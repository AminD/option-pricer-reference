"""Exotic options: Asian, barrier and American (Longstaff-Schwartz).

These payoffs depend on the whole path of the underlying or on an exercise
decision, so the plain Black-Scholes formula no longer applies. Each one is
priced here by simulation and checked against an independent method
(closed form, PDE or binomial tree).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from . import black_scholes, finite_difference
from .monte_carlo import MonteCarloResult, _summarise, simulate_paths
from .options import VanillaOption

# --------------------------------------------------------------------------
# Asian options: payoff on the average price over the life of the option.
# --------------------------------------------------------------------------


def geometric_asian_price(option: VanillaOption, n_fixings: int) -> float:
    """Closed-form price of a discretely monitored geometric-average Asian option.

    The geometric mean of lognormal prices is itself lognormal, so the price
    is a Black-Scholes-type formula. With fixings at t_i = i T / n:

        E[ln G]   = ln S0 + (r - q - sigma^2/2) T (n + 1) / (2n)
        Var[ln G] = sigma^2 T (n + 1)(2n + 1) / (6 n^2)
    """
    n = n_fixings
    mean = np.log(option.spot) + (option.carry - 0.5 * option.vol**2) * option.maturity * (n + 1) / (2 * n)
    variance = option.vol**2 * option.maturity * (n + 1) * (2 * n + 1) / (6 * n**2)
    std = np.sqrt(variance)

    d2 = (mean - np.log(option.strike)) / std
    d1 = d2 + std
    sign = option.sign
    forward_average = np.exp(mean + 0.5 * variance)
    return float(
        np.exp(-option.rate * option.maturity)
        * sign
        * (forward_average * norm.cdf(sign * d1) - option.strike * norm.cdf(sign * d2))
    )


def asian_price(
    option: VanillaOption,
    n_fixings: int = 50,
    n_paths: int = 100_000,
    seed: int | None = None,
    control_variate: bool = True,
    antithetic: bool = False,
) -> MonteCarloResult:
    """Monte Carlo price of an arithmetic-average Asian option.

    The arithmetic average has no closed form. The geometric-average option,
    which does, is almost perfectly correlated with it and makes an excellent
    control variate: it typically divides the standard error by 30 or more.
    """
    rng = np.random.default_rng(seed)
    paths = simulate_paths(option, n_fixings, n_paths, rng, antithetic=antithetic)[:, 1:]
    discount = np.exp(-option.rate * option.maturity)

    arithmetic = discount * option.payoff(paths.mean(axis=1))
    samples = arithmetic
    if control_variate:
        geometric = discount * option.payoff(np.exp(np.log(paths).mean(axis=1)))
        covariance = np.cov(arithmetic, geometric)
        beta = covariance[0, 1] / covariance[1, 1]
        samples = arithmetic - beta * (geometric - geometric_asian_price(option, n_fixings))

    if antithetic:
        # Average each path with its mirror image to get independent samples.
        half = n_paths // 2
        samples = 0.5 * (samples[:half] + samples[half:])
    return _summarise(samples, n_paths)


# --------------------------------------------------------------------------
# Barrier options: the contract appears or disappears if a level is touched.
# --------------------------------------------------------------------------

_BARRIER_TYPES = ("down-and-out", "down-and-in", "up-and-out", "up-and-in")


def _check_barrier(option: VanillaOption, barrier: float, barrier_type: str) -> tuple[bool, bool]:
    """Validate the inputs and return ``(is_down, is_knock_in)``."""
    if barrier_type not in _BARRIER_TYPES:
        raise ValueError(f"barrier_type must be one of {_BARRIER_TYPES}")
    is_down = barrier_type.startswith("down")
    if is_down and option.spot <= barrier:
        raise ValueError("spot must be above a down barrier")
    if not is_down and option.spot >= barrier:
        raise ValueError("spot must be below an up barrier")
    return is_down, barrier_type.endswith("in")


def barrier_price_analytic(option: VanillaOption, barrier: float, barrier_type: str) -> float:
    """Closed-form price of a continuously monitored single-barrier option.

    Reiner-Rubinstein formulas (no rebate), in the notation of Haug's
    "Complete Guide to Option Pricing Formulas". The knock-in price is built
    from four building blocks A-D; the knock-out price follows from

        knock-in + knock-out = vanilla.
    """
    is_down, is_knock_in = _check_barrier(option, barrier, barrier_type)
    s, k, t = option.spot, option.strike, option.maturity
    r, b, vol = option.rate, option.carry, option.vol
    phi = option.sign  # +1 call, -1 put
    eta = 1 if is_down else -1  # +1 down barrier, -1 up barrier

    vol_sqrt_t = vol * np.sqrt(t)
    mu = (b - 0.5 * vol**2) / vol**2
    shift = (1.0 + mu) * vol_sqrt_t
    x1 = np.log(s / k) / vol_sqrt_t + shift
    x2 = np.log(s / barrier) / vol_sqrt_t + shift
    y1 = np.log(barrier**2 / (s * k)) / vol_sqrt_t + shift
    y2 = np.log(barrier / s) / vol_sqrt_t + shift

    carry_discount = np.exp((b - r) * t)
    discount = np.exp(-r * t)
    reflection_spot = (barrier / s) ** (2.0 * (mu + 1.0))
    reflection_strike = (barrier / s) ** (2.0 * mu)

    term_a = phi * s * carry_discount * norm.cdf(phi * x1) - phi * k * discount * norm.cdf(phi * (x1 - vol_sqrt_t))
    term_b = phi * s * carry_discount * norm.cdf(phi * x2) - phi * k * discount * norm.cdf(phi * (x2 - vol_sqrt_t))
    term_c = phi * s * carry_discount * reflection_spot * norm.cdf(eta * y1) - phi * k * discount * (
        reflection_strike * norm.cdf(eta * (y1 - vol_sqrt_t))
    )
    term_d = phi * s * carry_discount * reflection_spot * norm.cdf(eta * y2) - phi * k * discount * (
        reflection_strike * norm.cdf(eta * (y2 - vol_sqrt_t))
    )

    strike_above_barrier = k > barrier
    if option.is_call and is_down:
        knock_in = term_c if strike_above_barrier else term_a - term_b + term_d
    elif option.is_call and not is_down:
        knock_in = term_a if strike_above_barrier else term_b - term_c + term_d
    elif not option.is_call and is_down:
        knock_in = term_b - term_c + term_d if strike_above_barrier else term_a
    else:
        knock_in = term_a - term_b + term_d if strike_above_barrier else term_c

    if is_knock_in:
        return float(knock_in)
    return float(black_scholes.price(option) - knock_in)


def barrier_price_mc(
    option: VanillaOption,
    barrier: float,
    barrier_type: str,
    n_steps: int = 100,
    n_paths: int = 100_000,
    seed: int | None = None,
    brownian_bridge: bool = True,
) -> MonteCarloResult:
    """Monte Carlo price of a single-barrier option.

    A simulated path is only observed at ``n_steps`` dates, but the true path
    can touch the barrier between two dates. Ignoring this overprices
    knock-outs. With ``brownian_bridge=True`` each step is weighted by the
    exact probability that a Brownian bridge between its two end points stays
    on the safe side of the barrier,

        P(no hit) = 1 - exp(-2 ln(S_i / H) ln(S_{i+1} / H) / (sigma^2 dt)),

    which removes the discretisation bias for continuous monitoring. With
    ``brownian_bridge=False`` the barrier is only checked at the grid dates
    (a discretely monitored barrier).
    """
    is_down, is_knock_in = _check_barrier(option, barrier, barrier_type)
    rng = np.random.default_rng(seed)
    paths = simulate_paths(option, n_steps, n_paths, rng)
    dt = option.maturity / n_steps

    # Signed log-distance to the barrier: positive while on the safe side.
    distance = np.log(paths / barrier) * (1.0 if is_down else -1.0)
    alive_at_dates = (distance > 0).all(axis=1)

    if brownian_bridge:
        safe = np.clip(distance, 0.0, None)
        hit_probability = np.exp(-2.0 * safe[:, :-1] * safe[:, 1:] / (option.vol**2 * dt))
        survival = alive_at_dates * np.prod(1.0 - hit_probability, axis=1)
    else:
        survival = alive_at_dates.astype(float)

    weight = 1.0 - survival if is_knock_in else survival
    samples = np.exp(-option.rate * option.maturity) * option.payoff(paths[:, -1]) * weight
    return _summarise(samples, n_paths)


def barrier_price_fd(
    option: VanillaOption,
    barrier: float,
    barrier_type: str,
    n_space: int = 800,
    n_time: int = 800,
) -> float:
    """Finite-difference price of a continuously monitored barrier option.

    A knock-out is the Black-Scholes PDE on a domain that stops at the
    barrier, with V = 0 there. A knock-in is obtained by parity.
    """
    is_down, is_knock_in = _check_barrier(option, barrier, barrier_type)
    if is_down:
        knock_out = finite_difference.price(
            option, n_space=n_space, n_time=n_time, s_min=barrier, lower_knock_out=True
        )
    else:
        knock_out = finite_difference.price(
            option, n_space=n_space, n_time=n_time, s_max=barrier, upper_knock_out=True
        )
    if is_knock_in:
        vanilla = finite_difference.price(option, n_space=n_space, n_time=n_time)
        return float(vanilla - knock_out)
    return float(knock_out)


# --------------------------------------------------------------------------
# American options by simulation: Longstaff-Schwartz least-squares Monte Carlo.
# --------------------------------------------------------------------------


def american_price_lsm(
    option: VanillaOption,
    n_steps: int = 50,
    n_paths: int = 100_000,
    seed: int | None = None,
    degree: int = 3,
) -> MonteCarloResult:
    """American option price by the Longstaff-Schwartz algorithm.

    Monte Carlo runs forward in time, but the exercise decision needs the
    value of *continuing*, which depends on the future. Longstaff and
    Schwartz estimate it by regression: working backwards from expiry, at
    each date the discounted future cash flows of the in-the-money paths are
    regressed on a polynomial of the current price. A path is exercised when
    its immediate payoff exceeds the fitted continuation value.

    The option can only be exercised at the ``n_steps`` simulation dates
    (a Bermudan approximation), and the estimated exercise rule is slightly
    sub-optimal, so the result is biased a little low.
    """
    rng = np.random.default_rng(seed)
    paths = simulate_paths(option, n_steps, n_paths, rng, antithetic=n_paths % 2 == 0)
    step_discount = np.exp(-option.rate * option.maturity / n_steps)

    # Cash flow each path receives, expressed in the money of the current date.
    cash_flows = option.payoff(paths[:, -1])

    for step in range(n_steps - 1, 0, -1):
        cash_flows = cash_flows * step_discount
        prices = paths[:, step]
        exercise = option.payoff(prices)
        in_the_money = exercise > 0
        if in_the_money.sum() <= degree + 1:
            continue

        # Regress on in-the-money paths only: these are the only ones for
        # which the decision matters, and it makes the fit much sharper.
        x = prices[in_the_money] / option.strike  # scaled for conditioning
        coefficients = np.polynomial.polynomial.polyfit(x, cash_flows[in_the_money], degree)
        continuation = np.polynomial.polynomial.polyval(x, coefficients)

        exercise_now = exercise[in_the_money] > continuation
        indices = np.flatnonzero(in_the_money)[exercise_now]
        cash_flows[indices] = exercise[indices]

    samples = cash_flows * step_discount
    if n_paths % 2 == 0:
        half = n_paths // 2
        samples = 0.5 * (samples[:half] + samples[half:])
    result = _summarise(samples, n_paths)
    # Exercising immediately is always possible.
    immediate = float(option.payoff(option.spot))
    return MonteCarloResult(max(result.price, immediate), result.std_error, n_paths)
