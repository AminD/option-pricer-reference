"""Monte Carlo pricing of European options, with variance reduction and Greeks.

The estimator is the discounted average payoff over simulated terminal prices.
Its error shrinks like 1/sqrt(N), which is slow: 100x more paths buy only one
extra digit. Two classic variance-reduction techniques are implemented:

* antithetic variates: every normal draw Z is reused as -Z;
* control variate: the discounted terminal price, whose expectation is known
  exactly, is used to cancel part of the sampling noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .options import VanillaOption


@dataclass(frozen=True)
class MonteCarloResult:
    """A Monte Carlo estimate together with its statistical uncertainty."""

    price: float
    std_error: float
    n_paths: int

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Two-sided confidence interval from the central limit theorem."""
        z = norm.ppf(0.5 + level / 2.0)
        return self.price - z * self.std_error, self.price + z * self.std_error


def _summarise(samples: np.ndarray, n_paths: int) -> MonteCarloResult:
    """Mean and standard error of i.i.d. samples."""
    std_error = samples.std(ddof=1) / np.sqrt(samples.size)
    return MonteCarloResult(float(samples.mean()), float(std_error), n_paths)


def terminal_prices(option: VanillaOption, normals: np.ndarray) -> np.ndarray:
    """Exact terminal prices S_T of a GBM for the given standard normal draws.

    S_T = S0 * exp((r - q - sigma^2 / 2) T + sigma sqrt(T) Z)

    No time-stepping is needed because the GBM has a closed-form solution.
    """
    drift = (option.carry - 0.5 * option.vol**2) * option.maturity
    diffusion = option.vol * np.sqrt(option.maturity) * normals
    return option.spot * np.exp(drift + diffusion)


def simulate_paths(
    option: VanillaOption,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate full GBM paths on a uniform time grid.

    Returns an array of shape ``(n_paths, n_steps + 1)`` whose first column is
    the spot. With ``antithetic=True`` the second half of the paths mirrors the
    first half (same draws with opposite sign), so ``n_paths`` must be even.
    """
    dt = option.maturity / n_steps
    if antithetic:
        if n_paths % 2:
            raise ValueError("n_paths must be even when antithetic=True")
        half = rng.standard_normal((n_paths // 2, n_steps))
        normals = np.concatenate([half, -half])
    else:
        normals = rng.standard_normal((n_paths, n_steps))

    # Sum the log-increments, then exponentiate: exact at every grid date.
    log_increments = (option.carry - 0.5 * option.vol**2) * dt + option.vol * np.sqrt(dt) * normals
    log_paths = np.cumsum(log_increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = option.spot
    paths[:, 1:] = option.spot * np.exp(log_paths)
    return paths


def price(
    option: VanillaOption,
    n_paths: int = 100_000,
    seed: int | None = None,
    antithetic: bool = False,
    control_variate: bool = False,
) -> MonteCarloResult:
    """Monte Carlo price of a European option.

    Parameters
    ----------
    n_paths : int
        Number of simulated terminal prices. With ``antithetic=True`` this is
        the total number of payoff evaluations (``n_paths / 2`` independent
        pairs), so the comparison with the plain estimator is at equal cost.
    seed : int, optional
        Seed for reproducibility.
    antithetic : bool
        Pair every draw Z with -Z and average the two payoffs.
    control_variate : bool
        Use the discounted terminal price as a control variate.
    """
    rng = np.random.default_rng(seed)
    discount = np.exp(-option.rate * option.maturity)

    if antithetic:
        normals = rng.standard_normal(n_paths // 2)
        s_up = terminal_prices(option, normals)
        s_down = terminal_prices(option, -normals)
        # One sample per pair: the two halves are negatively correlated, so
        # their average has a smaller variance than two independent draws.
        payoffs = 0.5 * discount * (option.payoff(s_up) + option.payoff(s_down))
        controls = 0.5 * discount * (s_up + s_down)
    else:
        s_t = terminal_prices(option, rng.standard_normal(n_paths))
        payoffs = discount * option.payoff(s_t)
        controls = discount * s_t

    if control_variate:
        # E[e^{-rT} S_T] = S0 e^{-qT} is known exactly. Subtracting
        # beta * (control - its mean) leaves the expectation unchanged and
        # removes the part of the payoff noise explained by the control.
        control_mean = option.spot * np.exp(-option.dividend_yield * option.maturity)
        covariance = np.cov(payoffs, controls)
        beta = covariance[0, 1] / covariance[1, 1]
        payoffs = payoffs - beta * (controls - control_mean)

    return _summarise(payoffs, n_paths)


# --------------------------------------------------------------------------
# Greeks estimated inside the simulation (no re-pricing needed).
# --------------------------------------------------------------------------


def delta_pathwise(option: VanillaOption, n_paths: int = 100_000, seed: int | None = None) -> MonteCarloResult:
    """Delta by the pathwise method: differentiate the payoff along each path.

    dS_T/dS0 = S_T / S0 and d payoff / dS_T = +-1 when in the money, so
    delta = e^{-rT} E[ sign * 1{ITM} * S_T / S0 ].
    """
    rng = np.random.default_rng(seed)
    s_t = terminal_prices(option, rng.standard_normal(n_paths))
    in_the_money = option.sign * (s_t - option.strike) > 0
    samples = np.exp(-option.rate * option.maturity) * option.sign * in_the_money * s_t / option.spot
    return _summarise(samples, n_paths)


def vega_pathwise(option: VanillaOption, n_paths: int = 100_000, seed: int | None = None) -> MonteCarloResult:
    """Vega by the pathwise method.

    dS_T/dsigma = S_T * (sqrt(T) Z - sigma T).
    """
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal(n_paths)
    s_t = terminal_prices(option, normals)
    in_the_money = option.sign * (s_t - option.strike) > 0
    ds_dvol = s_t * (np.sqrt(option.maturity) * normals - option.vol * option.maturity)
    samples = np.exp(-option.rate * option.maturity) * option.sign * in_the_money * ds_dvol
    return _summarise(samples, n_paths)


def delta_likelihood_ratio(
    option: VanillaOption, n_paths: int = 100_000, seed: int | None = None
) -> MonteCarloResult:
    """Delta by the likelihood-ratio method: differentiate the density instead.

    delta = e^{-rT} E[ payoff * Z / (S0 sigma sqrt(T)) ].

    Unlike the pathwise method it never differentiates the payoff, so it also
    works for discontinuous payoffs (digitals), at the cost of a larger variance.
    """
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal(n_paths)
    s_t = terminal_prices(option, normals)
    score = normals / (option.spot * option.vol * np.sqrt(option.maturity))
    samples = np.exp(-option.rate * option.maturity) * option.payoff(s_t) * score
    return _summarise(samples, n_paths)
