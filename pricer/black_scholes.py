"""Black-Scholes-Merton closed-form prices, Greeks and implied volatility.

Under the risk-neutral measure the underlying follows a geometric Brownian
motion

    dS = (r - q) S dt + sigma S dW,

and the price of a European option is the discounted expected payoff. For
calls and puts that expectation has the closed form implemented here. These
functions are the reference every numerical method in the package is tested
against.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .options import VanillaOption


def _d1_d2(option: VanillaOption) -> tuple[float, float]:
    """The two standardised distances that appear in every BS formula."""
    sqrt_t = np.sqrt(option.maturity)
    d1 = (
        np.log(option.spot / option.strike)
        + (option.carry + 0.5 * option.vol**2) * option.maturity
    ) / (option.vol * sqrt_t)
    d2 = d1 - option.vol * sqrt_t
    return d1, d2


def price(option: VanillaOption) -> float:
    """Black-Scholes price of a European call or put.

    C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

    Both are written as one expression using ``sign`` (+1 call, -1 put).
    """
    d1, d2 = _d1_d2(option)
    sign = option.sign
    discounted_spot = option.spot * np.exp(-option.dividend_yield * option.maturity)
    discounted_strike = option.strike * np.exp(-option.rate * option.maturity)
    return float(
        sign * (discounted_spot * norm.cdf(sign * d1) - discounted_strike * norm.cdf(sign * d2))
    )


# --------------------------------------------------------------------------
# Greeks: sensitivities of the price to each input.
# --------------------------------------------------------------------------


def delta(option: VanillaOption) -> float:
    """dV/dS: change in price for a one-unit move of the underlying."""
    d1, _ = _d1_d2(option)
    carry_discount = np.exp(-option.dividend_yield * option.maturity)
    return float(option.sign * carry_discount * norm.cdf(option.sign * d1))


def gamma(option: VanillaOption) -> float:
    """d2V/dS2: how fast delta changes. Identical for calls and puts."""
    d1, _ = _d1_d2(option)
    carry_discount = np.exp(-option.dividend_yield * option.maturity)
    return float(carry_discount * norm.pdf(d1) / (option.spot * option.vol * np.sqrt(option.maturity)))


def vega(option: VanillaOption) -> float:
    """dV/dsigma, per unit of volatility (divide by 100 for "per 1 vol point")."""
    d1, _ = _d1_d2(option)
    carry_discount = np.exp(-option.dividend_yield * option.maturity)
    return float(option.spot * carry_discount * norm.pdf(d1) * np.sqrt(option.maturity))


def theta(option: VanillaOption) -> float:
    """dV/dt: change in price per year of calendar time passing (usually < 0)."""
    d1, d2 = _d1_d2(option)
    sign = option.sign
    s, k, t = option.spot, option.strike, option.maturity
    r, q, vol = option.rate, option.dividend_yield, option.vol
    time_decay = -s * np.exp(-q * t) * norm.pdf(d1) * vol / (2.0 * np.sqrt(t))
    carry_term = sign * q * s * np.exp(-q * t) * norm.cdf(sign * d1)
    rate_term = -sign * r * k * np.exp(-r * t) * norm.cdf(sign * d2)
    return float(time_decay + carry_term + rate_term)


def rho(option: VanillaOption) -> float:
    """dV/dr, per unit of rate (divide by 100 for "per 1% rate move")."""
    _, d2 = _d1_d2(option)
    sign = option.sign
    return float(
        sign
        * option.strike
        * option.maturity
        * np.exp(-option.rate * option.maturity)
        * norm.cdf(sign * d2)
    )


def greeks(option: VanillaOption) -> dict[str, float]:
    """All five first/second-order Greeks in one dictionary."""
    return {
        "delta": delta(option),
        "gamma": gamma(option),
        "vega": vega(option),
        "theta": theta(option),
        "rho": rho(option),
    }


# --------------------------------------------------------------------------
# Implied volatility: invert the price formula.
# --------------------------------------------------------------------------


def implied_vol(
    market_price: float,
    option: VanillaOption,
    lower: float = 1e-6,
    upper: float = 5.0,
    tol: float = 1e-10,
) -> float:
    """Volatility that makes the Black-Scholes price equal ``market_price``.

    The price is strictly increasing in volatility, so the root is unique and
    Brent's bracketing method finds it reliably. ``option.vol`` is ignored.

    Raises
    ------
    ValueError
        If ``market_price`` is outside the range of prices reachable with a
        volatility in ``[lower, upper]`` (for example below intrinsic value).
    """

    def objective(vol: float) -> float:
        return price(option.with_(vol=vol)) - market_price

    if objective(lower) > 0 or objective(upper) < 0:
        raise ValueError("market_price is outside the attainable no-arbitrage range")
    return float(brentq(objective, lower, upper, xtol=tol))
