"""Model-independent Greeks by "bump and revalue".

Any pricing function can be differentiated numerically: change one input by
a small amount, price again, and take a finite difference. This is how
Greeks are obtained when no closed form exists (American options, PDE and
tree prices, exotics).
"""

from __future__ import annotations

from typing import Callable

from .options import VanillaOption

PriceFunction = Callable[[VanillaOption], float]


def bump_greeks(
    price_fn: PriceFunction,
    option: VanillaOption,
    spot_bump: float = 0.01,
    vol_bump: float = 1e-3,
    rate_bump: float = 1e-4,
    time_bump: float = 1.0 / 365.0,
) -> dict[str, float]:
    """Delta, gamma, vega, theta and rho of ``price_fn`` by central differences.

    Parameters
    ----------
    price_fn : callable
        Maps a :class:`VanillaOption` to a price, for example
        ``black_scholes.price`` or ``lambda o: binomial.price(o, american=True)``.
    spot_bump : float
        Relative bump of the spot (0.01 means +-1%).
    vol_bump, rate_bump : float
        Absolute bumps of volatility and rate.
    time_bump : float
        Reduction of the maturity, in years, used for theta (one day).

    Notes
    -----
    Do not use this with a Monte Carlo pricer unless the same random seed is
    used for every call; otherwise the sampling noise swamps the difference.
    """
    base = price_fn(option)
    ds = option.spot * spot_bump

    up = price_fn(option.with_(spot=option.spot + ds))
    down = price_fn(option.with_(spot=option.spot - ds))
    delta = (up - down) / (2.0 * ds)
    gamma = (up - 2.0 * base + down) / ds**2

    vega = (
        price_fn(option.with_(vol=option.vol + vol_bump))
        - price_fn(option.with_(vol=option.vol - vol_bump))
    ) / (2.0 * vol_bump)

    rho = (
        price_fn(option.with_(rate=option.rate + rate_bump))
        - price_fn(option.with_(rate=option.rate - rate_bump))
    ) / (2.0 * rate_bump)

    # Theta is the effect of time passing, i.e. of the maturity shrinking.
    time_bump = min(time_bump, 0.5 * option.maturity)
    theta = (price_fn(option.with_(maturity=option.maturity - time_bump)) - base) / time_bump

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
