"""Option pricing with closed forms, trees, Monte Carlo and finite differences.

Quick start
-----------
>>> from pricer import VanillaOption, black_scholes, binomial, monte_carlo, finite_difference
>>> option = VanillaOption(spot=100, strike=100, maturity=1.0, rate=0.05, vol=0.2, option_type="call")
>>> round(black_scholes.price(option), 4)
10.4506
>>> round(binomial.price(option, n_steps=1000), 2)
10.45
"""

from . import binomial, black_scholes, exotics, finite_difference, greeks, monte_carlo
from .options import OptionType, VanillaOption

__version__ = "1.0.0"

__all__ = [
    "OptionType",
    "VanillaOption",
    "binomial",
    "black_scholes",
    "exotics",
    "finite_difference",
    "greeks",
    "monte_carlo",
]
