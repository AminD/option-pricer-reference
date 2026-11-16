"""Contract and market description shared by every pricing method.

All pricers in this package take a :class:`VanillaOption` as input, so the
same object can be passed to the closed-form, tree, Monte Carlo and PDE
engines and the results compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np


class OptionType(str, Enum):
    """Call (right to buy) or put (right to sell)."""

    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> int:
        """+1 for a call, -1 for a put. Lets one formula cover both types."""
        return 1 if self is OptionType.CALL else -1


@dataclass(frozen=True)
class VanillaOption:
    """A European or American option together with its market parameters.

    Parameters
    ----------
    spot : float
        Current price of the underlying, S0.
    strike : float
        Exercise price, K.
    maturity : float
        Time to expiry in years, T.
    rate : float
        Continuously compounded risk-free rate, r (0.05 means 5%).
    vol : float
        Annualised volatility of the underlying, sigma (0.2 means 20%).
    option_type : OptionType or str
        ``"call"`` or ``"put"``.
    dividend_yield : float
        Continuous dividend yield, q. Defaults to 0 (no dividends).
    """

    spot: float
    strike: float
    maturity: float
    rate: float
    vol: float
    option_type: OptionType = OptionType.CALL
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        # Accept plain strings ("call" / "put") for convenience.
        object.__setattr__(self, "option_type", OptionType(self.option_type))
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity <= 0:
            raise ValueError("maturity must be positive")
        if self.vol <= 0:
            raise ValueError("vol must be positive")

    @property
    def is_call(self) -> bool:
        return self.option_type is OptionType.CALL

    @property
    def sign(self) -> int:
        """+1 for a call, -1 for a put."""
        return self.option_type.sign

    @property
    def carry(self) -> float:
        """Risk-neutral drift of the underlying, r - q."""
        return self.rate - self.dividend_yield

    def payoff(self, prices):
        """Exercise value max(S - K, 0) for a call, max(K - S, 0) for a put.

        Works on scalars and NumPy arrays alike.
        """
        return np.maximum(self.sign * (np.asarray(prices) - self.strike), 0.0)

    def with_(self, **changes) -> "VanillaOption":
        """Return a copy with some fields changed (the dataclass is immutable)."""
        return replace(self, **changes)
