"""Shared fixtures: one reference contract used across the test suite."""

import pytest

from pricer import VanillaOption


@pytest.fixture
def call() -> VanillaOption:
    """At-the-money one-year call. Its Black-Scholes price is 10.4506."""
    return VanillaOption(spot=100, strike=100, maturity=1.0, rate=0.05, vol=0.2, option_type="call")


@pytest.fixture
def put(call) -> VanillaOption:
    return call.with_(option_type="put")
