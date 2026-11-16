"""Binomial tree: convergence to Black-Scholes and American-option properties."""

import pytest

from pricer import VanillaOption, binomial, black_scholes


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("strike", [80, 100, 120])
def test_converges_to_black_scholes(option_type, strike):
    option = VanillaOption(100, strike, 1.0, 0.05, 0.2, option_type, dividend_yield=0.01)
    assert binomial.price(option, n_steps=2000) == pytest.approx(black_scholes.price(option), abs=5e-3)


def test_error_shrinks_with_more_steps(call):
    exact = black_scholes.price(call)
    coarse = abs(binomial.price(call, 50) - exact)
    fine = abs(binomial.price(call, 5000) - exact)
    assert fine < coarse / 20


def test_one_step_tree_by_hand():
    """A single step can be valued by hand with the risk-neutral probability."""
    import numpy as np

    option = VanillaOption(100, 100, 1.0, 0.0, np.log(1.2), "call")
    # With sigma = ln(1.2) the CRR tree goes to 120 or 83.33; check the value
    # against the risk-neutral expectation computed by hand.
    up, down = 1.2, 1 / 1.2
    prob_up = (1 - down) / (up - down)
    assert binomial.price(option, n_steps=1) == pytest.approx(prob_up * 20.0)


def test_american_call_without_dividends_equals_european(call):
    """Early exercise of a call on a non-dividend-paying stock is never optimal."""
    assert binomial.price(call, 500, american=True) == pytest.approx(binomial.price(call, 500), abs=1e-10)


def test_american_put_is_worth_more_than_european(put):
    american = binomial.price(put, 1000, american=True)
    european = binomial.price(put, 1000)
    assert american > european + 0.3
    # Reference value for S=K=100, T=1, r=5%, sigma=20%.
    assert american == pytest.approx(6.0896, abs=5e-3)


def test_american_price_is_at_least_intrinsic():
    deep_itm_put = VanillaOption(60, 100, 1.0, 0.05, 0.2, "put")
    assert binomial.price(deep_itm_put, 500, american=True) == pytest.approx(40.0)


def test_tree_delta(call):
    assert binomial.delta(call, 2000) == pytest.approx(black_scholes.delta(call), abs=1e-3)


def test_exercise_boundary_is_below_strike_for_a_put(put):
    times, boundary = binomial.early_exercise_boundary(put, 400)
    known = boundary[~(boundary != boundary)]  # drop NaN
    assert len(times) == 401
    assert (known <= put.strike + 1e-12).all()
