"""Exotics: each simulation is checked against an independent method."""

import pytest

from pricer import binomial, black_scholes, exotics, monte_carlo

BARRIERS = [("down-and-out", 90.0), ("down-and-in", 90.0), ("up-and-out", 130.0), ("up-and-in", 130.0)]


# ---------------------------------------------------------------- Asian


def test_geometric_asian_with_one_fixing_is_black_scholes(call):
    assert exotics.geometric_asian_price(call, 1) == pytest.approx(black_scholes.price(call))


def test_geometric_asian_closed_form_matches_simulation(call):
    """Simulate the geometric average directly and compare with the formula."""
    import numpy as np

    rng = np.random.default_rng(4)
    paths = monte_carlo.simulate_paths(call, 20, 200_000, rng)[:, 1:]
    payoffs = np.exp(-call.rate * call.maturity) * call.payoff(np.exp(np.log(paths).mean(axis=1)))
    std_error = payoffs.std(ddof=1) / np.sqrt(payoffs.size)
    assert abs(payoffs.mean() - exotics.geometric_asian_price(call, 20)) < 4 * std_error


def test_asian_ordering(call):
    """Geometric mean <= arithmetic mean, and averaging lowers the call price."""
    arithmetic = exotics.asian_price(call, 50, 100_000, seed=1).price
    assert exotics.geometric_asian_price(call, 50) < arithmetic < black_scholes.price(call)


def test_asian_control_variate_agrees_and_reduces_error(call):
    plain = exotics.asian_price(call, 50, 100_000, seed=2, control_variate=False)
    controlled = exotics.asian_price(call, 50, 100_000, seed=2, control_variate=True)
    assert abs(plain.price - controlled.price) < 4 * plain.std_error
    assert controlled.std_error < plain.std_error / 10


# -------------------------------------------------------------- Barrier


@pytest.mark.parametrize("barrier_type,barrier", BARRIERS)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_barrier_closed_form_matches_pde(call, barrier_type, barrier, option_type):
    option = call.with_(option_type=option_type, dividend_yield=0.01)
    analytic = exotics.barrier_price_analytic(option, barrier, barrier_type)
    assert exotics.barrier_price_fd(option, barrier, barrier_type) == pytest.approx(analytic, abs=5e-3)


@pytest.mark.parametrize("barrier_type,barrier", BARRIERS)
def test_barrier_monte_carlo_matches_closed_form(call, barrier_type, barrier):
    result = exotics.barrier_price_mc(call, barrier, barrier_type, n_paths=200_000, seed=3)
    assert abs(result.price - exotics.barrier_price_analytic(call, barrier, barrier_type)) < 4 * result.std_error


@pytest.mark.parametrize("direction,barrier", [("down", 90.0), ("up", 130.0)])
def test_in_out_parity(call, direction, barrier):
    knock_in = exotics.barrier_price_analytic(call, barrier, f"{direction}-and-in")
    knock_out = exotics.barrier_price_analytic(call, barrier, f"{direction}-and-out")
    assert knock_in + knock_out == pytest.approx(black_scholes.price(call))


def test_discrete_monitoring_overprices_knock_out(call):
    """Checking the barrier only at grid dates misses some crossings."""
    exact = exotics.barrier_price_analytic(call, 90.0, "down-and-out")
    discrete = exotics.barrier_price_mc(call, 90.0, "down-and-out", n_paths=200_000, seed=3, brownian_bridge=False)
    assert discrete.price - exact > 4 * discrete.std_error


def test_up_and_out_call_with_strike_above_barrier_is_worthless(call):
    assert exotics.barrier_price_analytic(call.with_(strike=120), 110.0, "up-and-out") == pytest.approx(0.0, abs=1e-10)


def test_barrier_on_the_wrong_side_raises(call):
    with pytest.raises(ValueError):
        exotics.barrier_price_analytic(call, 110.0, "down-and-out")
    with pytest.raises(ValueError):
        exotics.barrier_price_analytic(call, 90.0, "sideways")


# ------------------------------------------------------------- American


@pytest.mark.parametrize("strike", [90, 100, 110])
def test_longstaff_schwartz_matches_binomial(put, strike):
    option = put.with_(strike=strike)
    reference = binomial.price(option, 2000, american=True)
    result = exotics.american_price_lsm(option, n_steps=50, n_paths=200_000, seed=6)
    # Statistical error plus a small low bias from the Bermudan approximation.
    assert abs(result.price - reference) < 4 * result.std_error + 0.03


def test_longstaff_schwartz_exceeds_european(put):
    result = exotics.american_price_lsm(put, n_paths=200_000, seed=6)
    assert result.price > black_scholes.price(put) + 0.3
