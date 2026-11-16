"""Finite differences: accuracy, order of convergence and stability."""

import numpy as np
import pytest

from pricer import VanillaOption, binomial, black_scholes, finite_difference


@pytest.mark.parametrize("scheme", ["implicit", "crank-nicolson"])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_stable_schemes_match_black_scholes(scheme, option_type):
    option = VanillaOption(100, 105, 1.0, 0.05, 0.2, option_type, dividend_yield=0.02)
    fd_price = finite_difference.price(option, n_space=800, n_time=800, scheme=scheme)
    assert fd_price == pytest.approx(black_scholes.price(option), abs=1e-2)


def test_explicit_scheme_is_unstable_above_the_limit(call):
    """With dt far above the stability limit the explicit scheme blows up."""
    limit = finite_difference.explicit_stability_limit(call, n_space=200, s_max=400.0)
    assert call.maturity / 200 > limit
    assert not finite_difference.solve(call, n_space=200, n_time=200, scheme="explicit").is_stable


def test_explicit_scheme_is_accurate_below_the_limit(call):
    limit = finite_difference.explicit_stability_limit(call, n_space=200, s_max=400.0)
    n_time = int(np.ceil(call.maturity / limit)) + 1
    result = finite_difference.solve(call, n_space=200, n_time=n_time, scheme="explicit")
    assert result.is_stable
    assert result.price == pytest.approx(black_scholes.price(call), abs=2e-2)


def test_crank_nicolson_is_more_accurate_in_time_than_implicit(call):
    """On a fine price grid with few time steps, the time error dominates."""
    exact = black_scholes.price(call)
    implicit = abs(finite_difference.price(call, n_space=1600, n_time=20, scheme="implicit") - exact)
    crank = abs(finite_difference.price(call, n_space=1600, n_time=20, scheme="crank-nicolson") - exact)
    assert crank < implicit / 5


def test_american_put_matches_binomial(put):
    fd_price = finite_difference.price(put, n_space=800, n_time=800, american=True)
    assert fd_price == pytest.approx(binomial.price(put, 2000, american=True), abs=1e-2)


def test_solution_respects_bounds(call):
    result = finite_difference.solve(call)
    assert (result.values >= -1e-8).all()
    assert (result.values <= result.grid + 1e-8).all()
    assert result.grid[0] == 0.0 and len(result.grid) == len(result.values)


def test_unknown_scheme_raises(call):
    with pytest.raises(ValueError):
        finite_difference.price(call, scheme="leapfrog")
