"""Monte Carlo: unbiasedness, error scaling and variance reduction."""

import numpy as np
import pytest

from pricer import black_scholes, monte_carlo


@pytest.mark.parametrize("antithetic", [False, True])
@pytest.mark.parametrize("control_variate", [False, True])
def test_price_within_confidence_interval(call, antithetic, control_variate):
    """The exact price lies within 4 standard errors of the estimate."""
    result = monte_carlo.price(call, 200_000, seed=7, antithetic=antithetic, control_variate=control_variate)
    assert abs(result.price - black_scholes.price(call)) < 4 * result.std_error


def test_put_price(put):
    result = monte_carlo.price(put, 200_000, seed=11)
    assert abs(result.price - black_scholes.price(put)) < 4 * result.std_error


def test_same_seed_is_reproducible(call):
    assert monte_carlo.price(call, 10_000, seed=1) == monte_carlo.price(call, 10_000, seed=1)


def test_standard_error_scales_as_inverse_sqrt_n(call):
    """100x more paths should divide the standard error by about 10."""
    small = monte_carlo.price(call, 10_000, seed=3).std_error
    large = monte_carlo.price(call, 1_000_000, seed=3).std_error
    assert small / large == pytest.approx(10.0, rel=0.1)


def test_variance_reduction_helps(call):
    """Each technique lowers the standard error at equal number of paths."""
    plain = monte_carlo.price(call, 100_000, seed=5).std_error
    antithetic = monte_carlo.price(call, 100_000, seed=5, antithetic=True).std_error
    control = monte_carlo.price(call, 100_000, seed=5, control_variate=True).std_error
    both = monte_carlo.price(call, 100_000, seed=5, antithetic=True, control_variate=True).std_error
    assert antithetic < plain
    assert control < plain
    assert both < min(antithetic, control)


def test_confidence_interval_contains_estimate(call):
    result = monte_carlo.price(call, 50_000, seed=2)
    low, high = result.confidence_interval(0.95)
    assert low < result.price < high
    assert high - low == pytest.approx(2 * 1.959964 * result.std_error, rel=1e-5)


def test_pathwise_and_likelihood_ratio_greeks(call):
    pathwise = monte_carlo.delta_pathwise(call, 400_000, seed=9)
    likelihood = monte_carlo.delta_likelihood_ratio(call, 400_000, seed=9)
    vega = monte_carlo.vega_pathwise(call, 400_000, seed=9)
    assert abs(pathwise.price - black_scholes.delta(call)) < 4 * pathwise.std_error
    assert abs(likelihood.price - black_scholes.delta(call)) < 4 * likelihood.std_error
    assert abs(vega.price - black_scholes.vega(call)) < 4 * vega.std_error
    # The pathwise estimator is the more precise of the two for a vanilla payoff.
    assert pathwise.std_error < likelihood.std_error


def test_simulated_paths_have_the_right_mean(call):
    """E[S_T] = S0 e^{(r-q)T} under the risk-neutral measure."""
    rng = np.random.default_rng(0)
    paths = monte_carlo.simulate_paths(call, n_steps=12, n_paths=200_000, rng=rng, antithetic=True)
    assert paths.shape == (200_000, 13)
    assert np.all(paths[:, 0] == call.spot)
    assert paths[:, -1].mean() == pytest.approx(call.spot * np.exp(call.rate * call.maturity), rel=2e-3)
