"""Closed-form prices and Greeks: textbook values and no-arbitrage identities."""

import numpy as np
import pytest

from pricer import VanillaOption, black_scholes
from pricer.greeks import bump_greeks


def test_textbook_values(call, put):
    assert black_scholes.price(call) == pytest.approx(10.4506, abs=1e-4)
    assert black_scholes.price(put) == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("strike", [60, 90, 100, 110, 150])
@pytest.mark.parametrize("dividend_yield", [0.0, 0.03])
def test_put_call_parity(strike, dividend_yield):
    """C - P = S e^{-qT} - K e^{-rT}, whatever the model parameters."""
    call = VanillaOption(100, strike, 0.75, 0.04, 0.3, "call", dividend_yield)
    put = call.with_(option_type="put")
    forward_value = 100 * np.exp(-dividend_yield * 0.75) - strike * np.exp(-0.04 * 0.75)
    assert black_scholes.price(call) - black_scholes.price(put) == pytest.approx(forward_value, abs=1e-10)


def test_price_bounds(call, put):
    """A call is worth less than the stock and more than its intrinsic forward value."""
    lower = call.spot - call.strike * np.exp(-call.rate * call.maturity)
    assert lower < black_scholes.price(call) < call.spot
    assert 0 < black_scholes.price(put) < put.strike


def test_price_increases_with_volatility(call):
    prices = [black_scholes.price(call.with_(vol=v)) for v in (0.1, 0.2, 0.4, 0.8)]
    assert prices == sorted(prices)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("strike", [80, 100, 125])
def test_greeks_match_finite_differences(option_type, strike):
    """Analytic Greeks agree with bump-and-revalue on the analytic price."""
    option = VanillaOption(100, strike, 1.0, 0.05, 0.2, option_type, dividend_yield=0.02)
    analytic = black_scholes.greeks(option)
    numeric = bump_greeks(black_scholes.price, option, spot_bump=1e-3, time_bump=1e-5)
    for name in ("delta", "gamma", "vega", "rho"):
        assert numeric[name] == pytest.approx(analytic[name], rel=1e-4, abs=1e-6), name
    assert numeric["theta"] == pytest.approx(analytic["theta"], rel=1e-3)


def test_delta_parity(call, put):
    """Call delta minus put delta equals e^{-qT} (here 1, no dividends)."""
    assert black_scholes.delta(call) - black_scholes.delta(put) == pytest.approx(1.0)
    assert black_scholes.gamma(call) == pytest.approx(black_scholes.gamma(put))


@pytest.mark.parametrize("vol", [0.05, 0.2, 0.6, 1.5])
def test_implied_vol_round_trip(call, vol):
    market_price = black_scholes.price(call.with_(vol=vol))
    assert black_scholes.implied_vol(market_price, call) == pytest.approx(vol, abs=1e-7)


def test_implied_vol_rejects_arbitrage_price(call):
    with pytest.raises(ValueError):
        black_scholes.implied_vol(call.spot + 1.0, call)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        VanillaOption(-1, 100, 1, 0.05, 0.2)
    with pytest.raises(ValueError):
        VanillaOption(100, 100, 1, 0.05, 0.0)
    with pytest.raises(ValueError):
        VanillaOption(100, 100, 1, 0.05, 0.2, "straddle")
