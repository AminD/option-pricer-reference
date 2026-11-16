"""Interactive dashboard for the option pricer.

Run from the repository root::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make the ``pricer`` package importable when the app is launched directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pricer import VanillaOption, binomial, black_scholes, exotics, finite_difference, monte_carlo  # noqa: E402
from pricer.greeks import bump_greeks  # noqa: E402
from pricer.plotting import BLUE, MUTED, use_style  # noqa: E402

st.set_page_config(page_title="Option Pricer", page_icon="📈", layout="wide")
use_style()


# --------------------------------------------------------------------------
# Sidebar: the contract and the market.
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Contract")
    option_type = st.radio("Type", ["call", "put"], horizontal=True)
    american = st.toggle("American (early exercise)", value=False)
    spot = st.number_input("Spot price S", min_value=1.0, value=100.0, step=1.0)
    strike = st.number_input("Strike K", min_value=1.0, value=100.0, step=1.0)
    maturity = st.slider("Maturity T (years)", 0.05, 3.0, 1.0, 0.05)

    st.header("Market")
    rate = st.slider("Risk-free rate r", 0.0, 0.15, 0.05, 0.005, format="%.3f")
    vol = st.slider("Volatility σ", 0.05, 1.0, 0.20, 0.01)
    dividend = st.slider("Dividend yield q", 0.0, 0.10, 0.0, 0.005, format="%.3f")

    st.header("Numerical settings")
    n_steps = st.select_slider("Tree steps / PDE grid", [50, 100, 200, 400, 800], value=400)
    n_paths = st.select_slider("Monte Carlo paths", [10_000, 50_000, 100_000, 200_000], value=100_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

option = VanillaOption(spot, strike, maturity, rate, vol, option_type, dividend)


# --------------------------------------------------------------------------
# Cached computations: Streamlit reruns the script on every interaction.
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def price_table(option: VanillaOption, american: bool, n_steps: int, n_paths: int, seed: int) -> pd.DataFrame:
    """Price the contract with every available method."""
    rows = []
    if american:
        reference = binomial.price(option, max(n_steps, 500), american=True)
        rows.append(("Binomial tree (CRR)", reference, None))
        rows.append(("Finite differences (Crank-Nicolson)",
                     finite_difference.price(option, n_space=n_steps, n_time=n_steps, american=True), None))
        lsm = exotics.american_price_lsm(option, n_paths=n_paths, seed=seed)
        rows.append(("Longstaff-Schwartz Monte Carlo", lsm.price, 1.96 * lsm.std_error))
    else:
        reference = black_scholes.price(option)
        rows.append(("Black-Scholes (exact)", reference, None))
        rows.append(("Binomial tree (CRR)", binomial.price(option, n_steps), None))
        rows.append(("Finite differences (Crank-Nicolson)",
                     finite_difference.price(option, n_space=n_steps, n_time=n_steps), None))
        plain = monte_carlo.price(option, n_paths, seed)
        rows.append(("Monte Carlo (plain)", plain.price, 1.96 * plain.std_error))
        reduced = monte_carlo.price(option, n_paths, seed, antithetic=True, control_variate=True)
        rows.append(("Monte Carlo (antithetic + control variate)", reduced.price, 1.96 * reduced.std_error))

    table = pd.DataFrame(rows, columns=["Method", "Price", "95% CI half-width"])
    table["Difference vs reference"] = table["Price"] - reference
    return table


@st.cache_data(show_spinner=False)
def greeks_table(option: VanillaOption, american: bool) -> pd.DataFrame:
    """Greeks: closed form for European options, bump-and-revalue otherwise."""
    if american:
        values = bump_greeks(lambda o: binomial.price(o, 400, american=True), option)
        source = "Bump-and-revalue on the binomial tree"
    else:
        values = black_scholes.greeks(option)
        source = "Black-Scholes closed form"
    meaning = {
        "delta": "Price change for a +1 move of the underlying",
        "gamma": "Change of delta for a +1 move of the underlying",
        "vega": "Price change per unit of volatility (÷100 for one vol point)",
        "theta": "Price change per year of time passing (÷365 for one day)",
        "rho": "Price change per unit of interest rate (÷100 for 1%)",
    }
    table = pd.DataFrame({"Greek": list(values), "Value": list(values.values())})
    table["Meaning"] = table["Greek"].map(meaning)
    table.attrs["source"] = source
    return table


@st.cache_data(show_spinner=False)
def greek_profiles(option: VanillaOption) -> pd.DataFrame:
    spots = np.linspace(0.5 * option.strike, 1.5 * option.strike, 101)
    rows = [{"spot": s, **black_scholes.greeks(option.with_(spot=s)), "price": black_scholes.price(option.with_(spot=s))}
            for s in spots]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def convergence_data(option: VanillaOption, seed: int):
    exact = black_scholes.price(option)
    steps = np.arange(10, 201)
    tree = np.array([binomial.price(option, int(n)) for n in steps])
    paths = np.unique(np.logspace(2, 5.5, 30).astype(int))
    results = [monte_carlo.price(option, int(n), seed) for n in paths]
    mc_price = np.array([r.price for r in results])
    mc_half = 1.96 * np.array([r.std_error for r in results])
    return exact, steps, tree, paths, mc_price, mc_half


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

st.title("Option Pricer")
style = "American" if american else "European"
st.caption(
    f"{style} {option_type} · S = {spot:g} · K = {strike:g} · T = {maturity:g} y · "
    f"r = {rate:.1%} · σ = {vol:.0%} · q = {dividend:.1%}"
)

tab_price, tab_greeks, tab_convergence, tab_exotics = st.tabs(
    ["Price comparison", "Greeks", "Convergence", "Exotic options"]
)

with tab_price:
    table = price_table(option, american, n_steps, n_paths, int(seed))
    reference_price = table["Price"].iloc[0]
    intrinsic = float(option.payoff(spot))

    left, middle, right = st.columns(3)
    left.metric(table["Method"].iloc[0], f"{reference_price:.4f}")
    middle.metric("Intrinsic value", f"{intrinsic:.4f}")
    right.metric("Time value", f"{reference_price - intrinsic:.4f}")

    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Price": st.column_config.NumberColumn(format="%.4f"),
            "95% CI half-width": st.column_config.NumberColumn(format="%.4f"),
            "Difference vs reference": st.column_config.NumberColumn(format="%+.4f"),
        },
    )
    if american:
        premium = reference_price - black_scholes.price(option)
        st.info(f"Early-exercise premium over the European price: **{premium:.4f}**. "
                "There is no closed form for American options, so the binomial tree is the reference.")
    else:
        st.info("All methods solve the same model, so they must agree. The deterministic methods "
                "differ by discretisation error; Monte Carlo differs by sampling noise (see the confidence interval).")

with tab_greeks:
    table = greeks_table(option, american)
    st.caption(f"Source: {table.attrs['source']}")
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"Value": st.column_config.NumberColumn(format="%.4f")})

    st.subheader("Profiles against the spot (European, Black-Scholes)")
    profiles = greek_profiles(option)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, column, title in zip(axes, ["price", "delta", "gamma"], ["Price", "Delta", "Gamma"]):
        ax.plot(profiles["spot"], profiles[column], color=BLUE)
        ax.axvline(strike, color=MUTED, linewidth=1.0, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("Underlying price S (dotted line: strike)")
    st.pyplot(fig)
    plt.close(fig)

with tab_convergence:
    st.write("How fast each numerical method approaches the exact Black-Scholes price of the "
             "**European** version of the contract.")
    exact, steps, tree, paths, mc_price, mc_half = convergence_data(option, int(seed))
    fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(13, 4.0))

    left_ax.plot(steps, tree, color=BLUE, linewidth=1.2, label="Binomial tree")
    left_ax.axhline(exact, color=MUTED, linewidth=1.2, linestyle="--", label="Black-Scholes")
    left_ax.set_xlabel("Number of time steps")
    left_ax.set_ylabel("Price")
    left_ax.set_title("Binomial tree")
    left_ax.legend()

    right_ax.fill_between(paths, mc_price - mc_half, mc_price + mc_half, color=BLUE, alpha=0.18,
                          linewidth=0, label="95% confidence interval")
    right_ax.plot(paths, mc_price, color=BLUE, label="Monte Carlo")
    right_ax.axhline(exact, color=MUTED, linewidth=1.2, linestyle="--", label="Black-Scholes")
    right_ax.set_xscale("log")
    right_ax.set_xlabel("Number of simulated paths")
    right_ax.set_title("Monte Carlo")
    right_ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with tab_exotics:
    st.write("Path-dependent contracts on the same underlying. Each is priced by simulation and "
             "checked against an independent method.")
    asian_col, barrier_col = st.columns(2)

    with asian_col:
        st.subheader("Asian option")
        st.caption("Payoff on the average price over the life of the option.")
        n_fixings = st.slider("Number of averaging dates", 4, 252, 50)
        asian = exotics.asian_price(option, n_fixings, n_paths, int(seed))
        naive = exotics.asian_price(option, n_fixings, n_paths, int(seed), control_variate=False)
        st.metric("Arithmetic Asian (Monte Carlo)", f"{asian.price:.4f}")
        st.caption(f"95% confidence interval: ± {1.96 * asian.std_error:.4f}")
        st.dataframe(
            pd.DataFrame(
                {
                    "Quantity": ["Geometric Asian (closed form)", "European (Black-Scholes)",
                                 "Standard error without control variate", "Standard error with control variate"],
                    "Value": [exotics.geometric_asian_price(option, n_fixings), black_scholes.price(option),
                              naive.std_error, asian.std_error],
                }
            ),
            hide_index=True, width="stretch",
            column_config={"Value": st.column_config.NumberColumn(format="%.4f")},
        )

    with barrier_col:
        st.subheader("Barrier option")
        st.caption("The contract is cancelled (out) or activated (in) if the barrier is touched.")
        barrier_type = st.selectbox("Barrier type", ["down-and-out", "down-and-in", "up-and-out", "up-and-in"])
        is_down = barrier_type.startswith("down")
        default = round(0.85 * spot, 2) if is_down else round(1.25 * spot, 2)
        barrier = st.number_input("Barrier level", min_value=0.01, value=default, step=1.0,
                                  key=f"barrier-{'down' if is_down else 'up'}")
        if (is_down and barrier >= spot) or (not is_down and barrier <= spot):
            st.warning("A down barrier must be below the spot and an up barrier above it.")
        else:
            analytic = exotics.barrier_price_analytic(option, barrier, barrier_type)
            mc = exotics.barrier_price_mc(option, barrier, barrier_type, n_paths=n_paths, seed=int(seed))
            st.metric("Closed form", f"{analytic:.4f}")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Method": ["Closed form (Reiner-Rubinstein)", "Finite differences",
                                   "Monte Carlo (Brownian bridge)", "Vanilla option, no barrier"],
                        "Price": [analytic, exotics.barrier_price_fd(option, barrier, barrier_type, 400, 400),
                                  mc.price, black_scholes.price(option)],
                    }
                ),
                hide_index=True, width="stretch",
                column_config={"Price": st.column_config.NumberColumn(format="%.4f")},
            )

    st.caption("Exotic prices use the European exercise style, whatever the toggle in the sidebar.")
