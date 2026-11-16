"""Regenerate every figure shown in the README.

Run from the repository root::

    python scripts/make_figures.py

The figures are written to ``docs/figures``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pricer import VanillaOption, binomial, black_scholes, exotics, finite_difference, monte_carlo  # noqa: E402
from pricer.plotting import AQUA, BLUE, INK_SOFT, MUTED, ORANGE, YELLOW, use_style  # noqa: E402

FIGURES = ROOT / "docs" / "figures"
CALL = VanillaOption(spot=100, strike=100, maturity=1.0, rate=0.05, vol=0.2, option_type="call")
PUT = CALL.with_(option_type="put")


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"wrote docs/figures/{name}")


def monte_carlo_convergence() -> None:
    """Estimate and 95% confidence band as the number of paths grows."""
    exact = black_scholes.price(CALL)
    n_paths = np.unique(np.logspace(2, 6, 40).astype(int))
    results = [monte_carlo.price(CALL, int(n), seed=1) for n in n_paths]
    prices = np.array([r.price for r in results])
    half_width = 1.96 * np.array([r.std_error for r in results])

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.fill_between(n_paths, prices - half_width, prices + half_width, color=BLUE, alpha=0.18,
                    linewidth=0, label="95% confidence interval")
    ax.plot(n_paths, prices, color=BLUE, label="Monte Carlo estimate")
    ax.axhline(exact, color=MUTED, linewidth=1.2, linestyle="--", label=f"Black-Scholes ({exact:.4f})")
    ax.set_xscale("log")
    ax.set_ylim(exact - 2.5, exact + 2.5)
    ax.set_xlabel("Number of simulated paths")
    ax.set_ylabel("Call price")
    ax.set_title("Monte Carlo converges to the exact price like 1/sqrt(N)")
    ax.legend(loc="upper right")
    save(fig, "monte_carlo_convergence.png")


def variance_reduction() -> None:
    """Standard error of each estimator at equal number of paths."""
    n_paths = 100_000
    settings = [
        ("Plain", {}),
        ("Antithetic", {"antithetic": True}),
        ("Control variate", {"control_variate": True}),
        ("Antithetic + control", {"antithetic": True, "control_variate": True}),
    ]
    errors = [monte_carlo.price(CALL, n_paths, seed=1, **kwargs).std_error for _, kwargs in settings]
    labels = [name for name, _ in settings]

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    positions = np.arange(len(labels))[::-1]
    ax.barh(positions, errors, height=0.55, color=BLUE)
    for y, error in zip(positions, errors):
        # Equivalent speed-up: paths needed scale with the variance.
        speed_up = (errors[0] / error) ** 2
        note = "" if error == errors[0] else f"   ({speed_up:.0f}x fewer paths for the same accuracy)"
        ax.text(error, y, f"  {error:.4f}{note}", va="center", fontsize=9, color=INK_SOFT)
    ax.set_yticks(positions, labels)
    ax.set_xlim(0, errors[0] * 1.75)
    ax.set_xlabel(f"Standard error with {n_paths:,} paths")
    ax.set_title("Variance reduction: same cost, smaller error")
    ax.grid(axis="y", visible=False)
    save(fig, "variance_reduction.png")


def binomial_convergence() -> None:
    """The tree price oscillates around the exact price as steps are added."""
    exact = black_scholes.price(CALL.with_(strike=105))
    option = CALL.with_(strike=105)
    steps = np.arange(10, 301)
    prices = [binomial.price(option, int(n)) for n in steps]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, prices, color=BLUE, linewidth=1.2, label="Binomial tree (CRR)")
    ax.axhline(exact, color=MUTED, linewidth=1.2, linestyle="--", label=f"Black-Scholes ({exact:.4f})")
    ax.set_xlabel("Number of time steps")
    ax.set_ylabel("Call price (K = 105)")
    ax.set_title("The binomial tree converges with a saw-tooth pattern")
    ax.legend(loc="upper right")
    save(fig, "binomial_convergence.png")


def finite_difference_schemes() -> None:
    """Left: explicit instability. Right: time-stepping error of the stable schemes."""
    exact = black_scholes.price(CALL)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Explicit scheme run just above and just below its stability limit.
    n_space, s_max = 100, 400.0
    limit = finite_difference.explicit_stability_limit(CALL, n_space, s_max)
    n_stable = int(np.ceil(CALL.maturity / limit)) + 1
    stable = finite_difference.solve(CALL, n_space, n_stable, "explicit")
    # Reduce the number of steps until the oscillations become visible.
    n_unstable = n_stable
    unstable = stable
    while np.abs(unstable.values - stable.values).max() < 25.0:
        n_unstable -= 1
        unstable = finite_difference.solve(CALL, n_space, n_unstable, "explicit")
    left.plot(unstable.grid, unstable.values, color=ORANGE, linewidth=1.2,
              label=f"{n_unstable} time steps: oscillations grow")
    left.plot(stable.grid, stable.values, color=BLUE, label=f"{n_stable} time steps: stable")
    left.set_xlim(0, s_max)
    left.set_ylim(-60, 420)
    left.set_xlabel("Underlying price S")
    left.set_ylabel("Call value today")
    left.set_title("Explicit scheme needs small time steps")
    left.legend(loc="upper left")

    # Time discretisation error on a very fine price grid.
    time_steps = np.array([5, 10, 20, 40, 80, 160, 320])
    for scheme, color, label in [
        ("implicit", ORANGE, "Implicit (order 1)"),
        ("crank-nicolson", BLUE, "Crank-Nicolson (order 2)"),
    ]:
        reference = finite_difference.price(CALL, n_space=3200, n_time=20_000, scheme=scheme)
        errors = [abs(finite_difference.price(CALL, n_space=3200, n_time=int(n), scheme=scheme) - reference)
                  for n in time_steps]
        right.loglog(time_steps, errors, marker="o", color=color, label=label)
    right.set_xlabel("Number of time steps")
    right.set_ylabel("Time-stepping error")
    right.set_title("Time error: order 1 vs order 2")
    right.legend(loc="lower left")
    save(fig, "finite_difference_schemes.png")
    assert abs(stable.price - exact) < 0.05


def greeks_profiles() -> None:
    """Delta and gamma of a call against the spot, for three maturities."""
    spots = np.linspace(60, 140, 161)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0))
    for maturity, color in [(1.0, BLUE), (0.25, ORANGE), (1 / 52, AQUA)]:
        label = {1.0: "1 year", 0.25: "3 months", 1 / 52: "1 week"}[maturity]
        options = [CALL.with_(spot=s, maturity=maturity) for s in spots]
        left.plot(spots, [black_scholes.delta(o) for o in options], color=color, label=label)
        right.plot(spots, [black_scholes.gamma(o) for o in options], color=color, label=label)
    for ax, name in [(left, "Delta"), (right, "Gamma")]:
        ax.axvline(CALL.strike, color=MUTED, linewidth=1.0, linestyle=":")
        ax.set_xlabel("Underlying price S (strike = 100)")
        ax.set_ylabel(name)
        ax.legend(title="Time to expiry", loc="upper left")
    left.set_title("Delta steepens near expiry")
    right.set_title("Gamma concentrates at the strike")
    save(fig, "greeks.png")


def american_put() -> None:
    """Early-exercise premium and exercise boundary of an American put."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0))

    spots = np.linspace(60, 140, 81)
    european = [black_scholes.price(PUT.with_(spot=s)) for s in spots]
    american = [binomial.price(PUT.with_(spot=s), 500, american=True) for s in spots]
    left.plot(spots, american, color=BLUE, label="American put")
    left.plot(spots, european, color=ORANGE, label="European put")
    left.plot(spots, PUT.payoff(spots), color=MUTED, linewidth=1.2, linestyle="--", label="Exercise value")
    left.set_xlabel("Underlying price S")
    left.set_ylabel("Put value")
    left.set_title("The right to exercise early has a price")
    left.legend(loc="upper right")

    times, boundary = binomial.early_exercise_boundary(PUT, 2000)
    known = ~np.isnan(boundary)
    right.plot(times[known], boundary[known], color=BLUE, linewidth=1.2)
    right.fill_between(times[known], 60, boundary[known], color=BLUE, alpha=0.12, linewidth=0)
    right.text(0.45, 68, "Exercise", color=INK_SOFT, fontsize=10)
    right.text(0.25, 95, "Hold", color=INK_SOFT, fontsize=10)
    right.set_ylim(60, 102)
    right.set_xlabel("Time (years)")
    right.set_ylabel("Critical price S*(t)")
    right.set_title("Exercise once S falls below the boundary")
    save(fig, "american_put.png")


def barrier_monitoring() -> None:
    """Down-and-out call price against the barrier level: three ways to price it."""
    barriers = np.linspace(70, 98, 15)
    analytic = [exotics.barrier_price_analytic(CALL, b, "down-and-out") for b in barriers]
    bridge = [exotics.barrier_price_mc(CALL, b, "down-and-out", 50, 100_000, seed=2).price for b in barriers]
    discrete = [exotics.barrier_price_mc(CALL, b, "down-and-out", 50, 100_000, seed=2, brownian_bridge=False).price
                for b in barriers]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(barriers, analytic, color=BLUE, label="Closed form (continuous monitoring)")
    ax.plot(barriers, discrete, linestyle="none", marker="s", color=YELLOW, markeredgecolor=INK_SOFT,
            markeredgewidth=0.5, label="Monte Carlo, barrier checked at 50 dates only")
    ax.plot(barriers, bridge, linestyle="none", marker="o", color=AQUA,
            label="Monte Carlo with Brownian-bridge correction")
    ax.set_xlabel("Barrier level (spot = 100)")
    ax.set_ylabel("Down-and-out call price")
    ax.set_title("Naive simulation misses barrier crossings between dates")
    ax.legend(loc="lower left")
    save(fig, "barrier_monitoring.png")


def main() -> None:
    use_style()
    monte_carlo_convergence()
    variance_reduction()
    binomial_convergence()
    finite_difference_schemes()
    greeks_profiles()
    american_put()
    barrier_monitoring()


if __name__ == "__main__":
    main()
