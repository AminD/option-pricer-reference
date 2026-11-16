"""Finite-difference solvers for the Black-Scholes PDE.

Written in time to maturity tau = T - t, the option value V(S, tau) solves

    dV/dtau = 0.5 sigma^2 S^2 d2V/dS2 + (r - q) S dV/dS - r V,

with the payoff as initial condition at tau = 0. A log change of variable
turns this into the heat equation, so the three classic schemes of heat
conduction apply directly:

* explicit  (theta = 0):   cheap, but unstable unless dtau is tiny;
* implicit  (theta = 1):   unconditionally stable, first order in time;
* Crank-Nicolson (0.5):    unconditionally stable, second order in time.

All three are the same "theta scheme" and share one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .options import VanillaOption

_THETA = {"explicit": 0.0, "implicit": 1.0, "crank-nicolson": 0.5}


@dataclass(frozen=True)
class FiniteDifferenceResult:
    """Solution of the PDE on the whole price grid at valuation time."""

    price: float
    grid: np.ndarray
    values: np.ndarray

    @property
    def is_stable(self) -> bool:
        """False when the scheme blew up (values no longer finite or bounded)."""
        return bool(np.all(np.isfinite(self.values)) and np.abs(self.values).max() < 1e3 * self.grid[-1])


def explicit_stability_limit(option: VanillaOption, n_space: int, s_max: float) -> float:
    """Time step below which the explicit scheme is guaranteed to be stable.

    Same condition as for the heat equation (dt <= dx^2 / (2 D)), with the
    "diffusivity" 0.5 sigma^2 S^2 evaluated where it is largest, at S = s_max.
    It is a sufficient condition: the scheme usually survives time steps
    slightly above it, then blows up quickly.
    """
    ds = s_max / n_space
    return 1.0 / (option.vol**2 * (s_max / ds) ** 2 + option.rate)


def solve(
    option: VanillaOption,
    n_space: int = 400,
    n_time: int = 400,
    scheme: str = "crank-nicolson",
    american: bool = False,
    s_min: float = 0.0,
    s_max: float | None = None,
    lower_knock_out: bool = False,
    upper_knock_out: bool = False,
    rannacher_steps: int = 2,
) -> FiniteDifferenceResult:
    """Solve the Black-Scholes PDE on a uniform grid ``[s_min, s_max]``.

    Parameters
    ----------
    n_space, n_time : int
        Number of price intervals and of time steps.
    scheme : {"explicit", "implicit", "crank-nicolson"}
        Time-stepping scheme.
    american : bool
        Enforce V >= payoff after every time step (early exercise).
    s_min, s_max : float
        Grid bounds. ``s_max`` defaults to four times max(spot, strike), far
        enough for the far-field boundary condition to be accurate.
    lower_knock_out, upper_knock_out : bool
        Impose V = 0 at ``s_min`` / ``s_max``: the option dies when the
        underlying touches that level (knock-out barrier).
    rannacher_steps : int
        For Crank-Nicolson only: number of initial steps replaced by two
        implicit half-steps. This damps the oscillations that the kink of the
        payoff at the strike otherwise triggers in Crank-Nicolson.
    """
    if scheme not in _THETA:
        raise ValueError(f"scheme must be one of {sorted(_THETA)}")
    theta = _THETA[scheme]
    if s_max is None:
        s_max = 4.0 * max(option.spot, option.strike)

    grid = np.linspace(s_min, s_max, n_space + 1)
    ds = grid[1] - grid[0]
    dt = option.maturity / n_time
    r, q, vol = option.rate, option.dividend_yield, option.vol

    # Coefficients of the spatial operator at interior nodes, using central
    # differences:  (L V)_i = lower_i V_{i-1} + diag_i V_i + upper_i V_{i+1}.
    s_in = grid[1:-1]
    diffusion = 0.5 * vol**2 * s_in**2 / ds**2
    convection = (r - q) * s_in / (2.0 * ds)
    lower = diffusion - convection
    diag = -2.0 * diffusion - r
    upper = diffusion + convection

    def boundary_values(tau: float) -> tuple[float, float]:
        """Dirichlet values at s_min and s_max when time to maturity is tau."""
        if lower_knock_out:
            low = 0.0
        elif option.is_call:
            low = 0.0
        else:
            # Deep in-the-money put: worth K immediately if American,
            # otherwise the discounted forward payoff.
            low = (
                option.strike - s_min
                if american
                else option.strike * np.exp(-r * tau) - s_min * np.exp(-q * tau)
            )
        if upper_knock_out:
            high = 0.0
        elif option.is_call:
            # Deep in-the-money call: discounted forward payoff, floored at
            # the exercise value when early exercise is allowed.
            high = s_max * np.exp(-q * tau) - option.strike * np.exp(-r * tau)
            if american:
                high = max(high, s_max - option.strike)
        else:
            high = 0.0
        return max(low, 0.0), max(high, 0.0)

    def step(values: np.ndarray, tau_new: float, dt_step: float, theta_step: float) -> np.ndarray:
        """Advance the solution by one time step with the theta scheme.

        (I - theta dt L) V_new = (I + (1 - theta) dt L) V_old
        """
        inner = values[1:-1]
        explicit_part = lower * values[:-2] + diag * inner + upper * values[2:]
        rhs = inner + (1.0 - theta_step) * dt_step * explicit_part

        new = np.empty_like(values)
        new[0], new[-1] = boundary_values(tau_new)

        if theta_step == 0.0:
            new[1:-1] = rhs
        else:
            # The known boundary values move to the right-hand side.
            rhs[0] += theta_step * dt_step * lower[0] * new[0]
            rhs[-1] += theta_step * dt_step * upper[-1] * new[-1]
            # Tridiagonal system in the banded layout expected by SciPy.
            banded = np.zeros((3, inner.size))
            banded[0, 1:] = -theta_step * dt_step * upper[:-1]
            banded[1, :] = 1.0 - theta_step * dt_step * diag
            banded[2, :-1] = -theta_step * dt_step * lower[1:]
            new[1:-1] = solve_banded((1, 1), banded, rhs)

        if american:
            # Early exercise: the option is never worth less than its payoff.
            new[1:-1] = np.maximum(new[1:-1], payoff[1:-1])
        return new

    payoff = option.payoff(grid)
    values = payoff.copy()
    if lower_knock_out:
        values[0] = 0.0
    if upper_knock_out:
        values[-1] = 0.0

    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(n_time):
            tau_old = n * dt
            if theta == 0.5 and n < rannacher_steps:
                # Rannacher start-up: two implicit half-steps.
                values = step(values, tau_old + 0.5 * dt, 0.5 * dt, 1.0)
                values = step(values, tau_old + dt, 0.5 * dt, 1.0)
            else:
                values = step(values, tau_old + dt, dt, theta)

    return FiniteDifferenceResult(float(np.interp(option.spot, grid, values)), grid, values)


def price(option: VanillaOption, **kwargs) -> float:
    """Convenience wrapper returning only the price at the spot. See :func:`solve`."""
    return solve(option, **kwargs).price
