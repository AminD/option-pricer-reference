"""Cox-Ross-Rubinstein binomial tree for European and American options.

Time is cut into ``n_steps`` intervals. Over each one the underlying is
multiplied by ``u`` (up) or ``d = 1/u`` (down). The option is valued at
expiry and then rolled back through the tree, one step at a time, using the
risk-neutral probability of an up move. For an American option each node
also compares "exercise now" with "keep the option".
"""

from __future__ import annotations

import numpy as np

from .options import VanillaOption


def _tree_parameters(option: VanillaOption, n_steps: int) -> tuple[float, float, float, float]:
    """Up factor, down factor, risk-neutral up probability, one-step discount."""
    dt = option.maturity / n_steps
    up = np.exp(option.vol * np.sqrt(dt))
    down = 1.0 / up
    # Probability that makes the expected one-step growth equal to e^{(r-q) dt}.
    prob_up = (np.exp(option.carry * dt) - down) / (up - down)
    if not 0.0 < prob_up < 1.0:
        raise ValueError("n_steps is too small for these parameters (probability outside (0, 1))")
    return up, down, prob_up, np.exp(-option.rate * dt)


def _roll_back(option: VanillaOption, n_steps: int, american: bool, stop_at: int = 0) -> np.ndarray:
    """Backward induction from expiry to time step ``stop_at``.

    Returns the option values at the ``stop_at + 1`` nodes of that step,
    ordered from the lowest underlying price to the highest.
    """
    up, down, prob_up, discount = _tree_parameters(option, n_steps)

    # Node j of step n has had j up moves and n - j down moves.
    up_moves = np.arange(n_steps + 1)
    values = option.payoff(option.spot * up**up_moves * down ** (n_steps - up_moves))

    for step in range(n_steps - 1, stop_at - 1, -1):
        # Discounted risk-neutral expectation of the two child nodes.
        values = discount * (prob_up * values[1:] + (1.0 - prob_up) * values[:-1])
        if american:
            up_moves = np.arange(step + 1)
            prices = option.spot * up**up_moves * down ** (step - up_moves)
            values = np.maximum(values, option.payoff(prices))
    return values


def price(option: VanillaOption, n_steps: int = 500, american: bool = False) -> float:
    """Binomial price of a European (default) or American option."""
    return float(_roll_back(option, n_steps, american)[0])


def delta(option: VanillaOption, n_steps: int = 500, american: bool = False) -> float:
    """Delta read from the two nodes of the first time step."""
    up, down, _, _ = _tree_parameters(option, n_steps)
    value_down, value_up = _roll_back(option, n_steps, american, stop_at=1)
    return float((value_up - value_down) / (option.spot * (up - down)))


def early_exercise_boundary(option: VanillaOption, n_steps: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Critical underlying price below (put) or above (call) which exercise is optimal.

    Returns ``(times, boundary)``. Entries are NaN at dates where no node of
    the tree is in the exercise region.
    """
    up, down, prob_up, discount = _tree_parameters(option, n_steps)
    up_moves = np.arange(n_steps + 1)
    values = option.payoff(option.spot * up**up_moves * down ** (n_steps - up_moves))

    times = np.linspace(0.0, option.maturity, n_steps + 1)
    boundary = np.full(n_steps + 1, np.nan)
    boundary[-1] = option.strike

    for step in range(n_steps - 1, -1, -1):
        up_moves = np.arange(step + 1)
        prices = option.spot * up**up_moves * down ** (step - up_moves)
        continuation = discount * (prob_up * values[1:] + (1.0 - prob_up) * values[:-1])
        exercise = option.payoff(prices)
        exercised = (exercise > continuation) & (exercise > 0)
        if exercised.any():
            # Put: highest price still exercised. Call: lowest price exercised.
            boundary[step] = prices[exercised].max() if not option.is_call else prices[exercised].min()
        values = np.maximum(continuation, exercise)
    return times, boundary
