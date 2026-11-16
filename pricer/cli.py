"""Command-line interface.

Examples
--------
Price a European call with every method::

    python -m pricer --spot 100 --strike 100 --maturity 1 --rate 0.05 --vol 0.2 --type call

Price an American put::

    python -m pricer --spot 100 --strike 110 --maturity 1 --rate 0.05 --vol 0.2 --type put --american
"""

from __future__ import annotations

import argparse

from . import binomial, black_scholes, exotics, finite_difference, monte_carlo
from .options import VanillaOption


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="option-pricer",
        description="Price a vanilla option with several numerical methods and compare them.",
    )
    parser.add_argument("--spot", type=float, required=True, help="current price of the underlying")
    parser.add_argument("--strike", type=float, required=True, help="exercise price")
    parser.add_argument("--maturity", type=float, required=True, help="time to expiry in years")
    parser.add_argument("--rate", type=float, required=True, help="risk-free rate (0.05 = 5%%)")
    parser.add_argument("--vol", type=float, required=True, help="volatility (0.2 = 20%%)")
    parser.add_argument("--dividend", type=float, default=0.0, help="continuous dividend yield")
    parser.add_argument("--type", choices=["call", "put"], default="call", dest="option_type")
    parser.add_argument("--american", action="store_true", help="allow early exercise")
    parser.add_argument("--paths", type=int, default=200_000, help="Monte Carlo paths")
    parser.add_argument("--steps", type=int, default=500, help="tree steps / PDE grid size")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser


def _format_row(method: str, value: float, reference: float | None, note: str = "") -> str:
    difference = "" if reference is None else f"{value - reference:+.4f}"
    return f"{method:<28}{value:>10.4f}{difference:>12}   {note}"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    option = VanillaOption(
        spot=args.spot,
        strike=args.strike,
        maturity=args.maturity,
        rate=args.rate,
        vol=args.vol,
        option_type=args.option_type,
        dividend_yield=args.dividend,
    )

    style = "American" if args.american else "European"
    print(f"\n{style} {option.option_type.value}: S={option.spot:g} K={option.strike:g} "
          f"T={option.maturity:g} r={option.rate:g} sigma={option.vol:g} q={option.dividend_yield:g}\n")
    print(f"{'Method':<28}{'Price':>10}{'vs ref.':>12}")
    print("-" * 52)

    if args.american:
        # No closed form: the binomial tree is the reference.
        reference = binomial.price(option, n_steps=args.steps, american=True)
        print(_format_row("Binomial tree (CRR)", reference, None, "reference"))
        fd_price = finite_difference.price(option, n_space=args.steps, n_time=args.steps, american=True)
        print(_format_row("Finite differences (CN)", fd_price, reference))
        lsm = exotics.american_price_lsm(option, n_paths=args.paths, seed=args.seed)
        print(_format_row("Longstaff-Schwartz MC", lsm.price, reference, f"+/- {1.96 * lsm.std_error:.4f}"))
        european = black_scholes.price(option)
        print(f"\nEarly-exercise premium over the European price: {reference - european:.4f}")
    else:
        reference = black_scholes.price(option)
        print(_format_row("Black-Scholes (exact)", reference, None, "reference"))
        print(_format_row("Binomial tree (CRR)", binomial.price(option, n_steps=args.steps), reference))
        fd_price = finite_difference.price(option, n_space=args.steps, n_time=args.steps)
        print(_format_row("Finite differences (CN)", fd_price, reference))
        plain = monte_carlo.price(option, n_paths=args.paths, seed=args.seed)
        print(_format_row("Monte Carlo (plain)", plain.price, reference, f"+/- {1.96 * plain.std_error:.4f}"))
        reduced = monte_carlo.price(
            option, n_paths=args.paths, seed=args.seed, antithetic=True, control_variate=True
        )
        print(_format_row("Monte Carlo (reduced var.)", reduced.price, reference,
                          f"+/- {1.96 * reduced.std_error:.4f}"))

        print("\nGreeks (Black-Scholes)")
        print("-" * 52)
        for name, value in black_scholes.greeks(option).items():
            print(f"{name:<28}{value:>10.4f}")
    print()


if __name__ == "__main__":
    main()
