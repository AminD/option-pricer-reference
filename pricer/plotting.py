"""Shared Matplotlib style so every figure in the project looks the same."""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler

# Categorical colours, always used in this order (colour-blind-safe set).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]

INK = "#0b0b0b"        # titles and values
INK_SOFT = "#52514e"   # axis labels, legends
MUTED = "#898781"      # tick labels, reference lines
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def use_style() -> None:
    """Apply the project style to all subsequent Matplotlib figures."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "axes.prop_cycle": cycler(color=SERIES),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.titlecolor": INK,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SOFT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SOFT,
            "ytick.labelcolor": INK_SOFT,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_SOFT,
            "font.family": "DejaVu Sans",
        }
    )
