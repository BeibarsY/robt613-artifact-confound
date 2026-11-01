"""Shared figure helpers: one style, every figure saved under results/figures/."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
CMAP_SEQ = "viridis"
CMAP_DIV = "RdBu_r"


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure as PNG and PDF under results/figures/ and close it."""
    for ext in ("png", "pdf"):
        fig.savefig(config.FIGURES / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
