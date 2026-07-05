"""Fig 2 -- initial Latin-hypercube design (train/test split) over (z_PrOH, T).

Ported from ``fxns/mcmc_plotter.py``'s ``plot_lhs_2d`` (``fxns/plot_res.py``'s
``-m lhs_2D`` mode). Visual reproduction only (no golden pin) -- spot-check against
the archived ``gp_lhs_design.png``.
"""

import os

from . import _archive


def make(archive_dir, out_dir, img_fmt="png"):
    import matplotlib.pyplot as plt

    _archive.apply_plot_settings()
    os.makedirs(out_dir, exist_ok=True)
    train, test = _archive.load_lhs_design(archive_dir)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(train[:, 0], train[:, 1], "o", color="w", markeredgecolor="r", label="Train")
    if test is not None:
        ax.plot(test[:, 0], test[:, 1], "v", color="w", markeredgecolor="b", label="Test")
    ax.text(0.03, 0.95, "(a)", transform=ax.transAxes, fontweight="bold")
    ax.set_xlabel(r"Mole Fraction PrOH, $z_{\mathrm{PrOH}}$ [ ]")
    ax.set_ylabel(r"Temperature, $T$ [K]")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"gp_lhs_design.{img_fmt}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"path": out_path}
