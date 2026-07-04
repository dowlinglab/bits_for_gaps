"""Fig 8 -- T-x-y phase diagram (bubble/dew point) and liquid/vapor equilibrium curve.

Ported from ``new_phase_diagram.py``'s ``PhaseDiagram.plot_phase_diagram``. Two data
sources, reused per the Phase 7 approach (reuse the ported physics; don't re-run the
stochastic adaptive loop):

- **Surrogate ensemble** (many thin lines): the ARCHIVED ``phase_diagram_15`` --
  bubble/dew points from the paper's actual 15-iteration GP posterior samples. Not
  regenerated (that would mean re-running the adaptive HMC loop).
- **Wilson ground truth** (dashed line): recomputed FRESH via
  ``examples/vle_distillation.phase_diagram.wilson_gamma`` + ``vle_curve`` -- live
  Clapeyron calls, not the archived ``gt_Wilson_data`` -- demonstrating the ported
  physics reproduces the ground-truth curve the paper used. Needs Julia/Clapeyron.
"""
import os

import numpy as np

from . import _archive


def _reshape_long(data):
    """``phase_diagram_{iters}`` long format [z, sample_id, T_bub, y1] -> 2-D grids."""
    z, sample_id, T_bub, y1 = data[:, 0], data[:, 1].astype(int), data[:, 2], data[:, 3]
    order = np.lexsort((sample_id, z))
    z, sample_id, T_bub, y1 = z[order], sample_id[order], T_bub[order], y1[order]
    z_unique = np.unique(z)
    n_draws = np.unique(sample_id).size
    return z_unique, T_bub.reshape(-1, n_draws), y1.reshape(-1, n_draws)


def wilson_curve(n_grid=50):
    """Fresh Wilson ground-truth VLE curve via the ported Clapeyron physics."""
    from vle_distillation import phase_diagram as pd

    z_grid = np.linspace(0.0, 1.0, n_grid)
    return pd.vle_curve(pd.wilson_gamma, z_grid=z_grid)


def make(archive_dir, out_dir, iters=_archive.PUBLISHED_ITERS, img_fmt="png"):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _archive.apply_plot_settings()
    os.makedirs(out_dir, exist_ok=True)

    z_unique, T_bub, y1 = _reshape_long(_archive.load_phase_diagram_long(archive_dir, iters))
    z_wilson, T_wilson, y1_wilson = wilson_curve()

    # --- Panel (a): bubble/dew point vs temperature ---
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    n_draws = T_bub.shape[1]
    for q in range(n_draws):
        ax1.plot(y1[:, q], T_bub[:, q], linewidth=0.15, color="pink")
    for q in range(n_draws):
        ax1.plot(z_unique, T_bub[:, q], linewidth=0.15, color="purple")
    ax1.plot(z_wilson, T_wilson, color="purple", linestyle="--")
    ax1.plot(y1_wilson, T_wilson, color="pink", linestyle="--")
    ax1.set_xlabel(r"Mole Fraction PrOH, $z_{\mathrm{PrOH}}$ [ ]")
    ax1.set_ylabel("Temperature, T [K]")
    ax1.text(0.0, 0.97, "(a)", transform=ax1.transAxes, fontweight="bold")
    legend_elements = [
        Line2D([0], [0], color="grey", linestyle="-", label="Samples"),
        Line2D([0], [0], color="grey", linestyle="--", label="Wilson"),
        Line2D([0], [0], color="tab:pink", linestyle="-", label="Dew Point"),
        Line2D([0], [0], color="tab:purple", linestyle="-", label="Bubble Point"),
    ]
    ax1.legend(handles=legend_elements, loc="best")
    fig1.tight_layout()
    path1 = os.path.join(out_dir, f"phase_diagram_{iters}.{img_fmt}")
    fig1.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # --- Panel (b): liquid/vapor equilibrium curve ---
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.plot([0, 1], [0, 1], "k-", label="Parity")
    ax2.plot(z_wilson, y1_wilson, linestyle="--", color="b", label="Wilson")
    for q in range(n_draws):
        ax2.plot(z_unique, y1[:, q], color="b", linewidth=0.15)
    ax2.plot([], [], color="b", linewidth=1.0, label="Samples")  # legend proxy
    ax2.text(0.0, 0.97, "(b)", transform=ax2.transAxes, fontweight="bold")
    ax2.set_xlabel(r"Liquid Mol. Frac. PrOH, $z_{\mathrm{PrOH}}^{(\ell)}$ [ ]")
    ax2.set_ylabel(r"Vapor Mol. Frac. PrOH, $z_{\mathrm{PrOH}}^{(v)}$ [ ]")
    ax2.legend(loc="best")
    fig2.tight_layout()
    path2 = os.path.join(out_dir, f"phase_diagram_2_{iters}.{img_fmt}")
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    return {"path": [path1, path2], "wilson_curve": (z_wilson, T_wilson, y1_wilson)}
