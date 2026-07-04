"""Fig 3 -- predictive-entropy field over the (z_PrOH, T) design space, early iterations.

Ported from ``fxns/mcmc_plotter.py``'s ``plot_entropy_2D`` (``fxns/plot_res.py``'s
``-m entropy_2D`` mode), shown as a small multi-panel grid across the first few
sequential-design iterations (matching the paper's lettered-panel (a)-(f) scheme)
rather than one file per iteration. Visual reproduction only (no golden pin) --
spot-check against the archived ``entropy_surface_{iters}.png`` files.
"""
import os

import numpy as np

from . import _archive

N_PANELS = 6


def _panel(ax, archive_dir, it, letter):
    entropy = _archive.load_entropy(archive_dir, it)
    x1, x2, H = entropy[:, 0], entropy[:, 1], entropy[:, 2]
    n = int(round(np.sqrt(len(x1))))
    x1_grid = x1.reshape(n, n) if n * n == len(x1) else None

    if x1_grid is not None:
        x2_grid = x2.reshape(n, n)
        H_grid = H.reshape(n, n)
        cs = ax.contourf(x1_grid, x2_grid, H_grid, cmap="viridis", levels=30)
    else:
        cs = ax.tricontourf(x1, x2, H, cmap="viridis", levels=30)

    XData = _archive.load_activity_data(archive_dir, it)[:, 0:2]
    ax.plot(XData[:, 0], XData[:, 1], "o", color="w", markeredgecolor="k", markersize=5)

    x_max_row = _archive.load_activity_data(archive_dir, it + 1)[-1, 0:2]
    ax.plot(x_max_row[0], x_max_row[1], "s", color="r", markeredgecolor="k", markersize=8)

    ax.text(0.05, 0.9, letter, transform=ax.transAxes, fontweight="bold",
           backgroundcolor="w")
    ax.set_title(f"Iteration {it}", fontsize=10)
    return cs


def make(archive_dir, out_dir, n_panels=N_PANELS, img_fmt="png"):
    import matplotlib.pyplot as plt

    _archive.apply_plot_settings()
    os.makedirs(out_dir, exist_ok=True)

    available = _archive.available_entropy_iters(archive_dir)
    iters = [i for i in available if (i + 1) in available or i == available[-1]][:n_panels]
    letters = [f"({chr(ord('a') + i)})" for i in range(len(iters))]

    n_cols = 3
    n_rows = -(-len(iters) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    cs = None
    for ax, it, letter in zip(axes, iters, letters):
        cs = _panel(ax, archive_dir, it, letter)
        ax.set_xlabel(r"$z_{\mathrm{PrOH}}$ [ ]")
        ax.set_ylabel(r"$T$ [K]")
    for ax in axes[len(iters):]:
        ax.axis("off")

    if cs is not None:
        fig.colorbar(cs, ax=axes[:len(iters)].tolist(),
                    label=r"Entropy, $\mathcal{H}\{f(\mathbf{x}_*) \mid \mathbf{y}\}$")

    out_path = os.path.join(out_dir, f"entropy_surface_panels.{img_fmt}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"path": out_path, "iters": iters}
