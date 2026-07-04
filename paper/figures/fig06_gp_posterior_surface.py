"""Fig 6 -- GP posterior predictive surface (3-D), iterations 1 vs. 15.

Ported from ``fxns/mcmc_plotter.py``'s ``plot_gp_posterior_2D`` (``fxns/plot_res.py``'s
``-m gp_post_2D`` mode). Visual reproduction only (no golden pin) -- spot-check
against the archived ``gp_posterior_2D_{1,15}.png``.
"""
import os

import numpy as np

from . import _archive

ITERS = (1, 15)


def _panel(ax, archive_dir, it, letter):
    data = _archive.load_activity_data(archive_dir, it)
    XData, yData = data[:, 0:2], np.log(data[:, 2]) + 4

    prediction = _archive.load_gp_predict(archive_dir, it)
    XStar, yStar = prediction[:, :2], np.log(prediction[:, 2:]) + 4
    mean = yStar.mean(axis=1)
    lo, hi = np.percentile(yStar, [2.5, 97.5], axis=1)

    x1_unique = np.unique(XStar[:, 0])
    x2_unique = np.unique(XStar[:, 1])
    x1_grid, x2_grid = np.meshgrid(x1_unique, x2_unique)
    shape = x1_grid.shape
    mean_grid = mean.reshape(shape)
    lo_grid = lo.reshape(shape)
    hi_grid = hi.reshape(shape)

    ax.plot_surface(x1_grid, x2_grid, mean_grid, cmap="Oranges", alpha=0.8)
    ax.plot_wireframe(x1_grid, x2_grid, lo_grid, rstride=5, cstride=2, alpha=0.15,
                      color="k")
    ax.plot_wireframe(x1_grid, x2_grid, hi_grid, rstride=5, cstride=2, alpha=0.15,
                      color="k")
    ax.scatter(XData[:, 0], XData[:, 1], yData, c="k", alpha=1.0, s=40)
    ax.set_xlabel(r"$z_{\mathrm{PrOH}}$ [ ]", labelpad=10)
    ax.set_ylabel(r"$T$ [K]", labelpad=10)
    ax.set_zlabel("Log Act. Coeff.,\n" + r"$\log(\gamma_{\mathrm{PrOH}})$ [ ]", labelpad=10)
    ax.set_zlim(3, 7)
    ax.set_title(f"{letter} Iteration {it}")


def make(archive_dir, out_dir, img_fmt="png"):
    import matplotlib.pyplot as plt

    _archive.apply_plot_settings()
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(11, 5))
    for i, (it, letter) in enumerate(zip(ITERS, ["(a)", "(b)"])):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        _panel(ax, archive_dir, it, letter)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"gp_posterior_2D_{ITERS[0]}_{ITERS[-1]}.{img_fmt}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"path": out_path}
