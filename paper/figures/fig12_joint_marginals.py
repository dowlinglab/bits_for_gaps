"""Fig 12 -- joint (pairwise) posterior distributions of the GP kernel hyperparameters.

Ported from ``fxns/mcmc_plotter.py``'s ``plot_joint_marginals`` (``fxns/plot_res.py``'s
``-m joint_marginals`` mode), simplified: a lower-triangular grid of hexbin plots with
the MAP point and a 95% credible-interval box per pair -- dropped the KDE contour
overlay (a purely visual smoothing detail, not the quantitative content) for a
pragmatic reproduction. Visual reproduction only (no reference pin) -- spot-check against
the archived ``joint_marginals_15.png``.
"""

import os

import numpy as np

from . import _archive


def make(archive_dir, out_dir, iters=_archive.PUBLISHED_ITERS, img_fmt="png"):
    import matplotlib.pyplot as plt

    _archive.apply_plot_settings()
    os.makedirs(out_dir, exist_ok=True)
    params = _archive.load_param_posterior_samples(archive_dir, iters)
    n_params = params.shape[1]

    fig, axes = plt.subplots(
        n_params - 1,
        n_params - 1,
        figsize=(4 * (n_params - 1), 4 * (n_params - 1)),
        sharex="col",
        sharey="row",
        squeeze=False,
    )

    for i in range(1, n_params):
        for j in range(n_params - 1):
            ax = axes[i - 1, j]
            if j > i - 1:
                ax.axis("off")
                continue
            x, y = params[:, j], params[:, i]
            hb = ax.hexbin(x, y, gridsize=30, bins="log", cmap="viridis")
            map_x = x[np.argmin((x - np.median(x)) ** 2 + (y - np.median(y)) ** 2)]
            map_y = y[np.argmin((x - np.median(x)) ** 2 + (y - np.median(y)) ** 2)]
            ax.plot(map_x, map_y, "k*", markersize=12, label="MAP")
            xlo, xhi = np.quantile(x, [0.025, 0.975])
            ylo, yhi = np.quantile(y, [0.025, 0.975])
            ax.axvline(xlo, color="k", linestyle="--", linewidth=1)
            ax.axvline(xhi, color="k", linestyle="--", linewidth=1)
            ax.axhline(ylo, color="k", linestyle="--", linewidth=1)
            ax.axhline(yhi, color="k", linestyle="--", linewidth=1)
            if i == n_params - 1:
                ax.set_xlabel(rf"$\theta_{{{j + 1}}}$")
            if j == 0:
                ax.set_ylabel(rf"$\theta_{{{i + 1}}}$")

    fig.colorbar(hb, ax=axes.ravel().tolist(), label="Frequency (log)")
    axes[n_params - 2, 0].legend(loc="best", fontsize=10)

    out_path = os.path.join(out_dir, f"joint_marginals_{iters}.{img_fmt}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"path": out_path}
