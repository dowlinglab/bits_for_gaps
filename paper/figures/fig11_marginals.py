"""Fig 11 -- 1-D marginal posterior distributions of the GP kernel hyperparameters.

Ported from ``fxns/mcmc_plotter.py``'s ``plot_marginals`` (``fxns/plot_res.py``'s
``-m marginals`` mode). Visual reproduction only (no golden pin -- the underlying
posterior summary IS pinned quantitatively, in
``paper/golden/hyperparameter_posterior.json``) -- spot-check against the archived
``marginals_15.png``.
"""

import os

import numpy as np

from . import _archive

PARAM_LABELS = [r"$\theta_1$ (std_dev)", r"$\theta_2$ ($\ell_1$)", r"$\theta_3$ ($\ell_2$)"]


def make(archive_dir, out_dir, iters=_archive.PUBLISHED_ITERS, img_fmt="png"):
    import matplotlib.pyplot as plt

    _archive.apply_plot_settings()
    os.makedirs(out_dir, exist_ok=True)
    params = _archive.load_param_posterior_samples(archive_dir, iters)
    n_params = params.shape[1]
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    axes = np.atleast_1d(axes)
    for p in range(n_params):
        ax = axes[p]
        counts, bins, _ = ax.hist(params[:, p], bins=40, color="tab:blue", alpha=0.4, edgecolor="k")
        map_estimate = bins[np.argmax(counts)]
        lo, hi = np.quantile(params[:, p], [0.025, 0.975])
        ax.axvline(map_estimate, color="k", linestyle="-", label="MAP")
        ax.axvline(lo, color="k", linestyle="--", label="95% CI")
        ax.axvline(hi, color="k", linestyle="--")
        ax.text(
            0.04,
            0.91,
            letters[p % len(letters)],
            transform=ax.transAxes,
            fontweight="bold",
            backgroundcolor="w",
        )
        label = PARAM_LABELS[p] if p < len(PARAM_LABELS) else rf"$\theta_{{{p + 1}}}$"
        ax.set_xlabel(label)
        if p == 0:
            ax.set_ylabel("Density")
        if p == n_params - 1:
            ax.legend(loc="best")
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"marginals_{iters}.{img_fmt}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"path": out_path}
