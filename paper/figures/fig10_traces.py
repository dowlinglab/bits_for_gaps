"""Fig 10 -- HMC trace plots + R-hat/ESS convergence diagnostics.

Ported from ``fxns/mcmc_plotter.py``'s ``plot_all_traces`` (``fxns/plot_res.py``'s
``-m all_traces`` mode). One subplot per kernel hyperparameter, one line per HMC
chain, annotated with that parameter's R-hat and ESS.

Quantitatively pinned: ``diagnostics()`` returns the exact (rhat, ess) arrays checked
against ``paper/golden/hmc_diagnostics.json`` by the gated regression test.
"""

import os

import numpy as np

from . import _archive


def diagnostics(archive_dir, iters=_archive.PUBLISHED_ITERS):
    """(rhat, ess) arrays for the archived run -- the quantitative content of Fig 10."""
    return _archive.load_rhat_ess(archive_dir, iters)


def make(archive_dir, out_dir, iters=_archive.PUBLISHED_ITERS, img_fmt="png"):
    """Regenerate the trace plot into ``out_dir``; returns the (rhat, ess) diagnostics."""
    import matplotlib.pyplot as plt

    _archive.apply_plot_settings()
    trace = _archive.load_traces(archive_dir, iters)
    rhat, ess = diagnostics(archive_dir, iters)

    n_samples, n_chains, n_params = trace.shape
    fig_letters = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    fig, axes = plt.subplots(
        n_params, 1, figsize=(10, 2 * n_params), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    colors = plt.cm.get_cmap("viridis", n_chains)
    linestyles = ["-", "--", "dashdot", "dotted"]

    for p in range(n_params):
        for c in range(n_chains):
            axes[p].plot(
                range(n_samples),
                trace[:, c, p],
                color=colors(c),
                linestyle=linestyles[c % len(linestyles)],
                linewidth=1.0,
                alpha=0.7,
                label=f"Chain {c + 1}",
            )
        axes[p].text(
            0.98,
            0.05,
            rf"$\hat{{R}} = {rhat[p]:.3f}$" + "\n"
            rf"$\hat{{ESS}} = {ess[p]:.1f}$",
            transform=axes[p].transAxes,
            ha="right",
            va="bottom",
            backgroundcolor="w",
            bbox=dict(boxstyle="square", ec="k", fc="w"),
        )
        axes[p].text(
            0.01,
            0.85,
            fig_letters[p % len(fig_letters)],
            transform=axes[p].transAxes,
            fontweight="bold",
        )
        axes[p].set_ylabel(rf"$\theta_{{{p + 1}}}$")

    axes[0].legend(loc="upper center", ncol=n_chains, bbox_to_anchor=(0.5, 1.30))
    axes[-1].set_xlim(0, n_samples)
    axes[-1].set_xlabel("Samples")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"trace_all_{iters}.{img_fmt}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"path": out_path, "rhat": rhat, "ess": ess}
