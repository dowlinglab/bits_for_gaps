"""Fig 4 -- maximum predictive entropy vs. sequential-design iteration.

Ported from ``fxns/mcmc_plotter.py``'s ``plot_entropy_v_iters`` (``fxns/plot_res.py``'s
``-m ent_v_iters`` mode). Visual reproduction only (no reference pin) -- spot-check
against the archived ``entropy_v_iters.png``.
"""

import os

from . import _archive


def make(archive_dir, out_dir, img_fmt="png"):
    import matplotlib.pyplot as plt

    _archive.apply_plot_settings()
    os.makedirs(out_dir, exist_ok=True)

    iters = _archive.available_entropy_iters(archive_dir)
    max_entropy = [_archive.load_entropy(archive_dir, i)[:, 2].max() for i in iters]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(iters, max_entropy, "k-", linewidth=2, label="Maximum Entropy")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Maximum Entropy")
    ax1.margins(x=0.01)
    ax1.set_xticks(iters[:: max(1, len(iters) // 15)])

    ax2 = ax1.twinx()
    ax2.plot(iters, [-h for h in max_entropy], "b--", linewidth=2, label="Minimum Information")
    ax2.set_ylabel("Minimum Information", color="b")
    ax2.tick_params(axis="y", labelcolor="b")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"entropy_v_iters.{img_fmt}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"path": out_path, "iters": iters, "max_entropy": max_entropy}
