"""Fig 5 -- train/test parity plots + RMSE/MAE error box-and-whisker plots.

Simplified from the paper's own plotting code: dropped the zoomed inset (a purely
visual detail, not the quantitative content).

Quantitatively pinned: ``error_metrics()`` returns exactly the per-draw RMSE/MAE
distributions ``paper/reference/fig5_error_metrics.json`` was extracted from (same
formulas as ``paper/extract_reference.py``).
"""

import os

import numpy as np

from . import _archive

ITERS = (1, 15)


def error_metrics(archive_dir, iters):
    """Per-draw RMSE/MAE on train and test, shape (500,) each -- the paper's headline
    "surrogate accuracy improves over 15 iterations" claim.
    """
    y_train = _archive.load_activity_data(archive_dir, 1)[:, 2]
    y_test = _archive.load_activity_test_points(archive_dir)[:, 2]
    yhat_train = _archive.load_gp_predict_split(archive_dir, "train", iters)
    yhat_test = _archive.load_gp_predict_split(archive_dir, "test", iters)

    diff_train = y_train.reshape(-1, 1) - yhat_train
    diff_test = y_test.reshape(-1, 1) - yhat_test
    return {
        "rmse_train": np.sqrt((diff_train**2).mean(axis=0)),
        "mae_train": np.abs(diff_train).mean(axis=0),
        "rmse_test": np.sqrt((diff_test**2).mean(axis=0)),
        "mae_test": np.abs(diff_test).mean(axis=0),
    }


def _parity_panel(ax, y_true, yhat, color, marker, label):
    mean = yhat.mean(axis=1)
    lo, hi = np.quantile(yhat, [0.05, 0.95], axis=1)
    ax.errorbar(
        y_true,
        mean,
        yerr=[mean - lo, hi - mean],
        fmt=marker,
        color="w",
        markeredgecolor=color,
        ecolor=color,
        markersize=8,
        capsize=2,
        label=label,
    )


def _plot_parity(archive_dir, out_dir, img_fmt, loglog):
    import matplotlib.pyplot as plt

    y_train = _archive.load_activity_data(archive_dir, 1)[:, 2]
    y_test = _archive.load_activity_test_points(archive_dir)[:, 2]
    paths = []
    letters = {1: "(a)", 15: "(b)"}
    for it in ITERS:
        fig, ax = plt.subplots(figsize=(5, 5))
        yhat_train = _archive.load_gp_predict_split(archive_dir, "train", it)
        yhat_test = _archive.load_gp_predict_split(archive_dir, "test", it)
        _parity_panel(ax, y_train, yhat_train, "r", "o", "Train")
        _parity_panel(ax, y_test, yhat_test, "b", "v", "Test")
        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")
            line = np.logspace(-1, 1.5, 200)
            lims = (0.65, 25)
        else:
            line = np.array([0, 20])
            lims = (0, 20)
        ax.plot(line, line, "k--", label="Parity" if it == ITERS[-1] else None)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.text(0.03, 0.93, letters[it], transform=ax.transAxes, fontweight="bold")
        ax.set_title(f"Iteration {it}")
        ax.set_xlabel("Wilson Model")
        ax.set_ylabel("Surrogate Model")
        if it == ITERS[-1]:
            ax.legend(loc="lower right")
        fig.tight_layout()
        suffix = "parity_loglog" if loglog else "parity"
        out_path = os.path.join(out_dir, f"{suffix}_{it}.{img_fmt}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)
    return paths


def _plot_error_box(archive_dir, out_dir, img_fmt):
    import matplotlib.pyplot as plt

    paths = []
    letters = {1: "(c)", 15: "(d)"}
    for it in ITERS:
        metrics = error_metrics(archive_dir, it)
        data = [
            metrics["mae_train"],
            metrics["mae_test"],
            metrics["rmse_train"],
            metrics["rmse_test"],
        ]
        fig, ax = plt.subplots(figsize=(5, 5))
        bp = ax.boxplot(
            data,
            showfliers=False,
            patch_artist=True,
            tick_labels=["MAE\nTrain", "MAE\nTest", "RMSE\nTrain", "RMSE\nTest"],
        )
        for patch, color in zip(bp["boxes"], ["r", "b", "r", "b"]):
            patch.set_facecolor("w")
            patch.set_edgecolor(color)
            patch.set_linewidth(2.0)
        ax.set_ylim(0, 7)
        ax.text(0.03, 0.93, letters[it], transform=ax.transAxes, fontweight="bold")
        ax.set_ylabel("Error")
        ax.set_title(f"Iteration {it}")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"error_bar_chart_{it}.{img_fmt}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)
    return paths


def make(archive_dir, out_dir, img_fmt="png"):
    """Regenerate all three Fig 5 panels; returns the iter-15 error metrics."""
    _archive.apply_plot_settings()
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    paths += _plot_parity(archive_dir, out_dir, img_fmt, loglog=False)
    paths += _plot_parity(archive_dir, out_dir, img_fmt, loglog=True)
    paths += _plot_error_box(archive_dir, out_dir, img_fmt)
    return {"path": paths, "error_metrics": {it: error_metrics(archive_dir, it) for it in ITERS}}
