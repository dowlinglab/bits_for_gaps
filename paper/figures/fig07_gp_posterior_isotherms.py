"""Fig 7 -- GP posterior predictive at fixed isotherms, offset for readability.

Visual reproduction only (no reference pin) -- spot-check against the archived
``gp_posterior_isotherms_15.png``.
"""

import os

import numpy as np

from . import _archive

TEMPS = (350, 355, 360, 367)


def make(
    archive_dir,
    out_dir,
    iters=_archive.PUBLISHED_ITERS,
    offset_step=1.0,
    max_sample_lines=20,
    img_fmt="png",
):
    import matplotlib.pyplot as plt

    _archive.apply_plot_settings()
    os.makedirs(out_dir, exist_ok=True)

    prediction = _archive.load_gp_predict(archive_dir, iters)
    ground_truth = _archive.load_cont_data(archive_dir)
    colors = plt.cm.Oranges_r(np.linspace(0.0, 0.6, len(TEMPS)))

    fig, ax = plt.subplots(figsize=(5, 6))
    for i, T in enumerate(TEMPS):
        pred_idx = np.argmin(np.abs(prediction[:, 1] - T))
        closest_T = prediction[pred_idx, 1]
        mask = prediction[:, 1] == closest_T
        z = prediction[mask, 0]
        order = np.argsort(z)
        z = z[order]
        y_samples = np.log(prediction[mask, 2:][order] + 4)

        gt_mask = np.abs(ground_truth[:, 1] - T) == np.min(np.abs(ground_truth[:, 1] - T))
        z_gt = ground_truth[gt_mask, 0]
        gt_order = np.argsort(z_gt)
        gamma_gt = np.log(ground_truth[gt_mask, 2][gt_order] + 4)
        z_gt = z_gt[gt_order]

        offset = i * offset_step
        n_lines = min(max_sample_lines, y_samples.shape[1])
        for s in range(n_lines):
            ax.plot(z, y_samples[:, s] + offset, color=colors[i], alpha=0.3, linewidth=0.8)
        mean = y_samples.mean(axis=1)
        lo, hi = np.percentile(y_samples, [2.5, 97.5], axis=1)
        ax.plot(z, mean + offset, color=colors[i], linewidth=2, label=f"T={T} K (mean)")
        ax.plot(z, lo + offset, color=colors[i], linestyle="--", linewidth=1.5)
        ax.plot(z, hi + offset, color=colors[i], linestyle="--", linewidth=1.5)
        ax.plot(z_gt, gamma_gt + offset, "*", color=colors[i], markersize=6)

    ax.set_xlabel(r"Mole Fraction PrOH, $z_{\mathrm{PrOH}}$ [ ]")
    ax.set_ylabel(r"Shifted Log Act. Coeff., $\log(\gamma_{\mathrm{PrOH}}) + i\Delta$ [ ]")
    ax.set_ylim(top=7)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"gp_posterior_isotherms_{iters}.{img_fmt}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"path": out_path}
