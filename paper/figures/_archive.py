"""Shared loaders for the archived published run (iteration 15) + common plot style.

The archive is the PRIVATE old repo's ``results/less_x_new_manuscript_revisions/``
(REFACTOR_PLAN.md §7 decision 4: archive of record, not committed here, not on
Zenodo). Every figure module in this package reads from it via the functions below
rather than hardcoding paths -- ``paper/reproduce.py`` is the one place that resolves
the archive location (CLI arg / env var) and passes it down.

Iteration 15 is the published run: its ``rhat_value_15.txt``/``ess_value_15.txt``
match paper Fig 10 exactly (see ``paper/golden/hmc_diagnostics.json``).
"""
import os

import numpy as np

PUBLISHED_ITERS = 15


def require_archive(archive_dir):
    """Raise a clear error if the archive directory isn't where expected."""
    if not os.path.isdir(archive_dir):
        raise FileNotFoundError(
            f"Archive directory not found: {archive_dir!r}. This must point at the "
            f"private old repo's "
            f"'entropy_driven_hms/results/less_x_new_manuscript_revisions/' -- see "
            f"paper/DATA.md. Pass it via `paper/reproduce.py --archive <path>` or the "
            f"BFG_ARCHIVE_DIR environment variable."
        )
    return archive_dir


def apply_plot_settings():
    """Global rcParams matching the paper code's ``fxns/plot_settings.py`` /
    ``fxns/mcmc_plotter.py`` header -- shared look across figures."""
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["ytick.labelsize"] = 15
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["lines.linewidth"] = 3
    plt.rcParams["lines.markersize"] = 8
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True


def load_rhat_ess(archive_dir, iters=PUBLISHED_ITERS):
    """(rhat, ess) arrays, shape (n_hyperparameters,) each."""
    rhat = np.loadtxt(os.path.join(archive_dir, f"rhat_value_{iters}.txt"))
    ess = np.loadtxt(os.path.join(archive_dir, f"ess_value_{iters}.txt"))
    return rhat, ess


def load_traces(archive_dir, iters=PUBLISHED_ITERS):
    """HMC chain traces, shape (n_samples, n_chains, n_hyperparameters).

    Matches ``fxns/plot_res.py``'s ``all_traces`` mode exactly: collect every
    ``traces_chain_*_exp_{iters}`` file (sorted by filename, i.e. by chain index),
    stack, then transpose from (chain, sample, param) to (sample, chain, param).
    """
    suffix = f"_exp_{iters}"
    chains = []
    for name in sorted(os.listdir(archive_dir)):
        if name.startswith("traces_chain_") and name.endswith(suffix):
            chains.append(np.loadtxt(os.path.join(archive_dir, name)))
    if not chains:
        raise FileNotFoundError(f"No traces_chain_*{suffix} files found in {archive_dir}")
    return np.array(chains).transpose(1, 0, 2)


def load_param_posterior_samples(archive_dir, iters=PUBLISHED_ITERS):
    """Chain-0 constrained posterior samples, shape (n_samples, n_hyperparameters)."""
    return np.loadtxt(os.path.join(archive_dir, f"param_posterior_samples_{iters}"))


def load_activity_data(archive_dir, iters):
    """Evaluated design points, shape (n_points, 3): columns [z_PrOH, T, gamma_PrOH]."""
    return np.loadtxt(os.path.join(archive_dir, f"activity_data_{iters}"))


def load_activity_test_points(archive_dir):
    """Held-out test points, shape (n_test, 3): columns [z_PrOH, T, gamma_PrOH]."""
    return np.loadtxt(os.path.join(archive_dir, "activity_test_points"))


def load_gp_predict(archive_dir, iters=PUBLISHED_ITERS):
    """Full-grid GP posterior draws, shape (n_grid, 2 + n_draws): [z, T, draw_0, ...]."""
    return np.loadtxt(os.path.join(archive_dir, f"gp_predict_{iters}"))


def load_gp_predict_split(archive_dir, split, iters):
    """Train/test-only GP posterior draws (Fig 5): ``split`` is 'train' or 'test'."""
    return np.loadtxt(os.path.join(archive_dir, f"gp_predict_{split}_{iters}"))


def load_entropy(archive_dir, iters):
    """Entropy field, shape (n_grid, 3): columns [z_PrOH, T, H]."""
    return np.loadtxt(os.path.join(archive_dir, f"entropy_{iters}"))


def load_lhs_design(archive_dir):
    """(train, test) initial-design arrays, each shape (n, 2): columns [z_PrOH, T]."""
    train = np.loadtxt(os.path.join(archive_dir, "lhs_design"))
    test_path = os.path.join(archive_dir, "lhs_test_points")
    test = np.loadtxt(test_path) if os.path.isfile(test_path) else None
    return train, test


def load_cont_data(archive_dir):
    """Dense ground-truth activity-coefficient grid, shape (n, 4): [z, T, gamma_PrOH,
    gamma_H2O] (see ``proh_water_class.gen_cont_activities_2D``)."""
    return np.loadtxt(os.path.join(archive_dir, "cont_data"))


def load_phase_diagram_long(archive_dir, iters=PUBLISHED_ITERS):
    """Long-format surrogate phase-diagram draws: columns [z, sample_id, T_bub, y1]."""
    return np.loadtxt(os.path.join(archive_dir, f"phase_diagram_{iters}"))


def load_gt_wilson_data(archive_dir):
    """Wilson ground-truth VLE curve (archived), shape (n_z, 3): [z, T_bub, y1]."""
    return np.loadtxt(os.path.join(archive_dir, "gt_Wilson_data"))


def available_entropy_iters(archive_dir):
    """Sorted iteration numbers for every archived ``entropy_{i}`` file."""
    iters = []
    for name in os.listdir(archive_dir):
        if name.startswith("entropy_") and name[len("entropy_"):].isdigit():
            iters.append(int(name[len("entropy_"):]))
    return sorted(iters)
