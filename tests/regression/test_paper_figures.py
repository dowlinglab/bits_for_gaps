"""Gated regression: recompute the paper's quantitative figures from the archived
published run and diff against ``paper/golden/*``.

Needs the private archive (REFACTOR_PLAN.md §7 decision 4) -- gated behind
``@pytest.mark.vle``, reusing the existing "needs a private/special environment,
deselected by default" marker (Fig 9's gated test in ``test_mccabe_thiele.py``
established this convention; none of the tests below specifically need Julia, but all
need the archive, which is exactly as unavailable to most environments as Julia is).
Point at the archive via the ``BFG_ARCHIVE_DIR`` environment variable if it isn't at
the default path.

Fig 9's stage table already has its own gated test (``test_mccabe_thiele.py``) --
not duplicated here. Fig 8 (phase diagram) has no dedicated golden file (it's a
visual reproduction, not a pinned scalar target -- see ``paper/REPRODUCTION.md``);
instead this cross-checks the freshly-recomputed Wilson curve against the archived
``gt_Wilson_data`` the paper's own Fig 8 was built from.
"""
import os

import numpy as np
import pytest

ARCHIVE_DIR = os.environ.get(
    "BFG_ARCHIVE_DIR",
    os.path.expanduser(
        "~/DowlingLab/CAREER/entropy_driven_hybrid_models_code/entropy_driven_hms/"
        "results/less_x_new_manuscript_revisions"
    ),
)

pytestmark = [pytest.mark.vle, pytest.mark.slow]

requires_archive = pytest.mark.skipif(
    not os.path.isdir(ARCHIVE_DIR),
    reason=f"archive not found at {ARCHIVE_DIR!r} (private old repo; see paper/DATA.md)",
)


@requires_archive
def test_fig10_traces_match_golden(golden):
    from paper.figures import fig10_traces

    rhat, ess = fig10_traces.diagnostics(ARCHIVE_DIR)
    g = golden("hmc_diagnostics.json")
    np.testing.assert_allclose(rhat, g["rhat"], atol=g["tol"]["rhat_atol"])
    np.testing.assert_allclose(ess, g["ess"], rtol=g["tol"]["ess_rtol"])


@requires_archive
def test_fig05_parity_error_metrics_match_golden(golden):
    from paper.figures import fig05_parity

    g = golden("fig5_error_metrics.json")
    for it in (1, 15):
        computed = fig05_parity.error_metrics(ARCHIVE_DIR, it)
        expected = g[f"iter_{it}"]
        for key in ("rmse_train", "mae_train", "rmse_test", "mae_test"):
            median = float(np.median(computed[key]))
            assert median == pytest.approx(expected[key]["median"], rel=g["tol"]["rtol"])


@requires_archive
def test_hyperparameter_posterior_summary_matches_golden(golden):
    # Fig 11's quantitative backbone: recompute the same posterior summary statistics
    # extract_golden.py pinned, straight from the archived param_posterior_samples_15.
    from paper.figures import _archive

    params = _archive.load_param_posterior_samples(ARCHIVE_DIR)
    g = golden("hyperparameter_posterior.json")
    rtol = g["tol"]["rtol"]
    for j, name in enumerate(g["param_order"]):
        assert float(params[:, j].mean()) == pytest.approx(g["mean"][name], rel=rtol)
        assert float(np.median(params[:, j])) == pytest.approx(g["median"][name], rel=rtol)


@requires_archive
def test_fig08_wilson_curve_matches_archived_ground_truth():
    # No dedicated golden file for Fig 8 (visual reproduction) -- cross-check the
    # freshly recomputed Wilson curve (live Clapeyron) against the archived
    # gt_Wilson_data the paper's own Fig 8 was built from.
    from paper.figures import _archive
    from paper.figures.fig08_phase_diagram import wilson_curve

    gt = _archive.load_gt_wilson_data(ARCHIVE_DIR)
    z_gt, T_gt, y1_gt = gt[:, 0], gt[:, 1], gt[:, 2]

    z_fresh, T_fresh, y1_fresh = wilson_curve(n_grid=len(z_gt))
    np.testing.assert_allclose(z_fresh, z_gt, atol=1e-9)
    np.testing.assert_allclose(T_fresh, T_gt, atol=0.5)
    np.testing.assert_allclose(y1_fresh, y1_gt, atol=0.02)
