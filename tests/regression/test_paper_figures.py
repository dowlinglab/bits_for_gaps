"""Regression: recompute the paper's quantitative figures and diff against
``paper/reference/*``.

The data these tests read comes from the curated, COMMITTED ``paper/data/`` subset
(see ``paper/data/README.md``) -- no private-archive access needed, so the three
read-only tests below (Fig 10, Fig 5, the hyperparameter posterior behind Fig 11) run
in the DEFAULT suite. Only
``test_fig08_wilson_curve_matches_archived_ground_truth`` stays gated behind
``@pytest.mark.vle``: it *recomputes* the Wilson curve via live Clapeyron calls
(needs Julia), unlike the others, which only read committed text files.

Point ``BFG_ARCHIVE_DIR`` at another directory of run artifacts (same layout) instead
of ``paper/data/`` to check a different run.

Fig 9's stage table already has its own gated test (``test_mccabe_thiele.py``) --
not duplicated here. Fig 8 (phase diagram) has no dedicated reference file (it's a
visual reproduction, not a pinned scalar target -- see ``paper/REPRODUCTION.md``);
instead its test cross-checks the freshly-recomputed Wilson curve against the
committed ``gt_Wilson_data`` the paper's own Fig 8 was built from.
"""

import os
from pathlib import Path

import numpy as np
import pytest

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "paper" / "data"
ARCHIVE_DIR = os.environ.get("BFG_ARCHIVE_DIR", str(DEFAULT_DATA_DIR))

pytestmark = pytest.mark.slow

requires_archive = pytest.mark.skipif(
    not os.path.isdir(ARCHIVE_DIR),
    reason=f"data directory not found at {ARCHIVE_DIR!r} (see paper/data/README.md)",
)


@requires_archive
def test_fig10_traces_match_reference(reference):
    from paper.figures import fig10_traces

    rhat, ess = fig10_traces.diagnostics(ARCHIVE_DIR)
    g = reference("hmc_diagnostics.json")
    np.testing.assert_allclose(rhat, g["rhat"], atol=g["tol"]["rhat_atol"])
    np.testing.assert_allclose(ess, g["ess"], rtol=g["tol"]["ess_rtol"])


@requires_archive
def test_fig05_parity_error_metrics_match_reference(reference):
    from paper.figures import fig05_parity

    g = reference("fig5_error_metrics.json")
    for it in (1, 15):
        computed = fig05_parity.error_metrics(ARCHIVE_DIR, it)
        expected = g[f"iter_{it}"]
        for key in ("rmse_train", "mae_train", "rmse_test", "mae_test"):
            median = float(np.median(computed[key]))
            assert median == pytest.approx(expected[key]["median"], rel=g["tol"]["rtol"])


@requires_archive
def test_hyperparameter_posterior_summary_matches_reference(reference):
    # Fig 11's quantitative backbone: recompute the same posterior summary statistics
    # extract_reference.py pinned, straight from the committed param_posterior_samples_15.
    from paper.figures import _archive

    params = _archive.load_param_posterior_samples(ARCHIVE_DIR)
    g = reference("hyperparameter_posterior.json")
    rtol = g["tol"]["rtol"]
    for j, name in enumerate(g["param_order"]):
        assert float(params[:, j].mean()) == pytest.approx(g["mean"][name], rel=rtol)
        assert float(np.median(params[:, j])) == pytest.approx(g["median"][name], rel=rtol)


@requires_archive
@pytest.mark.vle
def test_fig08_wilson_curve_matches_archived_ground_truth():
    # No dedicated reference file for Fig 8 (visual reproduction) -- cross-check the
    # freshly recomputed Wilson curve (live Clapeyron -- needs Julia, unlike the
    # read-only tests above) against the committed gt_Wilson_data the paper's own
    # Fig 8 was built from.
    from paper.figures import _archive
    from paper.figures.fig08_phase_diagram import wilson_curve

    gt = _archive.load_gt_wilson_data(ARCHIVE_DIR)
    z_gt, T_gt, y1_gt = gt[:, 0], gt[:, 1], gt[:, 2]

    z_fresh, T_fresh, y1_fresh = wilson_curve(n_grid=len(z_gt))
    np.testing.assert_allclose(z_fresh, z_gt, atol=1e-9)
    np.testing.assert_allclose(T_fresh, T_gt, atol=0.5)
    np.testing.assert_allclose(y1_fresh, y1_gt, atol=0.02)
