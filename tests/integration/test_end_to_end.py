"""Seeded end-to-end integration test for the BITS-for-GAPS sampler.

Runs the full sequential-design decision pipeline of ``adaptiveEntropy`` on a *synthetic,
pure-Python* black box (a smooth 2-D function -- NO Julia), via the in-memory ``run()``
API. The run must complete and its outputs (the selected next point, the R-hat/ESS
shapes, and the entropy field) must be stable across two runs with the same seed, and
reproduce ``tests/integration/data/synthetic_baseline.json`` -- a hard pin of this run's
exact outputs.

We deliberately don't pass ``predict_grid=True``: it re-pickles the model and computes
the full-grid posterior-sample array used only for figures, takes ~20 s (100 full-
covariance draws over a 50x50 grid), and does not feed the acquisition --
``entropy_objective`` re-seeds NumPy and re-assigns every kernel hyperparameter before
each deterministic ``predict_f`` call, so the entropy field and the selected point are
identical whether or not it ran. (It is also not bitwise-reproducible in isolation --
see ``mixture.py``'s note on ``predict_f_samples``' ambient TF randomness.)
"""

import json
from pathlib import Path

import gpflow
import numpy as np
import pytest

from bits_for_gaps.kernels import AnisotropicSE
from bits_for_gaps.sampler import adaptiveEntropy

BOUNDS = [(0.0, 1.0), (0.0, 1.0)]
SEED = 123
BASELINE_PATH = Path(__file__).parent / "data" / "synthetic_baseline.json"


def _true_f(x1, x2):
    """A smooth synthetic black box over [0, 1]^2 (modest range, no Julia)."""
    return np.sin(3.0 * x1) + np.cos(3.0 * x2)


def _fwd_model(x1, x2):
    # The sampler calls FwdModel(*args, *xStar) in natural dimension order.
    return [float(_true_f(x1, x2))]


def _initial_design(n=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform([b[0] for b in BOUNDS], [b[1] for b in BOUNDS], size=(n, 2))
    y = np.array([_true_f(xi[0], xi[1]) for xi in X])
    return X, y


def _build_sampler(seed=SEED):
    s = adaptiveEntropy(
        exp_name="synthetic",
        iters=1,
        x_bounds=BOUNDS,
        likelihood_var=0.05,
        mean_fxn=gpflow.mean_functions.Zero(),
        kernel_fxn=AnisotropicSE(),
        fwd_model=_fwd_model,
        fwd_model_args=(),
    )
    s.seed = seed
    # Tiny, fast configuration (the point is stability, not statistical quality).
    s.noSamples = 100
    s.noBurnIn = 50
    s.noChains = 2
    s.noGaussians = 8
    s.entropyMesh = [4, 4]
    s.noRestarts = 3
    return s


def _run_once(seed=SEED, n_init=12):
    """One fully in-memory end-to-end pass via ``adaptiveEntropy.run`` (no disk I/O)."""
    X_init, y_init = _initial_design(n=n_init)
    s = _build_sampler(seed)
    record = s.run(X_init, y_init).last
    return {
        "n_init": n_init,
        "rhat": record.rhat,
        "ess": record.ess,
        "entropy": record.entropy_field,
        "xStar": record.xStar,
        "max_entropy": record.max_entropy,
        "next_data": np.column_stack([record.XData, record.yData]),
        "trace_shape": record.trace.shape,
    }


@pytest.fixture(scope="module")
def run_a():
    return _run_once()


@pytest.fixture(scope="module")
def run_b():
    return _run_once()


@pytest.mark.slow
def test_run_completes_with_expected_shapes(run_a):
    r = run_a
    assert r["rhat"].shape == (3,)
    assert r["ess"].shape == (3,)
    # trace = (noSamples, num_params) for chain 0
    assert r["trace_shape"] == (100, 3)
    # entropy field over the 4x4 mesh: 16 rows of [x1, x2, H]
    assert r["entropy"].shape == (16, 3)
    assert r["xStar"].shape == (2,)


@pytest.mark.slow
def test_outputs_are_finite_and_physical(run_a):
    r = run_a
    assert np.all(np.isfinite(r["rhat"])) and np.all(r["rhat"] > 0)
    assert np.all(np.isfinite(r["ess"])) and np.all(r["ess"] > 0)
    assert np.all(np.isfinite(r["entropy"]))
    assert np.isfinite(r["max_entropy"])
    # Selected point lies inside the search bounds.
    for j, (lo, hi) in enumerate(BOUNDS):
        assert lo <= r["xStar"][j] <= hi


@pytest.mark.slow
def test_next_point_appended_via_injected_fwd_model(run_a):
    r = run_a
    # call_model appends exactly one new (x1, x2, y) row.
    assert r["next_data"].shape == (r["n_init"] + 1, 3)
    x1, x2, y = r["next_data"][-1]
    # The appended y must equal the injected pure-Python black box at the new point.
    assert y == pytest.approx(_true_f(x1, x2), rel=1e-9)
    np.testing.assert_allclose([x1, x2], r["xStar"], atol=1e-9)


@pytest.mark.slow
def test_stable_across_two_runs_with_same_seed(run_a, run_b):
    # Same seed, same process => the sampler must be deterministic. This is the guard
    # against nondeterminism sneaking into the sampler's internals.
    a, b = run_a, run_b
    np.testing.assert_allclose(a["rhat"], b["rhat"], atol=1e-10)
    np.testing.assert_allclose(a["ess"], b["ess"], atol=1e-10)
    np.testing.assert_allclose(a["entropy"], b["entropy"], atol=1e-10)
    np.testing.assert_allclose(a["xStar"], b["xStar"], atol=1e-10)
    assert a["max_entropy"] == pytest.approx(b["max_entropy"], abs=1e-10)


@pytest.mark.slow
def test_matches_pre_phase4_baseline(run_a):
    # Hard pin against tests/integration/data/synthetic_baseline.json -- captured once
    # and never regenerated, so this must reproduce these exact numbers, not merely
    # match itself.
    with open(BASELINE_PATH) as f:
        base = json.load(f)
    r = run_a
    assert r["n_init"] == base["n_init"]
    assert r["trace_shape"] == tuple(base["trace_shape"])
    np.testing.assert_allclose(r["rhat"], base["rhat"], atol=1e-10)
    np.testing.assert_allclose(r["ess"], base["ess"], atol=1e-10)
    np.testing.assert_allclose(r["entropy"], base["entropy"], atol=1e-10)
    np.testing.assert_allclose(r["xStar"], base["xStar"], atol=1e-10)
    assert r["max_entropy"] == pytest.approx(base["max_entropy"], abs=1e-10)
    np.testing.assert_allclose(r["next_data"], base["next_data"], atol=1e-10)


@pytest.mark.slow
def test_run_writes_no_files_by_default(tmp_path, monkeypatch):
    # A full run must execute with zero disk writes unless the caller explicitly opts
    # in via checkpoint_dir.
    monkeypatch.chdir(tmp_path)
    X_init, y_init = _initial_design(n=12)
    s = _build_sampler()
    s.run(X_init, y_init)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.slow
def test_run_checkpoint_dir_is_opt_in(tmp_path):
    X_init, y_init = _initial_design(n=12)
    s = _build_sampler()
    checkpoint_dir = tmp_path / "checkpoints"
    s.run(X_init, y_init, checkpoint_dir=str(checkpoint_dir))
    written = {p.name for p in checkpoint_dir.iterdir()}
    assert {"rhat_value_1.txt", "ess_value_1.txt", "activity_data_2", "gp_model_1.pkl"} <= written


@pytest.mark.slow
def test_run_predict_grid_true_writes_grid_predictions(tmp_path):
    # predict_grid=True additionally computes the full-grid GP posterior-predictive
    # samples (the paper's gp_predict_2D plotting diagnostic) -- off by default, so no
    # other test in this file exercises it.
    X_init, y_init = _initial_design(n=12)
    s = _build_sampler()
    checkpoint_dir = tmp_path / "checkpoints"
    history = s.run(X_init, y_init, checkpoint_dir=str(checkpoint_dir), predict_grid=True)
    assert (checkpoint_dir / "gp_predict_1").exists()
    # 50x50 grid (mixture.predict_grid_2D's default n_grid), columns [x1, x2, *draws].
    grid = np.loadtxt(checkpoint_dir / "gp_predict_1")
    assert grid.shape[0] == 50 * 50
    assert grid.shape[1] > 2
    assert history.last.XData.shape[0] == 13


@pytest.mark.slow
def test_run_with_initial_lml_maximization():
    # initalLML=True runs an MLE fit before HMC (adaptiveEntropy.maximize_lml) -- off
    # by default, so no other test exercises this branch of run().
    X_init, y_init = _initial_design(n=12)
    s = _build_sampler()
    s.initalLML = True
    history = s.run(X_init, y_init)
    record = history.last
    assert record.lml_result is not None
    assert np.all(np.isfinite(record.rhat))


@pytest.mark.slow
def test_run_with_lower_bound_acquisition_objective_completes():
    # acquisitionObjective="lower_bound" selects the paper's closed-form entropy lower
    # bound (Theorem/SI-2) instead of the default 2nd-order Taylor estimator. Default
    # ("taylor") behavior/baselines are covered by every other test in this file; this
    # just confirms the alternative objective runs a full iteration end-to-end without
    # error.
    X_init, y_init = _initial_design(n=12)
    s = _build_sampler()
    s.acquisitionObjective = "lower_bound"
    record = s.run(X_init, y_init).last
    assert np.all(np.isfinite(record.xStar))
    for j, (lo, hi) in enumerate(BOUNDS):
        assert lo <= record.xStar[j] <= hi
    assert np.isfinite(record.max_entropy)


@pytest.mark.slow
def test_run_rejects_unknown_acquisition_objective():
    X_init, y_init = _initial_design(n=12)
    s = _build_sampler()
    s.acquisitionObjective = "bogus"
    with pytest.raises(ValueError, match="acquisitionObjective"):
        s.run(X_init, y_init)


@pytest.mark.slow
def test_iteration_record_gpmodel_survives_downstream_mixture_sampling():
    # Regression guard: a caller that reuses history.last.GPmodel for a second purpose
    # (e.g. building a surrogate phase diagram) AFTER something else
    # (mixture.sample_gp_posterior_mixture, e.g. for a test-RMSE metric) has already
    # been called on that same object must not find it left at an arbitrary leftover
    # hyperparameter state.
    from bits_for_gaps import mixture

    X_init, y_init = _initial_design(n=12)
    record = _build_sampler().run(X_init, y_init).last
    before = [float(p.numpy()) for p in record.GPmodel.kernel.hyperparameters]

    XGP, _ = record.GPmodel.data
    mixture.sample_gp_posterior_mixture(record.trace, record.GPmodel, XGP, seed=1, size=5)

    after = [float(p.numpy()) for p in record.GPmodel.kernel.hyperparameters]
    assert after == before
