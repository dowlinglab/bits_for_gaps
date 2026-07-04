"""Seeded end-to-end integration test for the BITS-for-GAPS sampler.

Runs the full sequential-design decision pipeline of ``adaptiveEntropy`` on a *synthetic,
pure-Python* black box (a smooth 2-D function -- NO Julia). This pins the sampler's
behavior BEFORE the Phase 4 decomposition: the run must complete and its outputs (the
selected next point, the R-hat/ESS shapes, and the entropy field) must be stable across
two runs with the same seed.

We mirror ``adaptiveEntropy.run_model`` step by step but deliberately skip the
plotting-only ``gp_predict_2D`` step: it re-pickles the model and writes the full-grid
posterior-sample file used for figures, takes ~20 s (100 full-covariance draws over a
50x50 grid), and does not feed the acquisition -- ``entropy_objective`` re-seeds NumPy and
re-assigns every kernel hyperparameter before each deterministic ``predict_f`` call, so
the entropy field and the selected point are identical whether or not it ran.

``tests/integration/data/synthetic_baseline.json`` is a hard pin of this run's exact
outputs, captured from the pre-Phase-4 (monolithic ``sampler.py``) code. It turns "did
the Phase 4 decomposition change behavior?" into a mechanical atol=1e-10 check -- the
module split (and, later, the disk-as-state removal) must reproduce these numbers
exactly, not just match itself run-to-run.
"""
import json
import os
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


def _fwd_model(x2, x1):
    # The sampler calls FwdModel(*args, x2, x1) and uses the [0]-th return value.
    return [float(_true_f(x1, x2))]


def _write_initial_design(path, n=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform([b[0] for b in BOUNDS], [b[1] for b in BOUNDS], size=(n, 2))
    y = np.array([_true_f(xi[0], xi[1]) for xi in X])
    np.savetxt(os.path.join(path, "activity_data_1"), np.column_stack([X, y]))
    return n


def _build_sampler(path, seed=SEED):
    s = adaptiveEntropy(
        exp_name="synthetic", iters=1, x_bounds=BOUNDS, likelihood_var=0.05,
        mean_fxn=gpflow.mean_functions.Zero(), kernel_fxn=AnisotropicSE(),
        fwd_model=_fwd_model, fwd_model_args=(),
    )
    s.path = str(path)          # write all artifacts into the pytest tmp dir
    s.seed = seed
    # Tiny, fast configuration (the point is stability, not statistical quality).
    s.noSamples = 100
    s.noBurnIn = 50
    s.noChains = 2
    s.noGaussians = 8
    s.entropyMesh = [4, 4]
    s.noRestarts = 3
    return s


def _run_once(path, seed=SEED, n_init=12):
    """One end-to-end pass mirroring run_model (minus the plotting-only gp_predict_2D)."""
    n_init = _write_initial_design(path, n=n_init)
    s = _build_sampler(path, seed)

    XData, yData = s.read_data(iters=1)
    XGP, yGP = s.trsf_data(XData, yData, iters=1)
    GPmodel = s.build_gp(XGP, yGP, iters=1)
    trace, GPmodel = s.run_mcmc(GPmodel, iters=1)

    rhat = np.loadtxt(os.path.join(path, "rhat_value_1.txt"))
    ess = np.loadtxt(os.path.join(path, "ess_value_1.txt"))

    s.gen_entropy_surface_data_2D(trace, GPmodel, iters=1)
    entropy = np.loadtxt(os.path.join(path, "entropy_1"))

    result = s.optimize_2D(trace, GPmodel)
    xStar = np.array([bkwd(result.x[i]) for i, bkwd in enumerate(s.XTrsfBkwd)])

    # Close the loop: evaluate the injected black box at the selected point.
    s.call_model(xStar=xStar, XData=XData, yData=yData, iters=2)
    next_data = np.loadtxt(os.path.join(path, "activity_data_2"))

    return {
        "n_init": n_init,
        "rhat": rhat,
        "ess": ess,
        "entropy": entropy,
        "xStar": xStar,
        "max_entropy": float(-result.fun),  # objective is -H, so -fun is the max entropy
        "next_data": next_data,
        "trace_shape": trace.shape,
    }


@pytest.fixture(scope="module")
def run_a(tmp_path_factory):
    return _run_once(tmp_path_factory.mktemp("run_a"))


@pytest.fixture(scope="module")
def run_b(tmp_path_factory):
    return _run_once(tmp_path_factory.mktemp("run_b"))


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
    # against nondeterminism sneaking in during the Phase 4 decomposition.
    a, b = run_a, run_b
    np.testing.assert_allclose(a["rhat"], b["rhat"], atol=1e-10)
    np.testing.assert_allclose(a["ess"], b["ess"], atol=1e-10)
    np.testing.assert_allclose(a["entropy"], b["entropy"], atol=1e-10)
    np.testing.assert_allclose(a["xStar"], b["xStar"], atol=1e-10)
    assert a["max_entropy"] == pytest.approx(b["max_entropy"], abs=1e-10)


@pytest.mark.slow
def test_matches_pre_phase4_baseline(run_a):
    # Hard pin against tests/integration/data/synthetic_baseline.json, captured from the
    # monolithic (pre-decomposition) sampler.py. The Phase 4 module split -- and later the
    # disk-as-state removal -- must reproduce these exact numbers, not merely match itself.
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
