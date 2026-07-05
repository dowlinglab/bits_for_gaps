"""Phase 5: N-D synthetic end-to-end tests via the ``BitsForGaps`` public API.

Runs a tiny, fully-seeded ``BitsForGaps.run(...)`` on pure-Python synthetic black boxes
(no Julia) at d=1 and d=3, proving the "already general" claim in code rather than just
at d=2. Mirrors ``test_end_to_end.py``'s tiny configuration and assertion style, but
parametrized over dimension.

The 3-D kernel deliberately mixes prior families across dimensions (LogNormal-positive,
Gamma-unconstrained, LogNormal-positive) -- the same per-dimension-Parameter design
point that matters for the paper's 2-D kernel, now exercised at d=3.
"""

import gpflow
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bits_for_gaps.kernels import AnisotropicSE
from bits_for_gaps.sampler import BitsForGaps

f64 = gpflow.utilities.to_default_float

SEED = 123


def _true_f_1d(x1):
    return np.sin(3.0 * x1) + 0.5 * x1


def _fwd_model_1d(x1):
    return [float(_true_f_1d(x1))]


def _kernel_1d():
    return AnisotropicSE(
        variance_prior=tfp.distributions.LogNormal(loc=tf.math.log(f64(1.0)), scale=f64(2.0)),
        lengthscale_priors=[
            tfp.distributions.LogNormal(loc=tf.math.log(f64(0.3)), scale=f64(0.5)),
        ],
    )


def _true_f_3d(x1, x2, x3):
    return np.sin(3.0 * x1) + np.cos(3.0 * x2) + 0.5 * x3


def _fwd_model_3d(x1, x2, x3):
    return [float(_true_f_3d(x1, x2, x3))]


def _kernel_3d():
    return AnisotropicSE(
        variance_prior=tfp.distributions.LogNormal(loc=tf.math.log(f64(1.0)), scale=f64(2.0)),
        lengthscale_priors=[
            tfp.distributions.LogNormal(loc=tf.math.log(f64(0.3)), scale=f64(0.5)),
            tfp.distributions.Gamma(concentration=f64(4.0), rate=f64(2.0)),
            tfp.distributions.LogNormal(loc=tf.math.log(f64(0.3)), scale=f64(0.5)),
        ],
        lengthscale_transforms=[gpflow.utilities.positive(), None, gpflow.utilities.positive()],
    )


CASES = {
    "1d": dict(
        bounds=[(0.0, 1.0)],
        kernel_factory=_kernel_1d,
        black_box=_fwd_model_1d,
        true_f=_true_f_1d,
        n_hyperparameters=2,
    ),
    "3d": dict(
        bounds=[(0.0, 1.0)] * 3,
        kernel_factory=_kernel_3d,
        black_box=_fwd_model_3d,
        true_f=_true_f_3d,
        n_hyperparameters=4,
    ),
}


def _initial_design(bounds, true_f, n=10, seed=0):
    rng = np.random.default_rng(seed)
    d = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    X = rng.uniform(lo, hi, size=(n, d))
    y = np.array([true_f(*row) for row in X])
    return X, y


def _build_bfg(case, seed=SEED):
    bfg = BitsForGaps(
        black_box=case["black_box"],
        bounds=case["bounds"],
        kernel=case["kernel_factory"](),
        likelihood_variance=0.05,
    )
    bfg.seed = seed
    # Tiny, fast configuration (the point is dimension-generality, not statistical quality).
    bfg.noSamples = 100
    bfg.noBurnIn = 50
    bfg.noChains = 2
    bfg.noGaussians = 8
    bfg.noRestarts = 3
    return bfg


def _run_once(case, seed=SEED, n_init=10):
    X_init, y_init = _initial_design(case["bounds"], case["true_f"], n=n_init)
    record = _build_bfg(case, seed).run(X_init, y_init).last
    return record


@pytest.fixture(scope="module", params=["1d", "3d"])
def case(request):
    return CASES[request.param]


@pytest.fixture(scope="module")
def run_a(case):
    return _run_once(case)


@pytest.fixture(scope="module")
def run_b(case):
    return _run_once(case)


@pytest.mark.slow
def test_shapes_match_dimension_and_hyperparameter_count(case, run_a):
    d = len(case["bounds"])
    n_hp = case["n_hyperparameters"]
    r = run_a
    assert r.rhat.shape == (n_hp,)
    assert r.ess.shape == (n_hp,)
    assert r.trace.shape[1] == n_hp
    assert r.xStar.shape == (d,)
    # entropy_surface_2D is 2-D-only; must be skipped (None), never computed, for d != 2.
    if d != 2:
        assert r.entropy_field is None


@pytest.mark.slow
def test_outputs_are_finite_and_in_bounds(case, run_a):
    r = run_a
    assert np.all(np.isfinite(r.rhat)) and np.all(r.rhat > 0)
    assert np.all(np.isfinite(r.ess)) and np.all(r.ess > 0)
    assert np.isfinite(r.max_entropy)
    for j, (lo, hi) in enumerate(case["bounds"]):
        assert lo <= r.xStar[j] <= hi


@pytest.mark.slow
def test_next_point_matches_injected_black_box(case, run_a):
    r = run_a
    d = len(case["bounds"])
    assert r.XData.shape[1] == d
    x_new = r.XData[-1]
    y_new = r.yData[-1, 0]
    np.testing.assert_allclose(x_new, r.xStar, atol=1e-9)
    assert y_new == pytest.approx(case["true_f"](*x_new), rel=1e-9)


@pytest.mark.slow
def test_stable_across_two_runs_with_same_seed(run_a, run_b):
    a, b = run_a, run_b
    np.testing.assert_allclose(a.rhat, b.rhat, atol=1e-10)
    np.testing.assert_allclose(a.ess, b.ess, atol=1e-10)
    np.testing.assert_allclose(a.trace, b.trace, atol=1e-10)
    np.testing.assert_allclose(a.xStar, b.xStar, atol=1e-10)
    assert a.max_entropy == pytest.approx(b.max_entropy, abs=1e-10)
