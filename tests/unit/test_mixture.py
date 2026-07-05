"""Unit tests for ``mixture.py``'s posterior-mixture sampling.

Phase 9c: the headline behavior under test is that ``sample_gp_posterior_mixture``
(and ``predict_grid_2D``, which calls it) leave the caller's ``GPmodel`` unchanged --
Phase 9b traced a real bug to this function leaving the kernel at an arbitrary leftover
hyperparameter state (see ``paper/PHASE9B_INVESTIGATION.md``).
"""

import gpflow
import numpy as np
import pytest

from bits_for_gaps.kernels import AnisotropicSE
from bits_for_gaps.mixture import predict_grid_2D, sample_gp_posterior_mixture


def _val(param):
    return float(param.numpy())


@pytest.fixture
def gp_model():
    rng = np.random.default_rng(0)
    X = rng.uniform([0.0, 350.0], [1.0, 367.0], size=(8, 2))
    y = np.sin(X[:, 0:1]) + 0.01 * (X[:, 1:2] - 358.0)
    model = gpflow.models.GPR(data=(X, y), kernel=AnisotropicSE())
    gpflow.set_trainable(model.likelihood.variance, False)
    model.likelihood.variance.assign(0.05)
    return model


@pytest.fixture
def trace():
    # A small, arbitrary-but-plausible "posterior" -- values well away from the
    # kernel's defaults, so a test that fails to restore is obviously wrong, not
    # accidentally passing because the trace happens to match the defaults.
    rng = np.random.default_rng(1)
    return rng.uniform([0.5, 0.5, 0.5], [3.0, 3.0, 3.0], size=(20, 3))


def test_sample_gp_posterior_mixture_restores_kernel_state(gp_model, trace):
    before = [_val(p) for p in gp_model.kernel.hyperparameters]
    XGP = np.array([[0.3, 0.4], [0.6, 0.5]])
    sample_gp_posterior_mixture(trace, gp_model, XGP, seed=42, size=5)
    after = [_val(p) for p in gp_model.kernel.hyperparameters]
    assert after == before


def test_sample_gp_posterior_mixture_restores_kernel_state_on_error(gp_model, trace):
    # Even if something downstream raises, the kernel must not be left mutated --
    # exercises the try/finally, not just the happy path.
    before = [_val(p) for p in gp_model.kernel.hyperparameters]
    bad_XGP = "not an array"  # forces predict_f_samples to raise
    # Exact exception type is TF/gpflow-internal and not the point of this test --
    # only that *some* failure happens, and that it doesn't leave the kernel mutated.
    with pytest.raises(Exception):  # noqa: B017
        sample_gp_posterior_mixture(trace, gp_model, bad_XGP, seed=42, size=5)
    after = [_val(p) for p in gp_model.kernel.hyperparameters]
    assert after == before


def test_predict_grid_2d_restores_kernel_state(gp_model, trace):
    before = [_val(p) for p in gp_model.kernel.hyperparameters]
    predict_grid_2D(
        trace,
        gp_model,
        x_bounds=[(0.0, 1.0), (350.0, 367.0)],
        x_trsf_fwd=[lambda x: x, lambda x: x],
        x_trsf_bkwd=[lambda x: x, lambda x: x],
        y_trsf_bkwd=lambda y: y,
        seed=42,
        size=5,
        n_grid=4,
    )
    after = [_val(p) for p in gp_model.kernel.hyperparameters]
    assert after == before


def test_sample_gp_posterior_mixture_tf_seed_is_reproducible(gp_model, trace):
    XGP = np.array([[0.3, 0.4], [0.6, 0.5]])
    draws_a = sample_gp_posterior_mixture(trace, gp_model, XGP, seed=42, size=5, tf_seed=7)
    draws_b = sample_gp_posterior_mixture(trace, gp_model, XGP, seed=42, size=5, tf_seed=7)
    np.testing.assert_array_equal(draws_a, draws_b)


def test_sample_gp_posterior_mixture_default_tf_seed_is_unset(gp_model, trace):
    # Default (tf_seed=None) leaves TF's ambient RNG alone -- the original,
    # documented-non-reproducible behavior must be unchanged.
    XGP = np.array([[0.3, 0.4], [0.6, 0.5]])
    draws_a = sample_gp_posterior_mixture(trace, gp_model, XGP, seed=42, size=5)
    draws_b = sample_gp_posterior_mixture(trace, gp_model, XGP, seed=42, size=5)
    assert not np.array_equal(draws_a, draws_b)


def test_predict_grid_2d_rejects_non_2d_bounds(gp_model, trace):
    # predict_grid_2D is a 2-D-only visualization diagnostic (Phase 5) -- it must
    # raise a clear error for other dimensions rather than silently misbehaving.
    with pytest.raises(ValueError, match="2-D-only"):
        predict_grid_2D(
            trace,
            gp_model,
            x_bounds=[(0.0, 1.0)],
            x_trsf_fwd=[lambda x: x],
            x_trsf_bkwd=[lambda x: x],
            y_trsf_bkwd=lambda y: y,
            seed=42,
            size=5,
            n_grid=4,
        )
