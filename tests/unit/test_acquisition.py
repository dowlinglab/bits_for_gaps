"""Unit tests for ``acquisition.py``'s entropy-maximization objective/optimizer.

The headline behavior under test is that ``entropy_objective`` (and its callers
``optimize``/``entropy_surface_2D``) leave the caller's ``GPmodel`` kernel unchanged --
since ``sampler.py``'s ``run()`` calls these on the same ``GPmodel`` object it later
stores in ``IterationRecord.GPmodel``, an unrestored kernel would silently corrupt
every iteration's returned model.
"""

import gpflow
import numpy as np
import pytest

from bits_for_gaps.acquisition import (
    ENTROPY_ESTIMATORS,
    entropy_objective,
    entropy_surface_2D,
    optimize,
)
from bits_for_gaps.kernels import AnisotropicSE


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
    rng = np.random.default_rng(1)
    return rng.uniform([0.5, 0.5, 0.5], [3.0, 3.0, 3.0], size=(20, 3))


def test_entropy_objective_restores_kernel_state(gp_model, trace):
    before = [_val(p) for p in gp_model.kernel.hyperparameters]
    xStarGP = np.array([0.4, 358.0])
    entropy_objective(xStarGP, trace, gp_model, seed=42, no_gaussians=5)
    after = [_val(p) for p in gp_model.kernel.hyperparameters]
    assert after == before


def test_entropy_objective_restores_kernel_state_on_error(gp_model, trace):
    # A NaN row triggers a mid-loop failure (inside assign_hyperparameters, AFTER the
    # kernel state has already been saved and at least one prior sample may have been
    # assigned) -- exercises the try/finally's restore path, not just the happy path.
    # no_gaussians=len(trace) selects every row (replace=False over the full trace),
    # guaranteeing the corrupted row is included regardless of RNG selection order.
    before = [_val(p) for p in gp_model.kernel.hyperparameters]
    bad_trace = trace.copy()
    bad_trace[3, 1] = np.nan
    xStarGP = np.array([0.4, 358.0])
    # Exact exception type is TF/gpflow-internal and not the point of this test --
    # only that *some* failure happens, and that it doesn't leave the kernel mutated.
    with pytest.raises(Exception):  # noqa: B017
        entropy_objective(xStarGP, bad_trace, gp_model, seed=42, no_gaussians=len(trace))
    after = [_val(p) for p in gp_model.kernel.hyperparameters]
    assert after == before


def test_optimize_restores_kernel_state(gp_model, trace):
    before = [_val(p) for p in gp_model.kernel.hyperparameters]
    optimize(
        trace,
        gp_model,
        x_bounds=[(0.0, 1.0), (350.0, 367.0)],
        x_trsf_fwd=[lambda x: x, lambda x: x],
        seed=42,
        no_gaussians=5,
        no_restarts=2,
    )
    after = [_val(p) for p in gp_model.kernel.hyperparameters]
    assert after == before


def test_entropy_surface_2d_restores_kernel_state(gp_model, trace):
    before = [_val(p) for p in gp_model.kernel.hyperparameters]
    entropy_surface_2D(
        trace,
        gp_model,
        x_bounds=[(0.0, 1.0), (350.0, 367.0)],
        mesh=[3, 3],
        x_trsf_fwd=[lambda x: x, lambda x: x],
        x_trsf_bkwd=[lambda x: x, lambda x: x],
        seed=42,
        no_gaussians=5,
    )
    after = [_val(p) for p in gp_model.kernel.hyperparameters]
    assert after == before


## ---------------------------------------------------------------------------
## Selectable acquisition objective (entropy_lower_bound as an alternative to Taylor).
## ---------------------------------------------------------------------------


def test_entropy_objective_default_is_taylor(gp_model, trace):
    xStarGP = np.array([0.4, 358.0])
    default = entropy_objective(xStarGP, trace, gp_model, seed=42, no_gaussians=5)
    explicit = entropy_objective(
        xStarGP, trace, gp_model, seed=42, no_gaussians=5, objective="taylor"
    )
    assert default == explicit


def test_entropy_objective_lower_bound_differs_from_taylor(gp_model, trace):
    xStarGP = np.array([0.4, 358.0])
    taylor = entropy_objective(
        xStarGP, trace, gp_model, seed=42, no_gaussians=5, objective="taylor"
    )
    lower_bound = entropy_objective(
        xStarGP, trace, gp_model, seed=42, no_gaussians=5, objective="lower_bound"
    )
    assert np.isfinite(taylor) and np.isfinite(lower_bound)
    # Different estimators, same inputs -- confirms `objective` actually switches
    # which one gets called rather than silently falling back to the default.
    assert taylor != lower_bound


def test_entropy_objective_rejects_unknown_objective(gp_model, trace):
    xStarGP = np.array([0.4, 358.0])
    with pytest.raises(ValueError, match="lower_bound"):
        entropy_objective(xStarGP, trace, gp_model, seed=42, no_gaussians=5, objective="bogus")


def test_optimize_accepts_lower_bound_objective(gp_model, trace):
    before = [_val(p) for p in gp_model.kernel.hyperparameters]
    result = optimize(
        trace,
        gp_model,
        x_bounds=[(0.0, 1.0), (350.0, 367.0)],
        x_trsf_fwd=[lambda x: x, lambda x: x],
        seed=42,
        no_gaussians=5,
        no_restarts=2,
        objective="lower_bound",
    )
    assert np.all(np.isfinite(result.x))
    after = [_val(p) for p in gp_model.kernel.hyperparameters]
    assert after == before


def test_entropy_estimators_registry_has_both_paper_estimators():
    assert set(ENTROPY_ESTIMATORS) == {"taylor", "lower_bound"}


def test_entropy_surface_2d_rejects_non_2d_bounds(gp_model, trace):
    # entropy_surface_2D is a 2-D-only visualization diagnostic -- it must raise a
    # clear error for other dimensions rather than silently misbehaving.
    with pytest.raises(ValueError, match="2-D-only"):
        entropy_surface_2D(
            trace,
            gp_model,
            x_bounds=[(0.0, 1.0), (350.0, 367.0), (0.0, 1.0)],
            mesh=[3, 3, 3],
            x_trsf_fwd=[lambda x: x, lambda x: x, lambda x: x],
            x_trsf_bkwd=[lambda x: x, lambda x: x, lambda x: x],
            seed=42,
            no_gaussians=5,
        )
