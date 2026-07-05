"""Unit tests for Phase 9c's public-API input validation on ``adaptiveEntropy``/
``BitsForGaps``: clear, early errors instead of a cryptic failure deep inside
GPflow/TensorFlow or a confusing downstream shape mismatch.
"""

import gpflow
import numpy as np
import pytest
import tensorflow_probability as tfp

from bits_for_gaps.kernels import AnisotropicSE
from bits_for_gaps.sampler import BitsForGaps, adaptiveEntropy

BOUNDS_2D = [(0.0, 1.0), (350.0, 367.0)]


def _fwd_model(x1, x2):
    return [float(np.sin(x1) + x2)]


def _build(bounds=BOUNDS_2D, kernel=None, iters=1):
    return adaptiveEntropy(
        exp_name="validation_test",
        iters=iters,
        x_bounds=bounds,
        likelihood_var=0.05,
        mean_fxn=gpflow.mean_functions.Zero(),
        kernel_fxn=kernel or AnisotropicSE(),
        fwd_model=_fwd_model,
        fwd_model_args=(),
    )


## ---------------------------------------------------------------------------
## __init__: bounds and kernel/bounds ndim validation
## ---------------------------------------------------------------------------


def test_valid_construction_succeeds():
    _build()  # must not raise


def test_rejects_lo_greater_than_hi():
    with pytest.raises(ValueError, match="lo must be strictly less than hi"):
        _build(bounds=[(1.0, 0.0), (350.0, 367.0)])


def test_rejects_lo_equal_hi():
    with pytest.raises(ValueError, match="lo must be strictly less than hi"):
        _build(bounds=[(0.5, 0.5), (350.0, 367.0)])


def test_rejects_malformed_bound_entry():
    with pytest.raises(ValueError, match="lo, hi. pair"):
        _build(bounds=[(0.0, 1.0, 2.0), (350.0, 367.0)])


def test_rejects_bounds_kernel_ndim_mismatch():
    # AnisotropicSE() defaults to ndim=2; giving it 3 bounds must fail clearly instead
    # of a cryptic shape error the first time the kernel is evaluated.
    with pytest.raises(ValueError, match="ndim=2"):
        _build(bounds=[(0.0, 1.0), (350.0, 367.0), (0.0, 1.0)])


def test_bits_for_gaps_facade_validates_too():
    # BitsForGaps is a thin subclass -- the same validation must apply.
    kernel_1d = AnisotropicSE(
        lengthscale_priors=[tfp.distributions.Gamma(concentration=4.0, rate=2.0)]
    )
    with pytest.raises(ValueError, match="lo must be strictly less than hi"):
        BitsForGaps(black_box=_fwd_model, bounds=[(1.0, 0.0)], kernel=kernel_1d)


## ---------------------------------------------------------------------------
## run(): config + X_init/y_init shape validation
## ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,bad_value",
    [
        ("noIters", 0),
        ("noIters", -1),
        ("noIters", 1.5),
        ("noSamples", 0),
        ("noChains", 0),
        ("noLeapfrogSteps", 0),
        ("noGaussians", 0),
        ("noRestarts", 0),
        ("noBurnIn", -1),
        ("noAdaptSteps", -1),
    ],
)
def test_run_rejects_non_positive_config(name, bad_value):
    s = _build()
    setattr(s, name, bad_value)
    X = np.array([[0.5, 358.0]])
    y = np.array([1.0])
    with pytest.raises(ValueError, match=name):
        s.run(X, y)


@pytest.mark.parametrize("bad_step_size", [0.0, -0.1])
def test_run_rejects_non_positive_step_size(bad_step_size):
    s = _build()
    s.stepSize = bad_step_size
    with pytest.raises(ValueError, match="stepSize"):
        s.run(np.array([[0.5, 358.0]]), np.array([1.0]))


@pytest.mark.parametrize("bad_target_accept", [0.0, 1.0, -0.1, 1.1])
def test_run_rejects_target_accept_outside_unit_interval(bad_target_accept):
    s = _build()
    s.targetAccept = bad_target_accept
    with pytest.raises(ValueError, match="targetAccept"):
        s.run(np.array([[0.5, 358.0]]), np.array([1.0]))


def test_run_rejects_non_positive_adapt_rate():
    s = _build()
    s.adaptRate = 0.0
    with pytest.raises(ValueError, match="adaptRate"):
        s.run(np.array([[0.5, 358.0]]), np.array([1.0]))


def test_run_rejects_x_init_dimension_mismatch():
    s = _build()  # 2-D bounds
    X_init = np.array([[0.5]])  # only 1 column
    y_init = np.array([1.0])
    with pytest.raises(ValueError, match="columns"):
        s.run(X_init, y_init)


def test_run_rejects_x_y_init_row_mismatch():
    s = _build()
    X_init = np.array([[0.5, 358.0], [0.6, 359.0]])
    y_init = np.array([1.0])  # only 1 entry for 2 rows
    with pytest.raises(ValueError, match="rows"):
        s.run(X_init, y_init)


## ---------------------------------------------------------------------------
## call_model: black-box output shape validation
## ---------------------------------------------------------------------------


def test_call_model_rejects_scalar_black_box_output():
    def bad_fwd_model(x1, x2):
        return float(x1 + x2)  # bare scalar, not a sequence -- the common mistake

    s = _build()
    s.FwdModel = bad_fwd_model
    with pytest.raises(ValueError, match="sequence with at least one element"):
        s.call_model([0.5, 358.0], np.array([[0.4, 357.0]]), np.array([1.0]))


def test_call_model_rejects_empty_black_box_output():
    def empty_fwd_model(x1, x2):
        return []

    s = _build()
    s.FwdModel = empty_fwd_model
    with pytest.raises(ValueError, match="sequence with at least one element"):
        s.call_model([0.5, 358.0], np.array([[0.4, 357.0]]), np.array([1.0]))


def test_call_model_accepts_well_formed_black_box_output():
    s = _build()
    XData_new, yData_new = s.call_model([0.5, 358.0], np.array([[0.4, 357.0]]), np.array([1.0]))
    assert XData_new.shape == (2, 2)
    assert yData_new.shape == (2, 1)
    np.testing.assert_array_equal(XData_new, [[0.4, 357.0], [0.5, 358.0]])
    expected_y = _fwd_model(0.5, 358.0)[0]
    assert yData_new[-1, 0] == pytest.approx(expected_y)


def test_call_model_extracts_first_element_of_multi_element_output():
    # A black box that (like Clapeyron's activity model) returns more than one value
    # must have only its first element appended -- the rest is silently dropped, by
    # design (call_model docstring): the GP models a single scalar output.
    def multi_output_fwd_model(x1, x2):
        return [float(x1 + x2), 999.0, -1.0]

    s = _build()
    s.FwdModel = multi_output_fwd_model
    XData_new, yData_new = s.call_model([0.5, 358.0], np.array([[0.4, 357.0]]), np.array([1.0]))
    assert yData_new[-1, 0] == pytest.approx(0.5 + 358.0)
