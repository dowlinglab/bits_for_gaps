"""Unit tests for the per-dimension Input/Output transform classes."""

import numpy as np
import pytest

from bits_for_gaps.transforms import InputTransform, OutputTransform


def test_input_transform_identity_default():
    t = InputTransform(ndim=2)
    X = np.array([[0.1, 350.0], [0.9, 366.0]])
    np.testing.assert_array_equal(t.forward(X), X)
    np.testing.assert_array_equal(t.backward(X), X)
    assert t.ndim == 2


def test_input_transform_custom_per_dimension():
    t = InputTransform(
        forward_fns=[lambda x: x * 2, lambda x: x - 1],
        backward_fns=[lambda x: x / 2, lambda x: x + 1],
    )
    X = np.array([[1.0, 10.0], [2.0, 20.0]])
    fwd = t.forward(X)
    np.testing.assert_allclose(fwd, np.array([[2.0, 9.0], [4.0, 19.0]]))
    np.testing.assert_allclose(t.backward(fwd), X)


def test_input_transform_requires_ndim_or_both_fn_lists():
    with pytest.raises(ValueError):
        InputTransform()
    with pytest.raises(ValueError):
        InputTransform(forward_fns=[lambda x: x])


def test_input_transform_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        InputTransform(forward_fns=[lambda x: x, lambda x: x], backward_fns=[lambda x: x])


def test_input_transform_1d_input_is_atleast_2d():
    # A bare length-d sequence (one point, not a batch) must be promoted to a single
    # (1, d) row and transformed correctly, not just reshaped.
    t = InputTransform(
        forward_fns=[lambda x: x * 2, lambda x: x - 1],
        backward_fns=[lambda x: x / 2, lambda x: x + 1],
    )
    out = t.forward([0.5, 1.5])
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out, [[1.0, 0.5]])
    back = t.backward(out[0])
    assert back.shape == (1, 2)
    np.testing.assert_allclose(back, [[0.5, 1.5]])


def test_output_transform_identity_default():
    t = OutputTransform()
    y = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(t.forward(y), y)
    np.testing.assert_array_equal(t.backward(y), y)


def test_output_transform_custom_roundtrip():
    t = OutputTransform(forward_fn=np.log, backward_fn=np.exp)
    y = np.array([1.0, 2.0, 5.0])
    np.testing.assert_allclose(t.backward(t.forward(y)), y, rtol=1e-12)
