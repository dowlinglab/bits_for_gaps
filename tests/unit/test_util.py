"""Unit tests for the small array helpers in ``bits_for_gaps._util``."""
import numpy as np
import tensorflow as tf

from bits_for_gaps._util import make_tensor, normalize, standardize


def test_standardize_zero_mean_unit_std():
    data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    z, mean, std = standardize(data)
    assert mean == np.mean(data)
    assert std == np.std(data)
    assert abs(np.mean(z)) < 1e-12
    np.testing.assert_allclose(np.std(z), 1.0, atol=1e-12)


def test_standardize_roundtrip():
    rng = np.random.default_rng(1)
    data = rng.normal(3.0, 2.0, size=100)
    z, mean, std = standardize(data)
    np.testing.assert_allclose(z * std + mean, data, rtol=1e-12)


def test_normalize_range_and_orientation():
    data = np.array([10.0, 20.0, 30.0, 40.0])
    z, mx, mn = normalize(data)
    assert mx == 40.0 and mn == 10.0
    # normalize is reversed: the max maps to 0, the min maps to 1.
    assert z[np.argmax(data)] == 0.0
    assert z[np.argmin(data)] == 1.0
    assert z.min() >= 0.0 and z.max() <= 1.0


def test_normalize_roundtrip():
    data = np.array([-3.0, 0.5, 2.0, 7.5])
    z, mx, mn = normalize(data)
    np.testing.assert_allclose(mx - z * (mx - mn), data, rtol=1e-12)


def test_make_tensor_from_1d():
    x = np.array([1.0, 2.0, 3.0])
    t = make_tensor(x)
    assert isinstance(t, tf.Tensor)
    assert t.dtype == tf.float64
    assert tuple(t.shape) == (3, 1)


def test_make_tensor_from_2d_preserves_shape():
    x = np.arange(6.0).reshape(2, 3)
    t = make_tensor(x)
    assert t.dtype == tf.float64
    assert tuple(t.shape) == (2, 3)
    np.testing.assert_array_equal(t.numpy(), x)
