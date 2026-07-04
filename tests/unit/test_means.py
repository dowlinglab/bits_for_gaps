"""Unit tests for FixedInverseMean.

Pins the physics-informed mean function's closed form, output shape, its dependence on
the mole-fraction column only, and its monotonic decrease in that column.
"""
import numpy as np
import pytest

from bits_for_gaps.means import FixedInverseMean


def _reference(x, epsilon=0.6, lambda_bc=0.1, eps=1e-8):
    base = (2.0 - x) / (x + epsilon)
    safe_base = np.maximum(base, eps)
    return (np.power(safe_base, lambda_bc) - 1.0) / lambda_bc


def test_output_shape():
    mean = FixedInverseMean()
    X = np.array([[0.2, 355.0], [0.5, 360.0], [0.9, 366.0]])
    out = mean(X).numpy()
    assert out.shape == (3, 1)


def test_matches_closed_form():
    mean = FixedInverseMean()
    x = np.array([0.05, 0.3, 0.7, 0.95])
    X = np.column_stack([x, np.full_like(x, 358.0)])
    out = mean(X).numpy().ravel()
    np.testing.assert_allclose(out, _reference(x), rtol=1e-10)


def test_zero_crossing_at_base_equals_one():
    # base = 1 when (2 - x) = (x + 0.6) => x = 0.7, where the transform is exactly 0.
    mean = FixedInverseMean()
    out = mean(np.array([[0.7, 0.0]])).numpy()[0, 0]
    assert out == pytest.approx(0.0, abs=1e-12)


def test_depends_only_on_first_column():
    mean = FixedInverseMean()
    a = mean(np.array([[0.4, 350.0]])).numpy()
    b = mean(np.array([[0.4, 999.0]])).numpy()
    np.testing.assert_array_equal(a, b)


def test_monotonic_decreasing_in_mole_fraction():
    mean = FixedInverseMean()
    x = np.linspace(0.01, 0.99, 50)
    X = np.column_stack([x, np.zeros_like(x)])
    out = mean(X).numpy().ravel()
    assert np.all(np.diff(out) < 0)


def test_custom_parameters():
    mean = FixedInverseMean(epsilon=0.3, lambda_bc=0.2)
    x = np.array([0.1, 0.6])
    X = np.column_stack([x, np.zeros_like(x)])
    out = mean(X).numpy().ravel()
    np.testing.assert_allclose(out, _reference(x, epsilon=0.3, lambda_bc=0.2), rtol=1e-10)
