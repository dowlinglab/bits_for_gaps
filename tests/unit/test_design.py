"""Unit tests for space-filling designs (pure NumPy/SciPy, no TensorFlow)."""

import numpy as np

from bits_for_gaps.design import full_factorial_design, latin_hypercube_design


def test_lhs_shapes_and_bounds():
    bounds = [(1e-6, 0.999), (350.0, 367.0)]
    train, test = latin_hypercube_design(bounds, n_train=10, n_test=10, seed=10)
    assert train.shape == (10, 2)
    assert test.shape == (10, 2)
    allpts = np.vstack([train, test])
    for i, (lo, hi) in enumerate(bounds):
        assert allpts[:, i].min() >= lo
        assert allpts[:, i].max() <= hi


def test_lhs_is_deterministic_with_seed():
    bounds = [(0.0, 1.0), (0.0, 10.0), (-5.0, 5.0)]
    a_train, a_test = latin_hypercube_design(bounds, n_train=12, n_test=4, seed=42)
    b_train, b_test = latin_hypercube_design(bounds, n_train=12, n_test=4, seed=42)
    np.testing.assert_array_equal(a_train, b_train)
    np.testing.assert_array_equal(a_test, b_test)


def test_lhs_different_seeds_differ():
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    a, _ = latin_hypercube_design(bounds, n_train=8, seed=1)
    b, _ = latin_hypercube_design(bounds, n_train=8, seed=2)
    assert not np.allclose(a, b)


def test_full_factorial_shapes_and_bounds():
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    train, test = full_factorial_design(bounds, n_train=12, n_test=4, seed=0)
    assert train.shape[0] == 12 and test.shape[0] == 4
    allpts = np.vstack([train, test])
    assert allpts.min() >= 0.0 and allpts.max() <= 1.0


def test_full_factorial_exact_grid_size_needs_no_trimming():
    # levels = ceil(9 ** (1/2)) = 3 -> a 3x3 grid has exactly 9 points, no trimming
    # (and therefore no `seed`-dependent `rng.choice` call) needed.
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    train, test = full_factorial_design(bounds, n_train=9, n_test=0, seed=None)
    assert train.shape == (9, 2)
    assert test.shape == (0, 2)


## NOTE: `full_factorial_design`'s "grid too small" `ValueError` (design.py:74-75) is
## unreachable for any (bounds, n_train, n_test): `levels = ceil(n_total ** (1/d))`
## guarantees `levels ** d >= n_total`. Defensive dead code, not a bug -- not exercised
## here since no input triggers it.
