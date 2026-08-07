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


def test_full_factorial_overshoot_grid_is_trimmed_and_seed_dependent():
    # d=2, n_total=10 -> levels = ceil(sqrt(10)) = 4 -> a 4x4=16-point grid, which
    # overshoots n_total=10 and must be randomly trimmed (the `rng.choice` branch
    # `test_full_factorial_exact_grid_size_needs_no_trimming` above doesn't reach,
    # since 9 is a perfect square).
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    train, test = full_factorial_design(bounds, n_train=10, n_test=0, seed=0)
    assert train.shape == (10, 2)
    assert test.shape == (0, 2)
    assert train.min() >= 0.0 and train.max() <= 1.0
    # No duplicate points -- confirms `replace=False` trimming, not resampling.
    assert len(np.unique(train, axis=0)) == 10

    # A different seed must select a different subset of the 16-point grid (proves
    # the trim is actually seeded, not silently deterministic regardless of `seed`).
    train_b, _ = full_factorial_design(bounds, n_train=10, n_test=0, seed=1)
    assert not np.array_equal(np.sort(train, axis=0), np.sort(train_b, axis=0))


## NOTE: `full_factorial_design`'s "grid too small" `ValueError` (design.py:72-73) is
## unreachable for any (bounds, n_train, n_test): `levels = ceil(n_total ** (1/d))`
## guarantees `levels ** d >= n_total`. Defensive dead code, not a bug -- marked
## `# pragma: no cover` at the source rather than faked with a test.
