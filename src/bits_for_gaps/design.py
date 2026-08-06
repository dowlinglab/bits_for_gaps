"""Space-filling initial designs over a bounded input space.

Pure, N-dimensional, and return arrays -- no files, no thermodynamics.
"""

from __future__ import annotations

from itertools import product
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.stats import qmc

Bounds = Sequence[Tuple[float, float]]


def _scale_to_bounds(unit_sample: np.ndarray, bounds: Bounds) -> np.ndarray:
    """Map samples on the unit cube to the physical box ``bounds``."""
    out = np.array(unit_sample, dtype=float)
    for i, (low, high) in enumerate(bounds):
        out[:, i] = low + (high - low) * out[:, i]
    return out


def _split(points: np.ndarray, n_test: int, seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle and split into (train, test)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(points))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return points[train_idx], points[test_idx]


def latin_hypercube_design(
    bounds: Bounds, n_train: int, n_test: int = 0, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Latin-hypercube design over ``bounds``, split into train/test.

    Parameters
    ----------
    bounds : sequence of (low, high)
        Per-dimension bounds; ``len(bounds)`` sets the dimensionality.
    n_train, n_test : int
        Number of training and test points.
    seed : int, optional
        Seed for both the LHS sampler and the train/test split.

    Returns
    -------
    (train, test) : tuple of np.ndarray
        Arrays of shape ``(n_train, d)`` and ``(n_test, d)``.
    """
    d = len(bounds)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    sample = sampler.random(n=n_train + n_test)
    points = _scale_to_bounds(sample, bounds)
    return _split(points, n_test, seed)


def full_factorial_design(
    bounds: Bounds, n_train: int, n_test: int = 0, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Full-factorial grid over ``bounds``, trimmed to ``n_train + n_test`` and split.

    The number of levels per dimension is ceil((n_train + n_test) ** (1/d)); if the grid
    overshoots the requested count it is randomly trimmed.
    """
    d = len(bounds)
    n_total = n_train + n_test
    levels = int(np.ceil(n_total ** (1 / d)))
    grid_unit = np.array(list(product(*[np.linspace(0, 1, levels) for _ in range(d)])))

    if len(grid_unit) < n_total:
        raise ValueError(f"Factorial grid has {len(grid_unit)} points, need {n_total}.")
    if len(grid_unit) > n_total:
        rng = np.random.default_rng(seed)
        grid_unit = rng.choice(grid_unit, size=n_total, replace=False)

    points = _scale_to_bounds(grid_unit, bounds)
    return _split(points, n_test, seed)
