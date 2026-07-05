"""Unit tests for ``vle_distillation.equilibrium`` (pure NumPy/SciPy -- no Julia)."""

import numpy as np
from vle_distillation.equilibrium import make_equilibrium_function


def test_interpolates_through_given_points():
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y = np.array([0.0, 0.4, 0.6, 0.85, 1.0])
    equil = make_equilibrium_function(x, y)
    np.testing.assert_allclose(equil(x), y, atol=1e-9)


def test_handles_unsorted_input():
    x = np.array([0.5, 0.0, 1.0, 0.25, 0.75])
    y = np.array([0.6, 0.0, 1.0, 0.4, 0.85])
    equil_unsorted = make_equilibrium_function(x, y)
    equil_sorted = make_equilibrium_function(np.sort(x), y[np.argsort(x)])
    probe = np.linspace(0.0, 1.0, 20)
    np.testing.assert_allclose(equil_unsorted(probe), equil_sorted(probe))


def test_extrapolates_beyond_range():
    x = np.array([0.2, 0.4, 0.6, 0.8])
    y = np.array([0.3, 0.5, 0.7, 0.9])
    equil = make_equilibrium_function(x, y)
    # fill_value="extrapolate" -- should not raise, and should return a finite value.
    assert np.isfinite(equil(0.0))
    assert np.isfinite(equil(1.0))
