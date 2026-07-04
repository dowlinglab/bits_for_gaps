"""Unit tests for the Gibbs-Duhem correction (``vle_distillation.gibbs_duhem``).

Pure NumPy -- no Julia. Validated against a closed-form case: for
``gamma_proh(z) = exp(a*z)``, the binary Gibbs-Duhem relation
``d ln(gamma_water) = -z/(1-z) d ln(gamma_proh)`` integrates analytically (via
``-a*z/(1-z) = a - a/(1-z)``) to
``ln(gamma_water) = a*(z - zmin) + a*(ln(1-z) - ln(1-zmin))``.
"""
import numpy as np
import pytest

from vle_distillation.gibbs_duhem import gamma_water_from_gamma_proh


def _analytic_log_gamma_water(a, z_star, zmin):
    return a * (z_star - zmin) + a * (np.log(1.0 - z_star) - np.log(1.0 - zmin))


def test_ideal_mixture_gives_ideal_water_coefficient():
    # gamma_proh == 1 everywhere => d ln(gamma_water) == 0 => gamma_water == 1.
    gamma_proh, gamma_water = gamma_water_from_gamma_proh(lambda z: np.ones_like(z), 0.5)
    assert gamma_proh == pytest.approx(1.0)
    assert gamma_water == pytest.approx(1.0)


@pytest.mark.parametrize("a,z_star", [(2.0, 0.5), (1.0, 0.3), (-1.5, 0.7)])
def test_matches_closed_form_within_discretization_error(a, z_star):
    zmin = 1e-5
    gamma_proh, gamma_water = gamma_water_from_gamma_proh(
        lambda z: np.exp(a * z), z_star, zmin=zmin)
    assert gamma_proh == pytest.approx(np.exp(a * z_star), rel=1e-10)
    expected = np.exp(_analytic_log_gamma_water(a, z_star, zmin))
    # n_steps=10 (the paper code's / this module's default) trapezoidal rule --
    # a coarse but fast approximation; verify it's within ~1% of the closed form.
    assert gamma_water == pytest.approx(expected, rel=1e-2)


def test_converges_to_closed_form_as_n_steps_increases():
    a, z_star, zmin = 2.0, 0.5, 1e-5
    expected = np.exp(_analytic_log_gamma_water(a, z_star, zmin))
    _, coarse = gamma_water_from_gamma_proh(lambda z: np.exp(a * z), z_star, zmin=zmin,
                                            n_steps=10)
    _, fine = gamma_water_from_gamma_proh(lambda z: np.exp(a * z), z_star, zmin=zmin,
                                          n_steps=1000)
    assert abs(fine - expected) < abs(coarse - expected)
    assert fine == pytest.approx(expected, rel=1e-6)


def test_z_star_clipped_to_zmax():
    # z_star beyond zmax should give the same result as z_star == zmax.
    gamma_proh_fn = lambda z: np.exp(1.5 * z)
    r_at_zmax = gamma_water_from_gamma_proh(gamma_proh_fn, 0.92, zmax=0.92)
    r_beyond = gamma_water_from_gamma_proh(gamma_proh_fn, 0.99, zmax=0.92)
    assert r_at_zmax == r_beyond
