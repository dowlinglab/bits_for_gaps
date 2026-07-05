"""Gibbs-Duhem correction: recover gamma_H2O from a modeled gamma_PrOH curve.

Ported from the paper code's ``new_phase_diagram.py`` (``PhaseDiagram.gibbs_duhem_fast``).
The GP surrogate used in this case study models only ``gamma_PrOH(z, T)`` -- the water
coefficient is derived from it via the binary Gibbs-Duhem relation, not learned by a
second GP output. For an isothermal, isobaric binary mixture:

    d ln(gamma_water) = -z / (1 - z) * d ln(gamma_proh)

integrated from a dilute reference ``z = zmin`` (where ``gamma_water ~= 1``, i.e. the
integration constant ``ln(gamma_water) = 0``) up to ``z = z_star``.
"""

import numpy as np


def gamma_water_from_gamma_proh(gamma_proh_fn, z_star, zmin=1e-5, zmax=0.92, n_steps=10):
    """Integrate the Gibbs-Duhem relation to get gamma_water at ``z_star``.

    Parameters
    ----------
    gamma_proh_fn : callable
        ``gamma_proh_fn(z_array) -> gamma_proh_array`` -- gamma_PrOH along a 1-D grid
        of liquid mole fractions (e.g. a GP mean prediction at fixed T).
    z_star : float
        Liquid mole fraction of PrOH at which to evaluate (assumed ``0 < z_star``;
        callers handle the ``z_star in {0, 1}`` pure-component edge cases).
    zmin, zmax : float
        Integration bounds; ``z_star`` is clipped to ``zmax`` (matches the paper code:
        gamma_PrOH is only modeled up to z=0.92).
    n_steps : int
        Number of grid points for the trapezoidal rule (10, matching the paper code).

    Returns
    -------
    (gamma_proh, gamma_water) : tuple of float
        ``gamma_proh`` at ``z_star`` (the last grid point) and the Gibbs-Duhem-derived
        ``gamma_water``.
    """
    z_hi = min(z_star, zmax)
    z_vals = np.linspace(zmin, z_hi, n_steps)
    gamma_proh = np.asarray(gamma_proh_fn(z_vals), dtype=float)
    integrand = -z_vals / (1.0 - z_vals)
    log_gamma_water = np.trapz(integrand, x=np.log(gamma_proh))
    gamma_water = np.exp(log_gamma_water)
    return float(gamma_proh[-1]), float(gamma_water)
