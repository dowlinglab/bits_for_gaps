"""H2O-PrOH bubble-point / dew-point phase diagram (Geankoplis Ex. 11.4-1 system).

Ported from the paper code's ``new_phase_diagram.py`` (``PhaseDiagram``). Two
interchangeable activity-coefficient sources feed the same bubble-point/dew-point
physics:

- **Ground truth (Wilson)**: :func:`wilson_gamma` calls Clapeyron.jl directly (via
  ``activity_model.py``) -- both PrOH and water coefficients come straight from the
  thermodynamic model.
- **Surrogate (GP)**: :func:`surrogate_gamma` predicts ``gamma_PrOH`` with a fitted
  ``bits_for_gaps`` GP (mean prediction; no Julia) and recovers ``gamma_water`` via
  the Gibbs-Duhem relation (``gibbs_duhem.py``) -- the GP never sees water directly.

Both feed :func:`vle_curve`, which solves the bubble-point temperature (extended
Raoult's law + Antoine vapor pressures) and the corresponding dew-point vapor
composition across a grid of liquid PrOH mole fractions -- the data behind paper
Fig 8/9's phase diagram and McCabe-Thiele column design.

Pure NumPy/SciPy (no Julia, no TensorFlow) except :func:`wilson_gamma`, which lazily
delegates to ``activity_model.py``.
"""

import numpy as np
from scipy.optimize import bisect

from . import activity_model, gibbs_duhem

# Antoine coefficients (vapor pressure in bar, T in K): log10(Pvap) = A - B / (C + T).
ANTOINE_CONSTANTS = {
    "PrOH": {"A": 5.31384, "B": 1690.864, "C": -51.804},
    "H2O": {"A": 4.6543, "B": 1435.264, "C": -64.848},
}
P_TOT_BAR = 1.01325  # 1 atm, in bar
Z_MESH = 50  # points across the z_PrOH grid, matching the paper code default


def pvap_antoine(T, constants):
    """Vapor pressure [bar] at temperature ``T`` [K] via Antoine's equation."""
    return 10.0 ** (constants["A"] - constants["B"] / (constants["C"] + T))


def wilson_gamma(z, T):
    """Ground-truth activity coefficients ``(gamma_proh, gamma_water)`` via Clapeyron."""
    return activity_model.activity_coefficients(z, T)


def surrogate_gamma(
    z, T, GPmodel, input_transform, output_transform, zmin=1e-5, zmax=0.92, n_steps=10
):
    """Surrogate activity coefficients: GP mean for PrOH, Gibbs-Duhem for water.

    Parameters
    ----------
    GPmodel : gpflow.models.GPR
        A fitted GP predicting (transformed) ``gamma_proh`` from (transformed)
        ``(z, T)`` -- e.g. the model inside a ``bits_for_gaps.state.IterationRecord``.
    input_transform, output_transform : bits_for_gaps.transforms.InputTransform /
    OutputTransform
        The same transforms the GP was trained with.

    Uses whatever hyperparameters ``GPmodel.kernel`` currently holds -- for a GP whose
    hyperparameters came from HMC (as opposed to a single MLE fit), that's an arbitrary
    single posterior sample (``bits_for_gaps.gp.run_mcmc`` leaves the kernel at HMC's
    last state), not a posterior summary. See :func:`surrogate_gamma_averaged` for a
    hyperparameter-posterior-averaged alternative when a robust curve independent of
    which single HMC sample happens to be live matters more than raw speed (e.g.
    feeding a McCabe-Thiele stage solver with a fixed initial guess -- see
    ``paper/PHASE9B_INVESTIGATION.md``). Also: ``GPmodel.kernel`` is mutated in place
    by anything that reassigns its hyperparameters (e.g.
    ``bits_for_gaps.mixture.sample_gp_posterior_mixture``, by design -- see its
    docstring), so callers that need this function's result to reflect a *specific*,
    stable hyperparameter state must not call such mutating functions on the same
    ``GPmodel`` object beforehand without restoring the kernel afterward.
    """

    def gamma_proh_curve(z_vals):
        z_vals = np.atleast_1d(np.asarray(z_vals, dtype=float))
        X_phys = np.column_stack([z_vals, np.full_like(z_vals, float(T))])
        X_gp = input_transform.forward(X_phys)
        mean, _variance = GPmodel.predict_f(X_gp)
        return output_transform.backward(mean.numpy().ravel())

    return gibbs_duhem.gamma_water_from_gamma_proh(
        gamma_proh_curve, z, zmin=zmin, zmax=zmax, n_steps=n_steps
    )


def surrogate_gamma_averaged(
    z,
    T,
    GPmodel,
    input_transform,
    output_transform,
    trace,
    n_draws=50,
    seed=42,
    zmin=1e-5,
    zmax=0.92,
    n_steps=10,
):
    """Hyperparameter-posterior-averaged surrogate activity coefficients.

    Matches the paper's own construction (the old repo's ``new_phase_diagram.py``'s
    ``PhaseDiagram.run``/``gibbs_duhem_fast``, and ``equilibrium.py``'s
    ``water_proh_eqm``, which fed the paper's *actual* Fig 9 surrogate column): draw
    ``n_draws`` independent samples from the HMC hyperparameter posterior (``trace``),
    evaluate this GP's own deterministic conditional mean under each one (not
    ``predict_f_samples`` -- no ambient-RNG, TF-non-reproducibility involved), and
    average ``gamma_proh`` pointwise. This is a Monte Carlo estimate of
    ``E_hyperparameter-posterior[gamma_proh(z, T)]``, smoother and more robust against
    any single unlucky/atypical posterior draw than :func:`surrogate_gamma`, which
    uses whatever single hyperparameter state ``GPmodel.kernel`` happens to hold.

    ``GPmodel.kernel`` is mutated in place, once per draw, and left at the *last*
    draw's hyperparameters on return -- same contract as
    ``bits_for_gaps.mixture.sample_gp_posterior_mixture``.

    Parameters
    ----------
    trace : np.ndarray, shape (n_samples, n_hyperparameters)
        HMC posterior samples (e.g. ``bits_for_gaps.state.IterationRecord.trace``), in
        ``GPmodel.kernel.hyperparameters``'s canonical order.
    n_draws : int
        Number of posterior draws to average (50, matching the paper's
        ``PhaseDiagram.n_draws``).
    """
    from bits_for_gaps.kernels import assign_hyperparameters

    rng = np.random.default_rng(seed)
    gamma_water_draws = np.zeros(n_draws)
    gamma_proh_draws = np.zeros(n_draws)
    for d in range(n_draws):
        assign_hyperparameters(GPmodel.kernel, trace[rng.integers(0, len(trace))])
        gamma_proh_draws[d], gamma_water_draws[d] = surrogate_gamma(
            z, T, GPmodel, input_transform, output_transform, zmin=zmin, zmax=zmax, n_steps=n_steps
        )
    return float(gamma_proh_draws.mean()), float(gamma_water_draws.mean())


def eqm_residual(T, z, gamma_fn, antoine=ANTOINE_CONSTANTS, P_tot=P_TOT_BAR):
    """Bubble-point residual: ``z*gamma_proh*Pvap_proh + (1-z)*gamma_water*Pvap_water - P_tot``.

    Eq (10), extended Raoult's law (``z_b^(v) P = z_b^(l) gamma_b P_b*``), summed over
    both components at the bubble point (where the two vapor partial pressures sum to
    ``P_tot``): this residual is zero exactly at the bubble-point temperature.

    ``gamma_fn(z, T) -> (gamma_proh, gamma_water)`` is not called at the pure-component
    compositions z=0/z=1 (activity-coefficient models are singular there) -- both
    coefficients are taken as 1.0, matching the paper code.
    """
    if z == 0.0 or z == 1.0:
        gamma_proh, gamma_water = 1.0, 1.0
    else:
        gamma_proh, gamma_water = gamma_fn(z, T)
    Pvap_proh = pvap_antoine(T, antoine["PrOH"])
    Pvap_water = pvap_antoine(T, antoine["H2O"])
    return z * gamma_proh * Pvap_proh + (1.0 - z) * gamma_water * Pvap_water - P_tot


def bubble_point_temperature(
    z,
    gamma_fn,
    antoine=ANTOINE_CONSTANTS,
    P_tot=P_TOT_BAR,
    Tbub_min=300.0,
    Tbub_max=400.0,
    tol=1e-6,
):
    """Solve the bubble-point temperature at liquid composition ``z`` via bisection."""
    return bisect(eqm_residual, Tbub_min, Tbub_max, args=(z, gamma_fn, antoine, P_tot), xtol=tol)


def dew_point_vapor_fraction(z, T_bub, gamma_fn, antoine=ANTOINE_CONSTANTS, P_tot=P_TOT_BAR):
    """Vapor PrOH mole fraction at the bubble point, via extended Raoult's law.

    Eq (10), solved for the vapor-phase mole fraction: ``z_proh^(v) = z_proh^(l) *
    gamma_proh * Pvap_proh / P_tot``.
    """
    if z == 1.0:
        return pvap_antoine(T_bub, antoine["PrOH"]) / P_tot
    if z == 0.0:
        return 0.0
    gamma_proh, _gamma_water = gamma_fn(z, T_bub)
    Pvap_proh = pvap_antoine(T_bub, antoine["PrOH"])
    return (z * Pvap_proh * gamma_proh) / P_tot


def vle_curve(
    gamma_fn,
    z_grid=None,
    n_grid=Z_MESH,
    antoine=ANTOINE_CONSTANTS,
    P_tot=P_TOT_BAR,
    Tbub_min=300.0,
    Tbub_max=400.0,
    tol=1e-6,
):
    """Bubble-point temperature and dew-point PrOH vapor fraction across a z_PrOH grid.

    Parameters
    ----------
    gamma_fn : callable
        ``gamma_fn(z, T) -> (gamma_proh, gamma_water)`` -- :func:`wilson_gamma` or a
        ``functools.partial`` of :func:`surrogate_gamma` binding the GP/transforms.
    z_grid : np.ndarray, optional
        Liquid PrOH mole fractions to evaluate; defaults to ``linspace(0, 1, n_grid)``.

    Returns
    -------
    z_grid, T_bub, y_proh : np.ndarray
        Liquid PrOH mole fraction, bubble-point temperature [K], and dew-point vapor
        PrOH mole fraction, one row per grid point.
    """
    if z_grid is None:
        z_grid = np.linspace(0.0, 1.0, n_grid)
    T_bub = np.array(
        [
            bubble_point_temperature(z, gamma_fn, antoine, P_tot, Tbub_min, Tbub_max, tol)
            for z in z_grid
        ]
    )
    y_proh = np.array(
        [dew_point_vapor_fraction(z, T, gamma_fn, antoine, P_tot) for z, T in zip(z_grid, T_bub)]
    )
    return z_grid, T_bub, y_proh
