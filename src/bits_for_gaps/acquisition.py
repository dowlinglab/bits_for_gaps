"""Entropy-maximization acquisition function (2nd-order Taylor estimator).

Moved from the paper code's ``driver_new.py`` (``adaptiveEntropy.entropy_objective`` /
``gen_entropy_surface_data_2D`` / ``optimize_2D``). Pure functions over explicit
arguments -- no disk I/O; ``entropy.py`` provides the underlying mixture-entropy math.

Phase 5: ``entropy_objective`` and ``optimize`` (was ``optimize_2D``) are dimension-
general -- kernel hyperparameters are assigned via ``kernels.assign_hyperparameters``
(the canonical-order contract, not hardcoded names), and ``optimize``'s Sobol restarts
and bounds scale with ``len(x_bounds)``. ``entropy_surface_2D`` stays 2-D-only: a dense
grid is exponential in d, and it is a visualization diagnostic that does not feed the
acquisition -- it raises a clear error for d != 2 rather than silently degrading.

Phase 9c: ``entropy_objective`` is called many times per ``optimize``/
``entropy_surface_2D`` call (once per Sobol restart x scipy.optimize.minimize
iteration, or once per grid point), each time reassigning ``GPmodel.kernel``'s
hyperparameters -- it used to leave the kernel at whichever sample the *last* such call
happened to use, not any meaningful state. Since ``sampler.py``'s ``run()`` calls
``optimize``/``entropy_surface_2D`` on the same ``GPmodel`` it then stores in
``IterationRecord.GPmodel`` (and optionally checkpoints), every iteration's returned
model used to carry this arbitrary leftover state -- the same class of footgun Phase 9b
found in ``mixture.sample_gp_posterior_mixture`` (see
``paper/PHASE9B_INVESTIGATION.md``). Now saves/restores the kernel around each call, so
``GPmodel`` is unchanged after ``entropy_objective`` returns -- behavior-preserving for
the entropy value itself (computed before the restore).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

from . import entropy as max_ent_design
from . import kernels


def entropy_objective(xStarGP, trace, GPmodel, seed, no_gaussians):
    """Negative 2nd-order-Taylor mixture entropy of the GP predictive posterior at a point.

    Negated because ``optimize`` minimizes it to maximize entropy. Dimension-general:
    ``xStarGP`` may have any number of columns. ``GPmodel.kernel``'s hyperparameters are
    reassigned once per posterior sample during the call, then restored before
    returning (see the module docstring) -- ``GPmodel`` is unchanged from the caller's
    perspective.
    """
    xStarGP = xStarGP.reshape(-1, 1).T
    np.random.seed(seed)
    subset_indices = np.random.choice(len(trace), size=no_gaussians, replace=False)
    sub_samples = trace[subset_indices]
    saved_hyperparameters = kernels.save_hyperparameters(GPmodel.kernel)
    try:
        means = []
        variances = []
        for sample in sub_samples:
            kernels.assign_hyperparameters(GPmodel.kernel, sample)
            mean, variance = GPmodel.predict_f(xStarGP, full_cov=True)
            means.append(mean.numpy().squeeze())
            variances.append(variance.numpy().squeeze())
    finally:
        kernels.assign_hyperparameters(GPmodel.kernel, saved_hyperparameters)
    means = np.array(means)
    variances = np.array(variances)
    H = max_ent_design.second_order_entropy(
        weights=np.ones(no_gaussians) * 1 / no_gaussians, means=means, covs=variances
    )
    return -H


def entropy_surface_2D(trace, GPmodel, x_bounds, mesh, x_trsf_fwd, x_trsf_bkwd, seed, no_gaussians):
    """Entropy field over a 2-D grid spanning ``x_bounds`` at ``mesh`` points per dim.

    Moved from ``adaptiveEntropy.gen_entropy_surface_data_2D``.

    2-D-ONLY VISUALIZATION DIAGNOSTIC (Phase 5): a dense grid is exponential in the
    input dimension d, so this is intentionally not generalized to N-D. It does not
    feed the acquisition -- see ``optimize`` for the N-D-general acquisition path.

    Returns
    -------
    np.ndarray, shape (mesh[0] * mesh[1], 3)
        Columns ``[x1, x2, H]`` in the physical (untransformed) input space.

    Raises
    ------
    ValueError
        If ``len(x_bounds) != 2``.
    """
    if len(x_bounds) != 2:
        raise ValueError(
            f"entropy_surface_2D is a 2-D-only visualization diagnostic (got "
            f"{len(x_bounds)} input dimensions); it does not feed the acquisition -- "
            f"see optimize() for the N-D-general acquisition optimizer."
        )
    x1 = np.linspace(*x_bounds[0], mesh[0])
    x2 = np.linspace(*x_bounds[1], mesh[1])
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    XStar = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T
    XStarGP = np.column_stack([fwd(XStar[:, j]) for j, fwd in enumerate(x_trsf_fwd)])
    H = np.array([-entropy_objective(x, trace, GPmodel, seed, no_gaussians) for x in XStarGP])
    XStar = np.column_stack([bkwd(XStarGP[:, j]) for j, bkwd in enumerate(x_trsf_bkwd)])
    return np.column_stack([XStar, H])


def optimize(trace, GPmodel, x_bounds, x_trsf_fwd, seed, no_gaussians, no_restarts):
    """Multistart optimization of the entropy objective over ``x_bounds`` (N-D).

    Moved from ``adaptiveEntropy.optimize_2D``; generalized (Phase 5) to arbitrary
    input dimension ``d = len(x_bounds)`` -- this is the acquisition path an N-D run
    actually depends on (contrast ``entropy_surface_2D``, a 2-D-only diagnostic).
    Sobol-scrambled restarts drawn in the physical space, optimized in GP (transformed)
    space.

    For ``d == 2`` this reproduces the pre-Phase-5 ``optimize_2D`` bit-for-bit: the
    vectorized bound-scaling below (``lo + x0 * (hi - lo)``) is the same floating-point
    operation order as the original per-dimension expression (IEEE 754 addition is
    commutative, so ``lo[j] + x0[j] * (hi[j] - lo[j])`` and the original
    ``x0[j] * (hi[j] - lo[j]) + lo[j]`` round identically).

    Returns
    -------
    scipy.optimize.OptimizeResult
        The best restart (lowest ``entropy_objective`` value); ``result.x`` is in GP
        (transformed) input space, ``-result.fun`` is the maximized entropy.
    """
    lo = np.array([b[0] for b in x_bounds], dtype=float)
    hi = np.array([b[1] for b in x_bounds], dtype=float)
    best_result = None
    best_value = np.inf
    sobol = Sobol(d=len(x_bounds), scramble=True, seed=seed)
    XBndsGP = [(f(b[0]), f(b[1])) for f, b in zip(x_trsf_fwd, x_bounds)]
    for _ in range(no_restarts):
        x0 = sobol.random()[0]
        x0 = lo + x0 * (hi - lo)
        x0GP = np.array([f(x0[i]) for i, f in enumerate(x_trsf_fwd)])
        result = minimize(
            entropy_objective, x0=x0GP, bounds=XBndsGP, args=(trace, GPmodel, seed, no_gaussians)
        )
        if result.fun < best_value:
            best_value = result.fun
            best_result = result
    return best_result
