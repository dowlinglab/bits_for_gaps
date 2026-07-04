"""Entropy-maximization acquisition function (2nd-order Taylor estimator).

Moved from the paper code's ``driver_new.py`` (``adaptiveEntropy.entropy_objective`` /
``gen_entropy_surface_data_2D`` / ``optimize_2D``). Pure functions over explicit
arguments -- no disk I/O; ``entropy.py`` provides the underlying mixture-entropy math.

TODO(Phase 5): hardcodes 2 input dimensions (grid/meshgrid, Sobol ``d=2``) and assigns
kernel hyperparameters by name (assumes the 3-hyperparameter ``AnisotropicSE`` kernel),
mirroring the sampler-level TODO. Keep the ``*_2D`` names until then.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

from . import entropy as max_ent_design


def entropy_objective(xStarGP, trace, GPmodel, seed, no_gaussians):
    """Negative 2nd-order-Taylor mixture entropy of the GP predictive posterior at a point.

    Negated because ``optimize_2D`` minimizes it to maximize entropy.
    """
    xStarGP = xStarGP.reshape(-1, 1).T
    np.random.seed(seed)
    subset_indices = np.random.choice(len(trace), size=no_gaussians, replace=False)
    sub_samples = trace[subset_indices]
    means = []
    variances = []
    for sample in sub_samples:
        GPmodel.kernel.std_dev.assign(sample[0])
        GPmodel.kernel.lengthscale_1.assign(sample[1])
        GPmodel.kernel.lengthscale_2.assign(sample[2])
        mean, variance = GPmodel.predict_f(xStarGP, full_cov=True)
        means.append(mean.numpy().squeeze())
        variances.append(variance.numpy().squeeze())
    means = np.array(means)
    variances = np.array(variances)
    H = max_ent_design.second_order_entropy(
        weights=np.ones(no_gaussians) * 1 / no_gaussians, means=means, covs=variances)
    return -H


def entropy_surface_2D(trace, GPmodel, x_bounds, mesh, x_trsf_fwd, x_trsf_bkwd, seed,
                       no_gaussians):
    """Entropy field over a 2-D grid spanning ``x_bounds`` at ``mesh`` points per dim.

    Moved from ``adaptiveEntropy.gen_entropy_surface_data_2D``.

    Returns
    -------
    np.ndarray, shape (mesh[0] * mesh[1], 3)
        Columns ``[x1, x2, H]`` in the physical (untransformed) input space.
    """
    x1 = np.linspace(*x_bounds[0], mesh[0])
    x2 = np.linspace(*x_bounds[1], mesh[1])
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    XStar = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T
    XStarGP = np.column_stack([fwd(XStar[:, j]) for j, fwd in enumerate(x_trsf_fwd)])
    H = np.array([
        -entropy_objective(x, trace, GPmodel, seed, no_gaussians) for x in XStarGP
    ])
    XStar = np.column_stack([bkwd(XStarGP[:, j]) for j, bkwd in enumerate(x_trsf_bkwd)])
    return np.column_stack([XStar, H])


def optimize_2D(trace, GPmodel, x_bounds, x_trsf_fwd, seed, no_gaussians, no_restarts):
    """Multistart optimization of the entropy objective over ``x_bounds`` (2-D).

    Moved from ``adaptiveEntropy.optimize_2D``. Sobol-scrambled restarts drawn in the
    physical space, optimized in GP (transformed) space.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The best restart (lowest ``entropy_objective`` value); ``result.x`` is in GP
        (transformed) input space, ``-result.fun`` is the maximized entropy.
    """
    best_result = None
    best_value = np.inf
    sobol = Sobol(d=2, scramble=True, seed=seed)
    XBndsGP = [(f(b[0]), f(b[1])) for f, b in zip(x_trsf_fwd, x_bounds)]
    for _ in range(no_restarts):
        x0 = sobol.random()[0]
        x0 = np.array([x0[0] * (x_bounds[0][1] - x_bounds[0][0]) + x_bounds[0][0],
                       x0[1] * (x_bounds[1][1] - x_bounds[1][0]) + x_bounds[1][0]])
        x0GP = np.array([f(x0[i]) for i, f in enumerate(x_trsf_fwd)])
        result = minimize(entropy_objective, x0=x0GP, bounds=XBndsGP,
                          args=(trace, GPmodel, seed, no_gaussians))
        if result.fun < best_value:
            best_value = result.fun
            best_result = result
    return best_result
