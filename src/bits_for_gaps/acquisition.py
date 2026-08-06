"""Entropy-maximization acquisition function (2nd-order Taylor estimator).

Pure functions over explicit arguments -- no disk I/O; ``entropy.py`` provides the
underlying mixture-entropy math.

``entropy_objective`` and ``optimize`` are dimension-general -- kernel hyperparameters
are assigned via ``kernels.assign_hyperparameters`` (the canonical-order contract, not
hardcoded names), and ``optimize``'s Sobol restarts and bounds scale with
``len(x_bounds)``. ``entropy_surface_2D`` stays 2-D-only: a dense grid is exponential in
d, and it is a visualization diagnostic that does not feed the acquisition -- it raises
a clear error for d != 2 rather than silently degrading.

CAUTION -- kernel mutation: ``entropy_objective`` is called many times per
``optimize``/``entropy_surface_2D`` call (once per Sobol restart x
``scipy.optimize.minimize`` iteration, or once per grid point), each time reassigning
``GPmodel.kernel``'s hyperparameters. It saves the kernel's hyperparameters before each
call and restores them in a ``finally``, so ``GPmodel`` is unchanged after
``entropy_objective`` returns regardless of how it exits -- ``sampler.py``'s ``run()``
relies on this, since it calls ``optimize``/``entropy_surface_2D`` on the same
``GPmodel`` it then stores in ``IterationRecord.GPmodel`` (and optionally checkpoints).

The paper derives TWO entropy estimators -- the 2nd-order Taylor approximation
(``entropy.second_order_entropy``, the one that actually drove acquisition in the paper
and remains the default here) and a closed-form lower bound
(``entropy.entropy_lower_bound``, paper Theorem/SI-2). ``objective`` selects between
them (default ``"taylor"``).
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import gpflow
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.stats.qmc import Sobol

from . import entropy as max_ent_design
from . import kernels

# The two paper-derived entropy estimators, selectable via `objective` below. Both
# accept (weights, means, covs_or_variances) positionally -- entropy_lower_bound's
# univariate-GMM assumption holds for every call site below, since entropy_objective
# always evaluates the entropy of the GP's (scalar) OUTPUT at one point, never a
# multi-point joint, so `means`/`variances` are always 1-D scalar arrays regardless of
# the input dimension d.
ENTROPY_ESTIMATORS = {
    "taylor": max_ent_design.second_order_entropy,
    "lower_bound": max_ent_design.entropy_lower_bound,
}


def entropy_objective(
    xStarGP: np.ndarray,
    trace: np.ndarray,
    GPmodel: gpflow.models.GPR,
    seed: int,
    no_gaussians: int,
    objective: str = "taylor",
) -> float:
    """Negative mixture entropy of the GP predictive posterior at a point.

    Implements Eq (1)'s entropy ``H{f(x*)}``, so that ``optimize`` (Eq 2: ``max_x*
    H{f(x*)}``) can call ``scipy.optimize.minimize`` on this function's negation.
    Negated because ``optimize`` minimizes it to maximize entropy. Dimension-general:
    ``xStarGP`` may have any number of columns. ``GPmodel.kernel``'s hyperparameters are
    reassigned once per posterior sample during the call, then restored before
    returning (see the module docstring) -- ``GPmodel`` is unchanged from the caller's
    perspective.

    Parameters
    ----------
    objective : {"taylor", "lower_bound"}
        Which of ``ENTROPY_ESTIMATORS`` to maximize -- see the module docstring.
    """
    if objective not in ENTROPY_ESTIMATORS:
        raise ValueError(
            f"objective must be one of {sorted(ENTROPY_ESTIMATORS)}, got {objective!r}"
        )
    xStarGP = xStarGP.reshape(-1, 1).T
    np.random.seed(seed)
    subset_indices = np.random.choice(len(trace), size=no_gaussians, replace=False)
    sub_samples = trace[subset_indices]
    saved_hyperparameters = kernels.save_hyperparameters(GPmodel.kernel)
    try:
        means = []
        variances = []
        for sample in sub_samples:
            # Eq (7): one mixture component per posterior draw theta^(s) -- assign it,
            # then read off that draw's own GP predictive mean/variance (Eq 5a/5b).
            kernels.assign_hyperparameters(GPmodel.kernel, sample)
            mean, variance = GPmodel.predict_f(xStarGP, full_cov=True)
            means.append(mean.numpy().squeeze())
            variances.append(variance.numpy().squeeze())
    finally:
        kernels.assign_hyperparameters(GPmodel.kernel, saved_hyperparameters)
    means = np.array(means)
    variances = np.array(variances)
    weights = np.ones(no_gaussians) * 1 / no_gaussians  # Eq (7): equal weights 1/S
    H = ENTROPY_ESTIMATORS[objective](weights, means, variances)  # Eq (9) or Theorem/SI-2
    return -H


def entropy_surface_2D(
    trace: np.ndarray,
    GPmodel: gpflow.models.GPR,
    x_bounds: Sequence[Tuple[float, float]],
    mesh: Sequence[int],
    x_trsf_fwd: Sequence[Callable[[np.ndarray], np.ndarray]],
    x_trsf_bkwd: Sequence[Callable[[np.ndarray], np.ndarray]],
    seed: int,
    no_gaussians: int,
    objective: str = "taylor",
) -> np.ndarray:
    """Entropy field over a 2-D grid spanning ``x_bounds`` at ``mesh`` points per dim.

    2-D-ONLY VISUALIZATION DIAGNOSTIC: a dense grid is exponential in the input
    dimension d, so this is intentionally not generalized to N-D. It does not feed the
    acquisition -- see ``optimize`` for the N-D-general acquisition path.

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
    H = np.array(
        [-entropy_objective(x, trace, GPmodel, seed, no_gaussians, objective) for x in XStarGP]
    )
    XStar = np.column_stack([bkwd(XStarGP[:, j]) for j, bkwd in enumerate(x_trsf_bkwd)])
    return np.column_stack([XStar, H])


def optimize(
    trace: np.ndarray,
    GPmodel: gpflow.models.GPR,
    x_bounds: Sequence[Tuple[float, float]],
    x_trsf_fwd: Sequence[Callable[[np.ndarray], np.ndarray]],
    seed: int,
    no_gaussians: int,
    no_restarts: int,
    objective: str = "taylor",
) -> OptimizeResult:
    """Multistart optimization of the entropy objective over ``x_bounds`` (N-D).

    Works at arbitrary input dimension ``d = len(x_bounds)`` -- this is the acquisition
    path an N-D run actually depends on (contrast ``entropy_surface_2D``, a 2-D-only
    diagnostic). Sobol-scrambled restarts drawn in the physical space, optimized in GP
    (transformed) space.

    The vectorized bound-scaling below (``lo + x0 * (hi - lo)``) is written in this
    operand order deliberately: it matters for bit-exactness against
    ``tests/integration/data/synthetic_baseline.json``'s pinned values at ``d == 2``
    (IEEE 754 addition is commutative, so ``x0[j] * (hi[j] - lo[j]) + lo[j]`` would be
    equally correct mathematically, but rounds differently).

    Parameters
    ----------
    objective : {"taylor", "lower_bound"}
        Which entropy estimator ``entropy_objective`` maximizes (default "taylor",
        the paper's estimator; unchanged behavior unless a caller opts into
        "lower_bound") -- see ``entropy_objective``'s docstring.

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
            entropy_objective,
            x0=x0GP,
            bounds=XBndsGP,
            args=(trace, GPmodel, seed, no_gaussians, objective),
        )
        if result.fun < best_value:
            best_value = result.fun
            best_result = result
    return best_result
