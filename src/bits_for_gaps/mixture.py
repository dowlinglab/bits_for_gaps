"""Gaussian-mixture predictive posterior from hyperparameter-posterior draws.

Moved from the paper code's ``driver_new.py`` (``adaptiveEntropy.sample_gp_posterior_mixture``
/ ``gp_predict_2D``). Each mixture component corresponds to one HMC posterior draw of the
GP kernel's hyperparameters (in that kernel's canonical order -- see
``kernels.AnisotropicSE.hyperparameters``); sampling reassigns the given GP model's kernel
parameters, once per draw, to walk through the mixture components, matching the paper code.

Phase 5: kernel hyperparameters are no longer assigned by hardcoded attribute name --
``kernels.assign_hyperparameters`` maps trace columns to a kernel's ``.hyperparameters``
generically, so this works for any dimension/kernel exposing that contract.

Phase 9c: :func:`sample_gp_posterior_mixture` used to leave ``GPmodel.kernel`` at
whatever the *last* draw's hyperparameters happened to be -- harmless for the values
this module itself returns (computed before the mutation matters), but a footgun for
any caller that reuses the same ``GPmodel`` object afterward: Phase 9b traced a
from-scratch validation script's spurious McCabe-Thiele non-convergence to exactly
this (see ``paper/PHASE9B_INVESTIGATION.md``) -- a test-RMSE step left the kernel at an
arbitrary leftover state before a later step reused the same model for the phase
diagram. Now saves the kernel's hyperparameters before sampling and restores them in a
``finally``, so the caller's model is unchanged after the call regardless of how it
exits. This changes only the model's POST-call state, not any value this module
computes and returns -- behavior-preserving for every existing test/baseline.

NOTE: ``GPmodel.predict_f_samples`` draws from GPflow/TensorFlow's *ambient* default
random generator, not ``numpy``'s -- the ``np.random.seed`` call below seeds which
posterior-sample components are selected, but not the draws themselves. This means
these two functions were never bitwise-reproducible in the original paper code either
(confirmed: two successive calls to ``predict_f_samples`` on the same inputs, same
process, differ). That is why the Phase 2 integration test deliberately excludes this
plotting-only step from its determinism/baseline pins. Phase 9c adds an OPTIONAL
``tf_seed`` to make a single call reproducible on request (seeds TF's global RNG right
before drawing) -- default ``None`` leaves the ambient-RNG behavior unchanged.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import gpflow
import numpy as np
import tensorflow as tf

from . import kernels


def sample_gp_posterior_mixture(
    trace: np.ndarray,
    GPmodel: gpflow.models.GPR,
    XGP: np.ndarray,
    seed: int,
    size: int,
    tf_seed: Optional[int] = None,
) -> np.ndarray:
    """Sample one full-covariance GP draw per selected hyperparameter-posterior component.

    Parameters
    ----------
    trace : np.ndarray, shape (n_trace, n_hyperparameters)
        HMC posterior samples, in ``GPmodel.kernel``'s canonical hyperparameter order.
    GPmodel : gpflow.models.GPR
        Its kernel hyperparameters are reassigned once per draw during the call, then
        restored to their original values before returning (even if an error occurs
        mid-loop) -- ``GPmodel`` is unchanged from the caller's perspective.
    XGP : array-like
        Points (in GP input space) at which to draw predictive samples.
    seed : int
    size : int
        Number of posterior-sample components to select (without replacement) from
        ``trace``. The number of mixture draws itself is fixed at 100, matching the
        paper code -- it draws *with* replacement over the ``size`` selected components.
    tf_seed : int, optional
        If given, seeds TensorFlow's global RNG immediately before drawing, making
        this call's ``predict_f_samples`` draws reproducible (see the module
        docstring's note on ambient-RNG non-reproducibility). ``None`` (default)
        leaves TF's ambient RNG untouched -- the original, non-reproducible behavior.

    Returns
    -------
    np.ndarray
        Stacked posterior-predictive samples, squeezed.
    """
    np.random.seed(seed)
    subset_indices = np.random.choice(len(trace), size=size, replace=False)
    sub_samples = trace[subset_indices]
    comp_ids = np.random.randint(0, len(sub_samples), size=100)
    if tf_seed is not None:
        tf.random.set_seed(tf_seed)
    saved_hyperparameters = kernels.save_hyperparameters(GPmodel.kernel)
    try:
        samples = []
        for comp_id in comp_ids:
            # Eq (7): one GMM component per selected hyperparameter-posterior draw --
            # assign theta^(s), then draw from that draw's own GP predictive (Eq 5a/5b).
            kernels.assign_hyperparameters(GPmodel.kernel, sub_samples[comp_id])
            gp_out = GPmodel.predict_f_samples(XGP, full_cov=True)
            samples.append(gp_out)
        return np.array(samples).squeeze()
    finally:
        kernels.assign_hyperparameters(GPmodel.kernel, saved_hyperparameters)


def predict_grid_2D(
    trace: np.ndarray,
    GPmodel: gpflow.models.GPR,
    x_bounds: Sequence[Tuple[float, float]],
    x_trsf_fwd: Sequence[Callable[[np.ndarray], np.ndarray]],
    x_trsf_bkwd: Sequence[Callable[[np.ndarray], np.ndarray]],
    y_trsf_bkwd: Callable[[np.ndarray], np.ndarray],
    seed: int,
    size: int,
    n_grid: int = 50,
    tf_seed: Optional[int] = None,
) -> np.ndarray:
    """Full-grid GP posterior-predictive samples, for 2-D plotting diagnostics only.

    Moved from ``adaptiveEntropy.gp_predict_2D``. This does **not** feed the acquisition
    (entropy/next-point selection) -- it is an expensive (``size`` full-covariance draws
    over an ``n_grid x n_grid`` grid), disk-write-oriented diagnostic kept only for
    parity with the paper's plotting pipeline. Callers should treat it as opt-in.

    2-D-ONLY (Phase 5): a dense grid is exponential in the input dimension d, so this
    is intentionally not generalized to N-D -- see ``acquisition.optimize`` for the
    N-D-general acquisition path this diagnostic does not feed.

    Parameters
    ----------
    x_trsf_fwd, x_trsf_bkwd : sequence of callables
        Per-dimension forward/backward transforms (see ``transforms.InputTransform``).
    y_trsf_bkwd : callable
        Backward transform for the GP output (see ``transforms.OutputTransform``).
    n_grid : int
        Points per dimension (50, matching the paper code's un-parameterized
        ``np.linspace`` default).
    tf_seed : int, optional
        Passed through to :func:`sample_gp_posterior_mixture`; see there.

    Returns
    -------
    np.ndarray, shape (n_grid**2, 2 + n_draws)
        Columns ``[x1, x2, y_draw_0, y_draw_1, ...]`` in the physical (untransformed)
        input/output space.

    Raises
    ------
    ValueError
        If ``len(x_bounds) != 2``.
    """
    if len(x_bounds) != 2:
        raise ValueError(
            f"predict_grid_2D is a 2-D-only visualization diagnostic (got "
            f"{len(x_bounds)} input dimensions); it does not feed the acquisition -- "
            f"see acquisition.optimize for the N-D-general acquisition path."
        )
    x1, x2 = np.linspace(*x_bounds[0], n_grid), np.linspace(*x_bounds[1], n_grid)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    xStar = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
    XStarGP = np.column_stack([fwd(xStar[:, j]) for j, fwd in enumerate(x_trsf_fwd)])
    yStarGP = sample_gp_posterior_mixture(trace, GPmodel, XStarGP, seed, size, tf_seed=tf_seed)
    xStar = np.column_stack([bkwd(XStarGP[:, j]) for j, bkwd in enumerate(x_trsf_bkwd)])
    yStar = y_trsf_bkwd(yStarGP).T
    return np.column_stack([xStar, yStar])
