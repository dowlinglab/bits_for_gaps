"""Gaussian-mixture predictive posterior from hyperparameter-posterior draws.

Moved from the paper code's ``driver_new.py`` (``adaptiveEntropy.sample_gp_posterior_mixture``
/ ``gp_predict_2D``). Each mixture component corresponds to one HMC posterior draw of the
GP kernel hyperparameters (``std_dev``, ``lengthscale_1``, ``lengthscale_2``); sampling
mutates the given GP model's kernel parameters in place, matching the paper code.

TODO(Phase 5): kernel hyperparameters are assigned by hardcoded attribute name (assumes
exactly the 3-hyperparameter ``AnisotropicSE`` kernel) -- mirrors the TODO in kernels.py.

NOTE: ``GPmodel.predict_f_samples`` draws from GPflow/TensorFlow's *ambient* default
random generator, not ``numpy``'s -- the ``np.random.seed`` call below seeds which
posterior-sample components are selected, but not the draws themselves. This means
these two functions were never bitwise-reproducible in the original paper code either
(confirmed: two successive calls to ``predict_f_samples`` on the same inputs, same
process, differ). That is why the Phase 2 integration test deliberately excludes this
plotting-only step from its determinism/baseline pins.
"""

import numpy as np


def sample_gp_posterior_mixture(trace, GPmodel, XGP, seed, size):
    """Sample one full-covariance GP draw per selected hyperparameter-posterior component.

    Parameters
    ----------
    trace : np.ndarray, shape (n_trace, 3)
        HMC posterior samples (std_dev, lengthscale_1, lengthscale_2).
    GPmodel : gpflow.models.GPR
        Mutated in place: its kernel hyperparameters are reassigned for every draw.
    XGP : array-like
        Points (in GP input space) at which to draw predictive samples.
    seed : int
    size : int
        Number of posterior-sample components to select (without replacement) from
        ``trace``. The number of mixture draws itself is fixed at 100, matching the
        paper code -- it draws *with* replacement over the ``size`` selected components.

    Returns
    -------
    np.ndarray
        Stacked posterior-predictive samples, squeezed.
    """
    np.random.seed(seed)
    subset_indices = np.random.choice(len(trace), size=size, replace=False)
    sub_samples = trace[subset_indices]
    comp_ids = np.random.randint(0, len(sub_samples), size=100)
    samples = []
    for comp_id in comp_ids:
        GPmodel.kernel.std_dev.assign(sub_samples[comp_id, 0])
        GPmodel.kernel.lengthscale_1.assign(sub_samples[comp_id, 1])
        GPmodel.kernel.lengthscale_2.assign(sub_samples[comp_id, 2])
        gp_out = GPmodel.predict_f_samples(XGP, full_cov=True)
        samples.append(gp_out)
    return np.array(samples).squeeze()


def predict_grid_2D(trace, GPmodel, x_bounds, x_trsf_fwd, x_trsf_bkwd, y_trsf_bkwd,
                    seed, size, n_grid=50):
    """Full-grid GP posterior-predictive samples, for 2-D plotting diagnostics only.

    Moved from ``adaptiveEntropy.gp_predict_2D``. This does **not** feed the acquisition
    (entropy/next-point selection) -- it is an expensive (``size`` full-covariance draws
    over an ``n_grid x n_grid`` grid), disk-write-oriented diagnostic kept only for
    parity with the paper's plotting pipeline. Callers should treat it as opt-in.

    Parameters
    ----------
    x_trsf_fwd, x_trsf_bkwd : sequence of callables
        Per-dimension forward/backward transforms (see ``transforms.InputTransform``).
    y_trsf_bkwd : callable
        Backward transform for the GP output (see ``transforms.OutputTransform``).
    n_grid : int
        Points per dimension (50, matching the paper code's un-parameterized
        ``np.linspace`` default).

    Returns
    -------
    np.ndarray, shape (n_grid**2, 2 + n_draws)
        Columns ``[x1, x2, y_draw_0, y_draw_1, ...]`` in the physical (untransformed)
        input/output space.
    """
    x1, x2 = np.linspace(*x_bounds[0], n_grid), np.linspace(*x_bounds[1], n_grid)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    xStar = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
    XStarGP = np.column_stack([fwd(xStar[:, j]) for j, fwd in enumerate(x_trsf_fwd)])
    yStarGP = sample_gp_posterior_mixture(trace, GPmodel, XStarGP, seed, size)
    xStar = np.column_stack([bkwd(XStarGP[:, j]) for j, bkwd in enumerate(x_trsf_bkwd)])
    yStar = y_trsf_bkwd(yStarGP).T
    return np.column_stack([xStar, yStar])
