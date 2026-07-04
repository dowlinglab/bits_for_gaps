"""GP covariance kernels with hierarchical (prior-bearing) hyperparameters.

Moved from the paper code's ``fxns/my_kermel_fxn.py`` (commented-out variants and the
misspelled module name dropped). The kernel's hyperparameters carry the tfp priors that
make the GP *hierarchical* -- HMC samples these to form the mixture predictive posterior.

Phase 5: generalized to N input dimensions. DESIGN DECISION -- each lengthscale (and the
kernel variance) is its own scalar ``gpflow.Parameter``, not a single vector-valued
Parameter. This is deliberate: the paper's method hinges on per-dimension PRIOR FAMILIES,
not just per-dimension prior parameters -- ``std_dev`` ~ LogNormal, ``lengthscale_1`` ~
LogNormal, ``lengthscale_2`` ~ Gamma *and unconstrained* (no positivity bijector). A
single vector Parameter carries exactly one prior distribution and one transform for the
whole vector, so it cannot express "lengthscale 1 is LogNormal-positive, lengthscale 2 is
Gamma-unconstrained" without slicing hacks. Per-dimension scalar Parameters also match
gpflow's HMC machinery directly: ``SamplingHelper`` takes a flat list of trainable
Parameters, each contributing its own prior term to the log-posterior and its own
bijector to the unconstrained state -- exactly what a list of scalar Parameters already
is, with no adapter needed.

CANONICAL HYPERPARAMETER ORDER (the contract ``gp.run_mcmc`` / ``mixture.py`` /
``acquisition.py`` rely on for generic introspection, HMC state order, and trace-column
assignment):

    [std_dev, lengthscale_1, lengthscale_2, ..., lengthscale_d]

exposed via the ``hyperparameters`` property (a list of ``gpflow.Parameter``), and by
name as ``.std_dev``, ``.lengthscale_1``, ... ``.lengthscale_d`` (kept for backward
compatibility with 2-D-era code addressing them by attribute name).
"""

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp

f64 = gpflow.utilities.to_default_float


class AnisotropicSE(gpflow.kernels.Kernel):
    """Anisotropic squared-exponential kernel with per-dimension lengthscales.

    Generalized (Phase 5) to N input dimensions. Each lengthscale -- and the kernel
    variance -- is its own ``gpflow.Parameter``, carrying its own prior distribution and
    (optionally) its own bijector, so per-dimension prior FAMILIES are supported (see
    the module docstring for why this matters).

    With no arguments, reproduces the paper's exact 2-D VLE configuration (mole
    fraction, temperature) -- equivalent to the explicit :meth:`paper_2d` factory.

    Parameters
    ----------
    variance_prior : tfp.distributions.Distribution, optional
        Prior on the kernel standard deviation (``std_dev``). Required together with
        ``lengthscale_priors`` for a custom (non-paper-default) configuration.
    lengthscale_priors : sequence of tfp.distributions.Distribution, optional
        One prior per input dimension; ``len(lengthscale_priors)`` sets ``ndim``.
    variance_init : float
        Initial value for ``std_dev``.
    lengthscale_inits : sequence of float, optional
        Initial value per dimension; defaults to ``1.0`` for every dimension.
    lengthscale_transforms : sequence of (tfp.bijectors.Bijector or None), optional
        Per-dimension bijector; ``None`` means gpflow's default (``Identity`` --
        *not* constrained to be positive). Defaults to ``positive()`` for every
        dimension. The paper's kernel deliberately leaves ``lengthscale_2`` (the
        temperature lengthscale) unconstrained (Gamma prior only, no bijector).
    """

    def __init__(self, variance_prior=None, lengthscale_priors=None, variance_init=1.25,
                 lengthscale_inits=None, lengthscale_transforms=None):
        super().__init__()

        if lengthscale_priors is None:
            if variance_prior is not None or lengthscale_inits is not None \
                    or lengthscale_transforms is not None:
                raise ValueError(
                    "lengthscale_priors is required whenever variance_prior / "
                    "lengthscale_inits / lengthscale_transforms are given explicitly."
                )
            # Paper's exact 2-D VLE configuration (see paper_2d).
            variance_prior = tfp.distributions.LogNormal(loc=tf.math.log(f64(1.0)),
                                                          scale=f64(2.0))
            lengthscale_priors = [
                tfp.distributions.LogNormal(loc=tf.math.log(f64(0.3)), scale=f64(0.5)),
                tfp.distributions.Gamma(concentration=f64(4.0), rate=f64(2.0)),
            ]
            lengthscale_inits = [2.0, 0.5]
            lengthscale_transforms = [gpflow.utilities.positive(), None]

        ndim = len(lengthscale_priors)
        if lengthscale_inits is None:
            lengthscale_inits = [1.0] * ndim
        if lengthscale_transforms is None:
            lengthscale_transforms = [gpflow.utilities.positive()] * ndim
        if not (len(lengthscale_inits) == len(lengthscale_transforms) == ndim):
            raise ValueError(
                "lengthscale_priors, lengthscale_inits, and lengthscale_transforms must "
                "all have the same length (one entry per input dimension)."
            )

        self.std_dev = gpflow.Parameter(
            name="std_dev",
            value=f64(variance_init),
            transform=gpflow.utilities.positive(),
            prior=variance_prior,
        )
        self._lengthscale_params = []
        for i in range(ndim):
            param = gpflow.Parameter(
                name=f"l{i + 1}",
                value=f64(lengthscale_inits[i]),
                transform=lengthscale_transforms[i],
                prior=lengthscale_priors[i],
            )
            setattr(self, f"lengthscale_{i + 1}", param)
            self._lengthscale_params.append(param)

    @classmethod
    def paper_2d(cls):
        """The paper's exact 2-D VLE kernel configuration (mole fraction, temperature).

        Equivalent to ``AnisotropicSE()``; an explicit, self-documenting factory for the
        2-D-faithful baseline this package is built to reproduce.
        """
        return cls()

    @property
    def ndim(self):
        return len(self._lengthscale_params)

    @property
    def hyperparameters(self):
        """All hierarchical hyperparameters, in the canonical order -- see the module
        docstring. This is the contract ``gp.run_mcmc`` (HMC parameter / trace-column
        order) and ``mixture.py``/``acquisition.py`` (assigning trace rows back onto
        this kernel, via ``assign_hyperparameters``) rely on.
        """
        return [self.std_dev] + list(self._lengthscale_params)

    @property
    def lengthscales(self):
        return tf.stack(self._lengthscale_params)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        X_scaled = X / self.lengthscales
        X2_scaled = X2 / self.lengthscales
        dist_sq = tf.reduce_sum((X_scaled[:, None, :] - X2_scaled[None, :, :]) ** 2, axis=-1)
        return self.std_dev ** 2 * tf.exp(-0.5 * dist_sq)

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.std_dev ** 2))


def assign_hyperparameters(kernel, values):
    """Assign ``values`` (in the canonical order -- see ``AnisotropicSE.hyperparameters``)
    onto ``kernel``'s hyperparameters, in place.

    The shared assignment contract ``mixture.py``/``acquisition.py`` use to replay one
    HMC trace row back onto the kernel: works for any kernel exposing a
    ``.hyperparameters`` property (a list of ``gpflow.Parameter``), not just
    ``AnisotropicSE``.
    """
    for param, value in zip(kernel.hyperparameters, values):
        param.assign(value)


def save_hyperparameters(kernel):
    """Snapshot ``kernel.hyperparameters``' current (constrained) values.

    Phase 9c: pairs with :func:`assign_hyperparameters` to save/restore a kernel's
    state around code that reassigns it in a loop (``mixture.sample_gp_posterior_mixture``,
    ``acquisition.entropy_objective``) -- see their docstrings. Returns a plain list of
    floats, not live references, so later mutating the kernel cannot change the snapshot.
    """
    return [float(param.numpy()) for param in kernel.hyperparameters]
