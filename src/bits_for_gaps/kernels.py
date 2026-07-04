"""GP covariance kernels with hierarchical (prior-bearing) hyperparameters.

Moved from the paper code's ``fxns/my_kermel_fxn.py`` (commented-out variants and the
misspelled module name dropped). The kernel's hyperparameters carry the tfp priors that
make the GP *hierarchical* -- HMC samples these to form the mixture predictive posterior.

TODO(Phase 5): this kernel hardcodes exactly two input dimensions (``lengthscale_1``,
``lengthscale_2``). Generalize to an N-dimensional anisotropic kernel whose per-dimension
lengthscales are addressable generically (so the sampler/mixture code need not assign
hyperparameters by hardcoded attribute name).
"""

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp

f64 = gpflow.utilities.to_default_float


class AnisotropicSE(gpflow.kernels.Kernel):
    """Anisotropic squared-exponential kernel with per-dimension lengthscales.

    Two input dimensions (mole fraction, temperature in the paper's VLE study).
    Hyperparameters and their priors match the manuscript:

    - ``std_dev``       ~ LogNormal(log 1.0, 2.0)   (kernel standard deviation)
    - ``lengthscale_1`` ~ LogNormal(log 0.3, 0.5)   (mole-fraction lengthscale)
    - ``lengthscale_2`` ~ Gamma(4.0, 2.0)           (temperature lengthscale)
    """

    def __init__(self):
        super().__init__()

        self.std_dev = gpflow.Parameter(
            name="std_dev",
            value=f64(1.25),
            transform=gpflow.utilities.positive(),
            prior=tfp.distributions.LogNormal(loc=tf.math.log(f64(1.0)), scale=f64(2.0)),
        )
        self.lengthscale_1 = gpflow.Parameter(
            name="l1",
            value=f64(2.0),
            transform=gpflow.utilities.positive(),
            prior=tfp.distributions.LogNormal(loc=tf.math.log(f64(0.3)), scale=f64(0.5)),
        )
        self.lengthscale_2 = gpflow.Parameter(
            name="l2",
            value=f64(0.5),
            prior=tfp.distributions.Gamma(concentration=f64(4.0), rate=f64(2.0)),
        )

    @property
    def lengthscales(self):
        return tf.stack([self.lengthscale_1, self.lengthscale_2])

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        X_scaled = X / self.lengthscales
        X2_scaled = X2 / self.lengthscales
        dist_sq = tf.reduce_sum((X_scaled[:, None, :] - X2_scaled[None, :, :]) ** 2, axis=-1)
        return self.std_dev ** 2 * tf.exp(-0.5 * dist_sq)

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.std_dev ** 2))
