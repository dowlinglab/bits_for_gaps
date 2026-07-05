"""GP mean functions.

Moved from the paper code's ``fxns/my_mean_fxn.py``. The paper's VLE study uses a
zero mean over the log-activity-coefficient output (encoding ideal mixing, gamma -> 1,
in the absence of data). ``FixedInverseMean`` is an alternative physics-informed mean
retained for reference.

TODO(Phase 5): ``FixedInverseMean`` assumes the mole fraction is input column 0;
generalize the input-column convention when the kernel goes N-D.
"""

from __future__ import annotations

import gpflow
import tensorflow as tf


class FixedInverseMean(gpflow.mean_functions.MeanFunction):
    """A fixed (non-trainable) Box-Cox-transformed inverse mean in the first input.

    m(X) = ((max(base, eps))^lambda - 1) / lambda,  base = (2 - x) / (x + epsilon),
    with x = X[:, 0] the mole fraction.
    """

    def __init__(self, epsilon: float = 0.6, lambda_bc: float = 0.1, eps: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.lambda_bc = lambda_bc
        self.eps = eps

    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        x = X[:, 0:1]
        base = (2.0 - x) / (x + self.epsilon)
        safe_base = tf.maximum(base, self.eps)
        transformed = (tf.pow(safe_base, self.lambda_bc) - 1.0) / self.lambda_bc
        return transformed
