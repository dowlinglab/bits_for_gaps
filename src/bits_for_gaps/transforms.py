"""Per-dimension input/output transforms for the GP surrogate.

Lifts the paper code's list-of-lambdas convention (``driver_new.py``'s
``adaptiveEntropy.XTrsfFwd``/``XTrsfBkwd``/``yTrsfFwd``/``yTrsfBkwd``, one identity
lambda per input dimension plus a scalar output lambda) into small, testable classes.
Identity by default -- matches the paper's VLE study, which trains the GP directly on
mole fraction / temperature / log-activity-coefficient with no rescaling.

``forward_fns``/``backward_fns`` (lists of per-dimension callables) are exposed
alongside the array-oriented ``forward``/``backward`` methods because the acquisition/
mixture code applies them element-wise to scalars (e.g. a single bound) as well as to
whole columns -- exactly the calling convention the paper code used.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

ElementwiseFn = Callable[[np.ndarray], np.ndarray]


def _identity(x: np.ndarray) -> np.ndarray:
    return x


class InputTransform:
    """Per-dimension forward/backward transforms for the GP input space.

    Parameters
    ----------
    forward_fns, backward_fns : sequence of callables, optional
        One elementwise callable per input dimension. Must be given together (or not
        at all, for the identity default).
    ndim : int, optional
        Number of input dimensions; required (and used to build the identity
        transforms) when ``forward_fns``/``backward_fns`` are omitted.
    """

    def __init__(
        self,
        forward_fns: Optional[Sequence[ElementwiseFn]] = None,
        backward_fns: Optional[Sequence[ElementwiseFn]] = None,
        ndim: Optional[int] = None,
    ) -> None:
        if forward_fns is None and backward_fns is None:
            if ndim is None:
                raise ValueError("ndim is required when forward_fns/backward_fns are omitted")
            forward_fns = [_identity] * ndim
            backward_fns = [_identity] * ndim
        elif forward_fns is None or backward_fns is None:
            raise ValueError("forward_fns and backward_fns must be given together")
        elif len(forward_fns) != len(backward_fns):
            raise ValueError("forward_fns and backward_fns must have the same length")
        self.forward_fns = list(forward_fns)
        self.backward_fns = list(backward_fns)

    @property
    def ndim(self) -> int:
        return len(self.forward_fns)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply the per-dimension forward transforms; ``X``: (n, d) -> (n, d)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.column_stack([f(X[:, j]) for j, f in enumerate(self.forward_fns)])

    def backward(self, X: np.ndarray) -> np.ndarray:
        """Apply the per-dimension backward transforms; ``X``: (n, d) -> (n, d)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.column_stack([f(X[:, j]) for j, f in enumerate(self.backward_fns)])


class OutputTransform:
    """Forward/backward transform for the (scalar) GP output column."""

    def __init__(
        self,
        forward_fn: Optional[ElementwiseFn] = None,
        backward_fn: Optional[ElementwiseFn] = None,
    ) -> None:
        self.forward_fn = forward_fn or _identity
        self.backward_fn = backward_fn or _identity

    def forward(self, y: np.ndarray) -> np.ndarray:
        return self.forward_fn(y)

    def backward(self, y: np.ndarray) -> np.ndarray:
        return self.backward_fn(y)
