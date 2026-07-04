"""BITS for GAPS -- Bayesian Information-Theoretic Sampling for hierarchical
GAussian Process Surrogates.

Public API. The pure NumPy/SciPy pieces (entropy, design) import eagerly; the
TensorFlow/GPflow-backed pieces (kernels, means, sampler) import lazily via PEP 562
``__getattr__`` so that entropy-only usage -- and lightweight CI -- need not pay the
TensorFlow import cost.
"""

from importlib import import_module

__version__ = "0.0.1.dev0"

# Pure (NumPy/SciPy) -- safe to import eagerly.
from . import design, entropy, transforms
from .design import full_factorial_design, latin_hypercube_design
from .entropy import (
    entropy_lower_bound,
    first_order_entropy_approx,
    gaussian_mixture_density,
    second_order_entropy,
)
from .transforms import InputTransform, OutputTransform

# TensorFlow/GPflow-backed -- imported on first access.
_LAZY = {
    "AnisotropicSE": ("kernels", "AnisotropicSE"),
    "FixedInverseMean": ("means", "FixedInverseMean"),
    "adaptiveEntropy": ("sampler", "adaptiveEntropy"),
    "BitsForGaps": ("sampler", "BitsForGaps"),  # target public API (Phase 4)
}


def __getattr__(name):
    if name in _LAZY:
        mod, attr = _LAZY[name]
        return getattr(import_module(f".{mod}", __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = [
    "second_order_entropy",
    "first_order_entropy_approx",
    "entropy_lower_bound",
    "gaussian_mixture_density",
    "latin_hypercube_design",
    "full_factorial_design",
    "AnisotropicSE",
    "FixedInverseMean",
    "adaptiveEntropy",
    "BitsForGaps",
    "InputTransform",
    "OutputTransform",
    "entropy",
    "design",
    "transforms",
]
