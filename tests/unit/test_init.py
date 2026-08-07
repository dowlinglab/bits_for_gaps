"""Unit tests for ``bits_for_gaps``'s top-level PEP 562 lazy-import machinery.

Every other test in this suite reaches the TF-backed classes via
``from bits_for_gaps.kernels import AnisotropicSE``-style direct submodule imports,
which never exercises ``__getattr__``'s lazy-resolution branch or ``__dir__``. Both are
part of the module's actual public contract (documented in its module docstring), so
they're tested directly here rather than excluded from coverage.
"""

import pytest

import bits_for_gaps
from bits_for_gaps.kernels import AnisotropicSE
from bits_for_gaps.means import FixedInverseMean
from bits_for_gaps.sampler import BitsForGaps, adaptiveEntropy


def test_lazy_attribute_resolves_to_the_real_class():
    # bits_for_gaps.AnisotropicSE (no direct submodule import) must be the exact same
    # object __getattr__ resolves it to, not a copy or a different definition.
    assert bits_for_gaps.AnisotropicSE is AnisotropicSE
    assert bits_for_gaps.FixedInverseMean is FixedInverseMean
    assert bits_for_gaps.adaptiveEntropy is adaptiveEntropy
    assert bits_for_gaps.BitsForGaps is BitsForGaps


def test_lazy_attribute_importable_via_from_import():
    # `from bits_for_gaps import BitsForGaps` is the documented top-level entry point
    # (see docs/quickstart.md) -- it must go through the same __getattr__ path.
    from bits_for_gaps import BitsForGaps as ReImported

    assert ReImported is BitsForGaps


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'NotARealName'"):
        _ = bits_for_gaps.NotARealName


def test_dir_includes_lazy_names_and_eager_names():
    names = dir(bits_for_gaps)
    # Lazy (TF-backed) names, resolved only via __getattr__.
    for lazy_name in ("AnisotropicSE", "FixedInverseMean", "adaptiveEntropy", "BitsForGaps"):
        assert lazy_name in names
    # Eager (pure NumPy/SciPy) names, already in the module's normal namespace.
    for eager_name in ("second_order_entropy", "latin_hypercube_design", "design", "entropy"):
        assert eager_name in names
    # dir() must not report duplicates.
    assert len(names) == len(set(names))
