API reference
=============

Public API
----------

The names re-exported at the top level (``from bits_for_gaps import ...``). The
TensorFlow/GPflow-backed names (:class:`~bits_for_gaps.kernels.AnisotropicSE`,
:class:`~bits_for_gaps.means.FixedInverseMean`,
:class:`~bits_for_gaps.sampler.adaptiveEntropy`,
:class:`~bits_for_gaps.sampler.BitsForGaps`) are imported lazily (PEP 562
``__getattr__``) so that pure entropy/design usage doesn't pay the TensorFlow import
cost -- see :mod:`bits_for_gaps`'s module docstring.

.. automodule:: bits_for_gaps
   :members:
   :undoc-members:
   :exclude-members: InputTransform,OutputTransform

..
   InputTransform/OutputTransform are excluded above (fully documented in "Design and
   transforms" below at their canonical bits_for_gaps.transforms location) -- without
   this, Sphinx indexes the same class under two names (the top-level re-export and
   the original module path), and any type hint that references them by their bare
   name (e.g. BitsForGaps.__init__'s input_transform/output_transform parameters)
   becomes an ambiguous cross-reference ("more than one target found"), which
   sphinx-build -W then fails on.

Sequential-design engine
-------------------------

.. automodule:: bits_for_gaps.sampler
   :members:
   :show-inheritance:

.. automodule:: bits_for_gaps.state
   :members:

GP model
--------

.. automodule:: bits_for_gaps.kernels
   :members:
   :show-inheritance:

.. automodule:: bits_for_gaps.means
   :members:
   :show-inheritance:

.. automodule:: bits_for_gaps.gp
   :members:

.. automodule:: bits_for_gaps.mixture
   :members:

.. automodule:: bits_for_gaps.diagnostics
   :members:

Entropy and acquisition
------------------------

.. automodule:: bits_for_gaps.entropy
   :members:

.. automodule:: bits_for_gaps.acquisition
   :members:

Design and transforms
----------------------

.. automodule:: bits_for_gaps.design
   :members:

.. automodule:: bits_for_gaps.transforms
   :members:
