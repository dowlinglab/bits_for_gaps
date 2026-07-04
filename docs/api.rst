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
