"""Wrap a VLE curve as an ``x_liquid -> y_vapor`` equilibrium function.

Ported from the paper code's ``equilibrium.py`` (``water_proh_eqm`` /
``water_proh_eqm_julia``), generalized to take the curve arrays directly (e.g. from
``phase_diagram.vle_curve``) instead of reading fixed archived filenames -- the
distillation solver (``distillation.py``) needs a plain ``x -> y`` callable.
"""

import numpy as np
from scipy.interpolate import interp1d


def make_equilibrium_function(x_liquid, y_vapor):
    """Cubic-interpolated ``x_liquid -> y_vapor`` equilibrium function.

    Parameters
    ----------
    x_liquid, y_vapor : array-like
        Liquid/vapor PrOH mole fractions, e.g. from ``phase_diagram.vle_curve(...)``.
    """
    x_liquid = np.asarray(x_liquid, dtype=float)
    y_vapor = np.asarray(y_vapor, dtype=float)
    order = np.argsort(x_liquid)
    return interp1d(x_liquid[order], y_vapor[order], kind="cubic", fill_value="extrapolate")
