"""Unit tests for ``vle_distillation.phase_diagram`` (pure NumPy/SciPy -- no Julia).

Uses a synthetic ideal-mixture activity function (``gamma_proh = gamma_water = 1``)
rather than Clapeyron, so these run in the default (no ``vle`` marker) suite and check
the bubble-point/dew-point/Antoine physics independent of the thermodynamic model.
"""
import numpy as np
import pytest

from vle_distillation import phase_diagram as pd

IDEAL_GAMMA = lambda z, T: (1.0, 1.0)


def test_pvap_antoine_proh_near_one_atm_at_normal_boiling_point():
    # 1-propanol's real normal boiling point is 370.35 K; Antoine's equation should
    # give ~1 atm (1.01325 bar) there.
    p = pd.pvap_antoine(370.35, pd.ANTOINE_CONSTANTS["PrOH"])
    assert p == pytest.approx(pd.P_TOT_BAR, rel=2e-3)


def test_pvap_antoine_water_near_one_atm_at_normal_boiling_point():
    # Water's real normal boiling point is 373.15 K; this Antoine correlation's own
    # fit is only accurate to ~1-2% here (it's fit for the water-PrOH VLE range, not
    # optimized at water's pure boiling point), so the tolerance is correspondingly
    # looser than the PrOH check above.
    p = pd.pvap_antoine(373.15, pd.ANTOINE_CONSTANTS["H2O"])
    assert p == pytest.approx(pd.P_TOT_BAR, rel=2e-2)


def test_ideal_bubble_point_at_pure_proh_matches_antoine_boiling_point():
    T_bub = pd.bubble_point_temperature(1.0, IDEAL_GAMMA)
    # At z=1 (ideal), the bubble point is exactly where Pvap_PrOH(T) == P_tot.
    assert pd.pvap_antoine(T_bub, pd.ANTOINE_CONSTANTS["PrOH"]) == pytest.approx(
        pd.P_TOT_BAR, rel=1e-6)


def test_ideal_bubble_point_at_pure_water_matches_antoine_boiling_point():
    T_bub = pd.bubble_point_temperature(0.0, IDEAL_GAMMA)
    assert pd.pvap_antoine(T_bub, pd.ANTOINE_CONSTANTS["H2O"]) == pytest.approx(
        pd.P_TOT_BAR, rel=1e-6)


def test_dew_point_pure_component_edge_cases():
    T1 = pd.bubble_point_temperature(1.0, IDEAL_GAMMA)
    T0 = pd.bubble_point_temperature(0.0, IDEAL_GAMMA)
    assert pd.dew_point_vapor_fraction(1.0, T1, IDEAL_GAMMA) == pytest.approx(1.0, abs=1e-6)
    assert pd.dew_point_vapor_fraction(0.0, T0, IDEAL_GAMMA) == pytest.approx(0.0, abs=1e-9)


def test_eqm_residual_zero_at_the_solved_bubble_point():
    z = 0.4
    T_bub = pd.bubble_point_temperature(z, IDEAL_GAMMA)
    assert pd.eqm_residual(T_bub, z, IDEAL_GAMMA) == pytest.approx(0.0, abs=1e-6)


def test_vle_curve_shapes_and_physical_bounds():
    z_grid = np.linspace(0.0, 1.0, 11)
    z, T_bub, y = pd.vle_curve(IDEAL_GAMMA, z_grid=z_grid)
    assert z.shape == T_bub.shape == y.shape == (11,)
    assert np.all(T_bub > 0)
    assert np.all((y >= -1e-9) & (y <= 1.0 + 1e-9))


def test_vle_curve_default_grid_size():
    z, T_bub, y = pd.vle_curve(IDEAL_GAMMA)
    assert z.shape == (pd.Z_MESH,)
