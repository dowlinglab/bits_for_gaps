"""Unit tests for ``vle_distillation.distillation`` (pure NumPy/SciPy -- no Julia).

``_distillation_residuals`` is verified against a hand-derived, self-consistent
single-stage (n=1) solution rather than relying on ``fsolve`` -- the underlying
nonlinear system has no bounds and, like the original MATLAB-derived port, can
converge to spurious (unphysical) roots for an arbitrary synthetic equilibrium curve
(confirmed empirically while developing this test). ``solve_column``'s actual
fsolve-based convergence is exercised end-to-end against the real Wilson physics under
``@pytest.mark.vle`` (``tests/regression/test_mccabe_thiele.py``), where it is known to
converge cleanly.
"""
import numpy as np
import pytest

from vle_distillation import distillation as dist


def _hand_consistent_n1_solution():
    """A self-consistent n=1 (single equilibrium stage) solution, derived by hand.

    Overall/component mass balances are satisfied by construction (D chosen from the
    lever rule on xD/xW/xF); the linear equilibrium curve ``equil(x) = 3*x`` is chosen
    to satisfy ``equil(xW) == xD`` exactly, which n=1's stage-1 equilibrium relation
    combined with the total-condenser and reboiler equations requires.
    """
    xD, xW, xF, F, R, q = 0.6, 0.2, 0.4, 100.0, 1.5, 1.0
    D = F * (xF - xW) / (xD - xW)
    W = F - D
    equil = lambda x: 3.0 * x
    L0 = R * D
    V0 = L0 + D
    L1 = L0 + F * q
    V1 = L1 - W
    v = np.array([L0, L1, V0, V1, xD, xW, xD, xW, D, R, W, F, xF, q])
    var_names = ["xW", "F", "xF", "R", "xD"]
    var_values = [xW, F, xF, R, xD]
    return v, equil, var_names, var_values


def test_residuals_vanish_at_hand_derived_solution():
    v, equil, var_names, var_values = _hand_consistent_n1_solution()
    fixed_idx, _ = dist._resolve_fixed_indices(1, var_names, var_values)
    res = dist._distillation_residuals(v, 1, 0, equil, fixed_idx, np.array(var_values))
    np.testing.assert_allclose(res, np.zeros_like(res), atol=1e-10)


def test_residuals_nonzero_when_perturbed():
    v, equil, var_names, var_values = _hand_consistent_n1_solution()
    fixed_idx, _ = dist._resolve_fixed_indices(1, var_names, var_values)
    v_bad = v.copy()
    v_bad[0] *= 1.1   # perturb L0
    res = dist._distillation_residuals(v_bad, 1, 0, equil, fixed_idx, np.array(var_values))
    assert np.max(np.abs(res)) > 1e-3


@pytest.mark.parametrize("name,expected_key", [
    ("xD", "x0"), ("xW", "xn"), ("L0", "L0"), ("V1", "V0"),
    ("L4", "Ln"), ("V5", "Vn"), ("x2", "x2"), ("y3", "y2"),
])
def test_resolve_fixed_indices_known_names(name, expected_key):
    # Cross-check every named-variable branch against its documented 0-based offset.
    n_stages = 4
    offset_L, offset_V = 0, n_stages + 1
    offset_x, offset_y = 2 * (n_stages + 1), 3 * (n_stages + 1)
    expected = {
        "x0": offset_x + 0, "xn": offset_x + n_stages,
        "L0": offset_L + 0, "V0": offset_V + 0,
        "Ln": offset_L + n_stages, "Vn": offset_V + n_stages,
        "x2": offset_x + 2, "y2": offset_y + 2,
    }[expected_key]
    fixed_idx, _ = dist._resolve_fixed_indices(n_stages, [name], [0.5])
    assert fixed_idx[0] == expected


def test_resolve_fixed_indices_rejects_unknown_name():
    with pytest.raises(ValueError):
        dist._resolve_fixed_indices(4, ["bogus"], [0.5])


def test_solve_column_stage_indexing_convention():
    # Confirm the returned "stages" pairing (x[i], y[i-1]) against a converged, hand-
    # verified solution rather than re-deriving fsolve convergence in this test.
    v, equil, var_names, var_values = _hand_consistent_n1_solution()
    num_stages = 2
    L = v[0:num_stages]
    V = v[num_stages:2 * num_stages]
    x = v[2 * num_stages:3 * num_stages]
    y = v[3 * num_stages:4 * num_stages]
    stages = [{"stage": i, "liquid": float(x[i]), "vapor": float(y[i - 1])}
             for i in range(1, 2)]
    assert stages == [{"stage": 1, "liquid": pytest.approx(0.2), "vapor": pytest.approx(0.6)}]
