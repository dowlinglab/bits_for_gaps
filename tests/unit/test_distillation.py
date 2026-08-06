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

    def equil(x):
        return 3.0 * x

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
    v_bad[0] *= 1.1  # perturb L0
    res = dist._distillation_residuals(v_bad, 1, 0, equil, fixed_idx, np.array(var_values))
    assert np.max(np.abs(res)) > 1e-3


@pytest.mark.parametrize(
    "name,expected_key",
    [
        ("xD", "x0"),
        ("xW", "xn"),
        ("L0", "L0"),
        ("V1", "V0"),
        ("L4", "Ln"),
        ("V5", "Vn"),
        ("x2", "x2"),
        ("y3", "y2"),
    ],
)
def test_resolve_fixed_indices_known_names(name, expected_key):
    # Cross-check every named-variable branch against its documented 0-based offset.
    n_stages = 4
    offset_L, offset_V = 0, n_stages + 1
    offset_x, offset_y = 2 * (n_stages + 1), 3 * (n_stages + 1)
    expected = {
        "x0": offset_x + 0,
        "xn": offset_x + n_stages,
        "L0": offset_L + 0,
        "V0": offset_V + 0,
        "Ln": offset_L + n_stages,
        "Vn": offset_V + n_stages,
        "x2": offset_x + 2,
        "y2": offset_y + 2,
    }[expected_key]
    fixed_idx, _ = dist._resolve_fixed_indices(n_stages, [name], [0.5])
    assert fixed_idx[0] == expected


def test_resolve_fixed_indices_rejects_unknown_name():
    with pytest.raises(ValueError):
        dist._resolve_fixed_indices(4, ["bogus"], [0.5])


## ---------------------------------------------------------------------------
## solve_column's retry-on-non-convergence orchestration.
##
## Stubs _try_solve_column rather than hunting for a real equilibrium curve that
## reproducibly fails to converge -- fsolve's behavior on synthetic curves is
## empirically finicky (see this file's module docstring), so a test relying on that
## would be fragile. This isolates the ORCHESTRATION logic (which initial guesses are
## tried, in what order, and what's returned) from fsolve's actual convergence, which
## is exercised for real under @pytest.mark.vle (test_mccabe_thiele.py) instead.
## ---------------------------------------------------------------------------


def _stub_result(converged, x_level=None):
    return {
        "converged": converged,
        "warnings": [] if converged else ["fsolve did not report convergence"],
        "stages": [{"stage": 1, "liquid": x_level, "vapor": x_level}],
        "L": np.array([1.0]),
        "V": np.array([1.0]),
        "x": np.array([0.0, 0.0]),
        "y": np.array([0.0, 0.0]),
        "D": 1.0,
        "R": 1.0,
        "W": 1.0,
        "F": 1.0,
        "xF": 0.5,
        "q": 1.0,
    }


def test_solve_column_returns_first_attempt_unmodified_when_it_converges(monkeypatch):
    calls = []

    def fake_try_solve(v0, n, feed_stage_idx, equil, fixed_idx, fixed_vals, num_stages):
        calls.append(float(v0[2 * num_stages + 1]))  # x[1]: interior, not fixed
        return _stub_result(converged=True)

    monkeypatch.setattr(dist, "_try_solve_column", fake_try_solve)
    # 4 stages (matches the paper's actual column spec): x[0]/x[4] are fixed
    # (xD/xW), x[1..3] stay at the initial-guess level, so x[1] is a valid probe.
    result = dist.solve_column(
        4, 3, lambda x: x, ["xW", "F", "xF", "R", "xD"], [0.01, 100.0, 0.10, 1.0, 0.43]
    )
    assert result["converged"] is True
    assert calls == [0.5]  # only the default guess was tried -- no retries needed
    assert result["warnings"] == []  # untouched: no retry note added


def test_solve_column_retries_alternate_guesses_on_non_convergence(monkeypatch):
    calls = []

    def fake_try_solve(v0, n, feed_stage_idx, equil, fixed_idx, fixed_vals, num_stages):
        x_level = float(v0[2 * num_stages + 1])  # x[1]: interior, not fixed
        calls.append(x_level)
        return _stub_result(converged=(x_level == pytest.approx(0.7)))

    monkeypatch.setattr(dist, "_try_solve_column", fake_try_solve)
    result = dist.solve_column(
        4, 3, lambda x: x, ["xW", "F", "xF", "R", "xD"], [0.01, 100.0, 0.10, 1.0, 0.43]
    )
    assert result["converged"] is True
    # Default (0.5) fails, then retries in _RETRY_INITIAL_GUESS_LEVELS order (0.3
    # fails, 0.7 succeeds) -- stops there, never tries the third (0.2) guess.
    assert calls == [0.5, 0.3, 0.7]
    assert any("alternate initial guess" in w for w in result["warnings"])


def test_solve_column_reports_when_all_retries_fail(monkeypatch):
    def always_fails(v0, n, feed_stage_idx, equil, fixed_idx, fixed_vals, num_stages):
        return _stub_result(converged=False)

    monkeypatch.setattr(dist, "_try_solve_column", always_fails)
    result = dist.solve_column(
        1, 1, lambda x: x, ["xW", "F", "xF", "R", "xD"], [0.2, 100.0, 0.4, 1.5, 0.6]
    )
    assert result["converged"] is False
    assert any("none converged" in w for w in result["warnings"])


def test_solve_column_stage_indexing_convention():
    # Confirm the returned "stages" pairing (x[i], y[i-1]) against a converged, hand-
    # verified solution rather than re-deriving fsolve convergence in this test.
    v, equil, var_names, var_values = _hand_consistent_n1_solution()
    num_stages = 2
    x = v[2 * num_stages : 3 * num_stages]
    y = v[3 * num_stages : 4 * num_stages]
    stages = [{"stage": i, "liquid": float(x[i]), "vapor": float(y[i - 1])} for i in range(1, 2)]
    assert stages == [{"stage": 1, "liquid": pytest.approx(0.2), "vapor": pytest.approx(0.6)}]
