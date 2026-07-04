"""McCabe-Thiele-consistent stage-by-stage distillation column solver.

Ported from the paper code's ``distillation_model.py`` + ``solve_distillation.py``
(itself a Python port of a course MATLAB script, ``distillation_nonlinear_equations.m``
-- see the inline notes on the reboiler equation below, kept verbatim from the original
port). Solves the column's nonlinear mass-balance + equilibrium-stage system via
``scipy.optimize.fsolve``, given an equilibrium function ``y = equil(x)`` (see
``equilibrium.py`` / ``phase_diagram.py``).

Variable vector layout (0-based), matching the original port exactly:
    v = [L_0..L_n, V_1..V_{n+1} (stored at V[0]..V[n]), x_0..x_n, y_1..y_{n+1}
         (stored at y[0]..y[n]), D, R, W, F, xF, q]
Stage 0 = condenser/distillate, stage n = reboiler/bottoms.

Pure NumPy/SciPy -- no Julia, no TensorFlow. Plotting (McCabe-Thiele diagram) is a
separate, lazily-imported-matplotlib function, :func:`plot_mccabe_thiele`, so this
module is importable without matplotlib.
"""
import numpy as np
from scipy.optimize import fsolve


def _distillation_residuals(v, n, feed_stage_idx, equil, fixed_idx, fixed_vals):
    """Residual vector for the column's nonlinear equation system.

    ``feed_stage_idx`` is the 0-based feed tray index; ``fixed_idx`` are 0-based
    indices into ``v`` for the fixed (specified) variables.
    """
    num_stages = n + 1  # includes condenser (stage 0) and reboiler (stage n)

    offset_L, offset_V = 0, num_stages
    offset_x, offset_y = 2 * num_stages, 3 * num_stages
    offset_params = 4 * num_stages  # D, R, W, F, xF, q

    L = v[offset_L: offset_L + num_stages]
    V = v[offset_V: offset_V + num_stages]
    x = v[offset_x: offset_x + num_stages]
    y = v[offset_y: offset_y + num_stages]
    D, R, W, F, xF, q = v[offset_params: offset_params + 6]

    f = np.zeros_like(v)

    # Overall mass balance (stages 1..n): L[i-1] + V[i] - L[i] - V[i-1] = 0.
    offset_eq = 0
    for idx in range(1, n + 1):
        f[offset_eq + idx - 1] = L[idx - 1] + V[idx] - L[idx] - V[idx - 1]
    f[offset_eq + feed_stage_idx] += F  # feed enters above this (0-based) stage

    # Component mass balance (stages 1..n).
    offset_eq = n
    for idx in range(1, n + 1):
        f[offset_eq + idx - 1] = (L[idx - 1] * x[idx - 1] + V[idx] * y[idx]
                                  - L[idx] * x[idx] - V[idx - 1] * y[idx - 1])
    f[offset_eq + feed_stage_idx] += F * xF

    # Constant molar overflow (stages 1..n), adjusted by feed quality q at the feed stage.
    offset_eq = 2 * n
    for idx in range(1, n + 1):
        f[offset_eq + idx - 1] = L[idx] - L[idx - 1]
    f[offset_eq + feed_stage_idx] -= F * q

    # Equilibrium (stages 1..n): y_i = equil(x_i).
    offset_eq = 3 * n
    for idx in range(1, n + 1):
        f[offset_eq + idx - 1] = y[idx - 1] - equil(x[idx])

    # Fixed (specified) variables.
    offset_eq = 4 * n
    for i in range(len(fixed_idx)):
        f[offset_eq + i] = v[fixed_idx[i]] - fixed_vals[i]

    # Condenser and reboiler equations.
    offset_eq = 4 * n + len(fixed_idx)
    f[offset_eq + 0] = y[0] - x[0]        # total condenser: y_1 = x_0
    # NOTE (kept from the original port): a partial reboiler is usually
    # y_{n+1} = equil(x_W). The source MATLAB instead sets x_n = y_{n+1} (liquid
    # leaving stage n equals the vapor leaving the reboiler); replicated as-is here
    # for physics parity with the paper's published column design.
    f[offset_eq + 1] = x[n] - y[n]
    f[offset_eq + 2] = L[0] - R * D       # reflux: L_0 = R * D
    f[offset_eq + 3] = V[0] - L[0] - D    # condenser balance: V_1 = L_0 + D
    f[offset_eq + 4] = L[n] - V[n] - W    # reboiler balance: L_n = V_{n+1} + W

    return f


def _resolve_fixed_indices(n_stages, var_names, var_values):
    """Map named fixed variables (e.g. ``'xD'``, ``'F'``) to 0-based indices in ``v``."""
    num_stages = n_stages + 1
    offset_L, offset_V = 0, num_stages
    offset_x, offset_y = 2 * num_stages, 3 * num_stages
    offset_params = 4 * num_stages

    fixed_idx = np.zeros(len(var_names), dtype=int)
    F_initial = 1.0

    for i, (name, val) in enumerate(zip(var_names, var_values)):
        if name == "xD":
            fixed_idx[i] = offset_x + 0
        elif name == "xW":
            fixed_idx[i] = offset_x + n_stages
        elif name == "L0":
            fixed_idx[i] = offset_L + 0
        elif name == "V1":
            fixed_idx[i] = offset_V + 0
        elif name == f"L{n_stages}":
            fixed_idx[i] = offset_L + n_stages
        elif name == f"V{n_stages + 1}":
            fixed_idx[i] = offset_V + n_stages
        elif len(name) > 1 and name[0] in ("L", "V", "x", "y") and name[1:].isdigit():
            stage_1_based = int(name[1:])
            if name[0] in ("L", "x"):
                fixed_idx[i] = (offset_L if name[0] == "L" else offset_x) + stage_1_based
            else:  # V, y map to array index (stage - 1)
                fixed_idx[i] = (offset_V if name[0] == "V" else offset_y) + stage_1_based - 1
        elif name == "D":
            fixed_idx[i] = offset_params + 0
        elif name == "R":
            fixed_idx[i] = offset_params + 1
        elif name == "W":
            fixed_idx[i] = offset_params + 2
        elif name == "F":
            fixed_idx[i] = offset_params + 3
            F_initial = val
        elif name == "xF":
            fixed_idx[i] = offset_params + 4
        elif name == "q":
            fixed_idx[i] = offset_params + 5
        else:
            raise ValueError(f"Unrecognized distillation variable name: {name!r}")

    return fixed_idx, F_initial


def solve_column(n_stages, feed_stage, equil, var_names, var_values):
    """Solve the column's nonlinear equation system and return the stage table.

    Parameters
    ----------
    n_stages : int
        Number of theoretical stages, excluding the reboiler (``n`` in the paper code).
    feed_stage : int
        1-based feed tray number.
    equil : callable
        Vapor PrOH mole fraction as a function of liquid PrOH mole fraction,
        ``y = equil(x)`` -- see ``equilibrium.make_equilibrium_function``.
    var_names, var_values : sequence
        Fixed-variable names/values, e.g. ``['xW', 'F', 'xF', 'R', 'xD']`` /
        ``[0.01, 100.0, 0.10, 1.0, 0.43]`` (the Geankoplis 11.4-1 column spec).

    Returns
    -------
    dict with keys ``"converged"`` (bool), ``"stages"`` (list of
    ``{"stage": i, "liquid": x_i, "vapor": y_i}`` for ``i = 1..n_stages``), the 0-based
    solution arrays ``"L"``, ``"V"``, ``"x"``, ``"y"``, and the scalars ``"D"``, ``"R"``,
    ``"W"``, ``"F"``, ``"xF"``, ``"q"``.
    """
    n = n_stages
    feed_stage_idx = feed_stage - 1
    num_stages = n + 1

    fixed_idx, F_initial = _resolve_fixed_indices(n, var_names, var_values)
    fixed_vals = np.array(var_values, dtype=float)

    v0 = np.concatenate([
        F_initial * np.ones(num_stages),   # L
        F_initial * np.ones(num_stages),   # V
        0.5 * np.ones(num_stages),         # x
        0.5 * np.ones(num_stages),         # y
        np.array([0.5 * F_initial, 2.0, 0.5 * F_initial, F_initial, 0.5, 1.0]),
    ])
    for i, idx in enumerate(fixed_idx):
        v0[idx] = fixed_vals[i]

    def residual_func(v_solve):
        return _distillation_residuals(v_solve, n, feed_stage_idx, equil, fixed_idx,
                                       fixed_vals)

    v_solution, _infodict, ier, _mesg = fsolve(residual_func, v0, full_output=True)

    offset_L, offset_V = 0, num_stages
    offset_x, offset_y = 2 * num_stages, 3 * num_stages
    offset_params = 4 * num_stages
    L = v_solution[offset_L: offset_L + num_stages]
    V = v_solution[offset_V: offset_V + num_stages]
    x = v_solution[offset_x: offset_x + num_stages]
    y = v_solution[offset_y: offset_y + num_stages]
    D, R, W, F, xF, q = v_solution[offset_params: offset_params + 6]

    # Stage i's liquid/vapor (both leaving stage i): x[i] / y[i-1] in the 0-based arrays.
    stages = [{"stage": i, "liquid": float(x[i]), "vapor": float(y[i - 1])}
             for i in range(1, n + 1)]

    # fsolve has no bounds: given a poorly-resolved equil() (e.g. too coarse a z-grid
    # in phase_diagram.vle_curve), it can converge (residual ~ 0) to a spurious root
    # outside the physical [0, 1] mole-fraction range. Flag it rather than silently
    # returning nonphysical numbers -- inherited from the original MATLAB-derived
    # port, which printed the same warnings.
    warnings = []
    if not (ier == 1):
        warnings.append("fsolve did not report convergence")
    if np.any((x < -1e-6) | (x > 1.0 + 1e-6)):
        warnings.append("liquid mole fractions outside [0, 1]")
    if np.any((y < -1e-6) | (y > 1.0 + 1e-6)):
        warnings.append("vapor mole fractions outside [0, 1]")
    if np.any(L <= 0) or np.any(V <= 0):
        warnings.append("non-positive liquid/vapor flow rates")

    return {
        "converged": ier == 1 and not warnings, "warnings": warnings, "stages": stages,
        "L": L, "V": V, "x": x, "y": y,
        "D": float(D), "R": float(R), "W": float(W), "F": float(F),
        "xF": float(xF), "q": float(q),
    }


def plot_mccabe_thiele(result, equil, feed_stage, ax=None):
    """McCabe-Thiele diagram for a :func:`solve_column` result (lazy matplotlib import)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(5, 5))

    x, y = result["x"], result["y"]
    n = len(x) - 1
    feed_stage_idx = feed_stage - 1

    ax.plot([0, 1], [0, 1], "k-", linewidth=2, label="Parity")
    x_eq = np.linspace(0, 1, 100)
    ax.plot(x_eq, equil(x_eq), "b-", label="Equilibrium")
    for i in range(n):
        ax.plot([x[i], x[i + 1]], [y[i], y[i]], "k--", linewidth=2.0)
        ax.plot([x[i + 1], x[i + 1]], [y[i], y[i + 1]], "k--", linewidth=2.0)
    ax.plot(x[:feed_stage_idx + 1], y[:feed_stage_idx + 1], "r.-", label="Enriching OL")
    ax.plot(x[feed_stage_idx + 1:], y[feed_stage_idx + 1:], "g.-", label="Stripping OL")
    ax.plot(x[1:], y[:-1], "wo", markeredgecolor="b", label="Stages")
    ax.plot([result["xF"]], [result["xF"]], "ko", label="Feed")
    ax.set_xlabel(r"Mole Fraction PrOH ($\ell$), $z_{\mathrm{PrOH}}^{(\ell)}$")
    ax.set_ylabel(r"Mole Fraction PrOH ($v$), $z_{\mathrm{PrOH}}^{(v)}$")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    return ax
