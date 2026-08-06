"""Fig 9 -- McCabe-Thiele distillation column diagram (Wilson vs. GP surrogate).

Built entirely on ``examples/vle_distillation``'s ``phase_diagram``/``equilibrium``/
``distillation`` modules -- this module only adds the plotting + LHS/GP setup on top.

``wilson_column()``/``surrogate_column()`` are the exact same recompute
``tests/regression/test_mccabe_thiele.py``'s gated stage-table regression pins against
``paper/reference/mccabe_thiele_stages.json`` -- defined here so the figure and the test
share one implementation instead of two copies (the test imports these functions rather
than redefining them). See that test's module docstring for why the surrogate uses a
fresh 30-point LHS + MLE fit rather than the full 15-iteration adaptive loop.
"""

import os

import numpy as np

# Geankoplis Ex. 11.4-1 column spec (paper Fig 9c) -- see paper/reference/mccabe_thiele_stages.json.
COLUMN_VAR_NAMES = ["xW", "F", "xF", "R", "xD"]
COLUMN_VAR_VALUES = [0.01, 100.0, 0.10, 1.0, 0.43]
COLUMN_N_STAGES = 4
COLUMN_FEED_STAGE = 3
Z_GRID_SIZE = 50  # the distillation solver's fsolve convergence is sensitive to how
# well-resolved the equilibrium curve is (confirmed empirically) --
# do not shrink this casually.

# Paper's exact 2-D VLE search space + GP input/output transforms (see
# examples/vle_distillation/run_case_study.py for the full paper-trail justification).
BOUNDS = [(1e-6, 0.999), (350.0, 367.0)]
SEED = 10
N_SURROGATE_TRAIN = 30


def wilson_column():
    """Recompute the Wilson (ground-truth) column via Clapeyron -- no archived data."""
    from vle_distillation import distillation, equilibrium
    from vle_distillation import phase_diagram as pd

    z_grid = np.linspace(0.0, 1.0, Z_GRID_SIZE)
    z, _T_bub, y = pd.vle_curve(pd.wilson_gamma, z_grid=z_grid)
    equil = equilibrium.make_equilibrium_function(z, y)
    result = distillation.solve_column(
        COLUMN_N_STAGES, COLUMN_FEED_STAGE, equil, COLUMN_VAR_NAMES, COLUMN_VAR_VALUES
    )
    result["equil"] = equil
    return result


def surrogate_column():
    """Recompute the surrogate column from a freshly-trained (non-adaptive) GP.

    Trains on an LHS design (seed matching the paper's ``my_system.seed = 10``)
    evaluated against the same Clapeyron Wilson model, using ``gp.maximize_lml`` (a
    fast, deterministic MLE fit) rather than the full adaptive
    ``BitsForGaps.run``/HMC loop -- reproducing the paper's exact 15-iteration
    *adaptively*-designed surrogate bit-for-bit is out of scope (would mean re-running
    the stochastic adaptive loop; see ``paper/REPRODUCTION.md``). A moderately dense
    LHS (30 points) gives a GP mean prediction close enough to Wilson for the
    distillation solver to converge to a physical column, which a sparser/adaptive-
    but-short design did not (confirmed empirically while developing this).
    """
    import gpflow
    from vle_distillation import activity_model, distillation, equilibrium
    from vle_distillation import phase_diagram as pd

    from bits_for_gaps import gp as gp_mod
    from bits_for_gaps.design import latin_hypercube_design
    from bits_for_gaps.kernels import AnisotropicSE
    from bits_for_gaps.transforms import InputTransform, OutputTransform

    input_transform = InputTransform(
        forward_fns=[lambda x: np.log(x + 0.1), lambda T: (T - BOUNDS[1][0]) / 17.0],
        backward_fns=[lambda x: np.exp(x) - 0.1, lambda T: 17.0 * T + BOUNDS[1][0]],
    )
    output_transform = OutputTransform(forward_fn=np.log, backward_fn=np.exp)

    X_train, _ = latin_hypercube_design(BOUNDS, n_train=N_SURROGATE_TRAIN, n_test=0, seed=SEED)
    y_train = np.array([activity_model.activity_coefficients(z, T)[0] for z, T in X_train])

    XGP = input_transform.forward(X_train)
    yGP = output_transform.forward(y_train.reshape(-1, 1))
    GPmodel = gp_mod.build_gp(
        XGP,
        yGP,
        mean_fxn=gpflow.mean_functions.Zero(),
        kernel_fxn=AnisotropicSE.paper_2d(),
        likelihood_var=0.1,
    )
    _result, GPmodel = gp_mod.maximize_lml(GPmodel)

    def surrogate_gamma_fn(z, T):
        return pd.surrogate_gamma(z, T, GPmodel, input_transform, output_transform)

    z_grid = np.linspace(0.0, 1.0, Z_GRID_SIZE)
    z, _T_bub, y = pd.vle_curve(surrogate_gamma_fn, z_grid=z_grid)
    equil = equilibrium.make_equilibrium_function(z, y)
    result = distillation.solve_column(
        COLUMN_N_STAGES, COLUMN_FEED_STAGE, equil, COLUMN_VAR_NAMES, COLUMN_VAR_VALUES
    )
    result["equil"] = equil
    return result


def make(archive_dir, out_dir, img_fmt="png"):
    """Regenerate both McCabe-Thiele panels (Wilson + surrogate); returns both columns.

    ``archive_dir`` is unused (Fig 9 is a pure physics recompute, no archived data
    needed) -- kept for the common ``make(archive_dir, out_dir)`` signature used by
    ``paper/reproduce.py``.
    """
    from vle_distillation import distillation

    os.makedirs(out_dir, exist_ok=True)
    wilson = wilson_column()
    surrogate = surrogate_column()

    paths = []
    for label, result, letter in [
        ("wilson", wilson, "(a) Wilson"),
        ("surrogate", surrogate, "(b) Surrogate"),
    ]:
        ax = distillation.plot_mccabe_thiele(result, result["equil"], COLUMN_FEED_STAGE)
        ax.set_xlim(-0.02, 0.5)
        ax.set_ylim(-0.02, 0.5)
        ax.text(0.0, 0.45, letter, fontweight="bold")
        out_path = os.path.join(out_dir, f"eight_stages_feed_on_3_{label}.{img_fmt}")
        ax.figure.savefig(out_path, dpi=300, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(ax.figure)
        paths.append(out_path)

    return {"path": paths, "wilson": wilson, "surrogate": surrogate}
