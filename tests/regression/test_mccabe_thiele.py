"""Regression: McCabe-Thiele equilibrium-stage compositions (paper Fig 9c).

The stage table is the paper's end-to-end validation: the distillation column designed
with the BITS-for-GAPS GP *surrogate* reproduces the column designed with the *Wilson*
ground-truth activity model. Reproducing it requires the Julia/Clapeyron VLE distillation
backend, ported in Phase 6 (``examples/vle_distillation/``); the recompute test below is
gated behind ``@pytest.mark.vle`` (needs Julia -- deselected by default).

The consistency checks below read only the committed golden JSON and pin the paper's
claim (surrogate == Wilson within tolerance), so they run in the default suite.
"""
import numpy as np
import pytest

STAGES = [1, 2, 3, 4]
PHASES = ["liquid", "vapor"]

# Geankoplis Ex. 11.4-1 column spec (paper Fig 9c) -- see paper/golden/mccabe_thiele_stages.json.
COLUMN_VAR_NAMES = ["xW", "F", "xF", "R", "xD"]
COLUMN_VAR_VALUES = [0.01, 100.0, 0.10, 1.0, 0.43]
COLUMN_N_STAGES = 4
COLUMN_FEED_STAGE = 3
Z_GRID_SIZE = 50   # matches phase_diagram.Z_MESH; the distillation solver's fsolve
                   # convergence is sensitive to how well-resolved the equilibrium
                   # curve is (confirmed empirically) -- do not shrink this casually.

# Paper's exact 2-D VLE search space + GP input/output transforms (see
# examples/vle_distillation/run_case_study.py for the full paper-trail justification).
BOUNDS = [(1e-6, 0.999), (350.0, 367.0)]
SEED = 10
N_SURROGATE_TRAIN = 30   # LHS points for the surrogate GP (see module docstring below)


def test_column_spec(golden):
    g = golden("mccabe_thiele_stages.json")
    spec = g["column_spec"]
    assert spec == {"xW": 0.01, "F": 100.0, "xF": 0.10, "R": 1.0, "xD": 0.43,
                    "n_stages": 4, "feed_stage": 3}
    assert g["component"] == "PrOH"
    assert [s["stage"] for s in g["stages"]] == STAGES


def test_mole_fractions_physical(golden):
    g = golden("mccabe_thiele_stages.json")
    for s in g["stages"]:
        for model in ("wilson", "surrogate"):
            for phase in PHASES:
                x = s[model][phase]
                assert 0.0 <= x <= 1.0


def test_vapor_richer_than_liquid_in_proh(golden):
    # PrOH is the more volatile key: vapor mole fraction >= liquid on every stage.
    g = golden("mccabe_thiele_stages.json")
    for s in g["stages"]:
        for model in ("wilson", "surrogate"):
            assert s[model]["vapor"] >= s[model]["liquid"]


def test_compositions_decrease_down_the_column(golden):
    # Stage 1 (top) is richest in the light key; compositions fall toward the reboiler.
    g = golden("mccabe_thiele_stages.json")
    for model in ("wilson", "surrogate"):
        liquid = [s[model]["liquid"] for s in g["stages"]]
        vapor = [s[model]["vapor"] for s in g["stages"]]
        assert liquid == sorted(liquid, reverse=True)
        assert vapor == sorted(vapor, reverse=True)


def test_surrogate_matches_wilson(golden):
    # The physics claim of Fig 9c: the surrogate-designed column matches the
    # Wilson-designed column stage-by-stage within the reported tolerance.
    g = golden("mccabe_thiele_stages.json")
    atol = g["tol"]["surrogate_vs_wilson_atol"]
    for s in g["stages"]:
        for phase in PHASES:
            assert abs(s["surrogate"][phase] - s["wilson"][phase]) <= atol


def _wilson_column():
    """Recompute the Wilson (ground-truth) column via Clapeyron -- no archived data."""
    from vle_distillation import distillation, equilibrium
    from vle_distillation import phase_diagram as pd

    z_grid = np.linspace(0.0, 1.0, Z_GRID_SIZE)
    z, _T_bub, y = pd.vle_curve(pd.wilson_gamma, z_grid=z_grid)
    equil = equilibrium.make_equilibrium_function(z, y)
    return distillation.solve_column(COLUMN_N_STAGES, COLUMN_FEED_STAGE, equil,
                                     COLUMN_VAR_NAMES, COLUMN_VAR_VALUES)


def _surrogate_column():
    """Recompute the surrogate column from a freshly-trained (non-adaptive) GP.

    Trains on an LHS design (seed matching the paper's ``my_system.seed = 10``)
    evaluated against the same Clapeyron Wilson model, using ``gp.maximize_lml`` (a
    fast, deterministic MLE fit) rather than the full adaptive
    ``BitsForGaps.run``/HMC loop -- reproducing the paper's exact 15-iteration
    *adaptively*-designed surrogate is Phase 7's job (full figure reproduction), not
    this Phase-6 backend-correctness check. A moderately dense LHS (30 points) gives
    a GP mean prediction close enough to Wilson for the distillation solver to
    converge to a physical column, which a sparser/adaptive-but-short design did not
    (confirmed empirically while developing this test).
    """
    import gpflow

    from bits_for_gaps import gp as gp_mod
    from bits_for_gaps.design import latin_hypercube_design
    from bits_for_gaps.kernels import AnisotropicSE
    from bits_for_gaps.transforms import InputTransform, OutputTransform
    from vle_distillation import activity_model, distillation, equilibrium
    from vle_distillation import phase_diagram as pd

    input_transform = InputTransform(
        forward_fns=[lambda x: np.log(x + 0.1), lambda T: (T - BOUNDS[1][0]) / 17.0],
        backward_fns=[lambda x: np.exp(x) - 0.1, lambda T: 17.0 * T + BOUNDS[1][0]],
    )
    output_transform = OutputTransform(forward_fn=np.log, backward_fn=np.exp)

    X_train, _ = latin_hypercube_design(BOUNDS, n_train=N_SURROGATE_TRAIN, n_test=0,
                                        seed=SEED)
    y_train = np.array([activity_model.activity_coefficients(z, T)[0]
                        for z, T in X_train])

    XGP = input_transform.forward(X_train)
    yGP = output_transform.forward(y_train.reshape(-1, 1))
    GPmodel = gp_mod.build_gp(XGP, yGP, mean_fxn=gpflow.mean_functions.Zero(),
                              kernel_fxn=AnisotropicSE.paper_2d(), likelihood_var=0.1)
    _result, GPmodel = gp_mod.maximize_lml(GPmodel)

    def surrogate_gamma_fn(z, T):
        return pd.surrogate_gamma(z, T, GPmodel, input_transform, output_transform)

    z_grid = np.linspace(0.0, 1.0, Z_GRID_SIZE)
    z, _T_bub, y = pd.vle_curve(surrogate_gamma_fn, z_grid=z_grid)
    equil = equilibrium.make_equilibrium_function(z, y)
    return distillation.solve_column(COLUMN_N_STAGES, COLUMN_FEED_STAGE, equil,
                                     COLUMN_VAR_NAMES, COLUMN_VAR_VALUES)


@pytest.mark.vle
def test_reproduce_stage_table_with_distillation_backend(golden):
    # Phase 6: recompute the stage table through the ported VLE distillation backend
    # and diff against the golden (transcribed from paper Fig 9c, hence the looser-
    # than-report atol below -- see paper/golden/README.md and HANDOFF.md).
    g = golden("mccabe_thiele_stages.json")

    wilson = _wilson_column()
    assert wilson["converged"], wilson["warnings"]
    surrogate = _surrogate_column()
    assert surrogate["converged"], surrogate["warnings"]

    for key, computed, atol in [("wilson", wilson["stages"], 0.015),
                               ("surrogate", surrogate["stages"], 0.05)]:
        for exp_stage, comp_stage in zip(g["stages"], computed):
            assert comp_stage["stage"] == exp_stage["stage"]
            for phase in PHASES:
                assert comp_stage[phase] == pytest.approx(exp_stage[key][phase], abs=atol)
