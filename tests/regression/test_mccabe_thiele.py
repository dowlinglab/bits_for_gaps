"""Regression: McCabe-Thiele equilibrium-stage compositions (paper Fig 9c).

The stage table is the paper's end-to-end validation: the distillation column designed
with the BITS-for-GAPS GP *surrogate* reproduces the column designed with the *Wilson*
ground-truth activity model. Reproducing it requires the Julia/Clapeyron VLE distillation
backend, ported in Phase 6 (``examples/vle_distillation/``); the recompute test below is
gated behind ``@pytest.mark.vle`` (needs Julia -- deselected by default).

The consistency checks below read only the committed golden JSON and pin the paper's
claim (surrogate == Wilson within tolerance), so they run in the default suite.

The recompute itself (``wilson_column``/``surrogate_column``) lives in
``paper.figures.fig09_mccabe_thiele`` (Phase 7) -- this test imports it rather than
reimplementing it, since Fig 9's plotting script needs the exact same columns.
"""
import pytest

STAGES = [1, 2, 3, 4]
PHASES = ["liquid", "vapor"]


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


@pytest.mark.vle
def test_reproduce_stage_table_with_distillation_backend(golden):
    # Phase 6/7: recompute the stage table through the ported VLE distillation backend
    # and diff against the golden (transcribed from paper Fig 9c, hence the looser-
    # than-report atol below -- see paper/golden/README.md and HANDOFF.md).
    from paper.figures.fig09_mccabe_thiele import surrogate_column, wilson_column

    g = golden("mccabe_thiele_stages.json")

    wilson = wilson_column()
    assert wilson["converged"], wilson["warnings"]
    surrogate = surrogate_column()
    assert surrogate["converged"], surrogate["warnings"]

    for key, computed, atol in [("wilson", wilson["stages"], 0.015),
                               ("surrogate", surrogate["stages"], 0.05)]:
        for exp_stage, comp_stage in zip(g["stages"], computed):
            assert comp_stage["stage"] == exp_stage["stage"]
            for phase in PHASES:
                assert comp_stage[phase] == pytest.approx(exp_stage[key][phase], abs=atol)
