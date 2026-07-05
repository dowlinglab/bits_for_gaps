"""Regression: GP hyperparameter posterior summary (paper Fig 10 marginals).

Reads ``paper/reference/hyperparameter_posterior.json`` -- the posterior mean/median/std
and 5%/95% quantiles of the three kernel hyperparameters at the published iteration --
and checks the summary is well-formed and physically sensible. This pins the target the
mixture-posterior code must reproduce after the Phase 4/5 refactor.

Pure (JSON only) -- no TensorFlow, no Julia.
"""

PARAMS = ["std_dev", "lengthscale_1", "lengthscale_2"]


def test_param_order(reference):
    g = reference("hyperparameter_posterior.json")
    assert g["param_order"] == PARAMS
    assert g["n_samples"] == 5000


def test_positive_and_finite(reference):
    g = reference("hyperparameter_posterior.json")
    for p in PARAMS:
        for stat in ("mean", "median", "std", "q05", "q95"):
            v = g[stat][p]
            assert v > 0.0
            assert v == v and abs(v) != float("inf")


def test_quantiles_bracket_median(reference):
    g = reference("hyperparameter_posterior.json")
    for p in PARAMS:
        assert g["q05"][p] < g["median"][p] < g["q95"][p]


def test_values_in_expected_ranges(reference):
    # Loose sanity envelope around the published posterior means; guards against a
    # reference regeneration that silently drifts far from the paper.
    g = reference("hyperparameter_posterior.json")
    assert 1.0 < g["mean"]["std_dev"] < 2.0
    assert 0.4 < g["mean"]["lengthscale_1"] < 1.5
    assert 2.0 < g["mean"]["lengthscale_2"] < 4.5
