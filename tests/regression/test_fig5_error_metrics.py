"""Regression: surrogate train/test error metrics (paper Fig 5).

Reads the committed reference snapshot (``paper/reference/fig5_error_metrics.json``) of the
RMSE/MAE distributions (over 500 mixture draws) at iterations 1 and 15, extracted from
the archived ``gp_predict_*`` files. Pins the paper's central result: the BITS-for-GAPS
design dramatically reduces *held-out test* error between the initial (iter 1) and final
(iter 15) designs, while keeping train error small.

Pure (JSON only) -- no TensorFlow, no Julia.
"""

import pytest

SUMMARY_KEYS = {"mean", "median", "std", "q05", "q95", "min", "max"}
METRICS = ["rmse_train", "mae_train", "rmse_test", "mae_test"]


def test_structure(reference):
    g = reference("fig5_error_metrics.json")
    assert g["n_train"] == 10 and g["n_test"] == 10 and g["n_draws"] == 500
    for it in ("iter_1", "iter_15"):
        assert it in g
        for m in METRICS:
            assert set(g[it][m]) >= SUMMARY_KEYS


def test_all_metrics_positive_and_finite(reference):
    g = reference("fig5_error_metrics.json")
    for it in ("iter_1", "iter_15"):
        for m in METRICS:
            for k in SUMMARY_KEYS:
                v = g[it][m][k]
                assert v > 0.0
                assert v == v and abs(v) != float("inf")


def test_quantiles_ordered(reference):
    g = reference("fig5_error_metrics.json")
    for it in ("iter_1", "iter_15"):
        for m in METRICS:
            s = g[it][m]
            assert s["min"] <= s["q05"] <= s["median"] <= s["q95"] <= s["max"]


@pytest.mark.parametrize("metric", ["rmse_test", "mae_test"])
def test_test_error_drops_from_iter1_to_iter15(reference, metric):
    # The paper's key claim: held-out test error falls sharply as BITS for GAPS
    # adds points. Median test RMSE ~4.3 -> ~0.67; require at least a 3x reduction.
    g = reference("fig5_error_metrics.json")
    early = g["iter_1"][metric]["median"]
    final = g["iter_15"][metric]["median"]
    assert final < early / 3.0


def test_train_error_small_and_not_worse(reference):
    # Train error is already small at iter 1 and does not get worse by iter 15.
    g = reference("fig5_error_metrics.json")
    assert g["iter_1"]["rmse_train"]["median"] < 1.0
    assert g["iter_15"]["rmse_train"]["median"] <= g["iter_1"]["rmse_train"]["median"]
