"""Extract golden scalar targets from the archived published run (iteration 15).

Pure NumPy; reads the read-only old-repo archive and writes small JSON files into
paper/golden/ (a sibling of this script). Run from anywhere:

    python paper/extract_golden.py

Archived run = results/less_x_new_manuscript_revisions, iteration 15 = the published
run whose R-hat/ESS match paper Fig 10 exactly.
"""
import json
import os

import numpy as np

ARCHIVE = os.path.expanduser(
    "~/DowlingLab/CAREER/entropy_driven_hybrid_models_code/entropy_driven_hms/"
    "results/less_x_new_manuscript_revisions"
)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(OUT, exist_ok=True)


def summarize(dist):
    """Summary stats of a 1-D distribution (over mixture draws)."""
    dist = np.asarray(dist, dtype=float)
    return {
        "mean": float(np.mean(dist)),
        "median": float(np.median(dist)),
        "std": float(np.std(dist)),
        "q05": float(np.quantile(dist, 0.05)),
        "q95": float(np.quantile(dist, 0.95)),
        "min": float(np.min(dist)),
        "max": float(np.max(dist)),
    }


def write_json(name, obj):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# 1. HMC convergence diagnostics (paper Fig 10)
# ---------------------------------------------------------------------------
rhat = np.loadtxt(os.path.join(ARCHIVE, "rhat_value_15.txt"))
ess = np.loadtxt(os.path.join(ARCHIVE, "ess_value_15.txt"))
write_json("hmc_diagnostics.json", {
    "description": "HMC convergence diagnostics for the hierarchical-GP hyperparameter "
                   "posterior at the published iteration (15).",
    "source": "results/less_x_new_manuscript_revisions/{rhat,ess}_value_15.txt",
    "paper_reference": "Jones & Dowling 2026, Figure 10",
    "param_order": ["std_dev", "lengthscale_1", "lengthscale_2"],
    "rhat": [float(v) for v in rhat],
    "ess": [float(v) for v in ess],
    "rhat_paper_rounded": [1.0052, 1.0073, 1.0088],
    "ess_paper_rounded": [1468.3, 2428.1, 653.1],
    "tol": {"rhat_atol": 5e-4, "ess_rtol": 1e-3},
})

# ---------------------------------------------------------------------------
# 2. Fig 5 train/test RMSE & MAE at iters 1 and 15
#    y_true from activity_data_1 (fixed 10 train pts) / activity_test_points (10 test).
#    Predictions: gp_predict_{train,test}_{iter} = (n_points, 500 mixture draws).
#    Metrics computed per-draw (axis=0 over points) => 500-length distributions,
#    exactly as fxns/train_test_split_proh.plot_error_bx_n_wskr.
# ---------------------------------------------------------------------------
y_train = np.loadtxt(os.path.join(ARCHIVE, "activity_data_1"))[:, 2]
y_test = np.loadtxt(os.path.join(ARCHIVE, "activity_test_points"))[:, 2]

fig5 = {
    "description": "Train/test RMSE & MAE of the GP surrogate vs the Wilson activity "
                   "coefficient, as distributions over 500 mixture draws (per-draw metric "
                   "over the fixed 10 train / 10 test points). Iter 1 = initial design, "
                   "iter 15 = final BITS-for-GAPS design.",
    "source": "gp_predict_{train,test}_{iter} vs activity_data_1[:,2] / "
              "activity_test_points[:,2]",
    "paper_reference": "Jones & Dowling 2026, Figure 5 (error box-and-whisker panels)",
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "n_draws": 500,
    "tol": {"rtol": 0.02, "atol": 1e-6},
}
for it in (1, 15):
    yhat_train = np.loadtxt(os.path.join(ARCHIVE, f"gp_predict_train_{it}"))
    yhat_test = np.loadtxt(os.path.join(ARCHIVE, f"gp_predict_test_{it}"))
    diff_train = y_train.reshape(-1, 1) - yhat_train
    diff_test = y_test.reshape(-1, 1) - yhat_test
    fig5[f"iter_{it}"] = {
        "rmse_train": summarize(np.sqrt((diff_train ** 2).mean(axis=0))),
        "mae_train": summarize(np.abs(diff_train).mean(axis=0)),
        "rmse_test": summarize(np.sqrt((diff_test ** 2).mean(axis=0))),
        "mae_test": summarize(np.abs(diff_test).mean(axis=0)),
    }
write_json("fig5_error_metrics.json", fig5)

# ---------------------------------------------------------------------------
# 3. Hyperparameter posterior summary (iter 15, chain-0 constrained samples)
#    cols = [std_dev, lengthscale_1, lengthscale_2]
# ---------------------------------------------------------------------------
theta = np.loadtxt(os.path.join(ARCHIVE, "param_posterior_samples_15"))
names = ["std_dev", "lengthscale_1", "lengthscale_2"]
write_json("hyperparameter_posterior.json", {
    "description": "Posterior summary of the GP kernel hyperparameters at the published "
                   "iteration (chain-0 constrained HMC samples).",
    "source": "results/less_x_new_manuscript_revisions/param_posterior_samples_15 (5000 x 3)",
    "paper_reference": "Jones & Dowling 2026, Fig 10 (marginals)",
    "param_order": names,
    "n_samples": int(theta.shape[0]),
    "mean": {names[j]: float(theta[:, j].mean()) for j in range(3)},
    "median": {names[j]: float(np.median(theta[:, j])) for j in range(3)},
    "std": {names[j]: float(theta[:, j].std()) for j in range(3)},
    "q05": {names[j]: float(np.quantile(theta[:, j], 0.05)) for j in range(3)},
    "q95": {names[j]: float(np.quantile(theta[:, j], 0.95)) for j in range(3)},
    "tol": {"rtol": 0.05},
})

# ---------------------------------------------------------------------------
# 4. McCabe-Thiele stage table (paper Fig 9c) -- values from the paper (the column
#    design + surrogate/Wilson stage compositions). Reproduction needs the Julia VLE
#    distillation backend (Phase 6 port), so the recompute test is @pytest.mark.vle.
# ---------------------------------------------------------------------------
write_json("mccabe_thiele_stages.json", {
    "description": "McCabe-Thiele equilibrium-stage liquid/vapor mole fractions of PrOH "
                   "for the Geankoplis 11.4-1 column, computed with the Wilson ground "
                   "truth vs the BITS-for-GAPS GP surrogate.",
    "source": "run_example.py: solve_distillation_model(n=4, feed_stage=3, ...) with "
              "equilibrium.water_proh_eqm_julia (Wilson) and water_proh_eqm (surrogate).",
    "paper_reference": "Jones & Dowling 2026, Figure 9c",
    "column_spec": {"xW": 0.01, "F": 100.0, "xF": 0.10, "R": 1.0, "xD": 0.43,
                    "n_stages": 4, "feed_stage": 3},
    "component": "PrOH",
    "stages": [
        {"stage": 1, "wilson": {"liquid": 0.22, "vapor": 0.43},
         "surrogate": {"liquid": 0.25, "vapor": 0.43}},
        {"stage": 2, "wilson": {"liquid": 0.05, "vapor": 0.32},
         "surrogate": {"liquid": 0.05, "vapor": 0.34}},
        {"stage": 3, "wilson": {"liquid": 0.03, "vapor": 0.24},
         "surrogate": {"liquid": 0.03, "vapor": 0.24}},
        {"stage": 4, "wilson": {"liquid": 0.02, "vapor": 0.14},
         "surrogate": {"liquid": 0.02, "vapor": 0.15}},
    ],
    # paper reports to 2 decimals; the surrogate-vs-Wilson agreement is the physics claim.
    "tol": {"report_atol": 0.005, "surrogate_vs_wilson_atol": 0.03},
})

print("\nDone.")
