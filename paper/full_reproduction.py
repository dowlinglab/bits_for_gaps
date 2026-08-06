"""From-scratch stochastic reproduction of the full adaptive loop.

Runs ``bits_for_gaps.sampler.BitsForGaps`` end-to-end against the same Wilson/
Clapeyron black box, HMC config, and 2-D input space as the paper's published run
(Jones & Dowling 2026) -- see ``examples/vle_distillation/run_case_study.py`` for the
same configuration at a shorter, demo-sized iteration count.

This is a ONE-TIME validation exercise, not part of the regression suite: results are
stochastic (HMC + posterior-mixture sampling both draw randomness TensorFlow does not
let us seed bitwise -- see ``bits_for_gaps.mixture``'s module docstring) and are
expected to be *qualitatively*, not bitwise, consistent with the paper. Documented
runtime is ~25-30 minutes on a laptop (15 outer iterations, each running a
4-chain/5000-sample HMC fit plus a 50-draw full-grid posterior-predictive diagnostic).

All artifacts go to ``--out-dir`` (default ``results_remaked/phase9_fullrun/``,
already gitignored) -- nothing this script produces is committed; only the numbers in
its ``full_run_summary.json`` feed the write-up appended to ``paper/REPRODUCTION.md``.

Usage::

    export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
    python paper/full_reproduction.py --out-dir results_remaked/phase9_fullrun
"""

import argparse
import json
import os
import sys
import time

import gpflow
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)).rsplit(os.sep, 1)[0]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

# This script dispatches thousands of tiny TF ops (one predict_f/predict_f_samples
# call per bisection step, per z, per posterior draw). On macOS, TF eager's default
# multi-threaded op dispatch spends most wall-clock time on thread wake-up/
# coordination for ops this small -- single-threading it removes a >10x slowdown.
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from bits_for_gaps import mixture
from bits_for_gaps.design import latin_hypercube_design
from bits_for_gaps.kernels import AnisotropicSE
from bits_for_gaps.sampler import BitsForGaps
from bits_for_gaps.transforms import InputTransform, OutputTransform

# Exactly matches examples/vle_distillation/run_case_study.py's paper-trail constants.
BOUNDS = [(1e-6, 0.999), (350.0, 367.0)]
SEED = 10
N_INIT = 10
N_TEST = 10
N_ITERS = 15

INPUT_TRANSFORM = InputTransform(
    forward_fns=[lambda x: np.log(x + 0.1), lambda T: (T - BOUNDS[1][0]) / 17.0],
    backward_fns=[lambda x: np.exp(x) - 0.1, lambda T: 17.0 * T + BOUNDS[1][0]],
)
OUTPUT_TRANSFORM = OutputTransform(forward_fn=np.log, backward_fn=np.exp)

COLUMN_VAR_NAMES = ["xW", "F", "xF", "R", "xD"]
COLUMN_VAR_VALUES = [0.01, 100.0, 0.10, 1.0, 0.43]
COLUMN_N_STAGES = 4
COLUMN_FEED_STAGE = 3
Z_GRID_SIZE = 50


def _predict_split(record, XGP, seed, size):
    """Physical-space mixture-posterior draws at held-out points -- matches the shape
    (n_points, n_draws) of the paper's ``gp_predict_{train,test}_{iter}`` files.

    Uses a 15-component hyperparameter-posterior subset (``noGaussians``, the same
    mixture size the acquisition function itself uses) rather than the paper's own
    dedicated 500-sample subset -- a documented simplification (see
    ``paper/REPRODUCTION.md``), not a bitwise match.
    """
    yGP_draws = mixture.sample_gp_posterior_mixture(
        record.trace, record.GPmodel, XGP, seed=seed, size=size
    )
    return OUTPUT_TRANSFORM.backward(yGP_draws).T


def run(out_dir, n_init=N_INIT, n_test=N_TEST, n_iters=N_ITERS, seed=SEED):
    from vle_distillation import activity_model, distillation, equilibrium
    from vle_distillation import phase_diagram as pd

    os.makedirs(out_dir, exist_ok=True)
    X_init, X_test = latin_hypercube_design(BOUNDS, n_train=n_init, n_test=n_test, seed=seed)
    y_init = np.array([activity_model.black_box(z, T)[0] for z, T in X_init])
    y_test = np.array([activity_model.black_box(z, T)[0] for z, T in X_test])

    np.savetxt(os.path.join(out_dir, "lhs_design"), X_init)
    np.savetxt(os.path.join(out_dir, "lhs_test_points"), X_test)
    np.savetxt(os.path.join(out_dir, "activity_data_1"), np.column_stack([X_init, y_init]))
    np.savetxt(os.path.join(out_dir, "activity_test_points"), np.column_stack([X_test, y_test]))

    bfg = BitsForGaps(
        black_box=activity_model.black_box,
        bounds=BOUNDS,
        kernel=AnisotropicSE.paper_2d(),
        mean_fxn=gpflow.mean_functions.Zero(),
        likelihood_variance=0.1,
        input_transform=INPUT_TRANSFORM,
        output_transform=OUTPUT_TRANSFORM,
        iters=n_iters,
        exp_name="phase9_fullrun",
    )
    bfg.seed = seed
    bfg.noSamples, bfg.noBurnIn = 5000, 0
    bfg.noChains, bfg.noLeapfrogSteps, bfg.stepSize = 4, 5, 0.05
    bfg.noAdaptSteps, bfg.targetAccept, bfg.adaptRate = 5, 0.9, 0.1
    bfg.noGaussians, bfg.noRestarts = 15, 10
    bfg.noGPpredictions = 50  # matches the paper run's 50 full-grid posterior draws

    print(
        f"Running {n_iters} adaptive design iterations from {n_init} initial points "
        f"({n_test} held out)...",
        flush=True,
    )
    t0 = time.time()
    history = bfg.run(X_init, y_init, checkpoint_dir=out_dir, predict_grid=True)
    elapsed = time.time() - t0
    print(
        f"Done in {elapsed / 3600:.2f} h. Final design has {history.last.XData.shape[0]} points.",
        flush=True,
    )

    # Fig 8/9-style phase diagram + McCabe-Thiele column from the final, genuinely
    # 15-iteration adaptively-trained GP -- contrast with
    # paper/figures/fig09_mccabe_thiele.py's surrogate_column(), which deliberately
    # trains a throwaway 30-point-LHS/MLE-fit GP instead of running this loop.
    #
    # MUST run before the test-RMSE loop below: `_predict_split` (via
    # `mixture.sample_gp_posterior_mixture`) mutates `record.GPmodel.kernel` in place,
    # and `history.last.GPmodel` is the SAME object -- running the RMSE loop first would
    # leave the kernel at an arbitrary leftover single-hyperparameter state and produce
    # a spurious non-converging column. Uses `surrogate_gamma_averaged` (matches the
    # paper's own posterior-averaging construction) rather than `surrogate_gamma`'s
    # single point-estimate, for the same reason: robust to any one hyperparameter draw
    # being atypical.
    GPmodel = history.last.GPmodel
    trace_final = history.last.trace
    z_grid = np.linspace(0.0, 1.0, Z_GRID_SIZE)
    z_w, T_w, y1_w = pd.vle_curve(pd.wilson_gamma, z_grid=z_grid)

    def surrogate_gamma_fn(z, T):
        return pd.surrogate_gamma_averaged(
            z, T, GPmodel, INPUT_TRANSFORM, OUTPUT_TRANSFORM, trace_final, seed=seed
        )

    z_s, T_s, y1_s = pd.vle_curve(surrogate_gamma_fn, z_grid=z_grid)
    np.savetxt(os.path.join(out_dir, "gt_Wilson_data"), np.column_stack([z_w, T_w, y1_w]))
    np.savetxt(
        os.path.join(out_dir, "phase_diagram_surrogate_final"), np.column_stack([z_s, T_s, y1_s])
    )

    # Fig-5-style train/test posterior-predictive draws at the first and last
    # iteration (mirrors gp_predict_{train,test}_{iter} in paper/data/), plus a
    # per-iteration test-RMSE trace (paper's headline "accuracy improves" claim).
    # Runs AFTER the phase-diagram/column above -- see the comment there.
    X_test_gp = INPUT_TRANSFORM.forward(X_test)
    X_train_gp = INPUT_TRANSFORM.forward(X_init)
    first_it, last_it = history[0].iteration, history[-1].iteration
    test_rmse_by_iter = {}
    for record in history:
        yhat_test = _predict_split(record, X_test_gp, seed, size=bfg.noGaussians)
        rmse = float(np.sqrt(((y_test.reshape(-1, 1) - yhat_test) ** 2).mean()))
        test_rmse_by_iter[record.iteration] = rmse
        if record.iteration in (first_it, last_it):
            yhat_train = _predict_split(record, X_train_gp, seed, size=bfg.noGaussians)
            np.savetxt(os.path.join(out_dir, f"gp_predict_train_{record.iteration}"), yhat_train)
            np.savetxt(os.path.join(out_dir, f"gp_predict_test_{record.iteration}"), yhat_test)

    equil_wilson = equilibrium.make_equilibrium_function(z_w, y1_w)
    equil_surrogate = equilibrium.make_equilibrium_function(z_s, y1_s)
    column_wilson = distillation.solve_column(
        COLUMN_N_STAGES, COLUMN_FEED_STAGE, equil_wilson, COLUMN_VAR_NAMES, COLUMN_VAR_VALUES
    )
    try:
        column_surrogate = distillation.solve_column(
            COLUMN_N_STAGES, COLUMN_FEED_STAGE, equil_surrogate, COLUMN_VAR_NAMES, COLUMN_VAR_VALUES
        )
    except Exception as exc:  # the adaptive surrogate may not be well-behaved enough
        column_surrogate = {"converged": False, "warnings": [repr(exc)], "stages": []}

    summary = {
        "config": {"n_init": n_init, "n_test": n_test, "n_iters": n_iters, "seed": seed},
        "elapsed_hours": elapsed / 3600,
        "final_n_points": int(history.last.XData.shape[0]),
        "rhat": {str(r.iteration): r.rhat.tolist() for r in history},
        "ess": {str(r.iteration): r.ess.tolist() for r in history},
        "max_entropy": {str(r.iteration): r.max_entropy for r in history},
        "test_rmse": {str(k): v for k, v in test_rmse_by_iter.items()},
        "hyperparameter_posterior_final": {
            "mean": history.last.trace.mean(axis=0).tolist(),
            "median": np.median(history.last.trace, axis=0).tolist(),
            "std": history.last.trace.std(axis=0).tolist(),
        },
        "column_wilson_converged": bool(column_wilson.get("converged", False)),
        "column_surrogate_converged": bool(column_surrogate.get("converged", False)),
        "column_wilson_stages": column_wilson.get("stages", []),
        "column_surrogate_stages": column_surrogate.get("stages", []),
    }
    summary_path = os.path.join(out_dir, "full_run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {summary_path}", flush=True)
    return history, summary


if __name__ == "__main__":
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        default=os.path.join(REPO_ROOT, "results_remaked", "phase9_fullrun"),
        help="Gitignored output directory (default: results_remaked/phase9_fullrun)",
    )
    parser.add_argument("--n-iters", type=int, default=N_ITERS)
    args = parser.parse_args()
    run(args.out_dir, n_iters=args.n_iters)
