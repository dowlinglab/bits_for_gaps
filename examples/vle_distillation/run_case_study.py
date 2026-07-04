"""End-to-end entry point for the H2O-PrOH VLE / distillation case study.

LHS initial design -> Clapeyron (Wilson) activity-coefficient evaluation ->
``BitsForGaps.run`` (adaptive entropy-driven design) -> phase diagram + McCabe-Thiele
distillation column, all on the public ``bits_for_gaps`` API.

Configuration (bounds, transforms, kernel, seed) matches the paper's published
``less_x_new_manuscript_revisions`` run (Jones & Dowling 2026) as closely as the
archived code allows to reconstruct -- see HANDOFF.md for the paper trail. This
script demonstrates the ported pipeline; it does not reproduce the paper's exact
15-iteration adaptive run bit-for-bit (that full reproduction is Phase 7's job).

Usage::

    export PYTHON_JULIACALL_HANDLE_SIGNALS=yes   # macOS: mandatory before any Julia use
    python examples/vle_distillation/run_case_study.py
"""
import os
import sys

import gpflow
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bits_for_gaps.design import latin_hypercube_design
from bits_for_gaps.kernels import AnisotropicSE
from bits_for_gaps.sampler import BitsForGaps
from bits_for_gaps.transforms import InputTransform, OutputTransform

from vle_distillation import activity_model, distillation, equilibrium
from vle_distillation import phase_diagram as pd

# Paper's exact 2-D VLE search space: liquid PrOH mole fraction, temperature [K].
BOUNDS = [(1e-6, 0.999), (350.0, 367.0)]
SEED = 10   # matches train_test_split_proh.py's `my_system.seed = 10`

# Paper's exact GP input/output transforms (new_phase_diagram.py's __main__ block):
# log(x + 0.1) keeps the mole-fraction lengthscale well-scaled near the dilute limit;
# min-max normalizing T to [0, 1] matches the kernel's O(1) lengthscale priors; log(y)
# trains the GP on log-activity-coefficient (always positive, roughly linear in log-x).
INPUT_TRANSFORM = InputTransform(
    forward_fns=[lambda x: np.log(x + 0.1), lambda T: (T - BOUNDS[1][0]) / 17.0],
    backward_fns=[lambda x: np.exp(x) - 0.1, lambda T: 17.0 * T + BOUNDS[1][0]],
)
OUTPUT_TRANSFORM = OutputTransform(forward_fn=np.log, backward_fn=np.exp)

N_INIT = 10          # matches PrOHwater(nObs=10, ...) for the manuscript run
N_ITERS = 5          # adaptive design iterations (paper ran 15; kept small here --
                     # full reproduction is Phase 7)
COLUMN_VAR_NAMES = ["xW", "F", "xF", "R", "xD"]
COLUMN_VAR_VALUES = [0.01, 100.0, 0.10, 1.0, 0.43]   # Geankoplis Ex. 11.4-1
COLUMN_N_STAGES = 4
COLUMN_FEED_STAGE = 3


def run(n_init=N_INIT, n_iters=N_ITERS, seed=SEED, z_grid_size=50):
    """Run the full pipeline; return the adaptive-design history and the column result."""
    X_init, _ = latin_hypercube_design(BOUNDS, n_train=n_init, n_test=0, seed=seed)
    y_init = np.array([activity_model.black_box(z, T)[0] for z, T in X_init])

    bfg = BitsForGaps(
        black_box=activity_model.black_box, bounds=BOUNDS, kernel=AnisotropicSE.paper_2d(),
        mean_fxn=gpflow.mean_functions.Zero(), likelihood_variance=0.1,
        input_transform=INPUT_TRANSFORM, output_transform=OUTPUT_TRANSFORM,
        iters=n_iters,
    )
    bfg.seed = seed

    print(f"Running {n_iters} adaptive design iterations from {n_init} initial points...")
    history = bfg.run(X_init, y_init)
    GPmodel = history.last.GPmodel
    print(f"Done. Final design has {history.last.XData.shape[0]} points.")

    print("Computing phase diagrams (Wilson ground truth + GP surrogate)...")
    z_grid = np.linspace(0.0, 1.0, z_grid_size)
    z_w, T_w, y_w = pd.vle_curve(pd.wilson_gamma, z_grid=z_grid)

    def surrogate_gamma_fn(z, T):
        return pd.surrogate_gamma(z, T, GPmodel, INPUT_TRANSFORM, OUTPUT_TRANSFORM)

    z_s, T_s, y_s = pd.vle_curve(surrogate_gamma_fn, z_grid=z_grid)

    print("Solving the distillation column (Wilson and surrogate equilibria)...")
    equil_wilson = equilibrium.make_equilibrium_function(z_w, y_w)
    equil_surrogate = equilibrium.make_equilibrium_function(z_s, y_s)
    column_wilson = distillation.solve_column(
        COLUMN_N_STAGES, COLUMN_FEED_STAGE, equil_wilson, COLUMN_VAR_NAMES,
        COLUMN_VAR_VALUES,
    )
    column_surrogate = distillation.solve_column(
        COLUMN_N_STAGES, COLUMN_FEED_STAGE, equil_surrogate, COLUMN_VAR_NAMES,
        COLUMN_VAR_VALUES,
    )

    if not column_wilson["converged"]:
        print(f"WARNING: Wilson column: {column_wilson['warnings']}")
    if not column_surrogate["converged"]:
        print(f"WARNING: surrogate column: {column_surrogate['warnings']} -- likely too "
             f"few design points for a well-behaved surrogate; try more n_init/n_iters.")

    print("\nStage table (liquid / vapor PrOH mole fraction):")
    print(f"{'stage':>5}  {'wilson liq':>10}  {'wilson vap':>10}  "
         f"{'surrogate liq':>13}  {'surrogate vap':>13}")
    for sw, ss in zip(column_wilson["stages"], column_surrogate["stages"]):
        print(f"{sw['stage']:>5}  {sw['liquid']:>10.4f}  {sw['vapor']:>10.4f}  "
             f"{ss['liquid']:>13.4f}  {ss['vapor']:>13.4f}")

    return {
        "history": history, "phase_diagram": {"wilson": (z_w, T_w, y_w),
                                              "surrogate": (z_s, T_s, y_s)},
        "column": {"wilson": column_wilson, "surrogate": column_surrogate},
    }


if __name__ == "__main__":
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    run()
