"""Synthetic, pure-Python BitsForGaps example -- no Julia, runs in well under a minute
on a laptop.

The same pattern `docs/quickstart.md` walks through, made runnable: a small sequential
design loop over a smooth 2-D closed-form function, using the public `BitsForGaps` API.
This is the Julia-free onboarding path -- `examples/vle_distillation/` is the paper's
real (Julia/Clapeyron-backed) case study, which needs a much heavier setup.

Usage::

    python examples/synthetic/run_example.py
"""

import numpy as np

from bits_for_gaps import AnisotropicSE, BitsForGaps, latin_hypercube_design

BOUNDS = [(0.0, 1.0), (0.0, 1.0)]  # one (low, high) tuple per input dimension
SEED = 0
N_INIT = 10
N_ITERS = 5


def black_box(x1, x2):
    """The function BitsForGaps is designing experiments for.

    Any closed-form or expensive-to-evaluate function works here -- only the
    ``(x1, x2) -> [y]`` calling convention matters (called as ``black_box(*xStar)``,
    expected to return a length-1 sequence).
    """
    return [float(np.sin(3.0 * x1) + np.cos(3.0 * x2))]


def run(n_init=N_INIT, n_iters=N_ITERS, seed=SEED):
    """Run the sequential design loop; return the full iteration history."""
    X_init, _ = latin_hypercube_design(BOUNDS, n_train=n_init, n_test=0, seed=seed)
    y_init = np.array([black_box(x1, x2)[0] for x1, x2 in X_init])

    bfg = BitsForGaps(
        black_box=black_box,
        bounds=BOUNDS,
        kernel=AnisotropicSE(),  # the paper's 2-D prior config -- see docs/theory.md
        likelihood_variance=0.05,
        iters=n_iters,
    )
    bfg.seed = seed
    # Small, fast HMC + acquisition config: the paper's defaults (noSamples=5000,
    # noChains=4, noGaussians=25, noRestarts=10, entropyMesh=[10,10], ...) are sized
    # for its harder, noisier problem -- a smooth synthetic function like this one
    # converges comfortably with far fewer samples/restarts/mixture components, and
    # runs in seconds instead of minutes.
    bfg.noSamples = 200
    bfg.noBurnIn = 100
    bfg.noChains = 2
    bfg.noGaussians = 8
    bfg.noRestarts = 3
    bfg.entropyMesh = [4, 4]
    # The paper's default acquisition objective is the 2nd-order Taylor entropy
    # approximation. Uncomment to try the alternative closed-form lower bound instead
    # (see docs/theory.md's "Entropy estimators" section):
    # bfg.acquisitionObjective = "lower_bound"

    print(f"Running {n_iters} adaptive design iterations from {n_init} initial points...")
    history = bfg.run(X_init, y_init)

    for record in history:
        print(
            f"  iter {record.iteration}: xStar={np.round(record.xStar, 4)}, "
            f"max_entropy={record.max_entropy:.4f}, rhat={np.round(record.rhat, 3)}"
        )

    print(f"\nDone. Final design has {history.last.XData.shape[0]} points.")
    return history


if __name__ == "__main__":
    run()
