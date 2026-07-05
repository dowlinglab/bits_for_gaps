# Quickstart

A minimal, pure-Python example -- no Julia, no archived data, runs in well under a
minute on a laptop. It sequentially designs experiments for a synthetic 2-D function
using `BitsForGaps`, the public entry point (a thin, friendlier-named wrapper around
the `adaptiveEntropy` sequential-design engine; see {doc}`api`).

```{note}
This page shows the code as a reference, without running it during the docs build
(HMC sampling is too slow for a docs build and doesn't belong in CI for a docs job).
A runnable version of exactly this example lives in
[`examples/synthetic/run_example.py`](https://github.com/dowlinglab/bits_for_gaps/blob/main/examples/synthetic/run_example.py) --
no Julia, runs in well under a minute:

```bash
python examples/synthetic/run_example.py
```

See `examples/synthetic/README.md` for setup. (`tests/integration/test_nd_synthetic.py`
and `test_end_to_end.py` exercise the same pipeline too, if you want to see it under
test rather than as a standalone script.)
```

## The pieces

```python
import numpy as np

from bits_for_gaps import AnisotropicSE, BitsForGaps

# The "black box" you're designing experiments for. BitsForGaps calls it as
# black_box(*xStar) -- xStar's components in the same order as `bounds` below --
# and expects an iterable of length 1 (the observed y) back.
def black_box(x1, x2):
    return [np.sin(3.0 * x1) + np.cos(3.0 * x2)]

BOUNDS = [(0.0, 1.0), (0.0, 1.0)]   # one (low, high) tuple per input dimension
```

## Seed an initial design

Any space-filling design works; `bits_for_gaps.design` provides a seeded
Latin-hypercube helper:

```python
from bits_for_gaps import latin_hypercube_design

X_init, _ = latin_hypercube_design(BOUNDS, n_train=10, n_test=0, seed=0)
y_init = np.array([black_box(x1, x2)[0] for x1, x2 in X_init])
```

## Build the model and run

```python
bfg = BitsForGaps(
    black_box=black_box,
    bounds=BOUNDS,
    kernel=AnisotropicSE(),          # the paper's 2-D prior config (see below)
    likelihood_variance=0.05,
)

# HMC/acquisition tuning knobs (defaults are set for the paper's problem size;
# a synthetic 2-D function like this one is comfortable with far fewer samples).
bfg.noSamples = 500
bfg.noBurnIn = 200
bfg.noChains = 2

history = bfg.run(X_init, y_init)   # runs bfg.noIters=1 sequential-design iteration
```

`run` returns a {class}`~bits_for_gaps.state.RunHistory`: one
{class}`~bits_for_gaps.state.IterationRecord` per iteration, holding the fitted GP,
the HMC trace, R-hat/ESS convergence diagnostics, and the newly selected point:

```python
record = history.last
print(record.xStar)         # the next point BitsForGaps chose to evaluate
print(record.max_entropy)   # the predictive entropy achieved there
print(record.rhat)          # R-hat per kernel hyperparameter -- check these are ~1
```

To run more than one sequential-design iteration, pass `iters=N` to `BitsForGaps(...)`
(or set `bfg.noIters = N`) and call `bfg.run(X_init, y_init)` once -- each iteration's
selected point is folded back into the design automatically.

## About that kernel

`AnisotropicSE()` with no arguments reproduces the paper's exact 2-D VLE kernel
configuration -- per-dimension lognormal/gamma priors on the lengthscales, a
lognormal prior on the output scale (equivalently, `AnisotropicSE.paper_2d()`). For
your own problem, specify priors explicitly, one per input dimension:

```python
import tensorflow as tf
import tensorflow_probability as tfp

kernel = AnisotropicSE(
    variance_prior=tfp.distributions.LogNormal(loc=tf.math.log(1.0), scale=1.0),
    lengthscale_priors=[
        tfp.distributions.LogNormal(loc=tf.math.log(0.3), scale=0.5),
        tfp.distributions.Gamma(concentration=2.0, rate=2.0),
        tfp.distributions.LogNormal(loc=tf.math.log(0.3), scale=0.5),
    ],   # one entry per input dimension -- this example is 3-D
)
```

Each lengthscale (and the output scale) is its own `gpflow.Parameter`, so different
input dimensions can carry entirely different prior *families*, not just different
prior parameters -- see {doc}`theory` and the `AnisotropicSE` API reference for why
that matters. `bounds` (and therefore the kernel's `lengthscale_priors`) can have any
number of entries -- the whole pipeline is dimension-general, proven in
`tests/integration/test_nd_synthetic.py` at 1-D and 3-D.
