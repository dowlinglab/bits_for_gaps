# Synthetic example (no Julia)

A small, runnable sequential-design loop over a smooth 2-D closed-form function --
the Julia-free onboarding path for `bits_for_gaps`. If you want the paper's real
case study (Julia/Clapeyron-backed, heavier setup), see `examples/vle_distillation/`
instead.

## Setup

```bash
conda env create -f environment.yml   # or: conda activate bits_for_gaps if it exists already
conda activate bits_for_gaps
pip install -e ".[dev]"               # core + test tools -- no [vle] extra needed here
```

## Run it

```bash
python examples/synthetic/run_example.py
```

Runs in well under a minute and prints, for each of 5 adaptive design iterations, the point
BitsForGaps chose to evaluate next (`xStar`), the predictive entropy achieved there
(`max_entropy`), and the R-hat convergence diagnostic for each kernel hyperparameter
(should be close to 1).

See `docs/quickstart.md` for a narrated, piece-by-piece walkthrough of the same code,
and `docs/theory.md` for the method behind it.
