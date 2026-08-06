# Reproducing the paper's results

This page assumes you've read Jones & Dowling (2026) once and know the figures by number.
(The article is bundled in the repository at
[`paper/bits_for_gaps_paper.pdf`](https://github.com/dowlinglab/bits_for_gaps/blob/main/paper/bits_for_gaps_paper.pdf)
if you want it open alongside.) Everything you need is committed here — there are no
additional downloads.

There are two different things you might mean by "reproduce the paper," and they have very
different costs:

|  | Regenerate the figures | Re-run the full adaptive loop |
|---|---|---|
| Script | `paper/reproduce.py` | `paper/full_reproduction.py` |
| What it does | Renders the paper's *already-collected* data (HMC traces, GP posterior draws, phase-diagram data, ...) through this package's plotting code | Actually runs the 15-iteration adaptive HMC + entropy-acquisition design loop from scratch |
| Cost | ~1-2 minutes | ~25-30 minutes on a laptop |
| Determinism | Deterministic (same inputs, same plots) | **Stochastic** -- will not reproduce the paper's exact numbers |
| Needs Julia? | Only Fig 8/9 | Yes, throughout |

Both are **repo-only** (`git clone` required; see {doc}`installation` -- neither is
part of the `pip install bits_for_gaps` package).

## Regenerate the figures (fast, deterministic)

All 11 of the paper's figures (2 through 12) regenerate from data already committed at
`paper/data/` (~16 MB -- exactly the plot-input files the figures read; see
`paper/data/README.md`). No private-archive access, and no HMC sampling, is needed:

```bash
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes    # macOS; only Fig 8/9 touch Julia
python paper/reproduce.py
```

**Measured cost:** ~80 seconds wall-clock on a laptop (M2 MacBook Pro) for all 11
figures, including the 2 that call live Clapeyron (Fig 8/9). Order of a minute, not
tens of minutes -- this path does no MCMC.

Output goes to `results_remaked/` by default (gitignored -- nothing this produces is
committed). Regenerate a subset with `--figures`, e.g. `python paper/reproduce.py
--figures 5 8 9 10`. See `paper/REPRODUCTION.md` for the full figure -> script ->
data -> reference-diff table, which figures are quantitatively pinned against
`paper/reference/*` (5, 8, 9, 10, 11) versus visually reproduced only (2, 3, 4, 6, 7,
12), and known discrepancies (e.g. the reference stage table's ~0.01 transcription
slop).

Only Fig 8 and Fig 9 touch Julia (they recompute the Wilson ground-truth curve live via
Clapeyron.jl); the other 9 figures render from pure Python/NumPy and need neither Julia
nor the `[vle]` extra. If you only want those 9: `pip install -e ".[dev]"` is enough,
and you can skip the `PYTHON_JULIACALL_HANDLE_SIGNALS` export.

Everything these figures read is committed in `paper/data/`, so nothing above requires
extra downloads. If you re-run the loop yourself (next section), point
`--archive`/`$BFG_ARCHIVE_DIR` at your run's output directory to plot your results
instead of the published ones.

## Re-run the full adaptive loop from scratch (slow, stochastic)

`paper/reproduce.py` renders what the paper's 15-iteration adaptive HMC loop already
produced -- it does not re-run that loop. `paper/full_reproduction.py` does: it drives
`bits_for_gaps.sampler.BitsForGaps` through the same 15-iteration adaptive HMC +
entropy-acquisition design loop as the paper's published run (same bounds, seed,
transforms, kernel, and HMC config -- see the script's module docstring for the exact
constants), starting from a *fresh* Latin-hypercube design and a live Clapeyron/Wilson
black box:

```bash
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
python paper/full_reproduction.py --out-dir results_remaked/full_reproduction
```

**Documented cost: ~25-30 minutes on a laptop** (15 outer iterations, each running a
4-chain x 5000-sample HMC fit plus live Clapeyron calls for the training data and a
full-grid posterior-predictive diagnostic). This is a one-time validation exercise, not
part of the test suite or the CI-gated regression checks -- there's no need to run it
to use the package, reproduce a figure, or verify your installation.

```{caution}
**This run is stochastic and will NOT reproduce the paper's exact numbers.** HMC
sampling and the entropy-maximizing acquisition both involve randomness; a fresh run
explores a different (though statistically similar) sequence of design points and
posterior samples than the paper's published run did.
```

**What SHOULD match** (qualitatively, run to run and against the paper):
- HMC convergence: R-hat < 1.1 for all three kernel hyperparameters, with healthy
  effective sample size relative to the ~20,000 raw HMC samples per iteration.
- The posterior orders the two lengthscales the same way the paper reports: the
  temperature lengthscale is *larger* than the mole-fraction lengthscale (the GP trusts
  nearby-temperature extrapolation more than nearby-composition extrapolation).
- Maximum entropy decays over successive iterations (Fig 4's shape) as the design fills
  in the space and uncertainty drops.
- Test-set predictive error drops from iteration 1 to iteration 15 (though not
  necessarily monotonically in between -- entropy-driven acquisition optimizes
  information gain, not held-out error, at each step).
- The surrogate's phase diagram and McCabe-Thiele stage table agree with the Wilson
  ground truth within a few 0.01 in mole fraction, the same way the paper's surrogate
  does.

**What will legitimately differ** from the paper and between your own runs:
- The exact sequence of sampled design points and their coordinates.
- Exact posterior samples, R-hat/ESS values, and entropy values (though the same order
  of magnitude and trend).
- The exact test-RMSE trajectory -- `GPmodel.predict_f_samples` (used for
  posterior-predictive draws, not for the HMC posterior itself) draws from
  TensorFlow's ambient RNG, which this package cannot seed bitwise (see
  `bits_for_gaps.mixture`'s module docstring); this was already true of the paper's
  own original code.
- Small (sub-0.05 mole fraction) shifts in the surrogate's phase diagram / stage table
  relative to the paper's exact reported numbers.

A completed run's numbers, compared directly against the paper's, are recorded in
`paper/REPRODUCTION.md`'s "Re-running the full adaptive loop from scratch" section --
useful as a sanity check for what "qualitatively similar" looks like in practice, if you
run it yourself and want something to compare against.

## The regression tests

Regression tests that check figure/column reproduction against `paper/reference/*`
(`tests/regression/test_paper_figures.py`, `tests/regression/test_mccabe_thiele.py`)
mostly run in the **default** `pytest -q` suite -- only the tests that recompute
something via live Clapeyron calls (needing Julia) stay behind `@pytest.mark.vle` (run
with `pytest -m vle`). You can also read `paper/figures/*.py` to see exactly how each
figure is built, and run `examples/vle_distillation/run_case_study.py`
({doc}`vle_example`) for a small, fast, fresh (non-archived) demonstration of the same
underlying pipeline at a shorter iteration count.
