# Reproducing the paper's figures

`paper/reproduce.py` regenerates the paper's figures (Jones & Dowling 2026, "BITS for
GAPS") from the published run's plot-input data (iteration 15 of the paper's original
run). It does **not** re-run the paper's 15-iteration adaptive HMC loop (stochastic,
expensive, and not what reproducing a figure requires) -- it loads what that loop
already produced and renders it through the `bits_for_gaps` package +
`examples/vle_distillation`.

**Everything needed is committed here** -- the data comes from `paper/data/` (~16 MB;
see `paper/data/README.md`), so this works from a fresh clone. Point
`--archive`/`$BFG_ARCHIVE_DIR` at another directory of run artifacts (same layout) to plot
a different run, such as one produced by `paper/full_reproduction.py`.

```bash
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes    # macOS; needed for Fig 8/9 (Clapeyron)
python paper/reproduce.py                      # uses the committed paper/data/
python paper/reproduce.py --archive /path/to/less_x_new_manuscript_revisions
python paper/reproduce.py --figures 5 8 9 10   # regenerate a subset
```

Output goes to `--out-dir` (default `results_remaked/`, gitignored) -- nothing this
script produces is committed.

For the from-scratch stochastic reproduction of the full adaptive loop (a separate,
one-time validation exercise, not part of `paper/reproduce.py`), see
["Re-running the full adaptive loop from scratch"](#re-running-the-full-adaptive-loop-from-scratch)
below.

## Figure map

| Fig | Content | Script (`paper/figures/`) | Data inputs (all from committed `paper/data/`) | Reference diff |
|---|---|---|---|---|
| 2 | Initial LHS design (train/test) | `fig02_lhs_design.py` | `lhs_design`, `lhs_test_points` | none -- visual |
| 3 | Entropy field, early iterations | `fig03_entropy_field.py` | `entropy_{1..6}`, `activity_data_{1..7}` | none -- visual |
| 4 | Max entropy vs. iteration | `fig04_entropy_evolution.py` | `entropy_{1..60}` | none -- visual |
| 5 | Parity + RMSE/MAE box plots | `fig05_parity.py` | `activity_data_1`, `activity_test_points`, `gp_predict_{train,test}_{1,15}` | **`fig5_error_metrics.json`** (`test_paper_figures.py`, default suite) |
| 6 | GP posterior 3-D surface, iter 1 vs. 15 | `fig06_gp_posterior_surface.py` | `activity_data_{1,15}`, `gp_predict_{1,15}` | none -- visual |
| 7 | GP posterior at 4 isotherms | `fig07_gp_posterior_isotherms.py` | `gp_predict_15`, `cont_data` | none -- visual |
| 8 | T-x-y phase diagram | `fig08_phase_diagram.py` | `phase_diagram_15` (surrogate ensemble); Wilson curve is **freshly recomputed** via live Clapeyron | `gt_Wilson_data` cross-check (`test_paper_figures.py`, **`@pytest.mark.vle`** -- needs Julia) |
| 9 | McCabe-Thiele column (Wilson vs. surrogate) | `fig09_mccabe_thiele.py` | none -- pure physics recompute (Wilson via live Clapeyron; surrogate via a fresh 30-pt LHS + MLE-fit GP) | **`mccabe_thiele_stages.json`** (`test_mccabe_thiele.py`, **`@pytest.mark.vle`** -- needs Julia) |
| 10 | HMC traces + R-hat/ESS | `fig10_traces.py` | `traces_chain_{0..3}_exp_15`, `rhat_value_15.txt`, `ess_value_15.txt` | **`hmc_diagnostics.json`** (`test_paper_figures.py`, default suite) |
| 11 | Hyperparameter marginals | `fig11_marginals.py` | `param_posterior_samples_15` | **`hyperparameter_posterior.json`** (`test_paper_figures.py`, default suite) |
| 12 | Hyperparameter joint marginals | `fig12_joint_marginals.py` | `param_posterior_samples_15` | none -- visual (same posterior samples Fig 11's numbers are pinned from) |

**Quantitatively pinned** (5, 8, 9, 10, 11): a committed reference JSON or a
`paper/data/`-vs-recompute cross-check is diffed within a stated tolerance. Only the
tests that *recompute* something via Clapeyron stay gated (`@pytest.mark.vle`,
deselected by default -- run with `pytest -m vle`, needs Julia): Fig 8's Wilson-curve
cross-check and Fig 9's full stage-table recompute. Fig 5/10/11's tests only *read* the
committed `paper/data/` text files (no Julia needed) and run in the
**default** `pytest -q` suite. Fig 8 has no dedicated reference *file* (it's a visual
reproduction, not a paper-reported scalar) -- its pin is a direct cross-check against
the committed `gt_Wilson_data` instead. `BFG_ARCHIVE_DIR` can point any of these at a
different run's artifacts instead of `paper/data/`, but that's not required for the
default-suite tests to run.

**Visually reproduced** (2, 3, 4, 6, 7, 12): no reference file exists for these (the
paper doesn't report them as scalars), so they're spot-checked by eye against the
archived PNGs' structure -- correct qualitative behavior (LHS space-filling, entropy
concentrating away from sampled points, CI narrowing with more data, etc.), not
pixel-identical figures.

## Simplifications from the original plotting code

This is reproduction code (`paper/figures/`), ported from the paper's original
plotting scripts pragmatically, not verbatim -- it is not a library API. Dropped,
since they're purely visual and don't change the figure's content or claim:

- **Fig 5**: the zoomed inset panel (the original's zoom into the low-error region).
  The main parity plot + error bars are unchanged.
- **Fig 12**: the KDE contour overlay on each hexbin panel. Kept the hexbin density,
  the MAP point, and the 95%-credible-interval box; the MAP point here is the sample
  nearest the coordinate-wise median (a cheap proxy), not a true density mode --
  fine for a visual reproduction, not something to treat as a precise point estimate
  (use `hyperparameter_posterior.json`'s `mean`/`median` for that).
- **Fig 3**: shown as a 2x3 grid of the first 6 iterations rather than one file per
  iteration (the archive has 60) -- the paper's own lettered-panel (a)-(f) scheme
  already implies a small multi-panel figure, not 60 separate files.
- **Fig 9**: the surrogate panel uses a freshly-trained (30-point LHS + MLE fit) GP,
  not the paper's actual 15-iteration adaptively-designed surrogate -- see
  `tests/regression/test_mccabe_thiele.py`; reproducing that surrogate exactly would
  mean re-running the full stochastic adaptive loop (see below), which this default
  figure-regeneration path deliberately does not do. The Wilson panel is an exact
  physics recompute either way.

## Known discrepancies (not bugs -- documented, not "fixed")

- **`paper/reference/mccabe_thiele_stages.json`'s `"wilson"` column has ~0.01-level
  transcription slop** (it was hand-read off paper Fig 9c, not computed from
  archived data -- see `paper/reference/README.md`). A fresh Clapeyron recompute in
  `fig09_mccabe_thiele.wilson_column()` hits real physical landmarks exactly (e.g.
  stage 1 vapor = `xD` = 0.43 by construction; pure-PrOH bubble point 370.35 K
  matches 1-propanol's real normal boiling point to 4 significant figures) yet
  differs from reference's `"wilson"` entries by up to 0.01 -- this is the reference
  file's own transcription precision limit, not a port error.

## Re-running the full adaptive loop from scratch

Everything above regenerates figures from the published run's *plot-input data* --
it never re-executes the paper's sequential-design loop. This section does: it runs
`paper/full_reproduction.py`, which drives `bits_for_gaps.sampler.BitsForGaps` through
the same 15-iteration adaptive HMC + entropy-acquisition loop as the paper's published
run (same bounds, seed, transforms, kernel, and HMC config -- see the script's module
docstring for the exact constants), starting from a *fresh* 10-train/10-test LHS
design and a live Clapeyron/Wilson black box -- no archived data read as input
anywhere in this section.

```bash
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
python paper/full_reproduction.py --out-dir results_remaked/full_reproduction   # ~25 min
```

This is a **one-time validation exercise**, not a regression test -- there is no
gated CI test for it (re-running it is expensive and its whole point is to check
statistical/qualitative, not bitwise, agreement). The numbers below are from a
completed run (artifacts stayed in the gitignored `results_remaked/`, as always);
they're recorded here rather than left as an unverifiable claim.

**Headline result: closer to bitwise than expected.** Everything in the loop that
runs through a `self.seed`-seeded path (LHS design, HMC sampling, the entropy-
acquisition optimizer) reproduced the published run to 6-8 significant figures --
not just "qualitatively similar." The only place real stochastic drift shows up is
the GP posterior-*predictive* mixture sampling (`gpflow`'s `predict_f_samples`, which
draws from TensorFlow's ambient RNG -- documented as non-reproducible in
`bits_for_gaps/mixture.py`'s module docstring, and non-reproducible in the paper's
original code too). That function feeds exactly two things: the test-RMSE curve
below, and the Fig-8-style surrogate phase diagram. Everything else pinned on the HMC
trace directly (R-hat/ESS, hyperparameter posterior, entropy field) lines up almost
exactly.

### HMC diagnostics (iteration 15, trained on the same 24 points as `gp_model_15`)

| | R-hat | ESS |
|---|---|---|
| Paper (`paper/reference/hmc_diagnostics.json`) | 1.0052, 1.0073, 1.0088 | 1468.3, 2428.1, 653.1 |
| Fresh run | 1.00523, 1.00730, 1.00879 | 1468.29, 2428.09, 653.15 |

All well under the R-hat < 1.1 convergence threshold, and ESS is healthy relative to
20,000 raw HMC samples (4 chains x 5000). The near-exact match here is a strong
confirmation that this package's decomposition of the original algorithm preserves
the HMC path bit-for-bit -- this isn't the paper's own number re-displayed, it's an
independently re-run HMC fit that happens to land on the same posterior.

### Hyperparameter posterior (`std_dev`, `lengthscale_1`, `lengthscale_2`)

| | mean | median |
|---|---|---|
| Paper (`paper/reference/hyperparameter_posterior.json`) | 1.35645, 0.86239, 3.19502 | 1.28619, 0.81949, 3.04407 |
| Fresh run | 1.35645, 0.86239, 3.19502 | 1.28619, 0.81949, 3.04407 |

Same ordering the paper reports in Fig 11: `lengthscale_2` (temperature) >
`lengthscale_1` (composition) -- the GP trusts nearby-temperature extrapolation more
than nearby-composition extrapolation, consistent with activity coefficients varying
more sharply in composition than in temperature over this range.

### Entropy decay (Fig 4 shape)

Per-iteration max entropy (grid max of `entropy_{i}`, both committed and fresh):

| Iteration | 1 | 2 | 3 | 10 | 14 | 15 |
|---|---|---|---|---|---|---|
| Paper (`paper/data/entropy_*`) | 1.4577 | 1.2470 | 0.6505 | 0.0215 | 0.0038 | -0.0084 |
| Fresh run | 1.4577 | 1.2470 | 0.6505 | 0.0215 | 0.0038 | -0.0084 |

Same monotonic-ish decay, same sign change (uncertainty-driven exploration
"exhausts" the space) landing between iterations 14 and 15 in both runs.

### Predictive accuracy (test RMSE, activity coefficient units)

| | iteration 1 | iteration 15 |
|---|---|---|
| Paper (paper's reported headline number) | ~4.34 | ~0.67 |
| Fresh run | 4.337 | 0.887 |

Iteration 1 matches almost exactly (same initial design, same black box -- no HMC
posterior involved yet beyond the seeded trace). Iteration 15 lands in the same
regime (~5x reduction vs. the paper's ~6.5x) but not on the same value -- expected,
since this metric is the one place `predict_f_samples`'s non-reproducible draws
enter, compounded by a documented simplification: this run's mixture uses a
15-component hyperparameter-posterior subset (`noGaussians`, the same size the
acquisition function itself uses) rather than the paper's own dedicated 500-sample
subset for this plot. The trend is non-monotonic in both runs (e.g. this run's RMSE
ticks up at iterations 8 and 12 before falling again) -- expected for an
entropy-driven acquisition, which optimizes information gain, not held-out error, at
each step.

### Phase diagram (Fig 8) and McCabe-Thiele stage table (Fig 9)

The freshly recomputed Wilson (ground-truth) curve matches the archived
`gt_Wilson_data` to within 0.20 K / 0.007 mole fraction (well inside the existing
gated test's 0.5 K / 0.02 tolerance) -- expected, since this is a deterministic
Clapeyron computation with no randomness anywhere in it.

The **genuinely** 15-iteration-adaptive surrogate GP (this run's final, 24-point
model -- not `fig09_mccabe_thiele.py`'s dedicated 30-point-LHS/MLE-fit stand-in) gives
a phase diagram within 0.83 K / 0.02 mole fraction of the Wilson curve -- a good
surrogate, and its McCabe-Thiele column converges and tracks the Wilson column
closely -- within 0.03 mole fraction (liquid) / 0.014 (vapor) at every stage.
`mixture.sample_gp_posterior_mixture` reassigns a GP's kernel hyperparameters once per
posterior draw during sampling (see its docstring); building this phase diagram from
`history.last.GPmodel` must happen before any test-RMSE-style step that calls that
function on the same model object, or the kernel is left at an arbitrary leftover
hyperparameter state. `paper/full_reproduction.py` orders its calls accordingly and
uses `surrogate_gamma_averaged` (a posterior-hyperparameter-averaged, rather than
single-draw, activity-coefficient estimate) for the phase diagram specifically because
that ordering matters.

### Summary

Qualitative/statistical agreement confirmed on every axis this exercise checked, and
considerably tighter than "qualitative" on the deterministic parts of the pipeline
(HMC diagnostics, hyperparameter posterior, entropy field). Reproduce via
`paper/full_reproduction.py` (module docstring has the exact constants used); not
gated in CI (see above).
