# Reproducing the paper's figures

`paper/reproduce.py` regenerates the paper's figures (Jones & Dowling 2026, "BITS for
GAPS") from the published run's plot-input data (iteration 15 of
`less_x_new_manuscript_revisions`). It does **not** re-run the paper's 15-iteration
adaptive HMC loop (stochastic, expensive, and not what reproducing a figure requires)
-- it loads what that loop already produced and renders it through the refactored
`bits_for_gaps` package + `examples/vle_distillation`.

**As of Phase 9, this needs no private-archive access by default** -- the data comes
from the curated, committed `paper/data/` subset (~16 MB; see `paper/data/README.md`).
Point `--archive`/`$BFG_ARCHIVE_DIR` at the full private archive only if you want
figures at iterations beyond that curated subset (see `paper/DATA.md`).

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
["Phase 9: from-scratch stochastic reproduction"](#phase-9-from-scratch-stochastic-reproduction)
below.

## Figure map

| Fig | Content | Script (`paper/figures/`) | Data inputs (all from committed `paper/data/`) | Golden diff |
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

**Quantitatively pinned** (5, 8, 9, 10, 11): a committed golden JSON or a
`paper/data/`-vs-recompute cross-check is diffed within a stated tolerance.
**As of Phase 9, only the tests that *recompute* something via Clapeyron stay
gated** (`@pytest.mark.vle`, deselected by default -- run with `pytest -m vle`,
needs Julia): Fig 8's Wilson-curve cross-check and Fig 9's full stage-table
recompute. Fig 5/10/11's tests only *read* the committed `paper/data/` text files
(no Julia, no private archive) and now run in the **default** `pytest -q` suite.
Fig 8 has no dedicated golden *file* (it's a visual reproduction, not a
paper-reported scalar) -- its pin is a direct cross-check against the committed
`gt_Wilson_data` instead. `BFG_ARCHIVE_DIR` can still point any of these at the full
private archive instead of `paper/data/` (e.g. to check other iterations), but
that's no longer required for the default-suite tests to run.

**Visually reproduced** (2, 3, 4, 6, 7, 12): no golden file exists for these (the
paper doesn't report them as scalars), so they're spot-checked by eye against the
archived PNGs' structure during development -- correct qualitative behavior (LHS
space-filling, entropy concentrating away from sampled points, CI narrowing with more
data, etc.), not pixel-identical figures.

## Simplifications from the original plotting code

This is reproduction code (`paper/figures/`), not a library API -- ported from
`fxns/mcmc_plotter.py` (847 lines) pragmatically, not verbatim. Dropped, since they're
purely visual and don't change the figure's content or claim:

- **Fig 5**: the zoomed inset panel (original `plot_parity`'s `inset_axes` zoom into
  the low-error region). The main parity plot + error bars are unchanged.
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
  `tests/regression/test_mccabe_thiele.py` and `HANDOFF.md` (Phase 6) for why
  reproducing that exactly is Phase 7 work that was deliberately *not* undertaken
  (it would mean re-running the stochastic adaptive loop, against this phase's
  guardrail). The Wilson panel is an exact physics recompute either way.

## Known discrepancies (not bugs -- documented, not "fixed")

- **`paper/golden/mccabe_thiele_stages.json`'s `"wilson"` column has ~0.01-level
  transcription slop** (it was hand-read off paper Fig 9c, not computed from
  archived data -- see `paper/golden/README.md`). The fresh Clapeyron recompute in
  `fig09_mccabe_thiele.wilson_column()` hits real physical landmarks exactly (e.g.
  stage 1 vapor = `xD` = 0.43 by construction; pure-PrOH bubble point 370.35 K
  matches 1-propanol's real normal boiling point to 4 significant figures) yet
  differs from golden's `"wilson"` entries by up to 0.01 -- this is the golden
  file's own transcription precision limit, not a port error (see `HANDOFF.md`
  Phase 6 for the full analysis).
