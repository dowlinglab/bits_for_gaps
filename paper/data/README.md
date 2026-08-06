# Curated plot-input subset (tracked)

**This directory IS tracked in git** (unlike `results_remaked/`, which is gitignored
output). It's a small, curated subset of the private archive's published run --
exactly the files `paper/figures/*.py` read, no more -- committed so that
`python paper/reproduce.py` works from a fresh clone with **no private-archive
access**.

## Provenance

Copied verbatim (unmodified) from the private old repo's archive of record:

```
~/DowlingLab/CAREER/entropy_driven_hybrid_models_code/entropy_driven_hms/
  results/less_x_new_manuscript_revisions/
```

**Iteration 15** = the published run (its `rhat_value_15.txt`/`ess_value_15.txt`
match paper Fig 10 exactly -- see `paper/reference/hmc_diagnostics.json`). See
`paper/DATA.md` for how this relates to the bulk (~2.5 GB, not committed anywhere)
archive.

## How this set was determined

Enumerated directly from `paper/figures/_archive.py`'s loader calls and each
`paper/figures/fig*.py`'s `make()` (not guessed/over-included) -- see
`paper/REPRODUCTION.md`'s figure table for the file -> figure mapping. Total: 87
files, ~16 MB.

| Files | Why |
|---|---|
| `lhs_design`, `lhs_test_points` | Fig 2 |
| `entropy_1` .. `entropy_60` (all) | Fig 3 (first 6), Fig 4 (all -- the whole evolution curve) |
| `activity_data_1` .. `activity_data_7` | Fig 3 (`it` and `it+1` for `it` in 1..6) |
| `activity_data_15` | Fig 6 |
| `activity_test_points` | Fig 5 |
| `gp_predict_train_{1,15}`, `gp_predict_test_{1,15}` | Fig 5 |
| `gp_predict_1`, `gp_predict_15` | Fig 6, Fig 7 (the bulk of this directory's size -- 6.1 MB each, full 50x50-grid x 100-draw text dumps) |
| `cont_data` | Fig 7 |
| `phase_diagram_15` | Fig 8 (the surrogate-ensemble panel; the Wilson ground-truth curve is recomputed fresh via live Clapeyron, not read from here) |
| `gt_Wilson_data` | Fig 8's regression cross-check only (`tests/regression/test_paper_figures.py`), not the figure itself |
| `traces_chain_{0,1,2,3}_exp_15`, `rhat_value_15.txt`, `ess_value_15.txt` | Fig 10 |
| `param_posterior_samples_15` | Fig 11, Fig 12 |

Fig 9 (McCabe-Thiele) needs none of this -- it's a pure physics recompute (Wilson via
live Clapeyron; the surrogate via a fresh LHS + MLE-fit GP, same as
`tests/regression/test_mccabe_thiele.py`).

## Why plain text, not compressed or `gp_model_*.pkl`-recomputed

`gp_predict_1`/`gp_predict_15` (6.1 MB each, ASCII `np.savetxt` dumps) are the bulk of
this directory. Two alternatives were considered and rejected for now:

- **Recompute from the archived `gp_model_{1,15}.pkl`** (the fitted GPR objects) at
  reproduce-time instead of committing the pre-computed draws. This would need
  `mixture.sample_gp_posterior_mixture`, which is **not bitwise-reproducible even in
  the original paper code** (`GPmodel.predict_f_samples` draws from TensorFlow's
  ambient, unseeded RNG -- see `mixture.py`'s module docstring) -- so it wouldn't
  save the committed data faithfully, it would produce a *different* (though
  qualitatively similar) draw every time. Not done.
- **Compress the text files** (e.g. gzip) and decompress at load time. Would roughly
  halve this directory's size. Not done because 16 MB total is already comfortably
  inside the ~30-50 MB budget this step was scoped to -- not worth the added
  `_archive.py` loader complexity (every `np.loadtxt` call would need a
  compression-aware wrapper) for a problem that doesn't exist yet. Revisit if this
  directory grows substantially (e.g. if a future figure needs `gp_predict` at more
  iterations).

## Do not edit by hand

These are frozen snapshots of a specific archived run. If the curated set ever needs
to change (a new figure needs a new file, or an existing one needs a different
iteration), copy the new file(s) from the private archive the same way -- don't
regenerate or edit any file here.
