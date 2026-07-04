# HANDOFF — bits_for_gaps

State of the fresh `bits_for_gaps` repo. Read this + `REFACTOR_PLAN.md` before continuing.

- Bootstrap (Phase 0 + Phase 3-lite): Opus, 2026-07-03.
- **Phase 2 (regression/test harness): done 2026-07-03 on branch `phase2-regression-harness`
  — awaiting review/merge to `main` before Phase 4 begins.**

## Phase 2 — regression/test harness (done; review gate before Phase 4)

Behavior is now pinned BEFORE any refactor. `pytest -q` = **56 passed, 1 deselected**
(the deselected one is the `vle` McCabe-Thiele recompute; the pure-Python core needs no
Julia). Full suite runs in ~16 s.

```
paper/golden/            golden scalars from the archived iter-15 published run (+ README)
  hmc_diagnostics.json         R-hat / ESS            (Fig 10)
  fig5_error_metrics.json      train/test RMSE & MAE, iters 1 & 15, over 500 draws (Fig 5)
  hyperparameter_posterior.json  kernel-hyperparam posterior summary (Fig 10 marginals)
  mccabe_thiele_stages.json    distillation stage table, Wilson vs surrogate (Fig 9c)
tests/conftest.py        `golden` loader fixture (resolves paper/golden/)
tests/unit/              + test_kernels.py, test_means.py, test_util.py  (was entropy/design)
tests/regression/        reads golden, pins vs published paper values; vle recompute gated
tests/integration/       test_end_to_end.py — seeded synthetic (no-Julia) adaptiveEntropy run
```

Key facts for the next session:
- **Golden extraction** was done offline by `scratchpad/extract_golden.py` (pure NumPy,
  reads the read-only old-repo archive). All but the McCabe-Thiele table came from
  archived data; the stage table is transcribed from paper Fig 9c (its recompute needs
  the Phase-6 VLE backend → `@pytest.mark.vle`, deselected by default via `addopts`).
- **The sampler is deterministic** for a fixed seed in-process (verified bitwise: R-hat,
  ESS, entropy field, selected point all diff = 0.0 across two runs). The integration
  test asserts this with `atol=1e-10` — it is the guard against nondeterminism creeping
  into the Phase 4 decomposition.
- The integration test **mirrors `run_model` but skips the plotting-only `gp_predict_2D`**
  (~20 s; 100 full-cov draws over a 50×50 grid). It does not feed the acquisition
  (`entropy_objective` re-seeds NumPy and re-assigns every kernel param before each
  deterministic `predict_f`), so entropy/next-point are unchanged by skipping it.
- Fig 5 golden captures the paper's headline: median **test** RMSE falls 4.34 → 0.67
  (iter 1 → 15); train RMSE 0.77 → 0.49. The regression test pins the ≥3× test-error drop.
- Markers registered in `pyproject.toml`: `vle` (Julia backend, deselected by default),
  `slow` (integration; still runs by default). Run gated tests with `pytest -m vle`.

## What exists now (Phase 0 + Phase 3-lite done)

A pip-installable package scaffold with the **clean core pieces moved in and tested**:

```
src/bits_for_gaps/
  __init__.py    public API; pure pieces eager, TF-backed pieces lazy (PEP 562)
  entropy.py     GMM density + 1st/2nd-order Taylor entropy + closed-form lower bound
                 (from fxns/max_ent_design.py; dead commented variants removed). PURE.
  design.py      latin_hypercube_design / full_factorial_design, N-D, pure, no disk I/O
                 (extracted from proh_water_class). PURE.
  kernels.py     AnisotropicSE (2-D, prior-bearing) from fxns/my_kermel_fxn.py
  means.py       FixedInverseMean from fxns/my_mean_fxn.py
  _util.py       standardize / normalize / make_tensor
  sampler.py     adaptiveEntropy engine from driver_new.py, DECOUPLED from the example
                 (no module-level juliacall / proh_water_class; example helpers removed;
                  `i += 50` -> `self.startIter`). Body otherwise faithful to the paper.
tests/unit/      test_entropy.py (analytic + regression pin), test_design.py  (+ Phase 2 tests)
```

`BitsForGaps` is exported as an alias of `adaptiveEntropy` (target public name).

## Environment (verified working)

```bash
conda env create -f environment.yml        # or reuse existing `bits_for_gaps` env
conda activate bits_for_gaps               # /opt/anaconda3/envs/bits_for_gaps
pip install -e ".[dev]"                     # add ,vle for the Julia example
pytest -q                                   # 14 pass
```
Stack: Python 3.9.23, gpflow 2.9.2, TF 2.16.2, TFP 0.24.0, numpy 1.26.4, scipy 1.13.1.
**macOS: `export PYTHON_JULIACALL_HANDLE_SIGNALS=yes`** before any juliacall import
(else SIGBUS). Run scripts that use bare imports with `PYTHONPATH` set to the dir.

## Where the original code + archived results live

Old repo: `~/DowlingLab/CAREER/entropy_driven_hybrid_models_code/entropy_driven_hms/`.
- Published run = `results/less_x_new_manuscript_revisions/`, **iteration 15** (R̂/ESS
  match paper Fig 10 exactly). Archived figure PNGs + data are all there.
- `driver_new.py` = active driver (NOT `driver.py`). `new_phase_diagram.py` = Fig 8.
  `run_example.py` = Fig 9. `train_test_split_proh.py` = Fig 5. `fxns/mcmc_plotter.py`
  + `fxns/plot_res.py` = figure library/CLI.
- Repro-fix already applied there: `equilibrium.water_proh_eqm_julia` now reads
  `gt_Wilson_data` (original path was missing).
- Do NOT copy `results/` here (2.5 GB); it will go to Zenodo (see paper/DATA.md TODO).

## Next steps (in order — see REFACTOR_PLAN.md phases)

1. **Phase 2 — regression harness FIRST (before refactoring sampler.py). ✅ DONE.**
   See the "Phase 2" section above. On branch `phase2-regression-harness`; merge to
   `main` after review, then start Phase 4. The green suite is the safety net.

2. **Phase 4 — decompose `sampler.py`** into gp / mixture / acquisition / state modules;
   replace disk-as-state with in-memory state + optional checkpointing. Keep tests green.

3. **Phase 5 — generalize to N-D.** Remove the 2-D / 3-hyperparameter hardcoding flagged
   by `TODO(Phase 5)` in kernels.py and sampler.py (run_mcmc trainable_parameters[2,0,1];
   by-name kernel param assignment; *_2D methods). Add 1-D and 3-D synthetic tests.

4. **Phase 6 — port the VLE example** into `examples/vle_distillation/` (activity model,
   gibbs_duhem, phase_diagram, distillation, equilibrium) onto the public API, injecting
   the Julia activity `fwd_model`. Pin Clapeyron via a `juliapkg.json`.

5. **Phase 7 — reproduce all paper figures** via `paper/reproduce.py`; diff vs golden.

6. **Phase 8 — docs (Sphinx/RTD)**; **Phase 9 — publish (TestPyPI -> PyPI) + Zenodo**.

## Known issues / decisions already made (do not re-litigate)

- Package name `bits_for_gaps`; examples+paper in this repo; 2-D faithful first then N-D;
  freeze the current dependency stack; core is pure-Python (Julia only for `[vle]`).
  (REFACTOR_PLAN.md §7.)
- `entropy_lower_bound` was unused in the paper code but is a real algorithm feature
  (the closed-form bound) — kept and tested. Wire it into acquisition as an option.
- CI (`.github/workflows/ci.yml`) is a skeleton; the frozen stack is finicky on Linux
  runners — keep Julia/regression tests in a separate gated job.
- LICENSE is BSD-3-Clause; copyright holders Alexander W. Dowling and Kyla D. Jones
  (University of Notre Dame). Authors set accordingly in pyproject.toml.
