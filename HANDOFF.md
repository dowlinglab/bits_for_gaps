# HANDOFF — bits_for_gaps

State of the fresh `bits_for_gaps` repo. Read this + `REFACTOR_PLAN.md` before continuing.

- Bootstrap (Phase 0 + Phase 3-lite): Opus, 2026-07-03.
- Phase 2 (regression/test harness): done 2026-07-03, merged to `main`.
- **Phase 4 (decompose `sampler.py`; retire disk-as-state): done 2026-07-03 on branch
  `phase4-decompose-sampler` — awaiting review/merge to `main` before Phase 5 begins.**

## Phase 4 — decompose sampler.py; retire disk-as-state (done; review gate before Phase 5)

`sampler.py`'s `adaptiveEntropy` god-class is now an orchestrator over decomposed,
independently-testable modules. Order of operations (per the task): pinned an exact-
value characterization baseline BEFORE touching any code, then extracted modules one at
a time (verifying bit-exact parity against the still-untouched monolith before wiring
each one in), then rewrote `sampler.py` itself and retired disk-as-state in one final
commit. `pytest -q` = **72 passed, 1 deselected** (same `vle` marker as Phase 2). Full
suite runs in ~35 s (was ~17 s before Phase 4 — the two new modules' extra HMC/optimize
runs and the `BitsForGaps` facade parity test each re-run the tiny synthetic pipeline).

```
src/bits_for_gaps/
  gp.py            build_gp / maximize_lml / run_mcmc (GP construction + HMC + R-hat/ESS)
  diagnostics.py   thin tfp.mcmc wrappers (potential_scale_reduction, effective_sample_size)
  mixture.py       sample_gp_posterior_mixture (GMM predictive posterior) + predict_grid_2D
                   (the plotting-only full-grid diagnostic, formerly gp_predict_2D)
  acquisition.py   entropy_objective / entropy_surface_2D (was gen_entropy_surface_data_2D)
                   / optimize_2D -- uses entropy.py; kept the *_2D names (Phase 5 generalizes)
  transforms.py    InputTransform / OutputTransform -- lifts the XTrsfFwd/XTrsfBkwd/
                   yTrsfFwd/yTrsfBkwd lambda-list convention into small testable classes,
                   identity by default. Exported eagerly from __init__ (pure NumPy).
  state.py         IterationRecord / RunHistory -- the in-memory replacement for disk-
                   as-state (np.savetxt/pickle under results/{exp_name}/, read back next
                   iteration)
  sampler.py       adaptiveEntropy is now the orchestrator: thin delegating wrappers over
                   the modules above, plus `run(X_init, y_init, checkpoint_dir=None,
                   predict_grid=False)` -- the new entry point. Also adds `BitsForGaps`,
                   a thin renamed-kwarg subclass (target public API, REFACTOR_PLAN §4).
tests/unit/        + test_transforms.py, test_state.py
tests/integration/ test_end_to_end.py rewritten to drive run() (in-memory, no files);
                   + data/synthetic_baseline.json (exact-value pin from before Phase 4)
                   + test_bits_for_gaps_facade.py (BitsForGaps reproduces the same pin)
```

Key facts for the next session:
- **Disk-as-state is retired.** `adaptiveEntropy.run(X_init, y_init)` takes the initial
  design directly in memory and returns a `state.RunHistory`; a full run writes **zero**
  files by default (`test_run_writes_no_files_by_default` asserts this). File output is
  opt-in via `run(..., checkpoint_dir=...)`, a best-effort equivalent of the paper code's
  per-iteration dump (`test_run_checkpoint_dir_is_opt_in` asserts the key files land).
  `run_model()` is kept as a deprecated, disk-based shim (reads `activity_data_1` from
  `self.path`, the original zero-arg precondition) for anything still relying on it.
- **`predict_grid_2D` (was `gp_predict_2D`) is opt-in, not run by default.** It's a
  plotting-only diagnostic (~20 s in real config: 100 full-covariance draws over a 50x50
  grid) that doesn't feed the acquisition. Pass `run(..., predict_grid=True)` to compute
  it anyway. **It was never bitwise-reproducible even in the original paper code** --
  confirmed directly: `GPmodel.predict_f_samples` draws from TensorFlow's ambient
  (unseeded) global RNG, not NumPy's, so two successive calls on identical inputs in the
  same process already differ. `np.random.seed(self.seed)` in `sample_gp_posterior_mixture`
  only controls *which* posterior components are selected, not the draws themselves. This
  is exactly why the Phase 2 integration test excluded it from the determinism/baseline
  pins in the first place -- documented now in `mixture.py`.
- **The decomposition is verified behavior-preserving, not just self-consistent.**
  `tests/integration/data/synthetic_baseline.json` pins the tiny seeded synthetic run's
  exact outputs (rhat, ess, entropy field, xStar, max_entropy, next_data) captured from
  the pre-Phase-4 monolith; `test_matches_pre_phase4_baseline` checks the post-
  decomposition `run()` reproduces them at atol=1e-10. Additionally, before rewriting
  `sampler.py`, each extracted module (gp.py, mixture.py's non-predict_grid parts,
  acquisition.py) was checked bit-exact (atol=1e-12) against the still-untouched
  monolithic methods on the same run.
- **`BitsForGaps` is a thin subclass, not a rewrite.** Renamed constructor kwargs
  (`black_box`, `bounds`, `kernel`, `likelihood_variance`, `input_transform`,
  `output_transform`) matching REFACTOR_PLAN §4's target API; inherits every method
  (including `run`) unchanged from `adaptiveEntropy` -- no new computation, so no
  numeric risk. `__init__.py`'s lazy map now resolves `BitsForGaps` to this real class
  (previously it was just an alias for `adaptiveEntropy`). Advanced config (HMC tuning,
  restarts, mesh density) is still set via the same instance attributes as
  `adaptiveEntropy` (e.g. `.noSamples`) -- an `mcmc=MCMCConfig(...)`-style kwarg is
  deferred to Phase 5/6, once the core is N-D and doesn't need this passthrough.
- **`entropy.py` / `design.py` / `kernels.py` / `means.py` are byte-for-byte untouched**
  (`git diff main...HEAD -- <those 4 files>` is empty) -- Phase 4 touched only the
  sequential-design engine, per the guardrails.
- `TODO(Phase 5)` markers preserved verbatim in `gp.py` (`run_mcmc`'s positional
  `trainable_parameters[2],[0],[1]`), `mixture.py`/`acquisition.py` (by-name kernel
  param assignment: `std_dev`, `lengthscale_1`, `lengthscale_2`), and `acquisition.py`/
  `sampler.py`'s `*_2D` methods (hardcoded `d=2`). None of the 2-D / 3-hyperparameter
  hardcoding was touched.

## Phase 2 — regression/test harness (done; merged to main)

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

## What exists now (Phase 0 + Phase 3-lite + Phase 4 done)

A pip-installable package with the **algorithm decomposed into focused, tested modules**
(see the "Phase 4" section above for the sampler-engine breakdown):

```
src/bits_for_gaps/
  __init__.py    public API; pure pieces eager (entropy, design, transforms),
                 TF-backed pieces lazy (kernels, means, sampler) via PEP 562
  entropy.py     GMM density + 1st/2nd-order Taylor entropy + closed-form lower bound
                 (from fxns/max_ent_design.py; dead commented variants removed). PURE.
  design.py      latin_hypercube_design / full_factorial_design, N-D, pure, no disk I/O
                 (extracted from proh_water_class). PURE.
  kernels.py     AnisotropicSE (2-D, prior-bearing) from fxns/my_kermel_fxn.py
  means.py       FixedInverseMean from fxns/my_mean_fxn.py
  _util.py       standardize / normalize / make_tensor
  transforms.py  InputTransform / OutputTransform (Phase 4). PURE.
  state.py       IterationRecord / RunHistory (Phase 4).
  diagnostics.py R-hat / ESS (Phase 4).
  gp.py          GP construction + HMC (Phase 4).
  mixture.py     GMM predictive posterior (Phase 4).
  acquisition.py entropy-maximization acquisition function (Phase 4).
  sampler.py     adaptiveEntropy (orchestrator, Phase 4) + BitsForGaps (public-API facade)
tests/unit/      entropy/design/kernels/means/_util/transforms/state
tests/integration/ end-to-end (in-memory run(), Phase 4) + BitsForGaps facade parity
tests/regression/  golden-file checks vs published paper values (Phase 2)
```

`BitsForGaps` (public-API facade, REFACTOR_PLAN §4 kwarg names) and `adaptiveEntropy`
(original name, kept for backward compatibility) are both real, independently
importable classes as of Phase 4 -- see the "Phase 4" section above.

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
   Merged to `main`.

2. **Phase 4 — decompose `sampler.py`. ✅ DONE.** See the "Phase 4" section above. On
   branch `phase4-decompose-sampler`; merge to `main` after review, then start Phase 5.
   The green suite (regression + integration baseline pin) is the safety net.

3. **Phase 5 — generalize to N-D.** Remove the 2-D / 3-hyperparameter hardcoding flagged
   by `TODO(Phase 5)` markers, now spread across `kernels.py` (per-dim lengthscale
   attrs), `gp.py` (`run_mcmc`'s positional `trainable_parameters[2],[0],[1]`),
   `mixture.py`/`acquisition.py` (by-name kernel param assignment: `std_dev`,
   `lengthscale_1`, `lengthscale_2`), and `acquisition.py`/`sampler.py`'s `*_2D` methods
   (hardcoded `d=2` grids/meshes/Sobol dimension). Add 1-D and 3-D synthetic tests. The
   Phase 4 module boundaries (gp/mixture/acquisition as pure functions over explicit
   args) should make this a *local* change per module rather than a `sampler.py` rewrite.

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
