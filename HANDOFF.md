# HANDOFF — bits_for_gaps

State of the fresh `bits_for_gaps` repo. Read this + `REFACTOR_PLAN.md` before continuing.

- Bootstrap (Phase 0 + Phase 3-lite): Opus, 2026-07-03.
- Phase 2 (regression/test harness): done 2026-07-03, merged to `main`.
- Phase 4 (decompose `sampler.py`; retire disk-as-state): done 2026-07-03, merged to `main`.
- Phase 5 (generalize to N-D): done 2026-07-04, merged to `main`.
- **Phase 6 (port the VLE/distillation example): done 2026-07-04 on branch
  `phase6-vle-example` — awaiting review/merge to `main` before Phase 7 begins.**

## Phase 6 — port the VLE/distillation example (done; review gate before Phase 7)

The paper's H2O-PrOH case study now lives at `examples/vle_distillation/`, on the
public `bits_for_gaps` API, with the Julia/Clapeyron activity model injected as the
black box. `pytest -q` (default) = **117 passed, 1 deselected** (was 88 before Phase 6:
+29 new no-Julia unit tests for the example modules). `pytest -m vle` = **1 passed, 117
deselected** (~54 s; needs Julia/Clapeyron) -- the gated stage-table regression now
actually recomputes and checks, instead of skipping.

```
examples/vle_distillation/       repo-only -- NOT in the pip wheel (verified via
                                 `python -m build --wheel`: contents are only
                                 bits_for_gaps/*, no examples/ or paper/)
  __init__.py
  activity_model.py               Wilson gamma via Clapeyron.jl; LAZY juliacall import
  calculate_activities.jl         ported from fxns/calculate_activities.jl
  juliapkg.json                   pins Clapeyron.jl =0.6.26
  gibbs_duhem.py                  gamma_water from a modeled gamma_proh curve. PURE.
  phase_diagram.py                Antoine + bubble/dew point; wilson_gamma (Julia) /
                                  surrogate_gamma (bits_for_gaps GP + Gibbs-Duhem)
  equilibrium.py                  wraps a VLE curve as x_liquid -> y_vapor. PURE.
  distillation.py                 McCabe-Thiele column solver (fsolve). PURE.
  run_case_study.py               LHS -> Clapeyron -> BitsForGaps.run -> phase
                                  diagram + column; paper's exact 2-D config
  README.md                       setup + run, for a fresh clone
tests/conftest.py                 + sys.path insert of examples/ (see below)
tests/unit/                       + test_gibbs_duhem/phase_diagram/equilibrium/
                                  distillation.py (no Julia, run by default)
tests/regression/test_mccabe_thiele.py  gated recompute wired up (was a Phase-2
                                  placeholder that skipped)
```

Key facts for the next session:

- **Packaging mechanism (REFACTOR_PLAN §7.3), concretely implemented:**
  `tests/conftest.py` does `sys.path.insert(0, REPO_ROOT / "examples")`, which makes
  `import vle_distillation.<mod>` work in dev/CI without installing `examples/` as a
  distribution. This same mechanism made `examples/vle_distillation/juliapkg.json`
  auto-discoverable: `juliapkg` scans every `<sys.path entry>/<subdir>/juliapkg.json`
  (confirmed directly by reading `juliapkg/deps.py` and by the resolution log), so
  once `examples/` is on `sys.path`, the Clapeyron pin is found with **no extra
  wiring** -- no `Pkg.add` step, no environment variable. `run_case_study.py` does the
  equivalent `sys.path` insert itself (of its own parent's parent) so it works when
  invoked as a standalone script, not just under pytest.
- **Clapeyron.jl is pinned to 0.6.26** (uuid `7c7805af-46cc-48c9-995b-ed0ed2dc909a`,
  the version already resolved in this machine's Julia depot -- read directly from
  `~/.julia/environments/pyjuliapkg/Manifest.toml`). A fresh machine's first
  `juliacall` import (anything calling into `activity_model.py`) auto-downloads Julia
  itself plus this exact Clapeyron version into a per-conda-env directory
  (`$CONDA_PREFIX/julia_env/`) -- no manual bootstrap step. See
  `examples/vle_distillation/README.md`.
- **Lazy Julia, verified both directions:** `import vle_distillation.activity_model`
  (and every other example module that imports it) succeeds with **zero** Julia
  touched (`sys.modules` has no `julia*` entries) -- confirmed directly. Calling
  `activity_coefficients`/`black_box`/`wilson_gamma` triggers the lazy import and, if
  Julia/`[vle]` isn't installed, raises a clear `ImportError` naming the fix. `import
  bits_for_gaps` (the core) is unaffected either way -- it never touches
  `examples/vle_distillation/` at all.
- **Black-box adapter convention (the critical Phase-5-carryover contract):**
  `adaptiveEntropy.call_model` calls `FwdModel(*FwdModelArgs, *xStar)` -- natural
  dimension order, `xStar = [z_PrOH, T]` for this case study.
  `activity_model.black_box(z_proh, temperature)` matches that signature directly and
  returns `[gamma_proh]` (a 1-list, matching `call_model`'s `np.array(self.FwdModel(...))`
  convention) -- **not** both coefficients: the GP surrogate models only
  `gamma_PrOH(z, T)`; `gamma_water` is recovered via Gibbs-Duhem
  (`gibbs_duhem.gamma_water_from_gamma_proh`), never learned by a second GP output.
  Verified: `activity_coefficients(0.5, 350.0) == (1.4695, 1.7464)`, matching the
  target sanity value `(1.469, 1.746)`.
- **The paper's exact as-run manuscript config isn't committed verbatim anywhere in
  the old repo** (`driver_new.py`/`driver.py`'s own `run_test()` examples use a
  different, exploratory `"testing_12"`/`"SAFTγMie"` config, not
  `"less_x_new_manuscript_revisions"`/`"Wilson"`). `run_case_study.py`'s config
  (bounds `[(1e-6, 0.999), (350, 367)]`, `XTrsfFwd = [log(x+0.1), (T-350)/17]`,
  `yTrsfFwd = log`, `thermoModel = "Wilson"`, seed `10`) was reconstructed from the
  files that DO reference that experiment directly:
  `train_test_split_proh.py`'s commented-out `run_test(...)` call (transforms, bounds,
  seed) and `new_phase_diagram.py`'s `__main__` block (same transforms, confirms
  `AnisotropicSE`'s exact 2-D config == `kernels.AnisotropicSE.paper_2d()`). This is a
  physically-faithful reconstruction, not a byte-for-byte-verified original script --
  flagged here for anyone who later finds the real one.
- **The distillation solver's `fsolve` is fragile for arbitrary equilibrium curves**
  (no bounds, inherited from the original MATLAB-derived port) -- confirmed
  empirically: several hand-picked synthetic constant-relative-volatility test curves
  either failed to converge or converged to spurious (unphysical, e.g. negative flow
  rate) roots, while the real Wilson curve (with a 50-point z-grid, `Z_MESH`/`Z_GRID_SIZE`)
  converges cleanly to the real Geankoplis 11.4-1 column every time it was tried.
  `distillation.solve_column` now returns a `"warnings"` list (nonphysical mole
  fractions/flow rates, fsolve non-convergence) instead of silently returning bad
  numbers; both `run_case_study.py` and the gated regression test check `"converged"`
  before trusting a result. **Do not shrink the z-grid casually** -- a coarser
  (15-point) grid was enough to flip an otherwise-clean Wilson solve to a spurious
  root in testing.
- **The gated regression test's "surrogate" recompute intentionally does not use the
  full adaptive `BitsForGaps.run`/HMC loop.** It trains a GP on a 30-point LHS design
  (seed 10) evaluated against the same Clapeyron Wilson model, then fits it with
  `gp.maximize_lml` (a fast, deterministic MLE point estimate) -- reproducing the
  paper's real 15-iteration *adaptively*-designed surrogate bit-for-bit is Phase 7's
  job (full figure reproduction), not this phase's backend-correctness check. This
  recompute matches golden's `"wilson"` column within `atol=0.015` and `"surrogate"`
  within `atol=0.05` (looser -- a non-adaptive, far-smaller-sample surrogate doesn't
  track Wilson as tightly in the most dilute region near `xW=0.01` as the paper's
  refined surrogate did; the biggest observed gap, ~0.03-0.04, is in stage 4's vapor
  fraction there). A quick empirical note: a 4-iteration adaptive run from only 15
  initial points did *worse* on this metric than the 30-point plain LHS + MLE fit
  (too little HMC/data to beat a well-covering static design) -- consistent with the
  paper's own thesis that adaptive design needs enough iterations to pay off, not
  evidence against the adaptive loop itself.
- **Golden's own `"wilson"` column has ~0.01-level transcription slop.** It was
  hand-transcribed from reading paper Fig 9c (Phase 2, `paper/golden/README.md`), not
  computed from archived data. My direct Clapeyron recompute reproduces real physical
  landmarks exactly (`stage 1 vapor == xD == 0.43` exactly, by construction; pure-PrOH
  bubble point `370.35 K` matches 1-propanol's real normal boiling point to 4
  significant figures) yet differs from golden's `"wilson"` entries by up to `0.01`
  (e.g. stage 4 liquid: recompute is exactly `xW = 0.01`, golden's transcription says
  `0.02`) -- this is the eyeballed-figure-reading precision limit of that column, not
  a bug in the port. `paper/golden/*` is unmodified (guardrail); the test's
  `atol=0.015` for Wilson already accounts for this.
- **`src/bits_for_gaps/` CORE is byte-for-byte untouched** (`git diff main...HEAD --
  src/bits_for_gaps/` is empty) -- Phase 6 only added `examples/` + tests +
  `tests/conftest.py`'s `sys.path` insert, per the guardrails.

## Phase 5 — generalize to N-D (done; merged to main)

The 2 inputs / 3 hyperparameters hardcoding flagged by the `TODO(Phase 5)` markers is
gone. `pytest -q` = **88 passed, 1 deselected** (same `vle` marker). Full suite runs in
~65 s (was ~35 s after Phase 4 — the new 1-D/3-D synthetic tests each run the tiny HMC
pipeline, same as the existing 2-D one, just at two more dimensions).

**The whole phase was executed as: pin the 2-D baseline as the regression oracle, then
generalize one module at a time, running the full suite (incl. the atol=1e-10 baseline
pin) after every change before moving to the next module.** Every commit in this phase
kept `test_matches_pre_phase4_baseline` green — the 2-D path is bit-exact with the
pre-Phase-5 code throughout, not just at the end.

### Design decision: per-dimension scalar Parameters (not a vector Parameter)

`kernels.AnisotropicSE` now takes `variance_prior` + `lengthscale_priors` (a list, one
prior per input dimension) instead of hardcoding `lengthscale_1`/`lengthscale_2`. Each
lengthscale — and the variance — is its **own `gpflow.Parameter`**, not a slice of one
vector-valued Parameter. This was the one real design choice in this phase, and it's not
arbitrary:

- **The paper's method depends on per-dimension prior *families*, not just per-dimension
  prior *parameters*.** `std_dev` ~ LogNormal, `lengthscale_1` ~ LogNormal, `lengthscale_2`
  ~ Gamma — three different distribution families, one of which (`lengthscale_2`) is also
  deliberately left **unconstrained** (`Identity` transform, no positivity bijector,
  confirmed empirically: `gpflow.Parameter`'s default `transform` is `Identity`, not
  `positive()` — this was already true of the pre-Phase-5 kernel and had to be preserved
  bit-for-bit). A single vector Parameter carries exactly one prior distribution and one
  bijector for the whole vector — it cannot express "component 2 is Gamma-unconstrained,
  the rest are LogNormal-positive" without slicing hacks that would themselves need
  per-component metadata, i.e. would reinvent per-dimension Parameters anyway.
- **It matches gpflow's HMC machinery with no adapter.** `gpflow.optimizers.SamplingHelper`
  takes a flat list of trainable `Parameter`s; each contributes its own prior term to the
  log-posterior and its own bijector to the unconstrained HMC state. A list of scalar
  Parameters already *is* that list — `GPmodel.kernel.hyperparameters` is passed straight
  through to `SamplingHelper` (see below), no wrapping/unwrapping needed.

The tradeoff (documented in `kernels.py`'s module docstring) is more Parameter objects to
manage than one vector — acceptable, since gpflow's own tooling (`print_summary`,
checkpointing, `trainable_parameters`) already expects a flat list of scalar Parameters.

### Canonical hyperparameter order (the contract across gp/mixture/acquisition)

```
[std_dev, lengthscale_1, lengthscale_2, ..., lengthscale_d]
```

Exposed as `AnisotropicSE.hyperparameters` (a list of `gpflow.Parameter`, in this exact
order) and by name as `.std_dev`, `.lengthscale_1`, ... `.lengthscale_d` (kept for
backward compatibility with 2-D-era code addressing them by attribute name — the existing
`test_kernels.py` tests referencing `.lengthscale_1`/`.lengthscale_2` still pass
unmodified). This order is used in exactly three places, all now consistent by
construction rather than by hardcoded agreement:

- `gp.run_mcmc`: `SamplingHelper(GPmodel.log_posterior_density, GPmodel.kernel.hyperparameters)`
  — was `[trainable_parameters[2], [0], [1]]`. **Verified identity-equal** (same Python
  `Parameter` objects, same order) to the old hardcoded indexing for the paper's kernel
  before making this change — so this is a pure refactor at d=2, not a behavior change.
  `trace`/`chains_states`/`rhat`/`ess` columns are in this order.
- `mixture.sample_gp_posterior_mixture` / `acquisition.entropy_objective`: replay a trace
  row onto the kernel via the new `kernels.assign_hyperparameters(kernel, values)` —
  `for param, value in zip(kernel.hyperparameters, values): param.assign(value)` — instead
  of three hardcoded `.assign()` calls by name. Works for any kernel exposing
  `.hyperparameters`, not just `AnisotropicSE`.
- Anywhere reading a trace column back out (regression tests, `paper/golden/*`) already
  used this same order by convention; nothing there changed.

### What's still 2-D-only, and why that's the right call

`acquisition.entropy_surface_2D` and `mixture.predict_grid_2D` (the dense-grid entropy
field and the full-grid GP-prediction plotting diagnostic) are **not** generalized to N-D
— a dense grid is exponential in the input dimension, and neither one feeds the
acquisition (the actual next-point decision). Both now raise a clear `ValueError` if
called with `len(x_bounds) != 2`. `adaptiveEntropy.run()` calls `entropy_surface_2D`
*only* when the input space is 2-D, leaving `entropy_field=None` otherwise — an N-D run
does not error, it just skips a diagnostic it was never going to use. If N-D
visualization is ever needed, the right tool is a *sparse* diagnostic (e.g. entropy along
1-D/2-D slices through the current best point), not a full grid — not built here since
nothing calls for it yet.

By contrast, **`acquisition.optimize` (was `optimize_2D`) — the function an N-D run
actually depends on — is fully dimension-general**: Sobol dimension and the restart
bound-scaling both derive from `len(x_bounds)`. For d=2 this reproduces the pre-Phase-5
`optimize_2D` bit-for-bit (verified via the baseline pin): the vectorized bound-scaling
`lo + x0 * (hi - lo)` is the same floating-point operation as the original
`x0[j] * (hi[j] - lo[j]) + lo[j]` (IEEE 754 addition is commutative, so reordering the
addends doesn't change the rounding).

### `call_model`'s black-box calling convention changed (interface, not algorithm)

Pre-Phase-5, `call_model` called the injected black box as `FwdModel(*FwdModelArgs, x2,
x1)` — reversed, 2-D-specific argument order inherited from the VLE example's Julia
activity-coefficient function (which took `(T, x)`). This doesn't generalize to N inputs.
Phase 5 changes it to `FwdModel(*FwdModelArgs, *xStar)` — `xStar`'s components in natural
dimension order. This is an **interface** change to how the sampler calls the user's
injected function, not an algorithm change: `tests/integration/test_end_to_end.py`'s
`_fwd_model` was updated from `_fwd_model(x2, x1)` to `_fwd_model(x1, x2)` to match, and
produces the exact same `(x1, x2, y)` values as before (verified via the atol=1e-10
baseline pin) — reordering which positional slot carries which value doesn't change the
value itself. **Phase 6 will need to account for this** when porting the VLE example's
Julia `fwd_model` wrapper (it can no longer rely on the reversed-argument convention; wrap
the Julia call so its own signature accepts `(x1, x2, ...)` in natural order).

### Other Phase 5 changes

- `sampler.py`: `adaptiveEntropy.optimize_2D` renamed to `.optimize` (matches
  `acquisition.optimize`); `predict_grid_2D`/`entropy_surface_2D` method names kept
  as-is (explicitly 2-D-only). `BitsForGaps`'s constructor and `.run()` are unaffected —
  it already accepted any `bounds`/`kernel`, so N-D "just worked" once the modules
  underneath it did (proven by `tests/integration/test_nd_synthetic.py` running 1-D/3-D
  problems through `BitsForGaps`, not `adaptiveEntropy` directly).
- New tests: `tests/integration/test_nd_synthetic.py` (1-D and 3-D synthetic problems,
  pure-Python, no Julia, via `BitsForGaps.run(...)`) and extensions to
  `tests/unit/test_kernels.py` (canonical order, `paper_2d()` parity,
  `assign_hyperparameters` round-trip, explicit 1-D/3-D construction with mixed prior
  families, constructor validation).
- `entropy.py` / `design.py` / `means.py` / `_util.py` are byte-for-byte untouched
  (`git diff main...HEAD -- <those 4 files>` is empty). `paper/golden/*` untouched;
  no `results/` committed.

## Phase 4 — decompose sampler.py; retire disk-as-state (done; merged to main)

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

## What exists now (Phase 0 + Phase 3-lite + Phase 4 + Phase 5 + Phase 6 done)

A pip-installable package with the **algorithm decomposed into focused, tested modules**,
**generalized to N input dimensions**, and a **ported VLE/distillation example** repo-side
(see the "Phase 4"/"Phase 5"/"Phase 6" sections above):

```
src/bits_for_gaps/
  __init__.py    public API; pure pieces eager (entropy, design, transforms),
                 TF-backed pieces lazy (kernels, means, sampler) via PEP 562
  entropy.py     GMM density + 1st/2nd-order Taylor entropy + closed-form lower bound
                 (from fxns/max_ent_design.py; dead commented variants removed). PURE.
  design.py      latin_hypercube_design / full_factorial_design, N-D, pure, no disk I/O
                 (extracted from proh_water_class). PURE.
  kernels.py     AnisotropicSE (N-D, per-dimension prior-bearing Parameters, Phase 5) +
                 assign_hyperparameters(kernel, values)
  means.py       FixedInverseMean from fxns/my_mean_fxn.py
  _util.py       standardize / normalize / make_tensor
  transforms.py  InputTransform / OutputTransform (Phase 4). PURE.
  state.py       IterationRecord / RunHistory (Phase 4).
  diagnostics.py R-hat / ESS (Phase 4).
  gp.py          GP construction + HMC, N-D via kernel.hyperparameters (Phase 4/5).
  mixture.py     GMM predictive posterior, N-D (Phase 4/5); predict_grid_2D stays 2-D-only.
  acquisition.py entropy-maximization acquisition; optimize is N-D (Phase 5),
                 entropy_surface_2D stays 2-D-only (raises for d != 2).
  sampler.py     adaptiveEntropy (orchestrator) + BitsForGaps (public-API facade) -- N-D
                 throughout except the 2-D-only diagnostics, which run() skips for d != 2.
tests/unit/      entropy/design/kernels(+N-D)/means/_util/transforms/state
tests/integration/ end-to-end (2-D, in-memory run()) + BitsForGaps facade parity +
                 nd_synthetic (1-D and 3-D, via BitsForGaps)
tests/regression/  golden-file checks vs published paper values (Phase 2, 2-D only --
                 the published run and paper/golden/* are inherently 2-D); +
                 test_mccabe_thiele.py's @pytest.mark.vle recompute (Phase 6)
examples/vle_distillation/  the H2O-PrOH case study on the public API (Phase 6) --
                 repo-only, not in the pip wheel; see the "Phase 6" section above
```

`BitsForGaps` (public-API facade, REFACTOR_PLAN §4 kwarg names) and `adaptiveEntropy`
(original name, kept for backward compatibility) are both real, independently
importable classes; both work at any input dimension as of Phase 5.

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

2. **Phase 4 — decompose `sampler.py`. ✅ DONE.** Merged to `main`.

3. **Phase 5 — generalize to N-D. ✅ DONE.** Merged to `main`.

4. **Phase 6 — port the VLE example. ✅ DONE.** See the "Phase 6" section above. On
   branch `phase6-vle-example`; merge to `main` after review, then start Phase 7. The
   green default + `-m vle` suites are the safety net.

5. **Phase 7 — reproduce all paper figures** via `paper/reproduce.py`; diff vs golden
   (all still 2-D -- the published run is 2-D, so this phase doesn't touch N-D at all).
   This is where the paper's real 15-iteration adaptive run needs reproducing
   bit-for-bit (or within tolerance) -- Phase 6's `run_case_study.py` demonstrates the
   pipeline but deliberately runs far fewer iterations, and the gated regression
   test's "surrogate" recompute deliberately skips the adaptive loop entirely (see the
   "Phase 6" section above for why). Also worth finding/committing the *actual*
   as-run manuscript script if it turns up -- Phase 6 had to reconstruct the config
   from indirect evidence (see "Phase 6" section); the McCabe-Thiele stage-table
   comparison is the one place this reconstruction is checked against golden.

6. **Phase 8 — docs (Sphinx/RTD)**; **Phase 9 — publish (TestPyPI -> PyPI) + Zenodo**.
   Phase 8 should document the N-D kernel construction (`AnisotropicSE(variance_prior=...,
   lengthscale_priors=[...])`) alongside the 2-D quickstart, plus the `paper_2d()` factory.

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
