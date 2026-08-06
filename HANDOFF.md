# HANDOFF — bits_for_gaps

State of the fresh `bits_for_gaps` repo. Read this + `REFACTOR_PLAN.md` before continuing.

- Bootstrap (Phase 0 + Phase 3-lite): Opus, 2026-07-03.
- Phase 2 (regression/test harness): done 2026-07-03, merged to `main`.
- Phase 4 (decompose `sampler.py`; retire disk-as-state): done 2026-07-03, merged to `main`.
- Phase 5 (generalize to N-D): done 2026-07-04, merged to `main`.
- Phase 6 (port the VLE/distillation example): done 2026-07-04, merged to `main`.
- Phase 7 (reproduce the paper's figures): done 2026-07-04, merged to `main`.
- Phase 8 (Sphinx + MyST + ReadTheDocs docs): done 2026-07-04, merged to `main`.
- Phase 9 (reproduce ALL paper results, including the from-scratch stochastic
  HMC+acquisition loop; archive-free figure reproduction): done 2026-07-04, merged to
  `main`.
- Phase 9b (investigate + fix the McCabe-Thiele non-convergence Phase 9 flagged as a
  discrepancy): done 2026-07-04, merged to `main`.
- Phase 9c (robustness hardening -- first sanctioned `src/bits_for_gaps/` core change
  since Phase 4): done 2026-07-04, merged to `main`.
- Phase 9d (whole-codebase polish: hygiene [ruff, type hints, CI, coverage] +
  faithfulness [selectable acquisition, MC-validation, synthetic example, hardening,
  CHANGELOG]): done 2026-07-04, merged to `main`.
- Phase 9e (docs/tests/comments quality pass -- behavior-preserving: `golden` ->
  `reference` rename, `theory.md` equation fidelity, test-coverage gap-filling,
  NumPy-docstring + inline equation-citation pass): done 2026-07-05, merged to `main`.
- **Phase 10 (release engineering for the v0.1.0 PyPI release -- packaging/CI only, no
  algorithm changes, no credentialed step performed): done 2026-08-06 on branch
  `phase10-release` — awaiting review/merge to `main`, then the maintainer-only steps
  in `RELEASE.md`.**

## Phase 10 — release engineering for v0.1.0 (done; review gate before the maintainer's release steps)

Packaging/CI/docs only -- `src/bits_for_gaps` logic and every regression value
untouched. `pytest -q`: 204 passed, 2 deselected throughout (unchanged from Phase 9e).

- **Version, single-sourced.** `pyproject.toml`'s `[project]` gained
  `dynamic = ["version"]` + a `[tool.hatch.version]` pointing at
  `src/bits_for_gaps/__init__.py` (hatchling's default "regex" version source matches
  `__version__ = "..."`), so `__version__` stays the one place edited to cut a
  release. Bumped `0.0.1.dev0` -> `0.1.0`.
- **PyPI metadata polish.** Classifiers: `Development Status :: 3 - Alpha` ->
  `4 - Beta`, plus a generic `Python :: 3` classifier and an OS-independent /
  AI-topic classifier. `project.urls` gained `Repository`/`Documentation`/`Changelog`
  alongside the existing `Homepage`/`Paper`.
- **Sdist scope, fixed.** `python -m build`'s sdist defaulted to hatchling's
  whole-repo file set (193 entries, 6.9 MB, including all of `paper/data/`'s curated
  ~16 MB plot-input subset, `tests/`, `examples/`, `docs/`) -- a real packaging gap,
  not caught by the wheel (which was already clean:
  `bits_for_gaps/*` + `py.typed` + dist-info only). Fixed via
  `[tool.hatch.build.targets.sdist]`'s `only-include` (not `include`, which *adds* to
  the default set -- tried first, still leaked several READMEs), restricting the
  sdist to `src/bits_for_gaps` + `LICENSE`/`README.md`/`CHANGELOG.md` (6.9 MB -> 33 KB).
  `twine check dist/*` passes for both artifacts; wheel `METADATA`'s `Version:` field
  matches the single-sourced `__version__`.
- **Clean-env install audit.** In a fresh, throwaway conda env (no dev tooling, no
  editable install), `pip install dist/bits_for_gaps-0.1.0-py3-none-any.whl` resolved
  every pinned dependency cleanly from wheel metadata alone; bare `import
  bits_for_gaps` stayed Julia-free/TF-lazy; a smoke test (`AnisotropicSE()`,
  `latin_hypercube_design`, `entropy.second_order_entropy`, and `docs/quickstart.md`'s
  exact `BitsForGaps(...)` construction snippet) matched the installed API with no
  doc changes needed. Env torn down after.
- **Trusted-publishing CI.** New `.github/workflows/publish.yml`: a shared build job
  (the same `python -m build` + `twine check` verified above) feeds two gated publish
  jobs via OIDC trusted publishing (`pypa/gh-action-pypi-publish`, a dedicated
  `release` environment, job-scoped `id-token: write`) -- no API tokens anywhere in
  the repo. A `v*` tag push reaches PyPI; a manual `workflow_dispatch` reaches
  TestPyPI only, for the pre-release dry run. `gh-action-pypi-publish` is referenced
  via PyPA's own recommended floating tag (`release/v1`) rather than a fabricated
  commit SHA (this session had no network access to verify one) -- `RELEASE.md`
  documents the optional SHA-pinning step for the maintainer.
- **CHANGELOG finalized, README badges added.** `[Unreleased]` renamed to
  `[0.1.0] - YYYY-MM-DD` (a placeholder; the maintainer fills in the actual date at
  tag time) with a fresh empty `[Unreleased]` left above it; added the Phase
  9e/10 bullets that weren't yet reflected. `README.md` gained PyPI-version and
  ReadTheDocs badges alongside the existing CI badge (both inert until the
  maintainer's release/RTD-import steps below actually happen).
- **`RELEASE.md`** — new maintainer checklist: the exact build-artifact contents
  audit, the clean-env install audit, and the four maintainer-only steps in order
  (register trusted publishers on PyPI + TestPyPI; TestPyPI dry run via
  `workflow_dispatch`; set the CHANGELOG date, `git tag v0.1.0 && git push --tags`;
  activate ReadTheDocs). None of these four were performed by this phase.

Verified at every commit: `pytest -q` (204 passed, 2 deselected), `pytest -m vle` (2
passed), `ruff check` clean, `sphinx-build -W` clean, and the Julia-free/TF-lazy
import contract. No PyPI/TestPyPI upload, no account configuration, no trusted-
publisher registration, no `git tag`, no RTD activation -- all maintainer-only, all
documented in `RELEASE.md`, none performed.

## Phase 9e — docs/tests/comments quality pass (done; review gate before Phase 10)

Three passes, each its own commit, suite green at every one. Behavior-preserving
throughout (docstrings/comments/tests/docs + one mechanical rename -- no algorithm
changes): baseline (atol 1e-10) + all reference regressions + `pytest -m vle` never
moved; `paper/data/` and the dependency stack untouched.

- **PASS 1 -- rename + docs.** `paper/golden/` -> `paper/reference/` everywhere
  (`git mv`, the `golden` pytest fixture -> `reference`, `extract_golden.py` ->
  `extract_reference.py`, every prose/docstring mention across ~30 files) --
  confirmed the 4 JSON files are byte-identical (only the directory moved) and zero
  "golden"/"Golden" text remains anywhere in the tracked repo. `docs/theory.md`
  rewritten to present the paper's key equations in its own notation with equation
  numbers -- Eq (1)/(2) entropy+acquisition, Eq (3) hyperparameter posterior, Eq (4)
  GP prior, Eq (5a)/(5b) predictive mean/variance, Eq (6) SE kernel + Table 1's exact
  priors, Eq (7) GMM predictive + Eq (8a)/(8b) moments, Eq (9) + the Huber et al.
  (2008) Taylor expansion and its Proposition truncation bound (SI-1), the closed-form
  entropy-lower-bound Theorem and its SI-2 cross-overlap term, Algorithm 1 (credible
  intervals, Lalchand & Rasmussen 2020), and Eq (10)/(11) + SI-4 for the VLE example --
  each linked to the implementing module. Every doc page's cross-links verified to
  resolve in the rendered HTML (not just `sphinx-build -W`, which doesn't catch
  unresolved `:func:`/`:class:` refs with `nitpicky=False`); fixed a stale test count
  in `installation.md` (117/5 -> 193/2). Docs page organization was already sound --
  no reordering needed.
- **PASS 2 -- test-coverage audit.** Added `tests/unit/test_diagnostics.py` (new --
  `potential_scale_reduction`/`effective_sample_size` had zero direct unit tests
  before, only indirect coverage via full HMC integration runs). Added tests
  exercising `entropy_surface_2D`/`predict_grid_2D`'s documented non-2-D `ValueError`
  (defined since Phase 5, never exercised by any test); direct tests for
  `entropy.gaussian_mixture_density` against `scipy.stats`; strengthened
  `InputTransform`'s 1-D-input test and `sampler.call_model`'s success test to check
  actual values, not just shapes; added a test confirming `call_model` extracts only
  the first element of a multi-element black-box output. One finding, not a bug:
  `design.full_factorial_design`'s "grid too small" `ValueError` is unreachable dead
  code for every `(bounds, n_train, n_test)` -- `levels = ceil(n_total ** (1/d))`
  mathematically guarantees `levels**d >= n_total` (verified by brute-force search);
  documented in a comment rather than given a fake test. `pytest -q`: 193 -> 204
  passed (2 deselected, unchanged).
- **PASS 3 -- docstrings + inline equation citations.** Filled NumPy-docstring gaps
  (missing Parameters/Returns/Notes: `entropy.py`'s `first_order_entropy_approx`/
  `cholesky`/`gradient_gaussian_mixture_density`/`second_order_entropy`, `gp.py`'s
  `build_gp`/`maximize_lml`/`run_mcmc`, `means.py`'s `FixedInverseMean`, `_util.py`'s
  three array helpers) and added inline comments citing the paper's equation numbers
  at the exact lines implementing them (verified against the paper + SI): Eq (6) in
  `kernels.AnisotropicSE.K`, Eq (3)/(4)/(5a)/(5b) in `gp.py`, Eq (7)/(9)/Theorem-SI-2
  in `entropy.py`, Eq (1)/(2)/(5a)/(5b)/(7) in `acquisition.py`/`mixture.py`, and
  Eq (10)/(11)/SI-4 in the VLE example's `phase_diagram.py`/`gibbs_duhem.py`/
  `distillation.py`.

## Phase 9d — whole-codebase polish: hygiene + faithfulness (done; merged to main)

Two batches, eight workstreams (A-D hygiene, E-H faithfulness/features), each its own
commit, suite green at every one. Behavior-preserving throughout: baseline (atol
1e-10) + all reference regressions + `pytest -m vle` never moved.

**Hygiene (A-D):**

- **A -- ruff lint + format, whole repo.** Added to the `[dev]` extra; configured
  (`E`/`F`/`W`/`I`/`B`, line-length 100) with per-file-ignores protecting the two
  load-bearing import orders (`__init__.py`'s eager-vs-lazy split;
  `PYTHON_JULIACALL_HANDLE_SIGNALS` set before the `juliacall`/TF-threading-config
  imports it protects, in `activity_model.py` and `paper/full_reproduction.py`). A
  handful of non-autofixable findings (assigned lambdas, one unused import, two
  intentionally-blind `pytest.raises(Exception)`) fixed by hand; the rest (52 files)
  is `ruff format`'s cosmetic whitespace/quote normalization.
- **B -- type hints across `src/bits_for_gaps/`, `py.typed` (PEP 561).** Every module
  annotated via `from __future__ import annotations` (stringized -- no new runtime
  imports, so the Julia-free/TF-lazy contract holds). `py.typed` confirmed included
  in the built wheel. Fixed a real `sphinx-build` regression the new hints exposed:
  `InputTransform`/`OutputTransform` are documented twice (top-level re-export +
  original location), and once a type hint referenced them by bare name,
  `autodoc-typehints` couldn't disambiguate ("more than one target found") -- fixed
  via `docs/api.rst`'s `:exclude-members:`.
- **C -- broadened CI + badge.** `.github/workflows/ci.yml` now runs `ruff check`
  plus the full default `pytest -q` (unit + integration + regression; still excludes
  `@pytest.mark.vle`), not just `tests/unit`. Verified in a fresh conda env with only
  the `[dev]` extra (no Julia) that this is genuinely Julia-free.
- **D -- coverage pass, 93% -> 97%.** `gp.py` had zero direct unit tests (only
  indirect coverage via full HMC integration runs); new `tests/unit/test_gp.py`
  closes it to 98%. A few real `sampler.py` gaps closed too (`read_data`,
  `BitsForGaps`'s custom transform override, `run(predict_grid=True)`,
  `run(initalLML=True)`). No coverage gate added to CI (informational only).

**Faithfulness/features (E-H):**

- **E -- `entropy_lower_bound` wired up as a selectable acquisition objective.** The
  paper derives two entropy estimators; the closed-form lower bound was implemented
  and unit-tested since Phase 2 but never usable for acquisition.
  `acquisition.entropy_objective`/`optimize`/`entropy_surface_2D` gain
  `objective="taylor"|"lower_bound"`; `BitsForGaps.acquisitionObjective` (default
  `"taylor"` -- unchanged existing behavior).
- **F -- Monte-Carlo validation of the entropy approximations.** New
  `tests/unit/test_entropy_mc_validation.py` estimates the true GMM differential
  entropy directly (sample the mixture, average -log of its own density at those
  samples) on several mixtures and checks the Taylor approximation stays close
  (rtol=0.15, calibrated empirically) and the lower bound stays at or below it --
  validates approximation *quality*, not just a regression pin.
- **G -- a pure-Python synthetic example.** `examples/synthetic/run_example.py`: a
  small, actually-runnable, Julia-free script (`python examples/synthetic/run_example.py`,
  well under a minute) -- the onboarding path that previously only pointed at test
  files. `docs/quickstart.md` now points to it.
- **H -- clear diagnostic for an unassignable hyperparameter value.**
  `kernels.assign_hyperparameters` (called deep in `mixture.py`/`acquisition.py`'s hot
  loops) now catches gpflow's low-level `InvalidArgumentError` for a value that can't
  round-trip through a parameter's transform (e.g. an extreme `lengthscale_2` outlier
  -- deliberately unconstrained, no positivity bijector) and re-raises a `ValueError`
  naming the parameter and value. Also added `CHANGELOG.md` (Keep a Changelog,
  Unreleased section summarizing every phase, targeting `v0.1.0`).

`pytest -q`: 166 -> 193 passed (2 deselected throughout). `paper/data/`, `paper/reference/*`,
and the pinned dependency stack untouched. `import bits_for_gaps` confirmed Julia-free
and TensorFlow-lazy after every workstream. Docs updated:
`docs/improvements_over_paper.md` (new sections for E/F/G/H), `docs/theory.md` and
`docs/reproduce_paper.md` (both had gone stale describing pre-Phase-9/9d behavior --
fixed in passing), `docs/quickstart.md`, `docs/api.rst`.

## Phase 9c — robustness hardening (done; merged to main)

Behavior-preserving only, by design: no numerical result, default, or seed changed.
The pre-Phase-4 baseline (atol 1e-10) + all `paper/reference/*` regressions +
`pytest -m vle` stayed green at every commit. 46 new tests added (120 -> 166 passed, 2
deselected).

**Mutation-footgun fix, broader than Phase 9b realized.** Phase 9b's bug (a script
reusing a `GPmodel` after `mixture.sample_gp_posterior_mixture` mutated its kernel and
left it at an arbitrary leftover state) turned out to be a symptom of a real footgun
in the library, not a one-off script mistake: `sampler.py`'s own `run()` calls
`entropy_surface_2D`/`optimize` (both funnel through `acquisition.entropy_objective`,
which reassigns kernel hyperparameters in the same way) on the *same* `GPmodel` object
it then stores in `IterationRecord.GPmodel` and optionally checkpoints. **Every run's
returned/checkpointed model** used to carry this arbitrary state, not just the one
Phase 9b patched. Both `sample_gp_posterior_mixture` and `entropy_objective` now
save the kernel's hyperparameters before mutating and restore them in a `finally` --
`kernels.py` gained `save_hyperparameters()` to pair with the existing
`assign_hyperparameters()`. Verified behavior-preserving for every value either
function computes/returns; new unit tests assert kernel state is unchanged
before/after (including on the error path), plus an integration test reproducing the
exact Phase 9b scenario end-to-end.

**Public-API input validation** (`adaptiveEntropy`/`BitsForGaps`): clear `ValueError`s
for `x_bounds` `lo >= hi`, bounds-vs-kernel-`ndim` mismatch (checked in `__init__`),
non-positive/out-of-range HMC/acquisition config and `X_init`/`y_init` shape
mismatches (checked in `run()`, since config is conventionally set via
post-construction attribute assignment throughout this codebase, not passed to
`__init__`), and a black-box output that isn't a non-empty sequence (`call_model`) --
previously cryptic failures deep inside GPflow/TensorFlow or a bare `IndexError`.

**SHOULD-fixes, confirmed valuable by the audit:** `entropy.py`'s `assert pl > 0`
(silently stripped under `python -O`) is now an explicit `ValueError`.
`examples/vle_distillation/distillation.py`'s `solve_column` retries a few generic
alternate `fsolve` initial guesses if the default doesn't converge -- the primary
attempt is byte-for-byte unchanged, so this only ever activates when the default
already reports non-convergence. `mixture.sample_gp_posterior_mixture`/
`predict_grid_2D` gained an optional `tf_seed` to make `predict_f_samples` draws
reproducible on request (default `None` leaves the documented ambient-RNG behavior
unchanged).

**Verification:** re-ran the full 15-iteration stochastic loop from scratch
(`results_remaked/phase9c_fullrun/`, gitignored; hit a one-off ~15 min TF
`tf.function`-retracing anomaly at iteration 2, unrelated to this phase's changes --
other iterations ran at the normal ~13 s/iteration pace). `column_surrogate_converged`
stayed `True`; R-hat/ESS/hyperparameter posterior matched Phase 9b's committed
`full_run_summary.json` to 5-6 significant figures (within the already-documented
run-to-run floating-point tolerance) -- confirming the hardening changed no numerical
behavior. No update to `paper/phase9_validation/` needed (values didn't move outside
that tolerance).

**Docs:** new `docs/improvements_over_paper.md` (wired into `docs/index.md`)
consolidates bug fixes (the `equilibrium.py` missing-path repoint, Phase 9b's mutation
bug), this phase's hardening, and the architecture wins from every prior phase.
`docs/reproduce_paper.md` was also fixed in passing -- found stale from before Phase 9
(still described the pre-archive-free, author-access-required flow).
`sphinx-build -W docs docs/_build/html` succeeds with zero warnings.

`paper/data/`, `paper/reference/*`, and the dependency stack were not touched.
`import bits_for_gaps` confirmed still Julia-free.

## Phase 9b — root-cause the McCabe-Thiele discrepancy (done; merged to main)

Phase 9's "genuine discrepancy" (the fully-adaptive surrogate's McCabe-Thiele column
not converging, attributed to entropy-driven acquisition) turned out to be **wrong** --
a shared-mutable-state bug in `paper/full_reproduction.py`: `_predict_split` (test-RMSE)
mutates `GPmodel.kernel` in place via `bits_for_gaps.mixture.sample_gp_posterior_mixture`
(by design -- documented, not a core bug), and the phase-diagram code reused that same
mutated object afterward. Confirmed by reproducing the exact reported failure
(stage-2 liquid=1.7258, to 4 decimal places) from the checkpointed `gp_model_15.pkl`.

Of the 4 hypotheses posed (draw-count/averaging, monotonicity, deterministic mean,
under-resolution), **none was the root cause** -- but testing them was still useful:
monotonicity is definitively rejected (all 5 reconstructed curves, including the failing
one, have zero violations), and the "deterministic posterior-mean" idea is falsified
(exactly as smooth as the working curves, yet fails -- a solver initial-guess-
sensitivity issue, not a curve-smoothness one). See `paper/PHASE9B_INVESTIGATION.md` for
the full analysis.

Fix (example layer only, `src/bits_for_gaps/` untouched): reordered
`paper/full_reproduction.py` to build the phase diagram before the mutating RMSE loop,
and added `examples/vle_distillation/phase_diagram.py`'s `surrogate_gamma_averaged`
(matches the paper's own `new_phase_diagram.py` hyperparameter-posterior-averaging
construction) for added robustness. Re-ran the full 15-iteration stochastic loop from
scratch with the fix: `column_surrogate_converged` now `True`, stage table within 0.03
mole fraction of Wilson at every stage. `pytest -q` (120/2) and `pytest -m vle` (2/120)
both still green; `paper/reproduce.py --figures 8 9` still works. Pre-fix and post-fix
`full_run_summary.json`s both committed under `paper/phase9_validation/` for the record.

## Phase 9 — full stochastic reproduction + archive-free figures (done; merged to main)

Two independent, separately-committed pieces (see REFACTOR_PLAN.md's Phase 9 entry
and §7 decision 4):

**STEP 1 -- archive-free figure reproduction.** `paper/reproduce.py` no longer needs
private-archive access by default. Enumerated (from the code, not guessed) exactly
which files every `paper/figures/*.py` module reads via `paper/figures/_archive.py`'s
loaders, and copied exactly that set (~16 MB, well under the 30-50 MB budget --
`gp_predict_2/3/4` turned out to not be read by anything, only `gp_predict_{1,15}`)
into a new **tracked** `paper/data/`, with provenance in `paper/data/README.md`.
Repointed `paper/reproduce.py`'s `DEFAULT_ARCHIVE` and
`tests/regression/test_paper_figures.py`'s `ARCHIVE_DIR` at it. Re-gated
`test_paper_figures.py`: the three tests that only *read* committed data (Fig 5 error
metrics, Fig 10 HMC diagnostics, the hyperparameter posterior behind Fig 11) moved out
of `@pytest.mark.vle` into the default suite; only the Fig 8 Wilson-curve cross-check
(recomputes via live Clapeyron) stays gated. `pytest -q` went from 117/5 to
**120 passed, 2 deselected**. `tests/regression/test_mccabe_thiele.py` needed **no**
change -- it has no `ARCHIVE_DIR` at all (Fig 9's stage table is a pure physics
recompute, no archived data read), contradicting the task's assumption that it needed
repointing.

**STEP 2/3 -- from-scratch stochastic reproduction.** New `paper/full_reproduction.py`
drives `bits_for_gaps.sampler.BitsForGaps` through the paper's exact 15-iteration
adaptive HMC+entropy-acquisition config (10 train/10 test LHS, seed 10, bounds
`[[1e-6,0.999],[350,367]]`, `AnisotropicSE.paper_2d()`, `likelihood_var=0.1`, HMC
`noSamples=5000/noBurnIn=0/noChains=4/noLeapfrogSteps=5/stepSize=0.05/noAdaptSteps=5/
targetAccept=0.9/adaptRate=0.1`, `noGaussians=15/noRestarts=10`) starting from a
*fresh* LHS design and a live Clapeyron/Wilson black box -- no archived data read.
Ran once (2026-07-04, ~25 min wall time on this machine); artifacts stayed in the
gitignored `results_remaked/phase9_fullrun/`. Committed: the script itself, and a
small (~200 KB) `paper/phase9_validation/` (2 summary PNGs + `full_run_summary.json`)
backing a new "Phase 9: from-scratch stochastic reproduction" section in
`paper/REPRODUCTION.md`. **Not gated in CI** -- one-time validation artifact, per the
task's explicit instruction.

Headline finding, stronger than expected: everything in the loop that runs through a
`self.seed`-seeded path (LHS design, HMC, the entropy-acquisition optimizer)
reproduced the published run's R-hat/ESS, hyperparameter posterior, and entropy-decay
curve to **6-8 significant figures** -- not just "qualitatively similar." The *only*
place real stochastic drift shows up is `gpflow`'s `predict_f_samples` (documented in
`bits_for_gaps/mixture.py` as drawing from TensorFlow's non-seedable ambient RNG),
which feeds the test-RMSE curve (4.337 -> 0.887 here vs. the paper's ~4.34 -> ~0.67 --
same regime, not the same value) and the surrogate phase diagram. One genuine,
documented discrepancy: the **genuinely** 15-iteration-adaptive surrogate GP's
McCabe-Thiele column did **not** converge to a physical solution (unlike
`fig09_mccabe_thiele.py`'s dedicated 30-point-LHS/MLE-fit stand-in, which does) --
entropy-driven acquisition optimizes for predictive accuracy at held-out points, not
for a globally smooth-enough equilibrium curve for the stage-stepping solver. See
`paper/REPRODUCTION.md`'s Phase 9 section for full numbers and discussion.

`src/bits_for_gaps/` core, `paper/reference/*`, and the dependency stack were not
touched. `import bits_for_gaps` confirmed still Julia-free.

## Phase 8 — documentation (done; merged to main)

`docs/` is a Sphinx 7 + MyST + furo site, repo-only (not shipped in the pip wheel --
same policy as `examples/`/`paper/`), built via the existing `[docs]` extra.
`sphinx-build -W docs docs/_build/html` succeeds with **zero warnings** (confirmed
on repeated clean builds). `pytest -q` is unaffected: **117 passed, 5 deselected**
(docs are additive; no source files under `src/bits_for_gaps/`, `examples/`, or
`paper/` changed -- `git diff main -- <path>` is empty for all three).

```
docs/
  conf.py            Sphinx config; see "RTD/TensorFlow decision" below
  index.md           overview, paper DOI, status, page map
  installation.md     core / dev / [docs] / optional [vle]+Julia install; explicit
                      "examples/ and paper/ are repo-only, clone required" note
  quickstart.md        pure-Python BitsForGaps(black_box, bounds, kernel=
                      AnisotropicSE()) example, as inert code blocks (not executed
                      at build time -- HMC is too slow for a docs build)
  theory.md            brief method summary + citation; links into the API reference
  vle_example.md       narrative walkthrough -> examples/vle_distillation/README.md
  reproduce_paper.md   narrative walkthrough -> paper/REPRODUCTION.md +
                      paper/DATA.md's private-archive-access note
  api.rst              automodule/autoclass over the public surface
  Makefile             `make html` (`sphinx-build -W`), `make clean`
.readthedocs.yaml       RTD build config: Python 3.9, ubuntu-24.04,
                      sphinx.fail_on_warning: true, `pip install .[docs]` only
```

Key facts for the next session:

- **RTD/TensorFlow build-robustness decision: install the real stack, don't mock
  it.** Autodoc imports `bits_for_gaps`, which imports gpflow/tensorflow/tfp for the
  TF-backed modules. `pip install ".[docs]"` (what `.readthedocs.yaml` runs) already
  pulls in the pinned TF 2.16.2/GPflow 2.9.2/TFP 0.24.0 stack, because pip's extras
  are *additive* to a package's base `dependencies` -- there is no way to install
  `[docs]` without also installing the base stack. This gives autodoc a real,
  importable `AnisotropicSE` (a `gpflow.kernels.Kernel` subclass) to
  introspect -- verified directly by inspecting the rendered HTML: its page shows
  the full constructor signature and "Bases: Kernel", which mocking the import out
  would have suppressed. `docs/conf.py` documents the fallback
  (`autodoc_mock_imports = [...]`, commented out) in case a future RTD build times
  out or OOMs installing TF -- not needed today, not exercised, just ready if it
  ever is. `juliacall`/`juliapkg` are **never** imported by the docs build at all
  (no autodoc directives target `examples/vle_distillation`; that page is narrative
  only) -- no mock needed for them, no `[vle]` extra installed by
  `.readthedocs.yaml`.
- **Two real (not cosmetic-noise) issues found and fixed while verifying the
  build:** (1) MyST's `dollarmath`/`amsmath` extensions were not enabled by
  default, so `theory.md`'s `$$...$$` canonical-hyperparameter-ordering equation
  silently rendered as literal text instead of math -- fixed by adding both to
  `myst_enable_extensions` plus `sphinx.ext.mathjax`; verified the equation now
  renders as an actual MathJax block. (2) `api.rst` had a cross-reference role
  (`` :class:`~bits_for_gaps.sampler.adaptiveEntropy` ``) that got line-wrapped
  mid-identifier when first drafted, which silently failed to resolve (a literal
  newline inside the backticks becomes a space, breaking the dotted path) --
  fixed by keeping the whole role on one line. Both were caught by rendering the
  actual HTML and checking for resolved anchors/MathJax output, not just by
  `sphinx-build -W` succeeding -- **a clean warnings-as-errors build does not by
  itself prove cross-references resolved or math rendered**, because
  `nitpicky = False` (the default, kept deliberately -- see next point) doesn't
  warn on unresolved `:class:`/`:func:` targets, and a `$$...$$` block that isn't
  recognized as math just becomes an ordinary (silently valid) paragraph.
- **`nitpicky` is deliberately left `False` (i.e., unset -- Sphinx's default),
  not enabled.** Turning it on was tried as a diagnostic (`sphinx-build -D
  nitpicky=1`) and immediately produced ~40 warnings, almost all from NumPy-style
  docstring parameter types (`np.ndarray, shape (n, m)`, `array-like`, `optional`,
  `tfp.distributions.Distribution`, ...) that napoleon turns into cross-reference
  attempts with no real Sphinx target -- these aren't bugs, they're the normal
  cost of NumPy-style docstrings without a full `intersphinx` wiring to
  numpy/scipy/tensorflow/gpflow/tfp's own docs. Fixing all of them would mean
  either adding `intersphinx_mapping` (network-dependent at build time -- a
  robustness risk for RTD builds, deliberately avoided; see the "no network
  fetches" reasoning in this same decision) or extensively rewriting docstrings
  purely for Sphinx's benefit, which the task explicitly says not to do ("docs
  should surface them \[docstrings\], not rewrite them"). One genuine, low-value
  broken reference remains as a result (`` :meth:`run` `` in a couple of
  `sampler.py` docstrings, referring to an *inherited* method that isn't
  separately autodoc'd since `:inherited-members:` isn't set) -- silently
  harmless under `nitpicky = False`, left as-is rather than touched for a
  docs-only cosmetic gain.
- **The quickstart is inert code, verified against the real API, not run.** Every
  name/signature/attribute the quickstart page uses
  (`BitsForGaps(black_box=..., bounds=..., kernel=..., likelihood_variance=...)`,
  `bfg.noSamples`/`.noBurnIn`/`.noChains`, `bfg.run(X_init, y_init)`,
  `history.last`, `record.xStar`/`.max_entropy`/`.rhat`, the
  `black_box(*xStar)` calling convention with `fwd_model_args=()` by default) was
  cross-checked directly against `sampler.py`'s `BitsForGaps.__init__` and
  `state.py`'s `IterationRecord`/`RunHistory`, not run through the actual HMC loop
  (which the task explicitly says not to execute at docs-build time).
- **RTD import steps for the user** (a maintainer action requiring the user's RTD
  account -- NOT attempted this session, per the guardrail):
  1. Sign in at [readthedocs.org](https://readthedocs.org) with the GitHub account
     that owns/has admin on `dowlinglab/bits_for_gaps`.
  2. "Add project" -> import `dowlinglab/bits_for_gaps` from the connected GitHub
     account (or "Import Manually" with the repo URL if the GitHub App isn't
     installed for this org yet).
  3. RTD auto-detects `.readthedocs.yaml` at the repo root -- no further build
     config needed in the web UI; confirm the default branch (`main`) and doc
     type (Sphinx) look right on the project's Admin > Settings page.
  4. Trigger a build (either automatically on import, or "Build Version" from the
     project dashboard) and check it completes; if TF's install ever times out or
     runs out of memory on RTD's builders, switch to the `autodoc_mock_imports`
     fallback documented in `docs/conf.py` and rebuild.
  5. Optional: enable "Build pull requests for this project" under Admin >
     Settings if PR doc previews are wanted; set up a custom domain / project
     slug as desired.
- Doc-only touch-ups outside `docs/`: `README.md` gained a short "Docs" section
  (build command + pointer to `docs/index.md`); no other file outside `docs/` and
  `.readthedocs.yaml` changed.

## Phase 7 — reproduce the paper's figures (done; merged to main)

All 11 figures (2-12) regenerate from the archived published run through
`paper/reproduce.py` + `paper/figures/`. `pytest -q` (default) = **117 passed, 5
deselected** (was 117/1 -- the 4 new gated tests in `test_paper_figures.py`).
`pytest -m vle` = **5 passed, 117 deselected** (~80 s; needs the archive, 4 of the 5
also need Julia). Full details, including known discrepancies and simplifications:
`paper/REPRODUCTION.md`.

```
paper/
  __init__.py                package marker (repo-only, not in the wheel -- same
                              policy as examples/, verified via `python -m build`)
  reproduce.py                CLI entry: --archive / $BFG_ARCHIVE_DIR (default: the
                              private old-repo path), --figures (subset),
                              --out-dir (default results_remaked/, gitignored)
  REPRODUCTION.md              figure -> script -> archived-inputs -> reference-diff table
  figures/
    _archive.py               shared loaders (rhat/ess, HMC traces, param_posterior_
                              samples, activity_data, gp_predict, entropy, lhs_design,
                              cont_data, phase_diagram, gt_Wilson_data) + apply_plot_
                              settings(); verified directly against the real archive
    fig02_lhs_design.py        Fig 2  -- visual
    fig03_entropy_field.py     Fig 3  -- visual
    fig04_entropy_evolution.py Fig 4  -- visual
    fig05_parity.py            Fig 5  -- PINNED (fig5_error_metrics.json)
    fig06_gp_posterior_surface.py  Fig 6  -- visual
    fig07_gp_posterior_isotherms.py  Fig 7  -- visual
    fig08_phase_diagram.py     Fig 8  -- archive cross-check (no dedicated reference file)
    fig09_mccabe_thiele.py     Fig 9  -- PINNED (mccabe_thiele_stages.json, Phase 6);
                              wilson_column()/surrogate_column() moved here from
                              tests/regression/test_mccabe_thiele.py (Phase 6), which
                              now imports them instead of reimplementing
    fig10_traces.py            Fig 10 -- PINNED (hmc_diagnostics.json)
    fig11_marginals.py         Fig 11 -- PINNED (hyperparameter_posterior.json)
    fig12_joint_marginals.py   Fig 12 -- visual
tests/regression/test_paper_figures.py  gated (@pytest.mark.vle) recompute-vs-reference
                              for Fig 5/8/10/11 (Fig 9 already gated in Phase 6)
tests/conftest.py              + repo root on sys.path (alongside examples/, Phase 6)
                              so `import paper.figures.*` works without installing it
```

Key facts for the next session:

- **The approach is "load archive, render through the new code" -- not "re-run the
  loop."** Every figure either reads archived text/pickle files (all 11) or calls
  live into `examples/vle_distillation`'s Clapeyron-backed physics for the
  ground-truth curve (Fig 8, 9) -- none of them re-run the paper's 15-iteration
  adaptive HMC loop (stochastic, expensive, and orthogonal to "does the new code
  reproduce the published figure"). This was an explicit guardrail, not just a time-
  saving shortcut: re-running would produce a *different* (though qualitatively
  similar) stochastic realization, not a reproduction of the specific published one.
- **Packaging mirrors `examples/`'s policy exactly.** `paper/__init__.py` +
  `paper/figures/__init__.py` make it a real package; `tests/conftest.py` now also
  puts the repo root on `sys.path` (added to the existing `examples/` insert from
  Phase 6) so `import paper.figures.fig10_traces` works without installing anything.
  Wheel exclusion is automatic (hatchling's `packages = ["src/bits_for_gaps"]` is an
  allowlist -- nothing outside `src/` is ever included regardless of `__init__.py`
  presence), not separately re-verified this phase (Phase 6 already confirmed the
  mechanism with `python -m build --wheel`).
- **Quantitative pins reuse `paper/reference/*` exactly as extracted in Phase 2** --
  no new reference files were added or existing ones modified (guardrail). Fig 8 has no
  dedicated reference *file* (the paper doesn't report its curve as a scalar target);
  its regression instead cross-checks the freshly-recomputed (live Clapeyron) Wilson
  curve against the archived `gt_Wilson_data` the paper's own Fig 8 was built from --
  confirmed matching (z exactly, since both use the same `linspace(0,1,75)` grid;
  T within 0.5 K; y1 within 0.02).
- **Fig 9's recompute logic moved, not duplicated.** `wilson_column()`/
  `surrogate_column()` lived in `tests/regression/test_mccabe_thiele.py` (Phase 6);
  Phase 7 moved them into `paper/figures/fig09_mccabe_thiele.py` (since the figure
  and the test need the exact same recompute) and the test now imports them. Verified
  the gated test still passes after the move.
- **The gated `vle` marker is reused for "needs the private archive," not just
  "needs Julia."** `test_paper_figures.py`'s 4 tests are marked `@pytest.mark.vle`
  even though 3 of them (`fig10_traces`, `fig05_parity`, hyperparameter-posterior)
  never touch Julia -- what actually gates all of them is needing the archive
  directory, which is exactly as unavailable to most environments/CI as Julia is.
  Introducing a separate marker for "needs archive" seemed like unwarranted plumbing
  for a distinction without a practical difference in this repo; each test also has
  its own `skipif` on the archive directory existing, so it degrades gracefully
  (skip, not error) if pointed at a missing path.
- **Simplifications from the 847-line `fxns/mcmc_plotter.py`** (this is reproduction
  code, not a library API -- ported pragmatically, not verbatim): dropped Fig 5's
  zoomed inset and Fig 12's KDE contour overlay (both purely visual, not the
  figure's quantitative content); Fig 3 is a 2x3 grid of the first 6 iterations
  rather than 60 separate per-iteration files; Fig 12's "MAP" marker is the sample
  nearest the coordinate-wise median (a cheap visual proxy), not a true density
  mode -- use `hyperparameter_posterior.json`'s `mean`/`median` for a real point
  estimate. Full list in `paper/REPRODUCTION.md`.
- **All 11 figures were visually spot-checked** against the archived PNGs' described
  structure while building them (not just "the code runs") -- e.g. Fig 8 shows the
  expected minimum-boiling-azeotrope T-x-y diagram with the archived surrogate
  ensemble tightly tracking the freshly-recomputed Wilson dashed curve; Fig 6 shows
  visibly tighter credible-interval wireframes and denser training coverage at
  iteration 15 vs. iteration 1; Fig 4 shows the expected monotonic-ish decreasing
  max-entropy trend across all 60 archived iterations.
- **`src/bits_for_gaps/` CORE, `examples/vle_distillation/` (Phase 6 physics), and
  `paper/reference/*` are all byte-for-byte untouched** (`git diff main...HEAD --
  <path>` empty for each) -- Phase 7 only added `paper/figures/`, `paper/reproduce.py`,
  `paper/REPRODUCTION.md`, one new test file, and extended `tests/conftest.py` +
  refactored (not rewrote) `test_mccabe_thiele.py`, per the guardrails.

## Phase 6 — port the VLE/distillation example (done; merged to main)

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
  recompute matches reference's `"wilson"` column within `atol=0.015` and `"surrogate"`
  within `atol=0.05` (looser -- a non-adaptive, far-smaller-sample surrogate doesn't
  track Wilson as tightly in the most dilute region near `xW=0.01` as the paper's
  refined surrogate did; the biggest observed gap, ~0.03-0.04, is in stage 4's vapor
  fraction there). A quick empirical note: a 4-iteration adaptive run from only 15
  initial points did *worse* on this metric than the 30-point plain LHS + MLE fit
  (too little HMC/data to beat a well-covering static design) -- consistent with the
  paper's own thesis that adaptive design needs enough iterations to pay off, not
  evidence against the adaptive loop itself.
- **Reference's own `"wilson"` column has ~0.01-level transcription slop.** It was
  hand-transcribed from reading paper Fig 9c (Phase 2, `paper/reference/README.md`), not
  computed from archived data. My direct Clapeyron recompute reproduces real physical
  landmarks exactly (`stage 1 vapor == xD == 0.43` exactly, by construction; pure-PrOH
  bubble point `370.35 K` matches 1-propanol's real normal boiling point to 4
  significant figures) yet differs from reference's `"wilson"` entries by up to `0.01`
  (e.g. stage 4 liquid: recompute is exactly `xW = 0.01`, reference's transcription says
  `0.02`) -- this is the eyeballed-figure-reading precision limit of that column, not
  a bug in the port. `paper/reference/*` is unmodified (guardrail); the test's
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
- Anywhere reading a trace column back out (regression tests, `paper/reference/*`) already
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
  (`git diff main...HEAD -- <those 4 files>` is empty). `paper/reference/*` untouched;
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
paper/reference/            reference scalars from the archived iter-15 published run (+ README)
  hmc_diagnostics.json         R-hat / ESS            (Fig 10)
  fig5_error_metrics.json      train/test RMSE & MAE, iters 1 & 15, over 500 draws (Fig 5)
  hyperparameter_posterior.json  kernel-hyperparam posterior summary (Fig 10 marginals)
  mccabe_thiele_stages.json    distillation stage table, Wilson vs surrogate (Fig 9c)
tests/conftest.py        `reference` loader fixture (resolves paper/reference/)
tests/unit/              + test_kernels.py, test_means.py, test_util.py  (was entropy/design)
tests/regression/        reads reference, pins vs published paper values; vle recompute gated
tests/integration/       test_end_to_end.py — seeded synthetic (no-Julia) adaptiveEntropy run
```

Key facts for the next session:
- **Reference extraction** was done offline by `scratchpad/extract_reference.py` (pure NumPy,
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
- Fig 5 reference captures the paper's headline: median **test** RMSE falls 4.34 → 0.67
  (iter 1 → 15); train RMSE 0.77 → 0.49. The regression test pins the ≥3× test-error drop.
- Markers registered in `pyproject.toml`: `vle` (Julia backend, deselected by default),
  `slow` (integration; still runs by default). Run gated tests with `pytest -m vle`.

## What exists now (Phase 0 + Phase 3-lite + Phases 4-9d done)

A pip-installable package with the **algorithm decomposed into focused, tested modules**,
**generalized to N input dimensions**, a **ported VLE/distillation example** repo-side,
**all 11 paper figures reproduced** repo-side (now archive-free by default),
a **Sphinx/MyST/RTD docs site**, and a **from-scratch stochastic reproduction** of the
full adaptive loop (see the "Phase 4"-"Phase 9" sections above):

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
tests/unit/      entropy/design/kernels(+N-D)/means/_util/transforms/state +
                 mixture/acquisition/sampler_validation (Phase 9c: mutation-footgun
                 state-restore + input-validation error paths) + gp/
                 sampler_legacy_and_transforms/entropy_mc_validation (Phase 9d)
tests/integration/ end-to-end (2-D, in-memory run()) + BitsForGaps facade parity +
                 nd_synthetic (1-D and 3-D, via BitsForGaps)
tests/regression/  reference-file checks vs published paper values (Phase 2, 2-D only --
                 the published run and paper/reference/* are inherently 2-D); +
                 test_mccabe_thiele.py's (Phase 6) and test_paper_figures.py's
                 (Phase 7) @pytest.mark.vle recomputes
examples/vle_distillation/  the H2O-PrOH case study on the public API (Phase 6) --
                 repo-only, not in the pip wheel; see the "Phase 6" section above
examples/synthetic/  Julia-free onboarding example (Phase 9d) -- repo-only, run
                 `python examples/synthetic/run_example.py`
src/bits_for_gaps/py.typed  PEP 561 marker (Phase 9d); every module in this
                 directory is type-hinted via `from __future__ import annotations`
paper/figures/ + paper/reproduce.py  all 11 paper figures reproduced (Phase 7),
                 archive-free by default as of Phase 9 via the tracked paper/data/
                 -- repo-only, not in the pip wheel; see the "Phase 7"/"Phase 9"
                 sections above
docs/ + .readthedocs.yaml  Sphinx/MyST/furo docs (Phase 8) -- repo-only, not in
                 the pip wheel; see the "Phase 8" section above for RTD import steps
paper/data/      curated ~16 MB plot-input subset (Phase 9) -- tracked, not
                 gitignored; see paper/data/README.md
paper/full_reproduction.py + paper/phase9_validation/  from-scratch stochastic
                 reproduction of the full adaptive loop (Phase 9) -- one-time
                 validation artifact, not gated in CI; see the "Phase 9" section
                 above and paper/REPRODUCTION.md
```

`BitsForGaps` (public-API facade, REFACTOR_PLAN §4 kwarg names) and `adaptiveEntropy`
(original name, kept for backward compatibility) are both real, independently
importable classes; both work at any input dimension as of Phase 5.

## Environment (verified working)

```bash
conda env create -f environment.yml        # or reuse existing `bits_for_gaps` env
conda activate bits_for_gaps               # /opt/anaconda3/envs/bits_for_gaps
pip install -e ".[dev,vle]"                 # ,vle needed for the Julia example
pytest -q                                   # 204 passed, 2 deselected (as of Phase 9e)
ruff check .                                 # lint (Phase 9d); ruff is in [dev]
```
Stack: Python 3.9.23, gpflow 2.9.2, TF 2.16.2, TFP 0.24.0, numpy 1.26.4, scipy 1.13.1.
**macOS: `export PYTHON_JULIACALL_HANDLE_SIGNALS=yes`** before any juliacall import
(else SIGBUS). Run scripts that use bare imports with `PYTHONPATH` set to the dir.

## Where the original code + archived results live

Old repo: `~/DowlingLab/CAREER/entropy_driven_hybrid_models_code/entropy_driven_hms/`.
- Published run = `results/less_x_new_manuscript_revisions/`, **iteration 15** (R̂/ESS
  match paper Fig 10 exactly). Archived figure PNGs + data are all there. Full run
  directory is 564 MB (not 2.5 GB -- that was the old repo's whole git history).
- `driver_new.py` = active driver (NOT `driver.py`). `new_phase_diagram.py` = Fig 8.
  `run_example.py` = Fig 9. `train_test_split_proh.py` = Fig 5. `fxns/mcmc_plotter.py`
  + `fxns/plot_res.py` = figure library/CLI.
- Repro-fix already applied there: `equilibrium.water_proh_eqm_julia` now reads
  `gt_Wilson_data` (original path was missing).
- **Data decision (updated Phase 9, supersedes the old "never copy results/ here"
  line): a curated ~16 MB plot-input subset IS committed, at `paper/data/`** (see
  `paper/data/README.md` for the exact file manifest and `paper/DATA.md` for the
  full policy). The full 564 MB run stays in this private old repo (archive of
  record) -- **no Zenodo deposit** (REFACTOR_PLAN.md §7 decision 4).

## Next steps (in order — see REFACTOR_PLAN.md phases)

1. **Phase 2 — regression harness FIRST (before refactoring sampler.py). ✅ DONE.**
   Merged to `main`.

2. **Phase 4 — decompose `sampler.py`. ✅ DONE.** Merged to `main`.

3. **Phase 5 — generalize to N-D. ✅ DONE.** Merged to `main`.

4. **Phase 6 — port the VLE example. ✅ DONE.** Merged to `main`.

5. **Phase 7 — reproduce all paper figures. ✅ DONE.** Merged to `main`.

6. **Phase 8 — docs (Sphinx/RTD). ✅ DONE.** Merged to `main`. See the "Phase 8"
   section above. Actually connecting the repo on readthedocs.org is a maintainer
   action requiring the user's RTD account -- not attempted this session (see the
   numbered steps in the "Phase 8" section); `.readthedocs.yaml` is ready and waiting
   for that one manual step.

7. **Phase 9 — full stochastic reproduction + archive-free figures. ✅ DONE.**
   Merged to `main`. See the "Phase 9" section above.

8. **Phase 9b — root-cause + fix the McCabe-Thiele discrepancy. ✅ DONE.** Merged to
   `main`. See the "Phase 9b" section above.

9. **Phase 9c — robustness hardening. ✅ DONE.** Merged to `main`. See the "Phase 9c"
   section above.

10. **Phase 9d — whole-codebase polish (hygiene + faithfulness). ✅ DONE.** Merged to
   `main`. See the "Phase 9d" section above.

11. **Phase 9e — docs/tests/comments quality pass. ✅ DONE.** Merged to `main`. See
   the "Phase 9e" section above.

12. **Phase 10 — release engineering for v0.1.0. ✅ DONE.** On branch
   `phase10-release`; merge to `main` after review, then the maintainer performs the
   steps in `RELEASE.md`. See the "Phase 10" section above. No Zenodo deposit
   (REFACTOR_PLAN.md §7 decision 4 -- the private old repo is the archive of record).

13. **Actual publish (maintainer-only, not part of any automated phase).** Everything
   up to this point is prepared and documented in `RELEASE.md`: register a PyPI +
   TestPyPI trusted publisher, dry-run via `workflow_dispatch` to TestPyPI, set the
   CHANGELOG date and `git tag v0.1.0 && git push --tags` to trigger the real PyPI
   publish, then activate ReadTheDocs (import steps in the "Phase 8" section above).

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
