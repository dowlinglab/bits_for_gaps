# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/); versioning
follows [Semantic Versioning](https://semver.org/). `v0.1.0` (not yet tagged -- see
`RELEASE.md`) is the first release; its entry below summarizes everything since the
package's extraction from the paper's research code (`entropy_driven_hybrid_models_code`)
began. See `HANDOFF.md` for the full phase-by-phase narrative and
`docs/improvements_over_paper.md` for a detailed comparison against the original code.

## [Unreleased]

Nothing yet -- `v0.1.0` below is the first release.

## [0.1.0] - YYYY-MM-DD

The first release. `YYYY-MM-DD` is a placeholder -- the maintainer sets the actual
release date at tag time (see `RELEASE.md`).

### Added

- Public API: `BitsForGaps` (friendly-named facade), `adaptiveEntropy` (original
  name, kept for backward compatibility), `AnisotropicSE`, `FixedInverseMean`,
  `InputTransform`/`OutputTransform`, `latin_hypercube_design`/`full_factorial_design`,
  `second_order_entropy`/`entropy_lower_bound`/`gaussian_mixture_density`.
- N-D generalization (Phase 5): the acquisition path a run actually depends on
  (`optimize`) works at any input dimension; proven at 1-D and 3-D, not just the
  paper's 2-D case.
- A selectable acquisition objective (Phase 9d): `BitsForGaps.acquisitionObjective`
  (or `objective=` on `acquisition.entropy_objective`/`optimize`) chooses between the
  paper's 2nd-order Taylor entropy approximation (`"taylor"`, default) and its
  closed-form lower bound (`"lower_bound"`, implemented since Phase 2 but never
  wired up as a usable choice before now).
- Public-API input validation (Phase 9c): clear `ValueError`s for bounds/kernel
  dimensionality mismatches, invalid bounds, non-positive HMC/acquisition config,
  `X_init`/`y_init` shape mismatches, and a malformed injected black-box output.
- An opt-in `tf_seed` on `mixture.sample_gp_posterior_mixture`/`predict_grid_2D`
  (Phase 9c) to make `predict_f_samples`' otherwise TF-ambient-RNG-driven draws
  reproducible on request.
- `py.typed` (PEP 561) and type hints across `src/bits_for_gaps/` (Phase 9d).
- Archive-free figure reproduction (Phase 9): a curated ~16 MB subset of the
  published run's plot-input data, committed to `paper/data/`, so
  `python paper/reproduce.py` regenerates every figure from a fresh clone with no
  private-archive access.
- `examples/vle_distillation/` (Phase 6): the paper's H2O-PrOH VLE/distillation case
  study, ported onto the public API (Julia/Clapeyron-backed, repo-only).
- `examples/synthetic/run_example.py` (Phase 9d): a small, Julia-free, runnable
  example for onboarding.
- Sphinx/MyST documentation (Phase 8), a CI workflow running the full default test
  suite plus `ruff check` (Phase 9d), and this changelog.
- `docs/theory.md` rewritten to present the paper's key equations in its own notation
  with equation numbers, each linked to the implementing module (Phase 9e).
- Release engineering (Phase 10): `.github/workflows/publish.yml` publishes to PyPI
  (on a `v*` tag) or TestPyPI (manual dry run) via OIDC trusted publishing -- no API
  tokens in the repo; see `RELEASE.md` for the maintainer checklist.

### Changed

- Decomposed the paper's monolithic `driver_new.py` into focused, independently
  testable modules (Phase 4): `gp`, `mixture`, `acquisition`, `entropy`, `transforms`,
  `state`, with `sampler.py`'s `adaptiveEntropy` reduced to a thin orchestrator.
- Retired disk-as-state (Phase 4): `run()` takes the initial design in memory and
  returns a `RunHistory`; a full run executes with zero disk writes by default,
  per-iteration file output is available but opt-in (`checkpoint_dir`).
- `mixture.sample_gp_posterior_mixture`/`acquisition.entropy_objective` now save and
  restore a GP's kernel hyperparameters around their internal reassignment loop
  (Phase 9c) -- behavior-preserving for every value either function returns, but the
  caller's model is no longer left at an arbitrary leftover state afterward.
- `examples/vle_distillation/distillation.py`'s `solve_column` retries a few generic
  alternate `fsolve` initial guesses if the default doesn't converge, before giving
  up (Phase 9c) -- the primary attempt is unchanged, so this only ever activates when
  it was already failing.
- Whole-repo `ruff` lint + format pass (Phase 9d), preserving the load-bearing import
  orders (`__init__.py`'s lazy-import split; `PYTHON_JULIACALL_HANDLE_SIGNALS` set
  before the `juliacall` import it protects).

### Fixed

- A missing data path in the original `equilibrium.py` (`water_proh_eqm_julia` read a
  file that didn't exist) -- repointed to the archived Wilson ground truth the rest
  of the pipeline already produces.
- A shared-mutable-state bug that produced a spurious "finding": a validation script
  reused a `GPmodel` for a second purpose after an earlier step had mutated its
  kernel and left it at an arbitrary state, producing a non-converging McCabe-Thiele
  column that a first write-up incorrectly attributed to a property of entropy-driven
  acquisition design (Phase 9b; see `paper/PHASE9B_INVESTIGATION.md`). The underlying
  footgun was then hardened away at its source (Phase 9c, above).
- `entropy.py`'s density-positivity check promoted from a bare `assert` (silently
  stripped under `python -O`) to an explicit `ValueError` (Phase 9c).
- `kernels.assign_hyperparameters` now raises a clear, specific error (naming the
  parameter and value) instead of a low-level gpflow/TensorFlow traceback when a
  value can't round-trip through a parameter's transform -- e.g. an extreme outlier
  posterior sample for `lengthscale_2`, deliberately left unconstrained (Phase 9d).
- A stale doc claim: `docs/reproduce_paper.md` and `docs/theory.md` described
  pre-Phase-9 behavior (archive access required; the entropy lower bound "not wired
  into the default acquisition path") after both had changed.

### Test / process

- A pre-refactor numerical baseline pin (`tests/integration/data/synthetic_baseline.json`,
  atol 1e-10) and reference scalars extracted from the published run (`paper/reference/*`)
  that every subsequent phase must reproduce exactly -- the original code had no
  automated tests at all.
- Coverage raised from 93% to 97% (Phase 9d), closing real gaps (`gp.py` had zero
  direct unit tests before).
- A Monte-Carlo validation of the entropy approximations against the true GMM
  differential entropy they approximate, not just a captured historical value
  (Phase 9d).
- 204 tests passing (up from 166 at Phase 9d's start), closing coverage gaps found in
  a function-by-function audit (`diagnostics.py` had zero direct unit tests; several
  documented `ValueError` paths were never exercised) and upgrading a few shape-only
  assertions to check actual values (Phase 9e).
