# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/); versioning
follows [Semantic Versioning](https://semver.org/). `v0.1.0` is the first release; its
entry below summarizes the library relative to the paper's original research code. See
`docs/improvements_over_paper.md` for a detailed comparison against the original code.

## [Unreleased]

Nothing yet.

## [0.1.1] - 2026-08-06

Bug-fix release. **Users on setuptools >= 81 should upgrade from 0.1.0.**

### Fixed

- **`import` of every TensorFlow-backed module failed on setuptools >= 81** with
  `ModuleNotFoundError: No module named 'pkg_resources'`. GPflow 2.9.2 does
  `import pkg_resources` at import time (`gpflow/versions.py`), and setuptools 81
  removed `pkg_resources`; setuptools 82.0.1 is the current default in a fresh
  environment. In 0.1.0 this left `AnisotropicSE`, `FixedInverseMean`, `BitsForGaps`,
  `adaptiveEntropy`, and the `gp`/`mixture`/`acquisition` modules unimportable for new
  users, while the pure NumPy/SciPy parts (`entropy`, `design`, `transforms`) kept
  working. Fixed by adding a `setuptools<81` bound to the runtime dependencies
  (setuptools is already a runtime dependency of GPflow; this adds the upper bound that
  setuptools' own deprecation warning recommends). Revisit when the frozen GPflow/TF
  stack is modernized.

  Why 0.1.0's release checks missed it: the clean-environment install audit exercised
  only the eagerly-imported pure modules, so the lazy `__getattr__` never loaded GPflow.
  `RELEASE.md`'s audit step now requires importing a TF-backed module too.

## [0.1.0] - 2026-08-06

The first release.

### Added

- Public API: `BitsForGaps` (friendly-named facade), `adaptiveEntropy` (original
  name, kept for backward compatibility), `AnisotropicSE`, `FixedInverseMean`,
  `InputTransform`/`OutputTransform`, `latin_hypercube_design`/`full_factorial_design`,
  `second_order_entropy`/`entropy_lower_bound`/`gaussian_mixture_density`.
- N-D generalization: the acquisition path a run actually depends on (`optimize`)
  works at any input dimension; proven at 1-D and 3-D, not just the paper's 2-D case.
- A selectable acquisition objective: `BitsForGaps.acquisitionObjective` (or
  `objective=` on `acquisition.entropy_objective`/`optimize`) chooses between the
  paper's 2nd-order Taylor entropy approximation (`"taylor"`, default) and its
  closed-form lower bound (`"lower_bound"`), now usable as an acquisition choice.
- Public-API input validation: clear `ValueError`s for bounds/kernel dimensionality
  mismatches, invalid bounds, non-positive HMC/acquisition config, `X_init`/`y_init`
  shape mismatches, and a malformed injected black-box output.
- An opt-in `tf_seed` on `mixture.sample_gp_posterior_mixture`/`predict_grid_2D` to
  make `predict_f_samples`' otherwise TF-ambient-RNG-driven draws reproducible on
  request.
- `py.typed` (PEP 561) and type hints across `src/bits_for_gaps/`.
- Archive-free figure reproduction: a curated ~16 MB subset of the published run's
  plot-input data, committed to `paper/data/`, so `python paper/reproduce.py`
  regenerates every figure from a fresh clone with no private-archive access.
- `examples/vle_distillation/`: the paper's H2O-PrOH VLE/distillation case study,
  built on the public API (Julia/Clapeyron-backed, repo-only).
- `examples/synthetic/run_example.py`: a small, Julia-free, runnable example for
  onboarding.
- Sphinx/MyST documentation, a CI workflow running the full default test suite plus
  `ruff check`, and this changelog.
- `docs/theory.md` presents the paper's key equations in its own notation with
  equation numbers, each linked to the implementing module.
- Release engineering: `.github/workflows/publish.yml` publishes to PyPI (on a `v*`
  tag) or TestPyPI (manual dry run) via OIDC trusted publishing -- no API tokens in
  the repo; see `RELEASE.md` for the maintainer checklist.

### Changed

- Decomposed the sequential-design engine into focused, independently testable
  modules: `gp`, `mixture`, `acquisition`, `entropy`, `transforms`, `state`, with
  `sampler.py`'s `adaptiveEntropy` reduced to a thin orchestrator.
- Retired disk-as-state: `run()` takes the initial design in memory and returns a
  `RunHistory`; a full run executes with zero disk writes by default, per-iteration
  file output is available but opt-in (`checkpoint_dir`).
- `mixture.sample_gp_posterior_mixture`/`acquisition.entropy_objective` now save and
  restore a GP's kernel hyperparameters around their internal reassignment loop, so
  the caller's model is not left at an arbitrary leftover state afterward.
- `examples/vle_distillation/distillation.py`'s `solve_column` retries a few generic
  alternate `fsolve` initial guesses if the default doesn't converge, before giving
  up -- the primary attempt is unchanged, so this only ever activates when it was
  already failing.
- Whole-repo `ruff` lint + format pass, preserving the load-bearing import orders
  (`__init__.py`'s lazy-import split; `PYTHON_JULIACALL_HANDLE_SIGNALS` set before
  the `juliacall` import it protects).

### Fixed

- A missing data path in the original `equilibrium.py` (`water_proh_eqm_julia` read a
  file that didn't exist) -- repointed to the archived Wilson ground truth the rest
  of the pipeline already produces.
- A shared-mutable-state bug that produced a spurious "finding": a validation script
  reused a `GPmodel` for a second purpose after an earlier step had mutated its
  kernel and left it at an arbitrary state, producing a non-converging McCabe-Thiele
  column that a first write-up incorrectly attributed to a property of entropy-driven
  acquisition design. The underlying footgun was then hardened away at its source
  (see "Changed", above).
- `entropy.py`'s density-positivity check promoted from a bare `assert` (silently
  stripped under `python -O`) to an explicit `ValueError`.
- `kernels.assign_hyperparameters` now raises a clear, specific error (naming the
  parameter and value) instead of a low-level gpflow/TensorFlow traceback when a
  value can't round-trip through a parameter's transform -- e.g. an extreme outlier
  posterior sample for `lengthscale_2`, deliberately left unconstrained.

### Testing

- A numerical baseline pin (`tests/integration/data/synthetic_baseline.json`,
  atol 1e-10) and reference scalars extracted from the published run
  (`paper/reference/*`) that the library must reproduce exactly -- the original code
  had no automated tests at all.
- Coverage raised to 97%, closing real gaps (`gp.py` and `diagnostics.py` had zero
  direct unit tests before; several documented `ValueError` paths were never
  exercised).
- A Monte-Carlo validation of the entropy approximations against the true GMM
  differential entropy they approximate, not just a captured historical value.
- 204 tests passing.
