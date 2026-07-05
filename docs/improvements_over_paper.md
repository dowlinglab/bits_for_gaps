# Improvements over the original paper code

This package is a from-scratch port of the code behind Jones & Dowling (2026), not a
copy. Every phase of the port (see `HANDOFF.md` for the full history) fixed something,
made something more robust, or restructured something for maintainability. This page
consolidates those changes honestly and specifically -- real diffs, not marketing.

```{note}
This is a pointer to real changes, each with more detail in the repository:
`HANDOFF.md` (phase-by-phase history), `paper/PHASE9B_INVESTIGATION.md` (the bug fix
below), and the module docstrings cited throughout (e.g. `mixture.py`,
`acquisition.py`, `sampler.py`).
```

## Bugs found and fixed

**A missing data path, silently wrong until traced down (pre-Phase-1).** The original
`equilibrium.py`'s `water_proh_eqm_julia` read from a file path
(`results/less_x/phase_diagram_data_saftgammamie`) that didn't exist on disk or in the
old repo's git history. Repointed to the archived Wilson ground truth
(`gt_Wilson_data`) that the rest of the pipeline already produces and uses.

**A shared-mutable-state bug that produced a spurious "finding" (Phase 9b).** A
validation script (`paper/full_reproduction.py`) computed a test-RMSE metric via
`mixture.sample_gp_posterior_mixture` -- which reassigns a GP's kernel hyperparameters
once per posterior draw, by design -- then reused that *same* GP object afterward to
build a McCabe-Thiele surrogate column. The column didn't converge, and the first
write-up of this attributed that to a property of entropy-driven acquisition design.
That attribution was wrong: reconstructing the failure from the checkpointed model
reproduced it to 4 decimal places, and the real cause was the leftover mutated kernel
state, not the design method. Full analysis, including which of four hypotheses held
and which didn't, in `paper/PHASE9B_INVESTIGATION.md`. Fixed by reordering the script
and adding a hyperparameter-posterior-averaged surrogate construction; the same
underlying footgun (see below) was then hardened away at its source.

## Hardening (Phase 9c) -- the same class of bugs made structurally harder to hit

The Phase 9b bug above was a symptom of a real footgun in the library itself, not just
a one-off script mistake:

- **State-mutation footgun, fixed at the source.** `mixture.sample_gp_posterior_mixture`
  and `acquisition.entropy_objective` both reassign a GP's kernel hyperparameters in a
  loop -- and both used to leave the kernel at whichever draw happened to run last,
  not any meaningful state. Since `sampler.py`'s `run()` calls these on the *same*
  `GPmodel` object it then stores in every `IterationRecord` (and optionally
  checkpoints to disk), **every run's returned model** used to carry this arbitrary
  leftover state, not just the one script Phase 9b happened to hit. Both functions now
  save the kernel's hyperparameters before mutating and restore them in a `finally` --
  behavior-preserving for every value either function computes and returns, verified
  against the pre-Phase-4 baseline (atol 1e-10) and all golden regressions.
- **Public-API input validation.** Constructing `BitsForGaps`/`adaptiveEntropy` with
  mismatched bounds/kernel dimensionality, `lo >= hi` bounds, or calling `run()` with
  non-positive HMC/acquisition config, mismatched `X_init`/`y_init` shapes, or an
  injected black box that returns something that isn't a non-empty sequence, used to
  fail deep inside GPflow/TensorFlow with a cryptic shape error (or an `IndexError`
  with no context) the first time something touched the bad value. All of these now
  raise a clear `ValueError` immediately, before any computation starts.
- **`assert` promoted to an explicit exception.** `entropy.py`'s density-positivity
  check was a bare `assert`, which Python silently strips under `python -O` -- meaning
  the exact scenario the check exists to catch (a degenerate mixture) would have
  silently propagated a `NaN`/garbage entropy value in an optimized build instead of
  failing. Now an explicit `ValueError` with a specific message.
- **Fragile `fsolve` convergence, given a fallback.** The McCabe-Thiele column solver
  (`examples/vle_distillation/distillation.py`) has no bounds and used a single fixed
  initial guess -- exactly the kind of solver fragility Phase 9b's investigation
  surfaced (a smooth, well-behaved equilibrium curve can still fail to converge from an
  unlucky initial guess). `solve_column` now retries a few generic alternate initial
  guesses if the default doesn't converge, before giving up. The primary attempt is
  byte-for-byte unchanged, so nothing that already converges is affected.
- **Opt-in reproducibility for the one documented non-reproducible step.**
  `GPmodel.predict_f_samples` (used by `sample_gp_posterior_mixture`/`predict_grid_2D`)
  draws from TensorFlow's ambient RNG, not NumPy's -- confirmed non-reproducible even
  within the same process, in the original paper code too. Both functions now accept
  an optional `tf_seed`; passing one makes that call's draws reproducible. Default
  (`None`) leaves the original, documented behavior unchanged.

46 new tests were added alongside this hardening (unit tests asserting kernel state is
identical before/after, including on the error path; validation-error-path tests; an
integration test reproducing the exact Phase 9b scenario end-to-end) -- all in addition
to, not replacing, the existing regression suite, which stayed green throughout.

## Architecture

- **Decomposed into focused, independently-testable modules** (Phase 4). The original
  `driver_new.py` was a single ~450-line class mixing GP construction, HMC, entropy
  math, acquisition optimization, and disk I/O. It's now
  {mod}`bits_for_gaps.gp`/{mod}`bits_for_gaps.mixture`/{mod}`bits_for_gaps.acquisition`/
  {mod}`bits_for_gaps.entropy`/{mod}`bits_for_gaps.transforms`/{mod}`bits_for_gaps.state`,
  each independently unit-tested, with `sampler.py`'s `adaptiveEntropy` reduced to a
  thin orchestrator over them.
- **In-memory state, checkpointing opt-in** (Phase 4). The original code used disk as
  its state-passing mechanism between iterations (`np.savetxt`/`pickle` under
  `results/{exp_name}/`, read back on the next call). `run()` now takes the initial
  design in memory and returns a `RunHistory` -- a full run executes with zero disk
  writes by default; per-iteration file output (mirroring the original layout) is
  available via an opt-in `checkpoint_dir` argument.
- **Generalized to N input dimensions** (Phase 5). The original kernel, acquisition,
  and mixture code hardcoded the 2-D VLE case (e.g. indexing `trainable_parameters` by
  position, a reversed 2-D-specific black-box calling convention). The acquisition
  path an actual run depends on (`optimize`, was `optimize_2D`) is now dimension-
  general; only the dense-grid 2-D-only visualization diagnostics (`entropy_surface_2D`,
  `predict_grid_2D`) stay 2-D (a dense grid is exponential in dimension, and neither
  feeds the acquisition) -- they raise a clear error for other dimensions rather than
  silently misbehaving.
- **Julia is an opt-in, lazily-imported extra** ({mod}`bits_for_gaps` core). The
  published run's activity-coefficient black box needs Julia + Clapeyron.jl; the
  sequential-design algorithm itself does not. `import bits_for_gaps` is Julia-free
  (verified: `pip install bits_for_gaps` pulls GPflow/TensorFlow/NumPy/SciPy only);
  `juliacall` is imported lazily, only when the VLE example's activity model is
  actually called, and only if you installed the `[vle]` extra.
- **Archive-free figure reproduction** (Phase 9). The original figures could only be
  regenerated with author access to the private, 564 MB archived run. A curated ~16 MB
  subset of exactly the files the figures read is committed to `paper/data/`
  (provenance in `paper/data/README.md`), so `python paper/reproduce.py` reproduces
  every figure from a fresh clone with no private-archive access.
- **A real regression/golden-file test suite** (Phase 2 onward). The original code had
  no automated tests. This package pins a pre-refactor numerical baseline
  (`tests/integration/data/synthetic_baseline.json`, atol 1e-10) plus golden scalars
  extracted from the published run (`paper/golden/*`) that every subsequent phase --
  including this hardening pass -- must reproduce exactly, with tests gated behind
  `@pytest.mark.vle` only where they genuinely need Julia or private data.
