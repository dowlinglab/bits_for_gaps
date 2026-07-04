# Phase 9b: why the fully-adaptive surrogate's McCabe-Thiele column didn't converge

Phase 9's from-scratch stochastic reproduction (`paper/REPRODUCTION.md`) documented one
discrepancy: the genuinely 15-iteration-adaptive surrogate GP's McCabe-Thiele column
didn't converge, unlike the paper's own Fig 9b surrogate column. The write-up attributed
this to a property of entropy-driven acquisition ("optimizes for GP predictive accuracy
at held-out points, not for a globally smooth-enough equilibrium curve"). **That
attribution was wrong.** The actual cause is a shared-mutable-state bug in
`paper/full_reproduction.py` itself, unrelated to entropy-driven design, the GP, or the
adaptive loop. This investigation found the bug, confirmed it by exactly reproducing the
original failure, fixed it, and re-ran the full stochastic loop to confirm the fix.

## Root cause

`paper/full_reproduction.py`'s original code computed the test-RMSE curve (via
`_predict_split` → `bits_for_gaps.mixture.sample_gp_posterior_mixture`) **before**
building the Fig 8/9-style surrogate phase diagram, and used the same in-memory
`GPmodel` object (`history.last.GPmodel`) for both. `sample_gp_posterior_mixture`
mutates `GPmodel.kernel` in place, by design (see its own docstring: "Mutated in
place: its kernel hyperparameters are reassigned for every draw") -- it's meant to be
called on a model you're about to discard or intentionally leave at that state. The
RMSE step called it ~30 times (once per HMC iteration, plus once more for the final
iteration's held-out train draws), each time reassigning `GPmodel.kernel` to a fresh
random draw from the HMC posterior and leaving it there. By the time the phase-diagram
code ran afterward and called `GPmodel.predict_f`, the kernel held whatever the *last*
of those ~30 unrelated draws happened to be -- not any principled state, and not what
the write-up assumed ("the final, genuinely 15-iteration adaptively-trained GP").

### Confirmation

Reconstructing the iteration-15 GP from its checkpointed `gp_model_15.pkl` and
replaying `full_reproduction.py`'s exact original call sequence (RMSE loop, seed=10,
then phase diagram) reproduced the reported failure to 4 decimal places:

```
hyperparams BEFORE any mutation: [1.6508907124750056, 1.0349426523888388, 4.2802100682522735]
hyperparams AFTER the mutation sequence: [2.0146229510942004, 0.7560132676044007, 2.4816506654063213]
converged: False
  stage 2: liquid=1.7258 vapor=0.3776   # full_run_summary.json (pre-fix) reported 1.725805727168204
```

That match is exact, not approximate -- this is the bug, not a contributing factor.

## Hypotheses: what held, what didn't

The task posed four hypotheses before this root cause was known. None of them turned
out to be it, but testing them anyway produced useful, honest findings -- reported here
rather than discarded.

**Setup**: five equilibrium curves were reconstructed over the same 50-point z-grid from
the Phase 9 run's checkpointed `gp_model_15.pkl` (24 training points) and
`param_posterior_samples_15` (the HMC trace, chain 0, constrained):

- **(a) fresh, HMC-as-left** -- `surrogate_gamma` using the kernel state exactly as
  `bits_for_gaps.gp.run_mcmc` left it (a single, un-mutated posterior sample).
- **(b) paper archive mean** -- pointwise mean of `paper/data/phase_diagram_15`'s y1
  column over its 50 draws, replicating `equilibrium.py`'s `water_proh_eqm` exactly.
- **(c) fresh, paper-method 50-draw average** -- 50 independent single-hyperparameter
  draws from the same trace, GP mean under each, averaged pointwise (the paper's own
  `new_phase_diagram.py` construction).
- **(d) fresh, posterior-mean hyperparameters** -- kernel set once to
  `trace.mean(axis=0)`, single deterministic curve.
- **(e) Wilson** -- ground truth, for reference.

| curve | monotonicity violations | roughness | column converged |
|---|---|---|---|
| (a) fresh, HMC-as-left | 0 | 0.00666 | **True** |
| (b) paper archive mean | 0 | 0.00343 | True |
| (c) fresh, 50-draw avg | 0 | 0.00650 | True |
| (d) fresh, mean-hyperparameters | 0 | 0.00661 | **False** |
| (e) Wilson | 0 | 0.00640 | True |

(`paper/phase9_validation/phase9b_curve_comparison.png` plots all five together, plus a
zoom on the dilute region where the column's stages concentrate.)

- **H1 (draw count/averaging)** -- the paper's method, replicated exactly on the fresh
  surrogate (c), does converge and tracks Wilson closely. **Confirmed as a good
  robustness practice** -- but note (a), a single un-mutated draw, *also* converged, so
  averaging was not *necessary* to fix the reported failure; the bug was. Averaging is
  adopted anyway (see "Fix" below) because it protects against exactly this class of
  fragility: any single hyperparameter draw, including a good one, is one accidental
  mutation away from being an arbitrary one.
- **H2 (monotonicity)** -- **rejected outright**. All five curves, including the one
  that failed to converge (d), have zero monotonicity violations. Whatever made (d) fail
  is not curve roughness or non-monotonicity.
- **H3 (deterministic mean, "likely the clean fix")** -- **falsified**. The
  posterior-mean-hyperparameters curve (d) is exactly as smooth as the others (roughness
  0.00661 vs. (a)'s 0.00666) yet is the *only* one of the five that doesn't converge.
  The arithmetic mean of each hyperparameter's marginal posterior is not a
  self-consistent joint point (`std_dev`/`lengthscale_1` are LogNormal-ish, positive;
  `lengthscale_2` is intentionally unconstrained -- see `kernels.py`), so it can land the
  curve's absolute shape somewhere `scipy.optimize.fsolve`'s fixed initial guess in
  `distillation.solve_column` can't reach a physical root from -- a solver
  initial-guess-sensitivity issue, not a smoothness issue. Using the posterior *mean* is
  not a safe substitute for a real posterior draw or a Monte-Carlo average over draws.
- **H4 (genuine under-resolution)** -- **rejected**. The un-mutated 24-point adaptive
  surrogate (a) converges and matches Wilson about as well as the paper's own
  archived-mean surrogate (b) does. Entropy-driven acquisition at this iteration count
  is not under-resolving the equilibrium curve; there is no methodological limitation to
  report here for this system. Phase 9's original framing of this as an entropy-driven-
  design-vs-space-filling-design tradeoff was an incorrect post-hoc explanation for a
  bug, not a real finding -- retracted below.

## Fix

Two changes, both in the **example layer** (`examples/vle_distillation/`,
`paper/full_reproduction.py`) -- `src/bits_for_gaps/` core is untouched, including
`mixture.sample_gp_posterior_mixture`'s in-place-mutation contract, which is correct and
documented as designed:

1. **`paper/full_reproduction.py`**: build the phase diagram/column from
   `history.last.GPmodel` *before* running the test-RMSE loop that mutates it, with a
   comment explaining why the order matters. This alone fixes the reported bug.
2. **`examples/vle_distillation/phase_diagram.py`**: added `surrogate_gamma_averaged`,
   matching the paper's own `new_phase_diagram.py`/`equilibrium.py` construction --
   draws `n_draws` (default 50) independent samples from a supplied HMC trace,
   evaluates this GP's deterministic conditional mean (`predict_f`, not
   `predict_f_samples` -- no TF-ambient-RNG non-reproducibility) under each, and
   averages `gamma_proh` pointwise. `full_reproduction.py` now uses this (not the
   single-point-estimate `surrogate_gamma`) for its phase diagram, for the added
   robustness H1 confirmed -- not because the point estimate doesn't work, but because
   a Monte-Carlo average over the posterior is inherently more robust to *any* single
   draw (mutated-into or genuinely unlucky) being atypical. `surrogate_gamma`'s
   docstring now documents the mutation hazard explicitly for future callers.
   `fig09_mccabe_thiele.py`'s non-adaptive, MLE-fit surrogate (no HMC trace, hence no
   posterior to average over) is unaffected and still uses `surrogate_gamma`, correctly.

## Result after the fix

Re-ran the full 15-iteration adaptive loop from scratch (fresh seed=10 run, ~26 min;
`results_remaked/phase9b_fullrun_fixed/`, gitignored) with the fixed
`paper/full_reproduction.py`:

```
column_wilson_converged:    True
column_surrogate_converged: True   # was False before the fix

  stage 1: wilson liq=0.2265 vap=0.4300 | surrogate liq=0.2537 vap=0.4300
  stage 2: wilson liq=0.0473 vap=0.3282 | surrogate liq=0.0433 vap=0.3419
  stage 3: wilson liq=0.0216 vap=0.2386 | surrogate liq=0.0197 vap=0.2366
  stage 4: wilson liq=0.0100 vap=0.1399 | surrogate liq=0.0100 vap=0.1435
```

The genuinely-adaptive surrogate's column now converges and tracks the Wilson
ground-truth column within 0.03 mole fraction (liquid) / 0.014 (vapor) at every stage --
the "satisfying full reproduction" this investigation set out to find. HMC diagnostics
(R-hat, ESS) and the hyperparameter posterior on this fresh run again matched the
originally-reported values to 6-8 significant figures (same near-bit reproducibility
Phase 9 established), confirming the fix changed only the phase-diagram/column
construction, not the adaptive loop itself. The test-RMSE curve differs slightly from
the original run's (4.75 vs. 4.34 at iteration 1) for the same already-documented reason
as Phase 9's other test-RMSE discrepancy: `predict_f_samples`'s non-seedable TF ambient
RNG.

Updated artifacts: `paper/phase9_validation/full_run_summary.json` (now the fixed run;
the original is preserved as `full_run_summary_pre_phase9b_fix.json` for the record),
`phase_diagram_fresh.png` (regenerated, now shows the converging surrogate), and
`phase9b_curve_comparison.png` (the five-curve diagnostic above). See
`paper/REPRODUCTION.md`'s Phase 9 section for the corrected discrepancy note.

## Takeaway

Not a methodological finding about entropy-driven design after all -- a straightforward
shared-mutable-state bug in a one-off analysis script, caught by refusing to accept the
first plausible-sounding explanation and instead reconstructing the failure from first
principles until it reproduced exactly. Worth remembering for any future script that
reuses a fitted model object across multiple purposes: check whether the functions
touching it document mutation, and if so, order calls (or copy the model) accordingly.
