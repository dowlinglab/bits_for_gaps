# Theory

This page summarizes the method well enough to read the API reference with context.
For derivations, proofs, and the H2O-PrOH case study's full results, see the paper:

> K. D. Jones and A. W. Dowling, "BITS for GAPS: Bayesian Information-Theoretic
> Sampling for hierarchical GAussian Process Surrogates," *Computers & Chemical
> Engineering* **211** (2026) 109650.
> [doi:10.1016/j.compchemeng.2026.109650](https://doi.org/10.1016/j.compchemeng.2026.109650)

## The loop

BITS for GAPS is a sequential experimental-design loop over a **hierarchical**
Gaussian-process surrogate ({func}`bits_for_gaps.sampler.adaptiveEntropy.run`):

1. **Initialize** -- a space-filling design ({mod}`bits_for_gaps.design`) over the
   input space; a GP prior on the black-box output ({mod}`bits_for_gaps.kernels`,
   {mod}`bits_for_gaps.means`); *priors on the GP's own hyperparameters* (the
   "hierarchical" part).
2. **Calibrate** -- evaluate the black box at the design points; run Hamiltonian
   Monte Carlo ({mod}`bits_for_gaps.gp`) to sample the hyperparameter posterior
   $p(\theta \mid \mathbf{y})$; propagate a subset of posterior draws through the GP
   to form a **Gaussian-mixture predictive posterior** ({mod}`bits_for_gaps.mixture`).
3. **Evaluate** -- score candidate points by the mixture's predictive differential
   entropy ({mod}`bits_for_gaps.entropy`, {mod}`bits_for_gaps.acquisition`); pick the
   maximizer as the next point. Repeat.

## Why hierarchical

A plain (non-hierarchical) GP conditions on a *point estimate* of its
hyperparameters (e.g. the MLE) and reports predictive variance around that one
setting. That variance says nothing about how much the data actually constrains the
hyperparameters themselves -- with few observations, very different hyperparameter
settings can fit the data almost equally well, and a plain GP's predictive variance
is blind to that. BITS for GAPS instead puts priors on the hyperparameters and
samples their posterior via HMC ({func}`bits_for_gaps.gp.run_mcmc`); the predictive
distribution becomes a mixture over posterior draws
({func}`bits_for_gaps.mixture.sample_gp_posterior_mixture`), so its spread reflects
*both* observation noise and hyperparameter uncertainty. Early in a design, when
hyperparameters are poorly constrained, this mixture is visibly wider than any single
draw -- which is exactly the situation where more targeted data collection helps most.

## Canonical hyperparameter ordering

Each kernel hyperparameter (the output scale and one lengthscale per input
dimension) is its own `gpflow.Parameter`, each carrying its own prior -- deliberately
not a single vector-valued parameter, because different dimensions can need
different prior *families* (see {class}`bits_for_gaps.kernels.AnisotropicSE`'s
paper-matching default: lognormal, lognormal, and gamma priors across its three
hyperparameters, one of them left deliberately unconstrained). The kernel exposes
these in a single canonical order via
{attr}`~bits_for_gaps.kernels.AnisotropicSE.hyperparameters`:

$$[\sigma, \ell_1, \ell_2, \ldots, \ell_d]$$

HMC samples in exactly this order ({func}`bits_for_gaps.gp.run_mcmc`), and every
downstream consumer -- the mixture sampler, the acquisition function -- replays a
posterior draw back onto the kernel via
{func}`bits_for_gaps.kernels.assign_hyperparameters`, which walks the same order.
This one contract is what makes the hierarchical machinery dimension-general instead
of hardcoded to the paper's 2-D, 3-hyperparameter case.

## Entropy estimators

The acquisition function maximizes the differential entropy of the Gaussian-mixture
predictive posterior at a candidate point. The exact entropy of a Gaussian mixture
has no closed form, so `bits_for_gaps.entropy` provides two estimators:

- {func}`~bits_for_gaps.entropy.second_order_entropy` -- a second-order Taylor
  approximation (Huber et al., 2008) around each mixture component. This is the
  **default** estimator {func}`bits_for_gaps.acquisition.entropy_objective`
  maximizes (`objective="taylor"`) -- the one that drove acquisition in the paper.
- {func}`~bits_for_gaps.entropy.entropy_lower_bound` -- a closed-form lower bound
  on the true mixture entropy (cheaper, less tight). Selectable as the acquisition
  objective via `objective="lower_bound"` (or, on `BitsForGaps`/`adaptiveEntropy`,
  `.acquisitionObjective = "lower_bound"`) -- implemented since Phase 2 but only
  wired up as a usable acquisition choice in Phase 9d.

Both reduce to the exact differential entropy of a single Gaussian in the
degenerate one-component case (see `tests/unit/test_entropy.py`, which checks this
directly against the analytic $\tfrac{1}{2}\log\big((2\pi e)^d |\Sigma|\big)$ formula).

## What's dimension-general, and what's deliberately not

The core sequential-design loop, the entropy math, and the N-D acquisition optimizer
({func}`bits_for_gaps.acquisition.optimize`) work at any input dimension $d$ -- proven
by `tests/integration/test_nd_synthetic.py` at $d=1$ and $d=3$, not just the paper's
$d=2$. Two purely-visual diagnostics stay 2-D-only on purpose, since a dense grid is
exponential in $d$ and neither feeds the acquisition:
{func}`bits_for_gaps.acquisition.entropy_surface_2D` (the entropy field over a 2-D
grid) and {func}`bits_for_gaps.mixture.predict_grid_2D` (full-grid GP posterior
samples for plotting). Both raise a clear error outside $d=2$ rather than silently
producing something wrong.
