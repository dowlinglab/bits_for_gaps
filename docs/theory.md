# Theory

This page summarizes the method in the paper's own notation, with equation numbers,
each linked to the module/function that implements it. For derivations and proofs
(the Proposition's truncation bound, the Theorem's lower bound, SI-1/SI-2/SI-4), see
the paper and its supplementary information (SI):

> K. D. Jones and A. W. Dowling, "BITS for GAPS: Bayesian Information-Theoretic
> Sampling for hierarchical GAussian Process Surrogates," *Computers & Chemical
> Engineering* **211** (2026) 109650.
> [doi:10.1016/j.compchemeng.2026.109650](https://doi.org/10.1016/j.compchemeng.2026.109650)

The article itself is bundled in the repository at
[`paper/bits_for_gaps_paper.pdf`](https://github.com/dowlinglab/bits_for_gaps/blob/main/paper/bits_for_gaps_paper.pdf)
(redistributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), so the
equation numbers cited below can be checked directly against it.

## The loop

BITS for GAPS is a sequential experimental-design loop over a **hierarchical**
Gaussian-process surrogate ({func}`bits_for_gaps.sampler.adaptiveEntropy.run`):

1. **Initialize** -- a space-filling design ({mod}`bits_for_gaps.design`) over the
   input space; a GP prior on the black-box output, Eq (4) below
   ({mod}`bits_for_gaps.kernels`, {mod}`bits_for_gaps.means`); *priors on the GP's own
   hyperparameters* $\theta$ (the "hierarchical" part).
2. **Calibrate** -- evaluate the black box at the design points; run Hamiltonian
   Monte Carlo ({func}`bits_for_gaps.gp.run_mcmc`) to sample the hyperparameter
   posterior $p(\theta \mid \mathbf{y})$, Eq (3); propagate the posterior draws
   through the GP predictive equations, Eq (5a)/(5b), to form a **Gaussian-mixture
   predictive posterior**, Eq (7) ({func}`bits_for_gaps.mixture.sample_gp_posterior_mixture`).
3. **Evaluate** -- score candidate points by the mixture's predictive differential
   entropy, Eq (1) ({mod}`bits_for_gaps.entropy`); pick the maximizer as the next
   point, Eq (2) ({func}`bits_for_gaps.acquisition.optimize`). Repeat.

## Why hierarchical

A plain (non-hierarchical) GP conditions on a *point estimate* of its
hyperparameters (e.g. the MLE) and reports predictive variance around that one
setting. That variance says nothing about how much the data actually constrains the
hyperparameters themselves -- with few observations, very different hyperparameter
settings can fit the data almost equally well, and a plain GP's predictive variance
is blind to that. BITS for GAPS instead puts priors $p(\theta)$ on the hyperparameters
and samples their posterior via HMC; the predictive distribution becomes a mixture
over posterior draws, so its spread reflects *both* observation noise and
hyperparameter uncertainty. Early in a design, when hyperparameters are poorly
constrained, this mixture is visibly wider than any single draw -- which is exactly
the situation where more targeted data collection helps most.

## The hierarchical GP

Given hyperparameters $\theta$ (kernel output scale, lengthscales, mean function),
the latent function values $\mathbf{f} = [f(\mathbf{x}_1), \ldots, f(\mathbf{x}_n)]^\top$
at the design points have the usual GP prior ({func}`bits_for_gaps.gp.build_gp`):

$$\mathbf{f} \sim \mathcal{N}(\mathbf{m}, \mathbf{K}), \qquad
\mathbf{m} = [m(\mathbf{x}_1), \ldots, m(\mathbf{x}_n)]^\top, \qquad
K_{i,j} = k(\mathbf{x}_i, \mathbf{x}_j) \tag{4}$$

with mean function $m$ ({mod}`bits_for_gaps.means`) and the anisotropic
squared-exponential covariance ({class}`bits_for_gaps.kernels.AnisotropicSE`):

$$k_{SE}(\mathbf{x}, \mathbf{x}') = \tau^{-1} \exp\!\left(-\tfrac12 \mathbf{d}^\top
\mathbf{L}^{-1} \mathbf{d}\right), \qquad \mathbf{d} = \mathbf{x} - \mathbf{x}' \tag{6}$$

where $\tau^{-1}$ is the process variance (`std_dev**2`) and $\mathbf{L} =
\mathrm{diag}(\ell_1, \ldots, \ell_d)$ is the diagonal ARD lengthscale matrix
(`lengthscale_1`, `lengthscale_2`, ...). At fixed $\theta$, conditioning on observed
data $\mathbf{y}$ (with i.i.d. Gaussian noise $\sigma_\varepsilon^2$) gives the
standard GP predictive mean and variance at a candidate point $\mathbf{x}_*$
(computed by {mod}`bits_for_gaps.gp`'s `gpflow.models.GPR.predict_f`, built via
{func}`~bits_for_gaps.gp.build_gp`):

$$\mu(\mathbf{x}_*) = m(\mathbf{x}_*) + \mathbf{k}_*^\top
(\mathbf{K} + \sigma_\varepsilon^2 \mathbf{I})^{-1} (\mathbf{y} - \mathbf{m}) \tag{5a}$$

$$\sigma^2(\mathbf{x}_*) = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^\top
(\mathbf{K} + \sigma_\varepsilon^2 \mathbf{I})^{-1} \mathbf{k}_* \tag{5b}$$

The hierarchical part puts priors $p(\theta)$ on the kernel hyperparameters and
samples the resulting posterior via HMC ({func}`bits_for_gaps.gp.run_mcmc`):

$$p(\theta \mid \mathbf{y}) = \frac{p(\mathbf{y} \mid \theta)\, p(\theta)}{p(\mathbf{y})} \tag{3}$$

**Table 1** (the paper's 2-D VLE case study) gives each hyperparameter's prior
family and parameters -- exactly `AnisotropicSE`'s default
({meth}`~bits_for_gaps.kernels.AnisotropicSE.paper_2d`):

| Hyperparameter | Symbol | Prior | Parameters |
|---|---|---|---|
| Kernel std. dev. ($\sqrt{\tau^{-1}}$) | $\theta_1$ | LogNormal | loc $=0$, scale $=2.0$ |
| Mole-fraction lengthscale ($\ell_1$) | $\theta_2$ | LogNormal | loc $=\log(0.3)$, scale $=0.5$ |
| Temperature lengthscale ($\ell_2$) | $\theta_3$ | Gamma | concentration $=4.0$, rate $=2.0$ |

$\theta_3$ (the temperature lengthscale) is deliberately left *unconstrained* (no
positivity bijector) in both the paper and this implementation -- see [Canonical
hyperparameter ordering](#canonical-hyperparameter-ordering) below.

## The Gaussian-mixture predictive posterior

Each of the $S$ HMC draws $\theta^{(s)}$ gives its own GP predictive distribution at
$\mathbf{x}_*$ via Eq (5a)/(5b) above. Averaging over draws gives a Monte Carlo
mixture approximation to the true hierarchical predictive posterior
({func}`bits_for_gaps.mixture.sample_gp_posterior_mixture`):

$$p\{f(\mathbf{x}_*) \mid \mathbf{y}\} \simeq \frac{1}{S} \sum_{s=1}^S
p\{f(\mathbf{x}_*) \mid \mathbf{y}, \theta^{(s)}\} \tag{7}$$

({func}`bits_for_gaps.entropy.gaussian_mixture_density` evaluates this density given
each component's $(\mu_s, \sigma_s^2)$ and equal weights $1/S$.) The mixture's total
mean and variance decompose into within- and between-component contributions:

$$\mathbb{E}[f(\mathbf{x}_*)] = \mu_*(\mathbf{x}_*) = \frac{1}{S} \sum_{s=1}^S \mu_s(\mathbf{x}_*) \tag{8a}$$

$$\mathbb{E}\{f(\mathbf{x}_*) - \mu_*(\mathbf{x}_*)\}^2 = \frac{1}{S} \sum_{s=1}^S \sigma_s^2(\mathbf{x}_*)
+ \frac{1}{S} \sum_{s=1}^S \{\mu_s(\mathbf{x}_*) - \mu_*(\mathbf{x}_*)\}^2 \tag{8b}$$

-- observation-noise-plus-fixed-hyperparameter variance ($\sigma_s^2$, averaged) plus
extra spread from disagreement *between* draws' means ($\mu_s$). This second term is
exactly the hyperparameter-uncertainty contribution a plain (non-hierarchical) GP
lacks.

## The acquisition function

Data acquisition maximizes the predictive differential entropy of the mixture at a
candidate point ({func}`bits_for_gaps.acquisition.entropy_objective`):

$$\mathcal{H}\{f(\mathbf{x}_*)\} := \mathbb{E}\big[-\log p\{f(\mathbf{x}_*)\}\big] \tag{1}$$

$$\max_{\mathbf{x}_* \in \mathcal{X}} \mathcal{H}\{f(\mathbf{x}_*)\}
= \min_{\mathbf{x}_* \in \mathcal{X}} \mathcal{I}\{\mathbf{x}_*\} \tag{2}$$

i.e. maximizing entropy is equivalent to minimizing the information $\mathcal{I}$
already available about $f$ at $\mathbf{x}_*$ -- the acquisition therefore favors
points where the hierarchical posterior is least certain. The maximization runs
L-BFGS-B from Sobol-sequence restarts over the search space $\mathcal{X}$
({func}`bits_for_gaps.acquisition.optimize`).

## Entropy estimators

The exact differential entropy of a Gaussian mixture,

$$\mathcal{H}(f_*) = -\int p(f_*) \log p(f_*)\, \mathrm{d}f_*, \qquad
p(f_*) \simeq \frac{1}{S} \sum_{s=1}^S p_s(f_*) \tag{9}$$

has no closed form, so `bits_for_gaps.entropy` provides two estimators:

- {func}`~bits_for_gaps.entropy.second_order_entropy` -- following Huber et al.
  (2008), expand $g(f_*) := \log p(f_*)$ in a Taylor series about each component
  mean $\mu_s$ and integrate term-by-term against $p_s(f_*) \sim
  \mathcal{N}(\mu_s, \sigma_s^2)$:

  $$\mathcal{H}(f_*) \simeq -\frac{1}{S} \sum_{s=1}^S \int p_s(f_*) \log p(f_*)\, \mathrm{d}f_*,
  \qquad g(f_*) = P_J(f_*) + R_J(f_*)$$

  where $P_J$ is the degree-$J$ Taylor polynomial of $g$ about $\mu_s$ and $R_J$ is
  the remainder. The truncation error is bounded (paper's Proposition; the absolute
  moments of $\mathcal{N}(0,1)$ used in the bound are derived in **SI-1**):

  $$\mathbb{E}[|R_J(f_*)|] \le C_{J+1}\, \sigma_s^{J+1}, \qquad
  C_{J+1} = \frac{M_{J+1}}{(J+1)!} \cdot \frac{2^{J/2+1}}{\sqrt{2\pi}}
  \Gamma\!\left(\frac{J+2}{2}\right)$$

  for $\sup_\zeta |g^{(J+1)}(\zeta)| \le M_{J+1}$ near $\mu_s$.
  {func}`~bits_for_gaps.entropy.first_order_entropy_approx` and
  {func}`~bits_for_gaps.entropy.second_order_entropy` are $J=1$ and $J=2$;
  `second_order_entropy` is the **default** estimator
  {func}`bits_for_gaps.acquisition.entropy_objective` maximizes
  (`objective="taylor"`) -- the one that drove acquisition in the paper.
- {func}`~bits_for_gaps.entropy.entropy_lower_bound` -- the paper's closed-form lower
  bound (unlabeled Theorem, proved via Jensen's inequality on $-\log$), cheaper but
  less tight:

  $$\mathcal{H}_{LB}(f_*) = -\frac{1}{S} \sum_{s=1}^S \log\!\left(\frac{1}{S}
  \sum_{s'=1}^S \xi_{s,s'}\right)$$

  where $\xi_{s,s'}$ is the pairwise cross-overlap between components $s$ and $s'$,
  a closed-form Gaussian integral derived in **SI-2**:

  $$\xi_{s,s'} = \int p_s(f_*)\, p_{s'}(f_*)\, \mathrm{d}f_* =
  \frac{1}{\sqrt{2\pi(\sigma_s^2 + \sigma_{s'}^2)}}
  \exp\!\left(-\frac12 \frac{(\mu_s - \mu_{s'})^2}{\sigma_s^2 + \sigma_{s'}^2}\right)$$

  Selectable as the acquisition objective via `objective="lower_bound"` (or, on
  `BitsForGaps`/`adaptiveEntropy`, `.acquisitionObjective = "lower_bound"`).

Both reduce to the exact differential entropy of a single Gaussian in the
degenerate one-component case (see `tests/unit/test_entropy.py`, which checks this
directly against the analytic $\tfrac{1}{2}\log\big((2\pi e)^d |\Sigma|\big)$ formula).

## Credible intervals

Because the GMM predictive posterior has no closed-form quantiles, the paper's
Algorithm 1 ("Pointwise Credible Region for Hierarchical Gaussian Process
Predictive Posterior," §4.7, following the empirical-quantile approach of Lalchand
and Rasmussen, 2020) draws samples from the mixture at each plotted point, sorts
them, and reports the empirical percentiles as the credible band -- exactly what
`paper/figures/fig06_gp_posterior_surface.py` and `fig07_gp_posterior_isotherms.py`
do with `np.percentile` on draws from
{func}`bits_for_gaps.mixture.sample_gp_posterior_mixture`.

## Canonical hyperparameter ordering

Each kernel hyperparameter (the output scale and one lengthscale per input
dimension) is its own `gpflow.Parameter`, each carrying its own prior -- deliberately
not a single vector-valued parameter, because different dimensions can need
different prior *families* (see {class}`bits_for_gaps.kernels.AnisotropicSE`'s
paper-matching default: LogNormal, LogNormal, and Gamma priors across its three
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

## Where the example's own physics lives

The paper's VLE case study layers its own thermodynamics on top of this method --
extended Raoult's law, the Gibbs-Duhem relation, and the McCabe-Thiele column model
(Eqs (10)-(11) and SI-4). Those equations are stated on the {doc}`vle_example` page
rather than here, because `bits_for_gaps` never sees them: the package only ever sees
the black-box activity coefficient it is designing experiments for.
