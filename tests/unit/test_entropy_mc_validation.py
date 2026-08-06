"""Monte-Carlo validation of the entropy estimators against the true GMM differential
entropy.

``test_entropy.py`` has closed-form checks (single-component exactness) and a
regression pin against the paper code's output, but nothing that validates
*approximation quality* against the actual quantity these estimators approximate: the
true differential entropy H(p) = -E_p[log p(X)] of a Gaussian mixture, which has no
closed form. This file estimates H(p) directly via Monte Carlo (sample the mixture,
average -log of its own density at those samples -- the standard MC estimator for
differential entropy of a distribution you can sample and evaluate) and checks:

- ``second_order_entropy`` (the Taylor approximation) is *close* to the MC estimate.
- ``entropy_lower_bound`` is *at or below* the MC estimate (it's a proven lower bound
  on the true H; MC noise gets a small absolute slack so this doesn't flake).

Tolerances were calibrated empirically (see the commit) rather than guessed: the
Taylor approximation was within 6% of a 300k-sample MC estimate on every mixture
tried, including a 3-component 1-D case and the paper's 5-component 2-D (Huber et al.)
mixture -- 15% leaves comfortable margin for a fixed test seed without being
loose enough to pass a broken estimator.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from bits_for_gaps.entropy import entropy_lower_bound, second_order_entropy

N_SAMPLES = 300_000
TAYLOR_MC_RTOL = 0.15
LOWER_BOUND_MC_SLACK = 0.05


def _mc_entropy_univariate(rng, weights, means, variances, n_samples=N_SAMPLES):
    """MC estimate of the differential entropy of a univariate GMM."""
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    stds = np.sqrt(np.asarray(variances, dtype=float))
    comp = rng.choice(len(weights), size=n_samples, p=weights)
    samples = rng.normal(means[comp], stds[comp])
    pdfs = norm.pdf(samples[:, None], loc=means[None, :], scale=stds[None, :])
    density = pdfs @ weights
    return -np.mean(np.log(density))


def _mc_entropy_multivariate(rng, weights, means, covs, n_samples=N_SAMPLES):
    """MC estimate of the differential entropy of a multivariate GMM."""
    weights = np.asarray(weights, dtype=float)
    counts = rng.multinomial(n_samples, weights)
    samples = np.concatenate(
        [
            rng.multivariate_normal(means[k], covs[k], size=counts[k])
            for k in range(len(weights))
            if counts[k] > 0
        ],
        axis=0,
    )
    density = np.zeros(len(samples))
    for k in range(len(weights)):
        density += weights[k] * multivariate_normal.pdf(samples, mean=means[k], cov=covs[k])
    return -np.mean(np.log(density))


UNIVARIATE_MIXTURES = {
    "well_separated_2comp": ([0.5, 0.5], [0.0, 6.0], [1.0, 1.0]),
    "overlapping_2comp": ([0.5, 0.5], [0.0, 0.5], [1.0, 1.0]),
    "three_comp_unequal_variance": ([0.3, 0.4, 0.3], [-3.0, 0.0, 3.0], [0.5, 1.0, 0.7]),
}


@pytest.mark.parametrize(
    "weights,means,variances", UNIVARIATE_MIXTURES.values(), ids=UNIVARIATE_MIXTURES.keys()
)
def test_second_order_entropy_close_to_mc_estimate_univariate(weights, means, variances):
    rng = np.random.default_rng(0)
    taylor = second_order_entropy(weights=weights, means=means, covs=variances)
    mc = _mc_entropy_univariate(rng, weights, means, variances)
    assert taylor == pytest.approx(mc, rel=TAYLOR_MC_RTOL)


@pytest.mark.parametrize(
    "weights,means,variances", UNIVARIATE_MIXTURES.values(), ids=UNIVARIATE_MIXTURES.keys()
)
def test_entropy_lower_bound_at_or_below_mc_estimate_univariate(weights, means, variances):
    rng = np.random.default_rng(0)
    lb = entropy_lower_bound(np.array(weights), np.array(means), np.array(variances))
    mc = _mc_entropy_univariate(rng, weights, means, variances)
    assert lb <= mc + LOWER_BOUND_MC_SLACK


def test_second_order_entropy_close_to_mc_estimate_multivariate():
    # The paper code's 5-component bivariate mixture (huber_et_al.py) -- also the
    # subject of test_entropy.py's exact regression pin; this checks it against the
    # actual quantity being approximated, not just a captured historical value.
    means = np.array([[0, 0], [3, 2], [1, -0.5], [2.5, 1.5], [1, 1]], dtype=float)
    covs = np.array(
        [
            np.diag([0.16, 1.0]),
            np.diag([1.0, 0.16]),
            np.diag([0.5, 0.5]),
            np.diag([0.5, 0.5]),
            np.diag([0.5, 0.5]),
        ]
    )
    weights = np.ones(5) * 0.2
    rng = np.random.default_rng(1)
    taylor = second_order_entropy(weights=weights, means=means, covs=covs)
    mc = _mc_entropy_multivariate(rng, weights, means, covs)
    assert taylor == pytest.approx(mc, rel=TAYLOR_MC_RTOL)
