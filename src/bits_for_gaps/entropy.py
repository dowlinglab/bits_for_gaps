"""Differential-entropy approximations for Gaussian-mixture predictive posteriors.

The BITS for GAPS acquisition maximizes the differential entropy of the hierarchical
GP predictive posterior, which is a Gaussian mixture (one component per hyperparameter
posterior sample). This module provides the entropy estimators used in the paper:

- :func:`second_order_entropy` -- second-order Taylor approximation (Huber et al., 2008),
  the estimator used to drive acquisition in the paper.
- :func:`entropy_lower_bound` -- the closed-form lower bound (paper, Theorem / SI-2),
  a cheaper analytic reference for the true mixture entropy.

Everything here is pure NumPy/SciPy and dimension-agnostic (univariate or multivariate
components), so it is straightforward to unit-test. Moved verbatim-in-spirit from the
paper code's ``fxns/max_ent_design.py`` (dead commented-out variants removed).
"""

import numpy as np
from scipy import stats


def gaussian_mixture_density(x, means, covs, weights):
    """Density of a Gaussian mixture at ``x``.

    Parameters
    ----------
    x : array-like or scalar
        Point at which to evaluate the density.
    means : sequence
        Component means (scalars for the univariate case, vectors otherwise).
    covs : sequence
        Component variances (univariate) or covariance matrices (multivariate).
    weights : sequence
        Mixture weights.

    Returns
    -------
    float
        The mixture density at ``x``.
    """
    density = 0
    if np.ndim(x) == 0:
        for loopA, w in enumerate(weights):
            density += w * stats.norm.pdf(x, loc=means[loopA], scale=np.sqrt(covs[loopA]))
    else:
        for loopA, w in enumerate(weights):
            density += w * stats.multivariate_normal.pdf(x, mean=means[loopA], cov=covs[loopA])
    return density


def first_order_entropy_approx(weights, means, covs, jitter=1e-10):
    """First-order Taylor approximation of the mixture differential entropy."""
    H = 0
    for loopA, wA in enumerate(weights):
        pl = gaussian_mixture_density(x=means[loopA], means=means, covs=covs, weights=weights)
        assert pl > 0, "Density must be positive"
        H -= wA * np.log(pl + jitter)
    return H


def cholesky(C):
    """Inverse of a covariance matrix via its Cholesky factor."""
    L = np.linalg.cholesky(C)
    L_inv = np.linalg.inv(L)
    return L_inv.T @ L_inv


def gradient_gaussian_mixture_density(x, means, covs, weights):
    """Gradient of the Gaussian-mixture density at ``x``."""
    grad = 0 if np.ndim(x) == 0 else np.zeros_like(x)

    if np.ndim(x) == 0:
        for loopA, w in enumerate(weights):
            mean = means[loopA]
            var = covs[loopA]
            density = stats.norm.pdf(x, loc=mean, scale=np.sqrt(var))
            grad_log_density = -(x - mean) / var
            grad += w * density * grad_log_density
    else:
        for loopA, w in enumerate(weights):
            mean = means[loopA]
            cov = covs[loopA]
            density = stats.multivariate_normal.pdf(x=x, mean=mean, cov=cov)
            grad -= w * density * np.linalg.solve(cov, (x - mean))

    return grad


def second_order_entropy(weights, means, covs):
    """Second-order Taylor approximation of the mixture differential entropy.

    This is the estimator maximized by the BITS for GAPS acquisition function.
    Handles both univariate (scalar means/variances) and multivariate
    (vector means / covariance-matrix) components.
    """
    H0 = first_order_entropy_approx(weights=weights, means=means, covs=covs)
    H = 0

    for loopA, wA in enumerate(weights):
        mu_l = means[loopA]
        sigma_l = covs[loopA]
        pl = gaussian_mixture_density(x=mu_l, means=means, covs=covs, weights=weights)
        grad_pl = gradient_gaussian_mixture_density(x=mu_l, means=means, covs=covs, weights=weights)
        Phi_mu_l = np.zeros_like(sigma_l)

        for loopB, wB in enumerate(weights):
            mu_k = means[loopB]

            if np.ndim(mu_l) == 0:
                sigma_k_inv = 1 / covs[loopB]
                pk_mu_l = gaussian_mixture_density(x=mu_l, means=[mu_k], covs=[covs[loopB]], weights=[1])
                term1 = (mu_l - mu_k) * grad_pl / pl
                term2 = (mu_l - mu_k) * sigma_k_inv * (mu_l - mu_k)
                Phi_mu_l += wB * sigma_k_inv * (term1 + term2 - 1) * pk_mu_l / pl
            else:
                sigma_k_inv = np.linalg.inv(covs[loopB])
                pk_mu_l = gaussian_mixture_density(x=mu_l, means=[mu_k], covs=[covs[loopB]], weights=[1])
                term1 = (mu_l - mu_k).reshape(-1, 1) @ grad_pl.reshape(1, -1) / pl
                term2 = (mu_l - mu_k).reshape(-1, 1) @ (sigma_k_inv @ (mu_l - mu_k)).reshape(1, -1)
                Phi_mu_l += wB * sigma_k_inv @ (term1 + term2 - np.eye(len(mu_l))) * pk_mu_l / pl

        H += 0.5 * wA * np.sum(Phi_mu_l * sigma_l)

    return H0 - H


def entropy_lower_bound(weights, means, variances):
    """Closed-form lower bound on the differential entropy of a univariate GMM.

    Implements the cross-overlap lower bound derived in the paper (Theorem / SI-2):
    ``H_LB = -sum_i w_i log( sum_j w_j z_{i,j} )`` where ``z_{i,j}`` is the Gaussian
    density of ``mu_i`` under ``N(mu_j, var_i + var_j)``.

    Parameters
    ----------
    weights, means, variances : np.ndarray, shape (L,)
        Mixture weights, component means, and component variances.

    Returns
    -------
    float
        The entropy lower bound ``H_LB``.
    """
    def gaussian_density_1d(mu1, mu2, var_sum):
        return (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-0.5 * ((mu1 - mu2) ** 2) / var_sum)

    L = len(weights)
    H_l = 0.0
    for i in range(L):
        inner_sum = 0.0
        for j in range(L):
            var_sum = variances[i] + variances[j]
            z_ij = gaussian_density_1d(means[i], means[j], var_sum)
            inner_sum += weights[j] * z_ij
        H_l += weights[i] * np.log(inner_sum)
    return -H_l
