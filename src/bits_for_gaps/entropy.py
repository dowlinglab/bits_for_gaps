"""Differential-entropy approximations for Gaussian-mixture predictive posteriors.

The BITS for GAPS acquisition maximizes the differential entropy of the hierarchical
GP predictive posterior, which is a Gaussian mixture (one component per hyperparameter
posterior sample), Eq (7): ``p(f*) ~= (1/S) sum_s p_s(f*)``. Computing that mixture's
exact differential entropy, Eq (9), has no closed form, so this module provides two
estimators used in the paper:

- :func:`second_order_entropy` -- second-order Taylor approximation (Huber et al.,
  2008) of Eq (9) about each component mean, the estimator used to drive acquisition
  in the paper. Its truncation error is bounded by the paper's Proposition (absolute
  moments of a standard normal, derived in SI-1).
- :func:`entropy_lower_bound` -- the closed-form lower bound (paper Theorem, proved
  via Jensen's inequality; its cross-overlap term is derived in SI-2), a cheaper
  analytic reference for the true mixture entropy.

Everything here is pure NumPy/SciPy and dimension-agnostic (univariate or multivariate
components), so it is straightforward to unit-test.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from scipy import stats

# Univariate components: scalar mean/variance. Multivariate: vector mean / covariance
# matrix. Both shapes flow through the same functions below (branched on np.ndim).
_Point = Union[float, np.ndarray]


def gaussian_mixture_density(
    x: _Point, means: Sequence[_Point], covs: Sequence[_Point], weights: Sequence[float]
) -> float:
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

    Notes
    -----
    Eq (7): ``p(x) = sum_s weights[s] * N(x; means[s], covs[s])`` -- each summand is
    one hyperparameter-posterior draw's GP predictive density (Eq 5a/5b).
    """
    density = 0
    if np.ndim(x) == 0:
        for loopA, w in enumerate(weights):
            density += w * stats.norm.pdf(x, loc=means[loopA], scale=np.sqrt(covs[loopA]))
    else:
        for loopA, w in enumerate(weights):
            density += w * stats.multivariate_normal.pdf(x, mean=means[loopA], cov=covs[loopA])
    return density


def first_order_entropy_approx(
    weights: Sequence[float], means: Sequence[_Point], covs: Sequence[_Point], jitter: float = 1e-10
) -> float:
    """First-order Taylor approximation of the mixture differential entropy.

    Eq (9): ``H(f*) ~= -(1/S) sum_s log p(mu_s)`` -- the Huber et al. (2008) Taylor
    expansion of ``log p(f*)`` about each component mean ``mu_s``, truncated at the
    zeroth-order (constant) term. This is ``H0`` in :func:`second_order_entropy`, which
    adds the second-order correction.

    Parameters
    ----------
    weights : sequence of float
        Mixture weights.
    means : sequence
        Component means (scalars for the univariate case, vectors otherwise).
    covs : sequence
        Component variances (univariate) or covariance matrices (multivariate).
    jitter : float
        Added to the density before taking its log, to avoid ``log(0)`` from residual
        floating-point underflow after the positivity check below has already passed.

    Returns
    -------
    float
        The first-order entropy estimate.

    Raises
    ------
    ValueError
        If the mixture density at some component mean is not positive (e.g. floating-
        point underflow for very small covariances / far-apart means) -- a data-
        dependent runtime condition, not a static invariant, hence an explicit
        exception rather than a bare ``assert`` (asserts are silently stripped under
        ``python -O``, which would let a NaN/garbage entropy value propagate instead of
        failing clearly).
    """
    H = 0
    for loopA, wA in enumerate(weights):
        pl = gaussian_mixture_density(x=means[loopA], means=means, covs=covs, weights=weights)
        if not (pl > 0):
            raise ValueError(
                f"Gaussian-mixture density must be positive, got {pl!r} at component "
                f"{loopA} -- check that means/covs/weights are well-formed (e.g. no "
                f"degenerate/zero covariances)."
            )
        H -= wA * np.log(pl + jitter)
    return H


def cholesky(C: np.ndarray) -> np.ndarray:
    """Inverse of a covariance matrix via its Cholesky factor.

    Parameters
    ----------
    C : np.ndarray, shape (d, d)
        A symmetric positive-definite covariance matrix.

    Returns
    -------
    np.ndarray, shape (d, d)
        ``C`` inverse, computed as ``L^-T L^-1`` where ``C = L L^T`` -- more
        numerically stable than a direct ``np.linalg.inv`` for well-conditioned
        covariance matrices. Not currently called elsewhere in this package (
        :func:`second_order_entropy`'s multivariate branch uses ``np.linalg.inv``
        directly); kept as a public, tested utility for callers who want the more
        numerically stable inverse.
    """
    L = np.linalg.cholesky(C)
    L_inv = np.linalg.inv(L)
    return L_inv.T @ L_inv


def gradient_gaussian_mixture_density(
    x: _Point, means: Sequence[_Point], covs: Sequence[_Point], weights: Sequence[float]
) -> _Point:
    """Gradient (with respect to ``x``) of the Gaussian-mixture density at ``x``.

    Used by :func:`second_order_entropy`'s Taylor expansion of ``log p(f*)`` (Eq 9):
    each summand's gradient contribution is ``weight * density * grad_log_density``,
    with ``grad_log_density = -Sigma^-1 (x - mean)`` for a single Gaussian.

    Parameters
    ----------
    x : array-like or scalar
        Point at which to evaluate the gradient.
    means : sequence
        Component means (scalars for the univariate case, vectors otherwise).
    covs : sequence
        Component variances (univariate) or covariance matrices (multivariate).
    weights : sequence of float
        Mixture weights.

    Returns
    -------
    float or np.ndarray
        The mixture density's gradient at ``x`` (same shape as ``x``).
    """
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


def second_order_entropy(
    weights: Sequence[float], means: Sequence[_Point], covs: Sequence[_Point]
) -> float:
    """Second-order Taylor approximation of the mixture differential entropy.

    This is the estimator maximized by the BITS for GAPS acquisition function.
    Handles both univariate (scalar means/variances) and multivariate
    (vector means / covariance-matrix) components.

    Eq (9), Huber et al. (2008): expand ``g(f*) = log p(f*)`` in a Taylor series about
    each component mean ``mu_s`` and integrate term-by-term against that component's
    own density. The zeroth-order term is :func:`first_order_entropy_approx` (``H0``
    below); this function adds the second-order (curvature) correction ``H``, so the
    estimate returned is ``H0 - H``. The per-component-pair ``Phi_mu_l`` accumulation
    below is ``g``'s Hessian at ``mu_l`` (via the log-density's gradient/curvature
    identities for a Gaussian mixture), weighted by each pair's relative density
    ``pk_mu_l / pl`` -- the paper's Proposition (SI-1) bounds the error this truncation
    introduces in terms of the absolute moments of a standard normal.

    Parameters
    ----------
    weights : sequence of float
        Mixture weights.
    means : sequence
        Component means (scalars for the univariate case, vectors otherwise).
    covs : sequence
        Component variances (univariate) or covariance matrices (multivariate).

    Returns
    -------
    float
        The second-order entropy estimate ``H0 - H``.
    """
    H0 = first_order_entropy_approx(weights=weights, means=means, covs=covs)  # Eq (9), 0th order
    H = 0

    for loopA, wA in enumerate(weights):
        mu_l = means[loopA]
        sigma_l = covs[loopA]
        pl = gaussian_mixture_density(x=mu_l, means=means, covs=covs, weights=weights)
        grad_pl = gradient_gaussian_mixture_density(x=mu_l, means=means, covs=covs, weights=weights)
        Phi_mu_l = np.zeros_like(sigma_l)  # accumulates the Hessian of log p(f*) at mu_l

        for loopB, wB in enumerate(weights):
            mu_k = means[loopB]

            if np.ndim(mu_l) == 0:
                sigma_k_inv = 1 / covs[loopB]
                pk_mu_l = gaussian_mixture_density(
                    x=mu_l, means=[mu_k], covs=[covs[loopB]], weights=[1]
                )
                term1 = (mu_l - mu_k) * grad_pl / pl
                term2 = (mu_l - mu_k) * sigma_k_inv * (mu_l - mu_k)
                Phi_mu_l += wB * sigma_k_inv * (term1 + term2 - 1) * pk_mu_l / pl
            else:
                sigma_k_inv = np.linalg.inv(covs[loopB])
                pk_mu_l = gaussian_mixture_density(
                    x=mu_l, means=[mu_k], covs=[covs[loopB]], weights=[1]
                )
                term1 = (mu_l - mu_k).reshape(-1, 1) @ grad_pl.reshape(1, -1) / pl
                term2 = (mu_l - mu_k).reshape(-1, 1) @ (sigma_k_inv @ (mu_l - mu_k)).reshape(1, -1)
                Phi_mu_l += wB * sigma_k_inv @ (term1 + term2 - np.eye(len(mu_l))) * pk_mu_l / pl

        # 2nd-order Taylor correction: 0.5 * tr(Hessian @ Sigma_l), summed over components.
        H += 0.5 * wA * np.sum(Phi_mu_l * sigma_l)

    return H0 - H


def entropy_lower_bound(weights: np.ndarray, means: np.ndarray, variances: np.ndarray) -> float:
    """Closed-form lower bound on the differential entropy of a univariate GMM.

    Implements the cross-overlap lower bound derived in the paper (Theorem, proved via
    Jensen's inequality on ``-log``; SI-2): ``H_LB = -sum_i w_i log( sum_j w_j
    xi_{i,j} )`` where ``xi_{i,j}`` is the pairwise cross-overlap between components
    ``i`` and ``j`` -- the Gaussian density of ``mu_i`` under ``N(mu_j, var_i +
    var_j)`` (SI-2's closed-form Gaussian-overlap integral).

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
        # SI-2: xi_{i,j} = integral of p_i(f*) p_j(f*) df* = N(mu_i; mu_j, var_i + var_j).
        return (1 / np.sqrt(2 * np.pi * var_sum)) * np.exp(-0.5 * ((mu1 - mu2) ** 2) / var_sum)

    L = len(weights)
    H_l = 0.0
    for i in range(L):
        inner_sum = 0.0
        for j in range(L):
            var_sum = variances[i] + variances[j]
            xi_ij = gaussian_density_1d(means[i], means[j], var_sum)
            inner_sum += weights[j] * xi_ij
        H_l += weights[i] * np.log(inner_sum)
    return -H_l
