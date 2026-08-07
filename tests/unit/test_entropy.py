"""Unit tests for the differential-entropy estimators.

These are pure NumPy/SciPy (no TensorFlow), so they run fast and in CI without the
GP stack. They combine closed-form correctness checks (no magic numbers) with one
regression pin against the paper's own 5-component mixture example.
"""

import numpy as np
import pytest
from scipy import stats

from bits_for_gaps.entropy import (
    cholesky,
    entropy_lower_bound,
    first_order_entropy_approx,
    gaussian_mixture_density,
    second_order_entropy,
)


def gaussian_entropy(det_cov, d):
    """Analytic differential entropy of a d-dim Gaussian: 0.5 log((2*pi*e)^d |Sigma|)."""
    return 0.5 * np.log((2 * np.pi * np.e) ** d * det_cov)


@pytest.mark.parametrize("sigma2", [0.5, 1.0, 2.0, 7.3])
def test_second_order_entropy_exact_for_univariate_gaussian(sigma2):
    # For a single-component mixture the 2nd-order Taylor estimator is exact.
    H = second_order_entropy(weights=[1.0], means=[0.5], covs=[sigma2])
    assert H == pytest.approx(gaussian_entropy(sigma2, d=1), rel=1e-6)


def test_second_order_entropy_exact_for_multivariate_gaussian():
    # Diagonal bivariate Gaussian; single component => exact.
    var = np.array([0.4, 1.7])
    cov = np.diag(var)
    H = second_order_entropy(weights=[1.0], means=[np.zeros(2)], covs=[cov])
    assert H == pytest.approx(gaussian_entropy(np.prod(var), d=2), rel=1e-6)


@pytest.mark.parametrize("sigma2", [0.5, 1.0, 2.0])
def test_lower_bound_is_below_true_entropy(sigma2):
    # For a single Gaussian: H_LB = 0.5 log(4*pi*sigma2) < true = 0.5 log(2*pi*e*sigma2).
    lb = entropy_lower_bound(
        weights=np.array([1.0]), means=np.array([0.0]), variances=np.array([sigma2])
    )
    assert lb == pytest.approx(0.5 * np.log(4 * np.pi * sigma2), rel=1e-12)
    assert lb < gaussian_entropy(sigma2, d=1)


def test_lower_bound_below_second_order_for_mixture():
    # The closed-form lower bound must not exceed the (near-true) Taylor estimate.
    rng = np.random.default_rng(0)
    w = np.ones(4) / 4
    means = rng.normal(size=4)
    variances = rng.uniform(0.3, 1.5, size=4)
    lb = entropy_lower_bound(weights=w, means=means, variances=variances)
    taylor = second_order_entropy(weights=w, means=list(means), covs=list(variances))
    assert lb <= taylor + 1e-9


def test_huber_5d_mixture_regression():
    # Regression pin on the paper's own 5-component bivariate mixture example,
    # with means[4] = [1, 1].
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
    H = second_order_entropy(weights=weights, means=means, covs=covs)
    assert H == pytest.approx(2.9564178831291565, rel=1e-9)


## ---------------------------------------------------------------------------
## Explicit exception (not a bare assert) for the runtime, data-dependent density
## check (asserts are silently stripped under python -O).
## ---------------------------------------------------------------------------


def test_first_order_entropy_approx_raises_on_nonpositive_density():
    # A zero-weight component contributes nothing to the mixture density, driving
    # pl to exactly 0 -- a real, reachable condition (e.g. a badly-conditioned or
    # degenerate mixture), not a contrived one.
    with pytest.raises(ValueError, match="must be positive"):
        first_order_entropy_approx(weights=[0.0], means=[0.0], covs=[1.0])


def test_first_order_entropy_approx_well_formed_mixture_unaffected():
    # The happy path must be completely unchanged by the assert -> raise conversion.
    H = first_order_entropy_approx(weights=[0.5, 0.5], means=[0.0, 1.0], covs=[1.0, 1.0])
    assert np.isfinite(H)


## ---------------------------------------------------------------------------
## gaussian_mixture_density (Eq 7): only exercised indirectly (via the entropy
## estimators above) until now -- verify it directly against scipy.stats.
## ---------------------------------------------------------------------------


def test_gaussian_mixture_density_single_component_matches_scipy_univariate():
    x = 0.3
    density = gaussian_mixture_density(x, means=[0.0], covs=[2.0], weights=[1.0])
    assert density == pytest.approx(stats.norm.pdf(x, loc=0.0, scale=np.sqrt(2.0)), rel=1e-12)


def test_gaussian_mixture_density_two_components_is_weighted_sum():
    x = 0.3
    means, covs, weights = [0.0, 1.5], [1.0, 0.5], [0.4, 0.6]
    density = gaussian_mixture_density(x, means, covs, weights)
    expected = 0.4 * stats.norm.pdf(x, loc=0.0, scale=1.0) + 0.6 * stats.norm.pdf(
        x, loc=1.5, scale=np.sqrt(0.5)
    )
    assert density == pytest.approx(expected, rel=1e-12)


def test_gaussian_mixture_density_multivariate_matches_scipy():
    x = np.array([0.2, -0.1])
    mean = np.zeros(2)
    cov = np.diag([1.0, 0.5])
    density = gaussian_mixture_density(x, means=[mean], covs=[cov], weights=[1.0])
    assert density == pytest.approx(stats.multivariate_normal.pdf(x, mean=mean, cov=cov), rel=1e-12)


## ---------------------------------------------------------------------------
## cholesky: not currently called elsewhere in this package (second_order_entropy's
## multivariate branch uses np.linalg.inv directly) but a public, documented utility
## a caller could rely on -- tested directly on its own mathematical contract.
## ---------------------------------------------------------------------------


def test_cholesky_matches_direct_matrix_inverse():
    C = np.array([[4.0, 1.0], [1.0, 3.0]])
    np.testing.assert_allclose(cholesky(C), np.linalg.inv(C), rtol=1e-12)


def test_cholesky_result_is_a_true_inverse():
    C = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])
    C_inv = cholesky(C)
    np.testing.assert_allclose(C @ C_inv, np.eye(3), atol=1e-12)
