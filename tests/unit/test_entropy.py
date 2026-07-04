"""Unit tests for the differential-entropy estimators.

These are pure NumPy/SciPy (no TensorFlow), so they run fast and in CI without the
GP stack. They combine closed-form correctness checks (no magic numbers) with one
regression pin against the paper code's 5-component mixture example (huber_et_al.py).
"""

import numpy as np
import pytest

from bits_for_gaps.entropy import entropy_lower_bound, second_order_entropy


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
    lb = entropy_lower_bound(weights=np.array([1.0]), means=np.array([0.0]),
                             variances=np.array([sigma2]))
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
    # Regression pin against the paper code (fxns/max_ent_design.second_order_entropy)
    # on the 5-component bivariate mixture from huber_et_al.py, with means[4] = [1, 1].
    means = np.array([[0, 0], [3, 2], [1, -0.5], [2.5, 1.5], [1, 1]], dtype=float)
    covs = np.array([
        np.diag([0.16, 1.0]), np.diag([1.0, 0.16]), np.diag([0.5, 0.5]),
        np.diag([0.5, 0.5]), np.diag([0.5, 0.5]),
    ])
    weights = np.ones(5) * 0.2
    H = second_order_entropy(weights=weights, means=means, covs=covs)
    assert H == pytest.approx(2.9564178831291565, rel=1e-9)
