"""Unit tests for the AnisotropicSE covariance kernel.

Pins the current (2-D) kernel behavior before the Phase 4/5 refactor: symmetry, positive
semi-definiteness, the (n, m) cross-covariance shape, the K_diag shortcut, and the
per-dimension lengthscale scaling.
"""
import numpy as np
import pytest

from bits_for_gaps.kernels import AnisotropicSE


@pytest.fixture
def kernel():
    return AnisotropicSE()


@pytest.fixture
def X():
    # 6 points in the 2-D input space (mole fraction, temperature-ish).
    rng = np.random.default_rng(0)
    return rng.uniform([0.0, 350.0], [1.0, 367.0], size=(6, 2))


def _val(param):
    return float(param.numpy())


def test_default_hyperparameters(kernel):
    assert _val(kernel.std_dev) == pytest.approx(1.25)
    assert _val(kernel.lengthscale_1) == pytest.approx(2.0)
    assert _val(kernel.lengthscale_2) == pytest.approx(0.5)


def test_K_symmetric(kernel, X):
    K = kernel.K(X).numpy()
    assert K.shape == (6, 6)
    np.testing.assert_allclose(K, K.T, rtol=1e-12, atol=1e-12)


def test_K_positive_semidefinite(kernel, X):
    K = kernel.K(X).numpy()
    eigvals = np.linalg.eigvalsh(K)
    assert eigvals.min() > -1e-8


def test_K_cross_covariance_shape(kernel, X):
    X2 = X[:4] + 0.01
    K = kernel.K(X, X2).numpy()
    assert K.shape == (6, 4)


def test_K_diag_matches_diagonal_and_variance(kernel, X):
    Kd = kernel.K_diag(X).numpy()
    K = kernel.K(X).numpy()
    assert Kd.shape == (6,)
    # SE kernel: K(x, x) = std_dev**2 on the diagonal.
    np.testing.assert_allclose(Kd, np.diag(K), rtol=1e-10)
    np.testing.assert_allclose(Kd, _val(kernel.std_dev) ** 2, rtol=1e-10)


def test_covariance_decays_with_distance(kernel):
    x0 = np.array([[0.5, 358.0]])
    near = np.array([[0.51, 358.0]])
    far = np.array([[0.99, 366.0]])
    k_self = kernel.K(x0, x0).numpy()[0, 0]
    k_near = kernel.K(x0, near).numpy()[0, 0]
    k_far = kernel.K(x0, far).numpy()[0, 0]
    assert k_self >= k_near >= k_far
    assert k_far >= 0.0


def test_anisotropic_lengthscale_scaling(kernel):
    # A displacement of one lengthscale in dim 1 gives the same covariance as a
    # displacement of one lengthscale in dim 2 (both are 1 scaled unit away).
    l1, l2 = _val(kernel.lengthscale_1), _val(kernel.lengthscale_2)
    x0 = np.array([[0.0, 0.0]])
    off1 = np.array([[l1, 0.0]])
    off2 = np.array([[0.0, l2]])
    k1 = kernel.K(x0, off1).numpy()[0, 0]
    k2 = kernel.K(x0, off2).numpy()[0, 0]
    np.testing.assert_allclose(k1, k2, rtol=1e-10)
    # both equal std_dev**2 * exp(-0.5) for a unit scaled distance
    np.testing.assert_allclose(k1, _val(kernel.std_dev) ** 2 * np.exp(-0.5), rtol=1e-10)
