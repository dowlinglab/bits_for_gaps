"""Unit tests for the AnisotropicSE covariance kernel.

Pins the 2-D kernel behavior (symmetry, positive semi-definiteness, the (n, m)
cross-covariance shape, the K_diag shortcut, per-dimension lengthscale scaling) that
predates the Phase 5 N-D generalization, plus new tests for the generalized API: the
canonical ``hyperparameters`` order, ``assign_hyperparameters`` round-tripping, and
explicit N-D (1-D and 3-D) construction with per-dimension prior families.
"""

import gpflow
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bits_for_gaps.kernels import AnisotropicSE, assign_hyperparameters, save_hyperparameters

f64 = gpflow.utilities.to_default_float


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


## ---------------------------------------------------------------------------
## Phase 5: N-D generalization -- canonical ordering, generic assignment, N-D construction
## ---------------------------------------------------------------------------


def test_hyperparameters_canonical_order(kernel):
    # [std_dev, lengthscale_1, lengthscale_2, ...] -- the contract gp.run_mcmc /
    # mixture.py / acquisition.py rely on for generic introspection.
    hp = kernel.hyperparameters
    assert hp == [kernel.std_dev, kernel.lengthscale_1, kernel.lengthscale_2]
    assert kernel.ndim == 2


def test_paper_2d_factory_matches_bare_constructor(X):
    a = AnisotropicSE()
    b = AnisotropicSE.paper_2d()
    np.testing.assert_allclose(a.K(X).numpy(), b.K(X).numpy(), rtol=1e-14, atol=1e-14)
    for pa, pb in zip(a.hyperparameters, b.hyperparameters):
        assert _val(pa) == _val(pb)


def test_default_transforms_preserved():
    # std_dev and lengthscale_1 are constrained positive (Softplus); lengthscale_2 is
    # deliberately left unconstrained (Identity, Gamma prior only) -- this is a real
    # part of the paper's method (it changes the HMC unconstrained parameterization),
    # not an oversight, so the generalized kernel must preserve it exactly.
    k = AnisotropicSE()
    assert isinstance(k.std_dev.transform, tfp.bijectors.Softplus)
    assert isinstance(k.lengthscale_1.transform, tfp.bijectors.Softplus)
    assert isinstance(k.lengthscale_2.transform, tfp.bijectors.Identity)


def test_assign_hyperparameters_round_trips(kernel):
    assign_hyperparameters(kernel, [9.0, 8.0, 7.0])
    assert _val(kernel.std_dev) == pytest.approx(9.0)
    assert _val(kernel.lengthscale_1) == pytest.approx(8.0)
    assert _val(kernel.lengthscale_2) == pytest.approx(7.0)
    # Read back through the same canonical-order property used to assign.
    np.testing.assert_allclose([_val(p) for p in kernel.hyperparameters], [9.0, 8.0, 7.0])


## ---------------------------------------------------------------------------
## Phase 9c: save/restore -- the save_hyperparameters half of the mutation-footgun fix
## (mixture.py/acquisition.py use these together; see their tests for the full loop).
## ---------------------------------------------------------------------------


def test_save_hyperparameters_captures_current_values(kernel):
    assign_hyperparameters(kernel, [9.0, 8.0, 7.0])
    saved = save_hyperparameters(kernel)
    assert saved == [9.0, 8.0, 7.0]


def test_save_then_assign_then_restore_round_trips(kernel):
    saved = save_hyperparameters(kernel)  # the defaults: [1.25, 2.0, 0.5]
    assign_hyperparameters(kernel, [9.0, 8.0, 7.0])
    assert [_val(p) for p in kernel.hyperparameters] != saved
    assign_hyperparameters(kernel, saved)
    np.testing.assert_allclose([_val(p) for p in kernel.hyperparameters], saved)


def test_save_hyperparameters_returns_plain_values_not_live_references(kernel):
    # A snapshot must not change if the kernel is mutated afterward.
    saved = save_hyperparameters(kernel)
    assign_hyperparameters(kernel, [9.0, 8.0, 7.0])
    assert saved == pytest.approx([1.25, 2.0, 0.5])


def test_construct_1d_kernel_with_explicit_priors():
    k = AnisotropicSE(
        variance_prior=tfp.distributions.LogNormal(loc=tf.math.log(f64(1.0)), scale=f64(2.0)),
        lengthscale_priors=[tfp.distributions.LogNormal(loc=tf.math.log(f64(0.3)), scale=f64(0.5))],
    )
    assert k.ndim == 1
    assert len(k.hyperparameters) == 2
    assert k.hyperparameters == [k.std_dev, k.lengthscale_1]
    X = np.array([[0.1], [0.5], [0.9]])
    K = k.K(X).numpy()
    assert K.shape == (3, 3)
    np.testing.assert_allclose(K, K.T, rtol=1e-12, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-8


def test_construct_3d_kernel_with_mixed_prior_families():
    # Mixes LogNormal-positive / Gamma-unconstrained / LogNormal-positive across the 3
    # dimensions -- the same per-dimension-prior-family design point that matters for
    # the paper's 2-D kernel, now exercised at d=3.
    k = AnisotropicSE(
        variance_prior=tfp.distributions.LogNormal(loc=tf.math.log(f64(1.0)), scale=f64(2.0)),
        lengthscale_priors=[
            tfp.distributions.LogNormal(loc=tf.math.log(f64(0.3)), scale=f64(0.5)),
            tfp.distributions.Gamma(concentration=f64(4.0), rate=f64(2.0)),
            tfp.distributions.LogNormal(loc=tf.math.log(f64(0.3)), scale=f64(0.5)),
        ],
        lengthscale_transforms=[gpflow.utilities.positive(), None, gpflow.utilities.positive()],
    )
    assert k.ndim == 3
    assert len(k.hyperparameters) == 4
    assert isinstance(k.lengthscale_1.transform, tfp.bijectors.Softplus)
    assert isinstance(k.lengthscale_2.transform, tfp.bijectors.Identity)
    assert isinstance(k.lengthscale_3.transform, tfp.bijectors.Softplus)
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(5, 3))
    K = k.K(X).numpy()
    assert K.shape == (5, 5)
    np.testing.assert_allclose(K, K.T, rtol=1e-12, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-8
    Kd = k.K_diag(X).numpy()
    np.testing.assert_allclose(Kd, np.diag(K), rtol=1e-10)


def test_construct_requires_matching_lengths():
    with pytest.raises(ValueError):
        AnisotropicSE(
            lengthscale_priors=[tfp.distributions.Gamma(concentration=f64(4.0), rate=f64(2.0))],
            lengthscale_inits=[1.0, 2.0],
        )


def test_variance_prior_requires_lengthscale_priors():
    with pytest.raises(ValueError):
        AnisotropicSE(
            variance_prior=tfp.distributions.LogNormal(loc=tf.math.log(f64(1.0)), scale=f64(2.0)),
        )


## ---------------------------------------------------------------------------
## Phase 9d: assign_hyperparameters raises a clear, specific error instead of a
## low-level gpflow/TF traceback for a value that can't round-trip through a
## parameter's transform -- the exact error class hit mid-investigation in Phase 9b/9c
## (see paper/PHASE9B_INVESTIGATION.md and kernels.py's assign_hyperparameters
## docstring). Behavior-preserving for every assignable value (every value seen in
## this codebase's tests/golden regressions/from-scratch reproduction runs).
## ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_std_dev", [0.0, -1.0])
def test_assign_hyperparameters_raises_clear_error_for_non_finite_unconstrained(
    kernel, bad_std_dev
):
    # std_dev is positivity-constrained (Softplus); 0.0/-1.0 have no finite
    # unconstrained (softplus-inverse) representation.
    with pytest.raises(ValueError, match="std_dev"):
        assign_hyperparameters(kernel, [bad_std_dev, 1.0, 1.0])


def test_assign_hyperparameters_error_names_the_offending_value(kernel):
    with pytest.raises(ValueError, match="-1.0"):
        assign_hyperparameters(kernel, [-1.0, 1.0, 1.0])


def test_assign_hyperparameters_still_works_for_valid_values_after_a_failed_call(kernel):
    # The guard must not leave the kernel (or assign_hyperparameters itself) broken
    # for subsequent valid calls.
    with pytest.raises(ValueError):
        assign_hyperparameters(kernel, [-1.0, 1.0, 1.0])
    assign_hyperparameters(kernel, [2.0, 3.0, 4.0])
    assert [_val(p) for p in kernel.hyperparameters] == [2.0, 3.0, 4.0]
