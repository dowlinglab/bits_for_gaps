"""Unit tests for ``diagnostics.py``'s R-hat / ESS wrappers.

These were previously exercised only indirectly, via full HMC runs in the
integration suite -- never in isolation against a toy array with a known answer.
"""

import numpy as np

from bits_for_gaps.diagnostics import effective_sample_size, potential_scale_reduction


def test_potential_scale_reduction_shape_and_near_one_for_identical_chains():
    # Two chains with identical samples (no between-chain variance) => R-hat ~= 1
    # for every parameter, the textbook well-converged case.
    rng = np.random.default_rng(0)
    one_chain = rng.normal(size=(500, 3))
    chains_states = np.stack([one_chain, one_chain], axis=1)  # (samples, chains, params)
    rhat = potential_scale_reduction(chains_states)
    assert rhat.shape == (3,)
    np.testing.assert_allclose(rhat, 1.0, atol=0.01)


def test_potential_scale_reduction_detects_diverged_chains():
    # Two chains sampling from very different distributions => R-hat well above 1.
    rng = np.random.default_rng(0)
    chain_a = rng.normal(loc=0.0, size=(500, 1))
    chain_b = rng.normal(loc=50.0, size=(500, 1))
    chains_states = np.stack([chain_a, chain_b], axis=1)
    rhat = potential_scale_reduction(chains_states)
    assert rhat.shape == (1,)
    assert rhat[0] > 1.5


def test_effective_sample_size_shape_and_bounded_by_sample_count():
    rng = np.random.default_rng(0)
    num_samples, num_chains, num_params = 300, 2, 3
    chains_states = rng.normal(size=(num_samples, num_chains, num_params))
    ess = effective_sample_size(chains_states)
    assert ess.shape == (num_params,)
    assert np.all(np.isfinite(ess))
    # ESS cannot exceed the total number of draws across all chains.
    assert np.all(ess <= num_samples * num_chains + 1e-6)


def test_effective_sample_size_lower_for_autocorrelated_chain():
    # An AR(1)-like strongly-autocorrelated chain has far fewer effective samples
    # than i.i.d. draws of the same length.
    rng = np.random.default_rng(0)
    n = 1000
    iid = rng.normal(size=(n, 2, 1))

    correlated = np.empty((n, 2, 1))
    for c in range(2):
        x = np.empty(n)
        x[0] = rng.normal()
        for t in range(1, n):
            x[t] = 0.98 * x[t - 1] + rng.normal(scale=0.1)
        correlated[:, c, 0] = x

    ess_iid = effective_sample_size(iid)[0]
    ess_correlated = effective_sample_size(correlated)[0]
    assert ess_correlated < ess_iid
