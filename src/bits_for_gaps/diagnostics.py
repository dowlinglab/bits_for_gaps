"""HMC convergence diagnostics.

Thin wrappers over ``tensorflow_probability.mcmc``, split out of ``driver_new.py``'s
``adaptiveEntropy.run_mcmc`` so ``gp.py``'s HMC driver doesn't need to know the
diagnostics API directly (and so these two lines are independently testable/reusable).
"""

import tensorflow_probability as tfp


def potential_scale_reduction(chains_states):
    """R-hat for each parameter.

    Parameters
    ----------
    chains_states : array-like, shape (num_samples, num_chains, num_params)

    Returns
    -------
    np.ndarray, shape (num_params,)
    """
    return tfp.mcmc.potential_scale_reduction(chains_states, independent_chain_ndims=1).numpy()


def effective_sample_size(chains_states):
    """Effective sample size for each parameter.

    Parameters
    ----------
    chains_states : array-like, shape (num_samples, num_chains, num_params)

    Returns
    -------
    np.ndarray, shape (num_params,)
    """
    return tfp.mcmc.effective_sample_size(chains_states, cross_chain_dims=1).numpy()
