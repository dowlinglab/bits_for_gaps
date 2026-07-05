"""GP construction, log-marginal-likelihood optimization, and HMC sampling.

Moved from the paper code's ``driver_new.py`` (``adaptiveEntropy.build_gp`` /
``maximize_lml`` / ``run_mcmc``). These are pure functions over explicit arguments --
no disk I/O; the orchestrator (``sampler.py``) decides whether/where to checkpoint.

Phase 5: ``run_mcmc`` no longer indexes ``GPmodel.trainable_parameters`` by hardcoded
position. It uses ``GPmodel.kernel.hyperparameters`` (see
``kernels.AnisotropicSE.hyperparameters``) directly as the HMC state -- any kernel that
exposes a ``.hyperparameters`` property (a list of ``gpflow.Parameter``, in whatever
order that kernel defines as canonical) works here, not just the paper's 3-hyperparameter
2-D kernel. For the paper's kernel this is verified identity-equal to the old
``[trainable_parameters[2], [0], [1]]`` indexing (same Parameter objects, same order),
so d=2 runs are bit-exact with the pre-Phase-5 code.
"""

import time

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from . import diagnostics

f64 = gpflow.utilities.to_default_float


def build_gp(XGP, yGP, mean_fxn, kernel_fxn, likelihood_var, summarize=False):
    """Construct a GPR model with a fixed (non-trainable) likelihood variance."""
    GPmodel = gpflow.models.GPR(data=(XGP, yGP), mean_function=mean_fxn, kernel=kernel_fxn)
    gpflow.set_trainable(GPmodel.likelihood.variance, False)
    GPmodel.likelihood.variance.assign(likelihood_var)
    if summarize:
        gpflow.utilities.print_summary(GPmodel)
    return GPmodel


def maximize_lml(GPmodel, debug_cov=False):
    """Maximize the GP log-marginal-likelihood over its trainable (kernel) parameters."""
    if debug_cov:
        XGP, _ = GPmodel.data
        K = GPmodel.kernel(XGP) + GPmodel.likelihood.variance * np.eye(len(XGP[:, 0]))
        condition_number = np.linalg.cond(K)
        print("Condition number of covariance matrix:", condition_number)
    opt = gpflow.optimizers.Scipy()
    result = opt.minimize(GPmodel.training_loss, GPmodel.trainable_variables)
    return result, GPmodel


def run_mcmc(
    GPmodel,
    seed,
    no_samples,
    no_burn_in,
    no_chains,
    no_leapfrog_steps,
    step_size,
    no_adapt_steps,
    target_accept,
    adapt_rate,
):
    """Run HMC over the GP kernel's hyperparameters and compute convergence diagnostics.

    Uses ``GPmodel.kernel.hyperparameters`` as the HMC state, in that kernel's own
    canonical order -- this is the trace-column-order contract ``mixture.py`` /
    ``acquisition.py`` rely on (via ``kernels.assign_hyperparameters``) when replaying a
    trace row back onto the kernel.

    Returns
    -------
    trace : np.ndarray, shape (no_samples, n_hyperparameters)
        Chain-0 posterior samples in the *constrained* (untransformed) parameter space.
    chains_states : np.ndarray, shape (no_samples, no_chains, n_hyperparameters)
        Every chain's samples in the unconstrained space (for diagnostics/checkpointing).
    rhat, ess : np.ndarray, shape (n_hyperparameters,)
    GPmodel : the same model instance (hyperparameters are left at HMC's last state).
    """
    hmc_helper = gpflow.optimizers.SamplingHelper(
        GPmodel.log_posterior_density,
        GPmodel.kernel.hyperparameters,
    )
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper.target_log_prob_fn,
        num_leapfrog_steps=no_leapfrog_steps,
        step_size=step_size,
    )
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc,
        num_adaptation_steps=no_adapt_steps,
        target_accept_prob=target_accept,
        adaptation_rate=adapt_rate,
    )

    @tf.function
    def run_chain_fn(chain_seed):
        return tfp.mcmc.sample_chain(
            num_results=no_samples,
            num_burnin_steps=no_burn_in,
            current_state=hmc_helper.current_state,
            kernel=adaptive_hmc,
            trace_fn=None,
            seed=chain_seed,
        )

    start_time = time.time()
    chains = []
    for i in tqdm(range(no_chains), desc="Running MCMC chains"):
        chains.append(run_chain_fn(seed + i))
    chains_states = tf.stack(chains, axis=1)
    chains_states = tf.transpose(chains_states, perm=[2, 1, 0])  # samples x chains x params
    print(f"\t Execution time: {time.time() - start_time} seconds.")

    rhat = diagnostics.potential_scale_reduction(chains_states)
    ess = diagnostics.effective_sample_size(chains_states)

    first_chain = chains_states[:, 0, :]
    num_params = first_chain.shape[-1]
    split_first_chain = tf.split(first_chain, num_or_size_splits=[1] * num_params, axis=-1)
    split_first_chain = [tf.squeeze(x, axis=-1) for x in split_first_chain]
    constrained_first_chain = hmc_helper.convert_to_constrained_values(split_first_chain)
    trace = np.stack([p.numpy() for p in constrained_first_chain], axis=-1)

    return trace, chains_states.numpy(), rhat, ess, GPmodel
