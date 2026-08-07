"""GP construction, log-marginal-likelihood optimization, and HMC sampling.

These are pure functions over explicit arguments -- no disk I/O; the orchestrator
(``sampler.py``) decides whether/where to checkpoint.

``run_mcmc`` uses ``GPmodel.kernel.hyperparameters`` (see
``kernels.AnisotropicSE.hyperparameters``) directly as the HMC state -- any kernel that
exposes a ``.hyperparameters`` property (a list of ``gpflow.Parameter``, in whatever
order that kernel defines as canonical) works here, not just the paper's 3-hyperparameter
2-D kernel.
"""

from __future__ import annotations

import time
from typing import Tuple

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from . import diagnostics

f64 = gpflow.utilities.to_default_float


def build_gp(
    XGP: np.ndarray,
    yGP: np.ndarray,
    mean_fxn: gpflow.mean_functions.MeanFunction,
    kernel_fxn: gpflow.kernels.Kernel,
    likelihood_var: float,
    summarize: bool = False,
) -> gpflow.models.GPR:
    """Construct a GPR model with a fixed (non-trainable) likelihood variance.

    Builds the Eq (4) GP prior ``f ~ N(m, K)`` (mean function ``mean_fxn``, covariance
    kernel ``kernel_fxn``) and conditions gpflow's ``GPR`` on it; ``GPmodel.predict_f``
    then implements the Eq (5a)/(5b) predictive mean/variance at fixed hyperparameters.

    Parameters
    ----------
    XGP, yGP : np.ndarray
        Training inputs/outputs (already in GP-transformed space).
    mean_fxn : gpflow.mean_functions.MeanFunction
        The prior mean function ``m`` in Eq (4).
    kernel_fxn : gpflow.kernels.Kernel
        The covariance kernel ``k`` in Eq (4) (e.g. :class:`~bits_for_gaps.kernels.AnisotropicSE`).
    likelihood_var : float
        Fixed (non-trainable) observation noise variance ``sigma_epsilon^2``.
    summarize : bool
        If True, print a gpflow parameter summary of the constructed model.

    Returns
    -------
    gpflow.models.GPR
        The constructed model, with its likelihood variance fixed at ``likelihood_var``.
    """
    GPmodel = gpflow.models.GPR(data=(XGP, yGP), mean_function=mean_fxn, kernel=kernel_fxn)
    gpflow.set_trainable(GPmodel.likelihood.variance, False)
    GPmodel.likelihood.variance.assign(likelihood_var)
    if summarize:
        gpflow.utilities.print_summary(GPmodel)
    return GPmodel


def maximize_lml(
    GPmodel: gpflow.models.GPR, debug_cov: bool = False
) -> Tuple[object, gpflow.models.GPR]:
    """Maximize the GP log-marginal-likelihood over its trainable (kernel) parameters.

    An MLE point-estimate fit of the hyperparameters ``theta`` -- used only for the
    optional ``initalLML`` warm-start (see ``sampler.py``), not part of the
    hierarchical (HMC posterior) pipeline itself.

    Parameters
    ----------
    GPmodel : gpflow.models.GPR
        Mutated in place by ``gpflow.optimizers.Scipy().minimize``.
    debug_cov : bool
        If True, print the training covariance matrix's condition number before
        optimizing (diagnostic for near-singular kernels).

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        The optimizer's result object.
    GPmodel : gpflow.models.GPR
        The same instance passed in, with its kernel/mean-function parameters updated
        in place.
    """
    if debug_cov:
        XGP, _ = GPmodel.data
        K = GPmodel.kernel(XGP) + GPmodel.likelihood.variance * np.eye(len(XGP[:, 0]))
        condition_number = np.linalg.cond(K)
        print("Condition number of covariance matrix:", condition_number)
    opt = gpflow.optimizers.Scipy()
    result = opt.minimize(GPmodel.training_loss, GPmodel.trainable_variables)
    return result, GPmodel


def run_mcmc(
    GPmodel: gpflow.models.GPR,
    seed: int,
    no_samples: int,
    no_burn_in: int,
    no_chains: int,
    no_leapfrog_steps: int,
    step_size: float,
    no_adapt_steps: int,
    target_accept: float,
    adapt_rate: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, gpflow.models.GPR]:
    """Run HMC over the GP kernel's hyperparameters and compute convergence diagnostics.

    Samples the hyperparameter posterior, Eq (3): ``p(theta | y) = p(y | theta) p(theta)
    / p(y)`` -- ``GPmodel.log_posterior_density`` is exactly ``log p(y | theta) +
    log p(theta)`` (gpflow sums the GP likelihood and each hyperparameter's prior
    log-density), so HMC's target log-density is the (unnormalized) log-posterior.

    Uses ``GPmodel.kernel.hyperparameters`` as the HMC state, in that kernel's own
    canonical order -- this is the trace-column-order contract ``mixture.py`` /
    ``acquisition.py`` rely on (via ``kernels.assign_hyperparameters``) when replaying a
    trace row back onto the kernel.

    Parameters
    ----------
    GPmodel : gpflow.models.GPR
        Its kernel's hyperparameters are the HMC state; left at HMC's last sampled
        state on return (not restored -- this is the orchestrator-level analogue of
        the mutation-footgun fix in ``mixture.py``/``acquisition.py``, which does not
        apply here since ``run_mcmc`` is expected to advance the model's state).
    seed : int
    no_samples, no_burn_in, no_chains : int
        HMC chain configuration: post-burn-in samples per chain, burn-in steps, and
        number of independent chains (for the R-hat/ESS diagnostics below).
    no_leapfrog_steps : int
        Leapfrog integration steps per HMC transition.
    step_size : float
        Initial HMC leapfrog step size (subsequently adapted).
    no_adapt_steps : int
        Number of ``SimpleStepSizeAdaptation`` steps.
    target_accept : float
        Target Metropolis acceptance probability for step-size adaptation.
    adapt_rate : float
        Step-size adaptation rate.

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
        # This line genuinely executes on every HMC run (every integration test that
        # calls run() exercises it, repeatedly) -- it shows as uncovered because
        # @tf.function's AutoGraph compiles this body into a TF graph, which runs
        # outside CPython's normal per-line trace hooks that coverage.py relies on.
        return tfp.mcmc.sample_chain(  # pragma: no cover
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
