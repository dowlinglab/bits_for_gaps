"""The BITS for GAPS sequential-design engine.

Moved from the paper code's ``driver_new.py`` and DECOUPLED from the VLE example:
module-level ``juliacall``/``proh_water_class`` imports and the example-only
``generateData``/``run_test`` helpers were removed. The black-box model is injected
via ``fwd_model``/``fwd_model_args`` -- the core has no Julia dependency.

The algorithm body is otherwise preserved from the manuscript run (Phase 3 = faithful
move). Known targets for later phases, deliberately left intact for now:

    TODO(Phase 4): decompose this class into gp/mixture/acquisition/state modules;
                   replace disk-as-state (np.savetxt/pickle under results/) with
                   in-memory state + optional checkpointing.
    TODO(Phase 5): remove 2-D / 3-hyperparameter hardcoding -- run_mcmc indexes
                   trainable_parameters[2],[0],[1]; the mixture/entropy code assigns
                   kernel params by name (std_dev, lengthscale_1, lengthscale_2);
                   gp_predict_2D / gen_entropy_surface_data_2D / optimize_2D assume d=2.

Regression tests (Phase 2) should pin current behavior BEFORE any of the above.
"""

import os
import pickle
import time

import numpy as np
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from tqdm import tqdm

from . import entropy as max_ent_design

f64 = gpflow.utilities.to_default_float


class adaptiveEntropy:
    def __init__(self, exp_name, iters, x_bounds, likelihood_var, mean_fxn, kernel_fxn,
                 fwd_model, fwd_model_args):
        """
        exp_name, iters, x_bounds, likelihood_var, mean_fxn, kernel_fxn,
        fwd_model, fwd_model_args
        """
        ## Miscellaneous
        self.path = os.path.join("results", exp_name)   # path to save/read results
        self.seed = 123                                 # random seed

        ## Data related
        self.XBnds          = x_bounds
        self.XTrsfFwd       = [lambda x: x for _ in self.XBnds]
        self.XTrsfBkwd      = [lambda x: x for _ in self.XBnds]
        self.yTrsfFwd       = lambda y: y
        self.yTrsfBkwd      = lambda y: y

        ## Sequential design
        self.noIters        = iters
        self.startIter      = 0     # iteration offset for resuming a prior run
                                    # (was a hardcoded ``i += 50`` in the manuscript run)
        self.noRestarts     = 10
        self.noGaussians    = 25
        self.entropyMesh    = [10 for _ in self.XBnds]
        self.optMethod      = None
        self.optOptions     = None
        self.FwdModel       = fwd_model
        self.FwdModelArgs   = fwd_model_args

        ## Gaussian Process
        self.likelihoodVar      = likelihood_var
        self.meanFxn            = mean_fxn
        self.kernelFxn          = kernel_fxn
        self.noGPpredictions    = 20
        self.summarizeGP        = False
        self.initalLML          = False
        self.debugCov           = False
        self.showLMLres         = False

        ## Hamiltonian MCMC
        self.noSamples          = 5000
        self.noBurnIn           = 2000
        self.noChains           = 4
        self.noLeapfrogSteps    = 5
        self.stepSize           = 0.05
        self.noAdaptSteps       = 5
        self.targetAccept       = 0.90
        self.adaptRate          = 0.10

    def read_data(self, iters):
        data = np.loadtxt(os.path.join(self.path, f"activity_data_{iters}"))
        nFeatures = len(self.XBnds)
        XData = data[:, :nFeatures]
        yData = data[:, [nFeatures]]
        return XData, yData

    def trsf_data(self, XData, yData, iters):
        XData = np.atleast_2d(XData)
        yData = np.atleast_1d(yData).reshape(-1, 1)
        XGP = np.column_stack([fwd(XData[:, j]) for j, fwd in enumerate(self.XTrsfFwd)])
        yGP = self.yTrsfFwd(yData)
        filepath = os.path.join(self.path, f"gp_training_data_{iters}")
        np.savetxt(filepath, np.concatenate((XGP, yGP), axis=1))
        return XGP, yGP

    def build_gp(self, XGP, yGP, iters):
        GPmodel = gpflow.models.GPR(data=(XGP, yGP),
                                    mean_function=self.meanFxn,
                                    kernel=self.kernelFxn)
        gpflow.set_trainable(GPmodel.likelihood.variance, False)
        GPmodel.likelihood.variance.assign(self.likelihoodVar)
        if self.summarizeGP:
            gpflow.utilities.print_summary(GPmodel)
        with open(os.path.join(self.path, f'gp_model_{iters}.pkl'), 'wb') as f:
            pickle.dump(GPmodel, f)
        return GPmodel

    def maximize_lml(self, GPmodel):
        if self.debugCov:
            XGP, _ = GPmodel.data
            K = GPmodel.kernel(XGP) + GPmodel.likelihood.variance * np.eye(len(XGP[:, 0]))
            condition_number = np.linalg.cond(K)
            print("Condition number of covariance matrix:", condition_number)
        opt = gpflow.optimizers.Scipy()
        result = opt.minimize(GPmodel.training_loss, GPmodel.trainable_variables)
        return result, GPmodel

    def run_mcmc(self, GPmodel, iters):
        ## Set up HMC
        hmc_helper = gpflow.optimizers.SamplingHelper(GPmodel.log_posterior_density,
                                                      [GPmodel.trainable_parameters[2],
                                                       GPmodel.trainable_parameters[0],
                                                       GPmodel.trainable_parameters[1]])  # order: variance, l1, l2
        hmc = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=hmc_helper.target_log_prob_fn,
                                             num_leapfrog_steps=self.noLeapfrogSteps,
                                             step_size=self.stepSize)
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(hmc,
                                                         num_adaptation_steps=self.noAdaptSteps,
                                                         target_accept_prob=self.targetAccept,
                                                         adaptation_rate=self.adaptRate)

        @tf.function
        def run_chain_fn(seed):
            return tfp.mcmc.sample_chain(num_results=self.noSamples,
                                         num_burnin_steps=self.noBurnIn,
                                         current_state=hmc_helper.current_state,
                                         kernel=adaptive_hmc,
                                         trace_fn=None,
                                         seed=seed)

        ## Run HMC
        start_time = time.time()
        chains_states = []
        for i in tqdm(range(self.noChains), desc="Running MCMC chains"):
            chain_seed = self.seed + i
            chain_state = run_chain_fn(chain_seed)
            chains_states.append(chain_state)
        chains_states = tf.stack(chains_states, axis=1)
        chains_states = tf.transpose(chains_states, perm=[2, 1, 0])  # num_samples x num_chains x num_params
        end_time = time.time()
        print(f"\t Execution time: {end_time - start_time} seconds.")

        ## Convergence diagnostics
        rhat_value = tfp.mcmc.potential_scale_reduction(chains_states, independent_chain_ndims=1).numpy()
        ess_value = tfp.mcmc.effective_sample_size(chains_states, cross_chain_dims=1).numpy()

        ## Post processing
        first_chain = chains_states[:, 0, :]
        num_params = first_chain.shape[-1]
        split_first_chain = tf.split(first_chain, num_or_size_splits=[1] * num_params, axis=-1)
        split_first_chain = [tf.squeeze(x, axis=-1) for x in split_first_chain]
        constrained_first_chain = hmc_helper.convert_to_constrained_values(split_first_chain)

        ## Save results
        np.savetxt(os.path.join(self.path, f"rhat_value_{iters}.txt"), rhat_value, header="R-hat value")
        np.savetxt(os.path.join(self.path, f"ess_value_{iters}.txt"), ess_value, header="ess value")
        for i in range(self.noChains):
            np.savetxt(os.path.join(self.path, f"traces_chain_{i}_exp_{iters}"), chains_states[:, i, :])
        trace = np.stack([p.numpy() for p in constrained_first_chain], axis=-1)
        np.savetxt(os.path.join(self.path, f"param_posterior_samples_{iters}"), trace)
        with open(os.path.join(self.path, f"gp_model_{iters}.pkl"), 'wb') as f:
            pickle.dump(GPmodel, f)
        return trace, GPmodel

    def sample_gp_posterior_mixture(self, trace, GPmodel, XGP, size=None):
        """Sample from a mixture of GPs (full covariance), one sample per component."""
        np.random.seed(self.seed)
        if size is None:
            size = self.noGaussians
        subset_indices = np.random.choice(len(trace), size=size, replace=False)
        sub_samples = trace[subset_indices]
        comp_ids = np.random.randint(0, len(sub_samples), size=100)
        samples = []
        for comp_id in comp_ids:
            GPmodel.kernel.std_dev.assign(sub_samples[comp_id, 0])
            GPmodel.kernel.lengthscale_1.assign(sub_samples[comp_id, 1])
            GPmodel.kernel.lengthscale_2.assign(sub_samples[comp_id, 2])
            gp_out = GPmodel.predict_f_samples(XGP, full_cov=True)
            samples.append(gp_out)
        out = np.array(samples).squeeze()
        return out

    def gp_predict_2D(self, GPmodel, trace, iters):
        x1, x2 = np.linspace(*self.XBnds[0]), np.linspace(*self.XBnds[1])
        x1_grid, x2_grid = np.meshgrid(x1, x2)
        xStar = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
        XStarGP = np.column_stack([fwd(xStar[:, j]) for j, fwd in enumerate(self.XTrsfFwd)])
        yStarGP = self.sample_gp_posterior_mixture(trace, GPmodel, XStarGP, self.noGPpredictions)
        xStar = np.column_stack([bkwd(XStarGP[:, j]) for j, bkwd in enumerate(self.XTrsfBkwd)])
        yStar = self.yTrsfBkwd(yStarGP).T
        predictions = np.column_stack([xStar, yStar])
        np.savetxt(os.path.join(self.path, f'gp_predict_{iters}'), predictions)
        with open(os.path.join(self.path, f"gp_model_{iters}.pkl"), 'wb') as f:
            pickle.dump(GPmodel, f)
        return GPmodel

    def entropy_objective(self, xStarGP, trace, GPmodel):
        xStarGP = xStarGP.reshape(-1, 1).T
        np.random.seed(self.seed)
        subset_indices = np.random.choice(len(trace), size=self.noGaussians, replace=False)
        sub_samples = trace[subset_indices]
        means = []
        variances = []
        for sample in sub_samples:
            GPmodel.kernel.std_dev.assign(sample[0])
            GPmodel.kernel.lengthscale_1.assign(sample[1])
            GPmodel.kernel.lengthscale_2.assign(sample[2])
            mean, variance = GPmodel.predict_f(xStarGP, full_cov=True)
            means.append(mean.numpy().squeeze())
            variances.append(variance.numpy().squeeze())
        means = np.array(means)
        variances = np.array(variances)
        H = max_ent_design.second_order_entropy(
            weights=np.ones(self.noGaussians) * 1 / self.noGaussians,
            means=means,
            covs=variances)
        return -H  # negative because we are minimizing

    def gen_entropy_surface_data_2D(self, trace, GPmodel, iters):
        x1 = np.linspace(*self.XBnds[0], self.entropyMesh[0])
        x2 = np.linspace(*self.XBnds[1], self.entropyMesh[1])
        x1_grid, x2_grid = np.meshgrid(x1, x2)
        XStar = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T
        cols = []
        for j, fwd in enumerate(self.XTrsfFwd):
            cols.append(fwd(XStar[:, j]))
        XStarGP = np.column_stack(cols)
        H = np.array([self.entropy_objective(xStarGP=x, trace=trace, GPmodel=GPmodel) * -1 for x in XStarGP])
        XStar = np.column_stack([bkwd(XStarGP[:, j]) for j, bkwd in enumerate(self.XTrsfBkwd)])
        np.savetxt(os.path.join(self.path, f"entropy_{iters}"), np.column_stack([XStar, H]))

    def optimize_2D(self, trace, GPmodel):
        best_result = None
        best_value = np.inf
        sobol = Sobol(d=2, scramble=True, seed=self.seed)
        XBndsGP = [(f(b[0]), f(b[1])) for f, b in zip(self.XTrsfFwd, self.XBnds)]
        for j in range(self.noRestarts):
            x0 = sobol.random()[0]
            x0 = np.array([x0[0] * (self.XBnds[0][1] - self.XBnds[0][0]) + self.XBnds[0][0],
                           x0[1] * (self.XBnds[1][1] - self.XBnds[1][0]) + self.XBnds[1][0]])
            x0GP = np.array([f(x0[i]) for i, f in enumerate(self.XTrsfFwd)])
            result = minimize(self.entropy_objective, x0=x0GP, bounds=XBndsGP, args=(trace, GPmodel))
            if result.fun < best_value:
                best_value = result.fun
                best_result = result
        return best_result

    def call_model(self, xStar, XData, yData, iters):
        x1Tot = np.append(XData[:, 0], xStar[0])
        x2Tot = np.append(XData[:, 1], xStar[1])
        yStar = np.array(self.FwdModel(*self.FwdModelArgs, x2Tot[-1], x1Tot[-1]))
        y1Tot = np.append(yData[:, 0], yStar[0])
        data = np.vstack((x1Tot, x2Tot, y1Tot)).T
        np.savetxt(os.path.join(self.path, f"activity_data_{iters}"), data)

    def run_model(self):
        for i in range(self.noIters):
            i += self.startIter
            print(f"On iteration {i + 1}")

            print("Reading in data....")
            XData, yData = self.read_data(iters=i + 1)
            print("Done.")

            print("Transforming training data....")
            XGP, yGP = self.trsf_data(XData=XData, yData=yData, iters=i + 1)
            print("Done.")

            print("Building GP....")
            GPmodel = self.build_gp(XGP=XGP, yGP=yGP, iters=i + 1)
            print("Done.")

            if self.initalLML:
                print("Initializing to log-marginal-likelihood solution...")
                result, GPmodel = self.maximize_lml(GPmodel=GPmodel)
                if self.showLMLres:
                    print(result)
                    gpflow.utilities.print_summary(GPmodel)
                print("Done.")

            print("Beginning HMC...")
            trace, GPmodel = self.run_mcmc(GPmodel=GPmodel, iters=i + 1)
            print("Done.")

            print("Predicting with GP...")
            GPmodel = self.gp_predict_2D(GPmodel=GPmodel, trace=trace, iters=i + 1)
            print("Done.")

            print("Generating entropy surface...")
            self.gen_entropy_surface_data_2D(trace=trace, GPmodel=GPmodel, iters=i + 1)
            print("Done.")

            print("Beginning entropy sampling...")
            result = self.optimize_2D(trace=trace, GPmodel=GPmodel)
            xStar = np.array([bkwd(result.x[i]) for i, bkwd in enumerate(self.XTrsfBkwd)])
            print(result)
            print(xStar)
            print("Done.")

            print("Predicting next point...")
            self.call_model(xStar=xStar, XData=XData, yData=yData, iters=i + 2)
            print("Done.")
