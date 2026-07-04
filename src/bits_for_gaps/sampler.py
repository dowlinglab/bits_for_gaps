"""The BITS for GAPS sequential-design engine.

Phase 4: ``adaptiveEntropy`` is now an *orchestrator* over the decomposed modules --
:mod:`bits_for_gaps.gp` (GP construction + HMC + R-hat/ESS), :mod:`bits_for_gaps.mixture`
(GMM predictive posterior), :mod:`bits_for_gaps.acquisition` (entropy objective + N-D
optimizer), :mod:`bits_for_gaps.transforms` (per-dimension input/output transforms), and
:mod:`bits_for_gaps.state` (in-memory run history). It no longer contains the algorithm
math itself -- that lives in the modules above, as pure functions over explicit
arguments, independently testable and reusable.

Disk-as-state is retired: :meth:`adaptiveEntropy.run` takes the initial design directly
(in memory) and returns a :class:`bits_for_gaps.state.RunHistory` -- a full run executes
with zero disk writes by default. File output (mirroring the paper code's per-iteration
``np.savetxt``/``pickle`` dump under ``results/{exp_name}/``) is available but opt-in via
``checkpoint_dir``. :meth:`run_model` is kept as a deprecated, disk-based shim for
scripts still relying on the original zero-argument convention.

The decomposition is behavior-preserving: ``tests/integration/data/synthetic_baseline.json``
pins the tiny seeded synthetic run's exact outputs from before this decomposition, and
the acquisition/gp/mixture modules were verified bit-exact (atol=1e-12) against the
pre-decomposition methods before this rewrite.

Phase 5: generalized to N input dimensions. ``optimize`` (was ``optimize_2D``) -- the
acquisition path a run actually depends on -- is dimension-general. ``predict_grid_2D``
and ``entropy_surface_2D`` stay 2-D-only (dense grids are exponential in d; they are
visualization diagnostics that don't feed the acquisition): :meth:`run` only calls
``entropy_surface_2D`` when ``len(self.XBnds) == 2``, leaving ``entropy_field=None``
otherwise, and both raise a clear ``ValueError`` if called directly for d != 2.
``call_model`` calls the injected black box as ``FwdModel(*FwdModelArgs, *xStar)`` --
``xStar``'s components in natural dimension order (was a 2-D-specific, reversed
``FwdModel(*args, x2, x1)`` convention inherited from the VLE example's Julia call).
"""

import os
import pickle

import gpflow
import numpy as np

from . import acquisition, gp, mixture
from .state import IterationRecord, RunHistory
from .transforms import InputTransform, OutputTransform

f64 = gpflow.utilities.to_default_float


class adaptiveEntropy:
    def __init__(self, exp_name, iters, x_bounds, likelihood_var, mean_fxn, kernel_fxn,
                 fwd_model, fwd_model_args):
        """
        exp_name, iters, x_bounds, likelihood_var, mean_fxn, kernel_fxn,
        fwd_model, fwd_model_args
        """
        ## Miscellaneous
        self.path = os.path.join("results", exp_name)   # default checkpoint dir (opt-in)
        self.seed = 123                                 # random seed

        ## Data related
        self.XBnds          = x_bounds
        self.input_transform  = InputTransform(ndim=len(x_bounds))
        self.output_transform = OutputTransform()

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
        self.kernelFxn           = kernel_fxn
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

    ## ------------------------------------------------------------------
    ## Optional disk I/O (legacy convenience; not used internally by run())
    ## ------------------------------------------------------------------

    def read_data(self, iters):
        """Read an ``activity_data_{iters}`` file from ``self.path`` (legacy convention).

        Not called by :meth:`run` (which takes the initial design in memory); kept for
        scripts that still want to seed a run from a file, and by :meth:`run_model`.
        """
        data = np.loadtxt(os.path.join(self.path, f"activity_data_{iters}"))
        nFeatures = len(self.XBnds)
        XData = data[:, :nFeatures]
        yData = data[:, [nFeatures]]
        return XData, yData

    ## ------------------------------------------------------------------
    ## Thin delegating wrappers over gp.py / mixture.py / acquisition.py
    ## ------------------------------------------------------------------

    def trsf_data(self, XData, yData):
        XData = np.atleast_2d(XData)
        yData = np.atleast_1d(yData).reshape(-1, 1)
        XGP = self.input_transform.forward(XData)
        yGP = self.output_transform.forward(yData)
        return XGP, yGP

    def build_gp(self, XGP, yGP):
        return gp.build_gp(XGP, yGP, mean_fxn=self.meanFxn, kernel_fxn=self.kernelFxn,
                           likelihood_var=self.likelihoodVar, summarize=self.summarizeGP)

    def maximize_lml(self, GPmodel):
        return gp.maximize_lml(GPmodel, debug_cov=self.debugCov)

    def run_mcmc(self, GPmodel):
        """Returns ``(trace, chains_states, rhat, ess, GPmodel)`` -- see ``gp.run_mcmc``."""
        return gp.run_mcmc(GPmodel, seed=self.seed, no_samples=self.noSamples,
                           no_burn_in=self.noBurnIn, no_chains=self.noChains,
                           no_leapfrog_steps=self.noLeapfrogSteps, step_size=self.stepSize,
                           no_adapt_steps=self.noAdaptSteps, target_accept=self.targetAccept,
                           adapt_rate=self.adaptRate)

    def sample_gp_posterior_mixture(self, trace, GPmodel, XGP, size=None):
        if size is None:
            size = self.noGaussians
        return mixture.sample_gp_posterior_mixture(trace, GPmodel, XGP, seed=self.seed,
                                                    size=size)

    def predict_grid_2D(self, GPmodel, trace):
        """Full-grid GP posterior-predictive samples (plotting-only diagnostic).

        Expensive and does not feed the acquisition -- not called by :meth:`run` unless
        ``predict_grid=True``. See ``mixture.predict_grid_2D`` for why its output is not
        bitwise-reproducible (``predict_f_samples`` draws from TF's ambient RNG).
        """
        return mixture.predict_grid_2D(
            trace, GPmodel, x_bounds=self.XBnds,
            x_trsf_fwd=self.input_transform.forward_fns,
            x_trsf_bkwd=self.input_transform.backward_fns,
            y_trsf_bkwd=self.output_transform.backward,
            seed=self.seed, size=self.noGPpredictions,
        )

    def entropy_objective(self, xStarGP, trace, GPmodel):
        return acquisition.entropy_objective(xStarGP, trace, GPmodel, seed=self.seed,
                                             no_gaussians=self.noGaussians)

    def entropy_surface_2D(self, trace, GPmodel):
        return acquisition.entropy_surface_2D(
            trace, GPmodel, x_bounds=self.XBnds, mesh=self.entropyMesh,
            x_trsf_fwd=self.input_transform.forward_fns,
            x_trsf_bkwd=self.input_transform.backward_fns,
            seed=self.seed, no_gaussians=self.noGaussians,
        )

    def optimize(self, trace, GPmodel):
        """Dimension-general acquisition optimizer (was ``optimize_2D``); see
        ``acquisition.optimize``.
        """
        return acquisition.optimize(
            trace, GPmodel, x_bounds=self.XBnds,
            x_trsf_fwd=self.input_transform.forward_fns,
            seed=self.seed, no_gaussians=self.noGaussians, no_restarts=self.noRestarts,
        )

    def call_model(self, xStar, XData, yData):
        """Evaluate the injected black box at ``xStar`` and append it to the design.

        Calls ``self.FwdModel(*self.FwdModelArgs, *xStar)`` -- ``xStar``'s components
        in natural dimension order (Phase 5: was a 2-D-specific, reversed
        ``FwdModel(*args, x2, x1)`` convention inherited from the VLE example's Julia
        call). Returns the extended ``(XData, yData)`` -- no disk write.
        """
        xStar = np.asarray(xStar, dtype=float).reshape(-1)
        XData = np.atleast_2d(np.asarray(XData, dtype=float))
        yData = np.atleast_1d(np.asarray(yData, dtype=float)).reshape(-1, 1)
        yStar = np.array(self.FwdModel(*self.FwdModelArgs, *xStar))
        XData_new = np.vstack([XData, xStar])
        yData_new = np.vstack([yData, [[float(yStar[0])]]])
        return XData_new, yData_new

    ## ------------------------------------------------------------------
    ## Orchestration
    ## ------------------------------------------------------------------

    def run(self, X_init, y_init, checkpoint_dir=None, predict_grid=False):
        """Run the sequential BITS-for-GAPS design loop for ``self.noIters`` iterations.

        Parameters
        ----------
        X_init, y_init : array-like
            Initial design (physical input space) and observed black-box outputs.
        checkpoint_dir : str, optional
            If given, write each iteration's artifacts to this directory (mirrors the
            paper code's per-iteration file dump). Off by default -- the run executes
            entirely in memory.
        predict_grid : bool
            If True, additionally compute (but discard, unless ``checkpoint_dir`` is
            set) the full-grid GP posterior-predictive samples used only for plotting
            (the paper code's ``gp_predict_2D``). Expensive, 2-D-only, and does not
            feed the acquisition -- off by default; raises if the input space is not
            2-D.

        Returns
        -------
        bits_for_gaps.state.RunHistory
        """
        XData = np.atleast_2d(np.asarray(X_init, dtype=float))
        yData = np.atleast_1d(np.asarray(y_init, dtype=float)).reshape(-1, 1)
        history = RunHistory()

        for i in range(self.noIters):
            it = self.startIter + i + 1
            print(f"On iteration {it}")

            XGP, yGP = self.trsf_data(XData, yData)
            GPmodel = self.build_gp(XGP, yGP)

            lml_result = None
            if self.initalLML:
                lml_result, GPmodel = self.maximize_lml(GPmodel)
                if self.showLMLres:
                    print(lml_result)
                    gpflow.utilities.print_summary(GPmodel)

            trace, chains_states, rhat, ess, GPmodel = self.run_mcmc(GPmodel)

            grid_predictions = None
            if predict_grid:
                grid_predictions = self.predict_grid_2D(GPmodel, trace)

            # entropy_surface_2D is a 2-D-only visualization diagnostic (dense grids
            # are exponential in d) -- skip it for N-D input spaces rather than error,
            # since it does not feed the acquisition below.
            entropy_field = None
            if len(self.XBnds) == 2:
                entropy_field = self.entropy_surface_2D(trace, GPmodel)

            opt_result = self.optimize(trace, GPmodel)
            xStar = self.input_transform.backward(opt_result.x.reshape(1, -1))[0]
            max_entropy = -float(opt_result.fun)

            XData, yData = self.call_model(xStar, XData, yData)

            record = IterationRecord(
                iteration=it, XData=XData, yData=yData, GPmodel=GPmodel, trace=trace,
                chains_states=chains_states, rhat=rhat, ess=ess,
                entropy_field=entropy_field, xStar=xStar, max_entropy=max_entropy,
                lml_result=lml_result,
            )
            history.append(record)

            if checkpoint_dir:
                self._write_checkpoint(checkpoint_dir, record, grid_predictions)

        return history

    def run_model(self):
        """Deprecated legacy entry point.

        Reads the iteration-1 design from ``activity_data_1`` under ``self.path`` (the
        original disk-as-state precondition) and writes checkpoints to ``self.path`` on
        every iteration, matching the paper code's ``run_model``. Prefer :meth:`run`,
        which takes the initial design in memory and makes checkpointing opt-in.
        """
        XData, yData = self.read_data(iters=1)
        return self.run(XData, yData, checkpoint_dir=self.path)

    def _write_checkpoint(self, checkpoint_dir, record, grid_predictions=None):
        """Persist one iteration's artifacts to disk (opt-in; off by default).

        A Phase-4, best-effort equivalent of the paper code's per-iteration file dump --
        not guaranteed byte-identical to the original file layout (e.g. the original
        also wrote an intermediate ``gp_training_data_`` file); intended for users who
        want on-disk artifacts, not as the mechanism for state hand-off between
        iterations (see module docstring).
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        it = record.iteration
        np.savetxt(os.path.join(checkpoint_dir, f"rhat_value_{it}.txt"), record.rhat,
                   header="R-hat value")
        np.savetxt(os.path.join(checkpoint_dir, f"ess_value_{it}.txt"), record.ess,
                   header="ess value")
        np.savetxt(os.path.join(checkpoint_dir, f"param_posterior_samples_{it}"), record.trace)
        for c in range(record.chains_states.shape[1]):
            np.savetxt(os.path.join(checkpoint_dir, f"traces_chain_{c}_exp_{it}"),
                      record.chains_states[:, c, :])
        if record.entropy_field is not None:
            np.savetxt(os.path.join(checkpoint_dir, f"entropy_{it}"), record.entropy_field)
        if grid_predictions is not None:
            np.savetxt(os.path.join(checkpoint_dir, f"gp_predict_{it}"), grid_predictions)
        np.savetxt(os.path.join(checkpoint_dir, f"activity_data_{it + 1}"),
                  np.column_stack([record.XData, record.yData]))
        with open(os.path.join(checkpoint_dir, f"gp_model_{it}.pkl"), "wb") as f:
            pickle.dump(record.GPmodel, f)


class BitsForGaps(adaptiveEntropy):
    """Public-API-friendly constructor for the BITS-for-GAPS sequential-design engine.

    A thin wrapper over :class:`adaptiveEntropy` (kept for backward compatibility)
    using the target public kwarg names from REFACTOR_PLAN.md Sec 4. Numerically
    identical to ``adaptiveEntropy`` -- no new computation, just friendlier
    constructor names; all methods (including :meth:`run`) are inherited unchanged.
    Advanced/legacy configuration (HMC tuning, restarts, mesh density, ...) is still set
    via the same instance attributes as ``adaptiveEntropy`` (e.g. ``.noSamples``).

    TODO(Phase 6): once the VLE example is ported onto this API, this can grow an
    ``mcmc=MCMCConfig(...)``-style kwarg for HMC tuning, replacing the passthrough
    instance attributes inherited from ``adaptiveEntropy`` (e.g. ``.noSamples``).
    """

    def __init__(self, black_box, bounds, kernel, mean_fxn=None, likelihood_variance=0.05,
                 exp_name="bits_for_gaps_run", iters=1, fwd_model_args=(),
                 input_transform=None, output_transform=None):
        super().__init__(
            exp_name=exp_name, iters=iters, x_bounds=bounds,
            likelihood_var=likelihood_variance,
            mean_fxn=mean_fxn if mean_fxn is not None else gpflow.mean_functions.Zero(),
            kernel_fxn=kernel, fwd_model=black_box, fwd_model_args=fwd_model_args,
        )
        if input_transform is not None:
            self.input_transform = input_transform
        if output_transform is not None:
            self.output_transform = output_transform
