"""Unit tests for a few `adaptiveEntropy`/`BitsForGaps` code paths the rest of the
suite doesn't reach directly: the legacy disk-based `read_data`, `BitsForGaps`'s
optional custom `input_transform`/`output_transform` override, and the thin
`sample_gp_posterior_mixture`/`entropy_objective` instance-method wrappers over
`mixture.py`/`acquisition.py` (used by callers who want the sampler's own
seed/noGaussians/acquisitionObjective config applied automatically, but not called by
`run()` itself -- `run()` and every other test in this suite reach the module-level
functions directly instead, e.g. `tests/integration/test_end_to_end.py`).
"""

import gpflow
import numpy as np
import pytest

from bits_for_gaps import acquisition
from bits_for_gaps.kernels import AnisotropicSE
from bits_for_gaps.sampler import BitsForGaps, adaptiveEntropy
from bits_for_gaps.transforms import InputTransform, OutputTransform

BOUNDS_2D = [(0.0, 1.0), (350.0, 367.0)]


def _fwd_model(x1, x2):
    return [float(np.sin(x1) + x2)]


def _build_sampler():
    return adaptiveEntropy(
        exp_name="wrapper_test",
        iters=1,
        x_bounds=BOUNDS_2D,
        likelihood_var=0.05,
        mean_fxn=gpflow.mean_functions.Zero(),
        kernel_fxn=AnisotropicSE(),
        fwd_model=_fwd_model,
        fwd_model_args=(),
    )


def _fitted_gp_model():
    rng = np.random.default_rng(0)
    X = rng.uniform([0.0, 350.0], [1.0, 367.0], size=(8, 2))
    y = np.sin(X[:, 0:1]) + 0.01 * (X[:, 1:2] - 358.0)
    model = gpflow.models.GPR(data=(X, y), kernel=AnisotropicSE())
    gpflow.set_trainable(model.likelihood.variance, False)
    model.likelihood.variance.assign(0.05)
    return model


def _small_trace():
    rng = np.random.default_rng(1)
    return rng.uniform([0.5, 0.5, 0.5], [3.0, 3.0, 3.0], size=(20, 3))


def test_read_data_splits_columns_correctly(tmp_path):
    # read_data is the legacy disk-as-state convention (run_model's precondition) --
    # not called by run(), but kept for scripts that still want to seed from a file.
    s = adaptiveEntropy(
        exp_name="legacy_test",
        iters=1,
        x_bounds=BOUNDS_2D,
        likelihood_var=0.05,
        mean_fxn=gpflow.mean_functions.Zero(),
        kernel_fxn=AnisotropicSE(),
        fwd_model=_fwd_model,
        fwd_model_args=(),
    )
    s.path = str(tmp_path)
    X = np.array([[0.1, 355.0], [0.2, 356.0], [0.3, 357.0]])
    y = np.array([[1.0], [2.0], [3.0]])
    np.savetxt(tmp_path / "activity_data_1", np.column_stack([X, y]))

    XData, yData = s.read_data(iters=1)
    np.testing.assert_allclose(XData, X)
    np.testing.assert_allclose(yData, y)


def test_bits_for_gaps_accepts_custom_input_output_transform():
    custom_input = InputTransform(
        forward_fns=[lambda x: np.log(x + 0.1), lambda t: (t - 350.0) / 17.0],
        backward_fns=[lambda x: np.exp(x) - 0.1, lambda t: 17.0 * t + 350.0],
    )
    custom_output = OutputTransform(forward_fn=np.log, backward_fn=np.exp)

    bfg = BitsForGaps(
        black_box=_fwd_model,
        bounds=BOUNDS_2D,
        kernel=AnisotropicSE(),
        input_transform=custom_input,
        output_transform=custom_output,
    )
    assert bfg.input_transform is custom_input
    assert bfg.output_transform is custom_output
    # And not silently ignored: forward/backward actually use the custom fns.
    np.testing.assert_allclose(bfg.input_transform.forward([[0.0, 350.0]])[0], [np.log(0.1), 0.0])
    assert bfg.output_transform.forward(1.0) == pytest.approx(0.0)


def test_bits_for_gaps_defaults_to_identity_transform_when_not_given():
    bfg = BitsForGaps(black_box=_fwd_model, bounds=BOUNDS_2D, kernel=AnisotropicSE())
    np.testing.assert_allclose(bfg.input_transform.forward([[0.3, 360.0]]), [[0.3, 360.0]])
    assert bfg.output_transform.forward(5.0) == 5.0


def test_sample_gp_posterior_mixture_forwards_seed_and_explicit_size(monkeypatch):
    # mixture.sample_gp_posterior_mixture draws from TF's ambient, unseeded RNG
    # (predict_f_samples -- see mixture.py's module docstring), and `size` selects
    # WHICH trace rows are eligible, not the output shape (fixed at 100 draws) -- so
    # neither is observable from the return value alone. What a caller actually
    # depends on is that the wrapper forwards `self.seed` and the given `size`
    # unchanged to the module-level function; verified directly here.
    calls = []

    def fake_sample(trace, GPmodel, XGP, seed, size, tf_seed=None):
        calls.append({"seed": seed, "size": size})
        return np.zeros((size, len(XGP)))

    monkeypatch.setattr("bits_for_gaps.sampler.mixture.sample_gp_posterior_mixture", fake_sample)

    s = _build_sampler()
    s.seed = 42
    GPmodel = _fitted_gp_model()
    trace = _small_trace()
    XGP = np.array([[0.3, 0.4], [0.6, 0.5]])

    s.sample_gp_posterior_mixture(trace, GPmodel, XGP, size=5)
    assert calls == [{"seed": 42, "size": 5}]


def test_sample_gp_posterior_mixture_default_size_uses_no_gaussians(monkeypatch):
    calls = []

    def fake_sample(trace, GPmodel, XGP, seed, size, tf_seed=None):
        calls.append({"seed": seed, "size": size})
        return np.zeros((size, len(XGP)))

    monkeypatch.setattr("bits_for_gaps.sampler.mixture.sample_gp_posterior_mixture", fake_sample)

    s = _build_sampler()
    s.seed = 7
    s.noGaussians = 11
    GPmodel = _fitted_gp_model()
    trace = _small_trace()
    XGP = np.array([[0.3, 0.4], [0.6, 0.5]])

    s.sample_gp_posterior_mixture(trace, GPmodel, XGP)  # size=None -> self.noGaussians
    assert calls == [{"seed": 7, "size": 11}]


def test_entropy_objective_delegates_instance_config():
    s = _build_sampler()
    s.seed = 42
    s.noGaussians = 5
    s.acquisitionObjective = "lower_bound"
    GPmodel = _fitted_gp_model()
    trace = _small_trace()
    xStarGP = np.array([0.4, 358.0])

    result = s.entropy_objective(xStarGP, trace, GPmodel)
    expected = acquisition.entropy_objective(
        xStarGP, trace, GPmodel, seed=42, no_gaussians=5, objective="lower_bound"
    )
    assert result == pytest.approx(expected)


def test_run_model_reads_disk_design_and_runs(tmp_path):
    # run_model is the deprecated zero-argument entry point: read_data(iters=1) then
    # run(..., checkpoint_dir=self.path) -- neither step is exercised together by any
    # other test (read_data alone is covered above; run() is always called directly
    # elsewhere with an in-memory design).
    s = _build_sampler()
    s.path = str(tmp_path)
    s.noSamples, s.noBurnIn, s.noChains = 50, 20, 2
    s.noGaussians, s.entropyMesh, s.noRestarts = 5, [3, 3], 2

    X = np.array([[0.1, 355.0], [0.3, 357.0], [0.5, 359.0], [0.7, 361.0], [0.9, 363.0]])
    y = np.array([[float(np.sin(x1) + x2)] for x1, x2 in X])
    np.savetxt(tmp_path / "activity_data_1", np.column_stack([X, y]))

    history = s.run_model()
    assert len(history) == 1
    assert history.last.XData.shape[0] == 6  # the 5 seed points + 1 newly selected
    assert (tmp_path / "activity_data_2").exists()  # checkpoint_dir=self.path, opt-in


def test_run_with_show_lml_results_prints_diagnostics(capsys):
    # showLMLres=True additionally prints the LML fit's result + a gpflow parameter
    # summary -- off by default (test_run_with_initial_lml_maximization in
    # test_end_to_end.py exercises initalLML=True alone, which stays silent).
    s = _build_sampler()
    s.initalLML = True
    s.showLMLres = True
    s.noSamples, s.noBurnIn, s.noChains = 50, 20, 2
    s.noGaussians, s.entropyMesh, s.noRestarts = 5, [3, 3], 2

    X = np.array([[0.1, 355.0], [0.3, 357.0], [0.5, 359.0], [0.7, 361.0], [0.9, 363.0]])
    y = np.array([float(np.sin(x1) + x2) for x1, x2 in X])

    history = s.run(X, y)
    assert history.last.lml_result is not None
    out = capsys.readouterr().out
    assert "std_dev" in out  # gpflow.utilities.print_summary's parameter table
