"""Unit tests for ``gp.py``'s GP construction and log-marginal-likelihood optimization.

Covers the ``summarize``/``debug_cov`` diagnostic branches the integration suite's full
HMC runs don't exercise. ``run_mcmc`` itself stays integration-tested (it's inherently a
full HMC run, not a cheap unit) -- see ``tests/integration/test_end_to_end.py``.
"""

import gpflow
import numpy as np
import pytest

from bits_for_gaps.gp import build_gp, maximize_lml
from bits_for_gaps.kernels import AnisotropicSE


@pytest.fixture
def training_data():
    rng = np.random.default_rng(0)
    X = rng.uniform([0.0, 350.0], [1.0, 367.0], size=(8, 2))
    y = np.sin(X[:, 0:1]) + 0.01 * (X[:, 1:2] - 358.0)
    return X, y


def test_build_gp_returns_configured_gpr(training_data):
    X, y = training_data
    model = build_gp(
        X, y, mean_fxn=gpflow.mean_functions.Zero(), kernel_fxn=AnisotropicSE(), likelihood_var=0.05
    )
    assert isinstance(model, gpflow.models.GPR)
    assert float(model.likelihood.variance.numpy()) == pytest.approx(0.05)
    assert model.likelihood.variance.trainable is False


def test_build_gp_summarize_true_does_not_raise(training_data, capsys):
    # summarize=True just prints a gpflow parameter summary -- exercising it here
    # closes the one branch the integration suite (summarize always False) never hits.
    X, y = training_data
    build_gp(
        X,
        y,
        mean_fxn=gpflow.mean_functions.Zero(),
        kernel_fxn=AnisotropicSE(),
        likelihood_var=0.05,
        summarize=True,
    )
    assert "std_dev" in capsys.readouterr().out


def test_maximize_lml_reduces_training_loss(training_data):
    X, y = training_data
    model = build_gp(
        X, y, mean_fxn=gpflow.mean_functions.Zero(), kernel_fxn=AnisotropicSE(), likelihood_var=0.05
    )
    loss_before = float(model.training_loss())
    _result, fitted = maximize_lml(model)
    assert fitted is model  # same instance, mutated in place
    assert float(model.training_loss()) <= loss_before


def test_maximize_lml_debug_cov_true_does_not_raise(training_data, capsys):
    X, y = training_data
    model = build_gp(
        X, y, mean_fxn=gpflow.mean_functions.Zero(), kernel_fxn=AnisotropicSE(), likelihood_var=0.05
    )
    maximize_lml(model, debug_cov=True)
    assert "Condition number" in capsys.readouterr().out
