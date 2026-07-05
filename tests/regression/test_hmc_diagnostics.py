"""Regression: HMC convergence diagnostics vs the published run (paper Fig 10).

Reads the committed golden snapshot (``paper/golden/hmc_diagnostics.json``) extracted
from the archived iteration-15 run and pins it against the values reported in the paper.
This locks the golden file against corruption and documents the reproduction tolerances
that ``paper/reproduce.py`` will diff against in Phase 7.

Pure (JSON only) -- no TensorFlow, no Julia.
"""

import numpy as np


def test_rhat_matches_paper_figure_10(golden):
    g = golden("hmc_diagnostics.json")
    rhat = np.array(g["rhat"])
    paper = np.array(g["rhat_paper_rounded"])
    assert rhat.shape == (3,)
    np.testing.assert_allclose(rhat, paper, atol=g["tol"]["rhat_atol"])


def test_ess_matches_paper_figure_10(golden):
    g = golden("hmc_diagnostics.json")
    ess = np.array(g["ess"])
    paper = np.array(g["ess_paper_rounded"])
    assert ess.shape == (3,)
    np.testing.assert_allclose(ess, paper, rtol=g["tol"]["ess_rtol"])


def test_chains_converged(golden):
    # R-hat within [1, 1.01] is the usual "converged" heuristic; the published run
    # sits comfortably inside it.
    g = golden("hmc_diagnostics.json")
    rhat = np.array(g["rhat"])
    assert np.all(rhat >= 1.0)
    assert np.all(rhat < 1.01)


def test_ess_positive_and_bounded(golden):
    # ESS must be positive and cannot exceed the nominal 5000 draws x 4 chains here.
    g = golden("hmc_diagnostics.json")
    ess = np.array(g["ess"])
    assert np.all(ess > 0)
    assert np.all(ess < 5000 * 4)
