"""Parity test for the ``BitsForGaps`` public-API facade (REFACTOR_PLAN.md Sec 4).

``BitsForGaps`` is a thin, renamed-kwarg subclass of ``adaptiveEntropy`` -- it adds no
new computation, so a run through it must reproduce the exact same pinned baseline as
``adaptiveEntropy`` (see ``tests/integration/test_end_to_end.py`` and
``tests/integration/data/synthetic_baseline.json``).
"""

import json

import numpy as np
import pytest
from test_end_to_end import BASELINE_PATH, BOUNDS, SEED, _fwd_model, _initial_design

from bits_for_gaps.kernels import AnisotropicSE
from bits_for_gaps.sampler import BitsForGaps


def _build_facade(seed=SEED):
    bfg = BitsForGaps(
        black_box=_fwd_model, bounds=BOUNDS, kernel=AnisotropicSE(), likelihood_variance=0.05
    )
    bfg.seed = seed
    bfg.noSamples = 100
    bfg.noBurnIn = 50
    bfg.noChains = 2
    bfg.noGaussians = 8
    bfg.entropyMesh = [4, 4]
    bfg.noRestarts = 3
    return bfg


@pytest.mark.slow
def test_bits_for_gaps_facade_matches_pinned_baseline():
    X_init, y_init = _initial_design(n=12)
    record = _build_facade().run(X_init, y_init).last

    with open(BASELINE_PATH) as f:
        base = json.load(f)

    np.testing.assert_allclose(record.rhat, base["rhat"], atol=1e-10)
    np.testing.assert_allclose(record.ess, base["ess"], atol=1e-10)
    np.testing.assert_allclose(record.entropy_field, base["entropy"], atol=1e-10)
    np.testing.assert_allclose(record.xStar, base["xStar"], atol=1e-10)
    assert record.max_entropy == pytest.approx(base["max_entropy"], abs=1e-10)
