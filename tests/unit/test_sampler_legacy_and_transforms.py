"""Unit tests for a few `adaptiveEntropy`/`BitsForGaps` code paths the rest of the
suite doesn't reach directly: the legacy disk-based `read_data`, and `BitsForGaps`'s
optional custom `input_transform`/`output_transform` override.
"""

import gpflow
import numpy as np
import pytest

from bits_for_gaps.kernels import AnisotropicSE
from bits_for_gaps.sampler import BitsForGaps, adaptiveEntropy
from bits_for_gaps.transforms import InputTransform, OutputTransform

BOUNDS_2D = [(0.0, 1.0), (350.0, 367.0)]


def _fwd_model(x1, x2):
    return [float(np.sin(x1) + x2)]


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
