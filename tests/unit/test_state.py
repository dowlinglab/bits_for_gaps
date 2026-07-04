"""Unit tests for the in-memory run-state containers."""
import numpy as np

from bits_for_gaps.state import IterationRecord, RunHistory


def _record(i):
    return IterationRecord(
        iteration=i, XData=np.zeros((i, 2)), yData=np.zeros((i, 1)), GPmodel=None,
        trace=np.zeros((3, 3)), chains_states=np.zeros((3, 2, 3)),
        rhat=np.ones(3), ess=np.ones(3) * 10,
    )


def test_empty_history():
    h = RunHistory()
    assert len(h) == 0


def test_append_and_index():
    h = RunHistory()
    h.append(_record(1))
    h.append(_record(2))
    assert len(h) == 2
    assert h[0].iteration == 1
    assert h[1].iteration == 2
    assert h.last.iteration == 2


def test_iterable():
    h = RunHistory()
    h.append(_record(1))
    h.append(_record(2))
    iters = [r.iteration for r in h]
    assert iters == [1, 2]


def test_iteration_record_optional_fields_default_none():
    r = _record(1)
    assert r.entropy_field is None
    assert r.xStar is None
    assert r.max_entropy is None
    assert r.lml_result is None


def test_iteration_record_optional_fields_settable():
    r = IterationRecord(
        iteration=1, XData=np.zeros((1, 2)), yData=np.zeros((1, 1)), GPmodel=None,
        trace=np.zeros((3, 3)), chains_states=np.zeros((3, 2, 3)),
        rhat=np.ones(3), ess=np.ones(3), entropy_field=np.ones((4, 3)),
        xStar=np.array([0.1, 0.2]), max_entropy=1.5,
    )
    assert r.max_entropy == 1.5
    np.testing.assert_array_equal(r.xStar, [0.1, 0.2])
