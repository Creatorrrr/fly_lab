"""Generated-data E265 checks; never construct the original neural model."""

import copy
import time

import numpy as np
import pytest

from tools import probe_c_banc_m3_lh_bridge_current as probe


def test_decomposition_signs_and_hypothesis():
    b = np.zeros((101, 3), np.float64)
    z = b.copy()
    z[:, 0] = np.arange(101) + 1.
    z[:, 1] = -.01
    drive = np.full(101, -.2)
    adaptation = np.full(101, .03)
    parts = probe.decomposition(z, b, drive, np.zeros(101), adaptation,
                                np.zeros(101), np.array([True, False, False]))
    assert np.allclose(parts["D"], np.arange(101) + .76)
    assert np.allclose(parts["S"], np.arange(101) + .8)
    assert np.allclose(parts["O"], -.04)
    assert probe.assess_h265(parts)[probe.HKEY] is True
    parts["O"][:] = 100
    assert probe.assess_h265(parts)["classification"] == "refuted"


@pytest.mark.parametrize("case", [None, 43, 56, 70])
def test_zero_denominator_inconclusive(case):
    d = np.ones(101)
    if case is None:
        d[:] = 0
    else:
        d[case] = 0
    result = probe.assess_h265({"D": d, "S": np.arange(101), "O": d * .1})
    assert result[probe.HKEY] is None
    assert result["classification"] == "inconclusive"
    if case is not None:
        assert result["cases"][str(case)]["other_fraction"] is None


def test_unsorted_roster_resolves_identity():
    assert probe.e264.observed_columns(np.array([9, 1, 7], np.int64),
        np.array([7, 9], np.int64)).tolist() == [2, 0]
    with pytest.raises(ValueError):
        probe.e264.observed_columns(np.array([9, 1, 7], np.int64), np.array([2], np.int64))


def test_packed_sum_lane_order_and_ftz():
    rates = np.ones(65, np.float32)
    weights = np.zeros(65, np.float32)
    weights[[0, 1, 32]] = [1e8, 1, -1e8]
    assert probe.packed_sum32(weights, rates, rates) == 1
    assert np.float32(weights.sum(dtype=np.float64)) == 1
    assert probe.packed_sum32(np.array([1e-40], np.float32), rates[:1], rates[:1]) == 0


class Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def __getitem__(self, item):
        return Tensor(self.array[item])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class GeneratedNetwork:
    def __init__(self, fail_current=False, mutate=False):
        self.state = {"rate": np.ones(probe.TARGET + 1, np.float32),
                      "output_mask": np.ones(probe.TARGET + 1, np.float32), "tick": 0}
        self.rate = Tensor(self.state["rate"])
        self.fail_current, self.mutate = fail_current, mutate

    def snapshot(self):
        return copy.deepcopy(self.state)

    def _current(self, rate):
        if self.mutate:
            self.state["tick"] += 1
        if self.fail_current:
            raise RuntimeError("first current failure")
        result = np.zeros(probe.TARGET + 1, np.float32)
        result[probe.TARGET] = 3
        return Tensor(result)


def make_probe_raw(folder):
    raw = {}
    for name, shape, dtype in (("incoming-rate.npy", (1, 3), np.float32),
                              ("incoming-mask.npy", (1, 3), np.float32),
                              ("edge-current64.npy", (1, 3), np.float64),
                              ("actual-current.npy", (1,), np.float32),
                              ("packed-current32.npy", (1,), np.float32),
                              ("rounding-difference64.npy", (1,), np.float64)):
        raw[name] = np.lib.format.open_memmap(folder / name, mode="w+", shape=shape, dtype=dtype)
        raw[name][:] = np.nan
    return raw


def exact(actual, expected, label):
    assert actual["tick"] == expected["tick"], label
    for key in ("rate", "output_mask"):
        assert probe.e254.same_bits(actual[key], expected[key]), label


@pytest.mark.parametrize("fail_current,mutate", [(False, False), (True, False), (False, True)])
def test_readonly_probe_intent_and_partial_raw(tmp_path, fail_current, mutate):
    network = GeneratedNetwork(fail_current, mutate)
    raw = make_probe_raw(tmp_path)
    context = {"weights": np.ones(3, np.float32), "spec": {"pre_ids": np.array([2, 5, 8])}}
    entry = {key: 0 for key in ("current_probe_requested", "current_probe_entered",
                               "current_probe_completed", "current_probe_snapshot_exact")}
    receipts = []
    action = lambda: probe.current_probe(network, 0, raw, entry, context,
                                         lambda: receipts.append(copy.deepcopy(entry)), exact=exact)
    if fail_current:
        with pytest.raises(RuntimeError, match="first current failure"):
            action()
    elif mutate:
        with pytest.raises(AssertionError):
            action()
    else:
        action()
        assert raw["actual-current.npy"][0] == 3
        assert entry["current_probe_snapshot_exact"] == 1
    assert np.array_equal(raw["edge-current64.npy"][0], np.ones(3))
    assert entry["current_probe_entered"] == 1
    assert entry["current_probe_completed"] == (0 if fail_current else 1)
    assert entry["unknown_current_partial"] is fail_current
    assert receipts


def test_failing_advance_keeps_intent():
    class Failing:
        def advance(self, drive, ticks):
            raise RuntimeError("partial advance")
    entry = {"completed_controls": 2}
    receipts = []
    with pytest.raises(RuntimeError, match="partial advance"):
        probe.advance_with_receipt(Failing(), np.zeros(4), 3, entry,
                                  lambda: receipts.append(copy.deepcopy(entry)))
    assert entry["entered_controls"] == 3
    assert entry["completed_controls"] == 2
    assert entry["unknown_neural_partial"] is True
    assert receipts[0]["unknown_neural_partial"] is True


def test_fresh_output_gate(tmp_path):
    out = tmp_path / "new"
    probe.require_fresh_output(out, out, 0)
    with pytest.raises(ValueError):
        probe.require_fresh_output(out, tmp_path / "other", 0)
    out.mkdir()
    with pytest.raises(FileExistsError):
        probe.require_fresh_output(out, out, 0)


def test_publication_failure_revokes_h_and_keeps_first_error(tmp_path):
    report = {"H265": {probe.HKEY: True, "classification": "supported"}}
    probe.e248.write_json(tmp_path / "factor-windows.json", report)
    result = {probe.HKEY: True, "H265_classification": "supported", "H265": copy.deepcopy(report["H265"])}
    ledger = {"arms": {}, "models": {}}
    errors = []
    first = RuntimeError("original error")
    def persist(path, value):
        if path.name == "output-manifest.json":
            raise OSError("publication failure")
        probe.e248.write_json(path, value)
    returned = probe.finish_receipts(tmp_path, {"resource_limits": {"internal_s": 100, "output_bytes": 100000}},
        ledger, result, errors, first, time.perf_counter(), persist=persist)
    assert returned is first
    assert result[probe.HKEY] is None
    assert result["H265_classification"] == "not_tested"
    assert result["first_error"]["message"] == "original error"
    assert probe.e248.read_json(tmp_path / "factor-windows.json")["H265"][probe.HKEY] is None
    assert not (tmp_path / "output-manifest.json").exists()
