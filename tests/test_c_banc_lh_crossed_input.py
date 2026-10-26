"""Generated-only contracts for E255; no original graph or CUDA execution."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pytest

from tools import probe_c_banc_m3_lh_crossed_input as mod
from tools import probe_c_banc_m3_lh_neural_response as io


def test_crossed_algebra_and_signed_cancellation():
    a = np.array([[0., 0.], [0., 0.]])
    b = np.array([[1., 2.], [0., 0.]])
    c = np.array([[10., 4.], [-2., 5.]])
    d = np.array([[9., -1.], [-3., 2.]])
    got = mod.crossed_effects(a, b, c, d)
    np.testing.assert_array_equal(got["T"], got["C"] + got["S"] + got["I"])
    np.testing.assert_array_equal(got["C1"], c - b)
    np.testing.assert_array_equal(got["S1"], c - d)
    np.testing.assert_array_equal(got["C_order_mean"],
                                  (got["C"] + got["C1"]) / 2)
    assert mod.signed_window_parts(np.array([3., -2.]), 0, 2) == {
        "signed": 1., "positive": 3., "negative": -2.}
    with pytest.raises(ValueError):
        mod.crossed_effects(a, b, c, np.full((2, 2), np.nan))


def test_dynamic_start_only_allows_identity_patch_metadata():
    a = {"rate": np.array([1., 2.], np.float32), "tick": 0,
         "drive": np.zeros(2, np.float32), "model_hash": "a",
         "core_adaptation": {"identity": "a"},
         "motor_adaptation": {"identity": "a"},
         "motor_timescale": {"identity": "a"}}
    b = copy.deepcopy(a)
    for key in mod.e250.METADATA:
        b[key] = "candidate"
    mod.dynamic_start_equal(a, b, exact=lambda x, y, label: _assert_exact(x, y))
    b["rate"][0] = 3.
    with pytest.raises(AssertionError):
        mod.dynamic_start_equal(a, b, exact=lambda x, y, label: _assert_exact(x, y))


def _assert_exact(a, b):
    assert set(a) == set(b)
    for key in a:
        if isinstance(a[key], np.ndarray):
            assert mod.e254.same_bits(a[key], b[key])
        else:
            assert a[key] == b[key]


def test_diagonal_exact_is_rowwise_and_bitwise():
    keys = ("endpoint-rate.npy", "endpoint-drive.npy",
            "endpoint-adaptation.npy", "actual-current.npy",
            "edge-current64.npy")
    raw = {key: np.zeros((2, 2), np.float64) for key in keys}
    ref = {key: value.copy() for key, value in raw.items()}
    mod.diagonal_exact(raw, ref, 1)
    ref["edge-current64.npy"][1, 0] = -0.0
    with pytest.raises(AssertionError, match="edge-current64"):
        mod.diagonal_exact(raw, ref, 1)


class FakeNetwork:
    def __init__(self):
        self.tick = 999
        self.rate = np.array([7., 8., 9.], np.float32)
        self.drives = []
        self.fail_advance = False

    def snapshot(self):
        return {"tick": self.tick, "rate": self.rate.copy()}

    def restore(self, saved):
        self.tick = saved["tick"]
        self.rate = saved["rate"].copy()

    def advance(self, drive, steps):
        self.drives.append(drive.copy())
        if self.fail_advance:
            raise RuntimeError("partial advance")
        self.rate += drive
        self.tick += steps


def _input(value):
    return {"sensory": {"old_filtered": np.full((2, 1), value, np.float64),
                         "source_ids": np.array([0]),
                         "returned_source_drive": np.full((2, 1), value,
                                                          np.float32)},
            "sensory_ids": np.array([0]), "sensory_ports": np.array([0])}


def _raw(path):
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                       shape=(2, 3))
    values[:] = np.nan
    return {"drive-values.npy": values}


def _entry():
    return {"reset_exact": False, "entered_controls": 0,
            "completed_controls": 0, "unknown_neural_partial": False,
            "status": "requested"}


def _observe(net, row, incoming, raw, entry):
    entry["observed_endpoint_count"] = row + 1


def _run(net, saved, value, entry, raw):
    def reconstruct(old, ids, ports, n, desc, gain, nav, forward):
        sensory = np.zeros(n, np.float32)
        sensory[ids] = old[ports].astype(np.float32)
        return sensory, sensory.copy()

    mod.run_crossed_arm(net, saved, _input(value), {}, np.arange(3),
                        np.array([], np.int64), np.array([], np.int64), raw,
                        entry, lambda: None, lambda: None, observe=_observe,
                        controls=2, n=3, steps=1, reconstruct=reconstruct)


def test_four_bindings_reuse_two_resets_and_input_order(tmp_path):
    saved = {"tick": 0, "rate": np.zeros(3, np.float32)}
    a, b = FakeNetwork(), FakeNetwork()
    raw = {}
    for arm, net in (("00", a), ("01", a), ("11", b), ("10", b)):
        entry, raw[arm] = _entry(), _raw(tmp_path / f"{arm}.npy")
        _run(net, saved, 1. if mod.BINDINGS[arm][1] == "baseline" else 2.,
             entry, raw[arm])
        assert entry["reset_exact"] and entry["completed_controls"] == 2
        assert net.tick == 2
    assert [x[0] for x in a.drives] == [1., 1., 2., 2.]
    assert [x[0] for x in b.drives] == [2., 2., 1., 1.]
    assert np.array_equal(raw["00"]["drive-values.npy"][:, 0], [1., 1.])
    assert np.array_equal(raw["10"]["drive-values.npy"][:, 0], [1., 1.])


def test_advance_failure_keeps_entered_and_unknown_partial(tmp_path):
    saved = {"tick": 0, "rate": np.zeros(3, np.float32)}
    net, raw, entry = FakeNetwork(), _raw(tmp_path / "partial.npy"), _entry()
    net.fail_advance = True
    with pytest.raises(RuntimeError, match="partial advance"):
        _run(net, saved, 1., entry, raw)
    assert entry["entered_controls"] == 1
    assert entry["completed_controls"] == 0
    assert entry["unknown_neural_partial"] is True
    assert np.isfinite(raw["drive-values.npy"][0]).all()
    assert np.isnan(raw["drive-values.npy"][1]).all()


def test_poststep_observation_failure_preserves_completed_tick(tmp_path):
    saved = {"tick": 0, "rate": np.zeros(3, np.float32)}
    net, raw, entry = FakeNetwork(), _raw(tmp_path / "poststep.npy"), _entry()

    def failing_observer(network, row, incoming, arrays, receipt):
        if row == 1:
            raise RuntimeError("poststep readback")
        _observe(network, row, incoming, arrays, receipt)

    def reconstruct(old, ids, ports, n, desc, gain, nav, forward):
        sensory = np.zeros(n, np.float32)
        sensory[ids] = old[ports].astype(np.float32)
        return sensory, sensory.copy()

    with pytest.raises(RuntimeError, match="poststep readback"):
        mod.run_crossed_arm(
            net, saved, _input(1.), {}, np.arange(3),
            np.array([], np.int64), np.array([], np.int64), raw, entry,
            lambda: None, lambda: None, observe=failing_observer,
            controls=2, n=3, steps=1, reconstruct=reconstruct)
    assert net.tick == 1
    assert entry["entered_controls"] == 1
    assert entry["completed_controls"] == 1
    assert entry["unknown_neural_partial"] is False
    assert entry["observed_endpoint_count"] == 1


def test_factor_report_h255_and_threshold_margin():
    union = np.unique(np.r_[mod.TRACKED, mod.TARGETS])
    incoming = {"union_ids": union,
                "edge_ids": np.array([444229, 777]),
                "pre_ids": np.array([135246, 20156]),
                "post_ids": np.array([6373, 22787]),
                "target_offsets": np.array([0, 1, 2, 2, 2, 2])}
    levels = {"00": 0., "01": 1., "11": 10., "10": 9.}
    raw = {}
    for arm, level in levels.items():
        raw[arm] = {"endpoint-rate.npy": np.full((101, len(union)), level,
                                                      np.float32),
                    "endpoint-drive.npy": np.zeros((101, len(union)), np.float32),
                    "endpoint-adaptation.npy": np.zeros((101, len(union)),
                                                    np.float32),
                    "actual-current.npy": np.zeros((101, 5), np.float32),
                    "edge-current64.npy": np.column_stack(
                        (np.full(101, level, np.float64), np.zeros(101)))}
    outputs, report = mod.factors(raw, incoming, np.full(5, 7.5, np.float32))
    assert report["H255"]["H255_connectivity_effect_retained"] is True
    assert report["H255"]["T"] == 990.
    assert report["H255"]["C"] == 891.
    assert outputs["margin64_00"][0, 0] == -7.5
    assert outputs["margin64_11"][0, 0] == 2.5
    assert outputs["edge_current64_I"].shape == (101, 2)


def test_target_parameters_require_actual_model_bytes():
    class Tensor:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, ids):
            np.testing.assert_array_equal(ids, mod.TARGETS)
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Model:
        pass

    fields = {"threshold": "threshold", "coefficient": "coefficient",
              "cap": "cap", "membrane_tau": "tau",
              "cell_gain": "_cell_gain",
              "cell_adaptation_tau": "_cell_adaptation_tau",
              "inverse_adaptation_tau": "_cell_inverse_tau"}
    model, expected = Model(), {}
    for key, attr in fields.items():
        expected[key] = np.full(5, 1., np.float32)
        setattr(model, attr, Tensor(expected[key].copy()))
    mod.verify_target_parameters(model, expected)
    expected["cell_gain"][0] = -0.0
    with pytest.raises(AssertionError, match="cell_gain"):
        mod.verify_target_parameters(model, expected)


def test_gate_rejects_wrong_sha_before_loading_original(tmp_path):
    with pytest.raises(ValueError, match="freeze/review SHA"):
        mod.gate(Path("missing"), "wrong", Path("also-missing"), "wrong",
                 tmp_path / "output")


@pytest.mark.parametrize("fault", ["preexisting_cleanup", "late_manifest"])
def test_late_publication_revokes_report_and_preserves_raw(tmp_path, fault):
    (tmp_path / "raw.npy").write_bytes(b"retained")
    io.write_json(tmp_path / "factor-windows.json", {"H255": True})
    ledger = {"models": {}, "arms": {}, "completed_work": 7}
    result = {"validity": mod.VALIDITY,
              "H255_connectivity_effect_retained": True,
              "factor_report_available": True,
              "H255": {"T": 10., "C": 9., "S": 1., "I": 0.,
                       "H255_connectivity_effect_retained": True}}
    errors = []

    def fail_manifest(path, value):
        if fault == "late_manifest" and path.name == "output-manifest.json":
            raise OSError("manifest failed")
        io.write_json(path, value)

    original = OSError("cleanup failed") if fault == "preexisting_cleanup" else None
    error = mod.finish_receipts(
        tmp_path, {"resource_limits": {"internal_s": 38,
                                       "output_bytes": 100000}},
        ledger, result, errors, original, time.perf_counter(),
        persist=fail_manifest)
    assert isinstance(error, OSError)
    assert str(error) == ("cleanup failed" if original is not None
                          else "manifest failed")
    assert result["H255_connectivity_effect_retained"] is False
    assert result["H255"]["H255_connectivity_effect_retained"] is False
    assert result["H255"]["T"] == 10.
    assert result["validity"] == "invalid_or_incomplete"
    assert (tmp_path / "raw.npy").read_bytes() == b"retained"
    assert not (tmp_path / "factor-windows.json").exists()
    published = io.read_json(tmp_path / "result.json")
    assert published["H255"]["H255_connectivity_effect_retained"] is False
    assert published["H255"]["T"] == 10.
    assert published["work"]["current_probe_calls_completed"] == 0
    assert io.read_json(tmp_path / "advance-ledger.json")["completed_work"] == 7
    if fault == "late_manifest":
        assert not (tmp_path / "output-manifest.json").exists()
