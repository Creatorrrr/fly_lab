"""Generated-only E248 contracts; never instantiate the original BANC/CUDA model."""

import json
import subprocess
import sys
import time

import numpy as np
import pytest

from tools import probe_c_banc_m3_lh_neural_response as e248


def _rate(value, width=5):
    return np.full((41, width), value, np.float32)


def test_mask_union_preserves_preexisting_zeros_and_only_requested_rows():
    original = np.array([1, 0, 1, 1, 1, 0], np.float32)
    source = np.array([0, 2], np.int64)
    bridge = np.array([3], np.int64)
    assert np.array_equal(e248.select_mask(original, source, bridge, "intact"), original)
    assert np.array_equal(e248.select_mask(original, source, bridge, "bridge"),
                          [1, 0, 1, 0, 1, 0])
    assert np.array_equal(e248.select_mask(original, source, bridge, "sensory"),
                          [0, 0, 0, 1, 1, 0])
    assert np.array_equal(original, [1, 0, 1, 1, 1, 0])
    with pytest.raises(ValueError):
        e248.select_mask(original, source, bridge, "not-an-arm")


def test_held_drive_updates_only_source_and_uses_float32_rounding():
    base = np.array([7., -2., 5000., 0.], np.float32)
    out = e248.held_drive(base, np.array([0, 3], np.int64), 8.758351424802523)
    assert e248.same_bits(out[[1, 2]], base[[1, 2]])
    assert out[0] == np.float32(8.758351424802523)
    assert out[3] == out[0]
    assert e248.same_bits(base, np.array([7., -2., 5000., 0.], np.float32))


def test_response_and_interaction_are_separate_and_row_zero_excluded():
    rates = {name: _rate(0.) for name in e248.ARM_NAMES}
    rates["intact_high"][0, 2] = 100.
    assert not e248.response_metrics(rates, np.array([2]))["response_present"]
    rates["intact_high"][1:, 2] = .04  # 2e-4 normalized response.
    rates["bridge_cut_high"][1:, 2] = .04
    result = e248.response_metrics(rates, np.array([2]))
    assert result["response_present"] and not result["H248_supported"]
    rates["bridge_cut_high"][1:, 2] = .00
    result = e248.response_metrics(rates, np.array([2]))
    assert result["H248_supported"]
    assert result["response_R"] == pytest.approx(.0002)
    assert result["interaction_D"] == pytest.approx(.0002)


def test_sensory_cut_tests_all_non_sources_by_bits_including_signed_zero():
    low = _rate(0.)
    high = low.copy()
    high[:, 1] = 3.
    check = e248.cut_validation(low, high, np.array([1]))
    assert check["non_source_bitexact"]
    assert check["source_differing_cell_time_samples"] == 41
    high[9, 4] = np.float32(-0.)
    assert not e248.cut_validation(low, high, np.array([1]))["non_source_bitexact"]


def test_ledger_distinguishes_entered_from_completed_and_partial(tmp_path, monkeypatch):
    class Tensor:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Network:
        def __init__(self, saved):
            self.state = saved.copy()
            self.output_mask = Tensor(saved["output_mask"].copy())
            self.calls = 0

        def restore(self, state):
            self.state = state.copy()
            self.output_mask = Tensor(state["output_mask"].copy())

        def snapshot(self):
            return self.state.copy()

        def set_muted(self, ids):
            self.output_mask.values[:] = 1
            self.output_mask.values[np.asarray(ids, np.int64)] = 0

        def readout(self):
            return self.state["rate"].copy()

        def advance(self, drive, steps):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("generated mid-control fault")
            self.state["rate"] = self.state["rate"] + np.float32(1.)

    saved = {"rate": np.array([0., 2., 0.], np.float32),
             "adaptation": np.zeros(3, np.float32),
             "drive": np.zeros(3, np.float32),
             "output_mask": np.array([1., 0., 1.], np.float32),
             "tick": 7}
    ledger = e248.new_ledger()
    net = Network(saved)
    monkeypatch.setattr(e248.time, "perf_counter", lambda: 1.)
    with pytest.raises(RuntimeError, match="mid-control"):
        e248.run_arm(net, saved, "intact_low", tmp_path / "intact_low", ledger,
                     np.array([0], np.int64), np.array([2], np.int64), 1., 2., 0.)
    entry = ledger["arms"]["intact_low"]
    assert entry["requested"] and entry["entered"] and not entry["completed"]
    assert (entry["entered_controls"], entry["completed_controls"],
            entry["unknown_partial"]) == (2, 1, True)
    assert ledger["science_ticks_completed"] == 20
    stored = np.load(tmp_path / "intact_low" / "rate.npy")
    assert np.array_equal(stored[0], saved["rate"])
    assert np.array_equal(stored[1], saved["rate"] + 1.)
    assert np.isnan(stored[2:]).all()
    receipt = json.loads((tmp_path / "advance-ledger.json").read_text())
    assert receipt["arms"]["intact_low"]["unknown_partial"]


def test_gate_rejects_review_before_loading_any_arrays(tmp_path, monkeypatch):
    freeze = tmp_path / "freeze.json"
    review = tmp_path / "review.json"
    freeze.write_text(json.dumps({"schema": e248.FREEZE_SCHEMA}), encoding="utf-8")
    review.write_text(json.dumps({"schema": e248.REVIEW_SCHEMA,
                                  "approved": False, "blocking_defects": ["unreviewed"]}),
                      encoding="utf-8")
    monkeypatch.setattr(e248, "FROZEN_SHA256", e248.sha256(freeze))
    monkeypatch.setattr(e248, "_load_inputs", lambda _: pytest.fail("numeric input loaded"))
    with pytest.raises(ValueError, match="did not approve"):
        e248.gate(freeze, e248.sha256(freeze), review, e248.sha256(review))


def test_complete_fake_arm_restores_baseline_and_records_actual_drive(tmp_path,
                                                                     monkeypatch):
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class FakeNetwork:
        def __init__(self, saved):
            self.saved = saved
            self.state = {k: v.copy() if isinstance(v, np.ndarray) else v
                          for k, v in saved.items()}
            self.output_mask = FakeTensor(self.state["output_mask"])

        def restore(self, saved):
            self.state = {k: v.copy() if isinstance(v, np.ndarray) else v
                          for k, v in saved.items()}
            self.output_mask = FakeTensor(self.state["output_mask"])

        def snapshot(self):
            return {k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in self.state.items()}

        def set_muted(self, ids):
            self.output_mask.values[:] = 1
            self.output_mask.values[np.asarray(ids, np.int64)] = 0

        def readout(self):
            return self.state["rate"].copy()

        def advance(self, drive, steps):
            self.state["drive"][:] = drive
            self.state["tick"] += steps
            self.state["rate"] += np.float32(.125)

    saved = {"rate": np.array([0., 1., 2.], np.float32),
             "adaptation": np.zeros(3, np.float32),
             "drive": np.array([0., 5000., -1.], np.float32),
             "output_mask": np.array([1., 0., 1.], np.float32), "tick": 40000}
    net = FakeNetwork(saved)
    ledger = e248.new_ledger()
    monkeypatch.setattr(e248.time, "perf_counter", lambda: 1.)
    end = e248.run_arm(net, saved, "sensory_cut_high", tmp_path / "sensory_cut_high",
                       ledger, np.array([0], np.int64), np.array([2], np.int64),
                       8., 13., 0.)
    assert ledger["arms"]["sensory_cut_high"]["completed"]
    assert ledger["science_ticks_completed"] == 800
    assert end["tick"] == 40800
    assert e248.same_bits(end["drive"], np.array([13., 5000., -1.], np.float32))
    assert e248.same_bits(end["output_mask"], np.array([0., 0., 1.], np.float32))
    raw = np.load(tmp_path / "sensory_cut_high" / "rate.npy")
    assert e248.same_bits(raw[0], saved["rate"])
    assert np.all(raw[40] == saved["rate"] + 5.)
    assert e248.same_bits(saved["rate"], np.array([0., 1., 2.], np.float32))


@pytest.mark.parametrize("field,value", [
    ("device", "cpu"), ("cuda_implementation", "reference"),
    ("interval_steps", 10), ("graph", None), ("n", 3),
])
def test_actual_backend_rejects_wrong_runtime(field, value):
    class Backend:
        device = "cuda"
        cuda_implementation = "packed"
        interval_steps = 20
        graph = object()
        n = e248.N

    model = Backend()
    assert e248.actual_backend(model)["captured_graph"]
    setattr(model, field, value)
    with pytest.raises(ValueError, match="CUDA packed"):
        e248.actual_backend(model)


def test_direct_script_import_path_without_model_or_workspace_cwd(tmp_path):
    source = e248.ROOT / "tools" / "probe_c_banc_m3_lh_neural_response.py"
    code = ("import importlib.util,runpy,sys; "
            "runpy.run_path(sys.argv[1],run_name='generated_import_only'); "
            "assert importlib.util.find_spec('flylab.c.storage') is not None; "
            "assert importlib.util.find_spec('tools.verify_c_banc_observation') is not None")
    completed = subprocess.run([sys.executable, "-I", "-c", code, str(source)],
                               cwd=tmp_path, capture_output=True, text=True,
                               timeout=15, check=False)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("persistent", [False, True])
def test_late_result_failure_revokes_metrics_and_keeps_manifest_consistent(
        tmp_path, persistent):
    ledger = e248.new_ledger()
    result = {"validity": e248.VALIDITY, "H248_supported": True,
              "future_exact": True, "first_error": None,
              "finalization_pending": True}
    e248.write_json(tmp_path / "result.json", {"validity": "incomplete",
                                            "H248_supported": False,
                                            "finalization_pending": True})
    e248.write_json(tmp_path / "metrics.json", {"response_R": .002,
                                             "interaction_D": .001,
                                             "H248_supported": True})
    e248.write_json(tmp_path / "advance-ledger.json", ledger)
    real_write = e248.write_json
    requested = [0]

    def fail_result(path, value):
        if path.name == "result.json":
            requested[0] += 1
            if persistent or requested[0] == 1:
                raise OSError("generated late result failure")
        real_write(path, value)

    freeze = {"resource_limits": {"internal_s": 44, "output_bytes": 1 << 20}}
    first = e248.finish_receipts(tmp_path, freeze, ledger, result, [], None,
                                 time.perf_counter(), persist=fail_result)
    assert isinstance(first, OSError)
    stored_metrics = e248.read_json(tmp_path / "metrics.json")
    stored_result = e248.read_json(tmp_path / "result.json")
    assert stored_metrics["response_R"] == .002
    assert stored_metrics["H248_measured_condition_met"]
    assert stored_metrics["H248_supported"] is False
    assert stored_result["validity"] != e248.VALIDITY
    assert stored_result["H248_supported"] is False
    assert "late result failure" in e248.read_json(tmp_path / "errors.json")["errors"][0]["message"]
    manifest = e248.read_json(tmp_path / "output-manifest.json")
    for name, item in manifest["files"].items():
        assert e248.sha256(tmp_path / name) == item["sha256"]
    if persistent:
        assert stored_result["finalization_pending"]


def test_science_primary_survives_secondary_publication_error(tmp_path):
    ledger = e248.new_ledger()
    result = {"validity": "invalid_or_incomplete", "H248_supported": False,
              "future_exact": False, "first_error": {"type": "ValueError", "message": "science"},
              "finalization_pending": True}
    primary = ValueError("science")
    e248.write_json(tmp_path / "result.json", result)
    real_write = e248.write_json

    def fail_manifest(path, value):
        if path.name == "output-manifest.json":
            raise OSError("generated manifest failure")
        real_write(path, value)

    freeze = {"resource_limits": {"internal_s": 44, "output_bytes": 1 << 20}}
    returned = e248.finish_receipts(tmp_path, freeze, ledger, result, [], primary,
                                     time.perf_counter(), persist=fail_manifest)
    assert returned is primary
    assert e248.read_json(tmp_path / "result.json")["first_error"]["message"] == "science"
    assert not (tmp_path / "output-manifest.json").exists()
