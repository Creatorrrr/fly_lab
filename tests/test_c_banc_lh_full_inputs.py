"""Small generated checks for E254; no original BANC/GPU/body work."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tools import replay_c_banc_m3_lh_full_inputs as e254


def test_drive_scatter_duplicate_overlap_and_float32_order():
    old = np.zeros(241, np.float64)
    old[:4] = [.1, .2, -0., 1.]
    ids = np.array([1, 1, 2, 20], np.int64)
    ports = np.arange(4, dtype=np.int64)
    nav = np.arange(20, dtype=np.int64)
    desc = np.array([20, 21], np.int64)
    sensory, actual = e254.reconstruct_drive(old, ids, ports, 32, desc,
                                              0., nav, 5000.)
    scalar = np.zeros(32, np.float32)
    for neuron, port in zip(ids, ports):
        scalar[neuron] += old[port]
    assert e254.same_bits(sensory, scalar)
    scalar[desc] += 0.
    tonic = np.zeros(32, np.float32)
    tonic[nav] = -5000.
    tonic[desc] = 5000.
    scalar += tonic
    assert e254.same_bits(actual, scalar)
    assert actual[1] == np.float32(sensory[1] - 5000.)
    assert actual[20] == np.float32(sensory[20] + 5000.)
    assert np.signbit(sensory[2]) is False or sensory[2] == 0.


def test_incoming_union_order_mask_and_signed_edge_math():
    n = 420
    matrix = csr_matrix((np.array([2., -3., 4.], np.float32),
                         (np.array([0, 1, 4]), np.array([419, 1, 418]))),
                        shape=(n, n), dtype=np.float32)
    mask = np.ones(n, np.float32)
    mask[1] = 0.
    rows = e254.incoming_rows(matrix, np.arange(5, dtype=np.int64),
                               np.arange(418, dtype=np.int64), mask)
    assert rows["union_ids"].tolist() == [*range(418), 418, 419]
    assert rows["target_offsets"].tolist() == [0, 1, 2, 2, 2, 3]
    assert rows["observed_edge_mask"].tolist() == [False, True, False]
    rates = np.zeros(420, np.float32)
    rates[418:420] = [5., 7.]
    rates[1] = 8.
    current = e254.edge_currents(rates, rows["union_ids"], rows["pre_ids"],
                                  rows["weights32"], mask)
    assert current.dtype == np.float64
    assert current.tolist() == [14., 0., 20.]
    with pytest.raises(ValueError):
        e254.edge_currents(rates[:-1], rows["union_ids"], rows["pre_ids"],
                           rows["weights32"], mask)


def test_endpoint_comparison_rejects_stale_or_misaligned_reference(tmp_path):
    raw = e254.memmaps(tmp_path, 3, 1, 2)
    raw["endpoint-rate.npy"][0] = [1., 2., 3.]
    raw["endpoint-drive.npy"][0] = [4., 5., 6.]
    raw["endpoint-adaptation.npy"][0] = [7., 8., 9.]
    arm = {"activity": {"observed_ids": np.array([0, 2], np.int64),
                         "rate": np.array([[1., 3.]], np.float32),
                         "drive": np.array([[4., 6.]], np.float32),
                         "adaptation": np.array([[7., 9.]], np.float32)},
           "controls": {"motor_rate32": np.array([[2.]], np.float32)}}
    e254.check_endpoint(0, raw, arm, np.arange(3), np.array([1]))
    arm["activity"]["rate"][0, 1] = np.float32(3.01)
    with pytest.raises(AssertionError, match="endpoint differs"):
        e254.check_endpoint(0, raw, arm, np.arange(3), np.array([1]))
    arm["activity"]["rate"][0, 1] = 3.
    arm["activity"]["observed_ids"][1] = 5
    with pytest.raises(AssertionError, match="endpoint differs"):
        e254.check_endpoint(0, raw, arm, np.arange(3), np.array([1]))


def test_current_probe_preserves_state_and_partial_readback(tmp_path):
    import torch

    class FakeNetwork:
        device = "cpu"

        def __init__(self):
            self.rate = torch.arange(6, dtype=torch.float32)
            self.drive = self.rate + 10
            self.adaptation = self.rate + 20
            self.fail_current = False

        def readout(self, ids=None):
            return self.rate[ids].detach().numpy().copy()

        def _current(self, rate):
            if self.fail_current:
                raise RuntimeError("probe failed")
            return rate * 2

        def snapshot(self):
            return {"rate": self.readout(), "drive": self.drive.numpy().copy(),
                    "adaptation": self.adaptation.numpy().copy()}

    network = FakeNetwork()
    raw = e254.memmaps(tmp_path, 6, 2, 1)
    incoming = {"target_ids": np.arange(5, dtype=np.int64),
                "pre_ids": np.array([0, 5], np.int64),
                "weights32": np.array([2., -1.], np.float32),
                "output_mask_start": np.ones(6, np.float32)}
    account = {"current_probe_requested": 0, "current_probe_entered": 0,
               "current_probe_completed": 0, "observed_endpoint_count": 0}
    before = network.snapshot()
    e254.observe_network(network, 0, np.arange(6), incoming, raw, account)
    assert e254.same_bits(network.snapshot()["rate"], before["rate"])
    assert raw["actual-current.npy"][0].tolist() == [0., 2., 4., 6., 8.]
    assert raw["edge-current64.npy"][0].tolist() == [0., -5.]
    assert account["current_probe_completed"] == account["observed_endpoint_count"] == 1
    network.fail_current = True
    with pytest.raises(RuntimeError, match="probe failed"):
        e254.observe_network(network, 1, np.arange(6), incoming, raw, account)
    assert raw["endpoint-rate.npy"][1].tolist() == list(range(6))
    assert np.isnan(raw["actual-current.npy"][1]).all()
    assert account["current_probe_entered"] == 2
    assert account["current_probe_completed"] == 1


def _fake_science(monkeypatch, tmp_path, *, fail_stage):
    monkeypatch.setattr(e254, "N", 32)
    monkeypatch.setattr(e254, "CONTROLS", 2)
    old = np.zeros((2, 241), np.float64)
    old[:, 0] = 1.
    source = np.array([22], np.int64)
    arm = {"sensory_ids": source, "sensory_ports": np.array([0], np.int64),
           "sensory": {"old_filtered": old, "source_ids": source,
                       "returned_source_drive": np.ones((2, 1), np.float32)}}
    raw = e254.memmaps(tmp_path, 1, 1, 22)
    entry = e254.begin_arm({"arms": {}}, "baseline")
    calls = []

    class FakeNetwork:
        completed = 0

        def advance(self, drive, steps):
            calls.append(("advance", self.completed + 1, steps))
            if fail_stage == "step" and self.completed == 0:
                raise RuntimeError("step failed")
            self.completed += 1

    def observe(network, index, union, incoming, sink, account):
        calls.append(("observe", index))
        if fail_stage == "observe" and index == 1:
            raise RuntimeError("readback failed")
        sink["endpoint-rate.npy"][index] = index
        sink["endpoint-rate.npy"].flush()
        account["observed_endpoint_count"] = index + 1

    try:
        e254.science_loop(FakeNetwork(), arm, raw, {"union_ids": source},
                          np.arange(22, dtype=np.int64), np.arange(20, dtype=np.int64),
                          np.array([20, 21], np.int64), source, entry,
                          lambda: None, lambda: None, observe=observe,
                          check=lambda *_args: None)
    except RuntimeError as error:
        assert str(error) == ("step failed" if fail_stage == "step"
                              else "readback failed")
    else:
        pytest.fail("injected failure must propagate")
    e254.flush_raw(raw)
    return entry, calls, raw


def test_step_failure_is_unknown_partial_and_raw_row_unwritten(monkeypatch, tmp_path):
    entry, calls, raw = _fake_science(monkeypatch, tmp_path, fail_stage="step")
    assert calls == [("observe", 0), ("advance", 1, 20)]
    assert entry["entered_controls"] == 1
    assert entry["completed_neural_controls"] == 0
    assert entry["unknown_neural_partial"] is True
    assert entry["observed_endpoint_count"] == 1
    assert np.isnan(raw["endpoint-rate.npy"][1:]).all()


def test_poststep_observation_failure_preserves_known_work(monkeypatch, tmp_path):
    entry, calls, raw = _fake_science(monkeypatch, tmp_path, fail_stage="observe")
    assert calls == [("observe", 0), ("advance", 1, 20), ("observe", 1)]
    assert entry["entered_controls"] == entry["completed_neural_controls"] == 1
    assert entry["unknown_neural_partial"] is False
    assert entry["observed_endpoint_count"] == 1
    assert np.isnan(raw["endpoint-rate.npy"][1:]).all()


def _receipts(tmp_path: Path):
    (tmp_path / "data.npy").write_bytes(b"raw")
    freeze = {"resource_limits": {"internal_s": 55, "output_bytes": 1_000_000}}
    ledger = {"arms": {"baseline": {"constructor_completed": True,
               "setup_executed_ticks": 60, "setup_capture_ticks": 20,
               "completed_neural_controls": 300, "current_probe_completed": 301,
               "unknown_neural_partial": False, "setup_unknown_partial": False}}}
    result = {"validity": e254.VALIDITY, "H254_supported": True,
              "finalization_pending": True}
    return freeze, ledger, result


def test_late_result_publication_failure_revokes_success(tmp_path):
    freeze, ledger, result = _receipts(tmp_path)
    def persist(path, value):
        if path.name == "result.json":
            raise OSError("result unavailable")
        e254.e248.write_json(path, value)
    error = e254.finish_receipts(tmp_path, freeze, ledger, result, [], None,
                                 e254.time.perf_counter(), persist=persist)
    assert isinstance(error, OSError)
    assert result["H254_supported"] is False
    assert result["validity"] == "invalid_or_incomplete"
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "output-manifest.json").exists()
    assert json.loads((tmp_path / "errors.json").read_text())["errors"]


def test_persistent_late_receipt_failure_never_leaves_supported_result(tmp_path):
    freeze, ledger, result = _receipts(tmp_path)
    first = RuntimeError("scientific readback failed")
    def persist(path, value):
        if path.name in {"errors.json", "result.json", "output-manifest.json"}:
            raise OSError("publication unavailable")
        e254.e248.write_json(path, value)
    error = e254.finish_receipts(tmp_path, freeze, ledger, result, [], first,
                                 e254.time.perf_counter(), persist=persist)
    assert error is first
    assert result["H254_supported"] is False
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "output-manifest.json").exists()


def test_prior_science_error_invalidates_even_if_publication_succeeds(tmp_path):
    freeze, ledger, result = _receipts(tmp_path)
    first = RuntimeError("original advance failure")
    error = e254.finish_receipts(tmp_path, freeze, ledger, result, [], first,
                                 e254.time.perf_counter())
    assert error is first
    final = json.loads((tmp_path / "result.json").read_text())
    assert final["validity"] == "invalid_or_incomplete"
    assert final["H254_supported"] is False
    assert final["first_error"]["message"] == "original advance failure"


def test_frozen_startup_hypothesis_and_window_edge_rank():
    edges = 6
    target = e254.TARGETS
    start = {"edge-current64.npy": np.zeros((301, edges), np.float64),
             "actual-current.npy": np.zeros((301, 5), np.float32),
             "endpoint-drive.npy": np.zeros((301, 5), np.float32),
             "endpoint-adaptation.npy": np.zeros((301, 5), np.float32)}
    other = {key: value.copy() for key, value in start.items()}
    other["edge-current64.npy"][1:100, :2] = [-1., 2.]
    incoming = {"target_ids": target, "target_offsets": np.array([0, 2, 3, 4, 5, 6]),
                "observed_edge_mask": np.array([True, False, True, True, True, True]),
                "union_ids": target.copy(), "edge_ids": np.arange(edges),
                "pre_ids": np.arange(edges)}
    table = e254.summarize(start, other, incoming,
                           {"case_controls": [64, 84], "hypothesis_motor_id": 6373},
                           {i: f"flywire:banc:888:{i:03}" for i in range(edges)})
    first = table["windows"]["startup"]["motors"]["6373"]
    assert first["observed_delta_sum64"] == -99.
    assert first["unobserved_delta_sum64"] == 198.
    assert table["H254_supported"] is True
    ranked = table["windows"]["startup"]["top_edges"]["6373"]
    assert ranked["positive"][0]["edge_id"] == 1
    assert ranked["negative"][0]["edge_id"] == 0


def test_temporal_cancellation_preserves_raw_positive_negative_parts():
    target = e254.TARGETS
    baseline = {"edge-current64.npy": np.zeros((301, 6), np.float64),
                "actual-current.npy": np.zeros((301, 5), np.float32),
                "endpoint-drive.npy": np.zeros((301, 5), np.float32),
                "endpoint-adaptation.npy": np.zeros((301, 5), np.float32)}
    candidate = {key: value.copy() for key, value in baseline.items()}
    candidate["edge-current64.npy"][1, :2] = [3., -3.]
    candidate["edge-current64.npy"][2, :2] = [-2., 5.]
    candidate["endpoint-drive.npy"][1:3, 0] = [2., -3.]
    candidate["endpoint-adaptation.npy"][1:3, 0] = [1., -4.]
    incoming = {"target_ids": target, "target_offsets": np.array([0, 2, 3, 4, 5, 6]),
                "observed_edge_mask": np.array([True, False, True, True, True, True]),
                "union_ids": target.copy(), "edge_ids": np.arange(6),
                "pre_ids": np.arange(6)}
    table = e254.summarize(baseline, candidate, incoming,
                           {"case_controls": [64, 84], "hypothesis_motor_id": 6373},
                           {i: f"flywire:banc:888:{i:03}" for i in range(6)})
    row = table["windows"]["startup"]["motors"]["6373"]
    assert (row["total_edge_delta_sum64"], row["positive_edge_delta_sum64"],
            row["negative_edge_delta_sum64"]) == (3., 8., -5.)
    assert (row["observed_delta_sum64"], row["observed_positive_delta_sum64"],
            row["observed_negative_delta_sum64"]) == (1., 3., -2.)
    assert (row["unobserved_delta_sum64"], row["unobserved_positive_delta_sum64"],
            row["unobserved_negative_delta_sum64"]) == (2., 5., -3.)
    assert (row["drive_delta_sum32"], row["drive_positive_delta_sum32"],
            row["drive_negative_delta_sum32"]) == (-1., 2., -3.)
    assert (row["negative_adaptation_delta_sum32"],
            row["negative_adaptation_positive_delta_sum32"],
            row["negative_adaptation_negative_delta_sum32"]) == (3., 4., -1.)
    assert (row["margin_delta_sum64"], row["margin_positive_delta_sum64"],
            row["margin_negative_delta_sum64"]) == (5., 5., 0.)
    top = table["windows"]["startup"]["top_edges"]["6373"]["positive"]
    assert [(item["signed_delta_sum64"], item["positive_delta_sum64"],
             item["negative_delta_sum64"]) for item in top] == [
                 (2., 5., -3.), (1., 3., -2.)]


def test_full_final_state_saved_before_mismatch_and_partial_keeps_first(tmp_path):
    events = []
    class FakeNetwork:
        def snapshot(self):
            events.append("snapshot")
            return {"tick": 6, "rate": np.array([1.], np.float32)}

    def save(path, state):
        events.append("save")
        path.mkdir()
        np.save(path / "rate.npy", state["rate"], allow_pickle=False)

    def exact(_actual, _expected, _label):
        events.append("exact")
        raise AssertionError("whole final mismatch")

    final = tmp_path / "final"
    with pytest.raises(AssertionError, match="whole final mismatch"):
        e254.save_final_before_parity(FakeNetwork(), final,
                                       {"tick": 6}, save=save, exact=exact)
    assert events == ["snapshot", "save", "exact"]
    assert np.load(final / "rate.npy", allow_pickle=False).tolist() == [1.]
    assert not (tmp_path / "partial-final").exists()
    primary = RuntimeError("first scientific failure")
    account = {}
    returned = e254.preserve_partial_state(FakeNetwork(), tmp_path / "partial-final",
                                            account, primary, save=save)
    assert returned is primary
    assert account["partial_full_state_available"] is True
    assert (tmp_path / "partial-final" / "rate.npy").exists()
    secondary_primary = RuntimeError("original parity failure")
    failed_account = {}
    def fail_save(_path, _state):
        raise OSError("disk lost")
    assert e254.preserve_partial_state(FakeNetwork(), tmp_path / "cannot-save",
                                       failed_account, secondary_primary,
                                       save=fail_save) is secondary_primary
    assert failed_account["partial_full_state_available"] is False
    assert any("disk lost" in note for note in secondary_primary.__notes__)


def test_gate_rejects_wrong_freeze_before_any_original_loading(tmp_path):
    with pytest.raises(FileNotFoundError):
        e254.gate(tmp_path / "missing-freeze.json", e254.FREEZE_SHA,
                  tmp_path / "review.json", "0" * 64, tmp_path / "out")
