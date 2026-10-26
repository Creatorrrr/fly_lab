"""Synthetic, native-free E189 contract tests."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from tools import probe_c_banc_m3_single_mn_edge as e189


def test_only_two_fixed_float32_weights_and_one_edge() -> None:
    for arm, multiplier in (("low", 0.98), ("high", 1.02)):
        expected = np.float32(np.float64(e189.SOURCE) * np.float64(multiplier))
        assert np.float32(e189.ARM_WEIGHTS[arm]).view(np.uint32) == expected.view(np.uint32)
        patch = e189.edge_patch(arm)
        assert patch["edge_ids"].tolist() == [10511590]
        assert patch["pre_ids"].tolist() == [54014]
        assert patch["post_ids"].tolist() == [144018]
        assert patch["source_values"].view(np.uint32).tolist() == [
            e189.SOURCE.view(np.uint32)]
        assert patch["target_values"].view(np.uint32).tolist() == [
            expected.view(np.uint32)]
    with pytest.raises(ValueError, match="arm"):
        e189.edge_patch("other")


def test_exact_state_transfer_excludes_only_runtime_and_declared_identity() -> None:
    source = {
        "schema": "old", "wall_s": 0.1,
        "network": {**{key: "old" for key in e189.NETWORK_IDENTITY_KEYS},
                    "rate": np.array([1, 2], np.float32),
                    "drive": np.array([3, 4], np.float32)},
        "body": {"time": 0.0},
    }
    candidate = copy.deepcopy(source)
    candidate.update(schema="new", wall_s=0.2, load_feedback_patch={"edge": 10511590})
    for key in e189.NETWORK_IDENTITY_KEYS:
        candidate["network"][key] = "new"
    e189.exact_state_transfer(source, candidate)
    candidate["network"]["rate"][0] = np.nextafter(np.float32(1), np.float32(2))
    with pytest.raises(ValueError, match="rate"):
        e189.exact_state_transfer(source, candidate)


def test_readiness_rechecks_raw_queue_and_stopped_service() -> None:
    gpu = ["2026/09/24 12:00:00.000, 0, 1, 100, 16384, 10, 100, 100"]
    parsed = e189.resource_readiness._gpu_telemetry({"gpu_csv": gpu})
    queue = {"phase": "readiness", "ready": True, "resource_clear": True,
             "resource_mode": "queue_empty", "queue": {"running": 0, "pending": 0},
             "clients": {key: "absent" for key in e189.EXPECTED_E182_CLIENTS},
             "gpu_csv": gpu, "gpu_telemetry": parsed}
    assert e189.readiness_clear(queue)
    forged = {**queue, "queue": {"running": 1, "pending": 0}}
    assert not e189.readiness_clear(forged)
    stopped = {**queue, "queue": None, "queue_error": "refused",
               "resource_mode": "service_stopped", "service_probe": {
                   "current_pid": 111, "tcp_connect": "refused",
                   "listener_inventory_complete": True,
                   "listener_8188": False, "process_inventory_complete": True,
                   "listeners_8188": [], "foreign_python": [],
                   "allowed_process_family": [{"pid": 111, "ppid": 0,
                                               "create_time": 1.0,
                                               "relation": "self"}],
                   "python_processes": [{"pid": 111, "ppid": 0,
                                         "create_time": 1.0, "name": "python.exe"}],
                   "ancestor_terminal": {"reason": "root", "pid": 0}}}
    assert e189.readiness_clear(stopped, 111)
    sibling = copy.deepcopy(stopped)
    sibling["service_probe"]["allowed_process_family"].append(
        {"pid": 222, "ppid": 0, "create_time": 2.0, "relation": "descendant"})
    assert not e189.readiness_clear(sibling, 111)
    forged_terminal = copy.deepcopy(stopped)
    forged_terminal["service_probe"]["ancestor_terminal"]["pid"] = 999
    assert not e189.readiness_clear(forged_terminal, 111)
    stopped["service_probe"]["foreign_python"] = [{"pid": 999}]
    assert not e189.readiness_clear(stopped, 111)
    assert not e189.readiness_clear({**queue, "clients": {"1": "absent"}})


def test_fixed_windows_use_native_clock_and_e186_owner_mapping() -> None:
    times = np.arange(1, 30001, dtype=float) * 0.0001
    forward = np.zeros((30000, 6), float)
    forward[:, 4] = -2.0
    times[4999] = np.nextafter(0.5, 0.0)
    times[-1] = 3.000000000002334
    force = {"time_s": times, "leg_heading_xy_force_bw": forward,
             "finite": np.ones(30000, bool), "available": np.ones(30000, bool),
             "error": np.full(30000, "", dtype="<U1")}
    native = {"time_s": times.copy(),
              "normal_load_bw": np.ones((30000, 6), float),
              "slip_mm_s": np.full((30000, 6), 3.0),
              "slip_valid": np.ones((30000, 6), bool),
              "nonfoot_contact": np.zeros(30000, bool),
              "upright": np.ones(30000, float)}
    value = e189.fixed_force_summary(force, native)
    assert value["main"]["samples"] == 24999
    assert value["main"]["signed_ap_impulse_bw_s"][4] == pytest.approx(-4.9998)
    assert value["main"]["negative_ap_magnitude_impulse_bw_s"][4] == pytest.approx(4.9998)
    assert value["main"]["slip_weighted_mm_s"] == pytest.approx(3.0)
    assert value["e186_post10_control_owner_mapping"]["521"]["native_indices"] == [26000, 26500]
    assert value["e186_post10_control_owner_mapping"]["521"]["control_rows"] == [521, 530]
    native["slip_valid"][10, 4] = False
    with pytest.raises(ValueError, match="every supported"):
        e189.fixed_force_summary(force, native)
    native["slip_valid"][10, 4] = True
    force["available"][10] = False
    with pytest.raises(ValueError, match="complete finite"):
        e189.fixed_force_summary(force, native)


def test_saved_e185_clock_uses_literal_boundary() -> None:
    base = Path(e189.ROOT) / e189.BASE
    with np.load(base / "e185-ground-reaction/ground-reaction.npz",
                 allow_pickle=False) as force_file, np.load(
                     base / "e185-ground-reaction/baseline/native-contact.npz",
                     allow_pickle=False) as native_file:
        force = {key: force_file[key] for key in force_file.files}
        native = {key: native_file[key] for key in native_file.files}
    result = e189.fixed_force_summary(force, native)
    assert result["main"]["samples"] == 24999
    assert result["e186_control_owner_rows_100_599"]["samples"] == 25000
    assert result["e186_post10_control_owner_mapping"]["521"]["native_indices"] == [26000, 26500]


def test_partial_selected_signal_is_typed() -> None:
    empty = e189.selected_arrays([])
    assert empty["control_tick"].shape == (0,)
    assert empty["control_tick"].dtype == np.int64
    assert empty["weight"].shape == (0,)
    assert empty["mn_current"].shape == (0,)


def test_csr_and_packed_proof_rejects_other_edge_stale_table_and_wrong_value(
    monkeypatch,
) -> None:
    from tools import probe_c_banc_m3_core_rhythm as rhythm

    n = 158706
    indptr = np.full(n + 1, 2, np.int32)
    indptr[:2] = [0, 1]
    indptr[2] = 1
    indices = np.array([0, 0], np.int32)
    original = sparse.csr_matrix(
        (np.array([0.1, e189.SOURCE], np.float32), indices, indptr),
        shape=(n, n))
    candidate = original.copy()
    candidate.data[1] = np.float32(e189.ARM_WEIGHTS["low"])

    class Tensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Network:
        device = "cuda"
        cuda_implementation = "packed"
        n = 158706

        def __init__(self, matrix):
            self.matrix = matrix
            self._rowptr = Tensor(indptr.copy())
            self._columns = Tensor(np.array([0, 1 << 18], np.int32))
            self._weight_values = Tensor(
                np.array([0.1, e189.ARM_WEIGHTS["low"]], np.float32))

    a, b = Network(original), Network(candidate)
    monkeypatch.setattr(rhythm, "source_matrix", lambda network: network.matrix)
    monkeypatch.setattr(e189, "EDGE", 1)
    monkeypatch.setattr(e189, "POST", 2)
    monkeypatch.setattr(e189, "PRE", 0)
    assert e189.exact_csr_and_packed(a, b, "low")["changed_words"] == 1
    b.matrix.data[0] = np.float32(0.2)
    with pytest.raises(ValueError, match="selected float32 word only"):
        e189.exact_csr_and_packed(a, b, "low")
    b.matrix.data[0] = a.matrix.data[0]
    b._weight_values = Tensor(np.array([0.1, e189.SOURCE], np.float32))
    with pytest.raises(ValueError, match="packed decoded"):
        e189.exact_csr_and_packed(a, b, "low")
    b._weight_values = Tensor(np.array([0.1, e189.ARM_WEIGHTS["low"]], np.float32))
    b.matrix.data[1] = np.float32(e189.ARM_WEIGHTS["high"])
    with pytest.raises(ValueError, match="selected float32 word only"):
        e189.exact_csr_and_packed(a, b, "low")


@pytest.mark.parametrize("fail_result_write", [False, True])
def test_post_callback_failure_closes_owner_and_keeps_partial_raw(
    tmp_path, monkeypatch, fail_result_write,
) -> None:
    from flylab.c.storage import StateStore
    from tools import optimize_c_banc_m3_native_timing as e181
    from tools import verify_c_banc_m3_native_timescale_transfer as protocol

    monkeypatch.setattr(e189, "ROOT", tmp_path)
    monkeypatch.setattr(e181, "_neutral", lambda _saved: None)
    monkeypatch.setattr(StateStore, "save", lambda *_args: None)
    output = tmp_path / e189.OUTPUT_ROOT / "low" / "body"
    output.parent.mkdir(parents=True)
    metadata = {"expected_graph_hash": "graph", "expected_network_identity": "original",
                "expected_adapter_identity": "adapter", "expected_body_hash": "body"}
    approved = e189.E189Gate("low", np.float32(e189.ARM_WEIGHTS["low"]),
                              metadata, {}, "a" * 64, "b" * 64, "c" * 64,
                              [{}, {}])
    class Owner:
        def __init__(self):
            self.closed = 0
            self.network = type("Network", (), {"tick": 0, "identity": "changed"})()
            self.body = type("Body", (), {"model_hash": "body"})()
            self.adapter = type("Adapter", (), {"identity": "adapter"})()

        def snapshot(self):
            return {}

        def close(self):
            self.closed += 1
            # Real BancWalkingSession.close releases this reference.
            self.network = None

    owner = Owner()
    monkeypatch.setattr(e189, "construct_arm", lambda *_args: (owner, {"changed_words": 1}))

    def after_callback(_label, callback, _rows, _persist, *, on_return):
        pair = callback()
        on_return(pair)
        raise RuntimeError("synthetic CUDA sync failure")

    monkeypatch.setattr(protocol, "profiled_construction", after_callback)
    if fail_result_write:
        original_write = protocol.write_json

        def write_except_result(path, value):
            if Path(path).name == "result.json":
                raise OSError("synthetic result disk failure")
            return original_write(path, value)

        monkeypatch.setattr(protocol, "write_json", write_except_result)
    graph = type("Graph", (), {"hash": "graph"})()
    neutral = {"network": {"model_hash": "original"}, "parameters": {},
               "monotone_recruitment_patch": {"adapter_identity": "adapter"}}
    if fail_result_write:
        with pytest.raises(RuntimeError, match="synthetic CUDA sync failure"):
            e189.run_arm(graph, neutral, output, arm="low", startup={},
                         approved=approved)
        assert not (output / "result.json").exists()
    else:
        result = e189.run_arm(graph, neutral, output, arm="low", startup={},
                               approved=approved)
        assert result["validity"] == "incomplete"
        assert result["primary_error"] == "RuntimeError('synthetic CUDA sync failure')"
    assert owner.closed == 1
    with np.load(output / "selected-signal.npz", allow_pickle=False) as raw:
        assert raw["control_tick"].shape == (0,)
    assert (output / "primary-error.json").is_file()


def test_child_rejects_missing_parent_receipt_before_work(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires live parent"):
        e189.execute(tmp_path / "freeze", tmp_path / "review", "low",
                     tmp_path / "output", "")


def test_constructor_cleanup_keeps_primary_error(monkeypatch) -> None:
    from tools import banc_m3_monotone_recruitment as monotone
    from tools import optimize_c_banc_m3_native_timing as e181

    class Owner:
        def snapshot(self):
            return {}

        def close(self):
            raise RuntimeError("cleanup secondary")

    monkeypatch.setattr(e181, "_neutral", lambda _saved: None)
    monkeypatch.setattr(monotone.MonotoneRecruitmentTrial, "from_checkpoint",
                        lambda *_args: Owner())

    def failed_restore(*_args):
        raise ValueError("primary state mismatch")

    monkeypatch.setattr(e181.protocol, "exact", failed_restore)
    with pytest.raises(ValueError, match="primary state mismatch") as error:
        e189.construct_arm(None, {}, "low")
    assert "cleanup secondary" in error.value.__notes__[0]
