"""Generated-only checks for the bounded E257 body comparison."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tools import analyze_c_banc_biological_gait as gait
from tools import verify_c_banc_m3_lh_partition_body as e257
from tools.banc_m3_monotone_recruitment import MonotoneRecruitmentTrial


def identity_snapshots():
    saved = {"model_hash": "old",
        "core_adaptation": {"base_model_hash": "old-core", "ids": np.array([1])},
        "motor_adaptation": {"source_core_identity": "old-motor",
                             "gain": np.array([.2])},
        "motor_timescale": {"source_motor_identity": "old-time",
                            "tau": np.array([.01])},
        "rate": np.array([.3], np.float32), "tick": 7}
    changed = copy.deepcopy(saved)
    changed["model_hash"] = "new"
    changed["core_adaptation"]["base_model_hash"] = "new-core"
    changed["motor_adaptation"]["source_core_identity"] = "new-motor"
    changed["motor_timescale"]["source_motor_identity"] = "new-time"
    return saved, changed


def test_exact_identity_transfer_and_dynamic_state():
    saved, constructed = identity_snapshots()
    baseline, fields = e257.transferred_network_state(saved, saved, False)
    assert baseline["model_hash"] == "old" and all(not value for value in fields.values())
    candidate, fields = e257.transferred_network_state(saved, constructed, True)
    assert candidate["model_hash"] == "new"
    assert candidate["tick"] == 7
    np.testing.assert_array_equal(candidate["rate"], saved["rate"])
    assert sum(len(value) for value in fields.values()) == 4
    bad = copy.deepcopy(constructed)
    bad["core_adaptation"]["ids"][0] = 2
    with pytest.raises((AssertionError, ValueError)):
        e257.transferred_network_state(saved, bad, True)


def test_six_existing_edges_only_leave_nineteen_and_csr_topology_exact():
    n = 111971
    indptr = np.zeros(n + 1, np.int32)
    indptr[1:] = 19
    indptr[111971:] = 25
    pre = np.arange(25, dtype=np.int32)
    original = np.full(25, .00003, np.float32)
    matrix = csr_matrix((original, pre, indptr), shape=(n, n))
    counts = np.array([1, 3, 1, 1, 3, 2], np.int64)
    target = (counts.astype(np.float64) * .03).astype(np.float32)
    selected = {"edge_ids": np.arange(19, 25),
                "pre_ids": pre[19:],
                "post_ids": np.full(6, 111970, np.int64),
                "source_values": (target.astype(np.float64) * .001).astype(np.float32),
                "target_values": target, "counts": counts}
    matrix.data[19:] = selected["source_values"]
    candidate = e257.e256.patch_six(matrix, selected, np.arange(19))
    assert e257.e250.prior.same_bits(candidate.indptr, matrix.indptr)
    assert e257.e250.prior.same_bits(candidate.indices, matrix.indices)
    assert e257.e250.prior.same_bits(candidate.data[:19], matrix.data[:19])
    assert e257.e250.prior.same_bits(candidate.data[19:], target)


def test_short_slip_missing_is_unavailable_and_forward_uses_fixed_heading():
    t = np.arange(1, 15001) * .0001
    native = {"time_s": t, "normal_load_bw": np.zeros((15000, 6)),
              "slip_mm_s": np.full((15000, 6), np.nan),
              "slip_valid": np.zeros((15000, 6), bool)}
    native["normal_load_bw"][4999:14999, 2] = .03
    native["slip_mm_s"][4999:14999, 2] = 2.
    native["slip_valid"][4999:14999, 2] = True
    result = e257.short_window_slip(native, complete=True)
    assert result["available"] and result["load_weighted_mm_s"] == 2.
    native["slip_valid"][7000, 2] = False
    result = e257.short_window_slip(native, complete=True)
    assert not result["available"] and result["missing_selected_slip"] == 1
    rows = [{"time_s": i*.005,
             "body": {"position": [i*.01, 0., 0.], "basis": np.eye(3).tolist()}}
            for i in range(301)]
    forward = e257.short_forward(rows, complete=True)
    assert forward["available"] and forward["window_samples"] == 200
    assert forward["fixed_initial_heading_forward_mm"] == pytest.approx(1.99)


def test_h257_is_conjunctive_and_short_only():
    arm = {"health": {"healthy": True},
           "startup": {"strict_E141_startup_screen": True},
           "forward": {"available": True, "fixed_initial_heading_forward_mm": 2.},
           "slip": {"available": True, "load_weighted_mm_s": 1.},
           "gait": {"window": {"longest_forward_bout_s": .31}}}
    baseline, candidate = copy.deepcopy(arm), copy.deepcopy(arm)
    assert e257.assess_h257(baseline, candidate)["H257_supported"]
    candidate["slip"]["available"] = False
    verdict = e257.assess_h257(baseline, candidate)
    assert not verdict["H257_supported"] and not verdict["F3_F5_10s_evaluated"]


def test_sensory_wrapper_retains_old_filter_drive_and_restores_instance():
    class Leaf:
        def __init__(self):
            self.filtered = np.zeros(241)
            self.velocity_lowpass = np.zeros(42)
            self.last_features = np.zeros(241)

        def encode(self):
            drive = np.zeros(10, np.float32)
            drive[2] = np.float32(self.filtered[85])
            self.last_features[85] = 3.
            self.filtered[85] = .75
            return drive

    leaf = Leaf()
    session = SimpleNamespace(adapter=SimpleNamespace(
        original=SimpleNamespace(original=leaf)), control_tick=0)
    records = []
    detach = e257.attach_sensory_observer(session, records, [2])
    assert leaf.encode()[2] == 0.
    assert records[0]["old_filtered"][85] == 0.
    assert records[0]["new_filtered"][85] == .75
    assert records[0]["returned_source_drive"][0] == 0.
    detach()
    assert "encode" not in leaf.__dict__
    assert leaf.encode()[2] == .75


def test_publication_late_failure_revokes_hypothesis(tmp_path):
    out = tmp_path
    result = {"validity": "valid_short_native_body_diagnostic",
              "H257": {"H257_supported": True}}
    ledger = {"status": "complete"}
    calls = 0

    def writer(path, payload):
        nonlocal calls
        calls += 1
        if path.name == "output-manifest.json":
            raise OSError("late manifest failed")
        path.write_text(json.dumps(payload))

    primary = e257.publish_final(out, ledger, result, [], None,
                                 e257.time.perf_counter(), 1024, writer)
    assert isinstance(primary, OSError)
    stored = json.loads((out / "result.json").read_text())
    assert stored["validity"] == "invalid_or_incomplete"
    assert stored["H257"]["H257_supported"] is False
    assert not (out / "output-manifest.json").exists()
    assert calls >= 4


def test_partial_clock_accounting_never_assumes_zero():
    class FakeSession:
        def __init__(self):
            self.control_tick = 0
            self.network = SimpleNamespace(tick=0)
            self.body = SimpleNamespace(physics_time=lambda: self.t)
            self.t = 0.

        def advance(self, _count):
            self.network.tick = 3
            self.t = .0002
            raise RuntimeError("partial physics")

    clock = e257.e113.AdvanceLedger()
    with pytest.raises(RuntimeError, match="partial physics"):
        clock.advance(FakeSession(), 1, "candidate")
    assert clock.events[0]["actual_neural_s"] == 3*.00025
    assert clock.events[0]["actual_physical_s"] == pytest.approx(.0002)
    assert clock.events[0]["advance_error"] == "RuntimeError"


def test_wrapper_takes_sole_body_ownership_without_changing_state():
    base = object.__new__(MonotoneRecruitmentTrial)
    base.closed = False
    base.body = object()
    base.network = object()
    wrapper = e257.LHPartitionTrial(base, {"variant": "baseline"})
    assert base.closed is True and wrapper.closed is False
    assert wrapper.body is base.body and wrapper.network is base.network


def test_core_activity_is_passive_and_keeps_partial_row_count(tmp_path):
    class Tensor:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, rows):
            return Tensor(self.values[rows])

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    counts = {"readout": 0, "advance": 0}

    def readout(rows):
        counts["readout"] += 1
        return np.arange(30, dtype=np.float32)[rows]

    def advance(*_args):
        counts["advance"] += 1

    net = SimpleNamespace(tick=400, readout=readout, advance=advance,
        drive=Tensor(np.arange(30, dtype=np.float32)),
        adaptation=Tensor(np.zeros(30, np.float32)))
    session = SimpleNamespace(network=net, control_tick=4,
                              body=SimpleNamespace(physics_time=lambda: .02))
    ids = np.arange(24, dtype=np.int64)
    row = e257.capture_core_activity(session, ids)
    e257.save_core_activity(tmp_path / "core-activity.npz", [row], ids)
    with np.load(tmp_path / "core-activity.npz", allow_pickle=False) as raw:
        assert raw["core_ids"].dtype == np.int64
        assert raw["rate"].shape == (1, 24)
        assert raw["rate"].dtype == raw["drive"].dtype == np.float32
        assert raw["control_tick"].tolist() == [4]
        assert raw["network_tick"].tolist() == [400]
    assert counts == {"readout": 1, "advance": 0}

    e257.save_core_activity(tmp_path / "empty-core.npz", [], ids)
    with np.load(tmp_path / "empty-core.npz", allow_pickle=False) as raw:
        assert raw["rate"].shape == raw["drive"].shape == (0, 24)
        assert raw["adaptation"].shape == (0, 24)
        assert raw["rate"].dtype == raw["drive"].dtype == np.float32
        assert raw["adaptation"].dtype == np.float32
        assert raw["control_tick"].shape == raw["network_tick"].shape == (0,)
        assert raw["control_tick"].dtype == raw["network_tick"].dtype == np.int64
        assert raw["physics_time_s"].dtype == np.float64

    net.readout = lambda rows: np.full(len(rows), np.nan, np.float32)
    with pytest.raises(ValueError, match="finite"):
        e257.capture_core_activity(session, ids)
    net.readout = readout
    session.body.physics_time = lambda: float("inf")
    with pytest.raises(ValueError, match="clock"):
        e257.capture_core_activity(session, ids)
    assert counts["advance"] == 0
    # An invalid next observation cannot erase a previously completed row.
    e257.save_core_activity(tmp_path / "partial-core.npz", [row], ids)
    with np.load(tmp_path / "partial-core.npz", allow_pickle=False) as raw:
        assert raw["rate"].shape == (1, 24)
        assert raw["control_tick"].tolist() == [4]


def test_partial_constructor_counts_actual_body_step_returns():
    class FakeBody:
        def __init__(self):
            self.d = SimpleNamespace(time=0.)

        def body_step(self):
            self.d.time += .0001

    body = FakeBody()

    class Fail(Exception):
        pass

    def constructor():
        body.body_step()
        body.body_step()
        raise Fail("after two actual steps")

    def fake_profile(_label, callback, _rows, _persist, *, on_return):
        value = callback()
        on_return(value)
        return value

    ledger = {"body_construction": [], "construction": []}
    with pytest.raises(Fail):
        e257.construction_with_body_count("generated", constructor, ledger,
            lambda: None, lambda _x: None, body_step_code=FakeBody.body_step.__code__,
            construction_profile=fake_profile)
    receipt = ledger["body_construction"][0]
    assert receipt["entered"] == receipt["returned"] == 2
    assert receipt["known_completed_settling_steps"] == 2
    assert receipt["trace_restored"]


def test_before_work_and_after_work_exception_receipts():
    class FakeBody:
        def __init__(self):
            self.d = SimpleNamespace(time=0.)

        def step(self, after):
            if after:
                self.d.time += .0001
            raise ValueError("fault")

    def profile(_label, callback, _rows, _persist, *, on_return):
        value = callback()
        on_return(value)
        return value

    for after, expected in ((False, 0), (True, 1)):
        body = FakeBody()
        ledger = {"body_construction": [], "construction": []}
        with pytest.raises(ValueError, match="fault"):
            e257.construction_with_body_count("fault", lambda b=body, a=after: b.step(a),
                ledger, lambda: None, lambda _x: None,
                body_step_code=FakeBody.step.__code__, construction_profile=profile)
        record = ledger["body_construction"][0]
        assert record["exception_unwinds"] == 1
        assert record["known_completed_settling_steps"] == expected
        assert record["unknown_step_entries"] == 0


def test_post_return_receipt_failure_releases_transferred_owner_once():
    owner = SimpleNamespace(closed=False, closes=0)

    def close():
        owner.closes += 1
        owner.closed = True

    owner.close = close
    holder = {}
    ledger = {"body_construction": [], "construction": []}
    calls = 0

    def persist():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("post-return receipt")

    def profile(_label, callback, _rows, _persist, *, on_return):
        value = callback()
        on_return(value)
        return value

    def noop():
        pass

    with pytest.raises(OSError, match="post-return receipt"):
        e257.construction_with_body_count("returned", lambda: owner, ledger,
            persist, lambda value: holder.setdefault("session", value),
            body_step_code=noop.__code__, construction_profile=profile)
    entry = {}
    recovered = e257.recover_transferred_owner(None, holder, entry)
    recovered.close()
    assert owner.closes == 1 and entry["post_return_owner_recovered"]


def test_only_measured_fault_is_valid_refutation():
    healthy = {"healthy": True}
    invalid_clock = {"healthy": False, "clock_exact": False,
                     "observed_clock_valid": False}
    valid_partial = {"healthy": False, "clock_exact": False,
                     "observed_clock_valid": True}
    assert e257.classify_short_outcome(None, None, healthy) == (
        "complete_healthy_short_diagnostic")
    assert e257.classify_short_outcome(None, None, invalid_clock) == (
        "invalid_or_incomplete")
    assert e257.classify_short_outcome(None, {"control": 3}, invalid_clock) == (
        "invalid_or_incomplete")
    assert e257.classify_short_outcome(None, {"control": 3}, valid_partial) == (
        "measured_native_health_stop")
    assert e257.classify_short_outcome(ValueError("clock"), {"control": 3}, healthy) == (
        "invalid_or_incomplete")


def test_saved_trace_passes_existing_gait_loader_and_threepart_analyzer(tmp_path):
    rows = []
    links = [[0., 0., 0.], [.2, 0., 0.], [.3, 0., .1],
             [.4, 0., .2], [.5, 0., .3]]
    for i in range(301):
        rows.append({"control_tick": i, "time_s": i*.005,
            "body": {"position": [i*.01, 0., 0.],
                     "basis": np.eye(3).tolist(), "yaw": 0.,
                     "legs": {leg: links for leg in e257.LEGS}},
            "feet": {"load_bearing": [False]*6}, "upright": 1., "fault": False})
    selection = {"source_rows": [], "bridge_rows": [],
                 "reachable_motor_rows": [], "observed_ids": []}
    failures = e257.save_arm_raw(tmp_path, rows, [], [], [], [], [],
        e257.e113.AdvanceLedger(), selection, "graph", "model", "body",
        "adapter", {"neural_dt_s": .00025})
    assert not failures
    loaded = gait.load_model(tmp_path / "trace.json")
    report, phase, events = gait.analyze(loaded)
    assert isinstance(report, dict) and isinstance(phase, np.ndarray)
    assert isinstance(events, list)
    assert "longest_forward_bout_s" in report["window"]


@pytest.mark.parametrize("metadata_fails", [False, True])
def test_arm_cleanup_uses_real_parameter_api_and_preserves_primary(
        tmp_path, monkeypatch, metadata_fails):
    def state_dict():
        if metadata_fails:
            raise KeyError("metadata failure")
        return {"neural_dt_s": .00025}

    # The actual session has p.state_dict(), not a parameters attribute.
    owner = SimpleNamespace(network=SimpleNamespace(n=0, identity="neural"),
        body=SimpleNamespace(model_hash="body"),
        adapter=SimpleNamespace(identity="adapter"),
        p=SimpleNamespace(state_dict=state_dict), closed=False, closes=0)

    def close():
        owner.closed = True
        owner.closes += 1

    owner.close = close
    owner.snapshot = lambda: {"schema": "generated-no-model"}

    def construct(_label, _callback, _ledger, _persist, on_return):
        on_return(owner)
        return owner

    monkeypatch.setattr(e257, "construction_with_body_count", construct)
    monkeypatch.setattr(e257.protocol.StateStore, "save", lambda *_args: None)
    freeze = {"selection": {"source_rows": [], "bridge_rows": [],
                            "reachable_motor_rows": []},
              "core_observer_rows": list(range(24)),
              "resource_limits": {"host_peak_bytes": 1024},
              "support_load_min_bw": .02}
    ledger = {"arms": {}}
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1))
    metrics, primary = e257.run_arm(SimpleNamespace(hash="graph"), {}, None, None,
        "baseline", freeze, tmp_path, ledger, e257.time.perf_counter(), None, process)
    assert isinstance(primary, ValueError)
    assert "CUDA/body neutral" in str(primary)
    assert metrics["status"] == "invalid_or_incomplete"
    assert owner.closed and owner.closes == 1
    recorded = json.loads((tmp_path / "baseline/trace.json").read_text())
    if not metadata_fails:
        assert recorded["parameters"] == {"neural_dt_s": .00025}
    else:
        assert any("metadata failure" in note for note in primary.__notes__)
