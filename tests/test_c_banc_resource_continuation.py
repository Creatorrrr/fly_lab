"""E310 generated nonzero whole continuation; no model/body/CUDA execution."""
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests import test_c_banc_recurrent_resource_body as approved
from tools import verify_c_banc_m3_resource_continuation as probe

COUNTS = {"fixtures": 0, "sessions": 0, "controls": 0, "neural_ticks": 0, "native_samples": 0,
          "snapshots": 0, "restores": 0, "partial_saves": 0, "original": 0, "model": 0, "body": 0, "CUDA": 0, "physics": 0, "fit": 0}
BEGAN = time.perf_counter()


@pytest.fixture(scope="session", autouse=True)
def generated_limits():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        yield
    import psutil
    memory = psutil.Process().memory_info()
    peak = getattr(memory, "peak_wset", memory.rss)
    wall = time.perf_counter() - BEGAN
    print(json.dumps({"e310_fake_counts": COUNTS, "fixture_wall_s": wall, "peak_rss_bytes": peak,
        "actual_original_model_neural_body_physics_CUDA_solver_fit": 0}, sort_keys=True))
    assert peak <= 1.5 * 1024 ** 3 and wall <= 90


def scientific_row(control):
    return {"control_tick": control, "time_s": control * .005, "fault": False,
            "body": {"position": [control * .01, 0., 0.], "basis": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}}


def records(control, arm="candidate"):
    active = arm == "candidate"
    common = {"control_tick": control, "network_tick": control * 20, "physics_time_s": control * .005}
    controlrow = {**common, "motor_rate32": np.full(2, .2, np.float32), "activation": np.array([.3], np.float64),
                  "command": np.array([.4], np.float64), "tendon": np.array([.5], np.float64)}
    activity = {**common, "complete": True, "rate": np.array([.2, .4], np.float32), "drive": np.array([.1, .3], np.float32),
                "adaptation": np.array([.05, .07], np.float32), "synaptic_resource": np.array([.9 if active else 1., 1.], np.float64)}
    recruitment = {"control_tick": control, "network_tick": control * 20, "time_s": control * .005, "complete": True,
        "bank": np.array([[.3, .2, .1]], np.float64), "activation": np.array([.3], np.float64),
        "original_activation": np.array([.1], np.float64), "motor_rate": np.array([.2], np.float64)}
    return controlrow, activity, recruitment


def prefix(arm="candidate"):
    rows = [scientific_row(c) for c in range(101)]
    native = {"time_s": np.arange(1, 5001) * .0001, "normal_load_bw": np.ones((5000, 6)),
              "slip_mm_s": np.ones((5000, 6)), "slip_valid": np.ones((5000, 6), bool),
              "nonfoot_contact": np.zeros(5000, bool), "upright": np.ones(5000)}
    all_records = [records(c, arm) for c in range(101)]
    values = {}
    for key, i in (("controls", 0), ("activity", 1), ("recruitment", 2)):
        values[key] = {name: np.asarray([r[i][name] for r in all_records]) for name in all_records[0][i]}
    values["recruitment"]["physics_time_s"] = values["recruitment"].pop("time_s")
    return {"trace": {"samples": rows, "model_hash": arm + "-fake", "graph_hash": "fake-graph", "parameters": {}}, "native": native, **values}


def saved_whole(arm):
    return {"control_tick": 100, "wall_s": 4.118, "network": {"tick": 2000, "synaptic_resource": np.array([.9 if arm == "candidate" else 1., 1.], np.float64),
        "rate": np.array([.2, .4], np.float32), "drive": np.array([.1, .3], np.float32), "adaptation": np.array([.05, .07], np.float32)},
        "adapter": {"bank": np.array([[.3, .2, .1]], np.float64)}, "body": {"time": .5, "qpos": np.array([.2, .1], np.float64),
        "warmstart": np.array([.2], np.float64)}, "sensory": {"filter": np.array([.6], np.float64)}, "held": {"command": np.array([.4], np.float64)}}


class FakeSession:
    def __init__(self, arm, deps, saved):
        COUNTS["sessions"] += 1
        COUNTS["restores"] += 1
        self.arm, self.deps, self.state = arm, deps, copy.deepcopy(saved)
        self.control_tick, self.time_s = 100, .5
        self.network = SimpleNamespace(tick=2000, identity=arm + "-fake", device="cuda", cuda_implementation="packed", interval_steps=20)
        self.adapter = SimpleNamespace(identity="adapter-fake")
        self.body = SimpleNamespace(fault=False, physics_time=lambda: self.time_s)
        self.fault, self.closed, self.close_calls = False, False, 0

    def snapshot(self):
        COUNTS["snapshots"] += 1
        return copy.deepcopy(self.state)

    def advance(self, controls):
        assert controls == 1 and not self.closed
        COUNTS["controls"] += 1
        mode = self.deps.mode if self.deps.bad_arm == self.arm else "pass"
        ticks = 7 if mode == "range" else 20
        self.network.tick += ticks
        COUNTS["neural_ticks"] += ticks
        if mode == "range":
            self.deps.original_error = RuntimeError("original fake range error")
            raise self.deps.original_error
        samples = 27 if mode in ("health", "partial") else 50
        first = round(self.time_s / .0001)
        for i in range(1, samples + 1):
            self.time_s = (first + i) * .0001
            self.native.append((self.time_s, np.ones(6), np.ones(6), np.ones(6, bool), False, 1.))
            COUNTS["native_samples"] += 1
        if mode in ("health", "partial"):
            raise approved.FakeHealthStop("first measured native health stop")
        self.control_tick += 1
        if mode == "clock":
            self.network.tick += 1
            COUNTS["neural_ticks"] += 1

    def close(self):
        self.closed = True
        self.close_calls += 1
        if self.deps.mode == "close" and self.deps.bad_arm == self.arm:
            raise OSError("fake close error")


class FakeDeps(approved.FakeDeps):
    def __init__(self, mode="pass", bad_arm="candidate"):
        super().__init__(mode)
        COUNTS["fixtures"] += 1
        self.bad_arm, self.combined_raw, self.raws = bad_arm, {}, {}

    def load(self, freeze):
        baseline = approved.metrics()
        baseline["slip"]["load_weighted_mm_s"] = 1.8026436968249955
        arms = {arm: {"ids": np.arange(2, dtype=np.int64), "arm": arm, "identity": arm + "-fake", "prefix": prefix(arm),
            "saved": saved_whole(arm), "freeze": {"adapter_identity": "adapter-fake"}, "baseline_short": baseline} for arm in ("neutral", "candidate")}
        return {"arms": arms, "baseline_short": baseline}

    def construct(self, arm, data, ledger, persist, on_return):
        row = {"label": arm, "known_completed_setup_steps": 3, "recorded_capture_setup_steps": 2,
               "unknown_setup_steps": int(self.mode == "unknown_setup" and arm == self.bad_arm)}
        ledger["construction"].append(row)
        ledger["body_construction"].append({"label": arm, "known_completed_settling_steps": 4,
            "unknown_step_entries": int(self.mode == "unknown_settling" and arm == self.bad_arm)})
        if self.mode == "constructor" and arm == self.bad_arm:
            raise RuntimeError("fake constructor error")
        if self.sessions:
            assert self.sessions[-1].closed
        session = FakeSession(arm, self, data["saved"])
        self.sessions.append(session)
        on_return(session)
        if self.mode == "post_return" and arm == self.bad_arm:
            raise OSError("fake ownership transfer failure")
        if self.mode == "initial" and arm == self.bad_arm:
            session.state["body"]["qpos"][0] += .1
        return session

    def initial(self, session, data, arm):
        return probe.NativeDependencies().initial(session, data, arm)

    def capture(self, session, raw):
        c = session.control_tick
        raw["rows"].append(scientific_row(c))
        if self.mode == "readout" and session.arm == self.bad_arm and c == 102:
            raise RuntimeError("fake readout error")
        controlrow, activity, recruitment = records(c, session.arm)
        raw["controls"].append(controlrow)
        raw["activity"].append(activity)
        raw["recruitment"].append(recruitment)
        if self.mode == "initial_readout" and session.arm == self.bad_arm and c == 100:
            raw["activity"][-1]["rate"][0] += .1

    def initial_readout(self, raw, data):
        return probe.compare_initial_readout(raw, data["prefix"])

    def save_state(self, path, session):
        if self.mode == "state" and session.arm == self.bad_arm and path.name == "final":
            raise OSError("fake final state failure")
        if self.mode in ("health", "partial", "range") and session.arm == self.bad_arm and path.name == "final":
            raise ValueError("unhealthy whole unavailable")
        self.write(path.with_suffix(".json"), session.snapshot())

    def save_partial(self, path, session):
        self.saved_partial += 1
        COUNTS["partial_saves"] += 1
        if self.mode == "partial":
            raise OSError("fake partial save failure")

    def save_raw(self, folder, raw, data, clock):
        self.raws[data["arm"]] = copy.deepcopy(raw)
        self.write(folder / "fake-raw.json", {"rows": len(raw["rows"]), "native": len(raw["native"])})
        if self.mode == "raw" and data["arm"] == self.bad_arm:
            raise OSError("fake raw error")
        return []

    def combined(self, raw, data):
        return probe.combine_raw(data["prefix"], raw)

    def save_combined(self, folder, combined, raw_folder, data):
        folder.mkdir()
        self.combined_raw[data["arm"]] = combined
        self.write(folder / "trace.json", {"samples": combined["rows"]})
        if self.mode == "combined" and data["arm"] == self.bad_arm:
            return [{"artifact": "fake-combined", "error": "fake combined storage failure"}]
        return []

    def short_metrics(self, combined, folder, clock, data):
        probe.validate_global_clocks(combined["rows"], combined["native_arrays"], complete=True)
        result = approved.metrics(data["arm"] == "candidate")
        result["health"] = probe.runner.short.short_health(combined["rows"], combined["native_arrays"], clock, False)
        result["gait"]["window"]["longest_forward_bout_s"] = .7 if data["arm"] == "candidate" else .2
        if self.mode == "candidate_short" and data["arm"] == "candidate":
            result["forward"]["fixed_initial_heading_forward_mm"] = .1
        return result


def execute(tmp_path, mode="pass", bad_arm="candidate", cleanup=None):
    deps = FakeDeps(mode, bad_arm)
    result, error = probe.execute({"diagnostic_id": "E310"}, tmp_path / "out", deps=deps, budget=lambda: None, cleanup=cleanup)
    return deps, result, error


@pytest.mark.parametrize("mode,bad_arm", [("pass", "candidate"), ("candidate_short", "candidate"), ("initial", "neutral"),
    ("initial_readout", "neutral"), ("constructor", "neutral"), ("constructor", "candidate"), ("post_return", "neutral"),
    ("health", "neutral"), ("health", "candidate"), ("range", "candidate"), ("partial", "candidate"), ("readout", "candidate"),
    ("clock", "candidate"), ("state", "candidate"), ("close", "neutral"), ("raw", "candidate"), ("combined", "candidate"),
    ("publication", "candidate"), ("unknown_setup", "candidate"), ("unknown_settling", "candidate"),
    ("unknown_setup", "neutral"), ("unknown_settling", "neutral")])
def test_actual_offset_loop_independent_arms_partial_costs_and_no_promotion(tmp_path, mode, bad_arm):
    cleanup = []
    deps, result, error = execute(tmp_path, mode, bad_arm, cleanup=lambda: cleanup.append(1))
    assert cleanup == [1]
    assert result["original_startup_pass"] == {"neutral": True, "candidate": False} and not result["startup_retested"]
    for key in ("candidate_adopted", "adopted", "F1_F8_complete", "F2_complete", "F6_exposed", "next_separate_10s_eligible",
                "actual_startup_pass", "short_gait_pass", "biological_claim", "causal_claim", "F2_future_disk_replay_verified"):
        assert result[key] is False
    assert all(s.close_calls == 1 for s in deps.sessions)
    if bad_arm == "neutral" and mode != "pass":
        assert "candidate" not in result["arms"] and all(s.arm == "neutral" for s in deps.sessions)
    if mode in ("pass", "candidate_short"):
        assert error is None and result["run_valid"] and [s.arm for s in deps.sessions] == ["neutral", "candidate"]
        assert all(s.control_tick == 300 and s.network.tick == 6000 for s in deps.sessions)
        assert result["cost"]["scientific_neural_ticks"] == 8000 and result["cost"]["scientific_physics_steps"] == 20000
        assert not result["arms"]["neutral"]["absolute_short_adjudication"]["pass"]
        assert result["arms"]["candidate"]["absolute_short_adjudication"]["pass"] is (mode == "pass")
        assert result["interval_resource_benefit_observed"] is (mode == "pass")
        for arm in ("neutral", "candidate"):
            assert len(deps.raws[arm]["rows"]) == 201 and len(deps.raws[arm]["native"]) == 10000
            assert len(deps.combined_raw[arm]["rows"]) == 301 and len(deps.combined_raw[arm]["native_arrays"]["time_s"]) == 15000
        ledger = json.loads((tmp_path / "out/advance-ledger.json").read_text("utf8"))
        for row in ledger["arms"].values():
            assert row["first_new_interval"]["control"] == 101 and row["first_new_interval"]["neural"] == 2020
            assert row["first_new_interval"]["physics_s"] == .505 and row["evaluation_clock"]["physical_s"] == 1.5
            assert row["actual_neural_s"] == pytest.approx(1.) and row["actual_physical_s"] == pytest.approx(1.)
    elif mode == "health":
        assert error is None and not result["run_valid"]
        assert result["cost"]["scientific_neural_ticks"] == (20 if bad_arm == "neutral" else 4020)
        assert result["cost"]["scientific_physics_steps"] == (27 if bad_arm == "neutral" else 10027)
    else:
        assert error is not None and not result["run_valid"]
        if mode == "range":
            assert error is deps.original_error and result["cost"]["scientific_neural_ticks"] == 4007
            assert result["cost"]["scientific_physics_steps"] == 10000 and deps.saved_partial == 1


@pytest.mark.parametrize("field", ["wall_s", "resource", "filter", "held", "warmstart", "bank", "scalar_type"])
def test_nonzero_full_restoration_preserves_every_component_and_blocks_mismatch(field):
    deps = FakeDeps()
    data = deps.load({})["arms"]["candidate"]
    session = FakeSession("candidate", deps, data["saved"])
    guard = probe.NativeDependencies()
    receipt = guard.initial(session, data, "candidate")
    assert receipt["whole_restoration_exact"] and receipt["excluded_fields"] == [] and receipt["numeric_tolerance"] == 0
    if field == "wall_s":
        session.state["wall_s"] += .01
    elif field == "resource":
        session.state["network"]["synaptic_resource"][0] = 1.
    elif field == "filter":
        session.state["sensory"]["filter"][0] = 0.
    elif field == "held":
        session.state["held"]["command"][0] = 0.
    elif field == "warmstart":
        session.state["body"]["warmstart"][0] = 0.
    elif field == "bank":
        session.state["adapter"]["bank"][0, 0] = 0.
    else:
        session.state["wall_s"] = np.float32(session.state["wall_s"])
    with pytest.raises(ValueError):
        guard.initial(session, data, "candidate")


@pytest.mark.parametrize("bad", ["prefix_unverified", "restored", "final", "unknown", "partial", "invalid", "nan"])
def test_evaluation_only_clock_offset_rejects_missing_or_partial_delta(bad):
    clock = SimpleNamespace(neural_s=1., physical_s=1., clock_valid=True)
    restored, final = {"control": 100, "neural": 2000, "physics": .5}, {"control": 300, "neural": 6000, "physics": 1.5}
    good = probe.evaluation_clock(clock, restored=restored, final=final, prefix_verified=True)
    assert good.neural_s == good.physical_s == 1.5 and clock.neural_s == clock.physical_s == 1.
    verified = bad != "prefix_unverified"
    if bad == "restored":
        restored["control"] = 0
    if bad == "final":
        final["neural"] = 5980
    if bad == "unknown":
        clock.physical_s = None
    if bad == "partial":
        clock.neural_s = .999
    if bad == "invalid":
        clock.clock_valid = False
    if bad == "nan":
        final["physics"] = float("nan")
    with pytest.raises(ValueError):
        probe.evaluation_clock(clock, restored=restored, final=final, prefix_verified=verified)


def test_original_heading_controls_forward_projection_and_boundary_merge():
    p = prefix()
    raw = {key: [] for key in ("rows", "controls", "activity", "recruitment", "native")}
    for c in range(100, 301):
        raw["rows"].append(scientific_row(c))
        a, b, d = records(c)
        raw["controls"].append(a); raw["activity"].append(b); raw["recruitment"].append(d)
    raw["native"] = [(i * .0001, np.ones(6), np.ones(6), np.ones(6, bool), False, 1.) for i in range(5001, 15001)]
    merged = probe.combine_raw(p, raw)
    assert [r["control_tick"] for r in merged["rows"]] == list(range(301))
    forward = probe.runner.short.short_forward(merged["rows"], complete=True)["fixed_initial_heading_forward_mm"]
    shifted = copy.deepcopy(merged["rows"])
    shifted[0]["body"]["basis"] = [[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]]
    assert forward > 0 and probe.runner.short.short_forward(shifted, complete=True)["fixed_initial_heading_forward_mm"] == 0.
    raw["rows"][1]["control_tick"] = 100
    with pytest.raises(ValueError):
        probe.combine_raw(p, raw)


def test_missing_matched_metrics_remain_unavailable():
    summary = probe.metric_summary({})
    assert summary["forward_mm"] is None and summary["longest_eligible_bout_s"] is None
    result = probe.matched_contrast(summary, summary)
    assert result["forward_mm_candidate_minus_neutral"] is None and not result["matched_bout_and_cycles_improved"]


@pytest.mark.parametrize("arm", ["neutral", "candidate"])
def test_production_construct_uses_saved_nonzero_wrapper_only(arm):
    seen = []
    frozen = saved_whole(arm)
    wrapper = SimpleNamespace(SSPRecoveryResourceTrial=SimpleNamespace(from_checkpoint=lambda graph, saved: seen.append(copy.deepcopy(saved)) or SimpleNamespace()))
    def construction(label, callback, ledger, persist, on_return):
        value = callback()
        on_return(value)
        return value
    deps = probe.NativeDependencies(wrapper=wrapper, construction=construction)
    result = deps.construct(arm, {"graph": "fake", "saved": frozen}, {}, lambda: None, lambda value: None)
    assert result._e291_arm == arm and len(seen) == 1
    probe.runner.layer.require_equal(seen[0], frozen, "Whole saved nonzero input unchanged")
    seen[0]["network"]["synaptic_resource"][0] = .2
    assert frozen["network"]["synaptic_resource"][0] == (.9 if arm == "candidate" else 1.)


@pytest.mark.parametrize("arm", ["neutral", "candidate"])
def test_production_endpoint_factors_use_each_actual_arm_selection(tmp_path, arm):
    raw = {"ids": np.arange(2, dtype=np.int64), "activity": [records(c, arm)[1] for c in (100, 101)]}
    data = {"source_ids": raw["ids"], "selection": {"utilization": np.array([.005 if arm == "candidate" else 0., 0.], np.float64),
            "recovery_tau": np.full(2, .15, np.float64)}}
    original = copy.deepcopy(data)
    probe.NativeDependencies().save_endpoints(tmp_path, raw, data)
    result = probe.runner.protocol.load_npz(tmp_path / "recovery-factor-endpoints.npz")
    assert result["control_tick"].tolist() == [100, 101] and result["network_tick"].tolist() == [2000, 2020]
    if arm == "neutral":
        assert np.all(result["g32"] == 1.) and raw["resource_summary"]["active_sources"] == 0
    else:
        assert np.all(result["g32"][:, 0] < 1.) and raw["resource_summary"]["active_sources"] == 1
    probe.runner.layer.require_equal(data, original, "Per-arm selection not modified")


def test_production_combined_artifacts_keep_boundary_and_original_heading(tmp_path):
    p = prefix()
    raw = {key: [] for key in ("rows", "controls", "activity", "recruitment", "native")}
    for c in range(100, 301):
        raw["rows"].append(scientific_row(c))
        a, b, d = records(c)
        raw["controls"].append(a); raw["activity"].append(b); raw["recruitment"].append(d)
    raw["native"] = [(i * .0001, np.ones(6), np.ones(6), np.ones(6, bool), False, 1.) for i in range(5001, 15001)]
    raw_folder = tmp_path / "continuation"
    raw_folder.mkdir()
    for key, filename in (("activity", "activity.npz"), ("recruitment", "native-recruitment-state.npz"), ("controls", "control-raw.npz")):
        arrays = {name: np.asarray([r[name] for r in raw[key]]) for name in raw[key][0]}
        if key == "recruitment":
            arrays["physics_time_s"] = arrays.pop("time_s")
        np.savez_compressed(raw_folder / filename, **arrays)
    joined = probe.combine_raw(p, raw)
    failures = probe.NativeDependencies().save_combined(tmp_path / "combined", joined, raw_folder, {"arm": "candidate", "prefix": p})
    assert failures == []
    trace = json.loads((tmp_path / "combined/trace.json").read_text("utf8"))
    assert trace["samples"][0] == p["trace"]["samples"][0] and len(trace["samples"]) == 301
    for filename in ("activity.npz", "native-recruitment-state.npz", "control-raw.npz"):
        arrays = probe.runner.protocol.load_npz(tmp_path / "combined" / filename)
        assert arrays["control_tick"].tolist() == list(range(301)) and len(arrays["network_tick"]) == 301


@pytest.mark.parametrize("change", ["verdict", "cost", "publication", "promoted", "clock", "closed"])
def test_producer_saved_verdict_and_actual_cost_semantics(change):
    cost = {"scientific_neural_ticks": 2000, "scientific_physics_steps": 5000, "unknown_cost_arms": [], "construction_calls": 1,
        "known_executed_setup_ticks": 280, "recorded_capture_setup_ticks": 140, "known_settling_physics_steps": 2000, "unknown_setup_calls": []}
    result = {"schema": probe.e309.RESULT_SCHEMA, "diagnostic_id": "E309", "run_valid": True, "publication_status": "complete",
        "actual_startup_pass": False, "cost": cost, "arms": {"candidate": {"startup": {"strict_E141_startup_screen": False}}},
        **{k: False for k in ("candidate_adopted", "adopted", "F1_F8_complete", "F2_complete", "F3_F8_pass", "F6_exposed",
            "natural_gait_complete", "ten_second_executed", "next_separate_10s_eligible", "biological_claim", "short_gait_pass")}}
    ledger = {"original_attempts": 1, "retries": 0, "fit_calls": 0, "arms": {"candidate": {"completed_controls": 100,
        "native_samples": 5000, "clock_valid": True, "close_returned": True, "actual_neural_s": .5, "actual_physical_s": .5}}}
    audit = {"cost": copy.deepcopy(cost), "schema": "flylab.m3.e309.root-original-raw-audit.v1", "pass": True,
        "diagnostic_run_valid": True, "original_strict_startup_pass": False}
    probe.validate_producer(result, ledger, audit, "candidate")
    if change == "verdict":
        result["actual_startup_pass"] = True
    elif change == "cost":
        result["cost"]["scientific_neural_ticks"] = 0
    elif change == "publication":
        result["publication_status"] = "invalid"
    elif change == "promoted":
        result["next_separate_10s_eligible"] = True
    elif change == "clock":
        ledger["arms"]["candidate"]["actual_physical_s"] = .49
    elif change == "closed":
        ledger["arms"]["candidate"]["close_returned"] = False
    with pytest.raises(ValueError):
        probe.validate_producer(result, ledger, audit, "candidate")


def test_primary_range_cleanup_publication_and_locked_invalidation_do_not_escape(tmp_path, monkeypatch):
    deps = FakeDeps("range")
    native_write = deps.write
    native_unlink = Path.unlink
    def write(path, value):
        if path.name == "result-sha256.json":
            raise OSError("secondary SHA write")
        if path.name == "result.json" and value.get("publication_status") == "invalid":
            raise PermissionError("secondary invalid result write")
        native_write(path, value)
    def unlink(path, *args, **kwargs):
        if path.name in ("output-manifest.json", "result-sha256.json", "result.json"):
            raise PermissionError("secondary locked invalidation")
        return native_unlink(path, *args, **kwargs)
    deps.write = write
    monkeypatch.setattr(Path, "unlink", unlink)
    def cleanup():
        raise OSError("secondary cleanup failure")
    result, error = probe.execute({"diagnostic_id": "E310"}, tmp_path / "out", deps=deps, budget=lambda: None, cleanup=cleanup)
    assert error is deps.original_error and result["first_error"]["message"] == "original fake range error"
    assert result["publication_status"] == "invalid" and not result["run_valid"] and not result["interval_resource_benefit_observed"]
    status = json.loads((tmp_path / "out/publication-status.json").read_text("utf8"))
    assert status["status"] == "invalid" and not status["success_receipts_trusted"]
    errors = json.loads((tmp_path / "out/errors.json").read_text("utf8"))["errors"]
    assert any("invalidate" in e["stage"] for e in errors) and any(e["stage"] == "runtime cleanup" for e in errors)
    assert not (tmp_path / "out/result-sha256.json").exists()


@pytest.mark.parametrize("real_type", [float, np.float16, np.float32, np.float64, np.longdouble])
def test_r1_production_summary_to_contrast_accepts_finite_numpy_real_scalars(real_type):
    neutral = {"forward": {"fixed_initial_heading_forward_mm": real_type(1.68)}, "slip": {"load_weighted_mm_s": real_type(1.94)},
        "gait": {"window": {"longest_forward_bout_s": real_type(.21), "bouts": []},
                 "events": {leg: {"complete_cycles": np.int64(0)} for leg in probe.runner.LEGS}}}
    candidate = copy.deepcopy(neutral)
    candidate["gait"]["window"]["longest_forward_bout_s"] = real_type(.205)
    result = probe.matched_contrast(probe.metric_summary(neutral), probe.metric_summary(candidate))
    delta = result["longest_continuous_forward_bout_s_candidate_minus_neutral"]
    assert type(delta) is float and delta == float(real_type(.205)) - float(real_type(.21)) and delta < 0
    assert result["complete_cycles_candidate_minus_neutral"] == {leg: 0 for leg in probe.runner.LEGS}
    assert not result["matched_bout_and_cycles_improved"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("invalid", [None, True, np.bool_(False), np.array(.21), np.array([.21]), complex(.21),
    np.complex128(.21), float("nan"), np.float64("inf"), np.float32("-inf"), ".21"])
def test_r1_production_summary_to_contrast_invalid_values_remain_unavailable(invalid):
    def summary(value):
        return probe.metric_summary({"gait": {"window": {"longest_forward_bout_s": value}}})
    for neutral, candidate in ((summary(invalid), summary(.21)), (summary(.21), summary(invalid))):
        result = probe.matched_contrast(neutral, candidate)
        assert result["longest_continuous_forward_bout_s_candidate_minus_neutral"] is None
        assert not result["matched_bout_and_cycles_improved"]


@pytest.mark.parametrize("integer_type", [int, np.int32, np.int64, np.uint64])
def test_r1_numpy_cycle_counts_are_actual_integers_and_negative_delta_does_not_wrap(integer_type):
    neutral = {"complete_cycles": {leg: integer_type(2) for leg in probe.runner.LEGS}}
    candidate = {"complete_cycles": {leg: integer_type(1) for leg in probe.runner.LEGS}}
    result = probe.matched_contrast(neutral, candidate)
    assert all(type(v) is int and v == -1 for v in result["complete_cycles_candidate_minus_neutral"].values())


@pytest.mark.parametrize("invalid", [True, np.bool_(False), 2., np.float64(2.), np.array(2), np.array([2]), complex(2), None])
def test_r1_cycle_count_nonintegers_remain_unavailable(invalid):
    neutral = {"complete_cycles": {leg: 0 for leg in probe.runner.LEGS}}
    candidate = {"complete_cycles": {leg: invalid for leg in probe.runner.LEGS}}
    result = probe.matched_contrast(neutral, candidate)
    assert all(v is None for v in result["complete_cycles_candidate_minus_neutral"].values())
    assert not result["matched_bout_and_cycles_improved"]
