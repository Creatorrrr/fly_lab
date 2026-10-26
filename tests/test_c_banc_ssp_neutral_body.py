"""E307 generated CPU fake contracts; actual neural/body/CUDA/original zero."""
import copy
import importlib.util
import json
import time
from types import SimpleNamespace

import numpy as np
import psutil
import pytest
from threadpoolctl import threadpool_limits

from tools import verify_c_banc_m3_ssp_neutral_body as probe

spec = importlib.util.spec_from_file_location("e307_approved_fake", probe.ROOT / probe.e306.OWNED[1])
approved = importlib.util.module_from_spec(spec)
spec.loader.exec_module(approved)
BEGAN = time.perf_counter()
COUNTS = {"fixtures": 0, "comparison_calls": 0}


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


def rows(count=101, difference=0.):
    return [{"control_tick": index, "time_s": index * .005, "joint_targets_rad": np.full(42, difference),
             "joint_angles_rad": np.full(42, difference * 2), "tendon_inputs": {f"{leg}_tarsus": float(difference) for leg in probe.runner.LEGS}}
            for index in range(count)]


class FakeDeps(approved.FakeDeps):
    def __init__(self, mode="pass"):
        super().__init__(mode)
        COUNTS["fixtures"] += 1
        self.guard = probe.NativeDependencies(wrapper=approved.FakeWrapper())
        self.guard.neural_references = {"neutral": approved.neural("neutral", 20)}
        self.guard.e306_first_reference = approved.neural("neutral", 20)
        self.mode = "neutral_mismatch" if mode == "first_mismatch" else mode

    def load(self, freeze):
        data = super().load(freeze)
        data.update(baseline_rows=rows(), e306_candidate_rows=rows(difference=.01),
                    startup={"target_command": np.zeros((99, 42)), "target_tendon": np.zeros((99, 6))})
        return data

    def construct(self, arm, data, ledger, persist, on_return):
        assert arm == "neutral"
        if self.mode == "constructor":
            raise RuntimeError("fake neutral constructor failure")
        session = super().construct(arm, data, ledger, persist, on_return)
        if self.mode in ("range", "health", "readout", "state", "partial", "unknown_setup"):
            def advance(count):
                assert count == 1
                counts = approved.approved.COUNTS
                counts["fake_controls"] += 1
                neural = 7 if self.mode in ("range", "partial") else 20
                session.network.tick += neural
                counts["fake_neural_ticks"] += neural
                if self.mode in ("range", "partial"):
                    self.original_error = RuntimeError("first fake SSP range error")
                    self.original_error.resource_fault_flags = 4
                    raise self.original_error
                native = 27 if self.mode == "health" else 50
                for _ in range(native):
                    session.time_s += .0001
                    session.native.append((session.time_s, np.ones(6), np.full(6, np.nan), np.zeros(6, bool), False, 1.))
                    counts["fake_native_samples"] += 1
                if self.mode == "health":
                    raise approved.approved.FakeHealthStop("fake neutral native health stop")
                session.control_tick += 1
            session.advance = advance
        if self.mode == "unknown_setup":
            ledger["construction"].append({"label": "fake unknown setup", "unknown_setup_steps": 1})
        return session

    def capture(self, session, raw):
        super().capture(session, raw)
        raw["rows"][-1]["joint_angles_rad"] = np.zeros(42, np.float64)
        if self.mode == "startup" and session.control_tick > 0:
            raw["rows"][-1]["joint_targets_rad"] = np.full(42, .07)
        if self.mode == "readout" and session.control_tick == 2:
            raise OSError("fake neutral readout failure")

    def startup(self, raw, data):
        return self.guard.startup(raw, data)

    def comparisons(self, raw, data):
        COUNTS["comparison_calls"] += 1
        return self.guard.comparisons(raw, data)

    def save_state(self, path, session):
        if self.mode in ("range", "partial") and path.name == "final":
            raise ValueError("fake sticky range final unavailable")
        if self.mode == "state" and path.name == "final":
            raise OSError("fake neutral final publication")
        return super().save_state(path, session)

    def save_partial(self, path, session):
        self.saved_partial += 1
        if self.mode == "partial":
            raise OSError("fake partial publication")


@pytest.mark.parametrize("mode,controls,valid", [("pass", 100, True), ("startup", 100, True), ("first_mismatch", 1, False),
    ("range", 0, False), ("health", 0, True), ("readout", 2, False), ("state", 100, False), ("partial", 0, False),
    ("close", 100, False), ("raw", 100, False), ("constructor", None, False), ("post_return", 0, False), ("publication", 100, False),
    ("unknown_setup", 100, False)])
def test_only_neutral_100_controls_no_continuation_and_cost(tmp_path, mode, controls, valid):
    deps = FakeDeps(mode)
    cleanup = []
    result, error = probe.execute({"diagnostic_id": "E307"}, tmp_path / "out", deps=deps, budget=lambda: None, cleanup=lambda: cleanup.append(1))
    assert cleanup == [1] and result["run_valid"] is valid
    assert len(deps.sessions) == int(controls is not None)
    assert set(result["arms"]) == {"neutral"}
    if deps.sessions:
        assert deps.sessions[0].arm == "neutral" and deps.sessions[0].control_tick == controls and deps.sessions[0].close_calls == 1
    assert result["schema"] == probe.RESULT_SCHEMA and result["arm_receipt_path"] == "tools.verify_c_banc_m3_ssp_neutral_body.run_arm"
    assert not result["candidate_adopted"] and not result["F1_F8_complete"] and not result["F2_complete"] and not result["F6_exposed"]
    assert not result["short_gait_pass"] and not result["next_separate_10s_eligible"] and not result["natural_gait_complete"] and not result["biological_claim"]
    if mode in ("pass", "startup"):
        assert error is None and result["cost"]["scientific_neural_ticks"] == 2000 and result["cost"]["scientific_physics_steps"] == 5000
        assert result["half_second_comparison_complete"] and result["ssp_neural_endpoint_exact"]
        assert result["actual_startup_pass"] is (mode == "pass")
        assert result["hypothesis_outcome"] == ("neutral_passes_original_startup" if mode == "pass" else "neutral_also_fails_original_startup")
        arm = result["arms"]["neutral"]
        assert arm["first_control_comparison"]["scope"] == "first .005-second interval only"
        assert arm["first_control_comparison"]["final_half_second_comparison"] is False
        assert arm["comparisons"]["controls"] == 100 and not arm["comparisons"]["new_acceptance_threshold"]
        assert len(deps.raw_records["neutral"]["rows"]) == 101
        pin = json.loads((tmp_path / "out/result-sha256.json").read_text())
        assert pin["sha256"] == probe.runner.protocol.sha256(tmp_path / "out/result.json")
    elif mode in ("range", "partial"):
        assert error is deps.original_error and result["cost"]["scientific_neural_ticks"] == 7 and result["cost"]["scientific_physics_steps"] == 0
        assert deps.saved_partial == 1 and deps.guard.resource_faults[-1]["flags"] == 4
    elif mode == "health":
        assert error is None and result["cost"]["scientific_neural_ticks"] == 20 and result["cost"]["scientific_physics_steps"] == 27
        assert not result["half_second_comparison_complete"] and result["hypothesis_outcome"] == "not_adjudicated"
    elif mode == "first_mismatch":
        assert result["cost"]["scientific_neural_ticks"] == 20 and "neutral" in deps.saved_neural
    if not valid:
        assert error is not None and result["hypothesis_outcome"] == "not_adjudicated"
    ledger = json.loads((tmp_path / "out/advance-ledger.json").read_text())
    assert ledger["arms"]["neutral"]["continuation_entered"] is False


@pytest.mark.parametrize("fault", ["control", "clock", "dtype", "shape", "nan", "missing", "short"])
def test_saved_trace_alignment_rejects_invalid(fault):
    actual, reference = rows(), rows()
    if fault == "control":
        reference[9]["control_tick"] = 10
    elif fault == "clock":
        reference[9]["time_s"] += .0001
    elif fault == "dtype":
        reference[9]["joint_angles_rad"] = np.zeros(42, np.float32)
        for row in reference:
            row["joint_angles_rad"] = np.asarray(row["joint_angles_rad"], np.float32)
    elif fault == "shape":
        reference[9]["joint_targets_rad"] = np.zeros(41)
    elif fault == "nan":
        reference[9]["joint_targets_rad"][8] = np.nan
    elif fault == "missing":
        reference[9].pop("tendon_inputs")
    else:
        reference.pop()
    with pytest.raises((ValueError, KeyError)):
        probe.compare_traces(actual, reference, "fake contrast")


def test_saved_comparison_peak_control_and_index_no_new_criterion():
    actual, reference = rows(), rows()
    actual[99]["joint_targets_rad"][8] = .06
    actual[50]["joint_angles_rad"][12] = .2
    arrays, report = probe.compare_traces(actual, reference, "fake")
    assert report["q"]["max_abs"] == .06 and report["q"]["max_control_tick"] == 99 and report["q"]["max_index"] == 8
    assert report["q"]["max_time_s"] == .495 and report["joint_angles"]["max_index"] == 12
    assert report["joint_angles"]["max_control_tick"] == 50 and report["q"]["selection_criterion"] is False
    assert arrays["q_delta"].shape == (101, 42) and arrays["LTM_delta"].shape == (101, 6)


def test_native_startup_health_criterion_preserved():
    deps = FakeDeps()
    data = deps.load({})
    raw = {"rows": rows(), "native": [(index * .0001, np.ones(6), np.full(6, np.nan), np.zeros(6, bool), False, 1.) for index in range(5000)]}
    assert deps.startup(raw, data)["strict_E141_startup_screen"]
    value = list(raw["native"][200])
    value[5] = .89
    raw["native"][200] = tuple(value)
    result = deps.startup(raw, data)
    assert result["strict_E141_command_screen"] and not result["strict_E141_startup_screen"] and not result["native_health_pass"]


def test_neutral_endpoint_reuses_existing_snapshot_and_all_metadata_checked():
    guard = probe.NativeDependencies()
    reference = approved.neural("neutral", 20)
    guard.neural_references["neutral"] = reference
    guard.e306_first_reference = copy.deepcopy(reference)
    class Network:
        def snapshot(self):
            raise AssertionError("Neutral must reuse existing whole snapshot")
        @property
        def resource_fault_flags(self):
            raise AssertionError("No per-control fault polling")
    raw = {"neutral_end_snapshot": {"network": copy.deepcopy(reference)}}
    guard.first_control(SimpleNamespace(_e291_arm="neutral", network=Network()), raw)
    assert raw["ssp_diagnostic_readouts"]["additional_neural_snapshots"] == 0
    assert raw["ssp_first_control_check"]["e306_neutral_endpoint_exact"]
    guard.e306_first_reference["recurrent_resource"]["precision"] = "bad-fake"
    with pytest.raises(ValueError):
        guard.first_control(SimpleNamespace(_e291_arm="neutral", network=Network()), raw)


def fake_e306_records():
    cost = {"construction_calls": 2, "known_executed_setup_ticks": 560, "recorded_capture_setup_ticks": 280, "known_settling_physics_steps": 4000,
            "scientific_neural_ticks": 2020, "scientific_physics_steps": 5050, "unknown_cost_arms": [], "unknown_setup_calls": []}
    flags = {key: False for key in ("candidate_adopted", "adopted", "F1_F8_complete", "F2_complete", "F3_F8_pass", "F6_exposed", "natural_gait_complete",
                                   "ten_second_executed", "actual_startup_pass", "short_gait_pass", "next_separate_10s_eligible", "biological_claim")}
    result = {"schema": probe.e306.RESULT_SCHEMA, "diagnostic_id": "E306", "run_valid": True, "publication_status": "complete",
              "neutral_prefix_valid": True, "ssp_neural_endpoint_exact": True, "cost": copy.deepcopy(cost), **flags,
              "arms": {"candidate": {"startup": {"strict_E141_startup_screen": False}}}}
    audit = {"schema": "flylab.m3.e306.root-original-raw-audit.v1", "pass": True, "diagnostic_run_valid": True, "candidate_rejected": True,
             "cost": copy.deepcopy(cost), **flags, "arms": {}}
    ledger = {"original_attempts": 1, "retries": 0, "fit_calls": 0, "arms": {},
              "construction": [{"known_completed_setup_steps": 280, "recorded_capture_setup_steps": 140, "unknown_setup_steps": 0, "unknown_capture_setup_steps": 0}] * 2,
              "body_construction": [{"known_completed_settling_steps": 2000, "unknown_step_entries": 0, "active_unreturned_frames": 0}] * 2}
    for arm, count in (("neutral", 1), ("candidate", 100)):
        audit["arms"][arm] = {"controls": count, "neural_ticks": count * 20, "native_samples": count * 50,
            "first_neural_endpoint_exact": True, "original_whole_start_exact": True, "closed": True}
        ledger["arms"][arm] = {"completed_controls": count, "native_samples": count * 50, "close_returned": True, "clock_valid": True,
            "continuation_entered": False, "actual_neural_s": count * .005, "actual_physical_s": count * .005}
    return result, ledger, audit


@pytest.mark.parametrize("fault", [None, "audit", "cost", "setup", "clock", "continued", "promoted", "startup"])
def test_actual_e306_valid_rejection_and_exact_cost_required(fault):
    result, ledger, audit = fake_e306_records()
    if fault == "audit":
        audit["pass"] = False
    elif fault == "cost":
        result["cost"]["scientific_neural_ticks"] = 2000
    elif fault == "setup":
        ledger["construction"][0]["unknown_setup_steps"] = 1
    elif fault == "clock":
        ledger["arms"]["candidate"]["actual_physical_s"] = .4
    elif fault == "continued":
        ledger["arms"]["candidate"]["continuation_entered"] = True
    elif fault == "promoted":
        result["F1_F8_complete"] = True
    elif fault == "startup":
        result["arms"]["candidate"]["startup"]["strict_E141_startup_screen"] = True
    if fault:
        with pytest.raises(ValueError):
            probe.validate_e306(result, ledger, audit)
    else:
        probe.validate_e306(result, ledger, audit)


@pytest.mark.parametrize("fail_result", [False, True])
def test_primary_error_survives_publication_cleanup_and_unlink(tmp_path, monkeypatch, fail_result):
    out = tmp_path / "out"
    class ConflictDeps(FakeDeps):
        def __init__(self):
            super().__init__("range")
            self.result_writes = 0
        def write(self, path, value):
            if path.name == "result-sha256.json":
                raise PermissionError("fake SHA lock")
            if path.name == "result.json":
                self.result_writes += 1
                if fail_result and self.result_writes > 1:
                    raise PermissionError("fake invalid result lock")
            return super().write(path, value)
    deps = ConflictDeps()
    original_unlink = probe.Path.unlink
    def locked(path, *args, **kwargs):
        if path.parent == out and path.name in {"output-manifest.json", "result-sha256.json", "result.json"}:
            raise PermissionError("fake locked stale success receipt")
        return original_unlink(path, *args, **kwargs)
    def cleanup():
        raise OSError("fake cleanup error")
    with monkeypatch.context() as context:
        context.setattr(probe.Path, "unlink", locked)
        result, error = probe.execute({"diagnostic_id": "E307"}, out, deps=deps, budget=lambda: None, cleanup=cleanup)
    assert error is deps.original_error and not result["run_valid"] and result["publication_status"] == "invalid"
    status = json.loads((out / "publication-status.json").read_text())
    assert status["success_receipts_trusted"] is False
    errors = json.loads((out / "errors.json").read_text())["errors"]
    assert any(row["stage"] == "runtime cleanup" for row in errors) and any(row["stage"] == "invalidate output-manifest.json" for row in errors)


def test_manifest_truthful_entrypoint_and_package_rules(tmp_path):
    path = tmp_path / "source-manifest.json"
    probe.NativeDependencies().write(path, {"source_sha256": {}, "input_sha256": {}})
    source = json.loads(path.read_text())
    assert source["executed_body_runner"] == "tools.verify_c_banc_m3_ssp_neutral_body.execute"
    assert source["executed_arm_runner"] == "tools.verify_c_banc_m3_ssp_neutral_body.run_arm"
    assert source["candidate_session_constructed"] is False
    assert set(probe.SOURCE_FILES) <= set(source["source_sha256"]) | set(source["owned_sha256"])
    probe.e306.validate_packages({"numpy": "fake", "python": "fake-python"}, {"numpy": "fake"}, {"numpy": "fake", "python": "fake-python"}, python_version="fake-python")
    with pytest.raises(ValueError):
        probe.e306.validate_packages({"numpy": "fake", "python": "fake-python", "extra": "bad"}, {"numpy": "fake"}, {"numpy": "fake", "python": "fake-python"}, python_version="fake-python")
    with pytest.raises(ValueError):
        probe.e306.require_pins({"frozen-source": "changed"}, {"frozen-source": "original"})



@pytest.mark.parametrize("fault", [None, "active_u", "negative_zero", "dtype", "shape", "source_order", "source_mismatch"])
def test_neutral_roster_exact_native_zero_contract(fault):
    data = {"source_ids": np.array([2, 4, 6], np.int64), "roster": {"source_ids": np.array([2, 4, 6], np.int64), "neutral_utilization": np.zeros(3, np.float64)}}
    if fault == "active_u":
        data["roster"]["neutral_utilization"][0] = .05
    elif fault == "negative_zero":
        data["roster"]["neutral_utilization"][0] = -0.
    elif fault == "dtype":
        data["roster"]["neutral_utilization"] = np.zeros(3, np.float32)
    elif fault == "shape":
        data["roster"]["neutral_utilization"] = np.zeros(2)
    elif fault == "source_order":
        data["source_ids"] = data["source_ids"][::-1]
    elif fault == "source_mismatch":
        data["roster"]["source_ids"][1] = 5
    if fault:
        with pytest.raises(ValueError):
            probe.validate_neutral_utilization(data)
    else:
        probe.validate_neutral_utilization(data)


def test_neutral_factor_unity_and_no_candidate_construction():
    arrays = {"g32": np.ones((101, 3), np.float32)}
    probe.require_neutral_factors(arrays)
    arrays["g32"][99, 1] = .9
    with pytest.raises(ValueError):
        probe.require_neutral_factors(arrays)
    with pytest.raises(ValueError, match="only one neutral"):
        probe.NativeDependencies().construct("candidate", {}, {}, lambda: None, lambda _value: None)


def test_literal_gate_and_generated_budget(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        probe.gate(path, "0" * 64, path, "0" * 64, tmp_path / "out")
    peak = getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)
    elapsed = time.perf_counter() - BEGAN
    assert elapsed < 60 and peak < 1536 << 20
    print(json.dumps({"generated_wall_s": elapsed, "host_peak_bytes": peak, "e307_fake_counts": COUNTS,
                      "e306_fake_counts": approved.COUNTS, "e291_fake_counts": approved.approved.COUNTS,
                      **{key: 0 for key in ("original", "model", "neural", "body", "physics", "CUDA", "solver", "fit")}}))
