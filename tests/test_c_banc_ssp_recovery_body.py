"""E306 generated fake session contracts; original/model/neural/CUDA/body zero."""
import copy
import importlib.util
import json
import time
from types import SimpleNamespace

import numpy as np
import psutil
import pytest
from threadpoolctl import threadpool_limits

from tools import verify_c_banc_m3_ssp_recovery_body as probe

spec = importlib.util.spec_from_file_location("e306_approved_fake", probe.ROOT / probe.runner.OWNED[1])
approved = importlib.util.module_from_spec(spec)
spec.loader.exec_module(approved)
BEGAN = time.perf_counter()
COUNTS = {"fixtures": 0, "diagnostic_snapshots": 0, "first_control_saved": 0}


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


def neural(arm, tick):
    return {"tick": tick, "model_hash": arm + "-fake-SSP", "recurrent_resource": {"schema": "fake-new-SSP", "precision": "fake-f64"},
            "rate": np.full(2, tick / 20, np.float32), "adaptation": np.zeros(2, np.float32),
            "drive": np.zeros(2, np.float32), "output_mask": np.ones(2, np.float32), "synaptic_resource": np.ones(3, np.float64)}


def whole(session):
    return {"network": session.network.snapshot(), "body": {"state": np.full(924, .125).tolist(), "action": {"joint_angles": np.full(42, .25).tolist()}}, "wall_s": .1}


class FakeWrapper:
    @staticmethod
    def remove_resource_layer(value):
        return copy.deepcopy(value)


class FakeDeps(approved.FakeDeps):
    def __init__(self, mode="pass"):
        super().__init__(mode)
        COUNTS["fixtures"] += 1
        self.guard = probe.NativeDependencies(wrapper=FakeWrapper())
        self.guard.neural_references = {arm: neural(arm, 20) for arm in ("neutral", "candidate")}
        self.saved_neural, self.raw_records, self.original_error = {}, {}, None

    def load(self, _freeze):
        return {"ids": np.arange(2, dtype=np.int64), "source_ids": np.arange(3, dtype=np.int64), "baseline_short": approved.metrics(),
                "saved": {"wall_s": 0.}, "baseline_end": {"wall_s": .2, "network": {"rate": np.zeros(2, np.float32), "adaptation": np.ones(2, np.float32)},
                  "body": {"state": np.zeros(924).tolist(), "action": {"joint_angles": np.zeros(42).tolist()}}}}

    def construct(self, arm, data, ledger, persist, on_return):
        session = super().construct(arm, data, ledger, persist, on_return)
        session._e291_arm = arm
        def snapshot():
            COUNTS["diagnostic_snapshots"] += 1
            state = neural(arm, session.network.tick)
            if session.control_tick == 1 and self.mode == arm + "_mismatch":
                state["recurrent_resource"]["precision"] = "mismatched-fake"
            return state
        session.network.snapshot = snapshot
        if self.mode == "range" and arm == "candidate":
            def failed(_controls):
                approved.COUNTS["fake_controls"] += 1
                approved.COUNTS["fake_neural_ticks"] += 7
                session.network.tick = 7
                self.original_error = RuntimeError("first fake SSP range fault")
                self.original_error.resource_fault_flags = 4
                raise self.original_error
            session.advance = failed
        return session

    def capture(self, session, raw):
        super().capture(session, raw)
        raw["activity"][-1]["synaptic_resource"] = np.ones(3, np.float64)
        if session.control_tick == 1:
            if session.arm == "neutral":
                raw["neutral_end_snapshot"] = whole(session)
            self.guard.first_control(session, raw)

    def compare_neutral(self, raw, data):
        return self.guard.compare_neutral(raw, data)

    def advance(self, session, arm, index, clock, journal):
        try:
            return super().advance(session, arm, index, clock, journal)
        except Exception as error:
            self.guard.record_fault(session, arm, index, error)
            raise

    def save_raw(self, folder, raw, data, clock):
        self.raw_records[folder.name] = copy.deepcopy(raw)
        if "ssp_first_control_neural" in raw:
            self.saved_neural[folder.name] = copy.deepcopy(raw["ssp_first_control_neural"])
            COUNTS["first_control_saved"] += 1
            np.savez_compressed(folder / "fake-first-control-neural.npz", rate=raw["ssp_first_control_neural"]["rate"])
        if "ssp_neutral_numeric_deltas" in raw:
            np.savez_compressed(folder / "ssp-neutral-numeric-deltas.npz", **raw["ssp_neutral_numeric_deltas"])
        self.write(folder / "resource-faults.json", {"events": self.guard.resource_faults})
        return super().save_raw(folder, raw, data, clock)

    def save_state(self, path, session):
        if self.mode == "range" and session.arm == "candidate" and path.name == "final":
            raise ValueError("sticky range fault snapshot unavailable")
        return super().save_state(path, session)


@pytest.mark.parametrize("mode,endpoints,valid", [("pass", [1, 300], True), ("startup", [1, 100], True),
    ("neutral_mismatch", [1], False), ("candidate_mismatch", [1, 1], False), ("range", [1, 0], False),
    ("health", [1, 0], True), ("readout", [1, 2], False), ("publication", [1, 300], False),
    ("close", [1], False), ("raw", [1], False), ("constructor", [1], False), ("post_return", [0], False)])
def test_honest_ssp_executor_gates_and_costs(tmp_path, mode, endpoints, valid):
    deps = FakeDeps(mode)
    cleanup_calls = []
    result, error = probe.execute({"diagnostic_id": "E306"}, tmp_path / "out", deps=deps, budget=lambda: None, cleanup=lambda: cleanup_calls.append(1))
    assert [session.control_tick for session in deps.sessions] == endpoints
    assert all(session.close_calls == 1 for session in deps.sessions)
    assert cleanup_calls == [1] and result["run_valid"] is valid
    assert result["schema"] == probe.RESULT_SCHEMA and result["diagnostic_id"] == "E306"
    assert "neutral_exact" not in result and "whole_end_exact" not in result
    assert not result["candidate_adopted"] and not result["natural_gait_complete"] and not result["F1_F8_complete"] and not result["F6_exposed"]
    if mode == "pass":
        assert error is None and result["actual_startup_pass"] and result["short_gait_pass"] and result["next_separate_10s_eligible"]
        assert result["cost"]["scientific_neural_ticks"] == 6020 and result["cost"]["scientific_physics_steps"] == 15050
        assert result["neutral_prefix_valid"] and result["ssp_neural_endpoint_exact"]
        comparisons = result["arms"]["neutral"]["neutral_prefix"]["old_RK4_numeric_comparisons"]
        assert comparisons["body_state"]["max_abs"] == .125 and comparisons["joint_angles"]["RMS"] == .25
        assert comparisons["rate"]["max_abs"] == 1. and not comparisons["rate"]["byteexact"]
        assert deps.raw_records["candidate"]["ssp_diagnostic_readouts"]["additional_neural_snapshots"] == 1
        result_pin = json.loads((tmp_path / "out/result-sha256.json").read_text())
        assert result_pin["sha256"] == probe.runner.protocol.sha256(tmp_path / "out/result.json")
    elif mode == "startup":
        assert error is None and result["cost"]["scientific_neural_ticks"] == 2020
        assert not result["actual_startup_pass"] and not result["next_separate_10s_eligible"]
    elif mode == "candidate_mismatch":
        assert "candidate" in deps.saved_neural and result["cost"]["scientific_neural_ticks"] == 40
        assert result["arms"]["candidate"]["completed_controls"] == 1 and not result["actual_startup_pass"]
    elif mode == "neutral_mismatch":
        assert "neutral" in deps.saved_neural and not result["neutral_prefix_valid"]
    elif mode == "range":
        assert error is deps.original_error and result["cost"]["scientific_neural_ticks"] == 27
        assert result["cost"]["scientific_physics_steps"] == 50 and deps.saved_partial == 1
        assert deps.guard.resource_faults[-1]["flags"] == 4 and deps.guard.resource_faults[-1]["actual_neural_tick"] == 7
    elif mode == "health":
        assert result["cost"]["scientific_physics_steps"] == 77 and result["cost"]["scientific_neural_ticks"] == 40
        assert result["arms"]["candidate"]["status"] == "measured_native_health_stop"
    if not valid:
        assert error is not None and not result["next_separate_10s_eligible"]
        if mode == "publication":
            assert not (tmp_path / "out/output-manifest.json").exists() and not (tmp_path / "out/result-sha256.json").exists()


def test_cleanup_failure_preserves_primary(tmp_path):
    deps = FakeDeps("range")
    def cleanup():
        raise OSError("fake cleanup secondary")
    result, error = probe.execute({"diagnostic_id": "E306"}, tmp_path / "out", deps=deps, budget=lambda: None, cleanup=cleanup)
    assert error is deps.original_error and not result["run_valid"]
    errors = json.loads((tmp_path / "out/errors.json").read_text())["errors"]
    assert any(item["stage"] == "runtime cleanup" for item in errors)


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "missing"])
def test_numeric_delta_rejects_invalid_fields(fault):
    a, b = np.zeros(4, np.float32), np.zeros(4, np.float32)
    if fault == "shape":
        a = a[:3]
    elif fault == "dtype":
        a = a.astype(np.float64)
    elif fault == "nan":
        a[0] = np.nan
    else:
        a = None
    with pytest.raises(ValueError):
        probe.numeric_delta(a, b, (4,), "fake")


def test_e305_full_metadata_endpoint_comparison_and_faultgetter_not_polled():
    guard = probe.NativeDependencies()
    guard.neural_references["candidate"] = neural("candidate", 20)
    class PacketNetwork:
        @property
        def resource_fault_flags(self):
            raise AssertionError("Successful path faultgetter prohibited")
        def snapshot(self):
            return neural("candidate", 20)
    session = SimpleNamespace(_e291_arm="candidate", network=PacketNetwork())
    raw = {}
    guard.first_control(session, raw)
    assert raw["ssp_first_control_check"]["ssp_neural_endpoint_exact"]
    guard.neural_references["candidate"]["recurrent_resource"]["precision"] = "wrong"
    with pytest.raises(ValueError):
        guard.first_control(session, {})


def fake_e305_records():
    arms = {arm: {"actual_tick": 20, "requested_steps": 20, "advance_requested": 1, "advance_returned": 1, "advance_failed": 0,
                  "partial_tick_unknown": False, "cleanup_returned": 1, "initial_snapshot_available": True, "final_snapshot_available": True} for arm in ("neutral", "candidate")}
    ledger = {"arms": arms, "models_requested": 2, "models_returned": 2, "scientific_ticks": 40, "unknown_tick_arms": [],
              "construction_profiles": [{"known_completed_setup_steps": 40, "recorded_capture_setup_steps": 20, "unknown_setup_steps": 0, "unknown_capture_setup_steps": 0}] * 2,
              **{key: 0 for key in ("body_constructors", "physics_steps", "fit_calls", "solver_calls", "retries")}}
    identities = {"neutral": "fake-neutral", "candidate": "fake-candidate"}
    result = {"schema": probe.e305.RESULT_SCHEMA, "diagnostic_id": "E305", "run_valid": True, "cost": copy.deepcopy(ledger), "identities": identities,
              "numeric_comparisons": {"fake": {"max_abs": 1}}, **probe.e305.FLAGS}
    audit = {"schema": "flylab.m3.e305.root-original-raw-audit.v1", "pass": True, "diagnostic_run_valid": True, "identities": identities,
              "numeric_comparisons": copy.deepcopy(result["numeric_comparisons"]), **probe.e305.FLAGS,
              "cost": {"models": 2, "scientific_ticks": 40, "executed_setup_ticks": 80, "captured_setup_ticks": 40, "unknown_setup_ticks": 0, "physics_steps": 0, "body_constructors": 0}}
    freeze = {arm + "_network_identity": identity for arm, identity in identities.items()}
    return result, ledger, audit, freeze


@pytest.mark.parametrize("fault", [None, "science", "setup", "unknown", "identity", "promotion", "audit", "arm"])
def test_actual_e305_semantics_required(fault):
    result, ledger, audit, freeze = fake_e305_records()
    if fault == "science":
        ledger["scientific_ticks"] = 39
    elif fault == "setup":
        ledger["construction_profiles"][0]["recorded_capture_setup_steps"] = 19
    elif fault == "unknown":
        ledger["unknown_tick_arms"] = ["candidate"]
    elif fault == "identity":
        freeze["candidate_network_identity"] = "old-fake"
    elif fault == "promotion":
        audit["F1_F8_complete"] = True
    elif fault == "audit":
        audit["pass"] = False
    elif fault == "arm":
        ledger["arms"]["candidate"]["actual_tick"] = 19
    if fault:
        with pytest.raises(ValueError):
            probe.validate_e305(result, ledger, audit, freeze)
    else:
        probe.validate_e305(result, ledger, audit, freeze)


def test_manifest_names_new_execute_and_original_run_arm(tmp_path):
    path = tmp_path / "source-manifest.json"
    probe.NativeDependencies().write(path, {"source_sha256": {"additional-frozen": "a" * 64}, "input_sha256": {}})
    data = json.loads(path.read_text())
    assert data["executed_body_runner"] == "tools.verify_c_banc_m3_ssp_recovery_body.execute"
    assert data["reused_arm_runner"] == "tools.verify_c_banc_m3_recurrent_resource_body.run_arm"
    assert set(data["owned_sha256"]) == set(probe.OWNED)
    assert set(probe.SOURCE_FILES) <= set(data["source_sha256"]) | set(data["owned_sha256"])



@pytest.mark.parametrize("fault", [None, "existing_version", "missing_python", "wrong_python", "extra_key", "missing_existing", "actual_differs"])
def test_e296_existing_packages_and_only_e305_current_python(fault):
    baseline = {"numpy": "fake1", "torch": "fake2"}
    ssp = {**baseline, "python": "fake-python"}
    actual = copy.deepcopy(ssp)
    if fault == "existing_version":
        ssp["torch"] = "fake-different"
    elif fault == "missing_python":
        ssp.pop("python")
    elif fault == "wrong_python":
        ssp["python"] = "fake-stale-python"
    elif fault == "extra_key":
        ssp["arbitrary"] = "new-package"
    elif fault == "missing_existing":
        ssp.pop("numpy")
    elif fault == "actual_differs":
        actual["numpy"] = "different"
    if fault:
        with pytest.raises(ValueError):
            probe.validate_packages(actual, baseline, ssp, python_version="fake-python")
    else:
        probe.validate_packages(actual, baseline, ssp, python_version="fake-python")
        assert baseline == {"numpy": "fake1", "torch": "fake2"}


@pytest.mark.parametrize("mode,fail_invalid_result", [("range", False), ("range", True), ("pass", True)])
def test_publication_and_unlink_errors_never_mask_primary(tmp_path, monkeypatch, mode, fail_invalid_result):
    out = tmp_path / "out"
    class ConflictDeps(FakeDeps):
        def __init__(self):
            super().__init__(mode)
            self.result_writes = 0
            self.sha_error = PermissionError("fake result SHA file locked")
        def write(self, path, value):
            if path.name == "result-sha256.json":
                raise self.sha_error
            if path.name == "result.json":
                self.result_writes += 1
                if fail_invalid_result and self.result_writes > 1:
                    raise PermissionError("fake invalid result locked")
            return super().write(path, value)
    deps = ConflictDeps()
    old_unlink = probe.Path.unlink
    def conflict_unlink(path, *args, **kwargs):
        if path.parent == out and path.name in {"output-manifest.json", "result-sha256.json", "result.json"}:
            raise PermissionError("fake stale success receipt locked " + path.name)
        return old_unlink(path, *args, **kwargs)
    with monkeypatch.context() as context:
        context.setattr(probe.Path, "unlink", conflict_unlink)
        result, error = probe.execute({"diagnostic_id": "E306"}, out, deps=deps, budget=lambda: None)
    expected = deps.original_error if mode == "range" else deps.sha_error
    assert error is expected and not result["run_valid"] and result["publication_status"] == "invalid"
    assert not result["next_separate_10s_eligible"]
    status = json.loads((out / "publication-status.json").read_text())
    assert status["status"] == "invalid" and status["success_receipts_trusted"] is False
    errors = json.loads((out / "errors.json").read_text())["errors"]
    assert any(item["stage"] == "invalidate output-manifest.json" for item in errors)
    assert any(item["stage"] == "invalidate result-sha256.json" for item in errors)
    if fail_invalid_result:
        assert any(item["stage"] == "invalidate result.json" for item in errors)
    else:
        assert json.loads((out / "result.json").read_text())["run_valid"] is False
    assert all(session.close_calls == 1 for session in deps.sessions)


def test_raw_readout_publication_keeps_previous_failures_and_later_artifacts(tmp_path):
    class RawDeps(probe.NativeDependencies):
        def __init__(self):
            super().__init__()
            self.attempts = []
        def inherited_save_raw(self, *_args):
            return [{"artifact": "previous-native-raw", "error": "earlier fake failure"}]
        def save_neural(self, path, _state):
            self.attempts.append(path.name)
        def write(self, path, value):
            self.attempts.append(path.name)
            if path.name == "ssp-diagnostic-readouts.json":
                raise OSError("fake diagnostic publication failure")
            return probe.runner.NativeDependencies.write(self, path, value)
    deps = RawDeps()
    raw = {"ssp_first_control_neural": neural("candidate", 20), "ssp_neutral_numeric_deltas": {"fake": np.zeros(4)}}
    failures = deps.save_raw(tmp_path, raw, {}, None)
    assert [entry["artifact"] for entry in failures] == ["previous-native-raw", "ssp-diagnostic-readouts.json"]
    assert deps.attempts == ["first-control-neural", "ssp-diagnostic-readouts.json", "ssp-first-control-check.json"]
    assert (tmp_path / "ssp-first-control-check.json").is_file() and (tmp_path / "ssp-neutral-numeric-deltas.npz").is_file()


def test_sha_gate_and_generated_budget(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        probe.gate(path, "0" * 64, path, "0" * 64, tmp_path / "out")
    memory = psutil.Process().memory_info()
    peak = getattr(memory, "peak_wset", memory.rss)
    elapsed = time.perf_counter() - BEGAN
    assert elapsed < 60 and peak < 1536 << 20
    print(json.dumps({"generated_wall_s": elapsed, "host_peak_bytes": peak, "e306_fake_counts": COUNTS, "approved_fake_counts": approved.COUNTS,
                      **{key: 0 for key in ("original", "model", "neural", "body", "physics", "CUDA", "solver", "fit")}}))
