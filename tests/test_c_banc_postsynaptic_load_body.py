"""E313 generated-only executor, diagnostic, and publication checks."""
import copy
import json
import time

import numpy as np
import psutil
import pytest
from threadpoolctl import threadpool_limits

from tests import test_c_banc_recurrent_resource_body as approved
from tools import verify_c_banc_m3_postsynaptic_load_body as probe

BEGAN = time.perf_counter()
COUNTS = {"fake_sessions": 0, "fake_controls": 0, "fake_neural_ticks": 0, "fake_native_samples": 0,
          "fake_diagnostic_packets": 0, "fake_raw_records": 0, "original": 0, "model": 0,
          "GraphStore": 0, "CUDA": 0, "native_body": 0, "solver": 0, "fit": 0}


@pytest.fixture(scope="session", autouse=True)
def generated_limits():
    before = {key: approved.COUNTS[key] for key in ("fake_sessions", "fake_controls", "fake_neural_ticks", "fake_native_samples")}
    with threadpool_limits(limits=1):
        yield
    for key, previous in before.items():
        COUNTS[key] = approved.COUNTS[key] - previous
    memory = psutil.Process().memory_info()
    peak = getattr(memory, "peak_wset", memory.rss)
    wall = time.perf_counter() - BEGAN
    print(json.dumps({"E313_body_fake_counts": COUNTS, "fixture_wall_s": wall, "peak_rss_bytes": peak}, sort_keys=True))
    assert peak < 2 * 1024 ** 3 and wall < 120


def diagnostic(control, arm="neutral"):
    h = np.array([control * .001, control * .002], np.float64)
    load_lambda = probe.LOAD_LAMBDA[arm]
    return {"target_ids": np.array([0, 2], np.int64), "h": h,
        "g": np.asarray(1. / (1. + load_lambda * h), np.float32),
        "gross": np.zeros(2, np.float32) if control == 0 else np.array([1., 2.], np.float32),
        "gross_available": control > 0, "gross_phase": "last-SSPRK33-stage-input",
        "gross_tick": control * 20 - 1 if control else None,
        "completed_tick": control * 20, "control_tick": control}


class FakeDeps(approved.FakeDeps):
    def __init__(self, mode="pass"):
        super().__init__(mode)
        self.raw_records = {}

    def load(self, freeze):
        return {"ids": np.arange(2, dtype=np.int64), "baseline_short": approved.metrics(),
            "matched_short": approved.metrics(False)}

    def construct(self, arm, data, ledger, persist, on_return):
        session = super().construct(arm, data, ledger, persist, on_return)
        session._e291_arm = arm
        ledger["construction"].append({"label": arm, "known_completed_setup_steps": 3,
            "recorded_capture_setup_steps": 2,
            "unknown_setup_steps": int(self.mode == "unknown_setup" and arm == "candidate"), "unknown_capture_setup_steps": 0})
        ledger["body_construction"].append({"label": arm, "known_completed_settling_steps": 4,
            "unknown_step_entries": int(self.mode == "unknown_body" and arm == "candidate")})
        return session

    def capture(self, session, raw):
        super().capture(session, raw)
        row = diagnostic(session.control_tick, session.arm)
        if self.mode == "bad_gross" and session.arm == "candidate" and session.control_tick == 1:
            row["gross_tick"] = 20
        raw.setdefault("load_diagnostics", []).append(row)
        COUNTS["fake_diagnostic_packets"] += 1
        probe.validate_diagnostic(row, session.network.tick, session.control_tick)
        if session.control_tick == 1:
            raw["ssp_first_control_neural"] = {"tick": 20, "rate": np.array([.1], np.float32)}

    def compare_neutral(self, raw, data):
        if self.mode == "neutral_mismatch":
            raise ValueError("fake exact lambda0 first interval mismatch")
        assert len(raw["rows"]) == 2 and len(raw["native"]) == 50
        return {"neutral_prefix_valid": True, "ssp_neural_endpoint_exact": True,
            "whole_memory_disk_exact": True, "load_metadata_separate": True}

    def save_raw(self, folder, raw, data, clock):
        self.raw_records[folder.name] = copy.deepcopy(raw)
        COUNTS["fake_raw_records"] += 1
        probe.save_diagnostics(folder / "load-diagnostics.npz", raw.get("load_diagnostics", []))
        return super().save_raw(folder, raw, data, clock)


def execute(tmp_path, mode="pass", cleanup=None):
    deps = FakeDeps(mode)
    result, error = probe.execute({"diagnostic_id": "E313"}, tmp_path / "out",
        deps=deps, budget=lambda: None, cleanup=cleanup)
    return deps, result, error


@pytest.mark.parametrize("mode,ends,valid", [("pass", [1, 300], True), ("startup", [1, 100], True),
    ("short", [1, 300], True), ("neutral_mismatch", [1], False),
    ("advance", [1, 0], False), ("readout", [1, 2], False),
    ("bad_gross", [1, 1], False), ("health", [1, 0], True),
    ("raw", [1], False), ("publication", [1, 300], False),
    ("close", [1], False), ("unknown_setup", [1, 300], False),
    ("unknown_body", [1, 300], False)])
def test_fixed_two_arm_protocol_and_fail_closed_costs(tmp_path, mode, ends, valid):
    deps, result, error = execute(tmp_path, mode)
    assert [s.control_tick for s in deps.sessions] == ends
    assert all(s.close_calls == 1 for s in deps.sessions)
    assert result["schema"] == probe.RESULT_SCHEMA and result["run_valid"] is valid
    assert result["candidate_adopted"] is False and result["biological_claim"] is False
    assert result["natural_gait_complete"] is False and result["F1_F8_complete"] is False and result["F6_exposed"] is False
    if mode == "pass":
        assert error is None and result["actual_startup_pass"] and result["short_gait_pass"]
        assert result["next_separate_10s_eligible"] and result["matched_contrast"]["complete_cycles_candidate_minus_neutral"]
        assert result["cost"]["scientific_neural_ticks"] == 6020
        assert result["cost"]["scientific_physics_steps"] == 15050
        assert len(deps.raw_records["neutral"]["load_diagnostics"]) == 2
        assert len(deps.raw_records["candidate"]["load_diagnostics"]) == 301
        with np.load(tmp_path / "out/candidate/load-diagnostics.npz", allow_pickle=False) as saved:
            assert saved["h"].shape == (301, 2) and saved["gross_tick"][1] == 19
    elif mode == "startup":
        assert error is None and not result["actual_startup_pass"] and not result["next_separate_10s_eligible"]
        assert result["cost"]["scientific_neural_ticks"] == 2020
    elif mode == "short":
        assert error is None and result["actual_startup_pass"] and not result["short_gait_pass"]
    elif mode == "health":
        assert error is None and not result["actual_startup_pass"]
        assert result["cost"]["scientific_physics_steps"] == 77
    if not valid:
        assert error is not None and result["next_separate_10s_eligible"] is False
        if mode == "publication":
            assert not (tmp_path / "out/result-sha256.json").exists()
            assert not (tmp_path / "out/output-manifest.json").exists()


def test_primary_science_error_survives_cleanup_and_publication_failure(tmp_path):
    class Failing(FakeDeps):
        def __init__(self):
            super().__init__("advance")
            self.first = None
        def advance(self, session, arm, index, clock, journal):
            try:
                return super().advance(session, arm, index, clock, journal)
            except Exception as error:
                self.first = error
                raise
        def write(self, path, value):
            if path.name == "result-sha256.json":
                raise PermissionError("fake receipt locked")
            return super().write(path, value)
    deps = Failing()
    def cleanup():
        raise OSError("fake cleanup secondary")
    result, error = probe.execute({"diagnostic_id": "E313"}, tmp_path / "out", deps=deps,
        budget=lambda: None, cleanup=cleanup)
    assert error is deps.first and not result["run_valid"]
    assert result["publication_status"] == "invalid"
    assert not (tmp_path / "out/result-sha256.json").exists()
    failures = json.loads((tmp_path / "out/errors.json").read_text())["errors"]
    assert any(x["stage"] == "runtime cleanup" for x in failures)
    assert any(x["stage"] == "publication" for x in failures)


@pytest.mark.parametrize("fault", ["ids", "h_dtype", "h_nan", "h_range", "g_shape", "gross_tick", "phase", "missing_gross"])
def test_actual_packet_contract_rejects_bad_scopes(fault):
    row = diagnostic(1, "candidate")
    if fault == "ids": row["target_ids"] = np.array([2, 0], np.int64)
    elif fault == "h_dtype": row["h"] = row["h"].astype(np.float32)
    elif fault == "h_nan": row["h"][0] = np.nan
    elif fault == "h_range": row["h"][0] = 1.01
    elif fault == "g_shape": row["g"] = np.array([1.], np.float32)
    elif fault == "gross_tick": row["gross_tick"] = 20
    elif fault == "phase": row["gross_phase"] = "endpoint"
    else: row["gross_available"] = False
    with pytest.raises(ValueError):
        probe.validate_diagnostic(row, 20, 1)


def test_scoped_neutral_first_command_mismatch_is_fatal():
    raw = {"controls": [], "recruitment": []}
    expected = {"load_reference_controls": {}, "load_reference_recruitment": {}}
    for key in ("motor_rate32", "activation", "command", "tendon"):
        values = np.ones((2, 1), np.float64)
        expected["load_reference_controls"][key] = values
        raw["controls"] = raw["controls"] or [{}, {}]
        for i in (0, 1): raw["controls"][i][key] = values[i].copy()
    for key in ("bank", "activation", "original_activation", "motor_rate"):
        values = np.ones((2, 1), np.float64)
        expected["load_reference_recruitment"][key] = values
        raw["recruitment"] = raw["recruitment"] or [{}, {}]
        for i in (0, 1): raw["recruitment"][i][key] = values[i].copy()
    probe.compare_neutral_readouts(raw, expected)
    raw["controls"][1]["command"][0] += .01
    with pytest.raises(ValueError): probe.compare_neutral_readouts(raw, expected)


def test_scalar_contrast_accepts_numpy_bout_and_keeps_original_absolute_judge():
    neutral, candidate = approved.metrics(False), approved.metrics(True)
    neutral["gait"]["window"]["longest_forward_bout_s"] = np.float64(.21)
    candidate["gait"]["window"]["longest_forward_bout_s"] = np.float64(.205)
    matched = probe.e310.matched_contrast(probe.e310.metric_summary(neutral), probe.e310.metric_summary(candidate))
    assert matched["longest_continuous_forward_bout_s_candidate_minus_neutral"] == pytest.approx(-.005)
    assert probe.runner.assess_short(neutral, candidate)["pass"] is True


@pytest.mark.parametrize("fault", [None, "source", "cuda", "future", "wrapper", "raw", "raw_original"])
def test_exact_e312_and_real_neural_fake_whole_producer_contract(monkeypatch, fault):
    source = {name: "a" * 64 for name in probe.ENGINE_FILES}
    wrapper = {name: "b" * 64 for name in probe.OWNED[:2]}
    evidence = {"generated/e312-evidence.json": "c" * 64}
    raw_path, raw_sha = "generated/e313-integration.json", "d" * 64
    pins = {**source, **wrapper, **evidence, raw_path: raw_sha}
    records = {"source_review": {"schema": "flylab.m3.e312.root-source-review.v1", "approved": True,
        "blocking_defects": [], "source_sha256": copy.deepcopy(source)},
        "qualification": {"schema": "flylab.m3.e312.postsynaptic-load-tiny-qualification.v1", "approved": True,
            "source_sha256": copy.deepcopy(source), "evidence_sha256": evidence,
            "checks": {key: True for key in ("cpu_pass", "cuda_pass", "independent_oracle_pass", "disk_future_exact")},
            "original_calls": 0, "native_body_constructors": 0, "F1_F8_complete": False, "F6_exposed": False},
        "whole_integration": {"schema": "flylab.m3.e313.root-tiny-integration.v1", "approved": True,
            "actual_tiny_neural": True, "native_body": False, "integration_path": raw_path,
            "integration_sha256": raw_sha, "wrapper_source_sha256": copy.deepcopy(wrapper)}}
    raw = {"schema": "flylab.m3.e313.real-neural-fake-body.v1", "pass": True,
        "original_calls": 0, "native_body_calls": 0, "exact_nonzero_disk_future": True}
    monkeypatch.setattr(probe.e305, "receipt", lambda *_: copy.deepcopy(raw))
    if fault == "source": records["source_review"]["source_sha256"][probe.ENGINE_FILES[0]] = "x" * 64
    elif fault == "cuda": records["qualification"]["checks"]["cuda_pass"] = False
    elif fault == "future": records["qualification"]["checks"]["disk_future_exact"] = False
    elif fault == "wrapper": records["whole_integration"]["wrapper_source_sha256"][probe.OWNED[0]] = "x" * 64
    elif fault == "raw": raw["exact_nonzero_disk_future"] = False
    elif fault == "raw_original": raw["original_calls"] = 1
    if fault is None:
        probe.validate_qualification(records, pins)
    else:
        with pytest.raises(ValueError): probe.validate_qualification(records, pins)


@pytest.mark.parametrize("fault", [None, "archive", "other_source", "review"])
def test_old_executed_e310_pair_and_current_r1_same_paths_remain_distinct(tmp_path, monkeypatch, fault):
    import hashlib
    def digest(value):
        return hashlib.sha256(value).hexdigest()
    names = probe.e310.OWNED
    old_pair = {name: digest(("old:" + name).encode()) for name in names}
    current_pair = {name: digest(("current-R1:" + name).encode()) for name in names}
    assert all(old_pair[name] != current_pair[name] for name in names)
    other_name, other_digest = "generated/unchanged-source.py", digest(b"unchanged")
    paths = {"freeze": probe.e310.OUTPUT_PATH + "-freeze.json",
        "review": probe.e310.OUTPUT_PATH + "-source-review.json",
        **{key: probe.e310.OUTPUT_PATH + "/" + filename for key, filename in
           (("result", "result.json"), ("result_sha256", "result-sha256.json"),
            ("source_manifest", "source-manifest.json"), ("output_manifest", "output-manifest.json"))},
        "root_audit": "verification/banc-natural-gait-m3-20260919/parent-e310-original-raw-audit.json",
        "correction": "verification/banc-natural-gait-m3-20260919/parent-e310-derived-contrast-correction.json",
        "neutral_arm_result": probe.MATCHED_SHORT_PATH}
    specs = {key: {"path": path, "sha256": digest(key.encode())} for key, path in paths.items()}
    archive = {}
    for name in names:
        archived_path = "generated/archive/" + name
        file = tmp_path / archived_path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(("old:" + name).encode())
        archive[name] = {"path": archived_path, "sha256": old_pair[name]}
    pins = {**{spec["path"]: spec["sha256"] for spec in specs.values()},
            **current_pair, other_name: other_digest,
            **{spec["path"]: spec["sha256"] for spec in archive.values()}}
    old = {"schema": probe.e310.FREEZE_SCHEMA, "resource_limits": probe.e310.LIMITS,
        "input_sha256": {}, "preservation_sha256": {},
        "reused_source_sha256": {**old_pair, other_name: other_digest}}
    result = {"schema": probe.e310.RESULT_SCHEMA, "run_valid": True, "publication_status": "complete",
        "arms": {"neutral": {"short": {"available": True}}},
        **{key: False for key in ("candidate_adopted", "F1_F8_complete", "F2_complete", "F6_exposed", "next_separate_10s_eligible")}}
    audit = {"pass": True, "diagnostic_run_valid": True,
        "result_sha256": specs["result"]["sha256"],
        "output_manifest_sha256": specs["output_manifest"]["sha256"]}
    records = {"freeze": old, "review": {"source_sha256": copy.deepcopy(old_pair)},
        "result": result, "result_sha256": {"sha256": specs["result"]["sha256"]},
        "source_manifest": {"owned_sha256": copy.deepcopy(old_pair)},
        "output_manifest": {"files": {}}, "root_audit": audit,
        "correction": {}, "neutral_arm_result": copy.deepcopy(result["arms"]["neutral"])}
    monkeypatch.setattr(probe.e305, "receipt", lambda spec, _pins: records[next(key for key, value in specs.items() if value == spec)])
    monkeypatch.setattr(probe, "safe", lambda path: tmp_path / path)
    freeze = {"e310_lineage": specs, "e310_executed_sources": archive}
    if fault == "archive":
        (tmp_path / archive[names[0]]["path"]).write_bytes(b"altered archive")
    elif fault == "other_source":
        pins.pop(other_name)
    elif fault == "review":
        records["review"]["source_sha256"][names[0]] = current_pair[names[0]]
    if fault is None:
        probe.matched_lineage(freeze, pins)
    else:
        with pytest.raises(ValueError): probe.matched_lineage(freeze, pins)


def test_diagnostic_raw_save_failure_does_not_erase_previous_artifact_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.runner.NativeDependencies, "save_raw",
        lambda *_: [{"artifact": "earlier-raw", "error": "saved failure"}])
    def broken(*_):
        raise OSError("late diagnostic storage error")
    monkeypatch.setattr(probe, "save_diagnostics", broken)
    deps = probe.NativeDependencies()
    folder = tmp_path / "candidate"
    folder.mkdir()
    failures = deps.save_raw(folder, {"load_diagnostics": []}, {}, None)
    assert [failure["artifact"] for failure in failures] == ["earlier-raw", "load-diagnostics.npz"]
    assert (folder / "load-diagnostics.json").exists()
    assert (folder / "resource-faults.json").exists()


def test_real_binary_parameter_file_is_read_as_npz_and_hash_changes_rejected(tmp_path, monkeypatch):
    path = tmp_path / "native-parameters.npz"
    threshold = np.full(4, 7.5, np.float32)
    tau = np.full(4, .02, np.float32)
    np.savez(path, threshold32=threshold, tau32=tau)
    monkeypatch.setattr(probe, "safe", lambda _: path)
    spec = {"path": probe.PARAMETER_PATH, "sha256": probe.runner.protocol.sha256(path)}
    params = probe.load_parameters(spec, 4)
    np.testing.assert_array_equal(params["threshold32"], threshold)
    np.testing.assert_array_equal(params["tau32"], tau)
    np.savez(path, threshold32=threshold + np.float32(1), tau32=tau)
    with pytest.raises(ValueError, match="SHA mismatch"):
        probe.load_parameters(spec, 4)
