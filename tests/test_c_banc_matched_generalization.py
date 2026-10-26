"""Generated-data checks for the E243 saved-data diagnostic."""

import hashlib
import json
import time

import numpy as np
import pytest

from tools import analyze_c_banc_m3_matched_generalization as diagnostic


def fixture():
    n, motors, k = 2001, 6, 2
    membership = np.array([9, 3, 15, 2, 21, 1])
    groups = [{"leg": "lf"} for _ in range(96)]
    for owner, leg in zip(membership, diagnostic.LEGS):
        groups[owner] = {"leg": leg}
    pool = np.zeros((96, motors))
    moment = np.zeros((42, 96))
    for i, owner in enumerate(membership):
        pool[owner, i] = 1
        moment[7*i, owner] = 2
    parameters = {"free_columns": np.arange(motors),
                  "fixed_columns": np.array([], dtype=np.int64),
                  "membership": membership, "pool": pool, "moment": moment,
                  "cap": np.ones(motors), "ltm": np.arange(90, 96)}
    descriptor = np.zeros((n, motors, 9))
    descriptor[:, 0, 0] = 1
    centers = np.zeros((motors, k, 9))
    centers[0, 0, 0] = 1
    centers[0, 1, 0] = 3
    geometry = {"centers": centers, "sigma2": np.ones(motors),
                "anchor_rows": np.tile(np.array([1, 1000]), (96, 1))}
    beta = np.zeros((96, k))
    beta[9] = [1, 0]
    target = np.zeros((n, 42))
    target[:, 0] = np.arange(n) * .001
    original = target + .01
    candidate = target + .005
    inputs = (parameters, geometry, {"descriptor": descriptor},
              {"target_command": target}, {"command": original},
              {"beta": beta, "q32": candidate}, groups,
              {"original": diagnostic.error_stats(original, target),
               "projected": diagnostic.error_stats(candidate, target)})
    return inputs


def test_rbf_gate_anatomy_and_metrics():
    report, raw = diagnostic.analyze_arrays(*fixture())
    assert int(raw["completed_mn"]) == 6
    assert raw["nearest_center"][100, 0] == 0
    assert raw["anchor_control"][100, 0] == 1
    np.testing.assert_allclose(raw["min_distance"][100, 0], 0)
    np.testing.assert_allclose(raw["gate"][100, 0], -np.expm1(-8))
    p = 1 / (1 + np.exp(-2))
    np.testing.assert_allclose(raw["instantaneous_target"][100, 0],
                               -np.expm1(-8) * p)
    np.testing.assert_allclose(raw["sensitivity_weight"].sum(axis=1), 1)
    assert report["e242_metric_max_abs_difference"] == 0
    assert report["descriptive_only"]
    assert report["observed_tail_independent_validation"] is False


def test_neighbor_exclusion_tie_and_control_velocity_bins():
    report, raw = diagnostic.analyze_arrays(*fixture())
    # Every descriptor in each leg is identical: earliest admissible training
    # control wins, including a query inside training and one in the tail.
    assert raw["nearest_train_control"][0, 0] == -1
    assert raw["nearest_train_distance"][0, 0] == 0
    assert raw["nearest_train_control"][1, 0] == 22
    assert raw["nearest_train_control"][1200, 0] == 1
    assert raw["nearest_train_control"][1201, 0] == 1
    assert raw["neighbor_teacher_q_difference"][1201, 0] == pytest.approx(-1.2)
    assert len(report["bins"]) == 10
    assert report["bins"][0]["first_control"] == 1
    assert report["bins"][0]["last_control"] == 200
    assert report["bins"][0]["global"]["original"]["velocity_rms_rad_s"] == pytest.approx(0)


def test_nonzero_euclidean_neighbor_and_bin_boundary_velocity():
    inputs = list(fixture())
    inputs[2]["descriptor"][1500, 0, 0] = 2
    inputs[4]["command"][201, 0] += .02
    inputs[-1]["original"] = diagnostic.error_stats(
        inputs[4]["command"], inputs[3]["target_command"])
    report, raw = diagnostic.analyze_arrays(*inputs)
    assert raw["nearest_train_control"][1500, 0] == 1
    assert raw["nearest_train_distance"][1500, 0] == pytest.approx(np.sqrt(.5))
    assert report["bins"][0]["global"]["original"]["velocity_max_abs_rad_s"] == pytest.approx(0)
    assert report["bins"][1]["global"]["original"]["velocity_max_abs_rad_s"] == pytest.approx(4)


def test_partial_raw_retained_at_completed_mn_boundary():
    raw = {}
    def stop(kind, number):
        if kind == "mn" and number == 2:
            raise TimeoutError("generated deadline")
    with pytest.raises(TimeoutError, match="generated deadline"):
        diagnostic.analyze_arrays(*fixture(), raw=raw, progress=stop)
    assert int(raw["completed_mn"]) == 2
    assert np.isfinite(raw["min_distance"][:, :2]).all()
    assert np.isnan(raw["min_distance"][:, 2:]).all()
    assert int(raw["completed_legs"]) == 0


def test_saved_metric_mismatch_rejected():
    inputs = list(fixture())
    inputs[-1]["projected"]["legs"]["lf"]["validation"]["q_rms_rad"] += .1
    with pytest.raises(ValueError, match="saved metric mismatch"):
        diagnostic.analyze_arrays(*inputs)


def test_exact_byte_publication_and_manifest(tmp_path):
    result = {"validity": "valid_saved_diagnostic"}
    ledger = {"completed_mn": 1, "errors": []}
    diagnostic.publish(tmp_path, result, {"x": np.array([1.])}, ledger)
    manifest = json.loads((tmp_path / "output-manifest.json").read_bytes())
    for name, entry in manifest["files"].items():
        data = (tmp_path / name).read_bytes()
        assert len(data) == entry["bytes"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    assert json.loads((tmp_path / "result.json").read_bytes()) == result


def test_deadline_prevents_success_receipt(tmp_path):
    with pytest.raises(TimeoutError, match="before result publication"):
        diagnostic.publish(tmp_path, {"validity": "valid_saved_diagnostic"},
                           {"x": np.array([1.])}, {"errors": []},
                           deadline=time.perf_counter()-1)
    assert not (tmp_path / "result.json").exists()


def test_first_ledger_error_keeps_raw_and_refreshes_manifest(tmp_path, monkeypatch):
    original = diagnostic.write_json
    failed = [False]
    def once(path, value):
        if path.name == "ledger.json" and not failed[0]:
            failed[0] = True
            raise OSError("generated first ledger failure")
        return original(path, value)
    monkeypatch.setattr(diagnostic, "write_json", once)
    result = {"validity": "valid_saved_diagnostic", "errors": []}
    ledger = {"errors": [], "completed_mn": 2}
    first = diagnostic.settle_publication(tmp_path, result,
        {"min_distance": np.array([[1., np.nan]])}, ledger,
        elapsed=lambda: .5, deadline=None)
    assert isinstance(first, OSError)
    assert first.args[0] == "generated first ledger failure"
    assert result["validity"] == "incomplete_or_invalid"
    assert result["finalization_pending"] is False
    with np.load(tmp_path / "diagnostic-raw.npz") as saved:
        assert np.isnan(saved["min_distance"][0, 1])
    assert len(json.loads((tmp_path / "errors.json").read_bytes())) == 1
    manifest = json.loads((tmp_path / "output-manifest.json").read_bytes())
    for name in ("errors.json", "ledger.json", "diagnostic-raw.npz"):
        assert manifest["files"][name]["sha256"] == hashlib.sha256(
            (tmp_path / name).read_bytes()).hexdigest()


def test_after_result_deadline_invalidates_and_refreshes(tmp_path, monkeypatch):
    clock = [0.]
    original = diagnostic.Path.write_bytes
    def advance(path, value):
        written = original(path, value)
        if path.name == "result.json" and clock[0] == 0:
            clock[0] = 2.
        return written
    monkeypatch.setattr(diagnostic.Path, "write_bytes", advance)
    monkeypatch.setattr(diagnostic.time, "perf_counter", lambda: clock[0])
    result = {"validity": "valid_saved_diagnostic", "errors": []}
    ledger = {"errors": []}
    first = diagnostic.settle_publication(tmp_path, result,
        {"x": np.array([1.])}, ledger, elapsed=lambda: clock[0], deadline=1.)
    assert isinstance(first, TimeoutError)
    saved = json.loads((tmp_path / "result.json").read_bytes())
    assert saved["validity"] == "incomplete_or_invalid"
    assert saved["finalization_pending"] is False
    assert saved["primary_error"]["type"] == "TimeoutError"
    manifest = json.loads((tmp_path / "output-manifest.json").read_bytes())
    for name in ("errors.json", "ledger.json"):
        assert manifest["files"][name]["sha256"] == hashlib.sha256(
            (tmp_path / name).read_bytes()).hexdigest()


def test_hash_gate_rejection_precedes_npz_load(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(diagnostic, "load_inputs", lambda _: called.append(True))
    with pytest.raises(ValueError, match="freeze SHA"):
        diagnostic.run(tmp_path / "missing", "bad", tmp_path / "review",
                       "bad", tmp_path / "out", generated_test_output=True)
    assert not called
    result = json.loads((tmp_path / "out" / "result.json").read_bytes())
    assert result["original_calculation_count"] == 0
    assert result["loaded_input_roles"] == []


@pytest.mark.parametrize("failed_name", ["output-manifest.json", "result.json"])
def test_late_write_failure_refreshes_all_receipts(tmp_path, monkeypatch, failed_name):
    original = diagnostic.Path.write_bytes
    seen = []

    def once(path, value):
        if path.name == failed_name and not seen:
            seen.append(True)
            raise OSError("generated late receipt failure")
        return original(path, value)

    monkeypatch.setattr(diagnostic.Path, "write_bytes", once)
    ledger = {"errors": []}
    result = {"validity": "valid_saved_diagnostic"}
    first = diagnostic.settle_publication(tmp_path, result, {"x": np.array([1.])},
        ledger, elapsed=lambda: .5, deadline=None)
    assert isinstance(first, OSError)
    saved = json.loads((tmp_path / "result.json").read_bytes())
    assert saved["validity"] == "incomplete_or_invalid"
    assert saved["finalization_pending"] is False
    assert saved["primary_error"]["message"] == "generated late receipt failure"
    assert saved["publication_wall_s"] >= 0
    inventory = json.loads((tmp_path / "output-manifest.json").read_bytes())["files"]
    assert set(inventory) == {"errors.json", "ledger.json", "diagnostic-raw.npz"}
    for name, item in inventory.items():
        assert item["sha256"] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()


def test_raw_cleanup_failure_retains_first_storage_error(tmp_path, monkeypatch):
    def fail_save(*_, **__):
        raise OSError("generated raw save failure")

    def fail_cleanup(*_, **__):
        raise OSError("generated cleanup failure")

    monkeypatch.setattr(diagnostic.np, "savez_compressed", fail_save)
    monkeypatch.setattr(diagnostic.Path, "unlink", fail_cleanup)
    ledger = {"errors": []}
    result = {"validity": "valid_saved_diagnostic"}
    first = diagnostic.settle_publication(tmp_path, result, {"x": np.array([1.])},
        ledger, elapsed=lambda: .5, deadline=None)
    assert first.args[0] == "generated raw save failure"
    saved = json.loads((tmp_path / "result.json").read_bytes())
    assert saved["validity"] == "incomplete_or_invalid"
    assert saved["finalization_pending"] is True
    assert saved["primary_error"]["message"] == "generated raw save failure"
    assert len(ledger["raw_secondary_errors"]) == 3
    assert saved["raw_secondary_errors"] == ledger["raw_secondary_errors"]
    assert all(e["message"] == "generated cleanup failure" for e in ledger["raw_secondary_errors"])
    assert len(ledger["errors"]) == 3


def test_original_work_error_precedes_thread_restore_error(tmp_path, monkeypatch):
    source = {}
    for name, value in {
            "source_result": {"validity": "valid_conditional_matched_teacher_fit",
                              "candidate_eligible": False},
            "source_ledger": {"solver_calls": 1},
            "source_proof": {"validity": "independently_valid"},
            "review": {"source_sha256": {},
                       "additional_reused_source_sha256": {},
                       "additional_input_sha256": {}}}.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value), encoding="utf-8")
        source[name] = path
    freeze = {"input_sha256": {}, "reused_source_sha256": {},
              "preservation_sha256": {}, "packages": {}}
    monkeypatch.setattr(diagnostic, "verify_gate", lambda *_: (freeze, source))
    monkeypatch.setattr(diagnostic, "repository_path", lambda _: source["review"])
    monkeypatch.setattr(diagnostic.shutil, "disk_usage",
                        lambda _: type("Disk", (), {"free": 1 << 40})())
    inputs = dict(zip(("parameters", "geometry", "descriptor", "targets",
                       "canonical_original", "candidate", "groups", "metrics"), fixture()))
    def load(_, *, progress):
        progress("parameters")
        return inputs
    monkeypatch.setattr(diagnostic, "load_inputs", load)
    monkeypatch.setattr(diagnostic, "validate_original_dimensions", lambda _: None)
    monkeypatch.setattr(diagnostic, "threadpool_info", list)
    class BrokenRestore:
        def restore_original_limits(self):
            raise RuntimeError("generated restore failure")
    monkeypatch.setattr(diagnostic, "threadpool_limits", lambda **_: BrokenRestore())
    def fail(*_, raw, progress):
        raw["partial"] = np.array([4.])
        progress("mn", 1)
        raise ValueError("generated analysis failure")
    monkeypatch.setattr(diagnostic, "analyze_arrays", fail)
    with pytest.raises(ValueError, match="generated analysis failure"):
        diagnostic.run(source["review"], "generated", source["review"],
                       "generated", tmp_path / "out", generated_test_output=True)
    out = tmp_path / "out"
    ledger = json.loads((out / "ledger.json").read_bytes())
    result = json.loads((out / "result.json").read_bytes())
    assert ledger["analysis_requested_count"] == 1
    assert ledger["analysis_entered_count"] == 1
    assert ledger["analysis_completed_count"] == 0
    assert ledger["original_calculation_count"] == 1
    assert ledger["completed_mn"] == 1
    assert ledger["threadpool_restore_completed"] is False
    assert [e["message"] for e in ledger["errors"]] == [
        "generated analysis failure", "generated restore failure"]
    assert result["primary_error"]["message"] == "generated analysis failure"
    assert result["finalization_pending"] is False
    assert result["full_banc_calls"] == result["physics_steps"] == 0
    with np.load(out / "diagnostic-raw.npz") as raw:
        assert raw["partial"][0] == 4
    manifest = json.loads((out / "output-manifest.json").read_bytes())
    for name in ("errors.json", "ledger.json", "diagnostic-raw.npz"):
        assert manifest["files"][name]["sha256"] == hashlib.sha256(
            (out / name).read_bytes()).hexdigest()
