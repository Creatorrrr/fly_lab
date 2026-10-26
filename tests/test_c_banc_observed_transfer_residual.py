"""Small saved-data fixtures only; original/model/body/CUDA/solver zero."""
import copy
import json
import time

import numpy as np
import psutil
import pytest

from tools import analyze_c_banc_m3_observed_transfer_residual as probe

BEGAN = time.perf_counter()


def fixture():
    source = np.array([2, 0, 1], np.int64)
    motors = np.array([5, 4], np.int64)
    matrix = np.array([[3, 2, -1], [-2, 0, 4]], np.float32)
    csr = {"indptr": np.array([0, 0, 0, 0, 0, 2, 5, 5], np.int32),
           "indices": np.array([1, 2, 0, 1, 2], np.int32), "data": np.array([4, -2, 2, -1, 3], np.float32)}
    ids = np.array([1, 5, 0, 4, 2], np.int64)
    b = {"observed_ids": ids, "rate": np.tile(np.array([4, 10, 3, 11, 2], np.float32), (100, 1)),
         "time_s": np.arange(1, 101, dtype=np.float64) * .005}
    ticks = np.arange(101, dtype=np.int64)
    a = {"observed_ids": np.array([5, 2, 4, 0], np.int64), "rate": np.tile(np.array([10, 5, 11, 7], np.float32), (101, 1)),
         "control_tick": ticks, "network_tick": ticks * 20, "physics_time_s": ticks * .005, "complete": np.ones(101, bool)}
    m = {"source_ids": source, "motor_ids": motors, "observed_ids": ids.copy(), "source_to_motor": matrix}
    return m, b, a, np.array([0], np.int64), csr


def test_unsorted_mapping_missing_nonzero_source_and_signed_hand_oracle():
    values = fixture()
    before = copy.deepcopy(values)
    result = probe.observed_transfer(*values)
    np.testing.assert_array_equal(result["common_source_ids"], [2, 0])
    np.testing.assert_array_equal(result["missing_source_ids"], [1])
    np.testing.assert_array_equal(result["missing_source_to_motor"], [[-1], [4]])
    np.testing.assert_array_equal(result["common_observed_signed_baseline"], np.tile([12., -4.], (100, 1)))
    np.testing.assert_array_equal(result["common_observed_signed_candidate"], np.tile([29., -10.], (100, 1)))
    np.testing.assert_array_equal(result["core_source_signed_delta"], np.tile([8., 0.], (100, 1)))
    np.testing.assert_array_equal(result["other_source_signed_delta"], np.tile([9., -6.], (100, 1)))
    for actual, expected in zip(values, before, strict=True):
        if isinstance(actual, dict):
            for key in actual:
                probe.p.equal_array(actual[key], expected[key], "Inputs unchanged")
        else:
            probe.p.equal_array(actual, expected, "Core unchanged")


@pytest.mark.parametrize("fault", ["duplicate", "missing_motor", "core_missing", "matrix", "dtype", "clock", "baseline_clock", "incomplete", "csr_duplicate", "csr_bounds", "nonfinite"])
def test_transfer_boundary_rejections(fault):
    m, b, a, core, csr = fixture()
    if fault == "duplicate":
        b["observed_ids"][0] = b["observed_ids"][1]
    elif fault == "missing_motor":
        a["observed_ids"][0] = 6
    elif fault == "core_missing":
        core = np.array([1], np.int64)
    elif fault == "matrix":
        m["source_to_motor"][0, 0] += 1
    elif fault == "dtype":
        a["rate"] = a["rate"].astype(np.float64)
    elif fault == "clock":
        a["physics_time_s"][50] += 2e-7
    elif fault == "baseline_clock":
        b["time_s"][0] = 0
    elif fault == "incomplete":
        a["complete"][2] = False
    elif fault == "csr_duplicate":
        csr["indices"][1] = csr["indices"][0]
    elif fault == "csr_bounds":
        csr["indptr"][-1] += 1
    else:
        a["rate"][0, 0] = np.nan
    with pytest.raises(ValueError):
        probe.observed_transfer(m, b, a, core, csr)


def test_empty_other_source_group_is_zero_and_no_imputation():
    m, b, a, _, csr = fixture()
    result = probe.observed_transfer(m, b, a, np.array([0, 2], np.int64), csr)
    assert np.all(result["other_source_signed_candidate"] == 0)
    assert result["candidate_rate"].shape == (100, 2)


def endpoint_fixture():
    m, b, a, core, _ = fixture()
    rate = np.zeros(158706, np.float32)
    rate[a["observed_ids"]] = a["rate"][100]
    rate[1] = 8
    state = {"control_tick": 100, "network": {"tick": 2000, "dt": .00025, "rate": rate},
             "body": {"state": [.7], "t0": .2}}
    return m, b, a, core, state


def test_full_endpoint_missing_source_contribution_only_at100():
    result = probe.endpoint_transfer(*endpoint_fixture())
    np.testing.assert_array_equal(result["missing_baseline_rate"], [[4.]])
    np.testing.assert_array_equal(result["missing_candidate_rate"], [[8.]])
    np.testing.assert_array_equal(result["missing_signed_delta"], [[-4., 16.]])
    np.testing.assert_array_equal(result["common_observed_signed_delta"], [[13., 10.]])
    assert result["candidate_rate"].shape == (1, 3)


@pytest.mark.parametrize("fault", ["tick", "physical", "observed", "dtype"])
def test_endpoint_clock_and_observed_bytes_reject(fault):
    values = endpoint_fixture()
    state = values[-1]
    if fault == "tick":
        state["network"]["tick"] = 1999
    elif fault == "physical":
        state["body"]["state"][0] += .001
    elif fault == "observed":
        state["network"]["rate"][5] += 1
    else:
        state["network"]["rate"] = state["network"]["rate"].astype(np.float64)
    with pytest.raises(ValueError):
        probe.endpoint_transfer(*values)


def test_startup_sse_cross_term_all_columns_and_99_vs100():
    b = np.tile(np.array([1., -2.]), (99, 1))
    c = b + [2., 1.]
    target = np.zeros_like(b)
    result = probe.startup_error_decomposition(b, c, target)
    columns = result["summary"]["all_columns"]
    assert columns[0]["candidate_SSE"] == 891 and columns[0]["cross_sum"] == 396
    assert columns[1]["candidate_SSE"] == 99 and columns[1]["cross_sum"] == -396
    assert result["arrays"]["candidate_error"].shape == (99, 2)
    assert all(row["candidate_error"]["peak_control"] == 1 for row in columns)
    with pytest.raises(ValueError):
        probe.startup_error_decomposition(np.vstack([b, b[:1]]), c, target)


def test_producer_manifest_exact_sha_and_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    folder = tmp_path / "producer"
    folder.mkdir()
    raw = folder / "raw.npz"
    raw.write_bytes(b"tiny")
    inputs = {"manifest": "producer/output-manifest.json", "raw": "producer/raw.npz"}
    sha = probe.p.sha256(raw)
    data = {"manifest": {"files": {"raw.npz": {"sha256": sha, "bytes": 4}}}}
    pins = {inputs["raw"]: sha}
    probe.producer_files(inputs, data, pins, "manifest", ("raw",))
    data["manifest"]["files"]["raw.npz"]["bytes"] = 5
    with pytest.raises(ValueError):
        probe.producer_files(inputs, data, pins, "manifest", ("raw",))
    with pytest.raises(ValueError):
        probe.pin_bytes([{inputs["raw"]: "0" * 64}])


def test_failure_receipt_no_scientific_promotion(tmp_path, monkeypatch):
    def fail(_inputs):
        raise ValueError("generated input failure")
    monkeypatch.setattr(probe, "load", fail)
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="generated"):
        probe.run({"inputs": {}}, output)
    result = json.loads((output / "failure.json").read_text())
    assert not result["analysis_valid"] and not result["candidate_selected"] and not result["causal_claim"]
    assert (output / "source-manifest.json").is_file() and all(value == 0 for value in result["execution"].values())
    assert not (output / "result.json").exists() and not (output / "output-manifest.json").exists()


@pytest.mark.parametrize("fault", [None, "result_sha", "path", "diagnostic", "schema", "cost", "startup", "flags"])
def test_manifest_before_result_publication_and_separate_result_binding(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    folder = tmp_path / probe.producer.OUTPUT_PATH
    folder.mkdir(parents=True)
    ledger = folder / "advance-ledger.json"
    ledger.write_text("{}")
    # Faithfully mirror publication order: manifest has ledger but no result.
    manifest = {"files": {ledger.name: {"sha256": probe.p.sha256(ledger), "bytes": ledger.stat().st_size}}}
    flags = {"run_valid": True, "actual_startup_pass": False, "short_gait_pass": False,
             "candidate_adopted": False, "F1_F8_complete": False, "F6_exposed": False}
    statistics = {"q": {"RMS": .019, "max_abs": .076}, "LTM": {"RMS": .0013, "max_abs": .013}}
    audit = {**flags, "cost": {"scientific_neural_ticks": 2020},
             "arms": {"candidate": {"controls": 100, "strict_startup": False, "startup": copy.deepcopy(statistics)}}}
    result = {**flags, "schema": "flylab.m3.e291.result.v1", "diagnostic_id": "E296", "cost": copy.deepcopy(audit["cost"]),
              "arms": {"candidate": {"completed_controls": 100, "health_stop": None, "status": "strict_startup_failed",
                "startup": {"strict_E141_startup_screen": False, "observed_controls": 99, "complete_original99_prefix": True,
                            "limits": probe.producer.STARTUP_LIMITS,
                            **{group: {**values, "available": True} for group, values in statistics.items()}}}}}
    if fault == "diagnostic":
        result["diagnostic_id"] = "E294"
    elif fault == "schema":
        result["schema"] = "wrong"
    elif fault == "cost":
        result["cost"]["scientific_neural_ticks"] += 1
    elif fault == "startup":
        result["arms"]["candidate"]["startup"]["q"]["RMS"] += .001
    elif fault == "flags":
        result["actual_startup_pass"] = True
    path = folder / ("alternate.json" if fault == "path" else "result.json")
    path.write_text(json.dumps(result))
    inputs = {"e296_result": path.relative_to(tmp_path).as_posix(), "ledger": ledger.relative_to(tmp_path).as_posix(),
              "manifest": (folder / "output-manifest.json").relative_to(tmp_path).as_posix()}
    data = {"e296_result": result, "e296_root_audit": audit, "e296_freeze": {"output": probe.producer.OUTPUT_PATH}, "manifest": manifest}
    pins = {inputs["e296_result"]: probe.p.sha256(path), inputs["ledger"]: probe.p.sha256(ledger)}
    probe.producer_files(inputs, data, pins, "manifest", ("ledger",))
    assert "result.json" not in manifest["files"]
    if fault == "result_sha":
        path.write_text(path.read_text() + " ")
    if fault:
        with pytest.raises(ValueError):
            probe.result_receipt(inputs, data, pins)
    else:
        probe.result_receipt(inputs, data, pins)


def test_gate_sha_and_generated_cost(tmp_path):
    file = tmp_path / "bad.json"
    file.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        probe.gate(file, "0" * 64, file, "0" * 64, tmp_path / "out")
    elapsed = time.perf_counter() - BEGAN
    memory = psutil.Process().memory_info()
    peak = getattr(memory, "peak_wset", memory.rss)
    assert elapsed < 45
    assert peak < 1 << 30
    print(json.dumps({"generated_wall_s": elapsed, "host_peak_bytes": peak, "original": 0, "model": 0, "neural": 0, "body": 0, "CUDA": 0, "solver": 0, "fit": 0}))
