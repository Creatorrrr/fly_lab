"""Generated tiny saved-state tests: model/body/CUDA/physics/solver/fit zero."""
import copy
import json
import time

import numpy as np
import pytest

from tools import analyze_c_banc_m3_resource_sign_balance as probe

BEGAN = time.perf_counter()


def circuit():
    s = {"premotor_ids": np.array([1, 2, 3], np.int64), "motor_ids": np.array([4], np.int64),
         "edge_ids": np.array([10, 20, 30, 40], np.int64),
         "pre_ids": np.array([1, 1, 2, 3], np.int64), "post_ids": np.array([2, 3, 3, 1], np.int64),
         "source_ids": np.array([1, 2, 3], np.int64), "edge_source_slots": np.array([0, 0, 1, 2], np.int32),
         "weights32": np.array([2, -3, -4, 1], np.float32)}
    a = {"observed_ids": np.array([1, 2, 3, 4], np.int64), "motor_ids": s["motor_ids"].copy(),
         "source_ids": s["source_ids"].copy(), "control_tick": np.array([0, 1], np.int64),
         "network_tick": np.array([0, 20], np.int64), "complete": np.ones(2, bool),
         "physics_time_s": np.array([0., .005]), "rate": np.array([[0, 0, 0, 0], [4, 2, 1, 7]], np.float32),
         "drive": np.zeros((2, 4), np.float32), "adaptation": np.zeros((2, 4), np.float32),
         "synaptic_resource": np.array([[1, 1, 1], [.5, .25, 1]], np.float64)}
    return s, a


def test_hand_circuit_signs_shared_source_and_input_immutability():
    s, a = circuit()
    before_s, before_a = copy.deepcopy(s), copy.deepcopy(a)
    result = probe.decompose(s, a)
    expected = {"J_exc": [1, 8, 0], "J_inh": [0, 0, -20],
                "J_exc_R": [1, 4, 0], "J_inh_R": [0, 0, -8],
                "excitation_loss": [0, 4, 0], "inhibition_relief": [0, 0, 12], "net_change": [0, -4, 12]}
    for key, values in expected.items():
        np.testing.assert_array_equal(result[key][1], values)
        assert result[key].dtype == np.float64
        np.testing.assert_array_equal(result[key][0], np.zeros(3))
    for key in s:
        probe.equal_array(s[key], before_s[key], key)
    for key in a:
        probe.equal_array(a[key], before_a[key], key)


@pytest.mark.parametrize("sign", [1, -1])
def test_empty_sign_and_neutral_resource(sign):
    s, a = circuit()
    s["weights32"] = np.abs(s["weights32"]) * np.float32(sign)
    a["synaptic_resource"][:] = 1
    result = probe.decompose(s, a)
    for key in ("excitation_loss", "inhibition_relief", "net_change"):
        assert not result[key].any()
    assert not result["J_inh" if sign > 0 else "J_exc"].any()


@pytest.mark.parametrize("mutation", ["observed_order", "source_order", "duplicate_edge", "duplicate_pop", "endpoint", "outside", "weights_dtype", "weights_zero", "weights_nan", "resource_dtype", "resource_shape", "resource_nan", "resource_low", "resource_high", "resource_initial", "slots", "rate_dtype", "rate_nan", "rate_negative", "control_duplicate", "network_clock", "physics_clock", "incomplete"])
def test_invalid_native_mapping_and_activity_rejected(mutation):
    s, a = circuit()
    if mutation == "observed_order":
        a["observed_ids"] = a["observed_ids"][::-1].copy()
    elif mutation == "source_order":
        a["source_ids"] = a["source_ids"][::-1].copy()
    elif mutation == "duplicate_edge":
        s["edge_ids"][1] = s["edge_ids"][0]
    elif mutation == "duplicate_pop":
        s["premotor_ids"][1] = 1
    elif mutation == "endpoint":
        s["post_ids"][0] = 4
    elif mutation == "outside":
        s["premotor_ids"][2] = 158706
    elif mutation == "weights_dtype":
        s["weights32"] = s["weights32"].astype(np.float64)
    elif mutation == "weights_zero":
        s["weights32"][0] = 0
    elif mutation == "weights_nan":
        s["weights32"][0] = np.nan
    elif mutation == "resource_dtype":
        a["synaptic_resource"] = a["synaptic_resource"].astype(np.float32)
    elif mutation == "resource_shape":
        a["synaptic_resource"] = a["synaptic_resource"][:, :2]
    elif mutation.startswith("resource_"):
        a["synaptic_resource"][0 if mutation == "resource_initial" else 1, 0] = {
            "resource_nan": np.nan, "resource_low": -.01, "resource_high": 1.01, "resource_initial": .5}[mutation]
    elif mutation == "slots":
        s["edge_source_slots"][1] = 1
    elif mutation == "rate_dtype":
        a["rate"] = a["rate"].astype(np.float64)
    elif mutation == "rate_nan":
        a["rate"][1, 0] = np.nan
    elif mutation == "rate_negative":
        a["rate"][1, 0] = -.1
    elif mutation == "control_duplicate":
        a["control_tick"][1] = 0
    elif mutation == "network_clock":
        a["network_tick"][1] = 21
    elif mutation == "physics_clock":
        a["physics_time_s"][1] += 2.1e-8
    else:
        a["complete"][1] = False
    with pytest.raises(ValueError):
        probe.decompose(s, a)


def recruitment():
    return {"control_tick": np.arange(3, dtype=np.int64), "network_tick": np.arange(3, dtype=np.int64) * 20,
        "physics_time_s": np.arange(3) * .005, "complete": np.ones(3, bool),
        "motor_rate": np.arange(3 * 391, dtype=np.float64).reshape(3, 391),
        **{key: np.ones(3, bool) for key in ("bank_available", "activation_available", "original_activation_available", "motor_rate_available")},
        "bank": np.zeros((3, 391, 3)), "activation": np.zeros((3, 96)), "original_activation": np.zeros((3, 96))}


def trace():
    return {"samples": [{"control_tick": i, "time_s": i * .005, "joint_targets_rad": [i] * 42, "upright": 1.} for i in range(3)]}


def test_baseline_matches_by_control_and_time():
    controls = np.array([0, 2], np.int64)
    value = recruitment()
    result = probe.matching_recruitment(value, controls, "baseline")
    np.testing.assert_array_equal(result, value["motor_rate"][[0, 2]])
    q, _ = probe.matching_trace(trace(), controls, "baseline")
    np.testing.assert_array_equal(q[:, 0], [0, 2])


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "time", "dtype", "complete"])
def test_baseline_recruitment_alignment_fail_closed(mutation):
    value = recruitment()
    if mutation == "duplicate":
        value["control_tick"][1] = 0
    elif mutation == "missing":
        value["control_tick"] += 1
    elif mutation == "time":
        value["physics_time_s"][1] += 2.1e-8
    elif mutation == "dtype":
        value["motor_rate"] = value["motor_rate"].astype(np.float32)
    else:
        value["complete"][1] = False
    with pytest.raises(ValueError):
        probe.matching_recruitment(value, np.arange(3, dtype=np.int64), "baseline")


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "time", "qshape", "qnan"])
def test_trace_alignment_fail_closed(mutation):
    value = trace()
    if mutation == "duplicate":
        value["samples"][1]["control_tick"] = 0
    elif mutation == "missing":
        value["samples"].pop()
    elif mutation == "time":
        value["samples"][1]["time_s"] += 2.1e-8
    elif mutation == "qshape":
        value["samples"][1]["joint_targets_rad"] = [0] * 41
    else:
        value["samples"][1]["joint_targets_rad"][0] = np.nan
    with pytest.raises(ValueError):
        probe.matching_trace(value, np.arange(3, dtype=np.int64), "baseline")


def native():
    return {"time_s": np.arange(1, 3125) * .0001, "normal_load_bw": np.ones((3124, 6)),
            "slip_mm_s": np.full((3124, 6), np.nan), "slip_valid": np.zeros((3124, 6), bool),
            "nonfoot_contact": np.zeros(3124, bool), "upright": np.r_[np.ones(3123), .89993517]}


def test_guard_records_actual_fields_without_merging_neural_clock():
    report = probe.native_guard(native())
    assert report["sample_index"] == 3123 and report["time_s"] == .3124
    assert report["upright_below_0p9"] and not report["nonfinite_guard"]
    assert not report["nonfoot_contact"] and report["slip_mm_s"] == [None] * 6


@pytest.mark.parametrize("cause", ["nonfoot", "upright_nan", "load_nan", "valid_slip_nan"])
def test_first_native_guard_observations_include_nonfinite_and_nonfoot(cause):
    value = native()
    value["upright"][-1] = 1.
    if cause == "nonfoot":
        value["nonfoot_contact"][-1] = True
    elif cause == "upright_nan":
        value["upright"][-1] = np.nan
    elif cause == "load_nan":
        value["normal_load_bw"][-1, 2] = np.nan
    else:
        value["slip_valid"][-1, 0] = True
    report = probe.native_guard(value)
    assert report["nonfoot_contact"] if cause == "nonfoot" else report["nonfinite_guard"]
    probe.canonical(report)


def test_earlier_guard_and_bad_native_clock_rejected():
    value = native()
    value["upright"][100] = .8
    with pytest.raises(ValueError):
        probe.native_guard(value)
    value = native()
    value["time_s"][100] += 2.1e-8
    with pytest.raises(ValueError):
        probe.native_guard(value)


def test_tiny_saved_analysis_output_is_descriptive(tmp_path, monkeypatch):
    selection, activity = circuit()
    data = {"roster": selection, "candidate_activity": activity, "candidate_native": native(),
            "current_path_source_audit": {"claim_scope": "source audit; no measured intermediate masks"}, "e292_result": {"cost": {}}}
    matched = {"motor_delta": np.zeros((2, 1)), "q_delta": np.zeros((2, 42)), "control_upright": np.ones(2),
               "native_guard": probe.native_guard(data["candidate_native"]), "partial_neural_s": .315, "partial_physical_s": .3124}
    monkeypatch.setattr(probe, "load_inputs", lambda _inputs: data)
    monkeypatch.setattr(probe, "validate_saved", lambda _data: matched)
    monkeypatch.setattr(probe, "FIXED_CONTROLS", (0, 1))
    result = probe.analyze({}, tmp_path / "out")
    assert result["analysis_valid"] and result["partial_neural_time_s"] == .315
    assert result["partial_physical_time_s"] == .3124
    assert not result["mask_evidence"]["intermediate_masks_measured"]
    assert not result["prefix_is_startup_adjudication"]
    assert all(result[key] is False for key in probe.FLAGS)
    assert result["execution"] == {key: 0 for key in ("original", "model", "CUDA", "body", "physics", "fit", "solver")}
    with np.load(tmp_path / "out/sign-decomposition.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["net_change"][1], [0, -4, 12])


def generated_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    def put(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return probe.sha256(path)
    sources = {name: put(name, {"generated_source": name}) for name in probe.SOURCE_FILES}
    inherited_name = "tools/inherited.py"
    inherited = {inherited_name: put(inherited_name, {"generated_source": "inherited"})}
    inputs, hashes = {}, {}
    for name in probe.INPUT_NAMES:
        if name.endswith("_checkpoint"):
            inputs[name] = "inputs/" + name
            for leaf in ("manifest.json", "state.json", "array_00000.npy"):
                path = inputs[name] + "/" + leaf
                hashes[path] = put(path, {"generated": leaf})
        else:
            inputs[name] = "inputs/" + name + ".json"
            hashes[inputs[name]] = put(inputs[name], {"generated": name})
    hashes[inputs["current_path_source_audit"]] = put(inputs["current_path_source_audit"], {
        "schema": "flylab.m3.e293.source-path-audit.v1", "approved": True, "no_runtime_mute": True,
        "checked_initial_cuts_all_false": True, "endpoint_masks_all_one": True,
        "source_sha256": {name: sources[name] for name in probe.PATH_SOURCES}, "inherited_source_sha256": inherited})
    design = "design.json"
    freeze = {"schema": probe.FREEZE_SCHEMA, "input_schema": probe.INPUT_SCHEMA, "owned_files": list(probe.OWNED),
              "inputs": inputs, "input_sha256": hashes, "source_sha256": sources, "resource_limits": probe.LIMITS,
              "fixed_controls": list(probe.FIXED_CONTROLS), "design": design, "design_sha256": put(design, {}), "output": "out"}
    def call():
        digest = put("freeze.json", freeze)
        review_sha = put("review.json", {"schema": probe.REVIEW_SCHEMA, "approved": True, "blocking_defects": [],
            "freeze_sha256": digest, "source_sha256": {name: sources[name] for name in probe.OWNED}})
        return probe.gate(tmp_path / "freeze.json", digest, tmp_path / "review.json", review_sha, tmp_path / "out")
    return freeze, call


def test_complete_generated_input_gate(tmp_path, monkeypatch):
    freeze, call = generated_gate(tmp_path, monkeypatch)
    assert call() == freeze


@pytest.mark.parametrize("mutation", ["input", "array_pin", "inherited", "source", "audit", "output", "path"])
def test_input_and_source_pin_tampering_rejected(tmp_path, monkeypatch, mutation):
    freeze, call = generated_gate(tmp_path, monkeypatch)
    if mutation == "input":
        (tmp_path / freeze["inputs"]["candidate_activity"]).write_text("mutated")
    elif mutation == "array_pin":
        del freeze["input_sha256"][freeze["inputs"]["partial_network_checkpoint"] + "/array_00000.npy"]
    elif mutation == "inherited":
        (tmp_path / "tools/inherited.py").write_text("mutated")
    elif mutation == "source":
        (tmp_path / probe.PATH_SOURCES[0]).write_text("mutated")
    elif mutation == "audit":
        path = tmp_path / freeze["inputs"]["current_path_source_audit"]
        audit = json.loads(path.read_text())
        audit["no_runtime_mute"] = False
        path.write_text(json.dumps(audit))
        freeze["input_sha256"][freeze["inputs"]["current_path_source_audit"]] = probe.sha256(path)
    elif mutation == "output":
        (tmp_path / "out").mkdir()
    else:
        freeze["inputs"]["candidate_activity"] = "../outside.json"
    with pytest.raises(ValueError):
        call()


def test_failed_publication_removes_success_receipt(tmp_path, monkeypatch):
    s, a = circuit()
    data = {"roster": s, "candidate_activity": a, "candidate_native": native(), "current_path_source_audit": {}, "e292_result": {"cost": {}}}
    matched = {"motor_delta": np.zeros((2, 1)), "q_delta": np.zeros((2, 42)), "control_upright": np.ones(2),
               "native_guard": {}, "partial_neural_s": .315, "partial_physical_s": .3124}
    monkeypatch.setattr(probe, "load_inputs", lambda _inputs: data)
    monkeypatch.setattr(probe, "validate_saved", lambda _data: matched)
    monkeypatch.setattr(probe, "FIXED_CONTROLS", (0, 1))
    original = probe.write_json
    def fail(path, value):
        if path.name == "result.json":
            raise OSError("generated publication failure")
        return original(path, value)
    monkeypatch.setattr(probe, "write_json", fail)
    with pytest.raises(OSError, match="generated publication"):
        probe.analyze({}, tmp_path / "out")
    assert not (tmp_path / "out/result.json").exists()
    assert not (tmp_path / "out/output-manifest.json").exists()
    assert not json.loads((tmp_path / "out/failure.json").read_text())["analysis_valid"]


def test_generated_cost_receipt():
    elapsed = time.perf_counter() - BEGAN
    assert elapsed < 60
    print(json.dumps({"wall_s": elapsed, "largest_generated_circuit_cells": 4, "edges": 4,
                      "original": 0, "model": 0, "CUDA": 0, "body": 0, "physics": 0, "solver": 0, "fit": 0}))
