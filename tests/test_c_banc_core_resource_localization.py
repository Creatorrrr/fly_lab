"""Generated tiny E295 localization/alignment/provenance tests; originals zero."""
import copy
import json
import time

import numpy as np
import pytest

from tools import analyze_c_banc_m3_core_resource_localization as probe

BEGAN = time.perf_counter()


def fixture():
    selection = {"premotor_ids": np.array([1, 2, 3], np.int64), "motor_ids": np.array([4], np.int64),
                 "source_ids": np.array([1, 2, 3], np.int64), "edge_ids": np.arange(4, dtype=np.int64),
                 "pre_ids": np.array([1, 1, 2, 3], np.int64), "post_ids": np.array([2, 3, 1, 2], np.int64),
                 "edge_source_slots": np.array([0, 0, 1, 2], np.int32), "weights32": np.array([2, 3, -4, 5], np.float32)}
    activity = {"observed_ids": np.array([1, 2, 3, 4], np.int64), "motor_ids": selection["motor_ids"].copy(),
                "source_ids": selection["source_ids"].copy(), "control_tick": np.array([0, 1], np.int64),
                "network_tick": np.array([0, 20], np.int64), "physics_time_s": np.array([0., .005]),
                "complete": np.ones(2, bool), "rate": np.array([[0, 0, 0, 0], [4, 2, 1, 7]], np.float32),
                "drive": np.zeros((2, 4), np.float32), "adaptation": np.zeros((2, 4), np.float32),
                "synaptic_resource": np.array([[1, 1, 1], [.5, 1, .25]], np.float64)}
    return selection, activity


def test_independent_hand_math_core_other_shared_source_unsorted_core_order():
    selection, activity = fixture()
    before_selection, before_activity = copy.deepcopy(selection), copy.deepcopy(activity)
    result = probe.core_decomposition(selection, activity, np.array([2, 1], np.int64))
    np.testing.assert_array_equal(result["core_source_Jpos"][1], [8, 0])
    np.testing.assert_array_equal(result["core_source_JposR"][1], [4, 0])
    np.testing.assert_array_equal(result["core_source_loss"][1], [4, 0])
    np.testing.assert_array_equal(result["other_source_Jpos"][1], [5, 0])
    np.testing.assert_array_equal(result["other_source_JposR"][1], [1.25, 0])
    np.testing.assert_array_equal(result["other_source_loss"][1], [3.75, 0])
    np.testing.assert_array_equal(result["core_source_Jneg"][1], [0, -8])
    for term in probe.TERMS:
        np.testing.assert_array_equal(result["all_selected_" + term], result["core_source_" + term] + result["other_source_" + term])
        assert result["all_selected_" + term].dtype == np.float64
    for group in probe.GROUPS:
        probe.prior.equal_array(result[group + "_Jneg"], result[group + "_JnegR"], "negative exact")
    for key in selection:
        probe.prior.equal_array(selection[key], before_selection[key], "input bytes")
    for key in activity:
        probe.prior.equal_array(activity[key], before_activity[key], "input bytes")


def test_empty_group_and_zero_loss_fraction_are_unavailable():
    selection, activity = fixture()
    result = probe.core_decomposition(selection, activity, selection["premotor_ids"])
    for term in probe.TERMS:
        assert not result["other_source_" + term].any()
    assert probe.loss_fraction(0., 0.) == {"available": False, "core_source_fraction": None, "other_source_fraction": None}
    assert probe.loss_fraction(3., 1.)["core_source_fraction"] == .75


@pytest.mark.parametrize("mutation", ["missing_core", "duplicate_core", "core_dtype", "resource_negative_source", "resource_nan", "dtype", "shape", "duplicate_id", "clock", "incomplete"])
def test_invalid_core_activity_boundary_fails_closed(mutation):
    selection, activity = fixture()
    core = np.array([1, 2], np.int64)
    if mutation == "missing_core":
        core[1] = 5
    elif mutation == "duplicate_core":
        core[1] = 1
    elif mutation == "core_dtype":
        core = core.astype(np.int32)
    elif mutation == "resource_negative_source":
        activity["synaptic_resource"][1, 1] = .9
    elif mutation == "resource_nan":
        activity["synaptic_resource"][1, 0] = np.nan
    elif mutation == "dtype":
        activity["rate"] = activity["rate"].astype(np.float64)
    elif mutation == "shape":
        activity["rate"] = activity["rate"][:, :-1]
    elif mutation == "duplicate_id":
        activity["observed_ids"][1] = 1
    elif mutation == "clock":
        activity["physics_time_s"][1] += 2.1e-8
    else:
        activity["complete"][1] = False
    with pytest.raises(ValueError):
        probe.core_decomposition(selection, activity, core)


def test_explicit_unsorted_baseline_id_map():
    np.testing.assert_array_equal(probe.id_columns(np.array([4, 1, 3, 2], np.int64), np.array([2, 1], np.int64)), [3, 1])


@pytest.mark.parametrize("observed,requested", [([4, 1, 1, 2], [2, 1]), ([4, 1, 3, 2], [2, 5]), ([4, 1, 3, 2], [1, 1])])
def test_unsorted_id_alignment_rejects_missing_or_duplicate(observed, requested):
    with pytest.raises(ValueError):
        probe.id_columns(np.asarray(observed, np.int64), np.asarray(requested, np.int64))


def test_field_and_all_cell_first_observed_difference_and_fixed_windows():
    baseline = np.zeros((100, 2), np.float32)
    candidate = baseline.copy()
    candidate[4:, 0] = 2
    candidate[19:, 1] = -1
    report, delta = probe.first_and_windows(baseline, candidate, np.arange(1, 101, dtype=np.int64), [7, 8])
    assert report["first_exact_difference_control"] == 5
    assert [cell["first_exact_difference_control"] for cell in report["all_cells"]] == [5, 20]
    assert report["all_cells"][0]["windows"][0]["mean_delta"] == 1.2
    assert report["all_cells"][1]["windows"][1]["samples"] == 10
    assert delta.dtype == np.float64 and len(report["all_cells"]) == 2


def test_first_exact_difference_includes_signed_zero_storage_change():
    baseline, candidate = np.zeros((100, 1)), np.zeros((100, 1))
    candidate[1, 0] = -0.
    report, delta = probe.first_and_windows(baseline, candidate, np.arange(1, 101, dtype=np.int64), [1])
    assert report["first_exact_difference_control"] == 2 and not delta.any()


@pytest.mark.parametrize("mutation", ["dtype", "incomplete", "duplicate_clock", "nan"])
def test_summary_invalid_pair_and_fixed_windows_rejected(mutation):
    baseline, candidate = np.zeros((100, 2), np.float32), np.zeros((100, 2), np.float32)
    controls = np.arange(1, 101, dtype=np.int64)
    if mutation == "dtype":
        candidate = candidate.astype(np.float64)
    elif mutation == "incomplete":
        baseline, candidate, controls = baseline[:-1], candidate[:-1], controls[:-1]
    elif mutation == "duplicate_clock":
        controls[1] = 1
    else:
        candidate[1, 0] = np.nan
    with pytest.raises(ValueError):
        probe.first_and_windows(baseline, candidate, controls, [1, 2])


def test_all_cell_group_windows_and_zero_denominator(monkeypatch):
    selection, activity = fixture()
    result = probe.core_decomposition(selection, activity, np.array([1, 2], np.int64))
    monkeypatch.setattr(probe, "WINDOWS", ((1, 1),))
    report = probe.decomposition_windows(result)
    assert len(report) == 2 and report[0]["core_id"] == 1
    assert not report[0]["windows"][0]["integrated_sample_loss_fraction"]["available"]
    assert report[1]["windows"][0]["components"]["core_source_loss"]["mean"] == 4


def test_sha_gate_fails_before_loading_any_data(tmp_path):
    file = tmp_path / "freeze.json"
    file.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        probe.gate(file, "0" * 64, file, "0" * 64, tmp_path / "out")


def generated_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    def put(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return probe.prior.sha256(path)
    sources = {name: put(name, {"source": name}) for name in probe.SOURCE_FILES}
    inputs = {name: "other/" + name + ".json" for name in probe.INPUT_NAMES}
    groups = (("baseline_e150_output_manifest", "e150", ("baseline_signals", "baseline_metadata", "baseline_recruitment", "baseline_trace")),
              ("baseline_e257_output_manifest", "e257", ("baseline_core_activity", "baseline_csr")),
              ("e294_output_manifest", "e294", ("candidate_activity", "candidate_native", "candidate_recruitment", "candidate_trace", "e294_ledger", "e294_source_manifest")))
    for manifest, directory, aliases in groups:
        inputs[manifest] = directory + "/output-manifest.json"
        for alias in aliases:
            inputs[alias] = directory + "/" + alias + ".json"
    hashes = {path: put(path, {}) for path in inputs.values()}
    original_sources = {name: sources[name] for name in probe.SOURCE_FILES[:2]}
    original_inputs = {inputs["positive_roster"]: hashes[inputs["positive_roster"]]}
    original_freeze = {"diagnostic_id": "E294", "input_sha256": original_inputs, "preservation_sha256": {},
                       "graph_hash": "graph", "source_network_identity": "network", "adapter_identity": "adapter"}
    hashes[inputs["e294_freeze"]] = put(inputs["e294_freeze"], original_freeze)
    hashes[inputs["e294_source_review"]] = put(inputs["e294_source_review"], {"approved": True, "blocking_defects": [],
        "freeze_sha256": hashes[inputs["e294_freeze"]], "source_sha256": original_sources})
    hashes[inputs["e294_source_manifest"]] = put(inputs["e294_source_manifest"], {
        "diagnostic_id": "E294", "owned_sha256": original_sources, "source_sha256": {}, "input_sha256": original_inputs})
    receipt = {"run_valid": True, "pass": True, **{key: False for key in ("actual_startup_pass", "short_gait_pass", "candidate_adopted", "F1_F8_complete", "F6_exposed")}}
    for name in ("e294_result", "e294_root_audit"):
        hashes[inputs[name]] = put(inputs[name], receipt)
    path_sources = {name: put(name, {"source": name}) for name in (
        "tools/verify_c_banc_m3_recurrent_resource_body.py", "tools/verify_c_banc_m3_positive_resource_body.py",
        "tools/banc_m3_recurrent_resource_trial.py", "tools/banc_m3_recurrent_resource_synapses.py", "flylab/c/banc_walking.py")}
    hashes[inputs["current_path_source_audit"]] = put(inputs["current_path_source_audit"], {"schema": "flylab.m3.e295.source-path-audit.v1",
        "approved": True, "no_runtime_mute": True, "checked_initial_cuts_all_false": True, "endpoint_masks_all_one": True, "source_sha256": path_sources})
    for manifest, directory, aliases in groups:
        entries = {inputs[name].removeprefix(directory + "/"): {"sha256": hashes[inputs[name]], "bytes": (tmp_path / inputs[name]).stat().st_size} for name in aliases}
        hashes[inputs[manifest]] = put(inputs[manifest], {"files": entries})
    design = "design.json"
    freeze = {"schema": probe.FREEZE_SCHEMA, "owned_files": list(probe.OWNED), "inputs": inputs, "input_sha256": hashes,
              "source_sha256": sources, "resource_limits": probe.LIMITS, "windows": [list(w) for w in probe.WINDOWS],
              "display_controls": list(probe.DISPLAY_CONTROLS), "output": probe.OUTPUT_PATH, "design": design, "design_sha256": put(design, {})}
    (tmp_path / probe.OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    def call():
        digest = put("freeze.json", freeze)
        review_sha = put("review.json", {"schema": probe.REVIEW_SCHEMA, "approved": True, "blocking_defects": [],
                    "freeze_sha256": digest, "source_sha256": {name: sources[name] for name in probe.OWNED}})
        return probe.gate(tmp_path / "freeze.json", digest, tmp_path / "review.json", review_sha, tmp_path / probe.OUTPUT_PATH)
    return freeze, call


def test_complete_generated_manifest_provenance_gate(tmp_path, monkeypatch):
    freeze, call = generated_gate(tmp_path, monkeypatch)
    assert call() == freeze


@pytest.mark.parametrize("mutation", ["input_hash", "producer_hash", "path_source", "incomplete_inputs", "output"])
def test_frozen_and_producer_manifest_tampering_rejected(tmp_path, monkeypatch, mutation):
    freeze, call = generated_gate(tmp_path, monkeypatch)
    if mutation == "input_hash":
        (tmp_path / freeze["inputs"]["candidate_activity"]).write_text("mutated")
    elif mutation == "producer_hash":
        path = tmp_path / freeze["inputs"]["baseline_e150_output_manifest"]
        manifest = json.loads(path.read_text())
        manifest["files"]["baseline_signals.json"]["sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))
        freeze["input_sha256"][freeze["inputs"]["baseline_e150_output_manifest"]] = probe.prior.sha256(path)
    elif mutation == "path_source":
        (tmp_path / "flylab/c/banc_walking.py").write_text("mutated")
    elif mutation == "incomplete_inputs":
        freeze["inputs"].pop("baseline_metadata")
    else:
        (tmp_path / probe.OUTPUT_PATH).mkdir()
    with pytest.raises(ValueError):
        call()


def test_generated_cost_receipt():
    elapsed = time.perf_counter() - BEGAN
    assert elapsed < 30
    print(json.dumps({"wall_s": elapsed, "tiny_cells": 4, "tiny_edges": 4,
                      "original": 0, "model": 0, "neural": 0, "body": 0, "physics": 0, "CUDA": 0, "fit": 0, "solver": 0}))
