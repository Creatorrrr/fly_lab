"""Generated, CPU-only contracts for the E177 native integration."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tools import banc_m3_load_feedback as feedback
from tools import verify_c_banc_m3_native_feedback as e177


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _freeze():
    return {
        "patch": {
            "edge_ids": [844054, 2400285, 7908675, 8578714, 11240394],
            "pre_ids": [107685, 30981, 107685, 30981, 30981],
            "post_ids": [12036, 32830, 108052, 117483, 153939],
            "source_values": [3e-5, 6e-5, 3e-5, 6e-5, 3e-5],
            "target_values": [0.03, 0.06, 0.03, 0.06, 0.03],
            "value_dtype": "<f4",
        }
    }


def test_arm_patch_and_full_csr_bytes():
    patch0 = e177.arm_patch(_freeze(), "baseline")
    patch1 = e177.arm_patch(_freeze(), "candidate")
    assert np.array_equal(patch0["source_values"], patch0["target_values"])
    assert patch1["target_values"].dtype == np.float32
    n = 11240395
    before = {
        "shape": (2, 2),
        "indptr": "same",
        "indices": "same",
        "data": np.zeros(n, np.float32),
    }
    before["data"][patch1["edge_ids"]] = patch1["source_values"]
    after = {**before, "data": before["data"].copy()}
    after["data"][patch1["edge_ids"]] = patch1["target_values"]
    assert e177.verify_csr_patch(before, after, patch1, "candidate")[
        "only_selected_five"
    ]
    after["data"][0] = 1
    with pytest.raises(ValueError, match="Non-selected"):
        e177.verify_csr_patch(before, after, patch1, "candidate")
    with pytest.raises(ValueError, match="Non-selected"):
        e177.verify_csr_patch(
            before, {**before, "data": after["data"]}, patch0, "baseline"
        )


def test_isolated_five_overrides_do_not_touch_protocol_globals():
    original = e177.protocol._native_body.__globals__
    factory = object()
    observer = object()
    saver = object()
    diagnostic = object()
    clone = e177.isolated_native_body(factory, observer, saver, diagnostic)
    assert clone.__code__ is e177.protocol._native_body.__code__
    assert clone.__defaults__ == e177.protocol._native_body.__defaults__
    assert clone.__closure__ is e177.protocol._native_body.__closure__
    assert clone.e177_overrides == e177.OVERRIDES
    assert [clone.__globals__[key] for key in e177.OVERRIDES] == [
        factory,
        observer,
        saver,
        250.0,
        diagnostic,
    ]
    assert original["MotorTimescaleTrial"] is not factory
    assert original["source_diagnostics"] is not diagnostic


def test_observer_one_current_masked_edges_and_partial_save(tmp_path, monkeypatch):
    calls = []
    ids = np.asarray(e177.FEEDBACK_IDS)
    n = max(ids) + 1
    rate = torch.zeros(n)
    rate[ids] = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    mask = torch.ones(n)
    mask[ids[1]] = 0.5
    drive = torch.arange(n, dtype=torch.float32)
    adaptation = torch.zeros(n)
    edge_ids = feedback.fixed_rm_load_patch()["edge_ids"]
    weights = torch.zeros(int(edge_ids.max()) + 1)
    weights[edge_ids] = torch.tensor([0.03, 0.06, 0.03, 0.06, 0.03])

    def current(_rate):
        calls.append(1)
        return _rate * 9

    net = SimpleNamespace(
        rate=rate,
        output_mask=mask,
        drive=drive,
        adaptation=adaptation,
        weights=SimpleNamespace(values=lambda: weights),
        _current=current,
        tick=20,
        readout=lambda selected: rate[selected].numpy(),
    )
    adapter = SimpleNamespace(
        bank=np.zeros((391, 3)),
        activation=np.zeros(96),
        original=SimpleNamespace(activation=np.zeros(96)),
        calibration={"motor_ids": np.arange(391, dtype=np.int64)},
    )
    body = SimpleNamespace(physics_time=lambda: 0.005)
    session = SimpleNamespace(control_tick=1, network=net, adapter=adapter, body=body)
    monkeypatch.setattr(
        e177,
        "contact_diagnostics",
        lambda _body: {"normal_load_bw": [0, 0, 0, 0, 1.25, 0]},
    )
    native, feedback_rows = [], []
    fields = e177.make_observer(native, feedback_rows)(session, np.array([0, 1]))
    assert len(calls) == 1
    assert fields["recurrent_current"].tolist() == [0.0, 0.0]
    assert feedback_rows[0]["direct_edge_current"][0] == pytest.approx(0.03 * 3 * 0.5)
    assert feedback_rows[0]["direct_edge_current"][1] == pytest.approx(0.06 * 2)
    assert feedback_rows[0]["rm_normal_load_bw"] == 1.25
    assert feedback_rows[0]["complete"] and native[0]["complete"]
    feedback_rows.append(
        {
            "control_tick": 2,
            "network_tick": 40,
            "rate": np.ones(7, np.float32),
            "error": "generated failure",
            "complete": False,
        }
    )
    path, receipt = tmp_path / "state.npz", tmp_path / "receipt.json"
    e177.save_feedback_state(path, receipt, feedback_rows)
    with np.load(path, allow_pickle=False) as raw:
        assert raw["rate"].shape == (2, 7)
        assert raw["edge_weight_available"].tolist() == [True, False]
        assert raw["complete"].tolist() == [True, False]
    assert json.loads(receipt.read_text())["partial"][0]["error"] == "generated failure"


def _write_state(path, clock=(0.0, 0.005), *, rates=None):
    controls = len(clock)
    fields = {
        "schema": np.array("fixture"),
        "control_tick": np.arange(controls, dtype=np.int64),
        "network_tick": np.arange(controls, dtype=np.int64) * 20,
        "physics_time_s": np.asarray(clock, np.float64),
        "complete": np.ones(controls, bool),
    }
    for key, shape in (
        ("bank", (391, 3)),
        ("activation", (96,)),
        ("original_activation", (96,)),
        ("motor_rate", (391,)),
    ):
        fields[key] = np.zeros((controls, *shape), np.float64)
        fields[key + "_available"] = np.ones(controls, bool)
    if rates is not None:
        fields["motor_rate"][0, 0] = rates
    np.savez_compressed(path, **fields)


def test_reproduction_physical_clock_only_tolerance(tmp_path):
    ref, got = tmp_path / "ref", tmp_path / "got"
    ref.mkdir()
    got.mkdir()
    _write_state(ref / "native-recruitment-state.npz")
    _write_state(got / "native-recruitment-state.npz", (1e-8, 0.005 + 1e-8))
    sample = {
        "control_tick": 0,
        "time_s": 0.0,
        "command_interval": [0.0],
        "joint_targets_rad": [0.0],
        "tendon_inputs": {"x": 0.0},
        "qpos": [0.0],
        "qvel": [0.0],
        "ctrl": [0.0],
        "body": {"position": [0.0], "basis": [1.0], "contactsBW": [0.0]},
        "upright": True,
    }
    samples = [copy.deepcopy(sample), copy.deepcopy(sample)]
    samples[1]["control_tick"] = 1
    samples[1]["time_s"] = 0.005
    for folder in (ref, got):
        (folder / "trace.json").write_text(json.dumps({"samples": samples}))
    freeze = {
        "baseline_reproduction": {
            "rows": 2,
            "physics_time_atol_s": 2e-8,
            "state_exact": [
                "control_tick",
                "network_tick",
                "complete",
                "bank",
                "bank_available",
                "activation",
                "activation_available",
                "original_activation",
                "original_activation_available",
                "motor_rate",
                "motor_rate_available",
            ],
            "trace_exact": [
                "control_tick",
                "time_s",
                "command_interval",
                "joint_targets_rad",
                "tendon_inputs",
                "qpos",
                "qvel",
                "ctrl",
                "body.position",
                "body.basis",
                "body.contactsBW",
                "upright",
            ],
        }
    }
    assert e177.compare_baseline_reproduction(ref, got, freeze)["exact"]
    _write_state(got / "native-recruitment-state.npz", (3e-8, 0.005))
    with pytest.raises(ValueError, match="physical time"):
        e177.compare_baseline_reproduction(ref, got, freeze)
    _write_state(got / "native-recruitment-state.npz")
    bad = copy.deepcopy(samples)
    bad[1]["body"]["position"][0] = -0.0
    (got / "trace.json").write_text(json.dumps({"samples": bad}))
    with pytest.raises(ValueError, match="float bits"):
        e177.compare_baseline_reproduction(ref, got, freeze)


def test_baseline_eligibility_startup_cut_and_reproduction():
    body = {
        "validity": "valid_full_BANC_native_transfer_diagnostic",
        "technical_failure": False,
        "stop_reason": "duration",
        "measured_health_stop": False,
        "clock_valid": True,
        "main_completed_controls": 2000,
        "main_native_samples": 100000,
        "clone_native_samples": 1000,
        "actual_neural_s": 10.1,
        "actual_physical_s": 10.1,
        "raw_persistence_errors": [],
        "analysis_errors": [],
        "strict_startup": {"strict_E141_startup_screen": True},
        "quality_summary": {
            "development_screens": {"F4_numeric": False, "F5_numeric": False}
        },
    }
    result = {
        "arm": "baseline",
        "technical_failure": False,
        "resource_limit_met": True,
        "body": body,
        "cut_restart": {"status": "completed", "accepted": True},
        "baseline_reproduction": {"exact": True},
    }
    assert e177.baseline_eligible(result)
    for key, value in (("strict_E141_startup_screen", False),):
        copy_result = copy.deepcopy(result)
        copy_result["body"]["strict_startup"][key] = value
        assert not e177.baseline_eligible(copy_result)
    result["baseline_reproduction"]["exact"] = False
    assert not e177.baseline_eligible(result)


def test_source_and_baseline_gate_reject_changed_bytes_before_models(
    tmp_path, monkeypatch
):
    # Construct a small stand-in freeze, preserving the exact gate ordering.
    root = tmp_path / "repo"
    root.mkdir()
    owned = {name: root / name for name in e177.OWNED}
    for path in owned.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"reviewed")
    design = root / "design.md"
    design.write_bytes(b"design")
    frozen = {
        "schema": "flylab.m3.e177.native-feedback.v1",
        "design_path": "design.md",
        "design_sha256": _sha(design),
        "owned_files": list(e177.OWNED),
        "allowed_protocol_global_overrides": list(e177.OVERRIDES),
        "body_protocol": "E141-code-with-isolated-factory-and-scoped-diagnostics",
        "arms": ["baseline", "candidate"],
        "prepare_wall_limit_s": 60,
        "body_wall_limit_s": 180,
        "internal_wall_limit_s": 250,
        "external_wall_limit_s": 270,
        "supervisor_limit_s": 285,
        "maximum_raw_bytes": 4 << 30,
        "minimum_free_disk_bytes": 12 << 30,
        "original_processes": 2,
        "automatic_retries": 0,
        "prior_conservative_actual_s": 3085.0114084999077,
        "prior_conservative_process_s": 3086,
        "prospective_cumulative_limit_s": 3656,
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "input_sha256": {str(i): "x" for i in range(102)},
        "reused_source_sha256": {str(i): "x" for i in range(90)},
        "preservation_sha256": {str(i): "x" for i in range(2013)},
        **_freeze(),
    }
    freeze_path = root / "freeze.json"
    freeze_path.write_text(json.dumps(frozen))
    review = {
        "schema": "flylab.m3.e177.parent-source-review.v1",
        "accepts_single_native_feedback": True,
        "blocking_defects": [],
        "freeze_sha256": _sha(freeze_path),
        "source_sha256": {name: _sha(path) for name, path in owned.items()},
    }
    review_path = root / "review.json"
    review_path.write_text(json.dumps(review))
    monkeypatch.setattr(e177, "ROOT", root)
    monkeypatch.setattr(e177, "FREEZE_SHA", _sha(freeze_path))
    monkeypatch.setattr(e177, "DESIGN_SHA", _sha(design))
    checked = []
    monkeypatch.setattr(
        e177.protocol,
        "_exact_mapping_hashes",
        lambda mapping, label: checked.append(label),
    )
    monkeypatch.setattr(
        e177.protocol.StateStore,
        "load",
        lambda *_: pytest.fail("model/values entered before gate"),
    )
    args = (freeze_path, _sha(freeze_path), review_path, _sha(review_path))
    e177.verified_inputs(*args, arm="baseline")
    assert checked == ["input_sha256", "reused_source_sha256", "preservation_sha256"]
    owned[e177.OWNED[0]].write_bytes(b"altered")
    with pytest.raises(ValueError, match="source changed"):
        e177.verified_inputs(*args, arm="baseline")
    with pytest.raises(ValueError, match="source changed"):
        e177.verified_inputs(
            *args,
            arm="candidate",
            baseline_review_path=root / "permit.json",
            baseline_review_sha256="invalid",
        )


def _complete_body():
    return {
        "validity": "valid_full_BANC_native_transfer_diagnostic",
        "technical_failure": False,
        "stop_reason": "duration",
        "measured_health_stop": False,
        "clock_valid": True,
        "main_completed_controls": 2000,
        "main_native_samples": 100000,
        "clone_native_samples": 1000,
        "actual_neural_s": 10.1,
        "actual_physical_s": 10.1,
        "raw_persistence_errors": [],
        "analysis_errors": [],
        "strict_startup": {"strict_E141_startup_screen": True},
        "hypothesis_verdict": "supported_development_diagnostic",
        "supported_slip": {"available": True, "load_weighted_mm_s": 3.0},
    }


def test_combined_outcome_keeps_measured_failures_distinct():
    body = _complete_body()
    cut = {"status": "completed", "clock_valid": True, "accepted": True}
    baseline_slip = {"available": True, "load_weighted_mm_s": 2.0}
    validity, verdict, technical, comparison = e177.combined_outcome(
        body, cut, "candidate", baseline_slip
    )
    assert (validity, verdict, technical) == (
        "valid_native_feedback_trial",
        "refuted_supported_slip",
        False,
    )
    assert comparison["candidate_no_worse"] is False
    body["strict_startup"]["strict_E141_startup_screen"] = False
    body["hypothesis_verdict"] = "refuted_numeric_or_startup"
    assert e177.combined_outcome(body, cut, "candidate", baseline_slip)[:3] == (
        "valid_native_feedback_trial",
        "refuted_numeric_or_startup",
        False,
    )
    body.update(
        validity="measured_native_health_stop",
        stop_reason="native_health_guard",
        measured_health_stop=True,
        hypothesis_verdict="refuted_health_stop",
    )
    assert e177.combined_outcome(
        body, {"status": "not_run_main_incomplete"}, "candidate", baseline_slip
    )[:3] == ("measured_native_health_stop", "refuted_health_stop", False)
    body = _complete_body()
    assert (
        e177.combined_outcome(body, cut, "candidate", None)[1]
        == "indeterminate_missing_slip_evidence"
    )
    cut["accepted"] = False
    assert e177.combined_outcome(body, cut, "candidate", baseline_slip)[:3] == (
        "valid_native_feedback_trial",
        "refuted_motor_cut_restart",
        False,
    )


def test_returned_wrapper_is_closed_on_postrestore_rejection(monkeypatch):
    closed = []

    class Returned:
        closed = False

        def snapshot(self):
            raise RuntimeError("generated postreturn failure")

        def close(self):
            self.closed = True
            closed.append(1)

    trial = Returned()
    patch_data = e177.arm_patch(_freeze(), "baseline")
    saved = {
        "schema": feedback.TRIAL_SCHEMA,
        "load_feedback_patch": {"edge_patch": patch_data},
    }
    monkeypatch.setattr(feedback.LoadFeedbackTrial, "from_checkpoint", lambda *_: trial)
    with pytest.raises(RuntimeError, match="postreturn"):
        e177.TrialFactory(patch_data).from_checkpoint(None, saved)
    assert closed == [1]


def test_frozen_assembled_bundle_reaches_generated_construction(tmp_path, monkeypatch):
    real = json.loads(
        (
            e177.ROOT / "verification/banc-natural-gait-m3-20260919/"
            "e177-native-feedback-freeze.json"
        ).read_text(encoding="utf-8")
    )
    freeze = copy.deepcopy(real)
    freeze["reference_directory"] = "reference"
    freeze["zero_checkpoint"] = "zero"
    freeze["graph_path"] = "graph"
    freeze["observation_path"] = "observation"
    freeze["startup_path"] = "startup"
    freeze["graph_hash"] = "generated"
    header = real["input_headers"][
        real["reference_directory"] + "/assembled-parameters.npz"
    ]
    freeze["input_headers"] = {"reference/assembled-parameters.npz": header}
    ref = tmp_path / "reference"
    ref.mkdir()
    for name in (
        "assembled-parameters.npz",
        "assembled-calibration.npz",
        "actual-global-csr.npz",
        "muscle-provenance.json",
    ):
        (ref / name).write_bytes(b"generated")
    assembled = {
        key: np.zeros(meta["shape"], np.dtype(meta["dtype"]))
        for key, meta in header.items()
    }
    monkeypatch.setattr(e177, "ROOT", tmp_path)
    monkeypatch.setattr(
        e177.protocol.StateStore, "load", lambda *_: {"schema": "generated"}
    )
    monkeypatch.setattr(
        e177.protocol.GraphStore,
        "load",
        lambda *_: SimpleNamespace(hash="generated", n=158706),
    )

    def load_npz(path):
        if Path(path).name == "assembled-parameters.npz":
            return copy.deepcopy(assembled)
        return {"motor_ids": assembled["motor_ids"].copy()}

    monkeypatch.setattr(e177.protocol, "load_npz", load_npz)
    monkeypatch.setattr(
        e177.protocol, "accounted_operation", lambda _label, callback, *_: callback()
    )

    class Reached(Exception):
        pass

    monkeypatch.setattr(
        e177.protocol,
        "profiled_construction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(Reached),
    )
    with pytest.raises(Reached):
        e177._prepare(
            freeze,
            "baseline",
            tmp_path,
            {"storage_operations": [], "construction_attempts": []},
            lambda: None,
        )


def test_preflight_finalization_error_reaches_caller(tmp_path, monkeypatch):
    freeze = {"output_directories": {"baseline": "unused"}}
    monkeypatch.setattr(e177, "ROOT", tmp_path)
    with (
        patch.object(e177, "verified_inputs", return_value=(freeze, {}, None)),
        patch.object(
            e177.protocol,
            "_persist_manifest",
            side_effect=OSError("generated manifest fault"),
        ),
        pytest.raises(OSError, match="manifest fault"),
    ):
        e177.run(
            "generated",
            "generated",
            "generated",
            "generated",
            "baseline",
            tmp_path / "preflight",
            preflight_only=True,
        )
    preliminary = json.loads((tmp_path / "preflight" / "result.json").read_text())
    assert preliminary["technical_failure"] is True


@pytest.mark.parametrize("partial_clock", [True, False])
def test_failed_cut_retains_known_main_and_partial_cost(
    tmp_path, monkeypatch, partial_clock
):
    freeze = {
        "output_directories": {"baseline": "run"},
        "minimum_free_disk_bytes": 0,
        "prepare_wall_limit_s": 60,
        "body_protocol": "generated",
    }
    out = tmp_path / "run"
    body = {
        "actual_neural_s": 10.1,
        "actual_physical_s": 10.1,
        "clock_valid": True,
        "body_analysis_wall_s": 1.0,
    }
    prepared = {
        key: None
        for key in (
            "graph",
            "cal",
            "observation",
            "startup",
            "neutral",
            "parameters",
            "anatomy",
            "factory",
        )
    }

    def native(*_args):
        (out / "checkpoint-1s").mkdir()
        return body

    native.e177_overrides = e177.OVERRIDES

    def cut(*_args):
        if partial_clock:
            (out / "cut-restart-ledger.json").write_text(
                json.dumps(
                    {"clock": {"neural_s": 0.015, "physical_s": 0.015, "valid": True}}
                )
            )
        raise RuntimeError("generated cut advance fault")

    monkeypatch.setattr(e177, "ROOT", tmp_path)
    monkeypatch.setattr(e177, "verified_inputs", lambda *_a, **_k: (freeze, {}, None))
    monkeypatch.setattr(e177, "_prepare", lambda *_a: prepared)
    monkeypatch.setattr(e177, "isolated_native_body", lambda *_a: native)
    monkeypatch.setattr(e177, "resource_checks", lambda *_a, **_k: [])
    monkeypatch.setattr(e177.e150, "full_main_complete", lambda _b: True)
    monkeypatch.setattr(e177.e150, "run_cut_restart", cut)
    monkeypatch.setattr(e177.protocol, "cleanup_cuda_runtime", lambda *_a: [])
    monkeypatch.setattr(
        e177.protocol.joint_fit,
        "one_cpu_thread",
        lambda: (SimpleNamespace(restore_original_limits=lambda: None), 1),
    )
    with pytest.raises(RuntimeError, match="generated cut advance fault"):
        e177.run("generated", "generated", "generated", "generated", "baseline", out)
    result = json.loads((out / "result.json").read_text())
    assert result["technical_failure"] is True
    assert (
        result["scientific_cost_by_stage"]["main_and_clone"]["actual_neural_s"] == 10.1
    )
    expected = 10.115 if partial_clock else None
    assert result["actual_scientific_neural_s"] == expected
    assert result["actual_scientific_physical_s"] == expected
