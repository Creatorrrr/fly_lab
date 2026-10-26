"""Generated-only E153 signed-contact tests; no original recording is opened."""

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import analyze_c_banc_m3_signed_contact as e153


def test_implicit_requests_are_limited_to_complete_uncut_e150_protocol():
    import copy

    row = {
        "neural_request": None,
        "joint_targets_rad": [0.0] * 42,
        "adhesion": [False] * 6,
        "tendon_inputs": {},
    }
    payload = {
        "schema": "flylab.m3.e150.native-monotone-body-trace.v1",
        "E150_provenance": {"numeric_protocol": "unchanged E141 native body protocol"},
        "samples": [copy.deepcopy(row) for _ in range(2001)],
    }
    saved = {
        "cuts": dict.fromkeys(("circuit", "descending", "motor", "sensory"), False),
        "body": {"push_left": 0.0, "push_force": [0.0] * 3},
    }
    request = e153.implicit_request_for_trace(payload, saved)
    _, _, command = e153.replay_command(row, implicit_request=request)
    assert command["step_kwargs"]["interpolate"] is True
    assert command["push_left_s"] == 0.0
    assert np.array_equal(command["push_force"], np.zeros(3))
    assert row["neural_request"] is None
    with pytest.raises(ValueError, match="Complete recorded"):
        e153.replay_command(row)
    for mutation in ("schema", "mixed", "missing", "cut", "push"):
        changed, state = copy.deepcopy(payload), copy.deepcopy(saved)
        if mutation == "schema":
            changed["schema"] = "unknown"
        elif mutation == "mixed":
            changed["samples"][2]["neural_request"] = request
        elif mutation == "missing":
            del changed["samples"][2]["neural_request"]
        elif mutation == "cut":
            state["cuts"]["motor"] = True
        else:
            state["body"]["push_force"][0] = 1.0
        with pytest.raises(ValueError):
            e153.implicit_request_for_trace(changed, state)


def synthetic_contacts():
    mapping = np.full(8, -1, dtype=np.int64)
    mapping[2] = 5
    mapping[3] = 10
    frame = np.repeat(np.eye(3)[None], 5, axis=0)
    wrench = np.zeros((5, 6))
    wrench[:, 0] = [2, 2, 3, 7, 11]
    return {
        "geom1": np.array([0, 2, 0, 0, 4]),
        "geom2": np.array([2, 0, 3, 2, 5]),
        "excluded": np.array([False, False, False, True, False]),
        "frame": frame,
        "wrench": wrench,
        "position_world": np.arange(15, dtype=float).reshape(5, 3),
        "feet_mapping": mapping,
        "support_geom_ids": {0},
        "weight0": 2.0,
        "rotation_world": np.eye(3),
    }


def test_geom_order_sign_rotation_and_contact_order():
    values = synthetic_contacts()
    force, forward, raw = e153.transform_contacts(**values)
    assert raw["contact_id"].tolist() == [0, 1, 2]
    assert raw["segment"].tolist() == [5, 5, 10]
    assert raw["leg"].tolist() == [1, 1, 2]
    assert raw["sign"].tolist() == [1, -1, 1]
    assert np.array_equal(force[1], np.zeros(3))
    assert np.array_equal(force[2], [1.5, 0, 0])
    assert forward[2] == 1.5

    rotated = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
    observed = e153.contact_force_world(rotated, [2, 3, 0, 0, 0, 0], 1, 2)
    assert np.array_equal(observed, [1.5, 1.0, 0])
    assert np.array_equal(
        e153.contact_force_world(rotated, [2, 3, 0, 0, 0, 0], -1, 2),
        -observed,
    )


def test_horizontal_heading_uses_rotation_column_and_removes_tilt():
    yaw_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert np.allclose(e153.horizontal_heading(yaw_90), [0, 1, 0])
    angle = 0.4
    tilted = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    assert np.allclose(e153.horizontal_heading(tilted), [1, 0, 0])
    vertical_forward = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    with pytest.raises(ValueError, match="no horizontal projection"):
        e153.horizontal_heading(vertical_forward)


def test_fixed_50_to_1_groups_signed_impulses_and_cancellation():
    force = np.zeros((100, 6))
    force[:50, 0] = 1.0
    force[:50, 1] = -1.0
    force[50:, 0] = -2.0
    force[50:, 1] = 0.5
    load = np.full((100, 6), 0.1)
    slip = np.full((100, 6), 2.0)
    valid = np.ones((100, 6), dtype=bool)
    groups = e153.aggregate_signed_contacts(force, load, slip, valid, [1.0, 0.0])
    assert groups["forward"]["sample_count"] == 50
    assert groups["interrupted"]["sample_count"] == 50
    assert groups["forward"]["legs"]["LF"]["positive_impulse_bw_s"] == 0.005
    assert groups["forward"]["legs"]["LM"]["negative_impulse_bw_s"] == -0.005
    # Total is summed across legs before sign separation, so cancellation is exact.
    assert groups["forward"]["total"]["positive_impulse_bw_s"] == 0
    assert groups["forward"]["total"]["negative_impulse_bw_s"] == 0
    assert groups["interrupted"]["total"]["negative_impulse_bw_s"] == pytest.approx(
        -0.0075
    )
    assert groups["all"]["total"]["load_weighted_slip_mm_s"] == 2.0
    assert groups["all"]["total"]["supported_samples"] == 600
    assert groups["all"]["total"]["mean_normal_load_bw"] == pytest.approx(0.6)
    assert groups["all"]["total"]["mean_normal_load_bw_per_leg"] == 0.1
    assert (
        groups["all"]["total"]["supported_negative_fraction_denominator"]
        == "supported leg samples across six legs"
    )


def test_supported_missing_slip_rejected_but_unsupported_nan_allowed():
    force = np.zeros((2, 6))
    load = np.zeros((2, 6))
    slip = np.full((2, 6), np.nan)
    valid = np.zeros((2, 6), dtype=bool)
    result = e153.aggregate_force_group(
        np.ones(2, dtype=bool), force, load, slip, valid
    )
    assert result["legs"]["LF"]["load_weighted_slip_mm_s"] is None
    load[0, 0] = e153.SUPPORT_BW
    with pytest.raises(ValueError, match="Supported slip is missing"):
        e153.aggregate_force_group(np.ones(2, dtype=bool), force, load, slip, valid)


def test_empty_group_uses_null_not_zero():
    force = np.zeros((2, 6))
    load = np.zeros((2, 6))
    slip = np.full((2, 6), np.nan)
    valid = np.zeros((2, 6), dtype=bool)
    result = e153.aggregate_force_group(
        np.zeros(2, dtype=bool), force, load, slip, valid
    )
    assert result == {"sample_count": 0, "legs": None, "total": None}


def test_raw_offsets_fields_and_defensive_copies(tmp_path):
    registry = e153.new_raw_registry()
    values = synthetic_contacts()
    force, _, contacts = e153.transform_contacts(**values)
    rotation = np.eye(3)
    load = np.arange(6, dtype=float) / 10
    slip = np.arange(6, dtype=float)
    valid = np.ones(6, dtype=bool)
    e153.append_native_raw(
        registry,
        time_s=0.0001,
        control_tick=201,
        rotation_world=rotation,
        force_world_bw=force,
        normal_load_bw=load,
        slip_mm_s=slip,
        slip_valid=valid,
        nonfoot_contact=False,
        upright=1.0,
        contacts=contacts,
    )
    load[:] = 99
    rotation[:] = 0
    native, raw, _ = e153.raw_arrays(registry)
    assert raw["offsets"].tolist() == [0, 3]
    assert raw["geom1"].tolist() == [0, 2, 0]
    assert native["normal_load_bw"][0, 0] == 0
    assert np.array_equal(native["rotation_world"][0], np.eye(3))
    e153.save_raw(tmp_path, registry)
    assert (tmp_path / "native-forces.npz").exists()
    assert (tmp_path / "contacts.npz").exists()
    assert (tmp_path / "replay-controls.npz").exists()


def test_gate_requires_freeze_budget_and_both_current_sources(tmp_path):
    sources = {name: e153.sha256(e153.ROOT / name) for name in e153.OWNED}
    valid = {
        "accepts_single_body_diagnostic": True,
        "freeze_sha256": e153.FREEZE_SHA,
        "original_processes_allowed": 1,
        "source_sha256": sources,
    }
    gate = tmp_path / "gate.json"

    def write(value):
        gate.write_text(json.dumps(value), encoding="utf-8")
        return hashlib.sha256(gate.read_bytes()).hexdigest()

    assert e153.validate_source_gate(gate, write(valid)) == valid
    for key, value, message in (
        ("freeze_sha256", "wrong", "different freeze"),
        ("original_processes_allowed", 0, "exactly one original process"),
        ("source_sha256", {e153.OWNED[0]: sources[e153.OWNED[0]]}, "both owned files"),
    ):
        with pytest.raises(ValueError, match=message):
            e153.validate_source_gate(gate, write({**valid, key: value}))


def test_partial_body_is_closed_when_new_body_configuration_raises():
    class FakeBody:
        last = None

        def __init__(self, *_args, **_kwargs):
            FakeBody.last = self
            self.closed = False

        def _step_physics(self):
            return None

        def close(self):
            self.closed = True

    module = SimpleNamespace(FlyGymBody=FakeBody)

    counters = {
        "body_constructor_attempts": 0,
        "body_constructor_completed": 0,
        "settling_physics_steps": 0,
    }

    def new_body(_saved):
        module.FlyGymBody()
        raise RuntimeError("generated configuration failure")

    module.new_body = new_body
    with pytest.raises(RuntimeError, match="generated configuration failure"):
        e153.construct_body_safely({}, module, counters)
    assert FakeBody.last.closed is True
    assert module.FlyGymBody is FakeBody
    assert counters == {
        "body_constructor_attempts": 1,
        "body_constructor_completed": 1,
        "settling_physics_steps": 0,
    }


def test_constructor_and_scientific_stage_counters_precede_failures():
    class FailingSetupBody:
        last = None

        def __init__(self, *_args, **_kwargs):
            FailingSetupBody.last = self
            self.closed = False
            for _ in range(3):
                self._step_physics()
            raise RuntimeError("setup failed")

        def _step_physics(self):
            return None

        def close(self):
            self.closed = True

    module = SimpleNamespace(FlyGymBody=FailingSetupBody)
    module.new_body = lambda _saved: module.FlyGymBody()
    counters = {
        "body_constructor_attempts": 0,
        "body_constructor_completed": 0,
        "settling_physics_steps": 0,
    }
    with pytest.raises(RuntimeError, match="setup failed"):
        e153.construct_body_safely({}, module, counters)
    assert counters == {
        "body_constructor_attempts": 1,
        "body_constructor_completed": 0,
        "settling_physics_steps": 3,
    }
    assert FailingSetupBody.last.closed is True
    assert module.FlyGymBody is FailingSetupBody

    stages = {
        "scientific_physics_steps": 0,
        "scientific_physics_steps_advanced": 0,
        "scientific_samples_observed": 0,
        "scientific_samples_verified": 0,
    }
    actual_steps = []
    e153.advance_scientific_step(lambda: actual_steps.append(1), stages)
    e153.mark_scientific_observed(stages)
    assert len(actual_steps) == 1
    assert stages["scientific_physics_steps"] == 1
    assert stages["scientific_physics_steps_advanced"] == 1
    assert stages["scientific_samples_observed"] == 1
    assert stages["scientific_samples_verified"] == 0


def test_settling_count_and_recorded_command_contract():
    assert e153.validate_settling_count(0.2, 0.0001, 2000) == 2000
    with pytest.raises(ValueError, match="settling count"):
        e153.validate_settling_count(0.1999, 0.0001, 2000)
    row = {
        "joint_targets_rad": np.arange(42, dtype=float),
        "adhesion": np.array([True, False, True, False, True, False]),
        "tendon_inputs": {"lf_tarsus": 0.1},
        "neural_request": {
            "external_force_at_start": {"push_left_s": 0.2, "push_force": [1, 2, 3]},
            "cuts": {"motor": True},
        },
    }
    joint, adhesion, command = e153.replay_command(row)
    assert np.array_equal(joint, np.arange(42))
    assert adhesion.tolist() == row["adhesion"].tolist()
    assert command["step_kwargs"]["interpolate"] is False
    assert np.array_equal(command["push_force"], [1, 2, 3])
    row["tendon_inputs"]["lf_tarsus"] = 9
    assert command["step_kwargs"]["tendon_inputs"]["lf_tarsus"] == 0.1


def test_body_parameter_validation_rejects_silent_new_mode():
    saved = {
        "parameters": {
            "continuous_joint_activation": False,
            "active_length": False,
            "adhesion_model": "legacy",
        }
    }
    assert e153.validate_body_parameters(saved)["graded_adhesion"] is False
    saved["parameters"]["adhesion_model"] = "unimplemented"
    with pytest.raises(ValueError, match="Unsupported"):
        e153.validate_body_parameters(saved)


def test_direct_cli_help_from_outside_repository(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(e153.ROOT / e153.OWNED[0]), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--freeze" in completed.stdout
    assert "--source-review" in completed.stdout
    assert "--output" in completed.stdout
