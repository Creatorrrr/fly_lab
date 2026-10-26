"""Generated-only E155 tests; no original checkpoint, NPZ, model, or physics is opened."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import analyze_c_banc_m3_rm_contact_jacobian as e155


def test_rotated_frame_moving_support_and_all_dof_groups_close():
    nv = 16
    qvel = np.arange(1, nv + 1, dtype=float) / 10
    angle = 0.37
    normal = np.array([np.sin(angle), 0.0, np.cos(angle)])
    tangent = np.array([np.cos(angle), 0.0, -np.sin(angle)])
    frame = np.vstack((normal, [0.0, 1.0, 0.0], tangent))
    jfoot = np.zeros((3, nv))
    jsupport = np.zeros((3, nv))
    jfoot[0] = np.linspace(-0.2, 0.4, nv)
    jfoot[1] = np.linspace(0.3, -0.1, nv)
    jsupport[0] = np.linspace(0.05, -0.15, nv)
    jsupport[2] = np.linspace(-0.3, 0.2, nv)
    heading = np.array([1.0, 0.0, 0.0])
    result = e155.decompose_contact(
        frame, qvel, jfoot, jsupport, heading, np.arange(6), np.arange(6, 13)
    )
    expected_tangent = (np.eye(3) - np.outer(normal, normal)) @ (jfoot - jsupport)
    expected_ap = (heading @ expected_tangent) * qvel
    np.testing.assert_allclose(result["tangent_jacobian"], expected_tangent)
    np.testing.assert_allclose(result["ap_dof"], expected_ap)
    assert result["ap_root"] == pytest.approx(expected_ap[:6].sum())
    np.testing.assert_allclose(result["ap_rm7"], expected_ap[6:13])
    assert result["ap_residual"] == pytest.approx(expected_ap[13:].sum())
    assert result["ap_total"] == pytest.approx(heading @ result["tangent_velocity"])
    assert (
        max(result["closure_error"], result["direct_error"], result["projection_error"])
        < 1e-14
    )


def test_support_jacobian_is_subtracted_and_inputs_are_not_mutated():
    frame = np.eye(3)
    qvel = np.ones(14)
    foot = np.zeros((3, 14))
    support = np.zeros((3, 14))
    foot[1, 0] = 4
    support[1, 0] = 1.5
    original = support.copy()
    result = e155.decompose_contact(
        frame, qvel, foot, support, [0, 1, 0], np.arange(6), np.arange(6, 13)
    )
    assert result["ap_dof"][0] == 2.5
    assert result["slip_speed"] == 2.5
    assert np.array_equal(support, original)


def test_weighted_slip_uses_contact_speed_and_support_threshold_only_for_validity():
    result = e155.combine_contact_decompositions(
        [0.01, 0.02],
        [2.0, 8.0],
        [-1.0, 3.0],
        [0.0, 1.0],
        np.arange(14, dtype=float).reshape(2, 7),
        [-1.0, 2.0],
    )
    assert result["normal_load"] == pytest.approx(0.03)
    assert result["slip"] == pytest.approx(6.0)
    assert result["slip"] != pytest.approx(np.mean([2.0, 8.0]))
    assert result["slip_valid"] is True
    assert result["ap_numerator"] == pytest.approx(0.05)
    unsupported = e155.combine_contact_decompositions(
        [0.019], [7.0], [1.0], [0.2], np.zeros((1, 7)), [0.8]
    )
    assert unsupported["normal_load"] == 0.019
    assert np.isnan(unsupported["slip"])
    assert unsupported["slip_valid"] is False
    assert unsupported["ap_numerator"] == pytest.approx(0.019)


def generated_controls() -> dict[str, np.ndarray]:
    return {
        "tick": np.array([201, 202, 203, 204]),
        "endpoint_speed_mm_s": np.array([1.0, 0.8, 0.2, 0.1]),
        "original_normal_load_bw": np.array([0.01, 0.03, 0.04, 0.0]),
        "original_force_forward_bw": np.array([0.2, -0.3, 0.4, 0.0]),
        "normal_load_denominator": np.array([0.01, 0.03, 0.04, 0.0]),
        "ap_numerator": np.array([0.01, -0.06, 0.12, 0.0]),
        "dof_numerator": np.arange(64, dtype=float).reshape(4, 16) / 100,
        "root_numerator": np.array([0.0, -0.03, 0.04, 0.0]),
        "rm7_numerator": np.arange(28, dtype=float).reshape(4, 7) / 100,
        "residual_numerator": np.array([0.01, -0.03, 0.08, 0.0]),
    }


def test_fixed_groups_use_all_positive_load_and_supported_force_masks():
    controls = generated_controls()
    groups = e155.aggregate_groups(controls)
    assert tuple(groups) == e155.GROUP_NAMES
    assert groups["all"]["sample_count"] == 4
    assert groups["all"]["supported_samples"] == 2
    assert groups["all"]["normal_load_denominator"] == pytest.approx(0.08)
    assert groups["all"]["weighted_contributions"][
        "all_dof_forward_velocity"
    ] == pytest.approx(0.875)
    assert groups["forward"]["sample_count"] == 2
    assert groups["interrupted"]["sample_count"] == 2
    assert groups["supported_RM_braking"]["sample_count"] == 1
    assert groups["supported_RM_propulsion"]["sample_count"] == 1


def test_empty_group_and_zero_denominator_publish_null_contributions():
    controls = generated_controls()
    empty = e155.aggregate_group(np.zeros(4, dtype=bool), controls)
    assert empty["sample_count"] == 0
    assert empty["supported_samples"] == 0
    assert empty["normal_load_denominator"] == 0.0
    assert empty["numerator_sums"]["rm7_forward_velocity"] == [0.0] * 7
    assert empty["numerator_sums"]["per_dof_forward_velocity"] == [0.0] * 16
    assert empty["weighted_contributions"] is None
    zero = e155.aggregate_group(np.array([False, False, False, True]), controls)
    assert zero["sample_count"] == 1
    assert zero["weighted_contributions"] is None


def test_contact_classification_enforces_order_sign_and_rm_tarsus2_or3():
    mapping = np.full(10, -1, dtype=np.int64)
    mapping[7] = 21
    assert e155.classify_saved_rm_contact(0, 7, 21, 4, 1, mapping, {0}) == (7, 0)
    assert e155.classify_saved_rm_contact(7, 0, 21, 4, -1, mapping, {0}) == (7, 0)
    with pytest.raises(ValueError, match="tarsus2/3"):
        e155.classify_saved_rm_contact(0, 7, 20, 4, 1, mapping, {0})
    with pytest.raises(ValueError, match="support geometry"):
        e155.classify_saved_rm_contact(3, 7, 21, 4, 1, mapping, {0})


def trace_payload() -> dict:
    samples = []
    for tick in range(401):
        samples.append(
            {
                "control_tick": tick,
                "time_s": tick * 0.005,
                "qpos": [0.0, 1.0],
                "qvel": [0.0],
                "ctrl": [0.0, 0.0, 0.0],
                "joint_targets_rad": [0.0] * 42,
                "adhesion": [False] * 6,
                "tendon_inputs": {},
                "neural_request": None,
            }
        )
    return {
        "schema": "flylab.m3.e150.native-monotone-body-trace.v1",
        "body_hash": "body",
        "samples": samples,
    }


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("tick", "ticks"),
        ("clock", "clock"),
        ("nan", "invalid qvel"),
        ("request", "null neural requests"),
        ("joint", "invalid joint command"),
    ],
)
def test_all_commands_and_poses_are_validated_before_model(mutation, message):
    payload = trace_payload()
    row = payload["samples"][250]
    if mutation == "tick":
        row["control_tick"] = 999
    elif mutation == "clock":
        row["time_s"] += 0.1
    elif mutation == "nan":
        row["qvel"][0] = np.nan
    elif mutation == "request":
        row["neural_request"] = {}
    else:
        row["joint_targets_rad"] = [0.0]
    with pytest.raises(ValueError, match=message):
        e155.validate_recorded_inputs(payload, "body")


def test_source_endpoint_offsets_and_feature_clock_are_exact():
    rows, shapes = e155.validate_recorded_inputs(trace_payload(), "body")
    indices = 49 + 50 * np.arange(200)
    native = {
        "time_s": np.zeros(10000),
        "rotation_world": np.zeros((10000, 3, 3)),
        "heading_horizontal": np.zeros((10000, 3)),
        "force_forward_bw": np.zeros((10000, 6)),
        "normal_load_bw": np.zeros((10000, 6)),
        "slip_mm_s": np.full((10000, 6), np.nan),
        "slip_valid": np.zeros((10000, 6), dtype=bool),
    }
    accumulated = 1.0 + np.cumsum(np.full(10000, 0.0001))
    native["time_s"][indices] = accumulated[indices]
    features = {"time_s": np.arange(500) * 0.005, "speed_mm_s": np.ones(500)}
    contacts = {"offsets": np.zeros(10001, dtype=np.int64)}
    for key in ("contact_id", "geom1", "geom2", "segment", "leg", "sign"):
        contacts[key] = np.empty(0, dtype=np.int64)
    contacts.update(
        frame=np.empty((0, 3, 3)),
        wrench=np.empty((0, 6)),
        position_world=np.empty((0, 3)),
    )
    assert np.array_equal(
        e155.validate_source_arrays(rows, shapes, native, contacts, features), indices
    )
    native["time_s"][indices[20]] += 3e-8
    with pytest.raises(ValueError, match="endpoint clocks"):
        e155.validate_source_arrays(rows, shapes, native, contacts, features)
    native["time_s"][indices[20]] -= 3e-8
    contacts["offsets"] = contacts["offsets"][:-1]
    with pytest.raises(ValueError, match="offsets incomplete"):
        e155.validate_source_arrays(rows, shapes, native, contacts, features)


class FakeMJ:
    mjtTrn = SimpleNamespace(mjTRN_JOINT=0)
    mjtJoint = SimpleNamespace(mjJNT_FREE=0, mjJNT_HINGE=3)
    mjtObj = SimpleNamespace(
        mjOBJ_JOINT=0, mjOBJ_GEOM=1, mjOBJ_BODY=2, mjOBJ_ACTUATOR=3
    )

    @staticmethod
    def mj_id2name(_model, kind, index):
        return f"kind{kind}_{index}"


def fake_layout_body():
    joint_ids = np.arange(1, 43)
    m = SimpleNamespace(
        nq=49,
        nv=48,
        nu=42,
        njnt=43,
        ngeom=3,
        nbody=3,
        actuator_trnid=np.column_stack((joint_ids, np.zeros(42, dtype=int))),
        actuator_trntype=np.zeros(42, dtype=int),
        jnt_type=np.array([0] + [3] * 42),
        jnt_dofadr=np.array([0] + list(range(6, 48))),
        geom_bodyid=np.arange(3),
    )
    return SimpleNamespace(
        m=m,
        mj=FakeMJ,
        act_ids=np.arange(42),
        joint_ids=joint_ids,
        qvel_ids=np.arange(6, 48),
        joint_names=[f"kind0_{index}" for index in joint_ids],
    )


def test_layout_maps_free_root_and_original_rm_actuator_slice():
    layout = e155.validate_layout(fake_layout_body())
    assert layout["root_dof_addresses"] == list(range(6))
    assert layout["rm_actuator_indices"] == list(range(28, 35))
    assert layout["rm_joint_ids"] == list(range(29, 36))
    assert layout["rm_dof_addresses"] == list(range(34, 41))
    assert set(layout["root_dof_addresses"]).isdisjoint(layout["rm_dof_addresses"])


def test_layout_rejects_nonjoint_transmission_duplicate_dof_and_freejoint_count():
    body = fake_layout_body()
    body.m.actuator_trntype[2] = 7
    with pytest.raises(ValueError, match="joint transmission"):
        e155.validate_layout(body)
    body = fake_layout_body()
    body.m.jnt_dofadr[2] = body.m.jnt_dofadr[1]
    with pytest.raises(ValueError, match="unique"):
        e155.validate_layout(body)
    body = fake_layout_body()
    body.m.jnt_type[1] = 0
    with pytest.raises(ValueError, match="hinge|free joint"):
        e155.validate_layout(body)


class GeometryMJ:
    @staticmethod
    def mj_forward(_m, _d):
        return None

    @staticmethod
    def mj_jac(_m, _d, target, _rotation, _point, body_id):
        target[:] = 0
        target[1, body_id] = body_id + 1


def test_generated_control_preserves_raw_before_reporting_slip_mismatch(tmp_path):
    nv = 14
    model = SimpleNamespace(nq=2, nv=nv, nu=3, geom_bodyid=np.arange(8))
    data = SimpleNamespace(
        qpos=np.zeros(2), qvel=np.zeros(nv), ctrl=np.zeros(3), time=0.0
    )
    body = SimpleNamespace(m=model, mj=GeometryMJ)
    shadow = SimpleNamespace(
        d=data,
        pose=lambda: (np.zeros(3), np.eye(3), None),
    )
    native = {
        "rotation_world": np.eye(3)[None],
        "heading_horizontal": np.array([[1.0, 0.0, 0.0]]),
        "normal_load_bw": np.array([[0, 0, 0, 0, 0.5, 0]]),
        "slip_mm_s": np.array([[np.nan, np.nan, np.nan, np.nan, 999.0, np.nan]]),
        "slip_valid": np.array([[False, False, False, False, True, False]]),
        "force_forward_bw": np.zeros((1, 6)),
    }
    contacts = {
        "offsets": np.array([0, 1]),
        "leg": np.array([4]),
        "wrench": np.array([[5.0, 0, 0, 0, 0, 0]]),
        "geom1": np.array([0]),
        "geom2": np.array([7]),
        "segment": np.array([21]),
        "sign": np.array([1]),
        "position_world": np.zeros((1, 3)),
        "frame": np.eye(3)[None],
        "contact_id": np.array([3]),
    }
    row = {
        "qpos": np.zeros(2),
        "qvel": np.ones(nv),
        "ctrl": np.zeros(3),
        "time_s": 1.005,
        "control_tick": 201,
    }
    registry = e155.new_registry(nv)
    ledger = {
        "shadow_forward_calls_entered": 0,
        "shadow_forward_calls_completed": 0,
        "jacobian_calls_requested": 0,
        "jacobian_calls_completed": 0,
        "contact_attempts": 0,
        "contact_attempts_incomplete": 0,
        "contacts_completed": 0,
        "controls_completed": 0,
    }
    errors = e155.process_saved_control(
        body=body,
        shadow=shadow,
        row=row,
        native=native,
        contacts=contacts,
        native_index=0,
        speed=0.2,
        control_offset=0,
        context={
            "feet_map": [-1] * 7 + [21],
            "support_geom_ids": [0],
            "weight0": 10.0,
        },
        layout={
            "root_dof_addresses": list(range(6)),
            "rm_dof_addresses": list(range(6, 13)),
        },
        registry=registry,
        ledger=ledger,
    )
    assert any("slip mismatch" in value for value in errors)
    assert len(registry["contacts"]["ap_dof"]) == 1
    assert len(registry["controls"]["tick"]) == 1
    assert ledger == {
        "shadow_forward_calls_entered": 1,
        "shadow_forward_calls_completed": 1,
        "jacobian_calls_requested": 2,
        "jacobian_calls_completed": 2,
        "contact_attempts": 1,
        "contact_attempts_incomplete": 0,
        "contacts_completed": 1,
        "controls_completed": 1,
    }

    class FailingMJ(GeometryMJ):
        calls = 0

        @classmethod
        def mj_jac(cls, *args):
            cls.calls += 1
            if cls.calls == 2:
                raise RuntimeError("generated support jacobian failure")
            return super().mj_jac(*args)

    body.mj = FailingMJ
    failed_registry = e155.new_registry(nv)
    failed_ledger = {
        "shadow_forward_calls_entered": 0,
        "shadow_forward_calls_completed": 0,
        "jacobian_calls_requested": 0,
        "jacobian_calls_completed": 0,
        "contact_attempts": 0,
        "contact_attempts_incomplete": 0,
        "contacts_completed": 0,
        "controls_completed": 0,
    }
    with pytest.raises(RuntimeError, match="support"):
        e155.process_saved_control(
            body=body,
            shadow=shadow,
            row=row,
            native=native,
            contacts=contacts,
            native_index=0,
            speed=0.2,
            control_offset=0,
            context={
                "feet_map": [-1] * 7 + [21],
                "support_geom_ids": [0],
                "weight0": 10.0,
            },
            layout={
                "root_dof_addresses": list(range(6)),
                "rm_dof_addresses": list(range(6, 13)),
            },
            registry=failed_registry,
            ledger=failed_ledger,
        )
    assert failed_ledger == {
        "shadow_forward_calls_entered": 1,
        "shadow_forward_calls_completed": 1,
        "jacobian_calls_requested": 2,
        "jacobian_calls_completed": 1,
        "contact_attempts": 1,
        "contact_attempts_incomplete": 1,
        "contacts_completed": 0,
        "controls_completed": 0,
    }
    assert len(failed_registry["contacts"]["ap_dof"]) == 0
    support_output = tmp_path / "support-failure"
    support_output.mkdir()
    e155.save_raw(support_output, failed_registry)
    with np.load(support_output / "contact-attempts.npz", allow_pickle=False) as raw:
        assert raw["jfoot_obtained"].tolist() == [True]
        assert raw["jsupport_obtained"].tolist() == [False]
        assert raw["decomposition_complete"].tolist() == [False]
        assert raw["completed_contact_row"].tolist() == [-1]
        assert np.isfinite(raw["jfoot"]).all()

    class NaNJacobianMJ(GeometryMJ):
        @staticmethod
        def mj_jac(_m, _d, target, _rotation, _point, body_id):
            GeometryMJ.mj_jac(_m, _d, target, _rotation, _point, body_id)
            if body_id == 0:
                target[0, 0] = np.nan

    body.mj = NaNJacobianMJ
    nan_registry = e155.new_registry(nv)
    nan_ledger = {
        "shadow_forward_calls_entered": 0,
        "shadow_forward_calls_completed": 0,
        "jacobian_calls_requested": 0,
        "jacobian_calls_completed": 0,
        "contact_attempts": 0,
        "contact_attempts_incomplete": 0,
        "contacts_completed": 0,
        "controls_completed": 0,
    }
    with pytest.raises(ValueError, match="Finite contact decomposition"):
        e155.process_saved_control(
            body=body,
            shadow=shadow,
            row=row,
            native=native,
            contacts=contacts,
            native_index=0,
            speed=0.2,
            control_offset=0,
            context={
                "feet_map": [-1] * 7 + [21],
                "support_geom_ids": [0],
                "weight0": 10.0,
            },
            layout={
                "root_dof_addresses": list(range(6)),
                "rm_dof_addresses": list(range(6, 13)),
            },
            registry=nan_registry,
            ledger=nan_ledger,
        )
    assert nan_ledger["jacobian_calls_completed"] == 2
    assert nan_ledger["contact_attempts_incomplete"] == 1
    assert nan_ledger["contacts_completed"] == 0
    nan_output = tmp_path / "nan-failure"
    nan_output.mkdir()
    e155.save_raw(nan_output, nan_registry)
    with np.load(nan_output / "contact-attempts.npz", allow_pickle=False) as raw:
        assert raw["jfoot_obtained"].tolist() == [True]
        assert raw["jsupport_obtained"].tolist() == [True]
        assert raw["decomposition_complete"].tolist() == [False]
        assert np.isnan(raw["jsupport"][0, 0, 0])


def test_partial_registry_is_saved_on_primary_failure(tmp_path):
    registry = e155.new_registry(14)
    for key in registry["controls"]:
        if key == "rm7_numerator":
            value = np.zeros(7)
        elif key == "dof_numerator":
            value = np.zeros(14)
        elif key == "rotation_world":
            value = np.eye(3)
        elif key == "heading_horizontal":
            value = np.array([1.0, 0.0, 0.0])
        else:
            value = False if key.endswith("valid") or key == "numeric_verified" else 0
        registry["controls"][key].append(value)
    registry["offsets"].append(0)
    result = {
        "accepted": False,
        "error": {"type": "RuntimeError", "message": "generated primary"},
    }
    primary = RuntimeError("generated primary")
    secondary = []
    e155._save_failure_artifacts(
        tmp_path,
        registry,
        {"partial": True},
        {"controls_completed": 1},
        result,
        0.0,
        1e20,
        primary,
        secondary,
    )
    assert (tmp_path / "contact-kinematics.npz").exists()
    assert (tmp_path / "control-kinematics.npz").exists()
    assert json.loads((tmp_path / "layout.json").read_text())["partial"] is True
    assert json.loads((tmp_path / "result.json").read_text())["accepted"] is False


def test_pre_model_empty_raw_registry_has_stable_zero_shapes(tmp_path):
    registry = e155.new_registry(0)
    e155.save_raw(tmp_path, registry)
    with np.load(tmp_path / "contact-kinematics.npz", allow_pickle=False) as raw:
        assert raw["offsets"].tolist() == [0]
        assert raw["jfoot"].shape == (0, 3, 0)
        assert raw["ap_dof"].shape == (0, 0)
    with np.load(tmp_path / "control-kinematics.npz", allow_pickle=False) as controls:
        assert controls["rotation_world"].shape == (0, 3, 3)
        assert controls["dof_numerator"].shape == (0, 0)


def test_source_gate_requires_exact_sha_freeze_one_process_and_both_sources(tmp_path):
    sources = {name: e155.sha256(e155.ROOT / name) for name in e155.OWNED}
    valid = {
        "accepts_single_kinematic_diagnostic": True,
        "freeze_sha256": e155.FREEZE_SHA,
        "original_processes_allowed": 1,
        "source_sha256": sources,
    }
    gate = tmp_path / "gate.json"

    def write(value):
        gate.write_text(json.dumps(value), encoding="utf-8")
        return hashlib.sha256(gate.read_bytes()).hexdigest()

    assert e155.validate_source_gate(gate, write(valid)) == valid
    for key, value, message in (
        ("accepts_single_kinematic_diagnostic", False, "did not accept"),
        ("freeze_sha256", "wrong", "different freeze"),
        ("original_processes_allowed", 0, "exactly one"),
        ("source_sha256", {e155.OWNED[0]: sources[e155.OWNED[0]]}, "both owned"),
    ):
        with pytest.raises(ValueError, match=message):
            e155.validate_source_gate(gate, write({**valid, key: value}))


def test_generated_full_run_returns_only_with_matching_manifest(monkeypatch, tmp_path):
    from flylab.c.storage import StateStore
    from tools import analyze_c_banc_m3_signed_contact as e154

    paths = {
        name: tmp_path / name
        for name in (
            "trace.json",
            "native.npz",
            "contacts.npz",
            "features.npz",
            "context.json",
            "checkpoint",
        )
    }
    payload = trace_payload()
    for row in payload["samples"]:
        row["qpos"] = [0.0] * 49
        row["qvel"] = [0.0] * 48
        row["ctrl"] = [0.0] * 42
    paths["trace.json"].write_text(json.dumps(payload), encoding="utf-8")
    indices = 49 + 50 * np.arange(200)
    native = {
        "time_s": 1.0 + np.cumsum(np.full(10000, 0.0001)),
        "rotation_world": np.repeat(np.eye(3)[None], 10000, axis=0),
        "heading_horizontal": np.repeat([[1.0, 0.0, 0.0]], 10000, axis=0),
        "force_forward_bw": np.zeros((10000, 6)),
        "normal_load_bw": np.zeros((10000, 6)),
        "slip_mm_s": np.full((10000, 6), np.nan),
        "slip_valid": np.zeros((10000, 6), dtype=bool),
    }
    native["normal_load_bw"][indices, 4] = 0.1
    native["slip_mm_s"][indices, 4] = 0.0
    native["slip_valid"][indices, 4] = True
    np.savez(paths["native.npz"], **native)
    counts = np.zeros(10000, dtype=np.int64)
    counts[indices] = 1
    contacts = {
        "offsets": np.concatenate(([0], np.cumsum(counts))),
        "contact_id": np.zeros(200, dtype=np.int64),
        "geom1": np.zeros(200, dtype=np.int64),
        "geom2": np.full(200, 7, dtype=np.int64),
        "segment": np.full(200, 21, dtype=np.int64),
        "leg": np.full(200, 4, dtype=np.int64),
        "sign": np.ones(200, dtype=np.int8),
        "frame": np.repeat(np.eye(3)[None], 200, axis=0),
        "wrench": np.repeat([[1.0, 0, 0, 0, 0, 0]], 200, axis=0),
        "position_world": np.zeros((200, 3)),
    }
    np.savez(paths["contacts.npz"], **contacts)
    np.savez(
        paths["features.npz"],
        time_s=np.arange(500) * 0.005,
        speed_mm_s=np.ones(500),
    )
    context = {
        "body_hash": "body",
        "body_parameters": {"generated": True},
        "feet_map": [-1] * 7 + [21],
        "support_geom_ids": [0],
        "weight0": 10.0,
    }
    paths["context.json"].write_text(json.dumps(context), encoding="utf-8")

    class RunMJ(FakeMJ):
        @staticmethod
        def MjData(model):
            return SimpleNamespace(
                qpos=np.zeros(model.nq),
                qvel=np.zeros(model.nv),
                ctrl=np.zeros(model.nu),
                time=0.0,
                xpos=np.zeros((model.nbody, 3)),
                xmat=np.repeat(np.eye(3).reshape(1, 9), model.nbody, axis=0),
            )

        @staticmethod
        def mj_forward(_model, _data):
            return None

        @staticmethod
        def mj_jac(_model, _data, target, _rotation, _point, _body_id):
            target[:] = 0

    body = fake_layout_body()
    body.mj = RunMJ
    body.m.ngeom = 8
    body.m.nbody = 8
    body.m.geom_bodyid = np.arange(8)
    body.m.opt = SimpleNamespace(timestep=0.0001)
    body.model_hash = "body"
    body.t0 = 0.2
    body.thorax = 1
    body.weight0 = 10.0
    body.support_geom_ids = {0}
    body._force_maps = {"feet": (np.array([-1] * 7 + [21]), None)}
    body.closed = False
    body.close = lambda: setattr(body, "closed", True)

    def construct(_saved, _observation, ledger):
        ledger["body_constructor_attempts"] += 1
        ledger["body_constructor_completed"] += 1
        ledger["settling_physics_steps"] += 2000
        return body

    monkeypatch.setattr(e154, "construct_body_safely", construct)
    monkeypatch.setattr(e154, "validate_settling_count", lambda *_args: 2000)
    monkeypatch.setattr(
        e154, "validate_body_parameters", lambda _saved: {"generated": True}
    )
    monkeypatch.setattr(
        StateStore,
        "load",
        staticmethod(
            lambda _path: {"control_tick": 200, "body": {"modelHash": "body"}}
        ),
    )
    freeze = {
        "trace": "trace.json",
        "native": "native.npz",
        "contacts": "contacts.npz",
        "features": "features.npz",
        "context": "context.json",
        "checkpoint": "checkpoint",
        "body_hash": "body",
        "input_sha256": {},
        "reused_source_sha256": {},
        "preservation_sha256": {},
        "resource_limits": {
            "internal_wall_s": 35,
            "external_wall_s": 45,
            "supervisor_wall_s": 55,
            "raw_bytes": 536870912,
            "free_bytes": 0,
            "prior_actual_s": 0,
            "prior_conservative_s": 0,
            "previous_allocation_s": 0,
            "prospective_cumulative_s": 55,
        },
    }
    monkeypatch.setattr(e155, "load_freeze", lambda *_args: freeze)
    monkeypatch.setattr(e155, "validate_source_gate", lambda *_args: {})
    monkeypatch.setattr(e155, "verify_hash_map", lambda _entries: None)
    monkeypatch.setattr(e155, "threadpool_info", list)
    monkeypatch.setattr(e155, "safe_path", lambda name: paths[name])
    output = tmp_path / "output"
    result = e155.run(Path("freeze"), "sha", Path("review"), "sha", output)
    assert result["accepted"] is True
    manifest = json.loads((output / "output-manifest.json").read_text())
    assert manifest["files"]
    for name, record in manifest["files"].items():
        assert e155.sha256(output / name) == record["sha256"]
    progress = (output / "progress.jsonl").read_text().splitlines()
    assert json.loads(progress[-1])["stage"] == "complete"
    assert body.closed is True


def test_direct_cli_help_from_outside_repository(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(e155.ROOT / e155.OWNED[0]), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--freeze-sha256" in completed.stdout
    assert "--source-review-sha256" in completed.stdout
    assert "--output" in completed.stdout


def test_direct_cli_rejects_wrong_freeze_before_output_or_original_access(tmp_path):
    output = tmp_path / "must-not-exist"
    completed = subprocess.run(
        [
            sys.executable,
            str(e155.ROOT / e155.OWNED[0]),
            "--freeze",
            str(
                e155.ROOT
                / "verification/banc-natural-gait-m3-20260919/e155-rm-contact-jacobian-freeze.json"
            ),
            "--freeze-sha256",
            "wrong",
            "--source-review",
            str(tmp_path / "absent.json"),
            "--source-review-sha256",
            "wrong",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "Exact E155 freeze required" in completed.stderr
    assert not output.exists()


def test_owned_paths_are_only_the_two_new_sources():
    assert e155.OWNED == (
        "tools/analyze_c_banc_m3_rm_contact_jacobian.py",
        "tests/test_c_banc_rm_contact_jacobian.py",
    )
    assert all(
        Path(value).name.startswith(("analyze_", "test_")) for value in e155.OWNED
    )
