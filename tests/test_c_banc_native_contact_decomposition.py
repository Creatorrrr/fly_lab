from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import analyze_c_banc_m3_native_contact_decomposition as e157


def test_frozen_design_contract_loads_without_original_inputs():
    path = (
        e157.ROOT
        / "verification/banc-natural-gait-m3-20260919/e157-native-contact-decomposition-freeze.json"
    )
    freeze = e157.load_freeze(path, e157.FREEZE_SHA)
    assert freeze["counts"]["jacobian_calls"] == 48840
    assert freeze["sample_contract"]["positive_contacts_by_leg"] == [
        3808,
        3720,
        5093,
        2363,
        6156,
        3280,
    ]


class FakeMJ:
    mjtTrn = SimpleNamespace(mjTRN_JOINT=0)
    mjtJoint = SimpleNamespace(mjJNT_FREE=0, mjJNT_HINGE=3)
    mjtObj = SimpleNamespace(mjOBJ_JOINT=0)

    @staticmethod
    def mj_id2name(_model, _kind, index):
        return f"joint_{index}"


def fake_layout_body(nv: int = 48):
    joint_ids = np.arange(1, 43)
    model = SimpleNamespace(
        nq=49,
        nv=nv,
        nu=42,
        njnt=43,
        actuator_trnid=np.column_stack((joint_ids, np.zeros(42, dtype=int))),
        actuator_trntype=np.zeros(42, dtype=int),
        jnt_type=np.array([0] + [3] * 42),
        jnt_dofadr=np.array([0] + list(range(6, 48))),
        geom_bodyid=np.arange(8),
    )
    return SimpleNamespace(
        m=model,
        mj=FakeMJ,
        act_ids=np.arange(42),
        joint_ids=joint_ids,
        qvel_ids=np.arange(6, 48),
        joint_names=[f"joint_{index}" for index in joint_ids],
    )


def test_layout_maps_six_leg_slices_and_root():
    layout = e157.validate_layout(fake_layout_body())
    assert layout["root_dof_addresses"] == list(range(6))
    assert layout["leg_dof_addresses"] == np.arange(6, 48).reshape(6, 7).tolist()
    assert layout["leg_joint_ids"] == np.arange(1, 43).reshape(6, 7).tolist()
    assert len({value for leg in layout["leg_dof_addresses"] for value in leg}) == 42


def test_layout_rejects_transmission_overlap_and_wrong_body_mapping():
    body = fake_layout_body()
    body.m.actuator_trntype[0] = 9
    with pytest.raises(ValueError, match="joint transmission"):
        e157.validate_layout(body)
    body = fake_layout_body()
    body.m.jnt_dofadr[2] = body.m.jnt_dofadr[1]
    with pytest.raises(ValueError, match="DOF mapping"):
        e157.validate_layout(body)
    body = fake_layout_body()
    body.qvel_ids = np.arange(42)
    with pytest.raises(ValueError, match="DOF mapping"):
        e157.validate_layout(body)


def test_generic_contact_classifier_covers_all_legs_signs_and_segments():
    mapping = np.array([-1, 0, 5, 10, 15, 20, 25])
    for leg in range(6):
        assert e157.classify_saved_contact(
            0, leg + 1, leg * 5, leg, 1, mapping, [0]
        ) == (leg + 1, 0)
        assert e157.classify_saved_contact(
            leg + 1, 0, leg * 5, leg, -1, mapping, [0]
        ) == (leg + 1, 0)
    with pytest.raises(ValueError, match="leg/segment"):
        e157.classify_saved_contact(0, 1, 0, 1, 1, mapping, [0])
    with pytest.raises(ValueError, match="exactly"):
        e157.classify_saved_contact(0, 1, 0, 0, 0, mapping, [0])


def independent_decomposition(
    frame, qvel, jfoot, jsupport, heading, root, own, wrench, sign, weight
):
    normal = frame[0]
    projector = np.eye(3) - np.outer(normal, normal)
    tangent = projector @ (jfoot - jsupport)
    velocity = tangent @ qvel
    force = sign * frame.T @ wrench[:3] / weight
    tangent_force = projector @ force
    ap = (heading @ tangent) * qvel
    power = (tangent_force @ tangent) * qvel
    rest = np.ones(qvel.size, dtype=bool)
    rest[root] = False
    rest[own] = False
    return tangent, velocity, force, tangent_force, ap, power, rest


def test_decompose_rotated_frame_moving_support_force_sign_and_closures():
    rng = np.random.default_rng(157)
    nv = 20
    qvel = rng.normal(size=nv)
    jfoot = rng.normal(size=(3, nv))
    jsupport = rng.normal(size=(3, nv))
    frame = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    heading = np.array([1.0, 0.0, 0.0])
    root = np.arange(6)
    own = np.arange(6, 13)
    wrench = np.array([8.0, -2.0, 3.0, 7.0, 11.0, 13.0])
    actual = e157.decompose_contact(
        frame, qvel, jfoot, jsupport, heading, root, own, wrench, -1, 4.0
    )
    tangent, velocity, force, tangent_force, ap, power, rest = (
        independent_decomposition(
            frame, qvel, jfoot, jsupport, heading, root, own, wrench, -1, 4.0
        )
    )
    np.testing.assert_allclose(actual["tangent_jacobian"], tangent)
    np.testing.assert_allclose(actual["tangent_velocity"], velocity)
    np.testing.assert_allclose(actual["force_world_bw"], force)
    np.testing.assert_allclose(actual["tangent_force_bw"], tangent_force)
    np.testing.assert_allclose(actual["ap_dof"], ap)
    np.testing.assert_allclose(actual["power_dof"], power)
    assert actual["ap_total"] == pytest.approx(heading @ velocity)
    assert actual["power_total"] == pytest.approx(tangent_force @ velocity)
    assert actual["ap_residual"] == pytest.approx(ap[rest].sum())
    assert actual["power_residual"] == pytest.approx(power[rest].sum())
    assert (
        max(
            actual[key]
            for key in (
                "ap_closure_error",
                "ap_direct_error",
                "power_closure_error",
                "power_direct_error",
                "projection_error",
            )
        )
        < 1e-12
    )


def test_power_includes_contact_force_once_without_normal_load_reweighting():
    qvel = np.arange(1.0, 15.0)
    jfoot = np.zeros((3, 14))
    jfoot[1] = np.linspace(0.1, 1.4, 14)
    result = e157.decompose_contact(
        np.eye(3),
        qvel,
        jfoot,
        np.zeros_like(jfoot),
        np.array([0.0, 1.0, 0.0]),
        np.arange(6),
        np.arange(6, 13),
        np.array([20.0, 5.0, 0.0, 0, 0, 0]),
        1,
        10.0,
    )
    expected = (0.5 * jfoot[1] * qvel).sum()
    assert result["normal_load_bw"] == 2.0
    assert result["power_total"] == pytest.approx(expected)
    assert result["power_total"] != pytest.approx(expected * result["normal_load_bw"])


@pytest.mark.parametrize("bad", ["nan", "heading", "overlap", "sign"])
def test_decomposition_rejects_invalid_inputs(bad):
    frame = np.eye(3)
    qvel = np.ones(14)
    jfoot = np.zeros((3, 14))
    jsupport = np.zeros((3, 14))
    heading = np.array([1.0, 0.0, 0.0])
    root = np.arange(6)
    own = np.arange(6, 13)
    sign = 1
    if bad == "nan":
        jfoot[0, 0] = np.nan
    if bad == "heading":
        heading = np.array([1.0, 0.0, 1.0])
    if bad == "overlap":
        own = np.arange(5, 12)
    if bad == "sign":
        sign = 0
    with pytest.raises(ValueError):
        e157.decompose_contact(
            frame, qvel, jfoot, jsupport, heading, root, own, np.ones(6), sign, 10.0
        )


class JacobianMJ:
    calls = 0
    fail_second = False
    nan_second = False

    @classmethod
    def mj_jac(cls, _m, _d, target, _rotation, _point, body_id):
        cls.calls += 1
        if cls.fail_second and cls.calls == 2:
            raise RuntimeError("generated second Jacobian failure")
        target[:] = 0
        target[1, body_id] = body_id + 1
        if cls.nan_second and cls.calls == 2:
            target[0, 0] = np.nan


def one_contact_inputs(nv=14):
    raw = {
        "contact_id": np.array([3]),
        "geom1": np.array([0]),
        "geom2": np.array([1]),
        "segment": np.array([0]),
        "leg": np.array([0]),
        "sign": np.array([1]),
        "frame": np.eye(3)[None],
        "wrench": np.array([[1.0, 0, 0, 0, 0, 0]]),
        "position_world": np.zeros((1, 3)),
    }
    source = {**raw, "offsets": np.array([0, 1])}
    body = SimpleNamespace(
        m=SimpleNamespace(nv=nv, geom_bodyid=np.arange(3)), mj=JacobianMJ
    )
    shadow = SimpleNamespace(d=SimpleNamespace())
    context = {"feet_map": [-1, 0], "support_geom_ids": [0], "weight0": 10.0}
    layout = {
        "root_dof_addresses": list(range(6)),
        "leg_dof_addresses": [list(range(6, 13))] * 6,
    }
    ledger = {
        "contact_attempts": 0,
        "contact_attempts_incomplete": 0,
        "jacobian_calls_requested": 0,
        "jacobian_calls_returned": 0,
        "contacts_completed": 0,
    }
    return raw, source, body, shadow, context, layout, ledger


def test_second_jacobian_failure_preserves_first_raw_and_counts(tmp_path):
    raw, source, body, shadow, context, layout, ledger = one_contact_inputs()
    registry = e157.new_registry(1, 14, 1)
    JacobianMJ.calls = 0
    JacobianMJ.fail_second = True
    JacobianMJ.nan_second = False
    with pytest.raises(RuntimeError, match="second"):
        e157.process_positive_contacts(
            body=body,
            shadow=shadow,
            native_index=0,
            raw_contacts=raw,
            source_contacts=source,
            source_start=0,
            heading=np.array([1.0, 0.0, 0.0]),
            qvel=np.ones(14),
            context=context,
            layout=layout,
            registry=registry,
            ledger=ledger,
        )
    assert ledger == {
        "contact_attempts": 1,
        "contact_attempts_incomplete": 1,
        "jacobian_calls_requested": 2,
        "jacobian_calls_returned": 1,
        "contacts_completed": 0,
    }
    e157.save_raw(tmp_path, registry)
    with np.load(tmp_path / "contact-attempts.npz") as saved:
        assert saved["jfoot_obtained"].tolist() == [True]
        assert saved["jsupport_obtained"].tolist() == [False]
        assert np.isfinite(saved["jfoot"]).all()


def test_nan_jacobian_is_saved_before_decomposition_assertion(tmp_path):
    raw, source, body, shadow, context, layout, ledger = one_contact_inputs()
    registry = e157.new_registry(1, 14, 1)
    JacobianMJ.calls = 0
    JacobianMJ.fail_second = False
    JacobianMJ.nan_second = True
    with pytest.raises(ValueError, match="Finite contact decomposition"):
        e157.process_positive_contacts(
            body=body,
            shadow=shadow,
            native_index=0,
            raw_contacts=raw,
            source_contacts=source,
            source_start=0,
            heading=np.array([1.0, 0.0, 0.0]),
            qvel=np.ones(14),
            context=context,
            layout=layout,
            registry=registry,
            ledger=ledger,
        )
    e157.save_raw(tmp_path, registry)
    with np.load(tmp_path / "contact-attempts.npz") as saved:
        assert saved["jfoot_obtained"].tolist() == [True]
        assert saved["jsupport_obtained"].tolist() == [True]
        assert saved["decomposition_complete"].tolist() == [False]
        assert np.isnan(saved["jsupport"][0, 0, 0])


def test_empty_and_partial_raw_have_fixed_shapes(tmp_path):
    registry = e157.new_registry(4, 14, 3)
    e157.save_raw(tmp_path, registry)
    with np.load(tmp_path / "native-replay.npz") as native:
        assert native["qpos"].shape == (0, 4)
        assert native["qvel"].shape == (0, 14)
    with np.load(tmp_path / "contact-decomposition.npz") as contact:
        assert contact["offsets"].tolist() == [0]
        assert contact["jfoot"].shape == (0, 3, 14)
        assert contact["power_dof"].shape == (0, 14)


def test_returned_observed_contact_is_saved_before_numeric_or_mapping_checks(tmp_path):
    registry = e157.new_registry(1, 14, 1)
    raw = {
        "contact_id": np.array([9]),
        "geom1": np.array([0]),
        "geom2": np.array([1]),
        "segment": np.array([0]),
        "leg": np.array([0]),
        "sign": np.array([1]),
        "frame": np.full((1, 3, 3), np.nan),
        "wrench": np.array([[1.0, np.nan, 0, 0, 0, 0]]),
        "position_world": np.array([[0.0, 0.0, np.nan]]),
    }
    e157.append_observed_contacts(registry, 4, 7, raw)
    e157.save_raw(tmp_path, registry)
    with np.load(tmp_path / "observed-contacts.npz") as saved:
        assert saved["offsets"].tolist() == [0, 1]
        assert saved["native_index"].tolist() == [4]
        assert saved["source_raw_index"].tolist() == [7]
        assert np.isnan(saved["frame"][0]).all()
        assert np.isnan(saved["wrench"][0, 1])


def generated_summary_inputs():
    native = {
        "time_s": np.arange(10000) * 0.0001,
        "normal_load_bw": np.zeros((10000, 6)),
        "force_forward_bw": np.zeros((10000, 6)),
    }
    native["normal_load_bw"][:, 0] = 0.1
    native["force_forward_bw"][:5000, 0] = -0.2
    native["force_forward_bw"][5000:, 0] = 0.3
    contacts = {
        "native_index": np.arange(10000),
        "leg": np.zeros(10000, dtype=int),
        "normal_load_bw": np.full(10000, 0.1),
        "slip_speed": np.ones(10000),
        "ap_total": np.ones(10000),
        "ap_root": np.full(10000, 0.2),
        "ap_own7": np.full((10000, 7), 0.1),
        "ap_residual": np.full(10000, 0.1),
        "power_total": np.full(10000, 2.0),
        "power_root": np.full(10000, 0.4),
        "power_own7": np.full((10000, 7), 0.2),
        "power_residual": np.full(10000, 0.2),
    }
    contacts["ap_dof"] = np.full((10000, 14), 1.0 / 14)
    contacts["power_dof"] = np.full((10000, 14), 2.0 / 14)
    speed = np.tile([0.4, 0.6], 100)
    return native, contacts, speed


def test_group_aggregation_uses_50_native_repeats_early_late_and_power_once():
    native, contacts, speed = generated_summary_inputs()
    groups = e157.aggregate_summary(native, contacts, speed)
    assert groups["all"]["native_count"] == 10000
    assert groups["forward"]["native_count"] == 5000
    assert groups["interrupted"]["native_count"] == 5000
    assert groups["early"]["native_count"] == 2000
    assert groups["late"]["native_count"] == 8000
    lf = groups["all"]["legs"]["lf"]
    assert lf["all_positive_contacts"]["load_weighted_velocity"][
        "slip"
    ] == pytest.approx(1.0)
    assert lf["all_positive_contacts"]["tangential_work_bw_mm"][
        "total"
    ] == pytest.approx(2.0)
    assert lf["supported_braking_native_count"] == 5000
    assert lf["supported_propulsion_native_count"] == 5000
    assert (
        groups["all"]["legs"]["rh"]["all_positive_contacts"]["load_weighted_velocity"]
        is None
    )


def test_source_gate_requires_exact_hash_freeze_empty_defects_and_one_process(tmp_path):
    sources = {name: e157.sha256(e157.ROOT / name) for name in e157.OWNED}
    valid = {
        "accepts_single_native_decomposition": True,
        "blocking_defects": [],
        "original_processes_allowed": 1,
        "freeze_sha256": e157.FREEZE_SHA,
        "source_sha256": sources,
    }
    path = tmp_path / "gate.json"

    def write(value):
        path.write_text(json.dumps(value), encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    assert e157.validate_source_gate(path, write(valid)) == valid
    for key, value in (
        ("accepts_single_native_decomposition", False),
        ("blocking_defects", ["x"]),
        ("original_processes_allowed", True),
        ("freeze_sha256", "x"),
        ("source_sha256", {}),
    ):
        with pytest.raises(ValueError):
            e157.validate_source_gate(path, write({**valid, key: value}))


def test_failure_publication_preserves_primary_and_matching_manifest(tmp_path):
    result = {
        "accepted": False,
        "error": {"type": "RuntimeError", "message": "primary"},
    }
    primary = RuntimeError("primary")
    secondary = []
    e157._save_failure(
        tmp_path,
        e157.new_registry(),
        {"partial": True},
        {"contacts_completed": 0},
        result,
        0.0,
        1e20,
        primary,
        secondary,
    )
    assert (
        json.loads((tmp_path / "result.json").read_text())["error"]["message"]
        == "primary"
    )
    manifest = json.loads((tmp_path / "output-manifest.json").read_text())
    for name, record in manifest["files"].items():
        assert e157.sha256(tmp_path / name) == record["sha256"]


def test_output_manifest_excludes_itself_and_counts_bytes(tmp_path):
    (tmp_path / "a").write_bytes(b"abc")
    manifest = e157.output_manifest(tmp_path)
    assert manifest["files"]["a"]["bytes"] == 3
    assert manifest["total_bytes"] == 3


def install_generated_full_run(
    monkeypatch,
    tmp_path,
    *,
    failure_stage=None,
    native_time_offset=0.0,
    cleanup_failure=False,
):
    from flylab.c import storage, walking_trace
    from tools import analyze_c_banc_m3_signed_contact as e154

    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "trace": tmp_path / "trace.json",
        "native": tmp_path / "native.npz",
        "contacts": tmp_path / "contacts.npz",
        "controls": tmp_path / "controls.npz",
        "features": tmp_path / "features.npz",
        "context": tmp_path / "context.json",
        "checkpoint": tmp_path / "checkpoint",
    }
    rows = []
    for tick in range(401):
        rows.append(
            {
                "control_tick": tick,
                "time_s": tick * 0.005,
                "qpos": [0.0] * 49,
                "qvel": [0.0] * 48,
                "ctrl": [0.0] * 42,
                "joint_targets_rad": [0.0] * 42,
                "adhesion": [False] * 6,
                "tendon_inputs": {},
            }
        )
    paths["trace"].write_text(
        json.dumps({"body_hash": "body", "samples": rows}), encoding="utf-8"
    )
    counts = np.array([3808, 3720, 5093, 2363, 6156, 3280])
    active = np.arange(10000)[:, None] < counts[None, :]
    normal = active.astype(float) * 0.1
    slip = np.where(active, 0.0, np.nan)
    times = np.empty(10000)
    generated_time = 1.0
    for index in range(10000):
        generated_time += 0.0001
        times[index] = generated_time
    times[0] += native_time_offset
    native = {
        "time_s": times,
        "rotation_world": np.repeat(np.eye(3)[None], 10000, axis=0),
        "force_world_bw": np.zeros((10000, 6, 3)),
        "force_forward_bw": np.zeros((10000, 6)),
        "normal_load_bw": normal,
        "slip_mm_s": slip,
        "slip_valid": active,
        "nonfoot_contact": np.zeros(10000, dtype=bool),
        "upright": np.ones(10000),
    }
    np.savez(paths["native"], **native)
    np.savez(
        paths["contacts"],
        offsets=np.zeros(10001, dtype=np.int64),
        contact_id=np.empty(0, dtype=np.int64),
        geom1=np.empty(0, dtype=np.int64),
        geom2=np.empty(0, dtype=np.int64),
        segment=np.empty(0, dtype=np.int64),
        leg=np.empty(0, dtype=np.int64),
        sign=np.empty(0, dtype=np.int8),
        frame=np.empty((0, 3, 3)),
        wrench=np.empty((0, 6)),
        position_world=np.empty((0, 3)),
    )
    np.savez(
        paths["controls"],
        tick=np.arange(200, 401),
        time_s=np.arange(200, 401) * 0.005,
        qpos=np.zeros((201, 49)),
        qvel=np.zeros((201, 48)),
        ctrl=np.zeros((201, 42)),
        verified=np.ones(201, dtype=bool),
    )
    np.savez(
        paths["features"],
        time_s=np.arange(401) * 0.005,
        speed_mm_s=np.tile([0.4, 0.6], 201)[:401],
    )
    context = {
        "body_hash": "body",
        "body_parameters": {"generated": True},
        "feet_map": [-1, 0, 5, 10, 15, 20, 25],
        "support_geom_ids": [0],
        "weight0": 10.0,
    }
    paths["context"].write_text(json.dumps(context), encoding="utf-8")

    class RunMJ(FakeMJ):
        mjtState = SimpleNamespace(mjSTATE_INTEGRATION=0)

        @staticmethod
        def MjData(model):
            return SimpleNamespace(
                qpos=np.zeros(model.nq),
                qvel=np.zeros(model.nv),
                ctrl=np.zeros(model.nu),
                time=1.0,
            )

        @staticmethod
        def mj_stateSize(_model, _state_type):
            return 1

        @staticmethod
        def mj_getState(_model, data, state, _state_type):
            state[0] = data.time

        @staticmethod
        def mj_setState(_model, data, state, _state_type):
            data.time = state[0]

        @staticmethod
        def mj_forward(_model, _data):
            if failure_stage == "shadow_forward":
                raise RuntimeError("generated shadow forward failure")

    class FakeBody:
        def __init__(self):
            layout = fake_layout_body()
            self.__dict__.update(layout.__dict__)
            self.mj = RunMJ
            self.m.geom_bodyid = np.arange(7)
            self.m.opt = SimpleNamespace(timestep=0.0001)
            self.d = RunMJ.MjData(self.m)
            self.t0 = 0.2
            self.model_hash = "body"
            self.weight0 = 10.0
            self.support_geom_ids = {0}
            self._force_maps = {"feet": (np.array(context["feet_map"]), None)}
            self.push_left = 0.0
            self.push_force = np.zeros(3)
            self.fault = None
            self.sample_index = 0
            self.closed = False
            self._step_physics = self.base_step

        def base_step(self):
            self.d.time += 0.0001
            self.sample_index += 1
            if failure_stage == "integration":
                self.d.qpos[0] = 17.0
                raise RuntimeError("generated integration failure")

        def physics_time(self):
            return self.d.time

        def step_joint_targets(self, _joint, _adhesion, **_kwargs):
            for _ in range(50):
                self._step_physics()

        def nonfoot_contact(self):
            return False

        def close(self):
            self.closed = True
            if cleanup_failure:
                raise RuntimeError("generated cleanup failure")

    body = FakeBody()

    def construct(_saved, _observation, ledger):
        ledger["body_constructor_attempts"] += 1
        ledger["body_constructor_completed"] += 1
        ledger["settling_physics_steps"] += 2000
        return body

    monkeypatch.setattr(e154, "construct_body_safely", construct)
    monkeypatch.setattr(
        e154, "validate_body_parameters", lambda _saved: {"generated": True}
    )
    monkeypatch.setattr(e154, "implicit_request_for_trace", lambda *_args: None)
    monkeypatch.setattr(
        e154,
        "replay_command",
        lambda *_args, **_kwargs: (
            np.zeros(42),
            np.zeros(6, dtype=bool),
            {
                "push_left_s": 0.0,
                "push_force": np.zeros(3),
                "step_kwargs": {},
            },
        ),
    )

    def observe(_body, _shadow):
        if failure_stage == "observer":
            raise RuntimeError("generated observer failure")
        count = 1 if failure_stage == "diagnostics" else 0
        return (
            np.zeros((6, 3)),
            {
                "contact_id": np.arange(count, dtype=np.int64),
                "geom1": np.zeros(count, dtype=np.int64),
                "geom2": np.ones(count, dtype=np.int64),
                "segment": np.zeros(count, dtype=np.int64),
                "leg": np.zeros(count, dtype=np.int64),
                "sign": np.ones(count, dtype=np.int8),
                "frame": np.repeat(np.eye(3)[None], count, axis=0),
                "wrench": np.repeat([[1.0, 0, 0, 0, 0, 0]], count, axis=0),
                "position_world": np.zeros((count, 3)),
            },
            np.eye(3),
        )

    monkeypatch.setattr(e154, "_observe_shadow", observe)

    def diagnostics(_shadow):
        if failure_stage == "diagnostics":
            raise RuntimeError("generated diagnostics failure")
        return {
            "normal_load_bw": normal[body.sample_index - 1],
            "slip_mm_s": [
                0.0 if value else None for value in active[body.sample_index - 1]
            ],
        }

    monkeypatch.setattr(
        walking_trace,
        "contact_diagnostics",
        diagnostics,
    )
    monkeypatch.setattr(
        storage.StateStore,
        "load",
        staticmethod(
            lambda _path: {
                "control_tick": 200,
                "body": {"modelHash": "body"},
            }
        ),
    )
    monkeypatch.setattr(
        e157,
        "validate_source_arrays",
        lambda *_args: (rows[200:401], np.tile([0.4, 0.6], 100)),
    )

    contact_native, contact_leg = np.nonzero(active)
    contact_load = np.full(len(contact_native), 0.1)
    synthetic_contacts = {
        "offsets": np.concatenate(([0], np.cumsum(active.sum(axis=1), dtype=np.int64))),
        "native_index": contact_native,
        "leg": contact_leg,
        "normal_load_bw": contact_load,
        "slip_speed": np.zeros(len(contact_native)),
    }
    for key in (
        "ap_total",
        "ap_root",
        "ap_residual",
        "power_total",
        "power_root",
        "power_residual",
    ):
        synthetic_contacts[key] = np.zeros(len(contact_native))
    synthetic_contacts["ap_own7"] = np.zeros((len(contact_native), 7))
    synthetic_contacts["power_own7"] = np.zeros((len(contact_native), 7))
    synthetic_contacts["ap_dof"] = np.zeros((len(contact_native), 48))
    synthetic_contacts["power_dof"] = np.zeros((len(contact_native), 48))

    def fake_process(**kwargs):
        index = kwargs["native_index"]
        count = int(active[index].sum())
        ledger = kwargs["ledger"]
        ledger["contact_attempts"] += count
        ledger["contact_attempts_incomplete"] += count
        ledger["jacobian_calls_requested"] += 2 * count
        ledger["jacobian_calls_returned"] += 2 * count
        ledger["contacts_completed"] += count
        ledger["contact_attempts_incomplete"] -= count

    original_registry_arrays = e157.registry_arrays

    def fake_registry_arrays(registry):
        native_arrays, attempts, _contacts, controls = original_registry_arrays(
            registry
        )
        return native_arrays, attempts, synthetic_contacts, controls

    def fake_save_raw(output, registry):
        native_arrays, attempts, actual_contacts, controls = original_registry_arrays(
            registry
        )
        e157.atomic_npz(output / "native-replay.npz", **native_arrays)
        e157.atomic_npz(
            output / "observed-contacts.npz", **e157.observed_arrays(registry)
        )
        e157.atomic_npz(output / "contact-attempts.npz", **attempts)
        e157.atomic_npz(
            output / "contact-decomposition.npz",
            **(
                synthetic_contacts
                if len(native_arrays["native_index"]) == 10000
                else actual_contacts
            ),
        )
        e157.atomic_npz(output / "replay-controls.npz", **controls)

    monkeypatch.setattr(e157, "process_positive_contacts", fake_process)
    monkeypatch.setattr(e157, "registry_arrays", fake_registry_arrays)
    monkeypatch.setattr(e157, "save_raw", fake_save_raw)
    monkeypatch.setattr(
        e157,
        "load_freeze",
        lambda *_args: {
            "trace": "trace",
            "native": "native",
            "contacts": "contacts",
            "baseline_controls": "controls",
            "features": "features",
            "context": "context",
            "checkpoint": "checkpoint",
            "body_hash": "body",
            "input_sha256": {},
            "reused_source_sha256": {},
            "preservation_sha256": {},
            "resource_limits": {
                "internal_s": 90,
                "external_s": 100,
                "supervisor_s": 110,
                "raw_bytes": 805306368,
                "free_bytes": 0,
            },
            "prior_actual_original_s": 0,
            "prior_conservative_s": 0,
            "previous_allocation_s": 0,
            "prospective_cumulative_s": 110,
        },
    )
    monkeypatch.setattr(e157, "validate_source_gate", lambda *_args: {})
    monkeypatch.setattr(e157, "verify_hash_map", lambda _entries: None)
    monkeypatch.setattr(e157, "safe_path", lambda name: paths[name])
    monkeypatch.setattr(e157, "threadpool_info", list)
    return body


def test_generated_full_run_publishes_matching_manifest(monkeypatch, tmp_path):
    body = install_generated_full_run(monkeypatch, tmp_path)
    output = tmp_path / "success"
    result = e157.run(Path("freeze"), "sha", Path("review"), "sha", output)
    assert result["accepted"] is True
    assert result["native_samples"] == 10000
    assert result["positive_contacts"] == 24420
    assert result["jacobian_calls"] == 48840
    manifest = json.loads((output / "output-manifest.json").read_text())
    for name, record in manifest["files"].items():
        assert e157.sha256(output / name) == record["sha256"]
    assert body.closed is True


def test_generated_final_deadline_failure_publishes_matching_failed_manifest(
    monkeypatch, tmp_path
):
    body = install_generated_full_run(monkeypatch, tmp_path)
    original_deadline = e157.deadline

    def final_timeout(started, limit_s, operation):
        if operation == "manifest publication":
            raise TimeoutError("generated final publication timeout")
        return original_deadline(started, limit_s, operation)

    monkeypatch.setattr(e157, "deadline", final_timeout)
    failed = tmp_path / "failed"
    with pytest.raises(TimeoutError, match="final publication"):
        e157.run(Path("freeze"), "sha", Path("review"), "sha", failed)
    failed_result = json.loads((failed / "result.json").read_text())
    assert failed_result["accepted"] is False
    failed_manifest = json.loads((failed / "output-manifest.json").read_text())
    for name, record in failed_manifest["files"].items():
        assert e157.sha256(failed / name) == record["sha256"]
    assert body.closed is True


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        (
            "integration",
            {
                "integration_complete": False,
                "state_captured": True,
                "state_copy_complete": False,
                "shadow_forward_complete": False,
                "observer_returned": False,
                "diagnostics_returned": False,
            },
        ),
        (
            "shadow_forward",
            {
                "integration_complete": True,
                "state_captured": True,
                "state_copy_complete": True,
                "shadow_forward_complete": False,
                "observer_returned": False,
                "diagnostics_returned": False,
            },
        ),
        (
            "observer",
            {
                "integration_complete": True,
                "state_captured": True,
                "state_copy_complete": True,
                "shadow_forward_complete": True,
                "observer_returned": False,
                "diagnostics_returned": False,
            },
        ),
        (
            "diagnostics",
            {
                "integration_complete": True,
                "state_captured": True,
                "state_copy_complete": True,
                "shadow_forward_complete": True,
                "observer_returned": True,
                "diagnostics_returned": False,
            },
        ),
    ],
)
def test_generated_native_failure_stages_preserve_attempt_and_available_raw(
    monkeypatch, tmp_path, stage, expected
):
    install_generated_full_run(monkeypatch, tmp_path, failure_stage=stage)
    output = tmp_path / "failed-stage"
    with pytest.raises(RuntimeError, match=stage.replace("_", " ")):
        e157.run(Path("freeze"), "sha", Path("review"), "sha", output)
    with np.load(output / "native-replay.npz") as raw:
        assert len(raw["native_index"]) == 1
        assert raw["native_index"].tolist() == [0]
        assert raw["step_requested"].tolist() == [True]
        for key, value in expected.items():
            assert raw[key].tolist() == [value]
        assert np.isfinite(raw["qpos"][0]).all()
        assert np.isfinite(raw["qvel"][0]).all()
        assert np.isfinite(raw["ctrl"][0]).all()
        if stage == "integration":
            assert raw["qpos"][0, 0] == 17.0
        if stage == "diagnostics":
            assert np.isfinite(raw["rotation_world"][0]).all()
            assert np.isfinite(raw["force_world_bw"][0]).all()
    with np.load(output / "observed-contacts.npz") as contacts:
        assert len(contacts["contact_id"]) == (1 if stage == "diagnostics" else 0)
    ledger = json.loads((output / "ledger.json").read_text())
    assert ledger["scientific_physics_steps_requested"] == 1
    assert ledger["scientific_physics_steps_advanced"] == (
        0 if stage == "integration" else 1
    )


def test_returned_diagnostics_survive_later_nonfoot_failure(monkeypatch, tmp_path):
    body = install_generated_full_run(monkeypatch, tmp_path)

    def fail_nonfoot(_self):
        raise RuntimeError("generated nonfoot observation failure")

    monkeypatch.setattr(type(body), "nonfoot_contact", fail_nonfoot)
    output = tmp_path / "failed-nonfoot"
    with pytest.raises(RuntimeError, match="nonfoot observation failure"):
        e157.run(Path("freeze"), "sha", Path("review"), "sha", output)
    with np.load(output / "native-replay.npz") as raw:
        np.testing.assert_array_equal(raw["normal_load_bw"], np.full((1, 6), 0.1))
        np.testing.assert_array_equal(raw["slip_mm_s"], np.zeros((1, 6)))
        np.testing.assert_array_equal(raw["slip_valid"], np.ones((1, 6), dtype=bool))
        assert raw["diagnostics_returned"].tolist() == [True]
        assert raw["nonfoot_returned"].tolist() == [False]
        assert raw["observation_complete"].tolist() == [False]
        assert raw["source_verified"].tolist() == [False]
    ledger = json.loads((output / "ledger.json").read_text())
    assert ledger["diagnostic_calls_returned"] == 1
    assert ledger["nonfoot_calls_requested"] == 1
    assert ledger["nonfoot_calls_returned"] == 0


def test_generated_primary_survives_cleanup_failure(monkeypatch, tmp_path):
    install_generated_full_run(
        monkeypatch,
        tmp_path,
        failure_stage="integration",
        cleanup_failure=True,
    )
    output = tmp_path / "failed-cleanup"
    with pytest.raises(RuntimeError, match="integration") as caught:
        e157.run(Path("freeze"), "sha", Path("review"), "sha", output)
    assert "cleanup failure" not in str(caught.value)
    result = json.loads((output / "result.json").read_text())
    assert result["error"]["message"] == "generated integration failure"
    assert any("body_close" in item for item in result["secondary_errors"])


def test_saved_native_time_requires_exact_parity_below_clock_tolerance(
    monkeypatch, tmp_path
):
    install_generated_full_run(monkeypatch, tmp_path, native_time_offset=1e-10)
    output = tmp_path / "time-mismatch"
    with pytest.raises(ValueError, match="native replay mismatch at 0: time_s"):
        e157.run(Path("freeze"), "sha", Path("review"), "sha", output)
    with np.load(output / "native-replay.npz") as raw:
        assert raw["integration_complete"].tolist() == [True]
        assert raw["observation_complete"].tolist() == [True]
        assert raw["source_verified"].tolist() == [False]
        assert abs(raw["time_s"][0] - 1.0001) < e157.CLOCK_ATOL


def test_cli_help_and_wrong_freeze_fail_before_output(tmp_path):
    source = e157.ROOT / e157.OWNED[0]
    help_run = subprocess.run(
        [sys.executable, str(source), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_run.returncode == 0 and "--source-review-sha256" in help_run.stdout
    output = tmp_path / "forbidden"
    bad = subprocess.run(
        [
            sys.executable,
            str(source),
            "--freeze",
            str(
                e157.ROOT
                / "verification/banc-natural-gait-m3-20260919/e157-native-contact-decomposition-freeze.json"
            ),
            "--freeze-sha256",
            "wrong",
            "--source-review",
            str(tmp_path / "missing"),
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
    assert bad.returncode != 0
    assert "Exact E157 freeze required" in bad.stderr
    assert not output.exists()


def test_owned_paths_are_only_new_e157_files():
    assert e157.OWNED == (
        "tools/analyze_c_banc_m3_native_contact_decomposition.py",
        "tests/test_c_banc_native_contact_decomposition.py",
    )
