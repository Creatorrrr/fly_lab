"""Generated-only E156 tests; no original state, model, physics, or NPZ is opened."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import banc_m3_monotone_recruitment as monotone
from tools import banc_m3_substep_recruitment as substep
from tools import verify_c_banc_m3_substep_body as e156


def test_helper_registry_preserves_attempt_before_failure():
    native = generated_native_state()
    rows = generated_rows()
    registry = e156.new_helper_registry()
    ledger = {
        "helper_intervals_requested": 0,
        "helper_intervals_returned": 0,
        "helper_intervals_completed": 0,
    }

    class FailingHelper:
        calls = 0

        def compute(self, *_args, **_kwargs):
            self.calls += 1
            raise RuntimeError("generated helper failure")

    adapter = SimpleNamespace(
        bank=np.zeros((391, 3)),
        activation=np.zeros(96),
        original=SimpleNamespace(activation=np.zeros(96)),
    )
    with pytest.raises(RuntimeError, match="generated helper"):
        e156.reconstruct_endpoints(
            FailingHelper(), adapter, rows, native, registry, ledger
        )
    arrays = e156.helper_arrays(registry)
    assert arrays["tick"].tolist() == [1]
    assert arrays["complete"].tolist() == [False]
    assert np.isfinite(arrays["bank_before"]).all()
    assert ledger == {
        "helper_intervals_requested": 1,
        "helper_intervals_returned": 0,
        "helper_intervals_completed": 0,
    }


def generated_native_state() -> dict[str, np.ndarray]:
    return {
        "schema": np.array(e156.STATE_SCHEMA),
        "control_tick": np.arange(2001, dtype=np.int64),
        "network_tick": 20 * np.arange(2001, dtype=np.int64),
        "physics_time_s": np.arange(2001, dtype=np.float64) * 0.005,
        "complete": np.ones(2001, dtype=bool),
        "bank": np.zeros((2001, 391, 3), dtype=np.float64),
        "bank_available": np.ones(2001, dtype=bool),
        "activation": np.zeros((2001, 96), dtype=np.float64),
        "activation_available": np.ones(2001, dtype=bool),
        "original_activation": np.zeros((2001, 96), dtype=np.float64),
        "original_activation_available": np.ones(2001, dtype=bool),
        "motor_rate": np.zeros((2001, 391), dtype=np.float64),
        "motor_rate_available": np.ones(2001, dtype=bool),
    }


def generated_rows() -> list[dict]:
    rows = []
    for tick in range(2001):
        angle = 1.0 + 0.2 * np.sin(tick * 0.1)
        points = np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [np.cos(angle), np.sin(angle), 0.0],
            ]
        )
        ui_points = points @ e156.UI_FROM_PHYSICS.T
        legs = {leg: ui_points.tolist() for leg in e156.LEGS}
        physical_position = np.array([tick / 2000, 0.0, 0.0])
        rows.append(
            {
                "control_tick": tick,
                "time_s": tick * 0.005,
                "motor_rates": [0.0] * 391,
                "joint_targets_rad": [0.0] * 42,
                "qpos": [tick / 2000, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "qvel": [0.0] * 6,
                "ctrl": [0.0] * 48,
                "adhesion": [False] * 6,
                "tendon_inputs": {},
                "fault": None,
                "neural_request": None,
                "body": {
                    "position": (e156.UI_FROM_PHYSICS @ physical_position).tolist(),
                    "basis": (e156.UI_FROM_PHYSICS @ np.eye(3)).tolist(),
                    "legs": legs,
                },
            }
        )
    return rows


def generated_parameters() -> dict[str, np.ndarray]:
    membership = np.arange(391, dtype=np.int64) % 96
    return {
        "tau": np.array([0.005, 0.01, 0.04], dtype=np.float64),
        "kappa": np.zeros(391, dtype=np.float64),
        "slopes": np.zeros((96, 391, 3, 2), dtype=np.float64),
        "membership": membership,
        "motor_ids": np.arange(391, dtype=np.int64),
    }


def generated_adapter_snapshot():
    parameters = generated_parameters()
    membership = parameters["membership"]
    counts = np.bincount(membership, minlength=96)
    pool = np.zeros((96, 391), dtype=np.float64)
    pool[membership, np.arange(391)] = 1 / counts[membership]
    slopes = np.zeros((96, 391, 3, 2), dtype=np.float64)
    slopes[:, :, 1, 0] = pool
    slopes[:, :, 1, 1] = pool
    parameters["slopes"] = slopes
    calibration = {
        "moment": np.zeros((42, 96), dtype=np.float64),
        "rest": np.zeros(42, dtype=np.float64),
        "pool": pool,
        "motor_ids": parameters["motor_ids"].copy(),
        "ltm": np.arange(6, dtype=np.int64),
    }
    groups = [
        {
            "leg": e156.LEGS[muscle // 16],
            "muscle": f"m{muscle}",
            "columns": np.flatnonzero(membership == muscle).tolist(),
        }
        for muscle in range(96)
    ]
    roster = [f"motor-{index}" for index in range(391)]
    original = SimpleNamespace(identity="a" * 64)
    identity = monotone._identity(
        original,
        calibration,
        groups,
        parameters["tau"],
        parameters["kappa"],
        parameters["slopes"],
        membership,
        roster,
    )
    fixed = np.arange(6, dtype=np.int64)
    minimum = np.min(slopes[membership, np.arange(391)], axis=-1).sum(axis=1)
    budget = np.bincount(membership, weights=minimum, minlength=96)
    snapshot_parameters = {
        "tau": parameters["tau"].copy(),
        "kappa": parameters["kappa"].copy(),
        "slopes": parameters["slopes"].copy(),
        "membership": membership.copy(),
        "motor_roster": roster,
        "fixed_muscles": fixed,
        "lower_budget": budget.copy(),
        "upper_budget": budget.copy(),
        "motor_minimum_slope": minimum,
    }
    saved = {
        "adapter": {
            "schema": monotone.ADAPTER_SCHEMA,
            "hash": identity,
            "parameters": snapshot_parameters,
            "calibration": calibration,
            "groups": groups,
            "original": {"hash": original.identity, "activation": np.zeros(96)},
            "bank": np.zeros((391, 3)),
            "activation": np.zeros(96),
            "last_motor": [],
        }
    }
    return saved, parameters


def test_helper_adapter_view_preserves_exact_identity_and_has_no_decode():
    saved, parameters = generated_adapter_snapshot()
    adapter = e156.make_helper_adapter(saved, parameters, np.zeros(42))
    helper = substep.CausalSubstepRecruitment(adapter)
    result = helper.compute(
        np.zeros((391, 3)),
        np.zeros(96),
        np.zeros(391),
        np.zeros(42),
        motor_cut=False,
        dt=0.005,
        physics_dt=0.0001,
    )
    assert result.joint_targets.shape == (50, 42)
    assert adapter.identity == saved["adapter"]["hash"]
    assert not hasattr(adapter, "decode")
    parameters["kappa"][0] = 1
    with pytest.raises(ValueError, match="Checkpoint parameter"):
        e156.make_helper_adapter(saved, parameters, np.zeros(42))


def test_all_2000_helper_intervals_complete_and_only_201_to_400_are_selected():
    native = generated_native_state()
    rows = generated_rows()

    class FakeResult:
        def __init__(self):
            self.joint_targets = np.zeros((50, 42))
            self.endpoint_error = {
                "bank_max_abs": 0.0,
                "fixed_activation_max_abs": 0.0,
                "activation_max_abs": 0.0,
                "target_max_abs": 0.0,
            }

    class Helper:
        def compute(self, *_args, **_kwargs):
            return FakeResult()

    adapter = SimpleNamespace(
        bank=np.zeros((391, 3)),
        activation=np.zeros(96),
        original=SimpleNamespace(activation=np.zeros(96)),
    )
    registry = e156.new_helper_registry()
    ledger = {
        "helper_intervals_requested": 0,
        "helper_intervals_returned": 0,
        "helper_intervals_completed": 0,
    }
    trajectory = e156.reconstruct_endpoints(
        Helper(), adapter, rows, native, registry, ledger
    )
    arrays = e156.helper_arrays(registry)
    assert trajectory.shape == (200, 50, 42)
    assert ledger["helper_intervals_completed"] == 2000
    assert arrays["selected"].sum() == 200
    assert arrays["tick"][arrays["selected"]][[0, -1]].tolist() == [201, 400]


def generated_trace(rows: list[dict]) -> dict:
    return {
        "schema": e156.TRACE_SCHEMA,
        "body_hash": "body",
        "E150_provenance": {"numeric_protocol": "unchanged E141 native body protocol"},
        "samples": rows,
    }


def test_saved_input_full_tick_rate_and_clock_contract():
    rows = generated_rows()
    native = generated_native_state()
    parameters = generated_parameters()
    assert (
        e156.validate_saved_inputs(generated_trace(rows), native, parameters, "body")
        is rows
    )
    native["physics_time_s"][777] += 1e-8
    e156.validate_saved_inputs(generated_trace(rows), native, parameters, "body")
    native["physics_time_s"][777] += 2e-8
    with pytest.raises(ValueError, match="physical clocks"):
        e156.validate_saved_inputs(generated_trace(rows), native, parameters, "body")


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("tick", "tick/index"),
        ("network", "network ticks"),
        ("rate", "motor rate"),
        ("availability", "Incomplete"),
        ("target", "joint target"),
    ],
)
def test_saved_input_fail_closed_mutations(mutation, message):
    rows = generated_rows()
    native = generated_native_state()
    parameters = generated_parameters()
    if mutation == "tick":
        rows[10]["control_tick"] = 11
    elif mutation == "network":
        native["network_tick"][10] += 1
    elif mutation == "rate":
        rows[10]["motor_rates"][3] = 1
    elif mutation == "availability":
        native["bank_available"][10] = False
    else:
        rows[10]["joint_targets_rad"] = [0.0]
    with pytest.raises(ValueError, match=message):
        e156.validate_saved_inputs(generated_trace(rows), native, parameters, "body")


def baseline_arrays():
    native = {
        "time_s": 1.0 + (np.arange(10000) + 1) * 0.0001,
        "control_tick": 201 + np.arange(10000) // 50,
        "rotation_world": np.broadcast_to(np.eye(3), (10000, 3, 3)).copy(),
        "heading_horizontal": np.broadcast_to([1.0, 0.0, 0.0], (10000, 3)).copy(),
        "force_world_bw": np.zeros((10000, 6, 3)),
        "force_forward_bw": np.zeros((10000, 6)),
        "normal_load_bw": np.full((10000, 6), 0.1),
        "slip_mm_s": np.full((10000, 6), 1.0),
        "slip_valid": np.ones((10000, 6), dtype=bool),
        "nonfoot_contact": np.zeros(10000, dtype=bool),
        "upright": np.ones(10000),
    }
    controls = {
        "tick": np.arange(200, 401, dtype=np.int64),
        "time_s": np.arange(200, 401) * 0.005,
        "qpos": np.column_stack((np.linspace(0, 1, 201), np.zeros((201, 6)))),
        "qvel": np.zeros((201, 6)),
        "ctrl": np.zeros((201, 48)),
        "verified": np.ones(201, dtype=bool),
    }
    context = {"body_hash": "body"}
    return native, controls, context


def test_baseline_requires_exact_shapes_masks_and_window():
    native, controls, context = baseline_arrays()
    e156.validate_baseline(native, controls, context, "body")
    native["slip_valid"][5, 2] = False
    with pytest.raises(ValueError, match="slip validity"):
        e156.validate_baseline(native, controls, context, "body")


def test_knee_geometry_and_movement_scalar_formulas():
    angles = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    legs = {
        leg: [
            [-1, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [np.cos(angle), np.sin(angle), 0],
        ]
        for leg, angle in zip(e156.LEGS, angles, strict=True)
    }
    np.testing.assert_allclose(e156.knee_angles_from_legs(legs), angles)
    positions = np.column_stack((np.linspace(0, 1, 201), np.zeros((201, 2))))
    rotations = np.broadcast_to(np.eye(3), (201, 3, 3)).copy()
    knee = 1 + np.sin(np.linspace(0, 2 * np.pi, 201))[:, None] * np.ones((1, 6))
    load = np.full((10000, 6), 0.1)
    slip = np.tile(np.arange(1, 7, dtype=float), (10000, 1))
    result = e156.movement_metrics(
        positions,
        rotations,
        knee,
        load,
        slip,
        np.ones_like(load, bool),
        np.zeros(10000, bool),
        np.ones(10000),
    )
    assert result["signed_forward_mm"] == pytest.approx(1)
    assert result["net_path_ratio"] == pytest.approx(1)
    assert result["load_weighted_slip_mm_s"] == pytest.approx(3.5)
    assert result["continuity"]["forward_fraction"] == 1
    assert result["continuity"]["longest_forward_bout_s"] == pytest.approx(1)
    missing = slip.copy()
    missing[10, 0] = np.nan
    missing_valid = np.ones_like(load, bool)
    missing_valid[10, 0] = False
    with pytest.raises(ValueError, match="Finite aligned|load-bearing"):
        e156.movement_metrics(
            positions,
            rotations,
            knee,
            load,
            missing,
            missing_valid,
            np.zeros(10000, bool),
            np.ones(10000),
        )


def test_signed_forward_uses_full_physical_basis_not_horizontal_normalization():
    angle = np.deg2rad(60)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, -np.sin(angle)],
            [0.0, 1.0, 0.0],
            [np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    rotations = np.broadcast_to(rotation, (201, 3, 3)).copy()
    forward = rotation[:, 0]
    positions = np.arange(201)[:, None] * forward[None, :] / 200
    knee = np.linspace(0.5, 1.5, 201)[:, None] * np.ones((1, 6))
    load = np.full((10000, 6), 0.1)
    result = e156.movement_metrics(
        positions,
        rotations,
        knee,
        load,
        np.ones_like(load),
        np.ones_like(load, bool),
        np.zeros(10000, bool),
        np.ones(10000),
    )
    assert result["signed_forward_mm"] == pytest.approx(1.0)
    assert result["horizontal_path_mm"] == pytest.approx(np.cos(angle))


def test_effect_gates_are_fixed_and_effect_failure_is_valid_result():
    widths = {leg: 1.0 for leg in e156.LEGS}
    baseline = {
        "load_weighted_slip_mm_s": 4.0,
        "signed_forward_mm": 1.0,
        "net_path_ratio": 0.8,
        "knee_width_rad": widths,
    }
    candidate = {
        "load_weighted_slip_mm_s": 3.2,
        "signed_forward_mm": 0.9,
        "net_path_ratio": 0.75,
        "knee_width_rad": {leg: 0.9 for leg in e156.LEGS},
        "nonfoot_samples": 0,
        "minimum_upright": 0.9,
    }
    decision = {
        "max_slip_ratio": 0.8,
        "min_forward_ratio": 0.9,
        "max_net_path_ratio_drop": 0.05,
        "min_knee_width_ratio": 0.9,
        "min_upright": 0.9,
        "nonfoot_allowed": 0,
    }
    result = e156.compare_effect(candidate, baseline, decision)
    assert result["advance_to_closed_loop"] is True
    candidate["load_weighted_slip_mm_s"] = 3.2000000001
    result = e156.compare_effect(candidate, baseline, decision)
    assert result["advance_to_closed_loop"] is False
    assert result["gates"]["slip"] is False


class FakeData:
    def __init__(self):
        self.qpos = np.array([0.0, 0.0, 0.0, 1, 0, 0, 0], dtype=float)
        self.qvel = np.zeros(6)
        self.ctrl = np.zeros(48)
        self.time = 1.0
        self.warning = SimpleNamespace(number=np.zeros(2, dtype=int))


class FakeMJ:
    mjtState = SimpleNamespace(mjSTATE_INTEGRATION=1)

    @staticmethod
    def MjData(_model):
        return FakeData()

    @staticmethod
    def mj_stateSize(_model, _state):
        return 62

    @staticmethod
    def mj_getState(_model, data, state, _kind):
        state[:] = 0
        state[:7] = data.qpos
        state[7:13] = data.qvel
        state[13:61] = data.ctrl
        state[61] = data.time

    @staticmethod
    def mj_setState(_model, data, state, _kind):
        data.qpos[:] = state[:7]
        data.qvel[:] = state[7:13]
        data.ctrl[:] = state[13:61]
        data.time = state[61]

    @staticmethod
    def mj_forward(_model, _data):
        return None


class FakeBody:
    def __init__(self, *, fail_step: int | None = None):
        self.m = SimpleNamespace(nq=7, nv=6, nu=48)
        self.mj = FakeMJ
        self.d = FakeData()
        self.act_ids = np.arange(42)
        self.fault = None
        self.push_left = 0.0
        self.push_force = np.zeros(3)
        self.steps = 0
        self.fail_step = fail_step
        self.model_hash = "body"
        self.neutral = np.zeros(42)
        self.t0 = 0.2
        self.m.opt = SimpleNamespace(timestep=e156.PHYSICS_DT)
        self._force_maps = {"feet": (np.array([-1, 0]), 30)}
        self.support_geom_ids = {0}
        self.weight0 = 1.0
        self.closed = False

    def physics_time(self):
        return self.d.time

    def _step_physics(self):
        self.steps += 1
        if self.fail_step == self.steps:
            raise RuntimeError("generated integration failure")
        self.d.time += e156.PHYSICS_DT
        self.d.qpos[0] += 1 / 10000

    def step_joint_targets(self, target, _adhesion, **_kwargs):
        self.d.ctrl[:42] = target
        for _ in range(50):
            self._step_physics()

    def pose(self):
        return self.d.qpos[:3].copy(), np.eye(3), np.zeros(6)

    def knee_kinematics(self, **_kwargs):
        phase = self.d.time * 2 * np.pi
        return {"flexion_rad": np.ones(6) + 0.2 * np.sin(phase)}

    def nonfoot_contact(self):
        return False

    def snapshot(self):
        return {"qpos": self.d.qpos.copy(), "qvel": self.d.qvel.copy()}

    def close(self):
        self.closed = True


def fake_contact(_body):
    return {"normal_load_bw": [0.1] * 6, "slip_mm_s": [0.5] * 6}


def fake_capture(session):
    position, rotation, _ = session.body.pose()
    angle = 1.0 + 0.2 * np.sin(float(session.body.physics_time()) * 2 * np.pi)
    points = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [np.cos(angle), np.sin(angle), 0.0],
        ]
    )
    ui_points = points @ e156.UI_FROM_PHYSICS.T
    return {
        "control_tick": session.control_tick,
        "fault": None,
        "body": {
            "position": (e156.UI_FROM_PHYSICS @ position).tolist(),
            "basis": (e156.UI_FROM_PHYSICS @ rotation).tolist(),
            "legs": {leg: ui_points.tolist() for leg in e156.LEGS},
        },
    }


def fake_replay(row, *, implicit_request=None):
    del implicit_request
    return (
        np.asarray(row["joint_targets_rad"]),
        np.asarray(row["adhesion"]),
        {
            "push_left_s": 0.0,
            "push_force": np.zeros(3),
            "step_kwargs": {"tendon_inputs": row["tendon_inputs"], "interpolate": True},
        },
    )


def minimal_rows():
    rows = [
        {
            "control_tick": tick,
            "joint_targets_rad": [0.0] * 42,
            "adhesion": [False] * 6,
            "tendon_inputs": {},
            "ctrl": [0.0] * 48,
        }
        for tick in range(401)
    ]
    return rows


def body_ledger():
    return {
        key: 0
        for key in (
            "scientific_steps_requested",
            "scientific_steps_completed",
            "scientific_samples_observed",
            "shadow_forward_requested",
            "shadow_forward_completed",
            "step_joint_targets_requested",
            "step_joint_targets_completed",
            "control_capture_requested",
            "control_capture_completed",
        )
    }


def test_generated_full_body_run_uses_50_substeps_and_restores_method(tmp_path):
    body = FakeBody()
    original_function = body._step_physics.__func__
    shadow = SimpleNamespace(
        d=FakeData(),
        pose=lambda: (np.zeros(3), np.eye(3), None),
        nonfoot_contact=lambda: False,
    )
    trajectory = np.zeros((200, 50, 42))
    registry = e156.new_body_registry(7, 6, 48)
    ledger = body_ledger()
    state = generated_native_state()
    e156.execute_intervention(
        body=body,
        shadow=shadow,
        rows=minimal_rows(),
        native_state=state,
        trajectory=trajectory,
        body_registry=registry,
        ledger=ledger,
        journal_path=tmp_path / "journal.jsonl",
        contact_diagnostics=fake_contact,
        capture_control=fake_capture,
        replay_command=fake_replay,
        implicit_request=None,
    )
    assert body.steps == 10000
    assert body._step_physics.__func__ is original_function
    assert ledger["scientific_steps_completed"] == 10000
    assert ledger["control_capture_completed"] == 200
    applied, native, attempts, controls = e156.body_arrays(registry)
    assert applied["ctrl"].shape == (10000, 48)
    assert np.all(applied["integration_complete"])
    assert np.all(applied["observation_complete"])
    assert native["time_s"].shape == (10000,)
    assert np.all(attempts["complete"])
    assert controls["tick"].tolist() == list(range(201, 401))
    journal = (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(journal) == 30000
    assert json.loads(journal[0])["integration_complete"] is False


def test_integration_failure_keeps_before_call_raw_and_known_counts(tmp_path):
    body = FakeBody(fail_step=17)
    original_function = body._step_physics.__func__
    shadow = SimpleNamespace(
        d=FakeData(),
        pose=lambda: (np.zeros(3), np.eye(3), None),
        nonfoot_contact=lambda: False,
    )
    registry = e156.new_body_registry(7, 6, 48)
    ledger = body_ledger()
    with pytest.raises(RuntimeError, match="integration failure"):
        e156.execute_intervention(
            body=body,
            shadow=shadow,
            rows=minimal_rows(),
            native_state=generated_native_state(),
            trajectory=np.zeros((200, 50, 42)),
            body_registry=registry,
            ledger=ledger,
            journal_path=tmp_path / "journal.jsonl",
            contact_diagnostics=fake_contact,
            capture_control=fake_capture,
            replay_command=fake_replay,
            implicit_request=None,
        )
    assert body._step_physics.__func__ is original_function
    applied, native, attempts, controls = e156.body_arrays(registry)
    assert len(applied["tick"]) == 17
    assert applied["integration_complete"].sum() == 16
    assert len(native["tick"]) == 16
    assert len(attempts["tick"]) == 16
    assert len(controls["tick"]) == 0
    assert ledger["scientific_steps_requested"] == 17
    assert ledger["scientific_steps_completed"] == 16


def test_observation_failure_preserves_completed_integration(tmp_path):
    body = FakeBody()
    shadow = SimpleNamespace(
        d=FakeData(),
        pose=lambda: (np.zeros(3), np.eye(3), None),
        nonfoot_contact=lambda: False,
    )

    def fail_contact(_body):
        raise RuntimeError("generated observation failure")

    registry = e156.new_body_registry(7, 6, 48)
    ledger = body_ledger()
    with pytest.raises(RuntimeError, match="observation failure"):
        e156.execute_intervention(
            body=body,
            shadow=shadow,
            rows=minimal_rows(),
            native_state=generated_native_state(),
            trajectory=np.zeros((200, 50, 42)),
            body_registry=registry,
            ledger=ledger,
            journal_path=tmp_path / "journal.jsonl",
            contact_diagnostics=fail_contact,
            capture_control=fake_capture,
            replay_command=fake_replay,
            implicit_request=None,
        )
    applied, native, attempts, _ = e156.body_arrays(registry)
    assert applied["integration_complete"].tolist() == [True]
    assert applied["observation_complete"].tolist() == [False]
    assert len(native["tick"]) == 0
    assert len(attempts["tick"]) == 0
    assert ledger["scientific_steps_completed"] == 1
    assert ledger["shadow_forward_completed"] == 1


def test_nan_observation_is_saved_as_incomplete_raw_before_rejection(tmp_path):
    body = FakeBody()
    shadow = SimpleNamespace(
        d=FakeData(),
        pose=lambda: (np.zeros(3), np.eye(3), None),
        nonfoot_contact=lambda: False,
    )

    def nan_contact(_body):
        return {"normal_load_bw": [np.nan] + [0.1] * 5, "slip_mm_s": [1.0] * 6}

    registry = e156.new_body_registry(7, 6, 48)
    ledger = body_ledger()
    with pytest.raises(ValueError, match="Finite native observation"):
        e156.execute_intervention(
            body=body,
            shadow=shadow,
            rows=minimal_rows(),
            native_state=generated_native_state(),
            trajectory=np.zeros((200, 50, 42)),
            body_registry=registry,
            ledger=ledger,
            journal_path=tmp_path / "journal.jsonl",
            contact_diagnostics=nan_contact,
            capture_control=fake_capture,
            replay_command=fake_replay,
            implicit_request=None,
        )
    applied, native, attempts, _ = e156.body_arrays(registry)
    assert applied["integration_complete"].tolist() == [True]
    assert applied["observation_complete"].tolist() == [False]
    assert len(native["tick"]) == 1
    assert np.isnan(native["normal_load_bw"][0, 0])
    assert native["observation_complete"].tolist() == [False]
    assert attempts["complete"].tolist() == [False]
    assert ledger["scientific_samples_observed"] == 0


def test_nonjoint_mutation_and_wrong_endpoint_fail_closed(tmp_path):
    rows = minimal_rows()
    rows[201]["ctrl"][47] = 1.0
    body = FakeBody()
    shadow = SimpleNamespace(
        d=FakeData(),
        pose=lambda: (np.zeros(3), np.eye(3), None),
        nonfoot_contact=lambda: False,
    )
    with pytest.raises(ValueError, match="Non-joint control|Final full ctrl"):
        e156.execute_intervention(
            body=body,
            shadow=shadow,
            rows=rows,
            native_state=generated_native_state(),
            trajectory=np.zeros((200, 50, 42)),
            body_registry=e156.new_body_registry(7, 6, 48),
            ledger=body_ledger(),
            journal_path=tmp_path / "journal.jsonl",
            contact_diagnostics=fake_contact,
            capture_control=fake_capture,
            replay_command=fake_replay,
            implicit_request=None,
        )
    rows = minimal_rows()
    rows[201]["joint_targets_rad"][0] = 1
    with pytest.raises(ValueError, match="Trajectory endpoint"):
        e156.execute_intervention(
            body=FakeBody(),
            shadow=shadow,
            rows=rows,
            native_state=generated_native_state(),
            trajectory=np.zeros((200, 50, 42)),
            body_registry=e156.new_body_registry(7, 6, 48),
            ledger=body_ledger(),
            journal_path=tmp_path / "journal2.jsonl",
            contact_diagnostics=fake_contact,
            capture_control=fake_capture,
            replay_command=fake_replay,
            implicit_request=None,
        )


def test_raw_and_manifest_include_partial_files(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    helper = e156.new_helper_registry()
    native = generated_native_state()
    e156._append_helper_attempt(helper, 1, native, generated_rows()[1])
    body = e156.new_body_registry(7, 6, 48)
    e156.save_raw(output, helper, np.empty((0, 50, 42)), body)
    e156.atomic_json(output / "result.json", {"accepted": False})
    manifest = e156.output_manifest(output)
    assert set(manifest["files"]) == {
        "applied-substeps.npz",
        "candidate-controls.npz",
        "candidate-native.npz",
        "candidate-trace.json",
        "helper-endpoints.npz",
        "joint-trajectory.npz",
        "native-attempts.npz",
        "result.json",
    }
    for name, record in manifest["files"].items():
        assert e156.sha256(output / name) == record["sha256"]


def test_source_gate_requires_exact_two_source_hashes(tmp_path, monkeypatch):
    review = tmp_path / "review.json"
    value = {
        "accepts_single_substep_body": True,
        "blocking_defects": [],
        "freeze_sha256": e156.FREEZE_SHA,
        "original_processes_allowed": 1,
        "source_sha256": {name: e156.sha256(e156.ROOT / name) for name in e156.OWNED},
    }
    review.write_text(json.dumps(value), encoding="utf-8")
    assert e156.validate_source_gate(review, e156.sha256(review)) == value
    value["source_sha256"].pop(e156.OWNED[1])
    review.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="two-source"):
        e156.validate_source_gate(review, e156.sha256(review))


def test_run_rejects_source_gate_before_any_output_or_model(tmp_path, monkeypatch):
    monkeypatch.setattr(e156, "load_freeze", lambda *_args: {"unused": True})

    def reject(*_args):
        raise ValueError("generated source gate rejection")

    monkeypatch.setattr(e156, "validate_source_gate", reject)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="source gate"):
        e156.run(tmp_path / "freeze", "sha", tmp_path / "review", "sha", output)
    assert not output.exists()


def test_generated_complete_run_writes_consistent_result_and_manifest(
    tmp_path, monkeypatch
):
    rows = generated_rows()
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps({"generated": True}), encoding="utf-8")
    native, controls, context = baseline_arrays()
    controls["qpos"] = np.asarray([rows[tick]["qpos"] for tick in range(200, 401)])
    context.update(
        body_parameters={"generated": True},
        feet_map=[-1, 0],
        support_geom_ids=[0],
        weight0=1.0,
    )
    state = generated_native_state()
    parameters = generated_parameters()
    saved = {
        "control_tick": 200,
        "network": {"tick": 4000},
        "body": {"modelHash": "body"},
        "adapter": {},
    }
    freeze = {
        "trace": "trace",
        "recruitment_state": "state",
        "parameter_bundle": "parameters",
        "baseline_native": "baseline-native",
        "baseline_controls": "baseline-controls",
        "baseline_context": "baseline-context",
        "checkpoint": "checkpoint",
        "body_hash": "body",
        "input_sha256": {},
        "reused_source_sha256": {},
        "preservation_sha256": {},
        "resource_limits": {
            "internal_s": 65,
            "external_s": 75,
            "supervisor_s": 85,
            "raw_bytes": 536870912,
            "free_bytes": 1,
        },
        "decision": {
            "max_slip_ratio": 0.8,
            "min_forward_ratio": 0.9,
            "max_net_path_ratio_drop": 0.05,
            "min_knee_width_ratio": 0.9,
            "min_upright": 0.9,
            "nonfoot_allowed": 0,
        },
        "prior_actual_original_s": 2535.0,
        "prior_conservative_s": 2536,
        "previous_allocation_s": 2567,
        "prospective_cumulative_s": 2621,
    }
    mapping = {
        "trace": trace_path,
        "state": tmp_path / "state.npz",
        "parameters": tmp_path / "parameters.npz",
        "baseline-native": tmp_path / "baseline-native.npz",
        "baseline-controls": tmp_path / "baseline-controls.npz",
        "baseline-context": tmp_path / "baseline-context.json",
        "checkpoint": tmp_path / "checkpoint",
    }
    for key, path in mapping.items():
        if key != "trace":
            path.touch() if path.suffix else path.mkdir()
    mapping["baseline-context"].write_text(json.dumps(context), encoding="utf-8")

    def fake_load_npz(path):
        if path == mapping["state"]:
            return state
        if path == mapping["parameters"]:
            return parameters
        if path == mapping["baseline-native"]:
            return native
        if path == mapping["baseline-controls"]:
            return controls
        raise AssertionError(path)

    body = FakeBody()
    body.d.qpos[:] = rows[200]["qpos"]

    import tools.analyze_c_banc_m3_signed_contact as e154
    import tools.banc_m3_substep_recruitment as helper_module
    from flylab.c import storage, walking_trace

    monkeypatch.setattr(e156, "load_freeze", lambda *_args: freeze)
    monkeypatch.setattr(e156, "validate_source_gate", lambda *_args: {})
    monkeypatch.setattr(e156, "verify_hash_map", lambda *_args: None)
    monkeypatch.setattr(e156, "threadpool_info", list)
    monkeypatch.setattr(e156, "safe_path", lambda name: mapping[name])
    monkeypatch.setattr(e156, "load_npz", fake_load_npz)
    monkeypatch.setattr(e156, "validate_saved_inputs", lambda *_args: rows)
    monkeypatch.setattr(e156, "validate_baseline", lambda *_args: None)
    monkeypatch.setattr(
        e156,
        "make_helper_adapter",
        lambda *_args: SimpleNamespace(
            identity="a",
            original=SimpleNamespace(identity="b", activation=np.zeros(96)),
            bank=np.zeros((391, 3)),
            activation=np.zeros(96),
        ),
    )
    monkeypatch.setattr(e156, "validate_checkpoint_anchor", lambda *_args: None)

    def fake_reconstruct(
        _helper,
        _adapter,
        _rows,
        _native,
        registry,
        ledger,
        *,
        on_interval=None,
    ):
        ledger["helper_intervals_requested"] = 2000
        ledger["helper_intervals_returned"] = 2000
        ledger["helper_intervals_completed"] = 2000
        registry["selected_tick"].extend(range(201, 401))
        registry["selected_joint_target"].extend(
            np.zeros((200, 50, 42), dtype=np.float64)
        )
        if on_interval is not None:
            on_interval(2000)
        return np.zeros((200, 50, 42))

    monkeypatch.setattr(e156, "reconstruct_endpoints", fake_reconstruct)
    monkeypatch.setattr(
        helper_module,
        "CausalSubstepRecruitment",
        lambda adapter: SimpleNamespace(adapter=adapter),
    )
    monkeypatch.setattr(storage.StateStore, "load", staticmethod(lambda _path: saved))

    def save_state(path, _snapshot):
        path.mkdir()
        (path / "generated.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(storage.StateStore, "save", staticmethod(save_state))
    monkeypatch.setattr(
        e154, "validate_body_parameters", lambda _saved: {"generated": True}
    )
    monkeypatch.setattr(e154, "implicit_request_for_trace", lambda *_args: {})

    def construct(_saved, _observation, ledger):
        ledger["body_constructor_attempts"] = 1
        ledger["body_constructor_completed"] = 1
        ledger["settling_physics_steps"] = 2000
        return body

    monkeypatch.setattr(e154, "construct_body_safely", construct)
    monkeypatch.setattr(e154, "validate_settling_count", lambda *_args: 2000)
    monkeypatch.setattr(e154, "replay_command", fake_replay)
    monkeypatch.setattr(walking_trace, "capture_control", fake_capture)
    monkeypatch.setattr(walking_trace, "contact_diagnostics", fake_contact)
    output = tmp_path / "output"
    result = e156.run(
        tmp_path / "freeze.json", "freeze", tmp_path / "review.json", "review", output
    )
    assert result["accepted"] is True
    assert result["publication_complete"] is True
    assert result["candidate_adopted"] is False
    manifest = json.loads((output / "output-manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["result.json"]["sha256"] == e156.sha256(
        output / "result.json"
    )
    assert manifest["files"]["applied-substeps.jsonl"]["bytes"] > 0
    context = json.loads((output / "context.json").read_text(encoding="utf-8"))
    assert context["joint_actuator_ids"] == list(range(42))
    assert context["nonjoint_actuator_ids"] == list(range(42, 48))
    assert context["body_hash"] == body.model_hash
    assert context["anchor_time_s"] == 1.0
    ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["scientific_steps_completed"] == 10000
    assert ledger["network_models"] == ledger["neural_steps"] == 0

    failed_body = FakeBody(fail_step=17)
    failed_body.d.qpos[:] = rows[200]["qpos"]

    def failing_construct(_saved, _observation, ledger):
        ledger["body_constructor_attempts"] = 1
        ledger["body_constructor_completed"] = 1
        ledger["settling_physics_steps"] = 2000
        return failed_body

    monkeypatch.setattr(e154, "construct_body_safely", failing_construct)
    failed_output = tmp_path / "failed-output"
    with pytest.raises(RuntimeError, match="integration failure"):
        e156.run(
            tmp_path / "freeze-2.json",
            "freeze",
            tmp_path / "review-2.json",
            "review",
            failed_output,
        )
    failed_result = json.loads(
        (failed_output / "result.json").read_text(encoding="utf-8")
    )
    assert failed_result["accepted"] is False
    with np.load(failed_output / "applied-substeps.npz", allow_pickle=False) as raw:
        assert len(raw["tick"]) == 17
        assert raw["integration_complete"].sum() == 16
    failed_manifest = json.loads(
        (failed_output / "output-manifest.json").read_text(encoding="utf-8")
    )
    for name, record in failed_manifest["files"].items():
        assert e156.sha256(failed_output / name) == record["sha256"]

    timeout_body = FakeBody()
    timeout_body.d.qpos[:] = rows[200]["qpos"]

    def timeout_construct(_saved, _observation, ledger):
        ledger["body_constructor_attempts"] = 1
        ledger["body_constructor_completed"] = 1
        ledger["settling_physics_steps"] = 2000
        return timeout_body

    def final_timeout(_started, _limit, operation):
        if operation == "after final manifest publication":
            raise TimeoutError("generated final publication timeout")
        return 0.0

    monkeypatch.setattr(e154, "construct_body_safely", timeout_construct)
    monkeypatch.setattr(e156, "deadline", final_timeout)
    timeout_output = tmp_path / "timeout-output"
    with pytest.raises(TimeoutError, match="final publication"):
        e156.run(
            tmp_path / "freeze-3.json",
            "freeze",
            tmp_path / "review-3.json",
            "review",
            timeout_output,
        )
    timeout_result = json.loads(
        (timeout_output / "result.json").read_text(encoding="utf-8")
    )
    assert timeout_result["accepted"] is False
    assert timeout_result["publication_complete"] is False
    timeout_manifest = json.loads(
        (timeout_output / "output-manifest.json").read_text(encoding="utf-8")
    )
    for name, record in timeout_manifest["files"].items():
        assert e156.sha256(timeout_output / name) == record["sha256"]
