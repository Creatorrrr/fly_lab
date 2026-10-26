"""Generated-only E165 contracts; never load an E150 recording or FlyBody."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from tools import analyze_c_banc_m3_command_lift as e165


def test_scalar_projection_and_rotated_heading() -> None:
    j = np.asarray(
        [[1, -2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [3, 4, 0, 0, 0, 0, 0]], float
    )
    delta = np.asarray(
        [
            [0.01, 0.02, 0, 0, 0, 0, 0],
            [-0.01, 0.02, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    heading = e165.horizontal_heading(
        np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    )
    assert np.array_equal(heading, [0, 1, 0])
    projection = e165.project_command(j, delta, heading)
    expected = np.asarray([j @ d for d in delta])
    np.testing.assert_allclose(projection["xyz_mm"], expected)
    np.testing.assert_allclose(projection["z_joint_mm"].sum(axis=1), expected[:, 2])
    np.testing.assert_allclose(projection["ap_joint_mm"].sum(axis=1), expected[:, 1])
    assert np.isclose(projection["z_mm"][0], 0.11)


def test_effective_controls_and_segment_signs() -> None:
    request = np.asarray([2.0, -2.0, 0.5])
    delivered = np.asarray([1.5, -3.0, 7.0])
    limits = np.asarray([True, True, False])
    ranges = np.asarray([[0, 1], [-1, 1], [-1, 1]], float)
    three, clipped = e165.effective_requests(request, delivered, limits, ranges)
    np.testing.assert_array_equal(three, [[2, -2, 0.5], [1.5, -3, 7], [1, -1, 7]])
    np.testing.assert_array_equal(clipped, [True, True, False])
    signs = e165.segment_signs(
        [[1, 2, -1, 3, 4], [0, -1, 0, -2, 0], [1, 2, 3, 4, 5]], 1e-12
    )
    np.testing.assert_array_equal(signs["mixed"], [True, False, False])
    np.testing.assert_array_equal(signs["all_nonpositive"], [False, True, False])
    np.testing.assert_array_equal(signs["all_positive"], [False, False, True])


def test_empty_nan_and_signed_weight_contract() -> None:
    assert e165.stats([np.nan, np.nan])["mean"] is None
    assert e165.stats([])["quantiles"] is None
    assert e165.weighted_mean([1, 2], [0, 0]) is None
    assert e165.weighted_mean([1, -2, 0], [1, 2, 3]) == -0.5
    assert e165.weighted_mean([np.nan, 2], [1, 1]) == 2
    with pytest.raises(ValueError):
        e165.weighted_mean([1], [-1])
    evidence = e165.weighted_evidence([1, -2, 0], [1, 2, 3])
    assert evidence["numerator_sum_mm_bw"] == -3
    assert evidence["denominator_sum_bw"] == 6
    assert evidence["mean_mm"] == -0.5
    assert e165.weighted_evidence([np.nan], [0])["mean_mm"] is None


def test_generated_position_servo_rejects_velocity_bias() -> None:
    gain, bias = e165.validate_position_servo([[1000, 0, 0]], [[0, -1000, -4]])
    assert gain[0, 0] == 1000 and bias[0, 2] == -4
    with pytest.raises(ValueError, match="position servo"):
        e165.validate_position_servo([[1000, 0, 0]], [[0, 0, -1000]])


def test_event_offset_availability() -> None:
    values = np.arange(6.0, dtype=float)
    stack, available = e165.event_stack(
        values, [1, 5], [-2, -1, 0, 1], [False, True, True, True, False, True]
    )
    np.testing.assert_array_equal(
        available, [[False, False, True, True], [True, False, True, False]]
    )
    assert np.isnan(stack[0, 0]) and np.isnan(stack[1, 3])
    assert stack[1, 2] == 5


def generated_chain():
    lines = [
        '<mujoco><option gravity="0 0 0"/><worldbody><body name="root" pos="0 0 0">'
    ]
    for index in range(7):
        lines.append(
            f'<body name="b{index}" pos="0.1 0.03 0.02"><joint name="j{index}" type="hinge" axis="0 1 0"/><geom type="sphere" size="0.005" mass="0.001"/>'
        )
    lines.extend(["</body>"] * 8)
    lines.append("</worldbody></mujoco>")
    model = mujoco.MjModel.from_xml_string("".join(lines))
    data = mujoco.MjData(model)
    return model, data


def test_generated_mujoco_jacobian_central_difference() -> None:
    model, data = generated_chain()
    data.qpos[:] = np.linspace(-0.2, 0.3, 7)
    mujoco.mj_forward(model, data)
    point_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b6")
    analytic = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, analytic, None, point_id)
    base = data.qpos.copy()
    numerical = np.zeros_like(analytic)
    h = 1e-6
    for column in range(model.nv):
        data.qpos[:] = base
        data.qpos[column] += h
        mujoco.mj_forward(model, data)
        plus = data.xpos[point_id].copy()
        data.qpos[column] -= 2 * h
        mujoco.mj_forward(model, data)
        minus = data.xpos[point_id].copy()
        numerical[:, column] = (plus - minus) / (2 * h)
    np.testing.assert_allclose(analytic, numerical, atol=1e-7, rtol=0)


class FakeMj:
    MjData = None
    mjtState = SimpleNamespace(mjSTATE_INTEGRATION=1)

    def __init__(self, fail: int | None = None):
        self.fail = fail
        self.forwards = 0
        self.jacobians = 0
        self.invalid_rotation_at = None
        self.origin_offset_at = None

    def mj_forward(self, m, d):
        self.forwards += 1
        if self.forwards == self.fail:
            raise RuntimeError("generated forward fault")
        d.xmat[0] = np.eye(3).reshape(-1)
        d.xpos[:] = 0
        if self.forwards == self.invalid_rotation_at:
            d.xmat[0] = (np.eye(3) * 2).reshape(-1)
        if self.forwards == self.origin_offset_at:
            d.xpos[1, 0] = 1

    def mj_jacBody(self, m, d, jac, _unused, body_id):
        self.jacobians += 1
        jac[:] = 0
        jac[2, (body_id - 1) // 5 * 7] = 0.01

    def mj_stateSize(self, m, state_type):
        return 109 + 108 + 82 + 1

    def mj_getState(self, m, d, output, state_type):
        output[:] = np.concatenate((d.qpos, d.qvel, d.ctrl, [d.time]))


def fake_pipeline(fail: int | None = None):
    fake = FakeMj(fail)
    model = SimpleNamespace(nq=109, nv=108, nu=82)

    def make_data(_):
        return SimpleNamespace(
            qpos=np.zeros(109),
            qvel=np.zeros(108),
            ctrl=np.zeros(82),
            time=0.0,
            xmat=np.zeros((31, 9)),
            xpos=np.zeros((31, 3)),
        )

    fake.MjData = make_data
    body = SimpleNamespace(m=model, mj=fake, thorax=0)
    layout = {
        "actuator_ids": list(range(42)),
        "joint_qpos_addresses": list(range(42)),
        "actuator_dof_addresses": list(range(42)),
        "tarsal_body_ids": np.arange(1, 31).reshape(6, 5).tolist(),
        "actuator_ctrllimited": [True] * 42,
        "actuator_ctrlrange": [[-1, 1]] * 42,
    }
    row = {"time_s": 0.0, "control_tick": 0}
    rows = [dict(row, time_s=t * 0.005, control_tick=t) for t in range(2001)]
    source = {
        "joint_angles_rad": np.zeros((2001, 42)),
        "joint_targets_rad": np.ones((2001, 42)) * 0.5,
        "ctrl": np.ones((2001, 82)) * 0.25,
        "qpos": np.zeros((2001, 109)),
        "qvel": np.zeros((2001, 108)),
        "normal_load_bw": np.zeros((2001, 6)),
        "feature_speed_mm_s": np.zeros(2001),
        "feature_ego_ap_velocity_mm_s": np.zeros((2001, 6)),
        "feature_world_forward_velocity_mm_s": np.zeros((2001, 6)),
        "basis": np.repeat(e165.S[None], 2001, axis=0),
        "tarsal_ui": np.zeros((2001, 6, 5, 3)),
    }
    raw = e165.new_raw(2000, 108)
    ledger = {
        "shadow_mjdata": 0,
        "body_models": 0,
        "forward_entered": 0,
        "forward_completed": 0,
        "jacobian_entered": 0,
        "jacobian_completed": 0,
        "rows_completed": 0,
    }
    return rows, source, body, layout, raw, ledger, fake


def test_full_generated_fake_body_pipeline() -> None:
    import time

    rows, source, body, layout, raw, ledger, _fake = fake_pipeline()
    errors = e165.process_rows(
        rows, source, body, layout, raw, ledger, time.perf_counter(), 90, 10
    )
    assert errors == []
    assert ledger["forward_completed"] == 2000
    assert ledger["jacobian_completed"] == 60000
    assert raw["completed"].all()
    e165.numeric_checks(raw, {"saved_atol": 1e-12})
    np.testing.assert_allclose(raw["z_mm"][0, 0], 0.005)
    np.testing.assert_allclose(raw["z_mm"][0, 1], 0.0025)


@pytest.mark.parametrize("fail", [1, 1000, 2000])
def test_partial_first_middle_last_generated_row(fail: int, tmp_path: Path) -> None:
    import time

    rows, source, body, layout, raw, ledger, _fake = fake_pipeline(fail)
    with pytest.raises(RuntimeError, match="generated forward fault"):
        e165.process_rows(
            rows, source, body, layout, raw, ledger, time.perf_counter(), 90, 10
        )
    e165.atomic_npz(tmp_path / "partial.npz", raw)
    with np.load(tmp_path / "partial.npz", allow_pickle=False) as saved:
        assert int(saved["completed"].sum()) == fail - 1
        assert not bool(saved["completed"][fail - 1])
        assert int(saved["forward_completed"].sum()) == fail - 1
    assert ledger["forward_entered"] == fail


def test_clock_mapping_and_gate_errors(tmp_path: Path) -> None:
    assert e165.safe_path("tools/analyze_c_banc_m3_command_lift.py").is_file()
    with pytest.raises(ValueError):
        e165.safe_path("../../outside")
    with pytest.raises(ValueError):
        e165.project_command(np.zeros((3, 6)), np.zeros((3, 7)), [1, 0, 0])
    with pytest.raises(ValueError):
        e165.horizontal_heading(np.eye(3) * 2)
    path = tmp_path / "gate.json"
    path.write_text(
        json.dumps(
            {
                "approved": True,
                "blocking_defects": [],
                "original_processes_allowed": True,
                "accepts_single_command_lift_diagnostic": True,
                "freeze_sha256": "f",
                "source_sha256": {},
            }
        )
    )
    with pytest.raises(ValueError, match="exactly one"):
        e165.validate_gate(path, e165.sha256(path), "f")


def test_persistent_terminal_save_failure_and_final_overrun(tmp_path: Path) -> None:
    # Terminal receipts cannot be treated as success if the writer or deadline fails.
    path = tmp_path / "receipt.json"
    path.write_text("old")
    (tmp_path / "receipt.json.tmp").write_text("stale")
    with pytest.raises(FileExistsError):
        e165.atomic_json(path, {"valid_diagnostic": True})
    assert path.read_text() == "old"
    assert (tmp_path / "receipt.json.tmp").exists()
    output = tmp_path / "persistent"
    output.mkdir()
    e165.atomic_json(output / "result.json", {"valid_diagnostic": False})
    (output / "result.json.tmp").write_text("stale")
    with pytest.raises(FileExistsError):
        e165.publish_success(
            output,
            {"valid_diagnostic": False},
            {"raw_bytes": 100000, "free_bytes": 0, "internal_s": 10},
            0,
            clock=lambda: 1,
        )
    assert json.loads((output / "result.json").read_text())["valid_diagnostic"] is False
    assert not (output / "output-manifest.json").exists()
    late = tmp_path / "late"
    late.mkdir()
    e165.atomic_json(late / "result.json", {"valid_diagnostic": False})
    with pytest.raises(ValueError, match="final save"):
        e165.publish_success(
            late,
            {"valid_diagnostic": False},
            {"raw_bytes": 100000, "free_bytes": 0, "internal_s": 10},
            0,
            clock=lambda: 11,
        )
    assert json.loads((late / "result.json").read_text())["valid_diagnostic"] is False
    assert not (late / "output-manifest.json").exists()
    assert (late / "result.invalidated.json").exists()
    assert (late / "output-manifest.invalidated.json").exists()


def generated_run_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    fail_forward: int | None = None,
    close_fails: bool = False,
):
    import threadpoolctl

    from flylab.c.storage import StateStore
    from tools import analyze_c_banc_m3_signed_contact as e154

    rows, source, body, layout, _raw, _ledger, fake = fake_pipeline(fail_forward)
    body.d = fake.MjData(body.m)
    body.d.ctrl[:] = source["ctrl"][0]
    body.t0 = 0.0
    body.model_hash = "generated-model"
    body.world_spec = {"bounds": [10, 10, 10], "obstacles": []}
    body._force_maps = {"feet": (np.zeros(3, dtype=int), None)}
    body.support_geom_ids = set()
    body.weight0 = 1.0
    body.m.opt = SimpleNamespace(timestep=0.001)
    body.physics_time = lambda: body.d.time
    closed = []

    def close():
        closed.append(True)
        if close_fails:
            raise RuntimeError("generated close fault")

    body.close = close
    saved = {
        "control_tick": 0,
        "seed": 7,
        "world": body.world_spec,
        "parameters": {},
        "body": {"modelHash": "generated-model"},
    }
    body_legs = {leg: [[0, 0, 0] for _ in range(8)] for leg in e165.LEGS}
    trace_rows = []
    for tick, row in enumerate(rows):
        interval = (
            None
            if tick == 0
            else {
                "start_tick": tick - 1,
                "end_tick": tick,
                "start_s": (tick - 1) * 0.005,
                "end_s": tick * 0.005,
            }
        )
        trace_rows.append(
            {
                **row,
                "command_interval": interval,
                "qpos": source["qpos"][tick].tolist(),
                "qvel": source["qvel"][tick].tolist(),
                "ctrl": source["ctrl"][tick].tolist(),
                "joint_angles_rad": source["joint_angles_rad"][tick].tolist(),
                "joint_targets_rad": source["joint_targets_rad"][tick].tolist(),
                "joint_tracking_error_rad": source["joint_targets_rad"][tick].tolist(),
                "body": {"legs": body_legs, "basis": e165.S.tolist()},
                "feet": {"normal_load_bw": [0] * 6},
                "neural_request": None,
            }
        )
    trace = {
        "schema": "flylab.m3.e150.native-monotone-body-trace.v1",
        "body_hash": "generated-model",
        "parameters": {"diagnostic": {"experiment": "E150"}},
        "samples": trace_rows,
    }
    data = tmp_path / "generated-data"
    data.mkdir()
    (data / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
    (data / "context.json").write_text(
        json.dumps(
            {
                "body_hash": "generated-model",
                "body_parameters": {},
                "feet_map": [0, 0, 0],
                "support_geom_ids": [],
                "weight0": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (data / "layout.json").write_text(
        json.dumps({"body_hash": "generated-model"}), encoding="utf-8"
    )
    (data / "design.md").write_text("generated design", encoding="utf-8")
    (data / "checkpoint").mkdir()
    ticks = np.arange(2001)
    features = {
        "time_s": ticks * 0.005,
        "window": (ticks * 0.005 >= 0.5) & (ticks * 0.005 < 10),
        "speed_mm_s": np.zeros(2001),
        "boundary_load_bw": np.zeros((2001, 6)),
        "ego_ap_velocity_mm_s": np.zeros((2001, 6)),
        "world_forward_velocity_mm_s": np.zeros((2001, 6)),
    }
    np.savez(data / "features.npz", **features)
    np.savez(
        data / "events.npz",
        event_index=np.arange(100, 146),
        offset_control=np.arange(-20, 21),
        offset_s=np.arange(-20, 21) * 0.005,
    )
    contract = {
        "legs": list(e165.LEGS),
        "segments": list(range(1, 6)),
        "request_kinds": list(e165.KINDS),
        "group_names": list(e165.GROUPS),
        "ui_from_world": e165.S.tolist(),
        "recorded_rows": 2001,
        "diagnostic_rows": 2000,
        "joints": 42,
        "nq": 109,
        "nv": 108,
        "nu": 82,
        "control_dt_s": 0.005,
        "neural_ticks_per_control": 20,
        "clock_atol": 2e-8,
        "saved_atol": 1e-12,
        "zero_mm": 1e-12,
        "support_threshold_bw": 0.02,
        "forward_threshold_mm_s": 0.5,
        "events": 46,
        "event_offsets": list(range(-20, 21)),
    }
    limits = {
        "original_processes": 1,
        "automatic_retries": 0,
        "cpu_threads": 1,
        "gpu_count": 0,
        "internal_s": 90,
        "external_s": 110,
        "supervisor_s": 120,
        "reserve_s": 10,
        "raw_bytes": 201326592,
        "free_bytes": 0,
    }
    frozen = {
        "schema": e165.SCHEMA,
        "owned_files": list(e165.OWNED),
        "design": "design",
        "design_sha256": e165.sha256(data / "design.md"),
        "body_hash": "generated-model",
        "sample_contract": contract,
        "resource_limits": limits,
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "F6_exposed": False,
        "input_sha256": {},
        "reused_source_sha256": {},
        "preservation_sha256": {},
        "paths": {
            "trace": "trace",
            "features": "features",
            "events": "events",
            "context": "context",
            "prior_layout": "layout",
            "checkpoint": "checkpoint",
        },
        "prior_conservative_s": 2818,
        "prospective_cumulative_s": 2938,
    }
    freeze_path = data / "freeze.json"
    freeze_path.write_text(json.dumps(frozen), encoding="utf-8")
    freeze_sha = e165.sha256(freeze_path)
    monkeypatch.setattr(e165, "FREEZE_SHA", freeze_sha)
    files = {
        "design": data / "design.md",
        "trace": data / "trace.json",
        "features": data / "features.npz",
        "events": data / "events.npz",
        "context": data / "context.json",
        "layout": data / "layout.json",
        "checkpoint": data / "checkpoint",
    }
    monkeypatch.setattr(e165, "safe_path", lambda name: files[name])
    source_hash = {name: e165.sha256(e165.ROOT / name) for name in e165.OWNED}
    gate = {
        "approved": True,
        "blocking_defects": [],
        "original_processes_allowed": 1,
        "accepts_single_command_lift_diagnostic": True,
        "freeze_sha256": freeze_sha,
        "source_sha256": source_hash,
    }
    review_path = data / "review.json"
    review_path.write_text(json.dumps(gate), encoding="utf-8")
    monkeypatch.setattr(StateStore, "load", lambda _path: saved)
    monkeypatch.setattr(e154, "validate_body_parameters", lambda _saved: {})

    def construct(_saved, _observation, ledger):
        ledger["body_constructor_attempts"] += 1
        ledger["body_constructor_completed"] += 1
        ledger["settling_physics_steps"] = 2000
        return body

    monkeypatch.setattr(e154, "construct_body_safely", construct)
    monkeypatch.setattr(e154, "validate_settling_count", lambda *_args: 2000)
    monkeypatch.setattr(e165, "validate_layout", lambda _body, _prior: layout)
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{"num_threads": 1}])
    return (
        freeze_path,
        freeze_sha,
        review_path,
        e165.sha256(review_path),
        data / "output",
        body,
        fake,
        closed,
    )


def test_generated_full_run_success(tmp_path: Path, monkeypatch) -> None:
    freeze, freeze_sha, review, review_sha, out, _body, fake, closed = (
        generated_run_fixture(tmp_path, monkeypatch)
    )
    result = e165.run(freeze, freeze_sha, review, review_sha, out)
    assert result["valid_diagnostic"] and result["publication_complete"]
    assert fake.forwards == 2000 and fake.jacobians == 60000
    assert len(closed) == 1
    assert json.loads((out / "result.json").read_text())["valid_diagnostic"]
    assert (out / "output-manifest.json").exists()
    assert (out / "actor-preservation-raw.npz").exists()
    assert (out / "event-stacks-raw.npz").exists()
    with np.load(out / "event-stacks-raw.npz", allow_pickle=False) as stack:
        for key in (
            "segment_min_z_mm",
            "segment_max_z_mm",
            "segment_mean_z_mm",
            "segment_min_ap_mm",
            "segment_max_ap_mm",
            "segment_mean_ap_mm",
        ):
            assert stack[key].shape == (46, 41, 3, 6)
    summary = json.loads((out / "summary.json").read_text())
    item = summary["groups"]["lf"]["supported"]["kinds"]["requested"]
    assert item["load_weighted_z_evidence"][0]["denominator_sum_bw"] == 0
    assert item["load_weighted_z_mm"][0] is None


@pytest.mark.parametrize("fail", [1, 1000, 2000])
def test_generated_full_run_partial_fault(
    tmp_path: Path, monkeypatch, fail: int
) -> None:
    freeze, freeze_sha, review, review_sha, out, _body, fake, closed = (
        generated_run_fixture(tmp_path, monkeypatch, fail_forward=fail)
    )
    with pytest.raises(RuntimeError, match="generated forward fault"):
        e165.run(freeze, freeze_sha, review, review_sha, out)
    assert fake.forwards == fail and len(closed) == 1
    with np.load(out / "command-lift-partial.npz", allow_pickle=False) as partial:
        assert int(partial["completed"].sum()) == fail - 1
    assert json.loads((out / "result.json").read_text())["valid_diagnostic"] is False


def test_generated_invalid_rotation_raw_before_failure(
    tmp_path: Path, monkeypatch
) -> None:
    freeze, freeze_sha, review, review_sha, out, _body, fake, _closed = (
        generated_run_fixture(tmp_path, monkeypatch)
    )
    fake.invalid_rotation_at = 1
    with pytest.raises(ValueError, match="not orthogonal"):
        e165.run(freeze, freeze_sha, review, review_sha, out)
    with np.load(out / "command-lift-partial.npz", allow_pickle=False) as partial:
        np.testing.assert_array_equal(partial["rotation_world"][0], np.eye(3) * 2)
        assert not bool(partial["completed"][0])


def test_generated_origin_error_survives_later_fault(
    tmp_path: Path, monkeypatch
) -> None:
    freeze, freeze_sha, review, review_sha, out, _body, fake, _closed = (
        generated_run_fixture(tmp_path, monkeypatch, fail_forward=2)
    )
    fake.origin_offset_at = 1
    with pytest.raises(RuntimeError, match="generated forward fault"):
        e165.run(freeze, freeze_sha, review, review_sha, out)
    evidence = json.loads((out / "input-failure-evidence.json").read_text())
    assert any(
        "tarsus1 UI origin mismatch" in error for error in evidence["comparison_errors"]
    )
    with np.load(out / "command-lift-partial.npz", allow_pickle=False) as partial:
        assert partial["origin_residual"][0, 0, 0] == 1
        assert bool(partial["completed"][0]) and not bool(partial["completed"][1])


@pytest.mark.parametrize(
    "fault",
    [
        "numeric",
        "cleanup",
        "posthash",
        "snapshot",
        "npz",
        "json",
        "capacity",
        "final_time",
    ],
)
def test_generated_full_run_terminal_faults(
    tmp_path: Path, monkeypatch, fault: str
) -> None:
    freeze, freeze_sha, review, review_sha, out, _body, fake, _closed = (
        generated_run_fixture(tmp_path, monkeypatch, close_fails=fault == "cleanup")
    )
    if fault == "numeric":
        original = fake.mj_jacBody

        def bad_jac(m, d, jac, unused, body_id):
            original(m, d, jac, unused, body_id)
            jac[0, 41] = 0.1 if body_id == 1 else jac[0, 41]

        fake.mj_jacBody = bad_jac
    elif fault == "posthash":
        original = e165.verify_hashes
        calls = []

        def fail_post(entries):
            calls.append(True)
            if len(calls) == 2:
                raise ValueError("generated posthash fault")
            return original(entries)

        monkeypatch.setattr(e165, "verify_hashes", fail_post)
    elif fault == "snapshot":
        monkeypatch.setattr(
            e165,
            "verify_snapshot_sources",
            lambda *_args: (_ for _ in ()).throw(
                ValueError("generated snapshot fault")
            ),
        )
    elif fault == "npz":
        original = e165.atomic_npz

        def fail_npz(path, arrays):
            if path.name == "event-stacks-raw.npz":
                raise OSError("generated NPZ fault")
            return original(path, arrays)

        monkeypatch.setattr(e165, "atomic_npz", fail_npz)
    elif fault == "json":
        original = e165.atomic_json

        def fail_json(path, value):
            if path.name == "summary.json":
                raise OSError("generated JSON fault")
            return original(path, value)

        monkeypatch.setattr(e165, "atomic_json", fail_json)
    elif fault in ("capacity", "final_time"):
        original = e165.publish_success

        def fail_terminal(path, pending, limits, started):
            revised = dict(limits)
            if fault == "capacity":
                revised["raw_bytes"] = 1
                return original(path, pending, revised, started)
            return original(
                path,
                pending,
                revised,
                started,
                clock=lambda: started + revised["internal_s"] + 1,
            )

        monkeypatch.setattr(e165, "publish_success", fail_terminal)
    with pytest.raises((RuntimeError, ValueError, OSError)):
        e165.run(freeze, freeze_sha, review, review_sha, out)
    result = json.loads((out / "result.json").read_text())
    assert result["valid_diagnostic"] is False
    assert result["publication_complete"] is False
    assert (out / "command-lift-partial.npz").exists()
    assert not (out / "output-manifest.json").exists()
    if fault == "numeric":
        assert (out / "command-lift-raw.npz").exists()
    if fault in ("capacity", "final_time"):
        assert (out / "result.invalidated.json").exists()
        assert (out / "output-manifest.invalidated.json").exists()


def test_generated_gate_changed_before_copy(tmp_path: Path, monkeypatch) -> None:
    freeze, freeze_sha, review, review_sha, out, _body, _fake, _closed = (
        generated_run_fixture(tmp_path, monkeypatch)
    )
    original = e165.validate_gate

    def altered_gate(*args):
        accepted = original(*args)
        review.write_text(review.read_text() + " ", encoding="utf-8")
        return accepted

    monkeypatch.setattr(e165, "validate_gate", altered_gate)
    with pytest.raises(ValueError, match="before copy"):
        e165.run(freeze, freeze_sha, review, review_sha, out)
    assert (out / "input-failure-evidence.json").exists()
    assert json.loads((out / "result.json").read_text())["valid_diagnostic"] is False


def test_generated_input_clock_failure_has_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    freeze, freeze_sha, review, review_sha, out, _body, _fake, _closed = (
        generated_run_fixture(tmp_path, monkeypatch)
    )
    trace_path = freeze.parent / "trace.json"
    trace = json.loads(trace_path.read_text())
    trace["samples"][5]["command_interval"]["end_tick"] = 6
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(ValueError, match="command interval"):
        e165.run(freeze, freeze_sha, review, review_sha, out)
    assert (out / "input-failure-evidence.json").exists()
    assert json.loads((out / "result.json").read_text())["valid_diagnostic"] is False


def test_generated_actor_state_change_rejected(tmp_path: Path, monkeypatch) -> None:
    freeze, freeze_sha, review, review_sha, out, body, fake, _closed = (
        generated_run_fixture(tmp_path, monkeypatch)
    )
    original = fake.mj_forward

    def mutate_actor(m, d):
        original(m, d)
        if fake.forwards == 1:
            body.d.qpos[0] = 0.125

    fake.mj_forward = mutate_actor
    with pytest.raises(ValueError, match="actor integration state changed"):
        e165.run(freeze, freeze_sha, review, review_sha, out)
    with np.load(out / "actor-preservation-raw.npz", allow_pickle=False) as raw:
        assert raw["after"][0] - raw["before"][0] == 0.125
    assert json.loads((out / "result.json").read_text())["valid_diagnostic"] is False


def test_generated_snapshot_byte_mutation_rejected(tmp_path: Path, monkeypatch) -> None:
    freeze, freeze_sha, review, review_sha, out, _body, _fake, _closed = (
        generated_run_fixture(tmp_path, monkeypatch)
    )
    original = e165.process_rows

    def tamper(*args):
        result = original(*args)
        (out / "source-snapshot" / "analyzer.py").write_bytes(
            b"tampered generated snapshot"
        )
        return result

    monkeypatch.setattr(e165, "process_rows", tamper)
    with pytest.raises(ValueError, match="source snapshot changed"):
        e165.run(freeze, freeze_sha, review, review_sha, out)
    assert json.loads((out / "result.json").read_text())["valid_diagnostic"] is False


def test_generated_final_free_space_failure(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "out"
    out.mkdir()
    e165.atomic_json(out / "result.json", {"valid_diagnostic": False})
    monkeypatch.setattr(
        e165.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0)
    )
    with pytest.raises(ValueError, match="free-space"):
        e165.publish_success(
            out,
            {"valid_diagnostic": False},
            {"raw_bytes": 100000, "free_bytes": 1, "internal_s": 10},
            0,
            clock=lambda: 1,
        )
    assert json.loads((out / "result.json").read_text())["valid_diagnostic"] is False
    assert (out / "output-manifest.invalidated.json").exists()


@pytest.mark.parametrize("field", ["body_origin_world_mm", "basis_residual", "xyz_mm"])
def test_nonfinite_shadow_data_is_rejected(field: str) -> None:
    raw = e165.new_raw(1, 108)
    for values in raw.values():
        values.fill(0)
    for name in ("completed", "forward_completed", "jacobian_completed"):
        raw[name].fill(True)
    raw[field].flat[0] = np.nan
    with pytest.raises(ValueError, match=f"Nonfinite E165 raw field: {field}"):
        e165.numeric_checks(raw, {"saved_atol": 1e-12})


def test_segment_extent_uses_the_same_numeric_zero_band() -> None:
    raw = e165.new_raw(1, 108)
    for values in raw.values():
        values.fill(0)
    raw["z_mm"].fill(1e-12)
    raw["ap_mm"].fill(1e-12)
    raw["normal_load_bw"].fill(0.02)
    result = e165.summaries(
        raw,
        {"feature_window": np.array([False, True])},
        {"zero_mm": 1e-12, "support_threshold_bw": 0.02, "forward_threshold_mm_s": 0.5},
    )
    item = result["lf"]["all"]["kinds"]["requested"]
    assert item["all_positive_fraction"] == 0
    for axis in ("z", "ap"):
        for extent in ("min", "max", "mean"):
            assert item[f"segment_{extent}_{axis}_mm"]["positive_fraction"] == 0


def test_e150_trace_metadata_does_not_relax_model_parameter_identity() -> None:
    saved = {"joint_activation_tau_s": 0.03}
    diagnostic = {"experiment": "E150", "fixed_candidate": True}
    assert (
        e165.validate_parameter_identity({**saved, "diagnostic": diagnostic}, saved)
        == diagnostic
    )
    with pytest.raises(ValueError, match="model parameters differ"):
        e165.validate_parameter_identity(
            {"joint_activation_tau_s": 0.04, "diagnostic": diagnostic}, saved
        )
    with pytest.raises(ValueError, match="model parameters differ"):
        e165.validate_parameter_identity(
            {**saved, "diagnostic": diagnostic, "unknown_model_setting": 1}, saved
        )
    with pytest.raises(ValueError, match="diagnostic provenance"):
        e165.validate_parameter_identity(
            {**saved, "diagnostic": {"experiment": "E151"}}, saved
        )


def test_direct_script_bootstrap_from_unrelated_cwd(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    script = e165.ROOT / e165.OWNED[0]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    code = (
        "import runpy, sys; runpy.run_path(sys.argv[1]); "
        "import flylab.c.storage; print('direct-script-import-ok')"
    )
    process = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "direct-script-import-ok"
