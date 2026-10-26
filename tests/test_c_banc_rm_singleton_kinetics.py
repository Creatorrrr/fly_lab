from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_c_banc_m3_rm_singleton_kinetics as e158


def test_ema_matches_explicit_recurrence_and_uses_saved_initial():
    rate = np.array([99.0, 1.0, 4.0, 2.0], dtype=np.float64)
    actual = e158.ema_from_saved_rate(rate, 0.25)
    alpha = -np.expm1(-0.005 / 0.04)
    expected = np.array([0.25, 0.25 + alpha * 0.75, 0.0, 0.0])
    expected[2] = expected[1] + alpha * (4 - expected[1])
    expected[3] = expected[2] + alpha * (2 - expected[2])
    assert np.array_equal(actual, expected)
    assert actual[0] == 0.25


def candidate_fixture(n=9):
    rate = np.zeros((n, 391), dtype=np.float64)
    rate[:, 364] = np.linspace(0, 4, n)
    bank = np.zeros((n, 391, 3), dtype=np.float64)
    bank[:, 364, 2] = e158.ema_from_saved_rate(rate[:, 364], 0.5)
    activation = np.arange(n * 96, dtype=np.float64).reshape(n, 96) / 1000
    moment = np.zeros((42, 96), dtype=np.float64)
    moment[29, 67] = 1
    moment[3, 4] = -0.25
    rest = np.linspace(-0.2, 0.2, 42, dtype=np.float64)
    command = rest + 0.02 * (activation @ moment.T)
    return bank, rate, activation, command, moment, rest


def test_build_candidate_changes_only_fixed_muscle_and_joint():
    values = candidate_fixture()
    result = e158.build_candidate_commands(
        bank=values[0],
        motor_rate=values[1],
        baseline_activation=values[2],
        baseline_command=values[3],
        moment=values[4],
        rest=values[5],
    )
    candidate = result["candidate_command"]
    active = result["candidate_activation"]
    assert np.array_equal(
        np.delete(candidate, 29, axis=1), np.delete(values[3], 29, axis=1)
    )
    assert np.array_equal(
        np.delete(active, 67, axis=1), np.delete(values[2], 67, axis=1)
    )
    assert np.array_equal(candidate[0], values[3][0])
    assert np.array_equal(candidate[1:, 29], values[5][29] + 0.02 * result["ema40"][1:])
    assert result["bank_max_abs_error"] == 0
    assert result["target_reconstruction_max_abs_error"] == 0


@pytest.mark.parametrize("which", ["bank", "command", "moment"])
def test_build_candidate_rejects_each_frozen_identity_violation(which):
    bank, rate, activation, command, moment, rest = candidate_fixture()
    if which == "bank":
        bank[4, 364, 2] += 2e-10
    elif which == "command":
        command[1, 1] += 2e-10
    else:
        moment[29, 3] = 1e-20
    with pytest.raises(ValueError):
        e158.build_candidate_commands(
            bank=bank,
            motor_rate=rate,
            baseline_activation=activation,
            baseline_command=command,
            moment=moment,
            rest=rest,
        )


def test_initial_observation_is_preserved_before_the_first_decoded_command():
    bank, rate, activation, command, moment, rest = candidate_fixture()
    command[0] += 0.33
    result = e158.build_candidate_commands(
        bank=bank,
        motor_rate=rate,
        baseline_activation=activation,
        baseline_command=command,
        moment=moment,
        rest=rest,
    )
    assert np.array_equal(result["candidate_command"][0], command[0])
    assert result["initial_observation_reconstruction_difference"] == pytest.approx(
        0.33
    )
    assert result["target_reconstruction_max_abs_error"] == 0
    assert np.array_equal(
        result["candidate_command"][1:, 29], rest[29] + 0.02 * result["ema40"][1:]
    )


def test_reconstruction_diagnostics_precede_a_failed_numerical_assertion():
    bank, rate, activation, command, moment, rest = candidate_fixture()
    command[4, 1] += 0.125
    captured = []
    with pytest.raises(ValueError, match="original target reconstruction"):
        e158.build_candidate_commands(
            bank=bank,
            motor_rate=rate,
            baseline_activation=activation,
            baseline_command=command,
            moment=moment,
            rest=rest,
            diagnostics=captured.append,
        )
    assert len(captured) == 1
    assert captured[0]["reconstruction_error"][4, 1] == pytest.approx(-0.125)
    assert np.array_equal(captured[0]["baseline_command"], command)


def test_startup_compatibility_uses_exact_99_prefix_and_four_limits():
    q = np.zeros((2001, 42))
    ltm = np.zeros((2001, 6))
    target_q = q[1:100].copy()
    target_ltm = ltm[1:100].copy()
    limits = {"q_RMS": 0.02, "q_max": 0.04, "LTM_RMS": 0.02, "LTM_max": 0.04}
    assert e158.startup_compatibility(q, ltm, target_q, target_ltm, limits)[
        "startup_pass"
    ]
    q[50, 2] = 0.04000001
    result = e158.startup_compatibility(q, ltm, target_q, target_ltm, limits)
    assert not result["startup_pass"]
    assert result["q"]["max_abs"] > limits["q_max"]


def test_singleton_mapping_binds_global_column_and_branch():
    membership = np.zeros(391, dtype=np.int64)
    membership[364] = 67
    motor_ids = np.arange(391, dtype=np.int64)
    motor_ids[364] = 144018
    parameters = {
        "tau": np.array([0.005, 0.01, 0.04]),
        "kappa": np.zeros(391),
        "slopes": np.zeros((96, 391, 3, 2)),
        "membership": membership,
        "motor_ids": motor_ids,
    }
    provenance = {
        "motor_ids": motor_ids.copy(),
        "moment": np.zeros((42, 96)),
        "rest": np.zeros(42),
        "pool": np.zeros((96, 391)),
        "membership": membership.copy(),
        "ltm": np.arange(6),
    }
    intervention = {"motor_column": 364, "muscle": 67, "graph_index": 144018}
    e158.validate_singleton_mapping(parameters, provenance, intervention)
    parameters["membership"] = membership.copy()
    parameters["membership"][363] = 67
    with pytest.raises(ValueError, match="singleton"):
        e158.validate_singleton_mapping(parameters, provenance, intervention)


def test_impulse_uses_canonical_ordinals_not_float_mask():
    time_s = np.arange(1, 100001) * 0.0001
    time_s[9999] += 1e-12
    force = np.zeros((100000, 6))
    force[10000:20000, 4] = -2
    force[9999, 4] = -1e6
    result = e158.impulse_summary(time_s, force)
    assert result["negative_forward_impulse_bw_s"]["rm"] == pytest.approx(2.0)
    baseline = e158.impulse_summary(time_s[10000:20000], force[10000:20000])
    assert baseline["negative_forward_impulse_bw_s"]["rm"] == pytest.approx(2.0)


def test_horizontal_heading_uses_world_z_as_vertical_with_yaw_and_pitch():
    yaw = np.deg2rad(45)
    pitch = np.deg2rad(20)
    rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    rotation = rz @ ry
    heading = e158.horizontal_heading(rotation)
    assert np.allclose(heading, [2**-0.5, 2**-0.5, 0], atol=1e-15)
    force = np.array([1.0, 0.0, 0.0])
    assert force @ heading == pytest.approx(2**-0.5)


def quality_fixture():
    return {
        "complete_10s_native_health": True,
        "fixed_initial_heading_net_forward_mm": 9.5,
        "net_over_path": 0.76,
        "longest_forward_bout_s": 0.3,
        "forward_eligible_complete_cycles": {leg: 1 for leg in e158.LEGS},
        "knee_width_deg": {"lf": 25, "lm": 0, "lh": 40, "rf": 30, "rm": 0, "rh": 35},
        "development_screens": {"F3_numeric": True},
    }


def thresholds_fixture():
    return {
        "forward_ratio_min": 0.95,
        "net_path_ratio_min": 0.95,
        "slip_ratio_max": 1.0,
        "longest_bout_s_min": 0.3,
        "all_leg_cycles_min": 1,
        "rm_backward_impulse_ratio_max": 0.9,
        "F3_numeric": True,
        "knee_width_deg_min": {"lf": 25, "rf": 30, "lh": 40, "rh": 35},
    }


def test_advancement_requires_every_fixed_gate_and_reports_ratios():
    impulses = {"negative_forward_impulse_bw_s": {"rm": 0.9}}
    baseline_impulses = {"negative_forward_impulse_bw_s": {"rm": 1.0}}
    result = e158.advancement_decision(
        complete_health=True,
        quality=quality_fixture(),
        slip={"available": True, "load_weighted_mm_s": 2.0},
        baseline_quality={
            "fixed_initial_heading_net_forward_mm": 10.0,
            "net_over_path": 0.8,
        },
        baseline_slip={"load_weighted_mm_s": 2.0},
        impulses=impulses,
        baseline_impulses=baseline_impulses,
        thresholds=thresholds_fixture(),
    )
    assert result["advance_to_closed_loop"]
    assert result["forward_ratio"] == pytest.approx(0.95)
    altered = copy.deepcopy(quality_fixture())
    altered["forward_eligible_complete_cycles"]["rh"] = 0
    result = e158.advancement_decision(
        complete_health=True,
        quality=altered,
        slip={"available": True, "load_weighted_mm_s": 2.0},
        baseline_quality={
            "fixed_initial_heading_net_forward_mm": 10.0,
            "net_over_path": 0.8,
        },
        baseline_slip={"load_weighted_mm_s": 2.0},
        impulses=impulses,
        baseline_impulses=baseline_impulses,
        thresholds=thresholds_fixture(),
    )
    assert not result["advance_to_closed_loop"]


class FakeMj:
    class mjtState:
        mjSTATE_INTEGRATION = 1

    class MjData:
        def __init__(self, model):
            self.model = model

    @staticmethod
    def mj_stateSize(model, state_type):
        return 1

    @staticmethod
    def mj_getState(model, data, state, state_type):
        state[0] = 1

    @staticmethod
    def mj_setState(model, data, state, state_type):
        return None

    @staticmethod
    def mj_forward(model, data):
        return None


class FakeShadow:
    def __init__(self):
        self.d = SimpleNamespace()

    def nonfoot_contact(self):
        return False


class FakeBody:
    def __init__(self):
        self.m = SimpleNamespace(nu=44, opt=SimpleNamespace(timestep=0.0001))
        self.mj = FakeMj()
        self.act_ids = np.arange(42)
        self.d = SimpleNamespace(
            ctrl=np.zeros(44),
            qpos=np.zeros(3),
            qvel=np.zeros(3),
            warning=SimpleNamespace(number=np.zeros(1)),
        )
        self.t = 0.0
        self.fault = None
        self.closed = False

    def physics_time(self):
        return self.t

    def _step_physics(self):
        self.t += 0.0001

    def step_joint_targets(self, joint, adhesion, **kwargs):
        self.d.ctrl[:42] = joint
        for _ in range(50):
            self._step_physics()

    def close(self):
        self.closed = True


def body_rows():
    return [
        {
            "time_s": tick * 0.005,
            "control_tick": tick,
            "joint_targets_rad": np.zeros(42),
            "adhesion": np.zeros(6, bool),
            "tendon_inputs": {key: 0.0 for key in e158.TENDONS},
            "ctrl": np.zeros(44),
        }
        for tick in range(2001)
    ]


def test_execute_body_retains_completed_integration_when_observer_fails():
    body = FakeBody()
    ledger = {
        key: 0
        for key in (
            "scientific_steps_requested",
            "scientific_steps_completed",
            "scientific_samples_observed",
            "shadow_forward_requested",
            "shadow_forward_completed",
        )
    }

    def replay(row, **kwargs):
        return (
            np.zeros(42),
            np.zeros(6, bool),
            {
                "push_left_s": 0.0,
                "push_force": np.zeros(3),
                "step_kwargs": {"tendon_inputs": row["tendon_inputs"]},
            },
        )

    def fail_observer(body, shadow):
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError) as caught:
        e158.execute_body(
            body=body,
            shadow=FakeShadow(),
            rows=body_rows(),
            candidate_command=np.zeros((2001, 42)),
            implicit_request=None,
            replay_command=replay,
            observe_shadow=fail_observer,
            contact_diagnostics=lambda shadow: {},
            capture_control=lambda view: {},
            ledger=ledger,
        )
    raw = caught.value.e158_native
    assert ledger["scientific_steps_requested"] == 1
    assert ledger["scientific_steps_completed"] == 1
    assert ledger["shadow_forward_completed"] == 1
    assert raw["integration_complete"].tolist() == [True]
    assert raw["shadow_complete"].tolist() == [True]
    assert raw["observation_complete"].tolist() == [False]
    assert raw["qpos"].shape == (1, 3)


def write_npz(path: Path, **values):
    np.savez(path, **values)


def generated_run_inputs(tmp_path: Path, *, reject=False):
    n = 2001
    bank, rate, activation, command, moment, rest = candidate_fixture(n)
    candidate = e158.build_candidate_commands(
        bank=bank,
        motor_rate=rate,
        baseline_activation=activation,
        baseline_command=command,
        moment=moment,
        rest=rest,
    )["candidate_command"]
    trace = tmp_path / "trace.json"
    rows = [
        {
            "time_s": i * 0.005,
            "control_tick": i,
            "joint_targets_rad": command[i].tolist(),
            "tendon_inputs": {key: 0.0 for key in e158.TENDONS},
            "qpos": np.zeros(3).tolist(),
            "qvel": np.zeros(3).tolist(),
            "ctrl": np.zeros(44).tolist(),
        }
        for i in range(n)
    ]
    trace.write_text(json.dumps({"samples": rows}), encoding="utf-8")
    recruitment = tmp_path / "recruitment.npz"
    write_npz(
        recruitment,
        bank=bank,
        activation=activation,
        motor_rate=rate,
        bank_available=np.ones(n, bool),
        activation_available=np.ones(n, bool),
        motor_rate_available=np.ones(n, bool),
        complete=np.ones(n, bool),
    )
    provenance = tmp_path / "provenance.npz"
    membership = np.arange(391, dtype=np.int64) % 96
    membership[membership == 67] = 66
    membership[364] = 67
    motor_ids = np.arange(391, dtype=np.int64) + 143654
    motor_ids[364] = 144018
    write_npz(
        provenance,
        moment=moment,
        rest=rest,
        membership=membership,
        motor_ids=motor_ids,
        pool=np.zeros((96, 391)),
        ltm=np.arange(6, dtype=np.int64),
    )
    parameters = tmp_path / "parameters.npz"
    write_npz(
        parameters,
        tau=np.array([0.005, 0.01, 0.04]),
        kappa=np.zeros(391),
        slopes=np.zeros((96, 391, 3, 2)),
        membership=membership,
        motor_ids=motor_ids,
    )
    startup = tmp_path / "startup.npz"
    target = candidate[1:100].copy()
    if reject:
        target[0, 29] += 0.05
    write_npz(startup, target_command=target, target_tendon=np.zeros((99, 6)))
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "quality-summary.json").write_text(
        json.dumps(
            {
                "fixed_initial_heading_net_forward_mm": 10.0,
                "net_over_path": 0.8,
            }
        )
    )
    (baseline / "supported-slip.json").write_text(
        json.dumps({"load_weighted_mm_s": 2.0})
    )
    forces = tmp_path / "forces.npz"
    write_npz(
        forces,
        time_s=np.arange(10001, 20001) * 0.0001,
        force_forward_bw=np.column_stack(
            [np.zeros((10000, 4)), -np.ones(10000), np.zeros(10000)]
        ),
    )
    return {
        "trace": trace,
        "recruitment": recruitment,
        "provenance": provenance,
        "parameters": parameters,
        "startup": startup,
        "baseline": baseline,
        "forces": forces,
    }


def fake_freeze(paths):
    return {
        "trace": "trace",
        "recruitment": "recruitment",
        "provenance": "provenance",
        "parameters": "parameters",
        "startup": "startup",
        "baseline_directory": "baseline",
        "baseline_forces": "forces",
        "intervention": {
            "muscle": 67,
            "motor_column": 364,
            "branch": 2,
            "joint_row": 29,
            "gain": 0.02,
            "tau_s": 0.04,
            "graph_index": 144018,
        },
        "startup_limits": {
            "q_RMS": 0.02,
            "q_max": 0.04,
            "LTM_RMS": 0.02,
            "LTM_max": 0.04,
        },
        "advancement": thresholds_fixture(),
        "resource_limits": {"internal_s": 260, "free_bytes": 0, "raw_bytes": 1 << 30},
        "prior_actual_original_s": 2619.309455999988,
        "prior_conservative_s": 2620,
        "checkpoint": "checkpoint",
        "body_hash": "generated-body",
    }


def test_run_startup_rejection_never_calls_body_and_saves_raw(monkeypatch, tmp_path):
    paths = generated_run_inputs(tmp_path, reject=True)
    freeze = fake_freeze(paths)
    monkeypatch.setattr(e158, "verified_inputs", lambda *args: (freeze, {}))
    monkeypatch.setattr(e158, "safe_path", lambda name: paths[name])
    called = False

    def runner(**kwargs):
        nonlocal called
        called = True
        raise AssertionError

    result = e158.run("f", "h", "r", "s", tmp_path / "out", body_runner=runner)
    assert result["validity"] == "valid_startup_rejected"
    assert result["physical_metrics"] is None
    assert not called
    assert (tmp_path / "out/candidate-commands.npz").exists()
    assert (tmp_path / "out/startup-original99.npz").exists()


def test_run_generated_complete_body_publishes_fixed_artifacts(monkeypatch, tmp_path):
    paths = generated_run_inputs(tmp_path)
    freeze = fake_freeze(paths)
    monkeypatch.setattr(e158, "verified_inputs", lambda *args: (freeze, {}))
    monkeypatch.setattr(e158, "safe_path", lambda name: paths[name])

    def runner(**kwargs):
        kwargs["ledger"].update(
            body_constructor_attempts=1,
            body_constructor_completed=1,
            settling_physics_steps=2000,
            body_models=1,
            shadow_mjdata=1,
            scientific_steps_requested=100000,
            scientific_steps_completed=100000,
            scientific_samples_observed=100000,
            shadow_forward_requested=100000,
            shadow_forward_completed=100000,
        )
        native = {
            "time_s": np.arange(1, 100001) * 0.0001,
            "force_forward_bw": np.column_stack(
                [
                    np.zeros((100000, 4)),
                    np.r_[np.zeros(10000), -0.9 * np.ones(10000), np.zeros(80000)],
                    np.zeros(100000),
                ]
            ),
        }
        return (
            [{} for _ in range(2001)],
            native,
            quality_fixture(),
            {
                "available": True,
                "load_weighted_mm_s": 2.0,
            },
        )

    result = e158.run("f", "h", "r", "s", tmp_path / "out", body_runner=runner)
    assert result["validity"] == "valid_body_contrast"
    assert result["physical_metrics"]["advancement"]["advance_to_closed_loop"]
    assert (tmp_path / "out/advancement.json").exists()
    assert (tmp_path / "out/output-manifest.json").exists()
    with np.load(tmp_path / "out/candidate-commands.npz") as raw:
        assert {
            "time_s",
            "baseline_command",
            "candidate_command",
            "baseline_activation",
            "candidate_activation",
            "ema40",
            "source_rate",
        } <= set(raw.files)


def complete_native():
    return {
        "time_s": np.arange(1, 100001) * 0.0001,
        "force_forward_bw": np.column_stack(
            [
                np.zeros((100000, 4)),
                np.r_[np.zeros(10000), -0.9 * np.ones(10000), np.zeros(80000)],
                np.zeros(100000),
            ]
        ),
        "normal_load_bw": np.ones((100000, 6)),
        "slip_mm_s": np.ones((100000, 6)),
        "slip_valid": np.ones((100000, 6), bool),
        "nonfoot_contact": np.zeros(100000, bool),
        "upright": np.ones(100000),
    }


def mark_complete_counts(ledger):
    ledger.update(
        body_constructor_attempts=1,
        body_constructor_completed=1,
        settling_physics_steps=2000,
        body_models=1,
        shadow_mjdata=1,
        scientific_steps_requested=100000,
        scientific_steps_completed=100000,
        scientific_samples_observed=100000,
        shadow_forward_requested=100000,
        shadow_forward_completed=100000,
    )


def test_actual_setup_branch_uses_analysis_observation_and_cleans_generated_body(
    monkeypatch, tmp_path
):
    paths = generated_run_inputs(tmp_path)
    paths["checkpoint"] = tmp_path / "checkpoint"
    freeze = fake_freeze(paths)
    monkeypatch.setattr(e158, "verified_inputs", lambda *args: (freeze, {}))
    monkeypatch.setattr(e158, "safe_path", lambda name: paths[name])
    marker = object()
    created = FakeBody()

    def construct(saved, observation, ledger):
        assert observation is marker
        ledger.update(
            body_constructor_attempts=1,
            body_constructor_completed=1,
            settling_physics_steps=2000,
        )
        return created

    def generated_execute(**kwargs):
        ledger = kwargs["ledger"]
        ledger.update(
            scientific_steps_requested=100000,
            scientific_steps_completed=100000,
            scientific_samples_observed=100000,
            shadow_forward_requested=100000,
            shadow_forward_completed=100000,
        )
        return kwargs["rows"], complete_native()

    gait = SimpleNamespace(
        load_model=lambda path: {},
        analyze=lambda data: (
            {"window": {"longest_forward_bout_s": 0.3}},
            None,
            None,
        ),
    )
    e113 = SimpleNamespace(
        body_quality=lambda *args, **kwargs: quality_fixture(),
        supported_slip=lambda *args, **kwargs: {
            "available": True,
            "load_weighted_mm_s": 2.0,
        },
    )
    dependencies = {
        "state_load": lambda path: {
            "control_tick": 0,
            "body": {"modelHash": "generated-body"},
        },
        "implicit_request": lambda payload, saved: None,
        "construct_body": construct,
        "observation": marker,
        "execute_body": generated_execute,
        "replay_command": object(),
        "observe_shadow": object(),
        "contact_diagnostics": object(),
        "capture_control": object(),
        "gait": gait,
        "e113": e113,
    }
    result = e158.run("f", "h", "r", "s", tmp_path / "out", dependencies=dependencies)
    assert result["validity"] == "valid_body_contrast"
    assert created.closed
    assert result["counts"]["settling_physics_steps"] == 2000


def test_health_stop_is_physical_and_retains_partial_raw(monkeypatch, tmp_path):
    paths = generated_run_inputs(tmp_path)
    freeze = fake_freeze(paths)
    monkeypatch.setattr(e158, "verified_inputs", lambda *args: (freeze, {}))
    monkeypatch.setattr(e158, "safe_path", lambda name: paths[name])

    def runner(**kwargs):
        kwargs["ledger"].update(
            scientific_steps_requested=1,
            scientific_steps_completed=1,
            scientific_samples_observed=1,
            shadow_forward_requested=1,
            shadow_forward_completed=1,
        )
        error = e158.NativeHealthStop("generated nonfoot contact")
        error.e158_native = {"time_s": np.array([0.0001]), "qpos": np.zeros((1, 3))}
        error.e158_trace_rows = [{"control_tick": 0}]
        raise error

    result = e158.run("f", "h", "r", "s", tmp_path / "out", body_runner=runner)
    assert result["validity"] == "measured_native_health_stop"
    assert result["hypothesis_verdict"] == "refuted_health_stop"
    assert result["body_executed"]
    assert result["technical_failure"] is False
    assert (tmp_path / "out/candidate-native-partial.npz").exists()
    assert (tmp_path / "out/candidate-trace-partial.json").exists()


def test_partial_save_failure_preserves_runner_error_and_attempts_trace(
    monkeypatch, tmp_path
):
    paths = generated_run_inputs(tmp_path)
    freeze = fake_freeze(paths)
    monkeypatch.setattr(e158, "verified_inputs", lambda *args: (freeze, {}))
    monkeypatch.setattr(e158, "safe_path", lambda name: paths[name])
    original_npz = e158.atomic_npz

    def fail_partial_npz(path, **arrays):
        if Path(path).name == "candidate-native-partial.npz":
            raise OSError("generated partial native save failure")
        return original_npz(path, **arrays)

    monkeypatch.setattr(e158, "atomic_npz", fail_partial_npz)

    def runner(**kwargs):
        kwargs["ledger"]["scientific_steps_requested"] = 1
        kwargs["ledger"]["scientific_steps_completed"] = 1
        error = RuntimeError("generated primary runner failure")
        error.e158_native = {"time_s": np.array([0.0001])}
        error.e158_trace_rows = [{"control_tick": 0}]
        raise error

    with pytest.raises(RuntimeError, match="primary runner"):
        e158.run("f", "h", "r", "s", tmp_path / "out", body_runner=runner)
    result = json.loads((tmp_path / "out/result.json").read_text())
    assert result["validity"] == "incomplete"
    assert result["body_executed"]
    assert result["primary_error"]["message"] == "generated primary runner failure"
    assert (tmp_path / "out/candidate-trace-partial.json").exists()


def test_final_ledger_failure_revokes_valid_startup_result(monkeypatch, tmp_path):
    paths = generated_run_inputs(tmp_path, reject=True)
    freeze = fake_freeze(paths)
    monkeypatch.setattr(e158, "verified_inputs", lambda *args: (freeze, {}))
    monkeypatch.setattr(e158, "safe_path", lambda name: paths[name])
    original = e158.atomic_json
    calls = 0

    def fail_third_ledger(path, value):
        nonlocal calls
        if Path(path).name == "ledger.json":
            calls += 1
            if calls == 3:
                raise OSError("generated final ledger failure")
        return original(path, value)

    monkeypatch.setattr(e158, "atomic_json", fail_third_ledger)
    with pytest.raises(OSError, match="final ledger"):
        e158.run("f", "h", "r", "s", tmp_path / "out", body_runner=lambda **k: None)
    result = json.loads((tmp_path / "out/result.json").read_text())
    assert result["validity"] == "incomplete"
    assert result["technical_failure"]


def test_post_manifest_deadline_has_fail_closed_unstale_manifest(monkeypatch, tmp_path):
    paths = generated_run_inputs(tmp_path, reject=True)
    freeze = fake_freeze(paths)
    monkeypatch.setattr(e158, "verified_inputs", lambda *args: (freeze, {}))
    monkeypatch.setattr(e158, "safe_path", lambda name: paths[name])
    original = e158.output_manifest

    def late_manifest(out):
        value = original(out)
        freeze["resource_limits"]["internal_s"] = -1
        return value

    monkeypatch.setattr(e158, "output_manifest", late_manifest)
    with pytest.raises(TimeoutError, match="post-manifest"):
        e158.run("f", "h", "r", "s", tmp_path / "out", body_runner=lambda **k: None)
    result = json.loads((tmp_path / "out/result.json").read_text())
    manifest = json.loads((tmp_path / "out/output-manifest.json").read_text())
    assert result["validity"] == "incomplete"
    assert "result.json" in manifest["files"]
    assert "ledger.json" in manifest["files"]
    for name, receipt in manifest["files"].items():
        assert e158.sha256(tmp_path / "out" / name) == receipt["sha256"]


def test_final_result_publication_failure_rewrites_fail_closed(monkeypatch, tmp_path):
    paths = generated_run_inputs(tmp_path, reject=True)
    freeze = fake_freeze(paths)
    monkeypatch.setattr(e158, "verified_inputs", lambda *args: (freeze, {}))
    monkeypatch.setattr(e158, "safe_path", lambda name: paths[name])
    original = e158.atomic_json
    failed = False

    def fail_first_result(path, value):
        nonlocal failed
        if Path(path).name == "result.json" and not failed:
            failed = True
            raise OSError("generated result publication failure")
        return original(path, value)

    monkeypatch.setattr(e158, "atomic_json", fail_first_result)
    with pytest.raises(OSError, match="result publication"):
        e158.run("f", "h", "r", "s", tmp_path / "out", body_runner=lambda **k: None)
    result = json.loads((tmp_path / "out/result.json").read_text())
    assert result["validity"] == "incomplete"
    assert result["technical_failure"]
    assert result["primary_error"]["message"] == "generated result publication failure"


@pytest.mark.parametrize("failure_stage", ["final_ledger", "manifest"])
def test_late_failure_revokes_all_advancement_and_covers_final_receipts(
    monkeypatch, tmp_path, failure_stage
):
    paths = generated_run_inputs(tmp_path)
    freeze = fake_freeze(paths)
    monkeypatch.setattr(e158, "verified_inputs", lambda *args: (freeze, {}))
    monkeypatch.setattr(e158, "safe_path", lambda name: paths[name])
    original_save = e158.atomic_json
    failed = False

    def publish(path, value):
        nonlocal failed
        selected = (
            failure_stage == "final_ledger"
            and path.name == "ledger.json"
            and value.get("status") == "body_complete"
        ) or (failure_stage == "manifest" and path.name == "output-manifest.json")
        if selected and not failed:
            failed = True
            raise OSError("generated late " + failure_stage)
        original_save(path, value)

    def runner(**kwargs):
        mark_complete_counts(kwargs["ledger"])
        return (
            [{} for _ in range(2001)],
            complete_native(),
            quality_fixture(),
            {"available": True, "load_weighted_mm_s": 2.0},
        )

    monkeypatch.setattr(e158, "atomic_json", publish)
    out = tmp_path / "out"
    with pytest.raises(OSError, match="generated late"):
        e158.run("f", "h", "r", "s", out, body_runner=runner)
    result = json.loads((out / "result.json").read_text())
    advancement = json.loads((out / "advancement.json").read_text())
    assert result["validity"] == "incomplete" and result["technical_failure"]
    assert result["advance_to_closed_loop"] is False
    assert result["physical_metrics"]["advancement"]["advance_to_closed_loop"] is False
    assert advancement["advance_to_closed_loop"] is False
    assert advancement["revoked_by"] == "technical_or_resource_failure"
    manifest = json.loads((out / "output-manifest.json").read_text())
    assert {"ledger.json", "result.json", "advancement.json"} <= set(manifest["files"])
    for name, receipt in manifest["files"].items():
        assert e158.sha256(out / name) == receipt["sha256"]


def test_complete_contrast_rejects_any_work_count_mismatch(monkeypatch, tmp_path):
    paths = generated_run_inputs(tmp_path)
    freeze = fake_freeze(paths)
    monkeypatch.setattr(e158, "verified_inputs", lambda *args: (freeze, {}))
    monkeypatch.setattr(e158, "safe_path", lambda name: paths[name])

    def runner(**kwargs):
        mark_complete_counts(kwargs["ledger"])
        kwargs["ledger"]["shadow_forward_completed"] -= 1
        return (
            [{} for _ in range(2001)],
            complete_native(),
            quality_fixture(),
            {"available": True, "load_weighted_mm_s": 2.0},
        )

    with pytest.raises(ValueError, match="Exact bounded"):
        e158.run("f", "h", "r", "s", tmp_path / "out", body_runner=runner)
    result = json.loads((tmp_path / "out/result.json").read_text())
    assert result["validity"] == "incomplete"


def test_verified_inputs_rejects_before_reading_bad_freeze(tmp_path):
    freeze = tmp_path / "freeze.json"
    review = tmp_path / "review.json"
    freeze.write_text("{}")
    review.write_text("{}")
    with pytest.raises(ValueError, match="SHA argument"):
        e158.verified_inputs(freeze, "bad", review, "bad")


def test_cli_requires_all_five_gate_arguments(monkeypatch):
    monkeypatch.setattr("sys.argv", ["e158"])
    with pytest.raises(SystemExit):
        e158.main()
