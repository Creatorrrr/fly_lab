from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import verify_c_banc_m3_rm_proximal_unload as e162


def test_stable_quadratic_exact_degeneracies():
    assert e162.stable_quadratic_interval(0.0, 0.0, 0.0)[:2] == (
        -np.inf,
        np.inf,
    )
    assert e162.stable_quadratic_interval(0.0, 0.0, 1.0)[:2] == (
        np.inf,
        -np.inf,
    )
    assert e162.stable_quadratic_interval(0.0, 2.0, -4.0)[:2] == (
        -np.inf,
        2.0,
    )
    assert e162.stable_quadratic_interval(0.0, -2.0, -4.0)[:2] == (
        -2.0,
        np.inf,
    )


def test_stable_quadratic_avoids_cancellation():
    lower, upper, discriminant = e162.stable_quadratic_interval(1.0, -1e8, 1.0)
    assert discriminant > 0
    assert lower == pytest.approx(1e-8, rel=1e-15)
    assert upper == pytest.approx(1e8, rel=1e-15)
    assert lower * upper == pytest.approx(1.0)


def arrays99(residual=0.0, direction=0.0):
    target = np.zeros((99, 42), dtype=np.float64)
    baseline = np.full((99, 42), residual, dtype=np.float64)
    delta = np.full((99, 42), direction, dtype=np.float64)
    return baseline, target, delta


def test_analytic_intersection_hard_cap_and_margin_with_zero_direction():
    result = e162.analytic_lambda_intersection(*arrays99())
    assert result["feasible"]
    assert result["hard_upper_active"]
    assert result["lambda"] == pytest.approx(0.25 * (1 - 1e-12))
    assert result["gain_scale"] >= 0.75


def test_analytic_intersection_element_max_bound():
    q, target, direction = arrays99()
    direction[4, 7] = 1.0
    result = e162.analytic_lambda_intersection(q, target, direction)
    assert result["intersection_upper_unmargined"] == pytest.approx(0.04)
    assert result["element_active_upper"][4, 7]
    assert result["lambda"] < 0.04
    assert np.max(np.abs(q + result["lambda"] * direction - target)) < 0.04


def test_analytic_intersection_rms_bound_and_signed_b():
    q, target, direction = arrays99(direction=1.0)
    result = e162.analytic_lambda_intersection(q, target, direction)
    assert result["quadratic_active_upper"]
    assert result["intersection_upper_unmargined"] == pytest.approx(0.02)
    assert result["quadratic_b"] == 0.0
    q[:] = 0.01
    direction[:] = -1.0
    shifted = e162.analytic_lambda_intersection(q, target, direction)
    assert shifted["quadratic_b"] < 0
    assert shifted["quadratic_upper"] == pytest.approx(0.03)


def test_analytic_intersection_baseline_failure_forces_zero():
    q, target, direction = arrays99(residual=0.041, direction=-1.0)
    result = e162.analytic_lambda_intersection(q, target, direction)
    assert not result["baseline_feasible"]
    assert result["lambda"] == 0
    assert not result["feasible"]


def test_analytic_intersection_exact_rms_boundary_with_negative_direction():
    q, target, direction = arrays99(residual=0.02, direction=-0.1)
    result = e162.analytic_lambda_intersection(q, target, direction)
    assert result["baseline_feasible"]
    assert result["quadratic_c"] == 0.0
    assert result["lambda"] == pytest.approx(0.25 * (1 - 1e-12))


def test_build_candidate_preserves_t0_and_only_moment_column():
    command = np.arange(2001 * 42, dtype=np.float64).reshape(2001, 42) / 1000
    activation = np.zeros((2001, 96), dtype=np.float64)
    activation[:, 70] = np.linspace(0, 2, 2001)
    moment = np.zeros((42, 96), dtype=np.float64)
    moment[32, 70] = 1
    result = e162.build_proximal_candidate(command, activation, moment, 0.25)
    candidate = result["candidate_command"]
    assert np.array_equal(candidate[0], command[0])
    assert np.array_equal(
        np.delete(candidate, 32, axis=1), np.delete(command, 32, axis=1)
    )
    assert np.array_equal(result["direction"][0], np.zeros(42))
    assert candidate[-1, 32] == pytest.approx(command[-1, 32] - 0.25 * 0.02 * 2)


def test_group70_mapping_requires_exact_two_global_motors_and_moment():
    membership = np.arange(391, dtype=np.int64) % 96
    membership[membership == 70] = 69
    membership[[192, 245]] = 70
    motor_ids = np.arange(391, dtype=np.int64)
    motor_ids[[192, 245]] = [77368, 99838]
    moment = np.zeros((42, 96))
    moment[32, 70] = 1
    parameters = {"membership": membership, "motor_ids": motor_ids}
    provenance = {
        "membership": membership.copy(),
        "motor_ids": motor_ids.copy(),
        "moment": moment,
    }
    intervention = {
        "muscle": 70,
        "motor_columns": [192, 245],
        "graph_indices": [77368, 99838],
        "joint_row": 32,
        "moment": 1.0,
    }
    e162.validate_group70_mapping(parameters, provenance, intervention)
    provenance["moment"][31, 70] = 1e-30
    with pytest.raises(ValueError, match="moment"):
        e162.validate_group70_mapping(parameters, provenance, intervention)


def write_npz(path: Path, **values):
    np.savez(path, **values)


def generated_inputs(tmp_path: Path, *, mode="positive"):
    n = 2001
    activation = np.zeros((n, 96), dtype=np.float64)
    if mode != "zero_direction":
        activation[:, 70] = np.linspace(0, 2, n)
    moment = np.zeros((42, 96), dtype=np.float64)
    moment[32, 70] = 1
    rest = np.zeros(42, dtype=np.float64)
    baseline = rest[None, :] + 0.02 * (activation @ moment.T)
    tendon = np.zeros((n, 6), dtype=np.float64)
    target = baseline[1:100].copy()
    if mode == "zero":
        target[0, 32] = baseline[1, 32] + 0.04
    elif mode == "baseline_fail":
        target[:, 0] = 0.05
    rows = [
        {
            "time_s": i * 0.005,
            "control_tick": i,
            "joint_targets_rad": baseline[i].tolist(),
            "tendon_inputs": {
                key: float(tendon[i, j]) for j, key in enumerate(e162.TENDONS)
            },
            "qpos": np.zeros(3).tolist(),
            "qvel": np.zeros(3).tolist(),
            "ctrl": np.zeros(44).tolist(),
        }
        for i in range(n)
    ]
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"samples": rows}), encoding="utf-8")
    recruitment = tmp_path / "recruitment.npz"
    write_npz(
        recruitment,
        activation=activation,
        activation_available=np.ones(n, bool),
        complete=np.ones(n, bool),
        control_tick=np.arange(n),
        physics_time_s=np.arange(n) * 0.005,
    )
    membership = np.arange(391, dtype=np.int64) % 96
    membership[membership == 70] = 69
    membership[[192, 245]] = 70
    motor_ids = np.arange(391, dtype=np.int64) + 70000
    motor_ids[[192, 245]] = [77368, 99838]
    provenance = tmp_path / "provenance.npz"
    write_npz(
        provenance,
        membership=membership,
        motor_ids=motor_ids,
        moment=moment,
        rest=rest,
    )
    parameters = tmp_path / "parameters.npz"
    write_npz(parameters, membership=membership, motor_ids=motor_ids)
    startup = tmp_path / "startup.npz"
    write_npz(
        startup,
        target_command=target,
        target_tendon=tendon[1:100],
        actual_command=baseline[1:100],
        actual_LTM=tendon[1:100],
    )
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "quality-summary.json").write_text(
        json.dumps({"fixed_initial_heading_net_forward_mm": 10.0, "net_over_path": 0.8})
    )
    (baseline_dir / "supported-slip.json").write_text(
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
        "baseline": baseline_dir,
        "forces": forces,
        "checkpoint": tmp_path / "checkpoint",
    }


def thresholds():
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


def fake_freeze():
    return {
        "trace": "trace",
        "recruitment": "recruitment",
        "provenance": "provenance",
        "parameters": "parameters",
        "startup": "startup",
        "baseline_directory": "baseline",
        "baseline_forces": "forces",
        "checkpoint": "checkpoint",
        "body_hash": "generated-body",
        "intervention": {
            "muscle": 70,
            "motor_columns": [192, 245],
            "graph_indices": [77368, 99838],
            "joint_row": 32,
            "moment": 1.0,
            "command_gain": 0.02,
            "lambda_min": 0.0,
            "lambda_max": 0.25,
            "positive_upper_margin": 1e-12,
        },
        "startup_limits": {
            "q_RMS": 0.02,
            "q_max": 0.04,
            "LTM_RMS": 0.02,
            "LTM_max": 0.04,
        },
        "advancement": thresholds(),
        "sample_contract": {
            "clock_atol": 2e-8,
            "reconstruction_atol": 1e-10,
        },
        "resource_limits": {"internal_s": 260, "free_bytes": 0, "raw_bytes": 1 << 30},
        "prior_actual_original_s": 2660.748491299979,
        "prior_conservative_s": 2661,
        "input_sha256": {},
        "reused_source_sha256": {},
        "preservation_sha256": {},
    }


def patch_inputs(monkeypatch, freeze, paths):
    monkeypatch.setattr(e162, "verified_inputs", lambda *args: (freeze, {}))
    monkeypatch.setattr(e162, "safe_path", lambda name: paths[name])


def native_arrays():
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


def quality():
    return {
        "complete_10s_native_health": True,
        "fixed_initial_heading_net_forward_mm": 9.5,
        "net_over_path": 0.76,
        "longest_forward_bout_s": 0.3,
        "forward_eligible_complete_cycles": {leg: 1 for leg in e162.LEGS},
        "knee_width_deg": {"lf": 25, "lm": 0, "lh": 40, "rf": 30, "rm": 0, "rh": 35},
        "development_screens": {"F3_numeric": True},
    }


def complete_counts(ledger):
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


def complete_runner(**kwargs):
    complete_counts(kwargs["ledger"])
    return (
        [{} for _ in range(2001)],
        native_arrays(),
        quality(),
        {"available": True, "load_weighted_mm_s": 2.0},
    )


def test_lambda_zero_skips_body_and_saves_selection(monkeypatch, tmp_path):
    paths = generated_inputs(tmp_path, mode="zero")
    freeze = fake_freeze()
    patch_inputs(monkeypatch, freeze, paths)
    called = False

    def runner(**kwargs):
        nonlocal called
        called = True

    result = e162.run("f", "h", "r", "s", tmp_path / "out", body_runner=runner)
    assert result["validity"] == "valid_lambda_zero_no_body"
    assert result["lambda_value"] == 0
    assert not called
    assert result["counts"]["body_models"] == 0
    assert (tmp_path / "out/lambda-constraints.npz").exists()


def test_baseline_failure_skips_body(monkeypatch, tmp_path):
    paths = generated_inputs(tmp_path, mode="baseline_fail")
    freeze = fake_freeze()
    patch_inputs(monkeypatch, freeze, paths)
    result = e162.run(
        "f",
        "h",
        "r",
        "s",
        tmp_path / "out",
        body_runner=lambda **k: pytest.fail("body"),
    )
    assert result["validity"] == "valid_baseline_startup_failure"
    assert result["body_executed"] is False


def test_zero_direction_full_run_serializes_unbounded_interval(monkeypatch, tmp_path):
    paths = generated_inputs(tmp_path, mode="zero_direction")
    freeze = fake_freeze()
    patch_inputs(monkeypatch, freeze, paths)
    result = e162.run("f", "h", "r", "s", tmp_path / "out", body_runner=complete_runner)
    assert result["lambda_value"] == pytest.approx(0.25 * (1 - 1e-12))
    assert result["selection"]["quadratic_lower"] == {"unbounded": "negative_infinity"}
    assert result["selection"]["quadratic_upper"] == {"unbounded": "positive_infinity"}


def test_complete_generated_pipeline_and_verdict(monkeypatch, tmp_path):
    paths = generated_inputs(tmp_path)
    freeze = fake_freeze()
    patch_inputs(monkeypatch, freeze, paths)
    result = e162.run("f", "h", "r", "s", tmp_path / "out", body_runner=complete_runner)
    assert result["validity"] == "valid_body_contrast"
    assert result["advance_to_closed_loop"]
    assert result["hypothesis_verdict"] == "supported_for_closed_loop_followup"
    assert result["selection_exposures"] == 1
    assert (tmp_path / "out/output-manifest.json").exists()


class FakeMj:
    class MjData:
        def __init__(self, model):
            self.model = model


class FakeBody:
    def __init__(self):
        self.m = SimpleNamespace(opt=SimpleNamespace(timestep=0.0001))
        self.mj = FakeMj()
        self.d = SimpleNamespace(qpos=np.zeros(3), qvel=np.zeros(3), ctrl=np.zeros(44))
        self.closed = False

    def physics_time(self):
        return 0.0

    def close(self):
        self.closed = True


def generated_dependencies(body):
    marker = object()

    def construct(saved, observation, ledger):
        assert observation is marker
        ledger.update(
            body_constructor_attempts=1,
            body_constructor_completed=1,
            settling_physics_steps=2000,
        )
        return body

    def execute(**kwargs):
        kwargs["ledger"].update(
            scientific_steps_requested=100000,
            scientific_steps_completed=100000,
            scientific_samples_observed=100000,
            shadow_forward_requested=100000,
            shadow_forward_completed=100000,
        )
        return kwargs["rows"], native_arrays()

    return {
        "state_load": lambda path: {
            "control_tick": 0,
            "body": {"modelHash": "generated-body"},
        },
        "implicit_request": lambda payload, saved: None,
        "construct_body": construct,
        "observation": marker,
        "execute_body": execute,
        "replay_command": object(),
        "observe_shadow": object(),
        "contact_diagnostics": object(),
        "capture_control": object(),
        "gait": SimpleNamespace(
            load_model=lambda path: {},
            analyze=lambda data: (
                {"window": {"longest_forward_bout_s": 0.3}},
                None,
                None,
            ),
        ),
        "e113": SimpleNamespace(
            body_quality=lambda *args, **kwargs: quality(),
            supported_slip=lambda *args, **kwargs: {
                "available": True,
                "load_weighted_mm_s": 2.0,
            },
        ),
    }


def test_actual_body_adapter_branch_with_generated_dependencies(monkeypatch, tmp_path):
    paths = generated_inputs(tmp_path)
    freeze = fake_freeze()
    patch_inputs(monkeypatch, freeze, paths)
    body = FakeBody()
    deps = generated_dependencies(body)
    result = e162.run("f", "h", "r", "s", tmp_path / "out", dependencies=deps)
    assert result["validity"] == "valid_body_contrast"
    assert body.closed


def test_recruitment_clock_mismatch_blocks_selection_and_body(monkeypatch, tmp_path):
    paths = generated_inputs(tmp_path)
    with np.load(paths["recruitment"]) as loaded:
        values = {key: loaded[key] for key in loaded.files}
    values["physics_time_s"] = values["physics_time_s"].copy()
    values["physics_time_s"][100] += 2.1e-8
    write_npz(paths["recruitment"], **values)
    freeze = fake_freeze()
    patch_inputs(monkeypatch, freeze, paths)
    with pytest.raises(ValueError, match="clock/ticks"):
        e162.run(
            "f",
            "h",
            "r",
            "s",
            tmp_path / "out",
            body_runner=lambda **kwargs: pytest.fail("body must remain blocked"),
        )
    ledger = json.loads((tmp_path / "out/ledger.json").read_text())
    assert ledger["selection_exposures"] == 0


def test_in_tolerance_physical_clock_drift_is_accepted(monkeypatch, tmp_path):
    paths = generated_inputs(tmp_path)
    with np.load(paths["recruitment"]) as loaded:
        values = {key: loaded[key] for key in loaded.files}
    values["physics_time_s"] = values["physics_time_s"].copy()
    values["physics_time_s"][100] += 1e-12
    write_npz(paths["recruitment"], **values)
    payload = json.loads(paths["trace"].read_text())
    payload["samples"][100]["time_s"] += 1e-12
    paths["trace"].write_text(json.dumps(payload))
    freeze = fake_freeze()
    patch_inputs(monkeypatch, freeze, paths)
    result = e162.run("f", "h", "r", "s", tmp_path / "out", body_runner=complete_runner)
    assert result["validity"] == "valid_body_contrast"


def test_cleanup_failure_revokes_complete_physical_result(monkeypatch, tmp_path):
    paths = generated_inputs(tmp_path)
    freeze = fake_freeze()
    patch_inputs(monkeypatch, freeze, paths)

    class CloseErrorBody(FakeBody):
        def close(self):
            self.closed = True
            raise OSError("generated close failure")

    body = CloseErrorBody()
    with pytest.raises(OSError, match="generated close failure"):
        e162.run(
            "f",
            "h",
            "r",
            "s",
            tmp_path / "out",
            dependencies=generated_dependencies(body),
        )
    result = json.loads((tmp_path / "out/result.json").read_text())
    assert result["validity"] == "incomplete"
    assert result["advance_to_closed_loop"] is False
    assert result["body_executed"] is True
    assert body.closed


def test_startup_actual_mismatch_blocks_selection_and_body(monkeypatch, tmp_path):
    paths = generated_inputs(tmp_path)
    with np.load(paths["startup"]) as loaded:
        values = {key: loaded[key] for key in loaded.files}
    values["actual_command"] = values["actual_command"].copy()
    values["actual_command"][0, 0] = 1e-15
    write_npz(paths["startup"], **values)
    freeze = fake_freeze()
    patch_inputs(monkeypatch, freeze, paths)
    with pytest.raises(ValueError, match="actual startup"):
        e162.run(
            "f",
            "h",
            "r",
            "s",
            tmp_path / "out",
            body_runner=lambda **k: pytest.fail("body"),
        )
    ledger = json.loads((tmp_path / "out/ledger.json").read_text())
    assert ledger["selection_exposures"] == 0


def test_decoder_reconstruction_error_saves_raw_then_blocks_selection(
    monkeypatch, tmp_path
):
    paths = generated_inputs(tmp_path)
    payload = json.loads(paths["trace"].read_text())
    payload["samples"][10]["joint_targets_rad"][8] = 1e-6
    paths["trace"].write_text(json.dumps(payload))
    freeze = fake_freeze()
    patch_inputs(monkeypatch, freeze, paths)
    with pytest.raises(ValueError, match="reconstruction"):
        e162.run(
            "f",
            "h",
            "r",
            "s",
            tmp_path / "out",
            body_runner=lambda **kwargs: pytest.fail("body must remain blocked"),
        )
    assert (tmp_path / "out/baseline-reconstruction.npz").exists()
    ledger = json.loads((tmp_path / "out/ledger.json").read_text())
    assert ledger["selection_exposures"] == 0


def test_partial_runner_error_preserves_raw_and_primary(monkeypatch, tmp_path):
    paths = generated_inputs(tmp_path)
    freeze = fake_freeze()
    patch_inputs(monkeypatch, freeze, paths)

    def runner(**kwargs):
        kwargs["ledger"].update(
            scientific_steps_requested=1, scientific_steps_completed=1
        )
        error = RuntimeError("generated body failure")
        error.e158_native = {"time_s": np.array([0.0001]), "qpos": np.zeros((1, 3))}
        error.e158_trace_rows = [{"control_tick": 0}]
        raise error

    with pytest.raises(RuntimeError, match="generated body"):
        e162.run("f", "h", "r", "s", tmp_path / "out", body_runner=runner)
    result = json.loads((tmp_path / "out/result.json").read_text())
    assert result["validity"] == "incomplete"
    assert result["body_executed"]
    assert (tmp_path / "out/candidate-native-partial.npz").exists()
    assert (tmp_path / "out/candidate-trace-partial.json").exists()


def test_health_stop_still_runs_terminal_integrity_verification(monkeypatch, tmp_path):
    paths = generated_inputs(tmp_path)
    freeze = fake_freeze()
    calls = 0

    def verify(*args):
        nonlocal calls
        calls += 1
        return freeze, {}

    monkeypatch.setattr(e162, "verified_inputs", verify)
    monkeypatch.setattr(e162, "safe_path", lambda name: paths[name])

    def runner(**kwargs):
        kwargs["ledger"].update(
            scientific_steps_requested=3,
            scientific_steps_completed=2,
            scientific_samples_observed=2,
        )
        error = e162.e160.NativeHealthStop("generated health stop")
        error.e158_native = {"time_s": np.array([0.0001, 0.0002])}
        error.e158_trace_rows = [{"control_tick": 0}, {"control_tick": 1}]
        raise error

    result = e162.run("f", "h", "r", "s", tmp_path / "out", body_runner=runner)
    assert result["validity"] == "measured_native_health_stop"
    assert result["advance_to_closed_loop"] is False
    assert calls == 2
    ledger = json.loads((tmp_path / "out/ledger.json").read_text())
    assert ledger["post_hashes_verified"] is True


@pytest.mark.parametrize(
    "receipt", ["ledger.json", "result.json", "output-manifest.json"]
)
def test_persistent_late_receipt_failures_never_leave_success(
    monkeypatch, tmp_path, receipt
):
    paths = generated_inputs(tmp_path)
    freeze = fake_freeze()
    patch_inputs(monkeypatch, freeze, paths)
    original = e162.atomic_json
    armed = False

    def fail_after_physical_success(path, value):
        nonlocal armed
        name = Path(path).name
        if name == "ledger.json" and value.get("status") == "body_complete":
            armed = True
        if name == "result.json" and value.get("validity") == "valid_body_contrast":
            armed = True
        if name == "output-manifest.json":
            armed = True
        if name == receipt and armed:
            raise OSError("generated repeated " + receipt)
        return original(path, value)

    monkeypatch.setattr(e162, "atomic_json", fail_after_physical_success)
    with pytest.raises(OSError, match="generated repeated"):
        e162.run("f", "h", "r", "s", tmp_path / "out", body_runner=complete_runner)
    if (tmp_path / "out/result.json").exists():
        result = json.loads((tmp_path / "out/result.json").read_text())
        assert result["validity"] == "incomplete"
        assert result["advance_to_closed_loop"] is False
    if (tmp_path / "out/advancement.json").exists():
        advance = json.loads((tmp_path / "out/advancement.json").read_text())
        assert advance["advance_to_closed_loop"] is False


def test_manifest_then_persistent_advancement_rewrite_failure_quarantines_success(
    monkeypatch, tmp_path
):
    paths = generated_inputs(tmp_path)
    freeze = fake_freeze()
    patch_inputs(monkeypatch, freeze, paths)
    original = e162.atomic_json

    def fail_publication(path, value):
        name = Path(path).name
        if name == "output-manifest.json":
            raise OSError("generated manifest failure")
        if name == "advancement.json" and value["advance_to_closed_loop"] is False:
            raise OSError("generated advancement rewrite failure")
        return original(path, value)

    monkeypatch.setattr(e162, "atomic_json", fail_publication)
    with pytest.raises(OSError, match="generated manifest failure"):
        e162.run("f", "h", "r", "s", tmp_path / "out", body_runner=complete_runner)
    assert not (tmp_path / "out/result.json").exists()
    assert not (tmp_path / "out/advancement.json").exists()
    stale_results = list((tmp_path / "out").glob("result.json.invalid-stale-*.json"))
    assert stale_results


def test_cli_requires_gate_arguments(monkeypatch):
    monkeypatch.setattr("sys.argv", ["e162"])
    with pytest.raises(SystemExit):
        e162.main()


@pytest.mark.parametrize("kind", ["json", "npz"])
def test_atomic_replace_failure_preserves_primary_error_and_bytes(
    monkeypatch, tmp_path, kind
):
    def fail_replace(self, target):
        raise OSError("primary replace failure")

    def fail_cleanup(self, *args, **kwargs):
        raise PermissionError("secondary cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        patch.setattr(Path, "unlink", fail_cleanup)
        with pytest.raises(OSError, match="primary replace failure"):
            if kind == "json":
                e162.atomic_json(tmp_path / "receipt.json", {"evidence": 162})
            else:
                e162.atomic_npz(tmp_path / "raw.npz", evidence=np.array([162]))
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    if kind == "json":
        assert json.loads(files[0].read_text()) == {"evidence": 162}
    else:
        with np.load(files[0], allow_pickle=False) as saved:
            assert saved["evidence"].tolist() == [162]


def test_quarantine_rename_failure_copies_evidence_before_removal(
    monkeypatch, tmp_path
):
    paths = generated_inputs(tmp_path)
    freeze = fake_freeze()
    patch_inputs(monkeypatch, freeze, paths)
    original_json = e162.atomic_json
    original_replace = Path.replace

    def publication_failure(path, value):
        if Path(path).name == "output-manifest.json":
            raise OSError("parent manifest failure")
        return original_json(path, value)

    def rename_failure(self, target):
        if ".invalid-stale-" in Path(target).name:
            raise PermissionError("generated quarantine rename denied")
        return original_replace(self, target)

    monkeypatch.setattr(e162, "atomic_json", publication_failure)
    monkeypatch.setattr(Path, "replace", rename_failure)
    with pytest.raises(OSError, match="parent manifest failure"):
        e162.run("f", "h", "r", "s", tmp_path / "out", body_runner=complete_runner)
    assert not (tmp_path / "out/result.json").exists()
    stale = list((tmp_path / "out").glob("result.json.invalid-stale-*.json"))
    assert any(
        json.loads(path.read_text())["validity"] == "valid_body_contrast"
        for path in stale
    )


def test_measured_numpy_bout_serializes_full_decision_and_return_value(
    monkeypatch, tmp_path
):
    paths = generated_inputs(tmp_path)
    freeze = fake_freeze()
    patch_inputs(monkeypatch, freeze, paths)

    def runner(**kwargs):
        rows, native, measured, slip = complete_runner(**kwargs)
        measured["longest_forward_bout_s"] = np.float64(0.4)
        return rows, native, measured, slip

    result = e162.run("f", "h", "r", "s", tmp_path / "out", body_runner=runner)
    assert result["validity"] == "valid_body_contrast"
    decision = result["physical_metrics"]["advancement"]
    assert decision["criteria"]["longest_bout"] is True
    assert json.loads(json.dumps(result, allow_nan=False))["advance_to_closed_loop"]
    saved = json.loads((tmp_path / "out/advancement.json").read_text())
    assert saved == decision
