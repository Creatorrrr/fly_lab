"""Generated-only E167 tests. No original NPZ or native model is opened."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import analyze_c_banc_m3_attenuation_envelope as e167


def contract() -> dict:
    fixed = [4, 15, 16, 19, 30, 31, 34, 35, 46, 47, 52, 63, 64, 67, 78, 82, 83, 94, 95]
    return {
        "fixed_muscles": fixed,
        "variable_muscles": [i for i in range(96) if i not in fixed],
        "control_dt": 0.005,
        "clock_atol": 2e-8,
        "command_gain": 0.02,
        "maximum_attenuation": 0.5,
        "reconstruction_atol": 1e-12,
        "padding_floor_rad": 1e-10,
        "padding_epsilon_multiplier": 4096,
        "decision_margin_rad": 1e-9,
        "required_rms_fraction": 0.8,
    }


def generated() -> tuple[dict[str, np.ndarray], dict]:
    c = contract()
    membership = np.arange(391, dtype=np.int64) % 96
    bank = np.ones((2001, 391, 3), dtype=np.float64)
    kappa = np.ones(391, dtype=np.float64)
    slopes = np.zeros((96, 391, 3, 2), dtype=np.float64)
    slopes[membership, np.arange(391), :, :] = 0.1
    contribution = e167.contribution_sum(bank, slopes, kappa, membership)
    activation = contribution.copy()
    moment = np.zeros((42, 96), dtype=np.float64)
    for muscle in range(96):
        moment[7 * (muscle // 16), muscle] = 0.01 if muscle % 2 else -0.01
    rest = np.zeros(42, dtype=np.float64)
    recorded = rest + 0.02 * (activation @ moment.T)
    ticks = np.arange(2001, dtype=np.int64)
    indices = np.arange(100, 1924, dtype=np.int64)
    raw = {
        "bank": bank,
        "activation": activation,
        "original_activation": activation.copy(),
        "prior_bank": bank.copy(),
        "prior_activation": activation.copy(),
        "prior_activation_residual": np.zeros_like(activation),
        "complete": np.ones(2001, dtype=np.bool_),
        "bank_available": np.ones(2001, dtype=np.bool_),
        "activation_available": np.ones(2001, dtype=np.bool_),
        "original_activation_available": np.ones(2001, dtype=np.bool_),
        "motor_rate_available": np.ones(2001, dtype=np.bool_),
        "control_tick": ticks,
        "network_tick": ticks * 20,
        "time_s": ticks * 0.005,
        "target_indices": indices,
        "target": recorded[indices].copy(),
        "recorded_command": recorded,
        "moment": moment,
        "rest": rest,
        "membership": membership,
        "parameter_membership": membership.copy(),
        "slopes": slopes,
        "kappa": kappa,
        "contract_fixed_muscles": np.asarray(c["fixed_muscles"], dtype=np.int64),
    }
    return raw, c


def sections(raw: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    return {
        "recruitment": {
            "bank": raw["bank"],
            "activation": raw["activation"],
            "original_activation": raw["original_activation"],
            "complete": raw["complete"],
            "bank_available": raw["bank_available"],
            "activation_available": raw["activation_available"],
            "original_activation_available": raw["original_activation_available"],
            "motor_rate_available": raw["motor_rate_available"],
            "control_tick": raw["control_tick"],
            "network_tick": raw["network_tick"],
            "physics_time_s": raw["time_s"],
        },
        "prior_raw": {
            "bank": raw["prior_bank"],
            "reconstructed_activation": raw["prior_activation"],
            "activation_residual": raw["prior_activation_residual"],
            "target_indices": raw["target_indices"],
            "target": raw["target"],
            "recorded_command": raw["recorded_command"],
        },
        "adapter": {
            "moment": raw["moment"],
            "rest": raw["rest"],
            "membership": raw["membership"],
        },
        "parameters": {
            "membership": raw["parameter_membership"],
            "slopes": raw["slopes"],
            "kappa": raw["kappa"],
        },
    }


def test_positive_hinge_and_signed_moment_numeric_example() -> None:
    bank = np.array([[[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    slopes = np.zeros((2, 2, 3, 2))
    slopes[0, 0, 0] = [2.0, 1.0]  # high-low < 0 yet c=2*1+1*2=4
    slopes[1, 1, 0] = [1.0, 1.0]
    total = e167.contribution_sum(bank, slopes, np.ones(2), np.array([0, 1]))
    assert total.tolist() == [[4.0, 1.0]]
    raw, c = generated()
    raw["moment"][0] = 0.0
    raw["moment"][0, 0] = 1.0
    raw["moment"][0, 1] = -1.0
    raw["recorded_command"] = raw["rest"] + 0.02 * (raw["activation"] @ raw["moment"].T)
    result = e167.envelope(raw, c)
    expected_lo = 0.02 * (result["activation_lower"][0, 0] - raw["activation"][0, 1])
    expected_hi = 0.02 * (raw["activation"][0, 0] - result["activation_lower"][0, 1])
    assert np.isclose(result["q_lower_unpadded"][0, 0], expected_lo)
    assert np.isclose(result["q_upper_unpadded"][0, 0], expected_hi)
    assert result["padding"][0, 0] >= c["padding_floor_rad"]
    assert np.array_equal(
        result["activation_lower"][:, c["fixed_muscles"]],
        raw["activation"][:, c["fixed_muscles"]],
    )
    e167.validate_numerics(raw, result, c)


@pytest.mark.parametrize(
    "key,mutate,match",
    [
        ("bank", lambda a: a.__setitem__((0, 0, 0), -1.0), "negative bank"),
        ("activation", lambda a: a.__setitem__((0, 0), np.nan), "nonfinite"),
        ("complete", lambda a: a.__setitem__(0, False), "incomplete"),
        ("target_indices", lambda a: a.__setitem__(0, 101), "target indices"),
        ("slopes", lambda a: a.__setitem__((1, 0, 0, 0), 0.1), "ownership"),
        (
            "original_activation",
            lambda a: a.__setitem__((0, 4), 50.0),
            "fixed activation",
        ),
        ("recorded_command", lambda a: a.__setitem__((1, 0), 1.0), "decoder"),
    ],
)
def test_invalid_inputs_and_numerics(key, mutate, match) -> None:
    raw, c = generated()
    mutate(raw[key])
    if key == "recorded_command":
        e167.validate_inputs(raw, c)
        result = e167.envelope(raw, c)
        with pytest.raises(ValueError, match=match):
            e167.validate_numerics(raw, result, c)
    else:
        with pytest.raises(ValueError, match=match):
            e167.validate_inputs(raw, c)


def test_dtype_and_shape_are_not_coerced() -> None:
    raw, c = generated()
    raw["target_indices"] = raw["target_indices"].astype(np.float64)
    with pytest.raises(ValueError, match="dtype"):
        e167.validate_inputs(raw, c)
    raw, c = generated()
    raw["bank"] = raw["bank"][:-1]
    with pytest.raises(ValueError, match="shape"):
        e167.validate_inputs(raw, c)


def test_parameter_without_fixed_field_uses_frozen_contract() -> None:
    raw, c = generated()
    source = sections(raw)
    assert "fixed_muscles" not in source["parameters"]
    extracted = e167.raw_inputs(source, c)
    assert np.array_equal(extracted["contract_fixed_muscles"], c["fixed_muscles"])
    e167.validate_inputs(extracted, c)


def test_pointwise_joint_intervals_do_not_imply_joint_feasibility() -> None:
    # M=[[1,-1],[1,1]], a=[2,1], gain=.02: target(0,.06) lies
    # in both pointwise intervals but requires incompatible activations.
    a = np.array([2.0, 1.0])
    m = np.array([[1.0, -1.0], [1.0, 1.0]])
    lo = np.array([1.0, 0.5])
    q0 = 0.02 * (lo @ np.maximum(m, 0).T + a @ np.minimum(m, 0).T)
    q1 = 0.02 * (a @ np.maximum(m, 0).T + lo @ np.minimum(m, 0).T)
    target = np.array([0.0, 0.06])
    assert np.all((q0 <= target) & (target <= q1))
    assert not np.allclose(0.02 * (np.array([1.0, 1.0]) @ m.T), target)


def fake_run(
    monkeypatch,
    tmp_path: Path,
    raw: dict[str, np.ndarray],
    c: dict,
    *,
    fail_publish=False,
    raw_limit=100663296,
    expected_error: str | None = None,
) -> Path:
    freeze = {
        "counts": {"body_models": 0, "solver_calls": 0},
        "resource_limits": {"internal_s": 40, "raw_bytes": raw_limit, "free_bytes": 0},
        "design_sha256": "design",
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(e167, "validate_freeze", lambda *_: c)
    monkeypatch.setattr(e167, "frozen_hashes", lambda *_: {})
    monkeypatch.setattr(e167, "validate_gate", lambda *_: {"source_sha256": {}})
    monkeypatch.setattr(e167, "snapshot", lambda *_: {})
    monkeypatch.setattr(e167, "verify_snapshot", lambda *_: None)
    if fail_publish:

        def failed_publish(out, *_):
            (out / "result.json").write_text("stale", encoding="utf-8")
            (out / "manifest.json").write_text("stale", encoding="utf-8")
            raise OSError("late publication")

        monkeypatch.setattr(e167, "publish", failed_publish)
    out = tmp_path / "out"
    if fail_publish or expected_error:
        with pytest.raises(
            (OSError, ValueError), match=expected_error or "late publication"
        ):
            e167.run(
                freeze_path,
                "generated",
                gate_path,
                "generated",
                out,
                input_loader=lambda _: sections(raw),
            )
    else:
        e167.run(
            freeze_path,
            "generated",
            gate_path,
            "generated",
            out,
            input_loader=lambda _: sections(raw),
        )
    return out


def test_full_generated_run_saves_raw_before_numerical_verdict(
    monkeypatch, tmp_path
) -> None:
    raw, c = generated()
    raw["activation"][100, 0] += 0.01
    raw["prior_activation"][100, 0] += 0.01
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="contribution reconstruction"):
        fake_run(monkeypatch, tmp_path, raw, c)
    assert (out / "input-raw.npz").exists()
    assert (out / "envelope-raw.npz").exists()
    assert (out / "failure.json").exists()
    assert not (out / "result.json").exists()


def test_full_generated_run_success_and_late_failure(monkeypatch, tmp_path) -> None:
    raw, c = generated()
    out = fake_run(monkeypatch, tmp_path, raw, c)
    assert (
        json.loads((out / "result.json").read_text())["validity"] == "valid_diagnostic"
    )
    assert (out / "manifest.json").exists()
    next_path = tmp_path / "late"
    next_path.mkdir()
    out2 = fake_run(monkeypatch, next_path, raw, c, fail_publish=True)
    assert (out2 / "failure.json").exists()
    assert not (out2 / "result.json").exists()
    assert not (out2 / "manifest.json").exists()


def test_real_publish_late_byte_growth_invalidates_success(
    monkeypatch, tmp_path
) -> None:
    raw, c = generated()
    original_write = e167.atomic_json

    def growing_write(path, value):
        original_write(path, value)
        if path.name == "result.json":
            with path.open("ab") as stream:
                stream.write(b"x" * 1024 * 1024)

    monkeypatch.setattr(e167, "atomic_json", growing_write)
    out = fake_run(
        monkeypatch,
        tmp_path,
        raw,
        c,
        raw_limit=50_000_000,
        expected_error="raw byte limit",
    )
    assert (out / "input-raw.npz").exists()
    assert (out / "envelope-raw.npz").exists()
    assert json.loads((out / "failure.json").read_text())["stage"] == "publication"
    assert not (out / "result.json").exists()
    assert not (out / "manifest.json").exists()


def test_real_publish_late_clock_growth_invalidates_success(
    monkeypatch, tmp_path
) -> None:
    raw, c = generated()
    clock = [1.0]
    monkeypatch.setattr(e167.time, "perf_counter", lambda: clock[0])
    original_write = e167.atomic_json

    def slow_write(path, value):
        original_write(path, value)
        if path.name == "result.json":
            clock[0] = 50.0

    monkeypatch.setattr(e167, "atomic_json", slow_write)
    out = fake_run(
        monkeypatch, tmp_path, raw, c, expected_error="internal time exceeded"
    )
    assert json.loads((out / "failure.json").read_text())["stage"] == "publication"
    assert not (out / "result.json").exists()
    assert not (out / "manifest.json").exists()


def test_hash_gate_and_snapshot_checks(tmp_path, monkeypatch) -> None:
    data = tmp_path / "file"
    data.write_bytes(b"original")
    monkeypatch.setattr(e167, "ROOT", tmp_path)
    hashes = {"file": e167.sha256(data)}
    e167.verify_hashes(hashes)
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen file changed"):
        e167.verify_hashes(hashes)
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema": e167.GATE_SCHEMA,
                "approved": True,
                "blocking_defects": [],
                "accepts_single_attenuation_envelope": True,
                "original_processes_allowed": 1,
                "automatic_retries": 0,
                "freeze_sha256": "f",
                "design_sha256": "d",
                "source_sha256": {name: "fake" for name in e167.OWNED},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        e167.validate_gate(gate, e167.sha256(gate), "f", "d")


def test_final_disk_check_is_in_timed_region(monkeypatch, tmp_path) -> None:
    raw, c = generated()
    clock = [1.0]
    monkeypatch.setattr(e167.time, "perf_counter", lambda: clock[0])
    disk_usage = e167.shutil.disk_usage

    def slow_disk(path):
        usage = disk_usage(path)
        clock[0] = 50.0
        return usage

    monkeypatch.setattr(e167.shutil, "disk_usage", slow_disk)
    out = fake_run(
        monkeypatch, tmp_path, raw, c, expected_error="internal time exceeded"
    )
    assert json.loads((out / "failure.json").read_text())["stage"] == "publication"
    assert (out / "envelope-raw.npz").exists()
    assert not (out / "result.json").exists()
    assert not (out / "manifest.json").exists()


def test_published_timing_is_scoped_and_matches_return_value(
    monkeypatch, tmp_path
) -> None:
    clock = [1.0]
    monkeypatch.setattr(e167.time, "perf_counter", lambda: clock[0])
    write = e167.atomic_json

    def advancing_write(path, value):
        write(path, value)
        clock[0] += 1.0

    monkeypatch.setattr(e167, "atomic_json", advancing_write)
    value = e167.publish(
        tmp_path,
        {"validity": "valid_diagnostic"},
        {"internal_s": 40, "raw_bytes": 10000, "free_bytes": 0},
        0.0,
    )
    assert json.loads((tmp_path / "result.json").read_text()) == value
    assert value["elapsed_before_publication_s"] == 1.0
    assert "elapsed_s" not in value


def test_snapshot_pre_and_post_source_integrity(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(e167, "ROOT", tmp_path)
    design = tmp_path / "design.md"
    design.write_text("design", encoding="utf-8")
    sources = {}
    for name in e167.OWNED:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        sources[name] = e167.sha256(path)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text("freeze", encoding="utf-8")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text("gate", encoding="utf-8")
    freeze = {"design": "design.md", "design_sha256": e167.sha256(design)}
    gate = {"source_sha256": sources}
    out = tmp_path / "out"
    out.mkdir()
    approved = e167.snapshot(
        out, freeze, freeze_path, gate_path, e167.sha256(gate_path), gate
    )
    e167.verify_snapshot(out, freeze, freeze_path, gate_path, approved)
    (out / "source-snapshot" / "analyzer.py").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="post snapshot"):
        e167.verify_snapshot(out, freeze, freeze_path, gate_path, approved)


def test_script_help_from_unrelated_cwd_without_pythonpath(tmp_path) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    script = Path(e167.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--source-review-sha256" in completed.stdout
