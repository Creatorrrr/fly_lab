"""Generated-only verification of E168 causal resource and single SOCP."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import fit_c_banc_m3_resource_attenuation as e168


def generated_e164_module():
    path = e168.ROOT / "tests" / "test_c_banc_onset_recruitment.py"
    spec = importlib.util.spec_from_file_location("generated_e164_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generated_full():
    fixture = generated_e164_module()
    data = fixture.generated_full_inputs()
    base = fixture.generated_full_freeze(data)
    freeze = json.loads(
        (
            e168.ROOT
            / "verification/banc-natural-gait-m3-20260919/e168-resource-attenuation-freeze.json"
        ).read_text(encoding="utf-8")
    )
    freeze["sample_contract"] = base["sample_contract"]
    raw: dict[str, np.ndarray] = {}
    pending: list[tuple[bool, str]] = []
    saved = e168.e164.prepare_saved(data, base, raw, pending)
    saved["group_legs"] = np.asarray(base["sample_contract"]["group_legs"])
    e168.resource_features(data, saved, freeze, raw, pending)
    e168.enforce(pending)
    return data, base, freeze, saved, raw


def test_held_endpoint_step_recovery_and_zero_conventions():
    rate = np.array(
        [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    kappa = np.array([1.0, 0.0, 0.0])
    r, drive = e168.held_resource(rate, kappa)
    assert np.array_equal(r[0], np.ones(3))
    assert drive[1, 0] == 0.5 and drive[1, 1] == 1 and drive[1, 2] == 0
    expected = 2 / 3 + (1 - 2 / 3) * math.exp(-1.5 * 0.005 / 0.07)
    assert r[1, 0] == pytest.approx(expected)
    assert r[2, 0] < r[1, 0] and r[3, 0] > r[2, 0]
    assert r[1, 1] < 1 and r[2, 1] > r[1, 1]
    assert np.array_equal(r[:, 2], np.ones(4))
    assert np.min(r) >= 0.5 and np.max(r) <= 1


def test_constant_drive_closed_form_and_bounds():
    rate = np.ones((101, 2), dtype=np.float64)
    kappa = np.array([1.0, 0.0])
    resource, drive = e168.held_resource(rate, kappa)
    for column, s in enumerate((0.5, 1.0)):
        steady = 1 / (1 + s)
        expected = steady + (1 - steady) * np.exp(
            -(1 + s) * np.arange(101) * 0.005 / 0.07
        )
        np.testing.assert_allclose(resource[:, column], expected, rtol=0, atol=2e-15)
    assert drive[1, 1] == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0, 1e308])
def test_held_input_rejects_invalid_or_overflow(bad):
    rate = np.ones((3, 2), dtype=np.float64)
    rate[1, 0] = bad
    with pytest.raises(ValueError):
        e168.held_resource(rate, np.array([1e308, 1.0]))


def test_positive_branch_and_post_update_feature_owned():
    membership = np.resize(np.arange(96), 391).astype(np.int64)
    slopes = np.zeros((96, 391, 3, 2))
    slopes[membership, np.arange(391), :, 0] = 2
    slopes[membership, np.arange(391), :, 1] = 1
    bank = np.zeros((2, 391, 3))
    bank[1, 0] = [0.5, 1.0, 2.0]
    kappa = np.ones(391)
    contribution = e168.positive_contribution(bank, slopes, kappa, membership)
    assert contribution[1, 0] == pytest.approx(2 * (0.5 + 1 + 1) + 1)
    resource, _ = e168.held_resource(np.ones((2, 391)), kappa)
    phi = (1 - resource) * contribution
    assert phi[0, 0] == 0 and phi[1, 0] > 0
    assert phi[1, 0] <= 0.5 * contribution[1, 0]


def test_generated_lambda_zero_fixed_and_signed_affine():
    _, _, freeze, saved, raw = generated_full()
    zero_metrics, zero = e168.candidate(saved, np.zeros(330))
    assert np.array_equal(zero["candidate_activation"], saved["activation"])
    assert np.array_equal(zero["candidate_command"], saved["recorded_command"])
    assert zero_metrics == saved["baseline_metrics"]
    assert np.max(np.abs(zero["decoder_residual"])) <= 1e-12
    lam = np.linspace(0, 0.8, 330)
    _, altered = e168.candidate(saved, lam)
    assert np.array_equal(
        altered["candidate_activation"][:, saved["fixed_muscles"]],
        saved["activation"][:, saved["fixed_muscles"]],
    )
    assert np.min(altered["candidate_delta_activation"]) < 0
    for leg in range(6):
        ticks = np.array([1, 100, 1923])
        matrix, constant = e168.leg_affine(saved, leg, ticks)
        joints = slice(7 * leg, 7 * (leg + 1))
        np.testing.assert_allclose(
            (matrix @ lam + constant).reshape(-1, 7),
            altered["candidate_command"][ticks, joints],
            rtol=0,
            atol=1e-14,
        )
    pending = []
    problem = e168.build_problem(saved, freeze, raw, pending)
    e168.enforce(pending)
    assert problem.P.shape == (331, 331) and problem.A.shape[1] == 331
    assert len(problem.criteria) == 15 and len(problem.cones) == 16
    assert np.array_equal(problem.lower, np.r_[np.zeros(330), -1])
    assert np.array_equal(problem.upper, np.r_[np.ones(330), 1])
    assert np.array_equal(raw["baseline_witness"], np.r_[np.zeros(330), -0.251])
    assert all(key.startswith("qr_") for key in raw if key.startswith("qr_"))
    assert np.max(np.abs(altered["decoder_residual"])) <= 1e-12


def test_all_history_selected_lambda_protection_and_exact_bounds():
    _, _, freeze, saved, _ = generated_full()
    lam = np.full(330, 0.2)
    metrics, arrays = e168.candidate(saved, lam)
    gates = e168.candidate_gates(saved, freeze, np.r_[lam, -0.251], metrics, arrays)
    assert gates["lambda_exact_bounds"] and gates["all_history_motor_floor"]
    assert gates["all_history_muscle_budget"] and gates["fixed_exact"]
    high = lam.copy()
    high[0] = 1.0 + 1e-15
    assert not e168.candidate_gates(saved, freeze, np.r_[high, 0], metrics, arrays)[
        "lambda_exact_bounds"
    ]
    assert not e168.candidate_gates(saved, freeze, np.r_[lam, 1.001], metrics, arrays)[
        "margin_bounds"
    ]


class FakeSolver:
    instances = 0
    solves = 0

    def __init__(self, _p, _q, a, b, _cones, settings):
        type(self).instances += 1
        self.a, self.b, self.settings = a, b, settings

    def set_termination_callback(self, callback):
        self.callback = callback

    def solve(self):
        type(self).solves += 1
        x = np.r_[np.zeros(330), -0.251]
        return SimpleNamespace(
            x=x,
            s=self.b - self.a @ x,
            z=np.zeros(self.a.shape[0]),
            status="AlmostSolved",
            iterations=1,
            solve_time=0.01,
            obj_val=0.251,
            obj_val_dual=0.251,
            r_prim=0.0,
            r_dual=0.0,
        )

    def get_settings(self):
        return self.settings


def test_generated_full_run_raw_dual_and_receipts(monkeypatch, tmp_path):
    data, base, freeze, _, _ = generated_full()
    freeze_path = tmp_path / "freeze.json"
    gate_path = tmp_path / "gate.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    gate_path.write_text("{}", encoding="utf-8")
    gate = {
        "source_sha256": {
            name: e168.sha256(e168.frozen_path(name)) for name in e168.OWNED
        }
    }
    monkeypatch.setattr(e168, "validate_freeze", lambda *_: (freeze, base, {}))
    monkeypatch.setattr(e168, "validate_gate", lambda *_: gate)
    FakeSolver.instances = FakeSolver.solves = 0
    out = tmp_path / "out"
    result = e168.run(
        freeze_path,
        "generated",
        gate_path,
        "generated",
        out,
        input_loader=lambda _: data,
        solver_factory=FakeSolver,
    )
    assert result["validity"] == "valid"
    assert result["solver_status"] == "AlmostSolved"
    assert (
        FakeSolver.instances == FakeSolver.solves == result["counts"]["fit_calls"] == 1
    )
    assert result["counts"]["resource_state_update_steps"] == 2000
    for name in (
        "pre-solve-raw.npz",
        "solver-return-raw.npz",
        "solver-derived-raw.npz",
        "candidate-raw.npz",
        "result.json",
        "ledger.json",
        "output-manifest.json",
    ):
        assert (out / name).exists()
    with np.load(out / "solver-derived-raw.npz") as raw:
        assert np.isfinite(raw["projected_dual_margin_upper_bound"])
        assert raw["solver_x"].shape == (331,)
    with np.load(out / "pre-solve-raw.npz") as raw:
        assert raw["problem_q"].shape == (331,)
        assert raw["resource"].shape == (2001, 391)


def test_generated_clarabel_one_solve_and_calibrated_dual():
    _, _, freeze, saved, raw = generated_full()
    pending = []
    problem = e168.build_problem(saved, freeze, raw, pending)
    e168.enforce(pending)
    solution, history, settings = e168.e145.solve_conic_problem(problem, freeze)
    assert len(solution.x) == 331 and history
    assert settings["max_threads"] == 1 and settings["direct_solve_method"] == "qdldl"
    derived = e168.e164.solver_raw(problem, solution, history, freeze)
    assert np.isfinite(derived["projected_dual_margin_upper_bound"])
    assert derived["solver_z_projected"].shape == (problem.A.shape[0],)


def test_gate_requires_exact_approved_source_hashes(tmp_path):
    freeze = {"design_sha256": e168.DESIGN_SHA}
    gate = {
        "schema": e168.GATE_SCHEMA,
        "approved": True,
        "blocking_defects": [],
        "accepts_single_resource_fit": True,
        "original_processes_allowed": 1,
        "automatic_retries": 0,
        "freeze_sha256": e168.FREEZE_SHA,
        "design_sha256": e168.DESIGN_SHA,
        "source_sha256": {
            name: e168.sha256(e168.frozen_path(name)) for name in e168.OWNED
        },
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    assert e168.validate_gate(path, e168.sha256(path), freeze) == gate
    gate["approved"] = False
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="blocks execution"):
        e168.validate_gate(path, e168.sha256(path), freeze)


def generated_gate(monkeypatch, tmp_path):
    data, base, freeze, _, _ = generated_full()
    freeze_path = tmp_path / "freeze.json"
    gate_path = tmp_path / "gate.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    gate_path.write_text("{}", encoding="utf-8")
    gate = {
        "source_sha256": {
            name: e168.sha256(e168.frozen_path(name)) for name in e168.OWNED
        }
    }
    monkeypatch.setattr(e168, "validate_freeze", lambda *_: (freeze, base, {}))
    monkeypatch.setattr(e168, "validate_gate", lambda *_: gate)
    return data, freeze_path, gate_path, tmp_path / "out"


def test_returned_bare_raw_survives_post_return_io_failure(monkeypatch, tmp_path):
    data, freeze_path, gate_path, out = generated_gate(monkeypatch, tmp_path)
    original = e168.atomic_json
    hit = False

    def fail_return(path, value):
        nonlocal hit
        if (
            not hit
            and path.name == "ledger.json"
            and value.get("status") == "solve_returned"
        ):
            hit = True
            raise OSError("generated post-return write failure")
        return original(path, value)

    monkeypatch.setattr(e168, "atomic_json", fail_return)
    with pytest.raises(OSError, match="generated post-return"):
        e168.run(
            freeze_path,
            "generated",
            gate_path,
            "generated",
            out,
            input_loader=lambda _: data,
            solver_factory=FakeSolver,
        )
    assert hit
    with np.load(out / "solver-partial.npz") as raw:
        assert raw["solver_x"].shape == (331,)
    receipt = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert receipt["validity"] == "incomplete" and receipt["counts"]["fit_calls"] == 1
    assert receipt["counts"]["solver_returns"] == 1


def test_late_manifest_failure_revokes_success(monkeypatch, tmp_path):
    data, freeze_path, gate_path, out = generated_gate(monkeypatch, tmp_path)
    original = e168.output_manifest
    hit = False

    def bad_manifest(path):
        nonlocal hit
        value = original(path)
        if not hit and "candidate-raw.npz" in value["files"]:
            hit = True
            value["files"]["candidate-raw.npz"]["sha256"] = "0" * 64
        return value

    monkeypatch.setattr(e168, "output_manifest", bad_manifest)
    with pytest.raises(ValueError, match="output manifest changed"):
        e168.run(
            freeze_path,
            "generated",
            gate_path,
            "generated",
            out,
            input_loader=lambda _: data,
            solver_factory=FakeSolver,
        )
    assert hit
    receipt = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert receipt["validity"] == "incomplete"
    assert receipt["conditional_resource_proposal"] is False
    assert receipt["primary_error"]["message"].startswith(
        "E168 output manifest changed"
    )
    assert list(out.glob("result.json.invalid-stale-*"))


@pytest.mark.parametrize(
    ("case", "raw_key", "dtype", "message"),
    [
        (
            "nan_activation",
            "loaded_recruitment_activation",
            np.float64,
            "nonfinite input",
        ),
        (
            "complex_original",
            "loaded_recruitment_original_activation",
            np.complex128,
            "shape/dtype differs",
        ),
        (
            "float32_clock",
            "loaded_recruitment_physics_time_s",
            np.float32,
            "shape/dtype differs",
        ),
        (
            "float_membership",
            "loaded_adapter_membership",
            np.float64,
            "shape/dtype differs",
        ),
    ],
)
def test_source_values_are_saved_before_dtype_finite_failure(
    monkeypatch, tmp_path, case, raw_key, dtype, message
):
    data, freeze_path, gate_path, out = generated_gate(monkeypatch, tmp_path)
    if case == "nan_activation":
        data["recruitment"]["activation"][1, 0] = np.nan
    elif case == "complex_original":
        data["recruitment"]["original_activation"] = (
            data["recruitment"]["original_activation"].astype(np.complex128) + 1j
        )
    elif case == "float32_clock":
        data["recruitment"]["physics_time_s"] = data["recruitment"][
            "physics_time_s"
        ].astype(np.float32)
    else:
        data["adapter"]["membership"] = data["adapter"]["membership"].astype(np.float64)
    FakeSolver.instances = FakeSolver.solves = 0
    with pytest.raises(ValueError, match=message):
        e168.run(
            freeze_path,
            "generated",
            gate_path,
            "generated",
            out,
            input_loader=lambda _: data,
            solver_factory=FakeSolver,
        )
    with np.load(out / "pre-solve-raw.npz", allow_pickle=False) as raw:
        assert raw[raw_key].dtype == np.dtype(dtype)
        if case == "nan_activation":
            assert np.isnan(raw[raw_key][1, 0])
        if case == "complex_original":
            assert np.all(raw[raw_key].imag == 1)
    receipt = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert receipt["validity"] == "incomplete"
    assert (
        receipt["counts"]["fit_calls"] == FakeSolver.instances == FakeSolver.solves == 0
    )


def test_unavailable_phase_outside_targets_remains_valid(monkeypatch, tmp_path):
    data, freeze_path, gate_path, out = generated_gate(monkeypatch, tmp_path)
    data["phase_diagnostic"]["e1_phase"][0, 0] = np.nan
    data["phase_diagnostic"]["e1_phase_available"][0, 0] = False
    result = e168.run(
        freeze_path,
        "generated",
        gate_path,
        "generated",
        out,
        input_loader=lambda _: data,
        solver_factory=FakeSolver,
    )
    assert result["validity"] == "valid"
    with np.load(out / "pre-solve-raw.npz", allow_pickle=False) as raw:
        assert np.isnan(raw["loaded_phase_diagnostic_e1_phase"][0, 0])


def test_completed_resource_survives_later_feature_failure(monkeypatch, tmp_path):
    data, freeze_path, gate_path, out = generated_gate(monkeypatch, tmp_path)

    def fail_contribution(*_args):
        raise RuntimeError("generated failure after resource")

    monkeypatch.setattr(e168, "positive_contribution", fail_contribution)
    FakeSolver.instances = FakeSolver.solves = 0
    with pytest.raises(RuntimeError, match="after resource"):
        e168.run(
            freeze_path,
            "generated",
            gate_path,
            "generated",
            out,
            input_loader=lambda _: data,
            solver_factory=FakeSolver,
        )
    receipt = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert receipt["counts"]["resource_state_update_steps"] == 2000
    assert receipt["counts"]["fit_calls"] == 0
    with np.load(out / "input-partial-raw.npz", allow_pickle=False) as raw:
        assert raw["resource"].shape == (2001, 391)
        assert np.isfinite(raw["resource"]).all()
        assert np.isfinite(raw["resource_drive"]).all()


def test_interrupted_resource_retains_completed_steps_in_memory(monkeypatch):
    rate = np.ones((5, 2), dtype=np.float64)
    kappa = np.ones(2, dtype=np.float64)
    raw = {}
    progress = {}
    original = np.exp
    calls = 0

    def fail_on_third(value):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("generated resource interruption")
        return original(value)

    monkeypatch.setattr(e168.np, "exp", fail_on_third)
    with pytest.raises(RuntimeError, match="resource interruption"):
        e168.held_resource(rate, kappa, progress=progress, raw=raw)
    assert progress["resource_state_update_steps"] == 2
    assert np.isfinite(raw["resource"][:3]).all()
    assert np.isnan(raw["resource"][3:]).all()
    assert raw["resource_drive"].shape == rate.shape


def test_unconverted_checkpoint_write_failure_prevents_prepare_and_fit(
    monkeypatch, tmp_path
):
    data, freeze_path, gate_path, out = generated_gate(monkeypatch, tmp_path)
    original = e168.atomic_npz
    attempted = False

    def fail_first_checkpoint(path, arrays):
        nonlocal attempted
        if path.name == "pre-solve-raw.npz" and not attempted:
            attempted = True
            raise OSError("generated unconverted checkpoint write failure")
        return original(path, arrays)

    def forbidden_prepare(*_args):
        raise AssertionError("E164 prepare must not run")

    monkeypatch.setattr(e168, "atomic_npz", fail_first_checkpoint)
    monkeypatch.setattr(e168.e164, "prepare_saved", forbidden_prepare)
    FakeSolver.instances = FakeSolver.solves = 0
    with pytest.raises(OSError, match="unconverted checkpoint"):
        e168.run(
            freeze_path,
            "generated",
            gate_path,
            "generated",
            out,
            input_loader=lambda _: data,
            solver_factory=FakeSolver,
        )
    assert attempted
    receipt = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert receipt["validity"] == "incomplete"
    assert (
        receipt["counts"]["fit_calls"] == FakeSolver.instances == FakeSolver.solves == 0
    )


def test_cli_from_unrelated_cwd_without_pythonpath(tmp_path):
    command = [sys.executable, "-X", "utf8", str(e168.ROOT / e168.OWNED[0]), "--help"]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0
    assert (
        "--freeze-sha256" in completed.stdout
        and "--source-review-sha256" in completed.stdout
    )
    invalid = subprocess.run(
        command[:-1]
        + [
            "--freeze",
            str(tmp_path / "missing.json"),
            "--freeze-sha256",
            "wrong",
            "--source-review",
            str(tmp_path / "missing-gate.json"),
            "--source-review-sha256",
            "wrong",
            "--out",
            str(tmp_path / "out"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        invalid.returncode == 1 and "Exact E168 freeze bytes required" in invalid.stderr
    )
    assert not (tmp_path / "out").exists()
