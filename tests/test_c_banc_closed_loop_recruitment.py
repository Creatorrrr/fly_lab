"""Generated-only checks for the frozen E169 positive-branch fit."""

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tests import test_c_banc_onset_recruitment as fixture
from tools import fit_c_banc_m3_closed_loop_recruitment as e169


def generated(*, varying_bank=False):
    data = fixture.generated_full_inputs()
    if varying_bank:
        ticks = np.arange(2001)[:, None]
        motors = np.arange(391)[None, :]
        rates = 1 + 0.2 * np.sin(0.013 * ticks + 0.071 * motors)
        bank = np.ones((2001, 391, 3))
        alpha = -np.expm1(-0.005 / np.array([0.005, 0.01, 0.04]))
        for tick in range(1, 2001):
            bank[tick] = bank[tick - 1] + alpha[None, :] * (
                rates[tick, :, None] - bank[tick - 1]
            )
        data["recruitment"]["motor_rate"] = rates
        data["recruitment"]["bank"] = bank
        for row, rate in zip(data["trace"]["samples"], rates, strict=True):
            row["motor_rates"] = rate.tolist()
    # A physical baseline with all 77 muscles above their direct 0.5 floor.
    membership = data["adapter"]["membership"]
    for motor in np.flatnonzero(np.isin(membership, fixture.VARIABLE)):
        data["parameters"]["slopes"][membership[motor], motor] = 0.06
    data["recruitment"]["activation"] = e169.e164.reconstruct_activation(
        data["recruitment"]["bank"],
        data["recruitment"]["original_activation"],
        data["parameters"]["slopes"],
        data["parameters"]["kappa"],
        membership,
        fixture.FIXED,
    )
    data["state"]["adapter"]["activation"] = data["recruitment"]["activation"][
        0
    ].tolist()
    command = data["adapter"]["rest"] + 0.02 * (
        data["recruitment"]["activation"] @ data["adapter"]["moment"].T
    )
    for row, q in zip(data["trace"]["samples"], command, strict=True):
        row["joint_targets_rad"] = q.tolist()
    data["startup"]["recorded_command"] = command
    data["startup"]["target_command"] = command[1:100] + 0.005
    base = fixture.generated_full_freeze(data)
    freeze = json.loads(
        (
            e169.ROOT
            / "verification/banc-natural-gait-m3-20260919/e169-closed-loop-recruitment-freeze.json"
        ).read_text(encoding="utf-8")
    )
    freeze["sample_contract"] = {**freeze["sample_contract"], **base["sample_contract"]}
    freeze["body_hash"] = base["body_hash"]
    freeze["resource_limits"] = {
        **freeze["resource_limits"],
        "free_bytes": 0,
        "internal_s": 120,
        "raw_bytes": 201326592,
    }
    return data, base, freeze


def saved_problem():
    data, base, freeze = generated()
    raw = {}
    e169.capture_loaded_input(data, raw)
    e169.validate_loaded_input(data, raw)
    checks = []
    saved = e169.e164.prepare_saved(data, base, raw, checks)
    e169.prepare_transfer(data, saved, freeze, raw, checks)
    e169.enforce(checks)
    problem_checks = []
    problem = e169.build_problem(saved, freeze, raw, problem_checks)
    e169.enforce(problem_checks)
    return data, base, freeze, saved, problem, raw


def test_generated_mapping_baseline_native_and_physical_envelope():
    data, base, freeze = generated()
    raw = {}
    saved = e169.e164.prepare_saved(data, base, raw, [])
    e169.prepare_transfer(data, saved, freeze, raw, [])
    assert raw["pair_motors"].shape == (990,)
    assert raw["w0"].shape == (1980,)
    np.testing.assert_array_equal(
        raw["pair_motors"], np.repeat(freeze["sample_contract"]["variable_motors"], 3)
    )
    np.testing.assert_array_equal(raw["pair_filters"], np.tile(np.arange(3), 330))
    metrics, arrays = e169.candidate(saved, saved["w0"])
    np.testing.assert_array_equal(arrays["candidate_activation"], saved["activation"])
    np.testing.assert_array_equal(
        arrays["candidate_command"], saved["recorded_command"]
    )
    assert metrics == saved["baseline_metrics"]
    np.testing.assert_allclose(
        e169.native_candidate(saved, saved["w0"])[1:],
        saved["activation"][1:],
        atol=1e-12,
        rtol=0,
    )
    w = saved["w0"].copy()
    w[0] += 0.01
    changed, output = e169.candidate(saved, w)
    assert np.any(output["candidate_delta_activation"])
    np.testing.assert_array_equal(
        output["candidate_activation"][:, fixture.FIXED],
        saved["activation"][:, fixture.FIXED],
    )
    np.testing.assert_allclose(
        e169.native_candidate(saved, w)[1:],
        output["candidate_activation"][1:],
        atol=1e-12,
        rtol=0,
    )
    assert changed["overall_command_RMS"] >= 0
    assert all(e169.physical_gates(saved, w, freeze).values())
    pair = w.reshape(990, 2)
    witness = np.r_[w, pair.min(axis=1), pair.max(axis=1), -0.251]
    assert witness.shape == (3961,)
    assert not e169.physical_gates(saved, np.r_[np.full(1980, 0.01)], freeze)[
        "muscle_sum_min_budget"
    ]


def test_generated_full_18_qr_15_soc_and_relaxed_dual():
    _, _, freeze, saved, problem, raw = saved_problem()
    assert problem.A.shape[1] == 3961
    assert problem.cone_dims[0] == 20682
    assert len(problem.criteria) == 15
    assert (
        len(
            [
                key
                for key in raw
                if key.startswith("qr_")
                and not key.endswith(("_active", "_direct_norms", "_compressed_norms"))
            ]
        )
        == 18
    )
    assert raw["baseline_witness"].shape == (3961,)
    assert np.min(raw["baseline_witness_slack"][:20682]) >= -1e-12
    assert all(e169.physical_gates(saved, raw["w0"], freeze).values())
    altered_envelope = raw["baseline_witness"].copy()
    altered_envelope[1980] += 0.01
    assert np.min((problem.b - problem.A @ altered_envelope)[:11882]) < 0
    altered_envelope = raw["baseline_witness"].copy()
    altered_envelope[2970] -= 0.01
    assert np.min((problem.b - problem.A @ altered_envelope)[:11882]) < 0
    fake = SimpleNamespace(
        x=raw["baseline_witness"],
        s=raw["baseline_witness_slack"],
        z=np.zeros(problem.A.shape[0]),
        status="Solved",
        iterations=0,
        solve_time=0,
    )
    derived = e169.solver_derived(problem, fake, [], freeze)
    changed = np.flatnonzero(derived["physical_budget_relaxed_b"] != problem.b)
    np.testing.assert_array_equal(changed, np.arange(11882, 12366))
    assert np.isfinite(derived["relaxed_projected_margin_upper_bound"])


def test_uncoerced_bad_input_is_captured_before_failure():
    data, _, _ = generated()
    data["recruitment"]["bank"] = data["recruitment"]["bank"].astype(np.float32)
    raw = {}
    e169.capture_loaded_input(data, raw)
    assert str(raw["loaded_recruitment_bank"].dtype) == "float32"
    with pytest.raises(ValueError, match="dtype"):
        e169.validate_loaded_input(data, raw)


def test_gate_rejects_wrong_source_even_when_approved(tmp_path, monkeypatch):
    freeze = {"design": "ignored"}
    gate = {
        "schema": e169.GATE_SCHEMA,
        "approved": True,
        "blocking_defects": [],
        "accepts_single_closed_loop_recruitment_fit": True,
        "original_processes_allowed": 1,
        "automatic_retries": 0,
        "freeze_sha256": e169.FREEZE_SHA,
        "design_sha256": e169.DESIGN_SHA,
        "source_sha256": {name: "wrong" for name in e169.OWNED},
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="source review provenance"):
        e169.validate_gate(path, e169.sha256(path), freeze)


def setup_generated_run(tmp_path, monkeypatch, *, varying_bank=True):
    data, base, freeze = generated(varying_bank=varying_bank)
    freeze_path = (
        e169.ROOT
        / "verification/banc-natural-gait-m3-20260919/e169-closed-loop-recruitment-freeze.json"
    )
    gate = {
        "schema": e169.GATE_SCHEMA,
        "approved": True,
        "blocking_defects": [],
        "accepts_single_closed_loop_recruitment_fit": True,
        "original_processes_allowed": 1,
        "automatic_retries": 0,
        "freeze_sha256": e169.FREEZE_SHA,
        "design_sha256": e169.DESIGN_SHA,
        "source_sha256": {
            name: e169.sha256(e169.frozen_path(name)) for name in e169.OWNED
        },
    }
    gate_path = tmp_path / "review.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    monkeypatch.setattr(
        e169, "validate_freeze", lambda path, digest: (freeze, base, {})
    )
    out = tmp_path / "out"
    args = (freeze_path, e169.FREEZE_SHA, gate_path, e169.sha256(gate_path), out)
    return args, out, freeze, lambda _: data


def test_generated_full_run_one_clarabel_solve_and_receipts(tmp_path, monkeypatch):
    args, out, freeze, loader = setup_generated_run(tmp_path, monkeypatch)
    result = e169.run(*args, input_loader=loader)
    assert result["validity"] == "valid"
    assert result["counts"]["fit_calls"] == result["counts"]["solver_returns"] == 1
    assert (out / "solver-return-raw.npz").exists()
    assert (out / "candidate-raw.npz").exists()
    assert (out / "output-manifest.json").exists()
    with np.load(out / "pre-solve-raw.npz", allow_pickle=False) as raw:
        assert raw["problem_A_shape"][1] == 3961
        assert raw["cone_dims"][0] == 20682
    with np.load(out / "solver-derived-raw.npz", allow_pickle=False) as raw:
        assert np.isfinite(raw["relaxed_projected_margin_upper_bound"])
    assert (
        sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
        <= freeze["resource_limits"]["raw_bytes"]
    )


def test_generated_solver_return_survives_derived_failure(tmp_path, monkeypatch):
    args, out, _, loader = setup_generated_run(tmp_path, monkeypatch)

    def fail_derived(*unused):
        raise RuntimeError("generated post-return failure")

    monkeypatch.setattr(e169, "solver_derived", fail_derived)
    with pytest.raises(RuntimeError, match="post-return failure"):
        e169.run(*args, input_loader=loader)
    with np.load(out / "solver-return-raw.npz", allow_pickle=False) as returned:
        assert returned["solver_x"].shape == (3961,)
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["validity"] == "incomplete"
    assert result["hypothesis_verdict"] == "invalid_not_tested"
    assert result["counts"]["solver_returns"] == 1
    assert result["primary_error"]["type"] == "RuntimeError"


def test_generated_late_publication_revokes_success(tmp_path, monkeypatch):
    args, out, _, loader = setup_generated_run(tmp_path, monkeypatch)
    original_manifest = e169.output_manifest
    calls = 0

    def late_manifest(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("generated late publication failure")
        return original_manifest(path)

    monkeypatch.setattr(e169, "output_manifest", late_manifest)
    with pytest.raises(OSError, match="late publication failure"):
        e169.run(*args, input_loader=loader)
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["validity"] == "incomplete"
    assert result["conditional_closed_loop_recruitment_proposal"] is False
    assert result["hypothesis_verdict"] == "invalid_not_tested"
    assert result["primary_error"]["type"] == "OSError"
    assert (out / "candidate-raw.npz").exists()


def test_generated_terminal_time_gate_revokes_success(tmp_path, monkeypatch):
    args, out, _, loader = setup_generated_run(tmp_path, monkeypatch)
    real_clock = time.perf_counter

    def terminal_clock():
        return real_clock() + (200 if (out / "output-manifest.json").exists() else 0)

    monkeypatch.setattr(e169, "time", SimpleNamespace(perf_counter=terminal_clock))
    with pytest.raises(TimeoutError, match="after terminal publication"):
        e169.run(*args, input_loader=loader)
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["validity"] == "incomplete"
    assert result["conditional_closed_loop_recruitment_proposal"] is False
    assert result["primary_error"]["type"] == "TimeoutError"
    assert any("invalid-stale" in path.name for path in out.iterdir())


def test_generated_terminal_byte_gate_revokes_success(tmp_path, monkeypatch):
    args, out, freeze, loader = setup_generated_run(tmp_path, monkeypatch)
    freeze["resource_limits"]["raw_bytes"] = 1
    with pytest.raises(ValueError, match="output byte limit"):
        e169.run(*args, input_loader=loader)
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["validity"] == "incomplete"
    assert result["conditional_closed_loop_recruitment_proposal"] is False
    assert result["primary_error"]["type"] == "ValueError"
    assert (out / "solver-return-raw.npz").exists()


class FakeSolver:
    def __init__(self, P, q, A, b, cones, settings):
        self.A, self.b, self.settings = A, b, settings

    def set_termination_callback(self, callback):
        self.callback = callback

    def get_settings(self):
        return self.settings

    def solve(self):
        x = np.r_[np.full(3960, 0.06), -0.251]
        return SimpleNamespace(
            x=x,
            s=self.b - self.A @ x,
            z=np.zeros(len(self.b)),
            status="GeneratedFakeSolve",
            iterations=0,
            solve_time=0.0,
            obj_val=0.251,
            obj_val_dual=0.0,
            r_prim=0.0,
            r_dual=0.0,
        )


@pytest.mark.parametrize(
    "target,expected",
    [
        ("solver-return-raw.npz", "solver_x"),
        ("candidate-raw.npz", "candidate_activation"),
    ],
)
@pytest.mark.parametrize("fallback_fails", [False, True])
def test_generated_fake_solver_raw_save_fallback(
    tmp_path, monkeypatch, target, expected, fallback_fails
):
    args, out, _, loader = setup_generated_run(
        tmp_path, monkeypatch, varying_bank=False
    )
    original = e169.atomic_npz
    fallback = target.replace("-raw.npz", "-fallback.npz")
    attempts = []

    def fail_raw(path, arrays):
        name = path.name
        attempts.append(name)
        if name == target and attempts.count(target) == 1:
            raise OSError("generated first raw save failure: " + name)
        if fallback_fails and name == fallback:
            raise OSError("generated secondary fallback failure: " + name)
        return original(path, arrays)

    monkeypatch.setattr(e169, "atomic_npz", fail_raw)
    with pytest.raises(OSError, match="first raw save failure"):
        e169.run(*args, input_loader=loader, solver_factory=FakeSolver)
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    ledger = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    assert result["validity"] == "incomplete"
    assert result["hypothesis_verdict"] == "invalid_not_tested"
    assert result["primary_error"]["message"].endswith(target)
    assert result["counts"]["fit_calls"] == result["counts"]["solver_returns"] == 1
    assert attempts.count(target) == 1
    assert attempts.count(fallback) == 1
    if fallback_fails:
        assert not (out / fallback).exists()
        assert any(
            item["stage"] == fallback.removesuffix("-fallback.npz") + " fallback raw"
            for item in ledger["secondary_errors"]
        )
    else:
        with np.load(out / fallback, allow_pickle=False) as raw:
            assert expected in raw
