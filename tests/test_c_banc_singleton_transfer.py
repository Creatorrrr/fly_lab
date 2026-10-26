"""Generated E191 contracts; no original NPZ payload is read here."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import fit_c_banc_m3_singleton_transfer as e191


def generated(rate_scale=1.):
    counts = np.full(96, 4)
    ltm = np.arange(6)
    singleton = np.arange(6, 19)
    counts[ltm] = 8
    counts[singleton] = 1
    counts[19:41] = 5
    assert counts.sum() == 391
    membership = np.repeat(np.arange(96), counts)
    # Graph columns have intentionally noncontiguous muscle ownership.
    membership = membership[np.argsort((np.arange(391) * 97) % 391)]
    pool = np.zeros((96, 391))
    pool[membership, np.arange(391)] = 1 / counts[membership]
    slopes = np.zeros((96, 391, 3, 2))
    slopes[:, :, 1] = pool[:, :, None]
    for motor in np.flatnonzero(~np.isin(membership, np.r_[ltm, singleton])):
        slopes[membership[motor], motor] = .06
    tick = np.arange(2001)[:, None]
    motor = np.arange(391)[None, :]
    rates = rate_scale * (.4 + .1 * np.sin(.015 * tick + .07 * motor))
    bank = np.zeros((2001, 391, 3))
    alpha = -np.expm1(-.005 / np.array([.005, .01, .04]))
    for row in range(1, 2001):
        bank[row] = bank[row - 1] + alpha[None, :] * (rates[row, :, None] - bank[row - 1])
    original_activation = bank[:, :, 1] @ pool.T
    data = {
        "loaded_recruitment_bank": bank,
        "loaded_recruitment_original_activation": original_activation,
        "loaded_adapter_membership": membership,
        "loaded_adapter_pool": pool,
        "loaded_adapter_ltm": ltm,
        "loaded_parameters_slopes": slopes,
        "loaded_parameters_kappa": np.full(391, .45),
    }
    data["loaded_recruitment_activation"] = np.zeros((2001, 96))
    data["loaded_recruitment_activation"] = e191.native_activation(data, slopes, np.r_[ltm, singleton])
    moment = np.zeros((42, 96))
    group_legs = np.repeat(np.arange(6), [17, 15, 16, 17, 15, 16])
    for muscle in range(96):
        moment[7 * group_legs[muscle]:7 * (group_legs[muscle] + 1), muscle] = .01
    data["loaded_adapter_moment"] = moment
    data["loaded_adapter_rest"] = np.zeros(42)
    command = e191.native_command(data, data["loaded_recruitment_activation"])
    tendon = .2 * np.tanh(data["loaded_recruitment_activation"][:, ltm] / 10)
    data.update(
        loaded_adapter_motor_ids=1000 + 3 * np.arange(391, dtype=np.int64),
        loaded_adapter_moment=moment,
        loaded_adapter_rest=np.zeros(42),
        loaded_parameters_membership=membership.copy(),
        loaded_parameters_motor_ids=np.arange(391, dtype=np.int64),
        loaded_parameters_tau=np.array([.005, .01, .04]),
        loaded_state_parameters_fixed_muscles=np.r_[ltm, singleton],
        loaded_state_group_columns_json=np.asarray(json.dumps([np.flatnonzero(membership == i).tolist() for i in range(96)])),
        loaded_trace_recorded_command=command,
        loaded_trace_tendon=tendon,
        loaded_trace_control_tick=np.arange(2001, dtype=np.int64),
        loaded_trace_time_s=np.arange(2001) * .005,
        loaded_trace_motor_rate=rates,
        loaded_startup_target_command=command[1:100] + .005,
        loaded_startup_target_tendon=tendon[1:100] + .001,
        loaded_startup_recorded_command=command.copy(),
        loaded_startup_recorded_tendon=tendon[1:].copy(),
        target=command[100:1924] + .01,
        target_velocity=(command[101:1924] - command[100:1923]) / .005 + .03,
        target_indices=np.arange(100, 1924, dtype=np.int64),
        affine_rows_command=np.arange(100, 1924, dtype=np.int64),
        affine_rows_velocity_start=np.arange(100, 1923, dtype=np.int64),
        affine_rows_velocity_end=np.arange(101, 1924, dtype=np.int64),
        affine_rows_startup=np.arange(1, 100, dtype=np.int64),
        loaded_recruitment_motor_rate=rates,
        loaded_recruitment_control_tick=np.arange(2001, dtype=np.int64),
        loaded_recruitment_network_tick=np.arange(2001, dtype=np.int64) * 20,
        loaded_recruitment_physics_time_s=np.arange(2001) * .005,
    )
    for field in ("complete", "bank_available", "activation_available", "original_activation_available", "motor_rate_available"):
        data["loaded_recruitment_" + field] = np.ones(2001, dtype=bool)
    for field in ("membership", "motor_ids", "moment", "rest", "pool", "ltm"):
        if field not in ("moment", "rest", "pool", "ltm"):
            data["loaded_parameters_" + field] = data["loaded_adapter_" + field].copy()
        data["loaded_calibration_" + field] = data["loaded_adapter_" + field].copy()
    for field in ("membership", "slopes", "kappa", "tau"):
        data["loaded_state_parameters_" + field] = data["loaded_parameters_" + field].copy()
    data["loaded_target_mask_target_indices"] = data["target_indices"].copy()
    data["loaded_target_mask_pair_start"] = data["affine_rows_velocity_start"].copy()
    data["loaded_target_mask_pair_end"] = data["affine_rows_velocity_end"].copy()
    data["loaded_target_mask_control_time_s"] = np.arange(2001) * .005
    data["loaded_target_mask_mask"] = np.isin(np.arange(2001), data["target_indices"])
    data["loaded_phase_diagnostic_control_time_s"] = np.arange(2001) * .005
    data["loaded_phase_diagnostic_e1_phase"] = np.zeros((2001, 6))
    data["loaded_phase_diagnostic_e1_phase_available"] = np.ones((2001, 6), dtype=bool)
    data["loaded_state_initial_bank"] = bank[0].copy()
    data["loaded_state_initial_activation"] = data["loaded_recruitment_activation"][0].copy()
    return data


def test_singletons_join_variable_family_and_preserve_ltm():
    data = generated()
    layout = e191.validate_input(data)
    assert layout["variable_motors"].size == 343
    assert layout["pair_motors"].size == 1029
    assert layout["w0"].size == 2058
    np.testing.assert_array_equal(layout["singleton_muscles"], np.arange(6, 19))
    assert np.any(np.diff(layout["pair_muscles"]) < 0)
    baseline_activation, affine = e191.affine_candidate(data, layout, layout["w0"])
    np.testing.assert_array_equal(baseline_activation, data["loaded_recruitment_activation"])
    np.testing.assert_array_equal(affine, data["loaded_trace_recorded_command"])
    w = layout["w0"].copy()
    singleton_pair = np.flatnonzero(layout["pair_muscles"] == 6)[0]
    w[2 * singleton_pair] += .01
    native = e191.candidate(data, layout, w)
    reference_command = np.asarray([data["loaded_adapter_rest"] + .02 * (data["loaded_adapter_moment"] @ row) for row in native["activation"]])
    reference_command[0] = data["loaded_trace_recorded_command"][0]
    np.testing.assert_array_equal(native["command"], reference_command)
    assert np.any(native["activation"][:, 6] != data["loaded_recruitment_activation"][:, 6])
    np.testing.assert_array_equal(native["activation"][:, data["loaded_adapter_ltm"]], data["loaded_recruitment_activation"][:, data["loaded_adapter_ltm"]])
    assert all(e191.physical_gates(data, layout, w).values())


def test_rejects_corrupt_alias_and_ownership():
    data = generated()
    data["loaded_state_parameters_membership"][0] = 95
    with pytest.raises(ValueError, match="alias"):
        e191.validate_input(data)
    data = generated()
    data["loaded_state_group_columns_json"] = np.asarray(json.dumps([[]] * 96))
    with pytest.raises(ValueError, match="ownership"):
        e191.validate_input(data)


def test_inner_problem_witness_and_proposal_screen():
    data = generated()
    layout = e191.validate_input(data)
    baseline, _ = e191.metrics(data, data["loaded_trace_recorded_command"])
    raw = {}
    _, q, matrix, b, cones, witness = e191.build_problem(data, layout, baseline, raw)
    assert matrix.shape[1] == 4117
    assert len(cones) == 16
    assert cones[0].dim == 21189
    assert witness[-1] == -.251
    assert np.min(b[:21189] - matrix[:21189] @ witness) >= -1e-12
    assert np.max(np.abs(q[:-1])) == 0 and q[-1] == -1
    assert len([key for key in raw if key.startswith("qr_") and not key.endswith(("_active", "_offset", "_direct_norms", "_compressed_norms"))]) == 18
    changed = witness.copy()
    changed[0] = -1e-5
    assert np.min(b[:21189] - matrix[:21189] @ changed) < 0
    changed = witness.copy()
    changed[2058] += .01
    assert np.min(b[:21189] - matrix[:21189] @ changed) < 0
    native = e191.candidate(data, layout, witness[:2058])
    actual, _ = e191.metrics(data, native["command"])
    gates = e191.proposal_gates(data, layout, witness, native, actual, baseline)
    assert not gates["overall_command_20pct"]
    assert gates["startup_q"]


def test_authority_rejects_hash_and_unsafe_input(tmp_path, monkeypatch):
    monkeypatch.setattr(e191, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    design = tmp_path / e191.DESIGN
    design.write_text("design", encoding="utf-8")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tests").mkdir()
    for name in e191.OWNED:
        (tmp_path / name).write_text("source", encoding="utf-8")
    source_files = {name: e191.sha256(tmp_path / name) for name in e191.OWNED}
    input_path = tmp_path / "input.npz"
    np.savez(input_path, x=np.array([1]))
    freeze = {"schema": e191.SCHEMA, "input_npz": {"path": "input.npz", "sha256": e191.sha256(input_path)}, "design": {"path": e191.DESIGN, "sha256": e191.sha256(design)}, "source_files": source_files, "cumulative_before_s": 4655.7939677997165, "stage_cap_s": 4994, "max_supervisor_s": 95, "max_internal_s": 80, "max_output_bytes": 268435456}
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    review = {"approved": True, "freeze_sha256": e191.sha256(freeze_path), "source_files": source_files}
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    e191.validate_authority(freeze_path, review["freeze_sha256"], review_path, e191.sha256(review_path))
    with pytest.raises(ValueError, match="SHA256"):
        e191.validate_authority(freeze_path, "0" * 64, review_path, e191.sha256(review_path))
    freeze["input_npz"]["path"] = "../escape.npz"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        e191.validate_authority(freeze_path, e191.sha256(freeze_path), review_path, e191.sha256(review_path))


def test_create_only_output_rejected_before_input_read(tmp_path, monkeypatch):
    monkeypatch.setattr(e191, "validate_authority", lambda *_: ({"input_npz": {"path": "missing.npz"}}, {}))
    monkeypatch.setattr(e191, "ROOT", tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="new workspace"):
        e191.run("freeze", "0" * 64, "review", "0" * 64, out)


def full_run_args(tmp_path, monkeypatch):
    monkeypatch.setattr(e191, "ROOT", tmp_path)
    for name in (*e191.OWNED, e191.DESIGN):
        source = Path(__file__).resolve().parents[1] / name
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    input_path = tmp_path / "synthetic.npz"
    # Zero input makes the full-sized real SOCP's optimum known: t = -0.25.
    # Varying-input ownership, QR and native transfer are tested separately.
    np.savez_compressed(input_path, **generated(rate_scale=0.))
    sources = {name: e191.sha256(tmp_path / name) for name in e191.OWNED}
    freeze = {"schema": e191.SCHEMA, "input_npz": {"path": "synthetic.npz", "sha256": e191.sha256(input_path)}, "design": {"path": e191.DESIGN, "sha256": e191.sha256(tmp_path / e191.DESIGN)}, "source_files": sources, "cumulative_before_s": 4655.7939677997165, "stage_cap_s": 4994, "max_supervisor_s": 95, "max_internal_s": 80, "max_output_bytes": 268435456}
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps({"approved": True, "freeze_sha256": e191.sha256(freeze_path), "source_files": sources}), encoding="utf-8")
    return freeze_path, e191.sha256(freeze_path), review_path, e191.sha256(review_path), tmp_path / "out"


def test_generated_full_run_and_immutable_manifest(tmp_path, monkeypatch):
    args = full_run_args(tmp_path, monkeypatch)
    result = e191.run(*args)
    assert result["validity"] == "valid"
    assert result["hypothesis_verdict"] == "valid_rejection_inconclusive"
    assert result["margin"] == pytest.approx(-.25, abs=1e-7)
    out = args[-1]
    ledger = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["fit_calls"] == ledger["solver_returns"] == 1
    assert ledger["fit_wall_s"] <= 20
    manifest = json.loads((out / "output-manifest.json").read_text(encoding="utf-8"))["files"]
    assert "ledger.json" not in manifest and "result.json" not in manifest
    assert all(e191.sha256(out / name) == receipt["sha256"] for name, receipt in manifest.items())
    with np.load(out / "pre-solve-raw.npz", allow_pickle=False) as raw:
        assert raw["problem_A_shape"][1] == 4117
        assert raw["cone_dims"][0] == 21189
    with np.load(out / "candidate-raw.npz", allow_pickle=False) as raw:
        assert raw["native_activation"].shape == (2001, 96)


def test_late_result_write_failure_revokes_success(tmp_path, monkeypatch):
    args = full_run_args(tmp_path, monkeypatch)
    original = e191.atom_json

    def fail_success_result(path, value):
        if path.name == "result.json" and value.get("validity") == "valid":
            raise OSError("synthetic final publication failure")
        original(path, value)

    monkeypatch.setattr(e191, "atom_json", fail_success_result)
    with pytest.raises(OSError, match="final publication"):
        e191.run(*args)
    receipt = json.loads((args[-1] / "result.json").read_text(encoding="utf-8"))
    assert receipt["validity"] == "incomplete"
    assert receipt["hypothesis_verdict"] == "invalid_not_tested"
    assert receipt["conditional_proposal"] is False
    assert (args[-1] / "candidate-raw.npz").exists()


def test_inclusive_fit_limit_preserves_candidate_and_rejects(tmp_path, monkeypatch):
    args = full_run_args(tmp_path, monkeypatch)
    real_solver = e191.clarabel.DefaultSolver
    real_clock = e191.time.perf_counter
    delayed = [0.]

    def solver_with_slow_construction(*solver_args):
        # Simulated constructor cost must count even when the real solve is fast.
        solver = real_solver(*solver_args)
        delayed[0] += 20.01
        return solver

    monkeypatch.setattr(e191, "time", SimpleNamespace(perf_counter=lambda: real_clock() + delayed[0]))
    monkeypatch.setattr(e191.clarabel, "DefaultSolver", solver_with_slow_construction)
    with pytest.raises(ValueError, match="inclusive 20-second"):
        e191.run(*args)
    receipt = json.loads((args[-1] / "result.json").read_text(encoding="utf-8"))
    ledger = json.loads((args[-1] / "ledger.json").read_text(encoding="utf-8"))
    assert receipt["validity"] == "incomplete"
    assert receipt["hypothesis_verdict"] == "invalid_not_tested"
    assert receipt["conditional_proposal"] is False
    assert ledger["fit_wall_s"] > 20
    assert (args[-1] / "candidate-raw.npz").exists()
    assert (args[-1] / "solver-return-raw.npz").exists()
    assert not (args[-1] / "output-manifest.json").exists()
