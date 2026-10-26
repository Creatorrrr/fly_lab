import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import clarabel
import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "fit_c_banc_m3_monotone_recruitment",
    TOOLS / "fit_c_banc_m3_monotone_recruitment.py",
)
assert SPEC is not None and SPEC.loader is not None
fit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fit
SPEC.loader.exec_module(fit)


def small_pool():
    membership = np.array([0, 0, 0, 1, 1, 2])
    pool = np.zeros((3, 6))
    pool[0, :3] = 1 / 3
    pool[1, 3:5] = 1 / 2
    pool[2, 5] = 1
    return pool, membership, np.array([1])


def test_knot_rule_basis_and_monotone_continuous_transfer():
    bank = np.zeros((5, 3, 3))
    bank[:, 0, 1] = [0, 2, 8, 4, 6]
    bank[:, 1, 1] = [0, 0, 0, 0, 0]
    bank[:, 2, 1] = [0.1, 0.2, 0.3, 0.4, 0.5]
    bank[:, :, 0] = np.array([0, 1, 3])
    bank[:, :, 2] = np.array([9, 2, 0.25])
    kappa, basis = fit.knot_and_basis(bank)
    np.testing.assert_array_equal(kappa, [5, 1, 1])
    assert np.all(basis >= 0)
    np.testing.assert_array_equal(basis.sum(axis=-1), bank)

    values = np.array([4.999999, 5.0, 5.000001])
    response = fit.piecewise_response(values, 5.0, np.array([0.4, 1.2]))
    assert np.all(np.diff(response) >= 0)
    assert response[1] == pytest.approx(2.0)
    left = fit.piecewise_response(5.0 - 1e-8, 5.0, np.array([0.4, 1.2]))
    right = fit.piecewise_response(5.0 + 1e-8, 5.0, np.array([0.4, 1.2]))
    assert abs(right - left) < 2e-8


def test_map_layout_witnesses_and_old_linear_inclusion():
    pool, membership, ltm = small_pool()
    old_weights = fit.gain.canonical_weights(pool) * 1.999999
    mapping = fit.build_recruitment_map(pool, membership, ltm, old_weights)
    np.testing.assert_array_equal(mapping.fitted_muscle_ids, [0])
    np.testing.assert_array_equal(mapping.fixed_muscles, [1, 2])
    np.testing.assert_array_equal(mapping.pair_flat, np.arange(9))
    np.testing.assert_array_equal(mapping.pair_motors, np.repeat(np.arange(3), 3))
    np.testing.assert_array_equal(mapping.pair_branches, np.tile(np.arange(3), 3))
    np.testing.assert_array_equal(mapping.coefficient_low_indices, 2 * np.arange(9))
    np.testing.assert_array_equal(
        mapping.coefficient_high_indices, 2 * np.arange(9) + 1
    )
    assert mapping.coefficient_slice == slice(0, 18)
    assert mapping.lower_slice == slice(18, 27)
    assert mapping.upper_slice == slice(27, 36)

    old_pair = old_weights[
        mapping.pair_muscles, mapping.pair_motors, mapping.pair_branches
    ]
    e147_slopes = mapping.e147_vector[mapping.coefficient_slice].reshape(-1, 2)
    np.testing.assert_array_equal(e147_slopes[:, 0], old_pair)
    np.testing.assert_array_equal(e147_slopes[:, 1], old_pair)
    lifted_slopes = mapping.lifted_vector[mapping.coefficient_slice].reshape(-1, 2)
    np.testing.assert_array_equal(lifted_slopes[:, 0], lifted_slopes[:, 1])
    np.testing.assert_allclose(
        mapping.gain_rows @ mapping.lifted_vector[mapping.lower_slice], 1
    )
    np.testing.assert_allclose(
        mapping.gain_rows @ mapping.lifted_vector[mapping.upper_slice], 1
    )

    rng = np.random.default_rng(14800)
    bank = rng.uniform(1e7, 1e8, size=(11, membership.size, 3))
    kappa, basis = fit.knot_and_basis(bank)
    saved = {
        "b": bank[:, :, 1],
        "bank": bank,
        "basis": basis,
        "kappa": kappa,
        "pool": pool,
        "membership": membership,
        "moment": rng.normal(size=(42, 3)),
        "rest": rng.normal(size=42),
        "target_indices": np.arange(1, 12),
        "pair_end": np.arange(2, 12),
        "ltm": ltm,
    }
    transfer_weights = fit.weights_from_vector(pool, mapping, mapping.e147_vector)
    transfer = fit.activation_from_weights(saved, transfer_weights, mapping)
    direct = fit.causal.activation_from_weights(
        bank, old_weights, bank[:, :, 1] @ pool.T, mapping.fitted_muscle_ids
    )
    np.testing.assert_array_equal(transfer, direct)
    direct_outputs = fit.activation_outputs(saved, direct)
    transfer_outputs = fit.activation_outputs(saved, transfer)
    for key in direct_outputs:
        np.testing.assert_array_equal(transfer_outputs[key], direct_outputs[key])
    canonical = bank[:, :, 1] @ pool.T
    # Muscle 1 is fixed by LTM and muscle 2 by singleton provenance.
    np.testing.assert_array_equal(
        transfer[:, mapping.fixed_muscles], canonical[:, [1, 2]]
    )


def test_nonnegative_rows_have_exact_order_and_reject_false_endpoint_budget():
    pool, membership, ltm = small_pool()
    mapping = fit.build_recruitment_map(
        pool, membership, ltm, fit.gain.canonical_weights(pool)
    )
    startup = sparse.csr_matrix(np.full((2, 18), 0.001))
    offset = np.array([0.002, -0.003])
    A, b = fit._make_nonnegative_rows(
        mapping, startup, offset, 0.04, (-1.0, 1.0), (0.500001, 1.999999)
    )
    point = np.concatenate((mapping.lifted_vector, [-0.251]))
    assert np.min(b - A @ point) >= -1e-12

    n = mapping.pair_flat.size
    # The first envelope row is ell_0 - coefficient_low_0 <= 0.
    envelope_start = 8 * n
    row = A.getrow(envelope_start).toarray()[0]
    assert row[mapping.lower_slice.start] == 1
    assert row[mapping.coefficient_low_indices[0]] == -1
    assert np.count_nonzero(row) == 2

    # Endpoint sums alone are insufficient: min-envelope catches crossed slopes.
    crossed = point.copy()
    slopes = crossed[mapping.coefficient_slice].reshape(-1, 2)
    slopes[:] = 0
    slopes[:5, 0] = 0.11
    slopes[5:, 1] = 0.14
    crossed[mapping.lower_slice] = np.min(slopes, axis=1)
    crossed[mapping.upper_slice] = np.max(slopes, axis=1)
    assert slopes[:, 0].sum() >= 0.5 and slopes[:, 1].sum() >= 0.5
    assert np.min(b - A @ crossed) < 0


def test_all_saved_output_inclusion_diagnostics_publish_before_failure():
    rng = np.random.default_rng(14802)
    pool, membership, ltm = small_pool()
    bank = rng.uniform(0.0, 20.0, size=(120, membership.size, 3))
    kappa, basis = fit.knot_and_basis(bank)
    old_weights = fit.gain.canonical_weights(pool)
    old_weights[0] *= 1.5
    mapping = fit.build_recruitment_map(pool, membership, ltm, old_weights)
    saved = {
        "bank": bank,
        "basis": basis,
        "kappa": kappa,
        "b": bank[:, :, 1],
        "pool": pool,
        "membership": membership,
        "moment": rng.normal(scale=0.1, size=(42, 3)),
        "rest": rng.normal(scale=0.1, size=42),
        "target_indices": np.arange(1, 121),
        "pair_end": np.arange(2, 121),
        "ltm": ltm,
        "witness147": old_weights,
    }
    direct = fit.causal.activation_from_weights(
        bank,
        old_weights,
        saved["b"] @ pool.T,
        mapping.fitted_muscle_ids,
    )
    outputs = fit.activation_outputs(saved, direct)
    saved["witness147_outputs"] = {
        key: outputs[key].copy()
        for key in (
            "activation",
            "selected_command_rad",
            "selected_velocity_rad_s",
            "startup_command_rad",
            "startup_tendon",
        )
    }
    saved["witness147_outputs"]["selected_velocity_rad_s"][20, 4] += 2e-9
    raw, partial = {}, {}
    with pytest.raises(ValueError, match="all saved E147 outputs"):
        fit.check_e147_linear_inclusion(saved, mapping, 1e-12, raw, partial)
    labels = (
        "direct_activation_vs_saved",
        "saved_selected_command_reconstruction",
        "saved_selected_velocity_reconstruction",
        "saved_startup_command_reconstruction",
        "saved_startup_tendon_reconstruction",
        "activation",
        "complete_command",
        "selected_command",
        "selected_velocity",
        "startup_command",
        "startup_tendon",
    )
    for label in labels:
        assert f"e147_inclusion_{label}_max_abs_error" in partial
        assert f"e147_inclusion_{label}_max_abs_index" in partial
    assert float(partial["e147_inclusion_selected_velocity_max_abs_error"]) > 1e-12


def generated_uneven_problem():
    rng = np.random.default_rng(14801)
    membership = np.repeat(np.arange(96), 2)
    pool = np.zeros((96, membership.size))
    pool[membership, np.arange(membership.size)] = 0.5
    ltm = np.array([0, 17, 32, 48, 65, 80])
    base_rate = rng.uniform(0.0, 2.0, size=(2000, membership.size))
    bank = fit.causal.causal_filter_bank(base_rate)
    kappa, basis = fit.knot_and_basis(bank)
    muscle_legs = np.repeat(np.arange(6), [17, 15, 16, 17, 15, 16])
    anatomy = {
        "muscles": [
            {
                "leg": fit.LEGS[int(muscle_legs[muscle])],
                "columns": np.flatnonzero(membership == muscle).tolist(),
            }
            for muscle in range(96)
        ]
    }
    moment = np.zeros((42, 96))
    for muscle, leg_index in enumerate(muscle_legs):
        moment[7 * leg_index : 7 * (leg_index + 1), muscle] = rng.normal(
            scale=0.2, size=7
        )
    rest = rng.normal(scale=0.1, size=42)
    baseline_activation = bank[:, :, 1] @ pool.T
    baseline_command = rest + fit.COMMAND_GAIN * (baseline_activation @ moment.T)
    target_indices = np.arange(174, 2001)
    pair_start = target_indices[:-1]
    pair_end = target_indices[1:]
    baseline_velocity = np.diff(baseline_command, axis=0)[pair_end - 2] / fit.DT
    saved = {
        "pool": pool,
        "membership": membership,
        "ltm": ltm,
        "bank": bank,
        "tau": np.array([0.005, 0.01, 0.04]),
        "b": bank[:, :, 1],
        "kappa": kappa,
        "basis": basis,
        "witness147": fit.gain.canonical_weights(pool),
        "moment": moment,
        "rest": rest,
        "baseline": baseline_command,
        "target_indices": target_indices,
        "pair_start": pair_start,
        "pair_end": pair_end,
        "target": baseline_command[target_indices - 1] + 0.2,
        "target_velocity": baseline_velocity + 0.2,
        "startup_target": baseline_command[:99].copy(),
        "startup_tendon_target": 0.2 * np.tanh(baseline_activation[:99, ltm] / 10),
    }
    witness_activation = fit.causal.activation_from_weights(
        bank,
        saved["witness147"],
        baseline_activation,
        np.setdiff1d(np.arange(96), ltm),
    )
    witness_outputs = fit.activation_outputs(saved, witness_activation)
    saved["witness147_outputs"] = {
        key: witness_outputs[key]
        for key in (
            "activation",
            "selected_command_rad",
            "selected_velocity_rad_s",
            "startup_command_rad",
            "startup_tendon",
        )
    }
    freeze = {
        "branch_guard_fraction": 1e-6,
        "proposal_command_RMS_ratio": 0.8,
        "proposal_velocity_RMS_ratio": 1.0,
        "startup_limits": {
            "q_RMS": 0.018,
            "q_max_abs": 0.04,
            "LTM_RMS": 0.009,
            "LTM_max_abs": 0.025,
        },
        "margin_bounds": [-1.0, 1.0],
        "solver_gain_bounds": [0.500001, 1.999999],
        "QR_absolute_atol": 1e-12,
        "QR_relative_atol": 1e-10,
        "baseline_feasibility_margin": -0.251,
        "original_MN_floor_fraction": 1e-6,
        "lower_bound_atol": 1e-12,
        "gain_bounds": [0.5, 2.0],
        "E147_inclusion_output_atol": 1e-12,
    }
    return saved, {"anatomy": anatomy}, freeze


def test_full_uneven_anatomy_15_soc_affines_match_direct_outputs():
    saved, data, freeze = generated_uneven_problem()
    problem, raw = fit.build_conic_problem(saved, data, freeze)
    assert len(problem.criteria) == 15
    assert problem.cone_kinds.count("SecondOrderConeT") == 15
    mapping = problem.anchor_map
    assert problem.startup_matrix.shape[1] == 4 * mapping.pair_flat.size
    weights = fit.weights_from_vector(saved["pool"], mapping, mapping.lifted_vector)
    arrays = fit.evaluate_composite(saved, weights)["arrays"]
    errors = {
        "overall_command": (
            arrays["candidate_selected_command_rad"] - saved["target"]
        ).reshape(-1),
        "overall_velocity": (
            arrays["candidate_selected_velocity_rad_s"] - saved["target_velocity"]
        ).reshape(-1),
        "overall_startup": (
            arrays["candidate_startup_command_rad"] - saved["startup_target"]
        ).reshape(-1),
    }
    for index, leg in enumerate(fit.LEGS):
        joint = slice(7 * index, 7 * (index + 1))
        errors[f"{leg}_command"] = (
            arrays["candidate_selected_command_rad"][:, joint]
            - saved["target"][:, joint]
        ).reshape(-1)
        errors[f"{leg}_velocity"] = (
            arrays["candidate_selected_velocity_rad_s"][:, joint]
            - saved["target_velocity"][:, joint]
        ).reshape(-1)
    for criterion in problem.criteria:
        affine = (
            criterion.direct_matrix @ mapping.lifted_vector + criterion.direct_offset
        )
        if criterion.name.startswith("overall_"):
            # Overall blocks are stacked leg-major; the direct output is time-major.
            assert np.linalg.norm(affine) == pytest.approx(
                np.linalg.norm(errors[criterion.name]), rel=3e-13, abs=3e-13
            )
        else:
            np.testing.assert_allclose(
                affine, errors[criterion.name], atol=3e-14, rtol=3e-13
            )
    np.testing.assert_array_equal(raw["pair_flat"], mapping.pair_flat)
    np.testing.assert_array_equal(
        raw["coefficient_low_indices"], mapping.coefficient_low_indices
    )
    np.testing.assert_array_equal(
        raw["coefficient_high_indices"], mapping.coefficient_high_indices
    )
    assert float(raw["e147_inclusion_activation_max_abs_error"]) <= 1e-12
    assert raw["lifted_baseline_cone_margins"].min() >= -1e-10


def test_fixed_outputs_are_exact_and_candidate_budgets_are_authoritative():
    saved, data, freeze = generated_uneven_problem()
    problem, _raw = fit.build_conic_problem(saved, data, freeze)
    mapping = problem.anchor_map
    vector = mapping.lifted_vector.copy()
    slopes = vector[mapping.coefficient_slice].reshape(-1, 2)
    slopes[:, 1] *= 1.1
    vector[mapping.lower_slice] = np.min(slopes, axis=1)
    vector[mapping.upper_slice] = np.max(slopes, axis=1)
    weights = fit.weights_from_vector(saved["pool"], mapping, vector)
    pair_lower, lower, upper = fit.physical_budgets(weights, mapping)
    np.testing.assert_array_equal(
        pair_lower,
        np.min(
            weights[mapping.pair_muscles, mapping.pair_motors, mapping.pair_branches],
            axis=1,
        ),
    )
    composite = fit.evaluate_composite(saved, weights)
    np.testing.assert_array_equal(
        composite["arrays"]["candidate_activation"][:, mapping.fixed_muscles],
        composite["arrays"]["baseline_activation"][:, mapping.fixed_muscles],
    )
    np.testing.assert_array_equal(
        composite["arrays"]["candidate_startup_tendon"],
        composite["arrays"]["baseline_startup_tendon"],
    )
    assessment = fit.assess_physical_candidate(
        saved, weights, lower, upper, composite, freeze
    )
    assert assessment["gates"]["authoritative_lower_budget_matches_weights"]
    assert assessment["gates"]["authoritative_upper_budget_matches_weights"]
    assert assessment["gates"]["original_motor_min_slope_floor"]


def test_derived_bank_knot_and_basis_survive_later_validation_failure():
    bank = np.zeros((2000, 391, 3))
    saved = {
        "b": bank[:, :, 1],
        "pool": np.zeros((96, 391)),
        "membership": np.arange(391) % 96,
        "moment": np.zeros((42, 96)),
        "rest": np.zeros(42),
        "ltm": np.array([0, 17, 32, 48, 65, 80]),
        "baseline": np.zeros((2000, 42)),
        "target_indices": np.arange(1827),
        "target": np.zeros((1827, 42)),
        "pair_start": np.arange(1826),
        "pair_end": np.arange(1, 1827),
        "target_velocity": np.zeros((1826, 42)),
        "startup_target": np.zeros((99, 42)),
        "startup_tendon_target": np.zeros((99, 6)),
    }
    key_map = {
        "baseline_pool": "pool",
        "membership": "membership",
        "moment": "moment",
        "rest": "rest",
        "ltm": "ltm",
        "baseline_endpoint_rad": "baseline",
        "target_indices": "target_indices",
        "target_rad": "target",
        "velocity_pair_start_indices": "pair_start",
        "velocity_pair_end_indices": "pair_end",
        "target_velocity_rad_s": "target_velocity",
        "startup_target_rad": "startup_target",
        "startup_target_tendon": "startup_tendon_target",
    }
    common = {raw_key: saved[saved_key] for raw_key, saved_key in key_map.items()}
    raw146 = {
        **common,
        "filter_bank": bank,
        "filter_tau_s": np.array([0.005, 0.01, 0.04]),
    }
    raw147 = {
        **raw146,
        "candidate_weights": np.empty((0,)),
    }
    partial, progress = {}, {}
    with pytest.raises(ValueError, match="E147 candidate weights"):
        fit.validate_saved_inputs(
            {"e146": raw146, "e147": raw147},
            {"tau_s": [0.005, 0.01, 0.04], "knot_minimum_hz": 1.0},
            base_validator=lambda *_: saved,
            partial=partial,
            progress=progress,
        )
    assert progress["reused_filter_vector_updates"] == 2000
    assert progress["derived_knot_values"] == 391
    assert progress["derived_basis_scalar_values"] == 4_692_000
    assert partial["recruitment_basis"].shape == (2000, 391, 3, 2)


def reference_problem():
    soc_A, soc_b = fit.make_soc_rows(sparse.csr_matrix([[1.0]]), [0.0], 1.0, 1)
    nonnegative_A = sparse.csr_matrix(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
    )
    A = sparse.vstack((nonnegative_A, soc_A), format="csc")
    b = np.concatenate((np.array([0.0, 2.0, 1.0, 1.0]), soc_b))
    return fit.conic.ConicProblem(
        sparse.csc_matrix((2, 2)),
        np.array([0.0, -1.0]),
        A,
        b,
        (clarabel.NonnegativeConeT(4), clarabel.SecondOrderConeT(2)),
        ("NonnegativeConeT", "SecondOrderConeT"),
        np.array([4, 2]),
        np.array([0.0, -1.0]),
        np.array([2.0, 1.0]),
        SimpleNamespace(),
        (),
        sparse.csr_matrix((0, 1)),
        np.empty(0),
    )


def solver_freeze():
    return {
        "solver_settings": {
            "max_threads": 1,
            "direct_solve_method": "qdldl",
            "max_iter": 100,
            "time_limit": 20.0,
            "verbose": False,
            "tol_gap_abs": 1e-10,
            "tol_gap_rel": 1e-10,
            "tol_feas": 1e-10,
            "tol_infeas_abs": 1e-10,
            "tol_infeas_rel": 1e-10,
        }
    }


def test_real_clarabel_small_problem_single_entry_and_dual_bound():
    problem = reference_problem()
    partial, progress = {}, {}
    solution, history, settings = fit.solve_conic_problem(
        problem, solver_freeze(), partial=partial, progress=progress
    )
    assert str(solution.status) == "Solved"
    np.testing.assert_allclose(solution.x, [0, 1], atol=2e-8)
    assert progress["fit_calls"] == 1 and history
    assert settings["max_threads"] == 1
    projected, _ = fit.conic.project_conic_dual(
        solution.z, problem.cone_kinds, problem.cone_dims, interior_pad=1e-12
    )
    bound = fit.conic.dual_margin_upper_bound(
        problem.q,
        problem.A,
        problem.b,
        projected,
        problem.lower,
        problem.upper,
        pad_factor=1e-8,
    )
    assert 1 - 2e-8 <= bound["margin_upper_bound"] <= 1 + 2e-7


def test_candidate_is_published_before_composite_failure(monkeypatch):
    pool, membership, ltm = small_pool()
    mapping = fit.build_recruitment_map(
        pool, membership, ltm, fit.gain.canonical_weights(pool)
    )
    problem = SimpleNamespace(anchor_map=mapping)
    solution = SimpleNamespace(
        x=np.concatenate((mapping.lifted_vector, [0.1])),
        s=np.empty(0),
        z=np.empty(0),
        status="Solved",
        iterations=1,
        solve_time=0.0,
        obj_val=-0.1,
        obj_val_dual=-0.1,
        r_prim=0.0,
        r_dual=0.0,
    )
    monkeypatch.setattr(
        fit,
        "solve_conic_problem",
        lambda *_args, **_kwargs: (solution, [], {"max_threads": 1}),
    )
    monkeypatch.setattr(fit, "solution_arrays", lambda *_: {"solver_x": solution.x})
    monkeypatch.setattr(
        fit,
        "evaluate_composite",
        lambda *_: (_ for _ in ()).throw(RuntimeError("generated composite failure")),
    )
    saved = {
        "pool": pool,
        "bank": np.ones((2, 6, 3)),
        "kappa": np.ones(6),
        "basis": np.ones((2, 6, 3, 2)),
    }
    partial = {}
    progress = {"timings_s": {}, "fit_calls": 1}
    with pytest.raises(RuntimeError, match="generated composite failure"):
        fit.calculate(
            {},
            {"total_fit_limit_s": 20},
            progress,
            partial,
            prepared=(problem, {"pre": np.array([1])}, saved),
        )
    np.testing.assert_array_equal(
        partial["candidate_weights"],
        fit.weights_from_vector(pool, mapping, mapping.lifted_vector),
    )
    assert "candidate_lower_budget" in partial
    assert "candidate_solver_lower_aux" in partial
    assert float(partial["returned_margin_t"]) == pytest.approx(0.1)


def minimal_freeze():
    return {
        "minimum_free_disk_bytes": 0,
        "internal_wall_limit_s": 60,
        "maximum_raw_bytes": 10_000_000,
        "total_fit_limit_s": 20,
    }


def run_stub_saved():
    return {
        "bank": np.ones((1, 1, 3)),
        "tau": np.array([0.005, 0.01, 0.04]),
        "kappa": np.ones(1),
        "basis": np.ones((1, 1, 3, 2)),
        "pool": np.ones((1, 1)),
        "membership": np.zeros(1, dtype=int),
    }


def test_problem_write_failure_preserves_problem_without_solver(monkeypatch, tmp_path):
    pre_raw = {"A_data_generated": np.array([0.3]), "qr_generated": np.array([0.4])}
    solver_entries = []
    monkeypatch.setattr(
        fit, "reviewed_inputs", lambda *_: (minimal_freeze(), {"source_sha256": {}})
    )
    monkeypatch.setattr(fit, "load_inputs", lambda *_: {})
    monkeypatch.setattr(
        fit, "validate_saved_inputs", lambda *_args, **_kwargs: run_stub_saved()
    )
    monkeypatch.setattr(
        fit, "build_conic_problem", lambda *_args, **_kwargs: (object(), pre_raw)
    )

    def forbidden_calculate(*_args, **_kwargs):
        solver_entries.append(1)
        raise AssertionError("solver path entered before problem persistence")

    monkeypatch.setattr(fit, "calculate", forbidden_calculate)
    original_save = fit.save_npz_new

    def fail_problem(path, arrays):
        if Path(path).name == "conic-problem.npz":
            raise OSError("generated problem write failure")
        original_save(path, arrays)

    monkeypatch.setattr(fit, "save_npz_new", fail_problem)
    output = tmp_path / "problem-failure"
    with pytest.raises(OSError, match="generated problem write failure"):
        fit.run(Path("f"), "f", Path("r"), "r", output)
    assert solver_entries == []
    with np.load(output / "partial-raw.npz", allow_pickle=False) as partial:
        assert "A_data_generated" in partial.files
        assert "qr_generated" in partial.files
    execution = json.loads((output / "execution-raw.json").read_text())
    assert execution["fit_calls"] == 0
    assert execution["derived_basis_scalar_values"] == 6
    assert execution["automatic_retries"] == 0
    assert execution["primary_failure"]["message"] == "generated problem write failure"


def test_final_raw_failure_preserves_candidate_arrays_and_primary(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        fit, "reviewed_inputs", lambda *_: (minimal_freeze(), {"source_sha256": {}})
    )
    monkeypatch.setattr(fit, "load_inputs", lambda *_: {})
    monkeypatch.setattr(
        fit, "validate_saved_inputs", lambda *_args, **_kwargs: run_stub_saved()
    )
    monkeypatch.setattr(
        fit,
        "build_conic_problem",
        lambda *_args, **_kwargs: (
            object(),
            {"A_data_generated": np.array([0.3])},
        ),
    )

    def calculate_stub(_data, _freeze, progress, partial, **_kwargs):
        progress["fit_calls"] = 1
        candidate = np.ones((1, 1, 3, 2))
        raw = {
            "candidate_weights": candidate,
            "candidate_lower_budget": np.ones(1),
            "candidate_selected_command_rad": np.ones((1, 42)),
            "candidate_command_RMS": np.asarray(0.1),
            "dual_projected_margin_upper_bound": np.asarray(0.2),
        }
        partial.update(raw)
        return raw, {"solver": {}}, {}, object()

    monkeypatch.setattr(fit, "calculate", calculate_stub)
    original_save = fit.save_npz_new

    def fail_final(path, arrays):
        if Path(path).name == "monotone-recruitment-raw.npz":
            raise OSError("generated final raw write failure")
        original_save(path, arrays)

    monkeypatch.setattr(fit, "save_npz_new", fail_final)
    output = tmp_path / "final-failure"
    with pytest.raises(OSError, match="generated final raw write failure"):
        fit.run(Path("f"), "f", Path("r"), "r", output)
    with np.load(output / "partial-raw.npz", allow_pickle=False) as partial:
        assert "candidate_weights" in partial.files
        assert "candidate_lower_budget" in partial.files
        assert "candidate_command_RMS" in partial.files
    execution = json.loads((output / "execution-raw.json").read_text())
    assert execution["fit_calls"] == 1
    assert execution["automatic_retries"] == 0
    assert (
        execution["primary_failure"]["message"] == "generated final raw write failure"
    )
