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
    "fit_c_banc_m3_muscle_gain", TOOLS / "fit_c_banc_m3_muscle_gain.py"
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


def test_direct_gain_map_is_muscle_then_motor_major_without_anchor():
    pool, membership, ltm = small_pool()
    mapping = fit.build_gain_map(pool, membership, ltm)
    np.testing.assert_array_equal(mapping.variable_muscles, [0])
    np.testing.assert_array_equal(mapping.fixed_muscles, [1, 2])
    np.testing.assert_array_equal(mapping.variable_flat, np.arange(9))
    np.testing.assert_array_equal(mapping.variable_motors, np.repeat(np.arange(3), 3))
    np.testing.assert_array_equal(mapping.variable_branches, np.tile(np.arange(3), 3))
    assert mapping.variable_flat.size == mapping.gain_rows.shape[1]
    assert mapping.variable_flat.size == mapping.canonical_vector.size
    np.testing.assert_allclose(mapping.gain_rows @ mapping.canonical_vector, [1])
    np.testing.assert_allclose(mapping.gain_rows @ mapping.lifted_vector, [1])
    np.testing.assert_allclose(mapping.lower, np.full(9, 1e-6 / 3))
    np.testing.assert_array_equal(mapping.upper, np.full(9, 2.0))


def test_candidate_weights_and_gain_are_direct_without_normalization():
    pool, membership, ltm = small_pool()
    mapping = fit.build_gain_map(pool, membership, ltm)
    vector = mapping.lifted_vector * 1.5
    weights = fit.weights_from_vector(pool, mapping, vector)
    gain = fit.gains_from_vector(mapping, vector)
    np.testing.assert_array_equal(
        weights[
            mapping.variable_muscle_for_coefficient,
            mapping.variable_motors,
            mapping.variable_branches,
        ],
        vector,
    )
    assert gain[0] == pytest.approx(1.5)
    np.testing.assert_array_equal(gain[1:], 1)
    np.testing.assert_array_equal(
        weights[mapping.fixed_muscles],
        fit.canonical_weights(pool)[mapping.fixed_muscles],
    )


def test_nonnegative_rows_enforce_branch_gain_box_t_and_startup():
    pool, membership, ltm = small_pool()
    mapping = fit.build_gain_map(pool, membership, ltm)
    startup_matrix = sparse.csr_matrix(np.ones((2, mapping.variable_flat.size)) * 0.001)
    startup_offset = np.array([0.002, -0.003])
    A, b = fit._make_nonnegative_rows(
        mapping,
        startup_matrix,
        startup_offset,
        0.04,
        (-1.0, 1.0),
        (0.500001, 1.999999),
    )
    point = np.concatenate((mapping.lifted_vector, [-0.251]))
    slack = b - A @ point
    assert np.min(slack) >= -1e-15
    too_low = point.copy()
    too_low[:-1] *= 0.49
    assert np.min(b - A @ too_low) < 0


def generated_uneven_problem():
    rng = np.random.default_rng(14701)
    membership = np.repeat(np.arange(96), 2)
    pool = np.zeros((96, membership.size))
    pool[membership, np.arange(membership.size)] = 0.5
    ltm = np.array([0, 17, 32, 48, 65, 80])
    bank = rng.uniform(0.0, 2.0, size=(2000, membership.size, 3))
    bank[:, :, 1] = causal_middle = fit.causal.causal_filter_bank(
        rng.uniform(size=(2000, membership.size))
    )[:, :, 1]
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
    baseline_activation = causal_middle @ pool.T
    baseline_command = rest + fit.COMMAND_GAIN * baseline_activation @ moment.T
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
        "b": causal_middle,
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
    freeze = {
        "branch_guard_fraction": 1e-6,
        "weight_upper_bound": 2.0,
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
    }
    return saved, {"anatomy": anatomy}, freeze


def test_full_uneven_anatomy_15_soc_affines_match_direct_physics():
    saved, data, freeze = generated_uneven_problem()
    problem, raw = fit.build_conic_problem(saved, data, freeze)
    assert len(problem.criteria) == 15
    assert problem.cone_kinds.count("SecondOrderConeT") == 15
    mapping = problem.anchor_map
    weights = fit.weights_from_vector(saved["pool"], mapping, mapping.lifted_vector)
    composite = fit.evaluate_composite(saved, weights)
    arrays = composite["arrays"]
    command_error = arrays["candidate_selected_command_rad"] - saved["target"]
    velocity_error = (
        arrays["candidate_selected_velocity_rad_s"] - saved["target_velocity"]
    )
    startup_error = arrays["candidate_startup_command_rad"] - saved["startup_target"]
    baseline_command_error = arrays["baseline_selected_command_rad"] - saved["target"]
    baseline_velocity_error = (
        arrays["baseline_selected_velocity_rad_s"] - saved["target_velocity"]
    )
    baseline_startup_error = (
        arrays["baseline_startup_command_rad"] - saved["startup_target"]
    )
    expected = {
        "overall_command": np.linalg.norm(command_error),
        "overall_velocity": np.linalg.norm(velocity_error),
        "overall_startup": np.linalg.norm(startup_error),
    }
    canonical_expected = {
        "overall_command": np.linalg.norm(baseline_command_error),
        "overall_velocity": np.linalg.norm(baseline_velocity_error),
        "overall_startup": np.linalg.norm(baseline_startup_error),
    }
    for index, leg in enumerate(fit.LEGS):
        expected[f"{leg}_command"] = np.linalg.norm(
            command_error[:, 7 * index : 7 * (index + 1)]
        )
        expected[f"{leg}_velocity"] = np.linalg.norm(
            velocity_error[:, 7 * index : 7 * (index + 1)]
        )
        canonical_expected[f"{leg}_command"] = np.linalg.norm(
            baseline_command_error[:, 7 * index : 7 * (index + 1)]
        )
        canonical_expected[f"{leg}_velocity"] = np.linalg.norm(
            baseline_velocity_error[:, 7 * index : 7 * (index + 1)]
        )
    for criterion in problem.criteria:
        actual = np.linalg.norm(
            criterion.direct_matrix @ mapping.lifted_vector + criterion.direct_offset
        )
        assert actual == pytest.approx(expected[criterion.name], rel=2e-12, abs=2e-12)
        canonical_actual = np.linalg.norm(
            criterion.direct_matrix @ mapping.canonical_vector + criterion.direct_offset
        )
        assert canonical_actual == pytest.approx(
            canonical_expected[criterion.name], rel=2e-12, abs=2e-12
        )
    np.testing.assert_array_equal(raw["variable_flat"], mapping.variable_flat)
    np.testing.assert_array_equal(raw["row_sum_matrix"], mapping.gain_rows.toarray())
    np.testing.assert_array_equal(
        problem.lower[:-1], np.zeros(mapping.variable_flat.size)
    )
    np.testing.assert_array_equal(
        problem.upper[:-1], np.full(mapping.variable_flat.size, 2.0)
    )


def test_fixed_activation_and_tendon_are_bit_exact():
    saved, _data, _freeze = generated_uneven_problem()
    mapping = fit.build_gain_map(saved["pool"], saved["membership"], saved["ltm"])
    saved["anchor"] = mapping
    weights = fit.weights_from_vector(
        saved["pool"], mapping, mapping.lifted_vector * 1.2
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


def test_physical_structure_requires_gain_branch_floor_and_support():
    saved, _data, freeze = generated_uneven_problem()
    mapping = fit.build_gain_map(saved["pool"], saved["membership"], saved["ltm"])
    saved["anchor"] = mapping
    weights = fit.weights_from_vector(saved["pool"], mapping, mapping.lifted_vector)
    gain = fit.gains_from_vector(mapping, mapping.lifted_vector)
    composite = fit.evaluate_composite(saved, weights)
    assessment = fit.assess_physical_candidate(saved, weights, gain, composite, freeze)
    for name in (
        "same_anatomical_support",
        "all_branches_nonnegative",
        "gain_bounds_with_tolerance",
        "original_motor_floor_with_tolerance",
        "fixed_LTM_and_singleton_rows_exact",
        "fixed_activation_output_exact",
    ):
        assert assessment["gates"][name] is True
    bad_vector = mapping.lifted_vector * 2.1
    bad_weights = fit.weights_from_vector(saved["pool"], mapping, bad_vector)
    bad_gain = fit.gains_from_vector(mapping, bad_vector)
    bad_composite = fit.evaluate_composite(saved, bad_weights)
    rejected = fit.assess_physical_candidate(
        saved, bad_weights, bad_gain, bad_composite, freeze
    )
    assert rejected["gates"]["gain_bounds_with_tolerance"] is False
    assert rejected["conditional_muscle_gain_proposal"] is False


def test_physical_gain_uses_effective_weight_sum_not_csr_rounding_order():
    membership = np.concatenate((np.zeros(13, dtype=int), np.arange(1, 96)))
    pool = np.zeros((96, membership.size))
    for muscle in range(96):
        mask = membership == muscle
        pool[muscle, mask] = 1.0 / np.count_nonzero(mask)
    mapping = fit.build_gain_map(pool, membership, np.array([1]))
    rng = np.random.default_rng(14703)
    selected = None
    for _draw in range(3000):
        vector = rng.uniform(size=39)
        vector *= 1.999999 / vector.sum()
        weights = fit.weights_from_vector(pool, mapping, vector)
        csr_gain = fit.gains_from_vector(mapping, vector)
        physical_gain = weights.sum(axis=(1, 2))
        if not np.allclose(csr_gain, physical_gain, atol=1e-15, rtol=0):
            selected = (weights, physical_gain, csr_gain)
            break
    assert selected is not None
    weights, physical_gain, csr_gain = selected
    assert abs(csr_gain[0] - physical_gain[0]) > 1e-15
    metrics = {
        "command_RMS": 0.01,
        "velocity_RMS": 0.01,
        "startup_q_RMS": 0.001,
        "startup_q_max_abs": 0.001,
        "startup_LTM_RMS": 0.0,
        "startup_LTM_max_abs": 0.0,
        "by_leg": {
            leg: {"command_RMS": 0.01, "velocity_RMS": 0.01} for leg in fit.LEGS
        },
    }
    baseline = {**metrics, "command_RMS": 1.0, "velocity_RMS": 1.0}
    composite = {
        "metrics": {"candidate": metrics, "baseline": baseline},
        "arrays": {
            "candidate_activation": np.zeros((1, 96)),
            "baseline_activation": np.zeros((1, 96)),
            "candidate_startup_tendon": np.zeros((1, 6)),
            "baseline_startup_tendon": np.zeros((1, 6)),
        },
    }
    freeze = {
        "original_MN_floor_fraction": 1e-6,
        "lower_bound_atol": 1e-12,
        "gain_bounds": [0.5, 2.0],
        "startup_limits": {
            "q_RMS": 0.018,
            "q_max_abs": 0.04,
            "LTM_RMS": 0.009,
            "LTM_max_abs": 0.025,
        },
        "proposal_command_RMS_ratio": 0.8,
        "proposal_velocity_RMS_ratio": 1.0,
    }
    assessment = fit.assess_physical_candidate(
        {
            "pool": pool,
            "membership": membership,
            "ltm": np.array([1]),
            "anchor": mapping,
        },
        weights,
        physical_gain,
        composite,
        freeze,
    )
    assert assessment["gates"]["candidate_gain_matches_effective_weights"] is True
    assert assessment["conditional_muscle_gain_proposal"] is True


def reference_problem():
    soc_A, soc_b = fit.make_soc_rows(sparse.csr_matrix([[1.0]]), [0.0], 1.0, 1)
    nonnegative_A = sparse.csr_matrix(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
    )
    nonnegative_b = np.array([0.0, 2.0, 1.0, 1.0])
    A = sparse.vstack((nonnegative_A, soc_A), format="csc")
    b = np.concatenate((nonnegative_b, soc_b))
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
            "time_limit": 30.0,
            "verbose": False,
            "tol_gap_abs": 1e-10,
            "tol_gap_rel": 1e-10,
            "tol_feas": 1e-10,
            "tol_infeas_abs": 1e-10,
            "tol_infeas_rel": 1e-10,
        }
    }


def test_real_clarabel_small_problem_and_dual_box_bound():
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


def test_validate_reused_bank_never_filters():
    pool = np.zeros((96, 391))
    membership = np.arange(391) % 96
    ltm = np.array([0, 17, 32, 48, 65, 80])
    b = np.arange(2000 * 391, dtype=np.float64).reshape(2000, 391) / 1000
    bank = np.stack((b + 1, b, b - 1), axis=2)
    saved = {
        "b": b,
        "pool": pool,
        "membership": membership,
        "moment": np.zeros((42, 96)),
        "rest": np.zeros(42),
        "ltm": ltm,
        "baseline": np.zeros((2000, 42)),
        "target_indices": np.empty(1827, dtype=int),
        "target": np.zeros((1827, 42)),
        "pair_start": np.empty(1826, dtype=int),
        "pair_end": np.empty(1826, dtype=int),
        "target_velocity": np.zeros((1826, 42)),
        "startup_target": np.zeros((99, 42)),
        "startup_tendon_target": np.zeros((99, 6)),
    }
    raw = {
        "filter_bank": bank,
        "filter_tau_s": np.array([0.005, 0.01, 0.04]),
        "baseline_pool": pool,
        "membership": membership,
        "moment": saved["moment"],
        "rest": saved["rest"],
        "ltm": ltm,
        "target_indices": saved["target_indices"],
        "target_rad": saved["target"],
        "velocity_pair_start_indices": saved["pair_start"],
        "velocity_pair_end_indices": saved["pair_end"],
        "target_velocity_rad_s": saved["target_velocity"],
        "startup_target_rad": saved["startup_target"],
        "startup_target_tendon": saved["startup_tendon_target"],
    }
    result = fit.validate_saved_inputs(
        {"e146": raw},
        {"tau_s": [0.005, 0.01, 0.04]},
        base_validator=lambda *_: saved,
    )
    np.testing.assert_array_equal(result["bank"], bank)
    np.testing.assert_array_equal(result["bank"][:, :, 1], b)


def test_candidate_is_published_before_composite_failure(monkeypatch):
    pool, membership, ltm = small_pool()
    mapping = fit.build_gain_map(pool, membership, ltm)
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
    partial = {}
    progress = {"timings_s": {}, "fit_calls": 1}
    with pytest.raises(RuntimeError, match="generated composite failure"):
        fit.calculate(
            {},
            {"total_fit_limit_s": 30},
            progress,
            partial,
            prepared=(problem, {"pre": np.array([1])}, {"pool": pool}),
        )
    np.testing.assert_array_equal(
        partial["candidate_weights"],
        fit.weights_from_vector(pool, mapping, mapping.lifted_vector),
    )
    np.testing.assert_allclose(partial["candidate_gain"], 1)
    assert float(partial["returned_margin_t"]) == pytest.approx(0.1)


def minimal_freeze():
    return {
        "minimum_free_disk_bytes": 0,
        "internal_wall_limit_s": 60,
        "maximum_raw_bytes": 10_000_000,
        "total_fit_limit_s": 30,
    }


def run_stub_saved():
    return {
        "bank": np.ones((1, 6, 3)),
        "tau": np.array([0.005, 0.01, 0.04]),
        "pool": small_pool()[0],
        "membership": small_pool()[1],
    }


def test_problem_write_failure_persists_built_problem_without_solver(
    monkeypatch, tmp_path
):
    pre_raw = {
        "A_data_generated": np.array([0.3]),
        "qr_generated": np.array([0.4]),
    }
    solver_entries = []
    monkeypatch.setattr(
        fit, "reviewed_inputs", lambda *_: (minimal_freeze(), {"source_sha256": {}})
    )
    monkeypatch.setattr(fit, "load_inputs", lambda *_: {})
    monkeypatch.setattr(
        fit, "validate_saved_inputs", lambda *_args, **_kwargs: run_stub_saved()
    )
    monkeypatch.setattr(fit, "build_conic_problem", lambda *_: (object(), pre_raw))

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
    assert execution["automatic_retries"] == 0
    assert execution["primary_failure"]["message"] == "generated problem write failure"


def test_final_raw_write_failure_persists_composite_and_primary(monkeypatch, tmp_path):
    saved = run_stub_saved()
    pool, membership, ltm = small_pool()
    mapping = fit.build_gain_map(pool, membership, ltm)
    saved["pool"] = pool
    saved["membership"] = membership
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
    metrics = {
        "command_RMS": 0.01,
        "velocity_RMS": 0.01,
        "startup_q_RMS": 0.001,
        "startup_q_max_abs": 0.001,
        "startup_LTM_RMS": 0.0,
        "startup_LTM_max_abs": 0.0,
        "by_leg": {
            leg: {"command_RMS": 0.01, "velocity_RMS": 0.01} for leg in fit.LEGS
        },
    }
    composite = {
        "metrics": {"candidate": metrics, "baseline": metrics},
        "arrays": {
            "candidate_selected_command_rad": np.full((1, 42), 0.01),
            "baseline_selected_command_rad": np.zeros((1, 42)),
        },
    }
    solver_entries = []
    monkeypatch.setattr(
        fit, "reviewed_inputs", lambda *_: (minimal_freeze(), {"source_sha256": {}})
    )
    monkeypatch.setattr(fit, "load_inputs", lambda *_: {})
    monkeypatch.setattr(fit, "validate_saved_inputs", lambda *_args, **_kwargs: saved)
    monkeypatch.setattr(
        fit,
        "build_conic_problem",
        lambda *_: (problem, {"A_data_generated": np.array([0.3])}),
    )

    def solve_stub(*_args, **kwargs):
        solver_entries.append(1)
        kwargs["progress"]["fit_calls"] = len(solver_entries)
        return solution, [], {"max_threads": 1}

    monkeypatch.setattr(fit, "solve_conic_problem", solve_stub)
    monkeypatch.setattr(fit, "solution_arrays", lambda *_: {"solver_x": solution.x})
    monkeypatch.setattr(fit, "evaluate_composite", lambda *_: composite)
    original_save = fit.save_npz_new

    def fail_final(path, arrays):
        if Path(path).name == "muscle-gain-raw.npz":
            raise OSError("generated final raw write failure")
        original_save(path, arrays)

    monkeypatch.setattr(fit, "save_npz_new", fail_final)
    output = tmp_path / "final-failure"
    with pytest.raises(OSError, match="generated final raw write failure"):
        fit.run(Path("f"), "f", Path("r"), "r", output)
    assert solver_entries == [1]
    with np.load(output / "partial-raw.npz", allow_pickle=False) as partial:
        assert "candidate_weights" in partial.files
        assert "candidate_gain" in partial.files
        assert "candidate_selected_command_rad" in partial.files
        assert "candidate_command_RMS" in partial.files
    execution = json.loads((output / "execution-raw.json").read_text())
    assert execution["fit_calls"] == 1
    assert execution["automatic_retries"] == 0
    assert (
        execution["primary_failure"]["message"] == "generated final raw write failure"
    )


def test_run_saves_raw_before_assessment(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        fit, "reviewed_inputs", lambda *_: (minimal_freeze(), {"source_sha256": {}})
    )
    monkeypatch.setattr(fit, "load_inputs", lambda *_: {})
    monkeypatch.setattr(
        fit,
        "validate_saved_inputs",
        lambda *_args, **_kwargs: {
            "bank": np.ones((1, 1, 3)),
            "tau": np.array([0.005, 0.01, 0.04]),
            "pool": np.ones((1, 1)),
            "membership": np.zeros(1, dtype=int),
        },
    )
    monkeypatch.setattr(
        fit, "build_conic_problem", lambda *_: (object(), {"pre": np.array([1])})
    )
    monkeypatch.setattr(
        fit,
        "calculate",
        lambda *_args, **_kwargs: (
            {
                "candidate_weights": np.ones((1, 1, 3)),
                "dual_projected_margin_upper_bound": np.asarray(0.2),
            },
            {"solver": {}},
            {
                "saved": {},
                "weights": np.ones((1, 1, 3)),
                "candidate_gain": np.ones(1),
                "composite": {},
            },
            object(),
        ),
    )
    original_save = fit.save_npz_new

    def recording_save(path, arrays):
        events.append(("save", Path(path).name))
        original_save(path, arrays)

    def assess(*_args):
        events.append(("assess",))
        return {"conditional_muscle_gain_proposal": False}

    monkeypatch.setattr(fit, "save_npz_new", recording_save)
    monkeypatch.setattr(fit, "assess_physical_candidate", assess)
    result = fit.run(Path("f"), "f", Path("r"), "r", tmp_path / "out")
    assert result["status"] == "pass"
    assert events == [
        ("save", "conic-problem.npz"),
        ("save", "muscle-gain-raw.npz"),
        ("assess",),
    ]


def test_primary_error_survives_late_resource_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fit, "reviewed_inputs", lambda *_: (minimal_freeze(), {"source_sha256": {}})
    )
    monkeypatch.setattr(fit, "load_inputs", lambda *_: {})
    monkeypatch.setattr(
        fit,
        "validate_saved_inputs",
        lambda *_args, **_kwargs: {
            "bank": np.ones((1, 1, 3)),
            "tau": np.array([0.005, 0.01, 0.04]),
            "pool": np.ones((1, 1)),
            "membership": np.zeros(1, dtype=int),
        },
    )
    monkeypatch.setattr(
        fit, "build_conic_problem", lambda *_: (object(), {"pre": np.array([1])})
    )

    def fail(_data, _freeze, _progress, partial, **_kwargs):
        partial["attempt"] = np.array([1])
        raise ValueError("primary generated calculation")

    monkeypatch.setattr(fit, "calculate", fail)
    monkeypatch.setattr(
        fit,
        "output_bytes",
        lambda *_: (_ for _ in ()).throw(OSError("late generated stat")),
    )
    with pytest.raises(ValueError, match="primary generated calculation") as caught:
        fit.run(Path("f"), "f", Path("r"), "r", tmp_path / "out")
    assert any(
        "late generated stat" in note for note in getattr(caught.value, "__notes__", [])
    )
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["failure"]["message"] == "primary generated calculation"
