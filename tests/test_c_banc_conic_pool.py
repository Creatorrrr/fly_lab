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
    "fit_c_banc_m3_conic_pool", TOOLS / "fit_c_banc_m3_conic_pool.py"
)
assert SPEC is not None and SPEC.loader is not None
fit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fit
SPEC.loader.exec_module(fit)


def small_anchor_inputs():
    membership = np.array([0, 0, 0, 1, 1, 2])
    pool = np.zeros((3, 6))
    pool[0, :3] = [0.2, 0.3, 0.5]
    pool[1, 3:5] = [0.4, 0.6]
    pool[2, 5] = 1
    return pool, membership, np.array([1])


def test_anchor_map_uses_last_sorted_member_and_exact_affine_baseline():
    pool, membership, ltm = small_anchor_inputs()
    anchor = fit.build_anchor_map(pool, membership, ltm)
    np.testing.assert_array_equal(anchor.columns, [0, 1, 2])
    np.testing.assert_array_equal(anchor.free_columns, [0, 1])
    np.testing.assert_array_equal(anchor.anchors, [2])
    np.testing.assert_allclose(
        anchor.constant + anchor.matrix @ anchor.baseline_free, pool[0, :3]
    )
    np.testing.assert_allclose(anchor.matrix, [[1, 0], [0, 1], [-1, -1]])


def test_anchor_affine_preserves_sum_and_has_no_posthoc_normalization():
    pool, membership, ltm = small_anchor_inputs()
    anchor = fit.build_anchor_map(pool, membership, ltm)
    free = np.array([0.1, 0.7])
    candidate = fit.candidate_pool_from_free(pool, anchor, free)
    np.testing.assert_allclose(candidate[0], [0.1, 0.7, 0.2, 0, 0, 0], atol=1e-15)
    assert candidate[0].sum() == pytest.approx(1)
    np.testing.assert_array_equal(candidate[1:], pool[1:])


def test_thin_qr_preserves_direct_norm_for_two_vectors():
    rng = np.random.default_rng(145)
    matrix = rng.normal(size=(41, 5))
    offset = rng.normal(size=41)
    r = fit.thin_qr_compress(matrix, offset)
    compressed_matrix, compressed_offset = fit.compressed_affine(r)
    assert r.shape == (6, 6)
    for vector in (rng.normal(size=5), np.linspace(-0.4, 0.7, 5)):
        assert np.linalg.norm(matrix @ vector + offset) == pytest.approx(
            np.linalg.norm(compressed_matrix @ vector + compressed_offset),
            rel=2e-14,
            abs=2e-14,
        )


def test_soc_row_signs_match_norm_le_one_minus_margin():
    matrix = sparse.csr_matrix([[2.0, -1.0], [0.5, 3.0]])
    offset = np.array([0.25, -0.75])
    rows, rhs = fit.make_soc_rows(matrix, offset, 2.5, 2)
    point = np.array([0.2, -0.1, 0.4])
    slack = rhs - rows @ point
    assert slack[0] == pytest.approx(1 - point[-1])
    np.testing.assert_allclose(slack[1:], (matrix @ point[:2] + offset) / 2.5)


def reference_problem() -> fit.ConicProblem:
    # max t subject to ||z|| <= 1-t and 0<=z<=1. Optimum is (z,t)=(0,1).
    soc_A, soc_b = fit.make_soc_rows(sparse.csr_matrix([[1.0]]), [0.0], 1.0, 1)
    nonnegative_A = sparse.csr_matrix(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ]
    )
    nonnegative_b = np.array([0.0, 1.0, 1.0, 1.0])
    A = sparse.vstack((nonnegative_A, soc_A), format="csc")
    b = np.concatenate((nonnegative_b, soc_b))
    anchor = fit.AnchorMap(
        np.array([0, 1]),
        np.array([0, 0]),
        np.array([1]),
        np.array([0]),
        np.array([0]),
        np.array([0.0, 1.0]),
        np.array([[1.0], [-1.0]]),
        np.array([0.5]),
        np.array([5e-7]),
        np.array([[1.0]]),
    )
    return fit.ConicProblem(
        sparse.csc_matrix((2, 2)),
        np.array([0.0, -1.0]),
        A,
        b,
        (clarabel.NonnegativeConeT(4), clarabel.SecondOrderConeT(2)),
        ("NonnegativeConeT", "SecondOrderConeT"),
        np.array([4, 2]),
        np.array([0.0, -1.0]),
        np.array([1.0, 1.0]),
        anchor,
        (),
        sparse.csr_matrix((0, 1)),
        np.empty(0),
    )


def fake_solution(problem, *, status="Solved"):
    return SimpleNamespace(
        x=np.array([0.0, 1.0]),
        s=np.asarray(problem.b - problem.A @ np.array([0.0, 1.0])),
        z=np.zeros(problem.b.size),
        status=status,
        iterations=3,
        solve_time=0.01,
        obj_val=-1.0,
        obj_val_dual=-1.0,
        r_prim=0.0,
        r_dual=0.0,
    )


def test_reference_known_small_socp_and_exact_settings():
    problem = reference_problem()
    freeze = {
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
    solution, history, settings = fit.solve_conic_problem(problem, freeze)
    assert str(solution.status) == "Solved"
    np.testing.assert_allclose(solution.x, [0, 1], atol=2e-8)
    assert history and settings["max_threads"] == 1
    assert settings["direct_solve_method"] == "qdldl"


def test_solver_factory_is_called_exactly_once():
    problem = reference_problem()
    calls = []

    class FakeSolver:
        def __init__(self, *_args):
            calls.append("construct")

        def set_termination_callback(self, callback):
            self.callback = callback

        def solve(self):
            calls.append("solve")
            return fake_solution(problem)

        def get_settings(self):
            return clarabel.DefaultSettings()

    solution, history, _settings = fit.solve_conic_problem(
        problem,
        {"solver_settings": {}},
        solver_factory=FakeSolver,
    )
    assert isinstance(solution, SimpleNamespace)
    assert history == []
    assert calls == ["construct", "solve"]


def test_constructor_failure_records_attempt_without_fit_call():
    problem = reference_problem()
    partial, progress, state, history = {}, {}, {}, []

    def failed_constructor(*_args):
        raise RuntimeError("generated constructor failure")

    with pytest.raises(RuntimeError, match="generated constructor failure"):
        fit.solve_conic_problem(
            problem,
            {"solver_settings": {}},
            solver_factory=failed_constructor,
            partial=partial,
            progress=progress,
            state=state,
            history=history,
        )
    assert progress["fit_calls"] == 0
    assert progress["solver_state"]["attempted"] is True
    assert progress["solver_state"]["constructed"] is False
    assert progress["solver_state"]["phase"] == "constructor_error"
    assert json.loads(str(partial["history_json"])) == []


def test_callback_registration_failure_records_constructed_without_fit_call():
    problem = reference_problem()
    partial, progress = {}, {}

    class FakeSolver:
        def __init__(self, *_args):
            pass

        def set_termination_callback(self, _callback):
            raise RuntimeError("generated callback registration failure")

    with pytest.raises(RuntimeError, match="generated callback registration failure"):
        fit.solve_conic_problem(
            problem,
            {"solver_settings": {}},
            solver_factory=FakeSolver,
            partial=partial,
            progress=progress,
        )
    assert progress["fit_calls"] == 0
    assert progress["solver_state"]["constructed"] is True
    assert progress["solver_state"]["callback_registered"] is False
    assert progress["solver_state"]["phase"] == "callback_registration_error"


def test_callback_then_solve_failure_preserves_completed_history_and_primary_error():
    problem = reference_problem()
    partial, progress = {}, {}

    class FakeSolver:
        def __init__(self, *_args):
            pass

        def set_termination_callback(self, callback):
            self.callback = callback

        def solve(self):
            self.callback(SimpleNamespace(iterations=7, status="Unsolved"))
            raise ValueError("generated solve failure after callback")

    with pytest.raises(ValueError, match="generated solve failure after callback"):
        fit.solve_conic_problem(
            problem,
            {"solver_settings": {}},
            solver_factory=FakeSolver,
            partial=partial,
            progress=progress,
        )
    assert progress["fit_calls"] == 1
    assert progress["solver_state"]["solve_entered"] is True
    assert progress["solver_state"]["solve_returned"] is False
    assert progress["solver_state"]["phase"] == "solve_error"
    assert json.loads(str(partial["history_json"])) == [
        {
            "mu": None,
            "sigma": None,
            "step_length": None,
            "iterations": 7.0,
            "cost_primal": None,
            "cost_dual": None,
            "res_primal": None,
            "res_dual": None,
            "res_primal_inf": None,
            "res_dual_inf": None,
            "gap_abs": None,
            "gap_rel": None,
            "ktratio": None,
            "solve_time": None,
            "status": "Unsolved",
        }
    ]


def test_project_conic_dual_is_in_each_self_dual_cone():
    raw = np.array([-2.0, 3.0, -1.0, 4.0, 0.0])
    projected, margins = fit.project_conic_dual(
        raw,
        ("NonnegativeConeT", "SecondOrderConeT"),
        [2, 3],
        interior_pad=1e-12,
    )
    assert np.all(projected[:2] >= 0)
    assert projected[2] > np.linalg.norm(projected[3:])
    assert np.min(margins) >= 0


def test_dual_box_bound_matches_reference_optimum_upper_bound():
    problem = reference_problem()
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    solution = clarabel.DefaultSolver(
        problem.P, problem.q, problem.A, problem.b, list(problem.cones), settings
    ).solve()
    projected, _ = fit.project_conic_dual(
        solution.z,
        problem.cone_kinds,
        problem.cone_dims,
        interior_pad=1e-12,
    )
    bound = fit.dual_margin_upper_bound(
        problem.q,
        problem.A,
        problem.b,
        projected,
        problem.lower,
        problem.upper,
        pad_factor=1e-8,
    )
    assert bound["margin_upper_bound"] >= 1 - 2e-8
    assert bound["margin_upper_bound"] <= 1 + 2e-7
    np.testing.assert_allclose(
        bound["objective_lower_bound"],
        -bound["margin_upper_bound_unpadded"],
    )


def test_solution_arrays_preserve_primal_stationarity_and_cone_order():
    problem = reference_problem()
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    solution = clarabel.DefaultSolver(
        problem.P, problem.q, problem.A, problem.b, list(problem.cones), settings
    ).solve()
    raw = fit.solution_arrays(
        problem,
        solution,
        [{"iterations": 0, "status": "Unsolved"}],
        {"dual_interior_pad": 1e-12, "dual_upper_bound_pad": 1e-8},
    )
    np.testing.assert_allclose(
        raw["primal_residual"], problem.A @ solution.x + solution.s - problem.b
    )
    np.testing.assert_allclose(
        raw["stationarity_raw"], problem.q + problem.A.T @ solution.z
    )
    assert json.loads(str(raw["history_json"]))[0]["iterations"] == 0


def test_nonnegative_rows_enforce_free_anchor_margin_and_startup_signs():
    pool, membership, ltm = small_anchor_inputs()
    anchor = fit.build_anchor_map(pool, membership, ltm)
    startup_matrix = sparse.csr_matrix([[2.0, -1.0]])
    startup_offset = np.array([0.01])
    A, b = fit._make_nonnegative_rows(
        anchor, startup_matrix, startup_offset, 0.04, (-1.0, 1.0)
    )
    point = np.array([0.2, 0.3, 0.1])
    slack = b - A @ point
    assert np.all(slack[: 2 * point[:-1].size] >= 0)
    assert slack[-2] == pytest.approx(
        1 - point[-1] - (startup_matrix @ point[:-1] + startup_offset)[0] / 0.04
    )
    assert slack[-1] == pytest.approx(
        1 - point[-1] + (startup_matrix @ point[:-1] + startup_offset)[0] / 0.04
    )


def test_physical_verdict_requires_returned_pool_positive_and_above_floor(monkeypatch):
    membership = np.arange(391) % 96
    candidate = np.zeros((96, 391))
    for muscle in range(96):
        columns = np.flatnonzero(membership == muscle)
        candidate[muscle, columns] = 1 / columns.size
    monkeypatch.setattr(fit, "validate_pool_contract", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        fit,
        "assess_composite",
        lambda *_: {
            "conditional_static_pool_proposal": True,
            "gates": {"all_original_physical_gates": True},
        },
    )
    freeze = {
        "row_sum_atol": 1e-12,
        "pool_floor_fraction": 1e-6,
        "lower_bound_atol": 1e-12,
    }
    passed = fit.assess_physical_candidate(
        {"membership": membership, "ltm": np.arange(6)}, candidate, {}, freeze
    )
    assert passed["conditional_static_pool_proposal"] is True
    failed = candidate.copy()
    failed[membership[0], 0] = 0
    rejected = fit.assess_physical_candidate(
        {"membership": membership, "ltm": np.arange(6)}, failed, {}, freeze
    )
    assert rejected["conditional_static_pool_proposal"] is False
    assert rejected["gates"]["strictly_positive_same_support"] is False


def minimal_freeze():
    return {
        "minimum_free_disk_bytes": 0,
        "internal_wall_limit_s": 60,
        "maximum_raw_bytes": 10_000_000,
    }


def generated_calculate_saved():
    return {
        key: np.empty(0)
        for key in (
            "b",
            "pool",
            "membership",
            "moment",
            "rest",
            "ltm",
            "target_indices",
            "target",
            "pair_start",
            "pair_end",
            "target_velocity",
            "startup_target",
            "startup_tendon_target",
        )
    }


def test_late_return_preserves_bare_solution_before_time_rejection(monkeypatch):
    problem = reference_problem()
    saved = generated_calculate_saved()
    partial, progress = {}, {"timings_s": {}}
    clock = iter([0.0, 31.0])
    monkeypatch.setattr(fit, "validate_saved_inputs", lambda *_: saved)
    monkeypatch.setattr(fit.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        fit,
        "solve_conic_problem",
        lambda *_args, **_kwargs: (
            fake_solution(problem, status="MaxTime"),
            [{"iterations": 5, "status": "MaxTime"}],
            {},
        ),
    )
    freeze = {"total_fit_limit_s": 30}
    with pytest.raises(ValueError, match="fit exceeded 30 seconds"):
        fit.calculate(
            {"e143": {}},
            freeze,
            progress,
            partial,
            prepared=(problem, fit._problem_arrays(problem)),
        )
    np.testing.assert_array_equal(partial["solver_x"], [0.0, 1.0])
    np.testing.assert_array_equal(partial["solver_s"], fake_solution(problem).s)
    np.testing.assert_array_equal(partial["solver_z_raw"], np.zeros(problem.b.size))
    assert str(partial["solver_status"]) == "MaxTime"
    assert json.loads(str(partial["history_json"])) == [
        {"iterations": 5, "status": "MaxTime"}
    ]


def test_derived_diagnostic_failure_keeps_bare_return_and_original_error(monkeypatch):
    problem = reference_problem()
    saved = generated_calculate_saved()
    partial, progress = {}, {"timings_s": {}}
    monkeypatch.setattr(fit, "validate_saved_inputs", lambda *_: saved)
    monkeypatch.setattr(
        fit,
        "solve_conic_problem",
        lambda *_args, **_kwargs: (fake_solution(problem), [], {}),
    )

    def failed_diagnostic(*_args):
        raise ArithmeticError("generated dual diagnostic failure")

    monkeypatch.setattr(fit, "solution_arrays", failed_diagnostic)
    with pytest.raises(ArithmeticError, match="generated dual diagnostic failure"):
        fit.calculate(
            {"e143": {}},
            {"total_fit_limit_s": 30},
            progress,
            partial,
            prepared=(problem, fit._problem_arrays(problem)),
        )
    np.testing.assert_array_equal(partial["solver_x"], [0.0, 1.0])
    assert str(partial["solver_status"]) == "Solved"
    assert "solver_z_projected" not in partial


def test_run_persists_problem_before_solver_raw_and_assessment(monkeypatch, tmp_path):
    events = []
    problem = reference_problem()
    monkeypatch.setattr(
        fit, "reviewed_inputs", lambda *_: (minimal_freeze(), {"source_sha256": {}})
    )
    monkeypatch.setattr(fit, "load_inputs", lambda *_: {})
    monkeypatch.setattr(fit, "validate_saved_inputs", lambda *_: {})
    monkeypatch.setattr(
        fit, "build_conic_problem", lambda *_: (problem, {"q": problem.q})
    )

    def calculation(*_args, **_kwargs):
        events.append("solve")
        return (
            {"solver_x": np.array([0.0, 1.0])},
            {},
            {"saved": {}, "candidate_pool": np.array([]), "composite": {}},
            problem,
        )

    monkeypatch.setattr(fit, "calculate", calculation)
    monkeypatch.setattr(
        fit,
        "assess_physical_candidate",
        lambda *_: (
            events.append("assess") or {"conditional_static_pool_proposal": False}
        ),
    )
    original_save = fit.save_npz_new

    def record_save(path, arrays):
        events.append(Path(path).name)
        original_save(path, arrays)

    monkeypatch.setattr(fit, "save_npz_new", record_save)
    result = fit.run(Path("f"), "f", Path("r"), "r", tmp_path / "out")
    assert result["status"] == "pass"
    assert events == ["conic-problem.npz", "solve", "conic-pool-raw.npz", "assess"]


def test_primary_error_survives_partial_persistence_failure(monkeypatch, tmp_path):
    problem = reference_problem()
    monkeypatch.setattr(
        fit, "reviewed_inputs", lambda *_: (minimal_freeze(), {"source_sha256": {}})
    )
    monkeypatch.setattr(fit, "load_inputs", lambda *_: {})
    monkeypatch.setattr(fit, "validate_saved_inputs", lambda *_: {})
    monkeypatch.setattr(
        fit, "build_conic_problem", lambda *_: (problem, {"q": problem.q})
    )

    def calculation(_data, _freeze, _progress, partial, prepared=None):
        del prepared
        partial["started"] = np.array([1])
        raise ValueError("primary calculation")

    monkeypatch.setattr(fit, "calculate", calculation)
    original_save = fit.save_npz_new

    def failed_partial(path, arrays):
        if Path(path).name == "partial-raw.npz":
            raise OSError("partial persistence")
        original_save(path, arrays)

    monkeypatch.setattr(fit, "save_npz_new", failed_partial)
    with pytest.raises(ValueError, match="primary calculation") as caught:
        fit.run(Path("f"), "f", Path("r"), "r", tmp_path / "out")
    assert any("partial persistence" in note for note in caught.value.__notes__)
    summary = json.loads(
        (tmp_path / "out" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["conditional_static_pool_proposal"] is False


def test_expected_summary_contract_on_failure():
    summary = fit.failure_summary(RuntimeError("x"), {})
    assert summary["candidate_adopted"] is False
    assert summary["F1_F8"] == "pending"
    assert summary["F6_exposed"] is False
    assert summary["active_UI_changed"] is False
    assert summary["whole_goal_complete"] is False


def test_freeze_and_design_hash_constants_match_files():
    assert fit.sha256(fit.DEFAULT_FREEZE) == fit.FREEZE_SHA
    assert (
        fit.sha256(ROOT / "docs/BANC_M3_E145_GLOBAL_CONIC_POOL_DESIGN_20260921.md")
        == fit.DESIGN_SHA
    )


def test_problem_arrays_include_independent_reviewer_contract():
    arrays = fit._problem_arrays(reference_problem())
    required = {
        "P_data",
        "P_indices",
        "P_indptr",
        "P_shape",
        "q",
        "A_data",
        "A_indices",
        "A_indptr",
        "A_shape",
        "b",
        "cone_kinds",
        "cone_dims",
        "anchor_columns",
        "free_columns",
        "anchor_constant",
        "anchor_matrix",
        "baseline_free",
    }
    assert required <= set(arrays)
    assert arrays["q"][-1] == -1
    assert arrays["P_data"].size == 0
