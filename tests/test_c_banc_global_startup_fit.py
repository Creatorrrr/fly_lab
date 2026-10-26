"""Cheap E211 sparse assembly, fail-closed gate, and mock solver checks."""

import json
from pathlib import Path

import clarabel
import numpy as np
import pytest
from scipy import sparse

from tools import fit_c_banc_m3_global_startup as fit


def synthetic_locals():
    records = []
    for i, leg in enumerate(fit.LEGS):
        M = sparse.csc_matrix([[float(i + 1)]])
        # Nonnegative beta box, then two SOCs, then local startup SOC.
        nonnegative = sparse.csc_matrix([[-1.], [1.]])
        q = sparse.csc_matrix([[0.], [-2.]])
        v = sparse.csc_matrix([[0.], [3.]])
        local, local_b = fit.global_startup_soc(M, np.array([i * .01]))
        A = sparse.vstack((nonnegative, q, v, local), format="csc")
        b = np.r_[0., 1., 4., 0., 4., 0., local_b]
        records.append({"leg": leg, "P": sparse.csc_matrix([[2. * (i + 1)]]),
            "A": A, "M": M, "linear": np.array([-.1 * (i + 1)]),
            "b": b, "dims": np.array([2, 2, 2, 2]),
            "kinds": ["NonnegativeConeT"] + ["SecondOrderConeT"] * 3,
            "motor_columns": np.array([i]), "startup_offset": np.array([i * .01]),
            "objective_constant": float(i + 1),
            "command_limit": 1., "velocity_limit": 1.})
    return records


def test_sparse_block_order_global_norm_and_objective():
    locals_ = synthetic_locals()
    problem = fit.assemble_global(locals_, strict=False)
    assert problem["A"].shape == (6 * 6 + 7, 6)
    assert problem["cone_dimensions"].tolist() == [2, 2, 2] * 6 + [7]
    assert problem["cone_types"] == (
        ["NonnegativeConeT", "SecondOrderConeT", "SecondOrderConeT"] * 6
        + ["SecondOrderConeT"])
    assert np.array_equal(problem["motor_columns"], np.arange(6))
    assert problem["objective_constant"] == 21.
    x = np.array([.1, .2, .3, .4, .5, .6])
    residual = problem["startup_matrix"] @ x + problem["startup_offset"]
    assert np.linalg.norm(residual) / np.sqrt(6) == pytest.approx(
        np.sqrt(np.mean([((i + 1) * x[i] + i * .01)**2 for i in range(6)])))
    np.testing.assert_allclose(problem["A"][-7:, :] @ x,
                               np.r_[0., -np.asarray(problem["startup_matrix"] @ x) / np.sqrt(6)])
    np.testing.assert_allclose(problem["b"][-7:],
                               np.r_[.018 * fit.PAD, problem["startup_offset"] / np.sqrt(6)])
    symmetric = problem["P"] + problem["P"].T - sparse.diags(problem["P"].diagonal())
    objective = .5 * x @ symmetric @ x + problem["linear"] @ x + problem["objective_constant"]
    direct = sum((i + 1) * x[i]**2 - .1 * (i + 1) * x[i] + (i + 1)
                 for i in range(6))
    assert objective == pytest.approx(direct)
    # The local startup SOC is absent; its stale per-leg .018 bound is gone.
    assert all(np.max(np.abs(problem["b"][-7:] - .018 * fit.PAD)) > 0 for _ in [0])


def test_reject_duplicate_motor_column_and_misordered_leg():
    locals_ = synthetic_locals()
    locals_[1]["motor_columns"] = locals_[0]["motor_columns"].copy()
    with pytest.raises(ValueError, match="overlap"):
        fit.assemble_global(locals_, strict=False)
    locals_ = synthetic_locals()
    locals_[0], locals_[1] = locals_[1], locals_[0]
    with pytest.raises(ValueError, match="six-leg order"):
        fit.assemble_global(locals_, strict=False)


def test_gate_rejects_before_input_reads_or_solver(tmp_path):
    freeze = Path("verification/banc-natural-gait-m3-20260919/"
                  "e211-global-startup-fit-freeze.json")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema": fit.REVIEW_SCHEMA, "approved": False,
        "blocking_defects": ["synthetic"], "freeze_sha256": fit.FREEZE_SHA,
        "source_sha256": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen SHA"):
        fit.verify_gate(freeze, "0" * 64, review, fit.file_sha(review))
    # A rejected review never authorizes an original solver call.
    with pytest.raises(ValueError, match="source review"):
        fit.verify_gate(freeze, fit.FREEZE_SHA, review, fit.file_sha(review))


def test_mock_non_solved_raw_saved_and_no_candidate(tmp_path):
    problem = fit.assemble_global(synthetic_locals(), strict=False)

    class FakeSolution:
        x = np.arange(6, dtype=float) / 10
        s = np.ones(problem["A"].shape[0])
        z = np.zeros(problem["A"].shape[0])
        status = "PrimalInfeasible"
        iterations = 12
        obj_val = float("nan")
        obj_val_dual = float("nan")
        r_prim = .1
        r_dual = .1
        solve_time = .001

    class FakeSolver:
        def __init__(self, *_args):
            pass

        def solve(self):
            return FakeSolution()

    ledger = {"solve_requested": False, "constructor_entered": False,
        "constructor_completed": False, "solve_called": False,
        "solver_returned": False, "solver_calls": 0, "solver_wall_s": 0.}
    raw, info = fit.solve_once(problem, tmp_path, ledger, elapsed=lambda: 0.,
                               solver_factory=FakeSolver)
    assessment = fit.assess_solution(problem, raw, info, synthetic_locals())
    assert not assessment["eligible_beta"]
    assert ledger["solver_calls"] == 1 and ledger["solver_returned"]
    with np.load(tmp_path / "solver-raw.npz") as saved:
        np.testing.assert_array_equal(saved["solver_x"], FakeSolution.x)
    assert (tmp_path / "solver-info.json").is_file()


def test_physical_limit_ignores_only_padding_roundoff():
    locals_ = synthetic_locals()
    within_original = .018 * fit.PAD + 1e-12
    for local in locals_:
        local["startup_offset"] = np.array([within_original])
    problem = fit.assemble_global(locals_, strict=False)
    x = np.zeros(6)
    raw = {"solver_x": x, "solver_s": problem["b"].copy(),
           "solver_z": np.zeros(len(problem["b"]))}
    info = {"status": "Solved", "iterations": 1, "objective_primal": 0.,
        "objective_dual": 0., "primal_residual": 0., "dual_residual": 0.,
        "solve_time_s": .001}
    report = fit.assess_solution(problem, raw, info, locals_)
    assert report["direct_training_pass"]
    assert report["padding_residual"]["startup_q_rms_above_padded"] > 0
    legs = [{"q_rms": fit.PAD + 1e-12,
             "velocity_rms": fit.PAD + 1e-12} for _ in locals_]
    diagnostics = fit.physical_limit_diagnostics(legs, locals_)
    assert all(diagnostics["leg_q_original_limit_pass"])
    assert all(diagnostics["leg_velocity_original_limit_pass"])
    assert all(value > 0 for value in diagnostics["leg_q_padding_residual"])
    legs[0]["q_rms"] = 1. + 1e-8
    assert not fit.physical_limit_diagnostics(
        legs, locals_)["leg_q_original_limit_pass"][0]
    for local in locals_:
        local["startup_offset"] = np.array([.018 + 1e-8])
    problem = fit.assemble_global(locals_, strict=False)
    raw["solver_s"] = problem["b"].copy()
    report = fit.assess_solution(problem, raw, info, locals_)
    assert not report["direct_training_pass"]


def test_soft_limit_status_retains_raw_without_candidate(tmp_path):
    problem = fit.assemble_global(synthetic_locals(), strict=False)

    class FakeSolution:
        x = np.zeros(6)
        s = np.zeros(problem["A"].shape[0])
        z = np.zeros(problem["A"].shape[0])
        status = "MaxTime"
        iterations = 150
        obj_val = .1
        obj_val_dual = .1
        r_prim = .01
        r_dual = .01
        solve_time = 180.25

    class FakeSolver:
        def __init__(self, *_args):
            pass

        def solve(self):
            return FakeSolution()

    ledger = {"solve_requested": False, "constructor_entered": False,
        "constructor_completed": False, "solve_called": False,
        "solver_returned": False, "solver_calls": 0, "solver_wall_s": 0.}
    raw, info = fit.solve_once(problem, tmp_path, ledger, elapsed=lambda: 0.,
                               solver_factory=FakeSolver)
    assert fit.soft_limit_overrun(ledger["solver_wall_s"], info["solve_time_s"]) == (
        pytest.approx(.25))
    assert ledger["solver_soft_limit_overrun_s"] == pytest.approx(.25)
    assert not fit.assess_solution(
        problem, raw, info, synthetic_locals())["eligible_beta"]
    assert (tmp_path / "solver-raw.npz").exists()


def test_shape_schema_failure_after_raw_storage_and_one_call_guard(tmp_path):
    problem = fit.assemble_global(synthetic_locals(), strict=False)

    class FakeSolution:
        x = np.zeros(6)
        s = np.zeros(2)  # saved malformed return
        z = np.zeros(problem["A"].shape[0])
        status = "Solved"
        iterations = 1
        obj_val = 0.
        obj_val_dual = 0.
        r_prim = 0.
        r_dual = 0.
        solve_time = .001

    calls = []

    class FakeSolver:
        def __init__(self, *_args):
            pass

        def solve(self):
            calls.append("solve")
            return FakeSolution()

    ledger = {"solve_requested": False, "constructor_entered": False,
        "constructor_completed": False, "solve_called": False,
        "solver_returned": False, "solver_calls": 0, "solver_wall_s": 0.}
    with pytest.raises(ValueError, match="schema shape mismatch: solver_s"):
        fit.solve_once(problem, tmp_path, ledger, elapsed=lambda: 0.,
                       solver_factory=FakeSolver)
    with np.load(tmp_path / "solver-raw.npz") as saved:
        assert saved["solver_s"].shape == (2,)
    assert (tmp_path / "solver-info.json").exists()
    with pytest.raises(RuntimeError, match="already called"):
        fit.solve_once(problem, tmp_path, ledger, elapsed=lambda: 0.,
                       solver_factory=FakeSolver)
    assert calls == ["solve"] and ledger["solver_calls"] == 1


def test_constructor_deadline_blocks_actual_solve(tmp_path):
    problem = fit.assemble_global(synthetic_locals(), strict=False)
    calls = []
    timeline = iter([0., 61.])

    class FakeSolver:
        def __init__(self, *_args):
            calls.append("construct")

        def solve(self):
            calls.append("solve")

    ledger = {"solve_requested": False, "constructor_entered": False,
        "constructor_completed": False, "solve_called": False,
        "solver_returned": False, "solver_calls": 0, "solver_wall_s": 0.}
    with pytest.raises(TimeoutError, match="allowance"):
        fit.solve_once(problem, tmp_path, ledger, elapsed=lambda: next(timeline),
                       solver_factory=FakeSolver)
    assert calls == ["construct"]
    assert ledger["solve_requested"] and ledger["constructor_completed"]
    assert not ledger["solve_called"] and ledger["solver_calls"] == 0


def test_mock_raw_save_failure_is_invalid_without_second_call(tmp_path, monkeypatch):
    problem = fit.assemble_global(synthetic_locals(), strict=False)
    calls = []

    class FakeSolution:
        x = np.zeros(6)
        s = np.zeros(problem["A"].shape[0])
        z = np.zeros(problem["A"].shape[0])

    class FakeSolver:
        def __init__(self, *_args):
            pass

        def solve(self):
            calls.append("solve")
            return FakeSolution()

    def fail_save(*_args, **_kwargs):
        raise OSError("synthetic raw storage failure")

    monkeypatch.setattr(fit.np, "savez", fail_save)
    ledger = {"solve_requested": False, "constructor_entered": False,
        "constructor_completed": False, "solve_called": False,
        "solver_returned": False, "solver_calls": 0, "solver_wall_s": 0.}
    with pytest.raises(OSError, match="raw storage"):
        fit.solve_once(problem, tmp_path, ledger, elapsed=lambda: 0.,
                       solver_factory=FakeSolver)
    assert calls == ["solve"] and ledger["solver_calls"] == 1
    assert not (tmp_path / "solver-info.json").exists()


def test_generated_clarabel_single_global_analytic_projection():
    """DEFERRED: exactly one real generated solve after parent FIRST/SECOND."""
    center = np.array([.8, .2])
    target = np.array([.1, .1])
    P = sparse.csc_matrix(2 * np.eye(2))
    linear = -2 * target
    box = sparse.vstack((-sparse.eye(2), sparse.eye(2)), format="csc")
    soc, rhs = fit.global_startup_soc(sparse.eye(2, format="csc"), -center,
                                      limit=.25 / fit.PAD)
    A = sparse.vstack((box, soc), format="csc")
    b = np.r_[np.zeros(2), np.ones(2), rhs]
    settings = fit.solver_settings()
    solver = clarabel.DefaultSolver(P, linear, A, b, [
        clarabel.NonnegativeConeT(4), clarabel.SecondOrderConeT(3)], settings)
    solution = solver.solve()
    x = np.asarray(solution.x)
    objective = float(np.sum((x - target) ** 2))
    constraint = float(np.sum((x - center) ** 2) - 2 * .25 ** 2)
    squared_error = float(np.sum((x - [.45, .15]) ** 2))
    print("E211_GENERATED_SOLVER_JSON", json.dumps({"calls_this_check": 1,
        "generated_calls_including_initial_failed_check": 2,
        "status": str(solution.status), "x": list(solution.x),
        "s": list(solution.s), "z": list(solution.z),
        "objective_primal_without_constant": solution.obj_val,
        "objective_dual_without_constant": solution.obj_val_dual,
        "primal_residual": solution.r_prim, "dual_residual": solution.r_dual,
        "f": objective, "g": constraint, "squared_position_error": squared_error,
        "analytic_x": [.45, .15], "analytic_f": .125,
        "tolerance_objective_constraint": 1e-8,
        "global_startup_rms": float(np.linalg.norm(x - center) / np.sqrt(2)),
        "solve_time_s": float(solution.solve_time), "cuda_calls": 0,
        "full_banc_calls": 0, "physics_steps": 0}))
    assert str(solution.status) == "Solved"
    assert np.all(x >= 0) and np.all(x <= 1)
    assert abs(objective - .125) <= 1e-8
    assert abs(constraint) <= 1e-8
    assert abs(objective + constraint - .125 - 2 * squared_error) <= 1e-14
    assert squared_error <= 1e-8
    assert abs(x[0] - center[0]) > .25


def test_budget_block_is_preserved_without_code_failure(tmp_path, monkeypatch):
    def budget_unavailable(*_args, **_kwargs):
        raise fit.SolverAllowanceUnavailable("synthetic pre-solve allowance exhausted")

    monkeypatch.setattr(fit, "verify_gate", budget_unavailable)
    out = tmp_path / "blocked"
    result = fit.run("unused", "unused", "unused", "unused", out)
    assert result["validity"] == "valid_bounded_unresolved_no_solver_call"
    assert result["solver_calls"] == 0 and result["raw_errors"] == []
    assert "primary_error" not in result
    ledger = json.loads((out / "ledger.json").read_text("utf-8"))
    assert ledger["blocked_before_solve"] and not ledger["solve_called"]
