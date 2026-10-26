from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import fit_c_banc_m3_static_pool as fit


def tiny_layout() -> fit.LegLayout:
    return fit.LegLayout(
        leg="lf",
        leg_index=0,
        joint_columns=np.array([0, 1]),
        muscles=np.array([0, 1]),
        variable_muscles=np.array([0, 1]),
        variable_columns=np.array([0, 1, 2]),
        variable_column_muscles=np.array([0, 0, 1]),
        x0=np.array([0.4, 0.6, 1.0]),
        lower=np.array([5e-7, 5e-7, 1e-6]),
        row_constraint=np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )


def tiny_problem() -> fit.LegProblem:
    layout = tiny_layout()
    command_design = np.array([[[1.0, -0.5, 0.2]], [[0.3, 0.1, -0.2]]])
    command_offset = np.array([[0.1], [-0.2]])
    command_target = np.array([[0.0], [0.0]])
    velocity_design = np.array([[[0.2, -0.1, 0.3]]])
    velocity_offset = np.array([[0.0]])
    velocity_target = np.array([[0.1]])
    startup_design = np.array([[[0.01, -0.01, 0.0]], [[0.0, 0.01, -0.01]]])
    startup_offset = -np.einsum("tjn,n->tj", startup_design, layout.x0)
    startup_target = np.zeros((2, 1))
    objective = fit.build_quadratic_objective(
        command_design,
        command_offset,
        command_target,
        velocity_design,
        velocity_offset,
        velocity_target,
        layout.x0,
        1.0,
        1.0,
    )
    return fit.LegProblem(
        layout,
        command_design,
        command_offset,
        command_target,
        velocity_design,
        velocity_offset,
        velocity_target,
        startup_design,
        startup_offset,
        startup_target,
        objective,
        0.01,
        0.04 - 1e-9,
        0.04,
        1e-12,
        1e-12,
    )


def test_affine_decomposition_and_velocity_pair_end_minus_two_indexing():
    rng = np.random.default_rng(12)
    layout = tiny_layout()
    b = rng.normal(size=(6, 3))
    moment = np.array([[2.0, -1.0], [0.5, 3.0]])
    fixed = rng.normal(size=(6, 2))
    # Spell the direct generated truth separately from the implementation expression.
    baseline = np.empty((6, 2))
    for t in range(6):
        for j in range(2):
            baseline[t, j] = fixed[t, j] + sum(
                fit.COMMAND_GAIN
                * b[t, n]
                * moment[j, layout.variable_column_muscles[n]]
                * layout.x0[n]
                for n in range(3)
            )
    design, offset = fit.build_affine_design(b, baseline, moment, layout, np.arange(6))
    x = np.array([0.25, 0.75, 1.0])
    actual = np.einsum("tjn,n->tj", design, x) + offset
    expected = fixed.copy()
    for t in range(6):
        for j in range(2):
            expected[t, j] += sum(
                fit.COMMAND_GAIN
                * b[t, n]
                * moment[j, layout.variable_column_muscles[n]]
                * x[n]
                for n in range(3)
            )
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
    velocity_design, velocity_offset = fit.velocity_affine_design(
        design, offset, [1, 4]
    )
    actual_velocity = np.einsum("tjn,n->tj", velocity_design, x) + velocity_offset
    np.testing.assert_allclose(
        actual_velocity,
        np.stack(
            ((expected[2] - expected[1]) / fit.DT, (expected[5] - expected[4]) / fit.DT)
        ),
        atol=5e-14,
        rtol=0,
    )


def test_quadratic_and_startup_rms_analytic_derivatives():
    rng = np.random.default_rng(32)
    command_design = rng.normal(size=(5, 2, 3))
    velocity_design = rng.normal(size=(4, 2, 3))
    command_offset = rng.normal(size=(5, 2))
    velocity_offset = rng.normal(size=(4, 2))
    command_target = rng.normal(size=(5, 2))
    velocity_target = rng.normal(size=(4, 2))
    x0 = np.array([0.2, 0.3, 0.5])
    model = fit.build_quadratic_objective(
        command_design,
        command_offset,
        command_target,
        velocity_design,
        velocity_offset,
        velocity_target,
        x0,
        0.7,
        1.2,
    )
    x = np.array([0.25, 0.27, 0.48])
    value, gradient = fit.quadratic_value_grad(model, x)
    eps = 1e-6
    numerical = np.empty(3)
    for index in range(3):
        step = np.zeros(3)
        step[index] = eps
        numerical[index] = (
            fit.quadratic_value_grad(model, x + step)[0]
            - fit.quadratic_value_grad(model, x - step)[0]
        ) / (2 * eps)
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-8)

    rms_value, rms_jac = fit.startup_rms_constraint_value_jac(
        command_design, command_offset, command_target, x, 0.9
    )
    numerical_rms = np.empty(3)
    for index in range(3):
        step = np.zeros(3)
        step[index] = eps
        numerical_rms[index] = (
            fit.startup_rms_constraint_value_jac(
                command_design, command_offset, command_target, x + step, 0.9
            )[0]
            - fit.startup_rms_constraint_value_jac(
                command_design, command_offset, command_target, x - step, 0.9
            )[0]
        ) / (2 * eps)
    assert np.isfinite(rms_value)
    np.testing.assert_allclose(rms_jac, numerical_rms, atol=2e-9, rtol=2e-8)


def test_proportional_startup_budgets_compose_to_strict_global_rms():
    rng = np.random.default_rng(4)
    error = rng.normal(size=(99, 42))
    strict = 0.018 - 1e-9
    budgets = fit.proportional_startup_budgets(error, strict)
    baseline_by_leg = np.asarray(
        [np.sqrt(np.mean(np.square(error[:, 7 * i : 7 * (i + 1)]))) for i in range(6)]
    )
    ratios = np.divide(
        budgets, baseline_by_leg, out=np.zeros(6), where=baseline_by_leg > 0
    )
    np.testing.assert_allclose(ratios, np.full(6, strict / np.sqrt(np.mean(error**2))))
    assert np.sqrt(np.mean(np.square(budgets))) <= strict + 1e-15
    np.testing.assert_array_equal(
        fit.proportional_startup_budgets(np.zeros((99, 42)), strict), 0
    )


def generated_full_pool():
    membership = np.arange(391) % 96
    pool = np.zeros((96, 391))
    for muscle in range(96):
        columns = np.flatnonzero(membership == muscle)
        pool[muscle, columns] = 1 / columns.size
    ltm = np.array([15, 31, 47, 63, 79, 95])
    anatomy = {
        "muscles": [
            {
                "leg": fit.LEGS[muscle // 16],
                "columns": np.flatnonzero(membership == muscle).tolist(),
            }
            for muscle in range(96)
        ]
    }
    moment = np.zeros((42, 96))
    for muscle in range(96):
        moment[7 * (muscle // 16), muscle] = 1
    return pool, membership, ltm, anatomy, moment


def test_layout_preserves_positive_support_and_fixed_ltm_and_singletons():
    pool, membership, ltm, anatomy, moment = generated_full_pool()
    # Make muscle zero a singleton while retaining a valid full support partition.
    zero_columns = np.flatnonzero(membership == 0)
    for column in zero_columns[1:]:
        membership[column] = 1
    pool[:] = 0
    for muscle in range(96):
        columns = np.flatnonzero(membership == muscle)
        pool[muscle, columns] = 1 / columns.size
        anatomy["muscles"][muscle]["columns"] = columns.tolist()
    layout = fit.build_leg_layout("lf", pool, membership, ltm, anatomy, moment)
    assert 0 not in layout.variable_muscles
    assert 15 not in layout.variable_muscles
    candidate_x = layout.x0.copy()
    for row in range(layout.variable_muscles.size):
        local = np.flatnonzero(layout.row_constraint[row])
        candidate_x[local] = np.linspace(1, 2, local.size)
        candidate_x[local] /= candidate_x[local].sum()
    candidate = fit.apply_selected_weights(
        pool,
        [(layout, candidate_x)]
        + [
            (
                fit.LegLayout(
                    leg,
                    i,
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([]),
                    np.array([]),
                    np.empty((0, 0)),
                ),
                np.array([]),
            )
            for i, leg in enumerate(fit.LEGS[1:], 1)
        ],
    )
    assert candidate is not None
    np.testing.assert_array_equal(candidate[0], pool[0])
    np.testing.assert_array_equal(candidate[15], pool[15])
    fit.validate_pool_contract(candidate, membership, ltm)


def test_strict_selector_uses_actual_vector_without_clipping_or_normalization():
    problem = tiny_problem()
    good = fit.evaluate_leg_vector(problem, problem.layout.x0)
    tied = fit.evaluate_leg_vector(problem, problem.layout.x0.copy())
    tied["objective"] = good["objective"]
    bad = dict(good)
    bad["x"] = good["x"].copy()
    bad["x"][0] = 0.0
    bad["feasible"] = False
    bad["objective"] = good["objective"] - 100
    assert fit.select_earliest_feasible([bad, good, tied]) == 1
    selected = fit.apply_selected_weights(
        np.eye(3),
        [(problem.layout, np.array([0.2, 0.8, 1.0]))]
        + [
            (
                fit.LegLayout(
                    leg,
                    i,
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([]),
                    np.array([]),
                    np.empty((0, 0)),
                ),
                np.array([]),
            )
            for i, leg in enumerate(fit.LEGS[1:], 1)
        ],
    )
    np.testing.assert_array_equal(selected[[0, 0, 1], [0, 1, 2]], [0.2, 0.8, 1.0])


def test_selector_uses_physical_point04_while_optimizer_keeps_margin():
    problem = tiny_problem()
    near_limit = 0.04 - 0.5e-9
    shifted = replace(
        problem,
        startup_design=np.zeros_like(problem.startup_design),
        startup_offset=np.full_like(problem.startup_offset, near_limit),
        startup_rms_budget=0.05,
    )
    evaluated = fit.evaluate_leg_vector(shifted, shifted.layout.x0)
    assert evaluated["feasible"] is True
    assert np.min(evaluated["startup_linear_constraints"]) < 0


def test_generated_slsqp_is_single_start_and_selects_only_evaluated_points():
    problem = tiny_problem()
    frozen = {
        "max_calls_per_leg": 140,
        "max_total_calls": 840,
        "solver_options": {"maxiter": 100, "ftol": 1e-10, "disp": False},
    }
    ledger, result = fit.solve_leg(problem, frozen, time.perf_counter() + 5, [0])
    assert result["returned"] is True
    assert result["unique_evaluations"] <= 140
    assert result["objective_calls"] <= 140
    assert ledger.records[0]["origin"] == "baseline"
    assert result["selected_index"] == fit.select_earliest_feasible(ledger.records)


def test_evaluation_ledger_enforces_unique_objective_and_deadline_caps():
    problem = tiny_problem()
    total = [0]
    ledger = fit.EvaluationLedger(
        problem,
        max_calls=1,
        deadline=time.perf_counter() + 5,
        total_counter=total,
        total_limit=1,
    )
    ledger.get(problem.layout.x0, origin="baseline")
    assert ledger.get(problem.layout.x0) is ledger.records[0]
    with pytest.raises(fit.EvaluationLimit, match="evaluation cap"):
        ledger.get(problem.layout.x0 + np.array([0.01, -0.01, 0]))
    np.testing.assert_array_equal(
        ledger.interrupted_attempts[-1]["x"],
        problem.layout.x0 + np.array([0.01, -0.01, 0]),
    )
    ledger.objective_calls = 1
    with pytest.raises(fit.EvaluationLimit, match="objective-call cap"):
        ledger.fun(problem.layout.x0)
    assert ledger.objective_calls_requested == 1
    expired = fit.EvaluationLedger(
        problem,
        max_calls=2,
        deadline=time.perf_counter() - 1,
        total_counter=[0],
        total_limit=2,
    )
    with pytest.raises(fit.EvaluationLimit, match="deadline"):
        expired.get(problem.layout.x0)


def test_cached_evaluation_is_rejected_after_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(fit.time, "perf_counter", lambda: clock[0])
    problem = tiny_problem()
    ledger = fit.EvaluationLedger(
        problem, max_calls=1, deadline=9.0, total_counter=[0], total_limit=1
    )
    ledger.get(problem.layout.x0, origin="baseline")
    clock[0] = 10.0
    with pytest.raises(fit.EvaluationLimit, match="deadline"):
        ledger.get(problem.layout.x0, origin="cached_after_deadline")
    assert ledger.interrupted_attempts[-1]["origin"] == "cached_after_deadline"


def test_unexpected_optimizer_error_preserves_active_history(monkeypatch):
    problem = tiny_problem()
    partial = {}
    progress = {"fit_calls": 0}

    def failed_minimize(fun, x0, **_kwargs):
        fun(x0)
        raise RuntimeError("generated optimizer failure")

    monkeypatch.setattr(fit, "minimize", failed_minimize)
    with pytest.raises(RuntimeError, match="generated optimizer failure"):
        fit.solve_leg(
            problem,
            {
                "max_calls_per_leg": 140,
                "max_total_calls": 840,
                "solver_options": {"maxiter": 100, "ftol": 1e-10, "disp": False},
            },
            time.perf_counter() + 5,
            [0],
            partial=partial,
            progress=progress,
        )
    assert partial["lf_evaluation_x"].shape[0] == 1
    assert partial["lf_optimizer_entered"]
    assert not partial["lf_optimizer_completed"]
    assert progress["fit_calls"] == 1


def test_baseline_evaluation_failure_preserves_attempt_without_optimizer_claim(
    monkeypatch,
):
    problem = tiny_problem()
    partial = {}
    progress = {"fit_calls": 0}

    def failed_evaluation(_problem, _x):
        raise ArithmeticError("generated analytic failure")

    monkeypatch.setattr(fit, "evaluate_leg_vector", failed_evaluation)
    with pytest.raises(ArithmeticError, match="generated analytic failure"):
        fit.solve_leg(
            problem,
            {"max_calls_per_leg": 140, "max_total_calls": 840},
            time.perf_counter() + 5,
            [0],
            partial=partial,
            progress=progress,
        )
    assert partial["lf_evaluation_x"].shape == (0, problem.layout.x0.size)
    np.testing.assert_array_equal(partial["lf_interrupted_x"][0], problem.layout.x0)
    assert partial["lf_baseline_attempted"]
    assert not partial["lf_baseline_completed"]
    assert not partial["lf_optimizer_entered"]
    assert progress["fit_calls"] == 0


def fake_saved_inputs():
    return {
        "b": np.empty((0, 0)),
        "pool": np.empty((0, 0)),
        "membership": np.empty(0, dtype=np.int64),
        "moment": np.empty((0, 0)),
        "rest": np.empty(0),
        "ltm": np.empty(0, dtype=np.int64),
        "baseline": np.zeros((99, 42)),
        "target_indices": np.empty(0, dtype=np.int64),
        "target": np.empty((0, 42)),
        "pair_start": np.empty(0, dtype=np.int64),
        "pair_end": np.empty(0, dtype=np.int64),
        "target_velocity": np.empty((0, 42)),
        "startup_target": np.zeros((99, 42)),
        "startup_tendon_target": np.zeros((99, 6)),
    }


def calculation_freeze():
    return {
        "startup_limits": {"q_RMS": 0.018},
        "startup_optimizer_margin": 1e-9,
        "total_fit_limit_s": 54,
        "leg_soft_limit_s": 9,
        "max_calls_per_leg": 140,
        "max_total_calls": 840,
        "solver_options": {"maxiter": 100, "ftol": 1e-10, "disp": False},
    }


def test_completed_leg_history_survives_next_leg_preparation_error(monkeypatch):
    saved = fake_saved_inputs()
    monkeypatch.setattr(fit, "validate_saved_inputs", lambda *_: saved)
    calls = {"count": 0}

    def problem_then_failure(*_args):
        calls["count"] += 1
        if calls["count"] == 2:
            raise ValueError("generated second leg preparation failure")
        return tiny_problem()

    monkeypatch.setattr(fit, "_problem_for_leg", problem_then_failure)
    partial = {}
    progress = {"timings_s": {}}
    with pytest.raises(ValueError, match="second leg preparation failure"):
        fit.calculate(
            {"e143": {}, "startup": {}}, calculation_freeze(), progress, partial
        )
    assert partial["lf_evaluation_x"].shape[0] >= 1
    assert int(partial["unique_evaluations"]) >= 1
    assert int(partial["fit_calls"]) == 1
    assert float(partial["fit_wall_s"]) >= 0
    assert partial["lm_preparation_attempted"]
    assert not partial["lm_optimizer_entered"]


def test_final_total_fit_overrun_preserves_all_completed_work(monkeypatch):
    saved = fake_saved_inputs()
    monkeypatch.setattr(fit, "validate_saved_inputs", lambda *_: saved)
    clock = [0.0]
    monkeypatch.setattr(fit.time, "perf_counter", lambda: clock[0])

    def prepared_problem(_saved, _data, _freeze, leg, _budget):
        clock[0] += 0.2
        problem = tiny_problem()
        layout = replace(problem.layout, leg=leg, leg_index=fit.LEGS.index(leg))
        return replace(problem, layout=layout)

    calls = {"count": 0}

    def completed_stub(problem, _freeze, _deadline, total, *, partial, progress):
        calls["count"] += 1
        progress["fit_calls"] += 1
        clock[0] += 8.9
        record = fit.evaluate_leg_vector(problem, problem.layout.x0)
        record.update(origin="baseline", elapsed_s=0.0)
        total[0] += 1
        ledger = SimpleNamespace(
            records=[record],
            interrupted_attempts=[],
            objective_calls=1,
            objective_calls_requested=1,
        )
        progress["leg_unique_evaluations"][problem.layout.leg] = 1
        progress["leg_objective_calls"][problem.layout.leg] = 1
        progress["leg_objective_calls_requested"][problem.layout.leg] = 1
        progress["leg_interrupted_attempts"][problem.layout.leg] = 0
        return ledger, {
            "returned": True,
            "success": True,
            "selected_index": 0,
            "objective_calls": 1,
            "objective_calls_requested": 1,
            "unique_evaluations": 1,
            "interrupted_attempts": 0,
        }

    monkeypatch.setattr(fit, "_problem_for_leg", prepared_problem)
    monkeypatch.setattr(fit, "solve_leg", completed_stub)
    partial = {}
    progress = {"timings_s": {}}
    with pytest.raises(
        TimeoutError, match="total fit limit reached after completed solve"
    ):
        fit.calculate(
            {"e143": {}, "startup": {}}, calculation_freeze(), progress, partial
        )
    assert calls["count"] == 6
    assert int(partial["fit_calls"]) == 6
    assert int(partial["unique_evaluations"]) == 6
    assert float(partial["fit_wall_s"]) > 54
    assert partial["lf_evaluation_x"].shape[0] == 1
    assert partial["rh_evaluation_x"].shape[0] == 1


def minimal_freeze() -> dict:
    return {
        "minimum_free_disk_bytes": 0,
        "internal_wall_limit_s": 70,
        "maximum_raw_bytes": 67108864,
    }


def test_run_persists_raw_before_outcome_assertions(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        fit, "reviewed_inputs", lambda *_: (minimal_freeze(), {"source_sha256": {}})
    )
    monkeypatch.setattr(fit, "load_inputs", lambda *_: {})
    monkeypatch.setattr(
        fit,
        "calculate",
        lambda *_: (
            {"evaluation_x": np.array([[1.0]])},
            {"legs": {}},
            {"saved": {}, "candidate_pool": None, "composite": {}},
        ),
    )
    original_save = fit.save_npz_new

    def recording_save(path, arrays):
        events.append(("raw", Path(path).name, sorted(arrays)))
        original_save(path, arrays)

    def assess(*_args):
        events.append(("assess",))
        return {"conditional_static_pool_proposal": False}

    monkeypatch.setattr(fit, "save_npz_new", recording_save)
    monkeypatch.setattr(fit, "assess_composite", assess)
    result = fit.run(Path("f"), "f", Path("r"), "r", tmp_path / "out")
    assert result["status"] == "pass"
    assert events[0][0] == "raw" and events[1][0] == "assess"


def test_primary_calculation_error_survives_late_stat_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fit, "reviewed_inputs", lambda *_: (minimal_freeze(), {"source_sha256": {}})
    )
    monkeypatch.setattr(fit, "load_inputs", lambda *_: {})

    def failed_calculation(_data, _freeze, _progress, partial):
        partial["started"] = np.array([1])
        raise ValueError("primary calculation")

    monkeypatch.setattr(fit, "calculate", failed_calculation)
    monkeypatch.setattr(
        fit, "output_bytes", lambda *_: (_ for _ in ()).throw(OSError("late stat"))
    )
    with pytest.raises(ValueError, match="primary calculation") as caught:
        fit.run(Path("f"), "f", Path("r"), "r", tmp_path / "out")
    assert any("late stat" in note for note in getattr(caught.value, "__notes__", []))
    summary = json.loads(
        (tmp_path / "out" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["failure"]["message"] == "primary calculation"


def test_late_resource_failure_invalidates_authoritative_success(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fit, "reviewed_inputs", lambda *_: (minimal_freeze(), {"source_sha256": {}})
    )
    monkeypatch.setattr(fit, "load_inputs", lambda *_: {})
    monkeypatch.setattr(
        fit,
        "calculate",
        lambda *_: (
            {"evaluation_x": np.array([[1.0]])},
            {"legs": {}},
            {"saved": {}, "candidate_pool": None, "composite": {}},
        ),
    )
    monkeypatch.setattr(
        fit, "assess_composite", lambda *_: {"conditional_static_pool_proposal": False}
    )
    calls = {"count": 0}

    def late_stat(_output):
        calls["count"] += 1
        if calls["count"] >= 3:
            raise OSError("post-summary stat")
        return 10

    monkeypatch.setattr(fit, "output_bytes", late_stat)
    with pytest.raises(OSError, match="post-summary stat"):
        fit.run(Path("f"), "f", Path("r"), "r", tmp_path / "out")
    summary = json.loads(
        (tmp_path / "out" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert (tmp_path / "out" / "invalidated-success-summary.json").exists()
