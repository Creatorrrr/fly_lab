"""Small generated CPU LP fixtures only; no original/conditional/GPU/body work."""

import ast
import inspect
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog

from tools import fit_c_banc_m3_feasible_source as fit

LP_COST = []


def test_neutral_mse_is_exact_across_array_memory_orders():
    rng = np.random.default_rng(27420260927)
    error = rng.normal(size=(1200, 7))
    arrays = [np.array(error, order="C"), np.array(error, order="F"),
              np.repeat(error, 2, axis=1)[:, ::2]]
    values = [fit.base.mean_square(x) for x in arrays]
    assert values[0] == values[1] == values[2]
    assert values[0] == float(np.mean(np.ascontiguousarray(error)**2))


def counted_lp(**kwargs):
    started = time.perf_counter()
    result = linprog(**kwargs)
    LP_COST.append(
        {
            "calls": 1,
            "iterations": int(result.nit),
            "wall_s": time.perf_counter() - started,
        }
    )
    return result


@pytest.fixture(scope="module", autouse=True)
def report_generated_lp_cost():
    yield
    print(
        "generated_e274_lp_cost",
        {
            "calls": len(LP_COST),
            "iterations": sum(row["iterations"] for row in LP_COST),
            "wall_s": sum(row["wall_s"] for row in LP_COST),
        },
    )


class GeneratedEvaluator:
    def __init__(self, output, *, nonlinear=False, budget=None):
        self.output = output
        self.budget = budget or fit.base.Budget()
        self.evaluations = []
        self.cache = {}
        self.baseline_mse = 0.125**2
        self.nonlinear = nonlinear

    def evaluate(self, x):
        key = x.tobytes()
        if key in self.cache:
            self.budget.cache_hits += 1
            self.budget.check()
            return self.cache[key]
        self.budget.check(new=True)
        self.budget.unique += 1
        self.budget.total_unique += 1
        c = x.reshape(1, 3).copy()
        error = x[1] - 0.125
        mse = float(error**2)
        gradient = np.array([0, 2 * error / self.baseline_mse, 0], np.float64)
        h = np.array([0.001 - x[1] ** 2]) if self.nonlinear else np.array([0.5 - x[1]])
        j = np.array([[0, -2 * x[1] if self.nonlinear else -1, 0]], np.float64)
        entry = {
            "coefficients": c,
            "gradient": gradient,
            "objective": mse / self.baseline_mse,
            "mse": mse,
            "baseline_mse": self.baseline_mse,
            "constraints": h,
            "constraint_jacobian": j,
            "valid": fit.base.coefficient_valid(c, interior=True) and np.all(h >= 0),
            "evaluation": len(self.evaluations),
        }
        self.evaluations.append(entry)
        self.cache[key] = entry
        return entry


def test_generated_feasible_lp_path(tmp_path):
    evaluator = GeneratedEvaluator(tmp_path)
    selected = fit.fit_feasible_leg(
        evaluator, fit.base.neutral(1), lp_solver=counted_lp
    )
    assert fit.eligible(selected) and selected["objective"] < 1e-10
    stats = evaluator.budget.e274_stats
    assert stats["leg_optimizer_calls"] == 1 and stats["slsqp_calls"] == 0
    assert 0 < stats["lp_calls"] <= 24 and stats["outer_steps"] <= 24
    assert stats["lp_calls"] == evaluator.budget.solver_calls
    history = json.loads((tmp_path / "feasible-history.json").read_text())
    assert all(row["time_limit_s"] <= 1 for row in history["lp"])
    assert all(row["minimum_linear_residual"] >= -1e-9 for row in history["lp"])
    assert all(row["options"]["maxiter"] == 2000 for row in history["lp"])
    assert all(
        row["options"]["primal_feasibility_tolerance"] == 1e-9 for row in history["lp"]
    )
    for proposal in history["proposals"]:
        if proposal["status"] == "accepted":
            assert proposal["minimum_actual_constraint"] >= 0
            assert proposal["directional_derivative"] < 0
            assert proposal["objective"] <= proposal["armijo_bound"]


def test_nonlinear_constraint_backtracking_keeps_rejected_evaluations(tmp_path):
    evaluator = GeneratedEvaluator(tmp_path, nonlinear=True)
    selected = fit.fit_feasible_leg(
        evaluator, fit.base.neutral(1), lp_solver=counted_lp
    )
    history = json.loads((tmp_path / "feasible-history.json").read_text())
    first = history["proposals"][0]
    assert first["alpha"] == 1 and first["minimum_actual_constraint"] < 0
    accepted = [p for p in history["proposals"] if p["status"] == "accepted"]
    assert accepted[0]["alpha"] == 0.5
    assert any(not e["valid"] for e in evaluator.evaluations)
    assert selected["constraints"][0] >= 0 and selected["objective"] < 1
    assert (
        selected["evaluation"]
        == min(
            (e for e in evaluator.evaluations if fit.eligible(e)),
            key=lambda e: e["objective"],
        )["evaluation"]
    )


def test_lp_formulation_keeps_global_outside_sse_and_boundary_coefficients():
    qerr = np.full((99, 7), 0.022)
    ltm = np.zeros((99, 1))
    qjac = np.zeros((99, 7, 3))
    qjac[:, :, 1] = 0.001
    ljac = np.zeros((99, 1, 3))
    h, j = fit.base.startup_constraints(qerr, ltm, qjac, ljac, (0.1, 0.002))
    assert h[0] > 0  # .022 leg RMS is permitted by the original 42-joint budget.
    c = np.array([[0.125 * (1 + 1e-6), 0, 0]], np.float64)
    entry = {
        "coefficients": c,
        "gradient": np.array([-1.0, -0.5, 0]),
        "constraints": h,
        "constraint_jacobian": j,
    }
    problem, _ = fit.lp_problem(entry, 0.05)
    np.testing.assert_array_equal(problem["A_ub"][: len(h)], -j)
    np.testing.assert_allclose(problem["b_ub"][: len(h)], 0.8 * h)
    np.testing.assert_array_equal(problem["bounds"][1:3, 0], [0, 0])
    d = np.array([0, 0.01, 0])
    assert np.min(fit.lp_residuals(problem, d)) >= 0
    bigger_h, _ = fit.base.startup_constraints(qerr, ltm, qjac, ljac, (1.1, 0.002))
    assert bigger_h[0] < 0


def test_lp_failure_is_bounded_without_retry(tmp_path):
    evaluator = GeneratedEvaluator(tmp_path)
    calls = []

    def failed(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            success=False, status=1, message="generated time limit", nit=3, x=None
        )

    selected = fit.fit_feasible_leg(evaluator, fit.base.neutral(1), lp_solver=failed)
    assert selected["status"] == "bounded_lp_failure" and len(calls) == 1
    assert len(evaluator.evaluations) == 1 and selected["evaluation"] == 0
    assert evaluator.budget.solver_calls == 1 and evaluator.budget.iterations == 3
    assert (tmp_path / "lp-000.npz").is_file()


def test_stationary_has_zero_lp_calls(tmp_path):
    evaluator = GeneratedEvaluator(tmp_path)
    initial = np.array([[1.0, 0.125, 0]], np.float64)

    def forbidden(**_):
        pytest.fail("zero gradient must not call LP")

    selected = fit.fit_feasible_leg(evaluator, initial, lp_solver=forbidden)
    assert selected["status"] == "stationary" and evaluator.budget.solver_calls == 0


@pytest.mark.parametrize(
    "scope", ["leg", "shared_fit", "original", "unique_evaluations"]
)
def test_budget_stop_preserves_completed_improvement(tmp_path, scope):
    now = [0.0]
    deadline = fit.base.OriginalDeadline(
        -200.0 if scope == "original" else 0.0, clock=lambda: now[0]
    )
    budget = fit.base.Budget(clock=lambda: now[0], deadline=deadline)
    evaluator = GeneratedEvaluator(tmp_path, budget=budget)
    original = evaluator.evaluate

    def late(x):
        entry = original(x)
        if entry["evaluation"] == 1:
            if scope == "unique_evaluations":
                budget.unique = 80
            else:
                now[0] = {"leg": 31.0, "shared_fit": 181.0, "original": 91.0}[scope]
            budget.check(new=scope == "unique_evaluations")
        return entry

    evaluator.evaluate = late

    def direction(**_):
        return SimpleNamespace(
            success=True,
            status=0,
            message="generated direction",
            nit=1,
            x=np.array([0, 0.05, 0], np.float64),
        )

    selected = fit.fit_feasible_leg(evaluator, fit.base.neutral(1), lp_solver=direction)
    assert selected["budget_scope"] == scope and selected["evaluation"] == 1
    assert selected["objective"] < 1 and fit.eligible(selected)
    assert evaluator.budget.solver_calls == 1
    assert json.loads((tmp_path / "selection.json").read_text())["evaluation"] == 1
    history = json.loads((tmp_path / "feasible-history.json").read_text())
    proposal = history["proposals"][0]
    assert proposal["status"] == "completed_budget_stop_before_iterate_adoption"
    assert proposal["evaluation"] == 1 and proposal["evaluation_complete"] is True
    assert proposal["budget_scope"] == scope and proposal["adopted_as_iterate"] is False
    assert proposal["objective"] == selected["objective"]
    assert proposal["mse"] == selected["mse"]
    assert proposal["actual_constraints"] == selected["constraints"].tolist()
    assert history["outer"][0]["evaluation"] == 0
    assert evaluator.budget.e274_stats["accepted_steps"] == 0


@pytest.mark.parametrize("blocked_name", ["selected.npz", "selection.json"])
def test_budget_stop_storage_failure_raises_same_primary(tmp_path, blocked_name):
    evaluator = GeneratedEvaluator(tmp_path)
    evaluate = evaluator.evaluate
    primary = fit.base.BudgetExceeded("leg")
    calls = []

    def late(x):
        entry = evaluate(x)
        if entry["evaluation"] == 1:
            (tmp_path / blocked_name).mkdir()
            raise primary
        return entry

    evaluator.evaluate = late

    def direction(**_):
        calls.append(1)
        return SimpleNamespace(
            success=True,
            status=0,
            message="generated direction",
            nit=0,
            x=np.array([0.0, 0.05, 0.0]),
        )

    with pytest.raises(fit.base.BudgetExceeded) as caught:
        fit.fit_feasible_leg(evaluator, fit.base.neutral(1), lp_solver=direction)
    assert caught.value is primary
    assert any("persistence failed" in note for note in primary.__notes__)
    assert len(calls) == 1 and len(evaluator.evaluations) == 2
    assert not (tmp_path / blocked_name).is_file()
    history = json.loads((tmp_path / "feasible-history.json").read_text())
    assert history["proposals"][0]["evaluation"] == 1
    assert (
        history["proposals"][0]["status"]
        == "completed_budget_stop_before_iterate_adoption"
    )


@pytest.mark.parametrize("startup_saved", [False, True])
def test_interrupted_proposal_without_cache_is_incomplete_and_not_reevaluated(
    tmp_path, startup_saved
):
    evaluator = GeneratedEvaluator(tmp_path)
    original = evaluator.evaluate
    calls = []
    primary = fit.base.BudgetExceeded("leg")

    def interrupted(x):
        calls.append(x.copy())
        if len(calls) == 1:
            return original(x)
        evaluator.budget.unique += 1
        evaluator.budget.total_unique += 1
        if startup_saved:
            np.savez(tmp_path / "eval-001-startup.npz", q=np.zeros((101, 7)))
        raise primary

    evaluator.evaluate = interrupted

    def direction(**_):
        return SimpleNamespace(
            success=True,
            status=0,
            message="generated direction",
            nit=0,
            x=np.array([0.0, 0.05, 0.0]),
        )

    selected = fit.fit_feasible_leg(evaluator, fit.base.neutral(1), lp_solver=direction)
    assert selected["evaluation"] == 0 and len(calls) == 2
    assert evaluator.budget.solver_calls == 1 and len(evaluator.evaluations) == 1
    history = json.loads((tmp_path / "feasible-history.json").read_text())
    proposal = history["proposals"][0]
    assert (
        proposal["status"] == "incomplete_budget_stop"
        and proposal["evaluation"] is None
    )
    assert (
        proposal["evaluation_complete"] is False
        and proposal["adopted_as_iterate"] is False
    )
    assert (
        proposal["budget_scope"] == "leg"
        and proposal["startup_raw_saved"] is startup_saved
    )
    assert proposal["teacher_raw_saved"] is False
    assert proposal["completion_source"] == (
        "startup_only_raw" if startup_saved else "no_completed_raw"
    )


@pytest.mark.parametrize("blocked_name", ["selected.npz", "selection.json"])
def test_budget_storage_error_aborts_entire_fit_pass(tmp_path, blocked_name):
    """A leg budget must not hide required storage failure at the outer boundary."""
    folder = tmp_path / "LF"
    folder.mkdir()
    evaluator = GeneratedEvaluator(folder)
    evaluate = evaluator.evaluate
    primary = fit.base.BudgetExceeded("leg")
    legs = []

    def late(x):
        entry = evaluate(x)
        if entry["evaluation"] == 1:
            (folder / blocked_name).mkdir()
            raise primary
        return entry

    evaluator.evaluate = late

    def direction(**_):
        return SimpleNamespace(success=True, status=0, message="generated direction",
                               nit=0, x=np.array([0.0, 0.05, 0.0]))

    def run_leg(index):
        legs.append(index)
        return fit.fit_feasible_leg(evaluator, fit.base.neutral(1), lp_solver=direction)

    with pytest.raises(fit.base.BudgetExceeded) as caught:
        fit.base.run_fit_pass(run_leg, evaluator.budget,
                              [np.array([i]) for i in range(6)], tmp_path)
    assert caught.value is primary and primary.abort_run is True
    assert legs == [0] and evaluator.budget.solver_calls == 1
    assert any("persistence failed" in note for note in primary.__notes__)
    assert not (tmp_path / "candidate.npz").exists()


def test_late_proposal_metadata_error_is_fatal(tmp_path, monkeypatch):
    evaluator = GeneratedEvaluator(tmp_path)
    evaluate = evaluator.evaluate
    primary = fit.base.BudgetExceeded("leg")

    def late(x):
        entry = evaluate(x)
        if entry["evaluation"] == 1:
            raise primary
        return entry

    def metadata_failure(*_):
        raise OSError("generated required metadata failure")

    evaluator.evaluate = late
    monkeypatch.setattr(fit, "interrupted_proposal", metadata_failure)
    with pytest.raises(fit.base.BudgetExceeded) as caught:
        fit.fit_feasible_leg(evaluator, fit.base.neutral(1), lp_solver=lambda **_: (
            SimpleNamespace(success=True, status=0, message="generated direction",
                            nit=0, x=np.array([0.0, 0.05, 0.0]))))
    assert caught.value is primary and primary.abort_run is True
    assert any("proposal metadata failed" in note for note in primary.__notes__)
    assert evaluator.budget.solver_calls == 1 and len(evaluator.evaluations) == 2


def test_lp_time_limit_is_minimum_remaining_budget(tmp_path):
    now = [29.8]
    budget = fit.base.Budget(clock=lambda: 0.0)
    budget.clock = lambda: now[0]
    budget.deadline = fit.base.OriginalDeadline(-260.0, clock=lambda: now[0])
    remaining, limit = fit.remaining_budget(budget)
    assert remaining["original_s"] == pytest.approx(0.2)
    assert limit == pytest.approx(0.2)
    assert remaining["leg_s"] == pytest.approx(0.2)


def test_calculation_error_survives_history_selection_storage_error(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("file, not output directory")
    evaluator = GeneratedEvaluator(blocked)
    primary = RuntimeError("generated LP calculation error")

    def explode(**_):
        raise primary

    with pytest.raises(RuntimeError, match="generated LP calculation") as caught:
        fit.fit_feasible_leg(evaluator, fit.base.neutral(1), lp_solver=explode)
    assert caught.value is primary
    assert any("persistence failed" in note for note in primary.__notes__)
    assert evaluator.budget.solver_calls == 1
    assert evaluator.budget.e274_stats["lp_wall_s"] >= 0


def test_invalid_neutral_never_calls_lp(tmp_path):
    evaluator = GeneratedEvaluator(tmp_path)
    initial = np.array([[0.125, 0, 0]], np.float64)

    def forbidden(**_):
        pytest.fail("invalid neutral must not call LP")

    with pytest.raises(ValueError, match="strict feasible"):
        fit.fit_feasible_leg(evaluator, initial, lp_solver=forbidden)
    assert evaluator.budget.solver_calls == 0


def test_execute_original_dispatch_is_keyword_only_and_preserves_default():
    signature = inspect.signature(fit.base.execute_original)
    assert signature.parameters["leg_fit"].kind == inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["leg_fit"].default is None
    tree = ast.parse(inspect.getsource(fit.base.execute_original))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.BoolOp)
        and isinstance(node.func.op, ast.Or)
    ]
    assert len(calls) == 1
    # Execute the actual dispatch expression in a small generated namespace.
    calls[0].args = [
        ast.Constant("generated evaluator"),
        ast.Constant("generated neutral"),
    ]
    expression = compile(
        ast.fix_missing_locations(ast.Expression(calls[0])),
        "generated-dispatch",
        "eval",
    )
    default = lambda *args: ("default", args)
    injected = lambda *args: ("feasible", args)
    assert eval(expression, {"leg_fit": None, "fit_leg": default})[0] == "default"
    assert eval(expression, {"leg_fit": injected, "fit_leg": default})[0] == "feasible"


def make_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(fit.base, "ROOT", tmp_path)
    sources = {}
    for name in fit.SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        sources[name] = fit.base.sha256(path)
    native = {
        "status": "pass",
        **{
            k: True
            for k in ("counters_valid", "raw_valid", "identity_valid", "decoder_valid")
        },
    }
    oldgate = {"status": "pass"}
    inputs = {}

    def file(alias, data):
        path = tmp_path / (alias + ".json")
        path.write_text(json.dumps(data))
        inputs[alias] = {"path": path.name, "sha256": fit.base.sha256(path)}

    file("native", native)
    file("cuda", oldgate)
    original = {
        "schema": fit.base.SCHEMA,
        "contract": fit.base.CONTRACT,
        "inputs": dict(inputs),
        "approved_sources": {"historical source": "historical SHA"},
        "contexts": {},
        "targets": {},
        "legs": {},
        "identity": {},
        "native_validation": {"input": "native"},
        "generated_cuda_gate": {"input": "cuda"},
    }
    file("original", original)
    fast = {
        "status": "pass",
        "exactness_valid": True,
        "speed_valid": True,
        "median_speedup": 1.5,
        "source_sha256": {k: sources[k] for k in fit.ENGINE_SOURCES},
    }
    file("fast", fast)
    freeze = {
        **original,
        "schema": fit.SCHEMA,
        "contract": fit.CONTRACT,
        "inputs": inputs,
        "approved_sources": sources,
        "e270_original_freeze": {"input": "original"},
        "generated_e273_gate": {"input": "fast"},
    }
    fp = tmp_path / "freeze.json"
    rp = tmp_path / "review.json"

    def save():
        fp.write_text(json.dumps(freeze))
        digest = fit.base.sha256(fp)
        rp.write_text(
            json.dumps(
                {
                    "schema": fit.REVIEW_SCHEMA,
                    "verdict": "approved",
                    "approved_sources": sources,
                    "freeze_sha256": digest,
                }
            )
        )
        return fp, digest, rp, fit.base.sha256(rp)

    return freeze, save, file, fast


def test_distinct_six_source_freeze_and_historical_pins(tmp_path, monkeypatch):
    freeze, save, _, _ = make_freeze(tmp_path, monkeypatch)
    assert fit.validate_freeze(*save()) == freeze
    freeze["approved_sources"].pop(fit.SOURCES[-1])
    with pytest.raises(ValueError, match="Six E274"):
        fit.validate_freeze(*save())


@pytest.mark.parametrize(
    "change", ["exactness", "speed", "source", "median", "context", "input"]
)
def test_freeze_requires_e273_exactness_speed_source_and_unchanged_inputs(
    tmp_path, monkeypatch, change
):
    freeze, save, file, fast = make_freeze(tmp_path, monkeypatch)
    if change == "exactness":
        fast["exactness_valid"] = False
    elif change == "speed":
        fast["speed_valid"] = False
    elif change == "source":
        fast["source_sha256"][fit.ENGINE_SOURCES[0]] = "f" * 64
    elif change == "median":
        fast["median_speedup"] = 1.49
    elif change == "context":
        freeze["contexts"] = {"changed": "source"}
    elif change == "input":
        file("native", {"status": "pass"})
    if change in ("exactness", "speed", "source", "median"):
        file("fast", fast)
    with pytest.raises(ValueError):
        fit.validate_freeze(*save())
