"""Small generated CPU/Clarabel cases only; no original/GPU/body/900-var run."""

import ast
import copy
import inspect
import json
import time
from types import SimpleNamespace

import clarabel
import numpy as np
import pytest

from tools import fit_c_banc_m3_conic_source as fit

SOLVER_COST = []


def counted_factory(*args):
    """Count actual tiny Clarabel constructor/solve separately."""
    began = time.perf_counter()
    solver = clarabel.DefaultSolver(*args)
    row = {"constructor_requests": 1, "constructor_completed": 1,
           "constructor_wall_s": time.perf_counter()-began, "solve_requests": 0,
           "solve_completed": 0, "iterations": 0, "solve_wall_s": 0.}
    SOLVER_COST.append(row)

    class Counted:
        def solve(self):
            row["solve_requests"] += 1
            started = time.perf_counter()
            try:
                result = solver.solve()
                row["solve_completed"] += 1
                row["iterations"] += int(result.iterations)
                return result
            finally:
                row["solve_wall_s"] += time.perf_counter()-started
    return Counted()


@pytest.fixture(scope="module", autouse=True)
def report_small_solver_cost():
    yield
    print("generated_e275_clarabel_cost", {k: sum(row[k] for row in SOLVER_COST) for k in
        ("constructor_requests", "constructor_completed", "constructor_wall_s", "solve_requests",
         "solve_completed", "iterations", "solve_wall_s")})


def entry(*, gradient=None, qerr=None, qjac=None, lerr=None, ljac=None):
    c = fit.base.neutral(1)
    qe = np.zeros((99, 7)) if qerr is None else qerr
    le = np.zeros((99, 1)) if lerr is None else lerr
    jq = np.zeros((99, 7, 3)) if qjac is None else qjac
    jl = np.zeros((99, 1, 3)) if ljac is None else ljac
    h, j = fit.base.startup_constraints(qe, le, jq, jl, (0., 0.))
    return {"coefficients": c, "gradient": np.array([0., -1., 0.]) if gradient is None else gradient,
            "qerr": qe, "lerr": le, "startup_q_jacobian": jq, "startup_ltm_jacobian": jl,
            "constraints": h, "constraint_jacobian": j, "objective": 1., "mse": .01,
            "baseline_mse": .01, "valid": True, "evaluation": 0}


def test_small_actual_clarabel_soc_sign_all_peak_and_settings():
    jq = np.zeros((99, 7, 3))
    jq[0, 0, 1], jq[77, 0, 1] = .2, 2.
    qe = np.zeros((99, 7))
    qe[0, 0], qe[77, 0] = .03, .02
    e = entry(qerr=qe, qjac=jq)
    problem = fit.conic_problem(e, (0., 0.), .05)
    solved = fit.solve_direction(problem, 2., solver_factory=counted_factory)
    assert solved["valid"] and solved["receipt"]["status"] == "Solved"
    # The current maximum is t0, but t77 becomes limiting under this direction.
    assert 0 < solved["direction"][1] <= .8*(.04*(1-1e-4)-.02)/2+1e-8
    assert problem["nonnegative_rows"] == 2*3+2*1+2*(99*7+99)
    slack = problem["b"]-problem["A"]@solved["direction"]
    offset = problem["nonnegative_rows"]
    for key, jac_key, size, radius in zip(("qerr", "lerr"), ("startup_q_jacobian", "startup_ltm_jacobian"),
                                        (693, 99), problem["radii"], strict=True):
        assert slack[offset] == radius
        np.testing.assert_allclose(slack[offset+1:offset+1+size],
            problem[key].ravel()+problem[jac_key].reshape(size, 3)@solved["direction"])
        offset += size+1
    assert solved["residuals"]["minimum"] >= -1e-8
    assert solved["receipt"]["settings"] == {"verbose": False, "max_iter": 100,
        "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8, "tol_feas": 1e-8, "time_limit": 2., "max_threads": 1}


def test_soc_includes_squared_term_missed_by_first_order_sse():
    jq = np.zeros((99, 7, 3))
    jq[:, :, 1] = 2
    problem = fit.conic_problem(entry(qjac=jq), (1.3465, 0.), .05)
    d = np.array([0., .04, 0.])
    first_order_sse = problem["current_sse"][0]  # error=0: first-order SSE slope=0.
    assert first_order_sse < problem["norm_squared_limits"][0]
    assert fit.conic_residuals(problem, d)["soc"][0] < 0
    solved = fit.solve_direction(problem, 2., solver_factory=counted_factory)
    assert solved["valid"] and np.linalg.norm(jq.reshape(693, 3)@solved["direction"]) <= problem["radii"][0]+1e-8


def test_actual_neutral_zero_boundary_uses_existing_box_residue_clip():
    problem = fit.conic_problem(entry(gradient=np.array([0., 1., .5])), (0., 0.), .05)
    solved = fit.solve_direction(problem, 2., solver_factory=counted_factory)
    assert solved["valid"]
    raw = problem["coefficients"].ravel()+solved["direction"]
    clipped = np.clip(raw, 0, 8).reshape(1, 3)
    assert fit.base.coefficient_valid(clipped, interior=True)
    assert np.max(np.abs(raw[1:])) <= 1e-8
    print("generated_e275_neutral_boundary", {
        "direction": solved["direction"].tolist(), "raw_coefficient_valid": fit.base.coefficient_valid(raw.reshape(1, 3), interior=True),
        "clipped_coefficients": clipped.tolist(), "minimum_residual": solved["residuals"]["minimum"]})


def test_conic_and_canonical_mse_equal_for_c_f_and_strided_arrays():
    rng = np.random.default_rng(27520260927)
    qe = rng.uniform(-.01, .01, (99, 7))
    jq = rng.normal(size=(99, 7, 3))*.01
    values = []
    problems = []
    for order in ("C", "F"):
        q, j = np.array(qe, order=order), np.array(jq, order=order)
        values.append(fit.base.mean_square(q))
        problems.append(fit.conic_problem(entry(qerr=q, qjac=j), (.2, .001), .05))
    values.append(fit.base.mean_square(np.repeat(qe, 2, axis=1)[:, ::2]))
    assert values[0] == values[1] == values[2]
    for key in ("b", "q", "current_sse", "radii", "bounds"):
        np.testing.assert_array_equal(problems[0][key], problems[1][key])
    np.testing.assert_array_equal(problems[0]["A"].toarray(), problems[1]["A"].toarray())


@pytest.mark.parametrize("change", ["fixed", "peak", "dtype", "shape", "radius", "gradient"])
def test_invalid_conic_inputs_rejected_before_constructor(change):
    e, fixed, radius = entry(), (0., 0.), .05
    if change == "fixed":
        fixed = (-1., 0.)
    elif change == "peak":
        e["qerr"][98, 6] = .04
    elif change == "dtype":
        e["startup_q_jacobian"] = e["startup_q_jacobian"].astype(np.float32)
    elif change == "shape":
        e["startup_q_jacobian"] = e["startup_q_jacobian"][:-1]
    elif change == "radius":
        radius = 1.
    else:
        e["gradient"].fill(0)
    with pytest.raises(ValueError):
        fit.conic_problem(e, fixed, radius)


class GeneratedEvaluator:
    def __init__(self, output, *, nonlinear=False, budget=None):
        self.output, self.nonlinear = output, nonlinear
        self.budget = budget or fit.base.Budget()
        self.fixed_sse, self.baseline_mse = (0., 0.), .1**2
        self.evaluations, self.cache = [], {}

    def evaluate(self, x):
        key = x.tobytes()
        if key in self.cache:
            self.budget.cache_hits += 1
            self.budget.check()
            return self.cache[key]
        self.budget.check(new=True)
        self.budget.unique += 1
        self.budget.total_unique += 1
        qe, jq = np.zeros((99, 7)), np.zeros((99, 7, 3))
        qe[0, 0] = x[1]+(30*x[1]**2 if self.nonlinear else 0)
        jq[0, 0, 1] = 1+(60*x[1] if self.nonlinear else 0)
        e = entry(qerr=qe, qjac=jq)
        e["coefficients"] = x.reshape(-1, 3).copy()
        delta = x-fit.base.neutral(1).ravel()
        e["mse"] = float((x[1]-.1)**2)
        e["baseline_mse"] = self.baseline_mse
        e["objective"] = e["mse"]/self.baseline_mse+1e-4*np.mean(delta**2)
        e["gradient"] = 2e-4*delta/delta.size
        e["gradient"][1] += 2*(x[1]-.1)/self.baseline_mse
        e["valid"] = fit.base.coefficient_valid(e["coefficients"], interior=True) and np.all(e["constraints"] >= 0)
        e["evaluation"] = len(self.evaluations)
        self.evaluations.append(e)
        self.cache[key] = e
        return e


def fake_factory(direction=None, *, status="Solved", constructor_error=None, solve_error=None):
    def factory(P, q, A, b, cones, settings):
        if constructor_error is not None:
            raise constructor_error
        class Solver:
            def solve(self):
                if solve_error is not None:
                    raise solve_error
                d = np.array([0., .02, 0.]) if direction is None else direction
                return SimpleNamespace(status=status, x=d.tolist(), s=(b-A@d).tolist(), z=np.zeros_like(b).tolist(),
                                       iterations=3, solve_time=0., obj_val=0.)
        return Solver()
    return factory


def test_actual_nonlinear_backtracking_and_completed_strict_selection(tmp_path):
    now = [0.]
    evaluator = GeneratedEvaluator(tmp_path, nonlinear=True, budget=fit.base.Budget(clock=lambda: now[0]))
    evaluate = evaluator.evaluate
    def after_valid(x):
        actual = evaluate(x)
        if actual["evaluation"] > 0 and actual["valid"]:
            now[0] = 31.
        return actual
    evaluator.evaluate = after_valid
    selected = fit.fit_conic_leg(evaluator, fit.base.neutral(1), solver_factory=counted_factory)
    history = json.loads((tmp_path/"conic-history.json").read_text())
    assert history["proposals"][0]["alpha"] == 1 and history["proposals"][0]["minimum_actual_constraint"] < 0
    accepted = [p for p in history["proposals"] if p["status"] == "accepted"]
    assert accepted[0]["alpha"] == .5 and selected["objective"] < 1
    assert fit.previous.eligible(selected) and selected["evaluation"] == 2
    stats = evaluator.budget.e275_stats
    assert stats["socp_solve_requests"] == evaluator.budget.solver_calls == 1
    assert stats["accepted_steps"] == 1 and stats["lp_calls"] == stats["slsqp_calls"] == 0
    audit = np.load(tmp_path/"socp-000.npz")
    assert audit["startup_q_jacobian"].shape == (99, 7, 3)
    assert audit["solver_s"].shape[0] == 2386


@pytest.mark.parametrize("phase", ["constructor", "solve"])
def test_solver_primary_fault_cost_and_no_retry(tmp_path, phase):
    evaluator = GeneratedEvaluator(tmp_path)
    primary = RuntimeError("generated solver primary")
    factory = fake_factory(**{phase+"_error": primary})
    with pytest.raises(RuntimeError, match="solver primary") as failure:
        fit.fit_conic_leg(evaluator, fit.base.neutral(1), solver_factory=factory)
    assert failure.value is primary
    stats = evaluator.budget.e275_stats
    assert stats["socp_constructor_requests"] == 1
    assert stats["socp_solve_requests"] == (1 if phase == "solve" else 0)
    assert evaluator.budget.solver_calls == (1 if phase == "solve" else 0)
    assert stats["socp_"+phase+"_failed"] == 1
    assert primary.conic_receipt[phase]["failed"] == 1


@pytest.mark.parametrize("status", ["MaxTime", "AlmostSolved", "NumericalError"])
def test_only_solved_status_is_accepted_without_retry(tmp_path, status):
    evaluator = GeneratedEvaluator(tmp_path)
    selected = fit.fit_conic_leg(evaluator, fit.base.neutral(1), solver_factory=fake_factory(status=status))
    assert selected["status"] == "bounded_socp_failure" and selected["evaluation"] == 0
    assert evaluator.budget.solver_calls == 1 and evaluator.budget.iterations == 3


def test_independent_residual_rejects_solved_infeasible_direction(tmp_path):
    evaluator = GeneratedEvaluator(tmp_path)
    selected = fit.fit_conic_leg(evaluator, fit.base.neutral(1), solver_factory=fake_factory(np.array([0., .5, 0.])))
    assert selected["status"] == "bounded_socp_failure"
    row = json.loads((tmp_path/"conic-history.json").read_text())["socp"][0]
    assert row["minimum_residual"] < -1e-8 and row["solution_valid"] is False


def test_zero_gradient_ends_without_constructor_or_solve(tmp_path):
    evaluator = GeneratedEvaluator(tmp_path)
    evaluate = evaluator.evaluate
    def stationary(x):
        e = evaluate(x)
        e["gradient"].fill(0)
        return e
    evaluator.evaluate = stationary
    def forbidden(*args):
        pytest.fail("stationary point must not construct a solver")
    selected = fit.fit_conic_leg(evaluator, fit.base.neutral(1), solver_factory=forbidden)
    assert selected["status"] == "stationary" and evaluator.budget.solver_calls == 0


@pytest.mark.parametrize("scope", ["leg", "shared_fit", "unique_evaluations"])
@pytest.mark.parametrize("complete", [False, True])
def test_interrupted_evaluation_preserves_completed_vs_unfinished(tmp_path, complete, scope):
    evaluator = GeneratedEvaluator(tmp_path)
    evaluate = evaluator.evaluate
    primary = fit.base.BudgetExceeded(scope)
    attempts = []
    def late(x):
        attempts.append(x.copy())
        if len(attempts) == 1:
            return evaluate(x)
        if complete:
            evaluate(x)
        else:
            np.savez(tmp_path/"eval-001-startup.npz", q=np.zeros((101, 7)))
        raise primary
    evaluator.evaluate = late
    selected = fit.fit_conic_leg(evaluator, fit.base.neutral(1), solver_factory=fake_factory())
    history = json.loads((tmp_path/"conic-history.json").read_text())
    proposal = history["proposals"][0]
    assert proposal["evaluation_complete"] is complete
    assert proposal["adopted_as_iterate"] is False and len(attempts) == 2
    assert selected["evaluation"] == (1 if complete else 0)
    assert selected["budget_scope"] == scope
    assert evaluator.budget.e275_stats["accepted_steps"] == 0


@pytest.mark.parametrize("blocked", [None, "selected.npz", "selection.json", "partial-candidate.npz", "fit-pass.json"])
@pytest.mark.parametrize("complete", [False, True])
def test_original_expiry_preserves_first_exception_through_upper_loop(tmp_path, complete, blocked):
    now = [270.]
    deadline = fit.base.OriginalDeadline(0., clock=lambda: now[0])
    budget = fit.base.Budget(clock=lambda: now[0], deadline=deadline)
    folder = tmp_path/"LF"
    folder.mkdir()
    evaluator = GeneratedEvaluator(folder, budget=budget)
    evaluate = evaluator.evaluate
    primary = fit.base.BudgetExceeded("original")
    primary.add_note("generated original primary evidence")
    evaluations, legs = [], []
    actual_solver_count = len(SOLVER_COST)
    def late(x):
        evaluations.append(x.copy())
        if len(evaluations) == 1:
            return evaluate(x)
        if complete:
            evaluate(x)
        else:
            np.savez(folder/"eval-001-startup.npz", q=np.zeros((101, 7)))
        if blocked is not None:
            parent = folder if blocked in ("selected.npz", "selection.json") else tmp_path
            (parent/blocked).mkdir()
        now[0] = 291.
        raise primary
    evaluator.evaluate = late
    def run_leg(index):
        legs.append(index)
        return fit.fit_conic_leg(evaluator, fit.base.neutral(1), solver_factory=fake_factory())
    with pytest.raises(fit.base.BudgetExceeded) as caught:
        fit.base.run_fit_pass(run_leg, budget, [np.array([i]) for i in range(6)], tmp_path)
    assert caught.value is primary and primary.scope == "original"
    assert primary.__notes__[0] == "generated original primary evidence"
    assert legs == [0] and len(evaluations) == 2 and budget.solver_calls == 1
    assert len(SOLVER_COST) == actual_solver_count  # Only the generated stub direction ran.
    assert len(evaluator.evaluations) == (2 if complete else 1)
    history = json.loads((folder/"conic-history.json").read_text())
    proposal = history["proposals"][0]
    assert proposal["evaluation_complete"] is complete and proposal["adopted_as_iterate"] is False
    if blocked not in ("selected.npz", "selection.json"):
        saved = json.loads((folder/"selection.json").read_text())
        assert saved["evaluation"] == (1 if complete else 0) and saved["budget_scope"] == "original"
    if blocked in ("selected.npz", "selection.json"):
        assert primary.abort_run is True
        assert any("persistence failed" in note for note in primary.__notes__)
    elif blocked is not None:
        assert any("Partial candidate receipt failed" in note for note in primary.__notes__)
    else:
        assert primary.__notes__ == ["generated original primary evidence"]


@pytest.mark.parametrize("blocked", ["selected.npz", "selection.json"])
def test_late_budget_storage_failure_aborts_outer_pass_same_primary(tmp_path, blocked):
    folder = tmp_path/"LF"
    folder.mkdir()
    evaluator = GeneratedEvaluator(folder)
    evaluate = evaluator.evaluate
    primary = fit.base.BudgetExceeded("leg")
    def late(x):
        e = evaluate(x)
        if e["evaluation"] == 1:
            (folder/blocked).mkdir()
            raise primary
        return e
    evaluator.evaluate = late
    legs = []
    def run_leg(index):
        legs.append(index)
        return fit.fit_conic_leg(evaluator, fit.base.neutral(1), solver_factory=fake_factory())
    with pytest.raises(fit.base.BudgetExceeded) as failure:
        fit.base.run_fit_pass(run_leg, evaluator.budget, [np.array([i]) for i in range(6)], tmp_path)
    assert failure.value is primary and primary.abort_run is True and legs == [0]
    assert any("persistence failed" in note for note in primary.__notes__)


def test_time_cap_uses_minimum_remaining_budget():
    budget = fit.base.Budget(clock=lambda: 0.)
    budget.clock = lambda: 29.8
    budget.deadline = fit.base.OriginalDeadline(-260., clock=budget.clock)
    remaining, limit = fit.remaining_budget(budget)
    assert limit == pytest.approx(.2) and remaining["original_s"] == pytest.approx(.2)
    settings, actual = fit.solver_settings(limit)
    assert settings.max_threads == actual["max_threads"] == 1


def test_explicit_factory_and_original_defaults_are_preserved(tmp_path, monkeypatch):
    signature = inspect.signature(fit.base.execute_original)
    assert signature.parameters["evaluator_factory"].kind == inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["evaluator_factory"].default is fit.base.LegEvaluator
    assert inspect.signature(fit.base.LegEvaluator).parameters["retain_startup_jacobians"].default is False
    tree = ast.parse(inspect.getsource(fit.base.execute_original))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "evaluator_factory"]
    assert len(calls) == 1
    calls[0].args = [ast.Constant("generated argument")]
    expr = compile(ast.fix_missing_locations(ast.Expression(calls[0])), "factory-dispatch", "eval")
    assert eval(expr, {"evaluator_factory": lambda x: ("injected", x)}) == ("injected", "generated argument")
    captured = []
    def execute(*args, **kwargs):
        captured.append(kwargs)
        return {"generated": True}
    monkeypatch.setattr(fit.base, "execute_original", execute)
    assert fit.original_run({}, tmp_path, time.perf_counter()) == {"generated": True}
    factory = captured[0]["evaluator_factory"]
    assert factory.func is fit.base.LegEvaluator and factory.keywords == {"retain_startup_jacobians": True}
    assert captured[0]["leg_fit"] is fit.fit_conic_leg


class GeneratedEngine:
    def __init__(self, rows):
        self.problem = {"samples": rows}
        self.calls = {"solve": 0, "host_copy": 0, "jacobian": 0, "cotangent": 0}

    def solve(self, c, *, derivative, host):
        assert derivative and not host
        self.calls["solve"] += 1
        rows = self.problem["samples"]
        return {"q": np.full((rows, 7), c[0, 1]*.001), "ltm": np.zeros((rows, 1)), "receipt": {}}

    def host_copy(self, value, *, label):
        self.calls["host_copy"] += 1
        return value.copy()

    def command_jacobian(self, result, samples, *, host):
        assert host and np.array_equal(samples, np.arange(1, 100))
        self.calls["jacobian"] += 1
        jq = np.zeros((99, 7, 3))
        jq[:, :, 1] = .001
        return {"q": jq, "ltm": np.zeros((99, 1, 3))}

    def activation_cotangent_product(self, result, cotangent, *, host):
        assert host
        self.calls["cotangent"] += 1
        return np.array([0., .05*np.sum(cotangent), 0.])


def test_retaining_generated_startup_jacobians_adds_only_cache_memory(tmp_path):
    evaluators, calls, entries = [], [], []
    for retain in (False, True):
        folder = tmp_path/str(retain)
        folder.mkdir()
        startup, teacher = GeneratedEngine(101), GeneratedEngine(1201)
        evaluator = fit.base.LegEvaluator(startup, teacher,
            {"fixed": np.zeros(1, bool), "moment": np.ones((7, 1))},
            np.zeros((99, 7)), np.zeros((99, 1)), np.full((1201, 7), .002),
            (0., 0.), 4e-6, fit.base.Budget(), folder, retain_startup_jacobians=retain)
        point = np.array([[1., .2, .1]])
        actual = evaluator.evaluate(point.ravel())
        assert evaluator.evaluate(point.ravel()) is actual
        evaluators.append(evaluator)
        entries.append(actual)
        calls.append((startup.calls.copy(), teacher.calls.copy()))
    assert calls[0] == calls[1] == (
        {"solve": 1, "host_copy": 2, "jacobian": 1, "cotangent": 0},
        {"solve": 1, "host_copy": 2, "jacobian": 0, "cotangent": 1})
    for key in ("startup_q_jacobian", "startup_ltm_jacobian"):
        assert key not in entries[0]
    assert entries[1]["startup_q_jacobian"].shape == (99, 7, 3)
    assert entries[1]["startup_ltm_jacobian"].shape == (99, 1, 3)
    for key in entries[0]:
        np.testing.assert_equal(entries[0][key], entries[1][key])
    assert fit.base.array_bytes(evaluators[1].cache)-fit.base.array_bytes(evaluators[0].cache) == 99*8*3*8


def make_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(fit.base, "ROOT", tmp_path)
    sources, inputs = {}, {}
    for name in fit.SOURCES:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        sources[name] = fit.base.sha256(path)

    def file(alias, data):
        path = tmp_path/(alias+".json")
        path.write_text(json.dumps(data))
        inputs[alias] = {"path": path.name, "sha256": fit.base.sha256(path)}

    file("native", {"status": "pass", **{k: True for k in
        ("counters_valid", "raw_valid", "identity_valid", "decoder_valid")}})
    file("cuda", {"status": "pass"})
    original = {"schema": fit.base.SCHEMA, "contract": fit.base.CONTRACT, "inputs": dict(inputs),
        "approved_sources": {"historical source": "historical SHA"}, "contexts": {}, "targets": {},
        "legs": {}, "identity": {}, "native_validation": {"input": "native"}, "generated_cuda_gate": {"input": "cuda"}}
    file("original", original)
    fast = {"status": "pass", "exactness_valid": True, "speed_valid": True, "median_speedup": 1.5,
        "source_sha256": {k: sources[k] for k in fit.ENGINE_SOURCES}}
    file("fast", fast)
    freeze = {**original, "schema": fit.SCHEMA, "contract": copy.deepcopy(fit.CONTRACT), "inputs": inputs,
        "approved_sources": sources, "e270_original_freeze": {"input": "original"}, "generated_e273_gate": {"input": "fast"}}
    fp, rp = tmp_path/"freeze.json", tmp_path/"review.json"
    def save():
        fp.write_text(json.dumps(freeze))
        digest = fit.base.sha256(fp)
        rp.write_text(json.dumps({"schema": fit.REVIEW_SCHEMA, "verdict": "approved",
            "approved_sources": sources, "freeze_sha256": digest}))
        return fp, digest, rp, fit.base.sha256(rp)
    return freeze, save, file, fast


def test_eight_current_source_pins_preserve_historical_original_freeze(tmp_path, monkeypatch):
    freeze, save, _, _ = make_freeze(tmp_path, monkeypatch)
    assert fit.validate_freeze(*save()) == freeze
    assert len(fit.SOURCES) == len(freeze["approved_sources"]) == 8
    freeze["approved_sources"].pop(fit.SOURCES[-1])
    with pytest.raises(ValueError, match="Eight E275"):
        fit.validate_freeze(*save())


@pytest.mark.parametrize("change", ["exactness", "speed", "source", "median", "context", "input", "contract", "version"])
def test_freeze_preserves_original_inputs_and_conic_contract(tmp_path, monkeypatch, change):
    freeze, save, file, fast = make_freeze(tmp_path, monkeypatch)
    if change in ("exactness", "speed"):
        fast[change+"_valid"] = False
    elif change == "source":
        fast["source_sha256"][fit.ENGINE_SOURCES[0]] = "f"*64
    elif change == "median":
        fast["median_speedup"] = 1.49
    elif change == "context":
        freeze["contexts"] = {"changed": "source"}
    elif change == "input":
        file("native", {"status": "pass"})
    elif change == "contract":
        freeze["contract"]["conic_max_iter"] = 101
    else:
        freeze["contract"]["clarabel_version"] = "changed"
    if change in ("exactness", "speed", "source", "median"):
        file("fast", fast)
    with pytest.raises(ValueError):
        fit.validate_freeze(*save())


@pytest.mark.parametrize("after_solve", [False, True])
def test_required_direction_archive_fault_preserves_primary_and_cost(tmp_path, monkeypatch, after_solve):
    evaluator, primary = GeneratedEvaluator(tmp_path), OSError("generated conic archive fault")
    original = fit.save_direction
    def blocked(path, problem, solved):
        if (solved is not None) is after_solve:
            raise primary
        original(path, problem, solved)
    monkeypatch.setattr(fit, "save_direction", blocked)
    with pytest.raises(OSError) as caught:
        fit.fit_conic_leg(evaluator, fit.base.neutral(1), solver_factory=fake_factory())
    assert caught.value is primary
    assert evaluator.budget.solver_calls == int(after_solve)
    assert len(evaluator.evaluations) == 1
    assert json.loads((tmp_path/"selection.json").read_text())["evaluation"] == 0


def test_budget_expires_during_input_persistence_before_solver(tmp_path, monkeypatch):
    now = [0.]
    evaluator = GeneratedEvaluator(tmp_path, budget=fit.base.Budget(clock=lambda: now[0]))
    original = fit.save_direction
    def slow_save(*args):
        original(*args)
        now[0] = 31.
    monkeypatch.setattr(fit, "save_direction", slow_save)
    def forbidden(*args):
        pytest.fail("expired input persistence must not construct Clarabel")
    selected = fit.fit_conic_leg(evaluator, fit.base.neutral(1), solver_factory=forbidden)
    assert selected["budget_scope"] == "leg" and evaluator.budget.solver_calls == 0


def test_interrupted_metadata_storage_fault_aborts_same_primary(tmp_path, monkeypatch):
    evaluator, primary = GeneratedEvaluator(tmp_path), fit.base.BudgetExceeded("leg")
    evaluate = evaluator.evaluate
    def late(x):
        actual = evaluate(x)
        if actual["evaluation"] == 1:
            raise primary
        return actual
    def metadata_fault(*args):
        raise OSError("generated metadata fault")
    evaluator.evaluate = late
    monkeypatch.setattr(fit.previous, "interrupted_proposal", metadata_fault)
    with pytest.raises(fit.base.BudgetExceeded) as caught:
        fit.fit_conic_leg(evaluator, fit.base.neutral(1), solver_factory=fake_factory())
    assert caught.value is primary and primary.abort_run is True
    assert any("proposal metadata failed" in note for note in primary.__notes__)


def test_original_failure_receipt_fault_preserves_primary_no_original(tmp_path, monkeypatch):
    primary = fit.base.BudgetExceeded("leg")
    def forbidden_execute(*args, **kwargs):
        raise primary
    def blocked_receipt(*args):
        raise OSError("generated failure receipt fault")
    monkeypatch.setattr(fit.base, "execute_original", forbidden_execute)
    monkeypatch.setattr(fit.base, "failure_record", blocked_receipt)
    with pytest.raises(fit.base.BudgetExceeded) as caught:
        fit.original_run({}, tmp_path, time.perf_counter())
    assert caught.value is primary and primary.abort_run is True
    assert any("original failure receipt failed" in note for note in primary.__notes__)
