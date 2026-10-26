"""Small generated certified fitter cases; no original/GPU/900-variable run."""

import copy
import inspect
import json
import time

import clarabel
import numpy as np
import pytest

from tools import fit_c_banc_m3_certified_source as fit

ACTUAL_COST = []


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
        qe, le, jq, jl = np.zeros((99, 7)), np.zeros((99, 1)), np.zeros((99, 7, 3)), np.zeros((99, 1, 3))
        qe[0, 0] = x[1]+(30*x[1]**2 if self.nonlinear else 0)
        jq[0, 0, 1] = 1+(60*x[1] if self.nonlinear else 0)
        h, j = fit.base.startup_constraints(qe, le, jq, jl, (0., 0.))
        delta = x-fit.base.neutral(1).ravel()
        mse = float((x[1]-.1)**2)
        gradient = 2e-4*delta/delta.size
        gradient[1] += 2*(x[1]-.1)/self.baseline_mse
        entry = {"coefficients": x.reshape(-1, 3).copy(), "gradient": gradient,
            "qerr": qe, "lerr": le, "startup_q_jacobian": jq, "startup_ltm_jacobian": jl,
            "constraints": h, "constraint_jacobian": j,
            "mse": mse, "baseline_mse": self.baseline_mse, "objective": mse/self.baseline_mse+1e-4*np.mean(delta**2),
            "valid": fit.base.coefficient_valid(x.reshape(-1, 3), interior=True) and np.all(h >= 0),
            "evaluation": len(self.evaluations)}
        self.evaluations.append(entry)
        self.cache[key] = entry
        return entry


def counted_factory(*args):
    began = time.perf_counter()
    row = {"constructor_requested": 1, "constructor_completed": 0, "constructor_wall_s": None,
           "solve_requested": 0, "solve_completed": 0, "solve_wall_s": 0., "iterations": 0}
    ACTUAL_COST.append(row)
    try:
        solver = clarabel.DefaultSolver(*args)
        row["constructor_completed"] = 1
    finally:
        row["constructor_wall_s"] = time.perf_counter()-began
    class Solver:
        def solve(self):
            row["solve_requested"] += 1
            began = time.perf_counter()
            try:
                result = solver.solve()
                row["solve_completed"] += 1
                row["iterations"] += int(result.iterations)
                return result
            finally:
                row["solve_wall_s"] += time.perf_counter()-began
    return Solver()


@pytest.fixture(scope="module", autouse=True)
def actual_cost():
    yield
    print("generated_e277_fitter_actual_cost", {k: sum(row[k] for row in ACTUAL_COST) for k in
        ("constructor_requested", "constructor_completed", "constructor_wall_s", "solve_requested",
         "solve_completed", "solve_wall_s", "iterations")})


def stub_certified(direction, *, valid=True, status="AlmostSolved"):
    """Generated loop-isolation stub; never a real solver/certificate call."""
    def solve(problem, *args, **kwargs):
        phase = {"requested": 1, "completed": 1, "failed": 0, "wall_s": 0.}
        receipt = {"constructor": phase.copy(), "solve": phase.copy(), "iterations": 1,
            "iterations_known": True, "status": status, "solution_valid": valid,
            "certificate": {**phase, "wall_known": True}, "certificate_receipt": {"valid": valid, "generated_stub": True}}
        d = np.asarray(direction, dtype=np.float64)
        exact = problem["b"]-problem["A"]@d
        return {"direction": d, "s": exact-.001, "z": np.zeros_like(exact), "valid": valid,
            "original_valid": False, "original_receipt": {}, "certificate": receipt["certificate_receipt"],
            "primal_slack": exact, "residuals": fit.conic_residuals(problem, d), "receipt": receipt}
    return solve


def test_actual_small_nonlinear_backtracking_certifies_and_shrinks_radius(tmp_path):
    now = [0.]
    evaluator = GeneratedEvaluator(tmp_path, nonlinear=True, budget=fit.base.Budget(clock=lambda: now[0]))
    evaluate = evaluator.evaluate
    def stop_after_accepted(x):
        actual = evaluate(x)
        if actual["evaluation"] > 0 and actual["valid"]:
            now[0] = 31.
        return actual
    evaluator.evaluate = stop_after_accepted
    selected = fit.fit_certified_leg(evaluator, fit.base.neutral(1), solver_factory=counted_factory)
    history = json.loads((tmp_path/"conic-history.json").read_text())
    accepted = [p for p in history["proposals"] if p["status"] == "accepted"]
    assert accepted[0]["alpha"] == .5 and accepted[0]["radius"] == .05 and accepted[0]["next_radius"] == .025
    assert history["proposals"][0]["minimum_actual_constraint"] < 0
    assert selected["evaluation"] == 2 and fit.previous.eligible(selected)
    certificate = history["socp"][0]["certificate_receipt"]
    assert certificate["valid"] and all(certificate["checks"].values())
    audit = np.load(tmp_path/"socp-000.npz")
    np.testing.assert_array_equal(audit["primal_slack"],
        fit.conic_problem(evaluator.evaluations[0], evaluator.fixed_sse, .05)["b"]
        -fit.conic_problem(evaluator.evaluations[0], evaluator.fixed_sse, .05)["A"]@audit["direction"])
    stats = evaluator.budget.e277_stats
    assert stats["socp_solve_requests"] == stats["certificate_completed"] == stats["certified_solved"] == 1


@pytest.mark.parametrize("nonlinear, expected_alpha, expected_radius", [(False, 1., .1), (True, .5, .025)])
def test_accepted_radius_is_used_in_next_problem(tmp_path, monkeypatch, nonlinear, expected_alpha, expected_radius):
    evaluator = GeneratedEvaluator(tmp_path, nonlinear=nonlinear)
    radii = []
    stub = stub_certified([0., .032, 0.])
    def solve(problem, *args, **kwargs):
        radii.append(problem["radius"])
        if len(radii) == 2:
            raise fit.base.BudgetExceeded("leg")
        return stub(problem)
    monkeypatch.setattr(fit, "solve_certified", solve)
    selected = fit.fit_certified_leg(evaluator, fit.base.neutral(1))
    history = json.loads((tmp_path/"conic-history.json").read_text())
    accepted = [p for p in history["proposals"] if p["status"] == "accepted"]
    assert radii == [.05, expected_radius] and accepted[0]["alpha"] == expected_alpha
    assert accepted[0]["next_radius"] == expected_radius and fit.previous.eligible(selected)


def test_all_rejected_halves_radius_and_keeps_completed_feasible_selection(tmp_path, monkeypatch):
    evaluator = GeneratedEvaluator(tmp_path)
    radii = []
    stub = stub_certified([0., .02, 0.])
    def solve(problem, *args, **kwargs):
        radii.append(problem["radius"])
        if len(radii) > 1:
            raise fit.base.BudgetExceeded("leg")
        return stub(problem)
    monkeypatch.setattr(fit, "solve_certified", solve)
    monkeypatch.setattr(fit.previous, "proposal_numbers", lambda *args: (1., -1.))
    selected = fit.fit_certified_leg(evaluator, fit.base.neutral(1))
    history = json.loads((tmp_path/"conic-history.json").read_text())
    assert radii == [.05, .025] and len(history["proposals"]) == 6
    assert all(p["next_radius"] is None for p in history["proposals"])
    assert selected["evaluation"] == 1 and selected["objective"] < 1  # Complete feasible candidates still participate.


def test_uncertified_direction_never_reaches_actual_evaluator(tmp_path, monkeypatch):
    evaluator = GeneratedEvaluator(tmp_path)
    monkeypatch.setattr(fit, "solve_certified", stub_certified([0., .02, 0.], valid=False, status="Solved"))
    selected = fit.fit_certified_leg(evaluator, fit.base.neutral(1))
    assert selected["status"] == "bounded_socp_failure" and len(evaluator.evaluations) == 1
    stats = evaluator.budget.e277_stats
    assert stats["certified_solved"] == stats["certified_almost_solved"] == 0


@pytest.mark.parametrize("scope", ["leg", "shared_fit", "unique_evaluations", "original"])
@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("blocked", [None, "selection.json", "fit-pass.json"])
def test_budget_primary_and_selection_survive_through_upper_loop(tmp_path, monkeypatch, scope, complete, blocked):
    now = [270.]
    budget = fit.base.Budget(clock=lambda: now[0], deadline=fit.base.OriginalDeadline(0., clock=lambda: now[0]))
    folder = tmp_path/"LF"
    folder.mkdir()
    evaluator = GeneratedEvaluator(folder, budget=budget)
    evaluate = evaluator.evaluate
    primary = fit.base.BudgetExceeded(scope)
    primary.add_note("generated original evidence")
    attempts, legs = [], []
    def late(x):
        attempts.append(x.copy())
        if len(attempts) == 1:
            return evaluate(x)
        if complete:
            evaluate(x)
        else:
            np.savez(folder/"eval-001-startup.npz", q=np.zeros((101, 7)))
        if blocked is not None:
            ((folder if blocked == "selection.json" else tmp_path)/blocked).mkdir()
        if scope == "original":
            now[0] = 291.
        raise primary
    evaluator.evaluate = late
    monkeypatch.setattr(fit, "solve_certified", stub_certified([0., .02, 0.]))
    def run_leg(index):
        legs.append(index)
        chosen = fit.fit_certified_leg(evaluator, fit.base.neutral(1))
        # End this generated upper pass after its chosen leg without another
        # actual evaluation. The saved selection is inspected below.
        now[0] = 451.
        return chosen
    if scope == "original" or blocked == "selection.json":
        with pytest.raises(fit.base.BudgetExceeded) as caught:
            fit.base.run_fit_pass(run_leg, budget, [np.array([i]) for i in range(6)], tmp_path)
        assert caught.value is primary and primary.__notes__[0] == "generated original evidence"
    elif blocked == "fit-pass.json":
        # The leg's recoverable budget result completed normally; a new upper
        # storage fault is the primary because no exception is in flight.
        with pytest.raises(OSError):
            fit.base.run_fit_pass(run_leg, budget, [np.array([i]) for i in range(6)], tmp_path)
    else:
        fit.base.run_fit_pass(run_leg, budget, [np.array([i]) for i in range(6)], tmp_path)
    assert legs == [0] and budget.solver_calls == 1 and len(attempts) == 2
    assert len(evaluator.evaluations) == (2 if complete else 1)
    history = json.loads((folder/"conic-history.json").read_text())
    assert history["proposals"][0]["evaluation_complete"] is complete
    if blocked != "selection.json":
        saved = json.loads((folder/"selection.json").read_text())
        assert saved["evaluation"] == (1 if complete else 0) and saved["budget_scope"] == scope
    if blocked == "selection.json":
        assert primary.abort_run is True
    if scope == "original" and blocked == "fit-pass.json":
        assert any("Partial candidate receipt failed" in note for note in primary.__notes__)


def test_certificate_primary_and_partial_raw_survive_secondary_archive_fault(tmp_path, monkeypatch):
    evaluator = GeneratedEvaluator(tmp_path)
    primary = RuntimeError("generated certificate primary")
    raw = stub_certified([0., .02, 0.])
    def fail(problem, *args, **kwargs):
        data = raw(problem)
        primary.conic_receipt = data["receipt"]
        primary.conic_receipt["certificate"] = {"requested": 1, "completed": 0, "failed": 1,
                                                "wall_s": None, "wall_known": False}
        primary.partial_certified_solution = data
        raise primary
    original = fit.save_direction
    def blocked(path, problem, solved):
        if solved is not None:
            raise OSError("generated raw archive fault")
        original(path, problem, solved)
    monkeypatch.setattr(fit, "solve_certified", fail)
    monkeypatch.setattr(fit, "save_direction", blocked)
    with pytest.raises(RuntimeError) as caught:
        fit.fit_certified_leg(evaluator, fit.base.neutral(1))
    assert caught.value is primary and primary.abort_run is True
    assert any("partial certified solution persistence failed" in note for note in primary.__notes__)
    assert evaluator.budget.e277_stats["certificate_wall_s"] is None
    assert evaluator.budget.e277_stats["certificate_wall_unknown_calls"] == 1


def test_stationary_point_requires_no_solver_or_certificate(tmp_path, monkeypatch):
    evaluator = GeneratedEvaluator(tmp_path)
    evaluate = evaluator.evaluate
    def zero(x):
        actual = evaluate(x)
        actual["gradient"].fill(0)
        return actual
    evaluator.evaluate = zero
    def forbidden(*args, **kwargs):
        pytest.fail("No direction solve at exact stationary point")
    monkeypatch.setattr(fit, "solve_certified", forbidden)
    assert fit.fit_certified_leg(evaluator, fit.base.neutral(1))["status"] == "stationary"
    assert evaluator.budget.solver_calls == evaluator.budget.e277_stats["certificate_requests"] == 0


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
    e275 = {**original, "schema": fit.conic.SCHEMA, "contract": fit.conic.CONTRACT, "inputs": dict(inputs),
        "approved_sources": {k: sources[k] for k in fit.conic.SOURCES}, "e270_original_freeze": {"input": "original"},
        "generated_e273_gate": {"input": "fast"}}
    file("e275", e275)
    freeze = {**e275, "schema": fit.SCHEMA, "contract": copy.deepcopy(fit.CONTRACT), "inputs": inputs,
        "approved_sources": sources, "e275_original_freeze": {"input": "e275"}}
    fp, rp = tmp_path/"freeze.json", tmp_path/"review.json"
    def save():
        fp.write_text(json.dumps(freeze))
        digest = fit.base.sha256(fp)
        rp.write_text(json.dumps({"schema": fit.REVIEW_SCHEMA, "verdict": "approved",
            "approved_sources": sources, "freeze_sha256": digest}))
        return fp, digest, rp, fit.base.sha256(rp)
    return freeze, save, file, e275, fast


def test_twelve_source_freeze_is_distinct_and_preserves_e275(tmp_path, monkeypatch):
    freeze, save, _, _, _ = make_freeze(tmp_path, monkeypatch)
    assert fit.validate_freeze(*save()) == freeze and len(fit.SOURCES) == 12
    freeze["approved_sources"].pop(fit.SOURCES[-1])
    with pytest.raises(ValueError, match="Twelve E277"):
        fit.validate_freeze(*save())


@pytest.mark.parametrize("bad", ["e275_sources", "e275_contract", "e275_context", "input", "certificate", "radius", "speed", "source"])
def test_freeze_rejects_changed_contract_sources_or_original_context(tmp_path, monkeypatch, bad):
    freeze, save, file, e275, fast = make_freeze(tmp_path, monkeypatch)
    if bad == "e275_sources":
        e275["approved_sources"][fit.conic.SOURCES[-1]] = "a"*64
    elif bad == "e275_contract":
        e275["contract"] = {"changed": True}
    elif bad == "e275_context":
        e275["contexts"] = {"changed": True}
    elif bad == "input":
        file("native", {"status": "pass"})
    elif bad == "certificate":
        freeze["contract"]["direction_certificate"]["dual_cone_tolerance"] = 1e-8
    elif bad == "radius":
        freeze["contract"]["accepted_alpha_radius"] = "changed"
    elif bad == "speed":
        fast["median_speedup"] = 1.49
    else:
        fast["source_sha256"][fit.ENGINE_SOURCES[0]] = "b"*64
    if bad.startswith("e275"):
        file("e275", e275)
    if bad in ("speed", "source"):
        file("fast", fast)
    with pytest.raises(ValueError):
        fit.validate_freeze(*save())


def test_explicit_factory_and_final_base_are_reused_without_original(tmp_path, monkeypatch):
    signature = inspect.signature(fit.base.execute_original)
    assert signature.parameters["evaluator_factory"].default is fit.base.LegEvaluator
    captured = []
    def execute(*args, **kwargs):
        captured.append(kwargs)
        return {"generated": True}
    monkeypatch.setattr(fit.base, "execute_original", execute)
    assert fit.original_run({}, tmp_path, time.perf_counter()) == {"generated": True}
    factory = captured[0]["evaluator_factory"]
    assert factory.func is fit.base.LegEvaluator and factory.keywords == {"retain_startup_jacobians": True}
    assert captured[0]["leg_fit"] is fit.fit_certified_leg
