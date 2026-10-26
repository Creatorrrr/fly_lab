"""Generated CPU checks only; no original input, GPU, neural/body execution."""

import json

import numpy as np
import pytest

from tools import fit_c_banc_m3_source_coefficients as fit


def finite_difference(fun, x, epsilon=1e-6):
    cols = []
    for p in range(len(x)):
        offset = np.zeros_like(x)
        offset[p] = epsilon
        cols.append((fun(x + offset) - fun(x - offset)) / (2 * epsilon))
    return np.stack(cols, axis=-1)


def test_objective_analytic_gradient():
    rng = np.random.default_rng(270)
    c = np.array([[1.1, 0.2, 0.1], [0.7, 0.3, 0.4]], np.float64)
    jac = rng.normal(size=(8, 7, 6))
    target = rng.normal(size=(8, 7))

    def scalar(x):
        return fit.objective(
            x.reshape(2, 3), np.einsum("sqp,p->sq", jac, x), jac, target, 0.3
        )[0]

    actual = fit.objective(c, np.einsum("sqp,p->sq", jac, c.ravel()), jac, target, 0.3)[
        1
    ]
    np.testing.assert_allclose(
        actual, finite_difference(scalar, c.ravel()), rtol=1e-7, atol=1e-7
    )


def test_global_rms_allows_leg_above_point018():
    q = np.full((99, 7), 0.022)
    l = np.zeros((99, 1))
    qj = np.zeros((99, 7, 3))
    lj = np.zeros((99, 1, 3))
    value, _ = fit.startup_constraints(q, l, qj, lj, (0, 0))
    assert value[0] > 0  # 42 joint global budget, not seven joint local budget.
    value, _ = fit.startup_constraints(q, l, qj, lj, (99 * 35 * 0.021**2, 0))
    assert value[0] < 0


def test_constraints_derivative_and_first_argmax():
    rng = np.random.default_rng(19)
    qbase = rng.normal(0, 0.003, (99, 7))
    lbase = rng.normal(0, 0.002, (99, 1))
    qj = rng.normal(0, 0.001, (99, 7, 4))
    lj = rng.normal(0, 0.001, (99, 1, 4))
    x = np.array([0.1, 0.2, 0.3, 0.4])

    def value(v):
        q = qbase + np.einsum("sqp,p->sq", qj, v)
        l = lbase + np.einsum("sqp,p->sq", lj, v)
        return fit.startup_constraints(q, l, qj, lj, (0.01, 0.002))

    np.testing.assert_allclose(
        value(x)[1], finite_difference(lambda v: value(v)[0], x), atol=1e-9
    )
    q = np.zeros((99, 7))
    q[0, 0] = 0.03
    q[1, 0] = -0.03
    qj[0, 0] = 2
    qj[1, 0] = 9
    _, grad = fit.startup_constraints(q, np.zeros((99, 1)), qj, lj, (0, 0))
    np.testing.assert_array_equal(grad[1], np.full(4, -2))


def test_bounds_selection_and_rounding():
    assert fit.coefficient_valid(fit.neutral(2), interior=True)
    assert not fit.coefficient_valid(
        np.array([[0.125, 0, 0]], np.float64), interior=True
    )
    assert fit.coefficient_valid(np.array([[0.125, 0, 0]], np.float64))
    linear = fit.triple_constraint(2)
    np.testing.assert_array_equal(linear.A @ np.ones(6), [3, 3])
    fallback = {"tag": "fallback"}
    entries = [
        {
            "valid": True,
            "mse": 0.9,
            "baseline_mse": 1,
            "objective": 0.8,
            "tag": "first",
        },
        {
            "valid": True,
            "mse": 0.7,
            "baseline_mse": 1,
            "objective": 0.8,
            "tag": "second",
        },
        {
            "valid": True,
            "mse": 1.1,
            "baseline_mse": 1,
            "objective": 0.1,
            "tag": "worse",
        },
        {
            "valid": False,
            "mse": 0.5,
            "baseline_mse": 1,
            "objective": 0.1,
            "tag": "unsafe",
        },
    ]
    assert fit.choose_candidate(entries, fallback)["tag"] == "first"
    assert fit.choose_candidate(entries[2:], fallback) is fallback


def test_budget_counts_and_global_clock():
    clock = [0.0]
    budget = fit.Budget(clock=lambda: clock[0])
    budget.unique = 80
    with pytest.raises(fit.BudgetExceeded):
        budget.check(new=True)
    budget.check()  # cached evaluations do not consume new quota
    clock[0] = 31
    with pytest.raises(fit.BudgetExceeded):
        budget.check()
    budget.start_leg()
    assert budget.unique == 0
    clock[0] = 180
    with pytest.raises(fit.BudgetExceeded):
        budget.start_leg()


class FakeEngine:
    """Linear generated response; shares settled engine transfer API."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = 0
        self.problem = {"samples": rows}

    def solve(self, c, derivative=True, host=False):
        assert derivative and not host
        self.calls += 1
        q = np.zeros((self.rows, 7)) + c[0, 1] * 0.001
        return {
            "q": q,
            "ltm": np.zeros((self.rows, 1)),
            "receipt": {"status": "complete"},
        }

    def host_copy(self, value, label=""):
        return value.copy()

    def command_jacobian(self, result, samples, host=True):
        j = np.zeros((len(samples), 7, 3))
        j[:, :, 1] = 0.001
        return {"q": j, "ltm": np.zeros((len(samples), 1, 3))}

    def activation_cotangent_product(self, result, cotangent, host=True):
        return np.array([0, 0.05 * np.sum(cotangent), 0])


def test_fake_engine_cache_and_no_development_selection(tmp_path):
    s, t = FakeEngine(101), FakeEngine(1201)
    budget = fit.Budget()
    target = np.full((1201, 7), 0.002)
    evaluator = fit.LegEvaluator(
        s,
        t,
        {"fixed": np.zeros(1, bool), "moment": np.ones((7, 1))},
        np.zeros((99, 7)),
        np.zeros((99, 1)),
        target,
        (0, 0),
        4e-6,
        budget,
        tmp_path,
    )
    c = np.array([[1, 0.2, 0.1]], np.float64)
    a = evaluator.evaluate(c.ravel())
    b = evaluator.evaluate(c.ravel())
    assert (
        a is b
        and s.calls == t.calls == 1
        and budget.unique == 1
        and budget.cache_hits == 1
    )
    assert a["mse"] == pytest.approx((0.0002 - 0.002) ** 2)
    assert a["valid"]
    assert (tmp_path / "eval-000-teacher.npz").is_file()


def test_development_only_final_adjudication():
    target = np.zeros((2001, 42))
    base = np.ones_like(target) * 0.1
    q = base.copy()
    q[1:1201] = 0.07
    q[1201:] = 0.2
    result = fit.adjudicate(
        q,
        base,
        target,
        np.zeros((99, 42)),
        np.zeros((99, 6)),
        np.repeat(np.arange(6), 7),
    )
    assert result["checks"]["train"] and not result["checks"]["development"]
    assert not result["conditional_support"] and not result["body_promotion"]


def test_review_hashes_and_exact_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, "ROOT", tmp_path)
    sources = {}
    for name in fit.SOURCES:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name)
        sources[name] = fit.sha256(p)
    native = tmp_path / "native.json"
    native.write_text(
        json.dumps(
            {
                "status": "pass",
                "counters_valid": True,
                "raw_valid": True,
                "identity_valid": True,
                "decoder_valid": True,
            }
        )
    )
    cuda = tmp_path / "cuda.json"
    cuda.write_text(json.dumps({"status": "pass"}))
    freeze = {
        "schema": fit.SCHEMA,
        "approved_sources": sources,
        "contract": fit.CONTRACT,
        "inputs": {
            k: {"path": p.name, "sha256": fit.sha256(p)}
            for k, p in [("native", native), ("cuda", cuda)]
        },
        "native_validation": {"input": "native"},
        "generated_cuda_gate": {"input": "cuda"},
    }
    fp = tmp_path / "freeze.json"
    fp.write_text(json.dumps(freeze))
    digest = fit.sha256(fp)
    review = {
        "schema": fit.REVIEW_SCHEMA,
        "verdict": "approved",
        "approved_sources": sources,
        "freeze_sha256": digest,
    }
    rp = tmp_path / "review.json"
    rp.write_text(json.dumps(review))
    assert fit.validate_freeze(fp, digest, rp, fit.sha256(rp)) == freeze
    (tmp_path / fit.SOURCES[0]).write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        fit.validate_freeze(fp, digest, rp, fit.sha256(rp))


def test_generated_single_slsqp_and_cost(tmp_path):
    class QuadraticEvaluator:
        def __init__(self):
            self.budget = fit.Budget()
            self.output = tmp_path
            self.baseline_mse = 1.0
            self.evaluations = []
            self.cache = {}

        def evaluate(self, x):
            key = x.tobytes()
            if key in self.cache:
                return self.cache[key]
            self.budget.check(new=True)
            self.budget.unique += 1
            self.budget.total_unique += 1
            diff = x - np.array([0.8, 0.1, 0.1])
            c = x.reshape(1, 3)
            entry = {
                "coefficients": c.copy(),
                "objective": float(diff @ diff),
                "gradient": 2 * diff,
                "mse": float(diff @ diff),
                "baseline_mse": 1.0,
                "constraints": np.ones(2),
                "constraint_jacobian": np.zeros((2, 3)),
                "valid": fit.coefficient_valid(c, interior=True),
                "evaluation": len(self.evaluations),
            }
            self.evaluations.append(entry)
            self.cache[key] = entry
            return entry

    evaluator = QuadraticEvaluator()
    selected = fit.fit_leg(evaluator, fit.neutral(1))
    np.testing.assert_allclose(selected["coefficients"], [[0.8, 0.1, 0.1]], atol=1e-7)
    assert evaluator.budget.solver_calls == 1 and evaluator.budget.iterations <= 60
    print("generated_cost", evaluator.budget.receipt())


def test_training_context_has_no_development_axes_or_rows(tmp_path):
    mapping = {
        key: np.zeros((160000, 1))
        for key in ("stage_total", "source_rate", "source_mask", "source_history")
    }
    mapping["stage_index"] = np.arange(160000, dtype=np.int64)
    mapping["source_history"] = np.zeros((160000, 1, 2))
    for key in ("stage_total", "source_rate", "source_mask", "source_history"):
        mapping[key][96000:] = 999
    train = fit.training_mapping(mapping)
    assert all(len(train[key]) == 96000 for key in mapping)
    assert all(not np.any(train[key] == 999) for key in mapping if key != "stage_index")
    full = FakeEngine(2001)
    kwargs = {
        "mapping": {"fixed": np.zeros(1, bool), "moment": np.ones((7, 1))},
        "target_q": np.zeros((99, 7)),
        "target_ltm": np.zeros((99, 1)),
        "target_teacher": np.zeros((1201, 7)),
        "fixed_sse": (0, 0),
        "baseline_mse": 1.0,
        "budget": fit.Budget(),
        "output": tmp_path,
    }
    with pytest.raises(ValueError, match="engine exposes"):
        fit.LegEvaluator(FakeEngine(101), full, **kwargs)
    kwargs["target_teacher"] = np.zeros((2001, 7))
    with pytest.raises(ValueError, match="target must exclude"):
        fit.LegEvaluator(FakeEngine(101), FakeEngine(1201), **kwargs)


@pytest.mark.parametrize(
    "stop", ["midleg", "betweenlegs", "first_evaluation", "preparation"]
)
def test_fake_clock_fit_pass_finalization_end_to_end(tmp_path, monkeypatch, stop):
    clock = [0.0]
    deadline = fit.OriginalDeadline(0.0, clock=lambda: clock[0])
    budget = fit.Budget(clock=lambda: clock[0])
    edges = [np.array([i], np.int64) for i in range(6)]
    started = []
    solver_invocations = []

    def solver(fun, x, **kwargs):
        solver_invocations.append(1)
        if stop == "first_evaluation":
            clock[0] = 181.0
            fun(x)  # no engine/evaluation may start
        else:
            fun(np.array([0.8, 0.2, 0.0]))
        raise AssertionError("budget stop should unwind the solver")

    monkeypatch.setattr(fit, "minimize", solver)

    def run_leg(leg):
        started.append(leg)
        if stop == "betweenlegs":
            clock[0] = 181.0
            return {
                "coefficients": np.array([[0.8, 0.2, 0.0]]),
                "status": "complete",
                "evaluation": 0,
            }
        if stop == "preparation":
            clock[0] = 181.0
            budget.check()

        class TimedEngine(FakeEngine):
            def activation_cotangent_product(self, result, cotangent, host=True):
                answer = super().activation_cotangent_product(
                    result, cotangent, host=host
                )
                clock[0] = 181.0
                return answer

        legdir = tmp_path / f"leg-{leg}"
        legdir.mkdir()
        evaluator = fit.LegEvaluator(
            FakeEngine(101),
            TimedEngine(1201),
            {"fixed": np.zeros(1, bool), "moment": np.ones((7, 1))},
            np.zeros((99, 7)),
            np.zeros((99, 1)),
            np.full((1201, 7), 0.002),
            (0, 0),
            4e-6,
            budget,
            legdir,
        )
        chosen = fit.fit_leg(evaluator, fit.neutral(1))
        if stop == "first_evaluation":
            assert evaluator.startup.calls == evaluator.teacher.calls == 0
        return chosen

    combined, statuses = fit.run_fit_pass(run_leg, budget, edges, tmp_path)
    assert started == [0]
    assert all(s["status"] == "neutral_not_started" for s in statuses[1:])
    np.testing.assert_array_equal(combined[1:], fit.neutral(5))
    if stop in ("midleg", "betweenlegs"):
        np.testing.assert_array_equal(combined[0], [0.8, 0.2, 0.0])
    else:
        np.testing.assert_array_equal(combined, fit.neutral(6))
    fit_wall = budget.receipt()["fit_wall_s"]
    finalized = []

    def predict(c):
        finalized.append(c.copy())
        clock[0] = 200.0
        return {"prediction": "complete"}

    result = fit.round_and_finalize(combined, predict, deadline, tmp_path)
    assert result == {"prediction": "complete"} and len(finalized) == 1
    assert budget.receipt()["fit_wall_s"] == fit_wall
    assert budget.failed == 0
    assert budget.solver_calls == (0 if stop in ("betweenlegs", "preparation") else 1)
    assert len(solver_invocations) == budget.solver_calls
    clock[0] = 291.0
    with pytest.raises(fit.BudgetExceeded, match="original"):
        fit.round_and_finalize(combined, predict, deadline, tmp_path)


def test_startup_failure_preserves_primary_if_rescue_fails(tmp_path):
    primary = RuntimeError("primary startup Jacobian failure")

    class Device:
        shape = (1,)

    primary.partial_response = {"q": Device()}
    primary.response_receipt = {"status": "failed", "completed_samples": 101}

    class BadStartup(FakeEngine):
        def __init__(self):
            super().__init__(101)
            self.labels = []

        def command_jacobian(self, *args, **kwargs):
            raise primary

        def host_copy(self, value, label=""):
            self.labels.append(label)
            if isinstance(value, Device):
                raise OSError("rescue copy failure")
            return super().host_copy(value, label=label)

    startup = BadStartup()
    teacher = FakeEngine(1201)
    evaluator = fit.LegEvaluator(
        startup,
        teacher,
        {"fixed": np.zeros(1, bool), "moment": np.ones((7, 1))},
        np.zeros((99, 7)),
        np.zeros((99, 1)),
        np.zeros((1201, 7)),
        (0, 0),
        1.0,
        fit.Budget(),
        tmp_path,
    )
    with pytest.raises(RuntimeError, match="primary startup") as caught:
        evaluator.evaluate(fit.neutral(1).ravel())
    assert caught.value is primary and teacher.calls == 0
    record = json.loads((tmp_path / "eval-000-error.json").read_text())
    assert record["response_receipt"]["completed_samples"] == 101
    rescue = json.loads((tmp_path / "eval-000-rescue-errors.json").read_text())
    assert rescue[0]["error"] == "rescue copy failure"
    assert any(":partial:q" in label for label in startup.labels)
    assert (tmp_path / "cost.json").is_file() and evaluator.budget.failed == 1


def test_static_mapping_and_target_validation_are_fail_closed():
    a = {key: np.ones(1, np.float64) for key in fit.STATIC_KEYS}
    a["stage_order"] = ("start", "half1", "half2", "end")
    b = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in a.items()
    }
    fit.validate_static_contexts(a, b)
    for key in fit.STATIC_KEYS:
        broken = dict(b)
        broken[key] = b[key] + 1
        with pytest.raises(ValueError, match=key):
            fit.validate_static_contexts(a, broken)
    targets = {
        "startup_q": np.zeros((99, 42)),
        "startup_ltm": np.zeros((99, 6)),
        "teacher_q": np.zeros((2001, 42)),
    }
    fit.validate_targets(targets)
    targets["startup_q"] = targets["startup_q"].astype(np.float32)
    with pytest.raises(ValueError, match="startup_q"):
        fit.validate_targets(targets)
    targets["startup_q"] = np.zeros((99, 42))
    targets["teacher_q"][0, 0] = np.nan
    with pytest.raises(ValueError, match="teacher_q"):
        fit.validate_targets(targets)


def test_exact_leg_ownership_and_named_numerical_errors():
    mapping = {
        "motor_ids": np.arange(6),
        "fixed": np.zeros(6, bool),
        "edge_ids": np.arange(6),
        "rest": np.zeros(42),
        "ltm": np.arange(6),
        "membership": np.arange(6),
        "edge_post": np.arange(6),
        "pool": np.eye(6),
        "moment": np.repeat(np.eye(6), 7, axis=0),
    }
    legs = {
        key: np.arange(6, dtype=np.int64) for key in ("motor", "muscle", "edge", "ltm")
    }
    legs["q"] = np.repeat(np.arange(6, dtype=np.int64), 7)
    fit.validate_leg_ownership(mapping, legs)
    mapping["pool"][0, 1] = 0.1
    with pytest.raises(ValueError, match="Cross-leg pool"):
        fit.validate_leg_ownership(mapping, legs)
    mapping["pool"][0, 1] = 0
    mapping["moment"][0, 1] = 1
    with pytest.raises(ValueError, match="Cross-leg joint"):
        fit.validate_leg_ownership(mapping, legs)
    gate = fit.numerical_gate(np.array([0.003]), np.array([0.0]), 0.002)
    assert gate == {"max_absolute_error": 0.003, "tolerance": 0.002, "pass": False}


def test_leg_clock_stop_keeps_neutral_and_continues_without_restart(tmp_path):
    clock = [0.0]
    budget = fit.Budget(clock=lambda: clock[0])
    attempts = []

    def run_leg(leg):
        attempts.append(leg)
        if leg == 0:
            clock[0] = 31.0
            budget.check()
        return {
            "coefficients": np.array([[0.9, 0.1, 0.0]]),
            "status": "complete",
            "evaluation": 0,
        }

    candidate, statuses = fit.run_fit_pass(
        run_leg, budget, [np.array([i]) for i in range(6)], tmp_path
    )
    assert attempts == list(range(6))
    assert statuses[0]["budget_scope"] == "leg"
    np.testing.assert_array_equal(candidate[0], [1, 0, 0])
    np.testing.assert_array_equal(candidate[1:], np.tile([0.9, 0.1, 0.0], (5, 1)))
    assert budget.solver_calls == 0 and budget.failed == 0


def test_fake_engine_cannot_return_development_despite_declared_training(tmp_path):
    teacher = FakeEngine(2001)
    teacher.problem["samples"] = 1201
    evaluator = fit.LegEvaluator(
        FakeEngine(101),
        teacher,
        {"fixed": np.zeros(1, bool), "moment": np.ones((7, 1))},
        np.zeros((99, 7)),
        np.zeros((99, 1)),
        np.zeros((1201, 7)),
        (0, 0),
        1.0,
        fit.Budget(),
        tmp_path,
    )
    with pytest.raises(ValueError, match="response exposes development"):
        evaluator.evaluate(fit.neutral(1).ravel())
    assert not evaluator.evaluations and evaluator.budget.failed == 1


def test_phase_failure_records_metered_construction_receipt(tmp_path):
    clock = [0.0]
    meter = fit.RunMeter(tmp_path, fit.OriginalDeadline(0.0, clock=lambda: clock[0]))
    primary = RuntimeError("generated construction error")
    primary.response_receipt = {"upload": {"requested_count": 1, "completed_count": 0}}

    def factory(_):
        raise primary

    with (
        pytest.raises(RuntimeError, match="generated construction") as caught,
        fit.engine_scope(factory, {}, meter, "neutral-LF-startup"),
    ):
        pytest.fail("factory failure must not enter solve")
    assert caught.value is primary
    record = json.loads((tmp_path / "neutral-LF-startup-error.json").read_text())
    assert record["response_receipt"]["upload"]["requested_count"] == 1
    assert (tmp_path / "cost.json").is_file()


def test_primary_exception_survives_receipt_storage_failure(tmp_path, monkeypatch):
    meter = fit.RunMeter(tmp_path, fit.OriginalDeadline(0.0, clock=lambda: 0.0))
    primary = RuntimeError("original constructor fault")

    def factory(_):
        raise primary

    def broken_storage(*args, **kwargs):
        raise OSError("generated full disk")

    monkeypatch.setattr(fit, "write_json", broken_storage)
    with (
        pytest.raises(RuntimeError, match="original constructor fault") as caught,
        fit.engine_scope(factory, {}, meter, "neutral-LF-startup"),
    ):
        pytest.fail("No engine exists")
    assert caught.value is primary
    assert any("generated full disk" in note for note in primary.__notes__)


def test_original_primary_survives_final_meter_failure(tmp_path, monkeypatch):
    primary = RuntimeError("original context fault")

    def broken_storage(*args, **kwargs):
        raise OSError("generated receipt storage fault")

    def execute(*args):
        monkeypatch.setattr(fit, "write_json", broken_storage)
        raise primary

    monkeypatch.setattr(fit, "execute_original", execute)
    with pytest.raises(RuntimeError, match="original context fault") as caught:
        fit.original_run({}, tmp_path, 0.0)
    assert caught.value is primary
    assert any("generated receipt storage fault" in note for note in primary.__notes__)


def test_fit_and_pass_keep_solver_error_when_candidate_storage_fails(tmp_path, monkeypatch):
    from types import SimpleNamespace
    primary = RuntimeError("generated solver entry fault")
    evaluator = SimpleNamespace(budget=fit.Budget(), baseline_mse=1.0, evaluations=[], output=tmp_path)

    def broken_solver(*args, **kwargs):
        raise primary

    def broken_storage(*args, **kwargs):
        raise OSError("generated candidate storage fault")

    monkeypatch.setattr(fit, "minimize", broken_solver)
    monkeypatch.setattr(fit.np, "savez", broken_storage)
    with pytest.raises(RuntimeError, match="generated solver entry fault") as caught:
        fit.run_fit_pass(lambda _: fit.fit_leg(evaluator, fit.neutral(1)), evaluator.budget,
                         [np.array([i]) for i in range(6)], tmp_path)
    assert caught.value is primary
    assert sum("storage fault" in note for note in primary.__notes__) == 2
