"""E280 frozen six-case conditional restoration diagnostic; never a fitter."""
from __future__ import annotations

# Preserve primary failures through receipt and cleanup errors.
# ruff: noqa: BLE001
import argparse
import copy
import gc
import importlib.metadata
import json
import platform
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import banc_m3_feasibility_restoration as restoration
from tools import fit_c_banc_m3_certified_source as inherited

base = inherited.base
SCHEMA = "flylab.m3.e280.feasibility-restoration.v1"
REVIEW_SCHEMA = "flylab.m3.e280.source-review.v1"
SOURCES = ("tools/banc_m3_feasibility_restoration.py", "tests/test_c_banc_feasibility_restoration.py",
           "tools/verify_c_banc_m3_feasibility_restoration.py", "tests/test_c_banc_feasibility_restoration_runner.py")
REQUIRED_INHERITED_SOURCES = inherited.SOURCES
CASES = [{"leg": leg, "alpha": alpha, "trial_evaluation": trial, "comparator_evaluation": comparator}
         for leg, alpha, trial, comparator in zip(base.LEGS, (.25, .5, 1., .25, .03125, .5),
                                                (3, 2, 1, 3, 6, 2), (4, 3, 2, 4, 0, 3), strict=True)]
CONTRACT = {**copy.deepcopy(restoration.CONTRACT), "cases": CASES, "case_order": list(base.LEGS),
    "science_wall_s": 45., "supervisor_wall_s": 70., "cleanup_reserve_s": 25.,
    "max_contexts_live": 2, "max_engines": 12, "max_solve_calls": 12, "max_evaluations": 24,
    "max_forward_ticks": 624000, "max_derivative_ticks": 624000,
    "ram_bytes": 6*1024**3, "gpu_additional_bytes": 4*1024**3,
    "output_bytes": 2*1024**3, "free_disk_bytes": 4*1024**3,
    "startup_samples": 101, "teacher_samples": 1201, "whole_BANC_calls": 0, "physics_steps": 0,
    "fit_calls": 0, "F6_exposed": False, "case_sequence": "before/trial-reproduction,corrections,release;stop-on-invalid"}


def package_versions():
    cupy = [(d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
            if d.metadata["Name"] and d.metadata["Name"].lower() in ("cupy", "cupy-cuda12x", "cupy-cuda13x", "cupy-cuda11x")]
    base.require(len(cupy) == 1, "Exactly one installed CuPy distribution required")
    return {"python": platform.python_version(), **{k: importlib.metadata.version(k) for k in
        ("numpy", "scipy", "clarabel", "psutil")}, "cupy_distribution": {"name": cupy[0][0], "version": cupy[0][1]}}


def _verify_spec(spec):
    base.require(isinstance(spec, dict) and set(spec) == {"path", "sha256"}, "Exact path/SHA spec required")
    path = base.safe_path(spec["path"])
    base.require(base.sha256(path) == spec["sha256"], "Frozen file changed: "+spec["path"])
    return path


def validate_freeze(path, digest, review_path, review_digest):
    freeze, review = base.verified_json(path, digest), base.verified_json(review_path, review_digest)
    base.require(freeze.get("schema") == SCHEMA and freeze.get("contract") == CONTRACT, "Exact E280 contract/schema required")
    sources = freeze.get("approved_sources")
    base.require(isinstance(sources, dict) and set(sources) == set(SOURCES), "Exactly four owned E280 sources required")
    for name, sha in sources.items():
        base.require(base.sha256(base.safe_path(name)) == sha, "Owned source changed: "+name)
    base.require(review.get("schema") == REVIEW_SCHEMA and review.get("verdict") == "approved"
        and review.get("approved_sources") == sources and review.get("freeze_sha256") == digest, "Exact E280 source review required")
    pins = freeze.get("inherited_sources")
    base.require(isinstance(pins, dict) and set(REQUIRED_INHERITED_SOURCES) <= set(pins), "All required inherited sources must be pinned")
    for name, sha in pins.items():
        base.require(base.sha256(base.safe_path(name)) == sha, "Inherited source changed: "+name)
    original = json.loads(_verify_spec(freeze["e277_freeze"]).read_text(encoding="utf-8"))
    base.require(original.get("schema") == inherited.SCHEMA and original.get("contract") == inherited.CONTRACT
        and original.get("approved_sources") == {k: pins[k] for k in inherited.SOURCES}, "Exact inherited E277 freeze required")
    base.require(freeze.get("inputs") == original["inputs"], "All inherited input pins must remain unchanged")
    for spec in freeze["inputs"].values():
        _verify_spec(spec)
    for key in ("native_validation", "generated_cuda_gate", "generated_e273_gate"):
        gate = inherited.previous.hashed_gate(original, key)
        base.require(gate.get("status") == "pass", "Inherited gate did not pass: "+key)
        if key == "native_validation":
            base.require(all(gate.get(k) is True for k in ("counters_valid", "raw_valid", "identity_valid", "decoder_valid")), "Native gate fields missing")
        if key == "generated_e273_gate":
            speed = gate.get("median_speedup")
            base.require(gate.get("exactness_valid") is True and gate.get("speed_valid") is True
                and isinstance(speed, (int, float)) and not isinstance(speed, bool) and np.isfinite(speed) and speed >= 1.5
                and gate.get("source_sha256") == {k: pins[k] for k in inherited.ENGINE_SOURCES}, "Exact E273 engine/speed gate required")
    base.require(freeze.get("packages") == package_versions(), "Frozen installed package versions required")
    cases = freeze.get("cases")
    base.require(isinstance(cases, list) and len(cases) == 6, "Exactly six fixed cases required")
    for row, expected in zip(cases, CASES, strict=True):
        base.require(isinstance(row, dict) and set(row) == {*expected, "path", "sha256"}
            and {k: row[k] for k in expected} == expected, "Fixed case roster/alpha/evaluation changed")
        with np.load(_verify_spec({k: row[k] for k in ("path", "sha256")}), allow_pickle=False) as z:
            case = {k: z[k] for k in z.files}
        restoration.validate_case(case)
        base.require(float(case["alpha"]) == expected["alpha"], "Case alpha differs from frozen roster")
        if expected["leg"] == "RM":
            base.require(np.array_equal(case["before_coefficients"], case["comparator_coefficients"])
                and float(case["reference_comparator_objective"]) == float(case["before_objective"]), "RM comparator must be its before neutral")
    return freeze, original


class ScienceDeadline:
    def __init__(self, started, *, clock=time.perf_counter):
        self.started, self.clock = started, clock
    def remaining(self):
        return 45-(self.clock()-self.started)
    def check(self):
        if self.remaining() >= 0:
            return
        raise base.BudgetExceeded("original")
    def receipt(self):
        return {"original_wall_s": self.clock()-self.started, "science_wall_limit_s": 45., "supervisor_wall_limit_s": 70.}


class RestorationBudget(base.Budget):
    def check(self, *, new=False):
        self.deadline.check()
        if new and (self.unique >= 4 or self.total_unique >= 24):
            raise base.BudgetExceeded("restoration_evaluations")


class RestorationMeter(base.RunMeter):
    def __init__(self, *args):
        self.case_results, self.engine_requests, self.cp = [], 0, None
        super().__init__(*args)
    def receipt(self):
        return {**super().receipt(), "case_results": self.case_results, "engine_requests": self.engine_requests,
                "fit_calls": 0, "whole_BANC_calls": 0, "physics_steps": 0, "F6_exposed": False}
    def enforce_memory(self):
        base.require(self.peak_estimate <= CONTRACT["ram_bytes"] and
            (self.peak_rss is None or self.peak_rss <= CONTRACT["ram_bytes"]), "Restoration RAM bound exceeded")
        live = [x for x in self.memory_events if x.get("live_gpu_estimate") is not None]
        if live:
            base.require(live[-1]["live_gpu_estimate"] <= CONTRACT["gpu_additional_bytes"], "Additional GPU bound exceeded")
    def cleanup(self, cp=None):
        started, primary = None, None
        row = {"status": "requested"}
        self.cleanup_events.append(row)
        try:
            started = self.deadline.clock()
            gc.collect()
            cp = self.cp if cp is None else cp
            if cp is not None:
                cp.get_default_memory_pool().free_all_blocks()
            row["status"] = "complete"
        except Exception as error:
            primary = error
            row.update(status="failed", error=str(error))
        finally:
            try:
                base.require(started is not None, "Cleanup start timing unavailable")
                row["wall_s"] = self.deadline.clock()-started
            except Exception as error:
                row["wall_s"] = None
                if primary is None:
                    primary = error
                else:
                    primary.add_note("Cleanup timing failed: "+repr(error))
            if primary is None:
                self.save()
            else:
                base.preserve_primary(primary, "Cleanup receipt failed", self.save)
        if primary is not None:
            raise primary


def _persist(folder, name, arrays, receipt):
    if arrays is not None:
        np.savez(folder/(name+".npz"), **arrays)
    base.write_json(folder/(name+".json"), receipt)
    total = sum(p.stat().st_size for p in folder.parent.rglob("*") if p.is_file())
    base.require(total <= CONTRACT["output_bytes"], "Restoration output bound exceeded")


def load_inputs(original):
    mappings = {name: {k: base.load_ref(original, ref) for k, ref in context["mapping"].items()}
                for name, context in original["contexts"].items()}
    base.require(set(mappings) == {"startup", "teacher"}, "Two original contexts required")
    for mapping in mappings.values():
        mapping["stage_order"] = tuple(mapping["stage_order"].tolist())
    base.validate_static_contexts(mappings["startup"], mappings["teacher"])
    legs = {k: base.load_ref(original, ref) for k, ref in original["legs"].items()}
    base.validate_leg_ownership(mappings["startup"], legs)
    indices = {i: tuple(np.flatnonzero(legs[k] == i) for k in ("motor", "muscle", "edge", "q", "ltm")) for i in range(6)}
    base.require([len(indices[i][2]) for i in range(6)] == [300, 282, 254, 261, 268, 295], "Original edge ownership changed")
    targets = {k: base.load_ref(original, ref) for k, ref in original["targets"].items()}
    base.validate_targets(targets)
    targets["teacher_q"] = targets["teacher_q"][:1201].copy()
    mappings["teacher"] = base.training_mapping(mappings["teacher"])
    return mappings, legs, indices, targets


def _enrich(evaluator, entry):
    with np.load(evaluator.output/f"eval-{entry['evaluation']:03d}-startup.npz", allow_pickle=False) as raw:
        entry["startup_q"], entry["startup_ltm"] = raw["q"], raw["ltm"]
    return entry


def make_evaluator_builder(inputs, factory, meter):
    mappings, legs, indices, targets = inputs
    @contextmanager
    def build(index, case, folder, budget):
        _, _, edge, q, ltm = indices[index]
        base.require(case["before_coefficients"].shape == (len(edge), 3), "Case coefficient axes differ from explicit edge roster")
        startup, startup_idx = base.contract_leg(mappings["startup"], legs, index)
        teacher, teacher_idx = base.contract_leg(mappings["teacher"], legs, index)
        base.require(all(np.array_equal(a, b) for a, b in zip(startup_idx, teacher_idx, strict=True)), "Context explicit rosters differ")
        budget.unique = 0
        budget.leg_started = budget.clock()
        def counted_factory(mapping):
            meter.engine_requests += 1
            base.require(meter.engine_requests <= 12, "At most twelve conditional engines")
            return factory(mapping)
        with base.engine_scope(counted_factory, startup, meter, "restore-"+base.LEGS[index]+"-startup") as es:  # noqa: SIM117
            with base.engine_scope(counted_factory, teacher, meter, "restore-"+base.LEGS[index]+"-training") as et:
                meter.memory_events[-1]["live_gpu_estimate"] = es.problem["gpu_bytes_estimate"]+et.problem["gpu_bytes_estimate"]
                meter.enforce_memory()
                evaluator = base.LegEvaluator(es, et, es.problem, targets["startup_q"][:, q], targets["startup_ltm"][:, ltm],
                    targets["teacher_q"][:, q], case["fixed_sse"], float(case["baseline_mse"]), budget, folder,
                    retain_startup_jacobians=True)
                def evaluate(flat):
                    try:
                        return _enrich(evaluator, evaluator.evaluate(flat))
                    except base.BudgetExceeded as error:
                        key = np.asarray(flat, np.float64).reshape(-1, 3).tobytes()
                        if key in evaluator.cache:
                            saved = base.preserve_primary(error, "Completed deadline evaluation rescue failed",
                                lambda: _enrich(evaluator, evaluator.cache[key]))
                            if saved is not None:
                                error.completed_restoration_entry = saved
                        raise
                try:
                    yield evaluate
                finally:
                    meter.peak_estimate = max(meter.peak_estimate, meter.host_bytes+meter.current_estimate+base.array_bytes(evaluator.cache))
    return build


def execute_cases(freeze, build_evaluator, meter):
    budget = RestorationBudget(clock=meter.deadline.clock, deadline=meter.deadline)
    meter.budget = budget
    results = []
    for index, row in enumerate(freeze["cases"]):
        meter.deadline.check()
        folder = meter.output/row["leg"]
        folder.mkdir()
        with np.load(base.safe_path(row["path"]), allow_pickle=False) as z:
            case = {k: z[k] for k in z.files}
        primary = None
        try:
            with build_evaluator(index, case, folder, budget) as evaluate:
                result = restoration.restore_case(case, evaluate, persist=lambda name, arrays, receipt, folder=folder:
                    _persist(folder, name, arrays, receipt), check=meter.deadline.check,
                    remaining=meter.deadline.remaining)
                results.append({"leg": row["leg"], **result})
                meter.case_results = results
                budget.solver_calls += result["cost"]["solve"]["requested"]
                budget.iterations += result["cost"]["iterations"]
                base.require(budget.solver_calls <= 12, "At most twelve restoration solves")
                meter.save()
        except Exception as error:
            primary = error
            partial = getattr(error, "restoration_case_result", None)
            if partial is not None:
                results.append({"leg": row["leg"], **partial})
                meter.case_results = results
                budget.solver_calls += partial["cost"]["solve"]["requested"]
                budget.iterations += partial["cost"]["iterations"]
            error.partial_restoration_results = results
            base.preserve_primary(error, "Restoration run failure receipt failed", meter.save)
            raise
        finally:
            if primary is None:
                meter.cleanup()
            else:
                base.preserve_primary(primary, "Restoration cleanup failed", meter.cleanup)
    return {"run_valid": True, "status": "complete", "cases": results,
        "method_supported": bool(len(results) == 6 and all(r["restored"] and r["comparator_improved"] for r in results)),
        "candidate_created": False, "body_promoted": False, "F6_exposed": False}


def original_run(freeze, original, output, started):
    from tools.banc_m3_source_coefficient_engine import CudaSourceCoefficientEngine
    meter = RestorationMeter(output, ScienceDeadline(started))
    primary = None
    try:
        base.require(shutil.disk_usage(output).free >= CONTRACT["free_disk_bytes"], "Four GiB free disk required")
        inputs = load_inputs(original)
        meter.host_bytes = base.array_bytes(inputs)
        meter.enforce_memory()
        # Only the recorded source engine's exact precomputed implementation.
        def factory(mapping):
            engine = CudaSourceCoefficientEngine(mapping, forward_implementation="precomputed")
            meter.cp = engine.cp
            return engine
        result = execute_cases(freeze, make_evaluator_builder(inputs, factory, meter), meter)
        return result
    except Exception as error:
        primary = error
        base.preserve_primary(error, "E280 original failure receipt failed",
            lambda stop=error: base.failure_record(stop, None, output, "e280-original", meter.receipt()))
        raise
    finally:
        if primary is None:
            meter.cleanup()
            meter.save()
        else:
            base.preserve_primary(primary, "E280 final cleanup failed", meter.cleanup)
            base.preserve_primary(primary, "E280 final cost failed", meter.save)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "freeze-sha256", "source-review", "source-review-sha256", "output"):
        parser.add_argument("--"+key, required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    base.require(not output.exists(), "Fresh E280 output directory required")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    base.write_json(output/"cost.json", {"phase": "freeze_validation", "original_calls": 0, "solver_calls": 0})
    try:
        freeze, original = validate_freeze(Path(args.freeze), args.freeze_sha256, Path(args.source_review), args.source_review_sha256)
        result = original_run(freeze, original, output, started)
        base.write_json(output/"result.json", result)
        return 0
    except Exception as error:
        result = {"run_valid": False, "status": "incomplete" if isinstance(error, base.BudgetExceeded) else "invalid",
            "error": str(error), "exception": type(error).__name__, "notes": getattr(error, "__notes__", []),
            "cases": getattr(error, "partial_restoration_results", []), "method_supported": False,
            "candidate_created": False, "body_promoted": False}
        base.preserve_primary(error, "E280 terminal result failed", lambda: base.write_json(output/"result.json", result))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
