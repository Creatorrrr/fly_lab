"""E281 frozen startup-only feasibility diagnostic, without teacher or fitting."""
from __future__ import annotations

# Retain primary errors when raw/cost persistence or cleanup also fails.
# ruff: noqa: BLE001
import argparse
import copy
import json
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import clarabel
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import banc_m3_startup_margin as margin
from tools import verify_c_banc_m3_feasibility_restoration as inherited

base = inherited.base
SCHEMA = "flylab.m3.e281.startup-margin-feasibility.v1"
REVIEW_SCHEMA = "flylab.m3.e281.source-review.v1"
SOURCES = ("tools/banc_m3_startup_margin.py", "tools/verify_c_banc_m3_startup_margin.py",
           "tests/test_c_banc_startup_margin.py", "tests/test_c_banc_startup_margin_runner.py")
REQUIRED_INHERITED_SOURCES = (*inherited.REQUIRED_INHERITED_SOURCES, *inherited.SOURCES)
CONTRACT = {**copy.deepcopy(margin.CONTRACT), "science_wall_s": 35., "supervisor_wall_s": 55., "cleanup_reserve_s": 20.,
    "case_order": list(base.LEGS), "startup_samples": 101, "max_engines": 6, "max_contexts_live": 1,
    "max_solve_calls": 12, "max_evaluations": 24, "max_forward_ticks": 48000, "max_derivative_ticks": 48000,
    "ram_bytes": 6*1024**3, "gpu_additional_bytes": 4*1024**3, "output_bytes": 1024**3,
    "free_disk_bytes": 4*1024**3, "whole_BANC_calls": 0, "physics_steps": 0, "fit_calls": 0,
    "F6_exposed": False, "sequence": "neutral reproduction,max2 corrections,f32 if f64 passed,release;scientific reject continues"}
package_versions = inherited.package_versions


def validate_freeze(path, digest, review_path, review_digest):
    freeze, review = base.verified_json(path, digest), base.verified_json(review_path, review_digest)
    base.require(freeze.get("schema") == SCHEMA and freeze.get("contract") == CONTRACT
        and clarabel.__version__ == "0.11.1", "Exact E281 schema/contract/Clarabel required")
    sources = freeze.get("approved_sources")
    base.require(isinstance(sources, dict) and set(sources) == set(SOURCES), "Exactly four E281 source pins required")
    for name, sha in sources.items():
        base.require(base.sha256(base.safe_path(name)) == sha, "Owned E281 source changed: "+name)
    base.require(review.get("schema") == REVIEW_SCHEMA and review.get("verdict") == "approved"
        and review.get("approved_sources") == sources and review.get("freeze_sha256") == digest, "Exact E281 source review required")
    pins = freeze.get("inherited_sources")
    base.require(isinstance(pins, dict) and set(REQUIRED_INHERITED_SOURCES) <= set(pins), "All E280/E277 inherited source pins required")
    for name, sha in pins.items():
        base.require(base.sha256(base.safe_path(name)) == sha, "Inherited source changed: "+name)
    old = json.loads(inherited._verify_spec(freeze["e280_freeze"]).read_text(encoding="utf-8"))
    base.require(old.get("schema") == inherited.SCHEMA and old.get("contract") == inherited.CONTRACT
        and old.get("approved_sources") == {k: pins[k] for k in inherited.SOURCES}
        and all(pins.get(k) == v for k, v in old.get("inherited_sources", {}).items()), "Exact original E280 source/freeze required")
    original = json.loads(inherited._verify_spec(old["e277_freeze"]).read_text(encoding="utf-8"))
    base.require(original.get("schema") == inherited.inherited.SCHEMA and original.get("contract") == inherited.inherited.CONTRACT
        and original.get("approved_sources") == {k: pins[k] for k in inherited.inherited.SOURCES}, "Original E277 source/freeze required")
    base.require(freeze.get("inputs") == old["inputs"] == original["inputs"], "Every inherited input SHA must remain unchanged")
    for spec in freeze["inputs"].values():
        inherited._verify_spec(spec)
    result = json.loads(inherited._verify_spec(freeze["e280_root_result"]).read_text(encoding="utf-8"))
    base.require(result.get("schema") == "flylab.m3.e280.root-original-audit.v1"
        and result.get("verdict") == "valid_method_supported_for_next_fit_design" and result.get("reproduction_exact") is True
        and len(result.get("cases", [])) == 6
        and all(row.get("leg") == leg and row.get("accepted") is True and row.get("beat_comparator") is True
                for row, leg in zip(result["cases"], base.LEGS, strict=True)), "Approved E280 root method result required")
    qualification = json.loads(inherited._verify_spec(freeze.get("e280_resource_qualification")).read_text(encoding="utf-8"))
    base.require(qualification.get("schema") == "flylab.m3.e280.resource-qualification.v1"
        and qualification.get("verdict") == "numerical_result_retained_actual_lifetime_assurance_withdrawn"
        and qualification.get("numeric_evidence_valid") is True
        and qualification.get("actual_live_engine_bound_verified") is False
        and qualification.get("future_reuse_requires_explicit_lifetime_release") is True
        and qualification.get("numeric_review") == freeze["e280_root_result"],
        "E280 numeric support must retain withdrawal of actual-live resource assurance")
    lifetime = json.loads(inherited._verify_spec(qualification.get("lifetime_review")).read_text(encoding="utf-8"))
    base.require(lifetime.get("schema") == "flylab.m3.e280.root-retrospective-lifetime.v1"
        and lifetime.get("source_sha256") == pins["tools/verify_c_banc_m3_feasibility_restoration.py"],
        "Linked lifetime evidence must identify the frozen E280 runner")
    for key in ("native_validation", "generated_cuda_gate", "generated_e273_gate"):
        gate = inherited.inherited.previous.hashed_gate(original, key)
        base.require(gate.get("status") == "pass", "Inherited gate failed: "+key)
        if key == "native_validation":
            base.require(all(gate.get(k) is True for k in ("counters_valid", "raw_valid", "identity_valid", "decoder_valid")), "Native integrity gate missing")
        if key == "generated_e273_gate":
            speed = gate.get("median_speedup")
            base.require(gate.get("exactness_valid") is True and gate.get("speed_valid") is True
                and isinstance(speed, (int, float)) and not isinstance(speed, bool) and np.isfinite(speed) and speed >= 1.5
                and gate.get("source_sha256") == {k: pins[k] for k in inherited.inherited.ENGINE_SOURCES}, "Reviewed exact E273 implementation required")
    base.require(freeze.get("packages") == package_versions(), "Frozen installed package versions required")
    cases, roster = freeze.get("cases"), freeze.get("engine_roster")
    base.require(isinstance(cases, list) and len(cases) == 6 and isinstance(roster, list) and len(roster) == 6,
                 "Six fixed cases and explicit startup engine rosters required")
    errors, prepared = [], []
    for row, engine, leg in zip(cases, roster, base.LEGS, strict=True):
        base.require(set(row) == {"leg", "path", "sha256", "neutral_sse", "share"} and row["leg"] == leg,
                     "Exact fixed baseline/quota row required")
        with np.load(inherited._verify_spec({k: row[k] for k in ("path", "sha256")}), allow_pickle=False) as z:
            case = {k: z[k] for k in z.files}
        margin.validate_case(case)
        keys = {"leg", "samples", "motor_indices", "muscle_indices", "edge_indices", "q_indices", "ltm_indices"}
        base.require(isinstance(engine, dict) and set(engine) == keys and engine["leg"] == leg and engine["samples"] == 101,
                     "Exact startup-only engine roster required")
        for key in keys-{"leg", "samples"}:
            values = engine[key]
            base.require(isinstance(values, list) and values and all(type(i) is int and i >= 0 for i in values)
                and values == sorted(set(values)), "Sorted explicit original index roster required: "+key)
        base.require(len(engine["edge_indices"]) == len(case["neutral_coefficients"])
            and len(engine["q_indices"]) == 7 and len(engine["ltm_indices"]) == 1, "Case/engine explicit axes differ")
        errors.append(case["neutral_startup_q"][1:100]-case["target_q"])
        prepared.append(case)
    for row, case, quota in zip(cases, prepared, margin.quota_rows(errors), strict=True):
        base.require(row["neutral_sse"] == quota["neutral_sse"] and row["share"] == quota["share"]
            and float(case["q_sse_quota"]) == quota["q_sse_quota"] and float(case["ltm_sse_quota"]) == quota["ltm_sse_quota"],
            "Frozen quotas must be calculated from all six neutral C-order errors")
    return freeze, original


class ScienceDeadline(inherited.ScienceDeadline):
    def remaining(self):
        return 35-(self.clock()-self.started)
    def receipt(self):
        return {"original_wall_s": self.clock()-self.started, "science_wall_limit_s": 35., "supervisor_wall_limit_s": 55.}


class StartupBudget:
    def __init__(self, deadline):
        self.deadline, self.leg_evaluations = deadline, 0
        self.counts = {k: 0 for k in ("evaluations_requested", "evaluations_completed", "evaluations_failed",
            "forward_requested_steps", "forward_completed_steps", "derivative_requested_steps", "derivative_completed_steps")}
    def request(self, derivative):
        self.deadline.check()
        if self.leg_evaluations >= 4 or self.counts["evaluations_requested"] >= 24:
            raise base.BudgetExceeded("startup_evaluations")
        self.leg_evaluations += 1
        self.counts["evaluations_requested"] += 1
        self.counts["forward_requested_steps"] += 2000
        self.counts["derivative_requested_steps"] += 2000*int(derivative)
        base.require(self.counts["forward_requested_steps"] <= 48000 and self.counts["derivative_requested_steps"] <= 48000,
                     "Startup neural step reservation exceeded")
    def receipt(self):
        return self.counts.copy()


class StartupMeter(inherited.RestorationMeter):
    def __init__(self, *args):
        self.socp_cost = {k: margin.restoration.phase() for k in ("constructor", "solve", "certificate")}
        self.socp_cost.update(iterations=0, iterations_unknown_calls=0)
        super().__init__(*args)
    def receipt(self):
        row = super().receipt()
        row["startup"] = row.pop("fit")
        row.update(socp_cost=self.socp_cost, teacher_calls=0, objective_calls=0)
        return row
    def memory(self, engine, phase):
        estimate = super().memory(engine, phase)
        try:
            base.require(self.active_contexts <= 1, "At most one live startup context")
            self.memory_events[-1]["live_gpu_estimate"] = engine.problem["gpu_bytes_estimate"]
            self.enforce_memory()
        except Exception as error:
            base.preserve_primary(error, "Startup rejected memory receipt release failed", lambda: self.release(engine, phase, estimate))
            raise
        return estimate
    def account(self, cost):
        for key in ("constructor", "solve", "certificate"):
            for field in ("requested", "completed", "failed"):
                self.socp_cost[key][field] += cost[key][field]
            if not cost[key]["wall_known"]:
                self.socp_cost[key].update(wall_s=None, wall_known=False)
            elif self.socp_cost[key]["wall_s"] is not None:
                self.socp_cost[key]["wall_s"] += cost[key]["wall_s"]
        self.socp_cost["iterations"] += cost["iterations"]
        self.socp_cost["iterations_unknown_calls"] += cost["iterations_unknown_calls"]
        base.require(self.socp_cost["solve"]["requested"] <= 12, "At most twelve startup correction solves")


def persist(folder, name, arrays, receipt):
    if arrays is not None:
        np.savez(folder/(name+".npz"), **arrays)
    base.write_json(folder/(name+".json"), receipt)
    base.require(sum(p.stat().st_size for p in folder.parent.rglob("*") if p.is_file()) <= CONTRACT["output_bytes"],
                 "Startup margin output bound exceeded")


def load_startup_inputs(original, roster):
    base.require(set(original["contexts"]) == {"startup", "teacher"}, "Inherited context schema required")
    # Deliberately read only the startup mapping and membership; no teacher target/mapping load.
    mapping = {k: base.load_ref(original, ref) for k, ref in original["contexts"]["startup"]["mapping"].items()}
    mapping["stage_order"] = tuple(mapping["stage_order"].tolist())
    legs = {k: base.load_ref(original, ref) for k, ref in original["legs"].items()}
    base.validate_leg_ownership(mapping, legs)
    prepared = []
    for index, leg in enumerate(base.LEGS):
        local, axes = base.contract_leg(mapping, legs, index)
        expected = {"leg": leg, "samples": 101, **{name+"_indices": values.tolist() for name, values in
                    zip(("motor", "muscle", "edge", "q", "ltm"), axes, strict=True)}}
        base.require(roster[index] == expected, "Frozen engine roster differs from explicit startup ownership")
        prepared.append(local)
    return mapping, legs, prepared


class StartupEvaluator:
    def __init__(self, engine, case, budget, output):
        base.require(engine.problem["samples"] == 101 and engine.problem["stages"] == 8000
            and engine.problem["params"] == case["neutral_coefficients"].size, "Exact startup-only problem required")
        self.engine, self.case, self.budget, self.output = engine, case, budget, output
        self.evaluation_index = 0
    def evaluate(self, c, *, derivative):
        base.require(type(derivative) is bool and base.coefficient_valid(c, interior=True), "Explicit valid startup evaluation required")
        self.budget.request(derivative)
        index, device, primary, entry = self.evaluation_index, None, None, None
        self.evaluation_index += 1
        try:
            device = self.engine.solve(c, derivative=derivative, host=False)
            q = self.engine.host_copy(device["q"], label=f"margin-{index}:q")
            ltm = self.engine.host_copy(device["ltm"], label=f"margin-{index}:ltm")
            entry = {"coefficients": c.copy(), "evaluation": index, "startup_q": q, "startup_ltm": ltm,
                     "qerr": q[1:100]-self.case["target_q"], "lerr": ltm[1:100]-self.case["target_ltm"]}
            if derivative:
                jacobian = self.engine.command_jacobian(device, np.arange(1, 100, dtype=np.int64), host=True)
                entry.update(startup_q_jacobian=jacobian["q"], startup_ltm_jacobian=jacobian["ltm"])
            margin.validate_entry(entry, derivative=derivative)
            entry["receipt"] = copy.deepcopy(device["receipt"])
            persist(self.output, f"eval-{index:03d}", {k: v for k, v in entry.items() if isinstance(v, np.ndarray)}, entry["receipt"])
            self.budget.counts["evaluations_completed"] += 1
            self.budget.deadline.check()
            return entry
        except Exception as error:
            primary = error
            self.budget.counts["evaluations_failed"] += 1
            if isinstance(error, base.BudgetExceeded) and entry is not None and "receipt" in entry:
                error.completed_margin_entry = entry
            base.preserve_primary(error, "Startup evaluation failure raw/cost failed",
                lambda error=error: base.failure_record(error, self.engine, self.output, f"eval-{index:03d}", self.budget.receipt()))
            raise
        finally:
            receipt = device.get("receipt", {}) if device is not None else getattr(primary, "response_receipt", {})
            receipt = receipt.get("input_response_receipt", receipt)
            self.budget.counts["forward_completed_steps"] += int(receipt.get("completed_steps", 0))
            self.budget.counts["derivative_completed_steps"] += int(receipt.get("derivative_completed_steps", 0))


def make_builder(inputs, factory, meter):
    _, _, prepared = inputs
    @contextmanager
    def build(index, case, folder, budget):
        budget.leg_evaluations = 0
        def counted_factory(mapping):
            meter.engine_requests += 1
            base.require(meter.engine_requests <= 6, "At most six startup engines")
            return factory(mapping)
        with base.engine_scope(counted_factory, prepared[index], meter, "margin-"+base.LEGS[index]+"-startup") as engine:
            evaluator = StartupEvaluator(engine, case, budget, folder)
            yield evaluator.evaluate
    return build


def execute_legs(freeze, builder, meter):
    meter.budget = StartupBudget(meter.deadline)
    results = []
    try:
        for index, row in enumerate(freeze["cases"]):
            meter.deadline.check()
            folder = meter.output/row["leg"]
            folder.mkdir()
            with np.load(base.safe_path(row["path"]), allow_pickle=False) as z:
                case = {k: z[k] for k in z.files}
            primary = None
            evaluate = None
            try:
                with builder(index, case, folder, meter.budget) as evaluate:
                    result = margin.run_leg(case, evaluate, persist=lambda name, arrays, receipt, folder=folder:
                        persist(folder, name, arrays, receipt), check=meter.deadline.check, remaining=meter.deadline.remaining)
                    results.append({"leg": row["leg"], **result})
                    meter.case_results = results
                    meter.account(result["cost"])
                    meter.save()
            except Exception as error:
                primary = error
                partial = getattr(error, "margin_leg_result", None)
                if partial is not None:
                    results.append({"leg": row["leg"], **partial})
                    meter.case_results = results
                    base.preserve_primary(error, "Partial startup solver cost aggregation failed", lambda partial=partial: meter.account(partial["cost"]))
                base.preserve_primary(error, "Partial startup cost persistence failed", meter.save)
                raise
            finally:
                evaluate = None
                if primary is None:
                    meter.cleanup()
                else:
                    base.preserve_primary(primary, "Startup cleanup failed", meter.cleanup)
        supported = len(results) == 6 and all(r["f64_success"] and r["f32_success"] for r in results)
        global_gate = None
        if supported:
            qerrors, lerrors = [], []
            for row in freeze["cases"]:
                with np.load(meter.output/row["leg"]/"f32-actual.npz", allow_pickle=False) as z:
                    qerrors.append(z["qerr"])
                    lerrors.append(z["lerr"])
            qe, le = np.concatenate(qerrors, axis=1), np.concatenate(lerrors, axis=1)
            qsse, lsse = margin.sum_square(qe), margin.sum_square(le)
            global_gate = {"q_sse": qsse, "ltm_sse": lsse, "q_peak": float(np.max(abs(qe))), "ltm_peak": float(np.max(abs(le))),
                "pass": bool(qsse <= 4158*(.0162*(1-1e-4))**2 and lsse <= 594*(.009*(1-1e-4))**2
                    and np.max(abs(qe)) <= .036*(1-1e-4) and np.max(abs(le)) <= .025*(1-1e-4))}
            supported = global_gate["pass"]
        return {"run_valid": True, "status": "complete", "cases": results, "initialization_supported": bool(supported),
                "global_margin_gate": global_gate, "candidate_created": False, "full_fit_started": False,
                "body_promoted": False, "F6_exposed": False}
    except Exception as error:
        error.partial_margin_results = results
        base.preserve_primary(error, "Final partial startup cost failed", meter.save)
        raise


def original_run(freeze, original, output, started):
    from tools.banc_m3_source_coefficient_engine import CudaSourceCoefficientEngine
    meter, primary = StartupMeter(output, ScienceDeadline(started)), None
    try:
        base.require(shutil.disk_usage(output).free >= CONTRACT["free_disk_bytes"], "Four GiB free disk required")
        inputs = load_startup_inputs(original, freeze["engine_roster"])
        meter.host_bytes = base.array_bytes(inputs)
        meter.enforce_memory()
        def factory(mapping):
            engine = CudaSourceCoefficientEngine(mapping, forward_implementation="precomputed")
            meter.cp = engine.cp
            return engine
        return execute_legs(freeze, make_builder(inputs, factory, meter), meter)
    except Exception as error:
        primary = error
        base.preserve_primary(error, "Original startup margin failure record failed",
            lambda error=error: base.failure_record(error, None, output, "e281-original", meter.receipt()))
        raise
    finally:
        if primary is None:
            meter.cleanup()
            meter.save()
        else:
            base.preserve_primary(primary, "Final startup margin cleanup failed", meter.cleanup)
            base.preserve_primary(primary, "Final startup margin cost failed", meter.save)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "freeze-sha256", "source-review", "source-review-sha256", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    base.require(not output.exists(), "Fresh startup margin output required")
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
            "cases": getattr(error, "partial_margin_results", []), "initialization_supported": False,
            "candidate_created": False, "full_fit_started": False, "body_promoted": False, "F6_exposed": False}
        base.preserve_primary(error, "Terminal startup result failed", lambda: base.write_json(output/"result.json", result))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
