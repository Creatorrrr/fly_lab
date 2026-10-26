"""E282 frozen joint startup diagnostic; each evaluation owns one new engine."""
from __future__ import annotations

# Secondary persistence/cleanup failures cannot replace the primary exception.
# ruff: noqa: BLE001
import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import banc_m3_joint_startup as joint
from tools import verify_c_banc_m3_startup_margin as inherited

base = inherited.base
SCHEMA, REVIEW_SCHEMA = "flylab.m3.e282.joint-startup-feasibility.v1", "flylab.m3.e282.source-review.v1"
SOURCES = ("tools/banc_m3_joint_startup.py", "tools/verify_c_banc_m3_joint_startup.py",
           "tests/test_c_banc_joint_startup.py", "tests/test_c_banc_joint_startup_runner.py")
REQUIRED_INHERITED_SOURCES = (*inherited.REQUIRED_INHERITED_SOURCES, *inherited.SOURCES)
CONTRACT = {**copy.deepcopy(joint.CONTRACT), "science_wall_s": 40., "cleanup_reserve_s": 15., "supervisor_wall_s": 55.,
    "legs": list(base.LEGS), "parameters": 4980, "edges": 1660, "startup_samples": 101, "max_engines": 36,
    "max_contexts_live": 1, "max_evaluations": 36, "max_derivative_evaluations": 30, "max_solve_calls": 4,
    "max_forward_ticks": 72000, "max_derivative_ticks": 60000, "ram_bytes": 6*1024**3,
    "gpu_additional_bytes": 4*1024**3, "output_bytes": 2*1024**3, "free_disk_bytes": 4*1024**3,
    "fit_calls": 0, "whole_BANC_calls": 0, "physics_steps": 0, "F6_exposed": False,
    "sequence": "six neutral reproductions;max4 joint corrections;one joint f32 round if f64 passed",
    "engine_lifetime": "new startup engine per evaluation;release all references and pool before next creation"}
package_versions = inherited.package_versions


def validate_roster(roster, cases):
    base.require(isinstance(roster, list) and len(roster) == len(cases) == 6, "Production requires six complete leg blocks")
    all_axes = {key: [] for key in ("motor", "muscle", "edge", "q", "ltm")}
    slices, cursor = [], 0
    for row, case, leg in zip(roster, cases, base.LEGS, strict=True):
        base.require(row["leg"] == leg and row["samples"] == 101, "Fixed startup-only leg order required")
        joint.margin.validate_case(case)
        for name, collected in all_axes.items():
            values = row[name+"_indices"]
            base.require(isinstance(values, list) and all(type(v) is int and v >= 0 for v in values)
                and values == sorted(set(values)), "Explicit sorted original axes required")
            collected.extend(values)
        edges = len(row["edge_indices"])
        base.require(case["neutral_coefficients"].shape == (edges, 3) and len(row["q_indices"]) == 7
            and len(row["ltm_indices"]) == 1, "Production case and original edge/q/LTM axes differ")
        slices.append((cursor, cursor+3*edges)); cursor += 3*edges
    for name, expected in (("motor", 391), ("muscle", 96), ("edge", 1660), ("q", 42), ("ltm", 6)):
        base.require(sorted(all_axes[name]) == list(range(expected)), "Production duplicate/missing original axis: "+name)
    base.require(cursor == 4980, "Production requires all original 4980 parameters")
    return {"leg_major_edge_indices": np.array(all_axes["edge"], np.int64), "parameter_slices": np.array(slices, np.int64),
            "q_indices_by_leg": np.array([r["q_indices"] for r in roster], np.int64),
            "ltm_indices_by_leg": np.array([r["ltm_indices"] for r in roster], np.int64)}


def validate_freeze(path, digest, review_path, review_digest):
    freeze, review = base.verified_json(path, digest), base.verified_json(review_path, review_digest)
    base.require(freeze.get("schema") == SCHEMA and freeze.get("contract") == CONTRACT, "Exact E282 schema/contract required")
    sources = freeze.get("approved_sources")
    base.require(isinstance(sources, dict) and set(sources) == set(SOURCES), "Four owned E282 sources required")
    for name, sha in sources.items(): base.require(base.sha256(base.safe_path(name)) == sha, "Owned E282 source changed: "+name)
    base.require(review.get("schema") == REVIEW_SCHEMA and review.get("verdict") == "approved"
        and review.get("approved_sources") == sources and review.get("freeze_sha256") == digest, "Exact E282 source review required")
    pins = freeze.get("inherited_sources")
    base.require(isinstance(pins, dict) and set(REQUIRED_INHERITED_SOURCES) <= set(pins), "All twenty inherited source pins required")
    for name, sha in pins.items(): base.require(base.sha256(base.safe_path(name)) == sha, "Inherited source changed: "+name)
    oldspec, oldreview = freeze["e281_freeze"], freeze["e281_source_review"]
    oldpath, reviewpath = inherited.inherited._verify_spec(oldspec), inherited.inherited._verify_spec(oldreview)
    old, original = inherited.validate_freeze(oldpath, oldspec["sha256"], reviewpath, oldreview["sha256"])
    base.require(all(pins.get(k) == v for k, v in {**old["approved_sources"], **old["inherited_sources"]}.items()), "Exact original E281 source identities required")
    for key in ("inputs", "cases", "engine_roster", "packages", "e280_resource_qualification"):
        base.require(freeze.get(key) == old[key], "Inherited E281 provenance/input changed: "+key)
    base.require(freeze["packages"] == package_versions(), "Installed frozen package versions required")
    result = json.loads(inherited.inherited._verify_spec(freeze["e281_root_result"]).read_text(encoding="utf-8"))
    base.require(result.get("schema") == "flylab.m3.e281.root-original-audit.v1"
        and result.get("verdict") == "valid_diagnostic_initialization_rejected" and result.get("reproduction_exact") is True,
        "Original E281 valid rejection must be preserved")
    cases = []
    for row in freeze["cases"]:
        with np.load(base.safe_path(row["path"]), allow_pickle=False) as z: cases.append({k: z[k] for k in z.files})
    validate_roster(freeze["engine_roster"], cases)
    return freeze, original


class ScienceDeadline(inherited.ScienceDeadline):
    def remaining(self): return 40-(self.clock()-self.started)
    def receipt(self): return {"original_wall_s": self.clock()-self.started, "science_wall_limit_s": 40., "supervisor_wall_limit_s": 55.}


class JointBudget(inherited.StartupBudget):
    def request(self, derivative):
        self.deadline.check()
        if self.counts["evaluations_requested"] >= 36 or (derivative and self.counts["derivative_requested_steps"] >= 60000):
            raise base.BudgetExceeded("joint_startup_evaluations")
        self.leg_evaluations += 1; self.counts["evaluations_requested"] += 1
        self.counts["forward_requested_steps"] += 2000; self.counts["derivative_requested_steps"] += 2000*int(derivative)


class JointMeter(inherited.StartupMeter):
    def __init__(self, *args):
        self.round_events, self.engine_creation = [], joint.metering.phase()
        super().__init__(*args)
    def receipt(self): return {**super().receipt(), "round_events": self.round_events, "engine_creation": self.engine_creation}
    def account(self, cost):
        super().account(cost)
        base.require(self.socp_cost["solve"]["requested"] <= 4, "At most four joint corrections")


def persist(output, folder, name, arrays, receipt):
    if arrays is not None: np.savez(folder/(name+".npz"), **arrays)
    base.write_json(folder/(name+".json"), receipt)
    base.require(sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) <= CONTRACT["output_bytes"], "Joint output bound exceeded")


def make_round_evaluator(inputs, cases, factory, meter):
    _, _, prepared = inputs
    def evaluate_round(cs, *, derivative, label):
        base.require(len(cs) == len(cases) == len(prepared) == 6 and type(derivative) is bool, "Production complete six-leg round required")
        event = {"label": label, "derivative": derivative, "evaluations_requested": 0, "evaluations_completed": 0, "status": "requested"}
        meter.round_events.append(event); rows = []
        try:
            for index, (c, case, mapping) in enumerate(zip(cs, cases, prepared, strict=True)):
                meter.deadline.check(); folder = meter.output/label/base.LEGS[index]; folder.mkdir(parents=True)
                engine, evaluator, primary = None, None, None
                meter.budget.leg_evaluations = 0
                def counted_factory(mapping):
                    meter.engine_requests += 1; base.require(meter.engine_requests <= 36 and meter.active_contexts == 0, "One actual startup engine at a time, at most36")
                    return joint.metering.metered({"constructor": meter.engine_creation}, "constructor", lambda: factory(mapping), meter.deadline.clock)
                try:
                    with base.engine_scope(counted_factory, mapping, meter, label+"-"+base.LEGS[index]) as engine:
                        evaluator = inherited.StartupEvaluator(engine, case, meter.budget, folder)
                        event["evaluations_requested"] += 1
                        entry = evaluator.evaluate(c, derivative=derivative)
                        rows.append(entry); event["evaluations_completed"] += 1
                        persist(meter.output, folder, "actual", {k: v for k, v in entry.items() if isinstance(v, np.ndarray)},
                            {k: v for k, v in entry.items() if not isinstance(v, np.ndarray)})
                except Exception as error:
                    primary = error
                    complete = getattr(error, "completed_margin_entry", None)
                    if complete is not None:
                        rows.append(complete); event["evaluations_completed"] += 1
                        base.preserve_primary(error, "Completed interrupted joint evaluation save failed",
                            lambda complete=complete, folder=folder: persist(meter.output, folder, "actual", {k: v for k, v in complete.items() if isinstance(v, np.ndarray)},
                                {k: v for k, v in complete.items() if not isinstance(v, np.ndarray)}))
                    raise
                finally:
                    # Both caller references must vanish before pool cleanup or the next constructor.
                    evaluator, engine = None, None
                    if primary is None: meter.cleanup()
                    else: base.preserve_primary(primary, "Joint evaluation cleanup failed", meter.cleanup)
            event["status"] = "complete"; meter.save(); return rows
        except Exception as error:
            event.update(status="incomplete" if isinstance(error, base.BudgetExceeded) else "invalid", error=str(error))
            error.partial_joint_round = rows
            base.preserve_primary(error, "Joint round partial cost failed", meter.save)
            raise
    return evaluate_round


def execute_joint(freeze, inputs, cases, factory, meter):
    meter.budget = JointBudget(meter.deadline)
    axes = validate_roster(freeze["engine_roster"], cases)
    persist(meter.output, meter.output, "internal-parameter-roster", axes, {"order": "leg-major", "product_candidate": False})
    evaluator = make_round_evaluator(inputs, cases, factory, meter)
    result, accounted = None, False
    try:
        result = joint.run_joint(cases, evaluator, persist=lambda n, a, r: persist(meter.output, meter.output, n, a, r),
                                 check=meter.deadline.check, remaining=meter.deadline.remaining)
        # Mark before the aggregation call: a failing call may already have
        # charged completed work, so an exception must never retry aggregation.
        accounted = True; meter.account(result["cost"]); meter.save()
        return {**result, "candidate_created": False, "full_fit_started": False, "body_promoted": False, "F6_exposed": False}
    except Exception as error:
        partial = getattr(error, "joint_result", result)
        if partial is not None:
            partial.update(run_valid=False, initialization_supported=False,
                           status="incomplete" if isinstance(error, base.BudgetExceeded) else "invalid")
            error.joint_result = partial
            if not accounted:
                base.preserve_primary(error, "Partial joint solver cost aggregation failed", lambda: meter.account(partial["cost"]))
        base.preserve_primary(error, "Partial joint final cost failed", meter.save)
        raise
    finally:
        evaluator = None


def original_run(freeze, original, output, started):
    from tools.banc_m3_source_coefficient_engine import CudaSourceCoefficientEngine
    meter, primary = JointMeter(output, ScienceDeadline(started)), None
    try:
        base.require(shutil.disk_usage(output).free >= CONTRACT["free_disk_bytes"], "Four GiB free disk required")
        inputs = inherited.load_startup_inputs(original, freeze["engine_roster"])
        cases = []
        for row in freeze["cases"]:
            with np.load(base.safe_path(row["path"]), allow_pickle=False) as z: cases.append({k: z[k] for k in z.files})
        meter.host_bytes = base.array_bytes((inputs, cases)); meter.enforce_memory()
        def factory(mapping):
            engine = CudaSourceCoefficientEngine(mapping, forward_implementation="precomputed"); meter.cp = engine.cp; return engine
        return execute_joint(freeze, inputs, cases, factory, meter)
    except Exception as error:
        primary = error
        base.preserve_primary(error, "Original joint failure receipt failed", lambda error=error: base.failure_record(error, None, output, "e282-original", meter.receipt()))
        raise
    finally:
        if primary is None: meter.cleanup(); meter.save()
        else:
            base.preserve_primary(primary, "Final joint cleanup failed", meter.cleanup)
            base.preserve_primary(primary, "Final joint cost failed", meter.save)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "freeze-sha256", "source-review", "source-review-sha256", "output"): parser.add_argument("--"+key, required=True)
    args = parser.parse_args(argv); output = Path(args.output).resolve(); base.require(not output.exists(), "Fresh E282 output required")
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=False)
    base.write_json(output/"cost.json", {"phase": "freeze_validation", "original_calls": 0, "solver_calls": 0})
    try:
        freeze, original = validate_freeze(Path(args.freeze), args.freeze_sha256, Path(args.source_review), args.source_review_sha256)
        base.write_json(output/"result.json", original_run(freeze, original, output, started)); return 0
    except Exception as error:
        result = {"run_valid": False, "status": "incomplete" if isinstance(error, base.BudgetExceeded) else "invalid", "error": str(error),
            "exception": type(error).__name__, "notes": getattr(error, "__notes__", []), "joint_result": getattr(error, "joint_result", None),
            "initialization_supported": False, "candidate_created": False, "full_fit_started": False, "body_promoted": False, "F6_exposed": False}
        base.preserve_primary(error, "Terminal joint result failed", lambda: base.write_json(output/"result.json", result)); return 1


if __name__ == "__main__": raise SystemExit(main())
