"""E284 frozen RF-only capacity diagnostic; startup only, new engine per evaluation."""
from __future__ import annotations

# Preserve the original exception when cleanup or storage also fails.
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
from tools import banc_m3_rf_capacity as capacity
from tools import verify_c_banc_m3_startup_continuation as prior
from tools import verify_c_banc_m3_startup_margin as startup

base = capacity.base
SCHEMA, REVIEW_SCHEMA = "flylab.m3.e284.rf-capacity.v1", "flylab.m3.e284.source-review.v1"
SOURCES = ("tools/banc_m3_rf_capacity.py", "tools/verify_c_banc_m3_rf_capacity.py", "tests/test_c_banc_rf_capacity.py")
REQUIRED_INHERITED_SOURCES = (*prior.REQUIRED_INHERITED_SOURCES, *prior.SOURCES)
CONTRACT = {**copy.deepcopy(capacity.CONTRACT), "rf_index": 3, "leg": "RF", "parameters": 783, "edges": 261,
    "startup_samples": 101, "max_engines": 6, "max_contexts_live": 1, "max_evaluations": 6, "max_derivative_evaluations": 5,
    "max_solve_calls": 4, "max_forward_ticks": 12000, "max_derivative_ticks": 10000,
    "science_wall_s": 20., "cleanup_reserve_s": 10., "supervisor_wall_s": 30., "ram_bytes": 4*1024**3,
    "gpu_additional_bytes": 2*1024**3, "output_bytes": 1024**3, "free_disk_bytes": 2*1024**3,
    "generated_cpu_wall_s": 120., "generated_tiny_solves": 20, "generated_max_variables": 6,
    "engine_lifetime": "new RF startup engine per evaluation;release engine/evaluator references before cleanup and next constructor",
    "whole_BANC_calls": 0, "physics_steps": 0, "fit_calls": 0, "F6_exposed": False,
    "candidate_created": False, "initialization_supported": False, "global_norm_supported": False}
package_versions = startup.package_versions
verify_spec = prior.inherited.inherited.inherited._verify_spec


def validate_rf_roster(row, case):
    capacity.margin.validate_case(case)
    names = ("motor", "muscle", "edge", "q", "ltm")
    base.require(isinstance(row, dict) and set(row) == {"leg", "samples", *[k+"_indices" for k in names]}
        and row["leg"] == "RF" and row["samples"] == 101, "Exact production RF startup roster required")
    for name, expected in zip(names, (70, 17, 261, 7, 1), strict=True):
        values = row[name+"_indices"]
        base.require(isinstance(values, list) and len(values) == expected and values == sorted(set(values))
            and all(type(v) is int and v >= 0 for v in values), "Explicit original RF index axes required: "+name)
    base.require(case["neutral_coefficients"].shape == (261, 3), "Production RF requires 261 edges/783 parameters")
    return {key: np.array(value, np.int64) for key, value in row.items() if key.endswith("_indices")}


def validate_prior_evidence(audit, proof, sensitivity):
    base.require(audit.get("schema") == "flylab.m3.e283.root-original-audit.v1"
        and audit.get("verdict") == "valid_diagnostic_initialization_rejected"
        and all(audit.get(k) is True for k in ("reproduction_exact", "sparse_equations_exact", "independent_primal_dual_checks_pass"))
        and audit.get("terminal_status") == "stage_0_two_corrections_rejected"
        and all(audit.get(k) is False for k in ("stage1_reached", "f32_evaluated", "candidate_created", "F6_exposed"))
        and audit.get("engine_requests") == audit.get("startup_evaluations") == 18
        and audit.get("solver_calls") == 2, "E283 actual intermediate-stage valid rejection must remain visible")
    labels = ("neutral", "stage-0-joint-00-actual", "stage-0-joint-01-actual")
    rows = proof.get("records", [])
    base.require(proof.get("schema") == "flylab.m3.e283.saved-fresh-rf-linear-bound.v1"
        and all(proof.get(k) == 0 for k in ("model_calls", "solver_calls", "original_calls"))
        and [r.get("label") for r in rows] == list(labels)
        and all(r.get("dual_nonnegative") is True and r.get("final_linear_infeasibility_supported") is True
            and np.isfinite(r.get("max_possible_weighted_slack_at_final_peak", np.nan))
            and r["max_possible_weighted_slack_at_final_peak"] < 0
            and np.isfinite(r.get("certified_linear_peak_lower_bound", np.nan))
            and r["certified_linear_peak_lower_bound"] > .036*(1-1e-4) for r in rows), "Saved RF fresh-J small-box infeasibility evidence required")
    verify_spec(proof.get("dual_source"))
    rows = sensitivity.get("records", [])
    base.require(sensitivity.get("schema") == "flylab.m3.e283.rf-radius-sensitivity.v1"
        and all(sensitivity.get(k) == 0 for k in ("model_calls", "solver_calls", "original_calls"))
        and [r.get("label") for r in rows] == [labels[0], labels[2]]
        and all(all(np.isfinite(r.get(k, np.nan)) and r[k] < 0 for k in ("bound_at_radius_005", "bound_at_radius_010"))
            and r.get("larger_radius_certificate_inconclusive_not_feasible") is False for r in rows), "Saved unchanged RF dual at radius .05 and .1 required")
    return {"e283_valid_rejection_preserved": True, "rf_small_box_linear_infeasibility_preserved": True,
            "scope": "fixed saved local J only;not a nonlinear family impossibility proof"}


def validate_freeze(path, digest, review_path, review_digest):
    freeze, review = base.verified_json(path, digest), base.verified_json(review_path, review_digest)
    base.require(freeze.get("schema") == SCHEMA and freeze.get("contract") == CONTRACT, "Exact E284 contract required")
    sources = freeze.get("approved_sources")
    base.require(isinstance(sources, dict) and set(sources) == set(SOURCES), "Exactly three E284 source pins required")
    for name, sha in sources.items(): base.require(base.sha256(base.safe_path(name)) == sha, "E284 source changed: "+name)
    base.require(review.get("schema") == REVIEW_SCHEMA and review.get("verdict") == "approved"
        and review.get("approved_sources") == sources and review.get("freeze_sha256") == digest, "Exact E284 source review required")
    pins = freeze.get("inherited_sources")
    base.require(isinstance(pins, dict) and set(REQUIRED_INHERITED_SOURCES) <= set(pins), "All26 inherited source identities required")
    for name, sha in pins.items(): base.require(base.sha256(base.safe_path(name)) == sha, "Inherited source changed: "+name)
    fs, rs = freeze["e283_freeze"], freeze["e283_source_review"]
    old, original = prior.validate_freeze(verify_spec(fs), fs["sha256"], verify_spec(rs), rs["sha256"])
    base.require(pins == {**old["inherited_sources"], **old["approved_sources"]}, "Exact E283 source chain required")
    for key in ("inputs", "cases", "engine_roster", "packages", "e280_resource_qualification"):
        base.require(freeze.get(key) == old[key], "Inherited startup provenance changed: "+key)
    base.require(freeze["packages"] == package_versions(), "Installed frozen package versions required")
    audit, proof, sensitivity = (json.loads(verify_spec(freeze[k]).read_text(encoding="utf-8")) for k in
                               ("e283_root_result", "e283_rf_linear_bound", "e283_radius_sensitivity"))
    validate_prior_evidence(audit, proof, sensitivity)
    base.require(type(freeze.get("rf_index")) is int and freeze["rf_index"] == 3
        and freeze.get("rf_case") == old["cases"][3] and freeze.get("rf_engine_roster") == old["engine_roster"][3],
        "RF must be original roster index3 with identical frozen case")
    with np.load(verify_spec({k: freeze["rf_case"][k] for k in ("path", "sha256")}), allow_pickle=False) as z:
        case = {k: z[k] for k in z.files}
    validate_rf_roster(freeze["rf_engine_roster"], case)
    return freeze, original


class ScienceDeadline(startup.ScienceDeadline):
    def remaining(self): return 20-(self.clock()-self.started)
    def receipt(self): return {"original_wall_s": self.clock()-self.started, "science_wall_limit_s": 20., "supervisor_wall_limit_s": 30.}


class CapacityBudget(startup.StartupBudget):
    def request(self, derivative):
        self.deadline.check()
        base.require(type(derivative) is bool, "Explicit RF derivative mode required")
        if self.counts["evaluations_requested"] >= 6 or (derivative and self.counts["derivative_requested_steps"] >= 10000):
            raise base.BudgetExceeded("rf_capacity_evaluations")
        self.leg_evaluations += 1; self.counts["evaluations_requested"] += 1
        self.counts["forward_requested_steps"] += 2000; self.counts["derivative_requested_steps"] += 2000*int(derivative)


class CapacityMeter(startup.StartupMeter):
    def __init__(self, *args):
        self.evaluation_events, self.engine_creation = [], capacity.metering.phase()
        super().__init__(*args)
    def receipt(self): return {**super().receipt(), "evaluation_events": self.evaluation_events, "engine_creation": self.engine_creation, "rf_only": True}
    def enforce_memory(self):
        base.require(self.peak_estimate <= CONTRACT["ram_bytes"] and (self.peak_rss is None or self.peak_rss <= CONTRACT["ram_bytes"]), "RF RAM bound exceeded")
        live = [r for r in self.memory_events if r.get("live_gpu_estimate") is not None]
        if live: base.require(live[-1]["live_gpu_estimate"] <= CONTRACT["gpu_additional_bytes"], "RF additional GPU bound exceeded")
    def account(self, cost):
        super().account(cost); base.require(self.socp_cost["solve"]["requested"] <= 4, "At most4 RF correction solves")


def persist(output, name, arrays, receipt):
    if arrays is not None: np.savez(output/(name+".npz"), **arrays)
    base.write_json(output/(name+".json"), receipt)
    base.require(sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) <= CONTRACT["output_bytes"], "RF output bound exceeded")


def load_rf_inputs(original, roster):
    mapping = {k: base.load_ref(original, ref) for k, ref in original["contexts"]["startup"]["mapping"].items()}
    mapping["stage_order"] = tuple(mapping["stage_order"].tolist())
    legs = {k: base.load_ref(original, ref) for k, ref in original["legs"].items()}
    base.validate_leg_ownership(mapping, legs)
    local, axes = base.contract_leg(mapping, legs, 3)
    expected = {"leg": "RF", "samples": 101, **{k+"_indices": v.tolist() for k, v in
                zip(("motor", "muscle", "edge", "q", "ltm"), axes, strict=True)}}
    base.require(roster == expected, "RF roster differs from original explicit leg ownership")
    return mapping, legs, local


def make_evaluator(mapping, case, factory, meter):
    def evaluate(c, *, derivative, label):
        meter.deadline.check(); folder = meter.output/label; folder.mkdir()
        event = {"label": label, "derivative": derivative, "status": "requested"}; meter.evaluation_events.append(event)
        engine, evaluator, primary, entry = None, None, None, None
        def counted_factory(mapping):
            meter.engine_requests += 1
            base.require(meter.engine_requests <= 6 and meter.active_contexts == 0, "At most6 engines;one actual RF engine at a time")
            return capacity.metering.metered({"constructor": meter.engine_creation}, "constructor", lambda: factory(mapping), meter.deadline.clock)
        try:
            with base.engine_scope(counted_factory, mapping, meter, "rf-"+label) as engine:
                evaluator = startup.StartupEvaluator(engine, case, meter.budget, folder)
                entry = evaluator.evaluate(c, derivative=derivative); event["status"] = "complete"
            return entry
        except Exception as error:
            primary = error; event.update(status="incomplete" if isinstance(error, base.BudgetExceeded) else "invalid", error=str(error))
            if entry is not None: error.completed_margin_entry = entry
            raise
        finally:
            evaluator, engine = None, None
            if primary is None:
                try: meter.cleanup()
                except Exception as error:
                    event.update(status="invalid", error=str(error))
                    if entry is not None: error.completed_margin_entry = entry
                    raise
            else: base.preserve_primary(primary, "RF evaluation cleanup failed", meter.cleanup)
    return evaluate


def execute_capacity(freeze, inputs, case, factory, meter):
    meter.budget = CapacityBudget(meter.deadline)
    axes = validate_rf_roster(freeze["rf_engine_roster"], case)
    persist(meter.output, "rf-original-roster", axes, {"rf_index": 3, "product_candidate": False})
    base.write_json(meter.output/"provenance.json", {"schema": SCHEMA, "rf_only": True,
        "e283_valid_rejection_preserved": True, "neutral_trust_applied": False, "step_trust_applied": False,
        "prior_evidence": {k: freeze[k] for k in ("e283_freeze", "e283_source_review", "e283_root_result", "e283_rf_linear_bound", "e283_radius_sensitivity", "e280_resource_qualification")},
        "inherited_sources": freeze["inherited_sources"], "approved_sources": freeze["approved_sources"]})
    evaluator = make_evaluator(inputs[2], case, factory, meter); result, accounted = None, False
    try:
        result = capacity.run_capacity(case, evaluator, persist=lambda n, a, r: persist(meter.output, n, a, r),
            check=meter.deadline.check, remaining=meter.deadline.remaining)
        accounted = True; meter.account(result["cost"]); meter.save()
        return {**result, "candidate_created": False, "full_fit_started": False, "body_promoted": False, "F6_exposed": False}
    except Exception as error:
        partial = getattr(error, "capacity_result", result)
        if partial is not None:
            partial.update(run_valid=False, rf_capacity_supported=False, status="incomplete" if isinstance(error, base.BudgetExceeded) else "invalid")
            error.capacity_result = partial
            if not accounted: base.preserve_primary(error, "Partial RF solver cost aggregation failed", lambda: meter.account(partial["cost"]))
        base.preserve_primary(error, "Partial RF final cost failed", meter.save)
        raise
    finally: evaluator = None


def original_run(freeze, original, output, started):
    from tools.banc_m3_source_coefficient_engine import CudaSourceCoefficientEngine
    meter, primary = CapacityMeter(output, ScienceDeadline(started)), None
    try:
        base.require(shutil.disk_usage(output).free >= CONTRACT["free_disk_bytes"], "Two GiB free disk required")
        inputs = load_rf_inputs(original, freeze["rf_engine_roster"])
        with np.load(base.safe_path(freeze["rf_case"]["path"]), allow_pickle=False) as z: case = {k: z[k] for k in z.files}
        meter.host_bytes = base.array_bytes((inputs, case)); meter.enforce_memory()
        def factory(mapping):
            engine = CudaSourceCoefficientEngine(mapping, forward_implementation="precomputed"); meter.cp = engine.cp; return engine
        return execute_capacity(freeze, inputs, case, factory, meter)
    except Exception as error:
        primary = error
        base.preserve_primary(error, "Original RF failure receipt failed", lambda error=error: base.failure_record(error, None, output, "e284-original", meter.receipt()))
        raise
    finally:
        if primary is None: meter.cleanup(); meter.save()
        else:
            base.preserve_primary(primary, "Final RF cleanup failed", meter.cleanup)
            base.preserve_primary(primary, "Final RF cost failed", meter.save)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "freeze-sha256", "source-review", "source-review-sha256", "output"): parser.add_argument("--"+key, required=True)
    args = parser.parse_args(argv); output = Path(args.output).resolve(); base.require(not output.exists(), "Fresh E284 output required")
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=False)
    base.write_json(output/"cost.json", {"phase": "freeze_validation", "original_calls": 0, "solver_calls": 0})
    try:
        freeze, original = validate_freeze(Path(args.freeze), args.freeze_sha256, Path(args.source_review), args.source_review_sha256)
        base.write_json(output/"result.json", original_run(freeze, original, output, started)); return 0
    except Exception as error:
        result = {"run_valid": False, "status": "incomplete" if isinstance(error, base.BudgetExceeded) else "invalid", "error": str(error),
            "exception": type(error).__name__, "notes": getattr(error, "__notes__", []), "capacity_result": getattr(error, "capacity_result", None),
            "rf_capacity_supported": False, "initialization_supported": False, "global_norm_supported": False,
            "candidate_created": False, "full_fit_started": False, "body_promoted": False, "F6_exposed": False}
        base.preserve_primary(error, "Terminal RF result failed", lambda: base.write_json(output/"result.json", result)); return 1


if __name__ == "__main__": raise SystemExit(main())
