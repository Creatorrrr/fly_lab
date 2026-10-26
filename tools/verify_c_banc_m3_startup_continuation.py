"""E283 bounded actual-J continuation; constraints scale, raw science does not."""
from __future__ import annotations

# Secondary storage and cleanup failures must preserve the primary exception.
# ruff: noqa: BLE001
import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import clarabel
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import banc_m3_joint_startup as joint
from tools import verify_c_banc_m3_joint_startup as inherited

base, margin, metering = joint.base, joint.margin, joint.metering
SCHEMA, REVIEW_SCHEMA = "flylab.m3.e283.startup-continuation.v1", "flylab.m3.e283.source-review.v1"
SOURCES = ("tools/verify_c_banc_m3_startup_continuation.py", "tests/test_c_banc_startup_continuation.py")
REQUIRED_INHERITED_SOURCES = (*inherited.REQUIRED_INHERITED_SOURCES, *inherited.SOURCES)
STAGES = ({"stage": 0, "q_scale": 18/19, "q_peak_limit": .038*(1-1e-4),
           "global_q_sse_limit": 4158*(.0171*(1-1e-4))**2, "max_corrections": 2},
          {"stage": 1, "q_scale": 1., "q_peak_limit": .036*(1-1e-4),
           "global_q_sse_limit": 4158*(.0162*(1-1e-4))**2, "max_corrections": 2})
CONTRACT = {**copy.deepcopy(inherited.CONTRACT), "stages": copy.deepcopy(list(STAGES)),
    "sequence": "neutral reproduction;stage0 max2;stage1 max2 using last actual fresh J;one f32 round only if final f64 passed",
    "stage_acceptance": "normalized strict AND actual-unit strict;zero tolerance;fixed original neutral trust",
    "constraint_view": "qerr and qJ only scaled;raw q/LTM/J unchanged;no transition evaluation/reset/clip/retry",
    "generated_cpu_wall_s": 120., "generated_tiny_evaluations": 20, "generated_max_variables": 6}
ScienceDeadline, JointBudget, JointMeter = inherited.ScienceDeadline, inherited.JointBudget, inherited.JointMeter
make_round_evaluator, validate_roster = inherited.make_round_evaluator, inherited.validate_roster
persist = inherited.persist


def stage_metadata(stage):
    base.require(type(stage) is int and stage in (0, 1), "Fixed continuation stage 0 or 1 required")
    return {**STAGES[stage], "ltm_peak_limit": .025*(1-1e-4),
            "per_leg_ltm_sse_limit": 99*(.009*(1-1e-4))**2,
            "linear_reserve_constraint_units": 1e-6, "raw_units_preserved": True}


def stage_view(entries, stage):
    """Separate q constraint arrays; never modify raw observed arrays."""
    scale = stage_metadata(stage)["q_scale"]
    views = []
    for entry in entries:
        margin.validate_entry(entry, derivative="startup_q_jacobian" in entry)
        view = dict(entry); view["qerr"] = entry["qerr"]*scale
        if "startup_q_jacobian" in entry: view["startup_q_jacobian"] = entry["startup_q_jacobian"]*scale
        views.append(view)
    return views


def strict_stage(entries, origins, stage):
    """Normalized and original-unit strict conditions, with zero tolerance."""
    limits = stage_metadata(stage)
    normalized = joint.strict_joint(stage_view(entries, stage), origins)
    actual = joint.strict_joint(entries, origins)
    checks = {}
    for index, row in enumerate(actual["per_leg"]):
        checks[str(index)] = bool(row["coefficient_checks"]["pass"]
            and row["q_peak"] <= limits["q_peak_limit"] and row["ltm_peak"] <= limits["ltm_peak_limit"]
            and row["ltm_sse"] <= limits["per_leg_ltm_sse_limit"])
    checks["global_q_sse"] = actual["q_sse"] <= limits["global_q_sse_limit"]
    actual.update(checks=checks, q_sse_limit=limits["global_q_sse_limit"])
    actual["pass"] = bool(all(checks.values()))
    return {"pass": bool(normalized["pass"] and actual["pass"]), "normalized": normalized,
            "actual": actual, "metadata": limits}


def run_continuation(cases, evaluate_round, *, persist, check=lambda: None, remaining=lambda: 4.,
              solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter, solve=joint.solve_joint):
    base.require(1 <= len(cases) <= 6, "Complete joint cases required")
    for case in cases: margin.validate_case(case)
    origins = [case["neutral_coefficients"] for case in cases]
    cost = {k: metering.phase() for k in ("round", "constructor", "solve", "certificate")}
    cost.update(iterations=0, iterations_unknown_calls=0, fresh_jacobian_rounds_used=0)
    result = {"status": "requested", "run_valid": False, "f64_success": False, "f32_success": False,
              "initialization_supported": False, "reproduction": [], "trajectory": [], "corrections": [], "stages": [], "cost": cost}
    primary = None
    def observe(cs, label, derivative):
        check()
        rows = None
        try:
            rows = metering.metered(cost, "round", lambda: evaluate_round(cs, derivative=derivative, label=label), wall_clock)
            base.require(len(rows) == len(cases), "Incomplete joint actual evaluation round")
            for entry, c, case in zip(rows, cs, cases, strict=True):
                margin.validate_entry(entry, derivative=derivative)
                base.require(np.array_equal(entry["coefficients"], c)
                    and np.array_equal(entry["qerr"], entry["startup_q"][1:100]-case["target_q"])
                    and np.array_equal(entry["lerr"], entry["startup_ltm"][1:100]-case["target_ltm"]), "Joint coefficient/error/target mismatch")
            check(); return rows
        except Exception as error:
            partial = rows if rows is not None else getattr(error, "partial_joint_round", None)
            if partial is None and getattr(error, "restoration_phase", None) == "round":
                partial = getattr(error, "phase_partial_result", None)
            if isinstance(partial, list):
                result["partial_round"] = {"label": label, "completed_blocks": len(partial), "expected_blocks": len(cases)}
                for index, entry in enumerate(partial):
                    arrays = {k: v for k, v in entry.items() if isinstance(v, np.ndarray)}
                    base.preserve_primary(error, "Partial joint round persistence failed",
                        lambda index=index, arrays=arrays: persist(f"{label}-partial-{index:02d}", arrays, {"derivative": derivative}))
            raise
    def account(receipt):
        for key in ("constructor", "solve", "certificate"):
            for field in ("requested", "completed", "failed"): cost[key][field] += receipt[key][field]
            if not receipt[key]["wall_known"]: cost[key].update(wall_s=None, wall_known=False)
            elif cost[key]["wall_s"] is not None: cost[key]["wall_s"] += receipt[key]["wall_s"]
        if receipt["iterations_known"]: cost["iterations"] += receipt["iterations"]
        elif receipt["solve"]["requested"]: cost["iterations_unknown_calls"] += 1
    try:
        current = observe([o.copy() for o in origins], "neutral", True)
        for row, case in zip(current, cases, strict=True):
            gates = {actual: base.numerical_gate(row[actual], case[saved], tolerance) for actual, saved, tolerance in
                (("startup_q", "neutral_startup_q", 1e-10), ("startup_ltm", "neutral_startup_ltm", 1e-10),
                 ("startup_q_jacobian", "neutral_q_jacobian", 1e-9), ("startup_ltm_jacobian", "neutral_ltm_jacobian", 1e-9))}
            result["reproduction"].append(gates)
        persist("reproduction", None, result["reproduction"])
        if not all(g["pass"] for row in result["reproduction"] for g in row.values()):
            raise metering.ReproductionError("Joint neutral q/LTM/J reproduction failed")
        for stage in range(2):
            judgement = strict_stage(current, origins, stage); stage_pass = judgement["pass"]
            stage_record = {**stage_metadata(stage), "entry_judgement": judgement, "corrections": 0, "pass": False}
            result["stages"].append(stage_record); result["trajectory"].append(judgement)
            for iteration in range(2):
                if stage_pass: break
                stage_record["corrections"] += 1
                check(); problem = joint.joint_problem(stage_view(current, stage), origins); cost["fresh_jacobian_rounds_used"] += 1
                prefix = f"stage-{stage}-joint-{iteration:02d}"
                persist(prefix+"-problem", joint.problem_arrays(problem), {**stage_metadata(stage), "iteration": iteration, "global_correction": len(result["corrections"])})
                check(); limit = min(4., remaining())
                if limit <= 0: raise base.BudgetExceeded("original")
                try:
                    solved = solve(problem, limit, solver_factory=solver_factory, wall_clock=wall_clock); account(solved["receipt"])
                except Exception as error:
                    receipt = getattr(error, "joint_solve_receipt", None)
                    if receipt is not None:
                        account(receipt); result["corrections"].append({"receipt": receipt, "stage": stage, "iteration": iteration, "status": "execution_failure"})
                        raw = getattr(error, "partial_joint_solution", None)
                        arrays = {} if raw is None else {k: np.asarray(getattr(raw, attr), np.float64) for k, attr in (("correction", "x"), ("solver_s", "s"), ("solver_y", "z"))}
                        base.preserve_primary(error, "Partial joint solve persistence failed", lambda prefix=prefix, arrays=arrays, receipt=receipt: persist(prefix+"-solve", arrays, receipt))
                    raise
                arrays = {k: solved[k] for k in ("correction", "solver_s", "solver_y", "primal_slack") if solved[k] is not None}
                if solved["residuals"] is not None: arrays.update({"residual_"+k: v for k, v in solved["residuals"].items()})
                persist(prefix+"-solve", arrays, solved["receipt"])
                row = {"stage": stage, "iteration": iteration, "receipt": solved["receipt"], "actual_evaluated": False}; result["corrections"].append(row)
                if not solved["valid"]:
                    result["status"] = "uncertified_joint_correction"; break
                begin, cs = 0, []
                for entry in current:
                    end = begin+entry["coefficients"].size
                    cs.append(entry["coefficients"]+solved["correction"][begin:end].reshape(entry["coefficients"].shape)); begin = end
                bounds = margin.coefficient_checks(np.concatenate(cs), np.concatenate(origins), solved["correction"])
                new_values = np.concatenate(cs).ravel()
                inward_pass = bool(np.all(new_values >= problem["coefficient_bounds"][:, 0])
                                   and np.all(new_values <= problem["coefficient_bounds"][:, 1]))
                original_pass = bounds["pass"]
                bounds.update(original_coefficient_checks_pass=original_pass, inward_coefficient_interval=inward_pass)
                bounds["pass"] = bool(original_pass and inward_pass)
                persist(prefix+"-coefficients", {"coefficients_leg_major": np.concatenate(cs)}, bounds)
                if not original_pass:
                    result["status"] = "strict_coefficient_trust_or_correction_rejection"; break
                if not inward_pass:
                    result["status"] = "strict_inward_coefficient_interval_rejection"; break
                current = observe(cs, prefix+"-actual", True); row["actual_evaluated"] = True
                judgement = strict_stage(current, origins, stage); result["trajectory"].append(judgement); row["judgement"] = judgement
                persist(prefix+"-judgement", None, row); stage_pass = judgement["pass"]
            stage_record.update(final_judgement=judgement)
            stage_record["pass"] = stage_pass
            persist(f"stage-{stage}-result", None, stage_record)
            if not stage_pass:
                if result["status"] == "requested": result["status"] = f"stage_{stage}_two_corrections_rejected"
                break
            # Carry the last actual response and fresh J without evaluation/reset.
            if stage == 1: result["f64_success"] = True
        if result["f64_success"]:
            cs32 = [entry["coefficients"].astype(np.float32) for entry in current]; cs = [c.astype(np.float64) for c in cs32]
            bounds = margin.coefficient_checks(np.concatenate(cs), np.concatenate(origins))
            persist("f32-coefficients", {"coefficients32_leg_major": np.concatenate(cs32), "coefficients_leg_major": np.concatenate(cs)}, bounds)
            if not bounds["pass"]: result["status"] = "f32_coefficient_or_trust_rejection_before_evaluation"
            else:
                final = observe(cs, "f32-actual", False); judgement = strict_stage(final, origins, 1)
                result.update(f32_success=judgement["pass"], f32_judgement=judgement, status="supported" if judgement["pass"] else "f32_joint_startup_rejection")
        result.update(run_valid=True, initialization_supported=bool(result["f64_success"] and result["f32_success"]))
    except Exception as error:
        primary = error; result.update(run_valid=False, status="incomplete" if isinstance(error, base.BudgetExceeded) else "invalid", error=str(error), exception=type(error).__name__)
    if primary is None:
        try: persist("continuation-result", None, result)
        except Exception as error:
            result.update(run_valid=False, initialization_supported=False, status="invalid", error=str(error), exception=type(error).__name__)
            error.joint_result, error.abort_run = result, True; raise
    else:
        primary.joint_result = result
        saved = base.preserve_primary(primary, "Joint failure persistence failed", lambda: persist("continuation-result", None, result) or True)
        if saved is not True: primary.abort_run = True
        raise primary
    return result


def validate_prior_evidence(audit, proof):
    base.require(audit.get("schema") == "flylab.m3.e282.root-original-audit.v1"
        and audit.get("verdict") == "valid_diagnostic_initialization_rejected"
        and audit.get("reproduction_exact") is True and audit.get("sparse_original_problem_exact") is True
        and audit.get("coefficient_evaluations") == audit.get("engine_requests") == 6
        and audit.get("solver_calls") == 1 and audit.get("candidate_created") is False,
        "E282 actual neutral-only valid rejection required")
    rows = audit.get("dual_infeasibility_checks", [])
    base.require(any(r.get("label") == "frozen_joint_problem"
        and r.get("bounded_linear_infeasibility_supported") is True
        and np.isfinite(r.get("max_possible_dual_slack_in_box", np.nan))
        and r["max_possible_dual_slack_in_box"] < 0 for r in rows), "E282 frozen linear infeasibility audit required")
    base.require(proof.get("schema") == "flylab.m3.e282.peak-only-infeasibility.v1"
        and all(proof.get(k) == 0 for k in ("solver_calls", "model_calls", "original_calls")), "Passive RF-only proof required")
    rf = [r for r in proof.get("proofs", []) if r.get("label") == "RF_q_peaks_and_coefficient_only"]
    base.require(len(rf) == 1 and rf[0].get("dual_nonnegative") is True
        and rf[0].get("certified_linear_peak_lower_bound") == .036805003120814854
        and rf[0].get("upper_at_strict_actual_peak", 0) < 0
        and rf[0].get("scope") == "Original f64 neutral +/-0.05 and coefficient/triple constraints, no norm or LTM requirements; neutral Jacobian only.",
        "Fixed RF-only neutral linear proof required")
    return {"e282_valid_rejection_preserved": True, "neutral_evaluations": 6, "solver_calls": 1,
            "rf_only_linear_peak_lower_bound": rf[0]["certified_linear_peak_lower_bound"],
            "scope": rf[0]["scope"], "nonlinear_impossibility_claim": False}


def validate_freeze(path, digest, review_path, review_digest):
    freeze, review = base.verified_json(path, digest), base.verified_json(review_path, review_digest)
    base.require(freeze.get("schema") == SCHEMA and freeze.get("contract") == CONTRACT, "Exact E283 contract required")
    sources = freeze.get("approved_sources")
    base.require(isinstance(sources, dict) and set(sources) == set(SOURCES), "Exactly two E283 sources required")
    for name, sha in sources.items(): base.require(base.sha256(base.safe_path(name)) == sha, "E283 source changed: "+name)
    base.require(review.get("schema") == REVIEW_SCHEMA and review.get("verdict") == "approved"
        and review.get("approved_sources") == sources and review.get("freeze_sha256") == digest, "Exact E283 source review required")
    pins = freeze.get("inherited_sources")
    base.require(isinstance(pins, dict) and set(REQUIRED_INHERITED_SOURCES) <= set(pins), "All24 inherited source identities required")
    for name, sha in pins.items(): base.require(base.sha256(base.safe_path(name)) == sha, "Inherited source changed: "+name)
    verify = inherited.inherited.inherited._verify_spec
    fs, rs = freeze["e282_freeze"], freeze["e282_source_review"]
    old, original = inherited.validate_freeze(verify(fs), fs["sha256"], verify(rs), rs["sha256"])
    base.require(pins == {**old["inherited_sources"], **old["approved_sources"]}, "Exact E282 source chain required")
    for key in ("inputs", "cases", "engine_roster", "packages", "e280_resource_qualification"):
        base.require(freeze.get(key) == old[key], "Inherited startup inputs/provenance changed: "+key)
    audit = json.loads(verify(freeze["e282_root_result"]).read_text(encoding="utf-8"))
    proof = json.loads(verify(freeze["e282_rf_peak_proof"]).read_text(encoding="utf-8"))
    validate_prior_evidence(audit, proof)
    return freeze, original


def execute_continuation(freeze, inputs, cases, factory, meter):
    meter.budget = JointBudget(meter.deadline)
    axes = validate_roster(freeze["engine_roster"], cases)
    persist(meter.output, meter.output, "internal-parameter-roster", axes, {"order": "leg-major", "product_candidate": False})
    base.write_json(meter.output/"provenance.json", {"schema": SCHEMA,
        "e282_valid_rejection_preserved": True, "prior_evidence": {k: freeze[k] for k in
            ("e282_freeze", "e282_source_review", "e282_root_result", "e282_rf_peak_proof")},
        "inherited_sources": freeze["inherited_sources"], "approved_sources": freeze["approved_sources"],
        "candidate_created": False, "F6_exposed": False})
    evaluator = make_round_evaluator(inputs, cases, factory, meter)
    result, accounted = None, False
    try:
        result = run_continuation(cases, evaluator, persist=lambda n, a, r: persist(meter.output, meter.output, n, a, r),
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
        inputs = inherited.inherited.load_startup_inputs(original, freeze["engine_roster"])
        cases = []
        for row in freeze["cases"]:
            with np.load(base.safe_path(row["path"]), allow_pickle=False) as z: cases.append({k: z[k] for k in z.files})
        meter.host_bytes = base.array_bytes((inputs, cases)); meter.enforce_memory()
        def factory(mapping):
            engine = CudaSourceCoefficientEngine(mapping, forward_implementation="precomputed"); meter.cp = engine.cp; return engine
        return execute_continuation(freeze, inputs, cases, factory, meter)
    except Exception as error:
        primary = error
        base.preserve_primary(error, "Original joint failure receipt failed", lambda error=error: base.failure_record(error, None, output, "e283-original", meter.receipt()))
        raise
    finally:
        if primary is None: meter.cleanup(); meter.save()
        else:
            base.preserve_primary(primary, "Final joint cleanup failed", meter.cleanup)
            base.preserve_primary(primary, "Final joint cost failed", meter.save)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "freeze-sha256", "source-review", "source-review-sha256", "output"): parser.add_argument("--"+key, required=True)
    args = parser.parse_args(argv); output = Path(args.output).resolve(); base.require(not output.exists(), "Fresh E283 output required")
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
