"""E286 two frozen RF state reproductions; no fit, Jacobian or adoption."""
from __future__ import annotations

# Preserve first failures through copies, clocks, storage and cleanup.
# ruff: noqa: BLE001
import argparse
import copy
import gc
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import banc_m3_source_coefficient_engine as response
from tools import verify_c_banc_m3_rf_elastic as prior

base, startup, metering = prior.base, prior.startup, prior.elastic.metering
SCHEMA, REVIEW_SCHEMA = "flylab.m3.e286.rf-state-attribution.v1", "flylab.m3.e286.source-review.v1"
SOURCES = ("tools/verify_c_banc_m3_rf_state_attribution.py", "tests/test_c_banc_rf_state_attribution.py")
REQUIRED_INHERITED_SOURCES = (*prior.REQUIRED_INHERITED_SOURCES, *prior.SOURCES)
COPY_KEYS = ("current", "rate", "adaptation", "phi", "branch", "bank", "activation", "original_activation", "q", "ltm")
DECODER_KEYS = ("motor_ids", "source_ids", "edge_ids", "edge_post", "edge_source", "membership", "pool", "low", "high", "kappa",
                "fixed", "moment", "rest", "ltm", "bank0", "activation0", "original_activation0", "initial_rate", "initial_adaptation")
Q0_IDS, POSITIVE_IDS, CONTROLS = (593, 74561, 97445, 121831, 141493, 156479), (593, 141493), (61, 83, 99)
CONTRACT = {"rf_index": 3, "leg": "RF", "parameters": 783, "edges": 261, "startup_samples": 101,
    "states": ["neutral", "plateau"], "max_engines": 2, "max_contexts_live": 1, "max_evaluations": 2,
    "max_forward_ticks": 4000, "max_derivative_ticks": 0, "max_derivative_evaluations": 0, "max_solve_calls": 0,
    "science_wall_s": 10., "cleanup_reserve_s": 5., "supervisor_wall_s": 15., "ram_bytes": 4*1024**3,
    "gpu_additional_bytes": 2*1024**3, "output_bytes": 256*1024**2, "free_disk_bytes": 1024**3,
    "generated_cpu_wall_s": 60., "generated_tiny_solves": 0, "copy_keys": list(COPY_KEYS),
    "q0_motor_ids": list(Q0_IDS), "positive_motor_ids": list(POSITIVE_IDS), "controls": list(CONTROLS),
    "stage_order": list(response.STAGE_ORDER), "ticks_per_control": 20, "stages_per_tick": 4,
    "reproduction_tolerance": 1e-10, "coefficient_target_error_exact": True, "host": False, "derivative": False,
    "engine_lifetime": "fresh engine per state;release response and engine before cleanup and next constructor",
    "fit_calls": 0, "teacher_calls": 0, "whole_BANC_calls": 0, "physics_steps": 0, "candidate_created": False,
    "initialization_supported": False, "global_capacity_supported": False, "F6_exposed": False}
verify_spec, load_rf_inputs, validate_rf_roster = prior.verify_spec, prior.load_rf_inputs, prior.validate_rf_roster


def validate_prior_evidence(audit, plateau_audit, plateau_spec):
    base.require(audit.get("schema") == "flylab.m3.e285.root-original-audit.v1"
        and audit.get("verdict") == "valid_diagnostic_rf_capacity_not_supported"
        and audit.get("terminal_status") == "six_outer_exhausted"
        and all(audit.get(k) is True for k in ("neutral_exact", "sparse_equations_exact", "actual_decisions_independently_verified"))
        and all(audit.get(k) is False for k in ("rf_capacity_supported", "f32_evaluated", "initialization_supported", "candidate_created", "F6_exposed"))
        and audit.get("final_q_peak") == .03693922689093462, "Fixed E285 valid plateau rejection required")
    rows = plateau_audit.get("records", [])
    base.require(plateau_audit.get("schema") == "flylab.m3.e285.plateau-audit.v1"
        and all(plateau_audit.get(k) == 0 for k in ("original_calls", "model_calls", "solver_calls"))
        and len(rows) == 6 and [r.get("outer") for r in rows] == list(range(6))
        and plateau_audit.get("final") == plateau_spec and plateau_audit.get("final_peak_control") == 83
        and plateau_audit.get("final_peak_local_joint") == 0
        and plateau_audit.get("final_peak_signed_error") == audit["final_q_peak"]
        and rows[-1].get("full_hard_domain_fixed_J_infeasible") is True
        and rows[-1].get("peak_lower_bound") == .036936296347271205, "Exact saved plateau/local-J evidence required")
    for row in rows:
        verify_spec(row["entry"]); verify_spec(row["solution"])
    return {"e285_rejected": True, "final_peak_control": 83, "final_q_peak": audit["final_q_peak"],
            "local_J_peak_lower_bound": rows[-1]["peak_lower_bound"], "nonlinear_impossibility_claim": False}


def validate_freeze(path, digest, review_path, review_digest):
    freeze, review = base.verified_json(path, digest), base.verified_json(review_path, review_digest)
    base.require(freeze.get("schema") == SCHEMA and freeze.get("contract") == CONTRACT, "Exact E286 contract required")
    sources = freeze.get("approved_sources")
    base.require(isinstance(sources, dict) and set(sources) == set(SOURCES), "Two E286 owned source pins required")
    for name, sha in sources.items(): base.require(base.sha256(base.safe_path(name)) == sha, "Owned source changed: "+name)
    base.require(review.get("schema") == REVIEW_SCHEMA and review.get("verdict") == "approved"
        and review.get("freeze_sha256") == digest and review.get("approved_sources") == sources, "Exact E286 source review required")
    pins = freeze.get("inherited_sources")
    base.require(isinstance(pins, dict) and set(REQUIRED_INHERITED_SOURCES) <= set(pins), "All32 inherited pins required")
    for name, sha in pins.items(): base.require(base.sha256(base.safe_path(name)) == sha, "Inherited source changed: "+name)
    fs, rs = freeze["e285_freeze"], freeze["e285_source_review"]
    old, original = prior.validate_freeze(verify_spec(fs), fs["sha256"], verify_spec(rs), rs["sha256"])
    base.require(pins == {**old["inherited_sources"], **old["approved_sources"]}, "Exact E285 source chain required")
    for key in ("inputs", "cases", "engine_roster", "packages", "e280_resource_qualification", "rf_index", "rf_case", "rf_engine_roster"):
        base.require(freeze.get(key) == old[key], "Original RF provenance changed: "+key)
    audit, proof = (json.loads(verify_spec(freeze[k]).read_text(encoding="utf-8")) for k in ("e285_root_result", "e285_plateau_audit"))
    validate_prior_evidence(audit, proof, freeze["plateau_actual"])
    base.require(Path(freeze["plateau_actual"]["path"]).name == "outer-05-alpha-0-actual.npz", "Fixed final actual artifact required")
    with np.load(verify_spec(freeze["plateau_actual"]), allow_pickle=False) as z: plateau = {k: z[k] for k in z.files}
    with np.load(verify_spec({k: freeze["rf_case"][k] for k in ("path", "sha256")}), allow_pickle=False) as z: case = {k: z[k] for k in z.files}
    startup.margin.validate_entry(plateau, derivative=False)
    base.require(plateau["coefficients"].shape == (261, 3) and base.coefficient_valid(plateau["coefficients"], interior=True)
        and np.array_equal(plateau["qerr"], plateau["startup_q"][1:100]-case["target_q"])
        and np.array_equal(plateau["lerr"], plateau["startup_ltm"][1:100]-case["target_ltm"]), "Exact plateau coefficients/targets/errors required")
    base.require(float(np.max(np.abs(plateau["qerr"]))) == audit["final_q_peak"]
        and float(plateau["qerr"][82, 0]) == audit["final_q_peak"], "Saved final control83 exact plateau peak required")
    return freeze, original


class ScienceDeadline(startup.ScienceDeadline):
    def remaining(self): return 10-(self.clock()-self.started)
    def receipt(self): return {"original_wall_s": self.clock()-self.started, "science_wall_limit_s": 10., "supervisor_wall_limit_s": 15.}


class StateBudget(startup.StartupBudget):
    def request(self, derivative):
        self.deadline.check()
        base.require(derivative is False, "No derivative evaluation allowed")
        if self.counts["evaluations_requested"] >= 2: raise base.BudgetExceeded("rf_state_evaluations")
        self.leg_evaluations += 1; self.counts["evaluations_requested"] += 1; self.counts["forward_requested_steps"] += 2000


class StateMeter(prior.ElasticMeter):
    def enforce_memory(self):
        base.require(self.peak_estimate <= CONTRACT["ram_bytes"] and (self.peak_rss is None or self.peak_rss <= CONTRACT["ram_bytes"]), "RF state RAM bound exceeded")
        rows = [r for r in self.memory_events if r.get("live_gpu_estimate") is not None]
        if rows: base.require(rows[-1]["live_gpu_estimate"] <= CONTRACT["gpu_additional_bytes"], "RF state GPU bound exceeded")
    def account(self, cost): raise ValueError("E286 forbids optimizer cost")
    def receipt(self): return {**super().receipt(), "solver_calls": 0, "jacobian_calls": 0, "candidate_created": False}


def persist(output, name, arrays, receipt):
    if arrays is not None: np.savez(output/(name+".npz"), **arrays)
    base.write_json(output/(name+".json"), receipt)
    base.require(sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) <= CONTRACT["output_bytes"], "E286 output bound exceeded")


def validate_raw(p, raw):
    n, m = len(p["motor_ids"]), len(p["fixed"])
    shapes = {"current": (8000, n), "rate": (101, n), "adaptation": (101, n), "phi": (8000, n),
              "branch": (8000, n), "bank": (101, n, 3), "activation": (101, m), "original_activation": (101, m),
              "q": (101, 7), "ltm": (101, 1)}
    for key, shape in shapes.items():
        v = raw[key]
        base.require(isinstance(v, np.ndarray) and v.shape == shape and np.isfinite(v).all(), "Complete finite RF raw required: "+key)
        base.require(v.dtype == (np.uint8 if key == "branch" else np.float64), "Exact RF raw dtype required: "+key)
    base.require(np.isin(raw["branch"], [0, 1]).all() and np.all(raw["rate"] >= 0) and np.all(raw["phi"] >= 0), "Native nonnegative rate/phi and binary branch required")


def reconstruct_decoder(p, raw):
    """Independent endpoint filter and decoder calculation, without neural work."""
    validate_raw(p, raw)
    bank, original = np.empty_like(raw["bank"]), np.empty_like(raw["original_activation"])
    bank[0], original[0] = p["bank0"], p["original_activation0"]
    for control in range(1, 101):
        bank[control] = bank[control-1]+response.FILTER_ALPHA*(raw["rate"][control, :, None]-bank[control-1])
        original[control] = original[control-1]+response.FIXED_ALPHA*(p["pool"]@raw["rate"][control]-original[control-1])
    channels = p["low"]*raw["bank"]+(p["high"]-p["low"])*np.maximum(raw["bank"]-p["kappa"][None, :, None], 0)
    motor_activation = channels.sum(2)
    activation = np.zeros_like(raw["activation"])
    for index, muscle in enumerate(p["membership"]): activation[:, muscle] += motor_activation[:, index]
    activation[:, p["fixed"]] = raw["original_activation"][:, p["fixed"]]
    q, ltm = p["rest"]+.02*(activation@p["moment"].T), .2*np.tanh(activation[:, p["ltm"]]/10)
    checks = {key: base.numerical_gate(value, raw[key], 1e-10) for key, value in
              (("bank", bank), ("original_activation", original), ("activation", activation), ("q", q), ("ltm", ltm))}
    checks["initial_rate"] = base.numerical_gate(raw["rate"][0], p["initial_rate"].astype(np.float64), 0)
    checks["initial_adaptation"] = base.numerical_gate(raw["adaptation"][0], p["initial_adaptation"].astype(np.float64), 0)
    return {"channels": channels, "motor_activation": motor_activation, "reconstructed_q": q,
            "reconstructed_ltm": ltm, "reconstructed_activation": activation}, {"pass": all(v["pass"] for v in checks.values()), "checks": checks}


def attribution(p, raw, target_q):
    reconstructed, decoder = reconstruct_decoder(p, raw)
    base.require(decoder["pass"], "Independent RF decoder reconstruction failed")
    ids = np.asarray(p["motor_ids"])
    base.require(ids.dtype == np.int64 and len(set(ids.tolist())) == len(ids), "Unique explicit motor identity axis required")
    indices = np.array([int(np.flatnonzero(ids == value)[0]) for value in Q0_IDS], np.int64)
    affected = ids[p["moment"][0, p["membership"]] != 0]
    base.require(sorted(affected.tolist()) == list(Q0_IDS), "Exact six RF q0 motor identities required")
    signs = p["moment"][0, p["membership"][indices]]
    positive = np.array([int(np.flatnonzero(ids == value)[0]) for value in POSITIVE_IDS], np.int64)
    base.require(np.all(p["moment"][0, p["membership"][positive]] > 0), "MN593/141493 must belong to positive q0 muscles")
    motor_q = .02*reconstructed["motor_activation"][:, indices]*signs
    motor_q[:, p["fixed"][p["membership"][indices]]] = 0
    muscle_q = .02*raw["activation"]*p["moment"][0]
    hinge = raw["bank"] > p["kappa"][None, :, None]
    windows = np.array([[(control-1)*80, control*80] for control in CONTROLS], np.int64)
    current, phi, branch = [np.stack([raw[key][begin:end, indices].reshape(20, 4, len(indices)) for begin, end in windows]) for key in ("current", "phi", "branch")]
    below = bool(np.all(raw["bank"][99, positive] < p["kappa"][positive, None]))
    summary = {"decoder": decoder, "h286": "supported" if below else "refuted",
        "hypothesis_scope": "control99 MN593/141493 all three banks strictly below kappa;high hinge inactive only",
        "rates_exact_zero": (raw["rate"][list(CONTROLS)][:, indices] == 0).tolist(),
        "rates_strictly_positive": (raw["rate"][list(CONTROLS)][:, indices] > 0).tolist(),
        "stage_current": {"minimum": current.min((1,2)).tolist(), "maximum": current.max((1,2)).tolist(), "mean": current.mean((1,2)).tolist()},
        "stage_phi": {"minimum": phi.min((1,2)).tolist(), "maximum": phi.max((1,2)).tolist(), "mean": phi.mean((1,2)).tolist()},
        "stage_branch_active_count": branch.sum((1,2)).tolist(), "endpoint_adaptation_is_not_RK4_internal_state": True,
        "global_capacity_supported": False, "natural_gait_supported": False, "candidate_created": False}
    arrays = {**reconstructed, "q0_motor_ids": ids[indices], "q0_motor_indices": indices, "q0_membership": p["membership"][indices],
        "q0_moment": signs, "q0_motor_contribution": motor_q, "q0_muscle_contribution": muscle_q,
        "q0_positive_contribution": np.maximum(muscle_q, 0).sum(1), "q0_negative_contribution": np.minimum(muscle_q, 0).sum(1),
        "controls": np.array(CONTROLS, np.int64), "window_stage_ranges": windows,
        "selected_rate": raw["rate"][list(CONTROLS)][:, indices], "selected_endpoint_adaptation": raw["adaptation"][list(CONTROLS)][:, indices],
        "selected_bank": raw["bank"][list(CONTROLS)][:, indices], "selected_kappa": p["kappa"][indices],
        "selected_low": p["low"][indices], "selected_high": p["high"][indices], "selected_hinge_active": hinge[list(CONTROLS)][:, indices],
        "selected_current": current, "selected_phi": phi, "selected_branch": branch, "target_q": target_q.copy(),
        "q0_error": raw["q"][1:100, 0]-target_q[:, 0]}
    return arrays, summary


def _capture(engine, coefficients, meter, event, label):
    """Keep response ownership until every requested copy has been attempted."""
    device, primary, raw = None, None, {}
    try:
        device = metering.metered(event, "forward", lambda: engine.solve(coefficients, derivative=False, host=False), meter.deadline.clock)
    except Exception as error:
        primary = error
        device = getattr(error, "phase_partial_result", None)
        if not isinstance(device, dict): device = getattr(error, "partial_response", {})
        if not isinstance(device, dict): device = {}
    try:
        for key in COPY_KEYS:
            try:
                base.require(key in device, "Missing current-response component: "+key)
                raw[key] = metering.metered(event, "copy", lambda key=key: engine.host_copy(device[key], label=label+":"+key), meter.deadline.clock)
            except Exception as error:
                partial = getattr(error, "phase_partial_result", None)
                if isinstance(partial, np.ndarray): raw[key] = partial
                event["copy_errors"].append({"key": key, "exception": type(error).__name__, "error": str(error)})
                if primary is None: primary = error
                else: primary.add_note("Additional response copy failure "+key+": "+repr(error))
        event["response_receipt"] = copy.deepcopy(getattr(engine, "last_receipt", device.get("receipt", {})))
        receipt = event["response_receipt"]
        meter.budget.counts["forward_completed_steps"] += int(receipt.get("completed_steps", 0))
        meter.budget.counts["derivative_completed_steps"] += int(receipt.get("derivative_completed_steps", 0))
        if event["forward"]["completed"]: meter.budget.counts["evaluations_completed"] += 1
        else: meter.budget.counts["evaluations_failed"] += 1
        event["copied_keys"] = list(raw)
        if primary is not None:
            primary.partial_host_response = raw
            # Engine-scope rescue must see host copies, never retained device references.
            primary.partial_response = raw
            if hasattr(primary, "phase_partial_result"): del primary.phase_partial_result
            raise primary
        return raw
    finally:
        device, engine = None, None


def evaluate_state(mapping, coefficients, reference, factory, meter, label):
    meter.deadline.check()
    base.require(label in CONTRACT["states"] and not any(e["label"] == label for e in meter.evaluation_events), "Each frozen neutral/plateau state exactly once")
    if meter.budget.counts["evaluations_requested"] >= 2: raise base.BudgetExceeded("rf_state_evaluations")
    event = {"label": label, "status": "requested", "copy_errors": [],
             **{k: metering.phase() for k in ("forward", "copy", "save", "reproduction", "decoder")}}
    meter.evaluation_events.append(event)
    engine, raw, problem, primary, result = None, {}, None, None, None
    def counted_factory(mapping):
        meter.engine_requests += 1
        base.require(meter.engine_requests <= 2 and meter.active_contexts == 0, "At most2 fresh RF engines;one live")
        try:
            return metering.metered({"constructor": meter.engine_creation}, "constructor", lambda: factory(mapping), meter.deadline.clock)
        except Exception as error:
            interrupted = getattr(error, "phase_partial_result", None)
            if interrupted is not None:
                estimate = None
                try:
                    estimate = meter.memory(interrupted, "interrupted-state-constructor")
                except Exception as secondary:
                    error.add_note("Interrupted constructor memory accounting failed: "+repr(secondary))
                finally:
                    if estimate is not None:
                        base.preserve_primary(error, "Interrupted constructor release failed", lambda: meter.release(interrupted, "interrupted-state-constructor", estimate))
                    interrupted = None
                    del error.phase_partial_result
            raise
    try:
        with base.engine_scope(counted_factory, mapping, meter, "rf-state-"+label) as engine:
            p = engine.problem
            base.require(p["samples"] == 101 and p["stages"] == 8000 and p["params"] == coefficients.size, "Exact startup shape required")
            for key, value in mapping.items():
                if isinstance(value, np.ndarray): base.require(key in p and np.array_equal(p[key], value), "Engine original input identity differs: "+key)
            for key in DECODER_KEYS: base.require(np.array_equal(p[key], mapping[key]), "Engine original decoder identity differs: "+key)
            problem = {k: p[k].copy() for k in DECODER_KEYS}
            metering.metered(event, "save", lambda: persist(meter.output, label+"-decoder-mapping", problem, {"stage_order": list(response.STAGE_ORDER)}), meter.deadline.clock)
            meter.budget.request(False)
            try: raw = _capture(engine, coefficients, meter, event, label)
            except Exception as error:
                raw = getattr(error, "partial_host_response", raw); raise
            raw.update(coefficients=coefficients.copy(), startup_q=raw["q"], startup_ltm=raw["ltm"],
                       qerr=raw["q"][1:100]-reference["target_q"], lerr=raw["ltm"][1:100]-reference["target_ltm"],
                       target_q=reference["target_q"].copy(), target_ltm=reference["target_ltm"].copy())
            metering.metered(event, "save", lambda: persist(meter.output, label, raw, event), meter.deadline.clock)
            def reproduce():
                gates = {key: base.numerical_gate(raw[key], reference[key], 1e-10) for key in ("startup_q", "startup_ltm")}
                base.require(np.array_equal(coefficients, reference["coefficients"]), "Frozen state coefficients changed")
                if "qerr" in reference:
                    base.require(np.array_equal(reference["qerr"], reference["startup_q"][1:100]-reference["target_q"])
                        and np.array_equal(reference["lerr"], reference["startup_ltm"][1:100]-reference["target_ltm"]), "Saved target/error relation changed")
                base.require(np.array_equal(raw["qerr"], reference["startup_q"][1:100]-reference["target_q"])
                    and np.array_equal(raw["lerr"], reference["startup_ltm"][1:100]-reference["target_ltm"]), "Exact reproduced target/error relation required")
                base.require(all(g["pass"] for g in gates.values()), "RF state reproduction failed")
                return gates
            gates = metering.metered(event, "reproduction", reproduce, meter.deadline.clock)
            arrays, report = metering.metered(event, "decoder", lambda: attribution(problem, raw, reference["target_q"]), meter.deadline.clock)
            result = {"label": label, "reproduction": gates, **report}
            metering.metered(event, "save", lambda: persist(meter.output, label+"-attribution", arrays, result), meter.deadline.clock)
            meter.deadline.check(); event["status"] = "complete"
        return result
    except Exception as error:
        primary = error; event.update(status="incomplete" if isinstance(error, base.BudgetExceeded) else "invalid", error=str(error), exception=type(error).__name__)
        error.partial_host_response = raw
        error.partial_response = raw
        base.preserve_primary(error, "Partial state raw publication failed", lambda: persist(meter.output, label+"-partial", raw, event))
        error.state_result = result
        raise
    finally:
        engine = None
        # Completed helper frames may retain the engine or device arrays after an
        # exception. Host copies and receipts are preserved before clearing them.
        if primary is not None: traceback.clear_frames(primary.__traceback__)
        gc.collect()
        if primary is None:
            try: meter.cleanup()
            except Exception as error:
                event.update(status="invalid", error=str(error)); error.state_result = result; raise
        else: base.preserve_primary(primary, "State evaluation cleanup failed", meter.cleanup)


def execute_attribution(freeze, inputs, case, plateau, factory, meter):
    meter.budget = StateBudget(meter.deadline)
    validate_rf_roster(freeze["rf_engine_roster"], case)
    base.require(base.coefficient_valid(plateau["coefficients"], interior=True), "Unmodified valid plateau coefficients required")
    refs = ({"coefficients": case["neutral_coefficients"], "startup_q": case["neutral_startup_q"], "startup_ltm": case["neutral_startup_ltm"]}, plateau)
    result = {"schema": SCHEMA, "run_valid": False, "status": "requested", "states": [], "H286": None,
              "candidate_created": False, "initialization_supported": False, "full_fit_started": False,
              "body_promoted": False, "F6_exposed": False, "global_capacity_supported": False}
    try:
        base.write_json(meter.output/"provenance.json", {"schema": SCHEMA, "e285_rejected": True,
            **{k: freeze[k] for k in ("e285_freeze", "e285_source_review", "e285_root_result", "e285_plateau_audit", "plateau_actual", "inherited_sources", "approved_sources")}})
        for label, reference in zip(CONTRACT["states"], refs, strict=True):
            reference = {**reference, "target_q": case["target_q"], "target_ltm": case["target_ltm"]}
            row = evaluate_state(inputs[2], reference["coefficients"], reference, factory, meter, label)
            result["states"].append(row); meter.case_results = result["states"]; meter.save()
        meter.save()
        result.update(run_valid=True, status="valid_state_attribution", H286=result["states"][1]["h286"])
        persist(meter.output, "attribution-result", None, result); return result
    except Exception as error:
        result.update(run_valid=False, H286=None, status="incomplete" if isinstance(error, base.BudgetExceeded) else "invalid", error=str(error), exception=type(error).__name__)
        if getattr(error, "state_result", None) is not None: result["partial_state"] = error.state_result
        error.attribution_result = result
        base.preserve_primary(error, "Failed attribution result publication failed", lambda: persist(meter.output, "attribution-result", None, result))
        base.preserve_primary(error, "Failed attribution final cost failed", meter.save)
        raise


def original_run(freeze, original, output, started):
    from tools.banc_m3_source_coefficient_engine import CudaSourceCoefficientEngine
    meter, primary, result = StateMeter(output, ScienceDeadline(started)), None, None
    try:
        base.require(shutil.disk_usage(output).free >= CONTRACT["free_disk_bytes"], "One GiB free disk required")
        inputs = load_rf_inputs(original, freeze["rf_engine_roster"])
        with np.load(verify_spec({k: freeze["rf_case"][k] for k in ("path", "sha256")}), allow_pickle=False) as z: case = {k: z[k] for k in z.files}
        with np.load(verify_spec(freeze["plateau_actual"]), allow_pickle=False) as z: plateau = {k: z[k] for k in z.files}
        meter.host_bytes = base.array_bytes((inputs, case, plateau)); meter.enforce_memory()
        def factory(mapping):
            engine = CudaSourceCoefficientEngine(mapping, forward_implementation="precomputed"); meter.cp = engine.cp; return engine
        result = execute_attribution(freeze, inputs, case, plateau, factory, meter)
    except Exception as error:
        primary = error
        base.preserve_primary(error, "Original attribution failure receipt failed", lambda error=error: base.failure_record(error, None, output, "e286-original", meter.receipt()))
        raise
    finally:
        if primary is None:
            try: meter.cleanup(); meter.save()
            except Exception as error:
                if result is not None:
                    result.update(run_valid=False, H286=None, status="invalid", error=str(error), exception=type(error).__name__)
                    error.attribution_result = result
                base.preserve_primary(error, "Final cleanup invalidation save failed", lambda: persist(output, "attribution-result", None, result))
                raise
        else:
            base.preserve_primary(primary, "Final attribution cleanup failed", meter.cleanup)
            base.preserve_primary(primary, "Final attribution cost failed", meter.save)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "freeze-sha256", "source-review", "source-review-sha256", "output"): parser.add_argument("--"+name, required=True)
    args = parser.parse_args(argv); output = Path(args.output).resolve(); base.require(not output.exists(), "Fresh E286 output required")
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=False)
    base.write_json(output/"cost.json", {"phase": "freeze_validation", "original_calls": 0, "solver_calls": 0})
    try:
        freeze, original = validate_freeze(Path(args.freeze), args.freeze_sha256, Path(args.source_review), args.source_review_sha256)
        result = original_run(freeze, original, output, started)
        base.write_json(output/"result.json", result); return 0
    except Exception as error:
        result = {"run_valid": False, "status": "incomplete" if isinstance(error, base.BudgetExceeded) else "invalid",
                  "error": str(error), "exception": type(error).__name__, "notes": getattr(error, "__notes__", []),
                  "attribution_result": getattr(error, "attribution_result", None), "H286": None,
                  "candidate_created": False, "initialization_supported": False, "F6_exposed": False}
        base.preserve_primary(error, "Terminal attribution publication failed", lambda: base.write_json(output/"result.json", result)); return 1


if __name__ == "__main__": raise SystemExit(main())
