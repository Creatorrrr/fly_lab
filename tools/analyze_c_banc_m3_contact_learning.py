"""Independent saved contact candidate/body arithmetic; no neural/model construction."""
# ruff: noqa: BLE001 -- preserve primary input/audit/publication failures.
from __future__ import annotations

import argparse
import json
import os
import sys


def candidate_audit(admission, result, publisher, saved, ledger, np):
    from tools import banc_m3_contact_learning as s
    from tools import verify_c_banc_m3_common_descent_body as body
    from tools import verify_c_banc_m3_coordination_parameter_direction as core
    from tools import verify_c_banc_m3_live_replay as replay
    from tools import verify_c_banc_m3_startup_ray as ray
    from tools.analyze_c_banc_m3_startup_tracking import CONTROL_FIELDS
    from tools.observe_c_banc_m3_local_foot_geometry import pose_fields
    artifacts = admission.request["artifacts"]
    report = saved.json(artifacts["report"])
    s.require(report["run_valid"] and report["first_error"] is None and not report["secondary_errors"], "complete finite direction before saved arithmetic")
    s.require(artifacts["report"]["bytes"] <= s.REPORT_CAP and artifacts["receipts"]["bytes"] <= s.RECEIPTS_CAP
              and report["receipt_artifact"] == artifacts["receipts"], "small direction report binds separate bounded complete receipts")
    receipts = saved.json(artifacts["receipts"])
    s.require(len(receipts) == report["receipt_record_count"], "complete receipt record count")
    metadata = saved.json(admission.roster["neutral_capture"]["geometry_metadata"])
    result["input_plan"] = s.preplan(admission, metadata, audit=True)
    raw_plan = saved.arrays(artifacts["raw"], plan_only=True)
    s.require(result["input_plan"]["selected_upper_bytes"] + raw_plan["expanded_bytes"] <= admission.limits["selected_expanded_cap_bytes"], "all saved audit headers plus legacy selected plan before arrays")
    result["input_plan"]["raw_header_plan"] = raw_plan
    endpoint_specs = s.endpoint_fields(admission.inputs["e318_startup"])
    for spec, fields in endpoint_specs:
        saved.arrays(spec, fields, plan_only=True)
    control_binding = admission.inputs["e318_startup"]["controls"]
    control_fields = {k: s.metadata_fields(control_binding)[k] for k in ("control_tick", "network_tick", "command", "tendon")}
    saved.arrays(control_binding, control_fields, plan_only=True)
    raw = saved.arrays(artifacts["raw"])
    s.require(all(value.dtype == np.bool_ and value.all() for key, value in raw.items() if key.endswith(".valid")), "complete actual acquired field bitmaps before candidate arithmetic")
    packet = {}
    actual_packet = 0
    for role in replay.PACKET_ROLES:
        packet[role] = core.load_selected(admission.root, admission.inputs["legacy_inventory"]["bindings"][role], ledger,
                                          timely=admission.check, generated=False)
        planned = next(x for x in result["input_plan"]["entries"] if x["path"] == admission.inputs["legacy_inventory"]["bindings"][role]["path"])
        s.require(set(packet[role]) == set(planned["members"])
                  and all(v.nbytes == planned["members"][k] for k, v in packet[role].items()), "audit actual legacy selected nbytes")
        actual_packet += planned["bytes"]
    geometry = saved.arrays(admission.roster["neutral_capture"]["geometry"], pose_fields(metadata["dimensions"]))
    control = saved.arrays(admission.roster["neutral_capture"]["control"], CONTROL_FIELDS)
    frozen = s.geometry_context(geometry, metadata, control["command"], admission.contract, np)
    coefficients = admission.contract["loss"]
    qn = raw["q_neutral"]
    checks = {}
    for j, c in enumerate(s.STARTS):
        checks[str(c)] = {"q": replay.compare_output(qn[j], control["command"][c+1:c+5], np),
                          "tendon": replay.compare_output(raw["tendon_neutral"][j], control["tendon"][c+1:c+5], np)}
        s.require(all(v["passed"] for v in checks[str(c)].values()), "nominal saved output arithmetic")
    d0 = np.stack([s.projection(qn[j, 0], frozen, j, np)[0] for j in range(4)])
    measured = np.stack([s.projection(control["command"][c+1], frozen, j, np)[0] for j, c in enumerate(s.STARTS)])
    result["qualified_saved_geometry_projection"] = s.qualified_projection(measured, frozen, saved.json(admission.roster["geometry_analysis"]), np)
    s.require(np.array_equal(d0, raw["d0"]) and np.array_equal(measured, raw["measured_saved_projection"]), "independent frozen baseline and measured projection")
    baseline = s.evaluate(qn, frozen, d0, coefficients, np)
    s.require(baseline == report["neutral_loss"] and baseline["keep"] == 0, "global loss and exact neutral keep0")
    for j in range(4):
        _, gq = s.window_loss(qn[j], frozen, d0, j, coefficients, np)
        s.require(np.array_equal(gq, raw["gq"][j]) and np.array_equal(gq[1:], np.zeros_like(gq[1:])), "analytic first-only gq independently reproduced")
    s.require(np.array_equal(raw["gtheta"].sum(axis=0), raw["sum_gtheta"]), "saved actual independent gradient sum")
    magnitude = float(np.max(np.abs(raw["sum_gtheta"])))
    if magnitude:
        s.require(np.array_equal(raw["direction"], -raw["sum_gtheta"] / magnitude), "normalized negative summed surrogate direction")
    else:
        s.require(report["no_candidate_reason"] == "finite_zero_gradient" and not report["alphas"] and report["probe_evaluations"] == 0, "scientific finite zero-gradient outcome")
    work = s.work_summary(receipts, admission.contract, evaluated=report["evaluated_alphas"], probes=report["probe_evaluations"], complete=True)
    s.require(work == report["work"], "independent separate complete receipts and small report work summary")
    if report["probe_evaluations"]:
        losses = {}
        represented = True
        for sign, name in ((1., "plus"), (-1., "minus")):
            theta = (1 + sign * admission.contract["eta"] * raw["direction"]).astype(np.float32).astype(np.float64)
            weights, palette = core.patch_weights(packet["actual_csr"], packet["recurrent_mask"], theta)
            s.require(np.array_equal(theta, raw["probe-" + name + ".theta_actual"])
                      and np.array_equal(weights, raw["probe-" + name + ".selected_weights_candidate"])
                      and palette == report["probe_representation"][name]["palette"], "actual probe rounded CSR/palette")
            represented = represented and not np.array_equal(weights, packet["actual_csr"]["data"][packet["recurrent_mask"]["positions"]])
            losses[name] = s.evaluate(raw["probe-" + name + ".q"], frozen, d0, coefficients, np)
        s.require(losses == report["finite_probe"]["losses"] and s.finite_probe_pass(baseline, losses["plus"], losses["minus"], represented) == report["finite_probe"]["passed"], "fixed rounded finite-descent screen arithmetic; secant diagnostic only")
        slope = float(raw["sum_gtheta"] @ raw["direction"])
        secant = (losses["plus"]["total"] - losses["minus"]["total"]) / (2 * admission.contract["eta"])
        s.require(slope == report["finite_probe"]["gradient_dot_direction"] and secant == report["finite_probe"]["central_secant"]
                  and report["finite_probe"]["secant_agreement_is_gate"] is False, "independent diagnostic slope; not derivative correctness gate")
        s.require(all(report["finite_probe"][k] == v for k, v in s.secant_diagnostic(slope, secant).items()), "fixed nongating1e-6/.01 secant diagnostic")
    target = core.load_selected(admission.root, admission.inputs["body_inventory"]["inventory"]["selected_archives"]["startup"], ledger,
                                timely=admission.check, generated=False)
    target_plan = next(x for x in result["input_plan"]["entries"] if x["path"] == admission.inputs["body_inventory"]["inventory"]["selected_archives"]["startup"]["path"])
    s.require(set(target) == set(target_plan["members"]) and all(v.nbytes == target_plan["members"][k] for k, v in target.items()), "audit actual startup target nbytes")
    s.require(body.startup99(raw["neutral_startup_q"], raw["neutral_startup_tendon"], **target) == report["neutral_startup"], "saved baseline startup four limits")
    e318_chunk, e318_recruitment = (saved.arrays(spec, fields) for spec, fields in endpoint_specs)
    result["independent_saved_E318_endpoint_checks"] = s.compare_E318_endpoint(raw, e318_chunk, e318_recruitment, np)
    e318_control = saved.arrays(control_binding, control_fields)
    s.require(np.array_equal(e318_control["control_tick"], np.arange(301, dtype=np.int64))
              and np.array_equal(e318_control["network_tick"], e318_control["control_tick"] * 20), "minimum selected E318 output clocks")
    result["input_plan"]["actual_selected_nbytes"] = actual_packet + target_plan["bytes"] + saved.actual_bytes
    s.require(result["input_plan"]["actual_selected_nbytes"] <= result["input_plan"]["selected_upper_bytes"] + raw_plan["expanded_bytes"], "audit actual combined selected512MiB reconciliation")
    result["independent_saved_E318_output_checks"] = {"q": replay.compare_output(raw["neutral_startup_q"][1:], e318_control["command"][1:100], np),
                                                       "tendon": replay.compare_output(raw["neutral_startup_tendon"][1:], e318_control["tendon"][1:100], np)}
    s.require(all(v["passed"] for v in result["independent_saved_E318_output_checks"].values()), "independent E318q/t1e-11/r0")
    result["current_full_startup_sensory_neural_equivalence"] = "NOT_OBSERVED"
    result["immutable_binder_NOT_RUN_unchanged"] = True
    s.require(replay.compare_output(raw["neutral_startup_q"], control["command"][:100], np)["passed"]
              and replay.compare_output(raw["neutral_startup_tendon"], control["tendon"][:100], np)["passed"], "current neutral startup output correspondence independently checked")
    records, eligible_indices = [], []
    for i, row in enumerate(report["alphas"]):
        evidence, arrays = ray.proposal(raw["sum_gtheta"], float(admission.contract["alphas"][i]), packet["actual_csr"], packet["recurrent_mask"], records)
        s.require(all(evidence[k] == row[k] for k in evidence), "saved proposal rounding/provenance/palette")
        if arrays:
            s.require(all(np.array_equal(v, raw[f"alpha-{i}." + k]) for k, v in arrays.items()), "selected actual candidate arithmetic")
        if row["forward_evaluated"]:
            corresponding = all(np.allclose(raw[f"alpha-{i}." + a], raw[f"alpha-{i}." + b], atol=1e-12, rtol=1e-10)
                                for a, b in (("q_relative", "q"), ("tendon_relative", "tendon"), ("startup_q_relative", "startup_q"), ("startup_tendon_relative", "startup_tendon")))
            corresponding = corresponding and all(np.allclose(raw[f"alpha-{i}.relative_" + prefix + key], raw[f"alpha-{i}.deployed_" + prefix + key], atol=1e-12, rtol=1e-10)
                                                   for prefix in ("endpoint-", "startup_endpoint-") for key in s.STATE_KEYS)
            s.require(corresponding and row["output_correspondence"], "saved actual relative/deployed output reconciliation")
            current = s.evaluate(raw[f"alpha-{i}.q"], frozen, d0, coefficients, np)
            startup = body.startup99(raw[f"alpha-{i}.startup_q"], raw[f"alpha-{i}.startup_tendon"], **target)
            s.require(current == row["loss"] and startup == row["startup"], "new contact loss/startup saved arithmetic")
            eligible = s.eligible(baseline, current, startup, evidence["eligible_representation"], corresponding, coefficients)
            s.require(eligible == row["eligible"], "same new first-eligible gate")
            if eligible:
                eligible_indices.append(i)
        else:
            s.require(row["eligible"] is False, "not-evaluated representation never candidate")
        records.append(row)
    candidate = report["candidate"]
    s.require(bool(eligible_indices) == report["candidate_present"] and len(eligible_indices) <= 1,
              "first eligible selected once with all negatives retained")
    if candidate is not None:
        i = eligible_indices[0]
        s.require(candidate["index"] == i == len(records) - 1 and candidate["schema"] == admission.contract["candidate_schema"]
                  and candidate["source_freeze"] == admission.request["source_freeze"] and candidate["contract"] == admission.request["contract"]
                  and candidate["theta_sha256"] == s.sha(raw[f"alpha-{i}.theta_actual"])
                  and candidate["selected_sha256"] == s.sha(raw[f"alpha-{i}.selected_weights_candidate"])
                  and candidate["palette"] == records[i]["palette"], "honest new contact identity")
        result["candidate_arrays"] = publisher.arrays("candidate.npz", {k: raw[f"alpha-{i}." + k] for k in ("theta_actual", "selected_weights_candidate")})
    elif report.get("finite_probe", {}).get("passed") is True:
        s.require(len(records) == len(admission.contract["alphas"]), "all fixed alpha negatives retained before no candidate")
    result.update(arithmetic_valid=True, candidate_present=report["candidate_present"], candidate=candidate, nominal_saved_checks=checks,
                  neutral_loss=baseline, evaluated_alphas=report["evaluated_alphas"], source_direction=artifacts["report"],
                  exact_native_VJP_correctness="UNVERIFIED", no_candidate_reason=report.get("no_candidate_reason"))


def body_audit(admission, result, publisher, saved, np):
    from tools import banc_m3_contact_learning as s
    from tools import verify_c_banc_m3_startup_ray as ray
    artifacts = admission.request["artifacts"]
    report = saved.json(artifacts["report"])
    from tools.banc_m3_contact_learning_admission import retry_candidate_identity
    retry_candidate_identity(admission.request, report, admission.read)
    plans = {key: saved.arrays(artifacts[key], plan_only=True) for key in ("control", "native", "evaluation")}
    s.require(sum(v["expanded_bytes"] for v in plans.values()) <= admission.limits["selected_expanded_cap_bytes"], "all body-audit selected headers before arrays")
    result["input_plan"] = plans
    control, native, evaluation = (saved.arrays(artifacts[k]) for k in ("control", "native", "evaluation"))
    trace = saved.json(artifacts["trace"])

    audit = ray.audit_body(report, control, native, trace, evaluation, admission.inputs["body_inventory"]["inventory"], output=publisher.folder)
    result.update(arithmetic_valid=True, candidate_present=True, candidate=report["candidate"], body_audit=audit,
                  screen_accepted=audit["screen_accepted"], classification="screen_accepted_pending_future_confirmation" if audit["screen_accepted"] else "scientific_rejected")


def audit_backend_guard(role):
    from tools import banc_m3_contact_learning as s

    s.require("mujoco" not in sys.modules and "flygym" not in sys.modules, "saved audit imports no physical simulator")
    tensor_math = sys.modules.get("torch")
    s.require(tensor_math is None or (role == "contact_body_audit" and not tensor_math.cuda.is_initialized()),
              "body audit permits existing CPU tensor losses only; no CUDA or neural rollout")


def worker(admission):
    from tools import banc_m3_contact_learning as s
    from tools.verify_c_banc_m3_contact_learning import failed
    from tools.verify_c_banc_m3_live_replay import publisher_for
    publisher, saved = publisher_for(admission), None
    result = {"schema": s.SCHEMA + ".audit.v1", "role": admission.role, "run_valid": False, "arithmetic_valid": False,
              "candidate_present": False, "candidate": None, "first_error": None, "secondary_errors": [], "classification": "technical_invalid",
              "body_model_neural_CUDA_constructors": 0, "forward_controls": 0, "physics_steps": 0,
              "candidate_adopted": False, "F1_F8": "INCOMPLETE", "biology": "UNPROVEN"}
    ledger, stage = [], "CPU-pre-import"
    try:
        s.require(all(os.environ.get(k) == "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
                  and "torch" not in sys.modules and "mujoco" not in sys.modules, "CPU audit thread1 no neural/model imports")
        import numpy as np
        saved = s.Saved(admission, np)
        stage = "saved-arithmetic"
        if admission.role == "contact_candidate_audit":
            candidate_audit(admission, result, publisher, saved, ledger, np)
        else:
            body_audit(admission, result, publisher, saved, np)
        audit_backend_guard(admission.role)
        result["existing_CPU_tensor_losses_only"] = admission.role == "contact_body_audit" and "torch" in sys.modules
        result["run_valid"] = True
        if admission.role == "contact_candidate_audit":
            result["classification"] = "candidate_audited" if result["candidate_present"] else "scientific_no_candidate"
    except Exception as error:
        failed(result, stage, error)
    finally:
        try:
            if saved is not None:
                result["input_pre_post_hashes"] = saved.posthash()
            from tools import verify_c_banc_m3_coordination_parameter_direction as core
            for entry in ledger:
                s.require(core.digest(core.safe(admission.root, entry["path"]), admission.check) == entry["expected_sha256"], "audit selected Original posthash")
        except Exception as error:
            failed(result, "audit-posthash", error)
        artifacts = {}
        if result.get("candidate_arrays"):
            artifacts["candidate_arrays"] = result["candidate_arrays"]
        try:
            artifacts["report"] = publisher.write("contact-report.json", s.wire(result, s.REPORT_CAP))
        except Exception as error:
            failed(result, "audit-publication", error)
            try:
                artifacts["report"] = publisher.write("contact-publication-failure.json", s.wire(result, s.REPORT_CAP))
            except Exception as secondary:
                failed(result, "audit-fallback", secondary)
        try:
            publisher.json("contact-manifest.json", {"schema": s.SCHEMA + ".manifest.v1", "role": admission.role,
                "source_freeze": admission.request["source_freeze"], "request": admission.request_spec, "contract": admission.request["contract"],
                "run_valid": result["run_valid"], "artifacts": artifacts})
        except Exception as error:
            failed(result, "audit-manifest", error)
            from tools.verify_c_banc_m3_contact_learning import terminal_failure
            terminal_failure(publisher, result, "audit-manifest")
    s.require(result["run_valid"], "saved arithmetic technical failure: " + json.dumps(result["first_error"]))
    return result


def main(argv=None):
    from tools.banc_m3_contact_learning_admission import Admission
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    origin = float(json.loads(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"])["entry_counter_s"])
    admission = Admission(args.request, entry=origin)
    if admission.role not in ("contact_candidate_audit", "contact_body_audit"):
        raise ValueError("saved arithmetic entrypoint role")
    result = worker(admission)
    print(json.dumps({"run_valid": result["run_valid"], "candidate_present": result["candidate_present"], "candidate_adopted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
