"""One finite contact surrogate direction and one conditional continuous body."""
# ruff: noqa: BLE001 -- actual partial receipts and primary errors are evidence.
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ENTRY = time.perf_counter()


def failed(result, stage, error):
    record = {"stage": stage, "type": type(error).__name__, "message": str(error)[:4096], "notes": [str(v)[:2048] for v in getattr(error, "__notes__", [])[:8]]}
    if result["first_error"] is None:
        result["first_error"] = record
    else:
        result["secondary_errors"].append(record)
    result["run_valid"], result["classification"] = False, "technical_invalid"


def publish(publisher, raw, result, ledger, admission):
    from tools import banc_m3_contact_learning as s
    artifacts = {}
    receipts = result.pop("receipts", {})
    def raw_action():
        s.require(len(raw) <= 256 and sum(v.nbytes for v in raw.values()) <= 128 << 20, "bounded contact raw members/temporary payload")
        return publisher.arrays("contact-raw.npz", raw)
    for key, name, action in (("raw", "contact-raw.npz", raw_action),
                              ("ledger", "selected-ledger.json", lambda: publisher.json("selected-ledger.json", ledger)),
                              ("receipts", "work-receipts.json", lambda: publisher.write("work-receipts.json", s.wire(receipts, s.RECEIPTS_CAP)))):
        try:
            artifacts[key] = action()
        except Exception as error:
            failed(result, name, error)
    result["receipt_artifact"] = artifacts.get("receipts")
    result["receipt_record_count"] = len(receipts)
    try:
        publisher.write("contact-report.md", ("# 접촉 조건부 유한 학습\n\n" + json.dumps({k: result.get(k) for k in ("run_valid", "candidate_present", "classification", "neutral_loss", "finite_probe", "work", "first_error")}, ensure_ascii=False, allow_nan=False)
                                             + "\n\n네 neutral 전이의 조건부 surrogate 방향입니다. Native VJP·candidate 기하·실제 보행·biology는 검증되지 않았습니다. 후보 채택false, F1–F8 INCOMPLETE.\n").encode("utf8"))
    except Exception as error:
        failed(result, "markdown-publication", error)
    try:
        artifacts["report"] = publisher.write("contact-report.json", s.wire(result, s.REPORT_CAP))
    except Exception as error:
        failed(result, "report-publication", error)
        try:
            artifacts["report"] = publisher.write("contact-publication-failure.json", s.wire(result, s.REPORT_CAP))
        except Exception as secondary:
            failed(result, "failure-publication", secondary)
    try:
        publisher.json("contact-manifest.json", {"schema": "flylab.m3.contact-learning.manifest.v1", "role": admission.role,
            "source_freeze": admission.request["source_freeze"], "request": admission.request_spec,
            "contract": admission.request["contract"], "run_valid": result["run_valid"], "artifacts": artifacts})
    except Exception as error:
        failed(result, "manifest-publication", error)
        terminal_failure(publisher, result, "manifest-publication")
    if not result["run_valid"]:
        print(json.dumps({"first_error": result["first_error"], "secondary_errors": result["secondary_errors"], "work": result.get("work")},
                         allow_nan=False, ensure_ascii=True)[:8192], file=sys.stderr)
    return artifacts


def terminal_failure(publisher, result, stage):
    from tools import banc_m3_contact_learning as s
    journal = getattr(getattr(publisher, "admission", None), "runtime_journal", None)
    if journal is not None:
        journal.acquire_errors(result)
        return  # body_worker makes the one separate terminal stream attempt last
    value = {"schema": s.SCHEMA + ".terminal-failure.v1", "run_valid": False, "stage": stage,
             "first_error": result["first_error"], "secondary_errors": result["secondary_errors"], "work": result.get("work")}
    try:
        publisher.write("contact-terminal-publication-failure.json", s.wire(value, 64 << 10))
    except Exception as error:
        failed(result, "terminal-failure-publication", error)
    print(json.dumps(value, ensure_ascii=True, allow_nan=False)[:8192], file=sys.stderr)


def load(admission, result, ledger, saved_box):
    import numpy as np

    from tools import analyze_c_banc_m3_startup_tracking as tracking
    from tools import banc_m3_contact_learning as science
    from tools import verify_c_banc_m3_coordination_parameter_direction as core
    from tools import verify_c_banc_m3_live_replay as replay
    from tools.banc_m3_startup_ray_inputs import load_e318
    from tools.observe_c_banc_m3_local_foot_geometry import pose_fields
    saved = science.Saved(admission, np)
    saved_box["reader"] = saved
    metadata = saved.json(admission.roster["neutral_capture"]["geometry_metadata"])
    result["input_plan"] = science.preplan(admission, metadata)
    packet = {"metadata": core.metadata_packet(admission.root, admission.inputs["legacy_inventory"])}
    actual_packet = 0
    for role in replay.PACKET_ROLES:
        admission.fence("pre_array")
        packet[role] = core.load_selected(admission.root, admission.inputs["legacy_inventory"]["bindings"][role], ledger,
                                          timely=admission.check, generated=False)
        planned = next(x for x in result["input_plan"]["entries"] if x["path"] == admission.inputs["legacy_inventory"]["bindings"][role]["path"])
        science.require(set(packet[role]) == set(planned["members"])
                        and all(v.nbytes == planned["members"][k] for k, v in packet[role].items()), "actual legacy packet selected nbytes")
        actual_packet += planned["bytes"]
    csr, mask, anatomical = packet["actual_csr"], packet["recurrent_mask"], packet["e289_roster"]
    science.require(np.array_equal(mask["positions"], anatomical["edge_ids"]) and np.array_equal(mask["population"], anatomical["premotor_ids"])
                    and np.array_equal(csr["data"][mask["positions"]], anatomical["weights32"])
                    and np.array_equal(np.unique(mask["edge_parameters"]), np.arange(8547)), "original edge/group topology and selected deployed weights")
    capture = admission.roster["neutral_capture"]
    context = saved.arrays(capture["context"], tracking.CONTEXT_FIELDS)
    control = saved.arrays(capture["control"], {k: (tuple(301 if x is None else x for x in shape), dtype) for k, (shape, dtype) in tracking.CONTROL_FIELDS.items()})
    report = saved.json(capture["body_report"])
    result["recorded_context"] = replay.validate_recorded(context, control, report, packet, np)
    pose = saved.arrays(capture["geometry"], pose_fields(metadata["dimensions"]))
    frozen = science.geometry_context(pose, metadata, control["command"], admission.contract, np)
    measured = np.stack([science.projection(control["command"][c + 1], frozen, j, np)[0] for j, c in enumerate(science.STARTS)])
    result["qualified_saved_geometry_projection"] = science.qualified_projection(measured, frozen, saved.json(admission.roster["geometry_analysis"]), np)
    result["input_plan"]["actual_packet_nbytes"] = actual_packet
    # All whole numeric headers are checked before StateStore allocates any leaf.
    whole_nbytes = 0
    for name, spec in admission.inputs["e318_startup"]["whole_files"].items():
        admission.fence("pre_array")
        p = core.safe(admission.root, name)
        science.require(p.stat().st_size == spec["bytes"] and core.digest(p, admission.check) == spec["sha256"], "whole preplan actual digest")
        if p.suffix == ".npy":
            with p.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                science.require(version in ((1, 0), (2, 0)), "whole numeric header version")
                shape, fortran, dtype = reader(stream, max_header_size=4096)
                science.require(not fortran and dtype.kind in "bifu", "whole selected numeric dtype")
                size = science.member_bytes(shape, dtype.str, admission.limits["member_cap_bytes"])
                science.require(size + stream.tell() == spec["bytes"], "whole exact selected header size")
                whole_nbytes += size
    result["input_plan"]["actual_whole_nbytes"] = whole_nbytes
    e318_headers = []
    binding = admission.inputs["e318_startup"]
    for spec in (*binding["chunks"], binding["controls"], binding["recruitment"], admission.inputs["body_inventory"]["inventory"]["selected_archives"]["startup"]):
        e318_headers.append(saved.arrays(spec, science.metadata_fields(spec), plan_only=True))
    result["input_plan"]["actual_E318_header_nbytes"] = sum(x["expanded_bytes"] for x in e318_headers)
    startup, resolution = load_e318(admission, core, ledger, admission.inputs["body_inventory"]["inventory"])
    result["E318_resolution"] = resolution
    science.require(startup["drive"].nbytes == result["input_plan"]["startup99_drive_bytes"], "actual99 drives62847576B")
    actual = actual_packet + saved.actual_bytes + whole_nbytes + result["input_plan"]["actual_E318_header_nbytes"]
    science.require(actual <= result["input_plan"]["selected_upper_bytes"], "all actual selected schema/header/nbytes reconcile within preplan")
    result["input_plan"]["actual_selected_nbytes"] = actual
    return packet, context, control, frozen, startup, saved


def collect(admission, packet, context, control, frozen, startup, result, raw, watch):
    import numpy as np
    import torch

    from tools import banc_m3_contact_learning as s
    from tools import verify_c_banc_m3_candidate_body_handoff as monitoring
    from tools import verify_c_banc_m3_common_descent_body as body
    from tools import verify_c_banc_m3_coordination_parameter_direction as core
    from tools import verify_c_banc_m3_live_replay as replay
    from tools import verify_c_banc_m3_startup_ray as ray
    coefficients, contract = admission.contract["loss"], admission.contract
    bundle = replay.minimal_assembly(packet)
    device = bundle["rollout"].device
    def tensor(x):
        return torch.as_tensor(x.copy(), device=device).contiguous()
    output_mask = tensor(context["output_mask"])
    ones = torch.ones(8547, dtype=torch.float64, device=device)
    receipts = result["receipts"]
    phases = s.PhaseBuffers(raw, result, np)

    def run(rollout, state, drives, theta, key, *, backward_seed=None, returned=None, gradient_returned=None):
        admission.check("science")
        receipt, primary = None, None
        try:
            if backward_seed is None:
                with torch.no_grad():
                    out = rollout.run(*(tensor(state[k]) for k in s.STATE_KEYS), tensor(drives), theta, output_mask)
            else:
                out = rollout.run(*(tensor(state[k]) for k in s.STATE_KEYS), tensor(drives), theta, output_mask)
            receipt = out["receipt"]
            q = out["q"].detach().cpu().numpy().copy()
            if returned is not None:
                returned("q", q)
            tendon = out["tendon"].detach().cpu().numpy().copy()
            if returned is not None:
                returned("tendon", tendon)
            receipt.confirm()
            s.require(np.isfinite(q).all() and np.isfinite(tendon).all(), "finite returned output " + key)
            if backward_seed is not None:
                seed = backward_seed(q)
                s.require(np.array_equal(seed[1:], np.zeros_like(seed[1:])), "other3 controls exactly zero seed")
                out["q"].backward(tensor(seed))  # exactly one backward per independent efficacy leaf
                s.require(theta.is_leaf and theta.grad is not None, "returned surrogate efficacy gradient")
                gradient = theta.grad.detach().cpu().numpy().copy()
                if gradient_returned is not None:
                    gradient_returned(gradient)
                receipt.confirm()
                s.require(np.isfinite(gradient).all(), "finite actual returned surrogate efficacy gradient")
            else:
                gradient = None
            endpoint = {k: out[k].detach().cpu().numpy().copy() for k in s.STATE_KEYS}
            return q, tendon, endpoint, gradient
        except Exception as error:
            primary = error
            receipt = receipt or getattr(error, "receipt", None)
            if receipt is not None and receipt.confirm_requested == receipt.confirm_returned:
                try:
                    receipt.confirm()
                except Exception as secondary:
                    error.add_note("partial device confirmation: " + str(secondary))
            raise
        finally:
            if receipt is not None:
                receipts[key] = core.receipt_dict(receipt)
            try:
                result["work"] = s.work_summary(receipts, contract, evaluated=result["evaluated_alphas"], probes=result["probe_evaluations"])
                monitoring.monitor_check(watch.snapshot(), admission.limits)
            except Exception as error:
                if primary is not None:
                    primary.add_note("partial work/memory observation: " + str(error))
                else:
                    raise

    def window_state(j):
        return {k: context[k][j] for k in s.STATE_KEYS}

    def windows(rollout, theta, prefix, qkey, tkey):
        phases.allocate(qkey, (4, 4, 42))
        phases.allocate(tkey, (4, 4, 6))
        endpoints = []
        for j, c in enumerate(s.STARTS):
            def returned(kind, value, index=j):
                phases.returned(qkey if kind == "q" else tkey, index, value)
            _q, _t, state, _ = run(rollout, window_state(j), context["drive"][4*j:4*j+4], theta,
                                  prefix + "-" + str(c), returned=returned)
            endpoints.append(state)
        return raw[qkey], raw[tkey], endpoints

    def startup99(rollout, theta, prefix, qkey, tkey):
        state = startup["control0"]
        phases.allocate(qkey, (100, 42))
        phases.allocate(tkey, (100, 6))
        phases.returned(qkey, 0, state["q"])
        phases.returned(tkey, 0, state["tendon"])
        for c in range(99):
            def returned(kind, value, index=c+1):
                phases.returned(qkey if kind == "q" else tkey, index, value[0])
            _q, _t, state, _ = run(rollout, state, startup["drive"][c:c+1], theta, prefix + "-" + str(c + 1), returned=returned)
        return raw[qkey], raw[tkey], state

    q0, t0, end0 = startup99(bundle["rollout"], ones, "neutral-startup", "neutral_startup_q", "neutral_startup_tendon")
    for key in s.STATE_KEYS:
        raw["neutral_endpoint-" + key] = end0[key].copy()  # returned counted99; no additional forward
    checks = {}
    for key, value in (("q", q0[1:]), ("tendon", t0[1:]), *end0.items()):
        target = startup["endpoint"][key]
        passed = core.exact(value, target) if key in ("rate", "adaptation") else np.allclose(value, target, rtol=0, atol=1e-11 if key in ("q", "tendon") else 1e-10)
        checks[key] = bool(passed)
    s.require(all(checks.values()), "current counted99 historical E318 reproduction")
    comparisons = {"command": replay.compare_output(q0, control["command"][:100], np), "tendon": replay.compare_output(t0, control["tendon"][:100], np)}
    result["startup_source_checks"] = {"E318": checks, "current_neutral": comparisons, "full_current_startup_sensory_neural_equivalence": "NOT_OBSERVED",
                                        "immutable_binder_NOT_RUN_unchanged": True}
    s.require(all(v["passed"] for v in comparisons.values()), "neutral/current startup outputs prerequisite")
    result["neutral_startup"] = body.startup99(q0, t0, **startup["target"])
    s.require(result["neutral_startup"]["passed"], "baseline startup four limits")
    for key, shape in (("q_neutral", (4, 4, 42)), ("tendon_neutral", (4, 4, 6)), ("gq", (4, 4, 42)),
                       ("gtheta", (4, 8547)), ("d0", (4, 6, 3)), ("measured_saved_projection", (4, 6, 3))):
        phases.allocate(key, shape)
    for j, c in enumerate(s.STARTS):
        theta = torch.ones(8547, dtype=torch.float64, device=device, requires_grad=True)
        def seed(q, window=j, control_id=c):
            ids = np.arange(control_id + 1, control_id + 5)
            s.require(replay.compare_output(q, control["command"][ids], np)["passed"], "nominal saved16 q correspondence before backward")
            d, _, _ = s.projection(q[0], frozen, window, np)
            observed, _, _ = s.projection(control["command"][control_id + 1], frozen, window, np)
            phases.returned("d0", window, d)
            phases.returned("measured_saved_projection", window, observed)
            baselines = np.zeros((4, 6, 3), np.float64)
            baselines[window] = d
            _, value = s.window_loss(q, frozen, baselines, window, coefficients, np)
            phases.returned("gq", window, value)
            return value
        def returned(kind, value, index=j):
            phases.returned("q_neutral" if kind == "q" else "tendon_neutral", index, value)
        def gradient_returned(value, index=j):
            phases.returned("gtheta", index, value)
        q, t, _state, gradient = run(bundle["rollout"], window_state(j), context["drive"][4*j:4*j+4], theta,
                                   "nominal-backward-" + str(c), backward_seed=seed, returned=returned, gradient_returned=gradient_returned)
        s.require(replay.compare_output(t, control["tendon"][c+1:c+5], np)["passed"], "nominal saved16 tendon correspondence")
        del theta, gradient, _state
    raw["sum_gtheta"] = raw["gtheta"].sum(axis=0)
    result["neutral_loss"] = s.evaluate(raw["q_neutral"], frozen, raw["d0"], coefficients, np)
    s.require(result["neutral_loss"]["keep"] == 0, "nominal fixed replay keep exactly zero")
    result["replay_measured_projection_max_drift_mm"] = float(np.max(np.abs(raw["d0"] - raw["measured_saved_projection"])))
    magnitude = float(np.max(np.abs(raw["sum_gtheta"])))
    s.require(np.isfinite(magnitude), "finite summed surrogate gradient")
    if magnitude == 0:
        result["no_candidate_reason"] = "finite_zero_gradient"
        return
    direction = -raw["sum_gtheta"] / magnitude
    raw["direction"] = direction
    probe_losses, represented = {}, True
    for sign, name in ((1., "plus"), (-1., "minus")):
        actual = (1 + sign * contract["eta"] * direction).astype(np.float32).astype(np.float64)
        weights, palette = core.patch_weights(packet["actual_csr"], packet["recurrent_mask"], actual)
        arrays = {"theta_actual": actual, "selected_weights_candidate": weights}
        changed = not np.array_equal(weights, packet["actual_csr"]["data"][packet["recurrent_mask"]["positions"]])
        represented = represented and changed and bool(np.all(actual > 0))
        raw["probe-" + name + ".theta_actual"], raw["probe-" + name + ".selected_weights_candidate"] = actual, weights
        deployed = ray.deployed_rollout(bundle, packet, arrays)
        q, t, endpoints = windows(deployed, ones, "probe-" + name, "probe-" + name + ".q", "probe-" + name + ".tendon")
        result["probe_evaluations"] += 1
        probe_losses[name] = s.evaluate(q, frozen, raw["d0"], coefficients, np)
        result.setdefault("probe_representation", {})[name] = {"changed": changed, "theta_sha256": s.sha(actual), "selected_sha256": s.sha(weights), "palette": palette}
        del deployed, endpoints
    slope = float(raw["sum_gtheta"] @ direction)
    secant = (probe_losses["plus"]["total"] - probe_losses["minus"]["total"]) / (2 * contract["eta"])
    s.require(np.isfinite(slope) and np.isfinite(secant), "finite diagnostic secant and surrogate dot")
    result["finite_probe"] = {"eta": contract["eta"], "losses": probe_losses, "gradient_dot_direction": slope, "central_secant": secant,
                              **s.secant_diagnostic(slope, secant), "exact_native_VJP_correctness": "UNVERIFIED",
                              "passed": s.finite_probe_pass(result["neutral_loss"], probe_losses["plus"], probe_losses["minus"], represented)}
    if not result["finite_probe"]["passed"]:
        result["no_candidate_reason"] = "completed_fixed_finite_descent_rejected"
        return
    for i, alpha in enumerate(contract["alphas"]):
        evidence, arrays = ray.proposal(raw["sum_gtheta"], float(alpha), packet["actual_csr"], packet["recurrent_mask"], result["alphas"])
        row = {**evidence, "index": i, "eligible": False, "forward_evaluated": False}
        result["alphas"].append(row)
        if arrays:
            for key, value in arrays.items():
                raw[f"alpha-{i}." + key] = value.copy()
        if not evidence["eligible_representation"]:
            continue
        deployed = ray.deployed_rollout(bundle, packet, arrays)
        qr, tr, er = windows(bundle["rollout"], tensor(arrays["theta_actual"]), f"alpha-{i}-relative", f"alpha-{i}.q_relative", f"alpha-{i}.tendon_relative")
        qd, td, ed = windows(deployed, ones, f"alpha-{i}-deployed", f"alpha-{i}.q", f"alpha-{i}.tendon")
        qs, ts, es = startup99(bundle["rollout"], tensor(arrays["theta_actual"]), f"alpha-{i}-relative-startup", f"alpha-{i}.startup_q_relative", f"alpha-{i}.startup_tendon_relative")
        qds, tds, eds = startup99(deployed, ones, f"alpha-{i}-deployed-startup", f"alpha-{i}.startup_q", f"alpha-{i}.startup_tendon")
        for key in s.STATE_KEYS:
            raw[f"alpha-{i}.relative_endpoint-" + key] = np.stack([v[key] for v in er])
            raw[f"alpha-{i}.deployed_endpoint-" + key] = np.stack([v[key] for v in ed])
            raw[f"alpha-{i}.relative_startup_endpoint-" + key] = es[key].copy()
            raw[f"alpha-{i}.deployed_startup_endpoint-" + key] = eds[key].copy()
        corresponding = all(np.allclose(a, b, atol=1e-12, rtol=1e-10) for a, b in ((qr, qd), (tr, td), (qs, qds), (ts, tds),
                        *((a[k], b[k]) for a, b in zip(er, ed, strict=True) for k in s.STATE_KEYS), *((es[k], eds[k]) for k in s.STATE_KEYS)))
        s.require(corresponding, "actual relative vs deployed q/tendon/state correspondence")
        row.update(forward_evaluated=True, output_correspondence=True, loss=s.evaluate(qd, frozen, raw["d0"], coefficients, np),
                   startup=body.startup99(qds, tds, **startup["target"]))
        row["eligible"] = s.eligible(result["neutral_loss"], row["loss"], row["startup"], True, corresponding, coefficients)
        result["evaluated_alphas"] += 1
        del deployed, er, ed, es, eds
        if row["eligible"]:
            result["candidate"] = {"schema": contract["candidate_schema"], "stage_id": contract["stage_id"], "index": i, "alpha": float(alpha),
                "theta_sha256": row["theta_sha256"], "selected_sha256": row["selected_sha256"], "palette": row["palette"],
                "source_freeze": admission.request["source_freeze"], "roster": admission.request["roster"], "contract": admission.request["contract"],
                "geometry": admission.roster["neutral_capture"]["geometry"], "loss": coefficients, "exact_native_VJP_correctness": "UNVERIFIED"}
            result["candidate_present"] = True
            return
    result["no_candidate_reason"] = "finite_no_eligible_alpha"


def direction_worker(admission):
    from tools import banc_m3_contact_learning as s
    from tools import verify_c_banc_m3_candidate_body_handoff as monitoring
    from tools.verify_c_banc_m3_live_replay import publisher_for
    admission.fence("pre_array")
    publisher, watch, saved = publisher_for(admission), None, None
    result = {"schema": s.SCHEMA + ".direction.v1", "role": admission.role, "run_valid": False, "candidate_present": False,
              "contract": admission.request["contract"], "source_freeze": admission.request["source_freeze"], "roster": admission.request["roster"],
              "loss_coefficients": admission.contract["loss"], "candidate_geometry": "UNOBSERVED", "exact_native_VJP_correctness": "UNVERIFIED",
              "candidate": None, "classification": "technical_invalid", "first_error": None, "secondary_errors": [], "receipts": {},
              "alphas": [], "evaluated_alphas": 0, "probe_evaluations": 0, "candidate_adopted": False, "F1_F8": "INCOMPLETE", "biology": "UNPROVEN"}
    raw, ledger, stage, saved_box = {}, [], "GPU-sampler", {}
    try:
        watch = monitoring.gpu_watch({"limits": admission.limits})
        stage = "bounded-input-preparation"
        packet, context, control, frozen, startup, saved = load(admission, result, ledger, saved_box)
        stage = "finite-direction"
        collect(admission, packet, context, control, frozen, startup, result, raw, watch)
        result["work"] = s.work_summary(result["receipts"], admission.contract, evaluated=result["evaluated_alphas"], probes=result["probe_evaluations"], complete=True)
        result.update(run_valid=True, classification="candidate_pending_arithmetic_audit" if result["candidate_present"] else "scientific_no_candidate")
    except Exception as error:
        failed(result, stage, error)
    finally:
        saved = saved or saved_box.get("reader")
        if watch is not None:
            try:
                result["sampler"] = watch.close()
                monitoring.monitor_check(result["sampler"], admission.limits, closed=True)
            except Exception as error:
                failed(result, "sampler-close", error)
        try:
            if saved is not None:
                result["input_pre_post_hashes"] = saved.posthash()
            from tools import verify_c_banc_m3_coordination_parameter_direction as core
            for entry in ledger:
                s.require(core.digest(core.safe(admission.root, entry["path"]), admission.check) == entry["expected_sha256"], "Original selected posthash")
        except Exception as error:
            failed(result, "input-posthash", error)
        publish(publisher, raw, result, ledger, admission)
    s.require(result["run_valid"], "technical direction failure retained")
    return result


def body_worker(admission):
    from tools import banc_m3_contact_learning as s
    from tools import verify_c_banc_m3_candidate_body_handoff as monitoring
    from tools import verify_c_banc_m3_common_descent_body as b
    from tools import verify_c_banc_m3_startup_ray as ray
    from tools.banc_m3_contact_runtime_journal import phase
    from tools.verify_c_banc_m3_live_replay import publisher_for
    journal = getattr(admission, "runtime_journal", None)
    saved, publisher = None, publisher_for(admission)
    result = {"schema": s.SCHEMA + ".body-failure.v1", "run_valid": False, "candidate_present": True,
              "candidate": None, "first_error": None, "secondary_errors": [], "classification": "technical_invalid",
              "work_observation": "NOT_STARTED_OR_NOT_RETURNED; no invented zero constructor receipt"}
    watch, stage, held_report = None, "new-candidate-input", {}
    original_json = publisher.json
    def observed_json(name, value):
        if name == "body-report.json":
            held_report["record"] = value  # retain acquired actual work if that publication refuses
        return original_json(name, value)
    publisher.json = observed_json
    try:
        import numpy as np
        with phase(admission, "body-entry", opaque=True):
            from tools import banc_m3_checkpoint_harness_runtime as harness
            from tools import banc_m3_checkpoint_runtime as checkpoint
            if admission.request['call_id'] in harness.CALLS:
                checkpoint = harness
            if admission.request['call_id'] in checkpoint.CALLS:
                checkpoint.body_binding(admission.request, admission.contract, admission.read)
            saved = s.Saved(admission, np)
            audit = saved.json(admission.request["artifacts"]["report"])
            s.require(audit["run_valid"] and audit["candidate_present"] and audit["arithmetic_valid"], "one independently audited new candidate")
            identity = audit["candidate"]
        with phase(admission, "body-plan", opaque=True):
            result["input_plan"] = s.body_plan(admission, saved, admission.request["artifacts"]["candidate_arrays"], np)
        with phase(admission, "candidate-load", opaque=True):
            arrays = saved.arrays(admission.request["artifacts"]["candidate_arrays"], {"theta_actual": ((8547,), "float64"), "selected_weights_candidate": ((725849,), "float32")})
        with phase(admission, "candidate-hash", opaque=True):
            admission.check("science")
            s.require(s.sha(arrays["theta_actual"]) == identity["theta_sha256"] and s.sha(arrays["selected_weights_candidate"]) == identity["selected_sha256"], "actual new candidate deployment identity")
        watch = monitoring.gpu_watch({"limits": admission.limits})
        stage = "existing-continuous-candidate-body"
        with phase(admission, "arm-data", opaque=True):
            data = ray.arm_data(arrays, admission.inputs["body_inventory"]["inventory"], admission, "candidate", b=b)
        input_plan = result["input_plan"]
        result = ray.continuous_arm(data, "candidate", identity, {"palette": identity["palette"]}, admission, publisher, watch)
        result["input_plan"] = input_plan
        result["candidate_present"] = True
    except Exception as error:
        result = held_report.get("record", result)
        if journal is not None:
            journal.acquire_errors(result)
            journal.error(stage, error)
        failed(result, stage, error)
    finally:
        try:
            if watch is not None and watch.snapshot()["state"] != "CLOSED":
                result["sampler_fallback"] = watch.close()
                monitoring.monitor_check(result["sampler_fallback"], admission.limits, closed=True)
        except Exception as error:
            failed(result, "body-worker-sampler-finally", error)
        try:
            if saved is not None:
                with phase(admission, "posthash", opaque=True):
                    result["selected_candidate_input_pre_post"] = saved.posthash()
        except Exception as error:
            failed(result, "body-worker-posthash", error)
        publisher.json = original_json
    result["candidate_present"] = True
    result.update(candidate_adopted=False, F1_F8="INCOMPLETE", biology="UNPROVEN", contact_stage=admission.request["contract"],
                  existing_continuous_arm="verify_c_banc_m3_startup_ray.continuous_arm", candidate_geometry="UNOBSERVED")
    artifacts = {}
    try:
        admission.check("science")  # no new artifact scan/hash after ENTRY+270
        artifacts = {key: ray.spec_for((publisher.folder / filename).relative_to(admission.root).as_posix(), admission.root, cap=admission.limits["output_cap_bytes"])
                     for key, filename in (("control", "control.npz"), ("native", "native.npz"), ("trace", "trace.json"), ("evaluation", "evaluation.npz"))
                     if (publisher.folder / filename).exists()}
    except Exception as error:
        failed(result, "body-artifact-descriptors", error)
    try:
        with phase(admission, "science-publication", opaque=True):
            admission.check("science")
            if journal is not None:
                journal.acquire_errors(result)
                result["runtime_journal"] = journal.snapshot()
            artifacts["report"] = publisher.write("contact-body-report.json", s.wire(monitoring.normalize(result), s.REPORT_CAP))
            publisher.json("contact-manifest.json", {"schema": s.SCHEMA + ".manifest.v1", "role": admission.role,
                "source_freeze": admission.request["source_freeze"], "request": admission.request_spec,
                "contract": admission.request["contract"], "run_valid": result["run_valid"], "artifacts": artifacts})
    except Exception as error:
        failed(result, "contact-body-publication", error)
        terminal_failure(publisher, result, "contact-body-publication")
        try:
            publisher.json("contact-body-publication-failure.json", result)
        except Exception as secondary:
            failed(result, "contact-body-fallback", secondary)
    if journal is not None:
        journal.acquire_errors(result)
        # This channel reads no Original or Source and never attests run validity.
        if not result["run_valid"]:
            journal.terminal_failure()
        else:
            journal.emit()
        result["runtime_journal"] = journal.snapshot()
    s.require(result["run_valid"], "actual new candidate technical failure retained: " + json.dumps(result["first_error"]))
    return result


def main(argv=None):
    from tools.banc_m3_contact_learning_admission import Admission
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    origin = float(json.loads(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"])["entry_counter_s"])
    owner = json.loads(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"])
    journal = None
    if owner["call_id"] in ("contact-learning-20261007-d-body", "contact-learning-20261008-e-body", "contact-learning-20261008-f-body"):
        from tools import banc_m3_startup_ray_supervisor as supervisor
        from tools.banc_m3_contact_runtime_journal import (
            RuntimeJournal,
            owned_stream_guard,
        )
        pid = os.getpid()
        creation = supervisor.creation_ticks(pid)
        journal = RuntimeJournal(origin, owner, guard=lambda: owned_stream_guard(owner, pid, creation))
    try:
        from contextlib import nullcontext
        if journal is not None:
            from tools import verify_c_banc_m3_startup_ray as ray
            journal.science()
            owner_request = ray.json_load(ray.leaf(owner["request"]))
            journal.science()
            if owner_request["science_request"]["path"] != args.request:
                raise ValueError("same acquired owner science request")
            journal.bind(owner_request, owner_request["science_request"])
        context = nullcontext() if journal is None else journal.phase("admission-entry", opaque=True)
        with context:
            admission = Admission(args.request, entry=origin, runtime_journal=journal)
        if admission.role not in ("contact_direction", "contact_candidate_body"):
            raise ValueError("direction/body entrypoint role")
        result = direction_worker(admission) if admission.role == "contact_direction" else body_worker(admission)
    except Exception as error:
        if journal is not None and not journal.terminal_attempted:
            journal.error("body-entry", error)
            journal.terminal_failure()
        raise
    print(json.dumps({"run_valid": result["run_valid"], "candidate_present": result.get("candidate_present", True), "candidate_adopted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
