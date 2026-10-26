"""Recorded-state output correspondence only:32 controls/640 SSP ticks, physics0."""
# ruff: noqa: BLE001 -- preserve partial raw/work and primary/cleanup errors.
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time

ENTRY = time.perf_counter()
SCHEMA = "flylab.m3.live-replay"
STARTS = (100, 164, 228, 292)
ENDS = tuple(c + j for c in STARTS for j in range(1, 5))
ARMS = ("tracking_neutral", "tracking_candidate")
PACKET_ROLES = ("actual_csr", "motor_configuration", "load_parameters", "recurrent_mask", "e289_roster", "calibration", "parameters", "inverse", "output_mask")
PACKET_SCHEMA = {
    "actual_csr": {"data": ((11584852,), "<f4"), "indices": ((11584852,), "<i4"), "indptr": ((158707,), "<i4")},
    "motor_configuration": {"cell_gain": ((158706,), "<f4"), "membrane_tau": ((158706,), "<f4"), "motor_ids": ((391,), "<i8"),
                            "actual_motor_inverse32": ((391,), "<f4"), "actual_motor_gain32": ((391,), "<f4")},
    "load_parameters": {"threshold32": ((158706,), "<f4"), "tau32": ((158706,), "<f4")},
    "recurrent_mask": {"population": ((7136,), "<i8"), "positions": ((725849,), "<i8"), "edge_parameters": ((725849,), "<i8")},
    "e289_roster": {"premotor_ids": ((7136,), "<i8"), "edge_ids": ((725849,), "<i8"), "weights32": ((725849,), "<f4")},
    "calibration": {"moment": ((42, 96), "<f8"), "rest": ((42,), "<f8"), "pool": ((96, 391), "<f8"), "motor_ids": ((391,), "<i8"), "ltm": ((6,), "<i8")},
    "parameters": {"tau": ((3,), "<f8"), "kappa": ((391,), "<f8"), "slopes": ((96, 391, 3, 2), "<f8"), "membership": ((391,), "<i8")},
    "inverse": {"value": ((158706,), "<f4")}, "output_mask": {"value": ((158706,), "<f4")},
}
CANDIDATE_FIELDS = {"theta_actual": ((8547,), "float64"), "selected_weights_candidate": ((725849,), "float32")}


def require(ok, label):
    if not ok:
        raise ValueError(label)


def sha(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def schema_nbytes(shape, dtype, member_cap):
    """Metadata arithmetic only; never construct a NumPy dtype/array."""
    widths = {"<f4": 4, "float32": 4, "<f8": 8, "float64": 8, "<i4": 4, "int32": 4,
              "<i8": 8, "int64": 8, "|b1": 1, "bool": 1}
    require(type(shape) in (tuple, list) and 1 <= len(shape) <= 4
            and all(type(axis) is int and axis > 0 for axis in shape) and type(dtype) is str and dtype in widths, "bounded selected metadata shape/dtype")
    count = widths[dtype]
    for axis in shape:
        count *= axis
    require(count <= member_cap, "selected member allocation cap")
    return count


def plan_inputs(bindings, captures, candidate, limits, saved_fields):
    """Plan the exact selected members, including both saved arms, before loading."""
    plan = {"packet": {}, "saved": {}, "packet_expanded_bytes": 0, "saved_expanded_bytes": 0}

    def selected_file(spec, fields, cap):
        require(type(spec["bytes"]) is int and 0 < spec["bytes"] <= limits["input_file_cap_bytes"], "selected file96MiB cap")
        members = {key: schema_nbytes(shape, dtype, cap) for key, (shape, dtype) in fields.items()}
        return {"path": spec["path"], "file_bytes": spec["bytes"], "member_bytes": members, "expanded_bytes": sum(members.values())}

    for role in PACKET_ROLES:
        binding, expected = bindings[role], PACKET_SCHEMA[role]
        require(binding["kind"] == ("npy" if role in ("inverse", "output_mask") else "npz")
                and set(binding["selected_schema"]) == set(expected), "exact selected packet kind/members")
        for key, (shape, dtype) in expected.items():
            spec = binding["selected_schema"][key]
            require(type(spec["shape"]) is list and spec["shape"] == list(shape) and spec["dtype"] == dtype
                    and spec.get("fortran_order", False) is False and spec["finite"] is True, "pinned selected packet shape/dtype " + role + "." + key)
        entry = selected_file(binding, expected, limits["legacy_packet_member_cap_bytes"])
        plan["packet"][role] = entry
        plan["packet_expanded_bytes"] += entry["expanded_bytes"]
    for arm in ARMS:
        for kind in ("context", "control"):
            key = arm + "." + kind
            plan["saved"][key] = selected_file(captures[arm][kind], saved_fields[kind], limits["member_cap_bytes"])
    plan["saved"]["candidate"] = selected_file(candidate, CANDIDATE_FIELDS, limits["member_cap_bytes"])
    plan["saved_expanded_bytes"] = sum(entry["expanded_bytes"] for entry in plan["saved"].values())
    plan["combined_expanded_bytes"] = plan["packet_expanded_bytes"] + plan["saved_expanded_bytes"]
    require(plan["combined_expanded_bytes"] <= limits["selected_expanded_cap_bytes"]
            and plan["combined_expanded_bytes"] <= limits["host_bytes"], "combined selected packet plus saved allocation plan")
    return plan


def reconcile_arrays(values, planned):
    require(set(values) == set(planned["member_bytes"])
            and all(value.nbytes == planned["member_bytes"][key] for key, value in values.items()), "actual selected nbytes reconcile with preallocation plan")
    return sum(value.nbytes for value in values.values())


def compare_output(actual, expected, np):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype == np.float64
            and np.isfinite(actual).all() and np.isfinite(expected).all(), "finite corresponding raw requested outputs")
    difference = actual - expected
    maximum = float(np.max(np.abs(difference)))
    return {"exact": bool(np.array_equal(actual, expected)), "max_abs": maximum,
            "rms": float(np.sqrt(np.mean(difference * difference))), "count": int(difference.size),
            "atol": 1e-11, "rtol": 0, "passed": maximum <= 1e-11}


def selected_deployment(source, positions, selected, np):
    """Patch already deployed f32 weights. This function never multiplies theta."""
    require(source.dtype == selected.dtype == np.float32 and positions.dtype == np.int64
            and positions.ndim == selected.ndim == 1 and positions.shape == selected.shape
            and len(np.unique(positions)) == len(positions) and np.all((positions >= 0) & (positions < len(source)))
            and np.isfinite(source).all() and np.isfinite(selected).all(), "actual selected deployment bytes")
    deployed = source.copy()
    deployed[positions] = selected
    return deployed


def validate_recorded(context, control, report, packet, np, *, n=158706, resource_n=7136):
    require(np.array_equal(context["pre_control"], STARTS) and np.array_equal(context["drive_control_end"], ENDS), "exact recorded windows/drive order")
    require(context["pre_valid"].all() and context["drive_valid"].all(), "four pre-states and16 finalized drives complete")
    require(np.array_equal(context["pre_network"], 20 * context["pre_control"])
            and np.all(np.abs(context["pre_time_s"] - .005 * context["pre_control"]) <= 2e-8)
            and np.array_equal(context["drive_network_pre"], 20 * (context["drive_control_end"] - 1))
            and np.all(np.abs(context["drive_time_pre_s"] - .005 * (context["drive_control_end"] - 1)) <= 2e-8), "recorded c+1 entry clocks")
    shapes = {"rate": ((4, n), np.float32), "adaptation": ((4, n), np.float32), "drive": ((16, n), np.float32),
              "bank": ((4, 391, 3), np.float64), "original_activation": ((4, 96), np.float64), "resource": ((4, resource_n), np.float64)}
    for name, (shape, dtype) in shapes.items():
        value = context[name]
        require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(), "actual respective state/drive " + name)
        if name != "drive":
            require(np.all(value >= 0), "nonnegative recorded state " + name)
    require(context["source_ids"].shape == (resource_n,) and context["source_ids"].dtype == np.int64
            and np.array_equal(context["source_ids"], np.unique(packet["actual_csr"]["indices"][packet["recurrent_mask"]["positions"]]))
            and np.all(context["utilization"] == 0) and np.all(context["resource"] == 1)
            and context["u_zero"].all() and context["resource_one"].all()
            and np.isfinite(context["recovery_tau_s"]).all() and np.all(context["recovery_tau_s"] > 0), "ordered source7136/resource1/u0 neutral proof")
    require(np.array_equal(context["motor_ids"], packet["calibration"]["motor_ids"])
            and np.array_equal(context["motor_ids"], packet["motor_configuration"]["motor_ids"])
            and np.array_equal(context["output_mask"], packet["output_mask"]["value"]), "actual ordered motor391/output mask")
    require(report["run_valid"] is True and report["capture_complete"] is True and report["all_owned_ended"] is True,
            "actual closed respective capture")
    require(all(report["context_array_sha256"].get(k) == sha(v) for k, v in context.items()), "recorded finite input byte identity")
    require(np.array_equal(control["control_tick"], np.arange(301))
            and np.array_equal(control["network_tick"], 20 * control["control_tick"])
            and np.all(np.abs(control["physics_time_s"] - .005 * control["control_tick"]) <= 2e-8), "saved output exact ordered control boundaries")
    require(report["transfer"]["source_network_identity"] == packet["metadata"]["model_identity"]
            and report["transfer"]["source_adapter_identity"] == packet["metadata"]["decoder_hash"], "original calibrated decoder/neural provenance")
    return {"recorded_input_identity": True, "windows": list(STARTS), "drive_ends": list(ENDS),
            "resource_interpretation": "u0/resource1 admits neutral recurrent current; no resource replay needed",
            "post_neural_decoder_endpoints": "NOT_OBSERVED", "readout_adhesion_full_neural_equivalence": "NOT_OBSERVED"}


def verify_receipts(receipts, *, full):
    expected_windows = len(receipts)
    total = {key: 0 for key in ("forward_controls_requested", "forward_controls_returned", "forward_neural_ticks_requested", "forward_neural_ticks_returned",
                              "recompute_controls_requested", "recompute_controls_returned", "recompute_neural_ticks_requested", "recompute_neural_ticks_returned",
                              "backward_blocks_requested", "backward_blocks_returned", "backward_block_ticks_returned")}
    confirmed_controls = confirmed_ticks = 0
    for value in receipts.values():
        for key in total:
            require(type(value[key]) is int and value[key] >= 0, "typed actual work receipt")
            total[key] += value[key]
        if full:
            require(value["status"] == "forward_returned" and value["forward_controls_requested"] == value["forward_controls_returned"] == 4
                    and value["forward_neural_ticks_requested"] == value["forward_neural_ticks_returned"] == 80
                    and value["confirm_requested"] == value["confirm_returned"] == 1, "one actual four-control forward/confirm")
        confirmed = value.get("confirmed")
        if confirmed:
            confirmed_controls += confirmed["forward_controls"]
            confirmed_ticks += confirmed["forward_neural_ticks"]
            require(confirmed["recompute_controls"] == confirmed["recompute_neural_ticks"] == confirmed["backward_blocks"] == confirmed["backward_block_ticks"] == 0, "confirmed no recompute/backward")
    require(total["forward_controls_requested"] <= 32 and total["forward_neural_ticks_requested"] <= 640
            and all(total[key] == 0 for key in total if key.startswith(("recompute", "backward"))), "fixed total640, recompute/backward0")
    if full:
        require(expected_windows == 8 and total["forward_controls_requested"] == total["forward_controls_returned"] == confirmed_controls == 32
                and total["forward_neural_ticks_requested"] == total["forward_neural_ticks_returned"] == confirmed_ticks == 640,
                "all32controls/640requested returned device-confirmed")
    return {**total, "forward_controls_device_confirmed": confirmed_controls, "forward_neural_ticks_device_confirmed": confirmed_ticks,
            "warmup_ticks": 0, "capture_ticks": 0, "body_constructors": 0, "physics_steps": 0, "extra_native_comparator_ticks": 0, "gradient": 0}


def minimal_assembly(packet, *, device="cuda"):
    """Existing constructors only; no objective, checkpoint body, warmup or forward."""
    import numpy as np
    import torch
    from scipy.sparse import csr_matrix

    from tools.banc_m3_checkpointed_rollout import CheckpointedControlRollout
    from tools.banc_m3_grouped_current_adjoint import GroupedCurrent
    from tools.banc_m3_monotone_adjoint import MonotoneDecoder
    from tools.banc_m3_ssp_adjoint import SSPAdjointModel

    csr, mask = packet["actual_csr"], packet["recurrent_mask"]
    n = len(csr["indptr"]) - 1
    weights = csr_matrix((csr["data"], csr["indices"], csr["indptr"]), shape=(n, n))
    operator = GroupedCurrent(weights, mask["positions"], mask["edge_parameters"], 8547, device)
    load, config = packet["load_parameters"], packet["motor_configuration"]
    require(np.array_equal(load["tau32"], config["membrane_tau"]) and np.all(load["threshold32"] == np.float32(7.5))
            and np.array_equal(packet["inverse"]["value"][config["motor_ids"]], config["actual_motor_inverse32"])
            and np.array_equal(config["cell_gain"][config["motor_ids"]], config["actual_motor_gain32"]), "exact calibrated deployed physiology")
    neural = SSPAdjointModel(operator, load["tau32"], np.full(n, np.float32(1 / 200), np.float32), load["threshold32"],
                             np.full(n, np.float32(200), np.float32), config["cell_gain"], packet["inverse"]["value"], .00025)
    p = packet["parameters"]
    decoder = MonotoneDecoder(packet["calibration"], p["tau"], p["kappa"], p["slopes"], p["membership"],
                              np.asarray(packet["metadata"]["neutral"], np.float64), device=device)
    rollout = CheckpointedControlRollout(neural, decoder, packet["calibration"]["motor_ids"], steps_per_control=20,
                                         control_dt=.005, checkpoint_controls=1, max_controls=4)
    torch.set_num_threads(1)
    return {"operator": operator, "rollout": rollout}


def publisher_for(admission):
    from tools import verify_c_banc_m3_startup_ray as ray

    class Publisher(ray.Publisher):
        def write(self, name, raw):
            require(self.total + len(raw) <= admission.limits["output_cap_bytes"] - (3 << 20), "replay result13MiB plus supervision reserve")
            return super().write(name, raw)

        def arrays(self, name, values):
            import numpy as np
            estimate = sum(v.nbytes for v in values.values()) + 1024 * len(values)
            require(estimate + self.total <= admission.limits["output_cap_bytes"] - (3 << 20), "temporary bounded raw serialization")
            buffer = io.BytesIO()
            np.savez(buffer, **values)
            require(buffer.tell() <= estimate and 2 * estimate <= admission.limits["host_bytes"], "serialization actual bytes/host")
            return self.write(name, buffer.getvalue())

    return Publisher(admission)


def markdown(result):
    lines = ["# 실제 저장 입력의 조건부 출력 대응", "", f"run_valid={result['run_valid']}, output_correspondence_passed={result['output_correspondence_passed']}.", ""]
    for key, value in result["comparisons"].items():
        lines.append(f"- {key}: command max/RMS={value['command']['max_abs']}/{value['command']['rms']}, tendon max/RMS={value['tendon']['max_abs']}/{value['tendon']['rms']}; 기준 abs≤1e-11/rtol0.")
    if result.get("first_error"):
        lines += ["", "최초 오류: " + json.dumps(result["first_error"], ensure_ascii=False)]
    if result.get("secondary_errors"):
        lines += ["후속 오류: " + json.dumps(result["secondary_errors"], ensure_ascii=False)]
    lines += ["", "합32controls/640ticks의 출력 대응입니다. post 신경·decoder endpoint/MN readout/adhesion은 NOT_OBSERVED이며 VJP·인과·보행 성공 증거가 아닙니다.",
              "body/physics/geometry/gradient/새 후보0. 후보 채택 false, F1–F8 INCOMPLETE, biology UNPROVEN.",
              "불일치면 대응 경로의 차이를 먼저 검토합니다. 대응 시 ROOT가 다음 과학적 변경이나 조건부 기하의 필요성을 판단합니다."]
    return "\n".join(lines) + "\n"


def publish_result(publisher, raw, source_ledger, result):
    """Try each bounded publication; preserve the primary error and partial work."""
    def failed(stage, error):
        record = {"stage": stage, "type": type(error).__name__, "message": str(error)}
        if result["first_error"] is None:
            result["first_error"] = record
        else:
            result["secondary_errors"].append(record)
        result["run_valid"] = result["output_correspondence_passed"] = False
        result["classification"] = "technical_invalid"

    operations = (("publish-raw", lambda: publisher.arrays("raw.npz", raw)),
                  ("publish-receipts", lambda: publisher.json("receipts.json", result["receipts"])),
                  ("publish-selected-ledger", lambda: publisher.json("selected-ledger.json", source_ledger)))
    for stage, operation in operations:
        try:
            operation()
        except Exception as error:
            failed(stage, error)
    result["durable_result_bytes_before_report"] = publisher.total
    try:
        publisher.write("report.md", markdown(result).encode("utf8"))
    except Exception as error:
        failed("publish-report-md", error)
    # JSON is authoritative and is written after any earlier publication refusal.
    try:
        publisher.json("report.json", result)
    except Exception as error:
        failed("publish-report-json", error)
        # New failure filenames do not retry/overwrite a possibly partial file.
        for stage, operation in (("publish-failure-json", lambda: publisher.json("publication-failure.json", result)),
                                 ("publish-failure-md", lambda: publisher.write("publication-failure.md", markdown(result).encode("utf8")))):
            try:
                operation()
            except Exception as secondary:
                failed(stage, secondary)
    return result


def worker(admission):
    require(admission.role == "live_replay" and admission.stage == "pre_array", "one Original replay-only role")
    admission.fence("pre_array")
    from tools import verify_c_banc_m3_candidate_body_handoff as monitoring
    publisher = publisher_for(admission)
    result = {"schema": SCHEMA + ".report.v1", "run_valid": False, "output_correspondence_passed": False,
              "classification": "technical_invalid", "first_error": None, "secondary_errors": [], "comparisons": {}, "receipts": {},
              "candidate_adopted": False, "F1_F8": "INCOMPLETE", "biology": "UNPROVEN", "M2_UI": "PRESERVED",
              "post_neural_decoder_endpoints": "NOT_OBSERVED", "VJP_live_credit": "UNVERIFIED", "body_physics_geometry_gradient_new_candidate": 0}
    raw, source_ledger, saved, watch, stage = {}, [], None, None, "gpu-sampler-start"
    try:
        watch = monitoring.gpu_watch({"limits": admission.limits})  # before Torch/numerical imports
        stage = "selected-input-preplan"
        from tools import analyze_c_banc_m3_startup_tracking as tracking
        from tools import verify_c_banc_m3_coordination_parameter_direction as core
        from tools import verify_c_banc_m3_startup_ray as ray
        inventory = admission.inputs["legacy_inventory"]
        saved_fields = {"context": tracking.CONTEXT_FIELDS,
                        "control": {key: (tuple(301 if axis is None else axis for axis in shape), dtype)
                                    for key, (shape, dtype) in tracking.CONTROL_FIELDS.items()}}
        result["selected_input_plan"] = plan_inputs(inventory["bindings"], admission.roster["captures"],
                                                    admission.roster["candidate_arrays"], admission.limits, saved_fields)
        plan = result["selected_input_plan"]
        stage = "numerical-load"
        import numpy as np
        import torch

        saved = tracking.SavedInputs(admission, np)
        packet = {"metadata": core.metadata_packet(admission.root, inventory)}
        packet_expanded = 0
        for role in PACKET_ROLES:
            admission.fence("pre_array")
            packet[role] = core.load_selected(admission.root, inventory["bindings"][role], source_ledger, timely=admission.check, generated=False)
            packet_expanded += reconcile_arrays(packet[role], plan["packet"][role])
        require(packet_expanded == plan["packet_expanded_bytes"], "actual full selected packet allocation reconciliation")
        csr, mask, roster = packet["actual_csr"], packet["recurrent_mask"], packet["e289_roster"]
        require(csr["data"].shape == csr["indices"].shape == (11584852,) and csr["indptr"].shape == (158707,)
                and csr["data"].dtype == np.float32 and csr["indices"].dtype == csr["indptr"].dtype == np.int32,
                "actual full deployed Original CSR")
        require(np.array_equal(mask["positions"], roster["edge_ids"]) and np.array_equal(mask["population"], roster["premotor_ids"])
                and np.array_equal(csr["data"][mask["positions"]], roster["weights32"])
                and np.array_equal(np.unique(mask["edge_parameters"]), np.arange(8547)), "unchanged selected anatomical weights/groups")
        selected = saved.json(admission.roster["selected_candidate"])
        audit = saved.json(admission.roster["candidate_audit"])
        fixed = admission.roster["candidate_identity"]
        require(selected["alpha"] == fixed["alpha"] == .0001 and selected["index"] == fixed["index"] == 4
                and audit["raw_valid"] is True and audit["selected_candidate"] == selected, "exact previously rejected selected candidate")
        arrays = saved.arrays(admission.roster["candidate_arrays"], CANDIDATE_FIELDS)
        reconcile_arrays(arrays, plan["saved"]["candidate"])
        require(sha(arrays["theta_actual"]) == selected["theta_sha256"] == fixed["theta_sha256"]
                and sha(arrays["selected_weights_candidate"]) == selected["selected_sha256"] == fixed["selected_sha256"], "saved selected float32 candidate and theta identity")
        neutral_bundle = minimal_assembly(packet)
        candidate_rollout = ray.deployed_rollout(neutral_bundle, packet, arrays)
        deployed = selected_deployment(csr["data"], mask["positions"], arrays["selected_weights_candidate"], np)
        require(sha(deployed) == selected["palette"]["full_data_c_order_sha256"], "exact patched deployed full CSR identity")
        del deployed
        ones = torch.ones(8547, dtype=torch.float64, device=neutral_bundle["rollout"].device)
        require(bool(torch.all(ones == 1)), "candidate theta1, no double multiplication")
        for arm in ARMS:
            stage = arm + ".load"
            bundle = admission.roster["captures"][arm]
            report = saved.json(bundle["body_report"])
            context = saved.arrays(bundle["context"], saved_fields["context"])
            control = saved.arrays(bundle["control"], saved_fields["control"])
            reconcile_arrays(context, plan["saved"][arm + ".context"])
            reconcile_arrays(control, plan["saved"][arm + ".control"])
            require(saved.expanded + packet_expanded <= plan["combined_expanded_bytes"], "actual combined selected allocation within preplan")
            result.setdefault("input_checks", {})[arm] = validate_recorded(context, control, report, packet, np)
            expected_data_sha = report["transfer"]["source_CSR_data_sha256"] if arm == ARMS[0] else report["transfer"]["candidate_CSR_data_sha256"]
            actual_data_sha = sha(csr["data"]) if arm == ARMS[0] else selected["palette"]["full_data_c_order_sha256"]
            require(actual_data_sha == expected_data_sha, "respective deployed full CSR/capture proof")
            rollout = neutral_bundle["rollout"] if arm == ARMS[0] else candidate_rollout
            raw[arm + ".q"] = np.zeros((4, 4, 42), np.float64)
            raw[arm + ".tendon"] = np.zeros((4, 4, 6), np.float64)
            raw[arm + ".valid"] = np.zeros((4, 4), bool)
            raw[arm + ".control_end"] = np.asarray(ENDS, np.int64).reshape(4, 4)
            for j, start in enumerate(STARTS):
                key = arm + "." + str(start)
                stage = key
                admission.check("science")
                inputs = [torch.as_tensor(context[name][j].copy(), device=rollout.device).contiguous() for name in ("rate", "adaptation", "bank", "original_activation")]
                drives = torch.as_tensor(context["drive"][4*j:4*j+4].copy(), device=rollout.device).contiguous()
                output_mask = torch.as_tensor(context["output_mask"].copy(), device=rollout.device).contiguous()
                receipt = None
                try:
                    with torch.no_grad():
                        output = rollout.run(*inputs, drives, ones, output_mask)
                    receipt = output["receipt"]
                    receipt.confirm()
                    q, tendon = output["q"].detach().cpu().numpy().copy(), output["tendon"].detach().cpu().numpy().copy()
                    require(q.shape == (4, 42) and tendon.shape == (4, 6) and np.isfinite(q).all() and np.isfinite(tendon).all(), "finite actual replay requested output")
                    raw[arm + ".q"][j], raw[arm + ".tendon"][j] = q, tendon
                    raw[arm + ".valid"][j] = True
                    ids = np.asarray(ENDS[4*j:4*j+4], np.int64)
                    result["comparisons"][key] = {"command": compare_output(q, control["command"][ids], np), "tendon": compare_output(tendon, control["tendon"][ids], np)}
                except Exception as error:
                    receipt = receipt or getattr(error, "receipt", None)
                    if receipt is not None and receipt.confirm_requested == 0:
                        try:
                            receipt.confirm()
                        except Exception as confirm_error:
                            error.add_note("device confirmation: " + str(confirm_error))
                    raise
                finally:
                    if receipt is not None:
                        result["receipts"][key] = core.receipt_dict(receipt)
                monitoring.monitor_check(watch.snapshot(), admission.limits)
            del context, control
        require(saved.expanded == plan["saved_expanded_bytes"] and saved.expanded + packet_expanded == plan["combined_expanded_bytes"],
                "all actual selected nbytes equal complete preallocation plan")
        result["selected_input_actual_nbytes"] = {"packet": packet_expanded, "saved": saved.expanded, "combined": saved.expanded + packet_expanded}
        result["work"] = verify_receipts(result["receipts"], full=True)
        result["input_pre_post_hashes"] = saved.posthash()
        for entry in source_ledger:
            require(core.digest(core.safe(admission.root, entry["path"]), admission.check) == entry["expected_sha256"], "selected original packet posthash")
        result["run_valid"] = True
        result["output_correspondence_passed"] = all(v[k]["passed"] for v in result["comparisons"].values() for k in ("command", "tendon"))
        result["classification"] = "conditional_output_correspondence" if result["output_correspondence_passed"] else "output_correspondence_rejected"
    except Exception as error:
        result["first_error"] = {"stage": stage, "type": type(error).__name__, "message": str(error), "notes": getattr(error, "__notes__", [])}
        try:
            result["work"] = verify_receipts(result["receipts"], full=False)
        except Exception as secondary:
            result["secondary_errors"].append({"stage": "partial-work", "type": type(secondary).__name__, "message": str(secondary)})
    finally:
        try:
            if watch is not None:
                closed = watch.close()
                result["sampler"] = closed
                monitoring.monitor_check(closed, admission.limits, closed=True)
        except Exception as error:
            result["run_valid"] = result["output_correspondence_passed"] = False
            result["classification"] = "technical_invalid"
            close_error = {"stage": "sampler-close", "type": type(error).__name__, "message": str(error)}
            if result["first_error"] is None:
                result["first_error"] = close_error
            else:
                result["secondary_errors"].append(close_error)
        if saved is not None and "input_pre_post_hashes" not in result:
            try:
                result["input_pre_post_hashes"] = saved.posthash()
            except Exception as error:
                result["secondary_errors"].append({"stage": "posthash", "type": type(error).__name__, "message": str(error)})
        publish_result(publisher, raw, source_ledger, result)
    require(result["run_valid"], "technical failure preserved: " + json.dumps(
        {"first_error": result["first_error"], "secondary_errors": result["secondary_errors"]}, ensure_ascii=True, allow_nan=False)[:8192])
    return result


def main(argv=None):
    from tools.banc_m3_live_replay_admission import Admission
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--pre-spawn", action="store_true")
    args = parser.parse_args(argv)
    entry = ENTRY if args.pre_spawn else float(json.loads(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"])["entry_counter_s"])
    admission = Admission(args.request, "pre_spawn" if args.pre_spawn else "pre_array", entry=entry)
    if args.pre_spawn:
        print(json.dumps({"role": "live_replay", "pre_spawn": True}))
        return 0
    result = worker(admission)
    print(json.dumps({"run_valid": result["run_valid"], "output_correspondence_passed": result["output_correspondence_passed"], "candidate_adopted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
