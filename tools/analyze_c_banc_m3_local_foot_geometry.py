"""Saved full-context local FK/J/FD only; CPU imports after Original Admission."""
# ruff: noqa: BLE001 -- preserve partial receipts and guarded failure reports.
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ENTRY = time.perf_counter()
SCHEMA = "flylab.m3.local-foot-geometry"
THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def require(ok, label):
    if not ok:
        raise ValueError(label)


def empty_work():
    return {"model_constructor_requested": 0, "model_constructor_returned": 0, "data_constructor_requested": 0,
            "data_constructor_returned": 0, "forward_requested": 0, "forward_returned": 0,
            "cpu_jacobian_requested": 0, "cpu_jacobian_returned": 0, "physics_steps": 0, "body_constructors": 0,
            "neural_ticks": 0, "Torch_CUDA": 0, "quaternion_correction": 0, "FD_ctrl_clamp": 0}


def qualification(result):
    """Technical validity, FK gates, common correspondence and stage pass differ."""
    fk = bool(result.get("checks")) and all(v["passed"] for v in result["checks"].values())
    common = result.get("baseline_comparison", {}).get("common_values_exact") is True
    result["FK_numerical_passed"], result["common_values_exact"] = fk, common
    result["historical_correspondence_qualified"] = bool(result["run_valid"] and common)
    result["qualified"] = bool(result["run_valid"] and fk and common and result.get("control_mapping_CPU_verified") is True)
    result["geometry_passed"] = result["next_learning_input_qualified"] = result["qualified"]
    result["classification"] = "technical_invalid" if not result["run_valid"] else "local_geometry_qualified" if result["qualified"] else "scientific_rejected"
    return result


def failed(result, stage, error):
    record = {"stage": stage, "type": type(error).__name__, "message": str(error), "notes": getattr(error, "__notes__", [])}
    if result["first_error"] is None:
        result["first_error"] = record
    else:
        result["secondary_errors"].append(record)
    result["run_valid"] = False
    qualification(result)


def numerical_modules():
    import mujoco as mj
    import numpy as np
    return mj, np


def construct_pair(mj, path, work, check):
    check("science")
    work["model_constructor_requested"] += 1
    model = mj.MjModel.from_binary_path(str(path))
    work["model_constructor_returned"] += 1
    check("science")
    work["data_constructor_requested"] += 1
    data = mj.MjData(model)
    work["data_constructor_returned"] += 1
    return model, data


def difference(actual, expected, atol, np):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype == np.float64
            and np.isfinite(actual).all() and np.isfinite(expected).all(), "corresponding finite geometry values")
    delta = actual - expected
    maximum = float(np.max(np.abs(delta)))
    return {"exact": bool(np.array_equal(actual, expected)), "max_abs": maximum, "rms": float(np.sqrt(np.mean(delta * delta))),
            "count": int(delta.size), "atol": atol, "rtol": 0, "passed": maximum <= atol}


def validate_mapping(model, metadata, mj, np):
    from tools.observe_c_banc_m3_local_foot_geometry import (
        CONTROLS,
        LEGS,
        control_identity,
        fk_identity,
        model_dimensions,
    )
    require(model_dimensions(model) == metadata["dimensions"] and fk_identity(model, np) == metadata["fk_identity"], "MJB compiled FK fields/dimensions exactly bound")
    mapping = metadata["control_mapping"]
    act = np.asarray(mapping["actuator_ids"], np.int64)
    require(control_identity(model, act, np) == metadata["control_identity"]
            and metadata["control_end_verified"] is True
            and metadata["control_boundary_verified"] == {str(c): True for c in CONTROLS}, "MJB and all live boundary/end compiled control identity")
    require(np.array_equal(model.actuator_trnid[act, 0], mapping["joint_ids"])
            and np.array_equal(model.actuator_ctrlrange[act], mapping["actuator_ctrlrange_rad"])
            and np.array_equal(model.actuator_ctrllimited[act].astype(bool), mapping["actuator_ctrllimited"])
            and np.array_equal(model.actuator_gear[act], mapping["actuator_gear"])
            and np.array_equal(model.actuator_trntype[act], mapping["actuator_trntype"])
            and int(model.opt.disableflags) == mapping["disableflags"] and int(model.opt.enableflags) == mapping["enableflags"], "recorded clamp/transmission metadata matches compiled mapping")
    ids = {key: np.asarray(metadata[key], np.int64) for key in ("active_joint_ids", "active_qpos_ids", "active_dof_ids", "tip_ids")}
    j, q, v, tips = (ids[k] for k in ("active_joint_ids", "active_qpos_ids", "active_dof_ids", "tip_ids"))
    require(j.shape == q.shape == v.shape == (42,) and len(np.unique(j)) == len(np.unique(q)) == len(np.unique(v)) == 42
            and np.array_equal(j, mapping["joint_ids"]) and np.array_equal(q, mapping["qpos_ids"])
            and np.all((j >= 0) & (j < model.njnt)) and np.all((q >= 0) & (q < model.nq)) and np.all((v >= 0) & (v < model.nv))
            and np.array_equal(model.jnt_qposadr[j], q) and np.array_equal(model.jnt_dofadr[j], v)
            and np.all(model.jnt_type[j] == int(mj.mjtJoint.mjJNT_HINGE)), "active hinge qpos/dof addresses and order")
    require(tips.shape == (6,) and len(np.unique(tips)) == 6 and np.all((tips > 0) & (tips < model.nbody))
            and 0 < metadata["thorax_id"] < model.nbody, "compiled six tip and thorax IDs")
    require([mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, int(i)) for i in j] == metadata["active_joint_names"]
            and [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(i)) for i in tips] == metadata["tip_names"]
            and mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, metadata["thorax_id"]) == metadata["thorax_name"]
            and all(name.endswith('/' + leg + '_tarsus5') for leg, name in zip(LEGS, metadata["tip_names"], strict=True)), "same named physical order")
    return ids


def restore_pose(data, pose, index, np):
    for key in ("qpos", "qvel", "mocap_pos", "mocap_quat"):
        value = pose[key][index]
        require(value.dtype == np.float64 and value.shape == getattr(data, key).shape and np.isfinite(value).all(), "full exact saved pose " + key)
        np.copyto(getattr(data, key), value)
    data.time = float(pose["model_time_absolute_s"][index])


def read_points(data, tip_ids, thorax_id, np):
    origin, axes = data.xpos[thorax_id], data.xmat[thorax_id].reshape(3, 3)
    return data.xpos[tip_ids].copy(), np.vstack((origin, origin + axes.T))


def reconstruct(model, data, pose, metadata, mj, np, check, *, receipts=None):
    from tools.observe_c_banc_m3_local_foot_geometry import (
        ANCHORS,
        CONTROLS,
        control_identity,
        fk_identity,
    )
    # Only construct_pair records actual constructor calls; passed objects prove none.
    receipts = empty_work() if receipts is None else receipts
    ids = validate_mapping(model, metadata, mj, np)
    absolute, normalized = pose["model_time_absolute_s"], pose["physics_time_normalized_s"]
    t0 = metadata["physics_t0_absolute_s"]
    require(np.array_equal(pose["control"], CONTROLS) and np.array_equal(pose["network"], 20 * pose["control"])
            and np.isfinite(t0) and np.isfinite(absolute).all() and np.isfinite(normalized).all()
            and np.all(np.abs(normalized - .005 * pose["control"]) <= 2e-8)
            and np.all(np.abs(absolute - normalized - t0) <= 2e-8)
            and np.array_equal(pose["anchor_control"], ANCHORS)
            and pose["valid"].all() and pose["jacobian_valid"].all() and pose["interval_valid"].all(), "eight exact complete post-forward boundaries/four anchors")
    points, frames = np.zeros((8, 6, 3), np.float64), np.zeros((8, 4, 3), np.float64)
    cpu_j, fd = np.zeros((4, 6, 3, model.nv), np.float64), np.zeros((4, 6, 3, 42), np.float64)
    jp, jr = np.zeros((3, model.nv), np.float64), np.zeros((3, model.nv), np.float64)

    def forward():
        check("science")
        require(receipts["forward_requested"] < 344, "fixed CPU344 forward cap")
        receipts["forward_requested"] += 1
        mj.mj_forward(model, data)
        receipts["forward_returned"] += 1

    try:
        for i, c in enumerate(CONTROLS):
            restore_pose(data, pose, i, np)
            forward()
            points[i], frames[i] = read_points(data, ids["tip_ids"], metadata["thorax_id"], np)
            if c in ANCHORS:
                j = ANCHORS.index(c)
                for leg, tip in enumerate(ids["tip_ids"]):
                    check("science")
                    receipts["cpu_jacobian_requested"] += 1
                    mj.mj_jacBody(model, data, jp, jr, int(tip))
                    receipts["cpu_jacobian_returned"] += 1
                    cpu_j[j, leg] = jp
        for j, c in enumerate(ANCHORS):
            i = CONTROLS.index(c)
            for active, qpos in enumerate(ids["active_qpos_ids"]):
                ends = []
                for sign in (-1., 1.):
                    restore_pose(data, pose, i, np)
                    data.qpos[int(qpos)] += sign * 1e-6
                    forward()
                    ends.append(data.xpos[ids["tip_ids"]].copy())
                fd[j, :, :, active] = (ends[1] - ends[0]) / 2e-6
        require(receipts["forward_requested"] == receipts["forward_returned"] == 344
                and receipts["cpu_jacobian_requested"] == receipts["cpu_jacobian_returned"] == 24
                and fk_identity(model, np) == metadata["fk_identity"]
                and control_identity(model, metadata["control_mapping"]["actuator_ids"], np) == metadata["control_identity"], "bounded CPU work and unchanged FK/control model")
        checks = {"tip_positions": difference(points, pose["tip_world_mm"], 1e-9, np),
                  "thorax_frame_points": difference(frames, pose["thorax_points_mm"], 1e-9, np),
                  "full_nv_native_CPU_J": difference(cpu_j, pose["jacobian_full_nv"], 1e-8, np),
                  "active_CPU_FD_J": difference(cpu_j[..., ids["active_dof_ids"]], fd, 1e-8, np)}
        return checks, receipts, {"CPU_points_mm": points, "CPU_thorax_points_mm": frames, "CPU_J_full_nv": cpu_j, "FD_J_active": fd}
    except Exception as error:
        error.geometry_receipts = receipts
        error.geometry_raw = {"CPU_points_mm": points, "CPU_thorax_points_mm": frames, "CPU_J_full_nv": cpu_j, "FD_J_active": fd}
        raise


def local_table(pose, metadata, np, *, control_identity_verified=False):
    from tools.observe_c_banc_m3_local_foot_geometry import ANCHORS, CONTROLS, LEGS
    from tools.observe_c_banc_m3_startup_tracking import derived_effective
    mapping = metadata["control_mapping"]
    require(mapping["activation_tau_s"] == 0 and mapping["interpolation"] == "held raw ctrl", "observed neutral tau0/held path")
    qpos, dofs = np.asarray(metadata["active_qpos_ids"], np.int64), np.asarray(metadata["active_dof_ids"], np.int64)
    rows = []
    for j, c in enumerate(ANCHORS):
        a, b = CONTROLS.index(c), CONTROLS.index(c + 1)
        raw = pose["interval_raw_ctrl_rad"][j]
        require(np.array_equal(raw, np.broadcast_to(raw[0], raw.shape)), "actual all50 held interval controls, not endpoint substitution")
        effective = derived_effective(raw[0], mapping, np)
        unit_verified = control_identity_verified and mapping["ctrl_to_q_mapping"]["status"] == "VERIFIED_UNIT_GEAR_JOINT_HINGE"
        # Explicit take keeps (6,3,42) despite NumPy advanced-index axis movement.
        active_j = np.take(pose["jacobian_full_nv"][j], dofs, axis=-1)
        actual = pose["tip_world_mm"][b] - pose["tip_world_mm"][a]
        active_delta = active_j @ (pose["qpos"][b, qpos] - pose["qpos"][a, qpos])
        command_delta = active_j @ (effective - pose["qpos"][a, qpos]) if effective is not None and unit_verified else None
        axes = pose["thorax_points_mm"][a, 1:] - pose["thorax_points_mm"][a, 0]
        own = []
        for k in (a, b):
            frame = pose["thorax_points_mm"][k, 1:] - pose["thorax_points_mm"][k, 0]
            own.append((pose["tip_world_mm"][k] - pose["thorax_points_mm"][k, 0]) @ frame.T)
        for leg, name in enumerate(LEGS):
            pre, post = bool(pose["normal_load_bw"][a, leg] >= .02), bool(pose["normal_load_bw"][b, leg] >= .02)
            def projection(vector, axes=axes):
                return {"start_thorax_AP_mm": float(vector @ axes[0]), "world_vertical_mm": float(vector[2]), "start_body_dorsal_mm": float(vector @ axes[2])}
            rows.append({"anchor_control": c, "leg": name, "normal_pre_bw": float(pose["normal_load_bw"][a, leg]),
                         "normal_post_bw": float(pose["normal_load_bw"][b, leg]), "loaded_pre": pre, "loaded_post": post,
                         "recorded_load_pre": bool(pose["recorded_load_bearing"][a, leg]), "recorded_load_post": bool(pose["recorded_load_bearing"][b, leg]),
                         "category": "loaded_loaded" if pre and post else "unloaded_unloaded" if not pre and not post else "transition",
                         "J_command": projection(command_delta[leg]) if command_delta is not None else None,
                         "command_status": "VERIFIED_ACTUAL_HELD_CLAMP" if command_delta is not None else "UNVERIFIED",
                         "J_active_delta": projection(active_delta[leg]), "observed_world_delta": projection(actual[leg]),
                         "unattributed_kinematic_residual": projection(actual[leg] - active_delta[leg]),
                         "body_relative_own_frame_AP_delta_mm": float(own[1][leg, 0] - own[0][leg, 0])})
    categories = {category: {"count": sum(row["category"] == category for row in rows),
                              "status": "OBSERVED" if any(row["category"] == category for row in rows) else "UNOBSERVED"}
                  for category in ("loaded_loaded", "unloaded_unloaded", "transition")}
    return rows, categories


def markdown(result):
    lines = ["# 네 neutral 전이의 국소 발 기하", "", f"run_valid={result['run_valid']}, FK_numerical_passed={result['FK_numerical_passed']}, common_values_exact={result['common_values_exact']}, qualified={result['qualified']}.",
             "AP=start-thorax anterior, vertical=world +Z; 별도 dorsal/own-frame AP. Tip은 tarsus5 body origin입니다.", ""]
    for row in result.get("local_table", []):
        lines.append(f"- c={row['anchor_control']} {row['leg']} {row['category']}: observed AP={row['observed_world_delta']['start_thorax_AP_mm']}, active AP={row['J_active_delta']['start_thorax_AP_mm']}; mixed residual은 미귀속.")
    lines += ["", "4개 국소 전이이며200control/full stride·힘·servo/contact 인과·candidate 기하로 확대하지 않습니다.",
              "F1–F8 INCOMPLETE, 후보 채택false, biology UNPROVEN, M2/UI/provenance 보존."]
    if result.get("first_error"):
        lines.append("최초 오류: " + json.dumps(result["first_error"], ensure_ascii=False))
    if result.get("secondary_errors"):
        lines.append("후속 오류: " + json.dumps(result["secondary_errors"], ensure_ascii=False))
    return "\n".join(lines) + "\n"


def initial_result():
    return qualification({"schema": SCHEMA + ".analysis.v1", "run_valid": False, "first_error": None,
              "secondary_errors": [], "candidate_adopted": False, "F1_F8": "INCOMPLETE", "biology": "UNPROVEN", "M2_UI": "PRESERVED",
              "work": empty_work(), "control_mapping_CPU_verified": False, "publication_valid": True,
              "hypothesis_status": "unidentified", "local_scope": "four unchanged neutral one-control transitions only"})


def publisher_for(admission):
    from tools.observe_c_banc_m3_startup_tracking import bounded_publisher
    return bounded_publisher(admission)


def publish_result(publisher, raw, source_ledger, result):
    """Existing bounded Publisher/fences apply to every independent/fallback write."""
    def rejected(stage, error):
        result["publication_valid"] = False
        failed(result, stage, error)

    operations = (("publish-partial-work", lambda: publisher.json("geometry-work.json", result["work"])),
                  ("publish-selected-ledger", lambda: publisher.json("geometry-selected-ledger.json", source_ledger)))
    if raw:
        operations = (("publish-raw", lambda: publisher.arrays("geometry-analysis.npz", raw)), *operations)
    for stage, operation in operations:
        try:
            operation()
        except Exception as error:
            rejected(stage, error)
    result["durable_result_bytes_before_report"] = publisher.total
    try:
        publisher.write("geometry-analysis.md", markdown(result).encode("utf8"))
    except Exception as error:
        rejected("publish-report-md", error)
    try:
        publisher.json("geometry-analysis.json", result)
    except Exception as error:
        rejected("publish-report-json", error)
        # Fresh failure names never overwrite/retry a potentially partial artifact.
        for stage, operation in (("publish-failure-json", lambda: publisher.json("geometry-publication-failure.json", result)),
                                 ("publish-failure-md", lambda: publisher.write("geometry-publication-failure.md", markdown(result).encode("utf8")))):
            try:
                operation()
            except Exception as secondary:
                rejected(stage, secondary)
    return result


def failure_text(result):
    return json.dumps({"first_error": result["first_error"], "work": result["work"], "secondary_errors": result["secondary_errors"]},
                      ensure_ascii=True, allow_nan=False)[:8192]


def worker(admission):
    result, raw, saved, publisher = initial_result(), {}, None, None
    stage = "publication-setup"
    try:
        require(admission.role == "geometry_analyze" and admission.stage == "pre_array", "saved Original geometry CPU only")
        admission.fence("pre_array")
        publisher = publisher_for(admission)
        stage = "CPU-thread-pre-import"
        result["pre_import_thread_environment"] = {key: os.environ.get(key) for key in THREAD_ENV}
        require(all(os.environ.get(k) == "1" for k in THREAD_ENV) and "torch" not in sys.modules, "CPU thread1 before numerical import/no Torch")
        stage = "numerical-import"
        mj, np = numerical_modules()
        stage = "saved-input-preparation"
        from tools import analyze_c_banc_m3_startup_tracking as old
        from tools.observe_c_banc_m3_local_foot_geometry import pose_fields
        saved = old.SavedInputs(admission, np)
        stage = "saved-input"
        bundle = admission.request["artifacts"]
        manifest, report, metadata = (saved.json(bundle[k]) for k in ("manifest", "body_report", "geometry_metadata"))
        require(manifest["role"] == "geometry_capture" and manifest["run_valid"] and manifest["capture_complete"]
                and report["geometry"]["complete"] and all(manifest["artifacts"][k] == v for k, v in bundle.items() if k != "manifest"), "exact complete recorded geometry bundle")
        pose = saved.arrays(bundle["geometry"], pose_fields(metadata["dimensions"]))
        path, stamp = saved.digest(bundle["model"])
        saved.ledger.append((bundle["model"], stamp))  # immediately after its actual digest, before allocation/constructors
        require(bundle["model"]["bytes"] * 16 + saved.expanded <= admission.limits["host_bytes"], "bounded model/data load before allocation")
        stage = "CPU-model-one-pair"
        model, data = construct_pair(mj, path, result["work"], admission.check)
        stage = "CPU-version-mapping"
        require(mj.__version__ == metadata["mujoco_version"], "same compiled MuJoCo version")
        stage = "CPU-FK-J-FD"
        result["checks"], result["work"], raw = reconstruct(model, data, pose, metadata, mj, np, admission.check, receipts=result["work"])
        result["control_mapping_CPU_verified"] = True
        result["local_table"], result["contact_categories"] = local_table(pose, metadata, np, control_identity_verified=True)
        stage = "common-neutral-correspondence"
        baseline = admission.roster["neutral_capture"]
        comparisons = {}
        for label, fields in (("control", old.CONTROL_FIELDS), ("native", old.NATIVE_FIELDS)):
            current, historic = saved.arrays(bundle[label], fields), saved.arrays(baseline[label], fields)
            comparisons[label] = {key: old.compare_native_slip(current, historic, np) if key == "slip_mm_s" else old.compare_values(current[key], historic[key], np) for key in fields}
        comparisons["trace"] = old.compare_trace(saved.json(bundle["trace"]), saved.json(baseline["trace"]), np)
        common_exact = all(value.get("exact") is True for group in (comparisons["control"], comparisons["native"]) for value in group.values()) and comparisons["trace"].get("exact") is True
        result["baseline_comparison"] = {"common_values_exact": common_exact, "fields": comparisons,
                                          "status": "EXACT_COMMON_NEUTRAL" if common_exact else "COMMON_DRIFT_NOT_HISTORICAL_CORRESPONDENCE_PASS",
                                          "full_pose_model_first_observed": True}
        result["input_pre_post_hashes"] = saved.posthash()
        require("torch" not in sys.modules and result["work"]["forward_returned"] == 344, "CPU-only bounded returned work")
        result["run_valid"] = True
        qualification(result)
    except Exception as error:
        result["work"] = getattr(error, "geometry_receipts", result["work"])
        raw = getattr(error, "geometry_raw", raw)
        failed(result, stage, error)
    finally:
        if saved is not None:
            try:
                result["input_pre_post_hashes"] = saved.posthash()
            except Exception as error:
                failed(result, "posthash", error)
        source_ledger = [{"descriptor": spec, "mtime_ns": stamp} for spec, stamp in saved.ledger] if saved is not None else []
        if publisher is not None:
            publish_result(publisher, raw, source_ledger, result)
        else:
            result["publication_valid"] = False
        if not result["run_valid"]:
            print("geometry technical failure: " + failure_text(result), file=sys.stderr)
    require(result["run_valid"], "technical-invalid geometry: " + failure_text(result))
    return result


def main(argv=None):
    from tools.banc_m3_local_foot_geometry_admission import Admission
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    entry = float(json.loads(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"])["entry_counter_s"])
    result = worker(Admission(args.request, entry=entry))
    print(json.dumps({"run_valid": result["run_valid"], "FK_numerical_passed": result["FK_numerical_passed"],
                      "common_values_exact": result["common_values_exact"], "qualified": result["qualified"], "candidate_adopted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
