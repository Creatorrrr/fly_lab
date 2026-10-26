"""E229 body-only contact-point Jacobian diagnostic at frozen event 25."""

# Keep the first integration/observer failure across independent raw saves.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.storage import StateStore
from tools import analyze_c_banc_m3_native_contact_decomposition as e157
from tools import analyze_c_banc_observation as observation
from tools import observe_c_banc_m3_effector_force_replay as replay
from tools import observe_c_banc_m3_ground_reaction as ground
from tools import probe_c_banc_m3_local_command_hold as e228
from tools import verify_c_banc_m3_e113_body as e113
from tools import verify_c_banc_m3_neural_current_body as prior
from tools.verify_c_banc_observation import exact

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
OUT = BASE / "e229-effector-contact-decomposition"
FREEZE_SHA = "52ae29a8db9cbe4a26b668d36b60efe383cf80df17a974a5b5596decc67cd71e"
FREEZE_SCHEMA = "flylab.m3.e229.effector-contact-decomposition.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e229.effector-contact-decomposition.review.v1"
OWNED = ("tools/observe_c_banc_m3_effector_contact_decomposition.py",
         "tests/test_c_banc_effector_contact_decomposition.py")
START, END = 1607, 1647
STEPS = 50
DT = .0001
PHASES = {"lead": [1608, 1620], "hold_window": [1620, 1636],
          "recovery": [1636, 1648], "during_break": [1623, 1631]}
CONTACT_SHAPES = {"position_world": (3,), "frame": (3, 3),
                  "wrench": (6,), "force_world_bw": (3,)}
CONTACT_INT = {"native_index", "contact_id", "geom1", "geom2", "fly_geom",
               "floor_geom", "leg", "segment", "sign"}
CONTACT_FIELDS = (*CONTACT_INT, *CONTACT_SHAPES, "normal_load_bw",
                  "positive_leg")
JAC_SHAPES = {"qvel": None, "jfly": None, "jfloor": None}
JAC_FIELDS = ("contact_row", "native_index", "qvel", "jfly", "jfloor",
              "jfly_requested", "jfly_returned", "jfloor_requested",
              "jfloor_returned", "decomposition_complete")
DECOMP_FIELDS = ("contact_row", "native_index", "leg", "normal_load_bw",
    "tangent_velocity", "force_world_bw", "tangent_force_bw", "ap_dof",
    "ap_total", "ap_root", "ap_own7", "ap_residual", "power_dof",
    "power_total", "power_root", "power_own7", "power_residual",
    "slip_speed", "ap_closure_error", "ap_direct_error", "power_closure_error",
    "power_direct_error", "projection_error")


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if (freeze_sha != FREEZE_SHA or replay.sha256(freeze_path) != FREEZE_SHA
            or replay.sha256(review_path) != review_sha):
        raise ValueError("Exact E229 freeze and parent review hashes required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or replay.sha256(replay.safe(freeze["design"])) != freeze.get(
                "design_sha256")
            or freeze.get("event") != {"event_id": 25,
                "start_control": START, "end_control": END,
                "entry_control": 1623, "exit_control": 1631}
            or freeze.get("phases") != PHASES
            or freeze.get("native") != {"control_dt_s": .005,
                "physics_dt_s": DT, "steps_per_control": STEPS,
                "controls": 40, "samples": 2000,
                "new_neural_ticks": 0, "support_threshold_bw": .02,
                "contact_rows_max": 16000, "jacobian_calls_max": 32000,
                "decomposition_atol": 1e-10, "force_sum_atol": 1e-10}
            or freeze.get("force_fields") != ["time_s", "available", "error",
                *replay.FORCE_SHAPES]
            or freeze.get("resource_limits") != {"cpu_threads": 1,
                "gpu_count": 0, "original_attempts": 1, "retries": 0,
                "body_constructors": 1, "shadow_constructors": 1,
                "fit_calls": 0, "solver_calls": 0,
                "internal_s": 30., "finalization_reserve_s": 5.,
                "external_s": 35., "supervisor_s": 40.,
                "resident_bytes": 6442450944,
                "output_bytes": 134217728,
                "minimum_free_bytes": 2147483648}
            or any(freeze.get(key) is not False for key in (
                "candidate_adopted", "F2_complete", "F6_exposed",
                "whole_goal_complete", "active_UI_changed"))):
        raise ValueError("E229 frozen diagnostic/resource contract changed")
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Parent-approved E229 source review required")
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze.get(label)
        if not isinstance(mapping, dict):
            raise TypeError("E229 frozen hash inventory required: " + label)
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or replay.sha256(replay.safe(name)) != expected):
                raise ValueError("Frozen E229 source/input changed: " + name)
    for mapping in (review["source_sha256"],
                    review.get("additional_reused_source_sha256", {})):
        if not isinstance(mapping, dict):
            raise TypeError("E229 reviewed hash mapping required")
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or replay.sha256(replay.safe(name)) != expected):
                raise ValueError("Reviewed E229 source changed: " + name)
    for package, expected in freeze["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("Frozen E229 package version changed: " + package)
    if output != OUT.resolve() or output.exists() or not output.parent.is_dir():
        raise FileExistsError("Exact new E229 output directory required")
    if shutil.disk_usage(output.parent).free < freeze["resource_limits"][
            "minimum_free_bytes"]:
        raise OSError("E229 output disk reserve unavailable")
    return freeze, review


def load_reference(freeze):
    paths = {key: replay.safe(name) for key, name in freeze["paths"].items()}
    trace = json.loads(paths["reference_trace"].read_text(encoding="utf-8"))["samples"]
    sham_dir = paths["e228_sham"]
    sham_rows = [json.loads(line) for line in (sham_dir / "control.jsonl").open(
        encoding="utf-8")]
    archives = {}
    for label, path in (("source_native", paths["reference_native"]),
                        ("source_force", paths["reference_force"]),
                        ("sham_native", sham_dir / "native.npz"),
                        ("sham_force", sham_dir / "force.npz")):
        with np.load(path, allow_pickle=False) as archive:
            archives[label] = {key: archive[key].copy() for key in archive.files}
    if (len(trace) != 1664 or len(sham_rows) != 41
            or [row["control_tick"] for row in sham_rows] != list(
                range(START, END + 1))
            or any(set(archives[label]) != set(e228.NATIVE_FIELDS)
                   for label in ("source_native", "sham_native"))
            or any(set(archives[label]) != set(freeze["force_fields"])
                   for label in ("source_force", "sham_force"))
            or any(len(value) != 2000 for value in archives["sham_native"].values())
            or any(len(value) != 2000 for value in archives["sham_force"].values())
            or any(len(value) != 53150 for value in archives["source_native"].values())
            or any(len(value) != 53150 for value in archives["source_force"].values())):
        raise ValueError("E229 exact E225/E227/E228 sham raw required")
    first = e228.source_slice(START + 1).start
    last = e228.source_slice(END).stop
    for label, source in (("sham_native", "source_native"),
                          ("sham_force", "source_force")):
        for key, values in archives[label].items():
            expected = archives[source][key][first:last]
            if (values.shape != expected.shape or values.dtype != expected.dtype
                    or values.tobytes() != expected.tobytes()):
                raise ValueError("E229 frozen sham/native/force mismatch: " + key)
    for offset, row in enumerate(sham_rows):
        replay.compare_physical_row(row, trace[START + offset])
    if any(any(trace[tick]["adhesion"]) or len(trace[tick]["tendon_inputs"]) != 8
           or np.asarray(trace[tick]["joint_targets_rad"]).shape != (42,)
           for tick in range(START, END + 1)):
        raise ValueError("E229 original all-off adhesion and 42-joint/eight-tendon path required")
    return paths, trace, sham_rows, archives


def layout_metadata(body):
    layout = e157.validate_layout(body)
    leg_map, leg_count = body._force_maps["legs"]
    foot_map, foot_count = body._force_maps["feet"]
    leg_map = np.asarray(leg_map, dtype=np.int64)
    foot_map = np.asarray(foot_map, dtype=np.int64)
    ground = np.asarray(body._ground_mask, dtype=bool)
    if (leg_count != 48 or foot_count != 30
            or not (leg_map.shape == foot_map.shape == ground.shape)
            or leg_map.shape != (int(body.m.ngeom),)
            or not np.array_equal(np.unique(leg_map[leg_map >= 0]),
                                  np.arange(48))
            or sorted(body.support_geom_ids) != sorted(set(body.support_geom_ids))):
        raise ValueError("Exact 48-leg/30-foot/support geometry mapping required")
    layout.update(leg_geom_map=leg_map.tolist(), foot_geom_map=foot_map.tolist(),
        support_geom_ids=sorted(int(value) for value in body.support_geom_ids),
        ground_mask=ground.tolist())
    return layout


def classify_all_contact(*, geom1, geom2, excluded, frame, wrench, position,
                         geom_body_ids, fly_body_ids, support_geom_ids,
                         ground_mask, leg_geom_map, weight0):
    """Return one active fly-floor row, including nonleg/nonpositive contacts."""
    if excluded:
        return None
    a, b = int(geom1), int(geom2)
    mapping = np.asarray(leg_geom_map, np.int64)
    bodies = np.asarray(geom_body_ids, np.int64)
    mask = np.asarray(ground_mask, bool)
    if min(a, b) < 0 or max(a, b) >= len(mapping):
        raise ValueError("E229 contact geometry index out of range")
    support = set(map(int, support_geom_ids))
    fly = set(map(int, fly_body_ids))
    floor_a, floor_b = a in support and mask[a], b in support and mask[b]
    fly_a, fly_b = int(bodies[a]) in fly, int(bodies[b]) in fly
    if floor_a and fly_b:
        sign, fly_geom, floor_geom = 1, b, a
    elif floor_b and fly_a:
        sign, fly_geom, floor_geom = -1, a, b
    else:
        return None
    axes = np.asarray(frame, np.float64).reshape(3, 3)
    six = np.asarray(wrench, np.float64).reshape(6)
    point = np.asarray(position, np.float64).reshape(3)
    if (not np.isfinite(axes).all() or not np.isfinite(six).all()
            or not np.isfinite(point).all() or not np.isfinite(weight0)
            or weight0 <= 0):
        raise ValueError("Finite raw contact geometry, wrench, and BW required")
    segment = int(mapping[fly_geom])
    leg = segment // 8 if segment >= 0 else -1
    if leg >= 6:
        raise ValueError("E229 leg segment outside six 8-link groups")
    return {"geom1": a, "geom2": b, "fly_geom": fly_geom,
        "floor_geom": floor_geom, "leg": leg, "segment": segment,
        "sign": sign, "position_world": point.copy(), "frame": axes.copy(),
        "wrench": six.copy(), "normal_load_bw": float(six[0] / weight0),
        "force_world_bw": sign * axes.T @ six[:3] / weight0,
        "positive_leg": bool(leg >= 0 and six[0] > 0)}


def new_registry(nq, nv, nu):
    return {"nq": nq, "nv": nv, "nu": nu,
        "native_state": {key: [] for key in ("native_index", "time_s",
            "control_tick", "qpos", "qvel", "ctrl", "state_captured",
            "shadow_forward_returned", "observation_returned")},
        "contact_offsets": [0],
        "contacts": {key: [] for key in CONTACT_FIELDS},
        "jacobian_attempts": {key: [] for key in JAC_FIELDS},
        "decomposition": {key: [] for key in DECOMP_FIELDS}}


def append(section, row):
    if set(row) != set(section):
        raise ValueError("E229 raw registry row schema differs")
    for key, value in row.items():
        section[key].append(np.array(value, copy=True) if isinstance(
            value, np.ndarray) else value)
    return len(next(iter(section.values()))) - 1


def observe_contact_step(body, shadow, registry, layout, inventory, heading,
                         ledger, limits):
    """Record all raw contacts, then request two Jacobians for positive leg rows."""
    native_index = len(registry["native_state"]["native_index"]) - 1
    if native_index < 0:
        raise ValueError("E229 native state must precede contact observation")
    map_leg = layout["leg_geom_map"]
    bodies = layout["geom_body_ids"]
    for contact_id, contact in enumerate(shadow.d.contact[:shadow.d.ncon]):
        wrench = np.empty(6, np.float64)
        if contact.exclude:
            ledger["excluded_contact_count"] += 1
            continue
        a, b = int(contact.geom1), int(contact.geom2)
        mapping = np.asarray(map_leg)
        floor_a = a in body.support_geom_ids and body._ground_mask[a]
        floor_b = b in body.support_geom_ids and body._ground_mask[b]
        fly_ids = set(inventory["body_ids"])
        if not ((floor_a and int(bodies[b]) in fly_ids)
                or (floor_b and int(bodies[a]) in fly_ids)):
            continue
        ledger["contact_force_calls_requested"] += 1
        body.mj.mj_contactForce(body.m, shadow.d, contact_id, wrench)
        ledger["contact_force_calls_returned"] += 1
        row = classify_all_contact(geom1=a, geom2=b, excluded=False,
            frame=contact.frame, wrench=wrench, position=contact.pos,
            geom_body_ids=bodies, fly_body_ids=inventory["body_ids"],
            support_geom_ids=layout["support_geom_ids"],
            ground_mask=layout["ground_mask"], leg_geom_map=mapping,
            weight0=inventory["weight0"])
        if row is None:
            raise ValueError("E229 active fly-floor contact classification lost")
        if ledger["contact_rows"] >= limits["contact_rows_max"]:
            raise MemoryError("E229 contact row cap exceeded")
        row = {"native_index": native_index, "contact_id": contact_id, **row}
        contact_row = append(registry["contacts"], row)
        ledger["contact_rows"] += 1
        if not row["positive_leg"]:
            if row["leg"] < 0:
                ledger["nonleg_contact_count"] += 1
            else:
                ledger["nonpositive_leg_contact_count"] += 1
            continue
        nv = registry["nv"]
        attempt = {"contact_row": contact_row, "native_index": native_index,
            "qvel": np.asarray(shadow.d.qvel).copy(),
            "jfly": np.full((3, nv), np.nan),
            "jfloor": np.full((3, nv), np.nan),
            "jfly_requested": False, "jfly_returned": False,
            "jfloor_requested": False, "jfloor_returned": False,
            "decomposition_complete": False}
        attempt_id = append(registry["jacobian_attempts"], attempt)
        point = row["position_world"]
        jfly = np.zeros((3, nv), np.float64)
        jfloor = np.zeros((3, nv), np.float64)
        if ledger["jacobian_calls_requested"] >= limits["jacobian_calls_max"]:
            raise MemoryError("E229 Jacobian call cap exceeded")
        ledger["jacobian_calls_requested"] += 1
        registry["jacobian_attempts"]["jfly_requested"][attempt_id] = True
        body.mj.mj_jac(body.m, shadow.d, jfly, None, point,
                       int(bodies[row["fly_geom"]]))
        ledger["jacobian_calls_returned"] += 1
        registry["jacobian_attempts"]["jfly"][attempt_id] = jfly.copy()
        registry["jacobian_attempts"]["jfly_returned"][attempt_id] = True
        if ledger["jacobian_calls_requested"] >= limits["jacobian_calls_max"]:
            raise MemoryError("E229 Jacobian call cap exceeded")
        ledger["jacobian_calls_requested"] += 1
        registry["jacobian_attempts"]["jfloor_requested"][attempt_id] = True
        body.mj.mj_jac(body.m, shadow.d, jfloor, None, point,
                       int(bodies[row["floor_geom"]]))
        ledger["jacobian_calls_returned"] += 1
        registry["jacobian_attempts"]["jfloor"][attempt_id] = jfloor.copy()
        registry["jacobian_attempts"]["jfloor_returned"][attempt_id] = True
        decomposed = e157.decompose_contact(row["frame"], attempt["qvel"],
            jfly, jfloor, heading, layout["root_dof_addresses"],
            layout["leg_dof_addresses"][row["leg"]], row["wrench"],
            row["sign"], inventory["weight0"])
        append(registry["decomposition"], {"contact_row": contact_row,
            "native_index": native_index, "leg": row["leg"],
            **{key: decomposed[key] for key in DECOMP_FIELDS
               if key not in {"contact_row", "native_index", "leg"}}})
        registry["jacobian_attempts"]["decomposition_complete"][attempt_id] = True
        ledger["decomposition_rows"] += 1


def arrays(registry):
    nq, nv, nu = (registry[key] for key in ("nq", "nv", "nu"))
    state_shapes = {"qpos": (nq,), "qvel": (nv,), "ctrl": (nu,)}
    jac_shapes = {"qvel": (nv,), "jfly": (3, nv), "jfloor": (3, nv)}
    decomp_shapes = {"tangent_velocity": (3,), "force_world_bw": (3,),
        "tangent_force_bw": (3,), "ap_dof": (nv,), "power_dof": (nv,),
        "ap_own7": (7,), "power_own7": (7,)}

    def pack(section, shapes, int_keys=()):
        packed = {}
        for key, values in section.items():
            dtype = (np.int64 if key in int_keys else bool if key in (
                "positive_leg", "state_captured", "shadow_forward_returned",
                "observation_returned", "jfly_requested", "jfly_returned",
                "jfloor_requested", "jfloor_returned", "decomposition_complete")
                else np.float64)
            packed[key] = np.asarray(values, dtype=dtype).reshape(
                (len(values), *shapes.get(key, ())))
        return packed

    state = pack(registry["native_state"], state_shapes,
                 ("native_index", "control_tick"))
    contacts = pack(registry["contacts"], CONTACT_SHAPES, CONTACT_INT)
    contacts["offsets"] = np.asarray(registry["contact_offsets"], np.int64)
    attempts = pack(registry["jacobian_attempts"], jac_shapes,
                    ("contact_row", "native_index"))
    decomposition = pack(registry["decomposition"], decomp_shapes,
        ("contact_row", "native_index", "leg"))
    return state, contacts, attempts, decomposition


def aggregate_phase(contacts, decomposition, phases, threshold, heading):
    """Aggregate each leg's supported and braking native samples once per phase."""
    native_count = 2000
    leg_load = np.zeros((native_count, 6), np.float64)
    leg_force = np.zeros((native_count, 6), np.float64)
    for row in range(len(contacts["native_index"])):
        leg = int(contacts["leg"][row])
        if leg < 0:
            continue
        native = int(contacts["native_index"][row])
        if contacts["normal_load_bw"][row] > 0:
            leg_load[native, leg] += contacts["normal_load_bw"][row]
        leg_force[native, leg] += float(contacts["force_world_bw"][row] @ heading)
    report = {}
    for phase, (first_tick, stop_tick) in phases.items():
        start = (first_tick - START - 1) * STEPS
        stop = (stop_tick - START - 1) * STEPS
        report[phase] = {}
        for leg in range(6):
            support = leg_load[start:stop, leg] >= threshold
            braking = support & (leg_force[start:stop, leg] < 0)
            groups = {}
            for name, native_mask in (("support", support), ("braking", braking)):
                selected_native = np.zeros(native_count, bool)
                selected_native[start:stop] = native_mask
                selected = (selected_native[decomposition["native_index"]]
                            & (decomposition["leg"] == leg))
                load = decomposition["normal_load_bw"][selected]
                denominator = float(load.sum())
                def weighted(key, chosen=selected, weights=load, total=denominator):
                    if total <= 0:
                        return None
                    value = np.asarray(decomposition[key][chosen], np.float64)
                    return (np.sum(value * weights.reshape((-1,) + (1,) *
                        (value.ndim - 1)), axis=0) / total).tolist()
                def integrated(key, chosen=selected):
                    value = np.asarray(decomposition[key][chosen], np.float64)
                    total = np.sum(value, axis=0) * DT
                    return total.tolist() if np.ndim(total) else float(total)
                values = leg_force[start:stop, leg][native_mask]
                groups[name] = {"unique_native_samples": int(native_mask.sum()),
                    "supported_time_s": float(native_mask.sum() * DT),
                    "positive_contact_count": int(selected.sum()),
                    "normal_load_sum_BW": denominator,
                    "AP_load_weighted_mm_s": {key: weighted(key) for key in (
                        "ap_total", "ap_root", "ap_own7", "ap_residual",
                        "slip_speed")},
                    "power_integral_BW_mm": {key: integrated(key) for key in (
                        "power_total", "power_root", "power_own7",
                        "power_residual")},
                    "signed_forward_force_BW_s": float(values.sum() * DT)}
            report[phase][e157.LEG_NAMES[leg]] = groups
    return report, leg_load, leg_force


def validate_complete(state, contacts, attempts, decomposition, force,
                      inventory, heading, limits):
    if (len(state["native_index"]) != 2000
            or len(contacts["offsets"]) != 2001
            or not np.all(attempts["jfly_returned"])
            or not np.all(attempts["jfloor_returned"])
            or not np.all(attempts["decomposition_complete"])
            or len(decomposition["contact_row"]) != len(attempts["contact_row"])
            or not np.array_equal(np.flatnonzero(contacts["positive_leg"]),
                                  attempts["contact_row"])
            or not np.array_equal(decomposition["contact_row"],
                                  attempts["contact_row"])):
        raise ValueError("E229 all native/contact Jacobian observations required")
    if (not np.array_equal(attempts["qvel"],
            state["qvel"][attempts["native_index"]])
            or not np.array_equal(decomposition["normal_load_bw"],
                                  contacts["normal_load_bw"][
                                      decomposition["contact_row"]])):
        raise ValueError("E229 pointwise qvel/load provenance differs")
    max_error = max((float(np.max(decomposition[key])) if len(
        decomposition[key]) else 0.) for key in (
        "ap_closure_error", "ap_direct_error", "power_closure_error",
        "power_direct_error", "projection_error"))
    if not np.isfinite(max_error) or max_error > limits["decomposition_atol"]:
        raise ValueError("E229 contact AP/power/projection closure differs")
    all_leg = np.zeros((2000, 6, 3), np.float64)
    full = np.zeros((2000, 3), np.float64)
    for row in range(len(contacts["native_index"])):
        native = int(contacts["native_index"][row])
        leg = int(contacts["leg"][row])
        value = contacts["force_world_bw"][row]
        full[native] += value
        if leg >= 0:
            all_leg[native, leg] += value
    leg_error = float(np.max(np.abs(all_leg - force["all_leg_force_bw"])))
    fly_error = float(np.max(np.abs(full - force["full_fly_floor_force_bw"])))
    residual_error = float(np.max(np.abs((full - all_leg.sum(axis=1)) -
        force["nonleg_floor_force_residual_bw"])))
    if max(leg_error, fly_error, residual_error) > limits["force_sum_atol"]:
        raise ValueError("E229 observed contact force does not close E227 shadow force")
    return {"decomposition_max_error": max_error,
        "all_leg_force_max_abs_BW": leg_error,
        "full_fly_force_max_abs_BW": fly_error,
        "nonleg_residual_force_max_abs_BW": residual_error,
        "positive_leg_contacts": len(attempts["contact_row"]),
        "all_active_fly_floor_contacts": len(contacts["native_index"]),
        "fixed_heading_native_xy": heading.tolist(),
        "weight0": inventory["weight0"]}


def save_npz(path, value):
    e157.atomic_npz(path, **value)


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze, review = gate(Path(freeze_path), freeze_sha,
                          Path(review_path), review_sha, out)
    out.mkdir(parents=False, exist_ok=False)
    limits = freeze["resource_limits"]
    science = freeze["native"]
    ledger = {"schema": "flylab.m3.e229.effector-contact-decomposition.ledger.v1",
        "body_constructors_requested": 0, "body_constructors_returned": 0,
        "shadow_constructors_requested": 0, "shadow_constructors_returned": 0,
        "constructor_settling_steps_nominal_source": 2000,
        "constructor_settling_steps_measured": None,
        "controls_requested": 0, "controls_returned": 0,
        "scientific_steps_requested": 0, "scientific_steps_observed": 0,
        "contact_force_calls_requested": 0, "contact_force_calls_returned": 0,
        "contact_rows": 0, "excluded_contact_count": 0,
        "nonleg_contact_count": 0, "nonpositive_leg_contact_count": 0,
        "jacobian_calls_requested": 0, "jacobian_calls_returned": 0,
        "decomposition_rows": 0, "new_neural_ticks": 0,
        "cuda_calls": 0, "fit_calls": 0, "solver_calls": 0,
        "retries": 0, "clock_events": [], "storage_errors": []}
    result = {"schema": "flylab.m3.e229.effector-contact-decomposition.result.v1",
        "validity": "incomplete", "execution_status": "running",
        "body_only": True, "original_neural_replayed": False,
        "candidate_adopted": False, "F2_complete": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "E225_10s_complete": False, "E225_all_six_eligible_cycles": 0,
        "E225_startup_previously_failed": True,
        "E225_startup_recorded_command_RMS_rad": .018052184585,
        "E225_startup_recorded_command_max_abs_rad": .05806575858,
        "E228_RF_and_RH_whole_leg_hold_supported": False}
    primary = measured = None
    body = restore_hook = thread_limits = process = None
    native = []
    force_sink = replay.ForceSink(freeze["force_fields"])
    registry = new_registry(0, 0, 0)
    rows = []

    def attempt(name, callback):
        nonlocal primary
        try:
            callback()
        except BaseException as error:
            ledger["storage_errors"].append({"artifact": name, "error": repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E229 " + name + ": " + repr(error))

    def persist():
        ledger["wall_s"] = time.perf_counter() - began
        prior.write_receipt(out / "advance-ledger.json", ledger)

    def budget():
        if time.perf_counter() - began >= limits["internal_s"] - limits[
                "finalization_reserve_s"]:
            raise TimeoutError("E229 finalization reserve reached")
        if process is not None:
            memory = process.memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0),
                                                peak)
            if peak > limits["resident_bytes"]:
                raise MemoryError("E229 resident memory cap exceeded")

    try:
        prior.write_receipt(out / "source-review-snapshot.json", review)
        paths, reference, sham_rows, sources = load_reference(freeze)
        base = StateStore.load(paths["base_checkpoint"])
        start = StateStore.load(paths["body_checkpoint"])
        from tools.fit_c_banc_m3_joint_adaptive_recruitment import one_cpu_thread
        thread_limits, threading = one_cpu_thread()
        ledger["threading"] = threading
        import psutil
        process = psutil.Process()
        budget()
        ledger["body_constructors_requested"] = 1
        persist()
        constructor_start = time.perf_counter()
        body = observation.new_body(base)
        ledger["constructor_wall_s"] = time.perf_counter() - constructor_start
        ledger["body_constructors_returned"] = 1
        body.restore(start)
        exact(start, body.snapshot(), "E229 E227 event25 body start")
        exact(start, StateStore.load(paths["e228_sham"] / "start"),
              "E229 E228 sham start")
        if not np.isclose(body.physics_time(), START * .005,
                          atol=2e-8, rtol=0):
            raise ValueError("E229 exact control1607 start clock required")
        if body.model_hash != start["modelHash"]:
            raise ValueError("E229 physical model hash differs")
        heading = replay.initial_horizontal_heading(reference[0])
        layout = layout_metadata(body)
        inventory = ground.fly_mass_inventory(body)
        prior.write_receipt(out / "layout.json", layout)
        prior.write_receipt(out / "inventory.json", inventory)
        StateStore.save(out / "body-start", body.snapshot())
        registry = new_registry(int(body.m.nq), int(body.m.nv), int(body.m.nu))
        start_row = replay.capture_physical_row(body, START,
            reference[START]["tendon_inputs"])
        replay.compare_physical_row(start_row, reference[START])
        replay.compare_physical_row(start_row, sham_rows[0])
        rows.append(start_row)
        native_step = body._step_physics
        ledger["shadow_constructors_requested"] = 1
        shadow = copy.copy(body)
        shadow.d = body.mj.MjData(body.m)
        ledger["shadow_constructors_returned"] = 1
        integration_type = body.mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(body.mj.mj_stateSize(body.m, integration_type))

        def observed_step():
            native_step()
            index = len(registry["native_state"]["native_index"])
            state_row = {"native_index": index,
                "time_s": float(body.physics_time()),
                "control_tick": START + 1 + index // STEPS,
                "qpos": np.asarray(body.d.qpos).copy(),
                "qvel": np.asarray(body.d.qvel).copy(),
                "ctrl": np.asarray(body.d.ctrl).copy(),
                "state_captured": True, "shadow_forward_returned": False,
                "observation_returned": False}
            append(registry["native_state"], state_row)
            try:
                body.mj.mj_getState(body.m, body.d, state, integration_type)
                body.mj.mj_setState(body.m, shadow.d, state, integration_type)
                body.mj.mj_forward(body.m, shadow.d)
                registry["native_state"]["shadow_forward_returned"][index] = True
                feet = replay.contact_diagnostics(shadow)
                _, rotation, _ = shadow.pose()
                rotation = np.asarray(rotation, float)
                loads = np.asarray(feet["normal_load_bw"], float)
                slip = np.asarray([np.nan if value is None else value
                    for value in feet["slip_mm_s"]], float)
                nonfoot = bool(shadow.nonfoot_contact())
                upright = float(rotation[2, 2])
                now = float(body.physics_time())
                native.append((now, loads.copy(), slip.copy(),
                    np.isfinite(slip), nonfoot, upright))
                health_bad = (not np.isfinite(loads).all()
                    or not math.isfinite(upright)
                    or not np.isfinite(body.d.qpos).all()
                    or not np.isfinite(body.d.qvel).all()
                    or nonfoot or upright < .9 or body.fault)
                force_error = None
                try:
                    force = ground.shadow_force_sample(shadow, inventory, rotation)
                    force.update(time_s=now, available=True, error="")
                    force_sink.append(force)
                except Exception as error:
                    force_sink.append({"time_s": now, "available": False,
                                       "error": repr(error)})
                    force_error = error
                if health_bad:
                    raise e113.NativeHealthStop("E229 native100us guard at " + str(now))
                if force_error is not None:
                    raise force_error
                observe_contact_step(body, shadow, registry, layout, inventory,
                    heading, ledger, science)
                registry["native_state"]["observation_returned"][index] = True
            finally:
                registry["contact_offsets"].append(len(
                    registry["contacts"]["native_index"]))

        body._step_physics = observed_step
        restore_hook = lambda: setattr(body, "_step_physics", native_step)
        with (out / "physical.jsonl").open("x", encoding="utf-8",
                                              buffering=1) as stream:
            stream.write(json.dumps(start_row, allow_nan=False) + "\n")
            for tick in range(START + 1, END + 1):
                budget()
                source = reference[tick]
                before = float(body.physics_time())
                old_native = len(native)
                receipt = {"control_tick": tick,
                    "physics_time_before_s": before,
                    "native_before": old_native, "returned": False}
                ledger["clock_events"].append(receipt)
                ledger["controls_requested"] += 1
                ledger["scientific_steps_requested"] += STEPS
                try:
                    body.step_joint_targets(np.asarray(source["joint_targets_rad"]),
                        np.asarray(source["adhesion"], bool),
                        tendon_inputs=source["tendon_inputs"], interpolate=True,
                        **({"joint_stiffness_scale": source[
                            "joint_stiffness_scale"]} if "joint_stiffness_scale"
                            in source else {}),
                        **({"adhesion_control": source["adhesion_control"]}
                            if "adhesion_control" in source else {}))
                    receipt["returned"] = True
                    ledger["controls_returned"] += 1
                finally:
                    after = float(body.physics_time())
                    interval = after - before
                    receipt["physics_time_after_s"] = after
                    receipt["actual_physical_s"] = interval if math.isfinite(
                        interval) else None
                    receipt["native_after"] = len(native)
                    receipt["state_after"] = len(
                        registry["native_state"]["native_index"])
                    ledger["scientific_steps_observed"] = len(
                        registry["native_state"]["native_index"])
                if (len(native) != old_native + STEPS
                        or len(force_sink) != len(native)):
                    raise ValueError("E229 exact50 native/force samples per control required")
                replay.compare_native_chunk(native[old_native:],
                    sources["source_native"], e228.source_slice(tick).start,
                    e228.source_slice(tick).stop)
                replay.compare_native_chunk(native[old_native:],
                    sources["sham_native"], e228.local_slice(tick).start,
                    e228.local_slice(tick).stop)
                e228.compare_force_chunk(force_sink.rows[old_native:],
                    sources["source_force"], tick)
                row = replay.capture_physical_row(body, tick,
                    source["tendon_inputs"])
                replay.compare_physical_row(row, source)
                replay.compare_physical_row(row, sham_rows[tick - START])
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        if (len(rows) != 41 or len(native) != 2000 or len(force_sink) != 2000):
            raise ValueError("E229 complete physical/native count differs")
        StateStore.save(out / "body-final", body.snapshot())
        exact(StateStore.load(paths["e228_sham"] / "final"), body.snapshot(),
              "E229 E228 sham final")
        result["physical_replay_exact"] = True
        result["native_replay_exact"] = True
        result["force_replay_exact"] = True
        state_array, contacts, attempts, decomposition = arrays(registry)
        force_array = force_sink.arrays()
        save_npz(out / "native-state.npz", state_array)
        save_npz(out / "contacts-all.npz", contacts)
        save_npz(out / "jacobian-attempts.npz", attempts)
        save_npz(out / "decomposition.npz", decomposition)
        save_npz(out / "native-contact.npz", e113.native_arrays(native))
        save_npz(out / "force-native.npz", force_array)
        checks = validate_complete(state_array, contacts, attempts, decomposition,
            force_array, inventory, heading, science)
        summary, leg_load, leg_force = aggregate_phase(contacts, decomposition,
            freeze["phases"], science["support_threshold_bw"], heading)
        save_npz(out / "leg-native.npz", {"normal_load_bw": leg_load,
            "initial_heading_force_bw": leg_force})
        during = summary["during_break"]["rf"]["braking"]
        own = during["AP_load_weighted_mm_s"]["ap_own7"]
        total = during["AP_load_weighted_mm_s"]["ap_total"]
        if during["unique_native_samples"] == 0 or total is None or own is None:
            hypothesis = "insufficient_observation"
        elif total > 0 and float(np.sum(own)) > 0:
            hypothesis = "local_kinematic_explanation_supported"
        elif float(np.sum(own)) <= 0:
            hypothesis = "own7_positive_AP_explanation_rejected"
        else:
            hypothesis = "local_kinematic_explanation_not_supported"
        result.update(validity="valid_contact_point_diagnostic",
            hypothesis_A=hypothesis,
            contact_checks=checks)
        prior.write_receipt(out / "summary.json", {"schema":
            "flylab.m3.e229.effector-contact-decomposition.summary.v1",
            "phases": summary, "contact_checks": checks,
            "hypothesis_A": hypothesis,
            "force_timing": "post-integration shadow mj_forward instantaneous force",
            "limits": ["not preceding solver impulse", "not biological muscle force",
                       "one event is not a causal joint intervention"]})
    except e113.NativeHealthStop as error:
        measured = error
        result.update(validity="measured_native_health_stop",
                      measured_error=repr(error))
    except (TimeoutError, MemoryError) as error:
        primary = error
        result.update(validity="resource_limited", primary_error=repr(error),
                      traceback=traceback.format_exc())
    except BaseException as error:
        primary = error
        result.update(validity="technical_failure", primary_error=repr(error),
                      traceback=traceback.format_exc())
    finally:
        state_array, contacts, attempts, decomposition = arrays(registry)
        attempt("native-state.npz", lambda: None if (out / "native-state.npz").exists()
                else save_npz(out / "native-state.npz", state_array))
        attempt("contacts-all.npz", lambda: None if (out / "contacts-all.npz").exists()
                else save_npz(out / "contacts-all.npz", contacts))
        attempt("jacobian-attempts.npz", lambda: None if
                (out / "jacobian-attempts.npz").exists() else save_npz(
                    out / "jacobian-attempts.npz", attempts))
        attempt("decomposition.npz", lambda: None if
                (out / "decomposition.npz").exists() else save_npz(
                    out / "decomposition.npz", decomposition))
        attempt("native-contact.npz", lambda: None if
                (out / "native-contact.npz").exists() else save_npz(
                    out / "native-contact.npz", e113.native_arrays(native)))
        attempt("force-native.npz", lambda: None if (out / "force-native.npz").exists()
                else save_npz(out / "force-native.npz", force_sink.arrays()))
        if restore_hook is not None:
            attempt("observer_cleanup", restore_hook)
        if body is not None:
            attempt("body_cleanup", body.close)
        if thread_limits is not None:
            attempt("cpu_thread_cleanup", thread_limits.restore_original_limits)
        ledger["wall_s"] = time.perf_counter() - began
        ledger["native_samples"] = len(native)
        ledger["force_samples"] = len(force_sink)
        ledger["output_bytes_before_final_receipts"] = sum(
            path.stat().st_size for path in out.rglob("*") if path.is_file())
        if ledger["wall_s"] > limits["internal_s"] and primary is None:
            primary = TimeoutError("E229 internal wall cap exceeded")
            result["validity"] = "resource_limited"
        if ledger["output_bytes_before_final_receipts"] > limits[
                "output_bytes"] and primary is None:
            primary = OSError("E229 output cap exceeded")
            result["validity"] = "technical_storage_failure"
        if primary is not None and result["validity"] == "valid_contact_point_diagnostic":
            result["validity"] = "technical_storage_failure"
        result["execution_status"] = "failed" if primary else "complete"
        if primary is not None:
            result["primary_error"] = repr(primary)
        if measured is not None:
            result["measured_error"] = repr(measured)
        attempt("result.json", lambda: prior.write_receipt(out / "result.json", result))
        attempt("advance-ledger.json", persist)
        attempt("output-manifest.json", lambda: prior.write_receipt(
            out / "output-manifest.json", replay.manifest(out)))
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.freeze, args.freeze_sha256, args.review,
                     args.review_sha256, args.out)
    except BaseException as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({"validity": result["validity"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
