"""E227 body-only replay of frozen effector commands with shadow force telemetry."""

# Preserve the first replay/health failure across partial raw cleanup.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import hashlib
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

from flylab.body import LEGS, LINKS
from flylab.c.storage import StateStore
from flylab.c.walking_trace import contact_diagnostics
from flylab.common import S, to_ui
from tools import analyze_c_banc_observation as observation
from tools import observe_c_banc_m3_ground_reaction as ground
from tools import verify_c_banc_m3_e113_body as e113
from tools import verify_c_banc_m3_neural_current_body as prior
from tools.verify_c_banc_observation import exact

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
OUT = BASE / "e227-effector-force-replay"
FREEZE_SHA = "4aca58c46354c332888f98bf83d26276f891f93c41ccc7be976150540477375c"
FREEZE_SCHEMA = "flylab.m3.e227.effector-force-replay.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e227.effector-force-replay.review.v1"
OWNED = ("tools/observe_c_banc_m3_effector_force_replay.py",
         "tests/test_c_banc_effector_force_replay.py")
FORCE_SHAPES = {"all_leg_force_bw": (6, 3),
    "full_fly_floor_force_bw": (3,),
    "nonleg_floor_force_residual_bw": (3,),
    "heading_xy": (3,), "com_position": (3,),
    "thorax_position": (3,),
    "all_leg_heading_xy_force_bw": (6,),
    "full_fly_heading_xy_force_bw": ()}
PHYSICAL_KEYS = ("body", "qpos", "qvel", "ctrl", "joint_angles_rad",
    "joint_targets_rad", "joint_tracking_error_rad", "adhesion",
    "tendon_inputs", "upright", "feet", "fault")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(relative):
    name = Path(relative)
    path = (ROOT / name).resolve()
    if name.is_absolute() or ".." in name.parts or not path.is_relative_to(ROOT):
        raise ValueError("Repository-relative E227 frozen input required")
    return path


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    """Verify all source/input/resource bytes before constructing a body."""
    if (freeze_sha != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA
            or sha256(review_path) != review_sha):
        raise ValueError("Exact E227 freeze and parent review SHA required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    native, events, limits = (freeze.get(key, {}) for key in
                              ("native", "events", "resource_limits"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("design_sha256") != sha256(safe(freeze["design"]))
            or native != {"start_control": 600, "end_control": 1663,
                "control_count": 1063, "source_rows": 1664,
                "continuation_rows": 1064, "native_samples": 53150,
                "control_dt_s": .005, "physics_dt_s": .0001,
                "steps_per_control": 50,
                "maximum_scientific_physics_steps": 53150,
                "new_neural_ticks": 0}
            or events != {"source_main_ids": [13, 14, 16, 19, 20, 21,
                22, 23, 24, 25], "pre_ticks": [-16, -4],
                "post_ticks": [4, 16], "support_threshold_bw": .02,
                "snapshot_offset_ticks": -16}
            or freeze.get("force_fields") != ["time_s", "available", "error",
                "all_leg_force_bw", "full_fly_floor_force_bw",
                "nonleg_floor_force_residual_bw", "heading_xy",
                "com_position", "thorax_position",
                "all_leg_heading_xy_force_bw", "full_fly_heading_xy_force_bw"]
            or limits != {"cpu_threads": 1, "gpu_count": 0,
                "original_attempts": 1, "retries": 0,
                "fit_calls": 0, "solver_calls": 0, "internal_s": 65.,
                "finalization_reserve_s": 8., "external_s": 75.,
                "supervisor_s": 80., "resident_bytes": 6442450944,
                "output_bytes": 134217728, "minimum_free_bytes": 2147483648}
            or any(freeze.get(key) is not False for key in (
                "candidate_adopted", "F2_complete", "F6_exposed",
                "whole_goal_complete", "active_UI_changed"))):
        raise ValueError("E227 fixed body replay/force/resource contract changed")
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Exact approved E227 source review required")
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze.get(label)
        if not isinstance(mapping, dict):
            raise TypeError("E227 frozen hash inventory required: " + label)
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("Frozen E227 input/source changed: " + name)
    for mapping in (review["source_sha256"],
                    review.get("additional_reused_source_sha256", {})):
        if not isinstance(mapping, dict):
            raise TypeError("Reviewed E227 hash mapping required")
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("Reviewed E227 source changed: " + name)
    for package, expected in freeze["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("E227 frozen package version changed: " + package)
    if output.exists() or not output.parent.is_dir() or output != OUT.resolve():
        raise FileExistsError("Exact new E227 output directory required")
    if shutil.disk_usage(output.parent).free < limits["minimum_free_bytes"]:
        raise OSError("E227 output disk reserve unavailable")
    return freeze, review


def load_reference(freeze):
    paths = {key: safe(name) for key, name in freeze["paths"].items()}
    trace = json.loads(paths["reference_trace"].read_text(encoding="utf-8"))
    rows = trace["samples"]
    with np.load(paths["reference_control"], allow_pickle=False) as archive:
        control = {key: archive[key].copy() for key in archive.files}
    with np.load(paths["reference_native"], allow_pickle=False) as archive:
        native = {key: archive[key].copy() for key in archive.files}
    events = json.loads(paths["events"].read_text(encoding="utf-8"))["events"]
    main_ids = freeze["events"]["source_main_ids"]
    if (len(rows) != 1664 or set(control) != set(prior.COMMON_FIELDS
             + prior.EFFECTOR_FIELDS) or set(native) != {"time_s",
             "normal_load_bw", "slip_mm_s", "slip_valid",
             "nonfoot_contact", "upright"}
            or len(events) <= max(main_ids)
            or [e["event_index"] for e in events if e["main_comparison"]] != main_ids
            or any(control[key].shape[0] != 1064 for key in control)
            or any(native[key].shape[0] != 53150 for key in native)
            or not np.array_equal(control["control_tick"], np.arange(600, 1664))
            or not np.array_equal(control["network_tick"],
                                  np.arange(600, 1664) * 20)
            or not np.allclose(control["physics_time_s"],
                               np.arange(600, 1664) * .005, atol=2e-8, rtol=0)
            or not np.allclose(native["time_s"],
                               3. + np.arange(1, 53151) * .0001,
                               atol=2e-8, rtol=0)
            or any(row["control_tick"] != index
                   or not np.isclose(row["time_s"], index * .005,
                                     atol=2e-8, rtol=0)
                   for index, row in enumerate(rows))):
        raise ValueError("Exact E225 partial raw/clock and E226 main events required")
    for offset, row in enumerate(rows[600:]):
        if (not np.array_equal(np.asarray(row["joint_targets_rad"]),
                               control["command"][offset])
                or not np.array_equal(np.asarray(row["motor_rates"], np.float32),
                                      control["motor_rate32"][offset])
                or not np.array_equal(np.asarray([row["tendon_inputs"][
                    leg + "_tarsus"] for leg in LEGS]),
                    control["tendon"][offset])):
            raise ValueError("E225 trace/control command seam changed")
    snapshots = {event_id: events[event_id]["entry_index"]
                 + freeze["events"]["snapshot_offset_ticks"]
                 for event_id in main_ids}
    if any(tick <= 600 or tick > 1663 for tick in snapshots.values()):
        raise ValueError("E226 force event snapshot outside replay")
    return paths, rows, control, native, events, snapshots


class ForceSink:
    """Retain only the preregistered force fields from E185's shadow sample."""

    def __init__(self, fields):
        if tuple(fields) != ("time_s", "available", "error", *FORCE_SHAPES):
            raise ValueError("Exact E227 force field order required")
        self.fields = tuple(fields)
        self.rows = []

    def __len__(self):
        return len(self.rows)

    def append(self, row):
        if not isinstance(row, dict) or not all(key in row for key in
                ("time_s", "available", "error")):
            raise ValueError("E185 force availability/error receipt required")
        kept = {key: np.array(row[key], copy=True) if isinstance(row[key], np.ndarray)
                else row[key] for key in self.fields if key in row}
        if row["available"] and set(kept) != set(self.fields):
            raise ValueError("Complete E227 force sample fields required")
        self.rows.append(kept)

    def arrays(self):
        arrays = {"time_s": np.asarray([row["time_s"] for row in self.rows],
                                       np.float64),
                  "available": np.asarray([row["available"] for row in self.rows],
                                          bool),
                  "error": np.asarray([row["error"] for row in self.rows], str)}
        for key, shape in FORCE_SHAPES.items():
            values = [row[key] if row["available"] else np.full(shape, np.nan)
                      for row in self.rows]
            arrays[key] = np.asarray(values, np.float64).reshape((len(values), *shape))
        return arrays


def capture_physical_row(body, tick, tendon_inputs):
    """Mirror only the measured physical fields of capture_control."""
    position, rotation, velocity = body.pose()
    feet = contact_diagnostics(body)
    positions = body.d.xpos
    segments = {name: to_ui(positions[int(body.body_ids[index])])
                for name, index in body.body_indices.items()
                if name in ("c_thorax", "c_head") or name.startswith("c_abdomen")}
    if "c_abdomen12" in segments:
        segments["c_abdomen"] = segments["c_abdomen12"]
    legs = {leg: [to_ui(positions[int(body.body_ids[
        body.body_indices[f"{leg}_{link}"]])]) for link in LINKS
        if f"{leg}_{link}" in body.body_indices] for leg in LEGS}
    actual = body.d.qpos[body.qpos_ids]
    targets = np.asarray(body.last_action.joint_angles)
    row = {"control_tick": tick, "time_s": tick * .005,
           "body": {"position": to_ui(position), "basis": (S @ rotation).tolist(),
               "yaw": math.atan2(-rotation[1, 0], rotation[0, 0]),
               "speed": float(velocity[3:] @ rotation[:, 0]),
               "legs": legs, "segments": segments,
               "contactsBW": feet["normal_load_bw"]},
           "qpos": body.d.qpos.tolist(), "qvel": body.d.qvel.tolist(),
           "ctrl": body.d.ctrl.tolist(),
           "joint_angles_rad": actual.tolist(),
           "joint_targets_rad": targets.tolist(),
           "joint_tracking_error_rad": (targets - actual).tolist(),
           "adhesion": np.asarray(body.last_action.adhesion_onoff,
                                  dtype=bool).tolist(),
           "tendon_inputs": dict(tendon_inputs),
           "upright": float(rotation[2, 2]), "feet": feet,
           "fault": body.fault}
    if hasattr(body, "active_length_scale"):
        row["joint_stiffness_scale"] = body.active_length_scale.tolist()
    if getattr(body, "graded_adhesion", False):
        row["adhesion_control"] = body.d.ctrl[
            body.sim._intern_adhesionactuatorids_by_fly[body.fly.name]].tolist()
    return row


def compare_physical_row(candidate, reference):
    if (candidate["control_tick"] != reference["control_tick"]
            or not np.isclose(candidate["time_s"], reference["time_s"],
                              atol=2e-8, rtol=0)):
        raise ValueError("E227 replay control clock differs")
    keys = PHYSICAL_KEYS + tuple(key for key in
        ("joint_stiffness_scale", "adhesion_control") if key in reference)
    if (set(candidate) != {"control_tick", "time_s", *keys}
            or any(key not in reference for key in keys)):
        raise ValueError("E227 physical replay fields changed")
    for key in keys:
        exact(reference[key], candidate[key], "E227 physical " + key)


def compare_native_chunk(actual, reference, start, stop):
    """Byte-compare every native field, preserving legitimate NaN slip payloads."""
    if len(actual) != stop - start:
        raise ValueError("E227 native interval sample count changed")
    arrays = e113.native_arrays(actual)
    for key in reference:
        left, right = arrays[key], reference[key][start:stop]
        if (left.shape != right.shape or left.dtype != right.dtype
                or left.tobytes() != right.tobytes()):
            raise ValueError("E227 exact native replay differs: " + key)


def initial_horizontal_heading(first_trace_row):
    ui = np.asarray(first_trace_row["body"]["basis"], np.float64)
    heading = S.T @ ui[:, 0]
    heading[2] = 0.
    norm = np.linalg.norm(heading)
    if not np.isfinite(heading).all() or norm <= 1e-12:
        raise ValueError("Recorded original control0 horizontal heading required")
    return heading / norm


def force_windows(force, reference_native, events, main_ids, frozen_events,
                  first_trace_row):
    """Split signed native force before time averaging; preserve zero denominators."""
    if (set(force) != {"time_s", "available", "error", *FORCE_SHAPES}
            or len(force["time_s"]) != 53150
            or not np.all(force["available"])
            or not np.allclose(force["time_s"], reference_native["time_s"],
                               atol=2e-8, rtol=0)):
        raise ValueError("Complete exact-clock E227 force observation required")
    heading0 = initial_horizontal_heading(first_trace_row)
    current = force["all_leg_heading_xy_force_bw"]
    original = np.einsum("tlj,j->tl", force["all_leg_force_bw"], heading0)
    load = reference_native["normal_load_bw"]
    support = load >= frozen_events["support_threshold_bw"]
    result = []
    for event_id in main_ids:
        item = events[event_id]
        entry, exit_ = item["entry_index"], item["exit_index"]
        windows = {"pre": (entry + frozen_events["pre_ticks"][0],
                           entry + frozen_events["pre_ticks"][1]),
                   "during": (entry, exit_),
                   "post": (exit_ + frozen_events["post_ticks"][0],
                            exit_ + frozen_events["post_ticks"][1])}
        report = {"event_index": event_id, "entry_index": entry,
                  "exit_index": exit_, "windows": {}}
        for name, (a, b) in windows.items():
            if a < 601 or b > 1664 or a >= b:
                report["windows"][name] = {"available": False,
                    "reason": "outside_complete_replay", "control_indices": None}
                continue
            first, last = (a - 601) * 50, (b - 601) * 50
            mask = support[first:last]
            def signed(values, start=first, stop=last, supported=mask):
                block = values[start:stop]
                plus = np.maximum(block, 0.)
                minus = np.maximum(-block, 0.)
                pos = (plus * supported).sum(axis=0) * .0001
                neg = (minus * supported).sum(axis=0) * .0001
                total_negative = float(neg.sum())
                observed = supported.sum(axis=0)
                means = np.divide((block * supported).sum(axis=0), observed,
                    out=np.full(6, np.nan), where=observed > 0)
                return {"positive_supported_BW_s": pos.tolist(),
                    "negative_supported_BW_s": neg.tolist(),
                    "negative_share": (neg / total_negative).tolist()
                        if total_negative > 0 else None,
                    "mean_supported_BW": [float(value) if count else None
                         for value, count in zip(means, observed, strict=True)]}
            report["windows"][name] = {"available": True,
                "control_indices": list(range(a, b)),
                "native_start_index": first,
                "native_stop_index_exclusive": last,
                "native_samples": last - first,
                "supported_time_s": (mask.sum(axis=0) * .0001).tolist(),
                "current_horizontal": signed(current),
                "original_initial_horizontal": signed(original),
                "full_fly_heading_xy_force_BW_s": float(
                    force["full_fly_heading_xy_force_bw"][first:last].sum() * .0001)}
        result.append(report)
    return {"source": "E185 post-integration shadow mj_forward force",
        "force_unit": "body-weight normalized instantaneous force; integral BW*s",
        "support_threshold_bw": frozen_events["support_threshold_bw"],
        "initial_heading_source": "original trace control0, native horizontal XY",
        "initial_heading_xy": heading0.tolist(),
        "current_heading_source": "per-native-sample shadow heading_xy",
        "events": result,
        "limitations": ["not preceding solver impulse", "not biological muscle force",
                        "descriptive contact-force association, not causal leg attribution"]}


def save_npz(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def manifest(out):
    return {str(path.relative_to(out)).replace("\\", "/"): {
        "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(out.rglob("*")) if path.is_file()
        and path.name != "output-manifest.json"}


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze, review = gate(Path(freeze_path), freeze_sha,
                          Path(review_path), review_sha, out)
    out.mkdir(parents=False, exist_ok=False)
    limits = freeze["resource_limits"]
    ledger = {"schema": "flylab.m3.e227.effector-force-replay.ledger.v1",
        "status": "gate_passed", "body_constructors_requested": 0,
        "body_constructors_returned": 0, "constructor_settling_steps_nominal_source": 2000,
        "constructor_settling_steps_measured": None,
        "scientific_steps_requested": 0, "scientific_steps_observed": 0,
        "scientific_controls_requested": 0, "scientific_controls_returned": 0,
        "new_neural_ticks": 0, "cuda_calls": 0, "fit_calls": 0,
        "solver_calls": 0, "retries": 0, "storage_errors": [],
        "clock_events": [], "snapshot_controls": []}
    result = {"schema": "flylab.m3.e227.effector-force-replay.result.v1",
        "validity": "incomplete", "execution_status": "running",
        "exact_physical_replay": False, "exact_native_replay": False,
        "force_complete": False, "body_only": True,
        "E225_10s_complete": False,
        "E225_startup_previously_failed": True,
        "E225_startup_recorded_command_RMS_rad": .018052184585,
        "E225_startup_recorded_command_max_abs_rad": .05806575858,
        "candidate_adopted": False, "F2_complete": False,
        "F6_exposed": False, "whole_goal_complete": False}
    primary = measured = None
    body = restore_hook = thread_limits = process = None
    native = []
    force_sink = ForceSink(freeze["force_fields"])
    inventory = {}
    rows = []
    durable_control = 600

    def persist():
        ledger["wall_s"] = time.perf_counter() - began
        prior.write_receipt(out / "advance-ledger.json", ledger)

    def attempt(name, callback):
        nonlocal primary
        try:
            callback()
        except BaseException as error:
            ledger["storage_errors"].append({"artifact": name, "error": repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E227 " + name + ": " + repr(error))

    def budget():
        if time.perf_counter() - began >= limits["internal_s"] - limits[
                "finalization_reserve_s"]:
            raise TimeoutError("E227 finalization reserve reached")
        if process is not None:
            memory = process.memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0),
                                                peak)
            if peak > limits["resident_bytes"]:
                raise MemoryError("E227 resident memory cap exceeded")

    try:
        prior.write_receipt(out / "source-review-snapshot.json", review)
        paths, reference, _reference_control, reference_native, events, snapshots = (
            load_reference(freeze))
        saved = StateStore.load(paths["checkpoint"])
        from tools.fit_c_banc_m3_joint_adaptive_recruitment import one_cpu_thread
        thread_limits, threads = one_cpu_thread()
        ledger["threading"] = threads
        import psutil
        process = psutil.Process()
        budget()
        ledger["status"] = "body_constructor_requested"
        ledger["body_constructors_requested"] = 1
        persist()
        constructor_start = time.perf_counter()
        body = observation.new_body(saved)
        ledger["constructor_wall_s"] = time.perf_counter() - constructor_start
        ledger["body_constructors_returned"] = 1
        exact(saved["body"], body.snapshot(), "E227 restored body-only state")
        if body.model_hash != saved["body"]["modelHash"]:
            raise ValueError("E227 restored physical model hash changed")
        start_row = capture_physical_row(body, 600, reference[600]["tendon_inputs"])
        compare_physical_row(start_row, reference[600])
        StateStore.save(out / "body-start", body.snapshot())
        ledger["snapshot_controls"].append(600)
        prior.write_receipt(out / "inventory.json", ground.fly_mass_inventory(body))
        restore_hook = ground.install_native_observer(body, native, force_sink,
            inventory, contact_diagnostics, e113.NativeHealthStop)
        with (out / "replay-control.jsonl").open("x", encoding="utf-8",
                                                buffering=1) as stream:
            stream.write(json.dumps(start_row, allow_nan=False) + "\n")
            for tick in range(601, 1664):
                budget()
                source = reference[tick]
                request = source.get("neural_request")
                if request is not None and (
                        request.get("external_force_at_start")
                        or (request.get("cuts") or {}).get("motor")):
                    raise ValueError("E227 frozen replay has an unmodeled external request")
                before_time = float(body.physics_time())
                before_native = len(native)
                event = {"control_tick": tick, "requested": True,
                    "returned": False, "native_before": before_native,
                    "physics_time_before_s": before_time}
                ledger["clock_events"].append(event)
                ledger["scientific_controls_requested"] += 1
                ledger["scientific_steps_requested"] += 50
                try:
                    body.step_joint_targets(np.asarray(source["joint_targets_rad"]),
                        np.asarray(source["adhesion"], bool),
                        tendon_inputs=source["tendon_inputs"], interpolate=True,
                        **({"joint_stiffness_scale": source["joint_stiffness_scale"]}
                           if "joint_stiffness_scale" in source else {}),
                        **({"adhesion_control": source["adhesion_control"]}
                           if "adhesion_control" in source else {}))
                    event["returned"] = True
                    ledger["scientific_controls_returned"] += 1
                finally:
                    after = float(body.physics_time())
                    event["physics_time_after_s"] = after
                    event["actual_physical_s"] = (after - before_time
                        if np.isfinite(after - before_time) else None)
                    event["native_after"] = len(native)
                    ledger["scientific_steps_observed"] = len(native)
                if len(native) != before_native + 50 or len(force_sink) != len(native):
                    raise ValueError("E227 exact50 native/force samples per control required")
                compare_native_chunk(native[before_native:], reference_native,
                                     before_native, len(native))
                row = capture_physical_row(body, tick, source["tendon_inputs"])
                compare_physical_row(row, source)
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                durable_control = tick
                for event_id, at in snapshots.items():
                    if at == tick:
                        StateStore.save(out / f"event-start-{event_id:02d}",
                                        body.snapshot())
                        ledger["snapshot_controls"].append(tick)
        if (len(native) != 53150 or len(force_sink) != 53150
                or durable_control != 1663):
            raise ValueError("E227 exact 5.315-second replay incomplete")
        StateStore.save(out / "body-final", body.snapshot())
        ledger["snapshot_controls"].append(1663)
        result["exact_physical_replay"] = True
        result["exact_native_replay"] = True
        arrays = force_sink.arrays()
        native_arrays = e113.native_arrays(native)
        save_npz(out / "force-native.npz", arrays)
        save_npz(out / "replay-native-contact.npz", native_arrays)
        result["force_complete"] = bool(np.all(arrays["available"]))
        if not result["force_complete"]:
            raise ValueError("E227 force observer incomplete")
        force_events = force_windows(arrays, reference_native, events,
            freeze["events"]["source_main_ids"], freeze["events"], reference[0])
        prior.write_receipt(out / "force-events.json", force_events)
        negative_by_leg = np.zeros(6, np.float64)
        for event in force_events["events"]:
            during = event["windows"]["during"]
            if during["available"]:
                negative_by_leg += np.asarray(during["current_horizontal"][
                    "negative_supported_BW_s"], np.float64)
        negative_total = float(negative_by_leg.sum())
        summary = {"schema": "flylab.m3.e227.effector-force-replay.summary.v1",
            "exact_physical_replay": True, "exact_native_replay": True,
            "force_complete": True, "scientific_native_samples": len(native),
            "main_event_count": len(force_events["events"]),
            "body_only": True, "neural_firing_replayed": False,
            "main_during_negative_supported_BW_s_by_leg": negative_by_leg.tolist(),
            "main_during_negative_share_by_leg": (
                negative_by_leg / negative_total).tolist()
                if negative_total > 0 else None,
            "E225_startup_previously_failed": True,
            "E225_10s_complete": False,
            "causal_leg_attribution_established": False,
            "force_timing": "post-integration shadow mj_forward instantaneous force"}
        prior.write_receipt(out / "force-summary.json", summary)
        result.update(validity="valid_exact_body_force_diagnostic",
                      summary=summary)
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
        attempt("force-native.npz", lambda: None if (out / "force-native.npz").exists()
                else save_npz(out / "force-native.npz", force_sink.arrays()))
        attempt("replay-native-contact.npz", lambda: None if
                (out / "replay-native-contact.npz").exists() else save_npz(
                    out / "replay-native-contact.npz", e113.native_arrays(native)))
        for label, callback in (("observer", restore_hook),
                                ("body", body.close if body else None),
                                ("cpu_thread", thread_limits.restore_original_limits
                                 if thread_limits else None)):
            if callback is not None:
                attempt("cleanup_" + label, callback)
        ledger["last_durable_control"] = durable_control
        ledger["native_samples"] = len(native)
        ledger["force_samples"] = len(force_sink)
        ledger["wall_s"] = time.perf_counter() - began
        if ledger["wall_s"] > limits["internal_s"]:
            error = TimeoutError("E227 internal wall cap exceeded")
            if primary is None:
                primary = error
                result["validity"] = "resource_limited"
            else:
                primary.add_note("Secondary E227 wall cap: " + repr(error))
        ledger["output_bytes"] = sum(path.stat().st_size for path in out.rglob("*")
                                     if path.is_file())
        if ledger["output_bytes"] > limits["output_bytes"]:
            error = OSError("E227 output cap exceeded")
            if primary is None:
                primary = error
                result["validity"] = "technical_storage_failure"
            else:
                primary.add_note("Secondary E227 output cap: " + repr(error))
        if primary is not None and result["validity"] == "valid_exact_body_force_diagnostic":
            result["validity"] = "technical_storage_failure"
        result["execution_status"] = "failed" if primary else "complete"
        if primary is not None:
            result["primary_error"] = repr(primary)
        if measured is not None:
            result["measured_error"] = repr(measured)
        attempt("result.json", lambda: prior.write_receipt(out / "result.json", result))
        attempt("advance-ledger.json", persist)
        attempt("output-manifest.json", lambda: prior.write_receipt(
            out / "output-manifest.json", manifest(out)))
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
