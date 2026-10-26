"""E230 body-only, one-joint RF tibia command hold with contact decomposition."""

# Retain the first physical, observation, resource, or storage failure.
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
from tools import analyze_c_banc_observation as observation
from tools import observe_c_banc_m3_effector_contact_decomposition as e229
from tools import observe_c_banc_m3_effector_force_replay as replay
from tools import observe_c_banc_m3_ground_reaction as ground
from tools import probe_c_banc_m3_local_command_hold as e228
from tools import verify_c_banc_m3_e113_body as e113
from tools import verify_c_banc_m3_neural_current_body as prior
from tools.verify_c_banc_observation import exact

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
OUT = BASE / "e230-rf-tibia-hold"
FREEZE_SHA = "4063f1087960dd0e5357000b486d5b30351402a2cf49d6be4f9eb7c869451c88"
FREEZE_SCHEMA = "flylab.m3.e230.rf-tibia-hold.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e230.rf-tibia-hold.review.v1"
OWNED = ("tools/probe_c_banc_m3_rf_tibia_hold.py",
         "tests/test_c_banc_rf_tibia_hold.py")
ARMS = ("sham", "rf_tibia_hold")
START, END, ANCHOR = 1607, 1647, 1619
HOLD_FIRST, HOLD_LAST = 1620, 1635
JOINT = 26
PHASES = dict(e229.PHASES)


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if (freeze_sha != FREEZE_SHA or replay.sha256(freeze_path) != FREEZE_SHA
            or replay.sha256(review_path) != review_sha):
        raise ValueError("Exact E230 freeze and parent source-review hashes required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or replay.sha256(replay.safe(freeze["design"])) != freeze.get(
                "design_sha256")
            or freeze.get("arms") != list(ARMS)
            or freeze.get("event") != {"event_id": 25,
                "start_control": START, "end_control": END,
                "entry_control": 1623, "exit_control": 1631,
                "anchor_control": ANCHOR, "hold_start_control": HOLD_FIRST,
                "hold_end_control": HOLD_LAST,
                "changed_joint_indices": [JOINT],
                "velocity_summary_controls": list(range(1620, 1647))}
            or freeze.get("phases") != PHASES
            or freeze.get("native") != {"control_dt_s": .005,
                "physics_dt_s": .0001, "steps_per_control": 50,
                "controls_per_arm": 40, "samples_per_arm": 2000,
                "max_scientific_steps": 4000, "new_neural_ticks": 0,
                "support_threshold_bw": .02, "contact_rows_max": 32000,
                "jacobian_calls_max": 64000,
                "decomposition_atol": 1e-10, "force_sum_atol": 1e-10}
            or freeze.get("force_fields") != ["time_s", "available", "error",
                *replay.FORCE_SHAPES]
            or freeze.get("diagnostic_thresholds") != {
                "backward_distance_reduction_fraction": .25,
                "minimum_forward_speed_gain_mm_s": .5,
                "maximum_forward_distance_loss_mm": .02,
                "rf_negative_integral_reduction_fraction": .2}
            or freeze.get("resource_limits") != {"cpu_threads": 1,
                "gpu_count": 0, "original_attempts": 1, "retries": 0,
                "body_constructors": 1, "shadow_constructors": 1,
                "fit_calls": 0, "solver_calls": 0, "internal_s": 30.,
                "finalization_reserve_s": 5., "external_s": 35.,
                "supervisor_s": 40., "resident_bytes": 6442450944,
                "output_bytes": 134217728,
                "minimum_free_bytes": 2147483648}
            or any(freeze.get(key) is not False for key in (
                "candidate_adopted", "F2_complete", "F6_exposed",
                "whole_goal_complete", "active_UI_changed"))):
        raise ValueError("E230 frozen arm/science/resource contract changed")
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Parent-approved E230 source review required")
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze.get(label)
        if not isinstance(mapping, dict):
            raise TypeError("E230 frozen hash inventory required: " + label)
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or replay.sha256(replay.safe(name)) != expected):
                raise ValueError("Frozen E230 input/source changed: " + name)
    for mapping in (review["source_sha256"],
                    review.get("additional_reused_source_sha256", {})):
        if not isinstance(mapping, dict):
            raise TypeError("Reviewed E230 hash mapping required")
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or replay.sha256(replay.safe(name)) != expected):
                raise ValueError("Reviewed E230 source changed: " + name)
    for package, expected in freeze["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("Frozen E230 package version changed: " + package)
    if output != OUT.resolve() or output.exists() or not output.parent.is_dir():
        raise FileExistsError("Exact new E230 output directory required")
    if shutil.disk_usage(output.parent).free < freeze["resource_limits"][
            "minimum_free_bytes"]:
        raise OSError("E230 output disk reserve unavailable")
    return freeze, review


def tibia_command(original, anchor, arm, tick):
    if arm not in ARMS or not START < tick <= END:
        raise ValueError("E230 fixed arm/control required")
    source = np.asarray(original, np.float64)
    held = np.asarray(anchor, np.float64)
    if (source.shape != (42,) or held.shape != (42,) or not np.isfinite(source).all()
            or not np.isfinite(held).all()):
        raise ValueError("Finite 42-joint original/anchor targets required")
    applied = source.copy()
    if arm == "rf_tibia_hold" and HOLD_FIRST <= tick <= HOLD_LAST:
        applied[JOINT] = held[JOINT]
    validate_tibia_command(source, applied, held, arm, tick)
    return applied


def validate_tibia_command(original, applied, anchor, arm, tick):
    expected = np.asarray(original, np.float64).copy()
    if arm == "rf_tibia_hold" and HOLD_FIRST <= tick <= HOLD_LAST:
        expected[JOINT] = np.asarray(anchor, np.float64)[JOINT]
    if not np.array_equal(np.asarray(applied), expected):
        raise ValueError("E230 command changed outside joint26/frozen hold window")


def load_reference(freeze):
    paths, trace, sham_rows, sources = e229.load_reference(freeze)
    original = paths["e229_reference"]
    archives = {}
    for key, filename in (("native_state", "native-state.npz"),
                          ("contacts", "contacts-all.npz"),
                          ("attempts", "jacobian-attempts.npz"),
                          ("decomposition", "decomposition.npz"),
                          ("native", "native-contact.npz"),
                          ("force", "force-native.npz")):
        with np.load(original / filename, allow_pickle=False) as archive:
            archives[key] = {name: archive[name].copy() for name in archive.files}
    reference_layout = json.loads((original / "layout.json").read_text(
        encoding="utf-8"))
    reference_inventory = json.loads((original / "inventory.json").read_text(
        encoding="utf-8"))
    if (len(archives["native_state"]["native_index"]) != 2000
            or len(archives["contacts"]["offsets"]) != 2001
            or len(archives["native"]["time_s"]) != 2000
            or len(archives["force"]["time_s"]) != 2000):
        raise ValueError("Exact E229 point-contact raw reference required")
    for key in sources["sham_native"]:
        left, right = sources["sham_native"][key], archives["native"][key]
        if (left.dtype != right.dtype or left.shape != right.shape
                or left.tobytes() != right.tobytes()):
            raise ValueError("E230 E229/E228 sham native source differs")
    for key in sources["sham_force"]:
        left, right = sources["sham_force"][key], archives["force"][key]
        if (left.dtype != right.dtype or left.shape != right.shape
                or left.tobytes() != right.tobytes()):
            raise ValueError("E230 E229/E228 sham force source differs")
    return paths, trace, sham_rows, sources, archives, reference_layout, reference_inventory


def compare_reference_slice(registry, source, first, stop):
    """Byte-exact E229 state/contact/Jac/derived rows for a completed prefix."""
    state, contacts, attempts, decomp = e229.arrays(registry)
    for label, actual in (("native_state", state), ("contacts", contacts),
                          ("attempts", attempts), ("decomposition", decomp)):
        expected = source[label]
        if set(actual) != set(expected):
            raise ValueError("E230 E229 " + label + " raw keys differ")
        if label == "native_state":
            actual_first, actual_stop = first, stop
            expected_first, expected_stop = first, stop
        elif label == "contacts":
            left, right = actual["offsets"], expected["offsets"]
            if (len(left) <= stop or not np.array_equal(left[first:stop + 1],
                                                         right[first:stop + 1])):
                raise ValueError("E230 E229 contact offsets differ")
            actual_first, actual_stop = int(left[first]), int(left[stop])
            expected_first, expected_stop = int(right[first]), int(right[stop])
        else:
            a_index = actual["native_index"]
            e_index = expected["native_index"]
            actual_first = int(np.searchsorted(a_index, first, side="left"))
            actual_stop = int(np.searchsorted(a_index, stop, side="left"))
            expected_first = int(np.searchsorted(e_index, first, side="left"))
            expected_stop = int(np.searchsorted(e_index, stop, side="left"))
        for key in actual:
            if key == "offsets":
                continue
            left = actual[key][actual_first:actual_stop]
            right = expected[key][expected_first:expected_stop]
            if (left.shape != right.shape or left.dtype != right.dtype
                    or left.tobytes() != right.tobytes()):
                raise ValueError("E230 E229 exact state/contact/Jac differs: " +
                                 label + "." + key)


def com_metrics(force, native_state, heading, root_dofs,
                source="initial_control0"):
    """Native COM gradient; during break is local native [750:1150)."""
    time_s = np.asarray(force["time_s"], np.float64)
    center = np.asarray(force["com_position"], np.float64)
    root = np.asarray(native_state["qvel"], np.float64)
    initial = np.asarray(heading, np.float64)
    root_indices = np.asarray(root_dofs, np.int64)
    if (time_s.shape != (2000,) or center.shape != (2000, 3)
            or root.ndim != 2 or len(root) != 2000
            or root_indices.shape != (6,) or np.any(root_indices >= root.shape[1])
            or initial.shape != (3,) or not np.isclose(np.linalg.norm(initial), 1.)
            or not np.allclose(time_s, 8.035 + (np.arange(2000) + 1) * .0001,
                               atol=2e-8, rtol=0)
            or not np.isfinite(center).all() or not np.isfinite(root).all()):
        raise ValueError("Complete finite E230 native COM/root clock required")
    position = center @ initial
    speed = np.gradient(position, .0001)
    during = speed[750:1150]
    summary = {"heading_source": source,
        "during_break_native_indices": [750, 1150],
        "during_break_backward_distance_mm": float(
            np.maximum(-during, 0.).sum() * .0001),
        "during_break_min_speed_mm_s": float(during.min()),
        "during_break_mean_speed_mm_s": float(during.mean()),
        "native_first_to_last_COM_displacement_mm": float(
            position[-1] - position[0]),
        "native_first_to_last_interval_s": float(time_s[-1] - time_s[0])}
    return {"time_s": time_s,
        "initial_heading_position_mm": position,
        "initial_heading_speed_mm_s": speed,
        "root_angular_velocity_rad_s": root[:, root_indices[3:6]]}, summary


def first_state_difference(registry, source):
    """Find the first retained native qpos/qvel/ctrl difference, even mid-control."""
    state, *_ = e229.arrays(registry)
    count = len(state["native_index"])
    first = None
    for key in ("qpos", "qvel", "ctrl"):
        offset = e228.first_array_difference(state[key], source["native_state"][
            key][:count])
        if offset is not None:
            first = offset if first is None else min(first, offset)
    return first


def save_arm_raw(out, registry, native, force_sink, on_error=None):
    state, contacts, attempts, decomposition = e229.arrays(registry)
    payloads = {"native-state.npz": state, "contacts-all.npz": contacts,
        "jacobian-attempts.npz": attempts,
        "decomposition.npz": decomposition,
        "native-contact.npz": e113.native_arrays(native),
        "force-native.npz": force_sink.arrays()}
    for name, payload in payloads.items():
        if not (out / name).exists():
            try:
                e229.save_npz(out / name, payload)
            except BaseException as error:
                if on_error is None:
                    raise
                on_error(name, error)
    return state, contacts, attempts, decomposition, payloads["force-native.npz"]


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze, review = gate(Path(freeze_path), freeze_sha,
                          Path(review_path), review_sha, out)
    out.mkdir(parents=False, exist_ok=False)
    limits, science = freeze["resource_limits"], freeze["native"]
    ledger = {"schema": "flylab.m3.e230.rf-tibia-hold.ledger.v1",
        "body_constructors_requested": 0, "body_constructors_returned": 0,
        "shadow_constructors_requested": 0, "shadow_constructors_returned": 0,
        "constructor_settling_steps_nominal_source": 2000,
        "constructor_settling_steps_measured": None,
        "controls_requested": 0, "controls_returned": 0,
        "scientific_steps_requested": 0, "scientific_steps_observed": 0,
        "new_neural_ticks": 0, "cuda_calls": 0, "fit_calls": 0,
        "solver_calls": 0, "retries": 0, "arms": {}, "storage_errors": []}
    result = {"schema": "flylab.m3.e230.rf-tibia-hold.result.v1",
        "validity": "incomplete", "execution_status": "running",
        "body_only_local_diagnostic": True, "arm_order": list(ARMS),
        "arms": {}, "candidate_adopted": False, "F2_complete": False,
        "F6_exposed": False, "whole_goal_complete": False,
        "E225_10s_complete": False, "E225_all_six_eligible_cycles": 0,
        "E225_startup_previously_failed": True,
        "E225_startup_recorded_command_RMS_rad": .018052184585,
        "E225_startup_recorded_command_max_abs_rad": .05806575858,
        "E228_RF_and_RH_whole_leg_hold_supported": False,
        "E229_own7_positive_AP_hypothesis_supported": False}
    primary = None
    body = restore_hook = thread_limits = process = None
    native = []
    force_sink = replay.ForceSink(freeze["force_fields"])
    registry = e229.new_registry(0, 0, 0)
    rows = []
    metrics = {}

    def record_storage_error(name, error):
        nonlocal primary
        ledger["storage_errors"].append({"artifact": name, "error": repr(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E230 " + name + ": " + repr(error))

    def attempt(name, callback):
        try:
            callback()
        except BaseException as error:
            record_storage_error(name, error)

    def persist():
        ledger["wall_s"] = time.perf_counter() - began
        prior.write_receipt(out / "advance-ledger.json", ledger)

    def budget():
        if time.perf_counter() - began >= limits["internal_s"] - limits[
                "finalization_reserve_s"]:
            raise TimeoutError("E230 finalization reserve reached")
        if process is not None:
            memory = process.memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0),
                                                peak)
            if peak > limits["resident_bytes"]:
                raise MemoryError("E230 resident memory cap exceeded")

    try:
        prior.write_receipt(out / "source-review-snapshot.json", review)
        paths, reference, sham_rows, sources, e229_reference, original_layout, (
            original_inventory) = load_reference(freeze)
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
        if body.model_hash != start["modelHash"]:
            raise ValueError("E230 frozen body model hash changed")
        layout = e229.layout_metadata(body)
        inventory = ground.fly_mass_inventory(body)
        exact(original_layout, layout, "E230 E229 anatomical layout")
        exact(original_inventory, inventory, "E230 E229 body inventory")
        prior.write_receipt(out / "layout.json", layout)
        prior.write_receipt(out / "inventory.json", inventory)
        heading = replay.initial_horizontal_heading(reference[0])
        ledger["shadow_constructors_requested"] = 1
        shadow = copy.copy(body)
        shadow.d = body.mj.MjData(body.m)
        ledger["shadow_constructors_returned"] = 1
        integration_type = body.mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(body.mj.mj_stateSize(body.m, integration_type))
        native_step = body._step_physics
        (out / "arms").mkdir()
        for arm in ARMS:
            budget()
            arm_out = out / "arms" / arm
            arm_out.mkdir()
            native = []
            force_sink = replay.ForceSink(freeze["force_fields"])
            registry = e229.new_registry(int(body.m.nq), int(body.m.nv),
                                         int(body.m.nu))
            rows = []
            arm_ledger = {"status": "restore_requested",
                "controls_requested": 0, "controls_returned": 0,
                "scientific_steps_requested": 0, "scientific_steps_observed": 0,
                "contact_force_calls_requested": 0,
                "contact_force_calls_returned": 0, "contact_rows": 0,
                "excluded_contact_count": 0, "nonleg_contact_count": 0,
                "nonpositive_leg_contact_count": 0,
                "jacobian_calls_requested": 0, "jacobian_calls_returned": 0,
                "decomposition_rows": 0, "clock_events": []}
            ledger["arms"][arm] = arm_ledger
            arm_result = {"arm": arm, "status": "running",
                "complete_healthy": False,
                "first_qpos_difference_control": None,
                "first_qvel_difference_control": None,
                "first_ctrl_difference_control": None,
                "first_native_difference_sample": None,
                "first_native_state_difference_sample_local": None,
                "first_physical_difference_control": None}
            result["arms"][arm] = arm_result

            def observed_step(registry=registry, native=native,
                              force_sink=force_sink, arm=arm,
                              arm_ledger=arm_ledger):
                native_step()
                index = len(registry["native_state"]["native_index"])
                e229.append(registry["native_state"], {
                    "native_index": index, "time_s": float(body.physics_time()),
                    "control_tick": START + 1 + index // 50,
                    "qpos": np.asarray(body.d.qpos).copy(),
                    "qvel": np.asarray(body.d.qvel).copy(),
                    "ctrl": np.asarray(body.d.ctrl).copy(),
                    "state_captured": True,
                    "shadow_forward_returned": False,
                    "observation_returned": False})
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
                        raise e113.NativeHealthStop("E230 native100us guard at " + str(now))
                    if force_error is not None:
                        raise force_error
                    arm_science = dict(science)
                    arm_science["contact_rows_max"] -= sum(
                        entry["contact_rows"] for name, entry in ledger[
                            "arms"].items() if name != arm)
                    arm_science["jacobian_calls_max"] -= sum(
                        entry["jacobian_calls_requested"] for name, entry in ledger[
                            "arms"].items() if name != arm)
                    e229.observe_contact_step(body, shadow, registry, layout,
                        inventory, heading, arm_ledger, arm_science)
                    registry["native_state"]["observation_returned"][index] = True
                finally:
                    registry["contact_offsets"].append(len(
                        registry["contacts"]["native_index"]))

            try:
                body.restore(start)
                exact(start, body.snapshot(), "E230 same event25 start " + arm)
                if not np.isclose(body.physics_time(), START * .005,
                                  atol=2e-8, rtol=0):
                    raise ValueError("E230 restored control1607 clock changed")
                start_row = replay.capture_physical_row(body, START,
                    reference[START]["tendon_inputs"])
                replay.compare_physical_row(start_row, reference[START])
                replay.compare_physical_row(start_row, sham_rows[0])
                StateStore.save(arm_out / "body-start", body.snapshot())
                rows.append(start_row)
                if restore_hook is not None:
                    restore_hook()
                    restore_hook = None
                body._step_physics = observed_step
                restore_hook = lambda: setattr(body, "_step_physics", native_step)
                arm_ledger["status"] = "start_exact"
                with ((arm_out / "commands.jsonl").open("x", encoding="utf-8",
                                                       buffering=1) as commands,
                      (arm_out / "physical.jsonl").open("x", encoding="utf-8",
                                                       buffering=1) as physical):
                    physical.write(json.dumps(start_row, allow_nan=False) + "\n")
                    anchor = reference[ANCHOR]["joint_targets_rad"]
                    for tick in range(START + 1, END + 1):
                        budget()
                        source = reference[tick]
                        applied = tibia_command(source["joint_targets_rad"],
                                                anchor, arm, tick)
                        commands.write(json.dumps({"control_tick": tick,
                            "requested_command": source["joint_targets_rad"],
                            "applied_command": applied.tolist(),
                            "adhesion": source["adhesion"],
                            "tendon_inputs": source["tendon_inputs"],
                            "joint_stiffness_scale": source.get(
                                "joint_stiffness_scale"),
                            "adhesion_control": source.get("adhesion_control")},
                            allow_nan=False) + "\n")
                        before = float(body.physics_time())
                        old = len(native)
                        receipt = {"control_tick": tick, "native_before": old,
                            "physics_time_before_s": before if math.isfinite(
                                before) else None, "returned": False}
                        arm_ledger["clock_events"].append(receipt)
                        arm_ledger["controls_requested"] += 1
                        ledger["controls_requested"] += 1
                        arm_ledger["scientific_steps_requested"] += 50
                        ledger["scientific_steps_requested"] += 50
                        try:
                            body.step_joint_targets(applied,
                                np.asarray(source["adhesion"], bool),
                                tendon_inputs=source["tendon_inputs"],
                                interpolate=True,
                                **({"joint_stiffness_scale": source[
                                    "joint_stiffness_scale"]} if
                                    "joint_stiffness_scale" in source else {}),
                                **({"adhesion_control": source["adhesion_control"]}
                                    if "adhesion_control" in source else {}))
                            receipt["returned"] = True
                            arm_ledger["controls_returned"] += 1
                            ledger["controls_returned"] += 1
                        finally:
                            after = float(body.physics_time())
                            interval = after - before
                            receipt["physics_time_after_s"] = after if math.isfinite(
                                after) else None
                            receipt["actual_physical_s"] = interval if math.isfinite(
                                interval) else None
                            receipt["native_after"] = len(native)
                            receipt["state_after"] = len(
                                registry["native_state"]["native_index"])
                            arm_ledger["scientific_steps_observed"] = len(
                                registry["native_state"]["native_index"])
                            ledger["scientific_steps_observed"] = sum(
                                item["scientific_steps_observed"] for item in
                                ledger["arms"].values())
                        if len(native) != old + 50 or len(force_sink) != len(native):
                            raise ValueError("E230 exact50 native/force per control required")
                        if arm == "sham" or tick <= ANCHOR:
                            replay.compare_native_chunk(native[old:],
                                sources["source_native"], e228.source_slice(tick).start,
                                e228.source_slice(tick).stop)
                            replay.compare_native_chunk(native[old:],
                                sources["sham_native"], e228.local_slice(tick).start,
                                e228.local_slice(tick).stop)
                            e228.compare_force_chunk(force_sink.rows[old:],
                                sources["source_force"], tick)
                            compare_reference_slice(registry, e229_reference,
                                e228.local_slice(tick).start,
                                e228.local_slice(tick).stop)
                        row = replay.capture_physical_row(body, tick,
                            source["tendon_inputs"])
                        exact(applied.tolist(), row["joint_targets_rad"],
                              "E230 applied target")
                        exact(source["adhesion"], row["adhesion"],
                              "E230 unchanged adhesion")
                        exact(source["tendon_inputs"], row["tendon_inputs"],
                              "E230 unchanged tendons")
                        if arm == "sham" or tick <= ANCHOR:
                            replay.compare_physical_row(row, source)
                            replay.compare_physical_row(row, sham_rows[tick - START])
                        else:
                            for key, receipt_key in (("qpos",
                                    "first_qpos_difference_control"),
                                    ("qvel", "first_qvel_difference_control"),
                                    ("ctrl", "first_ctrl_difference_control")):
                                if arm_result[receipt_key] is None:
                                    try:
                                        exact(source[key], row[key], key)
                                    except AssertionError:
                                        arm_result[receipt_key] = tick
                            if arm_result["first_physical_difference_control"] is None:
                                physical = (arm_result[
                                    "first_qpos_difference_control"], arm_result[
                                    "first_qvel_difference_control"], arm_result[
                                    "first_ctrl_difference_control"])
                                if any(value is not None for value in physical):
                                    arm_result["first_physical_difference_control"] = min(
                                        value for value in physical if value is not None)
                        rows.append(row)
                        physical.write(json.dumps(row, allow_nan=False) + "\n")
                if len(rows) != 41 or len(native) != 2000 or len(force_sink) != 2000:
                    raise ValueError("E230 completed arm count differs")
                StateStore.save(arm_out / "body-final", body.snapshot())
                if arm == "sham":
                    exact(StateStore.load(paths["e228_sham"] / "final"),
                          body.snapshot(), "E230 E228 sham final")
                    exact(StateStore.load(paths["e229_reference"] / "body-final"),
                          body.snapshot(), "E230 E229 sham final")
                state_array, contacts, attempts, decomposition, force_array = (
                    save_arm_raw(arm_out, registry, native, force_sink))
                checks = e229.validate_complete(state_array, contacts, attempts,
                    decomposition, force_array, inventory, heading, science)
                phases, load, forward_force = e229.aggregate_phase(contacts,
                    decomposition, PHASES, science["support_threshold_bw"], heading)
                e229.save_npz(arm_out / "leg-native.npz", {
                    "normal_load_bw": load,
                    "initial_heading_force_bw": forward_force})
                point_metrics = {"point_contact_phases": phases,
                    "contact_checks": checks,
                    "trajectory": e228.trajectory_metrics(rows, heading,
                        freeze["event"]["velocity_summary_controls"]),
                    "tarsus_support_phases": e228.phase_metrics(
                        e113.native_arrays(native), force_array, rows, PHASES,
                        science["support_threshold_bw"], heading)}
                com_raw, com_summary = com_metrics(force_array, state_array,
                    heading, layout["root_dof_addresses"])
                e229.save_npz(arm_out / "com-native.npz", com_raw)
                point_metrics["com_summary"] = com_summary
                metrics[arm] = point_metrics
                prior.write_receipt(arm_out / "metrics.json", point_metrics)
                arm_result.update(status="complete_healthy", complete_healthy=True,
                    contact_checks=checks)
                arm_ledger["status"] = "complete_healthy"
            except e113.NativeHealthStop as error:
                arm_result.update(status="measured_native_health_stop",
                                  measured_error=repr(error))
                arm_ledger["status"] = "measured_native_health_stop"
            except (TimeoutError, MemoryError) as error:
                arm_result.update(status="resource_limited", error=repr(error))
                arm_ledger["status"] = "resource_limited"
                raise
            except BaseException as error:
                arm_result.update(status="technical_failure", error=repr(error))
                arm_ledger["status"] = "technical_failure"
                raise
            finally:
                attempt(arm + "/raw", lambda arm_out=arm_out,
                        registry=registry, native=native, sink=force_sink,
                        arm_name=arm:
                    save_arm_raw(arm_out, registry, native, sink,
                        on_error=lambda filename, error:
                            record_storage_error(arm_name + "/" + filename, error)))
                if arm == "rf_tibia_hold":
                    def retain_partial(receipt=arm_result, samples=native,
                                       current_registry=registry):
                        first = e228.first_native_divergence(samples,
                            sources["source_native"], source_start=50350)
                        receipt["first_native_difference_sample"] = first
                        receipt["first_native_state_difference_sample_local"] = (
                            first_state_difference(current_registry, e229_reference))
                    attempt(arm + "/partial_native_difference", retain_partial)
                arm_result["native_samples"] = len(native)
                arm_result["state_samples"] = len(registry["native_state"][
                    "native_index"])
                arm_result["force_samples"] = len(force_sink)
                arm_result["last_physics_time_s"] = (float(body.physics_time())
                    if body is not None and np.isfinite(body.physics_time()) else None)
                attempt(arm + "/arm-result.json", lambda target=arm_out,
                        receipt=arm_result: prior.write_receipt(
                            target / "arm-result.json", receipt))
                attempt(arm + "/advance-ledger.json", persist)
            if primary is not None:
                raise primary
            if arm == "sham" and not arm_result["complete_healthy"]:
                result["validity"] = "measured_native_health_stop"
                result["intervention_not_executed_reason"] = "sham_health_stop"
                break
        result["local_comparison"] = e228.assess_local(
            {"trajectory": metrics["sham"]["trajectory"],
             "phases": metrics["sham"]["tarsus_support_phases"]}
            if "sham" in metrics else None,
            {"trajectory": metrics["rf_tibia_hold"]["trajectory"],
             "phases": metrics["rf_tibia_hold"]["tarsus_support_phases"]}
            if "rf_tibia_hold" in metrics else None,
            arm="rf_hold", thresholds=freeze["diagnostic_thresholds"])
        if result["local_comparison"]["available"]:
            result["local_comparison"]["arm"] = "rf_tibia_hold"
            result["local_comparison"]["RF_tibia_command_mechanical_contribution_supported"] = (
                result["local_comparison"][
                    "RF_command_mechanical_contribution_supported"])
        comparison = {"schema": "flylab.m3.e230.rf-tibia-hold.comparison.v1",
            "local_comparison": result["local_comparison"],
            "complete_healthy_by_arm": {name: result["arms"][name][
                "complete_healthy"] for name in result["arms"]},
            "com_summary_by_arm": {name: value["com_summary"] for name, value
                in metrics.items()},
            "point_contact_phases_by_arm": {name: value[
                "point_contact_phases"] for name, value in metrics.items()},
            "tarsus_support_phases_by_arm": {name: value[
                "tarsus_support_phases"] for name, value in metrics.items()}}
        prior.write_receipt(out / "comparison.json", comparison)
        if result["validity"] == "incomplete":
            result["validity"] = ("valid_local_rf_tibia_body_diagnostic"
                if all(result["arms"][name]["complete_healthy"] for name in ARMS)
                else "measured_native_health_stop")
    except (TimeoutError, MemoryError) as error:
        primary = error
        result.update(validity="resource_limited", primary_error=repr(error),
                      traceback=traceback.format_exc())
    except BaseException as error:
        primary = error
        result.update(validity="technical_failure", primary_error=repr(error),
                      traceback=traceback.format_exc())
    finally:
        if restore_hook is not None:
            attempt("observer_cleanup", restore_hook)
        if body is not None:
            attempt("body_cleanup", body.close)
        if thread_limits is not None:
            attempt("cpu_thread_cleanup", thread_limits.restore_original_limits)
        ledger["wall_s"] = time.perf_counter() - began
        for key in ("contact_rows", "jacobian_calls_requested",
                    "jacobian_calls_returned", "decomposition_rows"):
            ledger[key] = sum(arm[key] for arm in ledger["arms"].values())
        ledger["output_bytes_before_final_receipts"] = sum(
            path.stat().st_size for path in out.rglob("*") if path.is_file())
        if ledger["wall_s"] > limits["internal_s"] and primary is None:
            primary = TimeoutError("E230 internal wall cap exceeded")
            result["validity"] = "resource_limited"
        if ledger["output_bytes_before_final_receipts"] > limits[
                "output_bytes"] and primary is None:
            primary = OSError("E230 output cap exceeded")
            result["validity"] = "technical_storage_failure"
        if primary is not None and result["validity"] == (
                "valid_local_rf_tibia_body_diagnostic"):
            result["validity"] = "technical_storage_failure"
        result["execution_status"] = "failed" if primary else "complete"
        if primary is not None:
            result["primary_error"] = repr(primary)
        attempt("result.json", lambda: prior.write_receipt(out / "result.json", result))
        attempt("advance-ledger.json", persist)
        attempt("output-manifest.json", lambda: prior.write_receipt(
            out / "output-manifest.json", replay.manifest(out)))
        if primary is not None and result["validity"] == (
                "valid_local_rf_tibia_body_diagnostic"):
            result.update(validity="technical_storage_failure",
                          execution_status="failed", primary_error=repr(primary))
            attempt("result.json/invalidated", lambda: prior.write_receipt(
                out / "result.json", result))
            attempt("output-manifest.json/invalidated", lambda: prior.write_receipt(
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
