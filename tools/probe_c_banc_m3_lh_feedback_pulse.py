"""E261: one-state, four-branch physical LH target pulse diagnostic.

The pulse is applied at the body call boundary.  The decoder, sensory encoder,
neural graph, and all saved E257 controller parameters remain untouched.
"""

# Scientific and persistence failures are recorded independently.
# ruff: noqa: BLE001
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import probe_c_banc_m3_lh_edge_restoration as e250
from tools import probe_c_banc_m3_lh_timescale_partition as e256
from tools import verify_c_banc_m3_e113_body as e113
from tools import verify_c_banc_m3_lh_partition_body as e257
from tools import verify_c_banc_m3_native_monotone_body as e150
from tools import verify_c_banc_m3_native_timescale_transfer as protocol
from tools import verify_c_banc_m3_neural_current_body as e224
from tools.verify_c_banc_observation import exact

BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
FREEZE_SHA = "fbe7185f735aec9c1a033d1788f0f3b25853d15b5d2b3fc193b2667e0b4d536b"
FREEZE_SCHEMA = "flylab.m3.e261.lh-feedback-pulse.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e261.lh-feedback-pulse.review.v1"
OWNED = ("tools/probe_c_banc_m3_lh_feedback_pulse.py",
         "tests/test_c_banc_lh_feedback_pulse.py")
ARMS = ("sham0", "plus", "minus", "sham1")


class MeasuredStop(Exception):
    """A body-observed stop that does not imply technical instrument validity."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def safe(name: str) -> Path:
    relative = Path(name)
    path = (ROOT / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(ROOT):
        raise ValueError("Repository-relative E261 input required")
    return path


def write_json(path: Path, payload) -> None:
    e224.write_receipt(path, payload)


def gate(freeze_path, freeze_sha, review_path, review_sha, out) -> dict:
    freeze_path, review_path, out = map(Path, (freeze_path, review_path, out))
    if (freeze_sha != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA
            or sha256(review_path) != review_sha):
        raise ValueError("Exact frozen E261 source review required")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    limits = freeze.get("resource_limits", {})
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("owned_files") != list(OWNED)
            or sha256(safe(freeze["design"])) != freeze["design_sha256"]
            or freeze.get("output") != (BASE / "e261-lh-feedback-pulse").relative_to(ROOT).as_posix()
            or freeze.get("arm_order") != list(ARMS)
            or freeze.get("pulse_signs") != dict(zip(ARMS, (0, 1, -1, 0)))
            or freeze.get("prefix_controls") != 55 or freeze.get("fork_control") != 55
            or freeze.get("controls_per_arm") != 20 or freeze.get("pulse_control") != 56
            or freeze.get("pulse_joint") != 19
            or freeze.get("pulse_delta_rad") != .0013821410102253795
            or freeze.get("network_identity") !=
               "4d5bae3386bed2dce2ca8a2add5130449c60a70e3b0d95a51947feeb03c76354"
            or freeze.get("n") != 158706 or freeze.get("declared_edges") != 11584852
            or freeze.get("neural_ticks_per_control") != 20
            or freeze.get("physics_steps_per_control") != 50
            or freeze.get("return_window_controls") != [58, 76]
            or freeze.get("return_tolerance_rad") != 1e-8
            or freeze.get("detection_tolerance_rad") != 1e-8
            or limits.get("internal_s") != 45 or limits.get("finalization_reserve_s") != 8
            or limits.get("whole_constructors") != 1 or limits.get("neural_constructors") != 7
            or limits.get("body_constructors") != 1 or limits.get("in_place_restores") != 4
            or limits.get("scientific_neural_ticks") != 2700
            or limits.get("scientific_physics_steps") != 6750
            or limits.get("original_attempts") != 1 or limits.get("retries") != 0):
        raise ValueError("Frozen E261 scientific/resource contract differs")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("E261 parent source review differs")
    pins = {}
    for group in (freeze["input_sha256"], freeze["reused_source_sha256"],
                  freeze["preservation_sha256"], review["source_sha256"],
                  review.get("additional_input_sha256", {}),
                  review.get("additional_reused_source_sha256", {})):
        for name, digest in group.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E261 pin: " + name)
            pins[name] = digest
    for name, digest in pins.items():
        if len(digest) != 64 or sha256(safe(name)) != digest:
            raise ValueError("Pinned E261 bytes changed: " + name)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned E261 package differs: " + name)
    if out.resolve() != (BASE / "e261-lh-feedback-pulse").resolve() or out.exists():
        raise ValueError("Exact new E261 output required")
    if shutil.disk_usage(out.parent).free < limits["minimum_free_disk_bytes"]:
        raise OSError("E261 output disk reservation unavailable")
    return freeze


def load_inputs(freeze):
    """Read original scientific bytes only after gate has passed."""
    from scipy.sparse import csr_matrix

    def npz(key):
        return e257.read_npz(safe(freeze["paths"][key]))

    csr = npz("baseline_csr")
    baseline = csr_matrix((csr["data"], csr["indices"], csr["indptr"]),
                          shape=(freeze["n"], freeze["n"]), dtype=np.float32)
    baseline.check_format(full_check=True)
    candidate = npz("candidate_csr")
    e257_freeze = json.loads(safe(freeze["paths"]["e257_freeze"]).read_text(
        encoding="utf-8"))
    e256_patch = e257.read_npz(safe(e257_freeze["paths"]["e256_patch"]))
    patch = e256.patch_six(baseline, e256_patch,
                           np.asarray(e257_freeze["preserved_edge_ids"], np.int64))
    for key in ("data", "indices", "indptr"):
        if not e250.prior.same_bits(getattr(patch, key), candidate[key]):
            raise ValueError("E261 candidate CSR differs from frozen six-edge model: " + key)
    graph = protocol.GraphStore.load(safe(freeze["paths"]["graph_bundle"]))
    saved = protocol.StateStore.load(safe(freeze["paths"]["start"]))
    if (graph.hash != freeze["graph_hash"]
            or saved.get("schema") != e257.LHPartitionTrial.schema
            or saved["network"]["model_hash"] != freeze["network_identity"]
            or saved["body"]["modelHash"] != freeze["body_identity"]
            or saved["control_tick"] != 0):
        raise ValueError("E261 original start identity differs")
    selection = json.loads(safe(freeze["paths"]["pulse_selection"]).read_text(
        encoding="utf-8"))
    if (selection.get("pulse_delta_rad") != freeze["pulse_delta_rad"]
            or selection.get("control_tick") != freeze["pulse_control"]
            or selection.get("joint_column") != freeze["pulse_joint"]
            or not 0 < freeze["pulse_delta_rad"] <= .01):
        raise ValueError("E261 pulse selection differs")
    return graph, saved, baseline


def restore_in_place(session, fork: dict) -> None:
    """Restore every checkpointed dynamic component, then verify whole bytes."""
    if fork.get("schema") != e257.LHPartitionTrial.schema:
        raise ValueError("Exact E257 fork required")
    session.body.restore(fork["body"])
    session.network.restore(fork["network"])
    session.adapter.restore(fork["adapter"])
    session.set_cuts(fork["cuts"])
    session.tendon_inputs.clear()
    session.tendon_inputs.update(session.adapter.tendon_inputs)
    session.control_tick = int(fork["control_tick"])
    session.initial_position = np.asarray(fork["initial_position"]).copy()
    session.signed_forward_mm = float(fork["signed_forward_mm"])
    session.wall_s = float(fork["wall_s"])
    session.fault = None
    session.rate = session.network.readout(session.adapter.motor_ids)
    if session.navigation is not None:
        session.navigation.restore(fork["navigation"], time_s=session.control_tick * .005)
    session._clocks()
    exact(fork, session.snapshot(), "E261 in-place whole fork restoration")


def attach_pulse(session, records: list[dict], sign: int, delta: float,
                 *, pulse_control=56, joint=19):
    """Intercept just one body's same-control target call; never change decode."""
    if sign not in (-1, 0, 1) or not 0 < delta <= .01:
        raise ValueError("Frozen E261 pulse sign/magnitude required")
    body = session.body
    previous = body.__dict__.get("step_joint_targets")
    original = body.step_joint_targets

    def step(target, adhesion, **kwargs):
        control = int(session.control_tick) + 1
        source = np.asarray(target)
        if source.shape != (42,) or not np.isfinite(source).all():
            raise ValueError("Nonfinite or malformed decoded target")
        changed = bool(sign and control == pulse_control)
        applied = source.copy() if changed else source
        if changed:
            applied[joint] += sign * delta
            joint_id = int(body.joint_ids[joint])
            low, high = np.asarray(body.m.jnt_range[joint_id], np.float64)
            if not (low <= applied[joint] <= high):
                raise ValueError("E261 pulse exceeds measured joint range")
            actuator = int(body.act_ids[joint])
            if body.m.actuator_ctrllimited[actuator]:
                low, high = body.m.actuator_ctrlrange[actuator]
                if not low <= applied[joint] <= high:
                    raise ValueError("E261 pulse would be clipped by actuator control range")
        record = {"control_tick": control, "original_command": source.copy(),
                  "transmitted_command": np.asarray(applied).copy(),
                  "changed": changed, "body_call_entered": True,
                  "body_call_returned": False,
                  "adhesion": np.asarray(adhesion).copy(),
                  "tendon_inputs": copy.deepcopy(kwargs.get("tendon_inputs")),
                  "kwargs_keys": sorted(kwargs)}
        records.append(record)
        result = original(applied, adhesion, **kwargs)
        record["body_call_returned"] = True
        return result

    body.step_joint_targets = step

    def detach():
        if previous is None:
            del body.step_joint_targets
        else:
            body.step_joint_targets = previous

    return detach


def fork_geometry(body, joint=19, leg=2):
    measured = body.knee_kinematics(include_direction=True)
    if len(body._knee_layout) != 6:
        raise ValueError("Six physical knee geometries required")
    ids, knee_column = body._knee_layout[leg]
    column = int(body.qvel_ids[joint])
    jac = np.zeros((3, 3), np.float64)
    for i, body_id in enumerate(ids):
        full = np.zeros((3, body.m.nv))
        body.mj.mj_jacBody(body.m, body.d, full, None, int(body_id))
        jac[i] = full[:, column]
    if (column != int(knee_column) or not np.isfinite(jac).all()
            or not np.isfinite(measured["d_flexion_dq"][leg])):
        raise ValueError("LH knee coordinate or geometry invalid")
    return {"body_ids": np.asarray(ids, np.int64),
            "positions_xyz": np.asarray(body.d.xpos[ids], np.float64).copy(),
            "jacobian_linear_q19": jac,
            "d_flexion_dq": np.asarray(measured["d_flexion_dq"], np.float64).copy(),
            "flexion_rad": np.asarray(measured["flexion_rad"], np.float64).copy(),
            "velocity_rad_s": np.asarray(measured["velocity_rad_s"], np.float64).copy(),
            "joint_name": str(body.joint_names[joint]),
            "joint_id": int(body.joint_ids[joint]),
            "qpos_id": int(body.qpos_ids[joint]), "qvel_id": column,
            "actuator_id": int(body.act_ids[joint]),
            "actuator_control_limited": bool(body.m.actuator_ctrllimited[body.act_ids[joint]]),
            "actuator_control_range": np.asarray(body.m.actuator_ctrlrange[body.act_ids[joint]],
                                                np.float64).copy(),
            "joint_range": np.asarray(body.m.jnt_range[body.joint_ids[joint]],
                                      np.float64).copy()}


def capture_geometry(body, leg=2):
    value = body.knee_kinematics(include_direction=True)
    return {"flexion_rad": float(value["flexion_rad"][leg]),
            "velocity_rad_s": float(value["velocity_rad_s"][leg]),
            "d_flexion_dq": float(value["d_flexion_dq"][leg])}


def _finite_records(records, fields):
    for record in records:
        for key in fields:
            if key not in record:
                return False
            try:
                value = np.asarray(record[key], np.float64)
            except (TypeError, ValueError):
                return False
            if value.size == 0 or not np.isfinite(value).all():
                return False
    return True


def diagnostic_health(rows, native, clock, start_control, end_control, fault,
                      *, controls=None, activity=None, core=None, recruitment=None,
                      geometry=None, sensory=None, transfers=None):
    times = np.asarray([row["time_s"] for row in rows], np.float64)
    native_time = np.asarray(native["time_s"], np.float64)
    expected = np.arange(start_control, start_control + len(rows)) * .005
    expected_native = np.arange(start_control * 50 + 1,
                                start_control * 50 + 1 + len(native_time)) * .0001
    clocks = bool(clock.clock_valid and len(times) and np.isfinite(times).all()
                  and np.isfinite(native_time).all()
                  and np.allclose(times, expected, rtol=0, atol=2e-8)
                  and np.allclose(native_time, expected_native, rtol=0, atol=2e-8))
    upright = np.asarray(native["upright"])
    finite = bool(np.isfinite(upright).all()
                  and np.isfinite(native["normal_load_bw"]).all()
                  and all(not row.get("fault") for row in rows)
                  and _finite_records(rows, ("qpos", "qvel", "ctrl",
                      "joint_angles_rad", "joint_targets_rad",
                      "joint_tracking_error_rad", "motor_rates"))
                  and all(_finite_records([row.get("body", {})],
                      ("position", "basis")) for row in rows))
    for records, fields in (
            (controls, ("motor_rate32", "activation", "command", "tendon",
                        "source_drive32", "bridge_drive32")),
            (activity, ("rate", "drive", "adaptation")),
            (core, ("rate", "drive", "adaptation", "physics_time_s")),
            (recruitment, ("bank", "activation", "original_activation", "motor_rate",
                           "time_s")),
            (geometry, ("flexion_rad", "velocity_rad_s", "d_flexion_dq")),
            (transfers, ("original_command", "transmitted_command", "adhesion"))):
        if records is not None:
            finite = finite and _finite_records(records, fields)
    if sensory is not None:
        finite = bool(finite and len(sensory) == end_control - start_control
                      and all(row.get("complete") is True for row in sensory)
                      and _finite_records(sensory, ("old_filtered", "new_filtered",
                         "last_features", "old_velocity_lowpass",
                         "new_velocity_lowpass", "returned_source_drive")))
    minimum = float(upright.min()) if len(upright) and finite else None
    nonfoot = int(np.count_nonzero(native["nonfoot_contact"]))
    complete = len(rows) == end_control - start_control + 1 and len(native_time) == (
        end_control - start_control) * 50
    return {"complete": complete, "clock_exact": clocks, "finite": finite,
            "minimum_native_upright": minimum, "nonfoot_contact_count": nonfoot,
            "session_or_body_fault": bool(fault),
            "healthy": bool(complete and clocks and finite and not fault
                            and nonfoot == 0 and minimum is not None and minimum >= .9)}


def contrast(plus, minus, plus_events, minus_events, *, tolerance=1e-8):
    """Keep detection separate from the decoder-only delayed return."""
    if not (len(plus) == len(minus) == 21 and len(plus_events) == len(minus_events) == 20):
        return {"classification": "invalid_or_incomplete", "H261_supported": False}
    ticks = np.asarray([row["control_tick"] for row in plus], np.int64)
    if not np.array_equal(ticks, np.arange(55, 76)) or not np.array_equal(
            ticks, [row["control_tick"] for row in minus]):
        return {"classification": "invalid_or_incomplete", "H261_supported": False}
    p, m = plus_events[0], minus_events[0]
    if (p["control_tick"] != 56 or m["control_tick"] != 56
            or not p["changed"] or not m["changed"]):
        return {"classification": "invalid_or_incomplete", "H261_supported": False}
    applied = float(p["transmitted_command"][19] - m["transmitted_command"][19])
    physical = float(plus[1]["joint_angles_rad"][19] -
                     minus[1]["joint_angles_rad"][19])
    if (not e250.prior.same_bits(p["original_command"], m["original_command"])
            or any(not event["body_call_returned"] for event in (*plus_events, *minus_events))):
        return {"classification": "invalid_or_incomplete", "H261_supported": False}
    original = np.asarray([row["original_command"][19] for row in plus_events]) - np.asarray(
        [row["original_command"][19] for row in minus_events])
    response = float(np.mean(original[2:20]))  # Controls 58..75.
    detected = bool(np.isfinite(applied) and np.isfinite(physical)
                    and applied > tolerance and physical > tolerance)
    if not np.isfinite(response):
        classification = "invalid_or_incomplete"
    elif not detected:
        classification = "inconclusive_undetected_pulse"
    elif response > tolerance:
        classification = "supported_positive_return"
    elif response < -tolerance:
        classification = "refuted_compensatory_return"
    else:
        classification = "inconclusive_below_resolution"
    return {"classification": classification,
            "H261_supported": classification == "supported_positive_return",
            "physical_transmitted_plus_minus_q19_rad": applied,
            "physical_actual_plus_minus_q19_rad": physical,
            "decoder_return_mean_plus_minus_q19_rad": response,
            "decoder_return_controls": [58, 75], "detected": detected}


def compare_reference(segment, rows, controls, activity, core, native, recruitment,
                      reference: Path, begin: int, end: int):
    """Compare all E257 fields that existed in the saved candidate arm."""
    if len(rows) != end - begin + 1 or len(native) != (end - begin) * 50:
        raise ValueError("Incomplete E261 reference segment")
    saved = json.loads((reference / "trace.json").read_text(encoding="utf-8"))["samples"]
    exact(rows, saved[begin:end + 1], segment + " control/body trace")
    native_saved = e257.read_npz(reference / "native-contact.npz")
    current = e113.native_arrays(native)
    for key, value in current.items():
        if not e250.prior.same_bits(value, native_saved[key][begin * 50:end * 50]):
            raise AssertionError(segment + " native differs: " + key)
    for name, values, keys in (
            ("control-raw.npz", controls,
             ("control_tick", "network_tick", "physics_time_s", "motor_rate32",
              "activation", "command", "tendon")),
            ("activity.npz", activity, ("rate", "drive", "adaptation")),
            ("core-activity.npz", core,
             ("control_tick", "network_tick", "physics_time_s", "rate", "drive", "adaptation"))):
        saved_npz = e257.read_npz(reference / name)
        for key in keys:
            got = np.asarray([row[key] for row in values])
            if not e250.prior.same_bits(got, saved_npz[key][begin:end + 1]):
                raise AssertionError(segment + " " + name + ":" + key + " differs")
    saved_recruitment = e257.read_npz(reference / "native-recruitment-state.npz")
    for key in ("control_tick", "network_tick", "complete",
                "bank", "activation", "original_activation", "motor_rate"):
        # e150 writes optional motor_rate as float64, unlike live float32 readout.
        got = np.asarray([row[key] for row in recruitment],
                         dtype=np.float64 if key in ("bank", "activation",
                                                   "original_activation", "motor_rate") else None)
        if not e250.prior.same_bits(got, saved_recruitment[key][begin:end + 1]):
            raise AssertionError(segment + " recruitment:" + key + " differs")
    if not e250.prior.same_bits(np.asarray([row["time_s"] for row in recruitment]),
                                saved_recruitment["physics_time_s"][begin:end + 1]):
        raise AssertionError(segment + " recruitment:physics_time_s differs")


def _stack(path, records, keys):
    np.savez_compressed(path, **{key: np.asarray([row[key] for row in records])
                                for key in keys})


def _write_transfer(path, events):
    keys = ("control_tick", "original_command", "transmitted_command", "changed",
            "body_call_entered", "body_call_returned", "adhesion")
    _stack(path, events, keys)
    with (path.parent / "pulse-events.jsonl").open("x", encoding="utf-8") as stream:
        for row in events:
            cleaned, _ = e224.finite_receipt(protocol.jsonable(row))
            stream.write(json.dumps(cleaned, allow_nan=False) + "\n")


def decoded_target_series(controls, events, joint=19):
    """Held fork command plus same-control decoder originals, never pulsed body q."""
    if len(controls) != len(events) + 1:
        raise ValueError("E261 decoded control/event alignment differs")
    return np.asarray([controls[0]["command"][joint],
        *(event["original_command"][joint] for event in events)], np.float64)


def _capture(session, rows, controls, activity, core, recruitment, geometry, freeze):
    row = protocol.capture_control(session)
    rows.append(row)
    controls.append(e257.capture_control_raw(session, row))
    e150.make_observer(recruitment, original=lambda *_: {})(session,
                                                    np.empty(0, np.int64))
    activity.append(e257.capture_activity(session, freeze["observed_ids"]))
    core.append(e257.capture_core_activity(session, freeze["core_ids"]))
    geometry.append(capture_geometry(session.body, freeze["leg_index"]))
    drive = session.network.drive.detach().cpu().numpy()
    if drive.shape != (freeze["n"],) or drive.dtype != np.float32:
        raise ValueError("E261 complete original f32 neural drive required")
    controls[-1]["full_drive_sha256"] = hashlib.sha256(drive.tobytes()).hexdigest()
    controls[-1]["source_drive32"] = drive[np.asarray(freeze["source_ids"], np.int64)].copy()
    controls[-1]["bridge_drive32"] = drive[np.asarray(freeze["bridge_ids"], np.int64)].copy()
    return row


def run_segment(session, name, start, end, path, freeze, ledger, began,
                budget, *, pulse_sign=0, reference=None):
    """One already-restored segment; keep native/partial costs on any failure."""
    rows, controls, activity, core, native, recruitment, sensory, geometry, transfers = (
        [], [], [], [], [], [], [], [], [])
    clock = e113.AdvanceLedger()
    detach_native = detach_sensory = detach_pulse = journal = None
    primary = measured = None
    secondary = []
    entry = {"requested_controls": end - start, "entered_controls": 0,
             "completed_controls": 0, "native_samples": 0,
             "status": "requested", "clock_events": []}
    ledger["segments"][name] = entry
    path.mkdir(exist_ok=False)

    def absorb(stage, error):
        nonlocal primary
        secondary.append({"stage": stage, "error": repr(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E261 " + stage + ": " + repr(error))

    try:
        budget()
        if session.control_tick != start:
            raise ValueError("E261 segment starts at wrong control")
        protocol.StateStore.save(path / "observed-start", session.snapshot())
        detach_native = e113.install_native_observer(session.body, native)
        detach_sensory = e257.attach_sensory_observer(session, sensory, freeze["source_ids"])
        detach_pulse = attach_pulse(session, transfers, pulse_sign,
                                    freeze["pulse_delta_rad"],
                                    pulse_control=freeze["pulse_control"],
                                    joint=freeze["pulse_joint"])
        journal = e224.FiniteControlJournal(path / "control-journal.jsonl")
        _capture(session, rows, controls, activity, core, recruitment, geometry, freeze)
        with (path / "trace.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
            clean, _ = e224.finite_receipt(protocol.jsonable(rows[-1]))
            stream.write(json.dumps(clean, allow_nan=False) + "\n")
            for control in range(start + 1, end + 1):
                budget()
                entry["entered_controls"] += 1
                e257.write_json(path / "segment-ledger.json", entry)
                before = (session.control_tick, session.network.tick,
                          session.body.physics_time(), len(native))
                try:
                    event = protocol.advance_control(session, name, control, clock, journal)
                except e113.NativeHealthStop as error:
                    measured = {"control": control, "message": str(error)}
                    break
                try:
                    _capture(session, rows, controls, activity, core, recruitment,
                             geometry, freeze)
                    protocol.record_readout(journal, name, control, event, complete=True)
                except BaseException as error:
                    try:
                        protocol.record_readout(journal, name, control, event,
                                                complete=False, error=error)
                    except BaseException as other:
                        error.add_note("Secondary E261 journal: " + repr(other))
                    raise
                clean, _ = e224.finite_receipt(protocol.jsonable(rows[-1]))
                stream.write(json.dumps(clean, allow_nan=False) + "\n")
                if (session.control_tick != before[0] + 1
                        or session.network.tick != before[1] + 20
                        or abs(session.body.physics_time() - before[2] - .005) > 2e-8
                        or len(native) != before[3] + 50):
                    raise ValueError("Incomplete E261 control/native interval")
                entry["completed_controls"] += 1
                if session.fault or session.body.fault:
                    measured = {"control": control, "message": "measured session/body fault"}
                    break
        entry["status"] = "measured_health_stop" if measured else "scientific_complete"
    except BaseException as error:
        absorb("scientific", error)
        entry["status"] = "technical_or_resource_failure"
    finally:
        entry["native_samples"] = len(native)
        entry["clock_events"] = clock.events
        entry["measured_stop"] = measured
        for label, callback in (("pulse detach", detach_pulse),
                                ("sensory detach", detach_sensory),
                                ("native detach", detach_native),
                                ("journal close", journal.close if journal else None)):
            if callback:
                try:
                    callback()
                except BaseException as error:
                    absorb(label, error)
        # Each file is attempted independently, including when a control failed.
        selection = {"source_rows": freeze["source_ids"],
                     "bridge_rows": freeze["bridge_ids"],
                     "reachable_motor_rows": freeze["observed_ids"][27:],
                     "observed_ids": freeze["observed_ids"]}
        for failed in e257.save_arm_raw(path, rows, controls, activity, native,
                recruitment, sensory, clock, selection, freeze["graph_hash"],
                session.network.identity, session.body.model_hash,
                session.adapter.identity, session.p.state_dict()):
            absorb("raw " + failed["artifact"], RuntimeError(failed["error"]))
        for label, callback in (
                ("core-activity", lambda: e257.save_core_activity(
                    path / "core-activity.npz", core, freeze["core_ids"])),
                ("command-transfer", lambda: _write_transfer(
                    path / "command-transfer.npz", transfers)),
                ("geometry", lambda: _stack(path / "geometry.npz", geometry,
                    ("flexion_rad", "velocity_rad_s", "d_flexion_dq")))):
            try:
                callback()
            except BaseException as error:
                absorb("raw " + label, error)
        try:
            if measured is None:
                protocol.StateStore.save(path / "final", session.snapshot())
            else:
                protocol.StateStore.save(path / "partial-components", {
                    "body": session.body.snapshot(),
                    "network": session.network.snapshot(),
                    "adapter": session.adapter.snapshot(),
                    "control_tick": int(session.control_tick)})
                entry["whole_final_unavailable_due_to_measured_stop"] = True
        except BaseException as error:
            absorb("raw partial or final state", error)
        entry["secondary_errors"] = secondary
        try:
            e257.write_json(path / "segment-ledger.json", entry)
        except BaseException as error:
            absorb("segment ledger", error)
    native_arrays = e113.native_arrays(native)
    health = diagnostic_health(rows, native_arrays, clock, start, end,
                               bool(session.fault or session.body.fault),
                               controls=controls, activity=activity, core=core,
                               recruitment=recruitment, geometry=geometry,
                               sensory=sensory, transfers=transfers)
    if primary is None and measured is None and health["healthy"] and reference:
        try:
            compare_reference(name, rows, controls, activity, core, native,
                              recruitment, reference, start, end)
            entry["reference_exact"] = True
        except BaseException as error:
            absorb("reference exact", error)
    status = ("invalid_or_incomplete" if primary is not None else
              "measured_native_health_stop" if measured else
              "complete_healthy" if health["healthy"] else "invalid_or_incomplete")
    entry["status"] = status
    try:
        _stack(path / "drive-observation.npz", controls,
               ("source_drive32", "bridge_drive32", "full_drive_sha256"))
        e257.write_json(path / "segment-ledger.json", entry)
    except BaseException as error:
        absorb("drive observation or final ledger", error)
        status = "invalid_or_incomplete"
        entry["status"] = status
    return {"status": status, "health": health, "rows": rows,
            "controls": controls, "activity": activity, "core": core,
            "native": native, "recruitment": recruitment, "geometry": geometry,
            "transfers": transfers, "primary": primary, "measured": measured}


def output_manifest(out, limit):
    files, total = {}, 0
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in ("output-manifest.json", "result.json"):
            size = path.stat().st_size
            total += size
            if total > limit:
                raise RuntimeError("E261 output byte cap exceeded")
            files[path.relative_to(out).as_posix()] = {"sha256": sha256(path),
                                                       "bytes": size}
    return {"schema": "flylab.m3.e261.output-manifest.v1", "bytes": total,
            "files": files}


def finalize(out, ledger, result, errors, primary, began, limit,
             *, writer=write_json):
    """Publish last verdict only after durable work/errors/manifest receipts."""
    def absorb(stage, error):
        nonlocal primary
        errors.append({"stage": stage, "type": type(error).__name__,
                       "message": str(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E261 " + stage + ": " + repr(error))

    def revoke():
        result["validity"] = "invalid_or_incomplete"
        result["H261"] = {**result.get("H261", {}), "H261_supported": False,
                          "classification": "invalid_or_incomplete"}

    def revoke_contrast():
        canonical = out / "contrast.json"
        if not canonical.exists():
            return
        try:
            payload = json.loads(canonical.read_text(encoding="utf-8"))
            payload["H261_supported"] = False
            payload["classification"] = "unvalidated_after_failure"
            writer(canonical, payload)
        except BaseException as error:
            # A persistent writer failure must not leave canonical support true.
            try:
                canonical.replace(out / "contrast-unvalidated.json")
            except BaseException:
                canonical.unlink(missing_ok=True)
            errors.append({"stage": "contrast verdict withdrawal",
                           "type": type(error).__name__, "message": str(error)})

    result["finalization_pending"] = True
    if primary is not None:
        revoke()
        revoke_contrast()
    ledger["elapsed_s"] = time.perf_counter() - began
    for name, payload in (("advance-ledger.json", ledger), ("errors.json", errors)):
        try:
            writer(out / name, payload)
        except BaseException as error:
            absorb(name, error)
    try:
        writer(out / "output-manifest.json", output_manifest(out, limit))
    except BaseException as error:
        absorb("output-manifest.json", error)
    elapsed = time.perf_counter() - began
    if elapsed >= 45:
        absorb("internal_wall_s", TimeoutError("E261 45s internal deadline"))
    if primary is not None:
        revoke()
        revoke_contrast()
    result["elapsed_s"] = elapsed
    result["primary_error"] = (None if primary is None else
        {"type": type(primary).__name__, "message": str(primary),
         "notes": list(getattr(primary, "__notes__", []))})
    result["secondary_errors"] = errors.copy()
    result["finalization_pending"] = False
    try:
        writer(out / "result.json", result)
    except BaseException as error:
        absorb("result.json", error)
        (out / "result.json").unlink(missing_ok=True)
    if primary is None and time.perf_counter() - began >= 45:
        absorb("post_result_internal_wall_s", TimeoutError("E261 publication deadline"))
    if primary is not None:
        revoke()
        revoke_contrast()
        result["primary_error"] = {"type": type(primary).__name__, "message": str(primary)}
        result["secondary_errors"] = errors.copy()
        for name, payload in (("errors.json", errors),
                              ("advance-ledger.json", ledger),
                              ("result.json", result)):
            try:
                writer(out / name, payload)
            except BaseException:
                if name == "result.json":
                    (out / name).unlink(missing_ok=True)
                else:
                    (out / "output-manifest.json").unlink(missing_ok=True)
        if (out / "output-manifest.json").exists():
            try:
                writer(out / "output-manifest.json", output_manifest(out, limit))
            except BaseException:
                (out / "output-manifest.json").unlink(missing_ok=True)
    return primary


def execute(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze = gate(freeze_path, freeze_sha, review_path, review_sha, out)
    out.mkdir(exist_ok=False)
    ledger = {"schema": "flylab.m3.e261.lh-feedback-pulse.ledger.v1",
              "original_attempts": 1, "whole_constructors_requested": 0,
              "in_place_restores_requested": 0, "in_place_restores_completed": 0,
              "construction": [], "body_construction": [], "segments": {},
              "fit_calls": 0, "solver_calls": 0, "current_probes": 0,
              "candidate_adopted": False, "F6_exposed": False,
              "whole_goal_complete": False}
    result = {"schema": "flylab.m3.e261.lh-feedback-pulse.result.v1",
              "validity": "incomplete", "H261": {"H261_supported": False,
              "classification": "not_evaluated"}, "candidate_adopted": False,
              "F6_exposed": False, "whole_goal_complete": False,
              "F3_F5_10s_evaluated": False, "finalization_pending": True}
    primary, session, thread_limits = None, None, None
    holder, errors = {}, []

    def persist():
        write_json(out / "advance-ledger.json", ledger)

    def budget():
        if time.perf_counter() - began >= 37:
            raise TimeoutError("E261 8s finalization reserve reached")
        import psutil
        peak = psutil.Process().memory_info()
        resident = getattr(peak, "peak_wset", peak.rss)
        ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0), resident)
        if resident > freeze["resource_limits"]["process_memory_bytes"]:
            raise MemoryError("E261 host cap exceeded")
        import cupy as cp
        free_now, _ = cp.cuda.runtime.memGetInfo()
        added = max(0, ledger.get("gpu_free_before_bytes", free_now) - free_now)
        ledger["peak_added_gpu_bytes"] = max(ledger.get("peak_added_gpu_bytes", 0), added)
        if added > freeze["resource_limits"]["additional_gpu_memory_bytes"]:
            raise MemoryError("E261 added CUDA memory cap exceeded")

    try:
        write_json(out / "source-review-snapshot.json",
                   json.loads(Path(review_path).read_text(encoding="utf-8")))
        from tools.fit_c_banc_m3_joint_adaptive_recruitment import one_cpu_thread
        thread_limits, ledger["threading"] = one_cpu_thread()
        import cupy as cp
        before_gpu, _ = cp.cuda.runtime.memGetInfo()
        if before_gpu < freeze["resource_limits"]["minimum_free_gpu_bytes"]:
            raise MemoryError("E261 GPU free memory below reservation")
        ledger["gpu_free_before_bytes"] = before_gpu
        graph, saved, baseline = load_inputs(freeze)
        ledger["whole_constructors_requested"] = 1
        persist()
        session = e257.construction_with_body_count("E261 candidate", lambda:
            e257.LHPartitionTrial.from_checkpoint(graph, saved, baseline),
            ledger, persist, lambda value: holder.setdefault("session", value))
        ledger["whole_constructors_completed"] = 1
        exact(saved, session.snapshot(), "E261 candidate initial whole")
        if (session.network.identity != freeze["network_identity"]
                or session.body.model_hash != freeze["body_identity"]
                or session.network.n != freeze["n"] or session.network.device != "cuda"
                or session.network.cuda_implementation != "packed"
                or session.network.interval_steps != 20):
            raise ValueError("E261 actual packed whole backend differs")
        ledger["packed_csr"] = e250.packed_csr_receipt(session.network,
            e256.patch_six(baseline, saved["lh_partition_patch"]["edge_patch"],
                np.asarray(saved["lh_partition_patch"]["preserved_edge_ids"], np.int64)))
        protocol.StateStore.save(out / "observed-start", session.snapshot())
        reference = safe(freeze["paths"]["reference"])
        prefix = run_segment(session, "prefix", 0, 55, out / "prefix", freeze,
                             ledger, began, budget, reference=reference)
        if prefix["primary"] is not None:
            raise prefix["primary"]
        if prefix["status"] != "complete_healthy":
            result["prefix"] = prefix["status"]
            if prefix["status"] == "measured_native_health_stop":
                raise MeasuredStop("E261 observed prefix health stop")
            raise ValueError("E261 prefix observation/clock incomplete")
        fork = session.snapshot()
        protocol.StateStore.save(out / "fork", fork)
        geom = fork_geometry(session.body, freeze["pulse_joint"], freeze["leg_index"])
        np.savez_compressed(out / "fork-geometry.npz", **{key: value for key, value in geom.items()
            if not isinstance(value, str)})
        write_json(out / "fork-geometry.json", {key: value for key, value in geom.items()
                   if isinstance(value, str)})
        branches = {}
        (out / "arms").mkdir(exist_ok=False)
        for name in ARMS:
            budget()
            ledger["in_place_restores_requested"] += 1
            persist()
            restore_in_place(session, fork)
            ledger["in_place_restores_completed"] += 1
            branch = run_segment(session, name, 55, 75, out / "arms" / name,
                                 freeze, ledger, began, budget,
                                 pulse_sign=freeze["pulse_signs"][name],
                                 reference=reference if name.startswith("sham") else None)
            branches[name] = branch
            result[name] = branch["status"]
            if branch["primary"] is not None:
                raise branch["primary"]
            if branch["status"] != "complete_healthy":
                result["H261"]["classification"] = "inconclusive_health_stop"
                if branch["status"] == "measured_native_health_stop":
                    raise MeasuredStop("E261 observed branch health stop: " + name)
                raise ValueError("E261 branch observation/clock incomplete: " + name)
        if len(branches) == 4:
            for name in ARMS:
                if branches[name]["status"] != "complete_healthy":
                    raise ValueError("E261 complete healthy four-arm comparison required")
            exact(branches["sham0"]["rows"], branches["sham1"]["rows"],
                  "E261 sham complete rows")
            left = protocol.StateStore.load(out / "arms/sham0/final")
            right = protocol.StateStore.load(out / "arms/sham1/final")
            exact(left, right, "E261 sham final whole")
            for name in ("plus", "minus"):
                events = branches[name]["transfers"]
                if len(events) != 20 or sum(bool(e["changed"]) for e in events) != 1:
                    raise ValueError("E261 pulse was not isolated to control56")
                for event in events:
                    source = event["original_command"]
                    applied = event["transmitted_command"]
                    changed = np.flatnonzero([source[i].tobytes() != applied[i].tobytes()
                                              for i in range(42)])
                    expected = np.array([freeze["pulse_joint"]]) if event["control_tick"] == 56 else np.array([], int)
                    if not np.array_equal(changed, expected):
                        raise ValueError("E261 pulse leaked outside sole joint/control")
                exact(branches[name]["activity"][1], branches["sham0"]["activity"][1],
                      "E261 first pulse-control observed neural rates")
                if not e250.prior.same_bits(events[0]["original_command"],
                                            branches["sham0"]["transfers"][0]["original_command"]):
                    raise AssertionError("E261 first decoder output differs before pulse")
            result["H261"] = contrast(branches["plus"]["rows"],
                branches["minus"]["rows"], branches["plus"]["transfers"],
                branches["minus"]["transfers"])
            if result["H261"]["classification"] == "invalid_or_incomplete":
                raise ValueError("E261 contrast pulse or readout incomplete")
            result["validity"] = "valid_short_closed_loop_diagnostic"
            np.savez_compressed(out / "contrast.npz",
                controls=np.arange(55, 76, dtype=np.int64),
                decoded_q19_plus=decoded_target_series(
                    branches["plus"]["controls"], branches["plus"]["transfers"]),
                decoded_q19_minus=decoded_target_series(
                    branches["minus"]["controls"], branches["minus"]["transfers"]),
                transmitted_q19_plus=np.asarray([
                    row["command"][19] for row in branches["plus"]["controls"]]),
                transmitted_q19_minus=np.asarray([
                    row["command"][19] for row in branches["minus"]["controls"]]),
                flexion_plus=np.asarray([row["flexion_rad"] for row in branches["plus"]["geometry"]]),
                flexion_minus=np.asarray([row["flexion_rad"] for row in branches["minus"]["geometry"]]))
            write_json(out / "contrast.json", result["H261"])
    except MeasuredStop as stop:
        result["validity"] = "measured_native_health_stop"
        result["H261"] = {**result["H261"], "H261_supported": False,
                          "classification": "inconclusive_health_stop"}
        result["measured_stop"] = str(stop)
    except BaseException as error:
        primary = error
        errors.append({"stage": "execute", "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
    finally:
        try:
            session = e257.recover_transferred_owner(session, holder, ledger)
        except BaseException as error:
            errors.append({"stage": "construction ownership", "error": repr(error)})
            primary = primary or error
            session = session or holder.get("session")
        if session is not None:
            try:
                session.close()
            except BaseException as error:
                errors.append({"stage": "session close", "error": repr(error)})
                primary = primary or error
        if thread_limits is not None:
            try:
                thread_limits.restore_original_limits()
            except BaseException as error:
                errors.append({"stage": "thread close", "error": repr(error)})
                primary = primary or error
        ledger["whole_constructors_completed"] = int(
            ledger.get("whole_constructors_completed", 0))
        ledger["setup_neural_executed_ticks"] = sum(
            row.get("known_completed_setup_steps", 0) for row in ledger["construction"])
        ledger["setup_neural_capture_ticks"] = sum(
            row.get("recorded_capture_setup_steps", 0) for row in ledger["construction"])
        ledger["setup_physics_steps_known"] = sum(
            row.get("known_completed_settling_steps", 0)
            for row in ledger["body_construction"])
        events = [event for segment in ledger["segments"].values()
                  for event in segment["clock_events"]]
        ledger["scientific_neural_ticks_known"] = sum(
            max(0, int(event["neural_tick_after"] - event["neural_tick_before"]))
            for event in events if event.get("neural_tick_after") is not None)
        ledger["scientific_physics_steps_known"] = sum(
            max(0, round(event["actual_physical_s"] / .0001))
            for event in events if event.get("actual_physical_s") is not None)
        ledger["unknown_scientific_neural_intervals"] = sum(
            event.get("neural_tick_after") is None for event in events)
        ledger["unknown_scientific_physics_intervals"] = sum(
            event.get("actual_physical_s") is None for event in events)
        if (primary is None and result["validity"] in
                ("valid_short_closed_loop_diagnostic", "measured_native_health_stop")
                and (ledger["whole_constructors_completed"] != 1
                     or len(ledger["construction"]) != 1
                     or ledger["setup_neural_executed_ticks"] != 280
                     or ledger["setup_neural_capture_ticks"] != 140
                     or ledger["setup_physics_steps_known"] != 2000
                     or (result["validity"] == "valid_short_closed_loop_diagnostic"
                         and (ledger["scientific_neural_ticks_known"] != 2700
                              or ledger["scientific_physics_steps_known"] != 6750
                              or ledger["in_place_restores_completed"] != 4)))):
            primary = ValueError("E261 actual setup or scientific work differs")
        primary = finalize(out, ledger, result, errors, primary, began,
                           freeze["resource_limits"]["output_bytes"])
    return 0 if primary is None and result["validity"] in (
        "valid_short_closed_loop_diagnostic", "measured_native_health_stop") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    return execute(args.freeze, args.freeze_sha256, args.review,
                   args.review_sha256, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
