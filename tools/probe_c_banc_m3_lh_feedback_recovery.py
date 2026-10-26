"""E262: resume the E261 LH physical pulse from its verified control-55 fork."""

# Broad catches retain first science failures through independent raw and cleanup writes.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import probe_c_banc_m3_lh_edge_restoration as e250
from tools import probe_c_banc_m3_lh_feedback_pulse as e261
from tools import probe_c_banc_m3_lh_timescale_partition as e256
from tools import verify_c_banc_m3_lh_partition_body as e257
from tools import verify_c_banc_m3_native_timescale_transfer as protocol
from tools.verify_c_banc_observation import exact

BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
FREEZE_SHA = "fb9be645287ba98ab4757ed4538a285a7dab04099ea285ab19d24c3d0a85f079"
FREEZE_SCHEMA = "flylab.m3.e262.lh-feedback-recovery.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e262.lh-feedback-recovery.review.v1"
OWNED = ("tools/probe_c_banc_m3_lh_feedback_recovery.py",
         "tests/test_c_banc_lh_feedback_recovery.py")
ARMS = ("sham0", "plus", "minus", "sham1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def safe(name: str) -> Path:
    rel = Path(name)
    path = (ROOT / rel).resolve()
    if rel.is_absolute() or ".." in rel.parts or not path.is_relative_to(ROOT):
        raise ValueError("Repository-relative E262 input required")
    return path


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    freeze_path, review_path, output = map(Path, (freeze_path, review_path, output))
    if (freeze_sha != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA
            or sha256(review_path) != review_sha):
        raise ValueError("Exact E262 freeze/review bytes required")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    limits = freeze.get("resource_limits", {})
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("owned_files") != list(OWNED)
            or sha256(safe(freeze["design"])) != freeze["design_sha256"]
            or freeze.get("output") != (BASE / "e262-lh-feedback-recovery").relative_to(ROOT).as_posix()
            or freeze.get("paths", {}).get("start") !=
               "verification/banc-natural-gait-m3-20260919/e261-lh-feedback-pulse/fork"
            or freeze.get("prefix_controls") != 0 or freeze.get("fork_control") != 55
            or freeze.get("controls_per_arm") != 20 or freeze.get("arm_order") != list(ARMS)
            or freeze.get("pulse_signs") != dict(zip(ARMS, (0, 1, -1, 0)))
            or freeze.get("pulse_control") != 56 or freeze.get("pulse_joint") != 19
            or freeze.get("pulse_delta_rad") != .0013821410102253795
            or freeze.get("return_window_controls") != [58, 76]
            or freeze.get("return_tolerance_rad") != 1e-8
            or freeze.get("detection_tolerance_rad") != 1e-8
            or freeze.get("n") != 158706 or freeze.get("declared_edges") != 11584852
            or freeze.get("neural_ticks_per_control") != 20
            or freeze.get("physics_steps_per_control") != 50
            or limits.get("internal_s") != 45 or limits.get("finalization_reserve_s") != 8
            or limits.get("whole_constructors") != 1 or limits.get("neural_constructors") != 7
            or limits.get("setup_neural_executed_ticks") != 280
            or limits.get("setup_neural_capture_ticks") != 140
            or limits.get("body_constructors") != 1 or limits.get("setup_physics_steps") != 2000
            or limits.get("scientific_neural_ticks") != 1600
            or limits.get("scientific_physics_steps") != 4000
            or limits.get("in_place_restores") != 4 or limits.get("original_attempts") != 1
            or limits.get("retries") != 0):
        raise ValueError("Frozen E262 science/resource contract differs")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("E262 parent source review differs")
    pins = {}
    for group in (freeze["input_sha256"], freeze["reused_source_sha256"],
                  freeze["preservation_sha256"], review["source_sha256"],
                  review.get("additional_input_sha256", {}),
                  review.get("additional_reused_source_sha256", {})):
        for name, digest in group.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E262 pin " + name)
            pins[name] = digest
    for name, digest in pins.items():
        if len(digest) != 64 or sha256(safe(name)) != digest:
            raise ValueError("Pinned E262 input changed " + name)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned E262 package changed " + name)
    if output.resolve() != (BASE / "e262-lh-feedback-recovery").resolve() or output.exists():
        raise ValueError("Exact new E262 output required")
    if shutil.disk_usage(output.parent).free < limits["minimum_free_disk_bytes"]:
        raise OSError("E262 reserved disk unavailable")
    return freeze


def validated_held_inputs(sidecar: dict, saved_trace: dict | None = None) -> dict:
    if (sidecar.get("schema") != "flylab.m3.e262.held-tendon-inputs.v1"
            or sidecar.get("control_tick") != 55):
        raise ValueError("Exact E262 held tendon sidecar required")
    held = sidecar.get("tendon_inputs")
    if (not isinstance(held, dict) or len(held) != 8
            or any(not isinstance(key, str) for key in held)
            or any(type(value) not in (float, int) or not math.isfinite(value)
                   for value in held.values())):
        raise ValueError("Eight finite named held tendon values required")
    if (saved_trace is not None
            and (sidecar.get("source_trace_sha256") != sha256(safe(sidecar["source_trace"]))
                 or saved_trace["samples"][55]["tendon_inputs"] != held)):
        raise ValueError("Held tendon sidecar differs from original control55")
    return {key: float(value) for key, value in held.items()}


def install_held_inputs(session, held: dict) -> dict:
    """Replace all eight names; never merge with the previous branch's dict."""
    names = tuple(session.body.tendon_control.names)
    adapter = dict(session.adapter.tendon_inputs)
    if (len(names) != 8 or len(set(names)) != 8 or set(names) != set(held)
            or len(adapter) != 6 or not set(adapter).issubset(names)
            or any(np.float64(adapter[key]).tobytes() !=
                   np.float64(held[key]).tobytes() for key in adapter)):
        raise ValueError("E262 tendon roster or actual adapter six differs")
    session.tendon_inputs.clear()
    session.tendon_inputs.update({name: float(held[name]) for name in names})
    if (set(session.tendon_inputs) != set(names)
            or any(np.float64(session.tendon_inputs[name]).tobytes() !=
                   np.float64(held[name]).tobytes() for name in names)):
        raise AssertionError("E262 complete held tendon restoration failed")
    return held_receipt(session)


def held_receipt(session) -> dict:
    names = list(session.body.tendon_control.names)
    values = dict(session.tendon_inputs)
    if len(names) != 8 or set(names) != set(values):
        raise ValueError("E262 held tendon keys incomplete")
    array = np.asarray([values[key] for key in names], np.float64)
    if not np.isfinite(array).all():
        raise ValueError("E262 held tendon values nonfinite")
    return {"schema": "flylab.m3.e262.held-receipt.v1",
            "control_tick": int(session.control_tick), "tendon_names": names,
            "tendon_inputs": {name: float(values[name]) for name in names},
            "values": array.tolist()}


def restore_complete(session, fork: dict, held: dict) -> dict:
    """Reuse exact component restoration, then restore the unsnapshotted 8-input hold."""
    e261.restore_in_place(session, fork)
    receipt = install_held_inputs(session, held)
    exact(fork, session.snapshot(), "E262 entire fork after held input restoration")
    return receipt


def validate_end_held(session, start_held: dict, endpoint_row: dict) -> dict:
    """Six decoded tarsus values may move; two unmanaged abdomen values may not."""
    receipt = held_receipt(session)
    end = receipt["tendon_inputs"]
    adapter = dict(session.adapter.tendon_inputs)
    if (len(adapter) != 6 or not set(adapter).issubset(end)
            or set(end) != set(start_held)
            or set(endpoint_row.get("tendon_inputs", {})) != set(end)):
        raise ValueError("E262 endpoint held tendon roster differs")
    for name, value in end.items():
        expected = adapter[name] if name in adapter else start_held[name]
        observed = endpoint_row["tendon_inputs"][name]
        if (np.float64(value).tobytes() != np.float64(expected).tobytes()
                or np.float64(value).tobytes() != np.float64(observed).tobytes()):
            raise ValueError("E262 endpoint held tendon differs: " + name)
    return receipt


def save_arm_held_sidecars(arm_dir, start_receipt, session, science_primary,
                           errors, *, writer=e261.write_json):
    """Attempt both writes, retaining an earlier scientific failure as primary."""
    primary = science_primary
    end_receipt = None

    def absorb(label, error):
        nonlocal primary
        errors.append({"stage": label, "type": type(error).__name__,
                       "message": str(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E262 " + label + ": " + repr(error))

    try:
        writer(arm_dir / "start-held-tendon-inputs.json", start_receipt)
    except BaseException as error:
        absorb("start held sidecar", error)
    try:
        end_receipt = held_receipt(session)
        writer(arm_dir / "end-held-tendon-inputs.json", end_receipt)
    except BaseException as error:
        absorb("end held sidecar", error)
    return primary, end_receipt


def load_inputs(freeze):
    """All original arrays/state are read only after the complete source gate."""
    from scipy.sparse import csr_matrix

    path = lambda key: safe(freeze["paths"][key])
    baseline_data = e257.read_npz(path("baseline_csr"))
    baseline = csr_matrix((baseline_data["data"], baseline_data["indices"],
                           baseline_data["indptr"]), shape=(freeze["n"], freeze["n"]),
                          dtype=np.float32)
    baseline.check_format(full_check=True)
    frozen_e257 = json.loads(path("e257_freeze").read_text(encoding="utf-8"))
    patch = e257.read_npz(safe(frozen_e257["paths"]["e256_patch"]))
    candidate = e256.patch_six(baseline, patch,
                               np.asarray(frozen_e257["preserved_edge_ids"], np.int64))
    observed_csr = e257.read_npz(path("candidate_csr"))
    for key in ("data", "indices", "indptr"):
        if not e250.prior.same_bits(getattr(candidate, key), observed_csr[key]):
            raise ValueError("E262 candidate CSR differs from frozen six-edge input")
    saved = protocol.StateStore.load(path("start"))
    graph = protocol.GraphStore.load(path("graph_bundle"))
    if (saved.get("schema") != e257.LHPartitionTrial.schema
            or saved["control_tick"] != 55
            or saved["network"]["tick"] != 1100
            or saved["network"]["model_hash"] != freeze["network_identity"]
            or saved["body"]["modelHash"] != freeze["body_identity"]
            or graph.hash != freeze["graph_hash"]):
        raise ValueError("E262 exact E261 control55 whole fork required")
    sidecar = json.loads(path("held_tendon_inputs").read_text(encoding="utf-8"))
    original_trace = json.loads((path("reference") / "trace.json").read_text(
        encoding="utf-8"))
    held = validated_held_inputs(sidecar, original_trace)
    selection = json.loads(path("pulse_selection").read_text(encoding="utf-8"))
    if (selection.get("control_tick") != 56 or selection.get("joint_column") != 19
            or selection.get("pulse_delta_rad") != freeze["pulse_delta_rad"]):
        raise ValueError("E262 frozen E261 pulse selection differs")
    prefix = json.loads((path("prefix_evidence") / "segment-ledger.json").read_text(
        encoding="utf-8"))
    if (prefix.get("completed_controls") != 55 or prefix.get("reference_exact") is not True
            or prefix.get("status") != "complete_healthy"):
        raise ValueError("E261 verified fork prefix evidence incomplete")
    return graph, saved, baseline, candidate, held


def output_manifest(out, limit):
    result = e261.output_manifest(out, limit)
    result["schema"] = "flylab.m3.e262.output-manifest.v1"
    return result


def finalize(out, ledger, result, errors, primary, began, limit,
             *, writer=e261.write_json):
    """Own E262 terminal receipts; numerical raw survives failed publication."""
    def absorb(name, error):
        nonlocal primary
        errors.append({"stage": name, "type": type(error).__name__,
                       "message": str(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E262 " + name + ": " + repr(error))

    def revoke():
        result["validity"] = "invalid_or_incomplete"
        result["H262"] = {**result.get("H262", {}), "H261_supported": False,
                          "H262_supported": False, "classification": "invalid_or_incomplete"}
        canonical = out / "contrast.json"
        if canonical.exists():
            try:
                contrast = json.loads(canonical.read_text(encoding="utf-8"))
                contrast["H261_supported"] = False
                contrast["H262_supported"] = False
                contrast["classification"] = "unvalidated_after_failure"
                writer(canonical, contrast)
            except BaseException as error:
                try:
                    canonical.replace(out / "contrast-unvalidated.json")
                except BaseException:
                    canonical.unlink(missing_ok=True)
                errors.append({"stage": "contrast withdrawal", "error": repr(error)})

    result["finalization_pending"] = True
    if primary is not None:
        revoke()
    ledger["elapsed_s"] = time.perf_counter() - began
    for name, value in (("advance-ledger.json", ledger), ("errors.json", errors)):
        try:
            writer(out / name, value)
        except BaseException as error:
            absorb(name, error)
    try:
        writer(out / "output-manifest.json", output_manifest(out, limit))
    except BaseException as error:
        absorb("output-manifest.json", error)
    elapsed = time.perf_counter() - began
    if elapsed >= 45:
        absorb("internal_wall_s", TimeoutError("E262 45s internal deadline"))
    if primary is not None:
        revoke()
    result["elapsed_s"] = elapsed
    result["primary_error"] = None if primary is None else {
        "type": type(primary).__name__, "message": str(primary)}
    result["secondary_errors"] = errors.copy()
    result["finalization_pending"] = False
    try:
        writer(out / "result.json", result)
    except BaseException as error:
        absorb("result.json", error)
        (out / "result.json").unlink(missing_ok=True)
    if primary is None and time.perf_counter() - began >= 45:
        absorb("post_result_internal_wall_s", TimeoutError("E262 publication deadline"))
    if primary is not None:
        revoke()
        result["primary_error"] = {"type": type(primary).__name__, "message": str(primary)}
        result["secondary_errors"] = errors.copy()
        for name, value in (("errors.json", errors), ("advance-ledger.json", ledger),
                            ("result.json", result)):
            try:
                writer(out / name, value)
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
    ledger = {"schema": "flylab.m3.e262.lh-feedback-recovery.ledger.v1",
              "original_attempts": 1, "prefix_controls_reexecuted": 0,
              "whole_constructors_requested": 0, "whole_constructors_completed": 0,
              "in_place_restores_requested": 0, "in_place_restores_completed": 0,
              "construction": [], "body_construction": [], "segments": {},
              "fit_calls": 0, "solver_calls": 0, "current_probes": 0,
              "candidate_adopted": False, "F6_exposed": False,
              "whole_goal_complete": False}
    result = {"schema": "flylab.m3.e262.lh-feedback-recovery.result.v1",
              "validity": "incomplete", "H262": {"H261_supported": False,
              "H262_supported": False, "classification": "not_evaluated"},
              "candidate_adopted": False, "F6_exposed": False,
              "whole_goal_complete": False, "F3_F5_10s_evaluated": False,
              "finalization_pending": True}
    primary, session, thread_limits = None, None, None
    holder, errors = {}, []

    def persist():
        e261.write_json(out / "advance-ledger.json", ledger)

    def budget():
        if time.perf_counter() - began >= 37:
            raise TimeoutError("E262 8s finalization reserve reached")
        import psutil
        memory = psutil.Process().memory_info()
        resident = getattr(memory, "peak_wset", memory.rss)
        ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0), resident)
        if resident > freeze["resource_limits"]["process_memory_bytes"]:
            raise MemoryError("E262 host memory cap exceeded")
        import cupy as cp
        free_now, _ = cp.cuda.runtime.memGetInfo()
        added = max(0, ledger.get("gpu_free_before_bytes", free_now) - free_now)
        ledger["peak_added_gpu_bytes"] = max(ledger.get("peak_added_gpu_bytes", 0), added)
        if added > freeze["resource_limits"]["additional_gpu_memory_bytes"]:
            raise MemoryError("E262 added GPU cap exceeded")

    try:
        e261.write_json(out / "source-review-snapshot.json",
            json.loads(Path(review_path).read_text(encoding="utf-8")))
        from tools.fit_c_banc_m3_joint_adaptive_recruitment import one_cpu_thread
        thread_limits, ledger["threading"] = one_cpu_thread()
        import cupy as cp
        before_gpu, _ = cp.cuda.runtime.memGetInfo()
        if before_gpu < freeze["resource_limits"]["minimum_free_gpu_bytes"]:
            raise MemoryError("E262 GPU reservation unavailable")
        ledger["gpu_free_before_bytes"] = before_gpu
        graph, saved, baseline, candidate, held = load_inputs(freeze)
        ledger["whole_constructors_requested"] = 1
        persist()
        session = e257.construction_with_body_count("E262 fork", lambda:
            e257.LHPartitionTrial.from_checkpoint(graph, saved, baseline),
            ledger, persist, lambda value: holder.setdefault("session", value))
        ledger["whole_constructors_completed"] = 1
        exact(saved, session.snapshot(), "E262 restored E261 whole fork")
        if (session.network.identity != freeze["network_identity"]
                or session.body.model_hash != freeze["body_identity"]
                or session.control_tick != 55 or session.network.tick != 1100
                or session.network.n != freeze["n"] or session.network.device != "cuda"
                or session.network.cuda_implementation != "packed"
                or session.network.interval_steps != 20):
            raise ValueError("E262 fork CUDA/body identity differs")
        ledger["packed_csr"] = e250.packed_csr_receipt(session.network, candidate)
        install_held_inputs(session, held)
        if not e250.prior.same_bits(np.asarray(
                [session.body.d.ctrl[session.body.tendon_control.act_ids]], np.float64),
                np.asarray([[held[name] for name in session.body.tendon_control.names]],
                           np.float64)):
            raise ValueError("E262 actual fork tendon actuator values differ")
        protocol.StateStore.save(out / "observed-start", session.snapshot())
        protocol.StateStore.save(out / "fork", saved)
        e261.write_json(out / "fork-held-tendon-inputs.json", held_receipt(session))
        geom = e261.fork_geometry(session.body, freeze["pulse_joint"], freeze["leg_index"])
        np.savez_compressed(out / "fork-geometry.npz", **{key: value for key, value in geom.items()
            if not isinstance(value, str)})
        e261.write_json(out / "fork-geometry.json", {key: value for key, value in geom.items()
                        if isinstance(value, str)})
        reference = safe(freeze["paths"]["reference"])
        (out / "arms").mkdir(exist_ok=False)
        branches = {}
        end_held = {}
        for name in ARMS:
            budget()
            ledger["in_place_restores_requested"] += 1
            persist()
            receipt = restore_complete(session, saved, held)
            ledger["in_place_restores_completed"] += 1
            arm_dir = out / "arms" / name
            e261.write_json(out / f"start-held-{name}.json", receipt)
            branch = e261.run_segment(session, name, 55, 75, arm_dir,
                freeze, ledger, began, budget, pulse_sign=freeze["pulse_signs"][name],
                reference=reference if name.startswith("sham") else None)
            branches[name] = branch
            result[name] = branch["status"]
            branch_primary, end_receipt = save_arm_held_sidecars(
                arm_dir, receipt, session, branch["primary"], errors)
            if branch_primary is not None:
                raise branch_primary
            if branch["status"] != "complete_healthy":
                if branch["status"] == "measured_native_health_stop":
                    raise e261.MeasuredStop("E262 observed health stop: " + name)
                raise ValueError("E262 branch observation/clock incomplete: " + name)
            end_held[name] = validate_end_held(session, held, branch["rows"][-1])
            if end_receipt != end_held[name]:
                raise ValueError("E262 saved endpoint held sidecar differs")
        exact(branches["sham0"]["rows"], branches["sham1"]["rows"],
              "E262 sham entire physical trace")
        exact(end_held["sham0"], end_held["sham1"],
              "E262 sham final held tendon sidecars")
        left = protocol.StateStore.load(out / "arms/sham0/final")
        right = protocol.StateStore.load(out / "arms/sham1/final")
        exact(left, right, "E262 sham final whole state")
        for name in ("plus", "minus"):
            events = branches[name]["transfers"]
            if len(events) != 20 or sum(row["changed"] for row in events) != 1:
                raise ValueError("E262 exactly one pulse required")
            exact(branches[name]["activity"][1], branches["sham0"]["activity"][1],
                  "E262 first-control neural state")
            if not e250.prior.same_bits(events[0]["original_command"],
                                        branches["sham0"]["transfers"][0]["original_command"]):
                raise AssertionError("E262 first decoder output differs before pulse")
            for row in events:
                source, applied = row["original_command"], row["transmitted_command"]
                changed = np.flatnonzero([source[i].tobytes() != applied[i].tobytes()
                                          for i in range(42)])
                expected = np.array([19]) if row["control_tick"] == 56 else np.array([], int)
                if not np.array_equal(changed, expected):
                    raise ValueError("E262 pulse leaked from q19/control56")
        effect = e261.contrast(branches["plus"]["rows"], branches["minus"]["rows"],
                               branches["plus"]["transfers"], branches["minus"]["transfers"])
        if effect["classification"] == "invalid_or_incomplete":
            raise ValueError("E262 paired contrast invalid")
        result["H262"] = {**effect, "H262_supported": effect["H261_supported"]}
        result["validity"] = "valid_short_closed_loop_diagnostic"
        np.savez_compressed(out / "contrast.npz", controls=np.arange(55, 76),
            decoded_q19_plus=e261.decoded_target_series(
                branches["plus"]["controls"], branches["plus"]["transfers"]),
            decoded_q19_minus=e261.decoded_target_series(
                branches["minus"]["controls"], branches["minus"]["transfers"]),
            transmitted_q19_plus=np.asarray([row["command"][19]
                for row in branches["plus"]["controls"]]),
            transmitted_q19_minus=np.asarray([row["command"][19]
                for row in branches["minus"]["controls"]]),
            flexion_plus=np.asarray([row["flexion_rad"]
                for row in branches["plus"]["geometry"]]),
            flexion_minus=np.asarray([row["flexion_rad"]
                for row in branches["minus"]["geometry"]]))
        e261.write_json(out / "contrast.json", result["H262"])
    except e261.MeasuredStop as stop:
        result["validity"] = "measured_native_health_stop"
        result["H262"] = {**result["H262"], "H261_supported": False,
            "H262_supported": False, "classification": "inconclusive_health_stop"}
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
        ledger["setup_neural_executed_ticks"] = sum(row.get(
            "known_completed_setup_steps", 0) for row in ledger["construction"])
        ledger["setup_neural_capture_ticks"] = sum(row.get(
            "recorded_capture_setup_steps", 0) for row in ledger["construction"])
        ledger["setup_physics_steps_known"] = sum(row.get(
            "known_completed_settling_steps", 0) for row in ledger["body_construction"])
        events = [event for arm in ledger["segments"].values()
                  for event in arm["clock_events"]]
        ledger["scientific_neural_ticks_known"] = sum(
            max(0, int(event["neural_tick_after"] - event["neural_tick_before"]))
            for event in events if event.get("neural_tick_after") is not None)
        ledger["scientific_physics_steps_known"] = sum(
            max(0, round(event["actual_physical_s"] / .0001))
            for event in events if event.get("actual_physical_s") is not None)
        ledger["unknown_scientific_intervals"] = sum(
            event.get("neural_tick_after") is None or event.get("actual_physical_s") is None
            for event in events)
        if (primary is None and result["validity"] in
                ("valid_short_closed_loop_diagnostic", "measured_native_health_stop")
                and (ledger["whole_constructors_completed"] != 1
                     or ledger["setup_neural_executed_ticks"] != 280
                     or ledger["setup_neural_capture_ticks"] != 140
                     or ledger["setup_physics_steps_known"] != 2000
                     or (result["validity"] == "valid_short_closed_loop_diagnostic"
                         and (ledger["scientific_neural_ticks_known"] != 1600
                              or ledger["scientific_physics_steps_known"] != 4000
                              or ledger["in_place_restores_completed"] != 4)))):
            primary = ValueError("E262 actual setup/scientific work differs")
        primary = finalize(out, ledger, result, errors, primary, began,
                           freeze["resource_limits"]["output_bytes"])
    return 0 if primary is None and result["validity"] in (
        "valid_short_closed_loop_diagnostic", "measured_native_health_stop") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return execute(args.freeze, args.freeze_sha256, args.review,
                   args.review_sha256, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
