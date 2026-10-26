"""E240: apply E85 joint targets while retaining same-control native tendons."""
# Preserve the primary execution failure across cleanup and evidence publication.
# ruff: noqa: BLE001
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import observe_c_banc_m3_matched_teacher_inputs as original

BASE = original.BASE
ROOT = original.ROOT
FREEZE_SHA = "82bd3a85160c5eee59c9e66364ceec8f2a238e3a723789e5458daaa7d601e67f"
OWNED = ("tools/observe_c_banc_m3_native_tendon_teacher.py",
         "tests/test_c_banc_native_tendon_teacher.py")
OUT = BASE / "e240-native-tendon-teacher"
VALID = "valid_native_tendon_teacher_capture"


def verify_saved_proofs(parity, parity_result, matched, matched_result,
                        matched_ledger, matched_supervisor):
    if (parity.get("parity_validity") != "independently_valid_exact"
            or parity.get("arms", {}).get("parity", {}).get("completed_controls") != 99
            or parity_result.get("parity") != "valid_exact"
            or matched.get("validity") != "valid"
            or matched.get("arms", {}).get("teacher", {}).get("controls") != 2000
            or matched_result.get("validity") != "valid_matched_input_capture"
            or matched_result.get("teacher") != "valid_exact"
            or matched_result.get("candidate_adopted") is not False
            or matched_supervisor.get("exit_code") != 0
            or matched_supervisor.get("live") != []
            or matched_supervisor.get("terminated") != []):
        raise ValueError("Frozen E238 parity and complete E239 teacher proofs required")
    arms = matched_ledger.get("arms", [])
    if (len(arms) != 1 or arms[0].get("arm") != "teacher"
            or arms[0].get("valid") is not True
            or arms[0].get("controls_completed") != 2000
            or arms[0].get("native_samples") != 100000
            or arms[0].get("storage_errors") or matched_ledger.get("storage_errors")):
        raise ValueError("Complete E239 matched teacher execution required")


def gate(freeze_path, freeze_sha, review_path, review_sha, out):
    if (Path(freeze_path).resolve() != (BASE / "e240-native-tendon-teacher-freeze.json").resolve()
            or freeze_sha != FREEZE_SHA or original.sha256(freeze_path) != FREEZE_SHA
            or original.sha256(review_path) != review_sha):
        raise ValueError("Exact E240 freeze and review required")
    freeze = json.loads(Path(freeze_path).read_text("utf-8"))
    review = json.loads(Path(review_path).read_text("utf-8"))
    limits = freeze.get("resource_limits", {})
    if (freeze.get("schema") != "flylab.m3.e240.native-tendon-teacher.freeze.v1"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("graph_hash") !=
                "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"
            or freeze.get("expected_construction") != original.CONSTRUCTION
            or freeze.get("arms") != {"teacher": {"controls": 2000,
                "neural_ticks": 40000, "physics_steps": 100000, "stages": 160000}}
            or any(limits.get(k) != v for k, v in {
                "internal_s": 160, "finalization_reserve_s": 15,
                "external_s": 175, "supervisor_s": 190,
                "cpu_threads": 1, "gpu_count": 1, "original_attempts": 1,
                "retries": 0, "body_constructors": 1,
                "nominal_neural_constructors": 7, "fit_calls": 0,
                "solver_calls": 0, "resident_bytes": 6 << 30,
                "allocated_gpu_bytes": 4 << 30, "output_bytes": 2 << 30,
                "minimum_free_bytes": 3 << 30}.items())
            or freeze.get("original_cumulative_s") != 6408.073387799643
            or freeze.get("previous_cumulative_limit_s") != 6475
            or freeze.get("allocation_increase_s") != 130
            or freeze.get("original_cumulative_limit_s") != 6605
            or any(freeze.get(k) is not False for k in
                   ("candidate_adopted", "F2_complete", "F6_exposed",
                    "whole_goal_complete", "active_UI_changed"))
            or review.get("schema") != "flylab.m3.e240.native-tendon-teacher.review.v1"
            or review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Reviewed E240 native-tendon source required")
    for document, labels in [(freeze, ("input_sha256", "reused_source_sha256", "preservation_sha256")),
                             (review, ("source_sha256", "additional_reused_source_sha256", "additional_input_sha256"))]:
        for label in labels:
            for name, digest in document.get(label, {}).items():
                if original.sha256(original.safe(name)) != digest:
                    raise ValueError("E240 frozen source/input changed: " + name)
    if original.sha256(original.safe(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E240 frozen design changed")
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("E240 frozen package changed: " + name)
    paths = freeze["paths"]
    proof_paths = {"parity": BASE / "parent-e238-partial-output-review.json",
        "parity_result": BASE / "e238-matched-teacher-inputs/result.json",
        "matched": original.safe(paths["matched_proof"]),
        "matched_result": original.safe(paths["matched_result"]),
        "matched_ledger": original.safe(paths["matched_ledger"]),
        "matched_supervisor": original.safe(paths["matched_supervisor"])}
    proofs = {key: json.loads(path.read_text("utf-8"))
              for key, path in proof_paths.items()}
    verify_saved_proofs(*proofs.values())
    if out.resolve() != OUT.resolve() or out.exists() or not out.parent.is_dir():
        raise ValueError("New exact E240 output directory required")
    if shutil.disk_usage(out.parent).free < freeze["resource_limits"]["minimum_free_bytes"]:
        raise OSError("E240 minimum free disk unavailable")
    return freeze, review


class NativeTendonBoundary:
    """Apply only the teacher joint target; tendon is read in this decode call."""

    def __init__(self, session, command, names):
        self.session = session
        self.body = session.body
        self.command = np.asarray(command, np.float64).copy()
        self.names = tuple(names)
        if (self.command.shape != (42,) or not np.isfinite(self.command).all()
                or len(self.names) != 8 or len(set(self.names)) != 8):
            raise ValueError("Finite E85 q and original eight tendons required")
        self._had_instance = "step_joint_targets" in vars(self.body)
        self._previous = vars(self.body).get("step_joint_targets")
        self._original = self.body.step_joint_targets
        self.installed = False
        self.calls = 0
        self.receipt = None

    def _step(self, target, adhesion, *args, **kwargs):
        self.calls += 1
        if (self.calls != 1 or args or "tendon_inputs" not in kwargs
                or kwargs["tendon_inputs"] is not self.session.tendon_inputs):
            raise RuntimeError("One current decoder tendon action required")
        decoded = np.asarray(target, np.float64)
        actual = np.asarray([kwargs["tendon_inputs"][name] for name in self.names],
                            np.float64)
        if (decoded.shape != (42,) or actual.shape != (8,)
                or not np.isfinite(decoded).all() or not np.isfinite(actual).all()
                or np.asarray(adhesion).shape != (6,)
                or np.any(adhesion)):
            raise ValueError("Finite current decoder output required")
        self.receipt = {"decoded_command": decoded.copy(),
                        "applied_command": self.command.copy(),
                        "decoded_tendon": actual.copy(),
                        "applied_tendon": actual.copy(),
                        "applied_adhesion": np.zeros(6, bool)}
        # Pass the exact same live mapping from BancWalkingSession.advance.
        return self._original(self.command.copy(), np.zeros(6, bool), **kwargs)

    def install(self):
        if self.installed:
            raise RuntimeError("E240 body boundary already installed")
        self.body.step_joint_targets = self._step
        self.installed = True
        return self

    def restore(self):
        if not self.installed:
            return
        if self._had_instance:
            self.body.step_joint_targets = self._previous
        else:
            del self.body.step_joint_targets
        self.installed = False


def with_native_tendon_action(session, command, names, advance):
    boundary = NativeTendonBoundary(session, command, names).install()
    primary = None
    try:
        value = advance()
        if boundary.calls != 1 or boundary.receipt is None:
            raise RuntimeError("Exactly one E240 decoded/applied body action required")
        return value, boundary.receipt
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            boundary.restore()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note("Secondary E240 body boundary restore: " + repr(error))


def verify_first_readout(records, matched):
    if len(records) != 2:
        raise ValueError("E240 first control must complete before next control")
    for key in ("control_tick", "network_tick", "physics_time_s", "motor_rate32"):
        original.exact(np.asarray([row[key] for row in records]),
                       matched[key][:2], "E240 exact first readout " + key)


def verify_first_stage(out, freeze, observation_ids):
    for channel in original.CHANNELS:
        current = np.load(out / ("stage-" + channel + ".npy"), mmap_mode="r")
        matched = np.load(original.safe(freeze["paths"]["matched_" + channel]),
                          mmap_mode="r")
        if (current.dtype != np.float32 or current.shape != (160000, 391)
                or matched.dtype != np.float32
                or matched.shape != (160000, 391)):
            raise ValueError("E240 matched signed input array layout changed")
        original.exact(np.asarray(current[:80]), np.asarray(matched[:80]),
                       "E240 same start/input first control " + channel)
    if len(observation_ids) != 391 or len(np.unique(observation_ids)) != 391:
        raise ValueError("E240 full ordered native MN observation required")


def save_stage_raw(out, observer, names, tendon_actuator_ids):
    """Retain E239 stage schema and document the actual tendon ctrl columns."""
    ids = np.asarray(tendon_actuator_ids)
    if (ids.shape != (8,) or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != 8 or np.any(ids < 0)):
        raise ValueError("Eight ordered physical tendon actuator IDs required")
    primary = None
    try:
        original.save_stage_raw(out, observer, names)
    except BaseException as error:
        primary = error
    metadata_path = out / "observation.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text("utf-8"))
            metadata["tendon_actuator_ids"] = ids.astype(np.int64).tolist()
            original.e224.write_receipt(metadata_path, metadata)
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E240 tendon metadata: " + repr(error))
    if primary is not None:
        raise primary


def finalize_arm_receipt(account, primary, persist):
    """A completed scientific arm cannot remain valid after finalization fails."""
    def invalidate():
        if primary is not None and account.get("valid") is True:
            account.update(valid=False, status="technical_storage_or_cleanup_failure",
                           error=repr(primary))

    invalidate()
    try:
        persist()
    except BaseException as error:
        account["storage_errors"].append({"artifact": "arm-ledger", "error": repr(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E240 arm-ledger: " + repr(error))
        invalidate()
        try:
            persist()
        except BaseException as secondary:
            account["storage_errors"].append(
                {"artifact": "arm-ledger-retry", "error": repr(secondary)})
            primary.add_note("Secondary E240 arm-ledger retry: " + repr(secondary))
    return primary


def run_arm(freeze, graph, start, teacher_start, commands, matched_start,
            matched_first, out, ledger, persist, budget):
    arm_out = out / "teacher"
    arm_out.mkdir(exist_ok=False)
    account = {"arm": "teacher", "status": "requested", "controls_requested": 0,
               "controls_completed": 0, "body_constructors_requested": 1,
               "body_constructors_returned": 0,
               "observer_constructors_requested": 0,
               "observer_constructors_returned": 0,
               "construction": [], "clock_events": [], "storage_errors": []}
    ledger["arms"].append(account)
    persist()
    session = observer = restore_native = journal = None
    names = ()
    tendon_actuator_ids = np.empty(0, np.int64)
    rows, controls, native = [], [], []
    clock = original.e113.AdvanceLedger()
    primary = None
    trace_last = -1

    def attempt(label, callback):
        nonlocal primary
        if callback is None:
            return
        try:
            callback()
        except BaseException as error:
            account["storage_errors"].append({"artifact": label, "error": repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E240 " + label + ": " + repr(error))

    def own_session(value):
        nonlocal session
        session = value

    def own_observer(value):
        nonlocal observer
        observer = value

    try:
        budget()
        session = original.protocol.profiled_construction("restore_original",
            lambda: original.MotorTimescaleTrial.from_checkpoint(graph, start),
            account["construction"], persist, on_return=own_session)
        account["body_constructors_returned"] = 1
        original.exact(start, session.snapshot(), "E240 original MotorTimescale start")
        original.exact(session.body.snapshot(), teacher_start["body"],
                       "E240 identical E85 body origin")
        if (session.p.continuous_joint_activation or session.p.active_length
                or session.p.adhesion_model != "legacy" or session.control_tick != 0
                or session.network.tick != 0 or not session.fused_readout):
            raise ValueError("Original E150 body/decoder action mode required")
        names = tuple(session.body.tendon_control.names)
        if len(names) != 8 or len(set(names)) != 8:
            raise ValueError("Original eight tendon names required")
        tendon_actuator_ids = np.asarray(
            session.body.tendon_control.act_ids, np.int64).copy()
        if (tendon_actuator_ids.shape != (8,)
                or len(np.unique(tendon_actuator_ids)) != 8):
            raise ValueError("Original ordered eight tendon actuators required")
        account["observer_constructors_requested"] = 1
        persist()
        observer = original.protocol.profiled_construction("construct_observer",
            lambda: original.replace_with_signed_input_network(session,
                capacity_stages=160000), account["construction"], persist,
            on_return=own_observer)
        account["observer_constructors_returned"] = 1
        original.validate_construction(account["construction"])
        original.exact(start, session.snapshot(), "E240 passive observer start")
        original.protocol.StateStore.save(arm_out / "observed-start", session.snapshot())
        original.exact(matched_start, original.protocol.StateStore.load(
            arm_out / "observed-start"), "E240/E239 whole matched start")
        restore_native = original.e113.install_native_observer(session.body, native)
        journal = original.e224.FiniteControlJournal(arm_out / "control-journal.jsonl")
        row0 = original.protocol.capture_control(session)
        held_q = np.asarray(row0["joint_targets_rad"], np.float64)
        held_t = np.asarray([row0["tendon_inputs"][name] for name in names],
                            np.float64)
        controls.append({"control_tick": 0, "network_tick": 0,
            "physics_time_s": float(session.body.physics_time()),
            "motor_rate32": np.asarray(session.rate, np.float32).copy(),
            "applied_command": held_q.copy(), "decoded_command": held_q.copy(),
            "applied_tendon": held_t.copy(), "decoded_tendon": held_t.copy(),
            "applied_adhesion": np.asarray(row0["adhesion"], bool).copy()})
        rows.append(row0)
        observer.arm_observation()
        account["status"] = "scientific_advance"
        persist()
        with (arm_out / "trace.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
            stream.write(json.dumps(original.e224.finite_receipt(
                original.protocol.jsonable(row0))[0], allow_nan=False) + "\n")
            trace_last = 0
            for index in range(1, 2001):
                budget()
                before = (int(session.control_tick), int(session.network.tick),
                          float(session.body.physics_time()), len(native))
                account["controls_requested"] += 1

                def advance(index=index):
                    return original.protocol.advance_control(
                        session, "teacher", index, clock, journal)

                event, applied = with_native_tendon_action(session,
                    commands[index - 1], names, advance)
                try:
                    row, record = original.control_record(session, applied, names)
                    rows.append(row)
                    controls.append(record)
                    original.protocol.record_readout(journal, "teacher", index,
                                                     event, complete=True)
                except BaseException as error:
                    try:
                        original.protocol.record_readout(journal, "teacher", index,
                            event, complete=False, error=error)
                    except BaseException as secondary:
                        error.add_note("Secondary E240 readout journal: " + repr(secondary))
                    raise
                if index == 1:
                    verify_first_readout(controls, matched_first)
                if session.fault or session.body.fault:
                    raise original.e113.NativeHealthStop("E240 measured body fault")
                if (session.control_tick != before[0] + 1
                        or session.network.tick != before[1] + 20
                        or abs(session.body.physics_time() - before[2] - .005) > 2e-8
                        or len(native) != before[3] + 50):
                    raise ValueError("E240 incomplete control/native interval")
                stream.write(json.dumps(original.e224.finite_receipt(
                    original.protocol.jsonable(row))[0], allow_nan=False) + "\n")
                trace_last = index
                account["controls_completed"] += 1
        original.save_control(arm_out / "control-raw.npz", controls)
        original.e224.save_native_contact(arm_out / "native-contact.npz", native)
        original.protocol.StateStore.save(arm_out / "final", session.snapshot())
        save_stage_raw(arm_out, observer, names, tendon_actuator_ids)
        account["status"] = "raw_saved_before_validation"
        persist()
        if (len(controls) != 2001 or len(native) != 100000
                or session.control_tick != 2000 or observer.tick != 40000
                or not clock.clock_valid or session.fault or session.body.fault):
            raise ValueError("E240 complete original clocks/health required")
        with np.load(arm_out / "observation.npz", allow_pickle=False) as packet:
            original.validate_observation_packet(packet, observer, 160000)
        for channel in original.CHANNELS:
            stage = np.load(arm_out / ("stage-" + channel + ".npy"), mmap_mode="r")
            if (stage.dtype != np.float32 or stage.shape != (160000, 391)
                    or not np.isfinite(stage).all()):
                raise ValueError("E240 complete finite stage input required")
        verify_first_stage(arm_out, freeze, observer.observation_ids)
        with np.load(arm_out / "control-raw.npz", allow_pickle=False) as saved:
            if (not np.array_equal(saved["applied_command"][1:], commands)
                    or not np.array_equal(saved["applied_tendon"],
                                          saved["decoded_tendon"])
                    or np.any(saved["applied_adhesion"])):
                raise ValueError("E240 q-only/applied-current-tendon contract changed")
        measured = original.e113.native_arrays(native)
        if (np.any(measured["nonfoot_contact"])
                or not np.isfinite(measured["upright"]).all()
                or np.min(measured["upright"]) < .9):
            raise original.e113.NativeHealthStop("E240 measured native health stop")
        account.update(status="valid_native_tendon_teacher_capture", valid=True)
        persist()
    except original.e113.NativeHealthStop as error:
        primary = error
        account.update(status="measured_native_health_stop", error=repr(error))
    except (TimeoutError, MemoryError) as error:
        primary = error
        account.update(status="resource_limited", error=repr(error))
    except BaseException as error:
        primary = error
        account.update(status="technical_or_fidelity_failure", error=repr(error),
                       traceback=traceback.format_exc())
    finally:
        if observer is not None:
            attempt("stage-observer", lambda: save_stage_raw(
                arm_out, observer, names, tendon_actuator_ids))
        if not (arm_out / "control-raw.npz").exists():
            attempt("control-raw.npz", lambda: original.save_control(
                arm_out / "control-raw.npz", controls))
        if not (arm_out / "native-contact.npz").exists():
            attempt("native-contact.npz", lambda: original.e224.save_native_contact(
                arm_out / "native-contact.npz", native))
        for key, callback in (("native-observer", restore_native),
                              ("control-journal", journal.close if journal else None),
                              ("session", (lambda: None if session.closed else
                               session.close()) if session else None)):
            attempt("cleanup-" + key, callback)
        account.update(clock_events=clock.events, actual_neural_s=clock.neural_s,
                       actual_physical_s=clock.physical_s,
                       clock_valid=clock.clock_valid,
                       native_samples=len(native), control_rows=len(controls),
                       last_durable_trace_index=trace_last,
                       journal=journal.summary() if journal else None)
        observer = session = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E240 CUDA cleanup: " + repr(error))
        primary = finalize_arm_receipt(account, primary, persist)
    if primary is not None:
        raise primary
    return account


def publish_final_receipts(result, ledger, primary, write_ledger,
                           write_result, write_manifest):
    """Publish best-effort receipts; a late storage error invalidates capture."""
    def record(label, error):
        nonlocal primary
        ledger["storage_errors"].append({"artifact": label, "error": repr(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E240 " + label + ": " + repr(error))
        if result["validity"] == VALID:
            result["pre_storage_validity"] = VALID
            result["validity"] = "technical_storage_failure"
        result["execution_status"] = "failed"
        result["primary_error"] = repr(primary)

    for _ in range(2):
        before = len(ledger["storage_errors"])
        for label, callback in (("advance-ledger.json", write_ledger),
                                ("result.json", write_result),
                                ("output-manifest.json", write_manifest)):
            try:
                callback()
            except BaseException as error:
                record(label, error)
        if len(ledger["storage_errors"]) == before:
            break
    return primary


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze, review = gate(freeze_path, freeze_sha, review_path, review_sha, out)
    out.mkdir(exist_ok=False)
    limits = freeze["resource_limits"]
    ledger = {"schema": "flylab.m3.e240.native-tendon-teacher.ledger.v1",
              "status": "verified_saved_E238_E239", "original_processes": 1,
              "retries": 0, "fit_calls": 0, "solver_calls": 0,
              "arms": [], "storage_errors": [],
              "reused_parity_neural_ticks": 1980, "new_parity_neural_ticks": 0,
              "candidate_adopted": False, "F6_exposed": False, "whole_goal_complete": False}
    result = {"schema": "flylab.m3.e240.native-tendon-teacher.result.v1",
              "validity": "incomplete", "execution_status": "running",
              "parity_source": "independently verified saved E238, not re-executed",
              "matched_source": "independently verified saved E239, not re-executed",
              "controller_scope": "E85 q teacher with same-control original decoder tendons",
              "teacher": "not_run", "candidate_adopted": False, "F2_complete": False,
              "F6_exposed": False, "whole_goal_complete": False}
    primary = threads = process = gpu_free_before = None

    def persist():
        ledger["process_wall_s"] = time.perf_counter()-began
        original.e224.write_receipt(out/"advance-ledger.json", ledger)

    def budget():
        if time.perf_counter()-began >= limits["internal_s"]-limits["finalization_reserve_s"]:
            raise TimeoutError("E240 finalization reserve reached")
        if process is not None:
            memory = process.memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0), peak)
            if peak > limits["resident_bytes"]:
                raise MemoryError("E240 resident limit exceeded")
        if gpu_free_before is not None:
            import cupy as cp
            available, _ = cp.cuda.runtime.memGetInfo()
            added = max(0, gpu_free_before-available)
            ledger["peak_added_gpu_bytes"] = max(ledger.get("peak_added_gpu_bytes", 0), added)
            if added > limits["allocated_gpu_bytes"]:
                raise MemoryError("E240 additional GPU limit exceeded")

    try:
        original.e224.write_receipt(out/"source-review-snapshot.json", review)
        from tools.fit_c_banc_m3_joint_adaptive_recruitment import one_cpu_thread
        threads, ledger["threading"] = one_cpu_thread()
        import cupy as cp
        import psutil
        process = psutil.Process()
        gpu_free_before, _ = cp.cuda.runtime.memGetInfo()
        graph = original.protocol.GraphStore.load(original.safe(freeze["paths"]["graph"]))
        if graph.hash != freeze["graph_hash"] or graph.n != 158706:
            raise ValueError("Original whole BANC graph required")
        saved = original.protocol.StateStore.load(original.safe(freeze["paths"]["start"]))
        start, _, _ = original.unwrap_neural_current_checkpoint(saved)
        teacher_start = original.protocol.StateStore.load(original.safe(freeze["paths"]["teacher_start"]))
        original.exact(start["body"], teacher_start["body"],
                       "E240 identical teacher body origin")
        matched_start = original.protocol.StateStore.load(
            original.safe(freeze["paths"]["matched_start"]))
        with np.load(original.safe(freeze["paths"]["matched_control"]),
                     allow_pickle=False) as data:
            matched_first = {key: data[key][:2].copy() for key in
                             ("control_tick", "network_tick", "physics_time_s",
                              "motor_rate32")}
        with np.load(original.safe(freeze["paths"]["teacher_command"]), allow_pickle=False) as data:
            command = data["command"].copy()
        if command.shape != (2000, 42) or command.dtype != np.float64 or not np.isfinite(command).all():
            raise ValueError("Frozen full E85 teacher commands required")
        ledger["status"] = "teacher_requested_after_saved_proofs"
        persist()
        original.execute_registered_arm("teacher", lambda: run_arm(
            freeze, graph, start, teacher_start, command, matched_start,
            matched_first, out, ledger, persist, budget), result, ledger)
        result.update(validity=VALID, execution_status="complete",
                      teacher="valid_native_tendon_capture",
                      controls=2000, neural_ticks=40000, physics_steps=100000,
                      conclusion="q-only teacher with original decoder tendons; gait reference pending")
        ledger["status"] = "teacher_completed"
    except BaseException as error:
        primary = error
        validity = ("measured_native_health_stop" if isinstance(error, original.e113.NativeHealthStop)
                    else "resource_limited" if isinstance(error, (TimeoutError, MemoryError))
                    else "technical_or_fidelity_failure")
        result.update(validity=validity, error=repr(error), traceback=traceback.format_exc())
    finally:
        if threads is not None:
            try:
                threads.restore_original_limits()
            except BaseException as error:
                ledger["storage_errors"].append({"artifact": "thread_cleanup", "error": repr(error)})
                if primary is None:
                    primary = error
                else:
                    primary.add_note("Secondary E240 thread cleanup: " + repr(error))
        if time.perf_counter()-began > limits["internal_s"] and primary is None:
            primary = TimeoutError("E240 internal limit exceeded")
            result["validity"] = "resource_limited"
        if primary is not None:
            if result["validity"] == VALID:
                result["pre_storage_validity"] = result["validity"]
                result["validity"] = "technical_cleanup_failure"
            result.update(execution_status="failed", primary_error=repr(primary))
        primary = publish_final_receipts(result, ledger, primary, persist,
            lambda: original.e224.write_receipt(out/"result.json", result),
            lambda: original.e224.write_receipt(out/"output-manifest.json", original.manifest(out)))
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "freeze-sha256", "review", "review-sha256", "out"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    try:
        value = run(args.freeze, args.freeze_sha256, args.review, args.review_sha256, args.out)
    except BaseException as error:
        print(type(error).__name__+": "+str(error), file=sys.stderr)
        return 1
    print(json.dumps(value), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
