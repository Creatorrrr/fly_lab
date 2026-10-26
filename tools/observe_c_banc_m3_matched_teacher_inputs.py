"""E238: capture matched BANC inputs while applying frozen body commands."""

# Keep the first scientific/technical error across independent raw cleanup.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import gc
import hashlib
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

from tools import verify_c_banc_m3_e113_body as e113
from tools import verify_c_banc_m3_native_timescale_transfer as protocol
from tools import verify_c_banc_m3_neural_current_body as e224
from tools.banc_m3_motor_timescale import MotorTimescaleTrial
from tools.banc_m3_neural_current_trial import unwrap_neural_current_checkpoint
from tools.banc_m3_signed_input_channels import replace_with_signed_input_network
from tools.verify_c_banc_observation import exact

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
OUT = BASE / "e238-matched-teacher-inputs"
FREEZE_SHA = "c65b3940835e208cd67fddcef9f8cb2dac4977a56c2ab45afa8fb0e18e8569dc"
FREEZE_SCHEMA = "flylab.m3.e238.matched-teacher-inputs.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e238.matched-teacher-inputs.review.v1"
OWNED = ("tools/observe_c_banc_m3_matched_teacher_inputs.py",
         "tests/test_c_banc_matched_teacher_inputs.py")
CHANNELS = ("positive_recurrent", "negative_magnitude", "drive")
PHYSICAL_KEYS = ("body", "qpos", "qvel", "ctrl", "joint_angles_rad",
                 "joint_targets_rad", "joint_tracking_error_rad", "adhesion",
                 "upright", "feet", "fault", "control_tick", "time_s")
NATIVE_KEYS = ("time_s", "normal_load_bw", "slip_mm_s",
               "nonfoot_contact", "upright")
INTERNAL_S = 150.
RESERVE_S = 15.
CONSTRUCTION = {"restore_original": {"captured_interval_entries": 6,
    "executed_interval_entries": 12, "known_completed_setup_steps": 240,
    "recorded_capture_setup_steps": 120},
    "construct_observer": {"captured_interval_entries": 1,
    "executed_interval_entries": 2, "known_completed_setup_steps": 40,
    "recorded_capture_setup_steps": 20}}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(relative):
    path = Path(relative)
    resolved = (ROOT / path).resolve()
    if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(ROOT):
        raise ValueError("Repository-relative frozen E238 path required")
    return resolved


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if (Path(freeze_path).resolve() != (BASE /
            "e238-matched-teacher-inputs-freeze.json").resolve()
            or freeze_sha != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA
            or sha256(review_path) != review_sha):
        raise ValueError("Exact E238 freeze/review SHA required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    limits = freeze.get("resource_limits", {})
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or sha256(safe(freeze["design"])) != freeze.get("design_sha256")
            or freeze.get("graph_hash") !=
               "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"
            or freeze.get("native") != {"nodes": 158706, "motors": 391,
                "parity_free_motors": 343, "neural_dt_s": .00025,
                "control_dt_s": .005, "physics_dt_s": .0001,
                "neural_ticks_per_control": 20, "rk_stages_per_tick": 4,
                "physics_steps_per_control": 50}
            or freeze.get("arms") != {"parity": {"controls": 99,
                "neural_ticks": 1980, "physics_steps": 4950, "stages": 7920},
                "teacher": {"controls": 2000, "neural_ticks": 40000,
                "physics_steps": 100000, "stages": 160000}}
            or freeze.get("expected_construction") != CONSTRUCTION
            or any(limits.get(k) != v for k, v in {
                "internal_s": 150, "finalization_reserve_s": 15,
                "external_s": 170, "supervisor_s": 180, "cpu_threads": 1,
                "gpu_count": 1, "original_attempts": 1, "retries": 0,
                "body_constructors": 2, "nominal_neural_constructors": 14,
                "fit_calls": 0, "solver_calls": 0,
                "resident_bytes": 6 << 30, "allocated_gpu_bytes": 4 << 30,
                "output_bytes": 2 << 30, "minimum_free_bytes": 3 << 30}.items())
            or freeze.get("original_cumulative_s") != 6137.263854399662
            or freeze.get("previous_cumulative_limit_s") != 6215
            or freeze.get("allocation_increase_s") != 110
            or freeze.get("original_cumulative_limit_s") != 6325
            or any(freeze.get(key) is not False for key in
                   ("candidate_adopted", "F2_complete", "F6_exposed",
                    "whole_goal_complete", "active_UI_changed"))):
        raise ValueError("Frozen E238 scientific/resource contract changed")
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved E238 source review required")
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        mapping = freeze.get(label)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Frozen E238 inventory required: " + label)
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("Frozen E238 input/source changed: " + name)
    for label in ("source_sha256", "additional_reused_source_sha256",
                  "additional_input_sha256"):
        mapping = review.get(label, {})
        if not isinstance(mapping, dict):
            raise TypeError("Reviewed E238 hashes require mapping")
        for name, expected in mapping.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256(safe(name)) != expected):
                raise ValueError("Reviewed E238 input/source changed: " + name)
    for package, expected in freeze["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("E238 frozen package version changed: " + package)
    if output.resolve() != OUT.resolve() or output.exists() or not output.parent.is_dir():
        raise FileExistsError("New exact E238 output directory required")
    if shutil.disk_usage(output.parent).free < limits["minimum_free_bytes"]:
        raise OSError("E238 minimum free disk unavailable")
    return freeze, review


class AppliedCommandBoundary:
    """Replace only one body action inside the original session.advance call."""

    def __init__(self, session, command, tendon, names):
        self.session = session
        self.body = session.body
        self.command = np.asarray(command, np.float64).copy()
        self.tendon = np.asarray(tendon, np.float64).copy()
        self.names = tuple(names)
        if (self.command.shape != (42,) or self.tendon.shape != (len(self.names),)
                or len(self.names) != 8 or len(set(self.names)) != 8
                or not np.isfinite(self.command).all()
                or not np.isfinite(self.tendon).all()):
            raise ValueError("Finite frozen 42-joint/eight-tendon action required")
        self._had_instance_method = "step_joint_targets" in vars(self.body)
        self._old_instance_method = vars(self.body).get("step_joint_targets")
        self._original = self.body.step_joint_targets
        self.calls = 0
        self.decoded_command = None
        self.decoded_tendon = None
        self.applied_adhesion = np.zeros(6, bool)
        self.installed = False

    def _step(self, target, adhesion, *args, **kwargs):
        self.calls += 1
        if self.calls != 1 or args or "tendon_inputs" not in kwargs:
            raise RuntimeError("One original body action per control required")
        decoded = np.asarray(target, np.float64)
        previous_tendon = kwargs["tendon_inputs"]
        if (decoded.shape != (42,) or not np.isfinite(decoded).all()
                or set(previous_tendon) != set(self.names)
                or np.asarray(adhesion).shape != (6,)):
            raise ValueError("Original decoder action schema changed")
        self.decoded_command = decoded.copy()
        self.decoded_tendon = np.asarray([previous_tendon[name]
                                         for name in self.names], np.float64)
        applied = dict(zip(self.names, self.tendon, strict=True))
        self.session.tendon_inputs.clear()
        self.session.tendon_inputs.update(applied)
        kwargs["tendon_inputs"] = self.session.tendon_inputs
        return self._original(self.command.copy(), self.applied_adhesion.copy(),
                              **kwargs)

    def install(self):
        if self.installed:
            raise RuntimeError("Body action boundary already installed")
        self.body.step_joint_targets = self._step
        self.installed = True
        return self

    def restore(self):
        if not self.installed:
            return
        if self._had_instance_method:
            self.body.step_joint_targets = self._old_instance_method
        else:
            del self.body.step_joint_targets
        self.installed = False

    def receipt(self):
        if self.calls != 1 or self.decoded_command is None:
            raise RuntimeError("Exactly one decoded/applied body action required")
        return {"decoded_command": self.decoded_command.copy(),
                "decoded_tendon": self.decoded_tendon.copy(),
                "applied_command": self.command.copy(),
                "applied_tendon": self.tendon.copy(),
                "applied_adhesion": self.applied_adhesion.copy()}


def with_applied_action(session, command, tendon, names, advance):
    """Always restore the instance method, preserving the original failure."""
    boundary = AppliedCommandBoundary(session, command, tendon, names).install()
    primary = None
    try:
        value = advance()
        return value, boundary.receipt()
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            boundary.restore()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note("Secondary E238 body boundary restore: " + repr(error))


def control_record(session, action, names):
    row, value = e224.capture_state(session, "legacy_canonical")
    value = {"control_tick": value["control_tick"],
             "network_tick": value["network_tick"],
             "physics_time_s": value["physics_time_s"],
             "motor_rate32": value["motor_rate32"], **action}
    if (value["applied_tendon"].shape != (len(names),)
            or value["motor_rate32"].dtype != np.float32):
        raise ValueError("E238 control observation changed")
    return row, value


def save_control(path, values):
    fields = {"control_tick": ((), np.int64),
              "network_tick": ((), np.int64),
              "physics_time_s": ((), np.float64),
              "motor_rate32": ((391,), np.float32),
              "applied_command": ((42,), np.float64),
              "applied_tendon": ((8,), np.float64),
              "applied_adhesion": ((6,), bool),
              "decoded_command": ((42,), np.float64),
              "decoded_tendon": ((8,), np.float64)}
    arrays = {key: np.asarray([row[key] for row in values], dtype=dtype).reshape(
        (len(values), *shape)) for key, (shape, dtype) in fields.items()}
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def native_equal(actual, reference, *, parity):
    keys = NATIVE_KEYS + (("slip_valid",) if parity else ())
    for key in keys:
        left, right = actual[key], reference[key]
        if (left.shape != right.shape or left.dtype != right.dtype
                or left.tobytes() != right.tobytes()):
            raise ValueError("E238 native replay differs: " + key)


def physical_equal(rows, source, *, tendon_names):
    if len(rows) != len(source):
        raise ValueError("E238 physical trace row count changed")
    for index, (left, right) in enumerate(zip(rows, source, strict=True)):
        for key in PHYSICAL_KEYS:
            exact(left[key], right[key], "E238 physical " + key + " row " + str(index))
        if index:
            for name in tendon_names:
                exact(left["tendon_inputs"][name], right["tendon_inputs"][name],
                      "E238 applied tendon row " + str(index))


def compare_parity_final(actual, saved):
    """Ignore only unused canonical activation/last_motor and wall time."""
    for key in ("network", "body", "navigation", "control_tick", "cuts",
                "initial_position", "signed_forward_mm", "cpg_hash"):
        exact(actual[key], saved[key], "E238 parity final " + key)
    exact(actual["adapter"]["original"], saved["adapter"]["original"],
          "E238 underlying sensory adapter")


def require_parity_before_teacher(result, ledger):
    if (result.get("parity") != "valid_exact" or result.get("teacher") != "not_run"
            or not ledger["arms"] or ledger["arms"][-1].get("arm") != "parity"
            or ledger["arms"][-1].get("valid") is not True):
        raise RuntimeError("E238 exact parity must pass before teacher construction")


def execute_registered_arm(label, callback, result, ledger):
    """Leave the next arm not_run when the current one fails at any stage."""
    result[label] = "requested"
    try:
        result[label] = "running"
        value = callback()
    except BaseException:
        recent = ledger["arms"][-1] if ledger["arms"] else None
        result[label] = (recent.get("status", "failed")
                         if recent and recent.get("arm") == label
                         else "failed_before_arm_receipt")
        raise
    result[label] = "valid_exact"
    return value


def validate_observation_packet(packet, observer, stages):
    """SignedInputChannelNetwork returns one-element int32 flag arrays."""
    counters = packet["counters"]
    overflow = packet["overflow"]
    armed = packet["armed"]
    if (counters.dtype != np.int64 or counters.shape != (391,)
            or overflow.dtype != np.int32 or overflow.shape != (1,)
            or armed.dtype != np.int32 or armed.shape != (1,)
            or not np.all(counters == stages)
            or int(overflow.item()) != 0 or int(armed.item()) != 0
            or not np.array_equal(packet["motor_ids"], observer.observation_ids)):
        raise ValueError("E238 complete signed-input counters/flag shape required")


def publish_final_receipts(result, ledger, primary, write_ledger,
                           write_result, write_manifest):
    """Best-effort fail-closed publication; a science error remains primary."""
    def record(label, error):
        nonlocal primary
        ledger["storage_errors"].append({"artifact": label, "error": repr(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E238 " + label + ": " + repr(error))
        if result["validity"] == "valid_matched_input_capture":
            result["pre_storage_validity"] = result["validity"]
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


def validate_construction(rows):
    if len(rows) != 2 or [row.get("label") for row in rows] != list(CONSTRUCTION):
        raise ValueError("E238 synchronized restore/observer profile required")
    for row in rows:
        if (not row.get("ownership_transferred")
                or not row.get("synchronize_returned")
                or not row.get("profiler_restored")
                or row.get("unknown_setup_steps") != 0
                or row.get("unknown_capture_setup_steps") != 0
                or row.get("unknown_capture_state_entries") != 0
                or row.get("interval_entries") != row.get("interval_returns")
                or any(row.get(key) != value for key, value in
                       CONSTRUCTION[row["label"]].items())):
            raise ValueError("E238 construction/capture accounting changed")


def save_stage_raw(out, observer, names):
    if getattr(observer, "_observation_start_tick", None) is None:
        return
    observer.disable_observation()
    metadata = observer.observation_static_metadata()
    metadata["tendon_names"] = list(names)
    metadata["motor_ids"] = observer.observation_ids.tolist()
    errors = []
    for channel in CHANNELS:
        destination = out / ("stage-" + channel + ".npy")
        if not destination.exists():
            try:
                temporary = out / ("stage-" + channel + ".tmp.npy")
                np.save(temporary, observer.copy_values(channel), allow_pickle=False)
                temporary.replace(destination)
            except BaseException as error:
                errors.append((channel, error))
    def save_counters():
        temporary = out / "observation.tmp.npz"
        np.savez_compressed(temporary, counters=observer.copy_counters(),
            overflow=np.array(observer.copy_overflow(), np.int32),
            armed=np.array(observer.copy_armed(), np.int32),
            motor_ids=observer.observation_ids.copy())
        temporary.replace(out / "observation.npz")

    for label, callback in (("observation.json", lambda: e224.write_receipt(
            out / "observation.json", metadata)),
            ("observation.npz", save_counters)):
        if not (out / label).exists():
            try:
                callback()
            except BaseException as error:
                errors.append((label, error))
    if errors:
        first = errors[0][1]
        for label, error in errors[1:]:
            first.add_note("Secondary E238 stage raw " + label + ": " + repr(error))
        raise first


def manifest(out):
    return {str(path.relative_to(out)).replace("\\", "/"): {
        "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(out.rglob("*")) if path.is_file()
        and path.name != "output-manifest.json"}


def run_arm(label, freeze, graph, original_start, commands, tendons, out,
            ledger, persist, budget, *, owner=None, source_tendon_names=None):
    """One new BANC body and one signed-input observer; return only after close."""
    arm = freeze["arms"][label]
    arm_out = out / label
    arm_out.mkdir(exist_ok=False)
    account = {"arm": label, "status": "requested", "controls_requested": 0,
               "controls_completed": 0, "body_constructors_requested": 1,
               "body_constructors_returned": 0,
               "observer_constructors_requested": 0,
               "observer_constructors_returned": 0,
               "construction": [], "clock_events": [], "storage_errors": []}
    ledger["arms"].append(account)
    persist()
    session = observer = restore_native = journal = None
    rows, controls, native = [], [], []
    names = ()
    clock = e113.AdvanceLedger()
    primary = None
    trace_last = -1

    def attempt(name, callback):
        nonlocal primary
        if callback is None:
            return
        try:
            callback()
        except BaseException as error:
            account["storage_errors"].append({"artifact": name, "error": repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E238 " + name + ": " + repr(error))

    def own_session(value):
        nonlocal session
        session = value

    def own_observer(value):
        nonlocal observer
        observer = value

    try:
        budget()
        session = protocol.profiled_construction("restore_original",
            lambda: MotorTimescaleTrial.from_checkpoint(graph, original_start),
            account["construction"], persist, on_return=own_session)
        account["body_constructors_returned"] = 1
        exact(original_start, session.snapshot(), "E238 restored original start")
        if (session.p.continuous_joint_activation or session.p.active_length
                or session.p.adhesion_model != "legacy" or session.control_tick != 0
                or session.network.tick != 0 or not session.fused_readout):
            raise ValueError("Frozen original E150 body/action mode required")
        if owner is not None:
            exact(session.body.snapshot(), owner["body"],
                  "E238 teacher and E85 identical body start")
        names = tuple(session.body.tendon_control.names)
        if len(names) != 8 or len(set(names)) != 8:
            raise ValueError("Original eight named tendons required")
        if source_tendon_names is None:
            source_tendon_names = names
        if len(source_tendon_names) != 8 or set(source_tendon_names) != set(names):
            raise ValueError("Frozen eight-tendon source ownership changed")
        tendon_order = [source_tendon_names.index(name) for name in names]
        account["observer_constructors_requested"] = 1
        persist()
        observer = protocol.profiled_construction("construct_observer",
            lambda: replace_with_signed_input_network(session,
                capacity_stages=arm["stages"]), account["construction"],
            persist, on_return=own_observer)
        account["observer_constructors_returned"] = 1
        validate_construction(account["construction"])
        exact(original_start, session.snapshot(),
              "E238 signed recorder must preserve whole start state")
        protocol.StateStore.save(arm_out / "observed-start", session.snapshot())
        restore_native = e113.install_native_observer(session.body, native)
        journal = e224.FiniteControlJournal(arm_out / "control-journal.jsonl")
        row0 = protocol.capture_control(session)
        held = np.asarray(row0["joint_targets_rad"], np.float64)
        held_tendon = np.asarray([row0["tendon_inputs"][name] for name in names],
                                 np.float64)
        row0_value = {"control_tick": 0, "network_tick": 0,
            "physics_time_s": float(session.body.physics_time()),
            "motor_rate32": np.asarray(session.rate, np.float32).copy(),
            "applied_command": held.copy(), "decoded_command": held.copy(),
            "applied_tendon": held_tendon.copy(),
            "decoded_tendon": held_tendon.copy(),
            "applied_adhesion": np.asarray(row0["adhesion"], bool).copy()}
        rows.append(row0)
        controls.append(row0_value)
        observer.arm_observation()
        account["status"] = "scientific_advance"
        persist()
        with (arm_out / "trace.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
            stream.write(json.dumps(e224.finite_receipt(protocol.jsonable(row0))[0],
                                    allow_nan=False) + "\n")
            trace_last = 0
            for index in range(1, arm["controls"] + 1):
                budget()
                before = (int(session.control_tick), int(session.network.tick),
                          float(session.body.physics_time()), len(native))
                account["controls_requested"] += 1
                action = np.asarray(commands[index - 1], np.float64)
                tendon = np.asarray(tendons[index - 1, tendon_order], np.float64)

                def advance(index=index):
                    return protocol.advance_control(session, label, index,
                                                    clock, journal)

                event, applied = with_applied_action(session, action, tendon,
                                                     names, advance)
                try:
                    row, record = control_record(session, applied, names)
                    rows.append(row)
                    controls.append(record)
                    protocol.record_readout(journal, label, index, event,
                                            complete=True)
                except BaseException as error:
                    try:
                        protocol.record_readout(journal, label, index, event,
                                                complete=False, error=error)
                    except BaseException as secondary:
                        error.add_note("Secondary E238 readout journal: " + repr(secondary))
                    raise
                if session.fault or session.body.fault:
                    raise e113.NativeHealthStop("E238 measured body fault")
                if (session.control_tick != before[0] + 1
                        or session.network.tick != before[1] + 20
                        or abs(session.body.physics_time() - before[2] - .005) > 2e-8
                        or len(native) != before[3] + 50):
                    raise ValueError("E238 incomplete original control/native interval")
                stream.write(json.dumps(e224.finite_receipt(protocol.jsonable(row))[0],
                                        allow_nan=False) + "\n")
                trace_last = index
                account["controls_completed"] += 1
        # Seal each raw family before any source-fidelity assertion.
        save_control(arm_out / "control-raw.npz", controls)
        e224.save_native_contact(arm_out / "native-contact.npz", native)
        protocol.StateStore.save(arm_out / "final", session.snapshot())
        save_stage_raw(arm_out, observer, names)
        account["status"] = "raw_saved_before_validation"
        persist()
        if (len(rows) != arm["controls"] + 1 or len(native) != arm["physics_steps"]
                or session.control_tick != arm["controls"]
                or observer.tick != arm["neural_ticks"] or not clock.clock_valid
                or session.fault or session.body.fault):
            raise ValueError("E238 completed original clocks/health required")
        with np.load(arm_out / "observation.npz", allow_pickle=False) as packet:
            validate_observation_packet(packet, observer, arm["stages"])
        for channel in CHANNELS:
            value = np.load(arm_out / ("stage-" + channel + ".npy"), mmap_mode="r")
            if (value.shape != (arm["stages"], 391) or value.dtype != np.float32
                    or not np.isfinite(value).all()):
                raise ValueError("E238 complete finite stage input required: " + channel)
        source_rows = [json.loads(line) for line in safe(freeze["paths"][
            "parity_trace" if label == "parity" else "teacher_trace"]).read_text(
                encoding="utf-8").splitlines()]
        physical_equal(rows, source_rows, tendon_names=names)
        with (np.load(arm_out / "native-contact.npz", allow_pickle=False) as current,
              np.load(safe(freeze["paths"]["parity_native" if label == "parity"
                                            else "teacher_native"]),
                      allow_pickle=False) as expected):
            native_equal(current, expected, parity=label == "parity")
        if label == "parity":
            with (np.load(safe(freeze["paths"]["parity_control"]),
                          allow_pickle=False) as expected,
                  np.load(arm_out / "control-raw.npz", allow_pickle=False) as current):
                for key, source_key in (("motor_rate32", "motor_rate32"),
                                        ("control_tick", "control_tick"),
                                        ("network_tick", "network_tick"),
                                        ("physics_time_s", "physics_time_s")):
                    exact(current[key], expected[source_key],
                          "E238 parity control " + key)
            parameters = e224.load_npz(safe(freeze["paths"]["parameters"]))
            free = parameters["free_columns"]
            if not np.array_equal(observer.observation_ids,
                                  parameters["motor_ids"]):
                raise ValueError("E238 original ordered MN roster changed")
            for channel in CHANNELS:
                current = np.load(arm_out / ("stage-" + channel + ".npy"),
                                  mmap_mode="r")
                expected = np.load(safe(freeze["paths"]["parity_" + channel]),
                                   mmap_mode="r")
                exact(np.asarray(current[:, free]), expected,
                      "E238 parity free-MN stage " + channel)
            saved_final, _, _ = unwrap_neural_current_checkpoint(
                protocol.StateStore.load(safe(freeze["paths"]["parity_final"])))
            compare_parity_final(session.snapshot(), saved_final)
        else:
            exact(session.body.snapshot(), protocol.StateStore.load(
                safe(freeze["paths"]["teacher_final"])),
                "E238 teacher final physical body")
            native_packet = e113.native_arrays(native)
            if (np.any(native_packet["nonfoot_contact"])
                    or not np.isfinite(native_packet["upright"]).all()
                    or np.min(native_packet["upright"]) < .9):
                raise e113.NativeHealthStop("E238 measured teacher native health stop")
        account["status"] = "valid_matched_capture"
        account["valid"] = True
        persist()
    except e113.NativeHealthStop as error:
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
            attempt("stage-observer", lambda: save_stage_raw(arm_out, observer, names))
        if not (arm_out / "control-raw.npz").exists():
            attempt("control-raw.npz", lambda: save_control(
                arm_out / "control-raw.npz", controls))
        if not (arm_out / "native-contact.npz").exists():
            attempt("native-contact.npz", lambda: e224.save_native_contact(
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
        attempt("arm-ledger", persist)
        # No network/body owner survives into the next arm.
        observer = session = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E238 CUDA cache cleanup: " + repr(error))
    if primary is not None:
        raise primary
    return account


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze, review = gate(Path(freeze_path), freeze_sha, Path(review_path),
                          review_sha, out)
    out.mkdir(parents=False, exist_ok=False)
    limits = freeze["resource_limits"]
    ledger = {"schema": "flylab.m3.e238.matched-teacher-inputs.ledger.v1",
        "status": "gate_passed", "original_processes": 1, "retries": 0,
        "fit_calls": 0, "solver_calls": 0, "arms": [], "storage_errors": [],
        "candidate_adopted": False, "F6_exposed": False,
        "whole_goal_complete": False}
    result = {"schema": "flylab.m3.e238.matched-teacher-inputs.result.v1",
        "validity": "incomplete", "execution_status": "running",
        "parity": "not_run", "teacher": "not_run",
        "candidate_adopted": False, "F2_complete": False,
        "F6_exposed": False, "whole_goal_complete": False}
    primary = None
    threads = process = gpu_free_before = None

    def persist():
        ledger["process_wall_s"] = time.perf_counter() - began
        e224.write_receipt(out / "advance-ledger.json", ledger)

    def attempt(label, callback):
        nonlocal primary
        try:
            callback()
        except BaseException as error:
            ledger["storage_errors"].append({"artifact": label, "error": repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary E238 " + label + ": " + repr(error))

    def budget():
        if time.perf_counter() - began >= INTERNAL_S - RESERVE_S:
            raise TimeoutError("E238 finalization reserve reached")
        if process is not None:
            memory = process.memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0), peak)
            if peak > limits["resident_bytes"]:
                raise MemoryError("E238 resident memory cap exceeded")
        if gpu_free_before is not None:
            import cupy as cp
            available, _ = cp.cuda.runtime.memGetInfo()
            added = max(0, gpu_free_before - available)
            ledger["peak_added_gpu_bytes"] = max(
                ledger.get("peak_added_gpu_bytes", 0), added)
            if added > limits["allocated_gpu_bytes"]:
                raise MemoryError("E238 additional CUDA memory cap exceeded")

    try:
        e224.write_receipt(out / "source-review-snapshot.json", review)
        from tools.fit_c_banc_m3_joint_adaptive_recruitment import one_cpu_thread
        threads, ledger["threading"] = one_cpu_thread()
        import cupy as cp
        import psutil
        process = psutil.Process()
        gpu_free_before, _ = cp.cuda.runtime.memGetInfo()
        graph = protocol.GraphStore.load(safe(freeze["paths"]["graph"]))
        if graph.n != 158706 or graph.hash != freeze["graph_hash"]:
            raise ValueError("Exact E238 full BANC graph required")
        saved = protocol.StateStore.load(safe(freeze["paths"]["start"]))
        original_start, _, _ = unwrap_neural_current_checkpoint(saved)
        teacher_start = protocol.StateStore.load(safe(freeze["paths"]["teacher_start"]))
        with np.load(safe(freeze["paths"]["teacher_command"]),
                     allow_pickle=False) as packet:
            commands = packet["command"].copy()
        if (commands.dtype != np.float64 or commands.shape != (2000, 42)
                or not np.isfinite(commands).all()):
            raise ValueError("Exact finite E85 applied joint command required")
        parity_rows = [json.loads(line) for line in safe(freeze["paths"][
            "parity_trace"]).read_text(encoding="utf-8").splitlines()]
        if len(parity_rows) != 100:
            raise ValueError("E235 exact 100-row parity source required")
        tendon_names = tuple(parity_rows[0]["tendon_inputs"])
        parity_commands = np.asarray([row["joint_targets_rad"]
                                      for row in parity_rows[1:]], np.float64)
        parity_tendons = np.asarray([[row["tendon_inputs"][name]
                                     for name in tendon_names]
                                    for row in parity_rows[1:]], np.float64)
        if (parity_commands.shape != (99, 42)
                or parity_tendons.shape != (99, 8)
                or not np.isfinite(parity_commands).all()
                or not np.isfinite(parity_tendons).all()):
            raise ValueError("E235 exact parity body actions required")
        teacher_tendons = np.full((2000, 8), .2, np.float64)
        ledger["status"] = "parity_requested"
        persist()
        execute_registered_arm("parity", lambda: run_arm("parity", freeze,
            graph, original_start, parity_commands, parity_tendons, out,
            ledger, persist, budget, source_tendon_names=tendon_names),
            result, ledger)
        require_parity_before_teacher(result, ledger)
        budget()
        ledger["status"] = "teacher_requested_after_parity"
        persist()
        execute_registered_arm("teacher", lambda: run_arm("teacher", freeze,
            graph, original_start, commands, teacher_tendons, out, ledger,
            persist, budget, owner=teacher_start), result, ledger)
        result.update(validity="valid_matched_input_capture",
            execution_status="complete", controls=2099,
            neural_ticks=41980, physics_steps=104950,
            conclusion="matched teacher-body inputs and applied targets only")
    except e113.NativeHealthStop as error:
        primary = error
        result.update(validity="measured_native_health_stop", error=repr(error))
    except (TimeoutError, MemoryError) as error:
        primary = error
        result.update(validity="resource_limited", error=repr(error))
    except BaseException as error:
        primary = error
        result.update(validity="technical_or_fidelity_failure", error=repr(error),
                      traceback=traceback.format_exc())
    finally:
        if threads is not None:
            attempt("cpu-thread-restore", threads.restore_original_limits)
        ledger["process_wall_s"] = time.perf_counter() - began
        if ledger["process_wall_s"] > INTERNAL_S:
            error = TimeoutError("E238 internal wall limit exceeded")
            if primary is None:
                primary = error
                result["validity"] = "resource_limited"
            else:
                primary.add_note("Secondary E238 wall limit: " + repr(error))
        ledger["output_bytes"] = sum(path.stat().st_size for path in out.rglob("*")
                                     if path.is_file())
        if ledger["output_bytes"] > limits["output_bytes"]:
            error = OSError("E238 output cap exceeded")
            if primary is None:
                primary = error
                result["validity"] = "technical_storage_failure"
            else:
                primary.add_note("Secondary E238 output cap: " + repr(error))
        if primary is not None and result["validity"] == "valid_matched_input_capture":
            result["pre_storage_validity"] = result["validity"]
            result["validity"] = "technical_storage_failure"
        result["execution_status"] = "failed" if primary else "complete"
        if primary is not None:
            result["primary_error"] = repr(primary)
        primary = publish_final_receipts(result, ledger, primary, persist,
            lambda: e224.write_receipt(out / "result.json", result),
            lambda: e224.write_receipt(out / "output-manifest.json", manifest(out)))
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
        result = run(args.freeze, args.freeze_sha256,
                     args.review, args.review_sha256, args.out)
    except BaseException as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({"validity": result["validity"],
                      "parity": result["parity"], "teacher": result["teacher"]},
                     sort_keys=True))
    return 0 if result["validity"] == "valid_matched_input_capture" else 1


if __name__ == "__main__":
    raise SystemExit(main())
