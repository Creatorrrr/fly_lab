"""E245 passive receptor and selected-neuron observation of the E240 teacher arm."""
# Preserve primary execution faults across bounded cleanup/publication.
# ruff: noqa: BLE001
from __future__ import annotations

import argparse
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

from tools import observe_c_banc_m3_native_tendon_teacher as e240

BASE = e240.BASE
ROOT = e240.ROOT
FREEZE_SHA = "f864df608292de586f68df37afaf6fbf6ff849ecee3238424c1db7c92f4ec30b"
ROSTER_SHA = "f108afb5a709eda202137d29ce6b84c924da42aa6a12211d234b97cb612ec86b"
OWNED = ("tools/observe_c_banc_m3_passive_signals.py",
         "tests/test_c_banc_passive_signals.py")
OUT = BASE / "e245-passive-signals"
REVIEW_SCHEMA = "flylab.m3.e245.passive-signals.review.v1"
VALID = "valid_passive_signal_capture"
FLOAT_FIELDS = {
    "raw_angles_rad": (42,), "raw_velocities_rad_s": (42,),
    "support_load_bw": (6,), "non_support_load_bw": (6,),
    "knee_flexion_rad": (6,), "knee_velocity_rad_s": (6,),
    "old_filtered": (241,), "new_filtered": (241,),
    "last_features": (241,), "old_velocity_lowpass": (42,),
    "new_velocity_lowpass": (42,),
}
SIGNAL_FIELDS = {**FLOAT_FIELDS, "sensory_drive": (3604,),
                 "effective_drive": (4046,), "selected_rate": (4046,)}


def same_bits(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return (a.shape == b.shape and a.dtype == b.dtype
            and a.tobytes(order="C") == b.tobytes(order="C"))


def invalidate(result, ledger, primary, *, status=None):
    """One failure verdict for arm, result and H245 across all late paths."""
    if primary is None:
        return
    if result["validity"] == VALID:
        result["pre_failure_validity"] = VALID
        result["validity"] = status or "technical_cleanup_or_storage_failure"
    result.update(execution_status="failed", primary_error=repr(primary),
                  H245="not_tested")
    arm = ledger.get("arms", [])
    if arm:
        account = arm[-1]
        if account.get("valid") is True:
            account["pre_E245_validation_status"] = account.get("status")
        account.update(valid=False, status=result["validity"])
    result["teacher"] = result["validity"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate(freeze_path, freeze_sha, review_path, review_sha, out):
    expected = BASE / "e245-passive-signal-observation-freeze.json"
    if (Path(freeze_path).resolve() != expected.resolve()
            or freeze_sha != FREEZE_SHA or digest(expected) != FREEZE_SHA
            or digest(review_path) != review_sha):
        raise ValueError("Exact E245 freeze/review SHA required")
    freeze = json.loads(expected.read_text("utf-8"))
    review = json.loads(Path(review_path).read_text("utf-8"))
    required_limits = {"internal_s": 170, "finalization_reserve_s": 15,
                       "external_s": 185, "supervisor_s": 200,
                       "cpu_threads": 1, "gpu_count": 1,
                       "original_attempts": 1, "retries": 0,
                       "body_constructors": 1, "nominal_neural_constructors": 7,
                       "fit_calls": 0, "solver_calls": 0,
                       "resident_bytes": 6 << 30, "allocated_gpu_bytes": 4 << 30,
                       "output_bytes": 2 << 30, "minimum_free_bytes": 3 << 30}
    if (freeze.get("schema") != "flylab.m3.e245.passive-signals.freeze.v1"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("roster_sha256") != ROSTER_SHA
            or freeze.get("design_sha256") != digest(ROOT / freeze["design"])
            or freeze.get("native") != {
                "nodes": 158706, "motors": 391, "parity_free_motors": 343,
                "neural_dt_s": .00025, "control_dt_s": .005,
                "physics_dt_s": .0001, "neural_ticks_per_control": 20,
                "rk_stages_per_tick": 4, "physics_steps_per_control": 50}
            or freeze.get("arms") != {"teacher": {"controls": 2000,
                "neural_ticks": 40000, "physics_steps": 100000, "stages": 160000}}
            or freeze.get("expected_construction") != e240.original.CONSTRUCTION
            or any(freeze.get("resource_limits", {}).get(k) != v
                   for k, v in required_limits.items())
            or any(freeze.get(k) is not False for k in
                   ("candidate_adopted", "F2_complete", "F6_exposed",
                    "whole_goal_complete", "active_UI_changed"))
            or review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Reviewed E245 passive observation contract required")
    for record, labels in ((freeze, ("input_sha256", "reused_source_sha256",
                                     "preservation_sha256")),
                           (review, ("source_sha256", "additional_reused_source_sha256",
                                     "additional_input_sha256"))):
        for label in labels:
            for name, expected_hash in record.get(label, {}).items():
                if digest(e240.original.safe(name)) != expected_hash:
                    raise ValueError("Frozen E245 input/source changed: " + name)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Frozen E245 package changed: " + name)
    if (out.resolve() != OUT.resolve() or out.exists() or not out.parent.is_dir()
            or shutil.disk_usage(out.parent).free <
            freeze["resource_limits"]["minimum_free_bytes"]):
        raise ValueError("New E245 output and free disk required")
    roster_path = e240.original.safe(freeze["paths"]["roster"])
    if digest(roster_path) != ROSTER_SHA:
        raise ValueError("Frozen E245 roster changed")
    roster = json.loads(roster_path.read_text("utf-8"))
    if (roster.get("schema") != "flylab.m3.e245.passive-signal-roster.v1"
            or roster.get("graph_hash") != freeze["graph_hash"]
            or roster.get("counts") != freeze["roster_counts"]):
        raise ValueError("E245 roster schema/identity changed")
    proof = json.loads(e240.original.safe(freeze["paths"]["reference_proof"]).read_text("utf-8"))
    result = json.loads(e240.original.safe(freeze["paths"]["reference_result"]).read_text("utf-8"))
    if (proof.get("capture_validity") != "independently_valid"
            or result.get("validity") != e240.VALID
            or result.get("candidate_adopted") is not False):
        raise ValueError("Independently valid E240 reference required")
    return freeze, review, roster


def _copy(value, shape, dtype=np.float64):
    out = np.asarray(value)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError("Finite passive signal shape required: " + str(shape))
    return out.astype(dtype, copy=True)


class SignalCollector:
    """Copies only values returned by the actual encode and neural calls."""

    def __init__(self, roster, reference_controls=None, reference_rows=None,
                 stage_reference=None, prefix_path=None):
        self.roster = roster
        self.sensory = np.asarray(roster["sensory_node_rows"], np.int64)
        self.selected = np.asarray(roster["selected_node_rows"], np.int64)
        self.motor_columns = np.asarray(roster["motor_selected_columns"], np.int64)
        self.motors = np.asarray(roster["motor_node_rows"], np.int64)
        self.rows = []
        self.session = self.leaf = self.observer = None
        self._bindings = []
        self._pending = None
        self._calls = {"leg": 0, "knee": 0, "encode": 0, "neural": 0, "body": 0}
        self.reference_controls = reference_controls
        self.reference_rows = reference_rows
        self.stage_reference = stage_reference
        self.prefix_path = prefix_path
        self.prefix_verified = False
        self.failed_control = None

    def _bind(self, obj, name, replacement):
        own = vars(obj)
        self._bindings.append((obj, name, name in own, own.get(name)))
        setattr(obj, name, replacement)

    def restore(self):
        errors = []
        for obj, name, had, value in reversed(self._bindings):
            try:
                if had:
                    setattr(obj, name, value)
                else:
                    delattr(obj, name)
            except BaseException as error:
                errors.append(error)
        self._bindings.clear()
        if errors:
            for error in errors[1:]:
                errors[0].add_note("Secondary E245 binding restore: " + repr(error))
            raise errors[0]

    def install_session(self, session):
        self.session = session
        leaf = session.adapter.original.original
        if not hasattr(leaf, "filtered") or not hasattr(leaf, "velocity_lowpass"):
            raise ValueError("Original RateBodyAdapter leaf required")
        self.leaf = leaf
        if (not np.array_equal(leaf._sensory_ids, self.sensory)
                or not np.array_equal(leaf._sensory_ports,
                                      np.asarray(self.roster["sensory_ports"]))
                or not np.array_equal(leaf._port_features,
                                      np.asarray(self.roster["port_features"]))
                or [list(key) for key in leaf._receptor_keys] != self.roster["receptor_keys"]
                or len(leaf.ports) != 241
                or leaf.identity != self.roster["adapter_hash"]):
            raise ValueError("E245 actual sensory port roster changed")
        for port, record in zip(leaf.ports, self.roster["ports"], strict=True):
            leg, kind, cell, ids = port
            if (leg != record["leg_index"] or kind != record["kind"]
                    or cell != record["cell_type"]
                    or not np.array_equal(ids, record["node_rows"])):
                raise ValueError("E245 actual port anatomy changed")
        body = session.body
        old_leg, old_knee, old_encode = (body.leg_observation,
                                         body.knee_kinematics, leaf.encode)

        def leg():
            value = old_leg()
            if self._pending is not None:
                self._calls["leg"] += 1
                for key, shape, target in (("angles_rad", (42,), "raw_angles_rad"),
                                            ("velocities_rad_s", (42,), "raw_velocities_rad_s"),
                                            ("support_load_bw", (6,), "support_load_bw"),
                                            ("non_support_load_bw", (6,), "non_support_load_bw")):
                    self._pending[target] = _copy(value[key], shape)
            return value

        def knee(*args, **kwargs):
            value = old_knee(*args, **kwargs)
            if self._pending is not None:
                if args or kwargs != {"include_direction": False}:
                    raise ValueError("Original no-direction knee observation required")
                self._calls["knee"] += 1
                self._pending["knee_flexion_rad"] = _copy(value["flexion_rad"], (6,))
                self._pending["knee_velocity_rad_s"] = _copy(value["velocity_rad_s"], (6,))
            return value

        def encode(*args, **kwargs):
            if self._pending is not None:
                raise RuntimeError("Overlapping E245 encoder calls")
            index = int(session.control_tick) + 1
            if index != len(self.rows):
                raise ValueError("E245 encoder/control order changed")
            row = self._blank(index)
            row["old_filtered"] = _copy(leaf.filtered, (241,))
            row["old_velocity_lowpass"] = _copy(leaf.velocity_lowpass, (42,))
            self.rows.append(row)
            self._pending = row
            before = self._calls.copy()
            try:
                result = old_encode(*args, **kwargs)
                if (self._calls["leg"] != before["leg"] + 1
                        or self._calls["knee"] != before["knee"] + 1):
                    raise ValueError("Original sensory observation not called exactly once")
                row["sensory_drive"] = _copy(np.asarray(result)[self.sensory],
                                              (3604,), np.float32)
                row["new_filtered"] = _copy(leaf.filtered, (241,))
                row["last_features"] = _copy(leaf.last_features, (241,))
                row["new_velocity_lowpass"] = _copy(leaf.velocity_lowpass, (42,))
                row["encode_valid"] = True
                self._calls["encode"] += 1
                return result
            finally:
                self._pending = None

        self._bind(body, "leg_observation", leg)
        self._bind(body, "knee_kinematics", knee)
        self._bind(leaf, "encode", encode)

    def install_observer(self, observer):
        if self.session is None or self.observer is not None:
            raise RuntimeError("E245 observer ownership changed")
        self.observer = observer
        if (observer is not self.session.network
                or not np.array_equal(observer.core_ids,
                                      np.asarray(self.roster["core_adaptation_node_rows"]))
                or not np.array_equal(observer.motor_ids, self.motors)
                or observer.core_gain != 0.):
            raise ValueError("E245 native neural roster changed")
        if not np.array_equal(observer.observation_ids, self.motors):
            raise ValueError("E245 ordered MN stage observation changed")
        old = observer.advance_readout

        def advance(drive, indices, steps=10):
            index = int(self.session.control_tick) + 1
            if index != len(self.rows) - 1 or not self.rows[index]["encode_valid"]:
                raise ValueError("E245 neural/encode order changed")
            if steps != 20 or not np.array_equal(indices, self.motors):
                raise ValueError("E245 original MN readout request changed")
            row = self.rows[index]
            row["effective_drive"] = _copy(np.asarray(drive)[self.selected],
                                            (4046,), np.float32)
            output = old(drive, indices, steps)
            row["selected_rate"] = _copy(observer.readout(self.selected),
                                          (4046,), np.float32)
            if not same_bits(row["selected_rate"][self.motor_columns], output):
                raise ValueError("E245 passive selected MN return differs")
            row["network_tick"] = int(observer.tick)
            row["neural_valid"] = True
            self._calls["neural"] += 1
            return output

        self._bind(observer, "advance_readout", advance)
        first = self._blank(0)
        first["old_filtered"] = _copy(self.leaf.filtered, (241,))
        first["new_filtered"] = first["old_filtered"].copy()
        first["old_velocity_lowpass"] = _copy(self.leaf.velocity_lowpass, (42,))
        first["new_velocity_lowpass"] = first["old_velocity_lowpass"].copy()
        first["last_features"] = _copy(self.leaf.last_features, (241,))
        first["sensory_drive"] = _copy(observer.drive[self.sensory].cpu().numpy(),
                                       (3604,), np.float32)
        first["effective_drive"] = _copy(observer.drive[self.selected].cpu().numpy(),
                                         (4046,), np.float32)
        first["selected_rate"] = _copy(observer.readout(self.selected),
                                       (4046,), np.float32)
        first["network_tick"] = int(observer.tick)
        first["physics_time_s"] = float(self.session.body.physics_time())
        self.rows.append(first)

    @staticmethod
    def _blank(index):
        row = {"control_tick": index, "network_tick": -1,
               "physics_time_s": np.nan, "encode_valid": False,
               "neural_valid": False, "body_valid": False,
               "parity_valid": False}
        for field, shape in SIGNAL_FIELDS.items():
            dtype = np.float32 if field in ("sensory_drive", "effective_drive",
                                            "selected_rate") else np.float64
            row[field] = np.full(shape, np.nan, dtype)
        return row

    def completed_control(self, row, record):
        index = int(record["control_tick"])
        if (index != len(self.rows) - 1 or index < 1
                or not self.rows[index]["neural_valid"]
                or self.rows[index]["network_tick"] != int(record["network_tick"])):
            raise ValueError("E245 post-neural/body control order changed")
        self.rows[index]["physics_time_s"] = float(record["physics_time_s"])
        self.rows[index]["body_valid"] = True
        self._calls["body"] += 1
        try:
            if self.reference_controls is not None:
                for field in ("control_tick", "network_tick", "physics_time_s",
                              "motor_rate32", "applied_command", "applied_tendon",
                              "applied_adhesion", "decoded_command", "decoded_tendon"):
                    e240.original.exact(self.reference_controls[field][index],
                                        record[field], "E245 reference control " + field)
            if self.reference_rows is not None:
                recorded = e240.original.e224.finite_receipt(
                    e240.original.protocol.jsonable(row))[0]
                e240.original.exact(self.reference_rows[index], recorded,
                                    "E245 reference trace row")
            if index == 2:
                self.verify_first160()
            self.rows[index]["parity_valid"] = True
        except BaseException as error:
            self.failed_control = {"control_tick": index, "error": repr(error),
                                   "body_observed": True, "parity_valid": False}
            for label, actual in (("actual_trace_row", row),
                                  ("actual_control_record", record)):
                try:
                    self.failed_control[label] = e240.original.e224.finite_receipt(
                        e240.original.protocol.jsonable(actual))[0]
                except BaseException as secondary:
                    self.failed_control[label + "_capture_error"] = repr(secondary)
                    error.add_note("Secondary E245 failing row capture: " + repr(secondary))
            raise

    def verify_first160(self):
        """Read only completed existing graph buffers after blocking readout."""
        if self.stage_reference is None:
            return
        if self.prefix_verified or self.prefix_path is None or self.observer is None:
            raise RuntimeError("E245 first160 prefix ownership changed")
        observer = self.observer
        channels = {"positive_recurrent": observer._positive_values,
                    "negative_magnitude": observer._negative_values,
                    "drive": observer._drive_values}
        values = {key: tensor[:160].cpu().numpy().copy()
                  for key, tensor in channels.items()}
        values["counters"] = observer._observed_counters.cpu().numpy().copy()
        values["motor_ids"] = observer.observation_ids.copy()
        temp = self.prefix_path.with_name(self.prefix_path.name + ".tmp.npz")
        np.savez_compressed(temp, **values)
        temp.replace(self.prefix_path)
        if (not same_bits(values["motor_ids"], self.motors)
                or values["counters"].shape != (391,)
                or values["counters"].dtype != np.int64
                or not np.array_equal(values["counters"], np.full(391, 160, np.int64))):
            raise ValueError("E245 first160 stage ID/counter changed")
        for channel, expected in self.stage_reference.items():
            if not same_bits(values[channel], expected):
                raise ValueError("E245 first160 stage prefix differs: " + channel)
        self.prefix_verified = True

    def save(self, path):
        if not self.rows:
            return
        fields = {"control_tick": np.int64, "network_tick": np.int64,
                  "physics_time_s": np.float64, "encode_valid": bool,
                  "neural_valid": bool, "body_valid": bool,
                  "parity_valid": bool}
        fields.update({key: (np.float32 if key in
                        ("sensory_drive", "effective_drive", "selected_rate")
                        else np.float64) for key in SIGNAL_FIELDS})
        arrays = {key: np.asarray([row[key] for row in self.rows], dtype=dtype)
                  for key, dtype in fields.items()}
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)


def validate_signal(saved_path, collector, start, final, reference_control):
    with np.load(saved_path, allow_pickle=False) as data:
        if (len(collector.rows) != 2001
                or not np.array_equal(data["control_tick"], np.arange(2001))
                or data["encode_valid"][0] or data["neural_valid"][0]
                or data["body_valid"][0]
                or not all(np.all(data[key][1:]) for key in
                           ("encode_valid", "neural_valid", "body_valid",
                            "parity_valid"))
                or not np.array_equal(data["network_tick"], np.arange(2001) * 20)
                or np.max(np.abs(data["physics_time_s"] - np.arange(2001) * .005)) > 2e-8):
            raise ValueError("E245 complete control/encode/neural/body clock required")
        for key, shape in SIGNAL_FIELDS.items():
            expected = (2001, *shape)
            dtype = np.float32 if key in ("sensory_drive", "effective_drive",
                                          "selected_rate") else np.float64
            if (data[key].shape != expected or data[key].dtype != dtype
                    or not np.isfinite(data[key][1:]).all()):
                raise ValueError("E245 signal shape/dtype/finite changed: " + key)
        for key in ("old_filtered", "new_filtered", "old_velocity_lowpass",
                    "new_velocity_lowpass", "last_features", "effective_drive",
                    "selected_rate", "sensory_drive"):
            if not np.isfinite(data[key][0]).all():
                raise ValueError("E245 initial signal unavailable: " + key)
        if (not np.array_equal(data["new_filtered"][:-1],
                               data["old_filtered"][1:])
                or not np.array_equal(data["new_velocity_lowpass"][:-1],
                                      data["old_velocity_lowpass"][1:])):
            raise ValueError("E245 filter old/new causality changed")
        leaf_start = start["adapter"]["original"]["original"]
        leaf_final = final["adapter"]["original"]["original"]
        for field, raw in (("filtered", "new_filtered"),
                           ("velocity_lowpass", "new_velocity_lowpass"),
                           ("last_features", "last_features")):
            e240.original.exact(leaf_start[field], data[raw][0], "E245 initial " + field)
            e240.original.exact(leaf_final[field], data[raw][-1], "E245 final " + field)
        e240.original.exact(start["network"]["rate"][collector.selected],
                            data["selected_rate"][0], "E245 initial selected rate")
        e240.original.exact(final["network"]["rate"][collector.selected],
                            data["selected_rate"][-1], "E245 final selected rate")
        e240.original.exact(start["network"]["drive"][collector.selected],
                            data["effective_drive"][0], "E245 initial effective drive")
        e240.original.exact(final["network"]["drive"][collector.selected],
                            data["effective_drive"][-1], "E245 final effective drive")
        if not same_bits(data["selected_rate"][:, collector.motor_columns],
                         reference_control["motor_rate32"]):
            raise ValueError("E245 all MN rate bits differ")


def publish_receipts(result, ledger, primary, write_ledger, write_result,
                     write_manifest):
    """Late publication failure cancels a successful passive observation."""
    def failure(label, error):
        nonlocal primary
        ledger["storage_errors"].append({"artifact": label, "error": repr(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E245 " + label + ": " + repr(error))
        invalidate(result, ledger, primary, status="technical_storage_failure")

    for _ in range(2):
        before = len(ledger["storage_errors"])
        for name, callback in (("advance-ledger.json", write_ledger),
                               ("result.json", write_result),
                               ("output-manifest.json", write_manifest)):
            try:
                callback()
            except BaseException as error:
                failure(name, error)
        if len(ledger["storage_errors"]) == before:
            break
    return primary


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze, review, roster = gate(freeze_path, freeze_sha, review_path,
                                  review_sha, out)
    out.mkdir(exist_ok=False)
    limits = freeze["resource_limits"]
    ledger = {"schema": "flylab.m3.e245.passive-signals.ledger.v1",
              "status": "source_gate_passed", "original_processes": 1,
              "retries": 0, "fit_calls": 0, "solver_calls": 0,
              "arms": [], "storage_errors": [], "candidate_adopted": False,
              "F6_exposed": False, "whole_goal_complete": False}
    result = {"schema": "flylab.m3.e245.passive-signals.result.v1",
              "validity": "incomplete", "execution_status": "running",
              "teacher": "not_run", "H245": "not_tested",
              "candidate_adopted": False, "F2_complete": False,
              "F6_exposed": False, "whole_goal_complete": False}
    primary = threads = process = gpu_before = collector = None

    def persist():
        ledger["process_wall_s"] = time.perf_counter() - began
        e240.original.e224.write_receipt(out / "advance-ledger.json", ledger)

    def budget():
        elapsed = time.perf_counter() - began
        if elapsed >= limits["internal_s"] - limits["finalization_reserve_s"]:
            raise TimeoutError("E245 finalization reserve reached")
        if process is not None:
            memory = process.memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0), peak)
            if peak > limits["resident_bytes"]:
                raise MemoryError("E245 resident limit exceeded")
        if gpu_before is not None:
            import cupy as cp
            available, _ = cp.cuda.runtime.memGetInfo()
            added = max(0, gpu_before - available)
            ledger["peak_added_gpu_bytes"] = max(ledger.get("peak_added_gpu_bytes", 0), added)
            if added > limits["allocated_gpu_bytes"]:
                raise MemoryError("E245 additional GPU limit exceeded")

    # Frozen callbacks are temporarily intercepted only inside this process.
    protocol = e240.original.protocol
    old_profile = protocol.profiled_construction
    old_record = e240.original.control_record
    callbacks_installed = False
    try:
        e240.original.e224.write_receipt(out / "source-review-snapshot.json", review)
        from tools.fit_c_banc_m3_joint_adaptive_recruitment import one_cpu_thread
        threads, ledger["threading"] = one_cpu_thread()
        import cupy as cp
        import psutil
        process = psutil.Process()
        gpu_before, _ = cp.cuda.runtime.memGetInfo()
        graph = protocol.GraphStore.load(e240.original.safe(freeze["paths"]["graph"]))
        if graph.hash != freeze["graph_hash"] or graph.n != 158706:
            raise ValueError("Original whole BANC graph required")
        saved = protocol.StateStore.load(e240.original.safe(freeze["paths"]["start"]))
        start, _, _ = e240.original.unwrap_neural_current_checkpoint(saved)
        teacher_start = protocol.StateStore.load(e240.original.safe(freeze["paths"]["teacher_start"]))
        matched_start = protocol.StateStore.load(e240.original.safe(freeze["paths"]["matched_start"]))
        reference_start = protocol.StateStore.load(
            e240.original.safe(freeze["paths"]["reference_teacher"]) / "observed-start")
        e240.original.exact(start, reference_start, "E245 E240 start")
        reference_rows = [json.loads(line) for line in
                          (e240.original.safe(freeze["paths"]["reference_teacher"]) /
                           "trace.jsonl").read_text("utf-8").splitlines()]
        if len(reference_rows) != 2001:
            raise ValueError("Complete E240 trace required")
        with np.load(e240.original.safe(freeze["paths"]["reference_teacher"]) /
                     "control-raw.npz", allow_pickle=False) as packet:
            reference_control = {key: packet[key].copy() for key in packet.files}
        with np.load(e240.original.safe(freeze["paths"]["teacher_command"]),
                     allow_pickle=False) as packet:
            commands = packet["command"].copy()
        if (commands.shape != (2000, 42) or commands.dtype != np.float64
                or not np.isfinite(commands).all()):
            raise ValueError("Frozen E85 commands required")
        with np.load(e240.original.safe(freeze["paths"]["matched_control"]),
                     allow_pickle=False) as packet:
            matched_first = {key: packet[key][:2].copy() for key in
                             ("control_tick", "network_tick", "physics_time_s",
                              "motor_rate32")}
        stage_reference = {key: np.asarray(np.load(
            e240.original.safe(freeze["paths"]["reference_teacher"]) /
            ("stage-" + key + ".npy"), mmap_mode="r")[:160]).copy()
            for key in e240.original.CHANNELS}
        collector = SignalCollector(roster, reference_control, reference_rows,
            stage_reference, out / "teacher" / "stage-prefix.npz")

        def profiled(label, constructor, construction, save, *, on_return=None):
            def received(value):
                if on_return is not None:
                    on_return(value)
                if label == "restore_original":
                    collector.install_session(value)
                elif label == "construct_observer":
                    collector.install_observer(value)
            return old_profile(label, constructor, construction, save,
                               on_return=received)

        def record(session, action, names):
            row, value = old_record(session, action, names)
            collector.completed_control(row, value)
            return row, value

        callbacks_installed = True
        protocol.profiled_construction = profiled
        e240.original.control_record = record
        ledger["status"] = "teacher_requested"
        persist()
        e240.original.execute_registered_arm("teacher", lambda: e240.run_arm(
            freeze, graph, start, teacher_start, commands, matched_start,
            matched_first, out, ledger, persist, budget), result, ledger)
        # E240 closes the body before returning. Only saved evidence is read below.
        budget()
        signal_path = out / "teacher" / "signal-raw.npz"
        collector.save(signal_path)
        observed_start = protocol.StateStore.load(out / "teacher" / "observed-start")
        e240.original.exact(reference_start, observed_start, "E245 E240 whole initial")
        final = protocol.StateStore.load(out / "teacher" / "final")
        reference_final = protocol.StateStore.load(
            e240.original.safe(freeze["paths"]["reference_teacher"]) / "final")
        e240.original.exact(reference_final, final, "E245 E240 whole final")
        validate_signal(signal_path, collector, reference_start, final, reference_control)
        current = {key: np.load(out / "teacher" / ("stage-" + key + ".npy"),
                                mmap_mode="r") for key in e240.original.CHANNELS}
        for key, array in current.items():
            reference = np.load(e240.original.safe(freeze["paths"]["reference_teacher"]) /
                                ("stage-" + key + ".npy"), mmap_mode="r")
        if (array.shape != (160000, 391) or array.dtype != np.float32
                    or not same_bits(array[:160], reference[:160])
                    or not np.isfinite(array).all()):
                raise ValueError("E245 first two controls of MN stage input changed: " + key)
        if not collector.prefix_verified:
            raise ValueError("E245 first160 stage not verified before control3")
        stage_files = {key: {"sha256": digest(out / "teacher" /
                                              ("stage-" + key + ".npy")),
                             "shape": list(array.shape), "dtype": str(array.dtype)}
                       for key, array in current.items()}
        with np.load(out / "teacher" / "control-raw.npz", allow_pickle=False) as packet:
            for key, expected in reference_control.items():
                e240.original.exact(packet[key], expected, "E245 complete controls " + key)
        with (np.load(out / "teacher" / "native-contact.npz", allow_pickle=False) as packet,
              np.load(e240.original.safe(freeze["paths"]["reference_teacher"]) /
                      "native-contact.npz", allow_pickle=False) as expected):
            for key in expected.files:
                e240.original.exact(packet[key], expected[key], "E245 native " + key)
        trace_path = out / "teacher" / "trace.jsonl"
        if digest(trace_path) != digest(
                e240.original.safe(freeze["paths"]["reference_teacher"]) / "trace.jsonl"):
            raise ValueError("E245 complete trace differs from E240")
        leaf = collector.leaf
        body = collector.session.body
        metadata = {"schema": "flylab.m3.e245.passive-signals.metadata.v1",
                    "roster_sha256": ROSTER_SHA,
                    "selected_node_rows": roster["selected_node_rows"],
                    "sensory_node_rows": roster["sensory_node_rows"],
                    "motor_node_rows": roster["motor_node_rows"],
                    "sensory_selected_columns": roster["sensory_selected_columns"],
                    "motor_selected_columns": roster["motor_selected_columns"],
                    "leaf_joints": np.asarray(leaf.joints).tolist(),
                    "joint_columns": np.asarray(leaf.joints).tolist(),
                    "joint_names": list(body.joint_names),
                    "qpos_ids": np.asarray(body.qpos_ids).tolist(),
                    "qvel_ids": np.asarray(body.qvel_ids).tolist(),
                    "sensory_gain": float(leaf.sensory_gain),
                    "load_gain_multiplier": float(leaf.load_gain_multiplier),
                    "front_hair_plate": bool(leaf.front_hair_plate),
                    "load_half_bw": float(leaf.spec["load_half_bw"]),
                    "touch_half_bw": float(leaf.spec["touch_half_bw"]),
                    "receptor_keys": [list(key) for key in leaf._receptor_keys],
                    "port_features": leaf._port_features.tolist(),
                    "sensory_ports": leaf._sensory_ports.tolist(),
                    "encode_timing": "control c uses c-1 end body and old filter; new filter feeds c+1",
                    "neural_timing": "post-neural/pre-physics selected rate at control c",
                    "body_timing": "trace c follows current command and 50 native physics steps",
                    "row0": "initial state; no encode, neural advance, or decoder"}
        metadata["stage_files"] = stage_files
        metadata["core_selection_interpretation"] = (
            "Historical core_adaptation selection, not 442 effective gain-zero cells "
            "and not a biological CPG identity")
        e240.original.e224.write_receipt(out / "teacher" / "signal-metadata.json", metadata)
        output_bytes = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
        ledger["output_bytes_before_receipts"] = output_bytes
        if output_bytes > limits["output_bytes"]:
            raise MemoryError("E245 output byte limit exceeded")
        result.update(validity=VALID, execution_status="complete", teacher=VALID,
                      H245="supported_for_this_exact_teacher_run",
                      conclusion="Passive observation only; no gait or biological mechanism verdict")
        ledger["status"] = "complete_exact_passive_capture"
    except BaseException as error:
        primary = error
        label = ("measured_native_health_stop" if isinstance(error, e240.original.e113.NativeHealthStop)
                 else "resource_limited" if isinstance(error, (TimeoutError, MemoryError))
                 else "technical_or_fidelity_failure")
        result.update(validity=label, traceback=traceback.format_exc())
        invalidate(result, ledger, primary, status=label)
    finally:
        def secondary(label, callback):
            nonlocal primary
            try:
                callback()
            except BaseException as error:
                ledger["storage_errors"].append({"artifact": label, "error": repr(error)})
                if primary is None:
                    primary = error
                else:
                    primary.add_note("Secondary E245 " + label + ": " + repr(error))
        if callbacks_installed:
            protocol.profiled_construction = old_profile
            e240.original.control_record = old_record
        if collector is not None:
            secondary("restore_instance_bindings", collector.restore)
            path = out / "teacher" / "signal-raw.npz"
            if not path.exists():
                secondary("partial_signal_raw", lambda: collector.save(path))
            if collector.failed_control is not None:
                secondary("failing_control_row", lambda: e240.original.e224.write_receipt(
                    out / "teacher" / "parity-failure.json",
                    collector.failed_control))
            ledger["passive_calls"] = collector._calls.copy()
            ledger["passive_rows"] = len(collector.rows)
            ledger["first160_verified_before_control3"] = collector.prefix_verified
        if threads is not None:
            secondary("thread_cleanup", threads.restore_original_limits)
        elapsed = time.perf_counter() - began
        ledger["process_wall_s"] = elapsed
        if elapsed > limits["internal_s"] and primary is None:
            primary = TimeoutError("E245 internal including publication exceeded")
        invalidate(result, ledger, primary)
        primary = publish_receipts(result, ledger, primary, persist,
            lambda: e240.original.e224.write_receipt(out / "result.json", result),
            lambda: e240.original.e224.write_receipt(out / "output-manifest.json",
                e240.original.manifest(out)))
        if time.perf_counter() - began > limits["internal_s"] and primary is None:
            primary = TimeoutError("E245 publication exceeded internal limit")
            if result["validity"] == VALID:
                result["validity"] = "resource_limited"
            invalidate(result, ledger, primary, status="resource_limited")
            publish_receipts(result, ledger, primary, persist,
                lambda: e240.original.e224.write_receipt(out / "result.json", result),
                lambda: e240.original.e224.write_receipt(out / "output-manifest.json",
                    e240.original.manifest(out)))
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "freeze-sha256", "review", "review-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        value = run(args.freeze, args.freeze_sha256, args.review,
                    args.review_sha256, args.out)
    except BaseException as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(value, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
