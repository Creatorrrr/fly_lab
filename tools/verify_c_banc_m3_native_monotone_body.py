"""E150 one-shot native body check for the frozen monotone recruitment adapter."""

# Broad catches retain first failures while independently persisting partial raw,
# construction, clock, cleanup, and storage evidence.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import banc_m3_monotone_recruitment as monotone
from tools import continue_c_banc_m3_native_timescale_body as e142
from tools import verify_c_banc_m3_native_timescale_transfer as protocol

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
FREEZE_SHA = "688dcf37366d664ee2a10d9f567e817974fb9fc3260ecb195cc9ab23f0fc75fa"
DESIGN_SHA = "95fc313423437a9a48a551d4d5df7b32c1632ac82f57808579f69393bce31ef5"
OWNED = (
    "tools/verify_c_banc_m3_native_monotone_body.py",
    "tests/test_c_banc_native_monotone_body.py",
)
OVERRIDES = ("MotorTimescaleTrial", "_observe", "_save_body_raw", "INTERNAL_LIMIT_S")
TENDON_KEYS = tuple(leg + "_tarsus" for leg in ("lf", "lm", "lh", "rf", "rm", "rh"))
# Parent post-native correction: the 96-muscle adapter owns six LTM outputs.
# BancWalkingSession also retains the original two constant abdomen holds.
# The executed E150 source and incomplete cut result are archived unchanged.
NON_MOTOR_TENDON_HOLDS = {"abdomen_pitch": 0.2, "abdomen_yaw": 0.2}
NATIVE = {
    "neurons": 158706,
    "main_controls": 2000,
    "clone_controls": 20,
    "neural_steps_per_control": 20,
    "physics_steps_per_control": 50,
    "neural_dt_s": 0.00025,
    "physics_dt_s": 0.0001,
    "control_dt_s": 0.005,
    "actual_neural_s": 10.2,
    "actual_physical_s": 10.2,
    "neural_ticks": 40800,
    "physics_steps": 102000,
    "checkpoint_control": 200,
    "compare_control": 220,
    "development_window_s": [0.5, 10.0],
    "cut_controls": 4,
    "restart_controls": 16,
}
REQUIRED_OUTPUTS = (
    "advance-ledger.json",
    "source-review-snapshot.json",
    "spec.json",
    "candidate-neutral/manifest.json",
    "wrapper-provenance.json",
    "native-recruitment-state.npz",
    "cut-restart-ledger.json",
    "cut-restart-result.json",
    "E142-E150-same-window-comparison.json",
)


def _safe_path(relative):
    name = Path(relative)
    path = (ROOT / name).resolve()
    if name.is_absolute() or ".." in name.parts or not path.is_relative_to(ROOT):
        raise ValueError("E150 repository-relative path required")
    return path


def verified_inputs(freeze_path, freeze_sha, review_path, review_sha):
    """Verify every immutable byte and the parent source gate before model entry."""
    if freeze_sha != FREEZE_SHA or protocol.sha256(freeze_path) != FREEZE_SHA:
        raise ValueError("Exact E150 freeze required before model entry")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    fixed = {
        "schema": "flylab.m3.e150.native-monotone.v1",
        "design_sha256": DESIGN_SHA,
        "body_protocol": "E141-code-with-isolated-factory",
        "owned_files": list(OWNED),
        "prior_conservative_process_s": 2229,
        "prepare_wall_limit_s": 60,
        "body_wall_limit_s": 180,
        "internal_wall_limit_s": 250,
        "external_wall_limit_s": 270,
        "supervisor_limit_s": 285,
        "prospective_cumulative_limit_s": 2514,
        "original_program_cumulative_limit_s": 2514,
        "original_processes": 1,
        "automatic_retries": 0,
        "cpu_threads": 1,
        "gpu_count": 1,
        "maximum_raw_bytes": 4 << 30,
        "minimum_free_disk_bytes": 12 << 30,
        "native": NATIVE,
        "conditional_forward_calls": 0,
        "fit_calls": 0,
        "LP_calls": 0,
        "F6_exposed": False,
        "active_UI_changed": False,
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "allowed_protocol_global_overrides": list(OVERRIDES),
        "strict_curve_inclusion_atol": 1e-12,
        "curve_parameter_selection": False,
        "prior_conservative_actual_s": 2228.8917264000047,
        "previous_cumulative_allocation_s": 2271,
    }
    if (any(freeze.get(key) != value for key, value in fixed.items()) or
            freeze.get("design_sha256") != protocol.sha256(
                _safe_path(freeze.get("design_path", ""))) or
            len(freeze.get("input_sha256", {})) != 640 or
            len(freeze.get("reused_source_sha256", {})) != 84 or
            len(freeze.get("preservation_sha256", {})) != 2013):
        raise ValueError("Frozen E150 execution contract changed")
    if protocol.sha256(review_path) != review_sha:
        raise ValueError("Exact E150 parent source review required")
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (review.get("accepts_single_native_monotone") is not True or
            review.get("blocking_defects") != [] or
            review.get("freeze_sha256") != FREEZE_SHA or
            set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Accepted exact E150 two-source gate required")
    for name, digest in review["source_sha256"].items():
        if protocol.sha256(_safe_path(name)) != digest:
            raise ValueError("E150 reviewed source changed: " + name)
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        protocol._exact_mapping_hashes(freeze[label], label)
    return freeze, review


def load_parameters(path):
    value = protocol.load_npz(path)
    expected = {"tau": ((3,), np.float64), "kappa": ((391,), np.float64),
        "slopes": ((96, 391, 3, 2), np.float64),
        "membership": ((391,), np.int64), "motor_ids": ((391,), np.int64)}
    if set(value) != set(expected):
        raise ValueError("Complete frozen E150 parameter bundle required")
    for key, (shape, dtype) in expected.items():
        if (value[key].shape != shape or value[key].dtype != dtype or
                not np.isfinite(value[key]).all()):
            raise ValueError("Invalid E150 parameter array: " + key)
    return value


class TrialFactory:
    """Factory accepted by the unchanged E141 protocol's classmethod call site."""

    def __init__(self, parameters, *, base_cls=protocol.MotorTimescaleTrial,
                 wrapper_cls=monotone.MonotoneRecruitmentTrial):
        self.parameters = parameters
        self.base_cls = base_cls
        self.wrapper_cls = wrapper_cls
        self.source_restores = 0
        self.wrapper_restores = 0
        self.source_checks = []

    def from_checkpoint(self, graph, saved):
        if saved.get("schema") == monotone.TRIAL_SCHEMA:
            self.wrapper_restores += 1
            return self.wrapper_cls.from_checkpoint(graph, saved)
        self.source_restores += 1
        base = self.base_cls.from_checkpoint(graph, saved)
        candidate = None
        try:
            source = base.snapshot()
            protocol.exact(saved, source, "E150 exact source trial restoration")
            before_csr = csr_fingerprint(base.network)
            candidate = self.wrapper_cls(base, self.parameters["tau"],
                self.parameters["kappa"], self.parameters["slopes"],
                self.parameters["membership"])
            verify_wrapper_provenance(saved, source, base, candidate)
            after_csr = csr_fingerprint(candidate.network)
            protocol.exact(before_csr, after_csr,
                           "E150 actual source CSR contents unchanged")
            self.source_checks.append({"before_csr": before_csr,
                "after_csr": after_csr,
                "source_network_identity": source["network"]["model_hash"],
                "whole_unwrapped_snapshot_exact": True})
            return candidate
        except BaseException as error:
            owner = candidate if candidate is not None else base
            try:
                if not getattr(owner, "closed", False):
                    owner.close()
            except BaseException as secondary:
                error.add_note("Secondary E150 rejected wrapper close: " +
                               repr(secondary))
            raise

    def metadata(self):
        return {"source_restores": self.source_restores,
                "wrapper_restores": self.wrapper_restores,
                "source_checks": copy.deepcopy(self.source_checks)}


def csr_fingerprint(network):
    matrix = protocol.source_matrix(network)
    return {"shape": list(matrix.shape), **{key: protocol._hash_array(
        getattr(matrix, key)) for key in ("indptr", "indices", "data")}}


def verify_wrapper_provenance(original, source, base, candidate):
    if candidate.network is not base.network or candidate.body is not base.body:
        raise ValueError("E150 wrapper changed source network/body ownership")
    wrapped = candidate.snapshot()
    unchanged = ("graph_hash", "parameters", "seed", "world", "body", "network",
        "cuts", "control_tick", "initial_position", "signed_forward_mm",
        "cpg_hash", "wall_s", "navigation")
    for key in unchanged:
        protocol.exact(original[key], source[key], "E150 restored source " + key)
        protocol.exact(source[key], wrapped[key], "E150 wrapper source " + key)
    if (candidate.network.identity != base.network.identity or
            candidate.body.model_hash != base.body.model_hash):
        raise ValueError("E150 wrapper changed native model identity")
    patch = wrapped.get("monotone_recruitment_patch")
    if not isinstance(patch, dict):
        raise TypeError("E150 monotone patch provenance missing")
    protocol.exact(getattr(base, "motor_timescale_patch", None),
                   patch.get("source_motor_timescale_patch"),
                   "E150 source motor-timescale patch")
    unwrapped = copy.deepcopy(wrapped)
    unwrapped.pop("monotone_recruitment_patch")
    unwrapped["schema"] = source["schema"]
    unwrapped["adapter"] = copy.deepcopy(wrapped["adapter"]["original"])
    protocol.exact(source, unwrapped, "E150 complete unwrapped source snapshot")
    return wrapped


def _copy_array(value, shape):
    array = np.asarray(value)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("E150 native recruitment evidence shape/finite mismatch")
    return np.array(array, dtype=np.float64, copy=True)


def make_observer(records, original=protocol._observe):
    """Publish adapter state progressively, then retain the original readout."""
    def observe(session, detailed, available=None):
        if available is None:
            available = {}
        record = {"control_tick": int(session.control_tick),
            "network_tick": int(session.network.tick), "complete": False}
        records.append(record)
        available["e150_native"] = record
        try:
            record["time_s"] = float(session.body.physics_time())
            record["bank"] = _copy_array(session.adapter.bank, (391, 3))
            record["activation"] = _copy_array(
                session.adapter.activation, (96,))
            record["original_activation"] = _copy_array(
                session.adapter.original.activation, (96,))
            motor_ids = np.asarray(session.adapter.calibration["motor_ids"])
            if motor_ids.shape != (391,) or motor_ids.dtype.kind not in "iu":
                raise ValueError("E150 original 391-MN identity required")
            record["motor_rate"] = _copy_array(
                session.network.readout(motor_ids), (391,))
            fields = original(session, detailed, available)
            fields.update(monotone_bank=record["bank"].copy(),
                monotone_activation=record["activation"].copy(),
                original_activation=record["original_activation"].copy(),
                motor_rate_391=record["motor_rate"].copy(),
                control_tick=np.array(record["control_tick"], np.int64),
                network_tick=np.array(record["network_tick"], np.int64),
                physics_time_s=np.array(record["time_s"], np.float64))
            record["complete"] = True
            return fields
        except BaseException as error:
            record["error"] = repr(error)
            raise
    return observe


def _optional_stack(records, key, shape):
    result = np.full((len(records), *shape), np.nan, np.float64)
    available = np.zeros(len(records), bool)
    for index, record in enumerate(records):
        if key in record:
            result[index] = np.asarray(record[key], np.float64)
            available[index] = True
    return result, available


def save_native_state(path, records):
    bank, bank_ok = _optional_stack(records, "bank", (391, 3))
    active, active_ok = _optional_stack(records, "activation", (96,))
    original, original_ok = _optional_stack(records, "original_activation", (96,))
    rates, rate_ok = _optional_stack(records, "motor_rate", (391,))
    np.savez_compressed(path,
        schema=np.array("flylab.m3.e150.native-monotone-state.v1"),
        control_tick=np.asarray([row.get("control_tick", -1) for row in records],
                                np.int64),
        network_tick=np.asarray([row.get("network_tick", -1) for row in records],
                                np.int64),
        physics_time_s=np.asarray([row.get("time_s", np.nan) for row in records],
                                  np.float64),
        complete=np.asarray([row.get("complete", False) for row in records], bool),
        bank=bank, bank_available=bank_ok, activation=active,
        activation_available=active_ok, original_activation=original,
        original_activation_available=original_ok, motor_rate=rates,
        motor_rate_available=rate_ok)


def relabel_trace(path):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    original = {"schema": value.get("schema"),
                "controller": value.get("controller")}
    samples = copy.deepcopy(value.get("samples"))
    value.update(schema="flylab.m3.e150.native-monotone-body-trace.v1",
        controller="frozen E149 monotone recruitment; full native feedback",
        E150_provenance={
            "adapter": "flylab.m3-monotone-recruitment-adapter.v1",
            "candidate": "frozen E148 parameter vector",
            "numeric_protocol": "unchanged E141 native body protocol",
            "original_protocol_metadata": original})
    protocol.write_json(path, value)
    reread = json.loads(path.read_text(encoding="utf-8"))
    if reread.get("samples") != samples:
        raise ValueError("E150 trace relabel changed numeric samples")


def make_raw_saver(records, original=protocol._save_body_raw,
                   save_state=save_native_state, relabel=relabel_trace):
    def save(out, *args, **kwargs):
        primary = None
        try:
            errors = list(original(out, *args, **kwargs))
        except BaseException as error:
            primary = error
            errors = []
        for label, callback in (
            ("trace.json E150 provenance", lambda: relabel(Path(out) / "trace.json")),
            ("native-recruitment-state.npz", lambda: save_state(
                Path(out) / "native-recruitment-state.npz", records))):
            try:
                callback()
            except BaseException as error:
                errors.append({"artifact": label,
                    "error": type(error).__name__, "message": str(error),
                    "traceback": traceback.format_exc()})
                if primary is not None:
                    primary.add_note("Secondary E150 " + label + ": " + repr(error))
        if primary is not None:
            raise primary
        return errors
    return save


def isolated_native_body(factory, observer, raw_saver):
    """Copy E141 code/defaults/closure with exactly four frozen overrides."""
    source = protocol._native_body
    namespace = source.__globals__.copy()
    values = {"MotorTimescaleTrial": factory, "_observe": observer,
              "_save_body_raw": raw_saver, "INTERNAL_LIMIT_S": 250.0}
    if tuple(values) != OVERRIDES:
        raise RuntimeError("E150 override declaration changed")
    namespace.update(values)
    function = types.FunctionType(source.__code__, namespace, source.__name__,
                                  source.__defaults__, source.__closure__)
    function.__kwdefaults__ = copy.copy(source.__kwdefaults__)
    function.e150_overrides = tuple(values)
    return function


def close_preserving(primary, label, operation):
    try:
        operation()
    except BaseException as error:
        if primary is None:
            return error
        primary.add_note("Secondary " + label + ": " + repr(error))
    return primary


def prepare_neutral(graph, original, parameters, out, ledger, persist):
    factory = TrialFactory(parameters)
    owner = None
    primary = None
    snapshot = None
    try:
        owner = protocol.profiled_construction("prepare_monotone_neutral",
            lambda: factory.from_checkpoint(graph, original),
            ledger["construction_attempts"], persist)
        snapshot = owner.snapshot()
        for key in ("graph_hash", "parameters", "seed", "world", "body", "network",
                    "cuts", "control_tick", "initial_position",
                    "signed_forward_mm", "cpg_hash", "wall_s", "navigation"):
            protocol.exact(original[key], snapshot[key],
                           "E150 prepared wrapper source " + key)
        if (owner.network.device != "cuda" or owner.control_tick != 0 or
                owner.network.tick != 0 or abs(owner.body.physics_time()) > 1e-12 or
                not np.array_equal(owner.adapter.bank, np.zeros((391, 3))) or
                not np.array_equal(owner.adapter.activation, np.zeros(96)) or
                not np.array_equal(owner.adapter.original.activation, np.zeros(96)) or
                not np.array_equal(owner.adapter.calibration["motor_ids"],
                                   parameters["motor_ids"])):
            raise ValueError("E150 wrapper-neutral invariant changed")
        actual_csr = protocol.accounted_operation("load_fixed_csr_for_wrapper",
            lambda: protocol.load_npz(out / "actual-global-csr.npz"),
            ledger["storage_operations"], persist)
        expected_csr = {"shape": [158706, 158706], **{
            key: protocol._hash_array(actual_csr[key])
            for key in ("indptr", "indices", "data")}}
        protocol.exact(expected_csr, factory.source_checks[-1]["before_csr"],
                       "E150 live full CSR equals frozen original CSR")
        protocol.accounted_operation("save_wrapper_provenance",
            lambda: protocol.write_json(out / "wrapper-provenance.json", {
                "schema": "flylab.m3.e150.wrapper-provenance.v1",
                "expected_frozen_csr": expected_csr,
                "source_checks": copy.deepcopy(factory.source_checks),
                "same_network_and_body_objects": True,
                "whole_unwrapped_snapshot_exact": True}),
            ledger["storage_operations"], persist)
        protocol.accounted_operation("save_monotone_neutral",
            lambda: protocol.StateStore.save(out / "candidate-neutral", snapshot),
            ledger["storage_operations"], persist)
        loaded = protocol.accounted_operation("load_monotone_neutral",
            lambda: protocol.StateStore.load(out / "candidate-neutral"),
            ledger["storage_operations"], persist)
        protocol.exact(snapshot, loaded, "E150 full wrapper-neutral disk roundtrip")
    except BaseException as error:
        primary = error
    finally:
        if owner is not None:
            primary = close_preserving(primary, "prepared owner close", owner.close)
    if primary is not None:
        raise primary
    return snapshot, factory


def valid_cut_control(row, record):
    try:
        optional_state = [row[key] for key in ("qpos", "qvel", "ctrl", "motor_rates")
                          if key in row]
        return bool(not bool(row["fault"]) and
            np.asarray(row["joint_targets_rad"]).shape == (42,) and
            np.isfinite(row["joint_targets_rad"]).all() and
            set(row["tendon_inputs"]) == set(TENDON_KEYS).union(NON_MOTOR_TENDON_HOLDS) and
            np.isfinite(list(row["tendon_inputs"].values())).all() and
            all(row["tendon_inputs"][key] == value
                for key, value in NON_MOTOR_TENDON_HOLDS.items()) and
            all(np.isfinite(value).all() for value in optional_state) and
            record.get("complete") is True and
            record["bank"].shape == (391, 3) and
            record["activation"].shape == (96,) and
            record["original_activation"].shape == (96,) and
            record["motor_rate"].shape == (391,) and
            all(np.isfinite(record[key]).all() for key in
                ("bank", "activation", "original_activation", "motor_rate")))
    except (KeyError, TypeError, ValueError):
        return False


def full_main_complete(body):
    return bool(body.get("validity") ==
        "valid_full_BANC_native_transfer_diagnostic" and
        body.get("technical_failure") is False and
        body.get("stop_reason") == "duration" and
        body.get("measured_health_stop") is False and
        body.get("clock_valid") is True and
        body.get("main_completed_controls") == 2000 and
        body.get("main_native_samples") == 100000 and
        body.get("clone_native_samples") == 1000 and
        np.isclose(body.get("actual_neural_s", np.nan), 10.1, rtol=0, atol=1e-12) and
        np.isclose(body.get("actual_physical_s", np.nan), 10.1, rtol=0, atol=1e-8) and
        not body.get("raw_persistence_errors") and
        not body.get("analysis_errors"))


def validate_cut_restart(rows, records, neutral_target, clock, native_count):
    complete = bool(len(rows) == len(records) == 20 and native_count == 1000 and
        clock.clock_valid and np.isclose(clock.neural_s, .1, rtol=0, atol=1e-12) and
        np.isclose(clock.physical_s, .1, rtol=0, atol=1e-8) and
        all(valid_cut_control(row, record)
            for row, record in zip(rows, records, strict=True)))
    cut_rows, restart_rows = rows[:4], rows[4:]
    cut_records, restart_records = records[:4], records[4:]
    neutral_target = np.asarray(neutral_target, np.float64)
    if neutral_target.shape != (42,) or not np.isfinite(neutral_target).all():
        raise ValueError("E150 finite42 native neutral target required")
    cut_zero = bool(complete and all(
        np.array_equal(record["bank"], np.zeros((391, 3))) and
        np.array_equal(record["activation"], np.zeros(96)) and
        np.array_equal(record["original_activation"], np.zeros(96)) and
        np.array_equal(np.asarray(row["joint_targets_rad"]), neutral_target) and
        all(float(row["tendon_inputs"][key]) == 0 for key in TENDON_KEYS)
        for row, record in zip(cut_rows, cut_records, strict=True)))
    restarted = bool(complete and any(np.any(record["motor_rate"] > 0)
        for record in restart_records) and any(np.any(record["activation"] > 0)
        for record in restart_records) and any(not np.array_equal(
            np.asarray(row["joint_targets_rad"]), neutral_target)
            for row in restart_rows))
    return {"schema": "flylab.m3.e150.cut-restart-result.v1",
        "status": "completed" if complete else "incomplete",
        "accepted": bool(complete and cut_zero and restarted),
        "complete_clock_and_samples": complete, "cut_exact_zero": cut_zero,
        "restart_positive_non_neutral": restarted,
        "controls": len(rows), "native_samples": native_count,
        "actual_neural_s": clock.neural_s,
        "actual_physical_s": clock.physical_s,
        "clock_valid": clock.clock_valid,
        "F2_motor_transfer_evidence": bool(complete and cut_zero and restarted),
        "modeled_motor_tendon_keys": list(TENDON_KEYS),
        "preserved_non_motor_tendon_holds": dict(NON_MOTOR_TENDON_HOLDS),
        "F7_complete": False}


def run_cut_restart(out, graph, factory, ledger, persist, process_began,
                    wall_clock=None):
    if wall_clock is None:
        wall_clock = time.perf_counter
    cut_ledger = {"schema": "flylab.m3.e150.cut-restart-ledger.v1",
        "status": "created", "set_cuts": [], "rows": 0,
        "F7_complete": False}

    def cut_persist():
        protocol.write_json(out / "cut-restart-ledger.json", cut_ledger)
        persist()

    cut_persist()
    saved = protocol.accounted_operation("load_cut_restart_checkpoint",
        lambda: protocol.StateStore.load(out / "checkpoint-1s"),
        ledger["storage_operations"], cut_persist)
    session = restore = journal = None
    rows = []
    records = []
    native = []
    clock = protocol.AdvanceLedger()
    neutral_target = None
    runtime_error = None
    raw_errors = []
    try:
        session = protocol.profiled_construction("restore_cut_restart",
            lambda: factory.from_checkpoint(graph, saved),
            ledger["construction_attempts"], cut_persist)
        neutral_target = np.array(session.body.neutral, np.float64, copy=True)
        restore = protocol.install_native_observer(session.body, native)
        journal = protocol.ControlJournal(out / "cut-restart-journal.jsonl")
        observe = make_observer(records)
        detailed = np.asarray(session.adapter.calibration["motor_ids"][:1], np.int64)
        for value, count, stage in ((True, 4, "cut"), (False, 16, "restart")):
            transition = {"stage": stage, "requested": True, "returned": False,
                          "motor": value}
            cut_ledger["set_cuts"].append(transition)
            cut_persist()
            session.set_cuts({"motor": value})
            transition["returned"] = True
            cut_persist()
            for _ in range(count):
                if wall_clock() - process_began >= 250 - protocol.RESERVE_S:
                    raise TimeoutError("E150 reserve exhausted during cut/restart")
                index = len(rows) + 1
                before_native = len(native)
                before_tick = int(session.network.tick)
                before_control = int(session.control_tick)
                before_time = float(session.body.physics_time())
                event = protocol.advance_control(session, "cut_restart", index,
                                                 clock, journal)
                available = {}
                try:
                    row = protocol.capture_control(session)
                    observe(session, detailed, available)
                    rows.append(row)
                    if (len(native) - before_native != 50 or
                            session.network.tick != before_tick + 20 or
                            session.control_tick != before_control + 1 or
                            abs(session.body.physics_time() -
                                (before_time + .005)) > 1e-8 or
                            session.fault or session.body.fault or
                            not valid_cut_control(row, records[-1])):
                        raise ValueError("E150 complete healthy finite cut control required")
                    protocol.record_readout(journal, "cut_restart", index,
                                            event, complete=True)
                except BaseException as error:
                    try:
                        protocol.record_readout(journal, "cut_restart", index,
                                                event, complete=False, error=error)
                    except BaseException as secondary:
                        error.add_note("Secondary E150 cut journal: " + repr(secondary))
                    raise
        cut_ledger["status"] = "raw_pending"
    except BaseException as error:
        runtime_error = error
        cut_ledger.update(status="failed", primary_error=repr(error))
    finally:
        if journal is not None:
            cut_ledger["journal"] = journal.summary()
            runtime_error = close_preserving(runtime_error, "cut journal close",
                                             journal.close)
        cut_ledger.update(rows=len(rows), native_samples=len(native),
            clock={"neural_s": clock.neural_s, "physical_s": clock.physical_s,
                   "valid": clock.clock_valid, "events": clock.events})
        for label, callback in (
            ("cut-restart-control.json", lambda: protocol.write_json(
                out / "cut-restart-control.json", rows)),
            ("cut-restart-native-contact.npz", lambda: np.savez_compressed(
                out / "cut-restart-native-contact.npz",
                **protocol.native_arrays(native))),
            ("cut-restart-native-state.npz", lambda: save_native_state(
                out / "cut-restart-native-state.npz", records))):
            try:
                callback()
            except BaseException as error:
                raw_errors.append({"artifact": label, "error": repr(error)})
                if runtime_error is None:
                    runtime_error = error
                else:
                    runtime_error.add_note("Secondary " + label + ": " + repr(error))
        runtime_error = close_preserving(runtime_error, "cut observer close",
                                         restore or (lambda: None))
        if session is not None:
            runtime_error = close_preserving(runtime_error, "cut owner close",
                                             session.close)
        cut_ledger["raw_errors"] = raw_errors
        try:
            cut_persist()
        except BaseException as error:
            if runtime_error is None:
                runtime_error = error
            else:
                runtime_error.add_note("Secondary cut ledger: " + repr(error))
    if runtime_error is not None:
        raise runtime_error
    result = validate_cut_restart(rows, records, neutral_target, clock, len(native))
    cut_ledger.update(status=result["status"], result=result)
    protocol.write_json(out / "cut-restart-result.json", result)
    cut_persist()
    return result


def required_artifacts(out):
    names = list(REQUIRED_OUTPUTS)
    cut_result = Path(out) / "cut-restart-result.json"
    if cut_result.exists() and json.loads(cut_result.read_text(
            encoding="utf-8")).get("status") == "completed":
        names.extend(("cut-restart-journal.jsonl", "cut-restart-control.json",
            "cut-restart-native-contact.npz", "cut-restart-native-state.npz"))
    missing = [name for name in names if not (Path(out) / name).exists()]
    if missing:
        raise ValueError("E150 required artifacts missing: " + ",".join(missing))
    return names


def resource_checks(*, prepare_wall, body_wall, process_wall, raw_bytes,
                    free_disk_before, clock_valid):
    failures = []
    for failed, label in (
        (free_disk_before < 12 << 30, "minimum_free_disk_bytes"),
        (prepare_wall > 60, "prepare_wall_limit_s"),
        (body_wall > 180, "body_wall_limit_s"),
        (process_wall > 250, "internal_wall_limit_s"),
        (raw_bytes > 4 << 30, "maximum_raw_bytes"),
        (not clock_valid, "clock_valid"),
        (2229 + process_wall > 2514, "prospective_cumulative_limit_s")):
        if failed:
            failures.append(label)
    return failures


def persist_manifest_measure(out, began, *, clock=None,
                             persist_manifest=None, measure_raw=None):
    if clock is None:
        clock = time.perf_counter
    if persist_manifest is None:
        persist_manifest = protocol._persist_manifest
    if measure_raw is None:
        measure_raw = protocol._raw_bytes
    manifest = persist_manifest(out)
    elapsed = clock() - began
    raw_bytes = measure_raw(out)
    return manifest, elapsed, raw_bytes


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("Preserve single E150 output: " + str(out))
    out.mkdir(parents=True, exist_ok=False)
    ledger = {"schema": "flylab.m3.e150.native-monotone-ledger.v1",
        "status": "created", "construction_attempts": [],
        "storage_operations": [], "secondary_errors": [], "clock": {},
        "conditional_forward_calls": 0, "fit_calls": 0, "LP_calls": 0,
        "original_processes": 1, "candidate_adopted": False,
        "F6_exposed": False, "active_UI_changed": False,
        "whole_goal_complete": False}
    result = {"schema": "flylab.m3.e150.native-monotone-result.v1",
        "experiment": "M3-E150-native-monotone-body", "validity": "incomplete",
        "hypothesis_verdict": "invalid_not_tested", "body_started": False,
        "candidate_adopted": False, "F6_exposed": False,
        "active_UI_changed": False, "whole_goal_complete": False,
        "M3_complete": False, "fit_calls": 0, "LP_calls": 0,
        "conditional_forward_calls": 0}
    limits = None
    entered = False
    primary = None
    resource_failures = []

    def persist():
        ledger["construction_totals"] = protocol._construction_totals(
            ledger["construction_attempts"])
        ledger["process_wall_s"] = time.perf_counter() - began
        protocol.write_json(out / "advance-ledger.json", ledger)

    persist()
    try:
        freeze, review = verified_inputs(freeze_path, freeze_sha,
                                         review_path, review_sha)
        protocol.write_json(out / "source-review-snapshot.json", review)
        free_disk = shutil.disk_usage(out).free
        ledger["free_disk_before_original_bytes"] = free_disk
        if free_disk < freeze["minimum_free_disk_bytes"]:
            resource_failures.append("minimum_free_disk_bytes")
            raise OSError("E150 requires at least12 GiB free")
        limits, threads = protocol.joint_fit.one_cpu_thread()
        ledger["cpu_threads"] = threads
        cached = protocol.accounted_operation("load_cached_candidate_bundle",
            lambda: e142.load_cached_candidate(freeze, out),
            ledger["storage_operations"], persist)
        source_zero = out / "source-neutral"
        protocol.accounted_operation("rename_copied_source_zero",
            lambda: (out / "candidate-neutral").rename(source_zero),
            ledger["storage_operations"], persist)
        copied_zero = protocol.accounted_operation("reload_copied_source_zero",
            lambda: protocol.StateStore.load(source_zero),
            ledger["storage_operations"], persist)
        protocol.exact(cached["neutral"], copied_zero,
                       "E150 copied unchanged source zero")
        parameters = protocol.accounted_operation("load_monotone_parameters",
            lambda: load_parameters(_safe_path(freeze["parameter_bundle"])),
            ledger["storage_operations"], persist)
        ledger.update(cuda_runtime_entered=False,
                      status="wrapper_zero_construction_requested")
        persist()
        entered = True
        ledger.update(cuda_runtime_entered=True,
                      status="wrapper_zero_construction_entered")
        persist()
        neutral, factory = prepare_neutral(cached["graph"], cached["neutral"],
            parameters, out, ledger, persist)
        prepare_wall = time.perf_counter() - began
        ledger.update(prepare_wall_s=prepare_wall,
            factory_after_prepare=factory.metadata(), status="wrapper_zero_verified")
        if prepare_wall > freeze["prepare_wall_limit_s"]:
            resource_failures.append("prepare_wall_limit_s")
            raise TimeoutError("E150 preparation exceeded60 seconds")
        spec = {"experiment": "E150", "controller": "frozen E149 monotone recruitment",
            "body_protocol": "E141-code-with-isolated-factory",
            "continuation_of": "E148-fixed-vector-and-E149-adapter",
            "fixed_candidate": True, "new_parameter_selection": False,
            "main_controls": 2000, "clone_controls": 20,
            "cut_controls": 4, "restart_controls": 16,
            "freeze_sha256": FREEZE_SHA, "source_review_sha256": review_sha,
            "conditional_forward_calls": 0, "candidate_adopted": False,
            "F6_exposed": False, "active_UI_changed": False,
            "whole_goal_complete": False}
        protocol.write_json(out / "spec.json", spec)
        records = []
        observer = make_observer(records)
        raw_saver = make_raw_saver(records)
        native_body = isolated_native_body(factory, observer, raw_saver)
        ledger.update(status="native_body_requested",
            isolated_global_overrides=list(native_body.e150_overrides))
        persist()
        ledger.update(status="native_body_entered")
        result["body_started"] = True
        body = native_body(freeze, cached["graph"], cached["cal"],
            cached["observation"], cached["startup"], neutral,
            cached["parameters"], cached["anatomy"], out, spec, ledger,
            persist, began)
        if ((out / "checkpoint-1s").exists() and full_main_complete(body)):
            cut = run_cut_restart(out, cached["graph"], factory, ledger, persist,
                                  began)
        else:
            cut = {"status": "not_run_main_incomplete", "accepted": False,
                   "actual_neural_s": 0., "actual_physical_s": 0.,
                   "F7_complete": False}
            protocol.write_json(out / "cut-restart-ledger.json", {
                "schema": "flylab.m3.e150.cut-restart-ledger.v1",
                "status": cut["status"], "F7_complete": False})
            protocol.write_json(out / "cut-restart-result.json", cut)
        def load_baseline():
            return {key: json.loads((ROOT / freeze["baseline_directory"] / name)
                .read_text(encoding="utf-8")) for key, name in (
                    ("quality", "quality-summary.json"),
                    ("startup", "startup-diagnostic.json"),
                    ("result", "result.json"))}
        baseline = protocol.accounted_operation("load_E142_comparison",
            load_baseline, ledger["storage_operations"], persist)
        comparison = {"schema": "flylab.m3.e150.e142-comparison.v1",
            "window_s": [0.5, 10.0], "baseline_E142": baseline,
            "candidate_E150": {"quality": body.get("quality_summary"),
                "startup": body.get("strict_startup"), "cut_restart": cut},
            "same_original_protocol_numbers": True,
            "component_attribution_allowed": False}
        protocol.write_json(out / "E142-E150-same-window-comparison.json", comparison)
        actual_neural_s = body.get("actual_neural_s")
        actual_physical_s = body.get("actual_physical_s")
        if actual_neural_s is not None:
            actual_neural_s += cut["actual_neural_s"]
        if actual_physical_s is not None:
            actual_physical_s += cut["actual_physical_s"]
        verdict = body["hypothesis_verdict"]
        if cut["status"] == "completed" and not cut["accepted"]:
            verdict = "refuted_motor_cut_restart"
        result.update(body, cut_restart=cut, prepare_wall_s=prepare_wall,
            hypothesis_verdict=verdict, actual_scientific_neural_s=actual_neural_s,
            actual_scientific_physical_s=actual_physical_s,
            requested_scientific_neural_ticks=40800,
            requested_scientific_physics_steps=102000,
            actual_scientific_neural_ticks=(None if actual_neural_s is None else
                                            round(actual_neural_s / .00025)),
            actual_scientific_physics_steps=(None if actual_physical_s is None else
                                             round(actual_physical_s / .0001)),
            construction_accounting=protocol._construction_totals(
                ledger["construction_attempts"]), factory=factory.metadata(),
            candidate_adopted=False, F6_exposed=False,
            active_UI_changed=False, whole_goal_complete=False, M3_complete=False)
        if (body["body_analysis_wall_s"] > freeze["body_wall_limit_s"] or
                body["stop_reason"] == "resource_guard"):
            resource_failures.append("body_wall_limit_s")
            raise TimeoutError("E150 native body exceeded180 seconds")
        ledger["status"] = "result_ready"
    except BaseException as error:
        primary = error
        ledger["status"] = "technical_failed"
        ledger["primary_error"] = {"type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc(),
            "notes": list(getattr(error, "__notes__", [])),
            "response_progress": getattr(error, "response_progress", None)}
    finally:
        def absorb(error, label):
            nonlocal primary
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary " + label + ": " + repr(error))

        def update_final_result(elapsed):
            result.update(process_wall_s=elapsed,
                prior_conservative_actual_s=2228.8917264000047,
                prior_conservative_process_s=2229,
                conservative_cumulative_s=2229 + elapsed,
                resource_limit_met=not resource_failures,
                resource_failures=list(dict.fromkeys(resource_failures)),
                candidate_adopted=False, F6_exposed=False,
                active_UI_changed=False, whole_goal_complete=False,
                M3_complete=False)
            if primary is not None:
                result.update(validity=("invalid_resource" if resource_failures
                    else "incomplete"), hypothesis_verdict="invalid_not_tested",
                    technical_failure=True, primary_error={
                        "type": type(primary).__name__, "message": str(primary),
                        "notes": list(getattr(primary, "__notes__", []))})

        for label, error in protocol.cleanup_cuda_runtime(entered, ledger):
            ledger["secondary_errors"].append({"stage": label, "error": repr(error)})
            absorb(error, label)
        if limits is not None:
            primary = close_preserving(primary, "thread restore",
                                       limits.restore_original_limits)
        # Persist a complete ledger and a preliminary result before the manifest.
        # The manifest excludes result.json, so the final fail-closed result can
        # be rewritten without invalidating its hashes.
        try:
            persist()
        except BaseException as error:
            absorb(error, "E150 final ledger persistence")
        update_final_result(time.perf_counter() - began)
        preliminary = copy.deepcopy(result)
        preliminary.update(validity="incomplete",
            hypothesis_verdict="invalid_not_tested", technical_failure=True,
            finalization_pending=True)
        try:
            protocol.write_json(out / "result.json", preliminary)
        except BaseException as error:
            absorb(error, "E150 preliminary result persistence")
        raw_bytes = None
        try:
            result["output_manifest"], elapsed, raw_bytes = (
                persist_manifest_measure(out, began))
        except BaseException as error:
            absorb(error, "E150 manifest persistence/measurement")
            elapsed = time.perf_counter() - began
        try:
            if raw_bytes is None:
                raw_bytes = protocol._raw_bytes(out)
            if primary is None:
                result["required_artifacts"] = required_artifacts(out)
                if not (out / "output-manifest.json").exists():
                    raise ValueError("E150 output manifest missing")
                checks = resource_checks(
                    prepare_wall=ledger["prepare_wall_s"],
                    body_wall=result["body_analysis_wall_s"],
                    process_wall=elapsed, raw_bytes=raw_bytes,
                    free_disk_before=ledger["free_disk_before_original_bytes"],
                    clock_valid=bool(result["clock_valid"] and (
                        result["cut_restart"]["status"] != "completed" or
                        result["cut_restart"]["clock_valid"])))
                resource_failures.extend(checks)
                if checks:
                    raise RuntimeError("E150 resource checks failed: " +
                                       ",".join(checks))
            elif elapsed > 250:
                resource_failures.append("internal_wall_limit_s")
            if raw_bytes > 4 << 30:
                resource_failures.append("maximum_raw_bytes")
                raise RuntimeError("E150 final raw exceeded4 GiB")
        except BaseException as error:
            absorb(error, "E150 final resource checks")
        update_final_result(time.perf_counter() - began)
        result["finalization_pending"] = False
        publication_failed = False
        try:
            protocol.write_json(out / "result.json", result)
        except BaseException as error:
            absorb(error, "E150 final result persistence")
            publication_failed = True
        # A slow or failed final result write must revoke a prior success.
        late_changed = publication_failed
        try:
            if (protocol._raw_bytes(out) > 4 << 30 and
                    "maximum_raw_bytes" not in resource_failures):
                resource_failures.append("maximum_raw_bytes")
                absorb(RuntimeError("E150 final result exceeded4 GiB"),
                       "E150 late raw guard")
                late_changed = True
        except BaseException as error:
            absorb(error, "E150 late raw measurement")
            late_changed = True
        elapsed = time.perf_counter() - began
        if elapsed > 250 and "internal_wall_limit_s" not in resource_failures:
            resource_failures.append("internal_wall_limit_s")
            absorb(TimeoutError("E150 final artifacts exceeded250 seconds"),
                   "E150 late resource guard")
            late_changed = True
        if late_changed:
            update_final_result(elapsed)
            try:
                protocol.write_json(out / "result.json", result)
            except BaseException as error:
                absorb(error, "E150 fail-closed result persistence")
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "freeze-sha256", "source-review",
                 "source-review-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.freeze, args.freeze_sha256, args.source_review,
                         args.source_review_sha256, args.output)))


if __name__ == "__main__":
    main()
