"""E305 body-free SSP first held-drive control, with separate RK4 comparison."""
from __future__ import annotations

# Scientific exceptions remain primary through cleanup and publication.
# ruff: noqa: BLE001
import argparse
import copy
import gc
import importlib.metadata
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import probe_c_banc_m3_recovery_first_fault as prior
from tools import verify_c_banc_m3_recovery_resource_body as e301

OWNED = ("tools/verify_c_banc_m3_ssp_first_control.py", "tests/test_c_banc_ssp_first_control.py")
PRODUCERS = ("tools/banc_m3_ssp_recovery_synapses.py", "tests/test_c_banc_ssp_recovery_synapses.py")
SOURCE_FILES = tuple(dict.fromkeys((*OWNED, *PRODUCERS, *prior.OWNED, *e301.SOURCE_FILES,
                                 "flylab/c/storage.py", "tools/verify_c_banc_m3_native_timescale_transfer.py")))
INPUTS = prior.INPUTS
FREEZE_SCHEMA = "flylab.m3.e305.ssp-first-control.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e305.ssp-first-control.review.v1"
RESULT_SCHEMA = "flylab.m3.e305.ssp-first-control.result.v1"
QUALIFICATION_SCHEMA = "flylab.m3.e303.ssp-tiny-qualification.v1"
QUALIFICATION_CHECKS = {key: True for key in ("cpu_pass", "cuda_pass", "independent_oracle_pass", "disk_future_exact")}
OUTPUT_PATH = "verification/banc-natural-gait-m3-20260919/e305-ssp-first-control"
BASE_FREEZE_PATH = "verification/banc-natural-gait-m3-20260919/e302-recovery-first-fault-freeze.json"
DESIGN = "docs/BANC_M3_E305_SSP_FIRST_CONTROL_DESIGN_20260928.md"
DESIGN_SHA = "f6033ad0f5d074589fe623349d78df792d159beb0ed449448972be026bba17ba"
RECEIPTS = ("e302_freeze", "e302_review", "e302_result", "e302_ledger", "e302_source_manifest", "e302_output_manifest", "e302_root_audit",
            "e303_source_review", "e303_tiny_qualification")
LIMITS = {"supervisor_wall_s": 60, "internal_wall_s": 55, "science_wall_s": 45, "host_peak_bytes": 4 << 30,
          "additional_gpu_peak_bytes": 2 << 30, "output_bytes": 128 << 20, "minimum_free_disk_bytes": 4 << 30,
          "cpu_threads": 1, "models": 2, "scientific_ticks_per_arm": 20, "scientific_ticks_total": 40,
          "original_attempts": 1, "retries": 0, "body_constructors": 0, "physics_steps": 0, "solver_calls": 0, "fit_calls": 0}
FLAGS = {key: False for key in ("candidate_adopted", "F1_F8_complete", "F2_complete", "F6_exposed", "biological_claim", "behavior_criteria_evaluated")}
SOURCE_IDENTITY, INFERENCE = prior.SOURCE_IDENTITY, prior.INFERENCE


def pin_bytes(groups, root=ROOT):
    pins = {}
    for group in groups:
        for name, digest in group.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E305 pin")
            if not isinstance(digest, str) or len(digest) != 64 or prior.sha256(prior.safe(name, root)) != digest:
                raise ValueError("Frozen E305 byte differs: " + name)
            pins[name] = digest
    return pins


def receipt(spec, pins=None, root=ROOT):
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise ValueError("Exact independent receipt spec required")
    path = prior.safe(spec["path"], root)
    if prior.sha256(path) != spec["sha256"] or pins is not None and pins.get(spec["path"]) != spec["sha256"]:
        raise ValueError("Independent receipt SHA differs")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_e302(result, ledger, audit):
    if (result.get("schema") != prior.RESULT_SCHEMA or result.get("diagnostic_id") != "E302" or result.get("run_valid") is not True
            or result.get("first_fault_observed") is not True or result.get("all_valid_factors_unity_before_first_fault") is not True
            or audit.get("schema") != "flylab.m3.e302.root-original-raw-audit.v1" or audit.get("pass") is not True
            or audit.get("diagnostic_run_valid") is not True or audit.get("prior_valid_factor_stages") != 39
            or audit.get("all_prior_factors_one") is not True):
        raise ValueError("Audited E302 first internal RK4 fault required")
    for item in (result, audit):
        if any(item.get(key) is not False for key in ("candidate_adopted", "F1_F8_complete", "F6_exposed")):
            raise ValueError("E302 diagnostic must remain unpromoted")
    fault = result["first_fault"]
    if (fault.get("attempted_tick") != 10 or fault.get("completed_tick") != 9 or fault.get("stage_index") != 3
            or fault.get("actual_flags") != 4 or fault.get("negative_source_ids") != [96198] or fault.get("valid") is not False):
        raise ValueError("Exact tick10/stage3/source96198/flag4 boundary required")
    prior.same(fault, audit["first_fault"], "E302 independently audited fault")
    prior.same(result["cost"], ledger, "E302 result/ledger cost")
    science = ledger["science"]
    if (science.get("completed_ticks") != 9 or science.get("partial_ticks") != 1 or science.get("advance_requested") != 10
            or science.get("advance_completed") != 9 or science.get("advance_failed") != 1 or ledger.get("body_constructors") != 0):
        raise ValueError("E302 nine completed plus one partial cost required")
    rows = result["rows"]
    if len(rows) != 40 or any(row.get("valid") is not True or row.get("factor_nonunity_count") != 0 for row in rows[:39]):
        raise ValueError("Original 39 valid unity stages required")
    prior.same(rows[-1], fault, "E302 last observed stage")


def validate_qualification(review, qualification, producer_sha, root=ROOT):
    if (review.get("approved") is not True or review.get("blocking_defects") != [] or review.get("source_sha256") != producer_sha
            or qualification.get("schema") != QUALIFICATION_SCHEMA or qualification.get("approved") is not True
            or qualification.get("blocking_defects") != [] or qualification.get("source_sha256") != producer_sha
            or qualification.get("checks") != QUALIFICATION_CHECKS or not qualification.get("evidence_sha256")):
        raise ValueError("Actual source-bound E303 CPU/CUDA/oracle/disk qualification required")
    pin_bytes([qualification["evidence_sha256"]], root)


def gate(freeze_path, freeze_sha, review_path, review_sha, output, *, root=ROOT, version_reader=None):
    if prior.sha256(freeze_path) != freeze_sha or prior.sha256(review_path) != review_sha:
        raise ValueError("E305 freeze/review SHA differs")
    freeze, review = (json.loads(Path(path).read_text(encoding="utf-8")) for path in (freeze_path, review_path))
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("diagnostic_id") != "E305" or freeze.get("owned_files") != list(OWNED)
            or set(freeze.get("inputs", {})) != set(INPUTS) or freeze.get("limits") != LIMITS or freeze.get("output") != OUTPUT_PATH
            or freeze.get("design") != DESIGN or freeze.get("design_sha256") != DESIGN_SHA
            or freeze.get("source_network_identity") != SOURCE_IDENTITY or freeze.get("drive_inference") != INFERENCE
            or freeze.get("candidate_adopted") is not False):
        raise ValueError("Exact prospective E305 contract required")
    pins = pin_bytes([freeze["source_sha256"], freeze["input_sha256"], freeze["preservation_sha256"], {DESIGN: DESIGN_SHA}], root)
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha or review.get("source_sha256") != {key: pins[key] for key in OWNED}
            or not set(SOURCE_FILES) <= set(freeze["source_sha256"])):
        raise ValueError("Matching current E305 source approval required")
    records = {name: receipt(freeze[name], pins, root) for name in RECEIPTS}
    old = records["e302_freeze"]
    if (freeze["e302_freeze"]["path"] != BASE_FREEZE_PATH or old.get("schema") != prior.FREEZE_SCHEMA or old.get("diagnostic_id") != "E302"
            or old.get("source_network_identity") != SOURCE_IDENTITY or old.get("inputs") != freeze["inputs"]
            or old.get("drive_inference") != INFERENCE or freeze.get("packages") != old.get("packages")):
        raise ValueError("Literal unchanged old E302 inputs and context required")
    old_review, source = records["e302_review"], records["e302_source_manifest"]
    if (old_review.get("schema") != prior.REVIEW_SCHEMA or old_review.get("approved") is not True or old_review.get("blocking_defects") != []
            or old_review.get("freeze_sha256") != freeze["e302_freeze"]["sha256"]
            or old_review.get("source_sha256") != {name: old["source_sha256"][name] for name in prior.OWNED}
            or source.get("diagnostic_id") != "E302" or source.get("owned_sha256") != old_review["source_sha256"]
            or source.get("source_sha256") != old["source_sha256"] or source.get("input_sha256") != old["input_sha256"]):
        raise ValueError("Actual E302 source/execution lineage required")
    for group in (old["source_sha256"], old["input_sha256"], old["preservation_sha256"], {old["design"]: old["design_sha256"]}):
        if any(pins.get(name) != digest for name, digest in group.items()):
            raise ValueError("Unchanged complete E302 input/source/preservation chain required")
    earlier = {name: receipt(old[name], pins, root) for name in prior.RECEIPTS}
    prior.validate_e301_evidence(earlier["e301_freeze"], earlier["e301_result"], earlier["e301_ledger"], earlier["e301_root_audit"],
                                 earlier["e301_output_manifest"], earlier["e301_source_manifest"])
    validate_e302(records["e302_result"], records["e302_ledger"], records["e302_root_audit"])
    parent = prior.safe(freeze["e302_output_manifest"]["path"], root).parent
    if (records["e302_output_manifest"].get("schema") != "flylab.m3.e302.outputs.v1"
            or records["e302_output_manifest"].get("result_published_separately") is not True):
        raise ValueError("Frozen E302 separate-result publication contract required")
    for alias, filename in (("e302_result", "result.json"), ("e302_ledger", "ledger.json"),
                            ("e302_source_manifest", "source-manifest.json")):
        if prior.safe(freeze[alias]["path"], root) != parent / filename:
            raise ValueError("Actual E302 producer output path required")
    for name, entry in records["e302_output_manifest"]["files"].items():
        path = prior.safe((parent.relative_to(root) / name).as_posix(), root)
        if (pins.get(path.relative_to(root).as_posix()) != entry["sha256"] or prior.sha256(path) != entry["sha256"]
                or path.stat().st_size != entry["bytes"]):
            raise ValueError("Actual E302 output raw/manifest bytes differ")
    validate_qualification(records["e303_source_review"], records["e303_tiny_qualification"], {name: pins[name] for name in PRODUCERS}, root)
    for name in freeze["inputs"].values():
        path = prior.safe(name, root)
        files = [file for file in path.rglob("*") if file.is_file()] if path.is_dir() else [path]
        if not files or any(file.relative_to(root).as_posix() not in freeze["input_sha256"] for file in files):
            raise ValueError("All nine inputs and complete checkpoints must be pinned")
    versions = version_reader or (lambda name: platform.python_version() if name == "python" else importlib.metadata.version(name))
    if any(versions(name) != version for name, version in freeze["packages"].items()):
        raise ValueError("Pinned actual package differs")
    if output.resolve() != prior.safe(OUTPUT_PATH, root) or output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh literal E305 output required")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E305 disk reserve unavailable")
    return freeze


def engine_api():
    from tools import banc_m3_ssp_recovery_synapses
    return banc_m3_ssp_recovery_synapses


def load_inputs(freeze, *, engine=None, old_loader=None, checkpoint_loader=None, root=ROOT):
    old = receipt(freeze["e302_freeze"], root=root)
    if freeze["e302_freeze"]["path"] != BASE_FREEZE_PATH or old.get("schema") != prior.FREEZE_SCHEMA or old.get("inputs") != freeze["inputs"]:
        raise ValueError("Literal old E302 validation context required")
    if old_loader is None:
        if checkpoint_loader is None:
            from flylab.c.storage import StateStore
            checkpoint_loader = StateStore.load
        references = {}
        def wrapped(path):
            state = checkpoint_loader(path)
            if path == prior.safe(old["inputs"]["neutral_final"], root):
                references["rk4_reference"] = state["network"]
            return state
        data = prior.load_inputs(old, checkpoint_loader=wrapped, root=root)
        data.update(references)
    else:
        data = old_loader(old)
    api = engine_api() if engine is None else engine
    selections = {}
    for arm in ("neutral", "candidate"):
        selection = copy.deepcopy(data["selection"])
        if arm == "neutral":
            selection["utilization"] = np.zeros_like(selection["utilization"])
        metadata = api.resource_metadata(SOURCE_IDENTITY, selection)
        api.checked_resource_metadata(metadata)
        identity = api.resource_identity(metadata)
        if (identity != freeze[arm + "_network_identity"] or identity in
                (SOURCE_IDENTITY, old["candidate_network_identity"], data["rk4_reference"]["model_hash"])):
            raise ValueError("Separate new SSP identity required: " + arm)
        selections[arm] = selection
    data.update(selections=selections, freeze=freeze)
    return data


def endpoint_factor(state, selection):
    rate = state["rate"][selection["source_ids"]].astype(np.float64)
    resource, u, tau = state["synaptic_resource"], selection["utilization"], selection["recovery_tau"]
    return np.minimum(1., resource * (1 + (tau * u) * rate)).astype(np.float32)


def differences(left, right):
    if left.dtype != right.dtype or left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Finite matching saved numeric definitions required")
    delta = left.astype(np.float64) - right.astype(np.float64)
    return {"RMS": float(np.sqrt(np.mean(delta ** 2))), "max_abs": float(np.max(np.abs(delta))),
            "left_negative_count": int(np.count_nonzero(left < 0)), "right_negative_count": int(np.count_nonzero(right < 0)),
            "byteexact": left.tobytes() == right.tobytes()}, delta


class NativeDependencies:
    def __init__(self, *, engine=None, profiler=None, old_loader=None):
        self._engine, self.profiler, self.old_loader = engine, profiler, old_loader
        self.resources = prior.NativeDependencies()
    @property
    def engine(self):
        return engine_api() if self._engine is None else self._engine
    def load(self, freeze):
        return load_inputs(freeze, engine=self.engine, old_loader=self.old_loader)
    def construct(self, arm, data, profiles, persist, on_return):
        profiler = self.profiler
        if profiler is None:
            from tools.verify_c_banc_m3_native_timescale_transfer import (
                profiled_construction,
            )
            profiler = profiled_construction
        net, selection, tau = data["network"], data["selections"][arm], data["configuration"]["membrane_tau"]
        core, motor, timescale = net["core_adaptation"], net["motor_adaptation"], net["motor_timescale"]
        def callback():
            return self.engine.SSPRecoveryResourceNetwork(data["weights"], tau, np.ones(len(tau), np.float32),
                np.full(len(tau), 7.5, np.float32), np.full(len(tau), 200., np.float32), adaptation_gain=8., adaptation_tau_s=.3,
                dt=.00025, device="cuda", cuda_implementation="packed", capture_steps=20, core_ids=core["ids"], core_gain=core["gain"],
                motor_ids=motor["ids"], motor_gains=motor["gains"], motor_log_tau=timescale["log_tau"],
                edge_ids=selection["edge_ids"], premotor_ids=selection["premotor_ids"], utilization=selection["utilization"], recovery_tau=selection["recovery_tau"])
        return profiler("e305_ssp_" + arm, callback, profiles, persist, on_return=on_return)
    def initial(self, network, data, arm):
        prior.same(network.selection, data["selections"][arm], "Protected SSP selection")
        if (network.source_timescale_identity != SOURCE_IDENTITY or network.identity != data["freeze"][arm + "_network_identity"]
                or network.device != "cuda" or network.cuda_implementation != "packed" or network.interval_steps != 20):
            raise ValueError("Actual SSP identity/implementation differs")
        for key in ("indptr", "indices", "data"):
            prior.same(getattr(network._source_weights, key), getattr(data["weights"], key), "Original CSR " + key)
        network.import_base_snapshot(copy.deepcopy(data["network"]))
        state = network.snapshot()
        for key in ("rate", "adaptation", "drive", "output_mask", "tick"):
            prior.same(state[key], data["network"][key], "Imported original initial " + key)
        prior.same(state["synaptic_resource"], np.ones(len(network.selection["source_ids"]), np.float64), "Initial resource one")
        for name, expected in (("tau", data["configuration"]["membrane_tau"]), ("_cell_gain", data["configuration"]["cell_gain"]),
                ("_cell_adaptation_tau", data["network"]["motor_timescale"]["tau32"]), ("_cell_inverse_tau", data["network"]["motor_timescale"]["inverse32"])):
            prior.same(prior.host_array(getattr(network, name)), expected, "Original native tensor " + name)
        packed, lookup = prior.host_array(network._columns).view(np.uint32), prior.host_array(network._weight_values)
        prior.same((packed & 0x3ffff).astype(data["weights"].indices.dtype), data["weights"].indices, "Original packed columns")
        prior.same(lookup[packed >> 18], data["weights"].data, "Original packed weights")
        return state
    def final(self, network, data, arm):
        state = network.snapshot()  # Public endpoint guard verifies sticky flags=0.
        n, selection = len(data["network"]["rate"]), data["selections"][arm]
        if state["tick"] != 20 or state["model_hash"] != data["freeze"][arm + "_network_identity"]:
            raise ValueError("One SSP20-tick endpoint with new identity required")
        for key in ("rate", "adaptation", "drive", "output_mask"):
            value = state[key]
            if value.dtype != np.float32 or value.shape != (n,) or not np.isfinite(value).all():
                raise ValueError("Finite native SSP endpoint required: " + key)
        if np.any(state["rate"] < 0) or np.any(state["adaptation"] < 0):
            raise ValueError("Nonnegative SSP rate/adaptation required")
        prior.same(state["drive"], data["drive"], "Exact held drive")
        prior.same(state["output_mask"], np.ones(n, np.float32), "Original mask one")
        resource = state["synaptic_resource"]
        if resource.dtype != np.float64 or resource.shape != selection["source_ids"].shape or not np.isfinite(resource).all() or np.any(resource < 0) or np.any(resource > 1):
            raise ValueError("SSP endpoint resource bounds required")
        prior.same(resource[selection["utilization"] == 0], np.ones(np.count_nonzero(selection["utilization"] == 0), np.float64), "u0 endpoint resource one")
        if self.engine.resource_identity(state["recurrent_resource"]) != state["model_hash"]:
            raise ValueError("Snapshot SSP metadata identity differs")
        g = endpoint_factor(state, selection)
        if not np.isfinite(g).all() or np.any(g < resource.astype(np.float32)) or np.any(g > 1):
            raise ValueError("Endpoint derived factor bounds required")
        return state, g
    def resource_check(self, output):
        return self.resources.resource_check(output)
    def release(self, _network):
        return None  # Release occurs by removing all owning refs before cleanup.
    def cleanup(self):
        return self.resources.cleanup()


class Writer(prior.Writer):
    def checkpoint(self, name, state):
        from flylab.c.storage import StateStore
        def estimate(value):
            if isinstance(value, np.ndarray):
                return value.nbytes + 256
            if isinstance(value, dict):
                return sum(estimate(item) + len(key) + 16 for key, item in value.items())
            if isinstance(value, (list, tuple)):
                return sum(estimate(item) + 4 for item in value)
            return len(repr(value)) + 16
        self._check(estimate(state) + (1 << 20))  # Keep terminal receipt reserve.
        return StateStore.save(self.output / name, state)


def execute(freeze, output, *, dependencies=None, writer=None, clock=time.perf_counter):
    deps = NativeDependencies() if dependencies is None else dependencies
    output.mkdir(exist_ok=False)
    writer = Writer(output) if writer is None else writer
    began, primary = clock(), None
    ledger = {"construction_profiles": [], "arms": {}, "resource_samples": [], "models_requested": 0,
              "models_returned": 0, "body_constructors": 0, "physics_steps": 0, "solver_calls": 0, "fit_calls": 0, "retries": 0}
    result = {"schema": RESULT_SCHEMA, "diagnostic_id": "E305", "run_valid": False, "drive_inference": INFERENCE,
              "candidate_failed_drive_directly_measured": False, "neural_only_snapshots": True, "RK_stage_arrays_measured": False,
              "neutral_RK4_byteexact_required": False, **FLAGS}
    states, factors = {}, {}
    def fail(error, stage):
        nonlocal primary
        ledger.setdefault("errors", []).append({"stage": stage, "type": type(error).__name__, "message": str(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E305 " + stage + ": " + repr(error))
        traceback.clear_frames(error.__traceback__)
    def persist():
        writer.json("ledger.json", ledger)
    def sample(phase):
        ledger["resource_samples"].append({"phase": phase, **deps.resource_check(output)})
    def deadline(science=False):
        if clock() - began >= LIMITS["science_wall_s" if science else "internal_wall_s"]:
            raise TimeoutError("E305 science deadline" if science else "E305 final deadline")
    try:
        writer.json("source-manifest.json", {"schema": "flylab.m3.e305.sources.v1", "diagnostic_id": "E305",
            "owned_sha256": {name: prior.sha256(ROOT / name) for name in OWNED}, "source_sha256": freeze.get("source_sha256", {}),
            "input_sha256": freeze.get("input_sha256", {}), "packages": freeze.get("packages", {}), "design": DESIGN, "design_sha256": DESIGN_SHA,
            "receipt_pins": {key: freeze.get(key) for key in RECEIPTS}, "limits": LIMITS, "module_global_changes": False})
        data = deps.load(freeze)
        for arm in ("neutral", "candidate"):
            deadline(True)
            row = {"advance_requested": 0, "advance_returned": 0, "actual_tick": None, "partial_tick_unknown": False,
                   "advance_failed": 0, "advance_wall_s": 0., "requested_steps": 20,
                   "cleanup_requested": 0, "cleanup_returned": 0, "initial_snapshot_available": False, "final_snapshot_available": False}
            ledger["arms"][arm] = row
            holder, network = {}, None
            def returned(value, holder=holder):
                holder["network"] = value
                ledger["models_returned"] += 1
            try:
                sample(arm + ":before_constructor")
                ledger["models_requested"] += 1
                persist()
                network = deps.construct(arm, data, ledger["construction_profiles"], persist, returned)
                if holder.get("network") is not network:
                    raise ValueError("Constructor ownership callback required")
                sample(arm + ":owned_after_constructor")
                initial = deps.initial(network, data, arm)
                writer.checkpoint(arm + "/initial-neural", initial)
                row["initial_snapshot_available"] = True
                sample(arm + ":owned_after_initial")
                deadline(True)
                row["advance_requested"] = 1
                persist()
                advance_began = clock()
                try:
                    network.advance(data["drive"], steps=20)
                finally:
                    prior.charge_wall(row, "advance_wall_s", advance_began, clock)
                row["advance_returned"] = 1
                final, g = deps.final(network, data, arm)
                writer.checkpoint(arm + "/final-neural", final)
                row["final_snapshot_available"] = True
                states[arm], factors[arm] = final, g
                writer.raw(arm + "-endpoint-factor.npz", {"source_ids": data["selections"][arm]["source_ids"], "g32": g})
                sample(arm + ":owned_after_advance")
                deadline(True)
            except Exception as error:
                row["advance_failed"] = int(row["advance_requested"] and not row["advance_returned"])
                if row["advance_failed"]:
                    row["partial_tick_unknown"] = True
                    row["tick_scope"] = "Public committed tick; failed CUDA graph internal partial work unobserved"
                if getattr(error, "resource_fault_flags", None) is not None:
                    row["resource_fault_flags"] = prior.json_safe(error.resource_fault_flags)
                fail(error, arm + ":science")
                if network is None:
                    network = holder.get("network")
                if network is not None:
                    try:
                        writer.checkpoint(arm + "/partial-neural", network.snapshot())
                        row["partial_neural_snapshot_available"] = True
                    except Exception as secondary:
                        row["partial_neural_snapshot_available"] = False
                        row["partial_snapshot_error"] = repr(secondary)
                        fail(secondary, arm + ":partial_snapshot")
            finally:
                if network is None:
                    network = holder.get("network")
                if network is not None:
                    try:
                        row["actual_tick"] = int(network.tick)
                        if row["actual_tick"] < 0 or row["actual_tick"] > 20:
                            raise ValueError("Actual arm tick exceeds fixed20 budget")
                    except Exception as error:
                        row["partial_tick_unknown"] = True
                        fail(error, arm + ":tick_cost")
                    try:
                        sample(arm + ":owned_before_release")
                    except Exception as error:
                        fail(error, arm + ":owned_resource_audit")
                    try:
                        deps.release(network)
                    except Exception as error:
                        fail(error, arm + ":release")
                holder.clear()
                network = None
                gc.collect()
                row["cleanup_requested"] = 1
                try:
                    deps.cleanup()
                    row["cleanup_returned"] = 1
                except Exception as error:
                    fail(error, arm + ":cleanup")
                try:
                    sample(arm + ":after_cleanup")
                except Exception as error:
                    fail(error, arm + ":post_cleanup_resources")
                try:
                    persist()
                except Exception as error:
                    fail(error, arm + ":ledger_publication")
            if primary is not None:
                break
        if primary is None:
            summary, arrays = {}, {}
            for key in ("rate", "adaptation", "drive", "output_mask"):
                summary["candidate_minus_neutral_" + key], arrays[key + "_delta"] = differences(states["candidate"][key], states["neutral"][key])
            summary["candidate_minus_neutral_g32"], arrays["g32_delta"] = differences(factors["candidate"], factors["neutral"])
            for key in ("rate", "adaptation"):
                summary["new_neutral_minus_original_RK4_" + key], arrays["neutral_minus_RK4_" + key] = differences(states["neutral"][key], data["rk4_reference"][key])
            result["numeric_comparisons"] = summary
            result["identities"] = {"source_timescale": SOURCE_IDENTITY, "candidate": freeze["candidate_network_identity"],
                "neutral": freeze["neutral_network_identity"], "original_RK4_neutral": data["rk4_reference"]["model_hash"]}
            result["endpoint_factor_scope"] = "Final rate/R derived diagnostic; not RK-stage factor history"
            writer.raw("numeric-deltas.npz", arrays)
            deadline()
            result["run_valid"] = True
    except Exception as error:
        fail(error, "execution")
    finally:
        if primary is not None:
            traceback.clear_frames(primary.__traceback__)
        if not ledger["arms"]:
            ledger["pre_arm_cleanup_requested"] = 1
            try:
                deps.cleanup()
                ledger["pre_arm_cleanup_returned"] = 1
                sample("pre_arm:after_cleanup")
            except Exception as error:
                fail(error, "pre_arm_cleanup")
        if "data" in locals():
            data = None
        result["elapsed_s"] = clock() - began
        ticks = [row["actual_tick"] for row in ledger["arms"].values() if row["actual_tick"] is not None]
        ledger["scientific_ticks"] = sum(ticks)
        ledger["scientific_ticks_scope"] = "Sum of public committed endpoint ticks; failed graph internal partial work may be unknown"
        ledger["scientific_ticks_requested"] = 20 * sum(row["advance_requested"] for row in ledger["arms"].values())
        ledger["unknown_tick_arms"] = [arm for arm, row in ledger["arms"].items() if row["actual_tick"] is None or row["partial_tick_unknown"]]
        if primary is None and (ledger["models_requested"] > 2 or ledger["scientific_ticks"] > 40 or ledger["unknown_tick_arms"]):
            fail(RuntimeError("E305 actual construction/tick cost unknown or exceeded"), "cost")
        try:
            deadline()
        except Exception as error:
            fail(error, "final_deadline")
        result["run_valid"] = bool(result["run_valid"] and primary is None)
        def update_result():
            result["cost"] = copy.deepcopy(ledger)
            if primary is not None:
                result["first_error"] = {"type": type(primary).__name__, "message": str(primary), "notes": list(getattr(primary, "__notes__", []))}
        update_result()
        try:
            persist()
            files = {file.relative_to(output).as_posix(): {"sha256": prior.sha256(file), "bytes": file.stat().st_size}
                     for file in output.rglob("*") if file.is_file() and file.name not in ("result.json", "output-manifest.json")}
            writer.json("output-manifest.json", {"schema": "flylab.m3.e305.outputs.v1", "files": files, "result_published_separately": True,
                                                   "result_sha_published_separately": True})
            result["elapsed_s"] = clock() - began
            writer.json("result.json", result)
            writer.json("result-sha256.json", {"sha256": prior.sha256(output / "result.json")})
            deadline()
        except Exception as error:
            fail(error, "terminal_publication")
            result["run_valid"] = False
            update_result()
            # Remove any earlier success receipt before publishing failure.
            for name in ("result.json", "result-sha256.json", "output-manifest.json"):
                try:
                    (output / name).unlink(missing_ok=True)
                except Exception as secondary:
                    fail(secondary, "invalidate_success_receipt")
            try:
                writer.json("result.json", result)
                writer.json("result-sha256.json", {"sha256": prior.sha256(output / "result.json")})
                writer.json("failure.json", result)
            except Exception as secondary:
                fail(secondary, "failure_publication")
        if primary is not None:
            traceback.clear_frames(primary.__traceback__)
            primary.e305_partial_result = result
            primary.e305_ledger = ledger
    return result, primary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "source-review", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    for key in ("freeze-sha256", "source-review-sha256"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    freeze = gate(args.freeze, args.freeze_sha256, args.source_review, args.source_review_sha256, args.output)
    import torch
    from threadpoolctl import threadpool_limits
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            result, error = execute(freeze, args.output)
        return 0 if error is None and result["run_valid"] else 1
    finally:
        torch.set_num_threads(previous)


if __name__ == "__main__":
    raise SystemExit(main())
