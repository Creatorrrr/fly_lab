"""E177 fixed five-edge native feedback comparison; one arm per process.

The preflight reads hashes and JSON metadata only. Original arrays and models
are entered only after the independent parent source/permit gates have passed.
"""

# ruff: noqa: BLE001
from __future__ import annotations

import argparse
import copy
import json
import shutil
import struct
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.walking_trace import contact_diagnostics
from tools import banc_m3_load_feedback as feedback
from tools import verify_c_banc_m3_native_monotone_body as e150
from tools import verify_c_banc_m3_native_timescale_transfer as protocol

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = "41cdb1d6e24d9b1fb6b50d17b2302b4f5e30ebb6a8c8c6e8e427eb4ae2ef8590"
DESIGN_SHA = "37115ac5feb1f38f06bc6ca5aee3b4e2a3e1905a161eab9bf077450c0e7f8179"
OWNED = (
    "tools/verify_c_banc_m3_native_feedback.py",
    "tests/test_c_banc_native_feedback.py",
)
OVERRIDES = (
    "MotorTimescaleTrial",
    "_observe",
    "_save_body_raw",
    "INTERNAL_LIMIT_S",
    "source_diagnostics",
)
FEEDBACK_IDS = (30981, 107685, 12036, 32830, 108052, 117483, 153939)
RAW_NAMES = (
    "trace.json",
    "native-recruitment-state.npz",
    "feedback-boundary-state.npz",
    "feedback-boundary-state.json",
    "native-contact.npz",
    "clone-native-contact.npz",
    "cut-restart-control.json",
    "cut-restart-native-state.npz",
    "cut-restart-native-contact.npz",
    "cut-restart-result.json",
)


class _PreflightComplete(Exception):
    """Leave the full-body branch while still executing fail-closed finalization."""


def safe_path(relative):
    name = Path(relative)
    path = (ROOT / name).resolve()
    if name.is_absolute() or ".." in name.parts or not path.is_relative_to(ROOT):
        raise ValueError("Repository-relative E177 path required")
    return path


def arm_patch(freeze, arm):
    """Build one independently owned exact float32 patch from the freeze."""
    if arm not in ("baseline", "candidate"):
        raise ValueError("Frozen E177 arm required")
    fixed = feedback.fixed_rm_load_patch()
    recorded = freeze["patch"]
    for key in ("edge_ids", "pre_ids", "post_ids", "source_values", "target_values"):
        if not np.array_equal(
            fixed[key], np.asarray(recorded[key], dtype=fixed[key].dtype)
        ):
            raise ValueError("Frozen five-edge patch disagrees with E176")
    if recorded["value_dtype"] != "<f4":
        raise ValueError("Frozen float32 patch dtype changed")
    if arm == "baseline":
        fixed["target_values"] = fixed["source_values"].copy()
    return fixed


def _source_review(freeze, freeze_sha256, path, digest):
    if protocol.sha256(path) != digest:
        raise ValueError("Exact parent E177 source review hash required")
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        review.get("schema") != "flylab.m3.e177.parent-source-review.v1"
        or review.get("accepts_single_native_feedback") is not True
        or review.get("blocking_defects") != []
        or review.get("freeze_sha256") != freeze_sha256
        or set(review.get("source_sha256", {})) != set(OWNED)
    ):
        raise ValueError("Accepted exact two-source E177 parent gate required")
    for name, expected in review["source_sha256"].items():
        if protocol.sha256(safe_path(name)) != expected:
            raise ValueError("Reviewed E177 source changed: " + name)
    return review


def baseline_eligible(result):
    """F4/F5 hypothesis misses never revoke a technically valid baseline."""
    body = result.get("body", {})
    cut = result.get("cut_restart", {})
    reproduction = result.get("baseline_reproduction", {})
    return bool(
        result.get("arm") == "baseline"
        and result.get("technical_failure") is False
        and result.get("resource_limit_met") is True
        and e150.full_main_complete(body)
        and body.get("strict_startup", {}).get("strict_E141_startup_screen") is True
        and cut.get("status") == "completed"
        and cut.get("accepted") is True
        and reproduction.get("exact") is True
    )


def combined_outcome(body, cut, arm, baseline_slip=None):
    """Separate measured refutations, technical faults, and slip evidence."""
    candidate_slip = body.get("supported_slip", {})
    comparison = {
        "schema": "flylab.m3.e177.supported-slip-comparison.v1",
        "baseline": baseline_slip,
        "candidate": candidate_slip if arm == "candidate" else None,
        "available": False,
        "candidate_no_worse": None,
    }
    if arm == "candidate":
        left = baseline_slip or {}
        right = candidate_slip or {}
        a, b = left.get("load_weighted_mm_s"), right.get("load_weighted_mm_s")
        available = bool(
            left.get("available") is True
            and right.get("available") is True
            and isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and np.isfinite(a)
            and np.isfinite(b)
        )
        comparison.update(
            available=available,
            candidate_no_worse=(bool(b <= a) if available else None),
        )
    if (
        body.get("validity") == "measured_native_health_stop"
        and body.get("technical_failure") is False
        and body.get("clock_valid") is True
    ):
        return "measured_native_health_stop", "refuted_health_stop", False, comparison
    if not e150.full_main_complete(body):
        return "incomplete", "invalid_not_tested", True, comparison
    if cut.get("status") != "completed" or cut.get("clock_valid") is not True:
        return "incomplete", "invalid_not_tested", True, comparison
    if cut.get("accepted") is not True:
        return (
            "valid_native_feedback_trial",
            "refuted_motor_cut_restart",
            False,
            comparison,
        )
    verdict = body.get("hypothesis_verdict", "invalid_not_tested")
    if arm == "candidate" and verdict == "supported_development_diagnostic":
        if not comparison["available"]:
            verdict = "indeterminate_missing_slip_evidence"
        elif not comparison["candidate_no_worse"]:
            verdict = "refuted_supported_slip"
    return "valid_native_feedback_trial", verdict, False, comparison


def _baseline_review(freeze, freeze_sha256, source_digest, source_review, path, digest):
    if not path or not digest or protocol.sha256(path) != digest:
        raise ValueError("Exact parent E177 baseline permit required")
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        review.get("schema") != "flylab.m3.e177.parent-baseline-review.v1"
        or review.get("permits_candidate") is not True
        or review.get("blocking_defects") != []
        or review.get("freeze_sha256") != freeze_sha256
        or review.get("source_review_sha256") != source_digest
        or review.get("source_sha256") != source_review["source_sha256"]
    ):
        raise ValueError("Parent baseline permit binding changed")
    baseline = safe_path(freeze["output_directories"]["baseline"])
    if protocol.sha256(baseline / "result.json") != review.get(
        "baseline_result_sha256"
    ) or protocol.sha256(baseline / "output-manifest.json") != review.get(
        "baseline_manifest_sha256"
    ):
        raise ValueError("Baseline result or manifest changed after permit")
    manifest = json.loads(
        (baseline / "output-manifest.json").read_text(encoding="utf-8")
    )
    raw = review.get("baseline_raw_sha256")
    files = manifest.get("files", {})
    if (
        not isinstance(raw, dict)
        or not isinstance(files, dict)
        or not set(RAW_NAMES).issubset(raw)
        or set(raw) != set(files)
        or manifest.get("file_count") != len(files)
        or manifest.get("total_bytes")
        != sum(v.get("bytes", -1) for v in files.values())
    ):
        raise ValueError("Complete baseline manifest-byte permit required")
    actual_files = {
        p.relative_to(baseline).as_posix()
        for p in baseline.rglob("*")
        if p.is_file() and p.name not in {"result.json", "output-manifest.json"}
    }
    if actual_files != set(files):
        raise ValueError("Baseline manifest file set changed")
    for name, expected in raw.items():
        if (
            protocol.sha256(baseline / name) != expected
            or files[name].get("sha256") != expected
            or (baseline / name).stat().st_size != files[name].get("bytes")
        ):
            raise ValueError("Baseline raw bytes changed: " + name)
    result = json.loads((baseline / "result.json").read_text(encoding="utf-8"))
    if not baseline_eligible(result):
        raise ValueError("Baseline technical/reproduction/startup/cut gate failed")
    return review


def verified_inputs(
    freeze_path,
    freeze_sha256,
    source_review_path,
    source_review_sha256,
    *,
    arm,
    baseline_review_path=None,
    baseline_review_sha256=None,
):
    """Hash all originals before StateStore, GraphStore, NPZ values or CUDA."""
    if freeze_sha256 != FREEZE_SHA or protocol.sha256(freeze_path) != FREEZE_SHA:
        raise ValueError("Exact E177 freeze required")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    fixed = {
        "schema": "flylab.m3.e177.native-feedback.v1",
        "design_sha256": DESIGN_SHA,
        "owned_files": list(OWNED),
        "allowed_protocol_global_overrides": list(OVERRIDES),
        "body_protocol": "E141-code-with-isolated-factory-and-scoped-diagnostics",
        "arms": ["baseline", "candidate"],
        "prepare_wall_limit_s": 60,
        "body_wall_limit_s": 180,
        "internal_wall_limit_s": 250,
        "external_wall_limit_s": 270,
        "supervisor_limit_s": 285,
        "maximum_raw_bytes": 4 << 30,
        "minimum_free_disk_bytes": 12 << 30,
        "original_processes": 2,
        "automatic_retries": 0,
        "prior_conservative_actual_s": 3085.0114084999077,
        "prior_conservative_process_s": 3086,
        "prospective_cumulative_limit_s": 3656,
        "candidate_adopted": False,
        "whole_goal_complete": False,
    }
    if (
        arm not in freeze.get("arms", ())
        or any(freeze.get(k) != v for k, v in fixed.items())
        or protocol.sha256(safe_path(freeze["design_path"])) != DESIGN_SHA
        or len(freeze.get("input_sha256", {})) != 102
        or len(freeze.get("reused_source_sha256", {})) != 90
        or len(freeze.get("preservation_sha256", {})) != 2013
    ):
        raise ValueError("Frozen E177 execution contract changed")
    arm_patch(freeze, arm)
    review = _source_review(
        freeze, freeze_sha256, source_review_path, source_review_sha256
    )
    baseline_review = None
    if arm == "candidate":
        baseline_review = _baseline_review(
            freeze,
            freeze_sha256,
            source_review_sha256,
            review,
            baseline_review_path,
            baseline_review_sha256,
        )
    elif baseline_review_path or baseline_review_sha256:
        raise ValueError("Baseline arm cannot consume candidate permit")
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        protocol._exact_mapping_hashes(freeze[label], label)
    return freeze, review, baseline_review


def csr_fingerprint(network):
    return e150.csr_fingerprint(network)


def verify_csr_patch(before, after, patch, arm):
    """Inspect the complete CSR, allowing only five chosen data words."""
    for key in ("shape", "indptr", "indices"):
        if before[key] != after[key]:
            raise ValueError("CSR structure changed: " + key)
    before_data = np.asarray(before["data"])
    after_data = np.asarray(after["data"])
    if before_data.dtype != np.float32 or after_data.dtype != np.float32:
        raise ValueError("float32 CSR data required")
    if before_data.shape != after_data.shape:
        raise ValueError("CSR data shape changed")
    expected = before_data.copy()
    expected[patch["edge_ids"]] = patch["target_values"]
    if expected.tobytes() != after_data.tobytes():
        raise ValueError("Non-selected or incorrect CSR bytes changed")
    if arm == "baseline" and before_data.tobytes() != after_data.tobytes():
        raise ValueError("Baseline identity patch changed CSR")
    return {
        "before_sha256": protocol._hash_array(before_data),
        "after_sha256": protocol._hash_array(after_data),
        "selected_before": before_data[patch["edge_ids"]].tolist(),
        "selected_after": after_data[patch["edge_ids"]].tolist(),
        "only_selected_five": True,
    }


def verify_wrapper_state(source, wrapped, patch):
    """Prove whole E150 state after removing only four network identities."""
    provenance = wrapped["load_feedback_patch"]
    feedback._exact(patch, provenance["edge_patch"], "E177 exact arm patch")
    plain = copy.deepcopy(wrapped)
    plain.pop("load_feedback_patch")
    plain["schema"] = source["schema"]
    for key, value in provenance["source_network_metadata"].items():
        plain["network"][key] = value
    feedback._exact(source, plain, "E177 whole unwrapped source state")
    return True


class TrialFactory:
    def __init__(self, patch):
        self.patch = copy.deepcopy(patch)
        self.source_restores = 0
        self.wrapper_restores = 0
        self.checks = []

    def from_checkpoint(self, graph, saved):
        if saved.get("schema") == feedback.TRIAL_SCHEMA:
            self.wrapper_restores += 1
            feedback._exact(
                self.patch,
                saved["load_feedback_patch"]["edge_patch"],
                "E177 restored arm patch",
            )
            trial = feedback.LoadFeedbackTrial.from_checkpoint(graph, saved)
            try:
                feedback._exact(
                    saved, trial.snapshot(), "E177 restored whole checkpoint"
                )
                return trial
            except BaseException as error:
                try:
                    if not trial.closed:
                        trial.close()
                except BaseException as secondary:
                    error.add_note(
                        "Secondary E177 restored owner cleanup: " + repr(secondary)
                    )
                raise
        if saved.get("schema") != e150.monotone.TRIAL_SCHEMA:
            raise ValueError("Only E150/E176 checkpoint schemas accepted")
        self.source_restores += 1
        base = e150.monotone.MonotoneRecruitmentTrial.from_checkpoint(graph, saved)
        owner = base
        try:
            source = base.snapshot()
            feedback._exact(saved, source, "E177 restored E150 whole source")
            source_csr = protocol.source_matrix(base.network)
            before = {
                "shape": source_csr.shape,
                "indptr": protocol._hash_array(source_csr.indptr),
                "indices": protocol._hash_array(source_csr.indices),
                "data": source_csr.data.copy(),
            }
            trial = feedback.LoadFeedbackTrial(base, self.patch)
            owner = trial
            wrapped = trial.snapshot()
            verify_wrapper_state(source, wrapped, self.patch)
            current = protocol.source_matrix(trial.network)
            after = {
                "shape": current.shape,
                "indptr": protocol._hash_array(current.indptr),
                "indices": protocol._hash_array(current.indices),
                "data": current.data.copy(),
            }
            receipt = verify_csr_patch(
                before,
                after,
                self.patch,
                "baseline"
                if np.array_equal(
                    self.patch["source_values"], self.patch["target_values"]
                )
                else "candidate",
            )
            if trial.body is not base.body:
                raise ValueError("E176 body object changed")
            self.checks.append(receipt)
            return trial
        except BaseException as error:
            try:
                if not owner.closed:
                    owner.close()
            except BaseException as secondary:
                error.add_note("Secondary E177 owner cleanup: " + repr(secondary))
            raise

    def metadata(self):
        return {
            "source_restores": self.source_restores,
            "wrapper_restores": self.wrapper_restores,
            "checks": copy.deepcopy(self.checks),
        }


def _tensor_values(value, ids):
    return value[np.asarray(ids, np.int64)].detach().cpu().numpy().copy()


def make_observer(records, feedback_rows, ids=FEEDBACK_IDS):
    """One recurrent current evaluation serves legacy and seven-cell raw."""
    ids = np.asarray(ids, np.int64)
    if ids.shape != (7,) or tuple(ids.tolist()) != FEEDBACK_IDS:
        raise ValueError("Fixed seven feedback cell IDs required")

    def observe(session, detailed, available=None):
        if available is None:
            available = {}
        row = {
            "control_tick": int(session.control_tick),
            "network_tick": int(session.network.tick),
            "complete": False,
        }
        feedback_rows.append(row)
        available["e177_feedback"] = row
        record = {
            "control_tick": row["control_tick"],
            "network_tick": row["network_tick"],
            "complete": False,
        }
        records.append(record)
        available["e150_native"] = record
        try:
            row["physics_time_s"] = float(session.body.physics_time())
            record["time_s"] = row["physics_time_s"]
            record["bank"] = e150._copy_array(session.adapter.bank, (391, 3))
            record["activation"] = e150._copy_array(session.adapter.activation, (96,))
            record["original_activation"] = e150._copy_array(
                session.adapter.original.activation, (96,)
            )
            motor_ids = np.asarray(session.adapter.calibration["motor_ids"])
            if motor_ids.shape != (391,) or motor_ids.dtype.kind not in "iu":
                raise ValueError("E150 391-MN identity required")
            record["motor_rate"] = e150._copy_array(
                session.network.readout(motor_ids), (391,)
            )
            net = session.network
            current = net._current(net.rate)
            fields = available.setdefault("fields", {})
            fields["rate"] = net.readout(detailed)
            fields["recurrent_current"] = _tensor_values(current, detailed)
            fields["drive"] = _tensor_values(net.drive, detailed)
            fields["adaptation"] = _tensor_values(net.adaptation, detailed)
            fields["muscle_activation"] = record["activation"].copy()
            available["complete"] = True
            row["rate"] = _tensor_values(net.rate, ids)
            row["drive"] = _tensor_values(net.drive, ids)
            row["adaptation"] = _tensor_values(net.adaptation, ids)
            row["output_mask"] = _tensor_values(net.output_mask, ids)
            row["recurrent_current"] = _tensor_values(current, ids)
            edge_ids = feedback.fixed_rm_load_patch()["edge_ids"]
            pre_index = np.asarray([1, 0, 1, 0, 0], np.int64)
            row["edge_weight"] = _tensor_values(net.weights.values(), edge_ids)
            row["direct_edge_current"] = row["edge_weight"] * (
                row["rate"][pre_index] * row["output_mask"][pre_index]
            )
            load = contact_diagnostics(session.body)["normal_load_bw"]
            row["rm_normal_load_bw"] = np.asarray(load[4], np.float64)
            record["complete"] = True
            row["complete"] = True
            return fields
        except BaseException as error:
            row["error"] = repr(error)
            record["error"] = repr(error)
            raise

    return observe


def save_feedback_state(path, receipt_path, rows):
    shapes = {
        "rate": (7,),
        "drive": (7,),
        "adaptation": (7,),
        "output_mask": (7,),
        "recurrent_current": (7,),
        "edge_weight": (5,),
        "direct_edge_current": (5,),
        "rm_normal_load_bw": (),
    }
    payload = {
        "schema": np.array("flylab.m3.e177.feedback-boundary.v1"),
        "observed_ids": np.asarray(FEEDBACK_IDS, np.int64),
        "control_tick": np.asarray([r["control_tick"] for r in rows], np.int64),
        "network_tick": np.asarray([r["network_tick"] for r in rows], np.int64),
        "physics_time_s": np.asarray(
            [r.get("physics_time_s", np.nan) for r in rows], np.float64
        ),
        "complete": np.asarray([r["complete"] for r in rows], bool),
    }
    availability = {}
    for key, shape in shapes.items():
        dtype = next((np.asarray(r[key]).dtype for r in rows if key in r), np.float64)
        data = np.zeros((len(rows), *shape), dtype=dtype)
        mask = np.zeros(len(rows), bool)
        for i, row in enumerate(rows):
            if key in row:
                value = np.asarray(row[key])
                if value.shape != shape or value.dtype != dtype:
                    raise ValueError("E177 feedback field shape/dtype changed: " + key)
                data[i] = value
                mask[i] = True
        payload[key] = data
        payload[key + "_available"] = mask
        availability[key] = int(mask.sum())
    np.savez_compressed(path, **payload)
    protocol.write_json(
        receipt_path,
        {
            "schema": "flylab.m3.e177.feedback-boundary-receipt.v1",
            "rows": len(rows),
            "complete_rows": sum(bool(r["complete"]) for r in rows),
            "availability_counts": availability,
            "dtypes": {key: str(value.dtype) for key, value in payload.items()},
            "shapes": {key: list(value.shape) for key, value in payload.items()},
            "partial": [
                {
                    "row": i,
                    "fields": sorted(k for k in shapes if k in r),
                    "error": r.get("error"),
                }
                for i, r in enumerate(rows)
                if not r["complete"]
            ],
            "boundary_semantics": "completed control; held drive; instantaneous recurrent/direct current and RM load",
        },
    )


def make_raw_saver(records, feedback_rows, arm=None):
    def save(out, *args, **kwargs):
        errors = []
        primary = None
        try:
            errors.extend(protocol._save_body_raw(out, *args, **kwargs))
        except BaseException as error:
            primary = error
        for label, callback in (
            (
                "native-recruitment-state.npz",
                lambda: e150.save_native_state(
                    Path(out) / "native-recruitment-state.npz", records
                ),
            ),
            (
                "feedback-boundary-state.npz",
                lambda: save_feedback_state(
                    Path(out) / "feedback-boundary-state.npz",
                    Path(out) / "feedback-boundary-state.json",
                    feedback_rows,
                ),
            ),
            (
                "trace.json E177 labels",
                lambda: relabel_trace(Path(out) / "trace.json", arm),
            ),
        ):
            try:
                callback()
            except BaseException as error:
                errors.append(
                    {
                        "artifact": label,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                if primary is not None:
                    primary.add_note("Secondary " + label + ": " + repr(error))
        if primary is not None:
            raise primary
        return errors

    return save


def relabel_trace(path, arm=None):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = copy.deepcopy(value["samples"])
    prior = {"schema": value.get("schema"), "controller": value.get("controller")}
    value.update(
        schema="flylab.m3.e177.native-feedback-trace.v1",
        controller=f"E177 fixed-five {arm or 'arm'} native feedback",
        E177_provenance={"arm": arm, "legacy_metadata": prior},
    )
    protocol.write_json(path, value)
    if json.loads(Path(path).read_text(encoding="utf-8"))["samples"] != samples:
        raise ValueError("E177 trace relabel changed samples")


def scoped_source_diagnostics(_base, out, _cal, _neutral, health, reason):
    protocol.write_json(
        Path(out) / "e177-source-diagnostics.json",
        {
            "schema": "flylab.m3.e177.scoped-source-diagnostics.v1",
            "complete_native_health": bool(health),
            "stop_reason": reason,
            "teacher_arrays_read": False,
            "legacy_source_diagnostics_called": False,
        },
    )


def isolated_native_body(
    factory, observer, raw_saver, diagnostics=scoped_source_diagnostics
):
    source = protocol._native_body
    namespace = source.__globals__.copy()
    values = {
        "MotorTimescaleTrial": factory,
        "_observe": observer,
        "_save_body_raw": raw_saver,
        "INTERNAL_LIMIT_S": 250.0,
        "source_diagnostics": diagnostics,
    }
    if tuple(values) != OVERRIDES:
        raise RuntimeError("E177 exact five overrides changed")
    namespace.update(values)
    function = types.FunctionType(
        source.__code__,
        namespace,
        source.__name__,
        source.__defaults__,
        source.__closure__,
    )
    function.__kwdefaults__ = copy.copy(source.__kwdefaults__)
    function.e177_overrides = tuple(values)
    return function


def _strict_json_equal(left, right, label):
    if type(left) is not type(right):
        raise ValueError(label + " type changed")
    if isinstance(left, dict):
        if set(left) != set(right):
            raise ValueError(label + " keys changed")
        for key in left:
            _strict_json_equal(left[key], right[key], label + "." + key)
    elif isinstance(left, list):
        if len(left) != len(right):
            raise ValueError(label + " length changed")
        for i, (a, b) in enumerate(zip(left, right)):
            _strict_json_equal(a, b, f"{label}[{i}]")
    elif isinstance(left, float):
        if not np.isfinite(left) or struct.pack("!d", left) != struct.pack("!d", right):
            raise ValueError(label + " float bits changed/nonfinite")
    elif left != right:
        raise ValueError(label + " changed")


def compare_baseline_reproduction(reference_dir, baseline_dir, freeze):
    """Exact 2001 E150 rows; physical clock alone has a 2e-8 tolerance."""
    reference_dir, baseline_dir = Path(reference_dir), Path(baseline_dir)
    contract = freeze["baseline_reproduction"]
    with (
        np.load(
            reference_dir / "native-recruitment-state.npz", allow_pickle=False
        ) as ref,
        np.load(
            baseline_dir / "native-recruitment-state.npz", allow_pickle=False
        ) as got,
    ):
        if len(ref["control_tick"]) != contract["rows"] or set(ref.files) != set(
            got.files
        ):
            raise ValueError("E177 baseline state rows/fields differ")
        for name in contract["state_exact"]:
            a, b = ref[name], got[name]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                raise ValueError("E177 baseline state differs: " + name)
            if a.dtype.kind in "fc" and (
                not np.isfinite(a).all() or not np.isfinite(b).all()
            ):
                raise ValueError("E177 baseline nonfinite state: " + name)
        a, b = ref["physics_time_s"], got["physics_time_s"]
        if (
            a.dtype != b.dtype
            or a.shape != b.shape
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
            or not np.allclose(a, b, rtol=0, atol=contract["physics_time_atol_s"])
        ):
            raise ValueError("E177 baseline physical time differs")
    original = json.loads((reference_dir / "trace.json").read_text(encoding="utf-8"))[
        "samples"
    ]
    actual = json.loads((baseline_dir / "trace.json").read_text(encoding="utf-8"))[
        "samples"
    ]
    if len(original) != contract["rows"] or len(actual) != contract["rows"]:
        raise ValueError("E177 baseline trace rows differ")
    for i, (a, b) in enumerate(zip(original, actual, strict=True)):
        for field in contract["trace_exact"]:

            def get(row, path=field):
                value = row
                for part in path.split("."):
                    value = value[part]
                return value

            _strict_json_equal(get(a), get(b), f"trace[{i}].{field}")
    return {
        "schema": "flylab.m3.e177.baseline-reproduction.v1",
        "exact": True,
        "rows": contract["rows"],
        "physics_time_atol_s": contract["physics_time_atol_s"],
        "state_exact": contract["state_exact"],
        "trace_exact": contract["trace_exact"],
    }


def _prepare(freeze, arm, out, ledger, persist):
    ref = safe_path(freeze["reference_directory"])
    for name in (
        "assembled-calibration.npz",
        "assembled-parameters.npz",
        "actual-global-csr.npz",
        "muscle-provenance.json",
    ):
        protocol.accounted_operation(
            "copy_" + name,
            lambda n=name: shutil.copyfile(ref / n, out / n),
            ledger["storage_operations"],
            persist,
        )
    source = protocol.accounted_operation(
        "load_E150_zero",
        lambda: protocol.StateStore.load(safe_path(freeze["zero_checkpoint"])),
        ledger["storage_operations"],
        persist,
    )
    graph = protocol.accounted_operation(
        "load_graph",
        lambda: protocol.GraphStore.load(safe_path(freeze["graph_path"])),
        ledger["storage_operations"],
        persist,
    )
    if graph.hash != freeze["graph_hash"] or graph.n != 158706:
        raise ValueError("Original graph identity changed")
    params = protocol.accounted_operation(
        "load_E150_parameters",
        lambda: protocol.load_npz(ref / "assembled-parameters.npz"),
        ledger["storage_operations"],
        persist,
    )
    header = freeze["input_headers"][
        freeze["reference_directory"] + "/assembled-parameters.npz"
    ]
    if set(params) != set(header) or any(
        params[key].shape != tuple(meta["shape"])
        or params[key].dtype.str != meta["dtype"]
        or (params[key].dtype.kind in "fc" and not np.isfinite(params[key]).all())
        for key, meta in header.items()
    ):
        raise ValueError("Frozen E150 assembled parameter shape/dtype changed")
    cal = protocol.accounted_operation(
        "load_E150_calibration",
        lambda: protocol.load_npz(ref / "assembled-calibration.npz"),
        ledger["storage_operations"],
        persist,
    )
    if not np.array_equal(params["motor_ids"], cal["motor_ids"]):
        raise ValueError("E150 assembled motor IDs changed")
    observation = protocol.accounted_operation(
        "load_observation",
        lambda: protocol.load_npz(safe_path(freeze["observation_path"])),
        ledger["storage_operations"],
        persist,
    )
    startup = protocol.accounted_operation(
        "load_startup",
        lambda: protocol.load_npz(safe_path(freeze["startup_path"])),
        ledger["storage_operations"],
        persist,
    )
    original_csr = protocol.accounted_operation(
        "load_frozen_full_csr",
        lambda: protocol.load_npz(ref / "actual-global-csr.npz"),
        ledger["storage_operations"],
        persist,
    )
    factory = TrialFactory(arm_patch(freeze, arm))
    owner = None
    primary = None
    neutral = None
    try:
        owner = protocol.profiled_construction(
            "prepare_E177_" + arm,
            lambda: factory.from_checkpoint(graph, source),
            ledger["construction_attempts"],
            persist,
        )
        neutral = owner.snapshot()
        live = factory.checks[-1]
        expected = {
            "shape": [158706, 158706],
            **{
                k: protocol._hash_array(original_csr[k])
                for k in ("indptr", "indices", "data")
            },
        }
        source_live = protocol.source_matrix(owner.network)
        selected = factory.patch["edge_ids"]
        data = source_live.data.copy()
        data[selected] = factory.patch["source_values"]
        actual_source = {
            "shape": list(source_live.shape),
            "indptr": protocol._hash_array(source_live.indptr),
            "indices": protocol._hash_array(source_live.indices),
            "data": protocol._hash_array(data),
        }
        protocol.exact(expected, actual_source, "E177 full source CSR equals E150")
        arm_csr = {
            "indptr": source_live.indptr.copy(),
            "indices": source_live.indices.copy(),
            "data": source_live.data.copy(),
        }
        protocol.accounted_operation(
            "save_actual_arm_full_csr",
            lambda: np.savez_compressed(out / "actual-global-csr.npz", **arm_csr),
            ledger["storage_operations"],
            persist,
        )
        protocol.write_json(
            out / "wrapper-provenance.json",
            {
                "schema": "flylab.m3.e177.wrapper-provenance.v1",
                "source_csr": expected,
                "patch_receipt": live,
                "whole_unwrapped_state_exact": True,
                "source_restores": factory.source_restores,
            },
        )
        protocol.accounted_operation(
            "save_E177_neutral",
            lambda: protocol.StateStore.save(out / "candidate-neutral", neutral),
            ledger["storage_operations"],
            persist,
        )
        roundtrip = protocol.accounted_operation(
            "reload_E177_neutral",
            lambda: protocol.StateStore.load(out / "candidate-neutral"),
            ledger["storage_operations"],
            persist,
        )
        feedback._exact(neutral, roundtrip, "E177 whole neutral disk roundtrip")
    except BaseException as error:
        primary = error
    finally:
        if owner is not None:
            primary = e150.close_preserving(
                primary, "E177 prepared owner close", owner.close
            )
    if primary is not None:
        raise primary
    return {
        "graph": graph,
        "neutral": neutral,
        "factory": factory,
        "parameters": params,
        "cal": cal,
        "observation": observation,
        "startup": startup,
        "anatomy": json.loads(
            (ref / "muscle-provenance.json").read_text(encoding="utf-8")
        ),
    }


def resource_checks(
    freeze,
    *,
    prepare_wall,
    body_wall,
    process_wall,
    raw_bytes,
    free_disk_before,
    clock_valid,
):
    return [
        name
        for failed, name in (
            (
                free_disk_before < freeze["minimum_free_disk_bytes"],
                "minimum_free_disk_bytes",
            ),
            (prepare_wall > freeze["prepare_wall_limit_s"], "prepare_wall_limit_s"),
            (body_wall > freeze["body_wall_limit_s"], "body_wall_limit_s"),
            (process_wall > freeze["internal_wall_limit_s"], "internal_wall_limit_s"),
            (raw_bytes > freeze["maximum_raw_bytes"], "maximum_raw_bytes"),
            (not clock_valid, "clock_valid"),
            (
                freeze["prior_conservative_process_s"]
                + 2 * freeze["supervisor_limit_s"]
                > freeze["prospective_cumulative_limit_s"],
                "prospective_cumulative_limit_s",
            ),
        )
        if failed
    ]


def record_scientific_cost(result, body, cut):
    """Retain main work even if cut construction, advance, or persistence fails."""
    for suffix in ("neural_s", "physical_s"):
        main = body.get("actual_" + suffix)
        extra = cut.get("actual_" + suffix)
        result["actual_scientific_" + suffix] = (
            main + extra if main is not None and extra is not None else None
        )
    result["scientific_cost_by_stage"] = {
        "main_and_clone": {
            key: body.get(key) for key in ("actual_neural_s", "actual_physical_s")
        },
        "cut_restart": {
            key: cut.get(key) for key in ("actual_neural_s", "actual_physical_s")
        },
    }


def run(
    freeze_path,
    freeze_sha256,
    source_review_path,
    source_review_sha256,
    arm,
    output,
    *,
    baseline_review_path=None,
    baseline_review_sha256=None,
    preflight_only=False,
):
    began = time.perf_counter()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("Preserve one E177 arm output: " + str(out))
    out.mkdir(parents=True)
    ledger = {
        "schema": "flylab.m3.e177.native-feedback-ledger.v1",
        "status": "created",
        "arm": arm,
        "construction_attempts": [],
        "storage_operations": [],
        "secondary_errors": [],
        "conditional_forward_calls": 0,
        "fit_calls": 0,
        "LP_calls": 0,
        "automatic_retries": 0,
        "candidate_adopted": False,
        "whole_goal_complete": False,
    }
    result = {
        "schema": "flylab.m3.e177.native-feedback-result.v1",
        "arm": arm,
        "validity": "incomplete",
        "hypothesis_verdict": "invalid_not_tested",
        "technical_failure": True,
        "body_started": False,
        "candidate_adopted": False,
        "whole_goal_complete": False,
        "M3_complete": False,
        "F6_exposed": False,
    }
    primary = None
    entered = False
    limits = None
    resource_failures = []

    def persist():
        ledger["construction_totals"] = protocol._construction_totals(
            ledger["construction_attempts"]
        )
        ledger["process_wall_s"] = time.perf_counter() - began
        protocol.write_json(out / "advance-ledger.json", ledger)

    def absorb(error, label):
        nonlocal primary
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary " + label + ": " + repr(error))
            ledger["secondary_errors"].append({"stage": label, "error": repr(error)})

    try:
        freeze, review, permit = verified_inputs(
            freeze_path,
            freeze_sha256,
            source_review_path,
            source_review_sha256,
            arm=arm,
            baseline_review_path=baseline_review_path,
            baseline_review_sha256=baseline_review_sha256,
        )
        protocol.write_json(out / "source-review-snapshot.json", review)
        if permit is not None:
            protocol.write_json(out / "baseline-review-snapshot.json", permit)
        if preflight_only:
            ledger["status"] = "metadata_preflight_complete"
            result.update(
                validity="metadata_preflight_only",
                technical_failure=False,
                hypothesis_verdict="not_tested",
                preflight_only=True,
            )
            raise _PreflightComplete
        if out != safe_path(freeze["output_directories"][arm]):
            raise ValueError("Frozen one-shot E177 arm output directory required")
        free_disk = shutil.disk_usage(out).free
        ledger["free_disk_before_original_bytes"] = free_disk
        if free_disk < freeze["minimum_free_disk_bytes"]:
            resource_failures.append("minimum_free_disk_bytes")
            raise OSError("E177 requires 12 GiB free")
        limits, threads = protocol.joint_fit.one_cpu_thread()
        ledger["cpu_threads"] = threads
        entered = True
        ledger["cuda_runtime_entered"] = True
        persist()
        prepared = _prepare(freeze, arm, out, ledger, persist)
        prepare_wall = time.perf_counter() - began
        ledger["prepare_wall_s"] = prepare_wall
        result["prepare_wall_s"] = prepare_wall
        if prepare_wall > freeze["prepare_wall_limit_s"]:
            resource_failures.append("prepare_wall_limit_s")
            raise TimeoutError("E177 preparation exceeded 60 seconds")
        spec = {
            "experiment": "E177",
            "arm": arm,
            "body_protocol": freeze["body_protocol"],
            "freeze_sha256": freeze_sha256,
            "source_review_sha256": source_review_sha256,
            "baseline_review_sha256": baseline_review_sha256,
            "candidate_adopted": False,
            "whole_goal_complete": False,
        }
        protocol.write_json(out / "spec.json", spec)
        records, feedback_rows = [], []
        observer = make_observer(records, feedback_rows)
        raw_saver = make_raw_saver(records, feedback_rows, arm)
        native_body = isolated_native_body(prepared["factory"], observer, raw_saver)
        ledger["isolated_global_overrides"] = list(native_body.e177_overrides)
        ledger["status"] = "native_body_entered"
        result["body_started"] = True
        persist()
        body = native_body(
            freeze,
            prepared["graph"],
            prepared["cal"],
            prepared["observation"],
            prepared["startup"],
            prepared["neutral"],
            prepared["parameters"],
            prepared["anatomy"],
            out,
            spec,
            ledger,
            persist,
            began,
        )
        result["body"] = body
        result["cut_restart"] = {
            "status": "not_requested",
            "actual_neural_s": 0.0,
            "actual_physical_s": 0.0,
        }
        record_scientific_cost(result, body, result["cut_restart"])
        if e150.full_main_complete(body) and (out / "checkpoint-1s").exists():
            try:
                cut = e150.run_cut_restart(
                    out, prepared["graph"], prepared["factory"], ledger, persist, began
                )
            except BaseException as error:
                cut = {
                    "status": "failed",
                    "accepted": False,
                    "actual_neural_s": None,
                    "actual_physical_s": None,
                    "clock_valid": False,
                }
                try:
                    receipt = json.loads(
                        (out / "cut-restart-ledger.json").read_text(encoding="utf-8")
                    )
                    clock = receipt.get("clock", {})
                    cut.update(
                        actual_neural_s=clock.get("neural_s"),
                        actual_physical_s=clock.get("physical_s"),
                        clock_valid=clock.get("valid") is True,
                        partial_cost_source="cut-restart-ledger.json",
                    )
                except BaseException as secondary:
                    error.add_note("Partial cut cost unavailable: " + repr(secondary))
                result["cut_restart"] = cut
                record_scientific_cost(result, body, cut)
                raise
        else:
            cut = {
                "status": "not_run_main_incomplete",
                "accepted": False,
                "actual_neural_s": 0.0,
                "actual_physical_s": 0.0,
            }
            protocol.write_json(out / "cut-restart-result.json", cut)
        result["cut_restart"] = cut
        record_scientific_cost(result, body, cut)
        if arm == "baseline" and e150.full_main_complete(body):
            reproduction = compare_baseline_reproduction(
                safe_path(freeze["reference_directory"]), out, freeze
            )
            protocol.write_json(out / "baseline-reproduction.json", reproduction)
            result["baseline_reproduction"] = reproduction
        baseline_slip = None
        if arm == "candidate":

            def load_baseline_slip():
                base = safe_path(freeze["output_directories"]["baseline"])
                evidence = json.loads(
                    (base / "supported-slip.json").read_text(encoding="utf-8")
                )
                baseline_result = json.loads(
                    (base / "result.json").read_text(encoding="utf-8")
                )
                _strict_json_equal(
                    evidence,
                    baseline_result["body"]["supported_slip"],
                    "permitted baseline supported slip",
                )
                return evidence

            baseline_slip = protocol.accounted_operation(
                "load_permitted_baseline_slip",
                load_baseline_slip,
                ledger["storage_operations"],
                persist,
            )
        validity, verdict, technical, slip_comparison = combined_outcome(
            body, cut, arm, baseline_slip
        )
        if arm == "candidate":
            protocol.write_json(
                out / "E177-supported-slip-comparison.json", slip_comparison
            )
        result.update(
            technical_failure=technical,
            validity=validity,
            hypothesis_verdict=verdict,
            supported_slip_comparison=slip_comparison,
            factory=prepared["factory"].metadata(),
            prepare_wall_s=prepare_wall,
        )
        if body.get("body_analysis_wall_s", 0) > freeze["body_wall_limit_s"]:
            resource_failures.append("body_wall_limit_s")
            raise TimeoutError("E177 body exceeded 180 seconds")
        ledger["status"] = "result_ready"
    except _PreflightComplete:
        pass
    except BaseException as error:
        primary = error
        ledger["status"] = "technical_failed"
        ledger["primary_error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "notes": list(getattr(error, "__notes__", [])),
        }
    finally:
        if entered:
            for label, error in protocol.cleanup_cuda_runtime(True, ledger):
                absorb(error, label)
        if limits is not None:
            try:
                limits.restore_original_limits()
            except BaseException as error:
                absorb(error, "thread restore")
        try:
            persist()
        except BaseException as error:
            absorb(error, "ledger persistence")
        elapsed = time.perf_counter() - began
        result.update(
            process_wall_s=elapsed,
            prior_conservative_actual_s=3085.0114084999077,
            prior_conservative_process_s=3086,
            this_arm_plus_prior_estimate_s=3086 + elapsed,
            parent_supervisor_cumulative_authoritative=True,
            resource_limit_met=not resource_failures,
            resource_failures=resource_failures,
            construction_accounting=ledger.get("construction_totals"),
            candidate_adopted=False,
            whole_goal_complete=False,
            M3_complete=False,
        )
        if primary is not None:
            result.update(
                validity="invalid_resource" if resource_failures else "incomplete",
                technical_failure=True,
                hypothesis_verdict="invalid_not_tested",
                primary_error={
                    "type": type(primary).__name__,
                    "message": str(primary),
                    "notes": list(getattr(primary, "__notes__", [])),
                },
            )
        preliminary = copy.deepcopy(result)
        preliminary.update(
            validity="incomplete",
            technical_failure=True,
            hypothesis_verdict="invalid_not_tested",
            finalization_pending=True,
        )
        try:
            protocol.write_json(out / "result.json", preliminary)
        except BaseException as error:
            absorb(error, "preliminary result persistence")
        try:
            manifest = protocol._persist_manifest(out)
            result["output_manifest"] = manifest
            raw_bytes = protocol._raw_bytes(out)
            result["raw_bytes"] = raw_bytes
            if "body" in result:
                checks = resource_checks(
                    freeze,
                    prepare_wall=result["prepare_wall_s"],
                    body_wall=result["body"].get("body_analysis_wall_s", 0),
                    process_wall=time.perf_counter() - began,
                    raw_bytes=raw_bytes,
                    free_disk_before=ledger["free_disk_before_original_bytes"],
                    clock_valid=bool(
                        result["body"].get("clock_valid")
                        and (
                            result["cut_restart"]
                            .get("status", "")
                            .startswith("not_run")
                            or result["cut_restart"].get("clock_valid")
                        )
                    ),
                )
                resource_failures.extend(checks)
                if checks:
                    absorb(
                        RuntimeError("E177 resource checks: " + ",".join(checks)),
                        "resource checks",
                    )
            elif raw_bytes > 4 << 30:
                resource_failures.append("maximum_raw_bytes")
        except BaseException as error:
            absorb(error, "manifest/raw measurement")
        if time.perf_counter() - began > 250:
            resource_failures.append("internal_wall_limit_s")
            absorb(TimeoutError("E177 internal 250-second limit"), "late wall guard")
        result.update(
            process_wall_s=time.perf_counter() - began,
            this_arm_plus_prior_estimate_s=3086 + time.perf_counter() - began,
            resource_limit_met=not resource_failures,
            resource_failures=list(dict.fromkeys(resource_failures)),
            finalization_pending=False,
        )
        if primary is not None:
            result.update(
                technical_failure=True,
                validity="invalid_resource" if resource_failures else "incomplete",
                hypothesis_verdict="invalid_not_tested",
                primary_error={
                    "type": type(primary).__name__,
                    "message": str(primary),
                    "notes": list(getattr(primary, "__notes__", [])),
                },
            )
        try:
            protocol.write_json(out / "result.json", result)
        except BaseException as error:
            absorb(error, "final result persistence")
    if primary is not None:
        raise primary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "freeze",
        "freeze-sha256",
        "source-review",
        "source-review-sha256",
        "arm",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--baseline-review")
    parser.add_argument("--baseline-review-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.freeze,
                args.freeze_sha256,
                args.source_review,
                args.source_review_sha256,
                args.arm,
                args.output,
                baseline_review_path=args.baseline_review,
                baseline_review_sha256=args.baseline_review_sha256,
                preflight_only=args.preflight_only,
            )
        )
    )


if __name__ == "__main__":
    main()
