"""Existing-edge RM load-feedback efficacy comparison on the E150 trial.

The five recorded weights are rate-model float32 values, not anatomical
contact counts or measured physiological constants. No new neural path is made.
"""

from __future__ import annotations

import copy
import hashlib
import math
import struct
from typing import Any

import numpy as np
from scipy import sparse

from tools.banc_m3_monotone_recruitment import MonotoneRecruitmentTrial
from tools.banc_m3_motor_timescale import MotorTimescaleNetwork
from tools.probe_c_banc_m3_core_rhythm import source_matrix

PATCH_SCHEMA = "flylab.m3-rm-load-feedback-edges.v1"
TRIAL_SCHEMA = "flylab.m3-rm-load-feedback-walking.v1"
PROVENANCE_SCHEMA = "flylab.m3-rm-load-feedback-provenance.v1"
PATCH_KEYS = frozenset(
    {"schema", "edge_ids", "pre_ids", "post_ids", "source_values", "target_values"}
)
NETWORK_METADATA_KEYS = (
    "model_hash",
    "core_adaptation",
    "motor_adaptation",
    "motor_timescale",
)


def fixed_rm_load_patch() -> dict[str, Any]:
    """Return independent arrays for the five E161-verified rate-model edges."""
    return {
        "schema": PATCH_SCHEMA,
        "edge_ids": np.array(
            [844054, 2400285, 7908675, 8578714, 11240394], dtype=np.int64
        ),
        "pre_ids": np.array([107685, 30981, 107685, 30981, 30981], dtype=np.int64),
        "post_ids": np.array([12036, 32830, 108052, 117483, 153939], dtype=np.int64),
        "source_values": np.array(
            [0.00003, 0.00006, 0.00003, 0.00006, 0.00003], dtype=np.float32
        ),
        "target_values": np.array([0.03, 0.06, 0.03, 0.06, 0.03], dtype=np.float32),
    }


def _validated_patch(
    source: sparse.csr_matrix, patch: dict[str, Any]
) -> dict[str, Any]:
    if not sparse.isspmatrix_csr(source) or source.dtype != np.float32:
        raise ValueError("Square float32 CSR source required")
    source.check_format(full_check=True)
    if (
        source.shape[0] == 0
        or source.shape[0] != source.shape[1]
        or not source.has_canonical_format
        or not np.isfinite(source.data).all()
    ):
        raise ValueError("Canonical finite square float32 CSR source required")
    if (
        type(patch) is not dict
        or set(patch) != PATCH_KEYS
        or type(patch["schema"]) is not str
        or patch["schema"] != PATCH_SCHEMA
    ):
        raise ValueError("Complete RM load-feedback edge patch required")
    count = None
    for key, dtype in (
        ("edge_ids", np.int64),
        ("pre_ids", np.int64),
        ("post_ids", np.int64),
        ("source_values", np.float32),
        ("target_values", np.float32),
    ):
        value = patch[key]
        if type(value) is not np.ndarray or value.ndim != 1 or value.dtype != dtype:
            raise ValueError(f"Native one-dimensional {dtype} {key} required")
        if count is None:
            count = len(value)
        elif len(value) != count:
            raise ValueError("RM load-feedback edge patch lengths disagree")
    ids = patch["edge_ids"]
    pre = patch["pre_ids"]
    post = patch["post_ids"]
    before = patch["source_values"]
    after = patch["target_values"]
    if (
        not count
        or len(np.unique(ids)) != count
        or np.any((ids < 0) | (ids >= source.nnz))
        or np.any((pre < 0) | (pre >= source.shape[1]))
        or np.any((post < 0) | (post >= source.shape[0]))
        or not np.isfinite(before).all()
        or not np.isfinite(after).all()
        or np.any(before == 0)
        or np.any(after == 0)
        or np.any(np.signbit(before) != np.signbit(after))
    ):
        raise ValueError("Unique in-range same-sign nonzero finite edges required")
    rows = np.searchsorted(source.indptr, ids, side="right") - 1
    if (
        not np.array_equal(rows, post)
        or not np.array_equal(source.indices[ids], pre)
        or not np.array_equal(source.data[ids].view(np.uint32), before.view(np.uint32))
    ):
        raise ValueError("CSR endpoint or bit-exact source value disagrees")
    return copy.deepcopy(patch)


def patch_feedback_weights(
    source: sparse.csr_matrix, patch: dict[str, Any]
) -> sparse.csr_matrix:
    """Copy a CSR, changing only validated existing data entries."""
    checked = _validated_patch(source, patch)
    result = source.copy()
    result.data[checked["edge_ids"]] = checked["target_values"]
    return result


def _network_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    if type(snapshot) is not dict or any(
        key not in snapshot for key in NETWORK_METADATA_KEYS
    ):
        raise ValueError("Complete motor-timescale network metadata required")
    return {key: copy.deepcopy(snapshot[key]) for key in NETWORK_METADATA_KEYS}


def _digest_tree(value: Any) -> str:
    """Type- and byte-sensitive digest for an already materialized checkpoint."""
    digest = hashlib.sha256(b"flylab.m3-rm-load-feedback-checkpoint.v1")

    def put(raw: bytes) -> None:
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)

    def visit(item: Any) -> None:
        if type(item) is dict:
            put(b"dict")
            put(len(item).to_bytes(8, "little"))
            for key in sorted(item):
                if type(key) is not str:
                    raise TypeError("String checkpoint keys required")
                visit(key)
                visit(item[key])
        elif type(item) in (list, tuple):
            # StateStore serializes both sequences as JSON lists.
            put(b"list")
            put(len(item).to_bytes(8, "little"))
            for child in item:
                visit(child)
        elif type(item) is np.ndarray:
            if item.dtype.hasobject:
                raise TypeError("Object checkpoint arrays are unsupported")
            put(b"ndarray")
            put(item.dtype.str.encode("ascii"))
            put(repr(item.shape).encode("ascii"))
            put(item.tobytes(order="C"))
        elif isinstance(item, np.generic):
            # StateStore packs NumPy scalars using .item().
            visit(item.item())
        elif type(item) is str:
            put(b"str")
            put(item.encode("utf-8"))
        elif type(item) is bool:
            put(b"bool")
            put(b"1" if item else b"0")
        elif type(item) is int:
            put(b"int")
            put(str(item).encode("ascii"))
        elif type(item) is float:
            put(b"float")
            put(struct.pack("!d", item))
        elif item is None:
            put(b"none")
        else:
            raise TypeError(f"Unsupported checkpoint value: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def _checkpoint_digest(saved: dict[str, Any]) -> str:
    wall_s = saved.get("wall_s")
    if type(wall_s) is not float or not math.isfinite(wall_s) or wall_s < 0:
        raise ValueError("Finite nonnegative top-level runtime wall_s required")
    unsigned = copy.deepcopy(saved)
    unsigned["load_feedback_patch"].pop("snapshot_sha256", None)
    # Elapsed process time is restored, but is not part of the physical future.
    unsigned.pop("wall_s")
    return _digest_tree(unsigned)


def _exact(left: Any, right: Any, path: str = "checkpoint") -> None:
    """Compare exact values after StateStore's tuple/scalar normalization."""
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    if type(left) in (list, tuple) and type(right) in (list, tuple):
        if len(left) != len(right):
            raise ValueError(f"{path} length differs")
        for index, (a, b) in enumerate(zip(left, right)):
            _exact(a, b, f"{path}[{index}]")
        return
    if type(left) is not type(right):
        raise ValueError(f"{path} type differs")
    if type(left) is dict:
        if set(left) != set(right):
            raise ValueError(f"{path} keys differ")
        for key in left:
            _exact(left[key], right[key], f"{path}.{key}")
    elif type(left) is np.ndarray:
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or left.tobytes() != right.tobytes()
        ):
            raise ValueError(f"{path} bytes differ")
    elif type(left) is float:
        if struct.pack("!d", left) != struct.pack("!d", right):
            raise ValueError(f"{path} float bits differ")
    elif left != right:
        raise ValueError(f"{path} differs")


class LoadFeedbackTrial(MonotoneRecruitmentTrial):
    """Transfer an E150 trial's entire state to a newly built efficacy variant."""

    schema = TRIAL_SCHEMA

    def __init__(self, base: MonotoneRecruitmentTrial, patch: dict[str, Any]):
        if type(base) is not MonotoneRecruitmentTrial:
            raise TypeError("Exact original monotone recruitment trial required")
        if base.closed:
            raise ValueError("Open monotone recruitment trial required")
        self.__dict__.update(base.__dict__)
        base.closed = True  # The wrapper alone owns the same body from this point.
        try:
            old = self.network
            if type(old) is not MotorTimescaleNetwork:
                raise ValueError("Exact motor-timescale network layer required")
            source = source_matrix(old)
            checked = _validated_patch(source, patch)
            old_snapshot = old.snapshot()
            implementation = old.cuda_implementation
            if old.device == "cpu" and implementation == "torch-cpu-csr":
                implementation = "reference"
            candidate = MotorTimescaleNetwork(
                patch_feedback_weights(source, checked),
                *(
                    getattr(old, key).detach().cpu().numpy().copy()
                    for key in ("tau", "a", "threshold", "cap")
                ),
                adaptation_gain=old.adaptation_gain,
                adaptation_tau_s=old.adaptation_tau_s,
                dt=old.dt,
                device=old.device,
                cuda_implementation=implementation,
                capture_steps=old.interval_steps,
                core_ids=old.core_ids.copy(),
                core_gain=old.core_gain,
                motor_ids=old.motor_ids.copy(),
                motor_gains=old.motor_gains.copy(),
                motor_log_tau=old.motor_log_tau.copy(),
            )
            candidate_metadata = _network_metadata(candidate.snapshot())
            transferred = copy.deepcopy(old_snapshot)
            transferred.update(candidate_metadata)
            candidate.restore(transferred)
            self.load_feedback_patch = {
                "schema": PROVENANCE_SCHEMA,
                "edge_patch": checked,
                "source_network_metadata": _network_metadata(old_snapshot),
            }
            self.network = candidate
        except Exception as error:
            try:
                self.close()
            except Exception as close_error:  # noqa: BLE001
                error.add_note("Body cleanup failed: " + repr(close_error))
            raise

    def close(self) -> None:
        if not self.closed:
            super().close()

    def snapshot(self) -> dict[str, Any]:
        result = {
            **super().snapshot(),
            "load_feedback_patch": copy.deepcopy(self.load_feedback_patch),
        }
        result["load_feedback_patch"]["snapshot_sha256"] = _checkpoint_digest(result)
        return result

    @classmethod
    def from_checkpoint(cls, graph: Any, saved: dict[str, Any]) -> LoadFeedbackTrial:
        if type(saved) is not dict or saved.get("schema") != cls.schema:
            raise ValueError("Explicit load-feedback checkpoint required")
        provenance = saved.get("load_feedback_patch")
        if (
            type(provenance) is not dict
            or set(provenance)
            != {"schema", "edge_patch", "source_network_metadata", "snapshot_sha256"}
            or provenance["schema"] != PROVENANCE_SCHEMA
            or type(provenance["snapshot_sha256"]) is not str
            or provenance["snapshot_sha256"] != _checkpoint_digest(saved)
        ):
            raise ValueError("Complete, unmodified load-feedback provenance required")
        metadata = provenance["source_network_metadata"]
        if type(metadata) is not dict or set(metadata) != set(NETWORK_METADATA_KEYS):
            raise ValueError("Complete source network identity metadata required")
        if type(saved.get("network")) is not dict:
            raise ValueError("Complete saved network required")
        source = copy.deepcopy(saved)
        source.pop("load_feedback_patch")
        source["schema"] = MonotoneRecruitmentTrial.schema
        source["network"].update(copy.deepcopy(metadata))
        base = MonotoneRecruitmentTrial.from_checkpoint(graph, source)
        trial = None
        try:
            old_metadata = _network_metadata(base.network.snapshot())
            _exact(metadata, old_metadata, "source network metadata")
            trial = cls(base, provenance["edge_patch"])
            _exact(saved, trial.snapshot(), "whole load-feedback checkpoint")
            return trial
        except Exception as error:
            owner = trial if trial is not None else base
            try:
                if not owner.closed:
                    owner.close()
            except Exception as close_error:  # noqa: BLE001
                error.add_note("Body cleanup failed: " + repr(close_error))
            raise
