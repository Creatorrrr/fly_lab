"""E189: two fixed local perturbations of one relay-to-MN CSR edge.

Import and --help are inert. The parent owns the freeze, review, receipts and
sequential process supervisor. This child runs exactly one arm after its gate.
"""

# Broad catches preserve primary native failures and partial raw evidence.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_ENTRY_STARTED = time.perf_counter()

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import banc_m3_resource_readiness as resource_readiness

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
DESIGN = "docs/BANC_M3_E189_DIRECT_MN_EDGE_DESIGN_20260924.md"
METADATA = BASE + "/parent-e189-frozen-metadata.json"
SUPERVISOR = BASE + "/parent-e189-supervise-v2.py"
TERMINAL_RECOVERY = "docs/BANC_M3_E189_TERMINAL_RECOVERY_20260924.md"
OUTPUT_ROOT = BASE + "/e189-single-edge"
OWNED = ("tools/probe_c_banc_m3_single_mn_edge.py", "tests/test_c_banc_single_mn_edge.py")
FREEZE_SCHEMA = "flylab.m3.e189.single-mn-edge.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e189.single-mn-edge.source-review.v1"
METADATA_SCHEMA = "flylab.m3.e189.single-mn-edge.metadata.v1"
RECEIPT_SCHEMA = "flylab.m3.e189.single-mn-edge.parent-input-validation.v1"
EDGE = 10511590
PRE = 54014
POST = 144018
SOURCE = np.float32(0.2087680697441101)
ARM_WEIGHTS = {
    arm: float(np.float32(np.float64(SOURCE) * np.float64(multiplier)))
    for arm, multiplier in (("low", 0.98), ("high", 1.02))
}
PARAMETERS = {
    "control_dt_s": 0.005,
    "control_rows": 601,
    "native_rows": 30000,
    "neural_steps": 12000,
    "physics_steps": 30000,
    "cpu_threads": 1,
    "blas_threads": 1,
    "gpu_count": 1,
    "body_wall_s": 100,
    "child_internal_wall_s": 140,
    "parent_wall_s": 145,
    "cleanup_reserve_s": 5,
    "automatic_retries": 0,
    "maximum_output_bytes": 2 << 30,
    "maximum_body_bytes": 1 << 30,
    "minimum_free_bytes": 20 << 30,
    "cumulative_before_s": 4452.92451379972,
    "cumulative_cap_s": 4994,
    "stage_reserve_s": 150,
    "total_reserve_s": 300,
}
LIMITS = {"q_RMS": 0.02, "q_max_abs": 0.04, "LTM_RMS": 0.02, "LTM_max_abs": 0.04}
NETWORK_IDENTITY_KEYS = ("model_hash", "core_adaptation", "motor_adaptation", "motor_timescale")
EXPECTED_E182_CLIENTS = frozenset(("16248", "35496"))


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(name: str) -> Path:
    require(isinstance(name, str) and name and "\\" not in name, "unsafe path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ":" not in name and all(
        part not in ("", ".", "..") for part in relative.parts
    ), "unsafe path")
    path = ROOT.joinpath(*relative.parts)
    require(path.resolve().is_relative_to(ROOT.resolve()), "path escaped repository")
    require(not any(p.is_symlink() for p in (path, *path.parents) if p != ROOT.parent), "symlink source")
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(type(value) is dict, "JSON object required")
    return value


def write_new(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def input_stamps(inputs: dict[str, str]) -> dict[str, list[int]]:
    result = {}
    for name in inputs:
        stat = safe(name).stat()
        result[name] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
    return result


def readiness_clear(row: dict[str, Any], parent_pid: int | None = None,
                    *, monitor: bool = False) -> bool:
    """Recheck source fields from E188 resource helper, not only its ready flag."""
    try:
        clients = row["clients"]
        telemetry = resource_readiness._gpu_telemetry(row)
        common = (row.get("phase") == ("monitor" if monitor else "readiness")
                  and row.get("ready") is (not monitor)
                  and row.get("resource_clear") is True
                  and not row.get("client_error")
                  and not row.get("gpu_error")
                  and not row.get("gpu_telemetry_error")
                  and row.get("gpu_telemetry") == telemetry
                  and type(clients) is dict
                  and set(clients) == EXPECTED_E182_CLIENTS
                  and all(value in ("absent", "reused") for value in clients.values())
                  and bool(telemetry))
        if not common:
            return False
        if row.get("resource_mode") == "queue_empty":
            return row.get("queue") == resource_readiness.EMPTY_QUEUE and not row.get("queue_error")
        if row.get("resource_mode") == "service_stopped":
            proof = row.get("service_probe", {})
            family = proof.get("allowed_process_family")
            python = proof.get("python_processes")
            current = proof.get("current_pid")
            if (type(current) is not int or (parent_pid is not None and current != parent_pid)
                    or type(family) is not list or not family
                    or type(python) is not list or not python):
                return False
            family_by_pid = {item.get("pid"): item for item in family
                             if type(item) is dict}
            if (len(family_by_pid) != len(family)
                    or sum(item.get("relation") == "self" for item in family) != 1
                    or family_by_pid.get(current, {}).get("relation") != "self"
                    or any(type(item.get("pid")) is not int
                           or type(item.get("create_time")) not in (int, float)
                           or item.get("relation") not in ("self", "ancestor", "descendant")
                           for item in family)
                    or any(type(item) is not dict
                           or item.get("pid") not in family_by_pid
                           or item.get("create_time") != family_by_pid[item["pid"]]["create_time"]
                           for item in python)):
                return False
            terminal = proof.get("ancestor_terminal")
            if type(terminal) is not dict or terminal.get("reason") not in ("root", "absent_parent"):
                return False
            if terminal["reason"] == "absent_parent" and terminal.get("verified_absent") is not True:
                return False
            for item in family:
                if (item["pid"] <= 0 or type(item.get("ppid")) is not int
                        or item["ppid"] < 0 or not math.isfinite(item["create_time"])
                        or item["create_time"] <= 0):
                    return False
            ancestors: set[int] = set()
            cursor = current
            while family_by_pid[cursor]["ppid"] in family_by_pid:
                parent = family_by_pid[cursor]["ppid"]
                if (parent == current or parent in ancestors
                        or family_by_pid[parent]["create_time"]
                        > family_by_pid[cursor]["create_time"]):
                    return False
                ancestors.add(parent)
                cursor = parent
            missing = family_by_pid[cursor]["ppid"]
            if not ((missing == 0 and terminal == {"reason": "root", "pid": 0})
                    or (missing > 0 and terminal == {
                        "reason": "absent_parent", "pid": missing,
                        "child_pid": cursor, "verified_absent": True})):
                return False
            descendants = {current}
            while True:
                before = len(descendants)
                for pid, item in family_by_pid.items():
                    if pid not in descendants and item["ppid"] in descendants:
                        if family_by_pid[item["ppid"]]["create_time"] > item["create_time"]:
                            return False
                        descendants.add(pid)
                if len(descendants) == before:
                    break
            if (set(family_by_pid) != ancestors | descendants
                    or any(family_by_pid[pid]["relation"] != "ancestor" for pid in ancestors)
                    or any(family_by_pid[pid]["relation"] != "descendant"
                           for pid in descendants - {current})
                    or not any(item["pid"] == current for item in python)
                    or any(item.get("ppid") != family_by_pid[item["pid"]]["ppid"]
                           for item in python)):
                return False
            return bool(row.get("queue_error") and proof.get("tcp_connect") == "refused"
                        and proof.get("listener_inventory_complete") is True
                        and proof.get("listener_8188") is False
                        and proof.get("listeners_8188") == []
                        and proof.get("process_inventory_complete") is True
                        and proof.get("foreign_python") == []
                        and not proof.get("port_error") and not proof.get("process_error")
                        and (monitor or all(gpu["utilization_pct"] <= 5
                                            and gpu["memory_used_mib"] <= 2048
                                            for gpu in telemetry)))
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return False
    return False


@dataclass(frozen=True)
class E189Gate:
    arm: str
    weight: np.float32
    metadata: dict[str, Any]
    freeze: dict[str, Any]
    freeze_sha256: str
    review_sha256: str
    receipt_sha256: str
    readiness: list[dict[str, Any]]


def validate_receipt(
    receipt: dict[str, Any], receipt_sha: str, arm: str, output: Path,
    freeze_sha: str, review_sha: str, metadata_sha: str, inputs: dict[str, str],
) -> list[dict[str, Any]]:
    """Bind the current child, arm, assets and fresh parent readiness evidence."""
    import psutil

    require(len(receipt_sha) == 64, "parent receipt SHA")
    require(
        receipt.get("schema") == RECEIPT_SCHEMA
        and receipt.get("validated") is True
        and receipt.get("arm") == arm
        and receipt.get("output_path") == f"{OUTPUT_ROOT}/{arm}"
        and output.resolve() == safe(receipt["output_path"]).resolve()
        and receipt.get("freeze_sha256") == freeze_sha
        and receipt.get("source_review_sha256") == review_sha
        and receipt.get("metadata_sha256") == metadata_sha
        and receipt.get("input_sha256") == inputs
        and receipt.get("input_stamps") == input_stamps(inputs),
        "parent receipt contract",
    )
    parent = receipt.get("parent")
    require(type(parent) is dict and type(parent.get("pid")) is int, "parent process")
    direct_pid = os.getppid()
    if direct_pid != parent["pid"]:
        launcher = psutil.Process(direct_pid)
        require(
            os.name == "nt" and launcher.is_running()
            and launcher.ppid() == parent["pid"]
            and Path(launcher.exe()).resolve() == Path(sys.executable).resolve()
            and launcher.cmdline()[1:] == psutil.Process().cmdline()[1:],
            "different parent process or unverified venv launcher",
        )
    live = psutil.Process(parent["pid"])
    require(live.is_running() and abs(live.create_time() - parent["create_time"]) <= 0.001,
            "parent process identity changed")
    script = safe(SUPERVISOR).resolve()
    require(any(
        (Path(live.cwd()) / arg).resolve() == script
        for arg in live.cmdline() if str(arg).endswith(".py")
    ), "unexpected parent supervisor command")
    require(
        0 <= time.perf_counter() - receipt["issued_monotonic_s"] <= 145,
        "expired parent receipt",
    )
    readiness = receipt.get("resource_readiness_observations")
    require(
        type(readiness) is list and len(readiness) == 2
        and all(type(row) is dict and readiness_clear(row, parent["pid"])
                for row in readiness)
        and 0 <= time.perf_counter() - receipt["resource_readiness_finished_monotonic_s"] <= 10,
        "fresh two-observation readiness required",
    )
    return readiness


def gate(
    freeze_path: Path, review_path: Path, arm: str, output: Path,
    parent_validation_sha256: str | None = None,
) -> E189Gate:
    """Verify this E189 approval and exact inputs before any model import."""
    require(arm in ARM_WEIGHTS, "only fixed low/high arms allowed")
    require(not output.exists() and output.parent.is_dir(), "new arm output required")
    freeze, review = read_json(freeze_path), read_json(review_path)
    frozen_sha, review_sha = sha256(freeze_path), sha256(review_path)
    require(
        freeze.get("schema") == FREEZE_SCHEMA
        and review.get("schema") == REVIEW_SCHEMA
        and review.get("approved") is True
        and review.get("blocking_defects") == []
        and review.get("freeze_sha256") == frozen_sha,
        "E189 parent source approval",
    )
    require(
        freeze.get("design_path") == DESIGN
        and freeze.get("design_sha256") == sha256(safe(DESIGN))
        and freeze.get("metadata_path") == METADATA
        and freeze.get("metadata_sha256") == sha256(safe(METADATA))
        and freeze.get("output_root") == OUTPUT_ROOT
        and output.resolve() == safe(f"{OUTPUT_ROOT}/{arm}").resolve()
        and freeze.get("parameters") == PARAMETERS
        and freeze.get("arm_weights") == ARM_WEIGHTS
        and freeze.get("terminal_recovery_sha256") == sha256(safe(TERMINAL_RECOVERY))
        and freeze.get("parent_supervisor_sha256") == sha256(safe(SUPERVISOR)),
        "E189 frozen design/resources/arm",
    )
    metadata = read_json(safe(METADATA))
    require(
        metadata.get("schema") == METADATA_SCHEMA
        and metadata.get("design_path") == DESIGN
        and metadata.get("design_sha256") == freeze["design_sha256"]
        and metadata.get("owned_files") == list(OWNED)
        and metadata.get("original_native_runs") == 0
        and metadata.get("F6_exposed") == 0
        and metadata.get("candidate_adopted") is False
        and metadata.get("cumulative_before_s") == PARAMETERS["cumulative_before_s"]
        and metadata.get("cumulative_limit_s") == PARAMETERS["cumulative_cap_s"],
        "E189 metadata contract",
    )
    previous_path = metadata.get("e188_final_review_path")
    require(
        previous_path == BASE + "/parent-e188-final-review.json"
        and metadata.get("e188_final_review_sha256") == sha256(safe(previous_path)),
        "E188 final review hash",
    )
    previous = read_json(safe(previous_path))
    require(
        previous.get("schema") == "flylab.m3.e188.parent-final-review.v1"
        and previous.get("approved") is True
        and previous.get("run_validity") == "valid_all_incoming_observation"
        and previous.get("baseline_raw_exact") is True
        and previous.get("candidate_selection") == "not_selected"
        and previous.get("event521_post_shared_control") == 530,
        "E188 final review not accepted",
    )
    sources, inputs = freeze.get("source_hashes"), freeze.get("input_sha256")
    require(
        type(sources) is dict and {*OWNED, "tools/banc_m3_load_feedback.py"} <= set(sources)
        and type(inputs) is dict and inputs
        and inputs == metadata.get("inputs")
        and review.get("source_hashes") == sources
        and review.get("design_sha256") == freeze["design_sha256"]
        and review.get("metadata_sha256") == freeze["metadata_sha256"]
        and type(metadata.get("reused_sources")) is dict
        and set(metadata["reused_sources"]) <= set(sources),
        "E189 source/input inventory",
    )
    for name, expected in sources.items():
        require(type(expected) is str and len(expected) == 64 and sha256(safe(name)) == expected,
                "source changed: " + name)
    for name, expected in metadata["reused_sources"].items():
        require(sources[name] == expected, "reused source changed: " + name)
    for name, expected in inputs.items():
        require(type(expected) is str and len(expected) == 64,
                "invalid frozen input hash: " + name)
        if parent_validation_sha256 is None:
            require(sha256(safe(name)) == expected, "original input changed: " + name)
    require(
        inputs.get(previous_path) == metadata["e188_final_review_sha256"],
        "E188 review absent from original input set",
    )
    readiness: list[dict[str, Any]] = []
    if parent_validation_sha256 is not None:
        receipt_path = safe(f"{BASE}/parent-e189-{arm}-input-validation.json")
        require(sha256(receipt_path) == parent_validation_sha256, "parent receipt hash")
        readiness = validate_receipt(
            read_json(receipt_path), parent_validation_sha256, arm, output,
            frozen_sha, review_sha, freeze["metadata_sha256"], inputs,
        )
    return E189Gate(
        arm, np.float32(ARM_WEIGHTS[arm]), metadata, freeze,
        frozen_sha, review_sha, parent_validation_sha256 or "", readiness,
    )


def edge_patch(arm: str) -> dict[str, Any]:
    """Only one existing, same-sign relay54014 to MN144018 efficacy."""
    from tools.banc_m3_load_feedback import PATCH_SCHEMA

    require(arm in ARM_WEIGHTS, "arm")
    return {
        "schema": PATCH_SCHEMA,
        "edge_ids": np.array([EDGE], np.int64),
        "pre_ids": np.array([PRE], np.int64),
        "post_ids": np.array([POST], np.int64),
        "source_values": np.array([SOURCE], np.float32),
        "target_values": np.array([ARM_WEIGHTS[arm]], np.float32),
    }


def exact_csr_and_packed(original: Any, candidate: Any, arm: str) -> dict[str, Any]:
    """Check every CSR word and every packed decoded entry at zero clock."""
    from tools.probe_c_banc_m3_core_rhythm import source_matrix

    source, changed = source_matrix(original), source_matrix(candidate)
    require(
        source.shape == changed.shape == (158706, 158706)
        and source.data.dtype == changed.data.dtype == np.float32
        and source.nnz == changed.nnz and EDGE < source.nnz,
        "full float32 CSR dimensions",
    )
    require(
        np.array_equal(source.indptr, changed.indptr)
        and np.array_equal(source.indices, changed.indices)
        and int(np.searchsorted(source.indptr, EDGE, side="right") - 1) == POST
        and int(source.indices[EDGE]) == PRE,
        "whole CSR topology or selected endpoint changed",
    )
    before, after = source.data.view(np.uint32), changed.data.view(np.uint32)
    differing = np.flatnonzero(before != after)
    require(
        np.array_equal(differing, np.array([EDGE]))
        and before[EDGE] == SOURCE.view(np.uint32)
        and after[EDGE] == np.float32(ARM_WEIGHTS[arm]).view(np.uint32),
        "CSR must differ by selected float32 word only",
    )
    require(
        candidate.device == "cuda" and candidate.cuda_implementation == "packed"
        and candidate.n == 158706,
        "full packed CUDA network required",
    )
    rowptr = candidate._rowptr.detach().cpu().numpy().copy()
    columns = candidate._columns.detach().cpu().numpy().copy().view(np.uint32)
    lookup = candidate._weight_values.detach().cpu().numpy().copy()
    require(
        rowptr.shape == changed.indptr.shape
        and np.array_equal(rowptr.astype(np.int64), changed.indptr.astype(np.int64))
        and columns.shape == changed.indices.shape
        and lookup.dtype == np.float32,
        "whole packed rowptr/column table",
    )
    decoded_pre = columns & np.uint32((1 << 18) - 1)
    lookup_ids = columns >> np.uint32(18)
    require(
        np.array_equal(decoded_pre.astype(np.int64), changed.indices.astype(np.int64))
        and np.all(lookup_ids < len(lookup))
        and np.array_equal(lookup[lookup_ids].view(np.uint32), changed.data.view(np.uint32)),
        "whole packed decoded pre/weight differs from CSR",
    )
    return {
        "schema": "flylab.m3.e189.csr-packed-proof.v1",
        "edge_id": EDGE,
        "pre_id": PRE,
        "post_id": POST,
        "source_weight": float(SOURCE),
        "target_weight": ARM_WEIGHTS[arm],
        "csr_edges": int(source.nnz),
        "changed_words": 1,
        "whole_csr_topology_equal": True,
        "whole_packed_rowptr_pre_weight_equal_new_csr": True,
        "csr_indptr_sha256": hashlib.sha256(
            np.asarray(changed.indptr, np.int64).tobytes()).hexdigest(),
        "csr_indices_sha256": hashlib.sha256(
            np.asarray(changed.indices, np.int64).tobytes()).hexdigest(),
        "source_csr_values_sha256": hashlib.sha256(source.data.tobytes()).hexdigest(),
        "candidate_csr_values_sha256": hashlib.sha256(changed.data.tobytes()).hexdigest(),
        "packed_decoded_pre_sha256": hashlib.sha256(
            decoded_pre.astype(np.int64).tobytes()).hexdigest(),
        "packed_decoded_weight_sha256": hashlib.sha256(
            lookup[lookup_ids].tobytes()).hexdigest(),
        "selected_source_float32_word": int(before[EDGE]),
        "selected_candidate_float32_word": int(after[EDGE]),
        "device": str(candidate.device),
        "cuda_implementation": candidate.cuda_implementation,
    }


def exact_state_transfer(source: dict[str, Any], candidate: dict[str, Any]) -> None:
    """Compare every dynamic field; allow four identity keys and wrapper provenance."""
    from tools.banc_m3_load_feedback import _exact

    expected = copy.deepcopy(source)
    # Runtime accounting advances during CUDA construction, not physical time.
    expected["wall_s"] = candidate["wall_s"]
    expected["schema"] = candidate["schema"]
    for key in NETWORK_IDENTITY_KEYS:
        expected["network"][key] = copy.deepcopy(candidate["network"][key])
    expected["load_feedback_patch"] = copy.deepcopy(candidate["load_feedback_patch"])
    _exact(expected, candidate, "E189 transferred whole state")


def construct_arm(graph: Any, neutral: dict[str, Any], arm: str) -> tuple[Any, dict[str, Any]]:
    from tools import optimize_c_banc_m3_native_timing as e181
    from tools.banc_m3_load_feedback import LoadFeedbackTrial
    from tools.banc_m3_monotone_recruitment import MonotoneRecruitmentTrial

    require(arm in ARM_WEIGHTS, "arm")
    e181._neutral(neutral)
    base = MonotoneRecruitmentTrial.from_checkpoint(graph, neutral)
    owner = None
    try:
        e181.protocol.exact(neutral, base.snapshot(), "E189 exact E150 zero state")
        original_body = base.body
        original_body_hash = base.body.model_hash
        original_network = base.network
        owner = LoadFeedbackTrial(base, edge_patch(arm))
        require(owner.body is original_body and owner.body.model_hash == original_body_hash,
                "body object/model changed")
        proof = exact_csr_and_packed(original_network, owner.network, arm)
        snapshot = owner.snapshot()
        exact_state_transfer(neutral, snapshot)
        require(
            graph.n == 158706 and owner.network.tick == owner.control_tick == 0
            and abs(owner.body.physics_time()) <= 1e-8,
            "nonzero initial neural/control/body clock",
        )
        return owner, proof
    except BaseException as error:
        try:
            (owner or base).close()
        except BaseException as cleanup_error:
            error.add_note("Secondary E189 construction cleanup: " + repr(cleanup_error))
        raise


def selected_signal(session: Any) -> dict[str, Any]:
    """Read the chosen edge and MN without advancing neural or body time."""
    network = session.network
    before = (session.control_tick, network.tick, session.body.physics_time())
    fields = ("rate", "drive", "adaptation", "output_mask")
    originals = {key: getattr(network, key).clone() for key in fields}
    current = network._current(network.rate)
    rate = network.rate[[PRE, POST]].detach().cpu().numpy().copy()
    mask = network.output_mask[PRE : PRE + 1].detach().cpu().numpy().copy()
    mn_current = current[POST : POST + 1].detach().cpu().numpy().copy()
    require(
        before == (session.control_tick, network.tick, session.body.physics_time())
        and all(bool(network.torch.equal(originals[key], getattr(network, key)))
                for key in fields), "signal read changed state or clock",
    )
    return {
        "control_tick": int(before[0]),
        "pre_rate": float(rate[0]),
        "mn_rate": float(rate[1]),
        "pre_mask": float(mask[0]),
        "mn_current": float(mn_current[0]),
        "muscle67_activation": float(np.asarray(session.adapter.activation)[67]),
    }


EVENTS = (131, 219, 308, 391, 433, 480, 521, 564)
RM_COXA_ROLL = 29


def validate_rm_path(session: Any) -> None:
    names = session.body.joint_names
    moment = np.asarray(session.adapter.calibration["moment"])
    motor = np.asarray(session.adapter.calibration["motor_ids"])
    groups = session.adapter.groups
    require(len(names) == 42 and str(names[RM_COXA_ROLL]).endswith(
        "c_thorax-rm_coxa-roll"), "RM coxa-roll anatomical joint row")
    require(moment.shape == (42, 96)
            and np.array_equal(np.flatnonzero(moment[RM_COXA_ROLL]), [67])
            and float(moment[RM_COXA_ROLL, 67]) == 1.0
            and motor.shape == (391,) and int(motor[364]) == POST
            and groups[67]["columns"] == [364],
            "MN144018 / muscle67 / RM coxa-roll singleton path")


def fixed_force_summary(force: dict[str, np.ndarray],
                        native: dict[str, np.ndarray]) -> dict[str, Any]:
    """Use literal native clocks; E186 control bins are separately labelled."""
    times = np.asarray(force["time_s"])
    forward = np.asarray(force["leg_heading_xy_force_bw"])
    available = np.asarray(force["finite"])
    load = np.asarray(native["normal_load_bw"])
    slip = np.asarray(native["slip_mm_s"])
    slip_valid = np.asarray(native["slip_valid"])
    require(times.shape == (30000,) and forward.shape == load.shape == slip.shape
            == slip_valid.shape == (30000, 6)
            and np.allclose(times, np.arange(1, 30001) * 0.0001, rtol=0, atol=2e-8)
            and np.allclose(times, native["time_s"], rtol=0, atol=1e-12)
            and available.shape == (30000,) and bool(np.all(available))
            and np.asarray(force["available"]).shape == (30000,)
            and np.all(force["available"]) and np.all(force["error"] == "")
            and np.isfinite(forward).all() and np.isfinite(load).all(),
            "complete finite native force/contact clock")
    require(slip_valid.dtype == np.bool_ and np.all(load >= 0)
            and np.all(slip_valid[load >= 0.02])
            and np.isfinite(slip[load >= 0.02]).all(),
            "every supported native slip must be measured")

    def interval(mask: np.ndarray) -> dict[str, Any]:
        values = forward[mask]
        supported = (load[mask] >= 0.02)
        valid = supported & slip_valid[mask]
        numerator_by_leg = np.sum(
            load[mask] * np.where(valid, slip[mask], 0.0), axis=0)
        denominator_by_leg = np.sum(load[mask] * valid, axis=0)
        numerator = float(np.sum(numerator_by_leg))
        denominator = float(np.sum(denominator_by_leg))
        return {
            "samples": int(mask.sum()),
            "signed_ap_impulse_bw_s": (np.sum(values, axis=0) * 0.0001).tolist(),
            "negative_ap_magnitude_impulse_bw_s": (
                np.sum(np.maximum(-values, 0), axis=0) * 0.0001).tolist(),
            "contact_samples": (np.sum(load[mask] > 0, axis=0)).astype(int).tolist(),
            "support_samples": np.sum(supported, axis=0).astype(int).tolist(),
            "slip_valid_support_samples": np.sum(valid, axis=0).astype(int).tolist(),
            "slip_weighted_numerator": numerator,
            "slip_weighted_denominator": denominator,
            "slip_weighted_mm_s": numerator / denominator if denominator > 0 else None,
            "slip_weighted_numerator_by_leg": numerator_by_leg.tolist(),
            "slip_weighted_denominator_by_leg": denominator_by_leg.tolist(),
            "slip_weighted_mm_s_by_leg": [
                float(n / d) if d > 0 else None
                for n, d in zip(numerator_by_leg, denominator_by_leg)],
            "unsupported_slip_samples": int(np.sum(supported & ~slip_valid[mask])),
            "nonfoot_contact_samples": int(np.sum(native["nonfoot_contact"][mask])),
            "upright_min": float(np.min(native["upright"][mask])),
        }

    main = (times >= 0.5) & (times < 3.0)
    require(bool(main.any()), "native [0.5,3) sample count")
    owner_bins = np.zeros(len(times), dtype=bool)
    owner_bins[(100 - 1) * 50:599 * 50] = True
    posts = {}
    post_union = np.zeros(len(times), dtype=bool)
    for control in EVENTS:
        # E186 control k owns native indices (k-1)*50 through k*50-1.
        start, stop = (control - 1) * 50, (control + 9) * 50
        mask = np.zeros(len(times), dtype=bool)
        mask[start:stop] = True
        post_union |= mask
        posts[str(control)] = {"control_rows": [control, control + 9],
                               "native_indices": [start, stop], **interval(mask)}
    return {
        "schema": "flylab.m3.e189.fixed-force-windows.v1",
        "main_native_window_s": [0.5, 3.0],
        "force_projection": "leg_heading_xy_force_bw",
        "main": interval(main),
        "e186_control_owner_rows_100_599": interval(owner_bins),
        "e186_post10_union": interval(post_union),
        "whole_3s": interval(np.ones(len(times), dtype=bool)),
        "startup_99_controls": interval(np.arange(len(times)) < 99 * 50),
        "e186_post10_control_owner_mapping": posts,
        "control_rows_100_599_native_time_note":
            "E186 control-row owner bins span native .4951 through 2.995 s; main uses literal [.5,3).",
    }


def selected_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    floats = ("pre_rate", "mn_rate", "pre_mask", "mn_current",
              "muscle67_activation", "requested_rm_coxa_roll_rad",
              "actual_rm_coxa_roll_rad")
    result = {"control_tick": np.asarray([row["control_tick"] for row in rows], np.int64)}
    result.update({name: np.asarray([row[name] for row in rows], np.float64)
                   for name in floats})
    result["weight"] = np.full(len(rows), rows[0]["weight"] if rows else np.nan,
                               np.float32)
    return result


def capture_selected(session: Any, control: dict[str, Any], arm: str) -> dict[str, Any]:
    signal = selected_signal(session)
    signal.update(
        weight=ARM_WEIGHTS[arm],
        requested_rm_coxa_roll_rad=float(control["joint_targets_rad"][RM_COXA_ROLL]),
        actual_rm_coxa_roll_rad=float(control["joint_angles_rad"][RM_COXA_ROLL]),
    )
    return signal


def run_arm(graph: Any, neutral: dict[str, Any], output: Path, *, arm: str,
            startup: dict[str, np.ndarray], approved: E189Gate,
            process_began: float = _ENTRY_STARTED) -> dict[str, Any]:
    """Explicit E189 loop: original E181 clocks and guards with one changed edge."""
    from flylab.c.storage import StateStore
    from flylab.c.walking_trace import capture_control, contact_diagnostics
    from tools import observe_c_banc_m3_ground_reaction as e185
    from tools import optimize_c_banc_m3_native_timing as e181
    from tools import probe_c_banc_m3_native_objective as e180
    from tools import verify_c_banc_m3_native_monotone_body as e150
    from tools import verify_c_banc_m3_native_timescale_transfer as protocol
    from tools.analyze_c_banc_biological_gait import analyze, load_model

    require(isinstance(approved, E189Gate) and approved.arm == arm
            and len(approved.receipt_sha256) == 64 and len(approved.readiness) == 2
            and output.parent == safe(f"{OUTPUT_ROOT}/{arm}"),
            "live typed E189 arm gate required")
    require(LIMITS == e181.LIMITS, "startup99 limits changed")
    e181._neutral(neutral)
    require(graph.hash == approved.metadata["expected_graph_hash"]
            and neutral["network"]["model_hash"]
            == approved.metadata["expected_network_identity"],
            "source graph/network identity")
    output.mkdir(exist_ok=False)
    began = time.perf_counter()
    session = restore = journal = stream = None
    rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    native: list[Any] = []
    forces: list[dict[str, Any]] = []
    native_state: list[dict[str, Any]] = []
    inventory: dict[str, Any] = {}
    raw_errors: list[dict[str, str]] = []
    analysis_errors: list[dict[str, str]] = []
    clock = protocol.AdvanceLedger()
    reason, health_stop, primary = "duration", False, None
    actual_neural_steps = 0
    proof = None
    ledger = {
        "schema": "flylab.m3.e189.single-mn-edge.execution-ledger.v1",
        "construction_attempts": [], "clock": {}, "completed_controls": 0,
        "candidate_adopted": False, "native_evaluations": 1,
        "offline_solver_calls": 0, "automatic_retries": 0,
    }

    def persist() -> None:
        ledger["construction_totals"] = protocol._construction_totals(
            ledger["construction_attempts"])
        ledger["process_wall_s"] = time.perf_counter() - process_began
        protocol.write_json(output / "advance-ledger.json", ledger)

    def preserve(label: str, action: Any) -> None:
        try:
            action()
        except BaseException as error:
            raw_errors.append({"artifact": label, "error": repr(error)})

    persist()
    metadata = {
        "schema": "flylab.m3.e189.single-mn-edge.arm.v1", "arm": arm,
        "edge_id": EDGE, "pre_id": PRE, "post_id": POST,
        "source_weight": float(SOURCE), "target_weight": ARM_WEIGHTS[arm],
        "freeze_sha256": approved.freeze_sha256,
        "source_review_sha256": approved.review_sha256,
        "parent_validation_sha256": approved.receipt_sha256,
        "source_adapter_identity": neutral["monotone_recruitment_patch"]["adapter_identity"],
        "source_network_identity": neutral["network"]["model_hash"],
    }
    protocol.write_json(output / "arm-provenance.json", metadata)
    constructed: list[tuple[Any, dict[str, Any]]] = []

    def build() -> tuple[Any, dict[str, Any]]:
        pair = construct_arm(graph, neutral, arm)
        constructed.append(pair)
        return pair

    def claim(pair: tuple[Any, dict[str, Any]]) -> None:
        nonlocal session, proof
        session, proof = pair

    try:
        if time.perf_counter() - process_began >= PARAMETERS["body_wall_s"] - 10:
            raise e181.ResourceGuard("Body process reserve exhausted before construction")
        owner = protocol.profiled_construction(
            "construct_e189_single_edge", build,
            ledger["construction_attempts"], persist,
            on_return=claim,
        )
        require(owner[0] is session and owner[1] is proof,
                "construction owner transfer")
        protocol.write_json(output / "csr-packed-proof.json", proof)
        require(session.adapter.identity == approved.metadata["expected_adapter_identity"]
                and session.body.model_hash == approved.metadata["expected_body_hash"],
                "adapter/body identity changed")
        validate_rm_path(session)
        metadata.update(actual_device=str(session.network.device),
                        graph_neurons=int(graph.n), adapter_identity=session.adapter.identity,
                        body_model_hash=session.body.model_hash,
                        candidate_network_identity=session.network.identity)
        protocol.write_json(output / "arm-provenance.json", metadata)
        patch = session.monotone_recruitment_patch
        np.savez_compressed(output / "timing-parameters.npz",
                            tau=patch["tau"], kappa=patch["kappa"],
                            slopes=patch["slopes"], membership=patch["membership"],
                            motor_ids=patch["motor_ids"],
                            fixed_muscles=patch["fixed_muscles"])
        StateStore.save(output / "start", session.snapshot())
        rows.append(capture_control(session))
        selected.append(capture_selected(session, rows[-1], arm))
        restore = e185.install_native_observer(
            session.body, native, forces, inventory,
            contact_diagnostics, protocol.NativeHealthStop)
        journal = protocol.ControlJournal(output / "advance-journal.jsonl")
        observe = e150.make_observer(native_state, original=lambda *_: {})
        detailed = np.asarray(session.adapter.calibration["motor_ids"][:1], np.int64)
        stream = (output / "trace.jsonl").open("x", encoding="utf-8", buffering=1)
        stream.write(json.dumps(protocol.jsonable(rows[0]), allow_nan=False) + "\n")
        observe(session, detailed, {})
        for tick in range(1, 601):
            if time.perf_counter() - process_began >= PARAMETERS["body_wall_s"] - 10:
                reason = "resource_guard"
                break
            event = protocol.advance_control(session, "main", tick, clock, journal)
            try:
                row = capture_control(session)
                require(len(native) == len(forces) == 50 * tick
                        and session.network.tick == 20 * tick
                        and abs(session.body.physics_time() - 0.005 * tick) <= 1e-8,
                        "incomplete 20-neural/50-physics control")
                rows.append(row)
                selected.append(capture_selected(session, row, arm))
                stream.write(json.dumps(protocol.jsonable(row), allow_nan=False) + "\n")
                observe(session, detailed, {})
                protocol.record_readout(journal, "main", tick, event, complete=True)
            except Exception as error:
                protocol.record_readout(journal, "main", tick, event,
                                        complete=False, error=error)
                raise
            if session.fault or session.body.fault:
                reason, health_stop = "native_health_guard", True
                break
            if tick == 99:
                protocol.e121.save_startup_raw(output, rows, startup)
                screen = protocol.strict_startup(rows, native, startup, LIMITS)
                protocol.write_json(output / "startup-screen.json", screen)
                if not screen["strict_E141_startup_screen"]:
                    reason = "startup_guard"
                    break
            if tick % 100 == 0:
                ledger["completed_controls"] = tick
                ledger["clock"] = {"neural_s": clock.neural_s,
                                   "physical_s": clock.physical_s,
                                   "valid": clock.clock_valid}
                persist()
    except protocol.NativeHealthStop as error:
        reason, health_stop, primary = "native_health_guard", True, error
    except e181.ResourceGuard as error:
        reason, primary = "resource_guard", error
    except Exception as error:
        reason, primary = "execution_error", error
    finally:
        if session is None and constructed:
            preserve("construction_owner_close", constructed[0][0].close)
        if journal is not None:
            preserve("journal_close", journal.close)
        if stream is not None:
            preserve("trace_stream_close", stream.close)
        if session is not None:
            try:
                actual_neural_steps = int(session.network.tick)
            except BaseException as error:
                raw_errors.append({"artifact": "neural_counter_before_close", "error": repr(error)})
            preserve("final_checkpoint", lambda: StateStore.save(output / "final", session.snapshot()))
        preserve("trace", lambda: protocol.write_json(output / "trace.json", {
            "schema": "flylab.m3.e189.single-mn-edge.trace.v1",
            "controller": "E150 fixed monotone recruitment with one relay-to-MN edge",
            "graph_hash": graph.hash,
            "model_hash": session.network.identity if session else neutral["network"]["model_hash"],
            "body_hash": session.body.model_hash if session else None,
            "adapter_hash": session.adapter.identity if session else None,
            "parameters": neutral["parameters"], "samples": rows,
        }))
        preserve("native_contact", lambda: np.savez_compressed(
            output / "native-contact.npz", **protocol.native_arrays(native)))
        preserve("ground_reaction", lambda: np.savez_compressed(
            output / "ground-reaction.npz", **e185.force_arrays(forces)))
        preserve("ground_reaction_metadata", lambda: protocol.write_json(
            output / "ground-reaction-metadata.json", {
                "schema": "flylab.m3.e189.ground-reaction-metadata.v1",
                "inventory": inventory, "force_samples": len(forces),
                "native_samples": len(native)}))
        preserve("native_recruitment", lambda: e150.save_native_state(
            output / "native-recruitment-state.npz", native_state))
        preserve("selected_signal", lambda: np.savez_compressed(
            output / "selected-signal.npz", **selected_arrays(selected)))
        preserve("clock_events", lambda: protocol.write_json(output / "clock-events.json", clock.events))
        if restore is not None:
            preserve("observer_restore", restore)
        if session is not None:
            preserve("session_close", session.close)
        if primary is not None:
            preserve("primary_error", lambda: protocol.write_json(
                output / "primary-error.json",
                {"type": type(primary).__name__, "message": str(primary)}))
        ledger["completed_controls"] = max(0, len(rows) - 1)
        ledger["clock"] = {"neural_s": clock.neural_s, "physical_s": clock.physical_s,
                           "valid": clock.clock_valid, "actual_costs_by_owner": clock.by_owner()}
        ledger["stop_reason"] = reason
        preserve("ledger_final", persist)

    screen_path = output / "startup-screen.json"
    startup_ok = (read_json(screen_path)["strict_E141_startup_screen"]
                  if screen_path.exists() else None)
    if (reason == "duration" and not raw_errors
            and time.perf_counter() - process_began < PARAMETERS["body_wall_s"] - 3):
        try:
            model = load_model(output / "trace.json")
            quality, _phi, _events = analyze(model)
            protocol.write_json(output / "biological-gait-3s.json", quality)
            native_arrays = protocol.native_arrays(native)
            physical = e180.physical_summary(rows, native_arrays, quality)
            protocol.write_json(output / "physical-3s.json", physical)
            fixed = fixed_force_summary(e185.force_arrays(forces), native_arrays)
            protocol.write_json(output / "fixed-force-windows.json", fixed)
        except Exception as error:
            analysis_errors.append({"artifact": "fixed_3s_analysis", "error": repr(error)})
    elif reason == "duration":
        reason = "resource_guard"
    outcome = e181.summarize_arm(
        reason, health_stop, len(rows) - 1, len(native), clock.neural_s,
        clock.physical_s, clock.clock_valid, startup_ok, raw_errors, analysis_errors)
    outcome.update(schema="flylab.m3.e189.single-mn-edge.result.v1", arm=arm,
                   source_weight=float(SOURCE), target_weight=ARM_WEIGHTS[arm],
                   changed_csr_words=proof["changed_words"] if proof else None,
                   freeze_sha256=approved.freeze_sha256,
                   source_review_sha256=approved.review_sha256,
                   parent_validation_sha256=approved.receipt_sha256,
                   body_wall_s=time.perf_counter() - began,
                   process_wall_s=time.perf_counter() - process_began,
                   advance_journal_rows=journal.rows if journal else 0,
                   primary_error=repr(primary) if primary else None,
                   actual_neural_steps=actual_neural_steps,
                   actual_physics_steps=len(native),
                   raw_force_samples=len(forces), selected_signal_rows=len(selected))
    body_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    outcome["raw_bytes"] = body_bytes
    if body_bytes > PARAMETERS["maximum_body_bytes"]:
        outcome["technical_failure"] = True
        outcome["validity"] = "incomplete"
        outcome["raw_persistence_errors"].append(
            {"artifact": "body_size", "error": "1 GiB cap exceeded"})
    if time.perf_counter() - process_began > PARAMETERS["body_wall_s"]:
        outcome.update(validity="incomplete", technical_failure=True,
                       stop_reason="resource_guard", resource_limit_exceeded=True)
    try:
        protocol.write_json(output / "result.json", outcome)
    except BaseException as error:
        outcome["raw_persistence_errors"].append(
            {"artifact": "result", "error": repr(error)})
        outcome.update(technical_failure=True, validity="incomplete")
        if primary is not None:
            primary.add_note("Secondary E189 result persistence: " + repr(error))
            raise primary.with_traceback(primary.__traceback__) from error
        raise
    return outcome


def execute(freeze_path: Path, review_path: Path, arm: str, output: Path,
            parent_validation_sha256: str) -> dict[str, Any]:
    """Run one reviewed original arm and preserve even a partial failure."""
    require(type(parent_validation_sha256) is str
            and len(parent_validation_sha256) == 64,
            "child requires live parent validation SHA")
    approved = gate(freeze_path, review_path, arm, output,
                    parent_validation_sha256)
    require(shutil.disk_usage(output.parent).free >= PARAMETERS["minimum_free_bytes"],
            "20 GiB free disk required")
    require(time.perf_counter() - _ENTRY_STARTED < PARAMETERS["child_internal_wall_s"],
            "child budget exhausted before entry")
    output.mkdir(exist_ok=False)
    terminal: dict[str, Any] = {
        "schema": "flylab.m3.e189.single-mn-edge.terminal.v1",
        "status": "failed", "arm": arm, "original_runs": 0,
        "automatic_retries": 0, "candidate_adopted": False, "F6_exposed": 0,
        "freeze_sha256": approved.freeze_sha256,
        "source_review_sha256": approved.review_sha256,
        "parent_validation_sha256": approved.receipt_sha256,
    }
    write_new(output / "readiness.json", {
        "schema": "flylab.m3.e189.single-mn-edge.readiness.v1",
        "observations": approved.readiness, "native_permitted": True,
        "source": "same live parent validation receipt",
        "parent_validation_sha256": approved.receipt_sha256,
        "accepted_monotonic_s": time.perf_counter(),
    })
    monitor = None
    stop = threading.Event()
    monitor_errors: list[str] = []
    threads = None
    result = None
    primary: BaseException | None = None
    cleanup_errors: list[dict[str, str]] = []
    monitor_check: dict[str, Any] = {}
    try:
        from tools import optimize_c_banc_m3_native_timing as e181
        from tools import recover_c_banc_m3_native_timing as e182

        monitor = threading.Thread(
            target=resource_readiness.monitor,
            args=(output / "resource-monitor.jsonl", stop, monitor_errors, e182),
            daemon=True)
        monitor.start()
        threads, _ = e181.protocol.joint_fit.one_cpu_thread()
        metadata = approved.metadata
        startup_raw = e181.protocol.load_npz(safe(metadata["startup_path"]))
        startup = {key: startup_raw[key]
                   for key in ("target_command", "target_tendon")}
        graph = e181.GraphStore.load(safe(metadata["graph_path"]))
        require(graph.hash == metadata["expected_graph_hash"],
                "full graph identity changed")
        neutral = e181.StateStore.load(safe(metadata["neutral_checkpoint"]))
        require(neutral["network"]["model_hash"] == metadata["expected_network_identity"],
                "zero network identity changed")
        terminal["original_runs"] = 1
        result = run_arm(graph, neutral, output / "body", arm=arm,
                         startup=startup, approved=approved,
                         process_began=_ENTRY_STARTED)
    except BaseException as error:
        primary = error
    finally:
        if threads is not None:
            try:
                threads.restore_original_limits()
            except BaseException as error:
                cleanup_errors.append({"artifact": "cpu_thread_restore", "error": repr(error)})
        stop.set()
        if monitor is not None:
            try:
                monitor.join(timeout=5)
            except BaseException as error:
                cleanup_errors.append({"artifact": "monitor_join", "error": repr(error)})
            if monitor.is_alive():
                monitor_errors.append("monitor_thread_did_not_stop")
        resource_rows = []
        if monitor is not None:
            try:
                from tools import recover_c_banc_m3_native_timing as e182
                resource_rows = e182._resource_rows(output / "resource-monitor.jsonl")
            except BaseException as error:
                monitor_errors.append(repr(error))
        bad = [i for i, row in enumerate(resource_rows)
               if not resource_readiness.resource_clear(row)
               or not readiness_clear(row, monitor=True)]
        monitor_check = {
            "schema": "flylab.m3.e189.single-mn-edge.resource-monitor.v1",
            "samples": len(resource_rows), "bad_sample_indices": bad,
            "modes": [row.get("resource_mode") for row in resource_rows],
            "errors": monitor_errors,
            "thread_alive": bool(monitor and monitor.is_alive()),
            "resource_clear": bool(resource_rows and not bad and not monitor_errors
                                   and monitor is not None and not monitor.is_alive()),
        }
        try:
            write_new(output / "resource-monitor-check.json", monitor_check)
        except BaseException as error:
            cleanup_errors.append({"artifact": "resource_monitor_check", "error": repr(error)})
        terminal["resource_monitor"] = monitor_check
        terminal["cleanup_errors"] = cleanup_errors
        terminal["body_result"] = result
        terminal["child_wall_s"] = time.perf_counter() - _ENTRY_STARTED
        terminal["body_bytes"] = sum(
            p.stat().st_size for p in (output / "body").rglob("*") if p.is_file())
        terminal["output_bytes"] = sum(
            p.stat().st_size for p in output.rglob("*") if p.is_file())
        if primary is not None:
            terminal["primary_error"] = {
                "type": type(primary).__name__, "message": str(primary)}
        successful = bool(
            primary is None and result is not None
            and result.get("validity") == "valid_3s_diagnostic"
            and monitor_check["resource_clear"] and not cleanup_errors
            and terminal["child_wall_s"] <= PARAMETERS["child_internal_wall_s"]
            and terminal["body_bytes"] <= PARAMETERS["maximum_body_bytes"]
            and terminal["output_bytes"] <= PARAMETERS["maximum_output_bytes"]
            and shutil.disk_usage(output).free >= PARAMETERS["minimum_free_bytes"])
        terminal["status"] = "complete" if successful else "failed"
        try:
            write_new(output / "terminal.json", terminal)
        except BaseException as error:
            if primary is None:
                raise
            print("E189 terminal persistence failed: " + repr(error), file=sys.stderr)
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)
    require(terminal["status"] == "complete", "E189 arm incomplete; see terminal")
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARM_WEIGHTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-validation-sha256", required=True)
    args = parser.parse_args()
    execute(args.freeze, args.review, args.arm, args.output,
            args.parent_validation_sha256)


if __name__ == "__main__":
    main()
