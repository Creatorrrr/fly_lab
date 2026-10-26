"""E183 one-pass continuation over LH, RF, RM, RH.

E181 authentic LM validation certifies immutable scientific inputs. A separate
E183 gate authorizes the remaining axes. Import and --help never load originals.
"""

# Technical failures must survive in create-only terminal receipts.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil

_ENTRY_STARTED = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import optimize_c_banc_m3_native_timing as e181
from tools import recover_c_banc_m3_native_timing as e182

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "verification/banc-natural-gait-m3-20260919/e183-remaining-native-timing"
DESIGN = "docs/BANC_M3_E183_REMAINING_TIMING_DESIGN_20260924.md"
OWNED = (
    "tools/continue_c_banc_m3_native_timing.py",
    "tests/test_c_banc_native_timing_continuation.py",
)
AXES = ("lh", "rf", "rm", "rh")
SIGNS = ("plus", "minus")
SCHEMA = "flylab.m3.e183.remaining-timing-freeze.v1"
SOURCE_REVIEW_SCHEMA = "flylab.m3.e183.parent-source-review.v1"
E182_REVIEW_SCHEMA = "flylab.m3.e182.parent-final-review.v1"
AXIS_REVIEW_SCHEMA = "flylab.m3.e183.parent-axis-review.v1"
INHERITED_S = 3911.404136399855
LIMIT_S = 4994.0
RESERVE_S = 1020.0
AXIS_S = 270.0
SUPERVISOR_S = 110.0
CAMPAIGN_BYTES = 4 << 30


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object required")
    return value


def _sealed(path: Path, digest: str) -> dict[str, Any]:
    if not isinstance(digest, str) or len(digest) != 64 or e181._sha256(path) != digest:
        raise ValueError(f"Exact reviewed bytes required: {path}")
    return _read(path)


def _rel(relative: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Safe repository-relative path required")
    return e181._safe(relative)


def _raw_inventory(directory: Path, mapping: Any) -> None:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Complete raw inventory required")
    if directory.is_symlink() or any(
        path.is_symlink() for path in directory.rglob("*")
    ):
        raise ValueError("Symlink in reviewed raw")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if set(mapping) != actual:
        raise ValueError("Reviewed raw file inventory changed")
    for name, digest in mapping.items():
        if (
            not isinstance(name, str)
            or name.startswith("/")
            or "\\" in name
            or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or not (directory / name).resolve().is_relative_to(directory.resolve())
            or not isinstance(digest, str)
            or len(digest) != 64
            or e181._sha256(directory / name) != digest
        ):
            raise ValueError("Reviewed raw path or hash changed: " + str(name))


def _gate(args: argparse.Namespace) -> tuple[dict[str, Any], e181.ValidatedGate]:
    freeze = _sealed(args.freeze, args.freeze_sha256)
    required = {
        "schema": SCHEMA,
        "design_path": DESIGN,
        "owned_files": list(OWNED),
        "output_directory": OUTPUT,
        "axes": list(AXES),
        "backtrack_steps": list(e181.STEPS),
        "inherited_cumulative_s": INHERITED_S,
        "cumulative_limit_s": LIMIT_S,
        "reserve_max_s": RESERVE_S,
        "native_children_max": 8,
        "child_internal_s": 100,
        "child_external_s": 105,
        "supervisor_s": 110,
        "axis_wall_s": 270,
        "axis_external_s": 285,
        "minimum_free_disk_bytes": 20 << 30,
        "arm_raw_bytes_max": 1 << 30,
        "campaign_raw_bytes_max": CAMPAIGN_BYTES,
        "cpu_workers": 1,
        "gpu_workers": 1,
        "automatic_retries": 0,
        "candidate_adopted": False,
        "goal_status": "active",
    }
    if any(freeze.get(key) != value for key, value in required.items()):
        raise ValueError("E183 frozen continuation contract changed")
    if e181._sha256(_rel(DESIGN)) != freeze.get("design_sha256"):
        raise ValueError("E183 design changed")
    metadata = _sealed(
        _rel(freeze.get("parent_metadata_path")), freeze.get("parent_metadata_sha256")
    )
    if (
        metadata.get("design_sha256") != freeze["design_sha256"]
        or metadata.get("cumulative_before_s") != INHERITED_S
        or metadata.get("original_values_loaded") is not False
        or metadata.get("new_neural_steps") != 0
        or metadata.get("new_physics_steps") != 0
    ):
        raise ValueError("E183 preserved parent metadata changed")
    old = freeze.get("e181_gate")
    if not isinstance(old, dict) or set(old) != {
        "freeze_path",
        "freeze_sha256",
        "source_review_path",
        "source_review_sha256",
        "baseline_review_path",
        "baseline_review_sha256",
        "previous_lf_axis_sha256",
    }:
        raise ValueError("Complete authentic E181 gate identity required")
    gate = e181.verified_inputs(
        _rel(old["freeze_path"]),
        old["freeze_sha256"],
        _rel(old["source_review_path"]),
        old["source_review_sha256"],
        "lm",
        _rel(old["baseline_review_path"]),
        old["baseline_review_sha256"],
        old["previous_lf_axis_sha256"],
    )
    final = freeze.get("e182_final")
    names = (
        "freeze",
        "source_review",
        "baseline_result",
        "baseline_driver",
        "plus_result",
        "plus_driver",
        "minus_result",
        "minus_driver",
        "final_review",
        "baseline_review",
        "baseline_raw_review",
    )
    if not isinstance(final, dict) or set(final) != {
        suffix for name in names for suffix in (f"{name}_path", f"{name}_sha256")
    }:
        raise ValueError("Complete E182 final identity required")
    sealed = {
        name: _sealed(_rel(final[f"{name}_path"]), final[f"{name}_sha256"])
        for name in names
    }
    e182._source_gate(
        argparse.Namespace(
            freeze=_rel(final["freeze_path"]),
            freeze_sha256=final["freeze_sha256"],
            source_review=_rel(final["source_review_path"]),
            source_review_sha256=final["source_review_sha256"],
        )
    )
    base = sealed["baseline_result"]
    if (
        base.get("theta") != [0.0] * 6
        or base.get("validity") != "valid_3s_diagnostic"
        or sealed["baseline_driver"].get("error") is not None
        or any(sealed[f"{sign}_driver"].get("error") is not None for sign in SIGNS)
        or any(
            sealed[f"{sign}_result"].get("validity") != "valid_3s_diagnostic"
            for sign in SIGNS
        )
        or any(
            e181.screen_candidate(base, sealed[f"{sign}_result"])["eligible"]
            for sign in SIGNS
        )
        or any(
            sealed[f"{sign}_result"].get("theta") != e182.THETA[sign] for sign in SIGNS
        )
        or any(
            sealed[f"{sign}_driver"].get("result_sha256")
            != final[f"{sign}_result_sha256"]
            for sign in SIGNS
        )
    ):
        raise ValueError("E182 final theta0 and both rejected arms required")
    review = sealed["final_review"]
    baseline_review = sealed["baseline_review"]
    baseline_raw_review = sealed["baseline_raw_review"]
    if (
        review.get("schema") != E182_REVIEW_SCHEMA
        or review.get("freeze_sha256") != final["freeze_sha256"]
        or review.get("source_review_sha256") != final["source_review_sha256"]
        or review.get("selected_theta") != [0.0] * 6
        or review.get("selection_checked_independently") is not True
        or review.get("validity") != "valid_fixed_LM_pair"
        or review.get("hypothesis_verdict")
        != "both_LM_proposals_rejected_by_slip_constraint"
        or review.get("cost", {}).get("cumulative_after_s") != INHERITED_S
        or review.get("preservation", {}).get("accepted") is not True
        or review.get("source_sha256") != sealed["source_review"].get("source_sha256")
        or any(
            review.get("pair", {}).get(sign, {}).get("eligible") is not False
            for sign in SIGNS
        )
        or baseline_review.get("schema") != e182.BASELINE_REVIEW_SCHEMA
        or baseline_review.get("freeze_sha256") != final["freeze_sha256"]
        or baseline_review.get("baseline_result_sha256")
        != final["baseline_result_sha256"]
        or baseline_review.get("baseline_driver_sha256")
        != final["baseline_driver_sha256"]
        or baseline_review.get("accepts_baseline") is not True
        or baseline_raw_review.get("arm") != "baseline"
        or baseline_raw_review.get("accepted_recorded_data") is not True
        or baseline_raw_review.get("blocking_defects") != []
        or review.get("evidence_sha256", {}).get(
            Path(final["baseline_raw_review_path"]).name
        )
        != final["baseline_raw_review_sha256"]
    ):
        raise ValueError("Parent E182 final rejection review required")
    baseline_dir = _rel(e182.OUTPUT) / "baseline"
    if _rel(final["baseline_result_path"]) != baseline_dir / "result.json":
        raise ValueError("E182 baseline path changed")
    _raw_inventory(baseline_dir, baseline_review.get("baseline_artifact_sha256"))
    full_baseline = baseline_raw_review.get("arm_artifact_sha256")
    if (
        not isinstance(full_baseline, dict)
        or {
            f"{e182.OUTPUT}/baseline/{name}": digest
            for name, digest in baseline_review["baseline_artifact_sha256"].items()
        }
        != full_baseline
    ):
        raise ValueError("Independent E182 baseline raw review differs")
    for sign in SIGNS:
        raw_review_name = f"parent-e182-{sign}-raw-review.json"
        raw_review = _sealed(
            _rel(final["final_review_path"]).parent / raw_review_name,
            review.get("evidence_sha256", {}).get(raw_review_name),
        )
        arm_dir = _rel(e182.OUTPUT) / sign
        mapping = raw_review.get("arm_artifact_sha256")
        if (
            raw_review.get("arm") != sign
            or raw_review.get("accepted_recorded_data") is not True
            or raw_review.get("blocking_defects") != []
            or not isinstance(mapping, dict)
            or mapping.get(f"{e182.OUTPUT}/{sign}/result.json")
            != final[f"{sign}_result_sha256"]
        ):
            raise ValueError("Independent E182 candidate raw review differs")
        _raw_inventory(
            arm_dir,
            {
                Path(name).relative_to(f"{e182.OUTPUT}/{sign}").as_posix(): digest
                for name, digest in mapping.items()
            },
        )
    source = _sealed(args.source_review, args.source_review_sha256)
    if (
        source.get("schema") != SOURCE_REVIEW_SCHEMA
        or source.get("freeze_sha256") != args.freeze_sha256
        or source.get("accepts_continuation") is not True
        or source.get("blocking_defects") != []
        or set(source.get("source_sha256", {})) != set(OWNED)
    ):
        raise ValueError("E183 parent source review required")
    e181._check_hashes(source["source_sha256"])
    return freeze, gate


def _axis_root(axis: str) -> Path:
    return _rel(OUTPUT) / f"axis-{axis}"


def _prior_axis(
    axis: str, freeze_sha: str
) -> tuple[list[float], dict[str, Any], str | None]:
    if axis == AXES[0]:
        baseline = _read(_rel(e182.OUTPUT) / "baseline" / "result.json")
        return [0.0] * 6, baseline, None
    previous = AXES[AXES.index(axis) - 1]
    root = _axis_root(previous)
    result_path = root / "result.json"
    digest = e181._sha256(result_path)
    result = _read(result_path)
    review = _read(root / "parent-axis-review.json")
    if (
        result.get("axis") != previous
        or result.get("technical_failure") is not False
        or result.get("selection") is None
        or review.get("schema") != AXIS_REVIEW_SCHEMA
        or review.get("freeze_sha256") != freeze_sha
        or review.get("axis_result_sha256") != digest
        or review.get("selection_checked") is not True
        or review.get("accepts_next_axis") is not True
        or review.get("blocking_defects") != []
    ):
        raise ValueError("Exact parent-reviewed previous E183 axis required")
    for sign in SIGNS:
        request = root / f"{sign}-request.json"
        mapping = review.get("raw_sha256", {}).get(sign)
        if request.exists():
            _raw_inventory(root / sign, mapping)
        elif mapping is not None:
            raise ValueError("Review maps an unattempted raw arm")
    theta = result["selection"]["selected_theta"]
    selected = result["selected_result"]
    if theta != selected.get("theta") or not isinstance(theta, list):
        raise ValueError("Previous selected center differs from raw")
    selected_sign = result["selection"]["selected_direction"]
    source = root / selected_sign / "result.json" if selected_sign else None
    if (
        source is not None
        and e181._sha256(source) != result["arm_result_sha256"][selected_sign]
    ):
        raise ValueError("Previous selected raw changed")
    return theta, selected, digest


def _account(root: Path) -> dict[str, Any]:
    attempts = {
        f"{axis}/{sign}": e181._attempt_accounting(
            _axis_root(axis) / f"{sign}-request.json"
        )
        for axis in AXES
        for sign in SIGNS
    }
    return {
        "attempts": attempts,
        "native_upper_bound": sum(row["arms_upper_bound"] for row in attempts.values()),
        "native_exact": sum(row["arms_exact"] for row in attempts.values())
        if all(row["arms_exact"] is not None for row in attempts.values())
        else None,
        "native_accounted_s": sum(row["accounted_wall_s"] for row in attempts.values()),
        "native_cost_exact": all(row["cost_exact"] for row in attempts.values()),
    }


def _native_actual(root: Path) -> float:
    total = 0.0
    for axis in AXES:
        for sign in SIGNS:
            path = _axis_root(axis) / f"{sign}-request-supervisor.json"
            if path.is_file():
                receipt = _read(path)
                if receipt.get("started"):
                    value = receipt.get("child_wall_s")
                    if (
                        type(value) not in (int, float)
                        or not math.isfinite(value)
                        or value < 0
                    ):
                        raise ValueError("Invalid native elapsed receipt")
                    total += value
    return total


def _stage_cost(axis: str, began: float) -> dict[str, Any]:
    root = _rel(OUTPUT)
    previous_wall = sum(
        _read(_axis_root(name) / "result.json")["axis_wall_s"]
        for name in AXES[: AXES.index(axis)]
    )
    prior_gate_cost = sum(row["elapsed_s"] for row in _gate_failures(root))
    elapsed = time.perf_counter() - began
    account = _account(root)
    native_actual = _native_actual(root)
    unknown_overlap = 0.0
    for sign in SIGNS:
        request_path = _axis_root(axis) / f"{sign}-request.json"
        if (
            request_path.is_file()
            and not request_path.with_name(
                request_path.stem + "-supervisor.json"
            ).is_file()
        ):
            launched_at = _read(request_path).get("driver_elapsed_s")
            if (
                type(launched_at) not in (int, float)
                or not math.isfinite(launched_at)
                or launched_at < 0
            ):
                raise ValueError("Unknown attempt lacks launch elapsed")
            unknown_overlap += min(SUPERVISOR_S, max(0.0, elapsed - launched_at))
    non_native = (
        prior_gate_cost + previous_wall + elapsed - native_actual - unknown_overlap
    )
    if non_native < -1e-6:
        raise ValueError("Native elapsed exceeds driver elapsed")
    cost = max(0.0, non_native) + account["native_accounted_s"]
    return {
        **account,
        "non_native_elapsed_s": max(0.0, non_native),
        "native_actual_elapsed_s": native_actual,
        "unknown_native_overlap_reserved_s": unknown_overlap,
        "stage_accounted_s": cost,
        "cumulative_with_inherited_s": INHERITED_S + cost,
    }


def _capacity(axis: str, began: float, *, next_native: bool) -> dict[str, Any]:
    root = _rel(OUTPUT)
    if _gate_failures(root):
        raise e181.ResourceGuard("Earlier E183 driver gate failure blocks native")
    if psutil.disk_usage(str(root.parent)).free < 20 << 30:
        raise e181.ResourceGuard("20 GiB free disk required")
    if e181._directory_bytes(root) > CAMPAIGN_BYTES:
        raise e181.ResourceGuard("E183 campaign bytes exceeded")
    cost = _stage_cost(axis, began)
    reserve = SUPERVISOR_S if next_native else 0
    if (
        cost["native_upper_bound"] + int(next_native) > 8
        or cost["stage_accounted_s"] + reserve > RESERVE_S
        or cost["cumulative_with_inherited_s"] + reserve > LIMIT_S
        or time.perf_counter() - began + reserve > AXIS_S
        or time.perf_counter() - _ENTRY_STARTED + reserve > 285
    ):
        raise e181.ResourceGuard("E183 next 110s cannot fit hard budget")
    return cost


def _gate_failures(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    receipts = []
    for path in sorted(root.glob("axis-*-gate-failure.json")):
        row = _read(path)
        elapsed = row.get("elapsed_s")
        if (
            row.get("schema") != "flylab.m3.e183.driver-gate-failure.v1"
            or row.get("axis") not in AXES
            or type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
            or elapsed < 0
            or row.get("native_started") is not False
        ):
            raise ValueError("Invalid E183 driver gate failure receipt")
        receipts.append(row)
    return receipts


def _record_gate_failure(args: argparse.Namespace, error: Exception) -> None:
    root = _rel(OUTPUT)
    if root.is_symlink():
        raise ValueError("E183 output root may not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    e181._write_new(
        root / f"axis-{args.axis}-gate-failure.json",
        {
            "schema": "flylab.m3.e183.driver-gate-failure.v1",
            "axis": args.axis,
            "freeze_path": str(args.freeze),
            "freeze_sha256": args.freeze_sha256,
            "source_review_path": str(args.source_review),
            "source_review_sha256": args.source_review_sha256,
            "elapsed_s": time.perf_counter() - _ENTRY_STARTED,
            "native_started": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        },
    )


def _make_request(
    args: argparse.Namespace,
    gate: e181.ValidatedGate,
    sign: str,
    center: list[float],
    proposal: dict[str, Any],
    intent_sha: str,
    readiness_sha: str,
    driver_elapsed_s: float,
) -> dict[str, Any]:
    if sign not in SIGNS or proposal.get("selected_theta") is None:
        raise ValueError("Only a saved-startup selected direction may launch")
    step = proposal["attempts"][-1]["step"]
    if tuple(proposal["selected_theta"]) != e181.propose_axis(
        center, args.axis, sign, step
    ):
        raise ValueError("Saved startup theta differs from coordinate proposal")
    return {
        "schema": "flylab.m3.e183.child-request.v1",
        "axis": args.axis,
        "sign": sign,
        "parent_pid": os.getpid(),
        "parent_created_s": psutil.Process(os.getpid()).create_time(),
        "theta": proposal["selected_theta"],
        "center_theta": center,
        "step": step,
        "freeze_sha256": args.freeze_sha256,
        "source_review_sha256": args.source_review_sha256,
        "e181_freeze_sha256": gate.freeze_sha256,
        "intent_sha256": intent_sha,
        "readiness_sha256": readiness_sha,
        "driver_elapsed_s": driver_elapsed_s,
    }


def _classify_native_result(raw: dict[str, Any]) -> None:
    """Accept measured rejections while stopping technical or unknown outcomes."""
    if raw.get("technical_failure") is not False:
        raise ValueError("E183 child technical failure or unknown technical status")
    validity = raw.get("validity")
    if validity == "valid_3s_diagnostic":
        objective = raw.get("objective_mm_s")
        if (
            raw.get("startup_pass") is not True
            or raw.get("measured_health_stop") is not False
            or type(objective) not in (int, float)
            or not math.isfinite(objective)
            or objective < 0
        ):
            raise ValueError("Inconsistent E183 completed native result")
    elif validity == "measured_native_health_stop":
        if (
            raw.get("measured_health_stop") is not True
            or raw.get("objective_mm_s") is not None
        ):
            raise ValueError("Inconsistent E183 measured health stop")
    elif validity == "incomplete":
        if (
            raw.get("startup_pass") is not False
            or raw.get("measured_health_stop") is not False
            or raw.get("objective_mm_s") is not None
        ):
            raise ValueError("Unknown E183 incomplete native result")
    else:
        raise ValueError("Unknown E183 native validity")


def _child(args: argparse.Namespace, gate: e181.ValidatedGate) -> dict[str, Any]:
    axis = args.axis
    root = _axis_root(axis)
    request_path = root / f"{args.child_sign}-request.json"
    request = _sealed(request_path, args.child_request_sha256)
    intent_path = root / "intent.json"
    intent = _sealed(intent_path, request.get("intent_sha256"))
    preflight_path = root / f"{args.child_sign}-readiness.json"
    readiness = _sealed(preflight_path, request.get("readiness_sha256"))
    proposal = intent.get("preflight", {}).get(args.child_sign, {})
    attempts = proposal.get("attempts", [])
    selected = proposal.get("selected_theta")
    if (
        args.child_request.resolve() != request_path.resolve()
        or request.get("schema") != "flylab.m3.e183.child-request.v1"
        or request.get("axis") != axis
        or request.get("sign") != args.child_sign
        or request.get("freeze_sha256") != args.freeze_sha256
        or request.get("source_review_sha256") != args.source_review_sha256
        or request.get("e181_freeze_sha256") != gate.freeze_sha256
        or intent.get("schema") != "flylab.m3.e183.axis-intent.v1"
        or intent.get("axis") != axis
        or intent.get("freeze_sha256") != args.freeze_sha256
        or request.get("center_theta") != intent.get("center_theta")
        or request.get("theta") != selected
        or not attempts
        or attempts[-1].get("status") != "feasible"
        or request.get("step") != attempts[-1].get("step")
        or tuple(request.get("theta", []))
        != e181.propose_axis(
            request.get("center_theta"), axis, args.child_sign, request.get("step")
        )
        or readiness.get("schema") != "flylab.m3.e183.readiness.v1"
        or readiness.get("axis") != axis
        or readiness.get("sign") != args.child_sign
        or readiness.get("native_permitted") is not True
        or len(readiness.get("observations", [])) != 2
        or not all(row.get("ready") is True for row in readiness["observations"])
        or not e181.driver_ancestor(request)
    ):
        raise ValueError("Exact permitted live E183 child request required")
    center, _, previous_sha = _prior_axis(axis, args.freeze_sha256)
    if (
        center != intent["center_theta"]
        or previous_sha != intent["previous_axis_sha256"]
    ):
        raise ValueError("E183 center/order changed before original load")
    output = root / args.child_sign
    if output.exists():
        raise FileExistsError("E183 arm raw is create-only")
    if e181._directory_bytes(_rel(OUTPUT)) > CAMPAIGN_BYTES:
        raise e181.ResourceGuard("E183 campaign bytes exceeded")
    threads, _ = e181.protocol.joint_fit.one_cpu_thread()
    try:
        startup = e181._load_startup(gate.freeze)
        graph = e181.GraphStore.load(_rel(gate.freeze["graph_path"]))
        if graph.hash != gate.freeze["graph_hash"]:
            raise ValueError("E181 graph identity changed")
        neutral = e181.StateStore.load(_rel(gate.freeze["neutral_checkpoint"]))
        result = e181.run_arm(
            graph,
            neutral,
            tuple(request["theta"]),
            output,
            startup=startup,
            limits=e181.LIMITS,
            wall_limit_s=100,
            gate=gate,
            process_began=_ENTRY_STARTED,
        )
    finally:
        threads.restore_original_limits()
    e181._write_new(
        output / "e183-provenance.json",
        {
            "schema": "flylab.m3.e183.arm-provenance.v1",
            "axis": axis,
            "sign": args.child_sign,
            "theta": request["theta"],
            "center_theta": request["center_theta"],
            "e183_freeze_sha256": args.freeze_sha256,
            "e183_source_review_sha256": args.source_review_sha256,
            "e181_science_mode": gate.mode,
            "e181_freeze_sha256": gate.freeze_sha256,
            "e181_source_review_sha256": gate.source_review_sha256,
            "intent_sha256": request["intent_sha256"],
            "request_sha256": args.child_request_sha256,
            "e181_result_sha256": e181._sha256(output / "result.json"),
        },
    )
    if result.get("technical_failure"):
        raise RuntimeError("E183 native arm technical failure; raw retained")
    return result


def _driver(args: argparse.Namespace, gate: e181.ValidatedGate) -> dict[str, Any]:
    axis = args.axis
    root = _rel(OUTPUT)
    if axis == AXES[0]:
        root.mkdir(parents=True, exist_ok=False)
    out = _axis_root(axis)
    out.mkdir(exist_ok=False)
    began = _ENTRY_STARTED
    preflight: dict[str, Any] = {}
    directions: dict[str, Any] = {}
    readiness: dict[str, Any] = {}
    monitoring_errors: dict[str, list[str]] = {}
    center: list[float] | None = None
    center_result: dict[str, Any] | None = None
    previous_sha: str | None = None
    selection = None
    error: dict[str, Any] | None = None
    try:
        center, center_result, previous_sha = _prior_axis(axis, args.freeze_sha256)
        _capacity(axis, began, next_native=False)
        threads, _ = e181.protocol.joint_fit.one_cpu_thread()
        try:
            callback = e181._saved_startup_callback(gate)
            seen = {tuple(center)}
            for sign in SIGNS:
                try:
                    preflight[sign] = e181.preflight_direction(
                        center, axis, sign, callback, seen
                    )
                except e181.PreflightError as exc:
                    preflight[sign] = {
                        "axis": axis,
                        "direction": sign,
                        "attempts": exc.attempts,
                        "selected_theta": None,
                    }
                    raise
        finally:
            threads.restore_original_limits()
        intent = {
            "schema": "flylab.m3.e183.axis-intent.v1",
            "axis": axis,
            "center_theta": center,
            "center_objective_mm_s": center_result["objective_mm_s"],
            "baseline_result_sha256": e181._sha256(
                _rel(e182.OUTPUT) / "baseline" / "result.json"
            ),
            "previous_axis_sha256": previous_sha,
            "freeze_sha256": args.freeze_sha256,
            "source_review_sha256": args.source_review_sha256,
            "preflight": preflight,
            "native_arms_planned": sum(
                preflight[sign]["selected_theta"] is not None for sign in SIGNS
            ),
            "startup_only_attempts": sum(
                len(preflight[sign]["attempts"]) for sign in SIGNS
            ),
            "candidate_adopted": False,
        }
        e181._write_new(out / "intent.json", intent)
        for sign in SIGNS:
            proposal = preflight[sign]
            if proposal["selected_theta"] is None:
                continue
            _capacity(axis, began, next_native=True)
            readiness[sign] = e182._readiness()
            permitted = all(row["ready"] for row in readiness[sign])
            e181._write_new(
                out / f"{sign}-readiness.json",
                {
                    "schema": "flylab.m3.e183.readiness.v1",
                    "axis": axis,
                    "sign": sign,
                    "observations": readiness[sign],
                    "native_permitted": permitted,
                },
            )
            if not permitted:
                raise e181.ResourceGuard("Resource readiness blocked or unknown")
            _capacity(axis, began, next_native=True)
            request_path = out / f"{sign}-request.json"
            request = _make_request(
                args,
                gate,
                sign,
                center,
                proposal,
                e181._sha256(out / "intent.json"),
                e181._sha256(out / f"{sign}-readiness.json"),
                time.perf_counter() - began,
            )
            e181._write_new(request_path, request)
            command = [
                sys.executable,
                "-X",
                "utf8",
                str(Path(__file__).resolve()),
                "--axis",
                axis,
                "--freeze",
                str(args.freeze),
                "--freeze-sha256",
                args.freeze_sha256,
                "--source-review",
                str(args.source_review),
                "--source-review-sha256",
                args.source_review_sha256,
                "--child-request",
                str(request_path),
                "--child-request-sha256",
                e181._sha256(request_path),
                "--child-sign",
                sign,
            ]
            stop = threading.Event()
            errors: list[str] = []
            monitoring_errors[sign] = errors
            monitor = threading.Thread(
                target=e182._monitor,
                args=(out / f"{sign}-resources.jsonl", stop, errors),
                daemon=True,
            )
            monitor.start()
            try:
                supervisor = e181.supervise_command(
                    command, request_path, timeout_s=105
                )
            finally:
                stop.set()
                monitor.join(timeout=5)
                if monitor.is_alive():
                    errors.append("monitor_thread_did_not_stop")
            if supervisor["status"] != "completed" or supervisor["child_wall_s"] > 110:
                raise e181.ResourceGuard(
                    "E183 child incomplete; raw and supervisor retained"
                )
            raw_path = out / sign / "result.json"
            raw = _read(raw_path)
            if (
                raw.get("theta") != proposal["selected_theta"]
                or raw.get("freeze_sha256") != gate.freeze_sha256
                or raw.get("source_review_sha256") != gate.source_review_sha256
            ):
                raise ValueError("E183 child result/identity invalid")
            _classify_native_result(raw)
            directions[sign] = {
                "theta": proposal["selected_theta"],
                "result": raw,
                "result_sha256": e181._sha256(raw_path),
            }
        baseline = _read(_rel(e182.OUTPUT) / "baseline" / "result.json")
        selection = e181.select_axis(center, center_result, directions, baseline)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    selected_result = (
        directions[selection["selected_direction"]]["result"]
        if selection and selection["selected_direction"]
        else center_result
    )
    try:
        cost = _stage_cost(axis, began)
        bytes_used = e181._directory_bytes(root)
        if bytes_used > CAMPAIGN_BYTES or any(
            e181._directory_bytes(out / sign) > 1 << 30 for sign in SIGNS
        ):
            raise e181.ResourceGuard("E183 raw byte limit exceeded")
        if (
            time.perf_counter() - began > AXIS_S
            or time.perf_counter() - _ENTRY_STARTED > 285
        ):
            raise e181.ResourceGuard("E183 axis wall limit exceeded")
        if (
            cost["stage_accounted_s"] > RESERVE_S
            or cost["cumulative_with_inherited_s"] > LIMIT_S
        ):
            raise e181.ResourceGuard("E183 cumulative limit exceeded")
    except Exception as exc:
        cost = {"accounting_error": repr(exc)}
        if error is None:
            error = {"type": type(exc).__name__, "message": str(exc)}
    receipt = {
        "schema": "flylab.m3.e183.axis-result.v1",
        "axis": axis,
        "e183_freeze_sha256": args.freeze_sha256,
        "e183_source_review_sha256": args.source_review_sha256,
        "e181_science_mode": gate.mode,
        "e181_freeze_sha256": gate.freeze_sha256,
        "e181_source_review_sha256": gate.source_review_sha256,
        "previous_axis_sha256": previous_sha,
        "intent_sha256": e181._sha256(out / "intent.json")
        if (out / "intent.json").is_file()
        else None,
        "center_theta": center,
        "preflight": preflight,
        "readiness": readiness,
        "arm_result_sha256": {
            name: value["result_sha256"] for name, value in directions.items()
        },
        "selection": selection,
        "selected_result": selected_result,
        "accounting": cost,
        "axis_wall_s": time.perf_counter() - began,
        "monitoring_errors": monitoring_errors,
        "technical_failure": error is not None,
        "error": error,
        "candidate_adopted": False,
        "parent_review_required": True,
        "development_entry_proven": False,
        "F3_final": False,
        "F4_final": False,
        "F5_final": False,
    }
    e181._write_new(out / "result.json", receipt)
    if error:
        raise RuntimeError(
            "E183 axis stopped; terminal result and raw preserved: " + error["message"]
        )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", required=True, choices=AXES)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", required=True, type=Path)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--child-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-request-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--child-sign", choices=SIGNS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    child = args.child_request is not None
    if not child and _gate_failures(_rel(OUTPUT)):
        raise e181.ResourceGuard("Earlier E183 driver gate failure blocks retry")
    try:
        _, gate = _gate(args)
    except Exception as error:
        if not child:
            _record_gate_failure(args, error)
        raise
    if child:
        if args.child_request_sha256 is None or args.child_sign is None:
            raise ValueError("Exact child request hash and sign required")
        result = _child(args, gate)
    elif args.child_request_sha256 is not None or args.child_sign is not None:
        raise ValueError("Child-only argument without request")
    else:
        result = _driver(args, gate)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
