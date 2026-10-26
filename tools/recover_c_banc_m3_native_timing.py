"""E182 bounded recovery of the fixed E181 LM proposals.

The E181 verified gate remains the physics contract. The independent E182
freeze and parent reviews authorize only baseline, LM plus, then LM minus in
new create-only outputs. Import and --help never load original model data.
"""

# Broad catches here preserve technical failures and their accounting receipts.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ENTRY_STARTED = time.perf_counter()

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import optimize_c_banc_m3_native_timing as e181

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "verification/banc-natural-gait-m3-20260919/e182-native-timing-recovery"
DESIGN = "docs/BANC_M3_E182_RESOURCE_RECOVERY_DESIGN_20260923.md"
OWNED = (
    "tools/recover_c_banc_m3_native_timing.py",
    "tests/test_c_banc_native_timing_recovery.py",
)
SCHEMA = "flylab.m3.e182.recovery-freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e182.parent-source-review.v1"
BASELINE_REVIEW_SCHEMA = "flylab.m3.e182.parent-baseline-review.v1"
PLUS_REVIEW_SCHEMA = "flylab.m3.e182.parent-plus-review.v1"
ARMS = ("baseline", "plus", "minus")
REVIEWED_RAW = (
    "result.json",
    "trace.json",
    "startup-screen.json",
    "timing-parameters.npz",
    "native-recruitment-state.npz",
    "native-contact.npz",
    "advance-ledger.json",
    "start/state.json",
    "start/manifest.json",
    "final/state.json",
    "final/manifest.json",
)
THETA = {
    "baseline": [0.0] * 6,
    "plus": [0.0, 0.125, 0.0, 0.0, 0.0, 0.0],
    "minus": [0.0, -0.0625, 0.0, 0.0, 0.0, 0.0],
}
INHERITED_S = 3697.5422548998613
LIMIT_S = 4994.0
RESERVE_S = 340.0
STARTUP_S = 10.0
CAMPAIGN_BYTES = 4 << 30
QUEUE_URL = "http://127.0.0.1:8188/queue"
CLIENTS = {
    16248: "2026-09-23T14:37:08.417602+00:00",
    35496: "2026-09-23T14:37:08.433284+00:00",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object required")
    return value


def _sealed(path: Path, digest: str) -> dict[str, Any]:
    if not isinstance(digest, str) or len(digest) != 64 or e181._sha256(path) != digest:
        raise ValueError(f"Exact reviewed bytes required: {path}")
    return _read(path)


def _rel(path: str) -> Path:
    return e181._safe(path)


def _source_gate(args: argparse.Namespace) -> dict[str, Any]:
    freeze = _sealed(args.freeze, args.freeze_sha256)
    required = {
        "schema": SCHEMA,
        "design_path": DESIGN,
        "owned_files": list(OWNED),
        "output_directory": OUTPUT,
        "arms": list(ARMS),
        "theta": THETA,
        "duration_s": 3.0,
        "controls": 600,
        "neural_steps_per_control": 20,
        "physics_steps_per_control": 50,
        "objective_window_s": [0.5, 3.0],
        "inherited_cumulative_s": INHERITED_S,
        "cumulative_limit_s": LIMIT_S,
        "reserve_max_s": RESERVE_S,
        "startup_validation_reserve_s": STARTUP_S,
        "native_children_max": 3,
        "child_internal_s": 100,
        "child_external_s": 105,
        "supervisor_s": 110,
        "driver_wall_s": 150,
        "driver_external_s": 165,
        "minimum_free_disk_bytes": 20 << 30,
        "arm_raw_bytes_max": 1 << 30,
        "campaign_raw_bytes_max": CAMPAIGN_BYTES,
        "cpu_workers": 1,
        "gpu_workers": 1,
        "automatic_retries": 0,
        "candidate_adopted": False,
        "goal_status": "active",
    }
    if any(freeze.get(key) != expected for key, expected in required.items()):
        raise ValueError("E182 fixed recovery contract changed")
    if e181._sha256(_rel(DESIGN)) != freeze.get("design_sha256"):
        raise ValueError("E182 design changed")
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
    for name, sha in (
        ("freeze_path", "freeze_sha256"),
        ("source_review_path", "source_review_sha256"),
        ("baseline_review_path", "baseline_review_sha256"),
    ):
        _sealed(_rel(old[name]), old[sha])
    if (
        e181._sha256(_rel(f"{e181.OUTPUT_DIRECTORY}/axis-lf/result.json"))
        != old["previous_lf_axis_sha256"]
    ):
        raise ValueError("Reviewed E181 LF axis changed")
    review = _sealed(args.source_review, args.source_review_sha256)
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("freeze_sha256") != args.freeze_sha256
        or review.get("accepts_recovery") is not True
        or review.get("blocking_defects") != []
        or set(review.get("source_sha256", {})) != set(OWNED)
    ):
        raise ValueError("E182 parent source review required")
    e181._check_hashes(review["source_sha256"])
    return freeze


def _old_gate(freeze: dict[str, Any], arm: str) -> e181.ValidatedGate:
    old = freeze["e181_gate"]
    candidate = arm != "baseline"
    return e181.verified_inputs(
        _rel(old["freeze_path"]),
        old["freeze_sha256"],
        _rel(old["source_review_path"]),
        old["source_review_sha256"],
        "lm" if candidate else "baseline",
        _rel(old["baseline_review_path"]) if candidate else None,
        old["baseline_review_sha256"] if candidate else None,
        old["previous_lf_axis_sha256"] if candidate else None,
    )


def _check_reviewed_raw(root: Path, arm: str, mapping: Any) -> None:
    arm_root = root / arm
    if (
        not isinstance(mapping, dict)
        or not set(REVIEWED_RAW).issubset(mapping)
        or any(not isinstance(name, str) for name in mapping)
        or not any(
            name.startswith("start/array_") and name.endswith(".npy")
            for name in mapping
        )
        or not any(
            name.startswith("final/array_") and name.endswith(".npy")
            for name in mapping
        )
    ):
        raise ValueError("Complete reviewed E182 raw hash map required")
    for name in mapping:
        if (
            not isinstance(name, str)
            or name.startswith("/")
            or "\\" in name
            or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or not (arm_root / name).resolve().is_relative_to(arm_root.resolve())
        ):
            raise ValueError("Unsafe E182 raw path")
    if any(path.is_symlink() for path in arm_root.rglob("*")):
        raise ValueError("Symlink in reviewed E182 raw")
    actual = {
        path.relative_to(arm_root).as_posix()
        for path in arm_root.rglob("*")
        if path.is_file()
    }
    if set(mapping) != actual:
        raise ValueError("Reviewed E182 raw file inventory changed")
    for name, digest in mapping.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or e181._sha256(arm_root / name) != digest
        ):
            raise ValueError("Reviewed E182 raw changed: " + name)


def _prior_review(root: Path, arm: str, freeze_sha: str) -> None:
    if arm == "baseline":
        return
    baseline_result = root / "baseline" / "result.json"
    baseline_driver = root / "baseline-driver.json"
    comparison = root / "baseline-comparison.json"
    base = _read(baseline_driver)
    if (
        base.get("error") is not None
        or base.get("objective_completed") is not True
        or _read(baseline_result).get("validity") != "valid_3s_diagnostic"
        or _read(comparison).get("exact_reproduction") is not True
    ):
        raise ValueError("Complete E182 baseline and exact prefix required")
    review = _read(root / "parent-baseline-review.json")
    baseline_artifacts = review.get("baseline_artifact_sha256")
    if (
        review.get("schema") != BASELINE_REVIEW_SCHEMA
        or review.get("freeze_sha256") != freeze_sha
        or review.get("baseline_result_sha256") != e181._sha256(baseline_result)
        or review.get("baseline_comparison_sha256") != e181._sha256(comparison)
        or review.get("accepts_baseline") is not True
        or review.get("permits_candidates") is not True
        or review.get("blocking_defects") != []
    ):
        raise ValueError("E182 parent baseline review required")
    _check_reviewed_raw(root, "baseline", baseline_artifacts)
    if arm == "minus":
        plus_result = root / "plus" / "result.json"
        plus_driver = root / "plus-driver.json"
        plus = _read(plus_driver)
        if (
            plus.get("error") is not None
            or plus.get("objective_completed") is not True
            or _read(plus_result).get("validity") != "valid_3s_diagnostic"
        ):
            raise ValueError("Complete E182 plus arm required")
        review = _read(root / "parent-plus-review.json")
        plus_artifacts = review.get("plus_artifact_sha256")
        if (
            review.get("schema") != PLUS_REVIEW_SCHEMA
            or review.get("freeze_sha256") != freeze_sha
            or review.get("plus_result_sha256") != e181._sha256(plus_result)
            or review.get("accepts_minus") is not True
            or review.get("blocking_defects") != []
        ):
            raise ValueError("E182 parent plus review required")
        _check_reviewed_raw(root, "plus", plus_artifacts)


def _queue_counts() -> dict[str, int]:
    with urllib.request.urlopen(QUEUE_URL, timeout=2) as response:
        if response.status != 200:
            raise ValueError(f"Queue HTTP {response.status}")
        payload = json.load(response)
    running, pending = payload["queue_running"], payload["queue_pending"]
    if not isinstance(running, list) or not isinstance(pending, list):
        raise TypeError("Unknown queue schema")
    return {"running": len(running), "pending": len(pending)}


def _client_status() -> dict[str, str]:
    status = {}
    for pid, born in CLIENTS.items():
        try:
            process = psutil.Process(pid)
            expected = datetime.fromisoformat(born).timestamp()
            status[str(pid)] = (
                "live"
                if abs(process.create_time() - expected) <= 0.001
                and process.is_running()
                else "reused"
            )
        except psutil.NoSuchProcess:
            status[str(pid)] = "absent"
        except (psutil.Error, OSError) as error:
            status[str(pid)] = f"unknown:{type(error).__name__}"
    return status


def _observation() -> dict[str, Any]:
    row: dict[str, Any] = {"timestamp_utc": datetime.now(timezone.utc).isoformat()}
    try:
        row["queue"] = _queue_counts()
    except Exception as error:
        row["queue_error"] = repr(error)
    row["clients"] = _client_status()
    row["ready"] = row.get("queue") == {"running": 0, "pending": 0} and all(
        state in ("absent", "reused") for state in row["clients"].values()
    )
    return row


def _readiness() -> list[dict[str, Any]]:
    first = _observation()
    time.sleep(2)
    second = _observation()
    return [first, second]


def _sample_resources() -> dict[str, Any]:
    row: dict[str, Any] = {"timestamp_utc": datetime.now(timezone.utc).isoformat()}
    try:
        row["queue"] = _queue_counts()
    except Exception as error:
        row["queue_error"] = repr(error)
    try:
        command = [
            "nvidia-smi",
            (
                "--query-gpu=timestamp,index,utilization.gpu,memory.used,"
                "memory.total,power.draw,clocks.current.graphics,clocks.current.memory"
            ),
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=2, check=True
        )
        row["gpu_csv"] = result.stdout.strip().splitlines()
    except Exception as error:
        row["gpu_error"] = repr(error)
    return row


def _monitor(path: Path, stop: threading.Event, errors: list[str]) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            while not stop.is_set():
                stream.write(json.dumps(_sample_resources(), sort_keys=True) + "\n")
                stream.flush()
                if stop.wait(2):
                    break
    except Exception as error:
        errors.append(repr(error))


def _account(root: Path) -> dict[str, Any]:
    rows = {arm: e181._attempt_accounting(root / f"{arm}-request.json") for arm in ARMS}
    return {
        "arms": rows,
        "native_arms_upper_bound": sum(
            row["arms_upper_bound"] for row in rows.values()
        ),
        "native_arms_exact": sum(row["arms_exact"] for row in rows.values())
        if all(row["arms_exact"] is not None for row in rows.values())
        else None,
        "native_accounted_wall_s": sum(
            row["accounted_wall_s"] for row in rows.values()
        ),
        "native_cost_exact": all(row["cost_exact"] for row in rows.values()),
    }


def _preflight_receipts(
    root: Path, freeze_sha: str
) -> tuple[list[dict[str, Any]], float]:
    receipts = []
    for arm in ARMS:
        paths = sorted(root.glob(f"{arm}-preflight-*.json"))
        for number, path in enumerate(paths):
            if path.name != f"{arm}-preflight-{number:03d}.json":
                raise ValueError("E182 preflight sequence has a gap or unexpected name")
            row = _read(path)
            elapsed = row.get("elapsed_wall_s")
            if (
                row.get("schema") != "flylab.m3.e182.preflight.v1"
                or row.get("arm") != arm
                or row.get("freeze_sha256") != freeze_sha
                or type(row.get("native_permitted")) is not bool
                or type(elapsed) not in (int, float)
                or not math.isfinite(elapsed)
                or elapsed < 0
            ):
                raise ValueError("Invalid E182 preflight accounting receipt")
            receipts.append(
                {
                    "path": path.name,
                    "sha256": e181._sha256(path),
                    "elapsed_wall_s": elapsed,
                }
            )
    return receipts, sum(row["elapsed_wall_s"] for row in receipts)


def _driver_attempt_gate(root: Path, arm: str, freeze_sha: str) -> tuple[int, float]:
    receipts, prior_cost = _preflight_receipts(root, freeze_sha)
    current = [row for row in receipts if row["path"].startswith(f"{arm}-preflight-")]
    if any(_read(root / row["path"])["native_permitted"] for row in current):
        raise FileExistsError("E182 native launch already permitted for this arm")
    if (
        arm == "baseline"
        and root.exists()
        and any(
            not path.name.startswith("baseline-preflight-") or path.suffix != ".json"
            for path in root.iterdir()
        )
    ):
        raise FileExistsError("E182 baseline output is create-only")
    if any(
        (root / name).exists()
        for name in (f"{arm}-request.json", f"{arm}-driver.json", arm)
    ):
        raise FileExistsError("E182 arm output is create-only")
    if len(current) >= 100:
        raise e181.ResourceGuard("E182 readiness attempt record limit reached")
    return len(current), prior_cost


def _capacity(root: Path, arm: str, began: float, freeze_sha: str) -> None:
    if psutil.disk_usage(str(root.parent)).free < 20 << 30:
        raise e181.ResourceGuard("20 GiB free disk required")
    if e181._directory_bytes(root) > CAMPAIGN_BYTES:
        raise e181.ResourceGuard("E182 campaign raw limit reached")
    account = _account(root)
    if account["native_arms_upper_bound"] != ARMS.index(arm):
        raise e181.ResourceGuard("E182 native order or attempt count changed")
    _, prior = _preflight_receipts(root, freeze_sha)
    # Current elapsed is not yet in a receipt; every earlier attempt is there.
    overhead = max(STARTUP_S, prior + time.perf_counter() - began)
    if INHERITED_S + account["native_accounted_wall_s"] + 110 + overhead > LIMIT_S:
        raise e181.ResourceGuard("Inherited cumulative native reserve exhausted")
    if account["native_accounted_wall_s"] + 110 + overhead > RESERVE_S:
        raise e181.ResourceGuard("E182 340-second reserve exhausted")
    if time.perf_counter() - _ENTRY_STARTED + 110 > 150:
        raise e181.ResourceGuard("E182 driver wall reserve exhausted")


def _child(
    args: argparse.Namespace, freeze: dict[str, Any], gate: e181.ValidatedGate
) -> dict[str, Any]:
    root = _rel(OUTPUT)
    arm = args.arm
    request_path = root / f"{arm}-request.json"
    request = _sealed(request_path, args.child_request_sha256)
    if (
        args.child_request.resolve() != request_path.resolve()
        or request.get("schema") != "flylab.m3.e182.child-request.v1"
        or request.get("arm") != arm
        or request.get("theta") != THETA[arm]
        or request.get("freeze_sha256") != args.freeze_sha256
        or request.get("source_review_sha256") != args.source_review_sha256
        or request.get("e181_freeze_sha256") != gate.freeze_sha256
        or not e181.driver_ancestor(request)
    ):
        raise ValueError("Exact live E182 driver request required")
    preflight_name = request.get("preflight_name")
    if (
        not isinstance(preflight_name, str)
        or not preflight_name.startswith(f"{arm}-preflight-")
        or not preflight_name.endswith(".json")
        or Path(preflight_name).name != preflight_name
    ):
        raise ValueError("Exact E182 readiness receipt name required")
    preflight_path = root / preflight_name
    preflight = _sealed(preflight_path, request.get("preflight_sha256"))
    receipts, _ = _preflight_receipts(root, args.freeze_sha256)
    permitted = [
        row["path"]
        for row in receipts
        if row["path"].startswith(f"{arm}-preflight-")
        and _read(root / row["path"])["native_permitted"]
    ]
    if (
        preflight.get("schema") != "flylab.m3.e182.preflight.v1"
        or preflight.get("arm") != arm
        or preflight.get("freeze_sha256") != args.freeze_sha256
        or preflight.get("source_review_sha256") != args.source_review_sha256
        or preflight.get("native_permitted") is not True
        or preflight.get("error") is not None
        or len(preflight.get("observations", [])) != 2
        or not all(row.get("ready") is True for row in preflight["observations"])
        or permitted != [preflight_name]
    ):
        raise ValueError("Exact permitted E182 readiness receipt required")
    _prior_review(root, arm, args.freeze_sha256)
    output = root / arm
    if output.exists():
        raise FileExistsError("E182 raw output is create-only")
    if e181._directory_bytes(root) > CAMPAIGN_BYTES:
        raise e181.ResourceGuard("E182 campaign raw limit reached")
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
            tuple(THETA[arm]),
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
        output / "e182-provenance.json",
        {
            "schema": "flylab.m3.e182.arm-provenance.v1",
            "arm": arm,
            "theta": THETA[arm],
            "e182_freeze_sha256": args.freeze_sha256,
            "e182_source_review_sha256": args.source_review_sha256,
            "e181_freeze_sha256": gate.freeze_sha256,
            "e181_source_review_sha256": gate.source_review_sha256,
            "request_sha256": args.child_request_sha256,
            "e181_result_sha256": e181._sha256(output / "result.json"),
        },
    )
    if result.get("technical_failure"):
        raise RuntimeError("E182 arm technical failure; E181 raw retained")
    return result


def _driver(
    args: argparse.Namespace, freeze: dict[str, Any], gate: e181.ValidatedGate
) -> dict[str, Any]:
    root = _rel(OUTPUT)
    arm = args.arm
    began = _ENTRY_STARTED
    _prior_review(root, arm, args.freeze_sha256)
    attempt, prior_preflight_cost = _driver_attempt_gate(root, arm, args.freeze_sha256)
    if arm == "baseline" and not root.exists():
        root.mkdir(parents=True, exist_ok=False)
    readiness: list[dict[str, Any]] = []
    error: dict[str, str] | None = None
    pre_child = None
    selection = None
    monitoring_errors: list[str] = []
    preflight_path = root / f"{arm}-preflight-{attempt:03d}.json"
    try:
        _capacity(root, arm, began, args.freeze_sha256)
        readiness = _readiness()
        if not all(row["ready"] for row in readiness):
            raise e181.ResourceGuard(
                "Resource readiness blocked or unknown; native not started"
            )
        _capacity(root, arm, began, args.freeze_sha256)
        pre_child = time.perf_counter() - began
        e181._write_new(
            preflight_path,
            {
                "schema": "flylab.m3.e182.preflight.v1",
                "arm": arm,
                "freeze_sha256": args.freeze_sha256,
                "source_review_sha256": args.source_review_sha256,
                "observations": readiness,
                "native_permitted": True,
                "elapsed_wall_s": pre_child,
                "cumulative_pre_child_wall_s": prior_preflight_cost + pre_child,
                "resource_overhead_reserved_s": max(
                    STARTUP_S, prior_preflight_cost + pre_child
                ),
                "error": None,
            },
        )
        request_path = root / f"{arm}-request.json"
        request = {
            "schema": "flylab.m3.e182.child-request.v1",
            "arm": arm,
            "theta": THETA[arm],
            "parent_pid": os.getpid(),
            "parent_created_s": psutil.Process(os.getpid()).create_time(),
            "freeze_sha256": args.freeze_sha256,
            "source_review_sha256": args.source_review_sha256,
            "e181_freeze_sha256": gate.freeze_sha256,
            "preflight_name": preflight_path.name,
            "preflight_sha256": e181._sha256(preflight_path),
        }
        e181._write_new(request_path, request)
        command = [
            sys.executable,
            "-X",
            "utf8",
            str(Path(__file__).resolve()),
            "--arm",
            arm,
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
        ]
        stop = threading.Event()
        monitor = threading.Thread(
            target=_monitor,
            args=(root / f"{arm}-resources.jsonl", stop, monitoring_errors),
            daemon=True,
        )
        monitor.start()
        try:
            receipt = e181.supervise_command(command, request_path, timeout_s=105)
        finally:
            stop.set()
            monitor.join(timeout=5)
            if monitor.is_alive():
                monitoring_errors.append("monitor_thread_did_not_stop")
        if receipt["status"] != "completed" or receipt["child_wall_s"] > 110:
            raise e181.ResourceGuard(
                "E182 child incomplete; raw and supervisor receipt retained"
            )
        result = _read(root / arm / "result.json")
        if (
            result.get("theta") != THETA[arm]
            or result.get("freeze_sha256") != gate.freeze_sha256
            or result.get("source_review_sha256") != gate.source_review_sha256
        ):
            raise ValueError("E181 raw identity changed")
        if arm == "baseline" and result.get("validity") == "valid_3s_diagnostic":
            e180_comparison = e181.compare_baseline_prefix(
                root / "baseline", _rel(gate.freeze["reference_directory"])
            )
            e181_comparison = e181.compare_baseline_prefix(
                root / "baseline", _rel(f"{e181.OUTPUT_DIRECTORY}/baseline")
            )
            comparison = {
                "schema": "flylab.m3.e182.baseline-comparison.v1",
                "e180": e180_comparison,
                "e181": e181_comparison,
                "exact_reproduction": e180_comparison["exact_reproduction"]
                and e181_comparison["exact_reproduction"],
            }
            e181._write_new(root / "baseline-comparison.json", comparison)
            if comparison.get("exact_reproduction") is not True:
                raise ValueError("E182 zero baseline differs from E180 reference")
        if result.get("validity") != "valid_3s_diagnostic":
            raise ValueError("E182 arm did not complete valid 3s diagnostic")
        if arm == "minus":
            baseline = _read(root / "baseline" / "result.json")
            selection = e181.select_axis(
                THETA["baseline"],
                baseline,
                {
                    "plus": {
                        "theta": THETA["plus"],
                        "result": _read(root / "plus" / "result.json"),
                    },
                    "minus": {"theta": THETA["minus"], "result": result},
                },
                baseline,
            )
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    if not preflight_path.exists():
        failed_elapsed = time.perf_counter() - began
        e181._write_new(
            preflight_path,
            {
                "schema": "flylab.m3.e182.preflight.v1",
                "arm": arm,
                "freeze_sha256": args.freeze_sha256,
                "source_review_sha256": args.source_review_sha256,
                "observations": readiness,
                "native_permitted": False,
                "elapsed_wall_s": failed_elapsed,
                "cumulative_pre_child_wall_s": prior_preflight_cost + failed_elapsed,
                "resource_overhead_reserved_s": max(
                    STARTUP_S, prior_preflight_cost + failed_elapsed
                ),
                "error": error,
            },
        )
    if not (root / f"{arm}-request.json").is_file():
        raise RuntimeError(
            "E182 preflight stopped before native; receipt preserved: " + str(error)
        )
    account = _account(root)
    own = account["arms"][arm]
    raw_path = root / arm / "result.json"
    raw = _read(raw_path) if raw_path.is_file() else {}
    resource_rows = _resource_rows(root / f"{arm}-resources.jsonl")
    preflight_receipts, cumulative_pre_child = _preflight_receipts(
        root, args.freeze_sha256
    )
    cumulative_resource_cost = account["native_accounted_wall_s"] + max(
        STARTUP_S, cumulative_pre_child
    )
    driver = {
        "schema": "flylab.m3.e182.arm-driver.v1",
        "arm": arm,
        "theta": THETA[arm],
        "e182_freeze_sha256": args.freeze_sha256,
        "e182_source_review_sha256": args.source_review_sha256,
        "e181_freeze_sha256": gate.freeze_sha256,
        "e181_source_review_sha256": gate.source_review_sha256,
        "preflight_sha256": e181._sha256(preflight_path),
        "result_sha256": e181._sha256(raw_path) if raw_path.is_file() else None,
        "native_accounting": account,
        "native_started": own["started"],
        "objective_completed": raw.get("validity") == "valid_3s_diagnostic"
        and raw.get("objective_mm_s") is not None,
        "completed_controls": raw.get("completed_controls"),
        "actual_neural_s": raw.get("actual_neural_s"),
        "actual_physical_s": raw.get("actual_physical_s"),
        "pre_child_wall_s": pre_child
        if pre_child is not None
        else time.perf_counter() - began,
        "cumulative_pre_child_wall_s": cumulative_pre_child,
        "preflight_receipts": preflight_receipts,
        "cumulative_resource_budget_wall_s": cumulative_resource_cost,
        "cumulative_with_inherited_s": INHERITED_S + cumulative_resource_cost,
        "driver_wall_s": time.perf_counter() - began,
        "monitoring_errors": monitoring_errors,
        "resource_overlap_or_unknown": any(
            row.get("queue_error")
            or row.get("gpu_error")
            or row.get("queue", {}).get("running")
            or row.get("queue", {}).get("pending")
            for row in resource_rows
        )
        or bool(monitoring_errors)
        or not resource_rows,
        "error": error,
        "selection": selection,
        "candidate_adopted": False,
        "parent_review_required": True,
        "F3_final": False,
        "F4_final": False,
        "F5_final": False,
    }
    if (
        e181._directory_bytes(root) > CAMPAIGN_BYTES
        or e181._directory_bytes(root / arm) > 1 << 30
    ):
        driver["error"] = {
            "type": "ResourceGuard",
            "message": "E182 raw bytes exceeded",
        }
    if driver["driver_wall_s"] > 150:
        driver["error"] = {
            "type": "ResourceGuard",
            "message": "E182 driver wall exceeded",
        }
    if (
        cumulative_resource_cost > RESERVE_S
        or INHERITED_S + cumulative_resource_cost > LIMIT_S
    ):
        driver["error"] = {
            "type": "ResourceGuard",
            "message": "E182 cumulative resource budget exceeded",
        }
    e181._write_new(root / f"{arm}-driver.json", driver)
    if driver["error"]:
        raise RuntimeError(
            "E182 arm stopped; receipts and raw preserved: "
            + driver["error"]["message"]
        )
    return driver


def _resource_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, ValueError):
        return [{"queue_error": "resource_log_unreadable"}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", required=True, type=Path)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--child-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-request-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    freeze = _source_gate(args)
    gate = _old_gate(freeze, args.arm)
    if args.child_request is not None:
        if args.child_request_sha256 is None:
            raise ValueError("Exact child request hash required")
        result = _child(args, freeze, gate)
    elif args.child_request_sha256 is not None:
        raise ValueError("Child hash without request")
    else:
        result = _driver(args, freeze, gate)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
