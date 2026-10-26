"""E239: reuse verified E238 parity and collect the unchanged teacher arm once."""
# Preserve the primary execution failure across cleanup and evidence publication.
# ruff: noqa: BLE001
from __future__ import annotations

import argparse
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
FREEZE_SHA = "e68617b9408f09f5f9d1e7b39f228263ebce5968643fe01443314eb6e6d60b97"
OWNED = ("tools/observe_c_banc_m3_matched_teacher_recovery.py",
         "tests/test_c_banc_matched_teacher_recovery.py")
OUT = BASE / "e239-matched-teacher-recovery"


def verify_reused_parity(proof, result, ledger, supervisor):
    """A resource-limited teacher prefix is not a completed teacher collection."""
    if (proof.get("parity_validity") != "independently_valid_exact"
            or proof.get("complete_teacher_collection") is not False
            or proof.get("arms", {}).get("parity") != {
                "valid_saved_prefix": True, "completed_controls": 99,
                "neural_ticks": 1980, "physics_steps": 4950, "stage_rows": 7920,
                "minimum_upright": .9409362153417923, "complete_requested_arm": True}
            or result.get("parity") != "valid_exact"
            or result.get("teacher") != "resource_limited"
            or result.get("validity") != "resource_limited"
            or result.get("whole_goal_complete") is not False):
        raise ValueError("Independently valid E238 parity and retained teacher failure required")
    arms = ledger.get("arms", [])
    if (len(arms) != 2 or arms[0].get("arm") != "parity"
            or arms[0].get("valid") is not True
            or arms[0].get("controls_completed") != 99
            or arms[1].get("arm") != "teacher"
            or arms[1].get("controls_completed") != 1936
            or arms[1].get("status") != "resource_limited"
            or any(row.get("storage_errors") for row in arms)
            or ledger.get("storage_errors")
            or supervisor.get("exit_code") != 1
            or supervisor.get("live") != [] or supervisor.get("terminated") != []
            or supervisor.get("cumulative_after_s") != 6278.4757079996525):
        raise ValueError("Complete E238 execution and cleanup accounting required")


def gate(freeze_path, freeze_sha, review_path, review_sha, out):
    if (Path(freeze_path).resolve() != (BASE / "e239-matched-teacher-recovery-freeze.json").resolve()
            or freeze_sha != FREEZE_SHA or original.sha256(freeze_path) != FREEZE_SHA
            or original.sha256(review_path) != review_sha):
        raise ValueError("Exact E239 freeze and review required")
    freeze = json.loads(Path(freeze_path).read_text("utf-8"))
    review = json.loads(Path(review_path).read_text("utf-8"))
    if (freeze.get("schema") != "flylab.m3.e239.matched-teacher-recovery.freeze.v1"
            or freeze.get("owned_files") != list(OWNED)
            or review.get("schema") != "flylab.m3.e239.matched-teacher-recovery.review.v1"
            or review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Reviewed parent recovery source required")
    for document, labels in [(freeze, ("input_sha256", "reused_source_sha256", "preservation_sha256")),
                             (review, ("source_sha256", "additional_reused_source_sha256", "additional_input_sha256"))]:
        for label in labels:
            for name, digest in document.get(label, {}).items():
                if original.sha256(original.safe(name)) != digest:
                    raise ValueError("E239 frozen source/input changed: " + name)
    if original.sha256(original.safe(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E239 frozen design changed")
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("E239 frozen package changed: " + name)
    proofs = {key: json.loads(original.safe(path).read_text("utf-8"))
              for key, path in freeze["reused_parity"].items()}
    verify_reused_parity(proofs["proof"], proofs["previous_result"],
                         proofs["previous_ledger"], proofs["previous_supervisor"])
    if out.resolve() != OUT.resolve() or out.exists() or not out.parent.is_dir():
        raise ValueError("New exact E239 output directory required")
    if shutil.disk_usage(out.parent).free < freeze["resource_limits"]["minimum_free_bytes"]:
        raise OSError("E239 minimum free disk unavailable")
    return freeze, review


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze, review = gate(freeze_path, freeze_sha, review_path, review_sha, out)
    out.mkdir(exist_ok=False)
    limits = freeze["resource_limits"]
    ledger = {"schema": "flylab.m3.e239.matched-teacher-recovery.ledger.v1",
              "status": "verified_saved_parity", "original_processes": 1,
              "technical_attempt_in_family": 2, "retries": 0, "fit_calls": 0,
              "solver_calls": 0, "arms": [], "storage_errors": [],
              "reused_parity_neural_ticks": 1980, "new_parity_neural_ticks": 0,
              "candidate_adopted": False, "F6_exposed": False, "whole_goal_complete": False}
    result = {"schema": "flylab.m3.e239.matched-teacher-recovery.result.v1",
              "validity": "incomplete", "execution_status": "running", "parity": "valid_exact",
              "parity_source": "independently verified saved E238, not re-executed",
              "teacher": "not_run", "candidate_adopted": False, "F2_complete": False,
              "F6_exposed": False, "whole_goal_complete": False}
    primary = threads = process = gpu_free_before = None

    def persist():
        ledger["process_wall_s"] = time.perf_counter()-began
        original.e224.write_receipt(out/"advance-ledger.json", ledger)

    def budget():
        if time.perf_counter()-began >= limits["internal_s"]-limits["finalization_reserve_s"]:
            raise TimeoutError("E239 finalization reserve reached")
        if process is not None:
            memory = process.memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0), peak)
            if peak > limits["resident_bytes"]:
                raise MemoryError("E239 resident limit exceeded")
        if gpu_free_before is not None:
            import cupy as cp
            available, _ = cp.cuda.runtime.memGetInfo()
            added = max(0, gpu_free_before-available)
            ledger["peak_added_gpu_bytes"] = max(ledger.get("peak_added_gpu_bytes", 0), added)
            if added > limits["allocated_gpu_bytes"]:
                raise MemoryError("E239 additional GPU limit exceeded")

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
        original.exact(start["body"], teacher_start["body"], "E239 identical teacher body origin")
        with np.load(original.safe(freeze["paths"]["teacher_command"]), allow_pickle=False) as data:
            command = data["command"].copy()
        if command.shape != (2000, 42) or command.dtype != np.float64 or not np.isfinite(command).all():
            raise ValueError("Frozen full E85 teacher commands required")
        tendon = np.full((2000, 8), .2, np.float64)
        ledger["status"] = "teacher_requested_after_saved_parity"
        persist()
        original.execute_registered_arm("teacher", lambda: original.run_arm(
            "teacher", freeze, graph, start, command, tendon, out, ledger, persist,
            budget, owner=teacher_start), result, ledger)
        result.update(validity="valid_matched_input_capture", execution_status="complete",
                      controls=2000, neural_ticks=40000, physics_steps=100000,
                      conclusion="full matched teacher-body collection only; autonomous behavior untested")
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
                    primary.add_note("Secondary E239 thread cleanup: " + repr(error))
        if time.perf_counter()-began > limits["internal_s"] and primary is None:
            primary = TimeoutError("E239 internal limit exceeded")
            result["validity"] = "resource_limited"
        if primary is not None:
            if result["validity"] == "valid_matched_input_capture":
                result["pre_storage_validity"] = result["validity"]
                result["validity"] = "technical_cleanup_failure"
            result.update(execution_status="failed", primary_error=repr(primary))
        primary = original.publish_final_receipts(result, ledger, primary, persist,
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
