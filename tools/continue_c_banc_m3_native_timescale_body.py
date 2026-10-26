"""E142: first native body run from the independently verified E141 zero state.

Parent-owned residual preparation repair after Sol implementation/correction and
the parent's SECOND review. The frozen E141 body protocol is reused unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import verify_c_banc_m3_native_timescale_transfer as body_protocol

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
FREEZE_SHA = "c74dcf6c2a58d90c382cd422ce8f8167da8e449c941e1a19b5ea9542fe0abb68"
DESIGN_SHA = "25fa440798eb93dc356afb9d27364fdfdcbd6fd81ea30377f2a64f692034bb2f"
OWNED = (
    "tools/continue_c_banc_m3_native_timescale_body.py",
    "tests/test_c_banc_cached_native_continuation.py",
)
COPIED = (
    "assembled-parameters.npz",
    "assembled-calibration.npz",
    "actual-global-csr.npz",
    "csr-provenance.json",
    "motor-timescale-configuration.npz",
    "adapter-provenance.npz",
    "muscle-provenance.json",
    "network-layer-identities.json",
)


def verified_inputs(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != FREEZE_SHA or body_protocol.sha256(freeze_path) != FREEZE_SHA:
        raise ValueError("Exact E142 freeze required before CUDA")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    if (
        freeze["schema"] != "flylab.m3.e142.cached-native-continuation.v1"
        or freeze["design_sha256"] != DESIGN_SHA
        or body_protocol.sha256(ROOT / freeze["design_path"]) != DESIGN_SHA
        or tuple(freeze["owned_files"]) != OWNED
        or freeze["prior_conservative_process_s"] != 2033
        or freeze["prepare_wall_limit_s"] != 20
        or freeze["body_wall_limit_s"] != 180
        or freeze["internal_wall_limit_s"] != 210
        or freeze["external_wall_limit_s"] != 220
        or freeze["supervisor_limit_s"] != 235
        or freeze["prospective_cumulative_limit_s"] != 2268
        or freeze["original_program_cumulative_limit_s"] != 2271
        or freeze["conditional_forward_calls"] != 0
        or freeze["original_processes"] != 1
        or freeze["automatic_retries"] != 0
        or freeze["candidate_adopted"] is not False
        or freeze["F6_exposed"] is not False
    ):
        raise ValueError("Frozen E142 continuation contract changed")
    if body_protocol.sha256(review_path) != review_sha:
        raise ValueError("Exact E142 parent source review required")
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (
        review.get("accepted_for_single_cached_body_continuation") is not True
        or review.get("blocking_defects") != []
        or review.get("freeze_sha256") != FREEZE_SHA
        or review.get("source_sha256")
        != {name: body_protocol.sha256(ROOT / name) for name in OWNED}
    ):
        raise ValueError("Accepted exact E142 source gate required")
    for label in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        body_protocol._exact_mapping_hashes(freeze[label], label)
    return freeze, review


def verify_zero(neutral, original):
    for key in (
        "body",
        "world",
        "cuts",
        "navigation",
        "parameters",
        "seed",
        "control_tick",
        "cpg_hash",
        "initial_position",
        "signed_forward_mm",
    ):
        body_protocol.exact(neutral[key], original[key], "E142 original zero " + key)
    for key in ("rate", "adaptation", "drive", "output_mask", "tick"):
        body_protocol.exact(
            neutral["network"][key],
            original["network"][key],
            "E142 original neural zero " + key,
        )
    if (
        neutral["control_tick"] != 0
        or neutral["network"]["tick"] != 0
        or neutral["network"]["device"] != "cuda"
        or neutral["network"]["rate"].shape != (158706,)
    ):
        raise ValueError("Whole158706 CUDA zero checkpoint required")


def load_cached_candidate(freeze, out):
    """Validate and copy the sealed state without constructing a simulation."""
    cached = ROOT / freeze["cached_output_directory"]
    neutral = body_protocol.StateStore.load(ROOT / freeze["zero_checkpoint"])
    original = body_protocol.StateStore.load(ROOT / freeze["original_checkpoint_path"])
    verify_zero(neutral, original)
    for name in COPIED:
        shutil.copyfile(cached / name, out / name)
    shutil.copytree(ROOT / freeze["zero_checkpoint"], out / "candidate-neutral")
    body_protocol.exact(
        neutral,
        body_protocol.StateStore.load(out / "candidate-neutral"),
        "E142 exact copied zero checkpoint",
    )
    graph = body_protocol.GraphStore.load(ROOT / freeze["graph_path"])
    if graph.hash != freeze["graph_hash"] or graph.n != 158706:
        raise ValueError("Original graph identity changed")
    return {
        "graph": graph,
        "neutral": neutral,
        "parameters": body_protocol.load_npz(out / "assembled-parameters.npz"),
        "cal": body_protocol.load_npz(out / "assembled-calibration.npz"),
        "anatomy": json.loads(
            (out / "muscle-provenance.json").read_text(encoding="utf-8")
        ),
        "observation": body_protocol.load_npz(
            BASE / "parent-e113-expected-observation.npz"
        ),
        "startup": body_protocol.load_npz(
            BASE / "e103-startup-preserving-recruitment/startup-inputs.npz"
        ),
    }


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    ledger = {
        "schema": "flylab.m3.e142.cached-native-ledger.v1",
        "status": "created",
        "construction_attempts": [],
        "storage_operations": [],
        "secondary_errors": [],
        "clock": {},
        "main_recorded_controls": 0,
        "conditional_forward_calls": 0,
        "fit_calls": 0,
        "LP_calls": 0,
        "original_processes": 1,
        "candidate_adopted": False,
        "F6_exposed": False,
    }
    entered = False
    limits = None
    primary = None
    resource_failures = []
    result = {
        "experiment": "M3-E142-cached-native-continuation",
        "continuation_of": "E141",
        "validity": "incomplete",
        "hypothesis_verdict": "invalid_not_tested",
        "body_started": False,
        "conditional_forward_calls": 0,
        "historical_conditional_accepted": False,
        "historical_failed_legs": [2, 5],
        "candidate_adopted": False,
        "F6_exposed": False,
        "whole_goal_complete": False,
        "M3_complete": False,
    }

    def persist():
        ledger["construction_totals"] = body_protocol._construction_totals(
            ledger["construction_attempts"]
        )
        ledger["process_wall_s"] = time.perf_counter() - began
        body_protocol.write_json(out / "advance-ledger.json", ledger)

    persist()
    try:
        freeze, review = verified_inputs(
            freeze_path, freeze_sha, review_path, review_sha
        )
        body_protocol.write_json(out / "source-review-snapshot.json", review)
        if shutil.disk_usage(out).free < freeze["minimum_free_disk_bytes"]:
            resource_failures.append("minimum_free_disk_bytes")
            raise OSError("At least12 GiB free required")
        limits, threads = body_protocol.joint_fit.one_cpu_thread()
        ledger["cpu_threads"] = threads
        cached = load_cached_candidate(freeze, out)
        prepare_wall = time.perf_counter() - began
        ledger.update(prepare_wall_s=prepare_wall, status="copied_zero_verified")
        if prepare_wall > freeze["prepare_wall_limit_s"]:
            resource_failures.append("prepare_wall_limit_s")
            raise TimeoutError("E142 cached preparation exceeded20 seconds")
        spec = {
            "experiment": "E142",
            "body_protocol": "E141",
            "continuation_of": "E141",
            "fixed_candidate": True,
            "new_parameter_selection": False,
            "main_controls": 2000,
            "clone_controls": 20,
            "reused_static_provenance": list(COPIED),
            "reused_provenance_origin": "E141 actual preparation",
            "CUDA_cache": "warm after E141; wall time is not a cold-cache performance comparison",
            "freeze_sha256": FREEZE_SHA,
            "source_review_sha256": review_sha,
            "conditional_forward_calls": 0,
            "candidate_adopted": False,
            "F6_exposed": False,
            "whole_goal_complete": False,
        }
        body_protocol.write_json(out / "spec.json", spec)
        # All immutable input and zero-state checks precede the first native construction.
        ledger.update(cuda_runtime_entered=False, status="native_body_requested")
        persist()
        entered = True
        ledger.update(cuda_runtime_entered=True, status="native_body_entered")
        result["body_started"] = True
        body = body_protocol._native_body(
            freeze,
            cached["graph"],
            cached["cal"],
            cached["observation"],
            cached["startup"],
            cached["neutral"],
            cached["parameters"],
            cached["anatomy"],
            out,
            spec,
            ledger,
            persist,
            began,
        )
        result.update(
            body,
            prepare_wall_s=prepare_wall,
            construction_accounting=body_protocol._construction_totals(
                ledger["construction_attempts"]
            ),
            requested_scientific_neural_ticks=40400,
            requested_scientific_physics_steps=101000,
            actual_scientific_neural_ticks=None
            if body["actual_neural_s"] is None
            else round(body["actual_neural_s"] / 0.00025),
            actual_scientific_physics_steps=None
            if body["actual_physical_s"] is None
            else round(body["actual_physical_s"] / 0.0001),
        )
        if (
            body["body_analysis_wall_s"] > freeze["body_wall_limit_s"]
            or body["stop_reason"] == "resource_guard"
        ):
            resource_failures.append("body_wall_limit_s")
            raise TimeoutError("E142 native body/analysis exceeded180 seconds")
        ledger["status"] = "result_ready"
    except Exception as error:  # noqa: BLE001 - preserve and rethrow after finalization
        primary = error
        ledger["status"] = "technical_failed"
        ledger["primary_error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "notes": list(getattr(error, "__notes__", [])),
        }
    finally:
        for label, error in body_protocol.cleanup_cuda_runtime(entered, ledger):
            ledger["secondary_errors"].append({"stage": label, "error": repr(error)})
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary " + label + ": " + repr(error))
        if limits is not None:
            try:
                limits.restore_original_limits()
            except Exception as error:  # noqa: BLE001 - secondary must not mask primary
                if primary is None:
                    primary = error
                else:
                    primary.add_note("Secondary thread restore: " + repr(error))
        try:
            persist()
            if time.perf_counter() - began > 210:
                resource_failures.append("internal_wall_limit_s")
                raise TimeoutError("E142 process exceeded210 seconds")
            if body_protocol._raw_bytes(out) > 4 << 30:
                resource_failures.append("maximum_raw_bytes")
                raise RuntimeError("E142 raw exceeded4 GiB")
            result["output_manifest"] = body_protocol._persist_manifest(out)
        except Exception as error:  # noqa: BLE001 - preserve partial raw and primary
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary finalization: " + repr(error))
        elapsed = time.perf_counter() - began
        if elapsed > 210:
            resource_failures.append("internal_wall_limit_s")
            if primary is None:
                primary = TimeoutError("E142 manifest exceeded210 seconds")
        result.update(
            process_wall_s=elapsed,
            prior_conservative_process_s=2033,
            conservative_cumulative_s=2033 + elapsed,
            resource_limit_met=not resource_failures,
            resource_failures=list(dict.fromkeys(resource_failures)),
            construction_accounting=ledger["construction_totals"],
        )
        if primary is not None:
            result.update(
                validity="invalid_resource" if resource_failures else "incomplete",
                hypothesis_verdict="invalid_not_tested",
                technical_failure=True,
                primary_error={
                    "type": type(primary).__name__,
                    "message": str(primary),
                    "notes": list(getattr(primary, "__notes__", [])),
                },
            )
        try:
            body_protocol.write_json(out / "result.json", result)
        except Exception as error:  # noqa: BLE001 - propagate persistence failure
            if primary is None:
                primary = error
            else:
                primary.add_note("Secondary result persistence: " + repr(error))
    if primary is not None:
        raise primary
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in (
        "freeze",
        "freeze-sha256",
        "source-review",
        "source-review-sha256",
        "output",
    ):
        parser.add_argument("--" + argument, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.freeze,
                args.freeze_sha256,
                args.source_review,
                args.source_review_sha256,
                args.output,
            )
        )
    )
