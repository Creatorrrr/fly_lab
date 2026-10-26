"""E180 fixed physical-objective pilot. Import is inert; execution needs a parent gate.

The only perturbation is a 1% convex move of E150 slopes toward the original
10 ms anatomical pool. This module never fits parameters or changes a trial in
motion. A three-second arm is a diagnostic, not an F3/F4/F5 verdict.
"""

# Broad catches preserve first neural/body failures while independently saving raw
# state, clock, native contact, receipts, and secondary persistence failures.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ENTRY_STARTED = time.perf_counter()

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control
from tools import banc_m3_monotone_recruitment as monotone
from tools import verify_c_banc_m3_native_monotone_body as e150
from tools import verify_c_banc_m3_native_timescale_transfer as protocol
from tools.analyze_c_banc_biological_gait import analyze, load_model

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "flylab.m3.e180.native-objective-freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e180.parent-source-review.v1"
OWNED = (
    "tools/probe_c_banc_m3_native_objective.py",
    "tests/test_c_banc_native_objective.py",
)
LIMITS = {"q_RMS": 0.02, "q_max_abs": 0.04, "LTM_RMS": 0.02, "LTM_max_abs": 0.04}
CONTROLS = 600
ALPHAS = {"baseline": 0.0, "candidate": 0.01}


class ResourceGuard(RuntimeError):
    """Frozen process budget exhausted; scientific arm is incomplete."""


@dataclass(frozen=True)
class ValidatedGate:
    freeze: dict[str, Any]
    freeze_sha256: str
    arm: str


def blend_slopes(original: Any, pool: Any, fixed: Any, alpha: float) -> np.ndarray:
    """Return new slopes; alpha=0 copies original bytes without arithmetic."""
    source = np.asarray(original)
    anatomy = np.asarray(pool)
    rows = np.asarray(fixed)
    if (
        source.dtype != np.float64
        or source.ndim != 4
        or source.shape[2:] != (3, 2)
        or anatomy.dtype != np.float64
        or anatomy.shape != source.shape[:2]
        or rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or len(np.unique(rows)) != len(rows)
        or np.any(rows < 0)
        or np.any(rows >= len(anatomy))
        or not np.isfinite(source).all()
        or not np.isfinite(anatomy).all()
        or np.any(source < 0)
        or np.any(anatomy < 0)
        or not np.isfinite(alpha)
        or alpha not in (0.0, 0.01)
    ):
        raise ValueError("Exact finite E180 slope/pool/fixed/alpha contract required")
    if alpha == 0.0:
        return source.copy(order="K")
    canonical = np.zeros_like(source)
    canonical[:, :, 1, :] = anatomy[:, :, None]
    if not np.array_equal(source[rows], canonical[rows]):
        raise ValueError("Fixed rows must already equal canonical anatomy")
    result = (1.0 - alpha) * source + alpha * canonical
    result[rows] = source[rows]
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Frozen input must be repository-relative")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("Frozen input escaped repository")
    return resolved


def _check_hashes(mapping: dict[str, str]) -> None:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Nonempty frozen hash map required")
    for relative, expected in mapping.items():
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or _sha256(_safe(relative)) != expected
        ):
            raise ValueError("Frozen input/source changed: " + str(relative))


def verified_inputs(
    freeze_path: Path,
    freeze_sha: str,
    review_path: Path,
    review_sha: str,
    arm: str,
    baseline_review_path: Path | None = None,
    baseline_review_sha: str | None = None,
) -> ValidatedGate:
    """Validate all immutable bytes before GraphStore or StateStore access."""
    if arm not in ALPHAS or len(freeze_sha) != 64 or _sha256(freeze_path) != freeze_sha:
        raise ValueError("Exact E180 freeze and arm required")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    required = {
        "schema": SCHEMA,
        "alpha_baseline": 0.0,
        "alpha_candidate": 0.01,
        "duration_s": 3.0,
        "controls": CONTROLS,
        "startup_limits": LIMITS,
        "neural_steps_per_control": 20,
        "physics_steps_per_control": 50,
        "control_dt_s": 0.005,
        "neural_dt_s": 0.00025,
        "physics_dt_s": 0.0001,
        "internal_wall_limit_s": 100,
        "supervisor_limit_s": 110,
        "minimum_free_disk_bytes": 4 << 30,
        "maximum_raw_bytes": 1 << 30,
        "automatic_retries": 0,
        "original_processes_max": 2,
        "fit_evaluations": 0,
        "F6_exposures": 0,
        "candidate_adopted": False,
        "goal_status": "active",
        "cumulative_before_s": 3423.9201528000413,
        "cumulative_limit_s": 3656,
        "owned_files": list(OWNED),
    }
    if any(freeze.get(key) != value for key, value in required.items()):
        raise ValueError("E180 fixed experiment contract changed")
    if freeze.get(
        "design_path"
    ) != "docs/BANC_M3_E180_NATIVE_OBJECTIVE_DESIGN_20260923.md" or _sha256(
        _safe(freeze["design_path"])
    ) != freeze.get("design_sha256"):
        raise ValueError("E180 design changed")
    for key in (
        "graph_path",
        "neutral_checkpoint",
        "startup_path",
        "reference_directory",
    ):
        _safe(freeze[key])
    if (
        len(freeze.get("input_sha256", {})) != 92
        or len(freeze.get("reused_source_sha256", {})) != 90
        or len(freeze.get("preservation_sha256", {})) != 2013
    ):
        raise ValueError("Frozen E180 input/source/preservation inventory changed")
    for key in ("input_sha256", "reused_source_sha256", "preservation_sha256"):
        _check_hashes(freeze[key])
    if len(review_sha) != 64 or _sha256(review_path) != review_sha:
        raise ValueError("Exact parent source review required")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("accepts_native_objective") is not True
        or review.get("blocking_defects") != []
        or review.get("freeze_sha256") != freeze_sha
        or set(review.get("source_sha256", {})) != set(OWNED)
    ):
        raise ValueError("Accepted parent source gate required")
    for name, digest in review["source_sha256"].items():
        if _sha256(_safe(name)) != digest:
            raise ValueError("Reviewed E180 source changed: " + name)
    if arm == "candidate":
        if (
            baseline_review_path is None
            or baseline_review_sha is None
            or len(baseline_review_sha) != 64
            or _sha256(baseline_review_path) != baseline_review_sha
        ):
            raise ValueError("Candidate requires exact parent baseline review")
        baseline = json.loads(baseline_review_path.read_text(encoding="utf-8"))
        if (
            baseline.get("schema") != "flylab.m3.e180.parent-baseline-review.v1"
            or baseline.get("accepts_baseline") is not True
            or baseline.get("permits_candidate") is not True
            or baseline.get("blocking_defects") != []
            or baseline.get("freeze_sha256") != freeze_sha
            or not isinstance(baseline.get("baseline_artifact_sha256"), dict)
            or not baseline["baseline_artifact_sha256"]
        ):
            raise ValueError("Parent baseline comparison did not authorize candidate")
        _check_hashes(baseline["baseline_artifact_sha256"])
    return ValidatedGate(freeze, freeze_sha, arm)


def _neutral(saved: dict[str, Any]) -> None:
    if (
        saved.get("schema") != monotone.TRIAL_SCHEMA
        or saved.get("control_tick") != 0
        or not np.array_equal(saved["adapter"]["bank"], np.zeros((391, 3)))
        or not np.array_equal(saved["adapter"]["activation"], np.zeros(96))
    ):
        raise ValueError("Whole E150 monotone zero checkpoint required")


def restore_arm(graph: Any, saved: dict[str, Any], alpha: float):
    """Construct an independent original-schema trial at time zero only."""
    _neutral(saved)
    if alpha == 0.0:
        owner = monotone.MonotoneRecruitmentTrial.from_checkpoint(graph, saved)
        try:
            protocol.exact(saved, owner.snapshot(), "E180 alpha0 whole checkpoint")
            return owner
        except BaseException:
            owner.close()
            raise
    source = copy.deepcopy(saved)
    source.pop("monotone_recruitment_patch")
    source["schema"] = protocol.MotorTimescaleTrial.schema
    source["adapter"] = copy.deepcopy(saved["adapter"]["original"])
    base = protocol.MotorTimescaleTrial.from_checkpoint(graph, source)
    owner = None
    try:
        protocol.exact(source, base.snapshot(), "E180 exact unwrapped E150 source")
        patch = saved["monotone_recruitment_patch"]
        slopes = blend_slopes(
            patch["slopes"],
            base.adapter.calibration["pool"],
            patch["fixed_muscles"],
            alpha,
        )
        owner = monotone.MonotoneRecruitmentTrial(
            base, patch["tau"], patch["kappa"], slopes, patch["membership"]
        )
        e150.verify_wrapper_provenance(saved, source, base, owner)
        if not np.array_equal(
            owner.adapter.bank, np.zeros((391, 3))
        ) or not np.array_equal(owner.adapter.activation, np.zeros(96)):
            raise ValueError("Candidate started with nonzero adapter state")
        return owner
    except BaseException:
        (owner or base).close()
        raise


def summarize_arm(
    reason: str,
    health_stop: bool,
    controls: int,
    native_samples: int,
    neural_s: float | None,
    physical_s: float | None,
    clock_valid: bool,
    startup_ok: bool | None,
    raw_errors: list,
    analysis_errors: list,
) -> dict[str, Any]:
    complete = bool(
        reason == "duration"
        and controls == CONTROLS
        and native_samples == 50 * CONTROLS
        and clock_valid
        and neural_s is not None
        and physical_s is not None
        and abs(neural_s - 3.0) <= 1e-10
        and abs(physical_s - 3.0) <= 1e-8
    )
    technical = bool(raw_errors or analysis_errors or reason == "execution_error")
    validity = (
        "incomplete"
        if technical
        else "measured_native_health_stop"
        if health_stop
        else "valid_3s_diagnostic"
        if complete and startup_ok is True
        else "incomplete"
    )
    return {
        "validity": validity,
        "technical_failure": technical,
        "stop_reason": reason,
        "measured_health_stop": health_stop,
        "complete_3s": complete,
        "startup_pass": startup_ok,
        "completed_controls": controls,
        "native_samples": native_samples,
        "actual_neural_s": neural_s,
        "actual_physical_s": physical_s,
        "clock_valid": clock_valid,
        "raw_persistence_errors": raw_errors,
        "analysis_errors": analysis_errors,
        "F3_final": False,
        "F4_final": False,
        "F5_final": False,
        "candidate_adopted": False,
    }


def compare_arms(baseline: dict, candidate: dict) -> dict[str, Any]:
    """Compare only complete paired diagnostics; do not infer 10-second gates."""
    comparable = (
        baseline.get("validity") == "valid_3s_diagnostic"
        and candidate.get("validity") == "valid_3s_diagnostic"
    )
    b, c = baseline.get("physical_3s", {}), candidate.get("physical_3s", {})
    denominator_ok = bool(
        comparable
        and b.get("fixed_initial_heading_net_forward_mm") is not None
        and b["fixed_initial_heading_net_forward_mm"] > 0
        and b.get("net_over_path") is not None
        and b["net_over_path"] > 0
        and b.get("native_slip_load_weighted_mm_s") is not None
        and c.get("native_slip_load_weighted_mm_s") is not None
    )
    screens = {
        "forward_at_least_95pct": bool(
            denominator_ok
            and c.get("fixed_initial_heading_net_forward_mm") is not None
            and c["fixed_initial_heading_net_forward_mm"]
            >= 0.95 * b["fixed_initial_heading_net_forward_mm"]
        ),
        "net_over_path_at_least_95pct": bool(
            denominator_ok
            and c.get("net_over_path") is not None
            and c["net_over_path"] >= 0.95 * b["net_over_path"]
        ),
        "native_slip_not_increased": bool(
            denominator_ok
            and c["native_slip_load_weighted_mm_s"]
            <= b["native_slip_load_weighted_mm_s"]
        ),
        "longest_forward_bout_at_least_0p3s": bool(
            comparable
            and c.get("longest_forward_bout_s") is not None
            and c["longest_forward_bout_s"] >= 0.3
        ),
        "every_leg_one_eligible_cycle": bool(
            comparable
            and set(c.get("forward_eligible_complete_cycles", {})) == set(protocol.LEGS)
            and all(
                isinstance(c["forward_eligible_complete_cycles"][leg], int)
                and c["forward_eligible_complete_cycles"][leg] >= 1
                for leg in protocol.LEGS
            )
        ),
    }
    return {
        "paired_valid": comparable,
        "denominators_available": denominator_ok,
        "development_screens": screens,
        "development_entry": bool(comparable and all(screens.values())),
        "reason": "complete_pair" if comparable else "incomplete_or_invalid_arm",
        "F3_final": False,
        "F4_final": False,
        "F5_final": False,
        "candidate_adopted": False,
    }


def physical_summary(
    rows: list[dict], native: dict[str, np.ndarray], quality: dict[str, Any]
) -> dict[str, Any]:
    """Use original gait events and a matching [0.5,3) physical window."""
    times = np.asarray([row["time_s"] for row in rows], np.float64)
    position = np.asarray([row["body"]["position"] for row in rows], np.float64)
    basis = np.asarray(rows[0]["body"]["basis"], np.float64)
    window = (times >= 0.5) & (times < 3.0)
    if (
        times.shape != (601,)
        or position.shape != (601, 3)
        or int(window.sum()) != 500
        or not np.isfinite(position).all()
    ):
        raise ValueError("Complete 3s control positions and window required")
    forward = basis[[0, 2], 0]
    if not np.isfinite(forward).all() or np.linalg.norm(forward) <= 0:
        raise ValueError("Finite initial body heading required")
    forward = forward / np.linalg.norm(forward)
    planar = position[window][:, [0, 2]]
    delta = planar[-1] - planar[0]
    path = float(np.linalg.norm(np.diff(planar, axis=0), axis=1).sum())
    net = float(np.linalg.norm(delta))
    nt, loads, slip, valid = (
        native[key] for key in ("time_s", "normal_load_bw", "slip_mm_s", "slip_valid")
    )
    if (
        nt.shape != (30000,)
        or loads.shape != (30000, 6)
        or slip.shape != (30000, 6)
        or valid.shape != (30000, 6)
        or not np.allclose(nt, np.arange(1, 30001) * 0.0001, rtol=0, atol=2e-8)
        or not np.isfinite(loads).all()
    ):
        raise ValueError("Complete native contact clock and loads required")
    supported = ((nt >= 0.5) & (nt < 3.0))[:, None] & (loads >= 0.02)
    selected = supported & valid
    if (
        not selected.any()
        or np.any(supported & ~valid)
        or not np.isfinite(slip[selected]).all()
    ):
        weighted_slip = None
        support_count = 0
    else:
        weight = float(np.sum(loads[selected]))
        weighted_slip = (
            float(np.sum(loads[selected] * slip[selected]) / weight)
            if weight > 0
            else None
        )
        support_count = int(selected.sum())
    cycles = {leg: quality["events"][leg]["complete_cycles"] for leg in protocol.LEGS}
    return {
        "window_s": [0.5, 3.0],
        "fixed_initial_heading_net_forward_mm": float(delta @ forward),
        "net_displacement_mm": net,
        "planar_path_mm": path,
        "net_over_path": net / path if path > 0 else None,
        "native_slip_load_weighted_mm_s": weighted_slip,
        "native_support_samples": support_count,
        "longest_forward_bout_s": quality["window"]["longest_forward_bout_s"],
        "forward_eligible_complete_cycles": cycles,
    }


def run_arm(
    graph: Any,
    neutral: dict,
    alpha: float,
    output: Path,
    *,
    startup: dict[str, np.ndarray],
    limits: dict[str, float],
    wall_limit_s: float = 100.0,
    gate: ValidatedGate | None = None,
    process_began: float | None = None,
) -> dict[str, Any]:
    """Run exactly one reviewed arm; caller must first pass verified_inputs."""
    if (
        not isinstance(gate, ValidatedGate)
        or gate.freeze.get("schema") != SCHEMA
        or gate.arm not in ALPHAS
        or alpha != ALPHAS[gate.arm]
        or alpha not in ALPHAS.values()
        or limits != LIMITS
        or wall_limit_s > 100
    ):
        raise ValueError("Frozen E180 arm/limits required")
    _neutral(neutral)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    if process_began is None:
        process_began = began
    session = restore = journal = trace_stream = None
    rows, native, native_state, raw_errors, analysis_errors = [], [], [], [], []
    clock = protocol.AdvanceLedger()
    reason, health_stop, primary = "duration", False, None
    ledger = {
        "schema": "flylab.m3.e180.execution-ledger.v1",
        "construction_attempts": [],
        "clock": {},
        "completed_controls": 0,
        "candidate_adopted": False,
        "fit_evaluations": 0,
    }

    def persist():
        ledger["construction_totals"] = protocol._construction_totals(
            ledger["construction_attempts"]
        )
        ledger["process_wall_s"] = time.perf_counter() - process_began
        protocol.write_json(out / "advance-ledger.json", ledger)

    persist()
    metadata = {
        "schema": "flylab.m3.e180.arm.v1",
        "alpha": alpha,
        "source_adapter_identity": neutral["monotone_recruitment_patch"][
            "adapter_identity"
        ],
        "source_network_identity": neutral["network"]["model_hash"],
    }
    protocol.write_json(out / "arm-provenance.json", metadata)
    try:
        if time.perf_counter() - process_began >= wall_limit_s - 10:
            raise ResourceGuard("Whole E180 process reserve exhausted before model")
        session = protocol.profiled_construction(
            "restore_e180_arm",
            lambda: restore_arm(graph, neutral, alpha),
            ledger["construction_attempts"],
            persist,
            on_return=lambda _owner: None,
        )
        if session.network.device != "cuda":
            raise ValueError("Full BANC CUDA network required")
        if (
            graph.n != 158706
            or session.network.tick != 0
            or session.control_tick != 0
            or abs(session.body.physics_time()) > 1e-8
        ):
            raise ValueError(
                "Full 158706-neuron graph and zero neural/body clocks required"
            )
        metadata["actual_device"] = str(session.network.device)
        metadata["graph_neurons"] = int(graph.n)
        metadata["adapter_identity"] = session.adapter.identity
        metadata["body_model_hash"] = session.body.model_hash
        protocol.write_json(out / "arm-provenance.json", metadata)
        patch = session.monotone_recruitment_patch
        np.savez_compressed(
            out / "blended-parameters.npz",
            tau=patch["tau"],
            kappa=patch["kappa"],
            slopes=patch["slopes"],
            membership=patch["membership"],
            motor_ids=patch["motor_ids"],
            fixed_muscles=patch["fixed_muscles"],
        )
        StateStore.save(out / "start", session.snapshot())
        rows.append(capture_control(session))
        restore = protocol.install_native_observer(session.body, native)
        journal = protocol.ControlJournal(out / "advance-journal.jsonl")
        observe = e150.make_observer(native_state, original=lambda *_: {})
        detailed = np.asarray(session.adapter.calibration["motor_ids"][:1], np.int64)
        trace_stream = (out / "trace.jsonl").open("x", encoding="utf-8", buffering=1)
        trace_stream.write(
            json.dumps(protocol.jsonable(rows[0]), allow_nan=False) + "\n"
        )
        trace_stream.flush()
        observe(session, detailed, {})
        for tick in range(1, CONTROLS + 1):
            if time.perf_counter() - process_began >= wall_limit_s - 10:
                reason = "resource_guard"
                break
            event = protocol.advance_control(session, "main", tick, clock, journal)
            try:
                row = capture_control(session)
                if (
                    len(native) != 50 * tick
                    or session.network.tick != 20 * tick
                    or abs(session.body.physics_time() - 0.005 * tick) > 1e-8
                ):
                    raise ValueError("Incomplete 20-neural/50-physics control")
                rows.append(row)
                trace_stream.write(
                    json.dumps(protocol.jsonable(row), allow_nan=False) + "\n"
                )
                trace_stream.flush()
                observe(session, detailed, {})
                protocol.record_readout(journal, "main", tick, event, complete=True)
            except Exception as error:
                protocol.record_readout(
                    journal, "main", tick, event, complete=False, error=error
                )
                raise
            if session.fault or session.body.fault:
                reason, health_stop = "native_health_guard", True
                break
            if tick == 99:
                protocol.e121.save_startup_raw(out, rows, startup)
                screen = protocol.strict_startup(rows, native, startup, limits)
                protocol.write_json(out / "startup-screen.json", screen)
                if not screen["strict_E141_startup_screen"]:
                    reason = "startup_guard"
                    break
            if tick % 100 == 0:
                ledger["completed_controls"] = tick
                ledger["clock"] = {
                    "neural_s": clock.neural_s,
                    "physical_s": clock.physical_s,
                    "valid": clock.clock_valid,
                }
                persist()
    except protocol.NativeHealthStop as error:
        reason, health_stop, primary = "native_health_guard", True, error
    except ResourceGuard as error:
        reason, primary = "resource_guard", error
    except Exception as error:
        reason, primary = "execution_error", error
    finally:

        def preserve(label, operation):
            try:
                operation()
            except Exception as error:
                raw_errors.append({"artifact": label, "error": repr(error)})

        if journal is not None:
            preserve("journal_close", journal.close)
        if trace_stream is not None:
            preserve("trace_stream_close", trace_stream.close)
        if session is not None:
            preserve(
                "final_checkpoint",
                lambda: StateStore.save(out / "final", session.snapshot()),
            )
        preserve(
            "trace",
            lambda: protocol.write_json(
                out / "trace.json",
                {
                    "schema": "flylab.m3.e180.native-objective-trace.v1",
                    "controller": "E150 fixed monotone recruitment with E180 slope blend",
                    "graph_hash": graph.hash,
                    "model_hash": session.network.identity
                    if session
                    else neutral["network"]["model_hash"],
                    "body_hash": session.body.model_hash if session else None,
                    "adapter_hash": session.adapter.identity if session else None,
                    "parameters": neutral["parameters"],
                    "samples": rows,
                },
            ),
        )
        preserve(
            "native_contact",
            lambda: np.savez_compressed(
                out / "native-contact.npz", **protocol.native_arrays(native)
            ),
        )
        preserve(
            "native_recruitment_state",
            lambda: e150.save_native_state(
                out / "native-recruitment-state.npz", native_state
            ),
        )
        preserve(
            "clock_events",
            lambda: protocol.write_json(out / "clock-events.json", clock.events),
        )
        if restore is not None:
            preserve("observer_restore", restore)
        if session is not None:
            preserve("session_close", session.close)
        if primary is not None:
            protocol.write_json(
                out / "primary-error.json",
                {"type": type(primary).__name__, "message": str(primary)},
            )
        ledger["completed_controls"] = max(0, len(rows) - 1)
        ledger["clock"] = {
            "neural_s": clock.neural_s,
            "physical_s": clock.physical_s,
            "valid": clock.clock_valid,
            "actual_costs_by_owner": clock.by_owner(),
        }
        ledger["stop_reason"] = reason
        preserve("ledger_final", persist)
    screen_path = out / "startup-screen.json"
    startup_ok = (
        json.loads(screen_path.read_text(encoding="utf-8"))[
            "strict_E141_startup_screen"
        ]
        if screen_path.exists()
        else None
    )
    if (
        reason == "duration"
        and not raw_errors
        and time.perf_counter() - process_began < wall_limit_s - 3
    ):
        try:
            quality, _phi, _events = analyze(load_model(out / "trace.json"))
            protocol.write_json(out / "biological-gait-3s.json", quality)
            physical = physical_summary(rows, protocol.native_arrays(native), quality)
            protocol.write_json(out / "physical-3s.json", physical)
        except Exception as error:
            analysis_errors.append(
                {"artifact": "biological_gait_3s", "error": repr(error)}
            )
    elif reason == "duration":
        reason = "resource_guard"
    outcome = summarize_arm(
        reason,
        health_stop,
        len(rows) - 1,
        len(native),
        clock.neural_s,
        clock.physical_s,
        clock.clock_valid,
        startup_ok,
        raw_errors,
        analysis_errors,
    )
    outcome.update(
        alpha=alpha,
        body_wall_s=time.perf_counter() - began,
        process_wall_s=time.perf_counter() - process_began,
        advance_journal_rows=journal.rows if journal else 0,
        primary_error=repr(primary) if primary else None,
    )
    if reason == "duration" and not analysis_errors and not raw_errors:
        outcome["physical_3s"] = physical
    raw_bytes = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    outcome["raw_bytes"] = raw_bytes
    if raw_bytes > gate.freeze["maximum_raw_bytes"]:
        outcome["technical_failure"] = True
        outcome["validity"] = "incomplete"
        outcome["raw_persistence_errors"].append(
            {
                "artifact": "output_limit",
                "error": "Raw bytes exceeded frozen 1 GiB limit",
            }
        )
    if time.perf_counter() - process_began > wall_limit_s:
        outcome["validity"] = "incomplete"
        outcome["stop_reason"] = "resource_guard"
        outcome["resource_limit_exceeded"] = True
    persist()
    protocol.write_json(out / "result.json", outcome)
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=tuple(ALPHAS))
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", required=True, type=Path)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--baseline-review", type=Path)
    parser.add_argument("--baseline-review-sha256")
    args = parser.parse_args()
    process_began = _ENTRY_STARTED
    gate = verified_inputs(
        args.freeze,
        args.freeze_sha256,
        args.source_review,
        args.source_review_sha256,
        args.arm,
        args.baseline_review,
        args.baseline_review_sha256,
    )
    freeze = gate.freeze
    out = _safe(freeze["output_directories"][args.arm])
    if (
        out.exists()
        or shutil.disk_usage(out.parent).free < freeze["minimum_free_disk_bytes"]
    ):
        raise OSError("Create-only output and 4 GiB free disk required")
    thread_limits, _threads = protocol.joint_fit.one_cpu_thread()
    try:
        startup_raw = protocol.load_npz(_safe(freeze["startup_path"]))
        startup = {
            "target_command": startup_raw["target_command"],
            "target_tendon": startup_raw["target_tendon"],
        }
        graph = GraphStore.load(_safe(freeze["graph_path"]))
        if graph.hash != freeze["graph_hash"]:
            raise ValueError("Frozen graph identity changed")
        neutral = StateStore.load(_safe(freeze["neutral_checkpoint"]))
        result = run_arm(
            graph,
            neutral,
            ALPHAS[args.arm],
            out,
            startup=startup,
            limits=freeze["startup_limits"],
            wall_limit_s=freeze["internal_wall_limit_s"],
            gate=gate,
            process_began=process_began,
        )
    finally:
        thread_limits.restore_original_limits()
    print(json.dumps(result, sort_keys=True))
    if result["technical_failure"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
