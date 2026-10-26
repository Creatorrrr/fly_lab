"""E306 SSP body startup and short gait with frozen E305 neural endpoints."""
from __future__ import annotations

# Preserve scientific exceptions and secondary publication/cleanup failures.
# ruff: noqa: BLE001
import argparse
import copy
import importlib.metadata
import json
import os
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
from tools import verify_c_banc_m3_recovery_resource_body as e301
from tools import verify_c_banc_m3_ssp_first_control as e305

runner = e301.runner
OWNED = ("tools/verify_c_banc_m3_ssp_recovery_body.py", "tests/test_c_banc_ssp_recovery_body.py")
PRODUCERS = (*e305.PRODUCERS, "tools/banc_m3_ssp_recovery_trial.py", "tests/test_c_banc_ssp_recovery_trial.py")
SOURCE_FILES = tuple(dict.fromkeys((*OWNED, *PRODUCERS, *e305.SOURCE_FILES, *e301.SOURCE_FILES)))
PATHS, POLICY = e301.PATHS, e301.base.POLICY
BASE_KEYS = tuple(key for key in e301.BASE_KEYS if key not in ("candidate_network_identity", "packages"))
LIMITS, STARTUP_LIMITS = copy.deepcopy(e301.LIMITS), copy.deepcopy(e301.STARTUP_LIMITS)
FREEZE_SCHEMA = "flylab.m3.e306.ssp-recovery-body.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e306.ssp-recovery-body.review.v1"
RESULT_SCHEMA = "flylab.m3.e306.ssp-recovery-body.result.v1"
DESIGN = "docs/BANC_M3_E306_SSP_BODY_DESIGN_20260928.md"
DESIGN_SHA = "f83bd8045ec90e639eee8bfa34ec91c4ac59ece878e66df33c9952d39fd5ee8a"
OUTPUT_PATH = "verification/banc-natural-gait-m3-20260919/e306-ssp-recovery-body"
E305_FREEZE_SHA = "cc3e69c3fd90f35668abe814b0e1df8ffed6e564192ef1b67fc98bcc1f3a02ac"
E305_REVIEW_SHA = "d6dc8d1f4bf0af6bb80e168198c758500ab229411fc52e64afa90f015616a576"
RECEIPTS = ("e303_source_review", "e303_tiny_qualification", "e304_source_review", "e304_tiny_integration", "e304_actual_integration",
            "e305_freeze", "e305_review", "e305_result", "e305_ledger", "e305_source_manifest", "e305_output_manifest", "e305_root_audit")
CHECKPOINTS = {arm + "_" + phase: e305.OUTPUT_PATH + "/" + arm + "/" + phase + "-neural"
               for arm in ("neutral", "candidate") for phase in ("initial", "final")}
safe, verify_spec = e301.safe, e301.verify_spec


def engine_api():
    from tools import banc_m3_ssp_recovery_synapses
    return banc_m3_ssp_recovery_synapses


def wrapper_api():
    from tools import banc_m3_ssp_recovery_trial
    return banc_m3_ssp_recovery_trial


def require_pins(pins, *groups):
    for group in groups:
        if any(pins.get(name) != digest for name, digest in group.items()):
            raise ValueError("Complete unchanged inherited SSP provenance pins required")


def validate_e305(result, ledger, audit, freeze):
    if (result.get("schema") != e305.RESULT_SCHEMA or result.get("diagnostic_id") != "E305" or result.get("run_valid") is not True
            or audit.get("schema") != "flylab.m3.e305.root-original-raw-audit.v1" or audit.get("pass") is not True
            or audit.get("diagnostic_run_valid") is not True):
        raise ValueError("Actual audited E305 SSP first control required")
    for record in (result, audit):
        if any(record.get(key) is not False for key in e305.FLAGS):
            raise ValueError("E305 evidence must remain unpromoted")
    runner.layer.require_equal(result["cost"], ledger, "Actual E305 cost/ledger")
    cost = audit["cost"]
    expected = {"models": 2, "scientific_ticks": 40, "executed_setup_ticks": 80, "captured_setup_ticks": 40,
                "unknown_setup_ticks": 0, "physics_steps": 0, "body_constructors": 0}
    if cost != expected or ledger.get("models_requested") != 2 or ledger.get("models_returned") != 2 or ledger.get("scientific_ticks") != 40:
        raise ValueError("Exact E305 two-model/setup/science cost required")
    for key in ("body_constructors", "physics_steps", "fit_calls", "solver_calls", "retries"):
        if ledger.get(key) != 0:
            raise ValueError("E305 body/physics/fit/retry must be zero")
    if ledger.get("unknown_tick_arms") != []:
        raise ValueError("Known E305 neural cost required")
    profiles = ledger["construction_profiles"]
    if (len(profiles) != 2 or sum(row["known_completed_setup_steps"] for row in profiles) != 80
            or sum(row["recorded_capture_setup_steps"] for row in profiles) != 40
            or any(row.get("unknown_setup_steps") or row.get("unknown_capture_setup_steps") for row in profiles)):
        raise ValueError("Exact E305 setup and capture costs required")
    for arm in ("neutral", "candidate"):
        entry = ledger["arms"][arm]
        if (entry.get("actual_tick") != 20 or entry.get("requested_steps") != 20 or entry.get("advance_requested") != 1
                or entry.get("advance_returned") != 1 or entry.get("advance_failed") != 0 or entry.get("partial_tick_unknown") is not False
                or entry.get("cleanup_returned") != 1 or entry.get("initial_snapshot_available") is not True or entry.get("final_snapshot_available") is not True):
            raise ValueError("Actual completed E305 twenty-tick arm required")
        identity = freeze[arm + "_network_identity"]
        if result["identities"].get(arm) != identity or audit["identities"].get(arm) != identity:
            raise ValueError("Exact new E305 SSP identity required")
    runner.layer.require_equal(result["numeric_comparisons"], audit["numeric_comparisons"], "Audited E305 comparisons")


def ssp_lineage(freeze, pins):
    specs = freeze["ssp_lineage"]
    if set(specs) != set(RECEIPTS) or freeze.get("ssp_checkpoints") != CHECKPOINTS:
        raise ValueError("Complete explicit SSP receipts and literal four checkpoints required")
    records = {name: e305.receipt(specs[name], pins) for name in RECEIPTS}
    old, review = records["e305_freeze"], records["e305_review"]
    if (specs["e305_freeze"]["path"] != e305.OUTPUT_PATH + "-freeze.json" or specs["e305_freeze"]["sha256"] != E305_FREEZE_SHA
            or specs["e305_review"]["path"] != e305.OUTPUT_PATH + "-source-review.json" or specs["e305_review"]["sha256"] != E305_REVIEW_SHA
            or old.get("schema") != e305.FREEZE_SCHEMA or old.get("diagnostic_id") != "E305"
            or review.get("schema") != e305.REVIEW_SCHEMA or review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != E305_FREEZE_SHA or review.get("source_sha256") != {name: old["source_sha256"][name] for name in e305.OWNED}):
        raise ValueError("Literal approved E305 source freeze required")
    require_pins(pins, old["source_sha256"], old["input_sha256"], old["preservation_sha256"], {old["design"]: old["design_sha256"]})
    for name in e305.RECEIPTS:
        e305.receipt(old[name], pins)
    e305.validate_qualification(records["e303_source_review"], records["e303_tiny_qualification"], {name: pins[name] for name in PRODUCERS[:2]})
    wrapper_review, integration = records["e304_source_review"], records["e304_tiny_integration"]
    if (wrapper_review.get("schema") != "flylab.m3.e304.root-source-review.v1" or wrapper_review.get("approved") is not True
            or wrapper_review.get("blocking_defects") != [] or wrapper_review.get("source_sha256") != {name: pins[name] for name in PRODUCERS[2:]}
            or integration.get("schema") != "flylab.m3.e304.root-tiny-integration.v1" or integration.get("approved") is not True
            or integration.get("source_review_sha256") != specs["e304_source_review"]["sha256"]
            or integration.get("integration_sha256") != specs["e304_actual_integration"]["sha256"]
            or integration.get("e303_qualification_sha256") != specs["e303_tiny_qualification"]["sha256"]
            or integration.get("actual_tiny_neural") is not True or integration.get("native_body") is not False):
        raise ValueError("Actual source-bound E304 neural integration required")
    validate_e305(records["e305_result"], records["e305_ledger"], records["e305_root_audit"], freeze)
    source = records["e305_source_manifest"]
    if (source.get("source_sha256") != old["source_sha256"] or source.get("input_sha256") != old["input_sha256"]
            or source.get("owned_sha256") != review["source_sha256"]):
        raise ValueError("Actual E305 executed sources required")
    manifest = records["e305_output_manifest"]
    if manifest.get("schema") != "flylab.m3.e305.outputs.v1" or manifest.get("result_published_separately") is not True:
        raise ValueError("Original E305 manifest contract required")
    for alias, filename in (("e305_result", "result.json"), ("e305_ledger", "ledger.json"), ("e305_source_manifest", "source-manifest.json"), ("e305_output_manifest", "output-manifest.json")):
        if specs[alias]["path"] != e305.OUTPUT_PATH + "/" + filename:
            raise ValueError("Literal E305 producer output required")
    audit = records["e305_root_audit"]
    if (audit.get("result_sha256") != specs["e305_result"]["sha256"] or audit.get("output_manifest_sha256") != specs["e305_output_manifest"]["sha256"]):
        raise ValueError("E305 audit/result/manifest binding required")
    for name, entry in manifest["files"].items():
        path = safe(e305.OUTPUT_PATH + "/" + name)
        if pins.get(path.relative_to(ROOT).as_posix()) != entry["sha256"] or runner.protocol.sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("Actual E305 output manifest bytes differ")
    for folder in CHECKPOINTS.values():
        files = [path for path in safe(folder).rglob("*") if path.is_file()]
        if not files or any(path.relative_to(ROOT).as_posix() not in pins for path in files):
            raise ValueError("Every SSP checkpoint byte required")
    if freeze["packages"] != old["packages"]:
        raise ValueError("Inherited E305 packages required")
    return records


def validate_packages(actual, baseline, ssp, *, python_version=None):
    """Preserve all E296 entries and only inherit E305's pinned Python entry."""
    current = platform.python_version() if python_version is None else python_version
    if (not isinstance(actual, dict) or not isinstance(baseline, dict) or not isinstance(ssp, dict)
            or set(ssp) != set(baseline) | {"python"} or ssp.get("python") != current
            or any(ssp.get(name) != version for name, version in baseline.items()) or actual != ssp):
        raise ValueError("Exact existing E296 packages plus inherited current E305 Python required")


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if runner.protocol.sha256(freeze_path) != freeze_sha or runner.protocol.sha256(review_path) != review_sha:
        raise ValueError("E306 freeze/review SHA mismatch")
    freeze, review = (json.loads(Path(path).read_text(encoding="utf8")) for path in (freeze_path, review_path))
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("diagnostic_id") != "E306" or freeze.get("owned_files") != list(OWNED)
            or freeze.get("paths", {}).keys() != PATHS or freeze.get("policy") != POLICY or freeze.get("output") != OUTPUT_PATH
            or freeze.get("design") != DESIGN or freeze.get("design_sha256") != DESIGN_SHA or freeze.get("candidate_adopted") is not False
            or freeze.get("resource_limits") != LIMITS or freeze.get("startup_limits") != STARTUP_LIMITS):
        raise ValueError("Exact prospective E306 protected SSP body contract required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved matching E306 source review required")
    old, inherited = e301.base_lineage(freeze)
    diagnostic_pins = e301.diagnostic_lineage(freeze, old)
    for key in BASE_KEYS:
        if freeze.get(key) != old[key]:
            raise ValueError("Original E296 inherited context differs: " + key)
    if freeze["paths"] != old["paths"] or freeze["paths"]["protected_selection"] != e301.ROSTER_PATH or freeze["input_sha256"].get(e301.ROSTER_PATH) != e301.ROSTER_SHA:
        raise ValueError("Every original E296 input path and protected roster required")
    pins = e301.base.pin_groups(freeze["input_sha256"], freeze["reused_source_sha256"], freeze["preservation_sha256"], review["source_sha256"], {DESIGN: DESIGN_SHA})
    require_pins(pins, inherited, diagnostic_pins, old["input_sha256"])
    if not set(SOURCE_FILES) <= set(pins):
        raise ValueError("All actually reused E306 source pins required")
    records = ssp_lineage(freeze, pins)
    validate_packages(freeze["packages"], old["packages"], records["e305_freeze"]["packages"])
    for alias, name in freeze["paths"].items():
        path = safe(name)
        files = [item for item in path.rglob("*") if item.is_file()] if path.is_dir() else [path]
        if not files or any(item.relative_to(ROOT).as_posix() not in freeze["input_sha256"] for item in files):
            raise ValueError("Every original body input byte required: " + alias)
    for name, version in freeze["packages"].items():
        actual = platform.python_version() if name == "python" else importlib.metadata.version(name)
        if actual != version:
            raise ValueError("Frozen E306 package differs")
    NativeDependencies().load(freeze)
    if output.resolve() != safe(OUTPUT_PATH) or output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh fixed E306 output required")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E306 disk reserve unavailable")
    return freeze


def numeric_delta(actual, baseline, shape, label):
    """Strict finite numeric arrays; list storage is an explicit representation."""
    a, b = np.asarray(actual), np.asarray(baseline)
    if (a.shape != shape or b.shape != shape or a.dtype != b.dtype or a.dtype.kind not in "fi"
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Missing/shape/dtype/nonfinite neutral comparison: " + label)
    delta = a.astype(np.float64) - b.astype(np.float64)
    return delta, {"max_abs": float(np.max(np.abs(delta), initial=0)), "RMS": float(np.sqrt(np.mean(delta * delta))),
                   "byteexact": a.tobytes() == b.tobytes(), "shape": list(shape), "dtype": a.dtype.str,
                   "scope": label, "actual_representation": type(actual).__name__, "baseline_representation": type(baseline).__name__}


class NativeDependencies(e301.NativeDependencies):
    def __init__(self, *, checkpoint_loader=None, **kwargs):
        super().__init__(**kwargs)
        self._checkpoint_loader = checkpoint_loader
        self.neural_references, self.initial_references, self.first_control_checks = {}, {}, {}

    @property
    def engine(self):
        return engine_api() if self._engine is None else self._engine

    @property
    def wrapper(self):
        return wrapper_api() if self._wrapper is None else self._wrapper

    def load(self, freeze):
        data = super().load(freeze)
        loader = self._checkpoint_loader
        if loader is None:
            from flylab.c.storage import StateStore
            loader = StateStore.load
        for arm in ("neutral", "candidate"):
            initial = loader(safe(freeze["ssp_checkpoints"][arm + "_initial"]))
            final = loader(safe(freeze["ssp_checkpoints"][arm + "_final"]))
            if initial.get("tick") != 0 or final.get("tick") != 20 or final.get("model_hash") != freeze[arm + "_network_identity"]:
                raise ValueError("Original E305 complete SSP checkpoints required")
            self.neural_references[arm] = final
            self.initial_references[arm] = initial
        return data

    def construct(self, arm, data, ledger, persist, on_return):
        def callback():
            owned = self.wrapper.MonotoneRecruitmentTrial.from_checkpoint(data["graph"], copy.deepcopy(data["saved"]))
            try:
                selection = data["selection"]
                trial = self.wrapper.SSPRecoveryResourceTrial(owned, selection["edge_ids"], selection["premotor_ids"],
                    data["roster"]["neutral_utilization"] if arm == "neutral" else selection["utilization"], selection["recovery_tau"])
                trial._e291_arm = arm
                return trial
            except Exception as error:
                self.wrapper.cleanup_owned(owned, error)
                raise
        return self._construction(arm, callback, ledger, persist, on_return)

    def initial(self, session, data, arm):
        receipt = super().initial(session, data, arm)
        runner.layer.require_equal(session.network.snapshot(), self.initial_references[arm], "Entire original E305 SSP initial " + arm)
        return {**receipt, "e305_neural_initial_exact": True}

    def first_control(self, session, raw):
        arm = session._e291_arm
        # Neutral whole endpoint is already captured by the reused collector.
        state = raw["neutral_end_snapshot"]["network"] if arm == "neutral" else session.network.snapshot()
        raw["ssp_first_control_neural"] = copy.deepcopy(state)
        raw["ssp_diagnostic_readouts"] = {"additional_neural_snapshots": int(arm == "candidate"),
                                           "neutral_reused_whole_snapshot": arm == "neutral"}
        runner.layer.require_equal(state, self.neural_references[arm], "Entire E305 SSP first control " + arm)
        self.first_control_checks[arm] = {"ssp_neural_endpoint_exact": True, "control_tick": 1, "network_tick": 20,
                                          "comparison_scope": "Entire new SSP neural snapshot including metadata", "numeric_tolerance": 0}
        raw["ssp_first_control_check"] = copy.deepcopy(self.first_control_checks[arm])

    def capture(self, session, raw):
        super().capture(session, raw)
        if session.control_tick == 1:
            try:
                self.first_control(session, raw)
            except Exception as error:
                self.record_fault(session, session._e291_arm, 1, error)
                raise

    def compare_neutral(self, raw, data):
        if self.first_control_checks.get("neutral", {}).get("ssp_neural_endpoint_exact") is not True:
            raise ValueError("Entire E305 SSP neutral endpoint required")
        actual = self.wrapper.remove_resource_layer(raw["neutral_end_snapshot"])
        baseline = data["baseline_end"]
        deltas, summaries = {}, {}
        for key in ("rate", "adaptation"):
            expected = baseline["network"][key]
            deltas[key], summaries[key] = numeric_delta(actual["network"][key], expected, np.asarray(expected).shape, "new SSP minus old RK4 neural " + key)
        for key, a, b, size, label in (("body_state", actual["body"]["state"], baseline["body"]["state"], 924, "mixed-unit MuJoCo state vector delta"),
                                     ("joint_angles", actual["body"]["action"]["joint_angles"], baseline["body"]["action"]["joint_angles"], 42, "body action joint angles delta")):
            deltas[key], summaries[key] = numeric_delta(a, b, (size,), label)
        raw["ssp_neutral_numeric_deltas"] = deltas
        for row in raw["activity"]:
            runner.layer.require_equal(row["synaptic_resource"], np.ones(len(data["source_ids"]), np.float64), "Neutral resource one")
        return {"neutral_prefix_valid": True, **self.first_control_checks["neutral"], "old_RK4_whole_end_exact_required": False,
                "old_RK4_numeric_comparisons": summaries, "wall_observations": {"actual": actual["wall_s"], "baseline": baseline["wall_s"], "starting": data["saved"]["wall_s"]}}

    def save_neural(self, path, state):
        from flylab.c.storage import StateStore
        StateStore.save(path, state)

    def inherited_save_raw(self, folder, raw, data, clock):
        return super().save_raw(folder, raw, data, clock)

    def save_raw(self, folder, raw, data, clock):
        failures = self.inherited_save_raw(folder, raw, data, clock)
        if "ssp_first_control_neural" in raw:
            try:
                self.save_neural(folder / "first-control-neural", raw["ssp_first_control_neural"])
            except Exception as error:
                failures.append({"artifact": "first-control-neural", "error": repr(error)})
        for name, callback in (("ssp-diagnostic-readouts.json", lambda: self.write(folder / "ssp-diagnostic-readouts.json", raw.get("ssp_diagnostic_readouts", {}))),
                               ("ssp-first-control-check.json", lambda: self.write(folder / "ssp-first-control-check.json", raw.get("ssp_first_control_check", {}))),
                               ("ssp-neutral-numeric-deltas.npz", lambda: np.savez_compressed(folder / "ssp-neutral-numeric-deltas.npz", **raw["ssp_neutral_numeric_deltas"]))):
            if name.endswith(".npz") and "ssp_neutral_numeric_deltas" not in raw:
                continue
            try:
                callback()
            except Exception as error:
                failures.append({"artifact": name, "error": repr(error)})
        return failures

    def write(self, path, value):
        if path.name == "source-manifest.json":
            value = {**value, "schema": "flylab.m3.e306.sources.v1", "diagnostic_id": "E306",
                "owned_sha256": {name: runner.protocol.sha256(ROOT / name) for name in OWNED},
                "source_sha256": {**value.get("source_sha256", {}), **{name: runner.protocol.sha256(ROOT / name) for name in SOURCE_FILES if name not in OWNED}},
                "executed_body_runner": "tools.verify_c_banc_m3_ssp_recovery_body.execute",
                "reused_arm_runner": "tools.verify_c_banc_m3_recurrent_resource_body.run_arm"}
        return runner.NativeDependencies.write(self, path, value)


def execute(freeze, out, *, deps=None, budget=None, cleanup=None):
    deps = NativeDependencies() if deps is None else deps
    began = time.perf_counter()
    out.mkdir(exist_ok=False)
    ledger = {
        "schema": "flylab.m3.e306.ledger.v1",
        "arms": {},
        "construction": [],
        "body_construction": [],
        "original_attempts": 1,
        "retries": 0,
        "fit_calls": 0,
        "nonneutral_current_calls": 0,
    }
    result = {
        "schema": RESULT_SCHEMA,
        "validity": "incomplete",
        "publication_status": "pending",
        "candidate_adopted": False,
        "diagnostic_id": freeze.get("diagnostic_id", "E306"),
        "run_valid": False,
        "neutral_prefix_valid": False,
        "ssp_neural_endpoint_exact": False,
        "actual_startup_pass": False,
        "short_gait_pass": False,
        "adopted": False,
        "next_separate_10s_eligible": False,
        "natural_gait_complete": False,
        "F1_F8_complete": False,
        "arm_receipt_path": "tools.verify_c_banc_m3_recurrent_resource_body.run_arm",
        "E272_executed": False,
        "F6_exposed": False,
        "F2_complete": False,
        "F3_F8_pass": False,
        "ten_second_executed": False,
        "arms": {},
    }
    errors, primary = [], None

    def check():
        if time.perf_counter() - began >= LIMITS["science_wall_s"]:
            raise TimeoutError("E306 science deadline/reserve reached")
        if budget is not None:
            budget()
        elif hasattr(deps, "gpu_free_before"):
            import cupy
            import psutil

            memory = psutil.Process().memory_info()
            peak = getattr(memory, "peak_wset", memory.rss)
            free, _ = cupy.cuda.runtime.memGetInfo()
            added = max(0, deps.gpu_free_before - free)
            ledger["peak_rss_bytes"] = max(peak, ledger.get("peak_rss_bytes", 0))
            ledger["peak_added_gpu_bytes"] = max(
                added, ledger.get("peak_added_gpu_bytes", 0)
            )
            if (
                peak > LIMITS["host_peak_bytes"]
                or added > LIMITS["additional_gpu_peak_bytes"]
            ):
                raise MemoryError("E306 host/GPU cap exceeded")
            if shutil.disk_usage(out).free < LIMITS["minimum_free_disk_bytes"]:
                raise OSError("E306 output disk reserve reached")

    try:
        deps.write(
            out / "source-manifest.json",
            {
                "schema": "flylab.m3.e306.sources.v1",
                "input_sha256": freeze.get("input_sha256", {}),
                "source_sha256": freeze.get("reused_source_sha256", {}),
                "owned_sha256": {name: runner.protocol.sha256(ROOT / name) for name in OWNED},
            },
        )
        data = deps.load(freeze)
        neutral, primary = runner.run_arm(
            "neutral", data, freeze, out, ledger, deps, check
        )
        result["arms"]["neutral"] = neutral
        result["neutral_prefix_valid"] = (
            primary is None
            and neutral.get("neutral_prefix", {}).get("neutral_prefix_valid") is True
            and neutral.get("neutral_prefix", {}).get("ssp_neural_endpoint_exact") is True
        )
        result["ssp_neural_endpoint_exact"] = result["neutral_prefix_valid"]
        if primary is None and not result["neutral_prefix_valid"]:
            raise ValueError("E305 SSP neutral endpoint required before candidate")
        if result["neutral_prefix_valid"]:
            candidate, primary = runner.run_arm(
                "candidate", data, freeze, out, ledger, deps, check
            )
            result["arms"]["candidate"] = candidate
            result["actual_startup_pass"] = (
                primary is None
                and candidate.get("startup", {}).get("strict_E141_startup_screen")
                is True
            )
            if primary is None and candidate.get("completed_controls") == 300:
                result["short_adjudication"] = runner.assess_short(
                    data["baseline_short"], candidate.get("short", {})
                )
                result["short_gait_pass"] = result["short_adjudication"]["pass"]
            result["next_separate_10s_eligible"] = (
                result["actual_startup_pass"] and result["short_gait_pass"]
            )
        result["run_valid"] = primary is None
        result["validity"] = (
            "valid_ssp_recovery_body_diagnostic"
            if primary is None
            else "invalid_or_incomplete"
        )
    except Exception as error:
        primary = error if primary is None else primary
        errors.append(
            {
                "stage": "execution",
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    if cleanup is not None:
        try:
            cleanup()
        except Exception as error:
            primary = primary or error
            errors.append(
                {
                    "stage": "runtime cleanup",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
    if primary is not None:
        result.update(
            run_valid=False,
            actual_startup_pass=False,
            short_gait_pass=False,
            next_separate_10s_eligible=False,
            validity="invalid_or_incomplete",
            first_error={"type": type(primary).__name__, "message": str(primary)},
        )
        errors.append(
            {
                "stage": "primary",
                "type": type(primary).__name__,
                "message": str(primary),
                "notes": getattr(primary, "__notes__", []),
            }
        )
    result["elapsed_s"] = time.perf_counter() - began
    result["cost"] = {
        "scientific_neural_ticks": sum(
            round(v.get("actual_neural_s", 0) / 0.00025)
            for v in ledger["arms"].values()
            if v.get("actual_neural_s") is not None
        ),
        "scientific_physics_steps": sum(
            round(v.get("actual_physical_s", 0) / 0.0001)
            for v in ledger["arms"].values()
            if v.get("actual_physical_s") is not None
        ),
        "unknown_cost_arms": [
            k
            for k, v in ledger["arms"].items()
            if v.get("actual_neural_s") is None or v.get("actual_physical_s") is None
        ],
        "construction_calls": len(ledger["construction"]),
        "known_executed_setup_ticks": sum(
            v.get("known_completed_setup_steps", 0) for v in ledger["construction"]
        ),
        "recorded_capture_setup_ticks": sum(
            v.get("recorded_capture_setup_steps", 0) for v in ledger["construction"]
        ),
        "known_settling_physics_steps": sum(
            v.get("known_completed_settling_steps", 0)
            for v in ledger["body_construction"]
        ),
        "unknown_setup_calls": [
            v.get("label")
            for v in ledger["construction"]
            if v.get("unknown_setup_steps", 0)
            or v.get("unknown_capture_setup_steps", 0)
        ],
    }
    if (
        result["cost"]["scientific_neural_ticks"] > LIMITS["scientific_neural_ticks"]
        or result["cost"]["scientific_physics_steps"]
        > LIMITS["scientific_physics_steps"]
        or len(
            deps.sessions if hasattr(deps, "sessions") else ledger["body_construction"]
        )
        > 2
        or result["cost"]["unknown_cost_arms"]
    ):
        primary = primary or RuntimeError(
            "E306 scientific clocks/caps unavailable or exceeded"
        )
        result.update(
            run_valid=False,
            actual_startup_pass=False,
            short_gait_pass=False,
            next_separate_10s_eligible=False,
            validity="invalid_or_incomplete",
            first_error={"type": type(primary).__name__, "message": str(primary)},
        )
        errors.append({"stage": "cost", "message": str(primary)})
    finalization_began = time.perf_counter()
    try:
        deps.write(out / "advance-ledger.json", ledger)
        deps.write(out / "errors.json", {"errors": errors})
        manifest = runner.short.output_manifest(out, LIMITS["output_bytes"])
        manifest["schema"] = "flylab.m3.e306.output-manifest.v1"
        deps.write(out / "output-manifest.json", manifest)
        if (time.perf_counter() - began >= LIMITS["internal_wall_s"]
                or time.perf_counter() - finalization_began >= LIMITS["finalization_reserve_s"]):
            raise TimeoutError("E306 final publication deadline")
        result["publication_status"] = "complete"
        deps.write(out / "result.json", result)
        deps.write(out / "result-sha256.json", {"schema": "flylab.m3.e306.result-sha256.v1", "path": "result.json", "sha256": runner.protocol.sha256(out / "result.json")})
        if (time.perf_counter() - began >= LIMITS["internal_wall_s"]
                or time.perf_counter() - finalization_began >= LIMITS["finalization_reserve_s"]):
            raise TimeoutError("E306 late result publication deadline")
    except Exception as error:
        primary = primary or error
        result.update(
            run_valid=False,
            actual_startup_pass=False,
            short_gait_pass=False,
            next_separate_10s_eligible=False,
            validity="invalid_or_incomplete",
            first_error={"type": type(primary).__name__, "message": str(primary)},
        )
        for arm in result["arms"].values():
            arm["interpretation_validated"] = False
        result["publication_status"] = "invalid"
        errors.append({"stage": "publication", "type": type(error).__name__, "message": str(error)})
        def secondary(stage, failure):
            errors.append({"stage": stage, "type": type(failure).__name__, "message": str(failure)})
            primary.add_note("Secondary E306 " + stage + ": " + repr(failure))
        def clear_receipt(name):
            try:
                (out / name).unlink(missing_ok=True)
            except Exception as failure:
                secondary("invalidate " + name, failure)
        for name in ("output-manifest.json", "result-sha256.json"):
            clear_receipt(name)
        # Independent attempts make an earlier failed error write unable to
        # suppress the best available invalid result/status publication.
        for name, value in (("errors.json", {"errors": errors}), ("result.json", result),
                            ("publication-status.json", {"schema": "flylab.m3.e306.publication-status.v1", "status": "invalid",
                                "run_valid": False, "first_error": result["first_error"], "success_receipts_trusted": False})):
            try:
                deps.write(out / name, value)
            except Exception as failure:
                secondary("invalid publication " + name, failure)
                if name == "result.json":
                    clear_receipt(name)
        # Include secondary invalidation/write failures whenever still writable.
        try:
            deps.write(out / "errors.json", {"errors": errors})
        except Exception as failure:
            secondary("final errors publication", failure)
    return result, primary

def run(freeze_path, freeze_sha, review_path, review_sha, output):
    freeze = gate(freeze_path, freeze_sha, review_path, review_sha, output)
    import cupy
    import torch
    from threadpoolctl import threadpool_limits
    previous_threads, previous_env, limiter, cleaned = torch.get_num_threads(), {}, None, False
    def cleanup():
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        failures = []
        for callback in ((lambda: limiter.restore_original_limits()) if limiter else lambda: None, lambda: torch.set_num_threads(previous_threads)):
            try:
                callback()
            except Exception as error:
                failures.append(error)
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if failures:
            for error in failures[1:]:
                failures[0].add_note("Secondary E306 cleanup: " + repr(error))
            raise failures[0]
    try:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            previous_env[name] = os.environ.get(name)
            os.environ[name] = "1"
        torch.set_num_threads(1)
        limiter = threadpool_limits(limits=1)
        deps = NativeDependencies()
        deps.gpu_free_before, _ = cupy.cuda.runtime.memGetInfo()
        result, error = execute(freeze, output, deps=deps, cleanup=cleanup)
        return 0 if error is None and result["run_valid"] else 1
    finally:
        cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "source-review", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    for key in ("freeze-sha256", "source-review-sha256"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    return run(args.freeze, args.freeze_sha256, args.source_review, args.source_review_sha256, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
