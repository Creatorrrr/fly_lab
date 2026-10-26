"""E309 single reduced-u SSP candidate with unchanged startup and short screens."""
from __future__ import annotations

# First scientific errors stay primary through publication and cleanup.
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
from tools import analyze_c_banc_m3_resource_muscle_contrast as e308
from tools import verify_c_banc_m3_ssp_neutral_body as e307

runner, e306, e301, e305 = e307.runner, e307.e306, e307.e301, e307.e305
OWNED = ("tools/verify_c_banc_m3_reduced_resource_body.py", "tests/test_c_banc_reduced_resource_body.py")
SOURCE_FILES = tuple(dict.fromkeys((*OWNED, *e307.SOURCE_FILES, *e308.OWNED, *e308.REFERENCE_SOURCES)))
PATHS, POLICY, STARTUP_LIMITS = e307.PATHS, e307.POLICY, copy.deepcopy(e307.STARTUP_LIMITS)
LIMITS = {**copy.deepcopy(e306.LIMITS), "internal_wall_s": 110, "science_wall_s": 80, "finalization_reserve_s": 30,
          "body_constructors": 1, "scientific_neural_ticks": 6000, "scientific_physics_steps": 15000}
SUPERVISOR_WALL_S = 120
FREEZE_SCHEMA = "flylab.m3.e309.reduced-resource-body.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e309.reduced-resource-body.review.v1"
RESULT_SCHEMA = "flylab.m3.e309.reduced-resource-body.result.v1"
DESIGN = "docs/BANC_M3_E309_REDUCED_RESOURCE_BODY_DESIGN_20260928.md"
DESIGN_SHA = "34f480e62e987511788701d2a97e533efbc2fb9ef81bfaf6bd82804697dd4ba5"
OUTPUT_PATH = "verification/banc-natural-gait-m3-20260919/e309-reduced-resource-body"
REDUCED_U = .005
COUNTS = {"sources": 7136, "active": 1513, "edges": 725849, "motors": 391}
NEURAL_FIELDS = ("rate", "adaptation", "drive", "output_mask", "tick")
REFERENCE_PATHS = {"neural": e307.OUTPUT_PATH + "/neutral/first-control-neural", "trace": e307.OUTPUT_PATH + "/neutral/trace.json",
                   "native": e307.OUTPUT_PATH + "/neutral/native-contact.npz"}
FROZEN_SHAS = {"e307": {"freeze": "49bb80e8195eedf48d1759cdd6c15cbab85390aacf8a274066c1127d812aaeba",
                         "review": "9fa808ba6ff67903ba7a52e1a164c546ad0629daf840d8ee6edc62f4ef255c9d",
                         "result": "1ddd4521fa37e75ab8706c37a43e7b516d11a575ac78b1e58489f0ced6c5ddf3",
                         "output_manifest": "a5997b314a778a01328e3d0746a0de47cba8db790ef92743581c0a1602621ec4",
                         "root_audit": "68bf39f8780d3b9de97cf75c8d62fcf367c52489a041914ac37b673b9ac062c8"},
               "e308": {"freeze": "3080952373d006ff038c3760fb6d01743cb02197964e05006655239e3485b607",
                         "review": "40c94923cb4eb02d8f30ee59adf415dc191dfdf31fe568f743d6ad31a8dd2bbd",
                         "result": "b9312bb4e7947acbfd6bcdf3ef11aacc3e577267cdd73028a0afce321c4af670",
                         "output_manifest": "d69111a9c26a04cb00abb6a22f3757d8d00d29b205ce15429db746a57f2fb8ca",
                         "root_audit": "8a38ff3f16793ab19d11d2bb1b9dc2e9e2013ba4a169288b0dfa2cc12f8d68f9"}}
LINEAGE_PATHS = {number: {"freeze": prefix + "-freeze.json", "review": prefix + "-source-review.json",
                          **{alias: prefix + "/" + name for alias, name in (("result", "result.json"), ("result_sha256", "result-sha256.json"),
                              ("source_manifest", "source-manifest.json"), ("output_manifest", "output-manifest.json"))},
                          "root_audit": "verification/banc-natural-gait-m3-20260919/parent-" + number + "-original-raw-audit.json"}
                 for number, prefix in (("e307", e307.OUTPUT_PATH), ("e308", e308.OUTPUT))}
LINEAGE_PATHS["e307"]["ledger"] = e307.OUTPUT_PATH + "/advance-ledger.json"
safe = e307.safe


def reduced_selection(original, *, expected_counts=COUNTS):
    """Copy every native array; use literal f64 .005 solely on old .05 slots."""
    if not isinstance(original, dict) or set(original) != runner.layer.SELECTION_KEYS:
        raise ValueError("Complete original selection required")
    u = original["utilization"]
    if (not isinstance(u, np.ndarray) or u.dtype != np.float64 or u.ndim != 1 or not np.isfinite(u).all()
            or not np.isin(u, (np.float64(0), np.float64(.05))).all() or np.count_nonzero(u > 0) != expected_counts["active"]
            or len(u) != expected_counts["sources"] or len(original["edge_ids"]) != expected_counts["edges"]
            or len(original["motor_ids"]) != expected_counts["motors"]):
        raise ValueError("Original literal .05/zero protected mask and counts required")
    runner.layer.require_equal(original["recovery_tau"], np.full(u.shape, .15, np.float64), "Original unchanged tau .15")
    candidate = copy.deepcopy(original)
    candidate["utilization"] = np.where(u > 0, np.float64(.005), np.float64(0))
    for key in original:
        if key != "utilization":
            runner.layer.require_equal(candidate[key], original[key], "Every other original selection byte " + key)
    if any(np.shares_memory(candidate[key], original[key]) for key in candidate):
        raise ValueError("Candidate must not mutate original roster storage")
    return candidate


def validate_reduced_identity(identity, freeze):
    if (identity != freeze["reduced_candidate_network_identity"] or identity in
            (freeze["candidate_network_identity"], freeze["neutral_network_identity"], freeze["source_network_identity"])):
        raise ValueError("Distinct actual reduced-u SSP identity required")


def lineage_records(freeze, pins, number):
    specs = freeze[number + "_lineage"]
    if set(specs) != set(LINEAGE_PATHS[number]):
        raise ValueError("Complete " + number + " lineage receipts required")
    for alias, path in LINEAGE_PATHS[number].items():
        if specs[alias]["path"] != path or alias in FROZEN_SHAS[number] and specs[alias]["sha256"] != FROZEN_SHAS[number][alias]:
            raise ValueError("Literal immutable " + number + " lineage required")
    records = {alias: e305.receipt(spec, pins) for alias, spec in specs.items()}
    manifest = records["output_manifest"]
    for name, entry in manifest["files"].items():
        path = safe((e307.OUTPUT_PATH if number == "e307" else e308.OUTPUT) + "/" + name)
        digest = entry if isinstance(entry, str) else entry["sha256"]
        if pins.get(path.relative_to(ROOT).as_posix()) != digest or runner.protocol.sha256(path) != digest or isinstance(entry, dict) and path.stat().st_size != entry["bytes"]:
            raise ValueError("Every actual frozen output byte required")
    pin = records["result_sha256"]
    expected_schema = (e307.RESULT_SCHEMA if number == "e307" else e308.SCHEMA).replace("result", "result-sha256")
    # E307 inherited receipt deliberately uses its shorter recorded schema.
    if number == "e307":
        expected_schema = "flylab.m3.e307.result-sha256.v1"
    if pin != {"schema": expected_schema, "path": "result.json", "sha256": FROZEN_SHAS[number]["result"]}:
        raise ValueError("Independent actual result SHA receipt required")
    audit = records["root_audit"]
    if audit.get("result_sha256") != FROZEN_SHAS[number]["result"] or audit.get("output_manifest_sha256") != FROZEN_SHAS[number]["output_manifest"]:
        raise ValueError("Audited result/output manifest bindings required")
    return records


def validate_e308(result, audit):
    if (result.get("schema") != e308.SCHEMA or result.get("run_valid") is not True or result.get("publication_status") != "complete"
            or audit.get("schema") != "flylab.m3.e308.root-saved-raw-audit.v1" or audit.get("pass") is not True
            or audit.get("diagnostic_run_valid") is not True or audit.get("all_source_input_and_output_pins_verified") is not True
            or any(result.get(key) is not False or audit.get(key) is not False for key in
                ("candidate_adopted", "F1_F8_complete", "F2_complete", "F6_exposed", "biological_claim", "causal_claim"))
            or result.get("full_current_claim") is not False or result.get("cost") != audit.get("cost")):
        raise ValueError("Valid saved-only unpromoted E308 attribution required")
    for key in ("model_calls", "neural_ticks", "physics_steps", "cuda_calls", "fit_calls", "solver_calls", "retries"):
        if result["cost"].get(key) != 0:
            raise ValueError("E308 saved diagnostic may not create trajectories")


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if runner.protocol.sha256(freeze_path) != freeze_sha or runner.protocol.sha256(review_path) != review_sha:
        raise ValueError("E309 freeze/review SHA mismatch")
    freeze, review = (json.loads(Path(path).read_text(encoding="utf8")) for path in (freeze_path, review_path))
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("diagnostic_id") != "E309" or freeze.get("owned_files") != list(OWNED)
            or freeze.get("resource_limits") != LIMITS or freeze.get("supervisor_wall_s") != SUPERVISOR_WALL_S
            or freeze.get("startup_limits") != STARTUP_LIMITS or freeze.get("policy") != POLICY or freeze.get("candidate_adopted") is not False
            or freeze.get("candidate_utilization") != REDUCED_U or freeze.get("first_control_references") != REFERENCE_PATHS
            or freeze.get("output") != OUTPUT_PATH or freeze.get("design") != DESIGN or freeze.get("design_sha256") != DESIGN_SHA):
        raise ValueError("Exact prospective single reduced-u candidate contract required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved matching E309 source review required")
    pins = e301.base.pin_groups(freeze["input_sha256"], freeze["reused_source_sha256"], freeze["preservation_sha256"], review["source_sha256"], {DESIGN: DESIGN_SHA})
    if not set(SOURCE_FILES) <= set(pins):
        raise ValueError("Every actually reused E309 source pin required")
    neutral = lineage_records(freeze, pins, "e307")
    old = neutral["freeze"]
    e306.require_pins(pins, old["input_sha256"], old["reused_source_sha256"], old["preservation_sha256"],
                       neutral["review"]["source_sha256"], neutral["source_manifest"]["source_sha256"], {old["design"]: old["design_sha256"]})
    if (old.get("schema") != e307.FREEZE_SCHEMA or old.get("resource_limits") != e307.LIMITS
            or old.get("supervisor_wall_s") != e307.SUPERVISOR_WALL_S or neutral["review"].get("schema") != e307.REVIEW_SCHEMA
            or neutral["review"].get("approved") is not True or neutral["review"].get("blocking_defects") != []
            or neutral["review"].get("freeze_sha256") != FROZEN_SHAS["e307"]["freeze"]
            or neutral["source_manifest"].get("owned_sha256") != neutral["review"]["source_sha256"]
            or neutral["source_manifest"].get("input_sha256") != old["input_sha256"]):
        raise ValueError("Actual approved E307 source and context required")
    e307.e306_lineage(old, pins)
    for key in (*e307.BASE_KEYS, "packages", "paths", "ssp_lineage", "ssp_checkpoints", "e306_lineage", "comparator_paths", "neutral_network_identity", "candidate_network_identity"):
        if freeze.get(key) != old[key]:
            raise ValueError("Unchanged inherited E307 context required: " + key)
    saved = lineage_records(freeze, pins, "e308")
    saved_freeze, saved_review, saved_source = saved["freeze"], saved["review"], saved["source_manifest"]
    if (saved_freeze.get("schema") != e308.FREEZE_SCHEMA or saved_freeze.get("contract") != e308.CONTRACT
            or saved_review.get("schema") != e308.REVIEW_SCHEMA or saved_review.get("approved") is not True or saved_review.get("blocking_defects") != []
            or saved_review.get("freeze_sha256") != FROZEN_SHAS["e308"]["freeze"]
            or saved_review.get("source_sha256") != saved_freeze["owned_sha256"]
            or saved_source.get("owned_sha256") != saved_freeze["owned_sha256"] or saved_source.get("reference_source_sha256") != saved_freeze["reference_source_sha256"]):
        raise ValueError("Exact approved E308 saved attribution source lineage required")
    e306.require_pins(pins, saved_freeze["owned_sha256"], saved_freeze["reference_source_sha256"], saved_freeze["input_sha256"],
                       {saved_freeze["design"]["path"]: saved_freeze["design"]["sha256"]})
    validate_e308(saved["result"], saved["root_audit"])
    producer_pins = e308.Pins(ROOT, pins)
    e308.validate_producer("neutral", saved_freeze["arms"]["neutral"], producer_pins)
    e308.validate_producer("active", saved_freeze["arms"]["active"], producer_pins)
    for name, version in freeze["packages"].items():
        actual = platform.python_version() if name == "python" else importlib.metadata.version(name)
        if actual != version:
            raise ValueError("Pinned actual E309 package differs")
    NativeDependencies().load(freeze)
    if output.resolve() != safe(OUTPUT_PATH) or output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh literal E309 output required")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E309 disk reserve unavailable")
    return freeze


def compare_neural_fields(state, reference):
    for key in NEURAL_FIELDS:
        runner.layer.require_equal(state[key], reference[key], "Same first SSP neuron field " + key)
    return {"neural_fields_exact": True, "fields": list(NEURAL_FIELDS), "whole_network_reference_exact_claim": False,
            "excluded_from_reference_equality": ["synaptic_resource", "recurrent_resource", "model_hash"], "numeric_tolerance": 0}


def compare_first_body_native(raw, reference_rows, reference_native):
    if len(raw["rows"]) != 2 or len(raw["native"]) != 50 or len(reference_rows) < 2:
        raise ValueError("Complete same first body control prefix required")
    for index in (0, 1):
        runner.layer.require_equal(runner.protocol.jsonable(raw["rows"][index]), reference_rows[index], "Entire first scientific body trace row")
    arrays = runner.native.native_arrays(raw["native"])
    if set(arrays) != set(reference_native):
        raise ValueError("Complete same native reference channels required")
    for key, actual in arrays.items():
        runner.layer.require_equal(actual, reference_native[key][:50], "Same first native physical samples " + key)
    return {"body_trace_exact": True, "native_prefix_exact": True, "control_rows": 2, "native_samples": 50,
            "wall_time_scope": "Standalone observed wall cost is retained separately from scientific trace/native arrays"}


def summarize_resources(raw, data, factors):
    rows = [row for row in raw["activity"] if row.get("complete") is True]
    resource = np.stack([row["synaptic_resource"] for row in rows])
    u = data["selection"]["utilization"]
    g = factors["g32"]
    return {"active_sources": int(np.count_nonzero(u > 0)), "candidate_utilization": REDUCED_U,
            "minimum_resource": float(np.min(resource)), "minimum_endpoint_factor": float(np.min(g)),
            "factor_nonunity_samples": int(np.count_nonzero(g < 1)), "factor_nonunity_active_sources": int(np.count_nonzero(np.any(g[:, u > 0] < 1, axis=0))),
            "completed_activity_endpoints": len(rows), "scope": "Completed endpoint activity; not internal stage history",
            "baseline_policy": POLICY, "executed_policy": "Same protected active slots use literal float64 .005; all exempt slots remain zero",
            "physiological_adjudication": False, "selection_criterion": False}


class NativeDependencies(e306.NativeDependencies):
    def __init__(self, *, trace_loader=None, native_loader=None, **kwargs):
        super().__init__(**kwargs)
        self._trace_loader, self._native_loader = trace_loader, native_loader
        self.resource_summary = {}

    def load(self, freeze):
        data = super().load(freeze)
        data["baseline_selection"] = copy.deepcopy(data["selection"])
        data["baseline_metadata"] = self.engine.resource_metadata(data["saved"]["network"]["model_hash"], data["baseline_selection"])
        self.engine.checked_resource_metadata(data["baseline_metadata"])
        if self.engine.resource_identity(data["baseline_metadata"]) != freeze["candidate_network_identity"]:
            raise ValueError("Exact inherited active .05 metadata identity required")
        data["selection"] = reduced_selection(data["baseline_selection"])
        data["roster"] = copy.deepcopy(data["roster"])
        data["roster"]["utilization"] = data["selection"]["utilization"].copy()
        metadata = self.engine.resource_metadata(data["saved"]["network"]["model_hash"], data["selection"])
        self.engine.checked_resource_metadata(metadata)
        validate_reduced_identity(self.engine.resource_identity(metadata), freeze)
        data["reduced_metadata"] = metadata
        checkpoint_loader = self._checkpoint_loader
        if checkpoint_loader is None:
            from flylab.c.storage import StateStore
            checkpoint_loader = StateStore.load
        data["reference_neural"] = checkpoint_loader(safe(REFERENCE_PATHS["neural"]))
        runner.layer.require_equal(data["reference_neural"], self.neural_references["neutral"], "Actual E307/E305 u0 neural reference")
        trace_loader = self._trace_loader or (lambda path: json.loads(path.read_text(encoding="utf8"))["samples"])
        data["reference_rows"] = trace_loader(safe(REFERENCE_PATHS["trace"]))[:2]
        native_loader = self._native_loader or runner.protocol.load_npz
        data["reference_native"] = native_loader(safe(REFERENCE_PATHS["native"]))
        self.data = data
        return data

    def construct(self, arm, data, ledger, persist, on_return):
        if arm != "candidate":
            raise ValueError("E309 only constructs one reduced candidate")
        return super().construct(arm, data, ledger, persist, on_return)

    def initial(self, session, data, arm):
        changes, expected_changes = [], []
        saved = session.snapshot()
        actual = runner.prior.checkpoint_representation(self.wrapper.remove_resource_layer(saved), changes)
        expected = runner.prior.checkpoint_representation(data["saved"], expected_changes)
        runner.layer.require_equal(actual, expected, "Original whole initial bytes after new resource layer removal")
        self.validate_initial_original(actual)
        network = saved["network"]
        if (arm != "candidate" or session.network.identity != data["freeze"]["reduced_candidate_network_identity"]
                or session.adapter.identity != data["freeze"]["adapter_identity"] or session.control_tick != 0 or session.network.tick != 0
                or abs(session.body.physics_time()) > 2e-8 or session.network.device != "cuda" or session.network.cuda_implementation != "packed"
                or session.network.interval_steps != 20):
            raise ValueError("Actual reduced candidate identity and original CUDA clocks required")
        runner.layer.require_equal(session.network.selection, data["selection"], "Actual reduced selection")
        runner.layer.require_equal(network["recurrent_resource"], data["reduced_metadata"], "Actual reduced metadata")
        runner.layer.require_equal(network["synaptic_resource"], np.ones(len(data["source_ids"]), np.float64), "Initial fresh R1")
        return {"storage_representation_exact": True, "actual_representation_changes": changes, "expected_representation_changes": expected_changes,
                "numeric_tolerance": 0, "array_coercions": 0, "excluded_fields": [], "reduced_identity_exact": True,
                "whole_network_reference_exact_claim": False, "candidate_utilization": REDUCED_U}

    def first_control(self, session, raw):
        data = self.data
        state = session.network.snapshot()
        raw["ssp_first_control_neural"] = copy.deepcopy(state)
        raw["ssp_diagnostic_readouts"] = {"additional_neural_snapshots": 1, "neutral_reused_whole_snapshot": False}
        receipt = compare_neural_fields(state, data["reference_neural"])
        runner.layer.require_equal(state["recurrent_resource"], data["reduced_metadata"], "First actual reduced metadata")
        self.engine.checked_resource_metadata(state["recurrent_resource"])
        validate_reduced_identity(state["model_hash"], data["freeze"])
        n = len(data["saved"]["network"]["rate"])
        for key in ("rate", "adaptation", "drive", "output_mask"):
            value = state[key]
            if value.dtype != np.float32 or value.shape != (n,) or not np.isfinite(value).all():
                raise ValueError("Finite native actual first reduced neuron field required")
        resource = state["synaptic_resource"]
        if (state["tick"] != 20 or np.any(state["rate"] < 0) or np.any(state["adaptation"] < 0)
                or resource.dtype != np.float64 or resource.shape != data["source_ids"].shape or not np.isfinite(resource).all()
                or np.any(resource < 0) or np.any(resource > 1)):
            raise ValueError("First reduced SSP finite nonnegative/range contract required")
        inactive = data["selection"]["utilization"] == 0
        runner.layer.require_equal(resource[inactive], np.ones(np.count_nonzero(inactive), np.float64), "Exempt first resources one")
        g = e305.endpoint_factor(state, data["selection"])
        raw["reduced_first_resource"] = {"resource": resource.copy(), "g32": g.copy()}
        runner.layer.require_equal(g, np.ones_like(g), "Actual first reduced endpoint factors one")
        receipt.update(compare_first_body_native(raw, data["reference_rows"], data["reference_native"]))
        receipt.update(reduced_identity=state["model_hash"], new_metadata_exact=True, first_endpoint_g_unity=True,
                       first_resource_minimum=float(np.min(resource)), candidate_utilization=REDUCED_U)
        self.first_control_checks["candidate"] = receipt
        raw["ssp_first_control_check"] = copy.deepcopy(receipt)

    def startup(self, raw, data):
        result = super().startup(raw, data)
        arrays = runner.native.native_arrays(raw["native"])
        upright, nonfoot = arrays["upright"], arrays["nonfoot_contact"]
        healthy = bool(len(upright) == 5000 and np.isfinite(upright).all() and np.min(upright) >= .9 and not np.any(nonfoot))
        result.update(native_health_pass=healthy, minimum_native_upright=float(np.min(upright)) if len(upright) else None,
                      nonfoot_contact_count=int(np.count_nonzero(nonfoot)), strict_E141_command_screen=result["strict_E141_startup_screen"])
        result["strict_E141_startup_screen"] = result["strict_E141_startup_screen"] and healthy
        return result

    def save_endpoints(self, folder, raw, data):
        arrays, summary = e301.endpoint_factors(raw, data, "candidate")
        self.resource_summary = summarize_resources(raw, data, arrays)
        np.savez_compressed(folder / "recovery-factor-endpoints.npz", **arrays)
        self.write(folder / "recovery-factor-endpoints.json", {**summary, **self.resource_summary})

    def save_raw(self, folder, raw, data, clock):
        failures = super().save_raw(folder, raw, data, clock)
        for name, callback in (("reduced-parameter-selection.npz", lambda: np.savez_compressed(folder / "reduced-parameter-selection.npz",
                    **{prefix + key: value for prefix, selection in (("baseline_", data["baseline_selection"]), ("reduced_", data["selection"])) for key, value in selection.items()})),
                ("reduced-parameter-metadata.json", lambda: self.write(folder / "reduced-parameter-metadata.json",
                    {"baseline_policy": POLICY, "executed_utilization": REDUCED_U, "baseline_metadata": data["baseline_metadata"], "reduced_metadata": data["reduced_metadata"],
                     "baseline_candidate_identity": data["freeze"]["candidate_network_identity"], "reduced_candidate_identity": data["freeze"]["reduced_candidate_network_identity"]})),
                ("first-control-resource.npz", lambda: np.savez_compressed(folder / "first-control-resource.npz", **raw["reduced_first_resource"]))):
            if name == "first-control-resource.npz" and "reduced_first_resource" not in raw:
                continue
            try:
                callback()
            except Exception as error:
                failures.append({"artifact": name, "error": repr(error)})
        return failures

    def write(self, path, value):
        if path.name == "source-manifest.json":
            value = {**value, "schema": "flylab.m3.e309.sources.v1", "diagnostic_id": "E309",
                "owned_sha256": {name: runner.protocol.sha256(ROOT / name) for name in OWNED},
                "source_sha256": {**value.get("source_sha256", {}), **{name: runner.protocol.sha256(ROOT / name) for name in SOURCE_FILES if name not in OWNED}},
                "executed_body_runner": "tools.verify_c_banc_m3_reduced_resource_body.execute",
                "reused_arm_runner": "tools.verify_c_banc_m3_recurrent_resource_body.run_arm",
                "baseline_policy": POLICY, "executed_candidate_utilization": REDUCED_U, "neutral_session_constructed": False}
        return runner.NativeDependencies.write(self, path, value)


def execute(freeze, out, *, deps=None, budget=None, cleanup=None):
    if freeze.get("diagnostic_id") != "E309":
        raise ValueError("Explicit E309 candidate diagnostic required")
    deps = NativeDependencies() if deps is None else deps
    began = time.perf_counter()
    out.mkdir(exist_ok=False)
    ledger = {
        "schema": "flylab.m3.e309.ledger.v1",
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
        "diagnostic_id": freeze.get("diagnostic_id", "E309"),
        "run_valid": False,
        "first_control_neural_fields_exact": False,
        "first_control_body_native_exact": False,
        "whole_network_reference_exact_claim": False,
        "biological_claim": False,
        "candidate_utilization": .005,
        "resource_perturbation_observed": False,
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
            raise TimeoutError("E309 science deadline/reserve reached")
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
                raise MemoryError("E309 host/GPU cap exceeded")
            if shutil.disk_usage(out).free < LIMITS["minimum_free_disk_bytes"]:
                raise OSError("E309 output disk reserve reached")

    try:
        deps.write(
            out / "source-manifest.json",
            {
                "schema": "flylab.m3.e309.sources.v1",
                "input_sha256": freeze.get("input_sha256", {}),
                "source_sha256": freeze.get("reused_source_sha256", {}),
                "owned_sha256": {name: runner.protocol.sha256(ROOT / name) for name in OWNED},
            },
        )
        data = deps.load(freeze)
        candidate, primary = runner.run_arm("candidate", data, freeze, out, ledger, deps, check)
        result["arms"]["candidate"] = candidate
        receipt = getattr(deps, "first_control_checks", {}).get("candidate", {})
        result["first_control_comparison"] = copy.deepcopy(receipt)
        result["first_control_neural_fields_exact"] = receipt.get("neural_fields_exact") is True
        result["first_control_body_native_exact"] = receipt.get("body_trace_exact") is True and receipt.get("native_prefix_exact") is True
        result["resource_observation"] = copy.deepcopy(getattr(deps, "resource_summary", {}))
        result["resource_perturbation_observed"] = result["resource_observation"].get("factor_nonunity_samples", 0) > 0
        result["actual_startup_pass"] = primary is None and candidate.get("startup", {}).get("strict_E141_startup_screen") is True
        if primary is None and candidate.get("completed_controls") == 300:
            result["short_adjudication"] = runner.assess_short(data["baseline_short"], candidate.get("short", {}))
            result["short_gait_pass"] = result["short_adjudication"]["pass"]
        result["next_separate_10s_eligible"] = result["actual_startup_pass"] and result["short_gait_pass"]
        result["run_valid"] = primary is None
        result["validity"] = (
            "valid_reduced_resource_body_diagnostic"
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
        > LIMITS["body_constructors"]
        or result["cost"]["unknown_cost_arms"]
        or result["cost"]["unknown_setup_calls"]
        or any(row.get("unknown_step_entries") or row.get("active_unreturned_frames") for row in ledger["body_construction"])
    ):
        primary = primary or RuntimeError(
            "E309 scientific clocks/caps unavailable or exceeded"
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
        manifest["schema"] = "flylab.m3.e309.output-manifest.v1"
        deps.write(out / "output-manifest.json", manifest)
        if (time.perf_counter() - began >= LIMITS["internal_wall_s"]
                or time.perf_counter() - finalization_began >= LIMITS["finalization_reserve_s"]):
            raise TimeoutError("E309 final publication deadline")
        result["publication_status"] = "complete"
        deps.write(out / "result.json", result)
        deps.write(out / "result-sha256.json", {"schema": "flylab.m3.e309.result-sha256.v1", "path": "result.json", "sha256": runner.protocol.sha256(out / "result.json")})
        if (time.perf_counter() - began >= LIMITS["internal_wall_s"]
                or time.perf_counter() - finalization_began >= LIMITS["finalization_reserve_s"]):
            raise TimeoutError("E309 late result publication deadline")
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
            primary.add_note("Secondary E309 " + stage + ": " + repr(failure))
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
                            ("publication-status.json", {"schema": "flylab.m3.e309.publication-status.v1", "status": "invalid",
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
                failures[0].add_note("Secondary E309 cleanup: " + repr(error))
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
