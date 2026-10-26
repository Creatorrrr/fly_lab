"""E310 diagnostic continuation of fixed nonzero SSP whole checkpoints."""
from __future__ import annotations

# Preserve first scientific errors and secondary storage/cleanup failures.
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
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import verify_c_banc_m3_reduced_resource_body as e309

runner, e307, e306, e301, e305 = e309.runner, e309.e307, e309.e306, e309.e301, e309.e305
OWNED = ("tools/verify_c_banc_m3_resource_continuation.py", "tests/test_c_banc_resource_continuation.py")
SOURCE_FILES = tuple(dict.fromkeys((*OWNED, *e309.SOURCE_FILES)))
FREEZE_SCHEMA = "flylab.m3.e310.resource-continuation.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e310.resource-continuation.review.v1"
RESULT_SCHEMA = "flylab.m3.e310.resource-continuation.result.v1"
DESIGN = "docs/BANC_M3_E310_RESOURCE_CONTINUATION_DESIGN_20260928.md"
DESIGN_SHA = "40e25fd237fa62ced789c6b81e1b57f3cc19894a57e8b58fffbecbafc9e4e610"
OUTPUT_PATH = "verification/banc-natural-gait-m3-20260919/e310-resource-continuation"
LIMITS = {**copy.deepcopy(e309.LIMITS), "internal_wall_s": 205, "science_wall_s": 175,
          "finalization_reserve_s": 30, "body_constructors": 2, "scientific_neural_ticks": 8000, "scientific_physics_steps": 20000}
SUPERVISOR_WALL_S = 220
ORIGINAL_STARTUP_PASS = {"neutral": True, "candidate": False}
CONTINUATION_INPUTS = {arm: {"checkpoint": prefix + "/final", **{key: prefix + "/" + name for key, name in
    (("trace", "trace.json"), ("native", "native-contact.npz"), ("activity", "activity.npz"),
     ("recruitment", "native-recruitment-state.npz"), ("controls", "control-raw.npz"))}}
    for arm, prefix in (("neutral", e307.OUTPUT_PATH + "/neutral"), ("candidate", e309.OUTPUT_PATH + "/candidate"))}
LINEAGE_PATHS = {**copy.deepcopy(e309.LINEAGE_PATHS), "e309": {"freeze": e309.OUTPUT_PATH + "-freeze.json",
    "review": e309.OUTPUT_PATH + "-source-review.json", **{alias: e309.OUTPUT_PATH + "/" + name for alias, name in
    (("result", "result.json"), ("result_sha256", "result-sha256.json"), ("source_manifest", "source-manifest.json"),
     ("output_manifest", "output-manifest.json"), ("ledger", "advance-ledger.json"))},
    "root_audit": "verification/banc-natural-gait-m3-20260919/parent-e309-original-raw-audit.json"}}
FROZEN_SHAS = {**copy.deepcopy(e309.FROZEN_SHAS), "e309": {
    "freeze": "10a8ae10150e6de006d7705346272e977ce84b99b4833e1cf60d4f8205a17343",
    "review": "6585540f59985d3fabe6bb9431ada58e10342cc4545412b0a3b0d3369bfb1e9c",
    "result": "5c0c7332f391ba4a5d15c7b930c922b059b2d3146d40c755dbca6f9055d21574",
    "output_manifest": "f9f19871106ce50600f1784c8bb345376899d93800ace08c85e5e1ba886c935e",
    "root_audit": "a6ccdeee658864390b601d43a2b7d4fc709ec3b9038edceff827a7ec2a95937b"}}
safe = e309.safe


def validate_producer(result, ledger, audit, arm):
    """Keep the previous .5s verdict and charged costs exact."""
    module, number = (e307, "e307") if arm == "neutral" else (e309, "e309")
    expected = {"scientific_neural_ticks": 2000, "scientific_physics_steps": 5000,
        "unknown_cost_arms": [], "construction_calls": 1, "known_executed_setup_ticks": 280,
        "recorded_capture_setup_ticks": 140, "known_settling_physics_steps": 2000, "unknown_setup_calls": []}
    if (result.get("schema") != module.RESULT_SCHEMA or result.get("diagnostic_id") != number.upper()
            or result.get("run_valid") is not True or result.get("publication_status") != "complete"
            or result.get("actual_startup_pass") is not ORIGINAL_STARTUP_PASS[arm]
            or result.get("cost") != expected or audit.get("cost") != expected
            or audit.get("schema") != "flylab.m3." + number + ".root-original-raw-audit.v1"
            or audit.get("pass") is not True or audit.get("diagnostic_run_valid") is not True
            or audit.get("original_strict_startup_pass") is not ORIGINAL_STARTUP_PASS[arm]
            or ledger.get("original_attempts") != 1 or ledger.get("retries") != 0 or ledger.get("fit_calls") != 0):
        raise ValueError("Exact valid audited .5s producer and unchanged startup verdict required")
    for key in ("candidate_adopted", "adopted", "F1_F8_complete", "F2_complete", "F3_F8_pass", "F6_exposed",
                "natural_gait_complete", "ten_second_executed", "next_separate_10s_eligible", "biological_claim", "short_gait_pass"):
        if result.get(key) is not False:
            raise ValueError("Prior unpromoted diagnostic flags required")
    entry = ledger["arms"][arm]
    if (set(ledger["arms"]) != {arm} or entry.get("completed_controls") != 100 or entry.get("native_samples") != 5000
            or entry.get("clock_valid") is not True or entry.get("close_returned") is not True
            or round(entry.get("actual_neural_s", -1) / .00025) != 2000 or round(entry.get("actual_physical_s", -1) / .0001) != 5000):
        raise ValueError("Known closed exact producer .5s clock required")
    if result["arms"][arm]["startup"].get("strict_E141_startup_screen") is not ORIGINAL_STARTUP_PASS[arm]:
        raise ValueError("Producer arm startup verdict must remain exact")


def e309_lineage(freeze, pins):
    specs = freeze["e309_lineage"]
    if set(specs) != set(LINEAGE_PATHS["e309"]):
        raise ValueError("Complete E309 frozen lineage required")
    for alias, path in LINEAGE_PATHS["e309"].items():
        if specs[alias]["path"] != path or alias in FROZEN_SHAS["e309"] and specs[alias]["sha256"] != FROZEN_SHAS["e309"][alias]:
            raise ValueError("Literal immutable actual E309 receipt required")
    records = {alias: e305.receipt(spec, pins) for alias, spec in specs.items()}
    old, review, source = records["freeze"], records["review"], records["source_manifest"]
    if (old.get("schema") != e309.FREEZE_SCHEMA or old.get("diagnostic_id") != "E309" or old.get("resource_limits") != e309.LIMITS
            or old.get("supervisor_wall_s") != e309.SUPERVISOR_WALL_S or old.get("design") != e309.DESIGN or old.get("design_sha256") != e309.DESIGN_SHA
            or old.get("candidate_utilization") != .005 or old.get("first_control_references") != e309.REFERENCE_PATHS
            or review.get("schema") != e309.REVIEW_SCHEMA or review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FROZEN_SHAS["e309"]["freeze"] or set(review.get("source_sha256", {})) != set(e309.OWNED)
            or source.get("owned_sha256") != review["source_sha256"] or source.get("input_sha256") != old["input_sha256"]
            or source.get("executed_body_runner") != "tools.verify_c_banc_m3_reduced_resource_body.execute"
            or source.get("executed_candidate_utilization") != .005):
        raise ValueError("Exact actually approved/executed E309 source contract required")
    e306.require_pins(pins, old["input_sha256"], old["reused_source_sha256"], old["preservation_sha256"],
                       review["source_sha256"], source["source_sha256"], {old["design"]: old["design_sha256"]})
    neutral = e309.lineage_records(old, pins, "e307")
    e307.e306_lineage(neutral["freeze"], pins)
    e306.require_pins(pins, neutral["freeze"]["input_sha256"], neutral["freeze"]["reused_source_sha256"],
        neutral["freeze"]["preservation_sha256"], neutral["review"]["source_sha256"], neutral["source_manifest"]["source_sha256"],
        {neutral["freeze"]["design"]: neutral["freeze"]["design_sha256"]})
    saved = e309.lineage_records(old, pins, "e308")
    sf, sr, ss = saved["freeze"], saved["review"], saved["source_manifest"]
    if (sf.get("schema") != e309.e308.FREEZE_SCHEMA or sf.get("contract") != e309.e308.CONTRACT
            or sr.get("schema") != e309.e308.REVIEW_SCHEMA or sr.get("approved") is not True or sr.get("blocking_defects") != []
            or sr.get("freeze_sha256") != FROZEN_SHAS["e308"]["freeze"] or sr.get("source_sha256") != sf["owned_sha256"]
            or ss.get("owned_sha256") != sf["owned_sha256"] or ss.get("reference_source_sha256") != sf["reference_source_sha256"]):
        raise ValueError("Unchanged approved E308 lineage required")
    e306.require_pins(pins, sf["owned_sha256"], sf["reference_source_sha256"], sf["input_sha256"],
        {sf["design"]["path"]: sf["design"]["sha256"]})
    e309.validate_e308(saved["result"], saved["root_audit"])
    producer_pins = e309.e308.Pins(ROOT, pins)
    for arm in ("neutral", "active"):
        e309.e308.validate_producer(arm, sf["arms"][arm], producer_pins)
    for key in (*e307.BASE_KEYS, "packages", "paths", "ssp_lineage", "ssp_checkpoints", "e306_lineage", "comparator_paths",
                "neutral_network_identity", "candidate_network_identity", "e307_lineage", "e308_lineage", "candidate_utilization",
                "reduced_candidate_network_identity", "first_control_references"):
        if freeze.get(key) != old[key]:
            raise ValueError("Unchanged inherited E309 context required: " + key)
    for key in (*e307.BASE_KEYS, "packages", "paths", "ssp_lineage", "ssp_checkpoints", "e306_lineage", "comparator_paths",
                "neutral_network_identity", "candidate_network_identity"):
        if old.get(key) != neutral["freeze"][key]:
            raise ValueError("Exact original E307 context retained in E309 required")
    for number, rec in (("e307", neutral), ("e309", records)):
        prefix = e307.OUTPUT_PATH if number == "e307" else e309.OUTPUT_PATH
        for name, entry in rec["output_manifest"]["files"].items():
            path = safe(prefix + "/" + name)
            if pins.get(path.relative_to(ROOT).as_posix()) != entry["sha256"] or runner.protocol.sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                raise ValueError("Every frozen producer output byte required")
        if rec["result_sha256"] != {"schema": "flylab.m3." + number + ".result-sha256.v1", "path": "result.json", "sha256": FROZEN_SHAS[number]["result"]}:
            raise ValueError("Exact independent producer result hash required")
        if (rec["root_audit"].get("result_sha256") != FROZEN_SHAS[number]["result"]
                or rec["root_audit"].get("output_manifest_sha256") != FROZEN_SHAS[number]["output_manifest"]):
            raise ValueError("Root audited result/manifest bindings required")
    validate_producer(neutral["result"], neutral["ledger"], neutral["root_audit"], "neutral")
    validate_producer(records["result"], records["ledger"], records["root_audit"], "candidate")
    return {"neutral": neutral, "candidate": records}


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if runner.protocol.sha256(freeze_path) != freeze_sha or runner.protocol.sha256(review_path) != review_sha:
        raise ValueError("E310 freeze/review SHA mismatch")
    freeze, review = (json.loads(Path(path).read_text(encoding="utf8")) for path in (freeze_path, review_path))
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("diagnostic_id") != "E310" or freeze.get("owned_files") != list(OWNED)
            or freeze.get("resource_limits") != LIMITS or freeze.get("supervisor_wall_s") != SUPERVISOR_WALL_S
            or freeze.get("startup_limits") != e309.STARTUP_LIMITS or freeze.get("policy") != e309.POLICY
            or freeze.get("candidate_adopted") is not False or freeze.get("original_startup_pass") != ORIGINAL_STARTUP_PASS
            or freeze.get("continuation_inputs") != CONTINUATION_INPUTS or freeze.get("output") != OUTPUT_PATH
            or freeze.get("design") != DESIGN or freeze.get("design_sha256") != DESIGN_SHA):
        raise ValueError("Exact fixed two-arm nonzero continuation contract required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved matching E310 source review required")
    pins = e301.base.pin_groups(freeze["input_sha256"], freeze["reused_source_sha256"], freeze["preservation_sha256"], review["source_sha256"], {DESIGN: DESIGN_SHA})
    if not set(SOURCE_FILES) <= set(pins):
        raise ValueError("Every actual reused source pin required")
    e309_lineage(freeze, pins)
    for inputs in CONTINUATION_INPUTS.values():
        for name in inputs.values():
            path = safe(name)
            files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
            if not files or any(p.relative_to(ROOT).as_posix() not in freeze["input_sha256"] for p in files):
                raise ValueError("All whole checkpoint and prefix file bytes required")
    for name, version in freeze["packages"].items():
        actual = platform.python_version() if name == "python" else importlib.metadata.version(name)
        if actual != version:
            raise ValueError("Current pinned package differs")
    NativeDependencies().load(freeze)
    if output.resolve() != safe(OUTPUT_PATH) or output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh literal E310 output required")
    if shutil.disk_usage(output.parent).free < LIMITS["minimum_free_disk_bytes"]:
        raise OSError("E310 disk reserve unavailable")
    return freeze


def validate_global_clocks(rows, native, *, start=0, complete=False):
    controls = np.asarray([row["control_tick"] for row in rows])
    times = np.asarray([row["time_s"] for row in rows])
    if (not len(rows) or not np.array_equal(controls, np.arange(start, start + len(rows)))
            or not np.isfinite(times).all() or not np.allclose(times, controls * .005, rtol=0, atol=2e-8)):
        raise ValueError("Contiguous global control clocks required")
    t = native["time_s"]
    if not np.isfinite(t).all() or not np.allclose(t, (np.arange(1, len(t) + 1) + start * 50) * .0001, rtol=0, atol=2e-8):
        raise ValueError("Contiguous global native clocks required")
    if complete and (len(rows) != 301 - start or len(t) != (300 - start) * 50):
        raise ValueError("Complete exact global interval required")


def compare_initial_readout(raw, prefix):
    if any(len(raw[key]) != 1 for key in ("rows", "controls", "recruitment", "activity")):
        raise ValueError("One actual initial continuation readout required")
    runner.layer.require_equal(runner.protocol.jsonable(raw["rows"][0]), prefix["trace"]["samples"][-1], "Frozen original final scientific trace row")
    for record_key in ("controls", "activity", "recruitment"):
        record, reference = raw[record_key][0], prefix[record_key]
        keys = ("control_tick", "network_tick", "physics_time_s", "motor_rate32", "activation", "command", "tendon") if record_key == "controls" else (
            ("control_tick", "network_tick", "physics_time_s", "complete", "rate", "drive", "adaptation", "synaptic_resource") if record_key == "activity" else
            ("control_tick", "network_tick", "complete", "bank", "activation", "original_activation", "motor_rate"))
        for key in keys:
            runner.layer.require_equal(np.asarray(record[key]), np.asarray(reference[key][-1]), "Original last " + record_key + " " + key)
        if record_key == "recruitment":
            runner.layer.require_equal(np.asarray(record["time_s"]), np.asarray(reference["physics_time_s"][-1]), "Original last recruitment time")
    return {"trace_exact": True, "activity_exact": True, "recruitment_exact": True, "controls_exact": True, "control": 100}


def combine_raw(prefix, raw):
    """Evaluation-only merge: original row0 fixes the original heading."""
    arrays = runner.native.native_arrays(raw["native"])
    validate_global_clocks(prefix["trace"]["samples"], prefix["native"])
    if len(prefix["trace"]["samples"]) != 101 or len(prefix["native"]["time_s"]) != 5000:
        raise ValueError("Exact complete original .5s prefix required")
    validate_global_clocks(raw["rows"], arrays, start=100)
    compare_initial_readout({k: v[:1] if isinstance(v, list) else v for k, v in raw.items()}, prefix)
    joined_native = {}
    if set(arrays) != set(prefix["native"]):
        raise ValueError("Same complete native channels required")
    for key in arrays:
        original, new = prefix["native"][key], arrays[key]
        if original.dtype != new.dtype or original.shape[1:] != new.shape[1:]:
            raise ValueError("Native channel dtype/shape unchanged required")
        joined_native[key] = np.concatenate((original, new), axis=0)
    rows = copy.deepcopy(prefix["trace"]["samples"][:100]) + copy.deepcopy(raw["rows"])
    validate_global_clocks(rows, joined_native)
    return {"rows": rows, "native_arrays": joined_native, "trace_metadata": {k: v for k, v in prefix["trace"].items() if k != "samples"}}


def evaluation_clock(clock, *, restored, final, prefix_verified):
    values = (restored["physics"], final["physics"], clock.neural_s, clock.physical_s)
    if (any(v is None or not np.isfinite(v) for v in values)
            or not prefix_verified or restored["control"] != 100 or restored["neural"] != 2000 or abs(restored["physics"] - .5) > 2e-8
            or final["control"] != 300 or final["neural"] != 6000 or abs(final["physics"] - 1.5) > 2e-8
            or not clock.clock_valid or clock.neural_s is None or clock.physical_s is None
            or abs(clock.neural_s - 1.) > 2e-8 or abs(clock.physical_s - 1.) > 2e-8):
        raise ValueError("Verified whole prefix plus complete measured additional clocks required")
    return SimpleNamespace(neural_s=.5 + clock.neural_s, physical_s=.5 + clock.physical_s,
                           clock_valid=True, origin="Verified frozen .5s prefix + measured continuation delta; evaluation only")


def metric_summary(short):
    window = short.get("gait", {}).get("window", {})
    events = short.get("gait", {}).get("events", {})
    bouts = window.get("bouts")
    durations = [(b["samples"] - 1) * .005 for b in bouts] if isinstance(bouts, list) and all(type(b.get("samples")) is int for b in bouts) else None
    return {"forward_mm": short.get("forward", {}).get("fixed_initial_heading_forward_mm"),
        "slip_mm_s": short.get("slip", {}).get("load_weighted_mm_s"),
        "net_path": {"available": False, "value": None, "reason": "Original evaluator supplies no net-path metric"},
        "longest_continuous_forward_bout_s": window.get("longest_forward_bout_s"),
        "longest_eligible_bout_s": max(durations, default=0.) if durations is not None else None,
        "complete_cycles": {leg: events.get(leg, {}).get("complete_cycles") for leg in runner.LEGS},
        "health": copy.deepcopy(short.get("health", {}))}


def matched_contrast(neutral, candidate):
    def finite_real(value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            return None
        try:
            plain = float(value)
        except (OverflowError, ValueError):
            return None
        return plain if np.isfinite(plain) else None

    def integer(value):
        return int(value) if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) else None

    result = {"scope": "Closed-loop histories including different .5s prefixes; not instantaneous causal isolation"}
    for key in ("forward_mm", "slip_mm_s", "longest_continuous_forward_bout_s", "longest_eligible_bout_s"):
        a, b = finite_real(neutral.get(key)), finite_real(candidate.get(key))
        delta = b - a if a is not None and b is not None else None
        result[key + "_candidate_minus_neutral"] = delta if delta is not None and np.isfinite(delta) else None
    cycles = {}
    for leg in runner.LEGS:
        a, b = integer(neutral.get("complete_cycles", {}).get(leg)), integer(candidate.get("complete_cycles", {}).get(leg))
        cycles[leg] = b - a if a is not None and b is not None else None
    result["complete_cycles_candidate_minus_neutral"] = cycles
    bouts = result["longest_eligible_bout_s_candidate_minus_neutral"]
    cycles = list(result["complete_cycles_candidate_minus_neutral"].values())
    result["matched_bout_and_cycles_improved"] = bool(bouts is not None and bouts > 0 and all(v is not None and v > 0 for v in cycles))
    return result


class NativeDependencies(e306.NativeDependencies):
    def load(self, freeze):
        from flylab.c.storage import StateStore
        data = super().load(freeze)
        reduced = e309.reduced_selection(data["selection"])
        arms = {}
        for arm, paths in CONTINUATION_INPUTS.items():
            saved = StateStore.load(safe(paths["checkpoint"]))
            patch = self.wrapper.checked_checkpoint(saved)
            expected = copy.deepcopy(reduced if arm == "candidate" else data["selection"])
            if arm == "neutral":
                expected["utilization"] = data["roster"]["neutral_utilization"].copy()
            runner.layer.require_equal(patch["selection"], expected, "Exact frozen continuation selection bytes")
            identity = freeze["neutral_network_identity" if arm == "neutral" else "reduced_candidate_network_identity"]
            if (saved["control_tick"] != 100 or saved["network"]["tick"] != 2000 or saved["network"]["model_hash"] != identity
                    or saved["graph_hash"] != freeze["graph_hash"] or saved["adapter"]["hash"] != freeze["adapter_identity"]):
                raise ValueError("Exact fixed nonzero producer whole clocks and identities required")
            prefix = {key: json.loads(safe(path).read_text("utf8")) if key == "trace" else runner.protocol.load_npz(safe(path))
                      for key, path in paths.items() if key != "checkpoint"}
            validate_global_clocks(prefix["trace"]["samples"], prefix["native"])
            if len(prefix["trace"]["samples"]) != 101 or len(prefix["native"]["time_s"]) != 5000:
                raise ValueError("Complete .5s producer prefix required")
            for key in ("activity", "recruitment", "controls"):
                p = prefix[key]
                runner.layer.require_equal(p["control_tick"], np.arange(101, dtype=np.int64), "Exact saved prefix controls")
                runner.layer.require_equal(p["network_tick"], np.arange(101, dtype=np.int64) * 20, "Exact saved prefix neural ticks")
                if not np.allclose(p["physics_time_s"], np.arange(101) * .005, rtol=0, atol=2e-8):
                    raise ValueError("Exact saved prefix physical clocks required")
                if key != "controls" and not np.all(p["complete"]):
                    raise ValueError("Every producer endpoint complete required")
            roster = copy.deepcopy(data["roster"])
            roster["utilization"] = expected["utilization"].copy()
            arms[arm] = {**data, "arm": arm, "saved": saved, "selection": expected, "roster": roster, "prefix": prefix, "identity": identity}
        data["arms"] = arms
        return data

    def construct(self, arm, data, ledger, persist, on_return):
        def callback():
            session = self.wrapper.SSPRecoveryResourceTrial.from_checkpoint(data["graph"], copy.deepcopy(data["saved"]))
            session._e291_arm = arm
            return session
        return self._construction(arm, callback, ledger, persist, on_return)

    def initial(self, session, data, arm):
        observed = session.snapshot()
        receipt = self.wrapper.require_saved_whole_equal(observed, data["saved"], "Entire nonzero saved whole restoration including wall")
        if (session.control_tick != 100 or session.network.tick != 2000 or abs(session.body.physics_time() - .5) > 2e-8
                or session.network.identity != data["identity"] or session.adapter.identity != data["freeze"]["adapter_identity"]
                or session.network.device != "cuda" or session.network.cuda_implementation != "packed" or session.network.interval_steps != 20):
            raise ValueError("Actual nonzero CUDA restored clocks/identities required")
        runner.layer.require_equal(observed["network"]["synaptic_resource"], data["saved"]["network"]["synaptic_resource"], "Exact saved nonzero resource")
        receipt.update(whole_restoration_exact=True, original_wall_preserved=True, observed_wall_s=observed["wall_s"],
                       stored_wall_s=data["saved"]["wall_s"], prefix_control=100, prefix_neural=2000, prefix_native=5000,
                       prefix_physics_s=.5, future_replay_exact_claim=False)
        return receipt

    def capture(self, session, raw):
        return e301.NativeDependencies.capture(self, session, raw)

    def initial_readout(self, raw, data):
        return compare_initial_readout(raw, data["prefix"])

    def combined(self, raw, data):
        return combine_raw(data["prefix"], raw)

    def save_endpoints(self, folder, raw, data):
        rows = [row for row in raw["activity"] if row.get("complete") is True]
        if not rows:
            raise ValueError("No completed continuation resource endpoint available")
        columns = np.searchsorted(raw["ids"], data["source_ids"])
        if not np.array_equal(raw["ids"][columns], data["source_ids"]):
            raise ValueError("Actual source activity columns required")
        controls = np.asarray([r["control_tick"] for r in rows], np.int64)
        runner.layer.require_equal(controls, np.arange(100, 100 + len(rows), dtype=np.int64), "Contiguous continuation endpoints")
        u, tau = data["selection"]["utilization"], data["selection"]["recovery_tau"]
        factors, resources = [], []
        for row in rows:
            runner.validate_activity(row, len(raw["ids"]), len(data["source_ids"]))
            if row["network_tick"] != row["control_tick"] * 20 or abs(row["physics_time_s"] - row["control_tick"] * .005) > 2e-8:
                raise ValueError("Actual global resource endpoint clocks required")
            r, rate = row["synaptic_resource"], row["rate"][columns]
            g = np.minimum(1., r * (1 + (tau * u) * rate.astype(np.float64))).astype(np.float32)
            if not np.isfinite(g).all() or np.any(g < r.astype(np.float32)) or np.any(g > 1) or not np.all(r[u == 0] == 1) or not np.all(g[u == 0] == 1):
                raise ValueError("Finite endpoint resource/g and unchanged exemption required")
            resources.append(r)
            factors.append(g)
        resources, factors = np.stack(resources), np.stack(factors)
        summary = {"minimum_resource": float(resources.min()), "minimum_endpoint_factor": float(factors.min()),
            "active_sources": int(np.count_nonzero(u)), "factor_nonunity_samples": int(np.count_nonzero(factors < 1)),
            "scope": "Actual completed continuation endpoints including restored endpoint100; no internal stage claim"}
        raw["resource_summary"] = summary
        np.savez_compressed(folder / "recovery-factor-endpoints.npz", source_ids=data["source_ids"], control_tick=controls,
            network_tick=controls * 20, physics_time_s=np.asarray([r["physics_time_s"] for r in rows]), g32=factors)
        self.write(folder / "recovery-factor-endpoints.json", summary)

    def save_raw(self, folder, raw, data, clock):
        failures = runner.NativeDependencies.save_raw(self, folder, raw, data, clock)
        for name, callback in (("recovery-factor-endpoints.npz", lambda: self.save_endpoints(folder, raw, data)),
                              ("resource-faults.json", lambda: self.write(folder / "resource-faults.json", {"events": self.resource_faults})),
                              ("continuation-provenance.json", lambda: self.write(folder / "continuation-provenance.json",
                                {"schema": "flylab.m3.e310.continuation-provenance.v1", "source": CONTINUATION_INPUTS[data["arm"]],
                                 "original_startup_pass": ORIGINAL_STARTUP_PASS[data["arm"]], "new_controls": "100..300 endpoints; 101..300 executed",
                                 "prefix_runtime_replay": False, "cost_clock": {"neural_s": clock.neural_s, "physical_s": clock.physical_s, "valid": clock.clock_valid}}))):
            try:
                callback()
            except Exception as error:
                failures.append({"artifact": name, "error": repr(error)})
        return failures

    def save_combined(self, folder, combined, raw_folder, data):
        failures = []
        folder.mkdir(exist_ok=False)
        def attempt(name, callback):
            try:
                callback()
            except Exception as error:
                failures.append({"artifact": name, "error": repr(error)})
        attempt("trace.json", lambda: self.write(folder / "trace.json", {**combined["trace_metadata"], "samples": combined["rows"],
            "schema": "flylab.m3.e310.combined-evaluation-trace.v1", "E310_provenance": {"frozen_prefix_controls": "0..99",
                "actual_continuation_controls": "100..300", "source": CONTINUATION_INPUTS[data["arm"]], "evaluation_only": True, "original_heading_control": 0}}))
        attempt("native-contact.npz", lambda: np.savez_compressed(folder / "native-contact.npz", **combined["native_arrays"]))
        for key, filename in (("activity", "activity.npz"), ("recruitment", "native-recruitment-state.npz"), ("controls", "control-raw.npz")):
            def save(key=key, filename=filename):
                new, original = runner.protocol.load_npz(raw_folder / filename), data["prefix"][key]
                if new.keys() != original.keys():
                    raise ValueError("All combined observation channels unchanged required")
                joined = {}
                for name, value in new.items():
                    before = original[name]
                    if before.dtype != value.dtype or before.shape[1:] != value.shape[1:]:
                        raise ValueError("Combined observation channel dtype/shape required")
                    if name in ("schema", "observed_ids", "motor_ids", "source_ids"):
                        runner.layer.require_equal(value, before, "Combined static roster/schema")
                        joined[name] = value
                    else:
                        if len(before) != 101 or len(value) != len(combined["rows"]) - 100:
                            raise ValueError("Combined endpoint count required")
                        runner.layer.require_equal(value[0], before[-1], "Shared boundary endpoint " + name)
                        joined[name] = np.concatenate((before[:100], value), axis=0)
                np.savez_compressed(folder / filename, **joined)
            attempt(filename, save)
        return failures

    def short_metrics(self, combined, folder, clock, data):
        rows, arrays = combined["rows"], combined["native_arrays"]
        validate_global_clocks(rows, arrays, complete=True)
        health = runner.short.short_health(rows, arrays, clock, bool(rows[-1].get("fault")))
        result = {"health": health, "forward": runner.short.short_forward(rows, complete=True),
                  "slip": runner.short.short_window_slip(arrays, complete=True, minimum_load=.02)}
        result["gait"], _, _ = runner.gait.analyze(runner.gait.load_model(folder / "trace.json"))
        return result

    def write(self, path, value):
        if path.name == "source-manifest.json":
            value = {**value, "schema": "flylab.m3.e310.sources.v1", "diagnostic_id": "E310",
                "owned_sha256": {name: runner.protocol.sha256(ROOT / name) for name in OWNED},
                "source_sha256": {**value.get("source_sha256", {}), **{name: runner.protocol.sha256(ROOT / name) for name in SOURCE_FILES if name not in OWNED}},
                "executed_body_runner": "tools.verify_c_banc_m3_resource_continuation.execute",
                "executed_arm_runner": "tools.verify_c_banc_m3_resource_continuation.run_arm", "prefix_runtime_replay": False}
        return runner.NativeDependencies.write(self, path, value)


def run_arm(arm, data, freeze, out, ledger, deps, budget):
    folder = out / arm
    folder.mkdir()
    raw_folder, combined_folder = folder / "continuation", folder / "combined"
    raw_folder.mkdir()
    raw = {key: [] for key in ("rows", "controls", "recruitment", "activity", "native")}
    raw["ids"] = data["ids"]
    clock, holder = runner.native.AdvanceLedger(), {}
    session = detach = journal = None
    primary, measured_stop, errors = None, None, []
    entry = {"status": "requested", "completed_controls": 0, "final_control": None, "readout_endpoints": 0,
             "construction_requested": False, "construction_completed": False, "owned_session_recovered": False,
             "close_requested": False, "close_returned": False, "prefix_controls_not_charged": 100}
    ledger["arms"][arm] = entry
    metrics = {"original_startup_pass": ORIGINAL_STARTUP_PASS[arm], "startup_retested": False, "short": {"available": False},
               "diagnostics_only": True, "complete": False}
    construction_start, body_start = len(ledger["construction"]), len(ledger["body_construction"])
    def persist():
        deps.write(out / "advance-ledger.json", ledger)
    def fail(stage, error):
        nonlocal primary
        errors.append({"stage": stage, "type": type(error).__name__, "message": str(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E310 " + stage + ": " + repr(error))
    try:
        budget()
        entry["construction_requested"] = True
        persist()
        session = deps.construct(arm, data, ledger, persist, lambda value: holder.setdefault("session", value))
        entry["construction_completed"] = True
        if (any(row.get("unknown_setup_steps") or row.get("unknown_capture_setup_steps")
                for row in ledger["construction"][construction_start:])
                or any(row.get("unknown_step_entries") or row.get("active_unreturned_frames")
                       for row in ledger["body_construction"][body_start:])):
            raise RuntimeError("Current arm constructor/setup/settling costs unavailable")
        if hasattr(deps, "trace_metadata"):
            raw["trace_metadata"] = deps.trace_metadata(session)
        # Preserve actual observed state even when the ensuing equality fails.
        deps.save_state(folder / "observed-start", session)
        entry["initial_comparison"] = deps.initial(session, data, arm)
        entry["restored_clocks"] = {"control": session.control_tick, "neural": session.network.tick, "physics": session.body.physics_time()}
        detach = deps.install(session, raw["native"])
        journal = deps.journal(folder / "control-journal.jsonl")
        deps.capture(session, raw)
        entry["initial_readout_comparison"] = deps.initial_readout(raw, data)
        entry["readout_endpoints"] = 1
        entry["final_control"] = 100
        persist()
        for control in range(101, 301):
            budget()
            before_native = len(raw["native"])
            try:
                event = deps.advance(session, arm, control, clock, journal)
            except deps.health_stop as error:
                measured_stop = {"control": control, "message": str(error), "native_samples": len(raw["native"])}
                break
            entry["completed_controls"] += 1
            entry["final_control"] = control
            try:
                deps.capture(session, raw)
                deps.readout_receipt(journal, arm, control, event, True)
            except Exception as error:
                try:
                    deps.readout_receipt(journal, arm, control, event, False, error)
                except Exception as secondary:
                    error.add_note("Secondary E310 readout journal: " + repr(secondary))
                raise
            entry["readout_endpoints"] += 1
            if (session.control_tick != control or session.network.tick != control * 20
                    or abs(session.body.physics_time() - control * .005) > 2e-8 or len(raw["native"]) != before_native + 50):
                raise ValueError("Incomplete actual continuation control/neural/native interval")
            if control == 101:
                entry["first_new_interval"] = {"control": 101, "neural": 2020, "physics_s": session.body.physics_time(),
                    "new_neural_ticks": 20, "new_native_samples": 50, "normal_decoder_held_readout_completed": True,
                    "future_replay_exact_claim": False, "clock_event": copy.deepcopy(event)}
            persist()
            if session.fault or session.body.fault:
                measured_stop = {"control": control, "message": "Measured session/body fault"}
                break
        entry["final_clocks"] = {"control": session.control_tick, "neural": session.network.tick, "physics": session.body.physics_time()}
        entry["status"] = "measured_native_health_stop" if measured_stop is not None else "science_complete"
    except Exception as error:
        fail("science", error)
        entry["traceback"] = traceback.format_exc()
    finally:
        if session is None and "session" in holder:
            session = holder["session"]
            entry["owned_session_recovered"] = True
        entry.update(native_samples=len(raw["native"]), health_stop=measured_stop, actual_neural_s=clock.neural_s,
                     actual_physical_s=clock.physical_s, clock_events=clock.events, clock_valid=clock.clock_valid)
        for label, callback in (("native detach", detach), ("journal close", journal.close if journal else None)):
            if callback:
                try:
                    callback()
                except Exception as error:
                    fail(label, error)
        try:
            for failure in deps.save_raw(raw_folder, raw, data, clock):
                fail("raw " + failure["artifact"], RuntimeError(failure["error"]))
        except Exception as error:
            fail("raw publication", error)
        if session is not None:
            try:
                deps.save_state(folder / "final", session)
            except Exception as error:
                if not (measured_stop is not None and isinstance(error, ValueError)):
                    fail("whole final state", error)
                entry["whole_final_unavailable"] = repr(error)
                try:
                    deps.save_partial(folder / "partial-final", session)
                except Exception as partial:
                    fail("partial state", partial)
            entry["close_requested"] = True
            entry["close_started_wall_s"] = time.perf_counter()
            try:
                session.close()
                entry["close_returned"] = True
            except Exception as error:
                fail("session close", error)
            entry["close_finished_wall_s"] = time.perf_counter()
        if raw["rows"] and entry.get("initial_readout_comparison"):
            try:
                combined = deps.combined(raw, data)
                for failure in deps.save_combined(combined_folder, combined, raw_folder, data):
                    fail("combined " + failure["artifact"], RuntimeError(failure["error"]))
                if primary is None and measured_stop is None and entry["completed_controls"] == 200:
                    ec = evaluation_clock(clock, restored=entry["restored_clocks"], final=entry["final_clocks"], prefix_verified=True)
                    entry["evaluation_clock"] = {"neural_s": ec.neural_s, "physical_s": ec.physical_s, "valid": ec.clock_valid, "origin": ec.origin}
                    metrics["short"] = deps.short_metrics(combined, combined_folder, ec, data)
                    metrics["absolute_short_adjudication"] = runner.assess_short(data["baseline_short"], metrics["short"])
                    metrics["summary"] = metric_summary(metrics["short"])
                    metrics["complete"] = metrics["short"].get("health", {}).get("healthy") is True
            except Exception as error:
                fail("combined/evaluation", error)
        metrics["resource_observation"] = copy.deepcopy(raw.get("resource_summary", {}))
        entry["errors"] = errors
        if primary is not None:
            entry["status"] = "technical_or_resource_failure"
            metrics["first_error"] = {"type": type(primary).__name__, "message": str(primary)}
            metrics["complete"] = False
        metrics.update(validity="valid_complete_continuation" if primary is None and metrics["complete"] else "invalid_or_incomplete",
            status=entry["status"], completed_controls=entry["completed_controls"], final_control=entry["final_control"], health_stop=measured_stop,
            interpretation_validated=primary is None and metrics["complete"], technical_valid=primary is None)
        try:
            persist()
            deps.write(folder / "arm-result.json", metrics)
        except Exception as error:
            fail("arm publication", error)
            metrics.update(validity="invalid_or_incomplete", status="technical_or_resource_failure", interpretation_validated=False, complete=False, technical_valid=False)
    return metrics, primary


def execute(freeze, out, *, deps=None, budget=None, cleanup=None):
    if freeze.get("diagnostic_id") != "E310":
        raise ValueError("Explicit E310 candidate diagnostic required")
    deps = NativeDependencies() if deps is None else deps
    began = time.perf_counter()
    out.mkdir(exist_ok=False)
    ledger = {
        "schema": "flylab.m3.e310.ledger.v1",
        "arms": {},
        "construction": [],
        "body_construction": [],
        "original_attempts": 1,
        "retries": 0,
        "fit_calls": 0,
        "nonneutral_current_calls": 0,
    }
    result = {"schema": RESULT_SCHEMA, "validity": "incomplete", "publication_status": "pending", "diagnostic_id": "E310",
        "run_valid": False, "candidate_adopted": False, "adopted": False, "next_separate_10s_eligible": False,
        "natural_gait_complete": False, "F1_F8_complete": False, "F2_complete": False, "F3_F8_pass": False, "F6_exposed": False,
        "ten_second_executed": False, "E272_executed": False, "biological_claim": False, "causal_claim": False,
        "actual_startup_pass": False, "startup_retested": False, "original_startup_pass": copy.deepcopy(ORIGINAL_STARTUP_PASS),
        "diagnostics_only": True, "F2_future_disk_replay_verified": False, "short_gait_pass": False,
        "arm_receipt_path": "tools.verify_c_banc_m3_resource_continuation.run_arm", "arms": {}}
    errors, primary = [], None

    def check():
        if time.perf_counter() - began >= LIMITS["science_wall_s"]:
            raise TimeoutError("E310 science deadline/reserve reached")
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
                raise MemoryError("E310 host/GPU cap exceeded")
            if shutil.disk_usage(out).free < LIMITS["minimum_free_disk_bytes"]:
                raise OSError("E310 output disk reserve reached")

    try:
        deps.write(
            out / "source-manifest.json",
            {
                "schema": "flylab.m3.e310.sources.v1",
                "input_sha256": freeze.get("input_sha256", {}),
                "source_sha256": freeze.get("reused_source_sha256", {}),
                "owned_sha256": {name: runner.protocol.sha256(ROOT / name) for name in OWNED},
            },
        )
        data = deps.load(freeze)
        neutral, primary = run_arm("neutral", data["arms"]["neutral"], freeze, out, ledger, deps, check)
        result["arms"]["neutral"] = neutral
        result["candidate_executed"] = False
        if primary is None and neutral.get("complete") is True:
            candidate, primary = run_arm("candidate", data["arms"]["candidate"], freeze, out, ledger, deps, check)
            result["arms"]["candidate"] = candidate
            result["candidate_executed"] = True
            if primary is None and candidate.get("complete") is True:
                result["matched_contrast"] = matched_contrast(neutral.get("summary", {}), candidate.get("summary", {}))
                result["interval_resource_benefit_observed"] = candidate["absolute_short_adjudication"]["pass"] and result["matched_contrast"]["matched_bout_and_cycles_improved"]
        result["run_valid"] = primary is None and len(result["arms"]) == 2 and all(a.get("complete") is True for a in result["arms"].values())
        result["validity"] = "valid_fixed_resource_continuation_diagnostic" if result["run_valid"] else "invalid_or_incomplete"
        result["decision"] = "interval_benefit_observed_startup_still_rejected" if result.get("interval_resource_benefit_observed") else (
            "fixed_resource_branch_not_expanded" if result["run_valid"] or result.get("candidate_executed") and result["arms"]["candidate"].get("health_stop") else "interpretation_unavailable")
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
            interval_resource_benefit_observed=False, decision="interpretation_unavailable",
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
            "E310 scientific clocks/caps unavailable or exceeded"
        )
        result.update(
            run_valid=False,
            actual_startup_pass=False,
            short_gait_pass=False,
            next_separate_10s_eligible=False,
            validity="invalid_or_incomplete",
            interval_resource_benefit_observed=False, decision="interpretation_unavailable",
            first_error={"type": type(primary).__name__, "message": str(primary)},
        )
        errors.append({"stage": "cost", "message": str(primary)})
    finalization_began = time.perf_counter()
    try:
        deps.write(out / "advance-ledger.json", ledger)
        deps.write(out / "errors.json", {"errors": errors})
        manifest = runner.short.output_manifest(out, LIMITS["output_bytes"])
        manifest["schema"] = "flylab.m3.e310.output-manifest.v1"
        deps.write(out / "output-manifest.json", manifest)
        if (time.perf_counter() - began >= LIMITS["internal_wall_s"]
                or time.perf_counter() - finalization_began >= LIMITS["finalization_reserve_s"]):
            raise TimeoutError("E310 final publication deadline")
        result["publication_status"] = "complete"
        deps.write(out / "result.json", result)
        deps.write(out / "result-sha256.json", {"schema": "flylab.m3.e310.result-sha256.v1", "path": "result.json", "sha256": runner.protocol.sha256(out / "result.json")})
        if (time.perf_counter() - began >= LIMITS["internal_wall_s"]
                or time.perf_counter() - finalization_began >= LIMITS["finalization_reserve_s"]):
            raise TimeoutError("E310 late result publication deadline")
    except Exception as error:
        primary = primary or error
        result.update(
            run_valid=False,
            actual_startup_pass=False,
            short_gait_pass=False,
            next_separate_10s_eligible=False,
            validity="invalid_or_incomplete",
            interval_resource_benefit_observed=False, decision="interpretation_unavailable",
            first_error={"type": type(primary).__name__, "message": str(primary)},
        )
        for arm in result["arms"].values():
            arm["interpretation_validated"] = False
        result["publication_status"] = "invalid"
        errors.append({"stage": "publication", "type": type(error).__name__, "message": str(error)})
        def secondary(stage, failure):
            errors.append({"stage": stage, "type": type(failure).__name__, "message": str(failure)})
            primary.add_note("Secondary E310 " + stage + ": " + repr(failure))
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
                            ("publication-status.json", {"schema": "flylab.m3.e310.publication-status.v1", "status": "invalid",
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
                failures[0].add_note("Secondary E310 cleanup: " + repr(error))
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
