"""E297 saved observed transfer residual; no reconstruction or causal claim."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from flylab.c.storage import StateStore
from tools import analyze_c_banc_m3_core_resource_localization as prior
from tools import verify_c_banc_m3_core_afferent_resource_body as producer

p = prior.prior
OWNED = ("tools/analyze_c_banc_m3_observed_transfer_residual.py", "tests/test_c_banc_observed_transfer_residual.py")
SOURCE_FILES = tuple(dict.fromkeys((*OWNED, *prior.SOURCE_FILES, *producer.SOURCE_FILES)))
INPUT_NAMES = ("protected_roster", "candidate_activity", "candidate_native", "candidate_recruitment", "candidate_trace",
               "candidate_final_checkpoint", "e296_result", "e296_ledger", "e296_root_audit", "e296_source_manifest",
               "e296_output_manifest", "e296_freeze", "e296_source_review", "baseline_signals", "baseline_metadata",
               "baseline_recruitment", "baseline_trace", "baseline_core_activity", "baseline_csr",
               "baseline_e150_output_manifest", "baseline_e257_output_manifest", "original_startup")
NPZ_INPUTS = {"protected_roster", "candidate_activity", "candidate_native", "candidate_recruitment", "baseline_signals",
              "baseline_metadata", "baseline_recruitment", "baseline_core_activity", "baseline_csr", "original_startup"}
FREEZE_SCHEMA = "flylab.m3.e297.observed-transfer-residual.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e297.observed-transfer-residual.review.v1"
OUTPUT_PATH = "verification/banc-natural-gait-m3-20260919/e297-observed-transfer-residual"
LIMITS = {"wall_s": 45, "host_peak_bytes": 2 << 30, "output_bytes": 64 << 20, "cpu_threads": 1}
WINDOWS = prior.WINDOWS
FLAGS = {**prior.FLAGS, "all_current_observed": False, "CUDA_exact_current": False}
GROUPS = ("core_source", "other_source", "common_observed")
SIGNS = ("signed", "positive", "negative")


def safe(name):
    value = Path(name)
    path = (ROOT / value).resolve()
    if value.is_absolute() or ".." in value.parts or not path.is_relative_to(ROOT) or path.is_symlink():
        raise ValueError("Repository relative frozen path required")
    return path


def matrix_from_csr(metadata, csr):
    sources, motors = metadata["source_ids"], metadata["motor_ids"]
    prior.id_columns(sources, sources)
    prior.id_columns(motors, motors)
    if np.intersect1d(sources, motors).size:
        raise ValueError("Disjoint source and motor IDs required")
    indptr, indices, weights = (csr[key] for key in ("indptr", "indices", "data"))
    if (not isinstance(indptr, np.ndarray) or indptr.dtype != np.int32 or indptr.ndim != 1 or len(indptr) < 2
            or not isinstance(indices, np.ndarray) or indices.dtype != np.int32 or indices.ndim != 1):
        raise ValueError("Native original CSR topology required")
    p.array(weights, np.float32, indices.shape, "CSR weights")
    n = len(indptr) - 1
    if (indptr[0] != 0 or indptr[-1] != len(indices) or np.any(np.diff(indptr) < 0)
            or np.any((indices < 0) | (indices >= n)) or np.any(sources >= n) or np.any(motors >= n)):
        raise ValueError("CSR bounds or row pointers invalid")
    result = np.zeros((len(motors), len(sources)), np.float32)
    mapping = {int(cell): column for column, cell in enumerate(sources)}
    for row, cell in enumerate(motors):
        cols = indices[indptr[cell]:indptr[cell + 1]]
        values = weights[indptr[cell]:indptr[cell + 1]]
        if len(np.unique(cols)) != len(cols):
            raise ValueError("Duplicate original CSR column cannot be silently combined")
        for col, value in zip(cols, values, strict=True):
            if int(col) in mapping:
                result[row, mapping[int(col)]] = value
    p.array(metadata.get("source_to_motor"), np.float32, result.shape, "Original motor matrix")
    p.equal_array(result, metadata["source_to_motor"], "Byteexact original CSR motor rows/source columns")
    return result


def transfer_terms(matrix, baseline, candidate, inside):
    result = {}
    for group, chosen in (("core_source", inside), ("other_source", ~inside), ("common_observed", np.ones(len(inside), bool))):
        w = matrix[:, chosen].astype(np.float64)
        for sign, coefficient in (("signed", w), ("positive", np.maximum(w, 0)), ("negative", np.minimum(w, 0))):
            key = group + "_" + sign
            result[key + "_baseline"] = baseline[:, chosen].astype(np.float64) @ coefficient.T
            result[key + "_candidate"] = candidate[:, chosen].astype(np.float64) @ coefficient.T
            result[key + "_delta"] = result[key + "_candidate"] - result[key + "_baseline"]
    for phase in ("baseline", "candidate", "delta"):
        for sign in SIGNS:
            np.testing.assert_allclose(result["common_observed_" + sign + "_" + phase],
                result["core_source_" + sign + "_" + phase] + result["other_source_" + sign + "_" + phase], rtol=2e-12, atol=2e-10)
        for group in GROUPS:
            np.testing.assert_allclose(result[group + "_signed_" + phase],
                result[group + "_positive_" + phase] + result[group + "_negative_" + phase], rtol=2e-12, atol=2e-10)
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite descriptive transfer")
    return result


def observed_transfer(metadata, baseline_signals, candidate_activity, core_ids, csr):
    matrix = matrix_from_csr(metadata, csr)
    sources, motors = metadata["source_ids"], metadata["motor_ids"]
    b, a = baseline_signals, candidate_activity
    prior.id_columns(b["observed_ids"], sources)
    prior.id_columns(b["observed_ids"], motors)
    prior.id_columns(a["observed_ids"], motors)
    p.equal_array(metadata["observed_ids"], b["observed_ids"], "Original metadata signal ordering")
    common = np.isin(sources, a["observed_ids"])
    ids, missing = sources[common], sources[~common]
    prior.id_columns(ids, core_ids)
    if not len(ids):
        raise ValueError("No common observed sources")
    controls = np.arange(101, dtype=np.int64)
    p.equal_array(a.get("control_tick"), controls, "Candidate controls0..100")
    p.equal_array(a.get("network_tick"), controls * 20, "Candidate neural clock")
    times = p.array(a.get("physics_time_s"), np.float64, (101,), "Candidate physical clock")
    if np.any(np.abs(times - controls * .005) >= 2e-8) or not p.array(a.get("complete"), bool, (101,), "Complete activity").all():
        raise ValueError("Candidate clock/completeness mismatch")
    for value, rows, label in ((a, 101, "candidate"), (b, len(b["rate"]), "baseline")):
        p.array(value.get("rate"), np.float32, (rows, len(value["observed_ids"])), label + " rate")
    if len(b["rate"]) < 100:
        raise ValueError("Baseline100 rows required")
    bt = p.array(b.get("time_s"), np.float64, (len(b["rate"]),), "Baseline physical clock")
    if np.any(np.abs(bt[:100] - controls[1:] * .005) >= 2e-8):
        raise ValueError("Baseline controls1..100 only")
    baseline = b["rate"][:100, prior.id_columns(b["observed_ids"], ids)]
    candidate = a["rate"][1:, prior.id_columns(a["observed_ids"], ids)]
    return {"common_source_ids": ids.copy(), "missing_source_ids": missing.copy(), "motor_ids": motors.copy(),
            "core_ids": core_ids.copy(), "control_tick": controls[1:], "source_to_motor": matrix[:, common],
            "missing_source_to_motor": matrix[:, ~common], "baseline_rate": baseline.copy(), "candidate_rate": candidate.copy(),
            **transfer_terms(matrix[:, common], baseline, candidate, np.isin(ids, core_ids))}


def endpoint_transfer(metadata, baseline_signals, activity, core_ids, checkpoint):
    network = checkpoint["network"]
    if checkpoint.get("control_tick") != 100 or network.get("tick") != 2000 or network.get("dt") != .00025:
        raise ValueError("Final saved control100/neural2000 required")
    p.clock_close(checkpoint["body"]["state"][0] - checkpoint["body"]["t0"], .5, "Final physical endpoint")
    rate = network["rate"]
    p.array(rate, np.float32, (158706,), "Full final saved rate")
    p.equal_array(rate[activity["observed_ids"]], activity["rate"][100], "Final full rate/observed endpoint bytes")
    sources, motors = metadata["source_ids"], metadata["motor_ids"]
    prior.id_columns(activity["observed_ids"], motors)
    b = baseline_signals["rate"][99:100, prior.id_columns(baseline_signals["observed_ids"], sources)]
    c = rate[sources][None, :]
    missing = ~np.isin(sources, activity["observed_ids"])
    extra = {"missing_source_ids": sources[missing].copy(), "missing_baseline_rate": b[:, missing].copy(),
             "missing_candidate_rate": c[:, missing].copy(), "missing_source_to_motor": metadata["source_to_motor"][:, missing].copy()}
    for sign, coefficient in (("signed", metadata["source_to_motor"][:, missing]),
                              ("positive", np.maximum(metadata["source_to_motor"][:, missing], 0)),
                              ("negative", np.minimum(metadata["source_to_motor"][:, missing], 0))):
        extra["missing_" + sign + "_baseline"] = b[:, missing].astype(np.float64) @ coefficient.astype(np.float64).T
        extra["missing_" + sign + "_candidate"] = c[:, missing].astype(np.float64) @ coefficient.astype(np.float64).T
        extra["missing_" + sign + "_delta"] = extra["missing_" + sign + "_candidate"] - extra["missing_" + sign + "_baseline"]
    return {"control_tick": np.array([100], np.int64), "source_ids": sources.copy(), "motor_ids": motors.copy(),
            "baseline_rate": b.copy(), "candidate_rate": c.copy(), **extra,
            **transfer_terms(metadata["source_to_motor"], b, c, np.isin(sources, core_ids))}


def startup_error_decomposition(baseline_q, candidate_q, original_targets):
    shape = original_targets.shape
    if len(shape) != 2 or not shape[0] or not shape[1]:
        raise ValueError("Nonempty startup matrix required")
    for value in (baseline_q, candidate_q, original_targets):
        p.array(value, np.float64, shape, "Aligned startup commands")
    be, delta, ce = baseline_q - original_targets, candidate_q - baseline_q, candidate_q - original_targets
    cross = 2 * be * delta
    columns = []
    for col in range(shape[1]):
        terms = {key: float(np.sum(value[:, col] ** 2)) for key, value in (("baseline_SSE", be), ("delta_SSE", delta), ("candidate_SSE", ce))}
        terms["cross_sum"] = float(np.sum(cross[:, col]))
        np.testing.assert_allclose(terms["candidate_SSE"], terms["baseline_SSE"] + terms["delta_SSE"] + terms["cross_sum"], rtol=2e-12, atol=2e-12)
        columns.append({"column": col, **terms, **{key: {"RMS": float(np.sqrt(np.mean(value[:, col] ** 2))),
            "max_abs": float(np.max(np.abs(value[:, col]))), "peak_control": int(np.argmax(np.abs(value[:, col]))) + 1}
            for key, value in (("baseline_error", be), ("candidate_delta", delta), ("candidate_error", ce))}})
    return {"arrays": {"baseline_error": be, "candidate_delta": delta, "candidate_error": ce, "cross_term": cross},
            "summary": {"controls": [1, shape[0]], "all_columns": columns,
                **{key: {"RMS": float(np.sqrt(np.mean(value ** 2))), "max_abs": float(np.max(np.abs(value)))}
                   for key, value in (("baseline_error", be), ("candidate_error", ce))}}}


def pin_bytes(groups):
    pins = {}
    for group in groups:
        for name, digest in group.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting SHA")
            pins[name] = digest
    for name, digest in pins.items():
        if not isinstance(digest, str) or len(digest) != 64 or p.sha256(safe(name)) != digest:
            raise ValueError("Frozen E297 bytes changed: " + name)


def producer_files(inputs, data, pins, manifest_alias, aliases):
    parent = safe(inputs[manifest_alias]).parent
    entries = data[manifest_alias]["files"]
    for alias in aliases:
        path = safe(inputs[alias])
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for file in files:
            if not file.is_file():
                continue
            name = file.relative_to(ROOT).as_posix()
            if not file.is_relative_to(parent):
                raise ValueError("Raw outside producer")
            entry = entries.get(file.relative_to(parent).as_posix())
            if not isinstance(entry, dict) or entry.get("sha256") != pins.get(name) or entry.get("bytes") != file.stat().st_size:
                raise ValueError("Producer SHA/bytes mismatch: " + alias)


def result_receipt(inputs, data, pins):
    """E291 publishes its result after the output manifest; bind it separately."""
    frozen = data["e296_freeze"]
    name = inputs["e296_result"]
    if (frozen.get("output") != producer.OUTPUT_PATH or name != frozen["output"] + "/result.json"
            or p.sha256(safe(name)) != pins.get(name)):
        raise ValueError("Literal frozen E296 result path/SHA required")
    result, audit = data["e296_result"], data["e296_root_audit"]
    if (result.get("schema") != "flylab.m3.e291.result.v1" or result.get("diagnostic_id") != "E296"
            or result.get("cost") != audit.get("cost") or not isinstance(result.get("cost"), dict)):
        raise ValueError("Actual E296 result schema/identity/audited cost required")
    flags = ("run_valid", "actual_startup_pass", "short_gait_pass", "candidate_adopted", "F1_F8_complete", "F6_exposed")
    for flag in flags:
        expected = flag == "run_valid"
        if result.get(flag) is not expected or audit.get(flag) is not expected:
            raise ValueError("Result/audit rejection flags differ")
    candidate, audited = result["arms"]["candidate"], audit["arms"]["candidate"]
    startup = candidate["startup"]
    if (candidate.get("completed_controls") != 100 or candidate.get("health_stop") is not None
            or candidate.get("status") != "strict_startup_failed" or audited.get("controls") != 100
            or audited.get("strict_startup") is not False or startup.get("strict_E141_startup_screen") is not False
            or startup.get("observed_controls") != 99 or startup.get("complete_original99_prefix") is not True
            or startup.get("limits") != producer.STARTUP_LIMITS):
        raise ValueError("Result/audit original startup99 rejection differs")
    for group in ("q", "LTM"):
        if startup[group].get("available") is not True:
            raise ValueError("Available original startup summary required")
        for statistic in ("RMS", "max_abs"):
            actual, reference = startup[group][statistic], audited["startup"][group][statistic]
            if (type(actual) not in (int, float) or type(reference) not in (int, float)
                    or not np.isfinite(actual) or not np.isfinite(reference) or actual < 0 or reference < 0
                    or not np.isclose(actual, reference, rtol=2e-12, atol=2e-15)):
                raise ValueError("Result/audit startup statistic differs")


def provenance(inputs, data, pins):
    for manifest, aliases in (("baseline_e150_output_manifest", ("baseline_signals", "baseline_metadata", "baseline_recruitment", "baseline_trace")),
                             ("baseline_e257_output_manifest", ("baseline_core_activity", "baseline_csr")),
                             ("e296_output_manifest", ("candidate_activity", "candidate_native", "candidate_recruitment", "candidate_trace", "candidate_final_checkpoint", "e296_ledger", "e296_source_manifest"))):
        producer_files(inputs, data, pins, manifest, aliases)
    result_receipt(inputs, data, pins)
    f, r, s = (data[key] for key in ("e296_freeze", "e296_source_review", "e296_source_manifest"))
    if (f.get("schema") != producer.FREEZE_SCHEMA or f.get("diagnostic_id") != "E296" or f.get("owned_files") != list(producer.OWNED)
            or r.get("schema") != producer.REVIEW_SCHEMA or r.get("approved") is not True or r.get("blocking_defects") != []
            or r.get("freeze_sha256") != pins[inputs["e296_freeze"]] or set(r.get("source_sha256", {})) != set(producer.OWNED)
            or s.get("diagnostic_id") != "E296" or s.get("owned_sha256") != r["source_sha256"]
            or s.get("input_sha256") != f.get("input_sha256")):
        raise ValueError("Actual E296 source-review/execution lineage required")
    pin_bytes([s["owned_sha256"], s["source_sha256"], f["reused_source_sha256"], f["preservation_sha256"]])
    if not (set(producer.SOURCE_FILES) - set(producer.OWNED)) <= set(s["source_sha256"]):
        raise ValueError("Complete E296 inherited execution source pins required")
    for alias, old in (("protected_roster", "protected_selection"), ("original_startup", "startup"), ("baseline_core_activity", "core_reference")):
        name = inputs[alias]
        if f["paths"][old] != name or f["input_sha256"].get(name) != pins[name]:
            raise ValueError("Original E296 frozen definition differs: " + alias)
    for key in ("e296_result", "e296_root_audit"):
        receipt = data[key]
        if receipt.get("run_valid") is not True or any(receipt.get(flag) is not False for flag in ("actual_startup_pass", "short_gait_pass", "candidate_adopted", "F1_F8_complete", "F6_exposed")):
            raise ValueError("Validly rejected E296 original required")
    audit = data["e296_root_audit"]
    if audit.get("schema") != "flylab.m3.e296.root-original-raw-audit.v1" or audit.get("pass") is not True:
        raise ValueError("Passed actual E296 root raw audit required")
    arm = audit["arms"]["candidate"]
    ledger = data["e296_ledger"]["arms"]["candidate"]
    if (arm.get("controls") != 100 or arm.get("native_samples") != 5000 or arm.get("actual_neural_ticks") != 2000
            or arm.get("whole_final_available") is not True or arm.get("core_afferent_resource_exactly_one") is not True
            or ledger.get("completed_controls") != 100 or ledger.get("readout_endpoints") != 101 or ledger.get("native_samples") != 5000
            or ledger.get("close_returned") is not True):
        raise ValueError("Actual healthy100 control/final checkpoint required")
    p.clock_close(ledger["actual_neural_s"], .5, "Neural ledger")
    p.clock_close(ledger["actual_physical_s"], .5, "Physical ledger")


def load(inputs):
    return {key: StateStore.load(safe(name)) if key == "candidate_final_checkpoint" else
            p.npz(safe(name)) if key in NPZ_INPUTS else p.read_json(safe(name)) for key, name in inputs.items()}


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if p.sha256(freeze_path) != freeze_sha or p.sha256(review_path) != review_sha:
        raise ValueError("E297 freeze/review SHA mismatch")
    freeze, review = p.read_json(freeze_path), p.read_json(review_path)
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("owned_files") != list(OWNED) or set(freeze.get("inputs", {})) != set(INPUT_NAMES)
            or freeze.get("resource_limits") != LIMITS or freeze.get("windows") != [list(w) for w in WINDOWS] or freeze.get("output") != OUTPUT_PATH):
        raise ValueError("Exact E297 freeze required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved matching E297 source review required")
    names = set()
    for name in freeze["inputs"].values():
        path = safe(name)
        files = [file for file in path.rglob("*") if file.is_file()] if path.is_dir() else [path]
        if not files:
            raise ValueError("Missing frozen input")
        names.update(file.relative_to(ROOT).as_posix() for file in files)
    if set(freeze["input_sha256"]) != names or set(freeze["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("Exact all-input/source pins required")
    pin_bytes([freeze["input_sha256"], freeze["source_sha256"], review["source_sha256"], {freeze["design"]: freeze["design_sha256"]}])
    # Only receipts are needed before the explicitly approved saved analysis.
    receipts = {key: p.read_json(safe(name)) for key, name in freeze["inputs"].items() if key not in NPZ_INPUTS and key != "candidate_final_checkpoint"}
    provenance(freeze["inputs"], receipts, freeze["input_sha256"])
    if output.resolve() != safe(OUTPUT_PATH) or output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh fixed E297 output required")
    return freeze


def analyze(data):
    # Explicit inherited input adapter, no E294 provenance impersonation.
    matched = prior.aligned_signals({**data, "positive_roster": data["protected_roster"]})
    transfer = observed_transfer(data["baseline_metadata"], data["baseline_signals"], data["candidate_activity"], matched["core_ids"], data["baseline_csr"])
    if (len(transfer["motor_ids"]) != 391 or len(data["baseline_metadata"]["source_ids"]) != 55
            or len(transfer["common_source_ids"]) != 53 or len(transfer["core_ids"]) != 24
            or set(transfer["missing_source_ids"].tolist()) != {51845, 136232}):
        raise ValueError("Original391/55/common53/core24/missing2 coverage required")
    startup = data["original_startup"]
    p.array(startup.get("target_command"), np.float64, (99, 42), "Original startup99 targets")
    mask = p.array(startup.get("mask"), bool, (2001,), "Original startup mask")
    p.equal_array(np.flatnonzero(mask), np.arange(1, 100), "Original startup controls1..99")
    target_times = p.array(startup.get("time_s"), np.float64, (2001,), "Original startup clock")
    if np.any(np.abs(target_times[mask] - np.arange(1, 100) * .005) >= 2e-8):
        raise ValueError("Original startup time definition differs")
    errors = startup_error_decomposition(matched["q42_baseline"][:99], matched["q42_candidate"][:99], startup["target_command"])
    endpoint = endpoint_transfer(data["baseline_metadata"], data["baseline_signals"], data["candidate_activity"], matched["core_ids"], data["candidate_final_checkpoint"])
    summaries = {}
    for key in list(matched):
        if key.endswith("_baseline"):
            stem = key.removesuffix("_baseline")
            ids = matched["core_ids"] if stem.startswith("core_") else matched["motor_ids"] if stem.startswith("motor_") else np.arange(matched[key].shape[1])
            summaries[stem], matched[stem + "_delta"] = prior.first_and_windows(matched[key], matched[stem + "_candidate"], matched["control_tick"], ids)
    summaries["observed_source_rate"], transfer["rate_delta"] = prior.first_and_windows(transfer["baseline_rate"], transfer["candidate_rate"], transfer["control_tick"], transfer["common_source_ids"])
    for group in GROUPS:
        for sign in SIGNS:
            key = group + "_" + sign
            summaries[key], _ = prior.first_and_windows(transfer[key + "_baseline"], transfer[key + "_candidate"], transfer["control_tick"], transfer["motor_ids"])
    return matched, transfer, endpoint, errors, summaries


def run(freeze, output):
    import psutil
    from threadpoolctl import threadpool_limits
    began = time.perf_counter()
    output = Path(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh E297 output required")
    def budget():
        memory = psutil.Process().memory_info()
        if time.perf_counter() - began >= LIMITS["wall_s"]:
            raise TimeoutError("E297 saved analysis deadline")
        if getattr(memory, "peak_wset", memory.rss) > LIMITS["host_peak_bytes"]:
            raise MemoryError("E297 saved analysis host cap")
    output.mkdir()
    try:
        p.write_json(output / "source-manifest.json", {"schema": "flylab.m3.e297.sources.v1", "owned_files": list(OWNED),
            "source_sha256": freeze.get("source_sha256", {}), "input_sha256": freeze.get("input_sha256", {})})
        with threadpool_limits(limits=1):
            budget()
            data = load(freeze["inputs"])
            provenance(freeze["inputs"], data, freeze["input_sha256"])
            native = data["candidate_native"]
            times = p.array(native["time_s"], np.float64, (5000,), "Native clock")
            if np.any(np.abs(times - np.arange(1, 5001) * .0001) >= 2e-8) or np.any(p.array(native["upright"], np.float64, (5000,), "Native upright") < .9) or p.array(native["nonfoot_contact"], bool, (5000,), "Native nonfoot").any():
                raise ValueError("Actual healthy5000 native samples required")
            matched, transfer, endpoint, errors, summaries = analyze(data)
            result = {"schema": "flylab.m3.e297.observed-transfer-residual.result.v1", "analysis_valid": True,
                "summaries": summaries, "startup99": errors["summary"], "units": "model input units; not amperes",
                "trajectory_scope": "common53=core24+other29; missing2 unobserved before control100; no imputation",
                "endpoint_scope": "full55=core24+other31 at control100 only; not full network current",
                "missing_sources": [{"id": int(cell), "nonzero_motor_coefficients": int(np.count_nonzero(transfer["missing_source_to_motor"][:, col])),
                    "trajectory_observed": False} for col, cell in enumerate(transfer["missing_source_ids"])],
                "first_difference_scope": "5ms saved-grid observation; not substep onset or causal precedence",
                "joint_labels": "original42 column indices", "tendon_column_order": [leg + "_tarsus" for leg in prior.LEGS],
                "unobserved": ["metadata55-external afferents", "missing2 intermediate candidate rates", "unrecorded baseline premotor rates"],
                "saved_original_cost": data["e296_result"]["cost"],
                "execution": {key: 0 for key in ("original", "model", "neural", "body", "physics", "CUDA", "solver", "fit")}, **FLAGS}
            budget()
            payloads = {"aligned-signals-and-deltas.npz": matched, "observed-transfer.npz": transfer, "full55-endpoint.npz": endpoint, "startup99-errors.npz": errors["arrays"]}
            if sum(array.nbytes for values in payloads.values() for array in values.values()) > LIMITS["output_bytes"]:
                raise OSError("E297 raw output cap")
            for name, arrays in payloads.items():
                np.savez_compressed(output / name, **arrays)
            budget()
            result["elapsed_s"] = time.perf_counter() - began
            memory = psutil.Process().memory_info()
            result["host_peak_bytes"] = getattr(memory, "peak_wset", memory.rss)
            p.write_json(output / "result.json", result)
            p.write_json(output / "output-manifest.json", {"schema": "flylab.m3.e297.output-manifest.v1", "files": {file.name: {"bytes": file.stat().st_size, "sha256": p.sha256(file)} for file in output.iterdir() if file.is_file()}})
            budget()
            if sum(file.stat().st_size for file in output.iterdir()) > LIMITS["output_bytes"]:
                raise OSError("E297 published output cap")
            return result
    except Exception as error:
        (output / "result.json").unlink(missing_ok=True)
        (output / "output-manifest.json").unlink(missing_ok=True)
        p.write_json(output / "failure.json", {"analysis_valid": False, "first_error": str(error), "elapsed_s": time.perf_counter() - began,
            "execution": {key: 0 for key in ("original", "model", "neural", "body", "physics", "CUDA", "solver", "fit")}, **FLAGS})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "source-review", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    for key in ("freeze-sha256", "source-review-sha256"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    freeze = gate(args.freeze, args.freeze_sha256, args.source_review, args.source_review_sha256, args.output)
    old = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    try:
        for key in old:
            os.environ[key] = "1"
        run(freeze, args.output)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
