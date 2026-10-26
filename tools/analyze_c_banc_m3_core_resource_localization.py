"""E295 saved core-input localization; descriptive, no runtime target selection."""
from __future__ import annotations

# Failure receipts retain an incomplete diagnostic rather than scientific success.
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import analyze_c_banc_m3_resource_sign_balance as prior

OWNED = ("tools/analyze_c_banc_m3_core_resource_localization.py", "tests/test_c_banc_core_resource_localization.py")
SOURCE_FILES = (*OWNED, "tools/analyze_c_banc_m3_resource_sign_balance.py", "flylab/c/storage.py", "flylab/c/integrity.py")
INPUT_NAMES = ("positive_roster", "candidate_activity", "candidate_native", "candidate_recruitment", "candidate_trace",
               "e294_result", "e294_ledger", "e294_root_audit", "e294_source_manifest", "e294_output_manifest",
               "e294_freeze", "e294_source_review", "baseline_signals", "baseline_metadata", "baseline_recruitment",
               "baseline_trace", "baseline_core_activity", "baseline_csr", "baseline_e150_output_manifest", "baseline_e257_output_manifest",
               "current_path_source_audit")
FREEZE_SCHEMA = "flylab.m3.e295.core-resource-localization.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e295.core-resource-localization.review.v1"
LIMITS = {"wall_s": 30, "host_peak_bytes": 2 << 30, "output_bytes": 64 << 20, "cpu_threads": 1}
WINDOWS = ((1, 10), (11, 20), (21, 100))
DISPLAY_CONTROLS = (1, 5, 10, 15, 20, 40, 60, 80, 100)
OUTPUT_PATH = "verification/banc-natural-gait-m3-20260919/e295-core-resource-localization"
GROUPS, TERMS = ("core_source", "other_source", "all_selected"), ("Jpos", "JposR", "loss", "Jneg", "JnegR")
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
FLAGS = {**prior.FLAGS, "adopted": False, "causal_claim": False, "candidate_selected": False}
NPZ_INPUTS = {"positive_roster", "candidate_activity", "candidate_native", "candidate_recruitment", "baseline_signals",
              "baseline_metadata", "baseline_recruitment", "baseline_core_activity", "baseline_csr"}


def safe(name):
    value = Path(name)
    path = (ROOT / value).resolve()
    if value.is_absolute() or ".." in value.parts or not path.is_relative_to(ROOT):
        raise ValueError("Repository relative frozen path required")
    return path


def id_columns(observed_ids, requested_ids):
    """Explicit unique ID map, including deliberately unsorted baseline columns."""
    for label, ids in (("observed", observed_ids), ("requested", requested_ids)):
        if (not isinstance(ids, np.ndarray) or ids.dtype != np.int64 or ids.ndim != 1 or not len(ids)
                or np.any(ids < 0) or np.any(ids >= 158706) or len(np.unique(ids)) != len(ids)):
            raise ValueError("Native unique cell IDs required: " + label)
    mapping = {int(cell): column for column, cell in enumerate(observed_ids)}
    if any(int(cell) not in mapping for cell in requested_ids):
        raise ValueError("Required observed cell ID missing")
    return np.array([mapping[int(cell)] for cell in requested_ids], np.int64)


def core_decomposition(selection, activity, core_ids):
    """Selected core inputs only; same saved rates, float64 model input units."""
    _, observed, controls = prior.checked_activity(selection, activity)
    id_columns(observed, core_ids)
    if not np.isin(core_ids, selection["premotor_ids"]).all():
        raise ValueError("Original observed premotor core IDs required")
    core_map = {int(cell): column for column, cell in enumerate(core_ids)}
    chosen = np.flatnonzero(np.isin(selection["post_ids"], core_ids))
    source_columns = id_columns(observed, selection["source_ids"])
    weights = selection["weights32"][chosen].astype(np.float64)
    source_slots = selection["edge_source_slots"][chosen]
    posts = np.array([core_map[int(cell)] for cell in selection["post_ids"][chosen]], np.int64)
    inside = np.isin(selection["pre_ids"][chosen], core_ids)
    negative_sources = np.unique(selection["edge_source_slots"][selection["weights32"] < 0])
    if not np.all(activity["synaptic_resource"][:, negative_sources] == 1.):
        raise ValueError("E294 negative-present source resources must remain exact one")
    result = {"core_ids": core_ids.copy(), "control_tick": controls.copy(), "physics_time_s": activity["physics_time_s"].copy()}
    for group, members in (("core_source", inside), ("other_source", ~inside)):
        for term in TERMS:
            result[group + "_" + term] = np.zeros((len(controls), len(core_ids)), np.float64)
        for row in range(len(controls)):
            current = weights * activity["rate"][row, source_columns[source_slots]].astype(np.float64)
            scaled = current * activity["synaptic_resource"][row, source_slots]
            for term, values, sign in (("Jpos", current, weights > 0), ("JposR", scaled, weights > 0),
                                       ("Jneg", current, weights < 0), ("JnegR", scaled, weights < 0)):
                selected = members & sign
                result[group + "_" + term][row] = np.bincount(posts[selected], weights=values[selected], minlength=len(core_ids))
        result[group + "_loss"] = result[group + "_Jpos"] - result[group + "_JposR"]
        prior.equal_array(result[group + "_JnegR"], result[group + "_Jneg"], "Unchanged same-rate selected negative term")
    for term in TERMS:
        result["all_selected_" + term] = result["core_source_" + term] + result["other_source_" + term]
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite selected-core decomposition")
    return result


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if prior.sha256(freeze_path) != freeze_sha or prior.sha256(review_path) != review_sha:
        raise ValueError("E295 freeze/review SHA mismatch")
    freeze, review = prior.read_json(freeze_path), prior.read_json(review_path)
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("owned_files") != list(OWNED)
            or set(freeze.get("inputs", {})) != set(INPUT_NAMES) or freeze.get("resource_limits") != LIMITS
            or freeze.get("windows") != [list(w) for w in WINDOWS]
            or freeze.get("display_controls") != list(DISPLAY_CONTROLS) or freeze.get("output") != OUTPUT_PATH):
        raise ValueError("Exact E295 fixed-input freeze required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != [] or review.get("freeze_sha256") != freeze_sha
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved matching E295 source review required")
    names = set(freeze["inputs"].values())
    if set(freeze["input_sha256"]) != names or set(freeze["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("Every exact E295 input/source byte must be pinned")
    pins = {}
    for group in (freeze["input_sha256"], freeze["source_sha256"], review["source_sha256"],
                  {freeze["design"]: freeze["design_sha256"]}):
        for name, digest in group.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E295 SHA")
            pins[name] = digest
    for name, digest in pins.items():
        path = safe(name)
        if path.is_symlink() or not isinstance(digest, str) or len(digest) != 64 or prior.sha256(path) != digest:
            raise ValueError("Frozen E295 bytes changed: " + name)
    # Original run receipts independently bind every used raw to its producer.
    data = {name: prior.read_json(safe(freeze["inputs"][name])) for name in INPUT_NAMES if name not in NPZ_INPUTS}
    provenance(freeze["inputs"], data, freeze["input_sha256"])
    if output.resolve() != safe(OUTPUT_PATH) or output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh fixed E295 output required")
    return freeze


def provenance(inputs, data, pins):
    producer_groups = (("baseline_e150_output_manifest", ("baseline_signals", "baseline_metadata", "baseline_recruitment", "baseline_trace")),
                       ("baseline_e257_output_manifest", ("baseline_core_activity", "baseline_csr")),
                       ("e294_output_manifest", ("candidate_activity", "candidate_native", "candidate_recruitment", "candidate_trace", "e294_ledger", "e294_source_manifest")))
    for manifest_alias, aliases in producer_groups:
        manifest_path = safe(inputs[manifest_alias])
        entries = data[manifest_alias].get("files", {})
        for alias in aliases:
            path = safe(inputs[alias])
            if not path.is_relative_to(manifest_path.parent):
                raise ValueError("Raw is outside producer manifest directory")
            relative = path.relative_to(manifest_path.parent).as_posix()
            receipt = entries.get(relative)
            if not isinstance(receipt, dict) or receipt.get("sha256") != pins[inputs[alias]] or receipt.get("bytes") != path.stat().st_size:
                raise ValueError("Producer manifest/raw/inventory SHA mismatch: " + alias)
    path_audit = data["current_path_source_audit"]
    required_sources = {"tools/verify_c_banc_m3_recurrent_resource_body.py", "tools/verify_c_banc_m3_positive_resource_body.py",
                        "tools/banc_m3_recurrent_resource_trial.py", "tools/banc_m3_recurrent_resource_synapses.py", "flylab/c/banc_walking.py"}
    if (path_audit.get("schema") != "flylab.m3.e295.source-path-audit.v1" or path_audit.get("approved") is not True
            or path_audit.get("no_runtime_mute") is not True or path_audit.get("checked_initial_cuts_all_false") is not True
            or path_audit.get("endpoint_masks_all_one") is not True
            or not required_sources <= set(path_audit.get("source_sha256", {}))):
        raise ValueError("Approved exact E294 no-runtime-mute source-path evidence required")
    for name, digest in path_audit["source_sha256"].items():
        if prior.sha256(safe(name)) != digest:
            raise ValueError("Frozen source-path evidence changed")
    source_manifest, old_freeze, old_review = data["e294_source_manifest"], data["e294_freeze"], data["e294_source_review"]
    if (source_manifest.get("diagnostic_id") != "E294" or old_freeze.get("diagnostic_id") != "E294"
            or old_review.get("approved") is not True or old_review.get("blocking_defects") != []
            or old_review.get("freeze_sha256") != pins[inputs["e294_freeze"]]
            or old_review.get("source_sha256") != source_manifest.get("owned_sha256")
            or source_manifest.get("input_sha256") != old_freeze.get("input_sha256")):
        raise ValueError("Exact E294 run source/input/approved review lineage required")
    for group in (source_manifest.get("owned_sha256", {}), source_manifest.get("source_sha256", {}), old_freeze.get("preservation_sha256", {})):
        for name, digest in group.items():
            if prior.sha256(safe(name)) != digest:
                raise ValueError("Frozen E294 source/preservation bytes changed")
    if old_freeze.get("input_sha256", {}).get(inputs["positive_roster"]) != pins[inputs["positive_roster"]]:
        raise ValueError("Actual E294 frozen positive roster SHA differs")
    for alias in ("e294_result", "e294_root_audit"):
        result = data[alias]
        if (result.get("run_valid") is not True or any(result.get(key) is not False for key in
                ("actual_startup_pass", "short_gait_pass", "candidate_adopted", "F1_F8_complete", "F6_exposed"))):
            raise ValueError("Validly rejected original E294 receipts required")
    if data["e294_root_audit"].get("pass") is not True:
        raise ValueError("Root raw audit must pass")
    for identity in ("graph_hash", "source_network_identity", "adapter_identity"):
        if not isinstance(old_freeze.get(identity), str):
            raise TypeError("Frozen original graph/neural/adapter provenance required")


def load(inputs):
    return {name: prior.npz(safe(path)) if name in NPZ_INPUTS else prior.read_json(safe(path)) for name, path in inputs.items()}


def topology(selection, csr, core_ids):
    indptr, indices, weights = (csr.get(key) for key in ("indptr", "indices", "data"))
    prior.array(indptr, np.int32, (158707,), "original CSR row pointers")
    if not isinstance(indices, np.ndarray) or indices.dtype != np.int32 or indices.ndim != 1:
        raise ValueError("Original CSR native indices required")
    prior.array(weights, np.float32, indices.shape, "original CSR weights")
    if indptr[0] != 0 or indptr[-1] != len(indices) or np.any(indptr[1:] < indptr[:-1]) or np.any((indices < 0) | (indices >= 158706)):
        raise ValueError("Original CSR topology invalid")
    edge_ids = selection["edge_ids"]
    if np.any(edge_ids >= len(indices)):
        raise ValueError("Selected edge outside original CSR")
    prior.equal_array(indices[edge_ids].astype(np.int64), selection["pre_ids"], "Selected original pre IDs")
    prior.equal_array((np.searchsorted(indptr, edge_ids, side="right") - 1).astype(np.int64), selection["post_ids"], "Selected original post IDs")
    prior.equal_array(weights[edge_ids], selection["weights32"], "Original selected weight bytes")
    output = []
    for cell in core_ids:
        values = weights[indptr[cell]:indptr[cell + 1]]
        selected = selection["weights32"][selection["post_ids"] == cell]
        output.append({"core_id": int(cell), "all_nonzero_incoming": int(np.count_nonzero(values)),
                       "all_positive_incoming": int(np.count_nonzero(values > 0)), "all_negative_incoming": int(np.count_nonzero(values < 0)),
                       "selected_positive_incoming": int(np.count_nonzero(selected > 0)), "selected_negative_incoming": int(np.count_nonzero(selected < 0))})
    return output


def aligned_signals(data):
    a, s, b, c = data["candidate_activity"], data["positive_roster"], data["baseline_signals"], data["baseline_core_activity"]
    _, _, controls = prior.checked_activity(s, a)
    if len(controls) != 101 or len(s["source_ids"]) != 7136 or len(s["edge_ids"]) != 725849 or len(s["motor_ids"]) != 391:
        raise ValueError("Exact original E294 completed roster/101 rows required")
    core_ids = c.get("core_ids")
    if not isinstance(core_ids, np.ndarray) or core_ids.shape != (24,):
        raise ValueError("All original 24 core IDs required")
    ids = b.get("observed_ids")
    core_columns, motor_columns = id_columns(ids, core_ids), id_columns(ids, s["motor_ids"])
    prior.equal_array(data["baseline_metadata"].get("observed_ids"), ids, "Original signal column metadata")
    prior.equal_array(data["baseline_metadata"].get("motor_ids"), s["motor_ids"], "Original native motor ordering")
    for key in ("rate", "drive", "adaptation"):
        value = b.get(key)
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.ndim != 2 or value.shape[1] != len(ids) or len(value) < 100 or not np.isfinite(value).all():
            raise ValueError("Native complete original signals required: " + key)
    times = prior.array(b.get("time_s"), np.float64, (len(b["rate"]),), "baseline signals clock")
    if np.any(np.abs(times[:100] - np.arange(1, 101) * .005) >= 2e-8):
        raise ValueError("Baseline signals controls1..100 only; no fabricated control0")
    prior.equal_array(c["control_tick"][:101], controls, "Baseline core control alignment")
    prior.equal_array(c["network_tick"][:101], controls * 20, "Baseline core neural clock")
    if c["physics_time_s"].dtype != np.float64 or np.any(np.abs(c["physics_time_s"][:101] - controls * .005) >= 2e-8):
        raise ValueError("Baseline core physical clock")
    core_candidate = id_columns(a["observed_ids"], core_ids)
    motor_candidate = id_columns(a["observed_ids"], s["motor_ids"])
    arrays = {"control_tick": controls[1:].copy(), "baseline_physics_time_s": times[:100].copy(),
              "candidate_physics_time_s": a["physics_time_s"][1:].copy(), "core_ids": core_ids.copy(), "motor_ids": s["motor_ids"].copy()}
    for key in ("rate", "drive", "adaptation"):
        prior.array(c[key], np.float32, (len(c["control_tick"]), 24), "Original core " + key)
        prior.equal_array(b[key][:100, core_columns], c[key][1:101], "Exact E150/E257 core " + key)
        for group, baseline_columns, candidate_columns in (("core", core_columns, core_candidate), ("motor", motor_columns, motor_candidate)):
            arrays[group + "_" + key + "_baseline"] = b[key][:100, baseline_columns].copy()
            arrays[group + "_" + key + "_candidate"] = a[key][1:, candidate_columns].copy()
            arrays["candidate_raw_" + group + "_" + key] = a[key][:, candidate_columns].copy()
    baseline_motor = prior.matching_recruitment(data["baseline_recruitment"], controls[1:], "E150")
    candidate_motor = prior.matching_recruitment(data["candidate_recruitment"], controls, "E294")
    prior.equal_array(candidate_motor, a["rate"][:, motor_candidate].astype(np.float64), "Candidate native/activity motor definition")
    prior.equal_array(baseline_motor, b["rate"][:100, motor_columns].astype(np.float64), "Baseline native/signals motor definition")
    muscle = prior.array(b.get("muscle_activation"), np.float64, (len(times), 96), "Baseline signals muscle activation")
    prior.equal_array(data["baseline_recruitment"]["control_tick"][:101], controls, "Baseline native control0..100")
    prior.equal_array(muscle[:100], data["baseline_recruitment"]["activation"][1:101], "Baseline muscle definitions exact")
    arrays["muscle_activation_baseline"], arrays["muscle_activation_candidate"] = muscle[:100].copy(), data["candidate_recruitment"]["activation"][1:].copy()
    baseline_q, _ = prior.matching_trace(data["baseline_trace"], controls[1:], "E150")
    candidate_q, _ = prior.matching_trace(data["candidate_trace"], controls, "E294")
    if len(data["candidate_trace"]["samples"]) != 101:
        raise ValueError("Candidate trace complete original101 rows required")
    arrays["q42_baseline"], arrays["q42_candidate"] = baseline_q, candidate_q[1:]
    arrays["candidate_raw_q42"] = candidate_q
    arrays["candidate_raw_muscle_activation"] = data["candidate_recruitment"]["activation"].copy()
    for group, key in (("baseline", "baseline_trace"), ("candidate", "candidate_trace")):
        rows = {row["control_tick"]: row for row in data[key]["samples"]}
        tendon = np.asarray([[rows[int(t)]["tendon_inputs"][leg + "_tarsus"] for leg in LEGS] for t in controls[1:]], np.float64)
        prior.array(tendon, np.float64, (100, 6), "Original six tendon command order")
        arrays["tendon6_" + group] = tendon
    arrays["candidate_raw_control_tick"] = controls.copy()
    return arrays


def first_and_windows(baseline, candidate, controls, ids):
    if (baseline.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            or baseline.dtype != candidate.dtype or baseline.shape != candidate.shape or baseline.ndim != 2):
        raise ValueError("Exact paired saved definitions required")
    prior.array(controls, np.int64, (len(baseline),), "Aligned summary controls")
    prior.equal_array(controls, np.arange(1, 101, dtype=np.int64), "Exact complete summary controls1..100")
    if len(ids) != baseline.shape[1] or not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("Finite all-cell aligned summary required")
    delta = candidate.astype(np.float64) - baseline.astype(np.float64)
    bit_dtype = np.uint32 if baseline.dtype == np.float32 else np.uint64
    different = np.ascontiguousarray(baseline).view(bit_dtype) != np.ascontiguousarray(candidate).view(bit_dtype)
    cells = []
    for column, cell in enumerate(ids):
        changed = np.flatnonzero(different[:, column])
        windows = []
        for first, last in WINDOWS:
            chosen = (controls >= first) & (controls <= last)
            values = delta[chosen, column]
            if not len(values):
                raise ValueError("Complete fixed window missing")
            windows.append({"first_control": first, "last_control": last, "samples": int(np.count_nonzero(chosen)),
                            "mean_delta": float(np.mean(values)), "RMS_delta": float(np.sqrt(np.mean(values ** 2))),
                            "maximum_absolute_delta": float(np.max(np.abs(values)))})
        cells.append({"id": int(cell), "first_exact_difference_control": None if not len(changed) else int(controls[changed[0]]),
                      "windows": windows})
    changed = np.flatnonzero(different.any(axis=1))
    return {"first_exact_difference_control": None if not len(changed) else int(controls[changed[0]]), "all_cells": cells}, delta



def loss_fraction(inside, outside):
    total = float(inside + outside)
    return {"available": total > 0, "core_source_fraction": float(inside / total) if total > 0 else None,
            "other_source_fraction": float(outside / total) if total > 0 else None}


def decomposition_windows(decomposition):
    """Every core and component; no rank-selected subset or tonic/phasic inference."""
    controls = decomposition["control_tick"]
    output = []
    for column, cell in enumerate(decomposition["core_ids"]):
        windows = []
        for first, last in WINDOWS:
            chosen = (controls >= first) & (controls <= last)
            components = {}
            for group in GROUPS:
                for term in TERMS:
                    values = decomposition[group + "_" + term][chosen, column]
                    if not len(values):
                        raise ValueError("Missing fixed decomposition window")
                    components[group + "_" + term] = {"mean": float(np.mean(values)),
                        "RMS": float(np.sqrt(np.mean(values ** 2))), "maximum_absolute": float(np.max(np.abs(values)))}
            inside = np.sum(decomposition["core_source_loss"][chosen, column])
            outside = np.sum(decomposition["other_source_loss"][chosen, column])
            windows.append({"first_control": first, "last_control": last, "components": components,
                            "integrated_sample_loss_fraction": loss_fraction(inside, outside)})
        output.append({"core_id": int(cell), "windows": windows})
    return output


def validate_health(data):
    audit, ledger = data["e294_root_audit"], data["e294_ledger"]["arms"]["candidate"]
    arm = audit["arms"]["candidate"]
    if (arm.get("controls") != 100 or arm.get("native_samples") != 5000 or arm.get("actual_neural_ticks") != 2000
            or arm.get("whole_final_available") is not True or arm.get("negative_source_resource_exactly_one") is not True
            or ledger.get("completed_controls") != 100 or ledger.get("readout_endpoints") != 101
            or ledger.get("native_samples") != 5000 or ledger.get("close_returned") is not True):
        raise ValueError("Audited healthy100-control E294 rejection required")
    prior.clock_close(ledger.get("actual_neural_s"), .5, "E294 neural")
    prior.clock_close(ledger.get("actual_physical_s"), .5, "E294 physical")
    native = data["candidate_native"]
    times = prior.array(native["time_s"], np.float64, (5000,), "E294 native clock")
    if np.any(np.abs(times - np.arange(1, 5001) * .0001) >= 2e-8):
        raise ValueError("Complete native100us clock required")
    upright = prior.array(native["upright"], np.float64, (5000,), "E294 upright")
    nonfoot = prior.array(native["nonfoot_contact"], bool, (5000,), "E294 nonfoot")
    prior.array(native["normal_load_bw"], np.float64, (5000, 6), "E294 native load")
    if np.any(upright < .9) or nonfoot.any():
        raise ValueError("E294 saved healthy prefix must remain healthy")


def run(freeze, output):
    began = time.perf_counter()
    import psutil
    from threadpoolctl import threadpool_limits
    def budget():
        if time.perf_counter() - began >= LIMITS["wall_s"]:
            raise TimeoutError("E295 saved-analysis deadline")
        info = psutil.Process().memory_info()
        if getattr(info, "peak_wset", info.rss) > LIMITS["host_peak_bytes"]:
            raise MemoryError("E295 saved-analysis host cap")
    output = Path(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh E295 output required")
    with threadpool_limits(limits=1):
        budget()
        data = load(freeze["inputs"])
        provenance(freeze["inputs"], data, freeze["input_sha256"])
        validate_health(data)
        matched = aligned_signals(data)
        decomposition = core_decomposition(data["positive_roster"], data["candidate_activity"], matched["core_ids"])
        topology_receipt = topology(data["positive_roster"], data["baseline_csr"], matched["core_ids"])
        summaries = {}
        for group, terms, ids in (("core", ("rate", "drive", "adaptation"), matched["core_ids"]),
                                  ("motor", ("rate", "drive", "adaptation"), matched["motor_ids"]),
                                  ("muscle", ("activation",), np.arange(96)), ("q42", (None,), np.arange(42)), ("tendon6", (None,), np.arange(6))):
            for term in terms:
                key = group if term is None else group + "_" + term
                summaries[key], matched[key + "_delta"] = first_and_windows(matched[key + "_baseline"], matched[key + "_candidate"], matched["control_tick"], ids)
        loss_summary = []
        for row in DISPLAY_CONTROLS:
            inside, outside = decomposition["core_source_loss"][row], decomposition["other_source_loss"][row]
            total = inside + outside
            loss_summary.append({"control_tick": row, "all_core_cells": [{"core_id": int(cell),
                "core_source_loss": float(inside[column]), "other_source_loss": float(outside[column]),
                "total_loss": float(total[column]), "fraction_available": bool(total[column] > 0),
                "core_source_fraction": float(inside[column] / total[column]) if total[column] > 0 else None}
                for column, cell in enumerate(matched["core_ids"])]})
        result = {"schema": "flylab.m3.e295.core-resource-localization.result.v1", "analysis_valid": True,
            "units": "model input units; not amperes", "same_saved_rate_selected_inputs_only": True,
            "all_recurrent_current_observed": False, "CUDA_exact_current": False, "first_difference_scope": "5ms saved-grid observation; not substep onset or causal precedence",
            "baseline_controls": [1, 100], "candidate_raw_controls": [0, 100], "summaries": summaries,
            "display_loss": loss_summary, "original_incoming_topology_counts": topology_receipt,
            "tendon_column_order": [leg + "_tarsus" for leg in LEGS],
            "decomposition_windows": decomposition_windows(decomposition),
            "mask_evidence": {"per_tick_mask_measured": False, "source_path_audit": data["current_path_source_audit"]},
            "saved_original_cost": data["e294_result"]["cost"],
            "execution": {key: 0 for key in ("original", "model", "neural", "body", "physics", "CUDA", "solver", "fit")}, **FLAGS}
        budget()
        prior.canonical(result)
        if sum(x.nbytes for x in (*matched.values(), *decomposition.values())) > LIMITS["output_bytes"]:
            raise OSError("E295 output cap")
        output.mkdir()
        try:
            prior.write_json(output / "source-manifest.json", {"source_sha256": freeze["source_sha256"], "input_sha256": freeze["input_sha256"]})
            np.savez_compressed(output / "core-selected-inputs.npz", **decomposition)
            np.savez_compressed(output / "aligned-signals-and-deltas.npz", **matched)
            budget()
            result["elapsed_s"] = time.perf_counter() - began
            prior.write_json(output / "result.json", result)
            manifest = {"schema": "flylab.m3.e295.output-manifest.v1", "files": {p.name: {"sha256": prior.sha256(p), "bytes": p.stat().st_size} for p in output.iterdir() if p.is_file()}}
            prior.write_json(output / "output-manifest.json", manifest)
            budget()
            if sum(p.stat().st_size for p in output.iterdir()) > LIMITS["output_bytes"]:
                raise OSError("E295 published output cap")
        except Exception as error:
            (output / "result.json").unlink(missing_ok=True)
            (output / "output-manifest.json").unlink(missing_ok=True)
            prior.write_json(output / "failure.json", {"analysis_valid": False, "first_error": str(error), **FLAGS})
            raise
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", required=True, type=Path)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
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
