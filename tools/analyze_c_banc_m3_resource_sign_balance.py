"""E293 saved-state sign decomposition in model input units; no causal claim."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from flylab.c.integrity import canonical, read_json, write_json
from flylab.c.storage import StateStore

OWNED = ("tools/analyze_c_banc_m3_resource_sign_balance.py", "tests/test_c_banc_resource_sign_balance.py")
PATH_SOURCES = ("tools/verify_c_banc_m3_recurrent_resource_body.py", "tools/banc_m3_recurrent_resource_trial.py",
                "tools/banc_m3_recurrent_resource_synapses.py")
SOURCE_FILES = (*OWNED, "flylab/c/storage.py", "flylab/c/integrity.py", *PATH_SOURCES)
INPUT_NAMES = ("roster", "candidate_activity", "candidate_native", "candidate_recruitment", "candidate_trace",
               "baseline_trace", "baseline_recruitment", "observed_start_checkpoint", "partial_network_checkpoint",
               "e292_root_audit", "e292_result", "e292_ledger", "current_path_source_audit")
INPUT_SCHEMA = "flylab.m3.e293.resource-sign-balance.inputs.v1"
FREEZE_SCHEMA = "flylab.m3.e293.resource-sign-balance.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e293.resource-sign-balance.review.v1"
LIMITS = {"wall_s": 30, "host_peak_bytes": 2 << 30, "output_bytes": 128 << 20, "cpu_threads": 1}
FIXED_CONTROLS = (0, 10, 20, 40, 62)
COMPONENTS = ("J_exc", "J_inh", "J_exc_R", "J_inh_R", "excitation_loss", "inhibition_relief", "net_change")
SELECTION_KEYS = ("premotor_ids", "motor_ids", "edge_ids", "pre_ids", "post_ids", "source_ids", "edge_source_slots", "weights32")
FLAGS = {"candidate_adopted": False, "F1_F8_complete": False, "F6_exposed": False,
         "natural_gait_complete": False, "causal_gait_claim": False, "runtime_target_selection": False}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def equal_array(actual, expected, name):
    if (not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype
            or actual.shape != expected.shape or actual.tobytes() != expected.tobytes()):
        raise ValueError("Exact native array required: " + name)


def array(value, dtype, shape, name, *, finite=True):
    if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape
            or finite and not np.isfinite(value).all()):
        raise ValueError("Native dtype/shape/finite required: " + name)
    return value


def sorted_ids(value, name):
    if (not isinstance(value, np.ndarray) or value.dtype != np.int64 or value.ndim != 1
            or not len(value) or np.any(value < 0) or np.any(value[1:] <= value[:-1])):
        raise ValueError("Sorted unique nonnegative int64 IDs required: " + name)
    return value


def checked_activity(selection, activity):
    for key in SELECTION_KEYS:
        if key not in selection:
            raise ValueError("Missing original selected CSR field: " + key)
    pop, sources, edges = (sorted_ids(selection[key], key) for key in ("premotor_ids", "source_ids", "edge_ids"))
    if np.any(pop >= 158706):
        raise ValueError("Premotor cell ID out of original bounds")
    motors = selection["motor_ids"]
    if (not isinstance(motors, np.ndarray) or motors.dtype != np.int64 or motors.ndim != 1
            or not len(motors) or np.any(motors < 0) or np.any(motors >= 158706) or len(np.unique(motors)) != len(motors)
            or np.intersect1d(motors, pop).size):
        raise ValueError("Unique native disjoint motor IDs required")
    for key in ("pre_ids", "post_ids"):
        value = array(selection[key], np.int64, edges.shape, key)
        if not np.isin(value, pop).all():
            raise ValueError("Existing premotor endpoints required")
    equal_array(sources, np.unique(selection["pre_ids"]), "unique selected sources")
    equal_array(selection["edge_source_slots"], np.searchsorted(sources, selection["pre_ids"]).astype(np.int32), "shared source slots")
    weights = array(selection["weights32"], np.float32, edges.shape, "weights32")
    if np.any(weights == 0):
        raise ValueError("Original nonzero selected weight bytes required")
    observed = sorted_ids(activity.get("observed_ids"), "observed_ids")
    equal_array(observed, np.unique(np.r_[pop, motors]).astype(np.int64), "observed row IDs")
    equal_array(activity.get("source_ids"), sources, "activity resource columns")
    equal_array(activity.get("motor_ids"), motors, "activity motor IDs")
    controls = activity.get("control_tick")
    if not isinstance(controls, np.ndarray) or controls.dtype != np.int64 or controls.ndim != 1 or not len(controls):
        raise ValueError("Native control sequence required")
    rows = len(controls)
    equal_array(controls, np.arange(rows, dtype=np.int64), "control sequence from zero")
    equal_array(activity.get("network_tick"), controls * 20, "neural control clock")
    complete = array(activity.get("complete"), bool, (rows,), "activity complete")
    if not complete.all():
        raise ValueError("Incomplete activity readout")
    times = array(activity.get("physics_time_s"), np.float64, (rows,), "physics clock")
    if np.any(np.abs(times - controls * .005) >= 2e-8):
        raise ValueError("Saved physical control clock mismatch")
    for key in ("rate", "drive", "adaptation"):
        array(activity.get(key), np.float32, (rows, len(observed)), key)
    if np.any(activity["rate"] < 0):
        raise ValueError("Nonnegative saved rates required for sign losses")
    resource = array(activity.get("synaptic_resource"), np.float64, (rows, len(sources)), "resource")
    if np.any(resource < 0) or np.any(resource > 1):
        raise ValueError("Resource outside [0,1]")
    equal_array(resource[0], np.ones(len(sources), np.float64), "initial resource one")
    return pop, observed, controls


def decompose(selection, activity):
    """Float64 mathematical sums at SAME saved rates, not a CUDA current oracle."""
    pop, observed, controls = checked_activity(selection, activity)
    pre_columns = np.searchsorted(observed, selection["pre_ids"])
    post_columns = np.searchsorted(pop, selection["post_ids"])
    weights = selection["weights32"].astype(np.float64)
    positive, negative = weights > 0, weights < 0
    result = {key: np.zeros((len(controls), len(pop)), np.float64) for key in COMPONENTS}
    for row in range(len(controls)):
        current = weights * activity["rate"][row, pre_columns].astype(np.float64)
        scaled = current * activity["synaptic_resource"][row, selection["edge_source_slots"]]
        for key, values, chosen in (("J_exc", current, positive), ("J_inh", current, negative),
                                   ("J_exc_R", scaled, positive), ("J_inh_R", scaled, negative)):
            result[key][row] = np.bincount(post_columns[chosen], weights=values[chosen], minlength=len(pop))
    result["excitation_loss"] = result["J_exc"] - result["J_exc_R"]
    result["inhibition_relief"] = result["J_inh_R"] - result["J_inh"]
    result["net_change"] = result["inhibition_relief"] - result["excitation_loss"]
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite descriptive decomposition")
    return {"post_ids": pop.copy(), "control_tick": controls.copy(),
            "physics_time_s": activity["physics_time_s"].copy(), **result}


def safe(name):
    relative = Path(name)
    path = (ROOT / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(ROOT):
        raise ValueError("Repository relative pinned path required")
    return path


def input_files(inputs):
    if not isinstance(inputs, dict) or set(inputs) != set(INPUT_NAMES):
        raise ValueError("Exact flat E293 inputs mapping required")
    files = []
    for alias, name in inputs.items():
        path = safe(name)
        if alias.endswith("_checkpoint"):
            if not path.is_dir():
                raise ValueError("Checkpoint directory required: " + alias)
            children = list(path.iterdir())
            if not children or any(not child.is_file() or child.is_symlink() for child in children):
                raise ValueError("Complete flat passive checkpoint required")
            files.extend(children)
        elif not path.is_file() or path.is_symlink():
            raise ValueError("Regular frozen input required: " + alias)
        else:
            files.append(path)
    return files


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if sha256(freeze_path) != freeze_sha or sha256(review_path) != review_sha:
        raise ValueError("E293 freeze/review SHA mismatch")
    freeze, review = read_json(freeze_path), read_json(review_path)
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("input_schema") != INPUT_SCHEMA
            or freeze.get("owned_files") != list(OWNED) or freeze.get("resource_limits") != LIMITS
            or freeze.get("fixed_controls") != list(FIXED_CONTROLS)):
        raise ValueError("Exact E293 freeze required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != [] or review.get("freeze_sha256") != freeze_sha
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved matching E293 source review required")
    files = input_files(freeze["inputs"])
    expected = {path.relative_to(ROOT).as_posix() for path in files}
    if set(freeze["input_sha256"]) != expected:
        raise ValueError("Every input manifest/state/array byte must be pinned")
    pins = {}
    for group in (freeze["input_sha256"], freeze["source_sha256"], review["source_sha256"],
                  {freeze["design"]: freeze["design_sha256"]}):
        for name, digest in group.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E293 pin")
            pins[name] = digest
    if set(freeze["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("Exact actual E293/source-path dependencies required")
    for name, digest in pins.items():
        if not isinstance(digest, str) or len(digest) != 64 or sha256(safe(name)) != digest:
            raise ValueError("E293 pinned bytes changed: " + name)
    audit = read_json(safe(freeze["inputs"]["current_path_source_audit"]))
    if (audit.get("schema") != "flylab.m3.e293.source-path-audit.v1" or audit.get("approved") is not True
            or audit.get("no_runtime_mute") is not True
            or audit.get("source_sha256") != {name: pins[name] for name in PATH_SOURCES}):
        raise ValueError("Approved exact no-runtime-mute source audit required")
    inherited = audit.get("inherited_source_sha256")
    if not isinstance(inherited, dict) or not inherited:
        raise ValueError("Complete frozen inherited current path audit required")
    for name, digest in inherited.items():
        if (not isinstance(digest, str) or len(digest) != 64 or sha256(safe(name)) != digest
                or name in pins and pins[name] != digest):
            raise ValueError("Inherited no-mute source audit pin changed: " + name)
    if audit.get("checked_initial_cuts_all_false") is not True or audit.get("endpoint_masks_all_one") is not True:
        raise ValueError("Audited initial cuts and endpoint masks required")
    if output.resolve() != safe(freeze["output"]) or output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh fixed E293 output required")
    return freeze


def npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_inputs(inputs):
    input_files(inputs)
    data = {}
    for alias, name in inputs.items():
        if alias.endswith("_checkpoint"):
            data[alias] = StateStore.load(safe(name))
        elif alias in ("roster", "candidate_activity", "candidate_native", "candidate_recruitment", "baseline_recruitment"):
            data[alias] = npz(safe(name))
        else:
            data[alias] = read_json(safe(name))
    return data


def distribution(values):
    x = np.asarray(values, np.float64)
    if not x.size or not np.isfinite(x).all():
        raise ValueError("Finite descriptive distribution required")
    return {"minimum": float(np.min(x)), "maximum": float(np.max(x)), "mean": float(np.mean(x)),
            "p10": float(np.quantile(x, .1)), "median": float(np.median(x)), "p90": float(np.quantile(x, .9))}


def clock_close(actual, expected, name):
    if type(actual) not in (int, float) or not np.isfinite(actual) or abs(actual - expected) >= 2e-8:
        raise ValueError("Saved clock mismatch: " + name)


def matching_recruitment(value, controls, label):
    ticks = value.get("control_tick")
    if (not isinstance(ticks, np.ndarray) or ticks.dtype != np.int64 or ticks.ndim != 1
            or np.any(ticks[1:] <= ticks[:-1])):
        raise ValueError("Unique chronological recruitment rows required: " + label)
    positions = np.searchsorted(ticks, controls)
    if np.any(positions >= len(ticks)) or not np.array_equal(ticks[positions], controls):
        raise ValueError("Missing recruitment control: " + label)
    for key, dtype in (("network_tick", np.int64), ("physics_time_s", np.float64), ("complete", bool)):
        array(value.get(key), dtype, ticks.shape, label + key)
    equal_array(value["network_tick"][positions], controls * 20, label + "neural clock")
    if (np.any(np.abs(value["physics_time_s"][positions] - controls * .005) >= 2e-8)
            or not value["complete"][positions].all()):
        raise ValueError("Recruitment clock/completeness mismatch: " + label)
    motor = array(value.get("motor_rate"), np.float64, (len(ticks), 391), label + "motor rate")
    for key in ("bank_available", "activation_available", "original_activation_available", "motor_rate_available"):
        if not array(value.get(key), bool, ticks.shape, label + key)[positions].all():
            raise ValueError("Incomplete recruitment field: " + label + key)
    for key, shape in (("bank", (391, 3)), ("activation", (96,)), ("original_activation", (96,))):
        array(value.get(key), np.float64, (len(ticks), *shape), label + key)
    return motor[positions]


def matching_trace(value, controls, label):
    rows = value.get("samples")
    if not isinstance(rows, list):
        raise TypeError("Trace samples required")
    by_control = {}
    previous = -1
    for row in rows:
        tick = row.get("control_tick")
        if type(tick) is not int or tick <= previous:
            raise ValueError("Unique chronological trace controls required: " + label)
        previous = tick
        by_control[tick] = row
    commands, upright = [], []
    for tick in controls.tolist():
        if tick not in by_control:
            raise ValueError("Missing trace control: " + label)
        row = by_control[tick]
        clock_close(row.get("time_s"), tick * .005, label + "trace")
        q = np.asarray(row.get("joint_targets_rad"), np.float64)
        if q.shape != (42,) or not np.isfinite(q).all() or type(row.get("upright")) not in (float, int) or not np.isfinite(row["upright"]):
            raise ValueError("Finite q42/upright trace required: " + label)
        commands.append(q)
        upright.append(row["upright"])
    return np.stack(commands), np.asarray(upright, np.float64)


def native_guard(native):
    times = array(native.get("time_s"), np.float64, (3124,), "native times")
    if np.any(np.abs(times - np.arange(1, 3125) * .0001) >= 2e-8):
        raise ValueError("Exact native 100us sample clock required")
    load = array(native.get("normal_load_bw"), np.float64, (3124, 6), "native load", finite=False)
    slip = array(native.get("slip_mm_s"), np.float64, (3124, 6), "native slip", finite=False)
    valid = array(native.get("slip_valid"), bool, (3124, 6), "native slip validity")
    nonfoot = array(native.get("nonfoot_contact"), bool, (3124,), "native nonfoot")
    upright = array(native.get("upright"), np.float64, (3124,), "native upright", finite=False)
    bad_finite = (~np.isfinite(load).all(axis=1) | ~np.isfinite(upright)
                  | np.any(valid & ~np.isfinite(slip), axis=1))
    bad = bad_finite | nonfoot | (upright < .9)
    indices = np.flatnonzero(bad)
    if len(indices) != 1 or indices[0] != 3123:
        raise ValueError("Recorded E292 first guard must terminate final native sample")
    index = int(indices[0])
    def observed(value):
        return float(value) if np.isfinite(value) else None
    return {"sample_index": index, "time_s": float(times[index]), "upright": observed(upright[index]),
            "normal_load_bw": [observed(x) for x in load[index]],
            "slip_mm_s": [observed(x) for x in slip[index]], "slip_valid": valid[index].tolist(),
            "nonfoot_contact": bool(nonfoot[index]), "nonfinite_guard": bool(bad_finite[index]),
            "upright_below_0p9": bool(upright[index] < .9)}


def validate_saved(data):
    s, a = data["roster"], data["candidate_activity"]
    _, _, controls = checked_activity(s, a)
    if len(controls) != 63 or len(s["premotor_ids"]) != 7136 or len(s["edge_ids"]) != 725849 or len(s["motor_ids"]) != 391:
        raise ValueError("Fixed E292 roster and 63 complete endpoint rows required")
    for key, expected in (("utilization", .05), ("neutral_utilization", 0.), ("recovery_tau", .150)):
        equal_array(s.get(key), np.full(7136, expected, np.float64), "fixed roster " + key)
    audit, result, ledger = data["e292_root_audit"], data["e292_result"], data["e292_ledger"]
    if (audit.get("schema") != "flylab.m3.e292.root-original-raw-audit.v1" or audit.get("pass") is not True
            or audit.get("run_valid") is not True or result.get("diagnostic_id") != "E292"
            or result.get("run_valid") is not True or result.get("neutral_exact") is not True
            or any(result.get(key) is not False for key in ("actual_startup_pass", "short_gait_pass", "candidate_adopted", "F1_F8_complete", "F6_exposed", "natural_gait_complete"))):
        raise ValueError("Validly rejected E292 audit/result required")
    audited_arm = audit.get("arms", {}).get("candidate", {})
    result_arm = result.get("arms", {}).get("candidate", {})
    if (audited_arm.get("controls") != 62 or audited_arm.get("native_samples") != 3124
            or audited_arm.get("closed") is not True or audited_arm.get("whole_final_available") is not False
            or result_arm.get("completed_controls") != 62 or result_arm.get("startup", {}).get("available") is not False
            or result_arm.get("health_stop", {}).get("control") != 63):
        raise ValueError("Audited E292 incomplete-startup rejection required")
    arm = ledger.get("arms", {}).get("candidate", {})
    if (arm.get("completed_controls") != 62 or arm.get("readout_endpoints") != 63
            or arm.get("native_samples") != 3124 or arm.get("close_returned") is not True
            or arm.get("clock_valid") is not True or arm.get("status") != "measured_native_health_stop"):
        raise ValueError("Complete E292 partial arm ledger required")
    events = arm.get("clock_events")
    if not isinstance(events, list) or len(events) != 63:
        raise ValueError("Complete separate partial-control clock events required")
    last = events[-1]
    if (last.get("neural_tick_before") != 1240 or last.get("neural_tick_after") != 1260
            or last.get("control_tick_after") != 62 or last.get("advance_error") != "NativeHealthStop"):
        raise ValueError("Partial neural advancement must remain separate from completed controls")
    clock_close(last.get("physics_time_before_s"), .31, "last complete physical endpoint")
    clock_close(last.get("physics_time_after_s"), .3124, "partial physical event")
    clock_close(arm.get("actual_neural_s"), .315, "partial neural")
    clock_close(arm.get("actual_physical_s"), .3124, "partial physical")
    whole_start = data["observed_start_checkpoint"]
    cuts = whole_start.get("cuts")
    if not isinstance(cuts, dict) or set(cuts) != {"circuit", "descending", "motor", "sensory"} or any(value is not False for value in cuts.values()):
        raise ValueError("Original starting cuts all false required")
    start, partial = whole_start["network"], data["partial_network_checkpoint"]
    for label, state, tick in (("start", start, 0), ("partial", partial, 1260)):
        if type(state.get("tick")) is not int or state["tick"] != tick:
            raise ValueError("Separate saved neural endpoint tick required")
        equal_array(state.get("output_mask"), np.ones(158706, np.float32), label + "measured mask endpoint")
        for key in ("rate", "drive", "adaptation"):
            array(state.get(key), np.float32, (158706,), label + key)
        resource = array(state.get("synaptic_resource"), np.float64, (7136,), label + "resource")
        if np.any((resource < 0) | (resource > 1)):
            raise ValueError("Partial resource bounds")
        for key in SELECTION_KEYS:
            equal_array(state.get("recurrent_resource", {}).get(key), s[key], label + "selected metadata " + key)
    equal_array(start["synaptic_resource"], np.ones(7136, np.float64), "start resource")
    path_audit = data["current_path_source_audit"]
    if (path_audit.get("schema") != "flylab.m3.e293.source-path-audit.v1" or path_audit.get("approved") is not True
            or path_audit.get("no_runtime_mute") is not True
            or path_audit.get("checked_initial_cuts_all_false") is not True
            or path_audit.get("endpoint_masks_all_one") is not True
            or set(path_audit.get("source_sha256", {})) != set(PATH_SOURCES)):
        raise ValueError("Explicit source audit of intermediate mask path required")
    guard = native_guard(data["candidate_native"])
    baseline_motor = matching_recruitment(data["baseline_recruitment"], controls, "baseline")
    candidate_motor = matching_recruitment(data["candidate_recruitment"], controls, "candidate")
    columns = np.searchsorted(a["observed_ids"], s["motor_ids"])
    equal_array(candidate_motor, a["rate"][:, columns].astype(np.float64), "activity/recruitment motor correspondence")
    candidate_q, upright = matching_trace(data["candidate_trace"], controls, "candidate")
    baseline_q, _ = matching_trace(data["baseline_trace"], controls, "baseline")
    if len(data["candidate_trace"]["samples"]) != 63:
        raise ValueError("Candidate trace must contain only completed 63 rows")
    return {"motor_delta": candidate_motor - baseline_motor, "q_delta": candidate_q - baseline_q,
            "control_upright": upright, "native_guard": guard,
            "partial_neural_s": float(arm["actual_neural_s"]), "partial_physical_s": float(arm["actual_physical_s"])}


def summarize(decomposition, data, matched):
    summaries = []
    native = data["candidate_native"]
    for row in FIXED_CONTROLS:
        components = {key: distribution(decomposition[key][row]) for key in COMPONENTS}
        net = decomposition["net_change"][row]
        positive = np.flatnonzero(net > 0)
        negative = np.flatnonzero(net < 0)
        # Stable ties follow the already sorted original post IDs; descriptive only.
        top_positive = positive[np.argsort(-net[positive], kind="stable")[:10]]
        top_negative = negative[np.argsort(net[negative], kind="stable")[:10]]
        t = row * .005
        native_index = None if row == 0 else row * 50 - 1
        summaries.append({"control_tick": row, "time_s": t,
            "signed_total_model_input_units": {key: float(np.sum(decomposition[key][row], dtype=np.float64)) for key in COMPONENTS},
            "distributions_model_input_units": components,
            "net_positive_posts": len(positive), "net_negative_posts": len(negative),
            "descriptive_top_positive": [{"post_id": int(decomposition["post_ids"][i]), "net_change": float(net[i])} for i in top_positive],
            "descriptive_top_negative": [{"post_id": int(decomposition["post_ids"][i]), "net_change": float(net[i])} for i in top_negative],
            "motor_baseline_delta": distribution(matched["motor_delta"][row]),
            "motor_delta_RMS": float(np.sqrt(np.mean(matched["motor_delta"][row] ** 2))),
            "q42_command_delta_rad": matched["q_delta"][row].tolist(),
            "q_delta_max_abs_rad": float(np.max(np.abs(matched["q_delta"][row]))),
            "control_upright": float(matched["control_upright"][row]),
            "native_sample_index": native_index,
            "native_upright": None if native_index is None else float(native["upright"][native_index]),
            "native_normal_load_bw": None if native_index is None else native["normal_load_bw"][native_index].tolist(),
            "resource": distribution(data["candidate_activity"]["synaptic_resource"][row])})
    return summaries


def analyze(inputs, output):
    """Analyze frozen saved inputs once; constructors and runtime dynamics absent."""
    began = time.perf_counter()
    import psutil
    from threadpoolctl import threadpool_limits
    def budget():
        if time.perf_counter() - began >= LIMITS["wall_s"]:
            raise TimeoutError("E293 saved-analysis deadline")
        info = psutil.Process().memory_info()
        if getattr(info, "peak_wset", info.rss) > LIMITS["host_peak_bytes"]:
            raise MemoryError("E293 saved-analysis host cap")
    output = Path(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh saved-analysis output required")
    with threadpool_limits(limits=1):
        budget()
        data = load_inputs(inputs)
        budget()
        matched = validate_saved(data)
        decomposition = decompose(data["roster"], data["candidate_activity"])
        budget()
        summary = {"schema": "flylab.m3.e293.resource-sign-balance.result.v1", "analysis_valid": True,
            "same_saved_rate_descriptive_decomposition": True, "units": "model input units; not amperes",
            "original_baseline_current": False, "CUDA_FTZ_RK4_oracle": False,
            "future_counterfactual": False, "prefix_is_startup_adjudication": False,
            "complete_control_rows": 63, "last_complete_control_time_s": .31,
            "partial_neural_time_s": matched["partial_neural_s"], "partial_physical_time_s": matched["partial_physical_s"],
            "saved_original_cost_receipt": data["e292_result"].get("cost", {}),
            "native_guard": matched["native_guard"], "fixed_timepoints": summarize(decomposition, data, matched),
            "mask_evidence": {"observed_start_all_one": True, "partial_final_all_one": True,
                "intermediate_masks_measured": False, "no_runtime_mute_source_audit": data["current_path_source_audit"]},
            "execution": {"original": 0, "model": 0, "CUDA": 0, "body": 0, "physics": 0, "fit": 0, "solver": 0}, **FLAGS}
        # Validate JSON and all resource limits before publishing successful output.
        canonical(summary)
        if sum(value.nbytes for value in decomposition.values()) + matched["motor_delta"].nbytes + matched["q_delta"].nbytes > LIMITS["output_bytes"]:
            raise OSError("E293 output cap")
        output.mkdir()
        try:
            np.savez_compressed(output / "sign-decomposition.npz", **decomposition)
            np.savez_compressed(output / "matched-prefix-deltas.npz", control_tick=decomposition["control_tick"],
                motor_ids=data["roster"]["motor_ids"], motor_delta=matched["motor_delta"],
                q42_delta_rad=matched["q_delta"], control_upright=matched["control_upright"])
            budget()
            summary["elapsed_s"] = time.perf_counter() - began
            write_json(output / "result.json", summary)
            manifest = {"schema": "flylab.m3.e293.output-manifest.v1", "files": {
                p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
            write_json(output / "output-manifest.json", manifest)
            budget()
            if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > LIMITS["output_bytes"]:
                raise OSError("E293 published output cap")
        except Exception as error:
            (output / "result.json").unlink(missing_ok=True)
            (output / "output-manifest.json").unlink(missing_ok=True)
            write_json(output / "failure.json", {"analysis_valid": False, "first_error": str(error), **FLAGS})
            raise
    return summary


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
        analyze(freeze["inputs"], args.output)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
