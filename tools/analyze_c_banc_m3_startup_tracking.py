"""CPU-only analysis of saved Original tracking captures, after finite Admission."""
# ruff: noqa: BLE001 -- preserve a bounded failure report and immutable input hashes.
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import zipfile

ENTRY = time.perf_counter()
SCHEMA = "flylab.m3.startup-tracking"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
STARTS = (100, 164, 228, 292)
DRIVE_ENDS = tuple(c + j for c in STARTS for j in range(1, 5))


def require(ok, label):
    if not ok:
        raise ValueError(label)


class SavedInputs:
    """Selected numeric members only, explicit ZIP/header bounds, pre/post digest."""
    def __init__(self, admission, np):
        self.q, self.np, self.ledger, self.expanded = admission, np, [], 0

    def digest(self, descriptor):
        from tools.banc_m3_startup_tracking_admission import path
        self.q.check("science")
        p = path(descriptor["path"], self.q.root)
        st = p.stat()
        require(0 < st.st_size == descriptor["bytes"] <= self.q.limits["input_file_cap_bytes"], "bounded saved input bytes")
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                self.q.check("science")
                h.update(chunk)
        require(h.hexdigest() == descriptor["sha256"] and p.stat().st_mtime_ns == st.st_mtime_ns, "saved Original input digest")
        return p, st.st_mtime_ns

    def json(self, descriptor):
        from tools.banc_m3_startup_tracking_admission import json_load
        p, stamp = self.digest(descriptor)
        require(descriptor["bytes"] <= 16 << 20, "bounded saved JSON")
        result = json_load(p.read_bytes())
        self.ledger.append((descriptor, stamp))
        return result

    def arrays(self, descriptor, fields):
        np = self.np
        self.q.fence("pre_array")
        p, stamp = self.digest(descriptor)
        result = {}
        with zipfile.ZipFile(p) as archive:
            infos = archive.infolist()
            names = [v.filename for v in infos]
            require(len(names) == len(set(names)) <= 80 and all(v.endswith(".npy") and "/" not in v and "\\" not in v for v in names), "finite flat numeric archive")
            for name, (shape, dtype) in fields.items():
                self.q.check("science")
                member = name + ".npy"
                require(member in names, "required selected member " + name)
                info = archive.getinfo(member)
                require(not info.flag_bits & 1 and info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), "numeric compression only")
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    require(version in ((1, 0), (2, 0)), "bounded NPY version")
                    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                    actual_shape, fortran, actual_dtype = reader(stream, max_header_size=4096)
                    require(not fortran and actual_dtype == np.dtype(dtype) and actual_dtype.kind in "bifu"
                            and len(actual_shape) == len(shape) and all(a == b or b is None for a, b in zip(actual_shape, shape, strict=True)), "selected shape/dtype " + name)
                    require(all(type(v) is int and v >= 0 for v in actual_shape), "finite selected dimensions")
                    expected = math.prod(actual_shape) * actual_dtype.itemsize
                    require(info.file_size == stream.tell() + expected and expected <= 16 << 20, "exact selected NPY size")
                self.expanded += expected
                require(self.expanded <= self.q.limits["selected_expanded_cap_bytes"], "combined selected expanded input cap")
                with archive.open(member) as stream:
                    value = np.lib.format.read_array(stream, allow_pickle=False, max_header_size=4096)
                require(value.nbytes == expected, "selected saved byte count " + name)
                if name != "slip_mm_s":
                    require(np.isfinite(value).all(), "finite selected saved values " + name)
                value.flags.writeable = False
                result[name] = value
        if "slip_mm_s" in result:
            validate_native_slip(result, np)
        self.ledger.append((descriptor, stamp))
        self.digest(descriptor)
        return result

    def posthash(self):
        for spec, stamp in self.ledger:
            p, now = self.digest(spec)
            require(now == stamp and p.stat().st_size == spec["bytes"], "all saved Original inputs unchanged")
        return [{"path": d["path"], "bytes": d["bytes"], "sha256_before": d["sha256"], "sha256_after": d["sha256"], "unchanged": True} for d, _ in self.ledger]


CONTROL_FIELDS = {"control_tick": ((None,), "int64"), "network_tick": ((None,), "int64"),
                  "physics_time_s": ((None,), "float64"), "command": ((None, 42), "float64"), "tendon": ((None, 6), "float64")}
NATIVE_FIELDS = {"time_s": ((None,), "float64"), "normal_load_bw": ((None, 6), "float64"), "slip_mm_s": ((None, 6), "float64"),
                 "slip_valid": ((None, 6), "bool"), "nonfoot_contact": ((None,), "bool"), "upright": ((None,), "float64")}
TRACKING_FIELDS = {"boundary_control": ((None,), "int64"), "boundary_network": ((None,), "int64"),
                   "boundary_time_s": ((None,), "float64"), "boundary_valid": ((None,), "bool"),
                   "requested_target_rad": ((None, 42), "float64"), "actual_q_rad": ((None, 42), "float64")}
for _name, _shape, _dtype in (
    ("native_control_end", (None,), "int64"), ("native_substep", (None,), "int64"),
    ("native_pre_time_s", (None,), "float64"), ("native_post_time_s", (None,), "float64"),
    ("native_raw_ctrl_rad", (None, 42), "float64"), ("native_effective_ctrl_rad", (None, 42), "float64"),
    ("native_pre_q_rad", (None, 42), "float64"), ("native_post_q_rad", (None, 42), "float64"),
    ("native_started", (None,), "bool"), ("native_returned", (None,), "bool"),
    ("native_post_observed", (None,), "bool"), ("native_effective_verified", (None,), "bool"),
):
    TRACKING_FIELDS[_name] = (_shape, _dtype)
CONTEXT_FIELDS = {
    "pre_control": ((4,), "int64"), "pre_network": ((4,), "int64"), "pre_time_s": ((4,), "float64"), "pre_valid": ((4,), "bool"),
    "rate": ((4, 158706), "float32"), "adaptation": ((4, 158706), "float32"), "bank": ((4, 391, 3), "float64"),
    "original_activation": ((4, 96), "float64"), "resource": ((4, 7136), "float64"), "u_zero": ((4,), "bool"), "resource_one": ((4,), "bool"),
    "drive_control_end": ((16,), "int64"), "drive_network_pre": ((16,), "int64"), "drive_time_pre_s": ((16,), "float64"),
    "drive_valid": ((16,), "bool"), "drive": ((16, 158706), "float32"), "source_ids": ((7136,), "int64"),
    "utilization": ((7136,), "float64"), "recovery_tau_s": ((7136,), "float64"), "motor_ids": ((391,), "int64"), "output_mask": ((158706,), "float32"),
}


def stats(value, np):
    v = np.asarray(value, np.float64)
    require(np.isfinite(v).all(), "statistics receive finite selected values only")
    return {"count": int(v.size), "rms": float(np.sqrt(np.mean(v * v))) if v.size else None,
            "max_abs": float(np.max(np.abs(v))) if v.size else None}


def compare_values(a, b, np):
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"comparable": False, "a_shape": list(a.shape), "b_shape": list(b.shape), "a_dtype": str(a.dtype), "b_dtype": str(b.dtype)}
    return {"comparable": True, "exact": bool(np.array_equal(a, b)), **stats(a.astype(np.float64) - b.astype(np.float64), np)}


def validate_native_slip(raw, np):
    slip, valid = raw["slip_mm_s"], raw["slip_valid"]
    require(slip.shape == valid.shape and slip.dtype == np.float64 and valid.dtype == np.bool_, "native slip/valid shape and dtype")
    require(not np.isinf(slip).any() and np.isfinite(slip[valid]).all(), "valid native slip is finite; only invalid NaN is allowed")
    return {"valid_count": int(valid.sum()), "invalid_count": int((~valid).sum()), "invalid_nan_count": int(np.isnan(slip).sum())}


def compare_native_slip(a, b, np):
    va, vb = validate_native_slip(a, np), validate_native_slip(b, np)
    x, y = a["slip_mm_s"], b["slip_mm_s"]
    if x.shape != y.shape:
        return {"comparable": False, "a_shape": list(x.shape), "b_shape": list(y.shape), "a_validity": va, "b_validity": vb}
    finite_pair = np.isfinite(x) & np.isfinite(y)
    common_valid = a["slip_valid"] & b["slip_valid"]
    return {"comparable": True, "exact": bool(np.array_equal(x, y, equal_nan=True)),
            "validity_mask_exact": bool(np.array_equal(a["slip_valid"], b["slip_valid"])),
            "nan_position_exact": bool(np.array_equal(np.isnan(x), np.isnan(y))),
            "finite_pair_values": stats(x[finite_pair] - y[finite_pair], np),
            "both_valid_values": stats(x[common_valid] - y[common_valid], np),
            "a_validity": va, "b_validity": vb, "invalid_values_replaced_by_zero": False}


def compare_trace(a, b, np):
    """Every common lean numeric/bool leaf, including feet/basis/clocks; no zip claim."""
    mismatches, numeric = [], []

    def visit(x, y, label):
        if type(x) is dict and type(y) is dict:
            if set(x) != set(y):
                mismatches.append(label + ":keys")
            for key in sorted(set(x) & set(y)):
                visit(x[key], y[key], label + "." + key)
        elif type(x) is list and type(y) is list:
            if len(x) != len(y):
                mismatches.append(label + ":length")
            for i, (xx, yy) in enumerate(zip(x, y)):
                visit(xx, yy, label + "." + str(i))
        elif type(x) in (int, float, bool) and type(y) is type(x):
            require(math.isfinite(float(x)) and math.isfinite(float(y)), "finite trace leaf")
            numeric.append(float(x) - float(y))
            if x != y and len(mismatches) < 32:
                mismatches.append(label)
        elif type(x) is not type(y) or x != y:
            if len(mismatches) < 32:
                mismatches.append(label)

    visit(a, b, "trace")
    return {"exact": a == b, "numeric_leaf_comparison": stats(numeric, np), "first_mismatch_paths": mismatches[:32],
            "numeric_leaf_rows_are_independent_trials": False}


def validate_capture(control, native, tracking, context, report, np):
    """No assertion that a partial failed arm is a complete observation."""
    errors = []

    def check(ok, label):
        if not ok:
            errors.append(label)

    n, m = len(control["control_tick"]), len(tracking["native_started"])
    check(1 <= n <= 301 and 0 <= m <= 15000, "bounded rows/native count")
    check(all(len(v) == n for v in control.values()), "control lengths")
    check(all(len(v) == (m if k.startswith("native_") else n) for k, v in tracking.items()), "tracking lengths")
    check(np.array_equal(control["control_tick"], np.arange(n)), "ordered boundary0..N")
    check(np.array_equal(control["network_tick"], 20 * control["control_tick"]), "network boundary clocks")
    check(np.all(np.abs(control["physics_time_s"] - .005 * control["control_tick"]) <= 2e-8), "physics boundary clocks")
    check(np.array_equal(tracking["boundary_control"], control["control_tick"])
          and np.array_equal(tracking["boundary_network"], control["network_tick"])
          and np.array_equal(tracking["boundary_time_s"], control["physics_time_s"]), "same capture boundary clocks")
    check(tracking["boundary_valid"].all() and np.array_equal(tracking["requested_target_rad"], control["command"]), "raw requested target binding")
    check(np.array_equal(tracking["native_control_end"], np.arange(m) // 50 + 1)
          and np.array_equal(tracking["native_substep"], np.arange(m) % 50), "native control_end/substep")
    returned = tracking["native_returned"] & tracking["native_post_observed"]
    check(tracking["native_started"].all(), "native started bitmap")
    check(np.all(np.abs(tracking["native_pre_time_s"] - np.arange(m) * .0001) <= 2e-8), "native pre clocks")
    check(np.all(np.abs(tracking["native_post_time_s"][returned] - tracking["native_pre_time_s"][returned] - .0001) <= 2e-8), "native post clocks")
    k = len(native["time_s"])
    check(all(len(v) == k for v in native.values()), "native contact lengths")
    check(k <= m and np.all(np.abs(native["time_s"] - tracking["native_post_time_s"][:k]) <= 2e-8), "postcontact/native post clock join")
    check(np.array_equal(context["pre_control"], STARTS) and np.array_equal(context["drive_control_end"], DRIVE_ENDS), "fixed windows/drives")
    pre, drive = context["pre_valid"], context["drive_valid"]
    check(np.array_equal(context["pre_network"][pre], 20 * context["pre_control"][pre])
          and np.all(np.abs(context["pre_time_s"][pre] - .005 * context["pre_control"][pre]) <= 2e-8), "c+1 pre-state clocks")
    check(np.array_equal(context["drive_network_pre"][drive], 20 * (context["drive_control_end"][drive] - 1))
          and np.all(np.abs(context["drive_time_pre_s"][drive] - .005 * (context["drive_control_end"][drive] - 1)) <= 2e-8), "actual finalized drive clocks")
    check(np.all(context["utilization"] == 0) and np.all(context["recovery_tau_s"] > 0)
          and context["u_zero"][pre].all() and context["resource_one"][pre].all()
          and np.all(context["resource"][pre] == 1), "actual u0/resource1 proof")
    check(len(np.unique(context["source_ids"])) == 7136 and np.all((context["source_ids"] >= 0) & (context["source_ids"] < 158706))
          and len(np.unique(context["motor_ids"])) == 391 and np.all((context["motor_ids"] >= 0) & (context["motor_ids"] < 158706))
          and np.isin(context["output_mask"], (0, 1)).all(), "ordered neural leaves")
    check(all(report.get("context_array_sha256", {}).get(key) == hashlib.sha256(value.tobytes(order="C")).hexdigest()
              for key, value in context.items()), "saved context finite array hashes")
    if report.get("run_valid") is True:
        work = report.get("counters", {})
        expected = {"controls_requested": n - 1, "controls_host_returned": n - 1, "controls_completed": n - 1,
                    "neural_requested": 20 * (n - 1), "neural_host_returned": 20 * (n - 1), "neural_device_confirmed": 20 * (n - 1),
                    "physics_requested": 50 * (n - 1), "physics_host_returned": 50 * (n - 1), "physics_device_confirmed": 50 * (n - 1),
                    "observer_samples": 50 * (n - 1), "completed_rows": n}
        check(all(type(work.get(k)) is int and work[k] == v for k, v in expected.items())
              and work.get("neural_unreturned_unknown") is False and work.get("physics_unreturned_unknown") is False,
              "actual requested/returned/confirmed work")
        built = report.get("construction", {})
        constructor = {"body_constructors_requested": 1, "body_constructors_returned": 1,
                       "settling_physics_requested": 2000, "settling_physics_host_returned": 2000,
                       "neural_setup_requested": 320, "neural_setup_host_returned": 320,
                       "neural_capture_requested": 160, "neural_capture_recorded": 160}
        check(all(type(built.get(k)) is int and built[k] == v for k, v in constructor.items())
              and built.get("unknown_work") is False and built.get("device_confirmed") is True, "preserved actual constructor work")
    complete = bool(n == 301 and m == k == 15000 and returned.all() and pre.all() and drive.all())
    check(report.get("capture_complete") is complete, "reported versus measured completeness")
    run_valid = bool(report.get("run_valid") is True and report.get("all_owned_ended") is True and not errors)
    return {"run_valid": run_valid, "capture_complete": complete and run_valid, "raw_capture_complete": complete,
            "errors": errors, "boundary_rows": n, "native_started": m, "native_postcontact_rows": k,
            "native_known_returned": int(returned.sum()), "pre_complete_bitmap": pre.tolist(), "drive_complete_bitmap": drive.tolist(),
            "first_error": report.get("first_error"), "secondary_errors": report.get("secondary_errors", [])}


def common_lags(requested, actual, ticks, np, *, lower=100, upper=299):
    """command[t] versus measured[t+lag], same t intersection for every lag."""
    lookup = {int(t): i for i, t in enumerate(ticks)}
    base = [t for t in range(lower, upper + 1) if t in lookup and all(t + lag <= upper and t + lag in lookup for lag in range(5))]
    result = []
    for lag in range(5):
        left = np.asarray([lookup[t] for t in base], np.int64)
        right = np.asarray([lookup[t + lag] for t in base], np.int64)
        error = requested[left] - actual[right]
        result.append({"lag_controls": lag, "lag_ms": lag * 5, "pair_count": len(base), "scalar_denominator": int(error.size),
                       "error_rad": stats(error, np), "per_joint_rms_rad": np.sqrt(np.mean(error * error, axis=0)).tolist() if len(base) else None})
    return {"definition": "request boundary t versus measured boundary t+lag", "common_base_ticks": base,
            "window": [lower, upper], "includes_outside_window": False, "independent_trials": False,
            "lag_selection_is_causal_or_acceptance": False, "lags": result}


def endpoint_states(pre, post, np):
    return {"loaded_both": pre & post, "unloaded_both": ~pre & ~post, "transition": pre ^ post}


def control_pairs(ticks, np):
    physical = np.flatnonzero((ticks >= 100) & (ticks <= 299) & (ticks > 0))
    require(all(ticks[i - 1] == ticks[i] - 1 for i in physical), "physical pair preceding boundary exists")
    strict = np.asarray([i for i in physical if ticks[i - 1] >= 100], np.int64)
    return {"physical200_end100..299": physical, "both_in_window199_end101..299": strict}


def native_window_and_support(native, np):
    return ((native["time_s"] >= .5) & (native["time_s"] < 1.5), native["normal_load_bw"] >= .02)


def tracking_summary(control, native, tracking, trace, report, np):
    metadata = report["tracking_metadata"]
    names = metadata["joint_names"]
    require(len(names) == 42 and len(set(names)) == 42, "ordered joint names")
    from tools.observe_c_banc_m3_startup_tracking import leg_joint_indices
    require(metadata["leg_order"] == list(LEGS) and set(metadata["leg_joint_indices"]) == set(LEGS), "actual lower-case leg order")
    groups = leg_joint_indices(np.asarray([metadata["leg_joint_indices"][leg] for leg in LEGS], np.int64), names, np)
    ticks = tracking["boundary_control"]
    window = (ticks >= 100) & (ticks <= 299) & tracking["boundary_valid"]
    request, actual = tracking["requested_target_rad"][window], tracking["actual_q_rad"][window]
    error = request - actual
    per_joint = []
    actuator_range = np.asarray(metadata["actuator_ctrlrange_rad"], np.float64)
    actuator_limited = np.asarray(metadata["actuator_ctrllimited"], bool)
    joint_range = np.asarray(metadata["joint_range_rad"], np.float64)
    joint_limited = np.asarray(metadata["joint_limited"], bool)
    ctrl_outside = ((request < actuator_range[:, 0]) | (request > actuator_range[:, 1])) & actuator_limited
    constraint_outside = ((request < joint_range[:, 0]) | (request > joint_range[:, 1])) & joint_limited
    for j, name in enumerate(names):
        per_joint.append({"name": name, "requested_minus_measured_rad": stats(error[:, j], np),
                          "requested_p95_p05_span_rad": float(np.diff(np.percentile(request[:, j], [5, 95]))[0]) if len(request) else None,
                          "actual_p95_p05_span_rad": float(np.diff(np.percentile(actual[:, j], [5, 95]))[0]) if len(actual) else None,
                          "request_outside_ctrlrange_count": int(ctrl_outside[:, j].sum()),
                          "request_outside_joint_constraint_range_count": int(constraint_outside[:, j].sum())})
    control_status = {name: {} for name in ("loaded_both", "unloaded_both", "transition")}
    rows = trace["samples"]
    require(len(rows) == len(ticks) and all(row["control_tick"] == int(t) for row, t in zip(rows, ticks, strict=True)), "control trace exact join")
    loaded = np.asarray([row["feet"]["load_bearing"] for row in rows], bool)
    pairs_by_window = control_pairs(ticks, np)
    for leg, ids in groups.items():
        li = LEGS.index(leg)
        for label, pairs in pairs_by_window.items():
            states = endpoint_states(loaded[pairs - 1, li], loaded[pairs, li], np)
            for state, mask in states.items():
                control_status[state].setdefault(leg, {})[label] = {
                    "pair_count": int(mask.sum()), "requested_minus_post_measured_rad": stats((tracking["requested_target_rad"][pairs] - tracking["actual_q_rad"][pairs])[mask][:, ids], np),
                    "requested_delta_rad": stats((tracking["requested_target_rad"][pairs] - tracking["requested_target_rad"][pairs - 1])[mask][:, ids], np),
                    "actual_delta_rad": stats((tracking["actual_q_rad"][pairs] - tracking["actual_q_rad"][pairs - 1])[mask][:, ids], np),
                }
    m = min(len(native["time_s"]), len(tracking["native_started"]))
    native_window, all_postload = native_window_and_support(native, np)
    native_window = native_window[:m].copy()
    native_window &= tracking["native_returned"][:m] & tracking["native_post_observed"][:m]
    raw, preq, postq = (tracking[k][:m][native_window] for k in ("native_raw_ctrl_rad", "native_pre_q_rad", "native_post_q_rad"))
    verified = tracking["native_effective_verified"][:m][native_window]
    mapping_verified = metadata.get("ctrl_to_q_mapping", {}).get("status") == "VERIFIED_UNIT_GEAR_JOINT_HINGE"
    effective = tracking["native_effective_ctrl_rad"][:m][native_window]
    native_groups = {}
    postload = all_postload[:m]
    initial = np.asarray(metadata["initial_native_contact_load_bw"], np.float64) >= .02
    preload = np.vstack((initial[None], postload[:-1])) if m else np.zeros((0, 6), bool)
    for leg, ids in groups.items():
        li = LEGS.index(leg)
        states = endpoint_states(preload[native_window, li], postload[native_window, li], np)
        native_groups[leg] = {
            state: {"native_interval_count": int(mask.sum()), "raw_ctrl_minus_pre_q_rad": stats((raw - preq)[mask][:, ids], np) if mapping_verified else None,
                    "raw_ctrl_minus_post_q_rad": stats((raw - postq)[mask][:, ids], np) if mapping_verified else None,
                    "derived_effective_minus_post_q_rad": stats((effective - postq)[mask & verified][:, ids], np) if verified.any() and mapping_verified else None}
            for state, mask in states.items()
        }
    ap = {}
    for leg in LEGS:
        li = LEGS.index(leg)
        bodyfeet = np.asarray([(np.asarray(row["body"]["legs"][leg][-1]) - np.asarray(row["body"]["position"])) @ np.asarray(row["body"]["basis"]) for row in rows], np.float64)
        for label, pairs in pairs_by_window.items():
            delta = bodyfeet[pairs, 0] - bodyfeet[pairs - 1, 0]
            states = endpoint_states(loaded[pairs - 1, li], loaded[pairs, li], np)
            ap.setdefault(leg, {})[label] = {state: {"pair_count": int(mask.sum()), "body_frame_AP_sum_mm": float(delta[mask].sum()),
                                                     "body_frame_AP_delta_mm": stats(delta[mask], np)} for state, mask in states.items()}
    return {
        "official_boundary_window": [100, 299], "observed_boundary_count": int(window.sum()), "boundary0_excluded": True,
        "requested_minus_post_measured_rad": stats(error, np), "per_joint": per_joint,
        "per_leg": {leg: stats(error[:, ids], np) for leg, ids in groups.items()},
        "request_reachability": {"outside_ctrlrange_scalar_count": int(ctrl_outside.sum()), "outside_joint_constraint_scalar_count": int(constraint_outside.sum()),
                                 "joint_constraint_is_actuator_clamp": False, "large_raw_error_alone_proves_transfer_failure": False},
        "native_tracking": {"known_interval_count": int(native_window.sum()), "raw_ctrl_minus_pre_q_rad": stats(raw - preq, np) if mapping_verified else None,
                            "raw_ctrl_minus_post_q_rad": stats(raw - postq, np) if mapping_verified else None,
                            "derived_effective_minus_post_q_rad": stats((effective - postq)[verified], np) if verified.any() and mapping_verified else None,
                            "ctrl_to_q_mapping": metadata.get("ctrl_to_q_mapping"),
                            "effective_verified_interval_count": int(verified.sum()),
                            "clipped_scalar_count": int(np.count_nonzero(raw[verified] != effective[verified])) if verified.any() else None,
                            "clamp": metadata["control_clamp"], "interpolation": metadata["interpolation"], "activation_tau_s": metadata["activation_tau_s"]},
        "lag": common_lags(tracking["requested_target_rad"], tracking["actual_q_rad"], ticks, np),
        "control_contact_association": control_status, "native_contact_association": native_groups,
        "contact_definitions": {"control": "recorded feet.load_bearing boundary bool", "native": "recorded post-step normal_load_bw >= .02; pre is previous post or initial boundary0",
                                "control_pairs": "physical200 ends100..299 (includes99->100); strict199 ends101..299; neither is original temporal canonical200 ends101..300",
                                "native_window": "actual post-contact time_s .5 <= t < 1.5, preserving existing supported-slip window",
                                "control_interval_and_native_post_time_windows_identical": False,
                                "native_pre_q_equals_post_contact_time": False, "transitions_separate": True},
        "actual_body_frame_foot_AP": ap, "command_to_foot": "UNAVAILABLE",
        "interpretation": "관절 rad 추종·기록된 발 AP의 시간 연관이다. geometric knee degrees와 직접 차감하지 않는다. 힘/토크/인과/독립 반복 증거가 아니다.",
    }


def markdown(result):
    lines = ["# 저장 실제 관절 추종 진단", "", "기각 alpha .0001의 관측 진단입니다. 후보 채택 false, F1–F8 INCOMPLETE, 생물학적 기전 UNPROVEN입니다.", ""]
    for role, arm in result.get("arms", {}).items():
        valid, tracking = arm["capture_validity"], arm.get("tracking", {})
        lines += [f"- {role}: run_valid={valid['run_valid']}, capture_complete={valid['capture_complete']}; boundary {valid['boundary_rows']}, native {valid['native_started']}.",
                  f"  requested−measured RMS/max(rad): {tracking.get('requested_minus_post_measured_rad', {}).get('rms')} / {tracking.get('requested_minus_post_measured_rad', {}).get('max_abs')}.",
                  f"  baseline 공통 field exact={arm['baseline_comparison']['all_common_values_exact']}; mismatch는 새 관측 trajectory입니다."]
    lines += ["", "command→foot 기하는 UNAVAILABLE입니다. H1/H2 원인은 미식별이며 limits·activation·contact·신경 타이밍이 모두 양립합니다.",
              "lag0..4는 같은 pair/분모의 설명적 비교입니다. qualified cycles0은 다리가 움직이지 않았다는 뜻이 아닙니다.",
              "후속 우선 경로: 실제 한도·추종·접촉 전달 경로의 단위와 순서를 검토하고, 신경 방향 변경은 이 자료의 해석 이후 결정합니다."]
    return "\n".join(lines) + "\n"


def worker(admission):
    require(admission.role == "tracking_diagnosis" and admission.stage == "pre_array", "saved Original CPU only")
    admission.fence("pre_array")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(os.environ.get(name) == "1", "CPU thread1 before numerical import")
    import numpy as np

    from tools.observe_c_banc_m3_startup_tracking import bounded_publisher
    saved, publisher = SavedInputs(admission, np), bounded_publisher(admission)
    result = {"schema": SCHEMA + ".diagnosis.v1", "run_valid": False, "capture_complete": False,
              "arms": {}, "hypothesis_status": "unidentified", "candidate_adopted": False, "F1_F8": "INCOMPLETE",
              "biology": "UNPROVEN", "M2_UI": "PRESERVED", "offline_replay": 0, "gradient": 0, "body_work": 0,
              "command_to_foot": "UNAVAILABLE", "first_error": None, "secondary_errors": []}
    try:
        artifacts = admission.request["artifacts"]
        require(set(artifacts) == {"tracking_neutral", "tracking_candidate"}, "two saved arm bundles")
        for role, historic in (("tracking_neutral", "neutral_body"), ("tracking_candidate", "candidate_body")):
            admission.check("science")
            bundle = artifacts[role]
            require(set(bundle) == {"manifest", "body_report", "control", "native", "trace", "tracking", "context"}, "finite saved bundle")
            manifest = saved.json(bundle["manifest"])
            require(manifest["schema"] == SCHEMA + ".capture-manifest.v1" and manifest["role"] == role, "capture manifest role")
            require(manifest["historical_manifest"] == admission.manifest_spec and manifest["roster"] == admission.request["roster"], "same historical manifest/current roster")
            require(all(manifest["artifacts"][k] == d for k, d in bundle.items() if k != "manifest"), "manifest exact artifact binding")
            report = saved.json(bundle["body_report"])
            control, native = saved.arrays(bundle["control"], CONTROL_FIELDS), saved.arrays(bundle["native"], NATIVE_FIELDS)
            tracking, context = saved.arrays(bundle["tracking"], TRACKING_FIELDS), saved.arrays(bundle["context"], CONTEXT_FIELDS)
            trace = saved.json(bundle["trace"])
            old = admission.roster["historical_arms"][historic]
            old_control, old_native = saved.arrays(old["control.npz"], CONTROL_FIELDS), saved.arrays(old["native.npz"], NATIVE_FIELDS)
            old_trace = saved.json(old["trace.json"])
            comparisons = {"control": {k: compare_values(control[k], old_control[k], np) for k in CONTROL_FIELDS},
                           "native": {k: compare_native_slip(native, old_native, np) if k == "slip_mm_s" else compare_values(native[k], old_native[k], np) for k in NATIVE_FIELDS},
                           "lean_trace": compare_trace(trace, old_trace, np)}
            exact = all(v.get("exact") is True for group in (comparisons["control"], comparisons["native"]) for v in group.values()) and comparisons["lean_trace"]["exact"]
            comparison = {"all_common_values_exact": exact, "fields": comparisons, "native_drift": comparisons["native"],
                          "trajectory_interpretation": "common saved fields exact" if exact else "new observed trajectory; no posthoc tolerance or retry",
                          "new_actual_q_drive_context_historically_observed": False, "archive_hash_is_trajectory_equality": False}
            validity = validate_capture(control, native, tracking, context, report, np)
            arm = {"capture_validity": validity, "baseline_comparison": comparison,
                   "startup": report.get("startup"), "metrics": report.get("metrics"), "gates": report.get("gates"),
                   "candidate_adopted": False, "tracking": tracking_summary(control, native, tracking, trace, report, np)}
            result["arms"][role] = arm
            del control, native, tracking, context, old_control, old_native, trace, old_trace
        result["input_pre_post_hashes"] = saved.posthash()
        result["run_valid"] = all(a["capture_validity"]["run_valid"] for a in result["arms"].values())
        result["capture_complete"] = all(a["capture_validity"]["capture_complete"] for a in result["arms"].values())
        result["baseline_comparison"] = {role: a["baseline_comparison"]["all_common_values_exact"] for role, a in result["arms"].items()}
        result["hypothesis_status"] = "unidentified"
        result["hypothesis_limits"] = "H1 command→foot direction is unmeasured. H2 tracking/clamp/contact association alone is compatible, not causal. H3 mixed/unidentified remains permitted."
        result["next_development_path"] = "단위·관절 순서·한도/activation/contact 전달 경로를 saved 실제 추종과 대조; 새 신경 방향 설계는 ROOT의 해석 뒤에 결정"
    except Exception as error:
        result["first_error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            result["input_pre_post_hashes"] = saved.posthash()
        except Exception as secondary:
            result["secondary_errors"].append({"stage": "posthash", "type": type(secondary).__name__, "message": str(secondary)})
        publisher.json("report.json", result)
        publisher.write("report.md", markdown(result).encode("utf8"))
        raise
    result["output_usage_before_report"] = publisher.usage()
    publisher.json("report.json", result)
    publisher.write("report.md", markdown(result).encode("utf8"))
    require(result["run_valid"], "saved technical validity failed; partial diagnostic preserved")
    return result


def main(argv=None):
    from tools.banc_m3_startup_tracking_admission import Admission
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--pre-spawn", action="store_true")
    args = parser.parse_args(argv)
    entry = ENTRY if args.pre_spawn else float(json.loads(os.environ["FLYLAB_DIRECT_SCIENCE_OWNER"])["entry_counter_s"])
    admission = Admission(args.request, stage="pre_spawn" if args.pre_spawn else "pre_array", entry=entry)
    if args.pre_spawn:
        print(json.dumps({"pre_spawn": True, "role": admission.role}))
        return 0
    result = worker(admission)
    print(json.dumps({"run_valid": result["run_valid"], "capture_complete": result["capture_complete"], "hypothesis_status": result["hypothesis_status"], "candidate_adopted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
