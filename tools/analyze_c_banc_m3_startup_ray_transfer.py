"""Saved CPU association diagnostics. Source-only R1 proposal; not a gait gate.

Actual roles enter through --request and ROOT's pre-array Admission. --roster is
a local reader API, carrying no Original admission, supervisor or cost authority.
No neural replay, measured42 reconstruction, body constructor or stepping occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

ENTRY_S0 = time.perf_counter()
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
WINDOWS = (100, 164, 228, 292)
MIB = 1024 ** 2
CAPS = {"file": 96 * MIB, "member": 16 * MIB, "selected": 64 * MIB,
        "host": 2 * 1024 ** 3, "output": 16 * MIB}
SOURCE_PINS = {
    "tools/analyze_c_banc_biological_gait.py": "bcceb5d1e9f26293ffc800d09a1fbf063a1ab784d73cf5489b74c57ecdd14760",
    "tools/banc_m3_smooth_temporal.py": "3d9b893564766bff3bdb433a69149a7a80955c0bfe44793848591e2ac43cc44e",
    "tools/analyze_c_banc_m3_temporal_objective.py": "b7614c6fd2b8671d615eb2d557e86110e974626cf1b2b32091601adaa8cb6d48",
    "tools/banc_m3_phase_free_temporal.py": "a1e2e0d3032407ea4c1d38afee305a2da0039a6eb639e10794a09353ba18421e",
}
ROLES = {arm + "/" + name for arm in ("neutral_body", "candidate_body")
         for name in ("control.npz", "native.npz", "trace.json", "body-report.json", "evaluation.npz")}
ROLES |= {"ray/ray-raw.npz", "ray/ray-record.json", "ray/receipts.json"}
CONTROL = {"control_tick": ((301,), "int64"), "network_tick": ((301,), "int64"),
           "physics_time_s": ((301,), "float64"), "command": ((301, 42), "float64"),
           "tendon": ((301, 6), "float64")}
NATIVE = {"time_s": ((15000,), "float64"), "normal_load_bw": ((15000, 6), "float64"),
          "slip_mm_s": ((15000, 6), "float64"), "slip_valid": ((15000, 6), "bool"),
          "nonfoot_contact": ((15000,), "bool"), "upright": ((15000,), "float64")}
EVALUATION = {"reference.curves": ((6, 128, 7), "float64"),
              "reference.normalization": ((6,), "float64"),
              "reference.temperature": ((6,), "float64"),
              "reference.future_anchors": ((6, 128, 7), "float64"),
              "reference.phase_basis": ((128, 2), "float64"),
              "startup.target_command": ((99, 42), "float64"),
              "startup.target_tendon": ((99, 6), "float64")}
RAY = {"normalization": ((6,), "float64"), "target_q": ((4, 4, 42), "float64"),
       "window_starts": ((4,), "int64"), "q_neutral": ((4, 4, 42), "float64"),
       "R_neutral": ((4,), "float64"), "alpha-4.q_candidate": ((4, 4, 42), "float64")}
CADENCE = {"command": ((301, 42), "float64"), "control_tick": ((301,), "int64"),
           "control_time_s": ((301,), "float64")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def json_read(path):
    def nonfinite(value):
        raise ValueError("nonfinite JSON: " + value)
    def finite_float(value):
        number = float(value)
        require(math.isfinite(number), "nonfinite JSON float")
        return number
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=nonfinite, parse_float=finite_float)


def digest(path, check):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            check("hash")
            h.update(block)
    return h.hexdigest()


def beneath(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "relative input path")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "input outside root")
    return path


class Reader:
    """Only the thirteen supplied saved files plus cadence; no directory scanner."""
    def __init__(self, roster_path, root, check):
        self.root, self.check = Path(root).resolve(), check
        self.roster_path = Path(roster_path).resolve()
        require(self.roster_path.stat().st_size < MIB, "small roster required")
        self.roster_hash = digest(self.roster_path, check)
        self.roster = json_read(self.roster_path)
        require(self.roster.get("schema") == "flylab.m3.after-startup-ray.saved-roster.v1"
                and set(self.roster["files"]) == ROLES, "fixed thirteen-file roster")
        require(self.roster["window"] == {"start_s": .5, "end_exclusive_s": 1.5}, "fixed official window")
        self.specs = dict(self.roster["files"])
        self.specs["cadence"] = self.roster["cadence_label"]
        self.pins, self.members, self.expanded = {}, {}, 0
        for key, spec in self.specs.items():
            require(type(spec["bytes"]) is int and 0 < spec["bytes"] <= CAPS["file"], "input file cap")
            path = beneath(self.root, spec["path"])
            require(path.stat().st_size == spec["bytes"] and digest(path, check) == spec["sha256"], "input prehash: " + key)
            self.pins[key] = {"path": spec["path"], "bytes": spec["bytes"], "sha256": spec["sha256"]}
        self.source_pins = {}
        for relative, expected in SOURCE_PINS.items():
            path = beneath(self.root, relative)
            require(digest(path, check) == expected, "pure helper source drift: " + relative)
            self.source_pins[relative] = {"bytes": path.stat().st_size, "sha256": expected}

    def path(self, key):
        return beneath(self.root, self.specs[key]["path"])

    def metadata(self, key):
        self.check("json")
        return json_read(self.path(key))

    def numeric(self, key, schema, np):
        self.check("selected_members")
        result, ledger = {}, {}
        with zipfile.ZipFile(self.path(key)) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), "duplicate NPZ members")
            for member, (shape, dtype) in schema.items():
                self.check("member")
                info = archive.getinfo(member + ".npy")
                require(0 < info.file_size <= CAPS["member"], "selected member cap")
                self.expanded += info.file_size
                require(self.expanded <= CAPS["selected"], "selected expansion cap")
                raw = archive.read(info)
                header = io.BytesIO(raw)
                version = np.lib.format.read_magic(header)
                require(version in ((1, 0), (2, 0)), "supported small numeric NPY header")
                header_reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                 else np.lib.format.read_array_header_2_0)
                actual_shape, fortran, actual_dtype = header_reader(header, max_header_size=4096)
                require(actual_shape == shape and actual_dtype == np.dtype(dtype)
                        and not actual_dtype.hasobject and not fortran
                        and header.tell() + math.prod(shape) * actual_dtype.itemsize == len(raw),
                        "selected header/payload size before allocation")
                value = np.load(io.BytesIO(raw), allow_pickle=False)
                require(isinstance(value, np.ndarray) and value.shape == shape
                        and value.dtype == np.dtype(dtype) and value.dtype.kind in "biuf"
                        and value.nbytes <= CAPS["member"], "numeric schema: " + key + "/" + member)
                # Native unavailable slip is explicitly allowed, never coerced to zero.
                require(not np.isinf(value).any() if member == "slip_mm_s" else np.isfinite(value).all(),
                        "nonfinite selected values: " + member)
                result[member] = value
                ledger[member] = {"shape": list(shape), "dtype": dtype, "expanded_bytes": info.file_size,
                                  "c_order_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
        self.members[key] = ledger
        return result

    def posthash(self):
        for key, spec in self.specs.items():
            path = self.path(key)
            require(path.stat().st_size == spec["bytes"] and digest(path, self.check) == spec["sha256"], "input posthash: " + key)
            self.pins[key]["posthash_verified"] = True
        require(digest(self.roster_path, self.check) == self.roster_hash, "roster drift")
        for relative, expected in SOURCE_PINS.items():
            require(digest(beneath(self.root, relative), self.check) == expected, "helper posthash: " + relative)


def r_windows(command, np):
    require(command.shape == (301, 42), "R complete command301 required")
    return np.stack([command[c + 1:c + 5] for c in WINDOWS])


def r_score(command_windows, target, normalization, np):
    require(command_windows.shape == target.shape == (4, 4, 42)
            and normalization.shape == (6,) and np.all(normalization > 0), "R shapes/positive normalization")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = np.sum((command_windows - target) ** 2 / np.repeat(normalization, 7), axis=(1, 2)) / (16 * 42)
    require(np.isfinite(result).all(), "finite R")
    return result


def pair_slices(command, strict=False):
    # Recorded controls include boundary0: canonical ends at boundary300 (1.5s).
    stop = 299 if strict else 300
    return command[100:stop], command[101:stop + 1]


def percentiles(values, np):
    values = np.asarray(values)
    require(np.isfinite(values).all(), "finite summary")
    return None if not values.size else {"p05": float(np.percentile(values, 5)),
                                        "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def interval_ap(ap, contact, window, np):
    require(ap.shape == contact.shape and contact.dtype == np.bool_, "AP/contact schema")
    delta = np.diff(ap, axis=0)
    inside = window[:-1] & window[1:]
    loaded = contact[:-1] & contact[1:]
    unloaded = ~contact[:-1] & ~contact[1:]
    transition = contact[:-1] != contact[1:]
    result = {}
    for i, leg in enumerate(LEGS):
        result[leg] = {}
        for name, mask in (("loaded", loaded), ("unloaded", unloaded), ("transition", transition)):
            value = delta[inside & mask[:, i], i]
            result[leg][name] = {"intervals": int(value.size), "sum_leg_lengths": float(value.sum()),
                                 "delta_leg_lengths": percentiles(value, np)}
    return result


def slip_summary(native, np):
    t, load, slip, valid = (native[k] for k in ("time_s", "normal_load_bw", "slip_mm_s", "slip_valid"))
    selected = (t >= .5) & (t < 1.5)
    supported = selected[:, None] & (load >= .02)
    def one(mask):
        eligible = mask & valid & np.isfinite(slip)
        count, good = int(mask.sum()), int(eligible.sum())
        denominator, valid_denominator = float(load[mask].sum()), float(load[eligible].sum())
        available = count > 0 and good == count and denominator > 0
        value = float(np.sum(load[mask] * slip[mask]) / denominator) if available else None
        return {"available": available, "value_mm_s": value,
                "reason": None if available else ("no_supported_samples" if count == 0 else "invalid_supported_slip"),
                "supported_count": count, "valid_supported_count": good, "invalid_supported_count": count - good,
                "load_weight_denominator_bw": denominator, "valid_load_weight_denominator_bw": valid_denominator}
    return {
        "all": one(supported), "per_leg": {leg: one(supported & (np.arange(6)[None] == i))
        for i, leg in enumerate(LEGS)}, "native_window_rows": int(selected.sum()),
        "unit": "native world contact slip mm/s, weighted by normal_load_bw; not force or body AP"}


def temporal_summary(objective, command, torch, np, check):
    summary = {}
    for name, strict in (("canonical200", False), ("strict199", True)):
        check("temporal")
        previous, successor = pair_slices(command, strict)
        with torch.no_grad():
            result = objective.evaluate_pairs(torch.as_tensor(previous, dtype=torch.float64, device="cpu"),
                                              torch.as_tensor(successor, dtype=torch.float64, device="cpu"))
        def host(key, result=result):
            return result[key].detach().cpu().numpy()
        item = {key: float(host(key)) for key in ("training_loss", "successor_loss", "coordination_loss", "lr_coordination_loss")}
        item.update(pairs=len(previous), previous_controls=[100, 298 if strict else 299],
                    successor_controls=[101, 299 if strict else 300],
                    per_leg_successor_loss=dict(zip(LEGS, host("per_leg_successor_loss").tolist())),
                    per_leg_successor_mse_rad2=dict(zip(LEGS, host("per_leg_successor_mse").tolist())),
                    coordination_pair_loss=host("coordination_pair_loss").tolist())
        item["coordination_pair_order"] = [LEGS[a] + "-" + LEGS[b] for a in range(6) for b in range(a + 1, 6)]
        item["confidence"] = {}
        for side in ("previous", "next"):
            encoding = result[side + "_encoding"]
            item["confidence"][side] = {key: {leg: percentiles(encoding[key].detach().cpu().numpy()[:, i], np)
                for i, leg in enumerate(LEGS)} for key in ("R", "entropy", "maxweight")}
        summary[name] = item
    return summary


def geometry_summary(trace, data, np):
    rows = trace["samples"]
    basis = np.asarray([r["body"]["basis"] for r in rows], dtype=np.float64)
    origin = np.asarray([r["body"]["position"] for r in rows], dtype=np.float64)
    points = np.asarray([[r["body"]["legs"][leg] for leg in LEGS] for r in rows], dtype=np.float64)
    require(points.shape == (301, 6, 8, 3) and np.isfinite(points).all()
            and np.isfinite(basis).all() and np.isfinite(origin).all(), "native eight-landmark geometry")
    require(np.allclose(np.linalg.det(basis), 1, atol=1e-9, rtol=0), "right handed recorded basis")
    ego = np.einsum("tlpj,tjk->tlpk", points - origin[:, None, None], basis)
    window, lengths = data["window"], data["lengths"]
    require(np.isfinite(lengths).all() and np.all(lengths > 0), "positive leg lengths")
    result = {"interval_foot_AP": interval_ap(data["ap"], data["contact"], window, np),
              "leg_lengths_mm": dict(zip(LEGS, lengths.tolist())), "per_leg": {}}
    for i, leg in enumerate(LEGS):
        knee = percentiles(data["knee"][window, i], np)
        result["per_leg"][leg] = {"AP_leg_lengths": percentiles(data["ap"][window, i], np),
            "height_body_up_leg_lengths": percentiles(ego[window, i, -1, 2] / lengths[i], np),
            "height_world_y_mm": percentiles(points[window, i, -1, 1], np),
            "knee_interior_deg": knee, "knee_p95_minus_p05_deg": knee["p95"] - knee["p05"],
            "contact_fraction": float(data["contact"][window, i].mean()),
            "contact_transitions": int(np.count_nonzero(np.diff(data["contact"][window, i])))}
    support = data["contact"][window].sum(axis=1)
    result["simultaneous_support_counts"] = {str(n): int(np.count_nonzero(support == n)) for n in range(7)}
    result["units"] = "Recorded mm/world-y vs body-basis up; AP/height normalized by median leg length; knee interior degrees, never command radians"
    return result


def validate_receipts(receipts, record):
    keys = tuple(record["work"])
    totals = dict.fromkeys(keys, 0)
    for receipt in receipts.values():
        for key in keys:
            require(type(receipt[key]) is int and receipt[key] >= 0, "typed actual ray receipt count")
            totals[key] += receipt[key]
        require(type(receipt["confirm_requested"]) is int and receipt["confirm_requested"] == receipt["confirm_returned"] > 0,
                "actual returned confirmation")
        for name in ("forward_controls", "forward_neural_ticks", "recompute_controls", "recompute_neural_ticks", "backward_blocks", "backward_block_ticks"):
            require(receipt["confirmed"][name] == receipt[name + "_returned"], "device-confirmed receipt count")
    require(totals == record["work"], "actual receipt sums differ")
    for c in WINDOWS:
        for key in ("R-neutral-" + str(c), "alpha-4-R-" + str(c), "alpha-4-deployed-R-" + str(c)):
            require(receipts[key]["forward_controls_returned"] == 4 and receipts[key]["forward_neural_ticks_returned"] == 80,
                    "four-control R receipt alignment")
    return {"receipt_count": len(receipts), "actual_historical_counts": totals,
            "window_receipts": {str(c): ["R-neutral-" + str(c), "alpha-4-R-" + str(c), "alpha-4-deployed-R-" + str(c)] for c in WINDOWS},
            "meaning": "historical raw receipt counts, no new neural work or replay claim"}


def analyze(roster, root, check=lambda label="science": None):
    """CPU-only saved-reader entry; the caller supplies Original admission/time fencing."""
    reader = Reader(roster, root, check)
    check("pre_array")
    # Imports intentionally confined to this explicit saved CPU call.
    import numpy as np
    import torch

    from tools import analyze_c_banc_biological_gait as gait
    from tools.banc_m3_smooth_temporal import SmoothTemporalObjective
    torch.set_num_threads(1)
    arms, controls, traces, evaluations = {}, {}, {}, {}
    report = {"schema": "flylab.m3.startup-ray-transfer.saved-cpu.v1", "status": "technical-invalid",
              "candidate_adopted": False, "F1_F8": "INCOMPLETE", "tracking_42D": "UNOBSERVED",
              "live_neural_drive_state": "UNOBSERVED", "causal_status": "unidentified",
              "statistical_unit": "two saved arm trajectories, time rows are dependent observations",
              "scope": {"cpu_array_math": True, "tensor_device": "cpu", "grad_context": "no_grad",
                        "neural_replay": "NOT_PERFORMED", "model_body_constructor": "NOT_CALLED_BY_READER",
                        "cuda_api": "NOT_CALLED_BY_READER", "imported_runtime_side_effects_independently_audited": False,
                        "limits": CAPS, "host_cap_enforcement": "ROOT supervisor, no reader RSS certification",
                        "supervision_cost_once": "ROOT caller responsibility"}}
    try:
        ticks = np.arange(301, dtype=np.int64)
        cadence = reader.numeric("cadence", CADENCE, np)
        require(np.array_equal(cadence["control_tick"], ticks)
                and np.array_equal(cadence["control_time_s"], cadence["control_tick"] * .005),
                "cadence exact label clock, same as original align_label")
        report["clocks"] = {"cadence": {"rows": 301, "controls": [0, 300], "dt_s": .005,
                            "first_s": float(cadence["control_time_s"][0]), "last_s": float(cadence["control_time_s"][-1]),
                            "matching_rule": "np.array_equal(control_time_s, control_tick * .005); exact, no physics tolerance"},
                            "official_window": {"rows": 200, "controls": [100, 299], "start_s": .5, "end_exclusive_s": 1.5},
                            "arms": {}}
        for arm in ("neutral_body", "candidate_body"):
            check("arm")
            control, native = reader.numeric(arm + "/control.npz", CONTROL, np), reader.numeric(arm + "/native.npz", NATIVE, np)
            evaluation, trace = reader.numeric(arm + "/evaluation.npz", EVALUATION, np), reader.metadata(arm + "/trace.json")
            body_report = reader.metadata(arm + "/body-report.json")
            require(np.array_equal(control["control_tick"], ticks) and np.array_equal(control["network_tick"], ticks * 20)
                    and np.allclose(control["physics_time_s"], ticks * .005, atol=2e-8, rtol=0), "control clocks")
            require(np.allclose(native["time_s"], np.arange(1, 15001) * .0001, atol=2e-8, rtol=0)
                    and np.all(native["normal_load_bw"] >= 0), "native complete substep clock/load")
            require(len(trace["samples"]) == 301 and all(type(row["control_tick"]) is int and row["control_tick"] == i
                    and abs(row["time_s"] - .005 * i) <= 2e-8 for i, row in enumerate(trace["samples"])), "trace complete clock")
            require(all(len(row["feet"]["load_bearing"]) == 6 and all(type(v) is bool for v in row["feet"]["load_bearing"])
                        and math.isfinite(row["upright"]) and math.isfinite(row["body"]["yaw"])
                        for row in trace["samples"]), "trace contact/yaw/upright schema")
            require(trace["landmarks"] == {"scope": "actual_native", "count": 8,
                    "order": ["coxa", "trochanterfemur", "tibia", "tarsus1", "tarsus2", "tarsus3", "tarsus4", "tarsus5"]}, "landmark order")
            report["clocks"]["arms"][arm] = {"control_trace_rows": 301, "network_ticks_per_control": 20,
                "physics_substeps": 15000, "physics_dt_s": .0001, "controls": [0, 300],
                "trace_control_native_clocks_verified": True, "control_time_tolerance_s": 2e-8}
            data = gait.load_model(reader.path(arm + "/trace.json"))
            official = (ticks >= 100) & (ticks < 300)
            require(np.array_equal(data["window"], official), "official full-trace smoothed window100..299")
            gait_report, _, _ = gait.analyze(data)
            with torch.no_grad():
                objective = SmoothTemporalObjective(evaluation["reference.curves"], evaluation["startup.target_command"],
                    evaluation["startup.target_tendon"], device="cpu", normalization=evaluation["reference.normalization"])
            require(objective.device.type == "cpu", "explicit CPU objective")
            for key, actual in objective.canonical_constants.items():
                require(np.array_equal(actual, evaluation["reference." + key]), "same saved temporal constant: " + key)
            temporal = temporal_summary(objective, control["command"], torch, np, check)
            saved_loss = body_report["metrics"]["smooth"]["training_loss"]
            require(math.isclose(temporal["canonical200"]["training_loss"], saved_loss, abs_tol=1e-10, rel_tol=1e-10), "canonical score/body report correspondence")
            q = control["command"][official].reshape(200, 6, 7)
            forward_runs = gait.runs(data["moving"])
            arms[arm] = {"geometry": geometry_summary(trace, data, np), "native_slip": slip_summary(native, np),
                "speed_mm_s": percentiles(data["speed"][official], np),
                "speed_sign_counts": {name: int(mask.sum()) for name, mask in (("positive", data["speed"][official] > 0),
                    ("zero", data["speed"][official] == 0), ("negative", data["speed"][official] < 0))},
                "forward_bouts": {"count": len(forward_runs), "durations_s": [(hi - lo - 1) * .005 for lo, hi in forward_runs],
                                  "eligible": gait_report["window"]["bouts"], "longest_s": gait_report["window"]["longest_forward_bout_s"]},
                "qualified_cycles": {leg: gait_report["events"][leg]["complete_cycles"] for leg in LEGS},
                "command_rad": {leg: {"p95_minus_p05_per_joint": (np.percentile(q[:, i], 95, axis=0) - np.percentile(q[:, i], 5, axis=0)).tolist(),
                    "interval_rms_change_rad": float(np.sqrt(np.mean(np.diff(q[:, i], axis=0) ** 2)))} for i, leg in enumerate(LEGS)},
                "temporal": temporal, "saved_body_summary": body_report["metrics"]["summary"],
                "saved_classification": body_report["classification"], "startup": body_report["startup"],
                "source_identity": {k: trace[k] for k in ("body_hash", "model_hash", "adapter_hash", "graph_hash")}}
            startup_error = control["command"][1:100] - evaluation["startup.target_command"]
            tendon_error = control["tendon"][1:100] - evaluation["startup.target_tendon"]
            startup = {"q_rms": float(np.sqrt(np.mean(startup_error ** 2))), "q_max": float(np.abs(startup_error).max()),
                       "tendon_rms": float(np.sqrt(np.mean(tendon_error ** 2))), "tendon_max": float(np.abs(tendon_error).max())}
            thresholds = {"q_rms": .018, "q_max": .04, "tendon_rms": .009, "tendon_max": .025}
            arms[arm]["startup_control99"] = {"actual": startup, "unchanged_thresholds": thresholds,
                                            "margin": {key: thresholds[key] - value for key, value in startup.items()}}
            require(body_report["run_valid"] is True and body_report["classification"] == "scientific_rejected", "saved rejected body provenance")
            controls[arm], traces[arm], evaluations[arm] = control, trace, evaluation
        for key in ("body_hash", "adapter_hash", "graph_hash", "parameters", "landmarks"):
            require(traces["neutral_body"][key] == traces["candidate_body"][key], "same arm provenance: " + key)
        for key in EVALUATION:
            require(np.array_equal(evaluations["neutral_body"][key], evaluations["candidate_body"][key]), "same evaluation reference/startup")
        raw = reader.numeric("ray/ray-raw.npz", RAY, np)
        record, receipts = reader.metadata("ray/ray-record.json"), reader.metadata("ray/receipts.json")
        require(record["selected"] == 4 and record["records"][4]["alpha"] == .0001, "fixed selected rejected alpha")
        require(np.array_equal(raw["window_starts"], WINDOWS)
                and np.array_equal(raw["target_q"], r_windows(cadence["command"], np))
                and np.array_equal(raw["normalization"], evaluations["neutral_body"]["reference.normalization"]), "same cadence label/normalization")
        saved = {"neutral_body": r_score(raw["q_neutral"], raw["target_q"], raw["normalization"], np),
                 "candidate_body": r_score(raw["alpha-4.q_candidate"], raw["target_q"], raw["normalization"], np)}
        live = {arm: r_score(r_windows(control["command"], np), raw["target_q"], raw["normalization"], np) for arm, control in controls.items()}
        require(np.allclose(saved["neutral_body"], raw["R_neutral"], atol=1e-12, rtol=1e-10)
                and np.allclose(saved["candidate_body"] - saved["neutral_body"], record["records"][4]["analysis"]["O"], atol=1e-12, rtol=1e-10), "saved signed R/O arithmetic")
        report["R"] = {"window_controls": {str(c): list(range(c + 1, c + 5)) for c in WINDOWS}, "denominator_per_window": 672,
            "saved_per_window": {arm: value.tolist() for arm, value in saved.items()}, "live_per_window": {arm: value.tolist() for arm, value in live.items()},
            "saved_O_candidate_minus_neutral": (saved["candidate_body"] - saved["neutral_body"]).tolist(),
            "live_O_candidate_minus_neutral": (live["candidate_body"] - live["neutral_body"]).tolist(),
            "saved_total": {arm: float(value.sum()) for arm, value in saved.items()}, "live_total": {arm: float(value.sum()) for arm, value in live.items()},
            "live_minus_saved": {arm: (live[arm] - saved[arm]).tolist() for arm in arms},
            "live_command_minus_saved_window_rms_rad": {arm: np.sqrt(np.mean((r_windows(controls[arm]["command"], np)
                - raw["q_neutral" if arm == "neutral_body" else "alpha-4.q_candidate"]) ** 2, axis=(1, 2))).tolist() for arm in arms},
            "startup_saved_ray": {"neutral": record["neutral_startup"], "candidate": record["records"][4]["startup"]},
            "interpretation": "same label/controls; live command association only, neural replay/state/drive UNOBSERVED"}
        report["receipts"] = validate_receipts(receipts, record)
        report["arms"] = arms
        neutral, candidate = arms["neutral_body"], arms["candidate_body"]
        ns, cs = neutral["saved_body_summary"], candidate["saved_body_summary"]
        slip_n, slip_c = neutral["native_slip"]["all"]["value_mm_s"], candidate["native_slip"]["all"]["value_mm_s"]
        report["delta_candidate_minus_neutral"] = {"forward_mm": cs["forward_mm"] - ns["forward_mm"],
            "native_loaded_slip_mm_s": None if slip_n is None or slip_c is None else slip_c - slip_n,
            "speed_percentiles_mm_s": {k: candidate["speed_mm_s"][k] - neutral["speed_mm_s"][k] for k in ("p05", "median", "p95")},
            "per_leg": {leg: {"knee_span_deg": candidate["geometry"]["per_leg"][leg]["knee_p95_minus_p05_deg"] - neutral["geometry"]["per_leg"][leg]["knee_p95_minus_p05_deg"],
                "command_interval_rms_rad": candidate["command_rad"][leg]["interval_rms_change_rad"] - neutral["command_rad"][leg]["interval_rms_change_rad"]} for leg in LEGS},
            "temporal": {name: {key: candidate["temporal"][name][key] - neutral["temporal"][name][key]
                for key in ("training_loss", "successor_loss", "coordination_loss")} for name in ("canonical200", "strict199")}}
        delta = report["delta_candidate_minus_neutral"]
        delta["forward_bouts"] = {"count": candidate["forward_bouts"]["count"] - neutral["forward_bouts"]["count"],
                                  "longest_s": candidate["forward_bouts"]["longest_s"] - neutral["forward_bouts"]["longest_s"]}
        delta["simultaneous_support_counts"] = {str(n): candidate["geometry"]["simultaneous_support_counts"][str(n)]
                - neutral["geometry"]["simultaneous_support_counts"][str(n)] for n in range(7)}
        for leg in LEGS:
            geometry_n, geometry_c = neutral["geometry"]["per_leg"][leg], candidate["geometry"]["per_leg"][leg]
            item = delta["per_leg"][leg]
            for key in ("AP_leg_lengths", "height_body_up_leg_lengths", "height_world_y_mm", "knee_interior_deg"):
                item[key] = {p: geometry_c[key][p] - geometry_n[key][p] for p in ("p05", "median", "p95")}
            item["contact_fraction"] = geometry_c["contact_fraction"] - geometry_n["contact_fraction"]
            item["contact_transitions"] = geometry_c["contact_transitions"] - geometry_n["contact_transitions"]
            item["interval_foot_AP"] = {}
            for kind in ("loaded", "unloaded", "transition"):
                a, b = (arm["geometry"]["interval_foot_AP"][leg][kind] for arm in (neutral, candidate))
                item["interval_foot_AP"][kind] = {"intervals": b["intervals"] - a["intervals"],
                    "sum_leg_lengths": b["sum_leg_lengths"] - a["sum_leg_lengths"],
                    "delta_leg_lengths": None if a["delta_leg_lengths"] is None or b["delta_leg_lengths"] is None else
                    {p: b["delta_leg_lengths"][p] - a["delta_leg_lengths"][p] for p in ("p05", "median", "p95")}}
            a, b = (arm["native_slip"]["per_leg"][leg]["value_mm_s"] for arm in (neutral, candidate))
            item["native_loaded_slip_mm_s"] = None if a is None or b is None else b - a
        report["observed_association"] = {"classification": "mixed" if slip_n is not None and slip_c is not None and slip_c > slip_n else "unidentified",
            "command_space": "per-window signed loss/coordination association, compatible with H1; no causal selection",
            "contact": "native supported slip association compatible with contact part of H2" if slip_n is not None and slip_c is not None else "unavailable",
            "tracking_part_H2": "UNOBSERVED", "propulsive_force_torque": "UNOBSERVED"}
        report["limits"] = ["qualifiedcycles0 follows no >=0.3s eligible forward bout; transitions/oscillations/AP peaks are not steps",
            "temporal fixed points can be stationary; lower command-space surrogate loss is not propulsion/gait",
            "saved trajectories cannot identify H1/H2 causality or biological mechanism; thresholds unchanged",
            "next observation, if needed: neutral/rejected candidate <=1 each; synchronized pre/post applied42/actual42; four-window state/drive only if needed; ROOT fixes caps before admission"]
        report["status"] = "saved_diagnostic_complete"
    finally:
        reader.posthash()
    report["input_pins"] = reader.pins
    report["roster_sha256"] = reader.roster_hash
    report["source_pins"] = reader.source_pins
    report["selected_members"] = reader.members
    report["selected_expanded_bytes"] = reader.expanded
    return report


def publish(output, report):
    output = Path(output).resolve()
    raw = (json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    def number(value, signed=False):
        return "유효값 없음" if value is None else format(value, "+.9g" if signed else ".9g")
    status = "저장 CPU 진단 완료" if report["status"] == "saved_diagnostic_complete" else "기술적 무효·진단 미완료"
    text = "# 저장 startup-ray transfer 진단\n\n상태: " + status + ". alpha0.0001 후보는 기각 상태를 유지하며 F1–F8은 미완료입니다.\n\n"
    if "R" in report and "arms" in report:
        r = report["R"]
        text += "같은 cadence label과 control 경계를 비교했습니다. ΔR은 후보−중립이며 양수는 악화, 음수는 개선입니다. 각 창의 분모는 16×42=672입니다.\n\n"
        text += "| 창의 실제 controls | 저장 중립 R | 저장 후보 R | 저장 ΔR | 실제 명령 중립 R | 실제 명령 후보 R | 실제 명령 ΔR |\n"
        text += "|---|---:|---:|---:|---:|---:|---:|\n"
        for i, c in enumerate(WINDOWS):
            text += "| " + str(c + 1) + ".." + str(c + 4) + " | " + " | ".join([
                number(r["saved_per_window"]["neutral_body"][i]), number(r["saved_per_window"]["candidate_body"][i]),
                number(r["saved_O_candidate_minus_neutral"][i], True), number(r["live_per_window"]["neutral_body"][i]),
                number(r["live_per_window"]["candidate_body"][i]), number(r["live_O_candidate_minus_neutral"][i], True)]) + " |\n"
        text += "| 합계 | " + " | ".join([number(r["saved_total"]["neutral_body"]), number(r["saved_total"]["candidate_body"]),
            number(r["saved_total"]["candidate_body"] - r["saved_total"]["neutral_body"], True), number(r["live_total"]["neutral_body"]),
            number(r["live_total"]["candidate_body"]), number(r["live_total"]["candidate_body"] - r["live_total"]["neutral_body"], True)]) + " |\n\n"
        n, c = report["arms"]["neutral_body"], report["arms"]["candidate_body"]
        physical = [("forward 변위 (mm)", n["saved_body_summary"]["forward_mm"], c["saved_body_summary"]["forward_mm"]),
                    ("지지 하중 가중 native slip (mm/s)", n["native_slip"]["all"]["value_mm_s"], c["native_slip"]["all"]["value_mm_s"]),
                    ("최장 연속 forward bout (s)", n["forward_bouts"]["longest_s"], c["forward_bouts"]["longest_s"])]
        text += "공식 물리 창은 control100..299, 0.5≤t<1.5초입니다. slip은 load≥0.02BW·valid·전체 하중 분모를 유지하며, 유효값이 없으면 0으로 대체하지 않습니다.\n\n"
        text += "| 물리 관측값 | 중립 | 후보 | 후보−중립 |\n|---|---:|---:|---:|\n"
        for label, a, b in physical:
            text += "| " + label + " | " + number(a) + " | " + number(b) + " | " + number(None if a is None or b is None else b - a, True) + " |\n"
        text += "\n| temporal 명령 공간 진단 | 중립 | 후보 | 후보−중립 |\n|---|---:|---:|---:|\n"
        for name, label in (("canonical200", "기존 200쌍"), ("strict199", "양 끝 창내 199쌍")):
            for key, metric in (("successor_loss", "successor"), ("coordination_loss", "coordination")):
                a, b = n["temporal"][name][key], c["temporal"][name][key]
                text += "| " + label + " " + metric + " | " + number(a) + " | " + number(b) + " | " + number(b - a, True) + " |\n"
        text += "\n기존 200쌍은 100→101부터299→300까지로 1.5초 successor를 포함합니다. 별도 199쌍은 두 끝 모두 공식 창 안에 있습니다. temporal loss에는 정지 fixed point가 가능하므로 낮은 값이 추진력이나 보행 성공을 뜻하지 않습니다.\n\n"
    text += "실제 관절42 추종과 live neural state/drive는 **미관측(UNOBSERVED)**이며 인과는 **미식별**입니다. 저장 R와 실제 명령 R의 차이는 신경 replay 증거가 아닙니다. 기하학적 knee 각도와 command radians를 직접 차감하지 않습니다.\n\n"
    text += "qualified cycle 0은 0.3초 이상 eligible forward bout가 없다는 분석 결과입니다. 접촉 전환·명령 진동·발 AP peaks를 steps로 해석하지 않습니다. 두 저장 trajectory의 시간행은 독립 반복 표본이 아닙니다. 임계값과 기존 후보 기각 판정은 유지됩니다.\n"
    md = text.encode("utf-8")
    require(len(raw) + len(md) <= CAPS["output"], "output cap")
    output.mkdir(parents=True, exist_ok=False)
    for name, content in (("report.json", raw), ("report.md", md)):
        with (output / name).open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--request", type=Path)
    group.add_argument("--roster", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--max-seconds", type=float, default=100)
    args = parser.parse_args()
    require(math.isfinite(args.max_seconds) and 0 < args.max_seconds <= 100, "child cap <=100s")
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    admission = None
    if args.request:
        from tools.banc_m3_saved_original_admission import Admission
        admission = Admission(args.request.as_posix(), stage="pre_array", root=root, entry=ENTRY_S0)
        roster, output, root = admission.roster_path, admission.output, admission.root
        require(args.output is None or args.output.resolve() == Path(output).resolve(), "output differs from admission")
    else:
        require(args.output is not None, "local reader requires --output")
        roster, output = args.roster, args.output
    def check(label="science"):
        require(time.perf_counter() - ENTRY_S0 < args.max_seconds, "child CPU deadline exceeded")
        if admission is not None:
            admission.check(label)
    try:
        report = analyze(roster, root, check)
        report["admission_scope"] = "ROOT saved-Original role" if admission else "local reader; Original role authority unverified"
        check("publication")
        publish(output, report)
        return 0
    except Exception as error:  # noqa: BLE001 -- record technical-invalid outcome for ROOT supervisor.
        failure = {"status": "technical-invalid", "candidate_adopted": False, "F1_F8": "INCOMPLETE",
                   "error": {"type": type(error).__name__, "message": str(error)[:1000]}}
        if not Path(output).exists():
            publish(output, failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
