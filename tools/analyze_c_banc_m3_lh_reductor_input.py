"""E253 saved-only LH femur reductor decoder and existing-input audit."""

# Preserve primary analysis, cleanup, and publication failures independently.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
FREEZE_SHA = "5bae8584c1b8b6178aee61b837860b2ae2f04a19102378185810c3ad21bfb9a2"
SCHEMA = "flylab.m3.e253.lh-reductor-input"
OWNED = ("tools/analyze_c_banc_m3_lh_reductor_input.py",
         "tests/test_c_banc_lh_reductor_input.py")
MOTOR_COLUMNS = (13, 44, 128, 296, 343)
MOTOR_IDS = (6373, 22787, 52774, 121544, 139521)
BRIDGE_IDS = (20156, 63026, 111970, 152280)
WINDOWS = {"startup": (1, 100), "steady": (100, 300), "full": (0, 301)}
LIMITS = {"internal_s": 30, "finalization_reserve_s": 5, "external_s": 40,
          "supervisor_s": 45, "cpu_threads": 1, "gpu_count": 0,
          "original_attempts": 1, "retries": 0, "body_constructors": 0,
          "neural_constructors": 0, "neural_ticks": 0, "physics_steps": 0,
          "fit_calls": 0, "solver_calls": 0, "resident_bytes": 2147483648,
          "output_bytes": 67108864, "minimum_free_bytes": 536870912}
DECODE_FILES = {"output_mask": "array_00002.npy", "tau": "array_00012.npy",
                "kappa": "array_00013.npy", "slopes": "array_00014.npy",
                "membership": "array_00015.npy",
                "fixed_muscles": "array_00016.npy"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def save_json(path: Path, value: object) -> None:
    path.write_bytes(json_bytes(value))


def pinned(relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Repository-relative slash path required")
    name = Path(relative)
    if name.is_absolute() or any(part in (".", "..") for part in name.parts):
        raise ValueError("Unsafe frozen path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("Pinned path missing or outside repository")
    return path


def gate(freeze_path: str, freeze_sha: str, review_path: str,
         review_sha: str) -> tuple[dict, dict]:
    if freeze_sha != FREEZE_SHA or sha(Path(freeze_path)) != FREEZE_SHA:
        raise ValueError("E253 freeze hash mismatch")
    if sha(Path(review_path)) != review_sha:
        raise ValueError("E253 review hash mismatch")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != SCHEMA + ".freeze.v1"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("resource_limits") != LIMITS
            or freeze.get("controls") != 300
            or freeze.get("control_dt_s") != .005
            or freeze.get("n") != 158706
            or freeze.get("neural_ticks_per_control") != 20
            or freeze.get("muscle_index") != 41
            or freeze.get("joint_column") != 17
            or freeze.get("motor_columns") != list(MOTOR_COLUMNS)
            or freeze.get("motor_ids") != list(MOTOR_IDS)
            or freeze.get("bridge_ids") != list(BRIDGE_IDS)
            or freeze.get("observed_count") != 418
            or freeze.get("windows") != {key: list(value) for key, value in WINDOWS.items()}
            or freeze.get("case_controls") != [64, 84]
            or freeze.get("concentration_fraction") != .8
            or freeze.get("tie_break") != "ascending graph row ID"):
        raise ValueError("E253 frozen contract differs")
    if (review.get("schema") != SCHEMA + ".review.v1"
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or review.get("source_sha256") != {name: sha(pinned(name)) for name in OWNED}):
        raise ValueError("E253 approved source review required")
    if sha(pinned(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E253 design hash differs")
    for mapping in (freeze["input_sha256"], freeze["reused_source_sha256"],
                    freeze["preservation_sha256"],
                    review.get("additional_input_sha256", {}),
                    review.get("additional_reused_source_sha256", {})):
        if not isinstance(mapping, dict):
            raise TypeError("E253 hash map required")
        for name, expected in mapping.items():
            if sha(pinned(name)) != expected:
                raise ValueError("E253 pinned byte changed: " + name)
    if freeze["packages"] != {name: importlib.metadata.version(name)
                              for name in freeze["packages"]}:
        raise ValueError("E253 frozen packages differ")
    if set(freeze["arms"]) != {"baseline", "candidate"}:
        raise ValueError("E253 arm roles differ")
    return freeze, review


def load_npz(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        if not set(keys).issubset(saved.files):
            raise ValueError("Missing NPZ keys: " + str(path))
        return {key: saved[key].copy() for key in keys}


def normalize_route(value: dict) -> dict:
    if not isinstance(value, dict):
        raise TypeError("E248 route JSON object required")
    if "selected" in value:
        if not isinstance(value.get("rule"), str) or not value["rule"]:
            raise ValueError("E248 route rule missing")
        value = value["selected"]
    if (not isinstance(value, dict) or value.get("port") != 85
            or value.get("kind") != "claw"
            or value.get("reachable_count") != 55
            or not all(key in value for key in
                       ("source_rows", "bridge_rows", "reachable_motor_rows"))):
        raise ValueError("E248 selected LH claw route differs")
    return value


def load_inputs(freeze: dict, progress=None) -> dict:
    inputs = {}
    for role, keys in {
        "calibration": ("motor_ids", "moment", "rest", "pool", "membership", "ltm"),
        "edge_patch": ("edge_ids", "pre_ids", "post_ids", "source_values",
                       "target_values"),
        "baseline_csr": ("indptr", "indices", "data"),
        "candidate_csr": ("indptr", "indices", "data"),
    }.items():
        inputs[role] = load_npz(pinned(freeze["paths"][role]), keys)
        if progress:
            progress(role)
    for role in ("muscles", "selection", "route_selection", "e252_output_review",
                 "e252_final_review", "e252_supervisor"):
        inputs[role] = json.loads(pinned(freeze["paths"][role]).read_text(encoding="utf-8"))
        if role == "route_selection":
            inputs[role] = normalize_route(inputs[role])
        if progress:
            progress(role)
    for arm, folder in freeze["arms"].items():
        data = {}
        for role, filename, keys in (
            ("activity", "activity.npz", ("observed_ids", "source_ids", "bridge_ids",
                                          "reachable_motor_ids", "rate")),
            ("control", "control-raw.npz", ("control_tick", "network_tick",
                                               "physics_time_s", "motor_rate32",
                                               "activation", "command")),
            ("recruitment", "native-recruitment-state.npz",
             ("control_tick", "network_tick", "physics_time_s", "complete",
              "bank_available", "activation_available", "motor_rate_available",
              "bank", "activation", "motor_rate"))):
            data[role] = load_npz(pinned(f"{folder}/{filename}"), keys)
            if progress:
                progress(arm + "/" + role)
        state_path = pinned(f"{folder}/observed-start/state.json")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        adapter = state["adapter"]["parameters"]
        if (state["network"]["output_mask"] != {"__npy__": DECODE_FILES["output_mask"]}
                or any(adapter[key] != {"__npy__": filename}
                       for key, filename in DECODE_FILES.items() if key != "output_mask")):
            raise ValueError("E251 decoder state array references differ")
        data["decoder"] = {key: np.load(pinned(f"{folder}/observed-start/{filename}"),
                                        allow_pickle=False)
                           for key, filename in DECODE_FILES.items()}
        if progress:
            progress(arm + "/decoder")
        inputs[arm] = data
    return inputs


def checked(value, shape: tuple, dtype: str | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or (dtype is not None and array.dtype != np.dtype(dtype)):
        raise ValueError(f"E253 shape/dtype differs: {array.shape}/{array.dtype}, {shape}/{dtype}")
    if array.dtype.kind in "fiu" and not np.isfinite(array).all():
        raise ValueError("E253 nonfinite observed value")
    return array


def validate(inputs: dict, *, rows=301, motors=391, muscles=96,
             neurons=158706, muscle_index=41, motor_columns=MOTOR_COLUMNS,
             motor_ids=MOTOR_IDS, bridge_ids=BRIDGE_IDS) -> dict:
    columns = np.asarray(motor_columns, np.int64)
    target_ids = np.asarray(motor_ids, np.int64)
    bridges = np.asarray(bridge_ids, np.int64)
    cal = inputs["calibration"]
    roster = checked(cal["motor_ids"], (motors,), "int64")
    membership = checked(cal["membership"], (motors,), "int64")
    checked(cal["pool"], (muscles, motors), "float64")
    checked(cal["moment"], (42, muscles), "float64")
    checked(cal["rest"], (42,), "float64")
    checked(cal["ltm"], (6,), "int64")
    if (len(np.unique(roster)) != motors or not np.array_equal(roster, np.sort(roster))
            or not np.array_equal(roster[columns], target_ids)
            or not np.array_equal(membership[columns],
                                  np.full(5, muscle_index))
            or np.any((membership < 0) | (membership >= muscles))
            or np.any(cal["pool"] < 0)
            or not np.allclose(cal["pool"].sum(axis=1), 1, rtol=0, atol=1e-12)
            or not np.array_equal(np.argmax(cal["pool"], axis=0), membership)
            or not np.array_equal((cal["pool"] > 0).sum(axis=0), np.ones(motors))):
        raise ValueError("E253 original anatomical motor mapping differs")
    groups = inputs["muscles"]["muscles"]
    chosen = inputs["selection"]
    if (len(groups) != muscles or groups[muscle_index] != {
            "columns": columns.tolist(), "leg": "lh", "muscle": "femur_reductor_muscle"}
            or chosen.get("muscle_index") != muscle_index
            or chosen.get("muscle") != groups[muscle_index]
            or chosen.get("motor_ids") != target_ids.tolist()
            or chosen.get("bridge_ids") != bridges.tolist()):
        raise ValueError("E253 frozen whole-muscle selection differs")
    route = normalize_route(inputs["route_selection"])
    source = np.asarray(route["source_rows"], np.int64)
    reachable = np.asarray(route["reachable_motor_rows"], np.int64)
    if (source.shape != (23,) or reachable.shape != (55,)
            or not np.array_equal(route["bridge_rows"], bridges)
            or len(np.unique(np.r_[source, bridges, roster])) != 418
            or not np.isin(reachable, roster).all()):
        raise ValueError("E253 route source/bridge/MN union differs")
    observed = np.r_[source, bridges, roster]
    patch = inputs["edge_patch"]
    for name, dtype in (("edge_ids", "int64"), ("pre_ids", "int64"),
                        ("post_ids", "int64"), ("source_values", "float32"),
                        ("target_values", "float32")):
        checked(patch[name], (25,), dtype)
    if (len(np.unique(patch["edge_ids"])) != 25
            or not np.isin(patch["pre_ids"], source).all()
            or not np.isin(patch["post_ids"], bridges).all()):
        raise ValueError("E250 exact source-to-bridge patch differs")
    for arm, csr_name in (("baseline", "baseline_csr"),
                          ("candidate", "candidate_csr")):
        data = inputs[arm]
        activity, control, recruitment, decoder = (data[key] for key in
            ("activity", "control", "recruitment", "decoder"))
        if (not np.array_equal(activity["observed_ids"], np.r_[source, bridges, reachable])
                or not np.array_equal(activity["source_ids"], source)
                or not np.array_equal(activity["bridge_ids"], bridges)
                or not np.array_equal(activity["reachable_motor_ids"], reachable)):
            raise ValueError("E251 activity ID order differs")
        checked(activity["rate"], (rows, 82), "float32")
        for key, width, dtype in (("motor_rate32", motors, "float32"),
                                  ("activation", muscles, "float64"),
                                  ("command", 42, "float64")):
            checked(control[key], (rows, width), dtype)
        checked(recruitment["bank"], (rows, motors, 3), "float64")
        checked(recruitment["activation"], (rows, muscles), "float64")
        checked(recruitment["motor_rate"], (rows, motors), "float64")
        for key in ("complete", "bank_available", "activation_available",
                    "motor_rate_available"):
            if not checked(recruitment[key], (rows,), "bool").all():
                raise ValueError("Incomplete E251 recruitment rows")
        for key in ("control_tick", "network_tick"):
            checked(control[key], (rows,), "int64")
            checked(recruitment[key], (rows,), "int64")
        if (not np.array_equal(control["control_tick"], np.arange(rows))
                or not np.array_equal(control["network_tick"], np.arange(rows)*20)
                or not np.array_equal(recruitment["control_tick"], control["control_tick"])
                or not np.array_equal(recruitment["network_tick"], control["network_tick"])):
            raise ValueError("E251 endpoint control/neural clock differs")
        for key in ("physics_time_s",):
            checked(control[key], (rows,), "float64")
            checked(recruitment[key], (rows,), "float64")
            if (not np.allclose(control[key], np.arange(rows)*.005, rtol=0, atol=2e-8)
                    or not np.allclose(recruitment[key], control[key], rtol=0, atol=2e-8)):
                raise ValueError("E251 physics time alignment differs")
        if (not np.array_equal(control["activation"], recruitment["activation"])
                or not np.allclose(control["motor_rate32"], recruitment["motor_rate"],
                                   rtol=0, atol=1e-5)):
            raise ValueError("E251 duplicate control/recruitment values differ")
        index = {int(value): i for i, value in enumerate(roster)}
        reachable_columns = np.asarray([index[int(value)] for value in reachable])
        if not np.allclose(activity["rate"][:, 27:],
                           control["motor_rate32"][:, reachable_columns],
                           rtol=0, atol=1e-5):
            raise ValueError("E251 reachable motor activity differs from whole roster")
        checked(decoder["output_mask"], (neurons,), "float32")
        checked(decoder["tau"], (3,), "float64")
        checked(decoder["kappa"], (motors,), "float64")
        checked(decoder["slopes"], (muscles, motors, 3, 2), "float64")
        if not np.array_equal(checked(decoder["membership"], (motors,), "int64"),
                              membership):
            raise ValueError("E251 decoder membership differs")
        fixed = np.asarray(decoder["fixed_muscles"])
        if (fixed.dtype != np.int64 or fixed.ndim != 1 or muscle_index in fixed
                or not np.all((fixed >= 0) & (fixed < muscles))
                or len(np.unique(fixed)) != len(fixed)):
            raise ValueError("E253 target muscle must be variable")
        if (np.any(decoder["tau"] <= 0) or np.any(decoder["kappa"] < 0)
                or np.any(decoder["slopes"] < 0)
                or not np.isin(decoder["output_mask"], [0, 1]).all()):
            raise ValueError("E251 decoder parameters/mask invalid")
        matrix = inputs[csr_name]
        ptr = checked(matrix["indptr"], (neurons+1,), "int32")
        idx = np.asarray(matrix["indices"])
        weights = np.asarray(matrix["data"])
        if (idx.dtype != np.int32 or weights.dtype != np.float32
                or idx.shape != weights.shape or idx.ndim != 1
                or ptr[0] != 0 or ptr[-1] != len(idx)
                or np.any(np.diff(ptr) < 0) or np.any((idx < 0) | (idx >= neurons))
                or not np.isfinite(weights).all()):
            raise ValueError("Actual E251 CSR malformed")
        edge = patch["edge_ids"]
        if (np.any((edge < 0) | (edge >= len(idx)))
                or not np.array_equal(idx[edge], patch["pre_ids"])
                or not np.array_equal(np.searchsorted(ptr, edge, side="right")-1,
                                      patch["post_ids"])
                or not np.array_equal(weights[edge], patch[
                    "source_values" if arm == "baseline" else "target_values"])):
            raise ValueError("E250 restored 25-edge CSR provenance differs")
    for key in DECODE_FILES:
        if not np.array_equal(inputs["baseline"]["decoder"][key],
                              inputs["candidate"]["decoder"][key]):
            raise ValueError("E251 two arms decoder/mask differs")
    base = inputs["baseline_csr"]
    candidate = inputs["candidate_csr"]
    if (not np.array_equal(base["indptr"], candidate["indptr"])
            or not np.array_equal(base["indices"], candidate["indices"])):
        raise ValueError("E251 CSR structure changed")
    unchanged = np.ones(len(base["data"]), bool)
    unchanged[patch["edge_ids"]] = False
    if not np.array_equal(base["data"][unchanged], candidate["data"][unchanged]):
        raise ValueError("E251 CSR changed outside exact 25 restored edges")
    return {"motor_columns": columns, "motor_ids": target_ids,
            "source_ids": source, "bridge_ids": bridges, "observed_ids": observed,
            "membership": membership}


def _sum3(value: np.ndarray) -> dict:
    return {"signed": float(value.sum()),
            "positive": float(np.maximum(value, 0).sum()),
            "negative": float(np.minimum(value, 0).sum())}


def _concentration(positive: np.ndarray, ids: np.ndarray) -> dict:
    total = float(positive.sum())
    if total <= 0:
        return {"available": False, "total": total, "winner_id": None,
                "winner_fraction": None, "supported": False}
    order = np.lexsort((ids, -positive))
    winner = int(order[0])
    fraction = float(positive[winner] / total)
    return {"available": True, "total": total, "winner_index": winner,
            "winner_id": int(ids[winner]), "winner_fraction": fraction,
            "positive_by_id": {str(int(ids[i])): float(positive[i])
                               for i in range(len(ids))},
            "supported": bool(fraction >= .8)}


def analyze_arrays(inputs: dict, *, raw: dict | None = None, progress=None,
                   rows=301, motors=391, muscles=96, neurons=158706,
                   muscle_index=41, motor_columns=MOTOR_COLUMNS,
                   motor_ids=MOTOR_IDS, bridge_ids=BRIDGE_IDS) -> tuple[dict, dict]:
    """Analyze already loaded, fixed saved arrays; caller owns partial raw."""
    if raw is None:
        raw = {}
    mapping = validate(inputs, rows=rows, motors=motors, muscles=muscles,
                       neurons=neurons, muscle_index=muscle_index,
                       motor_columns=motor_columns, motor_ids=motor_ids,
                       bridge_ids=bridge_ids)
    raw.update({key: mapping[key] for key in
                ("motor_columns", "motor_ids", "source_ids", "bridge_ids",
                 "observed_ids", "membership")})
    baseline = inputs["baseline"]
    candidate = inputs["candidate"]
    decoder = baseline["decoder"]
    cols = mapping["motor_columns"]
    ids = mapping["motor_ids"]
    bridge = mapping["bridge_ids"]
    source = mapping["source_ids"]
    roster = inputs["calibration"]["motor_ids"]
    kappa = decoder["kappa"][cols]
    low = decoder["slopes"][muscle_index, cols, :, 0]
    high = decoder["slopes"][muscle_index, cols, :, 1]
    raw["kappa"] = kappa.copy()
    raw["tau"] = decoder["tau"].copy()
    raw["slope_low"] = low.copy()
    raw["slope_high"] = high.copy()
    raw["endpoint_control_tick"] = np.arange(rows, dtype=np.int64)
    raw["endpoint_time_s"] = baseline["control"]["physics_time_s"].copy()
    for arm in ("baseline", "candidate"):
        bank = inputs[arm]["recruitment"]["bank"][:, cols, :]
        hinge = np.maximum(bank - kappa[None, :, None], 0.)
        value = low[None, :, :]*bank + (high-low)[None, :, :]*hinge
        raw["bank_" + arm] = bank.copy()
        raw["hinge_" + arm] = (bank > kappa[None, :, None])
        raw["bank_contribution_" + arm] = value
        raw["mn_activation_" + arm] = value.sum(axis=2)
        reconstructed = value.sum(axis=(1, 2))
        actual = inputs[arm]["control"]["activation"][:, muscle_index]
        if not np.allclose(reconstructed, actual, rtol=0, atol=1e-10):
            raise ValueError("Exact E150 five-MN decoder reconstruction differs")
    raw["bank_contribution_delta"] = (raw["bank_contribution_candidate"]
                                      - raw["bank_contribution_baseline"])
    delta = raw["mn_activation_candidate"]-raw["mn_activation_baseline"]
    raw["mn_activation_delta"] = delta
    raw["muscle_activation_delta"] = delta.sum(axis=1)
    coefficient = .02*float(inputs["calibration"]["moment"][17, muscle_index])
    raw["joint17_mn_command_delta"] = coefficient*delta
    raw["joint17_muscle_command_delta"] = raw["joint17_mn_command_delta"].sum(axis=1)
    if not np.allclose(raw["joint17_muscle_command_delta"][1:],
                       candidate["control"]["command"][1:, 17]
                       - baseline["control"]["command"][1:, 17],
                       rtol=0, atol=1e-10):
        raise ValueError("E253 selected muscle does not reconstruct joint17 command delta")
    if progress:
        progress("decoder")

    # The observed union is source23, bridge4, then all original motor391.
    for arm in ("baseline", "candidate"):
        data = inputs[arm]
        raw["observed_rate_" + arm] = np.concatenate(
            (data["activity"]["rate"][:, :27],
             data["control"]["motor_rate32"]), axis=1)
    mask = decoder["output_mask"][mapping["observed_ids"]]
    raw["observed_output_mask"] = mask.copy()
    lookup = {int(value): j for j, value in enumerate(mapping["observed_ids"])}
    source_set = set(map(int, source))
    bridge_lookup = {int(value): j for j, value in enumerate(bridge)}
    motor_set = set(map(int, roster))
    offsets = [0]
    pre_rows = []
    post_rows = []
    edge_ids = []
    weights = []
    columns = []
    categories = []
    ptr = inputs["baseline_csr"]["indptr"]
    index = inputs["baseline_csr"]["indices"]
    data = inputs["baseline_csr"]["data"]
    other_data = inputs["candidate_csr"]["data"]
    for post in ids:
        start, stop = map(int, (ptr[post], ptr[post+1]))
        if not np.array_equal(data[start:stop], other_data[start:stop]):
            raise ValueError("E251 target-MN incoming CSR row changed")
        for edge in range(start, stop):
            pre = int(index[edge])
            edge_ids.append(edge)
            pre_rows.append(pre)
            post_rows.append(int(post))
            weights.append(float(data[edge]))
            columns.append(lookup.get(pre, -1))
            categories.append(bridge_lookup.get(pre,
                4 if pre in source_set else 5 if pre in motor_set else -1))
        offsets.append(len(edge_ids))
    edge_ids = np.asarray(edge_ids, np.int64)
    pre_rows = np.asarray(pre_rows, np.int64)
    post_rows = np.asarray(post_rows, np.int64)
    weights32 = np.asarray(weights, np.float32)
    columns = np.asarray(columns, np.int64)
    categories = np.asarray(categories, np.int8)
    observed = columns >= 0
    if not np.array_equal(observed, categories >= 0):
        raise ValueError("Observed CSR category partition differs")
    raw.update(incoming_offsets=np.asarray(offsets, np.int64),
               incoming_edge_id=edge_ids, incoming_pre=pre_rows,
               incoming_post=post_rows, incoming_weight=weights32,
               incoming_observed=observed, incoming_observed_column=columns,
               incoming_category=categories)
    for arm in ("baseline", "candidate"):
        current = np.full((rows, len(edge_ids)), np.nan, np.float64)
        current[:, observed] = (raw["observed_rate_" + arm][:, columns[observed]]
                                .astype(np.float64)
                                *mask[columns[observed]].astype(np.float64)
                                *weights32[observed].astype(np.float64))
        raw["incoming_current_" + arm] = current
    current_delta = np.full((rows, len(edge_ids)), np.nan, np.float64)
    current_delta[:, observed] = (raw["incoming_current_candidate"][:, observed]
                                  - raw["incoming_current_baseline"][:, observed])
    raw["incoming_current_delta"] = current_delta
    for arm in ("baseline", "candidate", "delta"):
        grouped = np.zeros((rows, 5, 6), np.float64)
        current = raw["incoming_current_" + arm]
        for m in range(5):
            for category in range(6):
                selected = np.arange(offsets[m], offsets[m+1])
                selected = selected[categories[selected] == category]
                grouped[:, m, category] = np.sum(current[:, selected], axis=1,
                                                 dtype=np.float64)
        raw["category_current_" + arm] = grouped
    raw["bridge_direct_delta"] = raw["category_current_delta"][:, :, :4].copy()
    if progress:
        progress("incoming_csr")

    coverage = {}
    absolute = np.abs(weights32.astype(np.float64))
    for m, neuron in enumerate(ids):
        lo, hi = offsets[m:m+2]
        available = observed[lo:hi]
        local = weights32[lo:hi]
        total_weight = float(absolute[lo:hi].sum())
        observed_weight = float(absolute[lo:hi][available].sum())
        coverage[str(int(neuron))] = {"incoming_edges": int(hi-lo),
            "observed_edges": int(available.sum()),
            "unobserved_edges": int((~available).sum()),
            "nonzero_edges": int(np.count_nonzero(local)),
            "observed_nonzero_edges": int(np.count_nonzero(local[available])),
            "unobserved_nonzero_edges": int(np.count_nonzero(local[~available])),
            "absolute_weight_total": total_weight,
            "absolute_weight_observed": observed_weight,
            "absolute_weight_unobserved": total_weight-observed_weight,
            "absolute_weight_coverage_fraction": (
                observed_weight/total_weight if total_weight > 0 else None)}
    windows = {}
    for name, (lo, hi) in WINDOWS.items():
        if rows != 301:  # Generated fixtures may still use the frozen 301 rows.
            raise ValueError("E253 fixed endpoint windows require 301 rows")
        mn = delta[lo:hi]
        bridge_delta = raw["bridge_direct_delta"][lo:hi]
        windows[name] = {"controls": [lo, hi],
            "mn_activation": {str(int(ids[i])): _sum3(mn[:, i]) for i in range(5)},
            "bridge_direct_current": {
                str(int(ids[i])): {str(int(bridge[j])): _sum3(bridge_delta[:, i, j])
                                   for j in range(4)} for i in range(5)},
            "sensory23_current": {
                str(int(ids[i])): _sum3(raw["category_current_delta"][lo:hi, i, 4])
                for i in range(5)},
            "observed_motor391_current": {
                str(int(ids[i])): _sum3(raw["category_current_delta"][lo:hi, i, 5])
                for i in range(5)},
            "observed_current_subtotal": {
                str(int(ids[i])): _sum3(raw["category_current_delta"][lo:hi, i].sum(axis=1))
                for i in range(5)}}
    positive_mn = np.maximum(delta[1:100], 0).sum(axis=0)
    motor_concentration = _concentration(positive_mn, ids)
    if motor_concentration["available"]:
        winner = motor_concentration["winner_index"]
        positive_bridge = np.maximum(raw["bridge_direct_delta"][1:100, winner], 0).sum(axis=0)
        bridge_concentration = _concentration(positive_bridge, bridge)
    else:
        bridge_concentration = {"available": False, "total": None,
                                "winner_id": None, "winner_fraction": None,
                                "supported": False}
    cases = {}
    for t in (64, 84):
        cases[str(t)] = {"endpoint_time_s": float(raw["endpoint_time_s"][t]),
            "mn_activation_baseline": raw["mn_activation_baseline"][t].tolist(),
            "mn_activation_candidate": raw["mn_activation_candidate"][t].tolist(),
            "mn_activation_delta": delta[t].tolist(),
            "joint17_mn_command_delta": raw["joint17_mn_command_delta"][t].tolist(),
            "bridge_direct_delta": raw["bridge_direct_delta"][t].tolist()}
    patch = inputs["edge_patch"]
    source_to_bridge = {str(int(item)): patch["edge_ids"][
        patch["post_ids"] == item].astype(np.int64).tolist() for item in bridge}
    bridge_weights = {str(int(ids[m])): {
        str(int(bridge[j])): weights32[
            (post_rows == ids[m]) & (pre_rows == bridge[j])].astype(np.float64).tolist()
        for j in range(4)} for m in range(5)}
    report = {"schema": SCHEMA + ".analysis.v1", "windows": windows,
        "coverage": coverage, "cases": cases, "source_to_bridge_restored_edge_ids": source_to_bridge,
        "bridge_to_motor_existing_weights": bridge_weights,
        "motor_concentration": motor_concentration,
        "bridge_concentration": bridge_concentration,
        "H253_single_observed_route": bool(motor_concentration["supported"]
                                           and bridge_concentration["supported"]),
        "target_muscle_index": muscle_index, "target_joint_column": 17,
        "observed_union_size": len(mapping["observed_ids"]),
        "unobserved_current_available": False,
        "observed_subtotal_is_full_recurrent_current": False,
        "claim_scope": "saved decoder attribution and partial existing input; no causal neural proof"}
    if progress:
        progress("summary")
    return report, raw


def error_record(error: BaseException, stage: str) -> dict:
    return {"stage": stage, "type": type(error).__name__, "message": str(error),
            "traceback": "".join(traceback.format_exception(error))}


def invalidate(result: dict) -> None:
    result["validity"] = "incomplete_or_invalid"
    result["H253_single_observed_route"] = None
    analysis = result.get("analysis")
    if isinstance(analysis, dict):
        analysis["H253_single_observed_route"] = None
        for name in ("motor_concentration", "bridge_concentration"):
            if isinstance(analysis.get(name), dict):
                analysis[name]["supported"] = None


def raw_once(out: Path, raw: dict) -> None:
    if not raw or (out / "diagnostic-raw.npz").exists():
        return
    temporary = out / "diagnostic-raw.npz.part"
    primary = None
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **raw)
        os.replace(temporary, out / "diagnostic-raw.npz")
    except BaseException as error:
        primary = error
    try:
        temporary.unlink(missing_ok=True)
    except BaseException as error:
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E253 raw cleanup: " + repr(error))
    if primary is not None:
        raise primary


def publish(out: Path, result: dict, raw: dict, ledger: dict,
            elapsed, deadline: float) -> None:
    raw_once(out, raw)
    ledger["process_wall_s"] = elapsed()
    save_json(out / "errors.json", ledger["errors"])
    save_json(out / "ledger.json", ledger)
    files = {item.name: {"sha256": sha(item), "bytes": item.stat().st_size}
             for item in sorted(out.iterdir()) if item.is_file()
             and item.name not in ("output-manifest.json", "result.json")}
    manifest = {"schema": SCHEMA + ".manifest.v1", "files": files,
                "total_bytes": sum(item["bytes"] for item in files.values())}
    result["process_wall_s"] = elapsed()
    manifest_payload = json_bytes(manifest)
    result_payload = json_bytes(result)
    if manifest["total_bytes"] + len(manifest_payload) + len(result_payload) > LIMITS["output_bytes"]:
        raise OSError("E253 output exceeds 64 MiB")
    if time.perf_counter() >= deadline:
        raise TimeoutError("E253 deadline before manifest")
    (out / "output-manifest.json").write_bytes(manifest_payload)
    if time.perf_counter() >= deadline:
        raise TimeoutError("E253 deadline before result")
    (out / "result.json").write_bytes(result_payload)
    if time.perf_counter() >= deadline:
        raise TimeoutError("E253 deadline after result")


def settle(out: Path, result: dict, raw: dict, ledger: dict,
           elapsed, deadline: float) -> BaseException | None:
    first = None
    for attempt in range(3):
        result["errors"] = ledger["errors"]
        result["finalization_pending"] = False
        try:
            publish(out, result, raw, ledger, elapsed, deadline)
            result["process_wall_s"] = elapsed()
            save_json(out / "result.json", result)
            if time.perf_counter() >= deadline:
                raise TimeoutError("E253 deadline after final receipt")
            return first
        except BaseException as error:
            if first is None:
                first = error
            ledger["errors"].append(error_record(error, f"publication_{attempt+1}"))
            result.setdefault("primary_error", {"type": type(error).__name__,
                                                "message": str(error)})
            invalidate(result)
            if time.perf_counter() >= deadline:
                deadline = float("inf")
    result["errors"] = ledger["errors"]
    try:
        publish(out, result, raw, ledger, elapsed, float("inf"))
        return first
    except BaseException as error:
        ledger["errors"].append(error_record(error, "terminal_publication"))
        result["finalization_pending"] = True
        result["errors"] = ledger["errors"]
        try:
            (out / "output-manifest.json").unlink(missing_ok=True)
            (out / "result.json").unlink(missing_ok=True)
            save_json(out / "publication-failure.json", ledger["errors"])
            save_json(out / "result.json", result)
        except BaseException as secondary:
            ledger["errors"].append(error_record(secondary, "terminal_receipt"))
        return first


def run(freeze_path: str, freeze_sha: str, review_path: str,
        review_sha: str, output: str) -> dict:
    began = time.perf_counter()
    out = Path(output).resolve()
    if out != (BASE / "e253-lh-reductor-input").resolve():
        raise ValueError("Exact E253 write-once output path required")
    if out.exists():
        raise FileExistsError("E253 output already exists")
    out.mkdir(parents=True)
    ledger = {"schema": SCHEMA + ".ledger.v1", "phase": "gate",
              "loaded_input_roles": [], "analysis_requested_count": 0,
              "analysis_entered_count": 0, "analysis_completed_count": 0,
              "original_calculation_count": 0, "completed_stages": [],
              "fit_calls": 0, "solver_calls": 0, "cuda_calls": 0,
              "neural_constructors": 0, "body_constructors": 0,
              "neural_ticks": 0, "physics_steps": 0, "retries": 0, "errors": []}
    result = {"schema": SCHEMA + ".result.v1", "validity": "incomplete_or_invalid",
              "H253_single_observed_route": None, "candidate_adopted": False,
              "F2_complete": False, "F6_exposed": False,
              "whole_goal_complete": False, "active_UI_changed": False,
              "H251_rejected_preserved": True, "finalization_pending": True}
    raw: dict = {}
    primary = None
    restore_error = None
    def elapsed():
        return time.perf_counter()-began
    def check_budget(stage):
        if elapsed() >= LIMITS["internal_s"]-LIMITS["finalization_reserve_s"]:
            raise TimeoutError("E253 prepublication deadline at " + stage)
        import psutil
        if psutil.Process().memory_info().rss > LIMITS["resident_bytes"]:
            raise MemoryError("E253 RSS exceeds 2 GiB")
    try:
        freeze, review = gate(freeze_path, freeze_sha, review_path, review_sha)
        result["provenance"] = {"freeze_sha256": freeze_sha,
            "review_sha256": review_sha, "source_sha256": review["source_sha256"],
            "input_sha256": freeze["input_sha256"],
            "reused_source_sha256": freeze["reused_source_sha256"],
            "preservation_sha256": freeze["preservation_sha256"],
            "packages": freeze["packages"]}
        if shutil.disk_usage(out).free < LIMITS["minimum_free_bytes"]:
            raise OSError("E253 requires 512 MiB free")
        limiter = None
        work_error = None
        try:
            ledger["threadpool_before"] = threadpool_info()
            limiter = threadpool_limits(limits=1)
            ledger["threadpool_observed"] = threadpool_info()
            if any(item.get("num_threads") != 1 for item in ledger["threadpool_observed"]):
                raise RuntimeError("E253 CPU thread limit differs")
            ledger["phase"] = "load"
            def loaded(role):
                ledger["loaded_input_roles"].append(role)
                check_budget(role)
            inputs = load_inputs(freeze, progress=loaded)
            ledger["phase"] = "analysis"
            ledger["analysis_requested_count"] = 1
            ledger["analysis_entered_count"] = 1
            ledger["original_calculation_count"] = 1
            def progress(stage):
                ledger["completed_stages"].append(stage)
                check_budget(stage)
            analysis, raw = analyze_arrays(inputs, raw=raw, progress=progress)
            ledger["analysis_completed_count"] = 1
        except BaseException as error:
            work_error = error
        finally:
            if limiter is not None:
                try:
                    limiter.restore_original_limits()
                    ledger["threadpool_restore_completed"] = True
                    ledger["threadpool_after"] = threadpool_info()
                except BaseException as error:
                    restore_error = error
                    ledger["threadpool_restore_completed"] = False
        if work_error is not None:
            raise work_error
        if restore_error is not None:
            raise restore_error
        result["analysis"] = analysis
        result["H253_single_observed_route"] = analysis["H253_single_observed_route"]
        result["validity"] = "valid_saved_diagnostic"
        ledger["phase"] = "publication"
    except BaseException as error:
        primary = error
        invalidate(result)
        ledger["errors"].append(error_record(error, ledger["phase"]))
        if restore_error is not None and error is not restore_error:
            ledger["errors"].append(error_record(restore_error, "threadpool_restore"))
    result.update(loaded_input_roles=ledger["loaded_input_roles"],
                  analysis_requested_count=ledger["analysis_requested_count"],
                  analysis_entered_count=ledger["analysis_entered_count"],
                  analysis_completed_count=ledger["analysis_completed_count"],
                  original_calculation_count=ledger["original_calculation_count"],
                  completed_stages=ledger["completed_stages"],
                  fit_calls=0, solver_calls=0, cuda_calls=0,
                  neural_constructors=0, body_constructors=0,
                  neural_ticks=0, physics_steps=0, errors=ledger["errors"])
    if primary is not None:
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
    publication_error = settle(out, result, raw, ledger, elapsed,
                               began+LIMITS["internal_s"])
    if primary is None:
        primary = publication_error
    if primary is not None:
        raise primary
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "freeze-sha256", "review", "review-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha256, args.review, args.review_sha256, args.out)


if __name__ == "__main__":
    main()
