"""E254: replay the two saved E251 neural inputs without constructing a body.

This is a conditional observation of the original network, not a gait candidate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import probe_c_banc_m3_lh_edge_restoration as e250
from tools import probe_c_banc_m3_lh_neural_response as e248

BASE = Path("verification/banc-natural-gait-m3-20260919")
FREEZE_SHA = "6a91c5e70e3b2e2c44746c2d1f907467c7443519b8d20933ca8faab5811f52b9"
FREEZE_SCHEMA = "flylab.m3.e254.lh-full-input-replay.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e254.lh-full-input-replay.review.v1"
OWNED = ("tools/replay_c_banc_m3_lh_full_inputs.py",
         "tests/test_c_banc_lh_full_inputs.py")
VALIDITY = "valid_exact_saved_neural_input_replay"
N = 158706
TARGETS = np.asarray([6373, 22787, 52774, 121544, 139521], np.int64)
WINDOWS = {"startup": (1, 100), "steady": (100, 300), "full": (0, 301)}
CONTROLS = 300
STEPS = 20


def same_bits(left, right) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def pinned(name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Slash-separated E254 frozen path required")
    relative = Path(name)
    path = (ROOT / relative).resolve()
    if (relative.is_absolute() or ".." in relative.parts
            or not path.is_relative_to(ROOT) or not path.is_file()):
        raise ValueError("Invalid E254 frozen path: " + name)
    return path


def gate(freeze_path: Path, freeze_sha: str, review_path: Path,
         review_sha: str, out: Path) -> dict:
    """Complete SHA/package/resource gate before loading any original array."""
    if (freeze_sha != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA
            or sha256(review_path) != review_sha):
        raise ValueError("E254 freeze or source-review SHA differs")
    freeze = e248.read_json(freeze_path)
    review = e248.read_json(review_path)
    limits = freeze.get("resource_limits", {})
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("owned_files") != list(OWNED)
            or freeze.get("output") != (BASE / "e254-lh-full-input-replay").as_posix()
            or freeze.get("n") != N or freeze.get("controls_per_arm") != CONTROLS
            or freeze.get("neural_ticks_per_control") != STEPS
            or freeze.get("initial_tick") != 0
            or freeze.get("motor_ids") != TARGETS.tolist()
            or freeze.get("windows") != {k: list(v) for k, v in WINDOWS.items()}
            or freeze.get("existing_observed_count") != 418
            or limits.get("internal_s") != 55 or limits.get("finalization_reserve_s") != 6
            or limits.get("neural_constructors") != 2
            or limits.get("setup_executed_ticks") != 120
            or limits.get("setup_capture_ticks") != 40
            or limits.get("scientific_neural_ticks") != 12000
            or limits.get("current_probe_calls") != 602
            or limits.get("body_constructors") != 0
            or limits.get("physics_steps") != 0
            or limits.get("fit_calls") != 0 or limits.get("solver_calls") != 0
            or limits.get("original_attempts") != 1 or limits.get("retries") != 0):
        raise ValueError("E254 fixed science/resource contract differs")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("E254 source review has not approved both owned files")
    pins = {}
    for mapping in (freeze["input_sha256"], freeze["reused_source_sha256"],
                    freeze["preservation_sha256"], review["source_sha256"],
                    review.get("additional_input_sha256", {}),
                    review.get("additional_reused_source_sha256", {}),
                    {freeze["design"]: freeze["design_sha256"]}):
        if not isinstance(mapping, dict):
            raise TypeError("E254 SHA mapping required")
        for name, digest in mapping.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E254 SHA pin: " + name)
            pins[name] = digest
    for name, digest in pins.items():
        if sha256(pinned(name)) != digest:
            raise ValueError("E254 pinned byte differs: " + name)
    for package, expected in freeze["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("E254 package version differs: " + package)
    if out.resolve() != (ROOT / BASE / "e254-lh-full-input-replay").resolve():
        raise ValueError("E254 output directory differs")
    if out.exists() or not out.parent.is_dir():
        raise FileExistsError("E254 output must be new")
    if shutil.disk_usage(out.parent).free < limits["minimum_free_disk_bytes"]:
        raise OSError("E254 minimum free disk unavailable")
    return freeze


def reconstruct_drive(old_filtered: np.ndarray, sensory_ids: np.ndarray,
                      sensory_ports: np.ndarray, n: int, descending_ids: np.ndarray,
                      descending_gain: float, navigation_ports: np.ndarray,
                      navigation_forward: float) -> tuple[np.ndarray, np.ndarray]:
    """Match encode scatter, then session descending and cut-navigation addition."""
    old = np.asarray(old_filtered)
    ids, ports = np.asarray(sensory_ids), np.asarray(sensory_ports)
    desc, nav = np.asarray(descending_ids), np.asarray(navigation_ports)
    if (old.ndim != 1 or old.dtype != np.float64 or not np.isfinite(old).all()
            or ids.ndim != 1 or ids.dtype.kind not in "iu" or ports.shape != ids.shape
            or ports.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= n)
            or np.any(ports < 0) or np.any(ports >= len(old))
            or desc.shape != (2,) or nav.shape != (20,)
            or np.unique(desc).size != 2 or np.unique(nav).size != 20
            or np.any(desc < 0) or np.any(desc >= n)
            or np.any(nav < 0) or np.any(nav >= n)
            or not np.isfinite([descending_gain, navigation_forward]).all()):
        raise ValueError("Invalid ordered E254 sensory/navigation binding")
    sensory = np.zeros(n, np.float32)
    if np.unique(ids).size == len(ids):
        sensory[ids] += old[ports]
    else:
        np.add.at(sensory, ids, old[ports])
    drive = sensory.copy()
    drive[desc] += descending_gain
    navigation = np.zeros(n, np.float32)
    navigation[nav] = -5000.0
    navigation[desc] = navigation_forward
    drive += navigation
    return sensory, drive


def incoming_rows(csr, target_ids, observed418_ids, output_mask):
    """Return ordered original CSR entries and one sorted readout cohort."""
    targets = np.asarray(target_ids, np.int64)
    observed = np.asarray(observed418_ids, np.int64)
    mask = np.asarray(output_mask)
    if (targets.ndim != 1 or len(targets) != 5 or len(np.unique(targets)) != 5
            or np.any(targets < 0) or np.any(targets >= csr.shape[0])
            or observed.ndim != 1 or len(observed) != 418
            or not np.array_equal(observed, np.unique(observed))
            or mask.shape != (csr.shape[0],) or mask.dtype != np.float32
            or not np.isin(mask, (0, 1)).all()):
        raise ValueError("E254 original target/observed/mask anatomy differs")
    offsets = np.asarray([int(csr.indptr[t]) for t in targets] +
                         [int(csr.indptr[targets[-1] + 1])], np.int64)
    counts = np.diff(csr.indptr)[targets].astype(np.int64)
    edge_ids = np.concatenate([np.arange(csr.indptr[t], csr.indptr[t + 1],
                                         dtype=np.int64) for t in targets])
    pre = np.asarray(csr.indices[edge_ids], np.int64)
    weights = np.asarray(csr.data[edge_ids], np.float32)
    if (not len(edge_ids) or not np.isfinite(weights).all()
            or not np.array_equal(np.diff(np.r_[0, np.cumsum(counts)]), counts)):
        raise ValueError("E254 complete finite incoming CSR rows required")
    union = np.union1d(observed, pre).astype(np.int64)
    return {"target_ids": targets, "edge_ids": edge_ids,
            "target_offsets": np.r_[0, np.cumsum(counts)].astype(np.int64),
            "csr_row_starts": offsets, "post_ids": np.repeat(targets, counts),
            "pre_ids": pre, "weights32": weights,
            "observed418_ids": observed,
            "observed_edge_mask": np.isin(pre, observed),
            "union_ids": union, "output_mask_start": mask.copy()}


def edge_currents(rate_selected, union_ids, pre_ids, weights32, mask):
    """Descriptive float64 edge terms; not the packed CUDA reduction."""
    union, pre = np.asarray(union_ids), np.asarray(pre_ids)
    rate, weights, output_mask = (np.asarray(rate_selected), np.asarray(weights32),
                                  np.asarray(mask))
    at = np.searchsorted(union, pre)
    if (union.ndim != 1 or rate.shape != union.shape or rate.dtype != np.float32
            or pre.ndim != 1 or weights.shape != pre.shape
            or weights.dtype != np.float32 or output_mask.dtype != np.float32
            or output_mask.ndim != 1 or np.any(pre < 0)
            or np.any(pre >= len(output_mask))
            or np.any(at >= len(union)) or not np.array_equal(union[at], pre)
            or not np.isfinite(rate).all() or not np.isfinite(weights).all()):
        raise ValueError("E254 incoming rate/edge alignment differs")
    masked = np.multiply(rate[at], output_mask[pre], dtype=np.float32)
    return weights.astype(np.float64) * masked.astype(np.float64)


def read_npz(path: Path, keys: tuple[str, ...]) -> dict:
    with np.load(path, allow_pickle=False) as file:
        if not set(keys).issubset(file.files):
            raise ValueError("Missing E254 NPZ keys: " + str(path))
        return {key: file[key].copy() for key in keys}


def prepare(freeze: dict):
    """Load original artifacts only after gate; no body or graph constructor."""
    from scipy.sparse import csr_matrix

    from flylab.c.storage import StateStore
    from tools.banc_m3_intrinsic_recruitment import patch_tau

    path = lambda key: pinned(freeze["paths"][key])
    config = read_npz(path("configuration"), ("membrane_tau", "cell_gain", "motor_ids"))
    cal = read_npz(path("calibration"), ("tau_ids", "tau_values"))
    tau = patch_tau(np.full(N, .02, np.float32), cal["tau_ids"], cal["tau_values"])
    if not same_bits(tau, config["membrane_tau"]):
        raise ValueError("E254 original native membrane tau differs")
    csr = {}
    for arm in ("baseline", "candidate"):
        values = read_npz(path(arm + "_csr"), ("indptr", "indices", "data"))
        if (values["data"].dtype != np.float32
                or len(values["data"]) != freeze["declared_edges"]
                or values["indptr"].shape != (N + 1,)
                or values["indices"].shape != values["data"].shape):
            raise ValueError("E254 original CSR shape/precision differs")
        csr[arm] = csr_matrix((values["data"], values["indices"],
                               values["indptr"]), shape=(N, N), dtype=np.float32)
        csr[arm].check_format(full_check=True)
    ports = e248.read_json(path("drive_ports"))
    if (ports.get("graph_hash") != freeze["graph_hash"] or ports.get("n") != N
            or ports.get("identity") is None or ports.get("navigation_cut") is not True
            or ports.get("navigation_forward_drive") != 5000.0
            or ports.get("p_descending_drive") != 0.0):
        raise ValueError("E254 original navigation port provenance differs")
    desc = np.asarray(ports["descending"], np.int64)
    nav = np.asarray(ports["ports"], np.int64)
    if (desc.shape != (2,) or nav.shape != (20,)
            or not np.array_equal(np.unique(np.r_[desc, nav]).size, 22)):
        raise ValueError("E254 20+2 original navigation binding differs")
    selection = e248.read_json(path("anatomical_selection"))
    prior_result = e248.read_json(path("e253_result"))
    if (prior_result.get("validity") is None or not isinstance(selection, dict)):
        raise ValueError("E254 prerequisite diagnostic provenance missing")
    arms = {}
    for arm, folder in freeze["arms"].items():
        source = ROOT / folder
        start = StateStore.load(source / "observed-start")
        final = StateStore.load(source / "final")
        sensory = read_npz(source / "sensory-input.npz",
                           ("control_tick_before", "complete", "old_filtered",
                            "source_ids", "returned_source_drive"))
        metadata = e248.read_json(source / "sensory-metadata.json")
        activity = read_npz(source / "activity.npz",
                            ("observed_ids", "rate", "drive", "adaptation"))
        controls = read_npz(source / "control-raw.npz",
                            ("control_tick", "network_tick", "motor_rate32"))
        if (start["schema"] != "flylab.m3.e251.lh-restoration-trial.v1"
                or start["control_tick"] != 0 or start["graph_hash"] != freeze["graph_hash"]
                or start["cuts"] != {key: False for key in
                                      ("sensory", "descending", "circuit", "motor")}
                or start["parameters"]["descending_drive"] != 0.
                or start["parameters"]["neural_dt_s"] != .00025
                or start["navigation"]["cut"] is not True
                or start["navigation"]["config"]["forward_drive"] != 5000.
                or start["navigation"]["encoder"]["hash"] != ports["identity"]
                or start["navigation"]["encoder"]["turn"] != 0
                or start["navigation"]["encoder"]["braking"] is not False
                or start["network"]["tick"] != 0 or final["network"]["tick"] != 6000
                or final["control_tick"] != 300
                or start["network"]["device"] != "cuda"
                or not same_bits(start["network"]["output_mask"],
                                 np.ones(N, np.float32))):
            raise ValueError("E254 frozen whole start/final/cuts/navigation differs")
        ids = np.asarray(metadata["sensory_ids"], np.int64)
        which = np.asarray(metadata["sensory_ports"], np.int64)
        if (sensory["old_filtered"].shape != (300, 241)
                or sensory["old_filtered"].dtype != np.float64
                or sensory["returned_source_drive"].dtype != np.float32
                or sensory["returned_source_drive"].shape != (300, 23)
                or not sensory["complete"].all()
                or not np.array_equal(sensory["control_tick_before"], np.arange(300))
                or not same_bits(sensory["source_ids"],
                                 np.asarray(metadata["source_rows"], np.int64))
                or ids.shape != which.shape or np.any(which < 0)
                or np.any(which >= 241)
                or controls["motor_rate32"].shape != (301, 391)
                or controls["motor_rate32"].dtype != np.float32
                or not np.array_equal(controls["control_tick"], np.arange(301))
                or not np.array_equal(controls["network_tick"], np.arange(301) * 20)
                or activity["rate"].shape != activity["drive"].shape
                or activity["rate"].shape != activity["adaptation"].shape
                or activity["rate"].shape != (301, len(activity["observed_ids"]))
                or not all(activity[key].dtype == np.float32 for key in
                           ("rate", "drive", "adaptation"))):
            raise ValueError("E254 source sensory/control/cohort shape or clock differs")
        arms[arm] = {"source": source, "start": start["network"],
                     "final": final["network"], "sensory": sensory,
                     "metadata": metadata, "activity": activity,
                     "controls": controls, "sensory_ids": ids, "sensory_ports": which}
    if set(arms) != {"baseline", "candidate"}:
        raise ValueError("E254 both original arms required")
    observed = np.union1d(np.union1d(arms["baseline"]["sensory"]["source_ids"],
                                       np.asarray(freeze["bridge_ids"], np.int64)),
                          config["motor_ids"]).astype(np.int64)
    if (len(observed) != 418 or not same_bits(arms["baseline"]["sensory"]["source_ids"],
                                           arms["candidate"]["sensory"]["source_ids"])
            or not same_bits(arms["baseline"]["sensory_ids"],
                             arms["candidate"]["sensory_ids"])
            or not same_bits(arms["baseline"]["sensory_ports"],
                             arms["candidate"]["sensory_ports"])
            or not same_bits(arms["baseline"]["activity"]["observed_ids"],
                             arms["candidate"]["activity"]["observed_ids"])):
        raise ValueError("E254 original 418-roster/cohort differs")
    rows = incoming_rows(csr["baseline"], TARGETS, observed,
                         arms["baseline"]["start"]["output_mask"])
    candidate_rows = incoming_rows(csr["candidate"], TARGETS, observed,
                                   arms["candidate"]["start"]["output_mask"])
    if any(not same_bits(rows[key], candidate_rows[key]) for key in rows):
        raise ValueError("E254 fixed5MN incoming CSR differs across E251 arms")
    drive_ids = np.unique(np.r_[arms["baseline"]["sensory_ids"],
                                arms["candidate"]["sensory_ids"], nav, desc]).astype(np.int64)
    return {"config": config, "tau": tau, "csr": csr, "ports": ports,
            "navigation_ports": nav, "descending_ids": desc,
            "arms": arms, "incoming": rows, "drive_ids": drive_ids}


def begin_arm(ledger: dict, arm: str) -> dict:
    entry = {"name": arm, "constructor_requested": True,
             "constructor_entered": False, "constructor_completed": False,
             "setup_unknown_partial": False, "setup_executed_ticks": None,
             "setup_capture_ticks": None, "entered_controls": 0,
             "completed_neural_controls": 0, "observed_endpoint_count": 0,
             "unknown_neural_partial": False, "current_probe_requested": 0,
             "current_probe_entered": 0, "current_probe_completed": 0,
             "initial_exact": False, "final_exact": False, "status": "requested"}
    ledger["arms"][arm] = entry
    return entry


def memmaps(arm_dir: Path, union_count: int, edge_count: int, drive_count: int):
    spec = {"endpoint-rate.npy": (301, union_count),
            "endpoint-drive.npy": (301, union_count),
            "endpoint-adaptation.npy": (301, union_count),
            "actual-current.npy": (301, 5),
            "edge-current64.npy": (301, edge_count),
            "drive-values.npy": (300, drive_count)}
    result = {}
    for name, shape in spec.items():
        dtype = np.float64 if name == "edge-current64.npy" else np.float32
        value = np.lib.format.open_memmap(arm_dir / name, mode="w+", dtype=dtype,
                                          shape=shape)
        value[:] = np.nan
        value.flush()
        result[name] = value
    return result


def flush_raw(raw):
    for value in raw.values():
        value.flush()


def observe_network(network, index, union_ids, incoming, raw, entry):
    """Copy recoverable fields separately; _current is a read-only probe."""
    import torch

    from tools.verify_c_banc_observation import exact

    pre_observation = network.snapshot() if index in (0, CONTROLS) else None

    row = network.readout(union_ids)
    raw["endpoint-rate.npy"][index] = row
    raw["endpoint-rate.npy"].flush()
    device_ids = torch.as_tensor(union_ids, dtype=torch.int64, device=network.device)
    for field, tensor in (("endpoint-drive.npy", network.drive),
                          ("endpoint-adaptation.npy", network.adaptation)):
        raw[field][index] = tensor.index_select(0, device_ids).detach().cpu().numpy()
        raw[field].flush()
    entry["current_probe_requested"] += 1
    entry["current_probe_entered"] += 1
    targets = torch.as_tensor(incoming["target_ids"], dtype=torch.int64,
                              device=network.device)
    with torch.inference_mode():
        current = network._current(network.rate)
        measured = current.index_select(0, targets).detach().cpu().numpy().copy()
    entry["current_probe_completed"] += 1
    raw["actual-current.npy"][index] = measured
    raw["actual-current.npy"].flush()
    raw["edge-current64.npy"][index] = edge_currents(
        np.asarray(row), union_ids, incoming["pre_ids"], incoming["weights32"],
        incoming["output_mask_start"])
    raw["edge-current64.npy"].flush()
    entry["observed_endpoint_count"] = index + 1
    if pre_observation is not None:
        exact(network.snapshot(), pre_observation,
              "E254 endpoint observation left whole neural state unchanged")


def check_endpoint(index, raw, arm_data, union_ids, motor_ids):
    rate = np.asarray(raw["endpoint-rate.npy"][index])
    drive = np.asarray(raw["endpoint-drive.npy"][index])
    adaptation = np.asarray(raw["endpoint-adaptation.npy"][index])
    activity = arm_data["activity"]
    at = np.searchsorted(union_ids, activity["observed_ids"])
    motor_at = np.searchsorted(union_ids, motor_ids)
    if (np.any(at >= len(union_ids)) or np.any(motor_at >= len(union_ids))
            or not np.array_equal(union_ids[at], activity["observed_ids"])
            or not np.array_equal(union_ids[motor_at], motor_ids)
            or not same_bits(rate[motor_at], arm_data["controls"]["motor_rate32"][index])
            or not same_bits(rate[at], activity["rate"][index])
            or not same_bits(drive[at], activity["drive"][index])
            or not same_bits(adaptation[at], activity["adaptation"][index])):
        raise AssertionError(f"E254 original E251 MN/cohort endpoint differs at {index}")


def science_loop(network, arm_data, raw, incoming, drive_ids, navigation_ports,
                 descending_ids, motor_ids, entry, persist, budget,
                 observe=observe_network, check=check_endpoint):
    """Advance bookkeeping is independent of endpoint observation/bookkeeping."""
    union = incoming["union_ids"]
    observe(network, 0, union, incoming, raw, entry)
    persist()
    check(0, raw, arm_data, union, motor_ids)
    for control in range(1, CONTROLS + 1):
        budget()
        sensory, drive = reconstruct_drive(
            arm_data["sensory"]["old_filtered"][control - 1],
            arm_data["sensory_ids"], arm_data["sensory_ports"], N,
            descending_ids, 0., navigation_ports, 5000.)
        sources = arm_data["sensory"]["source_ids"]
        if not same_bits(sensory[sources],
                         arm_data["sensory"]["returned_source_drive"][control - 1]):
            raise AssertionError(f"E254 original sensory drive differs at {control}")
        raw["drive-values.npy"][control - 1] = drive[drive_ids]
        raw["drive-values.npy"].flush()
        entry["entered_controls"] = control
        entry["unknown_neural_partial"] = True
        persist()
        network.advance(drive, STEPS)
        entry["completed_neural_controls"] = control
        entry["unknown_neural_partial"] = False
        persist()
        observe(network, control, union, incoming, raw, entry)
        persist()
        check(control, raw, arm_data, union, motor_ids)
    entry["status"] = "science_complete"


def signed_parts(values):
    """Split before reducing time or edges; cancellation remains visible."""
    data = np.asarray(values, np.float64)
    if not np.isfinite(data).all():
        raise ValueError("Finite E254 component values required")
    return {"signed": float(np.sum(data, dtype=np.float64)),
            "positive": float(np.sum(np.maximum(data, 0.), dtype=np.float64)),
            "negative": float(np.sum(np.minimum(data, 0.), dtype=np.float64))}


def summarize(raw_baseline, raw_candidate, incoming, freeze, pre_graph_ids):
    """Compute only preregistered endpoint descriptions after exact replay."""
    b = np.asarray(raw_baseline["edge-current64.npy"])
    c = np.asarray(raw_candidate["edge-current64.npy"])
    delta = c - b
    target = incoming["target_ids"]
    observed = incoming["observed_edge_mask"]
    starts = incoming["target_offsets"]
    union = incoming["union_ids"]
    target_at = np.searchsorted(union, target)
    windows = {}
    for name, (lo, hi) in WINDOWS.items():
        rows = {}
        top = {}
        edge_samples = delta[lo:hi]
        edge_window = np.sum(edge_samples, axis=0, dtype=np.float64)
        edge_positive = np.sum(np.maximum(edge_samples, 0.), axis=0, dtype=np.float64)
        edge_negative = np.sum(np.minimum(edge_samples, 0.), axis=0, dtype=np.float64)
        for i, motor in enumerate(target):
            span = slice(starts[i], starts[i + 1])
            mask = observed[span]
            this = edge_window[span]
            observed_samples = edge_samples[:, span][:, mask]
            unobserved_samples = edge_samples[:, span][:, ~mask]
            observed_part = signed_parts(observed_samples)
            unobserved_part = signed_parts(unobserved_samples)
            total_part = signed_parts(edge_samples[:, span])
            current_samples = (
                np.asarray(raw_candidate["actual-current.npy"])[lo:hi, i].astype(np.float64)
                - np.asarray(raw_baseline["actual-current.npy"])[lo:hi, i].astype(np.float64))
            drive_samples = (
                np.asarray(raw_candidate["endpoint-drive.npy"])[lo:hi, target_at[i]].astype(np.float64)
                - np.asarray(raw_baseline["endpoint-drive.npy"])[lo:hi, target_at[i]].astype(np.float64))
            negative_z_samples = (
                np.asarray(raw_baseline["endpoint-adaptation.npy"])[lo:hi, target_at[i]].astype(np.float64)
                - np.asarray(raw_candidate["endpoint-adaptation.npy"])[lo:hi, target_at[i]].astype(np.float64))
            current_part = signed_parts(current_samples)
            drive_part = signed_parts(drive_samples)
            negative_z_part = signed_parts(negative_z_samples)
            margin_part = signed_parts(np.sum(edge_samples[:, span], axis=1,
                                               dtype=np.float64)
                                       + drive_samples + negative_z_samples)
            rows[str(int(motor))] = {
                "observed_delta_sum64": observed_part["signed"],
                "observed_positive_delta_sum64": observed_part["positive"],
                "observed_negative_delta_sum64": observed_part["negative"],
                "unobserved_delta_sum64": unobserved_part["signed"],
                "unobserved_positive_delta_sum64": unobserved_part["positive"],
                "unobserved_negative_delta_sum64": unobserved_part["negative"],
                "total_edge_delta_sum64": total_part["signed"],
                "positive_edge_delta_sum64": total_part["positive"],
                "negative_edge_delta_sum64": total_part["negative"],
                "actual_current_delta_sum32": current_part["signed"],
                "actual_current_positive_delta_sum32": current_part["positive"],
                "actual_current_negative_delta_sum32": current_part["negative"],
                "drive_delta_sum32": drive_part["signed"],
                "drive_positive_delta_sum32": drive_part["positive"],
                "drive_negative_delta_sum32": drive_part["negative"],
                "negative_adaptation_delta_sum32": negative_z_part["signed"],
                "negative_adaptation_positive_delta_sum32": negative_z_part["positive"],
                "negative_adaptation_negative_delta_sum32": negative_z_part["negative"],
                "margin_delta_sum64": margin_part["signed"],
                "margin_positive_delta_sum64": margin_part["positive"],
                "margin_negative_delta_sum64": margin_part["negative"]}
            ids = incoming["edge_ids"][span]
            pre = incoming["pre_ids"][span]
            per_edge_positive = edge_positive[span]
            per_edge_negative = edge_negative[span]
            def item(j, this=this, ids=ids, pre=pre,
                     per_edge_positive=per_edge_positive,
                     per_edge_negative=per_edge_negative):
                signed = float(this[j])
                return {"edge_id": int(ids[j]), "pre_row": int(pre[j]),
                        "pre_graph_id": pre_graph_ids[int(pre[j])],
                        "signed_delta_sum64": signed,
                        "positive_delta_sum64": float(per_edge_positive[j]),
                        "negative_delta_sum64": float(per_edge_negative[j])}
            positive_rank = sorted((j for j in range(len(this)) if this[j] > 0),
                                   key=lambda j: (-this[j], pre_graph_ids[int(pre[j])], int(ids[j])))[:10]
            negative_rank = sorted((j for j in range(len(this)) if this[j] < 0),
                                   key=lambda j: (this[j], pre_graph_ids[int(pre[j])], int(ids[j])))[:10]
            top[str(int(motor))] = {"positive": [item(j) for j in positive_rank],
                                    "negative": [item(j) for j in negative_rank]}
        windows[name] = {"controls": [lo, hi], "motors": rows, "top_edges": top}
    cases = {}
    for index in freeze["case_controls"]:
        cases[str(index)] = {}
        for i, motor in enumerate(target):
            span = slice(starts[i], starts[i + 1])
            value = delta[index, span]
            o = float(np.sum(value[observed[span]], dtype=np.float64))
            u = float(np.sum(value[~observed[span]], dtype=np.float64))
            d = (float(np.asarray(raw_candidate["endpoint-drive.npy"])[index, target_at[i]])
                 - float(np.asarray(raw_baseline["endpoint-drive.npy"])[index, target_at[i]]))
            z = (float(np.asarray(raw_baseline["endpoint-adaptation.npy"])[index, target_at[i]])
                 - float(np.asarray(raw_candidate["endpoint-adaptation.npy"])[index, target_at[i]]))
            cases[str(index)][str(int(motor))] = {
                "observed_delta64": o, "unobserved_delta64": u,
                "total_edge_delta64": o + u,
                "actual_current_delta32": (
                    float(np.asarray(raw_candidate["actual-current.npy"])[index, i])
                    - float(np.asarray(raw_baseline["actual-current.npy"])[index, i])),
                "drive_delta32": d, "negative_adaptation_delta32": z,
                "margin_delta64": o + u + d + z}
    chosen = windows["startup"]["motors"][str(freeze["hypothesis_motor_id"])]
    o, u = chosen["observed_delta_sum64"], chosen["unobserved_delta_sum64"]
    return {"schema": "flylab.m3.e254.windows.v1", "windows": windows,
            "case_controls": cases, "H254_measured_condition_met": bool(o < 0 and u > 0 and u > -o),
            "H254_supported": bool(o < 0 and u > 0 and u > -o),
            "interpretation_status": "valid_exact_replay_only",
            "scope": "signed float64 endpoint decomposition; no causal or gait claim"}


def save_final_before_parity(network, path, expected, *, save=None, exact=None):
    """Durably keep the computed whole state before comparing to E251."""
    if save is None:
        from flylab.c.storage import StateStore
        save = StateStore.save
    if exact is None:
        from tools.verify_c_banc_observation import exact as original_exact
        exact = original_exact
    state = network.snapshot()
    save(path, state)
    if state["tick"] != expected["tick"]:
        raise AssertionError("E254 final network clock differs")
    exact(state, expected, "E254 original E251 full neural final")
    return state


def preserve_partial_state(network, path, entry, primary, *, save=None):
    """Best-effort whole state after an earlier failure; never mask its cause."""
    if network is None or path is None or path.exists():
        return primary
    if save is None:
        from flylab.c.storage import StateStore
        save = StateStore.save
    try:
        save(path, network.snapshot())
        entry["partial_full_state_available"] = True
    except Exception as secondary:  # noqa: BLE001 - earliest failure remains primary
        entry["partial_full_state_available"] = False
        entry["partial_full_state_error"] = repr(secondary)
        if primary is not None:
            primary.add_note("Secondary E254 whole-state preservation: " + repr(secondary))
        else:
            primary = secondary
    return primary


def node_ids_for(pre_ids, graph_hash):
    """Read only the pinned ordered graph labels; do not build a graph/model."""
    path = ROOT / "data/acquisitions/banc888-windows-20260910/bundle/nodes.jsonl"
    needed = set(map(int, pre_ids))
    found = {}
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in needed:
                found[index] = json.loads(line)["id"]
    if len(found) != len(needed) or not graph_hash:
        raise ValueError("E254 incomplete graph node provenance")
    return found


def output_manifest(out: Path, limit: int):
    files, total = {}, 0
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name in ("result.json", "output-manifest.json"):
            continue
        size = path.stat().st_size
        total += size
        if total > limit:
            raise OSError("E254 output bytes exceeded")
        files[path.relative_to(out).as_posix()] = {"sha256": sha256(path), "bytes": size}
    return {"schema": "flylab.m3.e254.output-manifest.v1", "bytes": total,
            "files": files}


def finish_receipts(out, freeze, ledger, result, errors, primary, began,
                    persist=e248.write_json):
    """No stale support/result/manifest survives a late publication error."""
    limit = freeze["resource_limits"]["internal_s"]
    def fail(error, stage):
        nonlocal primary
        if primary is None:
            primary = error
        errors.append({"stage": stage, "type": type(error).__name__,
                       "message": str(error)})
        result.update(validity="resource_limited" if isinstance(primary, TimeoutError)
                      else "invalid_or_incomplete", H254_supported=False,
                      finalization_pending=True,
                      first_error={"type": type(primary).__name__, "message": str(primary)})

    def revoke():
        path = out / "windows.json"
        if path.exists():
            window = e248.read_json(path)
            window["H254_supported"] = False
            window["interpretation_status"] = "not_tested_invalid_or_incomplete"
            persist(path, window)

    if primary is not None:
        fail(primary, "prior science/cleanup")

    result["work"] = {"constructors_requested": len(ledger["arms"]),
        "constructors_completed": sum(bool(x["constructor_completed"])
                                      for x in ledger["arms"].values()),
        "setup_executed_ticks": sum(x["setup_executed_ticks"] or 0
                                    for x in ledger["arms"].values()),
        "setup_capture_ticks": sum(x["setup_capture_ticks"] or 0
                                   for x in ledger["arms"].values()),
        "scientific_ticks_completed": STEPS * sum(x["completed_neural_controls"]
                                                    for x in ledger["arms"].values()),
        "current_probe_calls_completed": sum(x["current_probe_completed"]
                                             for x in ledger["arms"].values()),
        "unknown_partial_arms": [k for k, x in ledger["arms"].items()
                                 if x["unknown_neural_partial"] or x["setup_unknown_partial"]]}
    for name, value in (("advance-ledger.json", ledger),
                        ("errors.json", {"errors": errors})):
        try:
            persist(out / name, value)
        except Exception as error:  # noqa: BLE001 - preserve first failure
            fail(error, name)
    if time.perf_counter() - began >= limit:
        fail(TimeoutError("E254 deadline including finalization"), "deadline")
    if primary is not None:
        try:
            revoke()
        except Exception as error:  # noqa: BLE001
            fail(error, "windows revocation")
            (out / "windows.json").unlink(missing_ok=True)
    try:
        persist(out / "errors.json", {"errors": errors})
        persist(out / "output-manifest.json",
                output_manifest(out, freeze["resource_limits"]["output_bytes"]))
        result["elapsed_s"] = time.perf_counter() - began
        if result["elapsed_s"] >= limit:
            raise TimeoutError("E254 publication deadline")
        result["finalization_pending"] = False
        persist(out / "result.json", result)
        if time.perf_counter() - began >= limit:
            raise TimeoutError("E254 late final-result deadline")
    except Exception as error:  # noqa: BLE001
        fail(error, "publication")
        (out / "result.json").unlink(missing_ok=True)
        (out / "output-manifest.json").unlink(missing_ok=True)
        try:
            revoke()
        except Exception as secondary:  # noqa: BLE001
            fail(secondary, "windows revocation")
            (out / "windows.json").unlink(missing_ok=True)
        for name, value in (("advance-ledger.json", ledger),
                            ("result.json", result)):
            try:
                persist(out / name, value)
            except Exception as secondary:  # noqa: BLE001
                fail(secondary, "retry " + name)
                (out / name).unlink(missing_ok=True)
        try:
            persist(out / "errors.json", {"errors": errors})
        except Exception as secondary:  # noqa: BLE001
            fail(secondary, "errors retry")
            (out / "errors.json").unlink(missing_ok=True)
        # An incomplete publication has no manifest. All successful manifests
        # are complete and hash the final receipts; partial raw remain on disk.
    return primary


def execute(freeze: dict, out: Path) -> None:
    from flylab.c.storage import StateStore
    from tools.verify_c_banc_observation import exact

    began = time.perf_counter()
    out.mkdir(parents=False, exist_ok=False)
    ledger = {"schema": "flylab.m3.e254.ledger.v1", "status": "entered",
              "arms": {}, "original_attempts": 1, "retries": 0,
              "body_constructors": 0, "physics_steps": 0,
              "fit_calls": 0, "solver_calls": 0, "resources": []}
    result = {"schema": "flylab.m3.e254.result.v1", "validity": "incomplete",
              "H254_supported": False, "candidate_adopted": False,
              "F6_exposed": False, "whole_goal_pass": False,
              "finalization_pending": True}
    errors = []
    primary = None
    network = None
    active_arm_dir = None
    active_entry = None
    raws = {}
    thread_limiter = None
    old_torch_threads = None
    old_environment = {}
    try:
        import psutil
        import torch
        from threadpoolctl import threadpool_info, threadpool_limits

        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            old_environment[name] = os.environ.get(name)
            os.environ[name] = "1"
        old_torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        thread_limiter = threadpool_limits(limits=1)
        if (torch.get_num_threads() != 1
                or any(x["num_threads"] != 1 for x in threadpool_info())):
            raise RuntimeError("E254 effective CPU thread count differs")
        e248.write_json(out / "threading.json", {"torch_threads": 1,
                         "threadpools": threadpool_info()})
        if torch.cuda.mem_get_info()[0] < freeze["resource_limits"]["minimum_free_gpu_bytes"]:
            raise MemoryError("E254 minimum free GPU memory unavailable")
        torch.cuda.reset_peak_memory_stats()
        inputs = prepare(freeze)
        incoming = inputs["incoming"]
        drive_ids = inputs["drive_ids"]
        np.savez(out / "incoming-rows.npz", **incoming)
        np.savez(out / "drive-mapping.npz",
                 sensory_ids=inputs["arms"]["baseline"]["sensory_ids"],
                 sensory_ports=inputs["arms"]["baseline"]["sensory_ports"],
                 navigation_ports=inputs["navigation_ports"],
                 descending_ids=inputs["descending_ids"], drive_ids=drive_ids)
        for arm in ("baseline", "candidate"):
            arm_data = inputs["arms"][arm]
            arm_dir = out / arm
            arm_dir.mkdir(exist_ok=False)
            entry = begin_arm(ledger, arm)
            active_arm_dir = arm_dir
            active_entry = entry
            e248.write_json(out / "advance-ledger.json", ledger)
            if time.perf_counter() - began >= 49:
                raise TimeoutError("E254 constructor/science deadline")
            entry["constructor_entered"] = True
            entry["setup_unknown_partial"] = True
            e248.write_json(out / "advance-ledger.json", ledger)
            network, restored, identity_fields, backend = e250._constructor(
                inputs["csr"][arm], inputs["tau"], inputs["config"],
                arm_data["start"], expected_identity=arm_data["start"]["model_hash"],
                restore_saved=arm_data["start"])
            if identity_fields:
                raise AssertionError("E254 neural model identity was transplanted")
            entry.update(constructor_completed=True, setup_unknown_partial=False,
                         setup_executed_ticks=60, setup_capture_ticks=20,
                         backend=backend)
            exact(restored, arm_data["start"], "E254 untouched neural start")
            StateStore.save(arm_dir / "observed-start", network.snapshot())
            entry["initial_exact"] = True
            if arm == "baseline":
                parameters = {"target_ids": TARGETS}
                for key, source in (("threshold", network.threshold),
                                    ("coefficient", network.coefficient),
                                    ("cap", network.cap), ("membrane_tau", network.tau),
                                    ("cell_gain", network._cell_gain),
                                    ("cell_adaptation_tau", network._cell_adaptation_tau),
                                    ("inverse_adaptation_tau", network._cell_inverse_tau)):
                    parameters[key] = source[TARGETS].detach().cpu().numpy().copy()
                np.savez(out / "target-parameters.npz", **parameters)
            else:
                saved_params = read_npz(out / "target-parameters.npz",
                                        ("threshold", "coefficient", "cap", "membrane_tau",
                                         "cell_gain", "cell_adaptation_tau",
                                         "inverse_adaptation_tau"))
                for key, source in (("threshold", network.threshold),
                                    ("coefficient", network.coefficient),
                                    ("cap", network.cap), ("membrane_tau", network.tau),
                                    ("cell_gain", network._cell_gain),
                                    ("cell_adaptation_tau", network._cell_adaptation_tau),
                                    ("inverse_adaptation_tau", network._cell_inverse_tau)):
                    if not same_bits(saved_params[key],
                                     source[TARGETS].detach().cpu().numpy()):
                        raise AssertionError("E254 target physiological parameter changed")
            raw = memmaps(arm_dir, len(incoming["union_ids"]),
                          len(incoming["edge_ids"]), len(drive_ids))
            raws[arm] = raw
            start_snapshot = network.snapshot()

            def persist():
                e248.write_json(out / "advance-ledger.json", ledger)

            def budget():
                elapsed = time.perf_counter() - began
                memory = psutil.Process().memory_info()
                resource = {"elapsed_s": elapsed,
                            "resident_bytes": getattr(memory, "peak_wset", memory.rss),
                            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "gpu_free_bytes": torch.cuda.mem_get_info()[0],
                            "disk_free_bytes": shutil.disk_usage(out).free}
                if (elapsed >= 49
                        or resource["resident_bytes"] > freeze["resource_limits"]["process_memory_bytes"]
                        or resource["gpu_peak_allocated_bytes"] > freeze["resource_limits"]["additional_gpu_memory_bytes"]
                        or resource["disk_free_bytes"] < freeze["resource_limits"]["minimum_free_disk_bytes"]):
                    raise TimeoutError("E254 original science/resource cap reached")
                if len(ledger["resources"]) < 20:
                    ledger["resources"].append(resource)

            science_loop(network, arm_data, raw, incoming, drive_ids,
                         inputs["navigation_ports"], inputs["descending_ids"],
                         inputs["config"]["motor_ids"], entry, persist, budget)
            if not e248.same_bits(start_snapshot["rate"], arm_data["start"]["rate"]):
                raise AssertionError("E254 initial rate changed during observation")
            save_final_before_parity(network, arm_dir / "final", arm_data["final"],
                                     save=StateStore.save, exact=exact)
            entry["final_exact"] = True
            entry["status"] = "valid_exact_replay"
            persist()
            flush_raw(raw)
            network = None
            active_arm_dir = None
            active_entry = None
            gc.collect()
            torch.cuda.empty_cache()
        if (sum(x["setup_executed_ticks"] for x in ledger["arms"].values()) != 120
                or sum(x["setup_capture_ticks"] for x in ledger["arms"].values()) != 40
                or sum(x["completed_neural_controls"] for x in ledger["arms"].values()) != 600
                or sum(x["current_probe_completed"] for x in ledger["arms"].values()) != 602):
            raise AssertionError("E254 original work accounting differs")
        labels = node_ids_for(incoming["pre_ids"], freeze["graph_hash"])
        summary = summarize(raws["baseline"], raws["candidate"], incoming,
                            freeze, labels)
        e248.write_json(out / "windows.json", summary)
        e248.write_json(out / "top-edges.json", {"windows": {
            key: val["top_edges"] for key, val in summary["windows"].items()}})
        result.update(validity=VALIDITY,
                      H254_supported=summary["H254_supported"],
                      exact_replay_both=True,
                      interpretation="signed endpoint observation only")
    except Exception as error:  # noqa: BLE001 - raw/receipts survive scientific failure
        primary = error
        errors.append({"stage": "science", "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
        if (network is not None and active_arm_dir is not None
                and active_entry is not None
                and not (active_arm_dir / "final").exists()):
            primary = preserve_partial_state(network, active_arm_dir / "partial-final",
                                             active_entry, primary)
    finally:
        for raw in raws.values():
            try:
                flush_raw(raw)
            except Exception as error:  # noqa: BLE001
                if primary is None:
                    primary = error
                errors.append({"stage": "raw flush", "error": repr(error)})
        raws.clear()
        network = None
        gc.collect()
        if thread_limiter is not None:
            try:
                thread_limiter.restore_original_limits()
            except Exception as error:  # noqa: BLE001
                if primary is None:
                    primary = error
                errors.append({"stage": "threadpool restore", "error": repr(error)})
        if old_torch_threads is not None:
            try:
                import torch
                torch.set_num_threads(old_torch_threads)
            except Exception as error:  # noqa: BLE001
                if primary is None:
                    primary = error
                errors.append({"stage": "torch thread restore", "error": repr(error)})
        for name, old in old_environment.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
        primary = finish_receipts(out, freeze, ledger, result, errors, primary, began)
        if primary is not None:
            raise RuntimeError("E254 invalid: " + str(primary)) from primary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    freeze = gate(args.freeze.resolve(), args.freeze_sha256,
                  args.review.resolve(), args.review_sha256, out)
    execute(freeze, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
