"""E308 observed endpoint transfer audit. No model, integration, or simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
OUTPUT = BASE + "/e308-resource-muscle-contrast"
SCHEMA = "flylab.m3.e308.resource-muscle-contrast.result.v1"
FREEZE_SCHEMA = SCHEMA.replace("result", "freeze")
REVIEW_SCHEMA = SCHEMA.replace("result", "review")
DESIGN = "docs/BANC_M3_E308_RESOURCE_MUSCLE_ATTRIBUTION_DESIGN_20260928.md"
DESIGN_SHA = "0832e55945c14676ad574e99d6160b87e8a66b980cce766b1fc9333ee8d0e795"
OWNED = (
    "tools/analyze_c_banc_m3_resource_muscle_contrast.py",
    "tests/test_c_banc_resource_muscle_contrast.py",
)
REFERENCE_SOURCES = (
    "tools/banc_m3_monotone_recruitment.py",
    "tools/verify_c_banc_m3_native_monotone_body.py",
    "tools/verify_c_banc_m3_recurrent_resource_body.py",
    "tools/verify_c_banc_m3_ssp_recovery_body.py",
    "tools/verify_c_banc_m3_ssp_neutral_body.py",
    "tools/banc_m3_ssp_recovery_synapses.py",
    "flylab/c/storage.py",
    "flylab/c/integrity.py",
    "flylab/c/banc_walking.py",
)
ARM_KEYS = (
    "freeze",
    "review",
    "result",
    "result_sha",
    "output_manifest",
    "source_manifest",
    "ledger",
    "root_audit",
    "trace",
    "activity",
    "recruitment",
    "factors",
    "state_dir",
    "final_state_dir",
)
PATH_KEYS = ("startup_reference", "csr", "protected_roster", "layout")
CONTRACT = {
    "rows": 101,
    "motors": 391,
    "muscles": 96,
    "joints": 42,
    "sources": 7136,
    "observed": 7527,
    "neurons": 158706,
    "edges": 11584852,
    "control_dt": 0.005,
    "neural_dt": 0.00025,
    "neural_steps": 20,
    "reconstruction_atol": 1e-12,
    "direction": "active_E306_minus_neutral_E307",
    "anchor": "control0_predecode",
    "internal_s": 16,
    "compute_s": 10,
    "finalization_s": 6,
    "supervisor_s": 20,
    "cpu_threads": 1,
    "host_bytes": 3 * 1024**3,
    "output_bytes": 64 * 1024**2,
    "free_disk_bytes": 512 * 1024**2,
    "original_attempts": 1,
    "retries": 0,
    "model_calls": 0,
    "neural_ticks": 0,
    "physics_steps": 0,
    "cuda_calls": 0,
    "fit_calls": 0,
    "solver_calls": 0,
}
RESULT_SHAS = {
    "active": "6be280252bfa8a0a3978b0b63405f68474c5899a7615a6ad287e9c38a882e406",
    "neutral": "1ddd4521fa37e75ab8706c37a43e7b516d11a575ac78b1e58489f0ced6c5ddf3",
}
FLAGS = (
    "candidate_adopted",
    "adopted",
    "F1_F8_complete",
    "F2_complete",
    "F3_F8_pass",
    "F6_exposed",
    "natural_gait_complete",
    "ten_second_executed",
    "biological_claim",
)
LIMITATION = (
    "Observed endpoint float64 weight32*delta_rate32 sum, conditional on unit masks and no direct "
    "motor resource edge. Not packed CUDA current, RK-stage current, full drive, new trajectory, or causal intervention."
)


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def read_json(path):
    def reject(value):
        raise ValueError("Nonfinite JSON: " + value)

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe(root, name):
    require(
        isinstance(name, str)
        and name
        and "\\" not in name
        and not Path(name).is_absolute(),
        "Relative POSIX path required",
    )
    target = (Path(root) / name).resolve(strict=True)
    require(
        target.is_relative_to(Path(root).resolve()) and target.is_file(),
        "Unsafe pinned file",
    )
    require(not (Path(root) / name).is_symlink(), "Symlink input prohibited")
    return target


class Pins:
    """Hash each pinned input once; subsequent references must use that same pin."""

    def __init__(self, root, hashes, check=lambda: None):
        require(isinstance(hashes, dict) and hashes, "Input inventory required")
        self.root, self.hashes, self.checked, self.check = (
            Path(root),
            hashes,
            set(),
            check,
        )

    def file(self, name, expected=None):
        self.check()
        require(name in self.hashes, "Unpinned input: " + name)
        digest = self.hashes[name]
        require(
            isinstance(digest, str)
            and len(digest) == 64
            and (expected is None or digest == expected),
            "Pin identity differs",
        )
        path = safe(self.root, name)
        if name not in self.checked:
            require(sha256(path) == digest, "Input SHA mismatch: " + name)
            self.checked.add(name)
        return path

    def json(self, name):
        return read_json(self.file(name))


def verify_manifest(pins, name):
    manifest = pins.json(name)
    require(
        isinstance(manifest.get("files"), dict) and manifest["files"],
        "Manifest files required",
    )
    for relative, entry in manifest["files"].items():
        require(
            isinstance(relative, str)
            and Path(relative).as_posix() == relative
            and ".." not in Path(relative).parts,
            "Unsafe manifest path",
        )
        digest = entry if isinstance(entry, str) else entry["sha256"]
        path = pins.file((Path(name).parent / relative).as_posix(), digest)
        if isinstance(entry, dict):
            require(path.stat().st_size == entry["bytes"], "Manifest size differs")
    return manifest


def selective_state(pins, directory, *, adapter=False):
    """Validate full checkpoint inventory, unpack only adapter leaves and mask."""
    manifest_name = directory + "/manifest.json"
    manifest = verify_manifest(pins, manifest_name)
    require(
        manifest.get("schema") == "flylab.state_store.v3"
        and manifest.get("state_hash")
        == hashlib.sha256(canonical(manifest["files"])).hexdigest(),
        "StateStore identity differs",
    )
    state = pins.json(directory + "/state.json")

    def unpack(value):
        if isinstance(value, dict):
            if "__npy__" in value:
                require(set(value) == {"__npy__"}, "Malformed array leaf")
                leaf = value["__npy__"]
                require(
                    isinstance(leaf, str)
                    and Path(leaf).name == leaf
                    and leaf.endswith(".npy")
                    and leaf in manifest["files"],
                    "Unsafe array leaf",
                )
                return np.load(pins.file(directory + "/" + leaf), allow_pickle=False)
            return {k: unpack(v) for k, v in value.items()}
        if isinstance(value, list):
            return [unpack(v) for v in value]
        return value

    result = {
        "mask": unpack(state["network"]["output_mask"]),
        "cuts": state["cuts"],
        "parameters": state["parameters"],
        "body_hash": state["body"]["modelHash"],
        "control_tick": state["control_tick"],
        "network_tick": state["network"]["tick"],
    }
    if adapter:
        a = state["adapter"]
        result["adapter"] = {
            "hash": a["hash"],
            "schema": a["schema"],
            "groups": a["groups"],
            "parameters": unpack(a["parameters"]),
            "calibration": unpack(a["calibration"]),
            "bank": unpack(a["bank"]),
            "activation": unpack(a["activation"]),
            "original_activation": unpack(a["original"]["activation"]),
        }
    return result


def array(value, shape, dtype, label):
    require(
        isinstance(value, np.ndarray)
        and value.shape == shape
        and value.dtype == np.dtype(dtype)
        and np.isfinite(value).all(),
        "Invalid " + label,
    )
    return value


def indices(value, upper, label):
    require(
        isinstance(value, np.ndarray)
        and value.ndim == 1
        and value.dtype == np.int64
        and len(np.unique(value)) == len(value)
        and np.all((value >= 0) & (value < upper)),
        "Invalid " + label,
    )
    return value


def clocks(data, rows):
    ticks = array(data["control_tick"], (rows,), np.int64, "control clock")
    require(np.array_equal(ticks, np.arange(rows)), "Control ordering differs")
    require(
        np.array_equal(
            array(data["network_tick"], (rows,), np.int64, "neural clock"), ticks * 20
        ),
        "Neural clock differs",
    )
    require(
        np.max(
            np.abs(
                array(data["physics_time_s"], (rows,), np.float64, "physics clock")
                - ticks * 0.005
            )
        )
        <= 2e-8,
        "Physical clock differs",
    )
    if "complete" in data:
        require(
            array(data["complete"], (rows,), bool, "complete").all(),
            "Partial observed row",
        )


def validate_arm(arm, meta):
    t, j = arm["q"].shape
    n = len(meta["motor_ids"])
    m = len(meta["groups"])
    s = len(arm["source_ids"])
    o = len(arm["observed_ids"])
    for data in (arm["activity"], arm["recruitment"], arm["factors"]):
        clocks(data, t)
    for name, shape in {
        "q": (t, j),
        "ltm": (t, len(meta["ltm"])),
        "activation": (t, m),
        "original_activation": (t, m),
        "bank": (t, n, 3),
        "motor_rate": (t, n),
    }.items():
        array(arm[name], shape, np.float64, name)
    for name in ("bank", "activation", "original_activation", "motor_rate"):
        require(np.all(arm[name] >= 0), "Negative saved decoder state")
        require(
            array(
                arm["recruitment"][name + "_available"],
                (t,),
                bool,
                name + " availability",
            ).all(),
            "Unavailable decoder state",
        )
    for name in ("rate", "drive", "adaptation"):
        array(arm["activity"][name], (t, o), np.float32, name)
    require(
        np.all(arm["activity"]["rate"] >= 0)
        and np.all(arm["activity"]["adaptation"] >= 0),
        "Negative native state",
    )
    for name, data, dtype in (
        ("synaptic_resource", arm["activity"], np.float64),
        ("g32", arm["factors"], np.float32),
    ):
        v = array(data[name], (t, s), dtype, name)
        require(np.all((v >= 0) & (v <= 1)), "Resource/factor range differs")
    require(
        np.array_equal(arm["factors"]["source_ids"], arm["source_ids"]),
        "Factor source ordering differs",
    )
    require(
        np.array_equal(arm["activity"]["motor_ids"], meta["motor_ids"]),
        "Motor ordering differs",
    )
    observed = indices(arm["observed_ids"], len(arm["start_mask"]), "observed IDs")
    indices(arm["source_ids"], len(arm["start_mask"]), "source IDs")
    lookup = {int(v): i for i, v in enumerate(observed)}
    require(
        all(int(v) in lookup for v in np.r_[meta["motor_ids"], arm["source_ids"]]),
        "Required motor/source observation missing",
    )
    cols = [lookup[int(v)] for v in meta["motor_ids"]]
    require(
        np.array_equal(
            arm["motor_rate"], arm["activity"]["rate"][:, cols].astype(np.float64)
        ),
        "Motor recording disagreement",
    )
    for name in ("start_mask", "final_mask"):
        v = array(arm[name], (len(arm[name]),), np.float32, name)
        require(np.isin(v, (0.0, 1.0)).all(), "Invalid output mask")
    return lookup


def validate_metadata(meta):
    n = len(meta["motor_ids"])
    m = len(meta["groups"])
    j = len(meta["rest"])
    indices(meta["motor_ids"], 2**31, "motor IDs")
    for name, shape in {
        "rest": (j,),
        "moment": (j, m),
        "kappa": (n,),
        "slopes": (m, n, 3, 2),
        "tau": (3,),
    }.items():
        array(meta[name], shape, np.float64, name)
    array(meta["membership"], (n,), np.int64, "membership")
    indices(meta["fixed_muscles"], m, "fixed muscles")
    indices(meta["ltm"], m, "LTM")
    require(
        np.all((meta["membership"] >= 0) & (meta["membership"] < m))
        and np.all(meta["slopes"] >= 0)
        and np.all(meta["kappa"] >= 0)
        and np.array_equal(meta["tau"], np.array([0.005, 0.01, 0.04])),
        "Invalid transfer parameters",
    )
    for muscle, group in enumerate(meta["groups"]):
        require(
            group["columns"] == np.flatnonzero(meta["membership"] == muscle).tolist()
            and group.get("leg")
            and group.get("muscle"),
            "Group membership differs",
        )
    supported = np.zeros((m, n), bool)
    supported[meta["membership"], np.arange(n)] = True
    require(np.all(meta["slopes"][~supported] == 0), "Slope support differs")


def motor_contributions(bank, meta, muscle):
    """Static saved-bank hinge terms; no filter integration."""
    cols = np.flatnonzero(meta["membership"] == muscle)
    if muscle in meta["fixed_muscles"]:
        return {
            "available": False,
            "path": "saved original_activation",
            "columns": cols,
        }
    b = bank[:, cols, :]
    low = meta["slopes"][muscle, cols, :, 0]
    delta = meta["slopes"][muscle, cols, :, 1] - low
    return {
        "available": True,
        "path": "saved bank hinge",
        "columns": cols,
        "low": low[None, :, :] * b,
        "hinge": delta[None, :, :]
        * np.maximum(b - meta["kappa"][cols][None, :, None], 0.0),
    }


def residual(expected, observed):
    delta = expected - observed
    at = np.unravel_index(int(np.argmax(np.abs(delta))), delta.shape)
    return {
        "max_abs": float(abs(delta[at])),
        "index": list(map(int, at)),
        "pass": bool(abs(delta[at]) <= 1e-12),
    }


def peak(value):
    at = np.unravel_index(int(np.argmax(np.abs(value))), value.shape)
    return {
        "control_tick": int(at[0]),
        "joint": int(at[1]),
        "time_s": float(at[0] * 0.005),
        "signed_value": float(value[at]),
        "max_abs": float(abs(value[at])),
    }


def incoming_contributions(
    active, neutral, csr, motor, *, unit_masks, no_direct_resource
):
    ptr, source, weight = csr["indptr"], csr["indices"], csr["data"]
    a, b = int(ptr[motor]), int(ptr[motor + 1])
    ids = source[a:b].astype(np.int64)
    w = weight[a:b]
    lookup = {int(v): i for i, v in enumerate(active["observed_ids"])}
    available = np.array([int(v) in lookup for v in ids], bool)
    terms = np.full((len(active["q"]), len(ids)), np.nan, np.float64)
    cols = [lookup[int(v)] for v in ids[available]]
    terms[:, available] = w[available].astype(np.float64)[None, :] * (
        active["activity"]["rate"][:, cols].astype(np.float64)
        - neutral["activity"]["rate"][:, cols].astype(np.float64)
    )
    full = bool(available.all() and unit_masks and no_direct_resource)
    sums = {
        label: np.sum(terms[:, available & select], axis=1)
        for label, select in (
            ("positive", w > 0),
            ("negative", w < 0),
            ("zero", w == 0),
        )
    }
    return {
        "motor_id": int(motor),
        "edge_count": len(ids),
        "observed_edges": int(available.sum()),
        "coverage": float(np.mean(available)) if len(ids) else 1.0,
        "missing_ids": np.unique(ids[~available]).tolist(),
        "full_endpoint_sum_available": full,
        "unit_masks_conditional": bool(unit_masks),
        "no_direct_resource": bool(no_direct_resource),
        "limitation": LIMITATION,
    }, {
        "source_ids": ids,
        "weights32": w,
        "available": available,
        "terms64": terms,
        "observed_partial_sum": sum(sums.values()),
        **sums,
        "full_sum": sum(sums.values()) if full else np.full(len(terms), np.nan),
    }


def validate_csr(csr, neurons):
    ptr = csr["indptr"]
    ids = csr["indices"]
    weights = csr["data"]
    array(ptr, (neurons + 1,), np.int32, "CSR indptr")
    array(ids, (len(weights),), np.int32, "CSR indices")
    array(weights, (len(weights),), np.float32, "CSR data")
    require(
        ptr[0] == 0
        and ptr[-1] == len(ids)
        and np.all(np.diff(ptr) >= 0)
        and np.all((ids >= 0) & (ids < neurons)),
        "Invalid CSR storage",
    )


def analyze(
    active,
    neutral,
    metadata,
    startup_q,
    csr,
    roster,
    joint_names,
    *,
    check=lambda: None,
):
    """Generic tiny-compatible math; production shape is enforced by load_inputs."""
    meta = metadata
    validate_metadata(meta)
    lookups = [validate_arm(a, meta) for a in (active, neutral)]
    require(
        np.array_equal(active["observed_ids"], neutral["observed_ids"])
        and np.array_equal(active["source_ids"], neutral["source_ids"]),
        "Active/neutral ordering differs",
    )
    t, j = active["q"].shape
    m = len(meta["groups"])
    require(
        neutral["q"].shape == (t, j) and len(joint_names) == j, "Joint layout differs"
    )
    array(startup_q, (99, j), np.float64, "original startup reference")
    require(t == 101, "101 aligned observations required")
    validate_csr(csr, len(active["start_mask"]))
    checks = {}
    raw = {}
    activation_outputs = []
    for label, arm in (("active", active), ("neutral", neutral)):
        check()
        q = meta["rest"] + 0.02 * (arm["activation"] @ meta["moment"].T)
        ltm = 0.2 * np.tanh(arm["activation"][:, meta["ltm"]] / 10.0)
        checks[label + "_q"] = residual(q[1:], arm["q"][1:])
        checks[label + "_ltm"] = residual(ltm[1:], arm["ltm"][1:])
        reconstructed = arm["original_activation"].copy()
        for muscle in range(m):
            terms = motor_contributions(arm["bank"], meta, muscle)
            if terms["available"]:
                reconstructed[:, muscle] = np.sum(terms["low"], axis=(1, 2)) + np.sum(
                    terms["hinge"], axis=(1, 2)
                )
        checks[label + "_activation"] = residual(reconstructed, arm["activation"])
        activation_outputs.append(reconstructed)
        raw[label + "_anchor_q"] = arm["q"][0].copy()
        raw[label + "_anchor_ltm"] = arm["ltm"][0].copy()
        for field in ("bank", "motor_rate", "activation", "original_activation"):
            raw[label + "_" + field] = arm[field]
    delta = active["q"] - neutral["q"]
    da = active["activation"] - neutral["activation"]
    contributions = 0.02 * meta["moment"][None, :, :] * da[:, None, :]
    checks["delta_q"] = residual(contributions[1:].sum(axis=2), delta[1:])
    raw.update(
        delta_q=delta,
        delta_activation=da,
        muscle_contributions=contributions,
        delta_q_residual=contributions.sum(axis=2) - delta,
        active_q=active["q"],
        neutral_q=neutral["q"],
        startup_q=startup_q,
    )
    largest = peak(delta)
    error = peak(active["q"][1:100] - startup_q)
    error["control_tick"] += 1
    error["time_s"] = error["control_tick"] * 0.005
    per_joint = []
    vectors = []
    for joint in range(j):
        row = int(np.argmax(np.abs(delta[:, joint])))
        vector = contributions[row, joint]
        vectors.append(vector)
        ranking = np.argsort(-np.abs(vector), kind="stable")[:5]
        per_joint.append(
            {
                "joint": joint,
                "name": joint_names[joint],
                "control_tick": row,
                "signed_difference_rad": float(delta[row, joint]),
                "top_muscles": [
                    {
                        "index": int(k),
                        **meta["groups"][int(k)],
                        "signed_contribution_rad": float(vector[k]),
                    }
                    for k in ranking
                ],
            }
        )
    raw["all_muscles_at_joint_maxima"] = np.stack(vectors)
    row, joint = largest["control_tick"], largest["joint"]
    major = np.flatnonzero(meta["moment"][joint] != 0)
    raw["major_muscle_indices"] = major.astype(np.int64)
    motors = []
    muscles = []
    unit_masks = True
    all_required = np.unique(np.r_[active["source_ids"], meta["motor_ids"]])
    for arm in (active, neutral):
        unit_masks &= bool(
            np.all(arm["start_mask"][all_required] == 1)
            and np.all(arm["final_mask"][all_required] == 1)
            and arm["cuts_unchanged_false"]
        )
    edge_ids = roster["edge_ids"]
    require(
        isinstance(edge_ids, np.ndarray)
        and edge_ids.dtype == np.int64
        and edge_ids.ndim == 1
        and len(np.unique(edge_ids)) == len(edge_ids)
        and np.all((edge_ids >= 0) & (edge_ids < len(csr["data"]))),
        "Invalid protected edge roster",
    )
    post = np.searchsorted(csr["indptr"][1:], edge_ids, side="right")
    no_direct = not bool(np.isin(post, meta["motor_ids"]).any())
    for muscle in major:
        a = motor_contributions(active["bank"], meta, int(muscle))
        b = motor_contributions(neutral["bank"], meta, int(muscle))
        entry = {
            "index": int(muscle),
            **meta["groups"][int(muscle)],
            "path": a["path"],
            "motor_decomposition_available": a["available"],
        }
        if a["available"]:
            terms = (a["low"] + a["hinge"]) - (b["low"] + b["hinge"])
            raw[f"muscle_{muscle}_motor_filter_delta"] = terms
            raw[f"muscle_{muscle}_motor_ids"] = meta["motor_ids"][a["columns"]]
            entry["motor_contribution_residual"] = residual(
                terms.sum(axis=(1, 2)), da[:, muscle]
            )
        muscles.append(entry)
        for col in a["columns"]:
            motor = int(meta["motor_ids"][col])
            check()
            start, end = int(csr["indptr"][motor]), int(csr["indptr"][motor + 1])
            incoming_ids = csr["indices"][start:end]
            masks_for_row = unit_masks and all(
                bool(np.all(arm[k][incoming_ids] == 1))
                for arm in (active, neutral)
                for k in ("start_mask", "final_mask")
            )
            summary, arrays = incoming_contributions(
                active,
                neutral,
                csr,
                motor,
                unit_masks=masks_for_row,
                no_direct_resource=no_direct,
            )
            for key, value in arrays.items():
                raw[f"motor_{motor}_{key}"] = value
            raw[f"motor_{motor}_bank_delta"] = (
                active["bank"][:, col] - neutral["bank"][:, col]
            )
            raw[f"motor_{motor}_rate_delta"] = (
                active["motor_rate"][:, col] - neutral["motor_rate"][:, col]
            )
            oc = lookups[0][motor]
            for field in ("drive", "adaptation", "rate"):
                for arm_name, arm in (("active", active), ("neutral", neutral)):
                    raw[f"motor_{motor}_{arm_name}_{field}32"] = arm["activity"][field][
                        :, oc
                    ]
                raw[f"motor_{motor}_{field}_delta64"] = active["activity"][field][
                    :, oc
                ].astype(np.float64) - neutral["activity"][field][:, oc].astype(
                    np.float64
                )
            terms = arrays["terms64"][row]
            available = arrays["available"]
            ranking = np.flatnonzero(available)[
                np.argsort(-np.abs(terms[available]), kind="stable")[:5]
            ]
            summaries = []
            source_lookup = {int(v): k for k, v in enumerate(active["source_ids"])}
            for k in ranking:
                source = int(arrays["source_ids"][k])
                observed = lookups[0][source]
                slot = source_lookup.get(source)
                summaries.append(
                    {
                        "source_id": source,
                        "weight32": float(arrays["weights32"][k]),
                        "signed_term": float(terms[k]),
                        "active_rate32": float(
                            active["activity"]["rate"][row, observed]
                        ),
                        "neutral_rate32": float(
                            neutral["activity"]["rate"][row, observed]
                        ),
                        "resource_observed": slot is not None,
                        "active_R": None
                        if slot is None
                        else float(active["activity"]["synaptic_resource"][row, slot]),
                        "neutral_R": None
                        if slot is None
                        else float(neutral["activity"]["synaptic_resource"][row, slot]),
                        "active_g32": None
                        if slot is None
                        else float(active["factors"]["g32"][row, slot]),
                        "neutral_g32": None
                        if slot is None
                        else float(neutral["factors"]["g32"][row, slot]),
                    }
                )
            summary["largest_terms_at_joint_difference_peak"] = summaries
            summary["observed_partial_input_delta_at_peak"] = {
                key: float(arrays[key][row])
                for key in ("positive", "negative", "zero", "observed_partial_sum")
            }
            summary["external_drive_and_adaptation_scope"] = (
                "Saved observed motor drive/adaptation/rate time series and deltas in raw; synaptic subtotal is not entire effective input."
            )
            motors.append(summary)
    checks["neutral_resource_unity"] = {
        "pass": bool(
            np.all(neutral["activity"]["synaptic_resource"] == 1.0)
            and np.all(neutral["factors"]["g32"] == 1.0)
        )
    }
    checks["initial_resource_unity"] = {
        "pass": bool(
            all(
                np.all(arm["activity"]["synaptic_resource"][0] == 1.0)
                and np.all(arm["factors"]["g32"][0] == 1.0)
                for arm in (active, neutral)
            )
        )
    }
    valid = all(v["pass"] for v in checks.values()) and all(
        e.get("motor_contribution_residual", {"pass": True})["pass"] for e in muscles
    )
    return {
        "run_valid": bool(valid),
        "direction": CONTRACT["direction"],
        "units": {"q": "rad", "input": "native model input"},
        "checks": checks,
        "anchor_absolute_reconstruction_assessed": False,
        "largest_active_minus_neutral": largest,
        "separate_active_startup_error_peak": error,
        "all_joint_maxima": per_joint,
        "major_muscles": muscles,
        "motors": motors,
        "unit_mask_evidence": "start/final checkpoints plus unchanged cuts/run contract; not measured mask history",
        "unit_masks_supported": bool(unit_masks),
        "selected_motor_target_edges": int(np.isin(post, meta["motor_ids"]).sum()),
        "full_current_claim": False,
        "causal_claim": False,
        "limitation": LIMITATION,
    }, raw


def validate_producer(arm, spec, pins):
    number = 306 if arm == "active" else 307
    result = pins.json(spec["result"])
    audit = pins.json(spec["root_audit"])
    ledger = pins.json(spec["ledger"])
    require(
        pins.hashes[spec["result"]] == RESULT_SHAS[arm],
        "Literal producer result differs",
    )
    require(
        pins.json(spec["result_sha"]).get("sha256") == RESULT_SHAS[arm],
        "Producer result receipt differs",
    )
    require(
        result.get("schema")
        == f"flylab.m3.e{number}.ssp-{'recovery' if number == 306 else 'neutral'}-body.result.v1"
        and result.get("diagnostic_id") == f"E{number}"
        and result.get("run_valid") is True
        and result.get("publication_status") == "complete"
        and all(
            result.get(k) is False
            for k in FLAGS
            if k != "biological_claim" or number == 307
        )
        and (number != 306 or "biological_claim" not in result),
        "Producer invalid/promoted",
    )
    require(
        result.get("actual_startup_pass") is (arm == "neutral")
        and result.get("neutral_prefix_valid") is True
        and result.get("ssp_neural_endpoint_exact") is True,
        "Original producer startup status differs",
    )
    require(
        audit.get("schema") == f"flylab.m3.e{number}.root-original-raw-audit.v1"
        and audit.get("pass") is True
        and audit.get("diagnostic_run_valid") is True
        and audit.get(
            "actual_startup_pass" if number == 306 else "original_strict_startup_pass"
        )
        is (arm == "neutral"),
        "Root producer audit differs",
    )
    for flag in (
        "candidate_adopted",
        "F1_F8_complete",
        "F2_complete",
        "F6_exposed",
        "biological_claim",
    ):
        require(audit.get(flag) is False, "Root audit promotion differs")
    if number == 306:
        require(audit.get("candidate_rejected") is True, "Original rejection lost")
    count = 2 if number == 306 else 1
    cost = {
        "construction_calls": count,
        "known_executed_setup_ticks": 280 * count,
        "recorded_capture_setup_ticks": 140 * count,
        "known_settling_physics_steps": 2000 * count,
        "scientific_neural_ticks": 2020 if number == 306 else 2000,
        "scientific_physics_steps": 5050 if number == 306 else 5000,
        "unknown_cost_arms": [],
        "unknown_setup_calls": [],
    }
    require(
        result.get("cost") == cost and audit.get("cost") == cost,
        "Producer measured costs differ",
    )
    require(
        ledger.get("original_attempts") == 1
        and ledger.get("retries") == 0
        and ledger.get("fit_calls") == 0,
        "Producer attempts differ",
    )
    entry = ledger["arms"]["candidate" if number == 306 else "neutral"]
    require(
        entry.get("completed_controls") == 100
        and entry.get("native_samples") == 5000
        and entry.get("close_returned") is True
        and entry.get("clock_valid") is True
        and entry.get("continuation_entered") is False
        and round(entry.get("actual_neural_s", -1) / 0.00025) == 2000
        and round(entry.get("actual_physical_s", -1) / 0.0001) == 5000,
        "Producer completed arm differs",
    )
    frozen = pins.json(spec["freeze"])
    review = pins.json(spec["review"])
    require(
        review.get("schema")
        == f"flylab.m3.e{number}.ssp-{'recovery' if number == 306 else 'neutral'}-body.review.v1"
        and review.get("approved") is True
        and review.get("blocking_defects") == []
        and review.get("freeze_sha256") == pins.hashes[spec["freeze"]],
        "Producer review differs",
    )
    sources = pins.json(spec["source_manifest"])
    require(
        sources.get("schema") == f"flylab.m3.e{number}.sources.v1"
        and sources.get("diagnostic_id") == f"E{number}",
        "Producer source receipt differs",
    )
    for inventory in (
        sources["source_sha256"],
        sources["owned_sha256"],
        review["source_sha256"],
    ):
        for name, digest in inventory.items():
            pins.file(name, digest)
    require(
        frozen.get("schema")
        == f"flylab.m3.e{number}.ssp-{'recovery' if number == 306 else 'neutral'}-body.freeze.v1",
        "Producer freeze schema differs",
    )
    verify_manifest(pins, spec["output_manifest"])
    return {
        "schema": result["schema"],
        "result_sha256": RESULT_SHAS[arm],
        "cost": cost,
        "startup_pass": arm == "neutral",
    }


def verified_gate(
    freeze_path,
    freeze_sha256,
    review_path,
    review_sha256,
    *,
    root=ROOT,
    check=lambda: None,
):
    require(
        sha256(freeze_path) == freeze_sha256 and sha256(review_path) == review_sha256,
        "Freeze/review SHA differs",
    )
    freeze, review = read_json(freeze_path), read_json(review_path)
    require(
        freeze.get("schema") == FREEZE_SCHEMA
        and freeze.get("contract") == CONTRACT
        and freeze.get("output") == OUTPUT
        and freeze.get("design") == {"path": DESIGN, "sha256": DESIGN_SHA},
        "Literal E308 contract differs",
    )
    require(
        review.get("schema") == REVIEW_SCHEMA
        and review.get("approved") is True
        and review.get("blocking_defects") == []
        and review.get("freeze_sha256") == freeze_sha256
        and review.get("source_sha256") == freeze.get("owned_sha256"),
        "E308 review blocks analysis",
    )
    require(
        set(freeze["owned_sha256"]) == set(OWNED)
        and set(REFERENCE_SOURCES) <= set(freeze["reference_source_sha256"]),
        "Source roster differs",
    )
    hashes = dict(freeze["input_sha256"])
    for name, digest in {
        **freeze["owned_sha256"],
        **freeze["reference_source_sha256"],
        DESIGN: DESIGN_SHA,
    }.items():
        require(name not in hashes or hashes[name] == digest, "Conflicting source pin")
        hashes[name] = digest
    pins = Pins(root, hashes, check)
    for name in hashes:
        pins.file(name)
    require(
        set(freeze["arms"]) == {"active", "neutral"}
        and set(freeze["paths"]) == set(PATH_KEYS),
        "Input schema differs",
    )
    receipts = {}
    for arm, spec in freeze["arms"].items():
        require(set(spec) == set(ARM_KEYS), "Arm path keys differ")
        receipts[arm] = validate_producer(arm, spec, pins)
    return freeze, pins, receipts


def load_npz(pins, name):
    with np.load(pins.file(name), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def load_inputs(freeze, pins):
    arms = {}
    meta = None
    identities = None
    for label, spec in freeze["arms"].items():
        start = selective_state(pins, spec["state_dir"], adapter=True)
        final = selective_state(pins, spec["final_state_dir"])
        require(
            start["control_tick"] == start["network_tick"] == 0
            and final["control_tick"] == 100
            and final["network_tick"] == 2000,
            "Checkpoint clocks differ",
        )
        require(
            start["cuts"] == final["cuts"]
            and set(start["cuts"]) == {"circuit", "descending", "motor", "sensory"}
            and all(v is False for v in start["cuts"].values())
            and start["parameters"] == final["parameters"],
            "Mask-change/cuts contract differs",
        )
        require(
            start["parameters"].get("neural_dt_s") == 0.00025,
            "Original neural dt differs",
        )
        a = start["adapter"]
        require(
            final["body_hash"] == start["body_hash"],
            "Start/final body identity differs",
        )
        require(
            a["schema"] == "flylab.m3-monotone-recruitment-adapter.v1",
            "Adapter schema differs",
        )
        current = {**a["calibration"], **a["parameters"], "groups": a["groups"]}
        identity = {
            "hash": a["hash"],
            "body_hash": start["body_hash"],
            "parameters": start["parameters"],
        }
        if meta is None:
            meta, identities = current, identity
        else:
            require(
                identity == identities
                and final["body_hash"] == identities["body_hash"],
                "Adapter/body/parameters differ",
            )
            for k, v in meta.items():
                require(
                    np.array_equal(v, current[k])
                    if isinstance(v, np.ndarray)
                    else v == current[k],
                    "Shared calibration/order differs: " + k,
                )
        require(
            np.all(a["bank"] == 0)
            and np.all(a["activation"] == 0)
            and np.all(a["original_activation"] == 0),
            "Initial adapter anchor differs",
        )
        activity = load_npz(pins, spec["activity"])
        r = load_npz(pins, spec["recruitment"])
        f = load_npz(pins, spec["factors"])
        require(
            str(r["schema"]) == "flylab.m3.e150.native-monotone-state.v1",
            "Saved recruitment schema differs",
        )
        trace = pins.json(spec["trace"])
        require(
            trace.get("schema") == "flylab.m3.e291.trace.v1"
            and len(trace["samples"]) == 101,
            "Trace schema/length differs",
        )
        rows = trace["samples"]
        for i, row in enumerate(rows):
            require(
                row.get("control_tick") == i
                and type(row.get("time_s")) in (float, int)
                and abs(row["time_s"] - i * 0.005) <= 2e-8
                and row.get("fault") is None
                and row.get("neural_request") is None,
                "Trace time/order/fault/intervention differs",
            )
            if i:
                interval = row["command_interval"]
                require(
                    interval.get("start_tick") == i - 1
                    and interval.get("end_tick") == i
                    and abs(interval["start_s"] - (i - 1) * 0.005) <= 2e-8
                    and abs(interval["end_s"] - i * 0.005) <= 2e-8,
                    "Trace command interval differs",
                )
        ltm_names = [meta["groups"][int(k)]["leg"] + "_tarsus" for k in meta["ltm"]]
        arm = {
            "q": np.asarray([row["joint_targets_rad"] for row in rows], np.float64),
            "ltm": np.asarray(
                [[row["tendon_inputs"][k] for k in ltm_names] for row in rows],
                np.float64,
            ),
            **{
                k: r[k]
                for k in ("bank", "activation", "original_activation", "motor_rate")
            },
            "activity": activity,
            "recruitment": r,
            "factors": f,
            "observed_ids": activity["observed_ids"],
            "source_ids": activity["source_ids"],
            "start_mask": start["mask"],
            "final_mask": final["mask"],
            "cuts_unchanged_false": True,
        }
        require(
            arm["q"].shape == (101, 42)
            and len(arm["source_ids"]) == 7136
            and len(arm["observed_ids"]) == 7527
            and len(arm["start_mask"]) == 158706
            and len(arm["final_mask"]) == 158706,
            "Production observation axes differ",
        )
        arms[label] = arm
    require(
        len(meta["motor_ids"]) == 391
        and len(meta["groups"]) == 96
        and meta["moment"].shape == (42, 96)
        and meta["ltm"].shape == (6,),
        "Production decoder axes differ",
    )
    paths = freeze["paths"]
    startup = load_npz(pins, paths["startup_reference"])
    target = startup["target_command"]
    validate_startup_reference(startup, 42)
    csr = load_npz(pins, paths["csr"])
    require(csr["data"].shape == (11584852,), "Production CSR nnz differs")
    roster = load_npz(pins, paths["protected_roster"])
    require(
        np.array_equal(roster["motor_ids"], meta["motor_ids"])
        and np.array_equal(roster["source_ids"], arms["active"]["source_ids"]),
        "Protected roster order differs",
    )
    validate_csr(csr, 158706)
    edge = indices(roster["edge_ids"], len(csr["data"]), "Protected edge IDs")
    post = np.searchsorted(csr["indptr"][1:], edge, side="right")
    require(
        np.array_equal(roster["post_ids"], post)
        and np.array_equal(roster["pre_ids"], csr["indices"][edge])
        and np.array_equal(roster["weights32"], csr["data"][edge]),
        "Protected roster CSR identity differs",
    )
    layout = pins.json(paths["layout"])
    require(layout["body_hash"] == identities["body_hash"], "Layout body hash differs")
    leg_axes = layout["leg_actuator_indices"]
    require(
        isinstance(leg_axes, list)
        and len(leg_axes) == 6
        and all(len(row) == 7 for row in leg_axes),
        "Original leg layout axes differ",
    )
    names = [
        layout["actuator_joint_names"][int(k)]
        for k in np.concatenate([np.asarray(row) for row in leg_axes])
    ]
    require(len(names) == 42, "Original 42-joint layout differs")
    return (
        arms["active"],
        arms["neutral"],
        meta,
        target.copy(),
        csr,
        roster,
        names,
    )


def validate_startup_reference(startup, joints):
    array(
        startup["target_command"], (99, joints), np.float64, "Original99 target command"
    )
    time_s = array(startup["time_s"], (2001,), np.float64, "Original reference clock")
    mask = array(startup["mask"], (2001,), bool, "Original reference mask")
    require(
        np.array_equal(np.flatnonzero(mask), np.arange(1, 100))
        and np.max(np.abs(time_s[mask] - np.arange(1, 100) * 0.005)) <= 2e-8,
        "Original99 startup selection differs",
    )


def verify_publication(output):
    """A seal alone never establishes success after recorded invalidation."""
    output = Path(output)
    require(
        not (output / "invalidation.json").exists()
        and not (output / "failure.json").exists(),
        "Publication explicitly invalidated",
    )
    result = read_json(output / "result.json")
    seal = read_json(output / "result-sha256.json")
    require(
        result.get("schema") == SCHEMA
        and result.get("publication_status") == "complete"
        and seal.get("sha256") == sha256(output / "result.json"),
        "Result seal differs",
    )
    manifest = read_json(output / "output-manifest.json")
    require(
        manifest.get("schema") == SCHEMA.replace("result", "output-manifest"),
        "Output manifest schema differs",
    )
    for name, entry in manifest["files"].items():
        require(
            Path(name).name == name
            and sha256(output / name) == entry["sha256"]
            and (output / name).stat().st_size == entry["bytes"],
            "Output manifest differs",
        )
    return result


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(canonical(value))
    os.replace(temp, path)


def peak_rss_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (k, ctypes.c_size_t)
                for k in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        )
        require(
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb
            )
            != 0,
            "RSS measurement failed",
        )
        return int(c.PeakWorkingSetSize)
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def run(
    freeze_path,
    freeze_sha256,
    review_path,
    review_sha256,
    output,
    *,
    wall_clock=time.perf_counter,
    root=ROOT,
    gate=verified_gate,
    loader=load_inputs,
    analysis=analyze,
    writer=write_json,
    rss=peak_rss_bytes,
):
    """One attempt; original exception owns partial evidence, never retry."""
    began = wall_clock()
    output = Path(output)
    require(
        output.resolve() == (Path(root) / OUTPUT).resolve(),
        "Literal fresh output required",
    )
    require(not output.exists(), "Output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    cost = {
        "original_analysis_requested": 1,
        "original_analysis_completed": 0,
        "gate_requested": 0,
        "gate_completed": 0,
        "load_requested": 0,
        "load_completed": 0,
        "analysis_requested": 0,
        "analysis_completed": 0,
        "model_calls": 0,
        "neural_ticks": 0,
        "physics_steps": 0,
        "cuda_calls": 0,
        "solver_calls": 0,
        "fit_calls": 0,
        "retries": 0,
    }
    partial = {
        "schema": SCHEMA,
        "run_valid": False,
        "publication_status": "partial",
        "cost": cost,
    }

    def check():
        require(wall_clock() - began <= 16, "E308 internal deadline exceeded")
        require(rss() <= CONTRACT["host_bytes"], "Host RSS limit exceeded")

    try:
        require(
            shutil.disk_usage(output.parent).free >= CONTRACT["free_disk_bytes"],
            "Free disk limit exceeded",
        )
        cost["gate_requested"] += 1
        freeze, pins, receipts = gate(
            freeze_path,
            freeze_sha256,
            review_path,
            review_sha256,
            root=root,
            check=check,
        )
        cost["gate_completed"] += 1
        partial["producer_receipts"] = receipts
        writer(
            output / "input-receipt.json",
            {
                "schema": SCHEMA.replace("result", "inputs"),
                "input_sha256": freeze["input_sha256"],
                "freeze_sha256": freeze_sha256,
                "review_sha256": review_sha256,
            },
        )
        writer(
            output / "source-manifest.json",
            {
                "schema": SCHEMA.replace("result", "sources"),
                "owned_sha256": freeze["owned_sha256"],
                "reference_source_sha256": freeze["reference_source_sha256"],
                "design": freeze["design"],
                "no_model_imports": True,
            },
        )
        check()
        cost["load_requested"] += 1
        inputs = loader(freeze, pins)
        cost["load_completed"] += 1
        check()
        compute_start = wall_clock()
        cost["analysis_requested"] += 1

        def compute_check():
            check()
            require(
                wall_clock() - compute_start <= 10, "E308 compute deadline exceeded"
            )

        with threadpool_limits(limits=1):
            summary, raw = analysis(*inputs, check=compute_check)
        cost["analysis_completed"] += 1
        partial["summary"] = summary
        partial["raw_keys"] = list(raw)
        compute_check()
        cost["compute_wall_s"] = wall_clock() - compute_start
        final_start = wall_clock()
        np.savez_compressed(output / "contributions.npz", **raw)
        writer(output / "summary.json", summary)
        check()
        cost["original_analysis_completed"] += 1
        cost["wall_s"] = wall_clock() - began
        cost["peak_rss_bytes"] = rss()
        result = {
            **partial,
            "run_valid": summary["run_valid"],
            "publication_status": "complete",
            "candidate_adopted": False,
            "F1_F8_complete": False,
            "F2_complete": False,
            "F3_F8_pass": False,
            "F6_exposed": False,
            "M2_changed": False,
            "UI_changed": False,
            "natural_gait_complete": False,
            "biological_claim": False,
            "full_current_claim": False,
            "causal_claim": False,
        }
        writer(output / "result.json", result)
        writer(
            output / "result-sha256.json",
            {
                "path": "result.json",
                "schema": SCHEMA.replace("result", "result-sha256"),
                "sha256": sha256(output / "result.json"),
            },
        )
        files = {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in output.iterdir()
            if p.is_file()
        }
        writer(
            output / "output-manifest.json",
            {"schema": SCHEMA.replace("result", "output-manifest"), "files": files},
        )
        require(
            sum(p.stat().st_size for p in output.iterdir() if p.is_file())
            <= CONTRACT["output_bytes"],
            "Output byte cap exceeded",
        )
        require(wall_clock() - final_start <= 6, "E308 finalization deadline exceeded")
        check()
        return result
    except BaseException as error:
        error.partial_result = partial
        error.cost_receipt = cost
        error.partial_raw = locals().get("raw", {})
        try:
            cost["wall_s"] = wall_clock() - began
            cost["peak_rss_bytes"] = rss()
        except BaseException as secondary:  # noqa: BLE001 -- preserve first measurement/storage exception
            error.add_note("Failure metering error: " + repr(secondary))
        partial["error"] = repr(error)
        partial["publication_status"] = "invalid_partial"
        try:
            writer(
                output / "invalidation.json",
                {
                    "schema": SCHEMA.replace("result", "invalidation"),
                    "run_valid": False,
                    "reason": repr(error),
                },
            )
        except BaseException as secondary:  # noqa: BLE001 -- retain original I/O or cancellation
            error.add_note(
                "Explicit invalidation publication failure: " + repr(secondary)
            )
        for receipt in ("result-sha256.json", "output-manifest.json"):
            try:
                (output / receipt).unlink(missing_ok=True)
            except BaseException as secondary:  # noqa: BLE001 -- preserve primary even during cancellation
                error.add_note("Receipt withdrawal failure: " + repr(secondary))
        partial["error"] = repr(error)
        partial["publication_status"] = "invalid_partial"
        for name in ("failure.json", "result.json"):
            try:
                writer(output / name, partial)
            except BaseException as secondary:  # noqa: BLE001 -- preserve primary even during cancellation
                error.add_note("Partial publication failure: " + repr(secondary))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "source-review", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("freeze-sha256", "source-review-sha256"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    result = run(
        args.freeze,
        args.freeze_sha256,
        args.source_review,
        args.source_review_sha256,
        args.output,
    )
    print(
        json.dumps(
            {
                "run_valid": result["run_valid"],
                "publication_status": result["publication_status"],
            }
        )
    )
    return 0 if result["run_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
