"""E21: explicit DNg100 target laterality on the fixed author BANC network.

Metadata side is not the motor target side for these contralateral DNs.
This comparison does not replace the whole graph or validate body control.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.probe_c_banc_reference import (
    AREA_SHA,
    FAST_FETI,
    MATRIX_SHA,
    TABLE_SHA,
    checked_csv,
    load_reference,
    sha,
    write,
)

# Pugliese_2026 5626b731: Methods, Figure 3 cells 6/34/43 and the
# consistentColumns table's MANC matches. Do not invert unrelated cell types.
DNG100 = (
    ("left", "720575941500851362", "right", "id:10093"),
    ("right", "720575941626500746", "left", "id:10339"),
)
PARAMETERS_SHA = "0d7baa33ab121b03c422b2770d4082aa9a5f26a8d867baba5f97be04ecb334d1"
MODEL_HASHES = {
    "mean": "0ef7d5f0db4d98b2c78be6d1f8f8886ff2c97138782c5f6e76c6b004e55cb366",
    "seed1": "5b67d84b408ae3f2ac2022ec9db344e7acae29a0a84236874842da292253f5bb",
}


def descending_targets(rows):
    """Resolve exact IDs, validating source labels without deriving target side."""
    roots = [row["pt_root_id"] for row in rows]
    if len(roots) != len(set(roots)):
        raise ValueError("Ambiguous descending root IDs")
    lookup = {root: i for i, root in enumerate(roots)}
    result = {}
    for target_side, root, soma_side, match in DNG100:
        if root not in lookup:
            raise ValueError("Missing fixed DNg100 root")
        index = lookup[root]
        row = rows[index]
        if (
            row.get("cell_type") != "DNg100"
            or row.get("side") != soma_side
            or row.get("manc_121_match_id") != match
        ):
            raise ValueError("Fixed DNg100 source identity mismatch")
        result[target_side] = {
            "index": index,
            "root_id": root,
            "source_side": soma_side,
            "target_vnc_side": target_side,
            "manc_match": match,
        }
    return result


def load_parameters(path, roots, areas):
    if sha(path / "parameters.npz") != PARAMETERS_SHA:
        raise ValueError("Fixed E20 parameter checksum mismatch")
    with np.load(path / "parameters.npz", allow_pickle=False) as data:
        params = {key: data[key].copy() for key in data.files}
    if not np.array_equal(roots, params["root_ids"]) or not np.array_equal(
        areas, params["areas_um2"], equal_nan=True
    ):
        raise ValueError("Prepared parameter root/area mismatch")
    return params


def motor_groups(table):
    groups = {}
    for side in ("left", "right"):
        motors = [
            i
            for i, row in enumerate(table)
            if row["super_class"] == "motor" and row["side"] == side
        ]
        groups[side] = np.asarray(motors, np.int64)
        groups[side + "_front_leg"] = np.array(
            [i for i in motors if table[i]["body_part_effector"] == "front_leg"],
            np.int64,
        )
    return groups


def run(args):
    from scipy.sparse import csr_matrix

    from flylab.c.research_rate import ResearchRateNetwork

    began = time.perf_counter()
    weights, roots, areas, table = load_reference(
        args.matrix_reference, args.author_reference
    )
    rows = checked_csv(
        args.author_reference
        / "source/data/banc t1 premotor/wTable_20260217_fullData_consistentColumns.csv",
        AREA_SHA,
    )
    dns = descending_targets(rows)
    groups = motor_groups(table)
    lookup = {root: i for i, root in enumerate(roots)}
    pair = np.array([lookup[root] for root in FAST_FETI], np.int64)
    params = load_parameters(args.parameters, roots, areas)
    previous = json.loads((args.previous / "report.json").read_text(encoding="utf8"))
    controls = {}
    for profile, expected_hash in MODEL_HASHES.items():
        case = next(c for c in previous["cases"] if c["name"] == profile + "-dn400")
        path = args.previous / (case["name"] + ".npz")
        if case["model_hash"] != expected_hash or sha(path) != case["trace_sha256"]:
            raise ValueError("Previous comparison checksum mismatch")
        controls[profile] = case["trace_sha256"]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "parameters.npz").write_bytes(
        (args.parameters / "parameters.npz").read_bytes()
    )
    sources = (
        Path(__file__),
        Path("tools/probe_c_banc_reference.py"),
        Path("flylab/c/research_rate.py"),
        Path("GOAL_PLAN.md"),
    )
    (args.out / "source").mkdir()
    for path in sources:
        (args.out / "source" / path.name).write_bytes(path.read_bytes())
    spec = {
        "test": "E21 DNg100 target-side comparison",
        "simulated_neurons": len(roots),
        "source_signed_pairs": int(np.count_nonzero(weights)),
        "matrix_sha256": MATRIX_SHA,
        "table_sha256": TABLE_SHA,
        "area_sha256": AREA_SHA,
        "parameters_sha256": PARAMETERS_SHA,
        "model_hashes": MODEL_HASHES,
        "descending_targets": dns,
        "motor_groups": {name: ids.tolist() for name, ids in groups.items()},
        "fast_feti_indices": pair.tolist(),
        "previous_controls": controls,
        "conditions": ["left", "right", "both"],
        "dn_input_units": 400,
        "dn_on_ms": 20,
        "dn_off_ms": 1600,
        "end_ms": 2000,
        "neural_dt_s": 0.0001,
        "observe_dt_s": 0.001,
        "wall_limit_s": 180,
        "neural_model_limit_s": 12,
        "physics_model_seconds": 0,
        "sources": {str(path): sha(path) for path in sources},
        "exact_paper_run_reproduction": False,
        "controller_adoption": False,
    }
    write(args.out / "spec.json", spec)
    matrix = csr_matrix(weights.T.astype(np.float32) * np.float32(0.03))
    del weights
    cases = []
    for profile, expected_hash in MODEL_HASHES.items():
        network = ResearchRateNetwork(
            matrix,
            *[params[profile + "_" + key] for key in ("tau", "a", "threshold", "cap")],
            device="cuda",
        )
        if network.identity != expected_hash:
            raise ValueError("Fixed E20 model hash mismatch")
        for condition in spec["conditions"]:
            network.reset()
            sides = ("left", "right") if condition == "both" else (condition,)
            targets = np.array([dns[side]["index"] for side in sides], np.int64)
            values = np.zeros((2000, len(targets)), np.float32)
            values[20:1600] = 400
            rates = np.zeros((2001, len(roots)), np.float32)
            drive = np.zeros(len(roots), np.float32)
            for tick in range(2000):
                drive.fill(0)
                drive[targets] = values[tick]
                network.advance(drive)
                rates[tick + 1] = network.readout()
                if time.perf_counter() - began > 180:
                    raise TimeoutError("E21 total execution wall limit")
            prefix_error = None
            if condition == "right":
                with np.load(
                    args.previous / (profile + "-dn400.npz"), allow_pickle=False
                ) as old:
                    prefix_error = float(
                        np.abs(rates[:1601] - old["rates"][:1601]).max()
                    )
            valid = bool(
                np.isfinite(rates).all()
                and (rates >= 0).all()
                and not np.any(rates[:21])
                and network.tick == 20000
                and (prefix_error is None or prefix_error <= 0.001)
            )
            name = profile + "-" + condition
            path = args.out / (name + ".npz")
            np.savez_compressed(
                path,
                rates=rates,
                root_ids=roots,
                input_indices=targets,
                input_drive=values,
                time_s=np.arange(2001) * 0.001,
                neural_tick=network.tick,
                output_mask=network.output_mask.cpu().numpy(),
            )
            peaks = {
                name: float(rates[21:1601, ids].max()) for name, ids in groups.items()
            }
            recruited = {
                name: int((rates[21:1601, ids].max(axis=0) >= 2).sum())
                for name, ids in groups.items()
            }
            case = {
                "name": name,
                "profile": profile,
                "condition": condition,
                "valid": valid,
                "model_hash": network.identity,
                "device": network.device,
                "neural_model_seconds": network.tick * network.dt,
                "previous_prefix_max_error": prefix_error,
                "stimulus_peak": peaks,
                "recruited_motors_ge_2": recruited,
                "stimulus_peak_fast_feti": rates[21:1601, pair].max(axis=0).tolist(),
                "final_tail_fast_feti": rates[-50:, pair]
                .astype(np.float64)
                .mean(axis=0)
                .tolist(),
                "final_active_neurons_ge_2": int((rates[-1] >= 2).sum()),
                "trace_sha256": sha(path),
            }
            cases.append(case)
            write(args.out / "cases.json", cases)
            print(json.dumps(case), flush=True)
        del network
        gc.collect()
    outcomes = {}
    for profile in MODEL_HASHES:
        outcomes[profile] = {}
        for case in (c for c in cases if c["profile"] == profile):
            condition = case["condition"]
            left = case["stimulus_peak"]["left_front_leg"]
            right = case["stimulus_peak"]["right_front_leg"]
            passed = (
                (left >= 2 and right >= 2)
                if condition == "both"
                else (
                    left >= 2 and left > right
                    if condition == "left"
                    else right >= 2 and right > left
                )
            )
            outcomes[profile][condition] = (
                "not_tested"
                if not case["valid"]
                else ("supported" if passed else "refuted")
            )
    report = {
        "status": "COMPLETE",
        "validity": "valid" if all(c["valid"] for c in cases) else "invalid",
        "cases": cases,
        "outcomes": outcomes,
        "wall_seconds": time.perf_counter() - began,
        "neural_model_seconds": sum(c["neural_model_seconds"] for c in cases),
        "physics_model_seconds": 0,
        "F3_F6": "fail",
        "controller_adoption": False,
        "full_goal_completed": False,
    }
    write(args.out / "report.json", report)
    print(
        json.dumps({"outcomes": outcomes, "wall_seconds": report["wall_seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "matrix-reference",
        "author-reference",
        "parameters",
        "previous",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("out must not already exist; preserve previous evidence")
    try:
        run(args)
    except Exception as error:
        if args.out.is_dir() and not (args.out / "report.json").exists():
            write(
                args.out / "failure.json", {"status": "INVALID", "error": repr(error)}
            )
        raise
