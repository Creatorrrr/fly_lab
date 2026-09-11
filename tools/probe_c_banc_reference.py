"""E20: fixed author BANC matrix under a new, explicitly matched input task.

Prepare with the isolated reference JAX runtime; run with fly_lab's CUDA
runtime. The paper's exact run configuration is unknown. This diagnostic
does not replace the whole graph or certify a biological controller.
"""

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MATRIX_SHA = "83197529fa336b5f9ce400cf689f299fa3e8f6f5379947ae7df5ec004869141f"
TABLE_SHA = "f934f42bb142bc6f3de176752ab69ed094c08eb3cab7b2c6f35f0a1610709fa1"
AREA_SHA = "6f87bf62227e160754523418bfbac35fc8971a90e4ed54574bf769b08332e285"
GRAPH_HASH = "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"
FAST_FETI = ("720575941481179066", "720575941639281525")
# Historical E20 input: left soma, RIGHT VNC target. Keep the old ID and
# evidence; probe_c_banc_descending tests both target sides explicitly.
DN = "720575941626500746"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def checked_csv(path, expected):
    if sha(path) != expected:
        raise ValueError("Pinned source checksum mismatch: " + str(path))
    with path.open(encoding="utf8", newline="") as file:
        return list(csv.DictReader(file))


def load_reference(matrix_reference, author_reference):
    """Retain every author node, including IDs absent from the current graph."""
    table = checked_csv(matrix_reference / "wTable_20260217.csv", TABLE_SHA)
    area_file = (
        author_reference
        / "source/data/banc t1 premotor"
        / "wTable_20260217_fullData_consistentColumns.csv"
    )
    area_table = checked_csv(area_file, AREA_SHA)
    ids = [row["pt_root_id"] for row in table]
    if (
        len(ids) != 4963
        or len(set(ids)) != len(ids)
        or [int(row["w_idx"]) for row in table] != list(range(4963))
        or [row["pt_root_id"] for row in area_table] != ids
    ):
        raise ValueError("Pinned matrix/area root ordering mismatch")
    areas = np.array([float(row["surf_area_um2"] or "nan") for row in area_table])
    if (
        np.isinf(areas).any()
        or (areas <= 0).any()
        or np.isfinite(areas).sum() != 4823
        or not np.isclose(np.nanmedian(areas), 4961.393216, rtol=0, atol=1e-8)
    ):
        raise ValueError("Unexpected reference area values")
    path = matrix_reference / "W_20260217.npz"
    if sha(path) != MATRIX_SHA:
        raise ValueError("Pinned matrix checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        weights = data["arr_0"].copy()
    if (
        weights.shape != (4963, 4963)
        or not np.isfinite(weights).all()
        or not np.array_equal(weights, np.rint(weights))
        or np.count_nonzero(weights) != 496041
        or np.any((weights.min(axis=1) < 0) & (weights.max(axis=1) > 0))
    ):
        raise ValueError("Unexpected pre-row signed-count matrix")
    return weights, np.asarray(ids), areas, table


def prepare(args):
    import jax

    from tools.verify_c_rate_reference import load_author

    began = time.perf_counter()
    _, ids, areas, _ = load_reference(args.matrix_reference, args.author_reference)
    scope = load_author(args.author_reference)
    keys = jax.random.split(jax.random.PRNGKey(1), 5)
    arrays = {"root_ids": ids, "areas_um2": areas}
    for profile in ("mean", "seed1"):
        params = {}
        for k, (name, mean, std) in enumerate(
            (
                ("tau", 0.02, 0.002),
                ("a", 1, 0.1),
                ("threshold", 7.5, 0.6),
                ("cap", 200, 10),
            )
        ):
            params[name] = (
                np.full((1, len(ids)), mean, np.float32)
                if profile == "mean"
                else scope["sample_trunc_normal"](keys[k], mean, std, (1024, len(ids)))[
                    :1
                ]
            )
        params["a"], params["threshold"] = scope["set_sizes"](
            areas, params["a"], params["threshold"]
        )
        for name, value in params.items():
            values = np.asarray(value[0], dtype=np.float32)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("Invalid prepared neuron parameters")
            arrays[profile + "_" + name] = values
    if time.perf_counter() - began > 60:
        raise TimeoutError("E20 parameter preparation wall limit")
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "parameters.npz", **arrays)
    write(
        args.out / "spec.json",
        {
            "test": "E20 explicit diagnostic parameter realization",
            "seed": 1,
            "draw_shape": [1024, len(ids)],
            "selected_row": 0,
            "exact_paper_run_reproduction": False,
            "jax_version": jax.__version__,
            "matrix_sha256": MATRIX_SHA,
            "table_sha256": TABLE_SHA,
            "area_sha256": AREA_SHA,
            "parameters_sha256": sha(args.out / "parameters.npz"),
            "runner_sha256": sha(Path(__file__)),
            "author_source_manifest_sha256": sha(
                args.author_reference / "source-manifest.json"
            ),
            "plan_sha256": sha(Path("GOAL_PLAN.md")),
            "wall_seconds": time.perf_counter() - began,
        },
    )
    (args.out / "runner.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {"prepared_neurons": len(ids), "wall_seconds": time.perf_counter() - began}
        ),
        flush=True,
    )


def project_inputs(current_root_ids, reference_root_ids, input_indices, values):
    """Refuse to call a projection matched if it drops any nonzero input."""
    ids = np.asarray(input_indices)
    drive = np.asarray(values)
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or drive.ndim != 2
        or drive.shape[1] != len(ids)
        or not np.isfinite(drive).all()
        or len(np.unique(ids)) != len(ids)
        or (ids < 0).any()
        or (ids >= len(current_root_ids)).any()
        or len(set(reference_root_ids)) != len(reference_root_ids)
    ):
        raise ValueError("Invalid input alignment")
    lookup = {str(root): i for i, root in enumerate(reference_root_ids)}
    targets = np.array(
        [lookup.get(str(current_root_ids[i]), -1) for i in ids], dtype=np.int64
    )
    present = targets >= 0
    if np.any(drive[:, ~present] != 0):
        raise ValueError("Reference projection drops nonzero sensory input")
    return targets[present], drive[:, present].copy(), ids[~present]


def run(args):
    from scipy.sparse import csr_matrix

    from flylab.c.graph import GraphStore
    from flylab.c.research_rate import ResearchRateNetwork

    began = time.perf_counter()
    weights, roots, areas, table = load_reference(
        args.matrix_reference, args.author_reference
    )
    preparation = json.loads((args.parameters / "spec.json").read_text(encoding="utf8"))
    if sha(args.parameters / "parameters.npz") != preparation["parameters_sha256"]:
        raise ValueError("Prepared parameter checksum mismatch")
    with np.load(args.parameters / "parameters.npz", allow_pickle=False) as data:
        params = {key: data[key].copy() for key in data.files}
    if not np.array_equal(roots, params["root_ids"]) or not np.array_equal(
        areas, params["areas_um2"], equal_nan=True
    ):
        raise ValueError("Prepared parameter root/area mismatch")
    graph = GraphStore.load(args.graph)
    if graph.hash != GRAPH_HASH:
        raise ValueError("Fixed current graph identity required")
    current_roots = np.asarray([node["id"].rsplit(":", 1)[-1] for node in graph.nodes])
    lookup = {str(root): i for i, root in enumerate(roots)}
    motor_indices = np.array(
        [
            i
            for i, row in enumerate(table)
            if row["super_class"] == "motor"
            and row["side"] == "left"
            and row["body_part_effector"] == "front_leg"
        ],
        np.int32,
    )
    pair_indices = np.array([lookup[root] for root in FAST_FETI], np.int32)
    dn_index = lookup[DN]
    if table[dn_index]["cell_type"] != "DNg100":
        raise ValueError("Diagnostic DNg100 identity mismatch")
    inputs = []
    for label in ("q1", "q1.4", "q2", "q2.3"):
        path = args.sensory_reference / ("partial_area-" + label + ".npz")
        with np.load(path, allow_pickle=False) as data:
            if data["sensory_drive"].shape[0] != 600 or np.any(
                data["sensory_drive"][200:] != 0
            ):
                raise ValueError("E18 stimulus/removal timing mismatch")
            targets, values, absent = project_inputs(
                current_roots, roots, data["sensory_indices"], data["sensory_drive"]
            )
        inputs.append((label, targets, values, absent, sha(path)))
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("parameters.npz",):
        (args.out / name).write_bytes((args.parameters / name).read_bytes())
    sources = (Path(__file__), Path("flylab/c/research_rate.py"), Path("GOAL_PLAN.md"))
    (args.out / "source").mkdir()
    for path in sources:
        (args.out / "source" / path.name).write_bytes(path.read_bytes())
    spec = {
        "test": "E20 fixed author BANC matrix under new matched sensory input",
        "exact_paper_run_reproduction": False,
        "graph_hash_for_input_ids": graph.hash,
        "simulated_neurons": len(roots),
        "source_signed_pairs": int(np.count_nonzero(weights)),
        "source_ids_absent_from_current_graph": sorted(set(roots) - set(current_roots)),
        "matrix_sha256": MATRIX_SHA,
        "table_sha256": TABLE_SHA,
        "area_sha256": AREA_SHA,
        "parameters_sha256": preparation["parameters_sha256"],
        "profiles": ["mean", "seed1"],
        "observed_motor_indices": motor_indices.tolist(),
        "fast_feti_indices": pair_indices.tolist(),
        "dn_id": DN,
        "dn_soma_side": "left",
        "dn_target_vnc_side": "right",
        "dn_index": dn_index,
        "dn_units": 400,
        "dn_start_tick": 20,
        "dn_end_tick": 1999,
        "weight_multiplier": 0.03,
        "control_dt_s": 0.001,
        "neural_dt_s": 0.0001,
        "wall_limit_s": 180,
        "neural_model_limit_s": 10,
        "physics_model_seconds": 0,
        "sensory_input_alignment": [
            {
                "name": label,
                "input_sha256": checksum,
                "matched_channels": len(targets),
                "absent_zero_input_ids": current_roots[absent].tolist(),
            }
            for label, targets, _, absent, checksum in inputs
        ],
        "sources": {str(path): sha(path) for path in sources},
        "biological_validation": False,
        "controller_adoption": False,
    }
    write(args.out / "spec.json", spec)
    matrix = csr_matrix(weights.T.astype(np.float32) * np.float32(0.03))
    del weights
    cases = []
    for profile in spec["profiles"]:
        network = ResearchRateNetwork(
            matrix,
            *[
                params[profile + "_" + name]
                for name in ("tau", "a", "threshold", "cap")
            ],
            device="cuda",
        )
        for label in ("unstimulated", "dn400", "q1", "q1.4", "q2", "q2.3"):
            network.reset()
            count = 2000 if label == "dn400" else 600
            trace = np.zeros((count + 1, len(roots)), np.float32)
            targets = np.array([], np.int64)
            input_values = np.zeros((count, 0), np.float32)
            if label.startswith("q"):
                _, targets, input_values, _, _ = next(
                    row for row in inputs if row[0] == label
                )
            elif label == "dn400":
                targets = np.array([dn_index], np.int64)
                input_values = np.zeros((count, 1), np.float32)
                input_values[20:1999, 0] = 400
            drive = np.zeros(len(roots), np.float32)
            for tick in range(count):
                drive.fill(0)
                drive[targets] = input_values[tick]
                network.advance(drive)
                trace[tick + 1] = network.readout()
                if time.perf_counter() - began > 180:
                    raise TimeoutError("E20 total execution wall limit")
            name = profile + "-" + label
            valid = bool(
                np.isfinite(trace).all()
                and network.tick == count * 10
                and not np.any(trace[0])
            )
            if label == "unstimulated":
                valid = valid and not bool(np.any(trace))
            pair = trace[:, pair_indices].astype(np.float64)
            np.savez_compressed(
                args.out / (name + ".npz"),
                rates=trace,
                root_ids=roots,
                input_indices=targets,
                input_drive=input_values,
                time_s=np.arange(count + 1) * 0.001,
                neural_tick=network.tick,
                output_mask=network.output_mask.cpu().numpy(),
            )
            case = {
                "name": name,
                "profile": profile,
                "condition": label,
                "valid": valid,
                "model_hash": network.identity,
                "device": network.device,
                "neural_model_seconds": network.tick * network.dt,
                "stimulus_tail_fast_feti": pair[151:201].mean(axis=0).tolist(),
                "final_tail_fast_feti": pair[-50:].mean(axis=0).tolist(),
                "peak_lf_motor_rate": float(trace[:, motor_indices].max()),
                "final_active_neurons_ge_2": int((trace[-1] >= 2).sum()),
                "trace_sha256": sha(args.out / (name + ".npz")),
            }
            cases.append(case)
            write(args.out / "cases.json", cases)
            print(json.dumps(case), flush=True)
        del network
        gc.collect()
    qualification = {}
    for profile in spec["profiles"]:
        rows = {c["condition"]: c for c in cases if c["profile"] == profile}
        fast, antagonist = rows["q1"]["stimulus_tail_fast_feti"]
        other, feti = rows["q2"]["stimulus_tail_fast_feti"]
        directional = (
            fast >= 2 and antagonist <= 0.2 * fast and feti >= 2 and other <= 0.2 * feti
        )
        decayed = all(
            max(rows[label]["final_tail_fast_feti"]) < 2
            for label in ("q1", "q1.4", "q2", "q2.3")
        )
        qualification[profile] = {
            "bidirectional_transfer": directional,
            "all_four_inputs_decay_below_2": decayed,
            "next_physical_test_qualified": directional
            and decayed
            and all(c["valid"] for c in rows.values()),
        }
    report = {
        "status": "COMPLETE",
        "validity": "valid" if all(c["valid"] for c in cases) else "invalid",
        "cases": cases,
        "qualification": qualification,
        "wall_seconds": time.perf_counter() - began,
        "neural_model_seconds": sum(c["neural_model_seconds"] for c in cases),
        "physics_model_seconds": 0,
        "F3_F6": "fail",
        "controller_adoption": False,
        "full_goal_completed": False,
    }
    write(args.out / "report.json", report)
    print(
        json.dumps(
            {"qualification": qualification, "wall_seconds": report["wall_seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--matrix-reference", type=Path, required=True)
    parser.add_argument("--author-reference", type=Path, required=True)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--sensory-reference", type=Path)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "run" and (
        args.parameters is None or args.sensory_reference is None
    ):
        parser.error("run requires --parameters and --sensory-reference")
    if args.out.exists():
        parser.error("out must not already exist; preserve previous evidence")
    try:
        (prepare if args.mode == "prepare" else run)(args)
    except Exception as error:
        if args.out.is_dir() and not (args.out / "report.json").exists():
            write(
                args.out / "failure.json", {"status": "INVALID", "error": repr(error)}
            )
        raise
