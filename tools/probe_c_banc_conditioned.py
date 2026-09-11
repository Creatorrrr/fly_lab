"""E22/E24: preserved left-target DNg100 state and explicit sensory inputs.

The optional E24 candidate changes only declared FeCO homology input cells.
Omitting it retains the original E22 input task.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.probe_c_banc_descending import (
    MODEL_HASHES,
    PARAMETERS_SHA,
    load_parameters,
    motor_groups,
)
from tools.probe_c_banc_reference import (
    FAST_FETI,
    GRAPH_HASH,
    load_reference,
    project_inputs,
    sha,
    write,
)


def run(args):
    from scipy.sparse import csr_matrix

    from flylab.c.graph import GraphStore
    from flylab.c.research_rate import ResearchRateNetwork

    began = time.perf_counter()
    candidate_path = getattr(args, "claw_candidate", None)
    candidate = None
    experiment = "E24" if candidate_path is not None else "E22"
    if candidate_path is not None:
        from tools.probe_c_feco_identity import remap_inputs

        candidate = json.loads(candidate_path.read_text(encoding="utf8"))
        if candidate["graph_hash"] != GRAPH_HASH:
            raise ValueError("FeCO candidate graph mismatch")
        for source in candidate["sources"].values():
            if sha(Path(source["path"])) != source["sha256"]:
                raise ValueError("FeCO candidate source changed")
    weights, roots, areas, table = load_reference(
        args.matrix_reference, args.author_reference
    )
    params = load_parameters(args.parameters, roots, areas)
    previous = json.loads((args.previous / "report.json").read_text(encoding="utf8"))
    if previous["validity"] != "valid":
        raise ValueError("Valid E21 preserved state required")
    states = {}
    state_sources = {}
    for profile in MODEL_HASHES:
        name = profile + "-left"
        case = next(c for c in previous["cases"] if c["name"] == name)
        path = args.previous / (name + ".npz")
        if (
            sha(path) != case["trace_sha256"]
            or case["model_hash"] != MODEL_HASHES[profile]
        ):
            raise ValueError("E21 source checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["root_ids"], roots)
            np.testing.assert_array_equal(data["input_indices"], [1605])
            np.testing.assert_array_equal(data["input_drive"][1599], [400])
            states[profile] = data["rates"][1600].copy()
        state_sources[profile] = {
            "path": str(path),
            "sha256": sha(path),
            "saved_ms": 1600,
        }
    graph = GraphStore.load(args.graph)
    if graph.hash != GRAPH_HASH:
        raise ValueError("Fixed input graph required")
    current_roots = [node["id"].rsplit(":", 1)[-1] for node in graph.nodes]
    lookup = {root: i for i, root in enumerate(roots)}
    dn = lookup["720575941500851362"]
    if dn != 1605 or table[dn]["cell_type"] != "DNg100":
        raise ValueError("Fixed left-target DN required")
    pair = np.array([lookup[root] for root in FAST_FETI], np.int64)
    inputs = {"dn_only": (np.array([], np.int64), np.zeros((600, 0), np.float32))}
    input_sources = {}
    for label in ("q1", "q1.4", "q2", "q2.3"):
        path = args.sensory_reference / ("partial_area-" + label + ".npz")
        with np.load(path, allow_pickle=False) as data:
            targets, values, absent = project_inputs(
                current_roots, roots, data["sensory_indices"], data["sensory_drive"]
            )
        original_values = values.copy()
        if candidate is not None:
            values = remap_inputs(roots, targets, values, candidate)
        if values.shape[0] != 600 or np.any(values[200:]) or dn in targets:
            raise ValueError("Saved sensory/DN alignment or timing mismatch")
        inputs[label] = (targets, values)
        input_sources[label] = {
            "path": str(path),
            "sha256": sha(path),
            "matched_channels": len(targets),
            "absent_zero_input_ids": [current_roots[i] for i in absent],
        }
        if candidate is not None:
            input_sources[label]["changed_root_ids"] = roots[
                targets[np.any(values != original_values, axis=0)]
            ].tolist()
    groups = motor_groups(table)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "parameters.npz").write_bytes(
        (args.parameters / "parameters.npz").read_bytes()
    )
    sources = (
        Path(__file__),
        Path("tools/probe_c_banc_reference.py"),
        Path("tools/probe_c_banc_descending.py"),
        Path("flylab/c/research_rate.py"),
        Path("GOAL_PLAN.md"),
    )
    if candidate_path is not None:
        sources += (Path("tools/probe_c_feco_identity.py"), candidate_path)
    (args.out / "source").mkdir()
    for path in sources:
        (args.out / "source" / path.name).write_bytes(path.read_bytes())
    spec = {
        "test": experiment + " left-target DNg100 conditioned sensory transfer",
        "simulated_neurons": len(roots),
        "source_signed_pairs": int(np.count_nonzero(weights)),
        "state_sources": state_sources,
        "sensory_sources": input_sources,
        "parameters_sha256": PARAMETERS_SHA,
        "model_hashes": MODEL_HASHES,
        "graph_hash_for_sensory_ids": graph.hash,
        "dn_root": roots[dn],
        "dn_index": dn,
        "dn_soma_side": "right",
        "dn_target_vnc_side": "left",
        "dn_units": 400,
        "start_ms": 1600,
        "all_inputs_off_ms": 1800,
        "end_ms": 2200,
        "motor_groups": {key: ids.tolist() for key, ids in groups.items()},
        "fast_feti_indices": pair.tolist(),
        "neural_model_limit_s": 6,
        "wall_limit_s": 120,
        "sources": {str(path): sha(path) for path in sources},
        "exact_paper_run_reproduction": False,
        "controller_adoption": False,
    }
    if candidate_path is not None:
        spec["claw_candidate"] = {
            "path": str(candidate_path),
            "sha256": sha(candidate_path),
            "evidence_status": candidate["evidence_status"],
            "biological_validation": False,
            "assignments": candidate["assignments"],
        }
    write(args.out / "spec.json", spec)
    matrix = csr_matrix(weights.T.astype(np.float32) * np.float32(0.03))
    del weights
    cases = []
    for profile, expected in MODEL_HASHES.items():
        network = ResearchRateNetwork(
            matrix,
            *[params[profile + "_" + key] for key in ("tau", "a", "threshold", "cap")],
            device="cuda",
        )
        if network.identity != expected:
            raise ValueError("Prepared network changed from E21")
        for label, (sensory_indices, sensory_values) in inputs.items():
            state = network.snapshot()
            state["rate"] = states[profile]
            state["tick"] = 16000
            state["drive"].fill(0)
            state["drive"][dn] = 400
            state["output_mask"].fill(1)
            network.restore(state)
            rates = np.zeros((601, len(roots)), np.float32)
            rates[0] = network.readout()
            np.testing.assert_array_equal(rates[0], states[profile])
            targets = np.concatenate(([dn], sensory_indices)).astype(np.int64)
            values = np.zeros((600, len(targets)), np.float32)
            values[:200, 0] = 400
            values[:, 1:] = sensory_values
            drive = np.zeros(len(roots), np.float32)
            for tick in range(600):
                drive.fill(0)
                drive[targets] = values[tick]
                network.advance(drive)
                rates[tick + 1] = network.readout()
                if time.perf_counter() - began > 120:
                    raise TimeoutError(experiment + " cumulative wall limit")
            valid = bool(
                np.isfinite(rates).all()
                and (rates >= 0).all()
                and network.tick == 22000
            )
            name = profile + "-" + label
            path = args.out / (name + ".npz")
            np.savez_compressed(
                path,
                rates=rates,
                root_ids=roots,
                input_indices=targets,
                input_drive=values,
                time_s=1.6 + np.arange(601) * 0.001,
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
                "neural_model_seconds": (network.tick - 16000) * network.dt,
                "restored_state_max_error": float(
                    np.abs(rates[0] - states[profile]).max()
                ),
                "stimulus_tail_fast_feti": rates[151:201, pair]
                .astype(np.float64)
                .mean(axis=0)
                .tolist(),
                "final_tail_fast_feti": rates[-50:, pair]
                .astype(np.float64)
                .mean(axis=0)
                .tolist(),
                "stimulus_peak": {
                    key: float(rates[1:201, ids].max()) for key, ids in groups.items()
                },
                "final_active_neurons_ge_2": int((rates[-1] >= 2).sum()),
                "trace_sha256": sha(path),
            }
            cases.append(case)
            write(args.out / "cases.json", cases)
            print(json.dumps(case), flush=True)
        del network
        gc.collect()
    qualification = {}
    for profile in MODEL_HASHES:
        rows = {c["condition"]: c for c in cases if c["profile"] == profile}
        fast, antagonist = rows["q1"]["stimulus_tail_fast_feti"]
        other, feti = rows["q2"]["stimulus_tail_fast_feti"]
        directional = (
            fast >= 2 and antagonist <= 0.2 * fast and feti >= 2 and other <= 0.2 * feti
        )
        decayed = all(
            max(rows[q]["final_tail_fast_feti"]) < 2
            for q in ("q1", "q1.4", "q2", "q2.3")
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
        "controller_adoption": False,
        "F3_F6": "fail",
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
    for key in (
        "matrix-reference",
        "author-reference",
        "parameters",
        "previous",
        "sensory-reference",
        "out",
    ):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument(
        "--claw-candidate",
        type=Path,
        help="Explicit E24 homology candidate; default retains original E22 inputs",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    args = parser.parse_args()
    if args.out.exists():
        parser.error("out must not already exist; preserve prior evidence")
    try:
        run(args)
    except Exception as error:
        if args.out.is_dir() and not (args.out / "report.json").exists():
            write(
                args.out / "failure.json", {"status": "INVALID", "error": repr(error)}
            )
        raise
