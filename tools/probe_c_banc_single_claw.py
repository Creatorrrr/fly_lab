"""E26: separate existing claw input channels without changing BANC dynamics.

This bounded diagnostic tests indirect recruitment in the author reference.
It neither assigns physiological tuning nor replaces the complete BANC graph.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.probe_c_banc_descending import MODEL_HASHES, load_parameters
from tools.probe_c_banc_reference import FAST_FETI, load_reference, sha, write

SEAL_SHA = "46d8ce8519a72404133151460f736a6c7242ef05817631222728754c97a99709"
REMAP_SEAL_SHA = "148776458b787a8da7da5cb5299d6c9ca35d87bb75148b016e78844c11063a7b"
STATE_HASHES = {
    "mean": "c4f4566464a158351bc6694bf09e4a9dc0a321c8708f4b2572067758f1d4a320",
    "seed1": "db9c6e81e2cb7e1f319e6212c28c1c2cc8d509395b47b8c9a8583b033ae1db44",
}


def run(args):
    from scipy.sparse import csr_matrix

    from flylab.c.research_rate import ResearchRateNetwork

    began = time.perf_counter()
    root = args.evidence_root
    seal = root / "descending-evidence-manifest.json"
    if sha(seal) != SEAL_SHA:
        raise ValueError("Previous input evidence manifest changed")
    evidence = {
        str(Path(row["path"])): row["sha256"]
        for row in json.loads(seal.read_text(encoding="utf8"))["evidence"]
    }
    port_path = root / "feco-source-01/current-port-audit.json"
    curve_path = root / "banc-conditioned-01/mean-q1.npz"
    for path in (port_path, curve_path):
        if sha(path) != evidence[str(path.relative_to(root))]:
            raise ValueError("Sealed sensory evidence changed")
    port = json.loads(port_path.read_text(encoding="utf8"))
    claw_roots = sorted(port["external_claw_root_ids"])
    if len(set(claw_roots)) != len(claw_roots):
        raise ValueError("Invalid claw roster")
    if len(claw_roots) != 27 or "720575941615188220" in claw_roots:
        raise ValueError("Actual claw roster must exclude the orphan")
    remap_seal = root / "feco-evidence-manifest.json"
    if sha(remap_seal) != REMAP_SEAL_SHA:
        raise ValueError("Previous control evidence manifest changed")
    remap_hashes = {
        str(Path(row["path"])): row["sha256"]
        for row in json.loads(remap_seal.read_text(encoding="utf8"))["evidence"]
    }
    controls = {}
    for profile in MODEL_HASHES:
        path = root / "feco-remap-01" / (profile + "-dn_only.npz")
        if sha(path) != remap_hashes[str(path.relative_to(root))]:
            raise ValueError("Previous DN-only control changed")
        controls[profile] = {"path": str(path), "sha256": sha(path)}
    weights, roots, areas, table = load_reference(
        root / "banc-matrix-audit-01", root / "rate-reference"
    )
    parameters_path = root / "banc-reference-parameters-01"
    parameters = load_parameters(parameters_path, roots, areas)
    lookup = {value: index for index, value in enumerate(roots)}
    claw_indices = [lookup[value] for value in claw_roots]
    pair = [lookup[value] for value in FAST_FETI]
    dn = lookup["720575941500851362"]
    assert dn == 1605 and pair[0] == 1221
    with np.load(curve_path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["root_ids"], roots)
        input_roots = roots[data["input_indices"]].tolist()
        curve = data["input_drive"][:, input_roots.index("720575941478502209")].copy()
    if curve.shape != (600,) or np.any(curve[200:]) or not np.any(curve[:200]):
        raise ValueError("Saved sensory curve has incorrect duration")
    motors = [i for i, row in enumerate(table) if row["super_class"] == "motor"]
    tibia = [
        i
        for i in motors
        if table[i]["side"] == "left"
        and table[i]["body_part_effector"] == "front_leg"
        and table[i]["peripheral_target_type"]
        in (
            "tibia_flexor_muscle",
            "accessory_tibia_flexor_muscle",
            "tibia_extensor_muscle",
        )
    ]
    assert len(tibia) == 19 and set(pair) <= set(tibia)
    state_sources, states = {}, {}
    for profile, checksum in STATE_HASHES.items():
        path = root / "banc-descending-01" / (profile + "-left.npz")
        if sha(path) != checksum:
            raise ValueError("Preserved E21 state changed")
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["root_ids"], roots)
            np.testing.assert_array_equal(data["input_indices"], [dn])
            states[profile] = data["rates"][1600].copy()
        state_sources[profile] = {
            "path": str(path),
            "sha256": checksum,
            "saved_ms": 1600,
        }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "parameters.npz").write_bytes(
        (parameters_path / "parameters.npz").read_bytes()
    )
    (args.out / "source").mkdir()
    sources = (
        Path(__file__),
        Path("tools/probe_c_banc_reference.py"),
        Path("tools/probe_c_banc_descending.py"),
        Path("flylab/c/research_rate.py"),
        Path("GOAL_PLAN.md"),
    )
    for path in sources:
        (args.out / "source" / path.name).write_bytes(path.read_bytes())
    write(
        args.out / "spec.json",
        {
            "test": "E26 single claw indirect recruitment",
            "neurons": 4963,
            "sources": {str(path): sha(path) for path in sources},
            "parameters_sha256": sha(args.out / "parameters.npz"),
            "state_sources": state_sources,
            "control_sources": controls,
            "port_source": {"path": str(port_path), "sha256": sha(port_path)},
            "curve_source": {
                "path": str(curve_path),
                "sha256": sha(curve_path),
                "column_root": "720575941478502209",
            },
            "claw_roots": claw_roots,
            "claw_indices": claw_indices,
            "fast_feti_indices": pair,
            "motor_indices": motors,
            "lf_tibia_indices": tibia,
            "dn_index": dn,
            "dn_units": 400,
            "start_ms": 1600,
            "all_inputs_off_ms": 1800,
            "end_ms": 2200,
            "neural_model_limit_s": 34.8,
            "wall_limit_s": 180,
            "classification": "exploratory recruitment, no physiological tuning assignment",
            "controller_adoption": False,
        },
    )
    cases, comparison_errors = [], {}
    simulated_ms = 0
    for profile in MODEL_HASHES:
        network = ResearchRateNetwork(
            csr_matrix(weights.T.astype(np.float32) * np.float32(0.03)),
            *[
                parameters[profile + "_" + key]
                for key in ("tau", "a", "threshold", "cap")
            ],
            device="cuda",
        )
        if network.identity != MODEL_HASHES[profile]:
            raise ValueError("Original BANC reference model identity changed")
        tasks = [("dn-only", None, "control")]
        tasks += [
            ("claw-" + value, lookup[value], "single_claw") for value in claw_roots
        ]
        tasks += [("direct-fast", pair[0], "positive_control")]
        for label, stimulated, condition in tasks:
            name = profile + "-" + label
            network.reset()
            state = network.snapshot()
            state["rate"] = states[profile].copy()
            state["drive"][dn] = 400
            state["tick"] = 16000
            network.restore(state)
            rates = np.empty((601, 4963), np.float32)
            rates[0] = network.readout()
            np.testing.assert_array_equal(rates[0], states[profile])
            input_indices = np.array(
                [dn] if stimulated is None else [dn, stimulated], np.int64
            )
            inputs = np.zeros((600, len(input_indices)), np.float32)
            inputs[:200, 0] = 400
            if stimulated is not None:
                inputs[:, 1] = curve
            drive = np.zeros(4963, np.float32)
            for tick in range(600):
                drive.fill(0)
                drive[input_indices] = inputs[tick]
                network.advance(drive)
                rates[tick + 1] = network.readout()
                simulated_ms += 1
                if simulated_ms > 34800 or time.perf_counter() - began > 180:
                    raise TimeoutError("E26 model/wall limit")
            valid = bool(
                np.isfinite(rates).all()
                and (rates >= 0).all()
                and network.tick == 22000
            )
            stimulus = rates[151:201].astype(np.float64).mean(axis=0)
            final = rates[-50:].astype(np.float64).mean(axis=0)
            path = args.out / (name + ".npz")
            np.savez_compressed(
                path,
                rates=rates,
                root_ids=roots,
                input_indices=input_indices,
                input_drive=inputs,
                time_s=1.6 + np.arange(601) * 0.001,
                neural_tick=network.tick,
                output_mask=network.output_mask.cpu().numpy(),
            )
            case = {
                "name": name,
                "profile": profile,
                "condition": condition,
                "stimulated_root": str(roots[stimulated])
                if stimulated is not None
                else None,
                "valid": valid,
                "model_hash": network.identity,
                "device": network.device,
                "restored_state_max_error": 0.0,
                "neural_model_seconds": 0.6,
                "stimulus_tail_fast_feti": stimulus[pair].tolist(),
                "final_tail_fast_feti": final[pair].tolist(),
                "single_claw_candidate": bool(
                    condition == "single_claw"
                    and stimulus[pair[0]] >= 2
                    and stimulus[pair[1]] <= 0.2 * stimulus[pair[0]]
                    and max(final[pair]) < 2
                ),
                "motor_stimulus_tail": stimulus[motors].tolist(),
                "lf_tibia_final_tail": final[tibia].tolist(),
                "trace_sha256": sha(path),
            }
            if condition == "control":
                with np.load(controls[profile]["path"], allow_pickle=False) as old:
                    error = float(np.abs(old["rates"] - rates).max())
                if error > 0.001:
                    raise ValueError("Matched DN-only control changed")
                comparison_errors[profile] = error
            cases.append(case)
            write(args.out / "cases.json", cases)
            if len(cases) % 10 == 0 or condition != "single_claw":
                print(
                    json.dumps(
                        {
                            "case": name,
                            "completed": len(cases),
                            "fast_feti": case["stimulus_tail_fast_feti"],
                            "candidate": case["single_claw_candidate"],
                        }
                    ),
                    flush=True,
                )
        del network
        gc.collect()
    qualification = {}
    for profile in MODEL_HASHES:
        selected = [row for row in cases if row["profile"] == profile]
        positive = next(
            row for row in selected if row["condition"] == "positive_control"
        )
        qualification[profile] = {
            "single_claw_candidates": [
                row["stimulated_root"]
                for row in selected
                if row["single_claw_candidate"]
            ],
            "direct_fast_positive_control_ge_2": positive["stimulus_tail_fast_feti"][0]
            >= 2,
            "physiological_or_physical_control_qualified": False,
        }
    report = {
        "status": "COMPLETE",
        "validity": "valid" if all(row["valid"] for row in cases) else "invalid",
        "cases": cases,
        "qualification": qualification,
        "dn_only_max_errors": comparison_errors,
        "neural_model_seconds": simulated_ms * 0.001,
        "physics_model_seconds": 0,
        "wall_seconds": time.perf_counter() - began,
        "F3_F6": "fail",
        "controller_adoption": False,
        "full_goal_completed": False,
    }
    write(args.out / "report.json", report)
    print(
        json.dumps({key: value for key, value in report.items() if key != "cases"}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root", type=Path, default=Path("verification/body-control-20260911")
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("out must not exist; preserve earlier evidence")
    try:
        run(args)
    except Exception as error:
        if args.out.is_dir() and not (args.out / "report.json").exists():
            write(
                args.out / "failure.json", {"status": "INVALID", "error": repr(error)}
            )
        raise
