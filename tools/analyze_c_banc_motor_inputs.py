"""Inspect all LF tibia motor inputs in completed E27 traces without integration.

Keeps original per-cell annotations and separates input from emitted activity.
No motor is renamed slow and no input margin is converted to physical force.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.rate_observation import RateInputObserver
from tools.probe_c_banc_descending import load_parameters
from tools.probe_c_banc_reference import load_reference, sha, write

INPUT_SHA = {
    "spec.json": "00d0c417866e7ccda4260879f42da16b4e8fe512fb5cfdf1f89de2277a4dada5",
    "report.json": "d2477e23816cf51dfbe760eeb0a7053edbfb5fa8ca7f9b92e6dac298f6aad889",
    "independent-adjudication.json": "54fc305471798ea1bc85ca6f5b65dd175d6f815010a2234d624d3a6ee2221ce2",
}


def run(args):
    began = time.perf_counter()
    root, saved = args.evidence_root, args.evidence_root / "banc-dn-gate-01"
    for name, checksum in INPUT_SHA.items():
        if sha(saved / name) != checksum:
            raise ValueError("E27 input or completed independent validation changed")
    spec = json.loads((saved / "spec.json").read_text(encoding="utf8"))
    report = json.loads((saved / "report.json").read_text(encoding="utf8"))
    counts, roots, areas, table = load_reference(
        root / "banc-matrix-audit-01", root / "rate-reference"
    )
    parameters = load_parameters(root / "banc-reference-parameters-01", roots, areas)
    tibia = np.array(
        [
            i
            for i, row in enumerate(table)
            if row["super_class"] == "motor"
            and row["side"] == "left"
            and row["body_part_effector"] == "front_leg"
            and row["peripheral_target_type"]
            in (
                "tibia_flexor_muscle",
                "accessory_tibia_flexor_muscle",
                "tibia_extensor_muscle",
            )
        ]
    )
    np.testing.assert_array_equal(tibia, spec["lf_tibia_indices"])
    if len(tibia) != 19 or len(report["cases"]) != 24:
        raise ValueError("Complete E27 motor/case coverage required")
    # Author counts are pre-row; observer accepts already-scaled post/pre.
    weights = csr_matrix(counts.T * 0.03)
    observers = {
        profile: RateInputObserver(
            weights,
            *[
                parameters[profile + "_" + key]
                for key in ("tau", "a", "threshold", "cap")
            ],
            tibia,
        )
        for profile in spec["model_hashes"]
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "source").mkdir()
    sources = [
        Path(__file__),
        Path("flylab/c/rate_observation.py"),
        Path("tools/probe_c_banc_reference.py"),
        Path("tools/probe_c_banc_descending.py"),
        Path("GOAL_PLAN.md"),
    ]
    for path in sources:
        (args.out / "source" / path.name).write_bytes(path.read_bytes())
    write(
        args.out / "spec.json",
        {
            "schema": "flylab.banc-motor-input-audit.v1",
            "sources": {str(path): sha(path) for path in sources},
            "saved_inputs": INPUT_SHA,
            "model_hashes": spec["model_hashes"],
            "parameters_sha256": spec["parameters_sha256"],
            "lf_tibia_indices": tibia.tolist(),
            "neurons": 4963,
            "cases": 24,
            "alignment": "rate[t] with drive[t] for the interval starting at t",
            "summary_window": "relative input interval starts 150 through 199 ms inclusive",
            "matrix_scaling": "author float64 signed counts times 0.03, transposed to post/pre",
            "interpretation": "model inputs and rate targets; not membrane voltage, PSP, spikes or force",
            "posthoc": True,
            "neural_model_seconds": 0,
            "physics_model_seconds": 0,
            "wall_limit_s": 120,
            "controller_adoption": False,
        },
    )
    cases, summary = {}, {}
    for case in report["cases"]:
        name, profile = case["name"], case["profile"]
        source = saved / (name + ".npz")
        if sha(source) != case["trace_sha256"] or name in cases:
            raise ValueError("Changed or duplicate original trace")
        with np.load(source, allow_pickle=False) as data:
            np.testing.assert_array_equal(roots, data["root_ids"])
            np.testing.assert_array_equal(data["time_s"], 1.6 + np.arange(601) * 0.001)
            rates = data["rates"].copy()
            columns = data["input_indices"].copy()
            inputs = data["input_drive"].copy()
            mask = data["output_mask"].copy()
            if (
                rates.shape != (601, len(roots))
                or inputs.shape != (600, len(columns))
                or columns.ndim != 1
                or columns.dtype.kind not in "iu"
                or len(set(columns.tolist())) != len(columns)
                or (columns < 0).any()
                or (columns >= len(roots)).any()
                or int(data["neural_tick"]) != 22000
                or not np.all(mask == 1)
            ):
                raise ValueError("Incomplete or misaligned E27 state/input")
        drive = np.zeros((600, len(roots)))
        drive[:, columns] = inputs
        observation = observers[profile].observe(rates[:-1], drive, mask)
        tail = {key: value[150:200].mean(axis=0) for key, value in observation.items()}
        summary[name] = tail
        peak = observation["threshold_margin"][:200].max(axis=0)
        output = args.out / (name + ".npz")
        np.savez_compressed(
            output,
            **observation,
            time_s=1.6 + np.arange(600) * 0.001,
            motor_indices=tibia,
            motor_root_ids=roots[tibia],
        )
        cases[name] = {
            "profile": profile,
            "gate": case["gate"],
            "condition": case["condition"],
            "source_sha256": case["trace_sha256"],
            "observation_sha256": sha(output),
            "motors": [
                {
                    "index": int(index),
                    "root_id": roots[index],
                    "cell_type": table[index]["cell_type"],
                    "muscle": table[index]["peripheral_target_type"],
                    "physiological_slow_identity": "not_established",
                    "threshold": float(parameters[profile + "_threshold"][index]),
                    "tail": {key: float(value[j]) for key, value in tail.items()},
                    "peak_stimulus_threshold_margin": float(peak[j]),
                }
                for j, index in enumerate(tibia)
            ],
        }
        if time.perf_counter() - began > 120:
            raise TimeoutError("Posthoc observation wall limit")
    for name, row in cases.items():
        baseline = summary[row["profile"] + "-" + row["gate"] + "-dn_only"]
        for j, motor in enumerate(row["motors"]):
            motor["tail_difference_from_matched_dn_only"] = {
                key: float(value[j] - baseline[key][j])
                for key, value in summary[name].items()
            }
    result = {
        "status": "VALID_POSTHOC_OBSERVATION",
        "cases": cases,
        "neural_model_seconds": 0,
        "physics_model_seconds": 0,
        "wall_seconds": time.perf_counter() - began,
        "F3_F6": "fail",
        "controller_adoption": False,
        "qualification": "no physical or biological qualification from input margins",
    }
    write(args.out / "report.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root", type=Path, default=Path("verification/body-control-20260911")
    )
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
