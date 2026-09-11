"""E27: compare held/removed external DN drive from identical saved BANC states.

Neural outputs and recurrent connections remain intact. This rate-reference
diagnostic does not qualify full-brain or physical body control.
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

SEALS = {
    "feco-evidence-manifest.json": "148776458b787a8da7da5cb5299d6c9ca35d87bb75148b016e78844c11063a7b",
    "fanc-feedback-evidence-manifest.json": "25fb08791f922f239a8b861c9c7b8e8c11e767343fc242584cef93a47ec639ac",
}
LABELS = ("dn_only", "q1", "q1.4", "q2", "q2.3", "direct_fast")
INHIBITORY_ROOTS = ("720575941624450506", "720575941514435793", "720575941534733800")


def remove_external_dn(input_indices, inputs, dn_index):
    """Return a copy with the unique external DN channel removed."""
    indices = np.asarray(input_indices)
    values = np.asarray(inputs)
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or len(set(indices.tolist())) != len(indices)
        or values.ndim != 2
        or values.shape[1] != len(indices)
        or not np.isfinite(values).all()
        or np.count_nonzero(indices == dn_index) != 1
    ):
        raise ValueError("Unique aligned finite external DN input required")
    result = values.copy()
    result[:, int(np.flatnonzero(indices == dn_index)[0])] = 0
    return result


def run(args):
    from scipy.sparse import csr_matrix

    from flylab.c.research_rate import ResearchRateNetwork

    began = time.perf_counter()
    root = args.evidence_root
    sealed = {}
    for name, checksum in SEALS.items():
        path = root / name
        if sha(path) != checksum:
            raise ValueError("Previous evidence manifest changed")
        for row in json.loads(path.read_text(encoding="utf8"))["evidence"]:
            relative = str(Path(row["path"]))
            if relative in sealed and sealed[relative] != row["sha256"]:
                raise ValueError("Conflicting evidence hashes")
            sealed[relative] = row["sha256"]
    weights, roots, areas, table = load_reference(
        root / "banc-matrix-audit-01", root / "rate-reference"
    )
    parameter_dir = root / "banc-reference-parameters-01"
    parameters = load_parameters(parameter_dir, roots, areas)
    lookup = {value: index for index, value in enumerate(roots)}
    dn = lookup["720575941500851362"]
    pair = [lookup[value] for value in FAST_FETI]
    inhibitory = [lookup[value] for value in INHIBITORY_ROOTS]
    if (
        dn != 1605
        or pair[0] != 1221
        or any(weights[i, pair[0]] >= 0 for i in inhibitory)
    ):
        raise ValueError("Fixed target identity or inhibitory source changed")
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
    if len(tibia) != 19:
        raise ValueError("LF tibia motor membership changed")
    inputs, source_records, states = {}, {}, {}
    for profile in MODEL_HASHES:
        for label in LABELS:
            path = (
                (root / "banc-single-claw-01" / (profile + "-direct-fast.npz"))
                if label == "direct_fast"
                else (root / "feco-remap-01" / (profile + "-" + label + ".npz"))
            )
            if sha(path) != sealed[str(path.relative_to(root))]:
                raise ValueError("Saved matched input/trajectory changed")
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["root_ids"], roots)
                np.testing.assert_array_equal(
                    data["time_s"], 1.6 + np.arange(601) * 0.001
                )
                indices, values, rates = (
                    data["input_indices"].copy(),
                    data["input_drive"].copy(),
                    data["rates"].copy(),
                )
                if (
                    indices[0] != dn
                    or values.shape != (600, len(indices))
                    or rates.shape != (601, 4963)
                ):
                    raise ValueError("Source input/state alignment changed")
                expected = np.zeros(600, np.float32)
                expected[:200] = 400
                np.testing.assert_array_equal(values[:, 0], expected)
                if (
                    np.any(values[200:])
                    or not np.all(data["output_mask"] == 1)
                    or int(data["neural_tick"]) != 22000
                ):
                    raise ValueError("Saved input removal or state mask changed")
                if profile in states:
                    np.testing.assert_array_equal(rates[0], states[profile])
                states[profile] = rates[0].copy()
            inputs[(profile, label)] = (indices, values, rates)
            source_records[profile + "-" + label] = {
                "path": str(path),
                "sha256": sha(path),
            }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "parameters.npz").write_bytes(
        (parameter_dir / "parameters.npz").read_bytes()
    )
    sources = (
        Path(__file__),
        Path("tools/probe_c_banc_reference.py"),
        Path("tools/probe_c_banc_descending.py"),
        Path("flylab/c/research_rate.py"),
        Path("GOAL_PLAN.md"),
    )
    (args.out / "source").mkdir()
    for path in sources:
        (args.out / "source" / path.name).write_bytes(path.read_bytes())
    write(
        args.out / "spec.json",
        {
            "test": "E27 matched external descending input removal",
            "neurons": 4963,
            "sources": {str(path): sha(path) for path in sources},
            "source_evidence": source_records,
            "parameters_sha256": sha(args.out / "parameters.npz"),
            "model_hashes": MODEL_HASHES,
            "dn_index": dn,
            "dn_units": 400,
            "fast_feti_indices": pair,
            "motor_indices": motors,
            "lf_tibia_indices": tibia,
            "inhibitory_indices": inhibitory,
            "start_ms": 1600,
            "all_inputs_off_ms": 1800,
            "end_ms": 2200,
            "neural_model_limit_s": 14.4,
            "wall_limit_s": 180,
            "manipulation": "external DN channel only; retain neural state, output and recurrent weights",
            "controller_adoption": False,
        },
    )
    cases, hold_errors = [], {}
    simulated_ms = 0
    fast_column = weights[:, pair[0]] * 0.03
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
            raise ValueError("BANC reference dynamics changed")
        for gate in ("hold", "remove"):
            for label in LABELS:
                name = profile + "-" + gate + "-" + label
                indices, original_input, reference = inputs[(profile, label)]
                values = (
                    original_input.copy()
                    if gate == "hold"
                    else remove_external_dn(indices, original_input, dn)
                )
                network.reset()
                state = network.snapshot()
                state["rate"] = states[profile].copy()
                state["drive"][dn] = 400
                state["tick"] = 16000
                network.restore(state)
                rates = np.empty((601, 4963), np.float32)
                rates[0] = network.readout()
                np.testing.assert_array_equal(rates[0], states[profile])
                drive = np.zeros(4963, np.float32)
                for tick in range(600):
                    drive.fill(0)
                    drive[indices] = values[tick]
                    network.advance(drive)
                    rates[tick + 1] = network.readout()
                    simulated_ms += 1
                    if simulated_ms > 14400 or time.perf_counter() - began > 180:
                        raise TimeoutError("E27 model/wall limit")
                if gate == "hold":
                    error = float(np.abs(rates - reference).max())
                    if error > 0.001:
                        raise ValueError(
                            "Held DN condition differs from preserved reference"
                        )
                    hold_errors[name] = error
                stimulus = rates[151:201].astype(np.float64).mean(axis=0)
                final = rates[-50:].astype(np.float64).mean(axis=0)
                aligned_tail = rates[150:200].astype(np.float64).mean(axis=0)
                output = args.out / (name + ".npz")
                np.savez_compressed(
                    output,
                    rates=rates,
                    root_ids=roots,
                    input_indices=indices,
                    input_drive=values,
                    time_s=1.6 + np.arange(601) * 0.001,
                    neural_tick=network.tick,
                    output_mask=network.output_mask.cpu().numpy(),
                )
                case = {
                    "name": name,
                    "profile": profile,
                    "gate": gate,
                    "condition": label,
                    "valid": bool(
                        np.isfinite(rates).all()
                        and (rates >= 0).all()
                        and network.tick == 22000
                    ),
                    "device": network.device,
                    "model_hash": network.identity,
                    "restored_state_max_error": 0.0,
                    "stimulus_tail_fast_feti": stimulus[pair].tolist(),
                    "final_tail_fast_feti": final[pair].tolist(),
                    "motor_stimulus_tail": stimulus[motors].tolist(),
                    "lf_tibia_final_tail": final[tibia].tolist(),
                    "inhibitory_stimulus_tail": stimulus[inhibitory].tolist(),
                    "inhibitory_contributions_to_fast": (
                        aligned_tail[inhibitory] * fast_column[inhibitory]
                    ).tolist(),
                    "fast_synaptic_input_tail": {
                        "positive": float(
                            aligned_tail[fast_column > 0] @ fast_column[fast_column > 0]
                        ),
                        "negative": float(
                            aligned_tail[fast_column < 0] @ fast_column[fast_column < 0]
                        ),
                    },
                    "dn_stimulus_tail": float(stimulus[dn]),
                    "neural_model_seconds": 0.6,
                    "trace_sha256": sha(output),
                }
                cases.append(case)
                write(args.out / "cases.json", cases)
                print(
                    json.dumps(
                        {
                            "case": name,
                            "fast_feti": case["stimulus_tail_fast_feti"],
                            "dn_rate": case["dn_stimulus_tail"],
                        }
                    ),
                    flush=True,
                )
        del network
        gc.collect()
    qualification = {}
    for profile in MODEL_HASHES:
        selected = {
            (row["gate"], row["condition"]): row
            for row in cases
            if row["profile"] == profile
        }
        direct_hold = selected[("hold", "direct_fast")]["stimulus_tail_fast_feti"][0]
        direct_removed = selected[("remove", "direct_fast")]["stimulus_tail_fast_feti"][
            0
        ]
        fast, antagonist = selected[("remove", "q1")]["stimulus_tail_fast_feti"]
        other, feti = selected[("remove", "q2")]["stimulus_tail_fast_feti"]
        directional = (
            fast >= 2 and antagonist <= 0.2 * fast and feti >= 2 and other <= 0.2 * feti
        )
        decayed = all(
            max(selected[("remove", label)]["lf_tibia_final_tail"]) < 2
            for label in LABELS[1:5]
        )
        qualification[profile] = {
            "direct_fast_remove_minus_hold": direct_removed - direct_hold,
            "descending_suppresses_direct_fast": direct_removed >= 2
            and direct_removed - direct_hold >= 2,
            "removed_dn_bidirectional_sensory_transfer": directional,
            "all_four_inputs_tibia_decay_below_2": decayed,
            "conditional_neural_candidate": directional and decayed,
            "whole_BANC_or_physical_qualified": False,
        }
    report = {
        "status": "COMPLETE",
        "validity": "valid" if all(row["valid"] for row in cases) else "invalid",
        "cases": cases,
        "qualification": qualification,
        "matched_hold_max_errors": hold_errors,
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("out must not exist; preserve prior evidence")
    try:
        run(args)
    except Exception as error:
        if args.out.is_dir() and not (args.out / "report.json").exists():
            write(
                args.out / "failure.json", {"status": "INVALID", "error": repr(error)}
            )
        raise
