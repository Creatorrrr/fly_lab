"""Decompose saved E28 motor inputs on CPU, without advancing any model.

The 600 ms observation is a terminal vector-field diagnostic with external
input held at its declared post-removal value, not an observed future interval.
All values retain the research rate model's units and individual neuron IDs.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.rate_observation import RateInputObserver

SAMPLE_MS = (199, 200, 201, 400, 600)
FOCUS_ROOTS = (
    "720575941481179066",
    "720575941639281525",
    "720575941479472258",
    "720575941627155238",
)
EXPECTED_GRAPH = "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"
EXPECTED_MODEL = "fda2ac04f12dba959980cb49a9243660c5786e605202f567749066d541159164"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf8",
    )


def aligned_states(raw, neuron_count, sample_ms=SAMPLE_MS):
    """Use interval starts, not the motor trace's interval-end row indices."""
    columns = np.asarray(raw["input_indices"])
    inputs = np.asarray(raw["input_drive"])
    mask = np.asarray(raw["output_mask"])
    if (
        columns.ndim != 1
        or columns.dtype.kind not in "iu"
        or len(set(columns.tolist())) != len(columns)
        or np.any(columns < 0)
        or np.any(columns >= neuron_count)
        or inputs.shape != (600, len(columns))
        or mask.shape != (neuron_count,)
        or not np.isin(mask, (0, 1)).all()
        or not np.isfinite(inputs).all()
        or np.any(inputs[200:])
        or not sample_ms
        or any(not isinstance(t, int) or t < 0 or t > 600 for t in sample_ms)
        or len(set(sample_ms)) != len(sample_ms)
    ):
        raise ValueError(
            "Complete E28 timing, nonduplicated input IDs and mask required"
        )
    rates = np.stack([raw[f"rate_at_{t}ms"] for t in sample_ms])
    if (
        rates.shape != (len(sample_ms), neuron_count)
        or not np.isfinite(rates).all()
        or np.any(rates < 0)
    ):
        raise ValueError("Complete finite nonnegative saved states required")
    drive = np.zeros((len(sample_ms), neuron_count), np.float64)
    for row, t in enumerate(sample_ms):
        # At t=600 the predeclared zero-input continuation defines only the RHS.
        if t < 600:
            drive[row, columns] = inputs[t]
    return rates, drive, mask


def edge_observations(weights, parameters, indices, rates, drives, mask):
    """Independent incoming-edge sums; also preserve each source contribution."""
    tau, gain, threshold, cap = [np.asarray(p, np.float64) for p in parameters]
    emitted = np.asarray(rates, np.float64) * mask
    result = {
        key: []
        for key in (
            "rate",
            "excitatory_input",
            "inhibitory_input",
            "external_input",
            "threshold_margin",
            "instantaneous_target_rate",
            "rate_derivative",
        )
    }
    edges = {}
    for index in indices:
        start, end = weights.indptr[index : index + 2]
        pre = weights.indices[start:end]
        values = weights.data[start:end].astype(np.float64)
        contribution = emitted[:, pre] * values
        positive = contribution[:, values > 0].sum(axis=1)
        negative = contribution[:, values < 0].sum(axis=1)
        external = drives[:, index]
        margin = positive + negative + external - threshold[index]
        target = cap[index] * np.maximum(np.tanh(gain[index] * margin / cap[index]), 0)
        rate = rates[:, index].astype(np.float64)
        actual = (
            rate,
            positive,
            negative,
            external,
            margin,
            target,
            (target - rate) / tau[index],
        )
        for key, value in zip(result, actual):
            result[key].append(value)
        edges[int(index)] = (pre.copy(), values, contribution)
    return {key: np.stack(value, axis=1) for key, value in result.items()}, edges


def summarize_edges(graph, rows, values, contribution, sample_ms=SAMPLE_MS):
    """Nonoverlapping region groups and complete incoming-edge provenance."""
    source_rows = []
    for i, index in enumerate(rows):
        node = graph.nodes[index]
        source_rows.append(
            {
                "index": int(index),
                "root_id": node["root_id"],
                "cell_type": node.get("cell_type"),
                "super_class": node.get("super_class"),
                "nt_type": node.get("nt_type"),
                "regions": node.get("regions", []),
                "weight": float(values[i]),
            }
        )
    groups = {}
    for i, row in enumerate(source_rows):
        key = "+".join(sorted(row["regions"])) or "unannotated"
        groups.setdefault(key, []).append(i)
    by_region = {
        key: {
            "excitatory_input": contribution[:, [i for i in ids if values[i] > 0]]
            .sum(axis=1)
            .tolist(),
            "inhibitory_input": contribution[:, [i for i in ids if values[i] < 0]]
            .sum(axis=1)
            .tolist(),
        }
        for key, ids in groups.items()
    }
    top = {}
    for j, stamp in enumerate(sample_ms):
        top[str(stamp)] = {}
        for label, eligible, sign in (
            ("excitatory", np.flatnonzero(contribution[j] > 0), -1),
            ("inhibitory", np.flatnonzero(contribution[j] < 0), 1),
        ):
            selected = eligible[
                np.argsort(sign * contribution[j, eligible], kind="stable")[:10]
            ]
            top[str(stamp)][label] = [
                {**source_rows[i], "contribution": float(contribution[j, i])}
                for i in selected
            ]
    return {"sources": source_rows, "by_region": by_region, "top_sources": top}


def prior_comparison(root):
    result = {}
    for folder in ("rate-decay-02", "rate-regional-decay-01"):
        path = root / folder / "report.json"
        report = json.loads(path.read_text(encoding="utf8"))
        cases = []
        for case in report["cases"]:
            # Preserve verdicts already attached to each source experiment.
            row = {k: v for k, v in case.items() if not isinstance(v, (list, dict))}
            row["focus_motors"] = [
                motor
                for motor in case.get("tibia", [])
                if motor["id"].split(":")[-1] in FOCUS_ROOTS
            ]
            cases.append(row)
        result[folder] = {
            "source": str(path),
            "sha256": sha(path),
            "cases": cases,
            "comparison_limit": "Historical uniform profile; not a matched counterfactual of E28 morphology_area",
        }
    alignment_path = root / "motor-unit-source/alignment.json"
    inventory_path = root / "motor-unit-source/current-morphology-inventory.json"
    alignment = json.loads(alignment_path.read_text(encoding="utf8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf8"))
    result["accessory_identity"] = {
        "sources": {str(p): sha(p) for p in (alignment_path, inventory_path)},
        "original_xml_contains_accessory_muscle": alignment[
            "accessory_tibia_flexor_in_original_xml"
        ],
        "fixture_rows": [
            r for r in alignment["rows"] if r["id"].split(":")[-1] in FOCUS_ROOTS[2:]
        ],
        "current_annotation_rows": [
            r
            for r in inventory["lf_tibia_source_rows"]
            if r["root_888"] in FOCUS_ROOTS[2:]
        ],
        "physiological_slow_identity": "not_established",
        "force_path_adoption": False,
    }
    return result


def run(args):
    from flylab.c.graph import GraphStore
    from flylab.c.research_annotations import rate_weights

    began, cpu_began = time.perf_counter(), time.process_time()
    args.out.mkdir(parents=True, exist_ok=False)
    write(
        args.out / "analysis-contract.json",
        {
            "posthoc": True,
            "input": str(args.saved),
            "sample_ms": SAMPLE_MS,
            "question": "Does late Fast rate have ongoing recurrent excitation above threshold after the input is removed?",
            "competing_explanation": "Only passive decay of an earlier nonzero motor rate, or persistent external input",
            "predictions": "Nonzero late recurrent margin/target supports active circuit supply; zero margin/target with decreasing rate supports passive rate decay at that timestamp",
            "limits": "Descriptive input decomposition cannot localize a causal recurrent loop or prove biological correctness",
            "validity": "Original identities and trace hashes, state/input alignment, and independent incoming sums within 1e-8",
            "neural_model_seconds": 0,
            "physics_model_seconds": 0,
            "wall_limit_s": 120,
            "cpu_limit_s": 120,
            "controller_adoption": False,
        },
    )
    spec = json.loads((args.saved / "spec.json").read_text(encoding="utf8"))
    report = json.loads((args.saved / "report.json").read_text(encoding="utf8"))
    validation = json.loads(
        (args.saved / "independent-adjudication.json").read_text(encoding="utf8")
    )
    if (
        spec["model_hashes"] != {"morphology_area": EXPECTED_MODEL}
        or report["status"] != "VALID_COMPLETE"
        or len(report["cases"]) != 9
        or spec["input_timing_ms"] != {"on": 0, "off": 200, "end": 600}
        or sha(args.saved / "parameters.npz") != spec["parameters_sha256"]
    ):
        raise ValueError("Original valid morphology E28 evidence required")
    graph = GraphStore.load(spec["graph_path"])
    if (
        graph.hash != spec["graph_hash"]
        or graph.hash != EXPECTED_GRAPH
        or graph.n != 158706
    ):
        raise ValueError("Original complete BANC graph required")
    weights, audit = rate_weights(graph)
    if audit is not None or len(graph.indices) != spec["graph_pairs"]:
        raise ValueError("Unmodified complete signed-count weights required")
    roots = np.asarray([r["root_id"] for r in graph.nodes])
    with np.load(args.saved / "parameters.npz", allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw["root_ids"], roots)
        parameters = [
            raw["morphology_area_" + key].copy()
            for key in ("tau", "a", "threshold", "cap")
        ]
    observed = np.asarray([r["index"] for r in spec["observed"]])
    motors = [spec["observed"][i] for i in spec["tibia_columns"]]
    indices = np.asarray([r["index"] for r in motors])
    if len(motors) != 19 or len(observed) != 69:
        raise ValueError("All LF tibia and motor identities required")
    for row in motors:
        node = graph.nodes[row["index"]]
        if (
            node["id"],
            node["cell_type"],
            node["source_annotations"]["peripheral_target_type"],
        ) != (row["id"], row["cell_type"], row["target"]):
            raise ValueError("Motor annotation drift")
    observer = RateInputObserver(weights, *parameters, indices)
    cases, max_error, compared = {}, 0.0, 0
    for case in report["cases"]:
        name = case["name"]
        source = args.saved / (name + ".npz")
        if sha(source) != case["trace_sha256"] or not case["valid"] or name in cases:
            raise ValueError("Changed, invalid or duplicate original trace")
        with np.load(source, allow_pickle=False) as raw:
            rates, drive, mask = aligned_states(raw, graph.n)
            np.testing.assert_array_equal(raw["motor_indices"], observed)
            np.testing.assert_array_equal(
                raw["motor_rate_sample_neural_ticks"], np.arange(1, 601) * 10
            )
            if int(raw["neural_tick"]) != 6000:
                raise ValueError("Incomplete saved neural clock")
            for j, stamp in enumerate(SAMPLE_MS):
                np.testing.assert_array_equal(
                    rates[j, observed], raw["motor_rates"][stamp - 1]
                )
        expected_mask = np.ones(graph.n)
        if case["condition"] == "cut":
            expected_mask[spec["claw_cut_indices"]] = 0
        np.testing.assert_array_equal(mask, expected_mask)
        observation = observer.observe(rates, drive, mask)
        reference, edges = edge_observations(
            weights, parameters, indices, rates, drive, mask
        )
        error = max(
            float(np.max(np.abs(reference[key] - value)))
            for key, value in observation.items()
        )
        max_error = max(max_error, error)
        compared += sum(value.size for value in observation.values())
        if error > 1e-8:
            raise ValueError("Independent incoming-edge decomposition disagrees")
        payload = dict(observation)
        payload.update(
            sample_ms=np.asarray(SAMPLE_MS),
            motor_indices=indices,
            motor_root_ids=roots[indices],
        )
        focus = {}
        for j, motor in enumerate(motors):
            root = motor["id"].split(":")[-1]
            if root not in FOCUS_ROOTS:
                continue
            pre, values, contribution = edges[motor["index"]]
            payload[root + "_presynaptic_indices"] = pre
            payload[root + "_signed_weights"] = values
            payload[root + "_contribution"] = contribution
            focus[root] = {
                **motor,
                "threshold": float(parameters[2][motor["index"]]),
                "tau_s": float(parameters[0][motor["index"]]),
                "samples": [
                    {
                        "time_ms": stamp,
                        **{
                            key: float(value[k, j])
                            for key, value in observation.items()
                        },
                    }
                    for k, stamp in enumerate(SAMPLE_MS)
                ],
                **summarize_edges(graph, pre, values, contribution),
            }
        target = args.out / (name + ".npz")
        np.savez_compressed(target, **payload)
        cases[name] = {
            "source_sha256": case["trace_sha256"],
            "observation_sha256": sha(target),
            "condition": case["condition"],
            "pose": case["pose"],
            "focus_motors": focus,
        }
        if time.perf_counter() - began > 120 or time.process_time() - cpu_began > 120:
            raise TimeoutError("Posthoc analysis unit exceeded its 120 s bound")
    source_paths = [
        Path(__file__),
        Path("flylab/c/rate_observation.py"),
        Path("flylab/c/research_annotations.py"),
    ]
    (args.out / "source").mkdir()
    for path in source_paths:
        (args.out / "source" / path.name).write_bytes(path.read_bytes())
    result = {
        "schema": "flylab.late-motor-activity-analysis.v1",
        "status": "VALID_POSTHOC_OBSERVATION",
        "sources": {str(p): sha(p) for p in source_paths},
        "saved_evidence": {
            name: sha(args.saved / name)
            for name in (
                "spec.json",
                "report.json",
                "independent-adjudication.json",
                "parameters.npz",
            )
        },
        "original_independent_validation": validation,
        "graph_hash": graph.hash,
        "model_hash": EXPECTED_MODEL,
        "sample_ms": SAMPLE_MS,
        "alignment": "state at t, input for [t,t+1 ms); 600 ms is terminal zero-input RHS only",
        "units": "research model rate and input units, not physiological Hz, voltage or force",
        "cases": cases,
        "prior_comparison": prior_comparison(args.evidence_root),
        "verification": {
            "independent_max_error": max_error,
            "compared_values": compared,
            "tolerance": 1e-8,
        },
        "neural_model_seconds": 0,
        "physics_model_seconds": 0,
        "wall_seconds": time.perf_counter() - began,
        "cpu_seconds": time.process_time() - cpu_began,
        "controller_adoption": False,
        "physical_qualified": False,
        "F3_F6": "fail",
    }
    write(args.out / "report.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "verification",
                    "wall_seconds",
                    "cpu_seconds",
                    "neural_model_seconds",
                    "physics_model_seconds",
                    "controller_adoption",
                )
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--saved",
        type=Path,
        default=Path("verification/body-control-20260911/whole-motor-pool-02"),
    )
    parser.add_argument(
        "--evidence-root", type=Path, default=Path("verification/body-control-20260911")
    )
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
