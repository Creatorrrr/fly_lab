"""M3-E04: locate teacher-state information before the motor nonlinearity.

Offline linear decoding is an information diagnostic, not a deployed bypass
of BANC or an autonomous controller. The temporal split is predeclared.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.neuromuscular import LEGS, build_spec
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.identify_c_banc_m3_decoder import quality


def readout(x, y, train, valid, scale):
    # Fixed declared units avoid amplifying neurons nearly silent during train.
    mean = x[train].mean(axis=0)
    active = x[train].std(axis=0) > 1e-4
    x = (x[:, active] - mean[active]) / scale
    center = float(y[train].mean())
    a = x[train]
    dual = np.linalg.solve(a @ a.T + 10.0 * np.eye(len(a)), y[train] - center)
    prediction = x @ (a.T @ dual) + center
    return {
        "active_features": int(active.sum()),
        "train": quality(prediction, y, train),
        "validation": quality(prediction, y, valid),
    }, prediction


def run(source, out, graph_path):
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "spec.json",
        {
            "experiment": "M3-E04b",
            "source_sha256": sha256(source / "signals.npz"),
            "tool_sha256": sha256(__file__),
            "hypothesis": "Teacher kinematic information is present in measured receptors or direct presynaptic rates but not available to the current static motor decoder",
            "competing_explanation": "Relevant sensory features or upstream BANC activity are themselves insufficient/time-misaligned/nonstationary",
            "models": [
                "per-leg raw receptor features",
                "per-leg direct presynaptic rates with actual edges to knee muscles",
                "per-leg knee motor units",
            ],
            "fit": "Unrestricted ridge readouts with intercept, train-only centering and fixed physical scales (receptor1, rate200); fixed alpha=10; no phase clock, future samples or lag search",
            "split": "Train [.5,1.2), validation[1.5,2.2), same exposed E03b development data",
            "decision": "If all six presynaptic validation R2>=.5, investigate existing motor incoming efficacy/dynamics. If receptors succeed but presynaptic activity fails, inspect afferent transfer first. No adoption or biological claim from a linear probe.",
            "validity": "Graph/motor/source identity, only actual incoming edges, training-only normalization, finite predictions",
            "cost": "CPU offline only; no new GPU or physics trajectory",
        },
    )
    d = np.load(source / "signals.npz")
    collection = json.loads((source / "collection-result.json").read_text())
    graph = GraphStore.load(graph_path)
    if collection["graph_hash"] != graph.hash:
        raise ValueError("Graph mismatch")
    motor_ids = graph.resolve(collection["motor_roster"])
    t = d["time_s"]
    train = (t >= 0.5) & (t < 1.2)
    valid = (t >= 1.5) & (t < 2.2)
    spec = build_spec(graph, 2)
    sizes = [len(row["sensory"]) for row in spec["rows"]]
    edges = np.r_[0, np.cumsum(sizes)]
    if edges[-1] != d["sensory"].shape[1]:
        raise ValueError("Different receptor order")
    results, predictions = {}, {}
    for leg, index, j in zip(LEGS, range(6), d["joints"][:, 5]):
        cols = np.flatnonzero(d["moment"][j])
        neurons = motor_ids[cols]
        pre = np.unique(
            np.concatenate(
                [graph.indices[graph.indptr[i] : graph.indptr[i + 1]] for i in neurons]
            )
        )
        pc = np.searchsorted(d["presyn_ids"], pre)
        if not np.array_equal(d["presyn_ids"][pc], pre):
            raise ValueError("Missing presynaptic readout")
        target = d["target"][:, j] - d["neutral"][j]
        cases = {
            "receptor_features": d["sensory"][:, edges[index] : edges[index + 1]],
            "direct_presynaptic": d["presyn_rates"][:, pc].astype(float),
            "motor_units": d["motor"][:, cols].astype(float),
        }
        results[leg] = {}
        for name, x in cases.items():
            report, pred = readout(
                x, target, train, valid, 1.0 if name == "receptor_features" else 200.0
            )
            if not np.isfinite(pred).all():
                raise ValueError("Nonfinite probe")
            results[leg][name] = report
            predictions[leg + "_" + name] = pred
    score = [results[leg]["direct_presynaptic"]["validation"]["r2"] for leg in LEGS]
    result = {
        "experiment": "M3-E04b",
        "run_status": "completed",
        "validity": "valid",
        "hypothesis_verdict": "presynaptic_expression_supported"
        if all(v >= 0.5 for v in score)
        else "presynaptic_screen_failed",
        "adoption": "not_applicable",
        "milestone_outcome": "pending",
        "results": results,
    }
    write_json(out / "result.json", result)
    np.savez_compressed(out / "predictions.npz", **predictions)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    a = p.parse_args()
    run(a.source, a.out, a.graph)
