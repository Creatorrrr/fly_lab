"""M3-E02a: replay recorded motor units through the unchanged decoder.

No neural or physical stepping. Pooling/request/target are measured separately;
contralateral phases here are output diagnostics, not gait acceptance tests.
"""

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.graph import GraphStore
from flylab.c.neuromuscular import DOFS, LEGS
from flylab.c.rate_body import RateBodyAdapter
from flylab.c.storage import StateStore
from tools.analyze_c_banc_biological_gait import circular_summary, sha256, write_json
from tools.analyze_c_banc_observation import new_body


def adapter_for(graph, body, saved):
    keys = inspect.signature(RateBodyAdapter).parameters
    adapter = RateBodyAdapter(
        graph,
        body,
        **{key: value for key, value in saved["parameters"].items() if key in keys},
    )
    if adapter.identity != saved["adapter"]["hash"]:
        raise ValueError("Decoder constructor does not match source checkpoint")
    adapter.restore(saved["adapter"])
    return adapter


def pool(adapter, rates):
    result = np.empty(adapter._pool_count)
    for indices, columns in adapter._pool_groups:
        result[indices] = np.mean(rates[columns], axis=1)
    return result


def raw_request(adapter, pooled):
    vectors = adapter._moment_plan(pooled, adapter.motor_gain).copy()
    vectors[:, 0] *= adapter.coxa_sign
    vectors[:, 5] *= adapter.knee_sign
    vectors *= adapter.joint_gain_multipliers
    result = np.zeros(42)
    result[adapter.joints] = vectors
    return result


def widths(a):
    return np.percentile(a, 95, axis=0) - np.percentile(a, 5, axis=0)


def phases(x):
    angle = np.angle(hilbert(x - np.mean(x, axis=0), axis=0))[4:-4]
    return {
        LEGS[a] + "-" + LEGS[b]: circular_summary(
            np.mod((angle[:, b] - angle[:, a]) / (2 * np.pi), 1)
        )
        for a, b in ((0, 3), (1, 4), (2, 5))
    }


def run(source, out, graph_path):
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "spec.json",
        {
            "experiment": "M3-E02a",
            "hypothesis": "Small knee motion is already restricted upstream of the accurately tracked target; compare muscle phases before and after filtering",
            "competing_explanation": "Incorrect decoder reconstruction or physical tracking causes the apparent attenuation",
            "source_sha256": sha256(source / "trace.json"),
            "tool_sha256": sha256(__file__),
            "validity": "Source adapter identity, recorded motor roster, exact target bytes for every control; independent inverse-filter agreement",
            "decision": "Locate attenuation and contralateral phase instability before choosing a structural intervention; no gait adoption",
            "intervention": "Recorded motor outputs only; no physics or neural simulation",
            "window_s": [0.5, 10.0],
        },
    )
    began = time.perf_counter()
    saved = StateStore.load(source / "start")
    trace = json.loads((source / "trace.json").read_text())
    graph = GraphStore.load(graph_path)
    body = new_body(saved)
    try:
        adapter = adapter_for(graph, body, saved)
        if (
            adapter.pooling != "mean"
            or adapter.muscle_activation_first
            or adapter.active_length
            or adapter.feasible_activation
            or adapter.fast_joint_release
            or adapter.coxa_endpoint
        ):
            raise ValueError("This diagnostic is specified for the frozen M2 decoder")
        if graph.hash != trace["graph_hash"]:
            raise ValueError("Different source graph")
        expected_roster = [graph.nodes[i]["id"] for i in adapter.motor_ids]
        # The captured array uses sorted graph indices, as declared by WalkingTrace.
        if trace.get("motor_ids", expected_roster) != expected_roster:
            raise ValueError("Different recorded motor roster")
        rows = trace["samples"]
        t = np.array([r["time_s"] for r in rows])
        rates = np.array([r["motor_rates"] for r in rows], dtype=np.float32)
        raw, request, pooled_rows = [], [], []
        max_inverse_error = 0.0
        for k, row in enumerate(rows[1:], 1):
            pooled = pool(adapter, rates[k])
            unbounded = raw_request(adapter, pooled)
            bounded = 0.5 * np.tanh(unbounded / 0.5)
            previous = adapter.offset.copy()
            target, _ = adapter.decode(rates[k])
            recorded = np.asarray(row["joint_targets_rad"])
            if target.tobytes() != recorded.tobytes():
                raise AssertionError(
                    f"Target mismatch at {k}: {np.max(np.abs(target - recorded))}"
                )
            alpha = np.full(42, 1 - np.exp(-0.005 / adapter.joint_activation_tau_s))
            if adapter.fast_coxa_activation:
                alpha[adapter.joints[:, 0]] = 1 - np.exp(-0.005 / 0.03)
            inverse = previous + (adapter.offset - previous) / alpha
            max_inverse_error = max(
                max_inverse_error, float(np.max(np.abs(inverse - bounded)))
            )
            raw.append(unbounded)
            request.append(bounded)
            pooled_rows.append(pooled)
        if max_inverse_error > 1e-12:
            raise AssertionError(
                "Direct request disagrees with independent inverse filter"
            )
        raw, request, pooled = map(np.asarray, (raw, request, pooled_rows))
        actual = np.array([r["joint_angles_rad"] for r in rows[1:]])
        target = np.array([r["joint_targets_rad"] for r in rows[1:]])
        window = (t[1:] >= 0.5) & (t[1:] < 10)
        knee = adapter.joints[:, DOFS.index("tibia_pitch")]
        muscle_labels = [
            {
                "leg": leg,
                "muscle": name,
                "motor_ids": [graph.nodes[i]["id"] for i in ids],
                "moment": moment.tolist(),
            }
            for leg, groups in zip(LEGS, adapter.muscles)
            for name, ids, moment in groups
        ]
        result = {
            "experiment": "M3-E02a",
            "run_status": "completed",
            "validity": "valid",
            "hypothesis_verdict": "pending interpretation",
            "adoption": "not_applicable",
            "milestone_outcome": "pending",
            "exact_targets": len(rows) - 1,
            "max_inverse_request_error_rad": max_inverse_error,
            "body_hash": body.model_hash,
            "adapter_hash": adapter.identity,
            "graph_hash": graph.hash,
            "wall_s": time.perf_counter() - began,
            "knee": {
                leg: {
                    "raw_width_deg": float(np.degrees(widths(raw[window, j]))),
                    "bounded_request_width_deg": float(
                        np.degrees(widths(request[window, j]))
                    ),
                    "target_width_deg": float(np.degrees(widths(target[window, j]))),
                    "actual_width_deg": float(np.degrees(widths(actual[window, j]))),
                    "mean_abs_tracking_error_deg": float(
                        np.degrees(np.abs(target[window, j] - actual[window, j]).mean())
                    ),
                    "request_near_bound_fraction": float(
                        np.mean(np.abs(request[window, j]) > 0.45)
                    ),
                }
                for leg, j in zip(LEGS, knee)
            },
            "knee_diagnostic_phase": {
                name: phases(values[window][:, knee] * adapter.knee_sign)
                for name, values in (
                    ("raw", raw),
                    ("bounded", request),
                    ("target", target),
                    ("actual", actual),
                )
            },
            "phase_scope": "All recorded .5–10s, both forward and backward/stopped; Hilbert after channel centering, 20ms edge trim. Does not substitute for physical foot AP or eligible gait phases.",
            "muscles": muscle_labels,
            "pooled_mean": pooled[window].mean(axis=0).tolist(),
            "pooled_width_p05_p95": widths(pooled[window]).tolist(),
        }
        np.savez_compressed(
            out / "stages.npz",
            time_s=t[1:],
            motor=rates[1:],
            pooled=pooled,
            raw=raw,
            request=request,
            target=target,
            actual=actual,
            joints=adapter.joints,
            knee_sign=adapter.knee_sign,
            window=window,
        )
        write_json(out / "result.json", result)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("muscles", "pooled_mean", "pooled_width_p05_p95")
                }
            ),
            flush=True,
        )
    finally:
        body.close()


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
