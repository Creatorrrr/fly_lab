"""Independently audit saved E28 inputs, identities and local CPU integration."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flylab.c.graph import GraphStore


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rk4_interval(rate, weights, mask, parameters):
    tau, a, threshold, cap = [value.astype(np.float64) for value in parameters]

    # All external inputs have just been removed at the specified 200 ms.
    def rhs(value):
        current = weights @ (value * mask)
        return (
            np.maximum(cap * np.tanh((a / cap) * (current - threshold)), 0) - value
        ) / tau

    state = rate.astype(np.float64)
    dt = 0.0001
    for _ in range(10):
        k1 = rhs(state)
        k2 = rhs(state + dt * k1 / 2)
        k3 = rhs(state + dt * k2 / 2)
        k4 = rhs(state + dt * k3)
        state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return state


def run(out):
    root = out.parent
    spec = json.loads((out / "spec.json").read_text())
    report = json.loads((out / "report.json").read_text())
    old = root / "rate-morphology-01"
    old_spec = json.loads((old / "spec.json").read_text())
    for path, checksum in spec["sources"].items():
        assert sha(out / "source" / Path(path).name) == checksum
    seal = root / "morphology-evidence-manifest.json"
    assert (
        sha(seal)
        == spec["e18_seal_sha256"]
        == "53227dbe32a54be51957a357e2e860ef9039db187ac154f9f53a6a81611fe456"
    )
    previous = {
        str(Path(key)): value
        for key, value in json.loads(seal.read_text())["evidence"].items()
    }
    for name in ("spec.json", "report.json", "size-profiles.npz"):
        assert sha(old / name) == previous[str((old / name).relative_to(root))]
    graph = GraphStore.load(spec["graph_path"])
    assert (
        graph.hash
        == spec["graph_hash"]
        == "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"
    )
    assert graph.n == spec["neurons"] == 158706
    assert len(graph.indices) == spec["graph_pairs"] == 11584852
    roots = np.array([row["root_id"] for row in graph.nodes])
    assert sha(out / "parameters.npz") == spec["parameters_sha256"]
    with np.load(out / "parameters.npz", allow_pickle=False) as data:
        parameters = {key: data[key] for key in data.files}
    np.testing.assert_array_equal(parameters.pop("root_ids"), roots)
    with np.load(old / "size-profiles.npz", allow_pickle=False) as data:
        sizes = {"partial_area": data["baseline"], "morphology_area": data["candidate"]}
    values = np.asarray(np.sign(graph.weights) * graph.counts * 0.03, np.float32)
    matrix = csr_matrix(
        (values, graph.indices, graph.indptr), shape=(graph.n, graph.n), copy=True
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    for profile, checksum in spec["model_hashes"].items():
        size = sizes[profile]
        expected = (np.full(graph.n, 0.02), 1 / size, 7.5 * size, np.full(graph.n, 200))
        for key, value in zip(("tau", "a", "threshold", "cap"), expected):
            np.testing.assert_array_equal(
                parameters[profile + "_" + key], value.astype(np.float32)
            )
        identity = hashlib.sha256(b"flylab.research-rate.rk4.csr.v1")
        for value in (
            matrix.indptr,
            matrix.indices,
            matrix.data,
            *[
                parameters[profile + "_" + key]
                for key in ("tau", "a", "threshold", "cap")
            ],
        ):
            identity.update(np.ascontiguousarray(value).tobytes())
        identity.update(repr(0.0001).encode())
        assert identity.hexdigest() == checksum
    matrix = matrix.astype(np.float64)
    observed = [row["index"] for row in spec["observed"]]
    assert spec["observed"] == old_spec["observed"]
    for row in spec["observed"]:
        node = graph.nodes[row["index"]]
        assert node["id"] == row["id"] and node["cell_type"] == row["cell_type"]
        assert node["source_annotations"]["peripheral_target_type"] == row["target"]
    tibia = [
        j
        for j, row in enumerate(spec["observed"])
        if row["target"]
        in (
            "tibia_flexor_muscle",
            "accessory_tibia_flexor_muscle",
            "tibia_extensor_muscle",
        )
    ]
    assert tibia == spec["tibia_columns"] and len(tibia) == 19
    cand = root / "feco-identity-01/claw-input-candidate.json"
    assert (
        sha(cand)
        == spec["candidate_sha256"]
        == "2ee0bf3ac0844e9a6df855b6643bfb84f7cff84ed78729ae4af147f623e075e0"
    )
    candidate = json.loads(cand.read_text())
    root_lookup = {value: i for i, value in enumerate(roots)}
    claws = [root_lookup[value] for value in sorted(candidate["previous_features"])]
    assert claws == spec["claw_cut_indices"] and len(claws) == 27
    baseline_only = spec["baseline_only"]
    expected_names = {
        profile + "-" + kind + "-" + pose
        for profile in spec["model_hashes"]
        for kind, pose in (
            [("legacy", "q1")]
            if baseline_only
            else [("legacy", "q1")]
            + [(k, q) for k in ("full", "cut") for q in ("q1", "q1.4", "q2", "q2.3")]
        )
    }
    assert {row["name"] for row in report["cases"]} == expected_names
    assert len(report["cases"]) == len(expected_names)
    summaries, final, legacy_status, local_checks = {}, {}, {}, []
    for case in report["cases"]:
        name, profile, kind, pose = [
            case[key] for key in ("name", "profile", "condition", "pose")
        ]
        path = out / (name + ".npz")
        assert (
            sha(path) == case["trace_sha256"]
            and case["model_hash"] == spec["model_hashes"][profile]
        )
        reference = old / (profile + "-" + pose + ".npz")
        assert sha(reference) == previous[str(reference.relative_to(root))]
        with np.load(reference, allow_pickle=False) as data:
            legacy = {key: data[key] for key in data.files}
        with np.load(path, allow_pickle=False) as data:
            raw = {key: data[key] for key in data.files}
        inputs = legacy["sensory_drive"].copy()
        cols = legacy["sensory_indices"]
        np.testing.assert_array_equal(raw["input_indices"], cols)
        col_lookup = {roots[i]: j for j, i in enumerate(cols)}
        assert len(col_lookup) == len(cols) == 106
        if kind != "legacy":
            for change in candidate["assignments"]:
                sources = [
                    r
                    for r, feature in candidate["previous_features"].items()
                    if feature == change["candidate_feature"]
                ]
                assert sources
                curve = legacy["sensory_drive"][:, col_lookup[sources[0]]]
                for source in sources[1:]:
                    np.testing.assert_array_equal(
                        curve, legacy["sensory_drive"][:, col_lookup[source]]
                    )
                inputs[:, col_lookup[change["banc_root"]]] = curve
        np.testing.assert_array_equal(raw["input_drive"], inputs)
        assert not np.any(inputs[200:])
        mask = np.ones(graph.n, np.float32)
        if kind == "cut":
            mask[claws] = 0
        np.testing.assert_array_equal(raw["output_mask"], mask)
        np.testing.assert_array_equal(raw["motor_indices"], observed)
        np.testing.assert_array_equal(
            raw["motor_rate_sample_neural_ticks"], np.arange(1, 601) * 10
        )
        assert int(raw["neural_tick"]) == 6000
        rates = raw["motor_rates"]
        assert (
            rates.shape == (600, 69)
            and np.isfinite(rates).all()
            and not np.any(rates < 0)
        )
        for key, value in raw.items():
            if key.startswith("rate_at_"):
                assert (
                    value.shape == (graph.n,)
                    and np.isfinite(value).all()
                    and not np.any(value < 0)
                )
                ms = int(key[len("rate_at_") : -2])
                if ms:
                    np.testing.assert_array_equal(value[observed], rates[ms - 1])
                else:
                    np.testing.assert_array_equal(value, np.zeros(graph.n))
        before, after = (
            rates[150:200].astype(float).mean(axis=0),
            rates[-50:].astype(float).mean(axis=0),
        )
        np.testing.assert_array_equal(case["stimulus_tail"], before)
        np.testing.assert_array_equal(case["final_tail"], after)
        summaries[profile, kind, pose], final[profile, kind, pose] = (
            before[tibia],
            after[tibia],
        )
        if kind == "legacy":
            differences = np.abs(rates - legacy["motor_rates"]).max(axis=1)
            state_errors = {
                key: float(np.abs(raw[key] - legacy[key]).max())
                for key in (
                    "rate_at_0ms",
                    "rate_at_200ms",
                    "rate_at_400ms",
                    "rate_at_600ms",
                )
            }
            error = max(float(differences.max()), *state_errors.values())
            assert error == case["legacy_max_error"] and case["valid"] == (
                error <= 0.001
            )
            np.testing.assert_array_equal(
                case["legacy_error_detail"]["motor_max_error_by_ms"], differences
            )
            assert case["legacy_error_detail"]["full_state_max_errors"] == state_errors
            legacy_status[profile] = "PASS" if error <= 0.001 else "FAIL"
        check = (
            baseline_only
            or (kind == "full" and pose == "q1")
            or (kind == "cut" and pose == "q2")
        )
        if check:
            result = rk4_interval(
                raw["rate_at_200ms"],
                matrix,
                mask,
                [
                    parameters[profile + "_" + key]
                    for key in ("tau", "a", "threshold", "cap")
                ],
            )
            error = float(np.abs(result - raw["rate_at_201ms"]).max())
            assert np.isfinite(result).all() and np.all(result >= 0) and error <= 0.001
            local_checks.append({"case": name, "start_ms": 200, "max_error": error})
    if not baseline_only:
        flex = [
            i
            for i, j in enumerate(tibia)
            if spec["observed"][j]["target"] != "tibia_extensor_muscle"
        ]
        ext = [
            i
            for i, j in enumerate(tibia)
            if spec["observed"][j]["target"] == "tibia_extensor_muscle"
        ]
        for profile in spec["model_hashes"]:
            assert legacy_status[profile] == "PASS"
            q1, q2, c1, c2 = [
                summaries[profile, kind, pose]
                for kind, pose in (
                    ("full", "q1"),
                    ("full", "q2"),
                    ("cut", "q1"),
                    ("cut", "q2"),
                )
            ]
            decay = all(
                np.all(final[profile, kind, pose] < 2)
                for kind in ("full", "cut")
                for pose in ("q1", "q1.4", "q2", "q2.3")
            )
            expected_pairs = []
            for f in flex:
                for e in ext:
                    tests = {
                        "extension_flexor_ge_2": bool(q1[f] >= 2),
                        "extension_opponents_le_20pct": bool(
                            max(q1[ext]) <= 0.2 * q1[f]
                        ),
                        "flexion_extensor_ge_2": bool(q2[e] >= 2),
                        "flexion_opponents_le_20pct": bool(
                            max(q2[flex]) <= 0.2 * q2[e]
                        ),
                        "extension_claw_causal_delta_ge_2": bool(q1[f] - c1[f] >= 2),
                        "flexion_claw_causal_delta_ge_2": bool(q2[e] - c2[e] >= 2),
                        "all_tibia_decay_below_2": bool(decay),
                    }
                    expected_pairs.append(
                        {
                            "flexor_id": spec["observed"][tibia[f]]["id"],
                            "extensor_id": spec["observed"][tibia[e]]["id"],
                            "tests": tests,
                            "conditional_neural_candidate": all(tests.values()),
                        }
                    )
            qualification = report["qualification"][profile]
            assert qualification["pairs"] == expected_pairs
            assert qualification["candidate_pairs"] == sum(
                row["conditional_neural_candidate"] for row in expected_pairs
            )
            assert qualification["all_tibia_decay_below_2"] == bool(decay)
            assert qualification["physical_qualified"] is False
    np.testing.assert_allclose(
        report["neural_model_seconds"], len(expected_names) * 0.6, atol=1e-12, rtol=0
    )
    assert report["physics_model_seconds"] == 0 and report["F3_F6"] == "fail"
    result = {
        "status": "PASS_AUDIT",
        "cases": len(expected_names),
        "legacy_reproduction": legacy_status,
        "independent_cpu_intervals": local_checks,
        "additional_numerical_model_seconds": len(local_checks) * 0.001,
        "F3_F6": "fail",
        "controller_adoption": False,
    }
    (out / "independent-adjudication.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
