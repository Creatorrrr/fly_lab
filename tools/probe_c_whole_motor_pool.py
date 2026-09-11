"""E28: complete BANC, fixed partial sensory remap, individual tibia outputs."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.probe_c_banc_reference import GRAPH_HASH, sha, write
from tools.probe_c_feco_identity import remap_inputs

E18_SEAL = "53227dbe32a54be51957a357e2e860ef9039db187ac154f9f53a6a81611fe456"
CANDIDATE_SHA = "2ee0bf3ac0844e9a6df855b6643bfb84f7cff84ed78729ae4af147f623e075e0"
POSES = ("q1", "q1.4", "q2", "q2.3")
FLEXORS = ("tibia_flexor_muscle", "accessory_tibia_flexor_muscle")
EXTENSOR = "tibia_extensor_muscle"


def assess_motor_pairs(motors, full, cut, final_full, final_cut):
    """Enumerate every declared flexor/extensor pair, without a Fast-only gate."""
    flex = [i for i, row in enumerate(motors) if row["target"] in FLEXORS]
    ext = [i for i, row in enumerate(motors) if row["target"] == EXTENSOR]
    if not flex or not ext or len(flex) + len(ext) != len(motors):
        raise ValueError("Only a complete declared tibia motor cohort is supported")
    if len({row["id"] for row in motors}) != len(motors):
        raise ValueError("Duplicate motor identity")
    groups = (full, cut, final_full, final_cut)
    if any(set(group) != set(POSES) for group in groups) or any(
        np.shape(value) != (len(motors),)
        or not np.isfinite(value).all()
        or np.any(np.asarray(value) < 0)
        for group in groups
        for value in group.values()
    ):
        raise ValueError("Four complete finite nonnegative pose readouts required")
    decay = all(
        np.all(np.asarray(value) < 2)
        for group in groups[2:]
        for value in group.values()
    )
    pairs = []
    q1ext = max(full["q1"][i] for i in ext)
    q2flex = max(full["q2"][i] for i in flex)
    for f in flex:
        for e in ext:
            tests = {
                "extension_flexor_ge_2": bool(full["q1"][f] >= 2),
                "extension_opponents_le_20pct": bool(q1ext <= 0.2 * full["q1"][f]),
                "flexion_extensor_ge_2": bool(full["q2"][e] >= 2),
                "flexion_opponents_le_20pct": bool(q2flex <= 0.2 * full["q2"][e]),
                "extension_claw_causal_delta_ge_2": bool(
                    full["q1"][f] - cut["q1"][f] >= 2
                ),
                "flexion_claw_causal_delta_ge_2": bool(
                    full["q2"][e] - cut["q2"][e] >= 2
                ),
                "all_tibia_decay_below_2": bool(decay),
            }
            pairs.append(
                {
                    "flexor_id": motors[f]["id"],
                    "extensor_id": motors[e]["id"],
                    "tests": tests,
                    "conditional_neural_candidate": all(tests.values()),
                }
            )
    return {
        "pairs": pairs,
        "candidate_pairs": sum(p["conditional_neural_candidate"] for p in pairs),
        "all_tibia_decay_below_2": bool(decay),
        "physical_qualified": False,
    }


def run(args):
    from flylab.c.graph import GraphStore
    from flylab.c.research_annotations import rate_weights
    from flylab.c.research_rate import ResearchRateNetwork

    began = time.perf_counter()
    profile_count = 2 if args.profile == "both" else 1
    model_limit_ms = (600 if args.baseline_only else 5400) * profile_count
    wall_limit = 60 if args.baseline_only else (180 if profile_count == 2 else 120)
    root = args.evidence_root
    seal_path = root / "morphology-evidence-manifest.json"
    if sha(seal_path) != E18_SEAL:
        raise ValueError("E18 evidence seal changed")
    seal = json.loads(seal_path.read_text())
    hashes = {str(Path(key)): value for key, value in seal["evidence"].items()}
    saved = root / "rate-morphology-01"
    for path in saved.glob("*"):
        if (
            path.is_file()
            and path.suffix in (".npz", ".json")
            and sha(path) != hashes[str(path.relative_to(root))]
        ):
            raise ValueError("Saved E18 evidence changed")
    old = json.loads((saved / "spec.json").read_text())
    old_report = json.loads((saved / "report.json").read_text())
    candidate_path = root / "feco-identity-01/claw-input-candidate.json"
    if sha(candidate_path) != CANDIDATE_SHA:
        raise ValueError("Exact partial functional remap changed")
    candidate = json.loads(candidate_path.read_text())
    for name in ("flylab/c/research_rate.py", "flylab/c/research_annotations.py"):
        if sha(Path(name)) != old["sources"][name]:
            raise ValueError("Existing numerical/weight kernel changed")
    graph = GraphStore.load(args.graph)
    if (
        graph.hash != GRAPH_HASH
        or graph.n != 158706
        or candidate["graph_hash"] != graph.hash
    ):
        raise ValueError("Complete original current BANC graph required")
    roots = np.array([node["root_id"] for node in graph.nodes])
    observed = np.array([row["index"] for row in old["observed"]])
    for i, row in zip(observed, old["observed"]):
        node = graph.nodes[i]
        if (
            node["id"] != row["id"]
            or node["cell_type"] != row["cell_type"]
            or node["source_annotations"]["peripheral_target_type"] != row["target"]
        ):
            raise ValueError("Exact current motor annotations changed")
    tibia_columns = np.array(
        [
            j
            for j, row in enumerate(old["observed"])
            if row["target"] in (*FLEXORS, EXTENSOR)
        ]
    )
    motors = [old["observed"][j] for j in tibia_columns]
    if len(motors) != 19 or len(observed) != 69:
        raise ValueError("Complete LF motor inventory changed")
    lookup = {root: i for i, root in enumerate(roots)}
    claws = np.array(
        [lookup[value] for value in sorted(candidate["previous_features"])]
    )
    if len(claws) != 27:
        raise ValueError("Actual LF claw cohort changed")
    with np.load(saved / "size-profiles.npz", allow_pickle=False) as data:
        sizes = {
            "partial_area": data["baseline"].copy(),
            "morphology_area": data["candidate"].copy(),
        }
    if args.profile != "both":
        sizes = {args.profile: sizes[args.profile]}
    parameters = {}
    model_hashes = {}
    for profile, size_values in sizes.items():
        if (
            size_values.shape != (graph.n,)
            or not np.isfinite(size_values).all()
            or np.any(size_values <= 0)
        ):
            raise ValueError("Invalid complete saved size profile")
        values = (
            np.full(graph.n, 0.02),
            1 / size_values,
            7.5 * size_values,
            np.full(graph.n, 200),
        )
        for name, value in zip(("tau", "a", "threshold", "cap"), values):
            parameters[profile + "_" + name] = value.astype(np.float32)
        models = {
            row["model_hash"]
            for row in old_report["cases"]
            if row["profile"] == profile
        }
        if len(models) != 1:
            raise ValueError("Ambiguous original full BANC model")
        model_hashes[profile] = models.pop()
    stimuli = {}
    changed_roots = set()
    for profile in sizes:
        for pose in POSES:
            with np.load(
                saved / (profile + "-" + pose + ".npz"), allow_pickle=False
            ) as data:
                columns, values = (
                    data["sensory_indices"].copy(),
                    data["sensory_drive"].copy(),
                )
                revised = remap_inputs(roots, columns, values, candidate)
                if (
                    values.shape != (600, 106)
                    or np.any(values[200:])
                    or np.any(revised[200:])
                ):
                    raise ValueError("Fixed pose input timing changed")
                changed = {
                    roots[columns[j]]
                    for j in np.flatnonzero(np.any(values != revised, axis=0))
                }
                expected = {
                    row["banc_root"]
                    for row in candidate["assignments"]
                    if row["candidate_feature"] != row["previous_feature"]
                }
                if changed != expected or len(changed) != 4:
                    raise ValueError(
                        "Only four declared functional channels may change"
                    )
                changed_roots |= changed
                stimuli[profile, pose] = (columns, values, revised)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "source").mkdir()
    source_paths = [
        Path(__file__),
        Path("tools/probe_c_feco_identity.py"),
        Path("flylab/c/research_annotations.py"),
        Path("flylab/c/research_rate.py"),
        Path("GOAL_PLAN.md"),
    ]
    for path in source_paths:
        (args.out / "source" / path.name).write_bytes(path.read_bytes())
    np.savez_compressed(args.out / "parameters.npz", root_ids=roots, **parameters)
    weights, annotation_audit = rate_weights(graph)
    if annotation_audit is not None or weights.shape != (graph.n, graph.n):
        raise ValueError("Unmodified whole count-based weights required")
    spec = {
        "test": "E28 entire BANC fixed sensory remap and individual motor pairs",
        "graph_path": str(args.graph),
        "graph_hash": graph.hash,
        "neurons": graph.n,
        "graph_pairs": len(graph.indices),
        "model_hashes": model_hashes,
        "parameters_sha256": sha(args.out / "parameters.npz"),
        "e18_seal_sha256": E18_SEAL,
        "candidate_sha256": CANDIDATE_SHA,
        "changed_sensory_roots": sorted(changed_roots),
        "observed": old["observed"],
        "tibia_columns": tibia_columns.tolist(),
        "claw_cut_indices": claws.tolist(),
        "poses": list(POSES),
        "sources": {str(path): sha(path) for path in source_paths},
        "input_timing_ms": {"on": 0, "off": 200, "end": 600},
        "neural_model_limit_s": model_limit_ms / 1000,
        "wall_limit_s": wall_limit,
        "physics_model_seconds": 0,
        "controller_adoption": False,
        "baseline_only": args.baseline_only,
        "profile_selection": args.profile,
    }
    write(args.out / "spec.json", spec)
    cases, qualification = [], {}
    total_ms = 0
    for profile in sizes:
        network = ResearchRateNetwork(
            weights,
            *[
                parameters[profile + "_" + key]
                for key in ("tau", "a", "threshold", "cap")
            ],
            device="cuda",
        )
        if network.identity != model_hashes[profile]:
            raise ValueError("Entire original rate model identity changed")
        trials = [("legacy", "q1")] + (
            [(kind, pose) for kind in ("full", "cut") for pose in POSES]
            if not args.baseline_only
            else []
        )
        summaries = {kind: {} for kind in ("full", "cut")}
        finals = {kind: {} for kind in ("full", "cut")}
        for kind, pose in trials:
            name = profile + "-" + kind + "-" + pose
            columns, legacy, revised = stimuli[profile, pose]
            inputs = legacy if kind == "legacy" else revised
            network.reset()
            if kind == "cut":
                network.set_muted(claws)
            states = {"rate_at_0ms": network.readout()}
            rates = np.empty((600, 69), np.float32)
            drive = np.zeros(graph.n, np.float32)
            for tick in range(600):
                drive.fill(0)
                drive[columns] = inputs[tick]
                network.advance(drive)
                if bool((network.rate < 0).any()):
                    raise ValueError(
                        "Negative complete-neuron state at a saved interval"
                    )
                rates[tick] = network.readout(observed)
                total_ms += 1
                if tick + 1 in (1, 20, 50, 100, 150, 199, 200, 201, 400, 600):
                    states["rate_at_" + str(tick + 1) + "ms"] = network.readout()
                if (
                    total_ms > model_limit_ms
                    or time.perf_counter() - began > wall_limit
                ):
                    raise TimeoutError("E28 neural/wall limit reached")
            snapshot = network.snapshot()
            mask = np.ones(graph.n, np.float32)
            if kind == "cut":
                mask[claws] = 0
            np.testing.assert_array_equal(snapshot["output_mask"], mask)
            if (
                network.tick != 6000
                or np.any(snapshot["drive"])
                or any(
                    not np.isfinite(value).all() or np.any(value < 0)
                    for value in (rates, *states.values())
                )
            ):
                raise ValueError("Invalid complete raw state or actual input removal")
            error = None
            error_detail = None
            if kind == "legacy":
                with np.load(saved / (profile + "-q1.npz"), allow_pickle=False) as data:
                    motor_errors = np.abs(rates - data["motor_rates"]).max(axis=1)
                    error_detail = {
                        "motor_max_error_by_ms": motor_errors.tolist(),
                        "full_state_max_errors": {
                            key: float(np.abs(states[key] - data[key]).max())
                            for key in (
                                "rate_at_0ms",
                                "rate_at_200ms",
                                "rate_at_400ms",
                                "rate_at_600ms",
                            )
                        },
                    }
                    error = max(
                        float(motor_errors.max()),
                        *error_detail["full_state_max_errors"].values(),
                    )
            stimulus, final = (
                rates[150:200].astype(float).mean(axis=0),
                rates[-50:].astype(float).mean(axis=0),
            )
            if kind != "legacy":
                summaries[kind][pose], finals[kind][pose] = (
                    stimulus[tibia_columns],
                    final[tibia_columns],
                )
            path = args.out / (name + ".npz")
            np.savez_compressed(
                path,
                motor_rates=rates,
                motor_indices=observed,
                motor_rate_sample_neural_ticks=np.arange(1, 601) * 10,
                input_indices=columns,
                input_drive=inputs,
                output_mask=mask,
                neural_tick=network.tick,
                **states,
            )
            cases.append(
                {
                    "name": name,
                    "profile": profile,
                    "condition": kind,
                    "pose": pose,
                    "valid": error is None or error <= 0.001,
                    "device": network.device,
                    "model_hash": network.identity,
                    "legacy_max_error": error,
                    "legacy_error_detail": error_detail,
                    "stimulus_tail": stimulus.tolist(),
                    "final_tail": final.tolist(),
                    "trace_sha256": sha(path),
                    "neural_model_seconds": 0.6,
                }
            )
            write(args.out / "cases.json", cases)
            print(
                json.dumps(
                    {
                        "case": name,
                        "stimulus_tibia_max": float(stimulus[tibia_columns].max()),
                        "final_tibia_max": float(final[tibia_columns].max()),
                        "legacy_max_error": error,
                    }
                ),
                flush=True,
            )
            if error is not None and error > 0.001 and not args.baseline_only:
                write(
                    args.out / "failure.json",
                    {
                        "status": "INVALID_BASELINE_REPRODUCTION",
                        "case": name,
                        "legacy_max_error": error,
                        "neural_model_seconds": total_ms / 1000,
                        "physics_model_seconds": 0,
                        "wall_seconds": time.perf_counter() - began,
                        "trace_sha256": sha(path),
                        "controller_adoption": False,
                    },
                )
                raise ValueError(
                    "Whole baseline reproduction failed; raw evidence saved"
                )
        if not args.baseline_only:
            qualification[profile] = assess_motor_pairs(
                motors,
                summaries["full"],
                summaries["cut"],
                finals["full"],
                finals["cut"],
            )
        del network
        gc.collect()
    result = {
        "status": ("BASELINE_DIAGNOSTIC" if args.baseline_only else "VALID_COMPLETE"),
        "cases": cases,
        "qualification": qualification,
        "neural_model_seconds": total_ms / 1000,
        "physics_model_seconds": 0,
        "wall_seconds": time.perf_counter() - began,
        "F3_F6": "fail",
        "controller_adoption": False,
        "full_goal_completed": False,
    }
    write(args.out / "report.json", result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("cases", "qualification")
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root", type=Path, default=Path("verification/body-control-20260911")
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/acquisitions/banc888-windows-20260910/bundle"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Record both old-condition comparisons; never run or qualify the remap",
    )
    parser.add_argument(
        "--profile", choices=("both", "partial_area", "morphology_area"), default="both"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
