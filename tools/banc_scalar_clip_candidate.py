"""Bounded pure CPU probe of typed receptor scalar clipping; no model imports."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import statistics
import struct
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "flylab/c/neuromuscular.py"
OLD = "position=float(np.clip(angle/math.pi,0.,1.))"
NEW = (
    "position=angle/math.pi\n"
    "    position=float(0. if position<0. else 1. if position>1. else position)"
)


def scalar_reference(value):
    return float(np.clip(value, 0.0, 1.0))


def scalar_candidate(value):
    # Preserve the probed NaN payloads and signed zero with strict comparisons.
    return float(0.0 if value < 0.0 else 1.0 if value > 1.0 else value)  # noqa: FURB136


def packed(value):
    return struct.pack("=d", value)


def outcome(function, args):
    try:
        result = function(*args)
        return ("value", type(result).__name__, packed(result).hex())
    # Comparing exception type/message is part of this verification contract.
    except Exception as exc:  # noqa: BLE001
        return ("error", type(exc).__name__, str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args()
    output = (ROOT / args.out).resolve()
    if output.parent != ROOT / "verification" or not output.name.startswith(
        "e2-scalar-clip-"
    ):
        raise ValueError("Output must be a new verification/e2-scalar-clip-* directory")
    output.mkdir(exist_ok=False)
    source = SOURCE.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "typed_receptor_feature"
    )
    reference_source = "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])
    if reference_source.count(OLD) != 1:
        raise ValueError("Source clip expression differs from the reviewed candidate")
    candidate_source = reference_source.replace(OLD, NEW)
    deferred_source = reference_source.replace("    " + OLD + "\n", "").replace(
        "    if kind=='claw':", "    if kind=='claw':\n        " + OLD
    )
    namespace = {"np": np, "math": math}
    # Compile the fixed local function, never a supplied or downloaded program.
    exec(compile(reference_source, str(SOURCE), "exec"), namespace)  # noqa: S102
    reference = namespace["typed_receptor_feature"]
    namespace = {"np": np, "math": math}
    # Compile only the local function with the checked expression replacement.
    exec(  # noqa: S102
        compile(candidate_source, "candidate_typed_receptor_feature", "exec"), namespace
    )
    candidate = namespace["typed_receptor_feature"]
    namespace = {"np": np, "math": math}
    # Compile only the alternate local dependency placement for comparison.
    exec(compile(deferred_source, "deferred_typed_receptor_feature", "exec"), namespace)  # noqa: S102
    deferred = namespace["typed_receptor_feature"]
    (output / "reference.py.txt").write_text(reference_source, encoding="utf-8")
    (output / "candidate.py.txt").write_text(candidate_source, encoding="utf-8")
    (output / "deferred.py.txt").write_text(deferred_source, encoding="utf-8")
    spec = {
        "numpy": np.__version__,
        "source_sha256": source_hash,
        "pairs": 5,
        "iterations_per_measurement": args.iterations,
        "hypothesis": "Strict scalar comparisons preserve float64 clip bytes and reduce Python/NumPy overhead.",
        "criteria": "Every scalar and full typed output byte must match, including signed zero and NaN payloads.",
        "scope": "Synthetic pure CPU inputs only; unchanged expression order and other receptor math; no GPU or model run.",
    }
    (output / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    rng = np.random.default_rng(73020)
    special_bits = np.array(
        [
            0,
            0x8000000000000000,
            1,
            0x8000000000000001,
            0x000FFFFFFFFFFFFF,
            0x800FFFFFFFFFFFFF,
            0x0010000000000000,
            0x8010000000000000,
            0x7FEFFFFFFFFFFFFF,
            0xFFEFFFFFFFFFFFFF,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF8000000000000,
            0xFFF8000000000000,
            0x7FF8000000000001,
            0xFFF8000000000042,
            0x7FF0000000000001,
            0xFFF0000000000042,
        ],
        dtype=np.uint64,
    ).view(np.float64)
    near_bounds = np.array(
        [
            np.nextafter(0.0, -np.inf),
            np.nextafter(0.0, np.inf),
            np.nextafter(1.0, -np.inf),
            1.0,
            np.nextafter(1.0, np.inf),
        ]
    )
    raw_random = rng.integers(0, np.iinfo(np.uint64).max, 2048, dtype=np.uint64).view(
        np.float64
    )
    samples = np.concatenate(
        (
            special_bits,
            near_bounds,
            np.linspace(-2.0, 3.0, 4097),
            rng.normal(0.5, 20.0, 4096),
            raw_random,
        )
    )
    scalar_cases = 0
    failures = []
    for cast in (float, np.float64):
        for value in samples:
            value = cast(value)
            expected, actual = (
                packed(scalar_reference(value)),
                packed(scalar_candidate(value)),
            )
            scalar_cases += 1
            if expected != actual:
                failures.append(
                    {
                        "scope": "scalar",
                        "input": packed(value).hex(),
                        "expected": expected.hex(),
                        "actual": actual.hex(),
                    }
                )
                break
    receptor_spec = {"load_half_bw": 0.5, "touch_half_bw": 0.05}
    types = [
        ("claw", "SNpp50"),
        ("claw", "SNpp51"),
        ("claw", "unknown"),
        ("hook", "SNpp41"),
        ("hook", "SNpp39"),
        ("hook", "unknown"),
        ("club", "unknown"),
        ("load", "unknown"),
        ("touch", "unknown"),
        ("invalid", "unknown"),
    ]
    receptor_values = np.concatenate(
        (
            special_bits,
            near_bounds,
            np.linspace(-7.0, 7.0, 257),
            rng.normal(0.0, 20.0, 256),
        )
    )
    typed_cases = 0
    for cast in (float, np.float64):
        for value in receptor_values:
            value = cast(value)
            for kind, cell_type in types:
                arguments = (
                    kind,
                    cell_type,
                    value,
                    value,
                    cast(0.125),
                    value,
                    value,
                    receptor_spec,
                )
                expected, actual = (
                    outcome(reference, arguments),
                    outcome(candidate, arguments),
                )
                typed_cases += 1
                if expected != actual:
                    failures.append(
                        {
                            "scope": "typed",
                            "kind": kind,
                            "cell_type": cell_type,
                            "input": packed(value).hex(),
                            "expected": expected,
                            "actual": actual,
                        }
                    )
                alternative = outcome(deferred, arguments)
                if expected != alternative:
                    failures.append(
                        {
                            "scope": "typed_deferred",
                            "kind": kind,
                            "cell_type": cell_type,
                            "input": packed(value).hex(),
                            "expected": expected,
                            "actual": alternative,
                        }
                    )
    timings = {
        "scalar_reference": [],
        "scalar_candidate": [],
        "typed_reference": [],
        "typed_candidate": [],
        "typed_deferred": [],
    }
    scalar_inputs = list(rng.uniform(-1.0, 2.0, 256))
    typed_inputs = [
        (
            kind,
            cell_type,
            np.float64(0.7),
            np.float64(2.0),
            np.float64(0.1),
            np.float64(0.2),
            np.float64(0.03),
            receptor_spec,
        )
        for kind, cell_type in types[:-1]
    ]
    jobs = [
        ("scalar_reference", scalar_reference, [(x,) for x in scalar_inputs]),
        ("scalar_candidate", scalar_candidate, [(x,) for x in scalar_inputs]),
        ("typed_reference", reference, typed_inputs),
        ("typed_candidate", candidate, typed_inputs),
        ("typed_deferred", deferred, typed_inputs),
    ]
    if not failures:
        for _, function, inputs in jobs:
            for i in range(100):
                function(*inputs[i % len(inputs)])
        for pair in range(5):
            for name, function, inputs in (
                jobs if pair % 2 == 0 else list(reversed(jobs))
            ):
                start = time.perf_counter()
                for i in range(args.iterations):
                    function(*inputs[i % len(inputs)])
                timings[name].append((time.perf_counter() - start) / args.iterations)
    medians = {
        name: statistics.median(values) for name, values in timings.items() if values
    }
    if (
        hashlib.sha256(SOURCE.read_text(encoding="utf-8").encode()).hexdigest()
        != source_hash
    ):
        failures.append(
            {"scope": "source", "error": "Production source changed during probe"}
        )
    result = {
        "status": "PASS" if not failures else "FAIL",
        "production_changed": False,
        "numpy": np.__version__,
        "scalar_byte_cases": scalar_cases,
        "typed_output_or_exception_cases": typed_cases,
        "failures": failures[:20],
        "typed_deferred_output_or_exception_cases": typed_cases,
        "typed_timing_mix": "3 claw, 3 hook, 1 club, 1 load, 1 touch repeated; not a measured model port distribution",
        "coverage": "Dense/random/raw float64 bits, subnormal, signed zero, +/- infinity, signed quiet/signaling NaN payloads; Python float and NumPy float64.",
        "seconds_per_call": timings,
        "median_seconds_per_call": medians,
        "scope": "Microtiming only; no claim about complete simulation speed.",
    }
    if medians:
        result["scalar_speedup"] = (
            medians["scalar_reference"] / medians["scalar_candidate"]
        )
        result["typed_speedup"] = (
            medians["typed_reference"] / medians["typed_candidate"]
        )
        result["typed_deferred_speedup"] = (
            medians["typed_reference"] / medians["typed_deferred"]
        )
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raise SystemExit(main())
