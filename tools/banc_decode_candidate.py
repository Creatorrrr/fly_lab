"""Synthetic CPU probe for ordered BANC muscle moment accumulation only.

No graph, physics body, GPU, or production mutation. The complete production
decode function is read from its AST and compared with a single candidate edit.
"""

import argparse
import ast
import copy
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class OrderedMoments:
    """Two ordered multiplies followed by each leg's +0-prefixed addition scan."""

    def __init__(self, muscles):
        self.counts = np.array([len(groups) for groups in muscles])
        self.legs = np.arange(len(muscles))
        self.moments = np.asarray(
            [moment for groups in muscles for _, _, moment in groups], dtype=np.float64
        )
        self.row_leg = np.repeat(self.legs, self.counts)
        self.row_col = np.concatenate([np.arange(count) + 1 for count in self.counts])
        self.weighted = np.empty_like(self.moments)
        self.terms = np.zeros(
            (len(muscles), int(self.counts.max()) + 1, 7), dtype=np.float64
        )
        self.accumulated = np.empty_like(self.terms)

    def __call__(self, pooled, gain):
        # pooled is float64 in the production decoder. Explicit conversion also
        # matches Python float(rate) if a synthetic input uses float32.
        rates = np.asarray(pooled, dtype=np.float64)
        np.multiply(rates[:, None], self.moments, out=self.weighted)
        np.multiply(self.weighted, gain, out=self.weighted)
        self.terms[self.row_leg, self.row_col] = self.weighted
        np.add.accumulate(self.terms, axis=1, dtype=np.float64, out=self.accumulated)
        # End at the actual muscle count, ignoring any padding after that row.
        return self.accumulated[self.legs, self.counts]


def reference_moments(pooled, muscles, gain):
    rates = iter(pooled)
    vectors = []
    for groups in muscles:
        vector = np.zeros(7)
        for _, _, moment in groups:
            rate = float(next(rates))
            vector += rate * moment * gain
        vectors.append(vector)
    return np.asarray(vectors)


def load_decode():
    path = ROOT / "flylab/c/rate_body.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RateBodyAdapter"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "decode"
    )
    source = "\n".join(
        line[4:] for line in text.splitlines()[method.lineno - 1 : method.end_lineno]
    )
    scope = {"np": np, "math": math, "LEGS": ("lf", "lm", "lh", "rf", "rm", "rh")}
    # Compile the fixed local production method to compare its actual arithmetic.
    exec(compile(source, str(path), "exec"), scope)  # noqa: S102
    reference = scope["decode"]
    marker = "    rates_by_muscle = iter(pooled)"
    vector = "        vector, output = np.zeros(7), {}"
    addition = "            vector += rate * moment * self.motor_gain"
    if any(source.count(token) != 1 for token in (marker, vector, addition)):
        raise ValueError(
            "Production decoder differs from the preregistered candidate boundary"
        )
    candidate = source.replace(
        marker, "    vectors = self._moment_plan(pooled, self.motor_gain)\n" + marker
    )
    candidate = candidate.replace(vector, "        vector, output = vectors[leg], {}")
    candidate = candidate.replace(
        addition,
        "            # Ordered moment arithmetic was computed above; diagnostics retain Python float.",
    )
    # The candidate is the same local method with the checked replacement above.
    exec(compile(candidate, "ordered-candidate-decode", "exec"), scope)  # noqa: S102
    return (
        reference,
        scope["decode"],
        hashlib.sha256(text.encode()).hexdigest(),
        source,
        candidate,
    )


def source_moments():
    tree = ast.parse((ROOT / "flylab/c/neuromuscular.py").read_text(encoding="utf-8"))
    fields = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("MUSCLE_VECTORS", "DOFS")
        ):
            fields[node.targets[0].id] = ast.literal_eval(node.value)
    return {
        name: np.array(
            [value.get(dof, 0.0) for dof in fields["DOFS"]], dtype=np.float64
        )
        for name, value in fields["MUSCLE_VECTORS"].items()
    }


def groups_for(counts, rng, *, dense=False, ltm=False):
    mapping = source_moments()
    required = [
        "tarsus_depressor_muscle",
        "long_tendon_muscle",
        "tarsus_levator_muscle",
    ]
    names = [name for name in mapping if name not in required]
    groups = []
    column = 0
    for count in counts:
        present = names[: count - len(required)] + required
        row = []
        for name in present:
            moment = rng.normal(size=7) if dense else mapping[name].copy()
            if ltm and name == "long_tendon_muscle":
                moment.fill(0.0)
            row.append((name, np.array([column]), moment))
            column += 1
        groups.append(row)
    return groups


def fixture(groups, *, pooling="mean", gain=0.002, ltm=False, endpoint=False):
    count = sum(map(len, groups))
    return SimpleNamespace(
        muscles=groups,
        motor_ids=np.arange(count),
        _pool_count=count,
        _pool_groups=[(np.arange(count), np.arange(count).reshape(-1, 1))],
        pooling=pooling,
        motor_gain=float(gain),
        coxa_endpoint=endpoint,
        coxa_sign=np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
        knee_sign=np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
        joints=np.arange(42).reshape(6, 7),
        offset=np.zeros(42),
        ltm_tendons=ltm,
        tendon_activation=np.zeros(6),
        last_motor=[],
        _moment_plan=OrderedMoments(groups),
        body=SimpleNamespace(
            neutral=np.linspace(-0.2, 0.2, 42),
            foot_kinematics=lambda targets: {
                "target_lift_mm": np.asarray(targets).reshape(6, 7)[:, 6] * 0.1
            },
        ),
    )


def identical(left, right, name):
    a, b = np.asarray(left), np.asarray(right)
    if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
        raise AssertionError(name + " differs in dtype/shape/bits")


def validate(reference_decode, candidate_decode, cases, seed):
    rng = np.random.default_rng(seed)
    comparisons = 0
    for counts in ([16] * 6, [17, 15, 16, 17, 15, 16]):
        for dense in (False, True):
            groups = groups_for(counts, rng, dense=dense)
            plan = OrderedMoments(groups)
            n = sum(counts)
            fixed = [
                np.zeros(n),
                np.full(n, -0.0),
                np.full(n, np.nextafter(0.0, 1.0)),
                np.full(n, np.finfo(np.float64).tiny),
                np.ones(n),
                np.full(n, 1e200),
            ]
            fixed.append(
                np.resize(
                    np.array([0.0, -0.0, 1.0, np.nextafter(1.0, 2.0), 1e-300, 1e200]), n
                )
            )
            # A negative zero term must still be added to the original +0 accumulator.
            for gain in (0.0, -0.0, 0.002, 0.02):
                for pooled in fixed + [
                    np.exp(rng.uniform(-690.0, 450.0, n)) for _ in range(cases)
                ]:
                    identical(
                        reference_moments(pooled, groups, gain),
                        plan(pooled, gain),
                        "ordered vectors",
                    )
                    comparisons += 1
    full_comparisons = 0
    for pooling in ("mean", "unit_response"):
        for ltm, endpoint in ((False, False), (True, False), (False, True)):
            groups = groups_for([16] * 6, rng, dense=endpoint, ltm=ltm)
            left = fixture(groups, pooling=pooling, ltm=ltm, endpoint=endpoint)
            right = fixture(
                copy.deepcopy(groups), pooling=pooling, ltm=ltm, endpoint=endpoint
            )
            for i in range(cases):
                rates = rng.lognormal(mean=3.0, sigma=2.0, size=96).astype(
                    np.float32 if i % 2 else np.float64
                )
                if i < 2:
                    rates.fill(0.0 if i == 0 else -0.0)
                cut = i % 17 == 0
                a, b = (
                    reference_decode(left, rates, motor_cut=cut),
                    candidate_decode(right, rates, motor_cut=cut),
                )
                for j, (x, y) in enumerate(zip(a, b)):
                    identical(x, y, f"complete decoder return {j}")
                identical(left.offset, right.offset, "filtered offset")
                identical(
                    left.tendon_activation, right.tendon_activation, "tendon activation"
                )
                if list(left.last_motor) != list(right.last_motor):
                    raise AssertionError("diagnostic dictionary changed")
                # Dict equality alone does not distinguish +/-0 diagnostic rates.
                for x, y in zip(left.last_motor, right.last_motor):
                    identical(
                        list(x["muscle_rate_units"].values()),
                        list(y["muscle_rate_units"].values()),
                        "diagnostic rate bits",
                    )
                full_comparisons += 1
    return {
        "vector_cases_bit_exact": comparisons,
        "complete_decode_steps_bit_exact": full_comparisons,
        "fixed_cases": "positive/negative zero, subnormal, normal, large finite and heterogeneous rates",
        "random_seed": seed,
        "models": "synthetic anatomy-shaped sparse moments and dense signed moments; no real BANC/body run",
    }


def timing(reference_decode, candidate_decode, iterations, pairs, seed):
    rng = np.random.default_rng(seed)
    groups = groups_for([16] * 6, rng)
    plan = OrderedMoments(groups)
    rates = rng.uniform(0.0, 200.0, 96)
    operations = {
        "moment_reference": lambda: reference_moments(rates, groups, 0.002),
        "moment_candidate": lambda: plan(rates, 0.002),
    }
    for name, method in (
        ("decode_reference", reference_decode),
        ("decode_candidate", candidate_decode),
    ):
        adapter = fixture(copy.deepcopy(groups), ltm=True)
        operations[name] = lambda method=method, adapter=adapter: method(adapter, rates)
    for operation in operations.values():
        for _ in range(100):
            operation()
    samples = {key: [] for key in operations}
    for pair in range(pairs):
        order = list(operations) if pair % 2 == 0 else list(reversed(operations))
        for key in order:
            began = time.perf_counter()
            for _ in range(iterations):
                operations[key]()
            samples[key].append((time.perf_counter() - began) / iterations)
    return {
        "seconds_per_call": samples,
        "medians_seconds": {
            key: statistics.median(values) for key, values in samples.items()
        },
        "pairs": pairs,
        "iterations_per_measurement": iterations,
        "scope": "prepared 96-muscle synthetic CPU inputs, decoded downstream state/diagnostics retained; matrix preparation, graph/physics/CUDA excluded",
    }


def main(args):
    out = args.out.resolve()
    if out.parent != (ROOT / "verification").resolve() or not out.name.startswith(
        "e2-decode-"
    ):
        raise ValueError("Use a new verification/e2-decode-* directory")
    out.mkdir(exist_ok=False)
    reference, candidate, source_hash, source, edited = load_decode()
    spec = {
        "hypothesis": "ordered multiplication and +0-prefixed addition scans preserve exact decode state while reducing Python/NumPy dispatch",
        "alternative": "vectorization changes reduction order, signed zero or downstream state, or preparation/copy costs erase any benefit",
        "acceptance": "all synthetic bytes identical and five paired CPU repetitions; synthetic speedup is not whole-BANC speedup",
        "production_source_sha256": source_hash,
        "numpy": np.__version__,
        "args": {**vars(args), "out": str(out)},
    }
    (out / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (out / "reference-decode.py.txt").write_text(source, encoding="utf-8")
    (out / "candidate-decode.py.txt").write_text(edited, encoding="utf-8")
    report = {"status": "RUNNING", "production_changed": False}
    try:
        report["validation"] = validate(reference, candidate, args.cases, args.seed)
        report["timing"] = timing(
            reference, candidate, args.iterations, args.pairs, args.seed
        )
        medians = report["timing"]["medians_seconds"]
        report.update(
            status="PASS",
            moment_speedup=medians["moment_reference"] / medians["moment_candidate"],
            synthetic_decode_speedup=medians["decode_reference"]
            / medians["decode_candidate"],
        )
        if (
            source_hash
            != hashlib.sha256(
                (ROOT / "flylab/c/rate_body.py").read_text(encoding="utf-8").encode()
            ).hexdigest()
        ):
            raise AssertionError("Production source changed during the probe")
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (out / "result.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73019)
    parser.add_argument("--cases", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--pairs", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.cases < 1 or arguments.iterations < 1 or arguments.pairs < 5:
        parser.error("Positive case/iteration counts and at least five pairs required")
    main(arguments)
