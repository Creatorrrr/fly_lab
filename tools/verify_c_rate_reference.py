"""GOAL_PLAN E5: fixed author MANC rate reference and independent CUDA RK4.

Run reference/compare with the isolated JAX runtime and cuda with fly_lab's
PyTorch runtime. This is a numerical comparison, not a BANC controller.
The downloaded author functions are loaded only after SHA256 verification.
"""

import argparse
import ast
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf8")


def load_author(root):
    import diffrax
    import jax
    import jax.numpy as jnp
    from jax.scipy.signal import correlate

    manifest = json.loads((root / "source-manifest.json").read_text())
    for item in manifest["files"]:
        data = (root / "source" / item["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Author source checksum mismatch: {item['path']}")
    scope = {
        "jax": jax,
        "jnp": jnp,
        "np": np,
        "jit": jax.jit,
        "correlate": correlate,
        "ODETerm": diffrax.ODETerm,
        "Dopri5": diffrax.Dopri5,
        "SaveAt": diffrax.SaveAt,
        "PIDController": diffrax.PIDController,
        "diffeqsolve": diffrax.diffeqsolve,
    }
    wanted = {
        "src/utils/sim_utils.py": {
            "sample_trunc_normal",
            "set_sizes",
            "find_peaks_1d",
            "neuron_oscillation_score_helper_jax",
            "neuron_oscillation_score",
            "compute_oscillation_score",
        },
        "src/simulation/vnc_sim.py": {
            "rate_equation_half_tanh",
            "reweight_connectivity",
            "run_single_simulation",
        },
    }
    for relative, names in wanted.items():
        path = root / "source" / relative
        module = ast.parse(path.read_text(encoding="utf8"))
        functions = [
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        if {node.name for node in functions} != names:
            raise ValueError("Missing selected author function")
        for node in functions:
            if node.name == "run_single_simulation":
                # Observability only: preserve the original result, also return
                # the unsanitized solution and its solver status/statistics.
                if ast.unparse(node.body[-1]) != "return result":
                    raise ValueError("Unexpected author return statement")
                node.body[-1].value = ast.Tuple(
                    elts=[
                        ast.Name(id="result", ctx=ast.Load()),
                        ast.Name(id="solution", ctx=ast.Load()),
                    ],
                    ctx=ast.Load(),
                )
        selected = ast.fix_missing_locations(
            ast.Module(body=functions, type_ignores=[])
        )
        exec(compile(selected, str(path), "exec"), scope)  # noqa: S102 -- Reviewed, pinned source functions.
    return scope


def motor_metrics(scope, rates, motor_indices):
    import jax.numpy as jnp

    motor_rates = rates[motor_indices, 230:]
    active = np.max(motor_rates, axis=1) > 0.01
    score, frequency = scope["compute_oscillation_score"](
        jnp.asarray(motor_rates), jnp.asarray(active), prominence=0.05
    )
    score, frequency = float(score), float(frequency)
    if not np.isfinite(score):
        raise ValueError("Nonfinite author oscillation score")
    return {
        "active_motor_indices": motor_indices[active].tolist(),
        "oscillation_score": score,
        "oscillating": score >= 0.5,
        "frequency_cycles_per_sample": frequency if np.isfinite(frequency) else None,
        "motor_max_rate": float(np.max(motor_rates)),
    }


def reference(root, out):
    import diffrax
    import jax
    import jax.numpy as jnp
    import pandas as pd
    import yaml

    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    scope = load_author(root)
    source = root / "source"
    metadata = pd.read_csv(
        source
        / "data/manc t1 connectome data/wTable_20250813_DNtoMN_unsorted_withModules.csv",
        index_col=0,
    )
    table = pd.read_csv(
        source / "data/manc t1 connectome data/W_20250813_DNtoMN_unsorted.csv"
    )
    n = len(metadata)
    if not (
        np.array_equal(metadata.index, np.arange(n))
        and np.array_equal(table["bodyId_pre"], metadata["bodyId"])
        and table.columns[1:].tolist() == [str(i) for i in range(n)]
        and metadata.iloc[31]["type"] == "DNg100"
        and int(metadata.iloc[31]["bodyId"]) == 10093
    ):
        raise ValueError("Matrix/metadata/stimulus ordering mismatch")
    weights = jnp.array(table.drop(columns="bodyId_pre").to_numpy().astype(float))
    del table
    settings = yaml.safe_load(
        (source / "configs/neuron_params/default.yaml").read_text()
    )
    keys = jax.random.split(jax.random.PRNGKey(1), 5)
    params = {}
    for i, (name, prefix) in enumerate(
        (("tau", "tau"), ("a", "a"), ("threshold", "threshold"), ("fr_cap", "frcap"))
    ):
        # Keep the original draw shape; select realization zero before execution.
        params[name] = scope["sample_trunc_normal"](
            keys[i], settings[prefix + "Mean"], settings[prefix + "Stdv"], (1024, n)
        )
    params["a"], params["threshold"] = scope["set_sizes"](
        metadata["size"].values, params["a"], params["threshold"]
    )
    params = {name: values[0] for name, values in params.items()}
    weights = scope["reweight_connectivity"](
        weights, settings["excitatoryMultiplier"], settings["inhibitoryMultiplier"]
    )
    sim = yaml.safe_load((source / "configs/sim/default.yaml").read_text())
    for field in ("T", "dt", "pulseStart", "pulseEnd", "rtol", "atol"):
        sim[field] = float(sim[field])
    t_axis = jnp.unique(
        jnp.clip(
            jnp.sort(
                jnp.arange(0, sim["T"] + sim["dt"] / 2, sim["dt"], dtype=jnp.float32)
            ),
            0,
            sim["T"],
        )
    )
    seed = jax.random.split(keys[4], 1024)[0]
    motors = np.flatnonzero(metadata["class"].to_numpy() == "motor neuron")
    arrays = {name: np.array(value) for name, value in params.items()}
    arrays.update(
        weights=np.array(weights), time_s=np.array(t_axis), motor_indices=motors
    )
    np.savez_compressed(out / "inputs.npz", **arrays)
    write_json(
        out / "spec.json",
        {
            "test": "GOAL_PLAN E5 fixed MANC reference",
            "author_commit": json.loads((root / "source-manifest.json").read_text())[
                "commit"
            ],
            "neurons": n,
            "motor_neurons": len(motors),
            "seed": 1,
            "draw_shape": [1024, n],
            "realization": 0,
            "stimulus_index": 31,
            "stimulus_body_id": 10093,
            "stimulus_units": "author rate-model input, not mV",
            "stimulus_value": 250,
            "solver": "author Dopri5",
            "reference_platform": jax.default_backend(),
            "jax_x64": bool(jax.config.jax_enable_x64),
            "jax_prng": str(jax.config.jax_default_prng_impl),
            "versions": {
                p: version(p) for p in ("jax", "jaxlib", "diffrax", "numpy", "pandas")
            },
            "sim": sim,
            "cuda_rk4_dt_s": 0.0001,
            "maximum_rmse": 0.05,
            "maximum_absolute_difference": 0.5,
            "adoption_requires": "finite complete solutions, zero unstimulated state, trajectory limits and equal active motor set/oscillation classification",
            "source_instrumentation": "Selected author functions; return (original result, raw solution) only. No 1024xNxN mask.",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs_sha256": hashlib.sha256(
                (out / "inputs.npz").read_bytes()
            ).hexdigest(),
            "biological_validation": False,
            "banc_adopted": False,
        },
    )
    report = {"status": "RUNNING", "conditions": {}}
    for name, amplitude in (("unstimulated", 0), ("stimulated", 250)):
        condition_start = time.perf_counter()
        inputs = jnp.zeros(n).at[31].set(amplitude)
        processed, solution = scope["run_single_simulation"](
            weights,
            params["tau"],
            params["a"],
            params["threshold"],
            params["fr_cap"],
            inputs,
            0.0,
            t_axis,
            sim["T"],
            sim["dt"],
            sim["pulseStart"],
            sim["pulseEnd"],
            sim["rtol"],
            sim["atol"],
            seed,
        )
        raw, processed = np.array(solution.ys).T, np.array(processed)
        success = bool(solution.result == diffrax.RESULTS.successful)
        finite = bool(np.isfinite(raw).all())
        np.savez_compressed(
            out / f"reference-{name}.npz",
            raw=raw,
            processed=processed,
            time_s=np.array(solution.ts),
        )
        condition = {
            "solver_success": success,
            "raw_finite": finite,
            "last_saved_time_s": float(solution.ts[-1]),
            "raw_min": float(raw.min()) if finite else None,
            "raw_max": float(raw.max()) if finite else None,
            "postprocessing_max_difference": float(np.max(np.abs(raw - processed)))
            if finite
            else None,
            "stats": {k: int(v) for k, v in solution.stats.items()},
            "wall_s": time.perf_counter() - condition_start,
        }
        if finite and success:
            condition.update(motor_metrics(scope, raw, motors))
        report["conditions"][name] = condition
        write_json(out / "reference-report.json", report)
        print(name, json.dumps(condition), flush=True)
        if not (finite and success and float(solution.ts[-1]) == 2.0):
            raise RuntimeError("Invalid unsanitized reference solution")
        if time.perf_counter() - start > 900:
            raise TimeoutError("E5 shared run wall limit")
    report.update(status="COMPLETE", wall_s=time.perf_counter() - start)
    write_json(out / "reference-report.json", report)


def cuda(out):
    import torch

    if (out / "cuda-report.json").exists():
        raise FileExistsError("Preserve the previous CUDA comparison")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this declared comparison")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    start = time.perf_counter()
    with np.load(out / "inputs.npz") as data:
        weights, tau, a, threshold, cap = [
            torch.as_tensor(data[name].copy(), device="cuda")
            for name in ("weights", "tau", "a", "threshold", "fr_cap")
        ]
        times = data["time_s"].copy()
    n = weights.shape[0]
    state = torch.zeros(n, device="cuda")
    drive = torch.zeros_like(state)
    coefficient = a / cap
    dt = 0.0001

    def derivative(value):
        return (
            torch.clamp_min(
                cap
                * torch.tanh(
                    coefficient * (drive + torch.mv(weights, value) - threshold)
                ),
                0.0,
            )
            - value
        ) / tau

    def interval():
        for _ in range(10):
            k1 = derivative(state)
            k2 = derivative(state + (dt / 2) * k1)
            k3 = derivative(state + (dt / 2) * k2)
            k4 = derivative(state + dt * k3)
            state.add_((dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4))

    with torch.inference_mode():
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            interval()
            interval()
        torch.cuda.current_stream().wait_stream(warmup)
        state.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            interval()
        report = {
            "status": "RUNNING",
            "backend": "cuda",
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "tf32": False,
            "dt_s": dt,
            "method": "independent RK4, ten steps per CUDA graph; exact integer-ms stimulus boundaries",
            "conditions": {},
        }
        for name, amplitude in (("unstimulated", 0), ("stimulated", 250)):
            condition_start = time.perf_counter()
            state.zero_()
            drive.zero_()
            output = torch.empty((n, 2001), device="cuda")
            output[:, 0].copy_(state)
            for tick in range(2000):
                if tick == 20:
                    drive[31] = amplitude
                if tick == 1999:
                    drive.zero_()
                graph.replay()
                output[:, tick + 1].copy_(state)
                if tick % 100 == 99:
                    torch.cuda.synchronize()
                    if not bool(torch.isfinite(state).all()):
                        raise RuntimeError(f"Nonfinite CUDA state at interval {tick}")
                    if time.perf_counter() - start > 900:
                        raise TimeoutError("E5 shared run wall limit")
            values = output.cpu().numpy()
            np.savez_compressed(out / f"cuda-{name}.npz", raw=values, time_s=times)
            report["conditions"][name] = {
                "raw_finite": bool(np.isfinite(values).all()),
                "last_time_s": 2.0,
                "raw_min": float(values.min()),
                "raw_max": float(values.max()),
                "wall_s": time.perf_counter() - condition_start,
            }
            write_json(out / "cuda-report.json", report)
            print(name, json.dumps(report["conditions"][name]), flush=True)
        report.update(status="COMPLETE", wall_s=time.perf_counter() - start)
        write_json(out / "cuda-report.json", report)


def compare(root, out, candidate_out=None):
    candidate_out = candidate_out or out
    scope = load_author(root)
    with np.load(out / "inputs.npz") as data:
        motors = data["motor_indices"]
    reference_report = json.loads((out / "reference-report.json").read_text())
    cuda_report = json.loads((candidate_out / "cuda-report.json").read_text())
    results = {}
    for name in ("unstimulated", "stimulated"):
        with np.load(out / f"reference-{name}.npz") as data:
            ref = data["raw"]
        with np.load(candidate_out / f"cuda-{name}.npz") as data:
            candidate = data["raw"]
        diff = candidate.astype(np.float64) - ref
        error = float(np.sqrt(np.mean(diff**2)))
        maximum = float(np.max(np.abs(diff)))
        metrics = motor_metrics(scope, candidate, motors)
        original = reference_report["conditions"][name]
        same_active = (
            original["active_motor_indices"] == metrics["active_motor_indices"]
        )
        same_oscillation = original["oscillating"] == metrics["oscillating"]
        valid = bool(
            reference_report["status"] == cuda_report["status"] == "COMPLETE"
            and original["solver_success"]
            and original["raw_finite"]
            and cuda_report["conditions"][name]["raw_finite"]
            and np.isfinite(candidate).all()
            and np.isfinite(ref).all()
        )
        zero = bool(np.count_nonzero(ref) == np.count_nonzero(candidate) == 0)
        passed = (
            valid
            and error <= 0.05
            and maximum <= 0.5
            and same_active
            and same_oscillation
            and (name != "unstimulated" or zero)
        )
        results[name] = {
            "valid": valid,
            "passed": passed,
            "rmse": error,
            "max_absolute_difference": maximum,
            "same_active_motors": same_active,
            "same_oscillation_classification": same_oscillation,
            "zero_state": zero,
            "cuda_metrics": metrics,
        }
    report = {
        "conditions": results,
        "valid": all(r["valid"] for r in results.values()),
        "implementation_equivalent": all(r["passed"] for r in results.values()),
        "biological_validation": False,
        "banc_adopted": False,
        "full_goal_completed": False,
    }
    write_json(candidate_out / "comparison.json", report)
    print(json.dumps(report), flush=True)
    return report["implementation_equivalent"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reference", "cuda", "compare"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path)
    args = parser.parse_args()
    if args.phase == "reference":
        reference(args.root, args.out)
    elif args.phase == "cuda":
        cuda(args.out)
    else:
        raise SystemExit(0 if compare(args.root, args.out, args.candidate_out) else 1)
