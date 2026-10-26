"""E321 generated CUDA initialization-order diagnostic; no BANC data is loaded."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback


MODES = ("torch_only", "cupy_first", "torch_first", "torch_cuda_first")
SIZES = ((96, 1173), (96, 391), (42, 96))
RTOL = ATOL = 1e-12
TORCH_RESERVED_LIMIT_BYTES = 256 * 1024**2
POST_INIT_DEVICE_INCREASE_LIMIT_BYTES = 1024**3


def error_record(phase: str, exc: BaseException) -> dict:
    return {
        "phase": phase,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc() if sys.exc_info()[0] is not None else None,
    }


def capture_error(report: dict, phase: str, exc: BaseException) -> None:
    item = error_record(phase, exc)
    if report["first_error"] is None:
        report["first_error"] = item
    else:
        report["secondary_errors"].append(item)


def array_meta(array) -> dict:
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "strides_bytes": list(array.strides),
        "c_contiguous": bool(array.flags.c_contiguous),
        "f_contiguous": bool(array.flags.f_contiguous),
    }


def tensor_meta(tensor) -> dict:
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "strides_elements": list(tensor.stride()),
        "strides_bytes": [int(s) * tensor.element_size() for s in tensor.stride()],
        "device": str(tensor.device),
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def generated_inputs(np, rows: int, cols: int):
    # Recreate identical, contiguous float64 operands in each independent process.
    rng = np.random.default_rng(321000 + rows * 10000 + cols)
    matrix = np.ascontiguousarray(rng.standard_normal((rows, cols)), dtype=np.float64)
    vector = np.ascontiguousarray(rng.standard_normal(cols), dtype=np.float64)
    direction = np.ascontiguousarray(rng.standard_normal(rows), dtype=np.float64)
    return matrix, vector, direction


def inspect_dlls(report: dict) -> None:
    try:
        import psutil

        paths = sorted({mapping.path for mapping in psutil.Process().memory_maps()
                        if mapping.path and any(token in mapping.path.lower()
                                                for token in ("cublas", "cudart", "cudnn", "nvrtc", "cuda.dll"))})
        report["dlls"] = {"status": "available", "paths": paths}
    except Exception as exc:  # Windows can deny process map access.
        report["dlls"] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def memory_snapshot(report: dict, torch, phase: str) -> bool:
    try:
        free, total = torch.cuda.mem_get_info()
        snapshot = {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "device_free_bytes": int(free),
            "device_total_bytes": int(total),
        }
        report["torch_memory"][phase] = snapshot
        if phase == "post_init_baseline":
            report["resource_limits"]["post_init_baseline_free_bytes"] = int(free)
        baseline = report["resource_limits"]["post_init_baseline_free_bytes"]
        if baseline is not None:
            snapshot["device_used_increase_from_post_init_bytes"] = max(0, baseline - int(free))
        if snapshot["peak_reserved_bytes"] > TORCH_RESERVED_LIMIT_BYTES:
            report["resource_stop_reason"] = (
                f"Torch peak reserved exceeds {TORCH_RESERVED_LIMIT_BYTES} bytes at {phase}")
        elif snapshot.get("device_used_increase_from_post_init_bytes", 0) > POST_INIT_DEVICE_INCREASE_LIMIT_BYTES:
            report["resource_stop_reason"] = (
                f"Device used increase from post-init baseline exceeds "
                f"{POST_INIT_DEVICE_INCREASE_LIMIT_BYTES} bytes at {phase}")
        return report["resource_stop_reason"] is None
    except Exception as exc:
        capture_error(report, f"memory_{phase}", exc)
        report["resource_stop_reason"] = f"Resource observation unavailable at {phase}"
        return False


def run_gemv(report: dict, np, torch, rows: int, cols: int) -> None:
    name = f"{rows}x{cols}"
    case = {
        "name": name,
        "forward": {"requested": False, "returned": False, "synchronized": False,
                    "host_returned": False, "confirmed": False},
    }
    report["gemv"].append(case)
    try:
        matrix, vector, direction = generated_inputs(np, rows, cols)
        expected = matrix @ vector
        case["host"] = {"matrix": array_meta(matrix), "vector": array_meta(vector),
                        "direction": array_meta(direction), "expected": array_meta(expected)}
        dm = torch.from_numpy(matrix).to(device="cuda")
        dv = torch.from_numpy(vector).to(device="cuda")
        case["device"] = {"matrix": tensor_meta(dm), "vector": tensor_meta(dv)}
        case["forward"]["requested"] = True
        start = time.perf_counter()
        output = torch.mv(dm, dv)
        case["forward"]["returned"] = True
        case["device"]["output"] = tensor_meta(output)
        torch.cuda.synchronize()
        case["forward"]["synchronized"] = True
        actual = output.detach().cpu().numpy()
        case["forward"]["host_returned"] = True
        case["forward"]["elapsed_s"] = time.perf_counter() - start
        finite = bool(np.isfinite(actual).all())
        close = bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL))
        case["forward"].update({"finite": finite, "allclose": close,
                                "max_abs_error": float(np.max(np.abs(actual - expected))) if finite else None,
                                "confirmed": finite and close,
                                "rtol": RTOL, "atol": ATOL})
        if not (finite and close):
            capture_error(report, f"gemv_{name}_comparison",
                          ValueError("generated GEMV comparison failed"))
    except Exception as exc:
        capture_error(report, f"gemv_{name}_forward", exc)
        case["forward"]["error"] = error_record(f"gemv_{name}_forward", exc)

    if not memory_snapshot(report, torch, f"after_{name}_forward"):
        if (rows, cols) == SIZES[0]:
            case["gradient_skipped_reason"] = report["resource_stop_reason"]
        return
    if (rows, cols) == SIZES[0]:
        run_gradient(report, np, torch, case)
        memory_snapshot(report, torch, f"after_{name}_gradient")


def run_gradient(report: dict, np, torch, case: dict) -> None:
    gradient = {"requested": False, "returned": False, "synchronized": False,
                "host_returned": False, "confirmed": False}
    case["gradient"] = gradient
    try:
        rows, cols = SIZES[0]
        matrix, vector, direction = generated_inputs(np, rows, cols)
        expected_matrix = np.outer(direction, vector)
        expected_vector = matrix.T @ direction
        dm = torch.from_numpy(matrix).to(device="cuda").requires_grad_(True)
        dv = torch.from_numpy(vector).to(device="cuda").requires_grad_(True)
        dd = torch.from_numpy(direction).to(device="cuda")
        case.setdefault("device", {})["gradient_direction"] = tensor_meta(dd)
        gradient["requested"] = True
        start = time.perf_counter()
        (torch.mv(dm, dv) * dd).sum().backward()
        gradient["returned"] = True
        torch.cuda.synchronize()
        gradient["synchronized"] = True
        actual_matrix = dm.grad.detach().cpu().numpy()
        actual_vector = dv.grad.detach().cpu().numpy()
        gradient["host_returned"] = True
        gradient["elapsed_s"] = time.perf_counter() - start
        finite = bool(np.isfinite(actual_matrix).all() and np.isfinite(actual_vector).all())
        close_matrix = bool(np.allclose(actual_matrix, expected_matrix, rtol=RTOL, atol=ATOL))
        close_vector = bool(np.allclose(actual_vector, expected_vector, rtol=RTOL, atol=ATOL))
        gradient.update({"finite": finite, "matrix_allclose": close_matrix,
                         "vector_allclose": close_vector, "confirmed": finite and close_matrix and close_vector,
                         "max_matrix_abs_error": float(np.max(np.abs(actual_matrix - expected_matrix)))
                         if bool(np.isfinite(actual_matrix).all()) else None,
                         "max_vector_abs_error": float(np.max(np.abs(actual_vector - expected_vector)))
                         if bool(np.isfinite(actual_vector).all()) else None,
                         "rtol": RTOL, "atol": ATOL})
        case["device"]["matrix_gradient"] = tensor_meta(dm.grad)
        case["device"]["vector_gradient"] = tensor_meta(dv.grad)
        if not gradient["confirmed"]:
            capture_error(report, "gradient_comparison",
                          ValueError("generated GEMV gradient comparison failed"))
    except Exception as exc:
        capture_error(report, "gradient", exc)
        gradient["error"] = error_record("gradient", exc)


def execute(report: dict) -> None:
    sequence_start = time.perf_counter()
    import numpy as np

    report["versions"]["numpy"] = np.__version__
    report["versions"]["python"] = sys.version
    report["versions"]["module"] = __file__
    mode = report["mode"]
    torch = cp = None

    def mark(event: str) -> None:
        report["order"].append(event)
        report["timeline"].append({"event": event,
                                   "elapsed_s": time.perf_counter() - sequence_start})

    def import_torch():
        nonlocal torch
        mark("import torch requested")
        import torch as imported
        torch = imported
        mark("import torch returned")

    def import_cupy():
        nonlocal cp
        mark("import cupy requested")
        import cupy as imported
        cp = imported
        mark("import cupy returned")

    def cupy_context():
        mark("cupy memGetInfo requested")
        free, total = cp.cuda.runtime.memGetInfo()
        report["cupy_mem_info"] = {"free_bytes": int(free), "total_bytes": int(total)}
        mark("cupy memGetInfo returned")

    def torch_context():
        mark("torch cuda init requested")
        torch.cuda.init()
        torch.cuda.synchronize()
        mark("torch cuda init synchronized")

    try:
        if mode == "torch_only":
            import_torch()
            torch_context()
        elif mode == "cupy_first":
            import_cupy()
            import_torch()
            cupy_context()
            torch_context()
        elif mode == "torch_first":
            import_torch()
            import_cupy()
            cupy_context()
            torch_context()
        else:
            import_torch()
            torch_context()
            import_cupy()
            cupy_context()
        report["initialization_complete"] = True
    except Exception as exc:
        capture_error(report, "initialization", exc)
    finally:
        inspect_dlls(report)

    if torch is not None:
        report["versions"].update({"torch": torch.__version__,
                                   "torch_cuda_build": torch.version.cuda})
    if cp is not None:
        report["versions"]["cupy"] = cp.__version__

    if not report["initialization_complete"]:
        report["gemv_skipped_reason"] = "Selected initialization sequence did not complete"
        return
    try:
        report["device"] = {"index": int(torch.cuda.current_device()),
                            "name": torch.cuda.get_device_name(),
                            "capability": list(torch.cuda.get_device_capability())}
    except Exception as exc:
        capture_error(report, "torch_metadata", exc)
    if cp is not None:
        try:
            report["versions"]["cupy_cuda_runtime"] = int(cp.cuda.runtime.runtimeGetVersion())
        except Exception as exc:
            capture_error(report, "cupy_metadata", exc)
    if not memory_snapshot(report, torch, "post_init_baseline"):
        report["gemv_skipped_reason"] = report["resource_stop_reason"]
        return
    for rows, cols in SIZES:
        run_gemv(report, np, torch, rows, cols)
        if report["resource_stop_reason"] is not None:
            break
    inspect_dlls(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", required=True, type=Path,
                        help="New JSON output path; an existing file is rejected")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must name a fresh file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "E321_generated_cuda_initialization_v1", "mode": args.mode,
              "pid": os.getpid(), "generated_only": True,
              "original_data_reads": 0, "model_loads": 0, "body_runs": 0, "fit_runs": 0,
              "order": [], "timeline": [], "initialization_complete": False, "versions": {},
              "device": None, "cupy_mem_info": None,
              "dlls": {"status": "not_inspected"}, "torch_memory": {},
              "resource_limits": {
                  "torch_peak_reserved_limit_bytes": TORCH_RESERVED_LIMIT_BYTES,
                  "post_init_device_increase_limit_bytes": POST_INIT_DEVICE_INCREASE_LIMIT_BYTES,
                  "post_init_baseline_excludes_initialization_context": True,
                  "post_init_baseline_free_bytes": None,
              }, "resource_stop_reason": None, "gemv": [],
              "first_error": None, "secondary_errors": []}
    start = time.perf_counter()
    try:
        execute(report)
    except Exception as exc:
        capture_error(report, "top_level", exc)
    report["wall_s"] = time.perf_counter() - start
    report["diagnostic_complete"] = True
    # Exclusive creation ensures this run never overwrites prior evidence.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"E321 diagnostic saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
