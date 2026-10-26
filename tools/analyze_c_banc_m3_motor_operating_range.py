"""E198 saved E150 motor operating-range diagnostic; no state advancement."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {"all": slice(0, 2000), "startup": slice(0, 99), "gait": slice(99, 2000)}
QUANTILES = (0.0, 0.05, 0.5, 0.95, 1.0)
INPUT_FILES = {
    "signals": "recruitment-signals.npz",
    "metadata": "recruitment-metadata.npz",
    "native": "native-recruitment-state.npz",
}
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_INTERNAL_SECONDS = 15.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(record, key: str, shape: tuple[int, ...], dtype=None) -> np.ndarray:
    require(key in record, f"missing {key}")
    value = np.asarray(record[key])
    require(value.shape == shape, f"{key} shape {value.shape}, expected {shape}")
    if dtype is not None:
        require(value.dtype == dtype, f"{key} dtype {value.dtype}, expected {dtype}")
    require(value.dtype.kind in "biuf", f"{key} must be numeric")
    require(np.all(np.isfinite(value)), f"{key} is not finite")
    return value


def _counts(values: np.ndarray) -> dict[str, int]:
    return {"always_zero": int(np.count_nonzero(~np.any(values, axis=0))),
            "partly_active": int(np.count_nonzero(np.any(values, axis=0) & ~np.all(values, axis=0))),
            "always_active": int(np.count_nonzero(np.all(values, axis=0)))}


def analyze_arrays(signals: dict, metadata: dict, native: dict) -> tuple[dict, dict[str, np.ndarray]]:
    """Validate exact saved alignment, then calculate local static diagnostics.

    Output arrays use (window, motor) order, with window order all/startup/gait.
    The derivative at m=0 is chosen as zero; no future trajectory is predicted.
    """
    require("observed_ids" in metadata and "motor_ids" in metadata,
            "missing observed or motor IDs")
    n_observed = np.asarray(metadata["observed_ids"]).size
    n_motor = np.asarray(metadata["motor_ids"]).size
    observed = _array(metadata, "observed_ids", (n_observed,), np.dtype("int64"))
    motors = _array(metadata, "motor_ids", (n_motor,), np.dtype("int64"))
    require(n_observed > 0 and n_motor > 0 and
            len(np.unique(observed)) == n_observed and len(np.unique(motors)) == n_motor,
            "observed and motor IDs must be unique")
    require(np.array_equal(_array(signals, "observed_ids", (n_observed,), np.dtype("int64")), observed),
            "signal/metadata observed IDs differ")
    columns = np.searchsorted(np.sort(observed), motors)
    sorted_observed = np.sort(observed)
    require(np.all(columns < n_observed) and
            np.array_equal(sorted_observed[np.minimum(columns, n_observed - 1)], motors),
            "motor IDs are not an observed subset")
    inverse = {int(value): i for i, value in enumerate(observed)}
    columns = np.array([inverse[int(value)] for value in motors], dtype=np.int64)

    signal_time = _array(signals, "time_s", (2000,))
    require(np.max(np.abs(signal_time - np.arange(1, 2001) * 0.005)) <= 1e-9,
            "signal time grid")
    require(np.array_equal(_array(native, "control_tick", (2001,), np.dtype("int64")),
                           np.arange(2001)), "native control ticks")
    require(np.array_equal(_array(native, "network_tick", (2001,), np.dtype("int64")),
                           np.arange(2001) * 20), "native network ticks")
    require(np.max(np.abs(_array(native, "physics_time_s", (2001,)) -
                          np.arange(2001) * 0.005)) <= 1e-9,
            "native time grid")
    for key in ("complete", "motor_rate_available"):
        require(np.all(_array(native, key, (2001,), np.dtype("bool"))),
                f"native {key} incomplete")
    require(np.max(np.abs(signal_time - native["physics_time_s"][1:])) <= 1e-9,
            "signal/native time axes differ")
    rate = _array(signals, "rate", (2000, n_observed))[:, columns].astype(np.float64)
    native_rate = _array(native, "motor_rate", (2001, n_motor))
    require(np.array_equal(rate, native_rate[1:]), "signal/native motor rate differs")
    require(np.all(rate >= 0) and np.all(native_rate >= 0), "negative motor rate")

    drive = _array(signals, "drive", (2000, n_observed))[:, columns].astype(np.float64)
    current = _array(signals, "recurrent_current", (2000, n_observed))[:, columns].astype(np.float64)
    adaptation = _array(signals, "adaptation", (2000, n_observed))[:, columns].astype(np.float64)
    require(np.all(adaptation >= 0), "negative adaptation")
    parameters = {}
    for key in ("a", "threshold", "tau", "cap", "gain", "adaptation_log_tau"):
        values = _array(metadata, key, (n_observed,)).astype(np.float64)[columns]
        if key in ("a", "tau", "cap"):
            require(np.all(values > 0), f"{key} must be positive")
        parameters[key] = values

    u = drive + current - parameters["threshold"]
    m = u - adaptation
    h = np.tanh(parameters["a"] * m / parameters["cap"])
    relative_slope = 1.0 - h * h
    phi = np.maximum(parameters["cap"] * h, 0.0)
    active = m > 0
    input_limited = u <= 0
    adaptation_limited = (u > 0) & (m <= 0)
    saturated = active & (relative_slope <= 0.1)
    da = np.where(active, m * relative_slope, 0.0)
    dthreshold = np.where(active, -parameters["a"] * relative_slope, 0.0)
    require(np.all(np.isfinite(phi)) and np.all(np.isfinite(da)) and
            np.all(np.isfinite(dthreshold)), "nonfinite calculation")

    arrays: dict[str, np.ndarray] = {"motor_ids": motors.copy(), "window_names":
                                     np.array(tuple(WINDOWS), dtype="U8")}
    arrays.update({key: value for key, value in parameters.items()})
    for name, values in (("u", u), ("m", m), ("r", rate), ("phi", phi)):
        stats = np.stack([np.quantile(values[part], QUANTILES, axis=0, method="linear").T
                          for part in WINDOWS.values()])
        arrays[f"{name}_quantiles"] = stats
    for name, mask in (("active_fraction", active), ("input_limited_fraction", input_limited),
                       ("adaptation_limited_fraction", adaptation_limited),
                       ("saturated_fraction", saturated)):
        arrays[name] = np.stack([mask[part].mean(axis=0) for part in WINDOWS.values()])
    for name, derivative in (("dphi_da_rms", da), ("dphi_dthreshold_rms", dthreshold)):
        arrays[name] = np.stack([np.sqrt(np.mean(np.square(derivative[part]), axis=0))
                                 for part in WINDOWS.values()])
    summaries = {}
    for name, part in WINDOWS.items():
        target_counts = _counts(active[part])
        summaries[name] = {
            "ticks": [int(part.start + 1), int(part.stop)],
            "target_motor_counts": target_counts,
            "actual_rate_always_zero_count": int(np.count_nonzero(np.all(rate[part] == 0, axis=0))),
            "input_limited_only_count": int(np.count_nonzero(np.all(input_limited[part], axis=0))),
            "adaptation_limited_ever_count": int(np.count_nonzero(np.any(adaptation_limited[part], axis=0))),
            "saturated_ever_count": int(np.count_nonzero(np.any(saturated[part], axis=0))),
            "equal_motor_mean_fractions": {
                key: float(np.mean(arrays[key][list(WINDOWS).index(name)]))
                for key in ("active_fraction", "input_limited_fraction",
                            "adaptation_limited_fraction", "saturated_fraction")
            },
        }
    return ({"validity": "valid_saved_operating_range", "motor_count": int(n_motor),
             "quantile_method": "NumPy linear", "saturation_relative_slope_cutoff": 0.1,
             "derivative_at_m_zero": 0.0, "windows": summaries}, arrays)


def _checked_json(path: Path, expected_hash: str) -> dict:
    require(sha256(path) == expected_hash, f"SHA mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _source_path(relative: str) -> Path:
    require(isinstance(relative, str), "source path must be text")
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), "source path escapes repository")
    return path


def run(freeze_path: Path, freeze_sha256: str, review_path: Path,
        review_sha256: str, out: Path) -> dict:
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    phase = "hash_gate"
    loaded_input_roles: list[str] = []
    calculation_requested = False
    calculation_completed = False
    try:
        freeze = _checked_json(freeze_path, freeze_sha256)
        review = _checked_json(review_path, review_sha256)
        require(freeze.get("schema") == "flylab.m3.e198.motor-operating-range.freeze.v1",
                "freeze schema")
        require(review.get("schema") == "flylab.m3.e198.review.v1" and
                review.get("approved") is True and
                review.get("freeze_sha256") == freeze_sha256 and
                review.get("tool_sha256") == sha256(Path(__file__)) and
                review.get("test_sha256") == sha256(
                    ROOT / "tests/test_c_banc_motor_operating_range.py"), "review gate")
        require(freeze.get("shape") == {"samples": 2000, "observed": 446,
                                        "motors": 391, "dt_s": 0.005,
                                        "neural_steps_per_control": 20,
                                        "time_atol": 1e-9}, "freeze dimensions")
        require(freeze.get("windows") == {"all": [1, 2000], "startup": [1, 99],
                                           "gait": [100, 2000]}, "freeze windows")
        require(freeze.get("saturation_relative_slope") == 0.1 and
                freeze.get("quantile_method") == "linear" and
                freeze.get("kink_derivative") == 0.0, "freeze calculation contract")
        limits = freeze.get("resource_limits", {})
        require(limits.get("internal_s") == MAX_INTERNAL_SECONDS and
                limits.get("output_bytes") == MAX_OUTPUT_BYTES and
                limits.get("cpu_threads") == 1 and limits.get("gpu_count") == 0 and
                limits.get("original_attempts") == 1 and limits.get("retries") == 0,
                "freeze limits")
        require(freeze.get("owned_files") == [
                    "tools/analyze_c_banc_m3_motor_operating_range.py",
                    "tests/test_c_banc_motor_operating_range.py"], "freeze owned files")
        design = freeze["design"]
        require(design == "docs/BANC_M3_E198_MOTOR_OPERATING_RANGE_DESIGN_20260924.md" and
                sha256(_source_path(design)) == freeze["design_sha256"], "design SHA mismatch")
        preservation = freeze["preservation_sha256"]
        require(set(preservation) == {"configs/banc-walking/m2-calibrated-v1.json"},
                "preservation file contract")
        for relative, digest in preservation.items():
            require(sha256(_source_path(relative)) == digest,
                    f"preservation SHA mismatch: {relative}")
        provenance = {}
        for key, filename in INPUT_FILES.items():
            relative = freeze["input_roles"][key]
            path = _source_path(relative)
            require(path.name == filename and path.parent.name == "e150-native-monotone-body",
                    f"{key} source path")
            digest = freeze["input_sha256"][relative]
            require(sha256(path) == digest, f"{key} source SHA mismatch")
            provenance[key] = {"path": relative, "sha256": digest}
        require(set(freeze["input_sha256"]) == set(freeze["input_roles"].values()),
                "source role/hash mapping")
        manifest = freeze["source_manifest"]
        manifest_path = _source_path(manifest)
        require(manifest_path.name == "output-manifest.json" and
                manifest_path.parent.name == "e150-native-monotone-body",
                "source manifest path")
        require(sha256(manifest_path) == freeze["source_manifest_sha256"],
                "source manifest SHA mismatch")
        require(time.perf_counter() - start <= MAX_INTERNAL_SECONDS, "internal time limit")
        phase = "array_load"
        records = {}
        for key in INPUT_FILES:
            with np.load(_source_path(provenance[key]["path"]), allow_pickle=False) as archive:
                records[key] = {name: archive[name] for name in archive.files}
            loaded_input_roles.append(key)
        phase = "analysis"
        require(np.asarray(records["signals"]["rate"]).shape == (2000, 446) and
                np.asarray(records["metadata"]["observed_ids"]).shape == (446,) and
                np.asarray(records["metadata"]["motor_ids"]).shape == (391,) and
                np.asarray(records["native"]["motor_rate"]).shape == (2001, 391),
                "frozen source dimensions")
        calculation_requested = True
        result, arrays = analyze_arrays(records["signals"], records["metadata"], records["native"])
        calculation_completed = True
        require(time.perf_counter() - start <= MAX_INTERNAL_SECONDS, "internal time limit")
        phase = "output"
        np.savez_compressed(out / "per-mn.npz", **arrays)
        result.update({"experiment": "M3-E198", "run_status": "completed",
                       "freeze_sha256": freeze_sha256, "review_sha256": review_sha256,
                       "tool_sha256": review["tool_sha256"],
                       "test_sha256": review["test_sha256"],
                       "source_manifest": {"path": manifest,
                                           "sha256": freeze["source_manifest_sha256"]},
                       "source_inputs": provenance,
                       "loaded_input_roles": loaded_input_roles,
                       "calculation_requested": calculation_requested,
                       "calculation_completed": calculation_completed,
                       "original_calculation_count": 1, "fit_count": 0,
                       "neural_advance_count": 0, "physics_step_count": 0,
                       "boundary": "Saved E150 local static diagnostic; no gait or physiological adoption",
                       "wall_seconds": float(time.perf_counter() - start)})
        payload = (json.dumps(result, ensure_ascii=False, indent=2,
                              allow_nan=False) + "\n").encode("utf-8")
        require(sum(p.stat().st_size for p in out.iterdir()) + len(payload) <= MAX_OUTPUT_BYTES,
                "output byte limit")
        require(time.perf_counter() - start <= MAX_INTERNAL_SECONDS, "internal time limit")
        pending = out / "result.json.pending"
        pending.write_bytes(payload)
        require(time.perf_counter() - start <= MAX_INTERNAL_SECONDS, "internal time limit")
        pending.rename(out / "result.json")
        return result
    except Exception as exc:
        failure = {"experiment": "M3-E198", "run_status": "failed",
                   "validity": "inconclusive", "phase": phase,
                   "error_type": type(exc).__name__, "error": str(exc),
                   "partial": any(out.iterdir()),
                   "loaded_input_roles": loaded_input_roles,
                   "calculation_requested": calculation_requested,
                   "calculation_completed": calculation_completed,
                   "original_calculation_count": int(calculation_requested),
                   "fit_count": 0, "neural_advance_count": 0, "physics_step_count": 0,
                   "freeze_sha256": freeze_sha256, "review_sha256": review_sha256,
                   "wall_seconds": float(time.perf_counter() - start)}
        (out / "error.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.freeze, args.freeze_sha256, args.review,
                 args.review_sha256, args.out)
    print(json.dumps({"run_status": result["run_status"], "validity": result["validity"],
                      "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
