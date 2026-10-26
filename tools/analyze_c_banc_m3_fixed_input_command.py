"""E263: saved fixed-input three-muscle q19 command diagnostic (CPU only)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
SCHEMA = "flylab.m3.e263.fixed-input-command"
FREEZE_SHA = "d5203e4a1cd5c12224aa6d3c9178e2a1065e7bb6c2dc005a5e2c8ec2d0d60326"
OWNED = ("tools/analyze_c_banc_m3_fixed_input_command.py",
         "tests/test_c_banc_fixed_input_command.py")
ARMS = ("B", "C", "X")
MUSCLES = np.array([42, 43, 44], dtype=np.int64)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.write_bytes(encoded(value))


def pinned(name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Repository-relative slash path required")
    relative = Path(name)
    if relative.is_absolute() or any(part in (".", "..") for part in relative.parts):
        raise ValueError("Unsafe pinned path")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("Pinned path missing or outside repository")
    return path


def gate(freeze_path: str, freeze_sha: str, review_path: str,
         review_sha: str) -> tuple[dict, dict]:
    if freeze_sha != FREEZE_SHA or digest(Path(freeze_path)) != FREEZE_SHA:
        raise ValueError("E263 freeze identity differs")
    if digest(Path(review_path)) != review_sha:
        raise ValueError("E263 source review identity differs")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if (freeze.get("schema") != SCHEMA + ".freeze.v1"
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("rows") != 101
            or freeze.get("motor_count") != 391
            or freeze.get("muscle_count") != 96
            or freeze.get("joint_column") != 19
            or freeze.get("muscles") != MUSCLES.tolist()
            or freeze.get("startup_controls") != [1, 99]
            or freeze.get("cases") != [56, 70]
            or freeze.get("dt_s") != .005
            or freeze.get("gain") != .02
            or freeze.get("reconstruction_atol") != 1e-10
            or freeze.get("relative_tolerance") != .2
            or freeze.get("denominator_floor") != 1e-12
            or freeze.get("resource_limits", {}).get("original_attempts") != 1):
        raise ValueError("E263 frozen scientific contract differs")
    if (review.get("schema") != SCHEMA + ".review.v1"
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or review.get("source_sha256")
            != {name: digest(pinned(name)) for name in OWNED}):
        raise ValueError("E263 approved source review required")
    if digest(pinned(freeze["design"])) != freeze["design_sha256"]:
        raise ValueError("E263 design bytes differ")
    paths = freeze["paths"]
    if set(paths.values()) != set(freeze["input_sha256"]):
        raise ValueError("E263 input role/hash map differs")
    for mapping in (freeze["input_sha256"], freeze["reference_source_sha256"],
                    review.get("additional_input_sha256", {}),
                    review.get("additional_reused_source_sha256", {})):
        if not isinstance(mapping, dict):
            raise TypeError("E263 hash map required")
        for name, expected in mapping.items():
            if digest(pinned(name)) != expected:
                raise ValueError("E263 pinned bytes differ: " + name)
    if freeze["packages"] != {name: importlib.metadata.version(name)
                              for name in freeze["packages"]}:
        raise ValueError("E263 package versions differ")
    return freeze, review


def load_npz(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not set(keys).issubset(archive.files):
            raise ValueError("E263 required NPZ keys missing")
        return {key: archive[key].copy() for key in keys}


def load_inputs(freeze: dict, progress=None) -> dict:
    paths = freeze["paths"]
    specs = {
        "calibration": ("motor_ids", "moment", "rest", "pool", "membership"),
        "e260_decoder": ("motor_ids", "motor_columns", "bank0", "activation0",
                         "tau", "kappa", "slopes", "membership"),
        "e258_raw": ("muscle_joint_contribution", "difference_command",
                     "startup_baseline_error", "startup_candidate_error",
                     "baseline_command_reconstruction", "candidate_command_reconstruction"),
    }
    result = {}
    for role, keys in specs.items():
        result[role] = load_npz(pinned(paths[role]), keys)
        if progress:
            progress(role)
    for role in ("e260_prior_audit", "e260_final", "e262_final"):
        result[role] = json.loads(pinned(paths[role]).read_text(encoding="utf-8"))
        if progress:
            progress(role)
    for arm in ARMS:
        data = {"motor_rate": np.load(pinned(paths[arm + "_motor_rate"]),
                                      allow_pickle=False),
                "bank44": np.load(pinned(paths[arm + "_bank"]), allow_pickle=False),
                "activation44": np.load(pinned(paths[arm + "_activation44"]),
                                        allow_pickle=False)}
        if arm != "X":
            prefix = "B" if arm == "B" else "C"
            data["control"] = load_npz(pinned(paths[prefix + "_control_raw"]),
                                       ("control_tick", "network_tick", "physics_time_s",
                                        "motor_rate32", "activation", "command"))
            data["native"] = load_npz(pinned(paths[prefix + "_native_recruitment_state"]),
                                      ("control_tick", "network_tick", "physics_time_s",
                                       "complete", "bank_available", "activation_available",
                                       "motor_rate_available", "bank", "activation",
                                       "original_activation", "motor_rate"))
        result[arm] = data
        if progress:
            progress(arm)
    for role in ("tau", "kappa", "slopes", "membership", "fixed_muscles"):
        result[role] = np.load(pinned(paths[role]), allow_pickle=False)
        if progress:
            progress(role)
    return result


def checked(value, shape: tuple[int, ...], kind: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind != kind or not np.isfinite(array).all():
        raise ValueError(f"E263 required shape/type/finite differs: {shape}/{kind}")
    return array


def close(value: np.ndarray, reference: np.ndarray, label: str,
          tolerance: float = 1e-10) -> None:
    if value.shape != reference.shape or not np.allclose(value, reference,
                                                        rtol=0, atol=tolerance):
        raise ValueError("E263 saved alignment differs: " + label)


def validate(inputs: dict, *, rows: int = 101, motors: int = 391,
             muscles: int = 96) -> dict:
    calibration = inputs["calibration"]
    params = inputs["e260_decoder"]
    ids = checked(calibration["motor_ids"], (motors,), "i")
    membership = checked(calibration["membership"], (motors,), "i")
    pool = checked(calibration["pool"], (muscles, motors), "f")
    moment = checked(calibration["moment"], (42, muscles), "f")
    rest = checked(calibration["rest"], (42,), "f")
    if (len(np.unique(ids)) != motors or not np.array_equal(ids, np.sort(ids))
            or not np.array_equal(ids, checked(params["motor_ids"], (motors,), "i"))
            or not np.array_equal(membership, checked(inputs["membership"], (motors,), "i"))
            or np.any((membership < 0) | (membership >= muscles))
            or np.any(pool < 0) or not np.allclose(pool.sum(axis=1), 1, rtol=0, atol=1e-12)
            or not np.array_equal(np.argmax(pool, axis=0), membership)
            or not np.array_equal((pool > 0).sum(axis=0), np.ones(motors))):
        raise ValueError("E263 motor roster/ownership differs")
    expected_moment = np.array([-1.8088900095777785, -1.8088900095777785,
                                1.8088900095777785])
    nonzero = np.flatnonzero(moment[19] != 0)
    if (not np.array_equal(nonzero, MUSCLES)
            or not np.allclose(moment[19, MUSCLES], expected_moment, rtol=0, atol=1e-12)):
        raise ValueError("E263 q19 anatomical moment differs")
    fixed = checked(inputs["fixed_muscles"], (19,), "i")
    if (len(np.unique(fixed)) != 19 or np.any((fixed < 0) | (fixed >= muscles))
            or np.intersect1d(fixed, MUSCLES).size):
        raise ValueError("E263 fixed muscle roster differs")
    columns = checked(params["motor_columns"], (2,), "i")
    if (not np.array_equal(columns, np.flatnonzero(membership == 44))
            or not np.array_equal(columns, [68, 107])
            or not np.array_equal(checked(params["membership"], (2,), "i"),
                                  membership[columns])):
        raise ValueError("E263 muscle44 owner columns differ")
    tau = checked(inputs["tau"], (3,), "f")
    kappa = checked(inputs["kappa"], (motors,), "f")
    slopes = checked(inputs["slopes"], (muscles, motors, 3, 2), "f")
    if (np.any(tau <= 0) or np.any(kappa < 0) or np.any(slopes < 0)
            or not np.array_equal(tau, np.array([.005, .01, .04]))
            or not np.array_equal(tau, np.asarray(params["tau"]))
            or not np.array_equal(kappa[columns], checked(params["kappa"], (2,), "f"))
            or not np.array_equal(slopes[44, columns],
                                  checked(params["slopes"], (2, 3, 2), "f"))):
        raise ValueError("E263 recruitment parameters differ")
    bank0 = checked(inputs["B"]["native"]["bank"], (301, motors, 3), "f")[0]
    close(bank0[columns], checked(params["bank0"], (2, 3), "f"), "initial bank44")
    close(bank0, checked(inputs["C"]["native"]["bank"], (301, motors, 3), "f")[0],
          "shared initial bank")
    if np.any(bank0 < 0):
        raise ValueError("E263 negative initial bank")
    for arm in ARMS:
        data = inputs[arm]
        rate = checked(data["motor_rate"], (rows, motors), "f")
        checked(data["bank44"], (rows, 2, 3), "f")
        checked(data["activation44"], (rows,), "f")
        if np.any(rate < 0) or np.any(data["bank44"] < 0) or np.any(data["activation44"] < 0):
            raise ValueError("E263 negative saved rate/bank/activation")
        if arm == "X":
            continue
        control = data["control"]
        native = data["native"]
        expected_tick = np.arange(301, dtype=np.int64)
        for source in (control, native):
            if (not np.array_equal(checked(source["control_tick"], (301,), "i"), expected_tick)
                    or not np.array_equal(checked(source["network_tick"], (301,), "i"), 20*expected_tick)
                    or not np.allclose(checked(source["physics_time_s"], (301,), "f"),
                                       .005*expected_tick, atol=1e-9, rtol=0)):
                raise ValueError("E263 E257 clocks differ")
        for flag in ("complete", "bank_available", "activation_available",
                     "motor_rate_available"):
            if not np.all(np.asarray(native[flag])[:rows]):
                raise ValueError("E263 incomplete E257 native state")
        close(rate, checked(control["motor_rate32"], (301, motors), "f")[:rows],
              arm + " control motor rate", 0)
        close(rate, checked(native["motor_rate"], (301, motors), "f")[:rows],
              arm + " native motor rate", 0)
        checked(control["activation"], (301, muscles), "f")
        checked(native["activation"], (301, muscles), "f")
        checked(native["original_activation"], (301, muscles), "f")
        checked(control["command"], (301, 42), "f")
        if np.any(control["activation"] < 0) or np.any(native["activation"] < 0):
            raise ValueError("E263 negative native activation")
        close(control["activation"][:rows], native["activation"][:rows],
              arm + " activation", 0)
    e258 = inputs["e258_raw"]
    for key, shape in (("muscle_joint_contribution", (301, 42, muscles)),
                       ("difference_command", (301, 42)),
                       ("startup_baseline_error", (99, 42)),
                       ("startup_candidate_error", (99, 42)),
                       ("baseline_command_reconstruction", (301, 42)),
                       ("candidate_command_reconstruction", (301, 42))):
        checked(e258[key], shape, "f")
    if (inputs["e260_prior_audit"].get("all_passed") is not True
            or inputs["e260_final"].get("all_passed") is not True
            or inputs["e262_final"].get("all_passed") is not True):
        raise ValueError("E263 prior independent audit required")
    return {"moment": moment, "rest": rest, "pool": pool, "membership": membership,
            "bank0": bank0, "tau": tau, "kappa": kappa, "slopes": slopes,
            "columns": columns, "fixed": fixed}


def transfer(bank: np.ndarray, kappa: np.ndarray, slopes: np.ndarray,
             pool: np.ndarray) -> np.ndarray:
    hinge = np.maximum(bank - kappa[:, None], 0)
    values = np.empty(3, np.float64)
    for index, muscle in enumerate(MUSCLES):
        members = np.flatnonzero(pool[muscle] > 0)
        low = slopes[muscle, members, :, 0]
        delta = slopes[muscle, members, :, 1]-low
        values[index] = np.sum(low*bank[members])+np.sum(delta*hinge[members])
    return values


def classify_ratio(numerator: float, denominator: float,
                   *, floor: float = 1e-12) -> float | None:
    return None if denominator <= floor else numerator/denominator


def analyze_arrays(inputs: dict, *, rows: int = 101, motors: int = 391,
                   muscles: int = 96, raw: dict | None = None, progress=None,
                   check_budget=None, on_bank_update=None) -> tuple[dict, dict]:
    if raw is None:
        raw = {}
    mapping = validate(inputs, rows=rows, motors=motors, muscles=muscles)
    bank0 = mapping["bank0"]
    alpha = -np.expm1(-.005/mapping["tau"])
    moment = mapping["moment"][19, MUSCLES]
    for arm in ARMS:
        rate = np.asarray(inputs[arm]["motor_rate"], np.float64)
        bank = bank0.copy()
        activation = np.full((rows, 3), np.nan, np.float64)
        q19 = np.full(rows, np.nan, np.float64)
        bank44 = np.full((rows, 2, 3), np.nan, np.float64)
        raw[arm + "_activation"] = activation
        raw[arm + "_q19"] = q19
        raw[arm + "_bank44"] = bank44
        raw[arm + "_completed_endpoints"] = np.array(0, np.int64)
        for control in range(rows):
            if control:
                bank += alpha[None, :]*(rate[control, :, None]-bank)
                if on_bank_update:
                    on_bank_update(arm, control)
            if arm != "X":
                close(bank, inputs[arm]["native"]["bank"][control],
                      arm + " full bank")
            values = transfer(bank, mapping["kappa"], mapping["slopes"],
                              mapping["pool"])
            activation[control] = values
            q19[control] = mapping["rest"][19]+.02*np.dot(moment, values)
            bank44[control] = bank[mapping["columns"]]
            raw[arm + "_completed_endpoints"][...] = control+1
            if check_budget and control % 16 == 0:
                check_budget()
        if (not np.isfinite(bank).all() or not np.isfinite(activation).all()
                or np.any(bank < 0) or np.any(activation < -1e-12)):
            raise ValueError("E263 reconstructed state invalid")
        close(bank44, inputs[arm]["bank44"], arm + " E260 bank44")
        close(activation[:, 2], inputs[arm]["activation44"], arm + " E260 activation44")
        if arm != "X":
            native = inputs[arm]["native"]
            close(activation, native["activation"][:rows, MUSCLES],
                  arm + " three native activations")
            close(activation, inputs[arm]["control"]["activation"][:rows, MUSCLES],
                  arm + " three original activations")
            close(q19[1:], inputs[arm]["control"]["command"][1:rows, 19],
                  arm + " original q19 command")
            raw[arm + "_row0_held_minus_decoder"] = np.array(
                inputs[arm]["control"]["command"][0, 19]-q19[0], np.float64)
        if progress:
            progress(arm)
    b, c = (raw[arm + "_q19"] for arm in ("B", "C"))
    contributions = {}
    for key, left, right in (("T", "C", "B"), ("C0", "X", "B"),
                             ("S", "C", "X")):
        delta = raw[left + "_activation"]-raw[right + "_activation"]
        contribution = .02*delta*moment[None, :]
        raw[key + "_contribution"] = contribution
        raw[key + "_positive"] = np.maximum(contribution, 0)
        raw[key + "_negative"] = np.minimum(contribution, 0)
        raw[key + "_q19"] = contribution.sum(axis=1)
        close(raw[key + "_q19"], raw[left + "_q19"]-raw[right + "_q19"],
              key + " three-muscle sum")
        contributions[key] = contribution
    e258 = inputs["e258_raw"]
    close(contributions["T"], e258["muscle_joint_contribution"][:rows, 19, MUSCLES],
          "E258 three-muscle projection")
    close(raw["T_q19"][1:], e258["difference_command"][1:rows, 19],
          "E258 full q19 difference")
    close(b[1:], e258["baseline_command_reconstruction"][1:rows, 19],
          "E258 baseline command")
    close(c[1:], e258["candidate_command_reconstruction"][1:rows, 19],
          "E258 candidate command")
    be = e258["startup_baseline_error"][:, 19]
    ce = e258["startup_candidate_error"][:, 19]
    close(ce[:rows-2]-be[:rows-2], raw["T_q19"][1:rows-1],
          "E258 startup error change")
    raw["baseline_error"] = be[:rows-2].copy()
    raw["candidate_error"] = ce[:rows-2].copy()
    raw["virtual_X_error"] = raw["baseline_error"]+raw["C0_q19"][1:rows-1]
    raw["control_tick"] = np.arange(rows, dtype=np.int64)
    startup = slice(1, rows-1)
    t = raw["T_q19"][startup]
    c0 = raw["C0_q19"][startup]
    l1 = classify_ratio(float(np.sum(np.abs(t-c0))), float(np.sum(np.abs(t))))
    case_report = {}
    for control in (56, 70):
        ratio = classify_ratio(float(abs(raw["T_q19"][control]-raw["C0_q19"][control])),
                               float(abs(raw["T_q19"][control])))
        same_direction = bool(raw["C0_q19"][control]*ce[control-1] > 0)
        case_report[str(control)] = {"T_rad": float(raw["T_q19"][control]),
                                     "C0_rad": float(raw["C0_q19"][control]),
                                     "S_rad": float(raw["S_q19"][control]),
                                     "ratio": ratio,
                                     "C0_same_direction_as_candidate_error": same_direction,
                                     "baseline_error_rad": float(be[control-1]),
                                     "candidate_error_rad": float(ce[control-1]),
                                     "virtual_X_error_rad": float(raw["virtual_X_error"][control-1]),
                                     "T_contributions_rad": contributions["T"][control].tolist(),
                                     "C0_contributions_rad": contributions["C0"][control].tolist()}
    inconclusive = l1 is None or any(entry["ratio"] is None
                                    for entry in case_report.values())
    supported = None if inconclusive else bool(l1 <= .2 and all(
        entry["ratio"] <= .2 and entry["C0_same_direction_as_candidate_error"]
        for entry in case_report.values()))
    analysis = {"H263_supported": supported,
                "hypothesis_status": ("inconclusive" if inconclusive else
                                      "supported" if supported else "refuted"),
                "startup_L1_residual_ratio": l1,
                "cases": case_report,
                "startup_sum_abs_T_rad": float(np.sum(np.abs(t))),
                "startup_sum_abs_T_minus_C0_rad": float(np.sum(np.abs(t-c0))),
                "startup_new_q19_violations_C": int(np.count_nonzero(
                    (np.abs(be[:rows-2]) <= .04) & (np.abs(ce[:rows-2]) > .04))),
                "startup_virtual_X_q19_violations": int(np.count_nonzero(
                    np.abs(raw["virtual_X_error"]) > .04)),
                "startup_max_abs_baseline_error_rad": float(np.max(np.abs(be[:rows-2]))),
                "startup_max_abs_candidate_error_rad": float(np.max(np.abs(ce[:rows-2]))),
                "startup_max_abs_virtual_X_error_rad": float(np.max(np.abs(raw["virtual_X_error"]))),
                "interpretation": "Conditional saved fixed-input command re-expression; X error is virtual, not a body trajectory; E260 muscle44 effect was previously observed; 20% criterion selected after prior data exposure."}
    if progress:
        progress("comparison")
    return analysis, raw


def error_record(error: BaseException, stage: str) -> dict:
    return {"stage": stage, "type": type(error).__name__, "message": str(error)}


def invalidate(result: dict) -> None:
    result["validity"] = "incomplete_or_invalid"
    result["H263_supported"] = None
    result["hypothesis_status"] = "not_tested"
    if isinstance(result.get("analysis"), dict):
        result["analysis"]["H263_supported"] = None
        result["analysis"]["hypothesis_status"] = "not_tested"


def write_raw(out: Path, raw: dict) -> None:
    path = out / "raw.npz"
    if not path.exists() and raw:
        np.savez_compressed(path, **raw)


def publish(out: Path, result: dict, ledger: dict, raw: dict,
            limit_bytes: int, deadline: float) -> None:
    write_raw(out, raw)
    write_json(out / "errors.json", ledger["errors"])
    write_json(out / "ledger.json", ledger)
    files = {path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
             for path in sorted(out.iterdir()) if path.is_file()
             and path.name not in ("result.json", "output-manifest.json")}
    manifest = {"schema": SCHEMA + ".manifest.v1", "files": files}
    manifest_bytes = encoded(manifest)
    result_bytes = encoded(result)
    if sum(item["bytes"] for item in files.values()) + len(manifest_bytes) + len(result_bytes) > limit_bytes:
        raise OSError("E263 output byte limit exceeded")
    if time.perf_counter() >= deadline:
        raise TimeoutError("E263 publication deadline")
    (out / "output-manifest.json").write_bytes(manifest_bytes)
    (out / "result.json").write_bytes(result_bytes)
    if time.perf_counter() >= deadline:
        raise TimeoutError("E263 deadline after result")


def settle(out: Path, result: dict, ledger: dict, raw: dict,
           limit_bytes: int, deadline: float) -> BaseException | None:
    first = None
    for attempt in range(3):
        result["errors"] = ledger["errors"]
        result["finalization_pending"] = False
        try:
            publish(out, result, ledger, raw, limit_bytes, deadline)
            return first
        except Exception as error:  # noqa: BLE001
            if first is None:
                first = error
            ledger["errors"].append(error_record(error, f"publication_{attempt+1}"))
            result.setdefault("primary_error", error_record(error, "publication"))
            invalidate(result)
            deadline = float("inf")
    result["errors"] = ledger["errors"]
    try:
        publish(out, result, ledger, raw, limit_bytes, deadline)
        return first
    except Exception as error:  # noqa: BLE001
        ledger["errors"].append(error_record(error, "terminal_publication"))
        result["finalization_pending"] = True
        invalidate(result)
        for name in ("output-manifest.json", "result.json"):
            try:
                (out / name).unlink(missing_ok=True)
            except Exception as secondary:  # noqa: BLE001
                ledger["errors"].append(error_record(secondary, "unlink_" + name))
        for name, value in (("errors.json", ledger["errors"]),
                            ("ledger.json", ledger), ("result.json", result)):
            try:
                result["errors"] = ledger["errors"]
                write_json(out / name, value)
            except Exception as secondary:  # noqa: BLE001
                ledger["errors"].append(error_record(secondary, "terminal_" + name))
        return first


def run(freeze_path: str, freeze_sha: str, review_path: str,
        review_sha: str, output: str, timeout_s: float = 10) -> dict:
    began = time.perf_counter()
    out = Path(output).resolve()
    if out != (BASE / "e263-fixed-input-command").resolve():
        raise ValueError("Exact E263 write-once output path required")
    if out.exists():
        raise FileExistsError("E263 output already exists")
    freeze, review = gate(freeze_path, freeze_sha, review_path, review_sha)
    limits = freeze["resource_limits"]
    if not math.isfinite(timeout_s) or timeout_s <= 0 or timeout_s > limits["internal_s"]:
        raise ValueError("E263 timeout exceeds frozen internal budget")
    if shutil.disk_usage(out.parent).free < limits["minimum_free_bytes"]:
        raise OSError("E263 free disk below frozen minimum")
    out.mkdir(parents=True)
    ledger = {"schema": SCHEMA + ".ledger.v1", "phase": "load",
              "loaded_input_roles": [], "analysis_requested_count": 0,
              "analysis_entered_count": 0, "analysis_completed_count": 0,
              "original_calculation_count": 0, "completed_stages": [],
              "bank_update_count": 0, "neural_constructors": 0,
              "body_constructors": 0, "neural_ticks": 0,
              "physics_steps": 0, "fit_calls": 0, "solver_calls": 0,
              "gpu_calls": 0, "retries": 0, "errors": []}
    result = {"schema": SCHEMA + ".result.v1", "validity": "incomplete_or_invalid",
              "H263_supported": None, "candidate_adopted": False,
              "whole_goal_complete": False, "F6_exposed": False,
              "active_UI_changed": False, "finalization_pending": True,
              "provenance": {"freeze_sha256": freeze_sha, "review_sha256": review_sha,
                             "source_sha256": review["source_sha256"],
                             "input_sha256": freeze["input_sha256"],
                             "reference_source_sha256": freeze["reference_source_sha256"]}}
    raw: dict = {}
    primary = None
    deadline = began + timeout_s

    def budget() -> None:
        if time.perf_counter() >= deadline-limits["finalization_reserve_s"]:
            raise TimeoutError("E263 analysis deadline")
        import psutil
        if psutil.Process().memory_info().rss > limits["resident_bytes"]:
            raise MemoryError("E263 RSS limit")

    try:
        limiter = None
        work_error = None
        restore_error = None
        try:
            ledger["threadpool_before"] = threadpool_info()
            limiter = threadpool_limits(limits=1)
            ledger["threadpool_observed"] = threadpool_info()
            if any(item.get("num_threads") != 1 for item in ledger["threadpool_observed"]):
                raise RuntimeError("E263 CPU thread limit differs")

            def loaded(role):
                ledger["loaded_input_roles"].append(role)
                budget()

            inputs = load_inputs(freeze, progress=loaded)
            ledger["phase"] = "analysis"
            ledger["analysis_requested_count"] = 1
            ledger["analysis_entered_count"] = 1
            ledger["original_calculation_count"] = 1

            def progressed(stage):
                ledger["completed_stages"].append(stage)
                budget()

            def bank_updated(_arm, _control):
                ledger["bank_update_count"] += 1

            analysis, raw = analyze_arrays(inputs, raw=raw, progress=progressed,
                                          check_budget=budget,
                                          on_bank_update=bank_updated)
            ledger["analysis_completed_count"] = 1
        except Exception as error:  # noqa: BLE001
            work_error = error
        finally:
            if limiter is not None:
                try:
                    limiter.restore_original_limits()
                    ledger["threadpool_restore_completed"] = True
                    ledger["threadpool_after"] = threadpool_info()
                except Exception as error:  # noqa: BLE001
                    restore_error = error
                    ledger["threadpool_restore_completed"] = False
        if work_error is not None:
            raise work_error
        if restore_error is not None:
            raise restore_error
        result["analysis"] = analysis
        result["H263_supported"] = analysis["H263_supported"]
        result["hypothesis_status"] = analysis["hypothesis_status"]
        result["validity"] = "valid_saved_diagnostic"
        ledger["phase"] = "publication"
    except Exception as error:  # noqa: BLE001
        primary = error
        ledger["errors"].append(error_record(error, ledger["phase"]))
        if "restore_error" in locals() and restore_error is not None and error is not restore_error:
            ledger["errors"].append(error_record(restore_error, "threadpool_restore"))
        result["primary_error"] = error_record(error, ledger["phase"])
        invalidate(result)
    result.update(loaded_input_roles=ledger["loaded_input_roles"],
                  analysis_requested_count=ledger["analysis_requested_count"],
                  analysis_entered_count=ledger["analysis_entered_count"],
                  analysis_completed_count=ledger["analysis_completed_count"],
                  original_calculation_count=ledger["original_calculation_count"],
                  bank_update_count=ledger["bank_update_count"],
                  neural_constructors=0, body_constructors=0, neural_ticks=0,
                  physics_steps=0, fit_calls=0, solver_calls=0, gpu_calls=0)
    publication_error = settle(out, result, ledger, raw, limits["output_bytes"], deadline)
    if primary is None:
        primary = publication_error
    if primary is not None:
        raise primary
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "freeze-sha256", "review", "review-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--timeout-s", type=float, default=10)
    args = parser.parse_args()
    run(args.freeze, args.freeze_sha256, args.review, args.review_sha256,
        args.output, args.timeout_s)


if __name__ == "__main__":
    main()
