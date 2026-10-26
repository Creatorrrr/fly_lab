"""E264: fixed-input, body-free LH tonic-current compensation diagnostic."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import analyze_c_banc_m3_fixed_input_command as e263
from tools import probe_c_banc_m3_lh_edge_restoration as e250
from tools import probe_c_banc_m3_lh_extensor_ablation as e260
from tools import probe_c_banc_m3_lh_neural_response as e248
from tools import replay_c_banc_m3_lh_full_inputs as e254

BASE = Path("verification/banc-natural-gait-m3-20260919")
FREEZE_SHA = "f20239005881b3a4e8686241089fcd8e068d01d0ad59a39634f3a0f9605ca82b"
FREEZE_SCHEMA = "flylab.m3.e264.lh-tonic-compensation.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e264.lh-tonic-compensation.review.v1"
OWNED = ("tools/probe_c_banc_m3_lh_tonic_compensation.py",
         "tests/test_c_banc_lh_tonic_compensation.py")
ORDER = ("X_U0", "Z_U0", "X_low", "X_high", "Z_low", "Z_high", "disk_Z_low")
STARTUP = 100
RESPONSE = 40
STEPS = 20
TARGET = 111970
MUSCLES = np.array([42, 43, 44], np.int64)
VALIDITY = "valid_fixed_input_tonic_compensation"


def pinned(name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Slash-separated frozen path required")
    relative = Path(name)
    path = (ROOT / relative).resolve()
    if (relative.is_absolute() or ".." in relative.parts
            or not path.is_relative_to(ROOT) or not path.is_file()):
        raise ValueError("Invalid frozen path: " + name)
    return path


def gate(freeze_path: Path, freeze_sha: str, review_path: Path,
         review_sha: str, out: Path) -> dict:
    """Check every pinned byte and package before original data or model load."""
    if (freeze_sha != FREEZE_SHA or e254.sha256(freeze_path) != FREEZE_SHA
            or e254.sha256(review_path) != review_sha):
        raise ValueError("E264 freeze/review SHA differs")
    freeze, review = e248.read_json(freeze_path), e248.read_json(review_path)
    limits = freeze.get("resource_limits", {})
    expected_limits = {"internal_s": 45, "finalization_reserve_s": 8,
                       "external_s": 50, "supervisor_wall_s": 55,
                       "cpu_threads": 1, "gpu_processes": 1,
                       "original_attempts": 1, "retries": 0,
                       "neural_constructors": 2, "setup_executed_ticks": 120,
                       "setup_capture_ticks": 40,
                       "scientific_neural_ticks": 8000,
                       "body_constructors": 0, "physics_steps": 0,
                       "fit_calls": 0, "solver_calls": 0,
                       "process_memory_bytes": 6442450944,
                       "additional_gpu_memory_bytes": 3221225472,
                       "minimum_free_gpu_bytes": 3221225472,
                       "output_bytes": 268435456,
                       "minimum_free_disk_bytes": 1073741824}
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("output") != (BASE / "e264-lh-tonic-compensation").as_posix()
            or freeze.get("network_identity") !=
            "4d5bae3386bed2dce2ca8a2add5130449c60a70e3b0d95a51947feeb03c76354"
            or freeze.get("n") != e254.N
            or freeze.get("target_row") != TARGET
            or freeze.get("bias32") != -0.19308185577392578
            or freeze.get("bias32_hex") != "40b745be"
            or freeze.get("arm_order") != list(ORDER)
            or freeze.get("startup_controls") != STARTUP
            or freeze.get("response_controls") != RESPONSE
            or freeze.get("steps_per_control") != STEPS
            or freeze.get("control_dt_s") != .005
            or freeze.get("neural_dt_s") != .00025
            or freeze.get("q19_muscles") != MUSCLES.tolist()
            or freeze.get("decoder_gain") != .02
            or freeze.get("hypothesis") != {
                "name": "H264_tonic_compensation", "command_residual_fraction_max": .2,
                "startup_q19_error_max_rad": .04, "response_R_min": .0001,
                "response_retention_fraction_min": .8,
                "mean_rate_distance_nonincrease": True,
                "target_motor_id": 47329, "target_motor_column": 107,
                "denominator_floor": 1e-12, "activation_atol": 1e-10}
            or any(limits.get(k) != v for k, v in expected_limits.items())):
        raise ValueError("E264 frozen science/resource contract differs")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("E264 approved source review required")
    pins: dict[str, str] = {}
    for mapping in (freeze["input_sha256"], freeze["reused_source_sha256"],
                    freeze.get("preservation_sha256", {}), review["source_sha256"],
                    review.get("additional_input_sha256", {}),
                    review.get("additional_reused_source_sha256", {}),
                    {freeze["design"]: freeze["design_sha256"]}):
        if not isinstance(mapping, dict):
            raise TypeError("E264 SHA mapping required")
        for name, digest in mapping.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E264 SHA pin: " + name)
            pins[name] = digest
    for name, digest in pins.items():
        if e254.sha256(pinned(name)) != digest:
            raise ValueError("E264 frozen byte differs: " + name)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("E264 package version differs: " + name)
    if out.resolve() != (ROOT / freeze["output"]).resolve():
        raise ValueError("E264 output path differs")
    if out.exists() or not out.parent.is_dir():
        raise FileExistsError("E264 output must be new")
    if shutil.disk_usage(out.parent).free < limits["minimum_free_disk_bytes"]:
        raise OSError("E264 minimum free disk unavailable")
    return freeze


def drive_bias(base: np.ndarray, bias: np.float32, target: int = TARGET) -> np.ndarray:
    """Add one fixed f32 current without changing any other drive byte."""
    value = np.asarray(base)
    shift = np.asarray(bias)
    if (value.shape != (e254.N,) or value.dtype != np.float32
            or not np.isfinite(value).all() or shift.shape != ()
            or shift.dtype != np.float32 or not np.isfinite(shift)
            or target != TARGET or value[target].view(np.uint32) != np.float32(0).view(np.uint32)):
        raise ValueError("E264 original zero target drive/f32 bias required")
    result = value.copy()
    result[target] = np.float32(result[target] + shift)
    if (result[target].tobytes() != shift.tobytes()
            or result[:target].tobytes() != value[:target].tobytes()
            or result[target + 1:].tobytes() != value[target + 1:].tobytes()):
        raise AssertionError("E264 drive changed outside target")
    return result


def response_rms(low: np.ndarray, high: np.ndarray, motor_ids: np.ndarray) -> float:
    a, b, ids = np.asarray(low), np.asarray(high), np.asarray(motor_ids)
    if (a.shape != b.shape or a.ndim != 2 or len(a) != RESPONSE + 1
            or a.dtype != np.float32 or b.dtype != np.float32
            or ids.shape != (56,) or ids.dtype != np.int64
            or np.any(ids < 0) or np.any(ids >= a.shape[1])
            or len(np.unique(ids)) != 56 or not np.isfinite(a).all()
            or not np.isfinite(b).all()):
        raise ValueError("E264 complete aligned 56-MN response required")
    difference = b[1:, ids].astype(np.float64) - a[1:, ids].astype(np.float64)
    return float(np.sqrt(np.mean((difference / 200.) ** 2)))


def decoder_step(bank: np.ndarray, rate: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Original 5ms full-bank recurrence and three original muscle hinges."""
    previous, values = np.asarray(bank), np.asarray(rate)
    tau, kappa, slopes = (np.asarray(params[k]) for k in ("tau", "kappa", "slopes"))
    pool, moment, rest = (np.asarray(params[k]) for k in ("pool", "moment", "rest"))
    if (previous.shape != (391, 3) or previous.dtype != np.float64
            or values.shape != (391,) or values.dtype != np.float32
            or tau.shape != (3,) or kappa.shape != (391,)
            or slopes.shape != (96, 391, 3, 2)
            or pool.shape != (96, 391) or moment.shape != (42, 96)
            or rest.shape != (42,) or not np.isfinite(previous).all()
            or not np.isfinite(values).all()):
        raise ValueError("E264 complete finite original decoder state required")
    alpha = -np.expm1(-.005 / tau)
    updated = previous + alpha[None, :] * (values.astype(np.float64)[:, None] - previous)
    activation = e263.transfer(updated, kappa, slopes, pool)
    q19 = float(rest[19] + .02 * np.dot(moment[19, MUSCLES], activation))
    if (not np.isfinite(updated).all() or not np.isfinite(activation).all()
            or not np.isfinite(q19) or np.any(updated < 0)
            or np.any(activation < -1e-12)):
        raise ValueError("E264 decoder state invalid")
    return updated, activation, q19


def assess_h264(q_b, q_x, q_z, baseline_error, rates_b, rates_x, rates_z,
                r_x: float, r_z: float) -> dict:
    """Apply only the frozen neural-only H264 conjunction."""
    b, x, z, error = (np.asarray(v, np.float64) for v in
                      (q_b, q_x, q_z, baseline_error))
    rb, rx, rz = (np.asarray(v, np.float64) for v in (rates_b, rates_x, rates_z))
    if (b.shape != (101,) or x.shape != b.shape or z.shape != b.shape
            or error.shape != (99,) or rb.shape != b.shape or rx.shape != b.shape
            or rz.shape != b.shape or not all(np.isfinite(v).all() for v in
                (b, x, z, error, rb, rx, rz)) or not np.isfinite([r_x, r_z]).all()):
        raise ValueError("E264 complete finite H264 arrays required")
    delta_x, delta_z = x[1:100] - b[1:100], z[1:100] - b[1:100]
    denominator = float(np.sqrt(np.mean(delta_x ** 2)))
    residual = float(np.sqrt(np.mean(delta_z ** 2)))
    virtual_error_max = float(np.max(np.abs(error + delta_z)))
    mean_b, mean_x, mean_z = (float(np.mean(rate[1:100])) for rate in (rb, rx, rz))
    distance_x, distance_z = abs(mean_x - mean_b), abs(mean_z - mean_b)
    z_range = float(np.ptp(rz[1:100]))
    supported = None if denominator <= 1e-12 else bool(
        residual <= .2 * denominator and virtual_error_max <= .04
        and r_z >= .0001 and r_z >= .8 * r_x and distance_z <= distance_x
        and z_range > 0)
    return {"command_RMS_X_minus_B_rad": denominator,
            "command_RMS_Z_minus_B_rad": residual,
            "command_residual_fraction": None if denominator <= 1e-12 else residual / denominator,
            "virtual_Z_error_max_abs_rad": virtual_error_max,
            "response_R_X": float(r_x), "response_R_Z": float(r_z),
            "mean_rate_B": mean_b, "mean_rate_X": mean_x, "mean_rate_Z": mean_z,
            "mean_rate_distance_X": distance_x, "mean_rate_distance_Z": distance_z,
            "Z_rate_range": z_range,
            "H264_tonic_compensation": supported,
            "classification": ("inconclusive" if supported is None else
                               "supported" if supported else "refuted"),
            "scope": "fixed saved-input neural-only; no body or candidate adoption"}


def intervention_configuration(model_identity: str, freeze: dict) -> dict:
    """Intervention provenance is separate from the unchanged model hash."""
    digest = hashlib.sha256((model_identity + "|" + str(TARGET)
        + "|" + freeze["bias32_hex"] + "|baseline-source6-mean-controls1:99").encode())
    return {"model_identity": model_identity,
            "configuration_identity": digest.hexdigest(),
            "target_row": TARGET, "bias64": freeze["bias64"],
            "bias32": freeze["bias32"], "bias32_hex": freeze["bias32_hex"],
            "rule": "negative f64 mean original-source6 added endpoint current, then f32 once",
            "controls": [1, 99], "model_parameters_changed": False}


def read_intervention_configuration(path: Path, freeze: dict,
                                    model_identity: str) -> np.float32:
    """Disk future restores persisted bias, never an unchecked memory copy."""
    saved = e248.read_json(path)
    expected = intervention_configuration(model_identity, freeze)
    if (saved != expected or type(saved.get("bias32")) is not float
            or type(saved.get("bias64")) is not float):
        raise ValueError("E264 persisted intervention configuration differs")
    bias = np.float32(saved["bias32"])
    if (bias.tobytes().hex() != saved["bias32_hex"]
            or bias.tobytes() != np.float32(saved["bias64"]).tobytes()):
        raise ValueError("E264 persisted float32 tonic current differs")
    return bias


def calibration_current(rate: np.ndarray, mask: np.ndarray,
                        difference: np.ndarray, initial_mask: np.ndarray,
                        pre_ids: np.ndarray) -> np.ndarray:
    """Frozen six source masks broadcast through 99 endpoint observations."""
    if (rate.shape != (99, 6) or mask.shape != (6,)
            or difference.shape != (6,) or pre_ids.shape != (6,)
            or pre_ids.dtype != np.int64 or len(np.unique(pre_ids)) != 6
            or any(value.dtype != np.float64 or not np.isfinite(value).all()
                   for value in (rate, mask, difference))
            or initial_mask.shape != (e254.N,) or initial_mask.dtype != np.float32
            or np.any(pre_ids < 0) or np.any(pre_ids >= e254.N)
            or not np.isin(mask, (0., 1.)).all()
            or not e254.same_bits(mask, initial_mask[pre_ids].astype(np.float64))):
        raise ValueError("E264 mask6 differs from original output-mask bytes")
    return difference[None, :] * rate * mask[None, :]


def observed_columns(roster: np.ndarray, requested: np.ndarray) -> np.ndarray:
    """Resolve source identities without assuming the stored cohort is sorted."""
    ids, selected = np.asarray(roster), np.asarray(requested)
    if (ids.ndim != 1 or selected.ndim != 1 or ids.dtype != np.int64
            or selected.dtype != np.int64 or len(np.unique(ids)) != len(ids)
            or len(np.unique(selected)) != len(selected)):
        raise ValueError("E264 unique integer observation identities required")
    positions = {int(node): column for column, node in enumerate(ids)}
    if any(int(node) not in positions for node in selected):
        raise ValueError("E264 source missing from stored observation roster")
    return np.array([positions[int(node)] for node in selected], np.int64)


def prepare(freeze: dict) -> dict:
    """Load pinned inputs only after the external source/hash gate succeeds."""
    upstream = e248.read_json(pinned(freeze["paths"]["upstream_e260_freeze"]))
    data = e260.prepare(upstream)
    from flylab.c.storage import StateStore

    if (upstream["graph_hash"] != freeze["graph_hash"]
            or data["arms"]["candidate"]["start"]["model_hash"] != freeze["network_identity"]
            or not e254.same_bits(data["csr"]["W6"].data,
                                  data["csr"]["W6"].data.copy())):
        raise ValueError("E264 actual W6 model provenance differs")
    selection = e248.read_json(pinned(freeze["paths"]["selection"]))
    for key in ("source_rows", "bridge_rows", "original24_rows", "lh_free_motor_rows"):
        if selection[key] != freeze["selection"][key]:
            raise ValueError("E264 frozen LH selection differs: " + key)
    if (np.float32(selection["low32"]).tobytes() !=
            np.float32(freeze["selection"]["low32"]).tobytes()
            or np.float32(selection["high32"]).tobytes() !=
            np.float32(freeze["selection"]["high32"]).tobytes()):
        raise ValueError("E264 frozen response levels differ")
    with np.load(pinned(freeze["paths"]["tonic_calibration"]), allow_pickle=False) as source:
        calibration = {key: source[key].copy() for key in source.files}
    derivation = e248.read_json(pinned(freeze["paths"]["tonic_derivation"]))
    ids = calibration["edge_ids"]
    rate, mask, difference = (calibration[key] for key in
                              ("source_rate64", "output_mask64", "weight_difference64"))
    if (ids.shape != (6,) or ids.dtype != np.int64
            or len(np.unique(ids)) != 6 or rate.shape != (99, 6)
            or mask.shape != (6,) or difference.shape != (6,)
            or not all(x.dtype == np.float64 and np.isfinite(x).all()
                       for x in (rate, mask, difference))
            or not np.isin(mask, (0., 1.)).all()
            or not np.array_equal(calibration["controls"], np.arange(1, 100))
            or not np.all(calibration["post_ids"] == TARGET)):
        raise ValueError("E264 complete 99x6 calibration required")
    base_csr, candidate_csr = data["csr"]["W0"], data["csr"]["W6"]
    for arm in ("baseline", "candidate"):
        calibration_current(rate, mask, difference,
                            data["arms"][arm]["start"]["output_mask"],
                            calibration["pre_ids"])
    selected = np.flatnonzero(base_csr.data.view(np.uint32) !=
                              candidate_csr.data.view(np.uint32))
    if (not np.array_equal(ids, selected)
            or not np.array_equal(calibration["pre_ids"], base_csr.indices[ids])
            or not e254.same_bits(difference,
                candidate_csr.data[ids].astype(np.float64) -
                base_csr.data[ids].astype(np.float64))):
        raise ValueError("E264 exact original six-edge calibration differs")
    baseline = data["arms"]["baseline"]["reference"]
    observed_at = observed_columns(baseline["observed_ids"], calibration["pre_ids"])
    if (np.any(observed_at >= len(baseline["observed_ids"]))
            or not np.array_equal(baseline["observed_ids"][observed_at],
                                  calibration["pre_ids"])
            or not e254.same_bits(rate,
                baseline["observed-rate.npy"][1:100, observed_at].astype(np.float64))):
        raise ValueError("E264 calibration baseline source firing differs")
    measured = calibration_current(
        rate, mask, difference, data["arms"]["baseline"]["start"]["output_mask"],
        calibration["pre_ids"])
    summed = np.sum(measured, axis=1, dtype=np.float64)
    bias64 = -float(np.mean(summed))
    bias32 = np.float32(bias64)
    if (not e254.same_bits(measured, calibration["added_current_by_edge64"])
            or not e254.same_bits(summed, calibration["added_current64"])
            or not e254.same_bits(np.array(bias64), calibration["bias64"])
            or not e254.same_bits(np.array(bias32), calibration["bias32"])
            or bias64 != freeze["bias64"]
            or bias32.tobytes() != np.float32(freeze["bias32"]).tobytes()
            or derivation.get("bias32_hex") != freeze["bias32_hex"]):
        raise ValueError("E264 single frozen mean-current rule differs")
    source_row = data["arms"]["candidate"]["start"]["drive"][TARGET]
    if np.float32(source_row).tobytes() != np.float32(0).tobytes():
        raise ValueError("E264 original bridge drive must be +0")
    with np.load(pinned(freeze["paths"]["e263_output"] + "/raw.npz"),
                 allow_pickle=False) as raw:
        e263_reference = {key: raw[key].copy() for key in
                          ("B_q19", "X_q19", "X_activation", "baseline_error")}
    with np.load(pinned(freeze["paths"]["baseline_native"]),
                 allow_pickle=False) as native:
        baseline_bank = native["bank"][0].copy()
    with np.load(pinned("verification/banc-natural-gait-m3-20260919/"
                        "e150-native-monotone-body/assembled-calibration.npz"),
                 allow_pickle=False) as original:
        anatomy = {key: original[key].copy() for key in ("moment", "rest", "pool")}
    adapter = data["arms"]["candidate"]["whole"]["adapter"]
    if (not e254.same_bits(baseline_bank, adapter["bank"])
            or np.flatnonzero(anatomy["moment"][19]).tolist() != MUSCLES.tolist()):
        raise ValueError("E264 original decoder start/anatomy differs")
    decoder = {**anatomy,
               **{key: adapter["parameters"][key] for key in
                  ("tau", "kappa", "slopes")},
               "bank0": baseline_bank}
    e260_folder = ROOT / freeze["paths"]["e260_output"] / "X"
    e260_ref = {name: np.load(e260_folder / name, allow_pickle=False)
                for name in ("motor-rate.npy", "observed-rate.npy", "observed-drive.npy",
                             "observed-adaptation.npy", "core-rate.npy", "core-drive.npy",
                             "core-adaptation.npy", "bank.npy", "activation44.npy")}
    e260_ref["start"] = StateStore.load(e260_folder / "observed-start")
    e260_ref["final"] = StateStore.load(e260_folder / "final")
    data.update(calibration=calibration, bias32=bias32, decoder=decoder,
                e263_reference=e263_reference, e260_ref=e260_ref,
                frozen_selection=selection)
    return data


def raw_arrays(folder: Path, rows: int, union_count: int, drive_count: int) -> dict:
    specs = {"endpoint-rate.npy": ((rows, union_count), np.float32),
             "endpoint-drive.npy": ((rows, union_count), np.float32),
             "endpoint-adaptation.npy": ((rows, union_count), np.float32),
             "motor-rate.npy": ((rows, 391), np.float32),
             "observed-rate.npy": ((rows, 82), np.float32),
             "observed-drive.npy": ((rows, 82), np.float32),
             "observed-adaptation.npy": ((rows, 82), np.float32),
             "core-rate.npy": ((rows, 24), np.float32),
             "core-drive.npy": ((rows, 24), np.float32),
             "core-adaptation.npy": ((rows, 24), np.float32),
             "drive-values.npy": ((rows - 1, drive_count), np.float32),
             "bridge-drive.npy": ((rows,), np.float32),
             "bank.npy": ((rows, 391, 3), np.float64),
             "activation-three.npy": ((rows, 3), np.float64),
             "q19.npy": ((rows,), np.float64),
             "drive-sha256.npy": ((rows,), "S64")}
    result = {}
    for name, (shape, dtype) in specs.items():
        value = np.lib.format.open_memmap(folder / name, mode="w+",
                                          shape=shape, dtype=dtype)
        value[:] = b"" if name == "drive-sha256.npy" else np.nan
        value.flush()
        result[name] = value
    return result


def observe_endpoint(network, row: int, raw: dict, entry: dict,
                     context: dict) -> None:
    """Read original selected tensors; perform no current probe or advance."""
    import torch

    from tools.verify_c_banc_observation import exact

    before = network.snapshot() if row in (0, len(raw["motor-rate.npy"]) - 1) else None
    union = context["union_ids"]
    device_ids = context["device_union_ids"]
    rate = network.readout(union)
    raw["endpoint-rate.npy"][row] = rate
    for field, tensor in (("endpoint-drive.npy", network.drive),
                          ("endpoint-adaptation.npy", network.adaptation)):
        with torch.inference_mode():
            raw[field][row] = tensor.index_select(0, device_ids).detach().cpu().numpy()
        raw[field].flush()
    raw["endpoint-rate.npy"].flush()
    for cohort in ("motor", "observed", "core"):
        positions = context[cohort + "_at"]
        fields = ("rate",) if cohort == "motor" else ("rate", "drive", "adaptation")
        for field in fields:
            destination = "motor-rate.npy" if cohort == "motor" else cohort + "-" + field + ".npy"
            raw[destination][row] = raw["endpoint-" + field + ".npy"][row, positions]
            raw[destination].flush()
    if row:
        bank, activation, q19 = decoder_step(raw["bank.npy"][row - 1],
                                               raw["motor-rate.npy"][row],
                                               context["decoder"])
        raw["bank.npy"][row] = bank
        raw["activation-three.npy"][row] = activation
        raw["q19.npy"][row] = q19
    raw["bank.npy"].flush()
    raw["activation-three.npy"].flush()
    raw["q19.npy"].flush()
    drive = network.drive.detach().cpu().numpy().copy()
    if drive.shape != (e254.N,) or drive.dtype != np.float32 or not np.isfinite(drive).all():
        raise ValueError("E264 actual full f32 drive invalid")
    raw["bridge-drive.npy"][row] = drive[TARGET]
    raw["drive-sha256.npy"][row] = hashlib.sha256(drive.tobytes()).hexdigest().encode()
    raw["bridge-drive.npy"].flush()
    raw["drive-sha256.npy"].flush()
    entry["observed_endpoints"] = row + 1
    if before is not None:
        exact(network.snapshot(), before, "E264 read-only endpoint observation")


def context_for(network, data: dict) -> dict:
    import torch

    union = data["incoming"]["W6"]["union_ids"]
    motor = data["config"]["motor_ids"]
    observed = data["arms"]["baseline"]["reference"]["observed_ids"]
    core = data["core_ids"]
    positions = {name: np.searchsorted(union, ids)
                 for name, ids in (("motor", motor), ("observed", observed),
                                   ("core", core))}
    if (observed.shape != (82,) or core.shape != (24,)
            or any(np.any(at >= len(union)) or not np.array_equal(union[at], ids)
                   for name, ids, at in (("motor", motor, positions["motor"]),
                                         ("observed", observed, positions["observed"]),
                                         ("core", core, positions["core"])))):
        raise ValueError("E264 original motor/82/core24 union differs")
    return {"union_ids": union,
            "device_union_ids": torch.as_tensor(union, dtype=torch.int64,
                                                  device=network.device),
            **{name + "_at": at for name, at in positions.items()},
            "decoder": data["decoder"]}


def x_diagonal(row: int, raw: dict, reference: dict, e263_reference: dict) -> None:
    for key in ("motor-rate.npy", "observed-rate.npy", "observed-drive.npy",
                "observed-adaptation.npy", "core-rate.npy", "core-drive.npy",
                "core-adaptation.npy"):
        if not e254.same_bits(raw[key][row], reference[key][row]):
            raise AssertionError(f"E264 X original E260 {key} differs at {row}")
    if (not e254.same_bits(raw["bank.npy"][row, [68, 107]],
                           reference["bank.npy"][row])
            or not np.isclose(raw["activation-three.npy"][row, 2],
                              reference["activation44.npy"][row], atol=1e-10, rtol=0)
            or not np.isclose(raw["q19.npy"][row],
                              e263_reference["X_q19"][row], atol=1e-10, rtol=0)):
        raise AssertionError(f"E264 X original E260/E263 decoder differs at {row}")


def begin_arm(ledger: dict, name: str, controls: int) -> dict:
    entry = {"name": name, "requested_controls": controls,
             "entered_controls": 0, "completed_controls": 0,
             "unknown_neural_partial": False, "reset_requested": True,
             "reset_exact": False, "observed_endpoints": 0,
             "initial_saved": False, "final_saved": False,
             "status": "requested"}
    ledger["arms"][name] = entry
    return entry


def run_arm(network, saved: dict, bank0: np.ndarray, arm_data: dict,
            raw: dict, entry: dict, context: dict, data: dict, bias: np.float32,
            persist, budget, *, response_value: np.float32 | None = None,
            diagonal: bool = False) -> dict:
    """One restored original-input or held-input arm, with durable tick intent."""
    from tools.verify_c_banc_observation import exact

    rows = len(raw["motor-rate.npy"])
    controls = rows - 1
    network.restore(saved)
    exact(network.snapshot(), saved, "E264 arm restored whole neural start")
    entry["reset_exact"] = True
    persist()
    if (bank0.shape != (391, 3) or bank0.dtype != np.float64
            or not np.isfinite(bank0).all()):
        raise ValueError("E264 exact full decoder fork bank required")
    raw["bank.npy"][0] = bank0
    raw["activation-three.npy"][0] = e263.transfer(
        bank0, context["decoder"]["kappa"], context["decoder"]["slopes"],
        context["decoder"]["pool"])
    raw["q19.npy"][0] = (context["decoder"]["rest"][19]
                          + .02 * np.dot(context["decoder"]["moment"][19, MUSCLES],
                                         raw["activation-three.npy"][0]))
    for name in ("bank.npy", "activation-three.npy", "q19.npy"):
        raw[name].flush()
    observe_endpoint(network, 0, raw, entry, context)
    persist()
    if diagonal:
        x_diagonal(0, raw, data["e260_ref"], data["e263_reference"])
    source = arm_data["sensory"]["source_ids"]
    frozen_drive = None
    if response_value is not None:
        frozen_drive = e248.held_drive(saved["drive"], source, response_value)
        if np.float32(frozen_drive[TARGET]).tobytes() != np.float32(bias).tobytes():
            raise ValueError("E264 held response changed bridge tonic current")
    for control in range(1, controls + 1):
        budget()
        if frozen_drive is None:
            sensory, original_drive = e254.reconstruct_drive(
                arm_data["sensory"]["old_filtered"][control - 1],
                arm_data["sensory_ids"], arm_data["sensory_ports"], e254.N,
                data["descending_ids"], 0., data["navigation_ports"], 5000.)
            if not e254.same_bits(
                sensory[source],
                arm_data["sensory"]["returned_source_drive"][control - 1]):
                raise AssertionError(f"E264 source drive differs at {control}")
            drive = (drive_bias(original_drive, bias) if bias else original_drive)
            if (not e254.same_bits(drive[:TARGET], original_drive[:TARGET])
                    or not e254.same_bits(drive[TARGET + 1:],
                                          original_drive[TARGET + 1:])):
                raise AssertionError("E264 nonbridge drive changed")
        else:
            drive = frozen_drive
        raw["drive-values.npy"][control - 1] = drive[data["drive_ids"]]
        raw["drive-values.npy"].flush()
        entry["entered_controls"] = control
        entry["unknown_neural_partial"] = True
        persist()
        network.advance(drive, STEPS)
        entry["completed_controls"] = control
        entry["unknown_neural_partial"] = False
        persist()
        observe_endpoint(network, control, raw, entry, context)
        persist()
        if (not e254.same_bits(network.drive.detach().cpu().numpy(), drive)
                or raw["bridge-drive.npy"][control].tobytes() !=
                np.float32(bias).tobytes()):
            raise AssertionError("E264 actual drive does not match fixed intervention")
        if diagonal:
            x_diagonal(control, raw, data["e260_ref"], data["e263_reference"])
    final = network.snapshot()
    if (final["tick"] != saved["tick"] + controls * STEPS
            or not e254.same_bits(final["drive"], drive)
            or not e254.same_bits(final["output_mask"], saved["output_mask"])):
        raise AssertionError("E264 final neural clock/drive/mask differs")
    entry["status"] = "science_complete"
    return final


def arm_rate_stats(rate: np.ndarray, motor_column: int = 107) -> dict:
    values = np.asarray(rate)
    if (values.shape != (101, 391) or values.dtype != np.float32
            or not np.isfinite(values).all()):
        raise ValueError("E264 complete original motor rate required")
    selected = values[1:100, motor_column].astype(np.float64)
    return {"mean": float(np.mean(selected)), "min": float(np.min(selected)),
            "max": float(np.max(selected)), "range": float(np.ptp(selected)),
            "exact_zero_fraction": float(np.mean(selected == 0.)),
            "exact_cap200_fraction": float(np.mean(selected == 200.))}


def factors(raws: dict, data: dict, freeze: dict) -> tuple[dict, dict]:
    """Keep full numerical factors separately from the conditional verdict."""
    from tools.verify_c_banc_observation import exact

    prior = data["e263_reference"]
    b, x, z = (prior["B_q19"], np.asarray(raws["X_U0"]["q19.npy"]),
               np.asarray(raws["Z_U0"]["q19.npy"]))
    if not np.allclose(x, prior["X_q19"], atol=1e-10, rtol=0):
        raise AssertionError("E264 exact E263 X q19 reconstruction differs")
    motor_ids = data["config"]["motor_ids"]
    lh = np.asarray(freeze["selection"]["lh_free_motor_rows"], np.int64)
    columns = np.searchsorted(motor_ids, lh)
    if (columns.shape != (56,) or np.any(columns >= len(motor_ids))
            or not np.array_equal(motor_ids[columns], lh)):
        raise ValueError("E264 fixed LH motor membership differs")
    r_x = response_rms(raws["X_low"]["motor-rate.npy"],
                       raws["X_high"]["motor-rate.npy"], columns)
    r_z = response_rms(raws["Z_low"]["motor-rate.npy"],
                       raws["Z_high"]["motor-rate.npy"], columns)
    base_rate = data["arms"]["baseline"]["reference"]["motor-rate.npy"][:101]
    x_rate = np.asarray(raws["X_U0"]["motor-rate.npy"])
    z_rate = np.asarray(raws["Z_U0"]["motor-rate.npy"])
    report = assess_h264(b, x, z, prior["baseline_error"],
                         base_rate[:, 107], x_rate[:, 107], z_rate[:, 107],
                         r_x, r_z)
    report.update(rate_diagnostics={
        "B": arm_rate_stats(base_rate), "X": arm_rate_stats(x_rate),
        "Z": arm_rate_stats(z_rate)},
        core24_RMS_X_minus_B=float(np.sqrt(np.mean((
            raws["X_U0"]["core-rate.npy"][1:100].astype(np.float64)
            - data["arms"]["baseline"]["reference"]["core-rate.npy"][1:100]
            .astype(np.float64)) ** 2))),
        core24_RMS_Z_minus_B=float(np.sqrt(np.mean((
            raws["Z_U0"]["core-rate.npy"][1:100].astype(np.float64)
            - data["arms"]["baseline"]["reference"]["core-rate.npy"][1:100]
            .astype(np.float64)) ** 2))))
    arrays = {"control_tick": np.arange(101, dtype=np.int64),
              "q19_B": np.asarray(b, np.float64).copy(),
              "q19_X": np.asarray(x, np.float64).copy(),
              "q19_Z": np.asarray(z, np.float64).copy(),
              "q19_X_minus_B": np.asarray(x - b, np.float64),
              "q19_Z_minus_B": np.asarray(z - b, np.float64),
              "baseline_error": prior["baseline_error"].copy(),
              "virtual_Z_error": prior["baseline_error"] + (z - b)[1:100],
              "lh_motor_ids": lh, "lh_motor_columns": columns}
    for arm in ("X_U0", "Z_U0"):
        for name in ("activation-three.npy", "bank.npy", "motor-rate.npy",
                     "core-rate.npy", "core-drive.npy", "core-adaptation.npy"):
            arrays[arm + "__" + name.removesuffix(".npy")] = np.asarray(raws[arm][name]).copy()
    for arm in ("X_low", "X_high", "Z_low", "Z_high"):
        arrays[arm + "__lh_rate"] = np.asarray(raws[arm]["motor-rate.npy"][:, columns]).copy()
    arrays["response_X_high_minus_low"] = (
        arrays["X_high__lh_rate"].astype(np.float64)
        - arrays["X_low__lh_rate"].astype(np.float64))
    arrays["response_Z_high_minus_low"] = (
        arrays["Z_high__lh_rate"].astype(np.float64)
        - arrays["Z_low__lh_rate"].astype(np.float64))
    exact(np.asarray(raws["disk_Z_low"]["motor-rate.npy"]),
          np.asarray(raws["Z_low"]["motor-rate.npy"]),
          "E264 independently restored Z low full MN future")
    return arrays, {"schema": "flylab.m3.e264.factor-windows.v1",
                    "H264": report, "initial_row": "decoder hypothetical; held body command excluded",
                    "interpretation_status": "complete_neural_only"}


def output_manifest(out: Path, maximum: int) -> dict:
    files, size = {}, 0
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name in ("result.json", "output-manifest.json"):
            continue
        count = path.stat().st_size
        size += count
        if size > maximum:
            raise OSError("E264 output byte cap exceeded")
        files[path.relative_to(out).as_posix()] = {
            "sha256": e254.sha256(path), "bytes": count}
    return {"schema": "flylab.m3.e264.output-manifest.v1",
            "bytes": size, "files": files}


def revoke_report(out: Path, persist=e248.write_json) -> list[dict]:
    path = out / "factor-windows.json"
    if not path.exists():
        return []
    secondary = []
    try:
        report = e248.read_json(path)
        report["H264"]["H264_tonic_compensation"] = None
        report["H264"]["classification"] = "not_tested"
        report["interpretation_status"] = "not_tested"
        persist(path, report)
        return secondary
    except Exception as error:  # noqa: BLE001
        secondary.append({"stage": "report revocation", "error": repr(error)})
    try:
        path.replace(out / "factor-windows-unvalidated-raw.json")
    except Exception as error:  # noqa: BLE001
        secondary.append({"stage": "report fallback rename", "error": repr(error)})
        try:
            path.unlink(missing_ok=True)
        except Exception as removal:  # noqa: BLE001
            secondary.append({"stage": "stale report removal", "error": repr(removal)})
    return secondary


def finish_receipts(out: Path, freeze: dict, ledger: dict, result: dict,
                    errors: list, primary, began: float,
                    *, persist=e248.write_json):
    """Make all success interpretations conditional on completed publication."""
    limit = freeze["resource_limits"]["internal_s"]

    def fail(error, stage):
        nonlocal primary
        if primary is None:
            primary = error
        errors.append({"stage": stage, "type": type(error).__name__,
                       "message": str(error)})
        result.update(validity="resource_limited" if isinstance(primary, TimeoutError)
                      else "invalid_or_incomplete", H264_tonic_compensation=None,
                      H264_classification="not_tested",
                      factor_report_available=False, finalization_pending=True,
                      first_error={"type": type(primary).__name__,
                                   "message": str(primary)})
        if isinstance(result.get("H264"), dict):
            result["H264"]["H264_tonic_compensation"] = None
            result["H264"]["classification"] = "not_tested"
        errors.extend(revoke_report(out, persist=persist))

    if primary is not None:
        fail(primary, "prior science/cleanup")
    result["work"] = {
        "constructors_requested": sum(x["requested"] for x in ledger["models"].values()),
        "constructors_entered": sum(x["entered"] for x in ledger["models"].values()),
        "constructors_completed": sum(x["completed"] for x in ledger["models"].values()),
        "setup_executed_ticks_known": sum(x["setup_executed_ticks"] or 0
                                          for x in ledger["models"].values()),
        "setup_capture_ticks_known": sum(x["setup_capture_ticks"] or 0
                                         for x in ledger["models"].values()),
        "scientific_ticks_completed": STEPS * sum(x["completed_controls"]
                                                    for x in ledger["arms"].values()),
        "current_probe_calls_completed": 0,
        "unknown_partial_models": [name for name, x in ledger["models"].items()
                                   if x["setup_unknown_partial"]],
        "unknown_partial_arms": [name for name, x in ledger["arms"].items()
                                 if x["unknown_neural_partial"]]}
    for name, value in (("advance-ledger.json", ledger),
                        ("errors.json", {"errors": errors})):
        try:
            persist(out / name, value)
        except Exception as error:  # noqa: BLE001
            fail(error, name)
    if time.perf_counter() - began >= limit:
        fail(TimeoutError("E264 finalization deadline"), "deadline")
    try:
        persist(out / "errors.json", {"errors": errors})
        persist(out / "output-manifest.json",
                output_manifest(out, freeze["resource_limits"]["output_bytes"]))
        result["elapsed_s"] = time.perf_counter() - began
        if result["elapsed_s"] >= limit:
            raise TimeoutError("E264 publication deadline")
        result["finalization_pending"] = False
        persist(out / "result.json", result)
        if time.perf_counter() - began >= limit:
            raise TimeoutError("E264 late result deadline")
    except Exception as error:  # noqa: BLE001
        fail(error, "publication")
        for name in ("result.json", "output-manifest.json"):
            try:
                (out / name).unlink(missing_ok=True)
            except Exception as secondary:  # noqa: BLE001
                fail(secondary, "stale " + name + " removal")
        for name, value in (("advance-ledger.json", ledger),
                            ("result.json", result),
                            ("errors.json", {"errors": errors})):
            try:
                persist(out / name, value)
            except Exception as secondary:  # noqa: BLE001
                fail(secondary, "retry " + name)
                (out / name).unlink(missing_ok=True)
    return primary


def execute(freeze: dict, out: Path) -> None:
    from flylab.c.storage import StateStore
    from tools.verify_c_banc_observation import exact

    began = time.perf_counter()
    out.mkdir(parents=False, exist_ok=False)
    ledger = {"schema": "flylab.m3.e264.ledger.v1", "status": "entered",
              "models": {}, "arms": {}, "resources": [],
              "original_attempts": 1, "retries": 0,
              "body_constructors": 0, "physics_steps": 0,
              "fit_calls": 0, "solver_calls": 0, "current_probe_calls": 0}
    result = {"schema": "flylab.m3.e264.result.v1", "validity": "incomplete",
              "H264_tonic_compensation": None, "H264_classification": "not_tested",
              "factor_report_available": False,
              "candidate_adopted": False, "whole_goal_pass": False,
              "F6_exposed": False, "active_UI_changed": False,
              "finalization_pending": True,
              "interpretation": "fixed-input neural-only; no body or biology validation"}
    errors, primary = [], None
    network, active_dir, active_entry = None, None, None
    raw_by_arm: dict[str, dict] = {}
    limiter, old_torch_threads, old_environment = None, None, {}
    try:
        import torch
        from threadpoolctl import threadpool_info, threadpool_limits

        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            old_environment[name] = os.environ.get(name)
            os.environ[name] = "1"
        old_torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        limiter = threadpool_limits(limits=1)
        if (torch.get_num_threads() != 1
                or any(item["num_threads"] != 1 for item in threadpool_info())):
            raise RuntimeError("E264 effective CPU thread count differs")
        e248.write_json(out / "threading.json", {"torch_threads": 1,
                         "threadpools": threadpool_info()})
        if torch.cuda.mem_get_info()[0] < freeze["resource_limits"]["minimum_free_gpu_bytes"]:
            raise MemoryError("E264 minimum free GPU memory unavailable")
        torch.cuda.reset_peak_memory_stats()
        data = prepare(freeze)
        np.savez(out / "calibration.npz", **data["calibration"])
        data["incoming"] = {"W6": data["incoming"]["W6"]}
        np.savez(out / "incoming-W6.npz", **data["incoming"]["W6"])
        np.savez(out / "drive-mapping.npz",
                 sensory_ids=data["arms"]["baseline"]["sensory_ids"],
                 sensory_ports=data["arms"]["baseline"]["sensory_ports"],
                 source_ids=data["arms"]["baseline"]["sensory"]["source_ids"],
                 navigation_ports=data["navigation_ports"],
                 descending_ids=data["descending_ids"], drive_ids=data["drive_ids"])
        np.savez(out / "decoder-parameters.npz", **data["decoder"],
                 muscle_ids=MUSCLES)
        e248.write_json(out / "csr-fingerprint.json", {
            field: hashlib.sha256(np.ascontiguousarray(
                getattr(data["csr"]["W6"], field)).tobytes()).hexdigest()
            for field in ("indptr", "indices", "data")})

        def persist():
            e248.write_json(out / "advance-ledger.json", ledger)

        def budget():
            e260.resource_check(freeze, began, out, ledger)

        candidate = data["arms"]["candidate"]["start"]
        model_entry = {"requested": True, "entered": False, "completed": False,
                       "setup_unknown_partial": False,
                       "setup_executed_ticks": None, "setup_capture_ticks": None,
                       "status": "requested"}
        ledger["models"]["main_W6"] = model_entry
        persist()
        budget()
        model_entry.update(entered=True, setup_unknown_partial=True,
                           status="constructing")
        persist()
        network, restored, identity_fields, backend = e250._constructor(
            data["csr"]["W6"], data["tau"], data["config"], candidate,
            expected_identity=freeze["network_identity"], restore_saved=candidate)
        if identity_fields or network.identity != freeze["network_identity"]:
            raise AssertionError("E264 unchanged W6 model identity required")
        exact(restored, candidate, "E264 original candidate W6 start")
        model_entry.update(completed=True, setup_unknown_partial=False,
                           setup_executed_ticks=60, setup_capture_ticks=20,
                           status="ready", backend=backend,
                           model_hash=network.identity)
        persist()
        context = context_for(network, data)
        parameter_receipt = {"target_id": np.asarray(TARGET, np.int64)}
        for name, tensor in (("threshold", network.threshold),
                             ("coefficient", network.coefficient),
                             ("cap", network.cap), ("membrane_tau", network.tau),
                             ("cell_gain", network._cell_gain),
                             ("cell_adaptation_tau", network._cell_adaptation_tau),
                             ("inverse_adaptation_tau", network._cell_inverse_tau)):
            parameter_receipt[name] = tensor[TARGET].detach().cpu().numpy().copy()
        np.savez(out / "target-parameters.npz", **parameter_receipt)
        e248.write_json(out / "intervention-configuration.json",
                        intervention_configuration(network.identity, freeze))

        fork: dict[str, dict] = {}
        fork_bank: dict[str, np.ndarray] = {}
        for arm in ORDER[:6]:
            controls = STARTUP if arm.endswith("_U0") else RESPONSE
            folder = out / arm
            folder.mkdir(exist_ok=False)
            entry = begin_arm(ledger, arm, controls)
            active_dir, active_entry = folder, entry
            persist()
            raw = raw_arrays(folder, controls + 1,
                             len(context["union_ids"]), len(data["drive_ids"]))
            raw_by_arm[arm] = raw
            if arm == "X_U0":
                saved, bank0, bias = candidate, data["decoder"]["bank0"], np.float32(0)
                source = data["arms"]["baseline"]
                response = None
            elif arm == "Z_U0":
                saved, bank0, bias = candidate, data["decoder"]["bank0"], data["bias32"]
                source = data["arms"]["baseline"]
                response = None
            else:
                name, level = arm.split("_", 1)
                saved, bank0 = fork[name], fork_bank[name]
                bias = np.float32(0) if name == "X" else data["bias32"]
                source = data["arms"]["baseline"]
                response = np.float32(freeze["selection"][level + "32"])
            network.restore(saved)
            exact(network.snapshot(), saved, "E264 pre-arm whole restoration")
            StateStore.save(folder / "observed-start", network.snapshot())
            entry["initial_saved"] = True
            persist()
            final = run_arm(network, saved, bank0, source, raw, entry, context,
                            data, bias, persist, budget, response_value=response,
                            diagonal=(arm == "X_U0"))
            StateStore.save(folder / "final", final)
            entry["final_saved"] = True
            persist()
            if arm == "X_U0":
                exact(final, data["e260_ref"]["final"], "E264 X full E260 future")
            if arm in ("X_U0", "Z_U0"):
                name = arm[0]
                fork[name] = final
                fork_bank[name] = np.asarray(raw["bank.npy"][-1]).copy()
                StateStore.save(out / (name + "-fork"), final)
                np.save(out / (name + "-fork-bank.npy"), fork_bank[name],
                        allow_pickle=False)
            entry["status"] = "complete_exact_X" if arm == "X_U0" else "science_complete"
            persist()
            e254.flush_raw(raw)
            active_dir, active_entry = None, None
        network = None
        gc.collect()
        torch.cuda.empty_cache()

        disk_fork = StateStore.load(out / "Z-fork")
        exact(disk_fork, fork["Z"], "E264 saved Z fork exact")
        disk_bank = np.load(out / "Z-fork-bank.npy", allow_pickle=False)
        if (not e254.same_bits(disk_bank, fork_bank["Z"])
                or not e254.same_bits(disk_bank, raw_by_arm["Z_U0"]["bank.npy"][-1])):
            raise AssertionError("E264 persisted Z decoder fork bank differs")
        disk_bias = read_intervention_configuration(
            out / "intervention-configuration.json", freeze,
            freeze["network_identity"])
        if (not e254.same_bits(np.asarray(disk_bias), np.asarray(data["bias32"]))
                or disk_fork["drive"][TARGET].tobytes() != disk_bias.tobytes()):
            raise AssertionError("E264 persisted Z fork current differs")
        ledger["disk_configuration"] = {
            "loaded_exact": True,
            "sha256": e254.sha256(out / "intervention-configuration.json"),
            "configuration_identity": intervention_configuration(
                freeze["network_identity"], freeze)["configuration_identity"],
            "bias32_hex": disk_bias.tobytes().hex(),
            "fork_drive_exact": True, "fork_bank_exact": True}
        persist()
        model_entry = {"requested": True, "entered": False, "completed": False,
                       "setup_unknown_partial": False,
                       "setup_executed_ticks": None, "setup_capture_ticks": None,
                       "status": "requested"}
        ledger["models"]["disk_W6"] = model_entry
        persist()
        budget()
        model_entry.update(entered=True, setup_unknown_partial=True,
                           status="constructing")
        persist()
        network, restored, identity_fields, backend = e250._constructor(
            data["csr"]["W6"], data["tau"], data["config"], disk_fork,
            expected_identity=freeze["network_identity"], restore_saved=disk_fork)
        if identity_fields:
            raise AssertionError("E264 disk model identity transplant forbidden")
        exact(restored, disk_fork, "E264 disk Z fork restored exact")
        model_entry.update(completed=True, setup_unknown_partial=False,
                           setup_executed_ticks=60, setup_capture_ticks=20,
                           status="ready", backend=backend,
                           model_hash=network.identity)
        persist()
        context = context_for(network, data)
        folder = out / "disk_Z_low"
        folder.mkdir(exist_ok=False)
        entry = begin_arm(ledger, "disk_Z_low", RESPONSE)
        active_dir, active_entry = folder, entry
        persist()
        raw = raw_arrays(folder, RESPONSE + 1, len(context["union_ids"]),
                         len(data["drive_ids"]))
        raw_by_arm["disk_Z_low"] = raw
        StateStore.save(folder / "observed-start", network.snapshot())
        entry["initial_saved"] = True
        persist()
        final = run_arm(network, disk_fork, disk_bank, data["arms"]["baseline"],
                        raw, entry, context, data, disk_bias, persist, budget,
                        response_value=np.float32(freeze["selection"]["low32"]))
        StateStore.save(folder / "final", final)
        entry["final_saved"] = True
        e254.flush_raw(raw)
        for field in ("motor-rate.npy", "endpoint-rate.npy", "endpoint-drive.npy",
                      "endpoint-adaptation.npy", "bank.npy", "activation-three.npy",
                      "q19.npy", "bridge-drive.npy", "drive-sha256.npy"):
            if not e254.same_bits(raw[field], raw_by_arm["Z_low"][field]):
                raise AssertionError("E264 disk Z low future differs: " + field)
        exact(final, StateStore.load(out / "Z_low" / "final"),
              "E264 independently restored complete Z low final")
        entry["status"] = "complete_disk_exact"
        persist()
        active_dir, active_entry = None, None
        if (sum(x["setup_executed_ticks"] for x in ledger["models"].values()) != 120
                or sum(x["setup_capture_ticks"] for x in ledger["models"].values()) != 40
                or sum(x["completed_controls"] for x in ledger["arms"].values()) != 400
                or any(ledger["arms"][arm]["observed_endpoints"] !=
                       ledger["arms"][arm]["requested_controls"] + 1 for arm in ORDER)):
            raise AssertionError("E264 exact science/setup/endpoint accounting differs")
        arrays, report = factors(raw_by_arm, data, freeze)
        np.savez(out / "factor-effects.npz", **arrays)
        e248.write_json(out / "factor-windows.json", report)
        result.update(validity=VALIDITY, H264_tonic_compensation=report["H264"][
                          "H264_tonic_compensation"], H264=report["H264"],
                      H264_classification=report["H264"]["classification"],
                      factor_report_available=True, X_exact=True,
                      disk_Z_low_exact=True)
    except Exception as error:  # noqa: BLE001 - retain first science error and raw
        primary = error
        errors.append({"stage": "science", "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
        if (network is not None and active_dir is not None and active_entry is not None
                and not (active_dir / "final").exists()):
            primary = e254.preserve_partial_state(network, active_dir / "partial-final",
                                                   active_entry, primary)
    finally:
        for raw in raw_by_arm.values():
            try:
                e254.flush_raw(raw)
            except Exception as error:  # noqa: BLE001
                if primary is None:
                    primary = error
                errors.append({"stage": "raw flush", "error": repr(error)})
        raw_by_arm.clear()
        network = None
        try:
            gc.collect()
        except Exception as error:  # noqa: BLE001
            if primary is None:
                primary = error
            errors.append({"stage": "garbage collection", "error": repr(error)})
        if limiter is not None:
            try:
                limiter.restore_original_limits()
            except Exception as error:  # noqa: BLE001
                if primary is None:
                    primary = error
                errors.append({"stage": "threadpool restore", "error": repr(error)})
        if old_torch_threads is not None:
            try:
                import torch
                torch.set_num_threads(old_torch_threads)
            except Exception as error:  # noqa: BLE001
                if primary is None:
                    primary = error
                errors.append({"stage": "torch thread restore", "error": repr(error)})
        for name, old in old_environment.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
        primary = finish_receipts(out, freeze, ledger, result, errors, primary, began)
        if primary is not None:
            raise RuntimeError("E264 invalid: " + str(primary)) from primary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    freeze = gate(args.freeze, args.freeze_sha256,
                  args.review, args.review_sha256, args.out)
    execute(freeze, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
