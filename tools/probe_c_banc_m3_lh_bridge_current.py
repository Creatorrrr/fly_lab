"""E265: complete LH bridge incoming-current observation; no candidate change."""

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

from tools import probe_c_banc_m3_lh_edge_restoration as e250
from tools import probe_c_banc_m3_lh_extensor_ablation as e260
from tools import probe_c_banc_m3_lh_neural_response as e248
from tools import probe_c_banc_m3_lh_tonic_compensation as e264
from tools import replay_c_banc_m3_lh_full_inputs as e254

BASE = Path("verification/banc-natural-gait-m3-20260919")
FREEZE_SHA = "eacfaad2e92f3bce6ec593c0683cb712966e444f962a912a40b42f38730d2722"
FREEZE_SCHEMA = "flylab.m3.e265.lh-bridge-current.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e265.lh-bridge-current.review.v1"
OWNED = ("tools/probe_c_banc_m3_lh_bridge_current.py", "tests/test_c_banc_lh_bridge_current.py")
ORDER = ("B", "X", "Z")
TARGET = 111970
CONTROLS = 100
STEPS = 20
HKEY = "H265_six_input_residue"
VALIDITY = "valid_fixed_input_bridge_current"
pinned = e264.pinned


def require_fresh_output(out: Path, expected: Path, minimum_free: int) -> None:
    if out.resolve() != expected.resolve():
        raise ValueError("E265 output path differs")
    if out.exists() or not out.parent.is_dir():
        raise FileExistsError("E265 output must be new")
    if shutil.disk_usage(out.parent).free < minimum_free:
        raise OSError("E265 minimum free disk unavailable")


def gate(freeze_path: Path, freeze_sha: str, review_path: Path,
         review_sha: str, out: Path) -> dict:
    if (freeze_sha != FREEZE_SHA or e254.sha256(freeze_path) != FREEZE_SHA
            or e254.sha256(review_path) != review_sha):
        raise ValueError("E265 freeze/review SHA differs")
    freeze, review = e248.read_json(freeze_path), e248.read_json(review_path)
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("owned_files") != list(OWNED)
            or freeze.get("arm_order") != list(ORDER)
            or freeze.get("model_order") != ["W0", "W6"]
            or freeze.get("controls") != CONTROLS or freeze.get("steps_per_control") != STEPS
            or freeze.get("target_row") != TARGET or freeze.get("n") != e254.N
            or freeze.get("incoming_entries") != 434 or freeze.get("union_count") != 1402
            or freeze.get("output") != (BASE / "e265-lh-bridge-current").as_posix()):
        raise ValueError("E265 frozen contract differs")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("E265 approved source review required")
    pins = {}
    for mapping in (freeze["input_sha256"], freeze["reused_source_sha256"],
                    freeze.get("preservation_sha256", {}), review["source_sha256"],
                    review.get("additional_input_sha256", {}),
                    review.get("additional_reused_source_sha256", {}),
                    {freeze["design"]: freeze["design_sha256"]}):
        if not isinstance(mapping, dict):
            raise TypeError("E265 SHA mapping required")
        for name, digest in mapping.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E265 pin: " + name)
            pins[name] = digest
    for name, digest in pins.items():
        if e254.sha256(pinned(name)) != digest:
            raise ValueError("E265 frozen byte differs: " + name)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("E265 package differs: " + name)
    require_fresh_output(out, ROOT / freeze["output"],
                         freeze["resource_limits"]["minimum_free_disk_bytes"])
    return freeze


def prepare(freeze: dict) -> dict:
    """Explicit E264 preparation reuse; no model construction or global mutation."""
    from flylab.c.storage import StateStore

    f264 = e248.read_json(pinned(freeze["paths"]["upstream_e264_freeze"]))
    data = e264.prepare(f264)
    with np.load(pinned(freeze["paths"]["row_spec"]), allow_pickle=False) as source:
        spec = {key: source[key].copy() for key in source.files}
    union, old, pre, edges = (spec[k] for k in ("union_ids", "old_union_ids", "pre_ids", "edge_ids"))
    if (union.shape != (1402,) or old.shape != (1239,) or pre.shape != (434,)
            or edges.shape != (434,) or spec["six_mask"].sum() != 6
            or spec["previous_observed_mask"].sum() != 271
            or int(spec["target_id"]) != TARGET
            or not np.array_equal(union[spec["pre_columns"]], pre)
            or not np.array_equal(union[spec["old_union_columns"]], old)
            or not np.array_equal(old, data["incoming"]["W6"]["union_ids"])
            or not np.array_equal(edges[spec["six_mask"]], freeze["six_edge_ids"])):
        raise ValueError("E265 complete frozen bridge roster differs")
    for model in ("W0", "W6"):
        csr = data["csr"][model]
        ids = np.arange(csr.indptr[TARGET], csr.indptr[TARGET + 1], dtype=np.int64)
        if (not np.array_equal(ids, edges) or not np.array_equal(csr.indices[ids], pre)
                or not e254.same_bits(csr.data[ids], spec["weights_" + model])):
            raise ValueError("E265 frozen full CSR row differs: " + model)
    references = {}
    for arm in ORDER:
        folder = ROOT / freeze["paths"][arm + "_reference"]
        references[arm] = {p.name: np.load(p, allow_pickle=False)
                           for p in folder.glob("*.npy")}
        references[arm]["start"] = StateStore.load(folder / "observed-start")
        references[arm]["final"] = StateStore.load(folder / "final")
    with np.load(pinned(freeze["paths"]["e263_raw"]), allow_pickle=False) as source:
        prior = {key: source[key].copy() for key in ("B_q19", "B_activation")}
    data.update(spec=spec, references=references, prior=prior)
    return data


def ftz(value):
    """Match the sparse CUDA kernel's flush-to-zero arithmetic."""
    result = np.asarray(value, dtype=np.float32).copy()
    result[np.abs(result) < np.finfo(np.float32).tiny] = np.copysign(
        np.float32(0), result[np.abs(result) < np.finfo(np.float32).tiny])
    return result


def packed_sum32(weights, rates, masks) -> np.float32:
    weights, rates, masks = (np.asarray(v) for v in (weights, rates, masks))
    if (weights.ndim != 1 or rates.shape != weights.shape or masks.shape != weights.shape
            or any(v.dtype != np.float32 or not np.isfinite(v).all()
                   for v in (weights, rates, masks))):
        raise ValueError("E265 aligned finite f32 incoming values required")
    masked = ftz(rates * masks)
    products = ftz(ftz(weights) * ftz(masked))
    lanes = np.zeros(32, np.float32)
    for start in range(0, len(products), 32):
        chunk = products[start:start + 32]
        lanes[:len(chunk)] = ftz(ftz(lanes[:len(chunk)]) + chunk)
    for offset in (16, 8, 4, 2, 1):
        before = lanes.copy()
        lanes[:32 - offset] = ftz(ftz(before[:32 - offset]) + ftz(before[offset:]))
    return lanes[0]


def decomposition(edge_a, edge_b, drive_a, drive_b, adapt_a, adapt_b, six) -> dict:
    a, b = np.asarray(edge_a, np.float64), np.asarray(edge_b, np.float64)
    six = np.asarray(six)
    if (a.shape != b.shape or a.ndim != 2 or a.shape[0] != 101
            or six.shape != (a.shape[1],) or six.dtype != bool
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("E265 complete finite current contrast required")
    drive = np.asarray(drive_a, np.float64) - np.asarray(drive_b, np.float64)
    adapt = np.asarray(adapt_a, np.float64) - np.asarray(adapt_b, np.float64)
    if drive.shape != (101,) or adapt.shape != (101,) or not np.isfinite([drive, adapt]).all():
        raise ValueError("E265 complete finite drive/adaptation contrast required")
    delta = a - b
    total = delta.sum(axis=1, dtype=np.float64)
    source = delta[:, six].sum(axis=1, dtype=np.float64) + drive
    other = delta[:, ~six].sum(axis=1, dtype=np.float64) - adapt
    effective = total + drive - adapt
    if not np.allclose(effective, source + other, atol=1e-10, rtol=0):
        raise AssertionError("E265 D=S+O identity failed")
    return {"D": effective, "S": source, "O": other,
            "drive_delta": drive, "adaptation_delta": adapt, "edge_delta64": delta}


def assess_h265(parts: dict) -> dict:
    d, s, o = (np.asarray(parts[key], np.float64) for key in ("D", "S", "O"))
    if any(v.shape != (101,) or not np.isfinite(v).all() for v in (d, s, o)):
        raise ValueError("E265 complete finite hypothesis arrays required")
    denominator = float(np.sqrt(np.mean(d[1:100] ** 2)))
    numerator = float(np.sqrt(np.mean(o[1:100] ** 2)))
    cases = {str(i): {"D": float(d[i]), "S": float(s[i]), "O": float(o[i]),
                       "other_fraction": None if abs(d[i]) <= 1e-12 else float(abs(o[i]) / abs(d[i])),
                       "same_sign": bool(s[i] * d[i] > 0)} for i in (43, 56, 70)}
    undefined = denominator <= 1e-12 or any(abs(d[i]) <= 1e-12 for i in (43, 56, 70))
    std = float(np.std(s[1:100]))
    supported = None if undefined else bool(numerator / denominator <= .2
        and std > 1e-12 and all(v["other_fraction"] <= .2 and v["same_sign"] for v in cases.values()))
    return {HKEY: supported, "classification": "inconclusive" if supported is None else
            "supported" if supported else "refuted", "RMS_D": denominator, "RMS_O": numerator,
            "other_rms_fraction": None if denominator <= 1e-12 else numerator / denominator,
            "source_std": std, "cases": cases,
            "scope": "endpoint algebra only; no mediation, biology, gait or candidate adoption"}


def stats(value) -> dict:
    value = np.asarray(value, np.float64)
    return {"mean": float(value.mean()), "min": float(value.min()), "max": float(value.max()),
            "RMS": float(np.sqrt(np.mean(value ** 2))),
            "positive_sum": float(value[value > 0].sum()),
            "negative_sum": float(value[value < 0].sum()), "signed_sum": float(value.sum())}


def raw_arrays(folder: Path, union_count: int, drive_count: int) -> dict:
    raw = e264.raw_arrays(folder, 101, union_count, drive_count)
    for name, shape, dtype in (("incoming-rate.npy", (101, 434), np.float32),
                               ("incoming-mask.npy", (101, 434), np.float32),
                               ("edge-current64.npy", (101, 434), np.float64),
                               ("actual-current.npy", (101,), np.float32),
                               ("packed-current32.npy", (101,), np.float32),
                               ("rounding-difference64.npy", (101,), np.float64)):
        raw[name] = np.lib.format.open_memmap(folder / name, mode="w+", shape=shape, dtype=dtype)
        raw[name][:] = np.nan
        raw[name].flush()
    return raw


def context_for(network, data, model):
    import torch

    spec = data["spec"]
    union = spec["union_ids"]
    reference = data["arms"]["baseline"]["reference"]
    context = {"union_ids": union, "device_union_ids": torch.as_tensor(union,
               dtype=torch.int64, device=network.device), "decoder": data["decoder"],
               "spec": spec, "weights": spec["weights_" + model]}
    for name, ids in (("motor", data["config"]["motor_ids"]),
                      ("observed", reference["observed_ids"]), ("core", data["core_ids"])):
        context[name + "_at"] = e264.observed_columns(union, ids)
    return context


def current_probe(network, row, raw, entry, context, persist, *, exact=None):
    """One probe; preserve its intent and full-state invariance, even on error."""
    if exact is None:
        from tools.verify_c_banc_observation import exact
    before = network.snapshot()
    spec = context["spec"]
    rates = before["rate"][spec["pre_ids"]]
    masks = before["output_mask"][spec["pre_ids"]]
    raw["incoming-rate.npy"][row] = rates
    raw["incoming-mask.npy"][row] = masks
    edges = context["weights"].astype(np.float64) * rates.astype(np.float64) * masks.astype(np.float64)
    raw["edge-current64.npy"][row] = edges
    raw["packed-current32.npy"][row] = packed_sum32(context["weights"], rates, masks)
    for name in ("incoming-rate.npy", "incoming-mask.npy", "edge-current64.npy", "packed-current32.npy"):
        raw[name].flush()
    entry["current_probe_requested"] += 1
    persist()
    entry["current_probe_entered"] += 1
    entry["unknown_current_partial"] = True
    persist()
    error = None
    try:
        current = network._current(network.rate)
        raw["actual-current.npy"][row] = current[TARGET].detach().cpu().numpy()
        raw["actual-current.npy"].flush()
        entry["current_probe_completed"] += 1
        entry["unknown_current_partial"] = False
        persist()
    except Exception as failure:  # noqa: BLE001
        error = failure
    try:
        exact(network.snapshot(), before, "E265 full snapshot before/after current probe")
        entry["current_probe_snapshot_exact"] += 1
        persist()
    except Exception as failure:  # noqa: BLE001
        entry["probe_snapshot_error"] = repr(failure)
        if error is None:
            error = failure
    if error is not None:
        raise error
    if not e254.same_bits(raw["actual-current.npy"][row], raw["packed-current32.npy"][row]):
        raise AssertionError("E265 packed32 independent sum differs")
    raw["rounding-difference64.npy"][row] = float(raw["actual-current.npy"][row]) - edges.sum()
    raw["rounding-difference64.npy"].flush()


def observe_endpoint(network, row, raw, entry, context, persist):
    e264.observe_endpoint(network, row, raw, entry, context)
    current_probe(network, row, raw, entry, context, persist)


def diagonal(row, raw, arm, data):
    reference, spec = data["references"][arm], data["spec"]
    for field in ("motor-rate.npy", "observed-rate.npy", "observed-drive.npy",
                  "observed-adaptation.npy", "core-rate.npy", "core-drive.npy", "core-adaptation.npy",
                  "drive-values.npy", "drive-sha256.npy"):
        if arm == "B" and field == "drive-sha256.npy":
            # E260 B has no saved full-drive digest: reconstruct and verify the
            # full device drive each control, while retaining its actual SHA.
            continue
        index = row - 1 if field == "drive-values.npy" else row
        if index < 0:
            continue
        if not e254.same_bits(raw[field][index], reference[field][index]):
            raise AssertionError(f"E265 {arm} original {field} differs at {row}")
    for field in ("rate", "drive", "adaptation"):
        key = "endpoint-" + field + ".npy"
        if not e254.same_bits(raw[key][row, spec["old_union_columns"]], reference[key][row]):
            raise AssertionError(f"E265 {arm} original union1239 {field} differs at {row}")
    bank = raw["bank.npy"][row, [68, 107]] if arm == "B" else raw["bank.npy"][row]
    if not e254.same_bits(bank, reference["bank.npy"][row]):
        raise AssertionError("E265 original decoder bank differs")
    if arm == "B":
        q, activation = data["prior"]["B_q19"][row], data["prior"]["B_activation"][row]
    else:
        q, activation = reference["q19.npy"][row], reference["activation-three.npy"][row]
    if (not np.allclose(raw["activation-three.npy"][row], activation, atol=1e-10, rtol=0)
            or not np.isclose(raw["q19.npy"][row], q, atol=1e-10, rtol=0)):
        raise AssertionError("E265 original q19/three-muscle decoder differs")


def advance_with_receipt(network, drive, control, entry, persist):
    entry["entered_controls"] = control
    entry["unknown_neural_partial"] = True
    persist()
    network.advance(drive, STEPS)
    entry["completed_controls"] = control
    entry["unknown_neural_partial"] = False
    persist()


def run_arm(network, saved, arm, raw, entry, context, data, bias, persist, budget):
    from tools.verify_c_banc_observation import exact

    network.restore(saved)
    exact(network.snapshot(), saved, "E265 whole original start")
    entry["reset_exact"] = True
    bank = data["decoder"]["bank0"]
    raw["bank.npy"][0] = bank
    raw["activation-three.npy"][0] = e264.e263.transfer(bank, context["decoder"]["kappa"],
        context["decoder"]["slopes"], context["decoder"]["pool"])
    raw["q19.npy"][0] = context["decoder"]["rest"][19] + .02 * np.dot(
        context["decoder"]["moment"][19, e264.MUSCLES], raw["activation-three.npy"][0])
    observe_endpoint(network, 0, raw, entry, context, persist)
    diagonal(0, raw, arm, data)
    source = data["arms"]["baseline"]
    for control in range(1, 101):
        budget()
        sensory, original = e254.reconstruct_drive(source["sensory"]["old_filtered"][control - 1],
            source["sensory_ids"], source["sensory_ports"], e254.N,
            data["descending_ids"], 0., data["navigation_ports"], 5000.)
        if not e254.same_bits(sensory[source["sensory"]["source_ids"]],
                source["sensory"]["returned_source_drive"][control - 1]):
            raise AssertionError("E265 frozen original sensory drive differs")
        drive = e264.drive_bias(original, bias) if bias else original
        raw["drive-values.npy"][control - 1] = drive[data["drive_ids"]]
        raw["drive-values.npy"].flush()
        advance_with_receipt(network, drive, control, entry, persist)
        if not e254.same_bits(network.drive.detach().cpu().numpy(), drive):
            raise AssertionError("E265 actual whole drive differs")
        observe_endpoint(network, control, raw, entry, context, persist)
        diagonal(control, raw, arm, data)
    final = network.snapshot()
    if final["tick"] != saved["tick"] + 2000:
        raise AssertionError("E265 exact neural tick count differs")
    return final


def factors(raws, data):
    spec = data["spec"]
    at = int(spec["target_column"])
    arrays = {"control_tick": np.arange(101, dtype=np.int64), "time_s": np.arange(101) * .005}
    report = {"schema": "flylab.m3.e265.factor-windows.v1", "arms": {}, "contrasts": {},
              "previously_observed_entries": 271, "previously_missing_entries": 163,
              "initial_row": "original saved neural endpoint; hypothetical decoder",
              "interpretation_status": "complete_neural_only"}
    for arm in ORDER:
        raw = raws[arm]
        edge = np.asarray(raw["edge-current64.npy"])
        values = {"rate": raw["endpoint-rate.npy"][:, at],
                  "drive": raw["endpoint-drive.npy"][:, at],
                  "adaptation": raw["endpoint-adaptation.npy"][:, at],
                  "total_current64": edge.sum(axis=1),
                  "six_current64": edge[:, spec["six_mask"]].sum(axis=1),
                  "other_current64": edge[:, ~spec["six_mask"]].sum(axis=1),
                  "previous_observed_current64": edge[:, spec["previous_observed_mask"]].sum(axis=1),
                  "previous_missing_current64": edge[:, ~spec["previous_observed_mask"]].sum(axis=1),
                  "actual_current32": raw["actual-current.npy"],
                  "rounding_difference64": raw["rounding-difference64.npy"]}
        values["margin"] = values["total_current64"] + values["drive"] - values["adaptation"] - data["threshold"]
        report["arms"][arm] = {key: stats(value[1:100]) for key, value in values.items()}
        for key, value in values.items():
            arrays[arm + "__" + key] = np.asarray(value).copy()
    for left, right in (("X", "B"), ("Z", "B"), ("Z", "X")):
        a, b = raws[left], raws[right]
        parts = decomposition(a["edge-current64.npy"], b["edge-current64.npy"],
            a["endpoint-drive.npy"][:, at], b["endpoint-drive.npy"][:, at],
            a["endpoint-adaptation.npy"][:, at], b["endpoint-adaptation.npy"][:, at], spec["six_mask"])
        name = left + "_minus_" + right
        report["contrasts"][name] = {key: stats(value[1:100]) for key, value in parts.items()}
        for key in ("rate", "drive", "adaptation", "margin", "total_current64", "six_current64",
                    "other_current64", "previous_observed_current64", "previous_missing_current64",
                    "actual_current32", "rounding_difference64"):
            delta = arrays[left + "__" + key].astype(np.float64) - arrays[right + "__" + key].astype(np.float64)
            report["contrasts"][name][key] = stats(delta[1:100])
            arrays[name + "__" + key] = delta
        for key, value in parts.items():
            arrays[name + "__" + key] = value
        if name == "Z_minus_B":
            report["H265"] = assess_h265(parts)
    return arrays, report


def output_manifest(out, maximum):
    manifest = e264.output_manifest(out, maximum)
    manifest["schema"] = "flylab.m3.e265.output-manifest.v1"
    return manifest


def model_parameters(network):
    return {name: tensor.detach().cpu().numpy().copy() for name, tensor in
        (("threshold", network.threshold), ("coefficient", network.coefficient),
         ("cap", network.cap), ("membrane_tau", network.tau), ("cell_gain", network._cell_gain),
         ("cell_adaptation_tau", network._cell_adaptation_tau), ("inverse_adaptation_tau", network._cell_inverse_tau))}


def finish_receipts(out, freeze, ledger, result, errors, primary, began, *, persist=e248.write_json):
    """Publication errors invalidate inference while retaining the first error and raw."""
    def fail(error, stage):
        nonlocal primary
        if primary is None:
            primary = error
        errors.append({"stage": stage, "type": type(error).__name__, "message": str(error)})
        result.update(validity="resource_limited" if isinstance(primary, TimeoutError) else
            "invalid_or_incomplete", **{HKEY: None}, H265_classification="not_tested",
            factor_report_available=False, finalization_pending=True,
            first_error={"type": type(primary).__name__, "message": str(primary)})
        if isinstance(result.get("H265"), dict):
            result["H265"].update({HKEY: None, "classification": "not_tested"})
        path = out / "factor-windows.json"
        if path.exists():
            try:
                report = e248.read_json(path)
                report["H265"].update({HKEY: None, "classification": "not_tested"})
                report["interpretation_status"] = "not_tested"
                persist(path, report)
            except Exception as secondary:  # noqa: BLE001
                errors.append({"stage": "report revocation", "message": str(secondary)})
                try:
                    path.replace(out / "factor-windows-unvalidated-raw.json")
                except Exception as removal:  # noqa: BLE001
                    errors.append({"stage": "report rename", "message": str(removal)})
                    path.unlink(missing_ok=True)
    if primary is not None:
        fail(primary, "prior science/cleanup")
    arms, models = ledger["arms"], ledger["models"]
    result["work"] = {"constructors_requested": len(models),
        "constructors_entered": sum(v["entered"] for v in models.values()),
        "constructors_completed": sum(v["completed"] for v in models.values()),
        "setup_executed_ticks_known": sum(v["setup_executed_ticks"] or 0 for v in models.values()),
        "setup_capture_ticks_known": sum(v["setup_capture_ticks"] or 0 for v in models.values()),
        "scientific_ticks_completed": 20 * sum(v["completed_controls"] for v in arms.values()),
        **{key + "_total": sum(v[key] for v in arms.values()) for key in
            ("current_probe_requested", "current_probe_entered", "current_probe_completed", "current_probe_snapshot_exact")},
        "unknown_partial_models": [k for k, v in models.items() if v["setup_unknown_partial"]],
        "unknown_partial_arms": [k for k, v in arms.items() if v["unknown_neural_partial"]],
        "unknown_partial_current": [k for k, v in arms.items() if v["unknown_current_partial"]]}
    for name, value in (("advance-ledger.json", ledger), ("errors.json", {"errors": errors})):
        try:
            persist(out / name, value)
        except Exception as error:  # noqa: BLE001
            fail(error, name)
    try:
        if time.perf_counter() - began >= freeze["resource_limits"]["internal_s"]:
            raise TimeoutError("E265 finalization deadline")
        persist(out / "errors.json", {"errors": errors})
        persist(out / "output-manifest.json", output_manifest(out, freeze["resource_limits"]["output_bytes"]))
        result["elapsed_s"] = time.perf_counter() - began
        result["finalization_pending"] = primary is not None
        persist(out / "result.json", result)
        if time.perf_counter() - began >= freeze["resource_limits"]["internal_s"]:
            raise TimeoutError("E265 late publication deadline")
    except Exception as error:  # noqa: BLE001
        fail(error, "publication")
        (out / "output-manifest.json").unlink(missing_ok=True)
        for name, value in (("advance-ledger.json", ledger), ("errors.json", {"errors": errors}), ("result.json", result)):
            try:
                persist(out / name, value)
            except Exception as secondary:  # noqa: BLE001
                fail(secondary, "retry " + name)
                (out / name).unlink(missing_ok=True)
    return primary


def execute(freeze, out):
    from flylab.c.storage import StateStore
    from tools.verify_c_banc_observation import exact

    began = time.perf_counter()
    out.mkdir(exist_ok=False)
    ledger = {"schema": "flylab.m3.e265.ledger.v1", "models": {}, "arms": {}, "resources": [],
              "original_attempts": 1, "retries": 0, "body_constructors": 0,
              "physics_steps": 0, "fit_calls": 0, "solver_calls": 0}
    result = {"schema": "flylab.m3.e265.result.v1", "validity": "incomplete", HKEY: None,
              "H265_classification": "not_tested", "factor_report_available": False,
              "candidate_adopted": False, "whole_goal_pass": False, "F6_exposed": False,
              "active_UI_changed": False, "finalization_pending": True}
    errors, primary, raws = [], None, {}
    network = active_dir = active_entry = limiter = None
    old_threads, old_env = None, {}
    def persist():
        e248.write_json(out / "advance-ledger.json", ledger)
    def budget():
        e260.resource_check(freeze, began, out, ledger)
    try:
        import torch
        from threadpoolctl import threadpool_info, threadpool_limits

        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            old_env[key] = os.environ.get(key)
            os.environ[key] = "1"
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        limiter = threadpool_limits(limits=1)
        if any(v["num_threads"] != 1 for v in threadpool_info()):
            raise RuntimeError("E265 effective CPU threads differ")
        e248.write_json(out / "threading.json", {"torch_threads": torch.get_num_threads(), "threadpools": threadpool_info()})
        if torch.cuda.mem_get_info()[0] < freeze["resource_limits"]["minimum_free_gpu_bytes"]:
            raise MemoryError("E265 minimum free GPU unavailable")
        torch.cuda.reset_peak_memory_stats()
        data = prepare(freeze)
        np.savez(out / "incoming-row-spec.npz", **data["spec"])
        np.savez(out / "decoder-parameters.npz", **data["decoder"], muscle_ids=e264.MUSCLES)
        np.savez(out / "drive-mapping.npz", drive_ids=data["drive_ids"],
            sensory_ids=data["arms"]["baseline"]["sensory_ids"],
            sensory_ports=data["arms"]["baseline"]["sensory_ports"],
            source_ids=data["arms"]["baseline"]["sensory"]["source_ids"],
            navigation_ports=data["navigation_ports"], descending_ids=data["descending_ids"])
        e248.write_json(out / "source-manifest.json", {"freeze_sha256": FREEZE_SHA,
            "input_sha256": freeze["input_sha256"], "source_sha256": freeze["reused_source_sha256"],
            "owned_source_sha256": {p: e254.sha256(ROOT / p) for p in OWNED}})
        for model in ("W0", "W6"):
            if network is not None:
                network = None
                gc.collect()
                torch.cuda.empty_cache()
            arm_start = "B" if model == "W0" else "X"
            saved = data["references"][arm_start]["start"]
            model_entry = {"requested": True, "entered": False, "completed": False,
                "setup_unknown_partial": False, "setup_executed_ticks": None, "setup_capture_ticks": None}
            ledger["models"][model] = model_entry
            persist()
            budget()
            model_entry.update(entered=True, setup_unknown_partial=True)
            persist()
            network, restored, identity_fields, backend = e250._constructor(data["csr"][model],
                data["tau"], data["config"], saved, expected_identity=freeze["model_identity"][model], restore_saved=saved)
            if identity_fields or network.identity != freeze["model_identity"][model]:
                raise AssertionError("E265 unchanged whole model identity required")
            exact(restored, saved, "E265 constructor exact original start")
            model_entry.update(completed=True, setup_unknown_partial=False,
                setup_executed_ticks=60, setup_capture_ticks=20, backend=backend, model_hash=network.identity)
            persist()
            parameters = model_parameters(network)
            np.savez(out / ("model-parameters-" + model + ".npz"), **parameters)
            data["threshold"] = float(parameters["threshold"][TARGET])
            csr = data["csr"][model]
            e248.write_json(out / ("csr-" + model + ".json"), {field: hashlib.sha256(
                np.ascontiguousarray(getattr(csr, field)).tobytes()).hexdigest() for field in ("indptr", "indices", "data")})
            context = context_for(network, data, model)
            for arm in (("B",) if model == "W0" else ("X", "Z")):
                folder = out / arm
                folder.mkdir()
                entry = e264.begin_arm(ledger, arm, 100)
                entry.update(current_probe_requested=0, current_probe_entered=0, current_probe_completed=0,
                             current_probe_snapshot_exact=0, unknown_current_partial=False, weights=model)
                active_dir, active_entry = folder, entry
                persist()
                raw = raw_arrays(folder, 1402, len(data["drive_ids"]))
                raws[arm] = raw
                saved = data["references"][arm]["start"]
                network.restore(saved)
                exact(network.snapshot(), saved, "E265 original full restored start")
                StateStore.save(folder / "observed-start", network.snapshot())
                entry["initial_saved"] = True
                persist()
                final = run_arm(network, saved, arm, raw, entry, context, data,
                    np.float32(freeze["arm_bindings"][arm]["bias32"]), persist, budget)
                StateStore.save(folder / "final", final)
                entry["final_saved"] = True
                persist()
                exact(final, data["references"][arm]["final"], "E265 whole original final " + arm)
                if (entry["current_probe_completed"] != 101 or entry["current_probe_snapshot_exact"] != 101
                        or entry["completed_controls"] != 100 or entry["observed_endpoints"] != 101):
                    raise AssertionError("E265 endpoint/advance/probe accounting differs")
                entry["status"] = "complete_exact"
                unchanged = model_parameters(network)
                if (network.identity != freeze["model_identity"][model]
                        or any(not e254.same_bits(value, unchanged[key]) for key, value in parameters.items())):
                    raise AssertionError("E265 model/parameters changed during observation")
                entry["parameters_exact"] = True
                e254.flush_raw(raw)
                persist()
                active_dir = active_entry = None
        arrays, report = factors(raws, data)
        np.savez(out / "factor-effects.npz", **arrays)
        e248.write_json(out / "factor-windows.json", report)
        result.update(validity=VALIDITY, **{HKEY: report["H265"][HKEY]},
            H265=report["H265"], H265_classification=report["H265"]["classification"],
            factor_report_available=True, B_X_Z_exact=True)
    except Exception as error:  # noqa: BLE001
        primary = error
        errors.append({"stage": "science", "type": type(error).__name__, "message": str(error),
                       "traceback": traceback.format_exc()})
        if network is not None and active_dir is not None and not (active_dir / "final").exists():
            primary = e254.preserve_partial_state(network, active_dir / "partial-final", active_entry, primary)
    finally:
        for raw in raws.values():
            try:
                e254.flush_raw(raw)
            except Exception as error:  # noqa: BLE001
                if primary is None:
                    primary = error
                errors.append({"stage": "raw flush", "message": str(error)})
        network = None
        raws.clear()
        for stage, action in (("gc", gc.collect), ("threadpool restore", lambda: limiter.restore_original_limits() if limiter else None),
                              ("torch restore", lambda: torch.set_num_threads(old_threads) if old_threads is not None else None)):
            try:
                action()
            except Exception as error:  # noqa: BLE001
                if primary is None:
                    primary = error
                errors.append({"stage": stage, "message": str(error)})
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        primary = finish_receipts(out, freeze, ledger, result, errors, primary, began)
        if primary is not None:
            raise RuntimeError("E265 invalid: " + str(primary)) from primary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    freeze = gate(args.freeze, args.freeze_sha256, args.review, args.review_sha256, args.out)
    execute(freeze, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
