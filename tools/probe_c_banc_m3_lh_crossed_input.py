"""E255: bounded body-free crossing of two saved neural inputs and CSR models."""

from __future__ import annotations

import argparse
import gc
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
from tools import probe_c_banc_m3_lh_neural_response as e248
from tools import replay_c_banc_m3_lh_full_inputs as e254

BASE = Path("verification/banc-natural-gait-m3-20260919")
FREEZE_SHA = "5800a3a61c081f31b0c5382c52581602fa850a2dd18a7c11da693d2bf4f55605"
FREEZE_SCHEMA = "flylab.m3.e255.lh-crossed-input.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e255.lh-crossed-input.review.v1"
OWNED = ("tools/probe_c_banc_m3_lh_crossed_input.py",
         "tests/test_c_banc_lh_crossed_input.py")
VALIDITY = "valid_saved_input_crossed_neural_response"
ORDER = ("00", "01", "11", "10")
BINDINGS = {"00": ("baseline", "baseline"),
            "01": ("baseline", "candidate"),
            "11": ("candidate", "candidate"),
            "10": ("candidate", "baseline")}
CONTROLS = 100
STEPS = 20
N = e254.N
TRACKED = np.asarray([135246, 20156, 111970, 6373, 22787], np.int64)
TARGETS = e254.TARGETS


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
    """Verify frozen bytes and packages before any original array is loaded."""
    if (freeze_sha != FREEZE_SHA or e254.sha256(freeze_path) != FREEZE_SHA
            or e254.sha256(review_path) != review_sha):
        raise ValueError("E255 freeze/review SHA differs")
    freeze, review = e248.read_json(freeze_path), e248.read_json(review_path)
    limits = freeze.get("resource_limits", {})
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("e254_freeze_sha256") != e254.FREEZE_SHA
            or freeze.get("graph_hash") !=
            "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"
            or freeze.get("controls") != CONTROLS
            or freeze.get("steps_per_control") != STEPS
            or freeze.get("model_order") != ["baseline", "candidate"]
            or freeze.get("arm_order") != list(ORDER)
            or freeze.get("arm_bindings") != {k: {"weights": w, "input": u}
                                                 for k, (w, u) in BINDINGS.items()}
            or freeze.get("diagonal_references") != {"00": "baseline",
                                                        "11": "candidate"}
            or freeze.get("startup") != [1, 100]
            or freeze.get("case_controls") != [64, 84]
            or freeze.get("hypothesis") != {"edge_id": 444229,
                                             "pre_id": 135246,
                                             "post_id": 6373,
                                             "relative_tolerance": .2}
            or freeze.get("tracked_ids") != TRACKED.tolist()
            or freeze.get("motor_ids") != TARGETS.tolist()
            or freeze.get("additional_edge_pre_post") != [20156, 22787]
            or any(limits.get(key) != value for key, value in {
                "cpu_threads": 1, "gpu_processes": 1, "original_attempts": 1,
                "retries": 0, "constructors": 2, "setup_executed_ticks": 120,
                "setup_capture_ticks": 40, "scientific_ticks": 8000,
                "current_probes": 404, "body_constructors": 0,
                "physics_steps": 0, "fit_calls": 0, "solver_calls": 0,
                "internal_s": 38, "finalization_reserve_s": 6,
                "external_s": 43, "supervisor_wall_s": 45,
                "process_memory_bytes": 4294967296,
                "additional_gpu_memory_bytes": 3221225472,
                "minimum_free_gpu_bytes": 3221225472,
                "output_bytes": 268435456,
                "minimum_free_disk_bytes": 1073741824}.items())):
        raise ValueError("E255 fixed science/resource contract differs")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("E255 source review not approved")
    pins: dict[str, str] = {}
    for mapping in (freeze["input_sha256"], freeze["reused_source_sha256"],
                    freeze["preservation_sha256"], review["source_sha256"],
                    review.get("additional_input_sha256", {}),
                    review.get("additional_reused_source_sha256", {}),
                    {freeze["design"]: freeze["design_sha256"]}):
        if not isinstance(mapping, dict):
            raise TypeError("E255 SHA mapping required")
        for name, digest in mapping.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E255 SHA pin: " + name)
            pins[name] = digest
    for name, digest in pins.items():
        if e254.sha256(pinned(name)) != digest:
            raise ValueError("E255 frozen byte differs: " + name)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("E255 package version differs: " + name)
    if out.resolve() != (ROOT / BASE / "e255-lh-crossed-input").resolve():
        raise ValueError("E255 output path differs")
    if out.exists() or not out.parent.is_dir():
        raise FileExistsError("E255 output must be new")
    if shutil.disk_usage(out.parent).free < limits["minimum_free_disk_bytes"]:
        raise OSError("E255 minimum free disk unavailable")
    return freeze


def crossed_effects(y00, y01, y11, y10) -> dict[str, np.ndarray]:
    """Float64 factorial contrasts, with both conditional effect orders."""
    a, b, c, d = [np.asarray(x, np.float64) for x in (y00, y01, y11, y10)]
    if not (a.shape == b.shape == c.shape == d.shape
            and all(np.isfinite(x).all() for x in (a, b, c, d))):
        raise ValueError("Complete finite crossed observations required")
    result = {"T": c - a, "C": d - a, "S": b - a,
              "I": c - d - b + a, "C1": c - b, "S1": c - d}
    result["C_order_mean"] = (result["C"] + result["C1"]) / 2
    result["S_order_mean"] = (result["S"] + result["S1"]) / 2
    if not np.allclose(result["T"], result["C"] + result["S"] + result["I"],
                       atol=1e-10, rtol=1e-12):
        raise AssertionError("Crossed algebra failed")
    return result


def signed_window_parts(delta, lo: int, hi: int) -> dict[str, float]:
    """Separate signs at each endpoint, then sum; retain signed cancellation."""
    values = np.asarray(delta, np.float64)[lo:hi]
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Nonempty finite endpoint window required")
    return {"signed": float(values.sum(dtype=np.float64)),
            "positive": float(np.maximum(values, 0).sum(dtype=np.float64)),
            "negative": float(np.minimum(values, 0).sum(dtype=np.float64))}


def dynamic_start_equal(first: dict, second: dict, exact=None) -> None:
    """Allow only the documented model identity/patch metadata to differ."""
    if exact is None:
        from tools.verify_c_banc_observation import exact
    exclude = set(e250.METADATA)
    if set(first) != set(second):
        raise AssertionError("Crossed model snapshot fields differ")
    exact({k: v for k, v in first.items() if k not in exclude},
          {k: v for k, v in second.items() if k not in exclude},
          "E255 common unmodified neural dynamic start")


def diagonal_exact(raw: dict, reference: dict, row: int,
                   *, same=e254.same_bits) -> None:
    """Check one complete diagonal endpoint against frozen E254 bytes."""
    for key in ("endpoint-rate.npy", "endpoint-drive.npy",
                "endpoint-adaptation.npy", "actual-current.npy",
                "edge-current64.npy"):
        if not same(np.asarray(raw[key][row]), np.asarray(reference[key][row])):
            raise AssertionError(f"E255 diagonal {key} differs at {row}")


def reference_arrays(folder: Path) -> dict:
    keys = ("endpoint-rate.npy", "endpoint-drive.npy",
            "endpoint-adaptation.npy", "actual-current.npy",
            "edge-current64.npy", "drive-values.npy")
    return {key: np.load(folder / key, mmap_mode="r") for key in keys}


def raw_arrays(folder: Path, incoming: dict, drive_count: int) -> dict:
    specs = {"endpoint-rate.npy": (CONTROLS + 1, len(incoming["union_ids"])),
             "endpoint-drive.npy": (CONTROLS + 1, len(incoming["union_ids"])),
             "endpoint-adaptation.npy": (CONTROLS + 1, len(incoming["union_ids"])),
             "actual-current.npy": (CONTROLS + 1, len(TARGETS)),
             "edge-current64.npy": (CONTROLS + 1, len(incoming["edge_ids"])),
             "drive-values.npy": (CONTROLS, drive_count)}
    arrays = {}
    for key, shape in specs.items():
        dtype = np.float64 if key == "edge-current64.npy" else np.float32
        arrays[key] = np.lib.format.open_memmap(folder / key, mode="w+",
                                               dtype=dtype, shape=shape)
        arrays[key][:] = np.nan
        arrays[key].flush()
    return arrays


def observe_endpoint(network, row: int, incoming: dict, raw: dict,
                     entry: dict, *, probe=e254.observe_network) -> None:
    from tools.verify_c_banc_observation import exact

    before = network.snapshot() if row in (0, CONTROLS) else None
    probe(network, row, incoming["union_ids"], incoming, raw, entry)
    if before is not None:
        exact(network.snapshot(), before, "E255 read-only neural endpoint probe")


def run_crossed_arm(network, saved: dict, input_data: dict, incoming: dict,
                    drive_ids: np.ndarray, navigation_ports: np.ndarray,
                    descending_ids: np.ndarray, raw: dict, entry: dict,
                    persist, budget, reference: dict | None = None,
                    *, observe=observe_endpoint, controls=CONTROLS,
                    n=N, steps=STEPS,
                    reconstruct=e254.reconstruct_drive) -> None:
    """Restore exactly, then replay one original input sequence for 100 controls."""
    from tools.verify_c_banc_observation import exact

    network.restore(saved)
    exact(network.snapshot(), saved, "E255 arm restored dynamic start")
    entry["reset_exact"] = True
    persist()
    observe(network, 0, incoming, raw, entry)
    persist()
    if reference is not None:
        diagonal_exact(raw, reference, 0)
    for control in range(1, controls + 1):
        budget()
        sensory, drive = reconstruct(
            input_data["sensory"]["old_filtered"][control - 1],
            input_data["sensory_ids"], input_data["sensory_ports"], n,
            descending_ids, 0., navigation_ports, 5000.)
        sources = input_data["sensory"]["source_ids"]
        if not e254.same_bits(sensory[sources],
                              input_data["sensory"]["returned_source_drive"][control - 1]):
            raise AssertionError(f"E255 frozen sensory drive differs at {control}")
        raw["drive-values.npy"][control - 1] = drive[drive_ids]
        raw["drive-values.npy"].flush()
        if reference is not None and not e254.same_bits(
                raw["drive-values.npy"][control - 1],
                reference["drive-values.npy"][control - 1]):
            raise AssertionError(f"E255 diagonal injected drive differs at {control}")
        entry["entered_controls"] = control
        entry["unknown_neural_partial"] = True
        persist()
        network.advance(drive, steps)
        entry["completed_controls"] = control
        entry["unknown_neural_partial"] = False
        persist()
        observe(network, control, incoming, raw, entry)
        persist()
        if reference is not None:
            diagonal_exact(raw, reference, control)
    entry["status"] = "science_complete"


def selected_edges(incoming: dict) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(incoming["edge_ids"])
    primary = np.flatnonzero(ids == 444229)
    extra = np.flatnonzero((incoming["pre_ids"] == 20156)
                           & (incoming["post_ids"] == 22787))
    if len(primary) != 1 or len(extra) != 1:
        raise ValueError("E255 two fixed observed edges required")
    if (int(incoming["pre_ids"][primary[0]]) != 135246
            or int(incoming["post_ids"][primary[0]]) != 6373):
        raise ValueError("E255 fixed H255 edge binding differs")
    return np.r_[primary, extra].astype(np.int64), ids[np.r_[primary, extra]]


def verify_target_parameters(network, expected: dict) -> None:
    for key, source in (("threshold", network.threshold),
                        ("coefficient", network.coefficient),
                        ("cap", network.cap),
                        ("membrane_tau", network.tau),
                        ("cell_gain", network._cell_gain),
                        ("cell_adaptation_tau", network._cell_adaptation_tau),
                        ("inverse_adaptation_tau", network._cell_inverse_tau)):
        if not e254.same_bits(expected[key],
                              source[TARGETS].detach().cpu().numpy()):
            raise AssertionError("E255 target physiological parameter changed: " + key)


def factors(raws: dict, incoming: dict, threshold: np.ndarray) -> tuple[dict, dict]:
    """Make requested signed/current contrasts and fixed H255 report."""
    union = incoming["union_ids"]
    tracked_at = np.searchsorted(union, TRACKED)
    target_at = np.searchsorted(union, TARGETS)
    if (np.any(tracked_at >= len(union)) or np.any(target_at >= len(union))
            or not np.array_equal(union[tracked_at], TRACKED)
            or not np.array_equal(union[target_at], TARGETS)):
        raise ValueError("E255 tracked graph-row alignment differs")
    edge_at, edge_ids = selected_edges(incoming)
    offsets = incoming["target_offsets"]
    threshold = np.asarray(threshold)
    if (threshold.shape != (len(TARGETS),) or threshold.dtype != np.float32
            or not np.isfinite(threshold).all()):
        raise ValueError("E255 original target threshold required")
    observed = {}
    for arm in ORDER:
        raw = raws[arm]
        current64 = np.stack([
            np.asarray(raw["edge-current64.npy"][:, offsets[j]:offsets[j + 1]],
                       np.float64).sum(axis=1, dtype=np.float64)
            for j in range(len(TARGETS))], axis=1)
        observed[arm] = {
            "rate": np.asarray(raw["endpoint-rate.npy"][:, tracked_at], np.float64),
            "drive": np.asarray(raw["endpoint-drive.npy"][:, tracked_at], np.float64),
            "negative_adaptation": -np.asarray(
                raw["endpoint-adaptation.npy"][:, tracked_at], np.float64),
            "actual_current": np.asarray(raw["actual-current.npy"], np.float64),
            "current64": current64,
            "margin64": (current64 + np.asarray(
                raw["endpoint-drive.npy"][:, target_at], np.float64)
                - threshold.astype(np.float64)
                - np.asarray(raw["endpoint-adaptation.npy"][:, target_at],
                             np.float64)),
            "edge_current64": np.asarray(raw["edge-current64.npy"][:, edge_at],
                                         np.float64),
        }
    output = {"tracked_ids": TRACKED, "target_ids": TARGETS,
              "edge_ids": edge_ids, "edge_pre_ids": incoming["pre_ids"][edge_at],
              "edge_post_ids": incoming["post_ids"][edge_at]}
    for arm in ORDER:
        output[f"current64_{arm}"] = observed[arm]["current64"]
        output[f"margin64_{arm}"] = observed[arm]["margin64"]
    report = {"schema": "flylab.m3.e255.factor-windows.v1", "windows": {},
              "hypothesis_edge_id": 444229}
    windows = {"startup": (1, 100), "case64": (64, 65), "case84": (84, 85)}
    for variable in observed["00"]:
        effect = crossed_effects(*(observed[a][variable] for a in ORDER))
        for kind, values in effect.items():
            output[f"{variable}_{kind}"] = values
        report["windows"][variable] = {
            name: {kind: [signed_window_parts(values[:, j], lo, hi)
                          for j in range(values.shape[1])]
                   for kind, values in effect.items()}
            for name, (lo, hi) in windows.items()}
    h_idx = 0
    effect = crossed_effects(*(observed[a]["edge_current64"][:, h_idx]
                               for a in ORDER))
    t = signed_window_parts(effect["T"], 1, 100)["signed"]
    c = signed_window_parts(effect["C"], 1, 100)["signed"]
    report["H255"] = {"T": t, "C": c,
                      "S": signed_window_parts(effect["S"], 1, 100)["signed"],
                      "I": signed_window_parts(effect["I"], 1, 100)["signed"],
                      "C1": signed_window_parts(effect["C1"], 1, 100)["signed"],
                      "S1": signed_window_parts(effect["S1"], 1, 100)["signed"],
                      "C_order_mean": signed_window_parts(effect["C_order_mean"],
                                                          1, 100)["signed"],
                      "S_order_mean": signed_window_parts(effect["S_order_mean"],
                                                          1, 100)["signed"],
                      "relative_tolerance": .2,
                      "H255_connectivity_effect_retained": bool(
                          t > 0 and c > 0 and abs(t - c) <= .2 * abs(t))}
    return output, report


def begin_arm(ledger: dict, arm: str) -> dict:
    entry = {"weights": BINDINGS[arm][0], "input": BINDINGS[arm][1],
             "requested_controls": CONTROLS, "reset_requested": True,
             "reset_exact": False, "entered_controls": 0, "completed_controls": 0,
             "unknown_neural_partial": False, "current_probe_requested": 0,
             "current_probe_entered": 0, "current_probe_completed": 0,
             "observed_endpoint_count": 0, "initial_saved": False,
             "final_saved": False, "diagonal_exact": None, "status": "requested"}
    ledger["arms"][arm] = entry
    return entry


def resource_check(freeze: dict, began: float, out: Path, ledger: dict,
                   *, snapshot=None) -> None:
    import psutil
    import torch

    limits = freeze["resource_limits"]
    if snapshot is None:
        memory = psutil.Process().memory_info()
        snapshot = {"elapsed_s": time.perf_counter() - began,
                    "resident_bytes": getattr(memory, "peak_wset", memory.rss),
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "gpu_free_bytes": torch.cuda.mem_get_info()[0],
                    "disk_free_bytes": shutil.disk_usage(out).free}
    if len(ledger["resources"]) < 20:
        ledger["resources"].append(snapshot)
    if (snapshot["elapsed_s"] >= limits["internal_s"] - limits["finalization_reserve_s"]
            or snapshot["resident_bytes"] > limits["process_memory_bytes"]
            or snapshot["gpu_peak_allocated_bytes"] >
            limits["additional_gpu_memory_bytes"]
            or snapshot["gpu_free_bytes"] < limits["minimum_free_gpu_bytes"]
            or snapshot["disk_free_bytes"] < limits["minimum_free_disk_bytes"]):
        raise TimeoutError("E255 original science/resource cap reached")


def output_manifest(out: Path, limit: int) -> dict:
    files, total = {}, 0
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name in ("result.json", "output-manifest.json"):
            continue
        size = path.stat().st_size
        total += size
        if total > limit:
            raise OSError("E255 output bytes exceeded")
        files[path.relative_to(out).as_posix()] = {
            "sha256": e254.sha256(path), "bytes": size}
    return {"schema": "flylab.m3.e255.output-manifest.v1", "bytes": total,
            "files": files}


def finish_receipts(out: Path, freeze: dict, ledger: dict, result: dict,
                    errors: list, primary, began: float,
                    *, persist=e248.write_json):
    """Publish only complete consistent receipts; revoke on every late fault."""
    limit = freeze["resource_limits"]["internal_s"]

    def fail(error, stage):
        nonlocal primary
        if primary is None:
            primary = error
        errors.append({"stage": stage, "type": type(error).__name__,
                       "message": str(error)})
        result.update(validity="resource_limited" if isinstance(primary, TimeoutError)
                      else "invalid_or_incomplete",
                      H255_connectivity_effect_retained=False,
                      factor_report_available=False,
                      finalization_pending=True,
                      first_error={"type": type(primary).__name__,
                                   "message": str(primary)})
        if isinstance(result.get("H255"), dict):
            # Retain measured contrasts, never a support verdict from an
            # incomplete science/cleanup/publication path.
            result["H255"]["H255_connectivity_effect_retained"] = False
            result["H255"]["interpretation_status"] = "unvalidated_incomplete"

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
        "current_probe_calls_completed": sum(x["current_probe_completed"]
                                              for x in ledger["arms"].values()),
        "unknown_partial_models": [k for k, x in ledger["models"].items()
                                   if x["setup_unknown_partial"]],
        "unknown_partial_arms": [k for k, x in ledger["arms"].items()
                                 if x["unknown_neural_partial"]]}
    for name, value in (("advance-ledger.json", ledger),
                        ("errors.json", {"errors": errors})):
        try:
            persist(out / name, value)
        except Exception as error:  # noqa: BLE001
            fail(error, name)
    if time.perf_counter() - began >= limit:
        fail(TimeoutError("E255 finalization deadline"), "deadline")
    if primary is not None:
        (out / "factor-windows.json").unlink(missing_ok=True)
        (out / "factor-effects.npz").unlink(missing_ok=True)
    try:
        persist(out / "errors.json", {"errors": errors})
        persist(out / "output-manifest.json",
                output_manifest(out, freeze["resource_limits"]["output_bytes"]))
        result["elapsed_s"] = time.perf_counter() - began
        if result["elapsed_s"] >= limit:
            raise TimeoutError("E255 publication deadline")
        result["finalization_pending"] = False
        persist(out / "result.json", result)
        if time.perf_counter() - began >= limit:
            raise TimeoutError("E255 late result deadline")
    except Exception as error:  # noqa: BLE001
        fail(error, "publication")
        (out / "result.json").unlink(missing_ok=True)
        (out / "output-manifest.json").unlink(missing_ok=True)
        (out / "factor-windows.json").unlink(missing_ok=True)
        (out / "factor-effects.npz").unlink(missing_ok=True)
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
    ledger = {"schema": "flylab.m3.e255.ledger.v1", "status": "entered",
              "models": {}, "arms": {}, "resources": [],
              "original_attempts": 1, "retries": 0,
              "body_constructors": 0, "physics_steps": 0,
              "fit_calls": 0, "solver_calls": 0}
    result = {"schema": "flylab.m3.e255.result.v1", "validity": "incomplete",
              "H255_connectivity_effect_retained": False,
              "factor_report_available": False, "candidate_adopted": False,
              "whole_goal_pass": False, "F6_exposed": False,
              "active_UI_changed": False, "finalization_pending": True}
    errors, primary = [], None
    network, active_dir, active_arm = None, None, None
    raws, limiter, old_torch_threads, old_environment = {}, None, None, {}
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
                or any(x["num_threads"] != 1 for x in threadpool_info())):
            raise RuntimeError("E255 effective CPU thread count differs")
        e248.write_json(out / "threading.json", {"torch_threads": 1,
                         "threadpools": threadpool_info()})
        if torch.cuda.mem_get_info()[0] < freeze["resource_limits"]["minimum_free_gpu_bytes"]:
            raise MemoryError("E255 minimum free GPU memory unavailable")
        torch.cuda.reset_peak_memory_stats()
        e254_freeze = e248.read_json(pinned(freeze["e254_freeze_path"]))
        inputs = e254.prepare(e254_freeze)
        incoming = inputs["incoming"]
        drive_ids = inputs["drive_ids"]
        np.savez(out / "incoming-rows.npz", **incoming)
        np.savez(out / "drive-mapping.npz",
                 sensory_ids=inputs["arms"]["baseline"]["sensory_ids"],
                 sensory_ports=inputs["arms"]["baseline"]["sensory_ports"],
                 navigation_ports=inputs["navigation_ports"],
                 descending_ids=inputs["descending_ids"], drive_ids=drive_ids)
        target_parameters_path = ROOT / freeze["e254_output"] / "target-parameters.npz"
        shutil.copyfile(target_parameters_path, out / "target-parameters.npz")
        target_parameters = e254.read_npz(out / "target-parameters.npz",
                                          ("target_ids", "threshold", "coefficient",
                                           "cap", "membrane_tau", "cell_gain",
                                           "cell_adaptation_tau",
                                           "inverse_adaptation_tau"))
        if not e254.same_bits(target_parameters["target_ids"], TARGETS):
            raise ValueError("E255 fixed target parameter IDs differ")
        dynamic_start_equal(inputs["arms"]["baseline"]["start"],
                            inputs["arms"]["candidate"]["start"])
        for model in ("baseline", "candidate"):
            model_entry = {"requested": True, "entered": False,
                           "completed": False, "setup_unknown_partial": False,
                           "setup_executed_ticks": None,
                           "setup_capture_ticks": None,
                           "status": "requested"}
            ledger["models"][model] = model_entry
            e248.write_json(out / "advance-ledger.json", ledger)
            resource_check(freeze, began, out, ledger)
            model_entry.update(entered=True, setup_unknown_partial=True,
                               status="constructing")
            e248.write_json(out / "advance-ledger.json", ledger)
            source = inputs["arms"][model]
            network, restored, identity_fields, backend = e250._constructor(
                inputs["csr"][model], inputs["tau"], inputs["config"],
                source["start"], expected_identity=source["start"]["model_hash"],
                restore_saved=source["start"])
            if identity_fields:
                raise AssertionError("E255 model identity was transplanted")
            model_entry.update(completed=True, setup_unknown_partial=False,
                               setup_executed_ticks=60, setup_capture_ticks=20,
                               backend=backend, status="ready")
            exact(restored, source["start"], "E255 original model start")
            verify_target_parameters(network, target_parameters)
            if model == "candidate":
                dynamic_start_equal(inputs["arms"]["baseline"]["start"],
                                    network.snapshot())
            e248.write_json(out / f"{model}-binding.json", model_entry)
            e248.write_json(out / "advance-ledger.json", ledger)
            for arm in ("00", "01") if model == "baseline" else ("11", "10"):
                arm_dir = out / arm
                arm_dir.mkdir(exist_ok=False)
                entry = begin_arm(ledger, arm)
                active_dir, active_arm = arm_dir, entry
                e248.write_json(out / "advance-ledger.json", ledger)
                raw = raw_arrays(arm_dir, incoming, len(drive_ids))
                raws[arm] = raw
                reference = (reference_arrays(ROOT / freeze["e254_output"] / model)
                             if arm in ("00", "11") else None)

                def persist():
                    e248.write_json(out / "advance-ledger.json", ledger)

                def budget():
                    resource_check(freeze, began, out, ledger)

                # The observed start is saved after the arm-local exact restore.
                network.restore(source["start"])
                exact(network.snapshot(), source["start"], "E255 pre-arm exact reset")
                StateStore.save(arm_dir / "observed-start", network.snapshot())
                entry["initial_saved"] = True
                persist()
                run_crossed_arm(network, source["start"],
                                inputs["arms"][BINDINGS[arm][1]], incoming,
                                drive_ids, inputs["navigation_ports"],
                                inputs["descending_ids"], raw, entry,
                                persist, budget, reference)
                # Save the computed whole state before any end/diagonal verdict.
                StateStore.save(arm_dir / "final", network.snapshot())
                entry["final_saved"] = True
                if network.tick != CONTROLS * STEPS:
                    raise AssertionError("E255 final neural tick differs")
                entry["diagonal_exact"] = True if reference is not None else None
                entry["status"] = "valid_exact_diagonal" if reference is not None else "complete_cross"
                persist()
                e254.flush_raw(raw)
                active_dir, active_arm = None, None
            network = None
            gc.collect()
            torch.cuda.empty_cache()
        if (sum(x["setup_executed_ticks"] for x in ledger["models"].values()) != 120
                or sum(x["setup_capture_ticks"] for x in ledger["models"].values()) != 40
                or sum(x["completed_controls"] for x in ledger["arms"].values()) != 400
                or sum(x["current_probe_completed"] for x in ledger["arms"].values()) != 404
                or not all(ledger["arms"][arm]["diagonal_exact"] for arm in ("00", "11"))):
            raise AssertionError("E255 exact diagonal/work accounting differs")
        effect_arrays, report = factors(raws, incoming,
                                         target_parameters["threshold"])
        np.savez(out / "factor-effects.npz", **effect_arrays)
        e248.write_json(out / "factor-windows.json", report)
        result.update(validity=VALIDITY, factor_report_available=True,
                      H255_connectivity_effect_retained=report["H255"][
                          "H255_connectivity_effect_retained"],
                      H255=report["H255"], exact_diagonals=True,
                      interpretation="fixed saved-input neural-only cross; no body")
    except Exception as error:  # noqa: BLE001 - preserve partial scientific work
        primary = error
        errors.append({"stage": "science", "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
        if (network is not None and active_dir is not None
                and active_arm is not None and not (active_dir / "final").exists()):
            primary = e254.preserve_partial_state(
                network, active_dir / "partial-final", active_arm, primary)
    finally:
        for raw in raws.values():
            try:
                e254.flush_raw(raw)
            except Exception as error:  # noqa: BLE001
                if primary is None:
                    primary = error
                errors.append({"stage": "raw flush", "error": repr(error)})
        raws.clear()
        network = None
        gc.collect()
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
            raise RuntimeError("E255 invalid: " + str(primary)) from primary


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
