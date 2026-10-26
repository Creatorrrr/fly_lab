"""E266 conditional saved-current comparison. No neural or body model calls."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
FREEZE_SHA = "9df241422ea8a5daf7b8b941331ef9628ed57c042c3add6fd8f6a0026cd609b9"
PREFIX = "flylab.m3.e266.opponent-current"
OWNED = ["tools/analyze_c_banc_m3_opponent_current.py", "tests/test_c_banc_opponent_current.py"]
SIX = [8191121, 8191220, 8191247, 8191292, 8191340, 8191396]
LIMITS = {"cpu_threads": 1, "gpu_processes": 0, "original_attempts": 1, "retries": 0,
              "internal_s": 5., "external_child_s": 8., "supervisor_wall_s": 10.,
              "process_memory_bytes": 1073741824, "output_bytes": 16777216,
              "minimum_free_disk_bytes": 67108864}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pinned(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Slash-separated repository path required")
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("Unsafe pin")
    p = (ROOT / p).resolve()
    if not p.is_relative_to(ROOT) or not p.is_file():
        raise ValueError("Missing or external pin")
    return p


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, allow_nan=False, sort_keys=True, indent=2) + "\n", "utf-8")


def gate(freeze_path, freeze_sha, review_path, review_sha, check=lambda: None):
    if freeze_sha != FREEZE_SHA or digest(freeze_path) != FREEZE_SHA:
        raise ValueError("Frozen E266 identity differs")
    if digest(review_path) != review_sha:
        raise ValueError("Review bytes differ")
    f, r = read(freeze_path), read(review_path)
    contract = {"schema": PREFIX + ".freeze.v1", "owned_files": OWNED, "target": 111970,
                    "six_edge_ids": SIX, "opponent_edge_ids": [8191320], "startup_controls": [1, 99],
                    "fixed_cases": [43, 56, 70], "ratio_limit": .2, "undefined_epsilon": 1e-12,
                    "resource_limits": LIMITS}
    if any(f.get(k) != v for k, v in contract.items()):
        raise ValueError("Frozen contract differs")
    if (r.get("schema") != PREFIX + ".review.v1" or r.get("approved") is not True
            or r.get("blocking_defects") != [] or r.get("freeze_sha256") != FREEZE_SHA
            or r.get("source_sha256") != {p: digest(pinned(p)) for p in OWNED}):
        raise ValueError("Approved root source review required")
    if set(f["paths"].values()) != set(f["input_sha256"]):
        raise ValueError("Input role pins differ")
    maps = [f["input_sha256"], f["reused_source_sha256"],
            {f["design"]: f["design_sha256"]},
            r.get("additional_input_sha256", {}), r.get("additional_reused_source_sha256", {})]
    for mapping in maps:
        if not isinstance(mapping, dict):
            raise TypeError("Pin map required")
        for name, sha in mapping.items():
            if digest(pinned(name)) != sha:
                raise ValueError("Pin differs: " + name)
            check()
    if f["packages"] != {k: importlib.metadata.version(k) for k in f["packages"]}:
        raise ValueError("Package version differs")
    return f


def load_inputs(f, arrays, check=lambda: None):
    p = {k: pinned(v) for k, v in f["paths"].items()}
    with np.load(p["selected_input"], allow_pickle=False) as z:
        arrays.update({k: z[k].copy() for k in z.files})
    validate_arrays(arrays)
    check()
    with np.load(p["row_spec"], allow_pickle=False) as z:
        row = {k: z[k] for k in z.files}
    cols, ids, pres = arrays["original_row_columns"], arrays["edge_ids"], arrays["pre_ids"]
    if row["edge_ids"].shape != (434,) or row["target_id"].item() != 111970:
        raise ValueError("Incoming row identity differs")
    for selected, full in [("edge_ids", "edge_ids"), ("pre_ids", "pre_ids"),
                           ("post_ids", "post_ids"), ("W0", "weights_W0"),
                           ("W6", "weights_W6"), ("six_mask", "six_mask")]:
        if not np.array_equal(arrays[selected], row[full][cols]):
            raise ValueError("Selected row differs: " + selected)
    for role, key in [("incoming_rate", "rate"), ("incoming_mask", "mask")]:
        full = np.load(p[role], mmap_mode="r", allow_pickle=False)
        if full.shape != (101, 434) or not np.array_equal(full[:, cols], arrays[key]):
            raise ValueError("Saved B subset differs")
    bundle = {k: np.load(p["bundle_" + k], mmap_mode="r", allow_pickle=False)
              for k in ("indptr", "indices", "weights", "counts")}
    if not (np.all(ids >= bundle["indptr"][111970]) and np.all(ids < bundle["indptr"][111971])
            and np.array_equal(bundle["indices"][ids], pres)
            and np.array_equal(bundle["counts"][ids], arrays["counts"])):
        raise ValueError("Bundle endpoint/count differs")
    restored = (np.sign(bundle["weights"][ids]).astype(np.float64)
                * arrays["counts"].astype(np.float64) * .03).astype(np.float32)
    if not np.all(restored > 0) or not np.array_equal(restored, arrays["W7"]):
        raise ValueError("Bundle sign/restoration differs")
    annotation = read(p["selected_annotations"])
    roster = read(p["roster"])
    ports = [next(x for x in roster["ports"] if x["port"] == port) for port in (85, 86)]
    if annotation["ports"] != ports:
        raise ValueError("Roster selection differs")
    for port, cell, count in zip(ports, ("SNpp50", "SNpp51"), (23, 4)):
        if (port["leg"], port["kind"], port["cell_type"], len(port["node_rows"])) != ("lh", "claw", cell, count):
            raise ValueError("Port annotation differs")
    nodes = {}
    wanted = set(pres.tolist())
    with p["bundle_nodes"].open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if i in wanted:
                nodes[str(i)] = json.loads(line)
            if i % 4096 == 0:
                check()
    if nodes != annotation["nodes"]:
        raise ValueError("Original annotations differ")
    for i, pre in enumerate(pres):
        node, port = nodes[str(pre)], ports[0 if arrays["six_mask"][i] else 1]
        if (int(pre) not in port["node_rows"] or node["cell_type"] != port["cell_type"]
                or node["soma_side"] != "left" or node["nt_type"] != "ACH"
                or node["source_annotations"]["body_part_sensory"] != "hind_leg"):
            raise ValueError("Cell membership differs")
    prior = read(p["upstream_final_review"])
    if prior.get("all_passed") is not True or prior.get("run_validity") != "valid_fixed_input_bridge_current":
        raise ValueError("Valid upstream root review required")
    manifest = read(p["upstream_manifest"])
    for role, name in [("incoming_rate", "B/incoming-rate.npy"), ("incoming_mask", "B/incoming-mask.npy")]:
        if manifest["files"][name]["sha256"] != digest(p[role]):
            raise ValueError("Upstream manifest pin differs")
    sup = read(p["upstream_supervisor"])
    if sup["cumulative_after_s"] != f["original_cumulative_s"] or sup["cumulative_limit_s"] != f["original_cumulative_limit_s"]:
        raise ValueError("Upstream accounting differs")
    check()


def validate_arrays(a):
    specs = {"control_tick": ((101,), "i"), "time_s": ((101,), "f"),
             "rate": ((101, 7), "f"), "mask": ((101, 7), "f")}
    specs.update({k: ((7,), "i") for k in ("edge_ids", "pre_ids", "post_ids", "counts", "original_row_columns")})
    specs.update({k: ((7,), "f") for k in ("W0", "W6", "W7")})
    specs.update({k: ((7,), "b") for k in ("six_mask", "opponent_mask")})
    for k, (shape, kind) in specs.items():
        x = a[k]
        if x.shape != shape or x.dtype.kind != kind or not np.isfinite(x).all():
            raise ValueError("Required finite array differs: " + k)
    s, o = a["six_mask"], a["opponent_mask"]
    if (not np.array_equal(a["control_tick"], np.arange(101))
            or not np.array_equal(a["time_s"], np.arange(101) * .005)
            or not np.array_equal(o, ~s) or s.sum() != 6
            or a["edge_ids"][s].tolist() != SIX or a["edge_ids"][o].tolist() != [8191320]
            or a["pre_ids"][o].tolist() != [128621] or not np.all(a["post_ids"] == 111970)
            or not np.all(a["counts"] > 0) or a["counts"][o].tolist() != [2]
            or not np.all(a["rate"] >= 0) or not np.all((a["mask"] >= 0) & (a["mask"] <= 1))
            or not np.array_equal(a["W6"][s], a["W7"][s])
            or not np.array_equal(a["W6"][o], a["W0"][o])
            or not np.all(np.diff(a["original_row_columns"]) > 0)):
        raise ValueError("Selection/alignment contract differs")


def stats(x):
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
                "rms": float(np.sqrt(np.mean(x * x))), "maxabs": float(np.max(np.abs(x))),
                "positive": int(np.sum(x > 0)), "negative": int(np.sum(x < 0)), "zero": int(np.sum(x == 0))}


def compute(a):
    validate_arrays(a)
    effective = a["rate"].astype(np.float64) * a["mask"].astype(np.float64)
    currents = {name: effective * a[name].astype(np.float64) for name in ("W0", "W6", "W7")}
    six = effective * (a["W6"].astype(np.float64) - a["W0"].astype(np.float64))
    seven = effective * (a["W7"].astype(np.float64) - a["W0"].astype(np.float64))
    j6, j7 = six.sum(axis=1), seven.sum(axis=1)
    jopp = j7 - j6
    biases = np.array([-j6[1:100].mean(), -j7[1:100].mean()], np.float64)
    k6, k7 = j6 + biases[0], j7 + biases[1]
    arrays = {"effective_rate": effective, "six_edge_current": six, "seven_edge_current": seven,
                  "edge_delta6": six, "edge_delta7": seven,
                  "opponent_edge_current": seven - six, "J6": j6, "J7": j7, "Jopp": jopp,
                  "K6": k6, "K7": k7, "bias64": biases, "bias32": biases.astype(np.float32)}
    arrays.update({"current_" + k: v for k, v in currents.items()})
    startup = {k: stats(v[1:100]) for k, v in arrays.items() if k in ("J6", "J7", "Jopp", "K6", "K7")}
    rms6, rms7 = startup["K6"]["rms"], startup["K7"]["rms"]
    ratio = None if rms6 <= 1e-12 else rms7 / rms6
    verdict = None if ratio is None else ratio <= .2 and startup["K7"]["maxabs"] <= startup["K6"]["maxabs"]
    x, y = j6[1:100], jopp[1:100]
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    report = {"startup": startup, "covariance_J6_Jopp": covariance, "covariance_definition": "population; controls1..99",
                  "ratio": ratio, "H266_original_count_balance": verdict,
                  "H266_classification": "inconclusive" if verdict is None else "supported" if verdict else "refuted",
                  "fixed_cases": {str(c): {k: float(arrays[k][c]) for k in ("J6", "J7", "Jopp", "K6", "K7")}
                               for c in (43, 56, 70)},
                  "all_endpoints": {k: stats(arrays[k]) for k in ("J6", "J7", "Jopp", "K6", "K7")}}
    return arrays, report


def empty_result():
    return {"schema": "flylab.m3.e266.result.v1", "validity": "invalid",
                "H266_original_count_balance": None, "H266_classification": "not_tested", "analysis": None,
                "whole_goal_pass": False, "candidate_adopted": False, "F6_exposed": False, "active_UI_changed": False}


def run(args):
    start = time.perf_counter()
    out = Path(args.out).resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise ValueError("Fresh repository output directory required")
    out.mkdir(parents=True)
    arrays, errors, result = {}, [], empty_result()
    ledger = {"schema": PREFIX + ".ledger.v1", "original_attempts": 1, "retries": 0,
                  "neural_calls": 0, "body_calls": 0, "cuda_calls": 0, "physics_steps": 0, "fit_calls": 0,
                  "solver_calls": 0, "cpu_threads": 1, "calculation_completed": False}

    def check():
        if time.perf_counter() - start > 4.:
            raise TimeoutError("Computation deadline; reserve one second for publication")

    def record(exc, stage):
        errors.append({"stage": stage, "type": type(exc).__name__, "message": str(exc)})
        result.update(validity="invalid", H266_original_count_balance=None,
                      H266_classification="not_tested", analysis=None)

    try:
        with threadpool_limits(limits=1):
            f = gate(args.freeze, args.freeze_sha256, args.review, args.review_sha256, check)
            if out != (ROOT / f["output"]).resolve():
                raise ValueError("Frozen output differs")
            if shutil.disk_usage(out).free < LIMITS["minimum_free_disk_bytes"]:
                raise OSError("Minimum disk reserve unavailable")
            load_inputs(f, arrays, check)
            derived, report = compute(arrays)
            arrays.update(derived)
            ledger["calculation_completed"] = True
            check()
            result.update(validity="valid", analysis=report,
                          H266_original_count_balance=report["H266_original_count_balance"],
                          H266_classification=report["H266_classification"])
    except Exception as exc:  # noqa: BLE001 - preserve the first analysis failure.
        record(exc, "analysis")
    try:
        np.savez(out / "arrays.npz", **arrays)
    except Exception as exc:  # noqa: BLE001 - array publication invalidates the result.
        record(exc, "arrays-publication")
    ledger["wall_s"] = time.perf_counter() - start
    try:
        write(out / "ledger.json", ledger)
        write(out / "errors.json", errors)
        write(out / "result.json", result)
        files = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in out.iterdir() if p.is_file()}
        write(out / "output-manifest.json", {"schema": PREFIX + ".manifest.v1", "files": files,
                                                "bytes": sum(v["bytes"] for v in files.values())})
        if sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > LIMITS["output_bytes"]:
            raise OSError("Output byte limit exceeded")
        if time.perf_counter() - start > 5.:
            raise TimeoutError("Internal deadline exceeded")
    except Exception as exc:  # noqa: BLE001 - late publication must revoke success.
        record(exc, "publication")
        # Remove stale canonical success BEFORE trying any recovery writes.
        # Quarantined bytes are evidence only and are never a canonical result.
        for name in ("result.json", "output-manifest.json"):
            path = out / name
            if path.exists():
                try:
                    path.replace(out / (name + ".unverified-raw"))
                except OSError as secondary:
                    errors.append({"stage": "quarantine", "type": type(secondary).__name__, "message": str(secondary)})
                    # A failed quarantine must still invalidate stale success.
                    path.unlink(missing_ok=True)
        # Withdraw every canonical/nested success, including late manifest errors.
        for name, value in [("result.json", result), ("errors.json", errors), ("ledger.json", ledger)]:
            try:
                write(out / name, value)
            except Exception as secondary:  # noqa: BLE001 - best-effort error publication.
                errors.append({"stage": "recovery", "type": type(secondary).__name__, "message": str(secondary)})
        try:
            files = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in out.iterdir()
                     if p.is_file() and p.name != "output-manifest.json"}
            write(out / "output-manifest.json", {"schema": PREFIX + ".manifest.v1", "validity": "invalid", "files": files})
        except Exception as secondary:  # noqa: BLE001 - no success survives this failure.
            errors.append({"stage": "manifest-recovery", "type": type(secondary).__name__, "message": str(secondary)})
            path = out / "output-manifest.json"
            if path.exists():
                try:
                    path.replace(out / "output-manifest.json.recovery-unverified-raw")
                except OSError:
                    path.unlink(missing_ok=True)
    return 0 if result["validity"] == "valid" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "freeze-sha256", "review", "review-sha256", "out"):
        p.add_argument("--" + key, required=True)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
