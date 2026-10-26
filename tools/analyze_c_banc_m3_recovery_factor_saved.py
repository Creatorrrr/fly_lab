"""E298 unmasked same-state recovery-factor proxy; no new dynamics."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import analyze_c_banc_m3_observed_transfer_residual as prior

p = prior.p
OWNED = ("tools/analyze_c_banc_m3_recovery_factor_saved.py", "tests/test_c_banc_recovery_factor_saved.py")
SOURCE_FILES = tuple(dict.fromkeys((*OWNED, *prior.SOURCE_FILES)))
INPUT_NAMES = prior.INPUT_NAMES
FREEZE_SCHEMA = "flylab.m3.e298.recovery-factor-saved.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e298.recovery-factor-saved.review.v1"
OUTPUT_PATH = "verification/banc-natural-gait-m3-20260919/e298-recovery-factor-saved"
LIMITS = {"wall_s": 45, "host_peak_bytes": 2 << 30, "output_bytes": 128 << 20, "cpu_threads": 1}
WINDOWS = prior.WINDOWS
FLAGS = {**prior.FLAGS, "biological_claim": False, "gait_improved": False, "new_trajectory_verified": False,
         "new_model_nondegeneracy_verified": False, "kernel_implemented": False}


def recovery_factor(rate, resource, utilization, tau):
    """Explicit float64 arithmetic at saved unmasked rate/R, float32 factors."""
    if not isinstance(rate, np.ndarray) or rate.ndim not in (1, 2) or not rate.size:
        raise ValueError("Nonempty compatible source array required")
    p.array(rate, np.float32, rate.shape, "Saved source rates")
    p.array(resource, np.float64, rate.shape, "Saved source resource")
    for values, name in ((utilization, "Utilization"), (tau, "Recovery tau")):
        p.array(values, np.float64, (rate.shape[-1],), name)
    if (np.any(rate < 0) or np.any(resource < 0) or np.any(resource > 1)
            or np.any(utilization < 0) or np.any(tau <= 0)):
        raise ValueError("Nonnegative rates/u, bounded R and positive tau required")
    if not np.all(resource[..., utilization == 0] == 1):
        raise ValueError("u0 source resource must be exactly one")
    r = rate.astype(np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        a = resource * (1 + (tau * utilization) * r)
        rdot = (1 - resource) / tau - utilization * r * resource
        g64 = np.minimum(1., a)
        g32, old32 = g64.astype(np.float32), resource.astype(np.float32)
        np.testing.assert_allclose(a, 1 - tau * rdot, rtol=4e-15, atol=4e-15)
        np.testing.assert_allclose(g64, 1 - tau * np.maximum(rdot, 0), rtol=4e-15, atol=4e-15)
        if (np.any(g64 < resource) or np.any(g64 > 1) or np.any(g32 < old32) or np.any(g32 > 1)):
            raise ValueError("Recovery factor bounds violated")
        # Both conversions are round-to-nearest float32, including subnormals.
        for exact, rounded in ((g64, g32), (resource, old32)):
            allowance = np.spacing(rounded).astype(np.float64) * .5
            if np.any(np.abs(rounded.astype(np.float64) - exact) > allowance):
                raise ValueError("Float32 rounding bound violated")
        old_loss = r * (1 - old32.astype(np.float64))
        new_loss = r * (1 - g32.astype(np.float64))
    result = {"a": a, "g64": g64, "g32": g32, "old32": old32, "Rdot": rdot,
              "old_source_loss": old_loss, "new_source_loss": new_loss, "removed_source_loss": old_loss - new_loss}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite recovery-factor output")
    return result


def saved_proxy(roster, activity, core_ids, csr):
    ids = activity["observed_ids"]
    prior.prior.id_columns(ids, ids)
    order = np.argsort(ids)
    checked = {**activity, "observed_ids": ids[order]}
    for key in ("rate", "drive", "adaptation"):
        checked[key] = activity[key][:, order]
    posts, _, controls = p.checked_activity(roster, checked)
    prior.prior.id_columns(posts, core_ids)
    for key, value in (("utilization", .05), ("recovery_tau", .150)):
        p.array(roster[key], np.float64, roster["source_ids"].shape, key)
        if not np.isin(roster[key], [0., value] if key == "utilization" else [value]).all():
            raise ValueError("Frozen E296 u/tau required")
    indptr, indices, weights = (csr[key] for key in ("indptr", "indices", "data"))
    if (not isinstance(indptr, np.ndarray) or indptr.dtype != np.int32 or indptr.ndim != 1 or len(indptr) < 2
            or not isinstance(indices, np.ndarray) or indices.dtype != np.int32 or indices.ndim != 1):
        raise ValueError("Native original CSR required")
    p.array(weights, np.float32, indices.shape, "Original CSR weights")
    if (indptr[0] != 0 or indptr[-1] != len(indices) or np.any(np.diff(indptr) < 0)
            or np.any(indices < 0) or np.any(indices >= len(indptr) - 1) or np.any(roster["edge_ids"] >= len(indices))):
        raise ValueError("Invalid original CSR bounds")
    edges = roster["edge_ids"]
    p.equal_array(indices[edges].astype(np.int64), roster["pre_ids"], "Original selected presynaptic IDs")
    p.equal_array((np.searchsorted(indptr, edges, side="right") - 1).astype(np.int64), roster["post_ids"], "Original selected postsynaptic IDs")
    p.equal_array(weights[edges], roster["weights32"], "Original selected coefficient bytes")
    slots, w = roster["edge_source_slots"], roster["weights32"]
    core_edges = np.isin(roster["post_ids"], core_ids)
    if np.any(roster["utilization"][slots[(w < 0) | core_edges]] != 0):
        raise ValueError("Negative/core-afferent sources must be exempt")
    rate = activity["rate"][:, prior.prior.id_columns(ids, roster["source_ids"])]
    resource = activity["synaptic_resource"]
    factors = recovery_factor(rate, resource, roster["utilization"], roster["recovery_tau"])
    if not np.all(factors["g32"][:, slots[(w < 0) | core_edges]] == 1):
        raise ValueError("Negative/core selected factor must remain exact one")
    source = {"source_ids": roster["source_ids"].copy(), "control_tick": controls.copy(),
              "physics_time_s": activity["physics_time_s"].copy(), "rate": rate.copy(), "resource": resource.copy(),
              "utilization": roster["utilization"].copy(), "tau": roster["recovery_tau"].copy(), **factors}
    target = {"post_ids": posts.copy(), "core_ids": core_ids.copy(), "control_tick": controls.copy(),
              "physics_time_s": activity["physics_time_s"].copy()}
    chosen = w > 0
    post_slots = np.searchsorted(posts, roster["post_ids"][chosen])
    for name in ("old", "new"):
        values = factors[name + "_source_loss"][:, slots[chosen]]
        weighted = values * w[chosen].astype(np.float64)
        target[name + "_positive_loss"] = np.stack([np.bincount(post_slots, weights=row, minlength=len(posts)) for row in weighted])
    target["removed_positive_loss"] = target["old_positive_loss"] - target["new_positive_loss"]
    if np.any(target["new_positive_loss"] > target["old_positive_loss"]) or any(not np.isfinite(value).all() for value in target.values()):
        raise ValueError("Finite monotone unmasked proxy loss required")
    return source, target


def fixed_summary(source, target):
    controls = source["control_tick"]
    p.equal_array(controls, np.arange(101, dtype=np.int64), "Actual0..100 controls")
    active = source["utilization"] > 0
    if not active.any():
        raise ValueError("Active source denominator required")
    windows = []
    for first, last in ((0, 0), (1, 100), *WINDOWS):
        rows = (controls >= first) & (controls <= last)
        g = source["g32"][rows][:, active]
        components = {"one_minus_g32": 1 - g.astype(np.float64),
                      "g64_minus_R": source["g64"][rows][:, active] - source["resource"][rows][:, active],
                      **{key: source[key][rows][:, active] for key in ("old_source_loss", "new_source_loss", "removed_source_loss")}}
        posts = {}
        for group, members in (("core", np.isin(target["post_ids"], target["core_ids"])), ("other", ~np.isin(target["post_ids"], target["core_ids"]))):
            posts[group] = {key: {"sum": float(np.sum(target[key][rows][:, members])), "mean": float(np.mean(target[key][rows][:, members])),
                "maximum": float(np.max(target[key][rows][:, members]))} for key in ("old_positive_loss", "new_positive_loss", "removed_positive_loss")}
        windows.append({"first_control": first, "last_control": last, "samples": int(rows.sum()), "active_sources": int(active.sum()),
            "unity_fraction_active_only": float(np.mean(g == 1)), "suppressed_source_endpoints": int(np.count_nonzero(g < 1)),
            "any_suppressed_sources": int(np.count_nonzero(np.any(g < 1, axis=0))),
            "components": {key: {"mean": float(np.mean(value)), "maximum": float(np.max(value)), "sum": float(np.sum(value))} for key, value in components.items()},
            "target_groups": posts})
    return windows


def nontrivial_flags(source, target):
    active = source["utilization"] > 0
    factor = bool(np.any(source["g32"][1:, active] < 1))
    current = bool(np.any(target["new_positive_loss"][1:] > 0))
    return {"factor_nontrivial": factor, "current_proxy_nontrivial": current,
            "source_current_proxy_nontrivial": bool(np.any(source["new_source_loss"][1:, active] > 0)),
            "active_controls1_100_all_unity": not factor, "observed_saved_proxy_degenerate": not current}


def gate(freeze_path, freeze_sha, review_path, review_sha, output):
    if p.sha256(freeze_path) != freeze_sha or p.sha256(review_path) != review_sha:
        raise ValueError("E298 freeze/review SHA mismatch")
    freeze, review = p.read_json(freeze_path), p.read_json(review_path)
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("owned_files") != list(OWNED)
            or set(freeze.get("inputs", {})) != set(INPUT_NAMES) or freeze.get("resource_limits") != LIMITS
            or freeze.get("windows") != [list(w) for w in WINDOWS] or freeze.get("output") != OUTPUT_PATH):
        raise ValueError("Exact E298 fixed saved-input freeze required")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("Approved matching E298 source review required")
    names = set()
    for name in freeze["inputs"].values():
        path = prior.safe(name)
        files = [file for file in path.rglob("*") if file.is_file()] if path.is_dir() else [path]
        if not files:
            raise ValueError("Missing frozen E298 input")
        names.update(file.relative_to(ROOT).as_posix() for file in files)
    if set(freeze["input_sha256"]) != names or set(freeze["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("Exact all-input/source SHA set required")
    prior.pin_bytes([freeze["input_sha256"], freeze["source_sha256"], review["source_sha256"], {freeze["design"]: freeze["design_sha256"]}])
    receipts = {key: p.read_json(prior.safe(name)) for key, name in freeze["inputs"].items() if key not in prior.NPZ_INPUTS and key != "candidate_final_checkpoint"}
    prior.provenance(freeze["inputs"], receipts, freeze["input_sha256"])
    if output.resolve() != prior.safe(OUTPUT_PATH) or output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh fixed E298 output required")
    return freeze


def run(freeze, output):
    import psutil
    from threadpoolctl import threadpool_limits
    began = time.perf_counter()
    output = Path(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh E298 output required")
    def budget():
        memory = psutil.Process().memory_info()
        if time.perf_counter() - began >= LIMITS["wall_s"]:
            raise TimeoutError("E298 saved-analysis deadline")
        if getattr(memory, "peak_wset", memory.rss) > LIMITS["host_peak_bytes"]:
            raise MemoryError("E298 host memory cap")
    output.mkdir()
    try:
        p.write_json(output / "source-manifest.json", {"schema": "flylab.m3.e298.sources.v1", "owned_files": list(OWNED),
            "source_sha256": freeze.get("source_sha256", {}), "input_sha256": freeze.get("input_sha256", {})})
        with threadpool_limits(limits=1):
            budget()
            data = prior.load(freeze["inputs"])
            prior.provenance(freeze["inputs"], data, freeze["input_sha256"])
            roster, activity, core = data["protected_roster"], data["candidate_activity"], data["baseline_core_activity"]["core_ids"]
            source, target = saved_proxy(roster, activity, core, data["baseline_csr"])
            active = roster["utilization"] > 0
            if (len(source["control_tick"]) != 101 or len(source["source_ids"]) != 7136 or len(target["post_ids"]) != 7136
                    or len(core) != 24 or len(roster["edge_ids"]) != 725849 or active.sum() != 1513
                    or np.count_nonzero(active[roster["edge_source_slots"]]) != 117973
                    or np.count_nonzero(np.isin(roster["post_ids"], core)) != 6172):
                raise ValueError("Exact original E296 roster/active/core-afferent counts required")
            summaries = fixed_summary(source, target)
            result = {"schema": "flylab.m3.e298.recovery-factor-saved.result.v1", "analysis_valid": True, "summaries": summaries,
                **nontrivial_flags(source, target),
                "units": "model input units; not amperes", "scope": "unmasked same-state proxy using E296 rate/R only",
                "mask_scope": "intermediate masks unobserved; no per-tick unity measurement claimed",
                "future_unity_branch": "factor32==1 must use original packed CSR expression; future kernel not implemented",
                "constant_input_scope": "analytical property requires R>=Rinf; low arbitrary saved R may suppress even constant rate",
                "stability_scope": "bounded factors and scalar resource dt do not establish coupled network stability",
                "saved_original_cost": data["e296_result"]["cost"],
                "execution": {key: 0 for key in ("original", "model", "neural", "body", "physics", "CUDA", "solver", "fit")}, **FLAGS}
            budget()
            if sum(array.nbytes for values in (source, target) for array in values.values()) > LIMITS["output_bytes"]:
                raise OSError("E298 raw output cap")
            np.savez_compressed(output / "source-recovery-factor.npz", **source)
            np.savez_compressed(output / "target-positive-loss.npz", **target)
            budget()
            result["elapsed_s"] = time.perf_counter() - began
            memory = psutil.Process().memory_info()
            result["host_peak_bytes"] = getattr(memory, "peak_wset", memory.rss)
            p.write_json(output / "result.json", result)
            p.write_json(output / "output-manifest.json", {"schema": "flylab.m3.e298.output-manifest.v1", "files": {
                file.name: {"bytes": file.stat().st_size, "sha256": p.sha256(file)} for file in output.iterdir() if file.is_file()}})
            budget()
            if sum(file.stat().st_size for file in output.iterdir()) > LIMITS["output_bytes"]:
                raise OSError("E298 published output cap")
            return result
    except Exception as error:
        (output / "result.json").unlink(missing_ok=True)
        (output / "output-manifest.json").unlink(missing_ok=True)
        p.write_json(output / "failure.json", {"analysis_valid": False, "first_error": str(error), "elapsed_s": time.perf_counter() - began,
            "execution": {key: 0 for key in ("original", "model", "neural", "body", "physics", "CUDA", "solver", "fit")}, **FLAGS})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "source-review", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    for key in ("freeze-sha256", "source-review-sha256"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    freeze = gate(args.freeze, args.freeze_sha256, args.source_review, args.source_review_sha256, args.output)
    old = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    try:
        for key in old:
            os.environ[key] = "1"
        run(freeze, args.output)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
