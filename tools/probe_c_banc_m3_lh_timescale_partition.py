"""E256: existing six-edge neural timescale-partition contrast, without a body."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
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

BASE = Path("verification/banc-natural-gait-m3-20260919")
FREEZE_SHA = "abf8af0a97ced6f6884468ab7c53f88c223e2d722136f617c1569dd554bf7d95"
FREEZE_SCHEMA = "flylab.m3.e256.lh-timescale-partition.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e256.lh-timescale-partition.review.v1"
OWNED = ("tools/probe_c_banc_m3_lh_timescale_partition.py",
         "tests/test_c_banc_lh_timescale_partition.py")
VALIDITY = "valid_six_existing_edge_partition_contrast"
BASELINE = e250.BASELINE
CANDIDATE = e250.CANDIDATE
DISK = e250.DISK
N = e250.N
CONTROLS = e250.CONTROLS
STEPS = e250.STEPS
IDENTITY = e250.METADATA


def gate(freeze_path: Path, freeze_sha: str, review_path: Path,
         review_sha: str, out: Path) -> dict:
    """Fail before loading original matrices or constructing a neural model."""
    if (freeze_sha != FREEZE_SHA or e248.sha256(freeze_path) != FREEZE_SHA
            or e248.sha256(review_path) != review_sha):
        raise ValueError("E256 freeze/review SHA differs")
    freeze, review = e248.read_json(freeze_path), e248.read_json(review_path)
    limits = freeze.get("resource_limits", {})
    contract = {"internal_s": 34, "finalization_reserve_s": 5,
                "external_s": 35, "supervisor_s": 37, "cpu_threads": 1,
                "gpu_count": 1, "original_attempts": 1, "retries": 0,
                "neural_constructors": 3, "setup_neural_ticks_per_constructor": 60,
                "setup_capture_ticks_per_constructor": 20,
                "arm_count": 8, "reference_arm_count": 1,
                "controls_per_arm": 40, "neural_steps_per_control": 20,
                "scientific_neural_ticks": 7200, "body_constructors": 0,
                "physics_steps": 0, "fit_calls": 0, "solver_calls": 0,
                "resident_bytes": 6442450944,
                "additional_gpu_bytes": 6442450944,
                "output_bytes": 536870912,
                "minimum_free_bytes": 2147483648,
                "minimum_gpu_free_bytes": 7516192768,
                "baseline_arm_count": 2, "candidate_arm_count": 6,
                "supervisor_wall_s": 37}
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("n") != N or freeze.get("initial_tick") != 40000
            or freeze.get("declared_edges") != 11584852
            or freeze.get("network_identity") !=
            "a31f899cd948a4297a73db66569e5edc2210ac9aa3601aa83e96cb9ff68ce244"
            or freeze.get("graph_identity") !=
            "9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff"
            or freeze.get("baseline_arms") != list(BASELINE)
            or freeze.get("candidate_arms") != list(CANDIDATE)
            or freeze.get("reference_arm") != DISK
            or freeze.get("partition_post_rows") != [111970]
            or freeze.get("partition_edge_count") != 6
            or freeze.get("preserved_previous_edges") != 19
            or freeze.get("constructors") != 3
            or freeze.get("setup_ticks") != 180
            or freeze.get("scientific_ticks") != 7200
            or freeze.get("allocation_increase_s") != 0
            or freeze.get("rate_response_min") != 1e-4
            or freeze.get("interaction_min") != 1e-4
            or freeze.get("interaction_fraction_min") != .2
            or freeze.get("hypothesis") != {
                "R_min": 1e-4, "D_min": 1e-4, "D_over_R_min": .2,
                "Q6_over_Q25_max": .2, "Q_window": [1, 41],
                "core_roster": "original_24_node_rows"}
            or freeze.get("output") != (BASE / "e256-lh-timescale-partition").as_posix()
            or any(limits.get(k) != v for k, v in contract.items())):
        raise ValueError("E256 fixed design/resource contract differs")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("E256 source review not approved")
    pins = {}
    for mapping in (freeze["input_sha256"], freeze["reused_source_sha256"],
                    freeze["preservation_sha256"], review["source_sha256"],
                    review.get("additional_input_sha256", {}),
                    review.get("additional_reused_source_sha256", {}),
                    {freeze["design"]: freeze["design_sha256"]}):
        if not isinstance(mapping, dict):
            raise TypeError("E256 SHA mapping required")
        for name, digest in mapping.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E256 SHA pin: " + name)
            pins[name] = digest
    for name, digest in pins.items():
        path = (ROOT / name).resolve()
        if (not path.is_relative_to(ROOT) or not path.is_file()
                or e248.sha256(path) != digest):
            raise ValueError("E256 pinned byte differs: " + name)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("E256 package version differs: " + name)
    if out.resolve() != (ROOT / BASE / "e256-lh-timescale-partition").resolve():
        raise ValueError("E256 output path differs")
    if out.exists() or not out.parent.is_dir():
        raise FileExistsError("E256 output must be new")
    return freeze


def select_partition(patch: dict, roster: dict, prepared: dict) -> tuple[dict, np.ndarray]:
    """Select the six previously existing edges by frozen timescale membership."""
    ids = np.asarray(patch["edge_ids"], np.int64)
    post = np.asarray(patch["post_ids"], np.int64)
    intrinsic = np.asarray(roster["intrinsic_timescale_node_rows"], np.int64)
    core = np.asarray(roster["core_adaptation_node_rows"], np.int64)
    original24 = np.asarray(roster["original_24_node_rows"], np.int64)
    chosen = ~np.isin(post, intrinsic)
    if (ids.shape != (25,) or len(np.unique(ids)) != 25
            or intrinsic.shape != (38,) or original24.shape != (24,)
            or np.count_nonzero(chosen) != 6
            or set(post[chosen]) != {111970}
            or not np.isin(np.unique(post), core).all()
            or not np.array_equal(np.sort(original24),
                                  np.sort(np.asarray(prepared["original24_rows"])))
            or not np.array_equal(np.sort(intrinsic),
                                  np.sort(np.asarray(prepared["intrinsic_timescale38_rows"])))
            or prepared["selected_post_rows"] != [111970]
            or not np.array_equal(ids[chosen], prepared["selected_edges"]["edge_ids"])
            or not np.array_equal(ids[~chosen], prepared["preserved_edges"])):
        raise ValueError("E256 six/19 precommitted membership differs")
    selected = {key: np.asarray(values)[chosen].copy() for key, values in patch.items()
                if key in ("edge_ids", "pre_ids", "post_ids", "source_values",
                           "target_values", "counts")}
    for key, values in selected.items():
        expected = np.asarray(prepared["selected_edges"][key], dtype=values.dtype)
        if not e248.same_bits(values, expected):
            raise ValueError("E256 prepared six-edge bytes differ: " + key)
    return selected, ids[~chosen].copy()


def patch_six(baseline, selected: dict, preserved_ids: np.ndarray):
    """Change six CSR data values only; all endpoints and other bytes remain."""
    from scipy.sparse import csr_matrix

    if not isinstance(baseline, csr_matrix) or baseline.dtype != np.float32:
        raise TypeError("Original float32 CSR required")
    ids = np.asarray(selected["edge_ids"], np.int64)
    pre = np.asarray(selected["pre_ids"], np.int64)
    post = np.asarray(selected["post_ids"], np.int64)
    old = np.asarray(selected["source_values"], np.float32)
    new = np.asarray(selected["target_values"], np.float32)
    counts = np.asarray(selected["counts"], np.int64)
    if (ids.shape != (6,) or len(np.unique(ids)) != 6
            or any(x.shape != (6,) for x in (pre, post, old, new, counts))
            or np.any(ids < 0) or np.any(ids >= baseline.nnz)
            or np.any(post != 111970) or np.any(counts <= 0)
            or not e248.same_bits(new, (counts.astype(np.float64) * .03).astype(np.float32))
            or not e248.same_bits(old, (new.astype(np.float64) * .001).astype(np.float32))
            or not e248.same_bits(baseline.data[ids], old)
            or not e248.same_bits(baseline.indices[ids].astype(np.int64), pre)
            or not e248.same_bits((np.searchsorted(baseline.indptr, ids,
                                                   side="right") - 1).astype(np.int64), post)):
        raise ValueError("E256 selected existing CSR anatomy/value differs")
    changed = baseline.copy()
    changed.data[ids] = new
    other = np.ones(baseline.nnz, bool)
    other[ids] = False
    if (not e248.same_bits(changed.indptr, baseline.indptr)
            or not e248.same_bits(changed.indices, baseline.indices)
            or not e248.same_bits(changed.data[ids], new)
            or not e248.same_bits(changed.data[other], baseline.data[other])
            or not e248.same_bits(changed.data[preserved_ids],
                                  baseline.data[preserved_ids])):
        raise AssertionError("E256 nonselected CSR byte changed")
    return changed


def q_metrics(baseline: dict, candidate: dict, old25: dict,
              old25_baseline: dict, original24: np.ndarray,
              *, n=N) -> tuple[dict, dict]:
    """Fixed original-24 endpoint RMS and signed change components."""
    original24 = np.asarray(original24, np.int64)
    if original24.shape != (24,) or len(np.unique(original24)) != 24:
        raise ValueError("Frozen original24 neuron rows required")

    def diffs(b, c):
        pieces = []
        for arm in BASELINE:
            source, target = np.asarray(b[arm]), np.asarray(c[arm])
            if (source.shape != (41, n) or target.shape != source.shape
                    or source.dtype != np.float32 or target.dtype != np.float32
                    or not np.isfinite(source).all() or not np.isfinite(target).all()):
                raise ValueError("Complete original24 intact rate pair required")
            pieces.append((target[1:, original24].astype(np.float64)
                           - source[1:, original24].astype(np.float64)) / 200.)
        return np.stack(pieces, axis=0)

    six, all25 = diffs(baseline, candidate), diffs(old25_baseline, old25)
    q6, q25 = (float(np.sqrt(np.mean(x ** 2))) for x in (six, all25))
    summary = {"Q6": q6, "Q25": q25,
               "Q6_over_Q25": q6 / q25 if q25 > 0 else None,
               "Q25_positive": bool(q25 > 0),
               "Q80_protection_pass": bool(q25 > 0 and q6 <= .2 * q25),
               "original24_rows": original24.tolist()}
    for label, values in (("six", six), ("full25", all25)):
        summary[label + "_signed_sum_normalized"] = float(values.sum(dtype=np.float64))
        summary[label + "_positive_sum_normalized"] = float(
            np.maximum(values, 0).sum(dtype=np.float64))
        summary[label + "_negative_sum_normalized"] = float(
            np.minimum(values, 0).sum(dtype=np.float64))
        summary[label + "_min_normalized"] = float(values.min())
        summary[label + "_max_normalized"] = float(values.max())
    return summary, {"original24_rows": original24, "six_normalized": six,
                     "full25_normalized": all25}


def _manifest(out: Path, limit: int) -> dict:
    files, size = {}, 0
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name in ("result.json", "output-manifest.json"):
            continue
        bytes_ = path.stat().st_size
        size += bytes_
        if size > limit:
            raise OSError("E256 output bytes exceeded")
        files[path.relative_to(out).as_posix()] = {
            "sha256": e248.sha256(path), "bytes": bytes_}
    return {"schema": "flylab.m3.e256.output-manifest.v1", "bytes": size,
            "files": files}


def finish_receipts(out: Path, freeze: dict, ledger: dict, result: dict,
                    errors: list, primary, started: float,
                    *, persist=e248.write_json):
    """No top-level, nested, disk, or Q verdict remains valid after failure."""
    limit = freeze["resource_limits"]["internal_s"]
    result["work"] = {"constructors_requested": len(ledger["constructors"]),
        "constructors_entered": sum(bool(x["entered"]) for x in ledger["constructors"]),
        "constructors_completed": sum(bool(x["completed"]) for x in ledger["constructors"]),
        "setup_ticks_completed": ledger["setup_ticks_completed"],
        "setup_capture_ticks_completed": ledger["setup_capture_ticks_completed"],
        "science_ticks_requested": ledger["science_ticks_requested"],
        "science_ticks_completed": ledger["science_ticks_completed"],
        "arms_requested": len(ledger["arms"]),
        "arms_completed": sum(bool(x["completed"]) for x in ledger["arms"].values()),
        "unknown_partial_arms": [k for k, x in ledger["arms"].items()
                                 if x["unknown_partial"]],
        "unknown_partial_constructors": [x["name"] for x in ledger["constructors"]
                                         if x["unknown_partial"]]}

    def fail(error, stage):
        nonlocal primary
        if primary is None:
            primary = error
        errors.append({"stage": stage, "type": type(error).__name__,
                       "message": str(error)})
        result.update(validity="resource_limited" if isinstance(primary, TimeoutError)
                      else "invalid_or_incomplete", H256_supported=False,
                      future_exact=False, finalization_pending=True,
                      first_error={"type": type(primary).__name__,
                                   "message": str(primary)})
        if isinstance(result.get("metrics"), dict):
            revoke_support(result["metrics"])

    def revoke_support(metric):
        metric.update(H256_supported=False,
                      candidate_disk_future_exact=False,
                      candidate_sensory_cut_non_source_exact=False,
                      candidate_bridge_cut_nonbridge_exact=False,
                      interpretation_status="unvalidated_incomplete")
        if isinstance(metric.get("candidate_response"), dict):
            metric["candidate_response"]["H248_supported"] = False
        if isinstance(metric.get("Q"), dict):
            metric["Q"]["Q80_protection_pass"] = False

    def revoke_metrics():
        path = out / "metrics.json"
        if path.exists():
            saved = e248.read_json(path)
            revoke_support(saved)
            persist(path, saved)

    if primary is not None:
        fail(primary, "prior science/cleanup")
    for name, value in (("advance-ledger.json", ledger),
                        ("errors.json", {"errors": errors})):
        try:
            persist(out / name, value)
        except Exception as error:  # noqa: BLE001
            fail(error, name)
    if time.perf_counter() - started >= limit:
        fail(TimeoutError("E256 internal deadline including publication"), "deadline")
    if primary is not None:
        try:
            revoke_metrics()
        except Exception as error:  # noqa: BLE001
            fail(error, "metrics revocation")
            (out / "metrics.json").unlink(missing_ok=True)
    try:
        persist(out / "errors.json", {"errors": errors})
        persist(out / "output-manifest.json",
                _manifest(out, freeze["resource_limits"]["output_bytes"]))
        result["elapsed_s"] = time.perf_counter() - started
        if result["elapsed_s"] >= limit:
            raise TimeoutError("E256 result publication deadline")
        result["finalization_pending"] = False
        persist(out / "result.json", result)
        if time.perf_counter() - started >= limit:
            raise TimeoutError("E256 late result deadline")
    except Exception as error:  # noqa: BLE001
        fail(error, "publication")
        (out / "result.json").unlink(missing_ok=True)
        (out / "output-manifest.json").unlink(missing_ok=True)
        try:
            revoke_metrics()
        except Exception as secondary:  # noqa: BLE001
            fail(secondary, "metrics revocation")
            (out / "metrics.json").unlink(missing_ok=True)
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
    from tools.probe_c_banc_m3_core_rhythm import source_matrix
    from tools.verify_c_banc_observation import exact

    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    ledger = e250.new_ledger()
    ledger["schema"] = "flylab.m3.e256.lh-timescale-partition.ledger.v1"
    ledger["setup_capture_ticks_completed"] = 0
    result = {"schema": "flylab.m3.e256.lh-timescale-partition.result.v1",
              "validity": "incomplete", "H256_supported": False,
              "candidate_adopted": False, "whole_goal_pass": False,
              "F6_exposed": False, "active_UI_changed": False,
              "future_exact": False, "finalization_pending": True,
              "original_attempts": 1}
    errors, primary = [], None
    network, active_path, active_key = None, None, None
    try:
        result["cpu_runtime"] = e248._configure_cpu()
        e248._check_space(freeze, out)
        baseline, tau, config, saved, lh_rows = e250._load_inputs(freeze)
        source = np.asarray(freeze["selection"]["source_rows"], np.int64)
        bridge = np.asarray(freeze["selection"]["bridge_rows"], np.int64)
        low, high = (float(freeze["selection"][name])
                     for name in ("train_steady_min", "train_steady_max"))
        patch = e248.read_json(ROOT / freeze["paths"]["edge_preparation"])
        roster = e248.read_json(ROOT / freeze["paths"]["roster"])
        prepared = e248.read_json(ROOT / freeze["paths"]["partition_preparation"])
        if (patch.get("graph_hash") != freeze["graph_identity"]
                or roster.get("graph_hash") != freeze["graph_identity"]
                or prepared.get("graph_hash") != freeze["graph_identity"]):
            raise ValueError("E256 original graph/roster identity differs")
        for key, values in freeze["patch"].items():
            actual = np.asarray(patch[key])
            pinned = np.asarray(values, dtype=actual.dtype)
            if not e248.same_bits(actual, pinned):
                raise ValueError("E256 frozen 25-edge table differs: " + key)
        e250.verify_bundle_edges(freeze, patch)
        selected, preserved_ids = select_partition(patch, roster, prepared)
        candidate = patch_six(baseline, selected, preserved_ids)
        np.savez(out / "edge-patch.npz", **selected, preserved_edge_ids=preserved_ids)
        np.savez(out / "candidate-csr.npz", indptr=candidate.indptr,
                 indices=candidate.indices, data=candidate.data)
        e248.write_json(out / "partition.json", {
            "schema": "flylab.m3.e256.partition.v1",
            "rule": prepared["rule"], "selected_post_rows": [111970],
            "selected_edge_ids": selected["edge_ids"].tolist(),
            "preserved19_edge_ids": preserved_ids.tolist(),
            "all_bridge_core_adaptation": True,
            "outside_intrinsic_timescale38": True,
            "original24_not_biological_label": True})
        e248.write_json(out / "selection.json", {
            "source_rows": source.tolist(), "bridge_rows": bridge.tolist(),
            "lh_free_motor_rows": lh_rows.tolist(),
            "original24_rows": roster["original_24_node_rows"],
            "low32": float(np.float32(low)),
            "high32": float(np.float32(high))})

        def construct(name, weights, expected=None, restore_saved=None):
            nonlocal network
            e248._check_live_resources(freeze, started, out)
            entry = {"name": name, "requested": True, "entered": False,
                     "completed": False, "unknown_partial": False,
                     "setup_ticks_completed": None,
                     "setup_capture_ticks_completed": None}
            ledger["constructors"].append(entry)
            e248.write_json(out / "advance-ledger.json", ledger)
            entry["entered"] = True
            entry["unknown_partial"] = True
            try:
                e248.write_json(out / "advance-ledger.json", ledger)
            except Exception:
                # A failed pre-call receipt must not claim a CUDA constructor.
                entry["entered"] = False
                entry["unknown_partial"] = False
                raise
            network, state, fields, backend = e250._constructor(
                weights, tau, config, saved, expected_identity=expected,
                restore_saved=restore_saved)
            entry.update(completed=True, unknown_partial=False,
                         setup_ticks_completed=60,
                         setup_capture_ticks_completed=20,
                         actual_backend=backend, model_hash=network.identity)
            ledger["setup_ticks_completed"] += 60
            ledger["setup_capture_ticks_completed"] += 20
            e248.write_json(out / "advance-ledger.json", ledger)
            return state, fields

        original_state, _ = construct("fresh_baseline", baseline,
                                      freeze["network_identity"])
        StateStore.save(out / "baseline-neural-start", original_state)
        exact(StateStore.load(out / "baseline-neural-start"), saved)
        for arm in BASELINE:
            e248._check_live_resources(freeze, started, out)
            active_path = out / "baseline" / arm
            active_key = "baseline:" + arm
            e250.run_arm(network, original_state, arm, active_path, ledger,
                         source, bridge, low, high, started, scope="baseline")
            old = ROOT / freeze["paths"]["e248_output"] / arm
            if not e248.same_bits(np.load(active_path / "rate.npy", mmap_mode="r"),
                                  np.load(old / "rate.npy", mmap_mode="r")):
                raise AssertionError("E256 fresh baseline full-rate future differs")
            exact(StateStore.load(active_path / "final"), StateStore.load(old / "final"))
            active_path = active_key = None
        network = None
        gc.collect()

        candidate_state, identity_fields = construct("six_edge_candidate", candidate)
        StateStore.save(out / "candidate-neural-start", candidate_state)
        exact(StateStore.load(out / "candidate-neural-start"), candidate_state)
        e248.write_json(out / "provenance.json", {
            "original_identity": freeze["network_identity"],
            "candidate_identity": network.identity,
            "identity_fields": identity_fields,
            "changed_edge_count": 6,
            "actual_csr_data_sha256": hashlib.sha256(
                np.ascontiguousarray(source_matrix(network).data).tobytes()).hexdigest(),
            "baseline_csr_data_sha256": hashlib.sha256(
                np.ascontiguousarray(baseline.data).tobytes()).hexdigest(),
            "initial_tick": saved["tick"], "body_constructors": 0})
        for arm in CANDIDATE:
            e248._check_live_resources(freeze, started, out)
            active_path = out / "candidate" / arm
            active_key = "candidate:" + arm
            e250.run_arm(network, candidate_state, arm, active_path, ledger,
                         source, bridge, low, high, started, scope="candidate")
            active_path = active_key = None
        network = None
        gc.collect()

        disk = StateStore.load(out / "candidate-neural-start")
        exact(disk, candidate_state)
        _, _ = construct("disk_candidate_reference", candidate,
                         disk["model_hash"], restore_saved=disk)
        e248._check_live_resources(freeze, started, out)
        active_path = out / "candidate" / DISK
        active_key = "candidate:" + DISK
        e250.run_arm(network, disk, DISK, active_path, ledger,
                     source, bridge, low, high, started, scope="candidate")
        active_path = active_key = None
        network = None
        gc.collect()

        b = {name: np.load(out / "baseline" / name / "rate.npy", mmap_mode="r")
             for name in BASELINE}
        c = {name: np.load(out / "candidate" / name / "rate.npy", mmap_mode="r")
             for name in (*CANDIDATE, DISK)}
        old = ROOT / freeze["paths"]["e248_output"]
        for arm in ("sensory_cut_low", "sensory_cut_high"):
            if not e248.same_bits(c[arm], np.load(old / arm / "rate.npy", mmap_mode="r")):
                raise AssertionError("E256 sensory-cut whole-neural future differs")
            current_final = StateStore.load(out / "candidate" / arm / "final")
            old_final = StateStore.load(old / arm / "final")
            normalized = copy.deepcopy(current_final)
            for key in IDENTITY:
                normalized[key] = old_final[key]
            exact(normalized, old_final)
        for arm in ("bridge_cut_low", "bridge_cut_high"):
            other = np.ones(N, bool)
            other[bridge] = False
            reference = np.load(old / arm / "rate.npy", mmap_mode="r")
            if not e248.same_bits(c[arm][:, other], reference[:, other]):
                raise AssertionError("E256 bridge-cut nonbridge rate differs")
            actual_final = StateStore.load(out / "candidate" / arm / "final")
            previous_final = StateStore.load(old / arm / "final")
            for key in ("rate", "adaptation"):
                if not e248.same_bits(actual_final[key][other], previous_final[key][other]):
                    raise AssertionError("E256 bridge-cut nonbridge final differs")
        cut = e248.cut_validation(c["sensory_cut_low"], c["sensory_cut_high"], source)
        disk_exact = e248.same_bits(c["intact_low"], c[DISK])
        exact(StateStore.load(out / "candidate" / "intact_low" / "final"),
              StateStore.load(out / "candidate" / DISK / "final"))
        if not cut["non_source_bitexact"] or not disk_exact:
            raise AssertionError("E256 outgoing cut/disk future differs")
        for name, values in (*b.items(), *c.items()):
            if (not np.isfinite(values).all() or np.any(values < -1e-4)
                    or np.any(values > 200.001)):
                raise AssertionError("E256 physiological rate bounds differ: " + name)

        response = e248.response_metrics(c, lh_rows,
            response_min=freeze["rate_response_min"],
            interaction_min=freeze["interaction_min"],
            fraction_min=freeze["interaction_fraction_min"])
        baseline_delta = (b["intact_high"][1:, lh_rows].astype(np.float64)
                          - b["intact_low"][1:, lh_rows].astype(np.float64)) / 200.
        full25 = ROOT / freeze["paths"]["e250_output"]
        old_b = {name: np.load(full25 / "baseline" / name / "rate.npy", mmap_mode="r")
                 for name in BASELINE}
        old_c = {name: np.load(full25 / "candidate" / name / "rate.npy", mmap_mode="r")
                 for name in BASELINE}
        for arm in BASELINE:
            if not e248.same_bits(old_b[arm], b[arm]):
                raise AssertionError("E256 E250 saved baseline differs")
        q, q_raw = q_metrics(b, c, old_c, old_b,
                             np.asarray(roster["original_24_node_rows"], np.int64))
        np.savez(out / "q-comparison.npz", **q_raw)
        full25_metrics = e248.read_json(full25 / "metrics.json")
        metrics = {"schema": "flylab.m3.e256.metrics.v1",
                   "candidate_response": response,
                   "baseline_response_R": float(np.sqrt(np.mean(baseline_delta ** 2))),
                   "baseline_interaction_D": e248.read_json(old / "metrics.json")[
                       "interaction_D"],
                   "previous_full25_response_R": full25_metrics["response_R"],
                   "previous_full25_interaction_D": full25_metrics["interaction_D"],
                   "Q": q,
                   "candidate_disk_future_exact": disk_exact,
                   "candidate_sensory_cut_non_source_exact": cut["non_source_bitexact"],
                   "candidate_bridge_cut_nonbridge_exact": True,
                   "cut": cut,
                   "H256_supported": bool(response["H248_supported"]
                                          and q["Q80_protection_pass"]),
                   "interpretation": "200ms body-free neural response only"}
        e248.write_json(out / "metrics.json", metrics)
        result.update(validity=VALIDITY, H256_supported=metrics["H256_supported"],
                      future_exact=True, metrics=metrics)
        if (len(ledger["constructors"]) != 3
                or ledger["setup_ticks_completed"] != 180
                or ledger["setup_capture_ticks_completed"] != 60
                or len(ledger["arms"]) != 9
                or ledger["science_ticks_completed"] != 7200):
            raise AssertionError("E256 original work count differs")
    except Exception as error:  # noqa: BLE001
        primary = error
        errors.append({"type": type(error).__name__, "message": str(error),
                       "traceback": traceback.format_exc()})
        if (network is not None and active_path is not None
                and not (active_path / "final").exists()):
            try:
                StateStore.save(active_path / "partial-final", network.snapshot())
                if active_key is not None and active_key in ledger["arms"]:
                    ledger["arms"][active_key]["partial_full_state_saved"] = True
            except Exception as secondary:  # noqa: BLE001
                errors.append({"stage": "partial-final", "type": type(secondary).__name__,
                               "message": str(secondary)})
        result["first_error"] = {"type": type(primary).__name__,
                                 "message": str(primary)}
    finally:
        network = None
        try:
            gc.collect()
        except Exception as cleanup_error:  # noqa: BLE001
            if primary is None:
                primary = cleanup_error
            errors.append({"stage": "cleanup", "type": type(cleanup_error).__name__,
                           "message": str(cleanup_error)})
        primary = finish_receipts(out, freeze, ledger, result, errors, primary, started)
        if primary is not None:
            raise RuntimeError("E256 invalid: " + str(primary)) from primary


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
