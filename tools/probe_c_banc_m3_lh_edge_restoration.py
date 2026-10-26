"""E250: one body-free restoration of 25 existing LH sensory synapses."""

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

from tools import probe_c_banc_m3_lh_neural_response as prior

BASE = Path("verification/banc-natural-gait-m3-20260919")
FREEZE_SHA = "e21401a1eb65ab4a32dc04860de96241d7c0dbf8c633c1dee253146682f305be"
FREEZE_SCHEMA = "flylab.m3.e250.lh-edge-restoration.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e250.lh-edge-restoration.review.v1"
VALIDITY = "valid_existing_edge_restoration_contrast"
METADATA = ("model_hash", "core_adaptation", "motor_adaptation", "motor_timescale")
BASELINE = ("intact_low", "intact_high")
CANDIDATE = prior.ARM_NAMES
DISK = prior.REFERENCE
N = prior.N
STEPS = prior.STEPS
CONTROLS = prior.CONTROLS


def gate(freeze_path: Path, freeze_sha: str, review_path: Path, review_sha: str) -> dict:
    """Verify all frozen bytes and packages before loading scientific arrays."""
    if (freeze_sha != FREEZE_SHA or prior.sha256(freeze_path) != freeze_sha
            or prior.sha256(review_path) != review_sha):
        raise ValueError("E250 freeze/review hash differs")
    freeze, review = prior.read_json(freeze_path), prior.read_json(review_path)
    if (freeze.get("schema") != FREEZE_SCHEMA or review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != freeze_sha
            or set(review.get("source_sha256", {})) != set(freeze["owned_files"])):
        raise ValueError("E250 source review differs")
    pins = {}
    for group in (review["source_sha256"], freeze["input_sha256"],
                  freeze["reused_source_sha256"], freeze["preservation_sha256"],
                  review.get("additional_input_sha256", {}),
                  review.get("additional_reused_source_sha256", {}),
                  {freeze["design"]: freeze["design_sha256"]}):
        for name, digest in group.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting SHA pin: " + name)
            pins[name] = digest
    for name, digest in pins.items():
        if prior.sha256(ROOT / name) != digest:
            raise ValueError("E250 pinned byte differs: " + name)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("E250 package differs: " + name)
    limits = freeze["resource_limits"]
    if (freeze["n"] != N or freeze["initial_tick"] != 40000
            or freeze["declared_edges"] != 11584852
            or tuple(freeze["baseline_arms"]) != BASELINE
            or tuple(freeze["candidate_arms"]) != CANDIDATE
            or freeze["reference_arm"] != DISK
            or limits["neural_constructors"] != 3
            or limits["scientific_neural_ticks"] != 7200
            or limits["setup_neural_ticks_per_constructor"] != 60
            or freeze["output"] != (BASE / "e250-lh-edge-restoration").as_posix()):
        raise ValueError("E250 fixed work contract differs")
    return freeze


def patch_existing_edges(weights, patch: dict, source_rows, bridge_rows):
    """Replace only the frozen CSR data entries with original rate units."""
    from scipy.sparse import csr_matrix

    if not isinstance(weights, csr_matrix) or weights.dtype != np.float32:
        raise ValueError("Canonical float32 CSR required")
    ids = np.asarray(patch["edge_ids"], dtype=np.int64)
    pre = np.asarray(patch["pre_ids"], dtype=np.int64)
    post = np.asarray(patch["post_ids"], dtype=np.int64)
    old = np.asarray(patch["source_values"], dtype=np.float32)
    new = np.asarray(patch["target_values"], dtype=np.float32)
    counts = np.asarray(patch["counts"], dtype=np.int64)
    if (ids.shape != (25,) or len(np.unique(ids)) != 25
            or any(a.shape != ids.shape for a in (pre, post, old, new, counts))
            or np.any(ids < 0) or np.any(ids >= weights.nnz)
            or not np.isin(pre, source_rows).all()
            or not np.isin(post, bridge_rows).all()
            or not np.all(old > 0) or not np.all(counts > 0)
            or not prior.same_bits(new, (counts.astype(np.float64) * .03).astype(np.float32))
            or not prior.same_bits(old, (new.astype(np.float64) * .001).astype(np.float32))):
        raise ValueError("Frozen 25-edge anatomy or original rate values differ")
    actual_post = np.searchsorted(weights.indptr, ids, side="right") - 1
    if (not prior.same_bits(actual_post.astype(np.int64), post)
            or not prior.same_bits(weights.indices[ids].astype(np.int64), pre)
            or not prior.same_bits(weights.data[ids], old)):
        raise ValueError("Frozen 25-edge CSR endpoints/current values differ")
    changed = weights.copy()
    changed.data[ids] = new
    if (not prior.same_bits(changed.indptr, weights.indptr)
            or not prior.same_bits(changed.indices, weights.indices)
            or not prior.same_bits(changed.data[ids], new)):
        raise AssertionError("CSR topology or selected target differs")
    other = np.ones(weights.nnz, bool)
    other[ids] = False
    if not prior.same_bits(changed.data[other], weights.data[other]):
        raise AssertionError("Nonselected synapse changed")
    return changed


def verify_bundle_edges(freeze: dict, patch: dict) -> None:
    """Check each selected endpoint/count/sign in the pinned original bundle."""
    paths = freeze["paths"]
    indptr = np.load(ROOT / paths["bundle_indptr"], mmap_mode="r")
    indices = np.load(ROOT / paths["bundle_indices"], mmap_mode="r")
    weights = np.load(ROOT / paths["bundle_weights"], mmap_mode="r")
    counts = np.load(ROOT / paths["bundle_counts"], mmap_mode="r")
    ids = np.asarray(patch["edge_ids"], np.int64)
    pre = np.asarray(patch["pre_ids"], np.int64)
    post = np.asarray(patch["post_ids"], np.int64)
    original_counts = np.asarray(patch["counts"], np.int64)
    targets = np.asarray(patch["target_values"], np.float32)
    if (indptr.shape != (N + 1,) or indices.shape != weights.shape
            or counts.shape != weights.shape or np.any(ids >= len(indices))
            or not prior.same_bits((np.searchsorted(indptr, ids, side="right") - 1).astype(np.int64), post)
            or not prior.same_bits(np.asarray(indices[ids], np.int64), pre)
            or not prior.same_bits(np.asarray(counts[ids], np.int64), original_counts)
            or not np.all(np.asarray(weights[ids]) > 0)
            or not prior.same_bits((np.sign(weights[ids]) * counts[ids] * .03).astype(np.float32),
                                   targets)):
        raise ValueError("Original bundle signed-count provenance differs")


def transplant_identity(saved: dict, constructor_snapshot: dict) -> tuple[dict, dict]:
    """Transfer only newly generated hash-chain metadata, never dynamic state."""
    from tools.verify_c_banc_observation import exact

    if set(saved) != set(constructor_snapshot) or any(k not in saved for k in METADATA):
        raise ValueError("Complete same-schema native snapshots required")
    candidate = copy.deepcopy(saved)
    changed = {}
    for key in METADATA:
        before, after = saved[key], constructor_snapshot[key]
        if key == "model_hash":
            if before == after:
                raise ValueError("New CSR identity must differ")
            changed[key] = ["model_hash"]
        else:
            identity_field = {"core_adaptation": "base_model_hash",
                              "motor_adaptation": "source_core_identity",
                              "motor_timescale": "source_motor_identity"}[key]
            if set(before) != set(after):
                raise ValueError("Native layer metadata fields changed")
            old_values = {k: v for k, v in before.items() if k != identity_field}
            new_values = {k: v for k, v in after.items() if k != identity_field}
            exact(old_values, new_values)
            changed[key] = [identity_field] if before[identity_field] != after[identity_field] else []
        candidate[key] = copy.deepcopy(after)
    source_dynamic = {k: v for k, v in saved.items() if k not in METADATA}
    target_dynamic = {k: v for k, v in candidate.items() if k not in METADATA}
    exact(source_dynamic, target_dynamic)
    return candidate, changed


def new_ledger() -> dict:
    return {"schema": "flylab.m3.e250.lh-edge-restoration.ledger.v1",
            "constructors": [], "arms": {}, "setup_ticks_completed": 0,
            "science_ticks_requested": 0, "science_ticks_completed": 0,
            "body_constructors": 0, "physics_steps": 0, "fit_calls": 0,
            "solver_calls": 0, "retry_count": 0}


def run_arm(network, saved, arm, out, ledger, source_ids, bridge_ids,
            low, high, started, *, scope="candidate", persist=prior.write_json):
    """Journal intent and retain every completed row before later validation."""
    from flylab.c.storage import StateStore
    from tools.verify_c_banc_observation import exact

    ledger_key = f"{scope}:{arm}"
    ledger_path = out.parent.parent / "advance-ledger.json"
    if ledger_key in ledger["arms"]:
        raise ValueError("Repeated arm")
    entry = {"requested": True, "entered": False, "completed": False,
             "requested_controls": CONTROLS, "entered_controls": 0,
             "completed_controls": 0, "unknown_partial": False,
             "initial_restored_exact": False}
    ledger["arms"][ledger_key] = entry
    ledger["science_ticks_requested"] += CONTROLS * STEPS
    persist(ledger_path, ledger)
    entry["entered"] = True
    entry["unknown_partial"] = True
    persist(ledger_path, ledger)
    network.restore(saved)
    exact(network.snapshot(), saved)
    entry["initial_restored_exact"] = True
    entry["unknown_partial"] = False
    cut = "sensory" if arm.startswith("sensory") else "bridge" if arm.startswith("bridge") else "intact"
    value = high if arm.endswith("high") else low
    mask = prior.select_mask(saved["output_mask"], source_ids, bridge_ids, cut)
    drive = prior.held_drive(saved["drive"], source_ids, value)
    new_zeros = source_ids if cut == "sensory" else bridge_ids if cut == "bridge" else np.empty(0, np.int64)
    network.set_muted(np.unique(np.r_[np.flatnonzero(saved["output_mask"] == 0), new_zeros]))
    if not prior.same_bits(network.output_mask.detach().cpu().numpy(), mask):
        raise AssertionError("Actual outgoing mask differs")
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "drive.npy", drive, allow_pickle=False)
    np.save(out / "mask.npy", mask, allow_pickle=False)
    raw = np.lib.format.open_memmap(out / "rate.npy", mode="w+", dtype=np.float32,
                                    shape=(CONTROLS + 1, N))
    raw[:] = np.nan
    raw[0] = network.readout()
    raw.flush()
    if not prior.same_bits(raw[0], saved["rate"]):
        raise AssertionError("Restored initial full rate differs")
    persist(ledger_path, ledger)
    try:
        for control in range(1, CONTROLS + 1):
            if time.perf_counter() - started >= 29:
                raise TimeoutError("E250 science deadline")
            entry["entered_controls"] = control
            entry["unknown_partial"] = True
            persist(ledger_path, ledger)
            network.advance(drive, STEPS)
            raw[control] = network.readout()
            raw.flush()
            entry["completed_controls"] = control
            entry["unknown_partial"] = False
            ledger["science_ticks_completed"] += STEPS
            persist(ledger_path, ledger)
        final = network.snapshot()
        if (final["tick"] != saved["tick"] + CONTROLS * STEPS
                or not prior.same_bits(final["drive"], drive)
                or not prior.same_bits(final["output_mask"], mask)
                or not np.isfinite(final["adaptation"]).all()
                or np.any(final["adaptation"] < 0)):
            raise AssertionError("Native final clock/drive/mask/adaptation differs")
        StateStore.save(out / "final", final)
        entry["completed"] = True
        persist(ledger_path, ledger)
        return final
    finally:
        raw.flush()
        del raw


def _constructor(weights, tau, config, saved, *, expected_identity=None,
                 restore_saved=None):
    from tools.banc_m3_motor_timescale import MotorTimescaleNetwork
    from tools.probe_c_banc_m3_core_rhythm import source_matrix
    from tools.verify_c_banc_observation import exact

    ones = np.ones(N, np.float32)
    network = MotorTimescaleNetwork(weights, tau, ones,
        np.full(N, 7.5, np.float32), np.full(N, 200., np.float32),
        adaptation_gain=8., adaptation_tau_s=.3, dt=.00025,
        device="cuda", cuda_implementation="packed", capture_steps=20,
        core_ids=saved["core_adaptation"]["ids"],
        core_gain=saved["core_adaptation"]["gain"],
        motor_ids=saved["motor_adaptation"]["ids"],
        motor_gains=saved["motor_adaptation"]["gains"],
        motor_log_tau=saved["motor_timescale"]["log_tau"])
    backend = prior.actual_backend(network)
    if expected_identity is not None and network.identity != expected_identity:
        raise AssertionError("Constructed model identity differs")
    actual = source_matrix(network)
    if (not prior.same_bits(actual.indptr, weights.indptr)
            or not prior.same_bits(actual.indices, weights.indices)
            or not prior.same_bits(actual.data, weights.data)):
        raise AssertionError("Actual model CSR bytes differ")
    packed = packed_csr_receipt(network, weights)
    if (not prior.same_bits(network._cell_gain.detach().cpu().numpy(), config["cell_gain"])
            or not prior.same_bits(network.tau.detach().cpu().numpy(), config["membrane_tau"])):
        raise AssertionError("Neural parameter tensor differs")
    state = network.snapshot()
    if restore_saved is not None:
        restored, identity_fields = restore_saved, {}
    elif expected_identity is None:
        restored, identity_fields = transplant_identity(saved, state)
    else:
        restored, identity_fields = saved, {}
    network.restore(restored)
    exact(network.snapshot(), restored)
    return network, restored, identity_fields, {**backend, "packed_csr": packed}


def packed_csr_receipt(network, weights) -> dict:
    """Check the actual fused-kernel buffers, including lookup bit patterns."""
    rowptr = network._rowptr.detach().cpu().numpy()
    entries = network._columns.detach().cpu().numpy().view(np.uint32)
    lookup = network._weight_values.detach().cpu().numpy()
    if (rowptr.dtype != np.int32 or entries.shape != weights.indices.shape
            or lookup.dtype != np.float32
            or not prior.same_bits(rowptr.astype(np.int64), weights.indptr.astype(np.int64))
            or not prior.same_bits((entries & np.uint32((1 << 18) - 1)).astype(np.int64),
                                   weights.indices.astype(np.int64))
            or np.any((entries >> np.uint32(18)) >= len(lookup))
            or not prior.same_bits(lookup[entries >> np.uint32(18)], weights.data)):
        raise AssertionError("Actual packed CUDA CSR buffers differ")
    return {"rowptr_sha256": hashlib.sha256(np.ascontiguousarray(rowptr).tobytes()).hexdigest(),
            "entries_sha256": hashlib.sha256(np.ascontiguousarray(entries).tobytes()).hexdigest(),
            "lookup_sha256": hashlib.sha256(np.ascontiguousarray(lookup).tobytes()).hexdigest(),
            "decoded_data_sha256": hashlib.sha256(
                np.ascontiguousarray(lookup[entries >> np.uint32(18)]).tobytes()).hexdigest(),
            "lookup_unique_values": len(lookup), "edges": len(entries)}


def _load_inputs(freeze):
    from scipy.sparse import csr_matrix

    from flylab.c.storage import StateStore
    from tools.banc_m3_intrinsic_recruitment import patch_tau

    path = lambda key: ROOT / freeze["paths"][key]
    with np.load(path("native_csr"), allow_pickle=False) as file:
        indptr, indices, data = (file[k].copy() for k in ("indptr", "indices", "data"))
    if (data.dtype != np.float32 or len(data) != freeze["declared_edges"]
            or indptr.shape != (N + 1,) or indices.shape != data.shape
            or int(indptr[-1]) != len(data)):
        raise ValueError("Exact E150 CSR required")
    weights = csr_matrix((data, indices, indptr), shape=(N, N), dtype=np.float32)
    weights.check_format(full_check=True)
    with np.load(path("configuration"), allow_pickle=False) as file:
        config = {key: file[key].copy() for key in file.files}
    with np.load(path("calibration"), allow_pickle=False) as file:
        calibration = {key: file[key].copy() for key in file.files}
    tau = patch_tau(np.full(N, .02, np.float32),
                    calibration["tau_ids"], calibration["tau_values"])
    if not prior.same_bits(tau, config["membrane_tau"]):
        raise ValueError("Native membrane tau differs")
    saved = StateStore.load(path("checkpoint"))["network"]
    if (saved["model_hash"] != freeze["network_identity"]
            or saved["tick"] != freeze["initial_tick"] or saved["device"] != "cuda"
            or saved["dt"] != .00025):
        raise ValueError("Frozen full neural checkpoint differs")
    selection = prior.read_json(path("selection"))["selected"]
    if selection != freeze["selection"]:
        raise ValueError("Frozen structural selection differs")
    groups = prior.read_json(path("groups"))["groups"]
    with np.load(path("parameters"), allow_pickle=False) as file:
        motor_ids, free = file["motor_ids"].copy(), file["free_columns"].copy()
    lh_columns = sorted({int(column) for group in groups if group["leg"] == "lh"
                         for column in group["columns"] if column in free})
    lh_rows = motor_ids[lh_columns]
    if (len(lh_rows) != 56 or not prior.same_bits(motor_ids, config["motor_ids"])):
        raise ValueError("LH motor roster differs")
    return weights, tau, config, saved, lh_rows


def _manifest(out: Path, limit: int) -> dict:
    files, total = {}, 0
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name in ("result.json", "output-manifest.json"):
            continue
        size = path.stat().st_size
        total += size
        if total > limit:
            raise RuntimeError("E250 output bytes exceeded")
        files[path.relative_to(out).as_posix()] = {"sha256": prior.sha256(path), "bytes": size}
    return {"schema": "flylab.m3.e250.output-manifest.v1", "bytes": total,
            "files": files}


def finish_receipts(out, freeze, ledger, result, errors, primary, started,
                    persist=prior.write_json):
    """Publish failure-closed and retain the earliest scientific error."""
    limit = freeze["resource_limits"]["internal_s"]
    result["work"] = {"constructors_requested": len(ledger["constructors"]),
        "constructors_completed": sum(bool(x["completed"]) for x in ledger["constructors"]),
        "setup_ticks_completed": ledger["setup_ticks_completed"],
        "science_ticks_requested": ledger["science_ticks_requested"],
        "science_ticks_completed": ledger["science_ticks_completed"],
        "arms_requested": len(ledger["arms"]),
        "arms_completed": sum(bool(x["completed"]) for x in ledger["arms"].values()),
        "arms_unknown_partial": [k for k, x in ledger["arms"].items() if x["unknown_partial"]]}

    def fail(error, label):
        nonlocal primary
        if primary is None:
            primary = error
        errors.append({"type": type(error).__name__, "message": f"{label}: {error}"})
        result.update(validity="resource_limited" if isinstance(primary, TimeoutError)
                      else "invalid_or_incomplete", H250_supported=False,
                      future_exact=False, finalization_pending=True,
                      first_error={"type": type(primary).__name__, "message": str(primary)})

    def revoke():
        p = out / "metrics.json"
        if p.exists():
            m = prior.read_json(p)
            m.setdefault("H250_measured_condition_met", m.get("H250_supported"))
            m.update(H250_supported=False,
                     interpretation_status="not_tested_invalid_or_incomplete")
            persist(p, m)

    for label, action in (("ledger", lambda: persist(out / "advance-ledger.json", ledger)),
                          ("errors", lambda: persist(out / "errors.json", {"errors": errors}))):
        try:
            action()
        except Exception as error:  # noqa: BLE001
            fail(error, label)
    result["elapsed_s"] = time.perf_counter() - started
    if result["elapsed_s"] >= limit:
        fail(TimeoutError("E250 internal deadline including publication"), "deadline")
    if primary is not None:
        try:
            revoke()
        except Exception as error:  # noqa: BLE001
            fail(error, "metrics revocation")
            (out / "metrics.json").unlink(missing_ok=True)
    try:
        persist(out / "errors.json", {"errors": errors})
        persist(out / "output-manifest.json", _manifest(out, freeze["resource_limits"]["output_bytes"]))
        result["elapsed_s"] = time.perf_counter() - started
        if result["elapsed_s"] >= limit:
            raise TimeoutError("E250 result publication deadline")
        result["finalization_pending"] = False
        persist(out / "result.json", result)
        if time.perf_counter() - started >= limit:
            raise TimeoutError("E250 final result deadline")
    except Exception as error:  # noqa: BLE001
        fail(error, "publication")
        # A successful result might already be on disk when the final clock
        # crosses the deadline. Remove it before any fallible invalid rewrite.
        (out / "result.json").unlink(missing_ok=True)
        (out / "output-manifest.json").unlink(missing_ok=True)
        try:
            revoke()
        except Exception as secondary:  # noqa: BLE001
            fail(secondary, "metrics revocation")
            (out / "metrics.json").unlink(missing_ok=True)
        try:
            persist(out / "errors.json", {"errors": errors})
        except Exception as secondary:  # noqa: BLE001
            fail(secondary, "errors receipt")
            (out / "errors.json").unlink(missing_ok=True)
        try:
            persist(out / "result.json", result)
        except Exception as secondary:  # noqa: BLE001
            fail(secondary, "result retry")
            (out / "result.json").unlink(missing_ok=True)
            # Record the added secondary fault when storage permits. If it
            # does not, no stale success/manifest is left to imply validity.
            try:
                persist(out / "errors.json", {"errors": errors})
            except Exception as tertiary:  # noqa: BLE001
                fail(tertiary, "errors retry")
                (out / "errors.json").unlink(missing_ok=True)
    return primary


def execute(freeze: dict, out: Path) -> None:
    from flylab.c.storage import StateStore
    from tools.probe_c_banc_m3_core_rhythm import source_matrix
    from tools.verify_c_banc_observation import exact

    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    ledger = new_ledger()
    result = {"schema": "flylab.m3.e250.lh-edge-restoration.result.v1",
        "validity": "incomplete", "H250_supported": False,
        "candidate_adopted": False, "F6_exposed": False,
        "future_exact": False, "first_error": None,
        "finalization_pending": True, "original_attempts": 1}
    errors, primary = [], None
    prior.write_json(out / "advance-ledger.json", ledger)
    prior.write_json(out / "result.json", result)
    network = None
    try:
        result["cpu_runtime"] = prior._configure_cpu()
        prior._check_space(freeze, out)
        baseline, tau, config, saved, lh_rows = _load_inputs(freeze)
        source = np.asarray(freeze["selection"]["source_rows"], np.int64)
        bridge = np.asarray(freeze["selection"]["bridge_rows"], np.int64)
        low, high = (float(freeze["selection"][name])
                     for name in ("train_steady_min", "train_steady_max"))
        patch = prior.read_json(ROOT / freeze["paths"]["edge_preparation"])
        if patch["graph_hash"] != freeze["graph_identity"]:
            raise ValueError("Original graph identity differs")
        verify_bundle_edges(freeze, patch)
        candidate = patch_existing_edges(baseline, patch, source, bridge)
        np.savez(out / "edge-patch.npz", edge_ids=np.asarray(patch["edge_ids"], np.int64),
                 pre_ids=np.asarray(patch["pre_ids"], np.int64),
                 post_ids=np.asarray(patch["post_ids"], np.int64),
                 source_values=np.asarray(patch["source_values"], np.float32),
                 target_values=np.asarray(patch["target_values"], np.float32))
        np.savez(out / "candidate-csr.npz", indptr=candidate.indptr,
                 indices=candidate.indices, data=candidate.data)
        prior.write_json(out / "selection.json", {"port": freeze["selection"]["port"],
            "source_rows": source.tolist(), "bridge_rows": bridge.tolist(),
            "lh_free_motor_rows": lh_rows.tolist(), "low32": float(np.float32(low)),
            "high32": float(np.float32(high))})

        def construct(name, weights, expected=None, restore_saved=None):
            if time.perf_counter() - started >= 29:
                raise TimeoutError("E250 constructor deadline")
            entry = {"name": name, "requested": True, "entered": False,
                     "completed": False, "unknown_partial": False,
                     "setup_ticks_completed": None}
            ledger["constructors"].append(entry)
            prior.write_json(out / "advance-ledger.json", ledger)
            entry["entered"] = True
            entry["unknown_partial"] = True
            prior.write_json(out / "advance-ledger.json", ledger)
            instance, state, fields, backend = _constructor(weights, tau, config, saved,
                                                             expected_identity=expected,
                                                             restore_saved=restore_saved)
            entry.update(completed=True, unknown_partial=False,
                         setup_ticks_completed=60, actual_backend=backend,
                         model_hash=instance.identity)
            ledger["setup_ticks_completed"] += 60
            prior.write_json(out / "advance-ledger.json", ledger)
            return instance, state, fields

        network, original_state, _ = construct("fresh_baseline", baseline,
                                                freeze["network_identity"])
        StateStore.save(out / "baseline-neural-start", original_state)
        exact(StateStore.load(out / "baseline-neural-start"), saved)
        for name in BASELINE:
            prior._check_live_resources(freeze, started, out)
            run_arm(network, original_state, name, out / "baseline" / name, ledger,
                    source, bridge, low, high, started, scope="baseline")
            existing = ROOT / freeze["paths"]["e248_output"] / name
            if not prior.same_bits(np.load(out / "baseline" / name / "rate.npy", mmap_mode="r"),
                                   np.load(existing / "rate.npy", mmap_mode="r")):
                raise AssertionError("Fresh baseline full-rate future differs")
            exact(StateStore.load(out / "baseline" / name / "final"),
                  StateStore.load(existing / "final"))
        network = None
        gc.collect()

        network, candidate_state, identity_fields = construct("candidate", candidate)
        StateStore.save(out / "candidate-neural-start", candidate_state)
        exact(StateStore.load(out / "candidate-neural-start"), candidate_state)
        prior.write_json(out / "provenance.json", {
            "original_identity": freeze["network_identity"],
            "candidate_identity": network.identity,
            "identity_fields": identity_fields, "changed_edge_count": 25,
            "actual_csr_data_sha256": hashlib.sha256(
                np.ascontiguousarray(source_matrix(network).data).tobytes()).hexdigest(),
            "baseline_csr_data_sha256": hashlib.sha256(
                np.ascontiguousarray(baseline.data).tobytes()).hexdigest(),
            "initial_tick": saved["tick"], "body_constructors": 0})
        for name in CANDIDATE:
            prior._check_live_resources(freeze, started, out)
            run_arm(network, candidate_state, name, out / "candidate" / name,
                    ledger, source, bridge, low, high, started)
        network = None
        gc.collect()

        # Disk checkpoint, not an in-memory reuse of the candidate start.
        disk = StateStore.load(out / "candidate-neural-start")
        exact(disk, candidate_state)
        network, _, _ = construct("disk_candidate_reference", candidate,
                                   disk["model_hash"], restore_saved=disk)
        prior._check_live_resources(freeze, started, out)
        run_arm(network, disk, DISK, out / "candidate" / DISK,
                ledger, source, bridge, low, high, started)
        network = None
        gc.collect()

        b = {name: np.load(out / "baseline" / name / "rate.npy", mmap_mode="r")
             for name in BASELINE}
        c = {name: np.load(out / "candidate" / name / "rate.npy", mmap_mode="r")
             for name in (*CANDIDATE, DISK)}
        old = ROOT / freeze["paths"]["e248_output"]
        for name in ("sensory_cut_low", "sensory_cut_high"):
            if not prior.same_bits(c[name], np.load(old / name / "rate.npy", mmap_mode="r")):
                raise AssertionError("Sensory-cut whole-neural rate differs")
            current_final = StateStore.load(out / "candidate" / name / "final")
            old_final = StateStore.load(old / name / "final")
            normalized = copy.deepcopy(current_final)
            for key in METADATA:
                normalized[key] = old_final[key]
            exact(normalized, old_final)
        for name in ("bridge_cut_low", "bridge_cut_high"):
            old_rate = np.load(old / name / "rate.npy", mmap_mode="r")
            non_bridge = np.ones(N, bool)
            non_bridge[bridge] = False
            if not prior.same_bits(c[name][:, non_bridge], old_rate[:, non_bridge]):
                raise AssertionError("Bridge-cut nonbridge rate differs")
            current_final = StateStore.load(out / "candidate" / name / "final")
            old_final = StateStore.load(old / name / "final")
            for key in ("rate", "adaptation"):
                if not prior.same_bits(current_final[key][non_bridge], old_final[key][non_bridge]):
                    raise AssertionError("Bridge-cut nonbridge final state differs")
        cut = prior.cut_validation(c["sensory_cut_low"], c["sensory_cut_high"], source)
        disk_exact = prior.same_bits(c["intact_low"], c[DISK])
        exact(StateStore.load(out / "candidate" / "intact_low" / "final"),
              StateStore.load(out / "candidate" / DISK / "final"))
        if not cut["non_source_bitexact"] or not disk_exact:
            raise AssertionError("Outgoing sensory cut or disk future differs")
        for name, values in (*b.items(), *c.items()):
            if not np.isfinite(values).all() or np.any(values < -1e-4) or np.any(values > 200.001):
                raise AssertionError("Full rate outside original bounded physiology: " + name)
        metrics = prior.response_metrics(c, lh_rows,
            response_min=freeze["rate_response_min"],
            interaction_min=freeze["interaction_min"],
            fraction_min=freeze["interaction_fraction_min"])
        baseline_delta = (b["intact_high"][1:, lh_rows].astype(np.float64)
                          - b["intact_low"][1:, lh_rows].astype(np.float64)) / 200.
        candidate_delta = (c["intact_high"][1:, lh_rows].astype(np.float64)
                           - c["intact_low"][1:, lh_rows].astype(np.float64)) / 200.
        metrics.update({"H250_supported": metrics.pop("H248_supported"),
            "baseline_response_R": float(np.sqrt(np.mean(baseline_delta ** 2))),
            "baseline_interaction_D": prior.read_json(old / "metrics.json")["interaction_D"],
            "candidate_minus_baseline_delta_rms": float(np.sqrt(
                np.mean((candidate_delta - baseline_delta) ** 2))),
            "candidate_disk_future_exact": disk_exact,
            "candidate_sensory_cut_non_source_exact": cut["non_source_bitexact"],
            **cut})
        prior.write_json(out / "metrics.json", metrics)
        result["validity"] = VALIDITY
        result["H250_supported"] = metrics["H250_supported"]
        result["future_exact"] = True
    except Exception as error:  # noqa: BLE001
        primary = error
        result["first_error"] = {"type": type(error).__name__, "message": str(error)}
        result["validity"] = "resource_limited" if isinstance(error, TimeoutError) else "invalid_or_incomplete"
        errors.append({**result["first_error"], "traceback": traceback.format_exc()})
    finally:
        network = None
        gc.collect()
        primary = finish_receipts(out, freeze, ledger, result, errors, primary, started)
        if primary is not None:
            raise RuntimeError("E250 invalid: " + str(primary)) from primary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out != (ROOT / BASE / "e250-lh-edge-restoration").resolve():
        raise ValueError("Fixed E250 output directory required")
    freeze = gate(args.freeze.resolve(), args.freeze_sha256,
                  args.review.resolve(), args.review_sha256)
    execute(freeze, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
