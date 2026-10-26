"""E257: paired, bounded native-body test of six existing LH synapses.

The only scientific intervention is the frozen CSR replacement.  All trial
state is restored from the same E150 checkpoint; the original controller,
sensory encoder, anatomy, and body advance remain in charge of both arms.
"""

# Broad catches preserve the first scientific failure through independent raw
# and cleanup writes, including a partially advanced physical control.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import analyze_c_banc_biological_gait as gait
from tools import probe_c_banc_m3_lh_edge_restoration as e250
from tools import probe_c_banc_m3_lh_timescale_partition as e256
from tools import verify_c_banc_m3_e113_body as e113
from tools import verify_c_banc_m3_native_monotone_body as e150
from tools import verify_c_banc_m3_native_timescale_transfer as protocol
from tools.banc_m3_monotone_recruitment import MonotoneRecruitmentTrial

BASE = ROOT / "verification/banc-natural-gait-m3-20260919"
FREEZE_SHA = "4868cb88b3c6ea9840591c300d5f596530b89f0aa81229b64b2b58cff08a9709"
FREEZE_SCHEMA = "flylab.m3.e257.lh-partition-body.freeze.v1"
REVIEW_SCHEMA = "flylab.m3.e257.lh-partition-body.review.v1"
OWNED = ("tools/verify_c_banc_m3_lh_partition_body.py",
         "tests/test_c_banc_lh_partition_body.py")
LEGS = e113.LEGS
TENDONS = tuple(f"{leg}_tarsus" for leg in LEGS)
N = 158706


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(name: str) -> Path:
    rel = Path(name)
    value = (ROOT / rel).resolve()
    if rel.is_absolute() or ".." in rel.parts or not value.is_relative_to(ROOT):
        raise ValueError("Repository-relative E257 input required")
    return value


def write_json(path: Path, value) -> None:
    from tools.verify_c_banc_m3_neural_current_body import write_receipt

    write_receipt(path, value)


def read_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def gate(freeze_path: Path, freeze_sha: str, review_path: Path,
         review_sha: str, output: Path) -> dict:
    """Validate all pinned bytes/packages before any original state is loaded."""
    if (freeze_sha != FREEZE_SHA or sha256(freeze_path) != FREEZE_SHA
            or sha256(review_path) != review_sha):
        raise ValueError("Exact E257 freeze and approved source review required")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    limits = freeze.get("resource_limits", {})
    if (freeze.get("schema") != FREEZE_SCHEMA
            or freeze.get("owned_files") != list(OWNED)
            or freeze.get("design_sha256") != sha256(safe(freeze["design"]))
            or freeze.get("output") != (BASE / "e257-lh-partition-body").relative_to(ROOT).as_posix()
            or freeze.get("arms") != ["baseline", "candidate"]
            or freeze.get("controls_per_arm") != 300
            or freeze.get("neural_ticks_per_control") != 20
            or freeze.get("physics_steps_per_control") != 50
            or freeze.get("network_identity") is None
            or freeze.get("n") != N
            or freeze.get("declared_edges") != 11584852
            or freeze.get("initial_tick") != 0
            or freeze.get("candidate_network_identity") !=
            "4d5bae3386bed2dce2ca8a2add5130449c60a70e3b0d95a51947feeb03c76354"
            or freeze.get("changed_edge_count") != 6
            or len(freeze.get("preserved_edge_ids", [])) != 19
            or len(freeze.get("core_observer_rows", [])) != 24
            or freeze.get("startup_limits") != {"q_RMS": .018, "q_max_abs": .04,
                                                  "LTM_RMS": .009, "LTM_max_abs": .025}
            or limits.get("internal_wall_s") != 140
            or limits.get("finalization_reserve_s") != 15
            or limits.get("original_attempts") != 1
            or limits.get("retries") != 0
            or limits.get("whole_restores") != 2
            or limits.get("neural_constructors") != 14
            or limits.get("scientific_neural_ticks") != 12000
            or limits.get("scientific_physics_steps") != 30000):
        raise ValueError("Frozen E257 scientific/resource contract differs")
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("freeze_sha256") != FREEZE_SHA
            or set(review.get("source_sha256", {})) != set(OWNED)):
        raise ValueError("E257 parent source review differs")
    pins: dict[str, str] = {}
    for group in (freeze["input_sha256"], freeze["reused_source_sha256"],
                  freeze["preservation_sha256"], review["source_sha256"],
                  review.get("additional_input_sha256", {}),
                  review.get("additional_reused_source_sha256", {})):
        for name, digest in group.items():
            if name in pins and pins[name] != digest:
                raise ValueError("Conflicting E257 pin: " + name)
            pins[name] = digest
    for name, digest in pins.items():
        if len(digest) != 64 or sha256(safe(name)) != digest:
            raise ValueError("Pinned E257 bytes changed: " + name)
    for name, version in freeze["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned E257 package differs: " + name)
    if output.resolve() != (BASE / "e257-lh-partition-body").resolve():
        raise ValueError("Exact E257 output required")
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("E257 output must be new")
    if shutil.disk_usage(output.parent).free < limits["minimum_free_disk_bytes"]:
        raise OSError("Insufficient E257 output disk")
    return freeze


def exact_nonidentity(source: dict, target: dict) -> None:
    from tools.verify_c_banc_observation import exact

    for key, value in source.items():
        if key not in e250.METADATA:
            exact(value, target[key], "E257 nonidentity " + key)


def transferred_network_state(saved: dict, constructed: dict, candidate: bool):
    """Same dynamic bytes; baseline keeps identity, candidate changes four IDs."""
    from tools.verify_c_banc_observation import exact

    if candidate:
        target, changed = e250.transplant_identity(saved, constructed)
        if changed != {"model_hash": ["model_hash"],
                       "core_adaptation": ["base_model_hash"],
                       "motor_adaptation": ["source_core_identity"],
                       "motor_timescale": ["source_motor_identity"]}:
            raise ValueError("Only the four E257 model identity fields may change")
    else:
        for key in e250.METADATA:
            exact(saved[key], constructed[key], "E257 baseline model metadata " + key)
        target, changed = copy.deepcopy(saved), {key: [] for key in e250.METADATA}
    exact_nonidentity(saved, target)
    return target, changed


class LHPartitionTrial(MonotoneRecruitmentTrial):
    """Own a single original MonotoneRecruitmentTrial with explicit CSR identity."""

    schema = "flylab.m3.e257.lh-partition-trial.v1"

    def __init__(self, base, provenance: dict):
        from tools.banc_m3_monotone_recruitment import MonotoneRecruitmentTrial

        if type(base) is not MonotoneRecruitmentTrial or base.closed:
            raise TypeError("Exact open E150 monotone trial required")
        self.__dict__.update(base.__dict__)
        base.closed = True
        self.lh_partition_patch = copy.deepcopy(provenance)

    def snapshot(self):
        from tools.banc_m3_monotone_recruitment import MonotoneRecruitmentTrial

        value = MonotoneRecruitmentTrial.snapshot(self)
        value["schema"] = self.schema
        value["lh_partition_patch"] = copy.deepcopy(self.lh_partition_patch)
        return value

    @classmethod
    def from_checkpoint(cls, graph, saved, baseline_weights):
        if saved.get("schema") != cls.schema:
            raise ValueError("Explicit E257 whole trial checkpoint required")
        provenance = saved.get("lh_partition_patch")
        if not isinstance(provenance, dict) or provenance.get("schema") != cls.schema:
            raise ValueError("Complete E257 patch provenance required")
        source = copy.deepcopy(saved)
        source.pop("lh_partition_patch")
        source["schema"] = MonotoneRecruitmentTrial.schema
        source["network"].update(copy.deepcopy(provenance["source_network_metadata"]))
        patch = provenance.get("edge_patch")
        if provenance["variant"] == "candidate":
            weights = e256.patch_six(baseline_weights, patch,
                np.asarray(provenance["preserved_edge_ids"], np.int64))
        else:
            weights = baseline_weights
        trial = restore_trial(graph, source, baseline_weights, weights,
                              provenance["variant"], patch=patch,
                              source_rows=provenance.get("source_rows"),
                              bridge_rows=provenance.get("bridge_rows"),
                              preserved_edge_ids=provenance.get("preserved_edge_ids"),
                              expected_candidate_identity=provenance.get("new_model_hash"))
        try:
            from tools.verify_c_banc_observation import exact

            exact(saved, trial.snapshot(), "E257 whole disk checkpoint")
            return trial
        except BaseException as error:
            try:
                trial.close()
            except BaseException as secondary:
                error.add_note("Secondary E257 disk-restore close: " + repr(secondary))
            raise


def restore_trial(graph, saved: dict, baseline_weights, weights, variant: str,
                  *, patch=None, source_rows=None, bridge_rows=None,
                  preserved_edge_ids=None, expected_candidate_identity=None):
    """Use E150 whole restoration, then replace its top network once."""
    from tools.banc_m3_motor_timescale import MotorTimescaleNetwork
    from tools.probe_c_banc_m3_core_rhythm import source_matrix
    from tools.verify_c_banc_observation import exact

    if variant not in ("baseline", "candidate"):
        raise ValueError("E257 variant must be baseline or candidate")
    if saved.get("schema") != MonotoneRecruitmentTrial.schema:
        raise ValueError("Exact E150 monotone checkpoint required")
    base = None
    try:
        base = MonotoneRecruitmentTrial.from_checkpoint(graph, saved)
        exact(saved, base.snapshot(), "E257 original whole trial restoration")
        old = base.network
        actual_base = source_matrix(old)
        for field in ("indptr", "indices", "data"):
            if not e250.prior.same_bits(getattr(actual_base, field),
                                        getattr(baseline_weights, field)):
                raise ValueError("E257 lower original CSR differs")
        if variant == "baseline":
            if any(not e250.prior.same_bits(getattr(weights, field),
                                            getattr(baseline_weights, field))
                   for field in ("indptr", "indices", "data")):
                raise ValueError("Baseline CSR must be byte-identical")
        else:
            expected_csr = e256.patch_six(baseline_weights, patch,
                np.asarray(preserved_edge_ids, np.int64))
            if (weights.shape != baseline_weights.shape or any(
                    not e250.prior.same_bits(getattr(weights, field),
                                              getattr(expected_csr, field))
                    for field in ("indptr", "indices", "data"))):
                raise ValueError("Candidate CSR is not the exact six-edge patch")
        network = MotorTimescaleNetwork(weights,
            *(getattr(old, key).detach().cpu().numpy().copy()
              for key in ("tau", "a", "threshold", "cap")),
            adaptation_gain=old.adaptation_gain,
            adaptation_tau_s=old.adaptation_tau_s, dt=old.dt,
            device=old.device, cuda_implementation=old.cuda_implementation,
            capture_steps=old.interval_steps, core_ids=old.core_ids.copy(),
            core_gain=old.core_gain, motor_ids=old.motor_ids.copy(),
            motor_gains=old.motor_gains.copy(),
            motor_log_tau=old.motor_log_tau.copy())
        expected_identity = (saved["network"]["model_hash"] if variant == "baseline"
                             else expected_candidate_identity)
        if expected_identity is None or network.identity != expected_identity:
            raise ValueError("E257 constructed network identity differs")
        restored, changed = transferred_network_state(
            saved["network"], network.snapshot(), variant == "candidate")
        network.restore(restored)
        exact(network.snapshot(), restored, "E257 restored top neural state")
        base.network = network
        expected = copy.deepcopy(saved)
        expected["network"] = restored
        exact(expected, base.snapshot(), "E257 transferred entire trial")
        provenance = {"schema": LHPartitionTrial.schema, "variant": variant,
                 "source_model_hash": saved["network"]["model_hash"],
                 "new_model_hash": network.identity,
                 "changed_identity_fields": changed,
                 "source_network_metadata": {key: copy.deepcopy(saved["network"][key])
                                             for key in e250.METADATA},
                 "edge_patch": copy.deepcopy(patch),
                  "source_rows": np.asarray(source_rows, np.int64).copy(),
                  "bridge_rows": np.asarray(bridge_rows, np.int64).copy(),
                  "preserved_edge_ids": np.asarray(preserved_edge_ids, np.int64).copy()}
        return LHPartitionTrial(base, provenance)
    except BaseException as error:
        if base is not None and not getattr(base, "closed", False):
            try:
                base.close()
            except BaseException as secondary:
                error.add_note("Secondary E257 restore cleanup: " + repr(secondary))
        raise


def short_window_slip(native: dict, *, complete: bool, minimum_load=.02) -> dict:
    """Frozen E113 weighted-slip formula on only [.5, 1.5) seconds."""
    unavailable = {"available": False, "load_weighted_mm_s": None,
                   "support_samples": None, "support_load_sum_bw": None,
                   "missing_selected_slip": None}
    if not complete:
        return unavailable
    time_s = np.asarray(native["time_s"])
    load = np.asarray(native["normal_load_bw"])
    slip = np.asarray(native["slip_mm_s"])
    valid = np.asarray(native["slip_valid"])
    if (time_s.shape != (15000,) or load.shape != (15000, 6)
            or slip.shape != load.shape or valid.shape != load.shape
            or not np.isfinite(time_s).all() or not np.isfinite(load).all()):
        return unavailable
    support = ((time_s >= .5) & (time_s < 1.5))[:, None] & (load >= minimum_load)
    missing = support & (~valid | ~np.isfinite(slip))
    if missing.any() or not support.any():
        return {**unavailable, "missing_selected_slip": int(missing.sum())}
    weight = float(load[support].sum())
    if weight <= 0:
        return unavailable
    value = float(np.sum(load[support] * slip[support]) / weight)
    if not math.isfinite(value):
        return unavailable
    return {"available": True, "load_weighted_mm_s": value,
            "support_samples": int(support.sum()),
            "support_load_sum_bw": weight, "missing_selected_slip": 0}


def short_forward(rows: list[dict], *, complete: bool) -> dict:
    unavailable = {"available": False, "fixed_initial_heading_forward_mm": None,
                   "window_samples": None}
    if not complete or len(rows) != 301:
        return unavailable
    times = np.asarray([row["time_s"] for row in rows], np.float64)
    points = np.asarray([row["body"]["position"] for row in rows], np.float64)
    forward = np.asarray(rows[0]["body"]["basis"], np.float64)[[0, 2], 0]
    selected = (times >= .5) & (times < 1.5)
    if (not np.allclose(times, np.arange(301) * .005, rtol=0, atol=2e-8)
            or selected.sum() != 200 or points.shape != (301, 3)
            or not np.isfinite(points).all() or not np.isfinite(forward).all()
            or np.linalg.norm(forward) <= 0):
        return unavailable
    forward /= np.linalg.norm(forward)
    segment = points[selected][:, [0, 2]]
    return {"available": True, "fixed_initial_heading_forward_mm":
            float((segment[-1] - segment[0]) @ forward),
            "window_samples": 200, "last_included_control": 299}


def short_health(rows: list[dict], native: dict, clock, fault: bool) -> dict:
    complete = bool(len(rows) == 301 and len(native["time_s"]) == 15000
                    and clock.clock_valid and not fault)
    controls = np.asarray([row["time_s"] for row in rows], np.float64)
    observed_clock_valid = bool(clock.clock_valid
        and np.allclose(controls, np.arange(len(rows)) * .005, rtol=0, atol=2e-8)
        and np.allclose(native["time_s"],
                        np.arange(1, len(native["time_s"]) + 1) * .0001,
                        rtol=0, atol=2e-8))
    timing = bool(complete
        and np.allclose(controls, np.arange(301) * .005, rtol=0, atol=2e-8)
        and np.allclose(native["time_s"], np.arange(1, 15001) * .0001,
                        rtol=0, atol=2e-8))
    upright = np.asarray(native["upright"])
    finite = bool(np.isfinite(native["normal_load_bw"]).all()
                  and np.isfinite(upright).all()
                  and all(not row.get("fault") for row in rows))
    minimum = float(np.min(upright)) if len(upright) and np.isfinite(upright).all() else None
    nonfoot = int(np.count_nonzero(native["nonfoot_contact"]))
    return {"complete_1p5s": complete, "clock_exact": timing,
            "observed_clock_valid": observed_clock_valid,
            "finite": finite, "minimum_native_upright": minimum,
            "nonfoot_contact_count": nonfoot,
            "healthy": bool(timing and finite and nonfoot == 0
                            and minimum is not None and minimum >= .9)}


def assess_h257(baseline: dict, candidate: dict, *, bout_min=.3) -> dict:
    """The five frozen conjuncts; missing observations never imply support."""
    b, c = baseline, candidate
    complete = bool(b.get("health", {}).get("healthy")
                    and c.get("health", {}).get("healthy"))
    startup = bool(c.get("startup", {}).get("strict_E141_startup_screen"))
    bf = b.get("forward", {})
    cf = c.get("forward", {})
    forward = bool(bf.get("available") and cf.get("available")
                   and cf["fixed_initial_heading_forward_mm"]
                   >= bf["fixed_initial_heading_forward_mm"])
    bs = b.get("slip", {})
    cs = c.get("slip", {})
    slip = bool(bs.get("available") and cs.get("available")
                and cs["load_weighted_mm_s"] <= bs["load_weighted_mm_s"])
    bg = b.get("gait", {}).get("window", {}).get("longest_forward_bout_s")
    cg = c.get("gait", {}).get("window", {}).get("longest_forward_bout_s")
    bout = bool(bg is not None and cg is not None
                and np.isfinite([bg, cg]).all() and cg >= bout_min and cg >= bg)
    return {"complete_healthy_both": complete, "strict_startup": startup,
            "same_window_forward_nonworse": forward,
            "same_window_supported_slip_nonworse": slip,
            "longest_forward_bout": bout,
            "H257_supported": bool(complete and startup and forward and slip and bout),
            "F3_F5_10s_evaluated": False, "F2_future_tested": False,
            "candidate_adopted": False}


def sensory_leaf(session):
    return session.adapter.original.original


def sensory_metadata(session, source_ids):
    leaf = sensory_leaf(session)
    return {"schema": "flylab.m3.e257.sensory-metadata.v1",
            "leaf_adapter_identity": leaf.identity,
            "source_rows": np.asarray(source_ids, np.int64),
            "receptor_keys": leaf._receptor_keys,
            "port_features": np.asarray(leaf._port_features),
            "sensory_ids": np.asarray(leaf._sensory_ids),
            "sensory_ports": np.asarray(leaf._sensory_ports),
            "joint_columns": np.asarray(leaf.joints),
            "sensory_gain": leaf.sensory_gain,
            "load_gain_multiplier": leaf.load_gain_multiplier,
            "front_hair_plate": leaf.front_hair_plate}


def attach_sensory_observer(session, records, source_ids):
    """Observe one original encoder call and return exactly its original drive."""
    leaf = sensory_leaf(session)
    prior = leaf.__dict__.get("encode", None)
    original = leaf.encode
    source_ids = np.asarray(source_ids, np.int64)

    def encode(*args, **kwargs):
        record = {"control_tick_before": int(session.control_tick),
                  "old_filtered": np.asarray(leaf.filtered).copy(),
                  "old_velocity_lowpass": np.asarray(leaf.velocity_lowpass).copy()}
        records.append(record)
        try:
            drive = original(*args, **kwargs)
            record.update(new_filtered=np.asarray(leaf.filtered).copy(),
                          new_velocity_lowpass=np.asarray(leaf.velocity_lowpass).copy(),
                          last_features=np.asarray(leaf.last_features).copy(),
                          returned_source_drive=np.asarray(drive)[source_ids].copy(),
                          complete=True)
            return drive
        except BaseException as error:
            record.update(complete=False, error=repr(error))
            raise

    leaf.encode = encode

    def detach():
        if prior is None:
            del leaf.encode
        else:
            leaf.encode = prior
    return detach


def save_sensory(path: Path, records: list[dict], source_ids):
    arrays = {"control_tick_before": np.asarray(
        [row["control_tick_before"] for row in records], np.int64),
        "complete": np.asarray([row.get("complete", False) for row in records], bool),
        "source_ids": np.asarray(source_ids, np.int64)}
    shapes = {"old_filtered": (241,), "new_filtered": (241,),
              "last_features": (241,), "old_velocity_lowpass": (42,),
              "new_velocity_lowpass": (42,),
              "returned_source_drive": (len(source_ids),)}
    for key, shape in shapes.items():
        result = np.full((len(records), *shape), np.nan,
                         np.float32 if key == "returned_source_drive" else np.float64)
        for index, record in enumerate(records):
            if key in record:
                if np.asarray(record[key]).shape != shape:
                    raise ValueError("Partial E257 sensory shape differs: " + key)
                result[index] = record[key]
        arrays[key] = result
    np.savez_compressed(path, **arrays)


def capture_control_raw(session, row):
    ids = np.asarray(session.adapter.calibration["motor_ids"], np.int64)
    rate = np.asarray(session.network.readout(ids))
    if rate.shape != (391,) or rate.dtype != np.float32:
        raise ValueError("Actual 391 motor rate32 required")
    return {"control_tick": int(session.control_tick),
            "network_tick": int(session.network.tick),
            "physics_time_s": float(session.body.physics_time()),
            "motor_rate32": rate.copy(),
            "activation": np.asarray(session.adapter.activation).copy(),
            "command": np.asarray(row["joint_targets_rad"], np.float64).copy(),
            "tendon": np.asarray([row["tendon_inputs"][key] for key in TENDONS],
                                 np.float64)}


def capture_activity(session, ids):
    net = session.network
    ids = np.asarray(ids, np.int64)
    return {"rate": np.asarray(net.readout(ids)).copy(),
            "drive": net.drive[ids].detach().cpu().numpy().copy(),
            "adaptation": net.adaptation[ids].detach().cpu().numpy().copy()}


def capture_core_activity(session, ids):
    """Passive original24 endpoint readout; no network advance or drive write."""
    values = capture_activity(session, ids)
    for key in ("rate", "drive", "adaptation"):
        if (values[key].shape != (24,) or values[key].dtype != np.float32
                or not np.isfinite(values[key]).all()):
            raise ValueError("E257 original24 readout shape/dtype/finite differs")
    physics_time = float(session.body.physics_time())
    if not math.isfinite(physics_time):
        raise ValueError("E257 original24 physical clock nonfinite")
    return {**values, "control_tick": int(session.control_tick),
            "network_tick": int(session.network.tick),
            "physics_time_s": physics_time}


def save_core_activity(path, records, ids):
    core_ids = np.asarray(ids, np.int64)
    if core_ids.shape != (24,) or len(np.unique(core_ids)) != 24:
        raise ValueError("E257 original24 IDs invalid")
    values = {"core_ids": core_ids}
    for key in ("control_tick", "network_tick"):
        values[key] = np.asarray([row[key] for row in records], np.int64)
    values["physics_time_s"] = np.asarray(
        [row["physics_time_s"] for row in records], np.float64)
    for key in ("rate", "drive", "adaptation"):
        values[key] = (np.stack([row[key] for row in records]) if records
                       else np.empty((0, 24), np.float32))
        if (values[key].shape != (len(records), 24)
                or values[key].dtype != np.float32
                or not np.isfinite(values[key]).all()):
            raise ValueError("E257 original24 stored readout invalid: " + key)
    if not np.isfinite(values["physics_time_s"]).all():
        raise ValueError("E257 original24 stored physical clock nonfinite")
    np.savez_compressed(path, **values)


def save_stacked(path: Path, rows: list[dict], keys: tuple[str, ...]):
    np.savez_compressed(path, **{key: np.asarray([row[key] for row in rows])
                                for key in keys})


def baseline_prefix(rows, native, recruitment, freeze):
    """Exact E150 first 301 / 15000 source records before candidate starts."""
    from tools.verify_c_banc_observation import exact

    if len(rows) != 301 or len(native["time_s"]) != 15000:
        raise ValueError("Complete E257 baseline prefix required")
    previous = json.loads(safe(freeze["paths"]["baseline_trace"])
                          .read_text(encoding="utf-8"))["samples"]
    exact(rows, previous[:301], "E257 control0..300 original trace")
    with np.load(safe(freeze["paths"]["baseline_native"]), allow_pickle=False) as archive:
        for key, value in native.items():
            if not e250.prior.same_bits(value, archive[key][:15000]):
                raise AssertionError("E257 baseline native differs: " + key)
    with np.load(safe(freeze["paths"]["baseline_recruitment"]),
                 allow_pickle=False) as archive:
        for key, value in recruitment.items():
            if key == "schema":
                continue
            if not e250.prior.same_bits(value, archive[key][:301]):
                raise AssertionError("E257 baseline recruitment differs: " + key)
    return {"control_rows_exact": 301, "native_samples_exact": 15000,
            "recruitment_rows_exact": 301}


def construction_with_body_count(label, callback, ledger, persist, on_return,
                                 *, body_step_code=None, construction_profile=None):
    """Profile both CUDA interval setup and actual neutral-settling calls."""
    if body_step_code is None:
        from flylab.body import FlyGymBody

        body_step_code = FlyGymBody._step_physics.__code__
    if construction_profile is None:
        construction_profile = protocol.profiled_construction
    code = body_step_code
    previous = sys.gettrace()
    count = {"label": label, "entered": 0, "returned": 0,
             "exception_unwinds": 0, "unknown_step_entries": 0,
             "known_completed_settling_steps": 0}
    ledger["body_construction"].append(count)
    persist()

    active = {}

    def physical_clock(frame):
        body = frame.f_locals.get("self")
        try:
            clock = float(body.d.time)
        except (AttributeError, TypeError, ValueError):
            return None
        return clock if math.isfinite(clock) else None

    def trace(frame, event, arg):
        del arg
        if frame.f_code is code:
            if event == "call":
                count["entered"] += 1
                active[id(frame)] = {"before": physical_clock(frame),
                                     "exception": False}
            elif event == "exception":
                active[id(frame)]["exception"] = True
            elif event == "return":
                count["returned"] += 1
                state = active.pop(id(frame), {"before": None, "exception": True})
                if state["exception"]:
                    count["exception_unwinds"] += 1
                after = physical_clock(frame)
                before = state["before"]
                if before is None or after is None:
                    count["unknown_step_entries"] += 1
                else:
                    steps = round((after - before) / .0001)
                    if (steps < 0 or abs(after - before - steps*.0001) > 2e-8):
                        count["unknown_step_entries"] += 1
                    else:
                        count["known_completed_settling_steps"] += steps
            return trace
        return None

    sys.settrace(trace)
    primary = None
    try:
        return construction_profile(label, callback,
            ledger["construction"], persist, on_return=on_return)
    except BaseException as error:
        primary = error
        raise
    finally:
        sys.settrace(previous)
        count["trace_restored"] = sys.gettrace() is previous
        count["active_unreturned_frames"] = len(active)
        try:
            persist()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note("Secondary E257 body constructor receipt: " + repr(error))


def recover_transferred_owner(session, holder, entry):
    """Return the sole live owner even if a post-return receipt failed."""
    handed_off = holder.get("session")
    if session is not None and handed_off is not None and session is not handed_off:
        raise ValueError("E257 construction transferred two distinct trial owners")
    if session is None and handed_off is not None:
        entry["post_return_owner_recovered"] = True
    return session if session is not None else handed_off


def classify_short_outcome(primary, measured_stop, health):
    if primary is not None or health.get("observed_clock_valid") is False:
        return "invalid_or_incomplete"
    if measured_stop is not None:
        return "measured_native_health_stop"
    if health.get("healthy") is True:
        return "complete_healthy_short_diagnostic"
    return "invalid_or_incomplete"


def save_arm_raw(arm: Path, rows, controls, activity, native_samples,
                 recruitment, sensory, clock, selection, graph_hash, model_hash,
                 body_hash=None, adapter_hash=None, parameters=None):
    """Persist independent partial artifacts; caller owns each failure receipt."""
    from tools import verify_c_banc_m3_neural_current_body as e224

    failures = []

    def attempt(name, callback):
        try:
            callback()
        except BaseException as error:
            failures.append({"artifact": name, "error": repr(error)})

    attempt("native-contact.npz", lambda: np.savez_compressed(
        arm / "native-contact.npz", **e113.native_arrays(native_samples)))
    attempt("native-recruitment-state.npz", lambda: e150.save_native_state(
        arm / "native-recruitment-state.npz", recruitment))
    attempt("sensory-input.npz", lambda: save_sensory(
        arm / "sensory-input.npz", sensory, selection["source_rows"]))
    attempt("control-raw.npz", lambda: save_stacked(arm / "control-raw.npz",
        controls, ("control_tick", "network_tick", "physics_time_s",
                   "motor_rate32", "activation", "command", "tendon")))

    def activity_file():
        ids = np.asarray(selection["observed_ids"], np.int64)
        arrays = {"observed_ids": ids, "source_ids": np.asarray(
            selection["source_rows"], np.int64),
            "bridge_ids": np.asarray(selection["bridge_rows"], np.int64),
            "reachable_motor_ids": np.asarray(
                selection["reachable_motor_rows"], np.int64)}
        arrays.update({key: np.asarray([row[key] for row in activity])
                       for key in ("rate", "drive", "adaptation")})
        np.savez_compressed(arm / "activity.npz", **arrays)
    attempt("activity.npz", activity_file)

    def trace_file():
        cleaned, unavailable = e224.finite_receipt(rows)
        write_json(arm / "trace.json", {"schema": "flylab.m3.e257.trace.v1",
            "controller": "E150 monotone recruitment; original sensory/body feedback",
            "graph_hash": graph_hash, "model_hash": model_hash,
            "body_hash": body_hash, "adapter_hash": adapter_hash,
            "parameters": parameters,
            "samples": cleaned, "unavailable_nonfinite_fields": unavailable})
    attempt("trace.json", trace_file)
    attempt("clock-events.json", lambda: e224.write_receipt(
        arm / "clock-events.json", clock.events))
    return failures


def output_manifest(out: Path, limit: int) -> dict:
    files, total = {}, 0
    for path in sorted(out.rglob("*")):
        if (not path.is_file() or path.name in
                ("result.json", "output-manifest.json")):
            continue
        size = path.stat().st_size
        total += size
        if total > limit:
            raise RuntimeError("E257 output byte cap exceeded")
        files[path.relative_to(out).as_posix()] = {"sha256": sha256(path),
                                                  "bytes": size}
    return {"schema": "flylab.m3.e257.output-manifest.v1",
            "bytes": total, "files": files}


def publish_final(out, ledger, result, errors, primary, began, limit,
                  writer=write_json):
    """Result is last. A failed late write cannot leave a supported H257."""
    def absorb(label, error):
        nonlocal primary
        errors.append({"stage": label, "type": type(error).__name__,
                       "message": str(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E257 " + label + ": " + repr(error))

    result["finalization_pending"] = True
    if primary is not None:
        result["validity"] = "invalid_or_incomplete"
        result["H257"] = {**result.get("H257", {}), "H257_supported": False}
    ledger["elapsed_s"] = time.perf_counter() - began
    for name, payload in (("advance-ledger.json", ledger), ("errors.json", errors)):
        try:
            writer(out / name, payload)
        except BaseException as error:
            absorb(name, error)
    try:
        writer(out / "output-manifest.json", output_manifest(out, limit))
    except BaseException as error:
        absorb("output-manifest.json", error)
    elapsed = time.perf_counter() - began
    if elapsed > 140:
        absorb("internal_wall_s", TimeoutError("E257 final publication exceeded140s"))
    if primary is not None:
        result["validity"] = "invalid_or_incomplete"
        result["H257"] = {**result.get("H257", {}), "H257_supported": False}
    result["primary_error"] = (None if primary is None else
        {"type": type(primary).__name__, "message": str(primary),
         "notes": list(getattr(primary, "__notes__", []))})
    result["secondary_errors"] = errors.copy()
    result["elapsed_s"] = elapsed
    result["finalization_pending"] = False
    try:
        writer(out / "result.json", result)
    except BaseException as error:
        absorb("result.json", error)
        (out / "result.json").unlink(missing_ok=True)
    if primary is None and time.perf_counter() - began > 140:
        absorb("post_result_internal_wall_s",
               TimeoutError("E257 result publication exceeded140s"))
    if primary is not None:
        # A late error may have followed an already written result. Revoke it.
        result["validity"] = "invalid_or_incomplete"
        result["H257"]["H257_supported"] = False
        result["primary_error"] = {"type": type(primary).__name__,
                                   "message": str(primary)}
        result["secondary_errors"] = errors.copy()
        try:
            writer(out / "result.json", result)
        except BaseException:
            (out / "result.json").unlink(missing_ok=True)
        try:
            writer(out / "errors.json", errors)
            writer(out / "advance-ledger.json", ledger)
            writer(out / "output-manifest.json", output_manifest(out, limit))
        except BaseException:
            (out / "output-manifest.json").unlink(missing_ok=True)
    return primary


def run_arm(graph, saved, baseline_weights, weights, variant, freeze, out,
            ledger, began, gpu_free_before, process):
    from tools import verify_c_banc_m3_neural_current_body as e224
    from tools.verify_c_banc_observation import exact

    arm = out / variant
    arm.mkdir(exist_ok=False)
    rows, controls, activity, core_activity, native, recruitment, sensory = (
        [], [], [], [], [], [], [])
    clock = e113.AdvanceLedger()
    journal = session = detach_native = detach_sensory = None
    primary = None
    measured_stop = None
    storage_errors = []
    holder = {}
    selection = freeze["selection"]
    ids = np.asarray([*selection["source_rows"], *selection["bridge_rows"],
                      *selection["reachable_motor_rows"]], np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("E257 selected observation rows overlap")
    selection = {**selection, "observed_ids": ids.tolist()}
    core_ids = np.asarray(freeze["core_observer_rows"], np.int64)
    if (core_ids.shape != (24,) or len(np.unique(core_ids)) != 24
            or np.any(core_ids < 0) or np.any(core_ids >= N)):
        raise ValueError("E257 frozen original24 observer rows invalid")
    entry = {"status": "requested", "variant": variant,
             "requested_controls": 300, "completed_controls": 0,
             "native_samples": 0, "construction_requested": False,
             "construction_completed": False, "measured_health_stop": False,
             "clock_events": [], "storage_errors": []}
    ledger["arms"][variant] = entry

    def persist():
        e224.write_receipt(out / "advance-ledger.json", ledger)

    def absorb(stage, error):
        nonlocal primary
        storage_errors.append({"stage": stage, "error": repr(error)})
        if primary is None:
            primary = error
        else:
            primary.add_note("Secondary E257 " + stage + ": " + repr(error))

    def budget():
        if time.perf_counter() - began >= 125:
            raise TimeoutError("E257 15s finalization reserve reached")
        memory = process.memory_info()
        peak = getattr(memory, "peak_wset", memory.rss)
        ledger["peak_resident_bytes"] = max(ledger.get("peak_resident_bytes", 0), peak)
        if peak > freeze["resource_limits"]["host_peak_bytes"]:
            raise MemoryError("E257 host RSS cap exceeded")
        if gpu_free_before is not None:
            import cupy as cp

            current, _ = cp.cuda.runtime.memGetInfo()
            added = max(0, gpu_free_before - current)
            ledger["peak_added_gpu_bytes"] = max(
                ledger.get("peak_added_gpu_bytes", 0), added)
            if added > freeze["resource_limits"]["additional_gpu_peak_bytes"]:
                raise MemoryError("E257 additional CUDA memory cap exceeded")

    try:
        budget()
        entry["construction_requested"] = True
        persist()
        session = construction_with_body_count(variant,
            lambda: restore_trial(graph, saved, baseline_weights, weights,
                variant, patch=freeze["patch"],
                source_rows=selection["source_rows"],
                bridge_rows=selection["bridge_rows"],
                preserved_edge_ids=freeze["preserved_edge_ids"],
                expected_candidate_identity=freeze["candidate_network_identity"]),
            ledger, persist, lambda value: holder.setdefault("session", value))
        entry["construction_completed"] = True
        if (session.network.n != N or session.network.device != "cuda"
                or session.network.cuda_implementation != "packed"
                or session.network.interval_steps != 20
                or session.control_tick != 0 or session.network.tick != 0
                or abs(session.body.physics_time()) > 2e-8):
            raise ValueError("Exact E257 CUDA/body neutral start required")
        backend = e250.prior.actual_backend(session.network)
        packed = e250.packed_csr_receipt(session.network, weights)
        entry["backend"] = backend
        entry["packed_csr"] = packed
        plain = session.snapshot()
        plain.pop("lh_partition_patch")
        plain["schema"] = MonotoneRecruitmentTrial.schema
        expected = copy.deepcopy(saved)
        expected["network"] = session.network.snapshot()
        exact(expected, plain, "E257 entire transferred neutral start")
        protocol.StateStore.save(arm / "observed-start", session.snapshot())
        e224.write_receipt(arm / "sensory-metadata.json",
            sensory_metadata(session, selection["source_rows"]))
        detach_native = e113.install_native_observer(session.body, native)
        detach_sensory = attach_sensory_observer(session, sensory,
                                                  selection["source_rows"])
        journal = e224.FiniteControlJournal(arm / "control-journal.jsonl")
        observer = e150.make_observer(recruitment,
            original=lambda _session, _detailed, _available: {})
        row = protocol.capture_control(session)
        rows.append(row)
        controls.append(capture_control_raw(session, row))
        observer(session, np.empty(0, np.int64))
        activity.append(capture_activity(session, ids))
        core_activity.append(capture_core_activity(session, core_ids))
        with (arm / "trace.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
            clean, _ = e224.finite_receipt(protocol.jsonable(row))
            stream.write(json.dumps(clean, allow_nan=False) + "\n")
            for control in range(1, 301):
                if control == 1 or control % 20 == 0:
                    budget()
                previous = (session.control_tick, session.network.tick,
                            session.body.physics_time(), len(native))
                try:
                    event = protocol.advance_control(session, variant, control,
                                                     clock, journal)
                except e113.NativeHealthStop as error:
                    measured_stop = {"control": control, "message": str(error),
                                     "native_samples": len(native)}
                    break
                try:
                    row = protocol.capture_control(session)
                    rows.append(row)
                    controls.append(capture_control_raw(session, row))
                    observer(session, np.empty(0, np.int64))
                    activity.append(capture_activity(session, ids))
                    core_activity.append(capture_core_activity(session, core_ids))
                    protocol.record_readout(journal, variant, control, event,
                                            complete=True)
                except BaseException as error:
                    try:
                        protocol.record_readout(journal, variant, control, event,
                                                complete=False, error=error)
                    except BaseException as secondary:
                        error.add_note("Secondary E257 readout journal: " + repr(secondary))
                    raise
                clean, _ = e224.finite_receipt(protocol.jsonable(row))
                stream.write(json.dumps(clean, allow_nan=False) + "\n")
                if (session.control_tick != previous[0] + 1
                        or session.network.tick != previous[1] + 20
                        or abs(session.body.physics_time() - previous[2] - .005) > 2e-8
                        or len(native) != previous[3] + 50):
                    raise ValueError("E257 incomplete neural/physical interval")
                entry["completed_controls"] = control
                if session.fault or session.body.fault:
                    measured_stop = {"control": control,
                                     "message": "measured body/session fault",
                                     "native_samples": len(native)}
                    break
        entry["status"] = "scientific_complete" if measured_stop is None else "measured_health_stop"
    except BaseException as error:
        primary = error
        entry["status"] = "technical_or_resource_failure"
        entry["primary_error"] = {"type": type(error).__name__,
                                   "message": str(error),
                                   "traceback": traceback.format_exc()}
    finally:
        try:
            session = recover_transferred_owner(session, holder, entry)
        except BaseException as error:
            absorb("construction ownership", error)
            session = session or holder.get("session")
        entry["native_samples"] = len(native)
        entry["clock_events"] = clock.events
        entry["measured_health_stop"] = measured_stop is not None
        entry["health_stop"] = measured_stop
        for label, callback in (("sensory detach", detach_sensory),
                                ("native detach", detach_native),
                                ("journal close", journal.close if journal else None)):
            if callback is None:
                continue
            try:
                callback()
            except BaseException as error:
                absorb(label, error)
        trace_metadata = dict.fromkeys(
            ("model_hash", "body_hash", "adapter_hash", "parameters"))
        if session is not None:
            for key, capture in (
                    ("model_hash", lambda: session.network.identity),
                    ("body_hash", lambda: session.body.model_hash),
                    ("adapter_hash", lambda: session.adapter.identity),
                    ("parameters", lambda: session.p.state_dict())):
                try:
                    trace_metadata[key] = capture()
                except BaseException as error:
                    absorb("trace metadata " + key, error)
        for failure in save_arm_raw(arm, rows, controls, activity, native,
                                    recruitment, sensory, clock, selection,
                                    graph.hash, **trace_metadata):
            absorb("raw " + failure["artifact"], RuntimeError(failure["error"]))
        try:
            save_core_activity(arm / "core-activity.npz", core_activity, core_ids)
        except BaseException as error:
            absorb("raw core-activity.npz", error)
        if session is not None:
            try:
                protocol.StateStore.save(arm / "final", session.snapshot())
            except BaseException as error:
                absorb("final whole checkpoint", error)
            try:
                session.close()
            except BaseException as error:
                absorb("trial close", error)
        entry["storage_errors"] = storage_errors
        entry["actual_neural_s"] = clock.neural_s
        entry["actual_physical_s"] = clock.physical_s
        try:
            persist()
        except BaseException as error:
            absorb("arm ledger", error)

    native_arrays = e113.native_arrays(native)
    complete = len(rows) == 301 and len(native) == 15000 and primary is None
    health = short_health(rows, native_arrays, clock,
                          bool(rows and rows[-1].get("fault")))
    slip = short_window_slip(native_arrays, complete=complete,
                             minimum_load=freeze["support_load_min_bw"])
    forward = short_forward(rows, complete=complete)
    metrics = {"status": entry["status"], "complete": complete,
               "measured_health_stop": measured_stop is not None,
               "health": health, "slip": slip, "forward": forward,
               "startup": {"available": False}, "gait": {"available": False},
               "baseline_prefix": None,
               "primary_error": (None if primary is None else
                   {"type": type(primary).__name__, "message": str(primary)})}
    if primary is None and complete:
        try:
            startup = read_npz(safe(freeze["paths"]["startup"]))
            metrics["startup"] = protocol.strict_startup(rows, native, startup,
                                                           freeze["startup_limits"])
            metrics["gait"], _, _ = gait.analyze(
                gait.load_model(arm / "trace.json"))
            if variant == "baseline":
                recruitment_arrays = read_npz(arm / "native-recruitment-state.npz")
                metrics["baseline_prefix"] = baseline_prefix(
                    rows, native_arrays, recruitment_arrays, freeze)
        except BaseException as error:
            primary = error
            metrics["primary_error"] = {"type": type(error).__name__,
                                        "message": str(error)}
            entry["status"] = "technical_analysis_failure"
    metrics["status"] = classify_short_outcome(primary, measured_stop, health)
    try:
        e224.write_receipt(arm / "arm-result.json", metrics)
    except BaseException as error:
        absorb("arm-result", error)
        metrics["status"] = "invalid_or_incomplete"
    entry["status"] = metrics["status"]
    return metrics, primary


def load_scientific_inputs(freeze):
    """All heavy data loads happen only after the full source/input gate."""
    from scipy.sparse import csr_matrix

    path = lambda key: safe(freeze["paths"][key])
    original = read_npz(path("native_csr"))
    if (set(original) < {"indptr", "indices", "data"}
            or original["data"].dtype != np.float32
            or original["data"].shape != (freeze["declared_edges"],)
            or original["indptr"].shape != (N + 1,)):
        raise ValueError("Frozen E150 native CSR invalid")
    baseline = csr_matrix((original["data"], original["indices"],
                           original["indptr"]), shape=(N, N), dtype=np.float32)
    baseline.check_format(full_check=True)
    saved = protocol.StateStore.load(path("checkpoint"))
    if (saved.get("schema") != MonotoneRecruitmentTrial.schema
            or saved["network"]["model_hash"] != freeze["network_identity"]
            or saved["network"]["tick"] != 0
            or saved["control_tick"] != 0):
        raise ValueError("E150 original whole neutral state differs")
    patch = freeze["patch"]
    file_patch = read_npz(path("e256_patch"))
    for key, values in file_patch.items():
        expected = (freeze["preserved_edge_ids"] if key == "preserved_edge_ids"
                    else patch.get(key))
        if expected is None or not e250.prior.same_bits(
                values, np.asarray(expected, dtype=values.dtype)):
            raise ValueError("E256 frozen six-edge patch bytes differ: " + key)
    e250.verify_bundle_edges(freeze, patch)
    candidate = e256.patch_six(baseline, patch,
        np.asarray(freeze["preserved_edge_ids"], np.int64))
    e256_csr = read_npz(path("e256_candidate_csr"))
    for key in ("indptr", "indices", "data"):
        if not e250.prior.same_bits(getattr(candidate, key), e256_csr[key]):
            raise ValueError("E257 candidate CSR differs from frozen E256: " + key)
    e256_result = json.loads(path("e256_result").read_text(encoding="utf-8"))
    e256_review = json.loads(path("e256_review").read_text(encoding="utf-8"))
    if (e256_result.get("validity") != e256.VALIDITY
            or e256_result.get("H256_supported") is not True
            or e256_result.get("metrics", {}).get("Q", {}).get("original24_rows")
               != freeze["core_observer_rows"]
            or e256_review.get("run_status") != "completed"
            or e256_review.get("validity") != "valid"
            or e256_review.get("candidate_identity") != freeze["candidate_network_identity"]
            or e256_review.get("actual_packed_csr_verified") is not True):
        raise ValueError("E256 source result/review incomplete")
    graph = protocol.GraphStore.load(path("graph_bundle"))
    if graph.hash != freeze["graph_identity"]:
        raise ValueError("E257 original BANC graph identity differs")
    return graph, saved, baseline, candidate


def run(freeze_path, freeze_sha, review_path, review_sha, output):
    began = time.perf_counter()
    out = Path(output).resolve()
    freeze = gate(Path(freeze_path), freeze_sha, Path(review_path),
                  review_sha, out)
    out.mkdir(parents=False, exist_ok=False)
    ledger = {"schema": "flylab.m3.e257.lh-partition-body.ledger.v1",
        "status": "gate_passed", "original_attempts": 1,
        "whole_restores_requested": 0, "construction": [],
        "body_construction": [], "arms": {}, "fit_calls": 0,
        "solver_calls": 0, "retries": 0, "candidate_adopted": False,
        "F2_future_tested": False, "F6_exposed": False,
        "whole_goal_pass": False, "storage_errors": []}
    result = {"schema": "flylab.m3.e257.lh-partition-body.result.v1",
              "validity": "incomplete", "baseline": "not_run",
              "candidate": "not_run", "H257": {"H257_supported": False},
              "candidate_adopted": False, "F2_future_tested": False,
              "F3_F5_10s_evaluated": False, "F6_exposed": False,
              "whole_goal_pass": False, "finalization_pending": True}
    errors = []
    primary = None
    thread_limits = None
    process = None
    gpu_free_before = None

    def persist():
        write_json(out / "advance-ledger.json", ledger)

    try:
        write_json(out / "source-review-snapshot.json",
                   json.loads(Path(review_path).read_text(encoding="utf-8")))
        from tools.fit_c_banc_m3_joint_adaptive_recruitment import one_cpu_thread

        thread_limits, threading = one_cpu_thread()
        ledger["threading"] = threading
        import cupy as cp
        import psutil

        process = psutil.Process()
        gpu_free_before, _ = cp.cuda.runtime.memGetInfo()
        if gpu_free_before < freeze["resource_limits"]["minimum_free_gpu_bytes"]:
            raise MemoryError("E257 insufficient GPU free memory")
        ledger["gpu_free_before_bytes"] = gpu_free_before
        persist()
        graph, saved, baseline, candidate = load_scientific_inputs(freeze)
        for name, weights in (("baseline", baseline), ("candidate", candidate)):
            np.savez_compressed(out / f"{name}-csr.npz",
                indptr=weights.indptr, indices=weights.indices, data=weights.data)
        write_json(out / "csr-identities.json", {
            "source_model_hash": saved["network"]["model_hash"],
            "baseline_data_sha256": sha256(out / "baseline-csr.npz"),
            "candidate_data_sha256": sha256(out / "candidate-csr.npz"),
            "changed_edge_ids": freeze["patch"]["edge_ids"],
            "source_rows": freeze["selection"]["source_rows"],
            "bridge_rows": freeze["selection"]["bridge_rows"]})
        metrics = {}
        for variant, weights in (("baseline", baseline), ("candidate", candidate)):
            if variant == "candidate" and metrics["baseline"]["status"] != (
                    "complete_healthy_short_diagnostic"):
                result["candidate"] = "not_run_baseline_invalid"
                break
            ledger["whole_restores_requested"] += 1
            ledger["status"] = variant + "_requested"
            persist()
            arm_metrics, arm_error = run_arm(graph, saved, baseline, weights,
                variant, freeze, out, ledger, began, gpu_free_before, process)
            metrics[variant] = arm_metrics
            result[variant] = arm_metrics["status"]
            write_json(out / f"{variant}/arm-ledger.json", ledger["arms"][variant])
            if arm_error is not None:
                raise arm_error
            if variant == "baseline" and arm_metrics["baseline_prefix"] is None:
                raise ValueError("E257 baseline exact prefix not established")
        if "candidate" in metrics:
            result["H257"] = assess_h257(metrics["baseline"], metrics["candidate"],
                                          bout_min=freeze["bout_min_s"])
            result["validity"] = ("valid_short_native_body_diagnostic" if
                metrics["candidate"]["status"] in
                ("complete_healthy_short_diagnostic", "measured_native_health_stop")
                else "invalid_or_incomplete")
        ledger["status"] = "science_complete"
    except BaseException as error:
        primary = error
        ledger["status"] = "technical_or_resource_failure"
        errors.append({"stage": "run", "type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc()})
    finally:
        if thread_limits is not None:
            try:
                thread_limits.restore_original_limits()
            except BaseException as error:
                errors.append({"stage": "thread_cleanup", "error": repr(error)})
                if primary is None:
                    primary = error
                else:
                    primary.add_note("Secondary E257 thread restore: " + repr(error))
        ledger["whole_restores_completed"] = sum(
            bool(row.get("construction_completed")) for row in ledger["arms"].values())
        ledger["setup_neural_executed_ticks"] = sum(
            row["known_completed_setup_steps"] for row in ledger["construction"])
        ledger["setup_neural_capture_ticks"] = sum(
            row["recorded_capture_setup_steps"] for row in ledger["construction"])
        ledger["setup_physics_steps"] = sum(
            row["known_completed_settling_steps"] for row in ledger["body_construction"])
        ledger["scientific_neural_ticks"] = sum(
            max(0, int(event["neural_tick_after"] - event["neural_tick_before"]))
            for arm in ledger["arms"].values() for event in arm["clock_events"]
            if event.get("neural_tick_after") is not None)
        ledger["scientific_physics_steps_known"] = sum(
            max(0, round(event["actual_physical_s"] / .0001))
            for arm in ledger["arms"].values() for event in arm["clock_events"]
            if event.get("actual_physical_s") is not None)
        ledger["unknown_scientific_physics_intervals"] = sum(
            event.get("actual_physical_s") is None for arm in ledger["arms"].values()
            for event in arm["clock_events"])
        if (primary is None and result["validity"] == "valid_short_native_body_diagnostic"
                and (ledger["whole_restores_completed"] != 2
                     or len(ledger["construction"]) != 2
                     or ledger["setup_neural_executed_ticks"] != 560
                     or ledger["setup_neural_capture_ticks"] != 280
                     or ledger["setup_physics_steps"] != 4000
                     or (result["candidate"] == "complete_healthy_short_diagnostic"
                         and (ledger["scientific_neural_ticks"] != 12000
                              or ledger["scientific_physics_steps_known"] != 30000)))):
            primary = ValueError("E257 actual construction/science work differs")
        if primary is not None:
            result["validity"] = "invalid_or_incomplete"
            result["H257"] = {**result["H257"], "H257_supported": False}
        primary = publish_final(out, ledger, result, errors, primary, began,
                                freeze["resource_limits"]["output_bytes"])
    return 0 if primary is None and result["validity"] == (
        "valid_short_native_body_diagnostic") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.freeze, args.freeze_sha256, args.source_review,
               args.source_review_sha256, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
