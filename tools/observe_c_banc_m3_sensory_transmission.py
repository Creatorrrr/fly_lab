"""E193: one guarded, observational six-leg sensory transmission run.

Import and --help do not construct a model or access CUDA. The parent owns
source approval, frozen inputs, resource readiness, and the process supervisor.
"""

# Broad catches preserve the original model exception after partial persistence.
# ruff: noqa: BLE001
from __future__ import annotations

import argparse
import copy
import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

_ENTRY_STARTED = time.perf_counter()

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import banc_m3_resource_readiness as resources
from tools import observe_c_banc_m3_mn_inputs as e188

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
DESIGN = "docs/BANC_M3_E193_SENSORY_TRANSMISSION_DESIGN_20260924.md"
METADATA = BASE + "/parent-e193-frozen-metadata.json"
FREEZE = BASE + "/e193-sensory-transmission-freeze.json"
REVIEW = BASE + "/parent-e193-source-review.json"
RECEIPT = BASE + "/parent-e193-input-validation.json"
SUPERVISOR = BASE + "/parent-e193-supervise.py"
OUTPUT = BASE + "/e193-sensory-transmission"
OWNED = ("tools/observe_c_banc_m3_sensory_transmission.py", "tests/test_c_banc_sensory_transmission.py")
ROWS = 601
CONTROLS = 600
N = 158706
INTERNAL_S = 140.0
FREE_BYTES = 20 << 30
OUTPUT_BYTES = 2 << 30
BASELINE_BYTES = 1 << 30
PARAMETERS = {
    "control_dt_s": 0.005,
    "control_rows": 601,
    "native_rows": 30000,
    "neural_steps": 12000,
    "physics_steps": 30000,
    "cpu_threads": 1,
    "blas_threads": 1,
    "gpu_count": 1,
    "internal_wall_s": 140,
    "e181_wall_s": 100,
    "supervisor_wall_s": 150,
    "child_process_wall_s": 145,
    "cleanup_reserve_s": 5,
    "automatic_retries": 0,
    "maximum_output_bytes": 2147483648,
    "maximum_baseline_bytes": 1073741824,
    "minimum_free_bytes": 21474836480,
    "cumulative_before_s": 4670.873046799719,
    "cumulative_cap_s": 4994.0,
}

require = e188.require
safe = e188.safe
sha256 = e188.sha256
read_json = e188.read_json
write_new = e188.write_new
input_stamps = e188.input_stamps
tensor_array = e188.tensor_array


def validate_parent_inputs(inputs: dict[str, str], freeze_sha: str,
                           review_sha: str, metadata_sha: str, receipt_sha: str) -> dict:
    """Bind the child's stat checks to a fresh full-hash parent receipt."""
    import psutil

    path = safe(RECEIPT)
    require(len(receipt_sha) == 64 and sha256(path) == receipt_sha, "parent receipt SHA")
    receipt = read_json(path)
    require(
        receipt.get("schema") == "flylab.m3.e193.parent-input-validation.v1"
        and receipt.get("validated") is True
        and receipt.get("freeze_sha256") == freeze_sha
        and receipt.get("source_review_sha256") == review_sha
        and receipt.get("metadata_sha256") == metadata_sha
        and receipt.get("input_sha256") == inputs,
        "parent receipt contract",
    )
    parent = receipt["parent"]
    if parent["pid"] != os.getppid():
        launcher = psutil.Process(os.getppid())
        require(
            os.name == "nt" and launcher.is_running()
            and launcher.ppid() == parent["pid"]
            and Path(launcher.exe()).resolve() == Path(sys.executable).resolve()
            and launcher.cmdline()[1:] == psutil.Process().cmdline()[1:],
            "different parent or unverified venv launcher",
        )
    live = psutil.Process(parent["pid"])
    require(live.is_running() and abs(live.create_time() - parent["create_time"]) <= .001,
            "parent process identity")
    script = safe(SUPERVISOR).resolve()
    require(any((Path(live.cwd()) / arg).resolve() == script for arg in live.cmdline()
                if str(arg).endswith(".py")), "supervisor command")
    age = time.perf_counter() - receipt["issued_monotonic_s"]
    require(math.isfinite(age) and 0 <= age <= 60, "stale parent receipt")
    require(input_stamps(inputs) == receipt.get("input_stamps"), "input stat changed")
    return receipt


def gate(freeze_path: Path, review_path: Path, output: Path,
         parent_validation_sha256: str | None = None) -> tuple[dict, dict, str, dict | None]:
    require(not output.exists() and output.parent.is_dir(), "output exists or parent missing")
    require(freeze_path.resolve() == safe(FREEZE).resolve()
            and review_path.resolve() == safe(REVIEW).resolve(), "gate path")
    freeze, review = read_json(freeze_path), read_json(review_path)
    frozen_sha = sha256(freeze_path)
    require(freeze.get("schema") == "flylab.m3.e193.sensory-transmission.freeze.v1"
            and review.get("schema") == "flylab.m3.e193.sensory-transmission.source-review.v1",
            "gate schema")
    require(review.get("approved") is True and review.get("blocking_defects") == []
            and review.get("freeze_sha256") == frozen_sha, "parent source approval")
    require(freeze.get("design_path") == DESIGN
            and freeze.get("design_sha256") == sha256(safe(DESIGN))
            and freeze.get("metadata_path") == METADATA
            and freeze.get("metadata_sha256") == sha256(safe(METADATA))
            and freeze.get("output_path") == OUTPUT
            and output.resolve() == safe(OUTPUT).resolve(), "frozen paths and hashes")
    require(freeze.get("parameters") == PARAMETERS
            and freeze.get("parent_supervisor_sha256") == sha256(safe(SUPERVISOR)),
            "resource/supervisor contract")
    metadata = read_json(safe(METADATA))
    require(metadata.get("schema") == "flylab.m3.e193.sensory-transmission.metadata.v1"
            and metadata.get("design_path") == DESIGN
            and metadata.get("design_sha256") == freeze["design_sha256"]
            and metadata.get("owned_files") == list(OWNED)
            and metadata.get("original_native_runs") == 0
            and metadata.get("F6_exposed") == 0
            and metadata.get("candidate_adopted") is False
            and metadata.get("cumulative_before_s") == PARAMETERS["cumulative_before_s"]
            and metadata.get("cumulative_limit_s") == PARAMETERS["cumulative_cap_s"],
            "metadata contract")
    source_hashes = freeze.get("source_hashes")
    require(isinstance(source_hashes, dict) and set(OWNED) <= set(source_hashes)
            and isinstance(metadata.get("reused_sources"), dict)
            and set(metadata["reused_sources"]) <= set(source_hashes), "source inventory")
    require(review.get("source_hashes") == source_hashes
            and review.get("design_sha256") == freeze["design_sha256"]
            and review.get("metadata_sha256") == freeze["metadata_sha256"], "review identity")
    for name, expected in source_hashes.items():
        require(len(expected) == 64 and sha256(safe(name)) == expected,
                "source changed: " + name)
    for name, expected in metadata["reused_sources"].items():
        require(source_hashes[name] == expected, "reused source changed: " + name)
    inputs = metadata.get("inputs")
    require(isinstance(inputs, dict) and inputs and freeze.get("input_sha256") == inputs,
            "input inventory")
    receipt = None
    if parent_validation_sha256 is None:
        for name, expected in inputs.items():
            require(len(expected) == 64 and sha256(safe(name)) == expected,
                    "input changed: " + name)
    else:
        receipt = validate_parent_inputs(inputs, frozen_sha, sha256(review_path),
                                         freeze["metadata_sha256"], parent_validation_sha256)
    old = metadata["e181_gate"]
    for name, expected in ((old["freeze_path"], old["freeze_sha256"]),
                           (old["source_review_path"], old["source_review_sha256"])):
        require(sha256(safe(name)) == expected, "E181 gate changed")
    for field in ("graph_path", "neutral_checkpoint", "startup_path", "baseline_directory",
                  "expected_graph_hash", "expected_network_identity",
                  "expected_adapter_identity", "expected_body_hash"):
        require(isinstance(metadata.get(field), str) and metadata[field], field)
    return metadata, freeze, frozen_sha, receipt


def parent_readiness(receipt: dict) -> list[dict]:
    age = time.perf_counter() - receipt["resource_readiness_finished_monotonic_s"]
    require(math.isfinite(age) and 0 <= age <= 10, "parent readiness expired")
    rows = receipt.get("resource_readiness")
    require(isinstance(rows, list) and len(rows) == 2
            and all(row.get("phase") == "readiness" and row.get("ready") is True
                    and row.get("resource_clear") is True
                    and row.get("resource_mode") in ("queue_empty", "service_stopped")
                    for row in rows), "parent readiness")
    return rows


def leaf_adapter(adapter: Any) -> Any:
    """Descend a known wrapper chain without changing it."""
    from flylab.c.rate_body import RateBodyAdapter

    seen: set[int] = set()
    while not isinstance(adapter, RateBodyAdapter):
        require(id(adapter) not in seen and hasattr(adapter, "original"), "RateBodyAdapter chain")
        seen.add(id(adapter))
        adapter = adapter.original
    return adapter


def snapshot_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: snapshot_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def temporary_method(owner: Any, name: str, wrapper: Any):
    """Restore the exact instance slot state, including absence."""
    existed = name in vars(owner)
    prior = vars(owner).get(name)
    setattr(owner, name, wrapper)

    def restore():
        if existed:
            setattr(owner, name, prior)
        else:
            delattr(owner, name)

    return restore


class Collector:
    def __init__(self, metadata: dict):
        self.metadata = metadata
        self.encoder: list[dict] = []
        self.neural: list[dict] = []
        self.provenance: dict[str, Any] = {}
        self.error: str | None = None
        self.adapter: Any = None
        self.session: Any = None
        self.index: Any = None
        self.unique_ids: np.ndarray | None = None

    def attach(self, session: Any) -> None:
        require(self.session is None, "duplicate session attachment")
        adapter = leaf_adapter(session.adapter)
        require(session.graph.n == N and session.network.n == N
                and session.network.device == "cuda"
                and session.network.cuda_implementation == "packed", "full packed CUDA")
        require(session.graph.hash == self.metadata["expected_graph_hash"]
                and session.network.identity == self.metadata["expected_network_identity"]
                and session.adapter.identity == self.metadata["expected_adapter_identity"]
                and session.body.model_hash == self.metadata["expected_body_hash"],
                "runtime graph/network/adapter/body identity")
        require(adapter.body is session.body and adapter.graph is session.graph,
                "leaf adapter binding")
        require(adapter.sensory_gain == 18.0 and adapter.load_gain_multiplier == 1.0
                and adapter.front_hair_plate is False, "frozen receptor gain/options")
        expansion = np.asarray(adapter._sensory_ids, np.int64).copy()
        require(np.array_equal(expansion, np.concatenate([ids for *_, ids in adapter.ports]))
                and np.array_equal(adapter._sensory_ports,
                                   np.repeat(np.arange(len(adapter.ports)),
                                             [len(ids) for *_, ids in adapter.ports])),
                "port expansion order")
        unique = np.unique(expansion)
        require(np.all((unique >= 0) & (unique < N)) and len(unique) > 0,
                "sensory indices")
        self.index = session.network.torch.as_tensor(unique, device="cuda")
        require(not np.intersect1d(unique, np.asarray(session.descending)).size,
                "sensory/descending overlap")
        if session.navigation:
            nav = session.navigation.encoder
            nav_ids = np.concatenate((nav.ports, nav.descending))
            require(not np.intersect1d(unique, nav_ids).size,
                    "sensory/navigation overlap")
        graph_ids = [[session.graph.nodes[int(i)]["id"] for i in ids]
                     for *_, ids in adapter.ports]
        self.provenance = {
            "graph_hash": session.graph.hash,
            "graph_neurons": N,
            "network_identity": session.network.identity,
            "network_device": str(session.network.device),
            "network_cuda_implementation": session.network.cuda_implementation,
            "body_model_hash": session.body.model_hash,
            "adapter_identity": session.adapter.identity,
            "leaf_adapter_identity": adapter.identity,
            "sensory_gain_model_units": adapter.sensory_gain,
            "load_gain_multiplier": adapter.load_gain_multiplier,
            "front_hair_plate": adapter.front_hair_plate,
            "ports": [{"port": p, "leg_index": int(leg), "kind": kind,
                       "cell_type": cell, "indices": np.asarray(ids).tolist(),
                       "root_ids": roots}
                      for p, ((leg, kind, cell, ids), roots)
                      in enumerate(zip(adapter.ports, graph_ids))],
            "port_expansion_indices": expansion.tolist(),
            "port_expansion_ports": np.asarray(adapter._sensory_ports).tolist(),
            "unique_sensory_indices": unique.tolist(),
            "joints": adapter.joints.tolist(),
            "coxa_sign": adapter.coxa_sign.tolist(),
            "knee_sign": adapter.knee_sign.tolist(),
            "neutral": np.asarray(session.body.neutral)[adapter.joints].tolist(),
            "spec": json_value(adapter.spec),
            "network_parameters": {
                **{key: tensor_array(getattr(session.network, key)[self.index], np.float32,
                                      (len(unique),), key).tolist()
                   for key in ("cap", "coefficient", "threshold", "tau")},
                "adaptation_gain": float(session.network.adaptation_gain),
                "adaptation_tau_s": float(session.network.adaptation_tau_s),
            },
        }
        self.adapter, self.session, self.unique_ids = adapter, session, unique

    def capture_neural(self, session: Any) -> None:
        tick = int(session.control_tick)
        network = session.network
        row: dict[str, Any] = {"control_tick": tick, "complete": False}
        self.neural.append(row)
        try:
            clock = (tick, int(network.tick), float(session.body.physics_time()))
            row.update(network_tick=clock[1], time_s=clock[2])
            require(tick == len(self.neural) - 1 and tick <= CONTROLS
                    and clock[1] == tick * 20
                    and abs(clock[2] - tick * .005) <= 2e-8, "neural endpoint clock")
            if tick == 0:
                self.attach(session)
            else:
                require(session is self.session, "session changed")
            fields = ("rate", "drive", "adaptation", "output_mask")
            before = {name: getattr(network, name).clone() for name in fields}
            current = network._current(network.rate)
            require(all(bool(network.torch.equal(before[name], getattr(network, name)))
                        for name in fields), "neural readout changed state")
            for name in fields:
                row[name] = tensor_array(getattr(network, name)[self.index], np.float32,
                                         (len(self.unique_ids),), name)
            row["current"] = tensor_array(current[self.index], np.float32,
                                          (len(self.unique_ids),), "current")
            require(clock == (int(session.control_tick), int(network.tick),
                              float(session.body.physics_time())), "neural readout changed clock")
            row["complete"] = True
        except BaseException as error:
            row["error"] = repr(error)
            self.error = repr(error)
            raise

    def capture_encode(self, original: Any, *args: Any, **kwargs: Any) -> Any:
        adapter, session = self.adapter, self.session
        require(adapter is not None and session is not None, "encode before attachment")
        tick = int(session.control_tick)
        row: dict[str, Any] = {"control_tick": tick, "complete": False}
        self.encoder.append(row)
        body = adapter.body
        leg_original, knee_original = body.leg_observation, body.knee_kinematics
        calls = {"leg": 0, "knee": 0}
        restores = []
        primary: BaseException | None = None
        cleanup_failures: list[BaseException] = []

        def leg_wrapper(*a, **k):
            calls["leg"] += 1
            result = leg_original(*a, **k)
            row["leg_observation"] = snapshot_value(result)
            return result

        def knee_wrapper(*a, **k):
            calls["knee"] += 1
            result = knee_original(*a, **k)
            row["knee_kinematics"] = snapshot_value(result)
            return result

        try:
            clock = (tick, int(session.network.tick), float(body.physics_time()))
            row.update(network_tick=clock[1], time_s=clock[2])
            require(tick == len(self.encoder) - 1 and tick < CONTROLS
                    and clock[1] == tick * 20
                    and abs(clock[2] - tick * .005) <= 2e-8, "encode clock")
            row["old_filtered"] = adapter.filtered.copy()
            row["old_velocity_lowpass"] = adapter.velocity_lowpass.copy()
            restores.append(temporary_method(body, "leg_observation", leg_wrapper))
            restores.append(temporary_method(body, "knee_kinematics", knee_wrapper))
            result = original(*args, **kwargs)
            returned = np.asarray(result)
            require(returned.shape == (N,) and returned.dtype == np.float32,
                    "returned sensory drive shape/dtype")
            if self.unique_ids is not None:
                non_sensory = returned.copy()
                non_sensory[self.unique_ids] = 0
                require(not np.any(non_sensory), "returned nonsensory drive")
                row["returned_drive"] = returned[self.unique_ids].copy()
            row["last_features"] = adapter.last_features.copy()
            row["new_filtered"] = adapter.filtered.copy()
            row["new_velocity_lowpass"] = adapter.velocity_lowpass.copy()
            require(calls == {"leg": 1, "knee": 1}, "body observation call count")
            require(clock == (int(session.control_tick), int(session.network.tick),
                              float(body.physics_time())), "encode changed clock")
            row["complete"] = True
            return result
        except BaseException as error:
            primary = error
            row["error"] = repr(error)
            self.error = repr(error)
            raise
        finally:
            for restore in reversed(restores):
                try:
                    restore()
                except BaseException as cleanup_error:
                    cleanup_failures.append(cleanup_error)
                    row.setdefault("cleanup_errors", []).append(repr(cleanup_error))
            if cleanup_failures:
                row["complete"] = False
                if primary is None:
                    row["error"] = repr(cleanup_failures[0])
                    self.error = repr(cleanup_failures[0])
                    raise cleanup_failures[0]

    def arrays(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        ports = len(self.provenance.get("ports", []))
        cells = len(self.unique_ids) if self.unique_ids is not None else 0
        if not ports:
            ports = next((len(row["old_filtered"]) for row in self.encoder
                          if "old_filtered" in row), 0)

        def make(rows: list[dict], fields: dict[str, tuple[tuple[int, ...], Any, Any]]):
            def context(value: Any) -> Any:
                if isinstance(value, np.ndarray):
                    return {"shape": value.shape, "dtype": str(value.dtype)}
                if isinstance(value, dict):
                    return {str(key): context(item) for key, item in value.items()}
                return repr(value)[:160]

            errors = [str(row.get("error", "")) for row in rows]
            complete = np.asarray([bool(row.get("complete", False)) for row in rows], np.bool_)
            result = {"control_tick": np.asarray([row.get("control_tick", -1) for row in rows], np.int64),
                      "complete": complete,
                      "cleanup_error": np.asarray([repr(row.get("cleanup_errors", [])) for row in rows]),
                      "raw_context": np.asarray([repr({key: context(value)
                          for key, value in row.items() if key not in ("returned_drive",)})
                          for row in rows])}
            for key, (shape, dtype, fill) in fields.items():
                array = np.full((len(rows), *shape), fill, dtype=dtype)
                available = np.zeros(len(rows), np.bool_)
                for i, row in enumerate(rows):
                    if key in row:
                        try:
                            require(np.asarray(row[key]).shape == shape, key + " shape")
                            array[i] = row[key]
                            available[i] = True
                        except BaseException as error:
                            errors[i] += ("; " if errors[i] else "") + (
                                f"{key}: {error!r}; observed_shape={getattr(row[key], 'shape', None)}")
                            complete[i] = False
                result[key], result[key + "_available"] = array, available
            result["error"] = np.asarray(errors)
            return result

        encoder_fields = {
            "network_tick": ((), np.int64, -1), "time_s": ((), np.float64, np.nan),
            "old_filtered": ((ports,), np.float64, np.nan),
            "new_filtered": ((ports,), np.float64, np.nan),
            "old_velocity_lowpass": ((42,), np.float64, np.nan),
            "new_velocity_lowpass": ((42,), np.float64, np.nan),
            "last_features": ((ports,), np.float64, np.nan),
            "request": ((ports,), np.float64, np.nan),
            "angles_rad": ((42,), np.float64, np.nan),
            "velocities_rad_s": ((42,), np.float64, np.nan),
            "encoded_angles_rad": ((42,), np.float64, np.nan),
            "encoded_velocities_rad_s": ((42,), np.float64, np.nan),
            "support_load_bw": ((6,), np.float64, np.nan),
            "non_support_load_bw": ((6,), np.float64, np.nan),
            "flexion_rad": ((6,), np.float64, np.nan),
            "flexion_velocity_rad_s": ((6,), np.float64, np.nan),
            "returned_drive": ((cells,), np.float32, np.nan),
        }
        reduced = []
        for row in self.encoder:
            row = dict(row)
            try:
                if "leg_observation" in row:
                    obs = row["leg_observation"]
                    for key in ("angles_rad", "velocities_rad_s", "support_load_bw",
                                "non_support_load_bw"):
                        if key in obs:
                            row[key] = obs[key]
                if "knee_kinematics" in row:
                    knee = row["knee_kinematics"]
                    if "flexion_rad" in knee:
                        row["flexion_rad"] = knee["flexion_rad"]
                    if "velocity_rad_s" in knee:
                        row["flexion_velocity_rad_s"] = knee["velocity_rad_s"]
                if "last_features" in row and self.adapter is not None:
                    row["request"] = self.adapter.sensory_gain * row["last_features"]
                if "angles_rad" in row and "flexion_rad" in row and self.adapter is not None:
                    row["encoded_angles_rad"] = np.asarray(row["angles_rad"]).copy()
                    row["encoded_angles_rad"][self.adapter.joints[:, 5]] = row["flexion_rad"]
                if "velocities_rad_s" in row and "flexion_velocity_rad_s" in row and self.adapter is not None:
                    row["encoded_velocities_rad_s"] = np.asarray(row["velocities_rad_s"]).copy()
                    row["encoded_velocities_rad_s"][self.adapter.joints[:, 5]] = row["flexion_velocity_rad_s"]
            except BaseException as error:
                row["complete"] = False
                row["error"] = (row.get("error", "") + "; extraction: " + repr(error)).strip("; ")
            reduced.append(row)
        neural_fields = {"network_tick": ((), np.int64, -1),
                         "time_s": ((), np.float64, np.nan),
                         **{key: ((cells,), np.float32, np.nan)
                            for key in ("rate", "drive", "adaptation", "output_mask", "current")}}
        return make(reduced, encoder_fields), make(self.neural, neural_fields)


def install_factory(e181: Any, collector: Collector):
    original = e181.e150.make_observer
    installed: list[Any] = []

    def factory(*args, **kwargs):
        callback = original(*args, **kwargs)

        def observe(session, detailed, available=None):
            result = callback(session, detailed, available)
            collector.capture_neural(session)
            if session.control_tick == 0:
                adapter = collector.adapter
                encode_original = adapter.encode

                def observed_encode(*a, **k):
                    return collector.capture_encode(encode_original, *a, **k)

                installed.append(temporary_method(adapter, "encode", observed_encode))
            return result

        return observe

    e181.e150.make_observer = factory

    def restore():
        try:
            for undo in reversed(installed):
                undo()
        finally:
            e181.e150.make_observer = original

    return restore


def validate_signals(collector: Collector, encoder: dict, neural: dict) -> dict:
    adapter, provenance = collector.adapter, collector.provenance
    unique = collector.unique_ids
    require(len(encoder["control_tick"]) == CONTROLS
            and len(neural["control_tick"]) == ROWS
            and np.array_equal(encoder["control_tick"], np.arange(CONTROLS))
            and np.array_equal(neural["control_tick"], np.arange(ROWS))
            and encoder["complete"].all() and neural["complete"].all(), "signal rows")
    for data in (encoder, neural):
        ticks = data["control_tick"]
        require(np.array_equal(data["network_tick"], ticks * 20)
                and np.max(np.abs(data["time_s"] - ticks * .005)) <= 2e-8,
                "signal clock")
    require(np.array_equal(encoder["new_filtered"][:-1], encoder["old_filtered"][1:])
            and np.array_equal(encoder["new_velocity_lowpass"][:-1],
                               encoder["old_velocity_lowpass"][1:]), "filter continuity")
    expansion = np.asarray(provenance["port_expansion_indices"], np.int64)
    expansion_ports = np.asarray(provenance["port_expansion_ports"], np.int64)
    for tick in range(CONTROLS):
        q = encoder["angles_rad"][tick].copy()
        velocity = encoder["velocities_rad_s"][tick].copy()
        q[adapter.joints[:, 5]] = encoder["flexion_rad"][tick]
        velocity[adapter.joints[:, 5]] = encoder["flexion_velocity_rad_s"][tick]
        require(np.array_equal(q, encoder["encoded_angles_rad"][tick])
                and np.array_equal(velocity, encoder["encoded_velocities_rad_s"][tick]),
                "encoded knee substitution")
        from flylab.c.coxa_hair_plate import anterior_feature
        from flylab.c.neuromuscular import typed_receptor_feature

        feature_keys = adapter._receptor_keys
        values = np.array([
            anterior_feature(q[adapter.joints[leg, 0]],
                             adapter.body.neutral[adapter.joints[leg, 0]], adapter.coxa_sign[leg])
            if kind == "coxa_limit" else typed_receptor_feature(
                kind, cell, q[adapter.joints[leg, 5]], velocity[adapter.joints[leg, 5]],
                encoder["old_velocity_lowpass"][tick, adapter.joints[leg, 5]],
                encoder["support_load_bw"][tick, leg],
                encoder["non_support_load_bw"][tick, leg], adapter.spec)
            for leg, kind, cell in feature_keys])[adapter._port_features]
        np.testing.assert_allclose(values, encoder["last_features"][tick], atol=1e-12, rtol=0)
        request = adapter.sensory_gain * values
        np.testing.assert_allclose(request, encoder["request"][tick], atol=1e-12, rtol=0)
        expected_filtered = encoder["old_filtered"][tick] + (1-math.exp(-.005/.02)) * (
            request - encoder["old_filtered"][tick])
        expected_lp = encoder["old_velocity_lowpass"][tick] + (1-math.exp(-.005/.03)) * (
            velocity - encoder["old_velocity_lowpass"][tick])
        np.testing.assert_allclose(expected_filtered, encoder["new_filtered"][tick], atol=1e-12, rtol=0)
        np.testing.assert_allclose(expected_lp, encoder["new_velocity_lowpass"][tick], atol=1e-12, rtol=0)
        full = np.zeros(N, np.float32)
        if adapter._sensory_unique:
            full[expansion] += encoder["old_filtered"][tick, expansion_ports]
        else:
            np.add.at(full, expansion, encoder["old_filtered"][tick, expansion_ports])
        require(full[unique].tobytes() == encoder["returned_drive"][tick].tobytes(),
                "old filtered float32 returned drive")
        require(full[unique].tobytes() == neural["drive"][tick+1].tobytes(),
                "one-control delayed sensory network drive")
    return {"encoder_rows": CONTROLS, "neural_rows": ROWS,
            "typed_features_atol": 1e-12, "filter_atol": 1e-12,
            "returned_and_delayed_drive_bit_exact": True,
            "filter_continuity_exact": True}


def stats(values: np.ndarray) -> dict:
    value = np.asarray(values, np.float64).reshape(-1)
    require(value.size > 0 and np.isfinite(value).all(), "stats finite")
    return {"min": float(value.min()), "max": float(value.max()),
            "p5": float(np.percentile(value, 5)), "p95": float(np.percentile(value, 95)),
            "std": float(value.std()), "n": int(value.size)}


def analyze(collector: Collector, encoder: dict, neural: dict) -> dict:
    provenance = collector.provenance
    unique = collector.unique_ids
    cell_lookup = {int(index): p for p, index in enumerate(unique)}
    cap = np.asarray(provenance["network_parameters"]["cap"], np.float32)
    threshold = np.asarray(provenance["network_parameters"]["threshold"], np.float32)
    require(cap.shape == threshold.shape == (len(unique),), "network parameter shape")
    net = (neural["drive"].astype(np.float64) + neural["current"].astype(np.float64)
           - threshold.astype(np.float64)[None, :] - neural["adaptation"].astype(np.float64))
    phases = {"startup": (0, 100), "evaluation": (100, 600)}
    result = {"schema": "flylab.m3.e193.sensory-transmission.analysis.v1",
              "phase_boundaries": phases, "port_rows": [], "cell_rows": [],
              "endpoint_semantics": "network boundary k+1 paired to encode tick k; no RK intermediate current",
              "interpretation": "descriptive single-trajectory evidence only"}
    portmap = np.asarray(provenance["port_expansion_ports"])
    expansion = np.asarray(provenance["port_expansion_indices"])
    joint = np.asarray(provenance["joints"])
    for phase, (lo, hi) in phases.items():
        for p, port in enumerate(provenance["ports"]):
            leg = port["leg_index"]
            indices = [cell_lookup[int(index)] for index in expansion[portmap == p]]
            old = encoder["old_filtered"][lo:hi, p]
            feature = encoder["last_features"][lo:hi, p]
            request = provenance["sensory_gain_model_units"] * feature
            row = {"phase": phase, "port": p, "leg_index": leg,
                   "kind": port["kind"], "cell_type": port["cell_type"],
                   "cells": len(indices),
                   "physical": {"knee_rad": stats(encoder["flexion_rad"][lo:hi, leg]),
                                "knee_velocity_rad_s": stats(encoder["flexion_velocity_rad_s"][lo:hi, leg]),
                                "support_load_bw": stats(encoder["support_load_bw"][lo:hi, leg]),
                                "non_support_load_bw": stats(encoder["non_support_load_bw"][lo:hi, leg]),
                                "joint_velocity_rad_s": stats(encoder["velocities_rad_s"][lo:hi, joint[leg, 5]])},
                   "feature": stats(feature), "request": stats(request),
                   "old_filtered_drive": stats(old),
                   "feature_exact_zero_fraction": float(np.mean(feature == 0)),
                   "feature_exact_one_fraction": float(np.mean(feature == 1)),
                   "connected_rate": stats(neural["rate"][lo+1:hi+1, indices]),
                   "connected_rate_at_least_95pct_cap_fraction": float(np.mean(
                       neural["rate"][lo+1:hi+1, indices] >= .95 * cap[indices][None, :])),
                   "connected_endpoint_net_nonpositive_fraction": float(np.mean(
                       net[lo+1:hi+1, indices] <= 0))}
            result["port_rows"].append(row)
        for cell_pos, index in enumerate(unique):
            result["cell_rows"].append({"phase": phase, "index": int(index),
                "rate": stats(neural["rate"][lo+1:hi+1, cell_pos]),
                "drive": stats(neural["drive"][lo+1:hi+1, cell_pos]),
                "current": stats(neural["current"][lo+1:hi+1, cell_pos]),
                "adaptation": stats(neural["adaptation"][lo+1:hi+1, cell_pos]),
                "endpoint_net": stats(net[lo+1:hi+1, cell_pos]),
                "endpoint_net_nonpositive_fraction": float(np.mean(net[lo+1:hi+1, cell_pos] <= 0)),
                "rate_at_least_95pct_cap_fraction": float(np.mean(
                    neural["rate"][lo+1:hi+1, cell_pos] >= .95 * cap[cell_pos]))})
    return result


def execute(freeze_path: Path, review_path: Path, output: Path,
            parent_validation_sha256: str) -> None:
    began = _ENTRY_STARTED
    metadata, _, frozen_sha, receipt = gate(freeze_path, review_path, output,
                                            parent_validation_sha256)
    readiness = parent_readiness(receipt)
    require(shutil.disk_usage(output.parent).free >= FREE_BYTES, "20 GiB free")
    output.mkdir(exist_ok=False)
    terminal = {"schema": "flylab.m3.e193.sensory-transmission.terminal.v1",
                "status": "failed", "original_runs": 0, "automatic_retries": 0,
                "freeze_sha256": frozen_sha, "source_review_sha256": sha256(review_path),
                "parent_validation_sha256": parent_validation_sha256}
    collector = Collector(metadata)
    result = None
    monitor = None
    monitor_stop = threading.Event()
    monitor_errors: list[str] = []
    cleanup_errors: list[dict] = []
    execution_error: BaseException | None = None
    try:
        from tools import optimize_c_banc_m3_native_timing as e181
        from tools import recover_c_banc_m3_native_timing as e182

        write_new(output / "readiness.json", {
            "schema": "flylab.m3.e193.sensory-transmission.readiness.v1",
            "observations": readiness, "native_permitted": True,
            "source": "same live parent receipt"})
        old = metadata["e181_gate"]
        verified = e181.verified_inputs(safe(old["freeze_path"]), old["freeze_sha256"],
                                        safe(old["source_review_path"]),
                                        old["source_review_sha256"], mode="baseline")
        require(verified.mode == "baseline", "actual E181 baseline gate")
        monitor = threading.Thread(target=resources.monitor,
            args=(output / "resource-monitor.jsonl", monitor_stop, monitor_errors, e182),
            daemon=True)
        monitor.start()
        threads = restore = None
        primary = None
        try:
            threads, _ = e181.protocol.joint_fit.one_cpu_thread()
            startup = e181._load_startup(verified.freeze)
            graph = e181.GraphStore.load(safe(verified.freeze["graph_path"]))
            require(graph.hash == verified.freeze["graph_hash"], "graph identity")
            neutral = e181.StateStore.load(safe(verified.freeze["neutral_checkpoint"]))
            restore = install_factory(e181, collector)
            terminal["original_runs"] = 1
            result = e181.run_arm(graph, neutral, (0.0,) * 6, output / "baseline",
                startup=startup, limits=e181.LIMITS, wall_limit_s=100,
                gate=verified, process_began=began)
        except BaseException as error:
            primary = error
        finally:
            if restore is not None:
                try:
                    restore()
                except BaseException as error:
                    cleanup_errors.append({"artifact": "restore", "error": repr(error)})
            if threads is not None:
                try:
                    threads.restore_original_limits()
                except BaseException as error:
                    cleanup_errors.append({"artifact": "threads", "error": repr(error)})
            try:
                captured = collector.arrays()
            except BaseException as error:
                captured = ()
                cleanup_errors.append({"artifact": "partial_arrays", "error": repr(error)})
            for artifact, data in zip(("encoder-signals.npz", "sensory-signals.npz"), captured):
                try:
                    np.savez_compressed(output / artifact, **data)
                except BaseException as error:
                    cleanup_errors.append({"artifact": artifact, "error": repr(error)})
            try:
                write_new(output / "runtime-provenance.json", {
                    "schema": "flylab.m3.e193.sensory-transmission.runtime-provenance.v1",
                    **collector.provenance, "encoder_rows": len(collector.encoder),
                    "neural_rows": len(collector.neural), "observer_error": collector.error})
            except BaseException as error:
                cleanup_errors.append({"artifact": "runtime-provenance", "error": repr(error)})
            monitor_stop.set()
            if monitor is not None:
                monitor.join(timeout=5)
                if monitor.is_alive():
                    cleanup_errors.append({"artifact": "monitor_join", "error": "thread alive"})
        if primary is not None:
            raise primary.with_traceback(primary.__traceback__)
        require(not cleanup_errors, "observer cleanup/persistence error")
        rows = e182._resource_rows(output / "resource-monitor.jsonl")
        require(rows and not monitor_errors and all(resources.resource_clear(row) for row in rows),
                "resource monitor")
        require(time.perf_counter() - began <= INTERNAL_S, "internal 140s")
        encoder, neural = collector.arrays()
        e188.validate_run(result, neural)
        signal_check = validate_signals(collector, encoder, neural)
        comparison = e188.compare_baseline(output / "baseline", safe(metadata["baseline_directory"]))
        validity = {"schema": "flylab.m3.e193.sensory-transmission.validity.v1",
                    "valid_3s_diagnostic": True, "baseline_exact": comparison,
                    "signals": signal_check, "e181_result": result}
        write_new(output / "validity.json", validity)
        write_new(output / "analysis.json", analyze(collector, encoder, neural))
        require(time.perf_counter() - began <= INTERNAL_S, "internal 140s")
        baseline_bytes = sum(p.stat().st_size for p in (output / "baseline").rglob("*") if p.is_file())
        output_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        require(baseline_bytes <= BASELINE_BYTES and output_bytes <= OUTPUT_BYTES,
                "output size")
        terminal.update(status="complete", baseline_bytes=baseline_bytes,
                        output_bytes=output_bytes, encoder_rows=CONTROLS, neural_rows=ROWS)
    except BaseException as error:
        execution_error = error
        terminal["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if monitor is not None and monitor.is_alive():
            monitor_stop.set()
            monitor.join(timeout=5)
        terminal["elapsed_wall_s"] = time.perf_counter() - began
        terminal["e181_result_validity"] = result.get("validity") if result else None
        terminal["cleanup_persistence_errors"] = cleanup_errors
        try:
            write_new(output / "terminal.json", terminal)
        except BaseException as error:
            if execution_error is None:
                raise
            print("E193 terminal persistence error: " + repr(error), file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parent-validation-sha256", required=True)
    args = parser.parse_args()
    execute(args.freeze, args.review, args.output, args.parent_validation_sha256)


if __name__ == "__main__":
    main()
