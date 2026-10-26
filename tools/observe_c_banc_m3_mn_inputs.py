"""E188: one original, read-only CUDA observation of every MN144018 input.

Import and --help perform no model construction, CUDA work, or source analysis.
The parent owns the frozen metadata, source approval, and process supervisor.
"""

# Broad catches retain primary neural/body failures while saving partial evidence.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any

_ENTRY_STARTED = time.perf_counter()

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import banc_m3_resource_readiness as resource_readiness
from tools.analyze_c_banc_m3_saved_current import exact_bits, warp_current

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
DESIGN = "docs/BANC_M3_E188_ALL_INCOMING_DESIGN_20260924.md"
METADATA = BASE + "/parent-e188-frozen-metadata.json"
OUTPUT = BASE + "/e188-mn-inputs"
SUPERVISOR = BASE + "/parent-e188-supervise.py"
PARENT_VALIDATION = BASE + "/parent-e188-input-validation.json"
AMENDMENT = "docs/BANC_M3_E188_PARENT_GATE_AMENDMENT_20260924.md"
SERVICE_AMENDMENT = "docs/BANC_M3_E188_STOPPED_SERVICE_AMENDMENT_20260924.md"
OWNED = ("tools/observe_c_banc_m3_mn_inputs.py", "tests/test_c_banc_mn_inputs.py")
RESOURCE_OWNED = (
    "tools/banc_m3_resource_readiness.py",
    "tests/test_c_banc_resource_readiness.py",
)
POST = 144018
ROWS = 601
EDGES = 195
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
    "cumulative_before_s": 4351.0849566997495,
    "cumulative_cap_s": 4994.0,
}
SELECTED = (131, 219, 308, 391, 433, 480, 521, 564)
EVENTS = (131, 148, 219, 256, 308, 345, 391, 433, 480, 521, 564)
BANDS = {"pre": (-20, -10), "lead": (-10, 0), "post": (0, 10), "recover": (10, 21)}


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(name: str) -> Path:
    require(isinstance(name, str) and name and "\\" not in name, "unsafe path")
    relative = PurePosixPath(name)
    require(
        not relative.is_absolute()
        and ":" not in name
        and all(part not in ("", ".", "..") for part in relative.parts),
        "unsafe path",
    )
    path = ROOT.joinpath(*relative.parts)
    require(path.resolve().is_relative_to(ROOT.resolve()), "path escaped repository")
    require(
        not any(p.is_symlink() for p in (path, *path.parents) if p != ROOT.parent),
        "symlink source",
    )
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def input_stamps(inputs: dict[str, str]) -> dict[str, list[int]]:
    """Detect file replacement or edits around the parent's full SHA validation."""
    result = {}
    for name in inputs:
        relative = PurePosixPath(name)
        require(
            not relative.is_absolute()
            and ":" not in name
            and "\\" not in name
            and not any(part in (".", "..") for part in relative.parts),
            "unsafe input stamp path",
        )
        stat = ROOT.joinpath(*relative.parts).stat()
        result[name] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
    return result


def validate_parent_inputs(
    inputs: dict[str, str],
    freeze_sha: str,
    review_sha: str,
    metadata_sha: str,
    receipt_sha: str,
) -> None:
    """Accept only a fresh receipt from this live, approved supervisor process."""
    import psutil

    path = safe(PARENT_VALIDATION)
    require(
        len(receipt_sha) == 64 and sha256(path) == receipt_sha, "parent receipt SHA"
    )
    receipt = read_json(path)
    require(
        receipt.get("schema") == "flylab.m3.e188.parent-input-validation.v1"
        and receipt.get("validated") is True
        and receipt.get("freeze_sha256") == freeze_sha
        and receipt.get("source_review_sha256") == review_sha
        and receipt.get("metadata_sha256") == metadata_sha
        and receipt.get("input_sha256") == inputs,
        "parent receipt contract",
    )
    parent = receipt["parent"]
    if parent["pid"] != os.getppid():
        # Windows venv python.exe launches the base Python interpreter. Accept
        # exactly that one shim, with the same executable and child arguments.
        launcher = psutil.Process(os.getppid())
        require(
            os.name == "nt"
            and launcher.is_running()
            and getattr(launcher, "ppid", lambda: -1)() == parent["pid"]
            and Path(launcher.exe()).resolve() == Path(sys.executable).resolve()
            and launcher.cmdline()[1:] == psutil.Process().cmdline()[1:],
            "different parent process or unverified venv launcher",
        )
    live = psutil.Process(parent["pid"])
    require(
        live.is_running() and abs(live.create_time() - parent["create_time"]) <= 0.001,
        "parent process identity changed",
    )
    script = safe(SUPERVISOR).resolve()
    require(
        any(
            (Path(live.cwd()) / arg).resolve() == script
            for arg in live.cmdline()
            if str(arg).endswith(".py")
        ),
        "unexpected parent supervisor command",
    )
    require(
        0 <= time.perf_counter() - receipt["issued_monotonic_s"] <= 145,
        "expired parent receipt",
    )
    require(
        input_stamps(inputs) == receipt.get("input_stamps"), "parent input stat changed"
    )


def gate(
    freeze_path: Path,
    review_path: Path,
    output: Path,
    parent_validation_sha256: str | None = None,
) -> tuple[dict, dict, str]:
    """Parent review, exact source/data hashes, and prior evidence before E181 import."""
    require(
        not output.exists() and output.parent.is_dir(),
        "output exists or parent missing",
    )
    freeze, review = read_json(freeze_path), read_json(review_path)
    frozen_sha = sha256(freeze_path)
    require(
        freeze.get("schema") == "flylab.m3.e188.all-incoming.freeze.v1", "freeze schema"
    )
    require(
        review.get("schema") == "flylab.m3.e188.all-incoming.source-review.v1",
        "review schema",
    )
    require(
        review.get("approved") is True
        and review.get("blocking_defects") == []
        and review.get("freeze_sha256") == frozen_sha,
        "parent approval",
    )
    require(
        freeze.get("design_path") == DESIGN
        and freeze.get("design_sha256") == sha256(safe(DESIGN)),
        "design changed",
    )
    require(
        freeze.get("metadata_path") == METADATA
        and freeze.get("metadata_sha256") == sha256(safe(METADATA)),
        "metadata changed",
    )
    require(
        freeze.get("output_path") == OUTPUT
        and output.resolve() == safe(OUTPUT).resolve(),
        "output identity",
    )
    require(freeze.get("parameters") == PARAMETERS, "resource contract")
    require(
        freeze.get("amendment_path") == AMENDMENT
        and freeze.get("amendment_sha256") == sha256(safe(AMENDMENT))
        and review.get("amendment_sha256") == freeze["amendment_sha256"],
        "parent validation amendment changed",
    )
    require(
        freeze.get("service_amendment_path") == SERVICE_AMENDMENT
        and freeze.get("service_amendment_sha256") == sha256(safe(SERVICE_AMENDMENT))
        and review.get("service_amendment_sha256") == freeze["service_amendment_sha256"],
        "stopped service amendment changed",
    )
    require(
        freeze.get("parent_supervisor_sha256") == sha256(safe(SUPERVISOR)),
        "parent supervisor changed",
    )
    metadata = read_json(safe(METADATA))
    require(
        metadata.get("schema") == "flylab.m3.e188.all-incoming.metadata.v1"
        and metadata.get("design_path") == DESIGN
        and metadata.get("design_sha256") == freeze["design_sha256"]
        and metadata.get("owned_files") == list(OWNED)
        and metadata.get("original_native_runs") == 0
        and metadata.get("F6_exposed") == 0
        and metadata.get("candidate_adopted") is False
        and metadata.get("cumulative_before_s") == PARAMETERS["cumulative_before_s"]
        and metadata.get("cumulative_limit_s") == PARAMETERS["cumulative_cap_s"],
        "metadata contract",
    )
    source_hashes = freeze.get("source_hashes")
    require(
        isinstance(source_hashes, dict)
        and {*OWNED, *RESOURCE_OWNED} <= set(source_hashes)
        and isinstance(metadata.get("reused_sources"), dict),
        "source inventory",
    )
    require(
        review.get("source_hashes") == source_hashes
        and review.get("design_sha256") == freeze["design_sha256"]
        and review.get("metadata_sha256") == freeze["metadata_sha256"],
        "review identity",
    )
    require(
        set(metadata["reused_sources"]) <= set(source_hashes), "reused sources omitted"
    )
    for name, expected in source_hashes.items():
        require(
            len(expected) == 64 and sha256(safe(name)) == expected,
            "source changed: " + name,
        )
    for name, expected in metadata["reused_sources"].items():
        require(source_hashes[name] == expected, "reused source changed: " + name)
    inputs = metadata.get("inputs")
    require(
        isinstance(inputs, dict) and inputs and freeze.get("input_sha256") == inputs,
        "input inventory",
    )
    if parent_validation_sha256 is None:
        for name, expected in inputs.items():
            require(
                len(expected) == 64 and sha256(safe(name)) == expected,
                "input changed: " + name,
            )
    else:
        validate_parent_inputs(
            inputs,
            frozen_sha,
            sha256(review_path),
            freeze["metadata_sha256"],
            parent_validation_sha256,
        )
    e187_analysis = metadata["e187_directory"] + "/analysis.json"
    require(
        sha256(safe(e187_analysis)) == inputs[e187_analysis], "E187 analysis identity"
    )
    old = metadata["e181_gate"]
    for name, expected in (
        (old["freeze_path"], old["freeze_sha256"]),
        (old["source_review_path"], old["source_review_sha256"]),
    ):
        require(sha256(safe(name)) == expected, "E181 gate changed")
    previous = read_json(safe(metadata["previous_review_path"]))
    require(
        previous.get("schema") == "flylab.m3.e187.parent-final-review.v1"
        and previous.get("run_status") == "completed"
        and previous.get("run_validity") == "valid_saved_current_diagnostic"
        and previous.get("native_runs") == 0
        and previous.get("boundary_current_bit_exact_rows") == 25
        and previous.get("candidate_adopted") is False,
        "E187 parent review",
    )
    require(
        read_json(safe(metadata["e185_review_path"])).get("saved_raw_analysis_accepted")
        is True,
        "E185 parent review",
    )
    inventory = metadata.get("row_inventory")
    require(
        isinstance(inventory, list) and len(inventory) == EDGES,
        "195 incoming edges required",
    )
    edge_ids = np.asarray([row["edge_id"] for row in inventory], np.int64)
    pre_ids = np.asarray([row["pre_id"] for row in inventory], np.int64)
    weights = np.asarray([row["actual_weight"] for row in inventory], np.float32)
    require(
        np.array_equal(edge_ids, np.arange(edge_ids[0], edge_ids[0] + EDGES))
        and np.all((pre_ids >= 0) & (pre_ids < 158706))
        and np.isfinite(weights).all()
        and all(row.get("post_id") == POST for row in inventory),
        "incoming edge order",
    )
    return metadata, freeze, frozen_sha


def parent_readiness(receipt_sha: str) -> list[dict]:
    """Use the source-approved parent's fresh raw observations in its 150s budget."""
    path = safe(PARENT_VALIDATION)
    require(sha256(path) == receipt_sha, "parent readiness receipt changed")
    receipt = read_json(path)
    age = time.perf_counter() - receipt["resource_readiness_finished_monotonic_s"]
    require(math.isfinite(age) and 0 <= age <= 10, "parent readiness expired")
    rows = receipt.get("resource_readiness")
    require(
        isinstance(rows, list)
        and len(rows) == 2
        and all(
            row.get("phase") == "readiness"
            and row.get("ready") is True
            and row.get("resource_clear") is True
            and row.get("resource_mode") in ("queue_empty", "service_stopped")
            for row in rows
        ),
        "parent readiness not clear",
    )
    return rows


def tensor_array(
    value: Any, dtype: Any, shape: tuple[int, ...], label: str
) -> np.ndarray:
    array = value.detach().cpu().numpy().copy()
    require(
        array.shape == shape and array.dtype == dtype and np.isfinite(array).all(),
        label + " shape/dtype/finite",
    )
    return array


def validate_csr(network: Any, inventory: list[dict]) -> dict:
    """Confirm the intact CSR and packed lookup agree in original row order."""
    require(
        network.device == "cuda"
        and network.n == 158706
        and network.cuda_implementation == "packed",
        "full packed CUDA required",
    )
    weights = network.weights
    ptr = weights.crow_indices()[POST : POST + 2].detach().cpu().numpy().copy()
    require(ptr.shape == (2,) and ptr.dtype.kind in "iu", "CSR ptr")
    start, end = map(int, ptr)
    require(end - start == EDGES, "CSR row length")
    pre = weights.col_indices()[start:end].detach().cpu().numpy().copy()
    require(pre.shape == (EDGES,) and pre.dtype.kind in "iu", "CSR pre")
    actual = tensor_array(
        weights.values()[start:end], np.float32, (EDGES,), "CSR weight"
    )
    ids = np.asarray([row["edge_id"] for row in inventory], np.int64)
    expected_pre = np.asarray([row["pre_id"] for row in inventory], np.int64)
    expected_weights = np.asarray(
        [row["actual_weight"] for row in inventory], np.float32
    )
    require(
        np.array_equal(ids, np.arange(start, end))
        and np.array_equal(pre, expected_pre),
        "runtime CSR edge/pre order",
    )
    exact_bits(actual, expected_weights, "runtime CSR weights")
    packed_ptr = tensor_array(
        network._rowptr[POST : POST + 2], np.int32, (2,), "packed ptr"
    )
    require(np.array_equal(packed_ptr, ptr), "packed row pointer")
    encoded = tensor_array(
        network._columns[start:end], np.int32, (EDGES,), "packed columns"
    ).view(np.uint32)
    lookup = tensor_array(
        network._weight_values,
        np.float32,
        (int(network._weight_values.numel()),),
        "packed weights",
    )
    decoded_pre = (encoded & np.uint32((1 << 18) - 1)).astype(np.int64)
    weight_index = (encoded >> np.uint32(18)).astype(np.int64)
    require(
        np.array_equal(decoded_pre, expected_pre)
        and np.all(weight_index < len(lookup)),
        "packed pre/lookup index",
    )
    exact_bits(lookup[weight_index], actual, "packed weight lookup")
    return {
        "csr_start": start,
        "csr_end": end,
        "row_edges": EDGES,
        "csr_pre_order_equal": True,
        "csr_weight_bits_equal": True,
        "packed_pre_order_equal": True,
        "packed_weight_bits_equal": True,
        "cuda_implementation": network.cuda_implementation,
        "device": str(network.device),
    }


class Collector:
    def __init__(self, metadata: dict):
        self.inventory = metadata["row_inventory"]
        inventory = self.inventory
        self.metadata = metadata
        self.pre_ids = np.asarray([row["pre_id"] for row in inventory], np.int64)
        self.edge_ids = np.asarray([row["edge_id"] for row in inventory], np.int64)
        self.weights = np.asarray(
            [row["actual_weight"] for row in inventory], np.float32
        )
        self.rows: list[dict] = []
        self.provenance: dict[str, Any] = {}
        self.pre_index: Any = None
        self.error: str | None = None

    def capture(self, session: Any) -> None:
        network = session.network
        tick = int(session.control_tick)
        row: dict[str, Any] = {"control_tick": tick, "complete": False}
        self.rows.append(row)
        try:
            before = (tick, int(network.tick), float(session.body.physics_time()))
            row.update(network_tick=before[1], time_s=before[2])
            require(
                tick == len(self.rows) - 1
                and tick <= 600
                and before[1] == tick * 20
                and abs(before[2] - tick * 0.005) <= 2e-8,
                "observer clock/order",
            )
            if tick == 0:
                require(
                    session.graph.n == 158706 and network.tick == 0 and before[2] == 0,
                    "full graph/initial clocks",
                )
                self.provenance = validate_csr(network, self.inventory)
                require(
                    session.graph.hash == self.metadata["expected_graph_hash"]
                    and network.identity == self.metadata["expected_network_identity"]
                    and session.adapter.identity
                    == self.metadata["expected_adapter_identity"]
                    and session.body.model_hash == self.metadata["expected_body_hash"],
                    "runtime graph/network/adapter/body identity",
                )
                self.pre_index = network.torch.as_tensor(self.pre_ids, device="cuda")
                self.provenance.update(
                    graph_neurons=int(session.graph.n),
                    network_identity=network.identity,
                    adapter_identity=session.adapter.identity,
                    body_model_hash=session.body.model_hash,
                    initial_clock=list(before),
                )
            # GPU clones and equality checks prove the extra kernel is observational.
            fields = ("rate", "drive", "adaptation", "output_mask")
            originals = {name: getattr(network, name).clone() for name in fields}
            current = network._current(network.rate)
            same = all(
                bool(network.torch.equal(originals[name], getattr(network, name)))
                for name in fields
            )
            require(
                same
                and before
                == (
                    int(session.control_tick),
                    int(network.tick),
                    float(session.body.physics_time()),
                ),
                "readout changed state or clock",
            )
            row["pre_rate"] = tensor_array(
                network.rate[self.pre_index], np.float32, (EDGES,), "pre rate"
            )
            row["pre_mask"] = tensor_array(
                network.output_mask[self.pre_index], np.float32, (EDGES,), "pre mask"
            )
            for name in ("rate", "drive", "adaptation"):
                row["post_" + name] = tensor_array(
                    getattr(network, name)[POST : POST + 1],
                    np.float32,
                    (1,),
                    "post " + name,
                )[0]
            row["post_current"] = tensor_array(
                current[POST : POST + 1], np.float32, (1,), "post current"
            )[0]
            require(
                before
                == (
                    int(session.control_tick),
                    int(network.tick),
                    float(session.body.physics_time()),
                ),
                "readout clock changed",
            )
            if tick == 600:
                self.provenance["final_clock"] = list(before)
                validate_csr(network, self.inventory)
            row["complete"] = True
        except BaseException as error:
            row["error"] = repr(error)
            self.error = repr(error)
            raise

    def arrays(self) -> dict[str, np.ndarray]:
        n = len(self.rows)

        def stack(key: str, width: int | None = None, dtype: Any = np.float32):
            shape = (n,) if width is None else (n, width)
            fill = np.nan if np.dtype(dtype).kind == "f" else -1
            result = np.full(shape, fill, dtype=dtype)
            if dtype == np.bool_:
                result.fill(False)
            for i, row in enumerate(self.rows):
                if key in row:
                    result[i] = row[key]
            return result

        arrays = {
            "control_tick": stack("control_tick", dtype=np.int64),
            "network_tick": stack("network_tick", dtype=np.int64),
            "time_s": stack("time_s", dtype=np.float64),
            "pre_ids": self.pre_ids,
            "edge_ids": self.edge_ids,
            "weights": self.weights,
            "pre_rate": stack("pre_rate", EDGES),
            "pre_mask": stack("pre_mask", EDGES),
            "post_rate": stack("post_rate"),
            "post_drive": stack("post_drive"),
            "post_adaptation": stack("post_adaptation"),
            "post_current": stack("post_current"),
            "complete": stack("complete", dtype=np.bool_),
            "row_error": np.asarray([row.get("error", "") for row in self.rows]),
            "error": np.array(self.error or ""),
        }
        for name in (
            "pre_rate",
            "pre_mask",
            "post_rate",
            "post_drive",
            "post_adaptation",
            "post_current",
        ):
            arrays[name + "_available"] = np.asarray(
                [name in row for row in self.rows], np.bool_
            )
        return arrays


def install_factory(e181: Any, collector: Collector):
    """Call E181's original callback first and restore its factory in finally."""
    original = e181.e150.make_observer

    def factory(*args, **kwargs):
        callback = original(*args, **kwargs)

        def observe(session, detailed, available=None):
            try:
                result = callback(session, detailed, available)
                collector.capture(session)
                return result
            except BaseException as error:
                collector.error = repr(error)
                raise

        return observe

    e181.e150.make_observer = factory

    def restore():
        e181.e150.make_observer = original

    return restore


def exact_array(left: Any, right: Any, label: str) -> None:
    a, b = np.asarray(left), np.asarray(right)
    require(a.shape == b.shape and a.dtype == b.dtype, label + " header")
    if a.dtype.kind in "fc":
        require(a.tobytes() == b.tobytes(), label + " bits")
    else:
        require(np.array_equal(a, b), label + " values")


def compare_state(candidate: Path, reference: Path) -> None:
    a, b = read_json(candidate / "state.json"), read_json(reference / "state.json")

    def walk(x: Any, y: Any, key: str) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            if set(x) == set(y) == {"__npy__"}:
                exact_array(
                    np.load(candidate / x["__npy__"], allow_pickle=False),
                    np.load(reference / y["__npy__"], allow_pickle=False),
                    key,
                )
                return
            require(set(x) == set(y), "state keys " + key)
            for part in x:
                if not key and part == "wall_s":
                    continue
                walk(x[part], y[part], key + "/" + part)
        elif isinstance(x, list) and isinstance(y, list):
            require(len(x) == len(y), "state list " + key)
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, key + "/" + str(i))
        else:
            require(x == y, "state value " + key)

    walk(a, b, "")


def compare_baseline(candidate: Path, reference: Path) -> dict:
    left, right = (read_json(p / "trace.json") for p in (candidate, reference))
    require(
        len(left["samples"]) == len(right["samples"]) == ROWS, "baseline trace length"
    )
    require(left["samples"] == right["samples"], "baseline trace raw mismatch")
    for filename, expected in (
        ("native-recruitment-state.npz", ROWS),
        ("native-contact.npz", 30000),
    ):
        with (
            np.load(candidate / filename, allow_pickle=False) as a,
            np.load(reference / filename, allow_pickle=False) as b,
        ):
            require(set(a.files) == set(b.files), filename + " members")
            for key in a.files:
                require(
                    a[key].ndim == 0 or len(a[key]) == expected,
                    filename + "/" + key + " length",
                )
                exact_array(a[key], b[key], filename + "/" + key)
    for boundary in ("start", "final"):
        compare_state(candidate / boundary, reference / boundary)
    return {
        "reference": str(reference.relative_to(ROOT)),
        "trace_rows": ROWS,
        "recruitment_rows": ROWS,
        "native_contact_rows": 30000,
        "initial_final_state_equal": True,
        "all_raw_exact": True,
    }


def validate_run(result: dict, arrays: dict) -> None:
    require(
        result.get("validity") == "valid_3s_diagnostic"
        and result.get("complete_3s") is True
        and result.get("clock_valid") is True
        and result.get("startup_pass") is True
        and result.get("technical_failure") is False
        and result.get("measured_health_stop") is False
        and result.get("completed_controls") == 600
        and result.get("native_samples") == 30000
        and abs(float(result.get("actual_neural_s", -1)) - 3) <= 1e-10
        and abs(float(result.get("actual_physical_s", -1)) - 3) <= 1e-8,
        "E181 native 3s result invalid",
    )
    require(
        len(arrays["control_tick"]) == ROWS
        and np.array_equal(arrays["control_tick"], np.arange(ROWS))
        and np.array_equal(arrays["network_tick"], np.arange(ROWS) * 20)
        and np.max(np.abs(arrays["time_s"] - np.arange(ROWS) * 0.005)) <= 2e-8
        and np.all(arrays["complete"]),
        "incoming endpoint clock/completeness",
    )


def decomposition(arrays: dict, known: np.ndarray) -> dict[str, np.ndarray]:
    """Exact CUDA total and descriptive signed terms across all 195 inputs."""
    rates, masks, weights = arrays["pre_rate"], arrays["pre_mask"], arrays["weights"]
    require(
        rates.shape == masks.shape == (ROWS, EDGES)
        and weights.shape == (EDGES,)
        and rates.dtype == masks.dtype == weights.dtype == np.float32
        and known.shape == (EDGES,),
        "incoming series shape/dtype",
    )
    products = np.asarray(rates * masks, np.float32)
    terms = products.astype(np.float64) * weights.astype(np.float64)[None, :]
    gpu = arrays["post_current"]
    reconstructed = np.empty(ROWS, np.float32)
    full, old, new = (np.empty(ROWS, np.float64) for _ in range(3))
    for i in range(ROWS):
        reconstructed[i] = warp_current(weights, rates[i], masks[i], mode="packed")
        full[i] = math.fsum(terms[i])
        old[i] = math.fsum(terms[i, known])
        new[i] = math.fsum(terms[i, ~known])
    exact_bits(reconstructed, gpu, "601 CUDA endpoint currents")
    rounding = gpu.astype(np.float64) - full
    require(
        np.allclose(full, old + new, atol=1e-9, rtol=1e-12),
        "known/unknown decomposition",
    )
    return {
        "edge_contribution": terms,
        "full_sum": full,
        "known_sum": old,
        "unknown_sum": new,
        "rounding_residual": rounding,
        "reconstructed_current": reconstructed,
    }


def analyze(metadata: dict, arrays: dict) -> tuple[dict, dict]:
    from tools import analyze_c_banc_m3_mn_current as e187

    previous = read_json(safe(metadata["e187_directory"] + "/analysis.json"))
    require(
        previous["schema"] == "flylab.m3.e187.mn-current.analysis.v1"
        and len(previous["edge_inventory"]) >= EDGES,
        "E187 evidence",
    )
    with np.load(
        safe(metadata["e187_directory"] + "/aligned.npz"), allow_pickle=False
    ) as archive:
        old = {key: archive[key] for key in archive.files}
    with np.load(
        safe(metadata["e186_directory"] + "/aligned.npz"), allow_pickle=False
    ) as archive:
        force = {key: archive[key] for key in archive.files}
    windows = e187.windows_from_e186(force)
    inventory = metadata["row_inventory"]
    lookup = {
        int(row["edge_id"]): i for i, row in enumerate(previous["edge_inventory"])
    }
    old_positions = np.asarray([lookup[int(x)] for x in arrays["edge_ids"]], np.int64)
    known = old["edge_observed_pre"][old_positions]
    require(known.shape == (EDGES,) and known.sum() == 10, "E187 known-10 edge mask")
    for j, row in enumerate(inventory):
        previous_row = previous["edge_inventory"][int(old_positions[j])]
        require(row == previous_row, "E187 frozen edge inventory " + str(j))
    for new_key, old_key in (
        ("post_rate", "rate"),
        ("post_drive", "drive"),
        ("post_adaptation", "adaptation"),
        ("post_current", "recurrent_current"),
    ):
        exact_array(arrays[new_key], old[old_key][:, 0], "E187 " + new_key)
    calculated = decomposition(arrays, known)
    old_terms = old["edge_contribution"][:, old_positions]
    exact_array(
        calculated["edge_contribution"][:, known],
        old_terms[:, known],
        "E187 known edge contributions",
    )
    old_residual = old["residual"][:, 0]
    require(
        np.allclose(
            old_residual,
            calculated["unknown_sum"] + calculated["rounding_residual"],
            atol=1e-9,
            rtol=1e-12,
        ),
        "E187 residual explanation",
    )
    features = {
        "edge_contribution": calculated["edge_contribution"],
        "known_sum": calculated["known_sum"][:, None],
        "unknown_sum": calculated["unknown_sum"][:, None],
        "rounding_residual": calculated["rounding_residual"][:, None],
        "full_sum": calculated["full_sum"][:, None],
    }
    aligned = {
        "window_labels": force["window_labels"],
        "condition_native_count_rm": force["condition_native_count"][:, 4],
        "native_count_rm": force["native_count"][:, 4],
        "native_negative_magnitude_sum_bw_rm": force[
            "native_negative_magnitude_sum_bw"
        ][:, 4],
        "edge_ids": arrays["edge_ids"],
        "pre_ids": arrays["pre_ids"],
        "edge_observed_pre": known,
        **calculated,
    }
    for name, values in features.items():
        results = [
            e187.conditioned(
                values,
                rows,
                force["native_count"],
                force["native_negative_magnitude_sum_bw"],
            )
            for rows in windows.values()
        ]
        for index, label in enumerate(
            (
                "condition_mean",
                "condition_valid_count",
                "magnitude_mean",
                "magnitude_weight",
                "time_mean",
                "time_valid_count",
            )
        ):
            aligned[label + "_" + name] = np.asarray([row[index] for row in results])
    require(
        np.allclose(
            old["condition_mean_residual"][:, :, 0],
            (
                aligned["condition_mean_unknown_sum"]
                + aligned["condition_mean_rounding_residual"]
            )[:, :, 0],
            atol=1e-9,
            rtol=1e-12,
            equal_nan=True,
        ),
        "E187 conditioned residual explanation",
    )
    condition = aligned["condition_mean_edge_contribution"]
    neg, pos = condition[0, 0], condition[0, 2]
    score = np.abs(pos - neg)
    ranking = np.argsort(-score, kind="stable")
    summary = {
        "schema": "flylab.m3.e188.mn-inputs.analysis.v1",
        "validity": "single-trajectory descriptive attribution",
        "post_id": POST,
        "incoming_edges": inventory,
        "known_edge_count": int(known.sum()),
        "newly_observed_edge_count": int((~known).sum()),
        "cuda_current_bit_exact_601": True,
        "e187_known_series_exact": True,
        "e187_residual_decomposition_601": True,
        "e187_conditioned_residual_decomposition_52": True,
        "windows": list(windows),
        "all_events": list(EVENTS),
        "selected_events": list(SELECTED),
        "bands": BANDS,
        "whole_rm_negative_count": int(force["condition_native_count"][0, 4, 0]),
        "whole_rm_positive_count": int(force["condition_native_count"][0, 4, 2]),
        "whole_positive_minus_negative_by_edge": (pos - neg).tolist(),
        "whole_absolute_difference_ranking_edge_ids": arrays["edge_ids"][
            ranking
        ].tolist(),
        "event433_post_negative_count": int(
            force["condition_native_count"][list(windows).index("event433:post"), 4, 0]
        ),
        "event521_has_overlapping_sign_control": bool(
            force["native_count"][521, 4, 0] > 0
            and force["native_count"][521, 4, 2] > 0
        ),
        "candidate_adopted": False,
        "F6_exposed": False,
        "F1_F8_complete": False,
    }
    return summary, aligned


def finish_observation(
    collector: Collector,
    output: Path,
    restore: Any,
    threads: Any,
    monitor: Any,
    monitor_stop: threading.Event,
    monitor_errors: list[str],
    e182: Any,
) -> tuple[list[dict], dict]:
    """Save partial evidence and the monitor receipt without replacing a primary error."""
    errors: list[dict] = []

    def attempt(label: str, action: Any) -> None:
        try:
            action()
        except BaseException as error:
            errors.append({"artifact": label, "error": repr(error)})

    if restore is not None:
        attempt("observer_factory_restore", restore)
    if threads is not None:
        attempt("cpu_thread_restore", threads.restore_original_limits)
    signals = None
    try:
        signals = collector.arrays()
    except BaseException as error:
        errors.append({"artifact": "incoming_arrays", "error": repr(error)})
    if signals is not None:
        attempt(
            "incoming-signals.npz",
            lambda: np.savez_compressed(output / "incoming-signals.npz", **signals),
        )
        if any(row["artifact"] == "incoming-signals.npz" for row in errors):
            attempt(
                "incoming-signals-uncompressed.npz",
                lambda: np.savez(
                    output / "incoming-signals-uncompressed.npz", **signals
                ),
            )
    attempt(
        "runtime-provenance.json",
        lambda: write_new(
            output / "runtime-provenance.json",
            {
                "schema": "flylab.m3.e188.runtime-provenance.v1",
                **collector.provenance,
                "observed_rows": len(collector.rows),
                "error": collector.error,
            },
        ),
    )
    monitor_stop.set()
    if monitor is not None:
        attempt("monitor_join", lambda: monitor.join(timeout=5))
        if monitor.is_alive():
            monitor_errors.append("monitor_thread_did_not_stop")
    resource_rows: list[dict] = []
    if monitor is not None:
        try:
            resource_rows = e182._resource_rows(output / "resource-monitor.jsonl")
        except BaseException as error:
            monitor_errors.append(repr(error))
    bad = []
    try:
        bad = [
            i for i, row in enumerate(resource_rows)
            if not resource_readiness.resource_clear(row)
        ]
    except BaseException as error:
        monitor_errors.append(repr(error))
    check = {
        "schema": "flylab.m3.e188.resource-monitor.v1",
        "samples": len(resource_rows),
        "bad_sample_indices": bad,
        "modes": [row.get("resource_mode") for row in resource_rows],
        "errors": monitor_errors,
        "thread_alive": bool(monitor and monitor.is_alive()),
        "resource_clear": bool(
            resource_rows
            and not bad
            and not monitor_errors
            and monitor is not None
            and not monitor.is_alive()
        ),
    }
    attempt(
        "resource-monitor-check.json",
        lambda: write_new(output / "resource-monitor-check.json", check),
    )
    attempt(
        "observer-errors.json",
        lambda: write_new(
            output / "observer-errors.json",
            {"errors": errors, "observer_error": collector.error},
        ),
    )
    return errors, check


def propagate_observation_failure(
    primary: BaseException | None, errors: list[dict]
) -> None:
    if primary is not None:
        raise primary.with_traceback(primary.__traceback__)
    require(not errors, "observer cleanup/persistence failed")


def persist_terminal(
    output: Path, terminal: dict, primary: BaseException | None
) -> None:
    """A failed terminal write must fail success, without hiding a primary error."""
    try:
        write_new(output / "terminal.json", terminal)
    except BaseException as error:
        if primary is None:
            raise
        print("E188 terminal persistence error: " + repr(error), file=sys.stderr)


def execute(
    freeze_path: Path,
    review_path: Path,
    output: Path,
    parent_validation_sha256: str,
) -> None:
    began = _ENTRY_STARTED
    metadata, _freeze, frozen_sha = gate(
        freeze_path, review_path, output, parent_validation_sha256
    )
    prepared_readiness = parent_readiness(parent_validation_sha256)
    readiness_accepted = time.perf_counter()
    require(
        shutil.disk_usage(output.parent).free >= FREE_BYTES,
        "20 GiB free space required",
    )
    output.mkdir(exist_ok=False)
    terminal: dict[str, Any] = {
        "schema": "flylab.m3.e188.mn-inputs.terminal.v1",
        "status": "failed",
        "original_runs": 0,
        "automatic_retries": 0,
        "freeze_sha256": frozen_sha,
        "source_review_sha256": sha256(review_path),
        "parent_validation_sha256": parent_validation_sha256,
    }
    collector = Collector(metadata)
    result = None
    monitor = None
    monitor_stop = threading.Event()
    monitor_errors: list[str] = []
    monitor_check: dict[str, Any] | None = None
    cleanup_errors: list[dict] = []
    execution_error: BaseException | None = None
    try:
        # E182 imports E181 inertly. Both enter only after our complete source gate.
        from tools import optimize_c_banc_m3_native_timing as e181
        from tools import recover_c_banc_m3_native_timing as e182

        readiness = prepared_readiness
        permitted = bool(
            len(readiness) == 2 and all(row.get("ready") is True for row in readiness)
        )
        write_new(
            output / "readiness.json",
            {
                "schema": "flylab.m3.e188.readiness.v1",
                "freeze_sha256": frozen_sha,
                "observations": readiness,
                "native_permitted": permitted,
                "source": "same live parent validation receipt",
                "parent_validation_sha256": parent_validation_sha256,
                "accepted_monotonic_s": readiness_accepted,
            },
        )
        require(permitted, "E182 two-observation readiness blocked")
        old = metadata["e181_gate"]
        verified = e181.verified_inputs(
            safe(old["freeze_path"]),
            old["freeze_sha256"],
            safe(old["source_review_path"]),
            old["source_review_sha256"],
            mode="baseline",
        )
        require(verified.mode == "baseline", "actual E181 baseline gate")
        monitor = threading.Thread(
            target=resource_readiness.monitor,
            args=(output / "resource-monitor.jsonl", monitor_stop, monitor_errors, e182),
            daemon=True,
        )
        monitor.start()
        threads = None
        restore = None
        primary = None
        try:
            threads, _ = e181.protocol.joint_fit.one_cpu_thread()
            startup = e181._load_startup(verified.freeze)
            graph = e181.GraphStore.load(safe(verified.freeze["graph_path"]))
            require(graph.hash == verified.freeze["graph_hash"], "graph identity")
            neutral = e181.StateStore.load(safe(verified.freeze["neutral_checkpoint"]))
            restore = install_factory(e181, collector)
            terminal["original_runs"] = 1
            result = e181.run_arm(
                graph,
                neutral,
                (0.0,) * 6,
                output / "baseline",
                startup=startup,
                limits=e181.LIMITS,
                wall_limit_s=100,
                gate=verified,
                process_began=began,
            )
        except BaseException as error:
            primary = error
        finally:
            cleanup_errors, monitor_check = finish_observation(
                collector,
                output,
                restore,
                threads,
                monitor,
                monitor_stop,
                monitor_errors,
                e182,
            )
        propagate_observation_failure(primary, cleanup_errors)
        require(
            monitor_check["resource_clear"], "E182 resource monitor overlap/unknown"
        )
        require(time.perf_counter() - began <= INTERNAL_S, "internal 140s limit")
        arrays = collector.arrays()
        validate_run(result, arrays)
        comparison = compare_baseline(
            output / "baseline", safe(metadata["baseline_directory"])
        )
        validity = {
            "schema": "flylab.m3.e188.mn-inputs.validity.v1",
            "valid_3s_diagnostic": True,
            "baseline_exact": comparison,
            "incoming_rows": len(collector.rows),
            "e181_result": result,
        }
        write_new(output / "validity.json", validity)
        analysis, aligned = analyze(metadata, arrays)
        write_new(output / "analysis.json", analysis)
        np.savez_compressed(output / "aligned.npz", **aligned)
        require(time.perf_counter() - began <= INTERNAL_S, "internal 140s limit")
        base_bytes = sum(
            p.stat().st_size for p in (output / "baseline").rglob("*") if p.is_file()
        )
        output_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        require(
            base_bytes <= BASELINE_BYTES and output_bytes <= OUTPUT_BYTES,
            "output size limit",
        )
        terminal.update(
            status="complete",
            baseline_bytes=base_bytes,
            output_bytes=output_bytes,
            incoming_rows=ROWS,
        )
    except BaseException as error:
        execution_error = error
        terminal["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        final_cleanup_error: BaseException | None = None
        try:
            if monitor is not None and monitor.is_alive():
                monitor_stop.set()
                monitor.join(timeout=5)
                require(not monitor.is_alive(), "monitor thread remains live")
        except BaseException as error:
            final_cleanup_error = error
            cleanup_errors.append(
                {"artifact": "final_monitor_cleanup", "error": repr(error)}
            )
            terminal["status"] = "failed"
        terminal["elapsed_wall_s"] = time.perf_counter() - began
        terminal["e181_result_validity"] = result.get("validity") if result else None
        terminal["resource_monitor"] = monitor_check
        terminal["cleanup_persistence_errors"] = cleanup_errors
        persist_terminal(output, terminal, execution_error or final_cleanup_error)
        if execution_error is None and final_cleanup_error is not None:
            raise final_cleanup_error


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
