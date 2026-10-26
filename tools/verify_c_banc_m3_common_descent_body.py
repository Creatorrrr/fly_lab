"""E344 fixed candidate body diagnostic; original execution is separately gated."""

# ruff: noqa: BLE001 -- preserve first, partial and secondary lifecycle errors.
import time

ENTRY_S0 = time.perf_counter()
import argparse
import copy
import hashlib
import inspect
import io
import json
import math
import os
import stat
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

for _thread_key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_key] = "1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import verify_c_banc_m3_common_descent_direction as library

e = library.e
BASE = e.BASE
OWNED = (
    "tools/verify_c_banc_m3_common_descent_body.py",
    "tests/test_c_banc_common_descent_body.py",
    "tools/banc_m3_body_supervision.py",
)
REGISTRATION = BASE + "/parent-e346-reflected-design-registration.json"
REGISTRATION_SHA = "891c6219616087278597e4379cf7ad833972eb5543f170989feb23d610e55653"
INVENTORY = BASE + "/parent-e345-complete-input-inventory.json"
PREDECESSOR = {
    "path": BASE + "/parent-e344-reflected-input-inventory.json",
    "sha256": "702cbde1ebf5564ccc7312d8a54e4560eb3f462c9fd2a89dd60cc9e74c88742c",
}
DEPENDENCIES = BASE + "/parent-e344-native-dependency-descriptor-v2.json"
INPUT_LINKS = BASE + "/parent-e344-input-metadata-links.json"
INPUT_LINKS_SHA = "ac1bbdbf962f1f9c9ba74f02aee5e71cf59630e1a2d03634ccd819916f94777e"
FREEZE = BASE + "/e346-common-descent-body-freeze.json"
REVIEW = BASE + "/parent-e346-original-source-review.json"
CONTRACT = BASE + "/parent-e346-original-contract.json"
OUTPUT = BASE + "/e346-common-descent-body"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
NATIVE_LANDMARKS = (
    "coxa",
    "trochanterfemur",
    "tibia",
    "tarsus1",
    "tarsus2",
    "tarsus3",
    "tarsus4",
    "tarsus5",
)
LIMITS = {
    "science_s": 60,
    "internal_s": 72,
    "final_s": 8,
    "supervisor_s": 80,
    "audit_s": 12,
    "audit_internal_s": 10,
    "host_bytes": 6 * e.GIB,
    "sampled_added_GPU_bytes": 8 * e.GIB,
    "stage_bytes": 32 * e.MIB,
    "raw_bytes": 16 * e.MIB,
    "failure_bytes": 4 * e.MIB,
    "retry_count": 0,
}
QUOTAS = {
    "control-raw.npz": 8 * e.MIB,
    "native-contact.npz": 8 * e.MIB,
    "trace.json": 2 * e.MIB,
    "selected-ledger.json": 512 * 1024,
    "transfer-chunks.json": 512 * 1024,
    "result.json": 256 * 1024,
    "receipts.json": 256 * 1024,
    "stdout": 256 * 1024,
    "manifest.json": 16 * 1024,
    "sampler-completion.json": 64 * 1024,
    "revocation.json": 4 * e.MIB,
}
require = e.require
GPUMemoryWatch = library.GPUMemoryWatch
ProcessIdentityWatch = library.ProcessIdentityWatch
QUALIFICATION_BRANCHES = frozenset(("cpu_smoke", "native_smoke", "root_audit"))


def source_map(root=ROOT):
    return {name: e.digest(e.safe(root, name)) for name in OWNED}


def generated_path(root, name):
    path = e.safe(root, name)
    relative = path.relative_to(e.safe(root, BASE)).parts
    require(
        relative
        and relative[0].startswith(
            (
                "e344-sol-",
                "parent-e344-",
                "e345-sol-",
                "parent-e345-",
                "e346-sol-",
                "parent-e346-",
            )
        ),
        "owned E344 generated namespace",
    )
    return path


def preflight_inventory_metadata(inventory, predecessor):
    """Pure registered metadata check; never opens a scientific payload."""
    roles = {"recurrent_mask", "reference", "audited_candidate", "startup"}
    old_files = predecessor["files"]
    selected = inventory["selected_archives"]
    files = inventory["files"]
    require(
        len(old_files) == 131 and set(selected) == roles,
        "fixed predecessor and selected roles",
    )
    require(
        selected == predecessor["selected_archives"],
        "unchanged selected member schemas/bindings",
    )
    require(
        all(
            inventory[key] == predecessor[key]
            for key in (
                "initial_whole",
                "graph_bundle",
                "graph_hash",
                "source_network_identity",
                "source_adapter_identity",
                "same_theta_minus_original_raw_sha256",
            )
        ),
        "unchanged scientific metadata roots",
    )
    paths = [selected[role]["path"] for role in sorted(roles)]
    require(
        len(set(paths)) == 4 and not set(paths) & set(old_files),
        "distinct selected paths",
    )
    require(
        type(inventory["exact_distinct_file_count"]) is int
        and inventory["exact_distinct_file_count"] == len(files) == 135
        and set(files) == set(old_files) | set(paths),
        "exact whole124 graph7 selected4 union",
    )
    require(
        all(files[path] == entry for path, entry in old_files.items()),
        "unchanged original131 descriptors",
    )
    for role, spec in selected.items():
        entry = files[spec["path"]]
        require(
            entry["sha256"] == spec["sha256"]
            and type(entry["bytes"]) is int
            and entry["bytes"] == spec["bytes"]
            and entry.get("kind") == "npz"
            and entry.get("selected_archive_role") == role,
            "selected whitelist role/hash/bytes",
        )
    refs = inventory["initial_whole_array_references"]
    require(
        refs == predecessor["initial_whole_array_references"]
        and len(refs) == 122
        and len({v["state_leaf"] for v in refs}) == 122
        and len({v["path"] for v in refs}) == 122
        and {v["path"] for v in refs}
        == {
            p
            for p in old_files
            if p.endswith(".npy")
            and old_files[p].get("role") == "original_initial_whole"
        }
        and type(inventory["whole_NPY_count"]) is int
        and inventory["whole_NPY_count"] == 122,
        "unchanged whole122 reference coverage",
    )
    for path, entry in files.items():
        require(
            isinstance(path, str)
            and "\\" not in path
            and not Path(path).is_absolute()
            and ".." not in Path(path).parts
            and "\x00" not in path,
            "canonical inventory paths",
        )
        require(
            type(entry["bytes"]) is int and entry["bytes"] > 0, "typed inventory bytes"
        )
        e.check_sha(entry["sha256"])
    require(
        len({str(Path(path)).casefold() for path in files}) == 135, "no path aliases"
    )
    return {
        "files": 135,
        "selected_roles": sorted(roles),
        "whole_references": 122,
        "scientific_payload_reads": 0,
    }


def validate_registered_inventory(root=ROOT, *, timely=lambda: None):
    resolve_original_api()
    timely()
    registration = pinned_json(
        root, {"path": REGISTRATION, "sha256": REGISTRATION_SHA}, timely=timely
    )
    require(
        registration["inventory"]["path"] == INVENTORY, "current inventory registration"
    )
    inventory = pinned_json(root, registration["inventory"], timely=timely)
    predecessor = pinned_json(root, PREDECESSOR, timely=timely)
    preflight_inventory_metadata(inventory, predecessor)
    return inventory


def resolve_original_api():
    """Resolve actual public APIs and bind inert arguments without calling them."""
    from flylab.c.graph import GraphStore
    from flylab.c.storage import StateStore
    from tools import banc_m3_ssp_efficacy_trial as efficacy
    from tools import banc_m3_ssp_recovery_trial as recovery
    from tools import verify_c_banc_m3_e113_body as native

    inert = object()
    for function, args, keywords in (
        (GraphStore.load, (inert,), {}),
        (StateStore.load, (inert,), {"max_files": 123}),
        (recovery.checked_checkpoint, (inert,), {}),
        (recovery.SSPRecoveryResourceTrial.from_checkpoint, (inert, inert), {}),
        (efficacy.SSPEfficacyTrial, (inert, inert, inert, inert), {}),
        (native.install_native_observer, (inert, inert), {}),
    ):
        require(callable(function), "actual original API callable")
        inspect.signature(function).bind(*args, **keywords)
    return {
        "GraphStore": GraphStore,
        "StateStore": StateStore,
        "recovery": recovery,
        "efficacy": efficacy,
        "native": native,
    }


def resolve_source(root, name, digest, dependencies, *, timely=lambda: None):
    """Verify project bytes or one exactly registered absolute native source."""
    require(isinstance(name, str) and name and "\x00" not in name, "source path")
    e.check_sha(digest)
    path = Path(name.replace("\\", "/"))
    require(".." not in path.parts, "source traversal")
    root_path = Path(root).resolve()
    target = path if path.is_absolute() else root_path / path
    external = not target.resolve().is_relative_to(root_path)
    if external:
        require(path.is_absolute(), "absolute registered native source")
        key = path.as_posix()
        entry = dependencies["native_library_sources"].get(key)
        require(
            isinstance(entry, dict)
            and entry.get("sha256") == digest
            and type(entry.get("bytes")) is int
            and entry["bytes"] >= 0,
            "exact registered native source descriptor",
        )
    else:
        key = target.resolve().relative_to(root_path).as_posix()
        entry = None
    # Check the lexical path before resolving it: an ancestor link/junction must
    # not turn a registered name into a different execution source.
    for component in (target, *target.parents):
        require(
            not component.is_symlink()
            and not (
                getattr(component.lstat(), "st_file_attributes", 0)
                & stat.FILE_ATTRIBUTE_REPARSE_POINT
            ),
            "source path without links or reparse points",
        )
    require(target.is_file(), "source regular file")
    if entry is not None:
        require(target.stat().st_size == entry["bytes"], "native source byte count")
    require(e.digest(target, timely) == digest, "exact source bytes")
    return target, key


def json_bytes(value):
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf8")


def array_sha(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def same(actual, expected, label):
    require(type(actual) is type(expected), label + " type")
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), label + " keys")
        for key, value in expected.items():
            same(actual[key], value, label + "." + key)
    elif isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), label + " length")
        for a, b in zip(actual, expected, strict=True):
            same(a, b, label)
    elif hasattr(expected, "dtype"):
        require(
            actual.dtype == expected.dtype
            and actual.shape == expected.shape
            and actual.tobytes() == expected.tobytes(),
            label + " bytes",
        )
    else:
        require(actual == expected, label + " value")


def typed(value, shape, dtype, label, *, finite=True):
    import numpy as np

    require(
        isinstance(value, np.ndarray)
        and value.dtype == np.dtype(dtype)
        and value.shape == shape,
        label + " dtype/shape",
    )
    require(not finite or np.isfinite(value).all(), label + " finite")
    return value


def startup99(command, tendon, target_command, target_tendon):
    import numpy as np

    typed(command, (100, 42), "float64", "startup100 boundary rows")
    typed(tendon, (100, 6), "float64", "startup100 tendon")
    typed(target_command, (99, 42), "float64", "target99")
    typed(target_tendon, (99, 6), "float64", "target99 tendon")
    with np.errstate(over="raise", invalid="raise"):
        dq, dt = command[1:100] - target_command, tendon[1:100] - target_tendon
        result = {
            "q_rms": float(np.linalg.norm(dq) / math.sqrt(dq.size)),
            "q_max": float(np.abs(dq).max()),
            "tendon_rms": float(np.linalg.norm(dt) / math.sqrt(dt.size)),
            "tendon_max": float(np.abs(dt).max()),
        }
    require(all(math.isfinite(v) for v in result.values()), "finite startup99 errors")
    result["passed"] = all(
        result[k] <= v
        for k, v in (
            ("q_rms", 0.018),
            ("q_max", 0.04),
            ("tendon_rms", 0.009),
            ("tendon_max", 0.025),
        )
    )
    result["controls"] = 99
    return result


def validate_native(raw):
    import numpy as np

    m = len(raw["time_s"])
    for key, shape, dtype, finite in (
        ("time_s", (m,), "float64", True),
        ("normal_load_bw", (m, 6), "float64", True),
        ("slip_mm_s", (m, 6), "float64", False),
        ("slip_valid", (m, 6), "bool", True),
        ("nonfoot_contact", (m,), "bool", True),
        ("upright", (m,), "float64", True),
    ):
        typed(raw[key], shape, dtype, key, finite=finite)
    require(
        np.all(raw["normal_load_bw"] >= 0)
        and np.allclose(raw["time_s"], np.arange(1, m + 1) * 0.0001, rtol=0, atol=2e-8),
        "native prefix clock/load",
    )
    return m


def supported_slip(raw):
    import numpy as np

    validate_native(raw)
    support = ((raw["time_s"] >= 0.5) & (raw["time_s"] < 1.5))[:, None] & (
        raw["normal_load_bw"] >= 0.02
    )
    if not support.any():
        return {"available": False, "value": None, "reason": "no_supported_samples"}
    if (
        not np.all(raw["slip_valid"][support])
        or not np.isfinite(raw["slip_mm_s"][support]).all()
    ):
        return {"available": False, "value": None, "reason": "invalid_supported_slip"}
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        value = float(
            np.sum(raw["normal_load_bw"][support] * raw["slip_mm_s"][support])
            / np.sum(raw["normal_load_bw"][support])
        )
    require(math.isfinite(value), "finite supported slip")
    return {"available": True, "value": value, "reason": None}


def lean_row(row, *, point_count=None):
    import numpy as np

    body = row["body"]
    typed(np.asarray(body["position"]), (3,), "float64", "measured position")
    typed(np.asarray(body["basis"]), (3, 3), "float64", "measured basis")
    points = np.asarray([body["legs"][leg] for leg in LEGS])
    if point_count is None:
        require(points.shape in ((6, 5, 3), (6, 8, 3)), "explicit measured landmarks")
        point_count = points.shape[1]
    typed(
        points,
        (6, point_count, 3),
        "float64",
        "measured landmarks",
    )
    require(
        type(row["control_tick"]) is int
        and math.isfinite(row["time_s"])
        and math.isfinite(body["yaw"]),
        "trace clock/yaw",
    )
    require(
        len(row["feet"]["load_bearing"]) == 6
        and all(type(v) is bool for v in row["feet"]["load_bearing"]),
        "measured support",
    )
    return copy.deepcopy(
        {
            "time_s": row["time_s"],
            "control_tick": row["control_tick"],
            "body": {k: body[k] for k in ("position", "basis", "yaw", "legs")},
            "feet": {"load_bearing": row["feet"]["load_bearing"]},
            "upright": row["upright"],
            "fault": row["fault"],
            "navigation": {
                "brake_request": (row.get("navigation") or {}).get(
                    "brake_request", False
                )
            },
        }
    )


def lean_trace(rows, metadata):
    landmarks = metadata["landmarks"]
    require(
        landmarks
        in (
            {"scope": "actual_native", "count": 8, "order": list(NATIVE_LANDMARKS)},
            {"scope": "synthetic", "count": 8, "order": list(NATIVE_LANDMARKS)},
            {
                "scope": "synthetic",
                "count": 5,
                "order": ["fixture" + str(i) for i in range(5)],
            },
        ),
        "fixed native or explicit synthetic landmark contract",
    )
    payload = {
        "samples": [lean_row(row, point_count=landmarks["count"]) for row in rows],
        "landmarks": copy.deepcopy(landmarks),
        **{
            k: copy.deepcopy(metadata[k])
            for k in (
                "body_hash",
                "model_hash",
                "adapter_hash",
                "graph_hash",
                "parameters",
            )
        },
    }
    require(len(json_bytes(payload)) <= QUOTAS["trace.json"], "actual lean trace quota")
    return payload


def verify_chunk_coverage(proof):
    require(proof.get("schema") == "flylab.m3.e344.transfer-chunks.v1", "chunk schema")
    for label, leaf in proof["leaves"].items():
        require(type(leaf["count"]) is int and leaf["count"] >= 0, label + " count")
        end = 0
        for chunk in leaf["chunks"]:
            require(
                type(chunk["start"]) is int
                and type(chunk["end"]) is int
                and chunk["start"] == end
                and chunk["end"] > end
                and chunk["element_count"] == chunk["end"] - end
                and chunk["dtype"] == leaf["dtype"]
                and 0 < chunk["bytes"] <= 4 * e.MIB,
                label + " exact coverage",
            )
            require(
                all(
                    isinstance(chunk[k], str) and len(chunk[k]) == 64
                    for k in ("source_sha256", "expected_sha256", "actual_sha256")
                ),
                "chunk digests",
            )
            require(
                chunk["expected_sha256"] == chunk["actual_sha256"],
                label + " actual chunk",
            )
            end = chunk["end"]
        require(end == leaf["count"], label + " missing tail")
    return True


def verify_transfer(
    network,
    source_csr,
    positions,
    groups,
    theta,
    selected_expected,
    *,
    expected_palette=None,
    timely=lambda: None,
):
    import numpy as np

    from tools.banc_m3_ssp_efficacy_trial import patch_csr

    typed(theta, (len(theta),), "float64", "actual candidate theta")
    require(
        np.array_equal(theta, theta.astype(np.float32).astype(np.float64)),
        "actual f32 candidate",
    )
    expected, _original_selected, actual_selected = patch_csr(
        source_csr, positions, groups, theta
    )
    require(
        e.exact(actual_selected, selected_expected), "E343 audited selected minus bytes"
    )
    actual = host_csr(network)
    proof = {
        "schema": "flylab.m3.e344.transfer-chunks.v1",
        "leaves": {},
        "candidate_network_identity": network.identity,
        "source_CSR_data_sha256": array_sha(source_csr.data),
        "candidate_CSR_data_sha256": array_sha(expected.data),
        "selected_sha256": array_sha(actual_selected),
        "collector_scope": "actual host and packed comparison; audit covers saved digests, not GPU reacquisition",
    }

    def check(label, source, want, getter, count=None):
        count = len(want) if count is None else count
        chunks = []
        for begin in range(0, count, max(1, 4 * e.MIB // want.dtype.itemsize)):
            timely()
            end = min(count, begin + 4 * e.MIB // want.dtype.itemsize)
            observed = getter(begin, end)
            typed(observed, (end - begin,), want.dtype, label)
            require(e.exact(observed, want[begin:end]), label + " actual bytes")
            chunks.append(
                {
                    "start": begin,
                    "end": end,
                    "element_count": end - begin,
                    "dtype": want.dtype.str,
                    "bytes": observed.nbytes,
                    "source_sha256": array_sha(source[begin:end]),
                    "expected_sha256": array_sha(want[begin:end]),
                    "actual_sha256": array_sha(observed),
                }
            )
        proof["leaves"][label] = {
            "dtype": want.dtype.str,
            "count": count,
            "chunks": chunks,
        }

    for key, source, want, value in (
        ("host.rowptr", source_csr.indptr, expected.indptr, actual.indptr),
        ("host.columns", source_csr.indices, expected.indices, actual.indices),
        ("host.weights", source_csr.data, expected.data, actual.data),
    ):
        check(key, source, want, lambda a, b, value=value: value[a:b].copy())
    require(
        network.device == "cuda" and network.cuda_implementation == "packed",
        "actual packed native",
    )
    network._assert_pointers()

    def host(tensor, a, b):
        return tensor[a:b].detach().cpu().numpy().copy()

    palette = host(network._weight_values, 0, len(network._weight_values))
    typed(palette, (len(palette),), "float32", "packed palette")
    require(
        0 < len(palette) <= 16384
        and (expected_palette is None or len(palette) == expected_palette),
        "native palette cap",
    )
    proof["palette"] = len(palette)
    # Palette order belongs to the actual native representation. Independently
    # prove its bit-set and every decoded edge; do not impose a host ordering.
    require(
        len(np.unique(palette.view(np.uint32))) == len(palette)
        and set(map(int, palette.view(np.uint32)))
        == set(map(int, expected.data.view(np.uint32))),
        "actual palette full bit coverage",
    )
    check(
        "packed.palette",
        palette,
        palette,
        lambda a, b: host(network._weight_values, a, b),
    )
    rowptr = expected.indptr.astype(np.int32)
    check("packed.rowptr", rowptr, rowptr, lambda a, b: host(network._rowptr, a, b))
    palette_lookup = {int(v): i for i, v in enumerate(palette.view(np.uint32))}
    packed_want = (
        expected.indices.astype(np.uint32)
        | (
            np.asarray(
                [palette_lookup[int(v)] for v in expected.data.view(np.uint32)],
                np.uint32,
            )
            << 18
        )
    ).view(np.int32)

    def columns(a, b):
        packed = host(network._columns, a, b)
        typed(packed, (b - a,), "int32", "packed columns")
        bits = packed.view(np.uint32)
        idx = bits >> 18
        require(
            np.all(idx < len(palette))
            and np.array_equal((bits & np.uint32(0x3FFFF)), expected.indices[a:b]),
            "actual packed topology/index",
        )
        require(
            e.exact(palette[idx], expected.data[a:b]), "actual packed f32 weight bits"
        )
        return packed

    check("packed.columns", packed_want, packed_want, columns)
    network._assert_pointers()
    proof["pointer_version_guards_passed"] = True
    verify_chunk_coverage(proof)
    return proof


def host_csr(network):
    from scipy.sparse import csr_matrix

    return csr_matrix(
        (
            network.weights.values().cpu().numpy().copy(),
            network.weights.col_indices().cpu().numpy().copy(),
            network.weights.crow_indices().cpu().numpy().copy(),
        ),
        shape=(network.n, network.n),
    )


class Clock:
    def __init__(self, clock=time.perf_counter, s0=None):
        self.clock, self.s0 = clock, clock() if s0 is None else s0
        self.final_deadline = None

    def check(self, stage="science"):
        limit = self.s0 + (60 if stage == "science" else 72)
        if self.final_deadline is not None:
            limit = min(self.s0 + 72, self.final_deadline)
        now = self.clock()
        require(
            type(now) in (int, float) and math.isfinite(now) and now >= self.s0,
            "finite monotone clock",
        )
        if now > limit:
            raise TimeoutError("absolute E344 " + stage + " deadline")

    def finalize(self):
        self.check("internal")
        self.final_deadline = min(self.s0 + 72, self.clock() + 8)


@dataclass(frozen=True)
class GeneratedDimensions:
    controls: int = 1

    def __post_init__(self):
        require(
            type(self.controls) is int and 1 <= self.controls <= 300,
            "explicit generated controls",
        )


class Store:
    def __init__(self, output, clock):
        self.output, self.clock = Path(output), clock
        self.output.mkdir(parents=True, exist_ok=False)

    def check(self, extra=0, *, failure=False):
        total = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())
        require(
            total + extra + (0 if failure else 4 * e.MIB) <= 32 * e.MIB,
            "actual recursive staging quota",
        )
        self.clock.check("internal")

    def put(self, name, data, *, failure=False):
        require(name in QUOTAS and len(data) <= QUOTAS[name], "per-file write quota")
        self.check(len(data), failure=failure)
        path = self.output / name
        with path.open("xb") as stream:
            stream.write(data)
        require(path.read_bytes() == data, "actual write readback")
        self.check(failure=failure)

    def npz(self, name, arrays):
        import numpy as np

        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        self.put(name, buffer.getvalue())


def new_counters():
    return {
        "controls_requested": 0,
        "controls_host_returned": 0,
        "controls_completed": 0,
        "neural_requested": 0,
        "neural_host_returned": 0,
        "neural_device_confirmed": 0,
        "neural_unreturned_unknown": False,
        "physics_requested": 0,
        "physics_host_returned": 0,
        "physics_device_confirmed": 0,
        "physics_unreturned_unknown": False,
        "observer_samples": 0,
        "completed_rows": 0,
    }


class LiveObserver:
    """Instance-only native-return counter inside the contact observer wrapper."""

    def __init__(
        self, session, counters, samples, *, timely, install_contact, synchronize
    ):
        self.session, self.c, self.samples = session, counters, samples
        self.timely, self.install_contact, self.synchronize = (
            timely,
            install_contact,
            synchronize,
        )
        self.undo = []

    def install(self):
        network, body = self.session.network, self.session.body
        original_neural, original_physics = network.advance_readout, body._step_physics

        def neural(drive, ids, steps):
            self.timely()
            self.c["neural_requested"] += steps
            try:
                answer = original_neural(drive, ids, steps)
            except Exception:
                self.c["neural_unreturned_unknown"] = True
                raise
            self.c["neural_host_returned"] += steps
            self.synchronize()
            self.c["neural_device_confirmed"] += steps
            return answer

        def physics():
            self.timely()
            self.c["physics_requested"] += 1
            try:
                answer = original_physics()
            except Exception:
                self.c["physics_unreturned_unknown"] = True
                raise
            self.c["physics_host_returned"] += 1
            return answer

        network.advance_readout = neural
        body._step_physics = physics
        self.undo = [
            lambda: setattr(network, "advance_readout", original_neural),
            lambda: setattr(body, "_step_physics", original_physics),
        ]
        self.contact_undo = self.install_contact(body, self.samples)
        return self

    def close(self):
        errors = []
        for callback in [getattr(self, "contact_undo", None), *reversed(self.undo)]:
            if callback:
                try:
                    callback()
                except Exception as error:
                    errors.append(str(error))
        self.undo = []
        require(not errors, "owned observer detach: " + repr(errors))


class SetupObserver:
    """Constructor-only Python trace; no class or library-global patching."""

    def __init__(self, ledger, timely):
        self.ledger, self.timely, self.frames = ledger, timely, {}

    def __enter__(self):
        self.previous = sys.gettrace()
        self.ledger.update(
            body_constructors_requested=0,
            body_constructors_returned=0,
            settling_physics_requested=0,
            settling_physics_host_returned=0,
            neural_setup_requested=0,
            neural_setup_host_returned=0,
            neural_capture_recorded=0,
            neural_capture_requested=0,
            unknown_work=False,
            device_confirmed=False,
        )

        def trace(frame, event, arg):
            name, module = frame.f_code.co_name, frame.f_globals.get("__name__", "")
            obj = frame.f_locals.get("self")
            relevant = (
                (
                    module == "flylab.body"
                    and type(obj).__name__ == "FlyGymBody"
                    and name in ("__init__", "_step_physics")
                )
                or name == "_interval"
                and module.startswith("flylab.c")
            )
            if event == "call" and relevant:
                self.timely()
                capture = False
                if name == "_interval":
                    import torch

                    capture = bool(torch.cuda.is_current_stream_capturing())
                    count = obj.interval_steps
                    key = "neural_capture" if capture else "neural_setup"
                    self.ledger[key + "_requested"] += count
                else:
                    key, count = (
                        ("settling_physics", 1)
                        if name == "_step_physics"
                        else ("body_constructors", 1)
                    )
                    self.ledger[key + "_requested"] += count
                self.frames[id(frame)] = (key, count, False)
                return trace
            if id(frame) in self.frames:
                if event == "exception":
                    key, count, _ = self.frames[id(frame)]
                    self.frames[id(frame)] = (key, count, True)
                elif event == "return":
                    key, count, failed = self.frames.pop(id(frame))
                    if failed:
                        self.ledger["unknown_work"] = True
                    else:
                        suffix = (
                            "_recorded"
                            if key == "neural_capture"
                            else "_returned"
                            if key == "body_constructors"
                            else "_host_returned"
                        )
                        self.ledger[key + suffix] += count
            return trace if relevant else None

        sys.settrace(trace)
        return self

    def __exit__(self, kind, value, traceback):
        sys.settrace(self.previous)
        self.ledger["unknown_work"] |= kind is not None or bool(self.frames)


def native_arrays(samples):
    import numpy as np

    return {
        "time_s": np.asarray([v[0] for v in samples], np.float64),
        "normal_load_bw": np.asarray([v[1] for v in samples], np.float64).reshape(
            -1, 6
        ),
        "slip_mm_s": np.asarray([v[2] for v in samples], np.float64).reshape(-1, 6),
        "slip_valid": np.asarray([v[3] for v in samples], bool).reshape(-1, 6),
        "nonfoot_contact": np.asarray([v[4] for v in samples], bool),
        "upright": np.asarray([v[5] for v in samples], np.float64),
    }


def evaluate(output, control, native, data, startup):
    import numpy as np
    import torch

    from tools import analyze_c_banc_biological_gait as gait
    from tools.banc_m3_phase_free_shape import PhaseFreeShapeObjective
    from tools.banc_m3_smooth_temporal import SmoothTemporalObjective

    require(
        len(control["command"]) == 301 and validate_native(native) == 15000,
        "full evaluation only",
    )
    q, tendon = (
        torch.as_tensor(control["command"][1:]),
        torch.as_tensor(control["tendon"][1:]),
    )
    reference = data["reference"]
    objective = SmoothTemporalObjective(
        reference["curves"],
        data["startup"]["target_command"],
        data["startup"]["target_tendon"],
        device="cpu",
        normalization=reference["normalization"],
    )
    for key, expected in reference.items():
        require(
            e.exact(objective.canonical_constants[key], expected),
            "actual approved smooth constants",
        )
    legacy = PhaseFreeShapeObjective(
        reference["curves"],
        data["startup"]["target_command"],
        data["startup"]["target_tendon"],
        device="cpu",
    )
    with torch.no_grad():
        smooth, shape = objective.evaluate(q, tendon), legacy.evaluate(q, tendon)
    rows = e.read_json(Path(output) / "trace.json")["samples"]
    positions = np.asarray([row["body"]["position"] for row in rows])
    forward = np.asarray(rows[0]["body"]["basis"])[[0, 2], 0].copy()
    require(np.linalg.norm(forward) > 0, "physical initial forward axis")
    forward /= np.linalg.norm(forward)
    displacement = positions[299, [0, 2]] - positions[100, [0, 2]]
    slip = supported_slip(native)
    analyzed, _, _ = gait.analyze(gait.load_model(Path(output) / "trace.json"))
    health = {
        "healthy": bool(
            np.all(native["upright"] >= 0.9)
            and not native["nonfoot_contact"].any()
            and all(not row["fault"] for row in rows)
        ),
        "minimum_upright": float(native["upright"].min()),
        "nonfoot_contact_count": int(native["nonfoot_contact"].sum()),
    }
    return {
        "smooth": {"training_loss": float(smooth["training_loss"])},
        "shape": {"shape_loss": float(shape["shape_loss"])},
        "summary": {
            "forward_mm": float(displacement @ forward),
            "slip_mm_s": slip["value"],
            "slip": slip,
            "longest_continuous_forward_bout_s": float(
                analyzed["window"]["longest_forward_bout_s"]
            ),
            "complete_cycles": {
                leg: int(analyzed["events"][leg]["complete_cycles"]) for leg in LEGS
            },
            "health": health,
        },
        "startup": startup,
    }


def classify(metrics, neutral, startup, *, engineering_valid):
    summary = metrics["summary"]
    gates = {
        "engineering_valid": engineering_valid is True,
        "startup_four": startup.get("passed") is True,
        "smooth_better_than_saved_neutral": metrics["smooth"]["training_loss"]
        <= neutral["smooth"]["training_loss"] - 1e-9,
        "legacy_shape_better_than_saved_neutral": metrics["shape"]["shape_loss"]
        < neutral["shape"]["shape_loss"],
        "forward": summary["forward_mm"] >= 1.6790085779712631,
        "slip": summary["slip_mm_s"] is not None
        and summary["slip_mm_s"] <= 1.8026436968249955,
        "bout": summary["longest_continuous_forward_bout_s"] >= 0.3,
        "all_six_eligible_cycles": all(
            summary["complete_cycles"].get(leg, 0) > 0 for leg in LEGS
        ),
        "native_health": summary["health"]["healthy"] is True,
    }
    return {key: bool(value) for key, value in gates.items()}, all(gates.values())


def control_arrays(rows, session, data):
    import numpy as np

    return {
        "control_tick": np.asarray([row["control_tick"] for row in rows], np.int64),
        "network_tick": np.asarray([row["network_tick"] for row in rows], np.int64),
        "physics_time_s": np.asarray(
            [row["physics_time_s"] for row in rows], np.float64
        ),
        "command": np.asarray([row["command"] for row in rows], np.float64).reshape(
            -1, 42
        ),
        "tendon": np.asarray([row["tendon"] for row in rows], np.float64).reshape(
            -1, 6
        ),
        "theta": data["theta"].copy(),
        "selected_weights": data["selected_weights"].copy(),
    }


def capture(session):
    import numpy as np

    from flylab.c.walking_trace import capture_control

    row = capture_control(session)
    lean_row(row, point_count=8)
    return row, {
        "control_tick": int(session.control_tick),
        "network_tick": int(session.network.tick),
        "physics_time_s": float(session.body.physics_time()),
        "command": np.asarray(row["joint_targets_rad"], np.float64),
        "tendon": np.asarray(
            [session.tendon_inputs[leg + "_tarsus"] for leg in LEGS], np.float64
        ),
    }


def trace_metadata(session):
    return {
        "landmarks": {
            "scope": "actual_native",
            "count": 8,
            "order": list(NATIVE_LANDMARKS),
        },
        "body_hash": session.body.model_hash,
        "model_hash": session.network.identity,
        "adapter_hash": session.adapter.identity,
        "graph_hash": session.graph.hash,
        "parameters": session.p.state_dict()
        if hasattr(session.p, "state_dict")
        else asdict(session.p),
    }


def lifecycle(
    factory,
    data,
    *,
    root=ROOT,
    output=OUTPUT,
    dimensions=None,
    clock=time.perf_counter,
    s0=None,
    watcher=None,
    meter=e.usage,
    printer=e.stdout_bytes,
    fault=None,
):
    if dimensions is None:
        require(
            type(factory) is OriginalFactory
            and Path(root).resolve() == ROOT
            and output == OUTPUT
            and clock is time.perf_counter
            and meter is e.usage
            and printer is e.stdout_bytes
            and fault is None
            and s0 is not None,
            "original API overrides forbidden",
        )
    else:
        require(
            isinstance(dimensions, GeneratedDimensions),
            "explicit generated factory scope",
        )
        generated_path(root, output)
    import numpy as np

    timer, counters, construction = Clock(clock, s0), new_counters(), {}
    result = {
        "schema": "flylab.m3.e344.result.v1",
        "run_valid": False,
        "effective_qualified": False,
        "external_completion_pending": True,
        "independent_audit_pending": True,
        "source_sha256": source_map(),
        "mode": "generated" if dimensions else "original",
        "classification": "not_tested",
        "startup": {"status": "not_tested"},
        "metrics": None,
        "gates": None,
        "counters": counters,
        "construction": construction,
        "transfer": None,
        "first_error": None,
        "secondary_errors": [],
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "biology_proven": False,
        "full_original_factory_qualified": False,
        "retry_count": 0,
        "neural_backward_calls": 0,
        "full_parameter_Jacobian_independently_proven": False,
    }
    session = observer = store = None
    samples, controls, trace_rows, ledger, metadata = (
        [],
        [],
        [],
        data.get("ledger", []),
        {},
    )
    stage, reason, fault_hook = (
        "output",
        "duration",
        (lambda _: None) if fault is None else fault,
    )

    def failure(error, at):
        entry = {
            "stage": at,
            "type": type(error).__name__,
            "message": str(error)[:1024],
        }
        if result["first_error"] is None:
            result["first_error"] = entry
        else:
            result["secondary_errors"].append(entry)
        result["run_valid"] = False
        result["effective_qualified"] = False

    def timely():
        timer.check()
        fault_hook("deadline")

    def resources(record=True):
        value = meter("cuda")
        require(
            type(value["tree_rss_bytes"]) is int
            and 0 < value["tree_rss_bytes"] <= 6 * e.GIB,
            "host6GiB",
        )
        if record:
            result["resources"] = value
        if watcher:
            value = watcher.snapshot()
            require(
                not value["sampler_errors"]
                and value["sample_count"] > 0
                and value["sampled_additional_peak_bytes"]
                + value["quantization_allowance_bytes"]
                <= 8 * e.GIB,
                "context-included sampled GPU cap",
            )
            if record:
                result["GPU_memory_watch"] = value

    try:
        store = Store(e.safe(root, output), timer)
        stage = "precontext"
        if watcher and watcher.snapshot()["sample_count"] == 0:
            watcher.start()
        stage = "construct"
        session = factory.construct(data, construction, timely)
        resources()
        stage = "initial-transfer"
        result["transfer"] = factory.initial(session, data, timely)
        store.put("transfer-chunks.json", json_bytes(result["transfer"]))
        metadata = factory.metadata(session)
        row, boundary = factory.capture(session)
        trace_rows.append(row)
        controls.append(boundary)
        counters["completed_rows"] = 1
        observer = LiveObserver(
            session,
            counters,
            samples,
            timely=timely,
            install_contact=factory.install_contact,
            synchronize=factory.synchronize,
        ).install()
        count = dimensions.controls if dimensions else 300
        for control in range(1, count + 1):
            stage = "control-" + str(control)
            timely()
            counters["controls_requested"] += 1
            try:
                session.advance(1, capture=False)
            except factory.health_stop:
                reason = "native_health_stop"
                break
            counters["controls_host_returned"] += 1
            factory.synchronize()
            counters["physics_device_confirmed"] = counters["physics_host_returned"]
            row, boundary = factory.capture(session)
            trace_rows.append(row)
            controls.append(boundary)
            counters["observer_samples"] = len(samples)
            require(
                session.control_tick == control
                and session.network.tick == 20 * control
                and len(samples) == 50 * control
                and abs(session.body.physics_time() - control * 0.005) <= 2e-8,
                "actual completed control clock",
            )
            counters["controls_completed"] += 1
            counters["completed_rows"] = len(controls)
            if control == 99:
                result["startup"] = startup99(
                    np.stack([v["command"] for v in controls]),
                    np.stack([v["tendon"] for v in controls]),
                    **data["startup"],
                )
                if not result["startup"]["passed"]:
                    reason = "startup_rejected"
                    break
            resources()
        result["reason"] = reason
        stage = "raw-save"
        fault_hook(stage)
        raw = control_arrays(controls, session, data)
        native = native_arrays(samples)
        store.npz("control-raw.npz", raw)
        store.npz("native-contact.npz", native)
        store.put("trace.json", json_bytes(lean_trace(trace_rows, metadata)))
        stage = "typed-counters"
        validate_native(native)
        require(
            counters["completed_rows"] == counters["controls_completed"] + 1
            and counters["physics_host_returned"] == len(samples)
            and counters["neural_host_returned"] == counters["neural_device_confirmed"]
            and not counters["neural_unreturned_unknown"]
            and not counters["physics_unreturned_unknown"],
            "typed returned counter closure",
        )
        if reason == "duration" and count == 300:
            stage = "evaluation"
            result["metrics"] = evaluate(
                store.output, raw, native, data, result["startup"]
            )
            result["gates"], passed = classify(
                result["metrics"],
                data["saved_neutral"],
                result["startup"],
                engineering_valid=True,
            )
            result["classification"] = (
                "eligible_for_10s_design" if passed else "scientific_rejected"
            )
        elif reason != "duration":
            result["classification"] = "scientific_rejected"
        stage = "posthash"
        for entry in ledger:
            require(
                e.digest(e.safe(root, entry["path"]), lambda: timer.check("internal"))
                == entry["expected_sha256"],
                "selected posthash",
            )
        store.put(
            "selected-ledger.json",
            json_bytes(
                {"schema": "flylab.m3.e344.selected-ledger.v1", "inputs": ledger}
            ),
        )
        result["run_valid"] = True
    except Exception as error:
        failure(error, stage)
    finally:
        counters["observer_samples"] = len(samples)
        if session:
            try:
                factory.synchronize()
                counters["physics_device_confirmed"] = counters["physics_host_returned"]
            except Exception as error:
                failure(error, "partial-device-confirmation")
        for label, callback in (
            ("observer-close", observer.close if observer else None),
            ("body-close", session.close if session else None),
        ):
            if callback:
                try:
                    callback()
                except Exception as error:
                    failure(error, label)
        if store and controls:
            for name, callback in (
                (
                    "control-raw.npz",
                    lambda: store.npz(
                        "control-raw.npz", control_arrays(controls, session, data)
                    ),
                ),
                (
                    "native-contact.npz",
                    lambda: store.npz("native-contact.npz", native_arrays(samples)),
                ),
                (
                    "trace.json",
                    lambda: store.put(
                        "trace.json", json_bytes(lean_trace(trace_rows, metadata))
                    ),
                ),
            ):
                if not (store.output / name).exists():
                    try:
                        callback()
                    except Exception as error:
                        failure(error, "partial-" + name)
    try:
        stage = "final"
        timer.finalize()
        resources()
        result["actual_import_sha256"] = actual_imports()
        result["elapsed_before_publication_s"] = clock() - timer.s0
        payload = json_bytes(result)
        store.put("result.json", payload)
        store.put("receipts.json", payload)
        manifest = {
            p.name: {"sha256": e.digest(p), "bytes": p.stat().st_size}
            for p in store.output.iterdir()
            if p.is_file()
        }
        store.put(
            "manifest.json",
            json_bytes(
                {"schema": "flylab.m3.e344.output-manifest.v1", "files": manifest}
            ),
        )
        for name, entry in manifest.items():
            require(
                e.digest(store.output / name) == entry["sha256"],
                "nonselfmanifest readback",
            )
        fault_hook("readback")
        require(len(payload) <= QUOTAS["stdout"], "actual print quota")
        printer(payload)
        fault_hook("postpublication")
        resources(False)
        if watcher:
            completion = watcher.close()
            require(
                not completion["sampler_errors"]
                and not completion["owned_thread_alive"],
                "sampler cleanup",
            )
            store.put("sampler-completion.json", json_bytes(completion), failure=True)
        timer.check("final")
        store.check()
    except Exception as error:
        failure(error, stage)
    finally:
        if watcher:
            try:
                completion = watcher.close()
                require(
                    not completion["sampler_errors"]
                    and not completion["owned_thread_alive"],
                    "own sampler closed",
                )
            except Exception as error:
                failure(error, "sampler-close")
    if result["first_error"]:
        result["classification"] = "technical_invalid"
        result["revoked"] = True
        try:
            # A late deadline cannot erase the retained provisional bytes.
            payload = json_bytes(result)
            require(len(payload) <= QUOTAS["revocation.json"], "failure4MiB")
            require(
                sum(p.stat().st_size for p in store.output.rglob("*") if p.is_file())
                + len(payload)
                <= 32 * e.MIB,
                "emergency output cap",
            )
            with (store.output / "revocation.json").open("xb") as stream:
                stream.write(payload)
            require(
                (store.output / "revocation.json").read_bytes() == payload,
                "revocation readback",
            )
        except Exception as error:
            result["secondary_errors"].append(
                {
                    "stage": "emergency-publication",
                    "type": type(error).__name__,
                    "message": str(error)[:1024],
                }
            )
    return result, 0 if result["run_valid"] else 1


def actual_imports():
    result = e.actual_imports()
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name and str(name).endswith(".py") and ("flygym" in str(name).lower()):
            path = Path(name).resolve()
            key = (
                path.relative_to(ROOT).as_posix()
                if path.is_relative_to(ROOT)
                else path.as_posix()
            )
            merge_exact(result, {key: e.digest(path)})
    return result


class OriginalFactory:
    def __init__(self):
        api = resolve_original_api()
        self.recovery, self.efficacy, self.native = (
            api["recovery"],
            api["efficacy"],
            api["native"],
        )
        self.health_stop, self.install_contact = (
            self.native.NativeHealthStop,
            self.native.install_native_observer,
        )

    def synchronize(self):
        import torch

        torch.cuda.synchronize()

    def construct(self, data, ledger, timely):
        old = candidate = None
        try:
            with SetupObserver(ledger, timely):
                old = self.recovery.SSPRecoveryResourceTrial.from_checkpoint(
                    data["graph"], copy.deepcopy(data["saved"])
                )
                data["source_csr"] = host_csr(old.network)
                require(
                    old.network.identity
                    == data["inventory"]["source_network_identity"],
                    "actual original source identity",
                )
                candidate = self.efficacy.SSPEfficacyTrial(
                    old, data["positions"], data["groups"], data["theta"]
                )
            self.synchronize()
            ledger["device_confirmed"] = True
            require(
                ledger["body_constructors_requested"]
                == ledger["body_constructors_returned"]
                == 1
                and ledger["settling_physics_requested"]
                == ledger["settling_physics_host_returned"]
                == 2000
                and ledger["neural_setup_requested"]
                == ledger["neural_setup_host_returned"]
                == 320
                and ledger["neural_capture_requested"]
                == ledger["neural_capture_recorded"]
                == 160
                and not ledger["unknown_work"],
                "actual original constructor work mismatch",
            )
            return candidate
        except Exception as error:
            if candidate is not None:
                self.recovery.cleanup_owned(candidate, error)
            elif old is not None and not old.closed:
                self.recovery.cleanup_owned(old, error)
            raise

    def initial(self, session, data, timely):
        observed = session.snapshot()
        inverse = self.efficacy._source_representation(
            observed, observed[self.efficacy.PATCH_KEY]
        )
        self.recovery.require_saved_whole_equal(
            inverse, data["saved"], "exact initial inverse whole including wall"
        )
        require(
            session.control_tick == session.network.tick == 0
            and abs(session.body.physics_time()) <= 2e-8
            and session.graph.hash == data["inventory"]["graph_hash"]
            and session.adapter.identity == data["inventory"]["source_adapter_identity"]
            and session.network.identity != data["inventory"]["source_network_identity"]
            and session.network.interval_steps == 20,
            "original source/candidate/decoder/control0 identities",
        )
        proof = verify_transfer(
            session.network,
            data["source_csr"],
            data["positions"],
            data["groups"],
            data["theta"],
            data["selected_weights"],
            expected_palette=15461,
            timely=timely,
        )
        require(
            proof["candidate_CSR_data_sha256"]
            == "b389d72a116e16455a8837615ad511e5d1febe829a9b70a5290b8f2fc548bf2f"
            and proof["selected_sha256"]
            == "b503f0ea9d3e386cff8422104b6e4550a97456906fb4f2d75eddee1fadf390e1",
            "E343 exact actual candidate",
        )
        require(
            e.exact(host_csr(session.network).data, data["source_csr"].data) is False,
            "candidate CSR distinct from source",
        )
        proof.update(
            source_network_identity=data["inventory"]["source_network_identity"],
            source_inverse_whole_exact=True,
        )
        import numpy as np

        _, boundary = capture(session)
        target_tendon = data["saved"]["adapter"]["original"]["original"]["original"][
            "tendon_activation"
        ]
        require(
            e.exact(
                boundary["command"],
                np.asarray(data["saved"]["body"]["action"]["joint_angles"], np.float64),
            )
            and e.exact(boundary["tendon"], target_tendon),
            "actual source boundary command/tendon bytes",
        )
        proof["source_boundary_command_and_tendon_exact"] = True
        return proof

    capture = staticmethod(capture)
    metadata = staticmethod(trace_metadata)


def whole_metadata(root, inventory):
    base = inventory["initial_whole"]
    manifest = e.read_json(e.safe(root, base + "/manifest.json"))
    state = e.read_json(e.safe(root, base + "/state.json"))
    leaves = {}

    def walk(value, path=""):
        if isinstance(value, dict):
            if "__npy__" in value:
                require(
                    set(value) == {"__npy__"} and isinstance(value["__npy__"], str),
                    "typed current state reference",
                )
                leaves[path] = base + "/" + value["__npy__"]
            else:
                for key, leaf in value.items():
                    walk(leaf, path + "." + key if path else key)
        elif isinstance(value, list):
            for index, leaf in enumerate(value):
                walk(leaf, path + "." + str(index))

    walk(state)
    approved = {
        v["state_leaf"]: v["path"] for v in inventory["initial_whole_array_references"]
    }
    require(
        leaves == approved and len(leaves) == 122 and len(set(leaves.values())) == 122,
        "exact semantic122 restore leaves",
    )
    expected = {Path(v).name: inventory["files"][v]["sha256"] for v in leaves.values()}
    expected["state.json"] = inventory["files"][base + "/state.json"]["sha256"]
    require(
        manifest["schema"] == "flylab.state_store.v3"
        and manifest["files"] == expected
        and manifest["versions"] == inventory["whole_runtime_versions"],
        "exact manifest123/version/whitelist",
    )
    require(
        {p.name for p in e.safe(root, base).iterdir() if p.is_file()}
        == {*expected, "manifest.json"},
        "whole additional file forbidden",
    )
    require(
        state["control_tick"] == state["network"]["tick"] == 0, "source whole control0"
    )
    return state


def load_original(freeze, *, timely, ledger=None):
    import numpy as np

    api = resolve_original_api()
    GraphStore, StateStore, recovery = (
        api["GraphStore"],
        api["StateStore"],
        api["recovery"],
    )

    inventory = freeze["inventory"]
    ledger = [] if ledger is None else ledger
    whole_metadata(ROOT, inventory)
    graph = GraphStore.load(e.safe(ROOT, inventory["graph_bundle"]))
    saved = StateStore.load(e.safe(ROOT, inventory["initial_whole"]), max_files=123)
    require(graph.hash == inventory["graph_hash"], "current original graph hash")
    recovery.checked_checkpoint(saved)
    selection = {
        role: e.load_selected(ROOT, spec, ledger, timely=timely, generated=False)
        for role, spec in inventory["selected_archives"].items()
    }
    theta = selection["audited_candidate"]["theta_minus"]
    require(
        np.all(theta > 0)
        and e.exact(theta, theta.astype(np.float32).astype(np.float64)),
        "same actual f32 theta",
    )
    return {
        "graph": graph,
        "saved": saved,
        "inventory": inventory,
        "ledger": ledger,
        "theta": theta,
        "positions": selection["recurrent_mask"]["positions"],
        "groups": selection["recurrent_mask"]["edge_parameters"],
        "selected_weights": selection["audited_candidate"]["selected_weights_minus"],
        "reference": selection["reference"],
        "startup": selection["startup"],
        "saved_neutral": freeze["saved_neutral"],
    }


def merge_exact(target, added):
    for key, value in added.items():
        require(
            key not in target or target[key] == value, "collision-reject pin " + key
        )
        target[key] = value


def pinned_json(root, descriptor, *, timely=lambda: None):
    path = e.safe(root, descriptor["path"])
    require(e.digest(path, timely) == descriptor["sha256"], "exact JSON descriptor")
    return e.read_json(path)


def input_metadata(root, inventory, links, *, timely=lambda: None):
    """Finite explicit JSON producer edges; never recursively load old payloads."""
    require(
        links.get("schema") == "flylab.m3.e344.current-finite-input-metadata-links.v1"
        and len(links["producer_metadata_pins"]) == 15,
        "current finite15 producer links",
    )
    loaded = {}
    for path, entry in links["producer_metadata_pins"].items():
        require(
            e.safe(root, path).stat().st_size == entry["bytes"], "producer JSON size"
        )
        loaded[path] = pinned_json(
            root, {"path": path, "sha256": entry["sha256"]}, timely=timely
        )
    final = loaded[BASE + "/parent-e343-final-decision.json"]
    require(
        final.get("run_valid") is True
        and final.get("independent_audit_passed") is True
        and final.get("handoff_eligible_for_same_candidate_body_design") is True
        and final.get("direct_S_VJP_numerical_byte_identity_proven") is False,
        "E343 accepted limited handoff",
    )
    manifest = loaded[BASE + "/e343-common-descent-direction/manifest.json"]
    selected = inventory["selected_archives"]["audited_candidate"]
    require(
        manifest["files"]["raw.npz"]
        == {"sha256": selected["sha256"], "bytes": selected["bytes"]},
        "E343 selected actual candidate producer",
    )
    earlier = loaded[BASE + "/parent-e335-final-decision.json"]
    neutral = {
        "smooth": {
            "training_loss": earlier["metrics"]["smooth"]["neutral"]["training_loss"]
        },
        "shape": {"shape_loss": earlier["metrics"]["gait"]["neutral"]["shape_loss"]},
    }
    require(
        neutral
        == {
            "smooth": {"training_loss": links["saved_neutral"]["smooth_training_loss"]},
            "shape": {"shape_loss": links["saved_neutral"]["shape_loss"]},
        },
        "typed saved neutral ancestry",
    )
    require(
        inventory["files"][links["boundary_command"]["path"]]["sha256"]
        == links["boundary_command"]["sha256_inherited"]
        and inventory["files"][links["boundary_tendon"]["path"]]["sha256"]
        == links["boundary_tendon"]["sha256"],
        "source boundary producer leaves",
    )
    return neutral


def validate_qualification(root, proof, current, *, timely=lambda: None):
    require(
        proof.get("schema") == "flylab.m3.e344.current-generated-proof.v1"
        and proof.get("passed") is True
        and proof.get("source_sha256") == current
        and proof.get("original_calls") == 0,
        "current E344 generated proof",
    )
    evidence = proof["evidence_sha256"]
    require(
        0 < len(evidence) <= 256 and set(proof["branches"]) == QUALIFICATION_BRANCHES,
        "finite current proof roots",
    )
    for branch in proof["branches"].values():
        report = closed_execution(root, branch, evidence, current, timely=timely)
        contract = e.read_json(e.safe(root, branch["contract"]["path"]))
        require(
            all(
                contract["source_sha256"].get(path) == digest
                for path, digest in report.get("actual_import_sha256", {}).items()
            ),
            "actual definition imports executed source pins",
        )
    native = e.read_json(
        e.safe(root, proof["branches"]["native_smoke"]["report"]["path"])
    )
    require(
        native.get("actual_native_SSP_neural20") is True
        and native.get("actual_FlyGym_physics50") is True
        and native.get("full_original_factory_qualified") is False
        and native.get("biology_proven") is False,
        "tiny native scope",
    )
    audit = proof["same_audit_cli"]
    cases = {
        "valid",
        "wrong_theta",
        "wrong_clock",
        "wrong_boundary",
        "wrong_cycle",
        "wrong_slip",
        "wrong_result",
    }
    require(set(audit["cases"]) == cases, "same audit seven actual CLI cases")
    for name, branch in audit["cases"].items():
        closed_execution(
            root,
            branch,
            evidence,
            current,
            expected_exit=0 if name == "valid" else 1,
            timely=timely,
        )
        contract = e.read_json(e.safe(root, branch["contract"]["path"]))
        require(
            contract["source_sha256"].get(audit["source"]["path"])
            == audit["source"]["sha256"],
            "same audit source",
        )
    return True


def closed_execution(
    root, branch, evidence, current, *, expected_exit=0, timely=lambda: None
):
    """E344 6GiB envelope, with registered external native PY only."""
    require(
        set(branch) == {"contract", "supervision", "report", "pids"},
        "closed execution branch",
    )
    loaded = {
        key: e.read_json(e.descriptor(root, desc, evidence, timely))
        for key, desc in branch.items()
    }
    contract, supervision, report, pids = (
        loaded[key] for key in ("contract", "supervision", "report", "pids")
    )
    dependencies = e.read_json(e.safe(root, DEPENDENCIES))
    for name, digest in contract["source_sha256"].items():
        _, key = resolve_source(root, name, digest, dependencies, timely=timely)
        require(
            evidence.get(key) == digest,
            "executed finite source bytes",
        )
    require(
        contract["source_sha256"].get(e.SUPERVISOR) == e.SUPERVISOR_SHA
        and all(contract["source_sha256"].get(k) == v for k, v in current.items()),
        "actual supervisor/current source",
    )
    require(
        len(contract["commands"]) == len(supervision["attempts"]) == 1,
        "sole actual command",
    )
    command, attempt = contract["commands"][0], supervision["attempts"][0]
    argv = command["argv"]
    require(
        isinstance(argv, list)
        and all(isinstance(v, str) for v in argv)
        and len(argv) >= 7
        and argv[:4] == [".venv/Scripts/python.exe", "-X", "utf8", "-B"]
        and attempt["argv"] == argv,
        "actual Python argv",
    )
    helper = e.relative(root, argv[4])
    require(
        helper == e.relative(root, report["audit_source_path"])
        and contract["source_sha256"].get(helper) == report["audit_source_sha256"]
        and argv.count("--output") == 1
        and argv.index("--output") + 1 < len(argv)
        and e.relative(root, argv[argv.index("--output") + 1])
        == e.relative(root, branch["report"]["path"]),
        "actual helper report output",
    )
    require(
        supervision["contract_sha256"] == branch["contract"]["sha256"]
        and e.relative(root, contract["supervisor_output"])
        == e.relative(root, branch["supervision"]["path"]),
        "supervision contract linkage",
    )
    require(
        attempt.get("requested") is True
        and attempt.get("killed") is False
        and type(attempt.get("exit_code")) is int
        and attempt["exit_code"] == expected_exit
        and supervision.get("success") is (expected_exit == 0),
        "completed returned attempt",
    )
    require(
        e.positive_time(command.get("wall_s"))
        and e.positive_time(attempt.get("wall_s"))
        and e.positive_time(supervision.get("elapsed_s"))
        and attempt["wall_s"] <= supervision["elapsed_s"] <= command["wall_s"] <= 60,
        "actual finite wall envelope",
    )
    require(
        type(contract.get("max_tree_rss_bytes")) is int
        and 0 < contract["max_tree_rss_bytes"] <= 6 * e.GIB
        and type(supervision.get("max_tree_rss_bytes")) is int
        and 0 < supervision["max_tree_rss_bytes"] <= contract["max_tree_rss_bytes"],
        "actual positive RSS envelope",
    )
    ids = supervision.get("owned_pids")
    require(
        e.positive_pids(ids)
        and e.positive_pids(attempt.get("pids"))
        and e.positive_pids(pids.get("owned_pids"))
        and ids == attempt["pids"] == pids["owned_pids"],
        "exact owned positive unique PIDs",
    )
    presence, identities = (
        pids.get("present_after_return"),
        pids.get("pid_create_times"),
    )
    require(
        isinstance(presence, dict)
        and set(presence) == {str(v) for v in ids}
        and all(v is False for v in presence.values())
        and pids.get("all_owned_pids_ended") is True
        and isinstance(identities, dict)
        and set(identities) == set(presence)
        and all(
            type(v) in (int, float) and math.isfinite(v) and v > 0
            for v in identities.values()
        ),
        "exact absent own PID identities",
    )
    for label in ("stdout", "stderr"):
        key = e.relative(root, command[label])
        require(
            evidence.get(key) == e.digest(e.safe(root, key), timely),
            "closed actual logs",
        )
    require(
        all(
            type(value.get("original_calls")) is int and value["original_calls"] == 0
            for value in (contract, supervision, report)
        )
        and report.get("source_sha256") == current,
        "current generated source/counter",
    )
    checks = report.get("checks")
    require(
        isinstance(checks, dict)
        and checks
        and all(type(v) is bool for v in checks.values()),
        "typed actual report checks",
    )
    if expected_exit == 0:
        require(
            report.get("passed") is True
            and report.get("first_error") is None
            and all(checks.values()),
            "actual successful report",
        )
    else:
        require(
            report.get("passed") is False
            and isinstance(report.get("first_error"), dict)
            and not all(checks.values()),
            "actual fault rejection",
        )
    return report


def make_freeze(*, qualification, reservation):
    require(
        e.digest(e.safe(ROOT, REGISTRATION)) == REGISTRATION_SHA,
        "immutable reflected registration",
    )
    registration = e.read_json(e.safe(ROOT, REGISTRATION))
    inventory = pinned_json(ROOT, registration["inventory"])
    require(
        inventory == validate_registered_inventory(ROOT), "qualified metadata preflight"
    )
    dependencies = pinned_json(ROOT, registration["dependencies"])
    proof = pinned_json(ROOT, qualification)
    validate_qualification(ROOT, proof, source_map())
    reserve = pinned_json(ROOT, reservation)
    require(
        reserve.get("schema") == "flylab.m3.e346.root-reservation.v1"
        and reserve.get("approved") is True
        and reserve.get("source_sha256") == source_map()
        and reserve.get("original_limit_s") == 9900
        and reserve.get("original_cumulative_s") == 9700.57798869951
        and reserve.get("reserved_s") == 92
        and reserve.get("body_s") == 80
        and reserve.get("audit_s") == 12
        and reserve.get("retry_count") == 0,
        "ROOT prospective reservation after qualification",
    )
    metadata = {
        REGISTRATION: REGISTRATION_SHA,
        PREDECESSOR["path"]: PREDECESSOR["sha256"],
    }
    for descriptor in (
        registration["design"],
        registration["inventory"],
        registration["dependencies"],
        registration["actual_Astra_review"],
        registration["actual_Astra_evidence"],
        qualification,
        reservation,
    ):
        merge_exact(metadata, {descriptor["path"]: descriptor["sha256"]})
    for descriptor in registration.get("runtime_amendment_pins", []):
        pinned_json(ROOT, descriptor) if descriptor["path"].endswith(
            ".json"
        ) else require(
            e.digest(e.safe(ROOT, descriptor["path"])) == descriptor["sha256"],
            "reflected runtime amendment source",
        )
        merge_exact(metadata, {descriptor["path"]: descriptor["sha256"]})
    # These producer JSON roots are explicitly registered by the reflected review.
    review = pinned_json(ROOT, registration["actual_Astra_evidence"])
    authoritative = {
        v["path"]: v["sha256"] for v in review["text_files_read_and_SHA256"]
    }
    for path in (
        BASE + "/parent-e335-final-decision.json",
        BASE + "/parent-e343-final-decision.json",
        BASE + "/e318-full-drive-capture-freeze.json",
    ):
        require(path in authoritative, "registered producer root")
        merge_exact(metadata, {path: authoritative[path]})
    links = pinned_json(ROOT, {"path": INPUT_LINKS, "sha256": INPUT_LINKS_SHA})
    merge_exact(metadata, {INPUT_LINKS: INPUT_LINKS_SHA})
    merge_exact(
        metadata,
        {
            path: item["sha256"]
            for path, item in links["producer_metadata_pins"].items()
        },
    )
    neutral = input_metadata(ROOT, inventory, links)
    observed_sources = {}
    for branch in proof["branches"].values():
        report = pinned_json(ROOT, branch["report"])
        merge_exact(observed_sources, report.get("actual_import_sha256", {}))
    return {
        "schema": "flylab.m3.e344.freeze.v1",
        "source_sha256": source_map(),
        "observed_source_sha256": observed_sources,
        "inventory": inventory,
        "dependencies": dependencies,
        "metadata_sha256": metadata,
        "qualification": qualification,
        "reservation": reservation,
        "saved_neutral": neutral,
        "limits": LIMITS,
        "output": OUTPUT,
    }


def gate(freeze_sha256, review_sha256, *, timely=lambda: None):
    freeze = pinned_json(ROOT, {"path": FREEZE, "sha256": freeze_sha256}, timely=timely)
    review = pinned_json(ROOT, {"path": REVIEW, "sha256": review_sha256}, timely=timely)
    require(
        freeze.get("schema") == "flylab.m3.e344.freeze.v1"
        and freeze["source_sha256"] == source_map()
        and freeze["limits"] == LIMITS
        and freeze["output"] == OUTPUT,
        "fixed original E344 freeze",
    )
    require(
        review.get("approved") is True
        and review.get("blocking_defects") == []
        and review.get("source_sha256") == source_map()
        and review.get("freeze_sha256") == freeze_sha256,
        "current ROOT approved source",
    )
    for path, digest in freeze["metadata_sha256"].items():
        require(
            e.digest(e.safe(ROOT, path), timely) == digest,
            "producer/current metadata bytes",
        )
    reg = e.read_json(e.safe(ROOT, REGISTRATION))
    require(
        freeze["metadata_sha256"].get(REGISTRATION) == REGISTRATION_SHA
        and all(
            freeze["metadata_sha256"].get(reg[key]["path"]) == reg[key]["sha256"]
            for key in (
                "design",
                "inventory",
                "dependencies",
                "actual_Astra_review",
                "actual_Astra_evidence",
            )
        ),
        "exact reflected registration roots",
    )
    inv = pinned_json(ROOT, reg["inventory"], timely=timely)
    require(
        inv == validate_registered_inventory(ROOT, timely=timely),
        "original metadata preflight",
    )
    dep = pinned_json(ROOT, reg["dependencies"], timely=timely)
    require(
        inv == freeze["inventory"]
        and dep == freeze["dependencies"]
        and inv["exact_distinct_file_count"] == len(inv["files"]) == 135
        and inv["whole_NPY_count"] == 122,
        "exact typed scientific whitelist",
    )
    require(
        freeze["metadata_sha256"].get(INPUT_LINKS) == INPUT_LINKS_SHA
        and input_metadata(
            ROOT,
            inv,
            pinned_json(
                ROOT, {"path": INPUT_LINKS, "sha256": INPUT_LINKS_SHA}, timely=timely
            ),
            timely=timely,
        )
        == freeze["saved_neutral"],
        "finite typed original ancestry",
    )
    validate_qualification(
        ROOT,
        pinned_json(ROOT, freeze["qualification"], timely=timely),
        source_map(),
        timely=timely,
    )
    reserve = pinned_json(ROOT, freeze["reservation"], timely=timely)
    require(
        reserve.get("approved") is True
        and reserve.get("source_sha256") == source_map()
        and reserve.get("reserved_s") == 92
        and reserve.get("original_cumulative_s") == 9700.57798869951
        and reserve.get("original_limit_s") == 9900
        and reserve.get("schema") == "flylab.m3.e346.root-reservation.v1"
        and reserve.get("body_s") == 80
        and reserve.get("audit_s") == 12
        and reserve.get("retry_count") == 0,
        "active ROOT reservation",
    )
    sources = dict(dep["project_source_sha256"])
    merge_exact(sources, source_map())
    observed_sources = {}
    qualified = pinned_json(ROOT, freeze["qualification"], timely=timely)
    for branch in qualified["branches"].values():
        merge_exact(
            observed_sources,
            pinned_json(ROOT, branch["report"], timely=timely).get(
                "actual_import_sha256", {}
            ),
        )
    require(
        observed_sources == freeze.get("observed_source_sha256"),
        "exact qualified observed definition closure",
    )
    merge_exact(sources, observed_sources)
    for name, digest in sources.items():
        resolve_source(ROOT, name, digest, dep, timely=timely)
    for name, entry in dep["native_library_sources"].items():
        resolve_source(ROOT, name, entry["sha256"], dep, timely=timely)
    for name, entry in dep["native_constructor_asset_bytes"].items():
        path = Path(name)
        require(
            path.is_absolute()
            and not path.is_symlink()
            and e.digest(path, timely) == entry["sha256"]
            and path.stat().st_size == entry["bytes"],
            "finite native dependency bytes",
        )
    contract = e.read_json(e.safe(ROOT, CONTRACT))
    argv = [
        ".venv/Scripts/python.exe",
        "-X",
        "utf8",
        "-B",
        OWNED[0],
        "--run",
        "--freeze-sha256",
        freeze_sha256,
        "--review-sha256",
        review_sha256,
    ]
    require(
        type(contract["max_tree_rss_bytes"]) is int
        and contract["max_tree_rss_bytes"] == 6 * e.GIB
        and len(contract["commands"]) == 1
        and contract["commands"][0]["argv"] == argv
        and contract["commands"][0]["wall_s"] == 80
        and contract.get("original_calls") == 1
        and all(contract["source_sha256"].get(k) == v for k, v in sources.items()),
        "exact current original supervisor argv/cap",
    )
    freeze["current_contract_sha256"] = e.digest(e.safe(ROOT, CONTRACT))
    whole_metadata(ROOT, inv)
    for name, entry in inv["files"].items():
        path = e.safe(ROOT, name)
        require(
            path.stat().st_size == entry["bytes"]
            and e.digest(path, timely) == entry["sha256"],
            "original prehash whitelist",
        )
    require(not e.safe(ROOT, OUTPUT).exists(), "fresh fixed original output")
    return freeze


def run_generated(factory, data, *, root, output, dimensions, **options):
    require(isinstance(dimensions, GeneratedDimensions), "explicit generated lifecycle")
    return lifecycle(
        factory, data, root=root, output=output, dimensions=dimensions, **options
    )


def run(freeze_sha256, review_sha256):
    timer = Clock(s0=ENTRY_S0)
    freeze = gate(freeze_sha256, review_sha256, timely=timer.check)
    watcher = GPUMemoryWatch()
    ledger = []
    try:
        watcher.start()  # Before Torch context, factory or numeric packet imports.
        data = load_original(freeze, timely=timer.check, ledger=ledger)
        return lifecycle(OriginalFactory(), data, s0=ENTRY_S0, watcher=watcher)
    except Exception as error:
        return preload_failure(error, timer, watcher, ledger)


def preload_failure(error, timer, watcher, ledger, *, root=ROOT, output=OUTPUT):
    """No acquired forward arrays exist yet; retain partial read/cost evidence."""
    result = {
        "schema": "flylab.m3.e344.result.v1",
        "run_valid": False,
        "effective_qualified": False,
        "classification": "technical_invalid",
        "first_error": {
            "stage": "precontext-or-load",
            "type": type(error).__name__,
            "message": str(error)[:1024],
        },
        "secondary_errors": [],
        "counters": new_counters(),
        "construction": {"status": "not_started"},
        "acquired_raw_roster": [],
        "source_sha256": source_map(),
        "candidate_adopted": False,
        "F1_F8_complete": False,
        "biology_proven": False,
        "independent_audit_pending": True,
        "external_completion_pending": True,
    }
    try:
        result["GPU_memory_watch"] = watcher.close()
        require(
            not result["GPU_memory_watch"]["sampler_errors"]
            and not result["GPU_memory_watch"]["owned_thread_alive"],
            "owned sampler close",
        )
    except Exception as secondary:
        result["secondary_errors"].append(
            {
                "stage": "sampler-close",
                "type": type(secondary).__name__,
                "message": str(secondary)[:1024],
            }
        )
    destination = e.safe(root, output)
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in (
        (
            "selected-ledger.json",
            json_bytes(
                {"schema": "flylab.m3.e344.selected-ledger.v1", "inputs": ledger}
            ),
        ),
        ("result.json", json_bytes(result)),
        ("receipts.json", json_bytes(result)),
    ):
        require(len(payload) <= QUOTAS[name], "preload failure quota")
        with (destination / name).open("xb") as stream:
            stream.write(payload)
    try:
        timer.check("internal")
        e.stdout_bytes(json_bytes(result))
    except Exception as secondary:
        result["secondary_errors"].append(
            {
                "stage": "late-preload-publication",
                "type": type(secondary).__name__,
                "message": str(secondary)[:1024],
            }
        )
        with (destination / "revocation.json").open("xb") as stream:
            stream.write(json_bytes(result))
    return result, 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--review-sha256", required=True)
    args = parser.parse_args(argv)
    _, code = run(args.freeze_sha256, args.review_sha256)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
