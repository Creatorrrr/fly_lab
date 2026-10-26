"""E178: read-only decomposition of frozen E177 full-state recurrent current.

Only NumPy and standard-library readers are used. No model, StateStore, CUDA,
physics, neural integration, or fitting code is imported or constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OWNED = (
    "tools/analyze_c_banc_m3_saved_current.py",
    "tests/test_c_banc_saved_current.py",
)
FIELDS = ("rate", "drive", "output_mask", "adaptation")
SAMPLES = ("start", "checkpoint-1s", "main-1p1", "final")
ARMS = ("baseline", "candidate")
GROUPS = ("feedback", "navigation", "individual_fit", "other_calibration", "base_rate")
NAV_ROOTS = (
    "720575941535975873",
    "720575941537865165",
    "720575941504382999",
    "720575941553598180",
    "720575941541544028",
    "720575941594854251",
    "720575941567768423",
    "720575941429526743",
    "720575941450559566",
    "720575941592869928",
    "720575941480799395",
    "720575941496630400",
    "720575941509326430",
    "720575941438399807",
    "720575941496314697",
    "720575941572062729",
    "720575941561135887",
    "720575941623146186",
    "720575941613448668",
    "720575941503464587",
)


def fail(message: str) -> None:
    raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    name = Path(relative)
    target = (root / name).resolve()
    if (
        name.is_absolute()
        or ".." in name.parts
        or not target.is_relative_to(root.resolve())
    ):
        fail("unsafe repository-relative path: " + relative)
    return target


def exact_bits(
    actual: np.ndarray | np.float32, expected: np.ndarray | np.float32, label: str
) -> None:
    left = np.asarray(actual)
    right = np.asarray(expected)
    if (
        left.dtype != np.float32
        or right.dtype != np.float32
        or left.shape != right.shape
    ):
        fail(label + ": float32 shape/dtype mismatch")
    if not np.array_equal(left.view(np.uint32), right.view(np.uint32)):
        fail(label + ": float32 bit mismatch")


MIN_NORMAL = np.float32(np.finfo(np.float32).tiny)


def _ftz(value: np.float32) -> np.float32:
    value = np.float32(value)
    if np.float32(0) < np.abs(value) < MIN_NORMAL:
        return np.copysign(np.float32(0), value)
    return value


def _cuda_mul(left: np.float32, right: np.float32) -> np.float32:
    return _ftz(np.float32(_ftz(left) * _ftz(right)))


def _cuda_add(left: np.float32, right: np.float32) -> np.float32:
    return _ftz(np.float32(_ftz(left) + _ftz(right)))


def cuda_product(
    weight: np.float32, rate: np.float32, mask: np.float32, *, mode: str = "packed"
) -> np.float32:
    """Actual CSR product: Torch pre-mask in packed, CUDA FTZ throughout reference."""
    if mode not in ("packed", "reference"):
        fail("unknown CUDA CSR implementation")
    masked = np.float32(rate * mask) if mode == "packed" else _cuda_mul(rate, mask)
    return _cuda_mul(weight, masked)


def warp_current(
    weights: np.ndarray, rates: np.ndarray, masks: np.ndarray, *, mode: str = "packed"
) -> np.float32:
    """Exact lane-0 result of E177's 32-lane CSR kernel with --fmad=false.

    Inputs are edge-aligned float32 arrays in original CSR order. The CUDA
    kernel computes rate*mask, then weight*masked, then per-lane additions.
    """
    arrays = tuple(np.asarray(a) for a in (weights, rates, masks))
    if any(a.ndim != 1 or a.dtype != np.float32 for a in arrays):
        fail("warp inputs must be edge-aligned float32 vectors")
    if len({len(a) for a in arrays}) != 1 or any(
        not np.isfinite(a).all() for a in arrays
    ):
        fail("warp inputs differ in length or contain nonfinite values")
    if mode not in ("packed", "reference"):
        fail("unknown CUDA CSR implementation")
    lanes = [np.float32(0) for _ in range(32)]
    for lane in range(32):
        value = lanes[lane]
        for edge in range(lane, len(arrays[0]), 32):
            product = cuda_product(
                arrays[0][edge], arrays[1][edge], arrays[2][edge], mode=mode
            )
            value = _cuda_add(value, product)
        lanes[lane] = value
    for offset in (16, 8, 4, 2, 1):
        before = lanes.copy()
        for lane in range(32 - offset):
            lanes[lane] = _cuda_add(before[lane], before[lane + offset])
    return lanes[0]


def decompose_row(
    edge_ids: np.ndarray,
    pre_ids: np.ndarray,
    weights: np.ndarray,
    rates: np.ndarray,
    masks: np.ndarray,
    graph_counts: np.ndarray,
    graph_weights: np.ndarray,
    provenance_flags: dict[int, dict[str, bool]],
    *,
    mode: str = "packed",
) -> dict:
    """All edge terms and descriptive float64 sums for one complete CSR row."""
    integer = (np.asarray(edge_ids), np.asarray(pre_ids), np.asarray(graph_counts))
    floats = tuple(np.asarray(a) for a in (weights, rates, masks, graph_weights))
    n = len(integer[0])
    if any(a.ndim != 1 or len(a) != n for a in (*integer, *floats)):
        fail("row fields have incompatible lengths")
    if any(a.dtype.kind not in "iu" for a in integer) or any(
        a.dtype != np.float32 for a in floats
    ):
        fail("row fields have incompatible dtypes")
    if any(not np.isfinite(a).all() for a in floats) or np.any(integer[2] <= 0):
        fail("row has nonfinite values or invalid contacts")
    if n and (np.any(np.diff(integer[0]) != 1) or np.any(integer[1] < 0)):
        fail("row has noncontiguous edge IDs or negative source indices")
    edges = []
    for j in range(n):
        edge_id = int(integer[0][j])
        flags = provenance_flags.get(edge_id, {})
        flags = {key: bool(flags.get(key, False)) for key in GROUPS[:-1]}
        group = next((key for key in GROUPS[:-1] if flags[key]), "base_rate")
        masked = np.float32(floats[1][j] * floats[2][j])
        contribution = float(floats[0][j]) * float(masked)
        product = cuda_product(floats[0][j], floats[1][j], floats[2][j], mode=mode)
        ordinary_product = np.float32(floats[0][j] * masked)
        unmasked = float(floats[0][j]) * float(floats[1][j])
        graph_sign = float(np.sign(floats[3][j]))
        rate_base = float(np.float32(graph_sign * int(integer[2][j]) * 0.03))
        edge = {
            "edge_id": edge_id,
            "pre_id": int(integer[1][j]),
            "contacts": int(integer[2][j]),
            "graph_weight_mV": float(floats[3][j]),
            "rate_base_weight": rate_base,
            "actual_weight": float(floats[0][j]),
            "weight_to_rate_base": (
                float(floats[0][j]) / rate_base if rate_base else None
            ),
            "sign_changed_from_graph": bool(
                np.signbit(floats[0][j]) != np.signbit(floats[3][j])
            ),
            "rate": float(floats[1][j]),
            "mask": float(floats[2][j]),
            "masked_rate_float32": float(masked),
            "contribution": contribution,
            "cuda_product_float32": float(product),
            "ftz_effect": bool(
                np.asarray(product).view(np.uint32)
                != np.asarray(ordinary_product).view(np.uint32)
            ),
            "unmasked_contribution": unmasked,
            "exclusive_group": group,
            "overlapping_flags": flags,
        }
        edges.append(edge)
    terms = [e["contribution"] for e in edges]
    positive = math.fsum(x for x in terms if x > 0)
    negative = math.fsum(x for x in terms if x < 0)
    grouped = {
        key: math.fsum(e["contribution"] for e in edges if e["exclusive_group"] == key)
        for key in GROUPS
    }
    summary = {
        "edge_count": n,
        "positive_sum": positive,
        "negative_sum": negative,
        "double_sum": math.fsum(terms),
        "cuda_products_double_sum": math.fsum(e["cuda_product_float32"] for e in edges),
        "ftz_effect_count": sum(e["ftz_effect"] for e in edges),
        "ftz_product_difference_double_sum": math.fsum(
            e["contribution"] - e["cuda_product_float32"] for e in edges
        ),
        "unmasked_double_sum": math.fsum(e["unmasked_contribution"] for e in edges),
        "group_sums": grouped,
        "top_positive": [
            e["edge_id"]
            for e in sorted(edges, key=lambda e: (-e["contribution"], e["edge_id"]))
            if e["contribution"] > 0
        ][:10],
        "top_negative": [
            e["edge_id"]
            for e in sorted(edges, key=lambda e: (e["contribution"], e["edge_id"]))
            if e["contribution"] < 0
        ][:10],
    }
    if not math.isclose(
        math.fsum(grouped.values()), summary["double_sum"], abs_tol=1e-9
    ):
        fail("exclusive group decomposition does not close")
    return {"edges": edges, "summary": summary}


def _load_npz(path: Path, header: dict) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(header):
            fail(str(path) + ": NPZ member mismatch")
        loaded = {key: archive[key] for key in archive.files}
    for key, array in loaded.items():
        meta = header[key]
        if list(array.shape) != meta["shape"] or array.dtype.str != meta["dtype"]:
            fail(str(path) + ": NPZ shape/dtype mismatch: " + key)
        # The saved recruitment cohort uses NaN for unavailable motor_log_tau.
        # Its availability mask, rather than an invented value, carries meaning.
        if (
            array.dtype.kind in "fc"
            and key != "motor_log_tau"
            and not np.isfinite(array).all()
        ):
            fail(str(path) + ": nonfinite NPZ member: " + key)
    return loaded


def _load_npy(path: Path, header: dict, *, mmap: bool = False) -> np.ndarray:
    array = np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)
    if list(array.shape) != header["shape"] or array.dtype.str != header["dtype"]:
        fail(str(path) + ": NPY shape/dtype mismatch")
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        fail(str(path) + ": NPY nonfinite")
    return array


def _review_and_hashes(
    root: Path,
    freeze_path: Path,
    freeze_digest: str,
    review_path: Path,
    review_digest: str,
) -> dict:
    if sha256(freeze_path) != freeze_digest:
        fail("freeze sha256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != "flylab.m3.e178.saved-current.v1" or freeze.get(
        "owned_files"
    ) != list(OWNED):
        fail("E178 freeze contract mismatch")
    if sha256(review_path) != review_digest:
        fail("parent source review sha256 mismatch")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        review.get("schema") != freeze["source_gate_schema"]
        or review.get("accepts_saved_current") is not True
        or review.get("blocking_defects") != []
        or review.get("freeze_sha256") != freeze_digest
        or review.get("numeric_contract") != freeze.get("numeric_contract")
        or set(review.get("source_sha256", {})) != set(OWNED)
    ):
        fail("accepted exact parent source gate required")
    required_numeric = {
        "schema": "flylab.m3.e178.cuda-current-arithmetic.v1",
        "csr_ftz": True,
        "fmad": False,
        "packed_mask_product": "torch_float32_preserve_subnormals",
        "reference_mask_product": "cuda_ftz_float32",
        "implementation_request": "packed",
        "csr_eligibility": "n<=2**18 and unique_weight_bits<=2**14 and nnz<2**31",
        "descriptive_contribution": "float64(weight)*float32(rate*mask)",
        "bit_exact_required": True,
    }
    if freeze.get("numeric_contract") != required_numeric:
        fail("V2 numerical contract mismatch")
    for name in OWNED:
        if sha256(safe_path(root, name)) != review["source_sha256"][name]:
            fail("reviewed source sha256 mismatch: " + name)
    if sha256(safe_path(root, freeze["design_path"])) != freeze["design_sha256"]:
        fail("design sha256 mismatch")
    for mapping in ("input_sha256", "reused_source_sha256"):
        for name, expected in freeze[mapping].items():
            if sha256(safe_path(root, name)) != expected:
                fail(mapping + " mismatch: " + name)
    return freeze


def _snapshot_metadata(root: Path, freeze: dict, arm: str, snapshot: dict) -> None:
    directory = snapshot["directory"]
    state = json.loads(
        safe_path(root, directory + "/state.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        safe_path(root, directory + "/manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema") != "flylab.state_store.v3":
        fail("state manifest schema mismatch")
    network = state.get("network", {})
    if (
        network.get("tick") != snapshot["network_tick"]
        or state.get("control_tick") != snapshot["control_tick"]
        or network.get("dt") != snapshot["dt"]
        or network.get("model_hash") != snapshot["model_hash"]
    ):
        fail("saved state control/network tick/dt/model mismatch")
    if (
        state.get("motor_adaptation_patch", {}).get("source_cuda_implementation")
        != "packed"
    ):
        fail("saved source CUDA implementation request differs from packed")
    if snapshot["name"] not in SAMPLES or not directory.startswith(
        freeze["arms"][arm] + "/"
    ):
        fail("unexpected saved snapshot directory")
    for field in FIELDS:
        name = snapshot["fields"][field]
        if Path(name).parent != Path(directory):
            fail("selected array reference escaped frozen snapshot directory")
        if network.get(field) != {"__npy__": Path(name).name}:
            fail("saved state array reference mismatch: " + field)
        if manifest["files"].get(Path(name).name) != freeze["input_sha256"][name]:
            fail("saved state manifest array sha256 mismatch: " + field)
    if (
        manifest["files"].get("state.json")
        != freeze["input_sha256"][directory + "/state.json"]
    ):
        fail("saved state manifest state.json mismatch")


def _graph_nodes(
    path: Path, wanted: set[int], navigation_roots: set[str], n: int
) -> tuple[dict, set[int]]:
    selected = {}
    nav_ids = set()
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            node = json.loads(line)
            if node.get("id") != "flywire:banc:888:" + str(node.get("root_id")):
                fail("node namespace/root identity mismatch")
            if index in wanted:
                selected[index] = node
            if str(node["root_id"]) in navigation_roots:
                nav_ids.add(index)
    if (
        index + 1 != n
        or len(selected) != len(wanted)
        or len(nav_ids) != len(navigation_roots)
    ):
        fail("graph nodes missing, duplicate, or navigation roots unavailable")
    return selected, nav_ids


def _weight_provenance(
    graph: dict,
    actual: dict,
    calibration: dict,
    parameters: dict,
    freeze: dict,
    nav_ids: set[int],
) -> tuple[dict, dict]:
    n, m = freeze["node_count"], freeze["edge_count"]
    ptr, idx, data = actual["indptr"], actual["indices"], actual["data"]
    if (
        len(ptr) != n + 1
        or len(idx) != m
        or len(data) != m
        or int(ptr[0]) != 0
        or int(ptr[-1]) != m
        or np.any(np.diff(ptr) < 0)
        or np.any((idx < 0) | (idx >= n))
    ):
        fail("actual full CSR invalid")
    if not np.array_equal(ptr, graph["indptr"]) or not np.array_equal(
        idx, graph["indices"]
    ):
        fail("actual CSR topology differs from graph")
    base = np.asarray(np.sign(graph["weights"]) * graph["counts"] * 0.03, np.float32)
    if np.any(graph["counts"] < 1):
        fail("nonpositive graph contact count")
    motor_mask = np.zeros(n, bool)
    motor_mask[calibration["motor_ids"]] = True
    flags = {}
    for post in np.flatnonzero(motor_mask):
        for edge in range(int(ptr[post]), int(ptr[post + 1])):
            if int(idx[edge]) in nav_ids:
                if not base[edge] < 0:
                    fail("navigation port edge not inhibitory")
                base[edge] *= np.float32(64)
                flags[edge] = {"navigation": True}
    cal_ids = calibration["edge_ids"]
    cal_src = calibration["source_values"]
    cal_values = calibration["values"]
    if (
        len(np.unique(cal_ids)) != len(cal_ids)
        or np.any((cal_ids < 0) | (cal_ids >= m))
        or not np.array_equal(base[cal_ids].view(np.uint32), cal_src.view(np.uint32))
    ):
        fail(
            "assembled calibration source values differ from graph/navigation baseline"
        )
    if not np.array_equal(np.signbit(cal_src), np.signbit(cal_values)) or np.any(
        cal_values == 0
    ):
        fail("assembled calibration sign/zero mismatch")
    base[cal_ids] = cal_values
    individual = parameters["global_edge_ids"]
    pos = parameters["calibration_positions"]
    if (
        len(np.unique(individual)) != len(individual)
        or not np.array_equal(cal_ids[pos], individual)
        or not np.array_equal(
            cal_values[pos].view(np.uint32),
            parameters["native_fit_values32"].view(np.uint32),
        )
        or not np.array_equal(calibration["individual_edge_ids"], individual)
    ):
        fail("individual fit provenance differs from assembled calibration")
    for edge in cal_ids:
        flags.setdefault(int(edge), {})["other_calibration"] = True
    for edge in individual:
        flags.setdefault(int(edge), {})["individual_fit"] = True
    patch = freeze["feedback_patch"]
    feedback_ids = np.asarray(patch["edge_ids"], dtype=np.int64)
    if (
        len(feedback_ids) != 5
        or len(np.unique(feedback_ids)) != 5
        or not np.array_equal(
            base[feedback_ids].view(np.uint32),
            np.asarray(patch["source_values"], np.float32).view(np.uint32),
        )
    ):
        fail("feedback baseline source mismatch")
    for edge in feedback_ids:
        flags.setdefault(int(edge), {})["feedback"] = True
    return {"expected_baseline": base, "flags": flags}, {
        "calibrated_edges": len(cal_ids),
        "individual_edges": len(individual),
        "navigation_edges": sum(bool(v.get("navigation")) for v in flags.values()),
    }


def analyze(
    root: Path,
    freeze_path: Path,
    freeze_digest: str,
    review_path: Path,
    review_digest: str,
    output: Path,
) -> dict:
    started = time.monotonic()
    root = root.resolve()
    output = output.resolve()
    if output.exists():
        fail("output directory already exists")
    if shutil.disk_usage(output.parent).free < 1073741824:
        fail("less than 1 GiB free for E178 output")
    freeze = _review_and_hashes(
        root, freeze_path, freeze_digest, review_path, review_digest
    )
    if output != safe_path(root, freeze["output_directory"]):
        fail("output differs from frozen E178 directory")
    if (
        freeze["internal_wall_limit_s"] != 90
        or freeze["maximum_raw_bytes"] != 134217728
        or freeze["cpu_threads"] != 1
        or freeze["gpu_count"] != 0
        or freeze["model_constructions"] != 0
        or freeze["neural_steps"] != 0
        or freeze["physics_steps"] != 0
        or freeze["fit_calls"] != 0
    ):
        fail("E178 resource contract mismatch")
    if len(freeze["post_ids"]) != 7 or set(freeze["arms"]) != set(ARMS):
        fail("E178 observation contract mismatch")
    for arm in ARMS:
        samples = freeze["snapshots"][arm]
        if [x["name"] for x in samples] != list(SAMPLES):
            fail("snapshot order mismatch")
        for item in samples:
            _snapshot_metadata(root, freeze, arm, item)

    def load_npy(name: str, *, mmap: bool = False) -> np.ndarray:
        return _load_npy(
            safe_path(root, name), freeze["input_headers"][name], mmap=mmap
        )

    def load_npz(name: str) -> dict[str, np.ndarray]:
        return _load_npz(safe_path(root, name), freeze["input_headers"][name])

    graph_prefix = freeze["graph_path"] + "/"
    graph_manifest = json.loads(
        safe_path(root, graph_prefix + "manifest.json").read_text(encoding="utf-8")
    )
    if (
        graph_manifest.get("schema") != "flylab.graph.v3"
        or graph_manifest.get("graph_hash") != freeze["graph_hash"]
        or graph_manifest.get("simulated_node_count") != freeze["node_count"]
        or graph_manifest.get("pair_edge_count") != freeze["edge_count"]
        or graph_manifest.get("weight_model", {}).get("unit_weight_mV") != 0.275
    ):
        fail("frozen graph identity, shape, or mV unit mismatch")
    graph = {
        key: load_npy(graph_prefix + key + ".npy", mmap=True)
        for key in ("indptr", "indices", "counts", "weights")
    }
    arm_data = {}
    for arm in ARMS:
        prefix = freeze["arms"][arm] + "/"
        arm_data[arm] = {
            "csr": load_npz(prefix + "actual-global-csr.npz"),
            "cal": load_npz(prefix + "assembled-calibration.npz"),
            "params": load_npz(prefix + "assembled-parameters.npz"),
            "recruitment": load_npz(prefix + "recruitment-metadata.npz"),
            "boundary": load_npz(prefix + "feedback-boundary-state.npz"),
        }
    baseline = arm_data["baseline"]["csr"]
    candidate = arm_data["candidate"]["csr"]
    for key in ("indptr", "indices"):
        if not np.array_equal(baseline[key], candidate[key]):
            fail("two arm CSR topology differs")
    diff = np.flatnonzero(
        baseline["data"].view(np.uint32) != candidate["data"].view(np.uint32)
    )
    patch = freeze["feedback_patch"]
    if (
        set(diff.tolist()) != set(patch["edge_ids"])
        or not np.array_equal(
            baseline["data"][patch["edge_ids"]].view(np.uint32),
            np.asarray(patch["source_values"], np.float32).view(np.uint32),
        )
        or not np.array_equal(
            candidate["data"][patch["edge_ids"]].view(np.uint32),
            np.asarray(patch["target_values"], np.float32).view(np.uint32),
        )
    ):
        fail("two arm CSR differs beyond exact five feedback edges")
    for artifact in ("cal", "params"):
        left, right = arm_data["baseline"][artifact], arm_data["candidate"][artifact]
        if set(left) != set(right) or any(
            not np.array_equal(left[key].view(np.uint8), right[key].view(np.uint8))
            for key in left
        ):
            fail("two arm assembled " + artifact + " differs")
    for arm in ARMS:
        cohort = arm_data[arm]["recruitment"]
        for key in ("observed_ids", "motor_ids", "source_ids"):
            ids = cohort[key]
            if len(np.unique(ids)) != len(ids) or np.any(
                (ids < 0) | (ids >= freeze["node_count"])
            ):
                fail("duplicate or invalid recruitment cohort IDs: " + arm + "/" + key)
        if set(cohort["motor_ids"].tolist()) | set(
            cohort["source_ids"].tolist()
        ) != set(cohort["observed_ids"].tolist()) or set(
            cohort["motor_ids"].tolist()
        ) & set(cohort["source_ids"].tolist()):
            fail("recruitment cohort union or separation mismatch")
    if any(
        not np.array_equal(
            arm_data["baseline"]["recruitment"][key],
            arm_data["candidate"]["recruitment"][key],
        )
        for key in ("observed_ids", "motor_ids", "source_ids")
    ):
        fail("two arm recruitment cohort IDs differ")
    unique_weight_bits = len(np.unique(baseline["data"].view(np.uint32)))
    candidate_unique_weight_bits = len(np.unique(candidate["data"].view(np.uint32)))
    mode_by_arm = {
        "baseline": (
            "packed"
            if freeze["node_count"] <= 2**18
            and unique_weight_bits <= 2**14
            and freeze["edge_count"] < 2**31
            else "reference"
        ),
        "candidate": (
            "packed"
            if freeze["node_count"] <= 2**18
            and candidate_unique_weight_bits <= 2**14
            and freeze["edge_count"] < 2**31
            else "reference"
        ),
    }
    nav_roots = set(NAV_ROOTS)
    node_path = safe_path(root, graph_prefix + "nodes.jsonl")
    wanted_pre = set()
    for post in freeze["post_ids"]:
        wanted_pre.add(int(post))
        wanted_pre.update(
            int(x)
            for x in baseline["indices"][
                baseline["indptr"][post] : baseline["indptr"][post + 1]
            ]
        )
    nodes, nav_ids = _graph_nodes(
        node_path, wanted_pre, nav_roots, freeze["node_count"]
    )
    provenance, counts = _weight_provenance(
        graph,
        baseline,
        arm_data["baseline"]["cal"],
        arm_data["baseline"]["params"],
        freeze,
        nav_ids,
    )
    expected = provenance["expected_baseline"]
    if not np.array_equal(baseline["data"].view(np.uint32), expected.view(np.uint32)):
        fail("assembled baseline CSR differs from graph/navigation/calibration")
    expected_candidate = expected.copy()
    expected_candidate[patch["edge_ids"]] = np.asarray(
        patch["target_values"], np.float32
    )
    if not np.array_equal(
        candidate["data"].view(np.uint32), expected_candidate.view(np.uint32)
    ):
        fail("assembled candidate CSR differs from fixed feedback patch")
    flags = provenance["flags"]
    rows = []
    raw = {
        key: []
        for key in (
            "edge_id",
            "row_id",
            "pre_id",
            "contacts",
            "graph_weight_mV",
            "rate_base_weight",
            "actual_weight",
            "weight_to_rate_base",
            "sign_changed_from_graph",
            "rate",
            "mask",
            "masked_rate",
            "contribution",
            "cuda_product",
            "ftz_effect",
            "unmasked_contribution",
            "exclusive_group",
            "feedback",
            "navigation",
            "individual_fit",
            "other_calibration",
        )
    }
    post_ids = list(freeze["post_ids"])
    for arm in ARMS:
        source = arm_data[arm]
        boundary = source["boundary"]
        if str(
            boundary["schema"]
        ) != "flylab.m3.e177.feedback-boundary.v1" or not np.array_equal(
            boundary["observed_ids"], np.asarray(post_ids)
        ):
            fail("boundary schema or selected IDs mismatch")
        if not all(
            bool(boundary[name].all())
            for name in (
                "complete",
                "rate_available",
                "drive_available",
                "output_mask_available",
                "adaptation_available",
                "recurrent_current_available",
                "edge_weight_available",
                "direct_edge_current_available",
            )
        ):
            fail("boundary availability incomplete")
        cohort = source["recruitment"]
        cohort_index = {int(value): j for j, value in enumerate(cohort["observed_ids"])}
        for sample in freeze["snapshots"][arm]:
            if time.monotonic() - started > 90:
                fail("E178 90 second internal wall limit exceeded")
            tick = sample["control_tick"]
            if (
                int(boundary["control_tick"][tick]) != tick
                or int(boundary["network_tick"][tick]) != sample["network_tick"]
            ):
                fail("boundary tick mismatch")
            state = {name: load_npy(sample["fields"][name]) for name in FIELDS}
            if any(arr.shape != (freeze["node_count"],) for arr in state.values()):
                fail("selected state array is not full graph row")
            for j, post in enumerate(post_ids):
                for name in FIELDS:
                    exact_bits(
                        np.asarray(state[name][post]),
                        np.asarray(boundary[name][tick, j]),
                        arm + "/" + sample["name"] + "/" + str(post) + "/" + name,
                    )
                start, end = (
                    int(source["csr"]["indptr"][post]),
                    int(source["csr"]["indptr"][post + 1]),
                )
                pre = source["csr"]["indices"][start:end]
                weight = source["csr"]["data"][start:end]
                rate = state["rate"][pre]
                mask = state["output_mask"][pre]
                current = warp_current(weight, rate, mask, mode=mode_by_arm[arm])
                exact_bits(
                    np.asarray(current),
                    np.asarray(boundary["recurrent_current"][tick, j]),
                    arm + "/" + sample["name"] + "/" + str(post) + "/current",
                )
                edge_ids = np.arange(start, end, dtype=np.int64)
                piece = decompose_row(
                    edge_ids,
                    pre,
                    weight,
                    rate,
                    mask,
                    graph["counts"][start:end],
                    graph["weights"][start:end],
                    flags,
                    mode=mode_by_arm[arm],
                )
                for edge in piece["edges"]:
                    raw["edge_id"].append(edge["edge_id"])
                    raw["row_id"].append(len(rows))
                    for key in (
                        "pre_id",
                        "contacts",
                        "graph_weight_mV",
                        "rate_base_weight",
                        "actual_weight",
                        "weight_to_rate_base",
                        "sign_changed_from_graph",
                        "rate",
                        "mask",
                        "contribution",
                        "unmasked_contribution",
                        "exclusive_group",
                        "ftz_effect",
                    ):
                        raw[key].append(edge[key])
                    raw["masked_rate"].append(edge["masked_rate_float32"])
                    raw["cuda_product"].append(edge["cuda_product_float32"])
                    for key in GROUPS[:-1]:
                        raw[key].append(edge["overlapping_flags"][key])
                metadata = None
                if post in cohort_index:
                    k = cohort_index[post]
                    metadata = {
                        key: float(cohort[key][k])
                        for key in ("tau", "a", "threshold", "cap")
                    }
                    if metadata["tau"] <= 0 or metadata["cap"] <= 0:
                        fail("invalid recruitment cohort tau/cap")
                    metadata["effective_input"] = (
                        float(state["drive"][post])
                        + float(current)
                        - metadata["threshold"]
                        - float(state["adaptation"][post])
                    )
                    activation = (metadata["a"] / metadata["cap"]) * metadata[
                        "effective_input"
                    ]
                    metadata["instantaneous_target_rate"] = max(
                        0.0, metadata["cap"] * math.tanh(activation)
                    )
                row = {
                    "arm": arm,
                    "snapshot": sample["name"],
                    "control_tick": tick,
                    "network_tick": sample["network_tick"],
                    "post_id": post,
                    "csr_implementation": mode_by_arm[arm],
                    "rate": float(state["rate"][post]),
                    "drive": float(state["drive"][post]),
                    "output_mask": float(state["output_mask"][post]),
                    "adaptation": float(state["adaptation"][post]),
                    "warp_current": float(current),
                    "boundary_current": float(boundary["recurrent_current"][tick, j]),
                    "recruitment_metadata": metadata,
                    "recruitment_status": "available"
                    if metadata is not None
                    else "unavailable_not_in_cohort",
                    "edge_offset": len(raw["edge_id"]) - len(piece["edges"]),
                    "edge_count": len(piece["edges"]),
                    **piece["summary"],
                }
                rows.append(row)
            for k, (edge_id, pre_id, post_id) in enumerate(
                zip(patch["edge_ids"], patch["pre_ids"], patch["post_ids"], strict=True)
            ):
                if not (
                    int(source["csr"]["indices"][edge_id]) == pre_id
                    and int(
                        np.searchsorted(source["csr"]["indptr"], edge_id, side="right")
                        - 1
                    )
                    == post_id
                ):
                    fail("feedback patch edge endpoints differ from freeze")
                weight = np.asarray(source["csr"]["data"][edge_id])
                exact_bits(
                    weight,
                    np.asarray(boundary["edge_weight"][tick, k]),
                    arm + "/" + sample["name"] + "/feedback-weight",
                )
                direct = np.float32(
                    weight
                    * np.float32(state["rate"][pre_id] * state["output_mask"][pre_id])
                )
                exact_bits(
                    np.asarray(direct),
                    np.asarray(boundary["direct_edge_current"][tick, k]),
                    arm + "/" + sample["name"] + "/feedback-direct",
                )
    if len(rows) != 56:
        fail("E178 requires exactly 56 complete rows")
    columns = {}
    for key, values in raw.items():
        dtype = (
            np.int64
            if key == "edge_id"
            else np.int32
            if key in ("row_id", "pre_id", "contacts")
            else np.bool_
            if key in ("sign_changed_from_graph", "ftz_effect", *GROUPS[:-1])
            else np.int8
            if key == "exclusive_group"
            else np.float64
            if key in ("contribution", "unmasked_contribution")
            else np.float32
        )
        if key == "exclusive_group":
            values = [GROUPS.index(value) for value in values]
        if key == "weight_to_rate_base":
            values = [float("nan") if value is None else value for value in values]
        columns[key] = np.asarray(values, dtype=dtype)
    raw_bytes = sum(value.nbytes for value in columns.values())
    if raw_bytes > freeze["maximum_raw_bytes"]:
        fail("E178 raw edge output exceeds 128 MiB")
    if time.monotonic() - started > 90:
        fail("E178 90 second internal wall limit exceeded")
    staged = Path(tempfile.mkdtemp(prefix=".e178-staging-", dir=output.parent))
    try:
        if not staged.resolve().is_relative_to(root) or staged.parent != output.parent:
            fail("unsafe E178 staging directory")

        def write_json(name: str, value: object) -> None:
            (staged / name).write_text(
                json.dumps(
                    value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )

        np.savez_compressed(staged / "edge-contributions.npz", **columns)
        write_json("row-summary.json", rows)
        write_json(
            "source-nodes.json", {str(key): value for key, value in nodes.items()}
        )
        write_json(
            "input-receipt.json",
            {
                "freeze_sha256": freeze_digest,
                "source_review_sha256": review_digest,
                "verified_input_sha256": freeze["input_sha256"],
                "verified_reused_source_sha256": freeze["reused_source_sha256"],
                "exclusive_group_codes": {
                    str(i): name for i, name in enumerate(GROUPS)
                },
                "csr_implementation": mode_by_arm,
                "unique_weight_bits": {
                    "baseline": unique_weight_bits,
                    "candidate": candidate_unique_weight_bits,
                },
                "csr_provenance_counts": counts,
            },
        )
        elapsed = time.monotonic() - started
        if elapsed > 90:
            fail("E178 90 second internal wall limit exceeded")
        result = {
            "schema": "flylab.m3.e178.saved-current-result.v1",
            "technical_failure": False,
            "complete_rows": 56,
            "bitexact_current_rows": 56,
            "raw_bytes": raw_bytes,
            "wall_s": elapsed,
            "model_constructions": 0,
            "neural_steps": 0,
            "physics_steps": 0,
            "fit_calls": 0,
            "gpu_count": 0,
            "candidate_adopted": False,
            "F6_exposed": False,
            "whole_goal_complete": False,
        }
        write_json("result.json", result)
        expected_names = {
            "edge-contributions.npz",
            "row-summary.json",
            "source-nodes.json",
            "input-receipt.json",
            "result.json",
        }
        manifest = {
            "schema": "flylab.m3.e178.output-manifest.v1",
            "files": {
                name: {
                    "bytes": (staged / name).stat().st_size,
                    "sha256": sha256(staged / name),
                }
                for name in sorted(expected_names)
            },
        }
        write_json("output-manifest.json", manifest)
        if {p.name for p in staged.iterdir()} != expected_names | {
            "output-manifest.json"
        }:
            fail("E178 output inventory mismatch")
        if (
            sum(p.stat().st_size for p in staged.iterdir())
            > freeze["maximum_raw_bytes"]
        ):
            fail("E178 complete output exceeds 128 MiB")
        if time.monotonic() - started > 90:
            fail("E178 90 second internal wall limit exceeded")
        if output.exists():
            fail("E178 output appeared during staged write")
        staged.rename(output)
        return result
    finally:
        if staged.exists():
            if (
                not staged.resolve().is_relative_to(root)
                or staged.parent != output.parent
            ):
                fail("unsafe E178 staging cleanup target")
            shutil.rmtree(staged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    result = analyze(
        ROOT,
        args.freeze,
        args.freeze_sha256,
        args.source_review,
        args.source_review_sha256,
        args.output,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
