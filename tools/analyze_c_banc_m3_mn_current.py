"""E187 saved MN incoming-current attribution. No model or simulation imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path, PurePosixPath

import numpy as np

if __package__ in (None, ""):
    # The parent supervisor invokes this file by path with an empty PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analyze_c_banc_m3_saved_current import (
    NAV_ROOTS,
    cuda_product,
    exact_bits,
    warp_current,
)

ROOT = Path(__file__).resolve().parents[1]
DESIGN = "docs/BANC_M3_E187_MN_CURRENT_DESIGN_20260924.md"
METADATA = "verification/banc-natural-gait-m3-20260919/parent-e187-frozen-metadata.json"
OUTPUT = "verification/banc-natural-gait-m3-20260919/e187-mn-current"
SUPERVISOR = "verification/banc-natural-gait-m3-20260919/parent-e187-supervise.py"
OWNED = ("tools/analyze_c_banc_m3_mn_current.py", "tests/test_c_banc_mn_current.py")
FIELDS = ("rate", "drive", "output_mask", "adaptation")
POST_IDS = (144018, 97727, 43618, 54014, 144453)
SELECTED = (131, 219, 308, 391, 433, 480, 521, 564)
ALL_EVENTS = (131, 148, 219, 256, 308, 345, 391, 433, 480, 521, 564)
BANDS = {"pre": (-20, -10), "lead": (-10, 0), "post": (0, 10), "recover": (10, 21)}
PARAMETERS = {
    "control_dt_s": 0.005,
    "native_dt_s": 0.0001,
    "control_rows": 601,
    "native_rows": 30000,
    "cpu_threads": 1,
    "gpu_count": 0,
    "internal_wall_s": 40,
    "supervisor_wall_s": 45,
    "automatic_retries": 0,
    "maximum_output_bytes": 268435456,
    "minimum_free_bytes": 2147483648,
    "inherited_cumulative_wall_s": 4346.283222499653,
    "cumulative_cap_s": 4994.0,
}


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_path(name):
    require(isinstance(name, str) and name and "\\" not in name, "unsafe path")
    parts = PurePosixPath(name)
    require(
        not parts.is_absolute()
        and ":" not in name
        and all(part not in ("", ".", "..") for part in parts.parts),
        "unsafe path",
    )
    target = ROOT.joinpath(*parts.parts)
    require(target.resolve().is_relative_to(ROOT.resolve()), "path escaped root")
    require(
        not any(p.is_symlink() for p in (target, *target.parents) if p != ROOT.parent),
        "symlink path",
    )
    return target


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gate(freeze_path, review_path, output):
    """Hash all sources and original inputs before loading any numeric array."""
    freeze, review = read_json(freeze_path), read_json(review_path)
    require(
        freeze.get("schema") == "flylab.m3.e187.mn-current.freeze.v1", "freeze schema"
    )
    require(
        review.get("schema") == "flylab.m3.e187.mn-current.source-review.v1",
        "review schema",
    )
    freeze_sha = digest(freeze_path)
    require(
        review.get("approved") is True
        and review.get("blocking_defects") == []
        and review.get("freeze_sha256") == freeze_sha,
        "parent source approval",
    )
    require(
        freeze.get("design_path") == DESIGN
        and freeze.get("design_sha256") == digest(safe_path(DESIGN)),
        "design identity",
    )
    require(
        freeze.get("metadata_path") == METADATA
        and freeze.get("metadata_sha256") == digest(safe_path(METADATA)),
        "metadata identity",
    )
    require(
        freeze.get("output_path") == OUTPUT
        and output.resolve() == safe_path(OUTPUT).resolve(),
        "output identity",
    )
    require(freeze.get("parameters") == PARAMETERS, "resource contract")
    require(
        freeze.get("parent_supervisor_sha256") == digest(safe_path(SUPERVISOR)),
        "supervisor changed",
    )
    metadata = read_json(safe_path(METADATA))
    require(
        metadata.get("schema") == "flylab.m3.e187.mn-current.metadata.v1"
        and metadata.get("design_path") == DESIGN
        and metadata.get("design_sha256") == freeze["design_sha256"]
        and metadata.get("owned_files") == list(OWNED)
        and metadata.get("original_numeric_analysis_runs") == 0
        and metadata.get("new_models") == 0
        and metadata.get("gpu_work") == 0
        and metadata.get("F6_exposed") == 0
        and metadata.get("candidate_adopted") is False,
        "metadata contract",
    )
    inputs, reused = metadata.get("inputs"), metadata.get("reused_sources")
    require(
        isinstance(inputs, dict)
        and len(inputs) == 511
        and isinstance(reused, dict)
        and len(reused) == 11
        and isinstance(metadata.get("headers"), dict),
        "full inventories required",
    )
    sources = {name: digest(safe_path(name)) for name in (*OWNED, *reused)}
    require(
        len(sources) == 13
        and freeze.get("source_hashes") == sources
        and review.get("source_hashes") == sources
        and reused == {k: sources[k] for k in reused},
        "source changed",
    )
    require(
        review.get("design_sha256") == freeze["design_sha256"]
        and review.get("metadata_sha256") == freeze["metadata_sha256"],
        "review identity",
    )
    require(freeze.get("input_sha256") == inputs, "input inventory differs")
    for name, sha in inputs.items():
        require(
            isinstance(sha, str) and len(sha) == 64 and digest(safe_path(name)) == sha,
            "input changed: " + name,
        )
    require(set(metadata["headers"]) <= set(inputs), "header inventory escaped inputs")
    prior = read_json(safe_path(metadata["e185_review_path"]))
    require(
        prior.get("saved_raw_analysis_accepted") is True
        and prior.get("child_observation_valid") is True
        and prior.get("original_driver_accepted") is False,
        "E185 review changed",
    )
    previous = read_json(safe_path(metadata["previous_review_path"]))
    require(
        previous.get("run_validity") == "valid"
        and previous.get("candidate_adopted") is False,
        "E186 review changed",
    )
    return metadata, freeze_sha


def load_array(metadata, name):
    """Shape and dtype are frozen in parent metadata; no pickle or object arrays."""
    header = metadata["headers"].get(name)
    require(header is not None, "unfrozen array: " + name)
    path = safe_path(name)
    if name.endswith(".npz"):
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == set(header), "NPZ members differ: " + name)
            arrays = {key: archive[key] for key in archive.files}
        for key, array in arrays.items():
            check_header(array, header[key], name + "/" + key)
        return arrays
    require(name.endswith(".npy"), "unsupported array: " + name)
    array = np.load(path, allow_pickle=False)
    check_header(array, header, name)
    return array


def check_header(array, header, label):
    require(
        list(array.shape) == header["shape"] and array.dtype.str == header["dtype"],
        "array header differs: " + label,
    )
    if array.ndim > 1:
        require(
            bool(array.flags.f_contiguous and not array.flags.c_contiguous)
            == bool(header["fortran_order"]),
            "array order differs: " + label,
        )


def exact(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape and a.dtype == b.dtype, label + " shape/dtype")
    if a.dtype.kind in "fc":
        require(np.array_equal(np.isnan(a), np.isnan(b)), label + " NaN mask")
        require(np.array_equal(a, b, equal_nan=True), label + " values")
    else:
        require(np.array_equal(a, b), label + " values")


def state_semantics(metadata):
    """Compare full initial state, resolving NPY leaves irrespective of filenames."""
    a_dir = metadata["snapshots"][0]["directory"]
    b_dir = metadata["comparison_start_directory"]
    a, b = (
        read_json(safe_path(a_dir + "/state.json")),
        read_json(safe_path(b_dir + "/state.json")),
    )

    def walk(left, right, key):
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) == set(right) == {"__npy__"}:
                exact(
                    load_array(metadata, a_dir + "/" + left["__npy__"]),
                    load_array(metadata, b_dir + "/" + right["__npy__"]),
                    key,
                )
                return
            require(set(left) == set(right), "initial state keys " + key)
            for part in sorted(left):
                if key == "" and part == "wall_s":
                    continue
                walk(left[part], right[part], key + "/" + part)
        elif isinstance(left, list) and isinstance(right, list):
            require(len(left) == len(right), "initial state list " + key)
            for i, (x, y) in enumerate(zip(left, right)):
                walk(x, y, key + "/" + str(i))
        else:
            require(left == right, "initial state scalar " + key)

    walk(a, b, "")
    require(
        a["network"]["model_hash"]
        == b["network"]["model_hash"]
        == "a31f899cd948a4297a73db66569e5edc2210ac9aa3601aa83e96cb9ff68ce244"
        and a["graph_hash"] == metadata["graph_hash"]
        and a["network"]["dt"] == 0.00025
        and a["control_tick"] == a["network"]["tick"] == 0
        and all(v is False for v in a["cuts"].values())
        and a["motor_adaptation_patch"]["source_cuda_implementation"] == "packed",
        "initial model/cuts/CUDA identity",
    )
    return {
        "initial_state_semantically_equal": True,
        "network_model_hash": a["network"]["model_hash"],
        "graph_hash": a["graph_hash"],
        "cuts": a["cuts"],
    }


def prefix_semantics(metadata):
    directories = metadata["directories"]
    e150, e185 = directories["e150"], directories["e185"]
    t150, t185 = (
        read_json(safe_path(e150 + "/trace.json")),
        read_json(safe_path(e185 + "/trace.json")),
    )
    require(
        len(t150["samples"]) == 2001 and len(t185["samples"]) == 601, "trace length"
    )
    for name in ("graph_hash", "model_hash", "body_hash", "adapter_hash"):
        require(t150[name] == t185[name], "trace identity " + name)
    for i, (left, right) in enumerate(zip(t150["samples"], t185["samples"])):
        require(left == right, "trace prefix row " + str(i))
        require(
            left["control_tick"] == i and abs(left["time_s"] - i * 0.005) <= 2e-8,
            "trace clock " + str(i),
        )
    for filename, rows in (
        ("native-recruitment-state.npz", 601),
        ("native-contact.npz", 30000),
    ):
        left = load_array(metadata, e150 + "/" + filename)
        right = load_array(metadata, e185 + "/" + filename)
        require(set(left) == set(right), filename + " members")
        for key in left:
            if left[key].ndim == 0:
                exact(left[key], right[key], filename + "/" + key)
                continue
            require(
                len(left[key]) >= rows and len(right[key]) == rows,
                filename + " rows " + key,
            )
            exact(left[key][:rows], right[key], filename + "/" + key)
        if filename.startswith("native-recruitment"):
            require(
                np.array_equal(right["control_tick"], np.arange(601))
                and np.array_equal(right["network_tick"], np.arange(601) * 20)
                and np.max(np.abs(right["physics_time_s"] - np.arange(601) * 0.005))
                <= 2e-8
                and np.all(right["complete"]),
                "recruitment clock/completeness",
            )
        else:
            require(
                np.max(np.abs(right["time_s"] - np.arange(1, 30001) * 0.0001)) <= 2e-8,
                "native contact clock",
            )
    theta = read_json(safe_path(e185 + "/arm-provenance.json"))["theta"]
    require(
        isinstance(theta, list) and all(x == 0 for x in theta), "E185 theta wrapper"
    )
    return t150, {
        "trace_prefix_rows": 601,
        "recruitment_prefix_rows": 601,
        "native_contact_prefix_rows": 30000,
        "theta": theta,
    }


def graph_provenance(graph, actual, calibration, params, nav_ids, n, m):
    """E178 graph → navigation → calibration → individual mapping, without a feedback patch."""
    ptr, idx, data = actual["indptr"], actual["indices"], actual["data"]
    require(
        ptr.shape == (n + 1,)
        and idx.shape == data.shape == (m,)
        and ptr[0] == 0
        and ptr[-1] == m
        and np.all(np.diff(ptr) >= 0)
        and np.all((idx >= 0) & (idx < n)),
        "CSR shape/topology",
    )
    exact(ptr, graph["indptr"], "graph indptr")
    exact(idx, graph["indices"], "graph indices")
    base = np.asarray(np.sign(graph["weights"]) * graph["counts"] * 0.03, np.float32)
    require(np.all(graph["counts"] > 0), "graph contacts")
    flags = {}
    motor = np.zeros(n, bool)
    motor[calibration["motor_ids"]] = True
    for post in np.flatnonzero(motor):
        for edge in range(int(ptr[post]), int(ptr[post + 1])):
            if int(idx[edge]) in nav_ids:
                require(base[edge] < 0, "navigation sign")
                base[edge] *= np.float32(64)
                flags[edge] = {"navigation": True}
    ids = calibration["edge_ids"]
    require(
        len(np.unique(ids)) == len(ids) and np.all((ids >= 0) & (ids < m)),
        "calibration ids",
    )
    exact_bits(base[ids], calibration["source_values"], "calibration source")
    base[ids] = calibration["values"]
    positions = params["calibration_positions"]
    individual = params["global_edge_ids"]
    require(
        np.array_equal(ids[positions], individual)
        and np.array_equal(calibration["individual_edge_ids"], individual),
        "individual mapping",
    )
    exact_bits(
        calibration["values"][positions],
        params["native_fit_values32"],
        "individual values",
    )
    for edge in ids:
        flags.setdefault(int(edge), {})["calibration"] = True
    for edge in individual:
        flags.setdefault(int(edge), {})["individual_fit"] = True
    exact_bits(base, data, "actual assembled CSR")
    return flags


def windows_from_e186(aligned):
    require(
        np.array_equal(aligned["event_indices_all"], ALL_EVENTS)
        and np.array_equal(aligned["event_indices_selected"], SELECTED),
        "E186 event indices",
    )
    windows = {
        "whole": np.arange(1, 601),
        "evaluation": np.arange(100, 600),
        "startup99": np.arange(1, 100),
        "selected8": np.concatenate([np.arange(i - 20, i + 21) for i in SELECTED]),
    }
    for band, (a, b) in BANDS.items():
        windows["selected8:" + band] = np.concatenate(
            [np.arange(i + a, i + b) for i in SELECTED]
        )
    for i in ALL_EVENTS:
        for band, (a, b) in BANDS.items():
            windows[f"event{i}:{band}"] = np.arange(i + a, i + b)
    require(
        list(windows) == aligned["window_labels"].tolist() and len(windows) == 52,
        "E186 windows differ",
    )
    count = aligned["native_count"]
    signed = aligned["native_signed_force_sum_bw"]
    magnitude = aligned["native_negative_magnitude_sum_bw"]
    require(
        count.shape == (601, 6, 3)
        and signed.shape == count.shape
        and magnitude.shape == (601, 6)
        and np.all(count[0] == 0)
        and np.all(magnitude[0] == 0),
        "native force bins",
    )
    for j, rows in enumerate(windows.values()):
        exact(
            count[rows].sum(axis=0),
            aligned["condition_native_count"][j],
            "E186 condition count",
        )
        require(
            np.allclose(
                signed[rows].sum(axis=0),
                aligned["condition_signed_force_sum_bw"][j],
                atol=1e-10,
                rtol=0,
            ),
            "E186 signed force window",
        )
    return windows


def conditioned(values, rows, count, magnitude):
    """All columns belong to RM; invalid values remove their own weights."""
    x = np.asarray(values[rows], np.float64)
    require(x.ndim == 2, "conditioned feature shape")
    out = np.full((3, x.shape[1]), np.nan)
    denominators = np.zeros((3, x.shape[1]), np.float64)
    finite = np.isfinite(x)
    clean = np.where(finite, x, 0)
    for sign in range(3):
        weight = np.where(finite, count[rows, 4, sign, None], 0)
        denominators[sign] = weight.sum(axis=0)
        np.divide(
            (clean * weight).sum(axis=0),
            denominators[sign],
            out=out[sign],
            where=denominators[sign] > 0,
        )
    mag_weight = np.where(finite, magnitude[rows, 4, None], 0)
    mag_denominator = mag_weight.sum(axis=0)
    mag_mean = np.full(x.shape[1], np.nan)
    np.divide(
        (clean * mag_weight).sum(axis=0),
        mag_denominator,
        out=mag_mean,
        where=mag_denominator > 0,
    )
    mean = np.full(x.shape[1], np.nan)
    n = finite.sum(axis=0)
    np.divide(clean.sum(axis=0), n, out=mean, where=n > 0)
    return out, denominators, mag_mean, mag_denominator, mean, n


def partial_terms(observed_rate, observed_ids, edge_ids, pre_ids, weights, mask):
    """Known edges only; unknown pre-neurons remain NaN, never zero-filled."""
    rate = np.asarray(observed_rate)
    ids = np.asarray(observed_ids)
    edge_ids = np.asarray(edge_ids)
    pre_ids = np.asarray(pre_ids)
    weights = np.asarray(weights)
    mask = np.asarray(mask)
    require(
        rate.ndim == 2
        and rate.dtype == np.float32
        and ids.shape == (rate.shape[1],)
        and len(np.unique(ids)) == len(ids)
        and edge_ids.shape == pre_ids.shape
        and weights.ndim == 1
        and weights.dtype == np.float32
        and mask.ndim == 1
        and np.all((edge_ids >= 0) & (edge_ids < len(weights)))
        and np.all((pre_ids >= 0) & (pre_ids < len(mask))),
        "partial term shape/dtype",
    )
    known = np.isin(pre_ids, ids)
    terms = np.full((len(rate), len(pre_ids)), np.nan, np.float64)
    cuda = np.full((len(rate), len(pre_ids)), np.nan, np.float32)
    columns = {int(node): i for i, node in enumerate(ids)}
    for j in np.flatnonzero(known):
        pre = int(pre_ids[j])
        r = rate[:, columns[pre]]
        masked = np.asarray(r * mask[pre], np.float32)
        w = weights[edge_ids[j]]
        terms[:, j] = float(w) * masked.astype(np.float64)
        cuda[:, j] = np.array(
            [cuda_product(w, x, mask[pre], mode="packed") for x in r], np.float32
        )
    return known, terms, cuda


def json_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def boundary_coverage(boundaries, edge_ids, edge_post_ids, observed_pre):
    """Coverage at all five full-state endpoints, including 10s after the 3s prefix."""
    ids = np.asarray(edge_ids)
    posts = np.asarray(edge_post_ids)
    known = np.asarray(observed_pre)
    require(
        ids.ndim == posts.ndim == known.ndim == 1
        and ids.shape == posts.shape == known.shape
        and len(np.unique(ids)) == len(ids),
        "boundary coverage inventory",
    )
    lookup = {
        int(edge): (int(post), bool(is_known))
        for edge, post, is_known in zip(ids, posts, known)
    }
    coverage = []
    for boundary in boundaries:
        tick = int(boundary["control_tick"])
        for row in boundary["posts"]:
            post = int(row["post_id"])
            row_ids = row["edge_ids"]
            terms = np.asarray(row["edge_terms"], np.float64)
            require(row_ids == ids[posts == post].tolist(), "boundary CSR edge order")
            require(
                terms.shape == (len(row_ids),)
                and np.isfinite(terms).all()
                and all(lookup.get(int(edge), (None,))[0] == post for edge in row_ids),
                "boundary coverage row identity",
            )
            row_known = np.asarray([lookup[int(edge)][1] for edge in row_ids], bool)
            all_abs = math.fsum(abs(float(x)) for x in terms)
            require(
                math.isclose(
                    all_abs, row["absolute_input_sum"], rel_tol=0, abs_tol=1e-9
                ),
                "boundary absolute current",
            )
            local = terms[row_known]
            known_abs = math.fsum(abs(float(x)) for x in local)
            positive_abs = math.fsum(float(x) for x in local if x > 0)
            negative_abs = math.fsum(-float(x) for x in local if x < 0)
            coverage.append(
                {
                    "post_id": post,
                    "boundary": boundary["name"],
                    "control_tick": tick,
                    "known_absolute_fraction": known_abs / all_abs if all_abs else None,
                    "known_positive_absolute_fraction": positive_abs / all_abs
                    if all_abs
                    else None,
                    "known_negative_absolute_fraction": negative_abs / all_abs
                    if all_abs
                    else None,
                    "known_absolute_sum": known_abs,
                    "known_positive_absolute_sum": positive_abs,
                    "known_negative_absolute_sum": negative_abs,
                    "all_absolute_sum": all_abs,
                }
            )
    return coverage


def analyze(metadata):
    started = time.perf_counter()
    identity = state_semantics(metadata)
    trace, prefix = prefix_semantics(metadata)
    e150, e185 = metadata["directories"]["e150"], metadata["directories"]["e185"]
    signals = load_array(metadata, e150 + "/recruitment-signals.npz")
    anchor = load_array(metadata, e150 + "/start-observation.npz")
    cohort = load_array(metadata, e150 + "/recruitment-metadata.npz")
    native = load_array(metadata, e185 + "/native-recruitment-state.npz")
    aligned = load_array(metadata, metadata["e186_directory"] + "/aligned.npz")
    require(
        signals["rate"].shape == (2000, 446)
        and np.array_equal(signals["observed_ids"], anchor["observed_ids"])
        and np.array_equal(signals["observed_ids"], cohort["observed_ids"])
        and np.max(np.abs(signals["time_s"] - np.arange(1, 2001) * 0.005)) <= 2e-8,
        "E150 cohort/clock",
    )
    observed = signals["observed_ids"]
    require(
        len(np.unique(observed)) == 446 and set(POST_IDS) <= set(observed.tolist()),
        "five posts missing from E150 cohort",
    )
    motor_ids = cohort["motor_ids"]
    require(
        len(motor_ids) == 391 and np.array_equal(motor_ids, aligned["motor_ids"]),
        "motor roster IDs",
    )
    roster = read_json(safe_path(e150 + "/start/state.json"))["motor_adaptation_patch"][
        "motor_roster"
    ]
    require(roster == aligned["motor_roots_391"].tolist(), "motor roster roots")
    for i in range(601):
        exact(
            np.asarray(trace["samples"][i]["motor_rates"], np.float64),
            native["motor_rate"][i],
            "trace/native motor rate " + str(i),
        )
    exact(
        signals["rate"][
            :600, [int(np.flatnonzero(observed == m)[0]) for m in motor_ids]
        ].astype(np.float64),
        native["motor_rate"][1:601],
        "signal/native motor rate",
    )
    exact(
        signals["muscle_activation"][:600],
        native["activation"][1:601],
        "signal/native muscle activation",
    )
    exact(anchor["motor_rate_391"], native["motor_rate"][0], "anchor motor rate")
    exact(anchor["muscle_activation"], native["activation"][0], "anchor activation")
    windows = windows_from_e186(aligned)
    graph_dir = metadata["graph_directory"]
    manifest = read_json(safe_path(graph_dir + "/manifest.json"))
    require(
        manifest["graph_hash"] == metadata["graph_hash"]
        and manifest["simulated_node_count"] == metadata["node_count"]
        and manifest["pair_edge_count"] == metadata["edge_count"],
        "graph manifest",
    )
    graph = {
        k: load_array(metadata, graph_dir + "/" + k + ".npy")
        for k in ("indptr", "indices", "counts", "weights")
    }
    actual = load_array(metadata, e150 + "/actual-global-csr.npz")
    cal = load_array(metadata, e150 + "/assembled-calibration.npz")
    params = load_array(metadata, e150 + "/assembled-parameters.npz")
    # E185 persists CSR and adapter identity in its initial state; all state
    # arrays were compared above, including efficacy and network parameters.
    wanted = set(POST_IDS)
    ptr, idx, weight = actual["indptr"], actual["indices"], actual["data"]
    for post in POST_IDS:
        wanted.update(int(x) for x in idx[ptr[post] : ptr[post + 1]])
    roots = {}
    nav_ids = set()
    with safe_path(graph_dir + "/nodes.jsonl").open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            node = json.loads(line)
            require(
                node.get("id") == "flywire:banc:888:" + str(node.get("root_id")),
                "graph node namespace",
            )
            if i in wanted:
                roots[i] = node
            if str(node["root_id"]) in NAV_ROOTS:
                nav_ids.add(i)
    require(
        i + 1 == metadata["node_count"]
        and len(roots) == len(wanted)
        and len(nav_ids) == 20,
        "graph node inventory",
    )
    flags = graph_provenance(
        graph,
        actual,
        cal,
        params,
        nav_ids,
        metadata["node_count"],
        metadata["edge_count"],
    )
    snapshots = metadata["snapshots"]
    require(
        [s["control_tick"] for s in snapshots] == [0, 200, 220, 2000, 600],
        "boundary schedule",
    )
    boundary = []
    all_masks = []
    observed_pos = {int(node): j for j, node in enumerate(observed)}
    post_positions = np.array([observed_pos[node] for node in POST_IDS])
    series = {}
    for key in ("rate", "drive", "recurrent_current", "adaptation"):
        series[key] = np.concatenate([anchor[key][None], signals[key][:600]], axis=0)[
            :, post_positions
        ]
    series["net_input"] = (
        series["recurrent_current"].astype(np.float64)
        + series["drive"].astype(np.float64)
        - series["adaptation"].astype(np.float64)
    )
    # Threshold is a fixed network parameter, not fitted to this record.
    threshold = cohort["threshold"][post_positions].astype(np.float64)
    series["net_input"] -= threshold[None]
    for snap in snapshots:
        require(
            time.perf_counter() - started <= PARAMETERS["internal_wall_s"],
            "internal wall budget",
        )
        state = read_json(safe_path(snap["directory"] + "/state.json"))
        tick = snap["control_tick"]
        require(
            state["control_tick"] == tick
            and state["network"]["tick"] == tick * 20
            and state["network"]["dt"] == 0.00025
            and state["network"]["model_hash"] == identity["network_model_hash"]
            and all(v is False for v in state["cuts"].values()),
            "boundary state identity",
        )
        full = {key: load_array(metadata, snap["fields"][key]) for key in FIELDS}
        require(
            all(
                x.shape == (metadata["node_count"],) and x.dtype == np.float32
                for x in full.values()
            ),
            "full boundary shape",
        )
        all_masks.append(full["output_mask"])
        endpoint = anchor if tick == 0 else signals
        row = None if tick == 0 else tick - 1
        for key in ("rate", "drive", "adaptation"):
            exact_bits(
                full[key][observed],
                endpoint[key] if row is None else endpoint[key][row],
                snap["name"] + "/" + key,
            )
        per_post = []
        for post in POST_IDS:
            start, stop = int(ptr[post]), int(ptr[post + 1])
            pre = idx[start:stop]
            rates, masks = full["rate"][pre], full["output_mask"][pre]
            current = warp_current(weight[start:stop], rates, masks, mode="packed")
            expected = (
                endpoint["recurrent_current"][observed_pos[post]]
                if row is None
                else endpoint["recurrent_current"][row, observed_pos[post]]
            )
            exact_bits(
                np.asarray(current),
                np.asarray(expected),
                snap["name"] + "/current/" + str(post),
            )
            terms = np.array(
                [
                    float(w) * float(np.float32(r * m))
                    for w, r, m in zip(weight[start:stop], rates, masks)
                ],
                np.float64,
            )
            products = np.array(
                [
                    cuda_product(w, r, m, mode="packed")
                    for w, r, m in zip(weight[start:stop], rates, masks)
                ],
                np.float32,
            )
            per_post.append(
                {
                    "post_id": post,
                    "edge_ids": list(range(start, stop)),
                    "cuda_current_float32": float(current),
                    "descriptive_positive_sum": math.fsum(x for x in terms if x > 0),
                    "descriptive_negative_sum": math.fsum(x for x in terms if x < 0),
                    "descriptive_total": math.fsum(terms),
                    "edge_terms": terms.tolist(),
                    "cuda_products": products.tolist(),
                    "absolute_input_sum": math.fsum(abs(x) for x in terms),
                }
            )
        boundary.append({"name": snap["name"], "control_tick": tick, "posts": per_post})
    for mask in all_masks[1:]:
        exact_bits(mask, all_masks[0], "boundary output mask")
    rows = []
    edge_ids = []
    pre_ids = []
    post_ids = []
    known = []
    cal_lookup = {
        int(edge): (float(src), float(value))
        for edge, src, value in zip(
            cal["edge_ids"], cal["source_values"], cal["values"]
        )
    }
    for post in POST_IDS:
        start, stop = int(ptr[post]), int(ptr[post + 1])
        for edge in range(start, stop):
            pre = int(idx[edge])
            edge_ids.append(edge)
            pre_ids.append(pre)
            post_ids.append(post)
            known.append(pre in observed_pos)
            base_weight = float(
                np.float32(
                    np.sign(graph["weights"][edge]) * graph["counts"][edge] * 0.03
                )
            )
            nav_weight = (
                float(np.float32(base_weight * 64))
                if flags.get(edge, {}).get("navigation")
                else base_weight
            )
            cal_pair = cal_lookup.get(edge)
            rows.append(
                {
                    "edge_id": edge,
                    "post_id": post,
                    "post_root": roots[post]["root_id"],
                    "pre_id": pre,
                    "pre_root": roots[pre]["root_id"],
                    "pre_type": roots[pre].get("cell_type"),
                    "contacts": int(graph["counts"][edge]),
                    "graph_weight_mV": float(graph["weights"][edge]),
                    "rate_base_weight": base_weight,
                    "navigation_weight": nav_weight,
                    "calibration_source_weight": cal_pair[0] if cal_pair else None,
                    "calibration_replacement_weight": cal_pair[1] if cal_pair else None,
                    "navigation_scaled": bool(flags.get(edge, {}).get("navigation")),
                    "calibrated": bool(flags.get(edge, {}).get("calibration")),
                    "individual_fit": bool(flags.get(edge, {}).get("individual_fit")),
                    "actual_weight": float(weight[edge]),
                    "observed_pre": pre in observed_pos,
                }
            )
    edge_ids = np.asarray(edge_ids, np.int64)
    pre_ids = np.asarray(pre_ids, np.int64)
    post_ids = np.asarray(post_ids, np.int64)
    known = np.asarray(known, bool)
    observed_series = np.concatenate((anchor["rate"][None], signals["rate"][:600]))
    calculated_known, edge_terms, edge_cuda = partial_terms(
        observed_series, observed, edge_ids, pre_ids, weight, all_masks[0]
    )
    require(np.array_equal(known, calculated_known), "known incoming mask")
    known_positive = np.zeros((601, 5), np.float64)
    known_negative = np.zeros((601, 5), np.float64)
    known_total = np.zeros((601, 5), np.float64)
    residual = np.zeros((601, 5), np.float64)
    for p, post in enumerate(POST_IDS):
        terms = edge_terms[:, known & (post_ids == post)]
        known_positive[:, p] = np.where(terms > 0, terms, 0).sum(axis=1)
        known_negative[:, p] = np.where(terms < 0, terms, 0).sum(axis=1)
        known_total[:, p] = terms.sum(axis=1)
        residual[:, p] = series["recurrent_current"][:, p] - known_total[:, p]
    series.update(
        known_positive=known_positive,
        known_negative=known_negative,
        known_total=known_total,
        residual=residual,
    )
    for key in tuple(series):
        series[key + "_change"] = np.diff(series[key], axis=0, prepend=np.nan)
    coverage = boundary_coverage(boundary, edge_ids, post_ids, known)
    features = {"edge_contribution": edge_terms, **series}
    arrays = {
        "control_time_s": np.arange(601, dtype=np.float64) * 0.005,
        "post_ids": np.array(POST_IDS, np.int64),
        "edge_ids": edge_ids,
        "edge_pre_ids": pre_ids,
        "edge_post_ids": post_ids,
        "edge_observed_pre": known,
        "edge_contribution": edge_terms,
        "edge_cuda_product": edge_cuda,
        "window_labels": aligned["window_labels"],
        "condition_native_count": aligned["condition_native_count"],
        "native_count_rm": aligned["native_count"][:, 4, :],
        "native_negative_magnitude_sum_bw_rm": aligned[
            "native_negative_magnitude_sum_bw"
        ][:, 4],
        **series,
    }
    summaries = {}
    for key, values in features.items():
        means = []
        counts = []
        mag_means = []
        mag_counts = []
        time_means = []
        time_counts = []
        for window in windows.values():
            result = conditioned(
                values,
                window,
                aligned["native_count"],
                aligned["native_negative_magnitude_sum_bw"],
            )
            a, b, c, d, e, f = result
            means.append(a)
            counts.append(b)
            mag_means.append(c)
            mag_counts.append(d)
            time_means.append(e)
            time_counts.append(f)
        arrays["condition_mean_" + key] = np.asarray(means)
        arrays["condition_valid_count_" + key] = np.asarray(counts)
        arrays["magnitude_mean_" + key] = np.asarray(mag_means)
        arrays["magnitude_weight_" + key] = np.asarray(mag_counts)
        arrays["time_mean_" + key] = np.asarray(time_means)
        arrays["time_valid_count_" + key] = np.asarray(time_counts)
        summaries[key] = {
            "feature_count": values.shape[1],
            "zero_weight_cells": int(np.count_nonzero(np.asarray(counts) == 0)),
        }
    # Compact per-event counterexamples retain all 11, including excluded events.
    event_table = []
    for event in ALL_EVENTS:
        entry = {"index": event, "selected": event in SELECTED, "bands": {}}
        for band in BANDS:
            wi = list(windows).index(f"event{event}:{band}")
            entry["bands"][band] = {
                "rm_negative_count": int(aligned["condition_native_count"][wi, 4, 0]),
                "rm_zero_count": int(aligned["condition_native_count"][wi, 4, 1]),
                "rm_positive_count": int(aligned["condition_native_count"][wi, 4, 2]),
                "mn_rate_by_sign": [
                    [json_number(x) for x in row]
                    for row in arrays["condition_mean_rate"][wi]
                ],
                "known_positive_by_sign": [
                    [json_number(x) for x in row]
                    for row in arrays["condition_mean_known_positive"][wi]
                ],
                "known_negative_by_sign": [
                    [json_number(x) for x in row]
                    for row in arrays["condition_mean_known_negative"][wi]
                ],
                "residual_by_sign": [
                    [json_number(x) for x in row]
                    for row in arrays["condition_mean_residual"][wi]
                ],
            }
        event_table.append(entry)
    summary = {
        "schema": "flylab.m3.e187.mn-current.analysis.v1",
        "identity": identity,
        "prefix_reuse": prefix,
        "post_ids": list(POST_IDS),
        "post_roles": metadata["post_roles"],
        "selected_events": list(SELECTED),
        "all_events": list(ALL_EVENTS),
        "bands": BANDS,
        "endpoint_semantics": "current evaluates saved endpoint rate; command row i applies to (i-1,i]",
        "change_semantics": "*_change is a backward difference per 5ms control, not per second; anchor row0 is missing",
        "partial_observation": "residual includes unobserved inputs and floating-point summation difference",
        "mask_evidence_scope": "five exact full-state boundary masks; E150 main source has no set_cuts, cut-restart uses a separate restored session, and step does not write output_mask directly; no unrecorded intermediate full-state masks are inferred",
        "boundary": boundary,
        "edge_inventory": rows,
        "coverage": coverage,
        "feature_inventory": summaries,
        "event_counterexamples": event_table,
        "validity": "saved-data descriptive attribution only; no edge selected or intervention",
        "F1_F8_complete": False,
        "F6_exposed": False,
        "candidate_adopted": False,
        "threshold_used": threshold.tolist(),
    }
    return summary, arrays


def write_json_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def execute(freeze_path, review_path, output):
    began = time.perf_counter()
    require(output.resolve() == safe_path(OUTPUT).resolve(), "fixed output required")
    require(
        not output.exists() and output.parent.is_dir(),
        "output exists or parent missing",
    )
    require(
        shutil.disk_usage(output.parent).free >= PARAMETERS["minimum_free_bytes"],
        "disk reserve",
    )
    output.mkdir(exist_ok=False)
    terminal = {
        "schema": "flylab.m3.e187.mn-current.terminal.v1",
        "status": "failed",
        "attempted": True,
        "original_analysis_runs": 0,
        "model_constructions": 0,
        "neural_steps": 0,
        "physics_steps": 0,
        "gpu_count": 0,
        "inherited_cumulative_wall_s": PARAMETERS["inherited_cumulative_wall_s"],
    }
    try:
        metadata, freeze_sha = gate(freeze_path, review_path, output)
        terminal.update(
            freeze_sha256=freeze_sha, source_review_sha256=digest(review_path)
        )
        terminal["original_analysis_runs"] = 1
        summary, arrays = analyze(metadata)
        require(
            time.perf_counter() - began <= PARAMETERS["internal_wall_s"],
            "internal wall budget",
        )
        write_json_new(output / "analysis.json", summary)
        np.savez_compressed(output / "aligned.npz", **arrays)
        require(
            time.perf_counter() - began <= PARAMETERS["internal_wall_s"],
            "internal wall budget",
        )
        size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        require(size <= PARAMETERS["maximum_output_bytes"], "output size budget")
        terminal.update(status="complete", output_bytes=size, array_count=len(arrays))
    except Exception as exc:
        terminal["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        terminal["elapsed_wall_s"] = time.perf_counter() - began
        write_json_new(output / "terminal.json", terminal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--source-review", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    execute(args.freeze, args.source_review, args.out)


if __name__ == "__main__":
    main()
