"""E331 saved-only CPU residual analysis; no neural loader or body execution.

Root creates/pins make_contract(), qualifies this implementation, then invokes
--contract JSON --contract-sha256 SHA --output FRESH_DIRECTORY under its 120s
supervisor. All bytes are gated before numerical NPZ access. O/F are divided
once by 50; saved gtheta already contains that factor. Closure is algebra only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
ORIGINAL = BASE + "/e330-temporal-direction"
SCHEMA = "flylab.m3.e331.saved-residual.contract.v1"
RESULT_SCHEMA = "flylab.m3.e331.saved-residual.result.v1"
OWNED = ("tools/analyze_c_banc_m3_temporal_residual.py", "tests/test_c_banc_temporal_residual.py")
STARTS = tuple(range(100, 300, 4))
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
PAIRS = tuple((a, b) for a in range(6) for b in range(a + 1, 6))
TERMS = ("training_loss", "successor_loss", "coordination_loss", "lr_coordination_loss")
LIMITS = {"supervisor_wall_s": 120, "internal_wall_s": 110, "calculation_rehash_deadline_s": 100,
          "finalization_reserve_s": 10, "host_bytes": 2 << 30, "output_bytes": 128 << 20,
          "minimum_free_disk_bytes": 1 << 30, "cpu_threads": 1}
EVIDENCE = {"freeze": BASE + "/e330-temporal-direction-freeze.json",
            "review": BASE + "/e330-temporal-direction-source-review.json",
            "audit": BASE + "/parent-e330-original-raw-audit.json",
            "audit_source": BASE + "/parent-e330-original-raw-audit.py",
            "supervision": BASE + "/parent-e330-original-supervision.json",
            "decision": BASE + "/parent-e330-final-decision.json",
            "result": ORIGINAL + "/result.json", "ledger": ORIGINAL + "/advance-ledger.json",
            "manifest": ORIGINAL + "/output-manifest.json", "producer": ORIGINAL + "/source-manifest.json",
            "receipts": ORIGINAL + "/receipts.json",
            "design": "docs/BANC_M3_E331_SAVED_TEMPORAL_RESIDUAL_DESIGN_20261001.md",
            "astra": BASE + "/parent-e331-astra-design-review.md"}
INTERPRETATION = {
    "O_minus_F": "all neutral nearest-index fixing effects, including adjacent segment movement",
    "F_minus_Q": "finite objective response within fixed indices, including clamp, target knots and cosine",
    "Q_minus_T": "neural/decoder finite response, surrogate/rounding VJP and unknown CPU/GPU objective cotangent/branch/arithmetic differences",
    "Q_leg": "command-coordinate contribution including cross-leg coordination dependence; not biological responsibility",
    "leg_T_available": False, "leg_Q_minus_T_available": False,
    "GPU_gq_or_selected_m_saved": False, "closure_proves_correctness": False,
    "endpoint_change_proves_path_discontinuity": False,
    "tie_endpoint_knot_derivative": "selected program cotangent; unique mathematical derivative not claimed"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), object_pairs_hook=unique, parse_constant=invalid)


def safe(name, root=ROOT):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path outside repository")
    return path


def pin_map(pins, root=ROOT, timely=lambda: None):
    if not isinstance(pins, dict) or not pins:
        raise ValueError("nonempty pin map required")
    for name, digest in pins.items():
        timely()
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or sha256(safe(name, root)) != digest):
            raise ValueError("SHA256 mismatch: " + name)


def make_contract(*, root=ROOT):
    """Passive hash/JSON helper only; never opens original NPZ payloads."""
    freeze = read(safe(EVIDENCE["freeze"], root))
    manifest = read(safe(EVIDENCE["manifest"], root))
    sources = {**freeze["source_sha256"], **{name: sha256(safe(name, root)) for name in OWNED}}
    inputs = dict(freeze["input_sha256"])
    inputs.update({ORIGINAL + "/" + name: row["sha256"] for name, row in manifest["files"].items()})
    inputs.update({name: sha256(safe(name, root)) for name in EVIDENCE.values()})
    inputs.update(read(safe(EVIDENCE["decision"], root))["evidence_sha256"])
    return {"schema": SCHEMA, "evidence": EVIDENCE.copy(), "source_sha256": sources,
            "input_sha256": inputs, "limits": LIMITS.copy(), "window_starts": list(STARTS),
            "original_calls": 0, "candidate_adopted": False}


def gate(contract_path, contract_sha256, output, *, root=ROOT, timely=lambda: None):
    """Hash/source/proof gate must finish before load_npz or Torch import."""
    if output.exists() or output.resolve() == root.resolve():
        raise ValueError("fresh output directory required")
    if sha256(contract_path) != contract_sha256:
        raise ValueError("contract SHA256 mismatch")
    contract = read(contract_path)
    if (set(contract) != {"schema", "evidence", "source_sha256", "input_sha256", "limits", "window_starts",
                         "original_calls", "candidate_adopted"}
            or contract["schema"] != SCHEMA or contract["evidence"] != EVIDENCE
            or contract["limits"] != LIMITS or contract["window_starts"] != list(STARTS)
            or contract["original_calls"] != 0 or contract["candidate_adopted"] is not False):
        raise ValueError("exact E331 saved contract required")
    sources, inputs = contract["source_sha256"], contract["input_sha256"]
    pin_map(sources, root, timely)
    pin_map(inputs, root, timely)
    evidence = {role: read(safe(name, root)) for role, name in EVIDENCE.items()
                if role not in ("audit_source", "design", "astra")}
    freeze, review, producer, manifest = (evidence[k] for k in ("freeze", "review", "producer", "manifest"))
    if (freeze.get("schema") != "flylab.m3.e330.temporal-direction.freeze.v1"
            or review.get("schema") != "flylab.m3.e330.temporal-direction.review.v1"
            or review.get("approved") is not True or review.get("blocking_defects") != []
            or producer.get("schema") != "flylab.m3.e330.source-manifest.v1"
            or manifest.get("schema") != "flylab.m3.e330.output-manifest.v1"
            or producer.get("source_sha256") != freeze["source_sha256"]
            or producer.get("input_sha256") != freeze["input_sha256"]
            or review.get("source_sha256") != freeze["source_sha256"]
            or review.get("freeze_sha256") != inputs[EVIDENCE["freeze"]]
            or producer.get("freeze_sha256") != inputs[EVIDENCE["freeze"]]
            or producer.get("review_sha256") != inputs[EVIDENCE["review"]]):
        raise ValueError("E330 original source/input/review linkage differs")
    expected_inputs = dict(freeze["input_sha256"])
    expected_inputs.update({ORIGINAL + "/" + name: row["sha256"] for name, row in manifest["files"].items()})
    expected_inputs.update({name: inputs.get(name) for name in EVIDENCE.values()})
    expected_inputs.update(evidence["decision"]["evidence_sha256"])
    if inputs != expected_inputs or sources != {**freeze["source_sha256"], **{name: sources.get(name) for name in OWNED}}:
        raise ValueError("exact source/input closure required")
    for name in sources.keys() & inputs.keys():
        if sources[name] != inputs[name]:
            raise ValueError("contradictory source/input SHA")
    for name, row in manifest["files"].items():
        if row != {"sha256": inputs[ORIGINAL + "/" + name], "bytes": safe(ORIGINAL + "/" + name, root).stat().st_size}:
            raise ValueError("immutable output manifest descriptor differs")
    audit, result, ledger, supervision, decision = (evidence[k] for k in ("audit", "result", "ledger", "supervision", "decision"))
    if (audit.get("schema") != "flylab.m3.e330.root-original-raw-audit.v1"
            or audit.get("passed") is not True or audit.get("diagnostic_run_valid") is not True
            or audit.get("verdict") != "refuted" or len(audit.get("checks", {})) != 932
            or not all(v is True for v in audit["checks"].values())):
        raise ValueError("completed 932-check root audit required")
    for key, role in (("result_sha256", "result"), ("advance_ledger_sha256", "ledger"),
                      ("output_manifest_sha256", "manifest"), ("supervision_sha256", "supervision"),
                      ("audit_source_sha256", "audit_source")):
        if audit.get(key) != inputs[EVIDENCE[role]]:
            raise ValueError("root audit hash linkage differs: " + key)
    if (result.get("schema") != "flylab.m3.e330.temporal-direction.result.v1"
            or result.get("run_valid") is not True or result.get("local_direction") != "refuted"
            or result.get("candidate_adopted") is not False or result.get("errors") != []
            or result.get("continuation", {}).get("classification") != "not_tested"
            or result["continuation"].get("requested") is not False
            or result.get("body_not_executed") is not True or ledger.get("errors") != []
            or evidence["receipts"].get("schema") != "flylab.m3.e330.receipts.v1"
            or result.get("provenance", {}).get("freeze_sha256") != inputs[EVIDENCE["freeze"]]
            or result["provenance"].get("review_sha256") != inputs[EVIDENCE["review"]]):
        raise ValueError("E330 valid refuted result/receipt/ledger required")
    if (supervision.get("success") is not True or len(supervision.get("attempts", [])) != 1
            or supervision["attempts"][0].get("exit_code") != 0
            or supervision["attempts"][0].get("killed") is not False
            or not supervision.get("owned_pids") or any(psutil.pid_exists(pid) for pid in supervision["owned_pids"])
            or decision.get("root_raw_audit_pass") is not True
            or decision.get("local_direction") != "refuted"
            or decision.get("cumulative_after_s") != 9200.673308899522
            or decision.get("original_reserved_s") != 0):
        raise ValueError("completed supervisor/unchanged accounting required")
    for name, digest in decision.get("evidence_sha256", {}).items():
        if inputs.get(name) != digest:
            raise ValueError("final decision SHA linkage differs")
    paths = freeze["paths"]
    saved = read(safe(paths["e329_result"], root))
    prior_audit = read(safe(paths["e329_audit"], root))
    qualification = read(safe(paths["e329_qualification"], root))
    if (saved.get("schema") != "flylab.m3.e329.temporal.saved-characterization.v1"
            or saved.get("e330_input_qualification_pass") is not True
            or saved.get("reference_qualification", {}).get("pass") is not True
            or prior_audit.get("schema") != "flylab.m3.e329.root-saved-raw-audit.v1"
            or prior_audit.get("passed") is not True
            or prior_audit.get("result_sha256") != inputs[paths["e329_result"]]
            or prior_audit.get("contract_sha256") != inputs[paths["e329_contract"]]
            or qualification.get("approved") is not True
            or qualification.get("blocking_defects") != []
            or qualification.get("e330_input_qualification_pass") is not True):
        raise ValueError("E329 approved saved reference/qualification linkage differs")
    reference = freeze["temporal_reference"]
    if (reference.get("curves") != saved["reference_curves_metadata"]
            or reference.get("normalization") != saved["successor_normalization_metadata"]
            or reference.get("normalization_f64") != saved["successor_normalization_mse_rad"]
            or reference.get("frequency_hz") != 2.382760007990865
            or reference.get("control_dt_s") != .005
            or reference.get("coordination_coefficient") != 1 or reference.get("unordered_pairs") != 15):
        raise ValueError("E329 frozen objective/reference constants differ")
    for role, shape in (("curves", [6, 128, 7]), ("normalization", [6])):
        metadata = reference[role]
        if (metadata.get("shape") != shape or metadata.get("dtype") != "float64"
                or metadata.get("numpy_dtype") != "<f8" or metadata.get("layout") != "C"
                or metadata.get("leg_order") != list(LEGS)):
            raise ValueError("E329 reference byte metadata differs")
    protected = {contract_path.resolve(), *(safe(name, root) for name in sources.keys() | inputs.keys())}
    if any(path.is_relative_to(output.resolve()) for path in protected):
        raise ValueError("output contains protected input/source")
    return contract, evidence


def load_npz(path, keys):
    """Key-only loading; rate/adaptation/CSR arrays are never requested."""
    with np.load(path, allow_pickle=False) as archive:
        if not set(keys).issubset(archive.files):
            raise ValueError("missing required NPZ keys: " + str(path))
        return {key: archive[key] for key in keys}


def f64(value, shape, label):
    if not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError("finite float64 " + label + " shape " + str(shape) + " required")
    return value


def numpy_project(curves, q, fixed_segments=None):
    """Independent NumPy closed 7D polyline projection, first argmin tie."""
    f64(curves, (6, 128, 7), "curves")
    f64(q, (len(q), 42), "q")
    edge = np.roll(curves, -1, axis=1) - curves
    norm = np.sum(edge * edge, axis=-1)
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise ValueError("positive finite reference edges required")
    delta = q.reshape(-1, 6, 7)[:, :, None, :] - curves[None]
    raw = np.sum(delta * edge[None], axis=-1) / norm[None]
    u = np.clip(raw, 0, 1)
    nearest = curves[None] + u[..., None] * edge[None]
    distance = np.sum((q.reshape(-1, 6, 7)[:, :, None] - nearest) ** 2, axis=-1)
    if not np.isfinite(raw).all() or not np.isfinite(distance).all():
        raise ValueError("nonfinite projection")
    segment = distance.argmin(axis=-1) if fixed_segments is None else np.asarray(fixed_segments)
    if segment.shape != (len(q), 6) or segment.dtype.kind not in "iu" or np.any((segment < 0) | (segment >= 128)):
        raise ValueError("fixed segment index shape/range required")
    rows, legs = np.arange(len(q))[:, None], np.arange(6)[None]
    phase_all = np.remainder((np.arange(128)[None, None] + u) / 128, 1)
    return {"segment": segment.copy(), "raw_u": raw[rows, legs, segment], "u": u[rows, legs, segment],
            "phase": phase_all[rows, legs, segment], "distance_sq": distance[rows, legs, segment],
            "all_segment_distance_sq": distance, "all_segment_phase": phase_all}


def numpy_terms(curves, normalization, boundary, q, fixed_segments=None):
    """Only indices may be frozen; live q, clamp and target interpolation remain."""
    from tools.analyze_c_banc_m3_temporal_objective import numpy_interpolate
    f64(boundary, (42,), "boundary")
    f64(q, (4, 42), "window q")
    f64(normalization, (6,), "normalization")
    if np.any(normalization <= 0):
        raise ValueError("positive normalization required")
    previous = np.concatenate((boundary[None], q[:3]), axis=0)
    p = numpy_project(curves, previous, None if fixed_segments is None else fixed_segments[0])
    n = numpy_project(curves, q, None if fixed_segments is None else fixed_segments[1])
    target = numpy_interpolate(curves, p["phase"] + 2.382760007990865 * .005)
    successor_leg = np.mean((q - target).reshape(4, 6, 7) ** 2, axis=(0, 2)) / normalization
    pair = np.mean(np.stack([1 - np.cos(2 * np.pi * (n["phase"][:, a] - n["phase"][:, b]))
                             for a, b in PAIRS], axis=-1), axis=0)
    lr = [PAIRS.index(pair) for pair in ((0, 3), (1, 4), (2, 5))]
    successor, coordination = float(successor_leg.mean()), float(pair.mean())
    return {"terms": np.array([successor + coordination, successor, coordination, pair[lr].mean()], np.float64),
            "successor_leg": successor_leg, "coordination_pair": pair, "target": target,
            "previous": p, "next": n}


def geometry(projected):
    segment, phase = projected["segment"], projected["phase"]
    gap = np.abs(np.arange(128)[None, None] - segment[..., None])
    gap = np.minimum(gap, 128 - gap)
    phase_gap = np.abs(projected["all_segment_phase"] - phase[..., None])
    phase_gap = np.minimum(phase_gap, 1 - phase_gap)
    nonlocal_mask = (gap > 1) & (phase_gap > 2 / 128)
    competitor = np.min(np.where(nonlocal_mask, projected["all_segment_distance_sq"]
                                - projected["distance_sq"][..., None], np.inf), axis=-1)
    all_gap = projected["all_segment_distance_sq"] - projected["distance_sq"][..., None]
    selected = np.arange(128)[None, None] == segment[..., None]
    tie_gap = np.min(np.where(selected, np.inf, all_gap), axis=-1)
    raw = projected["raw_u"]
    regime = np.select((raw < 0, raw == 0, raw == 1, raw > 1), (-2, -1, 1, 2), default=0).astype(np.int8)
    coordinate = np.remainder(phase + 2.382760007990865 * .005, 1) * 128
    return {**{k: projected[k] for k in ("segment", "phase", "raw_u", "u", "distance_sq")},
            "nonlocal_gap": competitor, "tie_gap": tie_gap, "nonlocal_tie": np.isfinite(competitor) & (competitor <= 1e-12),
            "any_tie": tie_gap <= 1e-12, "clamp_regime": regime,
            "target_knot": coordinate == np.floor(coordinate)}


def branch_changes(neutral, arm):
    gap = np.abs(neutral["segment"] - arm["segment"])
    gap = np.minimum(gap, 128 - gap)
    phase_gap = np.abs(neutral["phase"] - arm["phase"])
    circular = np.minimum(phase_gap, 1 - phase_gap)
    return {"cyclic_segment_distance": gap, "circular_phase_difference": circular,
            "nonlocal_endpoint_change": (gap > 1) & (circular > 2 / 128),
            "adjacent_change": gap == 1, "phase_wrap": phase_gap > .5,
            "clamp_regime_change": neutral["clamp_regime"] != arm["clamp_regime"]}


def residuals(o, q, t, f):
    if not np.isfinite([o, q, t, f]).all():
        raise ValueError("finite residual inputs required")
    row = {"O": float(o), "Q": float(q), "T": float(t), "F": float(f), "O_minus_T": float(o - t),
           "O_minus_Q": float(o - q), "Q_minus_T": float(q - t),
           "O_minus_F": float(o - f), "F_minus_Q": float(f - q)}
    two = (o - q) + (q - t)
    three = (o - f) + (f - q) + (q - t)
    row.update(two_term_closure_error=float(abs(two - (o - t))), three_term_closure_error=float(abs(three - (o - t))))
    if max(row["two_term_closure_error"], row["three_term_closure_error"]) > 1e-12:
        raise ValueError("residual assembly closure failed")
    return row


def analyze_window(objective, boundary, arms, gtheta, delta_theta, *, start_control=100,
                   expected_terms=None, gpu_observed=None):
    import torch
    f64(boundary, (42,), "boundary")
    for label in ("neutral", "plus", "minus"):
        f64(arms[label], (4, 42), label)
    f64(gtheta, delta_theta.shape, "gtheta")
    f64(delta_theta, gtheta.shape, "delta theta")
    curves = objective.curves.detach().cpu().numpy()
    normalization = objective.successor_normalization.detach().cpu().numpy()
    independent, raw, actual, geometry_rows = {}, {"boundary": boundary.copy(), "delta_theta": delta_theta.copy(), "gtheta": gtheta.copy()}, {}, {}
    for label in ("neutral", "plus", "minus"):
        q = torch.tensor(arms[label], dtype=torch.float64, device="cpu", requires_grad=label == "neutral")
        previous = torch.cat((torch.tensor(boundary[None], dtype=torch.float64), q[:3]))
        values = objective.evaluate_pairs(previous, q)
        actual[label] = np.array([float(values[key].detach()) for key in TERMS], np.float64)
        independent[label] = numpy_terms(curves, normalization, boundary, arms[label])
        if np.max(np.abs(actual[label] - independent[label]["terms"])) > 1e-10:
            raise ValueError("independent NumPy loss mismatch")
        if expected_terms is not None and np.max(np.abs(actual[label] - expected_terms[label])) > 1e-10:
            raise ValueError("saved E330 loss mismatch")
        if label == "neutral":
            gq = torch.autograd.grad(values["training_loss"] / 50, q)[0].detach().numpy()
            f64(gq, (4, 42), "gq")
        raw[label + "_q"] = arms[label].copy()
        raw[label + "_terms"] = actual[label]
        for side in ("previous", "next"):
            geom = geometry(independent[label][side])
            geometry_rows[label + "_" + side] = {
                key: ([[float(x) if np.isfinite(x) else None for x in row] for row in value]
                      if key in ("nonlocal_gap", "tie_gap") else value.tolist()) for key, value in geom.items()}
            for key, value in geom.items():
                if key in ("nonlocal_gap", "tie_gap"):
                    raw[f"{label}_{side}_{key}_available"] = np.isfinite(value)
                    value = np.where(np.isfinite(value), value, 0)
                raw[f"{label}_{side}_{key}"] = value
        raw[label + "_target"] = independent[label]["target"]
    fixed = (independent["neutral"]["previous"]["segment"], independent["neutral"]["next"]["segment"])
    counter = {label: numpy_terms(curves, normalization, boundary, arms[label], fixed) for label in arms}
    if np.max(np.abs(counter["neutral"]["terms"] - actual["neutral"])) > 1e-10:
        raise ValueError("branch-fixed neutral loss mismatch")
    dq = arms["plus"] - arms["minus"]
    o = (actual["plus"][0] - actual["minus"][0]) / 50
    q = np.sum(gq * dq, dtype=np.float64)
    t = np.sum(gtheta * delta_theta, dtype=np.float64)
    f = (counter["plus"]["terms"][0] - counter["minus"]["terms"][0]) / 50
    result = {"start_control": start_control, "physics_time_s": start_control * .005,
              **residuals(o, q, t, f), "O_gpu_original": gpu_observed,
              "O_cpu_minus_gpu": None if gpu_observed is None else float(o - gpu_observed),
              "Q_coordinate_leg": np.sum((gq * dq).reshape(4, 6, 7), axis=(0, 2)).tolist()}
    for component, key in (("successor_leg", "successor_leg"), ("coordination_pair", "coordination_pair")):
        result[component + "_O"] = ((independent["plus"][key] - independent["minus"][key]) / 50).tolist()
        result[component + "_F"] = ((counter["plus"][key] - counter["minus"][key]) / 50).tolist()
    branch = {}
    for label in ("plus", "minus"):
        for side in ("previous", "next"):
            changes = branch_changes(geometry(independent["neutral"][side]), geometry(independent[label][side]))
            branch[label + "_" + side] = {key: int(np.count_nonzero(value)) for key, value in changes.items()
                                          if value.dtype == bool}
            for key, value in changes.items():
                raw[f"{label}_{side}_{key}"] = value
    result["branch_changes"] = branch
    result["geometry"] = geometry_rows
    raw.update(neutral_gq=gq, delta_q=dq)
    for label in counter:
        for key in ("terms", "successor_leg", "coordination_pair"):
            raw[label + "_fixed_" + key] = counter[label][key]
        for key in ("successor_leg", "coordination_pair"):
            raw[label + "_" + key] = independent[label][key]
    for key in ("O", "Q", "T", "F", "O_minus_T", "O_minus_Q", "Q_minus_T", "O_minus_F", "F_minus_Q"):
        raw[key] = np.array(result[key], np.float64)
    return result, raw


def aggregate(windows):
    denominator = sum(abs(row["O_minus_T"]) for row in windows)
    values = {}
    for key in ("O", "Q", "T", "F", "O_minus_T", "O_minus_Q", "Q_minus_T", "O_minus_F", "F_minus_Q"):
        absolute = sum(abs(row[key]) for row in windows)
        top = max(windows, key=lambda row: abs(row[key])) if windows else None
        values[key] = {"signed_sum": sum(row[key] for row in windows), "sum_abs": absolute,
                       "share_of_absolute_total_residual": None if denominator == 0 else absolute / denominator,
                       "max_abs": None if top is None else abs(top[key]),
                       "max_start_control": None if top is None else top["start_control"],
                       "max_physics_time_s": None if top is None else top["physics_time_s"]}
    return {"terms": values, "absolute_shares_add_to_one_claimed": False,
            "ranked_residual_controls": [row["start_control"] for row in sorted(windows, key=lambda x: abs(x["O_minus_T"]), reverse=True)]}


def write_json(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def _replace_terminal_json(path, value):
    """Replace only a terminal file in this run's fresh, owned output folder.

    Separate from the injected publication writer and the emergency journal:
    either can fail after committing a stale successful result. A failed temp
    write is retained, and replacement occurs only after a complete JSON write.
    """
    serialized = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(mode="x", encoding="utf8", dir=path.parent,
                                     prefix=path.name + ".repair-", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(serialized)
    os.replace(temporary, path)


def run(contract_path, contract_sha256, output, *, root=ROOT, writer=write_json,
        saver=np.savez_compressed, clock=time.perf_counter, gate_fn=gate, compute_fn=None):
    """Retain completed raw, first error, partial windows and publication errors."""
    began = clock()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("fresh output required")
    errors, rows, files = [], [], {}
    ledger = {"schema": "flylab.m3.e331.saved-residual.ledger.v1", "limits": LIMITS.copy(), "attempts": [],
              "original_calls": 0, "body_calls": 0, "CUDA_calls": 0, "model_calls": 0, "optimizer_steps": 0}
    result = {"schema": RESULT_SCHEMA, "analysis_valid": False, "candidate_adopted": False,
              "F1_F8_complete": False, "biological_gait_proven": False, "E330_local_direction": "refuted",
              "continuation": "not_tested", "interpretation": INTERPRETATION, "windows": rows,
              "original_calls": 0, "body_calls": 0, "CUDA_calls": 0, "whole_goal_status": "active",
              "original_cumulative_s": 9200.673308899522, "original_allocation_limit_s": 9360,
              "original_reserved_s": 0, "contract_sha256": contract_sha256}
    def timely():
        elapsed = clock() - began
        rss = psutil.Process().memory_info().rss
        ledger["max_host_rss_bytes"] = max(ledger.get("max_host_rss_bytes", 0), rss)
        if elapsed > 100 or rss > LIMITS["host_bytes"]:
            raise RuntimeError("calculation/rehash deadline or host limit exceeded")
    def error(stage, exception):
        errors.append({"stage": stage, "type": type(exception).__name__, "message": str(exception), "elapsed_s": clock() - began})
    def save(name, raw):
        timely()
        attempt = {"file": name, "requested": True, "returned": False}
        ledger["attempts"].append(attempt)
        saver(output / name, **raw)
        size = (output / name).stat().st_size
        if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > LIMITS["output_bytes"]:
            raise RuntimeError("output limit exceeded")
        files[name] = {"sha256": sha256(output / name), "bytes": size}
        attempt.update(returned=True, bytes=size)
    output.mkdir(parents=True)
    try:
        if shutil.disk_usage(output).free < LIMITS["minimum_free_disk_bytes"]:
            raise RuntimeError("free disk below 1GiB")
        # gate requires the fresh output to be absent; directory exists only to
        # retain failures, so gate receives the same absent path via removal.
        output.rmdir()
        try:
            contract, evidence = gate_fn(Path(contract_path), contract_sha256, output, root=root, timely=timely)
        finally:
            output.mkdir(exist_ok=True)
        timely()
        result["source_sha256"] = contract["source_sha256"]
        result["input_sha256"] = contract["input_sha256"]
        result["original_result_sha256"] = contract["input_sha256"][EVIDENCE["result"]]
        result["original_ledger_sha256"] = contract["input_sha256"][EVIDENCE["ledger"]]
        if compute_fn is not None:
            compute_fn(contract, evidence, rows, save, timely)
        else:
            compute_saved(contract, evidence, rows, save, timely, root=root)
        pin_map(contract["source_sha256"], root, timely)
        pin_map(contract["input_sha256"], root, timely)
        if sha256(contract_path) != contract_sha256:
            raise ValueError("post-computation contract SHA mismatch")
        timely()
        if [row["start_control"] for row in rows] != list(STARTS):
            raise ValueError("all 50 completed windows required")
        result["aggregate"] = aggregate(rows)
        result["analysis_valid"] = True
        result["calculation_and_rehash_elapsed_s"] = clock() - began
    except Exception as exception:  # noqa: BLE001
        error("analysis", exception)
    def terminal_state():
        elapsed = clock() - began
        first = errors[0] if errors else None
        common = {"errors": errors, "first_error": first, "completed_windows": len(rows),
                  "analysis_valid": result["analysis_valid"], "elapsed_s": elapsed,
                  "measured_boundary": "before_terminal_metadata_commit"}
        result.update(common)
        ledger.update(common)
    terminal_state()
    try:
        writer(output / "output-manifest.json", {"schema": "flylab.m3.e331.saved-residual.manifest.v1", "files": files,
                                                 "mutable_metadata_excluded": ["result.json", "advance-ledger.json", "publication-failure.json"]})
        writer(output / "advance-ledger.json", ledger)
        writer(output / "result.json", result)
        if clock() - began > 110 or sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > LIMITS["output_bytes"]:
            raise RuntimeError("internal finalization/output limit exceeded")
    except Exception as exception:  # noqa: BLE001
        error("publication", exception)
        result["analysis_valid"] = False
        terminal_state()
        try:
            write_json(output / "publication-failure.json", {"schema": "flylab.m3.e331.publication-failure.v1",
                       "analysis_valid": False, "first_error": errors[0], "errors": errors,
                       "completed_windows": len(rows), "elapsed_s": result["elapsed_s"]})
        except Exception as secondary:  # noqa: BLE001
            error("emergency_publication", secondary)
        terminal_state()
        repaired = []
        for name, value in (("advance-ledger.json", ledger), ("result.json", result)):
            try:
                _replace_terminal_json(output / name, value)
                repaired.append((name, value))
            except Exception as secondary:  # noqa: BLE001
                error("terminal_repair_" + name, secondary)
                terminal_state()
        # If one terminal sink failed, bring any working sink up to date with
        # those secondary errors. Never let recovery replace the first failure
        # at the API boundary, even when every output sink is unavailable.
        if any(row["stage"].startswith("terminal_repair_") for row in errors):
            terminal_state()
            for name, value in repaired:
                try:
                    _replace_terminal_json(output / name, value)
                except Exception as secondary:  # noqa: BLE001
                    error("terminal_refresh_" + name, secondary)
            terminal_state()
    return result


def compute_saved(contract, evidence, rows, save, timely, *, root=ROOT):
    """Called only after gate; original NPZ numbers are exclusively root-run."""
    import torch

    from tools.analyze_c_banc_m3_temporal_objective import curve128
    from tools.banc_m3_phase_free_temporal import PhaseFreeTemporalObjective
    torch.set_num_threads(1)
    freeze = evidence["freeze"]
    paths = freeze["paths"]
    reference = freeze["temporal_reference"]
    curves = curve128(load_npz(safe(paths["e85_reference"], root), ("time_s", "phase", "command")))
    if bytes_sha(curves) != reference["curves"]["c_order_sha256"]:
        raise ValueError("approved reference curve bytes differ")
    startup = load_npz(safe(paths["e103_startup"], root), ("target_command", "target_tendon"))
    objective = PhaseFreeTemporalObjective(curves, startup["target_command"], startup["target_tendon"], device="cpu")
    norm = np.array(reference["normalization_f64"], np.float64)
    f64(norm, (6,), "approved normalization")
    computed = objective.successor_normalization.detach().numpy()
    if not np.allclose(computed, norm, rtol=1e-10, atol=1e-12) or bytes_sha(norm) != reference["normalization"]["c_order_sha256"]:
        raise ValueError("CPU constructor or approved normalization differs")
    objective.successor_normalization = torch.tensor(norm.copy(), dtype=torch.float64)
    if bytes_sha(objective.successor_normalization.numpy()) != reference["normalization"]["c_order_sha256"]:
        raise ValueError("actual loss denominator bytes differ")
    save("reference.npz", {"curves": curves, "normalization": norm, "constructor_normalization": computed})
    raw = load_npz(safe(paths["e318_raw"], root), ("command", "control_tick", "network_tick", "physics_time_s"))
    f64(raw["command"], (301, 42), "E318 q")
    for key, expected in (("control_tick", np.arange(301)), ("network_tick", np.arange(301) * 20)):
        if raw[key].dtype.kind not in "iu" or not np.array_equal(raw[key], expected):
            raise ValueError("E318 clock differs")
    f64(raw["physics_time_s"], (301,), "physics clock")
    if not np.allclose(raw["physics_time_s"], np.arange(301) * .005, rtol=0, atol=2e-8):
        raise ValueError("E318 physics clock differs")
    direction = load_npz(safe(ORIGINAL + "/direction.npz", root), ("window_gradients", "theta_plus", "theta_minus"))
    f64(direction["window_gradients"], (50, 8547), "window gradients")
    for label in ("plus", "minus"):
        theta = f64(direction["theta_" + label], (8547,), "theta " + label)
        if not np.array_equal(theta, theta.astype(np.float32).astype(np.float64)):
            raise ValueError("actual f32 theta required")
        if bytes_sha(theta) != evidence["result"]["native_eligibility"][label]["theta64_sha256"]:
            raise ValueError("theta saved result linkage differs")
    delta = direction["theta_plus"] - direction["theta_minus"]
    original_windows = evidence["result"]["partial"]["windows"]
    if [row["start_control"] for row in original_windows] != list(STARTS):
        raise ValueError("original result window order differs")
    receipts = evidence["receipts"].get("receipts", {})
    for index, c in enumerate(STARTS):
        timely()
        arms, terms = {}, {}
        for label in ("neutral", "plus", "minus"):
            name = f"window-{c}-{label}"
            data = load_npz(safe(ORIGINAL + "/" + name + ".npz", root), ("q", "fixed_original_q_boundary", "loss_terms"))
            f64(data["fixed_original_q_boundary"], (42,), "saved boundary")
            if not np.array_equal(data["fixed_original_q_boundary"], raw["command"][c]):
                raise ValueError("fixed original boundary differs")
            arms[label] = f64(data["q"], (4, 42), name)
            if label == "neutral" and np.max(np.abs(arms[label] - raw["command"][c + 1:c + 5])) > 1e-11:
                raise ValueError("neutral q differs from approved E318 commands")
            terms[label] = f64(data["loss_terms"], (4,), "loss terms")
            original = evidence["result"]["partial"]["runs"][name]
            if original["raw_file"] != name + ".npz" or not np.allclose(terms[label], [original["metrics"][k] for k in TERMS], rtol=0, atol=1e-10):
                raise ValueError("original result/raw terms differ")
            if receipts.get(name) != original["receipt"]:
                raise ValueError("original receipt linkage differs")
        gradient = load_npz(safe(ORIGINAL + f"/window-{c}-neutral-gradient.npz", root), ("gradient",))["gradient"]
        f64(gradient, (8547,), "gradient")
        if not np.array_equal(gradient, direction["window_gradients"][index]):
            raise ValueError("window gradient order/value differs")
        original = original_windows[index]
        row, saved = analyze_window(objective, raw["command"][c], arms, gradient, delta,
                                    start_control=c, expected_terms=terms, gpu_observed=original["observed_contribution"])
        if abs(row["T"] - original["predicted_contribution"]) > 1e-12:
            raise ValueError("T original contribution differs")
        if abs((terms["plus"][0] - terms["minus"][0]) / 50 - original["observed_contribution"]) > 1e-12:
            raise ValueError("O original contribution differs")
        save(f"window-{c}.npz", saved)
        rows.append(row)
        timely()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = run(args.contract, args.contract_sha256, args.output)
    print(json.dumps({"analysis_valid": result["analysis_valid"], "completed_windows": result["completed_windows"],
                      "first_error": result["first_error"]}, allow_nan=False))
    return 0 if result["analysis_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
