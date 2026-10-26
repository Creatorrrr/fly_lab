"""E332 CPU saved-command analysis with passive inherited byte/proof closure.

No Torch import or NPZ payload access occurs before gate(). Old efficacy
cotangents are never read or relabeled. Root alone invokes the saved analysis;
generated CPU tests use synthetic inputs. Analysis validity and scientific input
qualification are separate, and neither claims a candidate or biological gait.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import psutil

from tools import analyze_c_banc_m3_temporal_residual as predecessor
from tools.analyze_c_banc_m3_temporal_residual import (
    _replace_terminal_json,
    bytes_sha,
    f64,
    load_npz,
    pin_map,
    read,
    safe,
    sha256,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
ORIGINAL = BASE + "/e330-temporal-direction"
PREVIOUS = BASE + "/e331-saved-temporal-residual"
SCHEMA = "flylab.m3.e332.smooth-temporal.contract.v1"
RESULT_SCHEMA = "flylab.m3.e332.smooth-temporal.result.v1"
OWNED = ("tools/banc_m3_smooth_temporal.py", "tools/analyze_c_banc_m3_smooth_temporal.py",
         "tests/test_c_banc_smooth_temporal.py")
STARTS = tuple(range(100, 300, 4))
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
PAIRS = tuple((a, b) for a in range(6) for b in range(a + 1, 6))
TERMS = ("training_loss", "successor_loss", "coordination_loss", "lr_coordination_loss")
LIMITS = predecessor.LIMITS.copy()
EVIDENCE = {
    "contract": BASE + "/e331-saved-temporal-residual-contract.json",
    "passive": BASE + "/parent-e331-passive-freeze.json",
    "second": BASE + "/parent-e331-second-source-review.json",
    "audit": BASE + "/parent-e331-saved-raw-audit.json",
    "audit_source": BASE + "/parent-e331-saved-raw-audit.py",
    "supervision": BASE + "/parent-e331-saved-supervision.json",
    "decision": BASE + "/parent-e331-final-decision.json",
    "result": PREVIOUS + "/result.json", "ledger": PREVIOUS + "/advance-ledger.json",
    "manifest": PREVIOUS + "/output-manifest.json",
    "prior_results": "docs/BANC_M3_E331_SAVED_TEMPORAL_RESIDUAL_RESULTS_20261001.md",
    "design": "docs/BANC_M3_E332_SMOOTH_TEMPORAL_DESIGN_20261001.md",
    "astra": BASE + "/parent-e332-astra-design-review.md",
}
FROZEN_PROOF = {
    "contract": "f0b3f78b19454959704c79ca7c7066f5247fe894bda31766fb01c6a548911a54",
    "second": "c48bff4ea0fe67e1a0f7234551c5d87208d6bc6c8393e1e9ae92e70265ca737d",
    "audit": "bf99091b0051f2995005ba79a0e8e4bf5d20ea518e340e8cada685b86e4e438d",
    "audit_source": "aefede05be2329817f01ee3e67297763b6d825ffdc5bb5e06b5ffc16a689a4c8",
    "supervision": "49bbb4bda9cb98e23d7393b33558137f196945fbabcf03d72baa40227a5345bb",
    "decision": "5e846542cb0a6153b362485603bbe660a96beae13d5daf525bbb4c4336df68aa",
    "manifest": "d2cce70a38b9f09e7df7da5ff5f262669516515170b5ba583661c9ef05bba9d0",
    "design": "4a3598435b61e1b5bade8cb3475a06cfa7ad48c031e4aef8ecf7adb1e6e92511",
    "astra": "0aad9c1397dc2522c754c4e6e8ed01ec102aa0c716e73d2f4aaedf8fe6f37770",
}
CALCULATION = {
    "frequency_hz": 2.382760007990865, "dt_s": .005, "anchors": 128,
    "temperature": "7*approved_normalization", "exponent": "-distance_squared/temperature",
    "constant_numpy_dtype": "<f8", "constant_layout": "C", "coordination_coefficient": 1,
    "unordered_pairs": [list(pair) for pair in PAIRS], "LR_pairs": [[0, 3], [1, 4], [2, 5]],
    "window_count": 50, "window_rows": 4, "loss_divisor": 50,
    "static_count": 129, "static_margin_minimum": 1e-8, "progress_margin_minimum": 1e-8,
    "scramble_leg_offsets": [0, 1/6, 2/6, 3/6, 4/6, 5/6],
    "command_relative_error_maximum": .05, "qualification": "signed_AND_window_L1_AND_progress_AND_static",
    "constant_rtol": 1e-12, "constant_atol": 1e-15,
    "numpy_value_rtol": 1e-10, "numpy_value_atol": 1e-10,
    "FD_h": 1e-6, "FD_rtol": 1e-6, "FD_atol": 1e-7, "FD_scaled_atol": 2e-9,
    "covariance_rtol": 1e-8, "covariance_atol": 1e-8, "covariance_scaled_atol": 2e-10,
    "CUDA_value_rtol": 1e-10, "CUDA_value_atol": 1e-10,
    "CUDA_gradient_rtol": 1e-8, "CUDA_gradient_atol": 1e-8, "CUDA_scaled_gradient_atol": 2e-10,
    "closure_rtol": 1e-12, "closure_atol": 1e-12,
    "steady_previous_slice": [100, 300], "steady_next_slice": [101, 301],
    "startup_targets": 99, "startup_guards": [.018, .04, .009, .025],
    "R_is_diagnostic": True, "z_normalization": False, "epsilon": False,
}
INTERPRETATION = {
    "Q_coordinate_leg": "command-coordinate contribution including cross-leg dependence; no causal leg responsibility",
    "component_Q": "derivative of each loss component, distinct from command-coordinate contributions",
    "E330_T_reused": False, "new_efficacy_gradient_computed": False,
    "neural_direction_qualified": False, "R_equals_one_assumed": False,
    "high_R_proves_true_phase": False, "static_fixedpoints_excluded": False,
    "scramble_is_time_permutation": False, "E318_E327_scores": "diagnostic only, no candidate adoption",
    "qualification_scope": "fixed saved command finite response, not neural adjoint or biology",
}


def _closure_maps(root):
    previous = read(safe(EVIDENCE["contract"], root))
    manifest = read(safe(EVIDENCE["manifest"], root))
    sources = {**previous["source_sha256"], **{name: sha256(safe(name, root)) for name in OWNED}}
    inputs = dict(previous["input_sha256"])
    inputs.update({PREVIOUS + "/" + name: row["sha256"] for name, row in manifest["files"].items()})
    inputs.update({name: sha256(safe(name, root)) for name in EVIDENCE.values()})
    for role in ("decision", "second"):
        inputs.update(read(safe(EVIDENCE[role], root))["evidence_sha256"])
    return previous, sources, inputs


def make_contract(*, root=ROOT):
    """Passive hashes/strict JSON only; zero Torch imports and NPZ payloads."""
    _, sources, inputs = _closure_maps(root)
    return {"schema": SCHEMA, "evidence": EVIDENCE.copy(), "source_sha256": sources,
            "input_sha256": inputs, "limits": LIMITS.copy(), "calculation": CALCULATION.copy(),
            "window_starts": list(STARTS), "original_calls": 0, "candidate_adopted": False}


def gate(contract_path, contract_sha256, output, *, root=ROOT, timely=lambda: None):
    """Exact E331 closure and its 3374-check completed proof before payload."""
    output = Path(output)
    if output.exists() or output.resolve() == root.resolve():
        raise ValueError("fresh output directory required")
    if sha256(contract_path) != contract_sha256:
        raise ValueError("contract SHA256 mismatch")
    contract = read(contract_path)
    if (set(contract) != {"schema", "evidence", "source_sha256", "input_sha256", "limits",
                         "calculation", "window_starts", "original_calls", "candidate_adopted"}
            or contract["schema"] != SCHEMA or contract["evidence"] != EVIDENCE
            or contract["limits"] != LIMITS or contract["calculation"] != CALCULATION
            or contract["window_starts"] != list(STARTS) or contract["original_calls"] != 0
            or contract["candidate_adopted"] is not False):
        raise ValueError("exact E332 saved contract required")
    pin_map(contract["source_sha256"], root, timely)
    pin_map(contract["input_sha256"], root, timely)
    for role, digest in FROZEN_PROOF.items():
        if contract["input_sha256"].get(EVIDENCE[role]) != digest:
            raise ValueError("frozen E331/E332 proof SHA differs: " + role)
    previous, expected_sources, expected_inputs = _closure_maps(root)
    if (len(previous["source_sha256"]) != 175 or len(previous["input_sha256"]) != 1470
            or contract["source_sha256"] != expected_sources or contract["input_sha256"] != expected_inputs):
        raise ValueError("exact inherited source/input maps required")
    for name in expected_sources.keys() & expected_inputs.keys():
        if expected_sources[name] != expected_inputs[name]:
            raise ValueError("contradictory source/input SHA")
    _, prior = predecessor.gate(safe(EVIDENCE["contract"], root), FROZEN_PROOF["contract"],
                                output, root=root, timely=timely)
    evidence = {role: read(safe(name, root)) for role, name in EVIDENCE.items()
                if role not in ("audit_source", "design", "astra", "prior_results")}
    result, ledger, manifest, audit, review, decision, passive, supervisor = (
        evidence[role] for role in ("result", "ledger", "manifest", "audit", "second", "decision", "passive", "supervision"))
    roster = {"reference.npz", *(f"window-{c}.npz" for c in STARTS)}
    if (manifest.get("schema") != "flylab.m3.e331.saved-residual.manifest.v1"
            or set(manifest.get("files", {})) != roster
            or manifest.get("mutable_metadata_excluded") != ["result.json", "advance-ledger.json", "publication-failure.json"]):
        raise ValueError("exact 51 immutable E331 raw files required")
    for name, row in manifest["files"].items():
        full = PREVIOUS + "/" + name
        if row != {"sha256": expected_inputs[full], "bytes": safe(full, root).stat().st_size}:
            raise ValueError("immutable E331 descriptor differs")
    if (audit.get("schema") != "flylab.m3.e331.root-saved-raw-audit.v1"
            or audit.get("passed") is not True or len(audit.get("checks", {})) != 3374
            or not all(value is True for value in audit["checks"].values())):
        raise ValueError("completed all-true 3374-check E331 root audit required")
    for key, role in (("result_sha256", "result"), ("advance_ledger_sha256", "ledger"),
                      ("output_manifest_sha256", "manifest"), ("contract_sha256", "contract"),
                      ("supervision_sha256", "supervision"), ("audit_source_sha256", "audit_source")):
        if audit.get(key) != expected_inputs[EVIDENCE[role]]:
            raise ValueError("E331 root audit hash link differs: " + key)
    if (review.get("schema") != "flylab.m3.e331.root-SECOND.v1" or review.get("approved") is not True
            or review.get("blocking_defects") != []
            or review.get("source_sha256") != {name: previous["source_sha256"][name] for name in predecessor.OWNED}
            or passive.get("passed") is not True or passive.get("source_count") != 175
            or passive.get("input_count") != 1470
            or passive.get("contract_sha256") != FROZEN_PROOF["contract"]
            or passive.get("source_review_sha256") != FROZEN_PROOF["second"]):
        raise ValueError("E331 SECOND-approved/passive proof required")
    for obj in (result, ledger):
        if (obj.get("analysis_valid") is not True or obj.get("completed_windows") != 50
                or obj.get("errors") != [] or obj.get("first_error") is not None):
            raise ValueError("completed valid E331 result/ledger required")
    if (result.get("schema") != "flylab.m3.e331.saved-residual.result.v1"
            or ledger.get("schema") != "flylab.m3.e331.saved-residual.ledger.v1"
            or result.get("source_sha256") != previous["source_sha256"]
            or result.get("input_sha256") != previous["input_sha256"]
            or result.get("contract_sha256") != FROZEN_PROOF["contract"]
            or [row.get("start_control") for row in result.get("windows", [])] != list(STARTS)
            or len(ledger.get("attempts", [])) != 51
            or {row.get("file") for row in ledger["attempts"]} != roster
            or any(row.get("returned") is not True for row in ledger["attempts"])
            or result.get("candidate_adopted") is not False or result.get("F1_F8_complete") is not False
            or result.get("E330_local_direction") != "refuted" or result.get("continuation") != "not_tested"):
        raise ValueError("E331 result/raw/source linkage differs")
    if (supervisor.get("success") is not True or len(supervisor.get("attempts", [])) != 1
            or supervisor["attempts"][0].get("exit_code") != 0 or supervisor["attempts"][0].get("killed") is not False
            or not supervisor.get("owned_pids") or any(psutil.pid_exists(pid) for pid in supervisor["owned_pids"])
            or decision.get("schema") != "flylab.m3.e331.root-final-decision.v1"
            or decision.get("saved_analysis_valid") is not True or decision.get("root_raw_audit_pass") is not True
            or decision.get("root_raw_checks") != 3374 or decision.get("source_review_approved") is not True
            or decision.get("candidate_adopted") is not False or decision.get("E330_local_direction") != "refuted"
            or decision.get("original_cumulative_s") != 9200.673308899522
            or decision.get("original_allocation_limit_s") != 9360 or decision.get("original_reserved_s") != 0):
        raise ValueError("completed E331 supervisor/final decision/accounting required")
    for role in ("decision", "second"):
        for name, digest in evidence[role]["evidence_sha256"].items():
            if expected_inputs.get(name) != digest:
                raise ValueError("E331 final proof link differs")
    e329 = read(safe(prior["freeze"]["paths"]["e329_contract"], root))
    paths = {}
    for role, descriptor in e329["inputs"].items():
        path = Path(descriptor["path"])
        resolved = (path if path.is_absolute() else safe(prior["freeze"]["paths"]["e329_contract"], root).parent / path).resolve()
        name = resolved.relative_to(root.resolve()).as_posix()
        if expected_inputs.get(name) != descriptor["sha256"]:
            raise ValueError("E329 exact approved raw path/hash differs")
        paths[role] = name
    if (paths["reference"] != prior["freeze"]["paths"]["e85_reference"]
            or paths["startup"] != prior["freeze"]["paths"]["e103_startup"]
            or paths["neutral"] != prior["freeze"]["paths"]["e318_raw"]
            or paths["rejected"] != BASE + "/e327-constrained-secant-body/control-raw.npz"):
        raise ValueError("E329 frozen E318/E327 reference/startup path differs")
    protected = {Path(contract_path).resolve(), *(safe(name, root) for name in expected_sources.keys() | expected_inputs.keys())}
    if any(path.is_relative_to(output.resolve()) for path in protected):
        raise ValueError("output contains protected input/source")
    evidence["prior"] = prior
    evidence["approved_paths"] = paths
    return contract, evidence


def _array(value):
    return value.detach().cpu().numpy().copy()


def _encoding_raw(prefix, encoded, raw):
    for key in ("weights", "target", "z", "R", "entropy", "maxweight", "zero_weight_count", "distance_sq"):
        raw[prefix + "_" + key] = _array(encoded[key])


def _values_raw(prefix, values, raw):
    raw[prefix + "_terms"] = np.array([float(values[key].detach()) for key in TERMS], np.float64)
    for key in ("per_leg_successor_loss", "per_leg_successor_mse", "successor_terms", "coordination_terms", "coordination_pair_loss"):
        raw[prefix + "_" + key] = _array(values[key])
    for side in ("previous", "next"):
        _encoding_raw(prefix + "_" + side, values[side + "_encoding"], raw)


def _gradient(torch, value, q):
    gradient = torch.autograd.grad(value / 50, q, retain_graph=True)[0]
    array = _array(gradient)
    f64(array, tuple(q.shape), "scaled command gradient")
    return array


def analyze_window(objective, boundary, arms, *, start_control=100):
    """New L/50 command gradient and finite response; no old neural T input."""
    import torch
    f64(boundary, (42,), "boundary")
    if set(arms) != {"neutral", "plus", "minus"}:
        raise ValueError("exact neutral/plus/minus arms required")
    for label in arms:
        f64(arms[label], (4, 42), label)
    raw, values = {"boundary": boundary.copy()}, {}
    for label in ("neutral", "plus", "minus"):
        q = torch.tensor(arms[label], dtype=torch.float64, device="cpu", requires_grad=label == "neutral")
        previous = torch.cat((torch.tensor(boundary[None], dtype=torch.float64), q[:3]))
        value = objective.evaluate_pairs(previous, q)
        _values_raw(label, value, raw)
        raw[label + "_q"] = arms[label].copy()
        values[label] = value
        if label == "neutral":
            gq = _gradient(torch, value["training_loss"], q)
            term_gq = np.stack([_gradient(torch, value[key], q) for key in TERMS])
            # Components are mean leg/pair losses, not coordinate leg shares.
            leg_gq = np.stack([_gradient(torch, value["per_leg_successor_loss"][leg], q) for leg in range(6)])
            pair_gq = np.stack([_gradient(torch, value["coordination_pair_loss"][pair], q) for pair in range(15)])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        dq = arms["plus"] - arms["minus"]
        term_o = (raw["plus_terms"] - raw["minus_terms"]) / 50
        term_q = np.sum(term_gq * dq[None], axis=(1, 2))
        o, predicted = float(term_o[0]), float(np.sum(gq * dq))
        leg_o = (raw["plus_per_leg_successor_loss"] - raw["minus_per_leg_successor_loss"]) / 50
        pair_o = (raw["plus_coordination_pair_loss"] - raw["minus_coordination_pair_loss"]) / 50
        leg_q = np.sum(leg_gq * dq[None], axis=(1, 2))
        pair_q = np.sum(pair_gq * dq[None], axis=(1, 2))
        coordinate_leg = np.sum((gq * dq).reshape(4, 6, 7), axis=(0, 2))
    finite = [dq, term_o, term_q, leg_o, pair_o, leg_q, pair_q, coordinate_leg]
    if not all(np.isfinite(value).all() for value in finite):
        raise ValueError("window contributions must be finite")
    for actual, expected in ((term_q[0], predicted), (term_o[0], term_o[1] + term_o[2]),
                             (term_q[0], term_q[1] + term_q[2]), (leg_o.mean(), term_o[1]),
                             (leg_q.mean(), term_q[1]), (pair_o.mean(), term_o[2]),
                             (pair_q.mean(), term_q[2]), (coordinate_leg.sum(), predicted)):
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise ValueError("component/coordinate closure failed")
    raw.update(neutral_gq=gq, neutral_term_gq=term_gq, delta_q=dq,
               successor_leg_gq=leg_gq, coordination_pair_gq=pair_gq,
               term_O=term_o, term_Q=term_q, successor_leg_O=leg_o, successor_leg_Q=leg_q,
               coordination_pair_O=pair_o, coordination_pair_Q=pair_q,
               Q_coordinate_leg=coordinate_leg, O=np.array(o), Q=np.array(predicted), O_minus_Q=np.array(o-predicted))
    row = {"start_control": start_control, "physics_time_s": start_control * .005,
           "O": o, "Q": predicted, "O_minus_Q": float(o-predicted),
           "term_O": term_o.tolist(), "term_Q": term_q.tolist(),
           "successor_leg_O": leg_o.tolist(), "successor_leg_Q": leg_q.tolist(),
           "coordination_pair_O": pair_o.tolist(), "coordination_pair_Q": pair_q.tolist(),
           "Q_coordinate_leg": coordinate_leg.tolist(), "command_delta_L2": float(np.linalg.norm(dq))}
    return row, raw


def aggregate(windows):
    o = np.array([row["O"] for row in windows], np.float64)
    q = np.array([row["Q"] for row in windows], np.float64)
    if not np.isfinite(o).all() or not np.isfinite(q).all():
        raise ValueError("aggregate finite O/Q required")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        residual = o - q
        signed = max(abs(float(o.sum())), abs(float(q.sum())))
        l1 = max(float(np.abs(o).sum()), float(np.abs(q).sum()))
        signed_error = None if signed == 0 else abs(float(residual.sum())) / signed
        l1_error = None if l1 == 0 else float(np.abs(residual).sum()) / l1
        terms = {name: {"signed_sum": float(value.sum()), "sum_abs": float(np.abs(value).sum())}
                 for name, value in (("O", o), ("Q", q), ("O_minus_Q", residual))}
    qualified = bool(signed > 0 and l1 > 0 and signed_error <= .05 and l1_error <= .05)
    result = {"terms": terms, "D_signed": signed, "D_L1": l1, "e_signed": signed_error, "e_L1": l1_error,
              "finite_command_qualification": qualified,
              "reason": "no_direction_information" if signed == 0 or l1 == 0 else ("pass" if qualified else "relative_error_exceeds_0.05"),
              "absolute_closure_valid": True,
              "ranked_residual_controls": [row["start_control"] for row in sorted(windows, key=lambda row: abs(row["O_minus_Q"]), reverse=True)]}
    for field in ("term_O", "term_Q", "successor_leg_O", "successor_leg_Q", "coordination_pair_O", "coordination_pair_Q", "Q_coordinate_leg"):
        if windows and all(field in row for row in windows):
            value = np.array([row[field] for row in windows], np.float64)
            if not np.isfinite(value).all():
                raise ValueError("finite aggregate components required")
            result[field] = {"signed_sum": value.sum(0).tolist(), "sum_abs": np.abs(value).sum(0).tolist()}
    # Validate reductions too: finite inputs can still overflow their sums.
    import json
    json.dumps(result, allow_nan=False)
    return result


def offcurve_distance(curves, q):
    """Detached diagnostic distance to the approved polyline; never loss math."""
    f64(curves, (6, 128, 7), "curves")
    f64(q, (len(q), 42), "offcurve q")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        edge = np.roll(curves, -1, axis=1) - curves
        delta = q.reshape(-1, 6, 7)[:, :, None] - curves[None]
        norm = np.sum(edge ** 2, axis=-1)
        coefficient = np.sum(delta * edge[None], axis=-1) / norm[None]
        projected = curves[None] + np.clip(coefficient, 0, 1)[..., None] * edge[None]
        distance = np.sum((q.reshape(-1, 6, 7)[:, :, None] - projected) ** 2, axis=-1).min(-1)
    if not np.isfinite(distance).all():
        raise ValueError("offcurve diagnostic must be finite")
    return distance / 7


def _summary(array):
    value = np.asarray(array)
    if not np.isfinite(value).all():
        raise ValueError("finite diagnostic summary required")
    return {"mean": float(value.mean()), "min": float(value.min()), "max": float(value.max())}


def reference_cases(objective):
    """Frozen 128 forward/static/reverse/scramble and 129 static minimum."""
    import torch

    from tools.analyze_c_banc_m3_temporal_objective import numpy_interpolate
    curves = objective.canonical_constants["curves"]
    phase = np.tile(np.arange(128, dtype=np.float64)[:, None] / 128, (1, 6))
    previous = numpy_interpolate(curves, phase)
    next_cases = {"forward": numpy_interpolate(curves, phase + objective.phase_increment),
                  "static": previous.copy(),
                  "reverse": numpy_interpolate(curves, phase - objective.phase_increment),
                  "scramble": numpy_interpolate(curves, phase + objective.phase_increment
                                                + np.arange(6, dtype=np.float64)[None] / 6)}
    raw, summaries = {"anchor_phase": phase.copy(), "reference_previous_q": previous.copy()}, {}
    losses = {}
    with torch.no_grad():
        for label, following in next_cases.items():
            values = objective.evaluate_pairs(torch.tensor(previous), torch.tensor(following))
            losses[label] = values
            _values_raw(label, values, raw)
            raw[label + "_q"] = following.copy()
            row_training = _array(values["successor_terms"]).mean(-1) + _array(values["coordination_terms"]).mean(-1)
            raw[label + "_row_training_loss"] = row_training
            for side, commands in (("previous", previous), ("next", following)):
                raw[f"{label}_{side}_offcurve_mse"] = offcurve_distance(curves, commands)
            summaries[label] = {"terms": raw[label + "_terms"].tolist(),
                                "per_leg_successor_loss": raw[label + "_per_leg_successor_loss"].tolist(),
                                "training_phase_range": _summary(row_training),
                                **{key: _summary(raw[f"{label}_next_{key}"])
                                   for key in ("R", "entropy", "maxweight", "zero_weight_count")},
                                "offcurve_mse": _summary(raw[label + "_next_offcurve_mse"])}
        static_points = np.vstack((curves.mean(axis=1).reshape(1, 42), previous))
        static_values = objective.evaluate_pairs(torch.tensor(static_points), torch.tensor(static_points))
        static_loss = _array(static_values["successor_terms"]).mean(-1) + _array(static_values["coordination_terms"]).mean(-1)
        _values_raw("known_static", static_values, raw)
        raw["known_static_q"] = static_points
        raw["known_static_training_loss"] = static_loss
        raw["known_static_offcurve_mse"] = offcurve_distance(curves, static_points)
        minimum_index = int(np.argmin(static_loss))  # fixed diagnostic reduction only
        point = static_points[minimum_index:minimum_index + 1]
        singleton = objective.evaluate_pairs(torch.tensor(point), torch.tensor(point))
        repeat = np.repeat(point, 4, axis=0)
        window = objective.evaluate_pairs(torch.tensor(np.vstack((point, repeat[:3]))), torch.tensor(repeat))
        minimum = float(static_loss[minimum_index])
        if (not np.isclose(float(singleton["training_loss"]), minimum, rtol=1e-12, atol=1e-12)
                or not np.isclose(float(window["training_loss"]), minimum, rtol=1e-12, atol=1e-12)):
            raise ValueError("single pair/static four-row identity failed")
    forward = raw["forward_terms"]
    margins = {label + "_per_leg_successor": (raw[label + "_per_leg_successor_loss"]
                                               - raw["forward_per_leg_successor_loss"]).tolist()
               for label in ("static", "reverse")}
    margins.update({label + "_aggregate_training": float(raw[label + "_terms"][0] - forward[0])
                    for label in ("static", "reverse")})
    margins["scramble_coordination"] = float(raw["scramble_terms"][2] - forward[2])
    progress = bool(all(np.all(np.asarray(value) > 1e-8) for value in margins.values()))
    static_margin = float(minimum - forward[0])
    raw["forward_target_blur_normalized_mse"] = (
        ((raw["forward_previous_target"] - next_cases["forward"]).reshape(128, 6, 7) ** 2)
        .mean(-1) / objective.canonical_constants["normalization"][None])
    summary = {"reference_progress_pass": progress, "progress_margins": margins,
               "known_static_discrimination": bool(static_margin > 1e-8),
               "known_static_count": 129, "known_static_minimum": minimum,
               "known_static_minimum_index": minimum_index,
               "known_static_minimum_kind": "curve_mean" if minimum_index == 0 else "anchor",
               "known_static_minimum_anchor": None if minimum_index == 0 else minimum_index - 1,
               "forward_training_mean": float(forward[0]), "static_margin": static_margin,
               "cases": summaries, "target_blur_normalized_mse": _summary(raw["forward_target_blur_normalized_mse"]),
               "R_is_confidence_and_geometry_diagnostic": True, "static_fixedpoints_excluded": False}
    return summary, raw


def _clock_commands(data, label):
    q = f64(data["command"], (301, 42), label + " command")
    for name, expected in (("control_tick", np.arange(301)), ("network_tick", np.arange(301) * 20)):
        value = data[name]
        if value.shape != (301,) or value.dtype.kind not in "iu" or not np.array_equal(value, expected):
            raise ValueError(label + " complete integer clock differs")
    times = f64(data["physics_time_s"], (301,), label + " physics clock")
    if not np.allclose(times, np.arange(301) * .005, rtol=0, atol=2e-8):
        raise ValueError(label + " 5ms physics clock differs")
    return q


def compute_saved(contract, evidence, rows, save, timely, *, root=ROOT):
    """Root-only saved execution, called exclusively after passive gate."""
    import torch

    from tools.analyze_c_banc_m3_temporal_objective import curve128
    from tools.banc_m3_smooth_temporal import SmoothTemporalObjective
    torch.set_num_threads(1)
    paths = evidence["approved_paths"]
    approved = evidence["prior"]["freeze"]["temporal_reference"]
    curves = curve128(load_npz(safe(paths["reference"], root), ("time_s", "phase", "command")))
    norm = np.array(approved["normalization_f64"], dtype="<f8", order="C")
    f64(norm, (6,), "approved normalization")
    prior_reference = load_npz(safe(PREVIOUS + "/reference.npz", root), ("curves", "normalization"))
    f64(prior_reference["curves"], (6, 128, 7), "E331 curves")
    f64(prior_reference["normalization"], (6,), "E331 normalization")
    if (bytes_sha(curves) != approved["curves"]["c_order_sha256"]
            or bytes_sha(norm) != approved["normalization"]["c_order_sha256"]
            or bytes_sha(curves) != bytes_sha(prior_reference["curves"])
            or bytes_sha(norm) != bytes_sha(prior_reference["normalization"])):
        raise ValueError("approved E329/E331 canonical reference bytes differ")
    startup = load_npz(safe(paths["startup"], root), ("target_command", "target_tendon"))
    objective = SmoothTemporalObjective(curves, startup["target_command"], startup["target_tendon"],
                                        normalization=norm, device="cpu")
    constants = {key: value.copy() for key, value in objective.canonical_constants.items()}
    constants.update({"constructor_" + key: value for key, value in objective.constructor_recomputed.items()})
    save("reference.npz", constants)
    timely()
    reference, raw = reference_cases(objective)
    save("reference-cases.npz", raw)
    reference["constant_metadata"] = objective.constant_metadata
    raw_commands = {}
    for label, role in (("E318", "neutral"), ("E327", "rejected")):
        data = load_npz(safe(paths[role], root), ("command", "control_tick", "network_tick", "physics_time_s"))
        commands = _clock_commands(data, label)
        raw_commands[label] = commands
        with torch.no_grad():
            values = objective.evaluate_pairs(torch.tensor(commands[100:300]), torch.tensor(commands[101:301]))
            raw = {"previous_q": commands[100:300], "q": commands[101:301],
                   **{key: data[key] for key in ("control_tick", "network_tick", "physics_time_s")}}
            _values_raw(label, values, raw)
        raw[label + "_next_offcurve_mse"] = offcurve_distance(curves, commands[101:301])
        reference[label + "_diagnostic"] = {"terms": raw[label + "_terms"].tolist(),
            "per_leg_successor_loss": raw[label + "_per_leg_successor_loss"].tolist(),
            **{key: _summary(raw[label + "_next_" + key]) for key in ("R", "entropy", "maxweight", "zero_weight_count")},
            "offcurve_mse": _summary(raw[label + "_next_offcurve_mse"]), "candidate_adopted": False}
        save(label.lower() + "-diagnostic.npz", raw)
        timely()
    original = evidence["prior"]["result"]
    if [row["start_control"] for row in original["partial"]["windows"]] != list(STARTS):
        raise ValueError("original E330 window ordering differs")
    for c in STARTS:
        timely()
        previous = load_npz(safe(PREVIOUS + f"/window-{c}.npz", root),
                            ("boundary", "neutral_q", "plus_q", "minus_q"))
        boundary = f64(previous["boundary"], (42,), "E331 boundary")
        if bytes_sha(boundary) != bytes_sha(raw_commands["E318"][c]):
            raise ValueError("fixed original q[c] boundary differs")
        arms = {}
        for label in ("neutral", "plus", "minus"):
            name = f"window-{c}-{label}"
            data = load_npz(safe(ORIGINAL + "/" + name + ".npz", root), ("q", "fixed_original_q_boundary"))
            arms[label] = f64(data["q"], (4, 42), name)
            original_boundary = f64(data["fixed_original_q_boundary"], (42,), name + " boundary")
            f64(previous[label + "_q"], (4, 42), "E331 " + label)
            if (bytes_sha(original_boundary) != bytes_sha(boundary)
                    or bytes_sha(arms[label]) != bytes_sha(previous[label + "_q"])):
                raise ValueError("E330/E331 exact q/boundary bytes differ")
            if original["partial"]["runs"][name]["raw_file"] != name + ".npz":
                raise ValueError("E330 original raw file linkage differs")
        row, saved = analyze_window(objective, boundary, arms, start_control=c)
        save(f"window-{c}.npz", saved)
        rows.append(row)
        timely()
    return reference


def run(contract_path, contract_sha256, output, *, root=ROOT, writer=write_json,
        saver=np.savez_compressed, clock=time.perf_counter, gate_fn=gate, compute_fn=None):
    """Retain raw, first error, partial costs and coherent disk/return status."""
    began = clock()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("fresh output required")
    errors, rows, files = [], [], {}
    ledger = {"schema": "flylab.m3.e332.smooth-temporal.ledger.v1", "limits": LIMITS.copy(), "attempts": [],
              "original_calls": 0, "body_calls": 0, "CUDA_calls": 0, "model_calls": 0, "optimizer_steps": 0}
    result = {"schema": RESULT_SCHEMA, "analysis_valid": False, "input_qualification_pass": False,
              "candidate_adopted": False, "F1_F8_complete": False, "biological_gait_proven": False,
              "E330_local_direction": "refuted", "continuation": "not_tested", "body_not_executed": True,
              "interpretation": INTERPRETATION, "windows": rows, "original_calls": 0, "body_calls": 0,
              "CUDA_calls": 0, "model_calls": 0, "optimizer_steps": 0, "whole_goal_status": "active",
              "original_cumulative_s": 9200.673308899522, "original_allocation_limit_s": 9360,
              "original_reserved_s": 0, "contract_sha256": contract_sha256}

    def timely():
        elapsed = clock() - began
        rss = psutil.Process().memory_info().rss
        ledger["max_host_rss_bytes"] = max(ledger.get("max_host_rss_bytes", 0), rss)
        if elapsed > LIMITS["calculation_rehash_deadline_s"] or rss > LIMITS["host_bytes"]:
            raise RuntimeError("calculation/rehash deadline or host limit exceeded")

    def error(stage, exception):
        errors.append({"stage": stage, "type": type(exception).__name__, "message": str(exception), "elapsed_s": clock() - began})

    def save(name, raw):
        timely()
        if Path(name).name != name or not name.endswith(".npz"):
            raise ValueError("owned raw basename required")
        if (output / name).exists():
            raise ValueError("fresh raw file required")
        if not all(np.isfinite(value).all() for value in raw.values()):
            raise ValueError("raw must contain finite arrays")
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
        output.rmdir()
        try:
            contract, evidence = gate_fn(Path(contract_path), contract_sha256, output, root=root, timely=timely)
        finally:
            output.mkdir(exist_ok=True)
        timely()
        result["source_sha256"] = contract["source_sha256"]
        result["input_sha256"] = contract["input_sha256"]
        result["predecessor_result_sha256"] = contract["input_sha256"][EVIDENCE["result"]]
        result["predecessor_ledger_sha256"] = contract["input_sha256"][EVIDENCE["ledger"]]
        reference = (compute_fn(contract, evidence, rows, save, timely) if compute_fn is not None
                     else compute_saved(contract, evidence, rows, save, timely, root=root))
        pin_map(contract["source_sha256"], root, timely)
        pin_map(contract["input_sha256"], root, timely)
        if sha256(contract_path) != contract_sha256:
            raise ValueError("post-computation contract SHA mismatch")
        timely()
        if [row["start_control"] for row in rows] != list(STARTS):
            raise ValueError("all 50 completed windows required")
        if (not isinstance(reference, dict) or type(reference.get("reference_progress_pass")) is not bool
                or type(reference.get("known_static_discrimination")) is not bool):
            raise ValueError("completed frozen reference qualification required")
        result["reference_qualification"] = reference
        result["aggregate"] = aggregate(rows)
        result["analysis_valid"] = True
        result["input_qualification_pass"] = bool(reference["reference_progress_pass"]
            and reference["known_static_discrimination"] and result["aggregate"]["finite_command_qualification"])
        result["calculation_and_rehash_elapsed_s"] = clock() - began
    except Exception as exception:  # noqa: BLE001
        error("analysis", exception)

    def terminal_state():
        elapsed = clock() - began
        first = errors[0] if errors else None
        if errors:
            result["analysis_valid"] = False
            result["input_qualification_pass"] = False
        common = {"errors": errors, "first_error": first, "completed_windows": len(rows),
                  "analysis_valid": result["analysis_valid"], "input_qualification_pass": result["input_qualification_pass"],
                  "elapsed_s": elapsed, "measured_boundary": "before_terminal_metadata_commit"}
        result.update(common)
        ledger.update(common)

    terminal_state()
    try:
        writer(output / "output-manifest.json", {"schema": "flylab.m3.e332.smooth-temporal.manifest.v1", "files": files,
            "mutable_metadata_excluded": ["result.json", "advance-ledger.json", "publication-failure.json"]})
        writer(output / "advance-ledger.json", ledger)
        writer(output / "result.json", result)
        if clock() - began > LIMITS["internal_wall_s"] or sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > LIMITS["output_bytes"]:
            raise RuntimeError("internal finalization/output limit exceeded")
    except Exception as exception:  # noqa: BLE001
        error("publication", exception)
        terminal_state()
        try:
            write_json(output / "publication-failure.json", {"schema": "flylab.m3.e332.publication-failure.v1",
                "analysis_valid": False, "input_qualification_pass": False, "first_error": errors[0], "errors": errors,
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
        if any(row["stage"].startswith("terminal_repair_") for row in errors):
            terminal_state()
            for name, value in repaired:
                try:
                    _replace_terminal_json(output / name, value)
                except Exception as secondary:  # noqa: BLE001
                    error("terminal_refresh_" + name, secondary)
            terminal_state()
    return result


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = run(args.contract, args.contract_sha256, args.output)
    print(json.dumps({"analysis_valid": result["analysis_valid"], "input_qualification_pass": result["input_qualification_pass"],
                      "completed_windows": result["completed_windows"], "first_error": result["first_error"]}, allow_nan=False))
    return 0 if result["analysis_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
