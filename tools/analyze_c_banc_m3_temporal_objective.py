"""E329 saved-only characterization with a strict pre-read input/source freeze.

Run with --contract CONTRACT_JSON --output NEW_JSON. The contract schema is
flylab.m3.e329.temporal.saved-contract.v1, inputs are exactly reference/startup/
neutral/rejected descriptors {path, sha256}, and source_sha256 pins exactly the
four OWNED_SOURCES below by repository-relative name. Relative input paths are
resolved from the contract's directory. All pins are checked before any NPZ is
opened. E318 is neutral and E327 remains rejected regardless of score ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from tools.banc_m3_phase_free_temporal import (
    ANCHORS, CONTROL_DT_S, FREQUENCY_HZ, PhaseFreeTemporalObjective,
)

ROOT = Path(__file__).resolve().parents[1]
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
SCHEMA = "flylab.m3.e329.temporal.saved-contract.v1"
OWNED_SOURCES = (
    "tools/banc_m3_phase_free_temporal.py",
    "tools/analyze_c_banc_m3_temporal_objective.py",
    "tests/test_c_banc_phase_free_temporal.py",
    "tests/test_c_banc_temporal_objective.py",
)
INPUT_ROLES = ("reference", "startup", "neutral", "rejected")
NONLOCAL_TIE_SQ = 1e-12
PHASE_TOLERANCE = 1e-8
SUCCESSOR_TOLERANCE_RAD = 1e-10
LOSS_TOLERANCE = 1e-10
DISTINGUISH_MARGIN = 1e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def verify_contract(contract_path: Path) -> tuple[dict, dict]:
    """Verify the exact roster before returning paths for numeric reads."""
    contract_path = contract_path.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf8"),
                          object_pairs_hook=_unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ValueError("nonfinite JSON constant: " + value)))
    if (not isinstance(contract, dict) or set(contract) != {"schema", "inputs", "source_sha256"}
            or contract["schema"] != SCHEMA
            or not isinstance(contract["inputs"], dict)
            or set(contract["inputs"]) != set(INPUT_ROLES)
            or not isinstance(contract["source_sha256"], dict)
            or set(contract["source_sha256"]) != set(OWNED_SOURCES)):
        raise ValueError("exact E329 schema, four input roles, and four source pins required")
    paths, pins = {}, {}
    for role in INPUT_ROLES:
        descriptor = contract["inputs"][role]
        if (not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}
                or not isinstance(descriptor["path"], str) or not descriptor["path"]):
            raise ValueError("invalid input descriptor: " + role)
        path = Path(descriptor["path"])
        path = (path if path.is_absolute() else contract_path.parent / path).resolve()
        paths[role] = path
        pins[role] = _pin(path, descriptor["sha256"], role)
    if len(set(paths.values())) != 4:
        raise ValueError("four distinct input files required")
    sources = {name: _pin(ROOT / name, contract["source_sha256"][name], name)
               for name in OWNED_SOURCES}
    return paths, {"contract": _pin(contract_path, sha256(contract_path), "contract"),
                   "inputs": pins, "sources": sources}


def _pin(path, expected, name):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("lowercase SHA256 required: " + name)
    digest = sha256(path)
    if digest != expected:
        raise ValueError("SHA256 mismatch: " + name)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def load_npz(path: Path, keys: tuple[str, ...]) -> dict:
    """Read only required command/reference arrays, never saved neural state."""
    with np.load(path, allow_pickle=False) as archive:
        if not set(keys).issubset(archive.files):
            raise ValueError("required NPZ arrays missing: " + str(path))
        return {name: archive[name] for name in keys}


def _f64(array, shape, name):
    if (not isinstance(array, np.ndarray) or array.dtype != np.float64
            or array.shape != shape or not np.isfinite(array).all()):
        raise ValueError("finite float64 " + name + " with shape " + str(shape) + " required")
    return array


def validate_saved(raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """Raw controls 0..300, network ticks 20/control, and fixed 5ms clock."""
    expected = np.arange(301)
    for key, values in (("control_tick", expected), ("network_tick", expected * 20)):
        ticks = raw[key]
        if (not isinstance(ticks, np.ndarray) or ticks.shape != (301,)
                or ticks.dtype.kind not in "iu" or not np.array_equal(ticks, values)):
            raise ValueError("complete integer " + key + " clock required")
    times = _f64(raw["physics_time_s"], (301,), "physics_time_s")
    if not np.allclose(times, expected * CONTROL_DT_S, rtol=0, atol=2e-8):
        raise ValueError("control timestamps must follow the fixed 5ms clock")
    return (_f64(raw["command"], (301, 42), "command"),
            _f64(raw["tendon"], (301, 6), "tendon"))


def curve128(reference: dict) -> np.ndarray:
    """Exact E319/E323 builder: time>=5, sorted LF global phase, periodic interp."""
    times = _f64(reference["time_s"], (2000,), "reference time_s")
    phase = _f64(reference["phase"], (2000, 6), "reference phase")
    command = _f64(reference["command"], (2000, 42), "reference command")
    if not np.allclose(np.diff(times), CONTROL_DT_S, rtol=0, atol=2e-8):
        raise ValueError("reference timestamps must follow the fixed 5ms clock")
    # This validation fixes the cadence independently of any saved candidate.
    cadence = np.diff(np.unwrap(phase[:, 0])) / (2 * np.pi * CONTROL_DT_S)
    if not np.isfinite(cadence).all() or not np.allclose(cadence, FREQUENCY_HZ, rtol=0, atol=1e-9):
        raise ValueError("reference LF cadence differs from fixed E329 frequency")
    late = times >= 5
    if not late.any():
        raise ValueError("late reference must not be empty")
    coordinates = np.mod(phase[late, 0] / (2 * np.pi), 1)
    order = np.argsort(coordinates)
    x, y = coordinates[order], command[late][order]
    extended_x = np.r_[x[-1] - 1, x, x[0] + 1]
    extended_y = np.vstack((y[-1], y, y[0]))
    query = np.arange(ANCHORS, dtype=np.float64) / ANCHORS
    rows = np.column_stack([np.interp(query, extended_x, extended_y[:, j]) for j in range(42)])
    return np.ascontiguousarray(rows.reshape(ANCHORS, 6, 7).transpose(1, 0, 2))


def numpy_interpolate(curves: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Independent CPU reference interpolation for qualification target values."""
    coordinate = np.remainder(phase, 1) * ANCHORS
    floor = np.floor(coordinate)
    index = floor.astype(np.int64) % ANCHORS
    fraction = coordinate - floor
    legs = np.arange(6)[None, :]
    first = curves[legs, index]
    last = curves[legs, (index + 1) % ANCHORS]
    return (first + fraction[..., None] * (last - first)).reshape(-1, 42)


def _tensor(array):
    return torch.as_tensor(array, dtype=torch.float64, device="cpu")


def _array(value):
    return value.detach().numpy()


def reference_metadata(objective: PhaseFreeTemporalObjective) -> dict:
    """Exact owned f64 reference bytes for the separately approved E330 linkage."""
    def array_metadata(array):
        contiguous = np.ascontiguousarray(array)
        return {"shape": list(contiguous.shape), "dtype": str(contiguous.dtype),
                "numpy_dtype": contiguous.dtype.str, "layout": "C",
                "leg_order": list(LEGS),
                "c_order_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()}
    curves = _array(objective.curves)
    normalization = np.ascontiguousarray(_array(objective.successor_normalization))
    curves_meta = array_metadata(curves)
    curves_meta["joint_index_order"] = list(range(7))
    normalization_meta = array_metadata(normalization)
    normalization_meta["per_leg_f64_sha256"] = {
        leg: hashlib.sha256(normalization[i:i + 1].tobytes(order="C")).hexdigest()
        for i, leg in enumerate(LEGS)}
    return {"reference_curves_metadata": curves_meta,
            "successor_normalization_metadata": normalization_meta}


def scores(result: dict) -> dict:
    names = ("training_loss", "successor_loss", "per_leg_successor_loss",
             "per_leg_successor_mse", "coordination_loss", "lr_coordination_loss",
             "shape_loss", "per_leg_shape_loss", "startup_q_rms", "startup_q_max",
             "startup_tendon_rms", "startup_tendon_max", "startup_pass")
    return {name: _array(result[name]).tolist() for name in names if name in result}


def projection_diagnostics(projection: dict) -> dict:
    """Raw per-sample diagnostics and nonlocal competitors, excluding seam ties."""
    phase = _array(projection["phase"])
    segment = _array(projection["segment"])
    distance = _array(projection["distance_sq"])
    all_distance = _array(projection["all_segment_distance_sq"])
    all_phase = _array(projection["all_segment_phase"])
    segment_gap = np.abs(np.arange(ANCHORS)[None, None, :] - segment[..., None])
    segment_gap = np.minimum(segment_gap, ANCHORS - segment_gap)
    phase_gap = np.abs(all_phase - phase[..., None])
    phase_gap = np.minimum(phase_gap, 1 - phase_gap)
    nonlocal_mask = (segment_gap > 1) & (phase_gap > 2 / ANCHORS)
    # None represents an unavailable competitor; no Inf enters emitted JSON.
    gap = np.where(nonlocal_mask, all_distance - distance[..., None], np.inf).min(axis=-1)
    available = np.isfinite(gap)
    ties = available & (gap <= NONLOCAL_TIE_SQ)
    raw_u, u = _array(projection["raw_u"]), _array(projection["u"])
    jump = np.abs(np.diff(phase, axis=0))
    jump = np.minimum(jump, 1 - jump)
    legs = {}
    for leg, name in enumerate(LEGS):
        valid_gap = gap[available[:, leg], leg]
        legs[name] = {
            "projection_distance_rms_rad": float(np.sqrt(distance[:, leg].mean() / 7)),
            "projection_distance_max_rad": float(np.sqrt(distance[:, leg].max())),
            "clamp_endpoint_fraction": float(np.mean((u[:, leg] == 0) | (u[:, leg] == 1))),
            "strict_clamp_fraction": float(np.mean((raw_u[:, leg] < 0) | (raw_u[:, leg] > 1))),
            "circular_jump_mean_cycle": float(jump[:, leg].mean()) if len(jump) else None,
            "circular_jump_max_cycle": float(jump[:, leg].max()) if len(jump) else None,
            "nonlocal_gap_min_sq_rad": float(valid_gap.min()) if valid_gap.size else None,
            "nonlocal_tie_count": int(ties[:, leg].sum()),
        }
    return {"legs": legs, "phase_proxy_cycle": phase.tolist(), "selected_segment": segment.tolist(),
            "projection_distance_sq_rad": distance.tolist(), "projection_u": u.tolist(),
            "projection_raw_u": raw_u.tolist(), "circular_jump_cycle": jump.tolist(),
            "nonlocal_competitor_gap_sq_rad": [[float(x) if np.isfinite(x) else None for x in row]
                                                for row in gap],
            "nonlocal_tie": ties.tolist(), "nonlocal_tie_count": int(ties.sum()),
            "nonlocal_definition": "cyclic segment distance >1 AND circular phase gap >2/128",
            "nonlocal_tie_max_gap_sq_rad": NONLOCAL_TIE_SQ}


def qualify_reference(objective: PhaseFreeTemporalObjective) -> dict:
    """Fixed actual-reference round-trip and counterexample qualification."""
    curves = _array(objective.curves)
    anchors = np.arange(ANCHORS, dtype=np.float64) / ANCHORS
    fixed = np.r_[anchors, anchors + .37 / ANCHORS, -.003, .999, .9997, 1.0003]
    phase = np.repeat(fixed[:, None], 6, axis=1)
    previous = numpy_interpolate(curves, phase)
    projected = objective.project(_tensor(previous))
    error = np.abs(_array(projected["phase"]) - np.remainder(phase, 1))
    error = np.minimum(error, 1 - error)
    expected = numpy_interpolate(curves, phase + objective.phase_increment)
    successor = objective.interpolate(projected["phase"] + objective.phase_increment)
    successor_error = float(np.max(np.abs(_array(successor) - expected)))
    anchor_phase = np.repeat(anchors[:, None], 6, axis=1)
    q_anchor = numpy_interpolate(curves, anchor_phase)
    correct = objective.evaluate_pairs(_tensor(q_anchor), _tensor(
        numpy_interpolate(curves, anchor_phase + objective.phase_increment)))
    static = objective.evaluate_pairs(_tensor(q_anchor), _tensor(q_anchor))
    reverse = objective.evaluate_pairs(_tensor(q_anchor), _tensor(
        numpy_interpolate(curves, anchor_phase - objective.phase_increment)))
    scrambled = anchor_phase[(np.arange(ANCHORS) * 37 + 11) % ANCHORS]
    scramble = objective.evaluate_pairs(_tensor(q_anchor), _tensor(
        numpy_interpolate(curves, scrambled + objective.phase_increment)))
    shifted_phase = anchor_phase.copy()
    shifted_phase[:, [0, 3]] += .125
    shifted = objective.evaluate_pairs(_tensor(numpy_interpolate(curves, shifted_phase)), _tensor(
        numpy_interpolate(curves, shifted_phase + objective.phase_increment)))
    arbitrary = {}
    for start in (.1234, .9997):
        p = np.repeat((start + np.arange(32) * objective.phase_increment)[:, None], 6, axis=1)
        arbitrary[str(start)] = scores(objective.evaluate_pairs(
            _tensor(numpy_interpolate(curves, p)),
            _tensor(numpy_interpolate(curves, p + objective.phase_increment))))
    results = {"correct": scores(correct), "static_all_128_anchors": scores(static),
               "reverse": scores(reverse), "noncyclic_scramble": scores(scramble),
               "one_lr_pair_shift_0_125_cycle": scores(shifted), "arbitrary_starts": arbitrary}
    good = results["correct"]["training_loss"]
    gates = {
        "phase_round_trip": bool(error.max() <= PHASE_TOLERANCE),
        "successor_round_trip": successor_error <= SUCCESSOR_TOLERANCE_RAD,
        "correct_cycles": good <= LOSS_TOLERANCE and all(
            row["training_loss"] <= LOSS_TOLERANCE for row in arbitrary.values()),
        "static_anchor_normalization": bool(np.allclose(
            results["static_all_128_anchors"]["per_leg_successor_loss"], 1, rtol=0, atol=LOSS_TOLERANCE)),
        "static_distinguished": results["static_all_128_anchors"]["training_loss"] > good + DISTINGUISH_MARGIN,
        "reverse_distinguished": results["reverse"]["training_loss"] > good + DISTINGUISH_MARGIN,
        "noncyclic_scramble_distinguished": results["noncyclic_scramble"]["training_loss"] > good + DISTINGUISH_MARGIN,
        "pair_shift_distinguished": results["one_lr_pair_shift_0_125_cycle"]["coordination_loss"] > DISTINGUISH_MARGIN,
        "lr_pair_shift_invariant": abs(results["one_lr_pair_shift_0_125_cycle"]["lr_coordination_loss"]
                                       - results["correct"]["lr_coordination_loss"]) <= LOSS_TOLERANCE,
    }
    return {"pass": all(gates.values()), "gates": gates,
            "sample_count": len(fixed), "anchors": ANCHORS,
            "interior_offset_in_segments": .37, "wrap_phases": [-.003, .999, .9997, 1.0003],
            "phase_round_trip_max_cycle": float(error.max()),
            "successor_round_trip_max_rad": successor_error, "scores": results,
            "tolerances": {"phase_cycle": PHASE_TOLERANCE, "successor_rad": SUCCESSOR_TOLERANCE_RAD,
                           "loss": LOSS_TOLERANCE, "distinguish_margin": DISTINGUISH_MARGIN}}


def characterize(paths: dict, pins: dict) -> dict:
    """Pure saved/generated CPU arithmetic; caller has already frozen bytes."""
    # Validate every saved clock/shape before doing reference qualification or
    # evaluating either arm. Merely having 301 rows does not prove time order.
    saved = {role: validate_saved(load_npz(paths[role], (
        "control_tick", "network_tick", "physics_time_s", "command", "tendon")))
             for role in ("neutral", "rejected")}
    startup = load_npz(paths["startup"], ("target_command", "target_tendon"))
    q0 = _f64(startup["target_command"], (99, 42), "startup target_command")
    t0 = _f64(startup["target_tendon"], (99, 6), "startup target_tendon")
    curves = curve128(load_npz(paths["reference"], ("time_s", "phase", "command")))
    objective = PhaseFreeTemporalObjective(curves, q0, t0, device="cpu")
    qualification = qualify_reference(objective)
    arms = {}
    for role, experiment in (("neutral", "E318"), ("rejected", "E327")):
        q, tendon = saved[role]
        result = objective.evaluate(_tensor(q[1:]), _tensor(tendon[1:]))
        # Both previous control100 and next controls101..300 are actual inputs
        # for E330. Startup-only ambiguity is reported separately, not gated.
        all_projection = projection_diagnostics(objective.project(_tensor(q)))
        steady_projection = projection_diagnostics(objective.project(_tensor(q[100:])))
        window_scores = [scores(objective.evaluate_pairs(
            _tensor(q[c:c + 4]), _tensor(q[c + 1:c + 5]))) for c in range(100, 300, 4)]
        values = scores(result)
        window_mean = {name: float(np.mean([row[name] for row in window_scores]))
                       for name in ("training_loss", "successor_loss", "coordination_loss")}
        if not all(np.isclose(window_mean[name], values[name], rtol=1e-10, atol=1e-10)
                   for name in window_mean):
            raise ValueError("50-window values disagree with 200-pair values")
        arms[role] = {"experiment": experiment, "classification": role, "candidate_adopted": False,
                      "scores": values, "all_controls_projection": all_projection,
                      "steady_pair_inputs_projection": steady_projection,
                      "steady_input_controls": [100, 300], "steady_next_controls": [101, 300],
                      "conditional_50_window_mean": window_mean,
                      "conditional_window_value_equivalence": True,
                      "conditional_window_gradient_equivalence_claimed": False}
    identifiable = all(arm["steady_pair_inputs_projection"]["nonlocal_tie_count"] == 0
                       for arm in arms.values())
    return {"schema": "flylab.m3.e329.temporal.saved-characterization.v1", "pins": pins,
            **reference_metadata(objective),
            "frequency_hz": FREQUENCY_HZ, "control_dt_s": CONTROL_DT_S,
            "phase_increment_cycle": objective.phase_increment,
            "successor_normalization_mse_rad": _array(objective.successor_normalization).tolist(),
            "reference_qualification": qualification, "arms": arms,
            "descriptive_training_loss_order": sorted(arms, key=lambda role: arms[role]["scores"]["training_loss"]),
            "ranking_is_acceptance_gate": False, "reference_phase_is_biological_observation": False,
            "all_pair_teacher_preference_may_exclude_other_good_gaits": True,
            "valid_input_nonlocal_identifiability_pass": identifiable,
            "e330_input_qualification_pass": qualification["pass"] and identifiable,
            "e330_original_execution_authorized": False,
            "original_calls": 0, "model_calls": 0, "body_calls": 0, "CUDA_calls": 0,
            "optimizer_steps": 0, "biological_pass": False, "F1_F8_pass": False,
            "candidate_adopted": False, "natural_gait_success": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    began = time.perf_counter()
    paths, pins = verify_contract(args.contract)
    output = args.output.resolve()
    protected = {args.contract.resolve(), *paths.values(), *(ROOT / name for name in OWNED_SOURCES)}
    if output in protected or output.exists():
        raise ValueError("output must be a new path outside all protected inputs/sources")
    torch.set_num_threads(1)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = characterize(paths, pins)
    # Detect a concurrent source/input change as well as gating the first read.
    _, after = verify_contract(args.contract)
    if after != pins:
        raise ValueError("input/source bytes changed during characterization")
    result["elapsed_s"] = time.perf_counter() - began
    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    if len(serialized.encode("utf8")) > 20 * 1024 * 1024:
        raise ValueError("characterization exceeds the fixed 20MiB output limit")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf8") as stream:
        stream.write(serialized + "\n")
    print(str(output))


if __name__ == "__main__":
    main()
