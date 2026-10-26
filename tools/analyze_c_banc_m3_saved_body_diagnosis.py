"""E328 saved-only descriptive comparison of E318 and E327 body recordings.

No model, controller, body, CUDA, or original simulation is constructed here.
The joint vector is interpreted in the producer's six contiguous seven-DOF
leg blocks; the trace has no per-joint name roster, so that interpretation is
declared rather than presented as independently measured anatomy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools import analyze_c_banc_biological_gait as gait

LEGS = gait.LEGS
ROOT = Path(__file__).resolve().parents[1]
BASE = Path(__file__).resolve().parents[1] / "verification/banc-natural-gait-m3-20260919"
FILES = ("trace.json", "control-raw.npz", "native-contact.npz")
BINS = ((0.5, 0.75), (0.75, 1.0), (1.0, 1.25), (1.25, 1.5))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def verify_sources(folder: Path, audit_path: Path, role: str) -> dict:
    """Pin the three selected bytes to their own producer manifest and audit."""
    manifest_path = folder / "output-manifest.json"
    result_path = folder / "result.json"
    manifest, result, audit = (read_json(path) for path in
                               (manifest_path, result_path, audit_path))
    expected_schema = f"flylab.m3.{ 'e318' if role == 'baseline' else 'e327'}.output-manifest.v1"
    if manifest.get("schema") != expected_schema or audit.get("pass") is not True:
        raise ValueError(f"{role}: producer manifest or original audit invalid")
    if audit.get("result_sha256") != sha256(result_path) or audit.get(
            "output_manifest_sha256") != sha256(manifest_path):
        raise ValueError(f"{role}: original audit does not pin result and manifest")
    if result.get("run_valid") is not True or result.get("first_error") is not None:
        raise ValueError(f"{role}: producer run invalid")
    if role == "candidate" and (audit.get("diagnostic_run_valid") is not True
                                or audit.get("candidate_adopted") is not False
                                or result.get("classification") != "rejected"):
        raise ValueError("candidate: E327 audit classification invalid")
    pins = {}
    for name in FILES:
        path = folder / name
        pin = manifest.get("files", {}).get(name)
        if not isinstance(pin, dict) or pin.get("bytes") != path.stat().st_size:
            raise ValueError(f"{role}: missing or wrong manifest size for {name}")
        digest = sha256(path)
        if pin.get("sha256") != digest:
            raise ValueError(f"{role}: manifest SHA256 mismatch for {name}")
        pins[name] = {"path": str(path.resolve()), "sha256": digest,
                      "bytes": path.stat().st_size}
    return {"files": pins, "manifest_sha256": sha256(manifest_path),
            "result_sha256": sha256(result_path), "audit_path": str(audit_path.resolve()),
            "audit_sha256": sha256(audit_path), "result": result, "audit": audit}


def validate_clocks(rows: list[dict], raw: dict, native: dict) -> None:
    if len(rows) != 301:
        raise ValueError("trace must contain 301 control rows")
    ticks = np.asarray([r["control_tick"] for r in rows])
    times = np.asarray([r["time_s"] for r in rows], dtype=float)
    expected = np.arange(301)
    if not np.array_equal(ticks, expected) or not np.isfinite(times).all() or not np.allclose(
            times, expected * .005, rtol=0, atol=2e-8):
        raise ValueError("trace control clock incomplete")
    for key, values in (("control_tick", expected), ("network_tick", expected * 20)):
        if not np.array_equal(np.asarray(raw[key]), values):
            raise ValueError(f"raw {key} incomplete")
    raw_time = np.asarray(raw["physics_time_s"], dtype=float)
    native_time = np.asarray(native["time_s"], dtype=float)
    if raw_time.shape != (301,) or not np.isfinite(raw_time).all() or not np.allclose(
            raw_time, expected * .005, rtol=0, atol=2e-8):
        raise ValueError("raw physical clock incomplete")
    if native_time.shape != (15000,) or not np.isfinite(native_time).all() or not np.allclose(
            native_time, np.arange(1, 15001) * .0001, rtol=0, atol=2e-8):
        raise ValueError("native physical clock incomplete")


def validate_joint_arrays(rows: list[dict], raw: dict) -> tuple[np.ndarray, np.ndarray]:
    measured = np.asarray([r["joint_angles_rad"] for r in rows], dtype=float)
    target = np.asarray([r["joint_targets_rad"] for r in rows], dtype=float)
    recorded_error = np.asarray([r["joint_tracking_error_rad"] for r in rows], dtype=float)
    command = np.asarray(raw["command"], dtype=float)
    if any(a.shape != (301, 42) or not np.isfinite(a).all()
           for a in (measured, target, recorded_error, command)):
        raise ValueError("complete finite 301x42 joint arrays required")
    if not np.allclose(recorded_error, target - measured, atol=1e-10, rtol=0):
        raise ValueError("trace joint error does not equal requested minus measured")
    if not np.allclose(target, command, atol=1e-10, rtol=0):
        raise ValueError("raw command and trace target order/value mismatch")
    return measured, command


def moving_runs(data: dict) -> list[dict]:
    t, dt = data["t"], data["dt"]
    return [{"start_s": float(t[lo]), "end_s": float(t[hi - 1]),
             "samples": int(hi - lo), "duration_s": float((hi - lo - 1) * dt),
             "eligible": bool((hi - lo - 1) * dt >= .3 - 1e-9)}
            for lo, hi in gait.runs(data["moving"])]


def support_breakdown(time_s, load, slip, valid) -> dict:
    """Exact selected-load denominator, shared by leg and fixed-bin numerators."""
    time_s = np.asarray(time_s, dtype=float)
    load = np.asarray(load, dtype=float)
    slip = np.asarray(slip, dtype=float)
    valid = np.asarray(valid)
    if (time_s.ndim != 1 or load.shape != (len(time_s), 6)
            or slip.shape != load.shape or valid.shape != load.shape
            or valid.dtype != np.bool_
            or not np.isfinite(time_s).all() or not np.isfinite(load).all()
            or (load < 0).any()):
        raise ValueError("invalid native load/slip shape or load")
    window = (time_s >= .5) & (time_s < 1.5)
    if not window.any():
        raise ValueError("native selected time window empty")
    support = window[:, None] & (load >= .02)
    if np.any(support & (~valid.astype(bool) | ~np.isfinite(slip))):
        raise ValueError("selected support has missing or invalid slip")
    if not np.isfinite(slip[valid]).all():
        raise ValueError("valid native slip nonfinite")
    denominator = float(np.sum(load[support]))
    total = None if denominator <= 0 else float(np.sum(load[support] * slip[support]) / denominator)

    def part(mask):
        selected = support & mask
        count = int(np.count_nonzero(selected))
        weight = float(np.sum(load[selected]))
        numerator = float(np.sum(load[selected] * slip[selected]))
        return {"support_samples": count, "support_load_sum_bw": weight,
                "weighted_slip_mm_s": None if weight == 0 else numerator / weight,
                "contribution_mm_s": None if total is None else numerator / denominator}

    legs = {}
    for i, leg in enumerate(LEGS):
        mask = np.zeros_like(support)
        mask[:, i] = True
        legs[leg] = {**part(mask), "support_duty": float(np.count_nonzero(support[:, i]) / window.sum())}
    bins = {}
    for lo, hi in BINS:
        mask = np.broadcast_to(((time_s >= lo) & (time_s < hi))[:, None], support.shape)
        bins[f"[{lo:g},{hi:g})"] = part(mask)
    if total is not None:
        for collection in (legs, bins):
            if abs(sum(v["contribution_mm_s"] for v in collection.values()) - total) > 1e-12:
                raise ValueError("slip contribution decomposition inconsistent")
    return {"window": "[0.5,1.5)", "threshold_bw": .02,
            "window_time_samples": int(window.sum()), "total_support_samples": int(support.sum()),
            "total_support_load_sum_bw": denominator, "weighted_slip_mm_s": total,
            "legs": legs, "time_bins": bins}


def _close(actual, expected, name):
    if (expected is None or actual is None or not np.isfinite(expected)
            or not np.isfinite(actual) or abs(actual - expected) > 1e-12):
        raise ValueError(f"official {name} mismatch")


def tracking_rms(measured: np.ndarray, command: np.ndarray) -> dict:
    """Whole-vector RMS is independent of any anatomical block assignment."""
    if (measured.shape != command.shape or measured.ndim != 2
            or measured.shape[1] != 42 or not measured.size
            or not np.isfinite(measured).all() or not np.isfinite(command).all()):
        raise ValueError("finite simultaneous 42-DOF measurements required")
    difference = measured - command
    per_leg = np.sqrt(np.mean(difference.reshape(-1, 6, 7) ** 2, axis=(0, 2)))
    return {"global_rms_rad": float(np.sqrt(np.mean(difference ** 2))),
            "source_inferred_per_leg_rms_rad": dict(zip(LEGS, map(float, per_leg)))}


def order_source_evidence(candidate: Path) -> dict:
    """Read source bytes only; local FlyGym files are not original-run pins."""
    local = ("flylab/body_options.py", "flylab/body.py", "flylab/c/walking_trace.py")
    external = ("src/flygym/compose/fly/base_fly.py",
                "src/flygym/compose/fly/flybody.py",
                "src/flygym/flybody/anatomy_flybody.py",
                "src/flygym/anatomy.py", "src/flygym/utils/math.py")
    manifest_path = candidate / "output-manifest.json"
    manifest = read_json(manifest_path)
    source_path = candidate / "source-manifest.json"
    source_pin = manifest.get("files", {}).get("source-manifest.json", {})
    if (source_pin.get("sha256") != sha256(source_path)
            or source_pin.get("bytes") != source_path.stat().st_size):
        raise ValueError("E327 source manifest not pinned by output manifest")
    sources = read_json(source_path).get("source_sha256", {})
    entries = {}
    for name in local:
        path = ROOT / name
        digest = sha256(path)
        if sources.get(name) != digest:
            raise ValueError("current local order source differs from audited E327: " + name)
        entries[name] = {"path": str(path.resolve()), "sha256": digest,
                         "audited_e327_source_pin": True}
    flygym_root = ROOT.parent / "flygym"
    for name in external:
        path = flygym_root / name
        entries["flygym/" + name] = {"path": str(path.resolve()),
                                     "sha256": sha256(path),
                                     "audited_e327_source_pin": False}
    return {"files": entries,
            "interpretation": "local producer sources match E327 audited source manifest; FlyGym library source hashes describe currently inspected source and are not independently original-run pinned"}


def analyze_arm(folder: Path, pinned: dict, official: dict) -> dict:
    rows = read_json(folder / "trace.json")["samples"]
    with np.load(folder / "control-raw.npz", allow_pickle=False) as source:
        raw = {key: source[key] for key in ("control_tick", "network_tick", "physics_time_s", "command")}
    with np.load(folder / "native-contact.npz", allow_pickle=False) as source:
        native = {key: source[key] for key in ("time_s", "normal_load_bw", "slip_mm_s",
                                                  "slip_valid", "upright", "nonfoot_contact")}
    validate_clocks(rows, raw, native)
    measured, command = validate_joint_arrays(rows, raw)
    data = gait.load_model(folder / "trace.json")
    if (any(not np.isfinite(data[key]).all() for key in
            ("t", "ap", "knee", "speed", "yaw_rate", "position", "upright", "lengths"))
            or np.asarray(native["upright"]).shape != (15000,)
            or np.asarray(native["nonfoot_contact"]).shape != (15000,)
            or not np.isfinite(native["upright"]).all()):
        raise ValueError("missing or nonfinite measured body/native data")
    result, _, _ = gait.analyze(data)
    if data["end_s"] != 1.5 or int(data["window"].sum()) != 200:
        raise ValueError("gait analysis did not retain fixed [0.5,1.5) interval")
    runs = moving_runs(data)
    slip = support_breakdown(*(native[key] for key in
                               ("time_s", "normal_load_bw", "slip_mm_s", "slip_valid")))
    if slip["weighted_slip_mm_s"] is None:
        raise ValueError("official comparison requires selected support")
    # The frozen forward statistic projects the first-to-last selected positions
    # onto the initial heading, rather than integrating current-heading speed.
    positions = np.asarray([r["body"]["position"] for r in rows], dtype=float)
    initial_basis = np.asarray(rows[0]["body"]["basis"], dtype=float)
    forward_axis = initial_basis[[0, 2], 0].copy()
    forward_axis /= np.linalg.norm(forward_axis)
    window = data["window"]
    selected_pos = positions[window][:, [0, 2]]
    forward = float((selected_pos[-1] - selected_pos[0]) @ forward_axis)
    cycles = {leg: result["events"][leg]["complete_cycles"] for leg in LEGS}
    healthy = bool(np.isfinite(native["upright"]).all()
                   and float(np.min(native["upright"])) >= .9
                   and not np.any(native["nonfoot_contact"])
                   and all(not r.get("fault") for r in rows))
    _close(forward, official["forward_mm"], "forward")
    _close(slip["weighted_slip_mm_s"], official["slip_mm_s"], "slip")
    _close(result["window"]["longest_forward_bout_s"],
           official["longest_continuous_forward_bout_s"], "bout")
    if cycles != official["complete_cycles"] or healthy is not official["healthy"]:
        raise ValueError("official cycles/health mismatch")
    _close(float(np.min(native["upright"])), official["minimum_native_upright"],
           "minimum native upright")
    if int(np.count_nonzero(native["nonfoot_contact"])) != official["nonfoot_contact_count"]:
        raise ValueError("official nonfoot contact mismatch")
    tracking = tracking_rms(measured[window], command[window])
    knee = {leg: float(np.percentile(data["knee"][window, i], 95)
                       - np.percentile(data["knee"][window, i], 5))
            for i, leg in enumerate(LEGS)}
    if pinned["role"] == "baseline":
        shape = pinned["candidate_result"]["neutral_metrics"]["per_leg_shape_loss"]
    else:
        shape = pinned["result"].get("metrics", {}).get("per_leg_shape_loss")
    if shape is not None and (len(shape) != 6 or not np.isfinite(shape).all()):
        raise ValueError("invalid official command shape per-leg data")
    return {"inputs": pinned["files"], "moving_runs": runs,
            "eligible_run_count": sum(r["eligible"] for r in runs),
            "official_complete_cycles": cycles, "longest_forward_bout_s": result["window"]["longest_forward_bout_s"],
            "forward_mm": forward, "healthy": healthy,
            "measured_geometric_knee_p95_minus_p05_deg": knee,
            "measured_minus_command_rms_rad": tracking,
            "command_shape_loss_per_leg": dict(zip(LEGS, shape)) if shape is not None else None,
            "native_support": slip}


def run(args) -> dict:
    baseline, candidate, audit, output = (Path(value).resolve() for value in
                                           (args.baseline, args.candidate, args.audit, args.output))
    if (baseline == candidate or output.exists() or output.is_relative_to(baseline)
            or output.is_relative_to(candidate)):
        raise ValueError("distinct inputs and fresh output directory required")
    if audit != (BASE / "parent-e327-original-raw-audit.json").resolve():
        raise ValueError("exact root E327 original audit path required")
    baseline_audit = BASE / "parent-e318-original-raw-audit.json"
    pins = {"baseline": verify_sources(baseline, baseline_audit, "baseline"),
            "candidate": verify_sources(candidate, audit, "candidate")}
    candidate_audit = pins["candidate"]["audit"]
    neutral_saved = pins["candidate"]["result"]["saved_neutral"]
    neutral_shape = pins["candidate"]["result"]["neutral_metrics"]
    if (neutral_saved["shape"] != neutral_shape
            or neutral_saved["short"]["gait"]["source"]["sha256"] != pins["baseline"]["files"]["trace.json"]["sha256"]
            or neutral_saved["short"]["gait"]["source"]["path"] != str((baseline / "trace.json").resolve())
            or neutral_shape["shape_loss"] != candidate_audit["neutral_metrics"]["shape_loss"]):
        raise ValueError("E327 saved neutral shape not linked to selected E318 trace/audit")
    pins["baseline"]["role"] = "baseline"
    pins["baseline"]["candidate_result"] = pins["candidate"]["result"]
    pins["candidate"]["role"] = "candidate"
    baseline_official = candidate_audit["neutral_metrics"]
    candidate_official = candidate_audit["metrics"]
    arms = {"baseline": analyze_arm(baseline, pins["baseline"], baseline_official),
            "candidate": analyze_arm(candidate, pins["candidate"], candidate_official)}
    for role in arms:
        if role == "candidate":
            official_shape = pins[role]["result"]["metrics"].get("per_leg_shape_loss")
            if arms[role]["command_shape_loss_per_leg"] != dict(zip(LEGS, official_shape)):
                raise ValueError("candidate command shape provenance mismatch")
    report = {"schema": "flylab.m3.e328.saved-body-diagnosis.v1",
              "scope": "exploratory saved-only comparison after E327 rejection; no biological or causal claim",
              "source_sha256": {"analyzer": sha256(Path(__file__)),
                                "frozen_gait": sha256(Path(gait.__file__))},
              "joint_order_source_evidence": order_source_evidence(candidate),
              "joint_order_interpretation": "FlyBody whole-body producer source constructs leg position actuators from LEGS_ACTIVE_ONLY in skeleton DFS order, with root legs lf,lm,lh,rf,rm,rh and seven actuated DOFs per leg; trace lacks a joint-name roster, so per-leg tracking uses this source-inferred order; target order/value cross-checked against raw command",
              "tracking_sample_interpretation": "simultaneous already-integrated control-boundary measured angle minus held requested command at each selected 5 ms row, all 7 DOFs pooled per leg",
              "inputs": {role: {key: value for key, value in pin.items() if key not in ("result", "audit", "candidate_result", "role")}
                         for role, pin in pins.items()}, "arms": arms,
              "comparison": {"candidate_minus_baseline": {
                  "forward_mm": arms["candidate"]["forward_mm"] - arms["baseline"]["forward_mm"],
                  "slip_mm_s": arms["candidate"]["native_support"]["weighted_slip_mm_s"] - arms["baseline"]["native_support"]["weighted_slip_mm_s"]}}}
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
