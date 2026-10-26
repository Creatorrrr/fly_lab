"""Generated E166 fixtures only; original E165/E163 arrays are never opened."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import analyze_c_banc_m3_matched_direction as m

ROOT = Path(__file__).resolve().parents[1]
REAL_FREEZE = (
    ROOT
    / "verification/banc-natural-gait-m3-20260919/e166-matched-direction-freeze.json"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def generated(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, np.ndarray], dict[str, np.ndarray]]:
    c = json.loads(REAL_FREEZE.read_text(encoding="utf-8"))
    n = 2000
    tick = np.arange(1, n + 1)
    phase = np.full((n + 1, 6), np.nan)
    available = np.zeros((n + 1, 6), bool)
    available[100:1924] = True
    phase[100:1924] = 0.2
    phase[101] = 0.99
    phase[102] = 0.01
    d = {
        "tick": tick,
        "time_s": tick * 0.005,
        "z_mm": np.full((n, 3, 6, 5), -0.1),
        "normal_load_bw": np.full((n, 6), 0.04),
        "ego_ap_velocity_mm_s": np.full((n, 6), 1.0),
        "speed_mm_s": np.full(n, 0.2),
        "completed": np.ones(n, bool),
        "forward_completed": np.ones(n, bool),
        "jacobian_completed": np.ones((n, 6, 5), bool),
    }
    d["speed_mm_s"][101] = 0.6  # control tick 102
    d["speed_mm_s"][109] = 0.6  # second control tick 110
    p = {
        "control_time_s": np.arange(n + 1) * 0.005,
        "e1_phase": phase,
        "e1_phase_available": available,
    }
    direction, phase_path = tmp_path / "direction.npz", tmp_path / "phase.npz"
    np.savez(direction, **d)
    np.savez(phase_path, **p)
    layout = tmp_path / "layout.json"
    write_json(
        layout,
        {
            "tarsal_body_names": [
                [f"{leg}_tarsus{i}" for i in range(1, 6)] for leg in m.LEGS
            ]
        },
    )
    dr, pr = tmp_path / "direction-review.json", tmp_path / "phase-review.json"
    write_json(dr, {"schema": "flylab.m3.e165.parent-final-review.v1", "valid": True})
    write_json(pr, {"original_execution_valid": True, "original_processes": 1})
    design = ROOT / c["design"]
    c["paths"] = {
        "direction": str(direction),
        "phase": str(phase_path),
        "layout": str(layout),
        "direction_review": str(dr),
        "phase_review": str(pr),
    }
    c["input_sha256"] = {str(v): digest(Path(v)) for v in c["paths"].values()}
    c["reused_source_sha256"] = {}
    c["preservation_sha256"] = {}
    c["resource_limits"]["internal_s"] = 60
    c["resource_limits"]["free_bytes"] = 0
    c["design"] = str(design)
    freeze_path = tmp_path / "freeze.json"
    write_json(freeze_path, c)
    source = {name: digest(ROOT / name) for name in m.OWNED}
    gate = {
        "schema": m.GATE_SCHEMA,
        "approved": True,
        "blocking_defects": [],
        "accepts_single_matched_direction": True,
        "original_processes_allowed": 1,
        "automatic_retries": 0,
        "freeze_sha256": digest(freeze_path),
        "design_sha256": c["design_sha256"],
        "source_sha256": source,
    }
    gate_path = tmp_path / "source-review.json"
    write_json(gate_path, gate)
    return freeze_path, gate_path, d, p


def execute(freeze: Path, gate: Path, out: Path) -> dict[str, object]:
    return m.run(freeze, digest(freeze), gate, digest(gate), out)


def test_match_leg_circular_inclusive_tie_reuse_unmatched() -> None:
    c = json.loads(REAL_FREEZE.read_text(encoding="utf-8"))["sample_contract"]
    tick = np.array([100, 101, 102, 103, 104, 105])
    time_s = tick * 0.005
    phase = np.array([0.99, 0.01, 0.01, 0.99, 0.99, 0.3])
    load = np.full(6, 0.04)
    ap = np.full(6, 1.0)
    z = np.arange(30.0).reshape(6, 5)
    case = np.array([1, 0, 0, 1, 1, 1], bool)
    control = np.array([0, 1, 1, 0, 0, 0], bool)
    pairs = m.match_leg(tick, time_s, phase, load, ap, z, case, control, c)
    assert pairs["control_tick"].tolist() == [101, 101, 101, -1]
    assert pairs["eligible_control_count"].tolist() == [2, 2, 2, 0]
    assert np.isclose(pairs["score"][0], 0.64)
    assert np.isnan(pairs["score"][-1])
    summary = m.summarize_pairs(
        pairs, z[case], z[control], tick[control], z[control], c, decision=True
    )
    assert summary["control_reuse_count_by_tick"] == {"101": 3}
    assert summary["decision"] == "inconclusive"


@pytest.mark.parametrize(
    "field,caliper", [("phase", 0.025), ("load", 0.02), ("ap", 0.5)]
)
def test_exact_caliper_and_nextafter_outside(field: str, caliper: float) -> None:
    c = json.loads(REAL_FREEZE.read_text(encoding="utf-8"))["sample_contract"]
    tick = np.array([100, 101, 102])
    time_s = tick * 0.005
    phase = np.zeros(3)
    load = np.zeros(3)
    ap = np.zeros(3)
    values = {"phase": phase, "load": load, "ap": ap}[field]
    values[1] = caliper
    values[2] = np.nextafter(caliper, np.inf)
    z = np.zeros((3, 5))
    pairs = m.match_leg(
        tick,
        time_s,
        phase,
        load,
        ap,
        z,
        np.array([True, False, False]),
        np.array([False, True, True]),
        c,
    )
    assert pairs["eligible_control_count"].tolist() == [1]
    assert pairs["control_tick"].tolist() == [101]
    assert pairs["score"][0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "case_z,control_z,expected",
    [
        (-1.0, -1.0, "nonpositive_sign_shared"),
        (-1.0, 1.0, "sign_separation_in_matched_subset"),
        (1.0, 1.0, "inconclusive"),
    ],
)
def test_decision_uses_paired_control_sign(
    case_z: float, control_z: float, expected: str
) -> None:
    c = json.loads(REAL_FREEZE.read_text(encoding="utf-8"))["sample_contract"]
    n = 20
    case = np.full((n, 5), case_z)
    control = np.full((n, 5), control_z)
    ticks = np.arange(100, 100 + n)
    pairs = {
        "case_tick": ticks,
        "control_tick": ticks + 100,
        "case_z_mm": case,
        "control_z_mm": control,
        "case_all_nonpositive": np.all(case <= c["zero_mm"], axis=1),
        "control_all_nonpositive": np.all(control <= c["zero_mm"], axis=1),
        "time_difference_s": np.zeros(n),
    }
    result = m.summarize_pairs(
        pairs, case, control, ticks + 100, control, c, decision=True
    )
    assert result["decision"] == expected


def test_generated_full_run_and_independent_cwd_subprocess(tmp_path: Path) -> None:
    freeze, gate, _, _ = generated(tmp_path)
    output = tmp_path / "run"
    result = execute(freeze, gate, output)
    assert result["valid_diagnostic"] is True
    assert (output / "raw-input-partial.npz").is_file()
    assert (output / "aligned-partial.npz").is_file()
    assert (output / "pairs-partial.npz").is_file()
    assert (output / "source-snapshot" / "analyzer.py").is_file()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["common_tick_count"] == 1824
    assert summary["legs"]["lf"]["all"]["matched_count"] > 0
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(ROOT / m.OWNED[0]),
        "--freeze",
        str(freeze),
        "--freeze-sha256",
        digest(freeze),
        "--source-review",
        str(gate),
        "--source-review-sha256",
        digest(gate),
        "--out",
        str(tmp_path / "subprocess"),
    ]
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=80,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (
        json.loads(
            (tmp_path / "subprocess" / "result.json").read_text(encoding="utf-8")
        )["publication_complete"]
        is True
    )


@pytest.mark.parametrize(
    "change,expected",
    [
        ("phase_radians", "Available phase outside cycle"),
        ("phase_unavailable_finite", "Unavailable phase must be NaN"),
        ("clock", "E165/E163 clocks differ"),
        ("shape", "shape differs"),
        ("nan_z", "effective_z_mm contains nonfinite"),
        ("tick_fraction", "direction_tick dtype differs"),
        ("nan_completed", "direction_completed dtype differs"),
        ("nan_available", "phase_e1_phase_available dtype differs"),
        ("numeric_mask", "direction_forward_completed dtype differs"),
        ("extra_common_phase", "Observed common phase mask differs"),
    ],
)
def test_bad_generated_arrays_preserve_partial(
    tmp_path: Path, change: str, expected: str
) -> None:
    freeze, gate, d, p = generated(tmp_path)
    if change == "phase_radians":
        p["e1_phase"][100, 0] = 2.0
    elif change == "phase_unavailable_finite":
        p["e1_phase"][1, 0] = 0.0
    elif change == "clock":
        p["control_time_s"][100] += 0.001
    elif change == "shape":
        d["z_mm"] = d["z_mm"][:, :, :, :4]
    elif change == "tick_fraction":
        d["tick"] = d["tick"].astype(float)
        d["tick"][100] = 101.75
    elif change == "nan_completed":
        d["completed"] = d["completed"].astype(float)
        d["completed"][100] = np.nan
    elif change == "nan_available":
        p["e1_phase_available"] = p["e1_phase_available"].astype(float)
        p["e1_phase_available"][100, 0] = np.nan
    elif change == "numeric_mask":
        d["forward_completed"] = d["forward_completed"].astype(float)
        d["forward_completed"][100] = 2.0
    elif change == "extra_common_phase":
        p["e1_phase"][1970] = 0.2
        p["e1_phase_available"][1970] = True
    else:
        d["z_mm"][100, 2, 0, 0] = np.nan
    direction, phase = tmp_path / "direction.npz", tmp_path / "phase.npz"
    np.savez(direction, **d)
    np.savez(phase, **p)
    f = json.loads(freeze.read_text(encoding="utf-8"))
    f["input_sha256"][str(direction)] = digest(direction)
    f["input_sha256"][str(phase)] = digest(phase)
    write_json(freeze, f)
    g = json.loads(gate.read_text(encoding="utf-8"))
    g["freeze_sha256"] = digest(freeze)
    write_json(gate, g)
    with pytest.raises(ValueError, match=expected):
        execute(freeze, gate, tmp_path / "bad")
    raw_path = tmp_path / "bad" / "raw-input-partial.npz"
    assert raw_path.is_file()
    with np.load(raw_path, allow_pickle=False) as raw:
        if change == "tick_fraction":
            assert raw["direction_tick"][100] == 101.75
        elif change == "nan_completed":
            assert np.isnan(raw["direction_completed"][100])
        elif change == "nan_available":
            assert np.isnan(raw["phase_e1_phase_available"][100, 0])
        elif change == "numeric_mask":
            assert raw["direction_forward_completed"][100] == 2.0
        elif change == "shape":
            assert raw["direction_z_mm"].shape == (2000, 3, 6, 4)
        elif change == "extra_common_phase":
            assert raw["phase_e1_phase_available"][1970].all()
    if change not in {
        "shape",
        "tick_fraction",
        "nan_completed",
        "nan_available",
        "numeric_mask",
    }:
        assert (tmp_path / "bad" / "aligned-partial.npz").is_file()
    if change == "extra_common_phase":
        with np.load(tmp_path / "bad" / "aligned-partial.npz") as aligned:
            assert aligned["observed_common_mask"].sum() == 1825
            assert aligned["common_tick_mask"].sum() == 1824
    assert (tmp_path / "bad" / "failure.json").is_file()


def test_gate_and_hash_reject_before_loading(tmp_path: Path) -> None:
    freeze, gate, _, _ = generated(tmp_path)
    g = json.loads(gate.read_text(encoding="utf-8"))
    g["approved"] = False
    write_json(gate, g)
    with pytest.raises(ValueError, match="blocking defects"):
        execute(freeze, gate, tmp_path / "gate-fail")
    g["approved"] = True
    write_json(gate, g)
    (tmp_path / "direction.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen file changed"):
        execute(freeze, gate, tmp_path / "hash-fail")


def test_snapshot_and_final_resource_failure_keep_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, gate, _, _ = generated(tmp_path)
    original = m.snapshot

    def corrupt(*args: object, **kwargs: object) -> dict[str, str]:
        approved = original(*args, **kwargs)
        out = args[0]
        assert isinstance(out, Path)
        (out / "source-snapshot" / "tests.py").write_bytes(b"corrupted")
        return approved

    monkeypatch.setattr(m, "snapshot", corrupt)
    with pytest.raises(ValueError, match="Post snapshot changed"):
        execute(freeze, gate, tmp_path / "snapshot-fail")
    assert (tmp_path / "snapshot-fail" / "failure.json").is_file()
    monkeypatch.setattr(m, "snapshot", original)
    f = json.loads(freeze.read_text(encoding="utf-8"))
    f["resource_limits"]["raw_bytes"] = 1
    write_json(freeze, f)
    g = json.loads(gate.read_text(encoding="utf-8"))
    g["freeze_sha256"] = digest(freeze)
    write_json(gate, g)
    with pytest.raises(ValueError, match="Final output byte limit"):
        execute(freeze, gate, tmp_path / "resource-fail")
    assert (tmp_path / "resource-fail" / "pairs-partial.npz").is_file()
    assert not (tmp_path / "resource-fail" / "result.json").exists()
    assert (tmp_path / "resource-fail" / "failure.json").is_file()


def test_post_hash_and_pair_save_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, gate, _, _ = generated(tmp_path)
    original_verify = m.verify_hashes
    calls = 0

    def fail_post(hashes: dict[str, str]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic post hash changed")
        original_verify(hashes)

    monkeypatch.setattr(m, "verify_hashes", fail_post)
    with pytest.raises(ValueError, match="synthetic post hash changed"):
        execute(freeze, gate, tmp_path / "posthash-fail")
    assert (tmp_path / "posthash-fail" / "pairs-partial.npz").is_file()
    assert not (tmp_path / "posthash-fail" / "result.json").exists()
    monkeypatch.setattr(m, "verify_hashes", original_verify)
    original_save = m.atomic_npz

    def fail_pairs(path: Path, arrays: dict[str, np.ndarray]) -> None:
        if path.name == "pairs-partial.npz":
            raise OSError("synthetic pair save failure")
        original_save(path, arrays)

    monkeypatch.setattr(m, "atomic_npz", fail_pairs)
    with pytest.raises(OSError, match="synthetic pair save failure"):
        execute(freeze, gate, tmp_path / "save-fail")
    assert (tmp_path / "save-fail" / "aligned-partial.npz").is_file()
    assert (tmp_path / "save-fail" / "failure.json").is_file()


@pytest.mark.parametrize(
    "failure", ["second_match", "first_summary", "second_pair_save"]
)
def test_completed_pair_windows_survive_later_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    freeze, gate, _, _ = generated(tmp_path)
    if failure == "second_match":
        original_match = m.match_leg
        calls = 0

        def match_then_fail(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic second match failure")
            return original_match(*args, **kwargs)

        monkeypatch.setattr(m, "match_leg", match_then_fail)
    elif failure == "first_summary":

        def summary_fail(*args: object, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("synthetic summary failure")

        monkeypatch.setattr(m, "summarize_pairs", summary_fail)
    else:
        original_save = m.atomic_npz
        pair_saves = 0

        def save_then_fail(path: Path, arrays: dict[str, np.ndarray]) -> None:
            nonlocal pair_saves
            if path.name == "pairs-partial.npz":
                pair_saves += 1
                if pair_saves == 2:
                    raise OSError("synthetic second pair save failure")
            original_save(path, arrays)

        monkeypatch.setattr(m, "atomic_npz", save_then_fail)
    output = tmp_path / failure
    with pytest.raises((RuntimeError, OSError), match="synthetic"):
        execute(freeze, gate, output)
    with np.load(output / "pairs-partial.npz", allow_pickle=False) as pairs:
        assert "lf_w0_case_tick" in pairs.files
        if failure == "first_summary":
            assert "lf_w2_case_tick" in pairs.files
        else:
            assert "lf_w1_case_tick" not in pairs.files
    assert (output / "raw-input-partial.npz").is_file()
    assert (output / "failure.json").is_file()
    assert not (output / "result.json").exists()


def test_publication_late_time_failure_invalidates_receipt(tmp_path: Path) -> None:
    out = tmp_path / "publication"
    out.mkdir()
    times = iter([0.0, 0.1, 2.0])
    limits = {"raw_bytes": 1 << 20, "free_bytes": 0, "internal_s": 1.0}
    with pytest.raises(ValueError, match="Post-publication time exceeded"):
        m.publish_success(
            out, {"schema": "test"}, limits, 0.0, clock=lambda: next(times)
        )
    assert not (out / "result.json").exists()
    assert (out / "result.json.invalidated").exists()
