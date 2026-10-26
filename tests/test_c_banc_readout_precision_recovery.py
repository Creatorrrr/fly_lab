"""Generated fixtures for the E220 saved-rate readout audit."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools import analyze_c_banc_m3_readout_precision_recovery as e220
from tools.analyze_c_banc_m3_readout_precision_recovery import (
    PROFILES,
    assess,
    compute_profiles,
    readout32,
    run,
)


def fixture_start():
    pool = np.zeros((96, 391), np.float64)
    pool[0, 0] = 1.125
    pool[1, 1] = .375
    moment = np.zeros((42, 96), np.float64)
    moment[0, 0] = 2.25
    moment[1, 1] = -.75
    rest = np.linspace(-.15, .15, 42)
    initial = np.zeros(96)
    initial[0], initial[1] = .135791113, .246802221
    ltm = np.array([5, 4, 3, 2, 1, 0], np.int64)
    groups = [{"leg": leg} for leg in ("lf", "lm", "lh", "rf", "rm", "rh")]
    groups += [{"leg": "lf"} for _ in range(90)]
    bank = np.zeros((391, 3))
    slopes = np.zeros((96, 391, 3, 2))
    slopes[0, 0, :, :] = .7
    slopes[1, 1, :, :] = .8
    return {"adapter": {"calibration": {"pool": pool, "moment": moment,
                "rest": rest, "ltm": ltm},
            "parameters": {"membership": np.zeros(391, np.int64),
                "kappa": np.zeros(391), "slopes": slopes,
                "tau": np.array([.005, .01, .04]),
                "fixed_muscles": np.array([2, 3, 4, 5, 6, 7])},
            "original": {"activation": initial}, "activation": initial.copy(),
            "bank": bank, "groups": groups},
            "body": {"action": {"joint_angles": np.full(42, .987654321)}}}


def fixture_rates():
    rates = np.zeros((4, 391), np.float64)
    rates[:, 0] = [.14, .29, .41, .67]
    rates[:, 1] = [.20, .35, .52, .81]
    return rates


def test_float32_step_initial_held_and_tendon_order():
    start, rates = fixture_start(), fixture_rates()
    result = readout32(start, rates)
    assert result["command"].dtype == np.float64
    assert np.array_equal(result["command"][0], np.full(42, .987654321))
    np.testing.assert_array_equal(result["activation"][0],
                                  start["adapter"]["original"]["activation"].astype(np.float32))
    p = np.float32(1.125) * np.float32(.29)
    a0 = np.float32(.135791113)
    alpha = np.float32(-np.expm1(-.005 / .01))
    expected_a = np.float32(a0 + alpha * np.float32(p - a0))
    assert result["activation"][1, 0] == float(expected_a)
    expected_q = np.float32(np.float32(start["adapter"]["calibration"]["rest"][0])
                            + np.float32(.02) * np.float32(np.float32(2.25) * expected_a))
    assert result["command"][1, 0] == float(expected_q)
    # ltm indices are reversed against lf,lm,lh,rf,rm,rh output order.
    expected_tendon = np.float32(.2) * np.tanh(
        np.float32(expected_a / np.float32(10)))
    assert result["tendon"][1, 0] == float(expected_tendon)
    assert result["tendon"][1, -1] == 0


def test_six_profiles_and_rate_roundback():
    start, original = fixture_start(), fixture_rates()
    new = original.copy()
    new[:, 0] += np.array([0, .0123456789, .026543219, .0377777777])
    profiles = compute_profiles(start, original, new)
    assert tuple(profiles) == PROFILES
    assert set(profiles["new_monotone"]) == {"bank", "activation", "command", "tendon"}
    assert set(profiles["readout32"]) == {"activation", "command", "tendon"}
    assert profiles["original_monotone"]["bank"].shape == (4, 391, 3)
    assert profiles["new_monotone"]["command"][1, 0] != profiles["new_canonical"]["command"][1, 0]
    from tools.prepare_c_banc_m3_causal_neural_basis import canonical_readout

    rounded = canonical_readout(start, new.astype(np.float32))
    np.testing.assert_array_equal(profiles["rate32_readout64"]["command"],
                                  rounded["command"])
    assert not np.array_equal(profiles["rate32_readout64"]["command"],
                              profiles["new_canonical"]["command"])


def test_assessment_requires_saved_candidate_restoration():
    start, rates = fixture_start(), fixture_rates()
    profiles = compute_profiles(start, rates, rates)
    trace = {"samples": [{"joint_targets_rad": row.tolist()}
                         for row in profiles["original_monotone"]["command"]]}
    canonical = profiles["original_canonical"]
    candidate = {"q_direct": profiles["new_canonical"]["command"].copy(),
                 "activation": profiles["new_canonical"]["activation"].copy(),
                 "tendon": profiles["new_canonical"]["tendon"].copy()}
    # Tiny generated targets permit the strict comparator while exercising restoration.
    target = {"indices": np.array([1, 2]), "pair_start": np.array([1]),
              "pair_end": np.array([2]),
              "target": profiles["new_canonical"]["command"][[1, 2]],
              "velocity": np.diff(profiles["new_canonical"]["command"][[1, 2]], axis=0) / .005,
              "startup_command": profiles["new_canonical"]["command"][1:100],
              "startup_tendon": profiles["new_canonical"]["tendon"][1:100],
              "baseline": {"velocity_rms": 1., "leg_command_rms": [1.]*6,
                           "leg_velocity_rms": [1.]*6}}
    reports, restoration = assess(profiles, target, trace, canonical, candidate)
    assert reports["new_canonical"]["strict_proxy"]["passed"]
    assert restoration["new_canonical_command_max_abs"] == 0
    candidate["q_direct"][1, 0] += 1e-5
    with pytest.raises(ValueError, match="restoration failed"):
        assess(profiles, target, trace, canonical, candidate)


def test_bad_freeze_rejected_before_input_load(tmp_path):
    freeze = tmp_path / "freeze.json"
    review = tmp_path / "review.json"
    freeze.write_text("{}", encoding="utf-8")
    review.write_text("{}", encoding="utf-8")
    result = run(freeze, "0" * 64, review, "0" * 64, tmp_path / "out")
    assert result["run_status"] == "failed"
    assert result["phase"] == "gate"
    assert result["loaded_input_roles"] == []
    assert result["original_calculation_count"] == 0
    assert (tmp_path / "out" / "failure.json").is_file()


def test_success_manifest_hashes_exact_published_bytes(tmp_path, monkeypatch):
    ids = np.arange(391, dtype=np.int64)
    rates = np.zeros((2001, 391), np.float64)
    np.savez(tmp_path / "endpoints.npz", motor_rate=rates, motor_ids=ids)
    np.savez(tmp_path / "candidate.npz", rate=rates, q_direct=np.zeros((2001, 42)),
             activation=np.zeros((2001, 96)), tendon=np.zeros((2001, 6)))
    np.savez(tmp_path / "canonical.npz", command=np.zeros((2, 42)),
             activation=np.zeros((2, 96)), tendon=np.zeros((2, 6)))
    np.savez(tmp_path / "parameters.npz", motor_ids=ids)
    (tmp_path / "trace.json").write_text("{}", encoding="utf-8")
    (tmp_path / "start").mkdir()
    paths = {"parameters": "parameters.npz", "endpoints": "endpoints.npz",
             "targets": "targets.npz", "trace": "trace.json", "start": "start",
             "candidate": "candidate.npz", "canonical_original": "canonical.npz"}
    freeze = {"paths": paths, "input_sha256": {"generated": "a"},
              "reused_source_sha256": {"generated": "b"},
              "preservation_sha256": {"generated": "c"}}
    monkeypatch.setattr(e220, "verify_gate", lambda *_: (freeze, {"tool_sha256": "abc"}))
    monkeypatch.setattr(e220, "source_path",
                        lambda relative, directory=False: tmp_path / relative)
    monkeypatch.setattr(e220.StateStore, "load",
                        lambda _: {"adapter": {"calibration": {"motor_ids": ids}}})
    monkeypatch.setattr(e220, "load_targets", lambda *_: {"baseline": {}})
    raw = {name: {"command": np.zeros((2, 42)),
                  "activation": np.zeros((2, 96)),
                  "tendon": np.zeros((2, 6))} for name in PROFILES}
    for name in ("original_monotone", "new_monotone"):
        raw[name]["bank"] = np.zeros((2, 391, 3))
    monkeypatch.setattr(e220, "compute_profiles", lambda *_: raw)
    monkeypatch.setattr(e220, "assess",
                        lambda *_: ({name: {"metrics": {}} for name in PROFILES},
                                    {"restored": 0.0}))
    out = tmp_path / "out"
    result = run(tmp_path / "freeze", "f" * 64, tmp_path / "review",
                 "r" * 64, out)
    assert result["run_status"] == "completed"
    manifest = json.loads((out / "output-manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"result.json", "metrics.json",
                                     *(name + ".npz" for name in PROFILES)}
    for name, record in manifest["files"].items():
        published = (out / name).read_bytes()
        assert record == {"sha256": hashlib.sha256(published).hexdigest(),
                          "bytes": len(published)}
    assert manifest["total_bytes"] + (out / "output-manifest.json").stat().st_size < 64 * 1024 * 1024
