"""Generated-only checks for E263 saved arithmetic and receipts."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from tools import analyze_c_banc_m3_fixed_input_command as e263


def fixture() -> dict:
    ids = np.arange(391, dtype=np.int64)
    membership = np.arange(391, dtype=np.int64) % 96
    membership[membership == 44] = 45
    membership[[0, 1]] = 42
    membership[[2, 3]] = 43
    membership[[68, 107]] = 44
    pool = np.zeros((96, 391), np.float64)
    pool[membership, ids] = 1
    pool /= pool.sum(axis=1, keepdims=True)
    # The original pool is normalized per muscle; six owner columns are enough
    # to exercise antagonistic positive and negative slopes.
    moment = np.zeros((42, 96), np.float64)
    moment[19, [42, 43, 44]] = [-1.8088900095777785,
                                 -1.8088900095777785, 1.8088900095777785]
    rest = np.zeros(42, np.float64)
    tau = np.array([.005, .01, .04], np.float64)
    kappa = np.full(391, .2, np.float64)
    slopes = np.zeros((96, 391, 3, 2), np.float64)
    slopes[membership, ids, :, 0] = .9
    slopes[membership, ids, :, 1] = .35
    bank0 = np.full((391, 3), .1, np.float64)
    base = np.full((101, 391), .4, np.float32)
    changed = base.copy()
    changed[1:, [0, 68, 107]] += .12
    x = changed.copy()
    x[1:, 2] += .01
    rates = {"B": base, "C": changed, "X": x}
    calibration = {"motor_ids": ids, "membership": membership,
                   "pool": pool, "moment": moment, "rest": rest}
    params = {"motor_ids": ids, "membership": membership[[68, 107]],
              "motor_columns": np.array([68, 107], np.int64),
              "bank0": bank0[[68, 107]], "activation0": np.float64(0),
              "tau": tau, "kappa": kappa[[68, 107]],
              "slopes": slopes[44, [68, 107]]}
    inputs = {"calibration": calibration, "e260_decoder": params,
              "tau": tau, "kappa": kappa, "slopes": slopes,
              "membership": membership, "fixed_muscles": np.arange(1, 20, dtype=np.int64),
              "e260_prior_audit": {"all_passed": True},
              "e260_final": {"all_passed": True},
              "e262_final": {"all_passed": True}}
    alpha = -np.expm1(-.005/tau)
    generated = {}
    for arm, rate in rates.items():
        bank = bank0.copy()
        banks = np.empty((301, 391, 3))
        activation = np.zeros((301, 96))
        commands = np.zeros((301, 42))
        for control in range(101):
            if control:
                bank += alpha[None, :]*(rate[control, :, None]-bank)
            banks[control] = bank
            activation[control, [42, 43, 44]] = e263.transfer(bank, kappa, slopes, pool)
            commands[control, 19] = .02*moment[19]@activation[control]
        banks[101:] = bank
        activation[101:] = activation[100]
        commands[101:] = commands[100]
        commands[0, 19] = 7  # held command is intentionally different
        data = {"motor_rate": rate, "bank44": banks[:101, [68, 107]],
                "activation44": activation[:101, 44]}
        if arm != "X":
            whole_rate = np.vstack([rate, np.tile(rate[-1], (200, 1))])
            clocks = {"control_tick": np.arange(301, dtype=np.int64),
                      "network_tick": 20*np.arange(301, dtype=np.int64),
                      "physics_time_s": .005*np.arange(301, dtype=np.float64)}
            data["control"] = {**clocks, "motor_rate32": whole_rate,
                               "activation": activation, "command": commands}
            data["native"] = {**clocks, "complete": np.ones(301, bool),
                              "bank_available": np.ones(301, bool),
                              "activation_available": np.ones(301, bool),
                              "motor_rate_available": np.ones(301, bool),
                              "bank": banks, "activation": activation,
                              "original_activation": np.zeros((301, 96)),
                              "motor_rate": whole_rate}
        inputs[arm] = data
        generated[arm] = (activation, commands)
    ba, bq = generated["B"]
    ca, cq = generated["C"]
    contribution = .02*(ca-ba)[:, None, :]*moment[None, :, :]
    difference = cq-bq
    target = np.zeros((99, 42))
    inputs["e258_raw"] = {
        "muscle_joint_contribution": contribution,
        "difference_command": difference,
        "startup_baseline_error": bq[1:100]-target,
        "startup_candidate_error": cq[1:100]-target,
        "baseline_command_reconstruction": bq,
        "candidate_command_reconstruction": cq,
    }
    return inputs


def test_alignment_antagonism_and_row_zero() -> None:
    inputs = fixture()
    analysis, raw = e263.analyze_arrays(inputs)
    assert raw["B_q19"].shape == (101,)
    assert raw["T_contribution"].shape == (101, 3)
    assert raw["B_row0_held_minus_decoder"] != 0
    assert np.allclose(raw["T_q19"][1:], raw["C_q19"][1:]-raw["B_q19"][1:])
    assert raw["T_contribution"][56, 0] < 0
    assert raw["T_contribution"][56, 2] > 0
    assert raw["T_positive"][56, 2] > 0
    assert raw["T_negative"][56, 0] < 0
    assert analysis["cases"]["56"]["virtual_X_error_rad"] == pytest.approx(
        raw["virtual_X_error"][55])


def test_decreasing_high_slope_and_boundary() -> None:
    inputs = fixture()
    pool = inputs["calibration"]["pool"]
    slopes = inputs["slopes"]
    kappa = inputs["kappa"]
    below = np.full((391, 3), .1)
    kink = np.full((391, 3), .2)
    above = np.full((391, 3), .3)
    f0 = e263.transfer(below, kappa, slopes, pool)
    f1 = e263.transfer(kink, kappa, slopes, pool)
    f2 = e263.transfer(above, kappa, slopes, pool)
    assert np.all(f1 > f0)
    assert np.all(f2 > f1)
    assert np.all((f2-f1) < (f1-f0))


def test_zero_denominator_and_invalid_identity() -> None:
    assert e263.classify_ratio(0, 0) is None
    assert e263.classify_ratio(1, 1e-12) is None
    inputs = fixture()
    inputs["fixed_muscles"][0] = 44
    with pytest.raises(ValueError, match="fixed muscle"):
        e263.analyze_arrays(inputs)
    inputs = fixture()
    inputs["C"]["motor_rate"][2, 0] += 1
    with pytest.raises(ValueError, match="motor rate"):
        e263.analyze_arrays(inputs)


def test_partial_raw_retained() -> None:
    inputs = fixture()
    inputs["X"]["activation44"][60] += 1
    raw = {}
    with pytest.raises(ValueError, match="activation44"):
        e263.analyze_arrays(inputs, raw=raw)
    assert "B_q19" in raw and "C_q19" in raw and "X_q19" in raw


def test_deadline_retains_completed_prefix_and_actual_updates() -> None:
    raw = {}
    counts = {"budget": 0, "updates": 0}

    def budget():
        counts["budget"] += 1
        if counts["budget"] == 2:
            raise TimeoutError("generated control16 deadline")

    def updated(_arm, _control):
        counts["updates"] += 1

    with pytest.raises(TimeoutError, match="control16"):
        e263.analyze_arrays(fixture(), raw=raw, check_budget=budget,
                            on_bank_update=updated)
    assert counts["updates"] == 16
    assert raw["B_completed_endpoints"].item() == 17
    assert np.isfinite(raw["B_q19"][:17]).all()
    assert np.isnan(raw["B_q19"][17:]).all()
    assert "C_q19" not in raw


def test_zero_denominator_is_inconclusive() -> None:
    inputs = fixture()
    inputs["C"] = copy.deepcopy(inputs["B"])
    for key in ("motor_rate", "bank44", "activation44"):
        inputs["X"][key] = inputs["B"][key].copy()
    e258 = inputs["e258_raw"]
    e258["muscle_joint_contribution"].fill(0)
    e258["difference_command"].fill(0)
    e258["candidate_command_reconstruction"] = e258["baseline_command_reconstruction"].copy()
    e258["startup_candidate_error"] = e258["startup_baseline_error"].copy()
    analysis, _raw = e263.analyze_arrays(inputs)
    assert analysis["H263_supported"] is None
    assert analysis["hypothesis_status"] == "inconclusive"


def test_late_publication_revokes_nested_verdict(tmp_path, monkeypatch) -> None:
    out = tmp_path / "result"
    out.mkdir()
    original = e263.publish
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("generated publication fault")
        return original(*args, **kwargs)

    monkeypatch.setattr(e263, "publish", fail_once)
    result = {"H263_supported": True, "hypothesis_status": "supported",
              "validity": "valid_saved_diagnostic",
              "analysis": {"H263_supported": True,
                           "hypothesis_status": "supported"}, "finalization_pending": True}
    ledger = {"errors": [], "phase": "publication"}
    error = e263.settle(out, result, ledger, {"row": np.arange(2)}, 1 << 20,
                        float("inf"))
    assert isinstance(error, OSError)
    saved = json.loads((out / "result.json").read_text())
    assert saved["H263_supported"] is None
    assert saved["analysis"]["H263_supported"] is None
    assert saved["hypothesis_status"] == "not_tested"
    assert saved["analysis"]["hypothesis_status"] == "not_tested"
    manifest = json.loads((out / "output-manifest.json").read_text())
    for name, entry in manifest["files"].items():
        assert e263.digest(out / name) == entry["sha256"]
    assert json.loads((out / "errors.json").read_text())[0]["message"] == (
        "generated publication fault")


def test_gate_rejects_before_input_load(monkeypatch) -> None:
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("input read")

    monkeypatch.setattr(e263, "load_inputs", forbidden)
    with pytest.raises(ValueError, match="freeze identity"):
        e263.gate("missing", "wrong", "missing", "wrong")
    assert not called


def test_nan_timeout_rejected_before_output_creation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(e263, "BASE", tmp_path)
    monkeypatch.setattr(e263, "gate", lambda *_args: (
        {"resource_limits": {"internal_s": 10}}, {}))
    out = tmp_path / "e263-fixed-input-command"
    with pytest.raises(ValueError, match="timeout"):
        e263.run("freeze", "sha", "review", "sha", str(out), float("nan"))
    assert not out.exists()
