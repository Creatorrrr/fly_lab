"""Generated saved-command cases for E200, without original numeric input."""

import json

import numpy as np
import pytest

from tools import analyze_c_banc_m3_silent_muscle_capacity as e200


def generated():
    indices = np.arange(1, 1825, dtype=np.int64)
    command = np.zeros((2001, 42))
    moment = np.zeros((42, 96))
    moment[0, e200.SILENT[0]] = 50  # B column +e0.
    moment[0, e200.SILENT[1]] = 50  # Dependent, nonunique NNLS weights.
    moment[1, e200.SILENT[2]] = -50  # B column -e1.
    target = np.zeros((1824, 42))
    target[0, 0] = 1
    target[1, 0] = -1
    saved = {"target_indices": indices, "target": target,
             "recorded_command": command,
             "loaded_adapter_moment": moment.copy(),
             "loaded_adapter_rest": np.zeros(42)}
    adapter = {"moment": moment, "rest": np.zeros(42)}
    native = {"control_tick": np.arange(2001, dtype=np.int64),
              "network_tick": 20 * np.arange(2001, dtype=np.int64),
              "physics_time_s": .005 * np.arange(2001),
              "activation": np.zeros((2001, 96))}
    for key in ("complete", "activation_available"):
        native[key] = np.ones(2001, bool)
    support = {"validity": "valid_saved_muscle_support", "run_status": "completed",
               "windows": {"all": {"ticks": [1, 2000],
                                   "activation_always_zero": [
                                       {"muscle_index": int(i)} for i in e200.SILENT]}}}
    return saved, native, adapter, support


def test_signed_projection_and_nonnegative_sign_constraint():
    result, raw = e200.analyze_arrays(*generated())
    assert raw["B"].shape == (42, 12)
    assert raw["Y"].shape == (1824, 42)
    assert raw["baseline_command"].shape == (1824, 42)
    assert result["svd_rank"] == 2
    np.testing.assert_allclose(raw["nnls_correction"][0, 0], 1, atol=1e-12)
    np.testing.assert_allclose(raw["nnls_correction"][1, 0], 0, atol=1e-12)
    np.testing.assert_allclose(raw["signed_correction"][1, 0], -1, atol=1e-12)
    np.testing.assert_allclose(raw["nnls_residual"], raw["nnls_correction"] - raw["Y"])
    assert np.min(raw["nnls_activation"]) >= 0
    assert result["hypothesis_verdict"] == "saved_nonnegative_command_headroom"
    assert result["nnls_kkt_gradient_min"] >= -1e-9
    json.dumps(result, allow_nan=False)


def test_signed_span_rejects_unreachable_joint_direction():
    args = generated()
    args[0]["target"][:] = 0
    args[0]["target"][0, 41] = 1
    result, raw = e200.analyze_arrays(*args)
    assert result["hypothesis_verdict"] == "reject_twelve_muscle_twenty_percent_hypothesis"
    assert result["whole"]["signed_span"]["rms_ratio"] == 1
    np.testing.assert_allclose(raw["signed_correction"], 0, atol=1e-12)


def test_initial_held_command_is_reported_but_generated_rows_match():
    args = generated()
    args[0]["recorded_command"][0, 0] = .3266829264
    result, raw = e200.analyze_arrays(*args)
    assert result["initial_command_difference_max_abs"] == .3266829264
    assert result["command_reconstruction_max_abs"] == 0
    assert result["command_reconstruction_rows"] == [1, 2000]
    assert raw["baseline_command"][0, 0] == 0
    assert raw["indices"][0] == 1


def test_reordering_and_saved_semantic_gates():
    for fault in ("indices", "moment", "silent_activation", "command", "roster"):
        args = generated()
        if fault == "indices":
            args[0]["target_indices"][[0, 1]] = args[0]["target_indices"][[1, 0]]
        elif fault == "moment":
            args[0]["loaded_adapter_moment"][0, 0] = 1
        elif fault == "silent_activation":
            args[1]["activation"][8, e200.SILENT[0]] = 1
        elif fault == "command":
            args[0]["recorded_command"][8, 0] = 1
        else:
            args[3]["windows"]["all"]["activation_always_zero"][0]["muscle_index"] = 9
        with pytest.raises(ValueError):
            e200.analyze_arrays(*args)


def test_freeze_hash_rejection_precedes_original_array_load(tmp_path, monkeypatch):
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}", encoding="utf-8")
    review = tmp_path / "review.json"
    review.write_text("{}", encoding="utf-8")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("original numeric array load reached")
    monkeypatch.setattr(e200.np, "load", forbidden)
    with pytest.raises(ValueError, match="freeze SHA mismatch"):
        e200.run(freeze, "0" * 64, review, "0" * 64, tmp_path / "out")
    failure = json.loads((tmp_path / "out" / "error.json").read_text(encoding="utf-8"))
    assert failure["phase"] == "hash_gate"
    assert failure["loaded_input_roles"] == []
    assert failure["original_calculation_count"] == 0
