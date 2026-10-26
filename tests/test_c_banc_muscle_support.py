"""Generated-array checks for E199 saved muscle support."""

import json

import numpy as np
import pytest

from tools import analyze_c_banc_m3_muscle_support as e199


def generated():
    ids = np.array([70, 10, 40, 90, 20, 60, 30, 80], np.int64)
    owner = np.array([0, 0, 1, 2, 3, 4, 5, 5], np.int64)
    pool = np.zeros((6, 8))
    pool[owner, np.arange(8)] = 1
    moment = np.zeros((42, 6))
    moment[np.arange(6), np.arange(6)] = 1
    adapter = {"motor_ids": ids.copy(), "membership": owner, "pool": pool,
               "moment": moment, "ltm": np.arange(6, dtype=np.int64)}
    muscles = {"motor_roster": [f"root-{i}" for i in ids],
               "muscles": [{"leg": "lf", "muscle": f"muscle-{j}",
                            "columns": np.flatnonzero(owner == j).tolist()}
                           for j in range(6)]}
    rate_max = np.array([0, 1, 0, 0, 1, 1, 1, 0], float)
    r = np.repeat(rate_max[None, :, None], 3, axis=0)
    r = np.repeat(r, 5, axis=2)
    r[1, 5] = 0  # Startup silence, followed by gait activity.
    phi = r.copy()
    active = (r[:, :, -1] > 0).astype(float)
    operating = {"motor_ids": ids.copy(),
                 "window_names": np.array(["all", "startup", "gait"]),
                 "r_quantiles": r, "phi_quantiles": phi,
                 "active_fraction": active}
    activation = np.ones((2001, 6))
    activation[0] = 0
    activation[:, 1] = 0
    activation[:, 2] = 0
    native = {"control_tick": np.arange(2001, dtype=np.int64),
              "network_tick": 20 * np.arange(2001, dtype=np.int64),
              "physics_time_s": .005 * np.arange(2001),
              "activation": activation,
              "motor_rate": np.zeros((2001, 8))}
    for key in ("complete", "activation_available", "motor_rate_available"):
        native[key] = np.ones(2001, bool)
    operating_result = {"run_status": "completed", "validity": "valid_saved_operating_range",
                        "motor_count": 8, "quantile_method": "NumPy linear"}
    operating_review = {"validity": "valid_saved_diagnostic",
                        "independent_checks": 43, "original_attempts": 1, "retries": 0}
    return operating, adapter, muscles, native, operating_result, operating_review


def test_noncontiguous_ownership_and_exact_zero_reporting():
    args = generated()
    result = e199.analyze_arrays(*args)
    all_window = result["windows"]["all"]
    assert result["muscle_count"] == 6
    assert len(all_window["muscles"]) == 6
    assert (all_window["all_member_rate_zero_count"],
            all_window["some_member_rate_zero_count"],
            all_window["no_member_rate_zero_count"]) == (2, 2, 2)
    assert all_window["muscles"][0]["zero_rate_motor_ids"] == [70]
    assert all_window["muscles"][0]["zero_rate_root_ids"] == ["root-70"]
    assert all_window["muscles"][0]["joint_rows"] == [0]
    assert all_window["muscles"][1]["activation_quantiles"] == [0] * 5
    assert all_window["muscles"][1]["activation_exact_zero_samples"] == 2000
    assert result["hypothesis_verdict"] == "some_muscles_without_saved_activation"
    json.dumps(result, allow_nan=False)


def test_all_window_contradiction_and_startup_residual_are_distinct():
    args = generated()
    args[3]["activation"][1:, 2] = 2
    result = e199.analyze_arrays(*args)
    assert result["hypothesis_verdict"] == "inconclusive_zero_initial_state_contradiction"
    assert [x["muscle_index"] for x in result["windows"]["all"][
        "all_zero_rate_with_positive_activation"]] == [2]
    assert [x["muscle_index"] for x in result["windows"]["startup"][
        "all_zero_rate_with_positive_activation"]] == [2, 4]
    assert [x["muscle_index"] for x in result["windows"]["gait"][
        "all_zero_rate_with_positive_activation"]] == [2]
    assert [x["muscle_index"] for x in result["windows"]["all"][
        "zero_initial_contradictions"]] == [2]
    assert [x["muscle_index"] for x in result["windows"]["startup"][
        "zero_initial_contradictions"]] == [2, 4]
    assert result["windows"]["startup"]["filter_residual_possible"] is False
    assert result["windows"]["gait"]["zero_initial_contradictions"] == []
    assert result["windows"]["gait"]["filter_residual_possible"] is True


def test_nonzero_initial_state_is_recorded_as_possible_carryover():
    args = generated()
    args[3]["activation"][:, 2] = 2
    result = e199.analyze_arrays(*args)
    assert result["hypothesis_verdict"] == "inconclusive_initial_filter_state"
    assert result["windows"]["all"]["zero_initial_contradictions"] == []
    assert result["windows"]["all"]["filter_residual_possible"] is True
    assert result["windows"]["all"]["muscles"][2]["initial_activation"] == 2


def test_ownership_and_availability_fail_closed():
    for fault in ("owner", "columns", "available", "motor_order"):
        args = generated()
        if fault == "owner":
            args[1]["pool"][0, 2] = 1
        elif fault == "columns":
            args[2]["muscles"][0]["columns"] = [0]
        elif fault == "available":
            args[3]["activation_available"][17] = False
        else:
            args[1]["motor_ids"][0] = 999
        with pytest.raises(ValueError):
            e199.analyze_arrays(*args)


def test_review_hash_gate_rejects_before_numeric_load(tmp_path, monkeypatch):
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}", encoding="utf-8")
    review = tmp_path / "review.json"
    review.write_text("{}", encoding="utf-8")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("numeric array loaded")
    monkeypatch.setattr(e199.np, "load", forbidden)
    with pytest.raises(ValueError, match="freeze SHA mismatch"):
        e199.run(freeze, "0" * 64, review, "0" * 64, tmp_path / "output")
    error = json.loads((tmp_path / "output" / "error.json").read_text(encoding="utf-8"))
    assert error["phase"] == "hash_gate"
    assert error["loaded_input_roles"] == []
    assert error["original_calculation_count"] == 0
