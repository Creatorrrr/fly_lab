"""Synthetic E238 action-boundary, parity, and partial-evidence checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools import observe_c_banc_m3_matched_teacher_inputs as m

NAMES = ("lf_tarsus", "lm_tarsus", "lh_tarsus", "rf_tarsus", "rm_tarsus",
         "rh_tarsus", "abdomen_left", "abdomen_right")


class Body:
    def __init__(self):
        self.calls = []

    def step_joint_targets(self, command, adhesion, *, tendon_inputs):
        self.calls.append((command.copy(), adhesion.copy(), tendon_inputs.copy()))
        return "advanced_once"


class Session:
    def __init__(self):
        self.body = Body()
        self.tendon_inputs = dict.fromkeys(NAMES, .1)


def test_body_boundary_applies_one_recorded_action_and_restores_method():
    session = Session()
    original = session.body.step_joint_targets
    desired = np.arange(42, dtype=np.float64) / 100
    desired_tendon = np.full(8, .2, np.float64)
    decoded = np.full(42, -.5, np.float64)

    def advance():
        return session.body.step_joint_targets(decoded, np.ones(6, bool),
            tendon_inputs=session.tendon_inputs)

    value, receipt = m.with_applied_action(session, desired, desired_tendon,
                                           NAMES, advance)
    assert value == "advanced_once"
    assert len(session.body.calls) == 1
    applied, adhesion, tendons = session.body.calls[0]
    np.testing.assert_array_equal(applied, desired)
    np.testing.assert_array_equal(adhesion, np.zeros(6, bool))
    assert all(tendons[name] == .2 for name in NAMES)
    np.testing.assert_array_equal(receipt["decoded_command"], decoded)
    np.testing.assert_array_equal(receipt["decoded_tendon"], np.full(8, .1))
    np.testing.assert_array_equal(receipt["applied_command"], desired)
    assert session.tendon_inputs == tendons
    assert "step_joint_targets" not in vars(session.body)
    assert session.body.step_joint_targets == original


def test_body_boundary_preserves_original_exception_and_restores():
    session = Session()

    def advance():
        session.body.step_joint_targets(np.zeros(42), np.zeros(6, bool),
                                        tendon_inputs=session.tendon_inputs)
        raise ArithmeticError("synthetic original advance failure")

    with pytest.raises(ArithmeticError, match="original advance failure"):
        m.with_applied_action(session, np.ones(42), np.full(8, .2), NAMES,
                              advance)
    assert len(session.body.calls) == 1
    assert "step_joint_targets" not in vars(session.body)
    assert session.tendon_inputs == dict.fromkeys(NAMES, .2)


def test_parity_gate_blocks_teacher_and_physical_row_zero_tendon_hold():
    result = {"parity": "not_run", "teacher": "not_run"}
    ledger = {"arms": []}
    with pytest.raises(RuntimeError, match="parity"):
        m.require_parity_before_teacher(result, ledger)
    ledger["arms"].append({"arm": "parity", "valid": False})
    result["parity"] = "valid_exact"
    with pytest.raises(RuntimeError, match="parity"):
        m.require_parity_before_teacher(result, ledger)
    ledger["arms"][0]["valid"] = True
    m.require_parity_before_teacher(result, ledger)
    row = {key: 0 for key in m.PHYSICAL_KEYS}
    row["tendon_inputs"] = dict.fromkeys(NAMES, .2)
    old = {**row, "tendon_inputs": dict.fromkeys(NAMES, 0.)}
    m.physical_equal([row], [old], tendon_names=NAMES)  # control0 hold is not applied
    with pytest.raises(AssertionError):
        m.physical_equal([row, row], [old, old], tendon_names=NAMES)


def test_partial_stage_persistence_keeps_each_available_channel(tmp_path):
    class Observer:
        observation_ids = np.array([2, 5], np.int64)
        _observation_start_tick = 0
        _observation_active = True

        def disable_observation(self):
            self._observation_active = False

        def observation_static_metadata(self):
            return {"capacity_stages": 4, "start_tick": 0, "end_tick": 1}

        def copy_values(self, channel):
            if channel == "negative_magnitude":
                raise OSError("synthetic one-channel readback failure")
            return np.full((4, 2), 3 if channel == "drive" else 2,
                           np.float32)

        def copy_counters(self):
            return np.array([4, 4], np.int64)

        def copy_overflow(self):
            return 0

        def copy_armed(self):
            return 0

    with pytest.raises(OSError, match="one-channel"):
        m.save_stage_raw(tmp_path, Observer(), NAMES)
    assert (tmp_path / "stage-positive_recurrent.npy").exists()
    assert (tmp_path / "stage-drive.npy").exists()
    assert (tmp_path / "observation.npz").exists()
    meta = json.loads((tmp_path / "observation.json").read_text())
    assert meta["tendon_names"] == list(NAMES)
    with np.load(tmp_path / "observation.npz") as data:
        np.testing.assert_array_equal(data["motor_ids"], [2, 5])


def test_native_exact_preserves_nan_bit_pattern():
    raw = {key: np.array([0.], np.float64) for key in m.NATIVE_KEYS}
    raw["slip_mm_s"] = np.array([[np.nan]], np.float64)
    raw["slip_valid"] = np.array([[False]])
    saved = {key: value.copy() for key, value in raw.items()}
    m.native_equal(raw, saved, parity=True)
    saved["normal_load_bw"][0] = 1
    with pytest.raises(ValueError, match="normal_load_bw"):
        m.native_equal(raw, saved, parity=True)


def test_actual_one_element_observation_flags_and_wrong_shape():
    class Observer:
        observation_ids = np.arange(391, dtype=np.int64)

    packet = {"counters": np.full(391, 7920, np.int64),
              "overflow": np.zeros(1, np.int32),
              "armed": np.zeros(1, np.int32),
              "motor_ids": np.arange(391, dtype=np.int64)}
    m.validate_observation_packet(packet, Observer(), 7920)
    packet["overflow"] = np.array(0, np.int32)
    with pytest.raises(ValueError, match="flag shape"):
        m.validate_observation_packet(packet, Observer(), 7920)


def test_arm_failure_is_recorded_and_teacher_not_run():
    result = {"parity": "not_run", "teacher": "not_run"}
    ledger = {"arms": []}

    def fail_parity():
        ledger["arms"].append({"arm": "parity", "status": "technical_or_fidelity_failure"})
        raise ValueError("synthetic parity mismatch")

    with pytest.raises(ValueError, match="parity mismatch"):
        m.execute_registered_arm("parity", fail_parity, result, ledger)
    assert result["parity"] == "technical_or_fidelity_failure"
    assert result["teacher"] == "not_run"
    with pytest.raises(RuntimeError, match="parity"):
        m.require_parity_before_teacher(result, ledger)


def test_late_receipt_failure_preserves_science_error_and_invalidates_success():
    def case(science_failure):
        science = ArithmeticError("scientific first failure") if science_failure else None
        result = {"validity": ("technical_or_fidelity_failure" if science_failure
                               else "valid_matched_input_capture"),
                  "execution_status": "failed" if science_failure else "complete"}
        ledger = {"storage_errors": []}
        saved = {}
        calls = {"ledger": 0, "manifest": 0}

        def write_ledger():
            calls["ledger"] += 1
            if science_failure and calls["ledger"] == 1:
                raise OSError("synthetic late ledger failure")
            saved["ledger"] = dict(ledger)

        def write_result():
            saved["result"] = dict(result)

        def write_manifest():
            calls["manifest"] += 1
            if not science_failure and calls["manifest"] == 1:
                raise OSError("synthetic late manifest failure")
            saved["manifest_result_validity"] = saved["result"]["validity"]

        primary = m.publish_final_receipts(result, ledger, science,
                                           write_ledger, write_result, write_manifest)
        assert primary is science if science_failure else isinstance(primary, OSError)
        assert calls["ledger"] == 2
        assert saved["result"]["validity"] == result["validity"]
        assert saved["manifest_result_validity"] == result["validity"]
        assert len(ledger["storage_errors"]) == 1
        assert result["execution_status"] == "failed"
        if science_failure:
            assert result["primary_error"] == repr(science)
            assert "late ledger" in science.__notes__[0]
        else:
            assert result["validity"] == "technical_storage_failure"

    case(False)
    case(True)
