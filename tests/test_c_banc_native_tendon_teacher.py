"""Generated-only E240 q-only action and evidence-publication checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools import observe_c_banc_m3_native_tendon_teacher as m

NAMES = ("lf_tarsus", "lm_tarsus", "lh_tarsus", "rf_tarsus",
         "rm_tarsus", "rh_tarsus", "abdomen_left", "abdomen_right")


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


def test_boundary_applies_q_only_with_current_decoder_tendons_and_restores():
    session = Session()
    original = session.body.step_joint_targets
    teacher_q = np.arange(42, dtype=np.float64) / 100
    decoded_q = np.full(42, -.5, np.float64)

    def advance():
        # Mimic the decoder updating this control's tendon mapping immediately
        # before BancWalkingSession calls the body method.
        session.tendon_inputs[NAMES[0]] = .37
        session.tendon_inputs[NAMES[-1]] = .26
        return session.body.step_joint_targets(decoded_q, np.zeros(6, bool),
                                               tendon_inputs=session.tendon_inputs)

    value, receipt = m.with_native_tendon_action(session, teacher_q, NAMES, advance)
    assert value == "advanced_once"
    assert len(session.body.calls) == 1
    applied_q, adhesion, tendons = session.body.calls[0]
    np.testing.assert_array_equal(applied_q, teacher_q)
    np.testing.assert_array_equal(adhesion, np.zeros(6, bool))
    assert tendons[NAMES[0]] == .37 and tendons[NAMES[-1]] == .26
    np.testing.assert_array_equal(receipt["decoded_command"], decoded_q)
    np.testing.assert_array_equal(receipt["applied_tendon"],
                                  receipt["decoded_tendon"])
    assert receipt["applied_tendon"][0] == .37
    assert "step_joint_targets" not in vars(session.body)
    assert session.body.step_joint_targets == original


def test_boundary_keeps_original_failure_and_rejects_stale_tendon_mapping():
    session = Session()

    def fail():
        session.body.step_joint_targets(np.zeros(42), np.zeros(6, bool),
                                        tendon_inputs=session.tendon_inputs)
        raise ArithmeticError("synthetic original advance failure")

    with pytest.raises(ArithmeticError, match="original advance failure"):
        m.with_native_tendon_action(session, np.ones(42), NAMES, fail)
    assert len(session.body.calls) == 1
    assert "step_joint_targets" not in vars(session.body)

    def stale():
        return session.body.step_joint_targets(np.zeros(42), np.zeros(6, bool),
                                               tendon_inputs=dict(session.tendon_inputs))

    with pytest.raises(RuntimeError, match="current decoder"):
        m.with_native_tendon_action(session, np.ones(42), NAMES, stale)
    assert len(session.body.calls) == 1
    assert "step_joint_targets" not in vars(session.body)


def test_first_motor_readout_is_checked_before_next_control():
    matched = {"control_tick": np.arange(2, dtype=np.int64),
               "network_tick": np.array([0, 20], np.int64),
               "physics_time_s": np.array([0., .005]),
               "motor_rate32": np.zeros((2, 391), np.float32)}
    rows = [{key: value[index] for key, value in matched.items()}
            for index in range(2)]
    m.verify_first_readout(rows, matched)
    rows[1]["motor_rate32"] = np.ones(391, np.float32)
    with pytest.raises(AssertionError, match="first readout"):
        m.verify_first_readout(rows, matched)


def test_stage_metadata_includes_physical_tendon_actuator_columns(tmp_path, monkeypatch):
    def fake_save(out, observer, names):
        (out / "observation.json").write_text(json.dumps({"tendon_names": names}),
                                               encoding="utf-8")

    monkeypatch.setattr(m.original, "save_stage_raw", fake_save)
    m.save_stage_raw(tmp_path, object(), NAMES, np.arange(50, 58, dtype=np.int32))
    metadata = json.loads((tmp_path / "observation.json").read_text("utf-8"))
    assert metadata["tendon_names"] == list(NAMES)
    assert metadata["tendon_actuator_ids"] == list(range(50, 58))
    with pytest.raises(ValueError, match="actuator IDs"):
        m.save_stage_raw(tmp_path, object(), NAMES, np.zeros(8, np.int32))


def test_late_persistence_invalidates_capture_without_replacing_science_error():
    def case(scientific_failure):
        science = ArithmeticError("science first") if scientific_failure else None
        result = {"validity": ("technical_or_fidelity_failure" if science
                               else m.VALID),
                  "execution_status": "failed" if science else "complete"}
        ledger = {"storage_errors": []}
        saved = {}
        calls = 0

        def write_ledger():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("late ledger storage failure")
            saved["ledger"] = dict(ledger)

        def write_result():
            saved["result"] = dict(result)

        def write_manifest():
            saved["manifest_validity"] = saved["result"]["validity"]

        primary = m.publish_final_receipts(result, ledger, science,
                                           write_ledger, write_result,
                                           write_manifest)
        assert primary is science if science else isinstance(primary, OSError)
        assert calls == 2
        assert len(ledger["storage_errors"]) == 1
        assert saved["result"]["validity"] == result["validity"]
        assert saved["manifest_validity"] == result["validity"]
        assert result["execution_status"] == "failed"
        if science:
            assert result["primary_error"] == repr(science)
        else:
            assert result["validity"] == "technical_storage_failure"

    case(False)
    case(True)


def test_completed_arm_late_cleanup_and_ledger_failures_revoke_success():
    def case(cleanup_error, fail_first_persist):
        account = {"arm": "teacher", "status": m.VALID, "valid": True,
                   "storage_errors": [], "controls_completed": 2000}
        ledger = {"arms": [account]}
        result = {"teacher": "not_run"}
        durable = []
        calls = 0

        def persist():
            nonlocal calls
            calls += 1
            if fail_first_persist and calls == 1:
                raise OSError("late arm ledger failure")
            durable.append((account["status"], account["valid"],
                            account["controls_completed"]))

        def finished_arm():
            primary = cleanup_error
            primary = m.finalize_arm_receipt(account, primary, persist)
            if primary is not None:
                raise primary
            return account

        expected = ArithmeticError if cleanup_error else OSError
        with pytest.raises(expected) as caught:
            m.original.execute_registered_arm("teacher", finished_arm,
                                              result, ledger)
        assert caught.value is cleanup_error if cleanup_error else isinstance(
            caught.value, OSError)
        assert account["valid"] is False
        assert account["status"] == "technical_storage_or_cleanup_failure"
        assert result["teacher"] == account["status"]
        assert durable[-1] == (account["status"], False, 2000)
        assert calls == (2 if fail_first_persist else 1)
        if fail_first_persist and cleanup_error:
            assert "late arm ledger" in cleanup_error.__notes__[0]

    case(ArithmeticError("scientific cleanup first"), True)
    case(None, True)
    case(ArithmeticError("late journal close"), False)
