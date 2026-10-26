"""Small CPU-only E245 probes; never construct the original body or network."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tools import observe_c_banc_m3_passive_signals as observed


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        return FakeTensor(self.value[key])

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeBody:
    def __init__(self):
        self.leg_calls = self.knee_calls = 0

    def leg_observation(self):
        self.leg_calls += 1
        return {"angles_rad": np.arange(42, dtype=float),
                "velocities_rad_s": np.arange(42, dtype=float) / 2,
                "support_load_bw": np.ones(6),
                "non_support_load_bw": np.full(6, .2)}

    def knee_kinematics(self, *, include_direction=True):
        assert include_direction is False
        self.knee_calls += 1
        return {"flexion_rad": np.full(6, .3),
                "velocity_rad_s": np.full(6, .4)}

    def physics_time(self):
        return 0.


class FakeLeaf:
    def __init__(self, body, roster):
        self.body = body
        self.identity = roster["adapter_hash"]
        self._sensory_ids = np.asarray(roster["sensory_node_rows"])
        self._sensory_ports = np.asarray(roster["sensory_ports"])
        self._port_features = np.asarray(roster["port_features"])
        self._receptor_keys = [tuple(value) for value in roster["receptor_keys"]]
        self.ports = [(item["leg_index"], item["kind"], item["cell_type"],
                       np.asarray(item["node_rows"])) for item in roster["ports"]]
        self.filtered = np.zeros(241)
        self.velocity_lowpass = np.zeros(42)
        self.last_features = np.zeros(241)
        self.calls = 0

    def encode(self, *, sensory_cut=False, dt=.005):
        assert not sensory_cut and dt == .005
        self.calls += 1
        self.body.leg_observation()
        self.body.knee_kinematics(include_direction=False)
        drive = np.zeros(4046, np.float32)
        drive[self._sensory_ids] = np.float32(self.filtered[0])
        self.filtered += 1
        self.velocity_lowpass += .5
        self.last_features.fill(.25)
        return drive


class FakeNetwork:
    def __init__(self, roster):
        self.motor_ids = np.asarray(roster["motor_node_rows"])
        self.observation_ids = self.motor_ids.copy()
        self.core_ids = np.asarray(roster["core_adaptation_node_rows"])
        self.core_gain = 0.
        self.drive = FakeTensor(np.zeros(4046, np.float32))
        self.rate = np.zeros(4046, np.float32)
        self.tick = 0
        self.calls = 0
        self._positive_values = FakeTensor(np.zeros((160, 391), np.float32))
        self._negative_values = FakeTensor(np.zeros((160, 391), np.float32))
        self._drive_values = FakeTensor(np.zeros((160, 391), np.float32))
        self._observed_counters = FakeTensor(np.full(391, 160, np.int64))

    def readout(self, ids):
        return self.rate[ids].copy()

    def advance_readout(self, drive, indices, steps=10):
        self.calls += 1
        self.drive.value = drive.copy()
        self.rate.fill(.5)
        self.tick += steps
        return self.rate[indices].copy()


def fixture():
    sensory = list(range(3604))
    selected = list(range(4046))
    motors = list(range(391))
    ports = [{"leg_index": 0, "kind": "claw", "cell_type": "SNpp50",
              "node_rows": sensory}] + [
                  {"leg_index": 0, "kind": "club", "cell_type": "",
                   "node_rows": []} for _ in range(240)]
    roster = {"adapter_hash": "fake", "sensory_node_rows": sensory,
              "selected_node_rows": selected, "motor_node_rows": motors,
              "motor_selected_columns": motors,
              "core_adaptation_node_rows": list(range(442)),
              "sensory_ports": [0] * 3604, "port_features": [0] * 241,
              "receptor_keys": [[0, "claw", "SNpp50"]], "ports": ports}
    body = FakeBody()
    leaf = FakeLeaf(body, roster)
    network = FakeNetwork(roster)
    session = SimpleNamespace(body=body, network=network, control_tick=0,
                              adapter=SimpleNamespace(
                                  original=SimpleNamespace(original=leaf)))
    return roster, body, leaf, network, session


def test_actual_return_hooks_copy_once_and_restore_bindings(tmp_path):
    roster, body, leaf, network, session = fixture()
    collector = observed.SignalCollector(roster)
    collector.install_session(session)
    collector.install_observer(network)
    assert len(collector.rows) == 1
    assert not collector.rows[0]["encode_valid"]
    assert np.array_equal(collector.rows[0]["old_filtered"], np.zeros(241))
    drive = leaf.encode(sensory_cut=False, dt=.005)
    before = drive.copy()
    output = network.advance_readout(drive, network.motor_ids, 20)
    assert np.array_equal(drive, before)
    assert np.array_equal(output, np.full(391, .5, np.float32))
    session.control_tick = 1
    collector.completed_control({}, {"control_tick": 1, "network_tick": 20,
                                     "physics_time_s": .005})
    assert (body.leg_calls, body.knee_calls, leaf.calls, network.calls) == (1, 1, 1, 1)
    assert collector._calls == {"leg": 1, "knee": 1, "encode": 1,
                                "neural": 1, "body": 1}
    path = tmp_path / "signal-raw.npz"
    collector.save(path)
    with np.load(path, allow_pickle=False) as packet:
        assert packet["selected_rate"].shape == (2, 4046)
        assert packet["old_filtered"][1, 0] == 0
        assert packet["new_filtered"][1, 0] == 1
        assert np.array_equal(packet["old_filtered"][1],
                              packet["new_filtered"][0])
        assert np.array_equal(packet["body_valid"], [False, True])
    collector.restore()
    assert "encode" not in vars(leaf)
    assert "advance_readout" not in vars(network)
    assert "leg_observation" not in vars(body)
    assert "knee_kinematics" not in vars(body)


def test_partial_neural_failure_retains_encoder_raw_and_binding(tmp_path):
    roster, _body, leaf, network, session = fixture()
    collector = observed.SignalCollector(roster)
    collector.install_session(session)
    collector.install_observer(network)
    original = network.advance_readout

    def fail(*args, **kwargs):
        raise RuntimeError("neural fault")

    # Replace the captured underlying callable, not a production module global.
    collector.restore()
    network.advance_readout = fail
    collector = observed.SignalCollector(roster)
    collector.install_session(session)
    collector.install_observer(network)
    drive = leaf.encode(sensory_cut=False, dt=.005)
    with pytest.raises(RuntimeError, match="neural fault"):
        network.advance_readout(drive, network.motor_ids, 20)
    path = tmp_path / "partial.npz"
    collector.save(path)
    with np.load(path, allow_pickle=False) as packet:
        assert packet["encode_valid"][1]
        assert not packet["neural_valid"][1]
        assert np.isfinite(packet["old_filtered"][1]).all()
        assert np.isnan(packet["selected_rate"][1]).all()
    collector.restore()
    assert network.advance_readout is fail
    network.advance_readout = original


def test_publication_failure_revokes_validity_and_keeps_primary():
    result = {"validity": observed.VALID, "execution_status": "complete",
              "H245": "supported_for_this_exact_teacher_run"}
    ledger = {"storage_errors": [], "arms": [{"valid": True, "status": "valid"}]}
    attempts = {"manifest": 0}

    def bad_manifest():
        attempts["manifest"] += 1
        raise OSError("manifest failed")

    primary = observed.publish_receipts(result, ledger, None, lambda: None,
                                         lambda: None, bad_manifest)
    assert isinstance(primary, OSError)
    assert str(primary) == "manifest failed"
    assert result["validity"] == "technical_storage_failure"
    assert result["execution_status"] == "failed"
    assert result["H245"] == "not_tested"
    assert result["teacher"] == "technical_storage_failure"
    assert ledger["arms"][0]["valid"] is False
    assert attempts["manifest"] == 2
    assert len(ledger["storage_errors"]) == 2


def test_existing_science_error_survives_late_publication_failure():
    science = ValueError("first science fault")
    result = {"validity": "technical_or_fidelity_failure",
              "execution_status": "failed", "H245": "not_tested"}
    ledger = {"storage_errors": []}

    def bad_ledger():
        raise OSError("late ledger")

    primary = observed.publish_receipts(result, ledger, science, bad_ledger,
                                         lambda: None, lambda: None)
    assert primary is science
    assert len(ledger["storage_errors"]) == 2
    assert result["primary_error"] == repr(science)


@pytest.mark.parametrize("error,status", [
    (OSError("cleanup failed"), "technical_cleanup_or_storage_failure"),
    (TimeoutError("late deadline"), "resource_limited")])
def test_single_invalidation_revokes_arm_and_hypothesis(error, status):
    result = {"validity": observed.VALID, "execution_status": "complete",
              "teacher": observed.VALID,
              "H245": "supported_for_this_exact_teacher_run"}
    ledger = {"arms": [{"valid": True, "status": "valid"}], "storage_errors": []}
    observed.invalidate(result, ledger, error, status=status)
    assert result["validity"] == status
    assert result["teacher"] == status
    assert result["H245"] == "not_tested"
    assert ledger["arms"][0]["valid"] is False
    assert ledger["arms"][0]["status"] == status
    primary = observed.publish_receipts(result, ledger, error, lambda: None,
                                         lambda: None, lambda: None)
    assert primary is error
    assert result["teacher"] == status


def test_failed_control_preserves_actual_body_and_command(tmp_path):
    roster, _body, leaf, network, session = fixture()
    reference = {"control_tick": np.array([0, 99], np.int64)}
    collector = observed.SignalCollector(roster, reference_controls=reference)
    collector.install_session(session)
    collector.install_observer(network)
    drive = leaf.encode(sensory_cut=False, dt=.005)
    network.advance_readout(drive, network.motor_ids, 20)
    session.control_tick = 1
    actual = {"control_tick": 1, "network_tick": 20, "physics_time_s": .005,
              "applied_command": np.ones(42)}
    body_row = {"joint_targets_rad": np.ones(42).tolist()}
    with pytest.raises(AssertionError, match="reference control"):
        collector.completed_control(body_row, actual)
    assert collector.rows[1]["body_valid"]
    assert not collector.rows[1]["parity_valid"]
    assert collector.failed_control["actual_control_record"]["applied_command"] == [1.] * 42
    assert collector.failed_control["actual_trace_row"] == body_row
    path = tmp_path / "partial.npz"
    collector.save(path)
    with np.load(path, allow_pickle=False) as packet:
        assert packet["body_valid"][1]
        assert not packet["parity_valid"][1]
    collector.restore()


@pytest.mark.parametrize("signed_zero_mismatch", [False, True])
def test_first160_checked_at_second_control_and_prefix_retained(
        tmp_path, signed_zero_mismatch):
    roster, _body, leaf, network, session = fixture()
    reference = {key: np.zeros((160, 391), np.float32)
                 for key in ("positive_recurrent", "negative_magnitude", "drive")}
    if signed_zero_mismatch:
        network._drive_values.value[0, 0] = np.float32(-0.)
    collector = observed.SignalCollector(roster, stage_reference=reference,
        prefix_path=tmp_path / "stage-prefix.npz")
    collector.install_session(session)
    collector.install_observer(network)
    for index in (1, 2):
        drive = leaf.encode(sensory_cut=False, dt=.005)
        network.advance_readout(drive, network.motor_ids, 20)
        session.control_tick = index
        actual = {"control_tick": index, "network_tick": index * 20,
                  "physics_time_s": index * .005}
        if index == 2 and signed_zero_mismatch:
            with pytest.raises(ValueError, match="first160 stage prefix differs"):
                collector.completed_control({}, actual)
        else:
            collector.completed_control({}, actual)
    assert len(collector.rows) == 3
    assert collector.rows[2]["old_filtered"][0] == 1.
    assert network.calls == 2
    with np.load(tmp_path / "stage-prefix.npz", allow_pickle=False) as packet:
        assert packet["drive"].shape == (160, 391)
        assert np.array_equal(packet["counters"], np.full(391, 160))
        if signed_zero_mismatch:
            assert np.signbit(packet["drive"][0, 0])
    assert collector.prefix_verified is not signed_zero_mismatch
    collector.restore()
