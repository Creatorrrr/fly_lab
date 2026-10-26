"""Synthetic E235 prefix, ownership, and partial-evidence tests only."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import observe_c_banc_m3_effector_startup_stages as stage


def test_same_index_float32_difference_and_transformed_mapping():
    original = np.zeros((85, 2), np.float32)
    observed = original.copy()
    observed[80, 1] = np.float32(2.)
    result = stage._channel_difference(
        observed, original, np.array([2., 4.]), "positive_recurrent")
    assert result["first_80_stages_exact"]
    assert result["first_difference_stage"] == 80
    assert result["first_difference_neural_tick"] == 21
    assert result["first_difference_control"] == 2
    assert result["first_difference_free_column"] == 1
    assert result["max_abs"] == 2.
    assert result["transformed_max_abs"] == pytest.approx(np.log1p(2.) / 4)

    drive = original.copy()
    drive[0, 0] = np.float32(-3.)
    signed = stage._channel_difference(
        drive, original, np.array([2., 4.]), "drive")
    assert not signed["first_80_stages_exact"]
    assert signed["first_difference_neural_tick"] == 1
    assert signed["transformed_max_abs"] == pytest.approx(np.arcsinh(3.) / 2)
    assert stage._channel_difference(original, original,
        np.array([1., 1.]), "negative_magnitude")["first_difference_stage"] is None


def test_exact_raw_prefix_preserves_dtype_shape_and_nan_bits():
    saved = {"time_s": np.array([0., np.nan, 2.], np.float64),
             "rate": np.array([[1.], [2.], [3.]], np.float32)}
    actual = {key: value[:2].copy() for key, value in saved.items()}
    stage.exact_control_prefix(actual, saved)
    actual["rate"][1, 0] = 2.0001
    with pytest.raises(ValueError, match="rate"):
        stage.exact_control_prefix(actual, saved)
    actual["rate"] = saved["rate"][:2].astype(np.float64)
    with pytest.raises(ValueError, match="rate"):
        stage.exact_control_prefix(actual, saved)


def test_observation_contract_checks_partial_without_filling_missing():
    fake = {}
    for key in stage.CHANNELS:
        shape = (7920, 343, 3) if key in ("fast64", "slow64") else (7920, 343)
        dtype = np.float64 if key.endswith("64") else np.float32
        fake[key] = SimpleNamespace(shape=shape, dtype=np.dtype(dtype))
    counter = np.zeros(343, np.int64)
    metadata = {"capacity": 7920, "free_count": 343,
                "start_tick": 0, "end_tick": 1980}
    stage.validate_observation(fake, counter, 1, 0, metadata, completed=False)
    with pytest.raises(ValueError, match="complete stage count"):
        stage.validate_observation(fake, counter, 1, 0, metadata, completed=True)
    with pytest.raises(ValueError, match="ownership"):
        stage.validate_observation(fake, counter[:-1], 1, 0, metadata,
                                   completed=False)


class FakeObserver:
    _observation_ever_armed = True

    def __init__(self, fail_channel=None):
        self.fail_channel = fail_channel
        self.stops = 0
        self.native_motor_ids = np.array([1, 6, 3, 5], np.int64)
        self.free_motor_ids = np.array([1, 3, 5], np.int64)

    def disable_observation(self):
        self.stops += 1

    def observation_static_metadata(self):
        return {"capacity": 4, "free_count": 3,
                "start_tick": 0, "end_tick": 1}

    def copy_values(self, channel):
        if channel == self.fail_channel:
            raise RuntimeError("one unavailable channel")
        shape = (4, 3, 3) if channel in ("fast64", "slow64") else (4, 3)
        dtype = np.float64 if channel.endswith("64") else np.float32
        return np.full(shape, np.nan, dtype)

    def copy_counters(self):
        return np.ones(3, np.int64)

    def copy_overflow(self):
        return 0

    def copy_armed(self):
        return 0


def test_stage_partial_channel_failure_preserves_other_raw(tmp_path):
    observer = FakeObserver("negative_magnitude")
    with pytest.raises(RuntimeError, match="unavailable channel"):
        stage.save_stage_raw(tmp_path, observer)
    assert observer.stops == 1
    assert (tmp_path / "stage-positive_recurrent.npy").exists()
    assert not (tmp_path / "stage-negative_magnitude.npy").exists()
    assert (tmp_path / "stage-slow64.npy").exists()
    assert (tmp_path / "observation.npz").exists()
    metadata = json.loads((tmp_path / "observation.json").read_text())
    assert metadata["free_motor_ids"] == [1, 3, 5]
    observer.fail_channel = None
    stage.save_stage_raw(tmp_path, observer)
    assert (tmp_path / "stage-negative_magnitude.npy").exists()


def test_replacement_rebinds_both_network_owners_and_preserves_checkpoint(monkeypatch):
    class Tensor:
        def cpu(self):
            return self

        def numpy(self):
            return np.ones(8, np.float32)

    class Old:
        identity = "same"
        adaptation_gain = 4.
        adaptation_tau_s = .15
        dt = .00025
        device = "cuda"
        cuda_implementation = "packed"
        interval_steps = 10
        core_ids = np.array([0], np.int64)
        core_gain = np.ones(8, np.float32)
        motor_ids = np.array([1, 6, 3, 5], np.int64)
        motor_gains = np.ones(4, np.float64)
        motor_log_tau = np.ones(4, np.float64)
        tau = a = threshold = cap = Tensor()

        def snapshot(self):
            return {"rate": np.zeros(8, np.float32), "model_hash": "same"}

    class Candidate:
        identity = "same"
        source_timescale_identity = "source"

        def __init__(self, *args, **kwargs):
            assert kwargs["observation_capacity"] == 7920
            assert kwargs["membership"].tolist() == [0, 0, 1]

        def restore(self, saved):
            self.saved = saved

        def snapshot(self):
            return self.saved

    old = Old()
    session = SimpleNamespace(network=old,
        adapter=SimpleNamespace(network=old),
        neural_current_trial_patch={"source_timescale_identity": "source",
            "parameters": {"free_columns": np.array([0, 2, 3]),
                "fixed_columns": np.array([1]),
                "membership": np.array([0, 3, 0, 1]),
                "scale": np.ones((4, 3))},
            "prototypes": {"centers": np.zeros((4, 64, 9)),
                           "sigma2": np.ones(4)},
            "beta": np.zeros((4, 64))})
    monkeypatch.setattr(stage, "source_matrix", lambda network: object())
    candidate = stage.replace_network(session, network_type=Candidate)
    assert candidate is session.network is session.adapter.network
    assert candidate.saved["model_hash"] == old.identity


def test_actual_construction_receipt_counts_and_incomplete_rejection():
    rows = []
    for label, expected in stage.EXPECTED_CONSTRUCTION.items():
        entries = (expected["captured_interval_entries"]
                   + expected["executed_interval_entries"])
        rows.append({"label": label, **expected,
            "synchronize_returned": True, "ownership_transferred": True,
            "profiler_restored": True, "unknown_setup_steps": 0,
            "unknown_capture_setup_steps": 0,
            "unknown_capture_state_entries": 0,
            "interval_entries": entries, "interval_returns": entries})
    assert stage.validate_construction(rows) == {
        "executed_setup_steps": 320, "captured_setup_steps": 160}
    rows[0]["synchronize_returned"] = False
    with pytest.raises(ValueError, match="profile"):
        stage.validate_construction(rows)
    rows[0]["synchronize_returned"] = True
    rows[0]["known_completed_setup_steps"] = 0
    with pytest.raises(ValueError, match="profile"):
        stage.validate_construction(rows)


def test_profiled_constructor_binds_owner_before_sync_failure():
    owner = {}
    closed = []

    class Result:
        def close(self):
            closed.append(True)

    def bad_sync():
        assert isinstance(owner["value"], Result)
        raise RuntimeError("synchronize failed")

    rows = []
    with pytest.raises(RuntimeError, match="synchronize failed"):
        stage.protocol.profiled_construction("synthetic", Result, rows,
            lambda: None, interval_code=(lambda: None).__code__,
            synchronize=bad_sync, on_return=lambda value: owner.update(value=value))
    assert len(rows) == 1 and rows[0]["ownership_transferred"]
    assert closed == [True]


def test_measured_health_cli_summary_remains_measured_and_non_success(monkeypatch,
                                                                       capsys):
    monkeypatch.setattr(stage, "run", lambda *args: {
        "validity": "measured_native_health_stop",
        "measured_error": "NativeHealthStop('upright')"})
    monkeypatch.setattr(stage.sys, "argv", ["e235", "--freeze", "f",
        "--freeze-sha256", "s", "--review", "r",
        "--review-sha256", "s", "--out", "o"])
    assert stage.main() == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["validity"] == "measured_native_health_stop"
    assert printed["first_5ms_inputs_exact"] is None
    assert "upright" in printed["measured_error"]
