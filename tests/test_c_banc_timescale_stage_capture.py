"""E201 pre-model gate and independent raw-preservation contracts."""

import json
from pathlib import Path

import numpy as np
import pytest

from tools import observe_c_banc_m3_timescale_stage_input as runner


def test_freeze_review_gate_rejects_before_model(tmp_path):
    freeze = Path("verification/banc-natural-gait-m3-20260919/"
                  "e201-native-stage-input-freeze.json")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema": runner.REVIEW_SCHEMA,
        "approved": False, "blocking_defects": ["synthetic reject"],
        "freeze_sha256": runner.file_sha(freeze),
        "source_sha256": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="source review"):
        runner.verify_contract(freeze, runner.file_sha(freeze),
            review, runner.file_sha(review))
    with pytest.raises(ValueError, match="freeze SHA"):
        runner.verify_contract(freeze, "0" * 64,
            review, runner.file_sha(review))
    with pytest.raises(ValueError, match="review SHA"):
        runner.verify_contract(freeze, runner.file_sha(freeze),
            review, "0" * 64)


def test_save_observation_keeps_full_raw_and_counters(tmp_path):
    class FakeObserver:
        def disable_observation(self):
            pass

        def observation_static_metadata(self):
            return {"motor_ids": np.array([1, 3], np.int64),
                    "start_tick": 0, "end_tick": 2,
                    "capacity_stages": 8, "observation_buffer_bytes": 64}

        def copy_counters(self):
            return np.array([8, 9], np.int64)

        def copy_overflow(self):
            return np.array([1], np.int32)

        def copy_armed(self):
            return np.array([0], np.int32)

        def copy_values(self):
            value = np.full((8, 2), np.nan, np.float32)
            value[:4] = [[1, 2], [3, 4], [5, 6], [7, 8]]
            return value

    errors = []
    metadata = runner.save_observation(tmp_path, FakeObserver(), errors)
    assert not errors
    assert metadata["counter_requested"].tolist() == [8, 9]
    assert metadata["counter_completed"].tolist() == [8, 8]
    assert metadata["overflow"].tolist() == [1]
    assert np.isnan(np.load(tmp_path / "motor-total-current.npy")[4:]).all()
    assert json.loads((tmp_path / "observation.json").read_text())["armed"] == [0]


def test_hostcopy_failure_still_writes_static_and_error(tmp_path):
    class Broken:
        def disable_observation(self):
            pass

        def observation_static_metadata(self):
            return {"start_tick": 0, "end_tick": 1, "capacity_stages": 4}

        def copy_counters(self):
            return np.array([4], np.int64)

        def copy_overflow(self):
            return np.array([0], np.int32)

        def copy_armed(self):
            return np.array([0], np.int32)

        def copy_values(self):
            raise RuntimeError("synthetic host copy failure")

    errors = []
    metadata = runner.save_observation(tmp_path, Broken(), errors)
    assert metadata["start_tick"] == 0
    assert len(errors) == 1 and errors[0]["stage"] == "current-hostcopy"
    assert (tmp_path / "observation.json").is_file()
    assert not (tmp_path / "motor-total-current.npy").exists()


def test_endpoint_npz_dtype_and_initial_held_command(tmp_path):
    motor_ids = np.arange(391, dtype=np.int64)
    records = []
    for control in range(2):
        records.append({"control_tick": control, "network_tick": control * 20,
            "physics_time_s": control * .005,
            "motor_rate": np.full(391, control + .25, np.float64),
            "adaptation": np.full(391, control + .5, np.float32),
            "activation": np.full(96, control + .75, np.float64),
            "bank": np.full((391, 3), control + 1., np.float64)})
    runner.save_endpoints(tmp_path / "endpoints.npz", records, motor_ids)
    with np.load(tmp_path / "endpoints.npz", allow_pickle=False) as saved:
        assert saved["motor_rate"].shape == (2, 391)
        assert saved["motor_rate"].dtype == np.float64
        assert saved["adaptation"].dtype == np.float32
        assert saved["bank"].shape == (2, 391, 3)
        assert saved["control_tick"].tolist() == [0, 1]
        assert saved["network_tick"].tolist() == [0, 20]
    assert "command" not in records[0]


def test_original_signal_adaptation_excludes_held_control_zero(tmp_path):
    ids = np.array([3, 1], np.int64)
    records = [{"adaptation": np.array([99, 99], np.float32)},
               {"adaptation": np.array([4, 2], np.float32)},
               {"adaptation": np.array([5, 3], np.float32)}]
    path = tmp_path / "signals.npz"
    np.savez_compressed(path, observed_ids=np.array([1, 3, 7], np.int64),
        adaptation=np.array([[2, 4, 0], [3, 5, 0]], np.float32))
    runner.compare_signal_adaptation(path, ids, records)
    np.savez_compressed(path, observed_ids=np.array([1, 3, 7], np.int64),
        adaptation=np.array([[99, 99, 0], [2, 4, 0], [3, 5, 0]], np.float32))
    with pytest.raises(ValueError, match="shape changed"):
        runner.compare_signal_adaptation(path, ids, records)


def test_manifest_excludes_mutable_result_and_retains_raw_hash(tmp_path):
    (tmp_path / "result.json").write_text("{}", encoding="utf-8")
    np.save(tmp_path / "motor-total-current.npy", np.array([[1]], np.float32))
    manifest = runner.output_manifest(tmp_path)
    assert "motor-total-current.npy" in manifest["files"]
    assert "result.json" not in manifest["files"]
    assert "output-manifest.json" not in manifest["files"]
    assert manifest["files"]["motor-total-current.npy"]["sha256"] == runner.file_sha(
        tmp_path / "motor-total-current.npy")
