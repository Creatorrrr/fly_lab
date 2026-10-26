"""Generated E208 signed-input contracts; original BANC never loaded here."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tools import observe_c_banc_m3_signed_input_channels as runner
from tools.banc_m3_motor_timescale import MotorTimescaleNetwork, _timescale_source
from tools.banc_m3_signed_input_channels import (
    SignedInputChannelNetwork,
    _signed_timescale_source,
)


def test_original_csr_reduction_and_stage_finish_preserved():
    for tanh in (False, True):
        base = _timescale_source(tanh)
        tapped = _signed_timescale_source(tanh, 2, 8)
        assert base.count("value += lookup[item >> 18] * masked[item & 0x3ffff];") == 1
        assert tapped.count("value += lookup[item >> 18] * masked[item & 0x3ffff];") == 1
        assert tapped.count("value += __shfl_down_sync(0xffffffff, value, offset);") == 1
        assert tapped.count("value = add_exact(drive[row], value);") == 1
        assert tapped.count("positive_observed[position * 2ull + column] = positive;") == 1
        assert tapped.count("negative_observed[position * 2ull + column] = negative;") == 1
        assert tapped.count("drive_observed[position * 2ull + column] = drive[row];") == 1
        assert tapped.split('extern "C" __global__ void stage')[1] == base.split(
            'extern "C" __global__ void stage')[1]


def test_exact_eighty_stage_target_indexing():
    controls = np.arange(1, 1825, dtype=np.int64)
    rows = runner.stage_rows_for_controls(controls)
    assert rows.shape == (1824 * 80,)
    assert rows[:80].tolist() == list(range(80))
    assert rows[80:160].tolist() == list(range(80, 160))
    assert rows[-1] == (1824 * 80 - 1)
    with pytest.raises(ValueError, match="1824"):
        runner.stage_rows_for_controls(np.arange(1824))


def test_signed_balance_uses_declared_float32_roundoff_not_exact_equality():
    total = np.array([[1.000002, -1.]], np.float32)
    positive = np.array([[2., 0.]], np.float32)
    negative = np.array([[2., 0.]], np.float32)
    drive = np.array([[1., -1.]], np.float32)
    error, fraction = runner.balance_block(total, positive, negative, drive)
    assert 0 < error < 1e-5 and fraction < 1
    with pytest.raises(ValueError, match="balance"):
        runner.balance_block(total + .1, positive, negative, drive)


def test_review_gate_blocks_before_model(tmp_path):
    freeze = Path("verification/banc-natural-gait-m3-20260919/"
                  "e208-signed-input-channels-freeze.json")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema": runner.REVIEW_SCHEMA,
        "approved": False, "blocking_defects": ["generated reject"],
        "freeze_sha256": runner.file_sha(freeze), "source_sha256": {}}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="source review"):
        runner.verify_contract(freeze, runner.file_sha(freeze),
            review, runner.file_sha(review))
    with pytest.raises(ValueError, match="freeze SHA"):
        runner.verify_contract(freeze, "0" * 64, review, runner.file_sha(review))


def test_four_independent_raw_channels_survive_one_hostcopy_failure(tmp_path):
    class Fake:
        def disable_observation(self):
            pass

        def observation_static_metadata(self):
            return {"motor_ids": np.array([1, 2]), "start_tick": 0,
                    "end_tick": 2, "capacity_stages": 8}

        def copy_counters(self):
            return np.array([9, 8], np.int64)

        def copy_overflow(self):
            return np.array([1], np.int32)

        def copy_armed(self):
            return np.array([0], np.int32)

        def copy_values(self, channel):
            if channel == "positive_recurrent":
                raise RuntimeError("generated hostcopy failure")
            value = np.full((8, 2), np.nan, np.float32)
            value[:4] = 1
            return value

    errors = []
    meta = runner.save_observation(tmp_path, Fake(), errors)
    assert meta["counter_completed"].tolist() == [8, 8]
    assert meta["overflow"].tolist() == [1]
    assert len(errors) == 1 and errors[0]["stage"] == "positive_recurrent-hostcopy"
    assert not (tmp_path / "motor-positive-recurrent.npy").exists()
    for name in ("motor-total-current.npy", "motor-negative-magnitude.npy",
                 "motor-drive.npy"):
        assert np.isnan(np.load(tmp_path / name)[4:]).all()
    assert json.loads((tmp_path / "observation.json").read_text())["armed"] == [0]


def test_generated_cuda_signed_channels_match_manual_first_stage():
    """ONE deferred CUDA test: two models total 8+4 setup ticks, 3+3+2 main."""
    import time

    import torch
    from threadpoolctl import threadpool_limits

    assert torch.cuda.is_available(), "Deferred actual generated CUDA comparison required"
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        matrix = csr_matrix(np.array([
            [0, 0, 0, 0], [.5, -.25, 0, 0],
            [0, 0, 0, 0], [-.75, 0, .125, 0]], np.float32))
        args = (matrix, np.full(4, .02, np.float32), np.ones(4, np.float32),
            np.full(4, 100., np.float32), np.full(4, 200., np.float32))
        kw = {"adaptation_gain": 8., "adaptation_tau_s": .3, "dt": .00025,
            "device": "cuda", "cuda_implementation": "packed", "capture_steps": 2,
            "core_ids": np.array([0], np.int64), "core_gain": 0.,
            "motor_ids": np.array([1, 3], np.int64),
            "motor_gains": np.array([6., 7.], np.float32),
            "motor_log_tau": np.zeros(2, np.float64)}
        began = time.perf_counter()
        base = MotorTimescaleNetwork(*args, **kw)
        observed = SignedInputChannelNetwork(*args,
            observation_ids=np.array([1, 3], np.int64), capacity_stages=16, **kw)
        setup_s = time.perf_counter() - began
        saved = base.snapshot()
        saved["rate"] = np.array([2., 1., 3., 4.], np.float32)
        base.restore(saved)
        observed.restore(saved)
        assert base.identity == observed.identity
        np.testing.assert_equal(base.snapshot(), observed.snapshot())
        initial_rate = observed.rate.cpu().numpy().copy()
        mask = observed.output_mask.cpu().numpy().copy()
        observed.arm_observation()
        drive = np.array([1., 2., 3., 4.], np.float32)
        began = time.perf_counter()
        base.advance(drive, steps=3)
        observed.advance(drive, steps=3)
        np.testing.assert_equal(base.snapshot(), observed.snapshot())
        observed.disable_observation()
        observed.advance(drive, steps=2)
        main_s = time.perf_counter() - began
        assert observed.copy_counters().tolist() == [12, 12]
        assert observed.copy_overflow().tolist() == [0]
        assert observed.copy_armed().tolist() == [0]
        positive = observed.copy_values("positive_recurrent")
        negative = observed.copy_values("negative_magnitude")
        actual_drive = observed.copy_values("drive")
        total = observed.copy_values("total_current")
        f = np.float32
        h = f(.000125)
        full = f(.00025)
        tau = f(.02)
        rate2 = (initial_rate + h * (-initial_rate / tau)).astype(np.float32)
        rate3 = (initial_rate + h * (-rate2 / tau)).astype(np.float32)
        rate4 = (initial_rate + full * (-rate3 / tau)).astype(np.float32)
        manual_masked = [rate * mask for rate in
                         (initial_rate, rate2, rate3, rate4)]
        expected_positive = np.asarray([
            [f(.5) * row[0], f(.125) * row[2]] for row in manual_masked],
            np.float32)
        expected_negative = np.asarray([
            [f(.25) * row[1], f(.75) * row[0]] for row in manual_masked],
            np.float32)
        assert not np.array_equal(expected_positive[0], expected_positive[1])
        np.testing.assert_allclose(positive[:4], expected_positive, rtol=0, atol=1e-6)
        np.testing.assert_allclose(negative[:4], expected_negative, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(actual_drive[:4],
            np.tile(drive[[1, 3]], (4, 1)))
        np.testing.assert_allclose(total[:4],
            actual_drive[:4] + expected_positive - expected_negative,
            rtol=0, atol=1e-6)
        assert all(np.isnan(value[12:]).all() for value in
            (positive, negative, actual_drive, total))
        print("E208_GENERATED_CUDA", {"models": 2,
            "warmup_native_ticks": 2 * (2 * kw["capture_steps"]),
            "capture_native_ticks": 2 * kw["capture_steps"],
            "main_native_ticks_base": 3,
            "main_native_ticks_observed": 5, "body_steps": 0,
            "setup_wall_s": setup_s, "main_wall_s": main_s})
