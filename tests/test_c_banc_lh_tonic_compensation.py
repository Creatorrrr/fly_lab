"""Generated-only checks for E264 frozen neural response protocol."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from tools import probe_c_banc_m3_lh_tonic_compensation as e264


def test_observation_identity_lookup_preserves_unsorted_saved_order() -> None:
    roster = np.array([70, 2, 91, 4, 30, 8], np.int64)
    selected = np.array([8, 70, 4], np.int64)
    columns = e264.observed_columns(roster, selected)
    np.testing.assert_array_equal(columns, [5, 0, 3])
    trace = np.arange(18).reshape(3, 6)
    np.testing.assert_array_equal(trace[:, columns], trace[:, [5, 0, 3]])
    np.testing.assert_array_equal(roster, [70, 2, 91, 4, 30, 8])
    with pytest.raises(ValueError, match="missing"):
        e264.observed_columns(roster, np.array([99], np.int64))
    with pytest.raises(ValueError, match="unique"):
        e264.observed_columns(np.array([8, 8], np.int64), selected)


def test_drive_bias_changes_only_existing_zero_bridge() -> None:
    base = np.arange(e264.e254.N, dtype=np.float32)
    base[e264.TARGET] = np.float32(0)
    result = e264.drive_bias(base, np.float32(-.19308185577392578))
    assert result[e264.TARGET].tobytes() == np.float32(-.19308185577392578).tobytes()
    assert result[:e264.TARGET].tobytes() == base[:e264.TARGET].tobytes()
    assert result[e264.TARGET + 1:].tobytes() == base[e264.TARGET + 1:].tobytes()
    assert base[e264.TARGET] == 0
    base[e264.TARGET] = 1
    with pytest.raises(ValueError):
        e264.drive_bias(base, np.float32(-.19308185577392578))
    with pytest.raises(ValueError):
        e264.drive_bias(base, np.float64(-.19308185577392578))


def test_decoder_step_preserves_held_row_and_original_expm1_formula() -> None:
    bank = np.zeros((391, 3), np.float64)
    bank[7] = [1., 2., 3.]
    rate = np.zeros(391, np.float32)
    rate[7] = np.float32(4.)
    pool = np.zeros((96, 391), np.float64)
    pool[42, 7] = 1.
    slopes = np.zeros((96, 391, 3, 2), np.float64)
    slopes[42, 7, :, :] = .5
    moment = np.zeros((42, 96), np.float64)
    moment[19, 42] = 2.
    rest = np.zeros(42, np.float64)
    rest[19] = .1
    params = {"tau": np.array([.005, .01, .04]),
              "kappa": np.zeros(391), "slopes": slopes, "pool": pool,
              "moment": moment, "rest": rest}
    updated, activation, q19 = e264.decoder_step(bank, rate, params)
    alpha = -np.expm1(-.005 / params["tau"])
    expected = bank[7] + alpha * (4. - bank[7])
    np.testing.assert_array_equal(updated[7], expected)
    assert activation[0] == .5 * expected.sum()
    assert q19 == .1 + .02 * 2 * activation[0]
    np.testing.assert_array_equal(bank[7], [1., 2., 3.])


def test_response_and_hypothesis_keep_magnitude_and_operating_point_distinct() -> None:
    low = np.zeros((41, 391), np.float32)
    high = low.copy()
    high[1:, :56] = 2.
    ids = np.arange(56, dtype=np.int64)
    assert e264.response_rms(low, high, ids) == .01
    b = np.zeros(101)
    x = np.r_[0, np.full(100, -.01)]
    z = np.r_[0, np.full(100, -.001)]
    rb = np.full(101, 20.)
    rx = np.full(101, 30.)
    rz = np.r_[20., np.linspace(20., 21., 100)]
    report = e264.assess_h264(b, x, z, np.zeros(99), rb, rx, rz, .01, .009)
    assert report["H264_tonic_compensation"] is True
    assert report["Z_rate_range"] > 0
    bad = e264.assess_h264(b, x, z, np.zeros(99), rb, rx, rz, .01, .00001)
    assert bad["H264_tonic_compensation"] is False
    inconclusive = e264.assess_h264(b, b, b, np.zeros(99), rb, rx, rz, .01, .009)
    assert inconclusive["command_residual_fraction"] is None
    assert inconclusive["H264_tonic_compensation"] is None
    assert inconclusive["classification"] == "inconclusive"


def test_six_fixed_source_masks_broadcast_and_detect_tamper() -> None:
    pre = np.array([2, 4, 6, 8, 10, 12], np.int64)
    actual = np.ones(e264.e254.N, np.float32)
    actual[6] = 0
    mask = actual[pre].astype(np.float64)
    rate = np.arange(99 * 6, dtype=np.float64).reshape(99, 6)
    delta = np.array([.01, .02, .03, .04, .05, .06], np.float64)
    result = e264.calibration_current(rate, mask, delta, actual, pre)
    np.testing.assert_array_equal(result, rate * delta[None] * mask[None])
    assert np.all(result[:, 2] == 0)
    mutated = mask.copy()
    mutated[2] = 1
    with pytest.raises(ValueError, match="mask6"):
        e264.calibration_current(rate, mutated, delta, actual, pre)
    with pytest.raises(ValueError, match="mask6"):
        e264.calibration_current(rate, np.tile(mask, (99, 1)), delta, actual, pre)


def test_disk_configuration_reloads_rule_identity_and_bias_bits(tmp_path) -> None:
    freeze = {"bias64": -0.19308186101907848,
              "bias32": -0.19308185577392578, "bias32_hex": "40b745be"}
    config = e264.intervention_configuration("fixed-model", freeze)
    path = tmp_path / "intervention-configuration.json"
    e264.e248.write_json(path, config)
    bias = e264.read_intervention_configuration(path, freeze, "fixed-model")
    assert bias.tobytes().hex() == freeze["bias32_hex"]
    for field, change in (("rule", "tampered"), ("model_identity", "other"),
                          ("bias32_hex", "00000000"), ("bias64", 0.)):
        broken = {**config, field: change}
        e264.e248.write_json(path, broken)
        with pytest.raises(ValueError, match="persisted"):
            e264.read_intervention_configuration(path, freeze, "fixed-model")


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Network:
    def __init__(self, saved):
        self.restore(saved)

    def restore(self, saved):
        self.tick = saved["tick"]
        self.drive = _Tensor(saved["drive"].copy())
        self.mask = saved["output_mask"].copy()

    def snapshot(self):
        return {"tick": self.tick, "drive": self.drive.value.copy(),
                "output_mask": self.mask.copy()}

    def advance(self, drive, steps):
        self.drive.value[:] = drive
        self.tick += steps


def test_branch_uses_held_source_and_tonic_in_every_control(tmp_path, monkeypatch) -> None:
    saved = {"tick": 2000, "drive": np.zeros(e264.e254.N, np.float32),
             "output_mask": np.ones(e264.e254.N, np.float32)}
    saved["drive"][e264.TARGET] = np.float32(-.19308185577392578)
    network = _Network(saved)
    raw = e264.raw_arrays(tmp_path, 2, 3, 3)
    seen = []

    def observe(net, row, arrays, entry, context):
        seen.append((row, net.drive.value.copy()))
        arrays["bridge-drive.npy"][row] = net.drive.value[e264.TARGET]
        entry["observed_endpoints"] = row + 1

    monkeypatch.setattr(e264, "observe_endpoint", observe)
    entry = {"reset_exact": False, "entered_controls": 0,
             "completed_controls": 0, "unknown_neural_partial": False}
    source = {"sensory": {"source_ids": np.array([2, 4], np.int64)}}
    decoder = {"kappa": np.zeros(391), "slopes": np.zeros((96, 391, 3, 2)),
               "pool": np.zeros((96, 391)), "rest": np.zeros(42),
               "moment": np.zeros((42, 96))}
    result = e264.run_arm(network, saved, np.zeros((391, 3), np.float64),
                          source, raw, entry, {"decoder": decoder},
                          {"drive_ids": np.array([2, 4, e264.TARGET])},
                          np.float32(-.19308185577392578), lambda: None,
                          lambda: None, response_value=np.float32(8.75))
    assert result["tick"] == 2020
    assert entry["completed_controls"] == 1
    assert seen[1][1][2] == seen[1][1][4] == np.float32(8.75)
    assert seen[1][1][e264.TARGET] == np.float32(-.19308185577392578)
    assert np.count_nonzero(seen[1][1]) == 3


def test_startup_bias_is_same_control_and_failed_advance_is_partial(tmp_path, monkeypatch) -> None:
    saved = {"tick": 0, "drive": np.zeros(e264.e254.N, np.float32),
             "output_mask": np.ones(e264.e254.N, np.float32)}
    network = _Network(saved)
    raw = e264.raw_arrays(tmp_path, 2, 3, 2)
    observed = []

    def observe(net, row, arrays, entry, context):
        observed.append((row, net.drive.value.copy()))
        arrays["bridge-drive.npy"][row] = net.drive.value[e264.TARGET]
        entry["observed_endpoints"] = row + 1

    def reconstruct(old, *_args):
        assert old[0] == 7.
        sensory = np.zeros(e264.e254.N, np.float32)
        sensory[3] = 7.
        return sensory, sensory.copy()

    monkeypatch.setattr(e264, "observe_endpoint", observe)
    monkeypatch.setattr(e264.e254, "reconstruct_drive", reconstruct)
    decoder = {"kappa": np.zeros(391), "slopes": np.zeros((96, 391, 3, 2)),
               "pool": np.zeros((96, 391)), "rest": np.zeros(42),
               "moment": np.zeros((42, 96))}
    source = {"sensory": {"source_ids": np.array([3], np.int64),
                          "old_filtered": np.array([[7.]]),
                          "returned_source_drive": np.array([[7.]], np.float32)},
              "sensory_ids": np.array([3], np.int64),
              "sensory_ports": np.array([0], np.int64)}
    data = {"drive_ids": np.array([3, e264.TARGET]),
            "descending_ids": np.array([], np.int64),
            "navigation_ports": np.array([], np.int64)}
    entry = {"reset_exact": False, "entered_controls": 0,
             "completed_controls": 0, "unknown_neural_partial": False}
    e264.run_arm(network, saved, np.zeros((391, 3), np.float64), source,
                 raw, entry, {"decoder": decoder}, data,
                 np.float32(-.19308185577392578), lambda: None, lambda: None)
    assert observed[0][1][e264.TARGET] == 0.
    assert observed[1][1][e264.TARGET] == np.float32(-.19308185577392578)
    assert observed[1][1][3] == 7.
    assert entry["entered_controls"] == entry["completed_controls"] == 1

    class FailedNetwork(_Network):
        def advance(self, drive, steps):
            raise RuntimeError("generated advance failure")

    failed = FailedNetwork(saved)
    failed_entry = {"reset_exact": False, "entered_controls": 0,
                    "completed_controls": 0, "unknown_neural_partial": False}
    with pytest.raises(RuntimeError, match="generated advance failure"):
        e264.run_arm(failed, saved, np.zeros((391, 3), np.float64), source,
                     raw, failed_entry, {"decoder": decoder}, data,
                     np.float32(-.19308185577392578), lambda: None, lambda: None)
    assert failed_entry["entered_controls"] == 1
    assert failed_entry["completed_controls"] == 0
    assert failed_entry["unknown_neural_partial"] is True


def test_late_publication_revokes_nested_and_canonical_report(tmp_path) -> None:
    e264.e248.write_json(tmp_path / "factor-windows.json", {
        "schema": "generated", "H264": {"H264_tonic_compensation": True},
        "interpretation_status": "complete_neural_only"})
    ledger = {"models": {}, "arms": {}}
    result = {"validity": e264.VALIDITY, "H264_tonic_compensation": True,
              "H264": {"H264_tonic_compensation": True},
              "finalization_pending": True}
    errors = []

    def persist(path, value):
        if path.name == "output-manifest.json":
            raise OSError("generated manifest failure")
        e264.e248.write_json(path, value)

    primary = e264.finish_receipts(tmp_path, {"resource_limits": {
        "internal_s": 45, "output_bytes": 100000}}, ledger, result, errors,
        None, e264.time.perf_counter(), persist=persist)
    assert isinstance(primary, OSError)
    assert result["H264_tonic_compensation"] is None
    assert result["H264"]["H264_tonic_compensation"] is None
    report = e264.e248.read_json(tmp_path / "factor-windows.json")
    assert report["H264"]["H264_tonic_compensation"] is None
    assert report["H264"]["classification"] == "not_tested"
    assert e264.e248.read_json(tmp_path / "result.json")["validity"] != e264.VALIDITY


def test_cli_help_does_not_load_originals_or_models() -> None:
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-m",
         "tools.probe_c_banc_m3_lh_tonic_compensation", "--help"],
        capture_output=True, text=True, check=True)
    assert "--freeze-sha256" in completed.stdout
    assert "--review-sha256" in completed.stdout
