"""Generated-only checks for E244's fixed causal contrast diagnosis."""

import hashlib
import json
import time

import numpy as np
import pytest

from tools import analyze_c_banc_m3_causal_contrast as contrast


def fixture(*, multiple=False):
    n = 2001
    owner = np.array([8, 2, 14, 1, 20, 4], dtype=np.int64)
    if multiple:
        owner = np.repeat(owner, 2)
    motors = len(owner)
    groups = [{"leg": "lf"} for _ in range(96)]
    pool = np.zeros((96, motors))
    moment = np.zeros((42, 96))
    for i, muscle in enumerate(owner):
        leg_index = i // 2 if multiple else i
        groups[muscle] = {"leg": contrast.LEGS[leg_index]}
        pool[muscle, i] = 1
        moment[7*leg_index, muscle] = 2
    cap = np.tile([.25, .75], 6) if multiple else np.ones(motors)
    parameters = {"free_columns": np.arange(motors, dtype=np.int64),
        "fixed_columns": np.array([], dtype=np.int64), "membership": owner,
        "pool": pool, "moment": moment, "cap": cap,
        "ltm": np.arange(90, 96)}
    descriptor = np.zeros((n, motors, 9))
    target = np.zeros((n, 42))
    for i in range(6):
        target[:, 7*i] = np.arange(n) * .001 * (i+1)
    old_index = np.full((n, 6), -1, np.int64)
    for query in range(1, n):
        old_index[query] = query + 21 if query <= 21 else 1
    old_distance = np.zeros((n, 6))
    old_difference = np.zeros((n, 42))
    for i in range(6):
        sl = slice(i*7, (i+1)*7)
        old_difference[1:, sl] = target[old_index[1:, i], sl] - target[1:, sl]
    weights = np.zeros((6, motors))
    for i in range(6):
        if multiple:
            weights[i, 2*i:2*i+2] = [.25, .75]
        else:
            weights[i, i] = 1
    baseline_raw = {"free_columns": parameters["free_columns"].copy(),
        "sensitivity_weight": weights, "nearest_train_control": old_index,
        "nearest_train_distance": old_distance,
        "neighbor_teacher_q_difference": old_difference}
    segment = {}
    for name, lo, hi in (("train", 1, 1201), ("steady", 100, 1201),
                         ("observed_tail", 1201, 2001)):
        segment[name] = {}
        for i, leg in enumerate(contrast.LEGS):
            segment[name][leg] = {"neighbor_teacher_q_difference_rms":
                contrast.rms(old_difference[lo:hi, i*7:(i+1)*7])}
    baseline_result = {"analysis": {"segments": segment}}
    return (parameters, {"descriptor": descriptor},
            {"target_command": target}, groups, baseline_raw, baseline_result)


def test_contrast_coordinate_order_training_std_and_future_exclusion():
    descriptor = np.zeros((32, 2, 9))
    t = np.arange(32, dtype=np.float64)
    descriptor[:, 0, :3] = np.stack((t, 2*t, 3*t), axis=1)
    descriptor[:, 0, 3:6] = np.stack((t/2, t, t), axis=1)
    descriptor[:, 0, 6:9] = np.stack((t/4, t/2, -t), axis=1)
    z, scale = contrast.contrast_and_scale(descriptor, train_last=22)
    np.testing.assert_allclose(z[7, 0], [3.5, 7, 14, 1.75, 3.5, 14])
    np.testing.assert_allclose(scale[0], np.std(z[1:23, 0], axis=0, ddof=0))
    np.testing.assert_array_equal(scale[1], np.full(6, 1e-4))
    descriptor[23:, :, 0] += 1e8
    _, future_scale = contrast.contrast_and_scale(descriptor, train_last=22)
    np.testing.assert_array_equal(scale, future_scale)
    assert not np.array_equal(z[23:], contrast.contrast_and_scale(
        descriptor, train_last=22)[0][23:])
    descriptor[:, :, :3] += 17
    descriptor[:, :, 3:6] += 17
    descriptor[:, :, 6:9] += 17
    shifted, _ = contrast.contrast_and_scale(descriptor, train_last=22)
    np.testing.assert_allclose(shifted[:23], z[:23])


def test_nonzero_euclidean_neighbor_exclusion_tie_and_hypothesis_failure():
    inputs = list(fixture())
    inputs[1]["descriptor"][1500, 0, 0] = 1e-4
    report, raw = contrast.analyze_arrays(*inputs)
    assert raw["nearest_train_control"].dtype == np.int64
    assert raw["nearest_train_control"][0, 0] == -1
    assert raw["nearest_train_control"][1, 0] == 22
    assert raw["nearest_train_control"][22, 0] == 1
    assert raw["nearest_train_control"][1500, 0] == 1
    assert raw["nearest_train_distance"][1500, 0] == pytest.approx(1.)
    assert raw["scale"][0, 0] == pytest.approx(1e-4)
    np.testing.assert_allclose(raw["contrast_teacher_q_difference"][1500, 0], -1.499)
    assert report["hypothesis"]["supported"] is False
    assert "observed_tail/global" in report["hypothesis"]["failed"]
    assert report["segments"]["observed_tail"]["contrast"]["global"]["q_rms_rad"] > 0
    assert len(report["bins"]) == 10
    assert report["bins"][1]["first_control"] == 201


def test_unequal_two_mn_weights_multicoordinate_euclidean_oracle():
    inputs = list(fixture(multiple=True))
    descriptor = inputs[1]["descriptor"]
    t = np.arange(1201, dtype=np.float64) / 1200
    descriptor[1:1201, 0, 0] = t[1:]
    descriptor[1:1201, 0, 3] = t[1:] / 2
    descriptor[1:1201, 1, 1] = 2*t[1:]
    descriptor[1:1201, 1, 4] = t[1:]
    descriptor[1500, [0, 1]] = descriptor[700, [0, 1]]
    descriptor[1500, 0, 0] += .03
    descriptor[1500, 1, 1] += .04
    _, raw = contrast.analyze_arrays(*inputs)
    np.testing.assert_array_equal(raw["sensitivity_weight"][0, :2], [.25, .75])
    z, scale = contrast.contrast_and_scale(descriptor)
    delta = (z[1500, None, :2] - z[1:1201, :2]) / scale[:2]
    manual = np.sqrt(np.sum(delta**2 * np.array([.25, .75])[None, :, None], axis=(1, 2)))
    nearest = int(np.argmin(manual)) + 1
    assert raw["nearest_train_control"][1500, 0] == nearest
    assert raw["nearest_train_distance"][1500, 0] == pytest.approx(manual[nearest-1])
    assert manual[nearest-1] > 0


def test_baseline_q_provenance_and_segment_rms_fail_closed():
    inputs = list(fixture())
    inputs[4]["neighbor_teacher_q_difference"][1201, 0] += .01
    with pytest.raises(ValueError, match="baseline teacher q provenance"):
        contrast.analyze_arrays(*inputs)
    inputs = list(fixture())
    inputs[5]["analysis"]["segments"]["steady"]["lf"][
        "neighbor_teacher_q_difference_rms"] += .01
    with pytest.raises(ValueError, match="baseline segment RMS"):
        contrast.analyze_arrays(*inputs)


def test_partial_batch_raw_prefix_retained():
    raw = {}
    def stop(kind, leg, rows):
        if kind == "neighbor_batch" and leg == 0 and rows == 128:
            raise TimeoutError("generated work deadline")
    with pytest.raises(TimeoutError, match="generated work deadline"):
        contrast.analyze_arrays(*fixture(), raw=raw, progress=stop)
    assert raw["completed_neighbor_batches"][0] == 1
    assert raw["completed_query_rows"][0] == 128
    assert raw["completed_neighbor_legs"] == 0
    assert np.isfinite(raw["nearest_train_distance"][1:129, 0]).all()
    assert np.isnan(raw["nearest_train_distance"][129:, 0]).all()


def test_ledger_failure_preserves_raw_and_latest_manifest(tmp_path, monkeypatch):
    original = contrast.write_json
    failed = [False]
    def once(path, value):
        if path.name == "ledger.json" and not failed[0]:
            failed[0] = True
            raise OSError("generated ledger failure")
        return original(path, value)
    monkeypatch.setattr(contrast, "write_json", once)
    ledger = {"errors": []}
    result = {"validity": "valid_saved_diagnostic"}
    error = contrast.settle_publication(tmp_path, result,
        {"partial": np.array([1., np.nan])}, ledger,
        elapsed=lambda: .1, deadline=None)
    assert isinstance(error, OSError)
    assert result["validity"] == "incomplete_or_invalid"
    assert result["finalization_pending"] is False
    with np.load(tmp_path / "diagnostic-raw.npz") as saved:
        assert np.isnan(saved["partial"][1])
    manifest = json.loads((tmp_path / "output-manifest.json").read_bytes())
    for name in ("errors.json", "ledger.json", "diagnostic-raw.npz"):
        assert manifest["files"][name]["sha256"] == hashlib.sha256(
            (tmp_path / name).read_bytes()).hexdigest()


def test_after_result_deadline_invalidates(tmp_path, monkeypatch):
    clock = [0.]
    original = contrast.Path.write_bytes
    def advance(path, value):
        written = original(path, value)
        if path.name == "result.json" and clock[0] == 0:
            clock[0] = 2.
        return written
    monkeypatch.setattr(contrast.Path, "write_bytes", advance)
    monkeypatch.setattr(contrast.time, "perf_counter", lambda: clock[0])
    result = {"validity": "valid_saved_diagnostic", "hypothesis_supported": True}
    ledger = {"errors": []}
    error = contrast.settle_publication(tmp_path, result,
        {"x": np.array([1.])}, ledger, elapsed=lambda: clock[0], deadline=1.)
    assert isinstance(error, TimeoutError)
    saved = json.loads((tmp_path / "result.json").read_bytes())
    assert saved["validity"] == "incomplete_or_invalid"
    assert saved["finalization_pending"] is False
    assert saved["hypothesis_supported"] is None
    manifest = json.loads((tmp_path / "output-manifest.json").read_bytes())
    for name in ("errors.json", "ledger.json"):
        assert manifest["files"][name]["sha256"] == hashlib.sha256(
            (tmp_path / name).read_bytes()).hexdigest()


def test_gate_rejects_before_data_load(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(contrast, "load_inputs", lambda *_args, **_kwargs: called.append(1))
    with pytest.raises(ValueError, match="freeze SHA"):
        contrast.run(tmp_path / "missing", "bad", tmp_path / "review", "bad",
                     tmp_path / "out", generated_test_output=True)
    assert not called
    result = json.loads((tmp_path / "out" / "result.json").read_bytes())
    assert result["original_calculation_count"] == 0
    assert result["loaded_input_roles"] == []


def test_deadline_before_result_has_no_success_receipt(tmp_path):
    with pytest.raises(TimeoutError, match="before result publication"):
        contrast.publish(tmp_path, {"validity": "valid_saved_diagnostic"},
                         {"x": np.array([1.])}, {"errors": []},
                         deadline=time.perf_counter()-1)
    assert not (tmp_path / "result.json").exists()
