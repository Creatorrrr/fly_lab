"""Generated-array checks for the saved E198 static diagnostic."""

import json
from pathlib import Path

import numpy as np
import pytest

from tools import analyze_c_banc_m3_motor_operating_range as e198


def generated():
    observed = np.arange(1000, 1446, dtype=np.int64)[::-1].copy()
    motors = observed[np.r_[400, 3, 211, 100, np.setdiff1d(np.arange(446),
                                                            [400, 3, 211, 100])[:387]]]
    zeros = np.zeros((2000, 446), np.float32)
    signals = {"observed_ids": observed, "time_s": np.arange(1, 2001) * .005,
               "rate": zeros.copy(), "drive": zeros.copy(),
               "recurrent_current": zeros.copy(), "adaptation": zeros.copy()}
    metadata = {"observed_ids": observed.copy(), "motor_ids": motors,
                "a": np.ones(446, np.float32), "threshold": zeros[0].copy(),
                "tau": np.ones(446, np.float32), "cap": np.ones(446, np.float32),
                "gain": np.ones(446, np.float32),
                "adaptation_log_tau": np.zeros(446, np.float64)}
    native = {"control_tick": np.arange(2001, dtype=np.int64),
              "network_tick": np.arange(2001, dtype=np.int64) * 20,
              "physics_time_s": np.arange(2001) * .005,
              "motor_rate": np.zeros((2001, 391), np.float64)}
    for key in ("complete", "motor_rate_available"):
        native[key] = np.ones(2001, bool)
    return signals, metadata, native


def test_noncontiguous_motor_order_and_quantiles():
    signals, metadata, native = generated()
    columns = [int(np.flatnonzero(signals["observed_ids"] == m)[0])
               for m in metadata["motor_ids"]]
    signals["drive"][:, columns[0]] = 2
    signals["drive"][:, columns[1]] = 1
    signals["rate"][:, columns[0]] = 4
    native["motor_rate"][1:, 0] = 4
    result, per_mn = e198.analyze_arrays(signals, metadata, native)
    assert result["motor_count"] == 391
    assert per_mn["motor_ids"].tolist() == metadata["motor_ids"].tolist()
    np.testing.assert_allclose(per_mn["u_quantiles"][0, :2, 2], [2, 1])
    np.testing.assert_allclose(per_mn["r_quantiles"][0, :2, 2], [4, 0])
    assert per_mn["u_quantiles"].shape == (3, 391, 5)


def test_exclusive_input_adaptation_saturation_and_kink():
    signals, metadata, native = generated()
    cols = [int(np.flatnonzero(signals["observed_ids"] == m)[0])
            for m in metadata["motor_ids"][:4]]
    signals["drive"][:, cols] = [-1, 1, 3, 0]
    signals["adaptation"][:, cols[1]] = 2
    result, a = e198.analyze_arrays(signals, metadata, native)
    np.testing.assert_array_equal(a["input_limited_fraction"][0, :4], [1, 0, 0, 1])
    np.testing.assert_array_equal(a["adaptation_limited_fraction"][0, :4], [0, 1, 0, 0])
    np.testing.assert_array_equal(a["active_fraction"][0, :4], [0, 0, 1, 0])
    np.testing.assert_array_equal(a["saturated_fraction"][0, :4], [0, 0, 1, 0])
    np.testing.assert_array_equal(a["dphi_da_rms"][0, [0, 1, 3]], 0)
    np.testing.assert_array_equal(a["dphi_dthreshold_rms"][0, [0, 1, 3]], 0)
    assert result["windows"]["all"]["target_motor_counts"]["always_active"] == 1
    assert result["windows"]["gait"]["ticks"] == [100, 2000]


@pytest.mark.parametrize("fault", ["rate", "signal_time", "native_time", "network_tick"])
def test_exact_rate_and_clock_alignment_rejected(fault):
    signals, metadata, native = generated()
    if fault == "rate":
        native["motor_rate"][1, 0] = np.nextafter(0., 1.)
    elif fault == "signal_time":
        signals["time_s"][9] += 2e-9
    elif fault == "native_time":
        native["physics_time_s"][9] += 2e-9
    else:
        native["network_tick"][9] += 1
    with pytest.raises(ValueError):
        e198.analyze_arrays(signals, metadata, native)


def test_two_time_axes_must_match_even_when_each_matches_expected_grid():
    signals, metadata, native = generated()
    signals["time_s"][9] += 0.9e-9
    native["physics_time_s"][10] -= 0.9e-9
    with pytest.raises(ValueError, match="time axes differ"):
        e198.analyze_arrays(signals, metadata, native)


def test_summary_is_pure_json_scalar():
    signals, metadata, native = generated()
    result, _ = e198.analyze_arrays(signals, metadata, native)
    assert json.loads(json.dumps(result, allow_nan=False))["windows"]["startup"]["ticks"] == [1, 99]
    assert all(type(value) is int for value in result["windows"]["all"]["target_motor_counts"].values())


def test_hash_rejection_precedes_array_load_and_records_error(tmp_path, monkeypatch):
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}", encoding="utf-8")
    review = tmp_path / "review.json"
    review.write_text("{}", encoding="utf-8")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("numeric array load reached")
    monkeypatch.setattr(e198.np, "load", forbidden)
    with pytest.raises(ValueError, match="SHA mismatch"):
        e198.run(freeze, "0" * 64, review, "0" * 64, tmp_path / "output")
    error = json.loads((tmp_path / "output" / "error.json").read_text(encoding="utf-8"))
    assert error["phase"] == "hash_gate"
    assert error["original_calculation_count"] == 0
    assert error["partial"] is False


def test_reviewed_test_hash_rejected_before_array_load(tmp_path, monkeypatch):
    freeze = {"schema": "flylab.m3.e198.motor-operating-range.freeze.v1"}
    review = {"schema": "flylab.m3.e198.review.v1", "approved": True,
              "freeze_sha256": "a" * 64,
              "tool_sha256": e198.sha256(Path(e198.__file__)),
              "test_sha256": "0" * 64}
    monkeypatch.setattr(e198, "_checked_json", lambda path, expected: freeze if path.name == "freeze" else review)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("numeric array load reached")
    monkeypatch.setattr(e198.np, "load", forbidden)
    with pytest.raises(ValueError, match="review gate"):
        e198.run(Path("freeze"), "a" * 64, Path("review"), "b" * 64,
                 tmp_path / "output")
    error = json.loads((tmp_path / "output" / "error.json").read_text(encoding="utf-8"))
    assert error["phase"] == "hash_gate" and error["loaded_input_roles"] == []
    assert error["calculation_requested"] is False


def test_partial_output_and_failure_are_retained_without_retry(tmp_path, monkeypatch):
    signals, metadata, native = generated()
    records = {"signals": signals, "metadata": metadata, "native": native}
    source = "verification/banc-natural-gait-m3-20260919/e150-native-monotone-body/"
    roles = {key: source + filename for key, filename in e198.INPUT_FILES.items()}
    digest = "a" * 64
    freeze = {"schema": "flylab.m3.e198.motor-operating-range.freeze.v1",
              "shape": {"samples": 2000, "observed": 446, "motors": 391, "dt_s": .005,
                        "neural_steps_per_control": 20, "time_atol": 1e-9},
              "windows": {"all": [1, 2000], "startup": [1, 99], "gait": [100, 2000]},
              "saturation_relative_slope": .1, "quantile_method": "linear",
              "kink_derivative": 0., "resource_limits": {
                  "internal_s": 15., "output_bytes": e198.MAX_OUTPUT_BYTES,
                  "cpu_threads": 1, "gpu_count": 0, "original_attempts": 1, "retries": 0},
              "owned_files": ["tools/analyze_c_banc_m3_motor_operating_range.py",
                              "tests/test_c_banc_motor_operating_range.py"],
              "design": "docs/BANC_M3_E198_MOTOR_OPERATING_RANGE_DESIGN_20260924.md",
              "design_sha256": digest,
              "preservation_sha256": {"configs/banc-walking/m2-calibrated-v1.json": digest},
              "input_roles": roles, "input_sha256": {v: digest for v in roles.values()},
              "source_manifest": source + "output-manifest.json",
              "source_manifest_sha256": digest}
    review = {"schema": "flylab.m3.e198.review.v1", "approved": True,
              "freeze_sha256": digest, "tool_sha256": digest,
              "test_sha256": digest}
    monkeypatch.setattr(e198, "_checked_json", lambda path, expected: freeze if path.name == "freeze" else review)
    monkeypatch.setattr(e198, "sha256", lambda path: digest)

    class Archive:
        def __init__(self, values):
            self.values = values
            self.files = list(values)
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def __getitem__(self, key):
            return self.values[key]

    def fake_load(path, **_kwargs):
        role = next(key for key, filename in e198.INPUT_FILES.items() if path.name == filename)
        return Archive(records[role])
    monkeypatch.setattr(e198.np, "load", fake_load)
    original_rate = signals["rate"]
    signals["rate"] = np.zeros((2000, 7), np.float32)
    with pytest.raises(ValueError, match="frozen source dimensions"):
        e198.run(Path("freeze"), digest, Path("review"), digest,
                 tmp_path / "dimension-failure")
    dimension_error = json.loads((tmp_path / "dimension-failure" / "error.json").read_text(
        encoding="utf-8"))
    assert dimension_error["loaded_input_roles"] == ["signals", "metadata", "native"]
    assert dimension_error["calculation_requested"] is False
    assert dimension_error["calculation_completed"] is False
    assert dimension_error["original_calculation_count"] == 0
    signals["rate"] = original_rate
    writes = []
    def failed_save(path, **_arrays):
        writes.append(path)
        path.write_bytes(b"partial generated output")
        raise OSError("generated write failure")
    monkeypatch.setattr(e198.np, "savez_compressed", failed_save)
    out = tmp_path / "output"
    with pytest.raises(OSError, match="generated write failure"):
        e198.run(Path("freeze"), digest, Path("review"), digest, out)
    error = json.loads((out / "error.json").read_text(encoding="utf-8"))
    assert len(writes) == 1
    assert (out / "per-mn.npz").read_bytes() == b"partial generated output"
    assert error["partial"] is True and error["original_calculation_count"] == 1
    assert error["calculation_requested"] is True
    assert error["calculation_completed"] is True
    assert error["phase"] == "output"
