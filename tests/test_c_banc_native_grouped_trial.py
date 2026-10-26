"""E222 trial checkpoint and readout boundary checks without model construction."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tools import banc_m3_native_grouped_trial as trial
from tools.banc_m3_monotone_recruitment import MonotoneRecruitmentTrial
from tools.banc_m3_motor_timescale import MotorTimescaleTrial


def test_unwrap_e150_preserves_canonical_source_and_input() -> None:
    timescale = {"name": "source", "ids": np.array([1, 2], np.int64)}
    saved = {"schema": MonotoneRecruitmentTrial.schema,
        "motor_timescale_patch": timescale,
        "monotone_recruitment_patch": {
            "source_adapter_identity": "canonical",
            "source_motor_timescale_patch": copy.deepcopy(timescale)},
        "adapter": {"hash": "monotone", "original": {"hash": "canonical",
                                                  "activation": np.array([2.])}},
        "network": {"rate": np.array([.25], np.float32)}}
    original = copy.deepcopy(saved)
    source = trial.unwrap_e150_checkpoint(saved)
    assert source["schema"] == MotorTimescaleTrial.schema
    assert source["adapter"]["hash"] == "canonical"
    assert "monotone_recruitment_patch" not in source
    np.testing.assert_array_equal(source["network"]["rate"], [.25])
    assert saved["schema"] == original["schema"]
    saved["monotone_recruitment_patch"]["source_adapter_identity"] = "wrong"
    with pytest.raises(ValueError, match="provenance"):
        trial.unwrap_e150_checkpoint(saved)


def test_native_checkpoint_unwrap_retains_full_rate_and_history() -> None:
    saved = {"schema": trial.SCHEMA,
        "native_grouped_patch": {"schema": trial.PATCH_SCHEMA,
                                  "source_timescale_identity": "old"},
        "network": {"model_hash": "new", "rate": np.array([1.], np.float32),
                    "native_grouped_response": {
                        "rate64": np.array([1.00000001]),
                        "fast64": np.array([[.1, .2, .3]])}},
        "adapter": {"hash": "canonical"}}
    source, patch, native = trial.unwrap_native_checkpoint(saved)
    assert patch["source_timescale_identity"] == "old"
    assert source["network"]["model_hash"] == "old"
    assert source["schema"] == MotorTimescaleTrial.schema
    assert "native_grouped_response" not in source["network"]
    np.testing.assert_array_equal(native["rate64"], [1.00000001])
    assert saved["network"]["model_hash"] == "new"
    broken = copy.deepcopy(saved)
    broken["network"].pop("native_grouped_response")
    with pytest.raises(ValueError, match="missing"):
        trial.unwrap_native_checkpoint(broken)


def test_canonical_decode_promotes_only_native_float32(monkeypatch) -> None:
    observed = {}

    def fake_decode(self, rates, *, motor_cut=False, dt=.005):
        del self
        observed.update(dtype=rates.dtype, motor_cut=motor_cut, dt=dt)
        return np.zeros(42), np.zeros(6, bool)

    monkeypatch.setattr(trial.AnatomicalRecruitmentAdapter, "decode", fake_decode)
    adapter = object.__new__(trial.CanonicalRate64Adapter)
    adapter.decode(np.ones(391, np.float32), motor_cut=True)
    assert observed == {"dtype": np.dtype("float64"), "motor_cut": True, "dt": .005}
    with pytest.raises(ValueError, match="float32"):
        adapter.decode(np.ones(391, np.float64))
