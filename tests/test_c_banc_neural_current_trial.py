"""CPU-only E224 decoder bridge and checkpoint provenance checks."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tools.banc_m3_native_grouped_trial import CanonicalRate64Adapter
from tools.banc_m3_neural_current_effector import NeuralCurrentEffectorNetwork
from tools.banc_m3_neural_current_trial import (
    BRIDGE_SCHEMA,
    PATCH_SCHEMA,
    SCHEMA,
    EffectorCanonicalBridge,
    unwrap_neural_current_checkpoint,
)


def fake_bridge():
    canonical = object.__new__(CanonicalRate64Adapter)
    canonical.identity = "canonical-anatomy"
    canonical.motor_ids = np.arange(391, dtype=np.int64)
    canonical.encode = lambda **kw: np.zeros(8, np.float32)
    decoded = []
    canonical.decode = lambda rates, **kw: (decoded.append(np.asarray(rates).copy()),
                                            (np.zeros(42), np.zeros(6, bool)))[1]
    canonical.snapshot = lambda: {"canonical": True}
    restored = []
    canonical.restore = lambda saved: restored.append(saved)
    network = object.__new__(NeuralCurrentEffectorNetwork)
    network.identity = "neural-current"
    network.native_motor_ids = canonical.motor_ids.copy()
    network.effector_readout = lambda: np.full(391, .25, np.float32)
    return EffectorCanonicalBridge(canonical, network), decoded, restored


def test_bridge_changes_decoder_only_and_saves_no_pointer():
    bridge, decoded, restored = fake_bridge()
    original = np.full(391, 2., np.float32)
    before = original.copy()
    target, adhesion = bridge.decode(original)
    assert target.shape == (42,) and adhesion.shape == (6,)
    np.testing.assert_array_equal(decoded[0], np.full(391, .25, np.float32))
    np.testing.assert_array_equal(bridge.last_decoder_input32, decoded[0])
    np.testing.assert_array_equal(original, before)
    assert bridge.encode().shape == (8,)
    saved = bridge.snapshot()
    assert set(saved) == {"schema", "hash", "canonical_identity",
                          "network_identity", "canonical", "last_decoder_input32"}
    assert not any(value is bridge.network for value in saved.values())
    bridge.restore(saved)
    assert restored == [{"canonical": True}]
    for key, changed in (("hash", "foreign"),
                         ("network_identity", "foreign"),
                         ("last_decoder_input32", np.zeros(391, np.float64))):
        invalid = copy.deepcopy(saved)
        invalid[key] = changed
        with pytest.raises(ValueError):
            bridge.restore(invalid)
    assert len(restored) == 1
    with pytest.raises(ValueError, match="neuronal"):
        bridge.decode(np.zeros(391, np.float64))


def test_explicit_legacy_and_effector_checkpoint_unwrap():
    patch = {"schema": PATCH_SCHEMA, "mode": "effector_canonical",
             "source_timescale_identity": "original-timescale"}
    saved = {"schema": SCHEMA, "neural_current_trial_patch": patch,
             "network": {"model_hash": "current", "rate": np.array([1.]),
                         "neural_current_effector": {"effector64": np.array([2.])}},
             "adapter": {"schema": BRIDGE_SCHEMA,
                         "canonical": {"schema": "canonical", "activation": 1}}}
    source, returned, effector = unwrap_neural_current_checkpoint(saved)
    assert source["network"]["model_hash"] == "original-timescale"
    assert "neural_current_effector" not in source["network"]
    assert source["adapter"] == saved["adapter"]["canonical"]
    assert effector["effector64"][0] == 2
    assert returned == patch
    assert "neural_current_effector" in saved["network"]
    legacy = copy.deepcopy(saved)
    legacy["neural_current_trial_patch"]["mode"] = "legacy_canonical"
    with pytest.raises(ValueError, match="Legacy"):
        unwrap_neural_current_checkpoint(legacy)
    legacy["network"].pop("neural_current_effector")
    legacy["adapter"] = {"schema": "canonical"}
    legacy["network"]["model_hash"] = "original-timescale"
    source, _, effector = unwrap_neural_current_checkpoint(legacy)
    assert effector is None and source["adapter"]["schema"] == "canonical"
    missing = copy.deepcopy(saved)
    missing["network"].pop("neural_current_effector")
    with pytest.raises(ValueError, match="effector"):
        unwrap_neural_current_checkpoint(missing)
