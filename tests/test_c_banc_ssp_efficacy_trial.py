"""Generated-only E327 CSR and whole efficacy ownership checks."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tools import banc_m3_ssp_efficacy_trial as layer


def test_patch_csr_exact_selected_rounding_and_topology():
    dense = np.array([[0, -.25, .5, 0], [.125, 0, 0, 0],
                      [0, 0, 0, .375], [0, 0, 0, 0]], np.float32)
    original = csr_matrix(dense)
    before = (original.indptr.tobytes(), original.indices.tobytes(),
              original.data.tobytes())
    positions = np.array([0, 2], np.int64)
    groups = np.array([0, 1], np.int64)
    theta = np.array([1., 1.125], np.float64)
    patched, old, actual = layer.patch_csr(original, positions, groups, theta)
    assert patched.indptr.tobytes() == before[0]
    assert patched.indices.tobytes() == before[1]
    assert original.data.tobytes() == before[2]
    assert actual[0].tobytes() == old[0].tobytes()
    assert actual[1] == np.float32(np.float64(old[1]) * theta[1])
    assert patched.data[1:2].tobytes() == original.data[1:2].tobytes()
    assert np.array_equal(np.signbit(actual), np.signbit(old))
    neutral, _, neutral_actual = layer.patch_csr(
        original, positions, groups, np.ones(2, np.float64))
    assert neutral.data.tobytes() == original.data.tobytes()
    assert neutral_actual.tobytes() == old.tobytes()
    with pytest.raises(ValueError):
        layer.patch_csr(original, positions, groups, np.array([1., -1.]))
    with pytest.raises(ValueError):
        layer.patch_csr(original, np.array([0, 0], np.int64), groups, theta)


def test_real_generated_ssp_builder_preserves_dynamic_and_owns_identity():
    from tools.banc_m3_ssp_recovery_synapses import SSPRecoveryResourceNetwork
    from tools.probe_c_banc_m3_core_rhythm import source_matrix

    dense = np.zeros((5, 5), np.float32)
    dense[0, 1:4] = np.array([.01, -.02, .03], np.float32)
    dense[1, 0] = .02
    weights = csr_matrix(dense)
    old = SSPRecoveryResourceNetwork(
        weights, np.full(5, .03, np.float32), np.ones(5, np.float32),
        np.full(5, 1.5, np.float32), np.full(5, 40., np.float32),
        adaptation_gain=.8, adaptation_tau_s=.3, dt=.00025,
        device="cpu", cuda_implementation="reference", capture_steps=20,
        core_ids=np.array([0, 1], np.int64), core_gain=.1,
        motor_ids=np.array([4], np.int64),
        motor_gains=np.ones(1, np.float32), motor_log_tau=np.zeros(1),
        edge_ids=np.array([0, 1, 3], np.int64),
        premotor_ids=np.arange(4, dtype=np.int64),
        utilization=np.zeros(3, np.float64),
        recovery_tau=np.full(3, .15, np.float64))
    before = old.snapshot()
    changed = copy.deepcopy(before)
    changed["rate"][:] = np.linspace(.01, .05, 5, dtype=np.float32)
    changed["adaptation"][:] = np.linspace(.001, .005, 5, dtype=np.float32)
    changed["drive"][:] = np.linspace(.1, .5, 5, dtype=np.float32)
    changed["output_mask"][2] = 0
    changed["tick"] = 7
    old.restore(changed)
    source_csr = source_matrix(old)
    candidate, original, actual = layer.make_candidate(
        old, np.array([0, 1], np.int64), np.array([0, 1], np.int64),
        np.array([1., 1.125], np.float64))
    assert type(candidate) is SSPRecoveryResourceNetwork
    assert candidate.identity != old.identity
    assert actual[0].tobytes() == original[0].tobytes()
    assert actual[1] != original[1]
    assert source_matrix(old).data.tobytes() == source_csr.data.tobytes()
    for key in layer.DYNAMIC:
        layer.require_equal(candidate.snapshot()[key], changed[key], key)


def _fixture_module():
    path = Path(__file__).with_name("test_c_banc_ssp_recovery_trial.py")
    spec = importlib.util.spec_from_file_location("generated_ssp_resource", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_whole_layer_roundtrip_tamper_and_failed_ownership(monkeypatch, tmp_path):
    from flylab.c.storage import StateStore

    generated = _fixture_module()
    generated.fakes.__wrapped__(monkeypatch)
    native_matrix = generated.FakeNetwork().matrix
    source_api = generated.API
    monkeypatch.setattr(layer.source, "network_api", lambda: source_api)

    def fake_candidate(old, positions, groups, theta):
        patched, original, actual = layer.patch_csr(native_matrix, positions,
                                                    groups, theta)
        kwargs = copy.deepcopy(old.kwargs)
        candidate = generated.FakeResource(
            patched, *(np.ones(4, np.float32) for _ in range(4)), **kwargs)
        snapshot = candidate.snapshot()
        old_snapshot = old.snapshot()
        for key in layer.DYNAMIC:
            snapshot[key] = copy.deepcopy(old_snapshot[key])
        candidate.restore(snapshot)
        return candidate, original, actual

    monkeypatch.setattr(layer, "make_candidate", fake_candidate)
    base = generated.wrap()
    base.network.advance(np.array([.2, .4, .3, .1], np.float32), 2)
    before = base.snapshot()
    body, adapter = base.body, base.adapter
    positions = np.array([0, 1], np.int64)
    groups = np.array([0, 1], np.int64)
    theta = np.array([1., 1.25], np.float64)
    candidate = layer.SSPEfficacyTrial(base, positions, groups, theta)
    assert base.closed and candidate.body is body and candidate.adapter is adapter
    assert body.close_calls == 0
    saved = candidate.snapshot()
    layer.source.require_saved_whole_equal(
        layer._source_representation(saved, saved[layer.PATCH_KEY]), before,
        "Entire source whole state")
    candidate.save(tmp_path / "whole")
    loaded = StateStore.load(tmp_path / "whole")
    layer.source.require_saved_whole_equal(loaded, saved, "Disk whole state")
    fresh = layer.SSPEfficacyTrial.from_checkpoint(candidate.graph, loaded)
    layer.source.require_saved_whole_equal(fresh.snapshot(), saved,
                                           "Restored efficacy whole state")
    broken = copy.deepcopy(saved)
    broken[layer.PATCH_KEY]["actual32"][0] *= 2
    with pytest.raises(ValueError):
        layer.SSPEfficacyTrial.from_checkpoint(candidate.graph, broken)
    assert body.close_calls == 0
    broken = copy.deepcopy(saved)
    broken["network"]["rate"][0] = np.nan
    with pytest.raises(ValueError):
        layer.SSPEfficacyTrial.from_checkpoint(candidate.graph, broken)
    assert body.close_calls == 0

    another = generated.wrap()
    original_body = another.body
    def broken_candidate(*_args):
        raise RuntimeError("generated candidate constructor failure")
    monkeypatch.setattr(layer, "make_candidate", broken_candidate)
    with pytest.raises(RuntimeError):
        layer.SSPEfficacyTrial(another, positions, groups, theta)
    assert not another.closed and original_body.close_calls == 0
