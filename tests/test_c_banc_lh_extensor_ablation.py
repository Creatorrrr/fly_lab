"""Generated CPU contracts for E260; no original arrays, model, or CUDA."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tools import probe_c_banc_m3_lh_extensor_ablation as mod
from tools import probe_c_banc_m3_lh_neural_response as io


def test_single_existing_csr_slot_retains_all_other_bytes(monkeypatch):
    monkeypatch.setattr(mod, "CUT_EDGE", 2)
    ptr = np.array([0, 1, 1, 3, 3], np.int32)
    indices = np.array([0, 1, 3], np.int32)
    values = np.array([1., -2., -0.8800315856933594], np.float32)
    csr = csr_matrix((values, indices, ptr), shape=(4, 4))
    monkeypatch.setattr(mod, "EDGE_PRE", 3)
    monkeypatch.setattr(mod, "EDGE_POST", 2)
    cut = mod.cut_one_existing_edge(csr, {"edge_id": 2, "pre_id": 3,
        "post_id": 2, "source_value": -0.8800315856933594,
        "target_value": 0.0})
    assert cut.data[2].tobytes() == np.float32(0).tobytes()
    np.testing.assert_array_equal(cut.data[:2].view(np.uint32),
                                  csr.data[:2].view(np.uint32))
    np.testing.assert_array_equal(cut.indices, csr.indices)
    np.testing.assert_array_equal(cut.indptr, csr.indptr)
    assert csr.data[2] < 0
    with pytest.raises(ValueError, match="binding"):
        mod.cut_one_existing_edge(csr, {"edge_id": 2,
            "source_value": -1., "target_value": 0.})


def test_f64_patch_values_exact_f32_roundtrip_only():
    original = np.array([0., -.0, .125, -.5, 1.25, -2.], np.float32)
    actual = mod.exact_float32_patch_values(original.astype(np.float64))
    np.testing.assert_array_equal(actual.view(np.uint32), original.view(np.uint32))
    lossy = original.astype(np.float64)
    lossy[2] += 1e-12
    with pytest.raises(ValueError, match="lossy"):
        mod.exact_float32_patch_values(lossy)
    with pytest.raises(ValueError, match="float64"):
        mod.exact_float32_patch_values(original)
    with pytest.raises(ValueError, match="float64"):
        mod.exact_float32_patch_values(np.full(6, np.nan, np.float64))


def test_two_target_incoming_union_preserves_original_edge_addresses():
    ptr = np.zeros(mod.N + 1, np.int32)
    ptr[mod.TARGETS[0] + 1:] += 2
    ptr[mod.TARGETS[1] + 1:] += 2
    csr = csr_matrix((np.array([1., -2., 3., 4.], np.float32),
                      np.array([8, 9, 10, 8], np.int32), ptr),
                     shape=(mod.N, mod.N))
    observed = np.arange(440, dtype=np.int64)
    out = mod.incoming_two(csr, mod.TARGETS, observed,
                           np.ones(mod.N, np.float32))
    np.testing.assert_array_equal(out["edge_ids"], [0, 1, 2, 3])
    np.testing.assert_array_equal(out["target_offsets"], [0, 2, 4])
    np.testing.assert_array_equal(out["post_ids"], [mod.TARGETS[0]] * 2
                                  + [mod.TARGETS[1]] * 2)
    np.testing.assert_array_equal(out["weights32"], csr.data)
    assert len(out["union_ids"]) == 440


def test_original_expm1_bank_and_hinge_with_nonzero_start():
    bank0 = np.array([[.2, .3, .4], [.5, .6, .7]], np.float64)
    rate = np.array([20., 5.], np.float32)
    tau = np.array([.01, .02, .03], np.float64)
    kappa = np.array([.1, .55], np.float64)
    slopes = np.array([[[.1, .2], [.2, .3], [.3, .4]],
                       [[.4, .5], [.5, .6], [.6, .7]]], np.float64)
    bank, activation = mod.bank_activation_step(
        bank0, rate, tau, kappa, slopes, np.array([44, 44]))
    expected = bank0 + (-np.expm1(-.005 / tau))[None, :] * (rate[:, None] - bank0)
    np.testing.assert_array_equal(bank, expected)
    expected_activation = (np.sum(slopes[:, :, 0] * expected)
        + np.sum((slopes[:, :, 1] - slopes[:, :, 0])
                 * np.maximum(expected - kappa[:, None], 0.)))
    assert activation == expected_activation
    np.testing.assert_array_equal(bank0, [[.2, .3, .4], [.5, .6, .7]])
    with pytest.raises(ValueError, match="binding"):
        mod.bank_activation_step(bank0, rate, tau, kappa, slopes, [44, 43])


def test_five_arm_contrasts_and_signed_endpoint_cancellation():
    values = {"B": np.array([0., 0.]), "C": np.array([-3., -2.]),
              "X": np.array([-2.8, -1.9]), "Bcut": np.array([1., 1.]),
              "Xcut": np.array([.9, .9])}
    result = mod.ablation_contrasts(values)
    np.testing.assert_allclose(result["T"], [-3., -2.])
    np.testing.assert_allclose(result["C0"], [-2.8, -1.9])
    np.testing.assert_allclose(result["Ccut"], [-.1, -.1])
    np.testing.assert_allclose(result["R"], result["C0"] - result["Ccut"])
    assert mod.signed_window_parts(np.array([3., -2.]), 0, 2) == {
        "signed": 1., "positive": 3., "negative": -2.}
    with pytest.raises(ValueError):
        mod.ablation_contrasts({**values, "C": np.array([np.nan, 1.])})


def test_diagonal_checks_saved_motor_cohorts_bank_and_activation():
    raw = {key: np.zeros((2, width), np.float32)
           for key, width in (("motor-rate.npy", 391),
                              ("observed-rate.npy", 82),
                              ("observed-drive.npy", 82),
                              ("observed-adaptation.npy", 82),
                              ("core-rate.npy", 24),
                              ("core-drive.npy", 24),
                              ("core-adaptation.npy", 24))}
    raw["bank.npy"] = np.zeros((2, 2, 3), np.float64)
    raw["activation44.npy"] = np.zeros(2, np.float64)
    reference = {k: v.copy() for k, v in raw.items()}
    mod.diagonal_exact(raw, reference, 1)
    reference["core-rate.npy"][1, 0] = -0.0
    with pytest.raises(AssertionError, match="core-rate"):
        mod.diagonal_exact(raw, reference, 1)
    reference["core-rate.npy"][1, 0] = 0.0
    reference["activation44.npy"][1] = 2e-10
    with pytest.raises(AssertionError, match="activation"):
        mod.diagonal_exact(raw, reference, 1)


def _raw(level: float, *, target_rate_dynamic=True):
    union = np.array([*range(200)], np.int64)
    rates = np.full((101, 391), 1., np.float32)
    if target_rate_dynamic:
        rates[1:100, 107] += np.arange(99, dtype=np.float32) / 100
    return {"motor-rate.npy": rates,
            "endpoint-drive.npy": np.zeros((101, len(union)), np.float32),
            "endpoint-adaptation.npy": np.zeros((101, len(union)), np.float32),
            "actual-current.npy": np.zeros((101, 2), np.float32),
            "edge-current64.npy": np.zeros((101, 2), np.float64),
            "bank.npy": np.zeros((101, 2, 3), np.float64),
            "activation44.npy": np.r_[0., np.full(99, level), 0.]}


def test_h260_sign_ratio_and_cut_rate_guard():
    incoming = {"union_ids": np.array([31172, 47329]),
                "edge_ids": np.array([4, mod.CUT_EDGE]),
                "pre_ids": np.array([3, mod.EDGE_PRE]),
                "post_ids": np.array([31172, mod.EDGE_POST]),
                "target_offsets": np.array([0, 1, 2])}
    raws = {arm: _raw(level) for arm, level in
            {"B": 10., "C": 5., "X": 5.5, "Bcut": 11., "Xcut": 10.8}.items()}
    output, report = mod.factors(raws, incoming, np.array([7.5, 7.5], np.float32))
    h = report["H260"]
    assert h["H260_existing_edge_attenuation"] is True
    assert h["T"] == -495.
    assert abs(h["C0"] - (-445.5)) < 1e-9
    assert output["activation44_R"].shape == (101, 1)
    assert output["margin64_B"][0, 0] == -7.5
    raws["Xcut"] = _raw(10.8, target_rate_dynamic=False)
    assert mod.factors(raws, incoming, np.array([7.5, 7.5], np.float32))[1][
        "H260"]["H260_existing_edge_attenuation"] is False


class FakeNetwork:
    def __init__(self):
        self.tick = 999
        self.rate = np.array([1., 2., 3.], np.float32)
        self.drives = []
        self.fail_advance = False

    def snapshot(self):
        return {"tick": self.tick, "rate": self.rate.copy()}

    def restore(self, saved):
        self.tick = saved["tick"]
        self.rate = saved["rate"].copy()

    def advance(self, drive, steps):
        self.drives.append(drive.copy())
        if self.fail_advance:
            raise RuntimeError("partial advance")
        self.rate += drive
        self.tick += steps


def _fake_run(net, entry, raw, *, failed_observation=False):
    saved = {"tick": 0, "rate": np.zeros(3, np.float32)}
    data = {"sensory": {"old_filtered": np.ones((2, 1), np.float64),
             "source_ids": np.array([0]),
             "returned_source_drive": np.ones((2, 1), np.float32)},
            "sensory_ids": np.array([0]), "sensory_ports": np.array([0])}

    def reconstruct(old, ids, ports, n, desc, gain, nav, forward):
        sensory = np.zeros(n, np.float32)
        sensory[ids] = old[ports].astype(np.float32)
        return sensory, sensory.copy()

    def observe(network, row, incoming, arrays, receipt, context):
        if failed_observation and row == 1:
            raise RuntimeError("poststep readback")
        receipt["observed_endpoint_count"] = row + 1

    mod.run_arm(net, saved, data, {}, np.arange(3), np.array([], np.int64),
                np.array([], np.int64), raw, entry, {}, lambda: None,
                lambda: None, observe=observe, controls=2, n=3, steps=1,
                reconstruct=reconstruct)


@pytest.mark.parametrize("failure", ["advance", "observation"])
def test_partial_neural_work_is_not_zero_filled(tmp_path, failure):
    net = FakeNetwork()
    net.fail_advance = failure == "advance"
    drive = np.lib.format.open_memmap(tmp_path / "drive.npy", mode="w+",
                                      dtype=np.float32, shape=(2, 3))
    drive[:] = np.nan
    raw = {"drive-values.npy": drive}
    entry = {"reset_exact": False, "entered_controls": 0,
             "completed_controls": 0, "unknown_neural_partial": False,
             "status": "requested"}
    with pytest.raises(RuntimeError):
        _fake_run(net, entry, raw, failed_observation=failure == "observation")
    assert entry["entered_controls"] == 1
    assert entry["completed_controls"] == (failure == "observation")
    assert entry["unknown_neural_partial"] == (failure == "advance")
    assert np.isfinite(raw["drive-values.npy"][0]).all()
    assert np.isnan(raw["drive-values.npy"][1]).all()


def test_dynamic_start_only_allows_identity_metadata():
    a = {"rate": np.array([1., 2.], np.float32), "tick": 0,
         "model_hash": "a", "core_adaptation": {},
         "motor_adaptation": {}, "motor_timescale": {}}
    b = copy.deepcopy(a)
    b["model_hash"] = "b"
    mod.dynamic_start_equal(a, b, exact=lambda x, y, label:
                            np.testing.assert_array_equal(x["rate"], y["rate"]))
    b["rate"][0] = 3.
    with pytest.raises(AssertionError):
        mod.dynamic_start_equal(a, b, exact=lambda x, y, label:
                                np.testing.assert_array_equal(x["rate"], y["rate"]))


def test_gate_rejects_wrong_sha_before_original_loading(tmp_path):
    with pytest.raises(ValueError, match="freeze/review SHA"):
        mod.gate(Path("missing"), "wrong", Path("missing-review"), "wrong",
                 tmp_path / "output")


@pytest.mark.parametrize("fault", ["prior_cleanup", "late_manifest"])
def test_late_failure_revokes_nested_h260_and_keeps_numeric_raw(tmp_path, fault):
    (tmp_path / "raw.npy").write_bytes(b"retained")
    io.write_json(tmp_path / "factor-windows.json", {"H260": {
        "T": -4., "H260_existing_edge_attenuation": True}})
    ledger = {"models": {}, "arms": {}, "resources": []}
    result = {"validity": mod.VALIDITY,
              "H260_existing_edge_attenuation": True,
              "factor_report_available": True,
              "H260": {"T": -4., "C0": -3.8, "Ccut": -.1,
                       "H260_existing_edge_attenuation": True}}
    errors = []

    def persist(path, value):
        if fault == "late_manifest" and path.name == "output-manifest.json":
            raise OSError("manifest failed")
        io.write_json(path, value)

    old = OSError("cleanup failed") if fault == "prior_cleanup" else None
    error = mod.finish_receipts(tmp_path, {"resource_limits": {
        "internal_s": 55, "output_bytes": 100000}}, ledger, result, errors,
        old, time.perf_counter(), persist=persist)
    assert isinstance(error, OSError)
    assert str(error) == ("cleanup failed" if old else "manifest failed")
    assert result["H260_existing_edge_attenuation"] is False
    assert result["H260"]["H260_existing_edge_attenuation"] is False
    assert result["H260"]["T"] == -4.
    assert result["validity"] == "invalid_or_incomplete"
    assert (tmp_path / "raw.npy").read_bytes() == b"retained"
    factor = io.read_json(tmp_path / "factor-windows.json")
    assert factor["H260"]["T"] == -4.
    assert factor["H260"]["H260_existing_edge_attenuation"] is False
    saved = io.read_json(tmp_path / "result.json")
    assert saved["H260"]["H260_existing_edge_attenuation"] is False


def test_failed_factor_rewrite_moves_stale_report_to_unvalidated_raw(tmp_path):
    io.write_json(tmp_path / "factor-windows.json", {"H260": {
        "T": -4., "H260_existing_edge_attenuation": True}})
    (tmp_path / "factor-effects.npz").write_bytes(b"numeric arrays retained")
    ledger = {"models": {}, "arms": {}}
    result = {"validity": mod.VALIDITY,
              "H260_existing_edge_attenuation": True,
              "H260": {"T": -4., "H260_existing_edge_attenuation": True}}
    errors = []

    def persist(path, value):
        if path.name == "factor-windows.json":
            raise OSError("factor rewrite denied")
        io.write_json(path, value)

    primary = OSError("original science failure")
    got = mod.finish_receipts(tmp_path, {"resource_limits": {
        "internal_s": 55, "output_bytes": 100000}}, ledger, result, errors,
        primary, time.perf_counter(), persist=persist)
    assert got is primary
    assert not (tmp_path / "factor-windows.json").exists()
    moved = io.read_json(tmp_path / "factor-windows-unvalidated-raw.json")
    assert moved["H260"]["T"] == -4.
    assert (tmp_path / "factor-effects.npz").exists()
    assert any(x["stage"] == "factor report invalidation" for x in errors)
    assert io.read_json(tmp_path / "result.json")["H260"][
        "H260_existing_edge_attenuation"] is False
