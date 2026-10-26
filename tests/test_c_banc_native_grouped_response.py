"""Generated CPU contract checks for E221; no CUDA model is constructed."""

from __future__ import annotations

import numpy as np
import pytest

from tools.banc_m3_native_grouped_response import (
    PRECISION_CONTRACT,
    NativeGroupedResponseNetwork,
    _native_source,
    _parameter_fingerprint,
    native_response_derivative,
    validate_native_parameters,
)


def parameters():
    motor = np.array([1, 6, 3, 5], np.int64)
    free = np.array([1, 3, 5], np.int64)
    fixed = np.array([6], np.int64)
    owner = np.array([0, 0, 1], np.int64)
    scale = np.ones((3, 3), np.float64)
    centers = np.zeros((3, 64, 9), np.float64)
    sigma = np.ones(3, np.float64)
    beta = np.full((2, 64), .5, np.float64)
    tau = np.full(3, .02, np.float64)
    cap = np.full(3, 10., np.float64)
    slow = np.full(3, 3., np.float64)
    return motor, free, fixed, owner, scale, centers, sigma, beta, tau, cap, slow


def test_ordered_partition_and_ownership() -> None:
    original = parameters()
    checked = validate_native_parameters(8, *original)
    assert all(np.array_equal(a, b) for a, b in zip(checked, original))
    checked[4][0, 0] = 7
    assert original[4][0, 0] == 1
    bad = list(original)
    bad[1] = np.array([3, 1, 5], np.int64)
    with pytest.raises(ValueError, match="partition"):
        validate_native_parameters(8, *bad)
    bad = list(original)
    bad[3] = np.array([0, 0, 2], np.int64)
    with pytest.raises(ValueError, match="bounded"):
        validate_native_parameters(8, *bad)
    sparse = list(original)
    sparse[3] = np.array([0, 0, 2], np.int64)
    sparse[7] = np.vstack((original[7][0], np.zeros(64), original[7][1]))
    assert validate_native_parameters(8, *sparse)[3].tolist() == [0, 0, 2]
    sparse[7][1, 7] = .25
    with pytest.raises(ValueError, match="zero beta"):
        validate_native_parameters(8, *sparse)


def test_reaction_and_signed_drive_cpu() -> None:
    _, _, _, owner, scale, centers, sigma, beta, tau, cap, slow = parameters()
    rate = np.array([0., 1., 2.], np.float64)
    fast = np.zeros((3, 3), np.float64)
    history = np.zeros_like(fast)
    zero = np.zeros(3, np.float64)
    dr, df, ds = native_response_derivative(zero, zero, zero, rate,
        fast, history, scale, centers, sigma, beta, owner, tau, cap, slow)
    np.testing.assert_array_equal(dr, -rate / tau)
    np.testing.assert_array_equal(df, 0)
    np.testing.assert_array_equal(ds, 0)
    e = np.array([2., 0., 1.], np.float64)
    d = np.array([-1., 3., 0.], np.float64)
    dr, df, ds = native_response_derivative(e, zero, d, rate, fast,
        history, scale, centers, sigma, beta, owner, tau, cap, slow)
    x = np.stack((np.log1p(e), zero, np.arcsinh(d)), axis=1)
    gate = -np.expm1(-8 * (x[:, 0] + np.abs(x[:, 2])))
    np.testing.assert_allclose(dr, (cap * gate * .5 - rate) / tau)
    np.testing.assert_allclose(df, x / .016)
    np.testing.assert_allclose(ds, x * slow[:, None])


def test_csr_total_order_and_native_stage_contract() -> None:
    source = _native_source(True)
    assert source.count("value += lookup[item >> 18] * masked[item & 0x3ffff];") == 1
    assert source.count("value = add_exact(drive[row], value);") == 1
    assert "if (weight > 0.0f) e += product;" in source
    assert "if (weight < 0.0f) inhibitory += -product;" in source
    assert "next_masked[i] = mul_exact(nr, mask[i]);" in source
    assert "masked[row]=mul_exact(cast,mask[row]);" in source
    assert 'extern "C" __global__ void native_stage(' in source
    assert 'extern "C" __global__ void native_finish(' in source
    assert "fmax(fast[3*j],0.)" in source
    assert "beta[64*owner[j]+a]" in source


def test_bad_native_state_rejected_before_live_base_restore() -> None:
    network = object.__new__(NativeGroupedResponseNetwork)
    (network.native_motor_ids, network.free_motor_ids, network.fixed_motor_ids,
     network.membership, network.scale, network.centers, network.sigma2,
     network.beta, network.free_tau, network.free_cap,
     network.inverse_slow) = parameters()
    network._free_count = 3
    network.n = 8
    network.identity = "candidate"
    network.source_timescale_identity = "source"
    network._native_fingerprint = "test-fingerprint"
    network._assert_native_parameters = lambda: None
    metadata = {"schema": "flylab.m3.native-grouped-response.v1",
        "precision_contract": PRECISION_CONTRACT,
        "parameter_fingerprint": "test-fingerprint",
        "source_timescale_identity": "source", "motor_ids": network.native_motor_ids.copy(),
        "free_ids": network.free_motor_ids.copy(), "fixed_ids": network.fixed_motor_ids.copy(),
        "membership": network.membership.copy(), "scale": network.scale.copy(),
        "centers": network.centers.copy(), "sigma2": network.sigma2.copy(),
        "beta": network.beta.copy(), "tau": network.free_tau.copy(),
        "cap": network.free_cap.copy(), "inverse_slow": network.inverse_slow.copy(),
        "rate64": np.zeros(3, np.float64), "fast64": np.zeros((3, 3), np.float64),
        "slow64": np.zeros((3, 3), np.float64)}
    saved = {"model_hash": "candidate", "rate": np.zeros(8, np.float32),
             "native_grouped_response": metadata}
    metadata["fast64"] = np.zeros((3, 3), np.float32)
    with pytest.raises(ValueError, match="state"):
        network.restore(saved)
    metadata["fast64"] = np.zeros((3, 3), np.float64)
    metadata["rate64"][0] = .1
    with pytest.raises(ValueError, match="full float32"):
        network.restore(saved)
    metadata["rate64"][0] = 0
    metadata["beta"][0, 0] = .25
    with pytest.raises(ValueError, match="parameter bytes"):
        network.restore(saved)


class _FakeTensor:
    def __init__(self, value):
        self.value = np.array(value, copy=True)

    def data_ptr(self):
        return id(self.value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def test_captured_mapping_bytes_and_immutable_fingerprint() -> None:
    values = list(parameters())
    values[8] = np.full(3, np.float64(np.float32(.02)), np.float64)
    network = object.__new__(NativeGroupedResponseNetwork)
    (network.native_motor_ids, network.free_motor_ids, network.fixed_motor_ids,
     network.membership, network.scale, network.centers, network.sigma2,
     network.beta, network.free_tau, network.free_cap,
     network.inverse_slow) = values
    network.n = 8
    network._free_count = 3
    network._native_fingerprint = _parameter_fingerprint(values)
    mapping = np.full(8, -1, np.int32)
    mapping[network.free_motor_ids] = np.arange(3, dtype=np.int32)
    network._free_row = _FakeTensor(network.free_motor_ids.astype(np.int32))
    network._owner = _FakeTensor(network.membership.astype(np.int32))
    network._free_index = _FakeTensor(mapping)
    for key in ("scale", "centers", "sigma2", "beta", "free_tau",
                "free_cap", "inverse_slow"):
        setattr(network, "_d_" + key, _FakeTensor(getattr(network, key)))
    tau = np.full(8, np.float32(.02), np.float32)
    cap = np.full(8, np.float32(10), np.float32)
    network.tau = _FakeTensor(tau)
    network.cap = _FakeTensor(cap)
    network._full_inverse32 = np.full(8, np.float32(3), np.float32)
    names = ("_free_row", "_owner", "_free_index") + tuple("_d_" + key for key in
        ("scale", "centers", "sigma2", "beta", "free_tau",
         "free_cap", "inverse_slow"))
    network._native_pointers = {key: getattr(network, key).data_ptr() for key in names}
    network._assert_native_parameters()
    for key in ("_free_row", "_owner", "_free_index"):
        tensor = getattr(network, key)
        old = tensor.value.copy()
        tensor.value.flat[0] += 1
        with pytest.raises(ValueError, match="map changed"):
            network._assert_native_parameters()
        tensor.value[:] = old
    network.beta[0, 0] = .75
    network._d_beta.value[0, 0] = .75
    with pytest.raises(ValueError, match="construction-time"):
        network._assert_native_parameters()
