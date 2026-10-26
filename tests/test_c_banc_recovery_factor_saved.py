"""E298 generated algebra/sparse fixtures, no original/model/neural/body/CUDA."""
import copy
import json
import time

import numpy as np
import psutil
import pytest
from threadpoolctl import threadpool_limits

from tools import analyze_c_banc_m3_recovery_factor_saved as probe

BEGAN = time.perf_counter()


@pytest.fixture(autouse=True)
def one_cpu_thread():
    with threadpool_limits(limits=1):
        yield


def factors(rate=None, resource=None):
    r = np.array([100, 200, 300, 0], np.float32) if rate is None else rate
    state = np.full(r.shape, .4, np.float64) if resource is None else resource
    return probe.recovery_factor(r, state, np.full(r.shape[-1], .05), np.full(r.shape[-1], .15))


def test_hand_calculation_and_zero_rate_loss():
    result = factors()
    np.testing.assert_allclose(result["g64"], [.7, 1, 1, .4], rtol=0, atol=2e-16)
    np.testing.assert_allclose(result["Rdot"], [2, 0, -2, 4], rtol=0, atol=1e-15)
    assert result["g32"].dtype == result["old32"].dtype == np.float32
    assert result["new_source_loss"][-1] == 0 and result["g32"][-1] < 1
    assert np.all(result["removed_source_loss"] >= 0)


def test_constant_rate_analytical_trajectory_and_low_arbitrary_state():
    rate = 100.
    equilibrium = 1 / (1 + .15 * .05 * rate)
    times = np.linspace(0, 2, 100)
    resource = equilibrium + (1 - equilibrium) * np.exp(-(1 / .15 + .05 * rate) * times)
    result = factors(np.full((100, 1), rate, np.float32), resource[:, None])
    assert np.all(result["g32"] == 1)
    assert factors()["g32"][0] < 1  # Constant input alone is insufficient if R<Rinf.
    r, state = 150., .4
    assert state * (1 + .15 * .05 * r) < 1
    assert state * (1 + 2 * .15 * .05 * r) > 1  # Uncapped slope does not prove stability.


def test_float32_unity_rounding_and_inactive_exact_branch():
    state = np.array([np.nextafter(1., 0.), 1.], np.float64)
    result = probe.recovery_factor(np.zeros(2, np.float32), state, np.array([.05, 0.]), np.full(2, .15))
    assert result["g64"][0] < 1 and result["g32"][0] == 1
    assert result["g64"][1] == result["old32"][1] == result["g32"][1] == 1


@pytest.mark.parametrize("fault", ["negative_rate", "nan", "rate_dtype", "resource_dtype", "shape", "negative_R", "R_above1", "inactive_R", "negative_u", "tau0"])
def test_factor_strict_boundaries(fault):
    rate, state, u, tau = np.ones((2, 2), np.float32), np.ones((2, 2), np.float64), np.full(2, .05), np.full(2, .15)
    if fault == "negative_rate":
        rate[1, 0] = -1e-6
    elif fault == "nan":
        rate[0, 0] = np.nan
    elif fault == "rate_dtype":
        rate = rate.astype(np.float64)
    elif fault == "resource_dtype":
        state = state.astype(np.float32)
    elif fault == "shape":
        state = state[:1]
    elif fault == "negative_R":
        state[0, 0] = -.1
    elif fault == "R_above1":
        state[0, 0] = 1.01
    elif fault == "inactive_R":
        u[0], state[0, 0] = 0, .9
    elif fault == "negative_u":
        u[0] = -.05
    else:
        tau[0] = 0
    with pytest.raises(ValueError):
        probe.recovery_factor(rate, state, u, tau)


def sparse_fixture():
    ids = np.arange(3, dtype=np.int64)
    s = {"premotor_ids": ids, "motor_ids": np.array([3], np.int64), "source_ids": ids.copy(), "edge_ids": np.arange(4, dtype=np.int64),
         "pre_ids": np.array([0, 0, 1, 2], np.int64), "post_ids": np.array([0, 1, 1, 2], np.int64),
         "edge_source_slots": np.array([0, 0, 1, 2], np.int32), "weights32": np.array([2, 4, -3, 5], np.float32),
         "utilization": np.array([.05, 0, 0]), "recovery_tau": np.full(3, .15)}
    ticks = np.arange(101, dtype=np.int64)
    rate = np.tile(np.array([20, 30, 100, 40], np.float32), (101, 1))  # Unsorted observed2,3,0,1.
    resource = np.ones((101, 3), np.float64)
    resource[1:, 0] = .4
    a = {"observed_ids": np.array([2, 3, 0, 1], np.int64), "source_ids": ids.copy(), "motor_ids": s["motor_ids"].copy(),
         "control_tick": ticks, "network_tick": ticks * 20, "physics_time_s": ticks * .005, "complete": np.ones(101, bool),
         "rate": rate, "drive": np.zeros_like(rate), "adaptation": np.zeros_like(rate), "synaptic_resource": resource}
    csr = {"indptr": np.array([0, 1, 3, 4, 4], np.int32), "indices": s["pre_ids"].astype(np.int32), "data": s["weights32"].copy()}
    return s, a, np.array([2], np.int64), csr


def test_unsorted_shared_source_sparse_loss_and_active_only_summaries():
    values = sparse_fixture()
    before = copy.deepcopy(values)
    source, target = probe.saved_proxy(*values)
    np.testing.assert_array_equal(source["rate"][1], [100, 40, 20])
    np.testing.assert_allclose(target["old_positive_loss"][1], [120, 240, 0], rtol=1e-7)
    np.testing.assert_allclose(target["new_positive_loss"][1], [60, 120, 0], rtol=1e-7)
    assert np.all(target["old_positive_loss"][0] == 0)
    assert np.all(source["g32"][:, 1:] == 1)
    summary = probe.fixed_summary(source, target)
    assert [(row["first_control"], row["last_control"]) for row in summary] == [(0, 0), (1, 100), (1, 10), (11, 20), (21, 100)]
    assert summary[1]["active_sources"] == 1 and summary[1]["unity_fraction_active_only"] == 0
    assert summary[1]["suppressed_source_endpoints"] == 100 and summary[1]["any_suppressed_sources"] == 1
    for value, saved in zip(values, before, strict=True):
        if isinstance(value, dict):
            for key in value:
                probe.p.equal_array(value[key], saved[key], "Input unchanged")


def test_nontrivial_factor_with_zero_rate_has_no_current_effect():
    values = sparse_fixture()
    values[1]["rate"][:] = 0
    source, target = probe.saved_proxy(*values)
    flags = probe.nontrivial_flags(source, target)
    assert flags["factor_nontrivial"] and not flags["current_proxy_nontrivial"]
    assert not flags["source_current_proxy_nontrivial"] and flags["observed_saved_proxy_degenerate"]
    assert np.all(source["new_source_loss"] == 0) and np.all(target["new_positive_loss"] == 0)


@pytest.mark.parametrize("fault", ["duplicate", "missing", "clock", "all_rate_negative", "weights", "slot", "core_u", "negative_u", "inactive_R"])
def test_sparse_alignment_original_bytes_and_exemption_rejections(fault):
    s, a, core, csr = sparse_fixture()
    if fault == "duplicate":
        a["observed_ids"][1] = a["observed_ids"][0]
    elif fault == "missing":
        a["observed_ids"][0] = 4
    elif fault == "clock":
        a["physics_time_s"][5] += .001
    elif fault == "all_rate_negative":
        a["rate"][1, 1] = -1e-6  # Even motor/non-source raw rates must be nonnegative.
    elif fault == "weights":
        csr["data"][0] += 1
    elif fault == "slot":
        s["edge_source_slots"][0] = 1
    elif fault == "core_u":
        s["utilization"][2] = .05
    elif fault == "negative_u":
        s["utilization"][1] = .05
    else:
        a["synaptic_resource"][1, 1] = .9
    with pytest.raises(ValueError):
        probe.saved_proxy(s, a, core, csr)


def test_failed_budget_receipt_and_fresh_output(tmp_path, monkeypatch):
    output = tmp_path / "out"
    # Own budget failure exercises publication without any dataset loading.
    monkeypatch.setattr(probe, "LIMITS", {**probe.LIMITS, "wall_s": 0})
    with pytest.raises(TimeoutError):
        probe.run({"inputs": {}}, output)
    receipt = json.loads((output / "failure.json").read_text())
    assert not receipt["analysis_valid"] and not receipt["kernel_implemented"] and not receipt["gait_improved"]
    assert all(value == 0 for value in receipt["execution"].values())
    assert (output / "source-manifest.json").exists() and not (output / "result.json").exists()
    with pytest.raises(ValueError, match="Fresh"):
        probe.run({"inputs": {}}, output)


def test_gate_sha_and_generated_cost(tmp_path):
    file = tmp_path / "bad.json"
    file.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        probe.gate(file, "0" * 64, file, "0" * 64, tmp_path / "out")
    memory = psutil.Process().memory_info()
    peak = getattr(memory, "peak_wset", memory.rss)
    elapsed = time.perf_counter() - BEGAN
    assert elapsed < 45 and peak < 1 << 30
    print(json.dumps({"generated_wall_s": elapsed, "host_peak_bytes": peak,
                      **{key: 0 for key in ("original", "model", "neural", "body", "physics", "CUDA", "solver", "fit")}}))
