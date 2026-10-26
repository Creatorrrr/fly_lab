"""Bounded generated E312 checks; root alone opts into CUDA qualification."""

import copy
import json
import os
import time

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from flylab.c.adaptive_rate import AdaptiveRateNetwork
from flylab.c.storage import StateStore
from tests.test_c_banc_recovery_resource_synapses import parameters, peak_rss_bytes
from tools import banc_m3_postsynaptic_load as p

CUDA_ENABLED = os.environ.get("RUN_E312_CUDA") == "1"
DT = 0.00025
COST = {
    name: 0
    for prefix in (
        "constructor",
        "actual_constructor",
        "step",
        "setup_step",
        "advance",
        "capture",
    )
    for name in (prefix + "_requested", prefix + "_completed", prefix + "_failed")
}
COST.update(
    {
        k: 0
        for k in (
            "logical_ticks",
            "oracle_ticks",
            "CUDA_calls",
            "original_calls",
            "body_calls",
            "solver_calls",
            "fit_calls",
        )
    }
)
COST.update(
    {
        k: 0.0
        for k in (
            "constructor_wall_s",
            "actual_constructor_wall_s",
            "step_wall_s",
            "advance_wall_s",
        )
    }
)


@pytest.fixture(scope="module", autouse=True)
def receipt():
    import torch

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    wall, cpu = time.perf_counter(), time.process_time()
    yield
    COST.update(
        wall_s=time.perf_counter() - wall,
        process_s=time.process_time() - cpu,
        peak_rss_bytes=peak_rss_bytes(),
    )
    torch.set_num_threads(old)
    print("E312_GENERATED_COST " + json.dumps(COST, sort_keys=True))
    assert COST["wall_s"] <= (60 if CUDA_ENABLED else 80)
    assert COST["peak_rss_bytes"] <= 2 * 1024**3
    assert COST["logical_ticks"] + COST["oracle_ticks"] <= 10000


@pytest.fixture(autouse=True)
def meter(monkeypatch):
    def wrap(method, prefix, setup=False):
        def metered(self, *args, **kwargs):
            if prefix == "constructor":
                assert args[0].shape[0] <= 64 and args[0].nnz <= 256
                self._generated_setup = True
                if kwargs.get("device") == "cuda":
                    assert CUDA_ENABLED
                    COST["CUDA_calls"] += 1
                    COST["capture_requested"] += 1
            COST[prefix + "_requested"] += 1
            started = time.perf_counter()
            tick = getattr(self, "tick", 0)
            is_setup = setup and getattr(self, "_generated_setup", False)
            if is_setup:
                COST["setup_step_requested"] += 1
            try:
                result = method(self, *args, **kwargs)
                COST[prefix + "_completed"] += 1
                if is_setup:
                    COST["setup_step_completed"] += 1
                if prefix == "constructor" and kwargs.get("device") == "cuda":
                    COST["capture_completed"] += 1
                return result
            except BaseException:
                COST[prefix + "_failed"] += 1
                if is_setup:
                    COST["setup_step_failed"] += 1
                if prefix == "constructor" and kwargs.get("device") == "cuda":
                    COST["capture_failed"] += 1
                raise
            finally:
                if prefix + "_wall_s" in COST:
                    COST[prefix + "_wall_s"] += time.perf_counter() - started
                if prefix == "advance":
                    COST["logical_ticks"] += getattr(self, "tick", tick) - tick
                if prefix == "constructor":
                    self._generated_setup = False

        return metered

    for cls, name, prefix in (
        (p.PostsynapticLoadNetwork, "__init__", "constructor"),
        (AdaptiveRateNetwork, "__init__", "actual_constructor"),
        (p.PostsynapticLoadNetwork, "_step", "step"),
        (p.PostsynapticLoadNetwork, "_advance", "advance"),
    ):
        monkeypatch.setattr(
            cls, name, wrap(getattr(cls, name), prefix, setup=prefix == "step")
        )
    for name, prefix in (
        ("__init__", "constructor"),
        ("_step", "step"),
        ("_advance", "advance"),
    ):
        method = getattr(p.ssp.SSPRecoveryResourceNetwork, name)
        metered = wrap(method, prefix, setup=prefix == "step")

        def only_baseline(self, *args, _method=method, _metered=metered, **kwargs):
            return (
                _metered(self, *args, **kwargs)
                if type(self) is p.ssp.SSPRecoveryResourceNetwork
                else _method(self, *args, **kwargs)
            )

        monkeypatch.setattr(p.ssp.SSPRecoveryResourceNetwork, name, only_baseline)


def model(value=1.0, device="cpu", args=None, kw=None):
    if args is None:
        args, kw = parameters(device)
    return p.PostsynapticLoadNetwork(
        *args,
        **kw,
        edge_ids=np.arange(4, dtype=np.int64),
        premotor_ids=np.arange(3, dtype=np.int64),
        utilization=np.zeros(3),
        recovery_tau=np.full(3, 0.15),
        load_lambda=value,
    )


def seed(n):
    state = n.snapshot()
    state.update(
        tick=17,
        rate=np.array([4, 3, 2, 5, 1, 6], np.float32),
        adaptation=np.array([0.1, 0.2, 0.15, 0.3, 0.2, 0.1], np.float32),
        drive=np.array([2.8, 3, 2.5, 3.1, 3, 2.7], np.float32),
    )
    state["postsynaptic_load_state"][:] = [0.2, 0.3, 0.1]
    n.restore(state)
    return state


def ftz(value):
    value = np.float32(value)
    return (
        np.copysign(np.float32(0), value)
        if abs(value) < np.finfo(np.float32).tiny
        else value
    )


def independent_current(weights, r, mask, edges, targets, h, strength):
    selected = set(edges.tolist())
    slots = {int(v): i for i, v in enumerate(targets)}
    current = np.zeros(len(r), np.float32)
    gross = np.zeros(len(targets), np.float32)
    for row in range(len(r)):
        lanes = np.zeros(32, np.float32)
        sums = np.zeros(32, np.float32)
        for lane in range(32):
            for edge in range(
                int(weights.indptr[row]) + lane, int(weights.indptr[row + 1]), 32
            ):
                pre = weights.indices[edge]
                v = ftz(ftz(weights.data[edge]) * ftz(np.float32(r[pre] * mask[pre])))
                if edge in selected:
                    sums[lane] = ftz(ftz(sums[lane]) + ftz(abs(v)))
                    if strength != 0:
                        g = np.float32(1.0 / (1.0 + strength * float(h[slots[row]])))
                        v = ftz(ftz(v) * ftz(g))
                lanes[lane] = ftz(ftz(lanes[lane]) + ftz(v))
        for offset in (16, 8, 4, 2, 1):
            old, old_sums = lanes.copy(), sums.copy()
            for lane in range(32 - offset):
                lanes[lane] = ftz(ftz(old[lane]) + ftz(old[lane + offset]))
                sums[lane] = ftz(ftz(old_sums[lane]) + ftz(old_sums[lane + offset]))
        current[row] = lanes[0]
        if row in slots:
            gross[slots[row]] = sums[0]
    return current, gross


def oracle_tick(n, state):
    args, _ = parameters()
    weights, tau, a, threshold, cap = args
    gain = np.array([0, 0, 8, 1, 2, 8], np.float32)
    inverse = (1.0 / np.array([0.3, 0.3, 0.3, 0.3, 0.6, 0.3])).astype(np.float32)
    targets = np.array([0, 1, 2], np.int64)
    r0, z0, h0 = (
        state[k].copy() for k in ("rate", "adaptation", "postsynaptic_load_state")
    )
    r, z, h = r0, z0, h0
    stages = []
    for b, v in ((0.0, 1.0), (0.75, 0.25), (1 / 3, 2 / 3)):
        stages.append((r.copy(), z.copy(), h.copy()))
        current, gross = independent_current(
            weights, r, state["output_mask"], np.arange(4), targets, h, n.load_lambda
        )
        target = np.maximum(
            cap
            * np.tanh(
                np.float32(a / cap) * (((state["drive"] + current) - threshold) - z)
            ),
            np.float32(0),
        )
        qr, qz = np.float32(DT) / tau, np.float32(DT) * inverse
        nr = np.float32(b) * r0 + np.float32(v) * (
            (np.float32(1) - qr) * r + qr * target
        )
        nz = np.float32(b) * z0 + np.float32(v) * (
            (np.float32(1) - qz) * z + qz * (gain * r)
        )
        nh = np.empty(3, np.float64)
        for i in range(3):
            U = float(gross[i])
            q = DT / float(tau[i])
            L = U / (float(threshold[i]) + U)
            nh[i] = b * h0[i] + v * ((1.0 - q) * h[i] + q * L)
        r, z, h = nr, nz, nh
    COST["oracle_ticks"] += 1
    result = copy.deepcopy(state)
    result.update(
        rate=r, adaptation=z, postsynaptic_load_state=h, tick=state["tick"] + 1
    )
    return result, stages


@pytest.mark.parametrize("strength", [0.0, 1.0])
def test_coupled_old_stage_independent_scalar_oracle(strength, monkeypatch):
    n = model(strength)
    seed(n)
    n.set_muted([1])
    state = n.snapshot()
    expected, stages = oracle_tick(n, state)
    old = p.packed_load_oracle
    seen = []

    def observe(weights, r, mask, selection, h, strength):
        seen.append((r.copy(), h.copy()))
        return old(weights, r, mask, selection, h, strength)

    monkeypatch.setattr(p, "packed_load_oracle", observe)
    n.advance(state["drive"], steps=1)
    assert len(seen) == 3
    for got, stage in zip(seen, stages, strict=True):
        np.testing.assert_allclose(got[0], stage[0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(got[1], stage[2], rtol=0, atol=1e-14)
    final = n.snapshot()
    for key in ("rate", "adaptation"):
        np.testing.assert_allclose(final[key], expected[key], rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        final["postsynaptic_load_state"],
        expected["postsynaptic_load_state"],
        rtol=0,
        atol=1e-14,
    )
    assert np.all(final["synaptic_resource"] == 1)


def test_lambda0_original_ssp_exact_while_h_observationally_changes():
    args, kw = parameters()
    n = model(0)
    baseline = p.ssp.SSPRecoveryResourceNetwork(
        *args,
        **kw,
        edge_ids=np.arange(4, dtype=np.int64),
        premotor_ids=np.arange(3, dtype=np.int64),
        utilization=np.zeros(3),
        recovery_tau=np.full(3, 0.15),
    )
    state = seed(n)
    old = baseline.snapshot()
    for k in ("tick", "rate", "adaptation", "drive", "output_mask"):
        old[k] = copy.deepcopy(state[k])
    baseline.restore(old)
    for _ in range(3):
        n.advance(state["drive"], steps=7)
        baseline.advance(state["drive"], steps=7)
        now, ref = n.snapshot(), baseline.snapshot()
        for k in (
            "rate",
            "adaptation",
            "drive",
            "output_mask",
            "tick",
            "synaptic_resource",
        ):
            assert p._equal(now[k], ref[k])
    assert not np.array_equal(
        n.snapshot()["postsynaptic_load_state"], state["postsynaptic_load_state"]
    )


def simple_pair():
    weights = csr_matrix(
        np.array([[0, 1, -1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], np.float32)
    )
    s = p.checked_selection(
        weights,
        np.array([0, 1], np.int64),
        np.arange(3, dtype=np.int64),
        np.zeros(2),
        np.full(2, 0.15),
        np.array([3], np.int64),
    )
    return weights, s


def test_equal_signed_net_different_gross_and_shared_post_history():
    w, s = simple_pair()
    mask = np.ones(4, np.float32)
    a = np.array([0, 2, 1, 0], np.float32)
    b = np.array([0, 5, 4, 0], np.float32)
    current_a, gross_a = p.packed_load_oracle(w, a, mask, s, np.zeros(1), 1.0)
    current_b, gross_b = p.packed_load_oracle(w, b, mask, s, np.zeros(1), 1.0)
    assert current_a[0] == current_b[0] == 1 and gross_a[0] == 3 and gross_b[0] == 9
    for gross in (gross_a, gross_b):
        h = p.load_euler(
            np.zeros(1), gross, np.ones(1, np.float32), np.full(1, 0.03, np.float32), DT
        )
        assert h[0] > 0
    assert (
        p.load_euler(
            np.zeros(1),
            gross_a,
            np.ones(1, np.float32),
            np.full(1, 0.03, np.float32),
            DT,
        )[0]
        < p.load_euler(
            np.zeros(1),
            gross_b,
            np.ones(1, np.float32),
            np.full(1, 0.03, np.float32),
            DT,
        )[0]
    )
    older, _ = p.packed_load_oracle(w, a, mask, s, np.array([0.8]), 1.0)
    younger, _ = p.packed_load_oracle(w, a, mask, s, np.array([0.2]), 1.0)
    assert older[0] < younger[0] < 1  # Same net input, different receiving history.


def wide_parameters(device="cpu"):
    weights = np.zeros((39, 39), np.float32)
    weights[0, 1:39] = np.array(
        [(0.01 if i % 2 else -0.02) for i in range(38)], np.float32
    )
    weights[1, 0] = 0.03
    args = (
        csr_matrix(weights),
        np.full(39, 0.03, np.float32),
        np.ones(39, np.float32),
        np.full(39, 1.5, np.float32),
        np.full(39, 40, np.float32),
    )
    kw = {
        "adaptation_gain": 8.0,
        "adaptation_tau_s": 0.3,
        "dt": DT,
        "device": device,
        "capture_steps": 5,
        "cuda_implementation": "packed",
        "core_ids": np.array([0, 1], np.int64),
        "core_gain": 0.0,
        "motor_ids": np.array([38], np.int64),
        "motor_gains": np.ones(1, np.float32),
        "motor_log_tau": np.zeros(1),
    }
    # Include all premotor incoming edges; row0's motor38 edge is unselected.
    edges = np.r_[np.arange(37), 38].astype(np.int64)
    return args, kw, edges


def wide_model(device="cpu", strength=1.0):
    args, kw, edges = wide_parameters(device)
    return p.PostsynapticLoadNetwork(
        *args,
        **kw,
        edge_ids=edges,
        premotor_ids=np.arange(38, dtype=np.int64),
        utilization=np.zeros(38),
        recovery_tau=np.full(38, 0.15),
        load_lambda=strength,
    )


def test_warp_over32_selected_unselected_mask_and_input_ownership():
    args, kw, edges = wide_parameters()
    w = args[0]
    s = p.checked_selection(
        w,
        edges,
        np.arange(38, dtype=np.int64),
        np.zeros(38),
        np.full(38, 0.15),
        kw["motor_ids"],
    )
    r = np.linspace(0, 50, 39).astype(np.float32)
    mask = np.ones(39, np.float32)
    mask[[7, 9]] = 0
    h = np.array([0.2, 0.8])
    prior = [v.copy() for v in (w.data, w.indices, w.indptr, r, mask, h)]
    got = p.packed_load_oracle(w, r, mask, s, h, 1.0)
    expected = independent_current(w, r, mask, edges, np.array([0, 1]), h, 1.0)
    for a, b in zip(got, expected, strict=True):
        assert a.tobytes() == b.tobytes()
    for a, b in zip(prior, (w.data, w.indices, w.indptr, r, mask, h), strict=True):
        assert a.tobytes() == b.tobytes()
    n = wide_model()
    assert n.load_metadata["target_ids"].tolist() == [0, 1]


def test_metadata_identity_ownership_order_and_zero_reset():
    n = model()
    neutral = model(0)
    m = n.load_metadata
    assert p.load_identity(m) == n.identity != neutral.identity
    assert p.checked_load_metadata(m)["target_ids"].tolist() == [0, 1, 2]
    m["selection"]["edge_ids"][0] = 100
    assert n.load_metadata["selection"]["edge_ids"][0] == 0
    seed(n)
    n.advance(np.full(6, 3, np.float32), steps=3)
    n.reset()
    assert (
        n.tick == 0
        and np.all(n.snapshot()["postsynaptic_load_state"] == 0)
        and n.load_fault_flags == 0
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "h_dtype",
        "h_shape",
        "h_nan",
        "h_negative",
        "h_over",
        "h_zero_tick",
        "old_schema",
        "lambda",
        "threshold",
        "target_order",
        "resource_active",
        "missing",
        "extra",
    ],
)
def test_invalid_restore_atomic_and_no_schema_mixing(mutation):
    n = model()
    before = seed(n)
    bad = copy.deepcopy(before)
    if mutation == "h_dtype":
        bad["postsynaptic_load_state"] = bad["postsynaptic_load_state"].astype(
            np.float32
        )
    elif mutation == "h_shape":
        bad["postsynaptic_load_state"] = np.zeros(2)
    elif mutation in ("h_nan", "h_negative", "h_over"):
        bad["postsynaptic_load_state"][0] = {
            "h_nan": np.nan,
            "h_negative": -1e-10,
            "h_over": 1.000001,
        }[mutation]
    elif mutation == "h_zero_tick":
        bad["tick"] = 0
    elif mutation == "old_schema":
        bad["postsynaptic_load"]["schema"] = p.ssp.SCHEMA
    elif mutation == "lambda":
        bad["postsynaptic_load"]["load_lambda"] = 0.0
    elif mutation == "threshold":
        bad["postsynaptic_load"]["threshold32"][0] += 0.1
    elif mutation == "target_order":
        bad["postsynaptic_load"]["target_ids"] = bad["postsynaptic_load"]["target_ids"][
            ::-1
        ].copy()
    elif mutation == "resource_active":
        bad["recurrent_resource"]["utilization"][0] = 0.05
    elif mutation == "missing":
        del bad["postsynaptic_load_state"]
    else:
        bad["extra"] = 1
    with pytest.raises((ValueError, TypeError)):
        n.restore(bad)
    assert p._equal(n.snapshot(), before)


def test_nonzero_disk_future_diagnostics_reset_and_restore_context(tmp_path):
    n = model()
    state = seed(n)
    n.advance(state["drive"], steps=5)
    saved = n.snapshot()
    assert np.any(saved["postsynaptic_load_state"] > 0)
    diag = n.load_readout()
    assert diag["gross_available"] and diag["gross_tick"] == n.tick - 1
    assert diag["gross_phase"] == "last-SSPRK33-stage-input"
    np.testing.assert_array_equal(
        diag["g"], p.load_factors(saved["postsynaptic_load_state"], 1.0)
    )
    StateStore.save(tmp_path / "cut", saved)
    n.advance(state["drive"], steps=7)
    future = n.snapshot()
    n.restore(StateStore.load(tmp_path / "cut"))
    assert not n.load_readout()["gross_available"]
    n.advance(state["drive"], steps=7)
    assert p._equal(n.snapshot(), future)
    n.reset()
    assert not n.load_readout()["gross_available"]


@pytest.mark.parametrize(
    "name",
    [
        "postsynaptic_load_state",
        "_load_work",
        "load_gross",
        "load_factor",
        "_load_target_ids",
        "_load_row_slots",
        "_load_threshold",
        "_load_tau",
    ],
)
def test_replaced_pointer_rejected_before_step_and_hostcopy(name, monkeypatch):
    n = model()
    setattr(n, name, getattr(n, name).clone())
    tick = n.tick
    monkeypatch.setattr(
        n.torch.Tensor,
        "cpu",
        lambda *a, **kw: pytest.fail("hostcopy after pointer corruption"),
    )
    with pytest.raises(ValueError):
        n.advance(np.zeros(6, np.float32), steps=1)
    with pytest.raises(ValueError):
        n.readout([3])
    assert n.tick == tick


@pytest.mark.parametrize(
    "name,value",
    [
        ("postsynaptic_load_state", -1e-10),
        ("postsynaptic_load_state", 1.00000001),
        ("postsynaptic_load_state", np.nan),
        ("rate", -float(np.nextafter(np.float32(0), np.float32(1)))),
        ("adaptation", -float(np.nextafter(np.float32(0), np.float32(1)))),
    ],
)
def test_sticky_invalid_live_state_not_repaired(name, value):
    n = model()
    good = seed(n)
    getattr(n, name)[0] = value
    with pytest.raises(RuntimeError) as caught:
        n.readout([3])
    assert caught.value.resource_fault_flags & (
        p.FAULT_LOAD if name == "postsynaptic_load_state" else 32
    )
    assert (
        np.isnan(getattr(n, name)[0])
        if np.isnan(value)
        else float(getattr(n, name)[0]) == value
    )
    getattr(n, name)[0] = 0.0
    with pytest.raises(RuntimeError):
        n.advance(good["drive"], steps=1)
    with pytest.raises(RuntimeError):
        n.snapshot()
    n.restore(good)
    assert n.load_fault_flags == 0


def test_positive_subnormal_acceptance_single_control_packet_and_small_q(monkeypatch):
    n = model()
    state = seed(n)
    state["rate"][0] = np.nextafter(np.float32(0), np.float32(1))
    n.restore(state)
    copies = [0]
    old = n.torch.Tensor.cpu

    def host(value, *args, **kwargs):
        copies[0] += 1
        return old(value, *args, **kwargs)

    monkeypatch.setattr(n.torch.Tensor, "cpu", host)
    n.advance_readout(state["drive"], [3, 4], steps=1)
    assert copies[0] == 1
    tiny = np.nextafter(np.float32(0), np.float32(1))
    h = p.load_euler(
        np.zeros(1),
        np.array([tiny]),
        np.array([tiny]),
        np.array([0.03], np.float32),
        DT,
    )
    assert h[0] == DT / float(np.float32(0.03)) * 0.5


@pytest.mark.parametrize("value", [True, 0.5, -1, np.nan, 2])
def test_bad_lambda_rejected_before_actual_constructor(value):
    prior = COST["actual_constructor_requested"]
    with pytest.raises(ValueError):
        model(value)
    assert COST["actual_constructor_requested"] == prior


def test_active_resource_or_invalid_threshold_rejected_before_native():
    args, kw = parameters()
    prior = COST["actual_constructor_requested"]
    with pytest.raises(ValueError):
        p.PostsynapticLoadNetwork(
            *args,
            **kw,
            edge_ids=np.arange(4, dtype=np.int64),
            premotor_ids=np.arange(3, dtype=np.int64),
            utilization=np.full(3, 0.05),
            recovery_tau=np.full(3, 0.15),
            load_lambda=1.0,
        )
    args[3][0] = 0.0
    with pytest.raises(ValueError):
        model(args=args, kw=kw)
    assert COST["actual_constructor_requested"] == prior


def test_base_import_exact_dynamic_state_and_new_metadata():
    n = model()
    initial = n.snapshot()
    base = {
        k: copy.deepcopy(v)
        for k, v in initial.items()
        if k
        not in (
            "recurrent_resource",
            "synaptic_resource",
            "postsynaptic_load",
            "postsynaptic_load_state",
        )
    }
    base["model_hash"] = n.source_timescale_identity
    n.import_base_snapshot(base)
    assert p._equal(n.snapshot(), initial)
    base["tick"] = 1
    with pytest.raises(ValueError):
        n.import_base_snapshot(base)
    assert p._equal(n.snapshot(), initial)


def test_active_legacy_current_and_derivative_hooks_refuse_missing_h():
    n = model(1.0)
    seed(n)
    before = n.snapshot()
    for legacy in (
        lambda: n._current(n.rate),
        lambda: n._derivative_pair(n.rate, n.adaptation),
        n.resource_current_oracle,
    ):
        with pytest.raises(RuntimeError, match="load_current_oracle"):
            legacy()
    current, gross = n.load_current_oracle()
    expected = independent_current(
        n._source_weights,
        before["rate"],
        before["output_mask"],
        np.arange(4),
        np.array([0, 1, 2]),
        before["postsynaptic_load_state"],
        1.0,
    )
    assert current.tobytes() == expected[0].tobytes()
    assert gross.tobytes() == expected[1].tobytes()
    assert p._equal(n.snapshot(), before)


def test_neutral_legacy_hooks_preserve_existing_ssp_meaning():
    n = model(0.0)
    seed(n)
    current = n._current(n.rate)
    derivatives = n._derivative_pair(n.rate, n.adaptation)
    original = n.resource_current_oracle()
    explicit, _ = n.load_current_oracle()
    np.testing.assert_array_equal(current.numpy(), original)
    np.testing.assert_array_equal(explicit, original)
    assert len(derivatives) == 2
    assert all(np.isfinite(value.numpy()).all() for value in derivatives)


def test_load_bounds_and_invalid_euler_no_clipping():
    for oldh in (0.0, 0.2, 1.0):
        for gross in (0.0, 1.0, 100.0):
            nh = p.load_euler(
                np.array([oldh]),
                np.array([gross], np.float32),
                np.array([1.0], np.float32),
                np.array([0.03], np.float32),
                float(np.float32(0.03)) / 2,
            )
            assert 0 <= nh[0] <= 1 and 0.5 <= p.load_factors(nh, 1.0)[0] <= 1
    with pytest.raises(ValueError):
        p.load_euler(
            np.array([1.00001]),
            np.zeros(1, np.float32),
            np.ones(1, np.float32),
            np.ones(1, np.float32),
            DT,
        )
    with pytest.raises(ValueError):
        p.load_euler(
            np.zeros(1),
            np.zeros(1, np.float32),
            np.ones(1, np.float32),
            np.ones(1, np.float32),
            0.500001,
        )


def test_cuda_source_pointwise_conversion_no_fma_original_lambda0_and_coupling():
    source = p.load_cuda_source()
    assert "total+=fabsf(original)" in source and "value+=original*gf" in source
    assert "value+=lookup[item>>18]*masked[item&0x3ffff]" in source
    assert "double fe=(1.0-q)*h[i]+q*L" in source and "recovery_f64_to_f32(g)" in source
    assert "cvt.f64.f32" in source and "recovery_bad_rate(total)" in source


@pytest.mark.skipif(
    not CUDA_ENABLED, reason="Root reservation required RUN_E312_CUDA=1"
)
@pytest.mark.parametrize("strength", [0.0, 1.0])
def test_cuda_wide_packed_direct_graph_disk_future(tmp_path, strength):
    cpu, gpu = wide_model(strength=strength), wide_model("cuda", strength)
    for n in (cpu, gpu):
        state = n.snapshot()
        state.update(
            tick=17,
            rate=np.linspace(0, 30, 39).astype(np.float32),
            drive=np.full(39, 3, np.float32),
        )
        state["postsynaptic_load_state"][:] = [0.3, 0.7]
        n.restore(state)
        n.set_muted([7])
        n.advance(state["drive"], steps=100)
    c, g = cpu.snapshot(), gpu.snapshot()
    for k in ("rate", "adaptation"):
        np.testing.assert_allclose(g[k], c[k], rtol=0, atol=2e-5)
    np.testing.assert_allclose(
        g["postsynaptic_load_state"], c["postsynaptic_load_state"], rtol=0, atol=2e-8
    )
    assert gpu.load_fault_flags == 0 and np.all(g["synaptic_resource"] == 1)
    StateStore.save(tmp_path / "gpu", g)
    gpu.advance(g["drive"], steps=10)
    expected = gpu.snapshot()
    gpu.restore(StateStore.load(tmp_path / "gpu"))
    gpu.graph = None
    gpu.advance(g["drive"], steps=10)
    assert p._equal(gpu.snapshot(), expected)
    if strength == 0:
        args, kw, edges = wide_parameters("cuda")
        baseline = p.ssp.SSPRecoveryResourceNetwork(
            *args,
            **kw,
            edge_ids=edges,
            premotor_ids=np.arange(38, dtype=np.int64),
            utilization=np.zeros(38),
            recovery_tau=np.full(38, 0.15),
        )
        old = baseline.snapshot()
        for key in ("tick", "rate", "adaptation", "drive", "output_mask"):
            old[key] = copy.deepcopy(g[key])
        baseline.restore(old)
        gpu.restore(g)
        baseline.advance(g["drive"], steps=5)
        gpu.advance(g["drive"], steps=5)
        for key in ("rate", "adaptation", "output_mask", "tick"):
            assert p._equal(baseline.snapshot()[key], gpu.snapshot()[key])


@pytest.mark.skipif(
    not CUDA_ENABLED, reason="Root reservation required RUN_E312_CUDA=1"
)
def test_cuda_subnormal_fault_and_load_pointer_rejection():
    n = wide_model("cuda")
    good = n.snapshot()
    n.rate[37] = -float(np.nextafter(np.float32(0), np.float32(1)))
    with pytest.raises(RuntimeError) as caught:
        n.advance(np.zeros(39, np.float32), steps=1)
    assert caught.value.resource_fault_flags & 32
    n.restore(good)
    n._load_work = n._load_work.clone()
    with pytest.raises(ValueError):
        n.advance(np.zeros(39, np.float32), steps=1)
