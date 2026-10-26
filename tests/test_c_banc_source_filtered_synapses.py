"""Generated E267 checks only; CUDA tests require root's separate reservation."""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

from flylab.c.storage import StateStore
from tools.banc_m3_motor_timescale import MotorTimescaleNetwork
from tools.banc_m3_source_filtered_synapses import (
    TAU,
    SourceFilteredSynapseNetwork,
    checked_selection,
    filtered_edge_value,
    history_derivative,
    packed_current_oracle,
    source_filtered_cuda_source,
)


@pytest.fixture(autouse=True)
def generated_cost(request, monkeypatch):
    from flylab.c.adaptive_rate import AdaptiveRateNetwork
    from flylab.c.research_rate import ResearchRateNetwork
    counts = {"test": request.node.name, "models": 0, "logical_steps": 0,
              "logical_neural_s": 0., "cuda_setup_steps": 0,
              "dop853_rhs_calls": 0, "dop853_wall_s": 0.}
    init, advance = AdaptiveRateNetwork.__init__, ResearchRateNetwork._advance

    def counted_init(self, *args, **kwargs):
        counts["models"] += 1
        init(self, *args, **kwargs)
        if self.device == "cuda":
            counts["cuda_setup_steps"] += 3*self.interval_steps

    def counted_advance(self, drive, steps, **kwargs):
        before = self.tick
        try:
            return advance(self, drive, steps, **kwargs)
        finally:
            counts["logical_steps"] += self.tick-before
            counts["logical_neural_s"] += (self.tick-before)*self.dt

    monkeypatch.setattr(AdaptiveRateNetwork, "__init__", counted_init)
    monkeypatch.setattr(ResearchRateNetwork, "_advance", counted_advance)
    started = time.perf_counter()
    yield counts
    counts["wall_s"] = time.perf_counter()-started
    print("E267_GENERATED_COST " + json.dumps(counts, sort_keys=True))


def graph():
    w = np.zeros((6, 6), np.float32)
    w[0, 2], w[1, 0], w[2, 1] = .07, -.08, .12
    w[3, 0], w[3, 1], w[4, 0], w[4, 2] = .15, -.1, .09, -.04
    w[5, 3], w[5, 4] = -.02, .13
    return csr_matrix(w)


def parameters(device):
    args = (graph(), np.full(6, .03, np.float32), np.ones(6, np.float32),
            np.full(6, 1.5, np.float32), np.full(6, 40., np.float32))
    kwargs = {"adaptation_gain": 8., "adaptation_tau_s": .3, "dt": .0001,
        "device": device, "capture_steps": 5, "cuda_implementation": "packed",
        "core_ids": np.array([0, 1], np.int64), "core_gain": 0.,
        "motor_ids": np.array([3, 4], np.int64), "motor_gains": np.array([1., 2.], np.float32),
        "motor_log_tau": np.array([0., np.log(2.)], np.float64)}
    edges = np.array([3, 4, 5, 6], np.int64)
    return args, kwargs, edges


def model(device="cpu", coefficients=None):
    args, kwargs, edges = parameters(device)
    c = np.tile(np.array([1., 0., 0.]), (4, 1)) if coefficients is None else coefficients
    return SourceFilteredSynapseNetwork(*args, **kwargs, edge_ids=edges,
        coefficients=c, free_motor_ids=np.array([3, 4], np.int64))


def base(device="cpu"):
    args, kwargs, _ = parameters(device)
    return MotorTimescaleNetwork(*args, **kwargs)


def initialize(network):
    saved = network.snapshot()
    saved["rate"] = np.array([4., 3., 2., 5., 1., 6.], np.float32)
    saved["adaptation"] = np.array([.1, .2, .15, .3, .2, .1], np.float32)
    saved["drive"] = np.array([2.8, 3., 2.5, 3.1, 3., 2.7], np.float32)
    saved["tick"] = 17
    if "source_history" in saved:
        saved["source_history"] = np.array([[.5, 1.], [1.5, 2.], [2.5, 3.]], np.float64)
    network.restore(saved)
    return saved


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    else:
        assert a == b


def test_cpu_selection_strict_topology_bytes_and_owned_parameters():
    args, kw, edges = parameters("cpu")
    c = np.tile(np.array([1., 0., 0.]), (4, 1))
    free = np.array([3, 4], np.int64)
    selected = checked_selection(args[0], edges, c, free)
    same(selected["pre_ids"], np.array([0, 1, 0, 2], np.int64))
    same(selected["source_ids"], np.array([0, 1, 2], np.int64))
    for bad_edges, bad_c, bad_free in (
        (edges[::-1], c, free), (np.array([3, 3, 5, 6], np.int64), c, free),
        (edges.astype(np.int32), c, free), (edges, c.astype(np.float32), free),
        (edges, c*9, free), (edges, c*.1, free), (edges, c*np.nan, free),
        (edges, c, free[::-1]), (edges, c, np.array([3], np.int64)),
        (np.array([0], np.int64), c[:1], free)):
        with pytest.raises(ValueError):
            checked_selection(args[0], bad_edges, bad_c, bad_free)
    bad = args[0].copy()
    bad.data[3] = 0
    with pytest.raises(ValueError):
        checked_selection(bad, edges, c, free)
    # Casting can cross the lower bound even though original float64 sum passes.
    crossing = np.tile(np.array([.125, 0., 0.]), (4, 1))
    crossing[0] = [.124999996, 4.000000001e-9, 0.]
    with pytest.raises(ValueError):
        checked_selection(args[0], edges, crossing, free)
    network = SourceFilteredSynapseNetwork(*args, **kw, edge_ids=edges,
        coefficients=c, free_motor_ids=free)
    before = network.snapshot()
    c[:] = 8
    edges[:] = 0
    free[:] = 0
    args[0].data[:] = 0
    args[1][:] = 2
    external = network.selection
    external["coefficients32"][:] = 0
    same(network.snapshot(), before)


def test_cpu_scalar_source_separation_shared_source_sign_and_mask():
    w = graph()
    c = np.array([[0., 1., 0.], [0., 0., 1.], [0., 0., 2.], [1., 0., 0.]])
    sel = checked_selection(w, np.array([3, 4, 5, 6], np.int64), c,
                            np.array([3, 4], np.int64))
    h = np.array([[2., 7.], [3., 11.], [5., 13.]])
    r, mask = np.arange(1, 7, dtype=np.float32), np.ones(6, np.float32)
    out = packed_current_oracle(w, r, mask, sel, h)
    assert out[3] == np.float32(np.float32(.15)*2+np.float32(-.1)*11)
    assert out[4] == np.float32(np.float32(.09)*14+np.float32(-.04)*3)
    mask[0] = 0
    muted = packed_current_oracle(w, r, mask, sel, h)
    assert muted[3] == np.float32(-.1)*11
    assert muted[4] == np.float32(-.04)*3
    # Muting does not enter the source history differential.
    same(history_derivative(r, h, sel["source_ids"]), (r[:3, None]-h)/TAU)
    assert filtered_edge_value(-.5, 2., 0., [90., 80.], [0., 1., 1.]) == 0
    tiny = np.nextafter(np.float32(0), np.float32(1))
    assert filtered_edge_value(1., tiny, 1., [0., 0.], [1., 0., 0.]) == 0


def test_cpu_neutral_bit_exact_import_and_same_stage_history():
    old, new = base(), model()
    saved = initialize(old)
    new.import_base_snapshot(saved)
    same(new.snapshot()["source_history"], np.repeat(saved["rate"][:3, None].astype(
        np.float64), 2, axis=1))
    # Independent reconstruction of source RK4 stages using original derivatives.
    r, z = old.rate.clone(), old.adaptation.clone()
    h = new.history.numpy().copy()
    stage_rates, derivatives = [], []
    rr, zz, hh = r, z, h
    for stage in range(4):
        stage_rates.append(rr.numpy().copy())
        dr, dz = old._derivative_pair(rr, zz)
        dh = (rr.numpy()[:3, None].astype(np.float64)-hh)/TAU
        derivatives.append(dh)
        if stage < 3:
            step = old.dt*(.5 if stage < 2 else 1)
            rr, zz, hh = r+step*dr, z+step*dz, h+step*dh
    expected = h+old.dt/6*(derivatives[0]+2*derivatives[1]+2*derivatives[2]+derivatives[3])
    assert not np.array_equal(stage_rates[0], stage_rates[1])
    drive = saved["drive"]
    old.advance(drive, steps=1)
    new.advance(drive, steps=1)
    same(new.snapshot()["source_history"], expected)
    for control in range(3):
        for network in (old, new):
            network.set_muted([0, 3] if control == 1 else [])
            network.advance(drive+np.float32(.1*control), steps=5)
        for key in ("rate", "adaptation", "drive", "output_mask", "tick"):
            same(old.snapshot()[key], new.snapshot()[key])
    with pytest.raises(ValueError):
        new.restore(saved)


def test_cpu_coupled_non_neutral_dop853_and_future_disk_reset(tmp_path, generated_cost):
    c = np.array([[.25, .5, .25], [0., .5, .5], [0., 1., 1.], [.5, .25, .25]])
    network = model(coefficients=c)
    saved = initialize(network)
    network.set_muted([0, 3])
    mask = network.output_mask.numpy().copy()
    w = graph().toarray().astype(np.float64)
    gain = network._cell_gain.numpy().astype(np.float64)
    tau = network._cell_adaptation_tau.numpy().astype(np.float64)
    sel = network.selection

    def rhs(_, value):
        generated_cost["dop853_rhs_calls"] += 1
        r, z, h = value[:6], value[6:12], value[12:].reshape(3, 2)
        current = w@(r*mask)
        for i, (pre, post) in enumerate(zip(sel["pre_ids"], sel["post_ids"])):
            mix = c[i, 0]*r[pre] + c[i, 1]*h[pre, 0] + c[i, 2]*h[pre, 1]
            current[post] += w[post, pre]*mask[pre]*(mix-r[pre])
        target = np.maximum(40*np.tanh((saved["drive"]+current-1.5-z)/40), 0)
        return np.r_[(target-r)/.03, (gain*r-z)/tau,
                      ((r[:3, None]-h)/TAU).ravel()]

    began = time.perf_counter()
    result = solve_ivp(rhs, (0., .002), np.r_[saved["rate"], saved["adaptation"],
        saved["source_history"].ravel()], method="DOP853", rtol=1e-11, atol=1e-12)
    generated_cost["dop853_wall_s"] = time.perf_counter()-began
    assert result.success
    network.advance(saved["drive"], steps=20)
    actual = network.snapshot()
    np.testing.assert_allclose(actual["rate"], result.y[:6, -1], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual["adaptation"], result.y[6:12, -1], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual["source_history"], result.y[12:, -1].reshape(3, 2),
                               rtol=2e-6, atol=2e-6)
    StateStore.save(tmp_path/"generated", actual)
    clone = model(coefficients=c)
    clone.restore(StateStore.load(tmp_path/"generated"))
    for candidate in (network, clone):
        candidate.advance(saved["drive"]+.2, steps=10)
    same(network.snapshot(), clone.snapshot())
    pointers = (network.history.data_ptr(), network._filter_coefficients.data_ptr())
    network.reset()
    assert not network.history.any() and network.tick == 0
    assert pointers == (network.history.data_ptr(), network._filter_coefficients.data_ptr())
    with pytest.raises(ValueError):
        network.import_base_snapshot(base().snapshot())


@pytest.mark.parametrize("key", ["rate", "adaptation", "drive", "output_mask",
    "tick", "source_history", "source_filter", "motor_timescale", "motor_adaptation",
    "core_adaptation", "adaptation_gain", "schema", "model_hash", "device", "dt"])
def test_cpu_all_checkpoint_fields_are_prevalidated_atomically(key):
    network = model()
    initialize(network)
    before = network.snapshot()
    bad = copy.deepcopy(before)
    # Set earlier valid fields to a different value; late failures must not copy them.
    bad["rate"][:] += 1
    bad["drive"][:] += 1
    if isinstance(bad[key], np.ndarray):
        bad[key].flat[-1] = np.nan
    elif isinstance(bad[key], dict):
        bad[key]["schema"] = "wrong"
    else:
        bad[key] = "wrong"
    with pytest.raises(ValueError):
        network.restore(bad)
    same(network.snapshot(), before)


def test_cpu_generated_cuda_source_retains_original_neutral_expression():
    source = source_filtered_cuda_source()
    assert source.count("value += lookup[item >> 18] * masked[item & 0x3ffff];") == 1
    assert "history_stage" in source and "history_finish" in source
    assert "mul_exact(mixture, mask[pre])" in source


def test_cpu_unsupported_derivatives_reject_source_filter_bypass():
    network = model(coefficients=np.tile(np.array([0., 1., 0.]), (4, 1)))
    initialize(network)
    before = network.snapshot()
    for call in (lambda: network._current(network.rate),
                 lambda: network._derivative_pair(network.rate, network.adaptation),
                 lambda: network._derivative(network.rate)):
        with pytest.raises(RuntimeError):
            call()
    same(network.source_current_oracle(), packed_current_oracle(graph(),
        before["rate"], before["output_mask"], network.selection, before["source_history"]))
    same(network.snapshot(), before)


def test_cpu_free_roster_rejects_existing_non_motor_target_before_construction(generated_cost):
    args, kwargs, _ = parameters("cpu")
    # Row zero is an existing nonzero target but is not in actual motor_ids.
    with pytest.raises(ValueError, match="actual native motor roster"):
        SourceFilteredSynapseNetwork(*args, **kwargs,
            edge_ids=np.array([0], np.int64), coefficients=np.array([[1., 0., 0.]]),
            free_motor_ids=np.array([0], np.int64))
    assert generated_cost["models"] == 0


def test_cpu_history_allocation_inside_inference_mode_remains_normally_mutable(monkeypatch):
    import torch
    original = SourceFilteredSynapseNetwork._init_history

    def warmup_like_init(self):
        with torch.inference_mode():
            return original(self)

    monkeypatch.setattr(SourceFilteredSynapseNetwork, "_init_history", warmup_like_init)
    network = model()
    tensors = [network.history, *network._history_buffers, network._edge_slot,
        network._source_slot, network._filter_coefficients, network._source_ids]
    assert all(not tensor.is_inference() for tensor in tensors)
    pointers = [tensor.data_ptr() for tensor in tensors]
    saved = initialize(network)
    network.reset()
    assert not network.history.any()
    network.restore(saved)
    same(network.snapshot(), saved)
    assert pointers == [tensor.data_ptr() for tensor in tensors]


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("Reserved generated CUDA test requires actual GPU")


def test_cuda_generated_neutral_native_capture_import_future_and_reset(tmp_path):
    require_cuda()
    old, new = base("cuda"), model("cuda")
    saved = initialize(old)
    new.import_base_snapshot(saved)
    assert new.graph is not None and new.cuda_implementation == "packed"
    ptrs = (new.history.data_ptr(), new._source_ids.data_ptr())
    for control in range(3):
        for network in (old, new):
            network.set_muted([0, 3] if control == 1 else [])
            network.advance(saved["drive"]+.1*control, steps=5)
        for key in ("rate", "adaptation", "drive", "output_mask", "tick"):
            same(old.snapshot()[key], new.snapshot()[key])
    StateStore.save(tmp_path/"cuda-generated", new.snapshot())
    clone = model("cuda")
    clone.restore(StateStore.load(tmp_path/"cuda-generated"))
    for network in (new, clone):
        network.advance(saved["drive"]+.3, steps=7)
    same(new.snapshot(), clone.snapshot())
    before = new.snapshot()
    bad = copy.deepcopy(before)
    bad["drive"][:] += 2
    bad["source_history"][0, 0] = np.nan
    with pytest.raises(ValueError):
        new.restore(bad)
    same(new.snapshot(), before)
    new.reset()
    assert ptrs == (new.history.data_ptr(), new._source_ids.data_ptr())
    assert not new.history.any()


def test_cuda_generated_non_neutral_coupled_host_oracle_mask_and_future():
    require_cuda()
    c = np.array([[.25, .5, .25], [0., .5, .5], [0., 1., 1.], [.5, .25, .25]])
    cpu, cuda = model(coefficients=c), model("cuda", coefficients=c)
    saved = initialize(cpu)
    initialize(cuda)
    for control in range(3):
        for network in (cpu, cuda):
            network.set_muted([0, 3] if control == 1 else [])
            network.advance(saved["drive"]+.1*control, steps=5)
        for key in ("rate", "adaptation", "source_history"):
            np.testing.assert_allclose(cpu.snapshot()[key], cuda.snapshot()[key],
                                       rtol=2e-6, atol=2e-6)
    clone = model("cuda", coefficients=c)
    clone.restore(cuda.snapshot())
    for network in (cuda, clone):
        network.advance(saved["drive"]+.3, steps=10)
    same(cuda.snapshot(), clone.snapshot())
