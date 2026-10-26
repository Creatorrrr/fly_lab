"""E289 generated CPU only by default; root explicitly opts into tiny CUDA."""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from flylab.c.adaptive_rate import AdaptiveRateNetwork
from flylab.c.research_rate import ResearchRateNetwork
from flylab.c.storage import StateStore
from tools import banc_m3_recurrent_resource_synapses as rr
from tools.banc_m3_motor_timescale import MotorTimescaleNetwork

CUDA_ENABLED = os.environ.get("RUN_E289_CUDA") == "1"
CUDA_RATE_ATOL, CUDA_RESOURCE_ATOL, CUDA_TICKS, CUDA_DT = 2e-5, 2e-8, 100, .00025
COST = {"constructor_requested": 0, "constructor_completed": 0, "constructor_failed": 0,
        "constructor_wall_s": 0., "actual_neural_constructors": 0, "step_requested": 0,
        "step_completed": 0, "step_failed": 0, "step_wall_s": 0., "setup_steps": 0,
        "advance_requested": 0, "advance_completed": 0, "advance_failed": 0,
        "logical_ticks": 0, "solver_calls": 0, "body_calls": 0, "original_calls": 0}


@pytest.fixture(scope="module", autouse=True)
def aggregate_cost():
    started, process = time.perf_counter(), time.process_time()
    yield
    COST.update(wall_s=time.perf_counter()-started, process_s=time.process_time()-process)
    print("E289_GENERATED_COST "+json.dumps(COST, sort_keys=True))
    assert COST["wall_s"] <= 120


@pytest.fixture(autouse=True)
def instrument(monkeypatch):
    original_resource_init = rr.RecurrentResourceNetwork.__init__
    original_base_init = MotorTimescaleNetwork.__init__
    adaptive_init = AdaptiveRateNetwork.__init__
    resource_step, base_step = rr.RecurrentResourceNetwork._step, AdaptiveRateNetwork._step
    advance = ResearchRateNetwork._advance
    def init_cost(method, self, args, kw):
        assert args[0].shape[0] <= 64 and args[0].nnz <= 256
        COST["constructor_requested"] += 1; began = time.perf_counter(); self._e289_setup = True
        try: method(self, *args, **kw); COST["constructor_completed"] += 1
        except Exception: COST["constructor_failed"] += 1; raise
        finally: COST["constructor_wall_s"] += time.perf_counter()-began; self._e289_setup = False
    def resource_init(self, *args, **kw): return init_cost(original_resource_init, self, args, kw)
    def base_init(self, *args, **kw):
        if type(self) is MotorTimescaleNetwork: return init_cost(original_base_init, self, args, kw)
        return original_base_init(self, *args, **kw)
    def actual_init(self, *args, **kw):
        COST["actual_neural_constructors"] += 1
        return adaptive_init(self, *args, **kw)
    def step_cost(method, self):
        COST["step_requested"] += 1; began = time.perf_counter()
        if getattr(self, "_e289_setup", False): COST["setup_steps"] += 1
        try: method(self); COST["step_completed"] += 1
        except Exception: COST["step_failed"] += 1; raise
        finally: COST["step_wall_s"] += time.perf_counter()-began
    def resource_step_cost(self): return step_cost(resource_step, self)
    def base_step_cost(self):
        if type(self) is MotorTimescaleNetwork: return step_cost(base_step, self)
        return base_step(self)
    def advance_cost(self, *args, **kw):
        COST["advance_requested"] += 1; tick = self.tick
        try: result = advance(self, *args, **kw); COST["advance_completed"] += 1; return result
        except Exception: COST["advance_failed"] += 1; raise
        finally: COST["logical_ticks"] += self.tick-tick
    monkeypatch.setattr(rr.RecurrentResourceNetwork, "__init__", resource_init)
    monkeypatch.setattr(MotorTimescaleNetwork, "__init__", base_init)
    monkeypatch.setattr(AdaptiveRateNetwork, "__init__", actual_init)
    monkeypatch.setattr(rr.RecurrentResourceNetwork, "_step", resource_step_cost)
    monkeypatch.setattr(AdaptiveRateNetwork, "_step", base_step_cost)
    monkeypatch.setattr(ResearchRateNetwork, "_advance", advance_cost)


def parameters(device="cpu"):
    w = np.zeros((6, 6), np.float32)
    w[0, 2], w[1, 0], w[2, 0], w[2, 1] = .07, -.08, .04, .12
    w[3, 0], w[3, 1], w[4, 0], w[4, 2], w[5, 3] = .15, -.1, .09, -.04, -.02
    args = (csr_matrix(w), np.full(6, .03, np.float32), np.ones(6, np.float32),
            np.full(6, 1.5, np.float32), np.full(6, 40., np.float32))
    kw = {"adaptation_gain": 8., "adaptation_tau_s": .3, "dt": CUDA_DT, "device": device,
          "capture_steps": 5, "cuda_implementation": "packed", "core_ids": np.array([0, 1], np.int64),
          "core_gain": 0., "motor_ids": np.array([3, 4], np.int64), "motor_gains": np.array([1., 2.], np.float32),
          "motor_log_tau": np.array([0., np.log(2.)], np.float64)}
    return args, kw


def model(device="cpu", u=None):
    args, kw = parameters(device)
    return rr.RecurrentResourceNetwork(*args, **kw, edge_ids=np.arange(4, dtype=np.int64),
        premotor_ids=np.arange(3, dtype=np.int64), utilization=np.array([.05, .1, .2]) if u is None else u,
        recovery_tau=np.array([.15, .12, .2]))


def same(a, b):
    assert rr._equal(a, b)


def seeded(network):
    s = network.snapshot(); s.update(tick=17, rate=np.array([4., 3., 2., 5., 1., 6.], np.float32),
        adaptation=np.array([.1, .2, .15, .3, .2, .1], np.float32),
        drive=np.array([2.8, 3., 2.5, 3.1, 3., 2.7], np.float32))
    s["synaptic_resource"] = np.array([.8, .7, .9]); s["synaptic_resource"][network.selection["utilization"] == 0] = 1.
    network.restore(s)
    return s


def independent_current(w, rate, mask, selection, resource):
    # Literal scalar CUDA warp accumulation, without calling the product oracle.
    def ftz(v):
        v = np.float32(v)
        return np.copysign(np.float32(0), v) if abs(v) < np.finfo(np.float32).tiny else v
    chosen = dict(zip(selection["edge_ids"], selection["edge_source_slots"], strict=True))
    result = np.zeros(w.shape[0], np.float32)
    for row in range(len(result)):
        lane = np.zeros(32, np.float32)
        for j in range(32):
            for edge in range(w.indptr[row]+j, w.indptr[row+1], 32):
                pre = w.indices[edge]; value = np.float32(rate[pre]*mask[pre])
                if edge in chosen and selection["utilization"][chosen[edge]] != 0:
                    value = ftz(ftz(value)*ftz(np.float32(resource[chosen[edge]])))
                value = ftz(ftz(w.data[edge])*ftz(value))
                lane[j] = ftz(ftz(lane[j])+ftz(value))
        for offset in (16, 8, 4, 2, 1):
            old = lane.copy()
            for j in range(32-offset): lane[j] = ftz(ftz(old[j])+ftz(old[j+offset]))
        result[row] = lane[0]
    return result


def independent_tick(network, state):
    r0, z0, R0 = (state[k].copy() for k in ("rate", "adaptation", "synaptic_resource"))
    r, z, R = r0, z0, R0; kr, kz, kR, stage_rates = [], [], [], []
    args, _ = parameters(); w, tau, a, threshold, cap = args; c = a/cap
    gain = np.array([0., 0., 8., 1., 2., 8.], np.float32)
    cell_tau = np.array([.3, .3, .3, .3, .6, .3], np.float32)
    s = network.selection
    for stage in range(4):
        stage_rates.append(r.copy())
        current = independent_current(w, r, state["output_mask"], s, R)
        target = np.maximum(cap*np.tanh(c*(state["drive"]+current-threshold-z)), np.float32(0))
        dr, dz = (target-r)/tau, (gain*r-z)/cell_tau
        dR = (1.-R)/s["recovery_tau"]-(s["utilization"]*r[s["source_ids"]].astype(np.float64))*R
        kr.append(dr); kz.append(dz); kR.append(dR)
        if stage < 3:
            step = network.dt*(.5 if stage < 2 else 1)
            r, z, R = r0+np.float32(step)*dr, z0+np.float32(step)*dz, R0+step*dR
    result = copy.deepcopy(state)
    for key, base, values in (("rate", r0, kr), ("adaptation", z0, kz), ("synaptic_resource", R0, kR)):
        step = network.dt/6 if key == "synaptic_resource" else np.float32(network.dt/6)
        result[key] = base+step*(values[0]+2*values[1]+2*values[2]+values[3])
    result["tick"] += 1
    return result, stage_rates


def test_selection_owned_shared_source_and_identity():
    args, kw = parameters(); edges = np.arange(4, dtype=np.int64); pop = np.arange(3, dtype=np.int64)
    u, tau = np.array([.05, .1, .2]), np.array([.15, .12, .2])
    selected = rr.checked_selection(args[0], edges, pop, u, tau, kw["motor_ids"])
    same(selected["source_ids"], np.arange(3, dtype=np.int64))
    same(selected["edge_source_slots"], np.array([2, 0, 0, 1], np.int32))
    network = rr.RecurrentResourceNetwork(*args, **kw, edge_ids=edges, premotor_ids=pop, utilization=u, recovery_tau=tau)
    before = network.snapshot(); metadata = before["recurrent_resource"]
    same(rr.checked_resource_metadata(metadata), metadata)
    assert rr.resource_identity(metadata) == network.identity
    assert metadata["source_model_hash"] == network.source_timescale_identity
    args[0].data[:] = 99; args[1][:] = 1; edges[:] = 0; pop[:] = 0; u[:] = .5; tau[:] = .5
    exposed = network.selection; exposed["utilization"][:] = 0
    same(network.snapshot(), before)


@pytest.mark.parametrize("device",["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not CUDA_ENABLED, reason="Root reservation required: RUN_E289_CUDA=1"))])
def test_unselected_explicit_zeros_preserve_positions_bytes_identity_and_neutral(device):
    args, kw = parameters(device); w = args[0]
    # Keep existing downstream CSR positions, including both signs of zero.
    w.data[4] = np.float32(0.); w.data[8] = np.float32(-0.)
    original = tuple(v.copy() for v in (w.data, w.indices, w.indptr))
    edges, pop = np.arange(4, dtype=np.int64), np.arange(3, dtype=np.int64)
    u, tau = np.zeros(3), np.full(3, .15)
    selection = rr.checked_selection(w, edges, pop, u, tau, kw["motor_ids"])
    assert w.nnz == 9 and np.count_nonzero(w.data == 0) == 2
    same(selection["edge_ids"], edges)
    same(selection["weights32"], original[0][:4])
    selected_zero = w.copy(); selected_zero.data[0] = np.float32(-0.)
    with pytest.raises(ValueError): rr.checked_selection(selected_zero, edges, pop, u, tau, kw["motor_ids"])
    base = MotorTimescaleNetwork(*args, **kw)
    network = rr.RecurrentResourceNetwork(*args, **kw, edge_ids=edges, premotor_ids=pop,
        utilization=u, recovery_tau=tau)
    assert network.source_timescale_identity == base.identity
    for actual, expected in zip((w.data, w.indices, w.indptr), original, strict=True): same(actual, expected)
    for actual, expected in zip((network._source_weights.data, network._source_weights.indices,
        network._source_weights.indptr), original, strict=True): same(actual, expected)
    same(network.weights.values().cpu().numpy(), original[0])
    network.import_base_snapshot(base.snapshot())
    for control in range(3):
        for n in (base, network):
            n.set_muted([0] if control == 1 else [])
            n.advance(np.full(6, 3.+.1*control, np.float32), steps=10)
        for key in ("rate", "adaptation", "drive", "output_mask", "tick"):
            same(base.snapshot()[key], network.snapshot()[key])
        assert np.all(network.resource.cpu().numpy() == 1)
    for actual, expected in zip((w.data, w.indices, w.indptr), original, strict=True): same(actual, expected)


@pytest.mark.parametrize("kind",["edge_unsorted", "edge_duplicate", "edge_dtype", "population", "motor", "u_dtype", "u_bound", "tau_bound", "shape", "zero", "weight_dtype", "csr_duplicate", "stale_sorted", "trailing_csr", "index_dtype", "nan"])
def test_bad_selection_no_canonicalization(kind):
    args, kw = parameters(); w = args[0].copy(); edge = np.arange(4, dtype=np.int64); pop = np.arange(3, dtype=np.int64)
    u, tau, motor = np.array([.05, .1, .2]), np.array([.15, .12, .2]), kw["motor_ids"]
    if kind == "edge_unsorted": edge = edge[::-1]
    if kind == "edge_duplicate": edge[1] = 0
    if kind == "edge_dtype": edge = edge.astype(np.int32)
    if kind == "population": pop = np.array([0, 1], np.int64)
    if kind == "motor": motor = np.array([0, 3], np.int64)
    if kind == "u_dtype": u = u.astype(np.float32)
    if kind == "u_bound": u[0] = .5000001
    if kind == "tau_bound": tau[0] = .01999999
    if kind == "shape": u = u[:2]
    if kind == "zero": w.data[0] = 0
    if kind == "weight_dtype": w = w.astype(np.float64)
    if kind == "csr_duplicate": w.indices[3] = w.indices[2]; w.has_canonical_format = False
    if kind == "stale_sorted":
        w.has_canonical_format = True; w.has_sorted_indices = True; w.indices[3] = w.indices[2]
    if kind == "trailing_csr": w.indptr[-1] -= 1
    if kind == "index_dtype": w.indices = w.indices.astype(np.float32)
    if kind == "nan": w.data[0] = np.nan
    before = (w.data.copy(), w.indices.copy(), w.indptr.copy())
    with pytest.raises(ValueError): rr.checked_selection(w, edge, pop, u, tau, motor)
    for actual, expected in zip((w.data, w.indices, w.indptr), before, strict=True):
        assert actual.tobytes() == expected.tobytes()


def test_derivative_equilibrium_recovery_mask_and_mixed_neutral():
    r = np.array([20., 10., 4.], np.float32); u = np.array([.05, .1, 0.]); tau = np.array([.15, .12, .2]); sources = np.arange(3, dtype=np.int64)
    equilibrium = 1/(1+tau*u*r.astype(np.float64))
    np.testing.assert_allclose(rr.resource_derivative(r, equilibrium, sources, u, tau), 0, atol=1e-15, rtol=0)
    recovered = equilibrium.copy(); dt = .00025
    for _ in range(100):
        def f(R): return rr.resource_derivative(np.zeros(3, np.float32), R, sources, u, tau)
        k1 = f(recovered); k2 = f(recovered+dt/2*k1); k3 = f(recovered+dt/2*k2); k4 = f(recovered+dt*k3)
        recovered += dt/6*(k1+2*k2+2*k3+k4)
    np.testing.assert_allclose(recovered, 1-(1-equilibrium)*np.exp(-100*dt/tau), atol=2e-14, rtol=0)
    assert recovered[2] == 1
    network = model(u=np.array([0., .1, .2])); s = seeded(network)
    network.set_muted([1]); s = network.snapshot()
    np.testing.assert_array_equal(network.resource_current_oracle(), independent_current(parameters()[0][0], s["rate"], s["output_mask"], network.selection, s["synaptic_resource"]))
    network.advance(s["drive"], steps=1)
    assert network.resource.numpy()[0] == 1 and network.resource.numpy()[1] != s["synaptic_resource"][1]
    assert rr.resource_edge_value(-.5, 20., 0., .2, .1) == 0
    assert rr.resource_edge_value(-.5, 20., 1., .2, 0.) == -10


def test_current_sign_selection_shared_not_other_outgoing_edges():
    n = model(); s = seeded(n); current = n.resource_current_oracle()
    expected = independent_current(parameters()[0][0], s["rate"], s["output_mask"], n.selection, s["synaptic_resource"])
    same(current, expected)
    # Source0 resource is shared only on two chosen recurrent edges, never MN3/4.
    changed = s["synaptic_resource"].copy(); changed[0] = .2
    alternate = n.resource_current_oracle(resource=changed)
    assert alternate[1] != current[1] and alternate[2] != current[2]
    same(alternate[3:], current[3:])
    tiny = np.nextafter(np.float32(0), np.float32(1))
    assert rr.resource_edge_value(1., tiny, 1., 1., .1) == 0


def test_joint_same_stage_independent_rk4_and_nonzero_future_disk(tmp_path):
    n = model(); s = seeded(n)
    expected, stages = independent_tick(n, s)
    assert not np.array_equal(stages[0], stages[1])
    n.advance(s["drive"], steps=1)
    actual = n.snapshot()
    for key in ("rate", "adaptation"): np.testing.assert_allclose(actual[key], expected[key], atol=2e-6, rtol=0)
    np.testing.assert_allclose(actual["synaptic_resource"], expected["synaptic_resource"], atol=2e-15, rtol=0)
    pointers = {key: getattr(n, key).data_ptr() for key in n._parameter_pointers}
    StateStore.save(tmp_path/"state", actual); saved = StateStore.load(tmp_path/"state")
    restored = model(); restored.restore(saved); same(restored.snapshot(), actual)
    for network in (n, restored): network.advance(s["drive"], steps=25)
    same(n.snapshot(), restored.snapshot())
    assert all(getattr(n, key).data_ptr() == value for key, value in pointers.items())
    n.reset(); assert n.tick == 0 and np.all(n.resource.numpy() == 1)
    same(n.snapshot()["recurrent_resource"], actual["recurrent_resource"])


def test_neutral_bit_exact_base_and_neutral_import_only():
    args, kw = parameters(); base = MotorTimescaleNetwork(*args, **kw); n = model(u=np.zeros(3))
    n.import_base_snapshot(base.snapshot())
    for control in range(4):
        drive = np.full(6, 3.+.1*control, np.float32)
        for network in (base, n): network.set_muted([0, 3] if control == 2 else []); network.advance(drive, steps=10)
        for key in ("rate", "adaptation", "drive", "output_mask", "tick"): same(base.snapshot()[key], n.snapshot()[key])
        assert np.all(n.resource.numpy() == 1)
    with pytest.raises(ValueError): n.import_base_snapshot(base.snapshot())
    base.reset(); nonneutral = model(); nonneutral.import_base_snapshot(base.snapshot())
    before = nonneutral.snapshot()
    for key in ("tick", "rate", "adaptation", "drive", "output_mask", "model_hash"):
        bad = base.snapshot()
        if key == "tick": bad[key] = 1
        elif key == "model_hash": bad[key] = "0"*64
        else: bad[key][0] = 0 if key == "output_mask" else 1
        with pytest.raises(ValueError): nonneutral.import_base_snapshot(bad)
        same(nonneutral.snapshot(), before)
    with pytest.raises(ValueError): nonneutral.restore(base.snapshot())


@pytest.mark.parametrize("kind",["resource_dtype", "resource_shape", "nan", "range", "u0", "tick0", "rate", "adaptation", "mask", "identity", "metadata", "precision", "source", "extra"])
def test_invalid_restore_is_atomic(kind):
    n = model(u=np.array([0., .1, .2])); before = seeded(n); bad = copy.deepcopy(before)
    if kind == "resource_dtype": bad["synaptic_resource"] = bad["synaptic_resource"].astype(np.float32)
    if kind == "resource_shape": bad["synaptic_resource"] = bad["synaptic_resource"][:2]
    if kind == "nan": bad["synaptic_resource"][1] = np.nan
    if kind == "range": bad["synaptic_resource"][1] = 1.000001
    if kind == "u0": bad["synaptic_resource"][0] = .9
    if kind == "tick0": bad["tick"] = 0
    if kind == "rate": bad["rate"][1] = -1
    if kind == "adaptation": bad["adaptation"][1] = -1
    if kind == "mask": bad["output_mask"][1] = .5
    if kind == "identity": bad["model_hash"] = "0"*64
    if kind == "metadata": bad["recurrent_resource"]["weights32"][0] *= -1
    if kind == "precision": bad["recurrent_resource"]["precision"] += "wrong"
    if kind == "source": bad["recurrent_resource"]["source_model_hash"] = "0"*64
    if kind == "extra": bad["extra"] = 1
    with pytest.raises(ValueError): n.restore(bad)
    same(n.snapshot(), before)


def test_restore_stages_all_arrays_before_mutation(monkeypatch):
    n = model(); before = seeded(n); bad = copy.deepcopy(before); bad["rate"] += 1
    original = n.torch.as_tensor; calls = [0]
    def fail(*args, **kw):
        calls[0] += 1
        if calls[0] == 5: raise MemoryError("last staging allocation")
        return original(*args, **kw)
    monkeypatch.setattr(n.torch, "as_tensor", fail)
    with pytest.raises(MemoryError): n.restore(bad)
    same(n.snapshot(), before)


@pytest.mark.parametrize("key",["resource", "_resource_u", "_resource_tau", "_resource_edge_source"])
def test_captured_pointer_or_parameter_tamper_rejected(key):
    n = model()
    setattr(n, key, getattr(n, key).clone())
    with pytest.raises(ValueError): n.snapshot()


def test_precision_kernel_stage_order_and_resolving_bounds():
    source = rr.resource_cuda_source()
    assert "u[s] == 0.0" in source and "masked[item & 0x3ffff] * (float)resource[s]" in source
    assert "const double*" in source and "cell_inverse_tau[i]" in source
    args, kw = parameters()
    for dt, cap in ((.0011, 40.), (.00025, 1000.)):
        kw["dt"] = dt; args = (*args[:4], np.full(6, cap, np.float32))
        with pytest.raises(ValueError): rr.RecurrentResourceNetwork(*args, **kw, edge_ids=np.arange(4, dtype=np.int64),
            premotor_ids=np.arange(3, dtype=np.int64), utilization=np.full(3, .5), recovery_tau=np.full(3, .02))


@pytest.mark.skipif(not CUDA_ENABLED, reason="Root reservation required: RUN_E289_CUDA=1")
def test_cuda_fixed_tiny_coupled_oracle_and_cut_disk_future(tmp_path):
    cpu, gpu = model(), model("cuda")
    for n in (cpu, gpu): seeded(n); n.set_muted([1])
    drive = np.full(6, 3., np.float32)
    for n in (cpu, gpu): n.advance(drive, steps=CUDA_TICKS)
    for key in ("rate", "adaptation"): np.testing.assert_allclose(gpu.snapshot()[key], cpu.snapshot()[key], atol=CUDA_RATE_ATOL, rtol=0)
    np.testing.assert_allclose(gpu.snapshot()["synaptic_resource"], cpu.snapshot()["synaptic_resource"], atol=CUDA_RESOURCE_ATOL, rtol=0)
    saved = gpu.snapshot(); pointers = dict(gpu._parameter_pointers)
    StateStore.save(tmp_path/"gpu", saved); gpu.reset(); gpu.restore(StateStore.load(tmp_path/"gpu"))
    assert all(getattr(gpu, key).data_ptr() == pointer for key, pointer in pointers.items())
    first = model("cuda"); first.restore(saved)
    for n in (first, gpu): n.advance(drive, steps=10)
    same(first.snapshot(), gpu.snapshot())


@pytest.mark.skipif(not CUDA_ENABLED, reason="Root reservation required: RUN_E289_CUDA=1")
def test_cuda_neutral_captured_advance_bit_exact():
    args, kw = parameters("cuda"); base = MotorTimescaleNetwork(*args, **kw); n = model("cuda", u=np.zeros(3))
    n.import_base_snapshot(base.snapshot())
    for control in range(3):
        for network in (base, n): network.set_muted([0] if control == 1 else []); network.advance(np.full(6, 3., np.float32), steps=10)
        for key in ("rate", "adaptation", "drive", "output_mask", "tick"): same(base.snapshot()[key], n.snapshot()[key])
        assert np.all(n.resource.cpu().numpy() == 1)
