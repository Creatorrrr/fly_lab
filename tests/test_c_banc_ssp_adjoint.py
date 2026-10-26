"""E314 generated CPU SSP forward/surrogate VJP; CUDA qualification is opt-in."""

import copy
import json
import os
import time

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from tools import banc_m3_ssp_adjoint as adjoint
from tools.banc_m3_ssp_adjoint import SSPAdjointModel

BEGAN = time.perf_counter()
COUNTS = {"generated_steps": 0, "finite_differences": 0, "fake_neurons": 0,
          "original": 0, "GraphStore": 0, "CUDA_executed": 0, "body": 0, "fit": 0}


@pytest.fixture(scope="session", autouse=True)
def generated_cpu_limits():
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        yield
    import psutil
    memory = psutil.Process().memory_info()
    peak = getattr(memory, "peak_wset", memory.rss)
    wall = time.perf_counter() - BEGAN
    print(json.dumps({"e314_generated_cpu": COUNTS, "fixture_wall_s": wall, "peak_rss_bytes": peak}, sort_keys=True))
    assert wall < 90 and peak < 2 * 1024 ** 3


class DenseCurrent:
    """Independent tiny differentiable current for validating the SSP chain."""

    def __init__(self, weights, groups):
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.groups = np.asarray(groups, np.int64)
        self.n = self.weights.shape[0]
        self.device = torch.device("cpu")
        self.group_count = int(self.groups.max()) + 1
        self.group_mask = [torch.tensor(self.groups == k) for k in range(self.group_count)]

    def current(self, masked_rate, efficacy):
        assert masked_rate.dtype == torch.float32 and masked_rate.is_contiguous()
        theta = efficacy.to(torch.float32)
        factors = sum(torch.where(mask, theta[k], 0.) for k, mask in enumerate(self.group_mask))
        return (self.weights * factors) @ masked_rate


def fixture(n=5):
    rng = np.random.default_rng(314)
    weights = rng.normal(0, .06, (n, n)).astype(np.float32)
    groups = np.arange(n * n, dtype=np.int64).reshape(n, n) % 3
    operator = DenseCurrent(weights, groups)
    arrays = {
        "tau": np.linspace(.003, .006, n, dtype=np.float32),
        "coefficient": np.linspace(.35, .7, n, dtype=np.float32),
        "threshold": np.linspace(.08, .15, n, dtype=np.float32),
        "cap": np.linspace(.9, 1.4, n, dtype=np.float32),
        "gain": np.linspace(0., 1.1, n, dtype=np.float32),
        "inverse_adaptation_tau": np.linspace(0., 17., n, dtype=np.float32),
    }
    model = SSPAdjointModel(operator, **arrays, dt=.00025)
    state = {
        "rate": torch.tensor(np.linspace(.13, .27, n, dtype=np.float32)),
        "adaptation": torch.tensor(np.linspace(.02, .06, n, dtype=np.float32)),
        "drive": torch.tensor(np.linspace(1.3, 1.8, n, dtype=np.float32)),
        "efficacy": torch.tensor([.8, 1., 1.3], dtype=torch.float64),
        "mask": torch.tensor([1. if i % 3 else 0. for i in range(n)], dtype=torch.float32),
    }
    COUNTS["fake_neurons"] += n
    return model, arrays, weights, groups, state


def independent_ssp_f64(arrays, weights, groups, state, steps=1):
    """Dense NumPy f64 recurrence, structurally separate from production code."""
    native = {name: np.asarray(value, np.float64) for name, value in arrays.items()}
    r, z = (np.asarray(state[key], np.float64).copy() for key in ("rate", "adaptation"))
    drive, efficacy, mask = (np.asarray(state[key], np.float64) for key in ("drive", "efficacy", "mask"))
    matrix = weights.astype(np.float64) * efficacy[groups]
    for _ in range(steps):
        base_r, base_z = r.copy(), z.copy()
        for b, w in ((0., 1.), (.75, .25), (1. / 3., 2. / 3.)):
            recurrent = matrix @ (r * mask)
            activation = native["coefficient"] * (drive + recurrent - native["threshold"] - z)
            target = np.maximum(native["cap"] * np.tanh(activation), 0.)
            qmem = .00025 / native["tau"]
            qadapt = .00025 * native["inverse_adaptation_tau"]
            next_r = b * base_r + w * ((1. - qmem) * r + qmem * target)
            next_z = b * base_z + w * ((1. - qadapt) * z + qadapt * native["gain"] * r)
            r, z = next_r, next_z
    return r, z


def scalar_f32(values, current, base_r, base_z, r, z, b, w):
    f = np.float32
    x = f(f(f(f(values["drive"] + current) - values["threshold"]) - z) * values["coefficient"])
    target = f(max(f(values["cap"] * f(np.tanh(x))), f(0.)))
    qr = f(f(.00025) / values["tau"])
    qz = f(f(.00025) * values["inverse_adaptation_tau"])
    rfe = f(f(f(f(1.) - qr) * r) + f(qr * target))
    zfe = f(f(f(f(1.) - qz) * z) + f(qz * f(values["gain"] * r)))
    return f(f(f(b) * base_r) + f(f(w) * rfe)), f(f(f(b) * base_z) + f(f(w) * zfe))


def test_cpu_native_pointwise_stage_order_and_input_nonmutation():
    class FixedCurrent:
        n, device, group_count = 1, torch.device("cpu"), 1
        def current(self, masked_rate, efficacy):
            return torch.tensor([np.float32(.125)], dtype=torch.float32)
    vals = {"tau": np.array([.003], np.float32), "coefficient": np.array([.625], np.float32),
        "threshold": np.array([.2], np.float32), "cap": np.array([1.1], np.float32),
        "gain": np.array([1.], np.float32), "inverse_adaptation_tau": np.array([7.], np.float32)}
    model = SSPAdjointModel(FixedCurrent(), **vals, dt=.00025)
    rate, adapt, drive, mask, eff = (torch.tensor([v], dtype=torch.float32) for v in (.2, .05, 1.3, 1., 1.))
    before = [v.clone() for v in (rate, adapt, drive, mask, eff)]
    out_r, out_z = model.step(rate, adapt, drive, eff, mask)
    scalar = {key: value[0] for key, value in vals.items()}
    scalar["drive"] = np.float32(1.3)
    r0, z0 = np.float32(.2), np.float32(.05)
    r, z = r0, z0
    for b, w in ((0., 1.), (.75, .25), (np.float32(1. / 3.), np.float32(2. / 3.))):
        r, z = scalar_f32(scalar, np.float32(.125), r0, z0, r, z, b, w)
    assert out_r.item() == pytest.approx(float(r), abs=2e-7)
    assert out_z.item() == pytest.approx(float(z), abs=2e-7)
    assert all(torch.equal(a, b) for a, b in zip(before, (rate, adapt, drive, mask, eff), strict=True))
    COUNTS["generated_steps"] += 1


def test_generated_39_cell_grouped_sparse_cpu_step_tracks_independent_dense_equation():
    from scipy.sparse import csr_matrix

    from tools.banc_m3_grouped_current_adjoint import GroupedCurrent

    dense = np.zeros((39, 39), np.float32)
    dense[0, 1:39] = np.array([.01 if i % 2 else -.02 for i in range(38)], np.float32)
    dense[1, 0] = .03
    weights = csr_matrix(dense)
    selected = np.r_[np.arange(37), 38].astype(np.int64)
    group_ids = np.arange(len(selected), dtype=np.int64) % 3
    operator = GroupedCurrent(weights, selected, group_ids, 3, "cpu")
    arrays = {"tau": np.linspace(.025, .035, 39, dtype=np.float32),
        "coefficient": np.linspace(.7, 1.1, 39, dtype=np.float32),
        "threshold": np.linspace(.2, .4, 39, dtype=np.float32),
        "cap": np.linspace(25., 40., 39, dtype=np.float32),
        "gain": np.linspace(0., 1., 39, dtype=np.float32),
        "inverse_adaptation_tau": np.linspace(0., 5., 39, dtype=np.float32)}
    model = SSPAdjointModel(operator, **arrays, dt=.00025)
    state = {"rate": torch.tensor(np.linspace(.02, .4, 39, dtype=np.float32)),
        "adaptation": torch.tensor(np.linspace(.001, .03, 39, dtype=np.float32)),
        "drive": torch.tensor(np.linspace(1.2, 1.5, 39, dtype=np.float32)),
        "efficacy": torch.tensor([.8, 1.2, 1.1], dtype=torch.float64),
        "mask": torch.tensor(np.where(np.arange(39) % 7 == 0, 0., 1.).astype(np.float32))}
    prepared = model.prepare_efficacy(state["efficacy"])
    got_r, got_z = model.step(state["rate"], state["adaptation"], state["drive"], prepared, state["mask"])
    direct_r, direct_z = model.step(**state)
    assert torch.equal(got_r, direct_r) and torch.equal(got_z, direct_z)
    native_weight = weights.data.copy()
    native_weight[selected] = (native_weight[selected] * state["efficacy"].numpy().astype(np.float32)[group_ids]).astype(np.float32)
    effective = csr_matrix((native_weight, weights.indices.copy(), weights.indptr.copy()), shape=weights.shape).toarray()
    reference = {key: value.detach().numpy().copy() for key, value in state.items()}
    reference["efficacy"] = np.ones(1, np.float64)
    rr, zz = independent_ssp_f64(arrays, effective, np.zeros((39, 39), np.int64), reference)
    assert np.allclose(got_r.detach().numpy(), rr, rtol=3e-6, atol=3e-6)
    assert np.allclose(got_z.detach().numpy(), zz, rtol=3e-6, atol=3e-6)
    state["efficacy"][0] += .01
    with pytest.raises(RuntimeError, match="mutated"):
        model.step(state["rate"], state["adaptation"], state["drive"], prepared, state["mask"])
    COUNTS["generated_steps"] += 1
    COUNTS["fake_neurons"] += 39


def test_surrogate_full_three_stage_and_two_step_vjp_against_independent_f64_fd():
    model, arrays, weights, groups, state = fixture()
    for key in ("rate", "adaptation", "drive", "efficacy", "mask"):
        state[key] = state[key].clone().requires_grad_(True)
    r1, z1 = model.step(**state)
    r2, z2 = model.step(r1, z1, state["drive"], state["efficacy"], state["mask"])
    weights_r = torch.tensor([.2, -.1, .3, .1, -.2], dtype=torch.float32)
    weights_z = torch.tensor([-.3, .1, .2, -.2, .1], dtype=torch.float32)
    objective = (r2 * weights_r).sum() + (z2 * weights_z).sum()
    objective.backward()
    reference = {key: value.detach().numpy().astype(np.float64) for key, value in state.items()}
    rr, zz = independent_ssp_f64(arrays, weights, groups, reference, steps=2)
    assert np.allclose(r2.detach().numpy(), rr, rtol=3e-7, atol=5e-7)
    assert np.allclose(z2.detach().numpy(), zz, rtol=3e-7, atol=5e-7)
    def f(values):
        r, z = independent_ssp_f64(arrays, weights, groups, values, steps=2)
        return float(r @ weights_r.numpy() + z @ weights_z.numpy())
    for key in ("rate", "adaptation", "drive", "efficacy"):
        observed = state[key].grad.detach().numpy()
        assert np.isfinite(observed).all() and np.any(np.abs(observed) > 1e-7)
        for index in range(len(observed)):
            plus, minus = copy.deepcopy(reference), copy.deepcopy(reference)
            plus[key][index] += 1e-5
            minus[key][index] -= 1e-5
            numeric = (f(plus) - f(minus)) / 2e-5
            COUNTS["finite_differences"] += 2
            assert observed[index] == pytest.approx(numeric, abs=2e-4, rel=1e-3)
    assert state["mask"].grad is not None and torch.isfinite(state["mask"].grad).all()
    COUNTS["generated_steps"] += 2


def test_two_live_efficacy_graphs_do_not_interfere_and_saved_tensor_mutation_rejected():
    model, _arrays, _weights, _groups, state = fixture()
    rate = state["rate"].requires_grad_()
    a = state["efficacy"].clone().requires_grad_()
    b = (state["efficacy"] * 1.1).clone().requires_grad_()
    ra, _ = model.step(rate, state["adaptation"], state["drive"], a, state["mask"])
    rb, _ = model.step(rate, state["adaptation"], state["drive"], b, state["mask"])
    (ra.sum() + 2. * rb.sum()).backward()
    assert a.grad is not None and b.grad is not None and not torch.equal(a.grad, b.grad)
    rate = state["rate"].clone().requires_grad_()
    result, _ = model.step(rate, state["adaptation"], state["drive"], a.detach(), state["mask"])
    with torch.no_grad():
        rate.add_(.01)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.sum().backward()


def test_exact_rectifier_kink_uses_zero_surrogate_subgradient():
    class ZeroCurrent:
        n, device, group_count = 1, torch.device("cpu"), 1
        def current(self, masked_rate, efficacy):
            return torch.zeros_like(masked_rate)
    values = {"tau": np.array([.01], np.float32), "coefficient": np.array([1.], np.float32),
        "threshold": np.array([.25], np.float32), "cap": np.array([1.], np.float32),
        "gain": np.array([0.], np.float32), "inverse_adaptation_tau": np.array([0.], np.float32)}
    model = SSPAdjointModel(ZeroCurrent(), **values, dt=.00025)
    drive = torch.tensor([.375], dtype=torch.float32, requires_grad=True)
    rate, _ = model.step(torch.tensor([0.], dtype=torch.float32), torch.tensor([.125], dtype=torch.float32),
        drive, torch.ones(1, dtype=torch.float32), torch.ones(1, dtype=torch.float32))
    rate.sum().backward()
    assert drive.grad.item() == 0.


def test_cuda_predicate_branch_never_converts_a_device_value_on_host(monkeypatch):
    class DevicePredicate:
        def __bool__(self):
            raise AssertionError("host value conversion forbidden")
    calls = []
    monkeypatch.setattr(adjoint.torch, "_assert_async", lambda value, message: calls.append((value, message)))
    predicate = DevicePredicate()
    adjoint._require(predicate, "generated async check", torch.device("cuda:0"))
    assert calls == [(predicate, "generated async check")]
    with pytest.raises(AssertionError, match="host value conversion"):
        adjoint._require(predicate, "CPU immediate check", torch.device("cpu"))


@pytest.mark.parametrize("field,mutation", [
    ("tau", lambda x: x.__setitem__(0, 0)),
    ("cap", lambda x: x.__setitem__(0, np.nan)),
    ("gain", lambda x: x.__setitem__(0, -.01)),
    ("inverse_adaptation_tau", lambda x: x.__setitem__(0, np.inf)),
    ("coefficient", lambda x: x.__setitem__(0, np.nan)),
    ("threshold", lambda x: x.__setitem__(0, np.inf)),
])
def test_resolved_physiology_rejects_invalid_arrays(field, mutation):
    _model, arrays, weights, groups, _state = fixture()
    bad = copy.deepcopy(arrays)
    mutation(bad[field])
    with pytest.raises(ValueError): SSPAdjointModel(DenseCurrent(weights, groups), **bad, dt=.00025)


def test_step_rejects_invalid_state_mask_current_and_stability():
    model, _arrays, _weights, _groups, state = fixture()
    with pytest.raises(ValueError): model.step(**{**state, "mask": torch.full((5,), .5)})
    with pytest.raises(ValueError): model.step(**{**state, "rate": torch.full((5,), -.1)})
    with pytest.raises(ValueError): model.step(**{**state, "drive": torch.full((5,), np.nan)})
    with pytest.raises(ValueError): model.step(**{**state, "efficacy": torch.tensor([1., -1., 1.], dtype=torch.float64)})
    with pytest.raises(ValueError): model.step(**{**state, "adaptation": state["adaptation"][:4]})
    with pytest.raises(ValueError): model.step(**{**state, "rate": torch.stack((state["rate"], state["rate"]), dim=1)[:, 0]})
    class BadCurrent(DenseCurrent):
        def current(self, masked_rate, efficacy):
            return torch.full((self.n,), np.nan)
    bad = BadCurrent(model.operator.weights.numpy(), model.operator.groups)
    altered = SSPAdjointModel(bad, model.tau, model.coefficient, model.threshold, model.cap,
        model.gain, model.inverse_adaptation_tau, model.dt)
    with pytest.raises(ValueError): altered.step(**state)
    with pytest.raises(ValueError): SSPAdjointModel(model.operator, model.tau, model.coefficient,
        model.threshold, model.cap, model.gain, model.inverse_adaptation_tau, dt=.002)


@pytest.mark.skipif(os.getenv("E314_CUDA_QUALIFY") != "1", reason="Root-only opt-in CUDA after source approval")
def test_cuda_generated_native_ssp_u0_exact_when_root_approves(monkeypatch):
    """Only root may execute the generated 39-cell native SSP parity gate."""
    from scipy.sparse import csr_matrix

    from tools.banc_m3_grouped_current_adjoint import GroupedCurrent
    from tools.banc_m3_ssp_recovery_synapses import SSPRecoveryResourceNetwork

    dense = np.zeros((39, 39), np.float32)
    dense[0, 1:39] = np.array([.01 if i % 2 else -.02 for i in range(38)], np.float32)
    dense[1, 0] = .03
    weights = csr_matrix(dense)
    # The final edge in row 0 belongs to motor38 and is unselected.
    selected = np.r_[np.arange(37), 38].astype(np.int64)
    grouped = GroupedCurrent(weights, selected, np.arange(len(selected), dtype=np.int64) % 3, 3, "cuda")
    native = SSPRecoveryResourceNetwork(weights, np.full(39, .03, np.float32), np.ones(39, np.float32),
        np.full(39, 1.5, np.float32), np.full(39, 40., np.float32),
        adaptation_gain=.8, adaptation_tau_s=.3, dt=.00025, device="cuda", cuda_implementation="packed",
        capture_steps=1, core_ids=np.array([0, 1], np.int64), core_gain=.1,
        motor_ids=np.array([38], np.int64), motor_gains=np.ones(1, np.float32), motor_log_tau=np.zeros(1),
        edge_ids=selected, premotor_ids=np.arange(38, dtype=np.int64),
        utilization=np.zeros(38, np.float64), recovery_tau=np.full(38, .15, np.float64))
    start = native.snapshot()
    start["rate"][:] = np.linspace(.02, .4, 39, dtype=np.float32)
    start["adaptation"][:] = np.linspace(.001, .03, 39, dtype=np.float32)
    start["drive"][:] = np.linspace(1.2, 1.5, 39, dtype=np.float32)
    start["output_mask"][7] = 0
    native.restore(start)
    model = SSPAdjointModel(grouped, native.tau, native.coefficient, native.threshold, native.cap,
        native._cell_gain, native._cell_inverse_tau, native.dt)
    rate = native.rate.detach().clone()
    adaptation = native.adaptation.detach().clone()
    drive = native.drive.detach().clone()
    mask = native.output_mask.detach().clone()
    efficacy = torch.ones(3, device="cuda", dtype=torch.float32)
    prepared = model.prepare_efficacy(efficacy)
    for _ in range(3):
        rate, adaptation = model.step(rate, adaptation, drive, prepared, mask)
        native.advance(start["drive"], steps=1)
        assert rate.detach().cpu().numpy().tobytes() == native.rate.detach().cpu().numpy().tobytes()
        assert adaptation.detach().cpu().numpy().tobytes() == native.adaptation.detach().cpu().numpy().tobytes()
    # The prepared hot step may launch validation kernels but cannot fetch a
    # CUDA scalar or copy any CUDA tensor back to the host.
    old_item, old_bool, old_cpu = torch.Tensor.item, torch.Tensor.__bool__, torch.Tensor.cpu
    def forbid_host(method):
        def checked(value, *args, **kwargs):
            if value.device.type == "cuda":
                raise AssertionError("CUDA hot step synchronized to host via " + method.__name__)
            return method(value, *args, **kwargs)
        return checked
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbid_host(old_item))
        patch.setattr(torch.Tensor, "__bool__", forbid_host(old_bool))
        patch.setattr(torch.Tensor, "cpu", forbid_host(old_cpu))
        model.step(rate, adaptation, drive, prepared, mask)
    assert np.array_equal(native.resource.detach().cpu().numpy(), np.ones(38, np.float64))
    COUNTS["CUDA_executed"] += 1
