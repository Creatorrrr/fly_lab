"""Tiny generated CPU fixtures: independent values, gradients, and E329 guards."""
import numpy as np
import pytest
import torch

from tools.banc_m3_phase_free_temporal import PhaseFreeTemporalObjective

FREQ = 2.382760007990865
DP = FREQ * .005


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def curves128():
    theta = 2 * np.pi * np.arange(128) / 128
    curves = np.zeros((6, 128, 7), dtype=np.float64)
    for leg in range(6):
        curves[leg, :, 0] = (.5 + .1 * leg) * np.cos(theta)
        curves[leg, :, 1] = (.6 + .08 * leg) * np.sin(theta)
        curves[leg, :, 2:] = .03 * leg
    return curves


def sample(curves, phases):
    result = np.empty((len(phases), 42))
    for row, per_leg in enumerate(phases):
        for leg, phase in enumerate(per_leg):
            position = (phase % 1) * 128
            index = int(np.floor(position))
            fraction = position - index
            a, b = curves[leg, index], curves[leg, (index + 1) % 128]
            result[row, leg * 7:(leg + 1) * 7] = (1 - fraction) * a + fraction * b
    return result


def numpy_project(curves, q):
    x = q.reshape(-1, 6, 7)
    edge = np.roll(curves, -1, axis=1) - curves
    delta = x[:, :, None] - curves[None]
    denominator = np.einsum("lmj,lmj->lm", edge, edge)
    u = np.clip(np.einsum("tlmj,lmj->tlm", delta, edge) / denominator[None], 0, 1)
    residual = delta - u[..., None] * edge[None]
    distance = np.einsum("tlmj,tlmj->tlm", residual, residual)
    index = np.argmin(distance, axis=2)
    chosen_u = np.take_along_axis(u, index[..., None], axis=2)[..., 0]
    phase = ((index + chosen_u) / 128) % 1
    return phase, np.take_along_axis(distance, index[..., None], axis=2)[..., 0] / 7


def numpy_value(curves, previous, next_q):
    anchors = np.repeat((np.arange(128) / 128)[:, None], 6, axis=1)
    increments = (sample(curves, anchors + DP) - sample(curves, anchors)).reshape(-1, 6, 7)
    normalization = np.mean(increments ** 2, axis=(0, 2))
    phase, _ = numpy_project(curves, previous)
    target = sample(curves, phase + DP)
    per_leg = np.mean((next_q - target).reshape(-1, 6, 7) ** 2, axis=(0, 2)) / normalization
    next_phase, shape = numpy_project(curves, next_q)
    coordination = np.mean([1 - np.cos(2 * np.pi * (next_phase[:, a] - next_phase[:, b]))
                            for a in range(6) for b in range(a + 1, 6)])
    lr = np.mean([1 - np.cos(2 * np.pi * (next_phase[:, a] - next_phase[:, a + 3]))
                  for a in range(3)])
    return {"training_loss": float(per_leg.mean() + coordination),
            "successor_loss": float(per_leg.mean()), "per_leg_successor_loss": per_leg,
            "coordination_loss": float(coordination), "lr_coordination_loss": float(lr),
            "per_leg_shape_loss": shape.mean(axis=0), "shape_loss": float(shape.mean())}


def objective(curves=None, startup_q=None, startup_t=None):
    return PhaseFreeTemporalObjective(curves128() if curves is None else curves,
                                      np.zeros((99, 42)) if startup_q is None else startup_q,
                                      np.zeros((99, 6)) if startup_t is None else startup_t)


def tensor(q, grad=False):
    return torch.tensor(q, dtype=torch.float64, requires_grad=grad)


def phase_fixture(count):
    return .137 + np.arange(count)[:, None] * DP * .91 + np.arange(6)[None, :] * .0021


def test_independent_numpy_value_and_separate_previous_next_gradients():
    curves = curves128()
    obj = objective(curves)
    previous = sample(curves, phase_fixture(3))
    next_q = sample(curves, phase_fixture(3) + DP * .93)
    next_q[:, 2::7] += .004
    prev, nxt = tensor(previous, True), tensor(next_q, True)
    actual = obj.evaluate_pairs(prev, nxt)
    expected = numpy_value(curves, previous, next_q)
    for key in expected:
        np.testing.assert_allclose(actual[key].detach().numpy(), expected[key], rtol=1e-10, atol=1e-10)
    actual["training_loss"].backward()
    rng = np.random.default_rng(32901)
    h = 1e-6
    for which, grad in (("previous", prev.grad.numpy()), ("next", nxt.grad.numpy())):
        direction = rng.normal(size=previous.shape)
        direction /= np.linalg.norm(direction)
        plus = numpy_value(curves, previous + (h * direction if which == "previous" else 0),
                           next_q + (h * direction if which == "next" else 0))["training_loss"]
        minus = numpy_value(curves, previous - (h * direction if which == "previous" else 0),
                            next_q - (h * direction if which == "next" else 0))["training_loss"]
        np.testing.assert_allclose(np.sum(grad * direction), (plus - minus) / (2 * h),
                                   rtol=1e-6, atol=1e-7)


def test_four_control_full_gradient_first_boundary_only_fixed_and_wrong_detach():
    curves = curves128()
    obj = objective(curves)
    phases = phase_fixture(5)
    boundary = sample(curves, phases[:1])
    commands = sample(curves, phases[1:])
    commands[:, 2::7] += np.arange(1, 5)[:, None] * .001
    q = tensor(commands, True)
    prev = torch.cat((tensor(boundary), q[:3]))
    terms = obj.evaluate_pairs(prev, q)
    actual = torch.autograd.grad(terms["training_loss"], q, retain_graph=True)[0].numpy()
    numeric = np.zeros_like(commands)
    h = 1e-6
    for index in np.ndindex(commands.shape):
        plus, minus = commands.copy(), commands.copy()
        plus[index] += h
        minus[index] -= h
        numeric[index] = (numpy_value(curves, np.vstack((boundary, plus[:3])), plus)["training_loss"]
                          - numpy_value(curves, np.vstack((boundary, minus[:3])), minus)["training_loss"]) / (2 * h)
    np.testing.assert_allclose(actual, numeric, rtol=1e-6, atol=1e-7)
    # Middle commands have contributions from next and from the next pair's
    # previous role. The incorrect detach preserves values but loses the latter.
    bad = obj.evaluate_pairs(prev.detach(), q)
    wrong = torch.autograd.grad(bad["training_loss"], q)[0].numpy()
    assert np.max(np.abs(actual[:3] - wrong[:3])) > 1e-4
    np.testing.assert_allclose(actual[-1], wrong[-1], rtol=1e-10, atol=1e-10)
    successor_grad = torch.autograd.grad(terms["successor_loss"], q, retain_graph=True)[0].numpy()
    coordination_grad = torch.autograd.grad(terms["coordination_loss"], q)[0].numpy()
    assert np.linalg.norm(successor_grad[-1]) > 1e-6
    assert np.linalg.norm(coordination_grad[-1]) > 1e-6
    np.testing.assert_allclose(actual[-1], successor_grad[-1] + coordination_grad[-1], rtol=1e-10, atol=1e-10)


def test_all15_counterexample_common_phase_and_static_reverse_scramble():
    curves = curves128()
    obj = objective(curves)
    p = np.repeat((np.arange(128) / 128)[:, None], 6, axis=1)
    correct = obj.evaluate_pairs(tensor(sample(curves, p)), tensor(sample(curves, p + DP)))
    assert correct["coordination_terms"].shape == (128, 15)
    assert correct["lr_coordination_terms"].shape == (128, 3)
    assert correct["training_loss"].item() <= 1e-10
    static = obj.evaluate_pairs(tensor(sample(curves, p)), tensor(sample(curves, p)))
    np.testing.assert_allclose(static["per_leg_successor_loss"].numpy(), 1, rtol=0, atol=1e-10)
    for altered in (p - DP, p[(np.arange(128) * 37 + 11) % 128] + DP):
        result = obj.evaluate_pairs(tensor(sample(curves, p)), tensor(sample(curves, altered)))
        assert result["training_loss"].item() > 1e-8
    p[:, [0, 3]] += .125
    shifted = obj.evaluate_pairs(tensor(sample(curves, p)), tensor(sample(curves, p + DP)))
    assert shifted["lr_coordination_loss"].item() <= 1e-10
    assert shifted["coordination_loss"].item() > .1
    common = obj.evaluate_pairs(tensor(sample(curves, p + .213)), tensor(sample(curves, p + .213 + DP)))
    np.testing.assert_allclose(common["coordination_loss"], shifted["coordination_loss"], rtol=1e-10, atol=1e-10)
    normal_p = np.repeat((.9997 + np.arange(12) * DP)[:, None], 6, axis=1)
    assert obj.evaluate_pairs(tensor(sample(curves, normal_p)), tensor(sample(curves, normal_p + DP)))["training_loss"].item() <= 1e-10


def test_wrap_first_tie_strict_clamp_and_fixed_reference_copy():
    curves = curves128()
    obj = objective(curves)
    p = np.repeat(np.array([-.003, .999, 1., 1.0003])[:, None], 6, axis=1)
    q = obj.interpolate(tensor(p))
    np.testing.assert_allclose(q.numpy(), sample(curves, p), rtol=1e-10, atol=1e-10)
    phase = obj.project(q)["phase"].numpy()
    difference = np.abs(phase - p % 1)
    assert np.minimum(difference, 1 - difference).max() <= 1e-10
    endpoint = tensor(curves[:, 0].reshape(1, 42))
    assert torch.equal(obj.project(endpoint)["segment"], torch.zeros((1, 6), dtype=torch.long))
    exterior = endpoint.clone().detach()
    exterior[:, 0::7] *= 2
    exterior.requires_grad_()
    clamp = obj.project(exterior)
    assert (clamp["raw_u"] < 0).all()
    clamp["phase"].sum().backward()
    assert exterior.grad.abs().max().item() == 0
    duplicate = curves.copy()
    duplicate[:, 64:] = duplicate[:, :64]
    # Keep both half loops closed and positive; duplicate nonlocal branches.
    loop = curves128()[:, ::2]
    duplicate[:, :64] = loop
    duplicate[:, 64:] = loop
    tie_obj = objective(duplicate)
    tie = tie_obj.project(tensor(duplicate[:, 10].reshape(1, 42)))
    assert (tie["segment"] < 64).all()
    assert torch.isfinite(tie["phase"]).all()
    before = obj.interpolate(tensor(p)).clone()
    curves[:] = 20
    torch.testing.assert_close(obj.interpolate(tensor(p)), before, rtol=0, atol=0)
    assert not obj.curves.requires_grad


def test_tiny_negative_and_subnormal_zero_neighbors_wrap_to_finite_first_anchor():
    curves = curves128()
    obj = objective(curves)
    # These negative remainders round to exactly one in f64. Expected values
    # come directly from the known seam anchor, not from a modulo helper.
    values = np.array([-1e-20, np.nextafter(0., -np.inf), -0., 0.,
                       np.nextafter(0., np.inf)], dtype=np.float64)
    phase = tensor(np.repeat(values[:, None], 6, axis=1), True)
    actual = obj.interpolate(phase)
    expected = np.repeat(curves[:, 0].reshape(1, 42), len(values), axis=0)
    expected[-1] += (values[-1] * 128 * (curves[:, 1] - curves[:, 0])).reshape(42)
    np.testing.assert_array_equal(actual.detach().numpy(), expected)
    assert torch.isfinite(actual).all()
    actual.sum().backward()
    assert torch.isfinite(phase.grad).all()
    projected = obj.project(actual.detach())
    np.testing.assert_array_equal(projected["phase"].numpy(), np.zeros((len(values), 6)))


def test_evaluate_exact_controls_startup_old_shape_and_window_value_mean():
    obj = objective()
    q = tensor(sample(curves128(), phase_fixture(300)))
    q[:99] = 0
    tendon = torch.zeros((300, 6), dtype=torch.float64)
    full = obj.evaluate(q, tendon)
    assert full["startup_pass"].item()
    expected = numpy_value(curves128(), q[99:299].numpy(), q[100:300].numpy())
    np.testing.assert_allclose(full["shape_loss"], expected["shape_loss"], rtol=1e-10, atol=1e-10)
    windows = [obj.evaluate_pairs(q[c - 1:c + 3], q[c:c + 4])["training_loss"].item()
               for c in range(100, 300, 4)]
    np.testing.assert_allclose(np.mean(windows), full["training_loss"], rtol=1e-10, atol=1e-10)
    changed = q.clone()
    changed[99, 0] += .1
    assert obj.evaluate(changed, tendon)["training_loss"] != full["training_loss"]
    assert obj.evaluate(changed, tendon)["startup_pass"].item()
    for key, delta in (("q_max", .041), ("q_rms", .019), ("t_max", .026), ("t_rms", .010)):
        cq, ct = q.clone(), tendon.clone()
        if key == "q_max":
            cq[98, 0] = delta
        elif key == "q_rms":
            cq[:99] = delta
        elif key == "t_max":
            ct[98, 0] = delta
        else:
            ct[:99] = delta
        assert not obj.evaluate(cq, ct)["startup_pass"].item()
    q.requires_grad_()
    tendon.requires_grad_()
    exact = obj.evaluate(q, tendon)
    (exact["startup_q_rms"] + exact["startup_tendon_rms"]).backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(tendon.grad).all()


def test_invalid_dtype_device_shape_fixed_clock_zero_edges_and_overflow():
    obj = objective()
    q = tensor(sample(curves128(), phase_fixture(2)))
    with pytest.raises(TypeError):
        obj.evaluate_pairs(q.float(), q)
    with pytest.raises(TypeError):
        obj.evaluate_pairs(q.numpy(), q)
    with pytest.raises(ValueError):
        obj.evaluate_pairs(torch.empty((2, 42), dtype=torch.float64, device="meta"), q)
    for bad in (q[:0], q[:, :41], q[:1]):
        with pytest.raises(ValueError):
            obj.evaluate_pairs(bad, q)
    with pytest.raises(ValueError):
        obj.evaluate(torch.zeros((299, 42), dtype=torch.float64), torch.zeros((299, 6), dtype=torch.float64))
    for value in (np.nan, np.inf, 1e308):
        bad = q.clone()
        bad[0, 0] = value
        with pytest.raises((ValueError, RuntimeError)):
            obj.evaluate_pairs(bad, q)
    for kwargs in ({"frequency": 1}, {"dt": .01}, {"frequency": np.nan}, {"dt": True}):
        with pytest.raises(ValueError):
            PhaseFreeTemporalObjective(curves128(), np.zeros((99, 42)), np.zeros((99, 6)), **kwargs)
    for bad in (curves128().astype(np.float32), curves128()[:, :127]):
        with pytest.raises((ValueError, TypeError)):
            objective(bad)
    for value in (0, np.nan, 1e308):
        bad = curves128()
        if value == 0:
            bad[:, 1] = bad[:, 0]
        else:
            bad[0, 0, 0] = value
        with pytest.raises(ValueError):
            objective(bad)
    # Squared edge lengths can be representable while fixed 5ms increment
    # normalization underflows to zero. This must not be epsilon-repaired.
    tiny = curves128() * 1e-161
    with pytest.raises(ValueError):
        objective(tiny)
