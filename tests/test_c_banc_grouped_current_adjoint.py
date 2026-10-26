"""Pure generated E314 sparse-current checks; CUDA is root opt-in only."""

import json
import os
import time

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from tools import banc_m3_grouped_current_adjoint as m

CUDA_ENABLED = os.environ.get("RUN_E314_CUDA") == "1"
COUNTS = dict.fromkeys(
    (
        "constructor_requested",
        "constructor_completed",
        "constructor_failed",
        "forward_requested",
        "forward_completed",
        "forward_failed",
        "backward_requested",
        "backward_completed",
        "backward_failed",
        "native_calls",
        "oracle_calls",
        "original_calls",
        "model_calls",
        "body_calls",
        "solver_calls",
        "fit_calls",
        "CUDA_calls",
    ),
    0,
)
COUNTS.update(constructor_wall_s=0.0, forward_wall_s=0.0, backward_wall_s=0.0)


@pytest.fixture(scope="module", autouse=True)
def metering():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    start, cpu = time.perf_counter(), time.process_time()
    yield
    COUNTS.update(wall_s=time.perf_counter() - start, cpu_s=time.process_time() - cpu)
    torch.set_num_threads(old)
    print("E314_GENERATED_COST " + json.dumps(COUNTS, sort_keys=True))
    assert COUNTS["wall_s"] <= 90
    assert (
        COUNTS["original_calls"] == COUNTS["model_calls"] == COUNTS["body_calls"] == 0
    )


@pytest.fixture(autouse=True)
def count(monkeypatch):
    constructor, forward, backward = (
        m.GroupedCurrent.__init__,
        m._Current.forward,
        m._Current.backward,
    )

    def wrap(method, kind):
        def call(*args, **kwargs):
            COUNTS[kind + "_requested"] += 1
            start = time.perf_counter()
            try:
                result = method(*args, **kwargs)
                COUNTS[kind + "_completed"] += 1
                return result
            except BaseException:
                COUNTS[kind + "_failed"] += 1
                raise
            finally:
                COUNTS[kind + "_wall_s"] += time.perf_counter() - start

        return call

    monkeypatch.setattr(m.GroupedCurrent, "__init__", wrap(constructor, "constructor"))
    monkeypatch.setattr(m._Current, "forward", staticmethod(wrap(forward, "forward")))
    monkeypatch.setattr(
        m._Current, "backward", staticmethod(wrap(backward, "backward"))
    )


def fixture():
    n = 39
    dense = np.zeros((n, n), np.float32)
    dense[0, :38] = np.array(
        [(-0.04 if i % 3 else 0.07) for i in range(38)], np.float32
    )
    dense[1, :8] = np.array([0.2, -0.1, 0.3, -0.2, 0.1, -0.3, 0.4, -0.5], np.float32)
    dense[2, 0], dense[38, 4] = 0.06, -0.08
    weights = csr_matrix(dense)
    # Explicit, unselected zero storage is part of the original topology.
    weights.data = np.r_[weights.data, np.float32(0)]
    weights.indices = np.r_[weights.indices, np.int32(10)]
    weights.indptr[39] += 1
    assert weights.has_sorted_indices and weights.has_canonical_format
    chosen = np.array([0, 3, 6, 10, 20, 33, 37, 39, 45], np.int64)
    groups = np.array([1, 0, 2, 1, 0, 2, 1, 0, 2], np.int64)
    rate = np.linspace(0.1, 2, n).astype(np.float32)
    rate[[7, 15]] = 0  # Caller-side mask already applied.
    return weights, chosen, groups, rate


def new(weights=None, ids=None, groups=None, device="cpu"):
    w, e, g, _ = fixture()
    return m.GroupedCurrent(
        w if weights is None else weights,
        e if ids is None else ids,
        g if groups is None else groups,
        3,
        device,
    )


def ftz(v):
    v = np.float32(v)
    return np.copysign(np.float32(0), v) if abs(v) < np.finfo(np.float32).tiny else v


def independent_native(weights, ids, groups, rate, theta):
    COUNTS["oracle_calls"] += 1
    chosen = dict(zip(ids, groups, strict=True))
    out = np.zeros(len(rate), np.float32)
    for row in range(len(rate)):
        lane = np.zeros(32, np.float32)
        for i in range(32):
            for edge in range(weights.indptr[row] + i, weights.indptr[row + 1], 32):
                original = weights.data[edge]
                group = chosen.get(edge)
                if group is not None and np.float32(theta[group]) != np.float32(1):
                    original = np.float32(original * np.float32(theta[group]))
                lane[i] = ftz(
                    ftz(lane[i]) + ftz(ftz(original) * ftz(rate[weights.indices[edge]]))
                )
        for stride in (16, 8, 4, 2, 1):
            previous = lane.copy()
            for i in range(32 - stride):
                lane[i] = ftz(ftz(previous[i]) + ftz(previous[i + stride]))
        out[row] = lane[0]
    return out


def dense_surrogate(weights, ids, groups, rate, theta, upstream):
    """Independent f64 real equation, retaining rounded selected f32 W(theta)."""
    post = np.searchsorted(weights.indptr, np.arange(weights.nnz), side="right") - 1
    selected = dict(zip(ids, groups, strict=True))
    W = np.zeros(weights.shape, np.float64)
    ge = np.zeros(3, np.float64)
    for edge, pre in enumerate(weights.indices):
        group = selected.get(edge)
        base = float(weights.data[edge])
        weight = (
            base
            if group is None or np.float32(theta[group]) == 1
            else float(np.float32(np.float32(base) * np.float32(theta[group])))
        )
        W[post[edge], pre] += weight
        if group is not None:
            ge[group] += float(upstream[post[edge]]) * base * float(rate[pre])
    return W.T @ upstream, ge


@pytest.mark.parametrize(
    "theta",
    [np.array([1.0, 1.0, 1.0]), np.array([0.7, 1.4, 0.3]), np.array([1.0, 0.5, 2.0])],
)
def test_packed_warp_order_neutral_and_grouped_native(theta):
    weights, ids, groups, rate = fixture()
    before = tuple(
        v.tobytes()
        for v in (weights.data, weights.indices, weights.indptr, ids, groups)
    )
    op = new(weights, ids[::-1], groups[::-1])
    actual = (
        op.current(torch.tensor(rate), torch.tensor(theta, dtype=torch.float64))
        .detach()
        .numpy()
    )
    expected = independent_native(weights, ids, groups, rate, theta)
    assert actual.tobytes() == expected.tobytes()
    assert before == tuple(
        v.tobytes()
        for v in (weights.data, weights.indices, weights.indptr, ids, groups)
    )
    assert np.array_equal(op.selected_edge_ids, np.sort(ids))
    if np.all(theta == 1):
        assert (
            actual.tobytes()
            == independent_native(weights, [], [], rate, theta).tobytes()
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_surrogate_real_dense_gradient_and_finite_difference(dtype):
    weights, ids, groups, rate = fixture()
    op = new()
    r = torch.tensor(rate, requires_grad=True)
    e = torch.tensor([0.7, 1.4, 0.3], dtype=dtype, requires_grad=True)
    y = np.linspace(-2, 3, 39).astype(np.float32)
    out = op.current(r, e)
    (out * torch.tensor(y)).sum().backward()
    dr, dg = dense_surrogate(weights, ids, groups, rate, e.detach().numpy(), y)
    np.testing.assert_allclose(r.grad.numpy(), dr, atol=2e-6, rtol=1e-6)
    np.testing.assert_allclose(e.grad.numpy(), dg, atol=2e-6, rtol=1e-6)
    # Finite difference of the smooth real surrogate (not rounded native output).
    eps = 1e-5
    post = np.searchsorted(weights.indptr, np.arange(weights.nnz), side="right") - 1
    selected = dict(zip(ids, groups, strict=True))

    def objective(rate_values, theta_values, *, rounded_weights):
        total = 0.0
        for edge, pre in enumerate(weights.indices):
            group = selected.get(edge)
            base = float(weights.data[edge])
            weight = base
            if group is not None:
                weight = (
                    float(
                        np.float32(np.float32(base) * np.float32(theta_values[group]))
                    )
                    if rounded_weights
                    else base * theta_values[group]
                )
            total += float(y[post[edge]]) * weight * float(rate_values[pre])
        return total

    theta = e.detach().numpy().astype(np.float64)
    for pre in (0, 4, 20):
        pos, neg = rate.astype(np.float64), rate.astype(np.float64)
        pos[pre] += eps
        neg[pre] -= eps
        # W(theta32) stays fixed for a rate perturbation.
        finite = (
            objective(pos, theta, rounded_weights=True)
            - objective(neg, theta, rounded_weights=True)
        ) / (2 * eps)
        np.testing.assert_allclose(finite, dr[pre], atol=2e-8)
    # The group surrogate is the unrounded real efficacy factor.
    for group in range(3):
        pos, neg = theta.copy(), theta.copy()
        pos[group] += eps
        neg[group] -= eps
        np.testing.assert_allclose(
            (
                objective(rate, pos, rounded_weights=False)
                - objective(rate, neg, rounded_weights=False)
            )
            / (2 * eps),
            dg[group],
            atol=2e-8,
        )


def test_two_live_theta_graphs_reentrant_and_prepared_context():
    w, ids, groups, rate = fixture()
    op = new(w, ids, groups)
    r = torch.tensor(rate, requires_grad=True)
    a = torch.tensor([0.5, 1.0, 1.2], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([1.3, 0.8, 0.7], dtype=torch.float64, requires_grad=True)
    pa, pb = op.prepare_efficacy(a), op.prepare_efficacy(b)
    first, second = op.current(r, pa), op.current(r, pb)
    y = torch.linspace(-2, 3, op.n)
    ((first * 2 + second * 3) * y).sum().backward()
    dr_a, dg_a = dense_surrogate(w, ids, groups, rate, a.detach().numpy(), y.numpy())
    dr_b, dg_b = dense_surrogate(w, ids, groups, rate, b.detach().numpy(), y.numpy())
    np.testing.assert_allclose(r.grad.numpy(), 2 * dr_a + 3 * dr_b, atol=4e-6)
    np.testing.assert_allclose(a.grad.numpy(), 2 * dg_a, atol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), 3 * dg_b, atol=1e-5)


@pytest.mark.parametrize("leaf", ["rate", "efficacy"])
def test_saved_leaf_version_detects_inplace_mutation(leaf):
    op = new()
    r = torch.ones(op.n, requires_grad=True)
    e = torch.ones(op.group_count, dtype=torch.float64, requires_grad=True)
    out = op.current(r, e)
    with torch.no_grad():
        (r if leaf == "rate" else e).add_(0.1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()


def test_prepared_version_detects_mutation_before_next_stage():
    op = new()
    e = torch.ones(op.group_count, requires_grad=True)
    prepared = op.prepare_efficacy(e)
    op.current(torch.ones(op.n), prepared)
    with torch.no_grad():
        e[0] = 2.0
    with pytest.raises(RuntimeError, match="changed after validation"):
        op.current(torch.ones(op.n), prepared)


def test_prepared_cached_theta_inplace_mutation_refused_before_current():
    weights = csr_matrix(np.array([[0, 2], [3, 0]], np.float32))
    op = m.GroupedCurrent(
        weights, np.array([0], np.int64), np.array([0], np.int64), 1, "cpu"
    )
    efficacy = torch.ones(1, requires_grad=True)
    prepared = op.prepare_efficacy(efficacy)
    rate = torch.tensor([1.0, 4.0])
    np.testing.assert_array_equal(
        op.current(rate, prepared).detach().numpy(), [8.0, 3.0]
    )
    prepared.theta.mul_(2)
    assert efficacy.item() == 1.0
    with pytest.raises(RuntimeError, match="changed after validation"):
        op.current(rate, prepared)
    np.testing.assert_array_equal(
        op.current(rate, op.prepare_efficacy(efficacy)).detach().numpy(), [8.0, 3.0]
    )


@pytest.mark.parametrize(
    "field", ["operator", "efficacy", "theta", "version", "theta_version"]
)
def test_prepared_field_replacement_is_explicitly_immutable(field):
    op = new()
    efficacy = torch.ones(op.group_count)
    prepared = op.prepare_efficacy(efficacy)
    with pytest.raises(AttributeError, match="immutable"):
        setattr(prepared, field, torch.zeros(op.group_count))
    assert torch.isfinite(op.current(torch.ones(op.n), prepared)).all()


def test_prepared_theta_mutation_after_forward_detected_by_saved_version():
    op = new()
    efficacy = torch.ones(op.group_count, requires_grad=True)
    prepared = op.prepare_efficacy(efficacy)
    out = op.current(torch.ones(op.n, requires_grad=True), prepared)
    prepared.theta.mul_(2)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()


def test_cuda_device_index_resolution_without_cuda_allocation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    implicit = new(device="cuda")
    explicit = new(device=torch.device("cuda:3"))
    assert implicit.device == torch.device("cuda:2")
    assert explicit.device == torch.device("cuda:3")
    assert new(device="cpu").device == torch.device("cpu")


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "zero",
        "dtype",
        "group_gap",
        "group_negative",
        "edge_oob",
        "uncanonical",
        "nonfinite",
        "nonsquare",
    ],
)
def test_constructor_bad_csr_selection_rejected(kind):
    weights, ids, groups, _ = fixture()
    if kind == "duplicate":
        ids[-1] = ids[0]
    elif kind == "zero":
        ids[-1] = weights.nnz - 1
    elif kind == "dtype":
        ids = ids.astype(np.int32)
    elif kind == "group_gap":
        groups[groups == 1] = 0
    elif kind == "group_negative":
        groups[0] = -1
    elif kind == "edge_oob":
        ids[-1] = weights.nnz
    elif kind == "uncanonical":
        weights.indices[1] = weights.indices[0]
    elif kind == "nonfinite":
        weights.data[0] = np.inf
    else:
        weights = csr_matrix(np.ones((2, 3), np.float32))
    with pytest.raises((ValueError, TypeError)):
        m.GroupedCurrent(weights, ids, groups, 3, "cpu")


@pytest.mark.parametrize(
    "kind",
    [
        "rate_dtype",
        "rate_shape",
        "rate_nan",
        "rate_stride",
        "eff_dtype",
        "eff_shape",
        "eff_zero",
        "eff_negative",
        "eff_nan",
        "eff_underflow",
        "eff_overflow",
    ],
)
def test_runtime_bad_inputs_rejected(kind):
    op = new()
    r = torch.ones(op.n)
    e = torch.ones(op.group_count, dtype=torch.float64)
    if kind == "rate_dtype":
        r = r.double()
    elif kind == "rate_shape":
        r = r[:-1]
    elif kind == "rate_nan":
        r[0] = torch.nan
    elif kind == "rate_stride":
        r = torch.ones(op.n * 2)[::2]
    elif kind == "eff_dtype":
        e = e.int()
    elif kind == "eff_shape":
        e = e[:-1]
    elif kind == "eff_zero":
        e[0] = 0
    elif kind == "eff_negative":
        e[0] = -1
    elif kind == "eff_nan":
        e[0] = torch.nan
    elif kind == "eff_underflow":
        e[0] = 1e-100
    else:
        e[0] = 1e100
    with pytest.raises(ValueError):
        op.current(r, e)


def test_subnormal_scaled_weight_preserved_before_ftz_product():
    tiny = np.nextafter(np.float32(0), np.float32(1))
    weights = csr_matrix(np.array([[tiny, 0.5], [0, 0]], np.float32))
    op = m.GroupedCurrent(
        weights, np.array([0], np.int64), np.array([0], np.int64), 1, "cpu"
    )
    r = torch.tensor([1.0, 1.0])
    actual = op.current(r, torch.tensor([2.0])).numpy()
    assert actual[0] == 0.5  # f32 scaled subnormal still FTZ at original CSR product.
    assert op.current(r, torch.tensor([1.0])).numpy()[0] == 0.5


def test_cuda_source_packed_no_repack_and_surrogate_nonatomic():
    source = m._CUDA_SOURCE
    assert "theta[group]==1.0f" in source
    assert "float weight=scale_exact" in source
    assert "value+=weight*rate" in source
    assert "double sum=0.0" in source and "atomicAdd" not in source


@pytest.mark.skipif(not CUDA_ENABLED, reason="Root-only RUN_E314_CUDA=1")
def test_cuda_native_warp_neutral_nonneutral_and_dense_vjp():
    COUNTS["CUDA_calls"] += 1
    weights, ids, groups, rate = fixture()
    gpu = m.GroupedCurrent(weights, ids, groups, 3, "cuda")
    for theta in (np.ones(3, np.float64), np.array([0.7, 1.4, 0.3])):
        r = torch.tensor(rate, device="cuda", requires_grad=True)
        e = torch.tensor(theta, device="cuda", dtype=torch.float64, requires_grad=True)
        output = gpu.current(r, gpu.prepare_efficacy(e))
        np.testing.assert_array_equal(
            output.detach().cpu().numpy(),
            independent_native(weights, ids, groups, rate, theta),
        )
        y = np.linspace(-2, 3, 39).astype(np.float32)
        (output * torch.tensor(y, device="cuda")).sum().backward()
        dr, dg = dense_surrogate(weights, ids, groups, rate, theta, y)
        np.testing.assert_allclose(r.grad.cpu().numpy(), dr, atol=2e-6)
        np.testing.assert_allclose(e.grad.cpu().numpy(), dg, atol=2e-6)


@pytest.mark.skipif(not CUDA_ENABLED, reason="Root-only RUN_E314_CUDA=1")
def test_cuda_two_graphs_and_mutation_rejection():
    COUNTS["CUDA_calls"] += 1
    op = new(device="cuda")
    r = torch.ones(op.n, device="cuda", requires_grad=True)
    e = torch.ones(
        op.group_count, device="cuda", dtype=torch.float64, requires_grad=True
    )
    first = op.current(r, op.prepare_efficacy(e))
    second = op.current(r, torch.full_like(e, 1.2, requires_grad=True))
    (first + second).sum().backward()
    assert torch.isfinite(r.grad).all()
    previous = op.current(r, op.prepare_efficacy(e))
    with torch.no_grad():
        e[0] = 1.3
    with pytest.raises(RuntimeError):
        previous.sum().backward()


@pytest.mark.skipif(not CUDA_ENABLED, reason="Root-only RUN_E314_CUDA=1")
def test_cuda_real_vjp_preserves_subnormal_inputs_while_forward_is_ftz():
    COUNTS["CUDA_calls"] += 1
    tiny = np.nextafter(np.float32(0), np.float32(1)) * np.float32(128)
    weights = csr_matrix(np.array([[0, 1e30], [2, 0]], np.float32))
    operator = m.GroupedCurrent(weights, np.arange(2, dtype=np.int64),
        np.arange(2, dtype=np.int64), 2, "cuda")
    r_values, y_values = np.array([1e-30, tiny], np.float32), np.array([tiny, 1], np.float32)
    rate = torch.tensor(r_values, device="cuda", requires_grad=True)
    efficacy = torch.ones(2, device="cuda", dtype=torch.float64, requires_grad=True)
    output = operator.current(rate, operator.prepare_efficacy(efficacy))
    (output * torch.tensor(y_values, device="cuda")).sum().backward()
    expected_rate = (weights.toarray().astype(np.float64).T @ y_values.astype(np.float64)).astype(np.float32)
    expected_group = weights.data.astype(np.float64) * r_values[[1, 0]].astype(np.float64) * y_values.astype(np.float64)
    np.testing.assert_array_equal(output.detach().cpu().numpy(), np.array([0, 2e-30], np.float32))
    np.testing.assert_allclose(rate.grad.cpu().numpy(), expected_rate, rtol=2e-6, atol=0)
    np.testing.assert_allclose(efficacy.grad.cpu().numpy(), expected_group, rtol=2e-6, atol=0)
