"""Generated E269 checks only. CUDA tests require root's explicit opt-in.

No original E268 dataset, fitter, network or physics is imported or executed.
"""

import copy
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from tools.banc_m3_source_coefficient_engine import (
    CUDA_SOURCE,
    FILTER_ALPHA,
    FIXED_ALPHA,
    MAX_CHUNK_STEPS,
    MAX_CURRENT_ELEMENTS,
    STAGE_ORDER,
    CudaSourceCoefficientEngine,
    activation_cotangent_product,
    checked_coefficients,
    command_jacobian,
    decode_rates,
    numpy_oracle,
    prepare_problem,
)


def generated_problem(controls=4):
    """3 variable MNs, 1 fixed LTM; signed edges and shared source inputs."""
    stages = controls*80
    phase = np.arange(stages, dtype=float)
    source = np.column_stack((1.1+.2*np.sin(phase*.13), .8+.1*np.cos(phase*.19))).astype(np.float32)
    history = np.stack((source.astype(float)*.7+.1, source.astype(float)*.3+.2), axis=2)
    # Exercise history native rounding instead of retaining all f64 digits.
    history += 2.3e-8
    mask = np.ones((stages, 2), np.float32)
    mask[7::13, 1] = 0
    mask[11::17, 0] = 0
    pool = np.array([[.2, .3, .5, 0], [0, 0, 0, 1.]])
    low = np.array([[.04, .07, .08], [.06, .1, .11], [.1, .14, .2], [0, 1, 0.]])
    high = low*np.array([[1.4], [.8], [1.15], [1.]])
    bank = np.array([[.2, .3, .4], [.35, .45, .55], [.5, .6, .7], [.6, .7, .8]])
    kappa = np.array([.15, .4, .65, .7])
    orig = np.array([.9, .7])
    act = np.array([np.sum(low[:3]*bank[:3]+(high[:3]-low[:3])*np.maximum(bank[:3]-kappa[:3, None], 0)), .7])
    total = (np.array([2.4, 2.1, 2.7, 2.2])[None, :]+.2*np.sin(phase[:, None]*.17)).astype(np.float32)
    return {"stage_total": total, "source_rate": source, "source_mask": mask,
            "source_history": history, "stage_index": np.arange(stages, dtype=np.int64),
            "stage_order": STAGE_ORDER,
            "initial_rate": np.array([.6, .8, .7, .9], np.float32),
            "initial_adaptation": np.array([.05, .07, .08, .03], np.float32),
            "tau": np.array([.006, .011, .02, .009], np.float32),
            "coefficient": np.array([.18, .16, .21, .17], np.float32),
            "threshold": np.array([.2, .25, .1, .3], np.float32),
            "cap": np.array([8, 7, 9, 6], np.float32),
            "cell_gain": np.array([.1, .2, .15, .3], np.float32),
            "cell_inverse_adaptation_tau": np.array([3.1, 7.2, 12.3, 4.5], np.float32),
            "motor_ids": np.array([101, 201, 301, 401], np.int64),
            "source_ids": np.array([21, 22], np.int64),
            "edge_ids": np.array([100, 110, 120, 130, 140, 150], np.int64),
            "edge_post": np.array([0, 0, 1, 1, 2, 3], np.int64),
            "edge_source": np.array([0, 1, 0, 1, 0, 1], np.int64),
            "weights": np.array([.4, -.15, .25, .1, -.2, .3], np.float32),
            "membership": np.array([0, 0, 0, 1], np.int64), "pool": pool,
            "low": low, "high": high, "kappa": kappa,
            "fixed": np.array([False, True]), "bank0": bank,
            "activation0": act, "original_activation0": orig,
            "moment": np.array([[1., -.7], [-.3, .8], [.4, .2]]),
            "rest": np.array([.1, -.2, .3]), "ltm": np.array([1], np.int64)}


def coefficients(problem, neutral=False):
    return np.tile(np.array([1., 0., 0.] if neutral else [.9, .16, .08]),
                   (len(problem["edge_ids"]), 1))


def test_prepare_copies_and_preserves_explicit_mapping():
    raw = generated_problem()
    p = prepare_problem(raw)
    assert p["samples"] == 5 and p["params"] == 18
    assert p["gpu_bytes_estimate"] < 4*1024**3
    np.testing.assert_array_equal(p["parameter_muscle"], np.repeat([0, 0, 0, 0, 0, 1], 3))
    np.testing.assert_array_equal(p["history32"], raw["source_history"].astype(np.float32))
    raw["stage_total"].fill(99)
    assert not np.all(p["stage_total"] == 99)
    assert not p["stage_total"].flags.writeable


@pytest.mark.parametrize("field,change", [
    ("source_mask", lambda x: x.__setitem__((0, 0), 1.01)),
    ("source_mask", lambda x: x.__setitem__((0, 0), .4)),
    ("source_rate", lambda x: x.__setitem__((0, 0), -1)),
    ("source_history", lambda x: x.__setitem__((0, 0, 0), np.inf)),
    ("stage_index", lambda x: x.__setitem__(slice(None), x[::-1].copy())),
    ("cell_inverse_adaptation_tau", lambda x: x.__setitem__(0, 0)),
    ("tau", lambda x: x.__setitem__(0, 0)),
    ("weights", lambda x: x.__setitem__(0, 0)),
    ("motor_ids", lambda x: x.__setitem__(0, x[1])),
    ("source_ids", lambda x: x.__setitem__(0, x[1])),
    ("edge_ids", lambda x: x.__setitem__(0, x[1])),
    ("edge_post", lambda x: x.__setitem__(0, 4)),
    ("edge_source", lambda x: x.__setitem__(0, 2)),
    ("membership", lambda x: x.__setitem__(0, 1)),
    ("pool", lambda x: x.__setitem__((1, 0), .1)),
    ("fixed", lambda x: x.__setitem__(1, False)),
    ("low", lambda x: x.__setitem__((3, 0), .1)),
    ("activation0", lambda x: x.__setitem__(0, x[0]+.1)),
])
def test_invalid_context_rejected_before_cuda(field, change):
    raw = generated_problem()
    change(raw[field])
    with pytest.raises(ValueError):
        CudaSourceCoefficientEngine(raw)


@pytest.mark.parametrize("field", ["edge_ids", "edge_post", "membership", "stage_index"])
def test_ids_require_exact_int64(field):
    p = generated_problem()
    p[field] = p[field].astype(np.int32)
    with pytest.raises(ValueError):
        prepare_problem(p)


def test_duplicate_edges_stage_order_and_extent_rejected():
    p = generated_problem()
    p["edge_source"][1] = 0
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_problem(p)
    p = generated_problem()
    p["stage_order"] = ("start", "half2", "half1", "end")
    with pytest.raises(ValueError, match="stage_order"):
        prepare_problem(p)
    p = generated_problem()
    p["stage_total"] = p["stage_total"][:-1]
    with pytest.raises(ValueError, match="stage totals"):
        prepare_problem(p)


def test_coefficients_native_rounding_and_bounds():
    c = np.array([[.125, 0., 0.], [8., 0., 0.]])
    assert checked_coefficients(c, 2, rounded=True).dtype == np.float32
    for bad in (c.astype(np.float32), c.ravel(), c-.01, c+8):
        with pytest.raises(ValueError):
            checked_coefficients(bad, 2)
    # Valid f64 sum can leave the interval after individual f32 rounding.
    boundary = np.array([[7.9999999, .0000001, 0.]])
    with pytest.raises(ValueError, match="Rounded"):
        checked_coefficients(boundary, 1, rounded=True)


def test_initial_row_and_actual_decoder_filters():
    p = generated_problem(1)
    rates = np.stack((p["initial_rate"].astype(float), np.array([1.2, .7, 1.4, 1.1])))
    result = decode_rates(p, rates)
    np.testing.assert_array_equal(result["bank"][0], p["bank0"])
    np.testing.assert_array_equal(result["activation"][0], p["activation0"])
    expected_bank = p["bank0"]+FILTER_ALPHA*(rates[1, :, None]-p["bank0"])
    np.testing.assert_array_equal(result["bank"][1], expected_bank)
    expected_var = np.sum(p["low"][:3]*expected_bank[:3]
                          + (p["high"][:3]-p["low"][:3])*np.maximum(expected_bank[:3]-p["kappa"][:3, None], 0))
    assert result["activation"][1, 0] == pytest.approx(expected_var, abs=1e-15)
    assert result["activation"][1, 1] == pytest.approx(.7+FIXED_ALPHA*(1.1-.7))
    np.testing.assert_allclose(result["q"], p["rest"]+.02*(result["activation"]@p["moment"].T))
    np.testing.assert_allclose(result["ltm"], .2*np.tanh(result["activation"][:, [1]]/10))


def test_neutral_masks_history_rounding_inverse_tau_and_half_stages():
    p = generated_problem()
    c = coefficients(p)
    neutral = numpy_oracle(p, coefficients(p, True))
    changed = copy.deepcopy(p)
    changed["source_rate"] *= 100
    changed["source_history"] *= 100
    np.testing.assert_array_equal(neutral["rate"], numpy_oracle(changed, coefficients(p, True))["rate"])
    zero_mask = copy.deepcopy(p)
    zero_mask["source_mask"].fill(0)
    a = numpy_oracle(zero_mask, c)
    np.testing.assert_array_equal(a["rate"], numpy_oracle(zero_mask, coefficients(p, True))["rate"])
    np.testing.assert_array_equal(a["B"], 0)
    rounded = copy.deepcopy(p)
    rounded["source_history"] = rounded["source_history"].astype(np.float32).astype(float)
    np.testing.assert_array_equal(numpy_oracle(p, c)["rate"], numpy_oracle(rounded, c)["rate"])
    changed = copy.deepcopy(p)
    changed["cell_inverse_adaptation_tau"][:] = changed["cell_inverse_adaptation_tau"][0]
    assert np.max(np.abs(numpy_oracle(p, c)["rate"]-numpy_oracle(changed, c)["rate"])) > 1e-6
    changed = copy.deepcopy(p)
    for key in ("stage_total", "source_rate", "source_mask", "source_history"):
        values = changed[key].reshape((-1, 4)+changed[key].shape[1:])
        values[:, [1, 2]] = values[:, [2, 1]]
    assert np.max(np.abs(numpy_oracle(p, c)["rate"]-numpy_oracle(changed, c)["rate"])) > 1e-9


def test_cpu_analytic_tangents_all_channels_commands_and_ltm():
    p = generated_problem()
    c = coefficients(p)
    analytic = numpy_oracle(p, c)
    jac = command_jacobian(p, analytic, np.arange(5, dtype=np.int64))
    eps = 1e-5
    for index in range(c.size):
        plus, minus = c.copy(), c.copy()
        plus.flat[index] += eps
        minus.flat[index] -= eps
        rp, rm = numpy_oracle(p, plus, derivative=False), numpy_oracle(p, minus, derivative=False)
        muscle = analytic["parameter_muscle"][index]
        for key, expected in (("rate", analytic["rate_sensitivity"][:, index]),
                              ("adaptation", analytic["adaptation_sensitivity"][:, index])):
            post = p["edge_post"][index//3]
            np.testing.assert_allclose((rp[key][:, post]-rm[key][:, post])/(2*eps), expected, rtol=3e-6, atol=2e-9)
        np.testing.assert_allclose((rp["activation"][:, muscle]-rm["activation"][:, muscle])/(2*eps),
                                   analytic["B"][:, index], rtol=3e-6, atol=2e-9)
        for key in ("q", "ltm"):
            np.testing.assert_allclose((rp[key]-rm[key])/(2*eps), jac[key][:, :, index], rtol=3e-6, atol=2e-10)
    cot = np.arange(10, dtype=float).reshape(5, 2)*.1
    gradient = activation_cotangent_product(analytic, cot)
    expected = np.array([np.dot(analytic["B"][:, i], cot[:, m])
                         for i, m in enumerate(analytic["parameter_muscle"])])
    np.testing.assert_allclose(gradient, expected, rtol=1e-15, atol=1e-15)


def test_oracle_size_and_chunk_source_contract():
    assert MAX_CHUNK_STEPS == 1000
    assert "for(int step=start;step<start+count;" in CUDA_SOURCE
    assert "*(double)inv[i]" in CUDA_SOURCE
    p = generated_problem(101)
    with pytest.raises(ValueError, match="generated"):
        numpy_oracle(p, coefficients(p))
    with pytest.raises(ValueError, match="Chunk"):
        CudaSourceCoefficientEngine(generated_problem(), chunk_steps=1001)


def test_cpu_transfer_accounting_without_gpu_objects():
    """Use a fake copy provider, without constructing/importing CUDA."""
    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def synchronize(self):
            return None

    class FakeCopy:
        ndarray = np.ndarray
        @staticmethod
        def asnumpy(value, **kwargs):
            return value.copy()

        @staticmethod
        def asarray(value):
            return value.copy()

    engine = object.__new__(CudaSourceCoefficientEngine)
    from tools.banc_m3_source_coefficient_engine import _receipt
    engine.receipt = {k: _receipt() for k in ("upload", "solve", "derivative", "host_copy")}
    engine.last_receipt = copy.deepcopy(engine.receipt)
    engine.cp, engine.stream, engine.device = FakeCopy(), FakeStream(), nullcontext()
    value = np.arange(6, dtype=float)
    engine._current_arrays = {id(value): value}
    engine._active_receipt = None
    np.testing.assert_array_equal(engine.host_copy(value, label="small_q"), value)
    assert engine.receipt["host_copy"]["completed_bytes"] == 48
    assert engine.last_receipt["host_copy"]["completed_count"] == 1
    assert engine.host_copy_events[0]["label"] == "small_q"
    engine._upload(value)
    assert engine.receipt["upload"]["completed_count"] == 1
    assert engine.last_receipt["upload"]["completed_bytes"] == 48
    newer = np.arange(6, dtype=float)
    engine._current_arrays = {id(newer): newer}
    old_count = engine.last_receipt["host_copy"]["requested_count"]
    with pytest.raises(ValueError, match="current response"):
        engine.host_copy(value, label="stale")
    assert engine.last_receipt["host_copy"]["requested_count"] == old_count


def test_rectification_boundary_chooses_zero_derivative():
    p = generated_problem(1)
    p["stage_total"].fill(0)
    p["threshold"].fill(0)
    p["initial_rate"].fill(0)
    p["initial_adaptation"].fill(0)
    result = numpy_oracle(p, coefficients(p, True))
    np.testing.assert_array_equal(result["phi"], 0)
    np.testing.assert_array_equal(result["B"], 0)
    np.testing.assert_array_equal(result["rate"], 0)


def generated_fake_engine(*, fault=None):
    """NumPy-backed bookkeeping harness; performs no CUDA/kernel integration."""
    from tools.banc_m3_source_coefficient_engine import _current_receipt, _receipt

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def synchronize(self):
            return None

    class FakeCopy:
        ndarray = np.ndarray
        float64, uint8, nan = np.float64, np.uint8, np.nan
        zeros, stack, tanh, sum = staticmethod(np.zeros), staticmethod(np.stack), staticmethod(np.tanh), staticmethod(np.sum)

        @staticmethod
        def full(*args):
            if fault == "allocation":
                raise RuntimeError("generated allocation failure")
            return np.full(*args)

        @staticmethod
        def asnumpy(value, **kwargs):
            return value.copy()

        @staticmethod
        def asarray(value):
            return value.copy()

    def kernel(name):
        def call(*args):
            if name == fault:
                raise RuntimeError("generated "+name+" failure")
        return call

    engine = object.__new__(CudaSourceCoefficientEngine)
    engine.problem = prepare_problem(generated_problem(3))
    engine.device_data = {k: v for k, v in engine.problem.items() if isinstance(v, np.ndarray)}
    engine.receipt = {k: _receipt() for k in ("upload", "solve", "derivative", "host_copy")}
    engine.receipt["products"] = {k: _receipt() for k in ("cotangent", "command_jacobian")}
    engine.receipt["current"] = _current_receipt()
    engine.forward_implementation = "reference"
    engine.cp, engine.stream, engine.device = FakeCopy(), FakeStream(), nullcontext()
    engine.chunk_steps = 20
    engine.kernels = {k: kernel(k) for k in ("forward", "current_precompute", "forward_precomputed",
                                          "bank_decode", "activation_decode", "tangent")}
    engine._current_response, engine._current_arrays, engine._active_receipt = None, {}, None
    return engine


@pytest.mark.parametrize("fault,forward_samples", [
    ("allocation", 0), ("forward", 0), ("bank_decode", 0),
    ("activation_decode", 0), ("tangent", 2),
])
def test_cpu_generated_first_chunk_failure_prefix(fault, forward_samples):
    engine = generated_fake_engine(fault=fault)
    with pytest.raises(RuntimeError, match="generated") as failure:
        engine.solve(coefficients(engine.problem))
    receipt = failure.value.response_receipt
    assert receipt["completed_samples"] == forward_samples
    assert receipt["completed_steps"] == (20 if forward_samples else 0)
    assert receipt["derivative_completed_samples"] == 0
    assert receipt["derivative_completed_steps"] == 0
    assert receipt["status"] == "failed"
    if fault == "forward":
        assert np.isnan(failure.value.partial_response["rate"][0]).all()


@pytest.mark.parametrize("fault", ["allocation", "bank_decode", "activation_decode"])
def test_cpu_generated_first_recorded_decoder_failure_prefix(fault):
    engine = generated_fake_engine(fault=fault)
    rates = np.tile(engine.problem["initial_rate"], (4, 1))
    with pytest.raises(RuntimeError, match="generated") as failure:
        engine.decode_rates(rates)
    assert failure.value.response_receipt["completed_samples"] == 0
    assert failure.value.response_receipt["completed_steps"] == 0
    assert failure.value.response_receipt["derivative_completed_samples"] == 0


def fake_bound_result(engine):
    result = numpy_oracle(generated_problem(3), coefficients(engine.problem))
    from tools.banc_m3_source_coefficient_engine import _receipt
    receipt = {k: _receipt() for k in ("upload", "solve", "derivative", "host_copy")}
    receipt.update(status="complete", completed_samples=4, derivative_completed_samples=4)
    engine._start_response(receipt)
    result["receipt"] = receipt
    engine._bind_response(result, receipt)
    return result


def test_cpu_generated_products_meter_current_owner_and_reject_stale_result():
    engine = generated_fake_engine()
    result = fake_bound_result(engine)
    cot = np.ones((4, 2), dtype=float)
    expected = activation_cotangent_product(result, cot)
    np.testing.assert_allclose(engine.activation_cotangent_product(result, cot), expected)
    assert result["receipt"]["products"]["cotangent"]["completed_count"] == 1
    assert result["receipt"]["products"]["cotangent"]["completed_bytes"] == 18*8
    assert engine.receipt["products"]["cotangent"]["completed_count"] == 1
    idx = np.array([0, 1, 3], np.int64)
    expected_jac = command_jacobian(engine.problem, result, idx)
    actual = engine.command_jacobian(result, idx)
    for key in expected_jac:
        np.testing.assert_allclose(actual[key], expected_jac[key])
    assert result["receipt"]["products"]["command_jacobian"]["completed_count"] == 1
    original_receipt = copy.deepcopy(result["receipt"])
    newer = fake_bound_result(engine)
    with pytest.raises(ValueError, match="current response") as failure:
        engine.activation_cotangent_product(result, cot)
    assert failure.value.response_receipt["requested_count"] == 0
    assert newer["receipt"]["products"]["cotangent"]["requested_count"] == 0
    assert result["receipt"] == original_receipt
    with pytest.raises(ValueError, match="current response"):
        engine.host_copy(result["q"], label="stale_q")
    assert newer["receipt"]["host_copy"]["requested_count"] == 0


@pytest.mark.parametrize("product,fault", [
    ("cotangent", "sum"), ("command_jacobian", "tanh"), ("cotangent", "asnumpy"),
])
def test_cpu_generated_product_primary_failure_receipt_and_partial(product, fault):
    engine = generated_fake_engine()
    result = fake_bound_result(engine)
    primary = RuntimeError("generated product primary failure")

    def fail(*args, **kwargs):
        raise primary

    setattr(engine.cp, fault, fail)
    with pytest.raises(RuntimeError, match="generated product primary failure") as failure:
        if product == "cotangent":
            engine.activation_cotangent_product(result, np.ones((4, 2)))
        else:
            engine.command_jacobian(result, np.array([0, 2], np.int64))
    assert failure.value is primary
    receipt = failure.value.response_receipt
    assert receipt["operation"] == product and receipt["status"] == "failed"
    assert receipt["requested_count"] == 1
    assert receipt["completed_count"] == (1 if fault == "asnumpy" else 0)
    assert receipt["wall_s"] >= 0
    assert receipt["input_response_receipt"]["upload"]["completed_count"] == 1
    assert "input_B" in failure.value.partial_response
    if fault == "tanh":
        assert "q" in failure.value.partial_response
    if fault == "asnumpy":
        assert "gradient" in failure.value.partial_response
        assert receipt["input_response_receipt"]["host_copy"]["requested_count"] == 1
        assert receipt["input_response_receipt"]["host_copy"]["completed_count"] == 0
    assert result["receipt"]["products"][product]["requested_count"] == 1
    assert engine.receipt["products"][product]["requested_count"] == 1


def test_cpu_generated_product_memory_guard_precedes_allocation():
    from tools.banc_m3_source_coefficient_engine import MAX_GPU_BYTES
    engine = generated_fake_engine()
    result = fake_bound_result(engine)
    engine._product_bytes = MAX_GPU_BYTES
    with pytest.raises(ValueError, match="4 GiB") as failure:
        engine.command_jacobian(result, np.array([0, 1], np.int64))
    assert failure.value.response_receipt["requested_count"] == 0
    assert result["receipt"]["upload"]["requested_count"] == 0
    assert result["receipt"]["products"]["command_jacobian"]["requested_count"] == 0


def generated_benchmark_problem(controls=1200, seed=27320260927):
    """Frozen E273 representative shape; synthetic inputs and existing edges."""
    rng = np.random.default_rng(seed)
    n, sources, edges, muscles, commands = 66, 51, 300, 16, 7
    stages = controls*80
    phase = np.arange(stages, dtype=float)[:, None]
    source = (1.2+.15*np.sin(phase*.003+np.arange(sources)[None, :]*.11)).astype(np.float32)
    history = np.stack((source.astype(float)*.7+.15, source.astype(float)*.4+.25), 2)
    history += 2.3e-8
    mask = np.ones_like(source)
    mask[::17, ::3] = 0
    membership = np.arange(n, dtype=np.int64) % muscles
    counts = np.bincount(membership, minlength=muscles)
    pool = np.zeros((muscles, n), dtype=float)
    pool[membership, np.arange(n)] = 1/counts[membership]
    factor = pool[membership, np.arange(n)]
    low = factor[:, None]*np.array([.2, .35, .45])
    high = low*np.array([1.1, 1.05, .95])
    fixed = np.zeros(muscles, dtype=bool)
    fixed[-1] = True
    fixed_mn = membership == muscles-1
    low[fixed_mn] = 0
    low[fixed_mn, 1] = factor[fixed_mn]
    high[fixed_mn] = low[fixed_mn]
    bank = .7+rng.random((n, 3))*.2
    kappa = .5+rng.random(n)*.1
    initial_rate = (.8+rng.random(n)*.2).astype(np.float32)
    orig = pool@initial_rate.astype(float)
    act = np.bincount(membership, weights=np.sum(low*bank+(high-low)*np.maximum(bank-kappa[:, None], 0), 1),
                      minlength=muscles)
    act[fixed] = orig[fixed]
    pairs = rng.choice(n*sources, size=edges, replace=False).astype(np.int64)
    weights = (rng.uniform(.02, .08, edges)*rng.choice([-1, 1], edges)).astype(np.float32)
    return {"stage_total": (2.4+.2*np.sin(phase*.005+np.arange(n)[None, :]*.07)).astype(np.float32),
            "source_rate": source, "source_mask": mask, "source_history": history,
            "stage_index": np.arange(stages, dtype=np.int64), "stage_order": STAGE_ORDER,
            "initial_rate": initial_rate,
            "initial_adaptation": rng.uniform(.03, .1, n).astype(np.float32),
            "tau": rng.uniform(.01, .03, n).astype(np.float32),
            "coefficient": rng.uniform(.15, .22, n).astype(np.float32),
            "threshold": rng.uniform(.1, .3, n).astype(np.float32),
            "cap": rng.uniform(6, 9, n).astype(np.float32),
            "cell_gain": rng.uniform(.1, .3, n).astype(np.float32),
            "cell_inverse_adaptation_tau": rng.uniform(3, 11, n).astype(np.float32),
            "motor_ids": np.arange(1001, 1001+n, dtype=np.int64),
            "source_ids": np.arange(101, 101+sources, dtype=np.int64),
            "edge_ids": np.arange(10001, 10001+edges, dtype=np.int64),
            "edge_post": pairs//sources, "edge_source": pairs%sources, "weights": weights,
            "membership": membership, "pool": pool, "low": low, "high": high,
            "kappa": kappa, "fixed": fixed, "bank0": bank, "activation0": act,
            "original_activation0": orig, "moment": rng.uniform(-1, 1, (commands, muscles)),
            "rest": rng.uniform(-.3, .3, commands), "ltm": np.array([muscles-1], np.int64)}


def literal_stage_current(problem, c):
    """Generated scalar reference sums, in the supplied original edge order."""
    result = problem["stage_total"].astype(float)
    history32 = problem["source_history"].astype(np.float32)
    for stage in range(len(result)):
        for edge, post in enumerate(problem["edge_post"]):
            source = problem["edge_source"][edge]
            u = ((float(c[edge, 0])-1)*float(problem["source_rate"][stage, source])
                 + float(c[edge, 1])*float(history32[stage, source, 0]))
            u += float(c[edge, 2])*float(history32[stage, source, 1])
            result[stage, post] += (float(problem["weights"][edge])
                                    * float(problem["source_mask"][stage, source]))*u
    return result


def test_cpu_e273_fixture_reference_preservation_and_memory_bound(monkeypatch):
    import tools.banc_m3_source_coefficient_engine as module
    p = generated_benchmark_problem(controls=2)
    prepared = prepare_problem(p)
    assert (prepared["neurons"], prepared["sources"], prepared["edges"], prepared["muscles"],
            prepared["commands"], p["ltm"].size) == (66, 51, 300, 16, 7, 1)
    assert prepared["current_bytes"] == 8*160*66
    archive = Path(__file__).resolve().parents[1]/"verification/banc-natural-gait-m3-20260919/e270-original-reviewed-sources/tools/banc_m3_source_coefficient_engine.py"
    text = archive.read_text(encoding="utf-8")
    start = text.index("extern \"C\" __global__ void forward(")
    end = text.index("extern \"C\" __global__ void bank_decode", start)
    assert CUDA_SOURCE.startswith("\n"+text[start:end])
    assert "int i=blockIdx.x;if(i>=n)return;" in CUDA_SOURCE
    with pytest.raises(ValueError, match="implementation"):
        CudaSourceCoefficientEngine(p, forward_implementation="automatic")
    monkeypatch.setattr(module, "MAX_GPU_BYTES", prepared["gpu_bytes_estimate"]-1)
    with pytest.raises(ValueError, match="4 GiB"):
        prepare_problem(p)


@pytest.mark.parametrize("failure_launch", [1, 2, None])
def test_cpu_e273_current_prefix_requests_and_zero_neural_before_ready(failure_launch):
    engine = generated_fake_engine()
    engine.problem = prepare_problem(generated_problem(206))
    engine.device_data = {k: v for k, v in engine.problem.items() if isinstance(v, np.ndarray)}
    engine.forward_implementation = "precomputed"
    launches = []
    primary = RuntimeError("generated current launch failure")

    def current(grid, block, args):
        first, count = int(args[-2]), int(args[-1])
        assert 0 < count <= MAX_CURRENT_ELEMENTS
        assert engine.last_receipt["completed_steps"] == 0
        assert engine.last_receipt["completed_samples"] == 0
        launches.append((first, count))
        if len(launches) == failure_launch:
            raise primary
        args[9].ravel()[first:first+count] = 12.34

    engine.kernels["current_precompute"] = current
    if failure_launch is None:
        result = engine.solve(coefficients(engine.problem), derivative=False)
        receipt = result["receipt"]
        assert receipt["current"]["requested_count"] == receipt["current"]["completed_count"] == 2
        assert receipt["current"]["completed_elements"] == 206*80*4
        assert receipt["completed_current_elements"] == 206*80*4
        assert receipt["pending_current_range"] is None
        assert receipt["completed_steps"] == 206*20
    else:
        with pytest.raises(RuntimeError, match="current launch failure") as failure:
            engine.solve(coefficients(engine.problem), derivative=False)
        assert failure.value is primary
        receipt = failure.value.response_receipt
        prefix = 0 if failure_launch == 1 else MAX_CURRENT_ELEMENTS
        assert receipt["completed_steps"] == receipt["completed_samples"] == 0
        assert receipt["completed_current_elements"] == prefix
        assert receipt["current"]["completed_elements"] == prefix
        assert receipt["current"]["completed_bytes"] == prefix*8
        assert receipt["current"]["requested_count"] == failure_launch
        assert receipt["current"]["completed_count"] == failure_launch-1
        assert receipt["pending_current_range"] == [prefix, launches[-1][0]+launches[-1][1]]
        raw = failure.value.partial_response["current"].ravel()
        np.testing.assert_array_equal(raw[:prefix], 12.34)
        assert np.isnan(raw[prefix:]).all()
    assert engine.receipt["current"] == engine.last_receipt["current"]


def test_cpu_e273_reference_zero_precompute_and_new_neural_first_fault():
    reference = generated_fake_engine()
    result = reference.solve(coefficients(reference.problem), derivative=False)
    assert "current" not in result
    assert result["receipt"]["current"]["requested_elements"] == 0
    assert result["receipt"]["current"]["completed_count"] == 0
    assert result["receipt"]["planned_current_elements"] == 0
    engine = generated_fake_engine(fault="forward_precomputed")
    engine.forward_implementation = "precomputed"
    with pytest.raises(RuntimeError, match="forward_precomputed") as failure:
        engine.solve(coefficients(engine.problem))
    receipt = failure.value.response_receipt
    assert receipt["completed_current_elements"] == 3*80*4
    assert receipt["completed_steps"] == receipt["completed_samples"] == 0


CUDA_ALLOWED = os.environ.get("E269_RUN_GENERATED_CUDA") == "1"


@pytest.mark.skipif(not CUDA_ALLOWED, reason="Root must reserve generated CUDA validation explicitly")
@pytest.mark.parametrize("neutral", [False, True])
def test_generated_cuda_oracle_and_chunk_boundaries(neutral):
    p = generated_problem(51)  # Cross the 1000-step boundary.
    c = coefficients(p, neutral)
    reference = numpy_oracle(p, c)
    engine = CudaSourceCoefficientEngine(p)
    actual = engine.solve(c, host=True)
    for key in ("rate", "adaptation", "phi", "bank", "activation", "original_activation",
                "q", "ltm", "B", "rate_sensitivity", "adaptation_sensitivity"):
        np.testing.assert_allclose(actual[key], reference[key], rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(actual["branch"].astype(bool), reference["branch"])
    assert actual["receipt"]["solve"]["completed_count"] == 2
    assert actual["receipt"]["completed_steps"] == 1020
    smaller = CudaSourceCoefficientEngine(p, chunk_steps=60).solve(c, host=True)
    for key in ("rate", "bank", "activation", "B", "q", "ltm"):
        np.testing.assert_array_equal(actual[key], smaller[key])


@pytest.mark.skipif(not CUDA_ALLOWED, reason="Root must reserve generated CUDA validation explicitly")
def test_generated_cuda_tangent_and_device_products():
    p = generated_problem()
    c = coefficients(p)
    engine = CudaSourceCoefficientEngine(p)
    device = engine.solve(c)
    actual = {k: engine.host_copy(v) for k, v in device.items() if k != "receipt"}
    oracle = numpy_oracle(p, c)
    cot = np.arange(10, dtype=float).reshape(5, 2)*.1
    np.testing.assert_allclose(engine.activation_cotangent_product(device, cot),
                               activation_cotangent_product(oracle, cot), rtol=2e-12, atol=2e-12)
    samples = np.array([0, 1, 4], np.int64)
    dj = engine.command_jacobian(device, samples)
    cj = command_jacobian(p, oracle, samples)
    for key in dj:
        np.testing.assert_allclose(dj[key], cj[key], rtol=2e-12, atol=2e-12)
    eps = 1e-5
    for index in range(c.size):
        plus, minus = c.copy(), c.copy()
        plus.flat[index] += eps
        minus.flat[index] -= eps
        rp = engine.solve(plus, derivative=False, host=True)
        rm = engine.solve(minus, derivative=False, host=True)
        m = actual["parameter_muscle"][index]
        np.testing.assert_allclose((rp["activation"][:, m]-rm["activation"][:, m])/(2*eps),
                                   actual["B"][:, index], rtol=3e-6, atol=2e-9)
        for key in ("q", "ltm"):
            np.testing.assert_allclose((rp[key][samples]-rm[key][samples])/(2*eps),
                                       dj[key][:, :, index], rtol=3e-6, atol=2e-10)
    p["source_mask"].fill(0)
    blocked = CudaSourceCoefficientEngine(p).solve(c, host=True)
    np.testing.assert_array_equal(blocked["B"], 0)


@pytest.mark.skipif(not CUDA_ALLOWED, reason="Root must reserve generated CUDA validation explicitly")
def test_generated_cuda_recorded_decoder_and_metered_selected_copies():
    p = generated_problem(51)
    rates = np.tile(p["initial_rate"], (52, 1)).astype(float)
    rates[1:] += np.arange(1, 52)[:, None]*.001
    engine = CudaSourceCoefficientEngine(p)
    device = engine.decode_rates(rates)
    reference = decode_rates(p, rates)
    for key in reference:
        actual = engine.host_copy(device[key], label="recorded_"+key)
        np.testing.assert_allclose(actual, reference[key], rtol=2e-12, atol=2e-12)
    assert device["receipt"]["completed_steps"] == 0
    assert device["receipt"]["completed_samples"] == 52
    assert device["receipt"]["host_copy"]["completed_count"] == 5
    assert engine.receipt["host_copy"]["completed_count"] == 5
    assert device["receipt"]["solve"]["completed_count"] == 2


@pytest.mark.skipif(not CUDA_ALLOWED, reason="Root must reserve generated CUDA validation explicitly")
def test_generated_cuda_failure_retains_synchronized_prefix():
    p = generated_problem(3)
    engine = CudaSourceCoefficientEngine(p, chunk_steps=20)
    original = engine.kernels["tangent"]
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("generated launch fault")
        return original(*args, **kwargs)

    engine.kernels["tangent"] = fail_second
    with pytest.raises(RuntimeError, match="generated launch fault") as failure:
        engine.solve(coefficients(p))
    receipt = failure.value.response_receipt
    assert receipt["completed_steps"] == 40
    assert receipt["derivative_completed_steps"] == 20
    assert receipt["derivative"]["requested_count"] == 2
    assert receipt["derivative"]["completed_count"] == 1
    assert "B" in failure.value.partial_response


@pytest.mark.skipif(not CUDA_ALLOWED, reason="Root must reserve generated CUDA validation explicitly")
@pytest.mark.parametrize("fault", ["allocation", "forward", "tangent", "bank_decode", "recorded_decoder"])
def test_generated_cuda_first_failure_prefix(fault, monkeypatch):
    p = generated_problem(3)
    engine = CudaSourceCoefficientEngine(p, chunk_steps=20, forward_implementation="reference")
    primary = RuntimeError("generated first "+fault+" failure")

    def fail(*args, **kwargs):
        raise primary

    if fault == "allocation":
        monkeypatch.setattr(engine.cp, "full", fail)
    else:
        engine.kernels["bank_decode" if fault == "recorded_decoder" else fault] = fail
    with pytest.raises(RuntimeError, match="generated first") as failure:
        if fault == "recorded_decoder":
            engine.decode_rates(np.tile(p["initial_rate"], (4, 1)))
        else:
            engine.solve(coefficients(p))
    assert failure.value is primary
    receipt = failure.value.response_receipt
    assert receipt["completed_samples"] == (2 if fault == "tangent" else 0)
    assert receipt["derivative_completed_samples"] == 0
    assert receipt["completed_steps"] == (20 if fault == "tangent" else 0)
    assert receipt["status"] == "failed"


@pytest.mark.skipif(not CUDA_ALLOWED, reason="Root must reserve generated CUDA validation explicitly")
@pytest.mark.parametrize("product,fault", [("cotangent", "sum"), ("command_jacobian", "tanh")])
def test_generated_cuda_product_failure_and_stale_ownership(product, fault, monkeypatch):
    p = generated_problem(3)
    engine = CudaSourceCoefficientEngine(p)
    old = engine.solve(coefficients(p))
    current = engine.solve(coefficients(p))
    with pytest.raises(ValueError, match="current response"):
        engine.command_jacobian(old, np.array([0, 1], np.int64))
    with pytest.raises(ValueError, match="current response"):
        engine.host_copy(old["q"], label="stale")
    assert current["receipt"]["host_copy"]["requested_count"] == 0
    assert current["receipt"]["products"]["command_jacobian"]["requested_count"] == 0
    primary = RuntimeError("generated product primary failure")

    def fail(*args, **kwargs):
        raise primary

    monkeypatch.setattr(engine.cp, fault, fail)
    with pytest.raises(RuntimeError, match="generated product primary failure") as failure:
        if product == "cotangent":
            engine.activation_cotangent_product(current, np.ones((4, 2)))
        else:
            engine.command_jacobian(current, np.array([0, 1], np.int64))
    assert failure.value is primary
    assert failure.value.response_receipt["requested_count"] == 1
    assert failure.value.response_receipt["completed_count"] == 0
    assert "input_B" in failure.value.partial_response
    if product == "command_jacobian":
        assert "q" in failure.value.partial_response


def assert_numpy_bit_exact(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    np.testing.assert_array_equal(left.view(np.uint8), right.view(np.uint8))


@pytest.mark.skipif(not CUDA_ALLOWED, reason="Root must reserve generated CUDA validation explicitly")
@pytest.mark.parametrize("case", ["neutral", "nonneutral", "blocked", "rectified"])
def test_e273_generated_cuda_bit_exact_reference_current_and_tangents(case):
    p = generated_problem(51)
    c = coefficients(p, neutral=case == "neutral")
    if case == "blocked":
        p["source_mask"].fill(0)
    if case == "rectified":
        p["stage_total"][:, 2] = -.3
    reference = CudaSourceCoefficientEngine(p, forward_implementation="reference").solve(c, host=True)
    engine = CudaSourceCoefficientEngine(p, forward_implementation="precomputed")
    actual = engine.solve(c, host=True)
    assert reference["receipt"]["current"]["requested_count"] == 0
    for key in reference:
        if key != "receipt":
            assert_numpy_bit_exact(actual[key], reference[key])
    assert_numpy_bit_exact(actual["current"], literal_stage_current(p, c))
    receipt = actual["receipt"]
    assert receipt["current"]["requested_elements"] == receipt["current"]["completed_elements"] == 51*80*4
    assert receipt["current"]["requested_bytes"] == receipt["current"]["completed_bytes"] == 51*80*4*8
    assert receipt["current"]["completed_count"] == 1
    assert receipt["pending_current_range"] is None
    assert engine.receipt["current"] == receipt["current"]


@pytest.mark.skipif(not CUDA_ALLOWED, reason="Root must reserve generated CUDA validation explicitly")
@pytest.mark.parametrize("failure_launch", [1, 2])
def test_e273_generated_cuda_current_launch_failure_prefix(failure_launch):
    p = generated_problem(206)
    engine = CudaSourceCoefficientEngine(p)
    original, calls = engine.kernels["current_precompute"], 0
    primary = RuntimeError("generated current launch failure")

    def fail_selected(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_launch:
            raise primary
        return original(*args, **kwargs)

    engine.kernels["current_precompute"] = fail_selected
    with pytest.raises(RuntimeError, match="current launch failure") as failure:
        engine.solve(coefficients(p))
    assert failure.value is primary
    receipt = failure.value.response_receipt
    prefix = 0 if failure_launch == 1 else MAX_CURRENT_ELEMENTS
    assert receipt["completed_current_elements"] == prefix
    assert receipt["completed_steps"] == receipt["completed_samples"] == 0
    assert receipt["pending_current_range"] == [prefix, min(prefix+MAX_CURRENT_ELEMENTS, 206*80*4)]
    current = engine.host_copy(failure.value.partial_response["current"], label="partial_current").ravel()
    assert np.isfinite(current[:prefix]).all()
    assert np.isnan(current[prefix:]).all()


@pytest.mark.skipif(not CUDA_ALLOWED, reason="Root must reserve generated CUDA validation explicitly")
def test_e273_generated_cuda_neural_first_failure_after_current_ready():
    p = generated_problem()
    engine = CudaSourceCoefficientEngine(p)
    primary = RuntimeError("generated precomputed forward failure")

    def fail(*args, **kwargs):
        raise primary

    engine.kernels["forward_precomputed"] = fail
    with pytest.raises(RuntimeError, match="precomputed forward failure") as failure:
        engine.solve(coefficients(p))
    assert failure.value is primary
    receipt = failure.value.response_receipt
    assert receipt["completed_current_elements"] == 4*80*4
    assert receipt["completed_steps"] == receipt["completed_samples"] == 0
    current = engine.host_copy(failure.value.partial_response["current"], label="prepared_current")
    assert_numpy_bit_exact(current, literal_stage_current(p, coefficients(p)))


def run_generated_e273_benchmark(*, output_path=None):
    """Prospective v2: 1 constructor, 2 warmups, 6 timed solves, no optimizer.

    One engine changes only its forward implementation, retaining identical
    device input pointers. Solve order is reference/precomputed repeated four
    times. After each timer ends, nine scientific arrays are copied to CPU and
    the local device-result reference is removed. CPU dtype/shape/raw-byte
    comparison, host-copy time and GPU memory snapshots are outside the timer.
    A fresh output path preserves the failed FIRST benchmark as separate data.
    """
    if os.environ.get("E273_RUN_GENERATED_CUDA_BENCHMARK") != "1":
        raise RuntimeError("Root must reserve the E273 generated CUDA benchmark explicitly")
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError("Prospective benchmark output already exists: "+str(output_path))
    p = generated_benchmark_problem()
    c = coefficients(p)
    names = ("reference", "precomputed")
    keys = ("rate", "adaptation", "phi", "branch", "bank", "activation",
            "original_activation", "q", "ltm")
    report = {"benchmark_version": "prospective-v2-memory-matched", "fixture_seed": 27320260927,
              "shape": {"neurons": 66, "sources": 51,
              "edges": 300, "muscles": 16, "commands": 7, "ltm": 1, "controls": 1200},
              "derivative": False, "planned_constructor_count": 1,
              "planned_warmup_solves": 2, "planned_timed_solves": 6,
              "planned_host_copy_count": 72, "planned_cpu_array_comparisons": 36,
              "scientific_keys": list(keys), "input_pointer_stable": True,
              "minimum_median_speedup": 1.5, "construction": {}, "warmup": [], "measurements": [],
              "solve_attempts": []}
    engine = None

    def persist():
        if output_path is not None:
            Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def compare(results):
        left, right = results["reference"], results["precomputed"]
        assert tuple(left) == tuple(right) == keys
        return {key: left[key].dtype == right[key].dtype and left[key].shape == right[key].shape
                and left[key].tobytes(order="C") == right[key].tobytes(order="C") for key in keys}

    def pointers():
        return {key: int(value.data.ptr) for key, value in engine.device_data.items()}

    def memory():
        with engine.device:
            pool = engine.cp.get_default_memory_pool()
            free, total = engine.cp.cuda.runtime.memGetInfo()
            return {"pool_used_bytes": int(pool.used_bytes()), "pool_total_bytes": int(pool.total_bytes()),
                    "device_free_bytes": int(free), "device_total_bytes": int(total)}

    def run_one(implementation):
        engine.forward_implementation = implementation
        before = pointers()
        row = {"implementation": implementation, "input_pointers_before": before, "memory_before": memory()}
        report["solve_attempts"].append(row)
        row["input_pointer_same_before"] = before == report["input_pointers"]
        report["input_pointer_stable"] &= row["input_pointer_same_before"]
        assert row["input_pointer_same_before"], "Device input pointers changed before solve"
        began = time.perf_counter()
        try:
            device_result = engine.solve(c, derivative=False, host=False)
        except Exception as error:
            row["status"] = "solve_failed"
            row["receipt"] = copy.deepcopy(getattr(error, "response_receipt", {}))
            raise
        finally:
            row["wall_s"] = time.perf_counter()-began
        try:
            row["memory_after_solve"] = memory()
            after = pointers()
            row["input_pointers_after"] = after
            row["input_pointer_same"] = after == before == report["input_pointers"]
            report["input_pointer_stable"] &= row["input_pointer_same"]
            assert row["input_pointer_same"], "Device input pointers changed during solve"
            began = time.perf_counter()
            try:
                cpu_result = {key: engine.host_copy(device_result[key], label="benchmark-v2:"+implementation+":"+key)
                              for key in keys}
            finally:
                row["host_copy_wall_s"] = time.perf_counter()-began
            row["host_copy_count"] = len(cpu_result)
            row["host_copy_bytes"] = sum(value.nbytes for value in cpu_result.values())
            row["receipt"] = copy.deepcopy(device_result["receipt"])
            row["status"] = "complete"
        except Exception:
            row["status"] = "post_solve_failed"
            row["receipt"] = copy.deepcopy(device_result["receipt"])
            raise
        finally:
            # The engine's current response persists only until the next
            # _start_response. No caller retains a previous large GPU result.
            del device_result
        row["memory_after_host_copy"] = memory()
        return cpu_result, row

    try:
        began = time.perf_counter()
        engine = CudaSourceCoefficientEngine(p, forward_implementation="reference")
        wall = time.perf_counter()-began
        report["construction"] = {"constructor_wall_s": wall, "jit_wall_s": engine.jit_wall_s,
            "constructor_without_jit_wall_s": wall-engine.jit_wall_s, "receipt": copy.deepcopy(engine.receipt),
            "memory_after_construction": memory()}
        report["input_pointers"] = pointers()
        cp = engine.cp
        props = cp.cuda.runtime.getDeviceProperties(engine.device.id)
        name = props["name"]
        report["environment"] = {"device_id": engine.device.id,
            "device_name": name.decode() if isinstance(name, bytes) else str(name),
            "cupy_version": cp.__version__, "numpy_version": np.__version__,
            "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion()}
        warm = {}
        for name in names:
            warm[name], row = run_one(name)
            report["warmup"].append(row)
            persist()
        began = time.perf_counter()
        report["warmup_bit_exact"] = compare(warm)
        report["warmup_cpu_compare_wall_s"] = time.perf_counter()-began
        del warm
        for iteration in range(3):
            results, rows = {}, {}
            for name in names:
                results[name], rows[name] = run_one(name)
            began = time.perf_counter()
            checks = compare(results)
            comparison_wall = time.perf_counter()-began
            report["measurements"].append({"iteration": iteration, "order": list(names),
                "wall_s": {name: rows[name]["wall_s"] for name in names}, "bit_exact": checks,
                "host_copy_wall_s": {name: rows[name]["host_copy_wall_s"] for name in names},
                "cpu_compare_wall_s": comparison_wall, "runs": rows,
                "receipts": {name: rows[name]["receipt"] for name in names}})
            del results
            persist()
        median = {name: float(np.median([row["wall_s"][name] for row in report["measurements"]]))
                  for name in names}
        report["median_wall_s"] = median
        report["median_speedup"] = median["reference"]/median["precomputed"]
        report["bit_exact"] = all(report["warmup_bit_exact"].values()) and all(
            all(row["bit_exact"].values()) for row in report["measurements"])
        report["passed"] = report["bit_exact"] and report["median_speedup"] >= 1.5
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        error.benchmark_receipt = copy.deepcopy(report)
        persist()
        raise
    persist()
    return report


@pytest.mark.skipif(os.environ.get("E273_RUN_GENERATED_CUDA_BENCHMARK") != "1",
                    reason="Root must reserve the separate representative CUDA benchmark")
def test_e273_generated_cuda_representative_benchmark(tmp_path):
    report = run_generated_e273_benchmark(output_path=tmp_path/"e273-generated-benchmark.json")
    assert report["bit_exact"]
    assert report["median_speedup"] >= report["minimum_median_speedup"], report


def benchmark_fake_constructor(events, *, mutate_pointer=False):
    """CPU-only harness proving benchmark counts and device-result lifetimes."""
    import weakref
    from types import SimpleNamespace

    class Response(dict):
        __slots__ = ("__weakref__",)

    class Device:
        id = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class Engine:
        def __init__(self, mapping, *, forward_implementation):
            events.append(("constructor", forward_implementation))
            self.forward_implementation = forward_implementation
            self.device_data = {key: SimpleNamespace(data=SimpleNamespace(ptr=100+index))
                                for index, key in enumerate(("stage_total", "source_rate"))}
            self.device = Device()
            self.jit_wall_s = 0.0
            self.receipt = {"host_copy": {"completed_count": 0}}
            self.current, self.previous = None, None
            self.solves = 0
            pool = SimpleNamespace(used_bytes=lambda: 100, total_bytes=lambda: 200)
            runtime = SimpleNamespace(memGetInfo=lambda: (700, 1000),
                getDeviceProperties=lambda device: {"name": b"CPU fake: no GPU"}, runtimeGetVersion=lambda: 0)
            self.cp = SimpleNamespace(get_default_memory_pool=lambda: pool,
                                      cuda=SimpleNamespace(runtime=runtime), __version__="CPU fake")

        def solve(self, c, *, derivative, host):
            assert derivative is False and host is False
            self.current = None  # Simulate actual _start_response releasing old output.
            if self.previous is not None:
                assert self.previous() is None, "Caller retained the previous large GPU response"
            self.solves += 1
            events.append(("solve", self.forward_implementation))
            if mutate_pointer and self.solves == 2:
                self.device_data["stage_total"].data.ptr += 1
            keys = ("rate", "adaptation", "phi", "branch", "bank", "activation",
                    "original_activation", "q", "ltm")
            self.current = Response({key: np.array([[0., -0.]]) for key in keys})
            self.current["branch"] = np.array([[0, 1]], np.uint8)
            self.current["receipt"] = {"status": "complete", "host_copy": {"completed_count": 0}}
            self.previous = weakref.ref(self.current)
            return self.current

        def host_copy(self, value, *, label):
            events.append(("host_copy", label))
            self.current["receipt"]["host_copy"]["completed_count"] += 1
            self.receipt["host_copy"]["completed_count"] += 1
            return value.copy()

    return Engine


def test_cpu_e273_v2_benchmark_one_engine_same_pointers_cpu_comparison(monkeypatch, tmp_path):
    import sys
    module = sys.modules[__name__]
    events = []
    small = generated_benchmark_problem(controls=2)
    monkeypatch.setenv("E273_RUN_GENERATED_CUDA_BENCHMARK", "1")
    monkeypatch.setattr(module, "generated_benchmark_problem", lambda: small)
    monkeypatch.setattr(module, "CudaSourceCoefficientEngine", benchmark_fake_constructor(events))
    path = tmp_path/"prospective-v2-cpu-harness.json"
    report = run_generated_e273_benchmark(output_path=path)
    assert len([event for event in events if event[0] == "constructor"]) == 1
    assert [event[1] for event in events if event[0] == "solve"] == ["reference", "precomputed"]*4
    assert len([event for event in events if event[0] == "host_copy"]) == 72
    assert report["planned_constructor_count"] == 1
    assert len(report["warmup"]) == 2 and len(report["measurements"]) == 3
    assert len(report["solve_attempts"]) == 8
    assert report["bit_exact"] and report["input_pointer_stable"]
    for row in report["solve_attempts"]:
        assert row["input_pointers_before"] == row["input_pointers_after"] == report["input_pointers"]
        assert row["host_copy_count"] == row["receipt"]["host_copy"]["completed_count"] == 9
        for key in ("memory_before", "memory_after_solve", "memory_after_host_copy"):
            assert row[key] == {"pool_used_bytes": 100, "pool_total_bytes": 200,
                                "device_free_bytes": 700, "device_total_bytes": 1000}
    assert json.loads(path.read_text(encoding="utf-8"))["planned_timed_solves"] == 6
    original_bytes = path.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        run_generated_e273_benchmark(output_path=path)
    assert path.read_bytes() == original_bytes
    assert len([event for event in events if event[0] == "constructor"]) == 1


def test_cpu_e273_v2_benchmark_rejects_changed_input_pointer_without_retry(monkeypatch, tmp_path):
    import sys
    module = sys.modules[__name__]
    events = []
    small = generated_benchmark_problem(controls=2)
    monkeypatch.setenv("E273_RUN_GENERATED_CUDA_BENCHMARK", "1")
    monkeypatch.setattr(module, "generated_benchmark_problem", lambda: small)
    monkeypatch.setattr(module, "CudaSourceCoefficientEngine",
                        benchmark_fake_constructor(events, mutate_pointer=True))
    path = tmp_path/"prospective-v2-cpu-failure.json"
    with pytest.raises(AssertionError, match="pointers changed") as failure:
        run_generated_e273_benchmark(output_path=path)
    report = failure.value.benchmark_receipt
    assert not report["input_pointer_stable"]
    assert len(report["solve_attempts"]) == 2
    assert report["solve_attempts"][-1]["status"] == "post_solve_failed"
    assert len([event for event in events if event[0] == "constructor"]) == 1
    assert len([event for event in events if event[0] == "solve"]) == 2
    assert json.loads(path.read_text(encoding="utf-8"))["error"]["type"] == "AssertionError"
