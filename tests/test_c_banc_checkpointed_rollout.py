"""Generated fake SSP chain plus actual fixed E316 decoder, no original model."""

import json
import time

import numpy as np
import pytest
import torch

from tools import banc_m3_checkpointed_rollout as c
from tools.banc_m3_monotone_adjoint import MonotoneDecoder

SEED = 31701
COST = dict.fromkeys(
    (
        "fake_neural_requested",
        "fake_neural_returned",
        "fake_neural_failed",
        "naive_ticks",
        "checkpointed_forward_ticks",
        "checkpointed_recompute_ticks",
        "backward_blocks_requested",
        "backward_blocks_returned",
        "original_calls",
        "model_calls",
        "body_calls",
        "solver_calls",
        "fit_calls",
        "CUDA_calls",
    ),
    0,
)


@pytest.fixture(scope="module", autouse=True)
def metering():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    wall, cpu = time.perf_counter(), time.process_time()
    yield
    COST.update(wall_s=time.perf_counter() - wall, cpu_s=time.process_time() - cpu)
    torch.set_num_threads(old)
    print("E317_GENERATED_COST " + json.dumps(COST, sort_keys=True))
    assert COST["wall_s"] <= 90
    assert (
        COST["original_calls"]
        == COST["model_calls"]
        == COST["body_calls"]
        == COST["CUDA_calls"]
        == 0
    )


class PreparedFake:
    def __init__(self, efficacy):
        self.efficacy, self.version = efficacy, efficacy._version


class FakeOperator:
    n, group_count, device = 400, 2, torch.device("cpu")

    def __init__(self):
        self._gpu = {"lookup": torch.tensor([0.25], dtype=torch.float32)}


class FakeNeural:
    n, device, dt = 400, torch.device("cpu"), 0.00025

    def __init__(self, *, native_dt=False):
        self.operator = FakeOperator()
        if native_dt:
            self.dt = float(np.float32(0.00025))
        self.fixed_coeff = torch.linspace(0.1, 0.3, self.n)
        self.trace = []
        self.fault = {"at": None}

    def prepare_efficacy(self, efficacy):
        return PreparedFake(efficacy)

    def step(self, rate, adaptation, drive, efficacy, mask):
        COST["fake_neural_requested"] += 1
        self.trace.append(1)
        if self.fault["at"] == len(self.trace):
            COST["fake_neural_failed"] += 1
            raise RuntimeError("generated neural fault")
        x = efficacy.efficacy if isinstance(efficacy, PreparedFake) else efficacy
        if isinstance(efficacy, PreparedFake) and efficacy.version != x._version:
            raise RuntimeError("prepared efficacy changed")
        masked = rate * mask
        total = masked.mean()
        target = torch.nn.functional.softplus(
            drive
            + 0.1 * x[0] * masked
            + 0.05 * x[1] * total
            + 0.01 * self.fixed_coeff
            - 0.03 * adaptation
        )
        next_rate = 0.98 * rate + 0.02 * target
        next_adaptation = 0.97 * adaptation + 0.03 * (0.4 * rate)
        COST["fake_neural_returned"] += 1
        return next_rate, next_adaptation


def decoder():
    membership = (np.arange(391) % 96).astype(np.int64)
    counts = np.bincount(membership, minlength=96)
    pool = np.zeros((96, 391), np.float64)
    pool[membership, np.arange(391)] = 1.0 / counts[membership]
    slopes = np.zeros((96, 391, 3, 2), np.float64)
    slopes[:, :, :, 0] = (0.8 / 3) * pool[:, :, None]
    slopes[:, :, :, 1] = (1.2 / 3) * pool[:, :, None]
    fixed = np.arange(6)
    slopes[fixed] = 0
    slopes[fixed, :, 1, :] = pool[fixed, :, None]
    calibration = {
        "moment": np.random.default_rng(SEED).normal(0, 0.01, (42, 96)),
        "rest": np.zeros(42),
        "pool": pool,
        "motor_ids": np.arange(9, 400, dtype=np.int64),
        "ltm": fixed.astype(np.int64),
    }
    return MonotoneDecoder(
        calibration,
        np.array([0.005, 0.01, 0.04]),
        np.full(391, 0.2),
        slopes,
        membership,
        np.zeros(42),
    )


def case(controls=5, *, steps=2, checkpoint=2):
    neural = FakeNeural()
    dec = decoder()
    rollout = c.CheckpointedControlRollout(
        neural,
        dec,
        np.arange(9, 400, dtype=np.int64),
        steps_per_control=steps,
        control_dt=steps * 0.00025,
        checkpoint_controls=checkpoint,
        max_controls=12,
    )
    rng = np.random.default_rng(SEED + controls + steps)
    r = torch.tensor(
        rng.uniform(0.1, 0.3, 400), dtype=torch.float32, requires_grad=True
    )
    z = torch.tensor(
        rng.uniform(0.01, 0.04, 400), dtype=torch.float32, requires_grad=True
    )
    bank = torch.tensor(
        rng.uniform(0.05, 0.2, (391, 3)), dtype=torch.float64, requires_grad=True
    )
    orig = torch.tensor(
        rng.uniform(0.03, 0.15, 96), dtype=torch.float64, requires_grad=True
    )
    drives = torch.tensor(rng.uniform(0.1, 0.3, (controls, 400)), dtype=torch.float32)
    efficacy = torch.tensor([0.8, 1.3], dtype=torch.float64, requires_grad=True)
    mask = torch.ones(400, dtype=torch.float32)
    mask[[9, 25, 300]] = 0
    return rollout, (r, z, bank, orig, drives, efficacy, mask)


def naive(rollout, inputs):
    r, z, bank, orig, drives, efficacy, mask = inputs
    prepared = rollout.neural.prepare_efficacy(efficacy)
    qs, tendons = [], []
    for drive in drives:
        for _ in range(rollout.steps_per_control):
            r, z = rollout.neural.step(r, z, drive, prepared, mask)
            COST["naive_ticks"] += 1
        decoded = rollout.decoder.step(
            r[rollout.motor_ids], bank, orig, dt=rollout.control_dt
        )
        bank, orig = decoded["bank"], decoded["original_activation"]
        qs.append(decoded["q"])
        tendons.append(decoded["tendon"])
    return {
        "q": torch.stack(qs),
        "tendon": torch.stack(tendons),
        "rate": r,
        "adaptation": z,
        "bank": bank,
        "original_activation": orig,
    }


def same_outputs(left, right):
    for name in ("q", "tendon", "rate", "adaptation", "bank", "original_activation"):
        assert torch.equal(left[name], right[name]), name


def loss(result, kind):
    if kind == "all":
        return (
            (result["q"] * torch.linspace(0.1, 1.2, 42)).sum()
            + (result["tendon"] * torch.linspace(0.2, 0.9, 6)).sum()
            + 0.01 * (result["rate"].sum() + result["adaptation"].sum())
            + 0.01 * (result["bank"].sum() + result["original_activation"].sum())
        )
    if kind == "final":
        return (
            0.02 * result["rate"].sum()
            + result["adaptation"].sum() * 0.01
            + (result["bank"].sum() + result["original_activation"].sum()) * 0.03
        )
    if kind == "outputs":
        return result["q"].square().mean() + result["tendon"].square().mean()
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "checkpoint,kind", [(1, "all"), (2, "outputs"), (2, "final"), (3, "all")]
)
def test_full_bptt_byte_forward_and_all_input_gradients(checkpoint, kind):
    rollout, inputs = case(5, checkpoint=checkpoint)
    raw = [v.detach().clone() for v in inputs]
    checked = rollout.run(*inputs)
    COST["checkpointed_forward_ticks"] += checked[
        "receipt"
    ].forward_neural_ticks_returned
    comparison = tuple(
        v.detach().clone().requires_grad_(v.requires_grad) for v in inputs
    )
    direct = naive(rollout, comparison)
    same_outputs(checked, direct)
    loss(checked, kind).backward()
    loss(direct, kind).backward()
    for actual, expected in zip(
        inputs[:4] + (inputs[5],), comparison[:4] + (comparison[5],), strict=True
    ):
        np.testing.assert_allclose(
            actual.grad.numpy(), expected.grad.numpy(), rtol=5e-6, atol=1e-8
        )
    for value, prior in zip(inputs, raw, strict=True):
        assert torch.equal(value.detach(), prior)
    receipt = checked["receipt"]
    COST["checkpointed_recompute_ticks"] += receipt.recompute_neural_ticks_returned
    COST["backward_blocks_requested"] += receipt.backward_blocks_requested
    COST["backward_blocks_returned"] += receipt.backward_blocks_returned
    assert (
        receipt.backward_blocks_requested
        == receipt.backward_blocks_returned
        == (5 + checkpoint - 1) // checkpoint
    )
    assert (
        receipt.backward_block_ticks_returned
        == receipt.recompute_neural_ticks_returned
        == 10
    )
    assert receipt.confirm()["forward_neural_ticks"] == 10
    assert receipt.confirmed["backward_block_ticks"] == 10


def test_mask_is_for_recurrence_only_motor_readout_is_raw():
    rollout, values = case(1, steps=1)
    result = rollout.run(*values)
    fake = rollout.neural
    with torch.no_grad():
        rate, z = fake.step(
            values[0],
            values[1],
            values[4][0],
            fake.prepare_efficacy(values[5]),
            values[6],
        )
        raw = rollout.decoder.step(
            rate[rollout.motor_ids], values[2], values[3], dt=rollout.control_dt
        )
        masked = rollout.decoder.step(
            (rate * values[6])[rollout.motor_ids],
            values[2],
            values[3],
            dt=rollout.control_dt,
        )
    assert torch.equal(result["q"], raw["q"][None])
    assert not torch.equal(raw["q"], masked["q"])
    assert z.shape == result["adaptation"].shape


def test_one_control_independent_dense_neural_and_decoder_formula():
    rollout, values = case(1, steps=1)
    result = rollout.run(*values)
    r, z, bank, original, drives, efficacy, mask = (
        value.detach().numpy() for value in values
    )
    neural, dec = rollout.neural, rollout.decoder
    masked = r * mask
    x = (
        drives[0].astype(np.float64)
        + 0.1 * efficacy[0] * masked
        + 0.05 * efficacy[1] * masked.mean()
        + 0.01 * neural.fixed_coeff.numpy()
        - 0.03 * z
    )
    next_r = 0.98 * r + 0.02 * np.logaddexp(0.0, x)
    next_z = 0.97 * z + 0.03 * (0.4 * r)
    np.testing.assert_allclose(
        result["rate"].detach().numpy(), next_r, rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        result["adaptation"].detach().numpy(), next_z, rtol=2e-6, atol=2e-7
    )
    selected = (
        result["rate"].detach().numpy()[rollout.motor_ids.numpy()].astype(np.float64)
    )
    new_original = original + (-np.expm1(-rollout.control_dt / 0.01)) * (
        dec.pool.numpy() @ selected - original
    )
    alpha = -np.expm1(-rollout.control_dt / dec.tau.numpy())
    new_bank = bank + alpha * (selected[:, None] - bank)
    activation = dec.low.numpy() @ new_bank.reshape(
        -1
    ) + dec.delta.numpy() @ np.maximum(
        new_bank - dec.kappa.numpy()[:, None], 0
    ).reshape(-1)
    activation[dec.fixed_mask.numpy()] = new_original[dec.fixed_mask.numpy()]
    expected_q = dec.rest.numpy() + 0.02 * (dec.moment.numpy() @ activation)
    expected_tendon = 0.2 * np.tanh(activation[dec.ltm.numpy()] / 10.0)
    np.testing.assert_allclose(
        result["q"].detach().numpy()[0], expected_q, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        result["tendon"].detach().numpy()[0], expected_tendon, rtol=1e-12, atol=1e-12
    )


def test_efficacy_directional_fd_against_independent_full_unroll():
    rollout, values = case(3, steps=2, checkpoint=2)
    output = rollout.run(*values)
    loss(output, "outputs").backward()
    direction = torch.tensor([0.6, -0.8], dtype=torch.float64)
    predicted = float(torch.dot(values[5].grad, direction))

    def shifted(sign):
        args = list(values)
        args[5] = (values[5].detach() + sign * 0.01 * direction).contiguous()
        with torch.no_grad():
            return float(loss(naive(rollout, args), "outputs"))

    observed = (shifted(1) - shifted(-1)) / 0.02
    np.testing.assert_allclose(predicted, observed, rtol=0.02, atol=2e-6)


def test_native_represented_dt_default_control_and_small_integrated_gradient():
    _, values = case(1, steps=20)
    neural = FakeNeural(native_dt=True)
    rollout = c.CheckpointedControlRollout(
        neural, decoder(), np.arange(9, 400, dtype=np.int64)
    )
    assert neural.dt == float(np.float32(0.00025))
    assert neural.dt * 20 != 0.005
    checked = rollout.run(*values)
    comparison = tuple(
        v.detach().clone().requires_grad_(v.requires_grad) for v in values
    )
    direct = naive(rollout, comparison)
    same_outputs(checked, direct)
    loss(checked, "all").backward()
    loss(direct, "all").backward()
    np.testing.assert_allclose(
        values[5].grad.numpy(), comparison[5].grad.numpy(), rtol=5e-6, atol=1e-8
    )
    assert checked["receipt"].forward_neural_ticks_returned == 20
    assert checked["receipt"].recompute_neural_ticks_returned == 20
    assert checked["receipt"].confirm()["backward_block_ticks"] == 20


@pytest.mark.parametrize(
    "control_dt",
    [0.004999, 0.005001, 0.005 + 20 * float(np.spacing(np.float32(0.00025)))],
)
def test_native_represented_dt_rejects_real_clock_mismatch(control_dt):
    with pytest.raises(ValueError, match="Control clock"):
        c.CheckpointedControlRollout(
            FakeNeural(native_dt=True),
            decoder(),
            np.arange(9, 400, dtype=np.int64),
            control_dt=control_dt,
        )


def test_two_live_call_state_receipt_and_cache_isolation():
    rollout, values = case(3, steps=1)
    first = rollout.run(*values)
    other = list(values)
    other[5] = values[5].detach().clone().requires_grad_()
    second = rollout.run(*other)
    assert first["receipt"] is not second["receipt"]
    (first["q"].sum() + first["rate"].sum() * 0.001).backward()
    (second["tendon"].sum() + second["bank"].sum() * 0.001).backward()
    assert first["receipt"].backward_blocks_returned == 2
    assert second["receipt"].backward_blocks_returned == 2
    assert values[5].grad is not None and other[5].grad is not None


@pytest.mark.parametrize("kind", ["rate", "bank", "efficacy", "drive", "mask"])
def test_input_mutation_detected_before_backward_and_receipt_not_complete(kind):
    rollout, values = case(2, steps=1)
    output = rollout.run(*values)
    index = {"rate": 0, "bank": 2, "efficacy": 5, "drive": 4, "mask": 6}[kind]
    with torch.no_grad():
        values[index].reshape(-1)[0] += 0.01
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output["q"].sum().backward()
    assert output["receipt"].status == "backward_failed"
    assert output["receipt"].backward_blocks_returned == 0


@pytest.mark.parametrize(
    "kind",
    [
        "tensor_version",
        "tensor_replace",
        "decoder_version",
        "operator_replace",
        "motor_order",
        "clock",
        "operator_buffer",
    ],
)
def test_frozen_model_identity_and_version_guard(kind):
    rollout, values = case(2, steps=1)
    output = rollout.run(*values)
    if kind == "tensor_version":
        rollout.neural.fixed_coeff[0] += 0.01
    elif kind == "tensor_replace":
        rollout.neural.fixed_coeff = rollout.neural.fixed_coeff.clone()
    elif kind == "decoder_version":
        rollout.decoder.moment[0, 0] += 0.01
    elif kind == "operator_replace":
        rollout.neural.operator = FakeOperator()
    elif kind == "motor_order":
        rollout.motor_ids[0] = 8
    elif kind == "operator_buffer":
        rollout.neural.operator._gpu["lookup"] += 0.01
    else:
        rollout.control_dt += 0.0001
    with pytest.raises(RuntimeError, match="changed"):
        output["q"].sum().backward()
    assert output["receipt"].status == "backward_failed"


def test_higher_order_explicit_rejection_and_no_block_completion():
    rollout, values = case(2, steps=1)
    output = rollout.run(*values)
    with pytest.raises(RuntimeError, match="Higher-order"):
        torch.autograd.grad(output["q"].sum(), values[5], create_graph=True)
    assert output["receipt"].status == "higher_order_rejected"
    assert output["receipt"].backward_blocks_returned == 0


def test_forward_partial_failure_host_returned_prefix_and_confirm():
    rollout, values = case(3, steps=2)
    rollout.neural.fault["at"] = 2
    with pytest.raises(RuntimeError, match="generated neural fault") as caught:
        rollout.run(*values)
    receipt = caught.value.receipt
    assert receipt.status == "forward_failed"
    assert (
        receipt.forward_controls_requested == 1
        and receipt.forward_controls_returned == 0
    )
    assert (
        receipt.forward_neural_ticks_requested == 2
        and receipt.forward_neural_ticks_returned == 1
    )
    assert receipt.confirm()["forward_neural_ticks"] == 1
    assert receipt.backward_blocks_returned == 0


def test_failed_recompute_block_never_claimed_returned():
    rollout, values = case(4, steps=1, checkpoint=2)
    output = rollout.run(*values)
    original_ticks = len(rollout.neural.trace)
    rollout.neural.fault["at"] = original_ticks + 2
    with pytest.raises(RuntimeError, match="generated neural fault"):
        output["q"].sum().backward()
    receipt = output["receipt"]
    assert (
        receipt.status == "backward_failed" and receipt.backward_blocks_requested == 1
    )
    assert (
        receipt.backward_blocks_returned == receipt.backward_block_ticks_returned == 0
    )
    assert (
        receipt.recompute_neural_ticks_requested == 2
        and receipt.recompute_neural_ticks_returned == 1
    )


@pytest.mark.parametrize(
    "field",
    [
        "rate_shape",
        "rate_dtype",
        "bank_negative",
        "efficacy_zero",
        "drive_grad",
        "mask_fraction",
        "mask_grad",
        "control_zero",
        "control_excess",
    ],
)
def test_input_validation_no_forward_tick_and_failure_receipt(field):
    rollout, values = case(2, steps=1)
    values = list(values)
    if field == "rate_shape":
        values[0] = values[0][:-1]
    elif field == "rate_dtype":
        values[0] = values[0].double()
    elif field == "bank_negative":
        values[2] = -values[2]
    elif field == "efficacy_zero":
        values[5] = torch.zeros_like(values[5])
    elif field == "drive_grad":
        values[4].requires_grad_(True)
    elif field == "mask_fraction":
        values[6][0] = 0.5
    elif field == "mask_grad":
        values[6].requires_grad_(True)
    elif field == "control_zero":
        values[4] = values[4][:0]
    else:
        values[4] = torch.cat((values[4], values[4].repeat(6, 1)))
    with pytest.raises(ValueError) as caught:
        rollout.run(*values)
    assert caught.value.receipt.forward_neural_ticks_requested == 0
    assert caught.value.receipt.status == "forward_failed"


@pytest.mark.parametrize(
    "field",
    [
        "steps_bool",
        "steps_float",
        "check_zero",
        "motor_unsorted",
        "motor_short",
        "clock_wrong",
        "max_bool",
    ],
)
def test_constructor_bad_contracts(field):
    neural, dec = FakeNeural(), decoder()
    ids = np.arange(9, 400, dtype=np.int64)
    kwargs = {"steps_per_control": 1, "control_dt": 0.00025}
    if field == "steps_bool":
        kwargs["steps_per_control"] = True
    elif field == "steps_float":
        kwargs["steps_per_control"] = 1.0
    elif field == "check_zero":
        kwargs["checkpoint_controls"] = 0
    elif field == "motor_unsorted":
        ids[[0, 1]] = ids[[1, 0]]
    elif field == "motor_short":
        ids = ids[:-1]
    elif field == "clock_wrong":
        kwargs["control_dt"] = 0.005
    else:
        kwargs["max_controls"] = True
    with pytest.raises((ValueError, TypeError)):
        c.CheckpointedControlRollout(neural, dec, ids, **kwargs)
