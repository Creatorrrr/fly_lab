"""E317 manual control-boundary recomputation for full first-order SSP BPTT.

This is an offline functional operator, not a body, controller or checkpoint.
The only persistent trajectory states are the four control-boundary tensors.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from scipy.sparse import spmatrix

SCHEMA = "flylab.m3.e317.checkpointed-rollout.v1"
_STATE_KEYS = ("rate", "adaptation", "bank", "original_activation")


def _require(predicate, message, device):
    if device.type == "cuda":
        torch._assert_async(predicate, message)
    elif not bool(predicate):
        raise ValueError(message)


def _digest(array):
    value = np.ascontiguousarray(array)
    checksum = hashlib.sha256()
    checksum.update(value.dtype.str.encode())
    checksum.update(np.asarray(value.shape, np.int64).tobytes())
    checksum.update(memoryview(value).cast("B"))
    return checksum.digest()


def _model_signature(neural, decoder):
    """Capture public/frozen tensors and owned native arrays, without copying."""
    entries = [(neural, "operator", neural.operator)]
    for owner in (neural, getattr(neural, "operator", None), decoder):
        if owner is None:
            continue
        for method in ("step", "prepare_efficacy", "current"):
            entries.append((owner, "class." + method, type(owner).__dict__.get(method)))
            entries.append((owner, "instance." + method, vars(owner).get(method)))
        for name, value in sorted(vars(owner).items()):
            if isinstance(value, torch.Tensor):
                try:
                    version = value._version
                except RuntimeError as error:
                    raise ValueError(
                        "Versioned frozen model tensors required"
                    ) from error
                entries.append(
                    (
                        owner,
                        name,
                        value,
                        version,
                        value.data_ptr(),
                        value.dtype,
                        tuple(value.shape),
                    )
                )
            elif isinstance(value, np.ndarray):
                entries.append((owner, name, value, _digest(value)))
            elif isinstance(value, spmatrix):
                entries.append(
                    (
                        owner,
                        name,
                        value,
                        *(
                            _digest(a)
                            for a in (value.indptr, value.indices, value.data)
                        ),
                    )
                )
            elif name == "_gpu" and isinstance(value, dict):
                buffers = {}
                for key, item in value.items():
                    if isinstance(item, torch.Tensor):
                        buffers[key] = (
                            item,
                            item._version,
                            item.data_ptr(),
                            item.dtype,
                            tuple(item.shape),
                        )
                entries.append(
                    (owner, "gpu-dict", value, tuple(value.items()), buffers)
                )
            elif type(value) in (int, float, bool, str, torch.device):
                entries.append((owner, name, value))
    return entries


def _check_signature(signature):
    for entry in signature:
        owner, name, previous, *extra = entry
        current = (
            type(owner).__dict__.get(name[6:])
            if name.startswith("class.")
            else vars(owner).get(name[9:])
            if name.startswith("instance.")
            else vars(owner).get("_gpu")
            if name == "gpu-dict"
            else vars(owner).get(name)
        )
        if name.startswith(("class.", "instance.")):
            if current is not previous:
                raise RuntimeError(
                    "Frozen model method changed while rollout graph was live"
                )
        elif isinstance(previous, torch.Tensor):
            version, pointer, dtype, shape = extra
            if (
                current is not previous
                or current._version != version
                or current.data_ptr() != pointer
                or current.dtype != dtype
                or tuple(current.shape) != shape
            ):
                raise RuntimeError(
                    "Frozen model tensor changed while rollout graph was live"
                )
        elif isinstance(previous, np.ndarray):
            if current is not previous or _digest(current) != extra[0]:
                raise RuntimeError(
                    "Frozen model array changed while rollout graph was live"
                )
        elif isinstance(previous, spmatrix):
            if current is not previous or tuple(
                _digest(a) for a in (current.indptr, current.indices, current.data)
            ) != tuple(extra):
                raise RuntimeError(
                    "Frozen model CSR changed while rollout graph was live"
                )
        elif name == "gpu-dict":
            if (
                current is not previous
                or set(current) != {key for key, _ in extra[0]}
                or any(current[key] is not item for key, item in extra[0])
            ):
                raise RuntimeError(
                    "Frozen CUDA current buffers changed while rollout graph was live"
                )
            if any(
                current.get(key) is not item
                or item._version != version
                or item.data_ptr() != pointer
                or item.dtype != dtype
                or tuple(item.shape) != shape
                for key, (item, version, pointer, dtype, shape) in extra[1].items()
            ):
                raise RuntimeError(
                    "Frozen CUDA current buffer changed while rollout graph was live"
                )
        elif name == "operator":
            if current is not previous:
                raise RuntimeError(
                    "Frozen current operator changed while rollout graph was live"
                )
        elif current != previous:
            raise RuntimeError(
                "Frozen model constant changed while rollout graph was live"
            )


class RolloutReceipt:
    """Host-returned counts are known; device completion requires confirm()."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.schema = SCHEMA
        self.status = "created"
        self.forward_controls_requested = 0
        self.forward_controls_returned = 0
        self.forward_neural_ticks_requested = 0
        self.forward_neural_ticks_returned = 0
        self.recompute_controls_requested = 0
        self.recompute_controls_returned = 0
        self.recompute_neural_ticks_requested = 0
        self.recompute_neural_ticks_returned = 0
        self.backward_blocks_requested = 0
        self.backward_blocks_returned = 0
        self.backward_block_ticks_returned = 0
        self.confirm_requested = 0
        self.confirm_returned = 0
        self.confirmed = None

    def confirm(self):
        self.confirm_requested += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.confirmed = {
            "forward_controls": self.forward_controls_returned,
            "forward_neural_ticks": self.forward_neural_ticks_returned,
            "recompute_controls": self.recompute_controls_returned,
            "recompute_neural_ticks": self.recompute_neural_ticks_returned,
            "backward_blocks": self.backward_blocks_returned,
            "backward_block_ticks": self.backward_block_ticks_returned,
        }
        self.confirm_returned += 1
        return self.confirmed.copy()


def _control(rollout, state, drive, efficacy, mask, receipt, *, recompute):
    r, z, bank, original = state
    prefix = "recompute" if recompute else "forward"
    setattr(
        receipt,
        prefix + "_controls_requested",
        getattr(receipt, prefix + "_controls_requested") + 1,
    )
    for _ in range(rollout.steps_per_control):
        field = prefix + "_neural_ticks_requested"
        setattr(receipt, field, getattr(receipt, field) + 1)
        r, z = rollout.neural.step(r, z, drive, efficacy, mask)
        field = prefix + "_neural_ticks_returned"
        setattr(receipt, field, getattr(receipt, field) + 1)
    selected = torch.index_select(r, 0, rollout.motor_ids)
    decoded = rollout.decoder.step(selected, bank, original, dt=rollout.control_dt)
    state = (r, z, decoded["bank"], decoded["original_activation"])
    setattr(
        receipt,
        prefix + "_controls_returned",
        getattr(receipt, prefix + "_controls_returned") + 1,
    )
    return state, decoded["q"], decoded["tendon"]


class _Checkpointed(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, rate, adaptation, bank, original, drives, efficacy, mask, rollout, receipt
    ):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(rate, adaptation, bank, original, drives, efficacy, mask)
        ctx.rollout, ctx.receipt = rollout, receipt
        ctx.neural, ctx.decoder = rollout.neural, rollout.decoder
        ctx.motor_ids = (
            rollout.motor_ids,
            rollout.motor_ids._version,
            rollout.motor_ids.data_ptr(),
        )
        ctx.settings = (
            rollout.steps_per_control,
            rollout.control_dt,
            rollout.checkpoint_controls,
            rollout.max_controls,
        )
        ctx.controls = int(drives.shape[0])
        ctx.backward_called = False
        state = (rate, adaptation, bank, original)
        checkpoints = [(0, tuple(v.detach().clone() for v in state))]
        qs, tendons = [], []
        receipt.status = "forward_running"
        try:
            prepared = rollout.neural.prepare_efficacy(efficacy)
            with torch.no_grad():
                for control in range(ctx.controls):
                    state, q, tendon = _control(
                        rollout,
                        state,
                        drives[control],
                        prepared,
                        mask,
                        receipt,
                        recompute=False,
                    )
                    qs.append(q)
                    tendons.append(tendon)
                    if (
                        control + 1
                    ) % rollout.checkpoint_controls == 0 or control + 1 == ctx.controls:
                        checkpoints.append(
                            (control + 1, tuple(v.detach().clone() for v in state))
                        )
            ctx.checkpoints = checkpoints
            ctx.signature = _model_signature(rollout.neural, rollout.decoder)
            receipt.status = "forward_returned"
            return torch.stack(qs), torch.stack(tendons), *state
        except BaseException:
            receipt.status = "forward_failed"
            raise

    @staticmethod
    def backward(ctx, grad_q, grad_tendon, grad_r, grad_z, grad_bank, grad_original):
        receipt = ctx.receipt
        if torch.is_grad_enabled():
            receipt.status = "higher_order_rejected"
            raise RuntimeError(
                "Higher-order checkpointed rollout differentiation is unsupported"
            )
        if ctx.backward_called:
            receipt.status = "backward_rejected"
            raise RuntimeError(
                "Repeated backward through one checkpointed rollout is unsupported"
            )
        ctx.backward_called = True
        receipt.status = "backward_running"
        try:
            rate, adaptation, bank, original, drives, efficacy, mask = ctx.saved_tensors
            rollout = ctx.rollout
            motor_ref, motor_version, motor_ptr = ctx.motor_ids
            if (
                rollout.neural is not ctx.neural
                or rollout.decoder is not ctx.decoder
                or rollout.motor_ids is not motor_ref
                or motor_ref._version != motor_version
                or motor_ref.data_ptr() != motor_ptr
                or (
                    rollout.steps_per_control,
                    rollout.control_dt,
                    rollout.checkpoint_controls,
                    rollout.max_controls,
                )
                != ctx.settings
            ):
                raise RuntimeError("Rollout model, motor order or clock changed")
            cotangents = tuple(
                torch.zeros_like(value) if grad is None else grad
                for value, grad in zip(
                    (rate, adaptation, bank, original),
                    (grad_r, grad_z, grad_bank, grad_original),
                    strict=True,
                )
            )
            total_eff = torch.zeros_like(efficacy, dtype=torch.float64)
            _check_signature(ctx.signature)
            with torch.enable_grad():
                effleaf = efficacy.detach().requires_grad_(True)
                prepared = rollout.neural.prepare_efficacy(effleaf)
                for block in range(len(ctx.checkpoints) - 2, -1, -1):
                    begin, frozen = ctx.checkpoints[block]
                    end, _ = ctx.checkpoints[block + 1]
                    receipt.backward_blocks_requested += 1
                    boundary = tuple(
                        value.detach().requires_grad_(True) for value in frozen
                    )
                    state = boundary
                    outputs, seeds = [], []
                    for control in range(begin, end):
                        state, q, tendon = _control(
                            rollout,
                            state,
                            drives[control],
                            prepared,
                            mask,
                            receipt,
                            recompute=True,
                        )
                        if grad_q is not None:
                            outputs.append(q)
                            seeds.append(grad_q[control])
                        if grad_tendon is not None:
                            outputs.append(tendon)
                            seeds.append(grad_tendon[control])
                    outputs.extend(state)
                    seeds.extend(cotangents)
                    valid = [
                        (output, seed)
                        for output, seed in zip(outputs, seeds, strict=True)
                        if output.requires_grad
                    ]
                    gradients = (
                        torch.autograd.grad(
                            [v[0] for v in valid],
                            (*boundary, effleaf),
                            grad_outputs=[v[1] for v in valid],
                            allow_unused=True,
                            create_graph=False,
                        )
                        if valid
                        else (None,) * 5
                    )
                    cotangents = tuple(
                        torch.zeros_like(value) if got is None else got
                        for value, got in zip(boundary, gradients[:4], strict=True)
                    )
                    if gradients[4] is not None:
                        total_eff = total_eff + gradients[4].to(torch.float64)
                    receipt.backward_blocks_returned += 1
                    receipt.backward_block_ticks_returned += (
                        end - begin
                    ) * rollout.steps_per_control
            receipt.status = "backward_returned"
            return (*cotangents, None, total_eff, None, None, None)
        except BaseException:
            receipt.status = "backward_failed"
            raise


class CheckpointedControlRollout:
    """Exact functional forward, bounded four-state checkpoints, full BPTT."""

    def __init__(
        self,
        neural,
        decoder,
        motor_ids,
        *,
        steps_per_control=20,
        control_dt=0.005,
        checkpoint_controls=1,
        max_controls=320,
    ):
        if (
            not callable(getattr(neural, "step", None))
            or not callable(getattr(neural, "prepare_efficacy", None))
            or not callable(getattr(decoder, "step", None))
            or type(getattr(getattr(neural, "operator", None), "group_count", None))
            is not int
            or type(getattr(neural, "n", None)) is not int
            or neural.n < 391
        ):
            raise TypeError("Functional native neural and decoder APIs required")
        self.neural, self.decoder = neural, decoder
        self.device = torch.device(neural.device)
        if self.device != torch.device(decoder.device):
            raise ValueError("Neural and decoder must share an exact resolved device")
        ids = np.asarray(motor_ids)
        if (
            ids.dtype != np.int64
            or ids.shape != (391,)
            or not np.array_equal(ids, np.unique(ids))
            or np.any(ids < 0)
            or np.any(ids >= neural.n)
        ):
            raise ValueError("Sorted unique 391 native motor IDs required")
        for label, value in (
            ("steps_per_control", steps_per_control),
            ("checkpoint_controls", checkpoint_controls),
            ("max_controls", max_controls),
        ):
            if type(value) is not int or value < 1:
                raise ValueError("Positive integral " + label + " required")
        # Native SSP stores its step as float32.  Compare the nominal per-tick
        # control interval to that represented step within half a float32 ULP;
        # multiplying the represented step first and rounding the product again
        # would reject the original 20 * 0.00025 -> 0.005 clock.
        nominal_step = (
            float(control_dt) / steps_per_control
            if type(control_dt) in (int, float)
            else float("nan")
        )
        step_rounding = (
            0.5 * float(np.spacing(np.float32(nominal_step)))
            if np.isfinite(nominal_step)
            else float("nan")
        )
        if (
            type(control_dt) not in (int, float)
            or not np.isfinite(control_dt)
            or control_dt <= 0
            or control_dt > 0.05
            or not np.isfinite(neural.dt)
            or abs(float(neural.dt) - nominal_step) > step_rounding
        ):
            raise ValueError(
                "Control clock must match the float32-represented neural timestep"
            )
        self.motor_ids = torch.as_tensor(
            ids.copy(), dtype=torch.int64, device=self.device
        )
        self.steps_per_control, self.control_dt = steps_per_control, float(control_dt)
        self.checkpoint_controls, self.max_controls = checkpoint_controls, max_controls

    def _validate(self, rate, adaptation, bank, original, drives, efficacy, mask):
        n, device = self.neural.n, self.device
        g = self.neural.operator.group_count
        if not isinstance(drives, torch.Tensor) or drives.ndim != 2:
            raise ValueError("Contiguous held drive matrix required")
        controls = drives.shape[0]
        specifications = (
            (rate, (n,), torch.float32, "rate"),
            (adaptation, (n,), torch.float32, "adaptation"),
            (bank, (391, 3), torch.float64, "bank"),
            (original, (96,), torch.float64, "original activation"),
            (drives, (controls, n), torch.float32, "drives"),
            (efficacy, (g,), torch.float64, "efficacy"),
            (mask, (n,), torch.float32, "mask"),
        )
        if controls < 1 or controls > self.max_controls:
            raise ValueError("Bounded positive control count required")
        for value, shape, dtype, name in specifications:
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != shape
                or value.dtype != dtype
                or value.device != device
                or not value.is_contiguous()
            ):
                raise ValueError(
                    "Contiguous same-device resolved input required: " + name
                )
            _require(
                torch.isfinite(value).all(), "Finite " + name + " required", device
            )
        if drives.requires_grad or mask.requires_grad:
            raise ValueError("Drive and mask gradients are not supported")
        _require(
            torch.all(rate >= 0)
            & torch.all(adaptation >= 0)
            & torch.all(bank >= 0)
            & torch.all(original >= 0)
            & torch.all(efficacy > 0),
            "Nonnegative state and positive efficacy required",
            device,
        )
        _require(
            torch.all((mask == 0) | (mask == 1)), "Binary output mask required", device
        )

    def run(self, rate, adaptation, bank, original_activation, drives, efficacy, mask):
        receipt = RolloutReceipt(self.device)
        try:
            self._validate(
                rate, adaptation, bank, original_activation, drives, efficacy, mask
            )
            outputs = _Checkpointed.apply(
                rate,
                adaptation,
                bank,
                original_activation,
                drives,
                efficacy,
                mask,
                self,
                receipt,
            )
            return {
                **dict(zip(("q", "tendon", *_STATE_KEYS), outputs, strict=True)),
                "receipt": receipt,
            }
        except BaseException as error:
            receipt.status = "forward_failed"
            error.receipt = receipt
            raise
