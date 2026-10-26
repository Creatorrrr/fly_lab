"""Functional differentiable form of the fixed E150 monotone muscle decoder.

This module owns no model, body or checkpoint.  All calibration tensors are
fixed copies; only the three arguments to ``step`` carry trajectory state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch

_TAU = np.array([0.005, 0.01, 0.04], dtype=np.float64)
_ATOL = 1e-12


def _array(value, shape, dtype, label):
    array = np.asarray(value)
    if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
        raise ValueError(f"Finite {label} with shape {shape} and dtype {dtype} required")
    return np.array(array, copy=True)


def _check_device_value(condition, message, device):
    if device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


class MonotoneDecoder:
    """Fixed E150 anatomy with explicit, differentiable control-step state."""

    def __init__(
        self, calibration, tau, kappa, slopes, membership, neutral, *, device="cpu"
    ):
        if not isinstance(calibration, Mapping):
            raise TypeError("Original anatomical calibration mapping required")
        try:
            moment = _array(calibration["moment"], (42, 96), np.float64, "moment")
            rest = _array(calibration["rest"], (42,), np.float64, "rest")
            pool = _array(calibration["pool"], (96, 391), np.float64, "pool")
            motor_ids = _array(calibration["motor_ids"], (391,), np.int64, "motor IDs")
            ltm = _array(calibration["ltm"], (6,), np.int64, "LTM rows")
        except KeyError as exc:
            raise ValueError("Incomplete original anatomical calibration") from exc
        tau = _array(tau, (3,), np.float64, "filter tau")
        kappa = _array(kappa, (391,), np.float64, "kappa")
        slopes = _array(slopes, (96, 391, 3, 2), np.float64, "slopes")
        membership = _array(membership, (391,), np.int64, "membership")
        neutral = _array(neutral, (42,), np.float64, "neutral")
        if not np.array_equal(tau, _TAU):
            raise ValueError("Exact original filter tau required")
        if not np.array_equal(motor_ids, np.sort(np.unique(motor_ids))):
            raise ValueError("Sorted unique original motor IDs required")
        if len(np.unique(ltm)) != 6 or np.any((ltm < 0) | (ltm >= 96)):
            raise ValueError("Six unique original LTM rows required")
        if np.max(np.abs(rest)) > 10 or np.any(pool < 0) or not np.allclose(
            pool.sum(axis=1), 1, rtol=0, atol=1e-15
        ):
            raise ValueError("Finite normalized original calibration required")
        if np.any(kappa < 0) or np.any(slopes < 0):
            raise ValueError("Nonnegative kappa and slopes required")
        if np.any((membership < 0) | (membership >= 96)):
            raise ValueError("Valid anatomical membership required")
        support = np.zeros((96, 391), dtype=bool)
        support[membership, np.arange(391)] = True
        if not np.array_equal((pool > 0).sum(axis=0), np.ones(391, dtype=int)) or not np.array_equal(
            pool.argmax(axis=0), membership
        ):
            raise ValueError("Membership differs from original anatomical pool")
        if np.any(slopes[~support] != 0):
            raise ValueError("Monotone slopes changed anatomical support")
        counts = np.bincount(membership, minlength=96)
        if np.any(counts == 0):
            raise ValueError("Every original muscle requires at least one MN")
        fixed = np.union1d(ltm, np.flatnonzero(counts == 1))
        canonical = np.zeros_like(slopes)
        canonical[:, :, 1, :] = pool[:, :, None]
        if not np.array_equal(slopes[fixed], canonical[fixed]):
            raise ValueError("Fixed LTM and singleton weights must equal original linear transfer")
        selected = slopes[membership, np.arange(391)]
        minimum = selected.min(axis=-1)
        maximum = selected.max(axis=-1)
        lower = np.bincount(membership, weights=minimum.sum(axis=1), minlength=96)
        upper = np.bincount(membership, weights=maximum.sum(axis=1), minlength=96)
        if np.any(lower < 0.5 - _ATOL) or np.any(upper > 2 + _ATOL):
            raise ValueError("Per-muscle slope sums must stay within 0.5-2")
        if np.any(minimum.sum(axis=1) < 1e-6 / counts[membership] - _ATOL):
            raise ValueError("Original per-MN minimum slope floor changed")

        self.device = torch.device(device)
        self.moment = torch.as_tensor(moment.copy(), device=self.device)
        self.device = self.moment.device  # Resolve an unspecified CUDA index once, outside step.
        self.rest = torch.as_tensor(rest.copy(), device=self.device)
        self.pool = torch.as_tensor(pool.copy(), device=self.device)
        self.tau = torch.as_tensor(tau.copy(), device=self.device)
        self.kappa = torch.as_tensor(kappa.copy(), device=self.device)
        self.neutral = torch.as_tensor(neutral.copy(), device=self.device)
        self.ltm = torch.as_tensor(ltm.copy(), device=self.device)
        fixed_mask = np.zeros(96, dtype=bool)
        fixed_mask[fixed] = True
        self.fixed_mask = torch.as_tensor(fixed_mask.copy(), device=self.device)
        self.low = torch.as_tensor(slopes[..., 0].reshape(96, 391 * 3).copy(), device=self.device)
        self.delta = torch.as_tensor(
            (slopes[..., 1] - slopes[..., 0]).reshape(96, 391 * 3).copy(), device=self.device
        )

    def step(self, rate, bank, original_activation, *, dt=0.005, motor_cut=False):
        if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, (int, float, np.floating)) or not math.isfinite(float(dt)) or not 0 < float(dt) <= 0.05:
            raise ValueError("Finite actual timestep in (0, 0.05] required")
        if type(motor_cut) is not bool:
            raise ValueError("Boolean motor cut required")
        for name, value, shape, dtype in (
            ("rate", rate, (391,), torch.float32),
            ("bank", bank, (391, 3), torch.float64),
            ("original activation", original_activation, (96,), torch.float64),
        ):
            if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype or value.device != self.device:
                raise ValueError(f"{name} must be a {dtype} tensor {shape} on {self.device}")
            _check_device_value(torch.isfinite(value).all() & (value >= 0).all(), f"Finite nonnegative {name} required", self.device)

        rate64 = rate.to(torch.float64)
        if motor_cut:
            # Retain a zero VJP to all three arguments, including from q.
            # A global sum can overflow for valid finite saved states. Selecting
            # one element still gives each full input a zero VJP via indexing.
            zero = rate64[0] * 0.0 + bank[0, 0] * 0.0 + original_activation[0] * 0.0
            new_bank = torch.zeros_like(bank) + zero
            new_original = torch.zeros_like(original_activation) + zero
            activation = torch.zeros_like(original_activation) + zero
            q = self.neutral + zero
            tendon = torch.zeros(6, dtype=torch.float64, device=self.device) + zero
        else:
            pooled = self.pool @ rate64
            original_alpha = -math.expm1(-float(dt) / 0.01)
            new_original = original_activation + original_alpha * (pooled - original_activation)
            alpha = -torch.expm1(-float(dt) / self.tau)
            new_bank = bank + alpha * (rate64[:, None] - bank)
            hinge = torch.relu(new_bank - self.kappa[:, None])
            variable = self.low @ new_bank.reshape(-1)
            variable = variable + self.delta @ hinge.reshape(-1)
            activation = torch.where(self.fixed_mask, new_original, variable)
            q = self.rest + 0.02 * (self.moment @ activation)
            tendon = 0.2 * torch.tanh(activation[self.ltm] / 10.0)
            valid = (
                torch.isfinite(new_bank).all()
                & torch.isfinite(new_original).all()
                & torch.isfinite(activation).all()
                & torch.isfinite(q).all()
                & torch.isfinite(tendon).all()
                & (new_bank >= 0).all()
                & (new_original >= 0).all()
                & (activation >= 0).all()
                & (tendon >= 0).all()
            )
            _check_device_value(valid, "Finite nonnegative muscle state and finite q required", self.device)
        return {
            "bank": new_bank,
            "original_activation": new_original,
            "activation": activation,
            "q": q,
            "tendon": tendon,
        }
