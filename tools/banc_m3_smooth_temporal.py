"""E332 fixed all-128-anchor Gaussian offline temporal objective.

Only constant references are detached. Commands, weights, successor targets and
unnormalized circular moments retain their full derivatives. Resultant length
and entropy are diagnostics; no runtime phase, controller or neural loader is
provided here. A static successor fixed point can exist, including a symmetric
circle's stationary center with coordination loss one and zero gradient.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

FREQUENCY_HZ = 2.382760007990865
CONTROL_DT_S = .005
ANCHORS = 128
ALL_PAIRS = tuple((a, b) for a in range(6) for b in range(a + 1, 6))
LR_PAIRS = ((0, 3), (1, 4), (2, 5))
CONSTANT_RTOL = 1e-12
CONSTANT_ATOL = 1e-15


def _host(value, name, shape):
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float64:
            raise TypeError(name + " must be float64")
        value = value.detach().cpu().numpy()
    if not isinstance(value, np.ndarray) or value.dtype != np.float64:
        raise TypeError(name + " must be float64")
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(name + " must have the required shape and be finite")
    return np.array(value, dtype="<f8", order="C", copy=True)


def _metadata(array):
    return {"shape": list(array.shape), "dtype": "float64", "numpy_dtype": "<f8",
            "layout": "C", "c_order_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def _guard(value, name):
    torch._assert_async(torch.isfinite(value).all(), name + " must be finite")
    if value.requires_grad:
        def finite_gradient(gradient):
            torch._assert_async(torch.isfinite(gradient).all(), name + " gradient must be finite")
            return gradient
        value.register_hook(finite_gradient)
    return value


class SmoothTemporalObjective:
    """Frozen frequency/dt, tau=7*n, all pairs with coefficient exactly one.

    ``normalization`` optionally supplies approved E329 bytes. The independent
    constructor recomputation must agree before any canonical loss constants
    are copied to the requested CPU/CUDA device. Supplied values are never
    silently overwritten. All actual loss constants are host-copied and hashed.
    """

    def __init__(self, curves, startup_q, startup_tendon, *, frequency=FREQUENCY_HZ,
                 dt=CONTROL_DT_S, device="cpu", normalization=None):
        if (isinstance(frequency, bool) or isinstance(dt, bool)
                or frequency != FREQUENCY_HZ or dt != CONTROL_DT_S):
            raise ValueError("frequency and dt must equal the fixed E332 constants")
        requested = torch.device(device)
        if requested.type not in ("cpu", "cuda"):
            raise ValueError("only CPU or CUDA devices are supported")
        if requested.type == "cpu" and requested.index not in (None, 0):
            raise ValueError("invalid CPU index")
        self.frequency, self.dt = FREQUENCY_HZ, CONTROL_DT_S
        self.phase_increment = self.frequency * self.dt
        c = _host(curves, "curves", (6, ANCHORS, 7))
        sq = _host(startup_q, "startup_q", (99, 42))
        st = _host(startup_tendon, "startup_tendon", (99, 6))
        # E329's approved wrapped interpolation is used solely for constants.
        from tools.analyze_c_banc_m3_temporal_objective import numpy_interpolate
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            edge_norm = np.sum((np.roll(c, -1, axis=1) - c) ** 2, axis=-1)
            if not np.isfinite(edge_norm).all() or np.any(edge_norm <= 0):
                raise ValueError("reference edges must have positive finite squared lengths")
            phase = np.tile(np.arange(ANCHORS, dtype=np.float64)[:, None] / ANCHORS, (1, 6))
            future = np.ascontiguousarray(numpy_interpolate(c, phase + self.phase_increment)
                                         .reshape(ANCHORS, 6, 7).transpose(1, 0, 2), dtype="<f8")
            computed_n = np.mean((future - c) ** 2, axis=(1, 2))
            n = (_host(normalization, "normalization", (6,)) if normalization is not None
                 else np.ascontiguousarray(computed_n, dtype="<f8"))
            if not np.allclose(computed_n, n, rtol=CONSTANT_RTOL, atol=CONSTANT_ATOL):
                raise ValueError("approved normalization differs from CPU recomputation")
            tau = np.ascontiguousarray(7 * n, dtype="<f8")
            angle = 2 * np.pi * (np.arange(ANCHORS, dtype=np.float64) / ANCHORS)
            basis = np.ascontiguousarray(np.column_stack((np.cos(angle), np.sin(angle))), dtype="<f8")
        if not np.isfinite(n).all() or np.any(n <= 0) or not np.isfinite(tau).all() or np.any(tau <= 0):
            raise ValueError("normalization and temperature must be positive finite")
        canonical = {"curves": c, "normalization": n, "temperature": tau,
                     "future_anchors": future, "phase_basis": basis}
        if not all(np.isfinite(array).all() for array in canonical.values()):
            raise ValueError("canonical constants must be finite")
        # Independent Torch interpolation/arithmetic on the requested device.
        # This happens before canonical n/tau/A/V are transferred for the loss.
        rec_c = torch.tensor(c, dtype=torch.float64, device=requested)
        rec_phase = (torch.arange(ANCHORS, dtype=torch.float64, device=requested)
                     / ANCHORS)[:, None].expand(-1, 6)
        coordinate = torch.remainder(rec_phase + self.phase_increment, 1) * ANCHORS
        floor = torch.floor(coordinate)
        index = torch.remainder(floor.to(torch.long), ANCHORS)
        legs = torch.arange(6, device=requested)[None]
        first = rec_c[legs, index]
        rec_a = (first + (coordinate - floor)[..., None]
                 * (rec_c[legs, (index + 1) % ANCHORS] - first)).permute(1, 0, 2).contiguous()
        rec_n = ((rec_a - rec_c) ** 2).mean(dim=(1, 2))
        rec_angle = 2 * math.pi * (torch.arange(ANCHORS, dtype=torch.float64, device=requested) / ANCHORS)
        recomputed = {"normalization": rec_n, "temperature": 7 * rec_n,
                      "future_anchors": rec_a, "phase_basis": torch.stack((rec_angle.cos(), rec_angle.sin()), -1)}
        self.constructor_recomputed = {}
        for name, tensor in recomputed.items():
            actual = tensor.detach().cpu().numpy().copy()
            if (not np.isfinite(actual).all()
                    or not np.allclose(actual, canonical[name], rtol=CONSTANT_RTOL, atol=CONSTANT_ATOL)):
                raise ValueError("independent constructor constant mismatch: " + name)
            self.constructor_recomputed[name] = actual
        self.canonical_constants = {name: array.copy() for name, array in canonical.items()}
        self.constant_metadata = {name: _metadata(array) for name, array in canonical.items()}
        attributes = {"curves": "curves", "normalization": "successor_normalization",
                      "temperature": "temperature", "future_anchors": "future_anchors",
                      "phase_basis": "phase_basis"}
        for name, attribute in attributes.items():
            tensor = torch.tensor(canonical[name], dtype=torch.float64, device=requested).detach()
            if _metadata(tensor.cpu().numpy()) != self.constant_metadata[name]:
                raise ValueError("actual loss constant bytes differ: " + name)
            setattr(self, attribute, tensor)
        self.device = self.curves.device
        self.startup_q = torch.tensor(sq, dtype=torch.float64, device=self.device)
        self.startup_tendon = torch.tensor(st, dtype=torch.float64, device=self.device)
        self._pair_a = torch.tensor([a for a, _ in ALL_PAIRS], device=self.device)
        self._pair_b = torch.tensor([b for _, b in ALL_PAIRS], device=self.device)
        self._lr_a = torch.tensor([a for a, _ in LR_PAIRS], device=self.device)
        self._lr_b = torch.tensor([b for _, b in LR_PAIRS], device=self.device)

    def _input(self, value, name, width):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
            raise TypeError(name + " must be a float64 torch tensor")
        if value.device != self.device:
            raise ValueError(name + " must be on the objective device")
        if value.ndim != 2 or value.shape[1] != width or value.shape[0] < 1:
            raise ValueError(name + " must have shape (T, " + str(width) + "), T>=1")
        _guard(value, name)

    def encode(self, q):
        """Return live weights, target, z plus finite confidence diagnostics."""
        self._input(q, "q", 42)
        delta = _guard(q.reshape(-1, 6, 7)[:, :, None] - self.curves[None], "anchor delta")
        squared = _guard(delta * delta, "squared anchor delta")
        distance = _guard(squared.sum(dim=-1), "anchor squared distance")
        logits = _guard(-distance / self.temperature[None, :, None], "Gaussian logits")
        weights = _guard(torch.softmax(logits, dim=-1), "Gaussian weights")
        target = _guard(torch.einsum("tls,lsj->tlj", weights, self.future_anchors)
                        .reshape(-1, 42), "successor target")
        z = _guard(torch.einsum("tls,sk->tlk", weights, self.phase_basis), "circular moment")
        diagnostic_weights = weights.detach()
        log_argument = torch.where(diagnostic_weights > 0, diagnostic_weights,
                                   torch.ones_like(diagnostic_weights))
        entropy = _guard(-(diagnostic_weights * log_argument.log()).sum(-1), "entropy")
        resultant = _guard(torch.linalg.vector_norm(z.detach(), dim=-1), "resultant length")
        return {"weights": weights, "target": target, "z": z, "R": resultant,
                "entropy": entropy, "maxweight": diagnostic_weights.max(-1).values,
                "zero_weight_count": (diagnostic_weights == 0).sum(-1), "distance_sq": distance}

    def evaluate_pairs(self, previous, next_q):
        self._input(previous, "previous", 42)
        self._input(next_q, "next_q", 42)
        if previous.shape != next_q.shape:
            raise ValueError("previous and next_q must have matching shapes")
        p, n = self.encode(previous), self.encode(next_q)
        residual = _guard((next_q - p["target"]).reshape(-1, 6, 7), "successor residual")
        square = _guard(residual * residual, "successor square")
        mse = _guard(square.mean(dim=-1), "successor MSE")
        normalized = _guard(mse / self.successor_normalization[None], "normalized successor")
        z = n["z"]
        pair = _guard(1 - (z[:, self._pair_a] * z[:, self._pair_b]).sum(-1), "coordination terms")
        lr = _guard(1 - (z[:, self._lr_a] * z[:, self._lr_b]).sum(-1), "LR coordination terms")
        per_leg = _guard(normalized.mean(0), "per-leg successor")
        successor = _guard(per_leg.mean(), "successor loss")
        coordination = _guard(pair.mean(), "coordination loss")
        loss = _guard(successor + coordination, "training loss")
        return {"training_loss": loss, "successor_loss": successor,
                "coordination_loss": coordination, "lr_coordination_loss": _guard(lr.mean(), "LR loss"),
                "per_leg_successor_loss": per_leg, "per_leg_successor_mse": mse.mean(0),
                "successor_terms": normalized, "coordination_terms": pair,
                "lr_coordination_terms": lr, "coordination_pair_loss": pair.mean(0),
                "successor_target": p["target"], "previous_encoding": p, "next_encoding": n}

    def evaluate(self, q, tendon):
        """Controls 1..300, exact startup99 and steady200 pairs/guards."""
        self._input(q, "q", 42)
        self._input(tendon, "tendon", 6)
        if q.shape != (300, 42) or tendon.shape != (300, 6):
            raise ValueError("evaluate requires q[300,42] and tendon[300,6]")
        result = self.evaluate_pairs(q[99:299], q[100:300])
        qe = _guard(q[:99] - self.startup_q, "startup q error")
        te = _guard(tendon[:99] - self.startup_tendon, "startup tendon error")
        qr = _guard(torch.linalg.vector_norm(qe) / math.sqrt(qe.numel()), "startup q RMS")
        tr = _guard(torch.linalg.vector_norm(te) / math.sqrt(te.numel()), "startup tendon RMS")
        qm, tm = _guard(qe.abs().max(), "startup q max"), _guard(te.abs().max(), "startup tendon max")
        result.update(startup_q_rms=qr, startup_q_max=qm, startup_tendon_rms=tr, startup_tendon_max=tm,
                      startup_pass=(qr <= .018) & (qm <= .04) & (tr <= .009) & (tm <= .025))
        return result
