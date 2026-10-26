"""E329 offline, differentiable command-space temporal objective.

The reference phase is a nearest-polyline proxy, not an observed biological
phase. This module imports no neural model, body, controller, or runtime clock.
Only reference tensors are detached; previous commands and successor targets
retain their gradients. Branch switches and polyline knots need not be smooth.
"""
from __future__ import annotations

import math

import torch

FREQUENCY_HZ = 2.382760007990865
CONTROL_DT_S = .005
ANCHORS = 128
ALL_PAIRS = tuple((a, b) for a in range(6) for b in range(a + 1, 6))
LR_PAIRS = ((0, 3), (1, 4), (2, 5))


class PhaseFreeTemporalObjective:
    """Fixed E85 reference, 5ms successor, and all-15-pair coordination.

    Construction may validate references on the host. All hot tensor paths use
    asynchronous device predicates, with no scalar extraction or host copies.
    Argmin selects the first segment in a tie; strict projection clamp exterior
    has zero phase derivative. evaluate() expects controls 1..300, while
    evaluate_pairs() also serves a future 4-control reset-window runner.
    """

    def __init__(self, curves, startup_q, startup_tendon, *,
                 frequency=FREQUENCY_HZ, dt=CONTROL_DT_S, device="cpu"):
        if (isinstance(frequency, bool) or isinstance(dt, bool)
                or frequency != FREQUENCY_HZ or dt != CONTROL_DT_S):
            raise ValueError("frequency and dt must equal the fixed E329 constants")
        requested_device = torch.device(device)
        self.curves = self._reference(curves, "curves", requested_device)
        self.startup_q = self._reference(startup_q, "startup_q", requested_device)
        self.startup_tendon = self._reference(startup_tendon, "startup_tendon", requested_device)
        self.device = self.curves.device
        if self.curves.shape != (6, ANCHORS, 7):
            raise ValueError("curves must have shape (6, 128, 7)")
        if self.startup_q.shape != (99, 42) or self.startup_tendon.shape != (99, 6):
            raise ValueError("startup references must have shapes (99, 42) and (99, 6)")
        self.frequency = FREQUENCY_HZ
        self.dt = CONTROL_DT_S
        self.phase_increment = FREQUENCY_HZ * CONTROL_DT_S
        self._edge = torch.roll(self.curves, -1, 1) - self.curves
        self._edge_norm_sq = (self._edge * self._edge).sum(dim=-1)
        if not bool(torch.isfinite(self._edge).all()
                    & torch.isfinite(self._edge_norm_sq).all()
                    & (self._edge_norm_sq > 0).all()):
            raise ValueError("reference edges must have positive finite squared lengths")
        self._legs = torch.arange(6, device=self.device)
        self._segments = torch.arange(ANCHORS, device=self.device, dtype=torch.float64)
        self._pair_a = torch.tensor([a for a, _ in ALL_PAIRS], device=self.device)
        self._pair_b = torch.tensor([b for _, b in ALL_PAIRS], device=self.device)
        self._lr_a = torch.tensor([a for a, _ in LR_PAIRS], device=self.device)
        self._lr_b = torch.tensor([b for _, b in LR_PAIRS], device=self.device)
        anchors = (self._segments / ANCHORS)[:, None].expand(-1, 6)
        increment = self._sample(anchors + self.phase_increment) - self.curves.permute(1, 0, 2)
        self.successor_normalization = (increment * increment).mean(dim=(0, 2))
        if not bool(torch.isfinite(increment).all()
                    & torch.isfinite(self.successor_normalization).all()
                    & (self.successor_normalization > 0).all()):
            raise ValueError("successor normalization must be positive and finite")

    @staticmethod
    def _reference(value, name, device):
        if not hasattr(value, "dtype") or value.dtype not in (torch.float64, "float64"):
            raise TypeError(f"{name} must be float64")
        result = torch.as_tensor(value, device=device).detach().clone()
        if not bool(torch.isfinite(result).all()):
            raise ValueError(f"{name} must contain only finite values")
        return result

    def _input(self, value, name, width):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch tensor")
        if value.dtype != torch.float64:
            raise TypeError(f"{name} must be float64")
        if value.device != self.device:
            raise ValueError(f"{name} must be on the objective device")
        if value.ndim != 2 or value.shape[1] != width or value.shape[0] < 1:
            raise ValueError(f"{name} must have shape (T, {width}), T >= 1")
        torch._assert_async(torch.isfinite(value).all(), f"{name} must contain only finite values")

    def _sample(self, phase):
        coordinate = torch.remainder(phase, 1.0) * ANCHORS
        floor = torch.floor(coordinate)
        # A tiny negative f64 phase can have remainder rounded to exactly one.
        # Wrap its gather index, but use the original floor for the fraction.
        index = torch.remainder(floor.to(torch.long), ANCHORS)
        fraction = coordinate - floor
        return (self.curves[self._legs[None, :], index]
                + fraction[..., None] * self._edge[self._legs[None, :], index])

    def interpolate(self, phase):
        """Interpolate wrapped global phase[T,6] to command[T,42]."""
        self._input(phase, "phase", 6)
        result = self._sample(phase).reshape(-1, 42)
        torch._assert_async(torch.isfinite(result).all(), "interpolated command must be finite")
        return result

    def project(self, q):
        """Nearest closed-polyline projection and device-local branch diagnostics.

        distance_sq uses the seven-dimensional squared Euclidean distance;
        mean_square_distance divides it by seven for the old shape semantics.
        all_segment_phase includes each segment's own clamped projected phase.
        """
        self._input(q, "q", 42)
        x = q.reshape(-1, 6, 7)
        delta = x[:, :, None, :] - self.curves[None, :, :, :]
        dot = (delta * self._edge[None, :, :, :]).sum(dim=-1)
        raw_u = dot / self._edge_norm_sq[None, :, :]
        # Check before clamp/min so overflow cannot be hidden by saturation or
        # by selecting another segment whose distance happens to be finite.
        torch._assert_async(torch.isfinite(delta).all() & torch.isfinite(dot).all()
                            & torch.isfinite(raw_u).all(), "projection coefficients must be finite")
        u = raw_u.clamp(0.0, 1.0)
        nearest = self.curves[None, :, :, :] + u[..., None] * self._edge[None, :, :, :]
        distances = ((x[:, :, None, :] - nearest) ** 2).sum(dim=-1)
        torch._assert_async(torch.isfinite(nearest).all() & torch.isfinite(distances).all(),
                            "projection distances must be finite")
        segment = distances.argmin(dim=-1)
        selected = segment[..., None]
        selected_u = u.gather(-1, selected).squeeze(-1)
        selected_raw = raw_u.gather(-1, selected).squeeze(-1)
        selected_distance = distances.gather(-1, selected).squeeze(-1)
        phases = torch.remainder((self._segments[None, None, :] + u) / ANCHORS, 1.0)
        phase = phases.gather(-1, selected).squeeze(-1)
        selected_nearest = nearest.gather(
            2, selected[..., None].expand(-1, -1, 1, 7)).squeeze(2).reshape(-1, 42)
        return {"phase": phase, "segment": segment, "u": selected_u,
                "raw_u": selected_raw, "distance_sq": selected_distance,
                "mean_square_distance": selected_distance / 7,
                "nearest": selected_nearest, "all_segment_distance_sq": distances,
                "all_segment_phase": phases}

    def evaluate_pairs(self, previous, next_q):
        """Evaluate arbitrary adjacent control pairs without detaching previous.

        A 4-control window supplies concat(fixed_boundary, Q[:3]) and Q. Only
        that first boundary is fixed by its caller; every Q retains both roles.
        Coordination and the old shape term are evaluated on next_q.
        """
        self._input(previous, "previous", 42)
        self._input(next_q, "next_q", 42)
        if previous.shape != next_q.shape:
            raise ValueError("previous and next_q must have matching shapes")
        prev_projection = self.project(previous)
        next_projection = self.project(next_q)
        target = self.interpolate(prev_projection["phase"] + self.phase_increment)
        residual = (next_q - target).reshape(-1, 6, 7)
        mse = (residual * residual).mean(dim=-1)
        normalized = mse / self.successor_normalization[None, :]
        phase = next_projection["phase"]
        terms = 1 - torch.cos(2 * math.pi * (phase[:, self._pair_a] - phase[:, self._pair_b]))
        lr_terms = 1 - torch.cos(2 * math.pi * (phase[:, self._lr_a] - phase[:, self._lr_b]))
        torch._assert_async(torch.isfinite(residual).all() & torch.isfinite(mse).all()
                            & torch.isfinite(normalized).all()
                            & torch.isfinite(terms).all() & torch.isfinite(lr_terms).all(),
                            "temporal losses must be finite")
        per_leg = normalized.mean(dim=0)
        successor = per_leg.mean()
        coordination = terms.mean()
        loss = successor + coordination
        torch._assert_async(torch.isfinite(loss), "training loss must be finite")
        shape = next_projection["mean_square_distance"].mean(dim=0)
        return {"training_loss": loss, "successor_loss": successor,
                "per_leg_successor_loss": per_leg, "per_leg_successor_mse": mse.mean(dim=0),
                "coordination_loss": coordination, "lr_coordination_loss": lr_terms.mean(),
                "coordination_terms": terms, "lr_coordination_terms": lr_terms,
                "shape_loss": shape.mean(), "per_leg_shape_loss": shape,
                "successor_target": target, "previous_projection": prev_projection,
                "next_projection": next_projection}

    def evaluate(self, q, tendon):
        """Controls 1..300: 200 steady pairs and exact controls 1..99 startup."""
        self._input(q, "q", 42)
        self._input(tendon, "tendon", 6)
        if q.shape != (300, 42) or tendon.shape != (300, 6):
            raise ValueError("evaluate requires q[300,42] and tendon[300,6]")
        result = self.evaluate_pairs(q[99:299], q[100:300])
        q_error = q[:99] - self.startup_q
        t_error = tendon[:99] - self.startup_tendon
        q_rms = torch.linalg.vector_norm(q_error) / math.sqrt(q_error.numel())
        t_rms = torch.linalg.vector_norm(t_error) / math.sqrt(t_error.numel())
        q_max = q_error.abs().max()
        t_max = t_error.abs().max()
        torch._assert_async(torch.isfinite(q_rms) & torch.isfinite(q_max)
                            & torch.isfinite(t_rms) & torch.isfinite(t_max),
                            "startup metrics must be finite")
        result.update({"startup_q_rms": q_rms, "startup_q_max": q_max,
                       "startup_tendon_rms": t_rms, "startup_tendon_max": t_max,
                       "startup_pass": ((q_rms <= .018) & (q_max <= .04)
                                        & (t_rms <= .009) & (t_max <= .025))})
        return result
