"""E319 generated-tensor, phase-free within-leg closed-curve objective.

This is an offline command-space loss. A low value alone says nothing about
traversal direction, cadence, inter-leg coordination, or physical gait.
"""

import math

import torch


class PhaseFreeShapeObjective:
    """Measure distance to each leg's closed reference polyline and startup drift."""

    def __init__(self, curves, startup_q, startup_tendon, *, device="cpu"):
        requested_device = torch.device(device)
        self.curves = self._reference(curves, "curves", requested_device)
        self.startup_q = self._reference(startup_q, "startup_q", requested_device)
        self.startup_tendon = self._reference(startup_tendon, "startup_tendon", requested_device)
        # Resolve an index-free request such as "cuda" to the owned tensor's
        # concrete device (for example cuda:0) exactly once at construction.
        self.device = self.curves.device
        if (self.curves.ndim != 3 or self.curves.shape[0] != 6
                or not 3 <= self.curves.shape[1] <= 512 or self.curves.shape[2] != 7):
            raise ValueError("curves must have shape (6, M, 7), with 3 <= M <= 512")
        if self.startup_q.shape != (99, 42):
            raise ValueError("startup_q must have shape (99, 42)")
        if self.startup_tendon.shape != (99, 6):
            raise ValueError("startup_tendon must have shape (99, 6)")
        self._start = self.curves
        self._edge = torch.roll(self.curves, shifts=-1, dims=1) - self.curves
        self._edge_norm_sq = (self._edge * self._edge).sum(dim=-1)
        if not bool(torch.isfinite(self._edge).all() & torch.isfinite(self._edge_norm_sq).all()):
            raise ValueError("curves produce nonfinite segment coefficients")
        self._safe_edge_norm_sq = torch.where(
            self._edge_norm_sq > 0, self._edge_norm_sq,
            torch.ones_like(self._edge_norm_sq),
        )

    @staticmethod
    def _reference(value, name, device):
        # as_tensor accepts generated NumPy references as well as torch tensors.
        if not hasattr(value, "dtype") or value.dtype not in (torch.float64, "float64"):
            # NumPy's dtype compares equal to the string "float64".
            raise TypeError(f"{name} must be float64")
        reference = torch.as_tensor(value, device=device).detach().clone()
        if not bool(torch.isfinite(reference).all()):
            raise ValueError(f"{name} must contain only finite values")
        return reference

    def evaluate(self, q, tendon):
        """Return differentiable losses and the exact control 1..99 startup gate.

        Tensor predicates stay on the input device. In particular, the hot
        evaluate path has no scalar extraction or host copy for CUDA inputs.
        """
        if not isinstance(q, torch.Tensor) or not isinstance(tendon, torch.Tensor):
            raise TypeError("q and tendon must be torch tensors")
        if q.dtype != torch.float64 or tendon.dtype != torch.float64:
            raise TypeError("q and tendon must be float64")
        if q.device != self.device or tendon.device != self.device:
            raise ValueError("q and tendon must be on the objective device")
        if q.ndim != 2 or q.shape[1] != 42 or not 101 <= q.shape[0] <= 320:
            raise ValueError("q must have shape (T, 42), with 101 <= T <= 320")
        if tendon.shape != (q.shape[0], 6):
            raise ValueError("tendon must have shape (T, 6)")
        torch._assert_async(torch.isfinite(q).all(), "q must contain only finite values")
        torch._assert_async(torch.isfinite(tendon).all(), "tendon must contain only finite values")

        # q[100] is control 101. Broadcast over the M segments for each leg.
        x = q[100:].reshape(-1, 6, 7)
        delta = x[:, :, None, :] - self._start[None, :, :, :]
        dot = (delta * self._edge[None, :, :, :]).sum(dim=-1)
        u = (dot / self._safe_edge_norm_sq[None, :, :]).clamp(0.0, 1.0)
        nearest = self._start[None, :, :, :] + u[..., None] * self._edge[None, :, :, :]
        dist_sq = ((x[:, :, None, :] - nearest) ** 2).sum(dim=-1) / 7.0
        per_leg = dist_sq.min(dim=-1).values.mean(dim=0)

        q_error = q[:99] - self.startup_q
        tendon_error = tendon[:99] - self.startup_tendon
        q_abs = q_error.abs()
        tendon_abs = tendon_error.abs()
        q_rms = torch.linalg.vector_norm(q_error) / math.sqrt(q_error.numel())
        q_max = q_abs.max()
        tendon_rms = torch.linalg.vector_norm(tendon_error) / math.sqrt(tendon_error.numel())
        tendon_max = tendon_abs.max()
        # A finite input may still overflow in projection or subtraction. The
        # asynchronous predicates preserve the CUDA hot path's device locality.
        torch._assert_async(torch.isfinite(per_leg).all(), "shape loss must be finite")
        torch._assert_async(torch.isfinite(q_rms) & torch.isfinite(q_max),
                            "startup q metrics must be finite")
        torch._assert_async(torch.isfinite(tendon_rms) & torch.isfinite(tendon_max),
                            "startup tendon metrics must be finite")
        startup_pass = ((q_rms <= .018) & (q_max <= .04)
                        & (tendon_rms <= .009) & (tendon_max <= .025))
        return {
            "shape_loss": per_leg.mean(),
            "per_leg_shape_loss": per_leg,
            "startup_q_rms": q_rms,
            "startup_q_max": q_max,
            "startup_tendon_rms": tendon_rms,
            "startup_tendon_max": tendon_max,
            "startup_pass": startup_pass,
        }
