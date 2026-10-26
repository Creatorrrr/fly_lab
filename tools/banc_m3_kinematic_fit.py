"""Differentiable static MuJoCo foot geometry for offline command calibration.

Only shadow mjData configurations are evaluated. No physical integration or
runtime trajectory generation is performed by this training helper.
"""

import numpy as np
import torch


class StaticFootGeometry:
    def __init__(self, body, qpos):
        self.m = body.m
        self.mj = body.mj
        self.qpos_ids = body.qpos_ids.copy()
        self.qvel_ids = body.qvel_ids.copy()
        self.tips = body._tip_ids.copy()
        self.thorax = body.thorax
        self.qpos = np.asarray(qpos, dtype=float).copy()
        if self.qpos.ndim != 2 or self.qpos.shape[1] != self.m.nq or not np.isfinite(self.qpos).all():
            raise ValueError('Finite fixed reference qpos contexts required')
        self.data = self.mj.MjData(self.m)
        self.jacobian = np.zeros((3, self.m.nv))

    def evaluate(self, commands, *, jacobian=False):
        commands = np.asarray(commands, dtype=float)
        if commands.shape != (len(self.qpos), 42) or not np.isfinite(commands).all():
            raise ValueError('One finite42joint command per frozen kinematic context required')
        points = np.empty((len(commands), 6, 3))
        derivatives = np.empty((len(commands), 6, 3, 42)) if jacobian else None
        for row, command in enumerate(commands):
            self.data.qpos[:] = self.qpos[row]
            self.data.qpos[self.qpos_ids] = command
            self.mj.mj_kinematics(self.m, self.data)
            self.mj.mj_comPos(self.m, self.data)
            origin = self.data.xpos[self.thorax]
            rotation = self.data.xmat[self.thorax].reshape(3, 3)
            points[row] = (self.data.xpos[self.tips] - origin) @ rotation
            if jacobian:
                for leg, tip in enumerate(self.tips):
                    self.mj.mj_jacBody(self.m, self.data, self.jacobian, None, int(tip))
                    derivatives[row, leg] = rotation.T @ self.jacobian[:, self.qvel_ids]
        return (points, derivatives) if jacobian else points

    def tensor(self, command):
        return _StaticFootFunction.apply(command, self)


class _StaticFootFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, commands, geometry):
        points, jacobian = geometry.evaluate(commands.detach().cpu().numpy(), jacobian=True)
        ctx.save_for_backward(torch.as_tensor(jacobian, dtype=commands.dtype, device=commands.device))
        return torch.as_tensor(points, dtype=commands.dtype, device=commands.device)

    @staticmethod
    def backward(ctx, gradient):
        jacobian, = ctx.saved_tensors
        return torch.einsum('tlx,tlxj->tj', gradient, jacobian), None
