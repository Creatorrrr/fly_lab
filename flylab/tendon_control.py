"""Named fixed-tendon motor inputs. No muscle or neural mapping is implied."""

from collections.abc import Mapping

import numpy as np

from .common import number


class TendonControl:
    def __init__(self, body):
        self.body = body
        m, mj = body.m, body.mj
        self.act_ids = np.asarray(
            body.sim._intern_actuatorids_by_type_by_fly[body.ActuatorType.TENDON][
                body.fly.name
            ],
            dtype=int,
        )
        self.tendon_ids = m.actuator_trnid[self.act_ids, 0]
        self.names = [
            mj.mj_id2name(m, mj.mjtObj.mjOBJ_TENDON, int(t)).split("/")[-1]
            for t in self.tendon_ids
        ]
        self.index = {name: i for i, name in enumerate(self.names)}
        self.limits = m.actuator_ctrlrange[self.act_ids].copy()
        self.joints = []
        for t in self.tendon_ids:
            a, n = int(m.tendon_adr[t]), int(m.tendon_num[t])
            if np.any(m.wrap_type[a : a + n] != mj.mjtWrap.mjWRAP_JOINT):
                raise ValueError("Only fixed joint tendons are supported")
            self.joints.append(m.wrap_objid[a : a + n].copy())
        self.joint_ids = np.unique(np.concatenate(self.joints)).astype(int)
        if len(self.index) != len(self.names) or not len(self.names):
            raise ValueError("Ambiguous tendon controls")
        if not np.all(m.actuator_ctrllimited[self.act_ids]):
            raise ValueError("Tendon motor limits are required")
        position_ids = body.sim._intern_actuatorids_by_type_by_fly[
            body.ActuatorType.POSITION
        ][body.fly.name]
        if set(m.actuator_trnid[position_ids, 0]) & set(self.joint_ids):
            raise ValueError("Position servos overlap tendon-driven joints")

    def command_vector(self, inputs):
        if not isinstance(inputs, Mapping) or not inputs:
            raise ValueError("Provide nonempty named tendon motor inputs")
        if any(name not in self.index for name in inputs):
            raise ValueError("Unknown tendon name")
        values = self.body.d.ctrl[self.act_ids].copy()
        for name, value in inputs.items():
            i = self.index[name]
            values[i] = number(value, name, *self.limits[i])
        return values

    def observation(self):
        b = self.body
        return {
            "schema": "flylab.tendons.v1",
            "names": self.names.copy(),
            "inputs": b.d.ctrl[self.act_ids].tolist(),
            "limits": self.limits.tolist(),
            "lengths": b.d.ten_length[self.tendon_ids].tolist(),
            "velocities": b.d.ten_velocity[self.tendon_ids].tolist(),
            "actuator_forces": b.d.actuator_force[self.act_ids].tolist(),
            "joint_groups": [
                [b.mj.mj_id2name(b.m, b.mj.mjtObj.mjOBJ_JOINT, int(j)) for j in group]
                for group in self.joints
            ],
            "angles_rad": [
                b.d.qpos[b.m.jnt_qposadr[group]].tolist() for group in self.joints
            ],
            "torques": [
                b.d.qfrc_actuator[b.m.jnt_dofadr[group]].tolist()
                for group in self.joints
            ],
            "actuator_model": "fixed-tendon motor; input is not a joint angle or Hill muscle activation",
            "neural_mapping": "unmapped",
            "biological_validation": False,
        }
