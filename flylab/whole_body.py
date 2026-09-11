"""Explicit whole-body servo interface for the composed FlyBody model.

Position servos are engineering actuators, not reconstructed muscle physiology.
Only ctrl is commanded. The floating root and passive joints remain physical.
All held commands live in MuJoCo's integration state and follow its checkpoint.
"""

from collections.abc import Mapping

import numpy as np

from . import PHYSICS_DT
from .common import number


def configure_tracking_servos(body):
    """Set engineering gains from compiled springs, including spring-dampers.

    The source MjsJoint stiffness can still be zero when MuJoCo will derive a
    nonzero stiffness from springdamper. Read the compiled physical parameters.
    This changes actuator gain/bias only, before initial settling and hashing.
    """
    m = body.m
    ids = body.sim._intern_actuatorids_by_type_by_fly[body.ActuatorType.POSITION][
        body.fly.name
    ]
    for dof, actuator in zip(body.full_order, ids):
        is_leg = dof.child.is_leg()
        if is_leg and body.body_options.servo_profile != "tracking_all":
            continue
        joint = m.actuator_trnid[actuator, 0]
        velocity = m.jnt_dofadr[joint]
        margin = 49 if dof.child.link.startswith("abdomen") else 19
        kp = max(m.actuator_gainprm[actuator, 0], margin * m.jnt_stiffness[joint])
        if is_leg:
            kp = max(kp, 1000.0)
        kv = max(
            0.0, 2 * np.sqrt(kp * m.dof_armature[velocity]) - m.dof_damping[velocity]
        )
        # Euler integrates actuator velocity feedback explicitly. Bound that
        # feedback at the declared step instead of changing body inertia.
        kv = min(kv, 0.5 * m.dof_armature[velocity] / m.opt.timestep)
        m.actuator_gainprm[actuator, 0] = kp
        m.actuator_biasprm[actuator, 1:3] = [-kp, -kv]


def apply_leg_action(sim, name, action, indices):
    """Allow the existing 42-DOF path to coexist with held nonleg commands."""
    from flygym.compose import ActuatorType

    ids = sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][name]
    values = sim.mj_data.ctrl[ids].copy()
    values[indices] = action.joint_angles
    sim.set_actuator_inputs(name, ActuatorType.POSITION, values)
    if action.adhesion_onoff is not None:
        sim.set_leg_adhesion_states(name, action.adhesion_onoff)


def region(dof):
    child = dof.child
    if child.is_leg():
        return child.pos
    link = child.link
    if link.startswith("abdomen"):
        return "abdomen"
    if link in ("rostrum", "haustellum", "labrum"):
        return "mouth"
    return link


class WholeBodyControl:
    def __init__(self, body):
        self.body = body
        self.order = list(body.full_order)
        self.names = [dof.name for dof in self.order]
        if len(self.names) != len(set(self.names)):
            raise ValueError("Duplicate full-body joint names")
        self.index = {name: i for i, name in enumerate(self.names)}
        self.act_ids = np.asarray(
            body.sim._intern_actuatorids_by_type_by_fly[body.ActuatorType.POSITION][
                body.fly.name
            ]
        )
        self.joint_ids = body.m.actuator_trnid[self.act_ids, 0]
        self.qpos_ids = body.m.jnt_qposadr[self.joint_ids]
        self.qvel_ids = body.m.jnt_dofadr[self.joint_ids]
        self.limits = body.m.actuator_ctrlrange[self.act_ids].copy()
        for i, jid in enumerate(self.joint_ids):
            if not body.m.actuator_ctrllimited[self.act_ids[i]]:
                raise ValueError("Whole-body servos must declare control limits")
            if body.m.jnt_limited[jid]:
                self.limits[i, 0] = max(self.limits[i, 0], body.m.jnt_range[jid, 0])
                self.limits[i, 1] = min(self.limits[i, 1], body.m.jnt_range[jid, 1])
        if not np.isfinite(self.limits).all() or np.any(
            self.limits[:, 0] >= self.limits[:, 1]
        ):
            raise ValueError("Invalid whole-body actuator limits")
        active = set(self.joint_ids.tolist())
        self.passive_ids = np.array(
            [
                j
                for j in range(body.m.njnt)
                if j not in active
                and body.m.jnt_type[j] == body.mj.mjtJoint.mjJNT_HINGE
            ],
            dtype=int,
        )
        self.neutral = body.d.ctrl[self.act_ids].copy()

    def command_vector(self, targets):
        if not isinstance(targets, Mapping) or not targets:
            raise ValueError("Provide nonempty named whole-body targets in radians")
        if any(name not in self.index for name in targets):
            raise ValueError("Unknown whole-body joint target")
        values = self.body.d.ctrl[self.act_ids].copy()
        for name, value in targets.items():
            i = self.index[name]
            values[i] = number(value, name, *self.limits[i])
        return values

    def step(self, targets, adhesion, dt):
        # Validate the entire request before changing any physical/control state.
        values = self.command_vector(targets)
        number(dt, "body control dt", PHYSICS_DT, 0.05)
        if abs(round(dt / PHYSICS_DT) * PHYSICS_DT - dt) > 1e-10:
            raise ValueError("Integral physical steps required")
        if adhesion is None:
            adhesion = self.body.last_action.adhesion_onoff.copy()
        adhesion = np.asarray(adhesion)
        if adhesion.shape != (6,) or adhesion.dtype != np.dtype(bool):
            raise ValueError("Six boolean adhesion states required")
        if self.body.fault:
            raise RuntimeError(self.body.fault)
        self.body.d.ctrl[self.act_ids] = values
        self.body.step_joint_targets(values[self.body.leg_action_indices], adhesion, dt)

    def observation(self):
        b = self.body
        passive_q = b.m.jnt_qposadr[self.passive_ids]
        passive_v = b.m.jnt_dofadr[self.passive_ids]
        return {
            "schema": "flylab.whole-body.v1",
            "time_s": b.physics_time(),
            "model_hash": b.model_hash,
            "names": self.names.copy(),
            "regions": [region(dof) for dof in self.order],
            "angles_rad": b.d.qpos[self.qpos_ids].tolist(),
            "velocities_rad_s": b.d.qvel[self.qvel_ids].tolist(),
            "targets_rad": b.d.ctrl[self.act_ids].tolist(),
            "limits_rad": self.limits.tolist(),
            "actuator_force": b.d.actuator_force[self.act_ids].tolist(),
            "actuator_torque": b.d.qfrc_actuator[self.qvel_ids].tolist(),
            "passive": {
                "names": [
                    b.mj.mj_id2name(b.m, b.mj.mjtObj.mjOBJ_JOINT, int(j))
                    for j in self.passive_ids
                ],
                "angles_rad": b.d.qpos[passive_q].tolist(),
                "velocities_rad_s": b.d.qvel[passive_v].tolist(),
            },
            "root_actuated": False,
            "root_position_mm": b.d.xpos[b.thorax].tolist(),
            "root_rotation": b.d.xmat[b.thorax].reshape(3, 3).tolist(),
            "actuator_model": "engineering position servos; gains are not measured muscle physiology",
            "servo_profile": b.body_options.servo_profile,
            "servo_kp": b.m.actuator_gainprm[self.act_ids, 0].tolist(),
            "servo_kv": (-b.m.actuator_biasprm[self.act_ids, 2]).tolist(),
            "torque_unit": "MuJoCo model units; not SI calibrated",
            "biological_validation": False,
            "neural_mapping_validated": False,
        }
