"""Explicit mechanical capacity hypotheses for the BANC body interface."""

import math

import numpy as np

from .integrity import digest


def configure_adhesion_units(body, enabled=False):
    """Convert the original FlyBody cm force gain to the body's mm units.

    Mass and time are unchanged in FlyGym's conversion, so normal force scales
    by ten. Keep this explicit for compatibility with existing recordings.
    """
    if type(enabled) is not bool:
        raise ValueError("Boolean adhesion unit conversion required")
    if not enabled:
        return
    ids = np.asarray(body.sim._intern_adhesionactuatorids_by_fly[body.fly.name])
    old = body.m.actuator_gainprm[ids, 0].copy()
    if ids.shape != (6,) or not np.array_equal(old, np.full(6, .985)):
        raise ValueError("Six original FlyBody adhesion gains required")
    if np.any(body.m.actuator_trntype[ids] != body.mj.mjtTrn.mjTRN_BODY):
        raise ValueError("Native adhesion body transmission required")
    body.m.actuator_gainprm[ids, 0] = old * 10
    body.model_hash = digest({
        "schema": "flylab.banc-adhesion-cm-to-mm.v1",
        "base_body_hash": body.model_hash,
        "actuator_ids": ids.tolist(), "original_gain": old.tolist(),
        "force_scale": 10, "gain": body.m.actuator_gainprm[ids, 0].tolist(),
        "application": "after common settling, before first sensory observation",
    })
    body.mj.mj_forward(body.m, body.d)


def configure_finite_pad_contact(body, enabled=False):
    """Activate existing torsional/rolling coefficients on tarsal support pairs."""
    if type(enabled) is not bool:
        raise ValueError("Boolean finite pad contact required")
    if not enabled:
        return
    mapping, _ = body._force_maps['feet']
    a, b = body.m.pair_geom1, body.m.pair_geom2
    support = np.asarray(sorted(body.support_geom_ids))
    selected = ((mapping[a] >= 0) & np.isin(b, support)) | ((mapping[b] >= 0) & np.isin(a, support))
    ids = np.flatnonzero(selected)
    legs = {int(mapping[g]) // 5 for p in ids for g in (a[p], b[p]) if mapping[g] >= 0}
    if legs != set(range(6)) or np.any(body.m.pair_dim[ids] != 3):
        raise ValueError("Explicit three-dimensional tarsal contacts on all six legs required")
    body.m.pair_dim[ids] = 6
    body.model_hash = digest({
        "schema": "flylab.banc-finite-pad-contact.v1",
        "base_body_hash": body.model_hash,
        "pair_ids": ids.tolist(), "pair_dim": body.m.pair_dim.tolist(),
        "friction": body.m.pair_friction[ids].tolist(),
        "application": "after common settling, before first sensory observation",
    })
    # Contact forces are derived state. Refresh before first sensing so fresh
    # sessions and reconstructed checkpoints see the same contact definition.
    body.mj.mj_forward(body.m, body.d)


def configure_active_length(body, enabled=False):
    """Engineering active force-length response, with no measured length curve."""
    if type(enabled) is not bool:
        raise ValueError("Boolean active length model required")
    if not enabled:
        return
    gain = body.m.actuator_gainprm[body.act_ids, 0].copy()
    bias = body.m.actuator_biasprm[body.act_ids, 1].copy()
    if (np.any(gain <= 0) or not np.array_equal(bias, -gain)
            or np.any(body.m.actuator_biasprm[body.act_ids, 0])):
        raise ValueError("Zero-bias position servos required for active length")
    if hasattr(body, '_active_length_base_gain'):
        raise ValueError("Active length already configured")
    body._active_length_base_gain = gain
    body._active_length_base_bias = bias
    body.active_length_scale = np.ones(42)
    body.model_hash = digest({
        "schema": "flylab.banc-active-length.v1",
        "base_body_hash": body.model_hash,
        "equation": "K0*D-K0*(1+s)*(q-q0)-C*qdot; native control limits retained",
        "activation": "s=LP30ms[tanh(sum(abs(moment)*muscle_rate*motor_gain)/.5)]",
        "base_gain": gain.tolist(), "base_bias": bias.tolist(),
    })


def set_active_length_scale(body, scale):
    value = np.asarray(scale)
    if (value.shape != (42,) or value.dtype.kind not in 'fiu'
            or not np.isfinite(value).all() or np.any((value < 1) | (value > 2))):
        raise ValueError("42 finite active length scales in [1,2] required")
    if not hasattr(body, '_active_length_base_gain'):
        raise ValueError("Active length model is not configured")
    body.m.actuator_gainprm[body.act_ids, 0] = body._active_length_base_gain * value
    body.m.actuator_biasprm[body.act_ids, 1] = body._active_length_base_bias * value
    body.active_length_scale = value.copy()


def configure_activation_trajectory(body, tau_s=0.0):
    """Apply the existing muscle time constant at each physical substep."""
    if (type(tau_s) not in (int, float) or not math.isfinite(tau_s)
            or (tau_s != 0 and not .005 <= tau_s <= .25)):
        raise ValueError("Invalid continuous joint activation time constant")
    if tau_s == 0:
        return
    body.joint_target_activation_tau_s = tau_s
    body.model_hash = digest({
        "schema": "flylab.banc-continuous-joint-activation.v1",
        "base_body_hash": body.model_hash,
        "activation_tau_s": tau_s,
        "trajectory": "exact exponential between consecutive activated targets at every physical step",
    })


def configure_torque_capacity(body, torque_limit_bw_mm=0.0):
    """Limit hinge torque to body weight times a model-mm lever arm.

    Applied after the common neutral settling, before any neural control. Zero
    preserves the legacy model. This is an engineering capacity envelope, not
    a measured fly muscle strength or a change in friction/contact mechanics.
    """
    if (type(torque_limit_bw_mm) not in (int, float)
            or not math.isfinite(torque_limit_bw_mm)
            or not 0 <= torque_limit_bw_mm <= 100):
        raise ValueError("Invalid BANC joint torque capacity")
    if torque_limit_bw_mm == 0:
        return
    gears = body.m.actuator_gear[body.act_ids]
    if not np.array_equal(gears[:, 0], np.ones(42)) or np.any(gears[:, 1:]):
        raise ValueError("Unit hinge transmission required for torque capacity")
    if np.any(body.m.jnt_type[body.joint_ids] != body.mj.mjtJoint.mjJNT_HINGE):
        raise ValueError("Hinge joints required for torque capacity")
    capacity = body.weight0 * torque_limit_bw_mm
    body.m.actuator_forcelimited[body.act_ids] = True
    body.m.actuator_forcerange[body.act_ids] = (-capacity, capacity)
    body.model_hash = digest({
        "schema": "flylab.banc-joint-torque-capacity.v1",
        "base_body_hash": body.model_hash,
        "torque_limit_bw_mm": torque_limit_bw_mm,
        "forcelimited": body.m.actuator_forcelimited.tolist(),
        "forcerange": body.m.actuator_forcerange.tolist(),
        "application": "after common neutral settling, before neural control",
    })
