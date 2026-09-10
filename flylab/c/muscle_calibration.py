"""Geometry diagnostics independent of muscle excitation and neural readout.

No tendon rerouting, range clamp or sign correction is performed. A tension
actuator has negative force in MuJoCo, so its torque sign is minus d(length)/dq.
"""
import xml.etree.ElementTree as ET
import numpy as np
from .integrity import bounded_int, finite
from .muscles import JOINT, MUSCLES, moment_matrix, reference_model


def direction_status(frame, tolerance=1e-9):
    arms = np.asarray(frame['moment_arm_mm'], dtype=float)
    if arms.shape != (2,) or not np.isfinite(arms).all():
        raise ValueError('Invalid flexor/extensor moment arms')
    expected = np.array([-1., 1.])
    reversed_names = [name for name, arm in zip(MUSCLES, arms*expected) if arm < -tolerance]
    singular_names = [name for name, arm in zip(MUSCLES, arms) if abs(arm) <= tolerance]
    return dict(status='REVERSED' if reversed_names else 'SINGULAR' if singular_names else 'CONSISTENT',
                reversed_muscles=reversed_names, singular_muscles=singular_names,
                convention='actuator moment = dL/dq; tensile muscle force is negative')


def site_path_geometry(mj, model, data, source):
    """Analytic site-line Jacobian, without reading actuator_moment/ten_J."""
    root = ET.parse(source).getroot()
    j = model.joint(JOINT).id
    anchor, axis = data.xanchor[j], data.xaxis[j]
    moving_body = model.jnt_bodyid[j]
    def moves(body):
        while body:
            if body == moving_body: return True
            body = model.body_parentid[body]
        return False
    paths = []
    for name in MUSCLES:
        tendon = root.find(f'./tendon/spatial[@name="{name}_tendon"]')
        if tendon is None or any(child.tag != 'site' for child in tendon):
            raise ValueError('Analytic calibration requires a site-only tendon')
        ids = [model.site(child.get('site')).id for child in tendon]
        p = data.site_xpos[ids]
        velocity = np.array([np.cross(axis, point-anchor) if moves(model.site_bodyid[i]) else np.zeros(3)
                             for i, point in zip(ids, p)])
        segments = np.diff(p, axis=0); lengths = np.linalg.norm(segments, axis=1)
        if np.any(lengths < 1e-12): raise ValueError('Coincident tendon sites')
        derivative = np.sum(segments/lengths[:, None]*np.diff(velocity, axis=0))
        paths.append(dict(muscle=name, sites=[model.site(i).name for i in ids],
            bodies=[model.body(model.site_bodyid[i]).name for i in ids],
            points_mm=p.tolist(), segment_lengths_mm=lengths.tolist(),
            length_mm=float(lengths.sum()), derivative_mm=float(derivative)))
    return paths


def interior_angle(model, data):
    """Angle from physical segment origins, rather than assuming pi - joint q."""
    p = [data.xpos[model.body(name).id] for name in ('LFFemur', 'LFTibia', 'LFTarsus1')]
    a, b = p[0]-p[1], p[2]-p[1]
    return float(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)), -1., 1.)))


def sweep(rig, samples=401, derivative_step=1e-6):
    """Original multi-joint vs fixed fixture, analytic vs finite-difference arms."""
    bounded_int(samples, 'geometry samples', 3, 10001)
    h = finite(derivative_step, 'derivative step', 1e-8, 1e-3)
    mj, model = rig.mj, rig.model
    _, reference, original = reference_model(mj, rig.source)
    j = reference.joint(JOINT).id
    qadr, dof = reference.jnt_qposadr[j], reference.jnt_dofadr[j]
    candidate = mj.MjData(model)
    rows = []
    for angle in np.linspace(*model.jnt_range[0], samples):
        candidate.qpos[0] = original.qpos[qadr] = angle
        mj.mj_forward(model, candidate); mj.mj_forward(reference, original)
        paths = site_path_geometry(mj, model, candidate, rig.source)
        arms = moment_matrix(mj, model, candidate)[:, 0]
        lengths = []
        for sign in (-1., 1.):
            candidate.qpos[0] = angle+sign*h; mj.mj_forward(model, candidate)
            lengths.append(candidate.actuator_length.copy())
        candidate.qpos[0] = angle; mj.mj_forward(model, candidate)
        analytic = np.array([p['derivative_mm'] for p in paths])
        rows.append(dict(q_rad=float(angle), interior_angle_rad=interior_angle(model, candidate),
            moment_arm_mm=arms.tolist(), analytic_arm_mm=analytic.tolist(),
            finite_difference_arm_mm=((lengths[1]-lengths[0])/(2*h)).tolist(),
            reference_arm_mm=moment_matrix(mj, reference, original)[:, dof].tolist(),
            length_mm=candidate.actuator_length.tolist(), paths=paths,
            pose_error_mm=float(np.max(np.abs(original.xpos-candidate.xpos))),
            orientation_error=float(np.max(np.abs(original.xmat-candidate.xmat)))))
    roots = []
    for i, name in enumerate(MUSCLES):
        for a, b in zip(rows[:-1], rows[1:]):
            if a['moment_arm_mm'][i]*b['moment_arm_mm'][i] >= 0: continue
            low, high, left = a['q_rad'], b['q_rad'], a['moment_arm_mm'][i]
            for _ in range(45):
                mid = (low+high)/2; candidate.qpos[0] = mid; mj.mj_forward(model, candidate)
                v = moment_matrix(mj, model, candidate)[i, 0]
                if left*v > 0: low = mid
                else: high = mid
            roots.append(dict(muscle=name, zero_angle_rad=(low+high)/2))
    arms = np.array([r['moment_arm_mm'] for r in rows])
    errors = {label:float(np.max(np.abs(arms-np.array([r[key] for r in rows]))))
              for label, key in [('analytic', 'analytic_arm_mm'), ('finite_difference', 'finite_difference_arm_mm'),
                                  ('reference', 'reference_arm_mm')]}
    errors.update(position=max(r['pose_error_mm'] for r in rows),
                  orientation=max(r['orientation_error'] for r in rows))
    checks = dict(analytic_agreement=errors['analytic'] < 1e-9,
                  finite_difference_agreement=errors['finite_difference'] < 1e-8,
                  source_geometry_preserved=max(errors['reference'], errors['position'], errors['orientation']) < 1e-9,
                  positive_q_flexes=all(b['interior_angle_rad'] < a['interior_angle_rad'] for a,b in zip(rows[:-1],rows[1:])))
    return dict(schema='flylab.muscle-geometry.v1', model_hash=rig.identity, samples=rows,
        checks=checks, errors=errors, zero_crossings=roots,
        status='PASS' if all(checks.values()) else 'FAIL',
        physiological_status='CALIBRATION_REQUIRED',
        diagnosis='Moment reversal reproduced by the original site routing, independent analytic geometry and numerical length derivative.',
        biological_validation=False, unit_contract=rig.metadata['unit_contract'])
