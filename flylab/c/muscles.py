"""Isolated Fe–Ti force rig derived from the pinned FlyMimic MJCF.

This is a calibration fixture, not the BANC walking body. All joints except
the left tibia are fixed at the upstream keyframe. Explicit body inertias and
the two routed Hill muscles are retained; meshes and contact are excluded.
The derivation checks the starting pose, tendon lengths and moment arms.
"""
from importlib.metadata import version
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .integrity import bounded_int, boolean, digest, file_hash, finite

JOINT = 'joint_LFTibia_pitch'
MUSCLES = ('LFTibia_flex_93434', 'LFTibia_extensor_93932')
SOURCE = 'https://neuromechfly.org/tutorials/6_muscle_imitation/'


def moment_matrix(mj, model, data):
    """MuJoCo 3.9 stores actuator moment arms as a sparse row matrix."""
    value = data.actuator_moment
    if value.shape == (model.nu, model.nv):
        return value.copy()
    dense = np.zeros((model.nu, model.nv))
    mj.mju_sparse2dense(dense, value, data.moment_rownnz,
                       data.moment_rowadr, data.moment_colind)
    return dense


def reference_model(mj, source):
    """Contact-free source with its original joints, for independent pose sweeps."""
    root = ET.parse(source).getroot()
    root.find('compiler').set('inertiafromgeom', 'false')
    # The shipped 100 us Euler step can overshoot activation to 2 at u=1:
    # its fastest activation time is 50 us. Resolve that time scale here.
    root.find('option').set('timestep', '0.00001')
    for body in root.findall('.//body'):
        if any(g.get('type') == 'mesh' for g in body.findall('geom')) and body.find('inertial') is None:
            raise ValueError('Mesh-dependent inertia cannot be removed: ' + str(body.get('name')))
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'mesh' or child.tag == 'geom' and child.get('type') == 'mesh':
                parent.remove(child)
    for geom in root.findall('.//geom'):
        geom.set('contype', '0'); geom.set('conaffinity', '0')
    for tag in ('contact', 'equality', 'sensor'):
        for element in root.findall(tag): root.remove(element)
    for actuator in list(root.find('actuator')):
        if actuator.get('name') not in MUSCLES: root.find('actuator').remove(actuator)
    for tendon in list(root.find('tendon')):
        if tendon.get('name') not in [n+'_tendon' for n in MUSCLES]: root.find('tendon').remove(tendon)
    if len(root.find('actuator')) != 2 or len(root.find('tendon')) != 2:
        raise ValueError('Expected the upstream tibia flexor/extensor pair')
    original = mj.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    initial = mj.MjData(original)
    mj.mj_resetDataKeyframe(original, initial, original.key('default-pose').id)
    mj.mj_forward(original, initial)
    return root, original, initial


def derive_fixture(mj, source):
    root, original, initial = reference_model(mj, source)
    joint = original.joint(JOINT).id
    q0 = float(initial.qpos[original.jnt_qposadr[joint]])
    for body in root.findall('.//body'):
        joints = body.findall('joint')
        if not joints or any(j.get('name') == JOINT for j in joints): continue
        index = original.body(body.get('name')).id
        parent = original.body_parentid[index]
        rotation = initial.xmat[parent].reshape(3, 3)
        relative = rotation.T @ initial.xmat[index].reshape(3, 3)
        quat = np.empty(4); mj.mju_mat2Quat(quat, relative.reshape(9))
        body.set('pos', ' '.join(map(str, rotation.T @ (initial.xpos[index]-initial.xpos[parent]))))
        for attr in ('euler', 'axisangle', 'xyaxes', 'zaxis'): body.attrib.pop(attr, None)
        body.set('quat', ' '.join(map(str, quat)))
        for j in joints: body.remove(j)
    key = root.find('./keyframe/key')
    key.set('qpos', str(q0))
    for attr in ('qvel', 'act', 'ctrl'): key.attrib.pop(attr, None)
    xml = ET.tostring(root, encoding='unicode')
    model = mj.MjModel.from_xml_string(xml); data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, model.key('default-pose').id)
    mj.mj_forward(model, data)
    errors = dict(position=float(np.max(np.abs(data.xpos-initial.xpos))),
                  orientation=float(np.max(np.abs(data.xmat-initial.xmat))),
                  tendon_length=float(np.max(np.abs(data.actuator_length-initial.actuator_length))),
                  moment_arm=float(np.max(np.abs(moment_matrix(mj, model, data)[:, 0]-
                      moment_matrix(mj, original, initial)[:, original.jnt_dofadr[joint]]))))
    if model.nv != 1 or model.nu != 2 or model.na != 2 or max(errors.values()) > 1e-9:
        raise ValueError('Fixture derivation changed the reference geometry: ' + str(errors))
    return model, data, xml, errors


class MuscleRig:
    def __init__(self, source=None):
        import mujoco as mj
        if source is None:
            from flygym.compose.fly.musculoskeletal import DEFAULT_MUSCULOSKELETAL_XML
            source = DEFAULT_MUSCULOSKELETAL_XML
        self.mj = mj; self.source = Path(source)
        self.model, self.data, self.xml, errors = derive_fixture(mj, self.source)
        self.metadata = dict(schema='flylab.muscle-fixture.v1', source_url=SOURCE,
            source_xml_sha256=file_hash(self.source), derived_xml_sha256=digest(self.xml),
            flygym_version=version('flygym'), mujoco_version=mj.__version__,
            physics_dt=float(self.model.opt.timestep), muscles=list(MUSCLES), joint=JOINT,
            derivation_error=errors, fixture='fixed-body-single-joint-no-contact',
            unit_contract=dict(time='s', angle='rad', length='mm',
                force='upstream model force units', torque='upstream model torque units',
                force_to_SI=None, torque_to_SI=None),
            parameter_evidence='upstream converted model; physiological force calibration pending',
            timestep_change='10 us fixture step; upstream 100 us can overshoot muscle activation',
            neural_binding=None, neural_control=False, biological_validation=False)
        self.identity = digest(self.metadata); self.tick = 0
        self.act_ids = np.array([self.model.actuator(n).id for n in MUSCLES])
        self.act_adr = self.model.actuator_actadr[self.act_ids]
        self.requested = np.zeros(2); self.connected = True
        self._set_control(self.requested, self.connected)
        mj.mj_forward(self.model, self.data)

    def _set_control(self, requested, connected):
        floor, ceiling = self.model.actuator_ctrlrange[self.act_ids].T
        self.data.ctrl[self.act_ids] = np.clip(requested if connected else np.zeros(2), floor, ceiling)

    def step(self, excitation=(0., 0.), *, steps=10, torque=0., connected=True, direction_guard=False):
        excitation = np.asarray(excitation, dtype=float)
        if excitation.shape != (2,) or not np.isfinite(excitation).all() or np.any((excitation < 0) | (excitation > 1)):
            raise ValueError('Two finite muscle excitations in 0..1 required')
        bounded_int(steps, 'muscle steps', 1, 10000); bounded_int(self.tick+steps, 'muscle tick', 0, 2**31-1)
        torque = finite(torque, 'external joint torque (model units)', -1., 1.)
        boolean(connected, 'motor connection')
        boolean(direction_guard, 'muscle direction guard')
        self.requested = excitation.copy(); self.connected = connected
        self._set_control(excitation, connected); self.data.qfrc_applied[0] = torque
        for _ in range(steps):
            self.mj.mj_step(self.model, self.data); self.tick += 1
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError('FAULT_MUSCLE_PHYSICS_NONFINITE')
            if direction_guard:
                # Evaluate the new pose every physical tick, not only at the
                # control boundary: a reversal can disappear before sampling.
                from .muscle_calibration import direction_status
                self.mj.mj_forward(self.model, self.data)
                arms = moment_matrix(self.mj, self.model, self.data)[self.act_ids, 0]
                if direction_status(dict(moment_arm_mm=arms))['status'] != 'CONSISTENT':
                    raise RuntimeError('FAULT_MUSCLE_DIRECTION')
        self.mj.mj_forward(self.model, self.data)
        return self.frame()

    def frame(self):
        m, d, mj = self.model, self.data, self.mj
        moment = moment_matrix(mj, m, d)[self.act_ids, 0]
        force = d.actuator_force[self.act_ids]
        passive = np.array([mj.mju_muscleBias(d.actuator_length[i], m.actuator_lengthrange[i],
            m.actuator_acc0[i], m.actuator_biasprm[i, :9]) for i in self.act_ids])
        inertia = np.empty((m.nv, m.nv)); mj.mj_fullM(m, inertia, d.qM)
        rhs = d.qfrc_actuator+d.qfrc_passive+d.qfrc_applied+d.qfrc_constraint-d.qfrc_bias
        return dict(tick=self.tick, seconds=self.tick*m.opt.timestep, model_hash=self.identity,
            q_rad=float(d.qpos[0]), qdot_rad_s=float(d.qvel[0]), qacc_rad_s2=float(d.qacc[0]),
            axis_world=d.xaxis[0].tolist(), joint_range_rad=m.jnt_range[0].tolist(),
            requested_excitation=self.requested.tolist(), motor_connected=self.connected,
            applied_excitation=d.ctrl[self.act_ids].tolist(), activation=d.act[self.act_adr].tolist(),
            minimum_excitation=m.actuator_ctrlrange[self.act_ids, 0].tolist(),
            muscle_force=force.tolist(), passive_muscle_force=passive.tolist(),
            active_muscle_force=(force-passive).tolist(), muscle_length_mm=d.actuator_length[self.act_ids].tolist(),
            muscle_velocity_mm_s=d.actuator_velocity[self.act_ids].tolist(),
            moment_arm_mm=moment.tolist(), muscle_torque=(moment*force).tolist(),
            actuator_torque=float(d.qfrc_actuator[0]), passive_joint_torque=float(d.qfrc_passive[0]),
            constraint_torque=float(d.qfrc_constraint[0]), external_torque=float(d.qfrc_applied[0]),
            bias_torque=float(d.qfrc_bias[0]), effective_inertia=float(inertia[0, 0]),
            torque_projection_error=float(abs((moment*force).sum()-d.qfrc_actuator[0])),
            dynamics_balance_error=float(np.max(np.abs(inertia@d.qacc-rhs))), contacts=int(d.ncon),
            unit_contract=self.metadata['unit_contract'], biological_validation=False)

    def snapshot(self):
        mask = self.mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(self.mj.mj_stateSize(self.model, mask))
        self.mj.mj_getState(self.model, self.data, state, mask)
        return dict(schema='flylab.muscle-state.v1', model_hash=self.identity, tick=self.tick,
                    requested=self.requested.copy(), connected=self.connected, state=state)

    def restore(self, saved):
        if saved.get('schema') != 'flylab.muscle-state.v1' or saved.get('model_hash') != self.identity:
            raise ValueError('Muscle fixture identity mismatch')
        tick = bounded_int(saved.get('tick'), 'muscle tick', 0, 2**31-1)
        connected = boolean(saved.get('connected'), 'motor connection')
        requested = saved.get('requested'); state = saved.get('state')
        mask = self.mj.mjtState.mjSTATE_INTEGRATION
        for a, shape in ((requested, (2,)), (state, (self.mj.mj_stateSize(self.model, mask),))):
            if not isinstance(a, np.ndarray) or a.dtype != np.float64 or a.shape != shape or not np.isfinite(a).all():
                raise ValueError('Invalid muscle state array')
        if np.any((requested < 0) | (requested > 1)): raise ValueError('Invalid muscle excitation')
        candidate = self.mj.MjData(self.model); self.mj.mj_setState(self.model, candidate, state, mask)
        floor, ceiling = self.model.actuator_ctrlrange[self.act_ids].T
        expected = np.clip(requested if connected else np.zeros(2), floor, ceiling)
        if (abs(candidate.time-tick*self.model.opt.timestep) > 1e-8 or
            np.any((candidate.act < 0) | (candidate.act > 1)) or
            not np.array_equal(candidate.ctrl[self.act_ids], expected) or
            np.any(np.abs(candidate.qfrc_applied) > 1.) or
            np.max(np.abs(candidate.qpos)) > 1000. or np.max(np.abs(candidate.qvel)) > 1e6):
            raise ValueError('Invalid muscle state range or clock')
        self.mj.mj_forward(self.model, candidate)
        if not np.isfinite(candidate.qacc).all(): raise ValueError('Invalid restored muscle dynamics')
        self.data = candidate; self.tick = tick; self.requested = requested.copy(); self.connected = connected
