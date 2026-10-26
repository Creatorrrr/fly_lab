"""Exact recorded-command body replay to diagnose foot propulsion asymmetry."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.body import LEGS
from flylab.c.storage import StateStore
from flylab.c.walking_trace import contact_diagnostics
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.analyze_c_banc_observation import new_body


def foot_wrenches(body):
    """Force on the fly, with moment about its instantaneous subtree COM.

    MuJoCo's contact-frame force acts on geom2. Frame rows are world axes;
    multiplying its transpose maps the wrench to world coordinates. The force
    changes sign when the foot is geom1. See the official contact convention:
    https://mujoco.readthedocs.io/en/stable/computation/#contact
    """
    mapping, _ = body._force_maps['feet']
    _, rotation, _ = body.pose()
    com = body.d.subtree_com[body.thorax]
    forces, torques = np.zeros((6, 3)), np.zeros((6, 3))
    wrench = np.empty(6)
    for index, contact in enumerate(body.d.contact[:body.d.ncon]):
        if contact.exclude:
            continue
        a, b = int(contact.geom1), int(contact.geom2)
        if a in body.support_geom_ids and mapping[b] >= 0:
            segment, sign = int(mapping[b]), 1.
        elif b in body.support_geom_ids and mapping[a] >= 0:
            segment, sign = int(mapping[a]), -1.
        else:
            continue
        body.mj.mj_contactForce(body.m, body.d, index, wrench)
        frame = np.asarray(contact.frame).reshape(3, 3)
        force = sign * (frame.T @ wrench[:3])
        torque = sign * (frame.T @ wrench[3:]) + np.cross(contact.pos - com, force)
        forces[segment // 5] += force
        torques[segment // 5] += torque
    return forces @ rotation / body.weight0, torques / body.weight0


def trajectory_summary(rows):
    selected = [r for r in rows if .5 <= r['time_s'] <= 2.2]
    position = np.asarray([r['body']['position'] for r in selected])[:, [0, 2]]
    yaw = np.unwrap([r['body']['yaw'] for r in selected])
    net = float(np.linalg.norm(position[-1] - position[0]))
    path = float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum())
    return {'window_s': [.5, 2.2], 'net_mm': net, 'path_mm': path,
        'net_path_ratio': net / path, 'UI_yaw_change_rad': float(yaw[-1] - yaw[0]),
        'UI_delta_xz_mm': (position[-1] - position[0]).tolist()}


def replay(source, saved_path, out):
    began = time.perf_counter()
    payload = json.loads((source / 'trace.json').read_text())
    saved = StateStore.load(saved_path)
    if saved['body']['modelHash'] != payload['body_hash']:
        raise ValueError('Original body identity required')
    rows = [r for r in payload['samples'] if r['time_s'] <= 2.2]
    if len(rows) != 441:
        raise ValueError('The same full original [0,2.2] recording is required')
    body = new_body(saved)
    shadow = copy.copy(body)
    shadow.d = body.mj.MjData(body.m)
    typ = body.mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(body.mj.mj_stateSize(body.m, typ))
    original_step = body._step_physics
    values = []
    def observed_step():
        original_step()
        body.mj.mj_getState(body.m, body.d, state, typ)
        body.mj.mj_setState(body.m, shadow.d, state, typ)
        body.mj.mj_forward(body.m, shadow.d)
        contact = contact_diagnostics(shadow)
        force, torque = foot_wrenches(shadow)
        values.append((float(body.physics_time()), force, torque,
            contact['normal_load_bw'],
            [np.nan if v is None else v for v in contact['slip_mm_s']]))
    body._step_physics = observed_step
    verified = 0
    try:
        for index, row in enumerate(rows):
            if index:
                request = row.get('neural_request')
                if request:
                    push = request['external_force_at_start']
                    body.push_left, body.push_force = push['push_left_s'], np.asarray(push['push_force'])
                body.step_joint_targets(np.asarray(row['joint_targets_rad']),
                    np.asarray(row['adhesion'], bool), tendon_inputs=row['tendon_inputs'],
                    interpolate=not (request and request['cuts']['motor']))
            for key in ('qpos', 'qvel', 'ctrl'):
                np.testing.assert_array_equal(getattr(body.d, key), np.asarray(row[key]),
                    err_msg=f'{source.name} control{index} {key}')
            if body.fault:
                raise ValueError(body.fault)
            verified = index
            if time.perf_counter() - began > 240:
                raise TimeoutError('Diagnostic replay wall limit')
        t = np.asarray([v[0] for v in values])
        force, torque = np.asarray([v[1] for v in values]), np.asarray([v[2] for v in values])
        load, slip = np.asarray([v[3] for v in values]), np.asarray([v[4] for v in values])
        selected = (t >= .5) & (t < 2.2)
        native_exact = None
        if (source / 'native-contact.npz').exists():
            original = np.load(source / 'native-contact.npz')
            np.testing.assert_array_equal(t, original['time_s'][:len(t)])
            np.testing.assert_array_equal(load, original['normal_load_bw'][:len(t)])
            np.testing.assert_array_equal(slip, original['slip_mm_s'][:len(t)])
            native_exact = True
        summary = {'source': str(source), 'trace_sha256': sha256(source / 'trace.json'),
            'validity': 'valid_exact_recorded_command_replay', 'verified_controls': verified,
            'qpos_qvel_ctrl_every_control_exact': True, 'recorded_native_contact_exact': native_exact,
            'native_sample_count': len(t), 'physical_replay_s': 2.2, 'neural_s': 0,
            'wall_s': time.perf_counter() - began, 'trajectory': trajectory_summary(rows),
            'legs': {}, 'claim': 'Descriptive force delivery, not causal isolation or final gait validation'}
        for i, leg in enumerate(LEGS):
            support = selected & (load[:, i] >= .02)
            summary['legs'][leg] = {
                'mean_force_body_xyz_bw': force[selected, i].mean(axis=0).tolist(),
                'mean_world_torque_xyz_bw_mm': torque[selected, i].mean(axis=0).tolist(),
                'support_fraction': float((load[selected, i] >= .02).mean()),
                'mean_supported_slip_mm_s': float(slip[support, i].mean()),
                'mean_load_bw': float(load[selected, i].mean())}
        np.savez_compressed(out / (source.name + '-wrenches.npz'), time_s=t,
            force_body_xyz_bw=force, torque_world_xyz_bw_mm=torque,
            normal_load_bw=load, slip_mm_s=slip)
        write_json(out / (source.name + '-result.json'), summary)
        return summary
    finally:
        body._step_physics = original_step
        body.close()


def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'spec.json', {'experiment': 'M3-E71-recorded-foot-propulsion',
        'question': 'Which source-to-body delivery asymmetries accompany E50 and E70 yaw drift, given the straight E42 reference?',
        'intervention': 'No change to controller or original traces. Restore each exact original neutral body and replay the already recorded commands once through2.2s. E42 is a teacher reference, E50 has external neural teacher, E70 is original intrinsic BANC recording.',
        'measurement': 'Require qpos/qvel/ctrl bit equality at every200Hzcontrol; E70 every100us native load/slip must also match. Observe separate forwarded shadow, never overwrite actor. At every100us sum supported foot force in bodyaxes and worldtorque about subtree COM. Compare[.5,2.2).',
        'hypothesis': 'A consistent ipsilateral delivery difference is a candidate design clue only; signed contact torque does not itself prove which muscle causes turning. No parameter fit or finalFpass.',
        'limits': 'Three sequential body-only2.2s replays=6.6physicalseconds, zero neural/GPUmodelseconds, <=240wallseconds each. Original M2/UI/F6 untouched.',
        'force_convention_source': 'https://mujoco.readthedocs.io/en/stable/computation/#contact',
        'tool_sha256': sha256(__file__)})
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    arms = [('e42-source-aligned-reference', root / 'e32-core-adaptation/candidate/start'),
        ('e50-tonic-full-body', root / 'e50-tonic-full-body/candidate-neutral'),
        ('e70-two-DN-adaptation-body', root / 'e70-two-DN-adaptation-body/start')]
    results = []
    for name, start in arms:
        result = replay(root / name, start, out)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'legs'}), flush=True)
    write_json(out / 'result.json', {'experiment': 'M3-E71-recorded-foot-propulsion',
        'validity': 'valid_exact_body_only_replays', 'hypothesis_verdict': 'descriptive_diagnosis',
        'adoption': 'not_applicable', 'milestone_outcome': 'pending', 'arms': results})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error), 'hypothesis_verdict': 'not_tested'})
        raise
