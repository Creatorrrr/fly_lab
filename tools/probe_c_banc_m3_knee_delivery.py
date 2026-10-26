"""F5-directed conditional body replay of the unchanged M2 motor activity."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import shgo
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control
from tools.analyze_c_banc_biological_gait import (
    LEGS,
    analyze,
    geometric_knee,
    load_model,
    sha256,
    write_json,
)
from tools.analyze_c_banc_m3_motor_transfer import adapter_for, pool, raw_request
from tools.analyze_c_banc_observation import new_body
from tools.verify_c_banc_observation import exact

SELECTED = np.array([0, 2, 3, 5])
THRESHOLDS = np.array([25., 40., 30., 35.])


def knee_curve(body, joints):
    """Exact rigid hinge angle curve, checked using separate MuJoCo data."""
    ids = np.array([[int(body.body_ids[body.body_indices[f'{leg}_{part}']])
        for part in ('trochanterfemur', 'tibia', 'tarsus1')] for leg in LEGS])
    q0 = body.d.qpos[body.qpos_ids[joints]].copy()
    points = body.d.xpos[ids]
    u = points[:, 0] - points[:, 1]; v = points[:, 2] - points[:, 1]
    u /= np.linalg.norm(u, axis=1)[:, None]; v /= np.linalg.norm(v, axis=1)[:, None]
    axis = body.d.xaxis[body.joint_ids[joints]]
    c = np.sum(u * axis, axis=1) * np.sum(v * axis, axis=1)
    a = np.sum(u * v, axis=1) - c
    b = np.sum(u * np.cross(axis, v), axis=1)
    def evaluate(q):
        delta = np.asarray(q) - q0
        return np.degrees(np.arccos(np.clip(a * np.cos(delta) + b * np.sin(delta) + c, -1, 1)))
    # Validation points cover every possible +/- .5rad decoder offset.
    shadow = body.mj.MjData(body.m); worst = 0.
    for delta in np.linspace(-.5, .5, 41):
        shadow.qpos[:] = body.d.qpos
        shadow.qpos[body.qpos_ids[joints]] = q0 + delta
        body.mj.mj_forward(body.m, shadow)
        packed = np.zeros((1, 6, 4, 3)); packed[0, :, 1:] = shadow.xpos[ids]
        actual = geometric_knee(packed)[0]
        worst = max(worst, float(np.max(abs(actual - evaluate(q0 + delta)))))
    if worst > 1e-8:
        raise ValueError('Rigid hinge formula differs from actual geometric knee')
    return evaluate, {'q0': q0.tolist(), 'a': a.tolist(), 'b': b.tolist(), 'c': c.tolist(),
        'maximum_fk_error_deg': worst}


def candidate_decode(adapter, rates, gains, *, motor_cut=False, dt=.005):
    """Keep source decoder state/rules, replacing only four knee requests/LPs."""
    if adapter.adhesion_model != 'legacy' or adapter.active_length or adapter.muscle_activation_first:
        raise ValueError('Knee trial requires the frozen M2 mechanical contract')
    selected = adapter.joints[SELECTED, 5]
    previous = adapter.offset[selected].copy()
    raw = raw_request(adapter, pool(adapter, rates))
    targets, adhesion = adapter.decode(rates, motor_cut=motor_cut, dt=dt)
    if not motor_cut:
        desired = .5 * np.tanh(gains * raw[selected] / .5)
        adapter.offset[selected] = previous + (1 - math.exp(-dt / .03)) * (desired - previous)
        targets = adapter.body.neutral + adapter.offset
        # Recompute the SAME source adhesion request and SAME physical lift veto
        # using the candidate target, rather than retaining the old knee veto.
        adhesion = np.array([row['muscle_rate_units'].get('tarsus_depressor_muscle', 0)
            + row['muscle_rate_units'].get('long_tendon_muscle', 0)
            > row['muscle_rate_units'].get('tarsus_levator_muscle', 0) + 1.
            for row in adapter.last_motor], dtype=bool)
        lift = np.asarray(adapter.body.foot_kinematics(targets)['target_lift_mm'])
        adhesion[lift > .02] = False
    return targets, adhesion


def run(source, graph_path, out, identified=None):
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': 'M3-E30c-f5-knee-delivery',
        'execution_revision': 'E30 failed numpy-bool JSON export before physics; E30b identified gains then stopped before first body step because adapter returns six leg tendons but the session adds two constant abdomen inputs. Mirror BancWalkingSession full8 initialization/update. Preserve both failed runs; no model/data/screen change. Reuse immutable E30b gains when supplied, with source and feasibility rechecks.',
        'hypothesis': 'The existing M2 motor signal can transmit the F5 front/hind knee excursions with a restricted decoder correction while the source body remains healthy',
        'motivation': 'E02 shows accurate physical target following but attenuation before it; fixed12Hz teacher reconstruction is not a final F criterion. Prior M2 uniform knee2x and broad faster-filter failures remain rejected.',
        'intervention': 'Only LF/LH/RF/RH tibia request gains and post-tanh LP. Fix30ms (existing fast-coxa engineering setting, not measured knee physiology). Identify smallest numerically found gain in[1,4] reaching1.05*F5 width on exposed0.5..2.2s M2 rates. Keep neutral, +/- .5tanh, joint/control limits, all other DOFs, mean unit pooling and live geometric adhesion veto/tendon rules.',
        'identification': 'Saved0..2.2s M2 raw input, saturation THEN30ms LP then control/joint range intersection then exact rigid-hinge CTr-FTi-TiTa angle. SHGO bounded scalar constrained optimization,n64,iters3 perleg; do not assume width monotonicity. A numerical minimum, not a formal global optimality claim. No physical gain/tau grid. If no feasible numerical solution, stop before candidate physics.',
        'comparator': 'Same initial M2 body; original2.2s rate replay must reproduce recorded targets/adhesion/tendons/qpos/qvel/ctrl exactly. Candidate runs once with same recorded motor inputs and recomputed source adhesion veto. Sensory input is not fed back into the recorded neural activity.',
        'screen': 'Four actual geometric95-5 knee widths meet fixedF5, full2.2s/no fault/nonfoot contact/minupright>=.9, positive signed forward. Report net/path, foot AP phase, cycles and tracking separately; F3/F4/F6/F7 remain untested and no gait adoption. A passing body diagnostic proceeds directly to actual full BANC feedback.',
        'limits': 'CPU only,2x2.2modelseconds,<=120wallsec; preserve numerical/physical failures and initial source. No live page changes.',
        'source_sha256': sha256(source / 'trace.json'), 'tool_sha256': sha256(__file__),
        'selected_legs': [LEGS[i] for i in SELECTED], 'f5_width_deg': THRESHOLDS.tolist()}
    prior_fits = None
    if identified:
        previous = json.loads((identified / 'spec.json').read_text())
        if previous['source_sha256'] != spec['source_sha256']:
            raise ValueError('Identification source mismatch')
        prior_fits = json.loads((identified / 'identification.json').read_text())['fits']
        spec['reused_identification_sha256'] = sha256(identified / 'identification.json')
    write_json(out / 'spec.json', spec); (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter(); saved = StateStore.load(source / 'start')
    graph = GraphStore.load(graph_path); rows = json.loads((source / 'trace.json').read_text())['samples'][:441]
    rates = np.array([row['motor_rates'] for row in rows[1:]], dtype=np.float32)
    times = np.array([row['time_s'] for row in rows[1:]])
    mask = (times >= .5 - 1e-9) & (times < 2.2 - 1e-9)
    body = new_body(saved)
    try:
        adapter = adapter_for(graph, body, saved)
        if saved['control_tick'] != 0 or np.any(adapter.offset != 0):
            raise ValueError('This inverse filter requires the frozen neutral zero-offset start')
        joints = adapter.joints[:, 5]; curve, curve_meta = knee_curve(body, joints)
        raw = np.array([raw_request(adapter, pool(adapter, rate))[joints] for rate in rates])
        controls = body.m.actuator_ctrlrange[body.act_ids[joints]]
        joint_ranges = body.m.jnt_range[body.joint_ids[joints]]
        lo = np.where(body.m.actuator_ctrllimited[body.act_ids[joints]], controls[:, 0], -np.inf)
        hi = np.where(body.m.actuator_ctrllimited[body.act_ids[joints]], controls[:, 1], np.inf)
        lo = np.maximum(lo, np.where(body.m.jnt_limited[body.joint_ids[joints]], joint_ranges[:, 0], -np.inf))
        hi = np.minimum(hi, np.where(body.m.jnt_limited[body.joint_ids[joints]], joint_ranges[:, 1], np.inf))
        neutral = body.neutral[joints].copy()
        source_knee = geometric_knee(np.array([[row['body']['legs'][leg] for leg in LEGS] for row in rows[1:]]))
        actual_q = np.array([row['joint_angles_rad'] for row in rows[1:]])[:, joints]
        source_geometry_error = float(np.max(abs(curve(actual_q) - source_knee)))
        if source_geometry_error > 1e-7:
            raise ValueError('Curve does not reproduce source geometric knee across body poses')
        alpha = 1 - np.exp(-.005 / .03)
        def prediction(leg, gain):
            request = .5 * np.tanh(gain * raw[:, leg] / .5)
            offset = lfilter([alpha], [1, alpha - 1], request)
            q = np.tile(neutral, (len(offset), 1))
            q[:, leg] = np.clip(neutral[leg] + offset, lo[leg], hi[leg])
            return curve(q)[:, leg]
        def width(leg, gain):
            q = prediction(leg, gain)[mask]
            return float(np.percentile(q, 95) - np.percentile(q, 5))
        gains = []; fits = []; passed = True
        for leg, target in zip(SELECTED, THRESHOLDS * 1.05):
            evaluations = []
            def constraint(x, leg=leg, target=target, evaluations=evaluations):
                value = width(leg, float(x[0])); evaluations.append((float(x[0]), value))
                return value - target
            if prior_fits is not None:
                entry = next(row for row in prior_fits if row['leg'] == LEGS[leg])
                if entry['target_with_headroom_deg'] != target or not entry['solver_success']:
                    raise ValueError('Identification contract or solver status differs')
                fit = SimpleNamespace(success=True, x=np.array([entry['gain']]),
                    message='Reused unchanged successful E30b solution; feasibility checked again')
            elif constraint([1.]) >= 0:
                fit = SimpleNamespace(success=True, x=np.array([1.]), message='gain1 already sufficient')
            else:
                fit = shgo(lambda x: float(x[0]), [(1., 4.)],
                    constraints={'type': 'ineq', 'fun': constraint}, n=64, iters=3,
                    options={'f_tol': 1e-9, 'minimize_every_iter': True})
            gain = float(fit.x[0]) if getattr(fit, 'x', None) is not None else None
            feasible = bool(gain is not None and 1 <= gain <= 4 and width(leg, gain) >= target - 1e-6)
            fits.append({'leg': LEGS[leg], 'target_with_headroom_deg': float(target),
                'gain1_width_deg': width(leg, 1.), 'solver_success': bool(fit.success),
                'solver_message': str(fit.message), 'numerically_feasible': feasible, 'gain': gain,
                'width_deg': width(leg, gain) if gain is not None else None,
                'evaluations': evaluations})
            gains.append(gain); passed &= bool(fit.success) and feasible
        np.savez_compressed(out / 'raw-input.npz', time_s=times, raw=raw, mask=mask,
            source_motor=rates, joint_indices=joints, control_joint_intersection=np.column_stack((lo, hi)))
        write_json(out / 'identification.json', {'fits': fits, 'curve': curve_meta,
            'source_geometric_max_error_deg': source_geometry_error, 'wall_s': time.perf_counter() - began})
    finally:
        body.close()
    if not passed:
        complete = all(row['solver_success'] for row in fits)
        write_json(out / 'result.json', {'run_status': 'completed_identification' if complete else 'incomplete_identification',
            'validity': 'valid_bounded_identification' if complete else 'incomplete_numerical_identification',
            'hypothesis_verdict': 'no_feasible_numerical_delivery' if complete else 'inconclusive',
            'adoption': 'reject', 'milestone_outcome': 'pending', 'physics_model_s': 0., 'fits': fits})
        print(json.dumps({'screen_passed': False, 'gains': gains, 'reason': 'identification_failed'}), flush=True)
        return
    gains = np.asarray(gains)
    write_json(out / 'frozen-candidate.json', {'selected_legs': [LEGS[i] for i in SELECTED],
        'gain': gains.tolist(), 'tau_s': .03, 'original_source_sha256': spec['source_sha256']})
    results = {}; model_s = 0.
    for arm in ('baseline', 'candidate'):
        destination = out / arm; destination.mkdir()
        body = new_body(saved); started = time.perf_counter()
        try:
            exact(saved['body'], body.snapshot(), 'E30 initial body')
            adapter = adapter_for(graph, body, saved)
            observer = SimpleNamespace(body=body, control_tick=0, rate=np.zeros(391),
                tendon_inputs={name: .2 for name in body.tendon_control.names}, fault=None)
            trace = [capture_control(observer)]; exact_controls = 0; signed_forward = 0.
            for tick, (rate, original) in enumerate(zip(rates, rows[1:]), 1):
                if time.perf_counter() - began > 115:
                    raise TimeoutError('E30 total CPU wall limit')
                target, adhesion = candidate_decode(adapter, rate, gains) if arm == 'candidate' else adapter.decode(rate)
                observer.tendon_inputs.update(adapter.tendon_inputs)
                if arm == 'baseline':
                    exact(target, np.array(original['joint_targets_rad']), 'E30 target')
                    exact(adhesion, np.array(original['adhesion'], bool), 'E30 adhesion')
                    exact(observer.tendon_inputs, original['tendon_inputs'], 'E30 tendons')
                observer.rate = rate.copy()
                prior, rotation, _ = body.pose()
                body.step_joint_targets(target, adhesion, tendon_inputs=observer.tendon_inputs)
                current = body.pose()[0]
                signed_forward += float((current - prior)[:2] @ rotation[:2, 0])
                observer.control_tick = tick; trace.append(capture_control(observer)); model_s += .005
                if arm == 'baseline':
                    for key in ('qpos', 'qvel', 'ctrl'):
                        exact(getattr(body.d, key), np.array(original[key]), 'E30 physical ' + key)
                    exact_controls += 1
                if body.fault or body.nonfoot_contact():
                    break
            StateStore.save(destination / 'final-body', body.snapshot())
            write_json(destination / 'trace.json', {'schema': 'flylab.m3-knee-delivery.v1',
                'controller': 'M2 recorded neural activity; body-only conditional replay',
                'body_hash': body.model_hash, 'model_hash': saved['network']['model_hash'],
                'adapter_hash': sha256(out / 'frozen-candidate.json') if arm == 'candidate' else adapter.identity,
                'graph_hash': graph.hash, 'parameters': spec, 'samples': trace})
            physical = {'control_count': observer.control_tick, 'wall_s': time.perf_counter() - started,
                'fault': body.fault, 'nonfoot_contact': bool(body.nonfoot_contact()),
                'initial_body_exact': True, 'exact_source_controls': exact_controls,
                'signed_forward_mm': signed_forward,
                'minimum_upright_entire_run': min(row['upright'] for row in trace),
                'contact_observation': '5ms boundaries; final F3 native contact verification remains required'}
        finally:
            body.close()
        measured, _, _ = analyze(load_model(destination / 'trace.json'))
        widths = {leg: values['p95'] - values['p05'] for leg, values in measured['knee_interior_deg'].items()}
        screen = physical['control_count'] == 440 and not physical['fault'] and not physical['nonfoot_contact']
        screen &= physical['minimum_upright_entire_run'] >= .9 and physical['signed_forward_mm'] > 0
        screen &= all(widths[LEGS[leg]] >= target for leg, target in zip(SELECTED, THRESHOLDS))
        results[arm] = {'physical': physical, 'quality': measured, 'knee_width_deg': widths, 'screen_passed': bool(screen)}
        write_json(destination / 'result.json', results[arm])
        print(json.dumps({'arm': arm, 'physical': physical, 'knee_width_deg': widths, 'screen_passed': bool(screen)}), flush=True)
        if arm == 'baseline' and exact_controls != 440:
            raise ValueError('Incomplete original body replay: do not start the candidate')
    write_json(out / 'result.json', {'experiment': spec['experiment'], 'run_status': 'completed',
        'validity': 'valid_conditional_body_replay', 'adoption': 'not_applicable', 'milestone_outcome': 'pending',
        'screen_passed': results['candidate']['screen_passed'], 'results': results,
        'physics_model_s': model_s, 'wall_s': time.perf_counter() - began})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'graph', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--identified', type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output directory must be new')
    try:
        run(args.source, args.graph, args.out, args.identified)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'error': type(error).__name__,
                'message': str(error), 'validity': 'incomplete', 'hypothesis_verdict': 'not_tested',
                'adoption': 'not_applicable', 'milestone_outcome': 'pending'})
        raise
