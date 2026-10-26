"""Test one measured posterior posture correction in exact command-driven bodies."""

import argparse
import copy
import json
import sys
import time
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.body import LEGS
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control, contact_diagnostics
from tools.analyze_c_banc_biological_gait import analyze, load_model, sha256, write_json
from tools.analyze_c_banc_m3_propulsion import trajectory_summary
from tools.analyze_c_banc_observation import new_body
from tools.fit_c_banc_m3_rhythmic_recruitment import phase_average


def measured_rest_correction(cal, signals, rows, reference):
    times = signals['time_s']
    command = np.asarray([r['joint_targets_rad'] for r in rows[1:]])
    np.testing.assert_array_equal(times, [r['time_s'] for r in rows[1:]])
    target_mean = phase_average(reference['time_s'], reference['command']).mean(axis=0)
    delta = np.zeros(42)
    diagnostics = []
    for group, indexes in ((4, [15, 16]), (0, [36, 37])):
        e1, = cal['core_ids'][(cal['core_groups'] == group) & (cal['core_cell_types'] == 0)]
        column, = np.flatnonzero(signals['observed_ids'] == e1)
        peaks, _ = find_peaks(signals['rate'][:, column], prominence=20, distance=20)
        peaks = peaks[(times[peaks] >= 3.5) & (times[peaks] < 9.5)]
        if len(peaks) < 20:
            raise ValueError('At least20 already exposed complete stable source cycles required')
        means = []
        for start, end in pairwise(peaks):
            means.append(np.trapezoid(command[start:end+1], times[start:end+1], axis=0)
                / (times[end] - times[start]))
        mean = np.mean(means, axis=0)
        for row in indexes:
            delta[row] = target_mean[row] - mean[row]
            diagnostics.append({'row': row, 'source_E1_graph_id': int(e1),
                'complete_neural_cycles': len(peaks)-1, 'observed_mean_command_rad': float(mean[row]),
                'target_phase_mean_command_rad': float(target_mean[row]), 'rest_delta_rad': float(delta[row])})
    if np.max(np.abs(delta)) > .1 or not np.isfinite(delta).all():
        raise ValueError('Measured correction exceeds fixed .1rad local posture bound')
    return delta, diagnostics


def body_arm(saved, payload, delta, out, *, reference, controls=440, command_shifts=None, capture_loads=False, wall_limit_s=100):
    out.mkdir()
    body = new_body(saved)
    oldstep = body._step_physics
    shadow = copy.copy(body)
    shadow.d = body.mj.MjData(body.m)
    typ = body.mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(body.mj.mj_stateSize(body.m, typ))
    native_health = []
    native_loads, native_times = [], []
    command_shifts = {} if command_shifts is None else dict(command_shifts)
    if (any(type(k) is not int or not 0 <= k < 6 or type(v) is not int or abs(v) > 20
            for k, v in command_shifts.items())
            or controls + max([0, *command_shifts.values()]) >= len(payload['samples'])):
        body.close()
        raise ValueError('Existing leg/control shifts within recorded future required')
    def step():
        oldstep()
        body.mj.mj_getState(body.m, body.d, state, typ)
        body.mj.mj_setState(body.m, shadow.d, state, typ)
        body.mj.mj_forward(body.m, shadow.d)
        _, rotation, _ = shadow.pose()
        native_health.append((bool(shadow.nonfoot_contact()), float(rotation[2, 2])))
        if capture_loads:
            native_loads.append(contact_diagnostics(shadow)['normal_load_bw'])
            native_times.append(float(body.physics_time()))
    body._step_physics = step
    observer = SimpleNamespace(body=body, control_tick=0, fault=None,
        rate=np.zeros(391), tendon_inputs=dict(payload['samples'][0]['tendon_inputs']))
    rows, reason = [capture_control(observer)], 'duration'
    began = time.perf_counter()
    try:
        for index, recorded in enumerate(payload['samples'][1:controls+1], 1):
            request = recorded['neural_request']
            if request:
                push = request['external_force_at_start']
                body.push_left, body.push_force = push['push_left_s'], np.asarray(push['push_force'])
            old_count = len(native_health)
            targets = np.asarray(recorded['joint_targets_rad']) + delta
            tendons = dict(recorded['tendon_inputs'])
            for leg, shift in command_shifts.items():
                shifted = payload['samples'][max(0, index+shift)]
                targets[7*leg:7*leg+7] = np.asarray(shifted['joint_targets_rad'])[7*leg:7*leg+7] + delta[7*leg:7*leg+7]
                tendon = LEGS[leg] + '_tarsus'
                tendons[tendon] = shifted['tendon_inputs'][tendon]
            body.step_joint_targets(targets, np.asarray(recorded['adhesion'], bool),
                tendon_inputs=tendons, interpolate=not (request and request['cuts']['motor']))
            observer.control_tick = index
            observer.tendon_inputs = tendons
            rows.append(capture_control(observer))
            if reference:
                for key in ('qpos', 'qvel', 'ctrl'):
                    np.testing.assert_array_equal(getattr(body.d, key), np.asarray(recorded[key]),
                        err_msg=f'Unchanged full reference body control{index} {key}')
            if len(native_health)-old_count != 50:
                raise ValueError('All native physical substeps required')
            if body.fault or any(c or u < .9 for c, u in native_health[old_count:]):
                reason = 'health_guard'; break
            if time.perf_counter() - began > wall_limit_s:
                reason = 'wall_limit'; break
        result = {'scope': 'Body-only command intervention; no neural simulation or final gait claim',
            'validity': 'valid_exact_baseline_body' if reference else 'valid_changed_command_body',
            'physical_s': .005*(len(rows)-1), 'neural_s': 0, 'wall_s': time.perf_counter()-began,
            'stop_reason': reason, 'minimum_upright': min(x[1] for x in native_health),
            'native_nonfoot_samples': sum(x[0] for x in native_health), 'native_samples': len(native_health),
            'fault': body.fault, 'original_qpos_qvel_ctrl_every_control_exact': True if reference else None}
        write_json(out / 'trace.json', {**{k: v for k, v in payload.items() if k != 'samples'},
            'schema': 'flylab.m3-body-only-command-rest-diagnostic.v1',
            'controller': 'Already recorded E73 commands with explicit diagnostic intervention; neural activity is not simulated',
            'diagnostic_rest_delta_rad': delta.tolist(), 'diagnostic_leg_control_shifts': command_shifts,
            'samples': rows})
        if capture_loads:
            np.savez_compressed(out / 'native-load.npz', time_s=native_times, normal_load_bw=native_loads)
        if len(rows) == controls+1:
            result['trajectory'] = trajectory_summary(rows)
            quality, _, _ = analyze(load_model(out / 'trace.json'))
            write_json(out / 'quality.json', quality)
            result['knees'] = {k: v['p95']-v['p05'] for k, v in quality['knee_interior_deg'].items()}
        write_json(out / 'result.json', result)
        return result
    finally:
        body._step_physics = oldstep
        body.close()


def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    source = root / 'e73-front-knee-delivery-body'
    calibration = root / 'e72-front-knee-delivery'
    write_json(out / 'spec.json', {'experiment': 'M3-E74b-posterior-mean-posture',
        'execution_revision': 'E74preserved: preparation complete but first body step was not executed because optional neural_request was None. Handle absent request exactly as the already validated E71replayer; same data-derived fourdelta/candidate/criteria, no hypothesis verdict from that execution error.',
        'question': 'Does posterior mean posture account for the yaw bias after E73frontamplitudes/10scoordination pass but direction fails?',
        'basis': 'LHcoxa roll varies with an active adductor whereas RHtwo anatomicalroll muscles are silent and its command stays-.069222rad. Both posterior coxa pitch commands stay .13401rad, above E42phase mean near.09. This is a specific posterior posture mismatch; front knee balancing inE73 increased earlyyaw, so it is not sufficient.',
        'intervention': 'One data-derived restdelta on four posterior coxa rows15/16/36/37, no gain/phase/weight/foot/root edit. E42phase-average mean commands are the targets; subtract E73cycle-averaged actual commands over complete sourceE1 cycles in exposed[3.5,9.5). Require eachdelta<=.1rad. No grid/retargeting or loop.',
        'assay': 'Restore original E73neutral body. First unchanged recorded commands[0,2.2] with every qpos/qvel/ctrl bit exact. Then separate samebody with samecommands plus fixed fourrest deltas. This is an exogenous recorded-command design intervention, not autonomous BANC. Native health guard remains; no F6exposure.',
        'decision': 'Require full2.2s healthy candidate, original fourknee minimums, and >=30% reduction in matched[.5,2.2] absolute yaw drift fromE73 .916880rad. If supported, prepare the static rest-only original neural decoder and test once with full158706BANC/body under original10s/F1-F8. If failed, preserve diagnosis without fit/threshold changes.',
        'limits': 'Sequential CPUbody only2x2.2s=4.4physicalseconds,0neural/GPUadvance,<=100wallseconds each. NoM2/UI/F6changes.',
        'source_sha256': sha256(source / 'trace.json'),
        'calibration_sha256': sha256(calibration / 'calibration.npz'), 'tool_sha256': sha256(__file__)})
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    cal = dict(np.load(calibration / 'calibration.npz'))
    payload = json.loads((source / 'trace.json').read_text())
    delta, diagnostic = measured_rest_correction(cal, np.load(source / 'signals.npz'), payload['samples'],
        np.load(root / 'e42-source-aligned-reference/commands.npz'))
    write_json(out / 'rest-hypothesis.json', {'diagnostic': diagnostic, 'delta_rad': delta.tolist()})
    cal['rest'] = cal['rest'] + delta
    np.savez_compressed(out / 'calibration.npz', **cal)
    saved = StateStore.load(source / 'start')
    baseline = body_arm(saved, payload, np.zeros(42), out / 'baseline', reference=True)
    print(json.dumps({'baseline': baseline}), flush=True)
    if baseline['stop_reason'] != 'duration':
        raise ValueError('Same original2.2s reference must complete before intervention')
    candidate = body_arm(saved, payload, delta, out / 'candidate', reference=False)
    ratio = (abs(candidate['trajectory']['UI_yaw_change_rad']) / abs(baseline['trajectory']['UI_yaw_change_rad'])) if 'trajectory' in candidate else None
    supported = (candidate['stop_reason'] == 'duration' and ratio <= .7 and
        all(candidate['knees'][leg] >= minimum for leg, minimum in {'lf':25, 'rf':30, 'lh':40, 'rh':35}.items()))
    result = {'experiment': 'M3-E74b-posterior-mean-posture', 'validity': 'valid_body_only_causal_intervention',
        'hypothesis_verdict': 'supported_for_recorded_command_body' if supported else 'not_supported',
        'adoption': 'not_applicable', 'milestone_outcome': 'pending', 'body_mechanism_screen': supported,
        'yaw_ratio': ratio, 'baseline': baseline, 'candidate': candidate}
    write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)


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
                'hypothesis_verdict': 'not_tested', 'error': type(error).__name__, 'message': str(error)})
        raise
