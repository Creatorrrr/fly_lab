"""Body-only reference matched to the exposed neural teaching period."""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control
from tools.analyze_c_banc_biological_gait import analyze, load_model, sha256, write_json
from tools.analyze_c_banc_observation import new_body
from tools.verify_c_banc_observation import exact


def run(source, waveform, out):
    out.mkdir(parents=True, exist_ok=False)
    wave = json.loads((waveform / 'teaching-waveform.json').read_text())
    spec = {'experiment': 'M3-E42-source-aligned-body-reference',
        'question': 'Can a slower moderate-amplitude version of the same measured FlyBody step provide a useful physical reference at the already frozen core teaching period?',
        'reference': 'Same local FlyBodyPreprogrammedSteps anatomical DOF mapping. Period .15s equals E37; amplitude .7 reduces the original large teacher excursions and translation while keeping the hind-knee target near its original .7*67degree span. This is an engineering choice, not measured biological amplitude or a fitted source gait.',
        'coordination': 'Same exposed E37 anatomical tripod groups; phase0 is the measured template PEP/swing onset. Magnitude ramps as .7*(1-exp(-t/.05)) from the source neutral body. Explicit externalphase/template used ONLY for reference/training; never a final neural runtime path.',
        'body': 'Exact original M2 neutral body, same servos/contact options. No adhesion; all existing tendons held .2 as in E01/E11 reference. Motor/neural simulation absent, no rootpose corrections. These teacher tendon conditions require separate neural-interface validation.',
        'decision': 'Healthy2.2s completion, positive signedforward, and actual four-knee F5 amplitude thresholds permit use as training reference. Report actual forward efficiency/APcycle/phase as measurements; do not require the prior arbitrary10mm-in2.2s teacher gate. Final F1-F8 unchanged and still require full neural control. Failure is retained without a cadence/amplitude grid.',
        'limits': 'CPU body only,2.2modelseconds,<=120wallsec,stopfault/nonfoot/upright<.9. SourceF6unexposed/UI/M2preserved.',
        'tool_sha256': sha256(__file__), 'waveform_sha256': sha256(waveform / 'teaching-waveform.json'),
        'step_source_sha256': sha256(Path('.venv/Lib/site-packages/flygym_demo/complex_terrain/preprogrammed.py')),
        'step_asset_sha256': sha256(Path('.venv/Lib/site-packages/flygym_demo/complex_terrain/assets/single_steps_flybody.npz'))}
    if wave['period_s'] != .15:
        raise ValueError('Frozen source period changed')
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    saved = StateStore.load(source / 'start')
    body = new_body(saved)
    began = time.perf_counter()
    rows, commands, phases = [], [], []
    try:
        exact(saved['body'], body.snapshot(), 'Reference exact source body')
        leg_phase = dict(zip(wave['group_leg_by_direct_motor_projection'], wave['group_training_phase_cycles']))
        offsets = np.array([leg_phase[leg] for leg in body.steps.legs])
        tendons = {name: .2 for name in body.tendon_control.names}
        observer = SimpleNamespace(body=body, control_tick=0, rate=np.zeros(391),
            tendon_inputs=tendons, fault=None)
        rows = [capture_control(observer)]
        signed_forward = 0.; reason = 'duration'
        for tick in range(1, 441):
            if time.perf_counter() - began > 115:
                reason = 'wall_limit'; break
            t = tick * .005
            phase = 2 * np.pi * ((t / wave['period_s'] + offsets) % 1)
            amplitude = np.full(6, .7 * (1 - np.exp(-t / .05)))
            q = body.steps.get_joint_angles_by_dof_order(phase, amplitude, body.order)
            body.step_joint_targets(q, np.zeros(6, dtype=bool), tendon_inputs=tendons)
            observer.control_tick = tick
            row = capture_control(observer)
            previous = rows[-1]['body']
            displacement = np.asarray(row['body']['position']) - np.asarray(previous['position'])
            signed_forward += float(displacement[[0, 2]] @ np.asarray(previous['basis'])[[0, 2], 0])
            rows.append(row); commands.append(q.copy()); phases.append(phase.copy())
            if body.fault or body.nonfoot_contact() or row['upright'] < .9:
                reason = 'health_guard'; break
        StateStore.save(out / 'final-body', body.snapshot())
        frame = {'signed_forward_mm': signed_forward, 'minimum_upright': min(r['upright'] for r in rows),
            'fault': body.fault, 'nonfoot_contact': bool(body.nonfoot_contact()), 'reason': reason}
        write_json(out / 'physical-result.json', {'frame': frame, 'controls': len(rows) - 1,
            'physical_model_s': (len(rows) - 1) * .005, 'wall_s': time.perf_counter() - began})
    finally:
        if rows:
            write_json(out / 'trace.json', {'schema': 'flylab.training.source-aligned-reference.v1',
                'controller': 'External source-period measured-step reference; no BANC execution',
                'graph_hash': saved['graph_hash'], 'body_hash': body.model_hash,
                'model_hash': 'no-neural-model', 'adapter_hash': sha256(out / 'spec.json'),
                'parameters': spec, 'samples': rows})
            np.savez_compressed(out / 'commands.npz', command=commands, phase=phases,
                time_s=np.arange(1, len(commands) + 1) * .005, neutral=body.neutral)
        body.close()
    quality = None; widths = {}
    if len(rows) > 102:
        quality, _, _ = analyze(load_model(out / 'trace.json'))
        widths = {leg: v['p95'] - v['p05'] for leg, v in quality['knee_interior_deg'].items()}
    amplitude_ok = all(widths.get(leg, 0) >= limit for leg, limit in [('lf', 25), ('rf', 30), ('lh', 40), ('rh', 35)])
    result = {'experiment': spec['experiment'], 'validity': 'valid_external_reference_body_trial',
        'adoption': 'not_applicable', 'milestone_outcome': 'pending', 'frame': frame,
        'controls': len(rows) - 1, 'knee_width_deg': widths, 'quality': quality,
        'reference_screen_pass': len(rows) == 441 and reason == 'duration' and signed_forward > 0 and amplitude_ok,
        'physical_model_s': (len(rows) - 1) * .005, 'neural_model_s': 0, 'wall_s': time.perf_counter() - began}
    write_json(out / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'quality'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'waveform', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    try:
        run(args.source, args.waveform, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error), 'hypothesis_verdict': 'not_tested'})
        raise
