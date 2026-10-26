"""Actual full-BANC/body test with externally taught core and unforced motor cells."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control
from tools.analyze_c_banc_biological_gait import analyze, load_model, sha256, write_json
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.banc_m3_synaptic_efficacy import EfficacyTrial
from tools.probe_c_banc_m3_core_teacher import CoreWaveTeacher
from tools.verify_c_banc_observation import exact


def run(source, calibration, teacher_source, graph_path, out, *, experiment='M3-E44-recruited-full-body'):
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': experiment,
        'question': 'Does the frozen source-constrained recruitment calibration produce healthy coordinated forward movement when all391motor cells and physical senses are unforced?',
        'intervention': 'Frozen source-efficacy/adaptation/anatomical decoder calibration identified by its filehash, retaining all source anatomical edges and signs. Same actual M2 neutral body, source158706neuron topology, servos/contact, real sensory feedback and original navigation-cut forward input. The only external waveform forces24core cells using exact E37native table fromzero. No motor clamp, teachercommand or rootcorrection.',
        'comparison': 'Compare measured body to same-period E42 reference and preserve E37 original coreteacher failure. Multiple model/interface changes are bundled; do not attribute causality to a single change.',
        'checkpoints': 'Before teaching, save new candidate schema; compare one5ms future to an independently disk-restored clone. Preserve exact source body and initial full neural rates/adaptation. Teaching checkpoint is separate training-only schema.',
        'screen': 'Healthy2.2s completion, positive signedforward, all3physicalAP LR pair R>=.5 and circular distance of mean from.5<=.15, and fouroriginalF5knee amplitude thresholds. This shortscreen is diagnostic, not finalF; preserve all raw traces even on early health stop. No surrogateRMS gate.',
        'decision': 'If this downstream screen passes, learn endogenous sourcecore/bridge recurrence to remove the teacher. Otherwise localize full-neural transfer vs physical-interface differences before any further tuning. No bound/waveform grid.',
        'limits': 'OneGPU,2.2main+.01clone/primary future neural/physicalseconds,<=180wallseconds. Stopfault/nonfoot5ms/upright<.9. Nativeallsubstepcontact andF1-F8remainpending; F6unexposed/M2UIpreserved.',
        'tool_sha256': sha256(__file__), 'interface_sha256': sha256(Path('tools/banc_m3_rhythmic_recruitment.py')),
        'calibration_sha256': sha256(calibration / 'calibration.npz'),
        'teacher_sha256': sha256(teacher_source / 'teaching-table.npz')}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    (out / 'interface.py').write_bytes(Path('tools/banc_m3_rhythmic_recruitment.py').read_bytes())
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    initial = StateStore.load(source / 'start')
    calibrated = dict(np.load(calibration / 'calibration.npz'))
    groups = json.loads((calibration / 'muscle-provenance.json').read_text())['muscles']
    base = EfficacyTrial(graph, initial, calibrated['edge_ids'], calibrated['source_values'],
        calibrated['values'], zero_adaptation_ids=calibrated['zero_adaptation_ids'])
    s = RecruitmentTrial(base, calibrated, groups)
    rows, signals = [], []
    model_s = 0
    reason = 'duration'
    try:
        neutral = s.snapshot()
        exact(initial['body'], neutral['body'], 'Original neutral physical state')
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            exact(initial['network'][key], neutral['network'][key], 'Original initial neural ' + key)
        StateStore.save(out / 'candidate-neutral', neutral)
        clone = RecruitmentTrial.from_checkpoint(graph, StateStore.load(out / 'candidate-neutral'))
        try:
            clone.advance(1); model_s += .005; future = clone.snapshot()
        finally:
            clone.close()
        s.advance(1); model_s += .005
        exact(future, s.snapshot(), 'Recruitment disk future')
        s.close()
        s = RecruitmentTrial.from_checkpoint(graph, StateStore.load(out / 'candidate-neutral'))
        n = s.network
        source_identity = n.identity
        n.identity = hashlib.sha256((n.identity + spec['experiment'] + 'external24coreteacher').encode()).hexdigest()
        wave = dict(np.load(teacher_source / 'teaching-table.npz'))
        teacher = CoreWaveTeacher(n, wave['ids'], wave['rate'])
        teacher.capture()
        rows.append(capture_control(s))
        extra_observed = calibrated.get('validation_observed_ids', np.array([], dtype=np.int64))
        observed = np.r_[calibrated['input_ids'], calibrated['motor_ids'], extra_observed]
        if len(np.unique(observed)) != len(observed):
            raise ValueError('Distinct source, motor and extra validation observations required')
        with (out / 'trace.jsonl').open('w', encoding='utf8') as stream:
            stream.write(json.dumps(rows[0]) + '\n')
            for tick in range(1, 441):
                if time.perf_counter() - began > 175:
                    reason = 'wall_limit'; break
                s.advance(1); model_s += .005
                rows.append(capture_control(s))
                stream.write(json.dumps(rows[-1]) + '\n')
                np.testing.assert_array_equal(n.readout(wave['ids']), wave['rate'][tick * 40])
                if n.tick != tick * 20 or int(teacher.cursor) != tick * 40:
                    raise ValueError('Native neural/body/teaching clocks differ')
                signals.append({'rate': n.readout(observed),
                    'motor_current': n._current(n.rate)[s.adapter.motor_ids].cpu().numpy().copy(),
                    'muscle_activation': s.adapter.activation.copy()})
                if s.fault or s.body.fault or s.body.nonfoot_contact() or rows[-1]['upright'] < .9:
                    reason = 'health_guard'; break
        StateStore.save(out / 'final-teaching-state', {'schema': 'flylab.training.recruited-body.v1',
            'autonomous_controller': False, 'whole_neural': n.snapshot(), 'body': s.body.snapshot(),
            'adapter': s.adapter.snapshot(), 'unforced_identity': source_identity,
            'teacher_half_step': int(teacher.cursor)})
        physical = {'controls': len(rows) - 1, 'physical_s': .005 * (len(rows) - 1),
            'signed_forward_mm': s.signed_forward_mm, 'minimum_upright': min(r['upright'] for r in rows),
            'fault': s.fault, 'body_fault': s.body.fault, 'stop_reason': reason,
            'disk_future_exact': True, 'total_neural_physical_s': model_s,
            'wall_s': time.perf_counter() - began}
        write_json(out / 'physical-result.json', physical)
    finally:
        if rows:
            write_json(out / 'trace.json', {'schema': 'flylab.training.recruited-body-trace.v1',
                'controller': 'External24core teacher, unforced391motor cells and actualbodyfeedback',
                'graph_hash': graph.hash, 'model_hash': s.network.identity, 'body_hash': s.body.model_hash,
                'adapter_hash': s.adapter.identity, 'parameters': {**initial['parameters'], 'diagnostic': spec},
                'samples': rows})
        if signals:
            np.savez_compressed(out / 'signals.npz', observed_ids=observed,
                time_s=np.arange(1, len(signals) + 1) * .005,
                **{key: np.array([r[key] for r in signals]) for key in signals[0]})
        s.close()
    quality, widths, coordination = None, {}, {}
    if len(rows) > 102:
        quality, _, _ = analyze(load_model(out / 'trace.json'))
        widths = {leg: row['p95'] - row['p05'] for leg, row in quality['knee_interior_deg'].items()}
        for key in ('lf-rf', 'lm-rm', 'lh-rh'):
            pair = quality['pair_phase'][key]
            mean, resultant = pair['mean_cycles'], pair['resultant_length']
            coordination[key] = mean is not None and resultant is not None and resultant >= .5 and abs((mean - .5 + .5) % 1 - .5) <= .15
    passed = (reason == 'duration' and len(rows) == 441 and physical['signed_forward_mm'] > 0
        and all(coordination.values()) and len(coordination) == 3
        and all(widths.get(leg, 0) >= value for leg, value in {'lf': 25, 'rf': 30, 'lh': 40, 'rh': 35}.items()))
    result = {'experiment': spec['experiment'], 'validity': 'valid_forced_core_actual_body_trial',
        'adoption': 'not_applicable', 'milestone_outcome': 'pending', 'downstream_screen': passed,
        'physical': physical, 'four_knee_width_deg': widths, 'coordination_screen': coordination,
        'quality': quality}
    write_json(out / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'quality'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'calibration', 'teacher-source', 'graph', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--experiment', default='M3-E44-recruited-full-body')
    args = p.parse_args()
    try:
        run(args.source, args.calibration, args.teacher_source, args.graph, args.out, experiment=args.experiment)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error), 'hypothesis_verdict': 'not_tested'})
        raise
