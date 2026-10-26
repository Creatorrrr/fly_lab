"""Actual full BANC feedback trial of the frozen four-knee delivery candidate."""

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.banc_walking import BancWalkingSession
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control
from tools.analyze_c_banc_biological_gait import (
    LEGS,
    analyze,
    load_model,
    sha256,
    write_json,
)
from tools.probe_c_banc_m3_knee_delivery import SELECTED, THRESHOLDS, candidate_decode
from tools.verify_c_banc_observation import exact


class KneeAdapter:
    schema = 'flylab.m3-four-knee-adapter.v1'

    def __init__(self, original, gains):
        self.original = original; self.gains = np.asarray(gains, dtype=float).copy()
        if self.gains.shape != (4,) or not np.isfinite(self.gains).all() or np.any((self.gains < 1) | (self.gains > 4)):
            raise ValueError('Four fixed knee gains required')
        self.identity = hashlib.sha256(json.dumps({'original': original.identity,
            'schema': self.schema, 'gains': self.gains.tolist(), 'tau_s': .03,
            'legs': [LEGS[i] for i in SELECTED]}, sort_keys=True).encode()).hexdigest()

    def __getattr__(self, key):
        return getattr(self.original, key)

    def decode(self, rates, *, motor_cut=False, dt=.005):
        return candidate_decode(self.original, rates, self.gains, motor_cut=motor_cut, dt=dt)

    def snapshot(self):
        return {'schema': self.schema, 'hash': self.identity, 'gains': self.gains.copy(),
            'original': self.original.snapshot()}

    def restore(self, saved):
        if saved['schema'] != self.schema or saved['hash'] != self.identity:
            raise ValueError('Knee adapter identity differs')
        exact(saved['gains'], self.gains, 'Knee adapter fixed gains')
        self.original.restore(saved['original'])


class KneeTrial(BancWalkingSession):
    schema = 'flylab.m3-four-knee-walking.v1'

    def __init__(self, graph, original_source, gains):
        base = BancWalkingSession.from_checkpoint(graph, original_source)
        self.__dict__.update(base.__dict__)
        self.adapter = KneeAdapter(self.adapter, gains)

    @classmethod
    def from_checkpoint(cls, graph, saved):
        if saved.get('schema') != cls.schema:
            raise ValueError('Explicit knee candidate checkpoint required')
        original = copy.deepcopy(saved)
        original['schema'] = BancWalkingSession.schema
        original['adapter'] = original['adapter']['original']
        session = cls(graph, original, saved['adapter']['gains'])
        try:
            session.adapter.restore(saved['adapter'])
            exact(saved, session.snapshot(), 'Knee trial complete checkpoint')
        except Exception:
            session.close(); raise
        return session


def run(source, identification, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    candidate = json.loads((identification / 'frozen-candidate.json').read_text())
    if candidate['selected_legs'] != [LEGS[i] for i in SELECTED] or candidate['tau_s'] != .03:
        raise ValueError('Frozen knee identity changed')
    if candidate['original_source_sha256'] != sha256(source / 'trace.json'):
        raise ValueError('Frozen knee source trace differs')
    if not json.loads((identification / 'result.json').read_text())['screen_passed']:
        raise ValueError('Conditional body feasibility required')
    spec = {'experiment': 'M3-E31-full-feedback-four-knee',
        'hypothesis': 'E30c four-knee delivery improves actual autonomous full-BANC geometric excursions without losing sustained forward motion',
        'intervention': 'Same original M2 full158706/11584852 graph, weights, cell dynamics, encoder, tonic navigation-cut forward input and physical body. Apply ONLY frozen E30c four-knee gains/30ms. Actual body sensory input each control; no recorded motor, teacher, phase, time input or root correction.',
        'checkpoints': 'Distinct candidate adapter/session schema and full hash. Original parameter/model physiology remains M2. Save at1s then compare complete .1s future to a disk-restored clone. Cloning runs only while main trial is paused and stays within shared neural model budget.',
        'measurement': 'Actual 200Hz body positions/feet/motor rates, original geometric F evaluator. Report candidate F3-F5. Native all-substep contact/slip, F6 and navigation F7 are still required separately. Control-boundary no-contact alone cannot pass final F3.',
        'limits': 'OneGPU,10main+.1clone modelsec,<=285wallsec. Stop at fault/nonfoot boundary contact or upright<.9; preserve complete raw trace before assertions. No adoption or live UI changes.',
        'candidate_sha256': sha256(identification / 'frozen-candidate.json'),
        'source_trace_sha256': sha256(source / 'trace.json'), 'tool_sha256': sha256(__file__),
        'evaluator_sha256': sha256(Path('tools/analyze_c_banc_biological_gait.py')),
        'decoder_tool_sha256': sha256(Path('tools/probe_c_banc_m3_knee_delivery.py')),
        'frozen_candidate': candidate}
    write_json(out / 'spec.json', spec); (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    (out / 'knee-delivery.py').write_bytes(Path('tools/probe_c_banc_m3_knee_delivery.py').read_bytes())
    began = time.perf_counter(); graph = GraphStore.load(graph_path); initial = StateStore.load(source / 'start')
    s = KneeTrial(graph, initial, candidate['gain']); clone_future = None; future_pass = False; reason = 'duration'
    raw = out / 'trace.jsonl'; rows = [capture_control(s)]; model_s = 0.
    try:
        StateStore.save(out / 'start', s.snapshot())
        exact(initial['body'], s.body.snapshot(), 'Same M2 initial body')
        with raw.open('w', encoding='utf8') as stream:
            stream.write(json.dumps(rows[0]) + '\n')
            for tick in range(1, 2001):
                if time.perf_counter() - began > 280:
                    reason = 'wall_limit'; break
                s.advance(1); model_s += .005
                rows.append(capture_control(s)); stream.write(json.dumps(rows[-1]) + '\n')
                if s.fault or s.body.fault or s.body.nonfoot_contact() or rows[-1]['upright'] < .9:
                    reason = 'health_guard'; break
                if tick == 200:
                    StateStore.save(out / 'checkpoint-1s', s.snapshot())
                    clone = KneeTrial.from_checkpoint(graph, StateStore.load(out / 'checkpoint-1s'))
                    try:
                        clone.advance(20); model_s += .1; clone_future = clone.snapshot()
                        StateStore.save(out / 'clone-future', clone_future)
                    finally:
                        clone.close()
                if tick == 220:
                    exact(clone_future, s.snapshot(), 'Disk-restored candidate future')
                    future_pass = True
                if tick % 400 == 0:
                    print(json.dumps({'model_s': tick * .005, 'signed_forward_mm': s.signed_forward_mm,
                        'upright': rows[-1]['upright'], 'wall_s': time.perf_counter() - began}), flush=True)
        if not s.fault and not s.body.fault:
            StateStore.save(out / 'final', s.snapshot())
        frame = s.frame()
    finally:
        write_json(out / 'trace.json', {'schema': 'flylab.m3-four-knee-trace.v1',
            'controller': 'Actual M2 full BANC dynamics with four-knee delivery correction',
            'graph_hash': graph.hash, 'model_hash': s.network.identity,
            'body_hash': s.body.model_hash, 'adapter_hash': s.adapter.identity,
            'parameters': {**initial['parameters'], 'four_knee_delivery': candidate}, 'samples': rows})
        s.close()
    if len(rows) > 102:
        measured, _, _ = analyze(load_model(out / 'trace.json'))
        widths = {leg: values['p95'] - values['p05'] for leg, values in measured['knee_interior_deg'].items()}
        amplitude_pass = all(widths[LEGS[leg]] >= threshold for leg, threshold in zip(SELECTED, THRESHOLDS))
    else:
        measured = None; widths = {}; amplitude_pass = False
    result = {'experiment': spec['experiment'], 'run_status': 'completed' if len(rows) == 2001 else 'stopped',
        'validity': 'valid_full_neural_physical_trial', 'stop_reason': reason,
        'adoption': 'defer_remaining_F' if amplitude_pass and reason == 'duration' else 'reject',
        'milestone_outcome': 'pending', 'frame': frame, 'quality': measured, 'knee_width_deg': widths,
        'F5_amplitude_only': amplitude_pass, 'checkpoint_future_exact': future_pass,
        'neural_physics_model_s': model_s, 'wall_s': time.perf_counter() - began}
    write_json(out / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('quality', 'frame')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'identification', 'graph', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output directory must be new')
    try:
        run(args.source, args.identification, args.graph, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'error': type(error).__name__, 'message': str(error),
                'validity': 'incomplete', 'hypothesis_verdict': 'not_tested', 'adoption': 'not_applicable',
                'milestone_outcome': 'pending'})
        raise
