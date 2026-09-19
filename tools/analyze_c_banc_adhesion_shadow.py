"""Same-state feasible-target adhesion audit; recorded physical commands only."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.body import LEGS
from flylab.c.graph import GraphStore
from flylab.c.integrity import file_hash, write_json
from flylab.c.rate_body import RateBodyAdapter
from flylab.c.storage import StateStore
from tools.analyze_c_banc_observation import new_body, validate_recording


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'spec.json', {
        'experiment': 'M2-E1', 'source': str(args.source),
        'evaluator_hash': file_hash(Path(__file__)),
        'hypothesis': 'Unreachable joint requests change the adhesion lift veto',
        'comparison': 'Same physical state and raw neural motor command; only shadow lift uses effective actuator target',
        'validity': 'Exact raw decoder commands and all replayed integration states',
        'wall_limit_s': 180,
    })
    began = time.perf_counter()
    payload = json.loads((args.source / 'trace.json').read_text(encoding='utf-8'))
    saved = StateStore.load(args.source / 'start')
    validate_recording(payload, saved)
    signals = np.load(args.source / 'recorded-signals.npz')
    graph = GraphStore.load(args.graph)
    body = new_body(saved)
    p = saved['parameters']
    adapter = RateBodyAdapter(graph, body, **{k: p[k] for k in (
        'sensory_gain', 'motor_gain', 'pooling', 'coxa_geometry', 'ltm_tendons', 'coxa_endpoint') if k in p})
    adapter.restore(saved['adapter'])
    limited = body.m.actuator_ctrllimited[body.act_ids].astype(bool)
    ranges = body.m.actuator_ctrlrange[body.act_ids]
    typ = body.mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(body.mj.mj_stateSize(body.m, typ))
    audit = []
    try:
        for i, row in enumerate(payload['samples'][1:]):
            targets, original = adapter.decode(signals['motor'][i])
            if targets.tobytes() != np.asarray(row['joint_targets_rad']).tobytes() or not np.array_equal(original, row['adhesion']):
                raise AssertionError(f'Raw decoder mismatch {i}')
            feasible = np.where(limited, np.clip(targets, ranges[:, 0], ranges[:, 1]), targets)
            raw_lift = body.foot_kinematics(targets)['target_lift_mm']
            effective_lift = body.foot_kinematics(feasible)['target_lift_mm']
            vote = np.array([
                r['muscle_rate_units'].get('tarsus_depressor_muscle', 0) + r['muscle_rate_units'].get('long_tendon_muscle', 0)
                > r['muscle_rate_units'].get('tarsus_levator_muscle', 0) + 1.0
                for r in adapter.last_motor])
            proposed = vote & (effective_lift <= .02)
            if not np.array_equal(original, vote & (raw_lift <= .02)):
                raise AssertionError('Adhesion reconstruction differs')
            audit.append({'time_s': row['time_s'], 'raw_lift_mm': raw_lift.tolist(),
                          'effective_lift_mm': effective_lift.tolist(), 'neural_vote': vote.tolist(),
                          'original_adhesion': original.tolist(), 'proposed_adhesion': proposed.tolist(),
                          'moving': not row['neural_request']['navigation_at_start']['brake_request']})
            external = row['neural_request']['external_force_at_start']
            body.push_left = external['push_left_s']
            body.push_force = np.asarray(external['push_force'])
            body.step_joint_targets(np.asarray(row['joint_targets_rad']), np.asarray(row['adhesion'], bool), tendon_inputs=row['tendon_inputs'])
            body.mj.mj_getState(body.m, body.d, state, typ)
            if state.tobytes() != signals['physical'][i].tobytes():
                raise AssertionError(f'Physical mismatch {i}')
            if time.perf_counter() - began > 180:
                raise TimeoutError('180s audit limit')
        moving = [r for r in audit if r['moving']]
        raw = np.asarray([r['original_adhesion'] for r in moving])
        proposed = np.asarray([r['proposed_adhesion'] for r in moving])
        result = {'validity': 'valid', 'run_status': 'completed', 'hypothesis_verdict': 'supported' if np.any(raw != proposed) else 'refuted',
                  'adoption': 'defer', 'milestone_outcome': 'pending', 'physical_and_raw_decoder_exact': True,
                  'samples': len(audit), 'moving_samples': len(moving),
                  'legs': {leg: {'changed': int(np.count_nonzero(raw[:, k] != proposed[:, k])),
                                  'release_to_attach': int(np.count_nonzero(~raw[:, k] & proposed[:, k])),
                                  'attach_to_release': int(np.count_nonzero(raw[:, k] & ~proposed[:, k]))}
                           for k, leg in enumerate(LEGS)},
                  'wall_seconds': time.perf_counter() - began}
        write_json(args.out / 'audit.json', audit)
        write_json(args.out / 'result.json', result)
        print(json.dumps(result), flush=True)
    finally:
        body.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--graph', type=Path, default=Path('data/acquisitions/banc888-windows-20260910/bundle'))
    run(p.parse_args())
