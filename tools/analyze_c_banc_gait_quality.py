"""M2's fixed moving-window quality calculation on existing BANC recordings."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.body import LEGS
from flylab.c.integrity import file_hash, write_json


def quality(payload):
    rows = payload['samples']
    time = np.asarray([r['time_s'] for r in rows])
    if len(time) < 2 or not np.allclose(np.diff(time), .005, atol=1e-8, rtol=0):
        raise ValueError('Continuous 5ms samples required')
    position = np.asarray([r['body']['position'] for r in rows])[:, [0, 2]]
    distance = np.asarray([r['navigation']['distance_mm'] for r in rows])
    if not np.isfinite(position).all() or not np.isfinite(distance).all():
        raise ValueError('Finite physical observations required')
    brake_start = 10.0
    for row in rows[1:]:
        request = row['neural_request']['navigation_at_start']
        if request['brake_request']:
            brake_start = min(10.0, row['time_s'] - .005)
            break
    selection = (time > .5) & (time <= brake_start + 1e-9)
    selected = [r for r, keep in zip(rows, selection) if keep]
    load = np.asarray([r['feet']['normal_load_bw'] for r in selected]).reshape(-1, 6)
    slip = np.asarray([[np.nan if s is None else s for s in r['feet']['slip_mm_s']] for r in selected]).reshape(-1, 6)
    support = load >= .02
    if np.any(support & ~np.isfinite(slip)):
        raise ValueError('Missing supported contact speed')
    legs = []
    for i, leg in enumerate(LEGS):
        v = slip[support[:, i], i]
        legs.append({'leg': leg, 'supported_samples': len(v),
                     'support_fraction': float(support[:, i].mean()) if len(selected) else 0,
                     'mean_slip_mm_s': float(v.mean()) if len(v) else None,
                     'p95_slip_mm_s': float(np.percentile(v, 95)) if len(v) else None})
    first = np.flatnonzero(distance <= 2.0)
    arrival = int(first[0]) if len(first) else None
    duration = len(selected) * .005
    speeds = np.r_[0., np.linalg.norm(np.diff(position, axis=0), axis=1) / .005]
    observed = duration >= 2.0 - 1e-9 and all(x['supported_samples'] >= 100 for x in legs)
    return {
        'schema': 'flylab.banc-gait-quality.v1', 'window_start_exclusive_s': .5,
        'window_end_s': brake_start, 'moving_observed_s': duration,
        'support_threshold_bw': .02, 'legs': legs,
        'balanced_mean_slip_mm_s': float(np.mean([x['mean_slip_mm_s'] for x in legs])) if all(x['mean_slip_mm_s'] is not None for x in legs) else None,
        'mean_support_legs': float(support.sum(axis=1).mean()) if len(selected) else 0,
        'moving_root_mean_speed_mm_s': float(speeds[selection].mean()) if len(selected) else 0,
        'first_arrival_s': float(time[arrival]) if arrival is not None else None,
        'approach_speed_mm_s': float((distance[0] - distance[arrival]) / time[arrival]) if arrival is not None and time[arrival] > 0 else None,
        'initial_distance_mm': float(distance[0]), 'final_distance_mm': float(distance[-1]),
        'minimum_upright': min(r['upright'] for r in rows),
        'faults': [r['fault'] for r in rows if r['fault']],
        'observation_valid': bool(observed),
        'limitations': '5ms physical contact samples; normal force includes adhesion reaction; no biological gait validation',
    }


def compare(base, candidate):
    observed = base['observation_valid'] and candidate['observation_valid']
    def ratio(key):
        a, b = base[key], candidate[key]
        return b / a if a is not None and b is not None and a > 0 else None
    ratios = {key: ratio(key) for key in ('balanced_mean_slip_mm_s', 'mean_support_legs', 'moving_root_mean_speed_mm_s', 'approach_speed_mm_s')}
    arrival_ok = base['first_arrival_s'] is not None and candidate['first_arrival_s'] is not None
    support_ok = ratios['mean_support_legs'] is not None and ratios['mean_support_legs'] >= .8
    approach_ok = ratios['approach_speed_mm_s'] is not None and ratios['approach_speed_mm_s'] >= .8
    speed_ok = (ratios['moving_root_mean_speed_mm_s'] is not None and ratios['moving_root_mean_speed_mm_s'] >= .8) or (arrival_ok and candidate['first_arrival_s'] <= base['first_arrival_s'])
    q2 = bool(observed and arrival_ok and support_ok and approach_ok and speed_ok)
    return {'ratios_candidate_over_baseline': ratios, 'Q1': bool(ratios['balanced_mean_slip_mm_s'] is not None and ratios['balanced_mean_slip_mm_s'] <= .7),
            'Q2': q2, 'observation_valid': bool(observed), 'arrival': bool(arrival_ok),
            'support': bool(support_ok), 'approach_speed': bool(approach_ok), 'moving_speed_or_earlier_arrival': bool(speed_ok)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads((args.source / 'trace.json').read_text(encoding='utf-8'))
    result = quality(payload)
    result['source'] = str(args.source)
    result['evaluator_hash'] = file_hash(Path(__file__))
    if args.baseline:
        baseline = quality(json.loads((args.baseline / 'trace.json').read_text(encoding='utf-8')))
        result['comparison'] = compare(baseline, result)
    if args.out.exists():
        raise FileExistsError(args.out)
    write_json(args.out, result)
    print(json.dumps(result), flush=True)
