"""Exogenous leg-time diagnostic: keep every within-leg command vector intact."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.body import LEGS
from flylab.c.storage import StateStore
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.probe_c_banc_m3_posterior_rest import body_arm


def stance_summary(trace, native, lo, hi):
    rows = json.loads(trace.read_text())['samples']
    times = np.array([r['time_s'] for r in rows])
    if times[-1] < hi+.04:
        return {'window_s': [lo, hi], 'validity': 'insufficient_observed_window_and_derivative_padding'}
    origin = np.array([r['body']['position'] for r in rows])
    forward = np.array([r['body']['basis'] for r in rows])[:, :, 0]
    tips = np.array([[r['body']['legs'][leg][-1] for leg in LEGS] for r in rows])
    ap = np.einsum('tlj,tj->tl', tips-origin[:, None, :], forward)
    velocity = np.gradient(gaussian_filter1d(ap, 2, axis=0, mode='nearest'), .005, axis=0)
    data = np.load(native)
    tn, load = data['time_s'], data['normal_load_bw']
    vn = np.column_stack([np.interp(tn, times, velocity[:, j]) for j in range(6)])
    window = (tn >= lo) & (tn < hi)
    w = np.where(window[:, None] & (load >= .02), load, 0)
    if np.any(w.sum(axis=0) == 0):
        raise ValueError('Every leg needs observed supporting load')
    selected = (times >= lo) & (times < hi)
    yaw = np.unwrap([r['body']['yaw'] for r in rows])[selected]
    return {'window_s': [lo, hi], 'validity': 'valid_descriptive_proxy',
        'load_fraction_with_forward_foot_motion': ((w*(vn > 1)).sum(axis=0)/w.sum(axis=0)).tolist(),
        'UI_yaw_change_rad': float(yaw[-1]-yaw[0]),
        'proxy': 'Body-relative tarsus5 AP in mm, Gaussian sigma10ms on200Hzpositions then derivative; interpolate velocity to100us actual all-tarsal load. Threshold>=.02BW and APvelocity>1mm/s. Not world-horizontal ground slip or force propulsion.'}


def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    source = root / 'e73-front-knee-delivery-body'
    ref = root / 'e42-source-aligned-reference'
    write_json(out / 'spec.json', {'experiment': 'M3-E75-leg-time-versus-shape',
        'question': 'Does correcting only LM/RHwhole-leg relative timing reduce early and sustained yaw without changing within-leg kinematic shape?',
        'basis': 'E73all3APphase/fourknees pass but net/path.2082fails. Frontamplitude and posterior mean-rest corrections did not improve yaw. Earlybody-relative forward-foot support is asymmetric LM/RH; lateE73this asymmetry mostly recedes while yaw continues. Thus timing is a competing hypothesis, not an established sole cause.',
        'intervention': 'Use original E73commands without rest/gain/shape edits. Derive LM/RHtime offsets once from original full-window E73minusE42LRmean phase and observed medianperiod. Move each selected leg all7joints and its own tendon together by nearest integer5ms sample, retaining exact recorded vectors. No phase fit to yaw. No adhesion in either source/arm.',
        'sign': 'Physical evaluator is phase(right)-phase(left). Advance LM to decrease middlepair phase; delay RH to decrease hindpair phase. This corrects an opposite preliminary verbal sign recommendation. Analytic sinusoid sign validation required before body run.',
        'boundary': 'Each body runs original neutral through9.6s: all advanced sourcecommands remain within recorded10s; delayed commands beforezero hold original neutral vector. No sourcefuture extrapolation, timeinterpolation, repeatedcycle extension, root edit or world change. Diagnostic9.6s is not final10sF3verification.',
        'measurement': 'First unmodified9.6s body qpos/qvel/ctrl everycontrol exactly original; also native100us load exact. Then separate changed-timing body with sameworld/nativehealthguard. Inspect early[.5,2.2) and late[3.5,9.5) yaw and identical descriptive load/footAP proxy. Preserve all raw observations and healthstops.',
        'decision': 'Whole timing mechanism supported only if complete healthy9.6s, originalfourknees and absolute yaw reduced>=30% in BOTH early andlatewindows. Early-only improvement or earlyguard cannot support full sustained mechanism. Failure narrows toward within-leg kinematics/force direction. Success still requires existing-source static neural implementation and new actualBANC/body F1-F8; recording shifts never enter finalcontroller.',
        'limits': 'Two sequentialCPUbodyreplays9.6seach,19.2physicalseconds/0neural,<=150wallseconds each. NoM2/UI/F6changes.',
        'source_sha256': sha256(source / 'trace.json'), 'reference_sha256': sha256(ref / 'result.json'),
        'tool_sha256': sha256(__file__), 'body_replayer_sha256': sha256(Path('tools/probe_c_banc_m3_posterior_rest.py'))})
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    (out / 'body_replayer.py').write_bytes(Path('tools/probe_c_banc_m3_posterior_rest.py').read_bytes())
    payload = json.loads((source / 'trace.json').read_text())
    quality = json.loads((source / 'result.json').read_text())['quality']
    reference_quality = json.loads((ref / 'result.json').read_text())['quality']
    shifts, proposal = {}, []
    for leg, pair, sign in ((1, 'lm-rm', 1), (5, 'lh-rh', -1)):
        phase = quality['pair_phase'][pair]['mean_cycles']
        reference_phase = reference_quality['pair_phase'][pair]['mean_cycles']
        difference = (phase-reference_phase+.5) % 1-.5
        period = quality['events'][LEGS[leg]]['period_s']['median']
        shift = round(sign*difference*period/.005)
        if not 0 < abs(shift) <= 6:
            raise ValueError('Single measured local timing change required')
        shifts[leg] = shift
        expected_phase = phase-sign*shift*.005/period
        # Two complete analytic cycles check actual time-advance phase sign.
        angle = 2*np.pi*np.arange(256)/128
        left = np.sin(angle + (2*np.pi*shift*.005/period if leg == 1 else 0))
        right = np.sin(angle + 2*np.pi*phase + (2*np.pi*shift*.005/period if leg == 5 else 0))
        measured = (np.angle(np.mean(np.exp(1j*(np.angle(hilbert(right))-np.angle(hilbert(left))))))/(2*np.pi)) % 1
        np.testing.assert_allclose(measured, expected_phase, rtol=0, atol=1e-14)
        proposal.append({'leg': LEGS[leg], 'pair': pair, 'original_mean_phase': phase,
            'reference_mean_phase': reference_phase, 'source_period_s': period,
            'shift_controls': shift, 'positive_shift_means': 'take a later sourcecommand, thus ADVANCE this leg',
            'idealized_new_phase': expected_phase, 'analytic_phase_sign_exact': True})
    if any(any(r['adhesion']) for r in payload['samples']):
        raise ValueError('This exact vector-shift diagnostic requires original noadhesion')
    write_json(out / 'timing-proposal.json', {'legs': proposal})
    saved = StateStore.load(source / 'start')
    baseline = body_arm(saved, payload, np.zeros(42), out / 'baseline', reference=True,
        controls=1920, capture_loads=True, wall_limit_s=150)
    if baseline['stop_reason'] != 'duration':
        raise ValueError('Original unmodified body reference must complete')
    observed = np.load(out / 'baseline/native-load.npz')
    original = np.load(source / 'native-contact.npz')
    for key in ('time_s', 'normal_load_bw'):
        np.testing.assert_array_equal(observed[key], original[key][:len(observed[key])])
    print(json.dumps({'baseline_physical_s': baseline['physical_s'], 'wall_s': baseline['wall_s'],
        'whole_reference_control_and_native_load_exact': True, 'proposal': proposal}), flush=True)
    candidate = body_arm(saved, payload, np.zeros(42), out / 'candidate', reference=False,
        controls=1920, command_shifts=shifts, capture_loads=True, wall_limit_s=150)
    comparisons = {}
    for label, lo, hi in (('early', .5, 2.2), ('late', 3.5, 9.5)):
        a = stance_summary(out / 'baseline/trace.json', out / 'baseline/native-load.npz', lo, hi)
        b = stance_summary(out / 'candidate/trace.json', out / 'candidate/native-load.npz', lo, hi)
        ratio = abs(b['UI_yaw_change_rad'])/abs(a['UI_yaw_change_rad']) if 'UI_yaw_change_rad' in b else None
        comparisons[label] = {'baseline': a, 'candidate': b, 'absolute_yaw_ratio': ratio}
    supported = (candidate['stop_reason'] == 'duration'
        and all(x['absolute_yaw_ratio'] is not None and x['absolute_yaw_ratio'] <= .7 for x in comparisons.values())
        and all(candidate.get('knees', {}).get(leg, 0) >= minimum for leg, minimum in {'lf':25, 'rf':30, 'lh':40, 'rh':35}.items()))
    result = {'experiment': 'M3-E75-leg-time-versus-shape', 'validity': 'valid_body_only_timing_intervention',
        'hypothesis_verdict': 'supported_for_body_diagnostic' if supported else 'not_supported',
        'adoption': 'not_applicable', 'milestone_outcome': 'pending', 'timing_mechanism_screen': supported,
        'baseline': baseline, 'candidate': candidate, 'comparisons': comparisons,
        'actual_neural_s': 0, 'physical_s': baseline['physical_s']+candidate['physical_s']}
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
