"""One conditional later-cycle phase fit for six existing BANC tonic taus."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.banc_m3_synaptic_efficacy import EfficacyTrial
from tools.probe_c_banc_m3_tonic_onset import (
    phase_trace,
    quadratic_peaks,
    reduced_rk4,
)
from tools.verify_c_banc_observation import exact

PAIRS = (('LF-RF', 2, 1), ('LM-RM', 5, 3), ('LH-RH', 4, 0))
WINDOWS = ((.5, 2.5), (2.5, 5.), (5., 9.))


def project_tonic_tau(values, upper, *, native_initial=False):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Tonic tau must be finite')
    corrected = values.copy()
    below = corrected < .02
    if native_initial:
        # Only the known native float32 encoding of .02 may be normalized.
        native_lower = float(np.float32(.02))
        if np.any(corrected[below] != native_lower):
            raise ValueError('Initial tonic tau is not native float32 .02')
        corrected[below] = .02
    elif np.any(below):
        raise ValueError('Optimized tonic tau below original closed bound')
    if np.any(corrected > upper):
        raise ValueError('Tonic tau above original closed bound')
    return corrected


def phase_objective(times, rates, e1_columns, period, queries):
    """Continuous antiphase residuals from bracketed measured E1 peaks."""
    peaks = [quadratic_peaks(times, rates[:, column]) for column in e1_columns]
    phases = np.array([phase_trace(queries, points) for points in peaks])
    if not np.isfinite(phases).all():
        raise ValueError('Every fixed phase query must be bracketed by actual E1 peaks')
    residual = []
    for _, left, right in PAIRS:
        difference = phases[left] - phases[right]
        for start, stop in WINDOWS:
            mask = (queries >= start) & (queries < stop)
            if not mask.any():
                raise ValueError('Nonempty fixed phase query window required')
            normalization = math.sqrt(int(mask.sum()))
            angle = 2 * np.pi * difference[mask]
            residual.extend(((np.cos(angle) + 1) / normalization).tolist())
            residual.extend((np.sin(angle) / normalization).tolist())
    periods = []
    for points in peaks:
        within = points[(points >= .5) & (points < 10)]
        if len(within) < 3:
            raise ValueError('At least two complete observed E1 periods required')
        periods.append(float(np.diff(within).mean()))
    residual.extend(((np.asarray(periods) / period - 1) / (.1 * math.sqrt(6))).tolist())
    return np.asarray(residual), peaks, phases, periods


def screens(times, rates, e1_columns, period):
    peaks = [quadratic_peaks(times, rates[:, column]) for column in e1_columns]
    selected = (times >= .5) & (times < 10)
    rhythms = []
    for column, points in zip(e1_columns, peaks):
        within = points[(points >= .5) & (points < 10)]
        intervals = np.diff(within)
        width = float(np.percentile(rates[selected, column], 95)
            - np.percentile(rates[selected, column], 5))
        mean = float(intervals.mean()) if len(intervals) else None
        cv = float(intervals.std() / intervals.mean()) if len(intervals) else None
        rhythms.append({'E1_column': int(column), 'complete_cycles': len(intervals),
            'width': width, 'mean_period_s': mean, 'CV': cv,
            'screen': bool(width >= 20 and len(intervals) >= 8 and cv is not None
                and cv <= .3 and mean is not None and abs(mean / period - 1) <= .1)})
    if any(len(points) < 2 for points in peaks):
        raise ValueError('At least two actual E1 peaks per leg for final LR screen')
    last_bracket = min(points[-1] for points in peaks)
    queries = np.arange(.5, min(10., last_bracket), .025)
    phases = np.array([phase_trace(queries, points) for points in peaks])
    if not np.isfinite(phases).all():
        raise ValueError('Final E1 phase queries must all be bracketed')
    pairs = {}
    for label, left, right in PAIRS:
        difference = phases[left] - phases[right]
        for start, stop in ((.5, 2.5), (2.5, 5.), (5., 10.)):
            mask = (queries >= start) & (queries < stop)
            if not mask.any():
                raise ValueError('Each final LR window needs bracketed samples')
            z = np.mean(np.exp(2j * np.pi * difference[mask]))
            mean = float((np.angle(z) / (2 * np.pi)) % 1)
            distance = abs((mean - .5 + .5) % 1 - .5)
            pairs[f'{label}:{start}-{stop}'] = {'mean_cycles': mean,
                'R': float(abs(z)), 'antiphase_distance': distance,
                'samples': int(mask.sum()),
                'screen': bool(distance <= .15 and abs(z) >= .8)}
    return rhythms, pairs


def run(root, graph_path, out, *, prior_execution_wall_s=0.):
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    e91 = root / 'e91-native-tonic-onset'
    e89 = root / 'e89-own-leg-native-recruitment'
    e90_metadata = root / 'e90-own-leg-native-recruitment-body/recruitment-metadata.npz'
    reference = root / 'e85-space-compatible-cadence-reference/commands.npz'
    period = float(np.load(reference)['period_s'])
    if not np.isclose(period, .4196813764904431, rtol=0, atol=1e-12):
        raise ValueError('Frozen E85 reference period required')
    spec = {'experiment': 'M3-E92-native-tonic-phase',
        'hypothesis': 'Changing only all six existing INXXX464 tonic taus can align later-cycle LR E1 phase under E90 conditional background, even though E91 first-peak delay failed.',
        'intervention': 'Use E91 exact36 local signed W/background/zero state and other30taus. Six tonic logtaus [.02,E85period], one initial point from E91 solver-call-05, one bounded least-squares max120 actual10s ODE calls. No external phase/rate in RHS or other network/body change.',
        'objective': 'Fixed queries .5..9s every25ms; each pair in three windows receives cos(2pi left-minus-right phase)+1 and sin term, each normalized by sqrt(window samples). Six mean-period residuals use (period/E85-1)/.1/sqrt6. Peak phases require actual bracketing.',
        'sampling': 'Native RK4 internal250us, recorded E1 and full36 rates only at5ms control boundaries; quadratic E1 peaks interpolate those5ms endpoints.',
        'limits': '<=120 actual ODE calls+one baseline,<=1210 reduced neural seconds,<=180total/150fit wallseconds. Full neural/physics0. Candidate only if all source rhythm and three-window LR screens pass; no F1-F8 claim.',
        'prior_preflight_execution_wall_s':prior_execution_wall_s,
        'preflight_correction':'Only E91 call05 entries exactly equal to native float32 .02 expanded to float64 are normalized to .02. Optimizer out-of-bound taus are rejected; original30 non-tonic taus remain exact.',
        'e91_inputs_sha256': sha256(e91 / 'reduced-inputs.npz'),
        'e91_baseline_sha256': sha256(e91 / 'baseline-reduced.npz'),
        'e91_initial_call_sha256': sha256(e91 / 'solver-call-05.npz'),
        'e90_native_metadata_sha256': sha256(e90_metadata),
        'e91_helper_sha256': sha256(Path(__file__).with_name('probe_c_banc_m3_tonic_onset.py')),
        'e89_calibration_sha256': sha256(e89 / 'calibration.npz'),
        'e89_manifest_sha256': sha256(e89 / 'candidate-neutral/manifest.json'),
        'e89_state_sha256': sha256(e89 / 'candidate-neutral/state.json'),
        'E85_commands_sha256': sha256(reference), 'E85_period_s': period,
        'tool_sha256': sha256(__file__)}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    (out / 'reduced-helper.py').write_bytes(Path(__file__).with_name(
        'probe_c_banc_m3_tonic_onset.py').read_bytes())
    prior = np.load(e91 / 'reduced-inputs.npz')
    prior_baseline = np.load(e91 / 'baseline-reduced.npz')
    source_call = np.load(e91 / 'solver-call-05.npz')
    weights = prior['local_weights']
    background = prior['background']
    old_tau = prior['old_tau']
    local_ids = prior['local_ids']
    times = prior['time_s']
    native_metadata = np.load(e90_metadata)
    native_tau = native_metadata['tau']
    native_ids = native_metadata['observed_ids']
    cal = {key: value.copy() for key, value in np.load(e89 / 'calibration.npz').items()}
    if (weights.shape != (36, 36) or background.shape != (2001, 36)
            or len(old_tau) != 36 or len(np.unique(local_ids)) != 36
            or not np.array_equal(local_ids[-6:], cal['intrinsic_tonic_ids'])
            or not np.allclose(old_tau[-6:], .02)):
        raise ValueError('Exact E91 36-cell source and tonic physiology required')
    initial_tau = source_call['tau'].astype(float)
    initial_raw_tau=initial_tau.copy()
    if not np.array_equal(initial_tau[:30], old_tau[:30]):
        raise ValueError('One exact admissible E91 call05 starting state required')
    native_lower = float(np.float32(.02))
    if not np.all(old_tau[-6:] == native_lower):
        raise ValueError('E91 tonic tau must retain native float32 .02 representation')
    tonic_native_columns = [int(np.flatnonzero(native_ids == cell)[0])
        for cell in local_ids[-6:]]
    if (native_tau.dtype != np.dtype('float32')
            or not np.array_equal(native_tau[tonic_native_columns],
                np.full(6, np.float32(.02)))
            or not np.array_equal(old_tau[-6:],
                native_tau[tonic_native_columns].astype(float))):
        raise ValueError('E91 tonic tau must match actual E90 native float32 state')
    initial_tau[-6:]=project_tonic_tau(initial_tau[-6:],period,
        native_initial=True)
    baseline = reduced_rk4(weights, background, old_tau, 2000)
    np.testing.assert_allclose(baseline, prior_baseline['rate'], rtol=0, atol=1e-12)
    np.savez_compressed(out / 'baseline-verified.npz', rate=baseline,
        old_tau=old_tau, local_ids=local_ids)
    reduced_s = 10.
    e1 = np.array([int(cal['core_ids'][(cal['core_groups'] == group)
        & (cal['core_cell_types'] == 0)][0]) for group in range(6)])
    e1_columns = np.array([np.flatnonzero(local_ids == value)[0] for value in e1])
    queries = np.arange(.5, 9.0001, .025)
    np.savez_compressed(out / 'fixed-queries.npz', time_s=queries,
        E1_ids=e1, E1_columns=e1_columns, initial_tau=initial_tau,
        initial_raw_tau=initial_raw_tau,
        initial_projection_max_abs=np.max(abs(initial_tau-initial_raw_tau)),
        native_tau_dtype=np.asarray(str(native_tau.dtype)),
        native_tonic_tau_bits=native_tau[tonic_native_columns].view(np.uint32),
        initial_raw_tau_float32_bits=initial_raw_tau[-6:].astype(np.float32).view(np.uint32))
    calls = 0
    history = []
    best = None
    fit_began = time.perf_counter()

    def objective(log_values):
        nonlocal calls, reduced_s, best
        if (calls >= 120 or reduced_s + 10 > 1210
                or time.perf_counter() - fit_began > 150
                or time.perf_counter() - began > 150-prior_execution_wall_s):
            raise TimeoutError('E92 bounded actual ODE call or wall allowance exhausted')
        tau = old_tau.copy()
        tau[-6:] = np.exp(log_values)
        tau[-6:] = project_tonic_tau(tau[-6:],period)
        rate = reduced_rk4(weights, background, tau, 2000)
        calls += 1
        reduced_s += 10
        record = {'call': calls, 'tau_s': tau[-6:].tolist(),
            'wall_s': time.perf_counter() - began}
        np.savez_compressed(out / f'call-{calls:03d}.npz',
            tau=tau, time_s=times, rate=rate,
            E1_rate=rate[:, e1_columns])
        try:
            residual, peaks, _, periods = phase_objective(
                times, rate, e1_columns, period, queries)
        except ValueError as error:
            record.update(valid=False, error=str(error))
            history.append(record)
            write_json(out / 'history.json', history)
            raise
        score = float(np.dot(residual, residual))
        record.update(valid=True, loss=score, periods_s=periods,
            peak_counts=[len(points) for points in peaks])
        history.append(record)
        write_json(out / 'history.json', history)
        np.savez_compressed(out / f'call-{calls:03d}-objective.npz',
            residual=residual, periods=periods,
            **{f'peaks_{i}': points for i, points in enumerate(peaks)})
        if best is None or score < best[0]:
            best = (score, calls, tau.copy(), rate.copy())
            np.savez_compressed(out / 'best-observed.npz', score=score,
                call=calls, tau=tau, rate=rate, residual=residual)
        return residual

    try:
        solution = least_squares(objective, np.log(initial_tau[-6:]),
            bounds=(np.log(np.full(6, .02)), np.log(np.full(6, period))),
            diff_step=1e-3, max_nfev=120,
            xtol=1e-8, ftol=1e-8, gtol=1e-8)
        status = {'success': bool(solution.success), 'message': solution.message,
            'nfev': int(solution.nfev), 'cost': float(solution.cost)}
    except (ValueError, TimeoutError) as error:
        status = {'success': False, 'message': str(error),
            'error': type(error).__name__}
    write_json(out / 'solver-status.json', status)
    if best is None:
        result = {'experiment': spec['experiment'], 'validity': 'invalid_candidate_phase',
            'hypothesis_verdict': 'not_tested', 'adoption': 'not_applicable',
            'calls': calls, 'reduced_neural_s': reduced_s,
            'actual_full_neural_s': 0, 'actual_physical_s': 0,
            'wall_s': time.perf_counter() - began, 'solver': status}
        write_json(out / 'result.json', result)
        return
    score, best_call, tau, rate = best
    rhythms, pairs = screens(times, rate, e1_columns, period)
    passed = all(item['screen'] for item in rhythms) and all(
        item['screen'] for item in pairs.values())
    result = {'experiment': spec['experiment'],
        'validity': 'valid_conditional_diagnostic',
        'hypothesis_verdict': 'conditional_later_phase_supported' if passed
            else 'not_supported', 'adoption': 'not_applicable',
        'E91_first_peak_failure_preserved': True,
        'screen': passed, 'rhythms': rhythms, 'LR_by_window': pairs,
        'best_call': best_call, 'best_loss': score,
        'best_tonic_tau_s': tau[-6:].tolist(),
        'solver': status, 'calls': calls,
        'reduced_neural_s': reduced_s,
        'actual_full_neural_s': 0, 'actual_physical_s': 0,
        'fit_wall_s': time.perf_counter() - fit_began,
        'wall_s': time.perf_counter() - began,
        'prior_execution_wall_s':prior_execution_wall_s,
        'initial_tau_projection_max_abs':float(np.max(abs(initial_tau-initial_raw_tau))),
        'native_integration_dt_s': .00025,
        'stored_rate_dt_s': .005,
        'body_validated': False}
    write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)
    if not passed:
        return
    if time.perf_counter() - began > 170:
        raise TimeoutError('Insufficient wall allowance for E92 neutral checkpoint')
    for cell, value in zip(local_ids[-6:], tau[-6:]):
        if cell in cal['tau_ids']:
            raise ValueError('Selected tonic cell already has a tau patch')
        cal['tau_ids'] = np.r_[cal['tau_ids'], cell]
        cal['tau_values'] = np.r_[cal['tau_values'], value]
    if len(cal['tau_ids']) != 38:
        raise ValueError('Original32 tau entries plus six tonic patches required')
    np.savez_compressed(out / 'calibration.npz', **cal)
    anatomy = json.loads((e89 / 'muscle-provenance.json').read_text())
    anatomy['tonic_later_phase_hypothesis'] = {'cell_ids': local_ids[-6:].tolist(),
        'tau_s': tau[-6:].tolist(), 'conditional_only': True,
        'first_peak_E91_failure_preserved': True}
    write_json(out / 'muscle-provenance.json', anatomy)
    graph = GraphStore.load(graph_path)
    original = StateStore.load(root / 'e32-core-adaptation/candidate/start')
    base = EfficacyTrial(graph, original, cal['edge_ids'], cal['source_values'],
        cal['values'], zero_adaptation_ids=cal['zero_adaptation_ids'])
    session = IntrinsicRecruitmentTrial(RecruitmentTrial(base, cal, anatomy['muscles']),
        cal['tau_ids'], cal['tau_values'])
    try:
        neutral = session.snapshot()
        exact(original['body'], neutral['body'], 'E92 original neutral body')
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            exact(original['network'][key], neutral['network'][key],
                'E92 original neutral neural ' + key)
        StateStore.save(out / 'candidate-neutral', neutral)
        clone = IntrinsicRecruitmentTrial.from_checkpoint(graph,
            StateStore.load(out / 'candidate-neutral'))
        try:
            exact(neutral, clone.snapshot(), 'E92 neutral disk exact')
        finally:
            clone.close()
    finally:
        session.close()
    result['neutral_disk_exact'] = True
    result['wall_s'] = time.perf_counter() - began
    write_json(out / 'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'graph', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--prior-execution-wall-s',type=float,default=0.)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve existing experiment output: ' + str(args.out))
    start = time.perf_counter()
    try:
        run(args.root, args.graph, args.out,
            prior_execution_wall_s=args.prior_execution_wall_s)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {
                'validity': 'incomplete', 'hypothesis_verdict': 'not_tested',
                'error': type(error).__name__, 'message': str(error),
                'execution_wall_s': time.perf_counter() - start,
                'actual_full_neural_s': 0, 'actual_physical_s': 0})
        raise
