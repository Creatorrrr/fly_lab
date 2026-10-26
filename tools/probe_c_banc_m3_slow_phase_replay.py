"""E106: bounded body-only replay of E105 commands at E85 relative phases."""

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flylab.body import LEGS
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control, contact_diagnostics
from tools.analyze_c_banc_biological_gait import analyze, load_model, sha256, write_json
from tools.analyze_c_banc_observation import new_body
from tools.probe_c_banc_m3_tonic_onset import phase_trace, quadratic_peaks

E1_IDS = (53861, 71439, 152280, 92648, 97727, 91167)
OFFSETS = np.array((0., .5, 0., .5, 0., .5))
COUNTERS = np.array((0, 0, 0, 0, 0, -1))
CONTROLS = 1900
SOURCE = ROOT / 'verification/banc-natural-gait-m3-20260919/e105-startup-native-body-diagnostic-v2/body'
REFERENCE = ROOT / 'verification/banc-natural-gait-m3-20260919/e85-space-compatible-cadence-reference/commands.npz'


def require_grid(source_t, query_t):
    source_t = np.asarray(source_t, dtype=np.float64)
    query_t = np.asarray(query_t, dtype=np.float64)
    if (source_t.ndim != 1 or len(source_t) != 2001
            or not np.array_equal(source_t, np.arange(2001) * .005)):
        raise ValueError('Complete exact 2001-point 0..10s source grid required')
    if (query_t.shape != (CONTROLS, 6) or not np.isfinite(query_t).all()
            or not np.all(np.diff(query_t, axis=0) > 0)
            or np.any(query_t < source_t[0]) or np.any(query_t > source_t[-1])):
        raise ValueError('Finite, increasing, bracketed six-leg query grid required')
    times = source_t[1:CONTROLS + 1, None]
    if (np.any(np.abs(query_t - times) > .125 + 1e-12)
            or not np.array_equal(query_t[:500], np.broadcast_to(times[:500], (500, 6)))
            or not np.array_equal(query_t[:, 0], times[:, 0])):
        raise ValueError('Source prefix, LF identity and 125ms bound required')


def require_accepted_review(review):
    if (not isinstance(review, dict)
            or review.get('accepted_as_valid_failed_body_diagnostic') is not True
            or review.get('blocking_defects') != []):
        raise ValueError('Explicit accepted E105 second review with zero blocking defects required')


def require_source_e1_rates(rates):
    rates = np.asarray(rates, dtype=np.float64)
    if rates.shape != (2000, 6) or not np.isfinite(rates).all():
        raise ValueError('Complete finite 2000x6 original E1 endpoint rates required')
    return rates


def require_source_tendons(series):
    expected = {'abdomen_pitch', 'abdomen_yaw', *(leg + '_tarsus' for leg in LEGS)}
    if set(series) != expected:
        raise ValueError('Exactly eight source tendon series required')
    for key, values in series.items():
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (2001,) or not np.isfinite(values).all():
            raise ValueError(f'Complete finite 2001-point source tendon series required: {key}')


def planar_initial_heading_forward(positions, initial_basis, lo, hi):
    """Project the late planar displacement on the normalized t=0 x/z heading."""
    positions = np.asarray(positions, dtype=np.float64)
    initial_basis = np.asarray(initial_basis, dtype=np.float64)
    if (positions.ndim != 2 or positions.shape[1] != 3
            or initial_basis.shape != (3, 3) or not np.isfinite(positions).all()
            or not np.isfinite(initial_basis).all() or not 0 <= lo < hi < len(positions)):
        raise ValueError('Finite 3D positions, initial basis and ordered indices required')
    heading = initial_basis[[0, 2], 0]
    norm = np.linalg.norm(heading)
    if norm <= 0:
        raise ValueError('Nonzero original planar forward heading required')
    return float(np.dot((positions[hi]-positions[lo])[[0, 2]], heading/norm))


def terminal_classification(baseline, candidate):
    """A measured candidate health rejection is a failed mechanism screen."""
    if (baseline is None or baseline.get('stop_reason') != 'duration'
            or baseline.get('controls') != CONTROLS or not baseline.get('native_reference_exact')):
        return 'incomplete_baseline', 'not_tested', False
    if candidate is None:
        return 'incomplete_candidate', 'not_tested', False
    if candidate.get('stop_reason') == 'health_guard':
        return 'valid_measured_health_rejection', 'not_supported', False
    if (candidate.get('stop_reason') != 'duration' or candidate.get('controls') != CONTROLS
            or candidate.get('native_samples') != 95000):
        return 'incomplete_candidate', 'not_tested', False
    return 'valid_complete_body_only_diagnostic', 'evaluate_full_window', True


def physical_advance_from_clocks(start_s, end_s, observed_native_samples):
    """Use advancing physics time; observation count can be short on an error."""
    if (not np.isfinite([start_s, end_s]).all() or end_s < start_s
            or observed_native_samples < 0):
        raise ValueError('Valid monotone physical clock and sample count required')
    return float(end_s-start_s)


def inclusive_arm_wall(start_s, end_s, body_only_wall_s):
    if (not np.isfinite([start_s, end_s, body_only_wall_s]).all()
            or end_s < start_s or body_only_wall_s < 0
            or end_s-start_s + 1e-9 < body_only_wall_s):
        raise ValueError('Inclusive arm timer must cover body, diagnostics and persistence')
    return float(end_s-start_s)


def add_physical_advance(total, advance):
    return None if total is None or advance is None else total + advance


def resource_screen(ledger, whole_wall_s):
    """Use the same inclusive arm limits on complete and rejected runs."""
    advanced = ledger['actual_physical_s']
    if (advanced is None or not np.isfinite(advanced) or not 0 <= advanced <= 19. + 1e-9
            or not np.isfinite(whole_wall_s) or not 0 <= whole_wall_s <= 300):
        return 'limit_exceeded_or_unverifiable'
    for name in ('baseline', 'candidate'):
        arm = ledger.get(name)
        if arm is not None:
            wall = arm.get('wall_s_including_late_diagnostics',
                           arm['wall_s_inclusive_close_persistence'])
            if not np.isfinite(wall) or not 0 <= wall <= 150:
                return 'limit_exceeded_or_unverifiable'
    return 'within_limits'


def phase_queries(source_t, peaks, *, offsets=OFFSETS, expected_counters=COUNTERS):
    """Positive desired phase maps to a later source time: phase advance."""
    source_t = np.asarray(source_t, dtype=np.float64)
    if len(peaks) != 6 or len(offsets) != 6:
        raise ValueError('Six original E1 peak grids required')
    if any(len(p) < 3 or not np.isfinite(p).all() or not np.all(np.diff(p) > 0)
           for p in peaks):
        raise ValueError('Finite increasing actual E1 peaks required')
    phase = np.stack([phase_trace(source_t, p) for p in peaks])
    align = (source_t >= 2.5) & (source_t < 9.5) & np.isfinite(phase).all(axis=0)
    if align.sum() != 1400:
        raise ValueError('Exact complete [2.5,9.5) phase alignment required')
    medians = np.median(phase[:, align] - phase[0, align] - np.asarray(offsets)[:, None], axis=1)
    counters = np.rint(medians).astype(int)
    if not np.array_equal(counters, expected_counters):
        raise ValueError('Frozen integer peak-counter alignment changed')
    times = source_t[1:CONTROLS + 1]
    query = np.broadcast_to(times[:, None], (CONTROLS, 6)).copy()
    active = times > 2.5
    x = np.clip((times[active] - 2.5) / .5, 0, 1)
    blend = 3*x*x - 2*x*x*x
    for leg in range(1, 6):
        desired = phase[0, 1:CONTROLS + 1][active] + offsets[leg] + counters[leg]
        if (not np.isfinite(desired).all() or desired.min() < 0
                or desired.max() > len(peaks[leg]) - 1):
            raise ValueError('Desired phase exceeds actual peak-time brackets')
        inverted = np.interp(desired, np.arange(len(peaks[leg])), peaks[leg])
        query[active, leg] = (1-blend)*times[active] + blend*inverted
    require_grid(source_t, query)
    return query, phase, counters, medians


def map_commands(source_t, query_t, source_commands, source_tendons):
    require_grid(source_t, query_t)
    require_source_tendons(source_tendons)
    commands = np.asarray(source_commands, dtype=np.float64)
    if commands.shape != (2001, 42) or not np.isfinite(commands).all():
        raise ValueError('Finite full 2001x42 source commands required')
    result = commands[1:CONTROLS + 1].copy()
    tendons = {key: np.asarray(value[1:CONTROLS + 1], dtype=np.float64).copy()
               for key, value in source_tendons.items()}
    for leg, name in enumerate(LEGS):
        # Interpolate the seven joint coordinates and that leg's tendon at one query.
        for joint in range(7):
            col = leg*7 + joint
            result[:, col] = np.interp(query_t[:, leg], source_t, commands[:, col])
        key = name + '_tarsus'
        tendons[key] = np.interp(query_t[:, leg], source_t, source_tendons[key])
    # NumPy interpolation at original sample times preserves recorded float values.
    result[:500] = commands[1:501]
    for key, value in tendons.items():
        value[:500] = source_tendons[key][1:501]
    if (not np.array_equal(result[:500], commands[1:501])
            or any(not np.array_equal(tendons[k][:500], source_tendons[k][1:501]) for k in tendons)):
        raise ValueError('2.5s prefix lost bit identity')
    if (not np.isfinite(result).all()
            or any(not np.isfinite(v).all() for v in tendons.values())):
        raise ValueError('Mapped commands must be finite')
    return result, tendons


class NativeGuard(Exception):
    pass


def body_arm(saved, source, command, tendons, out, *, baseline, original_native, whole_start,
             controller_description='External recorded E105 command timing; no neural advance'):
    out.mkdir(exist_ok=False)
    began = time.perf_counter()
    body = None
    rows, native = [], []
    reason = 'duration'
    error = None
    physics_start_s = None
    actual_physical_s = 0.
    try:
        body = new_body(saved)
        physics_start_s = float(body.physics_time())
        old_step = body._step_physics
        shadow = copy.copy(body)
        shadow.d = body.mj.MjData(body.m)
        state_type = body.mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(body.mj.mj_stateSize(body.m, state_type))

        def step():
            old_step()
            body.mj.mj_getState(body.m, body.d, state, state_type)
            body.mj.mj_setState(body.m, shadow.d, state, state_type)
            body.mj.mj_forward(body.m, shadow.d)
            feet = contact_diagnostics(shadow)
            _, rotation, _ = shadow.pose()
            item = (float(body.physics_time()), feet['normal_load_bw'],
                    [np.nan if v is None else v for v in feet['slip_mm_s']],
                    bool(shadow.nonfoot_contact()), float(rotation[2, 2]))
            native.append(item)
            if item[3] or item[4] < .9 or body.fault:
                raise NativeGuard('Native nonfoot/upright/fault guard')

        body._step_physics = step
        observer = SimpleNamespace(body=body, control_tick=0, fault=None,
                                   rate=np.zeros(391), tendon_inputs=dict(source['samples'][0]['tendon_inputs']))
        rows.append(capture_control(observer))
        if baseline:
            for key in ('qpos', 'qvel', 'ctrl'):
                if not np.array_equal(np.asarray(rows[0][key]), np.asarray(source['samples'][0][key])):
                    raise ValueError(f'Baseline {key} mismatch at control 0')
        with (out / 'trace.jsonl').open('w', encoding='utf-8') as stream:
            stream.write(json.dumps(rows[0]) + '\n')
            for tick in range(1, CONTROLS + 1):
                if time.perf_counter()-began > 140 or time.perf_counter()-whole_start > 285:
                    reason = 'wall_limit'
                    break
                original = source['samples'][tick]
                inputs = {key: float(value[tick-1]) for key, value in tendons.items()}
                previous = len(native)
                try:
                    body.step_joint_targets(command[tick-1], np.asarray(original['adhesion'], bool),
                                            tendon_inputs=inputs, interpolate=True)
                except NativeGuard:
                    reason = 'health_guard'
                    break
                observer.control_tick = tick
                observer.tendon_inputs = inputs
                rows.append(capture_control(observer))
                stream.write(json.dumps(rows[-1]) + '\n')
                if baseline:
                    for key in ('qpos', 'qvel', 'ctrl'):
                        if not np.array_equal(np.asarray(rows[-1][key]), np.asarray(original[key])):
                            raise ValueError(f'Baseline {key} mismatch at control {tick}')
                if len(native)-previous != 50:
                    raise ValueError(f'Native sample count mismatch at control {tick}')
                if tick % 200 == 0:
                    print(json.dumps({'arm': out.name, 'time_s': tick*.005,
                                      'wall_s': time.perf_counter()-began}), flush=True)
    except Exception as exc:  # noqa: BLE001 - preserve any partial actual arm
        reason = 'execution_error'
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    finally:
        if body is not None and physics_start_s is not None:
            # The native observer may throw after physics advances but before
            # it appends its diagnostic sample. The simulator clock is decisive.
            try:
                actual_physical_s = physical_advance_from_clocks(
                    physics_start_s, float(body.physics_time()), len(native))
            except Exception as exc:  # noqa: BLE001 - persist raw and close before reporting a bad clock
                actual_physical_s = None
                if error is None:
                    reason = 'physics_clock_error'
                    error = {'type': type(exc).__name__, 'message': str(exc),
                             'traceback': traceback.format_exc()}
        # Preserve raw observations before any fidelity or numerical assertions.
        try:
            payload = {key: value for key, value in source.items() if key != 'samples'}
            payload.update(schema='flylab.m3-e106-body-only-phase-replay.v1',
                           controller=controller_description, samples=rows)
            write_json(out / 'trace.json', payload)
            np.savez_compressed(out / 'native-contact.npz',
                                time_s=np.array([v[0] for v in native]),
                                normal_load_bw=np.array([v[1] for v in native]).reshape(-1, 6),
                                slip_mm_s=np.array([v[2] for v in native]).reshape(-1, 6),
                                nonfoot_contact=np.array([v[3] for v in native]),
                                upright=np.array([v[4] for v in native]))
        except Exception as exc:  # noqa: BLE001 - preserve the earlier arm error
            if error is None:
                error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
                reason = 'persistence_error'
        if body is not None:
            try:
                body.close()
            except Exception as exc:  # noqa: BLE001 - record close cost and error
                if error is None:
                    error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
                    reason = 'close_error'
    result = {'arm': out.name, 'stop_reason': reason, 'controls': max(0, len(rows)-1),
              'actual_physical_s': actual_physical_s, 'native_samples': len(native),
              'native_nonfoot_samples': sum(v[3] for v in native),
              'minimum_upright': min((v[4] for v in native), default=None),
              'fault': None if body is None else body.fault,
              'wall_s_inclusive_close_persistence': time.perf_counter()-began, 'error': error}
    if baseline and reason == 'duration' and len(native) == 95000:
        observed = np.load(out / 'native-contact.npz')
        for key in ('time_s', 'normal_load_bw', 'slip_mm_s', 'nonfoot_contact', 'upright'):
            if not np.array_equal(observed[key], original_native[key][:95000], equal_nan=True):
                result['stop_reason'] = 'native_reference_mismatch'
                result['error'] = {'type': 'ValueError', 'message': f'Baseline native {key} mismatch'}
                break
        else:
            result['native_reference_exact'] = True
    result['wall_s_inclusive_close_persistence'] = time.perf_counter()-began
    write_json(out / 'arm-result.json', result)
    return result


def late_metrics(trace, native_file):
    data = load_model(trace)
    window = (data['t'] >= 3.5) & (data['t'] < 9.0)
    data.update(window=window, moving=window & (data['speed'] > .5), start_s=3.5, end_s=9.)
    quality, _, _ = analyze(data)
    rows = json.loads(trace.read_text(encoding='utf-8'))['samples']
    positions = np.array([r['body']['position'] for r in rows])
    indices = np.flatnonzero(window)
    forward = planar_initial_heading_forward(positions, np.array(rows[0]['body']['basis']),
                                              int(indices[0]), int(indices[-1]))
    path = quality['physical']['path_mm']
    yaw = np.unwrap([r['body']['yaw'] for r in rows])
    native = np.load(native_file)
    selected = (native['time_s'] >= 3.5) & (native['time_s'] < 9.)
    loads = native['normal_load_bw'][selected]
    slip = native['slip_mm_s'][selected]
    support_mask = loads >= .02
    if not np.isfinite(slip[support_mask]).all():
        raise ValueError('Supported native slip must be finite')
    supported = np.where(support_mask, loads, 0.)
    safe_slip = np.where(support_mask, slip, 0.)
    weighted_slip = float(np.sum(supported * safe_slip) / supported.sum()) if supported.sum() else None
    return {'window_s': [3.5, 9.], 'quality': quality,
            'late_yaw_drift_rad': float(yaw[indices[-1]]-yaw[indices[0]]),
            'fixed_initial_heading_forward_mm': forward,
            'net_to_path': quality['physical']['net_mm']/path if path else None,
            'longest_forward_bout_s': quality['window']['longest_forward_bout_s'],
            'six_period_cv': {leg: quality['events'][leg]['period_cv'] for leg in LEGS},
            'four_knee_widths_deg': {leg: quality['knee_interior_deg'][leg]['p95']
                                     - quality['knee_interior_deg'][leg]['p05']
                                     for leg in ('lf', 'rf', 'lh', 'rh')},
            'supported_load_weighted_native_slip_mm_s': weighted_slip,
            'supported_native_load_sum_bw': float(supported.sum())}


def run(out):
    whole_start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    paths = {'trace': SOURCE/'trace.json', 'signals': SOURCE/'signals.npz',
             'start_manifest': SOURCE/'start/manifest.json',
             'native': SOURCE/'native-contact.npz',
             'accepted_review': SOURCE/'parent-second-review.json', 'E85_reference': REFERENCE}
    spec = {'experiment': 'M3-E106-slow-native-phase-replay',
            'scope': 'One precomputed exogenous E105 body-only phase-coherence diagnosis; never final neural gait',
            'source_sha256': {key: sha256(path) for key, path in paths.items()},
            'runner_sha256': sha256(__file__), 'limits': {'body_arms': 2, 'physical_s': 19.,
            'full_neural_s': 0., 'per_arm_wall_s': 150., 'whole_wall_s': 300.},
            'window_s': [3.5, 9.0], 'derivative_data_through_s': 9.5,
            'frozen_offsets_cycles': OFFSETS.tolist(), 'frozen_integer_counters': COUNTERS.tolist()}
    write_json(out/'spec.json', spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    ledger = {'preflight_wall_s': None, 'baseline': None, 'candidate': None,
              'actual_physical_s': 0., 'actual_full_neural_s': 0., 'errors': []}
    try:
        review = json.loads(paths['accepted_review'].read_text())
        require_accepted_review(review)
        source = json.loads(paths['trace'].read_text())
        rows = source['samples']
        if len(rows) != 2001 or any(r['control_tick'] != i or r['time_s'] != i*.005 for i, r in enumerate(rows)):
            raise ValueError('Original complete source trace required')
        if any(any(r['adhesion']) or r['neural_request'] is not None for r in rows):
            raise ValueError('Original no-adhesion/no-push source required')
        signals = np.load(paths['signals'])
        signal_t = signals['time_s']
        source_t = np.array([r['time_s'] for r in rows])
        if not np.array_equal(signal_t, source_t[1:]):
            raise ValueError('Exact signal/control endpoint alignment required')
        observed = signals['observed_ids']
        ids = [np.flatnonzero(observed == value) for value in E1_IDS]
        if any(len(x) != 1 for x in ids):
            raise ValueError('Six unique source E1 identities required')
        rate = require_source_e1_rates(signals['rate'][:, [x[0] for x in ids]])
        zero = np.zeros((1, 6), dtype=np.float64)
        saved = StateStore.load(SOURCE/'start')
        if saved['body']['push_left'] != 0. or not np.array_equal(saved['body']['push_force'], [0., 0., 0.]):
            raise ValueError('Original body must contain no pending external push')
        for leg, graph_id in enumerate(E1_IDS):
            if saved['network']['rate'][graph_id] != 0.:
                raise ValueError(f'Original verified zero E1 state required: {LEGS[leg]}')
        if not np.array_equal(source_t, np.r_[0., signal_t]):
            raise ValueError('Original source zero/endpoints required')
        reference = np.load(REFERENCE)
        ref_phase = reference['phase']
        if (ref_phase.shape != (2000, 6) or not np.isfinite(ref_phase).all()
                or not np.allclose((((ref_phase-ref_phase[:, [0]])/(2*np.pi)-OFFSETS+.5)%1)-.5,
                                   0., atol=1e-12, rtol=0)):
            raise ValueError('E85 stored six-phase tripod offsets required')
        peaks = [quadratic_peaks(source_t, np.r_[zero[0, leg], rate[:, leg]]) for leg in range(6)]
        query, phase, counters, medians = phase_queries(source_t, peaks)
        commands = np.array([r['joint_targets_rad'] for r in rows])
        tendon_keys = rows[0]['tendon_inputs'].keys()
        source_tendons = {key: np.array([r['tendon_inputs'][key] for r in rows]) for key in tendon_keys}
        require_source_tendons(source_tendons)
        mapped, mapped_tendons = map_commands(source_t, query, commands, source_tendons)
        np.savez_compressed(out/'phase-map-and-commands.npz', source_time_s=source_t,
                            control_time_s=source_t[1:CONTROLS+1], E1_ids=E1_IDS,
                            E1_rate_zero_and_endpoints=np.vstack((zero, rate)),
                            source_phase_cycles=phase, E85_reference_phase_rad=ref_phase,
                            reference_offsets_cycles=OFFSETS, integer_counters=counters,
                            median_counter_differences=medians, query_time_s=query,
                            source_commands_rad=commands, mapped_commands_rad=mapped,
                            **{'source_peak_times_'+leg: peak for leg, peak in zip(LEGS, peaks)},
                            **{'source_tendon_'+k: v for k, v in source_tendons.items()},
                            **{'mapped_tendon_'+k: v for k, v in mapped_tendons.items()})
        write_json(out/'phase-map-provenance.json', {'interpolation': 'linear, same query for all seven leg joints and own tarsus tendon',
            'zero_source': 'StateStore original E105 start network rate, six E1 graph IDs',
            'peak_prominence': 20, 'peak_counts': [len(p) for p in peaks],
            'integer_counters': counters.tolist(), 'median_differences': medians.tolist(),
            'max_absolute_query_shift_s': float(np.max(abs(query-source_t[1:CONTROLS+1, None]))),
            'prefix_exact_through_s': 2.5, 'LF_query_identity_exact': True,
            'source_no_adhesion_or_actual_push': True})
        ledger['preflight_wall_s'] = time.perf_counter()-whole_start
        write_json(out/'execution-ledger.json', ledger)
        if ledger['preflight_wall_s'] > 75:
            raise TimeoutError('Preflight exceeded reserved combined budget')
        original_native = np.load(paths['native'])
        baseline_commands = commands[1:CONTROLS+1]
        baseline_tendons = {key: value[1:CONTROLS+1] for key, value in source_tendons.items()}
        baseline_started = time.perf_counter()
        baseline = body_arm(saved, source, baseline_commands, baseline_tendons,
                            out/'baseline', baseline=True, original_native=original_native,
                            whole_start=whole_start)
        ledger['baseline'] = baseline
        ledger['actual_physical_s'] = add_physical_advance(
            ledger['actual_physical_s'], baseline['actual_physical_s'])
        write_json(out/'execution-ledger.json', ledger)
        if (baseline['stop_reason'] != 'duration' or baseline['controls'] != CONTROLS
                or not baseline.get('native_reference_exact') or baseline['wall_s_inclusive_close_persistence'] > 150):
            raise ValueError('Original exact healthy baseline incomplete: candidate forbidden')
        try:
            baseline_late = late_metrics(out/'baseline/trace.json', out/'baseline/native-contact.npz')
            write_json(out/'baseline-late-quality.json', baseline_late)
        finally:
            baseline['wall_s_including_late_diagnostics'] = inclusive_arm_wall(
                baseline_started, time.perf_counter(), baseline['wall_s_inclusive_close_persistence'])
            write_json(out/'execution-ledger.json', ledger)
        if baseline['wall_s_including_late_diagnostics'] > 150:
            raise TimeoutError('Baseline inclusive arm wall exceeded 150s')
        candidate_started = time.perf_counter()
        candidate = body_arm(saved, source, mapped, mapped_tendons,
                             out/'candidate', baseline=False, original_native=None,
                             whole_start=whole_start)
        ledger['candidate'] = candidate
        ledger['actual_physical_s'] = add_physical_advance(
            ledger['actual_physical_s'], candidate['actual_physical_s'])
        write_json(out/'execution-ledger.json', ledger)
        validity, terminal, full_window = terminal_classification(baseline, candidate)
        if terminal == 'not_supported' and not full_window:
            result = {'experiment': spec['experiment'], 'validity': validity,
                      'numerical_screen': 'not_supported', 'reason': 'measured_candidate_native_health_guard',
                      'full_window_metrics': 'unavailable_due_to_early_health_stop',
                      'resource_screen': 'pending_inclusive_accounting',
                      'adoption': 'not_applicable',
                      'actual_physical_s': ledger['actual_physical_s'], 'actual_full_neural_s': 0.}
            write_json(out/'result.json', result)
            candidate['wall_s_including_late_diagnostics'] = inclusive_arm_wall(
                candidate_started, time.perf_counter(), candidate['wall_s_inclusive_close_persistence'])
            result['resource_screen'] = resource_screen(ledger, time.perf_counter()-whole_start)
            write_json(out/'result.json', result)
            write_json(out/'execution-ledger.json', ledger)
            return
        if not full_window:
            raise ValueError('Candidate actual arm incomplete; full-window numerical metrics unavailable')
        try:
            candidate_late = late_metrics(out/'candidate/trace.json', out/'candidate/native-contact.npz')
            write_json(out/'candidate-late-quality.json', candidate_late)
        finally:
            candidate['wall_s_including_late_diagnostics'] = inclusive_arm_wall(
                candidate_started, time.perf_counter(), candidate['wall_s_inclusive_close_persistence'])
            write_json(out/'execution-ledger.json', ledger)
        comparisons = {'baseline': baseline_late, 'candidate': candidate_late}
        yaw_base = abs(comparisons['baseline']['late_yaw_drift_rad'])
        yaw_candidate = abs(comparisons['candidate']['late_yaw_drift_rad'])
        yaw_ratio = yaw_candidate/yaw_base if yaw_base > 0 else None
        quality = comparisons['candidate']['quality']
        cycles = {leg: quality['events'][leg]['complete_cycles'] for leg in LEGS}
        pairs = {key: quality['pair_phase'][key] for key in ('lf-rf', 'lm-rm', 'lh-rh')}
        pair_screen = all(v['mean_cycles'] is not None and
                          abs((v['mean_cycles']-.5+.5)%1-.5) <= .15 and
                          v['resultant_length'] is not None and v['resultant_length'] >= .5
                          for v in pairs.values())
        healthy = (candidate['native_samples'] == 95000 and
                   candidate['native_nonfoot_samples'] == 0 and
                   candidate['minimum_upright'] >= .9 and not candidate['fault'])
        supported = bool(healthy and all(v >= 10 for v in cycles.values())
                         and pair_screen and yaw_ratio is not None and yaw_ratio <= .7)
        result = {'experiment': spec['experiment'], 'validity': validity if healthy else 'incomplete',
                  'numerical_screen': 'supported' if supported else 'not_supported',
                  'resource_screen': resource_screen(ledger, time.perf_counter()-whole_start),
                  'adoption': 'not_applicable', 'final_F3_F5': 'not_assessed_at_9.5s',
                  'cycles': cycles, 'physical_LR_phase': pairs, 'late_absolute_yaw_ratio': yaw_ratio,
                  'baseline': comparisons['baseline'], 'candidate': comparisons['candidate'],
                  'actual_physical_s': ledger['actual_physical_s'], 'actual_full_neural_s': 0.,
                  'whole_wall_s': time.perf_counter()-whole_start}
        write_json(out/'result.json', result)
    except Exception as exc:
        ledger['errors'].append({'type': type(exc).__name__, 'message': str(exc),
                                 'traceback': traceback.format_exc(),
                                 'whole_wall_s_at_error': time.perf_counter()-whole_start})
        write_json(out/'execution-ledger.json', ledger)
        write_json(out/'execution-error.json', {'validity': 'incomplete',
                   'numerical_screen': 'not_tested', 'actual_physical_s': ledger['actual_physical_s'],
                   'actual_full_neural_s': 0., 'whole_wall_s': time.perf_counter()-whole_start,
                   'error': ledger['errors'][-1]})
        raise
    finally:
        ledger['whole_wall_s'] = time.perf_counter()-whole_start
        ledger['resource_screen'] = resource_screen(ledger, ledger['whole_wall_s'])
        write_json(out/'execution-ledger.json', ledger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
