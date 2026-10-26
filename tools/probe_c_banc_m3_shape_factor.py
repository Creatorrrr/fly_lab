"""E107: two frozen external E85 within-leg shape body factors."""

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flylab.body import LEGS
from flylab.c.storage import StateStore
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.fit_c_banc_m3_native_all_leg import reference_at_phase
from tools.probe_c_banc_m3_slow_phase_replay import (
    CONTROLS,
    E1_IDS,
    OFFSETS,
    REFERENCE,
    SOURCE,
    body_arm,
    inclusive_arm_wall,
    late_metrics,
    phase_queries,
    phase_trace,
    quadratic_peaks,
    require_accepted_review,
    require_grid,
    require_source_e1_rates,
    require_source_tendons,
)

ROOT_E106 = ROOT / 'verification/banc-natural-gait-m3-20260919/e106-slow-native-phase-replay'
CONTROLLER = ('External E85 own-leg reference shape blended into recorded E105 body commands; '
              'no live BANC neural advance')
KNEE_MIN = {'lf': 25., 'rf': 30., 'lh': 40., 'rh': 35.}


def blend_weights(times):
    times = np.asarray(times, dtype=np.float64)
    if times.shape != (CONTROLS,) or not np.isfinite(times).all():
        raise ValueError('Complete finite 1900 control times required')
    x = np.clip((times-2.5)/.5, 0., 1.)
    return 3*x*x-2*x*x*x


def shape_commands(source_t, query, native_base, peaks, reference):
    require_grid(source_t, query)
    base = np.asarray(native_base, dtype=np.float64)
    if base.shape != (CONTROLS, 42) or not np.isfinite(base).all():
        raise ValueError('Complete finite native base commands required')
    times = np.asarray(source_t)[1:CONTROLS+1]
    active = times > 2.5
    phase = np.empty((int(active.sum()), 6), dtype=np.float64)
    target = np.empty((int(active.sum()), 42), dtype=np.float64)
    for leg in range(6):
        query_leg = query[active, leg]
        peak = np.asarray(peaks[leg], dtype=np.float64)
        if (peak.ndim != 1 or len(peak) < 3 or not np.isfinite(peak).all()
                or not np.all(np.diff(peak) > 0)
                or query_leg.min() < peak[0] or query_leg.max() > peak[-1]):
            raise ValueError(f'Actual original E1 phase brackets required: {LEGS[leg]}')
        phase[:, leg] = phase_trace(query_leg, peak)
        rows = reference_at_phase(reference, phase[:, leg]-OFFSETS[leg], 0)
        target[:, 7*leg:7*(leg+1)] = rows[:, 7*leg:7*(leg+1)]
    if not np.isfinite(phase).all() or not np.isfinite(target).all():
        raise ValueError('Finite own-leg phases and reference target required')
    blend = blend_weights(times)
    output = base.copy()
    output[active] = (1-blend[active, None])*base[active]+blend[active, None]*target
    if not np.array_equal(output[:500], base[:500]) or not np.isfinite(output).all():
        raise ValueError('Bit-exact original prefix and finite 42-command output required')
    return output, phase, blend, target


def reference_validation(reference):
    time_s = np.asarray(reference['time_s'])
    command = np.asarray(reference['command'])
    phase = np.asarray(reference['phase'])
    if (time_s.shape != (2000,) or command.shape != (2000, 42)
            or phase.shape != (2000, 6) or not np.isfinite(command).all()
            or not np.isfinite(phase).all() or not np.array_equal(time_s, np.arange(1, 2001)*.005)):
        raise ValueError('Original complete finite E85 reference required')
    offsets = (((phase-phase[:, [0]])/(2*np.pi)-OFFSETS+.5)%1)-.5
    if not np.allclose(offsets, 0., rtol=0, atol=1e-12):
        raise ValueError('Original E85 six phase offsets required')
    train = (time_s >= 5.) & (time_s < 8.)
    holdout = (time_s >= 8.) & (time_s < 10.)
    reconstructed = reference_at_phase(reference, phase[holdout, 0]/(2*np.pi), 0, train)
    rms = float(np.sqrt(np.mean((reconstructed-command[holdout])**2)))
    if abs(rms-.0001314196828899847) > 1e-12 or rms > .002:
        raise ValueError('Frozen E85 5..8 to 8..10 interpolation validation failed')
    return rms


def prefix_exact(rows, reference_rows, native, reference_native):
    if len(rows) < 501 or len(reference_rows) < 501 or len(native['time_s']) < 25000:
        raise ValueError('Complete original 2.5s boundary/native prefix required')
    for tick in range(501):
        for key in ('qpos', 'qvel', 'ctrl'):
            if not np.array_equal(np.asarray(rows[tick][key]), np.asarray(reference_rows[tick][key])):
                raise ValueError(f'Original E106 baseline prefix {key} mismatch at {tick}')
    for key in ('time_s', 'normal_load_bw', 'slip_mm_s', 'nonfoot_contact', 'upright'):
        if not np.array_equal(native[key][:25000], reference_native[key][:25000], equal_nan=True):
            raise ValueError(f'Original E106 native prefix {key} mismatch')
    return True


def arm_terminal(arm):
    if arm['actual_physical_s'] is None:
        return 'technical_unknown_physics_clock', 'not_tested', False
    if arm['stop_reason'] == 'health_guard':
        return 'valid_measured_native_health_rejection', 'not_supported', False
    if (arm['stop_reason'] == 'duration' and arm['controls'] == CONTROLS
            and arm['native_samples'] == 95000 and arm['native_nonfoot_samples'] == 0
            and arm['minimum_upright'] is not None and arm['minimum_upright'] >= .9
            and not arm['fault']):
        return 'valid_complete_body_only_diagnostic', 'evaluate', True
    return 'technical_incomplete', 'not_tested', False


def shape_screen(metrics):
    quality = metrics['quality']
    cycles = {leg: quality['events'][leg]['complete_cycles'] for leg in LEGS}
    cv = {leg: quality['events'][leg]['period_cv'] for leg in LEGS}
    pairs = {key: quality['pair_phase'][key] for key in ('lf-rf', 'lm-rm', 'lh-rh')}
    knees = metrics['four_knee_widths_deg']
    cycle_ok = all(value >= 10 for value in cycles.values())
    cv_ok = all(value is not None and np.isfinite(value) and value <= .3 for value in cv.values())
    pair_ok = all(value['mean_cycles'] is not None and
                  abs((value['mean_cycles']-.5+.5)%1-.5) <= .15 and
                  value['resultant_length'] is not None and value['resultant_length'] >= .5
                  for value in pairs.values())
    knee_ok = all(knees[leg] >= limit for leg, limit in KNEE_MIN.items())
    return {'supported': bool(cycle_ok and cv_ok and pair_ok and knee_ok),
            'six_cycles': cycles, 'six_period_cv': cv, 'three_physical_LR_pairs': pairs,
            'four_knee_widths_deg': knees,
            'gate': {'six_cycles_at_least_10': cycle_ok, 'six_CV_at_most_point3': cv_ok,
                     'three_LR_antiphase': pair_ok, 'four_knee_minima': knee_ok}}


def resource_screen(ledger, elapsed):
    advance = ledger['actual_physical_s']
    if (advance is None or not np.isfinite(advance) or not 0 <= advance <= 19.+1e-9
            or not np.isfinite(elapsed) or not 0 <= elapsed <= 300):
        return 'limit_exceeded_or_unverifiable'
    for name in ('shape_only', 'shape_plus_timing'):
        arm = ledger[name]
        if arm is not None:
            wall = arm.get('wall_s_including_late_diagnostics',
                           arm['wall_s_inclusive_close_persistence'])
            if not np.isfinite(wall) or not 0 <= wall <= 150:
                return 'limit_exceeded_or_unverifiable'
    return 'within_limits'


def verify_shared_helper_metadata_only(path, reviewed_sha256):
    current = Path(path).read_text(encoding='utf-8')
    signature = ("def body_arm(saved, source, command, tendons, out, *, baseline, original_native, whole_start,\n"
                 "             controller_description='External recorded E105 command timing; no neural advance'):")
    prior_signature = "def body_arm(saved, source, command, tendons, out, *, baseline, original_native, whole_start):"
    label = 'controller=controller_description, samples=rows)'
    prior_label = "controller='External recorded E105 command timing; no neural advance', samples=rows)"
    if current.count(signature) != 1 or current.count(label) != 1:
        raise ValueError('E106 helper metadata-only revision not uniquely identifiable')
    reconstructed = current.replace(signature, prior_signature).replace(label, prior_label)
    if hashlib.sha256(reconstructed.encode('utf-8')).hexdigest() != reviewed_sha256:
        raise ValueError('E106 helper changed beyond optional controller metadata')
    return True


def run(out):
    began = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    helper = ROOT/'tools/probe_c_banc_m3_slow_phase_replay.py'
    review = json.loads((ROOT_E106/'parent-second-review.json').read_text(encoding='utf-8'))
    first_review = json.loads((ROOT_E106/'parent-first-review.json').read_text(encoding='utf-8'))
    e106_spec = json.loads((ROOT_E106/'spec.json').read_text(encoding='utf-8'))
    sources = {'E105_trace': SOURCE/'trace.json', 'E105_signals': SOURCE/'signals.npz',
               'E105_native': SOURCE/'native-contact.npz', 'E105_start': SOURCE/'start/manifest.json',
               'E105_review': SOURCE/'parent-second-review.json',
               'E106_review': ROOT_E106/'parent-second-review.json',
               'E106_phase_map': ROOT_E106/'phase-map-and-commands.npz',
               'E106_baseline_trace': ROOT_E106/'baseline/trace.json',
               'E106_baseline_native': ROOT_E106/'baseline/native-contact.npz',
               'E106_timing_trace': ROOT_E106/'candidate/trace.json',
               'E106_timing_native': ROOT_E106/'candidate/native-contact.npz',
               'E85_reference': REFERENCE,
               'evaluator': ROOT/'tools/analyze_c_banc_biological_gait.py',
               'reference_interpolator': ROOT/'tools/fit_c_banc_m3_native_all_leg.py'}
    helper_before = review['validation']['source_sha256']
    spec = {'experiment': 'M3-E107-within-leg-shape-factor',
            'scope': 'Two predetermined external recorded/reference shape body arms; no neural gait adoption',
            'source_sha256': {key: sha256(path) for key, path in sources.items()},
            'E106_shared_body_helper_before_metadata_change_sha256': helper_before,
            'E106_shared_body_helper_after_metadata_change_sha256': sha256(helper),
            'helper_change': 'Optional truthful controller_description only; numerical stepping unchanged',
            'runner_sha256': sha256(__file__),
            'limits': {'new_body_arms': 2, 'new_physical_s': 19., 'full_neural_s': 0.,
                       'per_arm_wall_s': 150., 'whole_wall_s': 300.},
            'window_s': [3.5, 9.0], 'derivative_data_through_s': 9.5,
            'order': ['shape_only', 'shape_plus_timing']}
    write_json(out/'spec.json', spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    ledger = {'preflight_wall_s': None, 'shape_only': None, 'shape_plus_timing': None,
              'actual_physical_s': 0., 'actual_full_neural_s': 0., 'errors': []}
    try:
        require_accepted_review(review)
        require_accepted_review(json.loads(sources['E105_review'].read_text(encoding='utf-8')))
        verify_shared_helper_metadata_only(helper, helper_before)
        for key, frozen in {'E105_trace': 'trace', 'E105_signals': 'signals',
                            'E105_native': 'native', 'E105_start': 'start_manifest',
                            'E85_reference': 'E85_reference'}.items():
            if sha256(sources[key]) != e106_spec['source_sha256'][frozen]:
                raise ValueError(f'Reviewed original source changed: {key}')
        for relative, expected in first_review['original_artifact_sha256'].items():
            if sha256(ROOT_E106/relative) != expected:
                raise ValueError(f'Reviewed E106 artifact changed: {relative}')
        if not all(review['original_16_file_hashes_unchanged'].values()):
            raise ValueError('E106 parent second review original manifest not intact')
        payload = json.loads(sources['E105_trace'].read_text(encoding='utf-8'))
        rows = payload['samples']
        source_t = np.array([row['time_s'] for row in rows])
        if (len(rows) != 2001 or not np.array_equal(source_t, np.arange(2001)*.005)
                or any(row['control_tick'] != i or any(row['adhesion'])
                       or row['neural_request'] is not None for i, row in enumerate(rows))):
            raise ValueError('Complete original no-adhesion/no-push E105 source required')
        saved = StateStore.load(SOURCE/'start')
        if (saved['body']['modelHash'] != payload['body_hash']
                or saved['network']['model_hash'] != payload['model_hash']
                or saved['adapter']['hash'] != payload['adapter_hash']
                or saved['graph_hash'] != payload['graph_hash']
                or any(payload['parameters'][key] != value for key, value in saved['parameters'].items())
                or saved['body']['push_left'] != 0.
                or not np.array_equal(saved['body']['push_force'], [0., 0., 0.])):
            raise ValueError('Original E105 body/model/adapter/graph/rest/world source mismatch')
        if sha256(sources['evaluator']) != payload['parameters']['diagnostic']['evaluator_sha256']:
            raise ValueError('Original frozen physical gait evaluator changed')
        signals = np.load(sources['E105_signals'])
        if not np.array_equal(signals['time_s'], source_t[1:]):
            raise ValueError('E105 signal endpoint grid mismatch')
        columns = [np.flatnonzero(signals['observed_ids'] == graph_id) for graph_id in E1_IDS]
        if any(len(column) != 1 for column in columns):
            raise ValueError('Six original E1 graph identities required')
        rates = require_source_e1_rates(signals['rate'][:, [int(c[0]) for c in columns]])
        if any(saved['network']['rate'][graph_id] != 0. for graph_id in E1_IDS):
            raise ValueError('Original E1 initial rates must all be zero')
        peaks = [quadratic_peaks(source_t, np.r_[0., rates[:, leg]]) for leg in range(6)]
        query, _, counters, _ = phase_queries(source_t, peaks)
        phase_map = np.load(sources['E106_phase_map'])
        if (not np.array_equal(query, phase_map['query_time_s'])
                or not np.array_equal(counters, phase_map['integer_counters'])):
            raise ValueError('Independently recomputed E106 source query differs')
        reference = np.load(REFERENCE)
        reference_rms = reference_validation(reference)
        source_commands = np.asarray([row['joint_targets_rad'] for row in rows], dtype=np.float64)
        if source_commands.shape != (2001, 42) or not np.isfinite(source_commands).all():
            raise ValueError('Complete finite original E105 commands required')
        source_tendons = {key: np.array([row['tendon_inputs'][key] for row in rows])
                          for key in rows[0]['tendon_inputs']}
        require_source_tendons(source_tendons)
        time_query = np.broadcast_to(source_t[1:CONTROLS+1, None], (CONTROLS, 6)).copy()
        bases = {'shape_only': source_commands[1:CONTROLS+1],
                 'shape_plus_timing': phase_map['mapped_commands_rad']}
        queries = {'shape_only': time_query, 'shape_plus_timing': query}
        tendons = {'shape_only': {key: value[1:CONTROLS+1] for key, value in source_tendons.items()},
                   'shape_plus_timing': {key: phase_map['mapped_tendon_'+key] for key in source_tendons}}
        proposals = {}
        frozen = {'source_time_s': source_t, 'control_time_s': source_t[1:CONTROLS+1],
                  'original_source_command_rad': source_commands, 'E1_ids': np.array(E1_IDS),
                  'E85_reference_phase_rad': reference['phase']}
        for name in ('shape_only', 'shape_plus_timing'):
            require_grid(source_t, queries[name])
            output, phases, blend, target = shape_commands(
                source_t, queries[name], bases[name], peaks, reference)
            if not np.array_equal(output[:500], source_commands[1:501]):
                raise ValueError('E107 first500 source commands not bitexact')
            if any(value.shape != (CONTROLS,) or not np.isfinite(value).all()
                   for value in tendons[name].values()):
                raise ValueError('Finite original unchanged E107 tendon commands required')
            proposals[name] = {'query': queries[name], 'base': bases[name],
                               'command': output, 'phase': phases, 'blend': blend,
                               'target': target, 'tendons': tendons[name]}
            frozen[name+'_query_s'] = queries[name]
            frozen[name+'_native_base_command_rad'] = bases[name]
            frozen[name+'_command_rad'] = output
            frozen[name+'_source_phase_cycles'] = phases
            frozen[name+'_reference_target_rad'] = target
            frozen[name+'_blend'] = blend
            frozen.update({name+'_tendon_'+key: value for key, value in tendons[name].items()})
        expected = np.load(ROOT/'verification/banc-natural-gait-m3-20260919/parent-e107-design-proposal.npz')
        for name in proposals:
            if not np.array_equal(proposals[name]['command'], expected[name+'_command']):
                raise ValueError(f'Frozen parent {name} command proposal differs')
            if not np.array_equal(proposals[name]['phase'], expected[name+'_source_phase']):
                raise ValueError(f'Frozen parent {name} source phase proposal differs')
        if not np.array_equal(blend[500:], expected['blend']):
            raise ValueError('Frozen E107 smoothstep differs')
        np.savez_compressed(out/'frozen-both-inputs.npz', **frozen)
        write_json(out/'inputs-lock.json', {'frozen_both_inputs_sha256': sha256(out/'frozen-both-inputs.npz'),
            'source_hashes': spec['source_sha256'], 'reference_validation_RMS_rad': reference_rms,
            'all_bracketed_finite': True, 'first500_original_commands_exact': True,
            'intervention': 'Only seven joint coordinates per leg blend to original E85 own-leg phase reference; tendons unchanged by shape',
            'max_abs_coordinate_change_rad': {name: float(np.max(abs(item['command']-item['base'])))
                                                for name, item in proposals.items()},
            'postblend_RMS_coordinate_change_rad': {name: float(np.sqrt(np.mean(
                (item['command'][source_t[1:CONTROLS+1] >= 3.]
                 - item['base'][source_t[1:CONTROLS+1] >= 3.])**2)))
                for name, item in proposals.items()}})
        # All four fixed control trajectories can be evaluated before the new arms.
        old_controls = {}
        for name, dirname in (('baseline', 'baseline'), ('timing_only', 'candidate')):
            value = late_metrics(ROOT_E106/dirname/'trace.json', ROOT_E106/dirname/'native-contact.npz')
            reviewed = json.loads((ROOT_E106/f'sol-first-corrected-{dirname}-late-quality.json').read_text(encoding='utf-8'))
            if value != reviewed:
                raise ValueError(f'Reviewed E106 {name} late metrics changed')
            old_controls[name] = value
        write_json(out/'reviewed-controls-late-quality.json', old_controls)
        reference_rows = json.loads((ROOT_E106/'baseline/trace.json').read_text(encoding='utf-8'))['samples'][:501]
        reference_native = np.load(ROOT_E106/'baseline/native-contact.npz')
        ledger['preflight_wall_s'] = time.perf_counter()-began
        write_json(out/'execution-ledger.json', ledger)
        if ledger['preflight_wall_s'] > 75 or time.perf_counter()-began > 75:
            raise TimeoutError('E107 preflight exceeded reserved whole experiment budget')
        outcomes = {}
        for name in ('shape_only', 'shape_plus_timing'):
            arm_start = time.perf_counter()
            arm = None
            try:
                arm = body_arm(saved, payload, proposals[name]['command'], proposals[name]['tendons'],
                               out/name, baseline=False, original_native=None, whole_start=began,
                               controller_description=CONTROLLER+'; arm='+name)
                ledger[name] = arm
                prior = ledger['actual_physical_s']
                ledger['actual_physical_s'] = None if prior is None or arm['actual_physical_s'] is None else prior+arm['actual_physical_s']
                write_json(out/'execution-ledger.json', ledger)
                # The original 2.5s was unchanged, so require every boundary and native sample.
                observed_rows = json.loads((out/name/'trace.json').read_text(encoding='utf-8'))['samples']
                observed_native = np.load(out/name/'native-contact.npz')
                prefix_exact(observed_rows, reference_rows, observed_native, reference_native)
                validity, verdict, full_window = arm_terminal(arm)
                if verdict == 'not_tested':
                    raise ValueError(f'{name} technical actual arm failure: {arm["stop_reason"]}')
                metrics = None
                screen = None
                if full_window:
                    metrics = late_metrics(out/name/'trace.json', out/name/'native-contact.npz')
                    write_json(out/f'{name}-late-quality.json', metrics)
                    screen = shape_screen(metrics)
                    verdict = 'supported' if screen['supported'] else 'not_supported'
                outcomes[name] = {'validity': validity, 'numerical_screen': verdict,
                                  'full_window_metrics': 'available' if full_window else 'unavailable_after_health_guard',
                                  'shape_screen': screen, 'late_metrics': metrics,
                                  'actual_physical_s': arm['actual_physical_s']}
                write_json(out/f'{name}-outcome.json', outcomes[name])
            finally:
                if arm is not None:
                    arm['wall_s_including_late_diagnostics'] = inclusive_arm_wall(
                        arm_start, time.perf_counter(), arm['wall_s_inclusive_close_persistence'])
                elif (out/name/'arm-result.json').exists():
                    recovered = json.loads((out/name/'arm-result.json').read_text(encoding='utf-8'))
                    ledger[name] = recovered
                    prior = ledger['actual_physical_s']
                    advance = recovered['actual_physical_s']
                    ledger['actual_physical_s'] = None if prior is None or advance is None else prior+advance
                    recovered['wall_s_including_late_diagnostics'] = time.perf_counter()-arm_start
                elif (out/name).exists():
                    ledger['actual_physical_s'] = None  # actual advancement is unknown
                write_json(out/'execution-ledger.json', ledger)
            if resource_screen(ledger, time.perf_counter()-began) != 'within_limits':
                raise TimeoutError(f'{name} physical or wall resource limit exceeded')
        result = {'experiment': spec['experiment'], 'validity': 'valid_completed_two_factor_body_diagnostic',
                  'resource_screen': resource_screen(ledger, time.perf_counter()-began),
                  'outcomes': outcomes, 'reviewed_controls': old_controls,
                  'actual_new_physical_s': ledger['actual_physical_s'],
                  'actual_full_neural_s': 0., 'adoption': 'not_applicable',
                  'final_10s_F3_F1_F8': 'not_assessed',
                  'whole_wall_s': time.perf_counter()-began}
        write_json(out/'result.json', result)
    except Exception as exc:
        ledger['errors'].append({'type': type(exc).__name__, 'message': str(exc),
                                 'traceback': traceback.format_exc(),
                                 'whole_wall_s_at_error': time.perf_counter()-began})
        write_json(out/'execution-ledger.json', ledger)
        write_json(out/'execution-error.json', {'validity': 'incomplete',
            'numerical_screen': 'not_tested_for_uncompleted_factor',
            'actual_new_physical_s': ledger['actual_physical_s'],
            'actual_full_neural_s': 0., 'error': ledger['errors'][-1]})
        raise
    finally:
        ledger['whole_wall_s'] = time.perf_counter()-began
        ledger['resource_screen'] = resource_screen(ledger, ledger['whole_wall_s'])
        write_json(out/'execution-ledger.json', ledger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
