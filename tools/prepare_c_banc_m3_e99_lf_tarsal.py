"""Prepare two existing LF tarsal premotor rows from frozen E98 observations."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.banc_m3_synaptic_efficacy import EfficacyTrial, patch_weights
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.probe_c_banc_m3_tonic_onset import phase_trace, quadratic_peaks, reduced_rk4
from tools.verify_c_banc_observation import exact


def inverse_native_rate(rate):
    rate = np.asarray(rate, dtype=float)
    if not np.isfinite(rate).all() or np.any(abs(rate) >= 200):
        raise ValueError('Inverse native rate requires finite values strictly inside cap200')
    return 7.5 + 200 * np.arctanh(rate / 200)


def deterministic_levator_map(i2, tonic_contribution, residual_background):
    """One fixed 5/95 anchor map; no search, clipping or fit."""
    low, high = np.percentile(i2, [5, 95])
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError('Observed LF I2 5/95 anchors must be distinct')
    k = float((inverse_native_rate(95) - inverse_native_rate(5)) / (high - low))
    intercept = float(inverse_native_rate(95) + k * low)
    denominator = float(np.mean(tonic_contribution))
    factor = float((intercept - np.mean(residual_background)) / denominator)
    if not np.isfinite([k, intercept, denominator, factor]).all() or factor <= 0 or denominator <= 0:
        raise ValueError('Analytical positive tonic factor required')
    return {'I2_p05': float(low), 'I2_p95': float(high), 'negative_weight': -k,
        'intercept': intercept, 'tonic_factor': factor,
        'retained_background_mean': float(np.mean(residual_background)),
        'current_tonic_contribution_mean': denominator}


def source_screen(times, predicted, presynaptic, target, period, *, sign):
    times = np.asarray(times)
    predicted = np.asarray(predicted)
    presynaptic = np.asarray(presynaptic)
    target = np.asarray(target)
    if (times.shape != (2001,) or not np.array_equal(times, np.arange(2001) * .005)
            or any(value.shape != times.shape for value in (predicted, presynaptic, target))):
        raise ValueError('Complete original 2001-point E98 source time grid required')
    mask = (times >= 2.5) & (times < 9.5)
    if (int(mask.sum()) != 1400 or not np.isfinite(predicted).all()
            or not np.isfinite(presynaptic[mask]).all()
            or not np.isfinite(target[mask]).all()):
        raise ValueError('Every fixed-window E98 own-LF source and target sample must be finite')
    def corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    width = float(np.percentile(predicted[mask], 95) - np.percentile(predicted[mask], 5))
    peaks = quadratic_peaks(times, predicted)
    within = peaks[(peaks >= 2.5) & (peaks < 9.5)]
    intervals = np.diff(within)
    mean = float(intervals.mean()) if len(intervals) else None
    cv = float(intervals.std() / intervals.mean()) if len(intervals) else None
    source_corr = corr(predicted[mask], sign * presynaptic[mask])
    target_corr = corr(predicted[mask], target[mask])
    passed = bool(width >= 20 and len(intervals) >= 12 and mean is not None
        and abs(mean / period - 1) <= .1 and cv is not None and cv <= .3
        and source_corr is not None and source_corr >= .8
        and target_corr is not None and target_corr >= .5)
    return {'width_p95_p05': width, 'complete_cycles': len(intervals),
        'period_mean_s': mean, 'period_CV': cv,
        'source_sign_Pearson': source_corr, 'own_LF_muscle_target_Pearson': target_corr,
        'samples': int(mask.sum()), 'screen': passed, 'peak_times_s': peaks.tolist()}


def run(root, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    e98 = root / 'e98-native-readout-body-discriminator'
    body = e98 / 'body'
    spec = {'experiment': 'M3-E99-LF-tarsal-source-preparation',
        'intervention': 'Existing60038 E2 positive input1.1707944, all other existing incoming original*.001, gain8to0; existing11088 original I2 negative plus two positive tonic inputs mapped once from E98 I2 5/95 anchors and observed retained background. No MN edge/tau/body/world/initial state change.',
        'conditional_diagnostic': 'Two native250us RK4 rate responses to linearly interpolated observed E98 endpoint currents. No full-network or physical advance, optimization, teacher, or runtime phase.',
        'limit': 'One preparation <=90 wall seconds including errors; no rerun/grid.',
        'e98_calibration_sha256': sha256(e98 / 'calibration.npz'),
        'e98_result_sha256': sha256(body / 'result.json'),
        'e98_signals_sha256': sha256(body / 'recruitment-signals.npz'),
        'e98_metadata_sha256': sha256(body / 'recruitment-metadata.npz'),
        'e98_parent_review_sha256': sha256(e98 / 'parent-first-review.json'),
        'e85_reference_sha256': sha256(root / 'e85-space-compatible-cadence-reference/commands.npz'),
        'warm_inverse_sha256': sha256(root / 'e89-own-leg-native-recruitment/warm-start.npz'),
        'tool_sha256': sha256(__file__)}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    graph = GraphStore.load(graph_path)
    cal = {key: value.copy() for key, value in np.load(e98 / 'calibration.npz').items()}
    old = {key: value.copy() for key, value in cal.items()}
    signal = np.load(body / 'recruitment-signals.npz')
    meta = np.load(body / 'recruitment-metadata.npz')
    ids = signal['observed_ids']
    if (len(ids) != 446 or not np.array_equal(ids, meta['observed_ids'])
            or signal['rate'].shape != (2000, 446)
            or not np.array_equal(signal['time_s'], np.arange(1, 2001) * .005)
            or len(cal['tau_ids']) != 38 or cal['theta'].shape != (96, 48)):
        raise ValueError('Exact complete E98 observations and 48-column legacy theta required')
    col = {int(cell): int(np.flatnonzero(ids == cell)[0]) for cell in
        (60038, 11088, 29608, 105382, 106670, 51845, 53861)}
    base, _ = rate_weights(graph)
    base = BancTargetEncoder(graph, forward_drive=5000).calibrate_weights(base, cal['motor_ids'])
    current = patch_weights(base, cal['edge_ids'], cal['source_values'], cal['values'])
    target_60038 = np.arange(graph.indptr[60038], graph.indptr[60038 + 1])
    target_60038 = target_60038[base.data[target_60038] != 0]
    e2 = target_60038[base.indices[target_60038] == 29608]
    target_11088 = np.arange(graph.indptr[11088], graph.indptr[11088 + 1])
    selected_11088 = {pre: target_11088[base.indices[target_11088] == pre]
        for pre in (105382, 106670, 51845)}
    if (len(e2) != 1 or any(len(v) != 1 for v in selected_11088.values())
            or 60038 in cal['input_ids'] or 60038 in cal['zero_adaptation_ids']
            or 60038 in cal['motor_ids'] or 11088 in cal['motor_ids']
            or 60038 in BancTargetEncoder(graph, forward_drive=5000).ports
            or 11088 in BancTargetEncoder(graph, forward_drive=5000).ports
            or not np.array_equal(current.data[target_60038], base.data[target_60038])
            or float(base.data[e2[0]]) <= 0
            or float(base.data[selected_11088[105382][0]]) >= 0
            or any(float(base.data[selected_11088[x][0]]) <= 0 for x in (106670, 51845))):
        raise ValueError('Existing disjoint LF input rows, signs and E98 row ownership required')
    for source, post, expected_root in ((29608, 60038, '720575941511186508'),
            (105382, 11088, '720575941433046798')):
        if str(graph.nodes[post]['root_id']) != expected_root or source == post:
            raise ValueError('Original LF anatomical identities required')
    if (float(meta['_cell_gain'][col[60038]]) != 8
            or float(meta['_cell_gain'][col[11088]]) != 0
            or any(float(meta['tau'][col[x]]) != float(np.float32(.02)) for x in (60038, 11088))
            or np.any(signal['drive'][:, [col[60038], col[11088]]] != 0)):
        raise ValueError('Exact original LF native gain, tau and drive required')
    times = np.r_[0., signal['time_s']]
    actual_rate = np.vstack((np.zeros((1, len(ids))), signal['rate']))
    recurrent = np.vstack((np.zeros((1, len(ids))), signal['recurrent_current']))
    adaptation = np.vstack((np.zeros((1, len(ids))), signal['adaptation']))
    mask = (times >= 2.5) & (times < 9.5)
    current_three = {pre: float(current.data[edge[0]]) for pre, edge in selected_11088.items()}
    rate = {pre: actual_rate[:, col[pre]] for pre in (29608, 105382, 106670, 51845)}
    background_11088 = recurrent[:, col[11088]] - sum(
        current_three[pre] * rate[pre] for pre in current_three)
    tonic = sum(current_three[pre] * rate[pre] for pre in (106670, 51845))
    mapping = deterministic_levator_map(rate[105382][mask], tonic[mask],
        background_11088[mask])
    values_60038 = base.data[target_60038] * np.float32(.001)
    values_60038[target_60038 == e2[0]] = np.float32(1.1707944)
    requested = {int(edge): float(value) for edge, value in zip(target_60038, values_60038)}
    requested.update({int(selected_11088[105382][0]): mapping['negative_weight']})
    requested.update({int(selected_11088[pre][0]): current_three[pre] * mapping['tonic_factor']
        for pre in (106670, 51845)})
    bounds = {edge: value / float(base.data[edge]) for edge, value in requested.items()}
    if (not np.isfinite(list(bounds.values())).all()
            or any(not .001 <= value <= 10 for value in bounds.values())):
        raise ValueError('Every original-source efficacy must be finite in [.001,10]')
    write_json(out / 'analytical-map.json', {**mapping,
        'requested_edge_ids': list(requested), 'original_efficacies': bounds,
        'changed_60038_incoming': len(target_60038),
        'changed_11088_incoming': 3})
    # Retain each row's actual unselected recurrent contribution. The whole-network
    # trajectory is recorded, not integrated again here.
    e2_original = float(current.data[e2[0]])
    background_60038 = (recurrent[:, col[60038]] - e2_original * rate[29608]) * .001
    proposed_60038 = background_60038 + float(np.float32(1.1707944)) * rate[29608]
    proposed_11088 = background_11088 + sum(
        float(np.float32(requested[int(selected_11088[pre][0])])) * rate[pre]
        for pre in (105382, 106670, 51845))
    inputs = np.column_stack((proposed_60038, proposed_11088))
    predicted = reduced_rk4(np.zeros((2, 2)), inputs, np.full(2, float(np.float32(.02))), 2000)
    warm = np.load(root / 'e89-own-leg-native-recruitment/warm-start.npz')
    anatomy = json.loads((e98 / 'muscle-provenance.json').read_text())
    if (anatomy['muscles'][14]['leg'] != 'lf'
            or anatomy['muscles'][14]['muscle'] != 'tarsus_depressor_muscle'
            or anatomy['muscles'][16]['leg'] != 'lf'
            or anatomy['muscles'][16]['muscle'] != 'tarsus_levator_muscle'
            or warm['warm_muscles'].shape != (84, 96)):
        raise ValueError('Frozen LF tarsal inverse target provenance required')
    peaks_lf = quadratic_peaks(times, actual_rate[:, col[53861]])
    phase_lf = phase_trace(times, peaks_lf)
    bins = warm['reference_phase_bins']
    target = np.full((len(times), 2), np.nan)
    valid = np.isfinite(phase_lf)
    for position, muscle in enumerate((14, 16)):
        target[valid, position] = np.interp(np.mod(phase_lf[valid], 1),
            np.r_[bins, 1], np.r_[warm['warm_muscles'][:, muscle], warm['warm_muscles'][0, muscle]])
    np.savez_compressed(out / 'conditional-raw.npz', time_s=times,
        source_ids=np.array([60038, 11088]), predicted_rate=predicted,
        proposed_input_current=inputs, retained_background=np.column_stack((
            background_60038, background_11088)), actual_rate=actual_rate[:, [col[60038], col[11088]]],
        actual_adaptation=adaptation[:, [col[60038], col[11088]]],
        E2_rate=rate[29608], I2_rate=rate[105382], tonic_contribution=tonic,
        LF_E1_peaks=peaks_lf, LF_phase=phase_lf, own_LF_inverse_target=target)
    period = float(np.load(root / 'e85-space-compatible-cadence-reference/commands.npz')['period_s'])
    screens = {
        'depressor_60038': source_screen(times, predicted[:, 0], rate[29608], target[:, 0], period, sign=1),
        'levator_11088': source_screen(times, predicted[:, 1], rate[105382], target[:, 1], period, sign=-1)}
    write_json(out / 'conditional-screen.json', screens)
    # All preflight and diagnostics above precede the one CUDA constructor.
    position = {int(edge): index for index, edge in enumerate(cal['edge_ids'])}
    for edge, value in requested.items():
        if edge in position:
            cal['values'][position[edge]] = np.float32(value)
        else:
            position[edge] = len(cal['edge_ids'])
            cal['edge_ids'] = np.r_[cal['edge_ids'], edge]
            cal['source_values'] = np.r_[cal['source_values'], base.data[edge]]
            cal['values'] = np.r_[cal['values'], np.float32(value)]
    cal['input_ids'] = np.r_[cal['input_ids'], 60038]
    cal['zero_adaptation_ids'] = np.union1d(cal['zero_adaptation_ids'], [60038])
    changed = patch_weights(base, cal['edge_ids'], cal['source_values'], cal['values'])
    other = np.ones(len(changed.data), dtype=bool)
    other[list(requested)] = False
    exact(changed.data[other], current.data[other], 'All nonselected original CSR data')
    exact(changed.indices, current.indices, 'Original CSR indices')
    exact(changed.indptr, current.indptr, 'Original CSR indptr')
    exact(changed[cal['motor_ids']].data, current[cal['motor_ids']].data, 'All391 motor receiving rows')
    for key, value in old.items():
        if key not in ('edge_ids', 'source_values', 'values', 'input_ids', 'zero_adaptation_ids'):
            exact(cal[key], value, 'E98 unchanged calibration ' + key)
    anatomy['input_nodes'].append(graph.nodes[60038])
    anatomy['E99_LF_tarsal_source'] = {'conditional_only': True,
        'existing_source_theta_48_columns_unchanged': True,
        'selected_source_count': 49, 'changed_input_rows': [60038, 11088],
        'screen': {key: value['screen'] for key, value in screens.items()}}
    np.savez_compressed(out / 'calibration.npz', **cal)
    write_json(out / 'muscle-provenance.json', anatomy)
    original = StateStore.load(root / 'e32-core-adaptation/candidate/start')
    base_trial = EfficacyTrial(graph, original, cal['edge_ids'], cal['source_values'],
        cal['values'], zero_adaptation_ids=cal['zero_adaptation_ids'])
    session = IntrinsicRecruitmentTrial(RecruitmentTrial(base_trial, cal,
        anatomy['muscles']), cal['tau_ids'], cal['tau_values'])
    try:
        if session.network.device != 'cuda':
            raise ValueError('Actual CUDA constructor required')
        actual = source_matrix(session.network)
        exact(actual.data, changed.data, 'E99 CUDA runtime all weights')
        exact(actual.indices, changed.indices, 'E99 CUDA CSR columns')
        exact(actual.indptr, changed.indptr, 'E99 CUDA CSR offsets')
        if (session.network._cell_gain[60038].item() != 0
                or session.network._cell_gain[11088].item() != 0
                or session.network.drive[60038].item() != 0
                or session.network.drive[11088].item() != 0
                or any(session.network.tau[i].item() != float(np.float32(.02))
                    for i in (60038, 11088))):
            raise ValueError('Native LF gain/drive/tau contract required')
        neutral = session.snapshot()
        exact(original['body'], neutral['body'], 'Original neutral physical start')
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            exact(original['network'][key], neutral['network'][key], 'Original neutral neural ' + key)
        StateStore.save(out / 'candidate-neutral', neutral)
    finally:
        session.close()
    clone = IntrinsicRecruitmentTrial.from_checkpoint(graph, StateStore.load(out / 'candidate-neutral'))
    try:
        exact(neutral, clone.snapshot(), 'Whole E99 neutral disk exact')
    finally:
        clone.close()
    wall = time.perf_counter() - began
    result = {'experiment': spec['experiment'],
        'validity': 'valid_conditional_preparation' if wall <= 90 else 'wall_limit_incomplete',
        'conditional_source_screen': bool(all(x['screen'] for x in screens.values())),
        'screen': screens, 'source_count': 49,
        'legacy_theta_columns': 48, 'changed_input_rows': [60038, 11088],
        'all_391_motor_rows_exact': True, 'all_38_tau_exact': True,
        'neutral_disk_exact': True, 'actual_neural_s': 0., 'actual_physical_s': 0.,
        'fit_calls': 0, 'wall_s': wall, 'resource_limit_met': wall <= 90,
        'body_trial': 'not_run'}
    write_json(out / 'result.json', result)
    if wall > 90:
        raise TimeoutError('E99 preparation exceeded 90 wall seconds')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'graph', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve existing experiment output: ' + str(args.out))
    started = time.perf_counter()
    try:
        run(args.root, args.graph, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error),
                'execution_wall_s': time.perf_counter() - started,
                'actual_neural_s': 0., 'actual_physical_s': 0., 'fit_calls': 0})
        raise
