"""One bounded native conditional fit of six existing coupling relays."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.banc_m3_synaptic_efficacy import EfficacyTrial, patch_weights
from tools.probe_c_banc_m3_feedforward_coordination import PATHS
from tools.probe_c_banc_m3_tonic_onset import reduced_rk4
from tools.probe_c_banc_m3_tonic_phase import phase_objective, screens
from tools.verify_c_banc_observation import exact


def existing_coupling_edges(graph, cal, local_ids, local_weights):
    """Resolve declared source relay paths to exact original CSR edge positions."""
    core = cal['core_ids']
    groups = cal['core_groups']
    cell_types = cal['core_cell_types']
    e1 = np.array([int(core[(groups == group) & (cell_types == 0)][0])
        for group in range(6)], dtype=np.int64)
    coupling = cal['coupling_ids']
    edges = []
    rows = []
    cols = []
    for _, target_group, root_id in PATHS:
        relay = int(graph.resolve(['flywire:banc:888:' + root_id])[0])
        if relay not in coupling:
            raise ValueError('Declared PATHS relay is not one of six selected coupling cells')
        target = int(e1[target_group])
        incoming = np.arange(graph.indptr[target], graph.indptr[target + 1])
        match = incoming[graph.indices[incoming] == relay]
        if len(match) != 1:
            raise ValueError('Exactly one existing relay-to-E1 CSR edge required')
        edge = int(match[0])
        row = int(np.flatnonzero(local_ids == target)[0])
        col = int(np.flatnonzero(local_ids == relay)[0])
        if local_weights[row, col] <= 0:
            raise ValueError('Existing selected relay-to-E1 local edge must be positive')
        edges.append(edge)
        rows.append(row)
        cols.append(col)
    if len(set(edges)) != 6 or len(set(cols)) != 6 or len(set(rows)) != 6:
        raise ValueError('Six unique selected output edges and relay cells required')
    return np.asarray(edges), np.asarray(rows), np.asarray(cols), e1


def output_bounds(original, current):
    """Intersect fixed original-source and E89-relative positive bounds."""
    if np.any(original <= 0) or np.any(current <= 0):
        raise ValueError('Six existing positive output weights required')
    lower = np.maximum(current * .5, original * .001)
    upper = np.minimum(current * 2., original * 10.)
    if np.any(lower > upper) or np.any(current < lower) or np.any(current > upper):
        raise ValueError('Existing output must lie inside nonempty frozen bounds')
    return lower, upper


def resource_budget(prior_reduced_neural_s):
    """Reserve the current baseline and selected runtime recheck in a shared cap."""
    if (not np.isfinite(prior_reduced_neural_s) or prior_reduced_neural_s < 0
            or prior_reduced_neural_s % 10 != 0):
        raise ValueError('Prior reduced neural time must be nonnegative 10s calls')
    cap = min(240, int((2420 - prior_reduced_neural_s - 20) // 10))
    if cap < 1:
        raise ValueError('No remaining optimizer calls within shared 2420s cap')
    return cap


def run(root, graph_path, out, *, prior_execution_wall_s=0.,
        prior_reduced_neural_s=0.):
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    call_cap = resource_budget(prior_reduced_neural_s)
    e91 = root / 'e91-native-tonic-onset'
    e92 = root / 'e92-native-tonic-phase-v3'
    e89 = root / 'e89-own-leg-native-recruitment'
    reference = root / 'e85-space-compatible-cadence-reference/commands.npz'
    period = float(np.load(reference)['period_s'])
    if not np.isclose(period, .4196813764904431, rtol=0, atol=1e-12):
        raise ValueError('Frozen E85 cadence period required')
    spec = {'experiment': 'M3-E93-native-coupling-phase',
        'intervention': 'Fix E92 best six tonic taus and E91 exact36 conditional background/zero-state. Fit only six existing coupling taus and six existing positive relay-to-E1 outputs in one log12 least-squares.',
        'bounds': {'coupling_tau_s': [.0125, .134851109828719],
            'output': 'intersection of [.5,2] times E89 output and [.001,10] times original signed source output'},
        'objective': 'Unchanged E92 fixed-query three-window LR and six period residuals; actual E1 peak brackets required. Invalid phase gets a fixed 1000-per-element barrier and cannot become best.',
        'screen': 'Unchanged E92 six cadence and all three LR pairs in each of three windows; both float64 and runtime float32 must pass for candidate.',
        'sampling': 'Native 250us RK4, 5ms stored full36 rates and quadratic E1 peak interpolation.',
        'limit': 'One baseline10s, <=240 optimizer10s calls, one selected float32 recheck10s; <=2420 reduced neural seconds, <=150 fit wall and <=180 total wall, zero full-network/physical stepping.',
        'prior_failed_preflight_wall_s': prior_execution_wall_s,
        'prior_reduced_neural_s': prior_reduced_neural_s,
        'remaining_optimizer_ODE_call_cap': call_cap,
        'preflight_correction': 'E89 calibration stores coupling tau in float64, while actual E90 native tau is float32. Compare calibration cast to native float32 bits with E91 recorded native tau; preserve v1 failure.',
        'E91_inputs_sha256': sha256(e91 / 'reduced-inputs.npz'),
        'E91_baseline_sha256': sha256(e91 / 'baseline-reduced.npz'),
        'E92_best_sha256': sha256(e92 / 'best-observed.npz'),
        'E92_result_sha256': sha256(e92 / 'result.json'),
        'E89_calibration_sha256': sha256(e89 / 'calibration.npz'),
        'E85_commands_sha256': sha256(reference),
        'E85_period_s': period,
        'reduced_helper_sha256': sha256(Path(__file__).with_name('probe_c_banc_m3_tonic_onset.py')),
        'phase_helper_sha256': sha256(Path(__file__).with_name('probe_c_banc_m3_tonic_phase.py')),
        'PATHS_helper_sha256': sha256(Path(__file__).with_name('probe_c_banc_m3_feedforward_coordination.py')),
        'tool_sha256': sha256(__file__)}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    prior = np.load(e91 / 'reduced-inputs.npz')
    original_baseline = np.load(e91 / 'baseline-reduced.npz')
    e92_best = np.load(e92 / 'best-observed.npz')
    cal = {key: value.copy() for key, value in np.load(e89 / 'calibration.npz').items()}
    graph = GraphStore.load(graph_path)
    local_ids = prior['local_ids']
    old_tau = prior['old_tau']
    old_w = prior['local_weights']
    background = prior['background']
    times = prior['time_s']
    if (len(local_ids) != 36 or len(np.unique(local_ids)) != 36
            or not np.array_equal(local_ids[24:30], cal['coupling_ids'])
            or not np.array_equal(local_ids[30:], cal['intrinsic_tonic_ids'])
            or old_w.shape != (36, 36) or background.shape != (2001, 36)
            or not np.array_equal(times, np.arange(2001) * .005)):
        raise ValueError('Exact E91 36-cell input provenance required')
    tonic = e92_best['tau'][30:].astype(float)
    if not np.array_equal(tonic, np.asarray(json.loads((e92 / 'result.json').read_text())['best_tonic_tau_s'])):
        raise ValueError('E92 selected six tonic tau provenance differs')
    if np.any((tonic < .02) | (tonic > period)):
        raise ValueError('E92 tonic tau outside selected bounds')
    edge_ids, rows, cols, e1 = existing_coupling_edges(graph, cal, local_ids, old_w)
    base, _ = rate_weights(graph)
    base = BancTargetEncoder(graph, forward_drive=5000).calibrate_weights(base, cal['motor_ids'])
    current = patch_weights(base, cal['edge_ids'], cal['source_values'], cal['values'])
    original_output = base.data[edge_ids].astype(float)
    current_output = current.data[edge_ids].astype(float)
    np.testing.assert_array_equal(current_output, old_w[rows, cols])
    if np.any(original_output <= 0):
        raise ValueError('Original selected output edges must be positive')
    old_patch_positions = []
    for edge in edge_ids:
        position = np.flatnonzero(cal['edge_ids'] == edge)
        if len(position) != 1:
            raise ValueError('Selected output must occur exactly once in existing E89 patch')
        old_patch_positions.append(int(position[0]))
    old_patch_positions = np.asarray(old_patch_positions)
    np.testing.assert_array_equal(cal['source_values'][old_patch_positions], base.data[edge_ids])
    np.testing.assert_array_equal(cal['values'][old_patch_positions], current.data[edge_ids])
    lower_output, upper_output = output_bounds(original_output, current_output)
    tau_low = .0125
    tau_high = float(old_tau[0])
    if not np.all(old_tau[:24] == tau_high) or tau_high > .134852 or tau_high < .13485:
        raise ValueError('Actual E86 core tau upper bound required')
    tau_patch_positions = []
    for cell in local_ids[24:30]:
        position = np.flatnonzero(cal['tau_ids'] == cell)
        if len(position) != 1:
            raise ValueError('Existing coupling tau patch required once')
        tau_patch_positions.append(int(position[0]))
    tau_patch_positions = np.asarray(tau_patch_positions)
    np.testing.assert_array_equal(cal['tau_values'][tau_patch_positions].astype(np.float32),
        old_tau[24:30].astype(np.float32))
    e1_columns = np.array([int(np.flatnonzero(local_ids == cell)[0]) for cell in e1])
    queries = np.arange(.5, 9.0001, .025)
    np.savez_compressed(out / 'fixed-inputs.npz', edge_ids=edge_ids,
        target_rows=rows, relay_columns=cols, E1_ids=e1,
        E1_columns=e1_columns, original_output=original_output,
        E89_output=current_output, output_lower=lower_output,
        output_upper=upper_output, coupling_ids=local_ids[24:30],
        original_coupling_tau=old_tau[24:30], tonic_tau=tonic,
        tau_lower=tau_low, tau_upper=tau_high, queries=queries,
        selected_cal_edge_positions=old_patch_positions,
        selected_cal_tau_positions=tau_patch_positions)
    initial = np.r_[old_tau[24:30], current_output]
    lower = np.r_[np.full(6, tau_low), lower_output]
    upper = np.r_[np.full(6, tau_high), upper_output]
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError('Frozen E89 starting parameters must be inside bounds')
    residual_size = len(phase_objective(times, e92_best['rate'],
        e1_columns, period, queries)[0])
    if time.perf_counter() - began > 150-prior_execution_wall_s:
        raise TimeoutError('E93 preflight exceeded bounded wall allowance')
    # Charge a started native ODE call conservatively even if it raises midway.
    write_json(out / 'execution-progress.json', {'baseline_ODE_calls': 1,
        'optimizer_ODE_calls': 0, 'runtime_recheck_ODE_calls': 0,
        'prior_reduced_neural_s': prior_reduced_neural_s,
        'current_reduced_neural_s': 10.,
        'cumulative_reduced_neural_s': prior_reduced_neural_s + 10.,
        'counting': 'Started calls charged as full 10s if interrupted'})
    baseline = reduced_rk4(old_w, background, old_tau, 2000)
    np.savez_compressed(out / 'baseline-verified.npz', time_s=times,
        rate=baseline, old_tau=old_tau, local_weights=old_w)
    write_json(out / 'execution-progress.json', {'baseline_ODE_calls': 1,
        'optimizer_ODE_calls': 0, 'runtime_recheck_ODE_calls': 0,
        'prior_reduced_neural_s': prior_reduced_neural_s,
        'current_reduced_neural_s': 10.,
        'cumulative_reduced_neural_s': prior_reduced_neural_s + 10.})
    np.testing.assert_allclose(baseline, original_baseline['rate'], rtol=0, atol=1e-12)
    reduced_s = 10.
    calls = 0
    history = []
    best = None
    fit_began = time.perf_counter()

    def objective(log_values):
        nonlocal calls, reduced_s, best
        if (calls >= call_cap or prior_reduced_neural_s + reduced_s + 20 > 2420
                or time.perf_counter() - fit_began > 150
                or time.perf_counter() - began > 150-prior_execution_wall_s):
            raise TimeoutError('E93 bounded ODE call or wall allowance exhausted')
        proposed = np.exp(log_values)
        if not np.isfinite(proposed).all() or np.any(proposed < lower) or np.any(proposed > upper):
            raise ValueError('Nonfinite or out-of-bound optimizer proposal')
        tau = old_tau.copy()
        tau[24:30] = proposed[:6]
        tau[30:] = tonic
        weights = old_w.copy()
        weights[rows, cols] = proposed[6:]
        calls += 1
        reduced_s += 10
        write_json(out / 'execution-progress.json', {'baseline_ODE_calls': 1,
            'optimizer_ODE_calls': calls, 'runtime_recheck_ODE_calls': 0,
            'prior_reduced_neural_s': prior_reduced_neural_s,
            'current_reduced_neural_s': reduced_s,
            'cumulative_reduced_neural_s': prior_reduced_neural_s + reduced_s,
            'counting': 'Started calls charged as full 10s if interrupted'})
        rate = reduced_rk4(weights, background, tau, 2000)
        record = {'call': calls, 'coupling_tau_s': proposed[:6].tolist(),
            'output_weights': proposed[6:].tolist(),
            'wall_s': time.perf_counter() - began}
        np.savez_compressed(out / f'call-{calls:03d}.npz',
            time_s=times, tau=tau, local_weights=weights, rate=rate,
            E1_rate=rate[:, e1_columns])
        if not np.isfinite(rate).all():
            record.update(valid=False, error='Nonfinite reduced neural state')
            history.append(record)
            write_json(out / 'history.json', history)
            raise ValueError('Nonfinite reduced neural state')
        try:
            residual, peaks, _, periods = phase_objective(times, rate,
                e1_columns, period, queries)
        except ValueError as error:
            record.update(valid=False, error=str(error), barrier_value=1000.)
            history.append(record)
            write_json(out / 'history.json', history)
            return np.full(residual_size, 1000.)
        loss = float(np.dot(residual, residual))
        record.update(valid=True, loss=loss, periods_s=periods,
            E1_peak_counts=[len(points) for points in peaks])
        history.append(record)
        write_json(out / 'history.json', history)
        np.savez_compressed(out / f'call-{calls:03d}-objective.npz',
            residual=residual, periods=periods,
            **{f'peaks_{i}': points for i, points in enumerate(peaks)})
        if best is None or loss < best[0]:
            best = (loss, calls, tau.copy(), weights.copy(), rate.copy())
            np.savez_compressed(out / 'best-observed.npz',
                score=loss, call=calls, tau=tau,
                local_weights=weights, rate=rate, residual=residual)
        return residual

    try:
        fit = least_squares(objective, np.log(initial),
            bounds=(np.log(lower), np.log(upper)),
            diff_step=1e-3, max_nfev=240,
            xtol=1e-8, ftol=1e-8, gtol=1e-8)
        solver = {'success': bool(fit.success), 'message': fit.message,
            'nfev': int(fit.nfev), 'cost': float(fit.cost)}
    except (ValueError, TimeoutError) as error:
        solver = {'success': False, 'message': str(error),
            'error': type(error).__name__}
    fit_wall = time.perf_counter() - fit_began
    write_json(out / 'solver-status.json', solver)
    if best is None:
        write_json(out / 'result.json', {'validity': 'invalid_candidate_phase',
            'calls': calls, 'reduced_neural_s': reduced_s,
            'fit_wall_s': fit_wall, 'wall_s': time.perf_counter() - began,
            'actual_full_neural_s': 0, 'actual_physical_s': 0, 'solver': solver})
        return
    loss, best_call, tau, weights, rate = best
    rhythms, pairs = screens(times, rate, e1_columns, period)
    float64_pass = all(item['screen'] for item in rhythms) and all(
        item['screen'] for item in pairs.values())
    runtime_tau = tau.astype(np.float32).astype(float)
    runtime_weights = weights.astype(np.float32).astype(float)
    reduced_s += 10
    write_json(out / 'execution-progress.json', {'baseline_ODE_calls': 1,
        'optimizer_ODE_calls': calls, 'runtime_recheck_ODE_calls': 1,
        'prior_reduced_neural_s': prior_reduced_neural_s,
        'current_reduced_neural_s': reduced_s,
        'cumulative_reduced_neural_s': prior_reduced_neural_s + reduced_s,
        'counting': 'Started calls charged as full 10s if interrupted'})
    runtime = reduced_rk4(runtime_weights, background, runtime_tau, 2000)
    np.savez_compressed(out / 'selected-runtime-float32.npz', time_s=times,
        tau=runtime_tau, local_weights=runtime_weights, rate=runtime,
        E1_rate=runtime[:, e1_columns])
    runtime_rhythms, runtime_pairs = screens(times, runtime, e1_columns, period)
    runtime_pass = all(item['screen'] for item in runtime_rhythms) and all(
        item['screen'] for item in runtime_pairs.values())
    result = {'experiment': spec['experiment'],
        'validity': 'valid_conditional_diagnostic',
        'hypothesis_verdict': 'conditional_coupling_supported' if float64_pass and runtime_pass else 'not_supported',
        'adoption': 'not_applicable', 'E91_first_peak_failure_preserved': True,
        'E92_later_phase_failure_preserved': True,
        'screen': bool(float64_pass and runtime_pass),
        'float64_screen': bool(float64_pass), 'runtime_float32_screen': bool(runtime_pass),
        'rhythms': rhythms, 'LR_by_window': pairs,
        'runtime_rhythms': runtime_rhythms, 'runtime_LR_by_window': runtime_pairs,
        'best_call': best_call, 'best_loss': loss,
        'best_coupling_tau_s': tau[24:30].tolist(),
        'best_output_weights': weights[rows, cols].tolist(),
        'solver': solver, 'calls': calls, 'reduced_neural_s': reduced_s,
        'prior_reduced_neural_s': prior_reduced_neural_s,
        'cumulative_reduced_neural_s': prior_reduced_neural_s + reduced_s,
        'actual_full_neural_s': 0, 'actual_physical_s': 0,
        'fit_wall_s': fit_wall, 'wall_s': time.perf_counter() - began,
        'prior_execution_wall_s': prior_execution_wall_s,
        'native_integration_dt_s': .00025, 'stored_rate_dt_s': .005,
        'body_validated': False}
    write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)
    if not result['screen']:
        return
    if time.perf_counter() - began > 170-prior_execution_wall_s:
        raise TimeoutError('Insufficient E93 wall allowance for neutral construction')
    next_cal = {key: value.copy() for key, value in cal.items()}
    next_cal['values'][old_patch_positions] = runtime_weights[rows, cols]
    next_cal['tau_values'][tau_patch_positions] = runtime_tau[24:30]
    next_cal['tau_ids'] = np.r_[next_cal['tau_ids'], local_ids[30:]]
    next_cal['tau_values'] = np.r_[next_cal['tau_values'], runtime_tau[30:]]
    if len(next_cal['tau_ids']) != 38:
        raise ValueError('Original32 tau plus tonic6 patches required')
    rebuilt = patch_weights(base, next_cal['edge_ids'], next_cal['source_values'], next_cal['values'])
    other = np.ones(len(rebuilt.data), dtype=bool)
    other[edge_ids] = False
    np.testing.assert_array_equal(rebuilt.data[other], current.data[other])
    np.testing.assert_array_equal(rebuilt.data[edge_ids], runtime_weights[rows, cols].astype(np.float32))
    for key, value in cal.items():
        if key not in ('values', 'tau_ids', 'tau_values'):
            np.testing.assert_array_equal(next_cal[key], value)
    np.savez_compressed(out / 'calibration.npz', **next_cal)
    anatomy = json.loads((e89 / 'muscle-provenance.json').read_text())
    anatomy['coupling_later_phase_hypothesis'] = {
        'selected_output_edges': edge_ids.tolist(),
        'coupling_tau_s': runtime_tau[24:30].tolist(),
        'tonic_tau_s': runtime_tau[30:].tolist(),
        'conditional_only': True}
    write_json(out / 'muscle-provenance.json', anatomy)
    original = StateStore.load(root / 'e32-core-adaptation/candidate/start')
    base_trial = EfficacyTrial(graph, original, next_cal['edge_ids'],
        next_cal['source_values'], next_cal['values'],
        zero_adaptation_ids=next_cal['zero_adaptation_ids'])
    session = IntrinsicRecruitmentTrial(RecruitmentTrial(base_trial, next_cal,
        anatomy['muscles']), next_cal['tau_ids'], next_cal['tau_values'])
    try:
        neutral = session.snapshot()
        exact(original['body'], neutral['body'], 'E93 original neutral body')
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            exact(original['network'][key], neutral['network'][key],
                'E93 original neutral neural ' + key)
        StateStore.save(out / 'candidate-neutral', neutral)
        clone = IntrinsicRecruitmentTrial.from_checkpoint(graph,
            StateStore.load(out / 'candidate-neutral'))
        try:
            exact(neutral, clone.snapshot(), 'E93 neutral disk exact')
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
    parser.add_argument('--prior-execution-wall-s', type=float, default=0.)
    parser.add_argument('--prior-reduced-neural-s', type=float, default=0.)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve existing experiment output: ' + str(args.out))
    start = time.perf_counter()
    try:
        run(args.root, args.graph, args.out,
            prior_execution_wall_s=args.prior_execution_wall_s,
            prior_reduced_neural_s=args.prior_reduced_neural_s)
    except Exception as error:
        if args.out.exists():
            progress_path = args.out / 'execution-progress.json'
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
                'baseline_ODE_calls': 0, 'optimizer_ODE_calls': 0,
                'runtime_recheck_ODE_calls': 0,
                'prior_reduced_neural_s': args.prior_reduced_neural_s,
                'current_reduced_neural_s': 0.,
                'cumulative_reduced_neural_s': args.prior_reduced_neural_s}
            write_json(args.out / 'execution-error.json', {
                'validity': 'incomplete', 'hypothesis_verdict': 'not_tested',
                'error': type(error).__name__, 'message': str(error),
                'execution_wall_s': time.perf_counter() - start,
                'actual_full_neural_s': 0, 'actual_physical_s': 0,
                **progress})
        raise
