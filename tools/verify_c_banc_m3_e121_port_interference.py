"""E121: one fixed target-port efficacy intervention in the native BANC body.

The only intervention is the existing 20 inhibitory port -> motor projection
divided by 64. This file must not be run before the frozen source review.
"""

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from flylab.c.walking_trace import capture_control
from tools.analyze_c_banc_biological_gait import analyze, load_model, sha256
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial, patch_tau
from tools.banc_m3_synaptic_efficacy import patch_weights
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.verify_c_banc_m3_e113_body import (
    AdvanceLedger,
    NativeHealthStop,
    body_quality,
    finite_error,
    install_native_observer,
    jsonable,
    load_npz,
    native_arrays,
    observation_ids,
    prepare_neutral_checkpoint,
    save_json,
    source_diagnostics,
    supported_slip,
)
from tools.verify_c_banc_observation import exact

EXPERIMENT = 'M3-E121-target-port-interference'
DESIGN_SHA = '8e1490a933d1c5b46f1df2bc2be34cf47ec07e8c8a04e95ed83144a1e5640a92'
FREEZE_SHA = '0b56b866823641412787b2c34bee8d9fe2291a7473fc453c11d3f364ef5c6d86'
ROOT = Path(__file__).resolve().parents[1]
PRIOR_WALL_S = 778.0
PREPARE_LIMIT_S = 45.0
BODY_LIMIT_S = 150.0
PROCESS_LIMIT_S = 195.0
SHARED_LIMIT_S = 988.0
RESERVE_S = 8.0
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
DIAGNOSTIC_INPUT_SHA = {
    'e85-space-compatible-cadence-reference/commands.npz':
        'a3e80d23ae458b26099abb4c347062a1b6f1a57eda898bf81fc6c7db58383eeb',
    'e89-own-leg-native-recruitment/warm-start.npz':
        'cd3ce65923fc3578bdaa31509da245ad1086e9f72a4c3a09331e9042543a89a7',
}


def reviewed_inputs(root):
    """Hash-check frozen data without constructing a graph, body, or network."""
    design = ROOT/'docs/BANC_M3_E121_TARGET_PORT_INTERFERENCE_DESIGN_20260920.md'
    freeze_path = root/'e121-frozen-design.json'
    if sha256(design) != DESIGN_SHA or sha256(freeze_path) != FREEZE_SHA:
        raise ValueError('Frozen E121 design or resource contract changed')
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    for relative, digest in freeze['input_sha256'].items():
        if sha256(root/relative) != digest:
            raise ValueError('Frozen E121 input changed: '+relative)
    for relative, digest in freeze['source_sha256'].items():
        if sha256(ROOT/relative) != digest:
            raise ValueError('Frozen E121 source changed: '+relative)
    for relative, digest in DIAGNOSTIC_INPUT_SHA.items():
        if sha256(root/relative) != digest:
            raise ValueError('Frozen E121 source diagnostic input changed: '+relative)
    review = json.loads((root/'e113-dense-foot-native-body/parent-final-review.json')
                        .read_text(encoding='utf-8'))
    if (review.get('accepted_as_valid_actual_diagnostic') is not True
            or review.get('blocking_defects') != []):
        raise ValueError('Accepted E113 actual native baseline required')
    originals = review.get('original_artifact_sha256')
    if not isinstance(originals, dict) or len(originals) != 207:
        raise ValueError('Original E113 207-artifact review required')
    for relative, digest in originals.items():
        name = Path(relative)
        if (name.is_absolute() or '..' in name.parts
                or sha256(root/'e113-dense-foot-native-body'/name) != digest):
            raise ValueError('Original E113 artifact changed: '+relative)
    cal = load_npz(root/'e113-dense-foot-native-body/calibration.npz')
    oracle = load_npz(root/'parent-e121-expected-calibration.npz')
    patch_oracle = load_npz(root/'parent-e121-port-patch.npz')
    observation = load_npz(root/'parent-e113-expected-observation.npz')
    startup = load_npz(root/'e103-startup-preserving-recruitment/startup-inputs.npz')
    if (set(cal) != set(oracle) or cal['edge_ids'].shape != (130943,)
            or cal['values'].dtype != np.float32 or len(cal['tau_ids']) != 38
            or len(cal['motor_ids']) != 391 or len(cal['individual_edge_ids']) != 1572
            or len(patch_oracle['port_ids']) != 20
            or len(patch_oracle['edge_ids']) != 872):
        raise ValueError('Frozen E113/parent calibration dimensions required')
    if not np.array_equal(cal['tau_ids'], observation['original_tau_ids']):
        raise ValueError('Original 38 tau IDs changed')
    exact(cal['tau_values'], observation['original_tau_values'], 'Original38 tau values')
    return cal, oracle, patch_oracle, observation, startup, freeze


def derive_port_patch(graph, cal, port_ids, base_unscaled, base_calibrated,
                      expected_cal=None, expected_patch=None):
    """Find declared-port CSR entries independently; map unsorted cal IDs explicitly."""
    ports = np.asarray(port_ids)
    motors = np.asarray(cal['motor_ids'])
    if (ports.shape != (20,) or len(np.unique(ports)) != 20
            or len(motors) != 391 or len(np.unique(motors)) != 391
            or np.any(ports < 0) or np.any(ports >= graph.n)):
        raise ValueError('Declared 20 port and 391 MN identity required')
    port_mask = np.zeros(graph.n, bool)
    motor_mask = np.zeros(graph.n, bool)
    port_mask[ports] = True
    motor_mask[motors] = True
    post = np.repeat(np.arange(graph.n, dtype=np.int64), np.diff(graph.indptr))
    edge_ids = np.flatnonzero(port_mask[graph.indices] & motor_mask[post])
    pre = np.asarray(graph.indices[edge_ids], dtype=np.int64)
    receiving = post[edge_ids]
    if (len(edge_ids) != 872 or len(np.unique(receiving)) != 382
            or np.intersect1d(edge_ids, cal['individual_edge_ids']).size
            or not np.array_equal(base_calibrated.indptr, base_unscaled.indptr)
            or not np.array_equal(base_calibrated.indices, base_unscaled.indices)):
        raise ValueError('Exact declared-port existing projection required')
    positions = {int(edge): i for i, edge in enumerate(cal['edge_ids'])}
    if len(positions) != len(cal['edge_ids']):
        raise ValueError('Duplicate calibration edge IDs')
    try:
        cal_positions = np.asarray([positions[int(edge)] for edge in edge_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError('Every port projection must already be calibrated') from error
    before = np.asarray(cal['values'][cal_positions], dtype=np.float32)
    after = np.asarray(before / np.float32(64), dtype=np.float32)
    if (not np.all(before < 0) or not np.all(after < 0)
            or not np.array_equal(cal['source_values'][cal_positions], before)
            or not np.array_equal(base_calibrated.data[edge_ids], before)
            or not np.array_equal(base_unscaled.data[edge_ids], after)
            or not np.array_equal(before, base_unscaled.data[edge_ids]*np.float32(64))):
        raise ValueError('Original64x signed port weights and source must agree exactly')
    changed = {key: value.copy() for key, value in cal.items()}
    changed['values'][cal_positions] = after
    unchanged = np.ones(len(cal['edge_ids']), bool)
    unchanged[cal_positions] = False
    for key in cal:
        if key == 'values':
            exact(changed[key][unchanged], cal[key][unchanged], 'Unchanged non-port values')
        else:
            exact(changed[key], cal[key], 'Unchanged E113 calibration '+key)
        if expected_cal is not None:
            exact(changed[key], expected_cal[key], 'Independent E121 oracle '+key)
    patch = {'edge_ids': edge_ids, 'calibration_positions': cal_positions,
             'port_ids': ports.astype(np.int32), 'edge_pre': pre,
             'edge_post': receiving, 'before': before, 'after': after,
             'unchanged_calibration_value_mask': unchanged}
    if expected_patch is not None:
        for key in expected_patch:
            exact(patch[key], expected_patch[key], 'Independent port patch '+key)
    return changed, patch


def prepare(root, graph_path, out, began):
    cal, oracle, expected_patch, obs, startup, freeze = reviewed_inputs(root)
    graph = GraphStore.load(graph_path)
    if (graph.hash != freeze['original_graph_hash'] or graph.n != 158706
            or graph.indptr[-1] != len(graph.indices)):
        raise ValueError('Frozen original full BANC graph required')
    original = StateStore.load(root/'e32-core-adaptation/candidate/start')
    prior_neutral = StateStore.load(root/'e113-dense-foot-native-body/candidate-neutral')
    exact(original['body'], prior_neutral['body'], 'Original E32/E113 body')
    exact(original['world'], prior_neutral['world'], 'Original E32/E113 world')
    for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
        exact(original['network'][key], prior_neutral['network'][key],
              'Original E32/E113 neural '+key)
    for key in ('cuts', 'navigation', 'parameters', 'seed', 'control_tick',
                'cpg_hash', 'initial_position', 'signed_forward_mm'):
        exact(original[key], prior_neutral[key], 'Original E32/E113 '+key)
    if original['network']['tick'] != 0 or original['control_tick'] != 0:
        raise ValueError('Original whole-network/body zero start required')
    tau = patch_tau(np.full(graph.n, .02, np.float32), cal['tau_ids'], cal['tau_values'])
    exact(tau[cal['motor_ids']], obs['native_motor_tau'], 'Original actual MN tau')
    encoder = BancTargetEncoder(graph, forward_drive=5000)
    unscaled, _ = rate_weights(graph)
    scaled = encoder.calibrate_weights(unscaled, cal['motor_ids'])
    changed, patch = derive_port_patch(graph, cal, encoder.ports, unscaled, scaled,
                                       oracle, expected_patch)
    before_w = patch_weights(scaled, cal['edge_ids'], cal['source_values'], cal['values'])
    after_w = patch_weights(scaled, changed['edge_ids'], changed['source_values'],
                            changed['values'])
    exact(before_w.indptr, after_w.indptr, 'Whole original CSR indptr')
    exact(before_w.indices, after_w.indices, 'Whole original CSR indices')
    other = np.ones(len(before_w.data), bool)
    other[patch['edge_ids']] = False
    exact(before_w.data[other], after_w.data[other], 'All non-port whole W')
    exact(after_w.data[patch['edge_ids']], patch['after'], 'Actual port whole W')
    exact(np.signbit(before_w.data), np.signbit(after_w.data), 'Whole W signs')
    source_dense = after_w[cal['motor_ids']][:, cal['individual_source_basis_ids']].toarray()
    exact(source_dense.astype(float), obs['actual_float32_source_weights'],
          'Unchanged original49 source-to-MN W')
    port_before = before_w[cal['motor_ids']][:, encoder.ports].toarray().astype(np.float32)
    port_after = after_w[cal['motor_ids']][:, encoder.ports].toarray().astype(np.float32)
    np.savez_compressed(out/'calibration.npz', **changed)
    np.savez_compressed(out/'port-patch.npz', **patch)
    np.savez_compressed(out/'port-projection.npz', port_ids=encoder.ports,
        motor_ids=cal['motor_ids'], before_64x_W=port_before, actual_W=port_after,
        source49_ids=cal['individual_source_basis_ids'], source49_W=source_dense)
    anatomy = json.loads((root/'e110-e109-native-body-diagnostic/muscle-provenance.json')
                         .read_text(encoding='utf-8'))
    save_json(out/'muscle-provenance.json', anatomy)
    prepare_neutral_checkpoint(graph, original, changed, anatomy, after_w, tau, out)
    actual_neutral = StateStore.load(out/'candidate-neutral')
    for key in ('cuts', 'navigation', 'parameters', 'seed', 'control_tick',
                'cpg_hash', 'initial_position', 'signed_forward_mm'):
        exact(original[key], actual_neutral[key], 'E121 unchanged original '+key)
    exact(prior_neutral['adapter'], actual_neutral['adapter'],
          'E121 unchanged E113 actual muscle adapter')
    elapsed = time.perf_counter()-began
    receipt = {'experiment': EXPERIMENT, 'validity': 'prepared' if elapsed <= PREPARE_LIMIT_S
               else 'resource_limit_incomplete', 'wall_s': elapsed,
               'resource_limit_met': bool(elapsed <= PREPARE_LIMIT_S),
               'changed_calibration_fields': ['values'], 'changed_edges': len(patch['edge_ids']),
               'actual_neural_s': 0.0, 'actual_physical_s': 0.0, 'fit_calls': 0,
               'full_tau_exact': True, 'whole_zero_disk_exact': True}
    save_json(out/'preparation-result.json', receipt)
    elapsed = time.perf_counter()-began
    if elapsed > PREPARE_LIMIT_S and receipt['resource_limit_met']:
        receipt.update(validity='resource_limit_incomplete', wall_s=elapsed,
                       resource_limit_met=False)
        save_json(out/'preparation-result.json', receipt)
    return graph, changed, obs, startup, patch, receipt


def original_startup(rows, native, startup):
    observed = max(0, min(len(rows)-1, 99))
    times = np.asarray([row['time_s'] for row in rows[1:observed+1]])
    q = np.asarray([row['joint_targets_rad'] for row in rows[1:observed+1]], dtype=float)
    order = tuple(f'{leg}_tarsus' for leg in LEGS)
    ltm = np.asarray([[row['tendon_inputs'][key] for key in order]
                      for row in rows[1:observed+1]], dtype=float)
    complete = bool(observed == 99 and len(native) >= 4950
        and np.array_equal(times, np.arange(1, 100)*.005)
        and q.shape == (99, 42) and ltm.shape == (99, 6)
        and startup['target_command'].shape == (99, 42)
        and startup['target_tendon'].shape == (99, 6))
    missing = {'RMS': None, 'max_abs': None, 'available': False}
    qm = finite_error(q, startup['target_command']) if complete else missing.copy()
    lm = finite_error(ltm, startup['target_tendon']) if complete else missing.copy()
    return {'complete_original99_prefix': complete, 'observed_controls': observed,
        'original_E100_E103_q': qm, 'original_E100_E103_LTM': lm,
        'original_E100_E103_startup_screen': bool(complete and qm['available']
            and lm['available'] and qm['RMS'] <= .02 and qm['max_abs'] <= .05
            and lm['RMS'] <= .01 and lm['max_abs'] <= .03),
        'tendon_order': list(order)}


def save_startup_raw(out, rows, startup):
    n = max(0, min(len(rows)-1, 99))
    q = np.asarray([r['joint_targets_rad'] for r in rows[1:n+1]], dtype=float).reshape(n, 42)
    ltm = np.asarray([[r['tendon_inputs'][f'{leg}_tarsus'] for leg in LEGS]
                      for r in rows[1:n+1]], dtype=float).reshape(n, 6)
    np.savez_compressed(out/'actual-startup-original99-raw.npz',
        time_s=np.asarray([r['time_s'] for r in rows[1:n+1]]), actual_command=q,
        actual_LTM=ltm, original_E100_E103_command=startup['target_command'],
        original_E100_E103_LTM=startup['target_tendon'])


def observed_signals(session, detailed, ports, available=None):
    """One actual recurrent-current evaluation supplies both observation sets."""
    if available is None:
        available = {}
    current = session.network._current(session.network.rate)
    available['current_detailed_tensor'] = current[detailed]
    available['current_port_tensor'] = current[ports]
    detailed_values = available.setdefault('detail', {})
    detailed_values['rate'] = session.network.readout(detailed)
    detailed_values['recurrent_current'] = (
        available['current_detailed_tensor'].cpu().numpy().copy())
    detailed_values['drive'] = session.network.drive[detailed].cpu().numpy().copy()
    detailed_values['adaptation'] = (
        session.network.adaptation[detailed].cpu().numpy().copy())
    detailed_values['muscle_activation'] = session.adapter.activation.copy()
    available['detail_complete'] = True
    port_values = available.setdefault('port', {})
    port_values['rate'] = session.network.readout(ports)
    port_values['recurrent_current'] = (
        available['current_port_tensor'].cpu().numpy().copy())
    port_values['drive'] = session.network.drive[ports].cpu().numpy().copy()
    port_values['adaptation'] = session.network.adaptation[ports].cpu().numpy().copy()
    available['port_complete'] = True
    return detailed_values, port_values


def port_contribution(port_rate, actual_w, before_w):
    """Fixed-rate arithmetic only; before64x is not an unrun trajectory."""
    rates = np.asarray(port_rate, dtype=np.float64)
    actual = np.asarray(actual_w, dtype=np.float64)
    before = np.asarray(before_w, dtype=np.float64)
    if (rates.ndim != 2 or rates.shape[1] != 20 or actual.shape != (391, 20)
            or before.shape != (391, 20) or not np.isfinite(rates).all()
            or not np.isfinite(actual).all() or not np.isfinite(before).all()):
        raise ValueError('Finite 20-port rates and actual20-to391 weights required')
    return rates @ actual.T, rates @ before.T


def save_raw(out, graph, neutral, session, rows, basic, detailed, dns,
             metadata_sources, signals, ports, port_rows, native, clone_native,
             ledger, spec, port_projection, partial_observations=None):
    """Try each independent prefix artifact before any numerical assertion."""
    errors = []

    def attempt(label, operation):
        try:
            operation()
        except Exception as error:  # noqa: BLE001 - continue independent saves
            errors.append({'artifact': label, 'error': type(error).__name__,
                           'message': str(error), 'traceback': traceback.format_exc()})

    attempt('clock-events.json', lambda: save_json(out/'clock-events.json', ledger.events))
    attempt('clock-summary.json', lambda: save_json(out/'clock-summary.json', {
        'actual_neural_s': ledger.neural_s, 'actual_physical_s': ledger.physical_s,
        'clock_valid': ledger.clock_valid, 'actual_costs_by_owner': ledger.by_owner(),
        'main_completed_controls': max(0, len(rows)-1),
        'main_recorded_controls': max(0, len(rows)-1),
        'main_control_tick_at_save': getattr(session, 'control_tick', None),
        'main_native_samples': len(native), 'clone_native_samples': len(clone_native),
        'incomplete_control_possible': bool(ledger.events and (
            ledger.events[-1]['advance_error'] is not None
            or ledger.events[-1]['clock_read_errors']))}))
    attempt('native-contact.npz', lambda: np.savez_compressed(
        out/'native-contact.npz', **native_arrays(native)))
    attempt('clone-native-contact.npz', lambda: np.savez_compressed(
        out/'clone-native-contact.npz', **native_arrays(clone_native)))
    if partial_observations:
        def partial_payload():
            raw = {}
            receipt = []
            for index, item in enumerate(partial_observations):
                available = item['available']
                fields = {}
                for group in ('detail', 'port'):
                    for field, value in available.get(group, {}).items():
                        key = f'partial{index}_{group}_{field}'
                        raw[key] = np.asarray(value).copy()
                        fields[key] = True
                for group in ('current_detailed_tensor', 'current_port_tensor'):
                    if group in available:
                        try:
                            key = f'partial{index}_{group}'
                            raw[key] = available[group].cpu().numpy().copy()
                            fields[key] = True
                        except Exception as error:  # noqa: BLE001 - retain returned fields
                            fields[group+'_transfer_error'] = type(error).__name__+': '+str(error)
                receipt.append({'time_s': item['time_s'],
                    'control_tick': item['control_tick'],
                    'native_control_complete': item['native_control_complete'],
                    'detail_complete': bool(available.get('detail_complete')),
                    'port_complete': bool(available.get('port_complete')),
                    'available_fields': fields})
            return raw, receipt

        partial_data = []
        attempt('partial-observation-materialization',
                lambda: partial_data.extend(partial_payload()))
        if len(partial_data) == 2:
            attempt('partial-observation.npz', lambda: np.savez_compressed(
                out/'partial-observation.npz', **partial_data[0]))
            attempt('partial-observation.json', lambda: save_json(
                out/'partial-observation.json', partial_data[1]))
    if session is not None:
        attempt('trace.json', lambda: save_json(out/'trace.json', {
            'schema': 'flylab.m3-e121-port-interference-native-trace.v1',
            'controller': 'Only 872 registered target-port-to-MN values divided by64; '
                          'full BANC native sensory/body feedback',
            'graph_hash': graph.hash, 'model_hash': session.network.identity,
            'body_hash': session.body.model_hash,
            'adapter_hash': session.adapter.identity,
            'parameters': {**neutral['parameters'], 'diagnostic': spec},
            'samples': rows}))
    if basic is not None and detailed is not None and dns is not None:
        def details():
            count = len(signals)
            times = np.asarray([row['time_s'] for row in rows[1:count+1]])
            values = {key: np.asarray([row[key] for row in signals]) for key in
                      ('rate', 'recurrent_current', 'drive', 'adaptation',
                       'muscle_activation')}
            return count, times, values

        def basic_signals():
            n, times, values = details()
            if n:
                columns = np.asarray([int(np.flatnonzero(detailed == cell)[0])
                                      for cell in basic])
                np.savez_compressed(out/'signals.npz', time_s=times, observed_ids=basic,
                    rate=values['rate'][:, columns], DN_ids=dns,
                    DN_drive=values['drive'][:, columns[-2:]],
                    DN_recurrent_current=values['recurrent_current'][:, columns[-2:]],
                    DN_adaptation=values['adaptation'][:, columns[-2:]])
            else:
                np.savez_compressed(out/'signals.npz', time_s=np.empty(0),
                    observed_ids=basic, rate=np.empty((0, 442), np.float32), DN_ids=dns,
                    DN_drive=np.empty((0, 2), np.float32),
                    DN_recurrent_current=np.empty((0, 2), np.float32),
                    DN_adaptation=np.empty((0, 2), np.float32))

        def detailed_signals():
            n, times, values = details()
            if n:
                np.savez_compressed(out/'recruitment-signals.npz', time_s=times,
                    observed_ids=detailed, **values)
            else:
                np.savez_compressed(out/'recruitment-signals.npz', time_s=np.empty(0),
                    observed_ids=detailed, rate=np.empty((0, 446), np.float32),
                    recurrent_current=np.empty((0, 446), np.float32),
                    drive=np.empty((0, 446), np.float32),
                    adaptation=np.empty((0, 446), np.float32),
                    muscle_activation=np.empty((0, 96)))

        attempt('signals.npz', basic_signals)
        attempt('recruitment-signals.npz', detailed_signals)

    if ports is not None:
        def port_signals():
            n = len(port_rows)
            nav = [row.get('navigation') for row in rows[:n]]
            values = {key: np.asarray([row[key] for row in port_rows],
                                      dtype=np.float32).reshape(n, 20)
                      for key in ('rate', 'recurrent_current', 'drive', 'adaptation')}
            np.savez_compressed(out/'target-port-signals.npz',
                time_s=np.asarray([row['time_s'] for row in rows[:n]]),
                port_ids=ports, motor_ids=port_projection['motor_ids'], **values,
                target_cue_cut=np.asarray([item['cue_cut'] if item else False
                                           for item in nav], dtype=bool),
                target_turn_request=np.asarray([item['turn_request'] if item else 0
                                                for item in nav], dtype=np.int8),
                target_brake_request=np.asarray([item['brake_request'] if item else False
                                                 for item in nav], dtype=bool),
                navigation_available=np.asarray([item is not None for item in nav], bool))

        def contribution():
            rates = np.asarray([row['rate'] for row in port_rows],
                               dtype=np.float64).reshape(len(port_rows), 20)
            actual, fixed_before = port_contribution(
                rates, port_projection['actual_W'], port_projection['before_64x_W'])
            np.savez_compressed(out/'port-MN-contribution.npz',
                time_s=np.asarray([row['time_s'] for row in rows[:len(port_rows)]]),
                port_ids=ports, motor_ids=port_projection['motor_ids'],
                actual_current_by_MN=actual,
                before64x_same_rate_arithmetic_by_MN=fixed_before,
                actual_W=port_projection['actual_W'],
                before_64x_W=port_projection['before_64x_W'])

        def identities():
            save_json(out/'target-port-identity.json', {
                'port_ids': ports.tolist(),
                'nodes': [{key: graph.nodes[int(i)].get(key) for key in
                           ('id', 'root_id', 'cell_type', 'nt_type', 'soma_side',
                            'output_side')} for i in ports],
                'interpretation': 'Existing declared encoder ports; before64x at '
                                  'observed rates is arithmetic, not an unrun response'})

        attempt('target-port-signals.npz', port_signals)
        attempt('port-MN-contribution.npz', contribution)
        attempt('target-port-identity.json', identities)
    return errors


def resource_screen(preparation, body_wall, process_wall, ledger):
    return bool(preparation['resource_limit_met'] and math.isfinite(body_wall)
        and math.isfinite(process_wall) and body_wall <= BODY_LIMIT_S
        and process_wall <= PROCESS_LIMIT_S
        and PRIOR_WALL_S+process_wall <= SHARED_LIMIT_S
        and ledger.clock_valid and ledger.neural_s is not None
        and ledger.physical_s is not None
        and ledger.neural_s <= 10.1000001 and ledger.physical_s <= 10.1000001)


def close_owned(primary, *operations):
    """Close all acquired objects, preserving the primary failure identity."""
    failures = []
    for label, operation in operations:
        if operation is None:
            continue
        try:
            operation()
        except Exception as error:  # noqa: BLE001 - remaining closes must run
            failures.append((label, error))
    if failures and primary is not None:
        for label, error in failures:
            primary.add_note(f'E121 {label} cleanup: {type(error).__name__}: {error}')
    elif failures:
        for label, error in failures[1:]:
            failures[0][1].add_note(
                f'E121 {label} cleanup: {type(error).__name__}: {error}')
        raise failures[0][1]
    return failures


def control_start_allowed(body_began, process_began):
    """Both deadlines are checked immediately before each new control."""
    now = time.perf_counter()
    return bool(now-body_began <= BODY_LIMIT_S-RESERVE_S
                and now-process_began <= PROCESS_LIMIT_S-RESERVE_S)


def run_body(root, out, graph, cal, obs, startup, patch, preparation, spec,
             process_began):
    body_began = time.perf_counter()
    neutral = StateStore.load(out/'candidate-neutral')
    projection = load_npz(out/'port-projection.npz')
    ledger = AdvanceLedger()
    session = clone = restore_main = restore_clone = None
    basic = detailed = metadata_sources = dns = None
    rows, signal_rows, port_rows, native, clone_native = [], [], [], [], []
    partial_observations = []
    reason = 'duration'
    runtime_error = None
    raw_errors = []
    future_ready = False
    try:
        session = IntrinsicRecruitmentTrial.from_checkpoint(graph, neutral)
        if session.network.device != 'cuda':
            raise ValueError('Original full BANC CUDA native body required')
        exact(neutral, session.snapshot(), 'E121 body original disk neutral')
        basic, detailed, metadata_sources, dns = observation_ids(graph, cal, obs)
        ports = projection['port_ids']
        if session.navigation is None or not np.array_equal(
                session.navigation.encoder.ports, ports):
            raise ValueError('Original target encoder20 identity required')
        weights = source_matrix(session.network)
        exact(weights[cal['motor_ids']][:, ports].toarray(), projection['actual_W'],
              'Runtime actual port-to-MN W')
        np.savez_compressed(out/'recruitment-metadata.npz', observed_ids=detailed,
            motor_ids=cal['motor_ids'], source_ids=metadata_sources,
            source_to_motor=weights[cal['motor_ids']][:, metadata_sources].toarray(),
            **{key: getattr(session.network, key).cpu().numpy()[detailed].copy()
               for key in ('tau', 'a', 'threshold', 'cap', '_cell_gain')})
        restore_main = install_native_observer(session.body, native)
        StateStore.save(out/'start', session.snapshot())
        rows.append(capture_control(session))
        t0_available = {}
        try:
            observed_signals(session, detailed, ports, t0_available)
        finally:
            if t0_available.get('port_complete'):
                port_rows.append(t0_available['port'])
            else:
                partial_observations.append({'time_s': rows[0]['time_s'],
                    'control_tick': 0, 'native_control_complete': False,
                    'available': t0_available})
        # Mandatory durable receipt before the first advance. Failure means 0 calls.
        save_json(out/'advance-start.json', {'experiment': EXPERIMENT,
            'body_wall_before_first_advance_s': time.perf_counter()-body_began,
            'main_requested_controls': 2000, 'clone_requested_controls': 20,
            'actual_neural_s': 0.0, 'actual_physical_s': 0.0})
        with (out/'trace.jsonl').open('w', encoding='utf-8') as stream:
            stream.write(json.dumps(jsonable(rows[0]), allow_nan=False)+'\n')
            for tick in range(1, 2001):
                if not control_start_allowed(body_began, process_began):
                    reason = 'resource_guard'
                    break
                previous_native = len(native)
                ledger.advance(session, 1, 'main')
                row = capture_control(session)
                if (len(native)-previous_native != 50
                        or session.network.tick != tick*20
                        or abs(session.body.physics_time()-tick*.005) > 1e-8):
                    raise ValueError('Complete5ms control requires20neural/50physics steps')
                rows.append(row)
                stream.write(json.dumps(jsonable(row), allow_nan=False)+'\n')
                available = {}
                try:
                    observed_signals(session, detailed, ports, available)
                finally:
                    if available.get('detail_complete'):
                        signal_rows.append(available['detail'])
                    if available.get('port_complete'):
                        port_rows.append(available['port'])
                    if not (available.get('detail_complete')
                            and available.get('port_complete')):
                        partial_observations.append({'time_s': row['time_s'],
                            'control_tick': tick, 'native_control_complete': True,
                            'available': available})
                if session.fault or session.body.fault:
                    reason = 'native_health_guard'
                    break
                if tick == 200:
                    StateStore.save(out/'checkpoint-1s', session.snapshot())
                    clone = IntrinsicRecruitmentTrial.from_checkpoint(
                        graph, StateStore.load(out/'checkpoint-1s'))
                    try:
                        if clone.network.device != 'cuda':
                            raise ValueError('Clone must keep original CUDA device')
                        restore_clone = install_native_observer(clone.body, clone_native)
                        for _ in range(20):
                            if not control_start_allowed(body_began, process_began):
                                reason = 'resource_guard'
                                break
                            ledger.advance(clone, 1, 'clone')
                        if reason != 'resource_guard':
                            if (clone.network.tick != 4400 or len(clone_native) != 1000
                                    or abs(clone.body.physics_time()-1.1) > 1e-8):
                                raise ValueError('Exact .1s clone clocks required')
                            StateStore.save(out/'clone-1p1', clone.snapshot())
                    finally:
                        primary = sys.exc_info()[1]
                        try:
                            close_owned(primary, ('clone observer', restore_clone),
                                        ('clone', clone.close if clone else None))
                        finally:
                            restore_clone = clone = None
                    if reason == 'resource_guard':
                        break
                if tick == 220:
                    StateStore.save(out/'main-1p1', session.snapshot())
                    future_ready = True
                if tick % 200 == 0:
                    print(json.dumps({'E121_main_time_s': tick*.005,
                        'body_wall_s': time.perf_counter()-body_began,
                        'actual_neural_s': ledger.neural_s,
                        'actual_physical_s': ledger.physical_s}), flush=True)
        if reason == 'duration' and not session.fault and not session.body.fault:
            StateStore.save(out/'final', session.snapshot())
    except NativeHealthStop as error:
        reason, runtime_error = 'native_health_guard', error
    except Exception as error:  # noqa: BLE001 - preserve primary and measured prefixes
        reason, runtime_error = 'execution_error', error
    finally:
        try:
            raw_errors.extend(save_raw(out, graph, neutral, session, rows, basic,
                detailed, dns, metadata_sources, signal_rows,
                projection['port_ids'] if basic is not None else None,
                port_rows, native, clone_native, ledger, spec, projection,
                partial_observations))
        except Exception as error:  # noqa: BLE001 - still close everything
            raw_errors.append({'artifact': 'save_raw', 'error': type(error).__name__,
                               'message': str(error), 'traceback': traceback.format_exc()})
        cleanup = []
        for label, operation in (('clone observer', restore_clone),
                                 ('clone', clone.close if clone else None),
                                 ('main observer', restore_main),
                                 ('main', session.close if session else None)):
            if operation is None:
                continue
            try:
                operation()
            except Exception as error:  # noqa: BLE001 - every owned object is attempted
                cleanup.append({'owner': label, 'error': type(error).__name__,
                                'message': str(error)})
                if runtime_error is None:
                    runtime_error = error
                    reason = 'execution_error'
                else:
                    runtime_error.add_note(f'E121 {label} cleanup: {error}')
        if cleanup:
            try:
                save_json(out/'cleanup-errors.json', cleanup)
            except Exception as error:  # noqa: BLE001 - keep primary
                runtime_error.add_note('Cleanup receipt failed: '+str(error))
        if runtime_error is not None:
            try:
                save_json(out/'runtime-error.json', {'stop_reason': reason,
                    'error': type(runtime_error).__name__, 'message': str(runtime_error),
                    'actual_neural_s': ledger.neural_s,
                    'actual_physical_s': ledger.physical_s,
                    'last_clock_event': ledger.events[-1] if ledger.events else None})
            except Exception as error:  # noqa: BLE001 - keep primary
                runtime_error.add_note('Runtime receipt failed: '+str(error))
        if raw_errors:
            try:
                save_json(out/'raw-persistence-errors.json', raw_errors)
            except Exception as error:  # noqa: BLE001 - keep primary if present
                if runtime_error is not None:
                    runtime_error.add_note('Raw-error receipt failed: '+str(error))
                else:
                    runtime_error, reason = error, 'execution_error'

    # Prefixes, native contacts, port currents and clocks are persisted above.
    analysis_errors = []
    secondary_exceptions = []

    def attempt_analysis(stage, operation, default=None):
        try:
            return operation()
        except Exception as error:  # noqa: BLE001 - preserve primary; continue others
            analysis_errors.append({'stage': stage, 'error': type(error).__name__,
                'message': str(error), 'traceback': traceback.format_exc()})
            secondary_exceptions.append(error)
            if runtime_error is not None:
                runtime_error.add_note(
                    f'E121 secondary {stage}: {type(error).__name__}: {error}')
            return default

    future_exact = False
    if future_ready and not raw_errors:
        def verify_future():
            exact(StateStore.load(out/'clone-1p1'), StateStore.load(out/'main-1p1'),
                  'Whole1s checkpoint future at1.1s')
            return True

        future_exact = attempt_analysis('checkpoint_future', verify_future, False)
    quality = None
    if len(rows) == 2001 and not raw_errors:
        analyzed = attempt_analysis('original_gait_evaluator',
            lambda: analyze(load_model(out/'trace.json')))
        if analyzed is not None:
            quality = analyzed[0]
            attempt_analysis('quality-analysis.json',
                             lambda: save_json(out/'quality-analysis.json', quality))
    fault = bool(runtime_error or raw_errors or (session is not None and
        (session.fault or session.body.fault)))
    missing_quality = {'complete_10s_native_health': False,
        'development_screens': {key: False for key in
                                ('F3_numeric', 'F4_numeric', 'F5_numeric')}}
    summary = attempt_analysis('body_quality',
        lambda: body_quality(rows, native, quality, reason=reason, fault=fault,
            clock_valid=ledger.clock_valid, future_exact=future_exact), missing_quality)
    attempt_analysis('quality-summary.json', lambda: save_json(
        out/'quality-summary.json', summary))
    attempt_analysis('actual-startup-original99-raw.npz',
                     lambda: save_startup_raw(out, rows, startup))
    missing_metric = {'RMS': None, 'max_abs': None, 'available': False}
    missing_startup = {'complete_original99_prefix': False,
        'observed_controls': max(0, min(len(rows)-1, 99)),
        'original_E100_E103_q': missing_metric.copy(),
        'original_E100_E103_LTM': missing_metric.copy(),
        'original_E100_E103_startup_screen': False}
    startup_summary = attempt_analysis('original_startup',
        lambda: original_startup(rows, native, startup), missing_startup)
    attempt_analysis('startup-diagnostic.json', lambda: save_json(
        out/'startup-diagnostic.json', startup_summary))
    missing_slip = {'load_weighted_mm_s': None, 'support_samples': None,
        'support_load_sum_bw': None, 'available': False}
    slip = attempt_analysis('supported_slip',
        lambda: supported_slip(native_arrays(native),
            complete=summary['complete_10s_native_health']), missing_slip)
    attempt_analysis('supported-slip.json', lambda: save_json(out/'supported-slip.json', slip))

    def read_baseline():
        prior = root/'e113-dense-foot-native-body'
        return {key: json.loads((prior/name).read_text(encoding='utf-8'))
            for key, name in (('quality', 'quality-summary.json'),
                              ('startup', 'startup-diagnostic.json'),
                              ('supported_slip', 'supported-slip.json'))}

    baseline = attempt_analysis('E113_baseline_read', read_baseline,
        {'quality': {'development_screens': {}}, 'startup': {}, 'supported_slip': {}})
    attempt_analysis('E113-E121-comparison.json', lambda: save_json(
        out/'E113-E121-comparison.json', {'baseline_E113': baseline,
        'intervention_E121': {'quality': summary, 'startup': startup_summary,
                              'supported_slip': slip}, 'same_window_s': [.5, 10.0],
        'E113_port_trajectory_available': False}))
    source_summary = {'source_screen': False}
    lf_summary = {'source_preparation_screen': False}
    if not raw_errors and (out/'recruitment-signals.npz').exists():
        source_summary, lf_summary = attempt_analysis('source_diagnostics',
            lambda: source_diagnostics(root, out, cal, neutral,
                summary['complete_10s_native_health'], reason),
            (source_summary, lf_summary))
    if analysis_errors:
        attempt_analysis('analysis-errors.json',
                         lambda: save_json(out/'analysis-errors.json', analysis_errors))
    if runtime_error is not None:
        attempt_analysis('runtime-error-final.json', lambda: save_json(
            out/'runtime-error.json', {'stop_reason': reason,
                'error': type(runtime_error).__name__, 'message': str(runtime_error),
                'notes': list(getattr(runtime_error, '__notes__', [])),
                'actual_neural_s': ledger.neural_s,
                'actual_physical_s': ledger.physical_s,
                'last_clock_event': ledger.events[-1] if ledger.events else None,
                'analysis_errors': analysis_errors}))
    body_wall = time.perf_counter()-body_began
    process_wall = time.perf_counter()-process_began
    resources = attempt_analysis('resource_screen',
        lambda: resource_screen(preparation, body_wall, process_wall, ledger), False)
    validity = ('valid_full_BANC_port_interference_diagnostic'
        if reason == 'duration' and summary['complete_10s_native_health']
        and future_exact and resources and not raw_errors and not analysis_errors
        else 'resource_limit_incomplete' if not resources else 'incomplete')
    screens = summary['development_screens']
    next_candidate = bool(validity.startswith('valid_') and all(screens.values())
        and startup_summary['original_E100_E103_startup_screen'])
    result = {'experiment': EXPERIMENT, 'validity': validity,
        'intervention': 'Existing872 declared20 GABA port-to-MN values /64 only',
        'baseline_E113_F3_F5': baseline['quality']['development_screens'],
        'candidate_adopted': False, 'supports_next_final_candidate_validation':
            next_candidate, 'F1_F8_goal': 'active_unmet', 'F6_exposed': False,
        'fit_calls': 0, 'LP_calls': 0, 'conditional_response_calls': 0,
        'stop_reason': reason, 'main_completed_controls': max(0, len(rows)-1),
        'main_control_tick_after': getattr(session, 'control_tick', None),
        'main_native_samples': len(native), 'clone_native_samples': len(clone_native),
        'actual_neural_s': ledger.neural_s, 'actual_physical_s': ledger.physical_s,
        'actual_costs_by_owner': ledger.by_owner(), 'clock_valid': ledger.clock_valid,
        'whole_disk_future_exact': future_exact, 'device': neutral['network']['device'],
        'quality_summary': summary, 'startup_diagnostic': startup_summary,
        'source_phase_summary': source_summary, 'actual_LF_source_summary': lf_summary,
        'supported_slip': slip, 'resource_limit_met': resources,
        'preparation_wall_s': preparation['wall_s'], 'body_wall_s': body_wall,
        'process_wall_s': process_wall, 'prior_conservative_wall_s': PRIOR_WALL_S,
        'conservative_shared_process_wall_s': PRIOR_WALL_S+process_wall,
        'raw_persistence_errors': len(raw_errors), 'analysis_errors': len(analysis_errors)}
    attempt_analysis('result.json', lambda: save_json(out/'result.json', result))
    # A slow final receipt is part of the charged process, not free post-run work.
    final_body_wall = time.perf_counter()-body_began
    final_process_wall = time.perf_counter()-process_began
    if not attempt_analysis('final_resource_screen',
            lambda: resource_screen(preparation, final_body_wall,
                                    final_process_wall, ledger), False):
        result.update(validity='resource_limit_incomplete',
            supports_next_final_candidate_validation=False, resource_limit_met=False,
            body_wall_s=final_body_wall, process_wall_s=final_process_wall,
            conservative_shared_process_wall_s=PRIOR_WALL_S+final_process_wall)
        attempt_analysis('final_result.json', lambda: save_json(out/'result.json', result))
    if runtime_error is not None and (reason == 'execution_error'
                                      or secondary_exceptions):
        raise runtime_error
    if raw_errors:
        raise RuntimeError('E121 raw persistence failed; see raw-persistence-errors.json')
    if secondary_exceptions:
        raise secondary_exceptions[0]
    return result


def main(root, graph_path, out):
    began = time.perf_counter()
    if out.exists():
        raise FileExistsError('Preserve single E121 output: '+str(out))
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': EXPERIMENT, 'question': 'Does removing only the existing '
        '64x target-port-to-MN amplification change actual native F3-F5?',
        'intervention': 'E113 existing872 calibrated values /64; unchanged original '
                        'whole-zero, tau, anatomy, all other W, sensory/body and cut',
        'controls': 2000, 'clone_controls': 20, 'neural_steps_per_control': 20,
        'physics_steps_per_control': 50, 'neural_dt_s': .00025,
        'physical_dt_s': .0001, 'F6_exposed': False, 'candidate_adopted': False,
        'limits': {'prepare_wall_s': PREPARE_LIMIT_S, 'body_wall_s': BODY_LIMIT_S,
                   'process_wall_s': PROCESS_LIMIT_S, 'prior_conservative_wall_s':
                   PRIOR_WALL_S, 'shared_wall_s': SHARED_LIMIT_S,
                   'reserve_s': RESERVE_S, 'actual_neural_s': 10.1,
                   'actual_physical_s': 10.1},
        'tool_sha256': sha256(__file__), 'E121_design_sha256': sha256(
            ROOT/'docs/BANC_M3_E121_TARGET_PORT_INTERFERENCE_DESIGN_20260920.md'),
        'E121_frozen_design_sha256': sha256(root/'e121-frozen-design.json'),
        'E113_calibration_sha256': sha256(root/'e113-dense-foot-native-body/calibration.npz'),
        'E113_parent_review_sha256': sha256(
            root/'e113-dense-foot-native-body/parent-final-review.json'),
        'diagnostic_input_sha256': DIAGNOSTIC_INPUT_SHA,
        'original_evaluator_sha256': sha256(
            ROOT/'tools/analyze_c_banc_biological_gait.py')}
    save_json(out/'spec.json', spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    try:
        graph, cal, obs, startup, patch, preparation = prepare(
            root, graph_path, out, began)
        if not preparation['resource_limit_met']:
            raise TimeoutError('E121 preparation exceeded45 wall seconds')
        return run_body(root, out, graph, cal, obs, startup, patch,
                        preparation, spec, began)
    except Exception as error:
        try:
            clock_path = out/'clock-summary.json'
            clock = json.loads(clock_path.read_text(encoding='utf-8')) if (
                clock_path.exists()) else {}
            save_json(out/'execution-error.json', {'experiment': EXPERIMENT,
                'validity': 'incomplete', 'error': type(error).__name__,
                'message': str(error), 'process_wall_s': time.perf_counter()-began,
                'notes': list(getattr(error, '__notes__', [])),
                'actual_neural_s': clock.get('actual_neural_s'),
                'actual_physical_s': clock.get('actual_physical_s'),
                'clock_valid': clock.get('clock_valid'), 'fit_calls': 0})
        except Exception as receipt_error:  # noqa: BLE001 - retain original
            error.add_note('E121 execution receipt failed: '+str(receipt_error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path,
        default=ROOT/'verification/banc-natural-gait-m3-20260919')
    parser.add_argument('--graph', type=Path,
        default=ROOT/'data/acquisitions/banc888-windows-20260910/bundle')
    parser.add_argument('--output', type=Path,
        default=ROOT/'verification/banc-natural-gait-m3-20260919/e121-port-interference')
    args = parser.parse_args()
    print(json.dumps(jsonable(main(args.root, args.graph, args.output))), flush=True)
