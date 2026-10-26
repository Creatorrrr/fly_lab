"""E114 recorded-context cross-response validation; no native simulation or fit.

This executable is reserved for one parent-approved CUDA batch.  Its preflight
and small synthetic unit tests do not construct a network or a body.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.banc_m3_context_response import (
    ContextResponse,
    audit_own_background,
    id_positions,
    validate_time_grid,
)
from tools.banc_m3_torch_foot_geometry import TorchFootGeometry

DESIGN_SHA = '5a9dbae12b70ad3a32ed965c697c96d970a034cef870efca8a8368c6664b0772'
FROZEN_SHA = 'be44fa8adca1947afb3e34113b6e4fabea4fe93cd1bd6276ebff6537f0a3d3c9'
INPUT_SHA = '8244c676ab907442168fb57a8b319fbd22e401f3ade9c286940eb48660f95f58'
INPUT_JSON_SHA = '0396655defb98e93a2520b496564236f9619f1fab962c4489ce07ca5b9097d64'
REVIEW_SHA = '77a8052a6165297105b1e85c81b5a2012f8b25c76ff91bdf6b9b713b6554bc6b'
GEOMETRY_HASH = '36a47d2f0f0d4e83d5ff8ecf1af95b9ce63269d5158d98642b4f89e0bc230953'
PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))
ORIGINAL = (.02, .05, .01, .03)
INNER = (.018, .04, .009, .025)
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
LIMIT_S = 50.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(safe_json(value), indent=2, ensure_ascii=False,
                               allow_nan=False), encoding='utf-8')


def check_calibration_weight_map(cal: dict, fixture: dict, context: int) -> None:
    """Check actual signed float32 W using IDs, never sorted-edge lookup."""
    rows, cols = fixture['edge_rows'], fixture['edge_cols']
    target_ids = fixture['edge_ids']
    positions = id_positions(cal['edge_ids'], target_ids)
    require(np.array_equal(cal['individual_edge_ids'], target_ids),
            'Selected edge ID order changed')
    require(np.array_equal(cal['individual_edge_motor_ids'], fixture['motor_ids'][rows])
            and np.array_equal(cal['individual_edge_source_ids'],
                               fixture['source_ids'][cols]),
            'Selected edge receiving/source ID mapping changed')
    require(np.array_equal(cal['individual_source_basis_ids'], fixture['source_ids']),
            'Calibration source basis order changed')
    require(np.array_equal(cal['source_values'][positions],
                           fixture['original_signed_edge_weights']),
            'Original signed float32 edge values changed')
    expected = (fixture['original_signed_edge_weights'] *
                cal['individual_edge_efficacies']).astype(np.float32)
    require(np.array_equal(cal['values'][positions], expected),
            'Calibration efficacy is not the actual float32 edge value')
    candidate = fixture['weights'][context]
    require(np.array_equal(candidate, candidate.astype(np.float32).astype(np.float64)),
            'Candidate W is not actual float32 weights')
    require(np.array_equal(candidate[rows, cols],
                           cal['values'][positions].astype(np.float64)),
            'Candidate W selected edge differs from calibration')


def reviewed_inputs(root: Path) -> tuple[dict, dict]:
    """Model-free, hash-exact review of the immutable two-context fixture."""
    root = Path(root)
    fixed = ROOT/'docs/BANC_M3_E114_CONTEXT_VALIDATION_DESIGN_20260920.md'
    require(sha256(fixed) == DESIGN_SHA, 'Frozen design changed')
    require(sha256(root/'e114-frozen-design.json') == FROZEN_SHA,
            'Frozen execution specification changed')
    require(sha256(root/'parent-e114-context-inputs.npz') == INPUT_SHA,
            'Parent input fixture changed')
    require(sha256(root/'parent-e114-context-inputs.json') == INPUT_JSON_SHA,
            'Parent input manifest changed')
    require(sha256(root/'parent-e114-preparation-review.json') == REVIEW_SHA,
            'Fixture acceptance review changed')
    frozen = json.loads((root/'e114-frozen-design.json').read_text(encoding='utf-8'))
    manifest = json.loads((root/'parent-e114-context-inputs.json').read_text(
        encoding='utf-8'))
    review = json.loads((root/'parent-e114-preparation-review.json').read_text(
        encoding='utf-8'))
    require(review['accepted_as_valid_recorded_input_fixture'] is True and
            review['checks']['original_only_failed_check_is_nonidentical_reconstruction']
            is True, 'Original fixture review is not the accepted addendum')
    require(frozen['actual_execution_authorized'] is False and
            frozen['pair_context_indices'] == [0, 0, 1, 1] and
            frozen['pair_weight_indices'] == [0, 1, 0, 1],
            'Frozen pair order or authorization changed')
    for relative, expected in {**manifest['input_sha256'],
                               **frozen['inputs_sha256']}.items():
        require(sha256(root/relative) == expected, f'Input hash changed: {relative}')
    fixture = load_npz(root/'parent-e114-context-inputs.npz')
    f = fixture
    require(set(map(str, f['context_names'])) == {'E110', 'E113'} and
            f['context_names'].tolist() == ['E110', 'E113'],
            'Context order changed')
    require(f['source_rate'].shape == (2, 2001, 49) and
            f['current'].shape == f['background'].shape == (2, 2001, 391) and
            f['weights'].shape == (2, 391, 49), 'Context tensor axes changed')
    validate_time_grid(f['time_s'])
    require(np.array_equal(f['time_s'], np.arange(2001, dtype=np.float64)*.005),
            'Original 5ms clock changed')
    require(np.array_equal(f['pair_context_indices'], [0, 0, 1, 1]) and
            np.array_equal(f['pair_weight_indices'], [0, 1, 0, 1]),
            'Fixture pair order changed')
    require(all(np.isfinite(f[key]).all() for key in ('source_rate', 'current',
            'background', 'weights', 'tau', 'a', 'threshold', 'cap', 'pool',
            'moment', 'rest', 'actual_command',
            'actual_tendon', 'fixed_target')),
            'Nonfinite fixture field')
    require(np.all(f['source_rate'][:, 0] == 0) and
            np.all(f['current'][:, 0] == 0),
            'Original whole-zero initial activity changed')
    require(np.array_equal(f['startup_mask'].nonzero()[0], np.arange(1, 100)) and
            np.array_equal(f['fixed_mask'].nonzero()[0], np.arange(100, 1928))
            and np.array_equal(f['own_mask'][0], f['fixed_mask']) and
            np.array_equal(f['own_mask'][1].nonzero()[0], np.arange(100, 1923))
            and np.array_equal(f['common_mask'], f['own_mask'][0] & f['own_mask'][1]),
            'Original99/fixed/own/common masks changed')
    require(np.array_equal(f['own_target'][0, f['fixed_mask']], f['fixed_target']),
            'E110 own-phase target differs from original fixed target')
    require(all(np.isfinite(f['own_target'][i, f['own_mask'][i]]).all()
                and np.isfinite(f['phase'][i, f['own_mask'][i]]).all()
                for i in range(2)),
            'Own-phase target/phase nonfinite inside bracketed support')
    require(f['startup_target_command'].shape == (99, 42) and
            f['startup_target_tendon'].shape == (99, 6) and
            f['actual_command'].shape == (2, 2001, 42) and
            f['actual_tendon'].shape == (2, 2001, 6) and
            f['phase'].shape == (2, 2001, 6),
            'Reference shapes changed')
    require(len(set(map(int, f['source_ids']))) == 49 and
            len(set(map(int, f['motor_ids']))) == 391 and
            len(set(map(int, f['edge_ids']))) == 1572 and
            np.all((0 <= f['edge_rows']) & (f['edge_rows'] < 391)) and
            np.all((0 <= f['edge_cols']) & (f['edge_cols'] < 49)),
            'Source/MN/edge IDs invalid')
    train = load_npz(root/'parent-e112-training-inputs.npz')
    for key in ('source_ids', 'motor_ids', 'tau', 'a', 'threshold', 'cap',
                'pool', 'moment', 'rest', 'ltm', 'edge_ids', 'edge_rows',
                'edge_cols', 'original_signed_edge_weights', 'joint_scale',
                'foot_scale', 'velocity_scale', 'source_e1_ids', 'offsets'):
        require(np.array_equal(f[key], train[key]), f'Original E112 {key} changed')
    require(np.array_equal(f['weights'][0], train['current_w']) and
            np.array_equal(f['source_rate'][0], train['source_rate']) and
            np.array_equal(f['current'][0], train['current']),
            'E110 input differs from reviewed training fixture')
    prediction = load_npz(root/'e112-dense-foot-recruitment-v2/prediction.npz')
    require(np.array_equal(f['weights'][1],
                           prediction['actual_float32_source_weights']),
            'E113 actual W differs from reviewed E112b candidate')
    for i, label in enumerate(('e110-e109-native-body-diagnostic',
                               'e113-dense-foot-native-body')):
        cal = load_npz(root/label/'calibration.npz')
        meta = load_npz(root/label/'recruitment-metadata.npz')
        check_calibration_weight_map(cal, f, i)
        require(meta['source_ids'].shape == (55,) and
                np.array_equal(f['motor_ids'], meta['motor_ids']),
                f'{label} original metadata order changed')
        positions = id_positions(meta['source_ids'], f['source_ids'])
        require(np.array_equal(meta['source_to_motor'][:, positions].astype(np.float64),
                               f['weights'][i]),
                f'{label} source49 and actual W mapping changed')
        trace = json.loads((root/label/'trace.json').read_text(encoding='utf-8'))
        samples = trace['samples']
        require(len(samples) == 2001 and
                np.array_equal(np.asarray([s['joint_targets_rad'] for s in samples],
                                          dtype=np.float64), f['actual_command'][i])
                and np.array_equal(np.asarray([s['time_s'] for s in samples],
                                              dtype=np.float64), f['time_s']),
                f'{label} literal native trace command/clock changed')
    return fixture, {'frozen': frozen, 'manifest': manifest, 'review': review}


def error_stats(predicted: np.ndarray, target: np.ndarray) -> dict:
    if predicted.shape != target.shape or predicted.size == 0:
        raise ValueError('Equal nonempty error arrays required')
    delta = np.asarray(predicted, dtype=np.float64) - target
    if not np.isfinite(delta).all():
        return {'rms': None, 'max': None, 'finite': False}
    return {'rms': float(np.sqrt(np.mean(np.square(delta)))),
            'max': float(np.max(np.abs(delta))), 'finite': True}


def response_metrics(raw: dict, f: dict, feet_fixed: np.ndarray,
                     own_target_feet: list[np.ndarray]) -> dict:
    """Keep original99, fixed-time and each own-phase target separate."""
    selected = np.flatnonzero(f['fixed_mask'])
    require(feet_fixed.shape == (4, len(selected), 6, 3),
            'Fixed proxy feet dimensions changed')
    startup = f['startup_mask']
    result = {'pairs': [], 'foot_semantics':
              'E111 fixed E32 geometry from conditional command; not actual native feet',
              'startup_time_s': [float(f['time_s'][1]), float(f['time_s'][99])],
              'fixed_time_s': [float(f['time_s'][selected[0]]),
                               float(f['time_s'][selected[-1]])],
              'fixed_count': len(selected),
              'own_counts': [int(row.sum()) for row in f['own_mask']],
              'common_count': int(f['common_mask'].sum())}
    for pair, (ci, wi) in enumerate(PAIRS):
        command, tendon = raw['command'][pair], raw['tendon'][pair]
        start_q = error_stats(command[startup], f['startup_target_command'])
        start_ltm = error_stats(tendon[startup], f['startup_target_tendon'])
        start_tuple = (start_q['rms'], start_q['max'], start_ltm['rms'],
                       start_ltm['max'])
        startup_result = {'q': start_q, 'LTM': start_ltm,
                          'original_limits': ORIGINAL, 'inner_limits': INNER,
                          'original_pass': bool(all(v is not None and v <= limit
                                                    for v, limit in zip(
                                                        start_tuple, ORIGINAL))),
                          'inner_pass': bool(all(v is not None and v <= limit
                                                 for v, limit in zip(
                                                     start_tuple, INNER)))}
        fixed_q = error_stats(command[selected], f['fixed_target'])
        foot = feet_fixed[pair]
        fixed_foot = error_stats(foot, f['fixed_target_feet'])
        velocity = np.diff(foot, axis=0)/.005
        fixed_velocity = error_stats(velocity, f['fixed_target_velocity'])
        own_indices = np.flatnonzero(f['own_mask'][ci])
        own_within_fixed = f['own_mask'][ci, selected]
        own_foot = foot[own_within_fixed]
        own_target = own_target_feet[ci]
        require(own_foot.shape == own_target.shape,
                'Own-phase geometry target alignment changed')
        own = {'q': error_stats(command[own_indices],
                                f['own_target'][ci, own_indices]),
               'foot_proxy': error_stats(own_foot, own_target),
               'velocity_proxy': error_stats(np.diff(own_foot, axis=0)/.005,
                                             np.diff(own_target, axis=0)/.005),
               'time_s': [float(f['time_s'][own_indices[0]]),
                          float(f['time_s'][own_indices[-1]])],
               'count': len(own_indices),
               'target': f'{f["context_names"][ci]} own E1 phase + fixed E85 cadence'}
        common = np.flatnonzero(f['common_mask'])
        result['pairs'].append({
            'pair': pair, 'input_context': str(f['context_names'][ci]),
            'candidate_W': str(f['context_names'][wi]),
            'original99': startup_result,
            'fixed_E110_1828': {
                'q': fixed_q, 'foot_proxy': fixed_foot,
                'velocity_proxy': fixed_velocity,
                'lf_row6': error_stats(command[selected, 6], f['fixed_target'][:, 6]),
                'per_leg_q': {leg: error_stats(command[selected, 7*j:7*j+7],
                    f['fixed_target'][:, 7*j:7*j+7])
                    for j, leg in enumerate(LEGS)}},
            'own_phase': own,
            'common_mask_own_target': {
                'q': error_stats(command[common], f['own_target'][ci, common]),
                'count': len(common), 'target_context': str(f['context_names'][ci])},
            'actual_self': None if ci != wi else {
                'q_after_t0': error_stats(command[1:], f['actual_command'][ci, 1:]),
                'LTM_after_t0': error_stats(tendon[1:], f['actual_tendon'][ci, 1:]),
                'q_original99': error_stats(command[startup],
                                            f['actual_command'][ci, startup]),
                'LTM_original99': error_stats(tendon[startup],
                                              f['actual_tendon'][ci, startup])}})
    signed = f['phase'][1] - f['phase'][0]
    circular = (signed + .5) % 1. - .5
    overlap = np.isfinite(f['phase'][0]) & np.isfinite(f['phase'][1])
    result['phase_difference_E113_minus_E110'] = {
        'units': 'cycles', 'absolute_time_aligned': True,
        'no_phase_retiming': True,
        'finite_overlap': {leg: {
            'count': int(overlap[:, j].sum()),
            'first_last_s': [float(f['time_s'][np.flatnonzero(overlap[:, j])[0]]),
                             float(f['time_s'][np.flatnonzero(overlap[:, j])[-1]])],
            'signed_unwrapped': error_stats(signed[overlap[:, j], j],
                np.zeros(int(overlap[:, j].sum()))),
            'circular': error_stats(circular[overlap[:, j], j],
                np.zeros(int(overlap[:, j].sum()))),
            'mean_signed_unwrapped': float(np.mean(signed[overlap[:, j], j]))}
                for j, leg in enumerate(LEGS)},
        'common_mask': {leg: {
            'signed_unwrapped': error_stats(signed[f['common_mask'], j],
                                            np.zeros(int(f['common_mask'].sum()))),
            'circular': error_stats(circular[f['common_mask'], j],
                                    np.zeros(int(f['common_mask'].sum())))}
            for j, leg in enumerate(LEGS)}}
    return result


def independent_baselines(raw: dict, root: Path) -> dict:
    files = ((0, 'parent-e112-latest-conditional-baseline.npz', 'full_current'),
             (1, 'e112-dense-foot-recruitment-v2/prediction.npz', 'current'),
             (3, 'parent-e113-actual-input-response-raw.npz', 'current'))
    checks = {}
    for pair, relative, current_key in files:
        reference = load_npz(root/relative)
        fields = (('current', current_key, 1e-8),
                  ('motor_rate', 'motor_rate', 1e-9),
                  ('muscle_activation', 'muscle_activation', 1e-9),
                  ('command', 'command', 1e-9), ('tendon', 'tendon', 1e-9))
        pair_result = {}
        for ours, theirs, threshold in fields:
            stats = error_stats(raw[ours][pair], reference[theirs])
            pair_result[ours] = {**stats, 'limit': threshold,
                                 'passed': bool(stats['finite'] and
                                                stats['max'] <= threshold)}
        checks[str(pair)] = {'file': relative, 'fields': pair_result,
                             'passed': all(value['passed']
                                           for value in pair_result.values())}
    return {'three_existing_responses': checks,
            'all_passed': all(x['passed'] for x in checks.values()),
            'missing_pair_2': 'E113 input / E110 W; independent parent NumPy once later'}


def detach_numpy(outputs: dict[str, torch.Tensor],
                 into: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Expose each completed array even if a later materialization fails."""
    result = {} if into is None else into
    for key, tensor in outputs.items():
        result[key] = tensor.detach().cpu().numpy().copy()
    return result


def rescue_error_raw(out: Path, raw: dict[str, np.ndarray],
                     output: dict[str, torch.Tensor] | None,
                     ledger: dict, primary: BaseException) -> None:
    """Best-effort independent files; never evaluate the conditional model again."""
    if output is not None:
        for key, tensor in output.items():
            if key in raw:
                continue
            try:
                raw[key] = tensor.detach().cpu().numpy().copy()
            except Exception as secondary:  # noqa: BLE001 - preserve primary error
                ledger['secondary_errors'].append({'at': f'materialize:{key}',
                    'type': type(secondary).__name__, 'message': str(secondary)})
                primary.add_note(f'E114 secondary {key} materialization failure: {secondary}')
    if not raw:
        return
    destination = out/'error-raw'
    try:
        destination.mkdir(exist_ok=True)
    except Exception as secondary:  # noqa: BLE001
        ledger['secondary_errors'].append({'at': 'error-raw directory',
            'type': type(secondary).__name__, 'message': str(secondary)})
        primary.add_note(f'E114 secondary error-raw directory failure: {secondary}')
        return
    ledger['error_raw_saved_fields'] = []
    for key, value in raw.items():
        try:
            np.save(destination/f'{key}.npy', value, allow_pickle=False)
            ledger['error_raw_saved_fields'].append(key)
        except Exception as secondary:  # noqa: BLE001
            ledger['secondary_errors'].append({'at': f'error-raw:{key}',
                'type': type(secondary).__name__, 'message': str(secondary)})
            primary.add_note(f'E114 secondary error-raw {key} failure: {secondary}')


def run_once(root: Path, out: Path) -> dict:
    """One CUDA batch, immutable raw first, then independent numerical gates."""
    start = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    ledger = {'stage': 'created', 'batch_started': 0, 'batch_materialized': 0,
              'batch_raw_saved': 0, 'pairs_started': 0,
              'pairs_materialized': 0, 'pairs_raw_saved': 0,
              'conditional_model_s_started': 0., 'conditional_model_s_materialized': 0.,
              'fit_calls': 0, 'native_network_calls': 0, 'body_calls': 0,
              'actual_neural_s': 0., 'actual_physical_s': 0.,
              'response_call_entered': False, 'batch_reservation_requested': 0,
              'internal_wall_s': None, 'internal_wall_status': 'not_yet_read',
              'secondary_errors': []}

    def refresh_clock(primary_error=None):
        try:
            elapsed = time.monotonic()-start
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Invalid monotonic elapsed wall time')
            ledger['internal_wall_s'] = elapsed
            ledger['internal_wall_status'] = 'available'
        except Exception as secondary:
            ledger['internal_wall_s'] = None
            ledger['internal_wall_status'] = 'unavailable'
            if primary_error is None:
                raise
            ledger['secondary_errors'].append({'at': 'internal clock',
                'type': type(secondary).__name__, 'message': str(secondary)})
            primary_error.add_note(f'E114 secondary internal clock failure: {secondary}')

    def persist():
        refresh_clock()
        # Required receipt: no conditional evaluation after a failed write.
        save_json(out/'execution-ledger.json', ledger)

    def failure_save(filename, value, primary_error):
        try:
            save_json(out/filename, value)
        except Exception as secondary:  # noqa: BLE001 - retain prior failure
            ledger['secondary_errors'].append({'at': filename,
                'type': type(secondary).__name__, 'message': str(secondary)})
            primary_error.add_note(f'E114 secondary {filename} save failure: {secondary}')

    def deadline():
        if time.monotonic()-start > LIMIT_S:
            raise TimeoutError('E114 internal 50s wall limit exceeded')

    primary = None
    output = None
    raw = {}
    try:
        for original, name in ((ROOT/'tools/banc_m3_context_response.py',
                                'banc_m3_context_response.py'),
                               (ROOT/'tools/verify_c_banc_m3_e114_contexts.py',
                                'verify_c_banc_m3_e114_contexts.py'),
                               (ROOT/'tests/test_c_banc_context_response.py',
                                'test_c_banc_context_response.py'),
                               (ROOT/'docs/BANC_M3_E114_CONTEXT_VALIDATION_DESIGN_20260920.md',
                                'frozen-design.md'),
                               (root/'e114-frozen-design.json', 'e114-frozen-design.json'),
                               (root/'parent-e114-context-inputs.json',
                                'parent-e114-context-inputs.json'),
                               (root/'parent-e114-preparation-review.json',
                                'parent-e114-preparation-review.json')):
            shutil.copy2(original, out/name)
        ledger['stage'] = 'preflight'
        persist()
        f, provenance = reviewed_inputs(root)
        deadline()
        require(torch.cuda.is_available(), 'CUDA unavailable')
        require(torch.cuda.device_count() > 0 and
                'RTX 4080' in torch.cuda.get_device_name(0),
                'Expected RTX 4080 cuda:0 device unavailable')
        import os
        require(all(os.environ.get(key) == '1' for key in
                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
                'Single-thread CPU environment required')
        torch.set_num_threads(1)
        require(torch.get_num_threads() == 1, 'Torch CPU thread count changed')
        ledger['environment'] = {'cuda_device': torch.cuda.get_device_name(0),
                                 'torch_version': torch.__version__,
                                 'torch_cuda_version': torch.version.cuda,
                                 'torch_cpu_threads': torch.get_num_threads(),
                                 'omp': os.environ['OMP_NUM_THREADS'],
                                 'mkl': os.environ['MKL_NUM_THREADS'],
                                 'openblas': os.environ['OPENBLAS_NUM_THREADS']}
        geometry = TorchFootGeometry.load(
            root/'e111-cuda-foot-geometry/geometry.npz', device='cuda:0')
        require(geometry.geometry_hash == GEOMETRY_HASH,
                'Original E32 fixed geometry hash changed')
        audit = [audit_own_background(f['current'][i], f['source_rate'][i],
                                      f['weights'][i], f['background'][i])
                 for i in range(2)]
        np.savez_compressed(out/'background-roundoff-raw.npz',
            derived=np.stack([x['derived'] for x in audit]),
            stored=f['background'], difference=np.stack([x['difference'] for x in audit]),
            bound=np.stack([x['bound'] for x in audit]))
        save_json(out/'background-roundoff.json', [{k: v for k, v in row.items()
                   if k not in ('derived', 'difference', 'bound')} for row in audit])
        require(all(x['passed'] for x in audit),
                'Own-W background reconstruction exceeded E112b gamma bound')
        deadline()
        staged = ContextResponse(time_s=f['time_s'], tau=f['tau'], a=f['a'],
            threshold=f['threshold'], cap=f['cap'], pool=f['pool'],
            moment=f['moment'], rest=f['rest'], ltm=f['ltm'], device='cuda:0')
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device='cuda:0')
        # Input transfer can fail before evaluate; charge only an entered response.
        response_inputs = (tensor(f['source_rate']), tensor(f['background']),
            tensor(f['weights']),
            torch.as_tensor(f['pair_context_indices'], device='cuda:0'),
            torch.as_tensor(f['pair_weight_indices'], device='cuda:0'))
        deadline()
        ledger.update(stage='response', batch_started=1, pairs_started=4,
                      conditional_model_s_started=40., batch_reservation_requested=4)
        try:
            persist()
        except BaseException:
            # The call was never entered; retain the attempted reservation.
            ledger.update(batch_started=0, pairs_started=0,
                          conditional_model_s_started=0.,
                          batch_start_receipt_failed=True)
            raise
        ledger['response_call_entered'] = True
        output = staged.evaluate(*response_inputs)
        detach_numpy(output, raw)
        torch.cuda.synchronize()
        ledger.update(stage='response_materialized', batch_materialized=1,
                      pairs_materialized=4, conditional_model_s_materialized=40.)
        persist()
        raw['pair_context_indices'] = f['pair_context_indices'].copy()
        raw['pair_weight_indices'] = f['pair_weight_indices'].copy()
        raw['time_s'] = f['time_s'].copy()
        np.savez_compressed(out/'response-raw.npz', **raw)
        ledger.update(stage='response_raw_saved', batch_raw_saved=1,
                      pairs_raw_saved=4)
        persist()
        deadline()
        selected = np.flatnonzero(f['fixed_mask'])
        command = tensor(raw['command'][:, selected].reshape(-1, 42))
        feet_fixed = geometry.full(command).detach().cpu().numpy().reshape(
            4, len(selected), 6, 3)
        own_target_feet = [geometry.full(tensor(f['own_target'][i, f['own_mask'][i]]))
            .detach().cpu().numpy() for i in range(2)]
        torch.cuda.synchronize()
        np.savez_compressed(out/'foot-proxy-raw.npz', feet_fixed=feet_fixed,
            own_target_feet_E110=own_target_feet[0],
            own_target_feet_E113=own_target_feet[1],
            fixed_mask=f['fixed_mask'], own_mask=f['own_mask'],
            common_mask=f['common_mask'])
        ledger['stage'] = 'geometry_raw_saved'
        persist()
        deadline()
        baseline = independent_baselines(raw, root)
        metrics = response_metrics(raw, f, feet_fixed, own_target_feet)
        save_json(out/'independent-baselines.json', baseline)
        save_json(out/'metrics.json', metrics)
        save_json(out/'provenance.json', {
            'input_sha256': INPUT_SHA, 'design_sha256': DESIGN_SHA,
            'frozen_design_sha256': FROZEN_SHA,
            'runner_sha256': sha256(ROOT/'tools/verify_c_banc_m3_e114_contexts.py'),
            'helper_sha256': sha256(ROOT/'tools/banc_m3_context_response.py'),
            'test_sha256': sha256(ROOT/'tests/test_c_banc_context_response.py'),
            'source_hashes': provenance['manifest']['input_sha256'],
            'geometry_hash': geometry.geometry_hash,
            'pair_order': PAIRS, 'response_semantics':
            'recorded source and own-W residual, candidate W; endpoint approximation'})
        require(baseline['all_passed'], 'Independent three-response numeric gate failed')
        deadline()
        ledger['stage'] = 'complete'
        persist()
        result = {'valid_conditional_validation': True,
                  'existing_three_baselines_pass': True,
                  'new_cross_pair_requires_parent_numpy_check': True,
                  'fit_or_native_gait_improvement_claim': False,
                  'internal_wall_s': time.monotonic()-start,
                  'internal_limit_s': LIMIT_S,
                  'ledger': ledger}
        save_json(out/'result.json', result)
        return result
    except BaseException as error:
        primary = error
        ledger['stage'] = 'error'
        ledger['primary_error'] = {'type': type(error).__name__,
                                   'message': str(error),
                                   'traceback': traceback.format_exc()}
        refresh_clock(error)
        failure_save('error.json', {'primary_error': ledger['primary_error'],
                     'internal_wall_s': ledger['internal_wall_s'],
                     'internal_wall_status': ledger['internal_wall_status']}, error)
        if ledger['batch_raw_saved'] == 0:
            rescue_error_raw(out, raw, output, ledger, error)
        raise
    finally:
        if primary is not None:
            refresh_clock(primary)
            failure_save('execution-ledger.json', ledger, primary)
            failure_save('error.json', {'primary_error': ledger['primary_error'],
                         'internal_wall_s': ledger['internal_wall_s'],
                         'internal_wall_status': ledger['internal_wall_status'],
                         'error_raw_saved_fields': ledger.get('error_raw_saved_fields', [])},
                         primary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/
        'verification/banc-natural-gait-m3-20260919')
    parser.add_argument('--out', type=Path, default=ROOT/
        'verification/banc-natural-gait-m3-20260919/e114-context-validation')
    args = parser.parse_args()
    run_once(args.root, args.out)


if __name__ == '__main__':
    main()
