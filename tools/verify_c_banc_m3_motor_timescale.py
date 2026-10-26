"""Single reviewed E138 held-input native timescale diagnostic.

The original graph is evaluated only by an explicit, SHA-gated CLI call.
This module does not construct a body or change the saved E121 CSR.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.storage import StateStore
from tools import verify_c_banc_m3_motor_adaptation as e123
from tools.banc_m3_motor_timescale import (
    LOG_TAU_LOWER,
    LOG_TAU_UPPER,
    MotorTimescaleNetwork,
    canonical_timescales,
    map_muscle_log_tau,
)
from tools.verify_c_banc_observation import exact

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'verification/banc-natural-gait-m3-20260919'
FREEZE = BASE/'e138-native-timescale-freeze.json'
FREEZE_SHA = '896ff32ddc7fb108d4c221b8d6e27c7e2d0a31f8db362eb887b377cff5ded6a1'
DESIGN_SHA = '37f7adc0d41ee79af7d78a7e182b9f45aaa3e5ef1741e68952e56688e0442dbc'
OWNED = ('tools/banc_m3_motor_timescale.py',
         'tools/verify_c_banc_m3_motor_timescale.py',
         'tests/test_c_banc_motor_timescale.py')
INTERNAL_LIMIT_S = 30.
RESERVE_S = 6.
RAW_LIMIT_BYTES = 1 << 30
ARMS = (('original_motor_gain_one', 10), ('zero_log_tau', 10),
        ('heterogeneous_log_tau', 10), ('heterogeneous_future', 1),
        ('clone_future', 1))


def _path_from_freeze(relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Only frozen repository-relative paths allowed')
    return ROOT/path


def reviewed_inputs(freeze_path, freeze_sha256, source_review_path,
                    source_review_sha256):
    """No graph/model allocation before exact review and preservation checks."""
    if freeze_sha256 != FREEZE_SHA or e123.sha256(freeze_path) != FREEZE_SHA:
        raise ValueError('Exact E138 freeze SHA required')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    protocol = freeze.get('protocol', {})
    conversion = freeze.get('conversion', {})
    if (freeze.get('design_sha256') != DESIGN_SHA
            or e123.sha256(_path_from_freeze(freeze['design_path'])) != DESIGN_SHA
            or freeze.get('e123_freeze_sha256') != e123.FREEZE_SHA
            or freeze.get('source_core_identity') != e123.CORE_IDENTITY
            or len(freeze.get('input_sha256', {})) != 89
            or len(freeze.get('reused_source_sha256', {})) != 43
            or tuple(freeze.get('owned_files', ())) != OWNED
            or freeze.get('dimensions') != {'full_neurons': 158706,
                'motor_neurons': 391, 'muscles': 96, 'lf_neurons': 69,
                'lf_muscles': 17}
            or [(a['name'], a['controls']) for a in protocol.get('arms', [])]
                != list(ARMS)
            or protocol.get('original_tick') != 4000
            or protocol.get('neural_dt_s') != .00025
            or protocol.get('steps_per_control') != 20
            or protocol.get('motor_gain') != 1.
            or protocol.get('baseline_tau_s') != .3
            or protocol.get('original_neural_s') != .16
            or protocol.get('original_neural_ticks') != 640
            or protocol.get('decoder_controls') != 32
            or protocol.get('full_model_constructions') != 4
            or protocol.get('cuda_setup_steps_per_model') != 60
            or protocol.get('cuda_setup_steps') != 240
            or conversion.get('log_tau_bounds') != [LOG_TAU_LOWER, LOG_TAU_UPPER]
            or conversion.get('log_tau_dtype') != 'float64'
            or conversion.get('cpu') != 'numerator / tau32'
            or conversion.get('packed_cuda') != 'mul_exact(numerator, inverse32)'
            or freeze.get('internal_wall_limit_s') != INTERNAL_LIMIT_S
            or freeze.get('reserve_s') != RESERVE_S
            or freeze.get('maximum_raw_bytes') != RAW_LIMIT_BYTES
            or freeze.get('cpu_threads') != 1 or freeze.get('gpu_count') != 1
            or freeze.get('original_processes') != 1 or freeze.get('retries') != 0
            or freeze.get('physical_body_calls') != 0 or freeze.get('fit_calls') != 0
            or freeze.get('LP_calls') != 0 or freeze.get('F6_exposed') is not False):
        raise ValueError('Frozen E138 protocol or resource contract changed')
    if (e123.sha256(source_review_path) != source_review_sha256
            or len(source_review_sha256) != 64):
        raise ValueError('Exact parent source review SHA required')
    review = json.loads(Path(source_review_path).read_text(encoding='utf-8'))
    if (review.get('accepted_for_single_original_native_timescale') is not True
            or review.get('blocking_defects') != []
            or review.get('freeze_sha256') != FREEZE_SHA
            or review.get('source_sha256') != {
                name: e123.sha256(ROOT/name) for name in OWNED}):
        raise ValueError('Accepted exact E138 parent source review required')
    for mapping in ('input_sha256', 'reused_source_sha256'):
        for relative, expected in freeze[mapping].items():
            if e123.sha256(_path_from_freeze(relative)) != expected:
                raise ValueError('Frozen '+mapping+' changed: '+relative)
    if (e123.sha256(_path_from_freeze(freeze['selected_raw_path']))
            != freeze['selected_raw_sha256']):
        raise ValueError('Selected E137 raw changed')
    e123.reviewed_inputs(_path_from_freeze(freeze['e123_freeze_path']))
    return freeze


def checked_timescale_inputs(freeze, cal, anatomy):
    """Recheck the saved anatomical proposal independently of its preparation."""
    data = e123.load_npz(_path_from_freeze(freeze['input_arrays_path']))
    expected_keys = {'motor_ids', 'membership', 'muscle_log_tau',
        'selected_muscle_indices', 'lf_motor_columns', 'motor_log_tau',
        'motor_tau64', 'motor_tau32', 'motor_inverse_tau32', 'motor_gains',
        'selected_raw_sha256'}
    if set(data) != expected_keys:
        raise ValueError('Exact E138 input array set required')
    ids, motor_ell, muscle_ell = map_muscle_log_tau(
        cal['motor_ids'], cal['pool'], cal['membership'], anatomy['muscles'],
        data['muscle_log_tau'])
    selected = data['selected_muscle_indices']
    lf_columns = data['lf_motor_columns']
    if (selected.shape != (17,) or selected.dtype != np.int64
            or len(np.unique(selected)) != 17
            or lf_columns.shape != (69,) or lf_columns.dtype != np.int64
            or not np.array_equal(lf_columns, np.flatnonzero(np.isin(
                cal['membership'], selected)))
            or np.any(muscle_ell[np.setdiff1d(np.arange(96), selected)] != 0)
            or not np.array_equal(ids, data['motor_ids'])
            or not np.array_equal(cal['membership'], data['membership'])
            or data['membership'].dtype != np.int64
            or data['motor_log_tau'].dtype != np.float64
            or motor_ell.tobytes() != data['motor_log_tau'].tobytes()
            or data['motor_gains'].dtype != np.float32
            or not np.array_equal(data['motor_gains'], np.ones(391, np.float32))
            or str(data['selected_raw_sha256']) != freeze['selected_raw_sha256']):
        raise ValueError('Original 17-muscle/391-MN proposal mapping changed')
    with np.load(_path_from_freeze(freeze['selected_raw_path']),
                 allow_pickle=False) as source:
        if (not np.array_equal(source['muscle_indices'], selected)
                or source['log_tau'].dtype != np.float64
                or source['log_tau'].tobytes() != muscle_ell[selected].tobytes()):
            raise ValueError('Selected E137 raw log-tau provenance changed')
    tau64, tau32, inverse32 = canonical_timescales(ids, motor_ell, 158706)
    for key, calculated in (('motor_tau64', tau64[ids]),
                            ('motor_tau32', tau32[ids]),
                            ('motor_inverse_tau32', inverse32[ids])):
        saved = data[key]
        if saved.dtype != calculated.dtype or saved.shape != (391,) \
                or saved.tobytes() != calculated.tobytes():
            raise ValueError('Frozen canonical actual '+key+' changed')
    changed = int(np.count_nonzero(inverse32[ids] != np.float32(1/.3)))
    if changed != 53:
        raise ValueError('Expected 53 actual changed native inverse cells')
    return data


def _save_timescale_configuration(path, network, ids, requested_ell):
    """Save requested and actual full physiology before comparing any arm."""
    if path.exists():
        raise FileExistsError('Timescale raw already exists: '+str(path))
    if isinstance(network, MotorTimescaleNetwork):
        pointers = (network._cell_adaptation_tau.data_ptr(),
                    network._cell_inverse_tau.data_ptr())
        tau32 = network._cell_adaptation_tau.detach().cpu().numpy().copy()
        inverse32 = network._cell_inverse_tau.detach().cpu().numpy().copy()
        tau64 = network._full_tau64.copy()
        origin = 'device_tensors'
    else:
        tau64 = np.full(network.n, .3, np.float64)
        tau32 = np.full(network.n, .3, np.float32)
        inverse32 = np.full(network.n, 1/.3, np.float32)
        origin = 'original_scalar'
    np.savez_compressed(path, model_hash=np.asarray(network.identity),
        source_motor_identity=np.asarray(getattr(network, 'source_motor_identity',
                                                  network.identity)),
        actual_origin=np.asarray(origin), motor_ids=np.asarray(ids, np.int64),
        motor_log_tau=np.asarray(requested_ell, np.float64),
        full_tau64=tau64, full_tau32=tau32, full_inverse32=inverse32,
        motor_tau64=tau64[ids], motor_tau32=tau32[ids],
        motor_inverse32=inverse32[ids])
    if isinstance(network, MotorTimescaleNetwork) and (
            pointers != (network._cell_tau_ptr, network._cell_inverse_ptr)
            or tau32.tobytes() != network._full_tau32.tobytes()
            or inverse32.tobytes() != network._full_inverse32.tobytes()):
        raise ValueError('Saved actual native timescale tensor changed')
    return e123.sha256(path)


def _finite_arm(raw, name):
    for key in ('rate', 'adaptation', 'drive', 'output_mask', 'motor_rate',
                'activation', 'request', 'tendon'):
        if not np.isfinite(raw[key]).all():
            raise ValueError(name+' has nonfinite '+key)
    if not np.all(raw['decoded']):
        raise ValueError(name+' has incomplete decoder observations')


def _raw_bytes(output):
    return sum(path.stat().st_size for path in output.rglob('*') if path.is_file())


def require_room(began, label):
    if time.perf_counter()-began+RESERVE_S >= INTERNAL_LIMIT_S:
        raise TimeoutError(label+' needs 6 s persistence/cleanup reserve')


def durable_preentry(receipt, row, label, unknown_key, secondary, began):
    """Persist intent and request, then enter; failed receipts mean zero work."""
    def rollback(error, phase):
        row.update(requested=False, entered=False, known_zero_work=True,
                   stage=label+'_'+phase+'_receipt_failed',
                   pre_entry_error=repr(error),
                   finished_wall_s=time.perf_counter()-began)
        row[unknown_key] = False
        try:
            receipt()  # Best-effort durable correction of a partial write.
        except Exception as repair_error:  # noqa: BLE001
            secondary.append({'stage': label+' pre-entry receipt correction',
                'error': type(repair_error).__name__,
                'message': str(repair_error)})

    try:
        receipt()  # Durable intent: no native or storage call has begun.
    except Exception as error:
        rollback(error, 'intent')
        raise
    row.update(requested=True, stage=label+'_requested')
    row[unknown_key] = True  # A crash after this receipt has uncertain work.
    try:
        receipt()  # Durable request before actual callback entry.
    except Exception as error:
        rollback(error, 'entry')
        raise
    row.update(entered=True, stage=label+'_entered')


def run(freeze_path, freeze_sha256, source_review_path,
        source_review_sha256, output):
    began = time.perf_counter()
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve single E138 original output: '+str(output))
    output.mkdir(parents=True, exist_ok=False)
    ledger = {'logical_steps': 0, 'logical_neural_s': 0.,
        'physical_steps': 0, 'decoder_controls': 0, 'arms': {},
        'setup_scientific_steps': 0, 'model_attempts': [],
        'configuration_sha256': {}, 'timescale_configuration_sha256': {},
        'cuda_setup_model_count': 0, 'cuda_setup_kernel_steps': 0,
        'advance_requests': [], 'state_store_requests': [],
        'raw_receipts': {}, 'raw_bytes': 0}
    secondary = []
    primary = None
    result = None
    thread_limits = None
    active = None
    network = hetero = clone = None
    cal = original = held = graph = weights = parameters = None

    def receipt():
        e123.write_json(output/'advance-ledger.json', ledger)

    def persist_active():
        nonlocal active
        if active is None:
            return
        name, rows, model, origin = active
        path = output/(name+'-raw.npz')
        try:
            e123.save_arm_raw(path, name, rows, model, cal, held, origin)
        except Exception as error:
            emergency = output/(name+'-raw-emergency.npz')
            try:
                e123.save_arm_raw(emergency, name, rows, model, cal, held, origin)
                ledger['raw_receipts'][name] = {'path': emergency.name,
                    'sha256': e123.sha256(emergency), 'rows': len(rows),
                    'canonical_save_error': repr(error)}
                active = None
            except Exception as fallback_error:  # noqa: BLE001
                secondary.append({'stage': name+' emergency raw save',
                    'error': type(fallback_error).__name__,
                    'message': str(fallback_error)})
            raise
        ledger['raw_receipts'][name] = {'path': path.name,
            'sha256': e123.sha256(path), 'rows': len(rows)}
        ledger['raw_bytes'] = _raw_bytes(output)
        active = None
        receipt()
        if ledger['raw_bytes'] > RAW_LIMIT_BYTES:
            raise MemoryError('E138 total raw exceeds 1 GiB')

    def start_model(name, ell=None):
        require_room(began, name+' constructor')
        attempt = {'owner': name, 'requested': False, 'entered': False,
            'returned': False, 'known_zero_work': False,
            'unknown_partial_setup': False, 'stage': 'intent',
            'cuda_setup_kernel_steps_completed': 0,
            'started_wall_s': time.perf_counter()-began}
        ledger['model_attempts'].append(attempt)
        durable_preentry(receipt, attempt, 'constructor',
                         'unknown_partial_setup', secondary, began)
        constructor_error = None
        try:
            if ell is None:
                candidate = e123.make_network(weights, parameters, original,
                    cal['motor_ids'], 1., setup_record=attempt)
            else:
                candidate = MotorTimescaleNetwork(weights, *parameters,
                    adaptation_gain=8., adaptation_tau_s=.3, dt=.00025,
                    device='cuda', cuda_implementation='packed', capture_steps=20,
                    core_ids=np.asarray(original['network']['core_adaptation']['ids']),
                    core_gain=0., motor_ids=cal['motor_ids'],
                    motor_gains=np.ones(len(cal['motor_ids']), np.float32),
                    motor_log_tau=ell)
                attempt.update(stage='constructor_and_capture_completed',
                               cuda_setup_kernel_steps_completed=(
                                   60 if candidate.device == 'cuda' else 0))
                if candidate.source_core_identity != e123.CORE_IDENTITY:
                    raise ValueError('Original core source identity changed')
                saved = copy.deepcopy(original['network'])
                saved['model_hash'] = candidate.identity
                saved['motor_adaptation'] = candidate.snapshot()['motor_adaptation']
                saved['motor_timescale'] = candidate.snapshot()['motor_timescale']
                candidate.restore(saved)
                if candidate.tick != 4000:
                    raise ValueError('New model changed original restore tick')
            if candidate.device != 'cuda':
                attempt['cuda_setup_kernel_steps_completed'] = 0
            attempt['returned'] = True
            attempt['unknown_partial_setup'] = False
            attempt['stage'] = 'restored_original_tick4000'
            return candidate
        except Exception as error:
            constructor_error = error
            attempt['unknown_partial_setup'] = (
                attempt['cuda_setup_kernel_steps_completed'] != 60)
            raise
        finally:
            attempt['finished_wall_s'] = time.perf_counter()-began
            ledger['cuda_setup_model_count'] = sum(
                row['cuda_setup_kernel_steps_completed'] == 60
                for row in ledger['model_attempts'])
            ledger['cuda_setup_kernel_steps'] = sum(
                row['cuda_setup_kernel_steps_completed']
                for row in ledger['model_attempts'])
            try:
                receipt()
            except Exception as save_error:
                if constructor_error is None:
                    raise
                secondary.append({'stage': name+' model receipt save',
                    'error': type(save_error).__name__,
                    'message': str(save_error)})

    def control(name, model, activation, rows):
        require_room(began, name+' advance')
        item = {'owner': name, 'stage': 'intent',
            'requested': False, 'entered': False,
            'returned': False, 'known_zero_work': False,
            'unknown_partial_advance': False, 'before_tick': int(model.tick),
            'requested_steps': 20, 'observed_delta': 0,
            'started_wall_s': time.perf_counter()-began}
        ledger['advance_requests'].append(item)
        durable_preentry(receipt, item, 'advance',
                         'unknown_partial_advance', secondary, began)
        advance_error = None
        row_count = len(rows)
        try:
            next_activation, row = e123.accounted_control(
                model, held, 20, cal, activation, ledger, name, rows)
            item['returned'] = True
            item['unknown_partial_advance'] = False
            item['synchronized_observation'] = True
            return next_activation, row
        except Exception as error:
            advance_error = error
            observed = len(rows) == row_count+1 and not rows[-1]['read_errors']
            item['synchronized_observation'] = bool(observed)
            item['unknown_partial_advance'] = not bool(observed)
            raise
        finally:
            item['observed_delta'] = int(model.tick)-item['before_tick']
            item['finished_wall_s'] = time.perf_counter()-began
            try:
                receipt()
            except Exception as save_error:
                if advance_error is None:
                    raise
                secondary.append({'stage': name+' advance receipt save',
                    'error': type(save_error).__name__,
                    'message': str(save_error)})

    def accounted_operation(label, path, callback):
        item = {'owner': label, 'path': path, 'stage': 'intent',
            'requested': False, 'entered': False, 'returned': False,
            'known_zero_work': False, 'unknown_partial_work': False,
            'started_wall_s': time.perf_counter()-began}
        if label == 'manifest':
            ledger['manifest'] = item
        else:
            ledger['state_store_requests'].append(item)
        durable_preentry(receipt, item, label, 'unknown_partial_work',
                         secondary, began)
        work_error = None
        try:
            value = callback()
            item.update(returned=True, unknown_partial_work=False,
                        stage=label+'_returned')
            return value
        except Exception as error:
            work_error = error
            item['stage'] = label+'_failed_after_entry'
            raise
        finally:
            item['finished_wall_s'] = time.perf_counter()-began
            try:
                receipt()
            except Exception as save_error:
                if work_error is None:
                    raise
                secondary.append({'stage': label+' return receipt save',
                    'error': type(save_error).__name__,
                    'message': str(save_error)})

    def arm(name, model, activation, controls, origin):
        nonlocal active
        ledger['arms'][name] = 0
        active = (name, [], model, origin)
        receipt()
        last = None
        for _ in range(controls):
            activation, last = control(name, model, activation, active[1])
        persist_active()
        return activation, last

    try:
        thread_limits, ledger['observed_cpu_threads'] = e123.configure_one_cpu_thread()
        freeze = reviewed_inputs(freeze_path, freeze_sha256,
                                 source_review_path, source_review_sha256)
        require_room(began, 'full graph preparation')
        graph, cal, _metadata, anatomy, original = e123.native_graph_and_inputs(freeze)
        proposal = checked_timescale_inputs(freeze, cal, anatomy)
        weights = e123.reconstruct_weights(graph, cal)
        parameters = e123.native_parameters(graph, cal)
        source_gains = np.full(graph.n, 8., np.float32)
        source_gains[original['network']['core_adaptation']['ids']] = 0.
        held = np.asarray(original['network']['drive'], np.float32).copy()
        if (not np.isfinite(held).all()
                or not np.isfinite(original['network']['rate']).all()
                or not np.isfinite(original['network']['adaptation']).all()):
            raise ValueError('Finite original saved full neural state required')
        e123.write_json(output/'spec.json', {'schema': 'flylab.m3-e138-timescale-diagnostic.v1',
            'freeze_sha256': FREEZE_SHA, 'design_sha256': DESIGN_SHA,
            'runner_sha256': e123.sha256(__file__),
            'network_tool_sha256': e123.sha256(ROOT/OWNED[0]),
            'source_review_sha256': source_review_sha256,
            'graph_hash': graph.hash, 'full_neurons': graph.n,
            'held_input': True, 'closed_loop': False, 'cpu_threads': 1,
            'internal_limit_s': INTERNAL_LIMIT_S, 'reserve_s': RESERVE_S,
            'candidate_adopted': False, 'F6_exposed': False,
            'whole_goal_complete': False})

        for name, ell in (('original_motor_gain_one', None),
                          ('zero_log_tau', np.zeros(len(cal['motor_ids']), np.float64)),
                          ('heterogeneous_log_tau', proposal['motor_log_tau'])):
            network = start_model(name, ell)
            if name == 'heterogeneous_log_tau':
                hetero = network
            ids = cal['motor_ids']
            actual_ell = np.zeros(len(ids), np.float64) if ell is None else ell
            ledger['configuration_sha256'][name] = e123.save_model_configuration(
                output/(name+'-configuration.npz'), network, weights, parameters,
                source_gains, ids, 1.)
            ledger['timescale_configuration_sha256'][name] = (
                _save_timescale_configuration(output/(name+'-timescale-configuration.npz'),
                                              network, ids, actual_ell))
            receipt()
            activation = np.asarray(original['adapter']['activation'],
                                    np.float64).copy()
            arm(name, network, activation, 10, original)
            if name == 'zero_log_tau':
                network = None
                gc.collect()
            elif name == 'original_motor_gain_one':
                network = None

        checkpoint = {'schema': 'flylab.m3-e138-held-checkpoint.v1',
            'graph_hash': graph.hash,
            'calibration_sha256': e123.sha256(_path_from_freeze(
                freeze['source_e121'])/'calibration.npz'),
            'calibration': {key: value.copy() for key, value in cal.items()},
            'held_drive': held.copy(),
            'activation': e123.load_npz(
                output/'heterogeneous_log_tau-raw.npz')['activation'][-1].copy(),
            'network': hetero.snapshot()}
        require_room(began, 'heterogeneous checkpoint')
        accounted_operation('state_store_save', 'heterogeneous-checkpoint',
            lambda: StateStore.save(output/'heterogeneous-checkpoint', checkpoint))
        disk = accounted_operation('state_store_load',
            'heterogeneous-checkpoint',
            lambda: StateStore.load(output/'heterogeneous-checkpoint'))
        exact(checkpoint, disk, 'E138 complete disk checkpoint')
        exact(cal, disk['calibration'], 'E138 original disk calibration')

        future_origin = e123.raw_origin(graph.hash, checkpoint['network'],
                                        checkpoint['activation'])
        main_activation, main_row = arm('heterogeneous_future', hetero,
            checkpoint['activation'].copy(), 1, future_origin)
        main_state = hetero.snapshot()
        clone = start_model('heterogeneous_disk_clone', proposal['motor_log_tau'])
        if (disk['schema'] != checkpoint['schema']
                or disk['graph_hash'] != graph.hash
                or disk['calibration_sha256'] != checkpoint['calibration_sha256']
                or not np.array_equal(disk['held_drive'], held)):
            raise ValueError('Disk source identity or held input changed')
        clone.restore(disk['network'])
        if clone.tick != 4200:
            raise ValueError('Disk clone must start at tick4200')
        ledger['configuration_sha256']['heterogeneous_disk_clone'] = (
            e123.save_model_configuration(output/'heterogeneous_disk_clone-configuration.npz',
                clone, weights, parameters, source_gains, cal['motor_ids'], 1.))
        ledger['timescale_configuration_sha256']['heterogeneous_disk_clone'] = (
            _save_timescale_configuration(
                output/'heterogeneous_disk_clone-timescale-configuration.npz',
                clone, cal['motor_ids'], proposal['motor_log_tau']))
        receipt()
        clone_origin = e123.raw_origin(graph.hash, disk['network'], disk['activation'])
        clone_activation, clone_row = arm('clone_future', clone,
            disk['activation'].copy(), 1, clone_origin)
        clone_state = clone.snapshot()

        # Numerical decisions occur only after all five arm raws are durable.
        raws = {name: e123.load_npz(output/(name+'-raw.npz')) for name, _ in ARMS}
        for name, raw in raws.items():
            _finite_arm(raw, name)
        original_times = e123.load_npz(
            output/'original_motor_gain_one-timescale-configuration.npz')
        zero_times = e123.load_npz(
            output/'zero_log_tau-timescale-configuration.npz')
        gate_errors = []
        try:
            e123.exact_arm(raws['original_motor_gain_one'], raws['zero_log_tau'],
                           'E138 gain1 versus ell0 whole state and decoder')
            for key in ('initial_rate', 'initial_adaptation', 'initial_output_mask',
                        'initial_tick', 'initial_activation'):
                exact(raws['original_motor_gain_one'][key],
                      raws['zero_log_tau'][key], 'E138 neutral initial '+key)
            for key in ('full_tau64', 'full_tau32', 'full_inverse32'):
                exact(original_times[key], zero_times[key],
                      'E138 neutral actual '+key)
        except AssertionError as error:
            gate_errors.append({'gate': 'neutral_parity', 'detail': str(error)})
        lf_ids = cal['motor_ids'][proposal['lf_motor_columns']]
        changed = raws['heterogeneous_log_tau']
        original_raw = raws['original_motor_gain_one']
        effect = not (
            np.array_equal(changed['rate'][:, lf_ids], original_raw['rate'][:, lf_ids])
            or np.array_equal(changed['adaptation'][:, lf_ids],
                              original_raw['adaptation'][:, lf_ids])
            or np.array_equal(changed['request'], original_raw['request']))
        if not effect:
            gate_errors.append({'gate': 'heterogeneous_actual_effect',
                'detail': 'LF rate, adaptation and request must all change'})
        hetero_times = e123.load_npz(
            output/'heterogeneous_log_tau-timescale-configuration.npz')
        unchanged = np.ones(graph.n, bool)
        unchanged[cal['motor_ids']] = False
        try:
            for key in ('full_tau64', 'full_tau32', 'full_inverse32'):
                exact(hetero_times[key][unchanged], original_times[key][unchanged],
                      'E138 unchanged non-MN '+key)
        except AssertionError as error:
            gate_errors.append({'gate': 'unchanged_non_motor_physiology',
                                'detail': str(error)})
        try:
            exact(main_state, clone_state, 'E138 disk clone whole neural future')
            exact(main_activation, clone_activation, 'E138 disk clone muscle future')
            exact(main_row['request'], clone_row['request'], 'E138 disk clone request')
            exact(main_row['tendon'], clone_row['tendon'], 'E138 disk clone tendon')
            e123.exact_arm(raws['heterogeneous_future'], raws['clone_future'],
                           'E138 disk future raw')
        except AssertionError as error:
            gate_errors.append({'gate': 'disk_future', 'detail': str(error)})
        if (ledger['logical_steps'] != 640 or ledger['decoder_controls'] != 32
                or ledger['logical_neural_s'] != .16
                or ledger['cuda_setup_model_count'] != 4
                or ledger['cuda_setup_kernel_steps'] != 240
                or ledger['physical_steps'] != 0 or _raw_bytes(output) > RAW_LIMIT_BYTES):
            raise ValueError('E138 fixed work/resource contract changed')
        require_room(began, 'final output manifest')

        def save_manifest():
            files = {str(path.relative_to(output)).replace('\\', '/'):
                e123.sha256(path) for path in sorted(output.rglob('*'))
                if path.is_file() and path.name not in (
                    'advance-ledger.json', 'output-manifest.json', 'result.json',
                    'execution-error.json')}
            e123.write_json(output/'output-manifest.json', {
                'schema': 'flylab.m3-e138-native-output-manifest.v1',
                'freeze_sha256': FREEZE_SHA,
                'source_review_sha256': source_review_sha256,
                'files': files, 'file_count': len(files)})

        accounted_operation('manifest', 'output-manifest.json', save_manifest)
        if time.perf_counter()-began > INTERNAL_LIMIT_S:
            raise TimeoutError('E138 internal30 s limit exceeded')
        result = {'validity': 'valid_full_BANC_held_neural_diagnostic',
            'hypothesis_verdict': 'refuted' if gate_errors else 'supported',
            'scope': 'native_timescale_capability_only',
            'neutral_whole_state_decoder_bit_exact': not any(
                item['gate'] == 'neutral_parity' for item in gate_errors),
            'heterogeneous_LF_actual_effect': effect,
            'disk_future_bit_exact': not any(
                item['gate'] == 'disk_future' for item in gate_errors),
            'numerical_gate_errors': gate_errors,
            'actual_neural_s': ledger['logical_neural_s'],
            'actual_neural_steps': ledger['logical_steps'],
            'decoder_controls': ledger['decoder_controls'],
            'actual_physical_s': 0., 'candidate_adopted': False,
            'F6_exposed': False, 'whole_goal_complete': False, 'wall_s': None}
    except Exception as error:  # noqa: BLE001 - never replace primary with cleanup
        primary = error
    finally:
        if active is not None:
            try:
                persist_active()
            except Exception as error:  # noqa: BLE001 - retain first failure
                if primary is None:
                    primary = error
                else:
                    secondary.append({'stage': 'partial raw persistence',
                        'error': type(error).__name__, 'message': str(error)})
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            active = network = hetero = clone = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as error:  # noqa: BLE001
            secondary.append({'stage': 'CUDA synchronization/cleanup',
                'error': type(error).__name__, 'message': str(error)})
        if thread_limits is not None:
            try:
                thread_limits.restore_original_limits()
            except Exception as error:  # noqa: BLE001
                secondary.append({'stage': 'CPU thread restoration',
                    'error': type(error).__name__, 'message': str(error)})
        ledger['raw_bytes'] = _raw_bytes(output)
        ledger['wall_s_before_final_receipts'] = time.perf_counter()-began
        try:
            receipt()
        except Exception as error:  # noqa: BLE001
            secondary.append({'stage': 'final ledger save',
                'error': type(error).__name__, 'message': str(error)})
        if primary is None and result is None:
            primary = RuntimeError('E138 ended without a complete result')
        if primary is None and secondary:
            primary = RuntimeError('E138 cleanup or persistence failed')
        if primary is None and time.perf_counter()-began > INTERNAL_LIMIT_S:
            primary = TimeoutError('E138 finalization exceeded internal30 s')
        if primary is None:
            result['wall_s'] = time.perf_counter()-began
            try:
                e123.write_json(output/'result.json', result)
            except Exception as error:  # noqa: BLE001
                primary = error
            if primary is None and time.perf_counter()-began > INTERNAL_LIMIT_S:
                primary = TimeoutError('E138 result save exceeded internal30 s')
        if primary is not None:
            if result is not None and (output/'result.json').exists():
                result['validity'] = ('resource_limit_incomplete' if isinstance(
                    primary, TimeoutError) else 'incomplete')
                result['hypothesis_verdict'] = 'invalid_not_tested'
                result['wall_s'] = time.perf_counter()-began
                try:
                    e123.write_json(output/'result.json', result)
                except Exception as error:  # noqa: BLE001
                    secondary.append({'stage': 'result invalidation',
                        'error': type(error).__name__, 'message': str(error)})
            try:
                e123.write_json(output/'execution-error.json', {
                    'primary': {'error': type(primary).__name__,
                        'message': str(primary),
                        'notes': getattr(primary, '__notes__', [])},
                    'secondary': secondary,
                    'actual_neural_steps': ledger['logical_steps'],
                    'actual_neural_s': ledger['logical_neural_s'],
                    'decoder_controls': ledger['decoder_controls'],
                    'model_attempts': ledger['model_attempts'],
                    'advance_requests': ledger['advance_requests'],
                    'state_store_requests': ledger['state_store_requests'],
                    'manifest': ledger.get('manifest'),
                    'raw_receipts': ledger['raw_receipts'],
                    'wall_s': time.perf_counter()-began,
                    'candidate_adopted': False, 'F6_exposed': False,
                    'whole_goal_complete': False})
            except Exception as error:  # noqa: BLE001
                primary.add_note('Execution error receipt failed: '+repr(error))
    if primary is not None:
        for item in secondary:
            primary.add_note('Secondary '+item['stage']+': '+item['message'])
        raise primary
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--source-review', type=Path, required=True)
    parser.add_argument('--source-review-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.freeze, args.freeze_sha256, args.source_review,
                         args.source_review_sha256, args.output)), flush=True)
