"""E123 held-input full-BANC neural diagnostic; run only after final source review.

The existing 1 s E121 checkpoint supplies the neural state and fixed drive.
No body, world, fit, sensory loop, or gait candidate is constructed here.
"""

import argparse
import copy
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.banc_m3_core_adaptation import CoreAdaptationNetwork
from tools.banc_m3_intrinsic_recruitment import patch_tau
from tools.banc_m3_motor_adaptation import MotorAdaptationNetwork
from tools.banc_m3_synaptic_efficacy import patch_weights
from tools.verify_c_banc_observation import exact

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT/'verification/banc-natural-gait-m3-20260919/e123-frozen-design.json'
FREEZE_SHA = '00e5823b2b602b5c101293020fc01b4e90d349e30d5c20aee5a3cfa8ff525ef8'
DESIGN_SHA = '170edb425476fc6e1a36c4ba0c0529b1d81890492d81078e08b52f4157a52909'
CORE_IDENTITY = 'bfc89ead7088892f8c7fb479d25010c35a9c590825e777b8e8c9b9a3d01674d7'
INTERNAL_LIMIT_S = 45.0
RESERVE_S = 8.0


def sha256(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False)+'\n', encoding='utf-8')


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def reviewed_inputs(freeze_path=FREEZE):
    """Check all frozen input/source files before graph or model construction."""
    if sha256(freeze_path) != FREEZE_SHA:
        raise ValueError('E123 frozen design SHA changed')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    if (freeze.get('design_sha256') != DESIGN_SHA
            or len(freeze.get('input_sha256', {})) != 78
            or len(freeze.get('reused_source_sha256', {})) != 40
            or {key: len(value) for key, value in freeze.get('preservation_sha256', {}).items()}
            != {'E121': 211, 'E122': 21, 'M2': 263}):
        raise ValueError('Exact E123 input/source/preservation contract required')
    for mapping in ('input_sha256', 'reused_source_sha256'):
        for relative, expected in freeze[mapping].items():
            name = Path(relative)
            if name.is_absolute() or '..' in name.parts or sha256(ROOT/name) != expected:
                raise ValueError('Frozen '+mapping+' changed: '+relative)
    for owner, files in freeze['preservation_sha256'].items():
        for relative, expected in files.items():
            name = Path(relative)
            if name.is_absolute() or '..' in name.parts or sha256(ROOT/name) != expected:
                raise ValueError('Original '+owner+' artifact changed: '+relative)
    source = ROOT/freeze['source_e121']
    for directory, count, review_key in (
            ('e121-port-interference', 211, 'accepted_as_valid_actual_diagnostic'),
            ('e122-support-transfer', 21, 'accepted_as_valid_recorded_diagnostic')):
        root = source.parent/directory
        manifest = json.loads((root/'original-output-manifest.json').read_text(encoding='utf-8'))
        review = json.loads((root/'parent-final-review.json').read_text(encoding='utf-8'))
        if (manifest.get('original_file_count') != count
                or len(manifest.get('original_files', {})) != count
                or review.get(review_key) is not True
                or review.get('blocking_defects') != []):
            raise ValueError('Accepted original '+directory+' manifest/review required')
    if sha256(ROOT/'verification/banc-gait-quality-m2-20260912/e76-final-manifest.json') != (
            freeze['input_sha256']['verification/banc-gait-quality-m2-20260912/e76-final-manifest.json']):
        raise ValueError('Original M2 manifest changed')
    return freeze


def native_graph_and_inputs(freeze):
    graph = GraphStore.load(ROOT/freeze['graph_directory'])
    if graph.n != 158706 or graph.hash != (
            '9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff'):
        raise ValueError('Original full BANC graph required')
    source = ROOT/freeze['source_e121']
    cal = load_npz(source/'calibration.npz')
    metadata = load_npz(source/'recruitment-metadata.npz')
    anatomy = json.loads((source/'muscle-provenance.json').read_text(encoding='utf-8'))
    original = StateStore.load(source/'checkpoint-1s')
    motor_ids = np.asarray(cal['motor_ids'], dtype=np.int64)
    if (motor_ids.shape != (391,) or len(np.unique(motor_ids)) != 391
            or not np.array_equal(motor_ids, metadata['motor_ids'])
            or anatomy['graph_hash'] != graph.hash
            or len(anatomy['motor_roster']) != 391
            or len(anatomy['muscles']) != 96
            or [graph.nodes[int(i)]['id'] for i in motor_ids] != anatomy['motor_roster']):
        raise ValueError('Original 391 root-ID roster and selected MN order required')
    observed = metadata['observed_ids']
    if (observed.shape != (446,) or len(np.unique(observed)) != 446
            or not np.array_equal(metadata['_cell_gain'][np.isin(observed, motor_ids)],
                                  np.zeros(391, np.float32))):
        raise ValueError('Original observed gain0 MN physiology required')
    if (cal['pool'].shape != (96, 391) or cal['membership'].shape != (391,)
            or not np.array_equal(np.argmax(cal['pool'], axis=0), cal['membership'])):
        raise ValueError('Original single-muscle MN membership required')
    neural = original['network']
    if (neural['model_hash'] != CORE_IDENTITY or neural['tick'] != 4000
            or neural['dt'] != .00025 or neural['device'] != 'cuda'
            or neural['adaptation_gain'] != 8
            or neural['adaptation_tau_s'] != .3
            or original['control_tick'] != 200
            or neural['rate'].shape != (graph.n,)
            or neural['adaptation'].shape != (graph.n,)
            or neural['drive'].shape != (graph.n,)
            or neural['output_mask'].shape != (graph.n,)
            or original['adapter']['activation'].shape != (96,)):
        raise ValueError('Original 1 s full neural and muscle checkpoint required')
    if not np.array_equal(neural['core_adaptation']['ids'],
                          np.union1d(cal['core_ids'], cal['zero_adaptation_ids'])):
        raise ValueError('Original source core and selected gain0 roster changed')
    if neural['core_adaptation']['gain'] != 0:
        raise ValueError('Original selected MN gain0 required')
    return graph, cal, metadata, anatomy, original


def reconstruct_weights(graph, cal):
    unscaled, _ = rate_weights(graph)
    scaled = BancTargetEncoder(graph, forward_drive=5000).calibrate_weights(
        unscaled, cal['motor_ids'])
    return patch_weights(scaled, cal['edge_ids'], cal['source_values'], cal['values'])


def native_parameters(graph, cal):
    tau = patch_tau(np.full(graph.n, .02, np.float32),
                    cal['tau_ids'], cal['tau_values'])
    return (tau, np.full(graph.n, 1, np.float32),
            np.full(graph.n, 7.5, np.float32),
            np.full(graph.n, 200, np.float32))


def make_network(weights, parameters, original, motor_ids=None, gain=None,
                 setup_record=None):
    kwargs = {'adaptation_gain': 8., 'adaptation_tau_s': .3, 'dt': .00025,
              'device': 'cuda', 'cuda_implementation': 'packed',
              'capture_steps': 20,
              'core_ids': np.asarray(original['network']['core_adaptation']['ids']),
              'core_gain': 0.}
    if motor_ids is None:
        network = CoreAdaptationNetwork(weights, *parameters, **kwargs)
        if setup_record is not None:
            setup_record.update(stage='constructor_and_capture_completed',
                                cuda_setup_kernel_steps_completed=60)
        if network.identity != CORE_IDENTITY:
            raise ValueError('Original full CSR/parameter identity mismatch')
        network.restore(original['network'])
    else:
        gains = np.full(len(motor_ids), gain, dtype=np.float32)
        network = MotorAdaptationNetwork(weights, *parameters, motor_ids=motor_ids,
                                         motor_gains=gains, **kwargs)
        if setup_record is not None:
            setup_record.update(stage='constructor_and_capture_completed',
                                cuda_setup_kernel_steps_completed=60)
        if network.source_core_identity != CORE_IDENTITY:
            raise ValueError('New model does not preserve source core identity')
        saved = copy.deepcopy(original['network'])
        saved['model_hash'] = network.identity
        saved['motor_adaptation'] = network.snapshot()['motor_adaptation']
        network.restore(saved)
    if network.tick != 4000:
        raise ValueError('CUDA setup/capture changed scientific start tick')
    if setup_record is not None:
        setup_record['stage'] = 'restored_original_tick4000'
    return network


def decode(cal, activation, rate):
    pooled = cal['pool'] @ rate[cal['motor_ids']]
    activation = activation + -math.expm1(-.005/.01)*(pooled-activation)
    request = cal['rest'] + .02*cal['moment'] @ activation
    tendon = .2*np.tanh(activation[cal['ltm']]/10)
    return activation, request, tendon


def _raw_tensor_packet(network, read_errors=()):
    result = {'tick': int(network.tick), 'read_errors': list(read_errors)}
    for key in ('rate', 'adaptation', 'drive', 'output_mask'):
        try:
            tensor = getattr(network, key)
            result[key] = tensor.detach().cpu().numpy().copy()
        except Exception as error:  # noqa: BLE001 - preserve other raw fields
            result[key] = np.full(network.n, np.nan, np.float32)
            result['read_errors'].append(key+': '+repr(error))
    return result


def raw_neural(network):
    """Read tensors directly, including nonfinite values rejected by restore."""
    return _raw_tensor_packet(network)


def accounted_control(network, drive, steps, cal, activation, ledger, owner, rows):
    """Charge tick delta even when post-advance validation raises."""
    before = int(network.tick)
    advance_error = None
    try:
        network.advance(drive, steps=steps)
    except Exception as error:  # noqa: BLE001 - inspect native tick after failure
        advance_error = error
    delta = int(network.tick)-before
    observation_error = None
    if delta > 0:
        ledger['logical_steps'] += delta
        ledger['logical_neural_s'] = ledger['logical_steps']*network.dt
        ledger['arms'][owner] = ledger['arms'].get(owner, 0)+delta
        try:
            try:
                raw = raw_neural(network)
            except Exception as error:  # noqa: BLE001 - retain primary and salvage fields
                observation_error = error
                raw = _raw_tensor_packet(network, ('Raw observation failed: '+repr(error),))
            row = {**raw, 'motor_rate': raw['rate'][cal['motor_ids']].copy(),
                   'activation': np.asarray(activation, np.float64).copy(),
                   'request': np.full(42, np.nan),
                   'tendon': np.full(6, np.nan), 'decoded': False}
            rows.append(row)  # Preserve the state before any validation or decoder.
            if raw['read_errors']:
                ledger.setdefault('observation_errors', []).append({
                    'owner': owner, 'tick': int(network.tick),
                    'errors': raw['read_errors'].copy()})
        except Exception as error:  # Keep the native failure primary.
            ledger.setdefault('observation_errors', []).append({
                'owner': owner, 'tick': int(network.tick), 'errors': [repr(error)]})
            if advance_error is not None:
                advance_error.add_note('Secondary raw observation failure: '+repr(error))
                raise advance_error from error
            raise
    if advance_error is not None:
        if observation_error is not None:
            advance_error.add_note('Secondary raw observation failure: '+repr(observation_error))
        elif delta > 0 and raw['read_errors']:
            advance_error.add_note('Secondary raw tensor reads: '+'; '.join(raw['read_errors']))
        raise advance_error
    if observation_error is not None:
        raise observation_error
    if delta != steps:
        raise ValueError('Successful advance must account all native ticks')
    # snapshot validates rates only; validate the remaining raw state explicitly.
    # All rejected values are already retained in the row above.
    network.snapshot()
    if row['read_errors']:
        raise RuntimeError('Raw neural tensor read failed')
    if (any(row[key].shape != (network.n,) or not np.isfinite(row[key]).all()
            for key in ('rate', 'adaptation', 'drive', 'output_mask'))
            or np.any(row['rate'] < -1e-4) or np.any(row['adaptation'] < 0)
            or not np.isin(row['output_mask'], (0, 1)).all()):
        raise ValueError('Invalid raw complete neural state before decoder')
    next_activation, request, tendon = decode(cal, activation, row['rate'])
    row.update(activation=next_activation.copy(), request=request.copy(),
               tendon=tendon.copy(), decoded=True)
    ledger['decoder_controls'] += 1
    return next_activation, row


def save_model_configuration(path, network, source_weights, parameters,
                             source_gains, motor_ids=(), motor_gain=None):
    """Save actual full physiology and CSR fingerprints, then validate it."""
    if path.exists():
        raise FileExistsError('Model configuration raw already exists')
    fields = {key: getattr(network, key).detach().cpu().numpy().copy()
              for key in ('tau', 'a', 'threshold', 'cap', '_cell_gain')}
    actual_csr = {key: value.detach().cpu().numpy().copy() for key, value in (
        ('indptr', network.weights.crow_indices()),
        ('indices', network.weights.col_indices()),
        ('values', network.weights.values()))}
    expected_csr = {'indptr': source_weights.indptr,
                    'indices': source_weights.indices,
                    'values': source_weights.data}
    digests = {key+'_sha256': np.asarray(hashlib.sha256(
        np.ascontiguousarray(value).tobytes()).hexdigest())
        for key, value in actual_csr.items()}
    np.savez_compressed(path, **fields, **digests,
        model_hash=np.asarray(network.identity),
        source_core_identity=np.asarray(getattr(network, 'source_core_identity',
                                                network.identity)),
        selected_ids=np.asarray(motor_ids, dtype=np.int64),
        selected_gain=np.asarray([] if motor_gain is None else [motor_gain],
                                 dtype=np.float32))
    for key, value in actual_csr.items():
        if not np.array_equal(value, expected_csr[key]):
            raise ValueError('Actual full CSR '+key+' changed')
    for key, expected in zip(('tau', 'a', 'threshold', 'cap'), parameters):
        if not np.array_equal(fields[key], expected):
            raise ValueError('Actual full '+key+' physiology changed')
    expected = np.asarray(source_gains, dtype=np.float32).copy()
    if motor_gain is not None:
        expected[np.asarray(motor_ids, dtype=np.int64)] = np.float32(motor_gain)
    if not np.array_equal(fields['_cell_gain'], expected):
        raise ValueError('Actual full per-cell adaptation gain changed')
    if (network.dt != .00025 or network.adaptation_gain != 8.
            or network.adaptation_tau_s != .3):
        raise ValueError('Original dt and adaptation physiology changed')
    return {key: str(value) for key, value in digests.items()}


def save_arm_raw(path, arm, rows, network, cal, held_drive, original):
    """Persist all observations already computed, even for an incomplete arm."""
    if path.exists():
        raise FileExistsError('Raw arm output already exists: '+str(path))
    values = {'arm': np.asarray(arm),
        'graph_hash': np.asarray(original['graph_hash']),
        'model_hash': np.asarray(network.identity),
        'source_core_identity': np.asarray(getattr(network, 'source_core_identity',
                                                   network.identity)),
        'motor_ids': np.asarray(cal['motor_ids']),
        'gain_ids': np.asarray(getattr(network, 'motor_ids', np.empty(0, np.int64))),
        'gains': np.asarray(getattr(network, 'motor_gains', np.empty(0, np.float32))),
        'held_drive': np.asarray(held_drive),
        'initial_rate': np.asarray(original['network']['rate'], dtype=np.float32),
        'initial_adaptation': np.asarray(original['network']['adaptation'], dtype=np.float32),
        'initial_output_mask': np.asarray(original['network']['output_mask'], dtype=np.float32),
        'initial_tick': np.asarray(original['network']['tick'], dtype=np.int64),
        'initial_activation': np.asarray(original['adapter']['activation'], dtype=np.float64),
        'ticks': np.asarray([row['tick'] for row in rows], dtype=np.int64),
        'rate': np.asarray([row['rate'] for row in rows], dtype=np.float32),
        'adaptation': np.asarray([row['adaptation'] for row in rows], dtype=np.float32),
        'drive': np.asarray([row['drive'] for row in rows], dtype=np.float32),
        'output_mask': np.asarray([row['output_mask'] for row in rows], dtype=np.float32),
        'motor_rate': np.asarray([row['motor_rate'] for row in rows], dtype=np.float32),
        'activation': np.asarray([row['activation'] for row in rows], dtype=np.float64),
        'request': np.asarray([row['request'] for row in rows], dtype=np.float64),
        'tendon': np.asarray([row['tendon'] for row in rows], dtype=np.float64),
        'decoded': np.asarray([row['decoded'] for row in rows], dtype=bool),
        'raw_read_errors': np.asarray(['; '.join(row['read_errors']) for row in rows])}
    np.savez_compressed(path, **values)


def exact_arm(left, right, name):
    for key in ('ticks', 'rate', 'adaptation', 'drive', 'output_mask',
                'motor_rate', 'activation', 'request', 'tendon'):
        exact(left[key], right[key], name+' '+key)


def configure_one_cpu_thread():
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    limits = threadpool_limits(limits=1)
    observed = threadpool_info()
    if (torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
            or any(item['num_threads'] != 1 for item in observed)):
        limits.restore_original_limits()
        raise RuntimeError('Observed CPU/BLAS thread count must be one')
    return limits, {'torch_intraop': torch.get_num_threads(),
                    'torch_interop': torch.get_num_interop_threads(),
                    'threadpools': observed}


def raw_origin(graph_hash, network, activation):
    return {'graph_hash': graph_hash, 'network': network,
            'adapter': {'activation': np.asarray(activation).copy()}}


def require_room(began, label, now=None):
    elapsed = (time.perf_counter() if now is None else now)-began
    if elapsed + RESERVE_S >= INTERNAL_LIMIT_S:
        raise TimeoutError(label+' needs 8 s persistence/cleanup reserve')


def run(freeze_path, output):
    began = time.perf_counter()
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve single E123 original output: '+str(output))
    output.mkdir(parents=True, exist_ok=False)
    ledger = {'logical_steps': 0, 'logical_neural_s': 0., 'physical_steps': 0,
              'decoder_controls': 0, 'arms': {}, 'setup_scientific_steps': 0,
              'model_attempts': [], 'configuration_sha256': {},
              'cuda_setup_model_count': 0, 'cuda_setup_kernel_steps': 0}
    secondary = []
    primary = None
    result = None
    thread_limits = None
    active_name = None
    active_rows = []
    active_network = None
    active_origin = None
    graph = cal = original = held = None
    network = positive = clone = None

    def room(label):
        require_room(began, label)

    def persist_active():
        nonlocal active_name, active_rows, active_network, active_origin
        if active_name is not None and active_network is not None:
            saved = False
            try:
                save_arm_raw(output/(active_name+'-raw.npz'), active_name,
                             active_rows, active_network, cal, held, active_origin)
                saved = True
            except Exception as error:  # noqa: BLE001 - preserve first failure and raw
                secondary.append({'stage': active_name+' raw save',
                                  'error': type(error).__name__, 'message': str(error)})
                try:
                    save_arm_raw(output/(active_name+'-raw-emergency.npz'),
                                 active_name, active_rows, active_network, cal, held,
                                 active_origin)
                    saved = True
                except Exception as fallback_error:  # noqa: BLE001 - report both
                    secondary.append({'stage': active_name+' emergency raw save',
                                      'error': type(fallback_error).__name__,
                                      'message': str(fallback_error)})
            if saved:
                active_name, active_rows, active_network, active_origin = None, [], None, None

    def start_model(owner, motor_ids=None, gain=None):
        room(owner+' model initialization')
        attempt = {'owner': owner, 'stage': 'attempted',
                   'started_wall_s': time.perf_counter()-began,
                   'cuda_setup_kernel_steps_completed': 0,
                   'partial_constructor_work_unmeasured': True}
        ledger['model_attempts'].append(attempt)
        try:
            candidate = make_network(weights, parameters, original,
                                     motor_ids, gain, setup_record=attempt)
            attempt['partial_constructor_work_unmeasured'] = False
            return candidate
        finally:
            attempt['finished_wall_s'] = time.perf_counter()-began
            if attempt['stage'] != 'attempted':
                attempt['partial_constructor_work_unmeasured'] = False
            ledger['cuda_setup_model_count'] = sum(
                item['cuda_setup_kernel_steps_completed'] == 60
                for item in ledger['model_attempts'])
            ledger['cuda_setup_kernel_steps'] = sum(
                item['cuda_setup_kernel_steps_completed']
                for item in ledger['model_attempts'])

    try:
        thread_limits, ledger['observed_cpu_threads'] = configure_one_cpu_thread()
        freeze = reviewed_inputs(freeze_path)
        room('full graph preparation')
        graph, cal, _metadata, _anatomy, original = native_graph_and_inputs(freeze)
        weights = reconstruct_weights(graph, cal)
        parameters = native_parameters(graph, cal)
        source_gains = np.full(graph.n, 8., np.float32)
        source_gains[original['network']['core_adaptation']['ids']] = 0.
        held = np.asarray(original['network']['drive'], dtype=np.float32).copy()
        if (not np.isfinite(held).all()
                or not np.isfinite(original['network']['rate']).all()
                or not np.isfinite(original['network']['adaptation']).all()):
            raise ValueError('Finite saved full neural state required')
        write_json(output/'spec.json', {
            'schema': 'flylab.m3-e123-held-neural-diagnostic.v1',
            'freeze_sha256': FREEZE_SHA, 'design_sha256': DESIGN_SHA,
            'runner_sha256': sha256(__file__),
            'network_tool_sha256': sha256(ROOT/'tools/banc_m3_motor_adaptation.py'),
            'full_neurons': graph.n, 'original_network_identity': CORE_IDENTITY,
            'held_input': True, 'closed_loop': False,
            'cpu_threads': 1, 'internal_limit_s': 45.,
            'child_limit_s': 55., 'external_limit_s': 60., 'reserve_s': 8.,
            'candidate_adopted': False, 'F6_exposed': False,
            'whole_goal_complete': False})
        if time.perf_counter()-began >= INTERNAL_LIMIT_S:
            raise TimeoutError('Preparation exceeded45 s internal limit')

        for name, gain, controls in (
                ('original_core', None, 10),
                ('zero_motor_gain', 0., 10),
                ('positive_motor_gain', 1., 10)):
            network = start_model(name,
                None if gain is None else cal['motor_ids'], gain)
            active_name, active_network, active_rows, active_origin = (
                name, network, [], original)
            ledger['configuration_sha256'][name] = save_model_configuration(
                output/(name+'-configuration.npz'), network, weights, parameters,
                source_gains, () if gain is None else cal['motor_ids'], gain)
            activation = np.asarray(original['adapter']['activation'],
                                    dtype=np.float64).copy()
            ledger['arms'][name] = 0
            for _ in range(controls):
                room(name+' control')
                activation, _ = accounted_control(network, held, 20, cal,
                    activation, ledger, name, active_rows)
            persist_active()
            if secondary:
                raise RuntimeError('Raw persistence failed before comparisons')
            if name == 'positive_motor_gain':
                positive = network
                positive_activation = activation.copy()
            else:
                del network

        with np.load(output/'original_core-raw.npz', allow_pickle=False) as source:
            baseline = {key: source[key].copy() for key in source.files}
        with np.load(output/'zero_motor_gain-raw.npz', allow_pickle=False) as source:
            zero = {key: source[key].copy() for key in source.files}
        with np.load(output/'positive_motor_gain-raw.npz', allow_pickle=False) as source:
            changed = {key: source[key].copy() for key in source.files}
        exact_arm(baseline, zero, 'Original core versus gain0 full state/decoder')
        if (not np.isfinite(changed['adaptation']).all()
                or not np.any(changed['adaptation'][-1, cal['motor_ids']]
                              > baseline['adaptation'][-1, cal['motor_ids']])
                or np.array_equal(changed['rate'], baseline['rate'])
                or np.array_equal(changed['request'], baseline['request'])):
            raise ValueError('Finite actual MN adaptation/rate/request effect required')
        if not (np.array_equal(positive._cell_gain.cpu().numpy()[cal['motor_ids']],
                               np.ones(391, np.float32))):
            raise ValueError('Actual selected CUDA gain vector changed')
        room('positive checkpoint')
        checkpoint = {'schema': 'flylab.m3-e123-held-checkpoint.v1',
                      'graph_hash': graph.hash, 'calibration_sha256':
                      sha256(ROOT/freeze['source_e121']/'calibration.npz'),
                      'calibration': {key: value.copy() for key, value in cal.items()},
                      'held_drive': held.copy(),
                      'activation': positive_activation.copy(),
                      'network': positive.snapshot()}
        StateStore.save(output/'positive-checkpoint', checkpoint)
        disk = StateStore.load(output/'positive-checkpoint')
        exact(checkpoint, disk, 'Positive full neural/muscle disk checkpoint')
        exact(cal, disk['calibration'], 'Disk-restored full original calibration')
        room('positive main future')
        active_name, active_network, active_rows, active_origin = (
            'positive-main-future', positive, [], raw_origin(
                graph.hash, checkpoint['network'], checkpoint['activation']))
        main_activation, main_row = accounted_control(positive, held, 20, cal,
            positive_activation, ledger, 'positive_main_future', active_rows)
        persist_active()  # Independent main future is durable before clone init.
        if secondary:
            raise RuntimeError('Main future raw persistence failed')
        main_state = positive.snapshot()
        clone = start_model('positive_disk_clone', cal['motor_ids'], 1.)
        if (disk['schema'] != checkpoint['schema'] or disk['graph_hash'] != graph.hash
                or disk['calibration_sha256'] != checkpoint['calibration_sha256']
                or not np.array_equal(disk['held_drive'], held)):
            raise ValueError('Disk clone source identity changed')
        clone.restore(disk['network'])
        ledger['configuration_sha256']['positive_disk_clone'] = save_model_configuration(
            output/'positive-disk-clone-configuration.npz', clone, weights,
            parameters, source_gains, cal['motor_ids'], 1.)
        active_name, active_network, active_rows, active_origin = (
            'positive-clone-future', clone, [], raw_origin(
                graph.hash, disk['network'], disk['activation']))
        room('positive disk clone future')
        clone_activation, clone_row = accounted_control(clone, held, 20,
            disk['calibration'], disk['activation'].copy(), ledger,
            'positive_clone_future', active_rows)
        persist_active()
        if secondary:
            raise RuntimeError('Clone future raw persistence failed')
        clone_state = clone.snapshot()
        exact(main_state, clone_state, 'Positive disk clone full neural future')
        exact(main_activation, clone_activation, 'Positive disk clone muscle future')
        exact(main_row['request'], clone_row['request'], 'Positive disk clone request future')
        exact(main_row['tendon'], clone_row['tendon'], 'Positive disk clone tendon future')
        if (ledger['logical_steps'] != 640 or ledger['decoder_controls'] != 32
                or ledger['logical_neural_s'] != .16 or ledger['physical_steps'] != 0):
            raise ValueError('Fixed E123 logical neural/decoder budget changed')
        if time.perf_counter()-began > INTERNAL_LIMIT_S:
            raise TimeoutError('E123 internal45 s limit exceeded')
        result = {'validity': 'valid_full_BANC_held_neural_diagnostic',
                  'gain0_bit_exact': True, 'positive_gain_actual_effect': True,
                  'disk_future_bit_exact': True, 'candidate_adopted': False,
                  'F6_exposed': False, 'whole_goal_complete': False,
                  'actual_neural_s': ledger['logical_neural_s'],
                  'actual_neural_steps': ledger['logical_steps'],
                  'actual_physical_s': 0., 'decoder_controls': 32,
                  'wall_s': None}
    except Exception as error:  # noqa: BLE001 - retain first execution failure
        primary = error
    finally:
        try:
            persist_active()
        except Exception as error:  # noqa: BLE001 - retain execution exception
            secondary.append({'stage': 'partial raw cleanup',
                              'error': type(error).__name__, 'message': str(error)})
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            active_network = network = positive = clone = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as error:  # noqa: BLE001 - report cleanup separately
            secondary.append({'stage': 'CUDA synchronize/resource cleanup',
                              'error': type(error).__name__, 'message': str(error)})
        if thread_limits is not None:
            try:
                thread_limits.restore_original_limits()
            except Exception as error:  # noqa: BLE001 - report cleanup separately
                secondary.append({'stage': 'CPU thread restoration',
                                  'error': type(error).__name__, 'message': str(error)})
        ledger['wall_s_before_final_receipts'] = time.perf_counter()-began
        try:
            write_json(output/'advance-ledger.json', ledger)
        except Exception as error:  # noqa: BLE001 - keep first execution failure
            secondary.append({'stage': 'ledger save', 'error': type(error).__name__,
                              'message': str(error)})
        if primary is None and result is None:
            primary = RuntimeError('E123 terminated without complete result')
        if primary is None and secondary:
            primary = RuntimeError('E123 raw/cleanup/ledger persistence failed')
        if primary is None and time.perf_counter()-began > INTERNAL_LIMIT_S:
            primary = TimeoutError('E123 finalization exceeded internal45 s limit')
        if primary is None:
            result['wall_s'] = time.perf_counter()-began
            try:
                write_json(output/'result.json', result)
            except Exception as error:  # noqa: BLE001 - preserve primary failure
                primary = error
            if primary is None and time.perf_counter()-began > INTERNAL_LIMIT_S:
                primary = TimeoutError('Final result save exceeded internal45 s limit')
        if primary is not None:
            if result is not None and (output/'result.json').exists():
                result['validity'] = 'resource_limit_incomplete' if isinstance(
                    primary, TimeoutError) else 'incomplete'
                result['wall_s'] = time.perf_counter()-began
                try:
                    write_json(output/'result.json', result)
                except Exception as error:  # noqa: BLE001 - retain primary
                    secondary.append({'stage': 'invalidate result',
                                      'error': type(error).__name__,
                                      'message': str(error)})
            try:
                write_json(output/'execution-error.json', {
                    'primary': {'error': type(primary).__name__,
                                'message': str(primary),
                                'notes': getattr(primary, '__notes__', [])},
                    'secondary': secondary,
                    'actual_neural_steps': ledger['logical_steps'],
                    'actual_neural_s': ledger['logical_neural_s'],
                    'actual_physical_s': 0., 'wall_s': time.perf_counter()-began,
                    'candidate_adopted': False, 'F6_exposed': False,
                    'whole_goal_complete': False})
            except Exception as error:  # noqa: BLE001 - retain primary
                primary.add_note('Execution receipt failed: '+repr(error))
    if primary is not None:
        for item in secondary:
            primary.add_note('Secondary '+item['stage']+': '+item['message'])
        raise primary
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, default=FREEZE)
    parser.add_argument('--output', type=Path,
        default=ROOT/'verification/banc-natural-gait-m3-20260919/e123-motor-adaptation')
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.freeze, arguments.output)), flush=True)
