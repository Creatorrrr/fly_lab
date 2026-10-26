"""E126: one reviewed 10 s native stage-input collection per original context.

Only the parent invokes original BANC/body responses. This runner never fits or
substitutes a conditional response for whole-circuit native dynamics.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.verify_c_banc_m3_native_current import (
    AdvanceLedger,
    configure_single_thread_runtime,
    error_record,
    file_sha,
    save_json,
)

SOURCE_FILES = (
    'tools/banc_m3_stage_dataset.py',
    'tools/verify_c_banc_m3_stage_dataset.py',
    'tests/test_c_banc_stage_dataset.py',
)
DESIGN = 'docs/BANC_M3_E126_STAGE_DATASET_DESIGN_20260921.md'
SCHEMA = 'flylab.m3.e126.stage-dataset-freeze.v1'
CONTEXTS = ('E110', 'E113', 'E121')
CONTROLS = 2000
STAGES = 160000
DT = .00025


def repository_path(relative, *, file=False):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('Repository-relative slash path required')
    name = Path(relative)
    if name.is_absolute() or '..' in name.parts or '.' in name.parts:
        raise ValueError('Unsafe repository-relative path: '+relative)
    path = ROOT/name
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Escaped repository-relative path: '+relative)
    if file and not path.is_file() or not file and not path.exists():
        raise ValueError('Missing frozen repository path: '+relative)
    return path


def verify_contract(freeze_path, review_path, context, *, freeze_sha=None,
                    review_sha=None, elapsed=None):
    """Refuse unreviewed, modified, or incomplete original inputs pre-model."""
    freeze_path, review_path = Path(freeze_path), Path(review_path)
    if context not in CONTEXTS:
        raise ValueError('One frozen original context E110/E113/E121 required')
    if freeze_sha is not None and file_sha(freeze_path) != freeze_sha:
        raise ValueError('Explicit E126 freeze SHA changed')
    if review_sha is not None and file_sha(review_path) != review_sha:
        raise ValueError('Explicit E126 review SHA changed')
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    review = json.loads(review_path.read_text(encoding='utf-8'))
    constants = {'schema': SCHEMA, 'design_path': DESIGN,
        'graph_n': 158706, 'graph_edges': 11584852,
        'graph_hash': '9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff',
        'controls': CONTROLS, 'neural_dt_s': DT, 'control_dt_s': .005,
        'physical_dt_s': .0001, 'capacity_stages': STAGES,
        'motor_count': 391, 'source_count': 69, 'edge_count': 2444,
        'fit_edge_count': 1572, 'raw_array_bytes_per_context': 294400000,
        'max_observation_bytes': 512*1024**2, 'future_control': 220,
        'internal_wall_limit_s': 150, 'finalize_reserve_s': 20,
        'external_wall_limit_s': 180, 'supervisor_limit_s': 185,
        'prior_conservative_process_s': 975,
        'shared_process_limit_s': 1530, 'processes': 3,
        'actual_neural_s_per_context': 10.,
        'actual_physical_s_per_context': 10.,
        'automatic_retries': 0, 'fit_calls': 0,
        'F6_exposed': False, 'candidate_adopted': False,
        'input_count': 289, 'reused_source_count': 24,
        'preservation_count': 1103}
    for key, value in constants.items():
        if freeze.get(key) != value or type(freeze[key]) is not type(value):
            raise ValueError('E126 frozen constant changed: '+key)
    if file_sha(repository_path(DESIGN, file=True)) != freeze.get('design_sha256'):
        raise ValueError('E126 design SHA changed')
    paths = ('graph_path', 'fixture_path', 'source_provenance_path',
             'E125_prefix_current_path')
    for key in paths:
        repository_path(freeze[key], file=key != 'graph_path')
    contexts = freeze.get('contexts')
    if (not isinstance(contexts, list) or len(contexts) != 3
            or [item.get('name') for item in contexts] != list(CONTEXTS)):
        raise ValueError('Exactly three ordered original contexts required')
    chosen = None
    for item in contexts:
        if not isinstance(item.get('neural_identity'), str) or len(item['neural_identity']) != 64:
            raise ValueError('Frozen context neural identity required')
        path_map = item.get('paths')
        if not isinstance(path_map, dict) or set(path_map) != {
                'start', 'future', 'final', 'calibration',
                'metadata', 'signals', 'trace'}:
            raise ValueError('Exact original context paths required')
        for key, relative in path_map.items():
            repository_path(relative, file=key not in ('start', 'future', 'final'))
        if item['name'] == context:
            chosen = item
    groups = (('input_sha256', 'input_count'),
              ('reused_source_sha256', 'reused_source_count'),
              ('preservation_sha256', 'preservation_count'))
    for label, count_key in groups:
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != freeze[count_key]:
            raise ValueError('Frozen E126 '+label+' cardinality changed')
        for relative, expected in hashes.items():
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError('Invalid E126 SHA entry')
            if file_sha(repository_path(relative, file=True)) != expected:
                raise ValueError('Frozen E126 file changed: '+relative)
            if elapsed is not None and elapsed() >= 130:
                raise TimeoutError('E126 preflight consumed finalization reserve')
    if (review.get('accepted_for_single_original_stage_dataset') is not True
            or review.get('blocking_defects') != []
            or review.get('freeze_sha256') != file_sha(freeze_path)
            or review.get('source_sha256') !=
                {name: file_sha(repository_path(name, file=True))
                 for name in SOURCE_FILES}):
        raise ValueError('Parent source review does not authorize these E126 sources')
    return freeze, chosen, review


def load_fixture(path):
    """Own copies of the frozen mapping; no GraphStore or response here."""
    with np.load(path, allow_pickle=False) as data:
        fixture = {name: data[name].copy() for name in data.files}
    names = set(fixture)
    expected = {'context_names', 'motor_ids', 'source_ids', 'edge_ids',
        'edge_pre', 'edge_post', 'edge_motor_column', 'edge_source_column',
        'calibration_positions', 'edge_values_by_context', 'fit_mask',
        'fit_edge_ids', 'initial_motor_rate', 'initial_motor_adaptation',
        'initial_source_masked_rate', 'pool', 'moment', 'rest', 'ltm'}
    if names != expected:
        raise ValueError('Frozen E126 fixture fields changed')
    shapes = {'context_names': (3,), 'motor_ids': (391,),
        'source_ids': (69,), 'edge_ids': (2444,), 'edge_pre': (2444,),
        'edge_post': (2444,), 'edge_motor_column': (2444,),
        'edge_source_column': (2444,), 'calibration_positions': (2444,),
        'edge_values_by_context': (3, 2444), 'fit_mask': (2444,),
        'fit_edge_ids': (1572,), 'initial_motor_rate': (3, 391),
        'initial_motor_adaptation': (3, 391),
        'initial_source_masked_rate': (3, 69),
        'pool': (96, 391), 'moment': (42, 96), 'rest': (42,),
        'ltm': (6,)}
    if any(fixture[name].shape != shape for name, shape in shapes.items()):
        raise ValueError('Frozen E126 fixture dimensions changed')
    if fixture['context_names'].tolist() != list(CONTEXTS):
        raise ValueError('Frozen E126 context order changed')
    for key in ('motor_ids', 'source_ids', 'edge_ids', 'edge_pre', 'edge_post',
                'edge_motor_column', 'edge_source_column',
                'calibration_positions', 'fit_edge_ids', 'ltm'):
        if fixture[key].dtype != np.dtype(np.int64):
            raise ValueError('Frozen E126 integer field changed: '+key)
    if (len(np.unique(fixture['motor_ids'])) != 391
            or len(np.unique(fixture['source_ids'])) != 69
            or len(np.unique(fixture['edge_ids'])) != 2444
            or 83877 not in fixture['source_ids']):
        raise ValueError('Frozen E126 existing source/MN/edge roster changed')
    bounded = {'motor_ids': 158706, 'source_ids': 158706,
        'edge_ids': 11584852, 'edge_pre': 158706, 'edge_post': 158706,
        'edge_motor_column': 391, 'edge_source_column': 69,
        'calibration_positions': 130943, 'fit_edge_ids': 11584852,
        'ltm': 96}
    for name, maximum in bounded.items():
        if np.any((fixture[name] < 0) | (fixture[name] >= maximum)):
            raise ValueError('Frozen E126 index outside original '+name)
    if (not np.array_equal(fixture['motor_ids'][fixture['edge_motor_column']],
                           fixture['edge_post'])
            or not np.array_equal(
                fixture['source_ids'][fixture['edge_source_column']],
                fixture['edge_pre'])):
        raise ValueError('Frozen E126 edge/source/MN mapping changed')
    for name in ('edge_values_by_context', 'initial_motor_rate',
                 'initial_motor_adaptation', 'initial_source_masked_rate',
                 'pool', 'moment', 'rest'):
        if not np.isfinite(fixture[name]).all():
            raise ValueError('Nonfinite frozen E126 fixture '+name)
    if fixture['fit_mask'].dtype != np.bool_ or fixture['fit_mask'].sum() != 1572:
        raise ValueError('Frozen E126 fit mask changed')
    if (len(np.unique(fixture['fit_edge_ids'])) != 1572
            or not np.array_equal(np.sort(fixture['edge_ids'][fixture['fit_mask']]),
                                  np.sort(fixture['fit_edge_ids']))):
        raise ValueError('Frozen E126 fit edge IDs changed')
    return fixture


def save_stage_arrays(out, observer, errors):
    """Try each GPU readback and .npy write even if its sibling fails."""
    out = Path(out)
    metadata = None
    try:
        observer.disable_observation()
    except BaseException as error:  # noqa: BLE001 - retain raw sibling after CUDA failure
        errors.append(error_record('stage-disable', error))
    try:
        metadata = observer.observation_static_metadata()
    except BaseException as error:  # noqa: BLE001 - preserve partial static metadata
        errors.append(error_record('stage-static-metadata', error))
        metadata = {'start_tick': None, 'end_tick': None}
    for name, getter in (
        ('current_counter_requested', observer.copy_current_counters),
        ('source_counter_requested', observer.copy_source_counters),
        ('current_overflow', observer.copy_current_overflow),
        ('source_overflow', observer.copy_source_overflow),
        ('armed', observer.copy_armed),
    ):
        try:
            metadata[name] = getter()
        except BaseException as error:  # noqa: BLE001 - independent GPU metadata readbacks
            errors.append(error_record(name, error))
            metadata[name] = None
    for prefix in ('current', 'source'):
        counts = metadata.get(prefix+'_counter_requested')
        metadata[prefix+'_counter_completed'] = (
            np.minimum(counts, metadata['capacity_stages'])
            if counts is not None and 'capacity_stages' in metadata else None)
    try:
        save_json(out/'stage-metadata.json', metadata)
    except BaseException as error:  # noqa: BLE001 - raw metadata save is independent
        errors.append(error_record('stage-metadata.json', error))

    results = {}
    for name, getter, filename in (
        ('current', observer.copy_current_values, 'motor-total-current.npy'),
        ('source', observer.copy_source_values, 'source-masked-rate.npy'),
    ):
        try:
            values = getter()
        except BaseException as error:  # noqa: BLE001 - try sibling hostcopy
            errors.append(error_record(name+'-hostcopy', error))
            results[name] = False
            continue
        try:
            np.save(out/filename, values, allow_pickle=False)
            results[name] = True
        except BaseException as error:  # noqa: BLE001 - preserve copied buffer
            errors.append(error_record(filename, error))
            results[name] = False
            try:
                np.save(out/(name+'-stage-rescue.npy'), values,
                        allow_pickle=False)
            except BaseException as rescue:  # noqa: BLE001 - try sibling save
                errors.append(error_record(name+'-stage-rescue.npy', rescue))
    return metadata, results


def replace_with_stage_network(session, fixture):
    from tools.banc_m3_stage_dataset import StageInputNetwork
    from tools.probe_c_banc_m3_core_rhythm import source_matrix
    from tools.verify_c_banc_observation import exact

    old = session.network
    saved = old.snapshot()
    weights = source_matrix(old)
    params = [getattr(old, name).cpu().numpy().copy()
              for name in ('tau', 'a', 'threshold', 'cap')]
    candidate = StageInputNetwork(weights, *params,
        adaptation_gain=old.adaptation_gain,
        adaptation_tau_s=old.adaptation_tau_s,
        dt=old.dt, device=old.device,
        cuda_implementation=old.cuda_implementation,
        capture_steps=old.interval_steps, core_ids=old.core_ids,
        core_gain=old.core_gain, motor_ids=fixture['motor_ids'],
        source_ids=fixture['source_ids'], capacity_stages=STAGES)
    exact(old.identity, candidate.identity, 'E126 native neural identity')
    candidate.restore(saved)
    exact(saved, candidate.snapshot(), 'E126 native neural state')
    session.network = candidate
    del old
    return candidate


def validate_fixture_against_original(fixture, context_index, graph, session):
    from tools.probe_c_banc_m3_core_rhythm import source_matrix
    from tools.verify_c_banc_observation import exact

    ids = fixture['edge_ids']
    exact(graph.indices[ids].astype(np.int64), fixture['edge_pre'],
          'E126 existing edge source rows')
    exact(np.searchsorted(graph.indptr[1:], ids, side='right').astype(np.int64),
          fixture['edge_post'], 'E126 existing receiving rows')
    exact(fixture['motor_ids'][fixture['edge_motor_column']],
          fixture['edge_post'], 'E126 MN receiving columns')
    exact(fixture['source_ids'][fixture['edge_source_column']],
          fixture['edge_pre'], 'E126 source columns')
    matrix = source_matrix(session.network)
    exact(matrix.data[ids], fixture['edge_values_by_context'][context_index],
          'E126 actual original selected synapses')
    exact(session.network.rate.cpu().numpy()[fixture['motor_ids']],
          fixture['initial_motor_rate'][context_index], 'E126 actual initial MN rates')
    exact(session.network.adaptation.cpu().numpy()[fixture['motor_ids']],
          fixture['initial_motor_adaptation'][context_index],
          'E126 actual initial MN adaptation')
    actual_source = (session.network.rate.cpu().numpy()[fixture['source_ids']]
                     * session.network.output_mask.cpu().numpy()[fixture['source_ids']])
    exact(actual_source, fixture['initial_source_masked_rate'][context_index],
          'E126 actual source masked initial rates')
    for key in ('pool', 'moment', 'rest', 'ltm'):
        exact(session.adapter.calibration[key], fixture[key],
              'E126 original anatomical '+key)


def _finite_npy(path, expected_shape, expected_dtype):
    data = np.load(path, mmap_mode='r', allow_pickle=False)
    if data.shape != expected_shape or data.dtype != expected_dtype:
        raise AssertionError('E126 raw stage array shape/dtype: '+str(path))
    for first in range(0, len(data), 4096):
        if not np.isfinite(data[first:first+4096]).all():
            raise AssertionError('E126 nonfinite raw stage array: '+str(path))
    return data


def compare_after_raw(freeze, chosen, context, out, fixture, rows,
                      motor_rate, activation, source_masked,
                      future_snapshot, final_snapshot, metadata, saved_arrays):
    from flylab.c.storage import StateStore
    from tools.verify_c_banc_observation import exact

    paths = {key: repository_path(value, file=key not in ('start', 'future', 'final'))
             for key, value in chosen['paths'].items()}
    if (len(rows) != CONTROLS+1 or motor_rate.shape != (CONTROLS, 391)
            or activation.shape != (CONTROLS, 96)
            or source_masked.shape != (CONTROLS+1, 69)
            or future_snapshot is None or final_snapshot is None
            or saved_arrays != {'current': True, 'source': True}):
        raise AssertionError('Complete E126 raw response required for parity')
    if (metadata['start_tick'] != 0 or metadata['end_tick'] != 40000
            or metadata['motor_ids'].tolist() != fixture['motor_ids'].tolist()
            or metadata['source_ids'].tolist() != fixture['source_ids'].tolist()
            or metadata['capacity_stages'] != STAGES
            or any(metadata[name] is None for name in (
                'current_counter_requested', 'source_counter_requested',
                'current_overflow', 'source_overflow', 'armed'))
            or not np.all(metadata['current_counter_requested'] == STAGES)
            or not np.all(metadata['source_counter_requested'] == STAGES)
            or not np.all(metadata['current_counter_completed'] == STAGES)
            or not np.all(metadata['source_counter_completed'] == STAGES)
            or metadata['current_overflow'].tolist() != [0]
            or metadata['source_overflow'].tolist() != [0]
            or metadata['armed'].tolist() != [0]):
        raise AssertionError('E126 dual native stage metadata invalid')
    current = _finite_npy(out/'motor-total-current.npy',
                          (STAGES, 391), np.dtype(np.float32))
    source = _finite_npy(out/'source-masked-rate.npy',
                         (STAGES, 69), np.dtype(np.float32))
    if current.nbytes + source.nbytes != freeze['raw_array_bytes_per_context']:
        raise AssertionError('E126 original raw stage payload bytes changed')
    exact(source[::80].copy(), source_masked[:-1],
          'E126 first stage uses preceding actual source masked endpoint')
    if context == 'E110':
        with np.load(repository_path(freeze['E125_prefix_current_path'], file=True),
                     allow_pickle=False) as e125:
            exact(current[:19200].copy(), e125['total_current'],
                  'E126 first 1.2s original E125 native current')
    exact(future_snapshot, StateStore.load(paths['future']),
          'E126 original full main-1p1 checkpoint')
    exact(final_snapshot, StateStore.load(paths['final']),
          'E126 original full 10s final checkpoint')
    original = json.loads(paths['trace'].read_text(encoding='utf-8'))
    exact(rows, original['samples'], 'E126 original 2001 control/body rows')
    with np.load(paths['signals'], allow_pickle=False) as signals:
        ids = signals['observed_ids']
        order = np.argsort(ids)
        columns = np.searchsorted(ids[order], fixture['motor_ids'])
        if (np.any(columns >= len(ids))
                or not np.array_equal(ids[order][columns], fixture['motor_ids'])):
            raise AssertionError('E126 original signal motor roster changed')
        exact(motor_rate, signals['rate'][:, order[columns]],
              'E126 original 2000 MN endpoints')
        exact(activation, signals['muscle_activation'],
              'E126 original 2000 activation endpoints')
    return {'whole_main_1p1_exact': True, 'whole_final_exact': True,
            'control_body_rows_exact': True, 'motor_rate_exact': True,
            'muscle_activation_exact': True,
            'source_first_stage_exact': True,
            'dual_stage_counts_exact': True,
            'E125_prefix_current_exact': context == 'E110'}


def run(freeze_path, review_path, context, out, *, freeze_sha=None,
        review_sha=None):
    began = time.perf_counter()
    out = Path(out)
    if out.exists():
        raise FileExistsError('One-shot E126 context output exists: '+str(out))
    out.mkdir(parents=True)
    errors = []
    primary = None
    session = observer = None
    fixture = freeze = chosen = None
    motor_ids = source_ids = tendon_names = None
    source_index = None
    rows = []
    motor_rate_rows, motor_z_rows, activation_rows = [], [], []
    source_rate_rows, source_mask_rows, source_masked_rows = [], [], []
    future_snapshot = final_snapshot = initial_snapshot = None
    metadata = None
    saved_arrays = {'current': False, 'source': False}
    comparison = None
    ledger = AdvanceLedger()
    setup = {'context': context, 'models_constructed': 0,
             'model_setup_completion': 'not_started',
             'warmup_native_ticks': 0, 'capture_native_ticks': 0,
             'actual_main_neural_ticks': None,
             'actual_main_physics_ticks': None,
             'attempted_controls': 0, 'synchronized_controls': 0,
             'recorded_control_endpoints': 0}

    def attempt(label, operation):
        try:
            return operation()
        except BaseException as error:  # noqa: BLE001 - each artifact is independent
            errors.append(error_record(label, error))
            return None

    try:
        freeze, chosen, review = verify_contract(freeze_path, review_path,
            context, freeze_sha=freeze_sha, review_sha=review_sha,
            elapsed=lambda: time.perf_counter()-began)
        paths = {key: repository_path(value, file=key not in ('start', 'future', 'final'))
                 for key, value in chosen['paths'].items()}
        save_json(out/'provenance.json', {'context': context,
            'freeze_sha256': file_sha(freeze_path),
            'source_review_sha256': file_sha(review_path),
            'design_sha256': freeze['design_sha256'],
            'source_sha256': review['source_sha256'],
            'fixture_sha256': file_sha(repository_path(freeze['fixture_path'], file=True)),
            'input_sha256': freeze['input_sha256'],
            'reused_source_sha256': freeze['reused_source_sha256'],
            'preservation_sha256': freeze['preservation_sha256']})
        reviewed = out/'reviewed-source'
        reviewed.mkdir()
        for name in SOURCE_FILES:
            copied = reviewed/Path(name).name
            shutil.copyfile(repository_path(name, file=True), copied)
            if file_sha(copied) != review['source_sha256'][name]:
                raise ValueError('Reviewed E126 source changed while copying')
        for source, name, expected in (
            (freeze_path, 'freeze.json', file_sha(freeze_path)),
            (review_path, 'source-review.json', file_sha(review_path)),
            (repository_path(DESIGN, file=True), 'design.md',
             freeze['design_sha256']),
            (repository_path(freeze['fixture_path'], file=True), 'fixture.npz',
             freeze['input_sha256'][freeze['fixture_path']])):
            copied = reviewed/name
            shutil.copyfile(source, copied)
            if file_sha(copied) != expected:
                raise ValueError('Frozen E126 input changed while copying')
        threads = configure_single_thread_runtime()
        save_json(out/'threading.json', threads)
        import torch

        from flylab.c.graph import GraphStore
        from flylab.c.storage import StateStore
        from flylab.c.walking_trace import capture_control
        from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
        from tools.verify_c_banc_observation import exact

        if not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA GPU required; no fallback')
        save_json(out/'environment.json', {'python': sys.version,
            'platform': platform.platform(), 'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(), 'pid': os.getpid(),
            'measured_threads': threads, 'actual_cuda': True})
        fixture = load_fixture(repository_path(freeze['fixture_path'], file=True))
        context_index = CONTEXTS.index(context)
        graph = GraphStore.load(repository_path(freeze['graph_path']))
        if (graph.n != freeze['graph_n'] or len(graph.indices) != freeze['graph_edges']
                or graph.hash != freeze['graph_hash']):
            raise ValueError('Frozen full BANC graph identity changed')
        start = StateStore.load(paths['start'])
        motor_ids, source_ids = fixture['motor_ids'], fixture['source_ids']
        with np.load(paths['calibration'], allow_pickle=False) as cal:
            exact(cal['motor_ids'], motor_ids, 'E126 frozen calibration MN roster')
        with np.load(paths['metadata'], allow_pickle=False) as original_metadata:
            exact(original_metadata['motor_ids'], motor_ids,
                  'E126 original signal MN roster')
        if (start['network']['tick'] != 0 or start['control_tick'] != 0
                or start['network']['model_hash'] != chosen['neural_identity']):
            raise ValueError('E126 original context start identity changed')
        setup['model_setup_completion'] = 'restore_requested_completion_unknown'
        setup['models_constructed'] = None
        setup['warmup_native_ticks'] = None
        setup['capture_native_ticks'] = None
        session = IntrinsicRecruitmentTrial.from_checkpoint(graph, start)
        setup['models_constructed'] = 4
        setup['warmup_native_ticks'] = 8*session.network.interval_steps
        setup['capture_native_ticks'] = 4*session.network.interval_steps
        setup['model_setup_completion'] = 'four_restored_models_complete'
        exact(start, session.snapshot(), 'E126 original restored whole start')
        validate_fixture_against_original(fixture, context_index, graph, session)
        setup['model_setup_completion'] = 'stage_network_requested_completion_unknown'
        observer = replace_with_stage_network(session, fixture)
        setup['models_constructed'] = 5
        setup['warmup_native_ticks'] += 2*observer.interval_steps
        setup['capture_native_ticks'] += observer.interval_steps
        setup['model_setup_completion'] = 'all_five_models_complete'
        exact(start, session.snapshot(), 'E126 observed whole start')
        tendon_names = np.asarray([
            session.adapter.groups[int(i)]['leg']+'_tarsus'
            for i in session.adapter.calibration['ltm']])
        if len(np.unique(tendon_names)) != 6:
            raise ValueError('Exactly six original LTM tendon names required')
        source_index = torch.as_tensor(source_ids, device='cuda')
        initial_snapshot = session.snapshot()
        np.savez_compressed(out/'initial-state.npz',
            rate=initial_snapshot['network']['rate'],
            adaptation=initial_snapshot['network']['adaptation'],
            drive=initial_snapshot['network']['drive'],
            output_mask=initial_snapshot['network']['output_mask'],
            tick=np.asarray(initial_snapshot['network']['tick']),
            tau=observer.tau.cpu().numpy().copy(),
            a=observer.a.cpu().numpy().copy(),
            threshold=observer.threshold.cpu().numpy().copy(),
            cap=observer.cap.cpu().numpy().copy(),
            motor_gain=observer._cell_gain.cpu().numpy()[motor_ids].copy(),
            adaptation_gain=np.asarray(observer.adaptation_gain),
            adaptation_tau_s=np.asarray(observer.adaptation_tau_s),
            neural_dt_s=np.asarray(observer.dt),
            motor_ids=motor_ids, source_ids=source_ids,
            pool=session.adapter.calibration['pool'],
            moment=session.adapter.calibration['moment'],
            rest=session.adapter.calibration['rest'],
            ltm=session.adapter.calibration['ltm'],
            tendon_names=tendon_names,
            activation=session.adapter.activation.copy())
        StateStore.save(out/'observed-start', initial_snapshot)

        def source_endpoint():
            rates = observer.rate[source_index]
            mask = observer.output_mask[source_index]
            return (rates.cpu().numpy().copy(), mask.cpu().numpy().copy(),
                    (rates*mask).cpu().numpy().copy())

        initial_source = source_endpoint()
        source_rate_rows.append(initial_source[0])
        source_mask_rows.append(initial_source[1])
        source_masked_rows.append(initial_source[2])
        exact(initial_source[2], fixture['initial_source_masked_rate'][context_index],
              'E126 frozen initial source masked rate')
        observer.arm_observation()
        rows.append(capture_control(session))
        for control in range(1, CONTROLS+1):
            if time.perf_counter()-began >= 130:
                raise TimeoutError('E126 stopped before finalization reserve')
            ledger.advance(session)
            if (observer.tick != control*20 or session.control_tick != control
                    or abs(session.body.physics_time()-control*.005) > 1e-8):
                raise ValueError('E126 original neural/body/control clocks diverged')
            rows.append(capture_control(session))
            motor_rate_rows.append(observer.readout(motor_ids))
            motor_z_rows.append(observer.adaptation[motor_ids].cpu().numpy().copy())
            activation_rows.append(session.adapter.activation.copy())
            rate, mask, masked = source_endpoint()
            source_rate_rows.append(rate)
            source_mask_rows.append(mask)
            source_masked_rows.append(masked)
            if not np.array_equal(mask, source_mask_rows[0]):
                raise ValueError('E126 source output mask changed within original control')
            if control == freeze['future_control']:
                future_snapshot = session.snapshot()
            if session.fault or session.body.fault:
                raise RuntimeError('E126 original native body health fault')
        final_snapshot = session.snapshot()
        torch.cuda.current_stream().synchronize()
    except BaseException as error:  # noqa: BLE001 - raw evidence must outlive primary
        primary = error
        errors.append(error_record('execution', error))
    finally:
        if observer is not None:
            metadata, saved_arrays = save_stage_arrays(out, observer, errors)
        attempt('advance-ledger.json', lambda: save_json(out/'advance-ledger.json',
            {'context': context, 'events': ledger.events, **ledger.totals()}))
        if session is not None:
            attempt('trace.json', lambda: save_json(out/'trace.json',
                {'schema': 'flylab.m3.e126.stage-dataset-trace.v1',
                 'context': context, 'samples': rows}))
            def save_endpoints():
                np.savez_compressed(out/'control-endpoints.npz',
                    motor_rate=np.asarray(motor_rate_rows,
                                          dtype=np.float32).reshape(-1, 391),
                    motor_adaptation=np.asarray(motor_z_rows,
                                                dtype=np.float32).reshape(-1, 391),
                    muscle_activation=np.asarray(activation_rows,
                                                 dtype=np.float64).reshape(-1, 96),
                    command=np.asarray([row['joint_targets_rad'] for row in rows[1:]],
                                       dtype=np.float64).reshape(-1, 42),
                    tendon=np.asarray([[row['tendon_inputs'][name]
                                        for name in tendon_names]
                                       for row in rows[1:]], dtype=np.float64).reshape(-1, 6),
                    tendon_names=tendon_names if tendon_names is not None
                        else np.empty(0, dtype='U1'),
                    source_rate=np.asarray(source_rate_rows,
                                           dtype=np.float32).reshape(-1, 69),
                    source_mask=np.asarray(source_mask_rows,
                                           dtype=np.float32).reshape(-1, 69),
                    source_masked_rate=np.asarray(source_masked_rows,
                                                  dtype=np.float32).reshape(-1, 69),
                    motor_ids=motor_ids if motor_ids is not None
                        else np.empty(0, np.int64),
                    source_ids=source_ids if source_ids is not None
                        else np.empty(0, np.int64),
                    recorded_controls=np.asarray(max(0, len(rows)-1)))
            attempt('control-endpoints.npz', save_endpoints)
            if future_snapshot is not None:
                attempt('observed-main-1p1', lambda: StateStore.save(
                    out/'observed-main-1p1', future_snapshot))
            def save_final_state():
                np.savez_compressed(out/'final-state.npz',
                    rate=session.network.rate.cpu().numpy().copy(),
                    adaptation=session.network.adaptation.cpu().numpy().copy(),
                    drive=session.network.drive.cpu().numpy().copy(),
                    output_mask=session.network.output_mask.cpu().numpy().copy(),
                    tick=np.asarray(session.network.tick),
                    tau=session.network.tau.cpu().numpy().copy(),
                    a=session.network.a.cpu().numpy().copy(),
                    threshold=session.network.threshold.cpu().numpy().copy(),
                    cap=session.network.cap.cpu().numpy().copy(),
                    motor_gain=session.network._cell_gain.cpu().numpy()[motor_ids].copy(),
                    activation=session.adapter.activation.copy())
            attempt('final-state.npz', save_final_state)
            if final_snapshot is not None:
                attempt('observed-final', lambda: StateStore.save(
                    out/'observed-final', final_snapshot))
            attempt('session-close', session.close)
        totals = ledger.totals()
        setup['attempted_controls'] = len(ledger.events)
        setup['synchronized_controls'] = totals['synchronized_controls']
        setup['recorded_control_endpoints'] = max(0, len(rows)-1)
        if totals['actual_neural_s'] is not None:
            setup['actual_main_neural_ticks'] = round(totals['actual_neural_s']/DT)
        if totals['actual_physical_s'] is not None:
            setup['actual_main_physics_ticks'] = round(totals['actual_physical_s']/.0001)
        attempt('setup-and-cost.json', lambda: save_json(out/'setup-and-cost.json',
            {**setup, **totals, 'wall_s': time.perf_counter()-began,
             'internal_wall_limit_s': 150, 'finalize_reserve_s': 20,
             'prior_conservative_process_s': 975,
             'shared_conservative_limit_s': 1530}))
    if primary is None and not errors:
        try:
            comparison = compare_after_raw(freeze, chosen, context, out,
                fixture, rows, np.asarray(motor_rate_rows, np.float32),
                np.asarray(activation_rows, np.float64),
                np.asarray(source_masked_rows, np.float32),
                future_snapshot, final_snapshot, metadata, saved_arrays)
            if time.perf_counter()-began > freeze['internal_wall_limit_s']:
                raise TimeoutError('E126 complete process exceeded internal 150s limit')
            save_json(out/'comparison.json', comparison)
        except BaseException as error:  # noqa: BLE001 - retain all raw on parity failure
            primary = error
            errors.append(error_record('comparison_after_raw', error))
    if errors:
        attempt('execution-errors.json', lambda: save_json(
            out/'execution-errors.json', errors))
        if primary is None:
            raise RuntimeError('E126 raw or cleanup failed; see execution-errors.json')
        raise primary
    result = {'status': 'valid_original_native_stage_dataset',
        'context': context, 'fit_calls': 0, 'candidate_adopted': False,
        'F1_F8': 'not_assessed', 'comparison': comparison,
        'actual_neural_s': ledger.totals()['actual_neural_s'],
        'actual_physical_s': ledger.totals()['actual_physical_s'],
        'cost': setup, 'wall_s': time.perf_counter()-began}
    save_json(out/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha', required=True)
    parser.add_argument('--source-review', type=Path, required=True)
    parser.add_argument('--source-review-sha', required=True)
    parser.add_argument('--context', choices=CONTEXTS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.freeze, args.source_review, args.context, args.out,
        freeze_sha=args.freeze_sha, review_sha=args.source_review_sha)


if __name__ == '__main__':
    main()
