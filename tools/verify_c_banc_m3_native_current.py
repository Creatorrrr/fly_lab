"""E125 A: one reviewed original E110 continuation with passive CUDA CSR taps.

The parent owns invocation on original inputs. This script never fits a model or
substitutes a decoded current for the native RK4 stage current.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE_FILES = (
    'tools/banc_m3_native_current.py',
    'tools/verify_c_banc_m3_native_current.py',
    'tests/test_c_banc_native_current.py',
)
SCHEMA = 'flylab.m3.e125.native-current-freeze.v1'
DESIGN = 'docs/BANC_M3_E125_NATIVE_CURRENT_DESIGN_20260921.md'
DT = .00025
CONTROLS = 240
STAGES = 19200
_THREAD_LIMITS = None


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def repository_file(relative):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('Repository-relative slash path required')
    name = Path(relative)
    if name.is_absolute() or '..' in name.parts or '.' in name.parts:
        raise ValueError('Unsafe repository-relative path: '+relative)
    path = ROOT/name
    if not path.resolve().is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError('Missing or escaped repository file: '+relative)
    return path


def repository_path(relative):
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('Repository-relative slash path required')
    name = Path(relative)
    if name.is_absolute() or '..' in name.parts or '.' in name.parts:
        raise ValueError('Unsafe repository-relative path: '+relative)
    path = ROOT/name
    if not path.resolve().is_relative_to(ROOT.resolve()) or not path.exists():
        raise ValueError('Missing or escaped repository path: '+relative)
    return path


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2,
        ensure_ascii=False, allow_nan=False), encoding='utf-8')


def error_record(label, error):
    return {'stage': label, 'type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc()}


def configure_single_thread_runtime():
    """Apply and measure the one-thread contract before any graph/model load."""
    global _THREAD_LIMITS
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError('Torch interop threading cannot be reduced to one')
    _THREAD_LIMITS = threadpool_limits(limits=1)
    pools = threadpool_info()
    measured = {'torch_intraop': torch.get_num_threads(),
                'torch_interop': torch.get_num_interop_threads(),
                'threadpools': pools}
    if (measured['torch_intraop'] != 1 or measured['torch_interop'] != 1
            or any(pool.get('num_threads') != 1 for pool in pools
                   if pool.get('user_api') in ('blas', 'openmp'))):
        raise RuntimeError('Measured Torch/BLAS/OpenMP threads must all equal one')
    return measured


def verify_contract(freeze_path, review_path, *, freeze_sha=None, review_sha=None,
                    clock=None):
    """Fail before the first model; no source or original hash may be omitted."""
    freeze_path, review_path = Path(freeze_path), Path(review_path)
    if freeze_sha is not None and file_sha(freeze_path) != freeze_sha:
        raise ValueError('Explicit E125 freeze SHA changed')
    if review_sha is not None and file_sha(review_path) != review_sha:
        raise ValueError('Explicit E125 source-review SHA changed')
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    review = json.loads(review_path.read_text(encoding='utf-8'))
    constants = {'schema': SCHEMA, 'design_path': DESIGN,
        'graph_n': 158706, 'graph_edges': 11584852,
        'graph_hash': '9bd8468f5e5e40c3e6daaffde9f444466825ba8e69189f1e347568bff85920ff',
        'controls': CONTROLS, 'neural_dt_s': DT, 'control_dt_s': .005,
        'physical_dt_s': .0001, 'capacity_stages': STAGES,
        'observed_neurons': 391, 'future_control': 220,
        'actual_neural_s': 1.2, 'actual_physical_s': 1.2,
        'internal_wall_limit_s': 90, 'finalize_reserve_s': 15,
        'external_wall_limit_s': 120, 'supervisor_limit_s': 125,
        'prior_conservative_process_s': 951,
        'shared_process_limit_s': 1076, 'automatic_retries': 0,
        'fit_calls': 0, 'F6_exposed': False, 'candidate_adopted': False}
    for key, value in constants.items():
        if freeze.get(key) != value or type(freeze[key]) is not type(value):
            raise ValueError('E125 freeze constant changed: '+key)
    if file_sha(repository_file(DESIGN)) != freeze.get('design_sha256'):
        raise ValueError('E125 design SHA changed')
    paths = freeze.get('paths')
    if not isinstance(paths, dict) or set(paths) != {
            'graph', 'start', 'future', 'calibration', 'metadata', 'signals', 'trace'}:
        raise ValueError('Exact seven E125 paths required')
    for name in paths.values():
        repository_path(name)
    groups = (('input_sha256', 70), ('reused_source_sha256', 12),
              ('preservation_sha256', 581))
    for label, count in groups:
        hashes = freeze.get(label)
        if not isinstance(hashes, dict) or len(hashes) != count:
            raise ValueError('E125 '+label+' cardinality changed')
        for name, expected in hashes.items():
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError('Invalid E125 SHA: '+str(name))
            if file_sha(repository_file(name)) != expected:
                raise ValueError('Frozen E125 file changed: '+name)
            if clock is not None and clock() > 75:
                raise TimeoutError('E125 preflight consumed execution reserve')
    if (review.get('accepted_for_single_original_native_current') is not True
            or review.get('blocking_defects') != []
            or review.get('freeze_sha256') != file_sha(freeze_path)
            or review.get('source_sha256') !=
                {name: file_sha(repository_file(name)) for name in SOURCE_FILES}):
        raise ValueError('Parent source review does not authorize these exact E125 sources')
    return freeze, review


class AdvanceLedger:
    """Distinguish completed synchronized work from requested or unknown work."""

    def __init__(self):
        self.events = []

    def advance(self, session):
        import torch

        event = {'requested_controls': 1, 'control_before': None,
            'control_after': None, 'neural_before': None, 'neural_after': None,
            'physics_before_s': None, 'physics_after_s': None,
            'neural_completed_s': None, 'physics_measured_s': None,
            'cuda_completion': 'not_started', 'advance_error': None,
            'clock_errors': []}
        self.events.append(event)
        event['control_before'] = int(session.control_tick)
        event['neural_before'] = int(session.network.tick)
        event['physics_before_s'] = float(session.body.physics_time())
        primary = None
        try:
            event['cuda_completion'] = 'requested'
            session.advance(1)
            torch.cuda.current_stream().synchronize()
            event['cuda_completion'] = 'synchronized'
        except BaseException as error:
            primary = error
            event['advance_error'] = type(error).__name__
            event['cuda_completion'] = 'unknown_after_error'
            raise
        finally:
            for key, operation in (
                ('control_after', lambda: int(session.control_tick)),
                ('neural_after', lambda: int(session.network.tick)),
                ('physics_after_s', lambda: float(session.body.physics_time()))):
                try:
                    event[key] = operation()
                except Exception as error:  # noqa: BLE001 - retain all independent clock failures
                    event['clock_errors'].append(key+': '+str(error))
            b, a = event['physics_before_s'], event['physics_after_s']
            if b is not None and a is not None and math.isfinite(a-b) and a >= b:
                event['physics_measured_s'] = a-b
            b, a = event['neural_before'], event['neural_after']
            if (event['cuda_completion'] == 'synchronized' and b is not None
                    and a is not None and a >= b):
                event['neural_completed_s'] = (a-b)*DT
            if primary is None and event['clock_errors']:
                raise RuntimeError('E125 advance clocks unreadable: '+
                                   ', '.join(event['clock_errors']))

    def totals(self):
        def total(key):
            values = [event[key] for event in self.events]
            return sum(values) if all(value is not None for value in values) else None
        return {'actual_neural_s': total('neural_completed_s'),
                'actual_physical_s': total('physics_measured_s'),
                'requested_controls': len(self.events),
                'synchronized_controls': sum(e['cuda_completion'] == 'synchronized'
                                             for e in self.events)}


def replace_with_observer(session, motor_ids, capacity):
    """Keep the existing E110 wrapper/body/adapter and all checkpoint fields."""
    from tools.banc_m3_native_current import ObservedCoreAdaptationNetwork
    from tools.probe_c_banc_m3_core_rhythm import source_matrix
    from tools.verify_c_banc_observation import exact

    old = session.network
    source = old.snapshot()
    weights = source_matrix(old)
    params = [getattr(old, name).cpu().numpy().copy()
              for name in ('tau', 'a', 'threshold', 'cap')]
    candidate = ObservedCoreAdaptationNetwork(weights, *params,
        adaptation_gain=old.adaptation_gain,
        adaptation_tau_s=old.adaptation_tau_s, dt=old.dt,
        device=old.device, cuda_implementation=old.cuda_implementation,
        capture_steps=old.interval_steps, core_ids=old.core_ids,
        core_gain=old.core_gain, observation_ids=motor_ids,
        capacity_stages=capacity)
    exact(old.identity, candidate.identity, 'E125 unchanged neural identity')
    candidate.restore(source)
    exact(source, candidate.snapshot(), 'E125 unchanged network checkpoint')
    session.network = candidate
    del old
    return candidate


def save_observation_raw(out, observer, errors):
    try:
        raw = observer.observation_raw()
    except Exception as error:  # noqa: BLE001 - preserve readback failure before raw salvage
        errors.append(error_record('native-stage-current-readback', error))
        return None
    try:
        np.savez_compressed(out/'native-stage-current.npz', **raw)
    except Exception as error:  # noqa: BLE001 - attempt independent uncompressed rescue
        errors.append(error_record('native-stage-current.npz', error))
        try:
            np.savez(out/'native-stage-current-rescue.npz', **raw)
        except Exception as rescue:  # noqa: BLE001 - retain original and rescue failures
            errors.append(error_record('native-stage-current-rescue.npz', rescue))
    return raw


def compare_after_raw(freeze, paths, rows, motor_ids, motor_rate, activation,
                      future_snapshot, stage_raw):
    from flylab.c.storage import StateStore
    from tools.verify_c_banc_observation import exact

    if len(rows) != CONTROLS+1 or future_snapshot is None or stage_raw is None:
        raise AssertionError('Full E125 observation required before comparisons')
    if (motor_rate.shape != (CONTROLS, 391)
            or activation.shape != (CONTROLS, 96)
            or not np.isfinite(motor_rate).all()
            or not np.isfinite(activation).all()):
        raise AssertionError('E125 endpoint rate/activation invalid')
    if (stage_raw['start_tick'] != 0 or stage_raw['end_tick'] != 4800
            or not np.all(stage_raw['counter_requested'] == STAGES)
            or not np.all(stage_raw['counter_completed'] == STAGES)
            or stage_raw['overflow'].tolist() != [0]
            or stage_raw['armed'].tolist() != [0]
            or stage_raw['total_current'].shape != (STAGES, 391)
            or not np.isfinite(stage_raw['total_current']).all()):
        raise AssertionError('E125 native stage cardinality/current invalid')
    exact(future_snapshot, StateStore.load(paths['future']),
          'E125 original full main-1p1 checkpoint')
    original = json.loads((paths['trace']).read_text(encoding='utf-8'))
    expected_rows = original['samples'][:CONTROLS+1]
    exact(rows, expected_rows, 'E125 original first240 control/body rows')
    with np.load(paths['signals'], allow_pickle=False) as source:
        ids = source['observed_ids']
        columns = np.searchsorted(np.sort(ids), motor_ids)
        order = np.argsort(ids)
        if not np.array_equal(ids[order][columns], motor_ids):
            raise AssertionError('E110 signal motor roster changed')
        columns = order[columns]
        exact(motor_rate, source['rate'][:CONTROLS, columns],
              'E125 original first240 motor rates')
        exact(activation, source['muscle_activation'][:CONTROLS],
              'E125 original first240 muscle activation')
    return {'whole_main_1p1_exact': True,
            'first240_control_body_trace_exact': True,
            'first240_motor_rate_exact': True,
            'first240_activation_exact': True,
            'native_stage_counts_exact': True}


def run(freeze_path, review_path, out, *, freeze_sha=None, review_sha=None):
    began = time.perf_counter()
    out = Path(out)
    if out.exists():
        raise FileExistsError('One-shot E125 output exists: '+str(out))
    out.mkdir(parents=True)
    errors = []
    ledger = AdvanceLedger()
    session = observer = None
    rows, rate_rows, adaptation_rows, activation_rows = [], [], [], []
    future_snapshot = stage_raw = initial = final = freeze = None
    motor_ids = tendon_names = None
    setup = {'models_constructed': 0, 'model_setup_completion': 'not_started',
             'warmup_native_ticks': 0, 'capture_native_ticks': 0,
             'actual_main_native_ticks': None,
             'actual_main_physics_ticks': None, 'actual_main_controls': 0}
    comparison = None
    primary = None
    try:
        freeze, review = verify_contract(freeze_path, review_path,
            freeze_sha=freeze_sha, review_sha=review_sha,
            clock=lambda: time.perf_counter()-began)
        paths = {key: repository_path(value) for key, value in freeze['paths'].items()}
        save_json(out/'provenance.json', {'freeze_sha256': file_sha(freeze_path),
            'source_review_sha256': file_sha(review_path),
            'design_sha256': freeze['design_sha256'],
            'source_sha256': review['source_sha256'],
            'input_sha256': freeze['input_sha256'],
            'reused_source_sha256': freeze['reused_source_sha256'],
            'preservation_sha256': freeze['preservation_sha256']})
        reviewed = out/'reviewed-source'
        reviewed.mkdir()
        for name in SOURCE_FILES:
            copied = reviewed/Path(name).name
            shutil.copyfile(repository_file(name), copied)
            if file_sha(copied) != review['source_sha256'][name]:
                raise ValueError('Reviewed source changed while copying: '+name)
        for source, name, expected in (
            (freeze_path, 'freeze.json', file_sha(freeze_path)),
            (review_path, 'source-review.json', file_sha(review_path)),
            (repository_file(DESIGN), 'design.md', freeze['design_sha256'])):
            copied = reviewed/name
            shutil.copyfile(source, copied)
            if file_sha(copied) != expected:
                raise ValueError('Reviewed freeze/design changed while copying')
        measured_threads = configure_single_thread_runtime()
        save_json(out/'threading.json', measured_threads)
        import torch

        from flylab.c.graph import GraphStore
        from flylab.c.storage import StateStore
        from flylab.c.walking_trace import capture_control
        from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
        from tools.verify_c_banc_observation import exact
        if not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA GPU required')
        save_json(out/'environment.json', {'python': sys.version,
            'platform': platform.platform(), 'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(),
            'cuda_available': True, 'pid': os.getpid(),
            'measured_threads': measured_threads})
        graph = GraphStore.load(paths['graph'])
        if (graph.n != freeze['graph_n'] or len(graph.indices) != freeze['graph_edges']
                or graph.hash != freeze['graph_hash']):
            raise ValueError('Frozen original full BANC graph mismatch')
        start = StateStore.load(paths['start'])
        cal = np.load(paths['calibration'], allow_pickle=False)
        motor_ids = cal['motor_ids'].copy()
        cal.close()
        with np.load(paths['metadata'], allow_pickle=False) as metadata:
            exact(metadata['motor_ids'], motor_ids, 'E125 original motor roster')
        if (motor_ids.shape != (391,) or len(np.unique(motor_ids)) != 391
                or start['network']['tick'] != 0 or start['control_tick'] != 0):
            raise ValueError('E125 full original zero start/motor roster required')
        # This existing restore chain constructs BancWalking, CoreTrial,
        # EfficacyTrial, then IntrinsicTrial networks; each captures once.
        setup['model_setup_completion'] = 'restore_requested_completion_unknown'
        setup['models_constructed'] = None
        setup['warmup_native_ticks'] = None
        setup['capture_native_ticks'] = None
        session = IntrinsicRecruitmentTrial.from_checkpoint(graph, start)
        setup['models_constructed'] = 4
        setup['warmup_native_ticks'] = 8*session.network.interval_steps
        setup['capture_native_ticks'] = 4*session.network.interval_steps
        setup['model_setup_completion'] = 'four_restored_models_complete'
        exact(start, session.snapshot(), 'E125 original restored start')
        setup['model_setup_completion'] = 'observer_requested_completion_unknown'
        observer = replace_with_observer(session, motor_ids, STAGES)
        setup['models_constructed'] = 5
        setup['warmup_native_ticks'] += 2*observer.interval_steps
        setup['capture_native_ticks'] += observer.interval_steps
        setup['model_setup_completion'] = 'all_five_models_complete'
        exact(start, session.snapshot(), 'E125 observed whole start')
        tendon_names = np.asarray([
            session.adapter.groups[int(i)]['leg']+'_tarsus'
            for i in session.adapter.calibration['ltm']])
        if len(np.unique(tendon_names)) != 6:
            raise ValueError('Exactly six unique frozen LTM tendon names required')
        initial = session.snapshot()
        np.savez_compressed(out/'initial-state.npz',
            rate=initial['network']['rate'],
            adaptation=initial['network']['adaptation'],
            tau=observer.tau.cpu().numpy().copy(),
            a=observer.a.cpu().numpy().copy(),
            threshold=observer.threshold.cpu().numpy().copy(),
            cap=observer.cap.cpu().numpy().copy(),
            motor_ids=motor_ids,
            motor_gain=observer._cell_gain.cpu().numpy()[motor_ids].copy(),
            adaptation_gain=np.asarray(observer.adaptation_gain),
            adaptation_tau_s=np.asarray(observer.adaptation_tau_s),
            neural_dt_s=np.asarray(observer.dt),
            pool=session.adapter.calibration['pool'],
            moment=session.adapter.calibration['moment'],
            rest=session.adapter.calibration['rest'],
            ltm=session.adapter.calibration['ltm'],
            tendon_names=tendon_names,
            initial_activation=session.adapter.activation.copy(),
            initial_drive=initial['network']['drive'],
            initial_mask=initial['network']['output_mask'])
        StateStore.save(out/'observed-start', initial)
        observer.arm_observation()
        rows.append(capture_control(session))
        for control in range(1, CONTROLS+1):
            if time.perf_counter()-began >= 75:
                raise TimeoutError('E125 stopped before raw-save reserve')
            ledger.advance(session)
            if (observer.tick != control*20 or session.control_tick != control
                    or abs(session.body.physics_time()-control*.005) > 1e-8):
                raise ValueError('E125 native/control/body clocks diverged')
            row = capture_control(session)
            rows.append(row)
            rate_rows.append(observer.readout(motor_ids))
            adaptation_rows.append(observer.adaptation[motor_ids].cpu().numpy().copy())
            activation_rows.append(session.adapter.activation.copy())
            if control == freeze['future_control']:
                future_snapshot = session.snapshot()
            if session.fault or session.body.fault:
                raise RuntimeError('E125 original native body health fault')
        torch.cuda.current_stream().synchronize()
    except BaseException as error:  # noqa: BLE001 - save partial work, then re-raise below
        primary = error
        errors.append(error_record('execution', error))
    finally:
        def attempt(label, operation):
            try:
                operation()
            except BaseException as error:  # noqa: BLE001 - attempt each independent artifact
                errors.append(error_record(label, error))

        # These independent saves precede every scientific parity assertion.
        if observer is not None:
            stage_raw = save_observation_raw(out, observer, errors)
        attempt('advance-ledger.json', lambda: save_json(out/'advance-ledger.json',
            {'events': ledger.events, **ledger.totals()}))
        if session is not None:
            attempt('trace.json', lambda: save_json(out/'trace.json',
                {'schema': 'flylab.m3.e125.native-current-trace.v1',
                 'samples': rows}))
            def save_endpoints():
                np.savez_compressed(out/'control-endpoints.npz',
                    motor_rate=np.asarray(rate_rows, dtype=np.float32).reshape(-1, 391),
                    motor_adaptation=np.asarray(adaptation_rows,
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
                    start_tick=np.asarray(initial['network']['tick']
                                          if initial is not None else -1),
                    start_tick_valid=np.asarray(initial is not None),
                    completed_controls=np.asarray(max(0, len(rows)-1)))
            attempt('control-endpoints.npz', save_endpoints)
            if future_snapshot is not None:
                attempt('observed-main-1p1', lambda: StateStore.save(
                    out/'observed-main-1p1', future_snapshot))
            def save_final():
                np.savez_compressed(out/'final-state.npz',
                    rate=session.network.rate.cpu().numpy().copy(),
                    adaptation=session.network.adaptation.cpu().numpy().copy(),
                    drive=session.network.drive.cpu().numpy().copy(),
                    output_mask=session.network.output_mask.cpu().numpy().copy(),
                    tau=session.network.tau.cpu().numpy().copy(),
                    a=session.network.a.cpu().numpy().copy(),
                    threshold=session.network.threshold.cpu().numpy().copy(),
                    cap=session.network.cap.cpu().numpy().copy(),
                    motor_gain=session.network._cell_gain.cpu().numpy()[motor_ids].copy(),
                    adaptation_gain=np.asarray(session.network.adaptation_gain),
                    adaptation_tau_s=np.asarray(session.network.adaptation_tau_s),
                    neural_dt_s=np.asarray(session.network.dt),
                    tick=np.asarray(session.network.tick),
                    activation=session.adapter.activation.copy())
            attempt('final-state.npz', save_final)
            if primary is None and not session.fault and not session.body.fault:
                def save_final_checkpoint():
                    nonlocal final
                    final = session.snapshot()
                    StateStore.save(out/'observed-final', final)
                attempt('observed-final', save_final_checkpoint)
            attempt('session-close', session.close)
        totals = ledger.totals()
        setup['attempted_main_controls'] = len(ledger.events)
        setup['actual_main_controls'] = totals['synchronized_controls']
        setup['recorded_control_endpoints'] = max(0, len(rows)-1)
        if totals['actual_neural_s'] is not None:
            setup['actual_main_native_ticks'] = round(totals['actual_neural_s']/DT)
        if totals['actual_physical_s'] is not None:
            setup['actual_main_physics_ticks'] = round(totals['actual_physical_s']/.0001)
        attempt('setup-and-cost.json', lambda: save_json(out/'setup-and-cost.json',
            {**setup, **totals, 'wall_s': time.perf_counter()-began,
             'main_wall_limit_s': 90, 'finalize_reserve_s': 15,
             'prior_conservative_process_s': 951,
             'shared_conservative_limit_s': 1076}))
    if primary is None and not errors:
        try:
            comparison = compare_after_raw(freeze, paths, rows, motor_ids,
                np.asarray(rate_rows, dtype=np.float32),
                np.asarray(activation_rows, dtype=np.float64),
                future_snapshot, stage_raw)
            if time.perf_counter()-began > freeze['internal_wall_limit_s']:
                raise TimeoutError('E125 complete process exceeded internal 90s limit')
            save_json(out/'comparison.json', comparison)
        except BaseException as error:  # noqa: BLE001 - retain raw and comparison failure
            primary = error
            errors.append(error_record('comparison_after_raw', error))
    if errors:
        try:
            save_json(out/'execution-errors.json', errors)
        except Exception as error:  # noqa: BLE001 - keep the primary exception intact
            if primary is not None:
                primary.add_note('Execution error report failed: '+str(error))
        if primary is None:
            raise RuntimeError('E125 raw or cleanup persistence failed; see execution-errors.json')
        raise primary
    result = {'status': 'valid_transparent_native_current_observation',
        'hypothesis_A_support': None, 'gait_candidate_adopted': False,
        'F1_F8': 'not_assessed', 'fit_calls': 0,
        'comparison': comparison, 'cost': setup,
        'actual_neural_s': ledger.totals()['actual_neural_s'],
        'actual_physical_s': ledger.totals()['actual_physical_s'],
        'wall_s': time.perf_counter()-began}
    save_json(out/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha')
    parser.add_argument('--source-review', type=Path, required=True)
    parser.add_argument('--source-review-sha')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.freeze, args.source_review, args.out,
        freeze_sha=args.freeze_sha, review_sha=args.source_review_sha)


if __name__ == '__main__':
    main()
