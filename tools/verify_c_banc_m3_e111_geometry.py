"""E111 one-shot static E32 foot-geometry validation; no neural/physics step.

The native run requires the parent's source review and explicit authorization.
This program neither fits a controller nor makes a walking-quality claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flylab.c.storage import StateStore
from tools import analyze_c_banc_observation as observation
from tools.banc_m3_kinematic_fit import StaticFootGeometry
from tools.banc_m3_torch_foot_geometry import (
    TorchFootGeometry,
    extract_from_body,
)
from tools.verify_c_banc_observation import exact

LIMIT_S = 60.
PRIOR_CONSERVATIVE_S = 310.
SHARED_LIMIT_S = 370.
SEED = 111
THRESHOLDS = {'position_mm': 1e-9, 'jacobian_mm_per_rad': 1e-8,
              'cpu_cuda_mm': 1e-10, 'cross_leg_mm_per_rad': 0.}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     ensure_ascii=False, allow_nan=False), encoding='utf-8')


def _error(error):
    return None if error is None else {'type': type(error).__name__,
                                      'message': str(error)}


def _try_save(path, value):
    try:
        save_json(path, value)
        return None
    except Exception as error:  # noqa: BLE001 - retain an earlier failure
        return _error(error)


def new_body_owned(saved, out, *, body_factory=None):
    """Own FlyBody immediately after construction, including restore failure."""
    if body_factory is None:
        body_factory = observation.FlyGymBody
    body = None
    try:
        body = body_factory(
            saved['seed'], saved['world'], observation.config_values(None),
            body_options=observation.BodyOptions(
                model='flybody', actuation='whole_body',
                servo_profile=saved['parameters'].get('servo_profile', 'tracking_all'),
                tendons='all'))
        params = saved['parameters']
        observation.configure_torque_capacity(
            body, params.get('torque_limit_bw_mm', 0.0))
        observation.configure_activation_trajectory(
            body, params.get('joint_activation_tau_s', .03)
            if params.get('continuous_joint_activation', False) else 0.0)
        observation.configure_active_length(body, params.get('active_length', False))
        observation.configure_finite_pad_contact(
            body, params.get('finite_pad_contact', False))
        observation.configure_adhesion_units(
            body, params.get('adhesion_unit_conversion', False))
        if params.get('adhesion_model') == 'graded':
            body.graded_adhesion = True
        body.restore(saved['body'])
        return body
    except Exception as error:  # retain original construction fault
        close_error = None
        if body is not None:
            try:
                body.close()
            except Exception as secondary:  # noqa: BLE001 - retain restore fault
                close_error = secondary
                error.add_note('E111 constructed-body close failed: '+str(secondary))
        save_error = _try_save(out/'construction-error.json', {
            'error': _error(error), 'close_error': _error(close_error),
            'constructed_body_closed': body is not None and close_error is None})
        if save_error is not None:
            error.add_note('E111 construction-error persistence failed: '+
                           str(save_error))
        raise


def input_commands(root):
    """Pure data preparation; the 256 RNG values stay in observed ranges."""
    e110 = root/'e110-e109-native-body-diagnostic'
    e85 = root/'e85-space-compatible-cadence-reference'
    parent = json.loads((e110/'parent-final-review.json').read_text(encoding='utf-8'))
    if parent.get('accepted_as_valid_actual_diagnostic') is not True or \
            parent.get('blocking_defects') != [] or \
            len(parent.get('original_artifact_sha256', {})) != 201:
        raise ValueError('Accepted E110 and all frozen original hashes required')
    frozen_hashes = {name.replace('\\', '/'): digest for name, digest in
                     parent['original_artifact_sha256'].items()}
    for name in ('trace.json', 'result.json', 'start/manifest.json'):
        if sha256(e110/name) != frozen_hashes[name]:
            raise ValueError('E110 source artifact changed: '+name)
    e110_trace = json.loads((e110/'trace.json').read_text(encoding='utf-8'))
    if len(e110_trace['samples']) != 2001 or \
            [item['control_tick'] for item in e110_trace['samples']] != list(range(2001)):
        raise ValueError('Complete actual E110 5ms control record required')
    actual = np.asarray([row['joint_targets_rad'] for row in
                         e110_trace['samples'][1:]], dtype=np.float64)
    with np.load(e85/'commands.npz', allow_pickle=False) as archive:
        reference = np.asarray(archive['command'], dtype=np.float64)
        time_s = np.asarray(archive['time_s'], dtype=np.float64)
    if actual.shape != (2000, 42) or reference.shape != actual.shape or \
            time_s.shape != (2000,) or not np.isfinite(actual).all() or \
            not np.isfinite(reference).all() or \
            np.max(np.abs(time_s - .005*np.arange(1, 2001))) > 1e-12:
        raise ValueError('E110/E85 fixed 2000x42 commands and 5ms grid required')
    lower = np.minimum(actual.min(axis=0), reference.min(axis=0))
    upper = np.maximum(actual.max(axis=0), reference.max(axis=0))
    random = np.random.default_rng(SEED).uniform(lower, upper, size=(256, 42))
    commands = np.concatenate((actual, reference, random), axis=0)
    provenance = {'seed': SEED, 'counts': {'E110': 2000, 'E85': 2000,
                                          'observed_range_random': 256},
                  'source_sha256': {
                      'E110_trace': sha256(e110/'trace.json'),
                      'E110_parent_review': sha256(e110/'parent-final-review.json'),
                      'E85_commands': sha256(e85/'commands.npz'),
                      'E32_start_manifest': sha256(root/'e32-core-adaptation/candidate/start/manifest.json'),
                  }, 'method': 'PCG64 seed111 independent uniform per joint in union of E110/E85 observed command ranges'}
    return commands, lower, upper, provenance


def _deadline(started):
    if time.perf_counter()-started >= LIMIT_S-.5:
        raise TimeoutError('E111 one-process 60s limit reached; no automatic retry')


def native_values(body, qref, commands, started, out):
    """Original CPU MuJoCo static points and full 42-column Jacobian."""
    points = np.empty((len(commands), 6, 3), dtype=np.float64)
    jacobian = np.empty((len(commands), 6, 3, 42), dtype=np.float64)
    completed = 0
    try:
        for first in range(0, len(commands), 256):
            _deadline(started)
            last = min(first+256, len(commands))
            geometry = StaticFootGeometry(
                body, np.repeat(qref[None], last-first, axis=0))
            points[first:last], jacobian[first:last] = geometry.evaluate(
                commands[first:last], jacobian=True)
            completed = last
    except Exception as error:  # preserve native failure
        try:
            np.savez_compressed(out/'native-prefix.npz', completed_rows=completed,
                                point=points[:completed], jacobian=jacobian[:completed])
        except Exception as secondary:  # noqa: BLE001 - prefix is best effort
            error.add_note('E111 native-prefix persistence failed: '+str(secondary))
            report_error = _try_save(out/'native-prefix-error.json', {
                'native_error': _error(error), 'persistence_error': _error(secondary),
                'completed_rows': completed})
            if report_error is not None:
                error.add_note('E111 native-prefix-error persistence failed: '+
                               str(report_error))
        raise
    return points, jacobian


def torch_values(geometry, commands, device, started):
    """Time-separable Jacobian: 18 output-component sums, never T-by-T."""
    _deadline(started)
    staged = geometry.to(device)
    source = torch.tensor(commands, dtype=torch.float64, device=device,
                          requires_grad=True)
    points = staged.full(source)
    jacobian = np.empty((len(commands), 6, 3, 42), dtype=np.float64)
    for leg in range(6):
        for component in range(3):
            _deadline(started)
            derivative, = torch.autograd.grad(points[:, leg, component].sum(),
                                               source, retain_graph=True)
            jacobian[:, leg, component] = derivative.detach().cpu().numpy()
    if device == 'cuda':
        torch.cuda.synchronize()
    return points.detach().cpu().numpy(), jacobian, staged


def leg_values(geometry, commands, started):
    """Exercise independent N7 API on every sample without full temporal Jacobian."""
    result = np.empty((len(commands), 6, 3), dtype=np.float64)
    jacobian = np.empty((len(commands), 6, 3, 7), dtype=np.float64)
    for leg in range(6):
        _deadline(started)
        indices = geometry.arrays['leg_frontend_indices'][leg]
        source = torch.tensor(commands[:, indices], dtype=torch.float64,
                              device=geometry.device, requires_grad=True)
        point = geometry.leg(source, leg)
        result[:, leg] = point.detach().cpu().numpy()
        for component in range(3):
            _deadline(started)
            derivative, = torch.autograd.grad(point[:, component].sum(),
                                               source, retain_graph=True)
            jacobian[:, leg, component] = derivative.detach().cpu().numpy()
    return result, jacobian


def benchmark_torch(geometry, commands, device, started):
    """Fixed point + one scalar-loss backward, 3 warmups then 20 repeats."""
    staged = geometry.to(device)
    source = torch.tensor(commands, dtype=torch.float64, device=device,
                          requires_grad=True)
    weight = torch.ones((len(commands), 6, 3), dtype=torch.float64,
                        device=device)
    durations = []
    for iteration in range(23):
        _deadline(started)
        source.grad = None
        if device == 'cuda':
            torch.cuda.synchronize()
        began = time.perf_counter()
        (staged.full(source)*weight).sum().backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-began
        if iteration >= 3:
            durations.append(elapsed)
    return durations


def benchmark_native(geometry, commands, started):
    """Three repeats of native CPU point plus all six full Jacobians."""
    durations = []
    for _ in range(3):
        _deadline(started)
        began = time.perf_counter()
        geometry.evaluate(commands, jacobian=True)
        durations.append(time.perf_counter()-began)
    return durations


def quality(native_p, native_j, cpu_p, cpu_j, cuda_p, cuda_j,
            cpu_leg, cpu_leg_j, cuda_leg, cuda_leg_j, geometry):
    tree = geometry.arrays['leg_frontend_indices']
    foreign = np.ones((6, 42), dtype=bool)
    for leg in range(6):
        foreign[leg, tree[leg]] = False
    cross = max(float(np.max(np.abs(cpu_j[:, leg, :, foreign[leg]])))
                for leg in range(6))
    cross_cuda = max(float(np.max(np.abs(cuda_j[:, leg, :, foreign[leg]])))
                     for leg in range(6))
    leg_cpu_jac = max(float(np.max(np.abs(
        cpu_j[:, leg][:, :, tree[leg]]-cpu_leg_j[:, leg]))) for leg in range(6))
    leg_cuda_jac = max(float(np.max(np.abs(
        cuda_j[:, leg][:, :, tree[leg]]-cuda_leg_j[:, leg]))) for leg in range(6))
    checks = {
        'native_cpu_point': float(np.max(np.abs(native_p-cpu_p))),
        'native_cuda_point': float(np.max(np.abs(native_p-cuda_p))),
        'native_cpu_jacobian': float(np.max(np.abs(native_j-cpu_j))),
        'native_cuda_jacobian': float(np.max(np.abs(native_j-cuda_j))),
        'cpu_cuda_point': float(np.max(np.abs(cpu_p-cuda_p))),
        'cpu_cuda_jacobian': float(np.max(np.abs(cpu_j-cuda_j))),
        'cpu_full_leg_point': float(np.max(np.abs(cpu_p-cpu_leg))),
        'cuda_full_leg_point': float(np.max(np.abs(cuda_p-cuda_leg))),
        'cpu_full_leg_jacobian': leg_cpu_jac,
        'cuda_full_leg_jacobian': leg_cuda_jac,
        'cross_leg_cpu_jacobian': cross,
        'cross_leg_cuda_jacobian': cross_cuda,
    }
    screens = {key: math.isfinite(value) and value <= (
        THRESHOLDS['jacobian_mm_per_rad'] if 'jacobian' in key
        and not key.startswith('cpu_cuda') and not key.startswith('cross_leg') else
        THRESHOLDS['cross_leg_mm_per_rad'] if key.startswith('cross_leg') else
        THRESHOLDS['cpu_cuda_mm'] if key.startswith('cpu_cuda') else
        THRESHOLDS['position_mm']) for key, value in checks.items()}
    return checks, screens


def run(root, out):
    started = time.perf_counter()
    if out.exists():
        raise FileExistsError('Single E111 output exists: '+str(out))
    out.mkdir(parents=True)
    save_json(out/'spec.json', {'experiment': 'M3-E111-CUDA-foot-geometry',
        'scope': 'static geometry only; no fit, neural/physical advance or F1-F8 claim',
        'thresholds': THRESHOLDS, 'process_limit_s': LIMIT_S,
        'prior_conservative_wall_s': PRIOR_CONSERVATIVE_S,
        'shared_limit_s': SHARED_LIMIT_S, 'tool_sha256': sha256(__file__)})
    body = None
    before = None
    before_clock = None
    primary = None
    result = None
    cleanup = {'snapshot_exact': None, 'clock_exact': None, 'closed': None,
               'snapshot_error': None, 'clock_error': None,
               'after_save_error': None, 'close_error': None}
    try:
        commands, lower, upper, provenance = input_commands(root)
        _deadline(started)
        np.savez_compressed(out/'inputs.npz', commands=commands,
                            observed_lower=lower, observed_upper=upper)
        save_json(out/'input-provenance.json', provenance)
        original = StateStore.load(root/'e32-core-adaptation/candidate/start')
        body = new_body_owned(original, out)
        before_clock = np.float64(body.d.time).tobytes()
        before = body.snapshot()
        save_json(out/'body-before.json', before)
        if body.model_hash != original['body']['modelHash']:
            raise ValueError('Original E32 body model hash changed')
        exact(original['body'], before, 'Original E32 restored body state')
        qref = np.asarray(body.d.qpos, dtype=np.float64).copy()
        geom = extract_from_body(body, qref)
        geom.save(out/'geometry.npz')
        loaded = TorchFootGeometry.load(out/'geometry.npz')
        if loaded.geometry_hash != geom.geometry_hash:
            raise ValueError('Fixed geometry roundtrip changed')
        save_json(out/'geometry-provenance.json', {
            'geometry_hash': geom.geometry_hash, 'model_hash': body.model_hash,
            'reference_qpos_sha256': hashlib.sha256(qref.tobytes()).hexdigest(),
            'frontend_joint_names': list(body.joint_names),
            'tree_frontend_indices': geom.arrays['tree_frontend_indices'].tolist(),
            'tree_joint_ids': geom.arrays['tree_joint_ids'].tolist(),
            'passive_joint_ids': geom.arrays['passive_joint_ids'].tolist(),
            'tip_body_ids': geom.arrays['tip_body_ids'].tolist(),
            'axes': geom.arrays['axes'].tolist(),
            'anchors': geom.arrays['anchors'].tolist(),
            'reference_tips': geom.arrays['tips'].tolist(),
            'geometry_npz_sha256': sha256(out/'geometry.npz')})
        native_p, native_j = native_values(body, qref, commands, started, out)
        np.savez_compressed(out/'native-raw.npz', point=native_p, jacobian=native_j)
        cpu_p, cpu_j, cpu_geometry = torch_values(loaded, commands, 'cpu', started)
        np.savez_compressed(out/'cpu-raw.npz', point=cpu_p, jacobian=cpu_j)
        cuda_p, cuda_j, cuda_geometry = torch_values(loaded, commands, 'cuda', started)
        np.savez_compressed(out/'cuda-raw.npz', point=cuda_p, jacobian=cuda_j)
        cpu_leg, cpu_leg_j = leg_values(cpu_geometry, commands, started)
        cuda_leg, cuda_leg_j = leg_values(cuda_geometry, commands, started)
        np.savez_compressed(out/'leg-raw.npz', cpu=cpu_leg, cuda=cuda_leg,
                            cpu_jacobian=cpu_leg_j, cuda_jacobian=cuda_leg_j)
        checks, screens = quality(native_p, native_j, cpu_p, cpu_j,
                                  cuda_p, cuda_j, cpu_leg, cpu_leg_j,
                                  cuda_leg, cuda_leg_j, loaded)
        save_json(out/'comparison.json', {'errors': checks, 'screens': screens,
            'raw_saved_before_assertions': True})
        if not all(screens.values()):
            raise ValueError('E111 native/Torch static geometry thresholds failed')
        benchmark = {'operation_scope': {
            'torch': 'batch six-foot point + one scalar weighted loss backward VJP',
            'native': 'CPU six-foot point + full 6x3x42 MuJoCo Jacobian',
            'comparison': 'different derivative output scope; timing is not an exact same-work speed ratio'},
            'warmup_torch': 3, 'repeat_torch': 20, 'repeat_native': 3,
            'durations_s': {}}
        for count in (92, 1827):
            subset = commands[:count]
            for device in ('cpu', 'cuda'):
                benchmark['durations_s'][f'{device}_{count}'] = benchmark_torch(
                    loaded, subset, device, started)
                save_json(out/'benchmark-progress.json', benchmark)
            native_bench_geometry = StaticFootGeometry(
                body, np.repeat(qref[None], count, axis=0))
            benchmark['durations_s'][f'native_cpu_{count}'] = benchmark_native(
                native_bench_geometry, subset, started)
            save_json(out/'benchmark-progress.json', benchmark)
        save_json(out/'benchmark.json', benchmark)
        result = {'validity': 'valid_static_geometry_diagnostic',
                  'all_geometry_screens_pass': True, 'sample_count': len(commands),
                  'geometry_hash': geom.geometry_hash, 'fit_calls': 0,
                  'neural_advance_s': 0., 'physical_advance_s': 0.,
                  'elapsed_in_process_s': time.perf_counter()-started,
                  'F1_F8': 'not_assessed', 'body_quality_improvement': 'not_assessed'}
    except Exception as error:  # noqa: BLE001 - preserve first error through cleanup
        primary = error
        actual_clock = None
        if body is not None:
            try:
                actual_clock = np.float64(body.d.time).tobytes()
            except Exception as clock_error:  # noqa: BLE001 - retain first error
                cleanup['clock_error'] = _error(clock_error)
        persistence_error = _try_save(out/'execution-error.json', {'error': _error(error),
            'traceback': traceback.format_exc(),
            'elapsed_in_process_s': time.perf_counter()-started,
            'fit_calls': 0, 'neural_advance_s': 0.,
            'physical_advance_s': 0. if before_clock is not None and
                                  actual_clock == before_clock else None})
        if persistence_error is not None:
            error.add_note('E111 execution-error persistence failed: '+
                           str(persistence_error))
    finally:
        if body is not None:
            try:
                cleanup['clock_exact'] = (before_clock ==
                    np.float64(body.d.time).tobytes()) if before_clock is not None else None
            except Exception as error:  # noqa: BLE001 - still snapshot and close
                cleanup['clock_error'] = _error(error)
            try:
                after = body.snapshot()
                cleanup['after_save_error'] = _try_save(out/'body-after.json', after)
                if before is not None:
                    exact(before, after, 'Original E32 static body snapshot')
                    cleanup['snapshot_exact'] = True
            except Exception as error:  # noqa: BLE001 - still close after snapshot failure
                cleanup['snapshot_exact'] = False if before is not None else None
                cleanup['snapshot_error'] = _error(error)
            try:
                body.close()
                cleanup['closed'] = True
            except Exception as error:  # noqa: BLE001 - record cleanup failure
                cleanup['closed'] = False
                cleanup['close_error'] = _error(error)
        cleanup_save_error = _try_save(out/'cleanup.json', cleanup)
        if cleanup_save_error is not None:
            if primary is not None:
                primary.add_note('E111 cleanup persistence failed: '+
                                 str(cleanup_save_error))
            else:
                primary = RuntimeError('E111 cleanup persistence failed: '+
                                       str(cleanup_save_error))
        if result is not None:
            result['body_snapshot_exact'] = cleanup['snapshot_exact']
            result['body_clock_exact'] = cleanup['clock_exact']
            result['body_closed'] = cleanup['closed']
            result['elapsed_in_process_s'] = time.perf_counter()-started
            result['resource_limit_met_internal'] = result['elapsed_in_process_s'] <= LIMIT_S
            result['conservative_shared_wall_s_internal'] = (
                PRIOR_CONSERVATIVE_S + result['elapsed_in_process_s'])
            if cleanup['snapshot_exact'] is not True or \
                    cleanup['clock_exact'] is not True or \
                    cleanup['closed'] is not True or \
                    cleanup['after_save_error'] is not None or \
                    not result['resource_limit_met_internal']:
                result['validity'] = 'incomplete'
            result_save_error = _try_save(out/'result.json', result)
            if result_save_error is not None:
                if primary is not None:
                    primary.add_note('E111 result persistence failed: '+
                                     str(result_save_error))
                else:
                    primary = RuntimeError('E111 result persistence failed: '+
                                           str(result_save_error))
    if primary is not None:
        raise primary
    if result['validity'] != 'valid_static_geometry_diagnostic':
        raise RuntimeError('E111 static validation incomplete; inspect cleanup/result')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.out), sort_keys=True), flush=True)
