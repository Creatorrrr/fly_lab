"""E130 isolated CUDA adapter: reuse one float64 expm1(delta) vector per solve.

The E128 source and its launch arguments remain immutable. Use only in an
isolated comparison process; the temporary factory is module global.
"""

from __future__ import annotations

import contextlib
import copy
import sys

import numpy as np

from tools import banc_m3_joint_adaptive_recruitment as neural

TARGET = 'expm1(delta[fit[j]])'
REPLACEMENT = 'prepared_expm1[fit[j]]'
PREPARE_SOURCE = r'''
extern "C" __global__ void prepare_expm1(
    const double *delta, double *prepared_expm1, int count) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < count) prepared_expm1[index] = expm1(delta[index]);
}
'''
_CANDIDATE = {}


def candidate_source():
    """Only add the vector argument and replace the single approved expression."""
    source = neural.CUDA_SOURCE
    if source.count(TARGET) != 1 or source.count('extern "C" __global__ void joint_forward(') != 1:
        raise ValueError('E128 CUDA source no longer has the one frozen expm1 site')
    signature = 'const double *initial_z,const double *gain,const double *delta,'
    if source.count(signature) != 1:
        raise ValueError('E128 joint_forward signature changed')
    source = source.replace(signature, signature + '\n    const double *prepared_expm1,', 1)
    return source.replace(TARGET, REPLACEMENT, 1)


def _candidate_kernel(name):
    import cupy as cp
    if name not in _CANDIDATE:
        source = candidate_source() if name == 'joint_forward' else PREPARE_SOURCE
        _CANDIDATE[name] = cp.RawKernel(source, name, options=('--std=c++11',))
    return _CANDIDATE[name]


class PrecomputedExpm1:
    """Per-engine launch receipts; unknown means a request was not synchronized."""

    def __init__(self, engine):
        self.engine = engine
        self.events = []
        self.gpu_temp_bytes_requested = 0
        self.gpu_temp_bytes_allocated = 0
        self.gpu_temp_bytes_peak = 0

    def snapshot(self):
        events = copy.deepcopy(self.events)
        counts = {kind: {state: sum(bool(e[state]) for e in events if e['kind'] == kind)
                         for state in ('requested', 'enqueued', 'synchronized',
                                       'unknown_requested')}
                  for kind in ('prepare', 'joint_forward')}
        return {'events': events, 'counts': counts,
                'gpu_temp_bytes_requested': self.gpu_temp_bytes_requested,
                'gpu_temp_bytes_allocated': self.gpu_temp_bytes_allocated,
                'gpu_temp_bytes_peak': self.gpu_temp_bytes_peak}

    def _event(self, kind, nbytes=0):
        event = {'kind': kind, 'bytes': int(nbytes), 'requested': True,
                 'enqueued': False, 'synchronized': False,
                 'unknown_requested': False, 'error': None}
        self.events.append(event)
        return event

    def _forward(self, grid, block, arguments, *rest, **kwargs):
        import cupy as cp
        delta = arguments[5]
        count = int(delta.size)
        if count != self.engine.edges or delta.dtype != cp.float64:
            raise ValueError('Candidate requires the unchanged E128 float64 delta')
        nbytes = int(delta.nbytes)
        prep = self._event('prepare', nbytes)
        self.gpu_temp_bytes_requested += nbytes
        try:
            prepared = cp.empty_like(delta)
            self.gpu_temp_bytes_allocated += nbytes
            self.gpu_temp_bytes_peak = max(self.gpu_temp_bytes_peak, nbytes)
            _candidate_kernel('prepare_expm1')(((count + 31) // 32,), (32,),
                                                (delta, prepared, np.int32(count)))
            prep['enqueued'] = True
            self.engine.stream.synchronize()
            prep['synchronized'] = True
        except BaseException as error:
            prep['unknown_requested'] = not prep['synchronized']
            prep['error'] = repr(error)
            error.precomputed_expm1_progress = self.snapshot()
            raise
        forward = self._event('joint_forward')
        try:
            # The original delta remains position 5 for the E128 edge bank.
            candidate_args = (*arguments[:6], prepared, *arguments[6:])
            value = _candidate_kernel('joint_forward')(
                grid, block, candidate_args, *rest, **kwargs)
            forward['enqueued'] = True
            return value
        except BaseException as error:
            forward['unknown_requested'] = True
            forward['error'] = repr(error)
            error.precomputed_expm1_progress = self.snapshot()
            raise

    def factory(self, original):
        def get(name):
            kernel = original(name) if name != 'joint_forward' else None
            if name != 'joint_forward':
                return kernel

            def launch(grid, block, arguments, *rest, **kwargs):
                if int(arguments[-2]):
                    return original(name)(grid, block, arguments, *rest, **kwargs)
                return self._forward(grid, block, arguments, *rest, **kwargs)
            return launch
        return get

    def solve(self, original):
        def call(*args, **kwargs):
            start = len(self.events)
            try:
                value = original(*args, **kwargs)
            except BaseException as error:
                progress = getattr(error, 'response_progress', {})
                completed = progress.get('forward_completion_class') == 'known_complete'
                for event in self.events[start:]:
                    if event['kind'] == 'joint_forward':
                        event['synchronized'] = completed and event['enqueued']
                        event['unknown_requested'] = not event['synchronized']
                error.precomputed_expm1_progress = self.snapshot()
                raise
            for event in self.events[start:]:
                if event['kind'] == 'joint_forward':
                    event['synchronized'] = True
            return value
        return call


@contextlib.contextmanager
def precomputed_expm1(engine):
    """Restore the exact factory and instance method even after CUDA faults."""
    if not isinstance(engine, neural.CudaJointAdaptiveEngine):
        raise TypeError('E130 requires a CudaJointAdaptiveEngine')
    original_factory = neural._kernel
    direct_solve = 'solve' in vars(engine)
    original_solve = vars(engine).get('solve')
    adapter = PrecomputedExpm1(engine)
    try:
        neural._kernel = adapter.factory(original_factory)
        engine.solve = adapter.solve(engine.solve)
        yield adapter
    finally:
        active = sys.exc_info()[1]
        neural._kernel = original_factory
        try:
            if direct_solve:
                engine.solve = original_solve
            else:
                del engine.solve
        except BaseException as secondary:
            if active is None:
                raise
            active.add_note('Secondary E130 solve restoration: ' + repr(secondary))
