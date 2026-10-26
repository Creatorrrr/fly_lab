"""Per-muscle adaptation timescales on the existing full-BANC motor layer.

The existing CSR, membrane constants, RK4 order and body interface are kept.
This is a research hypothesis, not measured single-cell physiology.
"""

from __future__ import annotations

import copy
import hashlib
import math

import numpy as np

from flylab.c.adaptive_fused import SOURCE, ExactRKMixin
from tools.banc_m3_motor_adaptation import (
    MotorAdaptationNetwork,
    MotorAdaptationTrial,
    map_muscle_gains,
)
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.verify_c_banc_observation import exact

TIMESCALE_SCHEMA = 'flylab.m3-motor-timescale.v1'
TRIAL_SCHEMA = 'flylab.m3-muscle-motor-timescale-patch.v1'
LOG_TAU_LOWER = -math.log(3)
LOG_TAU_UPPER = math.log(10)


def checked_motor_log_tau(motor_ids, motor_log_tau, n):
    """Require the ordered native IDs and requested float64 log seconds."""
    ids = np.asarray(motor_ids)
    ell = np.asarray(motor_log_tau)
    if (ids.ndim != 1 or ids.dtype != np.int64
            or len(np.unique(ids)) != len(ids)
            or np.any((ids < 0) | (ids >= n))
            or ell.shape != ids.shape or ell.dtype != np.float64
            or not np.isfinite(ell).all()
            or np.any(ell < LOG_TAU_LOWER) or np.any(ell > LOG_TAU_UPPER)):
        raise ValueError('Ordered native int64 IDs and bounded float64 motor log tau required')
    return ids.copy(), ell.copy()


def canonical_timescales(ids, ell, n):
    """Construct the exact frozen float64-to-float32 native conversion."""
    tau64 = np.full(n, .3, dtype=np.float64)
    tau64[ids] = np.float64(.3)*np.exp(ell)
    tau32 = tau64.astype(np.float32)
    inverse32 = (np.float64(1.)/tau64).astype(np.float32)
    if (not np.isfinite(tau64).all() or not np.isfinite(tau32).all()
            or not np.isfinite(inverse32).all() or np.any(tau32 <= 0)):
        raise ValueError('Finite positive canonical native adaptation timescales required')
    return tau64, tau32, inverse32


def _timescale_source(fuse_tanh):
    """Own guarded pointwise kernel transformation; no old module mutation."""
    signature = 'float gain, float inverse_adaptation_tau'
    gain = 'mul_exact(gain, rate[i])'
    inverse = '), inverse_adaptation_tau);'
    tanh = 'mul_exact(cap[i], activation[i])'
    if (SOURCE.count(signature) != 2 or SOURCE.count(gain) != 2
            or SOURCE.count(inverse) != 2 or SOURCE.count(tanh) != 2):
        raise RuntimeError('Frozen native RK4 source transformation points changed')
    result = SOURCE.replace(signature,
        'const float* cell_gain, const float* cell_inverse_tau, '
        'float gain, float inverse_adaptation_tau')
    result = result.replace(gain, 'mul_exact(cell_gain[i], rate[i])')
    result = result.replace(inverse, '), cell_inverse_tau[i]);')
    if fuse_tanh:
        result = result.replace(tanh, 'mul_exact(cap[i], matching_tanh(activation[i]))')
    if (result.count('mul_exact(cell_gain[i], rate[i])') != 2
            or result.count('), cell_inverse_tau[i]);') != 2):
        raise RuntimeError('Native timescale expression rewrite incomplete')
    return result


def _same_bits(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and left.tobytes(order='C') == right.tobytes(order='C'))


class MotorTimescaleNetwork(MotorAdaptationNetwork):
    """Keep E123 motor gains and replace only each cell's adaptation timescale."""

    def __init__(self, *args, core_ids, core_gain, motor_ids, motor_gains,
                 motor_log_tau, **kwargs):
        if kwargs.get('adaptation_tau_s') != .3:
            raise ValueError('E138 requires the original 0.3 s adaptation base')
        ids, ell = checked_motor_log_tau(motor_ids, motor_log_tau, args[0].shape[0])
        self.motor_log_tau = ell
        self._full_tau64, self._full_tau32, self._full_inverse32 = (
            canonical_timescales(ids, ell, args[0].shape[0]))
        if kwargs.get('dt', .0001) > float(np.min(self._full_tau32))/50:
            raise ValueError('Integration step must resolve every adaptation time constant')
        super().__init__(*args, core_ids=core_ids, core_gain=core_gain,
            motor_ids=ids, motor_gains=motor_gains, **kwargs)
        self._init_cell_timescale()
        self._cell_tau_ptr = self._cell_adaptation_tau.data_ptr()
        self._cell_inverse_ptr = self._cell_inverse_tau.data_ptr()
        self.source_motor_identity = self.identity
        checksum = hashlib.sha256(self.source_motor_identity.encode('ascii')
            + TIMESCALE_SCHEMA.encode('ascii'))
        for value in (self.motor_ids, self.motor_log_tau, self._full_tau64,
                      self._full_tau32, self._full_inverse32):
            checksum.update(np.ascontiguousarray(value).tobytes())
        self.identity = checksum.hexdigest()

    def _init_cell_timescale(self):
        if not hasattr(self, '_cell_adaptation_tau'):
            self._cell_adaptation_tau = self.torch.as_tensor(
                self._full_tau32.copy(), device=self.device)
            self._cell_inverse_tau = self.torch.as_tensor(
                self._full_inverse32.copy(), device=self.device)

    def _init_fusion(self):
        self._init_gains()
        self._init_cell_timescale()  # Stable pointers before warmup/capture.
        source = _timescale_source(self.fuse_tanh)
        self._fused_kernels = {name: self._cupy.RawKernel(
            source, name, options=('--fmad=false',))
            for name in ('csr_input', 'stage', 'finish')}
        for kernel in self._fused_kernels.values():
            kernel.compile()
        self._fused_buffers = [self.torch.empty_like(self.rate) for _ in range(11)]

    def _launch_fused(self, name, tensors, scalars=()):
        if name in ('stage', 'finish'):
            tensors = (*tensors, self._cell_gain, self._cell_inverse_tau)
        return ExactRKMixin._launch_fused(self, name, tensors, scalars)

    def _derivative_pair(self, rate, adaptation):
        self._init_gains()
        self._init_cell_timescale()
        dr, _ = super()._derivative_pair(rate, adaptation)
        return dr, (self._cell_gain*rate-adaptation)/self._cell_adaptation_tau

    def _assert_actual_timescales(self):
        if (self._cell_adaptation_tau.data_ptr() != self._cell_tau_ptr
                or self._cell_inverse_tau.data_ptr() != self._cell_inverse_ptr):
            raise ValueError('Captured adaptation timescale tensor pointer changed')
        actual_tau = self._cell_adaptation_tau.detach().cpu().numpy().copy()
        actual_inverse = self._cell_inverse_tau.detach().cpu().numpy().copy()
        if (not _same_bits(actual_tau, self._full_tau32)
                or not _same_bits(actual_inverse, self._full_inverse32)):
            raise ValueError('Captured adaptation timescale tensor values changed')

    def snapshot(self):
        self._assert_actual_timescales()
        return {**super().snapshot(), 'motor_timescale': {
            'schema': TIMESCALE_SCHEMA,
            'source_motor_identity': self.source_motor_identity,
            'ids': self.motor_ids.copy(),
            'log_tau': self.motor_log_tau.copy(),
            'tau64': self._full_tau64.copy(),
            'tau32': self._full_tau32.copy(),
            'inverse32': self._full_inverse32.copy()}}

    def restore(self, saved):
        patch = saved.get('motor_timescale')
        if not isinstance(patch, dict) or set(patch) != {
                'schema', 'source_motor_identity', 'ids', 'log_tau',
                'tau64', 'tau32', 'inverse32'}:
            raise ValueError('Complete motor timescale metadata required')
        arrays = {'ids': (self.motor_ids, np.int64),
            'log_tau': (self.motor_log_tau, np.float64),
            'tau64': (self._full_tau64, np.float64),
            'tau32': (self._full_tau32, np.float32),
            'inverse32': (self._full_inverse32, np.float32)}
        if (patch['schema'] != TIMESCALE_SCHEMA
                or patch['source_motor_identity'] != self.source_motor_identity
                or saved.get('model_hash') != self.identity
                or any(not isinstance(patch[key], np.ndarray)
                    or patch[key].dtype != dtype
                    or not _same_bits(patch[key], expected)
                    for key, (expected, dtype) in arrays.items())):
            raise ValueError('Motor timescale identity or native bytes mismatch')
        self._assert_actual_timescales()
        super().restore(saved)


def map_muscle_log_tau(motor_ids, pool, membership, muscles, muscle_log_tau):
    """Map exact named global muscle rows to ordered native MN rows."""
    ids, _, _ = map_muscle_gains(motor_ids, pool, membership, muscles,
                                 np.zeros(96, dtype=np.float32))
    ell = np.asarray(muscle_log_tau)
    if (ell.shape != (96,) or ell.dtype != np.float64
            or not np.isfinite(ell).all()
            or np.any(ell < LOG_TAU_LOWER) or np.any(ell > LOG_TAU_UPPER)):
        raise ValueError('Ordered bounded float64 96-muscle log tau required')
    motor_ell = ell[np.asarray(membership, dtype=np.int64)]
    return ids, motor_ell.copy(), ell.copy()


class MotorTimescaleTrial(MotorAdaptationTrial):
    """Research wrapper above E123, retaining its body and adapter state."""

    schema = 'flylab.m3-motor-timescale-walking.v1'

    def __init__(self, base, muscle_log_tau, muscles, membership):
        if not isinstance(base, MotorAdaptationTrial):
            raise TypeError('Existing motor adaptation trial layer required')
        self.__dict__.update(base.__dict__)
        old = self.network
        try:
            if type(old) is not MotorAdaptationNetwork:
                raise ValueError('Unmodified E123 motor adaptation layer required')
            calibration = self.adapter.calibration
            ids, motor_ell, muscle_ell = map_muscle_log_tau(
                calibration['motor_ids'], calibration['pool'], membership,
                muscles, muscle_log_tau)
            if not np.array_equal(ids, old.motor_ids):
                raise ValueError('Native and anatomical motor IDs disagree')
            if [(item['leg'], item['muscle']) for item in muscles] != [
                    (item['leg'], item['muscle']) for item in self.adapter.groups]:
                raise ValueError('Original named 96-muscle order changed')
            roster = [self.graph.nodes[int(i)]['id'] for i in ids]
            if len(set(roster)) != len(ids):
                raise ValueError('Unique original MN root roster required')
            saved = old.snapshot()
            implementation = old.cuda_implementation
            if old.device == 'cpu' and implementation == 'torch-cpu-csr':
                implementation = 'reference'
            candidate = MotorTimescaleNetwork(source_matrix(old),
                *(getattr(old, key).cpu().numpy().copy()
                  for key in ('tau', 'a', 'threshold', 'cap')),
                adaptation_gain=old.adaptation_gain,
                adaptation_tau_s=old.adaptation_tau_s, dt=old.dt,
                device=old.device, cuda_implementation=implementation,
                capture_steps=old.interval_steps, core_ids=old.core_ids,
                core_gain=old.core_gain, motor_ids=old.motor_ids,
                motor_gains=old.motor_gains, motor_log_tau=motor_ell)
            if candidate.source_motor_identity != old.identity:
                raise ValueError('Source motor identity changed during timescale patch')
            self.motor_timescale_patch = {'schema': TRIAL_SCHEMA,
                'muscle_log_tau': muscle_ell.copy(),
                'muscles': copy.deepcopy(muscles),
                'membership': np.asarray(membership, dtype=np.int64).copy(),
                'motor_ids': ids.copy(), 'motor_log_tau': motor_ell.copy(),
                'motor_tau64': candidate._full_tau64[ids].copy(),
                'motor_tau32': candidate._full_tau32[ids].copy(),
                'motor_inverse32': candidate._full_inverse32[ids].copy(),
                'motor_roster': roster,
                'source_motor_identity': old.identity,
                'source_adapter_identity': self.adapter.identity}
            saved['model_hash'] = candidate.identity
            saved['motor_timescale'] = candidate.snapshot()['motor_timescale']
            candidate.restore(saved)
            self.network = candidate
        except Exception as error:
            base.closed = True  # This wrapper owns the shared body after transfer.
            try:
                self.close()
            except Exception as close_error:  # noqa: BLE001
                error.add_note('Body cleanup failed: '+repr(close_error))
            raise

    def snapshot(self):
        return {**super().snapshot(),
                'motor_timescale_patch': copy.deepcopy(self.motor_timescale_patch)}

    @classmethod
    def from_checkpoint(cls, graph, saved):
        if saved.get('schema') != cls.schema:
            raise ValueError('Explicit motor timescale trial checkpoint required')
        patch = saved.get('motor_timescale_patch')
        if not isinstance(patch, dict) or patch.get('schema') != TRIAL_SCHEMA:
            raise ValueError('Complete muscle timescale provenance required')
        source = copy.deepcopy(saved)
        source.pop('motor_timescale_patch')
        source['schema'] = MotorAdaptationTrial.schema
        source['network'].pop('motor_timescale')
        source['network']['model_hash'] = patch['source_motor_identity']
        base = MotorAdaptationTrial.from_checkpoint(graph, source)
        try:
            if base.adapter.identity != patch['source_adapter_identity']:
                raise ValueError('Original anatomical adapter identity changed')
            trial = cls(base, patch['muscle_log_tau'], patch['muscles'],
                        patch['membership'])
            exact(saved, trial.snapshot(), 'Whole motor timescale trial checkpoint')
            return trial
        except Exception as error:
            try:
                if not getattr(base, 'closed', False):
                    base.close()
            except Exception as close_error:  # noqa: BLE001
                error.add_note('Body cleanup failed: '+repr(close_error))
            raise
