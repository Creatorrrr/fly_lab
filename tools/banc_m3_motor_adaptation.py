"""Bounded adaptation gains on existing BANC motor neurons.

The sparse graph, RK4 integration, and muscle/body interface remain native.
Gain values are research hypotheses, not measured single-cell physiology.
"""

import copy
import hashlib

import numpy as np

from tools.banc_m3_core_adaptation import CoreAdaptationNetwork
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.verify_c_banc_observation import exact

MOTOR_SCHEMA = 'flylab.m3-motor-adaptation.v1'
TRIAL_PATCH_SCHEMA = 'flylab.m3-muscle-motor-adaptation-patch.v1'


def checked_motor_gains(motor_ids, gains, n):
    ids = np.asarray(motor_ids)
    raw = np.asarray(gains)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids)
            or np.any((ids < 0) | (ids >= n)) or raw.ndim != 1
            or raw.shape != ids.shape or raw.dtype.kind not in 'fiu'):
        raise ValueError('Aligned unique existing motor IDs and numeric gains required')
    with np.errstate(over='ignore', invalid='ignore'):
        values = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 8)):
        raise ValueError('Finite float32 motor gains within [0,8] required')
    return ids.astype(np.int64, copy=True), values.copy()


class MotorAdaptationNetwork(CoreAdaptationNetwork):
    """Use the original per-cell RK4 derivative and fused CUDA kernel."""

    def __init__(self, *args, core_ids, core_gain, motor_ids, motor_gains, **kwargs):
        self.motor_ids, self.motor_gains = checked_motor_gains(
            motor_ids, motor_gains, args[0].shape[0])
        self._motor_gains_ready = True
        super().__init__(*args, core_ids=core_ids, core_gain=core_gain, **kwargs)
        self.source_core_identity = self.identity
        self.identity = hashlib.sha256(
            self.source_core_identity.encode('ascii') + MOTOR_SCHEMA.encode('ascii')
            + self.motor_ids.tobytes() + self.motor_gains.tobytes()).hexdigest()

    def _init_gains(self):
        if not hasattr(self, '_cell_gain'):
            gains = np.full(self.n, self.adaptation_gain, dtype=np.float32)
            gains[self.core_ids] = np.float32(self.core_gain)
            gains[self.motor_ids] = self.motor_gains
            # Called during AdaptiveRateNetwork.__init__ CUDA warmup. This
            # tensor is never replaced after the CUDA graph captures its pointer.
            self._cell_gain = self.torch.as_tensor(gains, device=self.device)

    def snapshot(self):
        return {**super().snapshot(), 'motor_adaptation': {
            'schema': MOTOR_SCHEMA, 'source_core_identity': self.source_core_identity,
            'ids': self.motor_ids.copy(), 'gains': self.motor_gains.copy()}}

    def restore(self, saved):
        patch = saved.get('motor_adaptation')
        if not isinstance(patch, dict) or set(patch) != {
                'schema', 'source_core_identity', 'ids', 'gains'}:
            raise ValueError('Complete motor adaptation metadata required')
        if (not isinstance(patch['ids'], np.ndarray)
                or patch['ids'].dtype != np.int64
                or not isinstance(patch['gains'], np.ndarray)
                or patch['gains'].dtype != np.float32):
            raise ValueError('Native integer IDs and float32 gain metadata required')
        try:
            ids, gains = checked_motor_gains(patch['ids'], patch['gains'], self.n)
        except (TypeError, ValueError) as error:
            raise ValueError('Invalid saved motor adaptation metadata') from error
        if (patch['schema'] != MOTOR_SCHEMA
                or patch['source_core_identity'] != self.source_core_identity
                or not np.array_equal(ids, self.motor_ids)
                or gains.dtype != np.float32
                or not np.array_equal(gains.view(np.uint32), self.motor_gains.view(np.uint32))
                or saved.get('model_hash') != self.identity):
            raise ValueError('Motor adaptation identity mismatch')
        # Core and rate restores validate their own metadata and all five
        # state fields. Their validation precedes any tensor mutation.
        super().restore(saved)


def map_muscle_gains(motor_ids, pool, membership, muscles, muscle_gains):
    """Map 96 named muscle rows to exactly one row per existing MN column."""
    ids = np.asarray(motor_ids)
    matrix = np.asarray(pool)
    belongs = np.asarray(membership)
    raw = np.asarray(muscle_gains)
    if (ids.shape != (391,) or ids.dtype.kind not in 'iu'
            or len(np.unique(ids)) != 391 or matrix.shape != (96, 391)
            or belongs.shape != (391,) or belongs.dtype.kind not in 'iu'
            or len(muscles) != 96 or raw.shape != (96,)):
        raise ValueError('Fixed 391 MN / 96 muscle single-membership mapping required')
    if (np.any((belongs < 0) | (belongs >= 96))
            or not np.isfinite(matrix).all() or np.any(matrix < 0)
            or np.any(np.count_nonzero(matrix, axis=0) != 1)
            or not np.array_equal(np.argmax(matrix, axis=0), belongs)):
        raise ValueError('Every native MN must have one anatomical muscle membership')
    seen = []
    for row, muscle in enumerate(muscles):
        columns = muscle.get('columns')
        if (not isinstance(columns, list) or not isinstance(muscle.get('leg'), str)
                or not isinstance(muscle.get('muscle'), str)):
            raise TypeError('Named native muscle columns required')
        expected = np.flatnonzero(belongs == row).tolist()
        if columns != expected:
            raise ValueError('Muscle column provenance disagrees with pool')
        seen.extend(columns)
    if sorted(seen) != list(range(391)):
        raise ValueError('Muscle columns must partition the 391 MN roster')
    _, values = checked_motor_gains(np.arange(96), raw, 96)
    return ids.astype(np.int64, copy=True), values[belongs].copy(), values


class MotorAdaptationTrial(IntrinsicRecruitmentTrial):
    """Research wrapper that keeps the original native body and adapter."""

    schema = 'flylab.m3-motor-adaptation-walking.v1'

    def __init__(self, base, muscle_gains, muscles, membership):
        if not isinstance(base, IntrinsicRecruitmentTrial):
            raise TypeError('Original intrinsic recruitment layer required')
        self.__dict__.update(base.__dict__)
        old = self.network
        try:
            if type(old) is not CoreAdaptationNetwork:
                raise ValueError('Unmodified intrinsic source core layer required')
            calibration = self.adapter.calibration
            ids, gains, muscle_values = map_muscle_gains(
                calibration['motor_ids'], calibration['pool'], membership,
                muscles, muscle_gains)
            if not np.array_equal(ids, self.adapter.motor_ids):
                raise ValueError('Adapter motor roster changed')
            if [(item['leg'], item['muscle']) for item in muscles] != [
                    (item['leg'], item['muscle']) for item in self.adapter.groups]:
                raise ValueError('Original 96 named muscle order changed')
            motor_roster = [self.graph.nodes[int(i)]['id'] for i in ids]
            if len(set(motor_roster)) != 391:
                raise ValueError('Unique source root-ID provenance required')
            saved = old.snapshot()
            implementation = old.cuda_implementation
            if old.device == 'cpu' and implementation == 'torch-cpu-csr':
                implementation = 'reference'  # CPU path remains torch-cpu-csr.
            candidate = MotorAdaptationNetwork(source_matrix(old),
                *(getattr(old, key).cpu().numpy().copy()
                  for key in ('tau', 'a', 'threshold', 'cap')),
                adaptation_gain=old.adaptation_gain,
                adaptation_tau_s=old.adaptation_tau_s, dt=old.dt,
                device=old.device, cuda_implementation=implementation,
                capture_steps=old.interval_steps, core_ids=old.core_ids,
                core_gain=old.core_gain, motor_ids=ids, motor_gains=gains)
            if candidate.source_core_identity != old.identity:
                raise ValueError('Source core identity changed during motor patch')
            self.motor_adaptation_patch = {
                'schema': TRIAL_PATCH_SCHEMA,
                'muscle_gains': muscle_values.copy(),
                'muscles': copy.deepcopy(muscles),
                'membership': np.asarray(membership, dtype=np.int64).copy(),
                'motor_ids': ids.copy(), 'motor_gains': gains.copy(),
                'motor_roster': motor_roster,
                'source_neural_identity': old.identity,
                'source_cuda_implementation': old.cuda_implementation,
                'source_core_adaptation': copy.deepcopy(saved['core_adaptation']),
                'source_adapter_identity': self.adapter.identity,
            }
            saved['model_hash'] = candidate.identity
            saved['motor_adaptation'] = candidate.snapshot()['motor_adaptation']
            candidate.restore(saved)
            self.network = candidate
        except Exception as error:
            base.closed = True  # The copied wrapper owns this same body now.
            try:
                self.close()
            except Exception as close_error:  # noqa: BLE001 - retain original failure
                error.add_note('Body cleanup failed: '+repr(close_error))
            raise

    def snapshot(self):
        return {**super().snapshot(),
                'motor_adaptation_patch': copy.deepcopy(self.motor_adaptation_patch)}

    @classmethod
    def from_checkpoint(cls, graph, saved):
        if saved.get('schema') != cls.schema:
            raise ValueError('Explicit motor adaptation trial checkpoint required')
        patch = saved.get('motor_adaptation_patch')
        if not isinstance(patch, dict) or patch.get('schema') != TRIAL_PATCH_SCHEMA:
            raise ValueError('Complete muscle gain provenance required')
        source = copy.deepcopy(saved)
        source.pop('motor_adaptation_patch')
        source['schema'] = IntrinsicRecruitmentTrial.schema
        source['network'].pop('motor_adaptation')
        source['network']['model_hash'] = patch['source_neural_identity']
        source['network']['core_adaptation'] = copy.deepcopy(patch['source_core_adaptation'])
        base = IntrinsicRecruitmentTrial.from_checkpoint(graph, source)
        try:
            if base.adapter.identity != patch['source_adapter_identity']:
                raise ValueError('Original muscle adapter identity changed')
            trial = cls(base, patch['muscle_gains'], patch['muscles'],
                        patch['membership'])
            exact(saved, trial.snapshot(), 'Whole motor adaptation trial checkpoint')
            return trial
        except Exception as error:
            try:
                if not getattr(base, 'closed', False):
                    base.close()
            except Exception as close_error:  # noqa: BLE001 - retain original failure
                error.add_note('Body cleanup failed: '+repr(close_error))
            raise
