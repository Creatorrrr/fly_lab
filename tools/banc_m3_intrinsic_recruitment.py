"""Checkpointed native neural timescale hypotheses for existing source cells."""

import copy

import numpy as np

from tools.banc_m3_core_adaptation import CoreAdaptationNetwork
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.probe_c_banc_m3_core_rhythm import source_matrix
from tools.verify_c_banc_observation import exact


def patch_tau(original, ids, values):
    original = np.asarray(original)
    ids = np.asarray(ids)
    values = np.asarray(values, dtype=np.float32)
    if (original.ndim != 1 or not np.isfinite(original).all() or np.any(original <= 0)
            or ids.ndim != 1 or ids.dtype.kind not in 'iu'
            or len(np.unique(ids)) != len(ids) or values.shape != ids.shape
            or np.any((ids < 0) | (ids >= len(original)))
            or not np.isfinite(values).all() or np.any(values <= 0)):
        raise ValueError('Unique existing cell IDs and finite positive native time constants required')
    result = np.array(original, dtype=np.float32, copy=True)
    result[ids] = values
    return result


class IntrinsicRecruitmentTrial(RecruitmentTrial):
    schema = 'flylab.m3-intrinsic-recruitment-walking.v1'

    def __init__(self, base, ids, tau):
        self.__dict__.update(base.__dict__)
        old = self.network
        saved = old.snapshot()
        try:
            original_tau = old.tau.cpu().numpy().copy()
            changed_tau = patch_tau(original_tau, ids, tau)
            candidate = CoreAdaptationNetwork(source_matrix(old), changed_tau,
                *(getattr(old, key).cpu().numpy().copy() for key in ('a', 'threshold', 'cap')),
                adaptation_gain=old.adaptation_gain, adaptation_tau_s=old.adaptation_tau_s,
                dt=old.dt, device=old.device, cuda_implementation=old.cuda_implementation,
                capture_steps=old.interval_steps, core_ids=old.core_ids, core_gain=old.core_gain)
            self.intrinsic_kinetics = {'schema': 'flylab.m3-source-time-constant-patch.v1',
                'ids': np.array(ids, dtype=np.int64), 'tau': np.array(tau, dtype=np.float32),
                'source_tau': original_tau[np.asarray(ids)], 'source_neural_identity': old.identity,
                'source_core_adaptation': copy.deepcopy(saved['core_adaptation'])}
            saved['model_hash'] = candidate.identity
            saved['core_adaptation'] = candidate.snapshot()['core_adaptation']
            candidate.restore(saved)
            self.network = candidate
        except Exception:
            self.close()
            raise

    def snapshot(self):
        return {**super().snapshot(), 'intrinsic_kinetics': copy.deepcopy(self.intrinsic_kinetics)}

    @classmethod
    def from_checkpoint(cls, graph, saved):
        if saved.get('schema') != cls.schema:
            raise ValueError('Explicit intrinsic recruitment checkpoint required')
        source = copy.deepcopy(saved)
        patch = source.pop('intrinsic_kinetics')
        if patch.get('schema') != 'flylab.m3-source-time-constant-patch.v1':
            raise ValueError('Explicit existing-cell kinetics provenance required')
        source['schema'] = RecruitmentTrial.schema
        source['network']['model_hash'] = patch['source_neural_identity']
        source['network']['core_adaptation'] = patch['source_core_adaptation']
        base = RecruitmentTrial.from_checkpoint(graph, source)
        try:
            exact(base.network.tau.cpu().numpy()[patch['ids']], patch['source_tau'], 'Original cell time constants')
            trial = cls(base, patch['ids'], patch['tau'])
            exact(saved, trial.snapshot(), 'Whole intrinsic recruitment checkpoint')
        except Exception:
            base.close()
            raise
        return trial
