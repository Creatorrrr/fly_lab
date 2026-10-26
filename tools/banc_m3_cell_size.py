"""A separately identified cell-size physiology hypothesis on existing cells."""

import copy

import numpy as np

from tools.banc_m3_core_adaptation import CoreAdaptationNetwork
from tools.probe_c_banc_m3_core_rhythm import CoreTrial, source_matrix
from tools.verify_c_banc_observation import exact


def patch_parameters(original_a, original_threshold, ids, a, threshold):
    ids = np.asarray(ids)
    a = np.asarray(a, dtype=np.float32)
    threshold = np.asarray(threshold, dtype=np.float32)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu'
            or len(np.unique(ids)) != len(ids)
            or np.any((ids < 0) | (ids >= len(original_a)))
            or a.shape != ids.shape or threshold.shape != ids.shape
            or not np.isfinite(a).all() or not np.isfinite(threshold).all()
            or np.any(a <= 0) or np.any(threshold <= 0)):
        raise ValueError('Unique existing cells and aligned positive physiology required')
    new_a = np.array(original_a, dtype=np.float32, copy=True)
    new_threshold = np.array(original_threshold, dtype=np.float32, copy=True)
    if new_a.shape != new_threshold.shape or new_a.ndim != 1:
        raise ValueError('Matching complete source physiology required')
    new_a[ids] = a
    new_threshold[ids] = threshold
    return new_a, new_threshold


class CellSizeTrial(CoreTrial):
    schema = 'flylab.m3-cell-size-walking.v1'

    def __init__(self, graph, core_source, ids, a, threshold):
        base = CoreTrial.from_checkpoint(graph, core_source)
        self.__dict__.update(base.__dict__)
        old = self.network
        saved = old.snapshot()
        try:
            new_a, new_threshold = patch_parameters(old.a.cpu().numpy(),
                old.threshold.cpu().numpy(), ids, a, threshold)
            candidate = CoreAdaptationNetwork(source_matrix(old),
                old.tau.cpu().numpy().copy(), new_a, new_threshold,
                old.cap.cpu().numpy().copy(), adaptation_gain=old.adaptation_gain,
                adaptation_tau_s=old.adaptation_tau_s, dt=old.dt,
                device=old.device, cuda_implementation=old.cuda_implementation,
                capture_steps=old.interval_steps, core_ids=old.core_ids,
                core_gain=old.core_gain)
            self.cell_size = {'schema': 'flylab.m3-cell-size-patch.v1',
                'ids': np.array(ids, dtype=np.int64),
                'a': np.array(a, dtype=np.float32),
                'threshold': np.array(threshold, dtype=np.float32),
                'source_core_identity': old.identity,
                'source_core_adaptation': copy.deepcopy(saved['core_adaptation'])}
            saved['model_hash'] = candidate.identity
            saved['core_adaptation'] = candidate.snapshot()['core_adaptation']
            candidate.restore(saved)
            self.network = candidate
        except Exception:
            self.close()
            raise

    def snapshot(self):
        return {**super().snapshot(), 'cell_size': copy.deepcopy(self.cell_size)}

    @classmethod
    def from_checkpoint(cls, graph, saved):
        if saved.get('schema') != cls.schema:
            raise ValueError('Explicit cell-size candidate checkpoint required')
        core = copy.deepcopy(saved)
        patch = core.pop('cell_size')
        if patch.get('schema') != 'flylab.m3-cell-size-patch.v1':
            raise ValueError('Explicit source physiology patch required')
        core['schema'] = CoreTrial.schema
        core['network']['model_hash'] = patch['source_core_identity']
        core['network']['core_adaptation'] = patch['source_core_adaptation']
        trial = cls(graph, core, patch['ids'], patch['a'], patch['threshold'])
        try:
            exact(saved, trial.snapshot(), 'Complete cell-size trial state')
        except Exception:
            trial.close()
            raise
        return trial
