"""Reversible efficacy hypotheses on existing BANC synapses only."""

import copy

import numpy as np

from tools.banc_m3_core_adaptation import CoreAdaptationNetwork
from tools.probe_c_banc_m3_core_rhythm import CoreTrial, source_matrix
from tools.verify_c_banc_observation import exact


def patch_weights(source, edge_ids, source_values, values):
    """Preserve CSR topology, source signs and every unselected source value."""
    ids = np.asarray(edge_ids)
    original = np.asarray(source_values, dtype=np.float32)
    changed = np.asarray(values, dtype=np.float32)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu'
            or len(np.unique(ids)) != len(ids)
            or np.any((ids < 0) | (ids >= len(source.data)))
            or original.shape != ids.shape or changed.shape != ids.shape
            or not np.isfinite(original).all() or not np.isfinite(changed).all()
            or np.any(original == 0) or np.any(changed == 0)
            or not np.array_equal(np.signbit(original), np.signbit(changed))):
        raise ValueError('Unique existing nonzero synapses and finite same-sign efficacy required')
    if not np.array_equal(source.data[ids], original):
        raise ValueError('Source synaptic values differ from the calibrated source')
    result = source.copy()
    result.data[ids] = changed
    return result


class EfficacyTrial(CoreTrial):
    schema = 'flylab.m3-synaptic-efficacy-walking.v1'

    def __init__(self, graph, core_source, edge_ids, source_values, values, *, zero_adaptation_ids=()):
        base = CoreTrial.from_checkpoint(graph, core_source)
        self.__dict__.update(base.__dict__)
        old = self.network
        saved = old.snapshot()
        try:
            weights = patch_weights(source_matrix(old), edge_ids, source_values, values)
            zero_ids = np.asarray(zero_adaptation_ids)
            if zero_ids.ndim != 1:
                raise ValueError('One-dimensional additional zero-adaptation IDs required')
            if not len(zero_ids):
                zero_ids = np.array([], dtype=np.int64)
            if (zero_ids.ndim != 1 or zero_ids.dtype.kind not in 'iu'
                    or len(np.unique(zero_ids)) != len(zero_ids)
                    or np.any((zero_ids < 0) | (zero_ids >= old.n))
                    or (len(zero_ids) and old.core_gain != 0)):
                raise ValueError('Unique existing zero-adaptation cells and source core gain0 required')
            selected = np.union1d(old.core_ids, zero_ids) if len(zero_ids) else old.core_ids
            candidate = CoreAdaptationNetwork(weights,
                *(getattr(old, key).cpu().numpy().copy() for key in ('tau', 'a', 'threshold', 'cap')),
                adaptation_gain=old.adaptation_gain, adaptation_tau_s=old.adaptation_tau_s,
                dt=old.dt, device=old.device, cuda_implementation=old.cuda_implementation,
                capture_steps=old.interval_steps, core_ids=selected, core_gain=old.core_gain)
            self.efficacy = {'schema': 'flylab.m3-source-efficacy-patch.v1',
                'edge_ids': np.array(edge_ids, dtype=np.int64),
                'source_values': np.array(source_values, dtype=np.float32),
                'values': np.array(values, dtype=np.float32),
                'zero_adaptation_ids': zero_ids.astype(np.int64, copy=True),
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
        return {**super().snapshot(), 'efficacy': copy.deepcopy(self.efficacy)}

    @classmethod
    def from_checkpoint(cls, graph, saved):
        if saved.get('schema') != cls.schema:
            raise ValueError('Explicit source-efficacy checkpoint required')
        core = copy.deepcopy(saved)
        patch = core.pop('efficacy')
        if patch.get('schema') != 'flylab.m3-source-efficacy-patch.v1':
            raise ValueError('Source efficacy metadata required')
        core['schema'] = CoreTrial.schema
        core['network']['model_hash'] = patch['source_core_identity']
        core['network']['core_adaptation'] = patch['source_core_adaptation']
        trial = cls(graph, core, patch['edge_ids'], patch['source_values'], patch['values'],
                    zero_adaptation_ids=patch['zero_adaptation_ids'])
        try:
            exact(saved, trial.snapshot(), 'Complete synaptic-efficacy trial state')
        except Exception:
            trial.close()
            raise
        return trial
