"""Anatomical rate-to-body interface for source-constrained recruitment trials."""

import copy
import hashlib
import json
import math

import numpy as np

from tools.banc_m3_synaptic_efficacy import EfficacyTrial
from tools.verify_c_banc_observation import exact


class AnatomicalRecruitmentAdapter:
    schema = 'flylab.m3-anatomical-recruitment-adapter.v1'

    def __init__(self, original, calibration, groups):
        self.original = original
        self.calibration = {key: np.array(calibration[key], copy=True)
            for key in ('moment', 'rest', 'pool', 'motor_ids', 'ltm')}
        self.groups = copy.deepcopy(groups)
        d = self.calibration
        if (d['moment'].shape != (42, 96) or d['rest'].shape != (42,)
                or d['pool'].shape != (96, 391) or d['ltm'].shape != (6,)
                or len(groups) != 96 or any(not np.isfinite(x).all() for x in d.values())
                or np.any(d['pool'] < 0) or np.max(np.abs(d['rest'])) > 10):
            raise ValueError('Finite fixed anatomical recruitment interface required')
        np.testing.assert_array_equal(d['motor_ids'], original.motor_ids)
        np.testing.assert_allclose(d['pool'].sum(axis=1), 1, rtol=0, atol=1e-15)
        self.identity = hashlib.sha256(json.dumps({'schema': self.schema,
            'original': original.identity, 'calibration': {k: v.tolist() for k, v in d.items()},
            'groups': groups, 'moment_scale': .02, 'muscle_tau_s': .01,
            'tendon_max': .2, 'tendon_rate_scale': 10}, sort_keys=True).encode()).hexdigest()
        self.activation = np.zeros(96)
        self.last_motor = []

    def __getattr__(self, key):
        return getattr(self.original, key)

    @property
    def tendon_inputs(self):
        return {self.groups[i]['leg'] + '_tarsus': float(.2 * np.tanh(self.activation[i] / 10))
            for i in self.calibration['ltm']}

    def decode(self, rates, *, motor_cut=False, dt=.005):
        rates = np.asarray(rates)
        if (rates.shape != (391,) or not np.isfinite(rates).all() or np.any(rates < 0)
                or type(motor_cut) is not bool or not math.isfinite(dt) or not 0 < dt <= .05):
            raise ValueError('Finite nonnegative source motor rates and timestep required')
        pooled = self.calibration['pool'] @ rates
        if motor_cut:
            self.activation.fill(0)
            target = self.body.neutral.copy()
        else:
            self.activation += -math.expm1(-dt / .01) * (pooled - self.activation)
            target = self.calibration['rest'] + .02 * self.calibration['moment'] @ self.activation
        diagnostic = {}
        for group, rate in zip(self.groups, pooled):
            diagnostic.setdefault(group['leg'], {})[group['muscle']] = float(rate)
        self.last_motor = [{'leg': leg, 'muscle_rate_units': values}
            for leg, values in diagnostic.items()]
        return target, np.zeros(6, dtype=bool)

    def snapshot(self):
        return {'schema': self.schema, 'hash': self.identity,
            'calibration': copy.deepcopy(self.calibration), 'groups': copy.deepcopy(self.groups),
            'original': self.original.snapshot(), 'activation': self.activation.copy(),
            'last_motor': copy.deepcopy(self.last_motor)}

    def restore(self, saved):
        if saved.get('schema') != self.schema or saved.get('hash') != self.identity:
            raise ValueError('Anatomical recruitment identity changed')
        exact(self.calibration, saved['calibration'], 'Fixed recruitment anatomy')
        exact(self.groups, saved['groups'], 'Fixed recruitment groups')
        active = np.asarray(saved['activation'])
        if active.shape != (96,) or not np.isfinite(active).all() or np.any((active < 0) | (active > 200)):
            raise ValueError('Finite bounded muscle activation state required')
        self.original.restore(saved['original'])
        self.activation[:] = active
        self.last_motor = copy.deepcopy(saved['last_motor'])


class RecruitmentTrial(EfficacyTrial):
    schema = 'flylab.m3-rhythmic-recruitment-walking.v1'

    def __init__(self, base, calibration, groups):
        self.__dict__.update(base.__dict__)
        self.adapter = AnatomicalRecruitmentAdapter(self.adapter, calibration, groups)
        self.tendon_inputs.update(self.adapter.tendon_inputs)

    @classmethod
    def from_checkpoint(cls, graph, saved):
        if saved.get('schema') != cls.schema:
            raise ValueError('Explicit recruitment checkpoint required')
        source = copy.deepcopy(saved)
        source['schema'] = EfficacyTrial.schema
        source['adapter'] = saved['adapter']['original']
        base = EfficacyTrial.from_checkpoint(graph, source)
        trial = cls(base, saved['adapter']['calibration'], saved['adapter']['groups'])
        try:
            trial.adapter.restore(saved['adapter'])
            trial.tendon_inputs.update(trial.adapter.tendon_inputs)
            exact(saved, trial.snapshot(), 'Whole recruitment checkpoint')
        except Exception:
            trial.close()
            raise
        return trial
