"""B's numerical model behind a readout boundary; state format stays v0.2.1."""
from .brain import NeuralController
from .common import clone, wrap


class LegacyBRate(NeuralController):
    backend = 'legacy_b_rate'

    def readout(self, ids):
        c = self.circuit
        return dict(ids=list(ids), values=[float(self.a[c.index[n]]) for n in ids],
                    means={k: self.mean(k) for k in c.pop}, heading=self.heading,
                    goal=self.goal, error=self.error, confidence=self.confidence,
                    headingBins=list(self.headingBins),
                    lesions=[c.nodes[i]['id'] for i in sorted(self.lesions)],
                    stimulated=[c.nodes[i]['id'] for i in self.stim],
                    output=clone(self.output), avoidanceActive=bool(self.turnMemory),
                    recoveryActive=bool(self.turnMemory and self.recoveryTime),
                    behavior=self.odorMode, recNorm=self.recNorm,
                    unit='1', kind='model-activation')

    def record_values(self):
        return self.a.tolist()

    def group_ids(self, group):
        return [self.circuit.nodes[i]['id'] for i in self.circuit.pop.get(group, [])]

    def heading_error(self, yaw):
        return abs(wrap(self.goal - yaw))
