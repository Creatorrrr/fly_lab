"""Explicit per-cell adaptation hypothesis for existing BANC core neurons.

The original sparse graph, transfer function, RK ordering and other cells are
unchanged. A zero gain is a physiological hypothesis, not a periodic reset.
"""

import hashlib

import numpy as np

from flylab.c.adaptive_fused import SOURCE
from flylab.c.adaptive_rate import AdaptiveRateNetwork

CORE_TYPES = ('IN17A001', 'INXXX466', 'IN16B036', 'IN19A007')


class CoreAdaptationNetwork(AdaptiveRateNetwork):
    def __init__(self, *args, core_ids, core_gain, **kwargs):
        core_ids = np.asarray(core_ids)
        n = args[0].shape[0]
        if (core_ids.ndim != 1 or core_ids.dtype.kind not in 'iu'
                or len(core_ids) != len(np.unique(core_ids))
                or np.any((core_ids < 0) | (core_ids >= n))
                or not np.isfinite(core_gain) or not 0 <= core_gain <= 100):
            raise ValueError('Unique existing core neuron IDs and bounded gain required')
        self.core_ids = core_ids.astype(np.int64, copy=True)
        self.core_gain = float(core_gain)
        if not kwargs.get('adaptation_gain', 0) > 0:
            raise ValueError('Positive unchanged non-core adaptation required')
        super().__init__(*args, **kwargs)
        self._init_gains()
        self.base_identity = self.identity
        self.identity = hashlib.sha256((self.base_identity + 'm3-core-adaptation-v1'
            + repr(self.core_gain)).encode() + self.core_ids.tobytes()).hexdigest()

    def _init_gains(self):
        if not hasattr(self, '_cell_gain'):
            gains = np.full(self.n, self.adaptation_gain, np.float32)
            gains[self.core_ids] = self.core_gain
            self._cell_gain = self.torch.as_tensor(gains, device=self.device)

    def _init_fusion(self):
        self._init_gains()
        source = SOURCE.replace('float gain, float inverse_adaptation_tau',
            'const float* cell_gain, float gain, float inverse_adaptation_tau')
        source = source.replace('mul_exact(gain, rate[i])', 'mul_exact(cell_gain[i], rate[i])')
        if self.fuse_tanh:
            source = source.replace('mul_exact(cap[i], activation[i])',
                'mul_exact(cap[i], matching_tanh(activation[i]))')
        self._fused_kernels = {name: self._cupy.RawKernel(source, name, options=('--fmad=false',))
            for name in ('csr_input', 'stage', 'finish')}
        for kernel in self._fused_kernels.values():
            kernel.compile()
        self._fused_buffers = [self.torch.empty_like(self.rate) for _ in range(11)]

    def _launch_fused(self, name, tensors, scalars=()):
        if name in ('stage', 'finish'):
            tensors = (*tensors, self._cell_gain)
        return super()._launch_fused(name, tensors, scalars)

    def _derivative_pair(self, rate, adaptation):
        self._init_gains()
        dr, _ = super()._derivative_pair(rate, adaptation)
        return dr, (self._cell_gain * rate - adaptation) / self.adaptation_tau_s

    def snapshot(self):
        return {**super().snapshot(), 'core_adaptation': {'schema': 'flylab.core-adaptation.v1',
            'ids': self.core_ids.copy(), 'gain': self.core_gain, 'base_model_hash': self.base_identity}}

    def restore(self, saved):
        core = saved.get('core_adaptation', {})
        if (core.get('schema') != 'flylab.core-adaptation.v1'
                or core.get('gain') != self.core_gain
                or core.get('base_model_hash') != self.base_identity
                or not np.array_equal(core.get('ids'), self.core_ids)):
            raise ValueError('Complete core adaptation identity required')
        super().restore(saved)
