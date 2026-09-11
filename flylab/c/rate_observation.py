"""Read-only input decomposition for the explicit research rate equation.

These are model input quantities and instantaneous rate targets, not measured
membrane voltages, PSPs, spikes or forces. Preserve individual motor identities
when interpreting them; silence does not imply absence of synaptic input.
"""

import numpy as np
from scipy.sparse import csr_matrix


def _numeric(value, shape, name):
    data = np.asarray(value)
    if (
        data.shape != shape
        or data.dtype.kind not in "fiu"
        or not np.isfinite(data).all()
    ):
        raise ValueError("Finite aligned numeric " + name + " required")
    return data.astype(np.float64, copy=True)


class RateInputObserver:
    """Inspect selected post rows while retaining every presynaptic source.

    Weights are already scaled W[post, pre]. No normalization, integration or
    parameter fitting occurs here. Time rows must align rate[t] with the held
    drive used during [t, t+dt), including the first interval after an input cut.
    """

    def __init__(self, weights_post_pre, tau, a, threshold, cap, indices):
        weights = csr_matrix(weights_post_pre, copy=True)
        if weights.dtype.kind not in "fiu":
            raise ValueError("Real numeric post/pre weights required")
        weights = weights.astype(np.float64)
        weights.check_format(full_check=True)
        weights.sum_duplicates()
        weights.sort_indices()
        if (
            weights.shape[0] != weights.shape[1]
            or weights.shape[0] == 0
            or not np.isfinite(weights.data).all()
        ):
            raise ValueError("Finite square post/pre weights required")
        self.n = weights.shape[0]
        ids = np.asarray(indices)
        if (
            ids.ndim != 1
            or not ids.size
            or ids.dtype.kind not in "iu"
            or (ids < 0).any()
            or (ids >= self.n).any()
            or len(set(ids.tolist())) != len(ids)
        ):
            raise ValueError("Unique nonempty observer target indices required")
        self.indices = ids.astype(np.int64, copy=True)
        parameters = [
            _numeric(value, (self.n,), name)
            for value, name in zip(
                (tau, a, threshold, cap), ("tau", "a", "threshold", "cap")
            )
        ]
        if any((value <= 0).any() for value in parameters):
            raise ValueError("Positive research rate parameters required")
        self.tau, self.a, self.threshold, self.cap = [
            value[self.indices] for value in parameters
        ]
        selected = weights[self.indices]
        self.positive = selected.copy()
        self.negative = selected.copy()
        self.positive.data = np.maximum(self.positive.data, 0)
        self.negative.data = np.minimum(self.negative.data, 0)
        self.positive.eliminate_zeros()
        self.negative.eliminate_zeros()

    def observe(self, rates, drives, output_mask):
        shape = np.shape(rates)
        if len(shape) != 2 or shape[0] == 0 or shape[1] != self.n:
            raise ValueError("Nonempty time by complete-neuron rates required")
        values = _numeric(rates, shape, "rates")
        inputs = _numeric(drives, shape, "held drives")
        mask = _numeric(output_mask, (self.n,), "outgoing mask")
        if (values < 0).any() or not np.isin(mask, (0, 1)).all():
            raise ValueError("Nonnegative rates and binary outgoing mask required")
        emitted = values * mask
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            positive = (self.positive @ emitted.T).T
            negative = (self.negative @ emitted.T).T
            external = inputs[:, self.indices]
            margin = external + positive + negative - self.threshold
            target = np.maximum(self.cap * np.tanh(self.a / self.cap * margin), 0)
            rate = values[:, self.indices]
            derivative = (target - rate) / self.tau
        result = {
            "rate": rate,
            "excitatory_input": positive,
            "inhibitory_input": negative,
            "external_input": external,
            "threshold_margin": margin,
            "instantaneous_target_rate": target,
            "rate_derivative": derivative,
        }
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Non-finite raw rate observation")
        return result
