"""Backend-independent necessary invariants of a reachable neural state."""
import numpy as np


def validate_neural_ranges(tick, parameters, *, v, h, queue, rate,
                           spike_count, refractory_until):
    if np.any(rate < 0) or np.any(spike_count < 0):
        raise ValueError('Negative neural rates or spike counters')
    if np.any(rate > 1/parameters.dt + 1) or np.any(spike_count > tick):
        raise ValueError('Neural rate or spike count out of range')
    if np.any(refractory_until < 0) or np.any(refractory_until > tick + round(parameters.refractory_s/parameters.dt)):
        raise ValueError('Refractory timer outside reachable range')
    for name, array, limit in (('voltage', v, 1e6), ('drive', h, 1e7), ('queue', queue, 1e7)):
        if np.any(np.abs(array) > limit):
            raise ValueError('Neural checkpoint '+name+' out of range')
