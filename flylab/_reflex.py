"""Batch the six pinned hybrid-controller reflexes without fast math.

CPG integration, contact projection and retraction-leg selection remain in
the upstream implementation. Arithmetic order within each joint is unchanged.
"""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def reflex_action(phases, magnitudes, spline, neutral, vectors, gains,
                  swing, swing_extension, retraction, stumbling, persistence,
                  stumbling_mask, selected, retraction_rates, stumbling_rates,
                  timestep, maximum, enable_adhesion):
    angles = np.empty((6, 7), np.float64)
    adhesion = np.empty(6, np.bool_)
    corrections = np.empty(6, np.float64)
    for i in range(6):
        if i == selected or persistence[i] > 0:
            retraction[i] += retraction_rates[0] * timestep
        else:
            retraction[i] = max(0., retraction[i] - retraction_rates[1] * timestep)
        if stumbling_mask[i]:
            stumbling[i] += stumbling_rates[0] * timestep
        else:
            stumbling[i] = max(0., stumbling[i] - stumbling_rates[1] * timestep)
        if retraction[i] > 0:
            correction = retraction[i]
            stumbling[i] = 0.
        else:
            correction = stumbling[i]
        correction = np.minimum(np.maximum(correction, 0.), maximum)
        phase = phases[i] % (2 * np.pi)
        scaled = correction * gains[i]
        for j in range(7):
            angle = neutral[i, j] + magnitudes[i] * (spline[i, j] - neutral[i, j])
            angles[i, j] = angle + scaled * vectors[i, j]
        corrections[i] = scaled
        adhesion[i] = enable_adhesion and not (swing[i, 0] < phase < swing[i, 1] + swing_extension)
    return angles, adhesion, corrections
