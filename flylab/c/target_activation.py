"""Engineering dynamics for signed position targets, not muscle physiology."""

import math

import numpy as np

RELEASE_TAU_S = 0.03


def advance_target_activation(current, requested, dt, growth_tau_s, *, reversals_only=False):
    """Integrate a held request with fast release and slower magnitude growth.

    On a reversal, integrate to the exact zero crossing using the release
    time constant, then use the growth time constant for the remaining time.
    The state is the existing signed offset; no gait phase or target cue enters.
    Inputs are validated and bounded by RateBodyAdapter.
    """
    growth_alpha = 1 - math.exp(-dt / growth_tau_s)
    release_alpha = 1 - math.exp(-dt / RELEASE_TAU_S)
    result = current + growth_alpha * (requested - current)
    if not reversals_only:
        same_direction_release = (current * requested >= 0) & (np.abs(requested) < np.abs(current))
        result[same_direction_release] = current[same_direction_release] + release_alpha * (
            requested[same_direction_release] - current[same_direction_release]
        )
    reversing = current * requested < 0
    if np.any(reversing):
        old, target = current[reversing], requested[reversing]
        crossing_s = RELEASE_TAU_S * np.log1p(-old / target)
        released = old + release_alpha * (target - old)
        crossed = dt > crossing_s
        released[crossed] = -target[crossed] * np.expm1(
            -(dt - crossing_s[crossed]) / growth_tau_s
        )
        result[reversing] = released
    return result
