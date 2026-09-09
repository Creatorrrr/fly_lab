"""Shared external-input contract, checked before either clock advances."""
import itertools
import math
import numpy as np
from .integrity import bounded_int

MAX_DRIVE_MV = 1000.


def validate_schedule(events, from_tick=0):
    """One sweep per target; also used when restoring or cancelling input."""
    changes = {}
    for event in events:
        if event['kind'] != 'stimulate' or event['expires_tick'] <= from_tick:
            continue
        for index in event['indices']:
            changes.setdefault(index, []).extend(((max(from_tick, event['at_tick']), event['amplitude_mV']),
                                                  (event['expires_tick'], -event['amplitude_mV'])))
    for index, rows in changes.items():
        total = 0.
        for tick, group in itertools.groupby(sorted(rows), key=lambda row: row[0]):
            total += math.fsum(delta for _, delta in group)
            if abs(total) > MAX_DRIVE_MV:
                raise InputRejected(f'Combined stimulus exceeds +/-1000 mV at neuron index {index}, tick {tick}')


class InputRejected(ValueError):
    """An invalid request, not a fault in an already advanced experiment."""


def validate_input(n, drive, steps, capture=(), pulses=None):
    bounded_int(steps, 'neural steps', 0, 10000)
    drive = np.asarray(drive)
    if drive.shape != (n,) or drive.dtype.kind not in 'fiu' or not np.isfinite(drive).all() or np.max(np.abs(drive)) > MAX_DRIVE_MV:
        raise InputRejected('Finite mV input per included neuron required; combined drive exceeds +/-1000 mV')
    capture = np.asarray(capture, dtype=np.int32)
    if capture.ndim != 1 or len(capture) > 1024 or (len(capture) and (capture.min() < 0 or capture.max() >= n)):
        raise InputRejected('Invalid bounded readout')
    if pulses is not None:
        if not isinstance(pulses, (list, tuple)) or len(pulses) != 2:
            raise InputRejected('Invalid external pulse matrix')
        pi, pv = map(np.asarray, pulses)
        if pi.ndim != 1 or pi.dtype.kind not in 'iu' or pv.shape != (steps, len(pi)) or pv.dtype.kind not in 'fiu' or not np.isfinite(pv).all():
            raise InputRejected('Invalid external pulse matrix')
        if len(pi) and (pi.min() < 0 or pi.max() >= n):
            raise InputRejected('Invalid external pulse targets')
        # Pulse currents and constant mV drive have different semantics. Check
        # representability, not the constant-drive limit, before float32 casts.
        if pv.size and np.max(np.abs(pv)) > np.finfo(np.float32).max / max(1, len(pi)):
            raise InputRejected('External pulse addition exceeds float32 range')
    return drive, capture


def validate_stimulus_schedule(candidate, events):
    """Reject overlapping excessive input; endpoints are half-open intervals."""
    if candidate['kind'] != 'stimulate':
        return
    start, end = candidate['at_tick'], candidate['expires_tick']
    changes = {i: [(start, candidate['amplitude_mV']), (end, -candidate['amplitude_mV'])]
               for i in candidate['indices']}
    selected = changes.keys()
    for event in events:
        if event['kind'] != 'stimulate':
            continue
        a, b = max(start, event['at_tick']), min(end, event['expires_tick'])
        if a >= b:
            continue
        for i in selected & set(event['indices']):
            changes[i].extend(((a, event['amplitude_mV']), (b, -event['amplitude_mV'])))
    for index, rows in changes.items():
        total = 0.
        for tick, group in itertools.groupby(sorted(rows), key=lambda row: row[0]):
            total += sum(delta for _, delta in group)
            if abs(total) > MAX_DRIVE_MV:
                raise InputRejected(f'Combined stimulus exceeds +/-1000 mV at neuron index {index}, tick {tick}')
