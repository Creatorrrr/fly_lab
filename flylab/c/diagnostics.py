"""Fault evidence is explicitly non-restorable and does not require a frame."""
from datetime import datetime, timezone
import math
import numpy as np


def diagnostic_value(value):
    if isinstance(value, np.ndarray):
        # Bounded diagnostics; a healthy checkpoint preserves full arrays.
        finite = np.isfinite(value) if value.dtype.kind in 'fiu' else None
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    nonfinite=int(np.count_nonzero(~finite)) if finite is not None else None,
                    sample=[diagnostic_value(v) for v in value.flat[:16]])
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return dict(nonfinite=repr(value))
    if isinstance(value, dict):
        return {str(k): diagnostic_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostic_value(v) for v in value[:20000]]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)[:2000]


def fault_report(engine):
    result = dict(schema='flylab.fault.v2', restorable=False,
                  created_at=datetime.now(timezone.utc).isoformat(), read_errors={})
    readers = dict(fault=lambda: engine.fault or engine.body.fault,
                   control_tick=lambda: engine.control_tick,
                   model_tick=lambda: engine.tick,
                   control_time_s=lambda: engine.control_tick*.005,
                   physics_time_s=lambda: engine.body.physics_time(),
                   neural_tick=lambda: engine.neural.tick if engine.neural else None,
                   neural_time_s=lambda: engine.neural.tick*engine.parameters.dt if engine.neural else None,
                   encoder_tick=lambda: engine.encoder.control_tick,
                   neural=lambda: engine.neural.snapshot() if engine.neural else None,
                   body=lambda: engine.body.snapshot(),
                   graph_hash=lambda: engine.graph.hash, binding_hash=lambda: engine.bindings.hash,
                   parameter_hash=lambda: engine.parameters.hash,
                   backend=lambda: engine.neural.backend if engine.neural else 'legacy_b_rate',
                   pending=lambda: engine.pending, active=lambda: engine.active,
                   last_command=lambda: engine.last_command, last_sensors=lambda: engine.last_sensors,
                   last_ports=lambda: engine.last_ports, events=lambda: engine.events)
    for key, read in readers.items():
        try:
            result[key] = diagnostic_value(read())
        except Exception as exc:
            result['read_errors'][key] = str(exc)
    return result
