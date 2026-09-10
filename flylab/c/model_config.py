"""One parameter and execution contract for engines, verifiers and jobs."""
from dataclasses import asdict
from collections.abc import Mapping
from . import MODES
from .neural import LIFParameters, NEURAL_BACKENDS
from .integrity import finite


def resolve_parameters(bindings, override=None):
    """Defaults < binding < explicit override; a parameter object is complete."""
    if isinstance(override, LIFParameters):
        return override
    values = dict(bindings.spec.get('neural_parameters', {}))
    if override is not None:
        if not isinstance(override, Mapping):
            raise ValueError('Neural parameters must be a mapping or LIFParameters')
        values.update(override)
    return LIFParameters(**values)


def model_identity(parameters):
    events = parameters.integration == 'exact-exponential-voltage-events-v1'
    return dict(parameter_hash=parameters.hash, parameters=asdict(parameters),
                integration=parameters.integration, dt=parameters.dt,
                input_semantics=dict(constant_drive='held_mV',
                    pulse_target='voltage_mV' if events else 'synaptic_drive_mV',
                    pulse_targets_bypass_refractory=events))


def model_ticks(seconds, dt, maximum=10000):
    seconds = finite(seconds, 'model seconds', dt, 3600.)
    ticks = round(seconds/dt)
    if abs(ticks*dt-seconds) > 1e-10 or not 1 <= ticks <= maximum:
        raise ValueError('Model duration must be an integral, bounded number of neural ticks')
    return ticks


def execution_capabilities(bindings):
    parameters = resolve_parameters(bindings)
    modes = ['C_STRICT'] if bindings.spec.get('neuromuscular') else list(MODES)
    backends = [b for b in NEURAL_BACKENDS
                if b != 'exp_lif_mps' or parameters.dtype == 'float32']
    return dict(modes=modes, pilot_modes=[m for m in ('C_STRICT','C_SHADOW','C_ASSISTED') if m in modes],
                neural_backends=backends, model=model_identity(parameters),
                motor_execution=bindings.motor_execution,
                availability_scope='configuration support; runtime device checks remain required')


def validate_execution(bindings, mode, backend, parameters=None):
    from .backend_selection import resolve_backend
    backend = resolve_backend(backend)
    supported = execution_capabilities(bindings)
    if mode not in supported['modes']:
        reason = ('The BANC neuromuscular profile requires C_STRICT'
                  if bindings.spec.get('neuromuscular') else 'Unknown C control mode')
        raise ValueError(reason + ': ' + str(mode))
    if backend not in NEURAL_BACKENDS:
        raise ValueError('Unknown neural backend: ' + str(backend))
    resolved = resolve_parameters(bindings, parameters)
    if mode != 'B_COMPAT' and backend == 'exp_lif_mps' and resolved.dtype != 'float32':
        raise ValueError('MPS requires float32 neural parameters')
    return resolved
