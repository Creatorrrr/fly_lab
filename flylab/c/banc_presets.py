"""Named target experiments; saved numerical parameters remain authoritative."""

import copy
import json
from functools import lru_cache
from pathlib import Path

from .banc_walking import BancWalkingParameters
from .target_sense import TargetSensor

_CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'banc-walking'
_NAMES = {
    'm1-target-v1': 'M1 기준',
    'm2-calibrated-v1': 'M2 보행 보정',
}
_FORWARD_DRIVE = 5000.0


@lru_cache(maxsize=2)
def _parameters(preset):
    if not isinstance(preset, str) or preset not in _NAMES:
        raise ValueError('Unknown BANC target preset')
    raw = json.loads((_CONFIG / (preset + '.json')).read_text(encoding='utf-8'))
    return BancWalkingParameters(**raw).state_dict()


def target_preset_catalog():
    return [dict(id=key, name=name, recommended=key == 'm2-calibrated-v1')
            for key, name in _NAMES.items()]


def resolve_target_preset(payload):
    """Resolve only new named requests, leaving legacy RPC values untouched."""
    if 'preset' not in payload:
        return payload
    if set(payload) - {'preset', 'seed', 'navigation'}:
        raise ValueError('A BANC target preset cannot be mixed with parameters')
    preset = payload['preset']
    if not isinstance(preset, str) or preset not in _NAMES:
        raise ValueError('Unknown BANC target preset')
    navigation = payload.get('navigation')
    if not isinstance(navigation, dict) or set(navigation) != {'target_xz_mm'}:
        raise ValueError('A BANC target preset requires only target coordinates')
    target = list(TargetSensor(navigation['target_xz_mm']).target_xz)
    return dict(seed=payload.get('seed', 42), parameters=copy.deepcopy(_parameters(preset)),
                navigation=dict(target_xz_mm=target, forward_drive=_FORWARD_DRIVE))


def observed_target_preset(observation):
    """Display the actual restored model, never the next UI selection."""
    custom = dict(id=None, name='직접 설정')
    if (not isinstance(observation.get('parameters'), dict)
            or not observation.get('navigation')
            or observation.get('descending_drive') != _FORWARD_DRIVE):
        return custom
    parameters = BancWalkingParameters(**observation['parameters']).state_dict()
    for preset, name in _NAMES.items():
        if parameters == _parameters(preset):
            return dict(id=preset, name=name)
    return custom
