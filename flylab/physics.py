"""Independent, explicit physics selection; CUDA neural selection is unchanged."""
from dataclasses import asdict, dataclass
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import sys


@dataclass(frozen=True)
class PhysicsProfile:
    backend: str = 'cpu'
    noslip_iterations: int | None = None
    multiccd: bool | None = None
    max_contacts: int = 1024
    max_constraints: int = 4096
    cuda_graph: bool = True
    control_backend: str = 'cpu'

    def __post_init__(self):
        if self.backend not in ('cpu', 'warp'):
            raise ValueError('Physics backend must be cpu or warp')
        if self.control_backend not in ('cpu','cuda') or (self.control_backend=='cuda' and self.backend!='warp'):
            raise ValueError('CUDA reflex control requires Warp physics')
        if self.noslip_iterations is None:
            object.__setattr__(self, 'noslip_iterations', 0 if self.backend == 'warp' else 5)
        if self.multiccd is None:
            object.__setattr__(self, 'multiccd', self.backend == 'cpu')
        if type(self.multiccd) is not bool:
            raise ValueError('multiccd must be boolean')
        if self.backend == 'warp' and self.multiccd:
            raise ValueError('This arena has contact margins incompatible with Warp MULTICCD')
        if type(self.noslip_iterations) is not int or not 0 <= self.noslip_iterations <= 100:
            raise ValueError('noslip_iterations must be an integer in 0..100')
        if self.backend == 'warp' and self.noslip_iterations != 0:
            raise ValueError('Warp does not support noslip; explicitly use the noslip=0 physics profile')
        for name in ('max_contacts', 'max_constraints'):
            value = getattr(self, name)
            if type(value) is not int or not 64 <= value <= 65536:
                raise ValueError(name + ' must be an integer in 64..65536')
        if type(self.cuda_graph) is not bool:
            raise ValueError('cuda_graph must be boolean')

    @property
    def hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def profile_values(value=None):
    if isinstance(value, PhysicsProfile):
        return value
    if value is not None and not isinstance(value, dict):
        raise ValueError('Physics profile must be an object')
    return PhysicsProfile(**(value or {}))


@dataclass(frozen=True)
class PhysicsBodyFactory:
    profile: PhysicsProfile

    def __call__(self, seed, world, config, **kwargs):
        from .body import FlyGymBody
        cls = FlyGymBody
        if self.profile.backend == 'warp':
            from .body_warp import WarpBody
            cls = WarpBody
            if self.profile.control_backend=='cuda':
                from .body_warp_resident import ResidentWarpBody
                cls=ResidentWarpBody
        return cls(seed, world, config, physics_profile=self.profile, **kwargs)


def body_factory(value=None):
    from .body import FlyGymBody
    profile = profile_values(value)
    return FlyGymBody if profile == PhysicsProfile() else PhysicsBodyFactory(profile)


def physics_availability():
    result = {'cpu': {'available': True}, 'warp': {'available': False, 'validated': False}}
    try:
        packages = {name: version(name) for name in ('warp-lang', 'mujoco-warp')}
        if packages != {'warp-lang': '1.14.0', 'mujoco-warp': '3.9.0'}:
            result['warp'].update(packages=packages, reason='Install the pinned requirements-warp.txt versions')
            return result
        # Warp initializes lazily and prints its device banner. Keep doctor
        # stdout machine-readable while retaining diagnostics on stderr.
        with redirect_stdout(sys.stderr):
            import warp as wp
            devices = [str(d) for d in wp.get_devices() if d.is_cuda]
        result['warp'].update(available=bool(devices), devices=devices, packages=packages)
    except (PackageNotFoundError, ImportError, RuntimeError) as exc:
        result['warp']['reason'] = str(exc)
    return result
