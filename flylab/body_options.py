"""Explicit comparison bodies and terrains; defaults preserve existing models."""
from dataclasses import dataclass,asdict
import math


@dataclass(frozen=True)
class BodyOptions:
    model: str = 'neuromechfly'
    terrain: str = 'flat'
    terrain_seed: int = 0
    slope_degrees: float = 10.
    render_camera: bool = False

    def __post_init__(self):
        if self.model not in ('neuromechfly','flybody'):raise ValueError('Unknown comparison body')
        if type(self.render_camera) is not bool:raise ValueError('render_camera must be boolean')
        if self.terrain not in ('flat','gaps','blocks','mixed','slope'):raise ValueError('Unknown terrain')
        if type(self.terrain_seed) is not int or not 0<=self.terrain_seed<2**32:raise ValueError('Invalid terrain seed')
        if type(self.slope_degrees) not in (int,float) or not math.isfinite(self.slope_degrees) or not -30<=self.slope_degrees<=30:
            raise ValueError('Invalid slope angle')

    @classmethod
    def parse(cls,value=None):return value if isinstance(value,cls) else cls(**(value or {}))


def make_world(options):
    from flygym.compose import FlatGroundWorld
    from flygym.compose.world.complex_terrain import GappedTerrainWorld,BlocksTerrainWorld,MixedTerrainWorld
    if options.terrain in ('flat','slope'):
        world=FlatGroundWorld(name='flylab_arena',half_size=100)
        if options.terrain=='slope':
            theta=math.radians(options.slope_degrees)/2
            world.ground_geom.quat=[math.cos(theta),0.,math.sin(theta),0.]
    elif options.terrain=='gaps':world=GappedTerrainWorld(name='flylab_arena',x_range=(-10,30),y_range=(-8,8))
    elif options.terrain=='blocks':world=BlocksTerrainWorld(name='flylab_arena',x_range=(-10,30),y_range=(-8,8),rand_seed=options.terrain_seed)
    else:world=MixedTerrainWorld(name='flylab_arena',y_range=(-8,8),rand_seed=options.terrain_seed)
    return world


def make_flybody(name):
    from flygym.compose import ActuatorType,KinematicPosePreset
    from flygym.compose.fly import FlyBody
    from flygym.flybody import FlyBodySkeleton,FlyBodyAxisOrder,FlyBodyJointPreset,FlyBodyActuatedDOFPreset
    fly=FlyBody(name=name)
    fly.add_joints(FlyBodySkeleton(axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH,
        joint_preset=FlyBodyJointPreset.LEGS_ONLY),KinematicPosePreset.FLYBODY_NEUTRAL)
    fly.add_actuators(fly.skeleton.get_actuated_dofs_from_preset(FlyBodyActuatedDOFPreset.LEGS_ACTIVE_ONLY),ActuatorType.POSITION,kp=100)
    fly.add_leg_adhesion()
    return fly


@dataclass(frozen=True)
class ComparisonBodyFactory:
    options: BodyOptions
    physics: object = None

    def __call__(self,seed,world,config,**kwargs):
        from .physics import body_factory
        return body_factory(self.physics)(seed,world,config,body_options=self.options,**kwargs)
