"""Explicit comparison bodies and terrains; defaults preserve existing models."""
from dataclasses import dataclass,asdict
import math
import numpy as np


@dataclass(frozen=True)
class BodyOptions:
    model: str = 'neuromechfly'
    terrain: str = 'flat'
    terrain_seed: int = 0
    slope_degrees: float = 10.
    render_camera: bool = False
    actuation: str = 'legs'
    servo_profile: str = 'asset'

    def __post_init__(self):
        if self.model not in ('neuromechfly','flybody'):raise ValueError('Unknown comparison body')
        if self.actuation not in ('legs','whole_body'):raise ValueError('Unknown body actuation scope')
        if self.actuation=='whole_body' and self.model!='flybody':raise ValueError('Whole-body actuation requires FlyBody')
        if self.servo_profile not in ('asset','tracking','tracking_all'):raise ValueError('Unknown servo profile')
        if self.servo_profile!='asset' and self.actuation!='whole_body':raise ValueError('Tracking servos require whole-body actuation')
        if type(self.render_camera) is not bool:raise ValueError('render_camera must be boolean')
        if self.terrain not in ('flat','gaps','blocks','mixed','slope'):raise ValueError('Unknown terrain')
        if type(self.terrain_seed) is not int or not 0<=self.terrain_seed<2**32:raise ValueError('Invalid terrain seed')
        if type(self.slope_degrees) not in (int,float) or not math.isfinite(self.slope_degrees) or not -30<=self.slope_degrees<=30:
            raise ValueError('Invalid slope angle')

    @classmethod
    def parse(cls,value=None):return value if isinstance(value,cls) else cls(**(value or {}))

    def model_identity(self):
        values=asdict(self)
        # The omitted default preserves existing explicit FlyBody/terrain hashes.
        if self.actuation=='legs':values.pop('actuation')
        if self.servo_profile=='asset':values.pop('servo_profile')
        return values


def outside_physical_domain(position,rotation,options):
    """Test height and uprightness against the slope's actual plane normal."""
    if not np.isfinite(position).all() or not np.isfinite(rotation).all():return True
    if options.terrain=='slope':
        angle=math.radians(options.slope_degrees)
        normal=np.array([math.sin(angle),0.,math.cos(angle)])
        height=float(position@normal)
        upright=float(rotation[:,2]@normal)
    else:
        height=float(position[2]);upright=float(rotation[2,2])
    return upright<.15 or height<0 or np.linalg.norm(position[:2])>80


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


def make_flybody(name,*,whole_body=False):
    from flygym.compose import ActuatorType,KinematicPosePreset
    from flygym.compose.fly import FlyBody
    from flygym.flybody import FlyBodySkeleton,FlyBodyAxisOrder,FlyBodyJointPreset,FlyBodyActuatedDOFPreset
    fly=FlyBody(name=name)
    fly.add_joints(FlyBodySkeleton(axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH,
        joint_preset=FlyBodyJointPreset.ALL_BIOLOGICAL if whole_body else FlyBodyJointPreset.LEGS_ONLY),KinematicPosePreset.FLYBODY_NEUTRAL)
    fly.add_actuators(fly.skeleton.get_actuated_dofs_from_preset(FlyBodyActuatedDOFPreset.LEGS_ACTIVE_ONLY),ActuatorType.POSITION,kp=100)
    if whole_body:
        # Keep the 24 distal tarsal hinges passive. Head, antenna, mouth,
        # abdomen, wings and halteres receive independent position servos.
        for dof in fly.skeleton.iter_jointdofs():
            if dof.child.is_leg():continue
            params=next((v['general'] for v in fly.actuator_config.values() if dof.name in v['apply_to']),{})
            gain=params.get('gainprm','1')
            kp=float(gain.split()[0] if isinstance(gain,str) else gain[0])
            fly.add_actuators([dof],ActuatorType.POSITION,neutral_input=KinematicPosePreset.FLYBODY_NEUTRAL,kp=kp)
    fly.add_leg_adhesion()
    return fly


@dataclass(frozen=True)
class ComparisonBodyFactory:
    options: BodyOptions
    physics: object = None

    def __call__(self,seed,world,config,**kwargs):
        from .physics import body_factory
        return body_factory(self.physics)(seed,world,config,body_options=self.options,**kwargs)
