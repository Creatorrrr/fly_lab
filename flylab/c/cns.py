"""Selected BANC leg motor outputs to physical joint targets; CPG is bypassed.

Anatomical muscle identity is source-backed. The activation-to-position map is
an explicitly separate engineering adapter, not a reconstructed muscle model.
"""
import math
import numpy as np
from .integrity import digest, finite

SOURCE='https://www.nature.com/articles/s41586-026-10735-w'

def tibia_bindings(graph):
    if graph.manifest['dataset_id']!='flywire_banc' or graph.manifest['snapshot_id']!='888':raise ValueError('BANC v888 required; no cross-specimen ID merge')
    rows=[]
    for side,letter in (('left','l'),('right','r')):
        for part,leg in (('front_leg','f'),('middle_leg','m'),('hind_leg','h')):
            groups={}
            for action in ('flexor','extensor'):
                ids=[n['id'] for n in graph.nodes if n['super_class']=='motor' and n['soma_side']==side
                     and n['source_annotations'].get('body_part_effector')==part
                     and n['source_annotations'].get('peripheral_target_type')=='tibia_'+action+'_muscle']
                if not ids:raise ValueError('Incomplete muscle annotation: '+side+' '+part+' '+action)
                groups[action]=ids
            rows.append(dict(leg=letter+leg,body_part=part,side=side,groups=groups,gain_rad_per_Hz=.005,
                             maximum_offset_rad=.5,activation_tau_s=.03,
                             uncertainty='Rate-to-joint-target proxy; no muscle force-length curve or recruitment calibration'))
    return dict(schema='flylab.cns-tibia-bindings.v1',graph_hash=graph.hash,rows=rows,evidence=[SOURCE],
                biological_validation=False,covered_dofs=6,total_leg_dofs=42,
                unbound='Other 36 DOFs held at neutral; no autonomous gait or sensory feedback inferred')

class TibiaDecoder:
    def __init__(self,graph,spec,body):
        if spec.get('schema')!='flylab.cns-tibia-bindings.v1' or spec.get('graph_hash')!=graph.hash:raise ValueError('CNS graph identity mismatch')
        self.graph=graph;self.spec=spec;self.hash=digest(spec);self.body=body
        self.offset=np.zeros(len(spec['rows']));self.tick=0;self.ports=[]
        if len(spec['rows'])!=6 or {r['leg'] for r in spec['rows']}!={'lf','lm','lh','rf','rm','rh'}:
            raise ValueError('Six unique leg ports required')
        for row in spec['rows']:
            finite(row['gain_rad_per_Hz'],'motor gain',0.,.1)
            finite(row['maximum_offset_rad'],'motor offset',.001,1.)
            finite(row['activation_tau_s'],'activation time constant',.001,1.)
            if set(row['groups'])!={'flexor','extensor'}:raise ValueError('Flexor/extensor motor pair required')
            suffix=f"/{row['leg']}_trochanterfemur-{row['leg']}_tibia-pitch"
            names=[i for i,name in enumerate(body.joint_names) if name.endswith(suffix)]
            if len(names)!=1:raise ValueError('Physical joint mapping not unique')
            indices={role:graph.resolve(ids) for role,ids in row['groups'].items()}
            for role,values in indices.items():
                for i in values:
                    n=graph.nodes[int(i)];a=n.get('source_annotations',{})
                    if n['super_class']!='motor' or n['soma_side']!=row['side'] or a.get('body_part_effector')!=row['body_part'] or a.get('peripheral_target_type')!='tibia_'+role+'_muscle':raise ValueError('Motor side/body-part/muscle mismatch')
            self.ports.append((row,names[0],indices))
        self.indices=np.asarray(sorted({int(i) for _,_,groups in self.ports for v in groups.values() for i in v}),np.int32)

    def decode(self,neural,dt=.005,disconnected=False):
        finite(dt,'actuator time interval',.0001,.05)
        r=neural.readout(self.indices)['rate_Hz'];rates={int(i):float(v) for i,v in zip(self.indices,r)}
        targets=self.body.neutral.copy();diagnostics=[]
        for k,(row,joint,groups) in enumerate(self.ports):
            values={name:float(np.mean([rates[int(i)] for i in ids])) for name,ids in groups.items()}
            # In this NeuroMechFly coordinate, positive pitch closes the tibia.
            requested=(values['flexor']-values['extensor'])*row['gain_rad_per_Hz']
            offset=float(np.clip(requested,-row['maximum_offset_rad'],row['maximum_offset_rad'])) if not disconnected else 0.
            self.offset[k]+=(1-math.exp(-dt/row['activation_tau_s']))*(offset-self.offset[k])
            targets[joint]+=self.offset[k]
            diagnostics.append(dict(leg=row['leg'],rates_Hz=values,offset_rad=float(self.offset[k]),target_rad=float(targets[joint])))
        self.tick+=1
        return targets,diagnostics

    def snapshot(self):return dict(hash=self.hash,tick=self.tick,offset=self.offset.copy())
    def restore(self,state):
        from .integrity import bounded_int
        if state.get('hash')!=self.hash:raise ValueError('CNS actuator identity mismatch')
        values=np.asarray(state['offset']);tick=bounded_int(state['tick'],'CNS control tick')
        if values.shape!=self.offset.shape or not np.isfinite(values).all() or any(abs(v)>r['maximum_offset_rad'] for v,(r,_,_) in zip(values,self.ports)):raise ValueError('CNS activation state out of range')
        self.offset[:]=values;self.tick=tick

def physical_joint_step(body,targets,dt=.005):
    """Actual actuator control and MuJoCo integration, with no root pose writes."""
    from .. import PHYSICS_DT
    targets=np.asarray(targets)
    if targets.shape!=body.neutral.shape or not np.isfinite(targets).all() or np.max(np.abs(targets))>10:raise ValueError('Invalid joint target array')
    steps=round(dt/PHYSICS_DT)
    if abs(steps*PHYSICS_DT-dt)>1e-9:raise ValueError('Integral physics dt required')
    for _ in range(steps):
        body.last_action=body.Action(joint_angles=targets.copy(),adhesion_onoff=np.ones(6,dtype=bool))
        body.apply(body.sim,body.fly.name,body.last_action);body.sim.step()
    body.mj.mj_forward(body.m,body.d)
    if not np.isfinite(body.d.qpos).all() or not np.isfinite(body.d.qvel).all():raise RuntimeError('Nonfinite physical state')
    p,R,_=body.pose();body.travel+=float(np.linalg.norm(p-body.last_position));body.last_position=p;body.walk_ticks+=1
    if R[2,2]<.15 or p[2]<0:body.fault='CNS circuit body fell; no automatic correction'
