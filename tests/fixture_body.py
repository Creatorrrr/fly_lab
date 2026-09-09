"""EXPLICIT TEST DOUBLE. NOT PHYSICS. Never selectable from run.py.
This fixture checks packet/UI contracts only; it does not validate MuJoCo.
"""
import math
import numpy as np
from flylab.common import clone,to_ui
from flylab.body import LEGS,MotorAdapter
from flylab import PHYSICS_DT,CONTROL_DT

class FixtureBody:
    backend='TEST-DOUBLE-NOT-PHYSICS';test_double=True
    def __init__(self,seed,world,config):
        self.world_spec=clone(world);self.config=clone(config);self.p=np.array([0.,0.,.9]);self.yaw=0.;self.time=0.;self.speed=0.;self.omega=0.
        self.travel=0.;self.collisions=0;self.fault=None;self.descending=np.zeros(2);self.push=0.;self.closed=False
    def set_world(self,w): self.world_spec=clone(w)
    def set_config(self,c):self.config=clone(c)
    def pose(self):
        a=-self.yaw;R=np.array([[math.cos(a),-math.sin(a),0.],[math.sin(a),math.cos(a),0.],[0.,0.,1.]])
        return self.p.copy(),R,np.array([0,0,-self.omega,*(R[:,0]*self.speed)])
    def ray(self,origin,direction,limit=20.):
        if direction[2]<-.5:return min(limit,float(origin[2]))
        return float(limit)
    def visual_panorama(self,origin,R):return [.92*math.exp(8*(math.cos(k/64*math.tau-math.pi+self.yaw)-1)) if self.world_spec['cueOn'] else 0 for k in range(64)]
    def nonfoot_contact(self):return False
    def perturb(self,bw,duration):self.push=bw*duration
    def step(self,command,dt):
        self.omega=command['yawRate'];self.yaw+=self.omega*dt;self.speed=command['forwardSpeed'];self.time+=dt
        _,R,_=self.pose();self.p+=R[:,0]*self.speed*dt;self.p[1]-=self.push;self.push=0;self.travel+=abs(self.speed)*dt;self.descending=MotorAdapter().map(command)
    def frame(self):
        p,R,vel=self.pose();legs={}
        for i,leg in enumerate(LEGS):
            side=1 if leg[0]=='l' else -1;front=1-(i%3)
            locals=[np.array([front*.43,side*y,z]) for y,z in [(.23,-.07),(.60,-.20),(.9,-.65),(1.1,-.9),(1.15,-.9),(1.2,-.9),(1.25,-.9),(1.3,-.9)]]
            legs[leg]=[to_ui(p+R@v) for v in locals]
        from flylab.common import S
        contact=[.33 if math.sin(self.time*7+i*math.pi)>0 else 0. for i in range(6)]
        angles=[.3*math.sin(self.time*7+i) for i in range(42)]
        body=dict(position=to_ui(p),yaw=self.yaw,yawRate=self.omega,speed=self.speed,verticalSpeed=0.,roll=0.,pitch=0.,phase=self.time,contact=0,collisions=0,travel=self.travel,legs=legs,segments={},contactsBW=contact,basis=(S@R).tolist())
        phy=dict(backend=self.backend,testDouble=True,physicalDt=PHYSICS_DT,controlDt=CONTROL_DT,physicsTime=self.time,settlingTime=0.,substeps=50,jointNames=[f'FIXTURE_JOINT_{i:02}' for i in range(42)],jointAngles=angles,jointVelocities=[0.]*42,jointTargets=angles,actuatorForces=[0.]*42,contactsBW=contact,contactVectorsBW=[[0.,v,0.] for v in contact],adhesion=[int(v>0) for v in contact],cpgPhases=[self.time]*6,descending=self.descending.tolist(),fault=None,gravity=self.config['gravity'],friction=self.config['friction'],bodyModelHash='TEST_ONLY')
        return body,phy
    def snapshot(self):return dict(backend=self.backend,modelHash='TEST_ONLY',p=self.p.tolist(),yaw=self.yaw,time=self.time,speed=self.speed,omega=self.omega,travel=self.travel,collisions=self.collisions,descending=self.descending.tolist(),push=self.push)
    def restore(self,s):
        if s['backend']!=self.backend:raise ValueError('Wrong fixture backend')
        for k in ('yaw','time','speed','omega','travel','collisions','push'):setattr(self,k,s[k])
        self.p=np.array(s['p']);self.descending=np.array(s['descending'])
    def close(self):self.closed=True
    def preview(self):return np.zeros((240,400,3),dtype=np.uint8)
