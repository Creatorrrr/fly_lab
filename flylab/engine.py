"""Request-driven experiment engine, independent of HTTP, browser and rendering."""
from __future__ import annotations
from collections import deque
import math, io, csv, json, base64, hashlib
from pathlib import Path
import numpy as np
from . import __version__, CONTROL_DT, PHYSICS_DT, PROTOCOL
from .common import clone,number,integer,safe_tree,wrap
from .brain import Circuit
from .legacy import LegacyBRate
from .body import FlyGymBody,LEGS
from .sensors import SensorAdapter,default_world,validate_world

DATA=Path(__file__).resolve().parents[1]/'data'/'circuit.json'
def default_graph(): return json.loads(DATA.read_text())
def config_values(p=None):
    c=dict(mode='walk',task='explore',goalAngle=1.,altitude=.8,sensorNoise=.018,graphGain=1.,motorCoupled=True,friction=1.,gravity=1.)
    if p is not None:
        if not isinstance(p,dict) or set(p)-set(c): raise ValueError('Unknown configuration keys')
        c.update(p)
    if c['mode']!='walk': raise ValueError('B는 물리 보행 전용입니다. 날개 공기역학/비행은 구현하지 않았습니다.')
    if c['task'] not in ('explore','heading'): raise ValueError('Unknown task')
    if type(c['motorCoupled']) is not bool: raise ValueError('motorCoupled must be boolean')
    for k,lo,hi in [('goalAngle',-math.pi,math.pi),('altitude',.1,14),('sensorNoise',0,.3),('graphGain',0,2),('friction',.2,2),('gravity',.25,2)]: c[k]=number(c[k],k,lo,hi)
    return c

class Engine:
    def __init__(self,graph=None,seed=42,config=None,body_factory=FlyGymBody,world=None):
        self.seed=integer(seed,'seed');self.config=config_values(config);self.world=clone(validate_world(world or default_world()))
        self.circuit=Circuit(graph or default_graph());self.brain=LegacyBRate(self.circuit,self.seed);self.sensors=SensorAdapter(self.seed)
        self.body_factory=body_factory;self.body=body_factory(self.seed,self.world,self.config)
        self.tick=0;self.events=deque(maxlen=80);self.history=deque(maxlen=1201);self.log=[];self.log_truncated=False;self.object_serial=0
        self.subscription=[n['id'] for n in self.circuit.nodes];self.last_sensors=self.sensors.observe(self.body,self.world,CONTROL_DT,self.config)
        self.sensor_tick=0;self.initial=None
        self.event('B 물리 실험 시작' if not self.body.test_double else 'TEST DOUBLE · 실제 물리 아님')
        self.initial=self.checkpoint();self._record()
    def event(self,message): self.events.append(dict(time=self.tick*CONTROL_DT,message=message))
    def close(self): self.body.close()
    def step(self,n):
        for _ in range(n):
            if self.body.fault: break
            self.last_sensors=self.sensors.observe(self.body,self.world,CONTROL_DT,self.config);self.sensor_tick=self.tick
            command=self.brain.step(self.last_sensors,CONTROL_DT,self.config)
            self.body.step(command,CONTROL_DT);self.tick+=1
            if self.tick%20==0: self._record()
        return self.frame()
    def frame(self):
        body,physics=self.body.frame()
        if abs(physics['physicsTime']-self.tick*CONTROL_DT)>1e-5: raise RuntimeError('Physics/control clock mismatch')
        return dict(schema='flylab.frame.v2',seed=self.seed,tick=self.tick,simTime=self.tick*CONTROL_DT,
            body=body,physics=physics,config=clone(self.config),world=clone(self.world),sensors=clone(self.last_sensors),sensorTick=self.sensor_tick,
            neural=self.brain.readout(self.subscription),
            events=list(self.events),recording=dict(samples=len(self.history),seconds=120,replayTruncated=self.log_truncated))
    def _record(self):
        body,phy=self.body.frame();self.history.append(dict(t=self.tick*CONTROL_DT,body=body,physics=phy,values=self.brain.record_values(),command=clone(self.brain.output)))
    def subscribe(self,ids):
        if not isinstance(ids,list) or len(ids)>512 or len(set(ids))!=len(ids) or any(id not in self.circuit.index for id in ids): raise ValueError('0..512 unique known IDs required')
        self.subscription=list(ids)
    def _set_world(self,w):
        if w['obstacles']==self.world['obstacles']:
            self.body.set_world(w);self.world=w;return
        # Recompile static geometry to keep MuJoCo's broadphase consistent.
        # Transfer integration/controller state, not a kinematic pose correction.
        state=self.body.snapshot();new=self.body_factory(self.seed,w,self.config)
        try:new.restore(state)
        except Exception:new.close();raise
        old=self.body;self.body=new;self.world=w;old.close()
    def command(self,msg,record=True):
        if not isinstance(msg,dict): raise ValueError('command object required')
        typ=msg.get('type');p=msg.get('payload',{})
        if not isinstance(p,dict): raise ValueError('payload object required')
        safe_tree(p)
        if typ=='configure':
            new=config_values({**self.config,**p});self.body.set_config(new);self.config=new
        elif typ in ('cue','food'):
            if type(p.get('enabled')) is not bool: raise ValueError('enabled must be boolean')
            self.world['cueOn' if typ=='cue' else 'foodOn']=p['enabled'];self.body.set_world(self.world)
        elif typ=='intervene':
            ids=p.get('ids');kind=p.get('kind')
            if not isinstance(ids,list) or not 1<=len(ids)<=512 or any(id not in self.circuit.index for id in ids): raise ValueError('Known neural IDs required')
            if kind not in ('stimulate','suppress','release'): raise ValueError('Unknown intervention')
            amp=number(p.get('amplitude',.65),'amplitude',-1,1);dur=number(p.get('duration',2),'duration',.005,60)
            self.brain.intervene(ids,kind,amp,dur)
        elif typ=='releaseAll': self.brain.lesions.clear();self.brain.stim.clear()
        elif typ=='place':
            if p.get('kind') not in ('food','hazard','obstacle'): raise ValueError('Unknown object kind')
            pos=p.get('position');objid='user-'+str(self.object_serial+1);w=clone(self.world)
            if p['kind']=='obstacle':
                r=number(p.get('radius',.8),'radius',.2,3)
                # Only physically supported ground obstacles; user Y is not a levitation controller.
                if not isinstance(pos,list) or len(pos)!=3: raise ValueError('Position required')
                pos=[pos[0],r,pos[2]];w['obstacles'].append(dict(id=objid,p=pos,r=r))
                body,_=self.body.frame()
                if np.linalg.norm(np.array(pos)-body['position'])<r+3: raise ValueError('현재 몸에서 3 mm 이상 떨어진 곳에 배치하세요.')
            else: w['sources'].append(dict(id=objid,kind=p['kind'],p=pos,strength=1.2))
            validate_world(w);self._set_world(w);self.object_serial+=1
        elif typ=='clearAdded':
            w=clone(self.world)
            for k in ('obstacles','sources'): w[k]=[o for o in w[k] if not o['id'].startswith('user-')]
            self._set_world(w)
        elif typ=='push': self.body.perturb(number(p.get('bw',.5),'bw',-2,2),number(p.get('duration',.05),'duration',.005,.2))
        else: raise ValueError('Unknown command: '+str(typ))
        self.event(str(typ)+' '+json.dumps(p,ensure_ascii=False)[:140])
        if record:
            if len(self.log)<20000:self.log.append(dict(tick=self.tick,command=clone(msg)))
            else:self.log_truncated=True
        return self.frame()
    def checkpoint(self):
        return dict(schema='flylab.checkpoint.v2',appVersion=__version__,engine='hybrid-rate-B',
            controlDt=CONTROL_DT,physicsDt=PHYSICS_DT,graph=clone(self.circuit.data),seed=self.seed,config=clone(self.config),world=clone(self.world),
            tick=self.tick,sensorTick=self.sensor_tick,objectSerial=self.object_serial,body=self.body.snapshot(),brain=self.brain.snapshot(),sensors=self.sensors.snapshot(),
            lastSensors=clone(self.last_sensors),subscription=list(self.subscription))
    @classmethod
    def from_checkpoint(cls,s,body_factory=FlyGymBody):
        safe_tree(s)
        if s.get('schema')!='flylab.checkpoint.v2' or s.get('appVersion')!=__version__ or s.get('controlDt')!=CONTROL_DT or s.get('physicsDt')!=PHYSICS_DT: raise ValueError('현재 B 버전의 체크포인트만 복원합니다. 이전 버전의 상태는 직접 호환되지 않습니다.')
        tick=integer(s.get('tick'),'tick',0,100_000_000);st=integer(s.get('sensorTick'),'sensorTick',0,tick)
        # Construction/validation occurs in a replacement instance. The live engine is untouched on failure.
        e=cls(s['graph'],s['seed'],s['config'],body_factory,s['world'])
        try:
            e.body.restore(s['body']);e.brain.restore(s['brain']);e.sensors.restore(s['sensors']);e.tick=tick;e.sensor_tick=st
            e.last_sensors=clone(s['lastSensors']);validate_sensor_packet(e.last_sensors)
            e.object_serial=integer(s['objectSerial'],'objectSerial',0,1000000);e.subscribe(s['subscription'])
            if abs(e.brain.clock-tick*CONTROL_DT)>1e-5: raise ValueError('Neural clock mismatch')
            e.history.clear();e.log=[];e.initial=e.checkpoint();e.event('체크포인트 복원');e.frame();e._record()
            return e
        except Exception:
            e.close();raise
    def replay_file(self):
        if self.log_truncated: raise ValueError('명령 기록이 잘렸습니다. 체크포인트를 저장하세요.')
        return dict(schema='flylab.replay.v2',initial=self.initial,commands=clone(self.log),endTick=self.tick)
    def csv(self,physical=False):
        out=io.StringIO();w=csv.writer(out);out.write('\ufeff')
        if not physical:
            w.writerow(['time_s','x_mm','height_mm','z_mm','yaw_rad','speed_mm_s','yaw_command_rad_s']+[n['id'] for n in self.circuit.nodes])
            for h in self.history:w.writerow([h['t'],*h['body']['position'],h['body']['yaw'],h['body']['speed'],h['command']['yawRate']]+h['values'])
        else:
            joints=self.history[-1]['physics']['jointNames'] if self.history else []
            w.writerow(['time_s']+[f'angle_rad:{j}' for j in joints]+[f'target_rad:{j}' for j in joints]+[f'actuator_native:{j}' for j in joints]+[f'contact_BW:{leg}' for leg in LEGS]+['CPG_left','CPG_right'])
            for h in self.history:
                p=h['physics'];w.writerow([h['t']]+p['jointAngles']+p['jointTargets']+p['actuatorForces']+p['contactsBW']+p['descending'])
        return out.getvalue()


def validate_sensor_packet(p):
    expected={'schema','panorama','nearRanges','odor','odorChange','danger','angularVelocity','forwardSpeed','clearanceDown','clearanceUp','contact'}
    if not isinstance(p,dict) or set(p)!=expected: raise ValueError('Sensor packet contains unsupported fields (including possible truth leakage)')
    for k,n,lo,hi in [('panorama',64,0,1),('nearRanges',9,0,10),('odor',2,0,1)]:
        a=p[k]
        if not isinstance(a,list) or len(a)!=n:raise ValueError('Invalid sensor shape')
        for v in a:number(v,k,lo,hi)
    for k in expected-{'schema','panorama','nearRanges','odor'}:number(p[k],k,-1e8,1e8)

class Dispatcher:
    def __init__(self,body_factory=FlyGymBody): self.engine=None;self.factory=body_factory
    def need(self):
        if self.engine is None:raise ValueError('Initialize the physical engine first')
        return self.engine
    def ready(self):
        e=self.need()
        return dict(capabilities=dict(engine='hybrid-rate-B',body=e.body.backend,physics=not e.body.test_double,testDouble=e.body.test_double,fullBrain=False,flight=False,neuralDt=CONTROL_DT,physicalDt=PHYSICS_DT,maxAdvance=10,maxSubscription=512,checkpoint=True,replay=True,preview=True),catalog=e.circuit.catalog(),graph=e.circuit.data,manifest=e.circuit.data.get('manifest',{}),frame=e.frame())
    def replace(self,e):
        old=self.engine;self.engine=e
        if old:old.close()
        return self.ready()
    def handle(self,msg):
        if not isinstance(msg,dict) or msg.get('protocol')!=PROTOCOL:raise ValueError('flylab.protocol.v2 required')
        rid=integer(msg.get('requestId'),'requestId',1,2**53-1);op=msg.get('op');p=msg.get('payload',{})
        if not isinstance(p,dict): raise ValueError('payload object required')
        result=None
        if op=='init':result=self.replace(Engine(p.get('graph'),p.get('seed',42),p.get('config'),self.factory))
        elif op=='initExperiment':
            if p.get('schema')!='flylab.experiment.v2':raise ValueError('Experiment v2 required')
            result=self.replace(Engine(p.get('graph'),p.get('seed',42),p.get('config'),self.factory,p.get('world')))
        elif op=='restore':result=self.replace(Engine.from_checkpoint(p,self.factory))
        elif op=='advance':result=self.need().step(integer(p.get('steps',1),'steps',0,10))
        elif op=='frame':result=self.need().frame()
        elif op=='command':result=self.need().command(p)
        elif op=='subscribe':self.need().subscribe(p.get('ids'));result=self.need().frame()
        elif op=='catalog':
            start=integer(p.get('offset',0),'offset',0,1000000);limit=integer(p.get('limit',100),'limit',1,512);q=p.get('query','')
            if not isinstance(q,str) or len(q)>100:raise ValueError('Invalid catalog query')
            ns=[n for n in self.need().circuit.catalog() if q.lower() in (n['id']+' '+n['name']).lower()];result=dict(items=ns[start:start+limit],total=len(ns),offset=start)
        elif op=='checkpoint':result=self.need().checkpoint()
        elif op=='graph':result=self.need().circuit.data
        elif op=='csv':result=self.need().csv()
        elif op=='physicsCsv':result=self.need().csv(True)
        elif op=='exportReplay':result=self.need().replay_file()
        elif op=='history':result=list(self.need().history)
        elif op=='replay':result=self._replay(p)
        elif op=='paired':result=self._paired(p)
        elif op=='preview':
            from PIL import Image
            image=self.need().body.preview();buf=io.BytesIO();Image.fromarray(image).save(buf,format='PNG');result=dict(image='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),tick=self.need().tick,label='MuJoCo physical model camera (not sensory input)')
        else:raise ValueError('Unsupported operation: '+str(op))
        return dict(protocol=PROTOCOL,requestId=rid,ok=True,result=result)
    def _replay(self,p):
        safe_tree(p)
        if p.get('schema')!='flylab.replay.v2':raise ValueError('B replay required')
        start=p['initial']['tick'];end=integer(p.get('endTick'),'endTick',start,start+12000) # 60 model seconds per request
        commands=p.get('commands')
        if not isinstance(commands,list) or len(commands)>20000:raise ValueError('Replay command limit')
        last=start
        for item in commands:
            last=integer(item['tick'],'command tick',last,end)
        e=Engine.from_checkpoint(p['initial'],self.factory)
        try:
            for item in commands:
                e.step(item['tick']-e.tick)
                if e.tick!=item['tick']:raise RuntimeError('Replay stopped due to physics fault')
                e.command(item['command'])
            e.step(end-e.tick)
            if e.tick!=end:raise RuntimeError('Replay stopped due to physics fault')
            return self.replace(e)
        except Exception:e.close();raise
    def _paired(self,p):
        kind=p.get('kind');seconds=number(p.get('seconds',4),'seconds',2,4)
        if kind not in ('left-suppression','graph-off','dark','motor-off'):raise ValueError('Unknown paired intervention')
        cp=self.need().checkpoint();runs=[];steps=round(seconds/CONTROL_DT);onset=round(1/CONTROL_DT)
        for variant in ('control','intervention'):
            e=Engine.from_checkpoint(cp,self.factory)
            try:
                e.command(dict(type='configure',payload=dict(task='heading',goalAngle=1.7,sensorNoise=.06)),False);e.command(dict(type='food',payload=dict(enabled=False)),False)
                path=[];errs=[];base=e.body.travel
                for i in range(steps):
                    if i==onset and variant=='intervention':
                        if kind=='left-suppression':cmd=dict(type='intervene',payload=dict(ids=e.brain.group_ids('PFL3-L'),kind='suppress'))
                        elif kind=='dark':cmd=dict(type='cue',payload=dict(enabled=False))
                        else:cmd=dict(type='configure',payload={'graphGain':0} if kind=='graph-off' else {'motorCoupled':False})
                        e.command(cmd,False)
                    e.step(1)
                    if e.body.fault:raise RuntimeError('Paired experiment stopped: '+e.body.fault)
                    if i%10==0:
                        b,_=e.body.frame();path.append(dict(t=(i+1)*CONTROL_DT,p=b['position']));errs.append(e.brain.heading_error(b['yaw'])*180/math.pi)
                runs.append(dict(variant=variant,trace=path,metrics=dict(distance=e.body.travel-base,meanHeadingErrorDeg=float(np.mean(errs)),collisions=e.body.collisions-cp['body']['collisions'])))
            finally:e.close()
        sep=float(np.mean([np.linalg.norm(np.array(a['p'])-b['p']) for a,b in zip(runs[0]['trace'],runs[1]['trace'])]))
        return dict(schema='flylab.paired.v2',kind=kind,seconds=seconds,onset=1.,runs=runs,meanPathSeparation=sep,physical=not self.need().body.test_double)
    def close(self):
        if self.engine:self.engine.close();self.engine=None
