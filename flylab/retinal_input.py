"""Clocked FlyGym Retina input with replayable temporal state.

Population features are engineering proxies. A per-ommatidium binding must
carry its own anatomical evidence; no neuronal ID is inferred from pixel order.
"""
import copy
import math
import numpy as np


class RetinalInput:
    def __init__(self, config):
        if not isinstance(config,dict) or set(config)-{'sample_hz','tau_s','dark_threshold'}:
            raise ValueError('Invalid retinal sampling configuration')
        self.config=dict(sample_hz=100,tau_s=.02,dark_threshold=.25)
        self.config.update(config)
        if self.config['sample_hz'] not in (100,200):raise ValueError('Retina supports 100 or 200 Hz')
        for key,lo,hi in [('tau_s',0.,10.),('dark_threshold',0.,1.)]:
            value=self.config[key]
            if type(value) not in (float,int) or not np.isfinite(value) or not lo<=value<=hi:
                raise ValueError('Invalid retinal '+key)
        self.period_ticks=round(10000/self.config['sample_hz'])
        self.next_tick=0;self.sample_tick=None;self.previous=None;self.filtered=None
        self.features={};self.samples=0

    def observe(self, body):
        now=round(body.physics_time()*10000)
        if abs(now/10000-body.physics_time())>1e-8:raise ValueError('Retinal clock is off the physics grid')
        if self.sample_tick is not None and now<self.sample_tick:raise ValueError('Retinal clock moved backwards')
        if now>=self.next_tick:
            # A caller may not silently skip scheduled samples.
            if now!=self.next_tick:raise ValueError('Retinal sampling deadline missed')
            observe=getattr(body,'retinal_observation',body.compound_eye_observation)
            observation=observe()
            if abs(observation['time_s']-now/10000)>1e-8:raise ValueError('Stale retinal frame')
            raw=np.asarray(observation['ommatidia'],dtype=np.float32)
            if raw.shape!=(2,721,2) or not np.isfinite(raw).all() or np.any(raw<0) or np.any(raw>1):
                raise ValueError('Invalid official Retina readout')
            # FlyGym readouts use normalized RGB intensity, with one active color
            # class per ommatidium. Summing the classes retains its intensity.
            luminance=raw.sum(axis=2)
            if np.any(luminance>1.00001):raise ValueError('Retina channel intensity exceeds 1')
            dt=1/self.config['sample_hz']
            first=self.previous is None
            motion=np.zeros(2) if first else np.mean(np.abs(luminance-self.previous),axis=1)/dt
            coverage=np.mean(luminance<self.config['dark_threshold'],axis=1)
            before=coverage if first else np.mean(self.previous<self.config['dark_threshold'],axis=1)
            looming=np.maximum(0.,(coverage-before)/dt)
            alpha=1. if not self.config['tau_s'] else -math.expm1(-dt/self.config['tau_s'])
            self.filtered=luminance.copy() if first else self.filtered+alpha*(luminance-self.filtered)
            for i,side in enumerate(('left','right')):
                self.features['retina_luminance_'+side]=float(self.filtered[i].mean())
                self.features['retina_motion_'+side]=float(motion[i])
                self.features['retina_loom_'+side]=float(looming[i])
            self.previous=luminance.copy();self.sample_tick=now;self.next_tick=now+self.period_ticks;self.samples+=1
        return dict(features=dict(self.features,retina_ommatidia=self.filtered.tolist()),
            sample_time_s=self.sample_tick/10000,next_time_s=self.next_tick/10000,
            age_s=(now-self.sample_tick)/10000,samples=self.samples,config=copy.deepcopy(self.config))

    def snapshot(self):
        return dict(schema='flylab.retinal-input.v2',config=copy.deepcopy(self.config),
            optics_contract='official-fisheye-single-sample-v2',
            next_tick=self.next_tick,sample_tick=self.sample_tick,samples=self.samples,
            previous=None if self.previous is None else self.previous.tolist(),
            filtered=None if self.filtered is None else self.filtered.tolist(),features=copy.deepcopy(self.features))

    def restore(self,state):
        if state.get('schema')!='flylab.retinal-input.v2' or state.get('config')!=self.config or state.get('optics_contract')!='official-fisheye-single-sample-v2':
            raise ValueError('Retinal checkpoint profile mismatch')
        sample,next_tick,count=state.get('sample_tick'),state.get('next_tick'),state.get('samples')
        if type(next_tick) is not int or type(count) is not int or count<0 or next_tick<0:
            raise ValueError('Invalid retinal clock')
        candidate=RetinalInput(self.config)
        if count==0:
            if sample is not None or next_tick!=0 or state.get('previous') is not None or state.get('filtered') is not None or state.get('features')!={}:
                raise ValueError('Nonempty initial retinal state')
        else:
            if type(sample) is not int or sample!=(count-1)*self.period_ticks or next_tick!=sample+self.period_ticks:
                raise ValueError('Inconsistent retinal sampling clock')
            for key in ('previous','filtered'):
                value=np.asarray(state.get(key),dtype=np.float32)
                if value.shape!=(2,721) or not np.isfinite(value).all() or np.any(value<0) or np.any(value>1.00001):
                    raise ValueError('Invalid retinal history')
                setattr(candidate,key,value.copy())
            expected={f'retina_{kind}_{side}' for kind in ('luminance','motion','loom') for side in ('left','right')}
            values=state.get('features')
            if not isinstance(values,dict) or set(values)!=expected or any(type(v) not in (float,int) or not np.isfinite(v) or not 0<=v<=200 for v in values.values()):
                raise ValueError('Invalid retinal features')
            candidate.features=copy.deepcopy(values)
        candidate.next_tick=next_tick;candidate.sample_tick=sample;candidate.samples=count
        self.__dict__.update(candidate.__dict__)
