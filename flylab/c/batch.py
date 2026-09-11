"""Full-connectome/sensor/body CUDA campaigns, with independent world state."""
import copy
from pathlib import Path
import numpy as np
from .engine import CEngine
from .neural_batch import CudaLIFBatch
from .ports import MotorArbiter
from .storage import StateStore
from ..warp_batch import WarpWorldBatch,BatchViewFactory
from ..physics import PhysicsProfile
from .. import CONTROL_DT
from ..common import to_ui


class BatchSession:
    def __init__(self,graph,bindings,seeds,*,mode='C_STRICT',body_options=None,world=None):
        if not 1<=len(seeds)<=32:raise ValueError('Use 1..32 worlds')
        self.graph,self.bindings=graph,bindings
        from dataclasses import replace
        from ..body_options import BodyOptions
        body_options=replace(BodyOptions.parse(body_options),render_camera=True)
        if mode not in ('C_STRICT','C_SHADOW','C_ASSISTED'):raise ValueError('Full neural batch requires a C mode')
        self.factory=BatchViewFactory(PhysicsProfile(backend='warp',control_backend='cuda'))
        self.engines={};self.status={};self.active=[];self.physics=self.neural=None;self.epoch=0
        self.closed=False;self.regroup_origin=None
        try:
            for i,seed in enumerate(seeds):
                key=str(i)
                self.engines[key]=CEngine(graph,bindings,seed=seed,mode=mode,backend='exp_lif_cuda',
                    body_factory=self.factory,body_options=body_options,world=world)
                self.status[key]='RUNNING'
            self._group()
        except Exception:
            try:self.close()
            except Exception:pass  # Keep the constructor/allocation failure.
            raise

    def _group(self, *, increment_epoch=True):
        self._require_open()
        if self.physics is not None:self.physics.close()
        self.active=[key for key in self.engines if self.status[key]=='RUNNING']
        self.physics=self.neural=None
        if not self.active:return
        engines=[self.engines[key] for key in self.active]
        try:
            self.neural=CudaLIFBatch([e.neural for e in engines])
            self.physics=WarpWorldBatch([e.body for e in engines])
        except Exception as exc:
            self._fail(exc)
            raise
        if increment_epoch:self.epoch+=1

    def _require_open(self):
        if self.closed:raise RuntimeError('Closed batch cannot be used; restore a checkpoint into a new session')

    def _fail(self, error):
        """A partially submitted GPU period must never be retried."""
        keys=list(self.active)
        self.active=[]
        physics=self.physics;self.physics=self.neural=None
        for key in keys:
            e=self.engines[key];self.status[key]='FAILED'
            e.fault=e.fault or str(error)
            if e.recorder:
                recorder=e.recorder;e.recorder=None
                try:recorder.close('FAILED',str(error))
                except Exception:pass  # Preserve the originating failure.
        if physics is not None:
            try:physics.close()
            except Exception:pass

    @staticmethod
    def _preparation_state(engine):
        return (engine.sensors.snapshot(),engine.encoder.snapshot(),
                engine.neuromuscular.snapshot(copy_diagnostics=False) if engine.neuromuscular else None)

    @staticmethod
    def _restore_preparation(engine,state):
        engine.sensors.restore(state[0]);engine.encoder.restore(state[1])
        if engine.neuromuscular:engine.neuromuscular.restore(state[2])

    def pause(self,world):
        self._require_open()
        if world not in self.engines:raise ValueError('Unknown world')
        if self.status[world]!='RUNNING':raise ValueError('World is not running')
        self.status[world]='PAUSED';self._group()

    def resume(self,world):
        self._require_open()
        if self.status.get(world)!='PAUSED':raise ValueError('World is not paused')
        self.status[world]='RUNNING';self._group()

    def cancel(self,world):
        self._require_open()
        if world not in self.engines or self.status[world] not in ('RUNNING','PAUSED'):raise ValueError('World is not cancellable')
        self.status[world]='CANCELLED'
        e=self.engines[world]
        if e.recorder:e.recorder.close('CANCELLED');e.recorder=None
        self._group()

    def step(self,controls=1):
        self._require_open()
        if type(controls) is not int or not 0<=controls<=200000:raise ValueError('Invalid batch step count')
        if any(s=='FAILED' for s in self.status.values()):
            raise RuntimeError('Faulted batch cannot advance; close it and restore a valid checkpoint')
        fault=next((self.engines[key].fault or self.engines[key].body.fault for key in self.active
                    if self.engines[key].fault or self.engines[key].body.fault),None)
        if fault:
            self._fail(fault)
            raise RuntimeError('Faulted batch cannot advance: '+str(fault))
        for _ in range(controls):
            if not self.active:break
            engines=[self.engines[key] for key in self.active]
            staged=[];commands=[];targets=[];pads=[]
            try:
                # Prepare every world before committing interventions or launching
                # kernels. Failure in a later world restores earlier sensor clocks.
                snapshots=[];preparations=[]
                try:
                    for e in engines:
                        snapshots.append((e,self._preparation_state(e)))
                        capture=np.unique(np.concatenate([e.subscription,e.record_indices if e.recorder else np.empty(0,np.int32)]))
                        preparations.append((e._prepare_control(capture),capture))
                except Exception:
                    for e,saved in reversed(snapshots):self._restore_preparation(e,saved)
                    raise
                for e,(values,capture) in zip(engines,preparations):
                    start=e.tick
                    prepared,packet,drive,pulses,ports=values
                    e._commit_interventions(prepared);e.last_sensors=packet;e.sensor_tick=start;e.last_ports=ports
                    legacy=e.legacy.step(packet,CONTROL_DT,e.config) if e.mode=='C_SHADOW' else None
                    command,e.last_motor_rates=e.decoder.decode(e.neural)
                    assist,reason=e.supervisor.step(packet,CONTROL_DT) if e.mode=='C_ASSISTED' and prepared['assist_enabled'] and prepared['coupled'] else (None,None)
                    e.last_command=MotorArbiter.choose(command,assist,mode=e.mode,legacy=legacy,motor_coupled=prepared['coupled'],stopped=e.stopped,assist_reason=reason)
                    e.last_command.update(sensor_tick=start,neural_readout_tick=start,interval_start_tick=start,interval_end_tick=start+e.substeps)
                    if e.neuromuscular:
                        target,adhesion=e.neuromuscular.decode(e.neural,disconnected=not prepared['coupled'] or e.stopped)
                        targets.append(target);pads.append(adhesion)
                        e.last_command.update(motor_execution=e.bindings.motor_execution,high_level_command_applied=False,joint_targets_rad=target.tolist(),adhesion=adhesion.tolist())
                    commands.append(e.last_command['u_final']);staged.append((drive,pulses,capture))
                self.neural.begin([s[0] for s in staged],engines[0].substeps,[s[2] for s in staged],[s[1] for s in staged])
                try:self.physics.advance(commands,np.asarray(targets) if targets else None,np.asarray(pads) if pads else None)
                finally:self.neural.finish()
                self.physics.sync()
                for key,e in zip(self.active,engines):
                    start=e.tick;e.control_tick+=1;e.timing_controls+=1
                    e.signal_start_tick=start;e.selected_events=[event for event in e.neural.last_events if event['index'] in e.subscription]
                    if e.metabolism:
                        p,R,velocity=e.body.pose();e.metabolism.step(CONTROL_DT,to_ui(p+R@np.array([.65,0.,-.1])),float(velocity[3:]@R[:,0]),e.world)
                    if e.recorder:
                        spikes=[event for event in e.neural.last_events if event['index'] in e.record_indices]
                        if spikes:e.recorder.event(dict(kind='spikes',tick=e.tick,events=spikes))
                        if e.control_tick%2==0:
                            r=e.neural.readout(e.record_indices);e.recorder.signal(e.tick,r['voltage_mV'],r['rate_Hz'])
                        e.recorder.event(dict(kind='motor_command',tick=start,details=e.last_command))
                        if e.control_tick%20==0:
                            body,physics=e.body.frame();e.recorder.body(dict(tick=e.tick,simTime=e.control_tick*CONTROL_DT,body=body,physics=physics,sensors=e.last_sensors,sensor_tick=e.sensor_tick,command=e.last_command,motion_expected=e.motion_expected))
                    e._clocks()
                    if e.body.fault:raise RuntimeError(e.body.fault)
            except Exception as exc:
                self._fail(exc)
                raise
        return self.summary()

    def summary(self):
        return dict(schema='flylab.full-cuda-batch.v1',graph_hash=self.graph.hash,neurons_per_world=self.graph.n,
            worlds={key:dict(status=self.status[key],tick=e.tick,model_seconds=e.control_tick*CONTROL_DT,fault=e.fault or e.body.fault,recording=e.recorder is not None) for key,e in self.engines.items()},
            active_worlds=list(self.active),epoch=self.epoch,closed=self.closed,regroup_origin=copy.deepcopy(self.regroup_origin),full_brain=self.graph.full_brain,
            shared_topology_bytes=self.neural.shared_topology_bytes if self.neural else 0,
            weights='world-owned',physics_backend='MuJoCo-Warp',neural_backend='vectorized ordered CUDA LIF',
            biological_validation=False,numerical_status='experimental; Warp replay/restore acceptance separate',
            regroup_semantics='pause/cancel freezes that world; resuming rebuilds a batch epoch from integration states')

    def checkpoint(self,path):
        self._require_open()
        if any(e.fault or e.body.fault for e in self.engines.values()):raise ValueError('Faulted batch is not a restorable checkpoint')
        return StateStore.save(path,dict(schema='flylab.full-batch-state.v1',graph_hash=self.graph.hash,binding_hash=self.bindings.hash,
            engines={key:e.checkpoint() for key,e in self.engines.items()},status=copy.deepcopy(self.status),
            active=list(self.active),epoch=self.epoch,regroup_origin=copy.deepcopy(self.regroup_origin),physics=self.physics.snapshot() if self.physics else None))

    @classmethod
    def restore(cls,graph,bindings,path,*,worlds=None):
        """Restore all worlds, or explicitly regroup selected checkpoint worlds.

        Regrouping preserves integration/neural/controller state and clocks, but
        rebuilds contacts and constraints in a new epoch, as pause/resume does.
        It does not claim bitwise continuation of the original Warp batch.
        """
        state=StateStore.load(path,max_files=4096)
        if state.get('schema')!='flylab.full-batch-state.v1' or state.get('graph_hash')!=graph.hash or state.get('binding_hash')!=bindings.hash:
            raise ValueError('Full batch checkpoint identity mismatch')
        keys=list(state.get('engines',{}))
        if not 1<=len(keys)<=32 or set(keys)!={str(i) for i in range(len(keys))}:
            raise ValueError('Invalid batch world inventory')
        keys=sorted(keys,key=int)
        status=state.get('status')
        if not isinstance(status,dict) or set(status)!=set(keys) or any(s not in ('RUNNING','PAUSED','CANCELLED') for s in status.values()):raise ValueError('Invalid batch statuses')
        if state.get('active')!=[key for key in keys if status[key]=='RUNNING']:raise ValueError('Active world order mismatch')
        selected=keys if worlds is None else worlds
        if not isinstance(selected,list) or not selected or any(key not in keys for key in selected) or len(selected)!=len(set(selected)):
            raise ValueError('Invalid selected checkpoint worlds')
        regroup=selected!=keys
        obj=cls.__new__(cls);obj.graph=graph;obj.bindings=bindings;obj.factory=BatchViewFactory(PhysicsProfile(backend='warp',control_backend='cuda'))
        if type(state.get('epoch')) is not int or state['epoch']<0:raise ValueError('Invalid batch epoch')
        obj.engines={};obj.status={str(i):status[key] for i,key in enumerate(selected)};obj.active=[];obj.physics=obj.neural=None;obj.epoch=state['epoch']+int(regroup)
        obj.closed=False;obj.regroup_origin=copy.deepcopy(state.get('regroup_origin'))
        if regroup:obj.regroup_origin=dict(checkpoint=Path(path).name,epoch=state['epoch'],worlds=list(selected),semantics='integration/controller states; contacts and constraints rebuilt')
        try:
            # JSON canonicalization sorts object keys lexically ("10" before
            # "2"). Device rows must retain the original numerical world order.
            for i,key in enumerate(selected):obj.engines[str(i)]=CEngine.from_checkpoint(graph,bindings,state['engines'][key],obj.factory)
            obj._group(increment_epoch=False)
            if obj.physics:
                if regroup:
                    rows=[state['active'].index(key) for key in selected if status[key]=='RUNNING']
                    obj.physics.restore_regrouped(state['physics'],rows)
                else:obj.physics.restore(state['physics'])
            return obj
        except Exception:
            try:obj.close()
            except Exception:pass  # Keep the restore/allocation failure.
            raise

    def close(self):
        if self.closed:return
        self.closed=True;self.active=[]
        physics=self.physics;self.physics=self.neural=None
        error=None
        for resource in ([physics] if physics is not None else [])+list(self.engines.values()):
            try:resource.close()
            except Exception as exc:
                if error is None:error=exc
        if error is not None:raise error
