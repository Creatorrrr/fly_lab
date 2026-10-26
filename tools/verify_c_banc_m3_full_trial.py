"""Bounded full-network/physical trials of frozen, unadopted M3 calibration.

Teacher reconstruction failures remain failures. This separate experiment asks
what the same frozen parameters actually do after all teaching is removed.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.adaptive_rate import AdaptiveRateNetwork
from flylab.c.banc_walking import BancWalkingSession
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control
from tools.analyze_c_banc_biological_gait import analyze, load_model, sha256, write_json
from tools.verify_c_banc_observation import exact


def calibrated_weights(graph, session, calibration, learned=True):
    weights,_=rate_weights(graph)
    weights=session.navigation.encoder.calibrate_weights(weights,session.adapter.motor_ids)
    if not np.array_equal(calibration['motor_ids'],session.adapter.motor_ids):
        raise ValueError('Calibrated motor roster differs')
    values=np.asarray(calibration['incoming_data'],np.float32)
    source=np.asarray(calibration['incoming_source_data'],np.float32)
    ratio=values/source
    if not np.isfinite(values).all() or np.any(source==0) or np.any((ratio<.1-1e-6)|(ratio>10+1e-6)):
        raise ValueError('Candidate violates positive efficacy bounds')
    changed=0
    for m,post in enumerate(session.adapter.motor_ids):
        original=np.arange(weights.indptr[post],weights.indptr[post+1])
        original=original[weights.data[original]!=0]
        sl=slice(calibration['incoming_indptr'][m],calibration['incoming_indptr'][m+1])
        if not np.array_equal(weights.indices[original],calibration['incoming_indices'][sl]) or not np.array_equal(weights.data[original],source[sl]):
            raise ValueError('Original source-edge weights or order changed')
        if learned:
            changed+=int(np.count_nonzero(weights.data[original]!=values[sl]))
            weights.data[original]=values[sl]
    return weights,changed


class TrialMotorAdapter:
    def __init__(self, original, network, calibration, calibration_hash, *, preserve_preload_on_cut=False):
        self.original=original
        self.rest=calibration['rest'].copy();self.moment=calibration['moment'].copy()
        self.group=calibration['motor_group'].copy()
        if self.moment.shape!=(42,96) or self.group.shape!=(391,) or not np.array_equal(self.rest,original.body.neutral):
            raise ValueError('Unexpected calibrated mechanical interface')
        self.pool=np.zeros((96,391))
        k=0
        for groups in original.muscles:
            for _name,ids,_moment in groups:
                columns=np.array([original.motor_columns[int(i)] for i in ids])
                if not np.all(self.group[columns]==k):raise ValueError('Anatomical muscle membership changed')
                self.pool[k,columns]=1/len(columns);k+=1
        self.activation=np.zeros(96);self.motor_cut=False
        self.preserve_preload_on_cut=bool(preserve_preload_on_cut)
        path=Path('verification/banc-natural-gait-m3-20260919/e05-afferent/candidate-source/flylab/c/afferent_rate.py')
        loader=importlib.util.spec_from_file_location('flylab.c._m3_archived_afferent',path)
        module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
        ids=np.unique(original._sensory_ids)
        self.encoding=module.AfferentRateEncoding(ids,
            cap=network.cap.cpu().numpy()[ids],slope=network.a.cpu().numpy()[ids],
            threshold=network.threshold.cpu().numpy()[ids],adaptation_gain=network.adaptation_gain,
            feature_gain=original.sensory_gain,target_rate=18)
        self.identity=hashlib.sha256((original.identity+self.encoding.identity+calibration_hash+'linear-anatomical-muscle-.02-tau.01-preload.2-v1').encode()).hexdigest()
        if self.preserve_preload_on_cut:
            self.identity=hashlib.sha256((self.identity+'preload-held-on-neural-cut-v2').encode()).hexdigest()

    def __getattr__(self,name):
        return getattr(self.original,name)

    def encode(self,**kwargs):
        return self.encoding(self.original.encode(**kwargs))

    def diagnostic(self,rates):
        pooled=self.pool@rates;diagnostic=[];k=0
        for leg,groups in zip(('lf','lm','lh','rf','rm','rh'),self.original.muscles):
            output={}
            for name,_ids,_moment in groups:
                output[name]=float(pooled[k]);k+=1
            diagnostic.append({'leg':leg,'muscle_rate_units':output})
        self.original.last_motor=diagnostic
        return pooled

    def decode(self,rates,*,motor_cut=False,dt=.005):
        rates=np.asarray(rates)
        if rates.shape!=(391,) or not np.isfinite(rates).all() or np.any(rates<0) or dt!=.005:
            raise ValueError('Invalid native motor output')
        pooled=self.diagnostic(rates)
        self.motor_cut=bool(motor_cut)
        if self.motor_cut:self.activation.fill(0)
        else:self.activation+=(1-math.exp(-dt/.01))*(pooled-self.activation)
        return self.rest+.02*self.moment@self.activation,np.zeros(6,bool)

    @property
    def tendon_inputs(self):
        value=0. if self.motor_cut and not self.preserve_preload_on_cut else .2
        return {leg+'_tarsus':value for leg in ('lf','lm','lh','rf','rm','rh')}

    def snapshot(self):
        return {'schema':'flylab.m3-trial-adapter.v1','hash':self.identity,
            'encoder':self.original.snapshot(),'activation':self.activation.copy(),'motor_cut':self.motor_cut}

    def restore(self,state):
        a=np.asarray(state['activation'])
        if state.get('schema')!='flylab.m3-trial-adapter.v1' or state.get('hash')!=self.identity or a.shape!=(96,) or not np.isfinite(a).all() or np.any((a<0)|(a>200.001)) or type(state.get('motor_cut')) is not bool:
            raise ValueError('Incompatible trial adapter state')
        self.original.restore(state['encoder']);self.activation[:]=a;self.motor_cut=state['motor_cut']


class CalibratedTrial(BancWalkingSession):
    schema='flylab.banc-calibrated-trial.v1'

    def __init__(self,graph,origin,calibration_path,*,learned=True,preserve_preload_on_cut=False):
        baseline=BancWalkingSession.from_checkpoint(graph,origin)
        self.__dict__.update(baseline.__dict__)
        self.calibration_hash=sha256(calibration_path)
        self.learned=learned
        calibration=dict(np.load(calibration_path))
        weights,self.changed_edges=calibrated_weights(graph,self,calibration,learned)
        n=self.network
        self.network=AdaptiveRateNetwork(weights,
            *[value.cpu().numpy().copy() for value in (n.tau,n.a,n.threshold,n.cap)],
            adaptation_gain=n.adaptation_gain,adaptation_tau_s=n.adaptation_tau_s,
            dt=n.dt,device='cuda',capture_steps=self.neural_steps_per_control,cuda_implementation='packed')
        if not learned and self.network.identity!=n.identity:raise ValueError('Theta1 neural baseline changed')
        start=copy.deepcopy(origin['network']);start['model_hash']=self.network.identity
        self.network.restore(start)
        self.adapter=TrialMotorAdapter(self.adapter,self.network,calibration,self.calibration_hash,
            preserve_preload_on_cut=preserve_preload_on_cut)
        self.tendon_inputs.update(self.adapter.tendon_inputs)
        self.set_navigation_cut(True)
        self.initialization='neutral'

    def frame(self):
        result=super().frame()
        result.update(calibration_hash=self.calibration_hash,learned_efficacy=self.learned,
            motor_model='Frozen anatomical linear muscle transfer, gain.02/tau10ms/preload.2; unadopted trial',
            initialization=self.initialization)
        return result

    def snapshot(self):
        result=super().snapshot()
        result.update(calibration_hash=self.calibration_hash,learned_efficacy=self.learned,
            initialization=self.initialization,tendon_inputs=self.tendon_inputs.copy())
        return result

    def restore_trial(self,saved):
        if saved.get('schema')!=self.schema or saved.get('calibration_hash')!=self.calibration_hash or saved.get('learned_efficacy')!=self.learned or saved.get('graph_hash')!=self.graph.hash:
            raise ValueError('Incompatible full trial checkpoint')
        self.body.restore(saved['body']);self.network.restore(saved['network']);self.adapter.restore(saved['adapter'])
        self.control_tick=saved['control_tick'];self.initial_position=saved['initial_position'].copy()
        self.signed_forward_mm=saved['signed_forward_mm'];self.wall_s=saved['wall_s']
        self.initialization=saved['initialization'];self.tendon_inputs=saved['tendon_inputs'].copy()
        self.navigation.restore(saved['navigation'],time_s=self.control_tick*.005)
        self.set_cuts(saved['cuts']);self.rate=self.network.readout(self.adapter.motor_ids)
        self._clocks()

    def release_teacher(self,teacher):
        if teacher.get('schema')!='flylab.m3-motor-teaching.v1' or teacher.get('control_tick')!=200:
            raise ValueError('Expected one-time1s teaching state')
        self.body.restore(teacher['body'])
        neural=copy.deepcopy(teacher['network']);neural['model_hash']=self.network.identity
        self.network.restore(neural)
        self.adapter.original.restore(teacher['adapter'])
        self.adapter.activation[:]=teacher['muscle_activation']
        self.control_tick=teacher['control_tick']
        self.rate=self.network.readout(self.adapter.motor_ids);self.adapter.diagnostic(self.rate)
        self.initial_position=self.body.pose()[0];self.signed_forward_mm=0.
        # Teaching did not record navigation history. Begin only the reporting
        # history at release; the cut encoder retains the same fixed tonic drive.
        self.navigation.set_target(self.navigation.sensor.target_xz,self.control_tick*.005,*self.body.pose()[:2])
        self.initialization='one-time teacher1s state; all teacher inputs removed'
        self._clocks()


def run(source,candidate,teacher_path,graph_path,out,fixed_preload_cut_only=False):
    out.mkdir(parents=True,exist_ok=False)
    spec={'experiment':'M3-E18-full-network-release',
        'hypothesis':'The frozen partially reconstructed candidate may produce useful full-network embodied motion, which need not exactly match the teacher trajectory',
        'prior_rejection':'E17b reconstruction screen/adoption reject is unchanged. This is a distinct bounded actual-gait diagnostic, not a relaxed reconstruction pass.',
        'intervention':'All158706BANC neurons and11584852source pair edges, native.25ms CUDA; only frozen eligible motor input efficacies changed. Continuous measured receptors through afferent-rate18 conversion, public fixed-forward DNg/20inhibitory inputs; anatomical linear muscle.02/tau10ms, tendon.2/no adhesion. NO teacher/clamp/phase/current replay/root correction during any rollout.',
        'arms':['learned-neutral2.2s','theta1-neutral2.2s','learned-taught-release2.2s','theta1-taught-release2.2s','learned-taught-motor-cut.3s'],
        'initialization':'Neutral restores original M2 full initial state; taught restores all neuralr/z, receptor filters, muscle activation, body integration and original1s clock ONCE. The cut navigation reporting history starts at release because teaching did not record it; held tonic inputs are unchanged. Release is not evidence of autonomous start. Cut arm distinguishes residual prepared motion.',
        'measurement':'Raw contiguous native-clock200Hz trace, model/adapter/graph hashes, output/cuts, posture/contact/fault. Descriptive gait window excludes first.5s AFTER each release; noF3ten-second orF6-F8 pass from this experiment. Same new interface at theta1 isolates learning from interface changes.',
        'stop':'Stop each arm on any body fault/nonfoot contact; preserve raw arrays before analysis. No candidate adoption in this diagnostic.',
        'limits':'One GPU job; each arm<=2.2modelsec, sum9.1modelsec plus small explicitly logged checkpoint checks; total<=300wallsec',
        'candidate_sha256':sha256(candidate/'candidate.npz'),'source_manifest_sha256':sha256(source/'start/manifest.json'),
        'teacher_manifest_sha256':sha256(teacher_path/'manifest.json'),'tool_sha256':sha256(__file__)}
    if fixed_preload_cut_only:
        spec.update(experiment='M3-E18b-preload-held-neural-cut',
            hypothesis='Separate the effect of removing decoded motor output from E18 original combined removal of motor output and tarsus preload',
            prior_rejection='All E18 noncut arms fell. Original cut changed six tarsus preloads .2 to0 and is retained as combined intervention, not pure neural-output cut.',
            arms=['learned-taught-motor-cut-fixed-preload.3s'],
            initialization='Same frozen learned candidate and one-time teacher1s state. Motor cut returns neutral joint targets and clears muscle activation, while all mechanical tendon preloads remain .2. Residual motion can include inertia, contact and neutral position-servo response; it is not evidence of passive inertia alone.',
            limits='One GPU job; .3modelsec/300wallsec; no repeated noncut trials or adoption')
    write_json(out/'spec.json',spec);(out/'runner.py').write_bytes(Path(__file__).read_bytes())
    original=StateStore.load(source/'start');teacher=StateStore.load(teacher_path);graph=GraphStore.load(graph_path)
    began=time.perf_counter();results={}
    arms=(
            ('learned-neutral',True,False,False,440),('theta1-neutral',False,False,False,440),
            ('learned-taught',True,True,False,440),('theta1-taught',False,True,False,440),
            ('learned-taught-cut',True,True,True,60))
    if fixed_preload_cut_only:arms=(('learned-taught-cut-fixed-preload',True,True,True,60),)
    for name,learned,taught,cut,controls in arms:
        arm=out/name;arm.mkdir();session=CalibratedTrial(graph,original,candidate/'candidate.npz',learned=learned,
            preserve_preload_on_cut=fixed_preload_cut_only)
        arm_began=time.perf_counter()
        try:
            if taught:session.release_teacher(teacher)
            else:exact(original['body'],session.body.snapshot(),'H18 neutral body')
            if cut:session.set_cuts({'motor':True})
            initial=session.snapshot();StateStore.save(arm/'start',initial)
            session.restore_trial(StateStore.load(arm/'start'))
            exact(initial,session.snapshot(),'H18 initial disk roundtrip')
            initial_tick=session.control_tick
            rows=[capture_control(session)]
            with (arm/'trace.jsonl').open('w') as stream:
                stream.write(json.dumps(rows[0])+'\n')
                for _ in range(controls):
                    if time.perf_counter()-began>285:raise TimeoutError('H18 complete-job wall limit')
                    session.advance(1,capture=True)
                    rows.append(capture_control(session));stream.write(json.dumps(rows[-1])+'\n')
                    if session.fault or session.body.nonfoot_contact():break
            physical={'model_s':(session.control_tick-initial_tick)*.005,'start_model_s':initial_tick*.005,
                'wall_s':time.perf_counter()-arm_began,'fault':session.fault,'nonfoot_contact':bool(session.body.nonfoot_contact()),
                'signed_forward_mm':session.signed_forward_mm,'minimum_upright':min(r['upright'] for r in rows),
                'changed_source_edges':session.changed_edges,'source_roster':graph.n,'source_edges':len(graph.indices),
                'device':session.network.device,'cuda_implementation':session.network.cuda_implementation,
                'cpg_unchanged':session._cpg_digest()==session.cpg_hash,'snapshot_disk_roundtrip_exact':True}
            write_json(arm/'physical-result.json',physical)
            StateStore.save(arm/'final-diagnostic',{'body':session.body.snapshot(),'network':session.network.snapshot(),
                'adapter':session.adapter.snapshot(),'frame':session.frame(),'schema':'flylab.m3-trial-final-diagnostic.v1'})
            write_json(arm/'trace.json',{'schema':'flylab.m3-full-trial-trace.v1','parameters':spec,
                'body_hash':session.body.model_hash,'model_hash':session.network.identity,
                'adapter_hash':session.adapter.identity,'graph_hash':graph.hash,'samples':rows,
                'analysis_time_origin_s':initial_tick*.005,'teacher_inputs_during_rollout':False})
            quality=None
            if physical['model_s']>.505:
                data=load_model(arm/'trace.json');lo=initial_tick*.005+.5;hi=float(data['t'][-1])
                window=(data['t']>=lo)&(data['t']<hi)
                data.update(window=window,moving=window&(data['speed']>.5),start_s=lo,end_s=hi)
                quality,_,_=analyze(data)
            result={'run_status':'completed','validity':'valid_full_trial','hypothesis_verdict':'descriptive_only',
                'adoption':'not_applicable','milestone_outcome':'pending','physical':physical,'quality':quality}
            write_json(arm/'result.json',result);results[name]=result
            print(json.dumps({'arm':name,'physical':physical,'quality_physical':quality['physical'] if quality else None}),flush=True)
        except Exception as exc:
            write_json(arm/'execution-error.json',{'type':type(exc).__name__,'message':str(exc),
                'wall_s':time.perf_counter()-arm_began,'adoption':'not_applicable','milestone_outcome':'pending'})
            raise
        finally:session.close()
    write_json(out/'result.json',{'experiment':spec['experiment'],'run_status':'completed','adoption':'not_applicable',
        'milestone_outcome':'pending','wall_s':time.perf_counter()-began,'arms':results})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','candidate','teacher','graph','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fixed-preload-cut-only',action='store_true')
    a=p.parse_args();run(a.source,a.candidate,a.teacher,a.graph,a.out,a.fixed_preload_cut_only)
