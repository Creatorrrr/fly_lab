"""Diagnostic-only coordinated core forcing through unforced BANC motor cells.

The time-indexed teaching table is intentionally incompatible with a final
autonomous controller. Its purpose is to test downstream sufficiency.
"""

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control
from tools.analyze_c_banc_biological_gait import analyze, load_model, sha256, write_json
from tools.analyze_c_banc_m3_motor_transfer import phases
from tools.banc_m3_core_adaptation import CORE_TYPES
from tools.collect_c_banc_m3_motor_teacher import MotorTeacher
from tools.probe_c_banc_m3_core_rhythm import CoreTrial, source_matrix


class CoreWaveTeacher(MotorTeacher):
    def __init__(self, network, ids, table):
        super().__init__(network, ids)
        self.wave = network.torch.as_tensor(table,dtype=network.rate.dtype,device=network.device)
        if self.wave.ndim != 2 or self.wave.shape[1] != len(ids):
            raise ValueError('Native half-step core teaching table required')
        self.cursor = network.torch.zeros((),dtype=network.torch.int64,device=network.device)
        self.zero_z = network.torch.zeros(len(ids),dtype=network.adaptation.dtype,device=network.device)

    def stage_states(self, rate, adaptation):
        torch = self.network.torch
        half = torch.index_select(self.wave,0,(self.cursor+1).reshape(1)).squeeze(0)
        end = torch.index_select(self.wave,0,(self.cursor+2).reshape(1)).squeeze(0)
        return (half,self.zero_z),(end,self.zero_z)

    def after_step(self):
        self.cursor.add_(2)

    def capture(self):
        super().capture()
        self.cursor.zero_()


def teaching_table(source, metadata, duration=2.2, dt=.00025):
    data = np.load(source/'core_zero.npz')
    rates,times = data['rate'],data['time_s']
    groups = np.asarray(metadata['core_groups'])
    # RF is identified by original direct leg-motor annotations, not by phase.
    group = np.flatnonzero(groups == 1)
    e1 = next(i for i in group if metadata['nodes'][i]['cell_type']=='IN17A001')
    active = np.flatnonzero((times>=.5)&(times<2.2))
    peaks,_ = find_peaks(rates[active,e1],prominence=20)
    peaks = active[peaks]
    columns = [next(i for i in group if metadata['nodes'][i]['cell_type']==kind) for kind in CORE_TYPES]
    phase = np.arange(128)/128
    cycles = [np.stack([np.interp(phase,np.linspace(0,1,b-a+1),rates[a:b+1,i])
        for i in columns],axis=1) for a,b in itertools.pairwise(peaks)]
    if len(cycles)<10:
        raise ValueError('At least10already exposed RF cycles required for fixed teaching waveform')
    cycle = np.mean(cycles,axis=0)
    native_t = np.arange(round((duration+.02)/(dt/2))+1)*(dt/2)
    table = np.empty((len(native_t),24))
    group_phase = np.array([.5,.5,0.,0.,0.,.5])
    for i,node in enumerate(metadata['nodes'][:24]):
        col = CORE_TYPES.index(node['cell_type'])
        x = (native_t/.15+group_phase[groups[i]])%1
        table[:,i] = np.interp(x,np.r_[phase,1.],np.r_[cycle[:,col],cycle[0,col]])
    table *= (1-np.exp(-native_t/.05))[:,None]
    if not np.isfinite(table).all() or np.any((table<0)|(table>200)):
        raise ValueError('Teaching rates must remain within original bounded range')
    return table.astype(np.float32), {'period_s':.15,'startup_envelope_tau_s':.05,
        'RF_source_cycles':len(cycles),'source_RF_period_s':((np.diff(peaks))*.005).tolist(),
        'mean_cycle':cycle.tolist(),'core_type_order':CORE_TYPES,
        'group_leg_by_direct_motor_projection':['rh','rf','lf','rm','lh','lm'],
        'group_training_phase_cycles':group_phase.tolist()}


def run(source, waveform, paths, graph_path, out):
    out.mkdir(parents=True,exist_ok=False)
    spec = {'experiment':'M3-E37-coordinated-core-teacher',
        'question':'Is a deliberately coordinated six-core signal sufficient to improve physical coordination through existing unforced motor pathways?',
        'scope':'Diagnostic only. Force24existing core rates at every nativeRKstage, with core adaptation0 already inE32. Original158682other cells evolve; all391motor units, actual body feedback, sourceW, tonic inputs, E31decoder and physics are unforced.',
        'teaching':'Use mean of all complete exposedRF E35core0cycles after.5s, matchfourcelltypes, stretch to.15s cycle, assign two anatomical tripod groups .5cycle apart. Ramp fromzero with50ms envelope. This arbitrary training coordination and externalclock are never autonomous evidence or biologicalparameters.',
        'decision':'If healthy2.2s completion and actualAP3LR R>=.5 anddistancefrom.5<=.15 with positiveforward hold, proceed to learning existing coreincoming efficacy. If not, reject this core-only forcing as a sufficient route and inspect downstream delivery before attempting to learn its rhythm. Failure is not proof that every possible corepattern is insufficient. No waveform grid.',
        'validity':'Sourcegraph/model/bodyandmotor mapping retained; sourceanatomical directMN projections establish6group leg identities; no motor/rootforcing. Native coreendpoints equal frozen table; tableclock matchesneural tick. Preserve imposed correction and actualcore/MN currents before summary.',
        'limits':'OneGPU,2.2modelsec,<=150wallsec; stopfault/5msnonfoot/upright<.9; F6notexposed/adoptionnotapplicable/Fallpending',
        'tool_sha256':sha256(__file__),'waveform_source_sha256':sha256(waveform/'core_zero.npz'),
        'source_trace_sha256':sha256(source/'trace.json'),'path_sha256':sha256(paths/'provenance.json')}
    write_json(out/'spec.json',spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    began=time.perf_counter()
    graph=GraphStore.load(graph_path)
    metadata=json.loads((paths/'provenance.json').read_text())
    ids=np.array(metadata['indices'][:24])
    table,wave_spec=teaching_table(waveform,metadata)
    np.savez_compressed(out/'teaching-table.npz',rate=table,ids=ids,half_dt_s=.000125)
    write_json(out/'teaching-waveform.json',wave_spec)
    session=CoreTrial.from_checkpoint(graph,StateStore.load(source/'start'))
    n=session.network
    if np.intersect1d(ids,session.adapter.motor_ids).size or n.core_gain!=0 or n.device!='cuda':
        raise ValueError('CUDA core0 and disjoint unforced motor population required')
    matrix=source_matrix(n)
    direct=matrix[session.adapter.motor_ids][:,ids].toarray().astype(float)
    projection=[]
    for group,leg in enumerate(wave_spec['group_leg_by_direct_motor_projection']):
        weights=direct[:,np.asarray(metadata['core_groups'])==group]
        motor_columns=np.flatnonzero(abs(weights).sum(axis=1)>0)
        expected_side='left' if leg.startswith('l') else 'right'
        expected_part={'f':'front_leg','m':'middle_leg','h':'hind_leg'}[leg[1]]
        nodes=[graph.nodes[session.adapter.motor_ids[i]] for i in motor_columns]
        if any(node['soma_side']!=expected_side or node['source_annotations']['body_part_effector']!=expected_part for node in nodes):
            raise ValueError('Source direct leg projection differs')
        projection.append({'group':group,'leg':leg,'motor_roots':[node['root_id'] for node in nodes]})
    write_json(out/'projection-provenance.json',projection)
    source_hash=n.identity
    n.identity=hashlib.sha256((source_hash+'diagnostic-core-wave-v1'+sha256(out/'teaching-table.npz')).encode()).hexdigest()
    teacher=CoreWaveTeacher(n,ids,table)
    teacher.capture()
    rows=[capture_control(session)]
    records=[]
    reason='duration'
    try:
        for tick in range(1,441):
            if time.perf_counter()-began>145:
                reason='wall_limit';break
            teacher.correction.zero_()
            session.advance(1)
            rows.append(capture_control(session))
            core=n.rate[teacher.ids].cpu().numpy().copy()
            np.testing.assert_array_equal(core,table[tick*40])
            if int(teacher.cursor)!=tick*40 or n.tick!=tick*20:
                raise ValueError('Teaching/native clocks differ')
            currents=n._current(n.rate)[session.adapter.motor_ids].cpu().numpy().copy()
            records.append({'core_rate':core,'motor_total_current':currents,
                'motor_core_current':direct@core,
                'core_correction_abs_sum':teacher.correction.cpu().numpy().copy()})
            if session.fault or session.body.fault or session.body.nonfoot_contact() or rows[-1]['upright']<.9:
                reason='health_guard';break
        frame=session.frame()
        StateStore.save(out/'teaching-state',{'schema':'flylab.training.core-wave.v1',
            'autonomous_controller':False,'source_model_hash':source_hash,
            'forced_core_ids':ids,'teacher_half_step':int(teacher.cursor),
            'whole_neural':n.snapshot(),'actual_body':session.body.snapshot(),
            'adapter':session.adapter.snapshot()})
    finally:
        write_json(out/'trace.json',{'schema':'flylab.training.core-wave-trace.v1',
            'controller':'External24coreteaching; never autonomousBANC',
            'graph_hash':graph.hash,'model_hash':n.identity,'body_hash':session.body.model_hash,
            'parameters':session.snapshot()['parameters'],'samples':rows})
        if records:
            np.savez_compressed(out/'observations.npz',
                **{key:np.asarray([r[key] for r in records]) for key in records[0]})
        session.close()
    analysis=phase=None
    if len(rows)>102:
        model=load_model(out/'trace.json')
        analysis,_,_=analyze(model)
        t=np.array([r['time_s'] for r in rows]);phase=phases(model['ap'][(t>=.5)&(t<2.2)])
    result={'experiment':spec['experiment'],'validity':'valid_externally_forced_core_diagnostic',
        'adoption':'not_applicable','milestone_outcome':'pending','reason':reason,'controls':len(rows)-1,
        'frame':frame,'analysis':analysis,'actual_AP_unrestricted_phase':phase,
        'minimum_upright_entire':min(r['upright'] for r in rows),'model_s':(len(rows)-1)*.005,
        'wall_s':time.perf_counter()-began,'forced_motor_neurons':0}
    write_json(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('frame','analysis')}),flush=True)
    print(json.dumps({'signed_forward_mm':frame['signed_forward_mm']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','waveform','paths','graph','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Output must be new')
    try:
        run(args.source,args.waveform,args.paths,args.graph,args.out)
    except Exception as error:
        if args.out.exists():write_json(args.out/'execution-error.json',{'error':type(error).__name__,
            'message':str(error),'validity':'incomplete','hypothesis_verdict':'not_tested'})
        raise
