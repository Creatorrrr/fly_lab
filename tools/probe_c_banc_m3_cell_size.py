"""One full-feedback trial of the frozen core/relay cell-size hypothesis."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control
from tools.analyze_c_banc_biological_gait import analyze, load_model, sha256, write_json
from tools.analyze_c_banc_m3_motor_transfer import phases
from tools.banc_m3_cell_size import CellSizeTrial
from tools.verify_c_banc_observation import exact

PARAMETERS_SHA = '18179373c3ba004bfe877eb0ab181c315fbe2811e1a393b0c43a68deb3aeab59'
MORPHOLOGY_SHA = 'aa0b9bd6860c030b62c879c34bc7346a8745cb5f1b6f0309d16f1f3d68c8a104'


def run(source, paths, parameters, morphology, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': 'M3-E36-core-relay-size',
        'hypothesis': 'A cell-size dependent response gain/threshold may make existing core/relay dynamics more useful for actual inter-leg coordination than uniform physiology.',
        'comparison': 'E32core0 with unchanged8/.3 adaptation inall noncorecells. Only188sourcecore/relay a/threshold are changed to existing frozen morphology_area mean arrays. tau=.02/cap200, originalW/all158706cells, drives, actual senses, E31decoder/body are unchanged.',
        'provenance': 'Measured areas and previously validated empirical morphology predictions are distinguished. Neither predicted surface area nor conversion to electrical response is a measured physiological parameter. Past partial/morphology-area sensory-to-motor failures remain rejected.',
        'counterevidence': 'On already observedE31states gain normalization reduces saturation but does not remove most below-threshold samples and can reduce absolute pathway response. The actual nonlinear future could still be worse; no gain/threshold grid follows this single trial.',
        'validity': 'Frozenfile hashes and158706rootorder;188sourceIDs; tau/cap constant; onlyselecteda/threshold changed; identity patch first source control matchesE32; disk candidate checkpoint .05s complete future exact when1s reached. ActualCUDA full feedback, no source replay/clock/phase/root correction.',
        'decision': 'Report actual unrestricted AP3LR phase and compareE32core0 progress/upright. Only healthy completion, positiveprogress, all3R>=.5 andmean distancefrom.5<=.15 support a subsequent frozen10sFtrial. AllF1-F8remainpending; short signal screen never passesF4.',
        'limits': 'OneGPU,2.2candidate+.005identity+.05clone=2.255maximumwholemodelsec,<=150wallsec. Stopfault/5msnonfoot/upright<.9. NoF6exposure/sourceM2/liveUIchange.',
        'parameters_sha256': PARAMETERS_SHA, 'morphology_sha256': MORPHOLOGY_SHA,
        'source_trace_sha256': sha256(source/'trace.json'), 'path_provenance_sha256': sha256(paths/'provenance.json'),
        'tool_sha256': sha256(__file__), 'class_sha256': sha256('tools/banc_m3_cell_size.py')}
    write_json(out/'spec.json', spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (out/'cell-size.py').write_bytes(Path('tools/banc_m3_cell_size.py').read_bytes())
    if sha256(parameters) != PARAMETERS_SHA or sha256(morphology) != MORPHOLOGY_SHA:
        raise ValueError('Frozen physiology evidence changed')
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    initial = StateStore.load(source/'start')
    source_trace = json.loads((source/'trace.json').read_text())
    metadata = json.loads((paths/'provenance.json').read_text())
    ids = np.array(metadata['indices'][:188], dtype=np.int64)
    if len(ids) != 188 or any(graph.nodes[i] != n for i,n in zip(ids, metadata['nodes'])):
        raise ValueError('Source core/relay alignment changed')
    root_ids = np.array([n['root_id'] for n in graph.nodes])
    with np.load(parameters) as p, np.load(morphology) as m:
        exact(root_ids, p['root_ids'], 'Parameter root order')
        exact(root_ids, m['root_ids'], 'Morphology root order')
        np.testing.assert_array_equal(p['morphology_area_tau'], np.full(len(root_ids),.02,np.float32))
        np.testing.assert_array_equal(p['morphology_area_cap'], np.full(len(root_ids),200.,np.float32))
        a = p['morphology_area_a'][ids].copy()
        threshold = p['morphology_area_threshold'][ids].copy()
        measured = np.isfinite(m['reference_area_um2'][ids]) & (m['reference_area_um2'][ids] > 0)
        predicted = ~measured & m['empirical_domain_eligible'][ids]
        provenance = {'nodes': metadata['nodes'][:188], 'indices': ids.tolist(),
            'core_groups': metadata['core_groups'], 'measured_area': measured.tolist(),
            'empirical_area': predicted.tolist(), 'assumed_area': (~measured & ~predicted).tolist(),
            'measured_count': int(measured.sum()), 'predicted_count': int(predicted.sum()),
            'assumed_count': int((~measured & ~predicted).sum()),
            'a': a.tolist(), 'threshold': threshold.tolist(),
            'claim_boundary': 'Cell-size electrical conversion is a hypothesis; no functional rhythm labels are inferred from anatomy alone.'}
    write_json(out/'provenance.json', provenance)
    baseline = CellSizeTrial(graph, initial, ids, np.ones(188), np.full(188,7.5))
    try:
        baseline_state = baseline.snapshot()
        exact(initial['network'], baseline_state['network'], 'Identity physiology patch')
        baseline.advance(1)
        row = capture_control(baseline)
        for key in ('qpos','qvel','ctrl','joint_targets_rad','adhesion','tendon_inputs','motor_rates'):
            exact(source_trace['samples'][1][key], row[key], 'Identity patch first control '+key)
    finally:
        baseline.close()
    session = CellSizeTrial(graph, initial, ids, a, threshold)
    rows = [capture_control(session)]
    observations = []
    clone_future = None
    restored = False
    model_s = .005
    reason = 'duration'
    def observe():
        n = session.network
        observations.append(np.stack((n.rate[ids].cpu().numpy().copy(),
            n.adaptation[ids].cpu().numpy().copy(),
            n._current(n.rate)[ids].cpu().numpy().copy(),
            n.drive[ids].cpu().numpy().copy()),axis=1))
    try:
        StateStore.save(out/'start', session.snapshot())
        observe()
        for tick in range(1,441):
            if time.perf_counter()-began > 145:
                reason = 'wall_limit'
                break
            session.advance(1)
            model_s += .005
            rows.append(capture_control(session))
            observe()
            if session.fault or session.body.fault or session.body.nonfoot_contact() or rows[-1]['upright'] < .9:
                reason = 'health_guard'
                break
            if tick == 200:
                StateStore.save(out/'checkpoint-1s', session.snapshot())
                clone = CellSizeTrial.from_checkpoint(graph, StateStore.load(out/'checkpoint-1s'))
                try:
                    clone.advance(10)
                    model_s += .05
                    clone_future = clone.snapshot()
                    StateStore.save(out/'clone-future',clone_future)
                finally:
                    clone.close()
            if tick == 210:
                exact(clone_future,session.snapshot(),'Cell-size disk future')
                restored = True
        if not session.fault and not session.body.fault:
            StateStore.save(out/'final',session.snapshot())
        frame = session.frame()
    finally:
        np.savez_compressed(out/'observations.npz', values=np.asarray(observations), ids=ids)
        write_json(out/'trace.json',{'schema':'flylab.m3-cell-size-trace.v1','graph_hash':graph.hash,
            'model_hash':session.network.identity,'adapter_hash':session.adapter.identity,
            'body_hash':session.body.model_hash,'parameters':initial['parameters'],'samples':rows})
        session.close()
    analysis = phase = None
    if len(rows)>102:
        loaded = load_model(out/'trace.json')
        analysis,_,_ = analyze(loaded)
        t = np.array([r['time_s'] for r in rows])
        phase = phases(loaded['ap'][(t>=.5)&(t<2.2)])
    observation = np.asarray(observations)
    margin = observation[:,:,2]+observation[:,:,3]-observation[:,:,1]-threshold
    relative_slope = np.where(margin>0,1-np.tanh(a*margin/200)**2,0)
    actual_slope = a*relative_slope
    np.savez_compressed(out/'operating-range.npz', margin=margin,
        relative_slope=relative_slope, actual_slope=actual_slope)
    result = {'experiment':spec['experiment'],'validity':'valid_full_feedback_trial',
        'adoption':'pending_adjudication','milestone_outcome':'pending','reason':reason,
        'controls':len(rows)-1,'disk_future_exact':restored,'frame':frame,'analysis':analysis,
        'actual_AP_unrestricted_phase':phase,'minimum_upright_entire':min(r['upright'] for r in rows),
        'model_s':model_s,'wall_s':time.perf_counter()-began,
        'observation_columns':['rate','adaptation','CUDA_recurrent_current','direct_drive']}
    write_json(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('analysis','frame')}),flush=True)
    print(json.dumps({'signed_forward_mm':frame['signed_forward_mm']}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','paths','parameters','morphology','graph','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output directory must be new')
    try:
        run(args.source,args.paths,args.parameters,args.morphology,args.graph,args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out/'execution-error.json',{'error':type(error).__name__,
                'message':str(error),'validity':'incomplete','hypothesis_verdict':'not_tested'})
        raise
