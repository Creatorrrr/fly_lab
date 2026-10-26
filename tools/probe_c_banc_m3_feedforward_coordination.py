"""Source-existing six-core coupling with one incoming relay per core."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.probe_c_banc_m3_pair_coupling import phase_summary, rate_step

PATHS = [(2, 1, '720575941563370038'), (1, 2, '720575941439734290'),
    (2, 5, '720575941703131610'), (5, 4, '720575941595194149'),
    (1, 3, '720575941461251027'), (3, 0, '720575941633363692')]


def run(root, out, sensitivity_fit=None):
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': 'M3-E59-six-source-feedforward-coupling',
        'question': 'Can the validated LF/RF intrinsic pair entrain both source-existing LF->LM->LH and RF->RM->RH chains without imposed phase?',
        'intervention': 'Same E52source24core recurrence94edges/tau/current. Keep E57LF/RFpositivepair and add four distinct IN19A006 source relays. Every designated inputeffective1.1707944/output1.4869378078382962,relaytau20ms/gain0/otherfixedfield0. Every core receives exactly one selected relay. Source missingedges stayabsent; all12efficacies .001..10.',
        'scope': 'Reduced30cell diagnostic with same E52constantcorefield. Hidden original relay self/cross inputs are absent from reduced model; eventual full model must weaken all nondesignated receiveredgesx.001, preserve original nonselected coreoutputsx.001 and verify actual neural/body feedback.',
        'arms': ['zero', 'exposed_six_offsets'],
        'initial_state': 'Zero all30rates versus already exposed E52lastcycles with groupg initialphaseg/6 and allrelay0. No subsequent phasecontroller, waveform, reset or learnedphasevariable. Neural state diversity is diagnostic, not F6bodypose exposure.',
        'screen': 'Both5s arms: all3 anatomical LR E1peakphase pairs and all6 directed sourcepath pairs in[3.5,5) must satisfy bothrhythms width>=20/completecycles>=8/CV<=.3 and phase mean distancefrom.5<=.15,R>=.8. EachLR pair mean must agree acrossstartswithin.1. These engineering neural criteria do not replace F4actualfeet.',
        'validity': 'Existing pathindices/weights/signs and distinctrelayroster checked; everycore one input, relayincoming/output normalization within frozenbounds. Inzeroarm LF/RF nativefirst.2s must numericallymatch E57coupled_zero, since newbranches provide no reducedfeedback to root.',
        'decision': 'Pass permits fullBANCsource30cell implementation then movingbody test. Failure diagnoses conditional entrainment and chain asymmetry; do not add missing edges or relaxscreen.',
        'limits': 'CPU2x5reducedneural seconds,<=60wallsec,0actualfullBANC/physical; F6unexposed/M2UIpreserved/noadoption.',
        'source_sha256': sha256(root / 'e34-coordination-field/observations.npz'),
        'e57_sha256': sha256(root / 'e57-native-fitted-positive-pair/parameters.npz'),
        'tool_sha256': sha256(__file__)}
    correction = None
    if sensitivity_fit is not None:
        correction = json.loads(sensitivity_fit.read_text())
        if correction.get('validity') != 'valid_native_source_sensitivity':
            raise ValueError('Exact native sensitivity result required')
        spec.update(experiment='M3-E62-native-measured-chain-correction',
            question='Does the E61exact-native Newton step improve chain startup after the assumption that downstream outputs must be strengthened was not supported?',
            intervention='Apply the single E61unconstrained derivative-derived Newton step: four downstream positive outputs times .9839929955549317 and sixteen downstream core taus times1.0098085087799304. Existing E59sourcepaths/94coreweights/field/relays and frontpair unchanged. This weakens output1.6percent and slows isolated downstream response. No parameter sweep, phase initialization change or external runtime phase.',
            direction_revision='E61 produced no candidate within its hypothesis-specific output multiplier>=1 interval. Retain that verdict. The actual frozen source-efficacy bounds are [.001,10]; its one Newton solution lies inside them and is tested here as a new local kinetics hypothesis, not an E61 pass.',
            prediction='E61exact tangent predicts hind start difference .08 with unchanged average downstream late period at first order. Finite native response can invalidate that prediction. Original E59two-start five-second/allpath/LR/.1agreement screen is decisive.',
            sensitivity_sha256=sha256(sensitivity_fit))
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    source = np.load(root / 'e52-homologue-oscillators/parameters.npz')
    history = np.load(root / 'e52-homologue-oscillators/native-state.npz')
    observed = np.load(root / 'e34-coordination-field/observations.npz')
    meta = json.loads((root / 'e34-coordination-field/provenance.json').read_text())
    reference = np.load(root / 'e57-native-fitted-positive-pair/coupled_zero.npz')
    e1 = np.array([int(np.flatnonzero((source['groups']==g)&(source['cell_type_index']==0))[0]) for g in range(6)])
    relays = np.array([next(i for i,n in enumerate(meta['nodes']) if n['root_id']==root_id) for _,_,root_id in PATHS])
    if len(set(relays)) != 6 or sorted(p[1] for p in PATHS) != list(range(6)):
        raise ValueError('Six distinct source relays and exactly one incoming path per core required')
    weight = np.zeros((30,30)); weight[:24,:24] = source['weights']
    original = np.array([[observed['local_weights'][relay,e1[a]],observed['local_weights'][e1[b],relay]]
        for (a,b,_),relay in zip(PATHS,relays)])
    effective = np.tile([1.1707944,1.4869378078382962],(6,1))
    if correction is not None:
        effective[2:,1] *= correction['proposed_output_multiplier']
    efficacy = effective/original
    if not np.isfinite(efficacy).all() or np.any((efficacy < .001)|(efficacy > 10)):
        raise ValueError('Existing positive source efficacy bounds required')
    for j,(a,b,_) in enumerate(PATHS):
        weight[24+j,e1[a]] = effective[j,0]
        weight[e1[b],24+j] = effective[j,1]
    current = np.r_[source['current'],np.zeros(6)]
    tau = np.r_[source['tau'],np.full(6,.02)]
    if correction is not None:
        ratio = correction['proposed_downstream_tau_multiplier']
        if not np.isfinite(ratio) or not .9 <= ratio <= 1.1:
            raise ValueError('Predeclared local tau correction bound required')
        tau[:24] *= np.where(np.isin(source['groups'],[0,3,4,5]),ratio,1.)
    np.savez_compressed(out / 'parameters.npz', weights=weight,current=current,tau=tau,
        core_ids=source['ids'],relay_ids=observed['indices'][relays],source_path_weights=original,
        path_weights=effective,efficacy=efficacy,
        source_E1_ids=source['ids'][[e1[a] for a,_,_ in PATHS]],
        target_E1_ids=source['ids'][[e1[b] for _,b,_ in PATHS]],core_groups=source['groups'])
    write_json(out / 'provenance.json',{'paths':[{'source':meta['nodes'][e1[a]],
        'relay':meta['nodes'][relay],'target':meta['nodes'][e1[b]],'original':original[j].tolist(),
        'effective':effective[j].tolist()} for j,((a,b,_),relay) in enumerate(zip(PATHS,relays))]})
    times = np.arange(20001)*.00025
    root_columns = np.array([int(np.flatnonzero((source['groups']==g)&(source['cell_type_index']==k))[0])
        for g in (2,1) for k in range(4)])
    results = {}
    for name in spec['arms']:
        r = np.zeros(30)
        if name != 'zero':
            for g in range(6):
                peaks,_ = find_peaks(history['rate'][:,e1[g]],prominence=20)
                a,b = peaks[-2:]
                index = a+round((b-a)*g/6)
                selected = source['groups']==g
                r[:24][selected] = history['rate'][index,selected]
        rates = [r.copy()]
        try:
            for step in range(1,20001):
                if step%20==0 and time.perf_counter()-began > 55:
                    raise TimeoutError('E59 bounded source entrainment assay exceeded wall limit')
                r = rate_step(r,weight,current,tau)
                rates.append(r.copy())
                if name=='zero' and step<=800:
                    np.testing.assert_allclose(r[root_columns],reference['rate'][step,:8],atol=1e-8,rtol=1e-10)
            if not np.isfinite(rates).all():
                raise ValueError('Finite source30cell dynamics required')
        finally:
            np.savez_compressed(out / (name+'.npz'),time_s=times[:len(rates)],rate=rates)
        rates = np.asarray(rates)
        def pair(a,b,current_rates=rates):
            values = np.zeros((len(times),5))
            values[:,0],values[:,4] = current_rates[:,e1[a]],current_rates[:,e1[b]]
            return phase_summary(times,values)
        anatomical = {label:pair(a,b) for label,a,b in [('LF-RF',2,1),('LM-RM',5,3),('LH-RH',4,0)]}
        edges = {f'{a}->{b}':pair(a,b) for a,b,_ in PATHS}
        results[name] = {'LR':anatomical,'source_paths':edges,
            'screen':all(p['screen'] for p in [*anatomical.values(),*edges.values()])}
        print(json.dumps({'arm':name,'screen':results[name]['screen'],
            'LR':{k:{p:v[p] for p in ('phase_mean','phase_R','screen')} for k,v in anatomical.items()}}),flush=True)
    agreement = {}
    for key in results['zero']['LR']:
        a,b = (results[name]['LR'][key]['phase_mean'] for name in spec['arms'])
        agreement[key] = abs((a-b+.5)%1-.5) if a is not None and b is not None else None
    passed = all(r['screen'] for r in results.values()) and all(d is not None and d<=.1 for d in agreement.values())
    result = {'experiment':spec['experiment'],'validity':'valid_reduced_source_six_core_assay',
        'hypothesis_verdict':'supported' if passed else 'not_supported','screen':passed,
        'adoption':'not_applicable','milestone_outcome':'pending','results':results,'start_agreement':agreement,
        'reduced_neural_model_s':10.,'actual_full_neural_model_s':0,'actual_physical_model_s':0,
        'wall_s':time.perf_counter()-began}
    write_json(out / 'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sensitivity-fit',type=Path)
    args=parser.parse_args()
    try:
        run(args.root,args.out,args.sensitivity_fit)
    except Exception as error:
        if args.out.exists():
            write_json(args.out/'execution-error.json',{'validity':'incomplete','error':type(error).__name__,
                'message':str(error),'hypothesis_verdict':'not_tested'})
        raise
