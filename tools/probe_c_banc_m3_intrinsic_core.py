"""Realize diagnostic core operating currents through existing source DN paths."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_neural import BancTargetEncoder
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.banc_m3_synaptic_efficacy import EfficacyTrial, patch_weights
from tools.probe_c_banc_m3_pair_coupling import phase_summary
from tools.verify_c_banc_observation import exact

TONIC_ROOTS = ['720575941703181786', '720575941603886582', '720575941571519483',
    '720575941564525480', '720575941450546254', '720575941483139514']


def build(graph, root, out, coupling_source=None, *, diagnose_reduced_failure=False,
        tonic_compensation_source=None):
    base = dict(np.load(root / 'e49-fixed-rest-tonic-recruitment/calibration.npz'))
    reduced = dict(np.load(root / 'e52-homologue-oscillators/parameters.npz'))
    meta = json.loads((root / 'e49-fixed-rest-tonic-recruitment/muscle-provenance.json').read_text())
    core, groups = reduced['ids'], reduced['groups']
    tonic_ratio = np.ones(len(core))
    compensation = None
    if tonic_compensation_source is not None:
        old_cal = np.load(tonic_compensation_source / 'calibration.npz')
        history = np.load(tonic_compensation_source / 'observations.npz')
        old_paths = np.load(coupling_source / 'parameters.npz')
        np.testing.assert_array_equal(old_cal['core_ids'], core)
        np.testing.assert_array_equal(old_cal['intrinsic_constant_target'], reduced['current'])
        np.testing.assert_array_equal(old_cal['coupling_ids'], old_paths['relay_ids'])
        late = (history['time_s'] >= 3.5) & (history['time_s'] < 5.)
        selected_current = np.zeros_like(history['coupling_current'][late])
        for k, post in enumerate(old_paths['target_E1_ids']):
            j = int(np.flatnonzero(core == post)[0])
            selected_current[:, j] = (old_paths['path_weights'][k, 1]
                * history['rate'][late, 32+k])
        tonic_mean = history['tonic_current'][late].mean(axis=0)
        residual_mean = (history['remaining_current'][late]
            + history['coupling_current'][late] - selected_current).mean(axis=0)
        tonic_ratio = (reduced['current']-residual_mean)/tonic_mean
        if not np.isfinite(tonic_ratio).all() or np.any(tonic_ratio <= 0):
            raise ValueError('Positive measured source-current compensation required')
        compensation = {'method': 'Single observed mean-field correction at fixed3.5..5s; selected rhythmic coupling excluded, no waveform fit',
            'source_sha256': sha256(tonic_compensation_source / 'observations.npz'),
            'tonic_mean': tonic_mean.tolist(), 'other_field_mean': residual_mean.tolist(),
            'original_target': reduced['current'].tolist(), 'positive_factor_ratio': tonic_ratio.tolist(),
            'scope': 'Actual full-source mean matching only; residual fluctuations and moving-body generalization remain unverified'}
        write_json(out / 'tonic-compensation.json', compensation)
    tonic = np.array(graph.resolve(['flywire:banc:888:' + r for r in TONIC_ROOTS]))
    dns = np.array(graph.resolve(['flywire:banc:888:720575941500851362', 'flywire:banc:888:720575941626500746']))
    original, _ = rate_weights(graph)
    original = BancTargetEncoder(graph, forward_drive=5000).calibrate_weights(original, base['motor_ids'])
    prior = StateStore.load(root / 'e50-tonic-full-body/final-teaching-state')['whole_neural']
    dnrate = prior['rate'][dns].astype(float)
    if np.any(dnrate < 190) or np.any(prior['drive'][dns] != 5000):
        raise ValueError('Observed bilateral existing forward drive5000 required')
    if (any(graph.nodes[i]['cell_type'] != 'INXXX464' or graph.nodes[i]['super_class'] != 'ventral_nerve_cord_intrinsic' for i in tonic)
            or np.intersect1d(tonic, base['input_ids']).size or np.intersect1d(tonic, base['motor_ids']).size):
        raise ValueError('Six distinct source tonic homologues without existing recruitment overlap required')
    dnweights = original[tonic][:, dns].toarray().astype(float)
    tonicrate = np.maximum(200 * np.tanh((dnweights @ dnrate - 7.5) / 200), 0)
    updates = dict(zip(base['edge_ids'].tolist(), base['values'].tolist()))
    for post in np.r_[core, tonic]:
        for edge in range(original.indptr[post], original.indptr[post + 1]):
            if original.data[edge] != 0:
                updates[edge] = float(original.data[edge]) * .001
    def set_value(post, pre, value):
        edges = np.arange(original.indptr[post], original.indptr[post + 1])
        found = edges[original.indices[edges] == pre]
        if len(found) != 1 or original.data[found[0]] == 0:
            raise ValueError('A new or absent source edge cannot be assigned')
        updates[int(found[0])] = float(value)
    for post in tonic:
        for pre in dns:
            if original[post, pre] != 0:
                set_value(post, pre, original[post, pre])
    core_rows = []
    for j, post in enumerate(core):
        for k, pre in enumerate(core):
            if reduced['weights'][j, k] != 0:
                set_value(post, pre, reduced['weights'][j, k])
        sources = np.r_[dns, tonic[groups[j]]]
        weights = original[post][:, sources].toarray().ravel().astype(float)
        expected_rate = np.r_[dnrate, tonicrate[groups[j]]]
        if np.any(weights < 0) or not weights.any():
            raise ValueError('Existing positive source tonic coverage required')
        factor = float(reduced['current'][j] / (weights @ expected_rate)) * tonic_ratio[j]
        if not .001 <= factor <= 10:
            raise ValueError('Required tonic efficacy exceeds frozen bounds')
        for pre, value in zip(sources, weights * factor):
            if value != 0:
                set_value(post, pre, value)
        core_rows.append({'node': graph.nodes[post], 'group': int(groups[j]),
            'sources': [graph.nodes[i]['root_id'] for i in sources], 'source_weights': weights.tolist(),
            'expected_source_rate': expected_rate.tolist(), 'efficacy': factor,
            'desired_constant_current': float(reduced['current'][j])})
    coupling_ids = np.array([], dtype=np.int64)
    if coupling_source is not None:
        pair = dict(np.load(coupling_source / 'parameters.npz'))
        prior_result = json.loads((coupling_source / 'result.json').read_text())
        diagnostic_exception = (diagnose_reduced_failure
            and prior_result.get('validity') == 'valid_reduced_source_six_core_assay'
            and all(r.get('screen') for r in prior_result.get('results', {}).values())
            and len(prior_result.get('results', {})) == 2)
        if not prior_result.get('screen') and not diagnostic_exception:
            raise ValueError('Reduced source coupling must pass or explicitly qualify for a whole-network failure diagnosis')
        coupling_ids = pair['relay_ids']
        source_e1 = pair.get('source_E1_ids', pair['core_ids'][[0, 4]])
        posts = pair.get('target_E1_ids', pair['core_ids'][[4, 0]])
        if (len(np.unique(coupling_ids)) != len(coupling_ids)
                or len(source_e1) != len(coupling_ids) or len(posts) != len(coupling_ids)
                or np.intersect1d(coupling_ids, np.r_[core, tonic, base['input_ids'], base['motor_ids'], dns]).size
                or not np.isin(np.r_[source_e1, posts], core[reduced['cell_type_index'] == 0]).all()):
            raise ValueError('Distinct anatomical coupling relays and source-core E1 endpoints required')
        if len(coupling_ids) == 6:
            np.testing.assert_array_equal(pair['core_ids'], core)
            if len(np.unique(posts)) != 6:
                raise ValueError('Exactly one selected input per core required')
            reduced['tau'] = pair['tau'][:24].copy()
        for k, relay in enumerate(coupling_ids):
            for edge in range(original.indptr[relay], original.indptr[relay + 1]):
                if original.data[edge] != 0:
                    updates[edge] = float(original.data[edge]) * .001
            np.testing.assert_array_equal(np.array([original[relay, source_e1[k]], original[posts[k], relay]]),
                pair['source_path_weights'][k].astype(np.float32))
            set_value(relay, source_e1[k], pair['path_weights'][k, 0])
            set_value(posts[k], relay, pair['path_weights'][k, 1])
        meta.update(coupling_nodes=[graph.nodes[i] for i in coupling_ids],
            coupling_presynaptic_roots=[graph.nodes[i]['root_id'] for i in source_e1],
            coupling_postsynaptic_roots=[graph.nodes[i]['root_id'] for i in posts],
            coupling_source_weights=pair['source_path_weights'].tolist(),
            coupling_effective_weights=pair['path_weights'].tolist(),
            reduced_coupling_screen=prior_result['screen'],
            reduced_failure_diagnostic=diagnostic_exception and not prior_result['screen'])
    ids = np.array(sorted(updates), dtype=np.int64)
    source_values = original.data[ids].copy()
    values = np.array([updates[int(i)] for i in ids], dtype=np.float32)
    effective = values.astype(float) / source_values
    if np.any((effective < .001 * (1-1e-6)) | (effective > 10 * (1+1e-6))):
        raise ValueError('Every selected source efficacy must retain frozen positive bounds')
    changed = patch_weights(original, ids, source_values, values)
    np.testing.assert_array_equal(changed.indptr, original.indptr)
    np.testing.assert_array_equal(changed.indices, original.indices)
    base.update(edge_ids=ids, source_values=source_values, values=values,
        zero_adaptation_ids=np.union1d(base['zero_adaptation_ids'], np.r_[tonic, coupling_ids]),
        core_ids=core, core_groups=groups, core_cell_types=reduced['cell_type_index'],
        tau_ids=core, tau_values=reduced['tau'], intrinsic_tonic_ids=tonic,
        intrinsic_tonic_predicted_rate=tonicrate, intrinsic_constant_target=reduced['current'],
        coupling_ids=coupling_ids)
    np.savez_compressed(out / 'calibration.npz', **base)
    meta.update(intrinsic_core=core_rows, intrinsic_tonic_nodes=[graph.nodes[i] for i in tonic],
        intrinsic_tonic_DN_weights=dnweights.tolist(), intrinsic_tonic_rate_prediction=tonicrate.tolist(),
        intrinsic_tau_s=reduced['tau'].tolist(), no_new_nodes_or_edges=True,
        no_runtime_current_injection=True, physiological_hypothesis_not_validated=True,
        tonic_compensation=compensation)
    write_json(out / 'muscle-provenance.json', meta)
    return base, meta, changed, dns


def run(root, graph_path, out, coupling_source=None, *, diagnose_reduced_failure=False,
        tonic_compensation_source=None):
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': 'M3-E53-intrinsic-source-full-BANC',
        'execution_revision': 'E53b: original E53 stopped before model construction because row enumeration selected explicitly stored zero source weights. Skip those unchanged zeros; retain every source CSR entry and the same scientific candidate/limits. Original error/spec preserved.',
        'question': 'Can the complete BANC network produce the six calibrated intrinsic core rhythms when diagnostic currents are replaced by existing bilateral DNg100/INXXX464 source connections?',
        'intervention': 'Keep E49 downstream anatomy/weights. Six source INXXX464 homologues retain original incoming DNg100 weights, other incomingx.001, adaptation0. All core noncore inputsx.001 except existing bilateral DNg100 and same-group tonic relay edges; a single positive factor per core sets their predicted constant current to E52target. Existing94coreefficacies and measured tau fromE52, no new edges/sign flips. All other158706cell network evolves.',
        'input': 'Use original neutral measured sensory vector and existing forward-navigation5000bilateralDN/-5000navigationport drive, held during neural-only assay. No injected core current/teacher/phase/table/preload/reset. Body remains at original neutral state, so this does not validate moving-body feedback or gait.',
        'screen': 'Six E1 after.5s: width>=20, >=8complete prominence20 cycles, CV<=.3, period.15s±10percent. Also retain late1.5..2.2source tonic rates, current decomposition and observed deviations. No interlegphase requirement until coupling is designed; no apparent movement claim.',
        'validity': 'All frozen source IDs/weights/signs/bounds checked before neural advance; initial fullr/z/drive/mask/tick/body exact. New candidate neutral disk5ms future matches independently restored clone. Neuralonly run separate diagnostic schema with actual ticks and stationarybody exact.',
        'decision': 'Pass supports proceeding to source-existing phasecoupling, not F2/F4 or adoption. Failure localizes tonic reception versus nativecore operatingpoint/recurrence. No blind gain/field/tau grid; F1-F8unchanged/F6unexposed/M2UIpreserved.',
        'limits': 'OneGPU,2.2main neural+.01clone/primary neuralphysical, .01actualphysicalseconds,<=180wallseconds. Finite and tick guards; no fullbodygait attempt.',
        'e49_sha256': sha256(root / 'e49-fixed-rest-tonic-recruitment/calibration.npz'),
        'e52_sha256': sha256(root / 'e52-homologue-oscillators/parameters.npz'),
        'tool_sha256': sha256(__file__), 'interface_sha256': sha256(Path('tools/banc_m3_intrinsic_recruitment.py'))}
    duration = 2.2
    if coupling_source is not None:
        duration = 5.
        spec.update(experiment='M3-E58-coupled-pair-full-BANC',
            execution_revision='First actual full-neural verification of frozen E57source-positive coupling. E53b original full core candidate is reconstructed identically before adding only two existing coupling relays and their source incoming/output efficacies.',
            question='Does E57source-positive LF/RF coupling establish half-cycle phase relation in the complete BANC network with its original held neutral sensory input?',
            intervention='E53bsourcecore/tonic/motor model retained. Two E57IN10B012 relay incoming edges other than designated E1x.001; designated E1effective1.1707944,relayadaptation0/tau20ms. Two original positive relay->oppositeE1 outputs use E57frozen1.4869378; no newedge/signflip/runtimephase/teacher. Otherfourcore recurrence unchanged.',
            screen='Keep six E1 rhythm criteria after.5(width>=20,>=8cycles,CV<=.3,period.15±10percent). Additionally LF/RF observed E1peakphase in[3.5,5):distance ofmeanfrom.5<=.15,R>=.8,and both rhythms>=8cycles. This is actual pair neural coordination only; whole-body and six-core coordination remain untested.',
            decision='Pass permits source-motif extension to all six groups with fixed actual recurrence and source references. Failure localizes full-network relay transmission/background effects before further source adjustments; no blindstrengthgrid.',
            limits='OneGPU5main+.01clone/primary neuralseconds/.01physicalseconds,<=180wallseconds; heldbody/F6unexposed/M2UIpreserved. Native whole r/z and pair current kept.',
            coupling_sha256=sha256(coupling_source / 'parameters.npz'))
        if diagnose_reduced_failure:
            spec.update(experiment='M3-E63-six-source-full-BANC-diagnosis',
                execution_revision='Diagnostic whole-network extension of unchanged E59, whose two per-start all-path/LR screens passed but .1hind start-agreement failed. E60b approximation failed and E62local correction worsened other-start coordination. Their failures remain; do not tune the reduced surrogate indefinitely or treat this as reduced approval.',
                question='Do all six source-coupled cores retain native rhythms and LR coordination in full158706BANC under the original neutral sensory/forward input?',
                intervention='Same E59six source pathways: E57frontpair plus four IN19A006 chain relays. Every nondesignated incoming edge to selected relay times.001, designated original E1input/output use exact E59parameters, adaptation0. Original source24tau/current-realization/E49downstream unchanged. No external waveform, phase, new source edge or initial-state preload.',
                screen='All six original rhythm criteria; all3 LR E1 peak phase in[3.5,5): mean distancefrom.5<=.15,R>=.8 and each>=8cycles. A single original neutral start diagnoses actual full-neural transfer; it does not pass E59cross-neural-start agreement, F4body gait or F6unused bodyposes.',
                decision='Regardless of whether held-body coordination passes, retain actual full source current/relay evidence to localize differences before one bounded movingbody diagnosis. No reduced failure or finalF threshold is changed; adoption remains not_applicable.')
        if tonic_compensation_source is not None:
            spec.update(experiment='M3-E65-measured-source-mean-field',
                question='Does matching the measured full-network mean core field to the original calibrated operating point improve six-core coordination?',
                intervention='Retain E63sourcecore94/coupling6/tau/E49motor/initialstate. Only each existing DN+sameleg INXXX464 positive input factor changes once by (originaldesiredfield - measuredmeanremainingfield - measuredmeannondesignatedcoupling)/measuredmeantonicfield. Use exposed E63[3.5,5). The selected six rhythmic E1 coupling currents are excluded from the compensation. Every actual source efficacy remains within[.001,10]. No direct injected field, waveform, phase, sourceedge/sign change or grid.',
                prediction='Matching the actual mean input may reduce unequal intrinsic frequencies. It does not eliminate fluctuating full-network inputs; failure indicates mean offset alone is insufficient and must not motivate repeated mean-matching until a pass.',
                execution_revision='Single data-derived source-input correction of E63 failed coordination. Preserve E59/E60b/E62/E63 and E64wall-contact/poor-coordination verdicts; no final threshold change.',
                decision='Compare unchanged5s full-neural LR/rhythm screens to E63. Report source-current ratios and remaining fluctuations. If supported, run original neutral actual movingbody; this does not pass F4/F6 or adopt M3.',
                compensation_source_sha256=sha256(tonic_compensation_source / 'observations.npz'))
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    (out / 'interface.py').write_bytes(Path('tools/banc_m3_intrinsic_recruitment.py').read_bytes())
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    cal, meta, weights, dns = build(graph, root, out, coupling_source,
        diagnose_reduced_failure=diagnose_reduced_failure,
        tonic_compensation_source=tonic_compensation_source)
    initial = StateStore.load(root / 'e32-core-adaptation/candidate/start')
    base = EfficacyTrial(graph, initial, cal['edge_ids'], cal['source_values'], cal['values'],
        zero_adaptation_ids=cal['zero_adaptation_ids'])
    s = IntrinsicRecruitmentTrial(RecruitmentTrial(base, cal, meta['muscles']), cal['tau_ids'], cal['tau_values'])
    neural_s, physical_s = 0., 0.
    records, body_saved = [], None
    core, tonic = cal['core_ids'], cal['intrinsic_tonic_ids']
    coupling_ids = cal['coupling_ids']
    observed = np.r_[core, tonic, dns, coupling_ids, cal['motor_ids']]
    try:
        neutral = s.snapshot()
        exact(initial['body'], neutral['body'], 'Original neutral source body')
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            exact(initial['network'][key], neutral['network'][key], 'Original neutral neural ' + key)
        StateStore.save(out / 'candidate-neutral', neutral)
        clone = IntrinsicRecruitmentTrial.from_checkpoint(graph, StateStore.load(out / 'candidate-neutral'))
        try:
            clone.advance(1); neural_s += .005; physical_s += .005
            future = clone.snapshot()
        finally:
            clone.close()
        s.advance(1); neural_s += .005; physical_s += .005
        exact(future, s.snapshot(), 'Intrinsic recruitment full disk future')
        s.close()
        s = IntrinsicRecruitmentTrial.from_checkpoint(graph, StateStore.load(out / 'candidate-neutral'))
        n = s.network
        body_saved = s.body.snapshot()
        position, rotation, _ = s.body.pose()
        drive = s.adapter.encode(sensory_cut=s.cuts['sensory'])
        drive[s.descending] += s.p.descending_drive
        drive += s.navigation.drive(position, rotation)
        if np.any(drive[np.r_[core, tonic, coupling_ids]] != 0) or np.any(drive[dns] != 5000):
            raise ValueError('No injected core/tonic drive and original bilateral forward input required')
        np.savez_compressed(out / 'held-input.npz', drive=drive)
        for control in range(round(duration/.005) + 1):
            if time.perf_counter() - began > 175:
                raise TimeoutError('E53 bounded native CUDA assay exceeded wall allowance')
            if control:
                n.advance_readout(drive, observed, 20); neural_s += .005
            rate = n.readout(observed)
            total = n._current(n.rate)[core].cpu().numpy().copy()
            core_current = weights[core][:, core] @ rate[:24].astype(float)
            tonic_current = weights[core][:, np.r_[tonic, dns]] @ rate[24:32].astype(float)
            coupling_current = weights[core][:, coupling_ids] @ rate[32:32+len(coupling_ids)].astype(float)
            records.append({'rate': rate, 'core_total_current': total,
                'core_recurrent_current': core_current, 'tonic_current': tonic_current,
                'coupling_current': coupling_current,
                'remaining_current': total - core_current - tonic_current - coupling_current})
            if n.tick != control * 20 or not np.isfinite(rate).all():
                raise ValueError('Finite native state and exact neural tick required')
        exact(body_saved, s.body.snapshot(), 'No physical advance in intrinsic neural assay')
        StateStore.save(out / 'final-neural-only', {'schema': 'flylab.diagnostic.intrinsic-neural-only.v1',
            'whole_neural': n.snapshot(), 'body': body_saved, 'native_ticks': n.tick,
            'body_control_tick': s.control_tick, 'autonomous_walking_validated': False})
    finally:
        if records:
            np.savez_compressed(out / 'observations.npz', time_s=np.arange(len(records)) * .005,
                observed_ids=observed, **{key: np.asarray([r[key] for r in records]) for key in records[0]})
        s.close()
    times = np.arange(len(records)) * .005
    rate = np.array([r['rate'] for r in records]); mask = (times >= .5) & (times < duration)
    rows = []
    for g in range(6):
        i = int(np.flatnonzero((cal['core_groups'] == g) & (cal['core_cell_types'] == 0))[0])
        values = rate[mask, i]; peaks, _ = find_peaks(values, prominence=20)
        periods = np.diff(peaks) * .005
        width = float(np.percentile(values, 95) - np.percentile(values, 5))
        cv = float(periods.std() / periods.mean()) if len(periods) else None
        rows.append({'group': g, 'E1_root': graph.nodes[core[i]]['root_id'], 'width': width,
            'complete_cycles': len(periods), 'mean_period_s': float(periods.mean()) if len(periods) else None,
            'CV': cv, 'screen': bool(width >= 20 and len(periods) >= 8 and cv <= .3 and abs(periods.mean()/.15-1) <= .1)})
    late = (times >= duration-.7) & (times < duration)
    measured_tonic = rate[late, 24:30].astype(float)
    pair_summary = None
    lr_summaries = {}
    if coupling_source is not None:
        pairs = [('LF-RF', 2, 1), ('LM-RM', 5, 3), ('LH-RH', 4, 0)] if len(coupling_ids) == 6 else [('LF-RF', 2, 1)]
        for label, a, b in pairs:
            pair_rate = np.zeros((len(times), 5))
            for g, col in ((a, 0), (b, 4)):
                i = int(np.flatnonzero((cal['core_groups'] == g) & (cal['core_cell_types'] == 0))[0])
                pair_rate[:, col] = rate[:, i]
            lr_summaries[label] = phase_summary(times, pair_rate)
        pair_summary = lr_summaries['LF-RF']
    passed = all(r['screen'] for r in rows) and all(r['screen'] for r in lr_summaries.values())
    result = {'experiment': spec['experiment'], 'validity': 'valid_full_BANC_held_sensory_neural_assay',
        'hypothesis_verdict': 'supported' if passed else 'not_supported',
        'screen': passed, 'adoption': 'not_applicable', 'milestone_outcome': 'pending',
        'LF_RF_coordination': pair_summary,
        'LR_coordination': lr_summaries, 'reduced_failure_diagnostic': diagnose_reduced_failure,
        'core': rows, 'tonic_mean': measured_tonic.mean(axis=0).tolist(),
        'tonic_width': (np.percentile(measured_tonic, 95, axis=0)-np.percentile(measured_tonic, 5, axis=0)).tolist(),
        'tonic_predicted': cal['intrinsic_tonic_predicted_rate'].tolist(),
        'disk_future_exact': True, 'neural_only_body_exact': True, 'runtime_external_teacher': False,
        'device': n.device, 'neural_model_s': neural_s, 'physical_model_s': physical_s,
        'wall_s': time.perf_counter() - began}
    write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'graph', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--coupling-source', type=Path)
    parser.add_argument('--diagnose-reduced-failure', action='store_true')
    parser.add_argument('--tonic-compensation-source', type=Path)
    args = parser.parse_args()
    try:
        run(args.root, args.graph, args.out, args.coupling_source,
            diagnose_reduced_failure=args.diagnose_reduced_failure,
            tonic_compensation_source=args.tonic_compensation_source)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete',
                'error': type(error).__name__, 'message': str(error), 'hypothesis_verdict': 'not_tested'})
        raise
