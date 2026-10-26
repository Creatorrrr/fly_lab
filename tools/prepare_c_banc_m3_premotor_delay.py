"""Prepare one source premotor time-constant hypothesis from measured transfer lag."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.banc_m3_synaptic_efficacy import EfficacyTrial
from tools.verify_c_banc_observation import exact


def run(root,graph_path,out):
    out.mkdir(parents=True,exist_ok=False)
    source=root/'e65-measured-source-mean-field'
    observed=root/'e66-compensated-intrinsic-body'
    spec={'experiment':'M3-E67b-source-premotor-path-delay-candidate',
        'question':'Can reducing the extra LM inhibitory relay delay remove part of the measured middle-pair transfer offset while preserving the core coordination?',
        'evidence':'E66 matched694eligible samples with same Hilbert phase convention: E1RM-minus-LM mean.58593 passes, actualfeet.69152 fails; relative transfer offset.106506/R.93148. The LF/RF and hind transfer offsets are only-.01427 and+.02767. Earlier E49source circuit adds LM E1->IN16B016->LMpleuralMN relative to direct/core source recruitment. This identifies a timing hypothesis, not causal proof.',
        'execution_revision':'E67single-relay proposal stopped before neural integration: its9.204ms tau violated the unchanged native dt<=min(tau)/50 guard. Preserve that untested preparation. E67b distributes the same total half-offset correction equally over the two existing first-order cells in the LMpath, IN16relay and LMpleuralMN. This keeps both taus above12.5ms without changing native dt or M2kernel.',
        'candidate':'Only IN16B016720575941590291084 and its source-connected LMpleuralMN720575941553503460 taus change from20ms to one common derived value. Use E66 LMmedianperiod; remove half the observed .106506cycle offset over the two cells: tau_new=tan(atan(omega*.02)-2pi*offset/4)/omega. All weights/coretau/adaptation/muscle decoder/current realization remain exactly E65.',
        'validity':'Existing selected sourceIN16 and LMmotor, source inhibitory edge retained, membership in prior input_ids/motor_ids and zero_adaptation_ids. Tau>=50*native_dt and<20ms. Exact E65edge/value/calibration/24coretau arrays, onlytwo appended kinetics IDs. Whole source neutral r/z/drive/body preserved; disk-restored entire candidate exact without any neural advance.',
        'decision':'Preparation is not neural/body evidence. Next one native fullBANC/actualbody run retains original10s/F1-F8/world/health criteria and records IN16 plus all previous source inputcells. No phase/time waveform at runtime. No repeated tau grid if refuted.',
        'limits':'OneCUDA setup<=90wallseconds,zero experimental neural/physicaladvance; F6/M2/UI preserved.',
        'phase_sha256':sha256(observed/'matched-neural-foot-phase.json'),
        'calibration_sha256':sha256(source/'calibration.npz'),'tool_sha256':sha256(__file__)}
    write_json(out/'spec.json',spec);(out/'runner.py').write_bytes(Path(__file__).read_bytes())
    began=time.perf_counter()
    phase=json.loads((observed/'matched-neural-foot-phase.json').read_text())
    quality=json.loads((observed/'result.json').read_text())['quality']
    offset=phase['pairs']['LM-RM']['relative_transfer_offset']['mean_cycles']
    period=quality['events']['lm']['period_s']['median']
    omega=2*math.pi/period
    tau=math.tan(math.atan(omega*.02)-2*math.pi*offset/4)/omega
    if not 50*.00025 <= tau < .02:
        raise ValueError('Measured correction exceeds predeclared native positive local tau range')
    graph=GraphStore.load(graph_path)
    relay=graph.resolve(['flywire:banc:888:720575941590291084'])[0]
    motor=graph.resolve(['flywire:banc:888:720575941553503460'])[0]
    cal=dict(np.load(source/'calibration.npz'))
    meta=json.loads((source/'muscle-provenance.json').read_text())
    if (graph.nodes[relay]['cell_type']!='IN16B016' or relay not in cal['input_ids']
            or relay not in cal['zero_adaptation_ids'] or relay in cal['tau_ids']):
        raise ValueError('Existing isolated LMsource relay with unmodified original tau required')
    if motor not in cal['motor_ids'] or motor not in cal['zero_adaptation_ids'] or motor in cal['tau_ids']:
        raise ValueError('Existing unadapted LMpleural motor required')
    cal['tau_ids']=np.r_[cal['tau_ids'],relay,motor]
    cal['tau_values']=np.r_[cal['tau_values'],tau,tau]
    hypothesis={'source_nodes':[graph.nodes[relay],graph.nodes[motor]],'old_tau_s':.02,'new_tau_s':tau,
        'observed_transfer_offset_cycle':offset,'removed_phase_cycle':offset/2,
        'reference_period_s':period,'predicted_body_LR_mean':quality['pair_phase']['lm-rm']['mean_cycles']-offset/2,
        'scope':'First-order delay design calculation only; actual nonlinear neuron/body verification required'}
    meta['premotor_timing_hypothesis']=hypothesis
    np.savez_compressed(out/'calibration.npz',**cal)
    write_json(out/'muscle-provenance.json',meta);write_json(out/'timing-hypothesis.json',hypothesis)
    original=StateStore.load(root/'e32-core-adaptation/candidate/start')
    base=EfficacyTrial(graph,original,cal['edge_ids'],cal['source_values'],cal['values'],zero_adaptation_ids=cal['zero_adaptation_ids'])
    np.testing.assert_allclose(base.network.tau[[relay,motor]].cpu().numpy(),.02,atol=1e-9,rtol=0)
    s=IntrinsicRecruitmentTrial(RecruitmentTrial(base,cal,meta['muscles']),cal['tau_ids'],cal['tau_values'])
    try:
        neutral=s.snapshot()
        exact(original['body'],neutral['body'],'Original neutral body')
        for key in ('rate','adaptation','drive','output_mask','tick'):
            exact(original['network'][key],neutral['network'][key],'Original neutral neural '+key)
        StateStore.save(out/'candidate-neutral',neutral)
    finally:s.close()
    restored=IntrinsicRecruitmentTrial.from_checkpoint(graph,StateStore.load(out/'candidate-neutral'))
    try:exact(neutral,restored.snapshot(),'Entire premotor candidate disk state')
    finally:restored.close()
    result={'experiment':spec['experiment'],'validity':'valid_prepared_source_kinetics_candidate',
        'hypothesis_verdict':'not_tested','adoption':'not_applicable','milestone_outcome':'pending',
        'candidate':hypothesis,'neutral_disk_exact':True,'actual_neural_model_s':0,'actual_physical_model_s':0,
        'wall_s':time.perf_counter()-began}
    write_json(out/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','graph','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.root,args.graph,args.out)
