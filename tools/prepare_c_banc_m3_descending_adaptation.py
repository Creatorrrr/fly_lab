"""Prepare a two-source-cell adaptation test without changing graph or inputs."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.banc_m3_intrinsic_recruitment import IntrinsicRecruitmentTrial
from tools.banc_m3_rhythmic_recruitment import RecruitmentTrial
from tools.banc_m3_synaptic_efficacy import EfficacyTrial
from tools.verify_c_banc_observation import exact


def run(root, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    source = root/'e67b-source-premotor-path-delay'
    evidence = root/'e68-premotor-timing-body/descending-transient-diagnosis.json'
    spec = {'experiment': 'M3-E69-two-DN-adaptation-candidate',
        'question': 'Is the assumed slow adaptation of the two stimulated DNg100 cells necessary for the transient loss of their output and subsequent coordination?',
        'basis': 'E68rightDNg100 at3.480s: drive5000 + recurrent(-3891.247) -adaptation1584.206 -threshold7.5 = -482.953. On the same recorded recurrent input, removing adaptation would leave at least1101.253. This arithmetic motivates a causal intervention but is not a replayed or actual counterfactual result.',
        'intervention': 'Keep exact E67bsource weights/taus/motor interface/initial state and all externalinputs. Only two existing DNg100 cells720575941500851362 and720575941626500746 change adaptation_gain8 to0. Existing adaptation equation and CUDA kernel are used; no periodicreset, rate clamp, newdrive, artificialphase or graph change. Other158704cells retain their prior individual physiology.',
        'scope': 'This is an engineered physiology hypothesis, not measured DNg100 electrophysiology. Test the same E67b/E68context with a known transient, while preserving the more stable E65/E66candidate. Preparation alone cannot support the hypothesis.',
        'validity': 'Original source identity/class and previous adaptation8 verified. Onlytwo adaptation membership IDs added; allcalibration arrays/26tau values/graph/weights/body/initialr/z/drive/mask/tick exact. Saved candidate disk restoration exact. No neural or walking advance during preparation.',
        'next_test': 'One actual fullBANC/body run under same10s/F1-F8/world/native contact/healthguard. Record bilateralDN rates/drive/current/adaptation and actual feet. Removing DN drops without gait improvement is partial mechanism evidence only. Do not weaken finalgates or repeat gain grids.',
        'limits': 'OneCUDA setup <=90wallseconds, zero experimental neural/physical advance; F6/M2/UI unchanged.',
        'calibration_sha256': sha256(source/'calibration.npz'), 'evidence_sha256': sha256(evidence),
        'tool_sha256': sha256(__file__)}
    write_json(out/'spec.json', spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    graph = GraphStore.load(graph_path)
    dns = np.array(graph.resolve(['flywire:banc:888:720575941500851362',
        'flywire:banc:888:720575941626500746']))
    original_cal = dict(np.load(source/'calibration.npz'))
    cal = {k: v.copy() for k, v in original_cal.items()}
    meta = json.loads((source/'muscle-provenance.json').read_text())
    if any(graph.nodes[i]['cell_type'] != 'DNg100' for i in dns) or np.intersect1d(dns, cal['zero_adaptation_ids']).size:
        raise ValueError('Two existing previously adapting DNg100 cells required')
    cal['zero_adaptation_ids'] = np.union1d(cal['zero_adaptation_ids'], dns)
    np.testing.assert_array_equal(np.setdiff1d(cal['zero_adaptation_ids'], original_cal['zero_adaptation_ids']), np.sort(dns))
    for key, value in original_cal.items():
        if key != 'zero_adaptation_ids':
            np.testing.assert_array_equal(cal[key], value)
    hypothesis = {'source_nodes': [graph.nodes[i] for i in dns], 'original_gain': 8.,
        'new_gain': 0., 'adaptation_tau_s': .3, 'runtime_reset': False, 'external_drive_changed': False,
        'physiological_validation': 'not_measured; engineered causal test'}
    meta['descending_adaptation_hypothesis'] = hypothesis
    np.savez_compressed(out/'calibration.npz', **cal)
    write_json(out/'muscle-provenance.json', meta)
    write_json(out/'descending-hypothesis.json', hypothesis)
    original = StateStore.load(root/'e32-core-adaptation/candidate/start')
    if original['parameters']['adaptation_gain'] != 8 or np.intersect1d(original['network']['core_adaptation']['ids'], dns).size:
        raise ValueError('Original per-DN adaptation8 required')
    base = EfficacyTrial(graph, original, cal['edge_ids'], cal['source_values'], cal['values'],
        zero_adaptation_ids=cal['zero_adaptation_ids'])
    s = IntrinsicRecruitmentTrial(RecruitmentTrial(base, cal, meta['muscles']), cal['tau_ids'], cal['tau_values'])
    try:
        gains = np.full(s.network.n, 8, np.float32)
        gains[np.union1d(original['network']['core_adaptation']['ids'], cal['zero_adaptation_ids'])] = 0
        np.testing.assert_array_equal(s.network._cell_gain.cpu().numpy(), gains)
        neutral = s.snapshot()
        exact(original['body'], neutral['body'], 'Original body')
        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
            exact(original['network'][key], neutral['network'][key], 'Original neural '+key)
        StateStore.save(out/'candidate-neutral', neutral)
    finally:
        s.close()
    clone = IntrinsicRecruitmentTrial.from_checkpoint(graph, StateStore.load(out/'candidate-neutral'))
    try:
        exact(neutral, clone.snapshot(), 'Entire two-DN candidate disk state')
    finally:
        clone.close()
    result = {'experiment': spec['experiment'], 'validity': 'valid_prepared_source_physiology_candidate',
        'hypothesis_verdict': 'not_tested', 'adoption': 'not_applicable', 'milestone_outcome': 'pending',
        'changed_neuron_ids': dns.tolist(), 'neutral_disk_exact': True,
        'actual_neural_model_s': 0, 'actual_physical_model_s': 0, 'wall_s': time.perf_counter()-began}
    write_json(out/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'graph', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.graph, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out/'execution-error.json', {'validity': 'incomplete_preparation',
                'hypothesis_verdict': 'not_tested', 'error': type(error).__name__, 'message': str(error)})
        raise
