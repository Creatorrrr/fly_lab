"""Matched full-body core-adaptation assay; never an external rhythm source."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control
from tools.analyze_c_banc_biological_gait import analyze, load_model, sha256, write_json
from tools.analyze_c_banc_m3_motor_transfer import phases, pool, raw_request
from tools.banc_m3_core_adaptation import CORE_TYPES, CoreAdaptationNetwork
from tools.verify_c_banc_m3_knee_trial import KneeTrial
from tools.verify_c_banc_observation import exact


def source_matrix(network):
    return csr_matrix((network.weights.values().cpu().numpy().copy(),
        network.weights.col_indices().cpu().numpy().copy(),
        network.weights.crow_indices().cpu().numpy().copy()), shape=(network.n, network.n))


class CoreTrial(KneeTrial):
    schema = 'flylab.m3-core-adaptation-walking.v1'

    def __init__(self, graph, knee_source, core_ids, core_gain):
        base = KneeTrial.from_checkpoint(graph, knee_source)
        self.__dict__.update(base.__dict__)
        old = self.network
        saved = old.snapshot()
        candidate = CoreAdaptationNetwork(source_matrix(old),
            *(getattr(old, name).cpu().numpy().copy() for name in ('tau', 'a', 'threshold', 'cap')),
            adaptation_gain=old.adaptation_gain, adaptation_tau_s=old.adaptation_tau_s,
            dt=old.dt, device=old.device, cuda_implementation=old.cuda_implementation,
            capture_steps=old.interval_steps, core_ids=core_ids, core_gain=core_gain)
        if candidate.base_identity != old.identity:
            self.close(); raise ValueError('Non-core physiology/weights unexpectedly changed')
        saved['model_hash'] = candidate.identity
        saved['core_adaptation'] = candidate.snapshot()['core_adaptation']
        candidate.restore(saved)
        self.network = candidate

    @classmethod
    def from_checkpoint(cls, graph, saved):
        if saved.get('schema') != cls.schema:
            raise ValueError('Explicit core trial checkpoint required')
        knee = copy.deepcopy(saved)
        core = knee['network'].pop('core_adaptation')
        knee['schema'] = KneeTrial.schema
        knee['network']['model_hash'] = core['base_model_hash']
        trial = cls(graph, knee, core['ids'], core['gain'])
        try:
            exact(saved, trial.snapshot(), 'Complete core trial state')
        except Exception:
            trial.close(); raise
        return trial


def core_observation(network, ids, internal):
    selected = network.torch.as_tensor(ids, device=network.device)
    rate = network.rate[selected].cpu().numpy().copy()
    adaptation = network.adaptation[selected].cpu().numpy().copy()
    current = network._current(network.rate)[selected].cpu().numpy().copy()
    within = internal.astype(float) @ rate
    return np.stack((rate, adaptation, current, within, current - within), axis=1)


def run(source, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    spec = {'experiment': 'M3-E32-core-adaptation',
        'hypothesis': 'The shared slow adaptation on the source CPG core suppresses useful endogenous motor coordination. Removing only that additional physiology may restore regular neural output through unchanged existing edges.',
        'counterevidence': 'E31 core external currents also vary strongly; the author circuit does not alone establish inter-leg phase locking. Source-area physiology remains unmatched. A faster core alone is not a gait improvement.',
        'intervention': 'Only24cells in four source types change adaptation_gain8 to0 from neutral zero state. Remaining158682cells retain8/.3, all edge weights/signs and source graph, original input/actual senses, E31 knee delivery and body unchanged. No timed resets/forcing/replay inputs.',
        'comparison': 'gain8 per-cell implementation must reproduce E31 all440body/MN controls and1s complete rate/adaptation exactly. Then run gain0 once fromsame source. All recorded cores/source provenance, internal and external recurrent currents, raw motor requests and body retained.',
        'decision': 'Reject as improvement if body worsens, core saturates/silences, or actual body-frame foot AP LR phase coupling remains belowR.5/distance.15 onall3pairs. This2.2s unrestricted-window signal screen is descriptive and cannot pass F4. Raw joint-axis phases are explanatory only because mirrored joint axes need not share a physical sign. Only a healthy better-coordinated output proceeds to original10sF3-F5; no adaptation scalar grid.',
        'limits': 'OneGPU,2x2.2modelsec,<=180wallsec,stopfault/nonfoot/upright<.9. Never change liveM2 or F6 states.',
        'core_types': CORE_TYPES, 'source_trace_sha256': sha256(source / 'trace.json'),
        'tool_sha256': sha256(__file__), 'network_tool_sha256': sha256(Path('tools/banc_m3_core_adaptation.py')),
        'primary_source': 'Pugliese_2026@5626b731; local Figure3 notebook explicit BANC4 roots; original paper core alone does not prove inter-leg gait'}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    (out / 'core-adaptation.py').write_bytes(Path('tools/banc_m3_core_adaptation.py').read_bytes())
    began = time.perf_counter(); graph = GraphStore.load(graph_path)
    ids = np.array([i for i, node in enumerate(graph.nodes) if node.get('cell_type') in CORE_TYPES])
    if len(ids) != 24 or any(sum(graph.nodes[i]['cell_type'] == kind for i in ids) != 6 for kind in CORE_TYPES):
        raise ValueError('Frozen24core mapping changed')
    initial = StateStore.load(source / 'start')
    expected = json.loads((source / 'trace.json').read_text())['samples'][:441]
    old1 = StateStore.load(source / 'checkpoint-1s')
    result = {}; total_model_s = 0.
    for arm, gain in (('baseline', 8.), ('candidate', 0.)):
        destination = out / arm; destination.mkdir()
        session = CoreTrial(graph, initial, ids, gain)
        rows = [capture_control(session)]; observed = []; requests = []; reason = 'duration'
        try:
            StateStore.save(destination / 'start', session.snapshot())
            matrix = source_matrix(session.network); internal = matrix[ids][:, ids].toarray()
            topology = csr_matrix(internal); topology.eliminate_zeros()
            nc, labels = connected_components(topology, directed=False)
            if nc != 6 or not np.all(np.bincount(labels) == 4):
                raise ValueError('Source core connectivity groups changed')
            write_json(destination / 'core-provenance.json', {'nodes': [graph.nodes[i] for i in ids],
                'indices': ids.tolist(), 'groups': labels.tolist(), 'core_weights': internal.tolist(),
                'gain': gain, 'interpretation': 'A group index does not by itself establish which leg it controls'})
            observed.append(core_observation(session.network, ids, internal))
            for tick in range(1, 441):
                if time.perf_counter() - began > 175:
                    reason = 'wall_limit'; break
                session.advance(1); total_model_s += .005
                rows.append(capture_control(session))
                observed.append(core_observation(session.network, ids, internal))
                requests.append(raw_request(session.adapter.original, pool(session.adapter.original, session.rate)))
                if arm == 'baseline':
                    for key in ('qpos', 'qvel', 'ctrl', 'joint_targets_rad', 'adhesion', 'tendon_inputs', 'motor_rates'):
                        exact(expected[tick][key], rows[-1][key], 'Matched E31 ' + key)
                    if tick == 200:
                        state = session.network.snapshot()
                        for key in ('rate', 'adaptation', 'drive', 'output_mask', 'tick'):
                            exact(old1['network'][key], state[key], 'Matched complete neural ' + key)
                if session.fault or session.body.fault or session.body.nonfoot_contact() or rows[-1]['upright'] < .9:
                    reason = 'health_guard'; break
            if not session.fault and not session.body.fault:
                StateStore.save(destination / 'final', session.snapshot())
            frame = session.frame()
        finally:
            write_json(destination / 'trace.json', {'schema': 'flylab.m3-core-trace.v1',
                'graph_hash': graph.hash, 'model_hash': session.network.identity,
                'adapter_hash': session.adapter.identity, 'body_hash': session.body.model_hash,
                'parameters': {**initial['parameters'], 'core_gain': gain}, 'samples': rows})
            np.savez_compressed(destination / 'neural-observation.npz', core_ids=ids,
                core_state=np.asarray(observed), raw_motor_request=np.asarray(requests))
            session.close()
        quality, _, _ = analyze(load_model(destination / 'trace.json')) if len(rows) > 102 else (None, None, None)
        times = np.array([r['time_s'] for r in rows[1:]])
        active = (times >= .5) & (times < 2.2)
        joints = session.adapter.joints
        signal = {name: phases(np.asarray(requests)[active][:, joints[:, joint]])
            for name, joint in (('knee', 5), ('coxa_yaw', 0))} if active.sum() > 20 else {}
        foot_phase = phases(load_model(destination / 'trace.json')['ap'][1:][active]) if active.sum() > 20 else {}
        armresult = {'control_count': len(rows) - 1, 'reason': reason, 'frame': frame,
            'quality': quality, 'raw_signal_phase': signal, 'actual_AP_unrestricted_phase': foot_phase,
            'minimum_upright_entire': min(r['upright'] for r in rows),
            'matched_E31': arm == 'baseline' and len(rows) == 441,
            'current_columns': ['rate', 'adaptation', 'CUDA_recurrent_current', 'float64_core_current', 'CUDA_minus_float64_core']}
        result[arm] = armresult
        write_json(destination / 'result.json', armresult)
        print(json.dumps({'arm': arm, 'controls': len(rows)-1, 'reason': reason,
            'signed_forward_mm': frame['signed_forward_mm'], 'minimum_upright': armresult['minimum_upright_entire'],
            'raw_signal_phase': signal, 'wall_s': time.perf_counter()-began}), flush=True)
        if arm == 'baseline' and len(rows) != 441:
            raise ValueError('Source reproduction did not complete')
    write_json(out / 'result.json', {'experiment': spec['experiment'], 'validity': 'valid_full_closed_loop_comparison',
        'adoption': 'pending_adjudication', 'milestone_outcome': 'pending',
        'models_s': total_model_s, 'wall_s': time.perf_counter()-began, 'results': result})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'graph', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output must be new')
    try:
        run(args.source, args.graph, args.out)
    except Exception as error:
        if args.out.exists():
            write_json(args.out / 'execution-error.json', {'error': type(error).__name__, 'message': str(error),
                'validity': 'incomplete', 'hypothesis_verdict': 'not_tested', 'adoption': 'not_applicable'})
        raise
