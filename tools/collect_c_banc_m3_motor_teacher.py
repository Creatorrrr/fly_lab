"""Training-only motor interventions through existing full BANC connections.

The forced neural states and body replay are teaching data, never an autonomous
walking checkpoint. A student must remove both interventions before evaluation.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flylab.c.banc_walking import BancWalkingSession
from flylab.c.graph import GraphStore
from flylab.c.research_annotations import rate_weights
from flylab.c.storage import StateStore
from flylab.c.target_sense import TargetObservation
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.verify_c_banc_observation import exact


def held_motor_state(rate, adaptation, target, seconds):
    """Analytic forced rate and its *consistent* adaptation at an RK stage."""
    er, ez = math.exp(-seconds / .02), math.exp(-seconds / .3)
    r = target + (rate - target) * er
    z = (adaptation * ez + 8 * target * (1 - ez)
         + 8 * (rate - target) * (.02 / (.02 - .3)) * (er - ez))
    return r, z


class MotorTeacher:
    def __init__(self, network, motor_ids):
        self.network = network
        self.ids = network.torch.as_tensor(motor_ids, device=network.device)
        self.target = network.torch.zeros(len(motor_ids), device=network.device)
        self.correction = network.torch.zeros_like(self.target)

    def step(self):
        n, ids = self.network, self.ids
        dt, r, z = n.dt, n.rate, n.adaptation
        (rm, zm), (re, ze) = self.stage_states(r[ids], z[ids])
        k1, a1 = n._derivative_pair(r, z)
        r2, z2 = r + dt * .5 * k1, z + dt * .5 * a1
        r2[ids], z2[ids] = rm, zm
        k2, a2 = n._derivative_pair(r2, z2)
        r3, z3 = r + dt * .5 * k2, z + dt * .5 * a2
        r3[ids], z3[ids] = rm, zm
        k3, a3 = n._derivative_pair(r3, z3)
        r4, z4 = r + dt * k3, z + dt * a3
        r4[ids], z4[ids] = re, ze
        k4, a4 = n._derivative_pair(r4, z4)
        r.add_(dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        z.add_(dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4))
        self.correction.add_(n.torch.abs(re - r[ids]))
        r[ids], z[ids] = re, ze
        self.after_step()

    def stage_states(self, rate, adaptation):
        return (held_motor_state(rate, adaptation, self.target, self.network.dt*.5),
                held_motor_state(rate, adaptation, self.target, self.network.dt))

    def after_step(self):
        pass

    def capture(self):
        n = self.network
        initial = n.snapshot()
        n.graph = None
        n.interval_steps = 10
        n._step = self.step
        with n.torch.inference_mode():
            n._interval()
            n._interval()
            n.torch.cuda.synchronize()
            n.graph = n.torch.cuda.CUDAGraph()
            with n.torch.cuda.graph(n.graph):
                n._interval()
        n.restore(initial)
        self.correction.zero_()


class ContinuousMotorTeacher(MotorTeacher):
    """Training-only exact v->u->r,z forcing; u and required J stay continuous."""
    def __init__(self, network, motor_ids):
        from scipy.linalg import expm
        super().__init__(network, motor_ids)
        self.activation = network.torch.zeros_like(self.target)
        matrix = np.array([[-50.,0.,50.,0.],[8/.3,-1/.3,0.,0.],
                           [0.,0.,-100.,100.],[0.,0.,0.,0.]])
        self.transitions = [expm(matrix*network.dt*f) for f in (.5,1.)]

    def stage_states(self, rate, adaptation):
        states=[]
        for matrix in self.transitions:
            state=(rate, adaptation, self.activation, self.target)
            states.append(tuple(sum(float(matrix[i,j])*state[j] for j in range(4)) for i in (0,1)))
        return states

    def after_step(self):
        e=math.exp(-self.network.dt/.01)
        self.activation.mul_(e).add_((1-e)*self.target)


def run(source, labels, identification, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    with np.load(labels/'signals.npz') as loaded:
        d={k:loaded[k] for k in loaded.files}
    continuous = 'continuous_drive_tau_s' in d
    encoding_path = Path('verification/banc-natural-gait-m3-20260919/e05-afferent/candidate-source/flylab/c/afferent_rate.py')
    spec = {
        'experiment': 'M3-E12-motor-teacher',
        'hypothesis': 'Dynamically feasible motor teaching propagates patterned activity through existing BANC edges and supplies a substrate for identifying autonomous motor currents',
        'competing_explanation': 'Current fitting may merely read clamped motor activity; the free student may not start or sustain the pattern',
        'intervention': 'Training only: all391 anatomical motor rates and their consistent adaptation follow E11 held-u analytic states at every native RK stage. All other158315 neurons retain original full BANC dynamics and connections. Body replays E11b commands; isolated-cell afferent encoding18 and public fixed forward tonic22 are retained.',
        'labels': 'Every motor in the same existing muscle group receives the same group rate. u is the bounded tanh target, not rate or external current. Required current=threshold+z+cap*atanh(u/cap)-drive for u>0; u=0 is an upper inequality. Tau20ms/cap200/threshold7.5/adaptation8,.3 unchanged.',
        'sampling': 'At2.5ms and5ms of each held5ms input, record actual presynaptic rates, motor states, imposed u and required current. No target or state reset is permitted in a future student.',
        'validity': 'Actual CUDA/full roster; every body qpos/qvel/ctrl byte exact E11b; forced endpoint motor rates within.002 of analytic float64 labels; finite full neural states; custom teaching identity/schema and correction recorded.',
        'decision': 'Collection alone is no autonomous improvement. Current reconstruction and free running from both neutral and taught states are separate gates. All data are development exposed, including the whole2.2s inverse labels. F1-F8 unchanged.',
        'limits': '2.2model sec/300wall sec, one GPU job; no online fit',
        'tool_sha256': sha256(__file__), 'label_sha256': sha256(labels/'signals.npz'),
        'teacher_trace_sha256': sha256(labels/'trace.json'),
        'encoding_sha256': sha256(encoding_path),
    }
    if continuous:
        spec.update(experiment='M3-E15-continuous-motor-teacher',
            intervention='Training only: E14 heldv drives continuous targetu(tau10ms), motorr(tau20ms), consistentz(tau.3/gain8) via exact coupled RK-stage states. All other full BANC dynamics and body replay remain explicit teaching conditions.',
            labels='Actual float32 continuous u at each2.5/5ms sample is transformed to required current, with actual adaptation. All391motor units retain existing muscle membership. No reset of u/r/z between control frames.',
            sampling='2.5/5ms actual presynaptic rates, motor r/z, continuousu and required current. u is continuous across the heldv control changes; no motor external drive introduced.')
    write_json(out/'spec.json', spec)
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    loader = importlib.util.spec_from_file_location('flylab.c._m3_archived_afferent', encoding_path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    saved = StateStore.load(source/'start')
    graph = GraphStore.load(graph_path)
    session = BancWalkingSession.from_checkpoint(graph, saved)
    began = time.perf_counter()
    records = {k: [] for k in ('time_s', 'presynaptic', 'rate', 'adaptation', 'target_activation', 'required_current', 'external_drive', 'source_current', 'correction', 'end_rate')}
    try:
        body, adapter, n = session.body, session.adapter, session.network
        if n.device != 'cuda':
            raise ValueError('Actual CUDA required')
        exact(saved['body'], body.snapshot(), 'E12 initial physical state')
        weights, _ = rate_weights(graph)
        weights = session.navigation.encoder.calibrate_weights(weights, adapter.motor_ids)
        incoming = weights[adapter.motor_ids].copy()
        incoming.eliminate_zeros()
        observed = np.unique(incoming.indices)
        original_model_hash = n.identity
        n.identity = hashlib.sha256((n.identity + spec['experiment'] + spec['label_sha256']).encode()).hexdigest()
        groups = json.loads((identification/'collection-result.json').read_text())['muscles']
        motor_group = np.full(len(adapter.motor_ids), -1)
        for k, g in enumerate(groups):
            if np.any(motor_group[g['columns']] >= 0):
                raise ValueError('Motor belongs to overlapping teaching groups')
            motor_group[g['columns']] = k
        if np.any(motor_group < 0) or len(groups) != d['input_activation'].shape[1]:
            raise ValueError('Incomplete anatomical motor teaching map')
        teacher = (ContinuousMotorTeacher if continuous else MotorTeacher)(n, adapter.motor_ids)
        teacher.capture()
        sensory_ids = np.unique(adapter._sensory_ids)
        encoding = module.AfferentRateEncoding(sensory_ids,
            cap=n.cap.cpu().numpy()[sensory_ids], slope=n.a.cpu().numpy()[sensory_ids],
            threshold=n.threshold.cpu().numpy()[sensory_ids], adaptation_gain=n.adaptation_gain,
            feature_gain=adapter.sensory_gain, target_rate=18)
        tonic = session.navigation.encoder.drive(TargetObservation(0, 20, True))
        if np.count_nonzero(tonic) != 22 or not np.all(tonic[session.descending] == 5000) or not np.all(tonic[session.navigation.encoder.ports] == -5000):
            raise ValueError('M2 forward tonic mismatch')
        trace = json.loads((labels/'trace.json').read_text())['samples'][1:441]
        write_json(out/'topology.json', {
            'graph_hash': graph.hash, 'neurons': graph.n, 'edges': len(graph.indices),
            'motor_ids': adapter.motor_ids.tolist(), 'presynaptic_ids': observed.tolist(),
            'motor_group': motor_group.tolist(), 'muscles': groups,
            'original_model_hash': original_model_hash, 'teacher_model_hash': n.identity,
            'incoming_nonzero_edges': incoming.nnz, 'encoding_hash': encoding.identity,
        })
        max_error = 0.
        for tick, row in enumerate(trace, 1):
            if time.perf_counter()-began > 300:
                raise TimeoutError('E12 wall limit')
            u = d['input_activation'][tick-1, motor_group]
            teacher.target.copy_(n.torch.as_tensor(u, dtype=n.torch.float32, device=n.device))
            teacher.correction.zero_()
            drive = encoding(adapter.encode()) + tonic
            for half in (1, 2):
                n.advance(drive, 10)
                full = n.readout()
                motor_z = n.adaptation[teacher.ids].cpu().numpy().copy()
                actual_u = teacher.activation.cpu().numpy().astype(float) if continuous else u
                required = 7.5 + motor_z + 200*np.arctanh(actual_u/200) - drive[adapter.motor_ids]
                records['time_s'].append((tick-1)*.005 + half*.0025)
                records['presynaptic'].append(full[observed].copy())
                records['rate'].append(full[adapter.motor_ids].copy())
                records['adaptation'].append(motor_z)
                records['target_activation'].append(actual_u.copy())
                records['required_current'].append(required)
                records['external_drive'].append(drive[adapter.motor_ids].copy())
                records['source_current'].append(incoming @ full)
            records['correction'].append(teacher.correction.cpu().numpy().copy())
            records['end_rate'].append(full[adapter.motor_ids].copy())
            max_error = max(max_error, float(np.max(np.abs(full[adapter.motor_ids]-d['neural_rate'][tick-1, motor_group]))))
            if max_error > .002:
                raise AssertionError(f'E12 analytic endpoint disagreement {max_error}')
            body.step_joint_targets(np.asarray(row['joint_targets_rad']), np.asarray(row['adhesion'], bool), tendon_inputs=row['tendon_inputs'])
            for name in ('qpos', 'qvel', 'ctrl'):
                if getattr(body.d, name).tobytes() != np.asarray(row[name]).tobytes():
                    raise AssertionError(f'E12 teacher body mismatch {tick}/{name}')
            session.control_tick = tick
            if tick in (200, 440):
                StateStore.save(out/f'teacher-state-{tick}', {
                    'schema': 'flylab.m3-motor-teaching.v1', 'not_autonomous_checkpoint': True,
                    'spec': spec, 'body': body.snapshot(), 'network': n.snapshot(),
                    'adapter': adapter.snapshot(), 'control_tick': tick,
                    'muscle_activation': d['muscle_activation'][tick-1],
                    'held_target': u, 'encoding_hash': encoding.identity,
                    'continuous_activation': teacher.activation.cpu().numpy().copy() if continuous else None,
                })
            if tick % 100 == 0:
                print(json.dumps({'tick': tick, 'wall_s': time.perf_counter()-began, 'max_rate_error': max_error}), flush=True)
        session._clocks()
        records = {k: np.asarray(v) for k, v in records.items()}
        if any(not np.isfinite(v).all() for v in records.values()) or not bool(n.torch.isfinite(n.adaptation).all()):
            raise ValueError('Nonfinite teaching data')
        np.savez_compressed(out/'signals.npz', **records,
            motor_ids=adapter.motor_ids, observed_ids=observed, motor_group=motor_group,
            moment=d['moment'], rest=d['rest'], command=d['command'], neural_rate=d['neural_rate'],
            incoming_indptr=incoming.indptr, incoming_indices=incoming.indices, incoming_data=incoming.data)
        result = {'experiment': spec['experiment'], 'run_status': 'completed', 'validity': 'valid_teaching_collection',
            'hypothesis_verdict': 'not_yet_tested', 'adoption': 'not_applicable', 'milestone_outcome': 'pending',
            'model_s': session.control_tick*.005, 'wall_s': time.perf_counter()-began,
            'device': n.device, 'neurons': graph.n, 'edges': len(graph.indices),
            'exact_physical_steps': session.control_tick, 'max_analytic_endpoint_rate_error': max_error,
            'mean_abs_native_clamp_correction': float(np.mean(records['correction'])/20),
            'teacher_model_hash': n.identity, 'source_model_hash': original_model_hash,
            'signals_sha256': sha256(out/'signals.npz')}
        write_json(out/'result.json', result)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        np.savez_compressed(out/'partial-signals.npz', **{k: np.asarray(v) for k,v in records.items()})
        write_json(out/'execution-error.json', {'type': type(exc).__name__, 'message': str(exc), 'wall_s': time.perf_counter()-began, 'adoption': 'not_applicable'})
        raise
    finally:
        session.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'labels', 'identification', 'graph', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.labels, a.identification, a.graph, a.out)
