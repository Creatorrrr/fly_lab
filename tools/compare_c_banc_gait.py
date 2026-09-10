#!/usr/bin/env python3
"""Bounded comparison of three explicit BANC actuator hypotheses.

All cases use the full connectome and native physics. A completed experiment
is separate from a successful walk. The v2 baseline and task gates are fixed;
this tool never changes a running server or the default binding profile.
"""
import argparse
import copy
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.behavior import trace_sample
from flylab.c.body_identity import source_identity
from flylab.c.engine import CEngine
from flylab.c.graph import GraphStore
from flylab.c.neural import BACKEND_CHOICES
from flylab.c.backend_selection import resolve_backend
from flylab.c.integrity import canonical, digest, file_hash, read_json, write_json
from flylab.c.neuromuscular import build_spec
from flylab.c.ports import PortBindings
from flylab.c.storage import StateStore
from flylab.c.tasks import evaluate
from flylab.sensors import default_world

PROFILES = ('baseline', 'unit_response', 'lift_release', 'combined')
INPUTS = ('feedback', 'DNg100')
BEHAVIOR_FIELDS = ('simTime', 'tick', 'position', 'yaw', 'forward_axis', 'contact',
                   'avoidance', 'motion_enabled', 'obstacle_engaged', 'fault')


def validate_protocol(protocol):
    """Reject registrations that disagree with this fixed experiment."""
    if protocol.get('schema') != 'flylab.banc-gait-improvement.v1':
        raise ValueError('Register the bounded gait-improvement protocol before running')
    expected = dict(seconds=10., seed=42, initial_pose=None, rate_half_Hz=100.,
                    lift_release_mm=.02, neural_dt_s=.0001, physical_dt_s=.0001,
                    control_dt_s=.005)
    fixed = protocol.get('fixed', {})
    if any(key not in fixed or fixed[key] != value for key, value in expected.items()):
        raise ValueError('Protocol parameters disagree with the fixed comparison')
    if [h.get('name') for h in protocol.get('hypotheses', [])] != list(PROFILES[1:]):
        raise ValueError('Protocol must register all three candidates and their ablations')
    if protocol.get('walking_gate') != dict(seconds=10., horizontal_net_mm=5.,
            signed_forward_mm=5., stuck='NOT_DETECTED', fault=None):
        raise ValueError('Walking gates must remain unchanged')


def candidate_bindings(graph, baseline):
    if baseline['neuromuscular']['schema'] != 'flylab.neuromuscular.v2':
        raise ValueError('A v2 baseline is required')
    result = {'baseline': copy.deepcopy(baseline)}
    for name in PROFILES[1:]:
        spec = copy.deepcopy(baseline)
        model = build_spec(graph, 3)
        # Preserve every existing v2 parameter, except the declared adhesion
        # ablation. The new anatomical rows contain the per-unit annotations.
        for key, value in baseline['neuromuscular'].items():
            if key not in ('schema', 'rows', 'muscle_model', 'adhesion_policy'):
                model[key] = copy.deepcopy(value)
        if name == 'unit_response':
            model['adhesion_policy'] = 'neural-tarsus-active-v2'
            model['adhesion_model'] = 'Unchanged v2 neural tarsal balance; no kinematic veto.'
        elif name == 'lift_release':
            model['motor_pooling'] = 'mean-linear-v2'
            model['muscle_model'] = baseline['neuromuscular']['muscle_model']
        spec.update(profile='banc888-gait-candidate-'+name, profile_version=3,
                    neuromuscular=model, biological_validation=False)
        PortBindings(graph, spec)
        result[name] = spec
    return result


def run_case(graph, binding, directory, input_name, seconds=10., backend='auto'):
    directory.mkdir()
    world = default_world()
    world['sources'] = []
    world['obstacles'] = []
    engine = CEngine(graph, PortBindings(graph, binding), world=world,
                     seed=42, mode='C_STRICT', backend=backend)
    started = time.perf_counter()
    samples, rates, counts, voltages = [], [], [], []
    feet_requested, feet_applied, feet_released, force_rows = [], [], [], []
    measured_feet=[]
    result = dict(status='RUNNING', physicalExecuted=True, biological_validation=False)
    try:
        if input_name == 'DNg100':
            ids = [n['id'] for n in graph.nodes if n['cell_type'] == 'DNg100']
            if len(ids) != 2:
                raise ValueError('Expected the two annotated BANC DNg100 neurons')
            engine.schedule(dict(kind='stimulate', ids=ids, amplitude_mV=20.,
                                 duration_controls=round(seconds/.005)+1))
        motor = engine.neuromuscular.motor_indices
        engine.subscribe([graph.nodes[int(i)]['id'] for i in motor])
        write_json(directory/'bindings.json', binding)
        write_json(directory/'provenance.json', engine.provenance())
        write_json(directory/'motor_units.json', [dict(index=int(i), id=graph.nodes[int(i)]['id'],
            cell_type=graph.nodes[int(i)]['cell_type'],
            annotations=graph.nodes[int(i)]['source_annotations']) for i in motor])
        StateStore.save(directory/'initial', engine.checkpoint())
        initial_cpg = digest(engine.body.snapshot()['cpg'])
        with (directory/'trace.jsonl').open('wb') as stream:
            for k in range(round(seconds/.005)+1):
                frame = engine.step() if k else engine.frame()
                row = trace_sample(frame)
                row['measured_feet']=engine.body.contact_probe()
                measured_feet.append(row['measured_feet'])
                stream.write(canonical(row)+b'\n')
                samples.append({key: row[key] for key in BEHAVIOR_FIELDS})
                readout = engine.neural.readout(motor)
                rates.append(readout['rate_Hz'])
                counts.append(readout['spike_count'])
                voltages.append(readout['voltage_mV'])
                groups = row['neuromuscular']['muscles']
                if groups:
                    feet_requested.append([g['adhesion_requested'] for g in groups])
                    feet_applied.append(row['leg_physics']['adhesion'])
                    feet_released.append([g.get('lift_release', False) for g in groups])
                    force_rows.append(row['neuromuscular']['contact']['adhesion_force_bw'])
                if frame['fault']:
                    break
                if k and k % 200 == 0:
                    print(directory.name, f'{frame["simTime"]:.0f}/{seconds:g} model s', flush=True)
        if not engine.fault:
            StateStore.save(directory/'final', engine.checkpoint())
        behavior = evaluate(samples, world, required_seconds=seconds)
        result.update(status='COMPLETE', input=input_name, behavior=behavior,
            fault=engine.fault, simulated_nodes=graph.n,
            cpg_unchanged=initial_cpg == digest(engine.body.snapshot()['cpg']),
            motor_units=len(motor),
            adhesion_requested_fraction=np.mean(feet_requested, axis=0).tolist(),
            adhesion_applied_fraction=np.mean(feet_applied, axis=0).tolist(),
            lift_release_fraction=np.mean(feet_released, axis=0).tolist(),
            mean_adhesion_force_bw=np.mean(force_rows, axis=0).tolist())
        result['foot_measurements']=dict(
            tip_height_range_mm=np.ptp(np.asarray([r['tip_position_native_mm'] for r in measured_feet])[:,:,2],axis=0).tolist(),
            contact_sample_duration_s=(np.asarray([r['floor_contact'] for r in measured_feet][1:]).sum(axis=0)*.005).tolist(),
            integrated_slip_mm=(np.asarray([r['slip_speed_mm_s'] for r in measured_feet][1:]).sum(axis=0)*.005).tolist(),
            sample_dt_s=.005,interpretation='Measured tip movement and sampled floor support; distinct from predicted Jacobian lift')
    except Exception as exc:
        result.update(status='FAIL', error=repr(exc), completed_model_seconds=engine.tick*.0001)
        raise
    finally:
        np.savez_compressed(directory/'motor_signals.npz', rates_Hz=np.asarray(rates),
            cumulative_spikes=np.asarray(counts), sampled_voltage_mV=np.asarray(voltages),
            dt_s=.005)
        result['wall_seconds'] = time.perf_counter()-started
        result['trace_sha256'] = file_hash(directory/'trace.jsonl') if (directory/'trace.jsonl').exists() else None
        write_json(directory/'result.json', result)
        engine.close()
    return result


def run(graph_path, binding_path, protocol_path, out, backend='auto'):
    backend=resolve_backend(backend)
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol_path, out/'registered_protocol.json')
    source = source_identity()
    root = Path(__file__).resolve().parents[1]
    for folder in ('flylab', 'tools'):
        shutil.copytree(root/folder, out/'source'/folder, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(root/'data/releases', out/'source/data/releases')
    shutil.copy2(root/'data/circuit.json', out/'source/data/circuit.json')
    write_json(out/'source_identity.json', source)
    report = dict(status='RUNNING', source=source, protocol_hash=digest(protocol),
                  cases={}, adoption='NOT_EVALUATED', biological_validation=False)
    write_json(out/'report.json', report)
    try:
        graph = GraphStore.load(graph_path)
        profiles = candidate_bindings(graph, read_json(binding_path))
        for name in PROFILES:
            for stimulus in INPUTS:
                case = name+'-'+stimulus
                report['cases'][case] = run_case(graph, profiles[name], out/case, stimulus, backend=backend)
                write_json(out/'report.json', report)
        qualified = [name for name in PROFILES[1:] if all(
            report['cases'][name+'-'+stimulus]['behavior']['task_status'] == 'PASS' and
            report['cases'][name+'-'+stimulus]['cpg_unchanged'] for stimulus in INPUTS)]
        report.update(status='COMPLETE', qualifying_candidates=qualified,
            adoption='HOLDOUT_REQUIRED' if qualified else 'REJECTED_ALL_CANDIDATES',
            model_seconds=sum(r['behavior']['elapsed_s'] for r in report['cases'].values()),
            live_or_default_changed=False)
    except Exception as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        write_json(out/'report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', default='data/acquisitions/banc888-v2-20260909/bundle')
    p.add_argument('--baseline', default='data/banc888/bindings-neuromuscular-v2.json')
    p.add_argument('--protocol', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--backend',choices=BACKEND_CHOICES,default='auto')
    a = p.parse_args()
    report = run(a.graph, a.baseline, a.protocol, a.out, a.backend)
    print(report['status'], report['adoption'])
