#!/usr/bin/env python3
"""Matched native force/receptor calibration assays; preserve every outcome."""
import argparse
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.integrity import digest, finite, write_json
from flylab.c.muscles import MuscleRig, MUSCLES
from flylab.c.receptors import JointReceptors, ReceptorParameters
from flylab.c.storage import StateStore


def run(out, seconds=.1):
    seconds = finite(seconds, 'assay seconds', .01, 1.)
    if abs(seconds/.001-round(seconds/.001)) > 1e-8: raise ValueError('Integral 1 ms samples required')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    cases = [dict(name='baseline', excitation=[0., 0.]),
             dict(name='flexor_low', excitation=[.1, 0.]), dict(name='extensor_low', excitation=[0., .1]),
             dict(name='flexor_high', excitation=[.35, 0.]), dict(name='extensor_high', excitation=[0., .35]),
             dict(name='coactivation', excitation=[.2, .2]),
             dict(name='motor_disconnected', excitation=[.35, .35], connected=False),
             dict(name='external_positive', excitation=[0., 0.], torque=.01),
             dict(name='external_negative', excitation=[0., 0.], torque=-.01)]
    spec = dict(schema='flylab.muscle-assay.v1', seconds=seconds, sample_dt=.001, cases=cases,
        thresholds=dict(geometry_error=1e-9, projection_error=1e-9, dynamics_error=1e-8),
        direction_hypothesis='flexor raises q relative to baseline; extensor lowers q',
        interpretation='technical calibration fixture; not gait, connectome or biological validation')
    write_json(out/'spec.json', spec)
    report = dict(schema='flylab.muscle-assay-results.v1', status='RUNNING', spec_hash=digest(spec),
        physicalExecuted=False, biological_validation=False, cases=[], checks={},
        contact_validation='NOT_APPLICABLE: fixture has no contacts',
        force_SI_calibration='NOT_VALIDATED', connectome_integration='NOT_IMPLEMENTED')
    write_json(out/'report.json', report); began = time.perf_counter()
    try:
        rig = MuscleRig(); initial = rig.snapshot()
        write_json(out/'model.json', rig.metadata)
        (out/'fixture.xml').write_text(rig.xml)
        shutil.copyfile(rig.source, out/'upstream.xml')
        StateStore.save(out/'initial', initial)
        sample_steps = round(.001/rig.model.opt.timestep)
        receptor_parameters = ReceptorParameters(*map(float, rig.model.jnt_range[0]), dt=.001)
        states = {}
        for case in cases:
            rig.restore(initial); receptor = JointReceptors(receptor_parameters)
            directory = out/case['name']; directory.mkdir()
            rows = [dict(physics=rig.frame(), receptors=None)]
            for _ in range(round(seconds/.001)):
                frame = rig.step(case['excitation'], steps=sample_steps,
                                 torque=case.get('torque', 0.), connected=case.get('connected', True))
                report['physicalExecuted'] = True
                sensory = receptor.step(frame['q_rad'], frame['qdot_rad_s'])
                rows.append(dict(physics=frame, receptors=sensory))
            saved = dict(body=rig.snapshot(), receptors=receptor.snapshot())
            StateStore.save(directory/'final', saved); states[case['name']] = saved
            write_json(directory/'trace.json', rows)
            # Continue, restore through portable files, and replay the same input.
            next_frame = rig.step(case['excitation'], steps=sample_steps, torque=case.get('torque', 0.),
                                  connected=case.get('connected', True))
            next_sensory = receptor.step(next_frame['q_rad'], next_frame['qdot_rad_s'])
            continued = rig.snapshot()['state'].copy()
            loaded = StateStore.load(directory/'final'); rig.restore(loaded['body']); receptor.restore(loaded['receptors'])
            replay_frame = rig.step(case['excitation'], steps=sample_steps, torque=case.get('torque', 0.),
                                    connected=case.get('connected', True))
            replay_sensory = receptor.step(replay_frame['q_rad'], replay_frame['qdot_rad_s'])
            reversals=[]
            for i,muscle in enumerate(MUSCLES):
                initial_arm=rows[0]['physics']['moment_arm_mm'][i]
                changed=[r['physics'] for r in rows if np.sign(initial_arm)*r['physics']['moment_arm_mm'][i] < -1e-9]
                if changed:
                    reversals.append(dict(muscle=muscle,samples=len(changed),initial_arm_mm=initial_arm,
                        first_time_s=changed[0]['seconds'],first_q_rad=changed[0]['q_rad'],
                        minimum_arm_mm=min(r['physics']['moment_arm_mm'][i] for r in rows),
                        maximum_arm_mm=max(r['physics']['moment_arm_mm'][i] for r in rows)))
            result = dict(name=case['name'], elapsed_s=rows[-1]['physics']['seconds'],
                moment_arm_reversals=reversals,
                final_angle_rad=rows[-1]['physics']['q_rad'],
                exact_continuation=np.array_equal(continued, rig.snapshot()['state']) and next_sensory == replay_sensory,
                max_projection_error=max(r['physics']['torque_projection_error'] for r in rows),
                max_dynamics_error=max(r['physics']['dynamics_balance_error'] for r in rows),
                activation_in_range=all(0 <= a <= 1 for r in rows for a in r['physics']['activation']),
                contacts=sum(r['physics']['contacts'] for r in rows))
            write_json(directory/'result.json', result); report['cases'].append(result)
            write_json(out/'report.json', report)
            print(case['name'], result['elapsed_s'], result['final_angle_rad'], 'restore', result['exact_continuation'], flush=True)
        angles = {r['name']: r['final_angle_rad'] for r in report['cases']}
        checks = dict(reference_geometry=max(rig.metadata['derivation_error'].values()) <= spec['thresholds']['geometry_error'],
            force_projects_to_torque=all(r['max_projection_error'] <= spec['thresholds']['projection_error'] for r in report['cases']),
            inertia_matches_torques=all(r['max_dynamics_error'] <= spec['thresholds']['dynamics_error'] for r in report['cases']),
            exact_continuation=all(r['exact_continuation'] for r in report['cases']),
            activation_in_range=all(r['activation_in_range'] for r in report['cases']),
            flexor_extensor_direction=angles['extensor_high'] < angles['extensor_low'] < angles['baseline'] < angles['flexor_low'] < angles['flexor_high'],
            external_torque_direction=angles['external_negative'] < angles['baseline'] < angles['external_positive'],
            disconnected_matches_zero_command=np.array_equal(states['baseline']['body']['state'], states['motor_disconnected']['body']['state']),
            no_contact_in_fixture=all(r['contacts'] == 0 for r in report['cases']))
        report.update(checks=checks, status='PASS' if all(checks.values()) else 'FAIL',
            physiological_status='CALIBRATION_REQUIRED',
            calibration_findings=[dict(case=r['name'],**finding) for r in report['cases'] for finding in r['moment_arm_reversals']],
            simulated_seconds=sum(r['elapsed_s'] for r in report['cases']),
            continuation_simulated_seconds=len(cases)*2*.001,
            receptor_profile=receptor.metadata, model_hash=rig.identity,
            motor_off_semantics='No excitation above upstream control floor; passive force and activation decay remain')
    except Exception as exc:
        if 'rig' in locals(): report['physicalExecuted'] = report['physicalExecuted'] or rig.tick > 0
        if 'directory' in locals():
            write_json(directory/'trace.json', rows)
            write_json(directory/'failure.json', dict(error=repr(exc), physical_tick=rig.tick))
            # NPY preserves even nonfinite failed solver state without executable pickle.
            np.save(directory/'failed_physics_state.npy', rig.snapshot()['state'], allow_pickle=False)
        report.update(status='FAIL' if report['physicalExecuted'] else 'BLOCKED', error=repr(exc))
    report['wall_seconds'] = time.perf_counter()-began
    write_json(out/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True); parser.add_argument('--seconds', type=float, default=.1)
    args = parser.parse_args(); result = run(args.out, args.seconds)
    print(result['status'], result.get('checks'), result.get('error', ''),
          'physiology:',result.get('physiological_status','NOT_VALIDATED'))
    raise SystemExit(0 if result['status'] == 'PASS' else 2 if result['status'] == 'BLOCKED' else 1)
