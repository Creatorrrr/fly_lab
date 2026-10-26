"""Fit two positive anatomical front-knee delivery scalars on exposed commands."""

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
from tools.banc_m3_rhythmic_recruitment import AnatomicalRecruitmentAdapter
from tools.verify_c_banc_observation import exact


def positive_scalar(source, target):
    """One unconstrained least-squares solution, rejected outside local bounds."""
    source, target = np.asarray(source), np.asarray(target)
    if (source.ndim != 1 or source.shape != target.shape or len(source) < 10
            or not np.isfinite(source).all() or not np.isfinite(target).all()
            or source @ source <= 1e-12):
        raise ValueError('Finite observed nonzero delivery required')
    gain = float((source @ target) / (source @ source))
    if not .5 <= gain <= 3:
        raise ValueError('Derived positive gain lies outside the fixed local [.5,3] bounds')
    return gain


def run(root, graph_path, out):
    out.mkdir(parents=True, exist_ok=False)
    source = root / 'e69-two-DN-adaptation'
    teacher = root / 'e50-tonic-full-body'
    reference = root / 'e42-source-aligned-reference'
    spec = {'experiment': 'M3-E72-front-knee-delivery-candidate',
        'question': 'Does restoring balanced front-knee command amplitudes reduce the yaw bias observed even under coordinated neural teacher?',
        'basis': 'E42 [.5,2.2] front knee command widthsLF/RF1.2565/1.2565rad; E50 .5266/.9729; E70 .5789/1.0759. E42yaw-.01385rad versusE50+.61312 andE70+.58232. Tracking RMS.0017-.0019rad and exactE71command/body replay rule out a large servo-tracking loss. This association does not establish the cause of turning; silentRHcoxa muscles and middleleg stance remain alternatives.',
        'intervention': 'Only moment rows5(LFtibia) and26(RFtibia) in E69anatomicaldecoder receive independently derived positive scalars. Rest, within-row muscle signs/ratios, all source neural weights/26taus/per-cell adaptation/inputs/pool/filter/tendons/body/world and whole zero start unchanged.',
        'fit': 'Use E50recorded actual96muscle activation and E42recorded command, same timestamps[1.5,2.2). Let x=.02*activation@oldmoment[row], y=referencecommand[row]-fixedrest[row]. Compute once gain=x@y/(x@x); require[.5,3], no search/clipping/phase shifts/refit. Verify reconstruction equals original E50 commands before fitting.',
        'decision': 'A valid scalar fit is preparation only. One fullBANC/no teacher movingbody test with original10s/F/world/nativeguard. Local mechanism prediction: matched[.5,2.2] absolute yaw drift decreases at least30% fromE70 while knee widths stay above frozen F5bounds. FullF3-F5 must still be evaluated; engineering fit is not biological moment calibration.',
        'limits': 'One CUDA setup <=90wallseconds, actual neural/physicsadvance0, noM2/UI/F6exposure.',
        'source_sha256': sha256(source / 'calibration.npz'),
        'signals_sha256': sha256(teacher / 'signals.npz'),
        'reference_sha256': sha256(reference / 'commands.npz'), 'tool_sha256': sha256(__file__)}
    write_json(out / 'spec.json', spec)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    began = time.perf_counter()
    original_cal = dict(np.load(source / 'calibration.npz'))
    cal = {k: v.copy() for k, v in original_cal.items()}
    signals = np.load(teacher / 'signals.npz')
    ref = np.load(reference / 'commands.npz')
    observed_rows = json.loads((teacher / 'trace.json').read_text())['samples'][1:]
    reference_rows = json.loads((reference / 'trace.json').read_text())['samples'][1:]
    np.testing.assert_array_equal(signals['time_s'], ref['time_s'])
    np.testing.assert_array_equal(ref['command'], [r['joint_targets_rad'] for r in reference_rows])
    time_s = signals['time_s']
    activation = signals['muscle_activation']
    reconstructed = cal['rest'] + .02 * activation @ cal['moment'].T
    np.testing.assert_allclose(reconstructed, [r['joint_targets_rad'] for r in observed_rows], rtol=0, atol=2e-14)
    selected = (time_s >= 1.5) & (time_s < 2.2)
    fits = []
    for row, leg in ((5, 'lf'), (26, 'rf')):
        x = .02 * activation[selected] @ cal['moment'][row]
        y = ref['command'][selected, row] - cal['rest'][row]
        gain = positive_scalar(x, y)
        cal['moment'][row] *= gain
        fits.append({'row': row, 'leg': leg, 'gain': gain,
            'before_RMS_rad': float(np.sqrt(np.mean((x-y)**2))),
            'after_RMS_rad': float(np.sqrt(np.mean((gain*x-y)**2))),
            'before_command_width_rad': float(np.diff(np.percentile(x, [5, 95]))[0]),
            'after_command_width_rad': float(np.diff(np.percentile(gain*x, [5, 95]))[0])})
    for key, value in original_cal.items():
        if key != 'moment':
            np.testing.assert_array_equal(cal[key], value)
    untouched = np.setdiff1d(np.arange(42), [5, 26])
    np.testing.assert_array_equal(cal['moment'][untouched], original_cal['moment'][untouched])
    np.testing.assert_array_equal(np.sign(cal['moment']), np.sign(original_cal['moment']))
    meta = json.loads((source / 'muscle-provenance.json').read_text())
    hypothesis = {'fits': fits, 'fixed_rest': True,
        'exposed_fit_window_s': [1.5, 2.2], 'runtime_waveform': False,
        'physiological_validation': 'not_measured; source-sign-preserving delivery hypothesis'}
    meta['front_knee_delivery_hypothesis'] = hypothesis
    np.savez_compressed(out / 'calibration.npz', **cal)
    write_json(out / 'muscle-provenance.json', meta)
    write_json(out / 'delivery-hypothesis.json', hypothesis)
    graph = GraphStore.load(graph_path)
    original = StateStore.load(source / 'candidate-neutral')
    s = IntrinsicRecruitmentTrial.from_checkpoint(graph, original)
    try:
        if 'lf_tibia' not in s.body.joint_names[5] or 'rf_tibia' not in s.body.joint_names[26]:
            raise ValueError('Expected source front tibia joint order')
        np.testing.assert_array_equal(s.adapter.activation, np.zeros(96))
        exact(original_cal['moment'], s.adapter.calibration['moment'], 'Original candidate moment')
        s.adapter = AnatomicalRecruitmentAdapter(s.adapter.original, cal, meta['muscles'])
        neutral = s.snapshot()
        for key, value in original.items():
            if key != 'adapter':
                exact(value, neutral[key], 'Unchanged candidate '+key)
        StateStore.save(out / 'candidate-neutral', neutral)
    finally:
        s.close()
    restored = IntrinsicRecruitmentTrial.from_checkpoint(graph, StateStore.load(out / 'candidate-neutral'))
    try:
        exact(neutral, restored.snapshot(), 'Whole front-knee candidate disk state')
    finally:
        restored.close()
    result = {'experiment': spec['experiment'], 'validity': 'valid_prepared_anatomical_delivery_candidate',
        'hypothesis_verdict': 'not_tested', 'adoption': 'not_applicable', 'milestone_outcome': 'pending',
        'fits': fits, 'neutral_disk_exact': True, 'actual_neural_model_s': 0, 'actual_physical_model_s': 0,
        'wall_s': time.perf_counter()-began}
    write_json(out / 'result.json', result)
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
            write_json(args.out / 'execution-error.json', {'validity': 'incomplete_preparation',
                'hypothesis_verdict': 'not_tested', 'error': type(error).__name__, 'message': str(error)})
        raise
