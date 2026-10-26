"""Data-only E106 first-review correction; never constructs or advances a body."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_c_banc_biological_gait import sha256, write_json
from tools.probe_c_banc_m3_slow_phase_replay import (
    SOURCE,
    late_metrics,
    require_accepted_review,
    require_source_e1_rates,
    require_source_tendons,
)

OUT = ROOT / 'verification/banc-natural-gait-m3-20260919/e106-slow-native-phase-replay'
RUNNER = ROOT / 'tools/probe_c_banc_m3_slow_phase_replay.py'
TEST = ROOT / 'tests/test_c_banc_slow_phase_replay.py'


def run():
    started = time.perf_counter()
    review = json.loads((OUT/'parent-first-review.json').read_text(encoding='utf-8'))
    originals = review['original_artifact_sha256']
    checks = {name: sha256(OUT/name) == expected for name, expected in originals.items()}
    if not all(checks.values()):
        raise ValueError('Original E106 artifact changed after parent first review')
    original_runner = OUT/'sol-first-original-runner.py'
    original_test = OUT/'sol-first-original-test.py'
    if sha256(original_runner) != originals['runner.py']:
        raise ValueError('Sol original runner snapshot mismatch')
    source_review = json.loads((SOURCE/'parent-second-review.json').read_text(encoding='utf-8'))
    require_accepted_review(source_review)
    import numpy as np
    signals = np.load(SOURCE/'signals.npz')
    ids = [53861, 71439, 152280, 92648, 97727, 91167]
    columns = [int(np.flatnonzero(signals['observed_ids'] == graph_id)[0]) for graph_id in ids]
    require_source_e1_rates(signals['rate'][:, columns])
    source_rows = json.loads((SOURCE/'trace.json').read_text(encoding='utf-8'))['samples']
    tendon_keys = source_rows[0]['tendon_inputs']
    source_tendons = {key: np.array([row['tendon_inputs'][key] for row in source_rows])
                      for key in tendon_keys}
    require_source_tendons(source_tendons)
    corrected = {}
    original = {}
    for arm in ('baseline', 'candidate'):
        corrected[arm] = late_metrics(OUT/arm/'trace.json', OUT/arm/'native-contact.npz')
        original[arm] = json.loads((OUT/f'{arm}-late-quality.json').read_text(encoding='utf-8'))
        if corrected[arm]['quality'] != original[arm]['quality']:
            raise ValueError(f'{arm} frozen evaluator result changed')
        for key in original[arm]:
            if key != 'fixed_initial_heading_forward_mm' and corrected[arm][key] != original[arm][key]:
                raise ValueError(f'{arm} unexpected first-review metric change: {key}')
        write_json(OUT/f'sol-first-corrected-{arm}-late-quality.json', corrected[arm])
    expected = {'baseline': 3.2701149952391133, 'candidate': 6.978098803976673}
    for arm, value in expected.items():
        if abs(corrected[arm]['fixed_initial_heading_forward_mm']-value) > 1e-12:
            raise ValueError(f'{arm} planar heading independent reference mismatch')
    correction_wall = time.perf_counter()-started
    tests = {}
    for label, command in (
        ('unit', [sys.executable, '-X', 'utf8', '-m', 'unittest',
                  'tests.test_c_banc_slow_phase_replay', '-v']),
        ('ruff', [sys.executable, '-X', 'utf8', '-m', 'ruff', 'check', str(RUNNER), str(TEST)]),
    ):
        check_start = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        tests[label] = {'returncode': completed.returncode,
                        'wall_s': time.perf_counter()-check_start,
                        'output': completed.stdout+completed.stderr}
        if completed.returncode:
            raise ValueError(f'{label} failed: {tests[label]["output"]}')
    original_ledger = json.loads((OUT/'execution-ledger.json').read_text(encoding='utf-8'))
    original_result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    if (original_result['numerical_screen'] != 'not_supported'
            or any(count != 0 for count in original_result['cycles'].values())):
        raise ValueError('Original numerical failure changed')
    whole = original_ledger['whole_wall_s']
    if whole > 150 or original_ledger['resource_screen'] != 'within_limits':
        raise ValueError('Conservative inclusive per-arm bound unavailable')
    amended = {
        'scope': 'E106 first-review data-only correction; original body and result artifacts preserved',
        'R1': {'original_full_3D_forward_mm': {arm: original[arm]['fixed_initial_heading_forward_mm']
                                               for arm in original},
               'corrected_planar_initial_heading_forward_mm': {
                   arm: corrected[arm]['fixed_initial_heading_forward_mm'] for arm in corrected},
               'late_window': [3.5, 9.0], 'original_evaluator_other_fields_exact': True},
        'R2': {'accepted_E105_review_explicit_true_empty_defects': True,
               'all_original_E1_endpoints_shape_and_finite': True,
               'all_eight_full_source_tendon_series_finite': True,
               'original_E105_source_unchanged': True},
        'R3': {'original_reported_body_only_arm_wall_s': {
                   arm: original_ledger[arm]['wall_s_inclusive_close_persistence']
                   for arm in ('baseline', 'candidate')},
               'original_historic_per_arm_late_diagnostic_cost_s': 'unknown_not_instrumented',
               'conservative_each_arm_inclusive_upper_bound_s': whole,
               'each_arm_inclusive_under_150_by_whole_bound': True,
               'original_whole_wall_s': whole,
               'original_actual_physical_s': original_ledger['actual_physical_s'],
               'original_full_neural_s': 0.,
               'future_clock_accounting_and_health_guard_classification_tested': True},
        'original_numerical_screen': original_result['numerical_screen'],
        'original_cycles': original_result['cycles'],
        'original_yaw_ratio': original_result['late_absolute_yaw_ratio'],
        'original_resource_screen': original_ledger['resource_screen'],
        'original_artifact_manifest_all_exact': checks,
        'original_runner_sha256': sha256(original_runner),
        'corrected_runner_sha256': sha256(RUNNER),
        'original_test_sha256': sha256(original_test),
        'corrected_test_sha256': sha256(TEST),
        'original_E105_trace_sha256': sha256(SOURCE/'trace.json'),
        'original_E105_signals_sha256': sha256(SOURCE/'signals.npz'),
        'data_only_recompute_wall_s': correction_wall,
        'separate_validation': tests,
        'new_body_or_neural_or_fit_advance': False,
    }
    write_json(OUT/'sol-first-correction-amendment.json', amended)
    print(json.dumps({'planar_forward': amended['R1']['corrected_planar_initial_heading_forward_mm'],
                      'original_manifest_exact': all(checks.values()),
                      'data_only_recompute_wall_s': correction_wall,
                      'tests': {key: value['returncode'] for key, value in tests.items()}}), flush=True)


if __name__ == '__main__':
    run()
