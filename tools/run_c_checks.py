#!/usr/bin/env python3
"""Explicit dependency tiers and counted test evidence; never overwrite a run."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def preflight(tier):
    required = []
    if tier in ('research', 'mps', 'cuda'):
        if importlib.util.find_spec('torch') is None:
            raise RuntimeError('BLOCKED_TORCH: install requirements-research.txt')
        import torch
        required.append(dict(name='torch', version=torch.__version__))
        if tier == 'mps' and not (torch.backends.mps.is_available() and hasattr(torch.mps, 'compile_shader')):
            raise RuntimeError('BLOCKED_MPS: actual Apple MPS hardware and compile_shader required')
        if tier == 'cuda':
            from flylab.c.backend_selection import backend_availability
            cuda=backend_availability()['exp_lif_cuda']
            if not cuda['available'] or not torch.cuda.is_available():
                raise RuntimeError('BLOCKED_CUDA: actual CUDA, CuPy and CUDA-enabled PyTorch required: '+str(cuda))
            required.append(cuda)
    if tier == 'native':
        from flylab.dependencies import dependency_report
        report = dependency_report()
        if not report['ready']:
            raise RuntimeError('BLOCKED_PHYSICS: '+str(report))
        required.append(report)
    return required


def test_ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from test_ids(test)
        else:
            yield test.id()


def main():
    # Keep diagnostic subprocesses consistent with the Windows UTF-8 launcher.
    if os.name=='nt':os.environ['PYTHONUTF8']='1'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', choices=('regression','research','native','mps','cuda'), default='regression')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='flylab.test-run.v1', tier=args.tier, status='RUNNING',
                  python=sys.version, platform=platform.platform(),
                  interpretation='Counts apply only to this test tier; inspect skipped tests and physical campaign evidence separately')
    began = time.perf_counter()
    try:
        report['required_dependencies'] = preflight(args.tier)
        modules = {
            'research':['tests.test_c_research','tests.test_c_contracts.ResearchContractTests'],
            'native':['tests.test_c_repairs','tests.test_c_sensorimotor','tests.test_runtime_optimization','tests.test_c_muscles','tests.test_c_single_joint'],
            'mps':['tests.test_c_mps','tests.test_c_contracts.ResearchContractTests','tests.test_c_research'],
            'cuda':['tests.test_c_cuda','tests.test_c_backend_selection'],
        }
        loader = unittest.TestLoader()
        suite = (loader.discover(str(ROOT/'tests'), top_level_dir=str(ROOT)) if args.tier=='regression'
                 else loader.loadTestsFromNames(modules[args.tier]))
        report['selected_tests'] = list(test_ids(suite))
        with (args.out/'tests.log').open('x') as log:
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        case_id=lambda test:getattr(test,'test_case',test).id()
        not_passed={case_id(t) for t,_ in result.skipped+result.failures+result.errors+result.expectedFailures}
        not_passed.update(case_id(t) for t in result.unexpectedSuccesses)
        report.update(total=result.testsRun, passed=result.testsRun-len(not_passed),
                      expected_failures=[dict(test=t.id(),traceback=m) for t,m in result.expectedFailures],
                      unexpected_successes=[t.id() for t in result.unexpectedSuccesses],
                      skipped=[dict(test=t.id(), reason=reason) for t,reason in result.skipped],
                      failures=[dict(test=t.id(),traceback=message) for t,message in result.failures],
                      errors=[dict(test=t.id(),traceback=message) for t,message in result.errors])
        report['status'] = ('PASS_WITH_SKIPS' if result.skipped else 'PASS') if result.wasSuccessful() else 'FAIL'
        code = 0 if result.wasSuccessful() else 1
    except Exception as exc:
        report.update(status='BLOCKED', error=str(exc))
        code = 2
    report['wall_seconds'] = time.perf_counter()-began
    (args.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('selected_tests','failures','errors')}, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
