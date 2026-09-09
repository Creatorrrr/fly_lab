"""표준 라이브러리만 사용하는 의존성 진단. 설치 전에도 JSON을 남긴다."""
from __future__ import annotations
from importlib.metadata import PackageNotFoundError, distribution
import json
import re
import sys
from . import FLYGYM_COMMIT


def _version_tuple(value: str | None) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r'\d+', value or '')[:3])


def dependency_report() -> dict:
    versions = {}
    for name in ('numpy', 'aiohttp', 'flygym', 'mujoco'):
        try:
            versions[name] = distribution(name).version
        except PackageNotFoundError:
            versions[name] = None
    source = {'expectedCommit': FLYGYM_COMMIT, 'installedCommit': None,
              'status': 'unavailable'}
    if versions['flygym']:
        try:
            raw = distribution('flygym').read_text('direct_url.json')
            info = json.loads(raw) if raw else {}
            commit = info.get('vcs_info', {}).get('commit_id')
            source.update(installedCommit=commit,
                          status=('match' if commit == FLYGYM_COMMIT else
                                  'mismatch' if commit else 'not-recorded'))
        except (ValueError, OSError, TypeError):
            source['status'] = 'unreadable'
    issues = []
    if not (3, 12) <= sys.version_info[:2] < (3, 15):
        issues.append('Python 3.12–3.14 required')
    for name in ('numpy', 'aiohttp', 'flygym', 'mujoco'):
        if not versions[name]:
            issues.append(f'Missing package: {name}')
    if versions['numpy'] and _version_tuple(versions['numpy'])[:1] != (2,):
        issues.append('NumPy 2.x required')
    if versions['aiohttp'] and not (3, 12) <= _version_tuple(versions['aiohttp']) < (4,):
        issues.append('aiohttp >=3.12,<4 required')
    if versions['flygym'] and versions['flygym'] != '2.1.0':
        issues.append('FlyGym 2.1.0 required')
    if versions['mujoco'] and _version_tuple(versions['mujoco'])[:2] != (3, 9):
        issues.append('MuJoCo 3.9.x required')
    if source['status'] == 'mismatch':
        issues.append('Installed FlyGym commit does not match the project pin')
    return dict(versions=versions, ready=not issues, physicsExecuted=False,
                issues=issues, source=source,
                hint='Python 3.12–3.14 가상환경에서 python -m pip install -r requirements.txt 를 실행하세요.',
                expected=dict(flygym='2.1.0', mujoco='3.9.x', sourceCommit=FLYGYM_COMMIT))
