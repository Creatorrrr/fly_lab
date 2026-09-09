#!/usr/bin/env python3
"""Run with Python 3.12–3.14; dependencies installed in an isolated venv."""
import sys
if not (3,12)<=sys.version_info[:2]<(3,15):
    raise SystemExit('FLY LAB B는 FlyGym 2.1.0 때문에 Python 3.12–3.14가 필요합니다.')
try:
    from flylab.server import main
except ModuleNotFoundError as e:
    raise SystemExit(f'필수 패키지가 없습니다: {e.name}\npython -m pip install -r requirements.txt') from e
if __name__=='__main__':raise SystemExit(main())
