#!/usr/bin/env python3
"""Separate C entry point. run.py and FLY_LAB_B.html remain compatible B."""
import sys
if not (3, 12) <= sys.version_info[:2] < (3, 15):
    raise SystemExit('FLY LAB C requires Python 3.12–3.14 for the pinned FlyGym body.')
if __name__ == '__main__':
    from flylab.c.server import main
    raise SystemExit(main())
