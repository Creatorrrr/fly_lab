#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  printf '%s\n' '먼저: python3.12 -m venv .venv' '.venv/bin/python -m pip install -r requirements.txt'
  exit 2
fi
exec .venv/bin/python run.py "$@"
