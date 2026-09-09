#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec .venv/bin/python run_c.py --backend exp_lif_mps "$@"
