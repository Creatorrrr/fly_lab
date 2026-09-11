#!/usr/bin/env python3
"""Acquire the pinned official FlyBody task code in a separate local checkout."""
import argparse
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.flybody_flight import SOURCE_COMMIT,validate_source


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not a.out.exists():
        subprocess.run(['git','clone','https://github.com/TuragaLab/flybody.git',str(a.out)],check=True)
        subprocess.run(['git','-C',str(a.out),'checkout','--detach',SOURCE_COMMIT],check=True)
    print(validate_source(a.out))
