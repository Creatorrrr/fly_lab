#!/usr/bin/env python3
"""Create or validate a portable C source/data/verification ZIP without overwrites."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.evidence import package,verify_archive

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path);p.add_argument('--include',type=Path,action='append',default=[])
    p.add_argument('--verify',type=Path);a=p.parse_args()
    if a.verify:
        r=verify_archive(a.verify);r.pop('manifest');print(r);return
    if not a.out:p.error('--out or --verify required')
    print(package(Path(__file__).resolve().parents[1],a.out,a.include))
if __name__=='__main__':main()
