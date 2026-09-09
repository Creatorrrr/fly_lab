#!/usr/bin/env python3
"""Fetch pinned public BANC or flybody inputs into a new acquisition directory."""
import argparse
from pathlib import Path
import subprocess
import sys
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.integrity import read_json,write_json,file_hash

def acquire(kind,destination):
    root=Path(__file__).resolve().parents[1];destination=Path(destination)
    if destination.exists():raise ValueError('Acquisition destination already exists')
    if kind=='flybody':
        ref=read_json(root/'data/releases/flybody-source.json')
        subprocess.run(['git','clone','--no-checkout',ref['repository'],str(destination)],check=True)
        subprocess.run(['git','-C',str(destination),'checkout','--detach',ref['commit']],check=True)
        return
    reference=read_json(root/'data/releases/banc888-v2-sources.json');destination.mkdir(parents=True)
    for item in reference['files']:
        path=destination/item['name']
        with urllib.request.urlopen(item['url'],timeout=60) as response,path.open('xb') as output:
            while block:=response.read(1024*1024):output.write(block)
        if path.stat().st_size!=item['size'] or file_hash(path)!=item['sha256']:raise ValueError('Pinned data checksum mismatch')
    write_json(destination/'download_manifest.json',reference)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=('banc','flybody'));p.add_argument('--destination',required=True)
    a=p.parse_args();acquire(a.kind,a.destination)
