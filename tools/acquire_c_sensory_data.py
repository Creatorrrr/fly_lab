#!/usr/bin/env python3
"""Acquire the hash-pinned, public DoOR and FAFB sensory source tables."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def acquire(directory,manifest):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    document=json.loads(Path(manifest).read_text(encoding='utf-8'))
    for row in document['sources']:
        target=(directory/row['path']).resolve()
        if not target.is_relative_to(directory) or not row['url'].startswith('https://'):
            raise ValueError('Invalid public asset destination/URL')
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']:
            continue
        if target.exists():raise ValueError('Existing asset hash mismatch: '+str(target))
        target.parent.mkdir(parents=True,exist_ok=True)
        part=target.with_name(target.name+'.partial')
        created=False
        try:
            digest=hashlib.sha256();size=0
            with urllib.request.urlopen(row['url'],timeout=60) as response,part.open('xb') as output:
                created=True
                while chunk:=response.read(1024*1024):
                    size+=len(chunk);digest.update(chunk);output.write(chunk)
            if size!=row['bytes'] or digest.hexdigest()!=row['sha256']:
                raise ValueError('Downloaded public asset hash/size mismatch: '+row['path'])
            part.rename(target)
        except Exception:
            # Only remove the exact temporary payload owned by this attempt.
            # An existing partial file is left for inspection.
            if created and part.exists():part.unlink()
            raise
        print('Verified '+row['path'],flush=True)
    (directory/'manifest.json').write_text(json.dumps(document,indent=2)+'\n',encoding='utf-8')
    return document


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'docs/C_SENSORY_ASSETS_20260910.json')
    a=p.parse_args();acquire(a.out,a.manifest)
