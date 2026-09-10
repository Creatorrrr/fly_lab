#!/usr/bin/env python3
"""Prefetch pinned FlyGym meshes concurrently, verifying upstream size/ETags."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import shutil
import tempfile
from flygym.utils import assets_lazy_loading as assets
from flygym.compose.fly.flybody import FLYBODY_FULLSIZE_MESH_DIR


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reuse',type=Path);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if not 1<=a.workers<=16:p.error('Use 1..16 download workers')
    cache=assets.get_cache_root();dest=cache/FLYBODY_FULLSIZE_MESH_DIR
    if dest.is_dir():print(dest);return
    prefix=assets.S3_ROOT_PREFIX+'/'+FLYBODY_FULLSIZE_MESH_DIR+'/'
    objects=assets._list_s3_prefix(prefix);stage=Path(tempfile.mkdtemp(prefix='flylab-prefetch-',suffix='.partial',dir=cache))
    jobs=[]
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        for obj in objects:
            relative=obj['key'][len(prefix):];target=(stage/relative).resolve()
            if not target.is_relative_to(stage.resolve()):raise ValueError('Invalid upstream asset path')
            previous=a.reuse/relative if a.reuse else None
            if previous and assets._is_up_to_date(previous,obj['size'],obj['etag']):
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(previous,target)
            else:jobs.append(executor.submit(assets._download_object,obj['key'],target,obj['size'],obj['etag']))
        for i,future in enumerate(as_completed(jobs)):
            future.result();print(f'Downloaded {i+1}/{len(jobs)}',flush=True)
    # The destination is a new cache directory; never replace existing assets.
    if not dest.exists():stage.rename(dest)
    print(dest)


if __name__=='__main__':main()
