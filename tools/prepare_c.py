#!/usr/bin/env python3
"""Build a new immutable acquisition; never overwrite bundled bindings."""
import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.build_c_graph import build, download, PRODUCTS
from tools.build_c_bindings import build as build_bindings
from tools.build_c_odor_profile import build as build_odor_profile
from tools.build_c_sensor_profile import build as build_sensor_profile
from flylab.c.data_identity import verify_snapshot, reference_for
from flylab.c.integrity import write_json


def prepare(destination, raw_source=None, reference=None):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    try:
        raw=destination/'raw'
        if raw_source:
            raw.mkdir()
            for name in [p+'.csv.gz' for p in PRODUCTS]+['download_manifest.json']:
                shutil.copy2(Path(raw_source)/name,raw/name)
        else:
            download(raw)
        graph=build(raw,destination/'bundle','mask_zero')
        validation=verify_snapshot(graph,reference)
        write_json(destination/'snapshot_validation.json',validation)
        if validation['status']!='PASS':
            raise ValueError('Snapshot identity not accepted: '+str(validation))
        bindings=build_bindings(graph)
        write_json(destination/'bindings.json',bindings)
        write_json(destination/'bindings-bilateral-geosmin-v1.json',build_odor_profile(graph,bindings))
        bilateral=build_odor_profile(graph,bindings,2)
        write_json(destination/'bindings-bilateral-geosmin-v2.json',bilateral)
        write_json(destination/'bindings-visual-head-contact-research-v1.json',build_sensor_profile(graph,bilateral))
        write_json(destination/'reference.json',reference_for(graph))
        result=dict(status='COMPLETE',graph=str((destination/'bundle').resolve()),
                    bindings=str((destination/'bindings.json').resolve()),graph_hash=graph.hash,
                    snapshot=validation,physicalExecuted=False)
        write_json(destination/'preparation.json',result)
        return result
    except Exception as exc:
        write_json(destination/'preparation.json',dict(status='FAILED',reason=str(exc),physicalExecuted=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--raw-source',type=Path)
    p.add_argument('--reference',type=Path)
    a=p.parse_args()
    print(prepare(a.destination,a.raw_source,a.reference))
