"""Build named-odorant and optional column-resolved experimental neural inputs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.graph import GraphStore
from flylab.sensory_evidence import chemical_profile,column_retinotopy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('graph','base','assets','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--odorants',nargs='+',default=['ethyl acetate','1-octen-3-ol'])
    p.add_argument('--columns',action='store_true');p.add_argument('--registration',type=Path)
    a=p.parse_args();g=GraphStore.load(a.graph)
    result=chemical_profile(g,json.loads(a.base.read_text(encoding='utf-8')),a.assets,a.odorants)
    if a.columns:result=column_retinotopy(g,result,a.assets,registration=json.loads(a.registration.read_text(encoding='utf-8')) if a.registration else None)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(path=str(a.out),ports=len(result['sensory']),receptors=len(result['sensor_model']['chemical_odor']['receptors']),mapping_status=result['mapping_status'])))


if __name__=='__main__':main()
