"""Create disjoint temporal clips; explicitly not an independent-animal dataset."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.integrity import file_hash


def main():
    from flygym_demo.muscle_imitation import MoCapDataset
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path)
    p.add_argument('--clip',default='0002');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();dataset=MoCapDataset(a.data) if a.data else MoCapDataset.default()
    clip=dataset.load(a.clip);n=clip.n_frames
    if n<200:raise ValueError('At least 200 frames required for a temporal split')
    windows=dict(train=(0,int(.6*n)),validation=(int(.8*n),n))
    a.out.mkdir(parents=True,exist_ok=False);files=[]
    for field in ('qpos','qvel','xipos','xivel'):
        value=getattr(clip,field)
        if value is None:continue
        (a.out/field).mkdir()
        for name,(start,end) in windows.items():
            dest=a.out/field/(name+'.npy');np.save(dest,value[start:end],allow_pickle=False)
            files.append(dict(path=str(dest.relative_to(a.out)),sha256=file_hash(dest)))
    report=dict(schema='flylab.imitation-split.v1',source_clip=a.clip,source_frames=n,windows=windows,
        source_files={field:file_hash(dataset.clip_dir/field/(a.clip+'.npy')) for field in ('qpos','qvel','xipos')},
        files=files,split='disjoint temporal windows with 20 percent gap; same animal and trial',
        independent_animal_validation=False)
    (a.out/'split.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':main()
