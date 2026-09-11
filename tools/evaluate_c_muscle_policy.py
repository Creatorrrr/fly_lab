"""Matched held-out-clip muscle comparisons, including optional LF contact."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from flylab.c.muscle_imitation import MuscleImitation
from flylab.c.integrity import write_json,file_hash


def main():
    from flygym_demo.muscle_imitation import MoCapDataset
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path);p.add_argument('--clip',default='0002')
    p.add_argument('--policy',type=Path);p.add_argument('--contact-platform',action='store_true')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    data=MoCapDataset(a.data) if a.data else MoCapDataset.default()
    modes=['passive','teacher']+(['untrained','trained'] if a.policy else [])
    report=dict(cases=[],clip=a.clip,policy_sha256=file_hash(a.policy) if a.policy else None,
        biological_validation=False,independent_animal_validation=False,policy_evaluation_device='cpu',
        metric_scope='matched full clip, final and per-frame joint RMSE; single seed')
    for mode in modes:
        rig=MuscleImitation(a.seed,dataset=data,clip=a.clip,contact_platform=a.contact_platform)
        rows=[]
        try:
            model=None
            if mode in ('trained','untrained'):
                from stable_baselines3 import PPO
                model=(PPO.load(a.policy,device='cpu') if mode=='trained' else
                    PPO('MlpPolicy',rig.env,device='cpu',seed=a.seed,n_steps=64,batch_size=64,policy_kwargs=dict(net_arch=[128,128])))
            while not rig.done:
                action=np.zeros(15,np.float32)
                if model is not None:action=np.clip(model.predict(rig.env._get_observation(),deterministic=True)[0],0,1)
                row=rig.step(teacher=True) if mode=='teacher' else rig.step(action)
                row['q_rmse_rad']=float(np.sqrt(np.mean((np.asarray(row['q'])-row['target_q'])**2)))
                rows.append(row)
            write_json(a.out/(mode+'.json'),rows)
            report['cases'].append(dict(mode=mode,identity=rig.identity,steps=len(rows),completed=rig.done,
                mean_reward=float(np.mean([r['reward'] for r in rows])),final_q_rmse_rad=rows[-1]['q_rmse_rad'],
                mean_q_rmse_rad=float(np.mean([r['q_rmse_rad'] for r in rows])),
                contact_samples=sum(bool(r['contacts']) for r in rows),
                max_contact_force=max((c['normal_force'] for r in rows for c in r['contacts']),default=0.),
                apparatus=rig.env.contact_apparatus))
            write_json(a.out/'report.json',report)
        finally:rig.close()
    print(json.dumps(report))


if __name__=='__main__':main()
