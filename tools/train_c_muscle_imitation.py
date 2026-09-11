#!/usr/bin/env python3
"""Train and evaluate the official LF muscle imitation policy on CUDA if available."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from flygym_demo.muscle_imitation import make_imitation_env,ImitationConfig,MoCapDataset
    from flylab.c.muscle_imitation import make_muscle_env
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=2048)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--eval-steps',type=int,default=100)
    p.add_argument('--resume',type=Path);p.add_argument('--data',type=Path)
    p.add_argument('--train-clip',default='0002');p.add_argument('--eval-clip',default='0002')
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--rollout-steps',type=int,default=64);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--contact-platform',action='store_true')
    a=p.parse_args()
    if not 64<=a.steps<=30_000_000 or not 1<=a.eval_steps<=100000:p.error('Invalid training/evaluation length')
    if not np.isfinite(a.learning_rate) or not 0<a.learning_rate<=.01 or not 64<=a.rollout_steps<=8192 or not 2<=a.batch_size<=a.rollout_steps or a.rollout_steps%a.batch_size:
        p.error('Invalid optimizer/rollout configuration')
    a.out.mkdir(parents=True,exist_ok=False)
    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    dataset=MoCapDataset(a.data) if a.data else MoCapDataset.default()
    from flylab.c.integrity import digest
    clip_hashes={key:digest({field:None if getattr(dataset.load(key),field) is None else getattr(dataset.load(key),field).tolist()
        for field in ('qpos','qvel','xipos','xivel')}) for key in (a.train_clip,a.eval_clip)}
    env=Monitor(make_muscle_env(dataset=dataset,config=ImitationConfig(clip=a.train_clip),contact_platform=a.contact_platform),str(a.out/'monitor.csv'))
    evaluation=make_muscle_env(dataset=dataset,config=ImitationConfig(clip=a.eval_clip,test=True,init_noise_scale=0.),contact_platform=a.contact_platform)
    try:
        model=PPO.load(a.resume,env=env,device=device) if a.resume else PPO('MlpPolicy',env,device=device,seed=a.seed,
            n_steps=a.rollout_steps,batch_size=a.batch_size,n_epochs=4,policy_kwargs=dict(net_arch=[128,128]),learning_rate=a.learning_rate,verbose=0)
        model.learn(total_timesteps=a.steps,reset_num_timesteps=a.resume is None)
        model.save(a.out/'policy')
        obs,_=evaluation.reset(seed=a.seed);rows=[]
        for _ in range(a.eval_steps):
            action,_=model.predict(obs,deterministic=True)
            obs,reward,done,truncated,_=evaluation.step(np.clip(action,0.,1.));rows.append(dict(reward=float(reward),activation=action.tolist()))
            if done or truncated:break
        output=dict(device=str(model.device),requested_train_steps=a.steps,total_train_steps=model.num_timesteps,
            mean_evaluation_reward=float(np.mean([r['reward'] for r in rows])),evaluation_steps=len(rows),
            training_clip=a.train_clip,evaluation_clip=a.eval_clip,
            different_clip_content=clip_hashes[a.eval_clip]!=clip_hashes[a.train_clip],clip_hashes=clip_hashes,
            evaluation_completed=bool(done or truncated),evaluation_q_rmse_rad=float(np.sqrt(np.mean((evaluation.sim.mj_data.qpos[evaluation._tracked_qposadrs]-evaluation._clip.qpos[min(evaluation._mocap_idx,evaluation._clip.n_frames-1)])**2))),
            evaluation_metric_scope='q error at final evaluated frame; mean reward covers all evaluated steps',
            optimizer=dict(learning_rate=float(model.lr_schedule(1.)),rollout_steps=model.n_steps,batch_size=model.batch_size),
            data_split=json.loads((a.data/'split.json').read_text(encoding='utf-8')) if a.data and (a.data/'split.json').is_file() else None,
            contact_apparatus=evaluation.contact_apparatus,
            held_out_biological_validation=False,neural_connectome_control=False,biological_validation=False,
            scope='LF muscle imitation training execution; convergence and biological control are separate assessments')
        (a.out/'report.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
        (a.out/'evaluation.json').write_text(json.dumps(rows),encoding='utf-8');print(json.dumps(output))
    finally:env.close();evaluation.close()


if __name__=='__main__':main()
