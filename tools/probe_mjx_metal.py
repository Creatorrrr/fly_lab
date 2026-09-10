#!/usr/bin/env python3
"""Probe the unchanged fly model on JAX CUDA/Metal; no CPU substitution.

The historical filename is retained. This is a diagnostic, not the C body backend.
"""
from pathlib import Path
import argparse
import importlib.metadata as metadata
import json
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('auto','cuda','metal'),default='auto')
    args=p.parse_args()
    if args.out.exists():p.error('Output already exists')
    versions={}
    for name in ('jax','jaxlib','jax-metal','mujoco','mujoco-mjx'):
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:versions[name]=None
    r={'status':'RUNNING','stage':'imports','model_modified':False,'physical_executed':False,
       'requested_device':args.device,'versions':versions}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    def save():args.out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    save()
    try:
        import numpy as np
        import mujoco
        import jax
        from mujoco import mjx
        device=None;r['device_errors']={}
        for backend in (('cuda','metal') if args.device=='auto' else (args.device,)):
            try:
                devices=jax.devices(backend)
                if not devices:raise RuntimeError('No devices')
                device=devices[0];r['selected_device']=backend;r['devices']=[str(d) for d in devices]
                break
            except Exception as exc:r['device_errors'][backend]=str(exc)
        if device is None:raise RuntimeError('No requested JAX GPU available: '+str(r['device_errors']))
        r['stage']=r['selected_device']+'_jit_smoke';save()
        with jax.default_device(device):
            x=jax.jit(lambda x:x*x+1)(np.arange(8,dtype=np.float32))
            r['jit_values']=np.asarray(x.block_until_ready()).tolist()
            r['stage']='unchanged_model_conversion';save()
            model=mujoco.MjModel.from_binary_path(str(args.model))
            r['physics_options']={'noslip_iterations':model.opt.noslip_iterations,
                                  'integrator':int(model.opt.integrator),'solver':int(model.opt.solver)}
            gpu_model=mjx.put_model(model,impl='jax',device=device)
            data=mujoco.MjData(model)
            for name,value in np.load(args.state,allow_pickle=False).items():
                if name=='time':data.time=float(value)
                else:getattr(data,name)[:]=value
            mujoco.mj_forward(model,data)
            gpu_data=mjx.put_data(model,data,impl='jax',device=device)
            r['stage']='compile_full_physics_step';save()
            step=jax.jit(mjx.step)
            begun=time.perf_counter();gpu_data=step(gpu_model,gpu_data)
            gpu_data.qpos.block_until_ready()
            r['cold_step_seconds']=time.perf_counter()-begun
            r['physical_executed']=True;r['stage']='warm_steps';save()
            begun=time.perf_counter()
            for _ in range(50):gpu_data=step(gpu_model,gpu_data)
            qpos=np.asarray(gpu_data.qpos.block_until_ready())
            r.update(warm_50_steps_seconds=time.perf_counter()-begun,qpos_finite=bool(np.isfinite(qpos).all()),status='EXECUTED_REQUIRES_PARITY')
    except Exception as e:
        r.update(status='BLOCKED',error=str(e),error_type=type(e).__name__,traceback=traceback.format_exc())
    save();print(json.dumps(r,indent=2))
    return 0 if r['physical_executed'] else 2


if __name__=='__main__':raise SystemExit(main())
