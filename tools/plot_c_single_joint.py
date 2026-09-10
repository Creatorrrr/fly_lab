#!/usr/bin/env python3
"""Export measured geometry and matched joint traces as standalone figures."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np


def plot(root,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=False)
    def read(path): return json.loads(path.read_text())
    geometry=read(root/'geometry/report.json')
    def trace(run,case): return read(root/run/case/'trace.json')
    def values(rows,key): return [row['physics'][key] for row in rows]
    def draw(ax,run,case,key,label,**kw):
        rows=trace(run,case);ax.plot([r['seconds']*1000 for r in rows],values(rows,key),label=label,**kw)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(15,8.3),layout='constrained')
    ax=axes[0,0];rows=geometry['samples'];q=[r['q_rad'] for r in rows]
    arms=np.array([r['moment_arm_mm'] for r in rows])
    ax.plot(q,arms[:,0],label='Flexor',color='#d76d39');ax.plot(q,arms[:,1],label='Extensor',color='#346cca')
    ax.axhline(0,color='#6c788c',lw=.7)
    crossing=geometry['zero_crossings'][0]['zero_angle_rad'];ax.axvline(crossing,color='#be3d42',ls='--',label=f'Reversal {crossing:.5f}')
    lo,hi=geometry['reference_clip']['tibia_angle_range_rad'];ax.axvspan(lo,hi,color='#52a687',alpha=.12,label='One motion clip range')
    ax.set(title='A  Original tendon routing',xlabel='Joint q (rad)',ylabel='d(tendon length)/dq (mm)');ax.legend(fontsize=8)
    ax=axes[0,1]
    for case,label,color in [('open_loop','Zero input','#657285'),('flexor','Fast flexor stimulus','#d76d39'),('extensor','FETi stimulus','#346cca')]:
        draw(ax,'banc_joint',case,'q_rad',label,color=color)
    ax.axvspan(10,50,color='#b6bcca',alpha=.15);ax.set(title='B  Motor spikes drive Hill muscles',xlabel='Model time (ms)',ylabel='Joint q (rad)');ax.legend(fontsize=8)
    ax=axes[0,2]
    for case,label,style in [('feedback','Feedback','-'),('open_loop','Feedback off','--'),('sensory_outgoing_muted','Sensory output muted',':')]:
        draw(ax,'banc_joint_gain72',case,'q_rad',label,ls=style)
    ax.set(title='C  Uncalibrated 72 mV hypothesis',xlabel='Model time (ms)',ylabel='Joint q (rad)');ax.legend(fontsize=8)
    ax=axes[1,0];rows=trace('banc_joint','feedback')
    profile=read(root/'banc_joint/feedback/profile.json');ids=rows[0]['signals']['ids']
    for motor,color in zip(profile['bindings']['motor'],('#d76d39','#346cca')):
        j=ids.index(motor['ids'][0]);ax.plot([r['seconds']*1000 for r in rows],[r['signals']['voltage_mV'][j] for r in rows],label=motor['cell_type'],color=color)
    ax.axhline(profile['neural_parameters']['threshold_mV'],color='#be3d42',ls='--',label='Spike threshold')
    ax.set(title='D  Default feedback: no motor spikes',xlabel='Model time (ms)',ylabel='Boundary voltage (mV)');ax.legend(fontsize=8)
    ax=axes[1,1]
    for case,index,label,color in [('flexor',0,'Fast flexor activation','#d76d39'),('extensor',1,'FETi activation','#346cca')]:
        rows=trace('banc_joint',case);ax.plot([r['seconds']*1000 for r in rows],[r['physics']['activation'][index] for r in rows],label=label,color=color)
    ax.axvspan(10,50,color='#b6bcca',alpha=.15);ax.set(title='E  Activation continues after stimulus',xlabel='Model time (ms)',ylabel='Muscle activation (0..1)');ax.legend(fontsize=8)
    ax=axes[1,2]
    for name,base,label,style in [('external_positive','feedback','Feedback: +torque','-'),('external_positive_open','open_loop','Off: +torque','--'),
                                 ('external_negative','feedback','Feedback: -torque','-'),('external_negative_open','open_loop','Off: -torque','--')]:
        rows,ref=trace('banc_joint_gain72',name),trace('banc_joint_gain72',base)
        count=min(len(rows),len(ref))
        ax.plot([r['seconds']*1000 for r in rows[:count]],np.array(values(rows[:count],'q_rad'))-np.array(values(ref[:count],'q_rad')),label=label,ls=style)
    ax.axvspan(50,80,color='#b6bcca',alpha=.15);ax.set(title='F  Perturbation minus matched baseline',xlabel='Model time (ms)',ylabel='Angle deviation (rad)');ax.legend(fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.15)
    fig.suptitle('Whole-BANC computation / single Fe-Ti muscle fixture\nFixed body, no contacts. Technical execution and physiological validity are separate.',fontsize=15)
    fig.savefig(out/'single_joint.png',dpi=170);fig.savefig(out/'single_joint.svg');plt.close(fig)
    from flylab.c.integrity import file_hash,write_json
    write_json(out/'figure_sources.json',dict(sources={str(p.relative_to(root)):file_hash(p) for p in
        [root/'geometry/report.json',root/'banc_joint/report.json',root/'banc_joint_gain72/report.json']},
        figures=['single_joint.png','single_joint.svg'],data_interpretation='Counterfactual error is measured relative to the same unperturbed profile'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();plot(a.root,a.out)
