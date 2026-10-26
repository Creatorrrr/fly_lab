"""Rate-dynamics-feasible muscle labels; teacher diagnostic, never runtime input."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import lsq_linear, minimize
from scipy.signal import lfilter
from scipy.sparse.linalg import LinearOperator

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flylab.c.storage import StateStore
from flylab.c.walking_trace import capture_control
from tools.analyze_c_banc_biological_gait import analyze, load_model, sha256, write_json
from tools.analyze_c_banc_observation import new_body
from tools.verify_c_banc_observation import exact


def rate_filter(x,tau,*,transpose=False):
    alpha=1-math.exp(-.005/tau)
    if transpose:
        return lfilter([alpha],[1,alpha-1],x[::-1],axis=0)[::-1]
    return lfilter([alpha],[1,alpha-1],x,axis=0)


def continuous_neural_filter(x,*,transpose=False):
    """Exact boundary samples of v -> u(10ms) -> r(20ms) continuous ODE."""
    au,ar=math.exp(-.005/.01),math.exp(-.005/.02)
    coupling=.01/(.01-.02)*(au-ar)
    direct=1-ar-coupling
    b=[direct,coupling*(1-au)-au*direct]
    a=[1,-au-ar,au*ar]
    if transpose:
        return lfilter(b,a,x[::-1],axis=0)[::-1]
    return lfilter(b,a,x,axis=0)


def inverse_operator(moment,frames,deadline,continuous_drive=False):
    groups=moment.shape[1]
    # A tiny energy penalty selects among mechanically equivalent coactivations.
    ridge=1e-5
    def check():
        if time.perf_counter()>deadline:
            raise TimeoutError('Dynamic inverse solver wall limit')
    def forward(u):
        check();u=np.asarray(u).reshape(frames,groups)
        r=continuous_neural_filter(u) if continuous_drive else rate_filter(u,.02)
        a=rate_filter(r,.01)
        q=.02*a@moment.T
        return np.r_[q.ravel(),ridge*u.ravel()]
    def transpose(v):
        check();v=np.asarray(v).ravel()
        q=v[:frames*42].reshape(frames,42)
        a=.02*q@moment
        r=rate_filter(a,.01,transpose=True)
        u=continuous_neural_filter(r,transpose=True) if continuous_drive else rate_filter(r,.02,transpose=True)
        return u.ravel()+ridge*v[frames*42:]
    return LinearOperator((frames*(42+groups),frames*groups),matvec=forward,rmatvec=transpose,dtype=np.float64)


def refine_continuous_solution(initial,target,moment,rest,deadline):
    """Same convex inverse problem, explicit separable CUDA float64 solve."""
    import torch
    from scipy.linalg import toeplitz
    if not torch.cuda.is_available():raise ValueError('CUDA inverse refinement required')
    frames=len(target)
    impulse=np.zeros(frames);impulse[0]=1
    response=rate_filter(continuous_neural_filter(impulse),.01)
    h=toeplitz(response,np.r_[response[0],np.zeros(frames-1)])
    h_t=torch.as_tensor(h,dtype=torch.float64,device='cuda')
    m_t=torch.as_tensor(moment.T,dtype=torch.float64,device='cuda')
    y=torch.as_tensor(target-rest,dtype=torch.float64,device='cuda')
    scale=np.sqrt((32*np.sum(h*h,axis=0)[:,None]*np.sum(moment*moment,axis=0)[None,:]+8e-6)/target.size)
    best={'value':float('inf'),'x':initial.copy()};history=[]
    def objective(value):
        if time.perf_counter()>deadline:raise TimeoutError('Continuous inverse refinement wall limit')
        v=torch.tensor(value.reshape(initial.shape)/scale,dtype=torch.float64,device='cuda',requires_grad=True)
        q=4*h_t@v@m_t
        loss=((q-y).square().sum()+(.002*v).square().sum())/target.size
        loss.backward()
        number=float(loss.detach().cpu())
        gradient=v.grad.detach().cpu().numpy()/scale
        if number<best['value']:best.update(value=number,x=v.detach().cpu().numpy()*200)
        history.append(number)
        if len(history)%500==0:print(json.dumps({'inverse_evaluation':len(history),'objective':number}),flush=True)
        return number,gradient.ravel()
    try:
        fit=minimize(objective,(initial/200*scale).ravel(),jac=True,method='L-BFGS-B',
            bounds=list(zip(np.zeros(initial.size),(.995*scale).ravel())),
            options={'maxiter':5000,'maxls':30,'maxcor':20,'ftol':1e-14,'gtol':1e-10})
        result=SimpleNamespace(x=best['x'].ravel(),success=fit.success,message=str(fit.message),
            optimality=float(np.max(abs(fit.jac))),nit=fit.nit)
    except TimeoutError as exc:
        result=SimpleNamespace(x=best['x'].ravel(),success=False,message=str(exc),optimality=None,nit=-1)
    # Independent CPU objective verifies the exact filter and energy convention.
    q=rest+.02*rate_filter(continuous_neural_filter(best['x']),.01)@moment.T
    result.cost=.5*(np.sum((q-target)**2)+np.sum((1e-5*best['x'])**2))
    if abs(2*result.cost/target.size-best['value'])>1e-10:
        raise ValueError('Explicit GPU inverse and independent CPU objective disagree')
    result.device=torch.cuda.get_device_name(0)
    result.evaluations=len(history)
    return result


def run(source,projected,out,evaluate_saved=None,continuous_drive=False,refine_saved=None):
    out.mkdir(parents=True,exist_ok=False)
    spec={
        'experiment':'M3-E11-dynamic-targets',
        'hypothesis':'Projected teacher commands remain useful when motor rate generation and muscle activation both have finite response times',
        'competing_explanation':'E06 feasible muscle inputs demand changes faster than the declared neural rate equation can generate',
        'intervention':'Inverse linear positive cascade: neural target activation u[0,199], cap200, motor tau20ms, muscle tau10ms, original signed moment*.02 and rest. Both rate and muscle states start zero; all commands generated from their own previous states. Constant .2 tendon preloads/no adhesion.',
        'equation':'dr/dt=(u-r)/.02 with u the bounded tanh target, dA/dt=(r-A)/.01; exact held-input neural rate sampling and control-boundary muscle activation. This only constrains the range/speed of neural output; fitting a recurrent circuit that produces u remains separate.',
        'fit':'Bounded least squares across the existing2.2s label trajectory, L2 command loss plus1e-5 input-energy coefficient; no independent validation claim from inverse labels',
        'screen':'Projected-command RMS<=.05rad on [.5,2.2), actual2.2s replay net>=10mm/upright>=.9/no fault or nonfoot contact; F1-F8 remain unchanged',
        'limits':'CPU inverse solver<=120wall sec, CPU physics<=2.2model sec; no BANC neural execution',
        'source_sha256':sha256(projected/'signals.npz'),'tool_sha256':sha256(__file__),
        'saved_feasible_candidate':str(evaluate_saved) if evaluate_saved else None,
        'saved_candidate_sha256':sha256(evaluate_saved/'signals.npz') if evaluate_saved else None,
        'saved_candidate_contract':'If explicitly supplied, evaluate its unchanged finite bounded solution even when the optimizer hit max_iter; do not claim optimality or erase the original optimization failure. The same physical screen applies.',
    }
    if continuous_drive:
        spec.update(experiment='M3-E14-continuous-current-targets',
            hypothesis='A continuously varying motor tanh target can still express the projected body gait and avoid the E11 discontinuous recurrent-current demand',
            competing_explanation='The additional continuity/response restriction prevents the existing12Hz teacher commands from being represented',
            intervention='Bounded v[0,199] is held5ms; du/dt=(v-u)/.01, dr/dt=(u-r)/.02 are integrated as a coupled continuous system exactly; muscle activation samples interval-end r with tau10ms. Original moment*.02/rest, all states initially zero. Desired recurrent current from u,z is now continuous across5ms boundaries.',
            feasibility_contract='A finite bounded solution may be physically evaluated even if the optimizer hits its iteration limit; report solver status/optimality separately and do not claim an optimum. Same .05rad/net10/upright.9/no-contact teacher screen; all F1-F8 unchanged.')
    if refine_saved:
        if not continuous_drive or evaluate_saved:raise ValueError('Refinement requires continuous drive and separate source')
        spec.update(experiment='M3-E14b-continuous-target-refinement',
            numerical_revision='Same convex objective/bounds/model/reference/screen as E14. Its TRF optimality.049 and incomplete iteration status justify an explicit separable Toeplitz float64 CUDA L-BFGS-B solve with diagonal training-objective preconditioning, starting at the unchanged saved E14 solution. No gain/cadence/bound search.',
            refinement_source_sha256=sha256(refine_saved/'signals.npz'),
            limits='One CUDA inverse fit<=285wall sec within300wall limit, CPU physical replay<=2.2model sec')
    write_json(out/'spec.json',spec);(out/'runner.py').write_bytes(Path(__file__).read_bytes())
    d=np.load(projected/'signals.npz');target=d['command'];moment=d['moment'];rest=d['rest']
    frames,groups=len(target),moment.shape[1]
    began=time.perf_counter()
    if refine_saved:
        candidate=np.load(refine_saved/'signals.npz')
        for name,value in (('target',target),('moment',moment),('rest',rest)):
            exact(candidate[name],value,'E14b source '+name)
        if float(candidate['continuous_drive_tau_s'])!=.01:raise ValueError('Wrong continuous target dynamics')
        result=refine_continuous_solution(candidate['input_activation'],target,moment,rest,began+275)
    elif evaluate_saved:
        candidate=np.load(evaluate_saved/'signals.npz')
        previous=json.loads((evaluate_saved/'fit-result.json').read_text())
        for name,value in (('target',target),('rest',rest),('moment',moment)):
            exact(candidate[name],value,'E11 saved feasible '+name)
        u=candidate['input_activation']
        if u.shape!=(frames,groups) or not np.isfinite(u).all() or np.any((u<0)|(u>199)):
            raise ValueError('Saved neural target is not feasible')
        result=SimpleNamespace(x=u.ravel(),success=previous['solver_success'],
            message=previous['solver_message'],optimality=previous['optimality'],
            cost=previous['cost'],nit=previous['iterations'])
    else:
        operator=inverse_operator(moment,frames,began+120,continuous_drive)
        rhs=np.r_[(target-rest).ravel(),np.zeros(frames*groups)]
        result=lsq_linear(operator,rhs,bounds=(0,199),tol=1e-6,lsmr_tol='auto',max_iter=200,lsmr_maxiter=250)
    u=result.x.reshape(frames,groups)
    r=continuous_neural_filter(u) if continuous_drive else rate_filter(u,.02)
    activation=rate_filter(r,.01)
    command=rest+.02*activation@moment.T
    if evaluate_saved:
        for name,value in (('neural_rate',r),('muscle_activation',activation),('command',command)):
            exact(candidate[name],value,'E11 saved dynamics '+name)
    extra={'continuous_drive_tau_s':np.array(.01),'target_activation':rate_filter(u,.01)} if continuous_drive else {}
    np.savez_compressed(out/'signals.npz',input_activation=u,neural_rate=r,
        muscle_activation=activation,command=command,target=target,rest=rest,moment=moment,
        time_s=np.arange(1,frames+1)*.005,**extra)
    fit={'solver_success':bool(result.success),'solver_message':result.message,
        'optimality':float(result.optimality) if result.optimality is not None else None,'cost':float(result.cost),'iterations':int(result.nit),
        'wall_s':time.perf_counter()-began,
        'rate_range':np.percentile(r,[0,50,99,100]).tolist(),
        'target_activation_range':np.percentile(u,[0,50,99,100]).tolist(),
        'command_rms_rad':float(np.sqrt(np.mean((command[100:]-target[100:])**2)))}
    if refine_saved:fit.update(device=result.device,evaluations=result.evaluations,solver='float64 CUDA L-BFGS-B')
    write_json(out/'fit-result.json',fit)
    if not result.success and evaluate_saved is None and not continuous_drive:
        write_json(out/'result.json',{'run_status':'solver_failed','validity':'incomplete_optimization',
            'hypothesis_verdict':'inconclusive','fit':fit,'adoption':'not_applicable','milestone_outcome':'pending'})
        print(json.dumps(fit));return
    saved=StateStore.load(source/'start');body=new_body(saved);began=time.perf_counter()
    try:
        exact(saved['body'],body.snapshot(),'E11 initial body')
        tendons={name:.2 for name in body.tendon_control.names}
        observer=SimpleNamespace(body=body,control_tick=0,rate=np.zeros(391),tendon_inputs=tendons,fault=None)
        rows=[capture_control(observer)]
        with (out/'trace.jsonl').open('w') as stream:
            stream.write(json.dumps(rows[0])+'\n')
            for tick,q in enumerate(command,1):
                body.step_joint_targets(q,np.zeros(6,bool),tendon_inputs=tendons)
                observer.control_tick=tick;rows.append(capture_control(observer))
                stream.write(json.dumps(rows[-1])+'\n')
                if body.fault or body.nonfoot_contact():break
        physical={'run_status':'completed','model_s':observer.control_tick*.005,
            'native_time_s':float(body.d.time),'wall_s':time.perf_counter()-began,
            'fault':body.fault,'nonfoot_contact':bool(body.nonfoot_contact()),
            'body_hash':body.model_hash,'initial_body_bit_exact':True,'BANC_neural_steps':0}
        write_json(out/'physical-result.json',physical)
        StateStore.save(out/'final-body',body.snapshot())
        write_json(out/'trace.json',{'schema':'flylab.m3-dynamic-labels.v1',
            'controller':'bounded synthetic neural-activation teacher; not BANC outputs',
            'body_hash':body.model_hash,'model_hash':'no-BANC-neural-execution',
            'adapter_hash':sha256(out/'signals.npz'),'graph_hash':saved['graph_hash'],
            'parameters':spec,'samples':rows})
    finally:body.close()
    quality,_,_=analyze(load_model(out/'trace.json'))
    passed=fit['command_rms_rad']<=.05 and physical['model_s']>=2.2 and not physical['fault'] and not physical['nonfoot_contact'] and quality['physical']['minimum_upright']>=.9 and quality['physical']['net_mm']>=10
    report={'experiment':spec['experiment'],'run_status':'completed','validity':'valid_teacher_diagnostic',
        'screen_passed':passed,'hypothesis_verdict':'supported' if passed else 'not_supported',
        'fit':fit,'physical':physical,'quality':quality,'adoption':'not_applicable','milestone_outcome':'pending'}
    write_json(out/'result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='quality'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','projected','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--evaluate-saved',type=Path)
    p.add_argument('--continuous-drive',action='store_true')
    p.add_argument('--refine-saved',type=Path)
    a=p.parse_args();run(a.source,a.projected,a.out,a.evaluate_saved,a.continuous_drive,a.refine_saved)
