"""One reviewed E128 conditional joint fit; original invocation belongs to parent.

The source review is a required independent gate. This runner never controls
the BANC graph, FlyBody, or a physical trial.
"""

# ruff: noqa: BLE001  # all broad catches retain primary and secondary receipts

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.banc_m3_joint_adaptive_recruitment import (
    FIELDS,
    CudaJointAdaptiveEngine,
    activation_cotangent_product,
    activation_parameter_map,
    initial_activation_bank,
)
from tools.banc_m3_torch_foot_geometry import TorchFootGeometry

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FREEZE_SHA = '8a2fb800fba151736fb9ef20ab66b1f132ad705ae6e6f510b5cf222153a1d343'
SOURCE_FILES = ('tools/banc_m3_joint_adaptive_recruitment.py',
                'tools/fit_c_banc_m3_joint_adaptive_recruitment.py',
                'tests/test_c_banc_joint_adaptive_recruitment.py')
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
INNER = np.array([.018,.04,.009,.025])
ORIGINAL = np.array([.02,.05,.01,.03])
OPTIMIZER_CAP = (1-2e-4)**2
SELECTOR_CAP = (1-1e-4)**2
CONSTRAINT_ROWS = 4761
LEG_LIMIT_S = 30.
LEG_RESERVE_S = 3.
FIT_LIMIT_S = 185.
INTERNAL_LIMIT_S = 215.
PROCESS_RESERVE_S = 5.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2,
        ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n',
        encoding='utf-8')


def checked_relative(relative, expected):
    name = Path(relative)
    if (name.is_absolute() or '..' in name.parts or
            not isinstance(expected, str) or len(expected) != 64):
        raise ValueError('Invalid frozen path/hash: '+str(relative))
    path = (ROOT/name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or sha256(path) != expected:
        raise ValueError('Frozen SHA mismatch: '+relative)
    return path


def reviewed_inputs(freeze_path, freeze_sha, review_path, review_sha):
    if freeze_sha != EXPECTED_FREEZE_SHA or sha256(freeze_path) != freeze_sha:
        raise ValueError('E128 freeze SHA mismatch')
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    if (freeze.get('schema') != 'flylab.m3.e128.joint-adaptive-fit-freeze.v1' or
            freeze.get('design_sha256') != sha256(ROOT/freeze['design_path']) or
            freeze.get('fixture_path') not in freeze.get('input_sha256', {}) or
            [x['name'] for x in freeze.get('contexts', [])] !=
                ['E110','E113','E121'] or
            any(freeze.get(k) != v for k,v in {
                'context_count':3,'motor_count':391,'source_count':69,
                'union_edge_count':2444,'fit_edge_count':1572,
                'muscle_count':96,'controls':2000,'stage_count':160000,
                'constraints_per_leg':4761,'leg_unique_evaluation_limit':100,
                'total_unique_evaluation_limit':600,
                'full_response_batch_limit':4,'solver':'SLSQP',
                'solver_maxiter':100,'solver_ftol':1e-9,
                'fit_wall_limit_s':185.,'internal_wall_limit_s':215.,
                'external_wall_limit_s':230.,'supervisor_limit_s':235.,
                'shared_process_limit_s':1530.,
                'prior_conservative_process_s':1212.,
                'allocated_conservative_end_s':1497.,
            }.items())):
        raise ValueError('E128 fixed design contract changed')
    for name, count in (('input_sha256',32), ('reused_source_sha256',42),
                        ('preservation_sha256',1460)):
        values = freeze.get(name)
        if not isinstance(values, dict) or len(values) != count:
            raise ValueError('E128 frozen map cardinality changed: '+name)
        for relative, expected in values.items():
            checked_relative(relative, expected)
    for context in freeze['contexts']:
        for field in ('current','source','endpoints'):
            if context[field] not in freeze['input_sha256']:
                raise ValueError('Unfrozen E128 context path')
    if sha256(review_path) != review_sha:
        raise ValueError('E128 source review SHA mismatch')
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if (review.get('accepted_for_single_original_joint_adaptive_fit') is not True or
            review.get('blocking_defects') != [] or
            review.get('freeze_sha256') != freeze_sha or
            not isinstance(review.get('source_sha256'), dict) or
            set(review['source_sha256']) != set(SOURCE_FILES)):
        raise ValueError('Accepted exact E128 three-source review required')
    for relative, digest in review['source_sha256'].items():
        checked_relative(relative, digest)
    return freeze, review


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key:z[key].copy() for key in z.files}


def frozen_problem(freeze):
    f = load_npz(ROOT/freeze['fixture_path'])
    if (f['context_names'].tolist() != ['E110','E113','E121'] or
            f['fit_time_mask'].sum(axis=1).tolist() != [1823,1823,1827] or
            f['own_mask'].sum(axis=1).tolist() != [1828,1823,1827] or
            f['startup_mask'].nonzero()[0].tolist() != list(range(1,100)) or
            f['leg_motor_mask'].sum(axis=1).tolist() != [69,63,63,70,63,63] or
            f['leg_edge_mask'].sum(axis=1).tolist() != [273,281,240,238,255,285] or
            f['leg_muscle_mask'].sum(axis=1).tolist() != [17,15,16,17,15,16]):
        raise ValueError('E128 frozen fixture masks changed')
    current = []
    source = []
    for item in freeze['contexts']:
        i = np.load(ROOT/item['current'], mmap_mode='r', allow_pickle=False)
        s = np.load(ROOT/item['source'], mmap_mode='r', allow_pickle=False)
        if (i.shape != (160000,391) or i.dtype != np.float32 or
                s.shape != (160000,69) or s.dtype != np.float32):
            raise ValueError('E128 native stage/source shape changed')
        current.append(i)
        source.append(s)
    f['stage_current'] = np.stack(current)
    f['source_rate'] = np.stack(source)
    return f


def synthetic_tiny_setup_problem():
    """Explicit one-neuron setup with no bytes from a recorded context."""
    stage=np.arange(80,dtype=np.float64)
    current=(9.+.1*np.sin(stage*.17)).astype(np.float32)[None,:,None]
    source=(.25+.05*np.cos(stage*.13)).astype(np.float32)[None,:,None]
    moment=np.zeros((42,1),np.float64)
    moment[0,0]=.1
    return {'stage_current':current,'source_rate':source,
        'time_s':np.array([0.,.005]),'initial_rate':np.zeros((1,1)),
        'initial_adaptation':np.zeros((1,1)),
        'initial_activation':np.zeros((1,1)),
        'tau':np.array([.02]),'a':np.array([1.]),
        'threshold':np.array([7.5]),'cap':np.array([200.]),
        'pool':np.array([[1.]]),'moment':moment,'rest':np.zeros(42),
        'ltm':np.array([0]),'membership':np.array([0]),
        'motor_ids':np.array([1001]),'source_ids':np.array([2001]),
        'union_edge_ids':np.array([3001]),'union_rows':np.array([0]),
        'union_cols':np.array([0]),'union_values':np.array([[1.]],np.float32),
        'fit_union_positions':np.array([0]),'edge_ids':np.array([3001]),
        'edge_rows':np.array([0]),'edge_cols':np.array([0]),
        'base_fit_weights':np.array([1.])}


def construct_engine(mapping,ledger,out,label,*subset):
    try:
        return CudaJointAdaptiveEngine(mapping,*subset)
    except Exception as error:
        event={'label':label,'type':type(error).__name__,
            'message':str(error),
            'fixed_upload_progress':getattr(error,'fixed_upload_progress',None),
            'fixed_upload_by_key':getattr(error,'fixed_upload_by_key',None)}
        ledger.setdefault('constructor_failures',[]).append(event)
        try:
            write_json(out/'advance-ledger.json',ledger)
        except Exception as secondary:
            error.add_note('Secondary constructor ledger save: '+repr(secondary))
        raise


def run_tiny_setup(out,ledger):
    """Own raw and progress even if synthetic compile or bank fails."""
    tiny=synthetic_tiny_setup_problem()
    setup={'status':'requested','synthetic_only':True,
        'requested_context_ticks':20,'requested_conditional_s':.005,
        'requested_edge_stage_ticks':80,
        'response_progress':None,'fixed_upload_progress':None,
        'raw_saved':False}
    ledger['cuda_tiny_setup']=setup
    write_json(out/'advance-ledger.json',ledger)
    engine=None
    try:
        engine=construct_engine(tiny,ledger,out,'tiny_synthetic')
        _,raw,timing=engine.solve(np.zeros(1),np.zeros(1),derivative=True)
        setup['response_progress']=timing['progress']
        setup['fixed_upload_progress']=engine.metadata()['fixed_upload']
        np.savez(out/'cuda-tiny-setup-raw.npz',
            synthetic_stage_current=tiny['stage_current'],
            synthetic_source_rate=tiny['source_rate'],**raw)
        setup.update(status='raw_saved',raw_saved=True,
            raw_sha256=sha256(out/'cuda-tiny-setup-raw.npz'),
            response_progress=timing['progress'],
            fixed_upload_progress=engine.metadata()['fixed_upload'],
            engine_metadata=engine.metadata(),
            finite=all(np.isfinite(raw[k]).all() for k in FIELDS))
        write_json(out/'advance-ledger.json',ledger)
        if not setup['finite']:
            raise ValueError('E128 synthetic tiny setup nonfinite')
        setup['status']='completed'
        return setup
    except Exception as error:
        setup['status']='failed'
        setup['response_progress']=getattr(error,'response_progress',
                                           setup['response_progress'])
        setup['fixed_upload_progress']=getattr(error,'fixed_upload_progress',
                                               setup['fixed_upload_progress'])
        if engine is not None:
            setup['engine_metadata']=engine.metadata()
        partial=getattr(error,'partial_response',None)
        if partial:
            try:
                np.savez(out/'cuda-tiny-setup-partial-raw.npz',
                    synthetic_stage_current=tiny['stage_current'],
                    synthetic_source_rate=tiny['source_rate'],**partial)
                setup['partial_raw_saved']=True
            except Exception as secondary:
                error.add_note('Secondary tiny setup partial save: '+repr(secondary))
        try:
            write_json(out/'advance-ledger.json',ledger)
        except Exception as secondary:
            error.add_note('Secondary tiny setup ledger save: '+repr(secondary))
        raise


def one_cpu_thread():
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    limits = threadpool_limits(limits=1)
    info = threadpool_info()
    if (torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1 or
            any(p['num_threads'] != 1 for p in info)):
        limits.restore_original_limits()
        raise RuntimeError('E128 one CPU thread contract failed')
    return limits, {'torch_intraop':torch.get_num_threads(),
        'torch_interop':torch.get_num_interop_threads(), 'threadpools':info,
        'environment':{k:os.environ.get(k) for k in
            ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS')}}


def constraint_values(losses, loss_jac, q, ltm, jq, jltm, t,
                      *, optimizer_cap=OPTIMIZER_CAP):
    """Three-context 4761 signed and RMS inequalities, exact Jacobian."""
    losses = np.asarray(losses, np.float64)
    loss_jac = np.asarray(loss_jac, np.float64)
    q = np.asarray(q, np.float64)
    ltm = np.asarray(ltm, np.float64)
    jq = np.asarray(jq, np.float64)
    jltm = np.asarray(jltm, np.float64)
    if (losses.shape != (3,) or q.shape != (3,99,7) or
            ltm.shape != (3,99) or jq.shape[:3] != q.shape or
            jltm.shape[:2] != ltm.shape or
            loss_jac.shape != (3,jq.shape[-1]) or
            jltm.shape[-1] != jq.shape[-1]):
        raise ValueError('Exact three-context initial99 constraint dimensions required')
    p = jq.shape[-1]
    root = math.sqrt(optimizer_cap)
    q_ratio = np.mean(q*q,axis=(1,2))/INNER[0]**2
    ltm_ratio = np.mean(ltm*ltm,axis=1)/INNER[2]**2
    value = np.r_[t-losses,
        (root-q/INNER[1]).ravel(), (root+q/INNER[1]).ravel(),
        (root-ltm/INNER[3]).ravel(), (root+ltm/INNER[3]).ravel(),
        optimizer_cap-q_ratio,optimizer_cap-ltm_ratio]
    jac = np.zeros((CONSTRAINT_ROWS,p+1),np.float64)
    jac[:3,:p] = -loss_jac
    jac[:3,p] = 1.
    at = 3
    for block in (-jq/INNER[1],jq/INNER[1],
                  -jltm/INNER[3],jltm/INNER[3]):
        count = int(np.prod(block.shape[:-1]))
        jac[at:at+count,:p] = block.reshape(count,p)
        at += count
    jac[at:at+3,:p] = -2*np.einsum('ctj,ctjp->cp',q,jq)/(
        99*7*INNER[0]**2)
    at += 3
    jac[at:at+3,:p] = -2*np.einsum('ct,ctp->cp',ltm,jltm)/(
        99*INNER[2]**2)
    if at+3 != CONSTRAINT_ROWS or value.shape != (CONSTRAINT_ROWS,):
        raise ValueError('E128 exact signed constraint row contract changed')
    ratios = np.stack((q_ratio, np.max(np.abs(q),axis=(1,2))**2/INNER[1]**2,
        ltm_ratio, np.max(np.abs(ltm),axis=1)**2/INNER[3]**2),axis=1)
    return value,jac,ratios


def selector_feasible(ratios):
    r = np.asarray(ratios,np.float64)
    return bool(r.shape == (3,4) and np.isfinite(r).all() and
                np.all(r <= SELECTOR_CAP))


def choose_records(records):
    finite = [x for x in records if np.isfinite(x['true_worst_objective'])]
    best_any = min(finite,key=lambda x:x['true_worst_objective']) if finite else None
    feasible = [x for x in finite if selector_feasible(x['ratios'])]
    best = min(feasible,key=lambda x:x['true_worst_objective']) if feasible else None
    return best_any,best


class FitLimit(Exception):
    pass


class EvaluationCache:
    """One float64 delta/gain byte key; epigraph t is never an engine input."""
    def __init__(self, evaluate, size, on_start, *, max_calls, deadline,
                 total_deadline):
        self.evaluate=evaluate
        self.size=size
        self.on_start=on_start
        self.max_calls=max_calls
        self.deadline=deadline
        self.total_deadline=total_deadline
        self.cache={}
        self.started=0
        self.completed=0
        self.hits=0

    def get(self, vector):
        x=np.asarray(vector,np.float64)
        if x.shape != (self.size+1,) or not np.isfinite(x).all():
            raise ValueError('Finite local delta/gain/epigraph required')
        key=x[:-1].tobytes()
        if key in self.cache:
            self.hits+=1
            return self.cache[key],float(x[-1])
        now=time.perf_counter()
        if self.started>=self.max_calls or now>=self.deadline-LEG_RESERVE_S:
            raise FitLimit('per_leg_unique_or_wall_limit')
        if now>=self.total_deadline-PROCESS_RESERVE_S:
            raise FitLimit('whole_fit_wall_limit')
        self.started+=1
        self.on_start(self.started,x[:-1])
        data=self.evaluate(x[:-1],self.started)
        self.cache[key]=data
        self.completed+=1
        return data,float(x[-1])


def objective_and_jac(data, t, edge_count):
    delta=data['delta']
    gain=data['gain']
    prior=6e-5*np.sum(delta*delta)/1572.+1e-5*np.sum(gain*gain)/96.
    jac=np.r_[1.2e-4*delta/1572.,2e-5*gain/96.,1.]
    return float(t+prior),jac,float(prior)


def solve_one_leg(cache, bounds, x0):
    def objective(x):
        data,t=cache.get(x)
        return objective_and_jac(data,t,len(data['delta']))[0]
    def objective_jac(x):
        data,t=cache.get(x)
        return objective_and_jac(data,t,len(data['delta']))[1]
    def values(x):
        data,t=cache.get(x)
        return constraint_values(data['losses'],data['loss_jac'],
            data['q_error'],data['ltm_error'],data['q_jac'],
            data['ltm_jac'],t)[0]
    def jac(x):
        data,t=cache.get(x)
        return constraint_values(data['losses'],data['loss_jac'],
            data['q_error'],data['ltm_error'],data['q_jac'],
            data['ltm_jac'],t)[1]
    return minimize(objective,x0,method='SLSQP',jac=objective_jac,
        bounds=[*bounds,(None,None)],
        constraints=[{'type':'ineq','fun':values,'jac':jac}],
        options={'maxiter':100,'ftol':1e-9,'disp':False})


def fit_cached_leg(cache, bounds, size):
    """Handle a budget expiring before the one explicit zero start."""
    try:
        start,_=cache.get(np.r_[np.zeros(size),0.])
        x0=np.r_[np.zeros(size),float(max(start['losses'])+.01)]
        return solve_one_leg(cache,bounds,x0),'scipy_finished'
    except FitLimit as error:
        return None,str(error)


def no_joint_result(chosen, leg_summaries):
    """Total numerical rejection receipt for six or fewer evaluated legs."""
    if len(chosen)>6:
        raise ValueError('At most six selected leg records')
    return {'status':'no_joint_feasible_candidate',
        'numerical_status':'not_supported',
        'technical_status':'complete_bounded_fit' if len(chosen)==6 else
            'fit_budget_exhausted',
        'feasible_legs':[LEGS[i] for i,x in enumerate(chosen) if x is not None],
        'unfeasible_legs':[LEGS[i] for i,x in enumerate(chosen) if x is None]+
            list(LEGS[len(chosen):]),
        'leg_summaries':leg_summaries,'candidate_adopted':False,
        'F6_exposed':False,'whole_goal_complete':False,
        'actual_neural_s':0.,'actual_physical_s':0.}


def torch_input_array(engine, value, device, label, *, integer=False):
    """Account a small fixed NumPy-to-Torch CUDA loss input."""
    import torch
    value=np.ascontiguousarray(value,dtype=np.int64 if integer else np.float64)
    amount=int(value.nbytes)
    bucket=engine.torch_input_transfer
    labeled=engine.torch_input_transfer_by_label.setdefault(label,{
        state:{'count':0,'bytes':0} for state in
        ('requested','enqueued','synchronized','unknown_requested')})
    bucket['requested']['count']+=1
    bucket['requested']['bytes']+=amount
    labeled['requested']['count']+=1
    labeled['requested']['bytes']+=amount
    try:
        result=torch.as_tensor(value,dtype=torch.int64 if integer else torch.float64,
                               device=device)
    except Exception:
        bucket['unknown_requested']['count']+=1
        bucket['unknown_requested']['bytes']+=amount
        labeled['unknown_requested']['count']+=1
        labeled['unknown_requested']['bytes']+=amount
        raise
    # Torch host-to-CUDA copies here use the default blocking transfer.
    for state in ('enqueued','synchronized'):
        bucket[state]['count']+=1
        bucket[state]['bytes']+=amount
        labeled[state]['count']+=1
        labeled[state]['bytes']+=amount
    return result


def torch_indices(engine, value, device, label):
    """Keep fixed loss-selection indices on CUDA with explicit copy costs."""
    value=np.ascontiguousarray(value,dtype=np.int64)
    key=(str(device),label,value.shape,value.tobytes())
    if key in engine.torch_index_cache:
        engine.torch_index_cache_hits+=1
        return engine.torch_index_cache[key]
    result=torch_input_array(engine,value,device,label,integer=True)
    engine.torch_index_cache[key]=result
    return result


def torch_leg_loss(engine, activation, fixture, leg, geometry, muscle_indices):
    """One context only; activation is the independent CUDA leaf."""
    device=activation.device
    start=7*leg
    moment=torch_input_array(engine,fixture['moment'][start:start+7][
        :,muscle_indices],device,'moment')
    command=torch_input_array(engine,fixture['rest'][start:start+7],
                              device,'rest')+.02*activation@moment.T
    ltm_global=int(fixture['ltm'][leg])
    where=np.flatnonzero(muscle_indices==ltm_global)
    if len(where)!=1:
        raise ValueError('One local leg LTM muscle required')
    import torch
    tendon=.2*torch.tanh(activation[:,int(where[0])]/10.)
    return command,tendon


def torch_host_array(engine, tensor, label):
    """Account one Torch CUDA-to-host materialization, including errors."""
    amount=int(tensor.numel()*tensor.element_size())
    bucket=engine.torch_host_transfer
    labeled=engine.torch_host_transfer_by_label.setdefault(label,{
        state:{'count':0,'bytes':0} for state in
        ('requested','enqueued','synchronized','unknown_requested')})
    bucket['requested']['count']+=1
    bucket['requested']['bytes']+=amount
    labeled['requested']['count']+=1
    labeled['requested']['bytes']+=amount
    try:
        result=tensor.detach().cpu().numpy().copy()
    except Exception:
        bucket['unknown_requested']['count']+=1
        bucket['unknown_requested']['bytes']+=amount
        labeled['unknown_requested']['count']+=1
        labeled['unknown_requested']['bytes']+=amount
        raise
    for state in ('enqueued','synchronized'):
        bucket[state]['count']+=1
        bucket[state]['bytes']+=amount
        labeled[state]['count']+=1
        labeled[state]['bytes']+=amount
    return result


def evaluate_leg(engine, fixture, geometry, leg, vector, index, out):
    import torch
    engine.geometry_backend=type(geometry).__name__
    engine.geometry_device=str(getattr(geometry,'device','synthetic_cuda'))
    p=engine.problem
    e=engine.edges
    delta=np.ascontiguousarray(vector[:e],np.float64)
    gain=np.ascontiguousarray(vector[e:],np.float64)
    try:
        d,h,timing=engine.solve(delta,gain,derivative=True,host=index==1)
    except Exception as error:
        partial={'delta':delta,'gain':gain,'edge_ids':p['edge_ids']}
        partial.update(getattr(error,'partial_response',{}))
        try:
            np.savez(out/f'leg-{leg}-evaluation-{index:03d}-partial-raw.npz',
                     **partial)
        except Exception as secondary:
            error.add_note('Secondary leg CUDA partial save: '+repr(secondary))
        raise
    muscle_indices=p['muscle_indices']
    starts=np.flatnonzero(fixture['startup_mask'])
    losses=np.empty(3)
    parts=np.empty((3,4))
    q_error=np.empty((3,99,7))
    ltm_error=np.empty((3,99))
    q_coefficient=np.empty((3,99,7,engine.muscles))
    ltm_coefficient=np.empty((3,99))
    raw={'delta':delta,'gain':gain,'edge_ids':p['edge_ids'],
         'motor_indices':p['motor_indices'],
         'muscle_indices':muscle_indices,
         'parameter_muscle_map':activation_parameter_map(p),
         'B_initial99':None,
         'timing_context_ticks':np.array(timing['context_rk4_ticks']),
         'timing_bank_complete':np.array(
             timing['progress']['derivative_bank_complete'])}
    try:
        engine.stream.synchronize()
        raw['B_initial99']=initial_activation_bank(engine,d,starts)
        # One independent activation leaf and one backward for all contexts.
        # No neural time-step is part of this Torch graph.
        activation_all=torch.from_dlpack(d['activation']).detach().requires_grad_(True)
        engine.torch_activation_shared_bytes+=int(d['activation'].nbytes)
        start_indices=torch_indices(engine,starts,activation_all.device,
                                     'startup_indices')
        loss_tensors=[]
        for ci in range(3):
            selected=np.flatnonzero(fixture['fit_time_mask'][ci])
            if len(selected)<2 or not np.all(np.diff(selected)==1):
                raise ValueError('Contiguous E128 selected fit mask required')
            # The private CuPy stream has completed. The Torch leaf has no
            # neural history graph; only fixed geometry/loss is differentiated.
            activation=activation_all[ci]
            selected_indices=torch_indices(engine,selected,activation.device,
                                            'fit_indices')
            command,tendon=torch_leg_loss(engine,activation,fixture,leg,geometry,
                                           muscle_indices)
            engine.geometry_cost['forward_rows_requested']+=len(selected)
            foot=geometry.leg(command[selected_indices],leg)
            engine.geometry_cost['forward_rows_enqueued']+=len(selected)
            velocity=(foot[1:]-foot[:-1])/.005
            activation_device=activation.device
            def to(x,label,device=activation_device):
                return torch_input_array(engine,x,device,label)
            start=7*leg
            q_target=to(fixture['own_target'][ci,selected,start:start+7],
                        'q_target')
            foot_target=to(fixture['target_feet'][ci,selected,leg],
                           'foot_target')
            v_target=(foot_target[1:]-foot_target[:-1])/.005
            q_loss=((command[selected_indices]-q_target)/
                    to(fixture['joint_scale'][start:start+7],
                       'joint_scale')).square().mean()
            foot_loss=((foot-foot_target)/
                       to(fixture['foot_scale'][leg],
                          'foot_scale')).square().mean()
            vel_loss=((velocity-v_target)/
                      to(fixture['velocity_scale'][leg],
                         'velocity_scale')).square().mean()
            ltm_loss=((tendon[selected_indices]-.2)/.2).square().mean()
            values=torch.stack((q_loss,foot_loss,vel_loss,ltm_loss))
            loss=values.sum()
            loss_tensors.append(loss)
            parts[ci]=torch_host_array(engine,values,'loss_parts')
            losses[ci]=float(torch_host_array(engine,loss,'loss_scalar'))
            q_error[ci]=torch_host_array(engine,command[start_indices]-to(
                fixture['startup_target_command'][:,start:start+7],
                'startup_target_command'),
                'initial_q_error')
            ltm_error[ci]=torch_host_array(engine,tendon[start_indices]-to(
                fixture['startup_target_tendon'][:,leg],
                'startup_target_tendon'),'initial_ltm_error')
            q_coefficient[ci]=np.broadcast_to(
                .02*fixture['moment'][start:start+7][:,muscle_indices][None,:,:],
                (99,7,engine.muscles))
            ltm_coefficient[ci]=(.02*(1.-np.tanh(
                torch_host_array(engine,activation[start_indices,int(
                    np.flatnonzero(muscle_indices==fixture['ltm'][leg])[0])],
                    'initial_ltm_activation')/10.)**2))
            raw[f'context{ci}_command']=torch_host_array(engine,command,
                'first_command') if index==1 else np.empty(0)
            raw[f'context{ci}_tendon']=torch_host_array(engine,tendon,
                'first_tendon') if index==1 else np.empty(0)
        engine.geometry_cost['backward_rows_requested']+=int(
            fixture['fit_time_mask'].sum())
        cotangent=torch.autograd.grad(sum(loss_tensors),activation_all)[0]
        engine.geometry_cost['backward_rows_enqueued']+=int(
            fixture['fit_time_mask'].sum())
        torch.cuda.synchronize(activation_all.device)
        engine.geometry_cost['forward_rows_completed']+=int(
            fixture['fit_time_mask'].sum())
        engine.geometry_cost['backward_rows_completed']+=int(
            fixture['fit_time_mask'].sum())
        loss_jac=activation_cotangent_product(engine,d,cotangent)
        cotangent=torch_host_array(engine,cotangent,'loss_cotangent_raw')
        B=raw['B_initial99']
        mapping=raw['parameter_muscle_map']
        q_jac=np.empty((3,99,7,len(mapping)),np.float64)
        ltm_jac=np.empty((3,99,len(mapping)),np.float64)
        ltm_local=int(np.flatnonzero(muscle_indices==fixture['ltm'][leg])[0])
        for param,muscle in enumerate(mapping):
            # B's muscle axis is local; the fixture moment axis is global.
            q_jac[:,:,:,param]=B[:,:,param,None]*(
                .02*fixture['moment'][7*leg:7*leg+7,
                                       muscle_indices[muscle]])[None,None,:]
            ltm_jac[:,:,param]=(
                B[:,:,param]*ltm_coefficient if muscle==ltm_local else 0.)
        _,_,ratios=constraint_values(losses,loss_jac,q_error,ltm_error,
                                      q_jac,ltm_jac,max(losses)+.01)
        raw.update(losses=losses,parts=parts,loss_jac=loss_jac,
            q_error=q_error,ltm_error=ltm_error,q_coefficient=q_coefficient,
            ltm_coefficient=ltm_coefficient,cotangent=cotangent,
            ratios=ratios,
            transfer_cost_json=np.array(json.dumps(jsonable({
                'host_transfer':engine.host_transfer,
                'host_transfer_by_label':engine.host_transfer_by_label,
                'torch_host_transfer':engine.torch_host_transfer,
                'torch_host_transfer_by_label':
                    engine.torch_host_transfer_by_label,
                'cotangent_upload':engine.cotangent_upload,
                'index_upload':engine.index_upload,
                'index_upload_by_label':engine.index_upload_by_label,
                'torch_input_transfer':engine.torch_input_transfer,
                'torch_input_transfer_by_label':
                    engine.torch_input_transfer_by_label,
                'torch_cupy_shared_bytes':engine.torch_cupy_shared_bytes,
                'torch_contiguous_copy_bytes':
                    engine.torch_contiguous_copy_bytes,
                'geometry_cost':engine.metadata()['geometry_cost']}))))
        if index==1:
            for key in (*FIELDS,'branch_bits'):
                raw[key]=h[key]
        path=out/f'leg-{leg}-evaluation-{index:03d}-raw.npz'
        np.savez(path,**raw)
        if any(not np.isfinite(value).all() for value in
               (losses,loss_jac,q_error,ltm_error,q_jac,ltm_jac,
                ratios,raw['B_initial99'])):
            raise ValueError('Nonfinite E128 leg raw saved: '+str(path))
        prior=6e-5*np.sum(delta*delta)/1572.+1e-5*np.sum(gain*gain)/96.
        return {'index':index,'delta':delta,'gain':gain,'losses':losses,
            'parts':parts,'loss_jac':loss_jac,'q_error':q_error,
            'ltm_error':ltm_error,'q_jac':q_jac,'ltm_jac':ltm_jac,
            'ratios':ratios,'prior':float(prior),
            'true_worst_objective':float(max(losses)+prior),
            'raw_path':str(path),'raw_sha256':sha256(path),
            'wall_s':timing['wall_s']}
    except Exception as error:
        error.response_progress=timing['progress']
        try:
            torch.cuda.synchronize()
            engine.geometry_cost['forward_rows_completed']=max(
                engine.geometry_cost['forward_rows_completed'],
                engine.geometry_cost['forward_rows_enqueued'])
            engine.geometry_cost['backward_rows_completed']=max(
                engine.geometry_cost['backward_rows_completed'],
                engine.geometry_cost['backward_rows_enqueued'])
        except Exception as secondary:
            error.add_note('Secondary geometry stream synchronization: '+
                           repr(secondary))
        rescued=dict(h)
        more,copy_errors=engine.rescue_device_state(d,starts)
        for key,value in more.items():
            rescued.setdefault(key,value)
        for item in copy_errors:
            error.add_note('Secondary leg response rescue '+
                           item['buffer']+': '+item['error'])
        error.engine_metadata=engine.metadata()
        raw.update(rescued)
        try:
            safe={key:value for key,value in raw.items()
                  if isinstance(value,np.ndarray) or np.isscalar(value)}
            safe['failure_type']=np.array(type(error).__name__)
            safe['failure_message']=np.array(str(error))
            safe['failure_progress_json']=np.array(json.dumps(jsonable({
                'response':timing['progress'],'engine':engine.metadata()})))
            np.savez(out/f'leg-{leg}-evaluation-{index:03d}-partial-raw.npz',
                     **safe)
        except Exception as secondary:
            error.add_note('Secondary partial raw save: '+repr(secondary))
        raise


def initial_metrics(response, fixture, leg=None):
    q=response['command'][:,fixture['startup_mask']]-fixture[
        'startup_target_command'][None,:,:]
    z=response['tendon'][:,fixture['startup_mask']]-fixture[
        'startup_target_tendon'][None,:,:]
    if leg is not None:
        q=q[:,:,7*leg:7*leg+7]
        z=z[:,:,leg]
        q_rms=np.sqrt(np.mean(q*q,axis=1 if q.ndim==2 else (1,2)))
        z_rms=np.sqrt(np.mean(z*z,axis=1))
        q_max=np.max(np.abs(q),axis=1 if q.ndim==2 else (1,2))
        z_max=np.max(np.abs(z),axis=1)
    else:
        q_rms=float(np.sqrt(np.mean(q*q)))
        z_rms=float(np.sqrt(np.mean(z*z)))
        q_max=float(np.max(np.abs(q)))
        z_max=float(np.max(np.abs(z)))
    return {'q_rms':q_rms,'q_max':q_max,'ltm_rms':z_rms,
            'ltm_max':z_max}


def score_context(command,tendon,feet,mask,target_q,target_feet,
                  scale_q,scale_foot,scale_velocity):
    selected=np.flatnonzero(mask)
    if len(selected)<2 or not np.all(np.diff(selected)==1):
        raise ValueError('Scored time mask must be contiguous')
    dq=command[selected]-target_q[selected]
    df=feet[selected]-target_feet[selected]
    dv=np.diff(df,axis=0)/.005
    dl=tendon[selected]-.2
    return {'count':len(selected),
        'q_rms':float(np.sqrt(np.mean(dq*dq))),
        'foot_rms':float(np.sqrt(np.mean(df*df))),
        'velocity_rms':float(np.sqrt(np.mean(dv*dv))),
        'ltm_rms':float(np.sqrt(np.mean(dl*dl))),
        'LF_row6_rms':float(np.sqrt(np.mean(dq[:,6]**2))),
        'per_leg_q_rms':[float(np.sqrt(np.mean(dq[:,7*i:7*i+7]**2))) for i in range(6)],
        'per_leg_foot_rms':[float(np.sqrt(np.mean(df[:,i]**2))) for i in range(6)],
        'per_leg_velocity_rms':[float(np.sqrt(np.mean(dv[:,i]**2))) for i in range(6)],
        'parts':[float(np.mean((dq/scale_q)**2)),
                 float(np.mean((df/scale_foot)**2)),
                 float(np.mean((dv/scale_velocity)**2)),
                 float(np.mean((dl/.2)**2))]}


def shape_screen(final,baseline):
    ratios={k:final[k]/baseline[k] for k in
            ('q_rms','foot_rms','velocity_rms')}
    ratios['supported']=bool(ratios['q_rms']<=1.05 and
        ratios['foot_rms']<=.8 and ratios['velocity_rms']<=.8)
    ratios['per_leg_q_ratios']=(np.asarray(final['per_leg_q_rms'])/
                                np.asarray(baseline['per_leg_q_rms'])).tolist()
    ratios['per_leg_q_supported']=bool(all(x<=1.05 for x in
                                           ratios['per_leg_q_ratios']))
    return ratios


def full_feet(geometry, response, ledger=None, label='full_geometry'):
    import torch
    cost={'label':label,'backend':type(geometry).__name__,
          'device':str(geometry.device),'dtype':'float64',
          'rows_requested':0,'rows_completed':0,
          'rows_unknown_requested':0,
          'command_h2d_bytes_requested':0,
          'command_h2d_bytes_completed':0,
          'command_h2d_bytes_unknown_requested':0,
          'foot_d2h_bytes_requested':0,
          'foot_d2h_bytes_completed':0,
          'foot_d2h_bytes_unknown_requested':0}
    if ledger is not None:
        ledger.setdefault('full_geometry',[]).append(cost)
    with torch.no_grad():
        values=[]
        for c in range(3):
            cost['command_h2d_bytes_requested']+=int(
                response['command'][c].nbytes)
            cost['command_h2d_bytes_unknown_requested']=(
                cost['command_h2d_bytes_requested']-
                cost['command_h2d_bytes_completed'])
            command=torch.as_tensor(response['command'][c],dtype=torch.float64,
                                    device=geometry.device)
            cost['command_h2d_bytes_completed']+=int(
                response['command'][c].nbytes)
            cost['command_h2d_bytes_unknown_requested']=(
                cost['command_h2d_bytes_requested']-
                cost['command_h2d_bytes_completed'])
            cost['rows_requested']+=len(command)
            cost['rows_unknown_requested']=(cost['rows_requested']-
                                            cost['rows_completed'])
            foot=geometry.full(command)
            cost['foot_d2h_bytes_requested']+=int(
                foot.numel()*foot.element_size())
            cost['foot_d2h_bytes_unknown_requested']=(
                cost['foot_d2h_bytes_requested']-
                cost['foot_d2h_bytes_completed'])
            values.append(foot.cpu().numpy())
            cost['foot_d2h_bytes_completed']+=int(
                foot.numel()*foot.element_size())
            cost['foot_d2h_bytes_unknown_requested']=(
                cost['foot_d2h_bytes_requested']-
                cost['foot_d2h_bytes_completed'])
            cost['rows_completed']+=len(command)
            cost['rows_unknown_requested']=(cost['rows_requested']-
                                            cost['rows_completed'])
    return np.stack(values)


def full_response(engine, delta, gain, name, out, ledger, *, native_mode=False,
                  extra=None):
    if ledger['full_response_batches_requested'] >= 4:
        raise FitLimit('E128 maximum four full batches')
    ledger['full_response_batches_requested']+=1
    ledger['full_requested_context_ticks']+=engine.contexts*engine.ticks
    ledger['full_mn_ticks_requested']+=engine.contexts*engine.ticks*engine.neurons
    ledger['full_decoder_ticks_requested']+=engine.contexts*(engine.samples-1)
    write_json(out/'advance-ledger.json',ledger)
    try:
        _device,host,timing=engine.solve(delta,gain,native_mode=native_mode)
    except Exception as error:
        progress=getattr(error,'response_progress',None)
        ledger['active_full_progress']=progress
        if progress is not None:
            completed=progress.get('completed_context_rk4_ticks')
            if completed is None:
                if progress.get('completion_class')!='not_enqueued':
                    ledger['full_context_ticks_unknown_requested']+=\
                        engine.contexts*engine.ticks
            else:
                ledger['full_context_ticks_completed']+=int(completed)
                ledger['full_conditional_s_completed']+=int(completed)*.00025
                ledger['full_mn_ticks_completed']+=int(completed)*engine.neurons
                if progress.get('completed_decoder_endpoint_ticks') is not None:
                    ledger['full_decoder_ticks_completed']+=int(
                        progress['completed_decoder_endpoint_ticks'])
            if (progress.get('completed_decoder_endpoint_ticks') is None and
                    progress.get('decoder_completion_class')=='unknown'):
                ledger['full_decoder_ticks_unknown_requested']=\
                    ledger.get('full_decoder_ticks_unknown_requested',0)+\
                    engine.contexts*(engine.samples-1)
        partial=getattr(error,'partial_response',{})
        if partial:
            try:
                np.savez(out/(name+'-partial-raw.npz'),**partial)
            except Exception as secondary:
                error.add_note('Secondary full partial save: '+repr(secondary))
        raise
    ledger['full_context_ticks_completed']+=timing['context_rk4_ticks']
    ledger['full_conditional_s_completed']+=timing['conditional_model_s']
    ledger['full_mn_ticks_completed']+=timing['context_rk4_ticks']*engine.neurons
    ledger['full_decoder_ticks_completed']+=timing['decoder_endpoint_ticks']
    path=out/(name+'-raw.npz')
    try:
        np.savez(path,gain96=np.asarray(gain,np.float64),
                 delta=np.asarray(delta,np.float64),**host,**(extra or {}))
    except Exception as error:
        try:
            np.savez(out/(name+'-emergency-raw.npz'),
                     gain96=np.asarray(gain,np.float64),
                     delta=np.asarray(delta,np.float64),**host,**(extra or {}))
        except Exception as secondary:
            error.add_note('Secondary full response emergency save: '+repr(secondary))
        raise
    ledger['full_response_batches_completed']+=1
    ledger['full_raw_sha256'][name]=sha256(path)
    ledger['active_full_progress']=timing['progress']
    write_json(out/'advance-ledger.json',ledger)
    if any(not np.isfinite(host[k]).all() for k in FIELDS):
        raise ValueError('Nonfinite saved E128 full response: '+name)
    return host


def compare_reference(actual, reference_path, atol):
    ref=load_npz(reference_path)
    fields={key:float(np.max(np.abs(actual[key]-ref[key]))) for key in FIELDS}
    branch=bool(np.array_equal(actual['branch_bits'],ref['branch_bits']))
    return {'field_max_abs':fields,'branch_exact':branch,
            'pass':bool(branch and all(v<=atol for v in fields.values()))}


def run(freeze_path,freeze_sha,review_path,review_sha,out):
    began=time.perf_counter()
    out=Path(out)
    if out.exists():
        raise FileExistsError('One-shot E128 output exists: '+str(out))
    out.mkdir(parents=True,exist_ok=False)
    ledger={'schema':'flylab.m3.e128.advance-ledger.v1',
        'freeze_sha256':freeze_sha,'source_review_sha256':review_sha,
        'full_response_batches_requested':0,'full_response_batches_completed':0,
        'full_requested_context_ticks':0,'full_context_ticks_completed':0,
        'full_context_ticks_unknown_requested':0,
        'full_mn_ticks_requested':0,'full_mn_ticks_completed':0,
        'full_decoder_ticks_requested':0,'full_decoder_ticks_completed':0,
        'full_decoder_ticks_unknown_requested':0,
        'full_conditional_s_completed':0.,'full_raw_sha256':{},
        'leg_unique_started':0,'leg_unique_completed':0,
        'leg_context_ticks_requested':0,'leg_context_ticks_completed':0,
        'leg_context_ticks_unknown_requested':0,
        'leg_mn_ticks_requested':0,'leg_mn_ticks_completed':0,
        'leg_edge_stages_requested':0,'leg_edge_stages_completed':0,
        'leg_edge_stages_unknown_requested':0,
        'leg_decoder_ticks_requested':0,'leg_decoder_ticks_completed':0,
        'leg_decoder_ticks_unknown_requested':0,
        'geometry_rows_completed':0,
        'derivative_banks_requested':0,
        'derivative_banks_unknown_requested':0,
        'leg_conditional_s_completed':0.,'derivative_banks_completed':0,
        'leg_attempts':[],'active_leg':None,'active_full_progress':None,
        'cuda_tiny_setup':None,'actual_neural_s':0.,'actual_physical_s':0.,
        'candidate_adopted':False,'F6_exposed':False,
        'whole_goal_complete':False}
    write_json(out/'advance-ledger.json',ledger)
    limits=full_engine=active_leg_engine=None
    primary=None
    secondary=[]
    receipt=None
    try:
        limits,threads=one_cpu_thread()
        freeze,review=reviewed_inputs(freeze_path,freeze_sha,
                                      review_path,review_sha)
        write_json(out/'source-review-snapshot.json',review)
        write_json(out/'spec.json',{'freeze_sha256':freeze_sha,
            'source_review_sha256':review_sha,
            'fixture_sha256':freeze['input_sha256'][freeze['fixture_path']],
            'design_sha256':freeze['design_sha256'],
            'source_sha256':review['source_sha256'],
            'geometry_hash':freeze['geometry_hash'],
            'conditional_only':True,'body_closed_loop':False,
            'candidate_adopted':False,'F6_exposed':False})
        if time.perf_counter()-began>=INTERNAL_LIMIT_S-PROCESS_RESERVE_S:
            raise FitLimit('E128 preflight time exhausted')
        fixture=frozen_problem(freeze)
        import torch
        geometry_upload={'backend':'TorchFootGeometry',
            'requested_device':'cuda','status':'requested',
            'fixed_h2d_bytes_requested':None,
            'fixed_h2d_bytes_enqueued':None,
            'fixed_h2d_bytes_completed':None,
            'fixed_h2d_bytes_unknown_requested':None}
        ledger['fixed_geometry_upload']=geometry_upload
        write_json(out/'advance-ledger.json',ledger)
        try:
            geometry=TorchFootGeometry.load(ROOT/freeze['geometry_path'],
                                            device='cuda')
            fixed_bytes=sum(int(geometry.arrays[key].nbytes) for key in
                ('axes','anchors','tips','reference_angles'))
            geometry_upload.update(device=str(geometry.device),
                fixed_h2d_bytes_requested=fixed_bytes,
                fixed_h2d_bytes_enqueued=fixed_bytes,
                status='enqueued')
            torch.cuda.synchronize(geometry.device)
            geometry_upload.update(fixed_h2d_bytes_completed=fixed_bytes,
                fixed_h2d_bytes_unknown_requested=0,status='completed')
        except Exception as error:
            geometry_upload['status']='unknown'
            geometry_upload['fixed_h2d_bytes_unknown_requested']=(
                None if geometry_upload['fixed_h2d_bytes_requested'] is None
                else geometry_upload['fixed_h2d_bytes_requested']-
                     (geometry_upload['fixed_h2d_bytes_completed'] or 0))
            try:
                write_json(out/'advance-ledger.json',ledger)
            except Exception as secondary:
                error.add_note('Secondary fixed geometry ledger save: '+
                               repr(secondary))
            raise
        if geometry.geometry_hash!=freeze['geometry_hash']:
            raise ValueError('E128 fixed E111 geometry hash changed')
        import cupy as cp
        gpu=cp.cuda.runtime.getDeviceProperties(0)['name']
        write_json(out/'environment.json',{'python':sys.version,
            'numpy':np.__version__,'cupy':cp.__version__,
            'torch':torch.__version__,'gpu':gpu.decode() if isinstance(gpu,bytes) else str(gpu),
            'cpu_threads':threads})
        # Separate explicit synthetic CUDA setup; no original stage is used.
        run_tiny_setup(out,ledger)
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        full_engine=construct_engine(fixture,ledger,out,'full_original_fixed')
        ledger['full_engine_metadata']=full_engine.metadata()
        write_json(out/'advance-ledger.json',ledger)
        zeros_e=np.zeros(1572)
        zeros_g=np.zeros(96)
        gain0=full_response(full_engine,zeros_e,zeros_g,'native-gain0',out,
                            ledger,native_mode=True)
        center=full_response(full_engine,zeros_e,fixture['center_gain'],
                            'native-center',out,ledger,native_mode=True)
        e127=ROOT/freeze['E127_directory']
        checks={'native_gain0':compare_reference(gain0,e127/'gain0-raw.npz',1e-8),
                'native_center':compare_reference(center,e127/'center-raw.npz',1e-8)}
        checks['native_pass']=bool(all(x['pass'] for x in checks.values()))
        if not checks['native_pass']:
            write_json(out/'preflight.json',checks)
            raise ValueError('E128 native E127 preflight mismatch')
        baseline=full_response(full_engine,zeros_e,zeros_g,'transported-baseline',
                               out,ledger)
        checks['E121_self_gain0']={key:float(np.max(np.abs(
            baseline[key][2]-gain0[key][2]))) for key in FIELDS}
        checks['E121_self_branch_exact']=bool(np.array_equal(
            baseline['branch_bits'][2],gain0['branch_bits'][2]))
        checks['transported_self_pass']=bool(
            checks['E121_self_branch_exact'] and
            max(checks['E121_self_gain0'].values())<=1e-8)
        write_json(out/'preflight.json',checks)
        if not checks['transported_self_pass']:
            raise ValueError('E128 E121 self transport mismatch')
        baseline_feet=full_feet(geometry,baseline,ledger,'transported_baseline')
        np.savez(out/'transported-baseline-feet.npz',feet=baseline_feet)
        baseline_own=[]
        for ci in range(3):
            baseline_own.append(score_context(baseline['command'][ci],
                baseline['tendon'][ci],baseline_feet[ci],
                fixture['own_mask'][ci],fixture['own_target'][ci],
                fixture['target_feet'][ci],fixture['joint_scale'],
                fixture['foot_scale'],fixture['velocity_scale']))
        write_json(out/'baseline-metrics.json',{'own_phase':baseline_own,
            'startup':initial_metrics(baseline,fixture),
            'preflight':checks})
        fit_began=time.perf_counter()
        fit_deadline=min(fit_began+FIT_LIMIT_S,began+INTERNAL_LIMIT_S)
        chosen=[]
        leg_summaries=[]
        for leg,name in enumerate(LEGS):
            leg_began=time.perf_counter()
            ledger['active_leg']=name
            motors=np.flatnonzero(fixture['leg_motor_mask'][leg])
            muscles=np.flatnonzero(fixture['leg_muscle_mask'][leg])
            edges=np.flatnonzero(fixture['leg_edge_mask'][leg])
            leg_engine=construct_engine(fixture,ledger,out,
                'leg_'+name,motors,muscles,edges)
            active_leg_engine=leg_engine
            records=[]
            def on_start(index,vector,_engine=leg_engine):
                ledger['leg_unique_started']+=1
                ledger['leg_context_ticks_requested']+=_engine.contexts*_engine.ticks
                ledger['leg_mn_ticks_requested']+=_engine.contexts*_engine.ticks*_engine.neurons
                ledger['leg_edge_stages_requested']+=_engine.contexts*_engine.ticks*4*_engine.edges
                ledger['leg_decoder_ticks_requested']+=_engine.contexts*(_engine.samples-1)
                ledger['derivative_banks_requested']+=1
                ledger['active_geometry_rows_before']=\
                    _engine.geometry_cost['forward_rows_completed']
                ledger['active_vector']=vector.tolist()
                write_json(out/'advance-ledger.json',ledger)
            def evaluate(vector,index,_engine=leg_engine,_leg=leg,
                         _name=name,_records=records):
                try:
                    record=evaluate_leg(_engine,fixture,geometry,_leg,vector,
                                        index,out)
                except Exception as error:
                    progress=getattr(error,'response_progress',None)
                    metadata=getattr(error,'engine_metadata',_engine.metadata())
                    ledger['active_leg_engine_metadata']=metadata
                    ledger['geometry_rows_completed']+=max(0,
                        _engine.geometry_cost['forward_rows_completed']-
                        ledger.get('active_geometry_rows_before',0))
                    if progress is not None:
                        completed=progress.get('completed_context_rk4_ticks')
                        if completed is None:
                            if progress.get('completion_class')!='not_enqueued':
                                ledger['leg_context_ticks_unknown_requested']+=\
                                    _engine.contexts*_engine.ticks
                        else:
                            ledger['leg_context_ticks_completed']+=int(completed)
                            ledger['leg_conditional_s_completed']+=\
                                int(completed)*.00025
                            ledger['leg_mn_ticks_completed']+=int(completed)*_engine.neurons
                        if progress.get('derivative_bank_complete'):
                            ledger['derivative_banks_completed']+=1
                            ledger['leg_edge_stages_completed']+=_engine.contexts*_engine.ticks*4*_engine.edges
                        elif progress.get('bank_completion_class')=='unknown':
                            ledger['derivative_banks_unknown_requested']=\
                                ledger.get('derivative_banks_unknown_requested',0)+1
                            ledger['leg_edge_stages_unknown_requested']=\
                                ledger.get('leg_edge_stages_unknown_requested',0)+\
                                _engine.contexts*_engine.ticks*4*_engine.edges
                        if progress.get('completed_decoder_endpoint_ticks') is not None:
                            ledger['leg_decoder_ticks_completed']+=int(
                                progress['completed_decoder_endpoint_ticks'])
                        elif progress.get('decoder_completion_class')=='unknown':
                            ledger['leg_decoder_ticks_unknown_requested']=\
                                ledger.get('leg_decoder_ticks_unknown_requested',0)+\
                                _engine.contexts*(_engine.samples-1)
                    ledger['leg_attempts'].append({'leg':_name,'index':index,
                        'status':'technical_error','error':repr(error),
                        'progress':progress,'engine_metadata':metadata,
                        'partial_raw_path':str(out/
                            f'leg-{_leg}-evaluation-{index:03d}-partial-raw.npz')})
                    ledger['active_leg_progress']=progress
                    write_json(out/'advance-ledger.json',ledger)
                    raise
                _records.append(record)
                ledger['leg_unique_completed']+=1
                ledger['leg_context_ticks_completed']+=_engine.contexts*_engine.ticks
                ledger['leg_conditional_s_completed']+=_engine.contexts*_engine.ticks*.00025
                ledger['leg_mn_ticks_completed']+=_engine.contexts*_engine.ticks*_engine.neurons
                ledger['leg_edge_stages_completed']+=_engine.contexts*_engine.ticks*4*_engine.edges
                ledger['leg_decoder_ticks_completed']+=_engine.contexts*(_engine.samples-1)
                ledger['geometry_rows_completed']+=int(fixture['fit_time_mask'].sum())
                ledger['derivative_banks_completed']+=1
                ledger['leg_attempts'].append({'leg':_name,'index':index,
                    'raw_sha256':record['raw_sha256'],
                    'true_worst_objective':record['true_worst_objective'],
                    'selector_feasible':selector_feasible(record['ratios'])})
                write_json(out/'advance-ledger.json',ledger)
                return record
            size=len(edges)+len(muscles)
            cache=EvaluationCache(evaluate,size,on_start,max_calls=100,
                deadline=leg_began+LEG_LIMIT_S,total_deadline=fit_deadline)
            # Explicit zero start; a pre-start cap creates no new response.
            bounds=[tuple(row) for row in fixture['delta_bounds'][edges]]+[
                (0.,8.)]*len(muscles)
            optimizer,stop=fit_cached_leg(cache,bounds,size)
            best_any,best=choose_records(records)
            summary={'leg':name,'stop':stop,'wall_s':time.perf_counter()-leg_began,
                'unique_started':cache.started,'unique_completed':cache.completed,
                'cache_hits':cache.hits,'candidate_feasible':best is not None,
                'best_any_index':None if best_any is None else best_any['index'],
                'best_feasible_index':None if best is None else best['index'],
                'best_feasible_objective':None if best is None else
                    best['true_worst_objective'],
                'optimizer':None if optimizer is None else {
                    'success':bool(optimizer.success),'status':int(optimizer.status),
                    'message':str(optimizer.message),'nit':int(optimizer.nit),
                    'nfev':int(optimizer.nfev)},
                'engine_metadata':leg_engine.metadata()}
            leg_summaries.append(summary)
            write_json(out/f'leg-{leg}-summary.json',summary)
            chosen.append(best)
            ledger['active_leg']=None
            ledger['active_vector']=None
            ledger['full_engine_metadata']=full_engine.metadata()
            write_json(out/'advance-ledger.json',ledger)
            del cache,on_start,evaluate,records,optimizer,leg_engine
            active_leg_engine=None
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            if time.perf_counter()>=fit_deadline-PROCESS_RESERVE_S:
                break
        ledger['fit_wall_s']=time.perf_counter()-fit_began
        if len(chosen)!=6 or any(x is None for x in chosen):
            write_json(out/'metrics.json',{'preflight':checks,
                'baseline_own_phase':baseline_own,
                'leg_summaries':leg_summaries,
                'numerical_status':'no_joint_feasible_candidate'})
            receipt=no_joint_result(chosen,leg_summaries)
        else:
            delta=np.zeros(1572)
            gain=np.zeros(96)
            for leg,record in enumerate(chosen):
                delta[fixture['leg_edge_mask'][leg]]=record['delta']
                gain[fixture['leg_muscle_mask'][leg]]=record['gain']
            efficacy=fixture['base_efficacy']*np.exp(delta)
            if np.any(efficacy<.001-1e-12) or np.any(efficacy>10.+1e-12):
                raise ValueError('Selected efficacy left native bounds')
            efficacy=np.clip(efficacy,.001,10.)
            from tools.fit_c_banc_m3_dense_foot_recruitment import (
                selected_actual_float32,
            )
            train=load_npz(ROOT/freeze['legacy_train_path'])
            cal=load_npz(ROOT/freeze['calibration_path'])
            if not np.array_equal(train['edge_ids'],fixture['edge_ids']):
                raise ValueError('Legacy export selected edge ID order changed')
            _,updated=selected_actual_float32(train,cal,efficacy)
            positions={int(edge):i for i,edge in enumerate(updated['edge_ids'])}
            actual_values=np.array([updated['values'][positions[int(edge)]]
                                    for edge in fixture['edge_ids']],np.float64)
            actual_gain=np.asarray(gain,np.float32).astype(np.float64)
            actual_delta=np.log(actual_values/fixture['base_fit_weights'])
            if not np.isfinite(actual_delta).all():
                raise ValueError('Nonfinite actual float32 selected delta')
            np.savez(out/'calibration.npz',**updated)
            np.savez(out/'selected-values-raw.npz',edge_ids=fixture['edge_ids'],
                selected_edge_values=actual_values,solver_delta=delta,
                actual_delta=actual_delta,solver_gain=gain,gain96=actual_gain,
                efficacy=efficacy)
            final=full_response(full_engine,actual_delta,actual_gain,
                'final-response',out,ledger,extra={
                    'selected_edge_values':actual_values,
                    'solver_delta':delta,'edge_ids':fixture['edge_ids']})
            final_feet=full_feet(geometry,final,ledger,'final_actual_float32')
            np.savez(out/'final-feet-raw.npz',feet=final_feet)
            from tools.fit_c_banc_m3_context_trust import frozen_output_metrics
            old_fixture=load_npz(ROOT/freeze['legacy_fixture_path'])
            old_parent=load_npz(ROOT/freeze['legacy_parent_data_path'])
            legacy_raw=load_npz(ROOT/
                'verification/banc-natural-gait-m3-20260919/e114-context-validation/response-raw.npz')
            legacy_feet=load_npz(ROOT/
                'verification/banc-natural-gait-m3-20260919/e114-context-validation/foot-proxy-raw.npz')
            legacy=frozen_output_metrics(
                {'command':final['command'][:2],
                 'tendon':final['tendon'][:2]},
                final_feet[:2,fixture['fixed_mask']],
                {'fixture':old_fixture,
                 'independent':{'common_target_feet':old_parent['common_target_feet']},
                 'e114':legacy_raw,'e114_feet':legacy_feet,
                 'root':ROOT/'verification/banc-natural-gait-m3-20260919'})
            scores=[]
            for ci in range(3):
                target=fixture['own_target'][ci]
                target_feet=fixture['target_feet'][ci]
                mask=fixture['fit_time_mask'][ci]
                f=score_context(final['command'][ci],final['tendon'][ci],
                    final_feet[ci],mask,target,target_feet,
                    fixture['joint_scale'],fixture['foot_scale'],
                    fixture['velocity_scale'])
                if ci<2:
                    base_foot=np.full_like(final_feet[ci],np.nan)
                    fixed=np.flatnonzero(fixture['fixed_mask'])
                    base_foot[fixed]=legacy_feet['feet_fixed'][1 if ci==0 else 3]
                    base=score_context(legacy_raw['command'][1 if ci==0 else 3],
                        legacy_raw['tendon'][1 if ci==0 else 3],base_foot,
                        mask,target,target_feet,fixture['joint_scale'],
                        fixture['foot_scale'],fixture['velocity_scale'])
                else:
                    base=score_context(baseline['command'][ci],
                        baseline['tendon'][ci],baseline_feet[ci],
                        mask,target,target_feet,fixture['joint_scale'],
                        fixture['foot_scale'],fixture['velocity_scale'])
                scores.append({'name':str(fixture['context_names'][ci]),
                    'final':f,'baseline':base,'ratios':shape_screen(f,base)})
            initial=initial_metrics(final,fixture)
            per_leg=[]
            for leg in range(6):
                q=final['command'][:,fixture['startup_mask'],7*leg:7*leg+7]-\
                    fixture['startup_target_command'][None,:,7*leg:7*leg+7]
                z=final['tendon'][:,fixture['startup_mask'],leg]-\
                    fixture['startup_target_tendon'][None,:,leg]
                values=np.stack((np.sqrt(np.mean(q*q,axis=(1,2))),
                    np.max(np.abs(q),axis=(1,2)),
                    np.sqrt(np.mean(z*z,axis=1)),
                    np.max(np.abs(z),axis=1)),axis=1)
                per_leg.append({'values':values,
                    'inner_ratios':(values/INNER)**2,
                    'inner_pass':bool(np.all((values/INNER)**2<=SELECTOR_CAP)),
                    'original_pass':bool(np.all(values<=ORIGINAL))})
            metrics={'contexts':scores,'original99':initial,
                'per_leg_original99':per_leg,
                'legacy_frozen_output_metrics':legacy,
                'solver_float32_max_edge_abs':float(np.max(np.abs(
                    fixture['base_fit_weights']*np.exp(delta)-actual_values))),
                'solver_float32_max_gain_abs':float(np.max(np.abs(gain-actual_gain))),
                'legacy_common_count':int(fixture['common_mask_legacy'].sum()),
                'legacy_fixed_count':int(fixture['fixed_mask'].sum()),
                'preflight':checks,'leg_summaries':leg_summaries}
            write_json(out/'metrics.json',metrics)
            screen=bool(legacy['robust_local_supported'] and
                        scores[2]['ratios']['supported'] and
                        all(x['inner_pass'] and x['original_pass'] for x in per_leg))
            receipt={'status':'valid_conditional_joint_candidate',
                'numerical_status':'conditional_screens_pass' if screen else
                    'conditional_screens_reject',
                'technical_status':'complete_bounded_fit',
                'conditional_screen_pass':screen,
                'parent_independent_reference_pending':True,
                'candidate_adopted':False,'F6_exposed':False,
                'whole_goal_complete':False,'actual_neural_s':0.,
                'actual_physical_s':0.,'leg_summaries':leg_summaries}
        receipt['wall_s']=time.perf_counter()-began
        receipt['fit_wall_s']=ledger.get('fit_wall_s')
        receipt['resource_limit_met']=bool(receipt['wall_s']<INTERNAL_LIMIT_S and
            ledger['fit_wall_s']<=FIT_LIMIT_S and
            all(x['wall_s']<=LEG_LIMIT_S for x in leg_summaries) and
            ledger['leg_unique_started']<=600 and
            ledger['leg_context_ticks_requested']<=72000000 and
            ledger['derivative_banks_completed']<=600 and
            ledger['full_response_batches_requested']<=4 and
            ledger['full_requested_context_ticks']<=480000 and
            1212.+receipt['wall_s']+50.<=1530.)
        if not receipt['resource_limit_met']:
            receipt['technical_status']='resource_limit_incomplete'
            receipt['conditional_screen_pass']=False
        write_json(out/'result.json',receipt)
    except Exception as error:
        primary=error
        try:
            write_json(out/'execution-error.json',{'primary':{
                'type':type(error).__name__,'message':str(error),
                'traceback':traceback.format_exc()},
                'active_leg':ledger.get('active_leg'),
                'active_vector':ledger.get('active_vector'),
                'active_full_progress':ledger.get('active_full_progress'),
                'actual_neural_s':0.,'actual_physical_s':0.})
        except Exception as save_error:
            secondary.append({'stage':'execution-error save','error':repr(save_error)})
    finally:
        if active_leg_engine is not None:
            try:
                ledger['active_leg_engine_metadata']=active_leg_engine.metadata()
            except Exception as error:
                secondary.append({'stage':'active leg metadata','error':repr(error)})
        if full_engine is not None:
            try:
                ledger['full_engine_metadata']=full_engine.metadata()
            except Exception as error:
                secondary.append({'stage':'engine metadata','error':repr(error)})
        ledger['wall_s']=time.perf_counter()-began
        ledger['conservative_shared_s']=1212.+ledger['wall_s']+50.
        ledger['secondary_errors']=secondary
        try:
            write_json(out/'advance-ledger.json',ledger)
        except Exception as error:
            secondary.append({'stage':'final ledger save','error':repr(error)})
        if limits is not None:
            try:
                limits.restore_original_limits()
            except Exception as error:
                secondary.append({'stage':'CPU thread restore','error':repr(error)})
        if primary is None and secondary:
            primary=RuntimeError('E128 secondary cleanup/persistence failure')
        if primary is not None:
            try:
                write_json(out/'result.json',{'status':'technical_incomplete',
                    'technical_status':'failure','numerical_status':'unassessed',
                    'primary_error':repr(primary),'secondary_errors':secondary,
                    'candidate_adopted':False,'F6_exposed':False,
                    'whole_goal_complete':False,'actual_neural_s':0.,
                    'actual_physical_s':0.,'wall_s':time.perf_counter()-began})
            except Exception as error:
                primary.add_note('Secondary result save: '+repr(error))
    if primary is not None:
        for item in secondary:
            primary.add_note('Secondary '+item['stage']+': '+item['error'])
        raise primary
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--freeze-sha',required=True)
    parser.add_argument('--source-review',type=Path,required=True)
    parser.add_argument('--source-review-sha',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.freeze,args.freeze_sha,args.source_review,
               args.source_review_sha,args.out)
    if result['technical_status']!='complete_bounded_fit':
        sys.exit(1)
