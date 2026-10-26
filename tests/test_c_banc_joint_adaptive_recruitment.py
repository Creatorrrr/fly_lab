"""Small synthetic tests only; original E128 responses are parent-owned."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from tools.banc_m3_dense_adaptive import solve_cuda as old_solve
from tools.banc_m3_joint_adaptive_recruitment import (
    CudaJointAdaptiveEngine,
    activation_cotangent_product,
    activation_parameter_map,
    initial_activation_bank,
    prepare_joint_problem,
)
from tools.fit_c_banc_m3_joint_adaptive_recruitment import (
    CONSTRAINT_ROWS,
    EvaluationCache,
    FitLimit,
    choose_records,
    constraint_values,
    evaluate_leg,
    fit_cached_leg,
    no_joint_result,
    objective_and_jac,
    run_tiny_setup,
    selector_feasible,
    synthetic_tiny_setup_problem,
)


def tiny_problem(*, stages=160):
    c,n,m,s=3,2,2,2
    stage=np.empty((c,stages,n),np.float32)
    source=np.empty((c,stages,s),np.float32)
    for ci in range(c):
        for j in range(stages):
            source[ci,j]=[.3+.03*ci+.1*math.sin(j*.07),
                          .2+.02*ci+.06*math.cos(j*.11)]
            stage[ci,j]=[11.+.2*ci+.1*math.sin(j*.09),
                         14.+.3*ci+.1*math.cos(j*.04)]
    return {'stage_current':stage,'source_rate':source,
        'time_s':np.arange(stages//80+1)*.005,
        'initial_rate':np.zeros((c,n)),
        'initial_adaptation':np.zeros((c,n)),
        'initial_activation':np.zeros((c,m)),
        'tau':np.array([.02,.035]),'a':np.array([1.,1.2]),
        'threshold':np.array([7.5,7.]),'cap':np.array([200.,150.]),
        'pool':np.array([[.7,0.],[0.,1.3]]),
        'moment':np.arange(84,dtype=float).reshape(42,2)*.001,
        'rest':np.linspace(-.1,.1,42),'ltm':np.array([0,1]),
        'membership':np.array([0,1]),
        'motor_ids':np.array([100,101]),'source_ids':np.array([7,8]),
        'union_edge_ids':np.array([17,18,19]),
        'union_rows':np.array([0,1,1]),
        'union_cols':np.array([0,1,0]),
        'union_values':np.array([[2.,-3.,.4],[2.5,-2.5,.5],
                                  [3.,-2.,.6]],np.float32),
        'fit_union_positions':np.array([0,1]),
        'edge_ids':np.array([17,18]),
        'edge_rows':np.array([0,1]),'edge_cols':np.array([0,1]),
        'base_fit_weights':np.array([3.,-2.])}


def _cuda():
    cp=pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount()<1:
        pytest.skip('Actual CUDA required')


def test_validation_before_large_copy_and_mapping():
    fixture=tiny_problem()
    p=prepare_joint_problem(fixture)
    assert p['union_ptr'].tolist()==[0,1,3]
    assert p['union_fit'].tolist()==[0,1,-1]
    assert p['edge_muscles'].tolist()==[0,1]
    assert p['union_diff'][2].tolist()==[0.,0.,0.]
    expected=fixture['stage_current'][0,0,1].astype(float)+sum(
        float(fixture['source_rate'][0,0,col])*(
            float(fixture['union_values'][2,j])-float(fixture['union_values'][0,j]))
        for j,col in ((1,1),(2,0)))
    assert expected==pytest.approx(float(fixture['stage_current'][0,0,1])+
        np.dot(fixture['source_rate'][0,0,[1,0]],p['union_diff'][0,1:]))
    with pytest.raises(ValueError):
        prepare_joint_problem({**fixture,'stage_current':
            np.empty((4,160,2),np.float32)})
    with pytest.raises(ValueError):
        prepare_joint_problem({**fixture,'stage_current':
            np.empty((3,160001,2),np.float32)})
    with pytest.raises(ValueError):
        prepare_joint_problem({**fixture,'source_rate':
            np.empty((3,160,70),np.float32)})
    with pytest.raises(ValueError):
        prepare_joint_problem({**fixture,'fit_union_positions':np.array([0,0])})


def test_prepared_arrays_are_owned_and_invalid_states_reject():
    f=tiny_problem()
    p=prepare_joint_problem(f)
    current=float(p['stage_current'][0,0,0])
    source=float(p['source_rate'][0,0,0])
    rest=float(p['rest'][0])
    time0=float(p['time_s'][1])
    f['stage_current'][0,0,0]=0
    f['source_rate'][0,0,0]=0
    f['rest'][0]=9
    f['time_s'][1]=9
    assert p['stage_current'][0,0,0]==current
    assert p['source_rate'][0,0,0]==source
    assert p['rest'][0]==rest
    assert p['time_s'][1]==time0
    for key,value in (
        ('initial_rate',np.full((3,2),-.1)),
        ('initial_adaptation',np.full((3,2),-.1)),
        ('initial_activation',np.full((3,2),-.1)),
        ('source_rate',np.full((3,160,2),-.1,np.float32)),
        ('motor_ids',np.array([100,100])),
        ('source_ids',np.array([7,7])),
        ('tau',np.array(['bad','input']))):
        bad=tiny_problem()
        bad[key]=value
        with pytest.raises(ValueError):
            prepare_joint_problem(bad)


def test_cuda_native_parity_and_exact_e121_self():
    _cuda()
    f=tiny_problem()
    engine=CudaJointAdaptiveEngine(f)
    gain=np.array([1.25,2.1])
    delta=np.zeros(2)
    _,native,timing=engine.solve(delta,gain,native_mode=True)
    legacy={k:f[k] for k in ('stage_current','time_s','initial_rate',
        'initial_adaptation','initial_activation','tau','a','threshold','cap',
        'pool','moment','rest','ltm','membership','motor_ids')}
    older,_=old_solve(legacy,gain)
    for key in ('rate','adaptation','rate_sensitivity',
                'adaptation_sensitivity','activation',
                'activation_sensitivity','command','tendon'):
        np.testing.assert_allclose(native[key],older[key],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(native['branch_bits'],older['branch_bits'])
    _,candidate,_=engine.solve(delta,gain)
    for key in native:
        np.testing.assert_allclose(candidate[key][2],native[key][2],rtol=0,atol=1e-12)
    assert timing['context_rk4_ticks']==3*40
    assert engine.metadata()['fixed_upload']['synchronized']['bytes']>0
    assert engine.metadata()['dynamic_upload']['synchronized']['count']==4


def test_edge_and_gain_bank_finite_difference_with_nonuniform_parameters():
    _cuda()
    f=tiny_problem(stages=800)
    engine=CudaJointAdaptiveEngine(f)
    delta=np.array([.03,-.025])
    gain=np.array([1.1,2.3])
    d,_,_=engine.solve(delta,gain,derivative=True,host=False)
    import cupy as cp
    with engine.device,engine.stream:
        B=cp.asnumpy(d['B'],stream=engine.stream)
    assert activation_parameter_map(engine.problem).tolist()==[0,1,0,1]
    h=1e-5
    for j in range(4):
        dp,dm=delta.copy(),delta.copy()
        gp,gm=gain.copy(),gain.copy()
        if j<2:
            dp[j]+=h
            dm[j]-=h
        else:
            gp[j-2]+=h
            gm[j-2]-=h
        _,plus,_=engine.solve(dp,gp)
        _,minus,_=engine.solve(dm,gm)
        muscle=j if j<2 else j-2
        fd=(plus['activation'][:,:,muscle]-minus['activation'][:,:,muscle])/(2*h)
        np.testing.assert_allclose(B[:,:,j],fd,rtol=2e-4,atol=3e-8)
    initial=initial_activation_bank(engine,d,np.array([0,1]))
    np.testing.assert_array_equal(initial,B[:,:2])
    cotangent=np.arange(B.shape[0]*B.shape[1]*2,dtype=float).reshape(
        B.shape[0],B.shape[1],2)*.001
    grad=activation_cotangent_product(engine,d,cotangent)
    expected=np.stack([np.sum(B[:,:,j]*cotangent[:,:,m],axis=1)
        for j,m in enumerate([0,1,0,1])],axis=1)
    np.testing.assert_allclose(grad,expected,rtol=0,atol=1e-10)


def test_three_context_4761_constraint_jacobian_and_selection():
    rng=np.random.default_rng(8)
    p=5
    losses=np.array([.4,.5,.6])
    loss_jac=rng.normal(size=(3,p))*.01
    q=rng.normal(size=(3,99,7))*.001
    z=rng.normal(size=(3,99))*.001
    jq=rng.normal(size=(3,99,7,p))*.0002
    jz=rng.normal(size=(3,99,p))*.0001
    values,jac,ratios=constraint_values(losses,loss_jac,q,z,jq,jz,.7)
    assert values.shape==(CONSTRAINT_ROWS,)
    assert jac.shape==(CONSTRAINT_ROWS,p+1)
    assert len(values)==3+4158+594+3+3
    np.testing.assert_allclose(jac[:3,-1],1.)
    np.testing.assert_allclose(values[:3],.7-losses)
    assert selector_feasible(ratios)
    # Directional finite difference of all nonlinear signed/RMS rows.
    h=1e-6
    direction=rng.normal(size=p)
    for sign,store in ((1,'plus'),(-1,'minus')):
        shifted=constraint_values(losses+sign*h*(loss_jac@direction),
            loss_jac,q+sign*h*np.einsum('ctjp,p->ctj',jq,direction),
            z+sign*h*np.einsum('ctp,p->ct',jz,direction),jq,jz,.7)[0]
        if store=='plus':
            plus=shifted
        else:
            minus=shifted
    np.testing.assert_allclose((plus-minus)/(2*h),jac[:,:p]@direction,
                               rtol=1e-6,atol=1e-8)
    records=[{'true_worst_objective':.9,'ratios':np.ones((3,4))*.99},
             {'true_worst_objective':.8,'ratios':np.ones((3,4))*1.1}]
    any_best,feasible=choose_records(records)
    assert any_best is records[1] and feasible is records[0]
    assert not selector_feasible(records[1]['ratios'])


def test_cache_ignores_epigraph_and_preserves_limit():
    seen=[]
    def evaluate(x,i):
        seen.append((i,x.copy()))
        return {'delta':x[:1], 'gain':x[1:], 'losses':np.zeros(3)}
    cache=EvaluationCache(evaluate,2,lambda i,x:None,max_calls=1,
        deadline=time.perf_counter()+20,total_deadline=time.perf_counter()+20)
    first,_=cache.get(np.array([.1,2.,.7]))
    second,t=cache.get(np.array([.1,2.,.9]))
    assert first is second and t==.9 and cache.hits==1 and len(seen)==1
    with pytest.raises(FitLimit):
        cache.get(np.array([.2,2.,.9]))
    objective,jac,prior=objective_and_jac({
        'delta':np.array([.1]),'gain':np.array([2.])},.7,1)
    assert objective==pytest.approx(.7+prior)
    assert jac.shape==(3,) and jac[-1]==1.


def test_no_feasible_and_prestart_budget_are_total():
    all_none=no_joint_result([None]*6,[{'leg':x} for x in range(6)])
    assert all_none['unfeasible_legs']==['lf','lm','lh','rf','rm','rh']
    assert all_none['technical_status']=='complete_bounded_fit'
    partial=no_joint_result([None],[{'leg':'lf'}])
    assert partial['unfeasible_legs']==['lf','lm','lh','rf','rm','rh']
    assert partial['technical_status']=='fit_budget_exhausted'
    calls=[]
    cache=EvaluationCache(lambda x,i:calls.append(i),2,
        lambda i,x:calls.append(('started',i)),max_calls=100,
        deadline=time.perf_counter(),total_deadline=time.perf_counter()+20)
    optimizer,stop=fit_cached_leg(cache,[(0,1),(0,8)],2)
    assert optimizer is None and stop=='per_leg_unique_or_wall_limit'
    assert cache.started==0 and calls==[]


def test_tiny_setup_uses_explicit_synthetic_arrays_and_saves_before_gate(tmp_path):
    _cuda()
    f=synthetic_tiny_setup_problem()
    assert f['stage_current'].shape==(1,80,1)
    assert f['source_rate'].shape==(1,80,1)
    assert not np.all(f['stage_current']==f['stage_current'][0,0,0])
    ledger={}
    setup=run_tiny_setup(tmp_path,ledger)
    assert setup['status']=='completed'
    assert setup['response_progress']['completed_context_rk4_ticks']==20
    assert setup['response_progress']['completed_bank_edge_stage_ticks']==80
    assert setup['raw_saved']
    with np.load(tmp_path/'cuda-tiny-setup-raw.npz',allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw['synthetic_stage_current'],
                                      f['stage_current'])
        assert raw['branch_bits'].shape==(1,20,1)


def test_small_cuda_loss_gradient_and_raw_before_checks(tmp_path):
    _cuda()
    import torch
    f=tiny_problem(stages=8000)
    samples=101
    f['fit_time_mask']=np.zeros((3,samples),bool)
    f['fit_time_mask'][:,1:]=True
    f['startup_mask']=np.zeros(samples,bool)
    f['startup_mask'][1:100]=True
    f['own_target']=np.broadcast_to(f['rest'],(3,samples,42)).copy()
    f['target_feet']=np.zeros((3,samples,6,3))
    f['startup_target_command']=np.broadcast_to(f['rest'],(99,42)).copy()
    f['startup_target_tendon']=np.zeros((99,6))
    f['joint_scale']=np.ones(42)
    f['foot_scale']=np.ones((6,3))
    f['velocity_scale']=np.ones((6,3))
    class Geometry:
        def leg(self,command,leg):
            assert leg==0
            return torch.stack((command[:,0],command[:,1],command[:,2]),dim=1)
    engine=CudaJointAdaptiveEngine(f,np.array([0]),np.array([0]),np.array([0]))
    vector=np.array([.02,1.1])
    record=evaluate_leg(engine,f,Geometry(),0,vector,1,tmp_path)
    assert (tmp_path/'leg-0-evaluation-001-raw.npz').exists()
    assert record['loss_jac'].shape==(3,2)
    metadata=engine.metadata()
    assert metadata['geometry_cost']['forward_rows_completed']==300
    assert metadata['geometry_cost']['backward_rows_completed']==300
    assert metadata['cotangent_upload']['requested']['bytes']==0
    assert metadata['torch_cupy_shared_bytes']>0
    assert metadata['host_transfer_by_label']['loss_gradient'][
        'synchronized']['bytes']>0
    assert metadata['torch_host_transfer_by_label']['loss_cotangent_raw'][
        'synchronized']['bytes']>0
    h=1e-5
    for j in range(2):
        plus=vector.copy()
        plus[j]+=h
        minus=vector.copy()
        minus[j]-=h
        rplus=evaluate_leg(engine,f,Geometry(),0,plus,2+j*2,tmp_path)
        rminus=evaluate_leg(engine,f,Geometry(),0,minus,3+j*2,tmp_path)
        np.testing.assert_allclose((rplus['losses']-rminus['losses'])/(2*h),
                                   record['loss_jac'][:,j],rtol=5e-4,atol=1e-7)
        np.testing.assert_allclose((rplus['q_error']-rminus['q_error'])/(2*h),
                                   record['q_jac'][:,:,:,j],rtol=5e-4,atol=1e-7)
    with np.load(tmp_path/'leg-0-evaluation-001-raw.npz') as raw:
        assert raw['B_initial99'].shape==(3,99,2)
        assert raw['branch_bits'].shape==(3,2000,1)


def test_permuted_nonzero_global_muscles_q_ltm_and_loss_fd(tmp_path):
    _cuda()
    import torch
    f=tiny_problem(stages=8000)
    samples=101
    f['fit_time_mask']=np.zeros((3,samples),bool)
    f['fit_time_mask'][:,1:]=True
    f['startup_mask']=np.zeros(samples,bool)
    f['startup_mask'][1:100]=True
    f['own_target']=np.broadcast_to(f['rest'],(3,samples,42)).copy()
    f['target_feet']=np.zeros((3,samples,6,3))
    f['startup_target_command']=np.broadcast_to(f['rest'],(99,42)).copy()
    f['startup_target_tendon']=np.zeros((99,6))
    f['joint_scale']=np.ones(42)
    f['foot_scale']=np.ones((6,3))
    f['velocity_scale']=np.ones((6,3))
    class Geometry:
        def leg(self,command,leg):
            return torch.stack((command[:,0],command[:,1],command[:,2]),dim=1)
    # Local muscle0 is global muscle1; local muscle1 is global muscle0.
    engine=CudaJointAdaptiveEngine(f,np.array([1,0]),np.array([1,0]),
                                   np.array([1,0]))
    np.testing.assert_array_equal(engine.problem['muscle_indices'],[1,0])
    np.testing.assert_array_equal(engine.problem['edge_muscles'],[0,1])
    x=np.array([-.02,.03,2.2,1.3])
    center=evaluate_leg(engine,f,Geometry(),0,x,1,tmp_path)
    h=1e-5
    for j in range(4):
        plus=x.copy();plus[j]+=h
        minus=x.copy();minus[j]-=h
        positive=evaluate_leg(engine,f,Geometry(),0,plus,2+2*j,tmp_path)
        negative=evaluate_leg(engine,f,Geometry(),0,minus,3+2*j,tmp_path)
        scale=1/(2*h)
        np.testing.assert_allclose((positive['q_error']-negative['q_error'])*scale,
            center['q_jac'][:,:,:,j],rtol=5e-4,atol=1e-7)
        np.testing.assert_allclose((positive['ltm_error']-negative['ltm_error'])*scale,
            center['ltm_jac'][:,:,j],rtol=5e-4,atol=1e-7)
        np.testing.assert_allclose((positive['losses']-negative['losses'])*scale,
            center['loss_jac'][:,j],rtol=5e-4,atol=1e-7)


def test_legacy_float32_export_maps_ids_and_preserves_other_fields():
    from tools.fit_c_banc_m3_dense_foot_recruitment import selected_actual_float32
    rows=np.arange(1572)//49
    cols=np.arange(1572)%49
    original=np.zeros((391,49))
    signed=np.where(np.arange(1572)%2==0,.125,-.25).astype(np.float32)
    original[rows,cols]=signed
    current=original.copy()
    ids=np.arange(1572,dtype=np.int64)+1000
    train={'edge_ids':ids,'edge_rows':rows,'edge_cols':cols,
        'original_signed_edge_weights':signed,'original_w':original,
        'current_w':current}
    cal={'edge_ids':ids[::-1],
         'values':signed[::-1].copy(),
         'tau_ids':np.array([2,5]),'tau_values':np.array([.02,.03]),
         'other_field':np.array([8.,9.])}
    efficacy=np.linspace(.8,1.2,1572)
    dense,updated=selected_actual_float32(train,cal,efficacy)
    np.testing.assert_array_equal(dense[rows,cols],
                                  np.float32(signed*efficacy).astype(np.float64))
    np.testing.assert_array_equal(updated['values'][::-1],
                                  np.float32(signed*efficacy))
    np.testing.assert_array_equal(updated['other_field'],cal['other_field'])
    np.testing.assert_array_equal(updated['tau_values'],cal['tau_values'])
    assert updated['values'].dtype==np.float32


def test_cuda_failure_marks_unknown_and_preserves_primary(monkeypatch,tmp_path):
    _cuda()
    import tools.banc_m3_joint_adaptive_recruitment as joint
    engine=joint.CudaJointAdaptiveEngine(tiny_problem())
    def fail(name):
        def launch(*args,**kwargs):
            raise RuntimeError('synthetic launch failure')
        return launch
    monkeypatch.setattr(joint,'_kernel',fail)
    with pytest.raises(RuntimeError,match='synthetic launch failure') as caught:
        engine.solve(np.zeros(2),np.zeros(2))
    progress=caught.value.response_progress
    assert progress['completion_class']=='unknown'
    assert progress['requested_context_rk4_ticks']==3*40
    assert progress['completed_context_rk4_ticks'] is None
    assert progress['dynamic_upload_synchronized']['count']==2
    assert engine.metadata()['dynamic_upload']['unknown_requested']['count']==0
    with pytest.raises(RuntimeError,match='synthetic launch failure'):
        evaluate_leg(engine,tiny_problem(),None,0,np.zeros(4),1,tmp_path)
    partial=tmp_path/'leg-0-evaluation-001-partial-raw.npz'
    assert partial.exists()
    with np.load(partial) as raw:
        assert raw['delta'].shape==(2,)
        assert raw['gain'].shape==(2,)
        np.testing.assert_array_equal(raw['edge_ids'],[17,18])


def test_second_dynamic_copy_failure_preserves_individual_costs(monkeypatch):
    _cuda()
    import cupy as cp
    engine=CudaJointAdaptiveEngine(tiny_problem())
    actual=cp.asarray
    count=0
    def fail_second(value,*args,**kwargs):
        nonlocal count
        count+=1
        if count==2:
            raise RuntimeError('synthetic second upload failure')
        return actual(value,*args,**kwargs)
    monkeypatch.setattr(cp,'asarray',fail_second)
    with pytest.raises(RuntimeError,match='synthetic second upload failure') as caught:
        engine.solve(np.zeros(2),np.zeros(2))
    progress=caught.value.response_progress
    assert progress['dynamic_upload_requested']['count']==2
    assert progress['dynamic_upload_enqueued']['count']==1
    assert progress['dynamic_upload_synchronized']['count']==0
    assert progress['dynamic_upload_events'][0]['enqueued']
    assert not progress['dynamic_upload_events'][1]['enqueued']
    by_name=engine.metadata()['dynamic_upload_by_name']
    assert by_name['delta']['enqueued']['count']==1
    assert by_name['gain']['enqueued']['count']==0
    assert by_name['delta']['unknown_requested']['count']==1
    assert by_name['gain']['unknown_requested']['count']==1


def test_post_forward_decoder_failure_persists_full_partial(monkeypatch,tmp_path):
    _cuda()
    import tools.banc_m3_joint_adaptive_recruitment as joint
    from tools.fit_c_banc_m3_joint_adaptive_recruitment import full_response
    engine=joint.CudaJointAdaptiveEngine(tiny_problem())
    actual=joint._kernel
    def fail_decoder(name):
        if name=='muscle_ema':
            raise RuntimeError('synthetic post-forward decoder failure')
        return actual(name)
    monkeypatch.setattr(joint,'_kernel',fail_decoder)
    ledger={'full_response_batches_requested':0,
        'full_requested_context_ticks':0,'full_mn_ticks_requested':0,
        'full_decoder_ticks_requested':0,'full_context_ticks_unknown_requested':0,
        'full_context_ticks_completed':0,'full_conditional_s_completed':0.,
        'full_mn_ticks_completed':0,'full_decoder_ticks_completed':0,
        'full_raw_sha256':{}}
    with pytest.raises(RuntimeError,match='synthetic post-forward decoder failure') as caught:
        full_response(engine,np.zeros(2),np.zeros(2),'synthetic-full',
                      tmp_path,ledger)
    progress=caught.value.response_progress
    assert progress['forward_completion_class']=='known_complete'
    assert progress['decoder_completion_class']=='unknown'
    assert progress['bank_completion_class']=='not_requested'
    assert ledger['full_context_ticks_completed']==120
    path=tmp_path/'synthetic-full-partial-raw.npz'
    assert path.exists()
    with np.load(path,allow_pickle=False) as raw:
        for name in raw.files:
            assert raw[name].dtype != object
        assert raw['rate'].shape==(3,3,2)
        assert raw['branch_bits'].shape==(3,40,2)


def test_post_bank_launch_failure_preserves_initial_bank(monkeypatch):
    _cuda()
    import tools.banc_m3_joint_adaptive_recruitment as joint
    engine=joint.CudaJointAdaptiveEngine(tiny_problem(stages=8000))
    actual=joint._kernel
    def after_enqueue(name):
        kernel=actual(name)
        if name!='edge_bank':
            return kernel
        def launch(*args,**kwargs):
            kernel(*args,**kwargs)
            raise RuntimeError('synthetic post-bank-enqueue failure')
        return launch
    monkeypatch.setattr(joint,'_kernel',after_enqueue)
    with pytest.raises(RuntimeError,match='synthetic post-bank-enqueue failure') as caught:
        engine.solve(np.zeros(2),np.ones(2),derivative=True,host=False)
    progress=caught.value.response_progress
    assert progress['forward_completion_class']=='known_complete'
    assert progress['decoder_completion_class']=='known_complete'
    assert progress['bank_completion_class']=='unknown'
    assert progress['completed_bank_edge_stage_ticks'] is None
    partial=caught.value.partial_response
    assert partial['B_initial99'].shape==(3,99,4)
    assert partial['rate'].shape==(3,101,2)
    assert partial['branch_bits'].shape==(3,2000,2)
    assert all(value.dtype!=object for value in partial.values())


def test_post_bank_geometry_failure_rescues_full_and_compact(monkeypatch,tmp_path):
    _cuda()
    f=tiny_problem(stages=8000)
    f['fit_time_mask']=np.zeros((3,101),bool)
    f['fit_time_mask'][:,1:]=True
    f['startup_mask']=np.zeros(101,bool)
    f['startup_mask'][1:100]=True
    f['startup_target_command']=np.zeros((99,42))
    f['startup_target_tendon']=np.zeros((99,6))
    class BrokenGeometry:
        def leg(self,commands,leg):
            raise RuntimeError('synthetic geometry failure after bank')
    engine=CudaJointAdaptiveEngine(f,np.array([0]),np.array([0]),
                                   np.array([0]))
    with pytest.raises(RuntimeError,match='synthetic geometry failure after bank') as caught:
        evaluate_leg(engine,f,BrokenGeometry(),0,np.array([.02,1.]),1,tmp_path)
    assert caught.value.response_progress['bank_completion_class']=='known_complete'
    cost=engine.metadata()['geometry_cost']
    assert cost['forward_rows_requested']==100
    assert cost['forward_rows_enqueued']==0
    assert cost['forward_rows_unknown_requested']==100
    path=tmp_path/'leg-0-evaluation-001-partial-raw.npz'
    assert path.exists()
    with np.load(path,allow_pickle=False) as raw:
        for name in raw.files:
            assert raw[name].dtype!=object
        assert raw['rate'].shape==(3,101,1)
        assert raw['B_initial99'].shape==(3,99,2)


def test_tiny_setup_bank_failure_keeps_setup_cost_and_partial(monkeypatch,tmp_path):
    _cuda()
    import tools.banc_m3_joint_adaptive_recruitment as joint
    actual=joint._kernel
    def fail_bank(name):
        if name=='edge_bank':
            raise RuntimeError('synthetic tiny bank failure')
        return actual(name)
    monkeypatch.setattr(joint,'_kernel',fail_bank)
    ledger={}
    with pytest.raises(RuntimeError,match='synthetic tiny bank failure'):
        run_tiny_setup(tmp_path,ledger)
    setup=ledger['cuda_tiny_setup']
    assert setup['status']=='failed'
    assert setup['requested_context_ticks']==20
    assert setup['response_progress']['forward_completion_class']=='known_complete'
    assert setup['response_progress']['decoder_completion_class']=='known_complete'
    assert setup['response_progress']['bank_completion_class']=='unknown'
    assert setup['partial_raw_saved']
    with np.load(tmp_path/'cuda-tiny-setup-partial-raw.npz',
                 allow_pickle=False) as raw:
        assert raw['synthetic_stage_current'].shape==(1,80,1)
        assert raw['B_initial99'].dtype!=object


def test_backward_failure_retains_completed_forward_and_unknown_backward(
        monkeypatch,tmp_path):
    _cuda()
    import torch
    f=tiny_problem(stages=8000)
    f['fit_time_mask']=np.zeros((3,101),bool)
    f['fit_time_mask'][:,1:]=True
    f['startup_mask']=np.zeros(101,bool)
    f['startup_mask'][1:100]=True
    f['startup_target_command']=np.zeros((99,42))
    f['startup_target_tendon']=np.zeros((99,6))
    f['own_target']=np.zeros((3,101,42))
    f['target_feet']=np.zeros((3,101,6,3))
    f['joint_scale']=np.ones(42)
    f['foot_scale']=np.ones((6,3))
    f['velocity_scale']=np.ones((6,3))
    class Geometry:
        def leg(self,command,leg):
            return torch.stack((command[:,0],command[:,1],command[:,2]),dim=1)
    def fail_backward(*args,**kwargs):
        raise RuntimeError('synthetic backward failure')
    monkeypatch.setattr(torch.autograd,'grad',fail_backward)
    engine=CudaJointAdaptiveEngine(f,np.array([0]),np.array([0]),
                                   np.array([0]))
    with pytest.raises(RuntimeError,match='synthetic backward failure'):
        evaluate_leg(engine,f,Geometry(),0,np.array([.01,1.]),1,tmp_path)
    cost=engine.metadata()['geometry_cost']
    assert cost['forward_rows_requested']==300
    assert cost['forward_rows_completed']==300
    assert cost['backward_rows_requested']==300
    assert cost['backward_rows_completed']==0
    assert cost['backward_rows_unknown_requested']==300
    with np.load(tmp_path/'leg-0-evaluation-001-partial-raw.npz',
                 allow_pickle=False) as raw:
        assert raw['B_initial99'].shape==(3,99,2)
        assert raw['rate'].shape==(3,101,1)


def test_partial_full_geometry_transfer_cost():
    _cuda()
    import torch

    from tools.fit_c_banc_m3_joint_adaptive_recruitment import full_feet
    class Geometry:
        device=torch.device('cuda')
        calls=0
        def full(self,command):
            self.calls+=1
            if self.calls==2:
                raise RuntimeError('synthetic second geometry failure')
            return command.new_zeros((len(command),6,3))
    ledger={}
    command=np.zeros((3,5,42))
    with pytest.raises(RuntimeError,match='synthetic second geometry failure'):
        full_feet(Geometry(),{'command':command},ledger,'synthetic_partial')
    cost=ledger['full_geometry'][0]
    assert cost['rows_requested']==10
    assert cost['rows_completed']==5
    assert cost['rows_unknown_requested']==5
    assert cost['command_h2d_bytes_requested']==2*command[0].nbytes
    assert cost['command_h2d_bytes_unknown_requested']==0
    assert cost['foot_d2h_bytes_completed']==5*6*3*8
    assert cost['foot_d2h_bytes_unknown_requested']==0


def test_constructor_failure_keeps_partial_fixed_upload(monkeypatch,tmp_path):
    _cuda()
    import cupy as cp

    from tools.fit_c_banc_m3_joint_adaptive_recruitment import construct_engine
    actual=cp.asarray
    count=0
    def fail_second(value,*args,**kwargs):
        nonlocal count
        count+=1
        if count==2:
            raise RuntimeError('synthetic fixed upload failure')
        return actual(value,*args,**kwargs)
    monkeypatch.setattr(cp,'asarray',fail_second)
    ledger={}
    with pytest.raises(RuntimeError,match='synthetic fixed upload failure'):
        construct_engine(tiny_problem(),ledger,tmp_path,'synthetic_constructor')
    event=ledger['constructor_failures'][0]
    assert event['fixed_upload_progress']['requested']['count']==2
    assert event['fixed_upload_progress']['enqueued']['count']==1
    assert event['fixed_upload_progress']['unknown_requested']['count']==2
    assert event['fixed_upload_by_key']['stage_current'][
        'enqueued']['count']==1
    assert (tmp_path/'advance-ledger.json').exists()


def test_review_gate_rejects_unreviewed_freeze(tmp_path):
    from tools.fit_c_banc_m3_joint_adaptive_recruitment import reviewed_inputs
    frozen=tmp_path/'freeze.json'
    review=tmp_path/'review.json'
    frozen.write_text('{}')
    review.write_text('{}')
    with pytest.raises(ValueError,match='freeze SHA mismatch'):
        reviewed_inputs(frozen,'0'*64,review,'0'*64)


def test_nonfinite_compact_raw_also_retains_full_failure_response(tmp_path):
    _cuda()
    import torch

    f=tiny_problem(stages=8000)
    f['fit_time_mask']=np.zeros((3,101),bool)
    f['fit_time_mask'][:,1:]=True
    f['startup_mask']=np.zeros(101,bool)
    f['startup_mask'][1:100]=True
    f['startup_target_command']=np.zeros((99,42))
    f['startup_target_tendon']=np.zeros((99,6))
    f['own_target']=np.full((3,101,42),np.nan)
    f['target_feet']=np.zeros((3,101,6,3))
    f['joint_scale']=np.ones(42)
    f['foot_scale']=np.ones((6,3))
    f['velocity_scale']=np.ones((6,3))

    class Geometry:
        def leg(self,command,leg):
            return torch.stack((command[:,0],command[:,1],command[:,2]),dim=1)

    engine=CudaJointAdaptiveEngine(f,np.array([0]),np.array([0]),np.array([0]))
    with pytest.raises(ValueError,match='Nonfinite E128 leg raw saved'):
        evaluate_leg(engine,f,Geometry(),0,np.array([.01,1.]),2,tmp_path)
    assert (tmp_path/'leg-0-evaluation-002-raw.npz').exists()
    with np.load(tmp_path/'leg-0-evaluation-002-partial-raw.npz',
                 allow_pickle=False) as raw:
        assert raw['rate'].shape==(3,101,1)
        assert np.isfinite(raw['rate']).all()
        assert raw['B_initial99'].shape==(3,99,2)
        assert raw['failure_type'].item()=='ValueError'
    assert engine.solve_calls==1


def test_index_uploads_are_explicit_owned_and_cached():
    _cuda()
    import torch

    from tools.fit_c_banc_m3_joint_adaptive_recruitment import torch_indices

    engine=CudaJointAdaptiveEngine(tiny_problem())
    host=np.array([1,0],np.int64)
    first=engine.device_indices(host,'synthetic_index')
    again=engine.device_indices(host,'synthetic_index')
    assert first is again
    assert engine.index_upload['synchronized']=={'count':1,'bytes':16}
    assert engine.index_cache_hits==1
    device=torch.device('cuda:0')
    tensor=torch_indices(engine,host,device,'synthetic_torch_index')
    assert tensor is torch_indices(engine,host,device,'synthetic_torch_index')
    assert tensor.dtype==torch.int64
    assert engine.torch_index_cache_hits==1
    assert engine.torch_input_transfer['synchronized']=={'count':1,'bytes':16}
    host[:]=0
    np.testing.assert_array_equal(first.get(),[1,0])
    np.testing.assert_array_equal(tensor.cpu().numpy(),[1,0])
    assert engine.solve_calls==0
