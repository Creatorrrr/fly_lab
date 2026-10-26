"""Current complete E359 generated boundaries: no original arrays/GPU/body."""
import copy
import json
import os
import sys
import time
from types import SimpleNamespace

import pytest

from tools import verify_c_banc_m3_startup_baseline as h


def counters99():
    construction={'body_constructors_requested':1,'body_constructors_returned':1,
                  'settling_physics_requested':2000,'settling_physics_host_returned':2000,
                  'neural_setup_requested':320,'neural_setup_host_returned':320,
                  'neural_capture_requested':160,'neural_capture_recorded':160,
                  'unknown_work':False,'device_confirmed':True}
    counters={'controls_requested':99,'controls_host_returned':99,'controls_completed':99,
              'neural_requested':1980,'neural_host_returned':1980,'neural_device_confirmed':1980,
              'physics_requested':4950,'physics_host_returned':4950,'physics_device_confirmed':4950,
              'observer_samples':4950,'completed_rows':100,'neural_unreturned_unknown':False,
              'physics_unreturned_unknown':False,'advance_unreturned_unknown':False}
    return counters,construction


def test_real_bounded_generated_fixture_closed_publisher():
    spec=json.loads(os.environ['E359_CURRENT_FIXTURE_EXIT'])
    closed=h.closed_exit(spec)
    assert closed['mode']=='generated' and closed['role']=='fixture' and closed['original_calls']==0
    c=h.strict_json(h.read_spec(closed['contract']))
    receipt=h.strict_json(h.safe(c['paths']['pre_spawn']).read_bytes())
    value=h.strict_json(h.read_spec(closed['envelope']))
    outer=h.strict_json(h.read_spec(closed['outer']))
    assert c['limits']['outer_s']==6 and receipt['authority']==c
    assert receipt['original_array_reads']==receipt['scientific_native_imports']==receipt['Popen_before_receipt']==0
    assert outer['observed_pid_create_times'] and outer['all_owned_ended'] and outer['natural_closed']
    assert value['original_array_reads']==0 and not value['natural_gait'] and not value['F1_F8_complete']
    fixture=h.strict_json(h.safe(c['paths']['output']+'/fixture.json').read_bytes())
    assert fixture=={'value':[1,2,3],'original_array_reads':0}


def test_exact_pre_array_receipt_authority_types(tmp_path,monkeypatch):
    # A generated metadata-only live fake parent reaches every receipt field.
    c={'role':'fixture','attempt_id':'fake','attempt':1,'mode':'generated','source_sha256':{},
       'cumulative_before_s':99.7460156,'cumulative_limit_s':150,'reserved_s':6,'pending_reservations':{'suite':12},
       'limits':{'outer_s':6,'science_s':3},'stage_used_s':19.556,'stage_limit_s':69.8099844,
       'original_calls':0,'max_tree_rss_bytes':2<<30,'retained_reserve_bytes':1<<20,'cpu_threads':1,
       'paths':{'pre_spawn':'receipt.json'}}
    spec={'path':'contract.json','bytes':1,'sha256':'a'*64}
    owner=SimpleNamespace(pid=123,create_time=lambda:456.)
    monkeypatch.setitem(sys.modules,'psutil',SimpleNamespace(Process=lambda pid:owner))
    monkeypatch.setattr(h,'validate_contract',lambda *a,**kw:(c,spec,{}))
    good={'schema':'flylab.m3.e359.actual-pre-spawn.v1','contract':spec,'authority':c,
          'entry_perf_counter':time.perf_counter(),'parent_pid':123,'parent_create_time':456.,
          'source_sha256':{},'reservation_held':True,'original_array_reads':0,'scientific_native_imports':0,
          'Popen_before_receipt':0,'cumulative_after_s':None}
    for key in ('role','attempt_id','attempt','cumulative_before_s','cumulative_limit_s','reserved_s',
                'pending_reservations','limits','stage_used_s','stage_limit_s','original_calls',
                'max_tree_rss_bytes','retained_reserve_bytes','cpu_threads'):
        good[key]=c[key]
    path=tmp_path/'receipt.json'
    path.write_bytes(h.json_bytes(good))
    assert h.pre_array('contract.json','receipt.json',root=tmp_path)[0]==c
    for key,value in (('attempt',True),('original_array_reads',False),('scientific_native_imports',False),
                      ('Popen_before_receipt',False),('cumulative_limit_s',150.),('reserved_s',6.),
                      ('stage_used_s',False),('parent_pid',True)):
        bad=copy.deepcopy(good);bad[key]=value
        path.write_bytes(h.json_bytes(bad))
        with pytest.raises(ValueError):
            h.pre_array('contract.json','receipt.json',root=tmp_path)
    bad=copy.deepcopy(good);bad['authority']['pending_reservations']={}
    path.write_bytes(h.json_bytes(bad))
    with pytest.raises(ValueError,match='authority'):
        h.pre_array('contract.json','receipt.json',root=tmp_path)


def test_complete_inherited_cost_roster_omission_duplicate_bool_change(tmp_path,monkeypatch):
    reg=h.registration()
    inherited=h.strict_json(h.read_spec(reg['inherited_cost_ledger']))
    original=copy.deepcopy(inherited['ledger'])
    original['e359_reservations']={}
    original.update(e359_original_local_limit_s=85,e359_generated_local_limit_s=25,
                    e359_original_local_used_s=0,e359_generated_local_used_s=0,e359_registration_path=h.REGISTRATION)
    path=tmp_path/'generated-ledger.json'
    real_safe=h.safe
    monkeypatch.setattr(h,'safe',lambda name,root=h.ROOT:path if name==reg['shared_ledger_path'] else real_safe(name,root))
    path.write_bytes(h.json_bytes(original))
    _,_,totals,local=h.ledger_state(reg)
    assert totals=={'original':78.18,'generated':19.556} and local=={'original':0.,'generated':0.}
    variants=[]
    bad=copy.deepcopy(original);bad['debits'].pop();variants.append(bad)
    bad=copy.deepcopy(original);bad['debits'].append(copy.deepcopy(bad['debits'][0]));variants.append(bad)
    bad=copy.deepcopy(original);bad['debits'][0]['actual_outer_full_return_s']=True;variants.append(bad)
    bad=copy.deepcopy(original);bad['debits'][0]['descriptor']['sha256']='0'*64;variants.append(bad)
    bad=copy.deepcopy(original);bad['original_stage_used_s']=True;variants.append(bad)
    bad=copy.deepcopy(original);bad['original_jobs_running']=True;variants.append(bad)
    for bad in variants:
        path.write_bytes(h.json_bytes(bad))
        with pytest.raises(ValueError):
            h.ledger_state(reg)
    held=copy.deepcopy(original)
    held['e359_reservations']={'generated-original-owner:baseline_audit':
                             {'mode':'original','role':'baseline_audit','reserved_s':12,'pending_reservations':{}}}
    held['original_reserved_s']=12
    held['original_jobs_running']=1
    path.write_bytes(h.json_bytes(held))
    assert h.ledger_state(reg)[1]['original_jobs_running']==1
    held['original_jobs_running']=0
    path.write_bytes(h.json_bytes(held))
    with pytest.raises(ValueError,match='held original'):
        h.ledger_state(reg)


def test_original_shared_owner_count_lock_and_PID_identity(tmp_path,monkeypatch):
    owner={'created':456.}
    monkeypatch.setitem(sys.modules,'psutil',SimpleNamespace(Process=lambda pid:SimpleNamespace(create_time=lambda:owner['created'])))
    path=h.safe(h.BASE+'/parent-e358-original-active.lock',tmp_path)
    path.parent.mkdir(parents=True)
    good={'pid':123,'created':456.,'contract':'current.json'}
    path.write_bytes(h.json_bytes(good))
    c={'mode':'original','ledger_semantics':{'original_jobs_running':1}}
    h.shared_owner(c,'current.json',123,456.,root=tmp_path)
    for changed in ({**good,'pid':124},{**good,'created':457.},{**good,'contract':'replayed.json'},
                    {**good,'pid':True}):
        path.write_bytes(h.json_bytes(changed))
        with pytest.raises(ValueError):
            h.shared_owner(c,'current.json',123,456.,root=tmp_path)
    path.write_bytes(h.json_bytes(good))
    owner['created']=457.
    with pytest.raises(ValueError,match='PID/create-time'):
        h.shared_owner(c,'current.json',123,456.,root=tmp_path)
    c['ledger_semantics']['original_jobs_running']=0
    with pytest.raises(ValueError,match='one held'):
        h.shared_owner(c,'current.json',123,456.,root=tmp_path)


def test_exact99_work_pass_fail_and_forbidden_prefix_evaluator():
    counters,construction=counters99()
    assert h.verify_work(counters,construction)
    for startup_passed in (True,False):
        result={'baseline_completed99':True,'startup':{'passed':startup_passed}}
        assert h.verify_work(counters,construction) and result['baseline_completed99']
    for count in (98,100,300,301,True):
        with pytest.raises(ValueError,match='exact99'):
            h.verify_work(counters,construction,count)
    source=h.safe(h.OWNED[0]).read_text('utf8')
    assert 'b.evaluate(' not in source and 'range(1,100)' in source
    counters['physics_unreturned_unknown']=True
    with pytest.raises(ValueError):
        h.verify_work(counters,construction)
    counters,construction=counters99();construction['neural_capture_recorded']=159
    with pytest.raises(ValueError):
        h.verify_work(counters,construction)


def test_actual_generated99_prefix_lifecycle_for_startup_pass_and_fail(monkeypatch):
    import numpy as np

    from tools import verify_c_banc_m3_common_descent_body as body_helpers
    monkeypatch.setattr(h,'monitor_check',lambda *a,**kw:True)
    for passed in (True,False):
        session=SimpleNamespace(control_tick=0,network=SimpleNamespace(tick=0),_cpg_digest=lambda:'same')
        session.body=SimpleNamespace(physics_time=lambda session=session:session.control_tick*.005)
        state={'counters':None,'samples':None,'detached':False}
        class Observer:
            def __init__(self,s,c,samples,state=state,**kw):state.update(counters=c,samples=samples)
            def install(self):return self
            def close(self,state=state):state['detached']=True
        def advance(count,capture,session=session,state=state):
            assert count==1 and capture is False
            session.control_tick+=1;session.network.tick+=20
            c=state['counters']
            for key in ('neural_requested','neural_host_returned','neural_device_confirmed'):c[key]+=20
            for key in ('physics_requested','physics_host_returned'):c[key]+=50
            state['samples'].extend([None]*50)
        session.advance=advance
        def capture(s,passed=passed):
            command=np.zeros(42,np.float64)
            if not passed and s.control_tick==99:command[0]=.041
            return {},{'control_tick':s.control_tick,'command':command,'tendon':np.zeros(6,np.float64)}
        counters,construction=counters99()
        empty={k:False if type(v) is bool else 0 for k,v in counters.items()}
        b=SimpleNamespace(new_counters=lambda empty=empty:copy.deepcopy(empty),trace_metadata=lambda s:{},capture=capture,LiveObserver=Observer)
        factory=SimpleNamespace(synchronize=lambda:None,install_contact=lambda *a:None)
        result={'first_error':None,'secondary_errors':[],'watch':SimpleNamespace(snapshot=dict)}
        timer=SimpleNamespace(check=lambda *a:None)
        controls,_,samples,observed,_=h.run_prefix(session,factory,{}, {'cpg_hash':'same'},
                                                   {'limits':{}},timer,b=b,result=result)
        assert h.verify_work(observed,construction) and session.control_tick==99 and len(controls)==100 and len(samples)==4950
        assert state['detached'] and result['cpg_hashes']==['same']*100
        startup=body_helpers.startup99(np.stack([v['command'] for v in controls]),np.stack([v['tendon'] for v in controls]),
                                      np.zeros((99,42),np.float64),np.zeros((99,6),np.float64))
        assert startup['passed'] is passed


def neutral_fake():
    import numpy as np
    source=SimpleNamespace(data=np.asarray([2.,-3.,4.],np.float32),indices=np.asarray([0,1,2],np.int32),
                           indptr=np.asarray([0,1,2,3],np.int32))
    data={'source_csr':source,'positions':np.asarray([0,1],np.int64),'theta':np.ones(8547,np.float64),
          'inventory':{'source_network_identity':'same'}}
    h.select_live_source(data)
    patch={'original32':data['selected_weights'].copy(),'actual32':data['selected_weights'].copy(),
           'source_network_identity':'same','network_identity':'same'}
    actual=SimpleNamespace(**{k:getattr(source,k).copy() for k in ('data','indices','indptr')})
    return data,patch,actual


def test_neutral_actual_source_bytes_identity_fabricated_selected_reject():
    data,patch,actual=neutral_fake()
    assert h.neutral_values(data,patch,actual,'same')
    assert data['selected_weights'].tobytes()==data['source_csr'].data[data['positions']].tobytes()
    with pytest.raises(ValueError,match='fabricated'):
        h.select_live_source(data)
    data['selected_weights'][0]=5
    with pytest.raises(ValueError,match='source selected'):
        h.neutral_values(data,patch,actual,'same')


def test_wrong_theta_CSR_patch_and_neutral_identity_reject():
    for changed in ('theta','data','indices','indptr','patch','identity'):
        data,patch,actual=neutral_fake()
        identity='same'
        if changed=='theta': data['theta'][0]=1.0001
        elif changed=='patch': patch['actual32'][0]=9
        elif changed=='identity': identity='learned'
        else: getattr(actual,changed)[0]+=1
        with pytest.raises(ValueError):
            h.neutral_values(data,patch,actual,identity)


def test_wrong_whole_CPG_and_boundary_initial_rejected(monkeypatch):
    data,patch,actual=neutral_fake()
    data.pop('selected_weights')
    data.update(saved={'cpg_hash':'cpg'},inventory={'source_network_identity':'same','graph_hash':'graph',
                                                 'source_adapter_identity':'adapter'})
    session=SimpleNamespace(snapshot=dict,network=SimpleNamespace(identity='same',tick=0,interval_steps=20),
                            control_tick=0,body=SimpleNamespace(physics_time=lambda:0),graph=SimpleNamespace(hash='graph'),
                            adapter=SimpleNamespace(identity='adapter'),neural_steps_per_control=20,
                            cpg_hash='cpg',_cpg_digest=lambda:'wrong')
    recovery=SimpleNamespace(require_saved_whole_equal=lambda *a:None)
    factory=SimpleNamespace(efficacy=SimpleNamespace(checked_checkpoint=lambda v:patch,_source_representation=lambda *a:{}),
                            recovery=recovery)
    b=SimpleNamespace(host_csr=lambda n:actual)
    timer=SimpleNamespace(check=lambda *a:None)
    with pytest.raises(ValueError,match='CPG'):
        h.initial_neutral(session,data,timer,b=b,factory=factory)
    data.pop('selected_weights')
    def wrong_whole(*a):
        raise ValueError('wrong inverse whole wall')
    recovery.require_saved_whole_equal=wrong_whole
    with pytest.raises(ValueError,match='whole wall'):
        h.initial_neutral(session,data,timer,b=b,factory=factory)
    data.pop('selected_weights')
    recovery.require_saved_whole_equal=lambda *a:None
    session._cpg_digest=lambda:'cpg'
    import numpy as np
    data['saved'].update(body={'action':{'joint_angles':[0.]*42}},
                         adapter={'original':{'original':{'original':{'tendon_activation':np.zeros(6,np.float64)}}}})
    b.capture=lambda s:({}, {'command':np.ones(42,np.float64),'tendon':np.zeros(6,np.float64)})
    b.e=SimpleNamespace(exact=np.array_equal)
    with pytest.raises(ValueError,match='boundary0'):
        h.initial_neutral(session,data,timer,b=b,factory=factory)


def test_fixed_startup_readback_pass_fail_tamper():
    import numpy as np

    from tools import verify_c_banc_m3_common_descent_body as b
    control={'command':np.zeros((100,42),np.float64),'tendon':np.zeros((100,6),np.float64)}
    target={'target_command':np.zeros((99,42),np.float64),'target_tendon':np.zeros((99,6),np.float64)}
    reported=b.startup99(**control,**target)
    assert h.startup_readback(control,target,reported,b=b)['passed']
    control['command'][1,0]=.041
    failed=b.startup99(**control,**target)
    assert failed['passed'] is False and h.startup_readback(control,target,failed,b=b)==failed
    with pytest.raises(ValueError,match='recomputation'):
        h.startup_readback(control,target,reported,b=b)


def generated_baseline(tmp_path,passed):
    """Finite generated source/world; real publisher and independent raw audit."""
    import numpy as np

    from tools import verify_c_banc_m3_common_descent_body as b
    data,patch,csr=neutral_fake()
    h.neutral_values(data,patch,csr,'same')
    control={k:np.zeros(shape,np.dtype(dtype)) for k,(shape,dtype,_) in h.control_fields(2).items()}
    control.update(control_tick=np.arange(100,dtype=np.int64),network_tick=np.arange(100,dtype=np.int64)*20,
                   physics_time_s=np.arange(100,dtype=np.float64)*.005,
                   theta=data['theta'].copy(),selected_weights=data['selected_weights'].copy())
    if not passed:
        control['command'][99,0]=.041
    target={'target_command':np.zeros((99,42),np.float64),'target_tendon':np.zeros((99,6),np.float64)}
    native={k:np.zeros(shape,np.dtype(dtype)) for k,(shape,dtype,_) in h.native_fields().items()}
    native.update(time_s=np.arange(1,4951,dtype=np.float64)*.0001,
                  upright=np.ones(4950,np.float64),slip_valid=np.ones((4950,6),bool))
    digest=b.array_sha(csr.data)
    leaves={}
    for label,dtype,count in (('host.rowptr','int32',4),('host.columns','int32',3),
                              ('host.weights','float32',3),('packed.rowptr','int32',4),
                              ('packed.columns','int32',3),('packed.palette','float32',3)):
        leaves[label]={'count':count,'dtype':dtype,'chunks':[{'start':0,'end':count,'element_count':count,
                       'dtype':dtype,'bytes':count*4,'source_sha256':digest,
                       'expected_sha256':digest,'actual_sha256':digest}]}
    proof={'schema':'flylab.m3.e344.transfer-chunks.v1','leaves':leaves,
           'source_network_identity':'same','candidate_network_identity':'same',
           'source_adapter_identity':'adapter','graph_hash':'graph','body_hash':'generated-world',
           'cpg_hash':'cpg','candidate_CSR_data_sha256':digest,'source_CSR_data_sha256':digest,
           'source_inverse_whole_exact':True,'source_boundary_command_and_tendon_exact':True,
           'neutral_source_CSR_equal':True,'pointer_version_guards_passed':True,'palette':3,
           'trial_schema':'flylab.m3-ssp-efficacy-trial.v1','patch_schema':'flylab.m3-ssp-efficacy-patch.v1',
           'source_trial_schema':'flylab.m3-ssprk3-recovery-resource-trial.v1',
           'patch_identity':'a'*64,'source_patch_identity':'b'*64,
           'boundary_command_sha256':b.array_sha(control['command'][0]),
           'boundary_tendon_sha256':b.array_sha(control['tendon'][0]),
           'theta_sha256':b.array_sha(control['theta']),'selected_sha256':b.array_sha(control['selected_weights']),
           'positions_sha256':b.array_sha(data['positions']),'groups_sha256':'c'*64}
    rows=[{'control_tick':i,'time_s':i*.005,'fault':None,'upright':1.,
           'body':{'position':[0.,0.,0.],'basis':np.eye(3).tolist(),'yaw':0.,
                   'legs':{leg:np.zeros((8,3),np.float64).tolist() for leg in b.LEGS}},
           'feet':{'load_bearing':[False]*6}} for i in range(100)]
    metadata={'landmarks':{'scope':'synthetic','count':8,'order':list(b.NATIVE_LANDMARKS)},
              'body_hash':proof['body_hash'],'model_hash':'same','graph_hash':'graph','adapter_hash':'adapter','parameters':{}}
    trace=b.lean_trace(rows,metadata)
    roster={'generated-source'+str(i):{'sha256':h.sha(str(i).encode()),'bytes':1} for i in range(133)}
    inventory={'graph_hash':'graph','source_network_identity':'same','source_adapter_identity':'adapter',
               'selected_archives':{'recurrent_mask':{'path':'generated-source0',
                  'selected_schema':{'positions':{'shape':[2],'dtype':'int64'}}},
                  'startup':{'path':'generated-source1','selected_schema':
                             {k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in target.items()}}}}
    inventory_path=tmp_path/'inventory.json'
    h.durable(inventory_path,h.json_bytes(inventory),cap=1<<20)
    reg={'body_inventory':h.file_spec('inventory.json',tmp_path),'source_input_roster':roster}
    ledger=[{'path':name,'expected_sha256':spec['sha256'],'bytes':spec['bytes']} for name,spec in roster.items()]
    ledger.extend([{'path':'generated-source0','selected':{'positions':{'c_order_sha256':proof['positions_sha256']},
                                                        'edge_parameters':{'c_order_sha256':proof['groups_sha256']}}},
                   {'path':'generated-source1','selected':{k:{'c_order_sha256':b.array_sha(v)} for k,v in target.items()}}])
    counters,construction=counters99()
    result={'run_valid':True,'baseline_completed99':True,'reason':'baseline_completed99','first_error':None,
            'secondary_errors':[],'transfer':proof,'counters':counters,'construction':construction,
            'cpg_unchanged':True,'cpg_hashes':['cpg']*100,'startup':h.normalize(b.startup99(control['command'],control['tendon'],**target)),
            'original_calls':0,'original_array_reads':0}
    c={'paths':{'output':'generated-baseline','contract':h.BASE+'/parent-e346-e359-generated.json'},
       'limits':{'output_cap_bytes':16<<20},'retained_reserve_bytes':1<<20}
    timer=SimpleNamespace(check=lambda *a:None)
    h.publish_baseline(c,result,reg,timer,control,native,trace,target,ledger,root=tmp_path)
    return c,result,reg,timer,b


def test_generated_complete_publisher_manifest_header_audit_PASS_FAIL(tmp_path):
    for passed in (True,False):
        root=tmp_path/str(passed)
        root.mkdir()
        c,result,reg,timer,b=generated_baseline(root,passed)
        startup,*_=h.audit_saved_baseline(c,result,reg,timer,b=b,root=root)
        assert startup['passed'] is passed and result['baseline_completed99']
        manifest=h.strict_json(h.read_spec(result['baseline_manifest'],root))
        assert manifest['all_fsynced_and_closed'] is True and len(manifest['files'])==7
        assert result['original_calls']==result['original_array_reads']==0


def test_generated_saved_raw_header_inventory_identity_startup_tamper(tmp_path):
    import numpy as np
    c,result,reg,timer,b=generated_baseline(tmp_path,True)
    manifest_path=h.safe(result['baseline_manifest']['path'],tmp_path)
    manifest_bytes=manifest_path.read_bytes()
    original_result=copy.deepcopy(result)
    manifest=h.strict_json(manifest_bytes)
    saved={name:h.safe(spec['path'],tmp_path).read_bytes() for name,spec in manifest['files'].items()}
    for mutation in ('bytes','header','identity','startup','leaf_omission','leaf_count','source_chunk'):
        active=copy.deepcopy(original_result)
        files=copy.deepcopy(manifest['files'])
        if mutation in ('bytes','header'):
            name='control-raw.npz'
            if mutation=='bytes':
                payload=saved[name]+b'bad'
            else:
                arrays=h.load_literal_npz(files[name],h.control_fields(2),timer,root=tmp_path)
                arrays['control_tick']=arrays['control_tick'].astype(np.int32)
                payload=h.npz_bytes(arrays,cap=8<<20)
            h.safe(files[name]['path'],tmp_path).write_bytes(payload)
            if mutation=='header':
                files[name]=h.file_spec(files[name]['path'],tmp_path)
        else:
            report=h.strict_json(saved['baseline-report.json'])
            if mutation in ('identity','leaf_omission','leaf_count','source_chunk'):
                proof=h.strict_json(saved['transfer-chunks.json'])
                if mutation=='identity':
                    proof['graph_hash']='different-source'
                elif mutation=='leaf_omission':
                    del proof['leaves']['packed.rowptr']
                elif mutation=='leaf_count':
                    proof['leaves']['host.weights']['count']=4
                    proof['leaves']['host.weights']['chunks'][0].update(end=4,element_count=4,bytes=16)
                else:
                    proof['leaves']['host.weights']['chunks'][0]['source_sha256']='0'*64
                active['transfer']=proof
                report['transfer']=proof
                h.safe(files['transfer-chunks.json']['path'],tmp_path).write_bytes(h.json_bytes(proof))
                files['transfer-chunks.json']=h.file_spec(files['transfer-chunks.json']['path'],tmp_path)
            else:
                report['startup']['passed']=False
            h.safe(files['baseline-report.json']['path'],tmp_path).write_bytes(h.json_bytes(report))
            files['baseline-report.json']=h.file_spec(files['baseline-report.json']['path'],tmp_path)
        manifest_path.write_bytes(h.json_bytes({**manifest,'files':files}))
        active['baseline_manifest']=h.file_spec(original_result['baseline_manifest']['path'],tmp_path)
        with pytest.raises(ValueError):
            h.audit_saved_baseline(c,active,reg,timer,b=b,root=tmp_path)
        for name,payload in saved.items():
            h.safe(manifest['files'][name]['path'],tmp_path).write_bytes(payload)
        manifest_path.write_bytes(manifest_bytes)


def test_generated_partial_prefix_preserves_unknown_samples_errors_and_diagnostic(tmp_path,monkeypatch):
    import numpy as np

    from tools import verify_c_banc_m3_common_descent_body as real_b
    monkeypatch.setattr(h,'monitor_check',lambda *a,**kw:True)
    session=SimpleNamespace(control_tick=0,network=SimpleNamespace(tick=0),_cpg_digest=lambda:'cpg')
    session.body=SimpleNamespace(physics_time=lambda:session.control_tick*.005)
    state={'detached':False}
    class Observer:
        def __init__(self,s,c,samples,**kw):
            state.update(counters=c,samples=samples)
        def install(self):
            return self
        def close(self):
            state['detached']=True
            raise RuntimeError('generated detach failure')
    def advance(count,capture):
        counters=state['counters']
        assert count==1 and capture is False
        for key in ('neural_requested','neural_host_returned','neural_device_confirmed'):
            counters[key]+=20
        counters['physics_requested']+=50
        steps=50 if session.control_tick==0 else 25
        counters['physics_host_returned']+=steps
        for _ in range(steps):
            state['samples'].append(((len(state['samples'])+1)*.0001,[0.]*6,[0.]*6,[True]*6,False,1.))
        if session.control_tick==1:
            counters['physics_unreturned_unknown']=True
            raise RuntimeError('generated advance first failure')
        session.control_tick+=1
        session.network.tick+=20
    session.advance=advance
    def capture(s):
        row={'control_tick':s.control_tick,'time_s':s.control_tick*.005,'fault':None,'upright':1.,
             'body':{'position':[0.,0.,0.],'basis':np.eye(3).tolist(),'yaw':0.,
                     'legs':{leg:np.zeros((8,3),np.float64).tolist() for leg in real_b.LEGS}},
             'feet':{'load_bearing':[False]*6}}
        return row,{'command':np.zeros(42,np.float64),'tendon':np.zeros(6,np.float64)}
    metadata={'landmarks':{'scope':'synthetic','count':8,'order':list(real_b.NATIVE_LANDMARKS)},
              'body_hash':'world','model_hash':'same','adapter_hash':'adapter','graph_hash':'graph','parameters':{}}
    b=SimpleNamespace(new_counters=real_b.new_counters,trace_metadata=lambda s:metadata,
                      capture=capture,LiveObserver=Observer,native_arrays=real_b.native_arrays,lean_trace=real_b.lean_trace)
    factory=SimpleNamespace(synchronize=lambda:None,install_contact=lambda *a:None)
    result={'first_error':None,'secondary_errors':[],'watch':SimpleNamespace(snapshot=dict)}
    c={'paths':{'output':'generated-partial','contract':h.BASE+'/parent-e346-e359-partial.json'},
       'limits':{'output_cap_bytes':16<<20},'retained_reserve_bytes':1<<20}
    timer=SimpleNamespace(check=lambda *a:None)
    progress={}
    with pytest.raises(RuntimeError,match='advance first'):
        h.run_prefix(session,factory,{}, {'cpg_hash':'cpg'},c,timer,b=b,result=result,progress=progress)
    assert len(progress['controls'])==len(progress['rows'])==2 and len(progress['samples'])==75
    assert result['counters']['observer_samples']==75 and result['counters']['completed_rows']==2
    assert result['counters']['advance_unreturned_unknown'] and result['counters']['physics_unreturned_unknown']
    assert result['first_error']['message']=='generated advance first failure'
    assert result['secondary_errors'][0]['message']=='generated detach failure' and state['detached']
    h.publish_partial(c,result,progress,timer,b=b,root=tmp_path)
    partial=h.strict_json(h.safe('generated-partial/partial-progress.json',tmp_path).read_bytes())
    assert partial['baseline_completed99'] is partial['qualified'] is False
    assert partial['progress']['native_samples']==75 and partial['first_error']==result['first_error']
    assert not h.safe('generated-partial/baseline-manifest.json',tmp_path).exists()
    spec=h.file_spec('generated-partial/partial-native.npz',tmp_path)
    fields={k:((75,*shape[1:]),dtype,finite) for k,(shape,dtype,finite) in h.native_fields().items()}
    assert real_b.validate_native(h.load_literal_npz(spec,fields,timer,root=tmp_path))==75


def test_historical_pair_requires_known99_CPG_publication_and_trace_identity():
    from tools import verify_c_banc_m3_common_descent_body as b
    counters,construction=counters99()
    proof={'schema':'flylab.m3.e344.transfer-chunks.v1','leaves':{},'source_network_identity':'source',
           'source_adapter_identity':'adapter','graph_hash':'graph','cpg_hash':'cpg','source_CSR_data_sha256':'d'*64,
           'body_hash':'world','candidate_network_identity':'candidate'}
    inventory={k:proof[k] for k in ('source_network_identity','source_adapter_identity','graph_hash')}
    old={'counters':counters,'construction':construction,'cpg_unchanged':True,'transfer':proof,
         'first_error':{'type':'ValueError','stage':'body-manifest-publication',
                        'message':'actual body imported source closure','notes':[]}}
    trace={'body_hash':'world','graph_hash':'graph','model_hash':'candidate','adapter_hash':'adapter','samples':[{}]*100}
    assert h.historical_identity(old,proof,trace,proof,inventory,b=b)
    for key in ('work','CPG','first_error','graph','model','decoder','world'):
        changed=copy.deepcopy(old)
        changed_trace=copy.deepcopy(trace)
        if key=='work':changed['counters']['controls_completed']=98
        elif key=='CPG':changed['cpg_unchanged']=False
        elif key=='first_error':changed['first_error']['message']='later repaired'
        else:changed_trace[{'graph':'graph_hash','model':'model_hash','decoder':'adapter_hash','world':'body_hash'}[key]]='wrong'
        with pytest.raises(ValueError):
            h.historical_identity(changed,proof,changed_trace,proof,inventory,b=b)


def test_source_alias_collision_and_actual_mismatch_preserved(monkeypatch):
    reg={'science_protected_text_sha256':{'same.py':'a'*64},'body_source_sha256':{'same.py':'b'*64},
         'unchanged_adapter_source_sha256':{}}
    with pytest.raises(ValueError,match='collision'):
        h.closure_sources(reg)
    reg['body_source_sha256']={str(h.ROOT/'same.py'):'b'*64}
    with pytest.raises(ValueError,match='alias pin collision'):
        h.closure_sources(reg)
    result={'first_error':None,'secondary_errors':[]}
    with pytest.raises(ValueError,match='closure') as caught:
        h.record_imports(result,{'source_sha256':{'same.py':'a'*64}},{'same.py':'b'*64})
    h.fail(result,caught.value,'source-closure')
    assert result['actual_import_sha256']=={'same.py':'b'*64}
    assert result['source_closure_mismatches'][0]['registered_sha256']=='a'*64 and result['revoked']


def test_historical_unqualified_unavailable_no_new_job(monkeypatch):
    timer=h.Clock(10.,{'science_s':9},clock=lambda:18.)
    result=h.historical_comparison({},None,None,None,timer,b=None,inventory=None)
    assert result['status']=='comparison_unavailable' and 'margin' in result['reason']
    assert not result['historical_run_qualified'] and not result['historical_original_publication_binding_proven']
    timer=h.Clock(10.,{'science_s':9},clock=lambda:10.)
    result=h.historical_comparison({'historical_diagnostic':{'metadata_files':{},'raw_paths':[]}},None,None,None,timer,
                                 b=None,inventory=None)
    assert result['status']=='comparison_unavailable' and result['reason'] and not result['numeric_alpha_search_performed']


def test_paired_error_locations_are_diagnostic_only():
    import numpy as np
    control={'command':np.zeros((100,42),np.float64),'tendon':np.zeros((100,6),np.float64)}
    historical={k:v.copy() for k,v in control.items()}
    historical['command'][5,7]=.05
    target={'target_command':np.zeros((99,42),np.float64),'target_tendon':np.zeros((99,6),np.float64)}
    paired=h.paired_errors(control,historical,target)
    assert paired['q']['historical_error']['max_location']=={'control_tick':5,'component':7}
    assert paired['q']['historical_error']['threshold_exceed_locations']==[{'control_tick':5,'component':7}]
    assert paired['q']['baseline_error']['max']==0 and paired['q']['prefix_difference']['max']==.05


def test_same_ENTRY_unknown_owned_close_first_secondary():
    calls=[]
    class Owner:
        def __init__(self,name):self.name=name
        def close(self):
            calls.append(self.name)
            raise RuntimeError(self.name)
    result={'first_error':None,'secondary_errors':[]}
    timer=h.Clock(10.,{'internal_s':5},clock=lambda:16.)
    h.finish_owned(Owner('detach'),Owner('body'),result,timer)
    assert calls==['detach','body'] and result['first_error']['message']=='detach'
    assert result['secondary_errors'][0]['message']=='body' and timer.entry==10.
    counters,construction=counters99();counters['advance_unreturned_unknown']=True
    with pytest.raises(ValueError,match='advance'):
        h.verify_work(counters,construction)


def test_matching_owned_identity_reuse_and_completion_timeout():
    state={'now':10.};kills=[]
    class Gone(Exception):pass
    class Process:
        def __init__(self,pid):self.pid=pid
        def create_time(self):return 11. if self.pid==1 else 99.
        def kill(self):kills.append(self.pid)
    api=SimpleNamespace(Process=Process,NoSuchProcess=Gone)
    timer=h.Clock(10.,{'outer_s':1},clock=lambda:state['now'])
    def sleep(amount):state['now']+=amount
    killed,errors=h.cleanup_matching({1:11.,2:22.},api,timer=timer)
    assert kills==[1] and killed==[{'pid':1,'create_time':11.}] and errors==[]
    matching,unknown,errors=h.bounded_owned_completion({1:11.,2:22.},timer,api,sleep=sleep)
    assert matching=={1:11.} and unknown=={} and errors[-1]['type']=='TimeoutError' and state['now']<=10.9


def test_late_outer_revocation_keeps_earliest_scientific_error(tmp_path,monkeypatch):
    (tmp_path/h.BASE).mkdir(parents=True)
    c={'paths':h.names('fake','fixture'),'limits':{'outer_s':6,'output_cap_bytes':16<<20}}
    first=h.error_record(RuntimeError('science first'),'science')
    record={'first_error':first,'secondary_errors':[],'natural_closed':True}
    timer=h.Clock(10.,c['limits'],clock=lambda:17.)
    result=h.publish_outer(c,record,timer,root=tmp_path)
    assert result['first_error']==first and result['secondary_errors'][0]['type']=='TimeoutError'
    assert not result['natural_closed'] and h.safe(c['paths']['outer'],tmp_path).exists()
    assert timer.entry==10.


def test_complete_current_case_three_phases_no_hidden_filter(monkeypatch):
    monkeypatch.setattr(h,'own_sources',lambda root:{'source':'sha'})
    monkeypatch.setattr(h,'test_functions',lambda root:['test_a','test_b'])
    row={'schema':'flylab.m3.e359.complete-pytest-receipt.v1','source_sha256':{'source':'sha'},
         'expected_functions':['test_a','test_b'],'collected_functions':['test_a','test_b'],'exit_code':0,
         'collection_errors':[],'original_calls':0,'original_array_reads':0,'complete_current_cases':True,
         'phases':{k:{v:'passed' for v in ('setup','call','teardown')} for k in ('test_a','test_b')}}
    assert h.validate_case_receipt(row)
    row['phases']['test_b']['teardown']='skipped'
    with pytest.raises(ValueError,match='setup/call/teardown'):
        h.validate_case_receipt(row)


def test_strict_finite_JSON_and_literal_source133_roles():
    reg=h.registration()
    assert reg['source_input_count']==133 and len(reg['source_input_roster'])==133
    assert not set(reg['excluded_baseline_science_inputs']).intersection(reg['source_input_roster'])
    for raw in (b'{"v":NaN}',b'{"v":1e999}',b'{"v":1,"v":2}'):
        with pytest.raises(ValueError):h.strict_json(raw)
