"""Complete E360 generated boundaries. No original payload or nested jobs."""
import contextlib
import copy
import json
import math
import os
import re
import sys
from types import FunctionType, SimpleNamespace

import pytest

from tools import verify_c_banc_m3_source_closure_diagnostic as h

# These fixtures execute only in the approved suite; FIRST performs Source extraction only.
_P30_CURRENT=h
_P18_CURRENT=None
_P17_CURRENT=None
_P16_CURRENT=None
_P15_CURRENT=h
_P15_HISTORY_CASES={
    'test_lint_recovery_P10_literal_exact82_scope11_preserves79_and_P9_caps',
    'test_lint_recovery_P10_once_pair_delta_no_grant_reset_and_uncached_selector',
    'test_lint_recovery_P10_historical_P9_whole_state_four_v2_relocations',
    'test_lint_recovery_P10_native_pair_delta_preserves_whole82_and_both_grants',
    'test_lint_recovery_P10_manual_Ruff_once_fresh_own9_capture_leaf_and_no_P11',
    'test_suite_recovery_current_full_suffix_and_immutable_historical_caps',
    'test_suite_recovery_actual_P9_reached_checkpoints_current_authority_and_source',
    'test_suite_recovery_preimport_scanner_helper_clocks_and_max84_journal',
    'test_execution_checkpoints_reached_actual_parent_ENTRY_tamper_and_bounds',
    'test_production_runner_journal_failure_all_phases_CASE_unqualified_child_and_restore',
}



_P18_LEGACY_STREAM_SOURCE=r'''def _p18_legacy_stream_bytes(spec,root):
    """Exact completed streams; positive generic Source bounds stay unchanged."""
    ADMIN_CHECK()
    require(type(spec) is dict and set(spec)=={'path','bytes','sha256'} and type(spec['path']) is str
        and type(spec['bytes']) is int and 0<=spec['bytes']<=1048576 and type(spec['sha256']) is str
        and re.fullmatch('[0-9a-f]{64}',spec['sha256']),'typed bounded historical stream')
    old_paths=tuple(BASE+'/parent-e360-generous-p'+str(n)+'-ruff-'+stream+'.txt'
        for n in (12,13,14) for stream in ('stdout','stderr'))
    additions={row['path']:row for row in _COMPLETED_STREAM_PINS}
    require(spec['path'] in old_paths or spec['path'] in additions,'fixed completed P12-P17 Ruff streams only')
    if spec['path'] in additions:typed_equal(spec,additions[spec['path']],'exact completed P15/P16/P17 stream descriptor')
    path=safe(spec['path'],root);require(path.stat().st_size==spec['bytes'],'historical stream bytes')
    with path.open('rb') as stream:raw=stream.read(spec['bytes']+1)
    require(len(raw)==spec['bytes'] and sha(raw)==spec['sha256'],'historical stream SHA')
    raw.decode('utf8',errors='strict')
    require(safe(spec['path'],root)==path and path.stat().st_size==spec['bytes'],'historical stream late path')
    with path.open('rb') as stream:again=stream.read(spec['bytes']+1)
    require(again==raw,'historical stream exact reread')
    scope=_AUTHORITY_SCOPE
    if scope is not None:
        try:
            scope.clock();require(not scope.failed,'failed stream scope cannot be reused')
            if spec['bytes']:scope._consumed(spec,root)
            else:
                empty_paths=(BASE+'/parent-e360-generous-p12-ruff-stderr.txt',BASE+'/parent-e360-generous-p13-ruff-stderr.txt',
                    BASE+'/parent-e360-generous-p14-ruff-stdout.txt',BASE+'/parent-e360-generous-p14-ruff-stderr.txt',
                    *(row['path'] for row in _COMPLETED_STREAM_PINS if row['bytes']==0))
                require(spec['path'] in empty_paths,'fixed empty completed stderr only')
                typed_equal(spec,{'path':spec['path'],'bytes':0,
                    'sha256':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'},'fixed empty stderr anchor')
                require(Path(root).resolve()==scope.root,'stream canonical root')
                canonical=os.path.normcase(str(path))
                if canonical in scope.descriptors:typed_equal(scope.descriptors[canonical],spec,'stream fence identity')
                else:
                    require(len(scope.descriptors)<512,'same stream descriptor bound')
                    scope.descriptors[canonical]=copy.deepcopy(spec)
            scope.clock()
        except Exception:
            scope.poison();raise
    ADMIN_CHECK()
    return raw
'''


@contextlib.contextmanager
def _p18_versioned_stream_bridge(module,source_path,frozen_spec):
    """Metadata-only bridge, old globals/default ROOT; old unbridged defect remains evidence."""
    current=_P18_CURRENT
    before=source_path.read_bytes()
    assert len(before)==frozen_spec['bytes'] and current.sha(before)==frozen_spec['sha256']
    old_reader=module.historical_p12_stream_bytes
    old_pins=module.__dict__.get('_COMPLETED_STREAM_PINS')
    anchor=current._COMPLETED_STREAM_PINS[1]
    positive=current._COMPLETED_STREAM_PINS[0]
    assert anchor['bytes']==0 and positive['bytes']>0
    with pytest.raises(ValueError,match='fixed completed P12/P13 Ruff streams only'):
        old_reader(anchor)
    original_clock=module.ADMIN_CHECK;original_scope=module._AUTHORITY_SCOPE
    # Only the three genuinely frozen users may use this old-API metadata bridge.
    assert module.__name__ in ('tools.e360_p17_frozen_fixture','tools.e360_p16v2_frozen_fixture',
        'tools.e360_p15v2_frozen_fixture')
    consumed_code=module.FixedAuthorityScope._consumed.__code__
    assert consumed_code.co_argcount==3 and consumed_code.co_varnames[:3]==('self','spec','root')
    legacy_definition=compile(_P18_LEGACY_STREAM_SOURCE,
        str(source_path)+'::P25-compatible-stream-metadata-bridge','exec')
    legacy_code=next(item for item in legacy_definition.co_consts
        if isinstance(item,type(old_reader.__code__)) and item.co_name=='_p18_legacy_stream_bytes')
    bridged=FunctionType(legacy_code,module.__dict__,
        name='historical_p12_stream_bytes',argdefs=(module.ROOT,))
    assert bridged.__globals__ is module.__dict__ and bridged.__defaults__==(module.ROOT,)
    module._COMPLETED_STREAM_PINS=copy.deepcopy(current._COMPLETED_STREAM_PINS)
    module.historical_p12_stream_bytes=bridged
    try:
        assert module.ADMIN_CHECK is original_clock and module._AUTHORITY_SCOPE is original_scope
        with module.fixed_authority_scope(check=original_clock) as scope:
            assert scope.depth==0 and bridged(anchor)==b''
            positive_raw=bridged(positive)
            assert scope.depth==0 and len(positive_raw)==positive['bytes'] and module.sha(positive_raw)==positive['sha256']
            assert scope.descriptors[module.os.path.normcase(str(module.safe(positive['path']))) ]==positive
            assert scope.descriptors[module.os.path.normcase(str(module.safe(anchor['path']))) ]==anchor
            scope.fence('versioned_metadata_bridge_proof')
        assert module._AUTHORITY_SCOPE is original_scope
        yield module
    finally:
        module.historical_p12_stream_bytes=old_reader
        module.ADMIN_CHECK=original_clock;module._AUTHORITY_SCOPE=original_scope
        if old_pins is None:module.__dict__.pop('_COMPLETED_STREAM_PINS',None)
        else:module._COMPLETED_STREAM_PINS=old_pins
        after=source_path.read_bytes()
        assert after==before and current.sha(after)==frozen_spec['sha256']


@pytest.fixture(scope='session')
def _p19_frozen_p18_source(tmp_path_factory):
    """Direct as-executed P18 Source/policy context; no current P19 reinterpretation."""
    import importlib.util
    from pathlib import Path
    global _P18_CURRENT
    current=_P30_CURRENT;policy=current.historical_p19_repair_policy()
    freeze=current.strict_json(current.read_spec(policy['predecessor_as_executed_freeze']))
    baseline=current.strict_json(current.read_spec(current.file_spec(current.BASE+'/parent-e360-p18-ROOT-actual-baseline.json')))
    root=tmp_path_factory.mktemp('immutable-p18-versioned-source')
    by_path={row['current']['path']:row['frozen'] for row in freeze['source_copies']}
    by_path[current.CURRENT_REPAIR_POLICY]=policy['historical_P18_policy']
    specs=[*baseline['protected322'],*baseline['old_Native29'],
        *[row['current'] for row in freeze['source_copies']],
        *[current.file_spec(name) for name in freeze['own_sources_sha256'] if name not in by_path],
        *freeze['all_P18_Native_sources'],freeze['collector_source'],policy['predecessor_as_executed_freeze'],
        policy['closed_predecessor_ledger'],current.P18_ORIGINAL_POLICY_SPEC,
        *current.P18_ACTUAL_RECORD_SPECS.values()]
    old=current.historical_p18_repair_policy()
    specs.extend(spec for spec in old.values() if type(spec) is dict and set(spec)=={'path','bytes','sha256'})
    specs.extend(current._COMPLETED_STREAM_PINS)
    pending=[current.strict_json(current.read_spec(policy['closed_predecessor_ledger']))];visited=set()
    ledger_name=current.registration()['shared_ledger_path']
    while pending:
        item=pending.pop()
        if type(item) is dict:
            if (set(item)=={'path','bytes','sha256'} and Path(item['path']).suffix=='.json'
                and not Path(item['path']).is_absolute() and item['path']!=ledger_name
                and item['path'].startswith(current.BASE+'/')):
                if item['path'] not in visited:
                    visited.add(item['path']);selected=by_path.get(item['path'],item)
                    raw=current.read_spec(selected);specs.append(item)
                    document=current.strict_json(raw)
                    if type(document) is dict and any(key in document for key in
                        ('actual_tool_return','contract','request','capture_return','collector_trace','envelope','paths')):pending.append(document)
            else:pending.extend(item.values())
        elif type(item) is list:pending.extend(item)
    for spec in specs:
        if Path(spec['path']).is_absolute():continue
        selected=by_path.get(spec['path'],spec)
        raw=current.historical_p12_stream_bytes(selected) if selected['bytes']==0 else current.read_spec(selected)
        assert len(raw)==spec['bytes'] and current.sha(raw)==spec['sha256']
        target=root/spec['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    for row in freeze['source_copies']:
        target=root/row['frozen']['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(current.read_spec(row['frozen']))
    # P18 fixed-history fixtures need predecessor96 while the actual closed99 remains separately pinned.
    target=root/ledger_name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(current.read_spec(old['closed_predecessor_ledger']))
    source_path=root/current.OWNED[0]
    loader=importlib.util.spec_from_file_location('tools.e360_p18_frozen_fixture',source_path)
    assert loader is not None and loader.loader is not None
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    assert module.ROOT==root and module.__package__=='tools'
    before_global={key:globals()[key] for key in ('h','_P18_CURRENT','_P30_CURRENT','_P17_CURRENT','_P16_CURRENT','_P15_CURRENT')}
    before_module={key:module.__dict__[key] for key in ('ROOT','CURRENT_REPAIR_SPEC','ADMIN_CHECK','_AUTHORITY_SCOPE')}
    _P18_CURRENT=module
    try:yield module
    finally:
        for key,item in before_global.items():globals()[key]=item
        for key,item in before_module.items():assert module.__dict__[key] is item
        for key,item in before_global.items():assert globals()[key] is item

def _p30_R1_saved_trace_cases(current,tmp_path,closed):
    """Actual immutable P18 trace positives; no full-history replay per negative."""
    for index,cap in ((96,3),(97,20),(98,25)):
        debit=closed['debits'][index]
        cost=current.strict_json(current.read_spec(debit['descriptor']))
        actual=current.strict_json(current.read_spec(cost['actual_tool_return']))
        call=closed['e360_calls'][cost['call_id']]
        request=current.strict_json(current.read_spec(call['request']))
        trace=current.strict_json(current.read_spec(actual['collector_trace']))
        assert trace['cap_s']==cap
        proved=current.historical_p18_collector_proof(actual['collector_trace'],request,call['request'],
            debit['actual_outer_full_return_s'],debit['actual_returned_exit_code'])
        assert proved['valid'] is True
        assert proved['full_return_s']==debit['actual_outer_full_return_s']
        if index==96:continue
        scratch=tmp_path/('p30-R1-historical-'+str(index));scratch.mkdir()
        collector=scratch/request['collector_source']['path'];collector.parent.mkdir(parents=True,exist_ok=True)
        collector.write_bytes(current.read_spec(request['collector_source']))
        target=scratch/actual['collector_trace']['path'];target.parent.mkdir(parents=True,exist_ok=True)
        changed=copy.deepcopy(trace);changed['cap_s']=30
        target.write_bytes((json.dumps(changed,separators=(',',':'))+'\n').encode('utf8'))
        before={name:getattr(current,name) for name in ('_AUTHORITY_SCOPE','_AUTHORITY_FENCE_STATE')}
        with pytest.MonkeyPatch.context() as patch:
            for name in before:patch.setattr(current,name,None)
            spec=current.file_spec(actual['collector_trace']['path'],scratch)
            with pytest.raises(ValueError,match='immutable role cap'):
                current.historical_p18_collector_proof(spec,request,call['request'],
                    debit['actual_outer_full_return_s'],debit['actual_returned_exit_code'],root=scratch)
        assert all(getattr(current,name) is item for name,item in before.items())
    # Connected historical selectors use the exact version's Source and original descriptor namespace.
    old=current.historical_p18_repair_policy()
    freeze=current.strict_json(current.read_spec(old['predecessor_as_executed_freeze']))
    for row in freeze['source_copies']:
        assert current.historical_fix_bytes(row['current'],pass_id='e360-generous-p17')==current.read_spec(row['frozen'])
    assert current.authority_spec_for_pass('e360-generous-p18')==current.P18_ORIGINAL_POLICY_SPEC
    assert current.authority_spec_for_pass('e360-generous-p19')==current.P19_ORIGINAL_POLICY_SPEC
    assert current.authority_spec_for_pass('e360-generous-p20')==current.P20_ORIGINAL_POLICY_SPEC
    assert current.authority_spec_for_pass('e360-generous-p21')==current.P21_ORIGINAL_POLICY_SPEC
    assert current.authority_spec_for_pass('e360-generous-p22')==current.P22_ORIGINAL_POLICY_SPEC
    assert current.authority_spec_for_pass('e360-generous-p23')==current.P23_ORIGINAL_POLICY_SPEC
    assert current.authority_spec_for_pass('e360-generous-p30')==current.CURRENT_REPAIR_SPEC

def _p30_R1_bootstrap_boundary_case(current,tmp_path,value,state,role):
    """Current real bootstrap -> inherited metadata -> actual SuiteDeadline constructor."""
    before={name:getattr(current,name) for name in ('ROOT','ADMIN_CHECK','_AUTHORITY_SCOPE','_AUTHORITY_FENCE_STATE','_SUITE_DEADLINE','CURRENT_REPAIR_SPEC')}
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(globals(),'h',current)
        for name in ('_AUTHORITY_SCOPE','_AUTHORITY_FENCE_STATE','_SUITE_DEADLINE'):patch.setattr(current,name,None)
        root=tmp_path/('p30-R1-bootstrap-boundary-'+role);root.mkdir()
        primary=125 if role=='fixture' else 65
        entry=current.time.perf_counter()
        name=current.BASE+'/parent-e346-e360-e360-generous-p30-R1-'+role+'.json'
        receipt_name=current.BASE+'/parent-e360-generous-p30-R1-'+role+'-receipt.json'
        c,_,_=finite_compact_contract(root,{'pass_id':'e360-generous-p30','mode':'generated','role':role,
            'future_administration_reserved_s':primary,'same_pending_fallback_reserved_s':120,
            'ledger_semantics':state,'limits':copy.deepcopy(value['effective_roles'][role]),
            'revalidation_amendment':copy.deepcopy(current.CURRENT_REPAIR_SPEC),
            'paths':{'contract':name,'pre_spawn':receipt_name}},entry=entry)
        target=current.safe(name,root);target.parent.mkdir(parents=True,exist_ok=True)
        spec=current.publish_current_contract(name,c,root)
        policy=current.safe(current.CURRENT_REPAIR_POLICY,root);policy.parent.mkdir(parents=True,exist_ok=True)
        policy.write_bytes(current.read_spec(current.CURRENT_REPAIR_SPEC))
        check=current.operation_clock(name,entry,root=root,stage='internal')
        admitted_clock=check.timer.clock
        assert check._admitted_root==root.resolve() and check._admitted_clock is admitted_clock
        assert check.timer.entry==entry and check.stage=='internal'
        assert check.timer.limits=={key:value['effective_roles'][role][key] for key in ('outer_s','child_s','internal_s','science_s')}
        with current.fixed_authority_scope(root=root,check=check) as boundary:
            current.enroll_json_descriptor(spec,root)
            with current.metadata_invocation(root) as admitted:
                assert admitted and current._AUTHORITY_SCOPE is boundary and boundary.depth==1
                assert current.strict_json(current.bounded_json_read(spec,root,cap=524288))==c
            assert boundary.depth==0 and boundary.check is check
            with current.descriptors_only_execution():
                assert current._AUTHORITY_SCOPE is None and current._AUTHORITY_FENCE_STATE is boundary
                with current.metadata_invocation(root) as inherited:
                    assert inherited and current._AUTHORITY_SCOPE is not boundary
                    assert current.strict_json(current.bounded_json_read(spec,root,cap=524288))==c
                assert current._AUTHORITY_SCOPE is None and current._AUTHORITY_FENCE_STATE is boundary
                if role=='suite':
                    receipt=current.compact_receipt(c,spec,entry,123,456.)
                    current.safe(receipt_name,root).write_bytes(current.json_bytes(receipt))
                    receipt_spec=current.file_spec(receipt_name,root)
                    timer=current.Clock(entry,c['limits'])
                    assert timer.clock is admitted_clock
                    with current.suite_deadline(c,spec,receipt_spec,timer,root) as deadline:
                        assert deadline.boundary is boundary and deadline.entry==entry
                        assert deadline.clock is admitted_clock and deadline.limits==c['limits']
                        with current.metadata_invocation(root) as inherited:
                            assert inherited and current._AUTHORITY_SCOPE is not boundary
                    assert current._SUITE_DEADLINE is None and current._AUTHORITY_FENCE_STATE is boundary
                    substituted=current.Clock(entry,c['limits'],clock=lambda:entry)
                    with pytest.raises(ValueError,match='admitted clock callable'):
                        current.SuiteDeadline(c,spec,receipt_spec,substituted,root)
                    assert not boundary.failed
            assert current._AUTHORITY_SCOPE is boundary
        assert current._AUTHORITY_SCOPE is None and current._AUTHORITY_FENCE_STATE is None
    assert all(getattr(current,name) is item for name,item in before.items())

def _p30_preimport_current_cases(current,tmp_path):
    """Actual current scanner/helper pre-import fragments, same ENTRY; no project import in the fragments."""
    import hashlib
    before={name:getattr(current,name) for name in ('ROOT','ADMIN_CHECK','_AUTHORITY_SCOPE','_AUTHORITY_FENCE_STATE','CURRENT_REPAIR_SPEC')}
    root=tmp_path/'current-p30-preimport';root.mkdir()
    source=(current.ROOT/current.OWNED[0]).read_text(encoding='utf8')
    scanner=source[source.index('_SCANNER_BOOTSTRAP_POLICY=None'):source.index('from tools import verify_c_banc_m3_candidate_body_handoff as lib')]
    helper_source=(current.ROOT/current.BASE/'parent-e360-root-metadata-helper.py').read_text(encoding='utf8')
    helper=helper_source[helper_source.index('EXECUTION_ADMISSION_PATH='):helper_source.index('from tools import verify_c_banc_m3_source_closure_diagnostic as h')]
    assert source.index('ENTRY=time.perf_counter()')<source.index('_request_bytes=')<source.index('from tools import')
    assert helper_source.index('ENTRY = time.perf_counter()')<helper_source.index('request_raw =')<helper_source.index('from tools import')
    policy=current.historical_p30_repair_policy()
    policy_path=root/current.CURRENT_REPAIR_POLICY;policy_path.parent.mkdir(parents=True,exist_ok=True)
    raw=current.read_spec(current.CURRENT_REPAIR_SPEC);policy_path.write_bytes(raw)
    request_path=root/'request.json'
    request={'pass_id':'e360-generous-p30','mode':'generated','revalidation_amendment':copy.deepcopy(current.CURRENT_REPAIR_SPEC)}
    try:
        request_path.write_bytes(current.json_bytes(request))
        scope={'ROOT':root,'__file__':str(root/'tools/current.py'),'os':SimpleNamespace(environ={'E360_CALL_REQUEST':'request.json'}),
            'sys':SimpleNamespace(argv=['tool','--scan']),'json':json,'hashlib':hashlib,'math':math,
            'time':SimpleNamespace(perf_counter=lambda:11.),'ENTRY':10.,'Path':type(current.ROOT)}
        exec(compile(scanner,'actual-P30-scanner-bootstrap','exec'),scope)  # noqa: S102 -- existing pre-import Source seam, no project imports.
        assert scope['_SCANNER_BOOTSTRAP_POLICY']==policy['scanner_limits'] and scope['ENTRY']==10.
        for mutation in ('oldpass','oldauthority','oldpolicy','deadline'):
            bad=copy.deepcopy(request)
            if mutation=='oldpass':bad['pass_id']='e360-generous-p23'
            elif mutation=='oldauthority':bad['revalidation_amendment']=copy.deepcopy(current.P23_ORIGINAL_POLICY_SPEC)
            elif mutation=='oldpolicy':policy_path.write_bytes(current.read_spec(current.P23_FROZEN_POLICY_SPEC))
            request_path.write_bytes(current.json_bytes(bad))
            probe={**scope,'request_value':copy.deepcopy(bad),'time':SimpleNamespace(perf_counter=lambda now=(1000. if mutation=='deadline' else 11.): now)}
            with pytest.raises((ValueError,TimeoutError)):exec(compile(scanner,'actual-P30-scanner-denial','exec'),probe)  # noqa: S102
            policy_path.write_bytes(raw)
        for role,limits in policy['effective_helper_limits'].items():
            if limits['mode']!='generated':continue
            request={'pass_id':'e360-generous-p30','mode':'generated','call_role':role,'action':limits['action'],
                'revalidation_amendment':copy.deepcopy(current.CURRENT_REPAIR_SPEC)}
            request_path.write_bytes(current.json_bytes(request))
            scope={'ROOT':root,'sys':SimpleNamespace(argv=['helper','request.json']),'json':json,'hashlib':hashlib,'request_value':copy.deepcopy(request),
                'time':SimpleNamespace(perf_counter=lambda:11.),'ENTRY':10.,'Path':type(current.ROOT)}
            exec(compile(helper,'actual-P30-helper-bootstrap','exec'),scope)  # noqa: S102 -- same actual ENTRY before project import.
            assert scope['policy']==limits and scope['policy_spec']==current.CURRENT_REPAIR_SPEC and scope['ENTRY']==10.
            for mutation in ('oldpass','oldauthority','oldpolicy','deadline'):
                bad=copy.deepcopy(request)
                if mutation=='oldpass':bad['pass_id']='e360-generous-p23'
                elif mutation=='oldauthority':bad['revalidation_amendment']=copy.deepcopy(current.P23_ORIGINAL_POLICY_SPEC)
                elif mutation=='oldpolicy':policy_path.write_bytes(current.read_spec(current.P23_FROZEN_POLICY_SPEC))
                request_path.write_bytes(current.json_bytes(bad))
                probe={**scope,'request_value':copy.deepcopy(bad),'time':SimpleNamespace(perf_counter=lambda now=(1000. if mutation=='deadline' else 11.): now)}
                with pytest.raises((ValueError,TimeoutError)):exec(compile(helper,'actual-P30-helper-denial','exec'),probe)  # noqa: S102
                policy_path.write_bytes(raw)
    finally:
        policy_path.write_bytes(raw)
        assert all(getattr(current,name) is value for name,value in before.items())

def _execution_actual_first_current(current):
    """Actual paid first row; later legitimate worker holds are permitted."""
    context=current.execution_context()
    assert context is not None
    debits,semantics,_totals,_local=current.ledger_state(current.registration())
    start=context['manifest']['start_index']
    assert len(debits)>start,'genuine newly paid Ruff is required before suite'
    debit=debits[start];cost=current.charged_cost(debit['descriptor'],'generated')
    actual=current.strict_json(current.read_spec(cost['actual_tool_return'],owner='dynamic'))
    call=semantics['e360_calls'][cost['call_id']]
    request=current.strict_json(current.read_spec(call['request'],owner='dynamic'))
    assert call['sequence_index']==0 and call['call_role']=='ruff' and call['state']=='charged'
    assert debit['cost_id']==cost['cost_id']==cost['call_id']==request['call_id']
    assert cost['source_sha256']==request['source_sha256']==context['freeze']['own_sources_sha256']
    assert current.validate_cost_authority(start,cost,actual)==context['manifest_spec']
    assert debit['actual_returned_exit_code']==cost['actual_returned_exit_code']==actual['actual_returned_exit_code']==0
    for wrong in (False,start-1,start+1):
        if wrong==start+1:
            bad=copy.deepcopy(cost);bad['pass_id']='e360-unadmitted-future'
            with pytest.raises(ValueError):current.validate_cost_authority(wrong,bad,actual)
        else:
            with pytest.raises(ValueError):current.validate_cost_authority(wrong,cost,actual)
    for owner,key,value in (('cost','bytes',False),('actual','path',current.CURRENT_REPAIR_POLICY),('cost','sha256','0'*64)):
        bad_cost=copy.deepcopy(cost);bad_actual=copy.deepcopy(actual)
        (bad_cost if owner=='cost' else bad_actual)['revalidation_amendment'][key]=value
        with pytest.raises(ValueError):current.validate_cost_authority(start,bad_cost,bad_actual)
    for key,wrong in (('generated_shared_cumulative_used_s',0.0),('generated_shared_remaining_s',0.0),
                      ('e360_budget_exceeded',not semantics['e360_budget_exceeded']),
                      ('generated_new_stage_retained_bytes',False),('generated_new_stage_retained_bytes',16777217)):
        bad=copy.deepcopy(semantics);bad[key]=wrong
        with pytest.raises(ValueError):current.execution_ledger_state(context,current.ROOT,bad)
    for key,wrong in (('failed_cap_s',False),('next_index',False),('failed_cost',{})):
        bad=copy.deepcopy(semantics);bad['e360_passes'][context['manifest']['pass_id']][key]=wrong
        with pytest.raises(ValueError):current.execution_ledger_state(context,current.ROOT,bad)
    bad=copy.deepcopy(semantics);bad['debits'][start]['actual_returned_exit_code']=False
    with pytest.raises(ValueError):current.execution_ledger_state(context,current.ROOT,bad)
    bad=copy.deepcopy(semantics);bad['debits'][start]['actual_outer_full_return_s']+=1.0
    with pytest.raises(ValueError):current.execution_ledger_state(context,current.ROOT,bad)
    return context

def _p30_actual_first_current(current):
    """Requires the genuine committed P30 Ruff; later calls and holds are permitted."""
    live=current.strict_json(current.read_spec(current.EXECUTION_ANCHOR_SPEC,owner='historical'))
    frozen=current.strict_json(current.read_spec(current.P30_FROZEN_FREEZE_SPEC,owner='historical'))
    expected={'path':current.CURRENT_REPAIR_POLICY,'bytes':12370,'sha256':'2b7412ea54d929a0a89df8679356e9bf8b1b6c643184f6291c4a8649006d6dfb'}
    assert current.CURRENT_REPAIR_SPEC==expected
    assert len(live['debits'])>109,'genuine P30 Ruff commit required before suite'
    prior=current.strict_json(current.read_spec(current.P29_CLOSED109_SPEC,owner='historical'))
    current.typed_equal(live['debits'][:109],prior['debits'],'unchanged actual109 paid prefix')
    debit=live['debits'][109];cost=current.charged_cost(debit['descriptor'],'generated')
    actual=current.strict_json(current.read_spec(cost['actual_tool_return'],owner='historical'))
    call=live['e360_calls']['e360-generous-p30-ruff']
    request=current.strict_json(current.read_spec(call['request'],owner='historical'))
    assert debit['cost_id']==cost['call_id']==cost['cost_id']==request['call_id']=='e360-generous-p30-ruff'
    for value in (cost,actual,request,call):
        assert value['pass_id']=='e360-generous-p30' and value['revalidation_amendment']==expected
    assert cost['charged_once'] is True and call['state']=='charged' and call['cost_id']==cost['cost_id']
    assert call['sequence_index']==0 and call['call_role']=='ruff'
    assert cost['source_sha256']==request['source_sha256']==frozen['own_sources_sha256']
    assert debit['descriptor']==current.file_spec(current.BASE+'/parent-e360-generous-p30-ruff-full-return.json')
    assert cost['actual_returned_exit_code']==actual['actual_returned_exit_code']==debit['actual_returned_exit_code']==0
    assert cost['actual_outer_full_return_s']==actual['actual_outer_full_return_s']==debit['actual_outer_full_return_s']>0
    return cost,actual,request,call,debit
def _p30_current_equivalents(current,tmp_path):
    value=current.strict_json(current.read_spec(current.P30_FROZEN_POLICY_SPEC,owner='historical'));closed=current.completed_historical_p29_proof()
    assert (len(closed['debits']),len(closed['e360_calls']),len(closed['e360_passes']))==(109,59,26)
    # Exact missing declarations are consumed by the real frozen metadata proof; no frozen P19/P20 module.
    assert current.HISTORICAL_P19_FIELDS==('e360_p19_recovery_activation','e360_p19_recovery_amendment')
    assert set(current.P19_ACTUAL_RECORD_SPECS)=={'full-return.json','final-request.json','actual.json',
        'ROOT-collector-trace.json','genuine-capture-return.json','output.json','stdout.txt','stderr.txt'}
    assert current.P19_ORIGINAL_POLICY_SPEC['path']==current.CURRENT_REPAIR_POLICY
    assert current.P19_FROZEN_POLICY_SPEC['path']!=current.CURRENT_REPAIR_POLICY
    assert all(current.P19_ORIGINAL_POLICY_SPEC[key]==current.P19_FROZEN_POLICY_SPEC[key] for key in ('bytes','sha256'))
    actual=current.strict_json(current.read_spec(current.P19_ACTUAL_RECORD_SPECS['actual.json']))
    capture=current.strict_json(current.read_spec(current.P19_ACTUAL_RECORD_SPECS['genuine-capture-return.json']))
    assert 'schema' not in actual and 'schema' not in capture and 'pass_id' not in capture
    old_freeze=current.historical_p19_source_freeze()
    assert 'current_policy_frozen' not in old_freeze
    assert closed['debits'][99]['descriptor']==current.P19_ACTUAL_RECORD_SPECS['full-return.json']
    assert closed['debits'][100]['descriptor']==current.P20_ACTUAL_RECORD_SPECS['full-return.json']
    assert closed['debits'][100]['actual_outer_full_return_s']==1.587
    assert current.historical_p12_stream_bytes(current.P20_ACTUAL_RECORD_SPECS['stderr.txt'])==b''
    frozen20=current.historical_p20_source_freeze()
    for row in frozen20['source_copies']:
        assert current.historical_fix_bytes(row['current'],pass_id='e360-generous-p20')==current.read_spec(row['frozen'])
    assert current.effective_policy(pass_id='e360-generous-p20')['primary_total_reserved_s']==768
    assert current.validate_historical_p20_activation_pair(closed)
    for field,wrong in (('new_generated_grant_s',False),('new_generated_grant_s',300),
        ('previous_revalidation_amendment',{}),('inherited100',{}),('as_executed_source_freeze',{})):
        bad=copy.deepcopy(closed);bad['e360_p20_recovery_activation'][field]=wrong
        with pytest.raises(ValueError):current.validate_historical_p20_activation_pair(bad)
    bad=copy.deepcopy(closed);bad.pop('e360_p20_recovery_amendment')
    with pytest.raises(ValueError):current.validate_historical_p20_activation_pair(bad)
    bad=copy.deepcopy(closed);bad['e360_p20_recovery_resource_grant']={}
    with pytest.raises(ValueError):current.validate_historical_p20_activation_pair(bad)
    assert closed['debits'][101]['descriptor']==current.P21_ACTUAL_RUFF_RECORD_SPECS['full-return.json']
    assert closed['debits'][102]['descriptor']==current.P21_ACTUAL_SCANNER_RECORD_SPECS['full-return.json']
    assert closed['e360_passes']['e360-generous-p21']['predecessor_failed_cap_s']==3
    assert closed['e360_passes']['e360-generous-p21']['failed_cap_s']==30
    assert current.historical_p12_stream_bytes(current.P21_ACTUAL_RUFF_RECORD_SPECS['stderr.txt'])==b''
    # Actual historical rows include scanner102 and Ruff103/104; current105 is explicitly synthetic.
    pairs=[]
    for index in (96,98,99,100,101,102,103,104,105,106,107,108):
        debit=closed['debits'][index]
        cost=current.strict_json(current.read_spec(debit['descriptor']))
        actual=current.strict_json(current.read_spec(cost['actual_tool_return']))
        pairs.append((index,cost,actual))
    by_index={index:(cost,actual) for index,cost,actual in pairs}
    for index,role,pass_id in ((102,'scanner','e360-generous-p21'),(103,'ruff','e360-generous-p22'),(104,'ruff','e360-generous-p23'),(105,'ruff','e360-generous-p25'),(106,'scanner','e360-generous-p25'),(107,'ruff','e360-generous-p27'),(108,'ruff','e360-generous-p29')):
        assert by_index[index][0]['call_role']==role and by_index[index][0]['pass_id']==pass_id
    real_cost,real_actual,_request,_call,_debit=_p30_actual_first_current(current)
    for index,cost,actual in (*pairs,(109,real_cost,real_actual)):
        assert current.validate_cost_authority(index,cost,actual)==current.authority_spec_for_pass(cost['pass_id'])
        for owner,key,wrong in (('cost','pass_id','e360-generous-p31'),('actual','pass_id','e360-generous-p30' if index<109 else 'e360-generous-p25'),
            ('cost','path',current.P23_FROZEN_POLICY_SPEC['path']),('actual','bytes',False),('cost','bytes',0),('actual','sha256','0'*64)):
            bad_cost=copy.deepcopy(cost);bad_actual=copy.deepcopy(actual)
            target=bad_cost if owner=='cost' else bad_actual
            if key=='pass_id':target[key]=wrong
            else:target['revalidation_amendment'][key]=wrong
            with pytest.raises(ValueError):current.validate_cost_authority(index,bad_cost,bad_actual)
        for wrong_index in (False,59.0,-1,{96:95,98:99,99:100,100:101,101:100,102:103,103:104,104:105,105:107,106:107,107:108,108:109,109:108}[index]):
            with pytest.raises(ValueError):current.validate_cost_authority(wrong_index,cost,actual)
        for owner,key in (('cost','pass_id'),('actual','revalidation_amendment')):
            bad_cost=copy.deepcopy(cost);bad_actual=copy.deepcopy(actual)
            (bad_cost if owner=='cost' else bad_actual).pop(key)
            with pytest.raises(ValueError):current.validate_cost_authority(index,bad_cost,bad_actual)
    assert closed['debits'][103]['descriptor']==current.P22_ACTUAL_RUFF_RECORD_SPECS['full-return.json']
    assert closed['e360_passes']['e360-generous-p22']['failed_cap_s']==3
    assert closed['e360_passes']['e360-generous-p22']['predecessor_failed_cap_s']==30
    assert current.historical_p12_stream_bytes(current.P22_ACTUAL_RUFF_RECORD_SPECS['stderr.txt'])==b''
    assert current.validate_historical_p22_activation_pair(closed)
    assert current.validate_historical_p21_activation_pair(closed)
    for field,wrong in (('new_generated_grant_s',False),('new_generated_grant_s',300),('inherited101',{}),('as_executed_source_freeze',{})):
        bad=copy.deepcopy(closed);bad['e360_p21_recovery_activation'][field]=wrong
        with pytest.raises(ValueError):current.validate_historical_p21_activation_pair(bad)
    bad=copy.deepcopy(closed);bad.pop('e360_p21_recovery_amendment')
    with pytest.raises(ValueError):current.validate_historical_p21_activation_pair(bad)
    assert closed['debits'][104]['descriptor']==current.P23_ACTUAL_RUFF_RECORD_SPECS['full-return.json']
    assert current.completed_historical_p23_proof()['generated_shared_cumulative_used_s']==666.1160156000001
    assert current.completed_historical_p23_proof()['e360_generated_local_used_s']==555.159
    assert current.historical_p12_stream_bytes(current.P23_ACTUAL_RUFF_RECORD_SPECS['stderr.txt'])==b''
    assert len(current.FIXED_READERS)==18
    _p30_R1_saved_trace_cases(current,tmp_path,closed)
    old_budget=current.historical_budget_authority(live=closed)
    assert old_budget['generated_shared_limit_s']==1700 and old_budget['generated_local_limit_s']==1589
    assert old_budget['p19_recovery_active']
    assert not any(key in closed for key in current.CURRENT_P30_FIELDS)
    freeze=current.strict_json(current.read_spec(current.P30_FROZEN_FREEZE_SPEC,owner='historical'))
    for row in freeze['source_copies']:
        assert row['current']['sha256']==row['frozen']['sha256']==freeze['own_sources_sha256'][row['current']['path']]
        assert current.sha(current.read_spec(row['frozen'],owner='historical'))==row['frozen']['sha256']
    assert current.effective_policy(pass_id='e360-generous-p30')['primary_total_reserved_s']==768
    assert current.historical_budget_authority()['generated_shared_limit_s']==1700
    assert current.historical_budget_authority()['generated_local_limit_s']==1589
    live=current.strict_json(current.read_spec(current.EXECUTION_ANCHOR_SPEC,owner='historical'))
    # One actual hot current Source ledger positive includes the first index99 comparison.
    if current.execution_bootstrap(current.ROOT) is None:
        checked_debits,semantics,_totals,_local=current.ledger_state(current.registration())
    else:
        semantics=current.strict_json(current.read_spec(current.EXECUTION_ANCHOR_SPEC,owner='historical'));checked_debits=semantics['debits']
    assert checked_debits[:109]==closed['debits']
    assert semantics['e360_p21_recovery_amendment']==current.P21_ORIGINAL_POLICY_SPEC
    assert semantics['e360_p23_recovery_amendment']==current.P23_ORIGINAL_POLICY_SPEC
    assert semantics['e360_p25_recovery_amendment']==current.P25_ORIGINAL_POLICY_SPEC
    assert semantics['e360_p27_recovery_amendment']==current.P27_ORIGINAL_POLICY_SPEC
    assert semantics['e360_p29_recovery_amendment']==current.P29_ORIGINAL_POLICY_SPEC
    saved_p27=current.strict_json(current.read_spec(current.P27_ACTUAL_RUFF_RECORD_SPECS['full-return.json']))
    assert saved_p27['internal_completion_valid'] is False and saved_p27['collector_proof_valid'] is True
    assert saved_p27['capture_proof_errors']==[{'stage':'ROOT-capture-proof','message':'Typed metadata descriptor'}]
    assert saved_p27['capture_return']['process_identity']['start_utc_ticks']==639268983001446681
    assert saved_p27['capture_process_identity'] is None and closed['debits'][107]['actual_returned_exit_code']==1
    assert semantics['e360_p30_recovery_amendment']==current.CURRENT_REPAIR_SPEC
    assert current.historical_p30_scope_authority(live=live)
    for key in current.CURRENT_P30_FIELDS:
        bad=copy.deepcopy(live);bad.pop(key)
        with pytest.raises(ValueError):current.historical_p30_scope_authority(live=bad)
    for identity in ('e360-generous-p23','e360-generous-p31','e360-original'):
        with pytest.raises(ValueError):current.check_call_request({'pass_id':identity,'mode':'generated'})
    missing_pair=copy.deepcopy(live)
    for key in current.HISTORICAL_P20_FIELDS:missing_pair.pop(key)
    with pytest.raises(ValueError):current.historical_budget_authority(live=missing_pair)
    expected_allocation=current.p19_expected_allocation()
    actual_allocation=current.strict_json(current.read_spec(current.file_spec(current.historical_p19_repair_policy()['allocation_record_path'])))
    assert current.validate_p19_allocation_document(actual_allocation,expected_allocation)==expected_allocation
    for key,wrong in [('generated_new_grant_s',False),('generated_new_grant_s',0),('generated_new_grant_s',600),('closed_predecessor_ledger',{}),('policy',{}),('human_authorization',{}),('ROOT_review_cycle_complete',False),('reset',True)]:
        bad=copy.deepcopy(actual_allocation);bad[key]=wrong
        with pytest.raises(ValueError):current.validate_p19_allocation_document(bad,expected_allocation)
    policy_only=copy.deepcopy(closed);policy_only.pop('e360_p19_recovery_activation');policy_only.pop('e360_p19_recovery_amendment')
    with pytest.raises(ValueError):current.historical_budget_authority(live=policy_only)
    assert current.effective_policy(pass_id='e360-generous-p18')['fixture_future_administration_reserved_s']==95
    assert current.effective_policy(pass_id='e360-generous-p18')['suite_future_administration_reserved_s']==50
    assert current.p30_recovery_activation_allowed(closed,lock_absent=True)
    activated=copy.deepcopy(closed)
    activated.update(e360_p30_recovery_activation=current.p30_recovery_event(),e360_p30_recovery_amendment=copy.deepcopy(current.CURRENT_REPAIR_SPEC))
    assert current.p30_recovery_delta_valid(closed,activated)
    assert current.historical_p30_scope_authority(live=activated)
    for key in closed:assert activated[key]==closed[key]
    for field in ('generated_shared_cumulative_limit_s','e360_generated_local_limit_s','generated_shared_cumulative_used_s','generated_shared_remaining_s','e360_generated_local_used_s'):
        bad=copy.deepcopy(activated);bad[field]+=1
        with pytest.raises(ValueError):current.p30_recovery_delta_valid(closed,bad)
    for wrong in (False,300,0.0,-1):
        bad=copy.deepcopy(activated);bad['e360_p30_recovery_activation']['new_generated_grant_s']=wrong
        with pytest.raises(ValueError):current.historical_p30_scope_authority(live=bad)
    bad=copy.deepcopy(activated);bad['e360_p30_recovery_resource_grant']={'new_generated_grant_s':300}
    with pytest.raises(ValueError):current.p30_recovery_delta_valid(closed,bad)
    with pytest.raises(ValueError):current.historical_p30_scope_authority(live=bad)
    with pytest.raises(ValueError):current.p30_recovery_activation_allowed(activated)
    with pytest.raises(ValueError):current.p30_recovery_activation_allowed(closed,lock_absent=False)
    for role,cap,suffix,total,primary in (('fixture',60,765,825,125),('suite',520,185,705,65)):
        call={'pass_id':'e360-generous-p30','mode':'generated','call_role':role+'_worker','reserved_s':cap,'future_reserved_s':suffix,'state':'begun'}
        state=copy.deepcopy(closed);state['e360_passes']['e360-generous-p30']={'next_index':4}
        state['e360_calls']['current']=call
        state['e360_reservations']['current']={'mode':'generated','role':role,'reserved_s':cap,'pending_reservations':{'suite':520} if role=='fixture' else {}}
        assert current.validate_p30_call_future(call,pass_id='e360-generous-p30')==total
        assert current.prospective_remaining(state,'generated')==total
        _p30_R1_bootstrap_boundary_case(current,tmp_path,value,state,role)
        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(globals(),'h',current)
            root=tmp_path/('p30-transport-'+role);root.mkdir()
            name=current.BASE+'/parent-e346-e360-e360-generous-p30-growth-'+role+'.json'
            contract={'pass_id':'e360-generous-p30','mode':'generated','role':role,'paths':{'contract':name},
                'revalidation_amendment':copy.deepcopy(current.CURRENT_REPAIR_SPEC),'limits':copy.deepcopy(value['effective_roles'][role]),
                'future_administration_reserved_s':primary,'same_pending_fallback_reserved_s':120,
                'ledger_semantics':state,'ledger_debits':copy.deepcopy(closed['debits']),
                'source_sha256':copy.deepcopy(freeze['own_sources_sha256'])}
            receipt_root=tmp_path/('p30-receipt-'+role);receipt_root.mkdir()
            c,compact_spec,receipt=finite_compact_contract(receipt_root,{'pass_id':'e360-generous-p30','mode':'generated','role':role,
                'future_administration_reserved_s':primary,'same_pending_fallback_reserved_s':120,'ledger_semantics':state})
            assert len(receipt)==34 and current.validate_current_receipt(c,compact_spec,receipt,root=receipt_root,entry=10.)==receipt
            for wrong in (None,False,0,-1,240,120.0):
                bad=copy.deepcopy(receipt);bad['same_pending_fallback_reserved_s']=wrong
                with pytest.raises(ValueError):current.validate_current_receipt(c,compact_spec,bad,root=receipt_root)
            # Growth is in meaningful full Source/debit metadata; no fields are dropped.
            contract['source_sha256'].update({f'growth/source-{i:04}.py':'a'*64 for i in range(2200)})
            raw=current.current_contract_bytes(contract,root=root)
            assert 262144<len(raw)<=524288 and current.strict_json(raw)==contract
            target=current.safe(name,root);target.parent.mkdir(parents=True,exist_ok=True)
            spec=current.publish_current_contract(name,contract,root)
            assert current.strict_json(current.bounded_json_read(spec,root,cap=524288))==contract
            for wrong in (None,False,0,-1,primary+120,float(primary),str(primary)):
                bad={**contract,'future_administration_reserved_s':wrong}
                with pytest.raises(ValueError):current.validate_p30_primary_future(bad)
            with pytest.raises(ValueError,match='bytes=.*cap=524288.*exists=true'):current.publish_current_contract(name,contract,root)
            absent=current.BASE+'/parent-e346-e360-e360-generous-p30-over-'+role+'.json'
            boundary=b' '*524288
            assert current.current_contract_bound(absent,boundary,root)==boundary
            with pytest.raises(ValueError,match='bytes=524289.*cap=524288.*exists=false'):current.current_contract_bound(absent,boundary+b' ',root)
            assert not current.safe(absent,root).exists()
            policy_target=current.safe(current.CURRENT_REPAIR_POLICY,root);policy_target.parent.mkdir(parents=True,exist_ok=True)
            policy_target.write_bytes(current.read_spec(current.CURRENT_REPAIR_SPEC))
            entry=current.time.perf_counter()
            clock=current.operation_clock(name,entry,root=root)
            assert clock.timer.entry==entry
            for key,wrong in [('pass_id','e360-generous-p23'),('revalidation_amendment',{}),('role','saved_prefix_diagnostic')]:
                bad=copy.deepcopy(contract);bad[key]=wrong;target.write_bytes((json.dumps(bad,separators=(',',':'))+'\n').encode('utf8'))
                with pytest.raises(ValueError):current.operation_clock(name,entry,root=root)
            target.write_bytes(raw)
            saved=(current.ROOT,current.ADMIN_CHECK,current._AUTHORITY_SCOPE,current.CURRENT_REPAIR_SPEC)
            with current.fixed_authority_scope(check=lambda:None) as scope:
                current.historical_p30_repair_policy();assert scope.depth==0
                scope.fence('current_P30_depth0_END')
            assert (current.ROOT,current.ADMIN_CHECK,current._AUTHORITY_SCOPE,current.CURRENT_REPAIR_SPEC)==saved
            assert all(a is b for a,b in zip((current.ROOT,current.ADMIN_CHECK,current._AUTHORITY_SCOPE,current.CURRENT_REPAIR_SPEC),saved))

@pytest.fixture(scope='session')

def _p18_frozen_p17_source(tmp_path_factory,_p19_frozen_p18_source):
    """Actual P17 Source4/Native4/collector/closed96, separate from hot P18."""
    import importlib.util
    from pathlib import Path
    global _P17_CURRENT
    current=_P18_CURRENT;value=current.current_repair_policy()
    freeze=current.strict_json(current.read_spec(value['predecessor_as_executed_freeze']))
    baseline=current.strict_json(current.read_spec(current.p17_recovery_amendment()['actual_P17_baseline_native_preservation']))
    root=tmp_path_factory.mktemp('immutable-p17-versioned-source')
    by_path={row['current']['path']:row['frozen'] for row in freeze['source_copies']}
    actual_own=[row['current'] for row in freeze['source_copies']]
    actual_own.extend(current.file_spec(name) for name in freeze['own_sources_sha256'] if name not in by_path)
    specs=[*baseline['protected322'],*actual_own,*baseline['old_Native24'],
        *freeze['all_P17_Native_sources'],freeze['collector_source'],value['predecessor_as_executed_freeze'],
        value['closed_predecessor_ledger'],*current._CURRENT_P18_PINS,*current.P17_SAVED_RECORD_SPECS.values()]
    old=current.p17_recovery_amendment()
    specs.extend(spec for spec in old.values() if type(spec) is dict and set(spec)=={'path','bytes','sha256'})
    specs.extend(old['current_P17_source_snapshots']);specs.extend(old['reviewed_source_snapshots'])
    specs.extend(row.get('snapshot',row.get('frozen')) for row in old['historical_concrete_fix_relocations'])
    pending=[current.strict_json(current.read_spec(value['closed_predecessor_ledger']))];visited=set()
    ledger_name=current.registration()['shared_ledger_path']
    while pending:
        item=pending.pop()
        if type(item) is dict:
            if (set(item)=={'path','bytes','sha256'} and Path(item['path']).suffix=='.json'
                and not Path(item['path']).is_absolute() and item['path']!=ledger_name
                and Path(item['path']).as_posix().startswith(current.BASE+'/')):
                if item['path'] not in visited:
                    visited.add(item['path']);raw=current.read_spec(item);specs.append(item)
                    document=current.strict_json(raw)
                    if type(document) is dict and any(key in document for key in
                        ('actual_tool_return','contract','request','capture_return','collector_trace','envelope','paths')):pending.append(document)
            else:pending.extend(item.values())
        elif type(item) is list:pending.extend(item)
    specs.extend(current._COMPLETED_STREAM_PINS)
    for spec in specs:
        if Path(spec['path']).is_absolute():continue
        selected=by_path.get(spec['path'],spec)
        raw=current.historical_p12_stream_bytes(selected) if selected['bytes']==0 else current.read_spec(selected)
        assert len(raw)==spec['bytes'] and current.sha(raw)==spec['sha256']
        target=root/spec['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    for row in freeze['source_copies']:
        target=root/row['frozen']['path'];target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(current.read_spec(row['frozen']))
    ledger_target=root/ledger_name;ledger_target.parent.mkdir(parents=True,exist_ok=True)
    ledger_target.write_bytes(current.read_spec(value['closed_predecessor_ledger']))
    source_path=root/current.OWNED[0];frozen_spec=by_path[current.OWNED[0]]
    loader=importlib.util.spec_from_file_location('tools.e360_p17_frozen_fixture',source_path)
    assert loader is not None and loader.loader is not None
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    assert module.ROOT==root and module.__package__=='tools'
    previous=_P17_CURRENT;_P17_CURRENT=module
    try:
        with _p18_versioned_stream_bridge(module,source_path,frozen_spec):yield module
    finally:_P17_CURRENT=previous

@pytest.fixture(scope='session')
def _p17_frozen_p16_source(tmp_path_factory,_p18_frozen_p17_source):
    """Actual executed P16v2 Source4/metadata, separate from accepted R1 and active P17."""
    import importlib.util
    from pathlib import Path
    global _P16_CURRENT
    current=_P17_CURRENT;value=current.p17_recovery_amendment()
    baseline=current.strict_json(current.read_spec(value['actual_P17_baseline_native_preservation']))
    freeze=current.strict_json(current.read_spec(value['historical_P16_final_source_freeze']))
    root=tmp_path_factory.mktemp('immutable-p16v2-versioned-source')
    frozen_by_path={row['current']['path']:row['frozen'] for row in freeze['source_copies']}
    old=current.p16_recovery_amendment()
    specs=[*baseline['protected322'],*baseline['current_own9'],*baseline['old_Native24'],
        current.file_spec(current.P16_RECOVERY_AMENDMENT),value['historical_P16_final_source_freeze'],
        *[spec for spec in old.values() if type(spec) is dict and set(spec)=={'path','bytes','sha256'}],
        *old['reviewed_source_snapshots'],*old['current_correction_source_snapshots'],*old['current_P16_source_snapshots'],
        *[row.get('snapshot',row.get('frozen')) for row in old['historical_concrete_fix_relocations']],
        *[value[key] for key in current.P16_SAVED_RECORD_KEYS.values()],value['historical_P16_closed_ledger94']]
    # Metadata-only dependency closure. Original array/header/archive descriptors are never opened.
    shared_ledger_path=current.registration()['shared_ledger_path']
    pending=[current.strict_json(current.read_spec(value['historical_P16_closed_ledger94']))]
    visited=set()
    while pending:
        item=pending.pop()
        if type(item) is dict:
            if (set(item)=={'path','bytes','sha256'} and Path(item['path']).suffix=='.json'
                and not Path(item['path']).is_absolute() and Path(item['path'])!=Path(shared_ledger_path)
                and Path(item['path']).as_posix().startswith(current.BASE+'/')):
                if item['path'] not in visited:
                    visited.add(item['path']);raw=current.read_spec(item);specs.append(item)
                    document=current.strict_json(raw)
                    # Paid-record dependencies only; reports/authority observations are retained as opaque pinned files.
                    if type(document) is dict and any(key in document for key in
                        ('actual_tool_return','contract','request','capture_return','collector_trace','envelope','paths')):
                        pending.append(document)
            else:pending.extend(item.values())
        elif type(item) is list:pending.extend(item)
    for spec in specs:
        if Path(spec['path']).is_absolute():continue
        assert Path(spec['path']).suffix in ('.py','.ps1','.json','.md','.txt','.ini')
        selected=frozen_by_path.get(spec['path'],spec)
        raw=current.historical_p12_stream_bytes(selected) if selected['bytes']==0 else current.read_spec(selected)
        assert len(raw)==spec['bytes'] and current.sha(raw)==spec['sha256']
        target=root/spec['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    for row in freeze['source_copies']:
        raw=current.read_spec(row['frozen'])
        assert len(raw)==row['current']['bytes'] and current.sha(raw)==row['current']['sha256']
        target=root/row['frozen']['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        assert current.sha((root/row['current']['path']).read_bytes())==row['current']['sha256']
    ledger_target=root/current.registration()['shared_ledger_path'];ledger_target.parent.mkdir(parents=True,exist_ok=True)
    ledger_target.write_bytes(current.read_spec(value['historical_P16_closed_ledger94']))
    source_path=root/current.OWNED[0]
    frozen_spec=next(row['frozen'] for row in freeze['source_copies'] if row['kind']=='tool')
    frozen=source_path.read_bytes()
    assert len(frozen)==frozen_spec['bytes'] and current.sha(frozen)==frozen_spec['sha256']
    loader_spec=importlib.util.spec_from_file_location('tools.e360_p16v2_frozen_fixture',source_path)
    assert loader_spec is not None and loader_spec.loader is not None
    module=importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(module)
    assert module.__file__==str(source_path) and module.__package__=='tools' and module.ROOT==root
    after_load=source_path.read_bytes()
    assert len(after_load)==frozen_spec['bytes'] and current.sha(after_load)==frozen_spec['sha256']
    previous=_P16_CURRENT;_P16_CURRENT=module
    try:
        with _p18_versioned_stream_bridge(module,source_path,frozen_spec):yield module
    finally:_P16_CURRENT=previous



@pytest.fixture(scope='session')
def _p16_frozen_p15_source(tmp_path_factory,_p17_frozen_p16_source):
    """Actual executed P15v2 Source4/metadata, separate from accepted R1 and active P16."""
    import importlib.util
    from pathlib import Path
    global _P15_CURRENT
    current=_P16_CURRENT;value=current.p16_recovery_amendment()
    baseline=current.strict_json(current.read_spec(value['actual_P16_baseline_native_preservation']))
    freeze=current.strict_json(current.read_spec(value['historical_P15_final_source_freeze']))
    root=tmp_path_factory.mktemp('immutable-p15v2-versioned-source')
    frozen_by_path={row['current']['path']:row['frozen'] for row in freeze['source_copies']}
    old=current.p15_recovery_amendment()
    specs=[*baseline['protected322'],*baseline['current_own9'],*baseline['old_Native19'],
        current.file_spec(current.P15_RECOVERY_AMENDMENT),value['historical_P15_final_source_freeze'],
        *[spec for spec in old.values() if type(spec) is dict and set(spec)=={'path','bytes','sha256'}],
        *old['reviewed_source_snapshots'],*old['current_correction_source_snapshots'],
        *[row.get('snapshot',row.get('frozen')) for row in old['historical_concrete_fix_relocations']],
        *[value[key] for key in current.P15_SAVED_RECORD_KEYS.values()],value['historical_P15_closed_ledger93'],
        *_P18_CURRENT._COMPLETED_STREAM_PINS[:2]]
    # Metadata-only dependency closure. Original array/header/archive descriptors are never opened.
    shared_ledger_path=current.registration()['shared_ledger_path']
    pending=[current.strict_json(current.read_spec(value['historical_P15_closed_ledger93']))]
    visited=set()
    while pending:
        item=pending.pop()
        if type(item) is dict:
            if (set(item)=={'path','bytes','sha256'} and Path(item['path']).suffix=='.json'
                and not Path(item['path']).is_absolute() and Path(item['path'])!=Path(shared_ledger_path)
                and Path(item['path']).as_posix().startswith(current.BASE+'/')):
                if item['path'] not in visited:
                    visited.add(item['path']);raw=current.read_spec(item);specs.append(item)
                    document=current.strict_json(raw)
                    # Paid-record dependencies only; reports/authority observations are retained as opaque pinned files.
                    if type(document) is dict and any(key in document for key in
                        ('actual_tool_return','contract','request','capture_return','collector_trace','envelope','paths')):
                        pending.append(document)
            else:pending.extend(item.values())
        elif type(item) is list:pending.extend(item)
    for spec in specs:
        if Path(spec['path']).is_absolute():continue
        assert Path(spec['path']).suffix in ('.py','.ps1','.json','.md','.txt','.ini')
        selected=frozen_by_path.get(spec['path'],spec)
        raw=current.historical_p12_stream_bytes(selected) if selected['bytes']==0 else current.read_spec(selected)
        assert len(raw)==spec['bytes'] and current.sha(raw)==spec['sha256']
        target=root/spec['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    for row in freeze['source_copies']:
        raw=current.read_spec(row['frozen'])
        assert len(raw)==row['current']['bytes'] and current.sha(raw)==row['current']['sha256']
        target=root/row['frozen']['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        assert current.sha((root/row['current']['path']).read_bytes())==row['current']['sha256']
    ledger_target=root/current.registration()['shared_ledger_path'];ledger_target.parent.mkdir(parents=True,exist_ok=True)
    ledger_target.write_bytes(current.read_spec(value['historical_P15_closed_ledger93']))
    source_path=root/current.OWNED[0]
    frozen_spec=next(row['frozen'] for row in freeze['source_copies'] if row['kind']=='tool')
    frozen=source_path.read_bytes()
    assert len(frozen)==frozen_spec['bytes'] and current.sha(frozen)==frozen_spec['sha256']
    loader_spec=importlib.util.spec_from_file_location('tools.e360_p15v2_frozen_fixture',source_path)
    assert loader_spec is not None and loader_spec.loader is not None
    module=importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(module)
    assert module.__file__==str(source_path) and module.__package__=='tools' and module.ROOT==root
    after_load=source_path.read_bytes()
    assert len(after_load)==frozen_spec['bytes'] and current.sha(after_load)==frozen_spec['sha256']
    previous=_P15_CURRENT;_P15_CURRENT=module
    try:
        with _p18_versioned_stream_bridge(module,source_path,frozen_spec):yield module
    finally:_P15_CURRENT=previous



def _p16_scope_bounds(root):
    root.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        _,raw,_=finite_scope_authority(root,patch)
        with h.fixed_authority_scope(root=root,check=lambda:None) as scope:
            for index in range(17):
                name=f'p16-literal{index}.json';(root/name).write_bytes(raw)
                patch.setitem(h.FIXED_READERS,'registration',(name,len(raw),h.sha(raw)))
                h.registration(root)
            patch.setitem(h.FIXED_READERS,'registration',('eighteenth.json',len(raw),h.sha(raw)))
            with pytest.raises(ValueError,match='entry bound'):h.registration(root)
            assert not scope.values and scope.failed and len(scope.descriptors)==18
            with pytest.raises(ValueError,match='cannot be reused'):h.registration(root)

def _p16_saved_documents(current):
    value=current.p16_recovery_amendment()
    documents={name:current.strict_json(current.read_spec(value[key]))
        for name,key in current.P15_SAVED_RECORD_KEYS.items()}
    documents['call']=current.strict_json(current.read_spec(value['historical_P15_closed_ledger93']))['e360_calls']['e360-generous-p15-ruff']
    return value,documents


def _p16_saved_record_cases(current):
    """Finite exact typed gates, one full positive; each negative touches the same narrow production validator."""
    _,documents=_p16_saved_documents(current);cases=[]
    for name,document in documents.items():
        key=next(iter(document));bad=copy.deepcopy(document);bad.pop(key)
        cases.append({'name':name+'_missing','leaf':name,'document':bad})
        bad=copy.deepcopy(document);bad['unrecorded_extra']=False
        cases.append({'name':name+'_extra','leaf':name,'document':bad})
    for name,path,wrong in [
        ('cost',('internal_completion_valid',),False),('cost',('actual_returned_exit_code',),True),
        ('actual',('actual_outer_full_return_s',),True),('request',('source_sha256',),{}),
        ('call',('reserved_s',),3.0),('call',('cost_id',),None),
        ('summary',('process_identity','start_utc_ticks'),639268015933086786),
        ('summary',('process_identity','start_utc_ticks'),float(639268015933086785)),
        ('manifest',('process_identity','process_id'),True),('trace',('session_id',),0),
        ('trace',('full_return_s',),1.800),('closure',('known_owned_ruff','process_id'),32628.0),
        ('closure',('known_owned_ruff','observed_start_utc_ticks'),639268015933086785),
        ('closure',('known_owned_ruff','matching_owned_live'),0),
        ('decision',('original_execution_authority',),0),('decision',('original_execution_authority',),None),
        ('decision',('original_GPU_body_neural_UI',),False),('decision',('first_failure',),{}),
    ]:
        bad=copy.deepcopy(documents[name]);target=bad
        for key in path[:-1]:target=target[key]
        target[path[-1]]=wrong
        cases.append({'name':name+'_typed_'+str(len(cases)),'leaf':name,'document':bad})
    return cases


def _p16_current_equivalents(current):
    value=current.p16_recovery_amendment();old=current.p15_recovery_amendment()
    assert len(current.FIXED_READERS)==17 and current.FIXED_SCOPE_POLICY['max_entries']==16
    assert value['fixed_authority_scope_policy']['reader_ids']==old['fixed_authority_scope_policy']['reader_ids']+['p16_recovery_amendment']
    assert [value['fixed_authority_scope_policy'][key] for key in ('max_entries','max_text_descriptors','max_literal_json_bytes','max_aggregate_literal_json_bytes')]==[17,512,262144,2097152]
    assert value['historical_concrete_fix_relocations'][:50]==old['historical_concrete_fix_relocations']
    assert len(value['historical_concrete_fix_relocations'])==54 and len(value['preserved_activation_fields'])==26
    before=current.completed_historical_p15_proof(value=value)
    assert before==current.strict_json(current.read_spec(value['historical_P15_closed_ledger93']))
    assert (len(before['debits']),len(before['e360_calls']),len(before['e360_passes']))==(93,43,15)
    _,documents=_p16_saved_documents(current)
    for name,document in documents.items():current.validate_saved_p15_record(name,document)
    for case in _p16_saved_record_cases(current):
        with pytest.raises(ValueError,match=re.escape('actual P15 '+case['leaf']+' exact typed record')):
            current.validate_saved_p15_record(case['leaf'],case['document'])
    after=copy.deepcopy(before)
    after.update(e360_p16_recovery_activation=current.p16_recovery_event(),e360_p16_recovery_amendment=current.file_spec(current.P16_RECOVERY_AMENDMENT))
    assert current.p16_recovery_activation_allowed(before,lock_absent=True)
    assert current.p16_recovery_delta_valid(before,after) and current.current_scope_authority(live=after)
    assert not current.current_scope_authority(live=before)
    for key in current.P16_RECOVERY_FIELDS:
        bad=copy.deepcopy(after);bad.pop(key)
        with pytest.raises(ValueError):current.current_scope_authority(live=bad)
    for identity in ('e360-generous-p15','e360-generous-p14','e360-generous-p17','e360-original','e360-auto'):
        with pytest.raises(ValueError):current.check_call_request({'pass_id':identity,'mode':'generated'})
        with pytest.raises(ValueError):current.validate_p16_pass(identity,{},before['debits'])
    with pytest.raises(ValueError):current.validate_p15_pass('e360-generous-p15',{},before['debits'])
    with pytest.raises(ValueError):current.effective_policy(pass_id='e360-generous-p17')
    assert current.effective_policy(pass_id='e360-generous-p16')['suite_future_administration_reserved_s']==50
    rows=value['historical_concrete_fix_relocations'][50:54]
    hot=[row['original'] for row in rows];frozen=[row['frozen'] for row in rows]
    assert all(current.file_spec(spec['path'])['sha256']!=spec['sha256'] for spec in hot)
    actual_read=current.read_spec;hits=[]
    def guarded_actual_read(spec,root=current.ROOT):
        if spec in hot:raise AssertionError('Current P16 read stale mutable actual P15 Source4')
        if spec in frozen:hits.append(copy.deepcopy(spec))
        return actual_read(spec,root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(current,'read_spec',guarded_actual_read)
        assert current.validate_live_calls(after,after['debits']) is None
    assert all(spec in hits for spec in frozen)
    assert after['debits']==before['debits'] and after['e360_calls']==before['e360_calls'] and after['e360_passes']==before['e360_passes']
    for spec in hot:
        with pytest.raises(ValueError,match='unregistered historical P15 mutable descriptor'):
            current.historical_fix_bytes(current.file_spec(spec['path']),pass_id='e360-generous-p15')


def _p17_scope_bounds(root):
    root.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        _,raw,_=finite_scope_authority(root,patch)
        with h.fixed_authority_scope(root=root,check=lambda:None) as scope:
            for index in range(18):
                name=f'p17-literal{index}.json';(root/name).write_bytes(raw)
                patch.setitem(h.FIXED_READERS,'registration',(name,len(raw),h.sha(raw)))
                h.registration(root)
            patch.setitem(h.FIXED_READERS,'registration',('nineteenth.json',len(raw),h.sha(raw)))
            with pytest.raises(ValueError,match='entry bound'):h.registration(root)
            assert not scope.values and scope.failed and len(scope.descriptors)==19
            with pytest.raises(ValueError,match='cannot be reused'):h.registration(root)

def _p17_saved_documents(current):
    value=current.p17_recovery_amendment()
    documents={name:current.strict_json(current.read_spec(value[key]))
        for name,key in current.P16_SAVED_RECORD_KEYS.items()}
    documents['call']=current.strict_json(current.read_spec(value['historical_P16_closed_ledger94']))['e360_calls']['e360-generous-p16-ruff']
    return value,documents


def _p17_saved_record_cases(current):
    """Finite exact typed gates, one full positive; each negative touches the same narrow production validator."""
    _,documents=_p17_saved_documents(current);cases=[]
    for name,document in documents.items():
        key=next(iter(document));bad=copy.deepcopy(document);bad.pop(key)
        cases.append({'name':name+'_missing','leaf':name,'document':bad})
        bad=copy.deepcopy(document);bad['unrecorded_extra']=False
        cases.append({'name':name+'_extra','leaf':name,'document':bad})
    for name,path,wrong in [
        ('cost',('internal_completion_valid',),False),('cost',('actual_returned_exit_code',),True),
        ('actual',('actual_outer_full_return_s',),True),('request',('source_sha256',),{}),
        ('call',('reserved_s',),3.0),('call',('cost_id',),None),
        ('summary',('process_identity','start_utc_ticks'),639268195783398393),
        ('summary',('process_identity','start_utc_ticks'),float(639268195783398392)),
        ('manifest',('process_identity','process_id'),True),('trace',('session_id',),0),
        ('trace',('full_return_s',),1.800),('closure',('known_owned_ruff','process_id'),10004.0),
        ('closure',('known_owned_ruff','observed_start_utc_ticks'),639268195783398392),
        ('closure',('known_owned_ruff','matching_owned_live'),0),
        ('decision',('original_execution_authority',),0),('decision',('original_execution_authority',),None),
        ('decision',('original_GPU_body_neural_UI',),False),('decision',('first_failure',),{}),
    ]:
        bad=copy.deepcopy(documents[name]);target=bad
        for key in path[:-1]:target=target[key]
        target[path[-1]]=wrong
        cases.append({'name':name+'_typed_'+str(len(cases)),'leaf':name,'document':bad})
    return cases


def _p18_compact_receipt_cases(current,tmp_path,semantics):
    names=('_SUITE_DEADLINE','_AUTHORITY_SCOPE','_AUTHORITY_FENCE_STATE')
    inherited=tuple(getattr(current,name) for name in names)
    try:
        """Same real constructor/complete validator; metadata fixture is not admission."""
        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(globals(),'h',current)
            isolate_compact_boundary(patch)
            for identity,count in (('e360-generous-p17',33),('e360-generous-p18',34)):
                root=tmp_path/('compact-'+identity);root.mkdir()
                contract={'pass_id':identity,'future_administration_reserved_s':95,
                    'ledger_semantics':copy.deepcopy(semantics)}
                if count==34:
                    contract['same_pending_fallback_reserved_s']=120
                    contract['ledger_semantics']['e360_passes'][identity]={'next_index':0}
                c,spec,receipt=finite_compact_contract(root,contract)
                assert len(receipt)==count
                assert ('same_pending_fallback_reserved_s' in receipt)==(count==34)
                assert current.validate_current_receipt(c,spec,receipt,root=root,entry=10.)==receipt
                for key,wrong in [('entry_perf_counter',True),('parent_pid',True),('parent_create_time',456),
                    ('registration',{}),('source_sha256',{'unrecorded':'a'*64})]:
                    bad=copy.deepcopy(receipt);bad[key]=wrong
                    with pytest.raises(ValueError):current.validate_current_receipt(c,spec,bad,root=root)
                if count==34:
                    for wrong in (False,0,-1,240,120.0):
                        bad={**receipt,'same_pending_fallback_reserved_s':wrong}
                        with pytest.raises(ValueError):current.validate_current_receipt(c,spec,bad,root=root)
                    bad=copy.deepcopy(receipt);bad.pop('same_pending_fallback_reserved_s')
                    with pytest.raises(ValueError):current.validate_current_receipt(c,spec,bad,root=root)
                    for wrong in (False,0,-1,240,120.0):
                        changed={**c,'same_pending_fallback_reserved_s':wrong}
                        with pytest.raises(ValueError):current.compact_receipt(changed,spec,10.,123,456.)
                else:
                    bad={**receipt,'same_pending_fallback_reserved_s':120}
                    with pytest.raises(ValueError):current.validate_current_receipt(c,spec,bad,root=root)
    finally:
        assert all(getattr(current,name) is original for name,original in zip(names,inherited))

def _p18_current_equivalents(current,tmp_path):
    value=current.current_repair_policy()
    assert len(current.FIXED_READERS)==18
    assert current.effective_policy(pass_id='e360-generous-p17')['effective_roles']['suite']['outer_s']==600
    frozen=current.completed_historical_p17_proof()
    assert (len(frozen['debits']),len(frozen['e360_calls']),len(frozen['e360_passes']))==(96,46,17)
    _p18_compact_receipt_cases(current,tmp_path,frozen)
    # Current complete Source freeze validator, distinct from immutable P17 history.
    actual_freeze=current.validate_p18_final_freeze()
    assert actual_freeze['own_sources_sha256']==current.own_sources()
    assert len(actual_freeze['source_copies'])==4 and len(actual_freeze['all_P18_Native_sources'])==4
    assert frozen['debits'][94]['actual_returned_exit_code']==0 and frozen['debits'][95]['actual_returned_exit_code']==1
    for name,spec in current.P17_SAVED_RECORD_SPECS.items():
        document=current.strict_json(current.read_spec(spec))
        current.validate_saved_p17_record(name,document)
        for mutation in ('missing','extra','type'):
            bad=copy.deepcopy(document)
            if mutation=='missing':bad.pop(next(iter(bad)))
            elif mutation=='extra':bad['unrecorded_extra']=False
            else:bad[next(iter(bad))]=False
            with pytest.raises(ValueError,match='actual P17 '+name+' exact typed record'):
                current.validate_saved_p17_record(name,bad)
    current.validate_current_p18_policy(value)
    for key,wrong in [('current_pass_id','e360-generous-p17'),('current_pass_id','e360-generous-p19'),
        ('primary_total_reserved_s',778),('separate_same_pending_fallback_reserved_s',False),
        ('separate_same_pending_fallback_reserved_s',0),('whole_needed_reserved_s',898),('reset',True)]:
        bad=copy.deepcopy(value);bad[key]=wrong
        with pytest.raises(ValueError):current.validate_current_p18_policy(bad)
    sem=copy.deepcopy(frozen)
    sem['e360_passes']['e360-generous-p18']={'next_index':0}
    assert current.validate_p18_fallback_balance(120,sem)==120
    for wrong in (None,False,0,-1,121,240,120.0):
        with pytest.raises(ValueError):current.validate_p18_fallback_balance(wrong,sem)
    sem['e360_calls']['current']={'pass_id':'e360-generous-p18','state':'actual_pending'}
    sem['e360_passes']['e360-generous-p18']['failed_cost']={'path':'actual-failed'}
    with pytest.raises(ValueError):current.validate_p18_fallback_balance(0,sem)
    sem['e360_calls']['current']['state']='charged'
    assert current.validate_p18_fallback_balance(0,sem)==0
    with pytest.raises(ValueError):current.validate_p18_fallback_balance(120,sem)
    for role,cap,suffix,total in (('fixture_worker',55,675,730),('suite_worker',460,170,630)):
        call={'pass_id':'e360-generous-p18','mode':'generated','call_role':role,'reserved_s':cap,
            'future_reserved_s':suffix,'state':'begun'}
        assert current.validate_p18_call_future(call)==total
        for wrong in (suffix-120,suffix+120,False):
            bad={**call,'future_reserved_s':wrong}
            with pytest.raises(ValueError):current.validate_p18_call_future(bad)
        routed=copy.deepcopy(frozen);routed['e360_passes']['e360-generous-p18']={'next_index':4}
        routed['e360_calls']['current']=call
        routed['e360_reservations']['current']={'mode':'generated','role':role.replace('_worker',''),
            'reserved_s':cap,'pending_reservations':{'suite':460} if role=='fixture_worker' else {}}
        assert current.prospective_remaining(routed,'generated')==total
    for identity in ('e360-generous-p17','e360-generous-p19','e360-original'):
        with pytest.raises(ValueError):current.check_call_request({'pass_id':identity,'mode':'generated'})
    for spec in value['exact_completed_stream_descriptors']:
        current.historical_p12_stream_bytes(spec)
        for key,wrong in [('bytes',False),('bytes',spec['bytes']+1),('sha256','0'*64),('path',spec['path']+'.other')]:
            bad=copy.deepcopy(spec);bad[key]=wrong
            with pytest.raises(ValueError):current.historical_p12_stream_bytes(bad)
    with current.fixed_authority_scope(check=lambda:None) as scope:
        current.current_repair_policy()
        assert scope.depth==0 and current.os.path.normcase(str(current.safe(current.CURRENT_REPAIR_POLICY))) in scope.descriptors
        assert not any(key[1].startswith('p18') for key in scope.values)
        scope.fence('current_P18_policy_depth0_END')
    # Same real production reader catches late empty-stream drift in metadata scratch only.
    scratch=tmp_path/'current-stream-fence';target=scratch/value['exact_completed_stream_descriptors'][1]['path']
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b'')
    _p30_copy_history_table(current,scratch)
    with current.fixed_authority_scope(root=scratch,check=lambda:None) as scope:
        current.historical_p12_stream_bytes(value['exact_completed_stream_descriptors'][1],scratch)
        target.write_bytes(b'late')
        with pytest.raises(ValueError):scope.fence('current_P18_late_stream')
        assert scope.failed and not scope.values

def _p17_current_equivalents(current):
    value=current.p17_recovery_amendment();old=current.p16_recovery_amendment()
    assert len(current.FIXED_READERS)==18 and current.FIXED_SCOPE_POLICY['max_entries']==16
    assert value['fixed_authority_scope_policy']['reader_ids']==old['fixed_authority_scope_policy']['reader_ids']+['p17_recovery_amendment']
    assert [value['fixed_authority_scope_policy'][key] for key in ('max_entries','max_text_descriptors','max_literal_json_bytes','max_aggregate_literal_json_bytes')]==[18,512,262144,2097152]
    assert value['historical_concrete_fix_relocations'][:54]==old['historical_concrete_fix_relocations']
    assert len(value['historical_concrete_fix_relocations'])==58 and len(value['preserved_activation_fields'])==28
    before=current.completed_historical_p16_proof(value=value)
    assert before==current.strict_json(current.read_spec(value['historical_P16_closed_ledger94']))
    assert (len(before['debits']),len(before['e360_calls']),len(before['e360_passes']))==(94,44,16)
    _,documents=_p17_saved_documents(current)
    for name,document in documents.items():current.validate_saved_p16_record(name,document)
    for case in _p17_saved_record_cases(current):
        with pytest.raises(ValueError,match=re.escape('actual P16 '+case['leaf']+' exact typed record')):
            current.validate_saved_p16_record(case['leaf'],case['document'])
    after=copy.deepcopy(before)
    after.update(e360_p17_recovery_activation=current.p17_recovery_event(),e360_p17_recovery_amendment=current.file_spec(current.P17_RECOVERY_AMENDMENT))
    assert current.p17_recovery_activation_allowed(before,lock_absent=True)
    assert current.p17_recovery_delta_valid(before,after) and current.current_scope_authority(live=after)
    assert not current.current_scope_authority(live=before)
    for key in current.P17_RECOVERY_FIELDS:
        bad=copy.deepcopy(after);bad.pop(key)
        with pytest.raises(ValueError):current.current_scope_authority(live=bad)
    for identity in ('e360-generous-p16','e360-generous-p14','e360-generous-p18','e360-original','e360-auto'):
        with pytest.raises(ValueError):current.check_call_request({'pass_id':identity,'mode':'generated'})
        with pytest.raises(ValueError):current.validate_p17_pass(identity,{},before['debits'])
    with pytest.raises(ValueError):current.validate_p16_pass('e360-generous-p16',{},before['debits'])
    with pytest.raises(ValueError):current.effective_policy(pass_id='e360-generous-p18')
    assert current.effective_policy(pass_id='e360-generous-p17')['suite_future_administration_reserved_s']==50
    rows=value['historical_concrete_fix_relocations'][54:58]
    hot=[row['original'] for row in rows];frozen=[row['frozen'] for row in rows]
    assert all(current.file_spec(spec['path'])['sha256']!=spec['sha256'] for spec in hot)
    actual_read=current.read_spec;hits=[]
    def guarded_actual_read(spec,root=current.ROOT):
        if spec in hot:raise AssertionError('Current P17 read stale mutable actual P16 Source4')
        if spec in frozen:hits.append(copy.deepcopy(spec))
        return actual_read(spec,root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(current,'read_spec',guarded_actual_read)
        assert current.validate_live_calls(after,after['debits']) is None
    assert all(spec in hits for spec in frozen)
    assert value['existing_human_authorization']==current.CURRENT_HUMAN_AUTONOMY
    assert value['current_human_autonomy']==current.CURRENT_HUMAN_AUTONOMY
    assert value['per_attempt_human_reapproval_required'] is False
    assert value['in_scope_next_corrected_attempt_selection_delegated'] is True
    assert value['current_only_admission_is_not_global_human_authority'] is True
    assert {name:len(document) for name,document in documents.items()}=={'cost':30,'actual':12,'request':15,'call':12,'trace':24,'summary':20,'manifest':24,'closure':23,'decision':61}
    assert len(documents['closure']['known_owned_ruff'])==7
    assert type(documents['decision']['first_failure']) is str
    assert documents['manifest']['schema']=='flylab.m3.e360.p16-ruff-capture.v1'
    assert documents['trace']['schema']=='flylab.m3.e360.p16.ROOT-genuine-collector-state.v1'
    assert documents['summary']['process_identity']=={'process_id':10004,'start_utc_ticks':639268195783398392}
    assert documents['cost']['actual_outer_full_return_s']==1.804 and documents['cost']['actual_returned_exit_code']==1
    assert before['e360_passes']['e360-generous-p16']['failed_cap_s']==3 and before['e360_passes']['e360-generous-p16']['next_index']==1
    assert documents['decision']['original_execution_authority'] is False and documents['decision']['original_GPU_body_neural_UI']==0
    for key in ('closed_ledger94','all_old93_once_debits_preserved','old_calls43_passes15_preserved','whole94_calls44_passes16'):
        assert key in documents['closure']
    assert after['debits']==before['debits'] and after['e360_calls']==before['e360_calls'] and after['e360_passes']==before['e360_passes']
    for spec in hot:
        with pytest.raises(ValueError,match='unregistered historical P16 mutable descriptor'):
            current.historical_fix_bytes(current.file_spec(spec['path']),pass_id='e360-generous-p16')



_P16_HISTORY_PROGRAM=r'''
if($p15ProbeCase.Contains('p16_history')){
    try{
        if($p15ProbeCase.Contains('deny_completed_p15')){
            Require-Prospective-P16-Pass 'e360-generous-p15' @{} $p.p16_history.authority $p.p16_history.spec @() $true
            return @{failure=$null;whole93=$false}
        }
        if($null -eq $p15ProbeCase.mutation){
            $authority=Checked-Read $p.p16_history.spec
            $whole=P16-Completed-Historical-P15-Proof $authority
            if(-not (Equal-Typed $whole $ledger)){throw 'Current P16 production proof must reach exact whole93 END'}
            return @{failure=$null;whole93=$true}
        }
        P16-Saved-P15-Record $p15ProbeCase.mutation.leaf $p15ProbeCase.mutation.document
        return @{failure=$null;whole93=$false}
    }catch{return @{failure=$_.Exception.Message;whole93=$false}}
}
'''


def _p16_native_history_payload():
    current=_P16_CURRENT;value=current.p16_recovery_amendment()
    programs=_p15_native_programs(current=current)
    program=programs['fixed'].replace(_P15_HISTORY_PROGRAM,_P16_HISTORY_PROGRAM+_P15_HISTORY_PROGRAM)
    cases=[{'name':'p16_whole93_positive','history':True,'p16_history':True,'root':str(current.ROOT),
        'ledger':value['historical_P15_closed_ledger93'],'p15_spec':current.file_spec(current.P15_RECOVERY_AMENDMENT),'mutation':None,'expected':None}]
    for case in _p16_saved_record_cases(current):
        cases.append({'name':case['name'],'history':True,'p16_history':True,'root':str(current.ROOT),'mutation':case,
            'expected':'P16 actual P15 '+case['leaf']+' exact typed record'})
    cases.append({'name':'p16_current_denies_completed_p15','history':True,'p16_history':True,'root':str(current.ROOT),
        'mutation':{},'deny_completed_p15':True,'expected':'Exact activated P16 paid-predecessor/freshSource9 only'})
    return {'program':program,'cases':cases,'authority':value,'spec':current.file_spec(current.P16_RECOVERY_AMENDMENT)}


_P16_NATIVE16_BODY=r'''
# Existing PROGRAM16 additionally invokes actual current P16 production proof and its narrow typed gates.
$p16Program=[ScriptBlock]::Create($p.p16_history.program)
foreach($p16Case in $p.p16_history.cases){
    $p15ProbeCase=$p16Case
    $observed=& $p16Program -Action callbegin -Mode generated -Request $p.p15_paid_replay.cli_request
    if($p16Case.name -ceq 'p16_whole93_positive'){
        if($null -ne $observed.failure -or -not $observed.whole93){throw ('P16 Native whole93 END: '+$observed.failure)}
    }elseif($observed.failure -cne $p16Case.expected){throw ('P16 intended same production semantic gate '+$p16Case.name+': '+$observed.failure)}
}
'''

def _p15_source_metadata_tree(destination):
    """Fixture-only copies of exact trusted Source/JSON; no original array/header/NPZ copy."""
    from pathlib import Path
    baseline=_P15_CURRENT.strict_json(_P15_CURRENT.read_spec(_P15_CURRENT.p15_recovery_amendment()['actual_P15_baseline_native_preservation']))
    specs=[*baseline['protected322_actual_baseline'],*baseline['actual_current_own9'],
        *baseline['actual_P13_Native4'],*baseline['actual_P14_Native4'],baseline['actual_P14_collector']]
    frozen_by_path={row['original']['path']:row['frozen'] for row in baseline['source_copies']}
    for spec in specs:
        if Path(spec['path']).is_absolute():continue  # Verified immutable external Source remains read-only.
        assert Path(spec['path']).suffix in ('.py','.ps1','.json','.md','.txt','.ini')
        source=_P15_CURRENT.safe(frozen_by_path.get(spec['path'],spec)['path']);raw=source.read_bytes()
        assert len(raw)==spec['bytes'] and _P15_CURRENT.sha(raw)==spec['sha256']
        target=destination/spec['path'];target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
    for row in baseline['source_copies']:
        raw=_P15_CURRENT.read_spec(row['frozen'])
        assert len(raw)==row['original']['bytes'] and _P15_CURRENT.sha(raw)==row['original']['sha256']
        target=destination/row['original']['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    return baseline


@pytest.fixture(autouse=True)
def _p15_versioned_history(request,tmp_path,_p16_frozen_p15_source):
    """Original P14 positives/denials use frozen P14 Source; current admission stays P15-only."""
    global h
    if request.node.name=='test_fixed_scope_entry_literal_aggregate_and_monotone_descriptor_bounds':
        prior=h;h=_P15_CURRENT
        try:yield
        finally:h=prior
        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(globals(),'h',_P16_CURRENT)
            _p16_scope_bounds(tmp_path/'current-p16-entry-bound')
        return
    if request.node.name not in _P15_HISTORY_CASES:
        yield
        return
    from types import ModuleType
    history_root=tmp_path/'immutable-p14-versioned-source';history_root.mkdir()
    baseline=_p15_source_metadata_tree(history_root)
    module=ModuleType('tools.e360_p14_frozen_fixture')
    module.__file__=str(history_root/_P15_CURRENT.OWNED[0]);module.__package__='tools'
    frozen=_P15_CURRENT.read_spec(next(row['frozen'] for row in baseline['source_copies'] if row['kind']=='tool')).decode('utf8')
    exec(compile(frozen,module.__file__,'exec'),module.__dict__)  # noqa: S102 -- exact frozen historical Source, inside approved suite only.
    prior=h;h=module
    try:yield
    finally:
        h=prior
    # Separate current equivalents are called after historical assertions. No extra Native process.


def _p15_corrected_history_cases(current):
    """Actual immutable leaf data; targeted mutations exercise the shared production semantic gate."""
    value=current.p15_recovery_amendment();old=current.p14_recovery_amendment()
    closure=current.strict_json(current.read_spec(value['historical_P14_independent_owned_closure']))
    decision=current.strict_json(current.read_spec(value['historical_P14_closed_decision']))
    approval=current.strict_json(current.read_spec(old['existing_human_authorization']))
    assert 'all_owned_ended' not in closure and 'original_execution_authority' not in decision
    assert type(decision['original_IO_GPU_body_neural_UI']) is bool and decision['original_IO_GPU_body_neural_UI'] is False
    leaves={'closure':(value['historical_P14_independent_owned_closure'],closure),
        'decision':(value['historical_P14_closed_decision'],decision),
        'literal':(value['previous_revalidation_amendment'],old),'approval':(old['existing_human_authorization'],approval)}
    cases=[]
    def add(name,leaf,document,gate):
        cases.append({'name':name,'spec':leaves[leaf][0],'document':document,'leaf':leaf,'gate':gate})
    owned=closure['known_owned_ruff']
    for key in owned:
        bad=copy.deepcopy(closure);bad['known_owned_ruff'].pop(key)
        add('owned_missing_'+key,'closure',bad,'actual known-owned Ruff closure7')
    bad=copy.deepcopy(closure);bad['known_owned_ruff']['all_owned_ended']=True
    add('owned_extra','closure',bad,'actual known-owned Ruff closure7')
    for key,wrong in [('process_id',True),('process_id',33356.0),('process_id',None),
        ('expected_start_utc_ticks',639267587608157544),('expected_start_utc_ticks',float(639267587608157543)),
        ('expected_start_utc_ticks',True),('observed_start_utc_ticks',639267587608157543),
        ('matching_owned_live',True),('matching_owned_live',0),('exit_observed',False),
        ('exit_observed',1),('both_streams_closed',False),('both_streams_closed',None),
        ('no_process_termination_performed',False)]:
        bad=copy.deepcopy(closure);bad['known_owned_ruff'][key]=wrong
        add('owned_'+key+'_'+str(len(cases)),'closure',bad,'actual known-owned Ruff closure7')
    for key,wrong,gate in [('schema','flylab.m3.e360.worker-owned-closure.v1','actual closure schema'),
        ('actual_actor','worker','actual closure actor'),('frozen_ledger92',value['historical_P14_inherited91'],'actual closure whole92'),
        ('whole92_calls42_passes14',False,'actual closure 92/42/14'),('all_old91_once_debits_preserved',False,'actual closure old91'),
        ('new_paid_calls',True,'actual P14 one paid Ruff'),('new_paid_seconds',1.841,'actual P14 paid Ruff wall'),
        *[(key,False,'actual closed ownership0 '+key) for key in
            ('pending_actuals','reservations','blocking_holds','unreturned_calls','original_jobs','original_reserved_s')]]:
        bad=copy.deepcopy(closure);bad[key]=wrong;add('closure_'+key,'closure',bad,gate)
    for leaf,key,gate in [('decision','original_IO_GPU_body_neural_UI','nonpaid predecessor original_IO_GPU_body_neural_UI'),
        ('literal','original_execution_authority','immutable P14 literal original authority'),
        ('approval','original_execution_authority','original_execution_authority')]:
        for wrong in (True,0,0.0,None,'false'):
            bad=copy.deepcopy(leaves[leaf][1]);bad[key]=wrong
            add(leaf+'_flag_'+str(len(cases)),leaf,bad,gate)
        bad=copy.deepcopy(leaves[leaf][1]);bad.pop(key)
        add(leaf+'_flag_missing',leaf,bad,key)
    return cases


def _p15_corrected_history_semantics(current):
    value=current.p15_recovery_amendment();old=current.p14_recovery_amendment()
    expected=current.strict_json(current.read_spec(value['historical_P14_closed_ledger92']))
    # Actual production proof returns at its END after closure -> decision -> separate authority.
    assert current.completed_historical_p14_proof(value=value)==expected
    original={'closure':current.strict_json(current.read_spec(value['historical_P14_independent_owned_closure'])),
        'decision':current.strict_json(current.read_spec(value['historical_P14_closed_decision'])),
        'literal':old,'approval':current.strict_json(current.read_spec(old['existing_human_authorization']))}
    for case in _p15_corrected_history_cases(current):
        documents=copy.deepcopy(original);documents[case['leaf']]=case['document']
        gate=case['gate']
        if case['leaf']=='approval' and 'flag_missing' not in case['name']:gate='immutable P14 human original authority'
        with pytest.raises((ValueError,KeyError),match=re.escape(gate)):
            current.validate_saved_p14_termination(documents['closure'],documents['decision'],
                documents['literal'],documents['approval'],value=value)
    assert current.completed_historical_p14_proof(value=value)==expected

_P15_HISTORY_PROGRAM=r'''
if($p15ProbeCase.Contains('history')){
    try {
        if($null -eq $p15ProbeCase.mutation){
            $literal=P15-Recovery-Literal
            $whole=P15-Completed-Historical-P14-Proof $literal.authority
            if(-not (Equal-Typed $whole $ledger)){throw 'Corrected production proof must reach exact whole92 END'}
            return @{failure=$null;whole92=$true}
        }
        $documents=@{};foreach($key in $p.p15_paid_replay.history_documents.Keys){$documents[$key]=$p.p15_paid_replay.history_documents[$key]}
        $changed=@{};foreach($key in $documents[$p15ProbeCase.mutation.leaf].Keys){$changed[$key]=$documents[$p15ProbeCase.mutation.leaf][$key]}
        $documents[$p15ProbeCase.mutation.leaf]=$changed
        if($p15ProbeCase.mutation.owned){
            $owned=@{};foreach($key in $changed.known_owned_ruff.Keys){$owned[$key]=$changed.known_owned_ruff[$key]}
            $changed.known_owned_ruff=$owned;$target=$owned
        }else{$target=$changed}
        if($p15ProbeCase.mutation.remove){$target.Remove($p15ProbeCase.mutation.key)}
        else{$target[$p15ProbeCase.mutation.key]=$p15ProbeCase.mutation.value}
        P15-Saved-P14-Termination $documents.closure $documents.decision $documents.literal $documents.approval $p.p15_paid_replay.history_authority
        return @{failure=$null;whole92=$false}
    }catch{return @{failure=$_.Exception.Message;whole92=$false}}
}
'''

_P15_HISTORY_NATIVE_BODY=r'''
# Same fragment16 and same one subprocess; actual current historical production proof.
foreach($historyCase in $p.p15_paid_replay.history_cases){
    $p15ProbeCase=$historyCase
    $program=$p15ProgramBlocks.fixed
    $observed=& $program -Action callbegin -Mode generated -Request $p.p15_paid_replay.cli_request
    if($historyCase.name -ceq 'history_positive'){
        if($null -ne $observed.failure -or -not $observed.whole92){throw ('Corrected Native historical END positive: '+$observed.failure)}
    }elseif($observed.failure -cne $historyCase.expected){
        throw ('Corrected Native intended semantic gate '+$historyCase.name+': '+$observed.failure)
    }
}
'''


def _p15_current_equivalents(current):
    _p15_corrected_history_semantics(current)
    value=current.p15_recovery_amendment();old=current.p14_recovery_amendment()
    assert len(current.FIXED_READERS)==16
    assert {k:v for k,v in current.FIXED_READERS.items() if k!='p15_recovery_amendment'}=={
        k:v for k,v in current.FIXED_READERS.items() if k in old['fixed_authority_scope_policy']['reader_ids']}
    assert value['fixed_authority_scope_policy']['reader_ids']==old['fixed_authority_scope_policy']['reader_ids']+['p15_recovery_amendment']
    assert value['historical_concrete_fix_relocations'][:46]==old['historical_concrete_fix_relocations']
    assert len(value['historical_concrete_fix_relocations'])==50 and len(value['preserved_activation_fields'])==24
    assert value['preserved_test_functions']==old['preserved_test_functions'] and len(value['preserved_test_functions'])==84
    before=current.completed_historical_p14_proof();after=copy.deepcopy(before)
    after.update(e360_p15_recovery_activation=current.p15_recovery_event(),e360_p15_recovery_amendment=current.file_spec(current.P15_RECOVERY_AMENDMENT))
    assert current.p15_recovery_activation_allowed(before,lock_absent=True)
    assert current.p15_recovery_delta_valid(before,after) and current.current_scope_authority(live=after)
    assert not current.current_scope_authority(live=before)
    passed=before['e360_passes']['e360-generous-p14']
    assert passed['failed_cost'] is None and passed['failed_cap_s']==0 and passed['next_index']==1
    assert before['debits'][91]['actual_outer_full_return_s']==1.840 and before['debits'][91]['actual_returned_exit_code']==0
    for key in current.P15_RECOVERY_FIELDS:
        bad=copy.deepcopy(after);bad.pop(key)
        with pytest.raises(ValueError):current.current_scope_authority(live=bad)
    for identity in ('e360-generous-p14','e360-generous-p13','e360-generous-p16','e360-original','e360-auto'):
        with pytest.raises(ValueError):current.check_call_request({'pass_id':identity,'mode':'generated'})
    for identity in ('e360-generous-p14','e360-generous-p16','e360-original','e360-auto'):
        with pytest.raises(ValueError):current.validate_p15_pass(identity,{},before['debits'])
    with pytest.raises(ValueError):current.effective_policy(pass_id='e360-generous-p16')
    assert current.effective_policy(pass_id='e360-generous-p15')['suite_future_administration_reserved_s']==50
    assert current.effective_policy(pass_id='e360-generous-p14')['fixture_future_administration_reserved_s']==95


def _p15_live_calls_current_relocated_history(current):
    """R1-F1: actual current validator, full typed P15 pair, unchanged whole92 and changed hot Source4."""
    value=current.p15_recovery_amendment()
    before=current.completed_historical_p14_proof()
    after=copy.deepcopy(before)
    after.update(e360_p15_recovery_activation=current.p15_recovery_event(),
        e360_p15_recovery_amendment=current.file_spec(current.P15_RECOVERY_AMENDMENT))
    assert current.current_scope_authority(live=after)
    rows=value['historical_concrete_fix_relocations'][46:50]
    assert len(rows)==4 and all(row['pass_id']=='e360-generous-p14' for row in rows)
    hot=[row['original'] for row in rows]
    frozen=[row['frozen'] for row in rows]
    # This executes after P15 Source adoption in the paid suite, not against a frozen P14 h view.
    assert all(current.file_spec(spec['path'])['sha256']!=spec['sha256'] for spec in hot)
    actual_read=current.read_spec
    frozen_hits=[]
    def guarded_actual_read(spec,root=current.ROOT):
        if spec in hot:
            raise AssertionError('R1-F1 stale P14 mutable Source4 read before historical relocation')
        if spec in frozen:frozen_hits.append(copy.deepcopy(spec))
        return actual_read(spec,root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(current,'read_spec',guarded_actual_read)
        # No admission/policy/proof functions are substituted. This is the real production validator.
        assert current.validate_live_calls(after,after['debits']) is None
    assert all(spec in frozen_hits for spec in frozen)
    assert after['debits']==before['debits'] and after['e360_calls']==before['e360_calls']
    assert after['e360_passes']==before['e360_passes']
    passed=after['e360_passes']['e360-generous-p14']
    assert passed['failed_cost'] is None and passed['failed_cap_s']==0 and passed['next_index']==1
    assert after['debits'][91]['actual_outer_full_return_s']==1.840 and after['debits'][91]['actual_returned_exit_code']==0
    # A current descriptor at an old own path must not substitute for the approved historical row.
    for spec in hot:
        current_spec=current.file_spec(spec['path'])
        with pytest.raises(ValueError,match='unregistered historical P14 mutable descriptor'):
            current.historical_fix_bytes(current_spec,pass_id='e360-generous-p14')
    assert current.completed_historical_p14_proof()==before



def _p15_slice(source,start,end):
    i=source.index(start);return source[i:source.index(end,i)]


_P15_CONTEXT_DEPENDENCIES=('Lint-Recovery-Literal','Lint-Recovery-Event','P11-Recovery-Literal','P11-Recovery-Event',
    'Assert-P12-Typed','P12-Recovery-Literal','P12-Recovery-Event','P12-Completed-Historical-P11-Proof','P12-Capture-Proof',
    'Assert-P13-Typed','P13-Recovery-Literal','P13-Recovery-Event','P13-Completed-Historical-P12-Proof',
    'P14-Recovery-Event','P11-Fresh-Source-Leaf','Read-P14-Historical-Stream')

def _p15_context_function_source(source,name):
    # Exact real version's text only. These finite functions all precede top-level administration.
    start=re.search(r'(?m)^function '+re.escape(name)+r'(?=[\s({])',source)
    assert start is not None,name
    end=re.search(r'(?m)^function ',source[start.end():])
    assert end is not None,name
    return source[start.start():start.end()+end.start()]
def _p15_native_programs(current=None):
    """Exact production PROGRAM text. Header and ordered loops are extracted, never mocked."""
    if current is None:current=_P15_CURRENT
    modern=current.safe(current.BASE+'/parent-e360-root-account.ps1').read_text('utf8')
    old=_P15_CURRENT.read_spec(_P15_CURRENT.p15_recovery_amendment()['reviewed_source_snapshots'][2]).decode('utf8')
    programs={}
    for version,source in (('fixed',modern),('collision',old)):
        header=source[:source.index("$ErrorActionPreference='Stop'")]
        definitions=(_p15_slice(source,'function Task-Path(','function Write-Json(')+
            _p15_slice(source,'function Describe(','function Equal-Typed(')+
            _p15_slice(source,'function Equal-Typed(','function Fallback-Closure-Proof(')+
            _p15_slice(source,'function P14-Collector-Proof(','function Assert-P15-Typed(' if version=='fixed' else 'function P14-Completed-Historical-P13-Proof('))
        if version=='fixed':
            definitions+=_p15_slice(source,'function Assert-P15-Typed(','function P15-Recovery-Manual-Pass(')
            definitions+=_p15_slice(source,'function P15-Collector-Proof(','function Historical-P11-Scope-Authority(')
        if version=='fixed':
            defined=set(re.findall(r'(?m)^function ([A-Za-z0-9-]+)',definitions))
            for name in _P15_CONTEXT_DEPENDENCIES:
                if name not in defined:definitions+=_p15_context_function_source(source,name)
            if current is _P30_CURRENT:
                prospective=_p15_slice(source,'function Sum-Actual(','# P30_CURRENT_ENTRY_GUARD')
                # Explicit historical invocation keeps definition-only production functions.
                # The live current-context selector is exercised later in this same PROGRAM16.
                if '\n$bancExecutionContext=Read-Execution-Context' in prospective:
                    prospective=prospective[:prospective.index('\n$bancExecutionContext=Read-Execution-Context')]
                definitions+=prospective
                definitions+=_p15_slice(source,'function Call-Policy(','function Prospective(')
                definitions+=_p15_slice(source,'function Validate-Call-State {','\nValidate-Call-State')
        cost_loop=_p15_slice(source,'$seen=@{}', '# GENEROUS_INTEGRATION_ACTIONS')
        trace_loop=_p15_slice(source,'for($i=59;', 'function Checked-Call(')
        # Explicit real production dependency in this same isolated PROGRAM context.
        # The original 37 ordered extraction slots are preserved; this dependency is versioned separately.
        context=(_P15_PROGRAM_CONTEXT if version=='fixed' else _P15_PROGRAM_CONTEXT_COLLISION)
        observe=_P15_PROGRAM_OBSERVE
        if version=='fixed' and current is _P30_CURRENT:
            context+=_P17_PROGRAM_CONTEXT+_P30_PROGRAM_CONTEXT;observe=_P17_PROGRAM_OBSERVE
        elif version=='fixed' and current is _P18_CURRENT:
            context+=_P17_PROGRAM_CONTEXT+_P18_PROGRAM_CONTEXT;observe=_P17_PROGRAM_OBSERVE
        elif version=='fixed' and current is _P17_CURRENT:
            context+=_P17_PROGRAM_CONTEXT;observe=_P17_PROGRAM_OBSERVE
        elif version=='fixed' and current is _P16_CURRENT:
            context+=_P16_PROGRAM_CONTEXT;observe=_P16_PROGRAM_OBSERVE
        if version=='fixed' and current is _P30_CURRENT:
            entry=_p15_slice(source,'# P30_CURRENT_ENTRY_GUARD','\n$ledger=Read-Json $ledgerName')
            assert entry.count('Assert-P30-Current-Policy-Entry (Describe ((P30-Compiled-Policy-Spec).path))')==1
            marker="$taskRoot=$p15ProbeCase.root;$taskBase='verification/banc-natural-gait-m3-20260919'\n"
            assert context.count(marker)==1
            context=context.replace(marker,marker+"if($p15ProbeCase.Contains('p30_history') -and $null -eq $p15ProbeCase.mutation){\n"+entry+"\n}\n",1)
        context+=_p15_slice(source,'function Checked-Historical-Fix(','$inherited59=')
        programs[version]=header+definitions+context+(_P15_HISTORY_PROGRAM if version=='fixed' else '')+'\ntry {\n'+cost_loop+trace_loop+observe
    return programs


_P15_PROGRAM_CONTEXT_COLLISION=r'''
$ErrorActionPreference='Stop'
$entryCLIRequest=$Request
$taskRoot=$p15ProbeCase.root;$taskBase='verification/banc-natural-gait-m3-20260919'
$ledger=Checked-Read $p15ProbeCase.ledger
$reg=Read-Json ($taskBase+'/parent-e360-reflected-registration.json')
$inherited=Checked-Read $reg.inherited_cost_ledger
$p15HistoricalLedger=$ledger
$oldRevalidationSpec=$ledger.e360_revalidation_amendment
$adminRevalidationSpec=$ledger.e360_active_revalidation_amendment
$sourceRevalidationSpec=$ledger.e360_source_recovery_amendment
$ruffRevalidationSpec=$ledger.e360_ruff_recovery_amendment
$scopeRevalidationSpec=$ledger.e360_authority_scope_amendment
$budgetSpec=$ledger.e360_execution_budget_amendment
$suiteSpec=$ledger.e360_suite_recovery_amendment
$lintSpec=$ledger.e360_lint_recovery_amendment
$p11Spec=$ledger.e360_p11_recovery_amendment
$p12Spec=$ledger.e360_p12_recovery_amendment
$p13Spec=$ledger.e360_p13_recovery_amendment
$p14Spec=$ledger.e360_p14_recovery_amendment
$p15Spec=$p15ProbeCase.p15_spec
$p11Active=$ledger.Contains('e360_p11_recovery_activation')
$p13Active=$ledger.Contains('e360_p13_recovery_activation')
$p14Active=$ledger.Contains('e360_p14_recovery_activation')
$p15Active=$false
'''

_P15_PROGRAM_CONTEXT=r'''
$ErrorActionPreference='Stop'
$entryCLIRequest=$Request
$taskRoot=$p15ProbeCase.root;$taskBase='verification/banc-natural-gait-m3-20260919'
if(-not $p15ProbeCase.Contains('history') -or $null -eq $p15ProbeCase.mutation){
$ledger=Checked-Read $p15ProbeCase.ledger
$reg=Read-Json ($taskBase+'/parent-e360-reflected-registration.json')
$inherited=Checked-Read $reg.inherited_cost_ledger
$p15HistoricalLedger=$ledger
$oldRevalidationSpec=$ledger.e360_revalidation_amendment
$adminRevalidationSpec=$ledger.e360_active_revalidation_amendment
$sourceRevalidationSpec=$ledger.e360_source_recovery_amendment
$ruffRevalidationSpec=$ledger.e360_ruff_recovery_amendment
$scopeRevalidationSpec=$ledger.e360_authority_scope_amendment
$budgetSpec=$ledger.e360_execution_budget_amendment
$suiteSpec=$ledger.e360_suite_recovery_amendment
$lintSpec=$ledger.e360_lint_recovery_amendment
$p11Spec=$ledger.e360_p11_recovery_amendment
$p12Spec=$ledger.e360_p12_recovery_amendment
$p13Spec=$ledger.e360_p13_recovery_amendment
$p14Spec=$ledger.e360_p14_recovery_amendment
$p15Spec=$p15ProbeCase.p15_spec
$p11Active=$ledger.Contains('e360_p11_recovery_activation')
$p13Active=$ledger.Contains('e360_p13_recovery_activation')
$p14Active=$ledger.Contains('e360_p14_recovery_activation')
$p15Active=$false
}
'''
_P15_PROGRAM_OBSERVE=r'''
    @{failure=$null;RequestValue=$Request;RequestType=$Request.GetType().FullName;entryRequest=$entryCLIRequest;
      metadata_mapping=($p15PriorAdmittedRequest -is [Collections.IDictionary]);collector_valid=$collectorProof.valid}
} catch {
    @{failure=$_.Exception.Message;RequestValue=$Request;RequestType=$Request.GetType().FullName;entryRequest=$entryCLIRequest;
      metadata_mapping=($p15PriorAdmittedRequest -is [Collections.IDictionary]);collector_valid=$null}
}
'''


_P17_HISTORY_PROGRAM=r'''
if($p15ProbeCase.Contains('p17_history')){
    try{
        if($p15ProbeCase.Contains('deny_completed_p16')){
            Require-Prospective-P17-Pass 'e360-generous-p16' @{} $p.p17_history.authority $p.p17_history.spec @() $true
            return @{failure=$null;whole94=$false}
        }
        if($null -eq $p15ProbeCase.mutation){
            $authority=Checked-Read $p.p17_history.spec
            $whole=P17-Completed-Historical-P16-Proof $authority
            if(-not (Equal-Typed $whole $ledger)){throw 'Current P17 production proof must reach exact whole94 END'}
            return @{failure=$null;whole94=$true}
        }
        P17-Saved-P16-Record $p15ProbeCase.mutation.leaf $p15ProbeCase.mutation.document
        return @{failure=$null;whole94=$false}
    }catch{return @{failure=$_.Exception.Message;whole94=$false}}
}
'''

_P30_PROGRAM_CONTEXT=r'''
if(-not $p15ProbeCase.Contains('history') -or $null -eq $p15ProbeCase.mutation){
    $p30HistoricalLedger=Checked-Read $p.p30_history.authority.closed_predecessor_ledger
    $p30Spec=$p.p30_history.spec;$p30Active=$ledger.Contains('e360_p30_recovery_activation')
}
'''
_P30_ROSTER_BINDINGS=r'''
# Explicit stored production bindings, using the already proven historical ledger;
# no target action, whole-history reconstruction, or substituted validator.
$p30HistoricalLedger=$ledger
$extensionSpec=$ledger.e360_resource_extension
$historicalExtension=Checked-Read $extensionSpec
$effectiveExtension=$historicalExtension|ConvertTo-Json -Depth 100 -Compress|ConvertFrom-Json -AsHashtable
$budget=Checked-Read $budgetSpec;$budgetActive=$ledger.Contains('e360_execution_budget_activation')
$suiteRecovery=Checked-Read $suiteSpec;$suiteActive=$ledger.Contains('e360_suite_recovery_activation')
$lintRecovery=Checked-Read $lintSpec;$lintActive=$ledger.Contains('e360_lint_recovery_activation')
$p11Recovery=Checked-Read $p11Spec;$p11Active=$ledger.Contains('e360_p11_recovery_activation')
$p12Recovery=Checked-Read $p12Spec;$p12Active=$ledger.Contains('e360_p12_recovery_activation')
$p13Recovery=Checked-Read $p13Spec;$p13Active=$ledger.Contains('e360_p13_recovery_activation')
$p14Recovery=Checked-Read $p14Spec;$p14Active=$ledger.Contains('e360_p14_recovery_activation')
$p15Recovery=Checked-Read $ledger.e360_p15_recovery_amendment;$p15Active=$ledger.Contains('e360_p15_recovery_activation')
$p16Recovery=Checked-Read $ledger.e360_p16_recovery_amendment;$p16Active=$ledger.Contains('e360_p16_recovery_activation')
$p17Recovery=Checked-Read $ledger.e360_p17_recovery_amendment;$p17Active=$ledger.Contains('e360_p17_recovery_activation')
$p30Recovery=(P30-Recovery-Literal).authority
$effectiveExtension.helper_limits=$p17Recovery.effective_helper_limits
$effectiveExtension.effective_roles=$p17Recovery.effective_roles
$effectiveExtension.scanner_limits=$p17Recovery.scanner_limits
$effectiveExtension.successful_suffix_after_s=$p17Recovery.successful_suffix_after_s
$effectiveExtension.generated_call_order=$p17Recovery.generated_call_order
$effectiveRevalidationSpec=$p17Spec
'''
_P30_HISTORY_PROGRAM=r'''
if($p15ProbeCase.Contains('p30_history')){
    $bancExecutionContext=$null;$executionPassID='e360-generous-p30'
    $activationChecked=$false;$activationAllowed=$null;$entryGuardChecked=$false
    try{
        if($null -eq $p15ProbeCase.mutation){
            $whole=P30-Completed-Historical-P29-Proof;$freeze=P30-Final-Freeze-Proof
            $activationAllowed=P30-Recovery-Activation-Allowed $whole $whole (P30-Recovery-Literal).authority $true
            $activationChecked=$true
            if($activationAllowed -isnot [bool] -or -not $activationAllowed){throw 'Actual closed105 production activation predicate must return Boolean true'}
            # Exercise the actual selector used immediately before activation commit.
            $scopeCandidate=$whole|ConvertTo-Json -Depth 100 -Compress|ConvertFrom-Json -AsHashtable
            $scopeCandidate.e360_p30_recovery_activation=P30-Recovery-Event
            $scopeCandidate.e360_p30_recovery_amendment=P30-Compiled-Policy-Spec
            $scopeAllowed=Current-Scope-Authority $scopeCandidate
            if($scopeAllowed -isnot [bool] -or -not $scopeAllowed){throw 'Actual closed109 plus current pair must have scope authority'}
            . ([ScriptBlock]::Create($p.p30_history.roster_bindings))
            . ([ScriptBlock]::Create($p.p30_history.cost_replay_loop))
            $p30PaidProofs=[Collections.Generic.List[object]]::new()
            . ([ScriptBlock]::Create($p.p30_history.paid_replay_observed_loop))
            $p30StoredPolicies=@{}
            foreach($id in @('e360-generous-p25-ruff','e360-generous-p25-scan','e360-generous-p27-ruff','e360-generous-p29-ruff')){
                $admitted=Checked-Read $ledger.e360_calls[$id].request
                $p30StoredPolicies[$id]=Call-Policy $admitted
            }
            Validate-Call-State
            $rosterEnd=$true
            Assert-P17-Typed $whole $ledger 'actual P30 direct whole109 END'
            return @{failure=$null;whole109=$true;freeze=$freeze;activation_checked=$activationChecked;activation_allowed=$activationAllowed;
                stored_p25_policies=$p30StoredPolicies;paid_row_proofs=$p30PaidProofs.ToArray();roster_END=$rosterEnd}
        }
        $m=$p15ProbeCase.mutation
        if($m.Contains('current_entry')){
            $entryGuardChecked=$true
            $null=Assert-P30-Current-Policy-Entry $m.current_entry
            return @{failure=$null;whole109=$false;entry_guard_checked=$entryGuardChecked}
        }
        if($m.Contains('bookkeeping_literal')){
            . ([ScriptBlock]::Create($p.p30_history.trace_function))
            $entryGuardChecked=$true
            Assert-E360P30CurrentPolicyTuple $m.bookkeeping_literal
            return @{failure=$null;whole109=$false;entry_guard_checked=$entryGuardChecked}
        }
        if($m.Contains('local_P25')){
            # Local negatives reuse the same actual functions/row slices, not whole history.
            $ledger=Checked-Read $p.p30_history.authority.closed_predecessor_ledger
            $p30HistoricalLedger=$ledger
            if($m.Contains('policy_request')){$null=Call-Policy $m.policy_request}
            elseif($m.Contains('paid_row')){
                $i=$m.paid_row.index;$row=$m.paid_row.cost;$actual=$m.paid_row.actual
                Assert-Cost-Authority $i $row $actual
                $collectorProof=$null;$p17PriorAdmittedRequest=$null;$p17PriorCall=$null
                . ([ScriptBlock]::Create($(if($i -eq 108){$p.p30_history.P29_paid_row_body}elseif($i -eq 107){$p.p30_history.P27_paid_row_body}else{$p.p30_history.P25_paid_row_body})))
                . ([ScriptBlock]::Create($p.p30_history.collector_validity_body))
            }elseif($m.Contains('roster_call')){
                $id=$m.roster_call_id;$ledger.e360_calls[$id]=$m.roster_call
                $extensionSpec=$ledger.e360_resource_extension;$seen=@{}
                foreach($debit in $ledger.debits){$seen[$debit.cost_id]=$true}
                $p30Spec=$p.p30_history.spec
                . ([ScriptBlock]::Create($p.p30_history.roster_call_body))
            }
            return @{failure=$null;whole109=$false}
        }
        if($m.Contains('activation_state')){
            $before=$m.activation_state|ConvertTo-Json -Depth 100 -Compress|ConvertFrom-Json -AsHashtable
            try{
                $activationAllowed=P30-Recovery-Activation-Allowed $m.activation_state $m.activation_frozen (P30-Recovery-Literal).authority $m.lock_absent
                $activationChecked=$true
            }finally{Assert-P17-Typed $m.activation_state $before 'Actual activation predicate leaves ledger unchanged'}
            if($activationAllowed -isnot [bool]){throw 'Actual activation predicate must return Boolean'}
            return @{failure=$null;whole109=$false;activation_checked=$activationChecked;activation_allowed=$activationAllowed}
        }
        elseif($m.Contains('cost_gate')){Assert-Cost-Authority $m.cost_gate.index $m.cost_gate.cost $m.cost_gate.actual}
        elseif($m.Contains('trace_gate')){
            . ([ScriptBlock]::Create($p.p30_history.trace_function));$b=$taskBase
            Assert-E360P30CollectorTrace $m.trace_gate.trace $m.trace_gate.path $m.trace_gate.collector $m.trace_gate.literal
        }
        elseif($m.Contains('dispatch_state')){
            . ([ScriptBlock]::Create($p.p30_history.dispatch_function))
            $null=Assert-E360P30CurrentActivation $m.dispatch_state $p.p30_history.spec $p.p30_history.authority.human_authorization
        }
        elseif($m.Contains('capture_state')){
            . ([ScriptBlock]::Create($p.p30_history.capture_functions))
            $root=$taskRoot;$base=$taskBase+'/';$utf8=[Text.UTF8Encoding]::new($false,$true)
            Assert-CaptureP30Activation $m.capture_state $p.p30_history.spec $p.p30_history.authority.human_authorization $p.p30_history.current_freeze
        }
        elseif($m.Contains('historical_objects')){
            Assert-P17-Typed (Describe ($taskBase+'/parent-e360-current-generated-repair-policy.json')) $p.p30_history.spec 'Actual slot is P30 during P19 reconstruction'
            $objects=P19-Activation-Objects $m.allocation_spec
            Assert-P17-Typed $objects $m.historical_objects 'Native historical P19 matches actual Python frozen event/grant'
        }
        elseif($m.Contains('fresh_request')){
            $before=$m.fresh_state|ConvertTo-Json -Depth 100 -Compress|ConvertFrom-Json -AsHashtable
            $authority=(P30-Recovery-Literal).authority
            try{Require-P30-Fresh-Pass $m.fresh_request $m.fresh_state $authority $p.p30_history.spec}
            finally{Assert-P17-Typed $m.fresh_state $before 'Fresh production guard must not mutate ledger before admission'}
        }
        elseif($m.Contains('historical_p20_pair')){if(-not (Historical-P20-Activation-Pair $m.historical_p20_pair)){throw 'Actual frozen P20 pair required'}}
        elseif($m.Contains('pair_state')){$null=Current-Scope-Authority $m.pair_state}
        elseif($m.Contains('delta_before')){if(-not (P30-Recovery-Delta-Valid $m.delta_before $m.delta_after $p.p30_history.event $p.p30_history.spec)){throw 'P30 exact pair2 delta denied'}}
        elseif($m.Contains('primary_proposal')){$null=P30-Primary-Future $m.primary_proposal}
        elseif($m.Contains('allocation')){Assert-P19-Allocation-Document $m.allocation $m.expected}
        elseif($m.Contains('state')){
            $before=$m.state|ConvertTo-Json -Depth 100 -Compress|ConvertFrom-Json -AsHashtable
            $budget=Current-Budget-Authority $m.state
            Assert-P17-Typed $m.state $before 'current Native budget selector cannot mutate historical state'
            return @{failure=$null;whole109=$false;budget=$budget;state=$m.state}
        }
        elseif($m.Contains('policy')){P30-Validate-Policy $m.policy}
        elseif($m.Contains('contract')){
            $payload=P30-Contract-Payload $m.path $m.contract
            if($payload.Length -le 262144 -or $payload.Length -gt 524288){throw 'Actual Native current >256KiB compact payload'}
            Write-P30-Contract $m.path $m.contract
            Assert-P17-Typed (Read-P30-Contract $m.path) $m.contract 'actual Native current publication roundtrip'
        }
        return @{failure=$null;whole109=$false}
    }catch{return @{failure=$_.Exception.Message;whole109=$false;activation_checked=$activationChecked;activation_allowed=$activationAllowed;entry_guard_checked=$entryGuardChecked}}
}
'''
def _p30_native_history_payload(tmp_path):
    current=_P30_CURRENT;value=current.strict_json(current.read_spec(current.P30_FROZEN_POLICY_SPEC,owner='historical'))
    program=_p15_native_programs(current=current)['fixed']
    program=program.replace(_P15_HISTORY_PROGRAM,_P30_HISTORY_PROGRAM+_P15_HISTORY_PROGRAM)
    state=current.strict_json(current.read_spec(current.EXECUTION_ANCHOR_SPEC,owner='historical'))
    cases=[{'name':'p30_whole109_positive','history':True,'p30_history':True,'root':str(current.ROOT),
        'ledger':value['closed_predecessor_ledger'],'p15_spec':current.file_spec(current.P15_RECOVERY_AMENDMENT),'mutation':None,'reject':False}]
    # Exact production early guard and canonical bookkeeping tuple; no history per mutation.
    for gate in ('current_entry','bookkeeping_literal'):
        cases.append({'name':'p30_'+gate+'_positive','history':True,'p30_history':True,'root':str(current.ROOT),
            'mutation':{gate:copy.deepcopy(current.CURRENT_REPAIR_SPEC)},'reject':False,'entry_guard_required':True})
        for key,wrong in (('path',current.BASE+'/parent-e360-p28-root-final-repair-policy.json'),('path','C:/foreign/current.json'),
            ('bytes',False),('bytes',str(current.CURRENT_REPAIR_SPEC['bytes'])),('bytes',current.CURRENT_REPAIR_SPEC['bytes']+1),
            ('bytes',0),('sha256','0'*64),('sha256',None),('sha256',current.P27_ORIGINAL_POLICY_SPEC['sha256'])):
            bad=copy.deepcopy(current.CURRENT_REPAIR_SPEC);bad[key]=wrong
            cases.append({'name':'p30_'+gate+'_reject_'+str(len(cases)),'history':True,'p30_history':True,'root':str(current.ROOT),
                'mutation':{gate:bad},'reject':True,'entry_guard_required':True,
                'error_contains':'Compiled canonical current P30 policy descriptor' if gate=='current_entry' else 'Compiled P30 bookkeeping policy'})
    closed=current.completed_historical_p29_proof()
    # The same real predicate returns false normally; no new history reconstruction or mirror guard.
    for label in ('count108','lock_held','already_current_pair'):
        activation_state=copy.deepcopy(closed);lock_absent=True
        if label=='count108':activation_state['debits']=activation_state['debits'][:108]
        elif label=='lock_held':lock_absent=False
        else:
            for key in current.CURRENT_P30_FIELDS:activation_state[key]=copy.deepcopy(state[key])
        cases.append({'name':'p30_activation_reject_'+label,'history':True,'p30_history':True,'root':str(current.ROOT),
            'mutation':{'activation_state':activation_state,'activation_frozen':closed,'lock_absent':lock_absent},
            'reject':True,'expected_activation_allowed':False})
    for label,snapshot,limits in (('closed107',closed,(1700,1589)),('complete_current',state,(1700,1589))):
        cases.append({'name':'p30_budget_'+label,'history':True,'p30_history':True,'root':str(current.ROOT),
            'mutation':{'state':snapshot},'reject':False,'expected_budget':{'active':True,'shared':limits[0],'local':limits[1]},
            'preserved_historical':{key:copy.deepcopy(snapshot[key]) for key in ('e360_p18_recovery_activation','e360_p18_recovery_amendment','e360_resource_grant','e360_suite_recovery_resource_grant','e360_p19_recovery_activation','e360_p19_recovery_amendment','e360_p19_recovery_resource_grant','e360_p20_recovery_activation','e360_p20_recovery_amendment','e360_p21_recovery_activation','e360_p21_recovery_amendment','e360_p22_recovery_activation','e360_p22_recovery_amendment','e360_p23_recovery_activation','e360_p23_recovery_amendment','e360_p25_recovery_activation','e360_p25_recovery_amendment')}})
    for key in ('e360_p18_recovery_activation','e360_p18_recovery_amendment','e360_resource_grant','e360_suite_recovery_resource_grant','e360_p19_recovery_activation','e360_p19_recovery_amendment','e360_p19_recovery_resource_grant','e360_p20_recovery_activation','e360_p20_recovery_amendment','e360_p21_recovery_activation','e360_p21_recovery_amendment','e360_p22_recovery_activation','e360_p22_recovery_amendment','e360_p23_recovery_activation','e360_p23_recovery_amendment','e360_p25_recovery_activation','e360_p25_recovery_amendment'):
        bad=copy.deepcopy(state);bad.pop(key)
        cases.append({'name':'p30_budget_missing_historical_'+key,'history':True,'p30_history':True,'root':str(current.ROOT),'mutation':{'state':bad},'reject':True})
    for field,wrong in (('new_generated_grant_s',False),('new_generated_grant_s',300),
        ('revalidation_amendment',{}),('inherited100',{}),('as_executed_source_freeze',{})):
        bad=copy.deepcopy(closed);bad['e360_p20_recovery_activation'][field]=wrong
        cases.append({'name':'p30_frozen_P20_pair_'+str(len(cases)),'history':True,'p30_history':True,
            'root':str(current.ROOT),'mutation':{'historical_p20_pair':bad},'reject':True})
    cases.append({'name':'p30_frozen_P20_pair_positive','history':True,'p30_history':True,
        'root':str(current.ROOT),'mutation':{'historical_p20_pair':closed},'reject':False})
    missing_pair=copy.deepcopy(state)
    for key in current.HISTORICAL_P20_FIELDS:missing_pair.pop(key)
    cases.append({'name':'p30_budget_missing_both_frozen_P20_fields','history':True,'p30_history':True,
        'root':str(current.ROOT),'mutation':{'state':missing_pair},'reject':True})
    expected=current.p19_expected_allocation()
    for key,wrong in [('generated_new_grant_s',False),('generated_new_grant_s',0),('generated_new_grant_s',600),('closed_predecessor_ledger',{}),('policy',{}),('human_authorization',{}),('ROOT_review_cycle_complete',False),('reset',True)]:
        bad=copy.deepcopy(expected);bad[key]=wrong
        cases.append({'name':'p30_allocation_'+key+'_'+str(len(cases)),'history':True,'p30_history':True,'root':str(current.ROOT),'mutation':{'allocation':bad,'expected':expected},'reject':True})
    for role,primary in (('fixture',125),('suite',65)):
        proposal={'pass_id':'e360-generous-p30','mode':'generated','role':role,'future_administration_reserved_s':primary}
        for wrong in (primary,None,False,0,-1,primary+120,float(primary),str(primary)):
            bad={**proposal,'future_administration_reserved_s':wrong}
            cases.append({'name':'p30_primary_'+str(len(cases)),'history':True,'p30_history':True,'root':str(current.ROOT),
                'mutation':{'primary_proposal':bad},'reject':type(wrong) is not int or wrong!=primary})
    for key in current.CURRENT_P30_FIELDS:
        bad=copy.deepcopy(state);bad.pop(key)
        cases.append({'name':'p30_partial_'+key,'history':True,'p30_history':True,'root':str(current.ROOT),'mutation':{'pair_state':bad},'reject':True})
    bad=copy.deepcopy(state);bad['e360_p30_recovery_resource_grant']={'new_generated_grant_s':300}
    cases.append({'name':'p30_new_grant_denied','history':True,'p30_history':True,'root':str(current.ROOT),'mutation':{'pair_state':bad},'reject':True})
    for wrong in (False,300,0.0,-1):
        bad=copy.deepcopy(state);bad['e360_p30_recovery_activation']['new_generated_grant_s']=wrong
        cases.append({'name':'p30_wrong_grant_'+str(len(cases)),'history':True,'p30_history':True,'root':str(current.ROOT),'mutation':{'pair_state':bad},'reject':True})
    before=closed;after=copy.deepcopy(closed)
    after.update(e360_p30_recovery_activation=current.p30_recovery_event(),e360_p30_recovery_amendment=copy.deepcopy(current.CURRENT_REPAIR_SPEC))
    for field,wrong in ((None,None),('generated_shared_cumulative_limit_s',2000),('e360_generated_local_limit_s',1889),('generated_shared_remaining_s',1347.6059844),('generated_shared_cumulative_used_s',0),('e360_generated_local_used_s',0)):
        bad=copy.deepcopy(after)
        if field is not None:bad[field]=wrong
        cases.append({'name':'p30_delta_'+str(field),'history':True,'p30_history':True,'root':str(current.ROOT),'mutation':{'delta_before':before,'delta_after':bad},'reject':field is not None})
    contract={'pass_id':'e360-generous-p30','mode':'generated','role':'fixture','ledger_semantics':state,
        'ledger_debits':copy.deepcopy(state['debits']),'source_sha256':{f'growth/source-{i:04}.py':'a'*64 for i in range(2300)},
        'future_administration_reserved_s':125,'same_pending_fallback_reserved_s':120}
    path='p30-current-growth-contract.json'
    # Scratch root only for this publication; actual current whole proof retains real ROOT.
    cases.append({'name':'p30_native_growth','history':True,'p30_history':True,'root':str(tmp_path),
        'mutation':{'contract':contract,'path':path},'reject':False})
    cases.append({'name':'p30_native_existing','history':True,'p30_history':True,'root':str(tmp_path),
        'mutation':{'contract':contract,'path':path},'reject':True,'error_contains':'cap=524288'})
    over=copy.deepcopy(contract);over['source_sha256'].update({f'overflow/source-{i:04}.py':'b'*64 for i in range(6000)})
    cases.append({'name':'p30_native_over','history':True,'p30_history':True,'root':str(tmp_path),
        'mutation':{'contract':over,'path':'p30-over-contract.json'},'reject':True,'error_contains':'exists=false'})
    native_text=(current.ROOT/current.BASE/'parent-e360-p30-ROOT-dispatch-adapter.ps1').read_text(encoding='utf8')
    dispatch_function=_p15_context_function_source(native_text,'Pin')+re.search(r'(?ms)^function Assert-E360P30CurrentActivation\([^\n]*\)\{\r?\n.*?^\}',native_text).group(0)
    cases.append({'name':'p30_dispatch_complete','history':True,'p30_history':True,'root':str(current.ROOT),
        'mutation':{'dispatch_state':copy.deepcopy(state)},'reject':False})
    for field in current.CURRENT_P30_FIELDS:
        bad=copy.deepcopy(state);bad.pop(field)
        cases.append({'name':'p30_dispatch_missing_'+field,'history':True,'p30_history':True,'root':str(current.ROOT),
            'mutation':{'dispatch_state':bad},'reject':True})
    for field in ('e360_p30_recovery_activation',):
        for wrong in (300,False,0.0):
            bad=copy.deepcopy(state);bad[field]['new_generated_grant_s']=wrong
            cases.append({'name':'p30_dispatch_grant_'+str(len(cases)),'history':True,'p30_history':True,'root':str(current.ROOT),
                'mutation':{'dispatch_state':bad},'reject':True})
    for field in ('revalidation_amendment','human_authorization','inherited109','as_executed_source_freeze'):
        bad=copy.deepcopy(state);bad['e360_p30_recovery_activation'][field]['sha256']='0'*64
        cases.append({'name':'p30_dispatch_binding_'+field,'history':True,'p30_history':True,'root':str(current.ROOT),
            'mutation':{'dispatch_state':bad},'reject':True})
    # Same PROGRAM16: actual production capture gate, without capture/process start.
    capture_text=(current.ROOT/current.BASE/'parent-e360-p30-capture-proposed.ps1').read_text(encoding='utf8')
    capture_functions=''
    for function_name in ('Assert-Capture','Test-CaptureEqual','Get-CapturePath','Get-CaptureDescriptor','Read-CaptureJson','Assert-CaptureP30Activation'):
        match=re.search(r'(?ms)^    function '+re.escape(function_name)+r'(?=[\s({]).*?^    }',capture_text)
        assert match is not None,function_name
        capture_functions+=match.group(0)+'\n'
    current_freeze=current.file_spec(current.BASE+'/parent-e360-p30-root-final-freeze.json')
    cases.append({'name':'p30_capture_pair2_complete','history':True,'p30_history':True,'root':str(current.ROOT),
        'mutation':{'capture_state':copy.deepcopy(state)},'reject':False})
    for key in current.CURRENT_P30_FIELDS:
        bad=copy.deepcopy(state);bad.pop(key)
        cases.append({'name':'p30_capture_missing_'+key,'history':True,'p30_history':True,'root':str(current.ROOT),
            'mutation':{'capture_state':bad},'reject':True})
    bad=copy.deepcopy(state);bad['e360_p30_recovery_resource_grant']={'new_generated_grant_s':300}
    cases.append({'name':'p30_capture_added_grant','history':True,'p30_history':True,'root':str(current.ROOT),
        'mutation':{'capture_state':bad},'reject':True})
    for wrong in (False,300,0.0,-1):
        bad=copy.deepcopy(state);bad['e360_p30_recovery_activation']['new_generated_grant_s']=wrong
        cases.append({'name':'p30_capture_typed_grant_'+str(len(cases)),'history':True,'p30_history':True,'root':str(current.ROOT),
            'mutation':{'capture_state':bad},'reject':True})
    for owner,key in (('e360_p30_recovery_activation','revalidation_amendment'),('e360_p30_recovery_activation','inherited109'),
        ('e360_p30_recovery_activation','as_executed_source_freeze'),('e360_p30_recovery_activation','resource_extension'),
        ('e360_p19_recovery_activation','revalidation_amendment'),('e360_p19_recovery_activation','ROOT_resource_allocation'),
        ('e360_p19_recovery_resource_grant','policy'),('e360_p19_recovery_resource_grant','allocation')):
        bad=copy.deepcopy(state);bad[owner][key]['sha256']='0'*64
        cases.append({'name':'p30_capture_binding_'+owner+'_'+key,'history':True,'p30_history':True,'root':str(current.ROOT),
            'mutation':{'capture_state':bad},'reject':True})
    allocation_spec=current.file_spec(current.historical_p19_repair_policy()['allocation_record_path'])
    old_event,old_grant=current.p19_activation_objects(allocation_spec)
    assert old_event==closed['e360_p19_recovery_activation'] and old_grant==closed['e360_p19_recovery_resource_grant']
    assert old_event['revalidation_amendment']==current.P19_ORIGINAL_POLICY_SPEC!=current.CURRENT_REPAIR_SPEC
    cases.append({'name':'p30_slot_historical_objects_match','history':True,'p30_history':True,'root':str(current.ROOT),
        'mutation':{'historical_objects':{'event':old_event,'grant':old_grant},'allocation_spec':allocation_spec},'reject':False})
    for label,snapshot in (('closed100',closed),('complete_current',state)):
        for owner,key in (('e360_p19_recovery_activation','revalidation_amendment'),('e360_p19_recovery_resource_grant','policy'),
            ('e360_p19_recovery_activation','ROOT_resource_allocation'),('e360_p19_recovery_resource_grant','allocation')):
            bad=copy.deepcopy(snapshot);bad[owner][key]['sha256']='0'*64
            cases.append({'name':'p30_slot_historical_binding_'+label+'_'+owner+'_'+key,'history':True,'p30_history':True,
                'root':str(current.ROOT),'mutation':{'state':bad},'reject':True})
    # Real fresh-pass production guard runs before any ledger mutation; reuse the same PROGRAM16.
    freeze=current.validate_p30_final_freeze()
    fresh_state=copy.deepcopy(closed)
    fresh_state.update(e360_p30_recovery_activation=copy.deepcopy(state['e360_p30_recovery_activation']),
        e360_p30_recovery_amendment=copy.deepcopy(state['e360_p30_recovery_amendment']))
    concrete=[current.file_spec(value['ROOT_reflected_design']['path']),
        *(current.file_spec(row['current']['path']) for row in freeze['source_copies']),current_freeze,
        *freeze['all_P30_Native_sources'],freeze['collector_source']]
    fresh_request={'pass_id':'e360-generous-p30','call_role':'ruff','mode':'generated',
        'paid_predecessor_success':copy.deepcopy(value['paid_predecessor_success']),'unpaid_Source_admission_failure':copy.deepcopy(value['unpaid_Source_admission_failure']),'admission_kind':value['current_admission_kind'],'predecessor_success_cap_s':3,'concrete_fix':concrete,
        'source_sha256':copy.deepcopy(freeze['own_sources_sha256'])}
    cases.append({'name':'p30_fresh_actual_P23_Ruff','history':True,'p30_history':True,'root':str(current.ROOT),
        'mutation':{'fresh_request':fresh_request,'fresh_state':fresh_state},'reject':False})
    for label in ('stale_P22','missing_failure','tampered_own_source','missing_concrete_source','missing_P25','tampered_prior_cap','tampered_prior_next','tampered_prior_cost'):
        bad_request=copy.deepcopy(fresh_request);bad_state=copy.deepcopy(fresh_state)
        if label=='stale_P22':bad_request['paid_predecessor_success']=copy.deepcopy(current.historical_p25_repair_policy()['predecessor_failed_ruff_cost'])
        elif label=='missing_failure':bad_request.pop('paid_predecessor_success')
        elif label=='tampered_own_source':bad_request['source_sha256'][current.OWNED[0]]='0'*64
        elif label=='missing_concrete_source':bad_request['concrete_fix']=[spec for spec in bad_request['concrete_fix'] if spec['path']!=current.OWNED[0]]
        elif label=='missing_P25':bad_state['e360_passes'].pop('e360-generous-p29')
        elif label=='tampered_prior_cap':bad_state['e360_passes']['e360-generous-p29']['failed_cap_s']=25
        elif label=='tampered_prior_next':bad_state['e360_passes']['e360-generous-p29']['next_index']=3
        else:bad_state['e360_passes']['e360-generous-p29']['failed_cost']=copy.deepcopy(current.historical_p25_repair_policy()['predecessor_failed_ruff_cost'])
        cases.append({'name':'p30_fresh_reject_'+label,'history':True,'p30_history':True,'root':str(current.ROOT),
            'mutation':{'fresh_request':bad_request,'fresh_state':bad_state},'reject':True})
    for key,wrong in (('unpaid_Source_admission_failure',{}),('admission_kind','failed-paid-predecessor'),
        ('predecessor_success_cap_s',False),('predecessor_success_cap_s',0),('paid_predecessor_success',current.P27_ACTUAL_RUFF_RECORD_SPECS['full-return.json'])):
        bad_request=copy.deepcopy(fresh_request);bad_request[key]=copy.deepcopy(wrong)
        cases.append({'name':'p30_fresh_transition_'+key+'_'+str(len(cases)),'history':True,'p30_history':True,'root':str(current.ROOT),
            'mutation':{'fresh_request':bad_request,'fresh_state':fresh_state},'reject':True})
    # The complete actual cost/replay loops precede real policy/roster END.
    account_text=(current.ROOT/current.BASE/'parent-e360-root-account.ps1').read_text(encoding='utf8')
    loop_start=account_text.index('for($i=59;')
    loop_end=account_text.index('    # BEGIN P16 PAID_REPLAY TRACE',loop_start)
    loop=account_text[loop_start:loop_end]
    first_authority_loop=loop+'}\n'
    assert 'Assert-Cost-Authority $i $row $actual' in first_authority_loop
    cost_replay_loop=_p15_slice(account_text,'$seen=@{}','# GENEROUS_INTEGRATION_ACTIONS')
    paid_replay_loop=_p15_slice(account_text,'for($i=59;','function Checked-Call(')
    assert 'Historical-P25-Collector-Proof $actual.collector_trace' in paid_replay_loop
    assert 'Fresh row-local collector proof required' in paid_replay_loop
    # Passive existing PROGRAM observation records the actual production proof after its gate.
    observer="""    if($i -ge 105 -and $i -lt 109){
        $p30PaidProofs.Add(@{index=[long]$i;request=$p17PriorCall.request;trace=$collectorProof.trace;
            full_return_s=$collectorProof.full_return_s;exit_code=$collectorProof.exit_code;valid=$collectorProof.valid;internal_completion_valid=$row.internal_completion_valid})
    }
"""
    marker='    # END P16 PAID_REPLAY TRACE'
    assert paid_replay_loop.count(marker)==1
    paid_replay_observed_loop=paid_replay_loop.replace(marker,observer+marker)
    paid_start=account_text.index('    }elseif($i -ge 105 -and $i -lt 107){')
    paid_start=account_text.index('\n',paid_start)+1
    paid_end=account_text.index('    }elseif($i -eq 107){',paid_start)
    P25_paid_row_body=account_text[paid_start:paid_end]
    p27_start=account_text.index('\n',paid_end)+1
    p27_end=account_text.index('    }elseif($i -eq 108){',p27_start)
    P27_paid_row_body=account_text[p27_start:p27_end]
    p29_start=account_text.index('\n',p27_end)+1
    p29_end=account_text.index('    }elseif($i -ge 109){',p29_start)
    P29_paid_row_body=account_text[p29_start:p29_end]
    validity_start=account_text.index('    if($i -ge 91){',paid_end)
    validity_end=account_text.index(marker,validity_start)
    collector_validity_body=account_text[validity_start:validity_end]
    roster_start=account_text.index('        $call=$ledger.e360_calls[$id]',account_text.index('function Validate-Call-State'))
    roster_end=account_text.index('    }\n    foreach($id in $ledger.e360_pending_actuals.Keys)',roster_start)
    roster_call_body=account_text[roster_start:roster_end]
    expected_paid_row_proofs=[]
    for index in (105,106,107,108):
        debit=closed['debits'][index]
        cost=current.strict_json(current.read_spec(debit['descriptor']))
        actual=current.strict_json(current.read_spec(cost['actual_tool_return']))
        call=closed['e360_calls'][cost['call_id']]
        request=current.strict_json(current.read_spec(call['request']))
        expected_paid_row_proofs.append({'index':index,'request':call['request'],'trace':actual['collector_trace'],
            'full_return_s':debit['actual_outer_full_return_s'],'exit_code':debit['actual_returned_exit_code'],
            'valid':actual['collector_proof_valid'],'internal_completion_valid':cost['internal_completion_valid']})
        for label in ('logical_current','own_source','mode_original'):
            bad=copy.deepcopy(request)
            if label=='logical_current':bad['revalidation_amendment']=copy.deepcopy(current.CURRENT_REPAIR_SPEC)
            elif label=='own_source':bad['source_sha256'][current.OWNED[0]]='0'*64
            else:bad['mode']='original'
            error={'logical_current':'historical P25 original request authority' if index<107 else ('historical P27 original request authority' if index==107 else 'historical P29 original request authority'),'own_source':'historical P25 own9' if index<107 else ('historical P27 own9' if index==107 else 'historical P29 own9'),'mode_original':'External role mode'}[label]
            cases.append({'name':'p30_P25_policy_'+str(index)+'_'+label,'history':True,'p30_history':True,
                'root':str(current.ROOT),'mutation':{'local_P25':True,'policy_request':bad},'reject':True,'error_contains':error})
        for label in ('other_trace','amount','exit','request'):
            bad=copy.deepcopy(actual)
            if label=='other_trace':bad['collector_trace']=copy.deepcopy(current.P25_ACTUAL_SCANNER_RECORD_SPECS['ROOT-collector-trace.json'] if index==105 else current.P25_ACTUAL_RUFF_RECORD_SPECS['ROOT-collector-trace.json'])
            elif label=='amount':bad['actual_outer_full_return_s']=0.5
            elif label=='exit':bad['actual_returned_exit_code']=False
            else:bad['request']=copy.deepcopy(current.P25_ACTUAL_SCANNER_RECORD_SPECS['final-request.json'] if index==105 else current.P25_ACTUAL_RUFF_RECORD_SPECS['final-request.json'])
            error={'other_trace':'Exact unique admitted collector trace','amount':'paid actual/debit amount','exit':'paid actual/debit exit','request':'paid actual request descriptor'}[label]
            cases.append({'name':'p30_P25_row_'+str(index)+'_'+label,'history':True,'p30_history':True,
                'root':str(current.ROOT),'mutation':{'local_P25':True,'paid_row':{'index':index,'cost':cost,'actual':bad}},'reject':True,'error_contains':error})
        for key,wrong in (('reserved_s',False),('future_reserved_s',0),('pass_id','e360-generous-p31')):
            bad=copy.deepcopy(call);bad[key]=wrong
            cases.append({'name':'p30_P25_roster_'+str(index)+'_'+key,'history':True,'p30_history':True,
                'root':str(current.ROOT),'mutation':{'local_P25':True,'roster_call':bad,'roster_call_id':cost['call_id']},'reject':True,'error_contains':'Exact call state authority'})
    pairs=[]
    for index in (96,98,99,100,101,102,103,104,105,106,107,108):
        cost=current.strict_json(current.read_spec(closed['debits'][index]['descriptor']))
        actual=current.strict_json(current.read_spec(cost['actual_tool_return']))
        pairs.append((index,cost,actual))
    by_index={index:(cost,actual) for index,cost,actual in pairs}
    for index,role,pass_id in ((102,'scanner','e360-generous-p21'),(103,'ruff','e360-generous-p22'),(104,'ruff','e360-generous-p23'),(105,'ruff','e360-generous-p25'),(106,'scanner','e360-generous-p25'),(107,'ruff','e360-generous-p27'),(108,'ruff','e360-generous-p29')):
        assert by_index[index][0]['call_role']==role and by_index[index][0]['pass_id']==pass_id
    real_cost,real_actual,_request,_call,_debit=_p30_actual_first_current(current)
    for index,cost,actual in (*pairs,(109,real_cost,real_actual)):
        pair={'index':index,'cost':cost,'actual':actual}
        label='actual_committed_current109' if index==109 else 'actual_historical_'+str(index)
        cases.append({'name':'p30_first_cost_'+label,'history':True,'p30_history':True,'root':str(current.ROOT),'mutation':{'cost_gate':pair},'reject':False})
        for owner,key,wrong in (('cost','pass_id','e360-generous-p31'),('actual','pass_id','e360-generous-p30' if index<109 else 'e360-generous-p25'),
            ('cost','path',current.P23_FROZEN_POLICY_SPEC['path']),('cost','bytes',False),('actual','bytes',0),('actual','sha256','0'*64)):
            bad=copy.deepcopy(pair);target=bad[owner]
            if key=='pass_id':target[key]=wrong
            else:target['revalidation_amendment'][key]=wrong
            cases.append({'name':'p30_first_cost_reject_'+str(len(cases)),'history':True,'p30_history':True,'root':str(current.ROOT),'mutation':{'cost_gate':bad},'reject':True})
        for wrong in (False,59.0,-1,{96:95,98:99,99:100,100:101,101:100,102:103,103:104,104:105,105:107,106:107,107:108,108:109,109:108}[index]):
            bad=copy.deepcopy(pair);bad['index']=wrong
            cases.append({'name':'p30_first_index_reject_'+str(len(cases)),'history':True,'p30_history':True,'root':str(current.ROOT),'mutation':{'cost_gate':bad},'reject':True})
        for owner,key in (('cost','pass_id'),('actual','revalidation_amendment')):
            bad=copy.deepcopy(pair);bad[owner].pop(key)
            cases.append({'name':'p30_first_missing_reject_'+str(len(cases)),'history':True,'p30_history':True,'root':str(current.ROOT),'mutation':{'cost_gate':bad},'reject':True})
    # The actual new canonical publisher calls this same pure guard before CreateNew.
    book=(current.ROOT/current.BASE/'parent-e360-p30-bookkeeping-native.ps1').read_text(encoding='utf8')
    trace_function=(_p15_context_function_source(book,'Get-E360P30CurrentPolicySpec')+
        _p15_context_function_source(book,'Assert-E360P30CurrentPolicyTuple')+
        _p15_context_function_source(book,'Assert-E360P30CollectorTrace'))
    assert 'Assert-E360P30CollectorTrace $value $Path (Pin ' in book
    collector=current.file_spec(current.BASE+'/parent-e360-p30-ROOT-genuine-collector.txt')
    trace=current.strict_json(current.read_spec(current.P23_ACTUAL_RUFF_RECORD_SPECS['ROOT-collector-trace.json']))
    trace.update(schema='flylab.m3.e360.p30.ROOT-genuine-collector-state.v1',pass_id='e360-generous-p30',call_id='e360-generous-p30-ruff',
        cost_name='e360-generous-p30-ruff',state_key='e360:p30:ROOT-genuine:e360-generous-p30-ruff',collector_source=collector)
    trace['request']['path']=current.BASE+'/parent-e360-generous-p30-ruff-final-request.json'
    path=current.BASE+'/parent-e360-generous-p30-ruff-ROOT-collector-trace.json'
    gate={'trace':trace,'path':path,'collector':collector,'literal':copy.deepcopy(current.CURRENT_REPAIR_SPEC)}
    cases.append({'name':'p30_actual_trace_guard_complete','history':True,'p30_history':True,'root':str(current.ROOT),'mutation':{'trace_gate':gate},'reject':False})
    for key,wrong in (('schema','flylab.m3.e360.p23.ROOT-genuine-collector-state.v1'),('schema',None),('pass_id','e360-generous-p23'),
        ('state_key','e360:p20:ROOT-genuine:e360-generous-p30-ruff'),('state_key',None),('call_id','e360-generous-p30-scan'),
        ('call_role','scanner'),('cap_s',False),('next_role_allowed',True),('collector_source',{})):
        bad=copy.deepcopy(gate);bad['trace'][key]=wrong
        cases.append({'name':'p30_actual_trace_guard_reject_'+str(len(cases)),'history':True,'p30_history':True,'root':str(current.ROOT),'mutation':{'trace_gate':bad},'reject':True})
    for key,wrong in (('path',current.BASE+'/parent-e360-generous-p23-ruff-ROOT-collector-trace.json'),('literal',copy.deepcopy(current.P23_ORIGINAL_POLICY_SPEC)),('collector',{})):
        bad=copy.deepcopy(gate);bad[key]=wrong
        cases.append({'name':'p30_trace_binding_reject_'+str(len(cases)),'history':True,'p30_history':True,'root':str(current.ROOT),'mutation':{'trace_gate':bad},'reject':True})
    return {'first_authority_loop':first_authority_loop,'cost_replay_loop':cost_replay_loop,'paid_replay_loop':paid_replay_loop,
        'paid_replay_observed_loop':paid_replay_observed_loop,'roster_bindings':_P30_ROSTER_BINDINGS,
        'P25_paid_row_body':P25_paid_row_body,'P27_paid_row_body':P27_paid_row_body,'P29_paid_row_body':P29_paid_row_body,'collector_validity_body':collector_validity_body,'roster_call_body':roster_call_body,
        'expected_paid_row_proofs':expected_paid_row_proofs,'expected_stored_p25_policies':{
            'e360-generous-p25-ruff':{'outer_s':3,'internal_s':3,'future_s':885},
            'e360-generous-p25-scan':{'outer_s':30,'internal_s':30,'future_s':855},
            'e360-generous-p27-ruff':{'outer_s':3,'internal_s':3,'future_s':885},
            'e360-generous-p29-ruff':{'outer_s':3,'internal_s':3,'future_s':885}},
        'trace_function':trace_function,'program':program,'cases':cases,'authority':value,'spec':copy.deepcopy(current.CURRENT_REPAIR_SPEC),
        'dispatch_function':dispatch_function,'event':current.p30_recovery_event(),'capture_functions':capture_functions,'current_freeze':current_freeze}

def _execution_native_current_payload(current):
    """Actual current state and local mutations for the existing PROGRAM16 process."""
    context=current.execution_context()
    assert context is not None
    _debits,state,_totals,_local=current.ledger_state(current.registration())
    start=context['manifest']['start_index']
    assert len(state['debits'])>start
    identity=context['manifest']['pass_id'];first=current.charged_cost(state['debits'][start]['descriptor'],'generated')
    mutations=[]
    def mutate(path,value):mutations.append({'path':path,'value':value})
    for key,value in (('generated_shared_cumulative_used_s',0.0),('generated_shared_remaining_s',0.0),
        ('generated_stage_used_s',state['generated_stage_used_s']+1.0),
        ('e360_generated_local_used_s',state['e360_generated_local_used_s']+1.0),
        ('e360_budget_exceeded',True),('generated_new_stage_retained_bytes',False),
        ('generated_new_stage_retained_bytes',16777217)):
        mutate([key],value)
    mutate(['execution_context_manifest','sha256'],'0'*64)
    mutate(['execution_context_activation','new_generated_grant_s'],False)
    mutate(['debits',0,'actual_outer_full_return_s'],state['debits'][0]['actual_outer_full_return_s']+1.0)
    mutate(['debits',start,'actual_returned_exit_code'],False)
    mutate(['debits',start,'actual_outer_full_return_s'],state['debits'][start]['actual_outer_full_return_s']+1.0)
    mutate(['e360_calls',first['call_id'],'reserved_s'],False)
    mutate(['e360_calls',first['call_id'],'future_reserved_s'],0)
    mutate(['e360_passes',identity,'failed_cap_s'],False)
    mutate(['e360_passes',identity,'next_index'],False)
    mutate(['e360_passes',identity,'failed_cost'],{})
    definitions=current.read_spec(context['freeze']['fixed_native_context'],owner='dynamic').decode('utf8')
    return {'root':str(current.ROOT),'base':current.BASE,'definitions':definitions,'state':state,
        'first_index':start,'first_call_id':first['call_id'],'mutations':mutations}

_P30_NATIVE16_BODY=r'''
# Same original PROGRAM16 process/timeout8; current P30 full production positive remains additional.
$p30Program=[ScriptBlock]::Create($p.p30_history.program)
foreach($p30Case in $p.p30_history.cases){
    $p15ProbeCase=$p30Case
    $observed=& $p30Program -Action callbegin -Mode generated -Request $p.p15_paid_replay.cli_request
    if($p30Case.Contains('entry_guard_required') -and ($observed.entry_guard_checked -isnot [bool] -or -not $observed.entry_guard_checked)){throw 'Actual production entry guard must be reached, not setup rejection'}
    if($p30Case.Contains('expected_activation_allowed')){
        if($null -ne $observed.failure -or $observed.activation_checked -isnot [bool] -or -not $observed.activation_checked -or
           $observed.activation_allowed -isnot [bool] -or -not (Equal-Typed $observed.activation_allowed $p30Case.expected_activation_allowed)){throw ('Actual activation Boolean observation '+$p30Case.name)}
    }elseif($p30Case.reject){
        if($null -eq $observed.failure){throw ('P30 current narrow negative accepted '+$p30Case.name)}
        if($p30Case.Contains('error_contains') -and -not $observed.failure.Contains($p30Case.error_contains)){throw 'Current Native first publication failure evidence'}
    }elseif($null -ne $observed.failure){throw ('P30 actual production positive '+$p30Case.name+': '+$observed.failure)}
    if($p30Case.Contains('expected_budget')){
        foreach($key in $p30Case.expected_budget.Keys){Assert-P17-Typed $observed.budget[$key] $p30Case.expected_budget[$key] 'current Native exact budget routing'}
        foreach($key in $p30Case.preserved_historical.Keys){Assert-P17-Typed $observed.state[$key] $p30Case.preserved_historical[$key] 'current Native preserves P18 pair and grants2'}
    }
    if($p30Case.name -ceq 'p30_whole109_positive' -and (-not $observed.whole109 -or $observed.freeze.all_P30_Native_sources.Count -ne 4 -or
       $observed.activation_checked -isnot [bool] -or -not $observed.activation_checked -or $observed.activation_allowed -isnot [bool] -or -not $observed.activation_allowed)){throw 'P30 direct whole109/full freeze/actual Boolean activation END'}
    if($p30Case.name -ceq 'p30_whole109_positive'){
        if($observed.roster_END -isnot [bool] -or -not $observed.roster_END){throw 'Actual Validate-Call-State roster END required'}
        Assert-P17-Typed $observed.stored_p25_policies $p.p30_history.expected_stored_p25_policies 'Both actual P25 Call-Policy 3/3/885 and30/30/855'
        Assert-P17-Typed $observed.paid_row_proofs $p.p30_history.expected_paid_row_proofs 'Actual105/106 distinct request/trace/seconds/exit proofs'
    }
}
# Same process, actual current production reducer; no additional subprocess or timeout.
& {
    param($current)
    $taskRoot=$current.root;$taskBase=$current.base
    . ([ScriptBlock]::Create($current.definitions))
    $context=Read-Execution-Context
    if($null -eq $context){throw 'Actual admitted current context required in PROGRAM16'}
    Assert-Execution-Delta $current.state $context
    $first=$current.state.debits[$current.first_index]
    $cost=Checked-Read $first.descriptor
    if($cost.call_id -cne $current.first_call_id -or $cost.call_role -cne 'ruff' -or $cost.actual_returned_exit_code -ne 0){throw 'Actual newly paid first current cost'}
    foreach($mutation in $current.mutations){
        $bad=$current.state|ConvertTo-Json -Depth 100 -Compress|ConvertFrom-Json -AsHashtable
        $cursor=$bad
        for($i=0;$i -lt $mutation.path.Count-1;$i++){$cursor=$cursor[$mutation.path[$i]]}
        $cursor[$mutation.path[-1]]=$mutation.value
        $denied=$false
        try{Assert-Execution-Delta $bad $context}catch{$denied=$true}
        if(-not $denied){throw ('Current production mutation accepted: '+($mutation.path -join '.'))}
    }
    Assert-Execution-Delta $current.state $context
} $p.execution_current
'''
_P18_HISTORY_PROGRAM=r'''
if($p15ProbeCase.Contains('p18_history')){
    try{
        if($p15ProbeCase.Contains('deny_completed_p17')){
            Require-Prospective-P18-Pass 'e360-generous-p17' @{} $p.p18_history.authority $p.p18_history.spec @() $true
            return @{failure=$null;whole96=$false}
        }
        if($null -eq $p15ProbeCase.mutation){
            $whole=P18-Completed-Historical-P17-Proof
            if(-not (Equal-Typed $whole $ledger)){throw 'Current P18 completed P17 production proof must reach whole96 END'}
            $dependencies=@('Describe','Checked-Read','Checked-Source','P11-Fresh-Source-Leaf','Assert-P17-Typed')
            foreach($name in $dependencies){$null=Get-Command -Name $name -CommandType Function -ErrorAction Stop}
            $freeze=P18-Final-Freeze-Proof
            return @{failure=$null;whole96=$true;freeze_complete=$true;freeze_dependencies=$dependencies;freeze=$freeze}
        }
        if($p15ProbeCase.mutation.Contains('primary_proposal')){
            $primary=P18-Primary-Future $p15ProbeCase.mutation.primary_proposal
            return @{failure=$null;whole96=$false;primary_future=$primary}
        }
        if($p15ProbeCase.mutation.Contains('policy')){P18-Validate-Policy $p15ProbeCase.mutation.policy}
        elseif($p15ProbeCase.mutation.Contains('balance')){P18-Fallback-Balance $p15ProbeCase.mutation.balance $p15ProbeCase.mutation.state|Out-Null}
        else{P18-Saved-Record $p15ProbeCase.mutation.leaf $p15ProbeCase.mutation.document}
        return @{failure=$null;whole96=$false}
    }catch{return @{failure=$_.Exception.Message;whole96=$false}}
}
'''
_P18_PROGRAM_CONTEXT=r'''
if(-not $p15ProbeCase.Contains('history') -or $null -eq $p15ProbeCase.mutation){
    $p17HistoricalLedger=Checked-Read $p.p17_history.authority.historical_P16_closed_ledger94
    $p17Spec=$p.p17_history.spec;$p17Active=$ledger.Contains('e360_p17_recovery_activation')
    $p18HistoricalLedger=Checked-Read $p.p18_history.authority.closed_predecessor_ledger
    $p18Spec=$p.p18_history.spec;$p18Active=$ledger.Contains('e360_p18_recovery_activation')
    $p17PriorAdmittedRequest=$null
}
'''

def _p18_native_history_payload():
    current=_P18_CURRENT;value=current.current_repair_policy()
    program=_p15_native_programs(current=current)['fixed']
    program=program.replace(_P15_HISTORY_PROGRAM,_P18_HISTORY_PROGRAM+_P15_HISTORY_PROGRAM)
    cases=[{'name':'p18_whole96_positive','history':True,'p18_history':True,'root':str(current.ROOT),
        'ledger':value['closed_predecessor_ledger'],'p15_spec':current.file_spec(current.P15_RECOVERY_AMENDMENT),'mutation':None,'expected':None}]
    for role,primary in (('fixture',95),('suite',50)):
        proposal={'pass_id':'e360-generous-p18','mode':'generated','role':role,'future_administration_reserved_s':primary}
        cases.append({'name':'p18_primary_'+role+'_positive','history':True,'p18_history':True,'root':str(current.ROOT),
            'mutation':{'primary_proposal':proposal},'expected':None,'expected_primary':primary})
        omitted=copy.deepcopy(proposal);omitted.pop('future_administration_reserved_s')
        cases.append({'name':'p18_primary_'+role+'_missing','history':True,'p18_history':True,'root':str(current.ROOT),
            'mutation':{'primary_proposal':omitted},'expected':'Missing current P18 primary future95/50'})
        for wrong in (None,0,False,-1,primary+120,primary*2,float(primary),str(primary)):
            bad={**proposal,'future_administration_reserved_s':wrong}
            expected='P17 P18 proposal primary future95/50' if type(wrong) is int else 'Typed current P18 primary future95/50'
            cases.append({'name':'p18_primary_'+role+'_'+str(len(cases)),'history':True,'p18_history':True,'root':str(current.ROOT),
                'mutation':{'primary_proposal':bad},'expected':expected})
    for name,spec in current.P17_SAVED_RECORD_SPECS.items():
        document=current.strict_json(current.read_spec(spec))
        for mutation in ('missing','extra','typed'):
            bad=copy.deepcopy(document)
            if mutation=='missing':bad.pop(next(iter(bad)))
            elif mutation=='extra':bad['unrecorded_extra']=False
            else:bad[next(iter(bad))]=False
            cases.append({'name':name+'_'+mutation,'history':True,'p18_history':True,'root':str(current.ROOT),
                'mutation':{'leaf':name,'document':bad},'expected':'P17 actual P17 '+name+' exact typed record'})
    for key,wrong in [('current_pass_id','e360-generous-p17'),('current_pass_id','e360-generous-p19'),
        ('separate_same_pending_fallback_reserved_s',False),('whole_needed_reserved_s',898)]:
        bad=copy.deepcopy(value);bad[key]=wrong
        cases.append({'name':'p18_policy_'+str(len(cases)),'history':True,'p18_history':True,'root':str(current.ROOT),
            'mutation':{'policy':bad},'expected':'P17 compiled current P18 '+key})
    state=current.completed_historical_p17_proof();state['e360_passes']['e360-generous-p18']={'next_index':0}
    for balance in (None,False,0,-1,240,120.0):
        cases.append({'name':'p18_fallback_'+str(len(cases)),'history':True,'p18_history':True,'root':str(current.ROOT),
            'mutation':{'balance':balance,'state':state},'expected':('P17 current fallback once until valid closure' if type(balance) is int and balance==0 else 'Typed current unused fallback balance')})
    for ended in (False,True):
        changed=copy.deepcopy(state)
        changed['e360_passes']['e360-generous-p18']['failed_cost']={'path':'actual-failed'}
        changed['e360_calls']['current']={'pass_id':'e360-generous-p18','state':'charged' if ended else 'actual_pending'}
        cases.append({'name':'p18_fallback_closed_'+str(ended),'history':True,'p18_history':True,'root':str(current.ROOT),
            'mutation':{'balance':120 if ended else 0,'state':changed},'expected':'P17 current fallback once until valid closure'})
    cases.append({'name':'p18_denies_closed_p17','history':True,'p18_history':True,'root':str(current.ROOT),
        'mutation':{},'deny_completed_p17':True,'expected':'Exact activated P18 paid-predecessor/freshSource9 only'})
    return {'program':program,'cases':cases,'authority':value,'spec':copy.deepcopy(current.CURRENT_REPAIR_SPEC)}

_P18_NATIVE16_BODY=r'''
# Same original PROGRAM16 subprocess/timeout8; current P18 history END and narrow gates.
$p18Program=[ScriptBlock]::Create($p.p18_history.program)
foreach($p18Case in $p.p18_history.cases){
    $p15ProbeCase=$p18Case
    $observed=& $p18Program -Action callbegin -Mode generated -Request $p.p15_paid_replay.cli_request
    if($p18Case.name -ceq 'p18_whole96_positive'){
        if($null -ne $observed.failure -or -not $observed.whole96 -or -not $observed.freeze_complete){throw ('P18 Native whole96/freeze END: '+$observed.failure)}
        $expectedDeps=@('Describe','Checked-Read','Checked-Source','P11-Fresh-Source-Leaf','Assert-P17-Typed')
        if(-not (Equal-Typed $observed.freeze_dependencies $expectedDeps) -or $observed.freeze.own_sources_sha256.Count -ne 9 -or
           $observed.freeze.source_copies.Count -ne 4 -or $observed.freeze.all_P18_Native_sources.Count -ne 4){throw 'P18 complete freeze production dependency resolution'}
    }elseif($p18Case.Contains('expected_primary')){
        if($null -ne $observed.failure -or -not (Equal-Typed $observed.primary_future $p18Case.expected_primary)){throw 'P18 actual proposal primary95/50 production gate'}
    }elseif($observed.failure -cne $p18Case.expected){throw ('P18 same narrow production gate '+$p18Case.name+': '+$observed.failure)}
}
'''

def _p17_native_history_payload():
    current=_P17_CURRENT;value=current.p17_recovery_amendment()
    program=_p15_native_programs(current=current)['fixed']
    program=program.replace(_P15_HISTORY_PROGRAM,_P17_HISTORY_PROGRAM+_P15_HISTORY_PROGRAM)
    cases=[{'name':'p17_whole94_positive','history':True,'p17_history':True,'root':str(current.ROOT),
        'ledger':value['historical_P16_closed_ledger94'],'p15_spec':current.file_spec(current.P15_RECOVERY_AMENDMENT),'mutation':None,'expected':None}]
    cases.append({'name':'p17_completed_trace_positive','root':str(current.ROOT),
        'ledger':value['historical_P16_closed_ledger94'],'p15_spec':current.file_spec(current.P15_RECOVERY_AMENDMENT),'mutation':None,'expected':None})
    for case in _p17_saved_record_cases(current):
        cases.append({'name':case['name'],'history':True,'p17_history':True,'root':str(current.ROOT),'mutation':case,
            'expected':'P17 actual P16 '+case['leaf']+' exact typed record'})
    cases.append({'name':'p17_current_denies_completed_p16','history':True,'p17_history':True,'root':str(current.ROOT),
        'mutation':{},'deny_completed_p16':True,'expected':'Exact activated P17 paid-predecessor/freshSource9 only'})
    return {'program':program,'cases':cases,'authority':value,'spec':current.file_spec(current.P17_RECOVERY_AMENDMENT)}

_P17_PROGRAM_CONTEXT=r'''
# Current P17 only. Historical p17_history typed negatives return without reading a ledger.
if(-not $p15ProbeCase.Contains('history') -or $null -eq $p15ProbeCase.mutation){
    $p16HistoricalLedger=Checked-Read $p.p17_history.authority.historical_P15_closed_ledger93
    $p16Spec=$p.p17_history.authority.historical_P16_literal
    $p16Active=$ledger.Contains('e360_p16_recovery_activation')
    $p17HistoricalLedger=Checked-Read $p.p17_history.authority.historical_P16_closed_ledger94
    $p17Spec=$p.p17_history.spec
    $p17Active=$ledger.Contains('e360_p17_recovery_activation')
    $p16PriorAdmittedRequest=$null;$p17PriorAdmittedRequest=$null
}
'''

_P16_PROGRAM_CONTEXT=r'''
# Actual frozen P16 PROGRAM only; original frozen P15 paid-P14 context stays separate.
if(-not $p15ProbeCase.Contains('history') -or $null -eq $p15ProbeCase.mutation){
    $p16HistoricalLedger=Checked-Read $p.p16_history.authority.historical_P15_closed_ledger93
    $p16Spec=$p.p16_history.spec
    $p16Active=$ledger.Contains('e360_p16_recovery_activation')
    $p16PriorAdmittedRequest=$null
}
'''

_P17_PROGRAM_OBSERVE=_P15_PROGRAM_OBSERVE.replace('$p15PriorAdmittedRequest','$p17PriorAdmittedRequest')
_P16_PROGRAM_OBSERVE=_P15_PROGRAM_OBSERVE.replace('$p15PriorAdmittedRequest','$p16PriorAdmittedRequest')

_P17_NATIVE16_BODY=r'''
# Existing PROGRAM16, same process and timeout: real whole94 END and narrow P16 typed leaf gates.
$p17Program=[ScriptBlock]::Create($p.p17_history.program)
foreach($p17Case in $p.p17_history.cases){
    $p15ProbeCase=$p17Case
    $observed=& $p17Program -Action callbegin -Mode generated -Request $p.p15_paid_replay.cli_request
    if($p17Case.name -ceq 'p17_whole94_positive'){
        if($null -ne $observed.failure -or -not $observed.whole94){throw ('P17 Native whole94 END: '+$observed.failure)}
    }elseif($p17Case.name -ceq 'p17_completed_trace_positive'){
        if($null -ne $observed.failure -or -not $observed.metadata_mapping -or -not $observed.collector_valid -or
           $observed.RequestType -cne 'System.String' -or $observed.RequestValue -cne $p.p15_paid_replay.cli_request -or
           $observed.entryRequest -cne $p.p15_paid_replay.cli_request){throw ('P17 exact version context/observer/typed CLI current PROGRAM trace: '+$observed.failure)}
    }elseif($observed.failure -cne $p17Case.expected){throw ('P17 intended same production semantic gate '+$p17Case.name+': '+$observed.failure)}
}
'''
def _p15_paid_replay_payload(tmp_path):
    """Actual baseline positive; fresh metadata scratch tree and consistent descriptor edges per negative."""
    from pathlib import Path
    current=_P15_CURRENT;value=current.p15_recovery_amendment()
    programs=_p15_native_programs();cases=[]
    mutations=[
        ('positive',None,None),('collision',None,'Exact unique admitted collector trace'),
        ('duplicate','ledger','Duplicate debit'),('debit_bool','ledger','Typed finite debit'),
        ('schema','cost','Typed exact actual cost schema'),('charged_once','cost','Actual once-cost linkage'),
        ('cost_amount','cost','Actual once-cost linkage'),('actual_amount','actual','Actual once-cost linkage'),
        ('cost_id','cost','Actual once-cost linkage'),('cost_exit','cost','Exact actual returned cost code'),
        ('request_call','request','P15 paid request call_id'),('request_pass','request','P15 paid request pass_id'),
        ('call_key','ledger','Paid dictionary call identity'),('call_pass','ledger','P15 paid call pass'),
        ('request_role','request','P15 paid request role'),('cost_role','cost','P15 ADMIN role'),
        ('actual_request','actual','P15 paid actual request descriptor'),
        ('authority','cost','P15 paid direct revalidation_amendment'),
        ('trace_call','trace','Collector admitted role identity'),
        ('trace_request','trace','Collector immutable request/source pins'),
        ('trace_clock','trace','Exact genuine first FINAL full return/exit; no clock replacement'),
        ('trace_session','trace','Contradictory shape cannot continue collection'),
        ('trace_repeat','trace','First FINAL and original integer clocks'),
    ]
    names={'cost':value['historical_P14_ruff_full_return']['path'],'actual':value['historical_P14_ruff_actual_tool_return']['path'],
        'request':value['historical_P14_ruff_actual_request']['path'],'trace':value['historical_P14_ruff_genuine_collector_trace']['path'],
        'ledger':value['historical_P14_closed_ledger92']['path']}
    for name,leaf,expected in mutations:
        if name in ('positive','collision'):
            root=current.ROOT;ledger_spec=value['historical_P14_closed_ledger92']
        else:
            root=tmp_path/('p15-semantic-'+name);root.mkdir();_p15_source_metadata_tree(root)
            # Baseline335 includes immutable proof leaves; clone any directly consumed trusted metadata missing there.
            pending=[current.strict_json(current.read_spec(value['historical_P14_closed_ledger92']))]
            visited=set()
            while pending:
                item=pending.pop()
                if type(item) is dict:
                    if set(item)=={'path','bytes','sha256'} and Path(item['path']).suffix=='.json' and not Path(item['path']).is_absolute():
                        if item['path'] not in visited:
                            visited.add(item['path']);raw=current.read_spec(item);target=root/item['path']
                            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);pending.append(current.strict_json(raw))
                    else:pending.extend(item.values())
                elif type(item) is list:pending.extend(item)
            for path in names.values():
                target=root/path
                if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(current.safe(path).read_bytes())
            path=names[leaf];obj=current.strict_json((root/path).read_bytes())
            if name=='duplicate':obj['debits'].append(copy.deepcopy(obj['debits'][91]))
            elif name=='debit_bool':obj['debits'][91]['actual_outer_full_return_s']=True
            elif name=='schema':obj['schema']='flylab.m3.e360.unapproved-schema.v1'
            elif name=='charged_once':obj['charged_once']=False
            elif name in ('cost_amount','actual_amount'):obj['actual_outer_full_return_s']=1.841
            elif name=='cost_id':obj['cost_id']='e360-substituted'
            elif name=='cost_exit':obj['actual_returned_exit_code']=1
            elif name=='request_call':obj['call_id']='e360-substituted'
            elif name=='request_pass':obj['pass_id']='e360-generous-p13'
            elif name=='call_key':obj['e360_calls']['e360-substituted']=obj['e360_calls'].pop('e360-generous-p14-ruff')
            elif name=='call_pass':obj['e360_calls']['e360-generous-p14-ruff']['pass_id']='e360-generous-p13'
            elif name in ('request_role','cost_role'):obj['call_role']='scanner'
            elif name=='actual_request':obj['request']=value['historical_P14_scanner_unadmitted_request']
            elif name=='authority':obj['revalidation_amendment']=current.file_spec(current.P13_RECOVERY_AMENDMENT)
            elif name=='trace_call':obj['call_id']='e360-substituted'
            elif name=='trace_request':obj['request']=value['historical_P14_scanner_unadmitted_request']
            elif name=='trace_clock':obj['full_return_s']=1.841
            elif name=='trace_session':obj['chunks'][0]['result']['session_id']=123
            elif name=='trace_repeat':obj['chunks'].append(copy.deepcopy(obj['chunks'][0]))
            old=current.descriptor(path,(root/path).read_bytes());newraw=current.json_bytes(obj);(root/path).write_bytes(newraw)
            changed={path:current.descriptor(path,newraw)}
            # Propagate descriptor edges through actual -> cost -> ledger. Every test case owns its tree.
            for _ in range(12):
                progress=False
                for file in sorted(root.rglob('*.json')):
                    raw=file.read_bytes();document=current.strict_json(raw)
                    def rewrite(item, *, changed_map=changed):
                        if type(item) is dict:
                            if set(item)=={'path','bytes','sha256'} and item['path'] in changed_map:return copy.deepcopy(changed_map[item['path']])
                            return {k:rewrite(v,changed_map=changed_map) for k,v in item.items()}
                        if type(item) is list:return [rewrite(v,changed_map=changed_map) for v in item]
                        return item
                    updated=rewrite(document)
                    if updated!=document:
                        encoded=current.json_bytes(updated);file.write_bytes(encoded)
                        relative=file.relative_to(root).as_posix();changed[relative]=current.descriptor(relative,encoded);progress=True
                if not progress:break
            else:raise AssertionError('metadata scratch descriptor edges did not converge')
            ledger_spec=current.descriptor(names['ledger'],(root/names['ledger']).read_bytes())
            assert old['sha256']!=changed[path]['sha256']
        cases.append({'name':name,'root':str(root),'ledger':ledger_spec,'expected':expected,
            'program':'collision' if name=='collision' else 'fixed','p15_spec':current.file_spec(current.P15_RECOVERY_AMENDMENT)})
    old=current.p14_recovery_amendment()
    history_documents={'closure':current.strict_json(current.read_spec(value['historical_P14_independent_owned_closure'])),
        'decision':current.strict_json(current.read_spec(value['historical_P14_closed_decision'])),
        'literal':old,'approval':current.strict_json(current.read_spec(old['existing_human_authorization']))}
    history_cases=[{'name':'history_positive','history':True,'root':str(current.ROOT),
        'ledger':value['historical_P14_closed_ledger92'],'p15_spec':current.file_spec(current.P15_RECOVERY_AMENDMENT),
        'mutation':None,'expected':None}]
    for case in _p15_corrected_history_cases(current):
        if case['leaf']=='literal':expected='P15 immutable P14 literal original authority'
        elif case['leaf']=='approval':expected='P15 immutable P14 human original authority'
        elif case['leaf']=='decision':expected='P15 nonpaid decision original_IO_GPU_body_neural_UI'
        else:expected='P15 '+case['gate']
        owned=case['name'].startswith('owned_')
        original=history_documents[case['leaf']]['known_owned_ruff'] if owned else history_documents[case['leaf']]
        changed=case['document']['known_owned_ruff'] if owned else case['document']
        keys=[key for key in set(original)|set(changed) if key not in original or key not in changed or
            type(original[key]) is not type(changed[key]) or original[key]!=changed[key]]
        assert len(keys)==1,case['name']
        key=keys[0];remove=key not in changed
        history_cases.append({'name':'history_'+case['name'],'history':True,'root':str(current.ROOT),
            'ledger':value['historical_P14_closed_ledger92'],'p15_spec':current.file_spec(current.P15_RECOVERY_AMENDMENT),
            'mutation':{'leaf':case['leaf'],'owned':owned,'key':key,'remove':remove,
                'value':None if remove else changed[key]},'expected':expected})
    return {'programs':programs,'cases':cases,'history_cases':history_cases,'history_documents':history_documents,'history_authority':value,'cli_request':value['historical_P14_scanner_unadmitted_request']['path']}


_P15_NATIVE16_BODY=r'''
# BEGIN P15 LOGICAL FRAGMENT16 ACTUAL PROGRAM CASES
$fixtureRequest=$request;$fixtureRequestType=$request.GetType().FullName
$p15ProgramBlocks=@{};foreach($key in $p.p15_paid_replay.programs.Keys){$p15ProgramBlocks[$key]=[ScriptBlock]::Create($p.p15_paid_replay.programs[$key])}
foreach($p15ProbeCase in $p.p15_paid_replay.cases){
    $p15Program=$p15ProgramBlocks[$p15ProbeCase.program]
    $p15Observed=& $p15Program -Action callbegin -Mode generated -Request $p.p15_paid_replay.cli_request
    if($p15ProbeCase.name -ceq 'positive'){
        if($null -ne $p15Observed.failure -or $p15Observed.RequestType -cne 'System.String' -or
           $p15Observed.RequestValue -cne $p.p15_paid_replay.cli_request -or -not $p15Observed.metadata_mapping -or
           -not $p15Observed.collector_valid){throw ('P15 actual typed PROGRAM positive: '+$p15Observed.failure)}
    }else{
        if($p15Observed.failure -cne $p15ProbeCase.expected){throw ('P15 intended first semantic rejection '+$p15ProbeCase.name+': '+$p15Observed.failure)}
        if($p15ProbeCase.name -ceq 'collision' -and ($p15Observed.RequestValue -ceq $p.p15_paid_replay.cli_request -or
           $p15Observed.RequestType -cne 'System.String')){throw 'Original frozen collision must observe CLI string corruption'}
    }
}

# Same fragment16 and same one subprocess; actual current historical production proof.
foreach($historyCase in $p.p15_paid_replay.history_cases){
    $p15ProbeCase=$historyCase
    $program=$p15ProgramBlocks.fixed
    $observed=& $program -Action callbegin -Mode generated -Request $p.p15_paid_replay.cli_request
    if($historyCase.name -ceq 'history_positive'){
        if($null -ne $observed.failure -or -not $observed.whole92){throw ('Corrected Native historical END positive: '+$observed.failure)}
    }elseif($observed.failure -cne $historyCase.expected){
        throw ('Corrected Native intended semantic gate '+$historyCase.name+': '+$observed.failure)
    }
}

if(-not (Equal-Typed $request $fixtureRequest) -or $request.GetType().FullName -cne $fixtureRequestType){throw 'Original untyped request fixture changed by isolated PROGRAM'}
# END P15 LOGICAL FRAGMENT16 ACTUAL PROGRAM CASES
'''


def timer():
    return SimpleNamespace(check=lambda *args:None)


def granted_live(*,activated=False,historical=False,admin_historical=False,source_historical=False,ruff_historical=False):
    if activated:
        if admin_historical:
            authority=h.active_revalidation_amendment()
            live=copy.deepcopy(h.strict_json(h.read_spec(authority['inherited60']))['ledger'])
            live.update(e360_active_revalidation_activation=h.active_revalidation_event(),
                        e360_active_revalidation_amendment=h.file_spec(h.ACTIVE_REVALIDATION_AMENDMENT))
            return live
        if source_historical:
            authority=h.source_recovery_amendment()
            live=copy.deepcopy(h.strict_json(h.read_spec(authority['inherited62']))['ledger'])
            live.update(e360_source_recovery_activation=h.source_recovery_event(),
                        e360_source_recovery_amendment=h.file_spec(h.SOURCE_RECOVERY_AMENDMENT))
            return live
        if ruff_historical:
            authority=h.ruff_recovery_amendment()
            live=copy.deepcopy(h.strict_json(h.read_spec(authority['inherited63']))['ledger'])
            live.update(e360_ruff_recovery_activation=h.ruff_recovery_event(),
                        e360_ruff_recovery_amendment=h.file_spec(h.RUFF_RECOVERY_AMENDMENT))
            return live
        if not historical:
            authority=h.current_revalidation_amendment()
            live=copy.deepcopy(h.strict_json(h.read_spec(authority['inherited68']))['ledger'])
            live.update(e360_authority_scope_activation=h.current_revalidation_event(),
                        e360_authority_scope_amendment=h.file_spec(h.REVALIDATION_AMENDMENT))
            return live
        authority=h.revalidation_amendment()
        live=copy.deepcopy(h.strict_json(h.read_spec(authority['inherited59']))['ledger'])
        live.update(e360_revalidation_activation=h.revalidation_event(),
                    e360_revalidation_amendment=h.file_spec(h.OLD_REVALIDATION_AMENDMENT))
        return live
    # Explicit finite historical v2 proof fixtures; production ledger_state requires activation.
    extension=h.resource_extension()
    live=copy.deepcopy(h.strict_json(h.read_spec(extension['inherited50']))['ledger'])
    spec=h.file_spec(h.RESOURCE_EXTENSION)
    live.update(e360_generated_local_limit_s=489,generated_shared_cumulative_limit_s=600,
                generated_shared_remaining_s=float(600-live['generated_shared_cumulative_used_s']),
                e360_resource_extension_path=h.RESOURCE_EXTENSION,e360_resource_extension=spec,
                e360_calls={},e360_passes={},e360_pending_actuals={},e360_blocking_holds={},
                e360_resource_grant={'resource_extension':spec,'inherited50':extension['inherited50'],
                                     'old_shared_s':150,'old_local_s':39,'new_shared_s':600,'new_local_s':489})
    return live


def suite_recovery_live():
    """Finite generated proof from immutable81; never writes the real ledger."""
    value=h.suite_recovery_amendment()
    live=copy.deepcopy(h.strict_json(h.read_spec(value['inherited81']))['ledger'])
    live.update(e360_suite_recovery_activation=h.suite_recovery_event(),
        e360_suite_recovery_amendment=h.file_spec(h.SUITE_RECOVERY_AMENDMENT),
        e360_suite_recovery_resource_grant=h.suite_recovery_grant(),
        generated_shared_cumulative_limit_s=1400,e360_generated_local_limit_s=1289,
        generated_shared_remaining_s=float(1400-live['generated_shared_cumulative_used_s']))
    return live


def install_suite_recovery_live(tmp_path,monkeypatch,live=None):
    live=suite_recovery_live() if live is None else live
    path=tmp_path/'finite-p9-ledger.json';path.write_bytes(h.json_bytes(live))
    shared=h.registration()['shared_ledger_path'];real_safe=h.safe
    monkeypatch.setattr(h,'safe',lambda name,root=h.ROOT:path if name==shared else real_safe(name,root))
    return path,live


def generated(tmp_path,passed=True):
    c={'paths':{'output':'generated-inputs','contract':h.BASE+'/parent-e346-e360-e360-test-fixture.json'},
       'retained_reserve_bytes':1<<20,'limits':{'output_cap_bytes':16<<20},'resource_extension':h.file_spec(h.RESOURCE_EXTENSION),'revalidation_amendment':h.file_spec(h.REVALIDATION_AMENDMENT)}
    with h.CPUOnly():
        reg=h.synthetic_inputs(c,timer(),passed=passed,root=tmp_path)
        records=h.historical_metadata(reg,root=tmp_path)
    return c,reg,records


def p11_recovery_live():
    value=h.p11_recovery_amendment()
    live=copy.deepcopy(h.strict_json(h.read_spec(value['inherited83']))['ledger'])
    live.update(e360_p11_recovery_activation=h.p11_recovery_event(),
        e360_p11_recovery_amendment=h.file_spec(h.P11_RECOVERY_AMENDMENT))
    return live


def p12_recovery_live():
    value=h.p12_recovery_amendment()
    live=copy.deepcopy(h.strict_json(h.read_spec(value['inherited88']))['ledger'])
    live.update(e360_p12_recovery_activation=h.p12_recovery_event(),
        e360_p12_recovery_amendment=h.file_spec(h.P12_RECOVERY_AMENDMENT))
    return live


def p13_recovery_live():
    value=h.p13_recovery_amendment()
    live=copy.deepcopy(h.strict_json(h.read_spec(value['inherited89']))['ledger'])
    live.update(e360_p13_recovery_activation=h.p13_recovery_event(),
        e360_p13_recovery_amendment=h.file_spec(h.P13_RECOVERY_AMENDMENT))
    return live


def p14_recovery_live():
    """Finite immutable91 proof; never mutates the real ledger."""
    value=h.p14_recovery_amendment()
    live=copy.deepcopy(h.strict_json(h.read_spec(value['inherited91']))['ledger'])
    live.update(e360_p14_recovery_activation=h.p14_recovery_event(),
        e360_p14_recovery_amendment=h.file_spec(h.P14_RECOVERY_AMENDMENT))
    return live

def test_lint_recovery_P10_literal_exact82_scope11_preserves79_and_P9_caps():
    value=h.p14_recovery_amendment();old=h.p13_recovery_amendment()
    assert h.FIXED_READERS['p11_recovery_amendment']==(h.P11_RECOVERY_AMENDMENT,33809,h.P11_RECOVERY_SHA)
    assert h.FIXED_READERS['p12_recovery_amendment']==(h.P12_RECOVERY_AMENDMENT,50865,h.P12_RECOVERY_SHA)
    assert len(old['fixed_authority_scope_policy']['reader_ids'])==14
    assert h.FIXED_READERS['p13_recovery_amendment']==(h.P13_RECOVERY_AMENDMENT,h.P13_RECOVERY_BYTES,h.P13_RECOVERY_SHA)
    assert h.FIXED_READERS['p14_recovery_amendment']==(h.P14_RECOVERY_AMENDMENT,h.P14_RECOVERY_BYTES,h.P14_RECOVERY_SHA)
    assert len(h.FIXED_READERS)==15
    assert value['fixed_authority_scope_policy']['reader_ids']==old['fixed_authority_scope_policy']['reader_ids']+['p14_recovery_amendment']
    assert [value['fixed_authority_scope_policy'][key] for key in
        ('max_entries','max_text_descriptors','max_literal_json_bytes','max_aggregate_literal_json_bytes')]==[16,512,262144,2097152]
    assert h.test_functions()==sorted(value['preserved_test_functions']) and len(h.test_functions())==84
    tree=h.ast.parse(h.safe(h.OWNED[1]).read_text('utf8'))
    assert [node.name for node in tree.body if isinstance(node,(h.ast.FunctionDef,h.ast.AsyncFunctionDef))
        and node.name.startswith('test_')]==value['preserved_test_functions']
    assert value['preserved_test_functions']==old['preserved_test_functions']
    assert value['generated_pass_limit']==14 and value['modified_generated_pass_limit']==13 and value['total_pass_limit']==15
    assert value['new_generated_grant_s']==0 and value['primary_total_reserved_s']==798
    assert len(value['preserved_activation_fields'])==22 and len(value['preserved_resource_grant_fields'])==2
    for number in range(2,15):
        policy=h.effective_policy(pass_id='e360-generous-p'+str(number))
        assert policy['effective_roles']['suite']['outer_s']==(600 if number>=9 else 120)
    with pytest.raises(ValueError):h.effective_policy(pass_id='e360-generous-p15')
    assert value['historical_concrete_fix_relocations'][:42]==old['historical_concrete_fix_relocations']


def test_lint_recovery_P10_once_pair_delta_no_grant_reset_and_uncached_selector(tmp_path,monkeypatch):
    value=h.p14_recovery_amendment();before=h.strict_json(h.read_spec(value['inherited91']))['ledger']
    after=p14_recovery_live()
    assert h.p14_recovery_activation_allowed(before,lock_absent=True)
    assert h.p14_recovery_delta_valid(before,after)
    assert h.current_budget_authority(live=after)==h.current_budget_authority(live=before)
    path,_=install_suite_recovery_live(tmp_path,monkeypatch,after)
    assert h.current_scope_authority() and h.prospective_authority_spec()==h.file_spec(h.P14_RECOVERY_AMENDMENT)
    for key in h.P14_RECOVERY_FIELDS:
        bad=copy.deepcopy(after);bad.pop(key)
        with pytest.raises(ValueError):h.current_scope_authority(live=bad)
    for field in ('generated_shared_cumulative_used_s','generated_shared_remaining_s','e360_generated_local_used_s'):
        bad=copy.deepcopy(after);bad[field]+=1
        with pytest.raises(ValueError):h.p14_recovery_delta_valid(before,bad)
    bad=copy.deepcopy(after);bad['e360_lint_resource_grant']={'grant_s':0}
    with pytest.raises(ValueError):h.p14_recovery_delta_valid(before,bad)
    with pytest.raises(ValueError):h.current_scope_authority(live=bad)
    bad=copy.deepcopy(after);bad['e360_p14_recovery_activation']['new_generated_grant_s']=False
    with pytest.raises(ValueError):h.current_scope_authority(live=bad)
    bad=copy.deepcopy(after);bad['e360_calls']['e360-generous-p10-ruff']['state']='begun'
    with pytest.raises(ValueError):h.current_scope_authority(live=bad)
    for source,lock in ((after,True),(before,False)):
        with pytest.raises(ValueError):h.p14_recovery_activation_allowed(source,lock_absent=lock)
    path.write_bytes(h.json_bytes(before))
    assert not h.current_scope_authority() and h.prospective_authority_spec()==h.file_spec(h.P13_RECOVERY_AMENDMENT)
    # Actual uncached selectors reread one finite metadata ledger; no injected live argument.
    for _ in range(2):
        path.write_bytes(h.json_bytes(after))
        assert h.current_scope_authority()
        assert h.prospective_authority_spec()==h.file_spec(h.P14_RECOVERY_AMENDMENT)
        for missing in h.P14_RECOVERY_FIELDS:
            partial=copy.deepcopy(after);partial.pop(missing)
            path.write_bytes(h.json_bytes(partial))
            with pytest.raises(ValueError):h.current_scope_authority()
            with pytest.raises(ValueError):h.prospective_authority_spec()
        path.write_bytes(h.json_bytes(before))
        assert not h.current_scope_authority()
        assert h.prospective_authority_spec()==h.file_spec(h.P13_RECOVERY_AMENDMENT)
    path.write_bytes(h.json_bytes(after))
    assert h.current_scope_authority()
    # Separate current P15 equivalent after all original P14 assertions; same existing test name.
    _p15_current_equivalents(_P15_CURRENT)
    _p15_live_calls_current_relocated_history(_P15_CURRENT)
    _p16_current_equivalents(_P16_CURRENT)
    _p17_current_equivalents(_P17_CURRENT)
    _p18_current_equivalents(_P18_CURRENT,tmp_path)
    if _P30_CURRENT.execution_bootstrap(_P30_CURRENT.ROOT) is not None:_execution_actual_first_current(_P30_CURRENT)
    _p30_current_equivalents(_P30_CURRENT,tmp_path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(globals(),'h',_P17_CURRENT)
        _p17_scope_bounds(tmp_path/'current-p17-entry-bound')

def test_lint_recovery_P10_historical_P9_whole_state_four_v2_relocations(tmp_path,monkeypatch):
    value=h.p12_recovery_amendment()
    frozen=h.completed_historical_p11_proof(value=value)
    assert frozen['debits'][83:88]==value['historical_P11_actual_paid_costs']
    assert frozen['debits'][86]['cost_id'].endswith('fseal') and frozen['debits'][87]['cost_id'].endswith(':fixture')
    assert frozen['e360_passes']['e360-generous-p11']['predecessor_failed_cap_s']==3
    assert frozen['e360_passes']['e360-generous-p11']['failed_cap_s']==55
    # Actual production historical proof must not touch any prospective admission path.
    def forbidden(*args,**kwargs):raise AssertionError('current admission used in completed historical proof')
    for name in ('current_scope_authority','own_sources','closed_exit','validate_contract','operation_clock'):
        monkeypatch.setattr(h,name,forbidden)
    assert h.completed_historical_p11_proof(value=value)==frozen
    original_read=h.read_spec
    for mutation in ('cap3','cap600','reorder','duplicate','qualified','receipt','admin','missingactual'):
        def read(spec,root=h.ROOT,mutation=mutation):
            raw=original_read(spec,root)
            if spec==value['inherited88']:
                item=h.strict_json(raw)
                if mutation in ('cap3','cap600'):
                    item['ledger']['e360_passes']['e360-generous-p11']['failed_cap_s']=3 if mutation=='cap3' else 600
                elif mutation=='reorder':item['ledger']['debits'][86:88]=item['ledger']['debits'][86:88][::-1]
                elif mutation=='duplicate':item['ledger']['debits'][87]=copy.deepcopy(item['ledger']['debits'][86])
                return h.json_bytes(item)
            if spec==value['actual_predecessor_failure'] and mutation in ('qualified','admin','missingactual'):
                item=h.strict_json(raw)
                if mutation=='qualified':item['qualified']=True
                elif mutation=='admin':item['schema']='flylab.m3.e360.administrative-full-return.v1'
                else:item.pop('actual_tool_return')
                return h.json_bytes(item)
            if spec==value['historical_P11_outer'] and mutation=='receipt':
                item=h.strict_json(raw);item['pre_spawn']={'forged':True};return h.json_bytes(item)
            return raw
        with monkeypatch.context() as patch:
            patch.setattr(h,'read_spec',read)
            with pytest.raises((ValueError,KeyError)):h.completed_historical_p11_proof(value=value)


    # Completed P12 admin proof is distinct from fresh P13 admission and from the P11 worker proof above.
    current=h.p13_recovery_amendment()
    completed=h.completed_historical_p12_proof(value=current)
    assert len(completed['debits'])==89 and len(completed['e360_calls'])==39 and len(completed['e360_passes'])==12
    assert completed['debits'][:88]==frozen['debits']
    assert completed['debits'][88:89]==current['historical_P12_actual_paid_costs']
    assert completed['e360_passes']['e360-generous-p12']['predecessor_failed_cap_s']==55
    assert completed['e360_passes']['e360-generous-p12']['failed_cap_s']==3
    assert current['historical_concrete_fix_relocations'][:38]==value['historical_concrete_fix_relocations']
    for row in current['historical_concrete_fix_relocations'][38:42]:
        assert row['pass_id']=='e360-generous-p12' and row['original']['bytes']==row['frozen']['bytes']
        assert row['original']['sha256']==row['frozen']['sha256']
    for mutation in ('internalfalse','schemaworker','elapsed','actualexit','request','own','EOF','ticks','once','decision','cap55','prior3','prefix','relocation'):
        def mutated(spec,root=h.ROOT,mutation=mutation):
            raw=original_read(spec,root)
            if spec==current['actual_predecessor_failure']:
                item=h.strict_json(raw)
                if mutation=='internalfalse':item['internal_completion_valid']=False
                elif mutation=='schemaworker':item['schema']='flylab.m3.e360.actual-full-return.v1'
                elif mutation=='elapsed':item['actual_outer_full_return_s']=1.424
                elif mutation=='once':item['charged_once']=False
                else:return raw
                return h.json_bytes(item)
            if spec==current['historical_P12_actual_tool_return'] and mutation=='actualexit':
                item=h.strict_json(raw);item['actual_returned_exit_code']=0;return h.json_bytes(item)
            if spec==current['historical_P12_actual_request'] and mutation in ('request','own'):
                item=h.strict_json(raw)
                if mutation=='request':item['call_id']='e360-substituted'
                else:item['source_sha256'].pop(next(iter(item['source_sha256'])))
                return h.json_bytes(item)
            if spec==current['historical_P12_genuine_capture_return'] and mutation in ('EOF','ticks'):
                item=h.strict_json(raw)
                if mutation=='EOF':item['streams'][1]['eof_observed']=False
                else:item['process_identity']['start_utc_ticks']+=1
                return h.json_bytes(item)
            if spec==current['failed_stage_decision'] and mutation=='decision':
                item=h.strict_json(raw);item['status']='PASS';return h.json_bytes(item)
            if spec==current['inherited89'] and mutation in ('cap55','prior3','prefix'):
                item=h.strict_json(raw)
                if mutation=='cap55':item['ledger']['e360_passes']['e360-generous-p12']['failed_cap_s']=55
                elif mutation=='prior3':item['ledger']['e360_passes']['e360-generous-p12']['predecessor_failed_cap_s']=3
                else:item['ledger']['debits'][86:88]=item['ledger']['debits'][86:88][::-1]
                return h.json_bytes(item)
            return raw
        with monkeypatch.context() as patch:
            patch.setattr(h,'read_spec',mutated)
            bad=copy.deepcopy(current)
            if mutation=='relocation':bad['historical_concrete_fix_relocations'][38:42]=bad['historical_concrete_fix_relocations'][38:42][::-1]
            with pytest.raises((ValueError,KeyError)):h.completed_historical_p12_proof(value=bad)
    assert h.completed_historical_p12_proof(value=current)==completed
    # Regression: historical call identity is the dictionary key, with cost_id in the row and no inner call_id.
    historical_call_id='e360-generous-p12-ruff'
    historical_call=completed['e360_calls'][historical_call_id]
    assert 'call_id' not in historical_call and historical_call['cost_id']==historical_call_id
    assert set(historical_call)=={'attempt_id','call_role','cost_id','future_reserved_s','mode','pass_id','request',
        'reserved_s','resource_extension','revalidation_amendment','sequence_index','state'}
    for mutation in ('callkey','costid','inner_call_id','missingcostid'):
        def altered_call(spec,root=h.ROOT,mutation=mutation):
            raw=original_read(spec,root)
            if spec in (current['inherited89'],current['historical_P12_closed_ledger89']):
                item=h.strict_json(raw)
                ledger=item['ledger'] if spec==current['inherited89'] else item
                call=ledger['e360_calls'][historical_call_id]
                if mutation=='callkey':ledger['e360_calls']['e360-substituted']=ledger['e360_calls'].pop(historical_call_id)
                elif mutation=='costid':call['cost_id']='e360-substituted'
                elif mutation=='inner_call_id':call['call_id']=historical_call_id
                else:call.pop('cost_id')
                return h.json_bytes(item)
            return raw
        with monkeypatch.context() as patch:
            patch.setattr(h,'read_spec',altered_call)
            with pytest.raises((ValueError,KeyError)):h.completed_historical_p12_proof(value=current)
    assert h.completed_historical_p12_proof(value=current)==completed
    # Actual zero-byte stream IO and descriptor fence retain their separate boundary.
    streams={name:current['historical_P12_'+name] for name in ('stdout','stderr')}
    for name,spec in streams.items():
        target=tmp_path/spec['path'];target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(h.historical_p12_stream_bytes(spec))
    with monkeypatch.context() as patch:
        patch.setattr(h,'_AUTHORITY_SCOPE',None)
        patch.setattr(h,'_AUTHORITY_FENCE_STATE',None)
        with h.fixed_authority_scope(root=tmp_path,check=lambda:None) as scope:
            scope.depth=1
            assert h.historical_p12_stream_bytes(streams['stderr'],tmp_path)==b''
            assert h.historical_p12_stream_bytes(streams['stdout'],tmp_path)
            scope.depth=0
            assert len(scope.descriptors)==2
            assert any(spec['bytes']==0 for spec in scope.descriptors.values())
            scope.fence('metadata_local_completion')
            with pytest.raises(ValueError):h.read_spec(streams['stderr'],tmp_path)
            (tmp_path/streams['stderr']['path']).write_bytes(b'x')
            with pytest.raises(ValueError):scope.fence('metadata_local_completion')


    # Completed P13 has two variants of the same ADMIN schema, not two Ruff records.
    p14=h.p14_recovery_amendment();closed=h.completed_historical_p13_proof(value=p14)
    assert len(closed['debits'])==91 and len(closed['e360_calls'])==41 and len(closed['e360_passes'])==13
    assert closed['debits'][:89]==completed['debits'] and closed['debits'][89:91]==p14['historical_P13_actual_paid_costs']
    assert p14['historical_concrete_fix_relocations'][:42]==current['historical_concrete_fix_relocations']
    assert [row['pass_id'] for row in p14['historical_concrete_fix_relocations'][42:46]]==['e360-generous-p13']*4
    p13=closed['e360_passes']['e360-generous-p13']
    assert p13['next_index']==2 and p13['predecessor_failed_cap_s']==3 and p13['failed_cap_s']==20
    costs=[h.strict_json(original_read(p14['historical_P13_'+role+'_full_return'])) for role in ('ruff','scanner')]
    assert [len(cost) for cost in costs]==[28,19]
    assert [cost['schema'] for cost in costs]==['flylab.m3.e360.administrative-full-return.v1']*2
    assert [cost['actual_outer_full_return_s'] for cost in costs]==[1.838,38.792]
    assert sum(cost['actual_outer_full_return_s'] for cost in costs)==40.630
    assert costs[1]['internal_completion_valid'] is True and not any(key.startswith('capture_') for key in costs[1])
    for label in ('ruff','scan'):
        call_id='e360-generous-p13-'+label;call=closed['e360_calls'][call_id]
        assert 'call_id' not in call and len(call)==12 and call['cost_id']==call_id
    aggregate=h.strict_json(original_read(p14['historical_P13_ROOT_genuine_actual_returns']))
    assert len(aggregate['actual_calls_in_dispatch_order'])==2 and len(aggregate['actual_once_debits'])==2
    assert aggregate['actual_calls_in_dispatch_order'][0]['final_return_ms']<aggregate['actual_calls_in_dispatch_order'][1]['dispatch_ms']
    summary=h.strict_json(original_read(p14['historical_P13_ruff_genuine_capture_return']))
    assert type(summary['process_identity']['start_utc_ticks']) is int
    assert summary['process_identity']['start_utc_ticks']==639267292820475860
    assert h.historical_p12_stream_bytes(p14['historical_P13_ruff_stderr'])==b''
    report=h.strict_json(original_read(p14['historical_P13_scanner_report']))
    assert report['diagnostic_ready'] is True and report['future_ready'] is False and 'internal_completion_valid' not in report
    closure=h.strict_json(original_read(p14['historical_P13_independent_owned_closure']))
    assert 'charged_failure' not in closure and closure['scanner_genuine_completed_FINAL_return'] is True
    assert closure['scanner_owned_worker_or_original_children_started']==0
    for mutation in ('scan_capture','ruff_missing_capture','callkey','inner_call_id','costid','elapsed','report_internal',
        'ticks','once_singular','order','live_session','cost_internal','failed_cap','next_index','closure_child','relocation'):
        def altered_p13(spec,root=h.ROOT,mutation=mutation):
            raw=original_read(spec,root)
            if spec==p14['historical_P13_scanner_full_return'] and mutation in ('scan_capture','elapsed','cost_internal'):
                item=h.strict_json(raw)
                if mutation=='scan_capture':item['capture_return']=p14['historical_P13_ruff_genuine_capture_return']
                elif mutation=='elapsed':item['actual_outer_full_return_s']=20.0
                else:item['internal_completion_valid']=False
                return h.json_bytes(item)
            if spec==p14['historical_P13_ruff_full_return'] and mutation=='ruff_missing_capture':
                item=h.strict_json(raw);item.pop('capture_return');return h.json_bytes(item)
            if spec in (p14['inherited91'],p14['historical_P13_closed_ledger91']) and mutation in ('callkey','inner_call_id','costid','failed_cap','next_index'):
                item=h.strict_json(raw);ledger=item['ledger'] if spec==p14['inherited91'] else item
                call_id='e360-generous-p13-scan';call=ledger['e360_calls'][call_id]
                if mutation=='callkey':ledger['e360_calls']['e360-substituted']=ledger['e360_calls'].pop(call_id)
                elif mutation=='inner_call_id':call['call_id']=call_id
                elif mutation=='costid':call['cost_id']='e360-substituted'
                elif mutation=='failed_cap':ledger['e360_passes']['e360-generous-p13']['failed_cap_s']=3
                else:ledger['e360_passes']['e360-generous-p13']['next_index']=1
                return h.json_bytes(item)
            if spec==p14['historical_P13_scanner_report'] and mutation=='report_internal':
                item=h.strict_json(raw);item['internal_completion_valid']=True;return h.json_bytes(item)
            if spec==p14['historical_P13_ruff_genuine_capture_return'] and mutation=='ticks':
                item=h.strict_json(raw);item['process_identity']['start_utc_ticks']+=1;return h.json_bytes(item)
            if spec==p14['historical_P13_ROOT_genuine_actual_returns'] and mutation in ('once_singular','order'):
                item=h.strict_json(raw)
                if mutation=='once_singular':item['actual_once_debit']=item.pop('actual_once_debits')
                else:item['actual_calls_in_dispatch_order'].reverse()
                return h.json_bytes(item)
            if spec==p14['historical_P13_scanner_genuine_tool_return'] and mutation=='live_session':
                item=h.strict_json(raw);item['result']['session_id']=43632;return h.json_bytes(item)
            if spec==p14['historical_P13_independent_owned_closure'] and mutation=='closure_child':
                item=h.strict_json(raw);item['scanner_owned_worker_or_original_children_started']=1;return h.json_bytes(item)
            return raw
        with monkeypatch.context() as patch:
            patch.setattr(h,'read_spec',altered_p13)
            bad=copy.deepcopy(p14)
            if mutation=='relocation':bad['historical_concrete_fix_relocations'][42:46]=bad['historical_concrete_fix_relocations'][42:46][::-1]
            with pytest.raises((ValueError,KeyError)):h.completed_historical_p13_proof(value=bad)
    assert h.completed_historical_p13_proof(value=p14)==closed


def test_lint_recovery_P10_native_pair_delta_preserves_whole82_and_both_grants(tmp_path):
    import subprocess
    source=h.safe(h.BASE+'/parent-e360-root-account.ps1').read_text('utf8')
    # Extract actual production functions; this probe runs only inside the admitted paid suite.
    equal=source[source.index('function Equal-Typed('):source.index('function Fallback-Closure-Proof(')]
    paths=source[source.index('function Task-Path('):source.index('function Write-Json(')]
    readers=source[source.index('function Describe('):source.index('function Equal-Typed(')]
    delta=source[source.index('function P14-Recovery-Delta-Valid('):source.index('function P14-Capture-Proof(')]
    activation=source[source.index('function P14-Recovery-Event('):source.index('function P14-Recovery-Delta-Valid(')]
    budget=source[source.index('function Suite-Recovery-Objects('):source.index('function Suite-Recovery-Activation-Allowed(')]
    scope=source[source.index('function Assert-P14-Typed('):source.index('function P14-Recovery-Manual-Pass(')]
    prior_literals=source[source.index('function Lint-Recovery-Event('):source.index('function Historical-Lint-Scope-Authority(')]+source[source.index('function P11-Recovery-Event('):source.index('function P12-Recovery-Event(')]+source[source.index('function P12-Recovery-Literal'):source.index('function Historical-P12-Scope-Authority(')]+source[source.index('function P10-Capture-Proof('):source.index('function Require-Prospective-P10-Pass(')]
    fresh=source[source.index('function P11-Fresh-Source-Leaf('):source.index('function P11-Recovery-Manual-Pass(')]
    historical=source[source.index('function P13-Completed-Historical-P12-Proof('):source.index('function Assert-P13-Typed(')]+source[source.index('function P14-Collector-Proof('):source.index('function Assert-P14-Typed(')]
    prior_helpers=source[source.index('function Assert-P12-Typed('):source.index('function P12-Recovery-Literal')]+source[source.index('function P12-Recovery-Event('):source.index('function P12-Recovery-Activation-Allowed(')]+source[source.index('function P12-Capture-Proof('):source.index('function Require-Prospective-P12-Pass(')]+source[source.index('function Assert-P13-Typed('):source.index('function P14-Recovery-Event(')]+source[source.index('function P13-Recovery-Event('):source.index('function P13-Recovery-Activation-Allowed(')]
    manual=source[source.index('function P14-Recovery-Manual-Pass('):source.index('function Historical-P11-Scope-Authority(')]
    capture=source[source.index('function P14-Capture-Proof('):source.index('function Require-Prospective-P14-Pass(')]
    routes=source[source.index('function Require-Prospective-P14-Pass('):source.index('function P14-Collector-Proof(')]
    future_start=source.index('function Future-Admin(')
    future=source[future_start:source.index("\nif($Action -eq 'begin')",future_start)]
    live_validation=source[source.index('function Validate-Call-State'):source.index('\nValidate-Call-State')]
    assert 'Require-Prospective-P14-Pass $passID $pass $p14Recovery $p14Spec $ledger.debits $p14Active' in live_validation
    value=h.p14_recovery_amendment();before=h.strict_json(h.read_spec(value['inherited91']))['ledger']
    payload={'before':before,'after':p14_recovery_live(),'event':h.p14_recovery_event(),'spec':h.file_spec(h.P14_RECOVERY_AMENDMENT)}
    payload.update(historical_p10_authority=h.lint_recovery_amendment(),historical_p10_spec=h.file_spec(h.LINT_RECOVERY_AMENDMENT),
        historical_p11_authority=h.p11_recovery_amendment(),historical_p11_spec=h.file_spec(h.P11_RECOVERY_AMENDMENT),
        historical_p12_authority=h.p12_recovery_amendment(),historical_p12_spec=h.file_spec(h.P12_RECOVERY_AMENDMENT),
        historical_p13_authority=h.p13_recovery_amendment(),historical_p13_spec=h.file_spec(h.P13_RECOVERY_AMENDMENT))
    fix_paths=(*h.OWNED,h.BASE+'/parent-e360-root-account.ps1',h.BASE+'/parent-e360-root-metadata-helper.py',
        value['ruff_capture_policy']['function_only_source_path'],*value['prospective_native_source_paths'],value['ROOT_genuine_collector_policy']['canonical_future_source_path'])
    payload.update(authority=value,root=str(h.ROOT),fixture_root=str(tmp_path),base=h.BASE,
        prospective={'mode':'generated','modified':True,'revalidation_amendment':payload['spec'],
            'actual_failure':value['actual_predecessor_failure'],'predecessor_failed_cap_s':20,
            'source_sha256':h.own_sources(),'concrete_fix':[value['concrete_correction_snapshot'],*(h.file_spec(name) for name in fix_paths)]})
    capture_dir=tmp_path/'capture-fixture';capture_dir.mkdir()
    (capture_dir/'stdout.txt').write_bytes(b'ruff diagnostic\n');(capture_dir/'stderr.txt').write_bytes(b'')
    stream_rows=[]
    for name in ('stdout','stderr'):
        spec=h.file_spec('capture-fixture/'+name+'.txt',tmp_path)
        stream_rows.append({'stream':name,'observed_bytes':spec['bytes'],'retained_bytes':spec['bytes'],
            'eof_observed':True,'read_fault':False,'overflow':False,'readback_verified':True,'descriptor':spec})
    desc={'path':'capture-fixture/request.json','bytes':1,'sha256':'a'*64}
    summary={'call_id':'e360-generous-p14-ruff','admission_complete':True,'publication_verified':True,'capture_complete':True,
        'exit_observed':True,'owned_state':'exit_observed','ruff_exit_code':0,'external_cap_s':3,
        'root_tool_exit_required':0,'process_identity':{'process_id':123,'start_utc_ticks':123456},'streams':stream_rows,
        'start_attempted':True,'started_observed':True,'start_known_failed':False,'first_failure':None,'wrapper_error_slots':[]}
    payload.update(summary=summary,manifest={'schema':'flylab.m3.e360.p14-ruff-capture.v1',
        'call_id':summary['call_id'],'pass_id':'e360-generous-p14','request':desc,'revalidation_amendment':payload['spec'],
        **{key:summary[key] for key in ('process_identity','owned_state','exit_observed','ruff_exit_code','capture_complete',
            'streams','start_attempted','started_observed','start_known_failed','first_failure')},'wrapper_errors':{}},
        capture_request={'call_id':summary['call_id'],'pass_id':'e360-generous-p14'},capture_request_spec=desc,
        capture_authority={'ruff_capture_policy':{'manifest_path':'capture-fixture/manifest.json',
            'stdout_path':'capture-fixture/stdout.txt','stderr_path':'capture-fixture/stderr.txt'}})
    payload.update(p15_paid_replay=_p15_paid_replay_payload(tmp_path),p16_history=_p16_native_history_payload(),p17_history=_p17_native_history_payload(),p18_history=_p18_native_history_payload(),p30_history=_p30_native_history_payload(tmp_path),execution_current=_execution_native_current_payload(_P30_CURRENT))
    inputs=tmp_path/'p10-pair-input.json';inputs.write_bytes(h.json_bytes(payload))
    probe=tmp_path/'p10-pair-production.ps1'
    probe.write_text("param([string]$InputPath)\n$ErrorActionPreference='Stop'\n"+equal+paths+readers+delta+activation+budget+prior_literals+scope+fresh+prior_helpers+historical+manual+capture+routes+future+r'''
$p=Get-Content -Raw -LiteralPath $InputPath|ConvertFrom-Json -AsHashtable
if(-not (P14-Recovery-Delta-Valid $p.before $p.after $p.event $p.spec)){throw 'Exact new pair rejected'}
function Clone($v){return ($v|ConvertTo-Json -Depth 100 -Compress|ConvertFrom-Json -AsHashtable)}
foreach($key in @('generated_shared_cumulative_used_s','generated_shared_remaining_s','e360_generated_local_used_s')){
    $bad=Clone $p.after;$bad[$key]+=1
    if(P14-Recovery-Delta-Valid $p.before $bad $p.event $p.spec){throw 'Budget/cost reset accepted'}
}
$bad=Clone $p.after;$bad.extra_grant=@{grant_s=0}
if(P14-Recovery-Delta-Valid $p.before $bad $p.event $p.spec){throw 'Third new field accepted'}
foreach($key in @('e360_resource_grant','e360_suite_recovery_resource_grant')){
    $bad=Clone $p.after;$bad[$key]=@{}
    if(P14-Recovery-Delta-Valid $p.before $bad $p.event $p.spec){throw 'Old grant changed'}
}
$bad=Clone $p.after;$bad.e360_calls['e360-generous-p10-ruff'].state='begun'
if(P14-Recovery-Delta-Valid $p.before $bad $p.event $p.spec){throw 'Historical whole call changed'}
$bad=Clone $p.after;$bad.e360_p14_recovery_activation.new_generated_grant_s=$false
if(P14-Recovery-Delta-Valid $p.before $bad $p.event $p.spec){throw 'Typed grant0 replaced by bool'}
$taskRoot=$p.root;$taskBase=$p.base
Require-Prospective-P14-Pass 'e360-generous-p14' $p.prospective $p.authority $p.spec $p.before.debits $true
function Reject($code){$rejected=$false;try{& $code}catch{$rejected=$true};if(-not $rejected){throw 'Invalid production boundary accepted'}}
if(-not (P14-Recovery-Activation-Allowed $p.before $p.before $p.authority $true)){throw 'Exact inherited91 activation denied'}
$short=Clone $p.before;$short.debits=@($short.debits[0..88])
if(P14-Recovery-Activation-Allowed $short $short $p.authority $true){throw 'Even equal typed89 cannot activate actual inherited91'}
if(P14-Recovery-Activation-Allowed $p.before $p.before $p.authority $false){throw 'Present original lock allowed activation'}
if(P14-Recovery-Activation-Allowed $p.after $p.before $p.authority $true){throw 'Repeated P14 activation allowed'}
$bad=Clone $p.before;$bad.e360_pending_actuals=@{one=@{}}
if(P14-Recovery-Activation-Allowed $bad $p.before $p.authority $true){throw 'Pending worker allowed activation'}
if(-not (Current-Scope-Authority $p.after)){throw 'Actual typed P14 selector denied complete pair'}
foreach($key in @('e360_p14_recovery_activation','e360_p14_recovery_amendment')){
    $bad=Clone $p.after;$null=$bad.Remove($key)
    Reject {Current-Scope-Authority $bad}
}
if(Current-Scope-Authority $p.before){throw 'Unactivated ledger supplied prospective P14'}
$completed=P13-Completed-Historical-P12-Proof $p.historical_p13_authority
# Regression: the actual P12 row uses dictionary-key ID/cost_id and has no nonexistent inner call_id.
$historicalCallID='e360-generous-p12-ruff';$historicalCall=$completed.e360_calls[$historicalCallID]
if($historicalCall.Contains('call_id') -or -not (Equal-Typed $historicalCall.cost_id $historicalCallID)){throw 'Historical dictionary-key call identity schema'}
$originalCheckedRead=(Get-Item Function:Checked-Read).ScriptBlock
try{
    function Checked-Read($spec){
        $item=& $originalCheckedRead $spec
        if((Equal-Typed $spec $p.historical_p13_authority.inherited89) -or (Equal-Typed $spec $p.historical_p13_authority.historical_P12_closed_ledger89)){
            $ledger=if(Equal-Typed $spec $p.historical_p13_authority.inherited89){$item.ledger}else{$item}
            $call=$ledger.e360_calls[$historicalCallID]
            switch($historicalCallMutation){
                'callkey'{$ledger.e360_calls['e360-substituted']=$call;$null=$ledger.e360_calls.Remove($historicalCallID)}
                'costid'{$call.cost_id='e360-substituted'}
                'inner_call_id'{$call.call_id=$historicalCallID}
                'missingcostid'{$null=$call.Remove('cost_id')}
            }
        }
        return $item
    }
    foreach($historicalCallMutation in @('callkey','costid','inner_call_id','missingcostid')){
        Reject {P13-Completed-Historical-P12-Proof $p.historical_p13_authority}
    }
}finally{Set-Item Function:Checked-Read -Value $originalCheckedRead}
if(-not (Equal-Typed (P13-Completed-Historical-P12-Proof $p.historical_p13_authority) $completed)){throw 'Historical proof changed after call-schema regression'}
$oldPass=$completed.e360_passes['e360-generous-p12']
if(-not (Equal-Typed $oldPass.predecessor_failed_cap_s ([long]55)) -or
   -not (Equal-Typed $oldPass.failed_cap_s ([long]3)) -or
   -not (Equal-Typed $oldPass.next_index ([long]1))){throw 'Historical predecessor55/failed3/next1 changed'}
if(-not (Equal-Typed @($completed.debits[88]) $p.historical_p13_authority.historical_P12_actual_paid_costs)){
    throw 'Actual one Ruff debit at index88 changed'
}
$completedP13=P14-Completed-Historical-P13-Proof $p.authority
if($completedP13.debits.Count -ne 91 -or $completedP13.e360_calls.Count -ne 41 -or $completedP13.e360_passes.Count -ne 13){throw 'P13 whole91/41/13'}
if(-not (Equal-Typed @($completedP13.debits[89..90]) $p.authority.historical_P13_actual_paid_costs)){throw 'P13 dual once debit order'}
$ruff=Checked-Read $p.authority.historical_P13_ruff_full_return;$scanner=Checked-Read $p.authority.historical_P13_scanner_full_return
if($ruff.Count -ne 28 -or $scanner.Count -ne 19 -or $ruff.schema -cne $scanner.schema -or
   $scanner.Contains('capture_return') -or $scanner.internal_completion_valid -ne $true){throw 'P13 ADMIN variants28/19'}
if($ruff.capture_process_identity.start_utc_ticks -isnot [long] -or
   -not (Equal-Typed $ruff.capture_process_identity.start_utc_ticks ([long]639267292820475860))){throw 'Exact native Int64 ticks'}
foreach($id in @('e360-generous-p13-ruff','e360-generous-p13-scan')){
    $row=$completedP13.e360_calls[$id]
    if($row.Count -ne 12 -or $row.Contains('call_id') -or $row.cost_id -cne $id){throw 'P13 dictionary call key/cost_id/12 fields'}
}
$closedPass=$completedP13.e360_passes['e360-generous-p13']
if(-not (Equal-Typed $closedPass.next_index ([long]2)) -or -not (Equal-Typed $closedPass.failed_cap_s ([long]20)) -or
   -not (Equal-Typed $closedPass.predecessor_failed_cap_s ([long]3))){throw 'P13 next2/failed20/previous3'}
$originalCheckedRead=(Get-Item Function:Checked-Read).ScriptBlock
try{
    function Checked-Read($spec){
        $item=& $originalCheckedRead $spec
        if((Equal-Typed $spec $p.authority.inherited91) -or (Equal-Typed $spec $p.authority.historical_P13_closed_ledger91)){
            $ledger=if(Equal-Typed $spec $p.authority.inherited91){$item.ledger}else{$item}
            $id='e360-generous-p13-scan';$row=$ledger.e360_calls[$id]
            switch($dualMutation){
                'inner_id'{$row.call_id=$id}
                'costid'{$row.cost_id='e360-substituted'}
                'next1'{$ledger.e360_passes['e360-generous-p13'].next_index=[long]1}
                'failed3'{$ledger.e360_passes['e360-generous-p13'].failed_cap_s=[long]3}
            }
        }
        if((Equal-Typed $spec $p.authority.historical_P13_scanner_full_return) -and $dualMutation -ceq 'scan_capture'){$item.capture_return=$p.authority.historical_P13_ruff_genuine_capture_return}
        if((Equal-Typed $spec $p.authority.historical_P13_ruff_full_return) -and $dualMutation -ceq 'ticks'){$item.capture_process_identity.start_utc_ticks+=1}
        if((Equal-Typed $spec $p.authority.historical_P13_ROOT_genuine_actual_returns) -and $dualMutation -ceq 'singular'){$item.actual_once_debit=$item.actual_once_debits;$null=$item.Remove('actual_once_debits')}
        return $item
    }
    foreach($dualMutation in @('inner_id','costid','next1','failed3','scan_capture','ticks','singular')){Reject {P14-Completed-Historical-P13-Proof $p.authority}}
}finally{Set-Item Function:Checked-Read -Value $originalCheckedRead}
if(-not (Equal-Typed (P14-Completed-Historical-P13-Proof $p.authority) $completedP13)){throw 'P13 proof stable after mutations'}
foreach($key in @($p.authority.preserved_activation_fields)+@($p.authority.preserved_resource_grant_fields)){
    $bad=Clone $p.after;$bad[$key]=$null
    if(P14-Recovery-Delta-Valid $p.before $bad $p.event $p.spec){throw 'Old20/grants2 reset accepted'}
}
if(Equal-Typed ([long]0) $false){throw 'Boolean compared equal to integer'}
if(Equal-Typed ([long]55) ([double]55.0)){throw 'Double compared equal to integer'}
$lintActive=$true;$p11Active=$true;$p12Active=$true;$p13Active=$true;$p14Active=$true
$lintRecovery=$p.historical_p10_authority;$lintSpec=$p.historical_p10_spec
$p11Recovery=$p.historical_p11_authority;$p11Spec=$p.historical_p11_spec
$p12Recovery=$p.historical_p12_authority;$p12Spec=$p.historical_p12_spec
$p13Recovery=$p.historical_p13_authority;$p13Spec=$p.historical_p13_spec
$p14Recovery=$p.authority;$p14Spec=$p.spec
foreach($id in @('e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p13','e360-generous-p14')){
    $expectedScope=switch($id){'e360-generous-p10'{$lintSpec};'e360-generous-p11'{$p11Spec};'e360-generous-p12'{$p12Spec};'e360-generous-p13'{$p13Spec};default{$p14Spec}}
    foreach($role in @('fixture','suite')){
        $current=@{pass_id=$id;mode='generated';role=$role;revalidation_amendment=$expectedScope}
        $expectedAdmin=if($role -ceq 'fixture'){[long]95}else{[long]50}
        if(-not (Equal-Typed (Future-Admin $current) $expectedAdmin)){throw 'Exact historical/current future95/50'}
        $bad=Clone $current;$bad.mode='original';Reject {Future-Admin $bad}
        $bad=Clone $current;$bad.revalidation_amendment=@{};Reject {Future-Admin $bad}
        switch($id){'e360-generous-p10'{$lintActive=$false};'e360-generous-p11'{$p11Active=$false};'e360-generous-p12'{$p12Active=$false};'e360-generous-p13'{$p13Active=$false};default{$p14Active=$false}}
        Reject {Future-Admin $current}
        $lintActive=$true;$p11Active=$true;$p12Active=$true;$p13Active=$true;$p14Active=$true
    }
}
foreach($id in @('e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p13','e360-generous-p15','e360-original')){
    Reject {Require-Prospective-P14-Pass $id $p.prospective $p.authority $p.spec $p.before.debits $true}
}
$bad=Clone $p.prospective;$bad.mode='original'
Reject {Require-Prospective-P14-Pass 'e360-generous-p14' $bad $p.authority $p.spec $p.before.debits $true}
Reject {Require-Prospective-P14-Pass 'e360-generous-p14' $p.prospective $p.authority $p.spec $p.before.debits $false}
$bad=Clone $p.prospective;$bad.revalidation_amendment=$p.authority.previous_revalidation_amendment
Reject {Require-Prospective-P14-Pass 'e360-generous-p14' $bad $p.authority $p.spec $p.before.debits $true}
foreach($cap in @([long]3,[long]55,[long]600,[double]20.0,$true)){
    $bad=Clone $p.prospective;$bad.predecessor_failed_cap_s=$cap
    Reject {Require-Prospective-P14-Pass 'e360-generous-p14' $bad $p.authority $p.spec $p.before.debits $true}
}
$bad=Clone $p.prospective;$bad.source_sha256=$p.before.e360_passes['e360-generous-p13'].source_sha256
Reject {Require-Prospective-P14-Pass 'e360-generous-p14' $bad $p.authority $p.spec $p.before.debits $true}
foreach($leaf in @($p.prospective.concrete_fix)){
    $bad=Clone $p.prospective;$bad.concrete_fix=@($bad.concrete_fix|Where-Object{-not (Equal-Typed $_ $leaf)})
    Reject {Require-Prospective-P14-Pass 'e360-generous-p14' $bad $p.authority $p.spec $p.before.debits $true}
}
foreach($mutation in @('missing','extra','substitute','drift')){
    $bad=Clone $p.prospective;$name=@($bad.source_sha256.Keys)[0]
    switch($mutation){
        'missing'{$null=$bad.source_sha256.Remove($name)}
        'extra'{$bad.source_sha256['unrelated.py']='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'}
        'substitute'{$digest=$bad.source_sha256[$name];$null=$bad.source_sha256.Remove($name);$bad.source_sha256['unrelated.py']=$digest}
        'drift'{$bad.source_sha256[$name]='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'}
    }
    Reject {Require-Prospective-P14-Pass 'e360-generous-p14' $bad $p.authority $p.spec $p.before.debits $true}
}
$duplicate=@($p.before.debits)+@($p.before.debits[90])
Reject {Require-Prospective-P14-Pass 'e360-generous-p14' $p.prospective $p.authority $p.spec $duplicate $true}
$taskRoot=$p.fixture_root
function Proof($summary,$manifest){
    $jsonPath=Task-Path $p.capture_authority.ruff_capture_policy.manifest_path
    [IO.File]::WriteAllText($jsonPath,($manifest|ConvertTo-Json -Depth 30 -Compress),[Text.UTF8Encoding]::new($false))
    $output=Describe $p.capture_authority.ruff_capture_policy.manifest_path
    $summary.manifest_descriptor=$output
    return P14-Capture-Proof $summary $output $p.capture_request $p.capture_request_spec $p.spec $p.capture_authority
}
$s=Clone $p.summary;$m=Clone $p.manifest;$proof=Proof $s $m
if(-not $proof.valid -or $proof.unknown){throw 'Complete exit0 rejected'}
$bad=Clone $p.summary;$bad.external_cap_s=$true
Reject {Proof $bad $p.manifest}
$bad=Clone $p.summary;$bad.external_cap_s=[double]3.0
Reject {Proof $bad $p.manifest}
$s.ruff_exit_code=[long]1;$m.ruff_exit_code=[long]1;$s.root_tool_exit_required=[long]1
$first=@{slot='ruff_exit_nonzero';kind='ruff_exit';observed_s=[double]0.25}
$s.first_failure=$first;$m.first_failure=Clone $first
$proof=Proof $s $m
if(-not $proof.valid -or $proof.unknown){throw 'Complete expected Ruff exit1 capture rejected'}
$bad=Clone $s;$bad.first_failure.observed_s=[double]0.5
Reject {Proof $bad $m}
$bad=Clone $s;$bad.first_failure.observed_s=1;$badManifest=Clone $m;$badManifest.first_failure.observed_s=1
Reject {Proof $bad $badManifest}
$bad=Clone $s;$bad.wrapper_error_slots=@('stderr_close')
Reject {Proof $bad $m}
$bad=Clone $s;$bad.ruff_exit_code=[long]0;$bad.root_tool_exit_required=[long]0
$badManifest=Clone $m;$badManifest.ruff_exit_code=[long]0
Reject {Proof $bad $badManifest}
$s.capture_complete=$false;$m.capture_complete=$false
$s.wrapper_error_slots=@('stderr_close');$m.wrapper_errors=@{stderr_close='later close failure'}
$proof=Proof $s $m
if($proof.valid -or $proof.unknown -or $proof.manifest.first_failure.kind -cne 'ruff_exit'){throw 'Later wrapper error erased first Ruff failure'}
$s.exit_observed=$false;$m.exit_observed=$false;$s.ruff_exit_code=$null;$m.ruff_exit_code=$null
$s.owned_state='unknown';$m.owned_state='unknown';$s.process_identity=$null;$m.process_identity=$null
$first=@{slot='cleanup';kind='wrapper';observed_s=[double]0.5}
$s.first_failure=$first;$m.first_failure=Clone $first;$s.wrapper_error_slots=@('cleanup');$m.wrapper_errors=@{cleanup='unknown owner'}
$proof=Proof $s $m
if($proof.valid -or -not $proof.unknown){throw 'Unknown ownership invented closure'}
$bad=Clone $s;$bad.root_tool_exit_required=0
Reject {Proof $bad $m}
$worker=@{call_id='e360-required-worker';pass_id='e360-generous-p14';attempt_id='e360-required';call_role='fixture_worker'}
$workerPath='capture-fixture/worker-request.json'
[IO.File]::WriteAllText((Task-Path $workerPath),($worker|ConvertTo-Json -Compress),[Text.UTF8Encoding]::new($false))
$workerSpec=Describe $workerPath
$actual=@{call_id=$worker.call_id;pass_id=$worker.pass_id;request=$workerSpec}
$actualPath='capture-fixture/worker-actual.json'
[IO.File]::WriteAllText((Task-Path $actualPath),($actual|ConvertTo-Json -Depth 10 -Compress),[Text.UTF8Encoding]::new($false))
$actualSpec=Describe $actualPath
$seal=@{pass_id=$worker.pass_id;attempt_id=$worker.attempt_id;call_role='fixture_seal'}
$state=@{e360_calls=@{};e360_pending_actuals=@{}}
$state.e360_calls[$worker.call_id]=@{state='actual_pending';mode='generated';pass_id=$worker.pass_id;
    attempt_id=$worker.attempt_id;call_role=$worker.call_role;request=$workerSpec}
$state.e360_pending_actuals.one=@{call_id=$worker.call_id;mode='generated';actual_tool_return=$actualSpec}
Require-Pending-P14-Seal $seal $state
$bad=Clone $seal;$bad.attempt_id='e360-other'
Reject {Require-Pending-P14-Seal $bad $state}
$bad=Clone $seal;$bad.call_role='suite_seal'
Reject {Require-Pending-P14-Seal $bad $state}
$bad=Clone $state;$bad.e360_calls[$worker.call_id].attempt_id='e360-substituted'
Reject {Require-Pending-P14-Seal $seal $bad}
$bad=Clone $state;$bad.e360_calls[$worker.call_id].call_role='suite_worker'
Reject {Require-Pending-P14-Seal $seal $bad}
$bad=Clone $state;$bad.e360_pending_actuals.two=Clone $bad.e360_pending_actuals.one
Reject {Require-Pending-P14-Seal $seal $bad}
$bad=Clone $state;$bad.e360_pending_actuals.one.call_id='e360-missing'
Reject {Require-Pending-P14-Seal $seal $bad}
$actual.call_id='e360-substituted'
[IO.File]::WriteAllText((Task-Path $actualPath),($actual|ConvertTo-Json -Depth 10 -Compress),[Text.UTF8Encoding]::new($false))
$state.e360_pending_actuals.one.actual_tool_return=Describe $actualPath
Reject {Require-Pending-P14-Seal $seal $state}
# The actual production collector replay is inside this same admitted probe, never another timing job.
[IO.Directory]::CreateDirectory((Task-Path $taskBase))|Out-Null
$collectorPath=$taskBase+'/parent-e360-p14-ROOT-genuine-collector.txt'
[IO.File]::WriteAllText((Task-Path $collectorPath),'opaque collector fixture',[Text.UTF8Encoding]::new($false))
Reject {Describe $collectorPath}
$collector=P14-Collector-Source-Descriptor
[IO.File]::WriteAllBytes((Task-Path $collectorPath),[byte[]]::new(262145))
Reject {P14-Collector-Source-Descriptor}
[IO.File]::WriteAllBytes((Task-Path $collectorPath),[byte[]]@(255))
Reject {P14-Collector-Source-Descriptor}
[IO.File]::WriteAllText((Task-Path $collectorPath),'opaque collector fixture',[Text.UTF8Encoding]::new($false))
if(-not (Equal-Typed (P14-Collector-Source-Descriptor) $collector)){throw 'Canonical Source reader changed after bounded UTF8 negatives'}
$request=@{call_id='e360-generous-p14-scan';pass_id='e360-generous-p14';call_role='scanner';collector_source=$collector}
$requestPath=$taskBase+'/parent-e360-generous-p14-scan-request.json'
[IO.File]::WriteAllText((Task-Path $requestPath),($request|ConvertTo-Json -Depth 10 -Compress),[Text.UTF8Encoding]::new($false))
$requestSpec=Describe $requestPath
$tracePath=$taskBase+'/parent-e360-generous-p14-scan-ROOT-collector-trace.json'
$trace=@{schema='flylab.m3.e360.p14.ROOT-genuine-collector-state.v1';pass_id=$request.pass_id;call_id=$request.call_id;
    call_role='scanner';request=$requestSpec;collector_source=$collector;state_key='e360:p14:ROOT-genuine:e360-generous-p14-scan';
    exec_args=@{cmd=("pwsh -NoProfile -File '"+$taskBase+"/parent-e360-p14-ROOT-dispatch-adapter.ps1' -CallRole 'scanner'");workdir='C:/Users/ckthd/Dev/Projects/fly_lab'};
    cost_name=$request.call_id;cap_s=[long]20;max_output_tokens=[long]10000;
    next_role_allowed=$false;dispatch_ms=[long]1000000000000;first_final_ms=[long]1000000001000;full_return_s=[double]1.0;
    session_id=[long]123;exit_code=[long]0;phase='FINAL_RECEIVED';publication='REQUIRED';first_error=$null;
    collection_status='FINAL_RECEIVED';collection_stopped=$false;transport_error=$null;
    chunks=@(@{received_ms=[long]1000000000100;result=@{session_id=[long]123;output=''}},
             @{received_ms=[long]1000000001000;result=@{exit_code=[long]0;output='{"start_utc_ticks":639267292820475860}'}})}
function CollectorProof($v){
    [IO.File]::WriteAllText((Task-Path $tracePath),($v|ConvertTo-Json -Depth 30 -Compress),[Text.UTF8Encoding]::new($false))
    return P14-Collector-Proof (Describe $tracePath) $request $requestSpec ([double]1.0) 0
}
if(-not (CollectorProof $trace).valid){throw 'Actual stable-session genuine trace rejected'}
$whole=Clone $trace;$whole.full_return_s=[long]1
if(-not (CollectorProof $whole).valid){throw 'Whole JSON numeric genuine elapsed rejected'}
foreach($duration in @($true,'1',$null,[long]0,[double]-1,[double]::NaN,[double]::PositiveInfinity)){
    $bad=Clone $trace;$bad.full_return_s=$duration;Reject {CollectorProof $bad}
}
$bad=Clone $trace;$bad.first_final_ms+=1;Reject {CollectorProof $bad}
$bad=Clone $trace;$bad.dispatch_ms+=1;Reject {CollectorProof $bad}
$bad=Clone $trace;$bad.chunks=@($bad.chunks[0]);Reject {CollectorProof $bad}
$bad=Clone $trace;$bad.chunks+=@($bad.chunks[1]);Reject {CollectorProof $bad}
$bad=Clone $trace;$bad.session_id=[long]124;Reject {CollectorProof $bad}
$bad=Clone $trace;$bad.phase='ERROR_HOLD';$bad.first_error=@{message='truncated genuine output';at_ms=[long]1000000001000}
$bad.chunks[1].result.output_truncated=$true
if((CollectorProof $bad).valid){throw 'Known FINAL truncation qualified next role'}
$bad=Clone $trace;$bad.phase='ERROR_HOLD';$bad.first_error=@{message='first missing pending output';at_ms=[long]1000000000100}
$null=$bad.chunks[0].result.Remove('output')
$bad.chunks=@($bad.chunks[0],@{received_ms=[long]1000000000200;result=@{session_id=[long]123;output='continued same pending'}},$bad.chunks[1])
if((CollectorProof $bad).valid -or $bad.first_error.message -cne 'first missing pending output' -or $bad.first_final_ms -ne $trace.first_final_ms){throw 'Stable pending continuation cleared error or replaced first FINAL'}
$bad=Clone $trace;$bad.phase='ERROR_HOLD';$bad.first_error=@{message='first pending clock reversal';at_ms=[long]1000000000050}
$bad.chunks=@($bad.chunks[0],@{received_ms=[long]1000000000050;result=@{session_id=[long]123;output='clock failure retained'}},$bad.chunks[1])
if((CollectorProof $bad).valid){throw 'Clock failure cleared at later genuine FINAL'}
foreach($mutation in @('shape','changed_session','transport','stopped')){
    $bad=Clone $trace;$bad.phase='ERROR_HOLD';$bad.first_error=@{message='original failed transport';at_ms=[long]1000000000100}
    switch($mutation){
        'shape'{$bad.chunks[0].result.exit_code=[long]0}
        'changed_session'{$bad.chunks=@($bad.chunks[0],@{received_ms=[long]1000000000200;result=@{session_id=[long]124;output=''}},$bad.chunks[1])}
        'transport'{$bad.transport_error=@{message='lost original FINAL'}}
        'stopped'{$bad.collection_stopped=$true}
    }
    Reject {CollectorProof $bad}
}
'P10-PAIR-DELTA-PASS'
'''+_P15_NATIVE16_BODY+_P16_NATIVE16_BODY+_P17_NATIVE16_BODY+_P18_NATIVE16_BODY+_P30_NATIVE16_BODY,encoding='utf8')
    result=subprocess.run(['pwsh','-NoProfile','-File',str(probe),'-InputPath',str(inputs)],capture_output=True,text=True,timeout=8,check=False)
    assert result.returncode==0 and 'P10-PAIR-DELTA-PASS' in result.stdout,(result.stdout,result.stderr)


def test_lint_recovery_P10_manual_Ruff_once_fresh_own9_capture_leaf_and_no_P11(tmp_path,monkeypatch):
    value=h.p14_recovery_amendment();live=p14_recovery_live()
    install_suite_recovery_live(tmp_path,monkeypatch,live)
    paths=(*h.OWNED,h.BASE+'/parent-e360-root-account.ps1',h.BASE+'/parent-e360-root-metadata-helper.py',
        *value['prospective_native_source_paths'],value['ruff_capture_policy']['function_only_source_path'],value['ROOT_genuine_collector_policy']['canonical_future_source_path'])
    passed={'mode':'generated','modified':True,'actual_failure':value['actual_predecessor_failure'],
        'predecessor_failed_cap_s':20,'source_sha256':h.own_sources(),
        'concrete_fix':[value['concrete_correction_snapshot'],*(h.file_spec(name) for name in paths)]}
    h.validate_p14_pass('e360-generous-p14',passed,live['debits'])
    assert len(passed['source_sha256'])==9 and paths[-1] not in passed['source_sha256']
    for identity in ('e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p13','e360-generous-p15','e360-original','e360-auto'):
        with pytest.raises(ValueError):h.validate_p14_pass(identity,passed,live['debits'])
    for key,item in (('modified',False),('predecessor_failed_cap_s',600),('concrete_fix',passed['concrete_fix'][:-1]),
        ('source_sha256',live['e360_passes']['e360-generous-p13']['source_sha256'])):
        with pytest.raises(ValueError):h.validate_p14_pass('e360-generous-p14',{**passed,key:item},live['debits'])
    for mutation in ('missing','extra','substitute','drift'):
        bad=copy.deepcopy(passed);name=next(iter(bad['source_sha256']))
        if mutation=='missing':bad['source_sha256'].pop(name)
        elif mutation=='extra':bad['source_sha256']['unrelated.py']='a'*64
        elif mutation=='substitute':bad['source_sha256']['unrelated.py']=bad['source_sha256'].pop(name)
        else:bad['source_sha256'][name]='a'*64
        with pytest.raises(ValueError):h.validate_p14_pass('e360-generous-p14',bad,live['debits'])
    # Current P14 retains capture/collector leaves and the completed scanner predecessor cap20.
    for cap in (3,55,600,20.0,True):
        with pytest.raises(ValueError):h.validate_p14_pass('e360-generous-p14',{**passed,'predecessor_failed_cap_s':cap},live['debits'])
    for leaf in passed['concrete_fix']:
        bad={**passed,'concrete_fix':[item for item in passed['concrete_fix'] if item!=leaf]}
        with pytest.raises(ValueError):h.validate_p14_pass('e360-generous-p14',bad,live['debits'])
    duplicate=next(row for row in live['debits'] if row['descriptor']==passed['actual_failure'])
    with pytest.raises(ValueError):h.validate_p14_pass('e360-generous-p14',passed,live['debits']+[duplicate])
    path,_=install_suite_recovery_live(tmp_path,monkeypatch,live)
    bad=copy.deepcopy(live);bad.pop('e360_p14_recovery_activation');bad.pop('e360_p14_recovery_amendment')
    path.write_bytes(h.json_bytes(bad))
    with pytest.raises(ValueError):h.validate_p14_pass('e360-generous-p14',passed,live['debits'])
    # Exercise the actual Python request boundary with already-admitted finite metadata.
    owned=h.own_sources();authority_spec=h.file_spec(h.P14_RECOVERY_AMENDMENT)
    extension_spec=h.file_spec(h.RESOURCE_EXTENSION);policy=h.effective_policy(pass_id='e360-generous-p14')
    (tmp_path/h.BASE).mkdir(parents=True,exist_ok=True)
    def save(name,value):
        (tmp_path/name).write_bytes(h.json_bytes(value))
        return h.file_spec(name,tmp_path)
    worker_id='e360-required-worker'
    worker={'call_id':worker_id,'pass_id':'e360-generous-p14','call_role':'fixture_worker','attempt_id':'e360-required'}
    worker_spec=save(h.BASE+'/parent-e360-required-worker-request.json',worker)
    actual={'call_id':worker_id,'pass_id':worker['pass_id'],'request':worker_spec}
    actual_name=h.BASE+'/parent-e360-required-worker-actual.json';actual_spec=save(actual_name,actual)
    seal_id='e360-required-seal';request_name=h.BASE+'/parent-e360-required-seal-request.json'
    report={'diagnostic_ready':True};closure_spec=save(h.BASE+'/parent-e360-required-closure.json',report)
    request={'schema':'flylab.m3.e360.helper-request.v1','call_id':seal_id,'pass_id':worker['pass_id'],
        'call_role':'fixture_seal','role':'fixture','mode':'generated','action':'seal','attempt_id':worker['attempt_id'],
        'expected_output':h.BASE+'/parent-e360-required-exit.json','completion':h.BASE+'/parent-e360-required-completion.json',
        'source_sha256':owned,'closure_report':closure_spec,'resource_extension':extension_spec,
        'revalidation_amendment':authority_spec,'collector_source':h.file_spec(paths[-1]),'_request_path':request_name}
    request_spec=save(request_name,request)
    sem={'e360_calls':{
        worker_id:{'state':'actual_pending','mode':'generated','pass_id':worker['pass_id'],'call_role':'fixture_worker',
            'attempt_id':worker['attempt_id'],'request':worker_spec},
        seal_id:{'state':'begun','pass_id':worker['pass_id'],'request':request_spec,
            'resource_extension':extension_spec,'revalidation_amendment':authority_spec}},
        'e360_pending_actuals':{'one':{'call_id':worker_id,'mode':'generated','actual_tool_return':actual_spec}},
        'e360_blocking_holds':{},'e360_passes':{worker['pass_id']:{'failed_cost':{'closed_failure':True}}}}
    real_spec=h.file_spec
    with monkeypatch.context() as patch:
        patch.setattr(h,'file_spec',lambda name,root=h.ROOT:authority_spec if name==h.P14_RECOVERY_AMENDMENT else
            extension_spec if name==h.RESOURCE_EXTENSION else request['collector_source'] if name==paths[-1] else real_spec(name,root))
        patch.setattr(h,'current_scope_authority',lambda *args,**kwargs:True)
        patch.setattr(h,'effective_policy',lambda *args,**kwargs:policy)
        patch.setattr(h,'own_sources',lambda *args:owned)
        patch.setattr(h,'registration',lambda *args:{})
        patch.setattr(h,'scan_current',lambda **kwargs:report)
        patch.setattr(h,'ledger_state',lambda *args:([],sem,{},{}))
        patch.setattr(h,'require_fit',lambda *args,**kwargs:pytest.fail('Necessary failed-worker seal re-admitted science'))
        assert h.check_call_request(request,root=tmp_path)==policy['helper_limits']['fixture_seal']
        for identity in ('e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p13','e360-generous-p15'):
            with pytest.raises(ValueError):h.check_call_request({**request,'pass_id':identity},root=tmp_path)
        for key,item in (('attempt_id','e360-other'),('call_role','suite_seal')):
            with pytest.raises(ValueError):h.require_pending_p10_seal({**request,key:item},sem,root=tmp_path)
        duplicate=copy.deepcopy(sem);duplicate['e360_pending_actuals']['two']=copy.deepcopy(duplicate['e360_pending_actuals']['one'])
        with pytest.raises(ValueError):h.require_pending_p10_seal(request,duplicate,root=tmp_path)
        wrong=copy.deepcopy(sem);wrong['e360_pending_actuals']['one']['call_id']='e360-missing'
        with pytest.raises(ValueError):h.require_pending_p10_seal(request,wrong,root=tmp_path)
        bad_actual=save(actual_name,{**actual,'call_id':'e360-substituted'})
        sem['e360_pending_actuals']['one']['actual_tool_return']=bad_actual
        with pytest.raises(ValueError):h.require_pending_p10_seal(request,sem,root=tmp_path)
        sem['e360_pending_actuals']['one']['actual_tool_return']=save(actual_name,actual)
        for mutation in ('attempt','worker','request','actual_request'):
            bad=copy.deepcopy(sem)
            if mutation=='attempt':bad['e360_calls'][worker_id]['attempt_id']='e360-other'
            elif mutation=='worker':bad['e360_calls'][worker_id]['call_role']='suite_worker'
            elif mutation=='request':
                bad['e360_calls'][worker_id]['request']=save(h.BASE+'/parent-e360-wrong-worker-request.json',{**worker,'call_id':'e360-substituted'})
            else:
                bad['e360_pending_actuals']['one']['actual_tool_return']=save(h.BASE+'/parent-e360-wrong-worker-actual.json',{**actual,'request':{}})
            with patch.context() as selected:
                selected.setattr(h,'ledger_state',lambda *args,bad=bad:([],bad,{},{}))
                with pytest.raises(ValueError):h.check_call_request(request,root=tmp_path)
        assert h.check_call_request(request,root=tmp_path)==policy['helper_limits']['fixture_seal']
        later={**request,'call_role':'scanner','action':'scanner'}
        with pytest.raises(ValueError,match='failed pass permits necessary seal only'):h.check_call_request(later,root=tmp_path)


    # Immutable FINAL replay uses opaque output; it never rounds capsule ticks or samples a clock.
    collector_path=h.BASE+'/parent-e360-p14-ROOT-genuine-collector.txt'
    (tmp_path/collector_path).write_bytes(b'opaque collector fixture')
    collector=h.file_spec(collector_path,tmp_path)
    admitted={'call_id':'e360-generous-p14-scan','pass_id':'e360-generous-p14','call_role':'scanner','collector_source':collector}
    admitted_spec=save(h.BASE+'/parent-e360-generous-p14-scan-request.json',admitted)
    trace_name=h.BASE+'/parent-e360-generous-p14-scan-ROOT-collector-trace.json'
    trace={'schema':'flylab.m3.e360.p14.ROOT-genuine-collector-state.v1',**admitted,'request':admitted_spec,
        'state_key':'e360:p14:ROOT-genuine:e360-generous-p14-scan','next_role_allowed':False,
        'exec_args':{'cmd':"pwsh -NoProfile -File '"+h.BASE+"/parent-e360-p14-ROOT-dispatch-adapter.ps1' -CallRole 'scanner'",'workdir':'C:/Users/ckthd/Dev/Projects/fly_lab'},
        'cost_name':admitted['call_id'],'cap_s':20,'max_output_tokens':10000,
        'dispatch_ms':1000000000000,'first_final_ms':1000000001000,'full_return_s':1.0,'session_id':123,'exit_code':0,
        'phase':'FINAL_RECEIVED','publication':'REQUIRED','first_error':None,
        'collection_status':'FINAL_RECEIVED','collection_stopped':False,'transport_error':None,
        'chunks':[{'received_ms':1000000000100,'result':{'session_id':123,'output':''}},
                  {'received_ms':1000000001000,'result':{'exit_code':0,'output':'{"start_utc_ticks":639267292820475860}'}}]}
    def proof(value):
        return h.p14_collector_proof(save(trace_name,value),admitted,admitted_spec,1.0,0,root=tmp_path)
    assert proof(trace)['valid'] is True
    assert proof({**trace,'full_return_s':1})['valid'] is True
    for duration in (True,'1',None,0,-1,float('nan'),float('inf')):
        with pytest.raises((ValueError,TypeError)):proof({**trace,'full_return_s':duration})
    assert trace['chunks'][-1]['result']['output']=='{"start_utc_ticks":639267292820475860}'
    for mutation in ('first_clock','dispatch_clock','lost_final','second_final','session','request','next_role','exit'):
        bad=copy.deepcopy(trace)
        if mutation=='first_clock':bad['first_final_ms']+=1
        elif mutation=='dispatch_clock':bad['dispatch_ms']+=1
        elif mutation=='lost_final':bad['chunks'].pop()
        elif mutation=='second_final':bad['chunks'].append(copy.deepcopy(bad['chunks'][-1]))
        elif mutation=='session':bad['session_id']=124
        elif mutation=='request':bad['request']={}
        elif mutation=='next_role':bad['next_role_allowed']=True
        else:bad['exit_code']=1
        with pytest.raises((ValueError,KeyError)):proof(bad)
    for mutation in ('truncated','shape','changed_session'):
        bad=copy.deepcopy(trace);bad['phase']='ERROR_HOLD'
        bad['first_error']={'message':'first genuine collector failure','at_ms':1000000001000}
        if mutation=='truncated':bad['chunks'][-1]['result']['output_truncated']=True
        elif mutation=='shape':bad['chunks'][0]['result']['exit_code']=0
        else:bad['chunks'].insert(1,{'received_ms':1000000000200,'result':{'session_id':124,'output':''}})
        if mutation=='truncated':assert proof(bad)['valid'] is False
        else:
            with pytest.raises(ValueError):proof(bad)
        assert bad['first_final_ms']==trace['first_final_ms'] and bad['session_id']==123
    for failure in ('missing_output','truncated_pending','clock'):
        bad=copy.deepcopy(trace);bad['phase']='ERROR_HOLD'
        bad['first_error']={'message':'immutable first '+failure,'at_ms':1000000000100}
        if failure=='missing_output':bad['chunks'][0]['result'].pop('output')
        elif failure=='truncated_pending':bad['chunks'][0]['result']['output_truncated']=True
        receive=1000000000050 if failure=='clock' else 1000000000200
        bad['chunks'].insert(1,{'received_ms':receive,'result':{'session_id':123,'output':'continued original pending'}})
        assert proof(bad)['valid'] is False
        assert bad['first_error']['message']=='immutable first '+failure and bad['first_final_ms']==trace['first_final_ms']
    for stopped in ({'collection_stopped':True},{'transport_error':{'message':'lost original FINAL'}},
                    {'collection_status':'PENDING'}):
        with pytest.raises(ValueError):proof({**trace,**stopped})
    collector_source=h.safe(h.BASE+'/parent-e360-p14-ROOT-genuine-collector.txt').read_text('utf8')
    assert 'while (state.collection_status === "PENDING" && !state.collection_stopped)' in collector_source
    assert 'if (state.first_error === null)' in collector_source and not re.search(r'\bJSON\.parse\s*\(',collector_source)
    assert collector_source.index('state.first_final_ms = received_ms')<collector_source.index('received_ms < previous_received_ms')
    assert proof(trace)['valid'] is True


def test_real_standalone_fixture_production_path_closed_once():
    spec=json.loads(os.environ['E360_CURRENT_FIXTURE_EXIT'])
    closed=h.closed_exit(spec)
    assert closed['role']=='fixture' and closed['mode']=='generated' and closed['original_calls']==0
    contract=h.strict_json(h.read_spec(closed['contract']))
    value=h.strict_json(h.read_spec(closed['envelope']))
    manifest=h.strict_json(h.read_spec(value['diagnostic_manifest']))
    diagnostic=h.strict_json(h.read_spec(manifest['files']['diagnostic.json']))
    assert manifest['all_fsynced_and_closed'] and diagnostic['synthetic_scope']
    assert diagnostic['startup_consistent'] and diagnostic['startup']['passed']
    assert diagnostic['historical_input_status']['qualified'] is False
    assert value['original_array_reads']==0 and diagnostic['body_reconstructed'] is False
    receipt=h.strict_json(h.safe(contract['paths']['pre_spawn']).read_bytes())
    assert receipt['original_array_reads']==receipt['scientific_native_imports']==receipt['Popen_before_receipt']==0
    assert len(receipt)==33 and receipt['schema']==h.CURRENT_PRESPAWN_SCHEMA and 'authority' not in receipt
    assert receipt['source_sha256']==contract['source_sha256'] and h.safe(contract['paths']['pre_spawn']).stat().st_size<=262144


def isolate_compact_boundary(monkeypatch):
    for name in ('_SUITE_DEADLINE','_AUTHORITY_SCOPE','_AUTHORITY_FENCE_STATE'):
        monkeypatch.setattr(h,name,None)


def finite_compact_contract(tmp_path,contract=None,*,entry=10.,pid=123,created=456.):
    """Structural metadata seam; never an actual admission or a runtime grant."""
    desc={'path':'finite-authority.json','bytes':2,'sha256':h.sha(b'{}')}
    c={'mode':'generated','role':'fixture','attempt_id':'e360-finite','attempt':1,
        'paths':{'contract':'contract.json','pre_spawn':'receipt.json'},'source_sha256':{},
        'limits':{'outer_s':30,'child_s':24,'internal_s':20,'science_s':16},'original_calls':0,
        'cumulative_before_s':274.1960156,'cumulative_limit_s':600,'reserved_s':30,
        'pending_reservations':{},'stage_used_s':194.006,'stage_limit_s':600,
        'max_tree_rss_bytes':2<<30,'retained_reserve_bytes':1<<20,'cpu_threads':1,
        'registration':desc,'amendment':desc,'resource_extension':desc,'revalidation_amendment':desc,
        'closure_report':desc,'future_administration_reserved_s':70,'local_used_s':163.239,'local_limit_s':489,
        'commands':[],'ledger_debits':[],'ledger_semantics':{},'inputs':{}}
    if contract is not None:c.update(copy.deepcopy(contract))
    (tmp_path/'contract.json').write_bytes(h.json_bytes(c))
    spec=h.file_spec('contract.json',tmp_path)
    receipt=h.compact_receipt(c,spec,entry,pid,created)
    return c,spec,receipt

def test_pre_array_typed_full_authority_and_actual_lock_owner(tmp_path,monkeypatch):
    isolate_compact_boundary(monkeypatch)
    entry=h.time.perf_counter()
    # P11 contributes structural metadata only; the synthetic validation seam is explicit.
    p11=h.strict_json(h.read_spec({'path':h.BASE+'/parent-e346-e360-e360-generous-p11-f1-fixture.json',
        'bytes':184035,'sha256':'d5456721c5e08d8d332cc795bcb7842d38c4ba2957653a3b8f1c0d5478880bc9'}))
    c,spec,_=finite_compact_contract(tmp_path,{'source_sha256':p11['source_sha256'],
        'commands':p11['commands'],'ledger_debits':p11['ledger_debits'],'inputs':{},
        'mode':'original','role':'saved_prefix_diagnostic','original_calls':1,
        'ledger_semantics':{'original_jobs_running':1},'limits':{'outer_s':12,'child_s':11,'internal_s':10,'science_s':9},
        'reserved_s':12,'cumulative_limit_s':10200,'stage_limit_s':300,'local_limit_s':20,
        'paths':{'contract':'contract.json','pre_spawn':'receipt.json'}},entry=entry,pid=h.os.getpid())
    assert len(c['source_sha256'])==552
    owner={'created':456.}
    monkeypatch.setitem(sys.modules,'psutil',SimpleNamespace(Process=lambda pid:SimpleNamespace(pid=pid,create_time=lambda:owner['created'])))
    monkeypatch.setattr(h,'sys',SimpleNamespace(modules={}))
    monkeypatch.setattr(h,'validate_contract',lambda *args,**kwargs:(copy.deepcopy(c),spec,{}))
    lock=h.safe(h.BASE+'/parent-e358-original-active.lock',tmp_path);lock.parent.mkdir(parents=True)
    lock.write_bytes(h.json_bytes({'pid':h.os.getpid(),'created':456.,'contract':'contract.json'}))
    _,_,receipt_spec,_=h.pre_spawn('contract.json',entry,root=tmp_path)
    path=tmp_path/'receipt.json';raw=path.read_bytes();receipt=h.strict_json(raw)
    assert receipt_spec==h.descriptor('receipt.json',raw) and len(raw)<=262144
    assert len(receipt)==33 and receipt['schema']==h.CURRENT_PRESPAWN_SCHEMA and 'authority' not in receipt
    assert receipt['source_sha256']==c['source_sha256']
    assert h.pre_array('contract.json','receipt.json',root=tmp_path)[0]==c
    # Capture the admitted worker ENTRY before the independently bounded reread.
    real_bounded=h.bounded_json_name;read_count={'receipt':0}
    def receipt_drift(name,root=h.ROOT,*,reader=real_bounded,counter=read_count,parent=receipt,target=path,**kwargs):
        observed=reader(name,root,**kwargs)
        if name=='receipt.json':
            counter['receipt']+=1
            if counter['receipt']==1:
                changed={**parent,'entry_perf_counter':parent['entry_perf_counter']+1.}
                target.write_bytes(h.json_bytes(changed))
        return observed
    with monkeypatch.context() as patch:
        patch.setattr(h,'bounded_json_name',receipt_drift)
        with pytest.raises(ValueError,match='ENTRY'):h.pre_array('contract.json','receipt.json',root=tmp_path)
    assert read_count['receipt']==2
    path.write_bytes(raw)
    for key,item in (('attempt',True),('original_array_reads',False),('scientific_native_imports',False),
        ('Popen_before_receipt',False),('cumulative_limit_s',10200.),('reserved_s',12.),
        ('schema','flylab.m3.e360.actual-pre-spawn.v1'),('authority',c),('parent_pid',True),
        ('parent_create_time',456),('entry_perf_counter',True),('source_sha256',{})):
        bad={**receipt,key:item};path.write_bytes(h.json_bytes(bad))
        with pytest.raises(ValueError):h.pre_array('contract.json','receipt.json',root=tmp_path)
    for removed in ('limits','registration','entry_perf_counter'):
        bad=copy.deepcopy(receipt);bad.pop(removed);path.write_bytes(h.json_bytes(bad))
        with pytest.raises((ValueError,KeyError)):h.pre_array('contract.json','receipt.json',root=tmp_path)
    for malformed in (b'\xff',b'{"schema":1,"schema":2}',b'{"bad":NaN}','{}'.encode('utf-16-le'),b' '*262145):
        path.write_bytes(malformed)
        with pytest.raises((ValueError,UnicodeError)):h.pre_array('contract.json','receipt.json',root=tmp_path)
    path.write_bytes(raw)
    contract_path=tmp_path/'contract.json';contract_raw=contract_path.read_bytes()
    for field,item in (('commands',[]),('ledger_debits',[]),('inputs',{'forged':True}),('ledger_semantics',{})):
        changed={**c,field:item};contract_path.write_bytes(h.json_bytes(changed))
        for recompute in (False,True):
            candidate=copy.deepcopy(receipt)
            if recompute:candidate['contract']=h.file_spec('contract.json',tmp_path)
            with pytest.raises(ValueError):h.validate_current_receipt(c,candidate['contract'],candidate,root=tmp_path)
        contract_path.write_bytes(contract_raw)
    for changed in ({**spec,'bytes':True},{**spec,'bytes':524289},{**spec,'sha256':'0'*64},
        {**spec,'path':'../contract.json'},{**spec,'path':'contract.py'},{**spec,'extra':1}):
        with pytest.raises(ValueError):h.bounded_json_read(changed,tmp_path,cap=524288)
    owner['created']=457.
    with pytest.raises(ValueError,match='identity'):h.pre_array('contract.json','receipt.json',root=tmp_path)
    owner['created']=456.;lock.write_bytes(h.json_bytes({'pid':h.os.getpid(),'created':457.,'contract':'contract.json'}))
    with pytest.raises(ValueError,match='lock'):h.pre_array('contract.json','receipt.json',root=tmp_path)
    # P14 reaches the immutable policy gate before any array/import or parent identity work.
    p14=h.p14_recovery_amendment()
    with monkeypatch.context() as patch:
        c14={**c,'pass_id':'e360-generous-p14','mode':'generated','role':'fixture',
             'limits':p14['effective_roles']['fixture'],'revalidation_amendment':h.file_spec(h.P14_RECOVERY_AMENDMENT)}
        patch.setattr(h,'validate_contract',lambda *args,**kwargs:(copy.deepcopy(c14),spec,{}))
        bad_policy=copy.deepcopy(p14);bad_policy['checkpoint_policy']['max_entries']+=1
        patch.setattr(h,'p14_recovery_amendment',lambda root=h.ROOT:copy.deepcopy(bad_policy))
        with pytest.raises(ValueError,match='pre-array immutable checkpoint policy'):
            h._pre_array('contract.json','receipt.json',root=tmp_path,admitted_entry=entry)

def test_CURRENT_actual_AST_closure_oracles_chains_and_source_stability():
    spec=json.loads(os.environ['E360_CURRENT_CLOSURE'])
    current=h.scan_current()
    assert current==h.strict_json(h.read_spec(spec)) and current['diagnostic_ready']
    assert current['oracle_counts']=={'project':88,'missing':15,'installed':35}
    reg=h.registration()
    for row in reg['saved_missing_pin_oracle']:
        chain=current['chains'][row['path']]
        assert chain[0] in reg['discovery_roots'] and chain[-1]==row['path']
        assert all(p in current['files'] for p in chain)
    assert current['arbitrary_dynamic_native_completeness_proven'] is False


def test_AST_cycle_relative_from_namespace_alias_constants_file_loader_and_unknown(tmp_path):
    sources={'tools/root.py':"import tools.a\nfrom tools import b as attr\nfrom importlib import import_module as im\nTARGET='tools.b'\nim(TARGET)\nimport importlib.util as iu\nCORE='tools/c.py'\niu.spec_from_file_location('alias', safe(CORE))\n",
             'tools/a.py':"from . import b\nimport tools.root\n",
             'tools/b.py':"def dormant():\n import tools.c\n",
             'tools/c.py':"import importlib\nimportlib.import_module(name)\n"}
    for name,source in sources.items():
        path=tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(source,encoding='utf8')
    report=h.scan_project(['tools/root.py'],root=tmp_path)
    assert set(report['files'])==set(sources) and report['future_ready'] is False
    assert any(v['back_edges'] for v in report['edges'])
    assert any(v['kind']=='literal-file-loader' and v['targets'][0]['path']=='tools/c.py' for v in report['edges'])
    assert report['unresolved'][0]['requested'] is None
    assert report['unresolved'][0]['importer']=='tools/c.py'
    assert report['chains']['tools/c.py'][0]=='tools/root.py'


def test_omitted_edge_oracle_does_not_seed_discovery(tmp_path,monkeypatch):
    (tmp_path/'tools').mkdir()
    (tmp_path/'tools/root.py').write_text('import tools.missing\n',encoding='utf8')
    (tmp_path/'tools/missing.py').write_text('VALUE=1\n',encoding='utf8')
    reg={'discovery_roots':['tools/root.py'],'saved_project_import_oracle':
         {'tools/missing.py':h.sha((tmp_path/'tools/missing.py').read_bytes())},'saved_missing_pin_oracle':[],
         'body_native_dependencies':{},'saved_installed_import_oracle':{}}
    # Remove a real edge while the oracle still names the known file; no
    # monkeypatched scanner result can seed the process cache.
    (tmp_path/'tools/root.py').write_text('VALUE=1\n',encoding='utf8')
    with pytest.raises(ValueError,match='not discovered'):
        h.scan_current(reg,root=tmp_path)


def test_AST_source_drift_and_path_escape_rejected(tmp_path):
    (tmp_path/'tools').mkdir()
    (tmp_path/'tools/root.py').write_text("import importlib.util\nimportlib.util.spec_from_file_location('bad', '../escape.py')\n",encoding='utf8')
    with pytest.raises(ValueError):
        h.scan_project(['tools/root.py'],root=tmp_path)
    with pytest.raises(ValueError):
        h.check_sources({'tools/root.py':'0'*64},root=tmp_path)


def test_production_reader_recompute_publisher_PASS_FAIL_and_NaN_scope(tmp_path):
    import numpy as np
    for passed in (True,False):
        root=tmp_path/str(passed)
        root.mkdir()
        c,reg,records=generated(root,passed)
        value=h.diagnostic(reg,records,timer(),root=root)
        assert value['startup']['passed'] is passed and value['startup_consistent']
        assert all(value[key] is False for key in h.SCOPES)
        assert value['historical_input_status']['first_error']==records['old']['first_error']
        assert value['H359a']=='not_independently_qualified'
        raw,_,_=h.raw_archive(reg['raw_inputs']['partial_native'],timer(),root=root)
        assert np.isnan(raw['slip_mm_s']).all() and not raw['slip_valid'].any()
        state={}
        before=copy.deepcopy(value)
        h.publish_diagnostic(c,state,value,timer(),root=root)
        manifest=h.strict_json(h.read_spec(state['diagnostic_manifest'],root))
        published=h.strict_json(h.read_spec(manifest['files']['diagnostic.json'],root))
        assert published=={**before,'resource_extension':c['resource_extension'],'revalidation_amendment':c['revalidation_amendment']}
        assert manifest['resource_extension']==published['resource_extension']
        assert state['diagnostic']==published and value==before
        extras=value['observations']['startup_selected_observation'][0]['header_only_extras']
        assert extras==['legitimate_header_only_extra']


def test_partial_archive_duplicate_extra_header_dtype_and_size_reject(tmp_path):
    import io
    import zipfile

    import numpy as np
    _,reg,_=generated(tmp_path)
    original=reg['raw_inputs']['partial_control']
    path=h.safe(original['path'],tmp_path)
    saved=path.read_bytes()
    cases=[]
    cases.append({'command':np.zeros((100,42),np.float32),'tendon':np.zeros((100,6),np.float64)})
    cases.append({'command':np.zeros((99,42),np.float64),'tendon':np.zeros((100,6),np.float64)})
    cases.append({'command':np.zeros((100,42),np.float64),'tendon':np.zeros((100,6),np.float64),'extra':np.zeros(1)})
    for arrays in cases:
        payload=h.lib.npz_bytes(arrays,cap=4<<20)
        path.write_bytes(payload)
        policy={**original,'expected_size_from_directory':len(payload)}
        with pytest.raises(ValueError):
            h.raw_archive(policy,timer(),root=tmp_path)
    duplicate=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(saved)) as archive,zipfile.ZipFile(duplicate,'w') as out:
        for name in archive.namelist():
            out.writestr(name,archive.read(name))
        out.writestr('command.npy',archive.read('command.npy'))
    path.write_bytes(duplicate.getvalue())
    with pytest.raises(ValueError,match='roster'):
        h.raw_archive({**original,'expected_size_from_directory':path.stat().st_size},timer(),root=tmp_path)
    path.write_bytes(saved)
    with pytest.raises(ValueError,match='byte length'):
        h.raw_archive({**original,'expected_size_from_directory':1},timer(),root=tmp_path)


def test_metadata_first_error_source15_work_CPG_flags_tamper_reject(tmp_path):
    _,reg,records=generated(tmp_path)
    spec=reg['historical_E359']['envelope.json']
    path=h.safe(spec['path'],tmp_path)
    original=path.read_bytes()
    for mutation in ('first','source','work','CPG','run_valid'):
        old=copy.deepcopy(records['old'])
        if mutation=='first':old['first_error']['message']='repaired'
        elif mutation=='source':old['source_closure_mismatches'].pop()
        elif mutation=='work':old['counters']['controls_completed']=98
        elif mutation=='CPG':old['cpg_hashes'][99]='b'*64
        else:old['run_valid']=True
        path.write_bytes(h.json_bytes(old))
        changed=copy.deepcopy(reg)
        changed['historical_E359']['envelope.json']=h.file_spec(spec['path'],tmp_path)
        changed_exit=h.strict_json(h.read_spec(reg['historical_E359']['full_exit'],tmp_path))
        changed_exit['envelope']=changed['historical_E359']['envelope.json']
        exit_path=h.safe(reg['historical_E359']['full_exit']['path'],tmp_path)
        original_exit=exit_path.read_bytes()
        exit_path.write_bytes(h.json_bytes(changed_exit))
        changed['historical_E359']['full_exit']=h.file_spec(reg['historical_E359']['full_exit']['path'],tmp_path)
        with pytest.raises(ValueError):
            h.historical_metadata(changed,root=tmp_path)
        exit_path.write_bytes(original_exit)
        path.write_bytes(original)


def test_raw_trace_target_startup_boundary_and_provenance_tamper_reject(tmp_path):
    import numpy as np

    from tools import verify_c_banc_m3_common_descent_body as b
    _,reg,records=generated(tmp_path)
    control,native,target,inv,_=h.read_prefix(reg,records,timer(),root=tmp_path)
    for mutation in ('trace','identity','boundary','target','startup','provenance','leaf'):
        c,n,t,r=copy.deepcopy(control),copy.deepcopy(native),copy.deepcopy(target),copy.deepcopy(records)
        if mutation=='trace':r['trace']['samples'][1]['time_s']=.009
        elif mutation=='identity':r['proof']['source_network_identity']='wrong'
        elif mutation=='boundary':c['command'][0,0]=.001
        elif mutation=='target':t['target_command'][0,0]=.1
        elif mutation=='startup':r['old']['startup']['q_max']=.001
        elif mutation=='provenance':r['proof']['patch_schema']='fake'
        else:r['proof']['leaves']['host.weights']['count']=4
        with pytest.raises(ValueError):
            h.validate_saved_prefix(c,n,t,r,inv,b=b)
    assert np.isnan(native['slip_mm_s']).all()


def test_production_CPU_guard_forbidden_actual_call_names(tmp_path,monkeypatch):
    _,reg,_records=generated(tmp_path)
    from tools import verify_c_banc_m3_common_descent_body as b
    # Real Python entrypoints are stopped at their call boundary before their
    # body can resolve physics/native dependencies or read an original input.
    actual=(lambda:b.OriginalFactory(),lambda:b.resolve_original_api(),
            lambda:b.OriginalFactory.construct(None,None,None,None),lambda:b.main())
    # Finite stand-ins model unavailable graph/state/session/native interfaces;
    # their real API names/paths reach the production child guard, not an audit
    # success stub. None of these stand-in bodies may execute.
    namespace={}
    source="""class GraphStore:
 def load(self): raise AssertionError('graph executed')
class StateStore:
 def load(self): raise AssertionError('state executed')
class SSPRecoveryResourceTrial:
 def from_checkpoint(self): raise AssertionError('trial executed')
class BancWalkingSession:
 def advance(self): raise AssertionError('session executed')
def install_native_observer(): raise AssertionError('native executed')
def evaluate(): raise AssertionError('full gait executed')
"""
    exec(compile(source,'/generated/tools/finite_interfaces.py','exec'),namespace)  # noqa: S102 -- fixed generated guard probes.
    standins=tuple(lambda cls=namespace[name],method=method:getattr(cls(),method)()
                   for name,method in (('GraphStore','load'),('StateStore','load'),
                                       ('SSPRecoveryResourceTrial','from_checkpoint'),('BancWalkingSession','advance')))
    standins += (namespace['install_native_observer'],namespace['evaluate'])
    for hardware_path,name in (('/generated/torch/cuda/__init__.py','_lazy_init'),
                               ('/generated/pynvml.py','nvmlInit')):
        isolated={}
        exec(compile('def '+name+"(): raise AssertionError('hardware executed')",hardware_path,'exec'),isolated)  # noqa: S102 -- fixed generated guard probes.
        standins += (isolated[name],)
    calls=[]
    current=[None]
    def forbidden_creation(*args,**kwargs):
        calls.append(current[0])
        return current[0]()
    monkeypatch.setattr(h,'synthetic_inputs',forbidden_creation)
    c={'role':'fixture','mode':'generated'}
    for callable_ in (*actual,*standins):
        current[0]=callable_
        with pytest.raises(ValueError,match='forbids'):
            h.execute_diagnostic_path(c,{},reg,timer(),root=tmp_path)
    assert len(calls)==len(actual)+len(standins)
    # Passive third-party class definitions are permitted, not their hardware
    # functions; this catches the cold-import SymPy class evaluate distinction.
    with h.CPUOnly():
        exec(compile('class evaluate: pass','/generated/sympy/core/parameters.py','exec'),{})  # noqa: S102 -- fixed passive class probe.
    previous=sys.getprofile()
    with h.CPUOnly():
        guarded=sys.getprofile()
        with h.CPUOnly():
            assert sys.getprofile() is guarded
            exec(compile('class evaluate: pass','/generated/sympy/core/parameters.py','exec'),{})  # noqa: S102 -- fixed passive class probe.
        assert sys.getprofile() is guarded
    assert sys.getprofile() is previous
    state={}
    with pytest.raises(ValueError,match='imported source closure'):
        h.actual_imports({'source_sha256':{}},state)
    assert h.OWNED[0] in state['actual_import_sha256'] and state['source_closure_mismatches']


def test_complete42_47_ledger_omission_duplicate_bool_and_changed_descriptor(tmp_path,monkeypatch):
    reg=h.registration()
    live=granted_live(activated=True)
    path=tmp_path/'ledger.json'
    original_safe=h.safe
    monkeypatch.setattr(h,'safe',lambda name,root=h.ROOT:path if name==reg['shared_ledger_path'] else original_safe(name,root))
    path.write_bytes(h.json_bytes(live))
    state=h.ledger_state(reg)
    assert state[2]=={'original':126.361,'generated':194.006} and state[3]['generated']==163.239
    for mutation in ('omitted','duplicate','bool','descriptor','old-local','hold-count','old42','offset47-reset','old25'):
        changed=copy.deepcopy(live)
        if mutation=='omitted':changed['debits'].pop()
        elif mutation=='duplicate':changed['debits'].append(copy.deepcopy(changed['debits'][0]))
        elif mutation=='bool':changed['debits'][0]['actual_outer_full_return_s']=True
        elif mutation=='descriptor':changed['debits'][0]['descriptor']['sha256']='0'*64
        elif mutation=='old-local':changed['e359_original_local_used_s']=0
        elif mutation=='hold-count':changed['original_jobs_running']=True
        elif mutation=='old42':changed['debits']=changed['debits'][:42]
        elif mutation=='offset47-reset':changed['e360_generated_local_used_s']=0
        else:changed['e360_generated_local_limit_s']=25
        path.write_bytes(h.json_bytes(changed))
        with pytest.raises(ValueError):
            h.ledger_state(reg)
    for index in range(42,68):
        for omit in (True,False):
            changed=copy.deepcopy(live)
            if omit:changed['debits'].pop(index)
            else:changed['debits'][index]['descriptor']['sha256']='0'*64
            path.write_bytes(h.json_bytes(changed))
            with pytest.raises(ValueError):
                h.ledger_state(reg)


def test_owned_identity_reuse_unknown_timeout_and_same_ENTRY():
    class Missing(Exception):
        pass
    class Denied(Exception):
        pass
    killed=[]
    class Process:
        def __init__(self,pid):
            self.pid=pid
        def create_time(self):
            return 2.
        def kill(self):
            killed.append(self.pid)
    api=SimpleNamespace(Process=Process,NoSuchProcess=Missing,AccessDenied=Denied)
    assert h.cleanup_matching({1:1.},api)==([],[])
    assert killed==[]
    clock=h.Clock(10.,{'outer_s':1,'internal_s':.5},clock=lambda:12.)
    with pytest.raises(TimeoutError):
        clock.check()
    remaining,unknown,errors=h.bounded_owned_completion({1:2.},clock,api,sleep=lambda delay:None)
    assert remaining=={1:2.} and not unknown and errors
    class Unknown(Process):
        def create_time(self):
            raise Denied('unknown')
    _,unknown,errors=h.owned_presence({1:2.},SimpleNamespace(Process=Unknown,NoSuchProcess=Missing,AccessDenied=Denied))
    assert unknown and errors and not killed


def test_late_publication_revokes_and_keeps_first_error(tmp_path):
    c={'paths':{'output':'late','contract':h.BASE+'/parent-e346-e360-e360-late-fixture.json'},
       'retained_reserve_bytes':1<<20,'limits':{'output_cap_bytes':16<<20},'resource_extension':h.file_spec(h.RESOURCE_EXTENSION),'revalidation_amendment':h.file_spec(h.REVALIDATION_AMENDMENT)}
    state={'first_error':None,'secondary_errors':[]}
    clock=h.Clock(10.,{'internal_s':1},clock=lambda:12.)
    with pytest.raises(TimeoutError):
        h.publish_diagnostic(c,state,{'diagnostic_only':True},clock,root=tmp_path)
    h.fail(state,RuntimeError('earliest'),'reader')
    h.fail(state,TimeoutError('late'),'publication')
    assert state['revoked'] and state['first_error']['message']=='earliest'
    assert state['secondary_errors'][0]['message']=='late'


def test_whole_current_receipt_three_phases_and_strict_JSON(monkeypatch):
    functions=h.test_functions()
    sources=h.own_sources()
    receipt={'schema':'flylab.m3.e360.complete-pytest-receipt.v1','source_sha256':sources,'source_before':sources,
             'expected_functions':functions,'collected_functions':functions,'exit_code':0,'collection_errors':[],
             'original_calls':0,'original_array_reads':0,'complete_current_cases':True,
             'phases':{f:{phase:'passed' for phase in ('setup','call','teardown')} for f in functions},
             'closure_report':json.loads(os.environ['E360_CURRENT_CLOSURE'])}
    receipt.update(registration=h.file_spec(h.REGISTRATION),amendment=h.file_spec(h.AMENDMENT),resource_extension=h.file_spec(h.RESOURCE_EXTENSION),revalidation_amendment=h.prospective_authority_spec())
    assert h.validate_case_receipt(receipt)
    receipt['phases'][functions[0]].pop('teardown')
    with pytest.raises(ValueError):
        h.validate_case_receipt(receipt)
    with pytest.raises(ValueError):
        h.json_bytes({'nonfinite':float('nan')})


def test_actual_E359_metadata_admission_preserves_separate_error_snapshots():
    # Only frozen JSON/source snapshots; never open/hash/header/decode a NPZ.
    reg=h.registration()
    records=h.historical_metadata(reg)
    assert type(reg['expected_historical_secondary_errors']) is list
    assert records['old']['secondary_errors']==reg['expected_historical_secondary_errors']
    assert records['old']['first_error']==reg['expected_historical_first_error']
    assert records['exit']['qualified'] is False and records['exit']['exit_code']==1
    assert records['exit']['first_error'] is not None
    assert type(records['exit']['secondary_errors']) is list
    assert records['partial']['baseline_completed99'] is False
    assert records['old']['baseline_completed99'] is True


def test_ROOT_finalize_semantic_keys_match_exact_ledger_source_schema():
    import ast
    import re
    tree=ast.parse(h.safe(h.OWNED[0]).read_text(encoding='utf-8-sig'))
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='ledger_state')
    dictionaries={}
    for node in ast.walk(function):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Dict):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('fixed','derived'):
                    dictionaries[target.id]={key.value for key in node.value.keys}
    expected=dictionaries['fixed']|dictionaries['derived']|{
        'e360_reservations','original_jobs_running','original_reserved_s',
        'e360_original_local_used_s','e360_generated_local_used_s','e360_calls','e360_passes','e360_pending_actuals',
        'e360_blocking_holds','e360_resource_grant','e360_execution_budget_activation','e360_execution_budget_amendment',
        'e360_suite_recovery_activation','e360_suite_recovery_amendment','e360_suite_recovery_resource_grant',
        'e360_lint_recovery_activation','e360_lint_recovery_amendment'}
    source=h.safe(h.BASE+'/parent-e360-root-account.ps1').read_text(encoding='utf-8-sig')
    finalize=source[source.index("if($Action -eq 'finalize'){"):source.index('# Actual overages')]
    matches=re.findall(r'\$keys=@\(([^)]*)\)\s*foreach\(\$key in \$keys\)',finalize)
    assert len(matches)==1
    literal=matches[0]
    actual=re.findall(r"'([^']+)'",literal)
    assert len(actual)==len(set(actual)) and set(actual)==expected
    assert {'e359_reservations','e359_original_local_used_s','e359_generated_local_used_s',
            'e359_original_local_limit_s','e359_generated_local_limit_s'}<=expected


def test_AST_lexical_shadow_builtin_loader_safe_and_absent_namespace_export(tmp_path):
    sources={'tools/root.py':"""import builtins
import importlib
from importlib.machinery import SourceFileLoader as Loader
from tools import absent_attribute
from tools.exports import defined, named, collections, VALUE, FLAG, nested, conditional, LEFT, RIGHT
CORE='tools/file.py'
def lazy(root):
 local=importlib.import_module
 local('tools.lazy')
 builtins.__import__('tools.builtin')
 Loader('file',safe(CORE,root))
def ambiguous(condition):
 local=importlib.import_module
 if condition: local=unknown
 local('tools.lazy')
def shadow(importlib):
 importlib.import_module('tools.lazy')
""",'tools/exports.py':"""import collections.abc
from tools.builtin import VALUE as named
def defined(): pass
VALUE=3
FLAG: int=4
def outer():
 def nested(): pass
if True:
 conditional=5
LEFT,RIGHT=(1,2)
""",'tools/file.py':'VALUE=1\n',
        'tools/lazy.py':'import tools.root\n','tools/builtin.py':'VALUE=1\n'}
    for path,source in sources.items():
        target=tmp_path/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(source,encoding='utf8')
    report=h.scan_project(['tools/root.py'],root=tmp_path)
    assert set(report['files'])==set(sources) and not report['future_ready']
    assert any(edge['resolution']=='exported_attribute' and edge['requested']=='tools.exports.defined'
               for edge in report['edges'])
    assert {edge['requested'] for edge in report['edges']
            if edge['resolution']=='exported_attribute' and edge['requested'].startswith('tools.exports.')}=={
        'tools.exports.'+name for name in ('defined','named','collections','VALUE','FLAG')}
    assert {'tools.exports.'+name for name in ('nested','conditional','LEFT','RIGHT')} <= {
        edge['requested'] for edge in report['unresolved']}
    assert report['parsed_sources']['tools/exports.py']==h.sha((tmp_path/'tools/exports.py').read_bytes())
    for name in ('nested','conditional','LEFT','RIGHT'):
        assert report['queried_layout']['tools/exports/'+name+'.py']['kind']=='absent'
    assert any(edge['requested']=='tools.absent_attribute' and 'no proved' in edge['reason']
               for edge in report['unresolved'])
    assert any(edge['requested']=='tools/file.py' and edge['kind']=='literal-file-loader' for edge in report['edges'])
    assert any(edge['requested']=='tools.builtin' and edge['kind']=='literal-dynamic' for edge in report['edges'])
    assert {tuple(edge['scope']) for edge in report['unresolved'] if edge['kind']=='ambiguous-dynamic-alias'}=={
        ('ambiguous',),('shadow',)}
    assert any(edge['back_edges'] for edge in report['edges'])
    adapter=tmp_path/h.OWNED[0]
    adapter.write_text('from tools import absent_attribute\n',encoding='utf8')
    dependency=tmp_path/'dependencies.json'
    dependency.write_bytes(h.json_bytes({'native_library_sources':{},'installed_package_sources':{}}))
    reg={'discovery_roots':[h.OWNED[0]],'saved_project_import_oracle':{},'saved_missing_pin_oracle':[],
         'saved_installed_import_oracle':{},'body_native_dependencies':h.file_spec('dependencies.json',tmp_path)}
    assert h.scan_current(reg,root=tmp_path)['diagnostic_ready'] is False


def test_supervise_pre_spawn_failure_records_no_Popen_and_unknown_creation(tmp_path,monkeypatch):
    import time
    frozen=h.registration()
    amended=h.amendment()
    extension=h.resource_extension()
    monkeypatch.setattr(h,'resource_extension',lambda root=h.ROOT:extension)
    monkeypatch.setattr(h,'registration',lambda root=h.ROOT:frozen)
    monkeypatch.setattr(h,'amendment',lambda root=h.ROOT:amended)
    api=SimpleNamespace(Process=lambda pid:SimpleNamespace(pid=pid,create_time=lambda:456.))
    monkeypatch.setitem(sys.modules,'psutil',api)
    calls=[]
    def never(*args,**kwargs):
        calls.append('Popen')
        raise RuntimeError('creation attempted but identity unreturned')
    monkeypatch.setattr(h.subprocess,'Popen',never)
    monkeypatch.setattr(h,'pre_popen_admission',lambda *args,**kwargs:None)  # Finite ownership-only process API probe.
    def rejected(*args,**kwargs):
        raise ValueError('exact failed pre-spawn authority')
    monkeypatch.setattr(h,'pre_spawn',rejected)
    for suffix in ('unspawned','unknown','late'):
        root=tmp_path/suffix
        for path in (h.REGISTRATION,h.AMENDMENT,h.RESOURCE_EXTENSION,h.REVALIDATION_AMENDMENT):
            target=h.safe(path,root)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(h.safe(path).read_bytes())
        paths=h.names('e360-'+suffix,'saved_prefix_diagnostic')
        c={'role':'saved_prefix_diagnostic','attempt_id':'e360-'+suffix,'attempt':1,'mode':'original',
           'paths':paths,'limits':frozen['roles']['saved_prefix_diagnostic'],'source_sha256':{},
           'cumulative_before_s':10020.115960899504,'cumulative_limit_s':10200,'stage_used_s':126.361,
           'stage_limit_s':300,'local_used_s':0,'local_limit_s':20,'reserved_s':12,'original_calls':1,
           'retained_reserve_bytes':1<<19,'ledger_semantics':{'original_jobs_running':1}}
        c.update(registration=h.file_spec(h.REGISTRATION,root),amendment=h.file_spec(h.AMENDMENT,root),resource_extension=h.file_spec(h.RESOURCE_EXTENSION,root),revalidation_amendment=h.file_spec(h.REVALIDATION_AMENDMENT,root),
                 future_administration_reserved_s=8)
        h.durable(h.safe(paths['contract'],root),h.json_bytes(c),cap=256<<10)
        if suffix=='unknown':
            monkeypatch.setattr(h,'pre_spawn',lambda *args,c=c,paths=paths,root=root,**kwargs:(c,h.file_spec(paths['contract'],root),{'synthetic_receipt':True},frozen))
        else:
            monkeypatch.setattr(h,'pre_spawn',rejected)
        entry=time.perf_counter()-(30 if suffix=='late' else 0)
        assert h.supervise(paths['contract'],entry=entry,root=root)==1
        outer=h.strict_json(h.safe(paths['outer'],root).read_bytes())
        actual={'actual_outer_full_return_s':30. if suffix=='late' else .01,'actual_returned_exit_code':1}
        h.durable(h.safe(paths['actual_return'],root),h.json_bytes(actual),cap=1024)
        sealed=h.seal_full_return(paths['contract'],h.file_spec(paths['actual_return'],root),root=root)
        assert not sealed['qualified'] and sealed['charged_once'] and sealed['first_error']==outer['first_error']
        if suffix=='unknown':
            assert outer['Popen_attempted'] and not outer['Popen_returned']
            assert not outer['all_owned_ended'] and not outer['reservation_released']
            assert 'Popen-unreturned' in outer['unknown_owned_identities']
            assert h.safe(h.BASE+'/parent-e358-original-active.lock',root).exists()
        else:
            assert outer['Popen_never_attempted'] and not outer['Popen_attempted']
            assert outer['all_owned_ended'] and outer['reservation_released']
            assert not h.safe(h.BASE+'/parent-e358-original-active.lock',root).exists()
            if suffix=='late':
                assert outer['first_error']['type']=='TimeoutError'
                assert 'outer deadline' in outer['first_error']['message']
                assert not sealed['natural_closed'] and sealed['secondary_errors']
            else:
                assert outer['first_error']['type']=='ValueError'
                assert outer['first_error']['message']=='exact failed pre-spawn authority'
    assert calls==['Popen']


def test_closed_exit_rejects_unreleased_unknown_missing_and_unadmitted_mutations(monkeypatch):
    spec=json.loads(os.environ['E360_CURRENT_FIXTURE_EXIT'])
    original=h.closed_exit(spec)
    reader=h.read_spec
    current=[None]
    monkeypatch.setattr(h,'read_spec',lambda s,root=h.ROOT,**kwargs:
                        h.json_bytes(current[0]) if s==spec else reader(s,root,**kwargs))
    monkeypatch.setattr(h,'charged_cost',lambda *args,**kwargs:current[0])
    for key,value in (('reservation_released',False),('unresolved_matching_identities',{'1':1.}),
                      ('unknown_owned_identities',{'1':'unknown'}),('missing_owned_creation_times',[1]),
                      ('authority_admitted',False),('Popen_returned',False)):
        current[0]={**original,key:value}
        with pytest.raises(ValueError,match='released owned'):
            h.closed_exit(spec)


def test_real_process_AST_cache_deepcopy_keys_source_and_layout_drift(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'_PROJECT_CACHE',{})
    parses=[]
    parser=h.ast.parse
    def real_parse(*args,**kwargs):
        parses.append(kwargs.get('filename'))
        return parser(*args,**kwargs)
    monkeypatch.setattr(h.ast,'parse',real_parse)
    roots=['tools/root.py']
    def seed(folder):
        (folder/'tools').mkdir(parents=True)
        (folder/'tools/root.py').write_text('import tools.known\nfrom tools import absent\n',encoding='utf8')
        (folder/'tools/known.py').write_text('VALUE=1\n',encoding='utf8')
        (folder/'scanner.py').write_text('IDENTITY=1\n',encoding='utf8')
        return {'scanner.py':h.sha((folder/'scanner.py').read_bytes())}
    cold_root=tmp_path/'uncommitted'
    cold_own=seed(cold_root)
    cold=h.cached_project(roots,{'registration':'finite-cold'},cold_own,root=cold_root,commit=False)
    expected_cold=h.json_bytes(cold)
    cold_parse_count=len(parses)
    assert not h._PROJECT_CACHE
    cold['edges'][0]['targets'][0]['sha256']='0'*64
    cold['queried_layout'].clear()
    completed=h.cached_project(roots,{'registration':'finite-cold'},cold_own,root=cold_root)
    assert h.json_bytes(completed)==expected_cold and len(parses)>cold_parse_count
    previous_keys=set(h._PROJECT_CACHE)
    real_scan=h._REAL_SCAN_PROJECT
    def completed_then_timeout(*args,**kwargs):
        real_scan(*args,**kwargs)
        raise TimeoutError('generated completed AST deadline')
    with monkeypatch.context() as patch:
        patch.setattr(h,'_REAL_SCAN_PROJECT',completed_then_timeout)
        with pytest.raises(TimeoutError,match='completed AST deadline'):
            h.cached_project(roots,{'registration':'finite-deadline'},cold_own,root=cold_root)
    assert set(h._PROJECT_CACHE)==previous_keys
    race_root=tmp_path/'own-race'
    race_own=seed(race_root)
    def completed_then_own_drift(*args,**kwargs):
        result=real_scan(*args,**kwargs)
        (race_root/'scanner.py').write_text('IDENTITY=2\n',encoding='utf8')
        return result
    with monkeypatch.context() as patch:
        patch.setattr(h,'_REAL_SCAN_PROJECT',completed_then_own_drift)
        with pytest.raises(ValueError,match='current source drift'):
            h.cached_project(roots,{'registration':'finite-own-race'},race_own,root=race_root,commit=False)
    assert set(h._PROJECT_CACHE)==previous_keys
    own=seed(tmp_path/'first')
    root=tmp_path/'first'
    report=h.cached_project(roots,{'registration':'finite-A','amendment':'finite-A'},own,root=root)
    expected=h.json_bytes(report)
    count=len(parses)
    assert report['queried_layout']['tools/absent.py']['kind']=='absent'
    assert report['queried_layout']['tools']['kind']=='directory'
    report['files'].clear()
    report['unresolved'].clear()
    report['future_ready']=True
    report['edges'][0]['targets'][0]['sha256']='0'*64
    report['chains']['tools/root.py'].append('caller mutation')
    report['queried_layout']['tools/known.py']['kind']='absent'
    fresh=h.cached_project(roots,{'registration':'finite-A','amendment':'finite-A'},own,root=root)
    assert fresh['files'] and fresh['unresolved'] and not fresh['future_ready'] and len(parses)==count
    assert h.json_bytes(fresh)==expected
    h.cached_project(roots,{'registration':'finite-B','amendment':'finite-B'},own,root=root)
    assert len(parses)>count
    other=tmp_path/'other'
    other_own=seed(other)
    h.cached_project(roots,{'registration':'finite-A','amendment':'finite-A'},other_own,root=other)
    for mutation in ('source','negative-module','negative-initializer','namespace','reparse','own'):
        isolated=tmp_path/mutation
        identity=seed(isolated)
        h.cached_project(roots,{'finite':mutation},identity,root=isolated)
        if mutation=='source':(isolated/'tools/root.py').write_text('VALUE=2\n',encoding='utf8')
        elif mutation=='negative-module':(isolated/'tools/absent.py').write_text('VALUE=1\n',encoding='utf8')
        elif mutation=='negative-initializer':(isolated/'tools/__init__.py').write_text('VALUE=1\n',encoding='utf8')
        elif mutation=='namespace':(isolated/'tools/absent').mkdir()
        elif mutation=='own':
            (isolated/'scanner.py').write_text('IDENTITY=2\n',encoding='utf8')
            identity={'scanner.py':h.sha((isolated/'scanner.py').read_bytes())}
        else:
            # A finite path API stand-in tests fail-closed reparse admission;
            # no symlink privilege or platform-specific junction is required.
            original=h.safe
            def forbidden(path,root=h.ROOT,isolated=isolated,original=original):
                if root==isolated and path=='tools/root.py':
                    raise ValueError('source no reparse')
                return original(path,root)
            monkeypatch.setattr(h,'safe',forbidden)
        with pytest.raises(ValueError,match='drift|reparse'):
            h.cached_project(roots,{'finite':mutation},identity,root=isolated)
        if mutation=='reparse':monkeypatch.setattr(h,'safe',original)


def test_warm_AST_never_caches_live_budget_and_full_future_scan_bound(tmp_path,monkeypatch):
    reg=h.registration()
    live=granted_live(activated=True)
    path=tmp_path/'live.json'
    original=h.safe
    monkeypatch.setattr(h,'safe',lambda name,root=h.ROOT:path if name==reg['shared_ledger_path'] else original(name,root))
    path.write_bytes(h.json_bytes(live))
    h.scan_current(reg)
    with h.fixed_authority_scope(check=lambda:None) as scope:
        assert h.registration()==h.registration()==reg and scope.values
        assert h.scan_admission(reg)==260
        live['e360_generated_local_used_s']=0
        path.write_bytes(h.json_bytes(live))
        with pytest.raises(ValueError,match='derived E360 local'):
            h.scan_admission(reg)  # Real fresh ledger/cost derivation despite literal hits.
        assert h.registration()==reg and scope.values
    # A cached AST does not make later preparation free or authorize excess.
    monkeypatch.setattr(h,'ledger_state',lambda *args,**kwargs:([],{'e360_reservations':{},'original_jobs_running':0},
                      {'original':126.361,'generated':346.713},{'original':0,'generated':315.946}))
    with pytest.raises(ValueError,match='downstream future bound'):
        h.scan_admission(reg)


def test_direct_closure_amendment_pins_and_late_report_drift(tmp_path):
    # Copy only the current source report text, never original numeric input.
    reg=h.registration()
    closure=json.loads(os.environ['E360_CURRENT_CLOSURE'])
    report=h.strict_json(h.read_spec(closure))
    pins=h.closure_sources(reg,report,closure_report=closure)
    assert pins[closure['path']]==closure['sha256'] and pins[h.AMENDMENT]==h.AMENDMENT_SHA
    amended=h.amendment()
    assert all(pins[amended[key]['path']]==amended[key]['sha256'] for key in
               ('inherited47','reviewed_design','actual_Astra_review','Astra_review_md','ROOT_reflection'))
    conflicting={**reg,'protected_text_sha256':{**reg['protected_text_sha256'],closure['path']:'0'*64}}
    with pytest.raises(ValueError,match='pin collision'):
        h.closure_sources(conflicting,report,closure_report=closure)
    path=tmp_path/'current-closure.json'
    path.write_bytes(h.read_spec(closure))
    bound={'current-closure.json':h.sha(path.read_bytes())}
    h.check_sources(bound,tmp_path)
    path.write_bytes(h.json_bytes({'late':'changed after pre-array'}))
    state={'first_error':None,'secondary_errors':[]}
    h.fail(state,RuntimeError('first scientific failure'),'reader')
    try:
        h.check_sources(bound,tmp_path)
    except ValueError as error:
        h.fail(state,error,'late-current-report')
    assert state['revoked'] and state['first_error']['message']=='first scientific failure'
    assert state['secondary_errors'][0]['stage']=='late-current-report'


def test_bounded_PowerShell_source_reader_UTF8_SHA_types_and_binary_rejection(tmp_path,monkeypatch):
    raw=b'param([string]$Name)\n# bounded metadata source\n'
    path=tmp_path/'source.ps1';path.write_bytes(raw)
    spec=h.descriptor('source.ps1',raw)
    assert h.read_spec(spec,tmp_path)==raw
    for changed in ({**spec,'bytes':True},{**spec,'bytes':(2<<20)+1},{**spec,'sha256':'0'*64},
                    {**spec,'extra':'unregistered'},{**spec,'path':'../escape.ps1'}):
        with pytest.raises(ValueError):h.read_spec(changed,tmp_path)
    path.write_bytes(b'\xff\xfe')
    with pytest.raises(ValueError):h.read_spec(h.descriptor('source.ps1',b'\xff\xfe'),tmp_path)
    real_open=h.Path.open
    def forbid_binary_open(self,*args,**kwargs):
        if self.suffix in ('.npz','.npy'):raise AssertionError('binary metadata reader bypass')
        return real_open(self,*args,**kwargs)
    monkeypatch.setattr(h.Path,'open',forbid_binary_open)
    for suffix in ('.npz','.npy'):
        with pytest.raises(ValueError):h.read_spec(h.descriptor('unopened'+suffix,b'generated'),tmp_path)


def test_generous_literal_v2_policy_and_source_authority(tmp_path):
    extension=h.resource_extension()
    assert extension['effective_shared_generated_limit_s']==600
    assert extension['effective_generated_local_limit_s']==489 and extension['local_cost_offset']==42
    assert extension['effective_roles']['fixture']['outer_s']==30
    assert extension['effective_roles']['suite']['science_s']==90
    assert extension['effective_roles']['saved_prefix_diagnostic']==h.registration()['roles']['saved_prefix_diagnostic']
    assert sum([3,20,20,30,15,20,120,15,20])==extension['primary_total_reserved_s']==263
    assert extension['successful_suffix_after_s']==[260,240,220,190,175,155,35,20,0]
    path=h.safe(h.RESOURCE_EXTENSION,tmp_path); path.parent.mkdir(parents=True)
    changed=copy.deepcopy(extension); changed['effective_generated_local_limit_s']=False
    path.write_bytes(h.json_bytes(changed))
    with pytest.raises(ValueError,match='literal generous v2'):
        h.resource_extension(tmp_path)
    live=granted_live()
    assert len(live['debits'])==50 and live['e360_generated_local_used_s']==15.946
    assert live['e359_original_local_used_s']==48.181 and live['e359_generated_local_used_s']==11.211


def test_same_budget_literal_activation_and_exact_historical_fix_relocation(tmp_path,monkeypatch):
    authority=h.revalidation_amendment()
    assert authority['new_generated_grant_s']==0 and authority['manual_generated_pass_id']=='e360-generous-p3'
    assert authority['required_exact_prefix_length']==59 and authority['local_cost_offset']==42
    inherited=h.strict_json(h.read_spec(authority['inherited59']))['ledger']
    live=granted_live(activated=True,historical=True)
    assert live['debits']==inherited['debits'] and live['e360_calls']==inherited['e360_calls']
    assert live['e360_passes']==inherited['e360_passes'] and live['e360_resource_grant']==inherited['e360_resource_grant']
    assert live['generated_shared_cumulative_used_s']==209.2320156 and live['e360_generated_local_used_s']==98.275
    for relocation in authority['historical_concrete_fix_relocations']:
        original=copy.deepcopy(relocation['original'])
        raw=h.historical_fix_bytes(original,pass_id='e360-generous-p2')
        assert h.sha(raw)==original['sha256'] and len(raw)==original['bytes']
        assert original==relocation['original']  # Recorded historical descriptor is never rewritten.
        with pytest.raises(ValueError):h.historical_fix_bytes({**original,'sha256':'0'*64},pass_id='e360-generous-p2')
    # The historical event remains valid evidence; current production requires the separate active event.
    live=granted_live(activated=True)
    reg=h.registration();path=tmp_path/'live.json';real_safe=h.safe
    monkeypatch.setattr(h,'safe',lambda name,root=h.ROOT:path if name==reg['shared_ledger_path'] else real_safe(name,root))
    path.write_bytes(h.json_bytes(live));assert len(h.ledger_state(reg)[0])==68
    for field in ('e360_revalidation_activation','e360_revalidation_amendment'):
        bad=copy.deepcopy(live);bad.pop(field);path.write_bytes(h.json_bytes(bad))
        with pytest.raises((ValueError,KeyError)):h.ledger_state(reg)
    for field in ('e360_calls','e360_passes','e360_resource_grant'):
        bad=copy.deepcopy(live);bad[field]={};path.write_bytes(h.json_bytes(bad))
        with pytest.raises((ValueError,KeyError)):h.ledger_state(reg)


def test_manual_p3_exact_once_suite_failure_no_pass4_or_fake_predecessor():
    authority=h.revalidation_amendment()
    live=granted_live(activated=True)
    # Verify genuine historic p3 evidence, including its four immutable relocations; never current admission.
    passed=copy.deepcopy(live['e360_passes']['e360-generous-p3'])
    h.validate_p3_pass('e360-generous-p3',passed,live['debits'])
    for identity in ('e360-generous-p4','e360-auto','e360-generous-p2'):
        with pytest.raises(ValueError):h.validate_p3_pass(identity,passed,live['debits'])
    for field,value in (('modified',1),('predecessor_failed_cap_s',True),('mode','original'),
                        ('actual_failure',live['e360_passes']['e360-generous-p1']['failed_cost']),('concrete_fix',[])):
        with pytest.raises(ValueError):h.validate_p3_pass('e360-generous-p3',{**passed,field:value},live['debits'])
    predecessors=[debit for debit in live['debits'] if debit['descriptor']==authority['actual_predecessor_failure']]
    assert len(predecessors)==1
    with pytest.raises(ValueError,match='once debited'):
        h.validate_p3_pass('e360-generous-p3',passed,live['debits']+[copy.deepcopy(predecessors[0])])


def test_active_admin_recovery60_history_event_and_six_pass_bound_relocations(tmp_path,monkeypatch):
    authority=h.active_revalidation_amendment()
    historical=h.revalidation_amendment()
    assert historical['pass4_authorized'] is False and authority['pass5_authorized'] is False
    assert authority['new_generated_grant_s']==0 and authority['manual_generated_pass_id']=='e360-generous-p4'
    inherited=h.strict_json(h.read_spec(authority['inherited60']))['ledger']
    live=granted_live(activated=True,admin_historical=True)
    for key in ('debits','e360_calls','e360_passes','e360_resource_grant','e360_revalidation_activation','e360_revalidation_amendment'):
        assert live[key]==inherited[key]
    generated=math.fsum(d['actual_outer_full_return_s'] for d in live['debits'] if d['mode']=='generated')
    local=math.fsum(d['actual_outer_full_return_s'] for d in live['debits'][42:] if d['mode']=='generated')
    assert type(live['generated_shared_cumulative_used_s']) is float
    assert round(h.registration()['generated_shared_baseline_s']+generated,9)==round(live['generated_shared_cumulative_used_s'],9)==209.6180156
    assert round(local,9)==live['e360_generated_local_used_s']==98.661
    assert inherited['e360_passes']['e360-generous-p3']['failed_cost']==authority['actual_predecessor_failure']
    failure=h.charged_cost(authority['actual_predecessor_failure'],'generated')
    assert failure['actual_returned_exit_code']==0 and failure['internal_completion_valid'] is False
    rows=authority['historical_concrete_fix_relocations'];assert len(rows)==6
    for row in rows:
        before=copy.deepcopy(row['original'])
        raw=h.historical_fix_bytes(before,pass_id=row['pass_id'])
        assert len(raw)==before['bytes'] and h.sha(raw)==before['sha256'] and before==row['original']
        with pytest.raises(ValueError):h.historical_fix_bytes(before,pass_id='e360-wrong-pass')
        with pytest.raises(ValueError):h.historical_fix_bytes({**before,'sha256':'0'*64},pass_id=row['pass_id'])
    reg=h.registration();path=tmp_path/'live.json';real_safe=h.safe
    monkeypatch.setattr(h,'safe',lambda name,root=h.ROOT:path if name==reg['shared_ledger_path'] else real_safe(name,root))
    h.validate_live_calls(live,live['debits'])  # Genuine preserved p4 history, not current admission.
    current=granted_live(activated=True)
    path.write_bytes(h.json_bytes(current));assert len(h.ledger_state(reg)[0])==68
    for field in ('e360_active_revalidation_activation','e360_active_revalidation_amendment'):
        bad=copy.deepcopy(current);bad.pop(field);path.write_bytes(h.json_bytes(bad))
        with pytest.raises((ValueError,KeyError)):h.ledger_state(reg)
        bad=copy.deepcopy(current);bad[field]=False;path.write_bytes(h.json_bytes(bad))
        with pytest.raises(ValueError):h.ledger_state(reg)
    bad=copy.deepcopy(current);bad['debits'][59]['actual_outer_full_return_s']=0.;path.write_bytes(h.json_bytes(bad))
    with pytest.raises(ValueError):h.ledger_state(reg)
    # A corrupted immutable snapshot rejects at the production descriptor reader, not a mocked success.
    row=rows[-1];target=h.safe(row['snapshot']['path'],tmp_path);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(h.read_spec(row['snapshot']))
    current_authority=h.current_revalidation_amendment()
    monkeypatch.setattr(h,'current_revalidation_amendment',lambda root=h.ROOT:current_authority)
    assert h.historical_fix_bytes(row['original'],pass_id=row['pass_id'],root=tmp_path)==h.read_spec(row['snapshot'])
    target.write_bytes(target.read_bytes()+b'\n# drift\n')
    with pytest.raises(ValueError):h.historical_fix_bytes(row['original'],pass_id=row['pass_id'],root=tmp_path)


def test_manual_p4_exact_failed_Ruff_correction_once_debit_no_p5():
    authority=h.active_revalidation_amendment();live=granted_live(activated=True)
    passed=copy.deepcopy(live['e360_passes']['e360-generous-p4'])
    h.validate_p4_pass('e360-generous-p4',passed,live['debits'])
    for identity in ('e360-generous-p5','e360-generous-p3','e360-auto'):
        with pytest.raises(ValueError):h.validate_p4_pass(identity,passed,live['debits'])
    for field,value in (('modified',1),('predecessor_failed_cap_s',True),('mode','original'),('concrete_fix',[]),
                        ('actual_failure',live['e360_passes']['e360-generous-p2']['failed_cost'])):
        with pytest.raises(ValueError):h.validate_p4_pass('e360-generous-p4',{**passed,field:value},live['debits'])
    predecessor=[d for d in live['debits'] if d['descriptor']==authority['actual_predecessor_failure']]
    assert len(predecessor)==1
    with pytest.raises(ValueError,match='once debited'):
        h.validate_p4_pass('e360-generous-p4',passed,live['debits']+[copy.deepcopy(predecessor[0])])


def test_current_PS_source_recovery62_fixed_history_and_ten_relocations(tmp_path,monkeypatch):
    authority=h.source_recovery_amendment()
    inherited=h.strict_json(h.read_spec(authority['inherited62']))['ledger']
    live=granted_live(activated=True,source_historical=True)
    assert authority['manual_generated_pass_id']=='e360-generous-p5' and authority['pass6_authorized'] is False
    assert authority['new_generated_grant_s']==0 and authority['primary_total_reserved_s']==263
    for key in ('debits','e360_calls','e360_passes','e360_resource_grant',*authority['old_activation_fields_preserved']):
        assert live[key]==inherited[key]
    assert round(live['generated_shared_cumulative_used_s'],9)==210.5610156 and live['e360_generated_local_used_s']==99.604
    assert len(authority['historical_concrete_fix_relocations'])==10
    for row in authority['historical_concrete_fix_relocations']:
        spec=copy.deepcopy(row['original']);raw=h.historical_fix_bytes(spec,pass_id=row['pass_id'])
        assert len(raw)==spec['bytes'] and h.sha(raw)==spec['sha256'] and spec==row['original']
        with pytest.raises(ValueError):h.historical_fix_bytes(spec,pass_id='e360-generous-p5')
        for field,value in (('path','tools/unregistered.py'),('bytes',True),('sha256','0'*64)):
            with pytest.raises((ValueError,FileNotFoundError)):h.historical_fix_bytes({**spec,field:value},pass_id=row['pass_id'])
    reg=h.registration();path=tmp_path/'current.json';original_safe=h.safe
    monkeypatch.setattr(h,'safe',lambda name,root=h.ROOT:path if name==reg['shared_ledger_path'] else original_safe(name,root))
    h.validate_live_calls(live,live['debits'])  # Genuine fixed p5 context, never current admission.
    current=granted_live(activated=True)
    path.write_bytes(h.json_bytes(current));assert len(h.ledger_state(reg)[0])==68
    for field in (*authority['old_activation_fields_preserved'],'e360_source_recovery_activation','e360_source_recovery_amendment'):
        for missing in (True,False):
            bad=copy.deepcopy(current)
            if missing:bad.pop(field)
            else:bad[field]=False
            path.write_bytes(h.json_bytes(bad))
            with pytest.raises((ValueError,KeyError)):h.ledger_state(reg)
    for old in (granted_live(activated=True,historical=True),granted_live(activated=True,admin_historical=True),live):
        path.write_bytes(h.json_bytes(old))
        with pytest.raises((ValueError,KeyError)):h.ledger_state(reg)


def test_manual_p5_actual_failed_scanner_C5_once_debit_and_no_p6():
    authority=h.source_recovery_amendment();live=granted_live(activated=True)
    passed=copy.deepcopy(live['e360_passes']['e360-generous-p5'])  # Actual as-executed63; no fresh file_spec.
    failure=h.charged_cost(passed['actual_failure'],'generated')
    assert failure['actual_returned_exit_code']==1 and failure['call_role']=='scanner'
    h.validate_p5_pass('e360-generous-p5',passed,live['debits'])
    for identity in ('e360-generous-p6','e360-auto','e360-generous-p4'):
        with pytest.raises(ValueError):h.validate_p5_pass(identity,passed,live['debits'])
    for field,value in (('modified',1),('predecessor_failed_cap_s',True),('mode','original'),('concrete_fix',[]),
                        ('concrete_fix',[authority['concrete_correction_snapshot']]),
                        ('actual_failure',h.active_revalidation_amendment()['actual_predecessor_failure'])):
        with pytest.raises(ValueError):h.validate_p5_pass('e360-generous-p5',{**passed,field:value},live['debits'])
    predecessor=[d for d in live['debits'] if d['descriptor']==authority['actual_predecessor_failure']]
    assert len(predecessor)==1
    for changed in (live['debits']+[copy.deepcopy(predecessor[0])],[d for d in live['debits'] if d not in predecessor]):
        with pytest.raises(ValueError,match='once debited'):h.validate_p5_pass('e360-generous-p5',passed,changed)
    for spec in passed['concrete_fix']:
        bad=copy.deepcopy(passed);bad['concrete_fix'].remove(spec)
        with pytest.raises(ValueError):h.validate_p5_pass('e360-generous-p5',bad,live['debits'])
    for key in passed['source_sha256']:
        for value in ('0'*64,True):
            bad=copy.deepcopy(passed);bad['source_sha256'][key]=value
            with pytest.raises(ValueError):h.validate_p5_pass('e360-generous-p5',bad,live['debits'])
    for field,value in (('source_sha256',{}),('source_sha256',False),('concrete_fix',False)):
        with pytest.raises(ValueError):h.validate_p5_pass('e360-generous-p5',{**passed,field:value},live['debits'])


def test_current_Ruff_recovery63_history_six_fields_fourteen_relocations(tmp_path,monkeypatch):
    authority=h.ruff_recovery_amendment()
    inherited=h.strict_json(h.read_spec(authority['inherited63']))['ledger']
    live=granted_live(activated=True,ruff_historical=True)
    assert authority['manual_generated_pass_id']=='e360-generous-p6' and authority['pass7_authorized'] is False
    assert authority['new_generated_grant_s']==0 and authority['primary_total_reserved_s']==263
    for key in ('debits','e360_calls','e360_passes','e360_resource_grant',*authority['old_activation_fields_preserved']):
        assert live[key]==inherited[key]
    assert round(live['generated_shared_cumulative_used_s'],9)==210.9630156 and live['e360_generated_local_used_s']==100.006
    rows=authority['historical_concrete_fix_relocations'];assert len(rows)==14
    for row in rows:
        raw=h.historical_fix_bytes(row['original'],pass_id=row['pass_id'])
        assert len(raw)==row['original']['bytes'] and h.sha(raw)==row['original']['sha256']
        with pytest.raises(ValueError):h.historical_fix_bytes(row['original'],pass_id='e360-wrong-pass')
    reg=h.registration();path=tmp_path/'current.json';real_safe=h.safe
    monkeypatch.setattr(h,'safe',lambda name,root=h.ROOT:path if name==reg['shared_ledger_path'] else real_safe(name,root))
    h.validate_live_calls(live,live['debits'])
    current=granted_live(activated=True)
    path.write_bytes(h.json_bytes(current));assert len(h.ledger_state(reg)[0])==68
    for field in (*authority['old_activation_fields_preserved'],'e360_ruff_recovery_activation','e360_ruff_recovery_amendment'):
        for value in (None,False):
            bad=copy.deepcopy(current)
            if value is None:bad.pop(field)
            else:bad[field]=value
            path.write_bytes(h.json_bytes(bad))
            with pytest.raises((ValueError,KeyError)):h.ledger_state(reg)
    path.write_bytes(h.json_bytes(granted_live(activated=True,source_historical=True)))
    with pytest.raises((ValueError,KeyError)):h.ledger_state(reg)
    row=rows[-1];target=h.safe(row['snapshot']['path'],tmp_path);target.parent.mkdir(parents=True,exist_ok=True)
    raw=h.read_spec(row['snapshot']);target.write_bytes(raw)
    monkeypatch.setattr(h,'current_revalidation_amendment',lambda root=h.ROOT:authority)
    assert h.historical_fix_bytes(row['original'],pass_id='e360-generous-p5',root=tmp_path)==raw
    target.write_bytes(raw+b'\n# source drift\n')
    with pytest.raises(ValueError):h.historical_fix_bytes(row['original'],pass_id='e360-generous-p5',root=tmp_path)


def test_manual_p6_exact_Ruff_exit1_internaltrue_C6_fresh_own9_and_no_p7():
    authority=h.ruff_recovery_amendment();live=granted_live(activated=True,ruff_historical=True)
    executed=h.strict_json(h.read_spec(h.current_revalidation_amendment()['inherited68']))['ledger']
    passed=copy.deepcopy(executed['e360_passes']['e360-generous-p6'])
    live=executed
    failure=h.charged_cost(passed['actual_failure'],'generated')
    assert failure['actual_returned_exit_code']==1 and failure['call_role']=='ruff' and failure['internal_completion_valid'] is True
    h.validate_p6_pass('e360-generous-p6',passed,live['debits'])
    for identity in ('e360-generous-p7','e360-auto','e360-generous-p5'):
        with pytest.raises(ValueError):h.validate_p6_pass(identity,passed,live['debits'])
    for field,value in (('modified',1),('predecessor_failed_cap_s',True),('mode','original'),('concrete_fix',[]),
                        ('actual_failure',h.source_recovery_amendment()['actual_predecessor_failure'])):
        with pytest.raises(ValueError):h.validate_p6_pass('e360-generous-p6',{**passed,field:value},live['debits'])
    for spec in passed['concrete_fix']:
        bad=copy.deepcopy(passed);bad['concrete_fix'].remove(spec)
        with pytest.raises(ValueError):h.validate_p6_pass('e360-generous-p6',bad,live['debits'])
    for name in passed['source_sha256']:
        bad=copy.deepcopy(passed);bad['source_sha256'][name]=False
        with pytest.raises(ValueError):h.validate_p6_pass('e360-generous-p6',bad,live['debits'])
    predecessor=[d for d in live['debits'] if d['descriptor']==authority['actual_predecessor_failure']];assert len(predecessor)==1
    with pytest.raises(ValueError,match='once debited'):
        h.validate_p6_pass('e360-generous-p6',passed,live['debits']+[copy.deepcopy(predecessor[0])])


def test_actual_call_admission_source_role_types_replay_and_live_suffix(tmp_path,monkeypatch):
    # This finite old-pass request exercises its original 600/489 admission boundary.
    historical=granted_live(activated=True)
    ledger_path=tmp_path/'historical-live.json';ledger_path.write_bytes(h.json_bytes(historical))
    shared=h.registration()['shared_ledger_path'];real_safe=h.safe
    monkeypatch.setattr(h,'safe',lambda name,root=h.ROOT:ledger_path if name==shared else real_safe(name,root))
    extension=h.file_spec(h.RESOURCE_EXTENSION)
    closure=json.loads(os.environ['E360_CURRENT_CLOSURE'])
    request={'schema':'flylab.m3.e360.helper-request.v1','call_id':'e360-finite-build','pass_id':'e360-finite-pass',
             'call_role':'fixture_build','mode':'generated','action':'build','role':'fixture',
             'expected_output':h.BASE+'/parent-e360-finite-proposal.json',
             'completion':h.BASE+'/parent-e360-finite-completion.json','source_sha256':h.own_sources(),
             'closure_report':closure,'resource_extension':extension,'revalidation_amendment':h.file_spec(h.REVALIDATION_AMENDMENT)}
    raw=h.json_bytes(request); path=tmp_path/'request.json';path.write_bytes(raw)
    req_spec=h.descriptor('request.json',raw)
    original_spec=h.file_spec
    monkeypatch.setattr(h,'file_spec',lambda name,root=h.ROOT: req_spec if name=='request.json' else original_spec(name,root))
    request['_request_path']='request.json'
    sem=granted_live()
    sem['e360_calls'][request['call_id']]={'state':'begun','request':req_spec,'pass_id':request['pass_id'],
        'mode':'generated','resource_extension':extension,'revalidation_amendment':h.file_spec(h.REVALIDATION_AMENDMENT),'reserved_s':20,'future_reserved_s':220}
    monkeypatch.setattr(h,'ledger_state',lambda *args,**kwargs:([],sem,{},{}))
    assert h.check_call_request(request)['outer_s']==20
    for key,value in (('mode',True),('role','suite'),('resource_extension',{**extension,'sha256':'0'*64}),
                      ('source_sha256',{}),('pass_id','e360-replayed')):
        bad={**request,key:value}
        with pytest.raises(ValueError):h.check_call_request(bad)
    sem['e360_calls'][request['call_id']]['state']='charged'
    with pytest.raises(ValueError,match='consumed call'):h.check_call_request(request)
    sem['e360_calls'][request['call_id']]['state']='begun'
    sem['e360_generated_local_used_s']=480.
    with pytest.raises(ValueError,match='future admission fits'):h.check_call_request(request)
    sem['e360_generated_local_used_s']=15.946
    sem['e360_blocking_holds']={'owned-unknown':{}}
    with pytest.raises(ValueError,match='unknown owned'):h.check_call_request(request)


def test_pending_actual_substitution_order_and_debit_backed_unknown(monkeypatch):
    live=granted_live(); extension=h.resource_extension(); specs={}
    real_read=h.read_spec
    def save(name,value):
        raw=h.json_bytes(value); spec=h.descriptor(name,raw);specs[name]=raw
        return spec
    def read(spec,root=h.ROOT):
        if spec['path'] not in specs:return real_read(spec,root)
        raw=specs[spec['path']]
        assert len(raw)==spec['bytes'] and h.sha(raw)==spec['sha256']
        return raw
    monkeypatch.setattr(h,'read_spec',read)
    pass_id='e360-finite-pass';sources=h.own_sources()
    live['e360_passes'][pass_id]={'mode':'generated','next_index':4,'source_sha256':sources,'failed_cost':None,'failed_cap_s':0}
    debits=copy.deepcopy(live['debits'])
    roles=extension['generated_call_order'][:4]
    for index,role in enumerate(roles):
        call_id='e360-finite-'+role.replace('_','-')
        request={'schema':'flylab.m3.e360.helper-request.v1','call_id':call_id,'pass_id':pass_id,'call_role':role,'mode':'generated',
                 'resource_extension':h.file_spec(h.RESOURCE_EXTENSION),'revalidation_amendment':h.file_spec(h.REVALIDATION_AMENDMENT),'source_sha256':sources}
        cap=[3,20,20,30][index]
        call={'state':'charged' if index<3 else 'actual_pending','request':save(call_id+'.json',request),
              'pass_id':pass_id,'call_role':role,'mode':'generated','sequence_index':index,'resource_extension':request['resource_extension'],
              'reserved_s':cap,'future_reserved_s':extension['successful_suffix_after_s'][index],
              'cost_id':call_id if index<3 else None}
        live['e360_calls'][call_id]=call
        if index<3:debits.append({'cost_id':call_id})
    worker_id='e360-finite:fixture';call_id='e360-finite-fixture-worker'
    actual=save('pending.json',{'actual_outer_full_return_s':1.25,'actual_returned_exit_code':1})
    live['e360_reservations'][worker_id]={'mode':'generated','role':'fixture','reserved_s':30,'pending_reservations':{'suite':120}}
    live['e360_pending_actuals'][worker_id]={'mode':'generated','call_id':call_id,'actual_tool_return':actual,
                                          'resource_extension':h.file_spec(h.RESOURCE_EXTENSION),'revalidation_amendment':h.file_spec(h.REVALIDATION_AMENDMENT)}
    h.validate_live_calls(live,debits)
    # Worker's30 cap is replaced by actual1.25; no future helper is double counted.
    assert h.prospective_remaining(live,'generated')==1.25
    live['e360_calls']['e360-finite-fixture-build']['state']='begun'
    assert h.prospective_remaining(live,'generated')==241.25
    live['e360_calls']['e360-finite-fixture-build']['state']='charged'
    bad=copy.deepcopy(live); bad['e360_calls'][call_id]['reserved_s']=True
    with pytest.raises(ValueError,match='call cap'):h.validate_live_calls(bad,debits)
    bad=copy.deepcopy(live); bad['e360_calls'][call_id]['sequence_index']=False
    with pytest.raises(ValueError,match='sequence'):h.validate_live_calls(bad,debits)
    live['e360_pending_actuals'].clear();live['e360_reservations'].clear()
    live['e360_calls'][call_id].update(state='charged',cost_id=worker_id);debits.append({'cost_id':worker_id})
    live['e360_blocking_holds'][worker_id]={'mode':'generated','debit_backed':True,'numeric_reserved_s':0,
                                         'resource_extension':h.file_spec(h.RESOURCE_EXTENSION),'revalidation_amendment':h.file_spec(h.REVALIDATION_AMENDMENT)}
    h.validate_live_calls(live,debits)
    assert h.prospective_remaining(live,'generated')==0
    live['e360_blocking_holds'][worker_id]['numeric_reserved_s']=30
    with pytest.raises(ValueError,match='nonnumeric hold'):h.validate_live_calls(live,debits)


def test_Q_helper_actual_timely_descriptor_completion_and_late_rejection(tmp_path,monkeypatch):
    ext=h.file_spec(h.RESOURCE_EXTENSION);sources=h.own_sources();specs={};real_read=h.read_spec
    def save(name,value):
        raw=h.json_bytes(value);specs[name]=raw;return h.descriptor(name,raw)
    def read(spec,root=h.ROOT):
        if spec['path'] not in specs:return real_read(spec,root)
        raw=specs[spec['path']];assert len(raw)==spec['bytes'] and h.sha(raw)==spec['sha256'];return raw
    monkeypatch.setattr(h,'read_spec',read)
    q={'qualification_call_id':'e360-finite-Q','resource_extension':ext,'revalidation_amendment':h.file_spec(h.REVALIDATION_AMENDMENT),'source_sha256':sources}
    qspec=save('Q.json',q);req=save('request.json',{'action':'qualification'})
    call={'state':'charged','call_role':'qualification','mode':'generated','cost_id':'e360-finite-Q',
          'request':req,'pass_id':'e360-finite-pass','revalidation_amendment':h.file_spec(h.REVALIDATION_AMENDMENT)}
    actual=save('actual.json',{'actual_outer_full_return_s':1.,'actual_returned_exit_code':0})
    complete={'schema':'flylab.m3.e360.helper-internal-completion.v1','completed':True,'entry_perf_counter':10.,
              'pass_id':call['pass_id'],'call_role':'qualification','internal_full_return_s':.5,'call_id':q['qualification_call_id'],'request':req,
              'resource_extension':ext,'revalidation_amendment':h.file_spec(h.REVALIDATION_AMENDMENT),'source_sha256':sources,'expected_output':qspec}
    completion=save('completion.json',complete)
    cost={'schema':'flylab.m3.e360.administrative-full-return.v1','mode':'generated','charged_once':True,
          'cost_id':'e360-finite-Q','actual_outer_full_return_s':1.,'actual_returned_exit_code':0,'actual_tool_return':actual,
          'request':req,'call_id':q['qualification_call_id'],'call_role':'qualification','pass_id':call['pass_id'],
          'resource_extension':ext,'revalidation_amendment':h.file_spec(h.REVALIDATION_AMENDMENT),'source_sha256':sources,'expected_output':qspec,'internal_completion':completion,'internal_completion_valid':True}
    sem={'e360_calls':{q['qualification_call_id']:call}}
    good=save('cost.json',cost);debits=[{'cost_id':call['cost_id'],'descriptor':good}]
    assert h.qualification_admin_proof(q,qspec,sem,debits)==cost
    for field,value in (('actual_returned_exit_code',1),('actual_outer_full_return_s',21.),('internal_completion_valid',False),
                        ('expected_output',{**qspec,'sha256':'0'*64}),('source_sha256',{}),('revalidation_amendment',{})):
        changed={**cost,field:value}
        if field=='actual_outer_full_return_s':changed['actual_tool_return']=save('late-actual.json',{'actual_outer_full_return_s':21.,'actual_returned_exit_code':0})
        changed_spec=save('changed.json',changed)
        with pytest.raises(ValueError):h.qualification_admin_proof(q,qspec,sem,[{'cost_id':call['cost_id'],'descriptor':changed_spec}])
    for elapsed in (True,19.,float('inf')):
        if elapsed==float('inf'):
            with pytest.raises(ValueError):h.json_bytes({**complete,'internal_full_return_s':elapsed})
            continue
        bad=save('bad-completion.json',{**complete,'internal_full_return_s':elapsed})
        cost_spec=save('bad-cost.json',{**cost,'internal_completion':bad})
        with pytest.raises(ValueError):h.qualification_admin_proof(q,qspec,sem,[{'cost_id':call['cost_id'],'descriptor':cost_spec}])
    with pytest.raises(ValueError,match='exactly one'):h.qualification_admin_proof(q,qspec,sem,debits+debits)
    call['state']='begun'
    with pytest.raises(ValueError,match='once charged'):h.qualification_admin_proof(q,qspec,sem,debits)


    # Actual full Q value consumer, with explicit finite closed/admission seams.
    isolate_compact_boundary(monkeypatch)
    names=h.test_functions();authority=h.suite_recovery_amendment()
    direct={key:h.file_spec(path) for key,path in (('registration',h.REGISTRATION),('amendment',h.AMENDMENT),
        ('resource_extension',h.RESOURCE_EXTENSION),('revalidation_amendment',h.SUITE_RECOVERY_AMENDMENT))}
    report={'finite_semantic_closure':True};closure=save('q-closure.json',report)
    full_sources={**sources,closure['path']:closure['sha256']}
    assert len(sources)==9 and len(full_sources)>9
    closed_rows={};contracts={}
    exits={role:save('q-'+role+'-exit.json',{'finite_closed_seam':role}) for role in ('fixture','suite')}
    case={'schema':'flylab.m3.e360.current-pytest-case.v1','source_sha256':sources,
        'expected_functions':names,'receipt':'q-case-receipt.json','pytest_ini':'empty.ini',
        'fixture_exit':exits['fixture'],'closure_report':closure,**direct}
    case_pin=save('q-case.json',case)
    case_receipt={'schema':'flylab.m3.e360.complete-pytest-receipt.v1','source_sha256':sources,'source_before':sources,
        'closure_report':closure,**direct,'case_contract':case_pin,'expected_functions':names,
        'collected_functions':names,'phases':{name:{phase:'passed' for phase in ('setup','call','teardown')} for name in names},
        'collection_errors':[],'exit_code':0,'original_calls':0,'original_array_reads':0,'complete_current_cases':True}
    case_receipt_pin=save('q-case-receipt.json',case_receipt)
    for role in ('fixture','suite'):
        contract={'role':role,'mode':'generated','pass_id':'e360-generous-p9','source_sha256':full_sources,
            'closure_report':closure,**direct}
        name='q-'+role+'-contract.json';(tmp_path/name).write_bytes(h.json_bytes(contract))
        contracts[role]=contract
        closed_rows[role]={'role':role,'mode':'generated','pass_id':'e360-generous-p9','original_calls':0,
            'source_sha256':copy.deepcopy(full_sources),'contract':h.file_spec(name,tmp_path),
            'envelope':save('q-'+role+'-envelope.json',{'case_receipt':case_receipt_pin}),
            'closure_report':closure,**direct}
    full_q={'schema':'flylab.m3.e360.adapter-qualification.v1','source_sha256':sources,
        'original_calls':0,'original_array_reads':0,'closed':True,'fixture_exit':exits['fixture'],
        'suite_exit':exits['suite'],'case_receipt':case_receipt_pin,'closure_report':closure,**direct,
        'qualification_call_id':'e360-finite-q-value'}
    actual_file_spec=h.file_spec
    monkeypatch.setattr(h,'file_spec',lambda name,root=h.ROOT:next((pin for pin in direct.values() if pin['path']==name),None)
        or actual_file_spec(name,root))
    monkeypatch.setattr(h,'own_sources',lambda root=h.ROOT:copy.deepcopy(sources))
    monkeypatch.setattr(h,'test_functions',lambda root=h.ROOT:list(names))
    monkeypatch.setattr(h,'registration',lambda root=h.ROOT:{})
    monkeypatch.setattr(h,'suite_recovery_amendment',lambda root=h.ROOT:authority)
    monkeypatch.setattr(h,'prospective_authority_spec',lambda root=h.ROOT:direct['revalidation_amendment'])
    monkeypatch.setattr(h,'scan_current',lambda **kwargs:copy.deepcopy(report))
    monkeypatch.setattr(h,'closure_sources',lambda reg,actual_report,root=h.ROOT,*,closure_report=None:
        copy.deepcopy(full_sources) if actual_report==report and closure_report==closure else {})
    monkeypatch.setattr(h,'check_sources',lambda pins,root=h.ROOT:h.typed_equal(pins,full_sources,'finite complete checked closure'))
    monkeypatch.setattr(h,'closed_exit',lambda pin,**kwargs:copy.deepcopy(closed_rows[next(role for role in exits if pin==exits[role])]))
    assert h.validate_qualification_value(full_q,root=tmp_path)==full_q
    for role in ('fixture','suite'):
        original=copy.deepcopy(closed_rows[role])
        for mutation in ('missing','unrelated','stale','pass','role','direct','report','contract_pin'):
            changed=copy.deepcopy(original)
            if mutation=='missing':changed['source_sha256'].pop(closure['path'])
            elif mutation=='unrelated':changed['source_sha256']['unrelated.py']='a'*64
            elif mutation=='stale':changed['source_sha256'][closure['path']]='0'*64
            elif mutation=='pass':changed['pass_id']='e360-generous-p10'
            elif mutation=='role':changed['role']='suite' if role=='fixture' else 'fixture'
            elif mutation=='direct':changed['registration']={}
            elif mutation=='report':changed['closure_report']={}
            else:changed['contract']={**changed['contract'],'sha256':'0'*64}
            closed_rows[role]=changed
            with pytest.raises(ValueError):h.validate_qualification_value(full_q,root=tmp_path)
        closed_rows[role]=original
        contract_path=tmp_path/('q-'+role+'-contract.json');raw=contract_path.read_bytes()
        contract_path.write_bytes(h.json_bytes({**contracts[role],'source_sha256':sources}))
        closed_rows[role]['contract']=actual_file_spec(contract_path.name,tmp_path)
        with pytest.raises(ValueError):h.validate_qualification_value(full_q,root=tmp_path)
        contract_path.write_bytes(raw);closed_rows[role]=original
    assert h.validate_qualification_value(full_q,root=tmp_path)==full_q

def test_helper_same_ENTRY_callback_cold_read_and_durable_late_failure(tmp_path,monkeypatch):
    now=[10.];clock=h.Clock(10.,{'internal_s':1},clock=lambda:now[0])
    monkeypatch.setattr(h,'ADMIN_CHECK',clock.check)
    path=tmp_path/'metadata.json';path.write_bytes(h.json_bytes({'metadata_only':True}))
    spec=h.file_spec('metadata.json',tmp_path)
    real=h.lib.read_spec
    def slow(spec,root):
        value=real(spec,root);now[0]=12.;return value
    monkeypatch.setattr(h.lib,'read_spec',slow)
    with pytest.raises(TimeoutError,match='same-ENTRY'):h.read_spec(spec,tmp_path)
    assert path.exists() and h.strict_json(path.read_bytes())=={'metadata_only':True}
    with pytest.raises(TimeoutError):h.durable(tmp_path/'late-Q.json',b'{}\n',cap=64)
    assert not (tmp_path/'late-Q.json').exists()


def test_ROOT_helper_entry_before_import_prepublication_and_pending_source_protocol(tmp_path):
    helper=h.safe(h.BASE+'/parent-e360-root-metadata-helper.py').read_text('utf8')
    account=h.safe(h.BASE+'/parent-e360-root-account.ps1').read_text('utf8')
    assert helper.index('ENTRY = time.perf_counter()')<helper.index('request_raw =')<helper.index('from tools import')
    assert helper.index('h.validate_qualification_value(qualification')<helper.index('h.durable(h.safe(request["qualification"])')
    actual=account[account.index("if($Action -eq 'actual'){"):account.index("if($Action -eq 'admin' -or")]
    assert actual.index('$ledger.e360_pending_actuals[$costID]')<actual.index('Write-Json $ledgerName')<actual.index('exit 0')
    assert 'numeric_reserved_s=0' in account and "'fallback'" in account and 'Validate-Call-State' in account
    assert 'Prospective $req.mode $callPolicy.outer_s $callPolicy.future_s' in account


    # SOURCE seam: execute the actual helper boundary function, never its bootstrap or main.
    tree=h.ast.parse(helper)
    function=next(node for node in tree.body if isinstance(node,h.ast.FunctionDef) and node.name=='validate_closed_full_map')
    actual_suite=next(node for node in tree.body if isinstance(node,h.ast.FunctionDef) and node.name=='main')
    assert sum(isinstance(node,h.ast.Call) and isinstance(node.func,h.ast.Name)
               and node.func.id=='validate_closed_full_map' for node in h.ast.walk(actual_suite))==2
    suite_if=next(node for node in h.ast.walk(actual_suite) if isinstance(node,h.ast.If)
                  and isinstance(node.test,h.ast.Compare) and isinstance(node.test.left,h.ast.Subscript)
                  and isinstance(node.test.left.value,h.ast.Name) and node.test.left.value.id=='request'
                  and isinstance(node.test.comparators[0],h.ast.Constant) and node.test.comparators[0].value=='suite')
    qualification_if=next(node for node in h.ast.walk(actual_suite) if isinstance(node,h.ast.If)
                         and isinstance(node.test,h.ast.Compare) and isinstance(node.test.left,h.ast.Name)
                         and node.test.left.id=='action' and isinstance(node.test.comparators[0],h.ast.Constant)
                         and node.test.comparators[0].value=='qualification')
    suite_prefix=suite_if.body[:4]
    qualification_prefix=qualification_if.body[:4]
    # Both prefixes end at the production validation calls, before CASE/Q publication.
    assert isinstance(suite_prefix[-1],h.ast.Expr) and suite_prefix[-1].value.func.id=='validate_closed_full_map'
    assert isinstance(qualification_prefix[-1],h.ast.For)
    namespace={}
    exec(compile(h.ast.Module(body=[function],type_ignores=[]),'actual-helper-full-map-SOURCE-seam','exec'),namespace)  # noqa: S102 -- exact production function, finite Source seam.
    boundary=namespace['validate_closed_full_map']
    # Real finite text files let actual check_sources invoke its callback per source.
    full={}
    for name in [*(f'own-{index}.py' for index in range(9)),'non-own.py','closure.json']:
        raw=(name+'\n').encode('utf8');(tmp_path/name).write_bytes(raw);full[name]=h.sha(raw)
    own={key:full[key] for key in full if key.startswith('own-')}
    direct={key:{'path':key+'.json','bytes':1,'sha256':'d'*64} for key in
            ('registration','amendment','resource_extension','revalidation_amendment','closure_report')}
    request={'pass_id':'e360-generous-p14','role':'suite',
             'fixture_exit':{'path':'fixture.json'},'suite_exit':{'path':'suite.json'},'source_sha256':copy.deepcopy(own),
             'closure_report':direct['closure_report'],'revalidation_amendment':direct['revalidation_amendment']}
    report={'fresh':'direct-report'}
    observed=[]
    for role in ('fixture','suite'):
        contract={'mode':'generated','role':role,'pass_id':request['pass_id'],'source_sha256':copy.deepcopy(full),**copy.deepcopy(direct)}
        closed={**copy.deepcopy(contract),'contract':{'path':role+'.json','bytes':1,'sha256':'c'*64}}
        state={'contract':copy.deepcopy(contract),'closed':copy.deepcopy(closed),'current':copy.deepcopy(full),'request':copy.deepcopy(request),'own':copy.deepcopy(own),'report':copy.deepcopy(report)}
        events=[]
        def verified_sources(pins,*,check,events=events):
            events.append('source-begin')
            h.check_sources(pins,root=tmp_path,check=check)
            observed.append(copy.deepcopy(pins))
            events.append('source-end')
        def make_api(state,*,role=role,contract=contract,closed=closed,verify=verified_sources):
            return SimpleNamespace(
                require=h.require,typed_equal=h.typed_equal,strict_json=h.strict_json,
                bounded_json_read=lambda spec,**kwargs:h.json_bytes(state['contract'] if spec['path']==role+'.json' else {**copy.deepcopy(contract),'role':'suite' if role=='fixture' else 'fixture'}),
                closed_exit=lambda spec:copy.deepcopy(state['closed'] if spec['path']==role+'.json' else {**copy.deepcopy(closed),'role':'suite' if role=='fixture' else 'fixture','contract':{'path':spec['path'],'bytes':1,'sha256':'c'*64}}),
                read_spec=lambda spec:h.json_bytes(state['report']),
                scan_current=lambda:copy.deepcopy(report),closure_sources=lambda *args,**kwargs:copy.deepcopy(state['current']),
                registration=dict,own_sources=lambda:copy.deepcopy(state['own']),
                REGISTRATION='registration',AMENDMENT='amendment',RESOURCE_EXTENSION='resource_extension',
                file_spec=lambda key:copy.deepcopy(direct[key]),
                prospective_authority_spec=lambda:copy.deepcopy(direct['revalidation_amendment']),
                check_sources=verify)
        checks=[]
        def same_entry(checks=checks,events=events):
            checks.append('same-ENTRY');events.append('check')
        assert boundary(make_api(state),same_entry,state['closed'],state['request'],role)==contract
        assert len(checks)==3+len(full) and observed[-1]==full and len(observed[-1])>9
        assert events==['check','check','source-begin',*(['check']*len(full)),'source-end','check']
        # A late same-ENTRY failure inside actual source IO propagates; no accepted result.
        calls=[];before=len(observed);events.clear()
        def late_check(calls=calls):
            calls.append('same-ENTRY')
            if len(calls)==4:raise TimeoutError('finite same-ENTRY source timeout')
        with pytest.raises(TimeoutError,match='same-ENTRY source timeout'):
            boundary(make_api(state),late_check,state['closed'],state['request'],role)
        assert len(calls)==4 and len(observed)==before and events==['source-begin']
        # Metadata maps remain equal while real non-own bytes drift: actual IO must reject.
        raw=(tmp_path/'non-own.py').read_bytes();before=len(observed)
        (tmp_path/'non-own.py').write_bytes(b'finite non-own drift\n')
        with pytest.raises(ValueError,match='current source closure'):
            boundary(make_api(state),lambda:None,state['closed'],state['request'],role)
        assert len(observed)==before
        (tmp_path/'non-own.py').write_bytes(raw)
        assert boundary(make_api(state),lambda:None,state['closed'],state['request'],role)==contract
        prefixes=[qualification_prefix]+([suite_prefix] if role=='fixture' else [])
        for prefix in prefixes:
            branch_namespace={'h':make_api(state),'check':lambda:None,'request':state['request'],
                              'validate_closed_full_map':boundary}
            exec(compile(h.ast.Module(body=prefix,type_ignores=[]),'actual-helper-suite-Q-SOURCE-prefix','exec'),branch_namespace)  # noqa: S102 -- actual production branch prefix, no publication/bootstrap.
        for mutation in ('changed','missing','extra','oldpass','report','authority','ownkey','contract_nonown','current_nonown'):
            state={'contract':copy.deepcopy(contract),'closed':copy.deepcopy(closed),'current':copy.deepcopy(full),'request':copy.deepcopy(request),'own':copy.deepcopy(own),'report':copy.deepcopy(report)}
            if mutation=='changed':state['closed']['source_sha256']['non-own.py']='e'*64
            elif mutation=='missing':state['closed']['source_sha256'].pop('non-own.py')
            elif mutation=='extra':state['closed']['source_sha256']['unexpected.py']='f'*64
            elif mutation=='oldpass':state['closed']['pass_id']='e360-generous-p11'
            elif mutation=='report':state['report']={'stale':'report'}
            elif mutation=='authority':state['contract']['revalidation_amendment']={}
            elif mutation=='ownkey':state['own']['own-drift.py']=state['own'].pop('own-0.py')
            elif mutation=='contract_nonown':state['contract']['source_sha256']['non-own.py']='e'*64
            else:state['current']['non-own.py']='e'*64
            before=len(observed)
            with pytest.raises(ValueError):boundary(make_api(state),lambda:None,state['closed'],state['request'],role)
            assert len(observed)==before
            for prefix in prefixes:
                branch_namespace={'h':make_api(state),'check':lambda:None,'request':state['request'],
                                  'validate_closed_full_map':boundary}
                with pytest.raises(ValueError):
                    exec(compile(h.ast.Module(body=prefix,type_ignores=[]),'actual-helper-rejected-SOURCE-prefix','exec'),branch_namespace)  # noqa: S102 -- actual production rejection seam.

def test_modified_pass_semantic_worker_failure_literal_and_once_charged(monkeypatch):
    worker={'schema':'flylab.m3.e360.actual-full-return.v1','mode':'generated','charged_once':True,
            'exit_code':0,'actual_outer_full_return_s':1.,'qualified':False}
    assert h.concrete_failed_predecessor(worker,30.)
    assert h.concrete_failed_predecessor({**worker,'actual_returned_exit_code':0},30.)
    for key,value in (('qualified',True),('qualified',0),('qualified',None),('charged_once',False),
                      ('charged_once',1),('mode','original'),('exit_code',False),('actual_outer_full_return_s',True)):
        assert not h.concrete_failed_predecessor({**worker,key:value},30.)
    # A qualified worker is never a fabricated modified-pass failure, even with contradictory code/cap.
    assert not h.concrete_failed_predecessor({**worker,'qualified':True,'exit_code':1,'actual_outer_full_return_s':31.},30.)
    admin={**worker,'schema':'flylab.m3.e360.administrative-full-return.v1','actual_returned_exit_code':0}
    assert not h.concrete_failed_predecessor(admin,30.)  # Arbitrary admin qualified=false does not authorize a retry.
    assert h.concrete_failed_predecessor({**admin,'internal_completion_valid':False},30.)
    assert h.concrete_failed_predecessor({**admin,'actual_returned_exit_code':1},30.)
    live=granted_live();real_read=h.read_spec;generated_bytes={}
    def save(name,value):
        raw=h.json_bytes(value);generated_bytes[name]=raw;return h.descriptor(name,raw)
    def read(spec,root=h.ROOT):
        if spec['path'] not in generated_bytes:return real_read(spec,root)
        raw=generated_bytes[spec['path']]
        assert h.sha(raw)==spec['sha256'] and len(raw)==spec['bytes']
        return raw
    monkeypatch.setattr(h,'read_spec',read)
    actual=save('semantic-actual.json',{'actual_outer_full_return_s':1.,'actual_returned_exit_code':0})
    failure=save('semantic-worker.json',{**worker,'cost_id':'e360-semantic:fixture','actual_tool_return':actual})
    debit={'cost_id':'e360-semantic:fixture','mode':'generated','actual_outer_full_return_s':1.,
           'actual_returned_exit_code':0,'descriptor':failure}
    debits=live['debits']+[debit]
    live['e360_passes']['e360-modified']={'mode':'generated','next_index':0,'source_sha256':h.own_sources(),
        'modified':True,'actual_failure':failure,'concrete_fix':[h.file_spec(h.OWNED[0])],'predecessor_failed_cap_s':30.}
    h.validate_live_calls(live,debits)
    with pytest.raises(ValueError,match='exactly once'):h.validate_live_calls(live,debits+[debit])
    with pytest.raises(ValueError,match='exactly once'):h.validate_live_calls(live,live['debits'])
    qualified=save('qualified-worker.json',{**worker,'qualified':True,'cost_id':'e360-semantic:fixture','actual_tool_return':actual})
    live['e360_passes']['e360-modified']['actual_failure']=qualified
    with pytest.raises(ValueError,match='concrete failed actual predecessor'):
        h.validate_live_calls(live,live['debits']+[{**debit,'descriptor':qualified}])


def test_ROOT_production_fallback_Q_and_modified_pass_pure_boundaries(tmp_path):
    """One bounded stdlib/PowerShell generated probe; no accounting entrypoint or live ledger."""
    import subprocess
    source=h.safe(h.BASE+'/parent-e360-root-account.ps1').read_text('utf8')
    # These actual production functions are used by fallback and original callbegin;
    # run them directly against finite JSON, rather than duplicating their predicates.
    functions=source[source.index('function Equal-Typed('):source.index('function Sum-Actual(')]
    activation=source[source.index('$ruffRecoveryActivation=@{'):source.index("if($Action -cnotin @('recover-ruff','activate-ruff'))")]
    internal_completion=source[source.index('        # Metadata preparation never consumes an original experiment attempt.'):
                               source.index('        $cost.internal_completion_valid=$internalGood')]
    desc={'path':'generated.json','bytes':1,'sha256':'a'*64}
    current={'source_sha256':{'tools/generated.py':'b'*64},'registration':desc,'amendment':desc,'revalidation_amendment':desc}
    outer={'schema':'flylab.m3.e360.outer-provisional.v1','contract':desc,'resource_extension':desc,'revalidation_amendment':desc,
           **current,'all_owned_ended':True,'reservation_released':True,'natural_closed':True,
           'unresolved_matching_identities':{},'unknown_owned_identities':{},'missing_owned_creation_times':[],
           'known_identity_cleanup':[],'observed_pid_create_times':{'123':10.},
           'Popen_attempted':True,'Popen_returned':True,'Popen_never_attempted':False}
    call={'state':'charged','call_role':'qualification','mode':'generated','cost_id':'e360-Q',
          'pass_id':'e360-pass','request':desc,'revalidation_amendment':desc}
    q={'qualification_call_id':'e360-Q','source_sha256':current['source_sha256'],'resource_extension':desc,'revalidation_amendment':desc}
    cost={'schema':'flylab.m3.e360.administrative-full-return.v1','mode':'generated','charged_once':True,
          'cost_id':'e360-Q','call_id':'e360-Q','pass_id':'e360-pass','call_role':'qualification','request':desc,
          'actual_returned_exit_code':0,'actual_outer_full_return_s':1.,'internal_completion_valid':True,
          'expected_output':desc,'source_sha256':q['source_sha256'],'resource_extension':desc,'revalidation_amendment':desc}
    completion={'schema':'flylab.m3.e360.helper-internal-completion.v1','completed':True,'call_id':'e360-Q',
                'pass_id':'e360-pass','call_role':'qualification','request':desc,'resource_extension':desc,'revalidation_amendment':desc,
                'source_sha256':q['source_sha256'],'expected_output':desc,'entry_perf_counter':10.,'internal_full_return_s':1.}
    payload={'outer':outer,'current':current,'descriptor':desc,'q':q,'call':call,'cost':cost,'completion':completion,
             'revalidation':h.ruff_recovery_amendment(),
             'activation_ledger':h.strict_json(h.read_spec(h.ruff_recovery_amendment()['inherited63']))['ledger'],
             'ruff_failure':h.charged_cost(h.ruff_recovery_amendment()['actual_predecessor_failure'],'generated'),
             'manual_sources':[h.file_spec(name) for name in (*h.OWNED,h.BASE+'/parent-e360-root-account.ps1',h.BASE+'/parent-e360-root-metadata-helper.py')],
             'manual_source_sha256':h.own_sources(),
             'source_authority':h.source_recovery_amendment(),
             'historical_p5':h.strict_json(h.read_spec(h.ruff_recovery_amendment()['inherited63']))['ledger']['e360_passes']['e360-generous-p5'],
             'historical_scanner_failure':h.charged_cost(h.source_recovery_amendment()['actual_predecessor_failure'],'generated'),
             'request':{'source_sha256':q['source_sha256'],'revalidation_amendment':desc},'policy':{'outer_s':20,'internal_s':18},
             'qualification_request':{'schema':'flylab.m3.e360.helper-request.v1','call_id':'e360-Q','pass_id':'e360-pass',
                                      'call_role':'qualification','action':'qualification','role':'suite','mode':'generated',
                                      'expected_output':desc['path'],'source_sha256':q['source_sha256'],'resource_extension':desc,'revalidation_amendment':desc}}
    data=tmp_path/'pure-input.json';data.write_bytes(h.json_bytes(payload))
    script=tmp_path/'pure-boundary.ps1'
    script.write_text("param([string]$InputPath)\n$ErrorActionPreference='Stop'\n"+functions+r'''
$p=[IO.File]::ReadAllText($InputPath)|ConvertFrom-Json -AsHashtable
function Clone($value){return ($value|ConvertTo-Json -Depth 100 -Compress|ConvertFrom-Json -AsHashtable)}
function Check([bool]$condition,[string]$message){if(-not $condition){throw $message}}
$revalidation=$p.revalidation;$revalidationSpec=$p.descriptor;$extensionSpec=$p.descriptor
'''+activation+r'''
$roundTrip=Clone $ruffRecoveryActivation
Check (Equal-Typed $roundTrip $ruffRecoveryActivation) 'production current activation rejected after durable JSON round trip'
Check ($roundTrip.schema -ceq 'flylab.m3.e360.ruff-recovery-activation.v1' -and $roundTrip.manual_generated_pass_id -ceq 'e360-generous-p6') 'wrong current activation constructor'
$bad=Clone $roundTrip;$bad.new_generated_grant_s=$false
Check (-not (Equal-Typed $bad $ruffRecoveryActivation)) 'boolean grant admitted by current activation comparison'
Check (Ruff-Recovery-Activation-Allowed $p.activation_ledger $true) 'exact63 initial activation rejected'
$bad=Clone $p.activation_ledger;$bad.e360_ruff_recovery_activation=$roundTrip
Check (-not (Ruff-Recovery-Activation-Allowed $bad $true)) 'repeated activation admitted'
$bad=Clone $p.activation_ledger;$bad.original_jobs_running=$false
Check (-not (Ruff-Recovery-Activation-Allowed $bad $true)) 'boolean zero ownership admitted'
$bad=Clone $p.activation_ledger;$bad.e360_pending_actuals=@{pending=@{}}
Check (-not (Ruff-Recovery-Activation-Allowed $bad $true)) 'pending activation admitted'
Check (-not (Ruff-Recovery-Activation-Allowed $p.activation_ledger $false)) 'present lock activation admitted'
$predecessor=Clone @{failed_cap_s=$revalidation.predecessor_failed_cap_s}
Check (Equal-Typed $predecessor.failed_cap_s $revalidation.predecessor_failed_cap_s) 'authority cap JSON round trip mismatch'
function Probe-Internal-Completion([string]$role,$record){
    $extension=@{helper_limits=@{qualification=$p.policy}}
    $req=@{call_role=$role;call_id=$p.completion.call_id;pass_id=$p.completion.pass_id;source_sha256=$p.q.source_sha256}
    $admitted=@{spec=$p.descriptor};$outputSpec=$p.descriptor;$callPolicy=$p.policy
    $completionSpec=if($null -eq $record){$null}else{$p.descriptor}
    function Checked-Read($spec){return $record}
'''+internal_completion+r'''
    return $internalGood
}
Check (Probe-Internal-Completion 'ruff' $null) 'Ruff incorrectly requires Python helper completion'
Check (Probe-Internal-Completion 'qualification' $p.completion) 'timely helper completion rejected'
Check (-not (Probe-Internal-Completion 'qualification' $null)) 'missing helper completion admitted'
$badCompletion=Clone $p.completion;$badCompletion.internal_full_return_s=$true
Check (-not (Probe-Internal-Completion 'qualification' $badCompletion)) 'boolean helper clock admitted'
$good=Fallback-Closure-Proof $p.outer $p.current $p.descriptor $p.descriptor $true
Check $good.closed 'genuine closed proof rejected'
foreach($field in @('unknown_owned_identities','unresolved_matching_identities','missing_owned_creation_times','known_identity_cleanup','source_sha256','contract','resource_extension','revalidation_amendment','all_owned_ended','reservation_released','observed_pid_create_times','Popen_returned')){
    $bad=Clone $p.outer;$null=$bad.Remove($field)
    $proof=Fallback-Closure-Proof $bad $p.current $p.descriptor $p.descriptor $true
    Check (-not $proof.closed -and $proof.unknown_owned_identities.ContainsKey('fallback-closure-proof')) ('omission released: '+$field)
}
foreach($value in @($null,0,$false,'malformed')){
    $bad=Clone $p.outer;$bad.unknown_owned_identities=$value
    $proof=Fallback-Closure-Proof $bad $p.current $p.descriptor $p.descriptor $true
    Check (-not $proof.closed) 'null/malformed ownership released'
}
$bad=Clone $p.outer;$bad.unknown_owned_identities=@{known_unknown='must retain'};$bad.missing_owned_creation_times=@(42)
$proof=Fallback-Closure-Proof $bad $p.current $p.descriptor $p.descriptor $true
Check (-not $proof.closed -and $proof.unknown_owned_identities.known_unknown -ceq 'must retain' -and $proof.missing_owned_creation_times[0] -eq 42) 'supplied ownership lost'
$bad=Clone $p.outer;$bad.contract.sha256='c'*64
Check (-not (Fallback-Closure-Proof $bad $p.current $p.descriptor $p.descriptor $true).closed) 'contract alias released'
Check (-not (Fallback-Closure-Proof $p.outer $p.current $p.descriptor $p.descriptor $false).closed) 'present original lock released'
Require-Q-Helper-Proof $p.q $p.descriptor $p.call $p.cost $p.completion $p.request $p.policy $p.descriptor $p.qualification_request
foreach($field in @('completed','schema','call_id','pass_id','call_role','request','expected_output','source_sha256','resource_extension','revalidation_amendment','entry_perf_counter','internal_full_return_s')){
    $bad=Clone $p.completion
    switch($field){
        'completed' {$bad[$field]=$false}
        'schema' {$bad[$field]='wrong'}
        'entry_perf_counter' {$bad[$field]=$true}
        'internal_full_return_s' {$bad[$field]=[double]19}
        default {$bad[$field]='mismatched'}
    }
    $rejected=$false
    try{Require-Q-Helper-Proof $p.q $p.descriptor $p.call $p.cost $bad $p.request $p.policy $p.descriptor $p.qualification_request}catch{$rejected=$true}
    Check $rejected ('bad Q internal proof admitted: '+$field)
}
$bad=Clone $p.cost;$bad.internal_completion_valid=$false;$rejected=$false
try{Require-Q-Helper-Proof $p.q $p.descriptor $p.call $bad $p.completion $p.request $p.policy $p.descriptor $p.qualification_request}catch{$rejected=$true}
Check $rejected 'late Q external exit0 admitted original helper'
$worker=Clone $p.cost;$worker.schema='flylab.m3.e360.actual-full-return.v1';$worker.qualified=$false
Check (Concrete-Failed-Predecessor $worker 30) 'semantic exit0 unqualified worker not recognized'
$worker.qualified=$true;$worker.actual_returned_exit_code=1
Check (-not (Concrete-Failed-Predecessor $worker 30)) 'qualified worker used as fake failure'
$worker.qualified=0
Check (-not (Concrete-Failed-Predecessor $worker 30)) 'nonbool worker qualification accepted'
$admin=Clone $p.cost;$admin.internal_completion_valid=$false
Check (Admin-Recovery-Predecessor $admin 3) 'actual exit0/internalfalse administrative predecessor rejected'
$admin.internal_completion_valid=$true
Check (-not (Admin-Recovery-Predecessor $admin 3)) 'successful admin fabricated as predecessor'
$admin.internal_completion_valid=0
Check (-not (Admin-Recovery-Predecessor $admin 3)) 'nonbool admin failure accepted'
Check (Ruff-Recovery-Predecessor $p.ruff_failure $revalidation.predecessor_failed_cap_s) 'actual p5 Ruff failure rejected'
$manual=@{mode='generated';modified=$true;actual_failure=$revalidation.actual_predecessor_failure;
    predecessor_failed_cap_s=$revalidation.predecessor_failed_cap_s;
    concrete_fix=@($revalidation.concrete_correction_snapshot)+@($p.manual_sources);source_sha256=$p.manual_source_sha256}
$manual=Clone $manual
Check (Ruff-Recovery-Manual-Pass 'e360-generous-p6' $manual $revalidation $p.activation_ledger.debits $p.ruff_failure) 'actual once-debited manual p6 rejected'
Check (Source-Recovery-Historical-Pass $p.historical_p5 $p.historical_p5 $p.source_authority $p.activation_ledger.debits $p.historical_scanner_failure) 'as-executed historical p5 proof rejected after current source changes'
$bad=Clone $p.historical_p5;$bad.source_sha256=$p.manual_source_sha256
Check (-not (Source-Recovery-Historical-Pass $bad $p.historical_p5 $p.source_authority $p.activation_ledger.debits $p.historical_scanner_failure)) 'current source map substituted into historical p5'
$bad=Clone $p.historical_p5;$bad.concrete_fix=@($revalidation.concrete_correction_snapshot)+@($p.manual_sources)
Check (-not (Source-Recovery-Historical-Pass $bad $p.historical_p5 $p.source_authority $p.activation_ledger.debits $p.historical_scanner_failure)) 'current evidence substituted into historical p5'
Check (Equal-Typed $manual.predecessor_failed_cap_s $revalidation.predecessor_failed_cap_s) 'parsed authority cap changed after manual JSON roundtrip'
$bad=Clone $manual;$bad.concrete_fix=@($bad.concrete_fix|Where-Object{$_.path -cne $revalidation.concrete_correction_snapshot.path})
Check (-not (Ruff-Recovery-Manual-Pass 'e360-generous-p6' $bad $revalidation $p.activation_ledger.debits $p.ruff_failure)) 'missing C6 evidence admitted'
foreach($spec in $p.manual_sources){
    $bad=Clone $manual;$bad.concrete_fix=@($bad.concrete_fix|Where-Object{$_.path -cne $spec.path})
    Check (-not (Ruff-Recovery-Manual-Pass 'e360-generous-p6' $bad $revalidation $p.activation_ledger.debits $p.ruff_failure)) ('missing current source admitted: '+$spec.path)
    $bad=Clone $manual;$bad.source_sha256[$spec.path]='0'*64
    Check (-not (Ruff-Recovery-Manual-Pass 'e360-generous-p6' $bad $revalidation $p.activation_ledger.debits $p.ruff_failure)) ('wrong source SHA admitted: '+$spec.path)
    $bad=Clone $manual;$bad.source_sha256[$spec.path]=$true
    Check (-not (Ruff-Recovery-Manual-Pass 'e360-generous-p6' $bad $revalidation $p.activation_ledger.debits $p.ruff_failure)) ('boolean source SHA admitted: '+$spec.path)
    foreach($field in @('bytes','sha256','path')){
        $bad=Clone $manual;$entry=@($bad.concrete_fix|Where-Object{$_.path -ceq $spec.path})[0];$entry[$field]=$true
        Check (-not (Ruff-Recovery-Manual-Pass 'e360-generous-p6' $bad $revalidation $p.activation_ledger.debits $p.ruff_failure)) ('non-typed current source admitted: '+$field)
    }
}
$bad=Clone $manual;$bad.source_sha256=$false
Check (-not (Ruff-Recovery-Manual-Pass 'e360-generous-p6' $bad $revalidation $p.activation_ledger.debits $p.ruff_failure)) 'malformed current source map admitted'
Check (-not (Ruff-Recovery-Manual-Pass 'e360-generous-p7' $manual $revalidation $p.activation_ledger.debits $p.ruff_failure)) 'p7 admitted'
$bad=Clone $manual;$bad.modified=1
Check (-not (Ruff-Recovery-Manual-Pass 'e360-generous-p6' $bad $revalidation $p.activation_ledger.debits $p.ruff_failure)) 'nonbool modified admitted'
foreach($cap in @($true,20)){
    $bad=Clone $manual;$bad.predecessor_failed_cap_s=$cap
    Check (-not (Ruff-Recovery-Manual-Pass 'e360-generous-p6' $bad $revalidation $p.activation_ledger.debits $p.ruff_failure)) 'wrong/bool cap admitted'
}
$bad=Clone $manual;$bad.actual_failure=$p.source_authority.actual_predecessor_failure
Check (-not (Ruff-Recovery-Manual-Pass 'e360-generous-p6' $bad $revalidation $p.activation_ledger.debits $p.historical_scanner_failure)) 'historical scanner used as Ruff predecessor'
$predecessor=@($p.activation_ledger.debits|Where-Object{Equal-Typed $_.descriptor $manual.actual_failure})
Check ($predecessor.Count -eq 1) 'predecessor fixture not once debited'
Check (-not (Ruff-Recovery-Manual-Pass 'e360-generous-p6' $manual $revalidation (@($p.activation_ledger.debits)+@($predecessor[0])) $p.ruff_failure)) 'replayed predecessor admitted'
$bad=Clone $p.ruff_failure;$bad.actual_returned_exit_code=$true
Check (-not (Ruff-Recovery-Predecessor $bad $revalidation.predecessor_failed_cap_s)) 'boolean Ruff exit admitted'
$bad=Clone $p.ruff_failure;$bad.actual_returned_exit_code=0
Check (-not (Ruff-Recovery-Predecessor $bad $revalidation.predecessor_failed_cap_s)) 'old exit0/internalfalse Ruff predicate accepted for current Ruff'
$bad=Clone $p.ruff_failure;$bad.internal_completion_valid=$false
Check (-not (Ruff-Recovery-Predecessor $bad $revalidation.predecessor_failed_cap_s)) 'historical internalfalse classification substituted into actual internaltrue Ruff'
@{schema='flylab.m3.e360.generated-pure-boundaries.v1';resource_extension=$p.descriptor;revalidation_amendment=$p.descriptor;active_activation_JSON_roundtrip=$true;administrative_exit0_failure=$true;fallback_closed_good=$true;malformed_unknown_retained=$true;Q_before_original_internal_proof=$true;semantic_modified_failure=$true;original_calls=0;live_ledger_writes=0}|ConvertTo-Json -Compress
''',encoding='utf8')
    result=subprocess.run(['pwsh','-NoProfile','-NonInteractive','-File',str(script),str(data)],
                          capture_output=True,text=True,timeout=8,check=False)
    assert result.returncode==0,result.stderr
    proof=h.strict_json(result.stdout.encode('utf8'))
    assert set(proof)=={'schema','resource_extension','revalidation_amendment','fallback_closed_good',
                        'malformed_unknown_retained','Q_before_original_internal_proof','semantic_modified_failure',
                        'active_activation_JSON_roundtrip','administrative_exit0_failure','original_calls','live_ledger_writes'}
    assert proof['schema']=='flylab.m3.e360.generated-pure-boundaries.v1'
    assert proof['resource_extension']==proof['revalidation_amendment']==desc
    assert proof['fallback_closed_good'] and proof['malformed_unknown_retained']
    assert proof['Q_before_original_internal_proof'] and proof['semantic_modified_failure']
    assert proof['active_activation_JSON_roundtrip'] and proof['administrative_exit0_failure']
    assert proof['original_calls']==proof['live_ledger_writes']==0


def _p30_copy_history_table(current,root):
    # Exact same table bytes are copied, but the genuine scope validates this root's own file.
    spec=current.HISTORY_CATALOGUE_SPEC;raw=(current.ROOT/spec['path']).read_bytes()
    assert len(raw)==spec['bytes'] and current.sha(raw)==spec['sha256']
    path=root/spec['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    assert path.read_bytes()==raw

def _p30_history_leaf(current,root,spec):
    raw=(current.ROOT/spec['path']).read_bytes()
    assert len(raw)==spec['bytes'] and current.sha(raw)==spec['sha256']
    raw.decode('utf8',errors='strict')
    path=root/spec['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    return path

def _p30_owned_scope_cases(current,tmp_path):
    # Focused metadata-only mutations use the same production registration/fences; no history replay.
    root=tmp_path/'p30-owned';root.mkdir();_p30_copy_history_table(current,root)
    positive=current.P27_ACTUAL_RUFF_RECORD_SPECS['stdout.txt']
    empty=current.P27_ACTUAL_RUFF_RECORD_SPECS['stderr.txt']
    leaf=_p30_history_leaf(current,root,positive);_p30_history_leaf(current,root,empty)
    with current.fixed_authority_scope(root=root,check=lambda:None) as boundary:
        current.historical_p12_stream_bytes(positive,root)
        current.historical_p12_stream_bytes(empty,root)
        current.read_spec(positive,root,owner='historical')
        assert boundary.depth==0 and len(boundary.history_descriptors)==2 and len(boundary.descriptors)==1
        assert not set(boundary.descriptors)&set(boundary.history_descriptors)
        boundary.values['scratch']={'value':True};boundary.clear_parsed()
        assert not boundary.values and len(boundary.history_descriptors)==2
        with current.descriptors_only_execution():
            assert current._AUTHORITY_FENCE_STATE is boundary and len(boundary.history_descriptors)==2
            current.authority_fence('prepublication')
        boundary.fence('post_pre_array_before_first_numeric_native_or_array')
        boundary.fence('pre_Popen')
    assert not boundary.descriptors and not boundary.history_descriptors
    unknown={'path':'unknown-history.txt','bytes':1,'sha256':current.sha(b'x')};(root/unknown['path']).write_bytes(b'x')
    for route in ('private','public','read_spec'):
        with current.fixed_authority_scope(root=root,check=lambda:None) as boundary:
            with pytest.raises(ValueError,match='unknown historical'), current.dependency_ownership(root,'historical'):
                if route=='private':boundary._consumed(unknown,root)
                elif route=='public':boundary.depth=1;boundary.consumed(unknown,root)
                else:current.read_spec(unknown,root,owner='historical')
            assert boundary.failed and len(boundary.descriptors)<512 and not boundary.history_descriptors
    unknown_empty={'path':'p30-synthetic-completed-stderr.txt','bytes':0,
        'sha256':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}
    (root/unknown_empty['path']).write_bytes(b'')
    original_completed=current._COMPLETED_STREAM_PINS
    original_catalogue=copy.deepcopy(current.HISTORY_CATALOGUE_SPEC)
    with pytest.MonkeyPatch.context() as synthetic_admission:
        synthetic_admission.setattr(current,'_COMPLETED_STREAM_PINS',(*original_completed,unknown_empty))
        with current.fixed_authority_scope(root=root,check=lambda:None) as boundary:
            with pytest.raises(ValueError,match='unknown historical physical dependency'):
                current.historical_p12_stream_bytes(unknown_empty,root)
            assert boundary.depth==0 and boundary.failed and not boundary.values
            assert len(boundary.descriptors)<512 and not boundary.history_descriptors
            assert current.HISTORY_CATALOGUE_SPEC==original_catalogue
    assert current._COMPLETED_STREAM_PINS is original_completed
    for conflict in ('identity','owner'):
        with current.fixed_authority_scope(root=root,check=lambda:None) as boundary:
            current.historical_p12_stream_bytes(positive,root)
            with pytest.raises(ValueError):
                if conflict=='identity':boundary._consumed({**positive,'sha256':'0'*64},root,owner='historical')
                else:boundary._consumed(positive,root,owner='dynamic')
            assert boundary.failed and len(boundary.history_descriptors)==1
    for point in ('prepublication','pre_Popen','post_pre_array_before_first_numeric_native_or_array','failure_prepublication'):
        leaf.write_bytes(b'All checks passed!\n')
        with current.fixed_authority_scope(root=root,check=lambda:None) as boundary:
            current.historical_p12_stream_bytes(positive,root)
            leaf.write_bytes(b'late')
            with pytest.raises(ValueError):boundary.fence(point)
            assert boundary.failed and boundary.history_descriptors
    with current.fixed_authority_scope(root=root,check=lambda:None) as boundary:
        for index in range(512):
            name='generic-'+str(index)+'.txt';(root/name).write_bytes(b'g')
            boundary._consumed({'path':name,'bytes':1,'sha256':current.sha(b'g')},root,owner='dynamic')
        assert len(boundary.descriptors)==512 and not boundary.history_descriptors
        (root/'generic-overflow.txt').write_bytes(b'g')
        with pytest.raises(ValueError,match='descriptor bound'):
            boundary._consumed({'path':'generic-overflow.txt','bytes':1,'sha256':current.sha(b'g')},root,owner='dynamic')
        assert boundary.failed
def finite_scope_authority(tmp_path,monkeypatch):
    isolate_compact_boundary(monkeypatch)
    raw=h.json_bytes({'schema':'finite.generated.fixed-literal','items':[1]})
    path=tmp_path/'literal.json';path.write_bytes(raw)
    dependency=tmp_path/'dependency.py';dependency.write_bytes(b'VALUE=1\n')
    calls=[]
    def validate(root):
        name=h.FIXED_READERS['registration'][0]
        value=h.strict_json(h.literal_bytes(name,root))
        h.typed_equal(value['schema'],'finite.generated.fixed-literal','finite typed authority')
        h.read_spec(h.file_spec('dependency.py',root),root)
        calls.append(name)
        return value
    monkeypatch.setitem(h.FIXED_READERS,'registration',('literal.json',len(raw),h.sha(raw)))
    monkeypatch.setattr(h,'_registration',validate)
    return path,raw,calls


def test_fixed_scope_hits_deepcopy_nested_root_descriptor_and_isolation(tmp_path,monkeypatch):
    isolate_compact_boundary(monkeypatch)
    _p30_owned_scope_cases(h,tmp_path)
    path,raw,calls=finite_scope_authority(tmp_path,monkeypatch)
    ticks=[]
    with h.fixed_authority_scope(root=tmp_path,check=lambda:ticks.append(1)) as scope:
        value=h.registration(tmp_path);value['items'].append(2)
        assert h.registration(tmp_path)['items']==[1] and calls==['literal.json']
        with h.fixed_authority_scope(root=tmp_path,check=lambda:ticks.append(2)) as nested:
            assert nested is scope and h.registration(tmp_path)['items']==[1]
        assert len(ticks)>=5 and len(scope.descriptors)==2
        with pytest.raises(ValueError,match='root'):h.registration(tmp_path/'other')
        assert scope.failed and not scope.values and scope.descriptors
        with pytest.raises(ValueError,match='nested fixed scope root'), h.fixed_authority_scope(root=tmp_path/'other',check=lambda:None):pass
        with pytest.raises(ValueError,match='cannot be reused'):h.registration(tmp_path)
    with h.fixed_authority_scope(root=tmp_path,check=lambda:None) as scope:
        h.registration(tmp_path)
        monkeypatch.setitem(h.FIXED_READERS,'registration',('literal.json',len(raw), '0'*64))
        with pytest.raises(ValueError,match='descriptor'):h.registration(tmp_path)
        assert scope.failed and not scope.values and scope.descriptors
        with pytest.raises(ValueError,match='cannot be reused'):h.registration(tmp_path)
    assert not scope.values and not scope.descriptors and h._AUTHORITY_SCOPE is None
    monkeypatch.setitem(h.FIXED_READERS,'registration',('literal.json',len(raw),h.sha(raw)))
    with h.fixed_authority_scope(root=tmp_path,check=lambda:None):h.registration(tmp_path)
    assert len(calls)==4 and path.read_bytes()==raw


def test_fixed_scope_hit_deadline_failure_nonreuse_and_first_error_cleanup(tmp_path,monkeypatch):
    finite_scope_authority(tmp_path,monkeypatch)
    late={'value':False}
    def check():
        if late['value']:raise TimeoutError('actual ENTRY expired')
    with pytest.raises(TimeoutError,match='actual ENTRY'), h.fixed_authority_scope(root=tmp_path,check=check) as scope:
        h.registration(tmp_path)
        late['value']=True
        h.registration(tmp_path)
    assert not scope.values and not scope.descriptors and h._AUTHORITY_SCOPE is None
    late['value']=False
    def failed(root):
        h.literal_bytes('literal.json',root)
        raise ValueError('earliest invalid authority')
    monkeypatch.setattr(h,'_registration',failed)
    with h.fixed_authority_scope(root=tmp_path,check=check) as scope:
        with pytest.raises(ValueError,match='earliest'):h.registration(tmp_path)
        with pytest.raises(ValueError,match='cannot be reused'):h.registration(tmp_path)
        assert not scope.values and scope.descriptors
    assert h._AUTHORITY_SCOPE is None


def test_fixed_scope_entry_literal_aggregate_and_monotone_descriptor_bounds(tmp_path,monkeypatch):
    _,raw,_=finite_scope_authority(tmp_path,monkeypatch)
    with h.fixed_authority_scope(root=tmp_path,check=lambda:None) as scope:
        for index in range(16):
            name=f'literal{index}.json';(tmp_path/name).write_bytes(raw)
            monkeypatch.setitem(h.FIXED_READERS,'registration',(name,len(raw),h.sha(raw)))
            h.registration(tmp_path)
        monkeypatch.setitem(h.FIXED_READERS,'registration',('seventeenth.json',len(raw),h.sha(raw)))
        with pytest.raises(ValueError,match='entry bound'):h.registration(tmp_path)
        assert not scope.values and scope.failed and len(scope.descriptors)==17
        with pytest.raises(ValueError,match='cannot be reused'):h.registration(tmp_path)
    with h.fixed_authority_scope(root=tmp_path,check=lambda:None):
        monkeypatch.setitem(h.FIXED_READERS,'registration',('oversize.json',262145,h.sha(raw)))
        with pytest.raises(ValueError,match='literal byte'):h.registration(tmp_path)
    padded=raw+b' '*(262144-len(raw))
    with h.fixed_authority_scope(root=tmp_path,check=lambda:None) as scope:
        for index in range(8):
            name=f'large{index}.json';(tmp_path/name).write_bytes(padded)
            monkeypatch.setitem(h.FIXED_READERS,'registration',(name,len(padded),h.sha(padded)))
            h.registration(tmp_path)
        assert scope.literal_bytes==2097152
        monkeypatch.setitem(h.FIXED_READERS,'registration',('ninth.json',len(raw),h.sha(raw)))
        with pytest.raises(ValueError,match='literal byte'):h.registration(tmp_path)
        assert scope.failed and not scope.values and scope.descriptors
    with h.fixed_authority_scope(root=tmp_path,check=lambda:None) as scope:
        scope.depth=1
        try:
            for index in range(512):
                name=f'text{index}.json';(tmp_path/name).write_bytes(b'{}')
                h.file_spec(name,tmp_path)
            assert len(scope.descriptors)==512
            (tmp_path/'overflow.json').write_bytes(b'{}')
            with pytest.raises(ValueError,match='descriptor bound'):h.file_spec('overflow.json',tmp_path)
            assert len(scope.descriptors)==512 and scope.failed and not scope.values
        finally:scope.depth=0


def test_fixed_scope_publication_pre_post_fences_bytes_path_and_late_error(tmp_path,monkeypatch):
    path,raw,_=finite_scope_authority(tmp_path,monkeypatch)
    with h.fixed_authority_scope(root=tmp_path,check=lambda:None) as scope:
        h.registration(tmp_path);h.registration(tmp_path)
        path.write_bytes(raw+b' ')
        with pytest.raises(ValueError,match='fence'):h.durable(tmp_path/'never.json',b'{}',cap=16)
        assert not (tmp_path/'never.json').exists()
        assert scope.failed and scope.descriptors and not scope.values
        with pytest.raises(ValueError,match='cannot be reused'):h.registration(tmp_path)
        path.write_bytes(raw)
        with pytest.raises(ValueError,match='failed fixed authority fence'):
            h.durable(tmp_path/'blocked-after-restore.json',b'{}',cap=16)
        assert not (tmp_path/'blocked-after-restore.json').exists()
        with h.descriptors_only_execution(), pytest.raises(ValueError,match='failed fixed authority fence'):
            h.authority_fence('post_pre_array_before_first_numeric_native_or_array')
    path.write_bytes(raw)
    original=h.lib.durable
    def publish_then_drift(*args,**kwargs):
        result=original(*args,**kwargs)
        path.write_bytes(raw+b' ')
        return result
    monkeypatch.setattr(h.lib,'durable',publish_then_drift)
    with h.fixed_authority_scope(root=tmp_path,check=lambda:None):
        h.registration(tmp_path)
        with pytest.raises(ValueError,match='fence'):h.durable(tmp_path/'closed-but-unqualified.json',b'{}',cap=16)
        assert (tmp_path/'closed-but-unqualified.json').read_bytes()==b'{}'
    path.write_bytes(raw)
    real_safe=h.safe;alias=tmp_path/'alias.json';alias.write_bytes(raw)
    with h.fixed_authority_scope(root=tmp_path,check=lambda:None):
        h.registration(tmp_path)
        monkeypatch.setattr(h,'safe',lambda name,root=h.ROOT:alias if name=='literal.json' else real_safe(name,root))
        with pytest.raises(ValueError,match='canonical path'):h.authority_fence('prepublication')
    monkeypatch.setattr(h,'safe',real_safe)
    with h.fixed_authority_scope(root=tmp_path,check=lambda:None) as scope:
        h.registration(tmp_path)
        def reparse(name,root=h.ROOT):
            if name=='literal.json':raise ValueError('finite no-reparse stand-in')
            return real_safe(name,root)
        monkeypatch.setattr(h,'safe',reparse)
        with pytest.raises(ValueError,match='no-reparse'):h.authority_fence('prepublication')
        assert scope.failed and scope.descriptors and not scope.values
        with pytest.raises(ValueError,match='cannot be reused'):h.registration(tmp_path)


def test_fixed_scope_actual_pre_popen_fence_and_fresh_dynamic_admission(tmp_path,monkeypatch):
    import time
    path,raw,_=finite_scope_authority(tmp_path,monkeypatch)
    paths=h.names('e360-scope-boundary','fixture')
    desc={'path':'finite-authority.json','bytes':1,'sha256':'a'*64}
    c={'mode':'generated','role':'fixture','attempt_id':'e360-scope-boundary','attempt':1,'source_sha256':{},
       'paths':paths,'limits':{'outer_s':30,'child_s':24,'internal_s':20,'science_s':16,'output_cap_bytes':16<<20},
       'original_calls':0,'cumulative_before_s':274.1960156,'cumulative_limit_s':600,'reserved_s':30,
       'pending_reservations':{},'stage_used_s':194.006,'stage_limit_s':600,'max_tree_rss_bytes':2<<30,
       'retained_reserve_bytes':1<<20,'cpu_threads':1,'registration':desc,'amendment':desc,
       'resource_extension':desc,'revalidation_amendment':desc,'closure_report':desc,
       'future_administration_reserved_s':70,'local_used_s':163.239,'local_limit_s':489}
    target=h.safe(paths['contract'],tmp_path);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(h.json_bytes(c))
    def validate(name,*,root,**kwargs):
        h.registration(root);h.registration(root)  # Actual fixed-reader hit.
        current=h.file_spec(name,root)
        h.typed_equal(h.strict_json(h.read_spec(current,root)),c,'finite current contract')
        return copy.deepcopy(c),current,{}
    monkeypatch.setattr(h,'validate_contract',validate)
    monkeypatch.setattr(h,'failed_authority_view',lambda *args,**kwargs:copy.deepcopy(c))
    monkeypatch.setattr(h,'sys',SimpleNamespace(modules={}))
    monkeypatch.setitem(sys.modules,'psutil',SimpleNamespace(Process=lambda pid:SimpleNamespace(pid=pid,create_time=lambda:456.)))
    invoked=[]
    monkeypatch.setattr(h.subprocess,'Popen',lambda *args,**kwargs:invoked.append('Popen'))
    real=h._pre_spawn
    def stored_then_drift(*args,**kwargs):
        admitted=real(*args,**kwargs)
        receipt=h.strict_json(h.safe(paths['pre_spawn'],tmp_path).read_bytes())
        assert receipt['original_array_reads']==receipt['scientific_native_imports']==0
        path.write_bytes(raw.replace(b'1',b'2',1))
        return admitted
    monkeypatch.setattr(h,'_pre_spawn',stored_then_drift)
    entry=time.perf_counter()
    assert h.supervise(paths['contract'],entry=entry,root=tmp_path)==1
    outer=h.strict_json(h.safe(paths['outer'],tmp_path).read_bytes())
    assert invoked==[] and outer['Popen_never_attempted'] and not outer['Popen_attempted']
    assert outer['pre_spawn'] and outer['authority_admitted'] and outer['first_error']['stage']=='outer'
    assert 'fence' in outer['first_error']['message'] and outer['secondary_errors']
    assert outer['all_owned_ended'] and outer['reservation_released'] and outer['observed_pid_create_times']=={}
    assert outer['unknown_owned_identities']=={} and outer['actual_full_return_pending']
    assert outer['failure_publication']['entry_perf_counter']==entry and not outer['natural_closed']


    # New contract/receipt descriptors are enrolled even at depth0.
    for mutation in ('contract','receipt','expired'):
        with monkeypatch.context() as patch:
            root=tmp_path/('compact-'+mutation);root.mkdir()
            isolate_compact_boundary(patch)
            current,pin,_=finite_compact_contract(root)
            original=copy.deepcopy(current);entry=h.time.perf_counter()
            def fresh(name,*,root,original=original,**kwargs):
                raw=h.bounded_json_name(name,root,cap=524288)
                actual=h.strict_json(raw);h.typed_equal(actual,original,'finite fresh dynamic reproduction')
                return actual,h.descriptor(name,raw),{}
            patch.setattr(h,'validate_contract',fresh)
            check=h.operation_clock('contract.json',entry,root=root)
            with h.fixed_authority_scope(root=root,check=check) as boundary:
                admitted,pin,_parent,_=h.pre_spawn('contract.json',entry,root=root)
                assert boundary.depth==0 and len(boundary.descriptors)==2 and not boundary.values
                target=root/('contract.json' if mutation=='contract' else 'receipt.json')
                if mutation=='expired':check.timer.clock=lambda entry=entry:entry+31.
                else:
                    raw=target.read_bytes();changed=raw.replace(b'e360-finite',b'e360-forged')
                    assert len(changed)==len(raw) and changed!=raw;target.write_bytes(changed)
                with pytest.raises((ValueError,TimeoutError)):h.pre_popen_admission(admitted,'contract.json',pin,root=root)
                assert boundary.failed and boundary.descriptors and not boundary.values
                with pytest.raises((ValueError,TimeoutError)):h.authority_fence('post_pre_array_before_first_numeric_native_or_array')
    assert invoked==[]

def test_current_authority_scope68_eight_fields_eighteen_relocations_and_no_reset(tmp_path,monkeypatch):
    authority=h.current_revalidation_amendment()
    inherited=h.strict_json(h.read_spec(authority['inherited68']))['ledger'];live=granted_live(activated=True)
    assert authority['fixed_authority_scope_policy']==h.FIXED_SCOPE_POLICY
    assert authority['manual_generated_pass_id']=='e360-generous-p7' and authority['pass8_authorized'] is False
    for key in ('debits','e360_calls','e360_passes','e360_resource_grant',*authority['old_activation_fields_preserved']):assert live[key]==inherited[key]
    assert round(live['generated_shared_cumulative_used_s'],9)==274.1960156 and live['e360_generated_local_used_s']==163.239
    rows=authority['historical_concrete_fix_relocations'];assert len(rows)==18
    for row in rows[-4:]:
        raw=h.historical_fix_bytes(row['original'],pass_id='e360-generous-p6')
        assert len(raw)==row['original']['bytes'] and h.sha(raw)==row['original']['sha256']
        with pytest.raises(ValueError):h.historical_fix_bytes(row['original'],pass_id='e360-generous-p7')
    reg=h.registration();real_safe=h.safe;path=tmp_path/'live.json'
    monkeypatch.setattr(h,'safe',lambda name,root=h.ROOT:path if name==reg['shared_ledger_path'] else real_safe(name,root))
    path.write_bytes(h.json_bytes(live));assert len(h.ledger_state(reg)[0])==68
    for key in (*authority['old_activation_fields_preserved'],*authority['new_active_fields']):
        bad=copy.deepcopy(live);bad.pop(key);path.write_bytes(h.json_bytes(bad))
        with pytest.raises((ValueError,KeyError)):h.ledger_state(reg)
    path.write_bytes(h.json_bytes(granted_live(activated=True,ruff_historical=True)))
    with pytest.raises((ValueError,KeyError)):h.ledger_state(reg)


def test_manual_p7_exact_closed_fixture_exit1_C7_fresh_sources_no_p8(monkeypatch):
    authority=h.current_revalidation_amendment();live=granted_live(activated=True)
    frozen=h.strict_json(h.read_spec(h.execution_budget_amendment()['inherited73']))['ledger']['e360_passes']['e360-generous-p7']
    passed={'mode':'generated','modified':True,'actual_failure':authority['actual_predecessor_failure'],
            'predecessor_failed_cap_s':30,'source_sha256':copy.deepcopy(frozen['source_sha256']),
            'concrete_fix':copy.deepcopy(frozen['concrete_fix'])}
    failure=h.charged_cost(passed['actual_failure'],'generated')
    assert failure['exit_code']==1 and failure['qualified'] is False and failure['role']=='fixture'
    h.validate_p7_pass('e360-generous-p7',passed,live['debits'])
    for identity in ('e360-generous-p8','e360-auto','e360-generous-p6'):
        with pytest.raises(ValueError):h.validate_p7_pass(identity,passed,live['debits'])
    for spec in passed['concrete_fix']:
        bad=copy.deepcopy(passed);bad['concrete_fix'].remove(spec)
        with pytest.raises(ValueError):h.validate_p7_pass('e360-generous-p7',bad,live['debits'])
    for key in passed['source_sha256']:
        bad=copy.deepcopy(passed);bad['source_sha256'][key]=False
        with pytest.raises(ValueError):h.validate_p7_pass('e360-generous-p7',bad,live['debits'])
    predecessor=[d for d in live['debits'] if d['descriptor']==passed['actual_failure']];assert len(predecessor)==1
    with pytest.raises(ValueError,match='once debited'):h.validate_p7_pass('e360-generous-p7',passed,live['debits']+[copy.deepcopy(predecessor[0])])
    real=h.charged_cost
    for field,value in (('exit_code',True),('qualified',True),('all_owned_ended',False),('reservation_released',False),('role','suite')):
        changed={**failure,field:value}
        monkeypatch.setattr(h,'charged_cost',lambda *args,_changed=changed,**kwargs:_changed)
        with pytest.raises(ValueError):h.validate_p7_pass('e360-generous-p7',passed,live['debits'])
    monkeypatch.setattr(h,'charged_cost',real)

def test_ROOT_current_scope_activation_and_complete_fixture_predecessor_production_probe(tmp_path):
    import subprocess
    source=h.safe(h.BASE+'/parent-e360-root-account.ps1').read_text('utf8')
    functions=source[source.index('function Task-Path('):source.index('function Sum-Actual(')]
    activation=source[source.index('$authorityScopeActivation=@{'):source.index("if($Action -cnotin @('recover-scope','activate-scope'))")]
    authority=h.current_revalidation_amendment();inherited=h.strict_json(h.read_spec(authority['inherited68']))['ledger']
    historical=inherited['e360_passes']['e360-generous-p6']
    sources=[h.file_spec(name) for name in (*h.OWNED,h.BASE+'/parent-e360-root-account.ps1',h.BASE+'/parent-e360-root-metadata-helper.py')]
    fresh=b'VALUE=1\n';(tmp_path/'fresh.py').write_bytes(fresh)
    payload={'authority':authority,'state':inherited,'failure':h.charged_cost(authority['actual_predecessor_failure'],'generated'),
             'sources':sources,'own9':h.own_sources(),'descriptor':h.file_spec(h.REVALIDATION_AMENDMENT),
             'extension':h.file_spec(h.RESOURCE_EXTENSION),'ruff_authority':h.ruff_recovery_amendment(),
             'historical':historical,'ruff_failure':h.charged_cost(h.ruff_recovery_amendment()['actual_predecessor_failure'],'generated'),
             'fresh':h.descriptor('fresh.py',fresh)}
    data=tmp_path/'input.json';data.write_bytes(h.json_bytes(payload));script=tmp_path/'current-scope.ps1'
    script.write_text("param([string]$InputPath)\n$ErrorActionPreference='Stop'\n$taskRoot=[IO.Path]::GetDirectoryName($InputPath)\n"+functions+r'''
$p=[IO.File]::ReadAllText($InputPath)|ConvertFrom-Json -AsHashtable
function Clone($value){return ($value|ConvertTo-Json -Depth 100 -Compress|ConvertFrom-Json -AsHashtable)}
function Check([bool]$condition,[string]$message){if(-not $condition){throw $message}}
$revalidation=$p.authority;$revalidationSpec=$p.descriptor;$extensionSpec=$p.extension
'''+activation+r'''
$round=Clone $authorityScopeActivation
Check (Equal-Typed $round $authorityScopeActivation) 'actual constructor JSON mismatch'
Check ($round.manual_generated_pass_id -ceq 'e360-generous-p7' -and $round.schema -ceq 'flylab.m3.e360.authority-scope-activation.v1') 'wrong current constructor'
$bad=Clone $round;$bad.new_generated_grant_s=$false
Check (-not (Equal-Typed $bad $authorityScopeActivation)) 'bool grant admitted'
Check (Authority-Scope-Activation-Allowed $p.state $true) 'exact68 activation rejected'
$bad=Clone $p.state;$bad.e360_authority_scope_activation=$round
Check (-not (Authority-Scope-Activation-Allowed $bad $true)) 'repeated activation admitted'
$bad=Clone $p.state;$bad.original_jobs_running=$false
Check (-not (Authority-Scope-Activation-Allowed $bad $true)) 'bool ownership admitted'
Check (-not (Authority-Scope-Activation-Allowed $p.state $false)) 'present lock admitted'
Check (Ruff-Recovery-Historical-Pass $p.historical $p.historical $p.ruff_authority $p.state.debits $p.ruff_failure) 'actual p6 as-executed history rejected'
$bad=Clone $p.historical;$bad.source_sha256['tools/verify_c_banc_m3_source_closure_diagnostic.py']=$p.own9['tools/verify_c_banc_m3_source_closure_diagnostic.py']
Check (-not (Ruff-Recovery-Historical-Pass $bad $p.historical $p.ruff_authority $p.state.debits $p.ruff_failure)) 'current source substituted into p6 history'
$manual=@{mode='generated';modified=$true;actual_failure=$revalidation.actual_predecessor_failure;
 predecessor_failed_cap_s=$revalidation.predecessor_failed_cap_s;concrete_fix=@($revalidation.concrete_correction_snapshot)+@($p.sources);source_sha256=$p.own9}
Check (Authority-Scope-Manual-Pass 'e360-generous-p7' $manual $revalidation $p.state.debits $p.failure) 'complete actual p7 admission rejected'
Check (-not (Authority-Scope-Manual-Pass 'e360-generous-p8' $manual $revalidation $p.state.debits $p.failure)) 'p8 admitted'
$bad=Clone $manual;$bad.modified=1
Check (-not (Authority-Scope-Manual-Pass 'e360-generous-p7' $bad $revalidation $p.state.debits $p.failure)) 'nonbool modified admitted'
$bad=Clone $manual;$bad.predecessor_failed_cap_s=$true
Check (-not (Authority-Scope-Manual-Pass 'e360-generous-p7' $bad $revalidation $p.state.debits $p.failure)) 'bool cap admitted'
$bad=Clone $manual;$bad.concrete_fix=@($p.sources)
Check (-not (Authority-Scope-Manual-Pass 'e360-generous-p7' $bad $revalidation $p.state.debits $p.failure)) 'missing C7 admitted'
foreach($spec in $p.sources){
 $bad=Clone $manual;$bad.concrete_fix=@($bad.concrete_fix|Where-Object{$_.path -cne $spec.path})
 Check (-not (Authority-Scope-Manual-Pass 'e360-generous-p7' $bad $revalidation $p.state.debits $p.failure)) 'missing current source admitted'
 $bad=Clone $manual;$bad.source_sha256[$spec.path]='0'*64
 Check (-not (Authority-Scope-Manual-Pass 'e360-generous-p7' $bad $revalidation $p.state.debits $p.failure)) 'wrong source SHA admitted'
 foreach($field in @('bytes','sha256')){
  $bad=Clone $manual;foreach($item in $bad.concrete_fix){if($item.path -ceq $spec.path){$item[$field]=$true}}
  Check (-not (Authority-Scope-Manual-Pass 'e360-generous-p7' $bad $revalidation $p.state.debits $p.failure)) 'source type admitted'
 }
}
$predecessor=@($p.state.debits|Where-Object{Equal-Typed $_.descriptor $manual.actual_failure})
Check ($predecessor.Count -eq 1) 'worker predecessor not once-debited'
Check (-not (Authority-Scope-Manual-Pass 'e360-generous-p7' $manual $revalidation (@($p.state.debits)+@($predecessor[0])) $p.failure)) 'replayed worker admitted'
foreach($field in @('exit_code','qualified','all_owned_ended','reservation_released')){
 $bad=Clone $p.failure;$bad[$field]=if($field -ceq 'exit_code'){$true}elseif($field -ceq 'qualified'){$true}else{$false}
 Check (-not (Authority-Scope-Predecessor $bad $revalidation.predecessor_failed_cap_s)) 'bad worker proof admitted'
}
$null=Checked-Source $p.fresh
$bad=Clone $p.fresh;$bad.bytes++
$rejected=$false;try{$null=Checked-Source $bad}catch{$rejected=$true};Check $rejected 'wrong source bytes admitted by actual fresh reader'
$bad=Clone $p.fresh;$bad.sha256='0'*64
$rejected=$false;try{$null=Checked-Source $bad}catch{$rejected=$true};Check $rejected 'wrong source SHA admitted by actual fresh reader'
@{schema='flylab.m3.e360.generated-current-scope-probe.v1';original_calls=0;live_ledger_writes=0;current_constructor=$round;complete_pass=$true;historical_p6=$true}|ConvertTo-Json -Depth 100 -Compress
''',encoding='utf8')
    result=subprocess.run(['pwsh','-NoProfile','-NonInteractive','-File',str(script),str(data)],capture_output=True,text=True,timeout=8,check=False)
    assert result.returncode==0,result.stderr
    proof=h.strict_json(result.stdout.encode('utf8'))
    assert proof['schema']=='flylab.m3.e360.generated-current-scope-probe.v1'
    assert proof['current_constructor']==h.current_revalidation_event() and proof['complete_pass'] and proof['historical_p6']
    assert proof['original_calls']==proof['live_ledger_writes']==0

def test_fixed_scope_actual_pre_spawn_and_pre_array_fences_before_numeric(tmp_path,monkeypatch):
    import time
    path,raw,_=finite_scope_authority(tmp_path,monkeypatch)
    desc={'path':'authority.json','bytes':1,'sha256':'a'*64}
    c={'mode':'generated','role':'fixture','attempt_id':'e360-finite','attempt':1,'source_sha256':{},
       'paths':{'pre_spawn':'receipt.json'},'limits':{'outer_s':30,'child_s':24,'internal_s':20,'science_s':16},'original_calls':0,
       'cumulative_before_s':274.1960156,'cumulative_limit_s':600,'reserved_s':30,'pending_reservations':{},
       'stage_used_s':194.006,'stage_limit_s':600,'max_tree_rss_bytes':2<<30,'retained_reserve_bytes':1<<20,
       'cpu_threads':1,'registration':desc,'amendment':desc,'resource_extension':desc,'revalidation_amendment':desc,
       'closure_report':desc,'future_administration_reserved_s':70,'local_used_s':163.239,'local_limit_s':489}
    (tmp_path/'contract.json').write_bytes(h.json_bytes(c));spec=h.file_spec('contract.json',tmp_path)
    monkeypatch.setattr(h,'validate_contract',lambda *args,**kwargs:(copy.deepcopy(c),spec,{}))
    monkeypatch.setattr(h,'sys',SimpleNamespace(modules={}))  # Finite unavailable scientific process dependencies.
    monkeypatch.setitem(sys.modules,'psutil',SimpleNamespace(Process=lambda pid:SimpleNamespace(pid=pid,create_time=lambda:456.)))
    entry=time.perf_counter()
    h.pre_spawn('contract.json',entry,root=tmp_path)
    stored=h.strict_json((tmp_path/'receipt.json').read_bytes())
    assert stored['entry_perf_counter']==entry and stored['original_array_reads']==stored['scientific_native_imports']==0
    with h.fixed_authority_scope(root=tmp_path,check=lambda:None):
        h.registration(tmp_path);h.registration(tmp_path)
        assert h.pre_array('contract.json','receipt.json',root=tmp_path)[0]==c
        path.write_bytes(raw.replace(b'1',b'2',1))
        with pytest.raises(ValueError,match='fence'):h.pre_array('contract.json','receipt.json',root=tmp_path)
    path.write_bytes(raw);c['paths']['pre_spawn']='rejected-receipt.json'
    (tmp_path/'contract.json').write_bytes(h.json_bytes(c));spec=h.file_spec('contract.json',tmp_path)
    with h.fixed_authority_scope(root=tmp_path,check=lambda:None):
        h.registration(tmp_path)
        path.write_bytes(raw.replace(b'1',b'2',1))
        with pytest.raises(ValueError,match='fence'):h.pre_spawn('contract.json',entry,root=tmp_path)
        assert not (tmp_path/'rejected-receipt.json').exists()


def test_fixed_scope_failure_outer_retains_primary_unknown_pending_and_closed_evidence(tmp_path,monkeypatch):
    for mode in ('expired','drift'):
        root=tmp_path/mode;root.mkdir()
        path,raw,_=finite_scope_authority(root,monkeypatch)
        paths=h.names('e360-failed-'+mode,'fixture')
        c={'paths':paths,'limits':{'output_cap_bytes':16<<20}}
        now=[10.]
        clock=h.Clock(10.,{'outer_s':30},clock=lambda _now=now:_now[0])
        record={'schema':'flylab.m3.e360.outer-provisional.v1','first_error':None,'secondary_errors':[],
                'natural_closed':False,'all_owned_ended':False,'reservation_released':False,
                'observed_pid_create_times':{'123':456.},'unknown_owned_identities':{'Popen-unreturned':'unknown'},
                'unresolved_matching_identities':{},'missing_owned_creation_times':[123],
                'actual_full_return_pending':True,'contract':{'path':paths['contract'],'bytes':1,'sha256':'a'*64}}
        h.fail(record,RuntimeError('earliest scientific failure'),'science')
        with h.fixed_authority_scope(root=root,check=lambda _clock=clock:_clock.check('outer')) as scope:
            h.registration(root);h.registration(root)
            if mode=='expired':now[0]=41.
            else:path.write_bytes(raw+b' ')
            assert h.publish_outer(c,record,clock,root=root) is record
            assert scope.failed and scope.descriptors and not scope.values
        saved=h.strict_json(h.safe(paths['outer'],root).read_bytes())
        assert saved['first_error']['message']=='earliest scientific failure' and saved['secondary_errors']
        assert saved['unknown_owned_identities']==record['unknown_owned_identities']
        assert saved['observed_pid_create_times']=={'123':456.} and saved['missing_owned_creation_times']==[123]
        assert not saved['all_owned_ended'] and not saved['reservation_released'] and saved['actual_full_return_pending']
        assert saved['revoked'] and not saved['run_valid'] and not saved['effective_qualified']
        evidence=saved['failure_publication']['closed_failed_evidence']
        assert evidence and h.strict_json(h.read_spec(evidence,root))['first_error']==saved['first_error']
        assert saved['failure_publication']['entry_perf_counter']==10. and not saved['failure_publication']['timely_success']


def test_fixed_scope_actual_child_failed_envelope_revocation_survives_drift_and_expiry(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'sys',SimpleNamespace(modules={},getprofile=sys.getprofile,setprofile=sys.setprofile))
    for mode in ('expired','drift'):
        root=tmp_path/mode;root.mkdir()
        path,raw,_=finite_scope_authority(root,monkeypatch)
        paths=h.names('e360-child-failed-'+mode,'fixture')
        desc={'path':'finite.json','bytes':1,'sha256':'a'*64}
        c={'paths':paths,'mode':'generated','role':'fixture','attempt_id':'e360-child-failed-'+mode,
           'source_sha256':{},'registration':desc,'amendment':desc,'resource_extension':desc,
           'revalidation_amendment':desc,'closure_report':desc,'adapter_qualification':None,
           'cumulative_before_s':274.1960156,'cumulative_limit_s':600,
           'limits':{'output_cap_bytes':16<<20},'retained_reserve_bytes':1<<20}
        now=[10.];clock=h.Clock(10.,{'internal_s':20,'science_s':16},clock=lambda _now=now:_now[0])
        def diagnostic(*args,_mode=mode,_path=path,_raw=raw,_now=now,**kwargs):
            if _mode=='expired':_now[0]=31.
            else:_path.write_bytes(_raw+b' ')
            raise RuntimeError('earliest generated child failure')
        monkeypatch.setattr(h,'execute_diagnostic_path',diagnostic)
        with h.fixed_authority_scope(root=root,check=clock.check):
            h.registration(root);h.registration(root)
            with h.descriptors_only_execution():assert h._child(c,desc,desc,{},clock,root=root)==1
        for name in ('envelope.json','revocation.json'):
            value=h.strict_json(h.safe(paths['output']+'/'+name,root).read_bytes())
            assert value['first_error']['message']=='earliest generated child failure' and value['secondary_errors']
            assert value['actual_full_return_pending'] and value['revoked'] and not value['run_valid']
            assert not value['effective_qualified'] and value['failure_publication']['entry_perf_counter']==10.
            assert value['failure_publication']['closed_failed_evidence']


def test_fixed_scope_parent_ENTRY_science_and_internal_publication_caps(tmp_path,monkeypatch):
    finite_scope_authority(tmp_path,monkeypatch)
    for role,science,internal,outer,child in (('fixture',16,20,30,24),('suite',90,100,120,110),
                                             ('saved_prefix_diagnostic',9,10,12,11)):
        (tmp_path/'stage.json').write_bytes(h.json_bytes({'role':role,'limits':{
            'outer_s':outer,'child_s':child,'internal_s':internal,'science_s':science}}))
        check=h.operation_clock('stage.json',10.,root=tmp_path,stage='internal')
        assert check.timer.entry==10. and check.timer.limits=={
            'outer_s':outer,'child_s':child,'internal_s':internal,'science_s':science}
        now=[10.+science-.1];check.timer.clock=lambda _now=now:_now[0]
        with h.fixed_authority_scope(root=tmp_path,check=check) as scope:
            h.registration(tmp_path);h.registration(tmp_path)
            h.authority_fence('post_pre_array_before_first_numeric_native_or_array')
            now[0]=10.+science+.1
            with h.descriptors_only_execution():
                h.durable(tmp_path/(role+'-within-internal.json'),b'{}',cap=16)
                with pytest.raises(TimeoutError,match='science'):h.authority_fence('post_pre_array_before_first_numeric_native_or_array')
            assert scope.failed and scope.descriptors and not scope.values
        now[0]=10.+internal+.1
        with pytest.raises(TimeoutError,match='internal'), h.fixed_authority_scope(root=tmp_path,check=check):pass
        assert not (tmp_path/(role+'-late.json')).exists()


def test_execution_budget_v2_exact73_ninth_reader_historical_caps_and_no_P9():
    value=h.execution_budget_amendment()
    assert h.FIXED_READERS['execution_budget_amendment']==(h.EXECUTION_BUDGET_AMENDMENT,24334,h.EXECUTION_BUDGET_SHA)
    assert len(value['fixed_authority_scope_policy']['reader_ids'])==9 and len(h.FIXED_READERS)==15 and value['primary_total_reserved_s']==273
    assert value['generated_pass_limit']==8 and value['modified_generated_pass_limit']==7 and value['total_pass_limit']==9
    assert value['new_generated_grant_s']==value['automatic_retries']==0 and not value['pass9_authorized']
    assert value['superseded_unexecuted_registration']['path']!=h.EXECUTION_BUDGET_AMENDMENT
    inherited=h.strict_json(h.read_spec(value['inherited73']))['ledger']
    assert len(inherited['debits'])==73 and len(value['historical_concrete_fix_relocations'])==22
    assert inherited['generated_shared_cumulative_used_s']==317.1840156 and inherited['e360_generated_local_used_s']==206.227
    assert set(value['preserved_test_functions'])<=set(h.test_functions()) and len(value['preserved_test_functions'])==52
    for number in range(1,8):
        policy=h.effective_policy(pass_id='e360-generous-p'+str(number))
        assert policy['effective_roles']['fixture']['outer_s']==30
        assert policy['scanner_limits']['outer_s']==20 and policy['helper_limits']['fixture_build']['outer_s']==20
    policy=h.effective_policy(pass_id='e360-generous-p8')
    assert policy['effective_roles']['fixture']==value['effective_roles']['fixture']
    assert policy['successful_suffix_after_s']==[270,255,240,185,175,155,35,20,0]
    assert sum(row['outer_s'] for row in value['generated_call_schedule'])==273
    assert h.effective_policy(pass_id='e360-generous-p9')['effective_roles']['suite']['outer_s']==600
    for identity in ('e360-generous-p13','e360-generous-p80'):
        with pytest.raises(ValueError):h.effective_policy(pass_id=identity)


def test_suite_recovery_P9_literal_tenth_reader_approval_and_all71_preserved():
    value=h.suite_recovery_amendment();old=h.execution_budget_amendment()
    assert len(h.FIXED_READERS)==15 and h.FIXED_READERS['suite_recovery_amendment']==(h.SUITE_RECOVERY_AMENDMENT,28728,h.SUITE_RECOVERY_SHA)
    assert value['fixed_authority_scope_policy']['reader_ids'][-2:]==['execution_budget_amendment','suite_recovery_amendment']
    assert [value['fixed_authority_scope_policy'][key] for key in ('max_entries','max_text_descriptors','max_literal_json_bytes','max_aggregate_literal_json_bytes')]==[16,512,262144,2097152]
    assert value['historical_concrete_fix_relocations'][:22]==old['historical_concrete_fix_relocations']
    assert len(value['historical_concrete_fix_relocations'])==26
    assert set(value['preserved_test_functions'])<=set(h.test_functions()) and len(value['preserved_test_functions'])==71
    assert value['generated_pass_limit']==9 and value['modified_generated_pass_limit']==8 and value['total_pass_limit']==10
    assert value['original_pass_limit']==1 and value['automatic_retries']==0 and not value['pass10_authorized'] and not value['original_execution_authority']
    assert value['generated_scope_ends_after_Q_debit'] and not value['source_hash_cycle_allowed']
    assert sum(row['outer_s'] for row in value['generated_call_schedule'])==798
    event=h.suite_recovery_event();grant=h.suite_recovery_grant()
    assert set(event)==set(value['activation_policy']['activation_keys']) and len(event)==8
    assert set(grant)==set(value['activation_policy']['resource_grant_keys']) and len(grant)==10
    assert event['resource_extension']==value['resource_extension']==h.file_spec(h.RESOURCE_EXTENSION)
    assert event['human_authorization']==grant['authorization']==value['existing_human_authorization']
    approval=h.strict_json(h.read_spec(grant['authorization']))
    assert approval['new_generated_grant_s']==800 and approval['inherited81']==value['inherited81']


def test_suite_recovery_live_selector_partial_extra_types_uncached_nonrecursive(tmp_path,monkeypatch):
    value=h.suite_recovery_amendment();base=h.strict_json(h.read_spec(value['inherited81']))['ledger']
    live=suite_recovery_live()
    assert h.current_budget_authority(live=base)=={'suite_recovery_active':False,'generated_shared_limit_s':600,'generated_local_limit_s':489}
    def forbidden(*args,**kwargs):raise AssertionError('recursive budget selector')
    with monkeypatch.context() as patch:
        patch.setattr(h,'ledger_state',forbidden);patch.setattr(h,'validate_live_calls',forbidden);patch.setattr(h,'effective_policy',forbidden)
        assert h.current_budget_authority(live=live)=={'suite_recovery_active':True,'generated_shared_limit_s':1400,'generated_local_limit_s':1289}
    for mask in range(1,7):
        changed=copy.deepcopy(base)
        for index,key in enumerate(h.SUITE_RECOVERY_FIELDS):
            if mask&(1<<index):changed[key]=live[key]
        with pytest.raises(ValueError,match='complete new pair'):h.current_budget_authority(live=changed)
    for container in ('e360_suite_recovery_activation','e360_suite_recovery_resource_grant'):
        for field,item in (('new_generated_grant_s',800.),('new_generated_grant_s',True),('unexpected',1)):
            changed=copy.deepcopy(live);changed[container][field]=item
            with pytest.raises(ValueError):h.current_budget_authority(live=changed)
        changed=copy.deepcopy(live);changed[container].pop('inherited81')
        with pytest.raises(ValueError):h.current_budget_authority(live=changed)
    for field,item in (('generated_shared_cumulative_limit_s',1400.),('e360_generated_local_limit_s',True)):
        changed=copy.deepcopy(live);changed[field]=item
        with pytest.raises(ValueError):h.current_budget_authority(live=changed)
    path,_=install_suite_recovery_live(tmp_path,monkeypatch)
    with h.fixed_authority_scope(check=lambda:None):
        assert h.current_budget_authority()['suite_recovery_active']
        changed=copy.deepcopy(live);changed.pop(h.SUITE_RECOVERY_FIELDS[2]);path.write_bytes(h.json_bytes(changed))
        with pytest.raises(ValueError):h.current_budget_authority()
    path.write_bytes(h.json_bytes(base))
    with pytest.raises(ValueError,match='preactivation'):
        h.check_call_request({'pass_id':'e360-generous-p9'})


def test_suite_recovery_current_full_suffix_and_immutable_historical_caps(tmp_path,monkeypatch):
    path,live=install_suite_recovery_live(tmp_path,monkeypatch)
    assert h.effective_policy()['effective_roles']['suite']['outer_s']==600
    assert h.effective_policy(pass_id='e360-generous-p8')['effective_roles']['suite']['outer_s']==120
    assert h.effective_policy(pass_id='e360-generous-p7')['effective_roles']['fixture']['outer_s']==30
    policy=h.effective_policy(pass_id='e360-generous-p9')
    assert policy['successful_suffix_after_s']==[795,775,750,695,680,650,50,30,0]
    assert [policy['helper_limits'][key]['outer_s'] for key in ('fixture_build','fixture_seal','suite_build','suite_seal','qualification')]==[25,15,30,20,30]
    assert [policy['helper_limits'][key]['internal_s'] for key in ('fixture_build','fixture_seal','suite_build','suite_seal','qualification')]==[22,13,27,18,27]
    assert policy['effective_roles']['suite']==h.suite_recovery_amendment()['effective_roles']['suite']
    assert policy['effective_roles']['saved_prefix_diagnostic']==h.resource_extension()['effective_roles']['saved_prefix_diagnostic']
    semantics={**live,'e360_calls':{'finite':{'mode':'generated','state':'begun','reserved_s':3,'future_reserved_s':795}},'e360_pending_actuals':{}}
    reg=h.registration();h.require_fit(reg,semantics,'generated')
    assert h.prospective_remaining(semantics,'generated')==798
    too_large={**semantics,'e360_generated_local_used_s':492.}
    with pytest.raises(ValueError,match='future admission fits'):h.require_fit(reg,too_large,'generated')
    too_large={**semantics,'generated_shared_cumulative_used_s':603.}
    with pytest.raises(ValueError,match='future admission fits'):h.require_fit(reg,too_large,'generated')
    inherited=h.strict_json(h.read_spec(h.suite_recovery_amendment()['inherited81']))['ledger']
    path.write_bytes(h.json_bytes(inherited))
    assert h.effective_policy()['effective_roles']['suite']['outer_s']==120
    with pytest.raises(ValueError,match='future admission fits'):h.require_fit(reg,semantics,'generated')
    for identity in ('e360-generous-p13','e360-generous-p99'):
        with pytest.raises(ValueError):h.effective_policy(pass_id=identity)
    # Real build/failure functions with finite metadata seams exercise the P14 conditions.
    current=p14_recovery_live();budget=h.current_budget_authority(live=current)
    assert budget['generated_shared_limit_s']==1400 and budget['generated_local_limit_s']==1289
    report={'diagnostic_ready':True};report_spec={'path':'finite-report.json','bytes':1,'sha256':'a'*64}
    actual_read=h.read_spec
    with monkeypatch.context() as patch:
        patch.setattr(h,'current_budget_authority',lambda root=h.ROOT,**kwargs:copy.deepcopy(budget))
        patch.setattr(h,'prospective_authority_spec',lambda root=h.ROOT:h.file_spec(h.P14_RECOVERY_AMENDMENT,root))
        patch.setattr(h,'read_spec',lambda spec,root=h.ROOT:h.json_bytes(report) if spec==report_spec else actual_read(spec,root))
        patch.setattr(h,'scan_current',lambda *args,**kwargs:copy.deepcopy(report))
        patch.setattr(h,'closure_sources',lambda *args,**kwargs:h.own_sources())
        for role,reserve in (('fixture',95),('suite',50)):
            call={'call_role':role+'_build','attempt_id':'e360-r1-finite','pass_id':'e360-generous-p14'}
            patch.setattr(h,'ledger_state',lambda *args,call=call,**kwargs:([],{'e360_calls':{'finite':call}},
                {'generated':0.,'original':0.},{'generated':0.,'original':0.}))
            built=h.build(role,'e360-r1-finite',closure_report=report_spec)
            assert built['future_administration_reserved_s']==reserve and built['pass_id']=='e360-generous-p14'
            assert built['cumulative_limit_s']==1400 and built['local_limit_s']==1289
            failure=h.failed_authority_view({'pass_id':'e360-generous-p14'},h.names('e360-r1-finite',role)['contract'])
            assert failure['cumulative_limit_s']==1400 and failure['local_limit_s']==1289
            if role=='fixture':fixture14=copy.deepcopy(built)
    # Production outer failure publication retains the actual P14 parent ENTRY before Popen.
    with monkeypatch.context() as patch:
        observed=[];contract_name=fixture14['paths']['contract']
        patch.setitem(sys.modules,'psutil',SimpleNamespace(Process=lambda pid:SimpleNamespace(create_time=lambda:456.)))
        patch.setattr(h,'bounded_json_name',lambda *args,**kwargs:h.json_bytes(fixture14))
        patch.setattr(h,'failed_authority_view',lambda *args,**kwargs:copy.deepcopy(fixture14))
        patch.setattr(h,'file_spec',lambda name,root=h.ROOT:{'path':name,'bytes':1,'sha256':'a'*64})
        patch.setattr(h,'bounded_owned_completion',lambda *args,**kwargs:({},{},[]))
        patch.setattr(h,'publish_outer',lambda c,record,*args,**kwargs:observed.append(copy.deepcopy(record)) or record)
        patch.setattr(h.time,'perf_counter',lambda:11.)
        def forbidden_popen(*args,**kwargs):raise AssertionError('finite failed ENTRY must not dispatch')
        patch.setattr(h.subprocess,'Popen',forbidden_popen)
        assert h._supervise(contract_name,entry=10.,root=tmp_path,initial_error=ValueError('first P14 admission failure'))==1
        assert observed[0]['entry_perf_counter']==10. and observed[0]['Popen_never_attempted'] is True
        assert observed[0]['first_error']['message']=='first P14 admission failure'


def test_suite_recovery_native_once_activation_whitelist_grant_and_history(tmp_path):
    import subprocess
    source=h.safe(h.BASE+'/parent-e360-root-account.ps1').read_text('utf8')
    functions=source[source.index('function Task-Path('):source.index('function Sum-Actual(')]
    historical_reader=source[source.index('function Checked-Historical-Fix('):source.index('$inherited59=')]
    value=h.suite_recovery_amendment();frozen=h.strict_json(h.read_spec(value['inherited81']))['ledger']
    p8=h.execution_budget_amendment()
    historical7=h.strict_json(h.read_spec(p8['inherited73']))['ledger']['e360_passes']['e360-generous-p7']
    actual_failure=h.strict_json(h.read_spec(value['actual_predecessor_failure']))
    manual_pass={'mode':'generated','modified':True,'actual_failure':value['actual_predecessor_failure'],
        'predecessor_failed_cap_s':120,'source_sha256':h.own_sources(),
        'concrete_fix':[value['concrete_correction_snapshot'],*(h.file_spec(name) for name in
            (*h.OWNED,h.BASE+'/parent-e360-root-account.ps1',h.BASE+'/parent-e360-root-metadata-helper.py'))]}
    payload={'authority':value,'state':frozen,'spec':h.file_spec(h.SUITE_RECOVERY_AMENDMENT),'extension':h.file_spec(h.RESOURCE_EXTENSION),'root':str(h.ROOT),'base':h.BASE,
        'historical7':historical7,'historical8':frozen['e360_passes']['e360-generous-p8'],
        'old_rows':h.current_revalidation_amendment()['historical_concrete_fix_relocations'],
        'rows7':p8['historical_concrete_fix_relocations'][18:],'rows8':value['historical_concrete_fix_relocations'][22:],
        'current_own':h.file_spec(h.OWNED[0]),'actual_failure':actual_failure,'manual_pass':manual_pass}
    data=tmp_path/'input.json';data.write_bytes(h.json_bytes(payload))
    script=tmp_path/'suite-recovery.ps1'
    script.write_text("param([string]$InputPath)\n$ErrorActionPreference='Stop'\n"+functions+historical_reader+r'''
$p=[IO.File]::ReadAllText($InputPath)|ConvertFrom-Json -AsHashtable
$taskRoot=$p.root;$taskBase=$p.base
function Check([bool]$ok,[string]$message){if(-not $ok){throw $message}}
function Clone($value){return ($value|ConvertTo-Json -Depth 100 -Compress|ConvertFrom-Json -AsHashtable)}
$inactive=Current-Budget-Authority $p.state
Check (-not $inactive.active -and $inactive.shared -eq 600 -and $inactive.local -eq 489) 'exact inactive inherited81 selector'
foreach($field in @('generated_shared_cumulative_limit_s','e360_generated_local_limit_s')){
    foreach($wrong in @([bool]$true,[double]$p.state[$field],[long]0)){
        $bad=Clone $p.state;$bad[$field]=$wrong
        $rejected=$false;try{$null=Current-Budget-Authority $bad}catch{$rejected=$true}
        Check $rejected 'inactive wrong scalar value or bool/float type denied directly'
    }
    $bad=Clone $p.state;$null=$bad.Remove($field)
    $rejected=$false;try{$null=Current-Budget-Authority $bad}catch{$rejected=$true}
    Check $rejected 'inactive missing scalar denied directly'
}
foreach($row in $p.old_rows){$null=Checked-Historical-Fix $row.original $row.pass_id}
foreach($entry in @(@{pass='e360-generous-p7';proof=$p.historical7;rows=$p.rows7},@{pass='e360-generous-p8';proof=$p.historical8;rows=$p.rows8})){
    foreach($spec in $entry.proof.concrete_fix){$null=Checked-Historical-Fix $spec $entry.pass}
    foreach($row in $entry.rows){
        $null=Checked-Historical-Fix $row.original $entry.pass
        $wrongPass=if($entry.pass -ceq 'e360-generous-p7'){'e360-generous-p8'}else{'e360-generous-p7'}
        $rejected=$false;try{$null=Checked-Historical-Fix $row.original $wrongPass}catch{$rejected=$true}
        Check $rejected 'actual native exact historical pass binding'
        foreach($field in @('bytes','sha256')){
            $bad=Clone $row.original
            if($field -ceq 'bytes'){$bad.bytes++}else{$bad.sha256='0'*64}
            $rejected=$false;try{$null=Checked-Historical-Fix $bad $entry.pass}catch{$rejected=$true}
            Check $rejected 'actual native full historical descriptor drift denied'
        }
        $bad=Clone $row.original;$bad.extra='unauthorized descriptor member'
        $rejected=$false;try{$null=Checked-Historical-Fix $bad $entry.pass}catch{$rejected=$true}
        Check $rejected 'actual native descriptor extra member denied'
    }
    $rejected=$false;try{$null=Checked-Historical-Fix $p.current_own $entry.pass}catch{$rejected=$true}
    Check $rejected 'unregistered current mutable own descriptor never replaces historical source'
}
$rejected=$false;try{$null=Checked-Historical-Fix $p.current_own 'e360-generous-p9'}catch{$rejected=$true}
Check $rejected 'current P9 source requires separate fresh proof rather than historical reader'
Check ($p.actual_failure.source_sha256.Count -gt $p.historical8.source_sha256.Count) 'real full historical closure distinct from own9'
Check (Suite-Recovery-Manual-Pass 'e360-generous-p9' $p.manual_pass $p.authority $p.state.debits $p.actual_failure) 'actual closed P8 full closure and own9 projection admit fresh P9'
$bad=Clone $p.actual_failure;$bad.source_sha256=Clone $p.historical8.source_sha256
Check (-not (Suite-Recovery-Manual-Pass 'e360-generous-p9' $p.manual_pass $p.authority $p.state.debits $bad)) 'own9 cannot replace complete historical closure'
$outside=@($p.actual_failure.source_sha256.Keys|Where-Object{-not $p.historical8.source_sha256.Contains($_)})[0]
$bad=Clone $p.actual_failure;$null=$bad.source_sha256.Remove($outside)
Check (-not (Suite-Recovery-Manual-Pass 'e360-generous-p9' $p.manual_pass $p.authority $p.state.debits $bad)) 'dependency omission from historical closure denied'
$bad=Clone $p.actual_failure;$bad.source_sha256[$outside]='0'*64
Check (-not (Suite-Recovery-Manual-Pass 'e360-generous-p9' $p.manual_pass $p.authority $p.state.debits $bad)) 'dependency drift in historical closure denied'
$objects=Suite-Recovery-Objects $p.authority $p.spec $p.extension
Check ($objects.event.Count -eq 8 -and $objects.grant.Count -eq 10) 'exact event8 and distinct grant10'
Check (Equal-Typed (Clone $objects) $objects) 'native constructors stable after JSON typed roundtrip'
Check (Suite-Recovery-Activation-Allowed $p.state $p.state $p.authority $true) 'exact81 full798 grant fit'
Check (-not (Suite-Recovery-Activation-Allowed $p.state $p.state $p.authority $false)) 'original lock forbids activation'
$after=Clone $p.state
$after.e360_suite_recovery_activation=$objects.event;$after.e360_suite_recovery_amendment=$p.spec
$after.e360_suite_recovery_resource_grant=$objects.grant
$after.generated_shared_cumulative_limit_s=$p.authority.generated_shared_limit_s
$after.e360_generated_local_limit_s=$p.authority.generated_local_limit_s
$after.generated_shared_remaining_s=[double]($p.authority.generated_shared_limit_s-$p.state.generated_shared_cumulative_used_s)
Check (Suite-Recovery-Delta-Valid $p.state $after $p.authority $objects $p.spec) 'only new3 changed3 accepted'
Check ((Current-Budget-Authority $after).shared -eq 1400) 'actual nonrecursive live grant selector'
Check (-not (Suite-Recovery-Activation-Allowed $after $p.state $p.authority $true)) 'replay denied'
foreach($field in @('generated_shared_cumulative_used_s','e360_generated_local_used_s','phase')){
    $bad=Clone $after;$bad[$field]='changed'
    Check (-not (Suite-Recovery-Delta-Valid $p.state $bad $p.authority $objects $p.spec)) 'old values and phase preserved'
}
$bad=Clone $after;$bad.extra='new unauthorized key'
Check (-not (Suite-Recovery-Delta-Valid $p.state $bad $p.authority $objects $p.spec)) 'extra state key forbidden'
$bad=Clone $after;$bad.debits[80].actual_outer_full_return_s=[double]0
Check (-not (Suite-Recovery-Delta-Valid $p.state $bad $p.authority $objects $p.spec)) 'entire81 debit prefix preserved'
foreach($key in $p.authority.activation_policy.new_ledger_fields){
    $bad=Clone $after;$bad.Remove($key)
    $rejected=$false;try{$null=Current-Budget-Authority $bad}catch{$rejected=$true}
    Check $rejected 'partial pair/grant denied'
}
foreach($wrong in @([double]800,[bool]$true)){
    $bad=Clone $after;$bad.e360_suite_recovery_resource_grant.new_generated_grant_s=$wrong
    $rejected=$false;try{$null=Current-Budget-Authority $bad}catch{$rejected=$true}
    Check $rejected 'grant integer type substitution denied'
}
foreach($field in @('generated_shared_cumulative_used_s','e360_generated_local_used_s')){
    $bad=Clone $p.state;$bad[$field]=[double]700
    Check (-not (Suite-Recovery-Activation-Allowed $bad $bad $p.authority $true)) 'whole798 after grant required before write'
}
foreach($key in @('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations')){
    $bad=Clone $p.state;$bad[$key]=@{unknown=@{}}
    Check (-not (Suite-Recovery-Activation-Allowed $bad $bad $p.authority $true)) 'unknown ownership denied'
}
''',encoding='utf8')
    done=subprocess.run(['pwsh','-NoProfile','-NonInteractive','-File',str(script),str(data)],capture_output=True,text=True,
        timeout=8,check=False,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert done.returncode==0,done.stdout+'\n'+done.stderr


def test_suite_recovery_manual_P9_closed_killed_P8_failure_fresh_sources(tmp_path,monkeypatch):
    live=h.strict_json(h.read_spec(h.lint_recovery_amendment()['inherited82']))['ledger']
    install_suite_recovery_live(tmp_path,monkeypatch,live)
    passed=copy.deepcopy(live['e360_passes']['e360-generous-p9'])
    h.validate_p9_pass('e360-generous-p9',passed,live['debits'])
    for field,item in (('modified',False),('predecessor_failed_cap_s',600),('source_sha256',live['e360_passes']['e360-generous-p8']['source_sha256']),('concrete_fix',[])):
        with pytest.raises(ValueError):h.validate_p9_pass('e360-generous-p9',{**passed,field:item},live['debits'])
    for identity in ('e360-generous-p8','e360-generous-p10','e360-auto'):
        with pytest.raises(ValueError):h.validate_p9_pass(identity,passed,live['debits'])
    duplicate=next(row for row in live['debits'] if row['descriptor']==passed['actual_failure'])
    with pytest.raises(ValueError):h.validate_p9_pass('e360-generous-p9',passed,live['debits']+[duplicate])
    real=h.charged_cost;failure=real(passed['actual_failure'],'generated')
    assert failure['natural_closed'] is False and failure['exit_code']==1 and failure['actual_outer_full_return_s']==120.053
    frozen=live['e360_passes']['e360-generous-p8']['source_sha256']
    assert len(failure['source_sha256'])>len(frozen)==9
    outside=next(name for name in failure['source_sha256'] if name not in frozen)
    for changed_source in (frozen,{name:item for name,item in failure['source_sha256'].items() if name!=outside},
                           {**failure['source_sha256'],outside:'0'*64}):
        with monkeypatch.context() as patch:
            patch.setattr(h,'charged_cost',lambda *args,_source=changed_source,**kwargs:{**failure,'source_sha256':_source})
            with pytest.raises(ValueError):h.validate_p9_pass('e360-generous-p9',passed,live['debits'])
    for field,item in (('natural_closed',True),('qualified',True),('actual_outer_full_return_s',120.),('all_owned_ended',False),
        ('reservation_released',False),('unresolved_matching_identities',{'still-live':1}),('role','fixture'),('pass_id','e360-generous-p7')):
        with monkeypatch.context() as patch:
            patch.setattr(h,'charged_cost',lambda *args,_field=field,_item=item,**kwargs:{**failure,_field:_item})
            with pytest.raises(ValueError):h.validate_p9_pass('e360-generous-p9',passed,live['debits'])


def test_suite_recovery_exact_P8_C9_relocation_never_current_own9():
    value=h.suite_recovery_amendment();frozen=h.strict_json(h.read_spec(value['inherited81']))['ledger']['e360_passes']['e360-generous-p8']
    for row in value['historical_concrete_fix_relocations'][22:]:
        assert row['pass_id']=='e360-generous-p8' and row['original']['sha256']==frozen['source_sha256'][row['original']['path']]
        assert h.historical_fix_bytes(row['original'],pass_id='e360-generous-p8')==h.read_spec(row['frozen'])
        with pytest.raises(ValueError):h.historical_fix_bytes(row['original'],pass_id='e360-generous-p7')
        changed={**row['original'],'sha256':'0'*64}
        with pytest.raises(ValueError):h.historical_fix_bytes(changed,pass_id='e360-generous-p8')
    assert frozen['source_sha256']!=h.own_sources()
    h.validate_p8_pass('e360-generous-p8',frozen,h.strict_json(h.read_spec(value['inherited81']))['ledger']['debits'])
    fake={**frozen,'source_sha256':h.own_sources()}
    with pytest.raises(ValueError):h.validate_p8_pass('e360-generous-p8',fake,h.strict_json(h.read_spec(value['inherited81']))['ledger']['debits'])


def test_suite_recovery_actual_P9_reached_checkpoints_current_authority_and_source(tmp_path,monkeypatch):
    isolate_compact_boundary(monkeypatch)
    path,_=install_suite_recovery_live(tmp_path,monkeypatch);value=h.suite_recovery_amendment()
    contract={'pass_id':'e360-generous-p9','mode':'generated','role':'fixture','source_sha256':h.own_sources(),
        'revalidation_amendment':h.file_spec(h.SUITE_RECOVERY_AMENDMENT),'limits':value['effective_roles']['fixture']}
    contract,descriptor,receipt=finite_compact_contract(tmp_path,contract)
    actual_safe=h.safe
    monkeypatch.setattr(h,'safe',lambda name,root=h.ROOT:tmp_path/'contract.json' if name=='contract.json' else actual_safe(name,root))
    result={'contract':descriptor};timer_=h.Clock(10.,contract['limits']);now=[11.]
    monkeypatch.setattr(h.time,'perf_counter',lambda:now[0])
    for stage in h.CHECKPOINT_STAGES:
        h.checkpoint(contract,result,timer_,stage);now[0]+=.1
    result['elapsed_before_publication_s']=2.
    assert len(h.validate_execution_checkpoints(result,contract,receipt)['stages'])==7
    changed=copy.deepcopy(contract);changed['source_sha256'][h.OWNED[0]]='0'*64
    with pytest.raises(ValueError):h.validate_execution_checkpoints(result,changed,h.compact_receipt(changed,descriptor,10.,123,456.))
    changed={**contract,'revalidation_amendment':h.file_spec(h.EXECUTION_BUDGET_AMENDMENT)}
    with pytest.raises(ValueError):h.validate_execution_checkpoints(result,changed,h.compact_receipt(changed,descriptor,10.,123,456.))
    partial=copy.deepcopy(result);partial['execution_checkpoints']['stages']=partial['execution_checkpoints']['stages'][:3]
    partial.update(run_valid=False,effective_qualified=False)
    assert len(h.validate_execution_checkpoints(partial,contract,receipt,successful=False)['stages'])==3
    with pytest.raises(ValueError):h.validate_execution_checkpoints(partial,contract,receipt)
    path.write_bytes(h.json_bytes(h.strict_json(h.read_spec(value['inherited81']))['ledger']))
    with pytest.raises(ValueError,match='current activation'):h.validate_execution_checkpoints(result,contract,receipt)
    with monkeypatch.context() as patch:
        p14=h.p14_recovery_amendment();active=[True]
        c14,d14,r14=finite_compact_contract(tmp_path,{'pass_id':'e360-generous-p14','mode':'generated','role':'fixture',
            'source_sha256':h.own_sources(),'revalidation_amendment':h.file_spec(h.P14_RECOVERY_AMENDMENT),
            'limits':p14['effective_roles']['fixture']})
        patch.setattr(h,'current_scope_authority',lambda *args,**kwargs:active[0])
        t14=h.Clock(10.,c14['limits']);v14={'contract':d14};now[0]=11.
        for stage in h.CHECKPOINT_STAGES:
            h.checkpoint(c14,v14,t14,stage);now[0]+=.1
        v14['elapsed_before_publication_s']=2.
        assert len(h.validate_execution_checkpoints(v14,c14,r14)['stages'])==7
        active[0]=False
        with pytest.raises(ValueError,match='P14 checkpoints require actual uncached current scope'):
            h.validate_execution_checkpoints(v14,c14,r14)
        active[0]=True
        with pytest.raises(ValueError,match='ENTRY'):
            h.validate_execution_checkpoints(v14,c14,{**r14,'entry_perf_counter':11.})


def test_suite_recovery_preimport_scanner_helper_clocks_and_max84_journal(tmp_path,monkeypatch):
    import hashlib
    source=h.safe(h.OWNED[0]).read_text('utf8')
    bootstrap=source[source.index('_SCANNER_BOOTSTRAP_POLICY=None'):source.index('from tools import verify_c_banc_m3_candidate_body_handoff as lib')]
    request={'pass_id':'e360-generous-p9','revalidation_amendment':h.file_spec(h.SUITE_RECOVERY_AMENDMENT)}
    request_path=tmp_path/'request.json';request_path.write_bytes(h.json_bytes(request))
    literal=h.read_spec(request['revalidation_amendment'])
    class Root:
        def __truediv__(self,name):
            return request_path if name=='request.json' else self
        def read_bytes(self):return literal
    # Execute the actual pre-import fragment with finite contained text reads and the original ENTRY.
    monkeypatch.setattr(h.Path,'is_relative_to',lambda *args:True)
    scope={'ROOT':Root(),'os':SimpleNamespace(environ={'E360_CALL_REQUEST':'request.json'}),
        'sys':SimpleNamespace(argv=['tool','--scan']),'json':json,'hashlib':hashlib,'math':math,
        'time':SimpleNamespace(perf_counter=lambda:11.),'ENTRY':10.,'Path':type(h.ROOT)}
    exec(compile(bootstrap,'actual-scanner-bootstrap','exec'),scope)  # noqa: S102 -- actual pinned production bootstrap in a finite regression.
    assert scope['_SCANNER_BOOTSTRAP_POLICY']['outer_s']==20 and scope['_SCANNER_BOOTSTRAP_POLICY']['internal_s']==18
    request['revalidation_amendment']['bytes']=24334;request_path.write_bytes(h.json_bytes(request))
    with pytest.raises(ValueError,match='direct authority'):exec(compile(bootstrap,'actual-scanner-bootstrap','exec'),scope)  # noqa: S102 -- actual pinned production bootstrap in a finite regression.
    request={'pass_id':'e360-generous-p14','mode':'generated','revalidation_amendment':h.file_spec(h.P14_RECOVERY_AMENDMENT)}
    literal=h.read_spec(request['revalidation_amendment']);request_path.write_bytes(h.json_bytes(request))
    exec(compile(bootstrap,'actual-P14-scanner-bootstrap','exec'),scope)  # noqa: S102 -- same production pre-import fragment inside the admitted suite.
    assert scope['_SCANNER_BOOTSTRAP_POLICY']['outer_s']==20 and scope['_SCANNER_BOOTSTRAP_POLICY']['internal_s']==18
    for identity,mode in (('e360-generous-p13','generated'),('e360-generous-p15','generated'),('e360-generous-p14','original')):
        request_path.write_bytes(h.json_bytes({**request,'pass_id':identity,'mode':mode}))
        with pytest.raises(ValueError,match='cannot scan'):exec(compile(bootstrap,'actual-P14-scanner-denial','exec'),scope)  # noqa: S102 -- no project import reached by denied identities.
    helper=h.safe(h.BASE+'/parent-e360-root-metadata-helper.py').read_text('utf8')
    helper_bootstrap=helper[helper.index('request_path = ROOT / sys.argv[1]'):helper.index('from tools import verify_c_banc_m3_source_closure_diagnostic as h')]
    request={'pass_id':'e360-generous-p9','revalidation_amendment':h.file_spec(h.SUITE_RECOVERY_AMENDMENT),
        'call_role':'suite_build','action':'build','mode':'generated'}
    request_path.write_bytes(h.json_bytes(request))
    class HelperRoot:
        def __truediv__(self,name):return request_path if name=='request.json' else h.ROOT/name
    scope={'ROOT':HelperRoot(),'sys':SimpleNamespace(argv=['helper','request.json']),
        'json':json,'hashlib':hashlib,'time':SimpleNamespace(perf_counter=lambda:11.),'ENTRY':10.,'Path':type(h.ROOT)}
    exec(compile(helper_bootstrap,'actual-helper-bootstrap','exec'),scope)  # noqa: S102 -- actual pinned production bootstrap in a finite regression.
    assert scope['policy']['outer_s']==30 and scope['policy']['internal_s']==27 and scope['ENTRY']==10.
    request={'pass_id':'e360-generous-p14','revalidation_amendment':h.file_spec(h.P14_RECOVERY_AMENDMENT),
        'call_role':'suite_build','action':'build','mode':'generated'}
    request_path.write_bytes(h.json_bytes(request))
    exec(compile(helper_bootstrap,'actual-P14-helper-bootstrap','exec'),scope)  # noqa: S102 -- same actual pre-import fragment, original ENTRY.
    assert scope['policy']['outer_s']==30 and scope['policy']['internal_s']==27 and scope['ENTRY']==10.
    for identity,mode in (('e360-generous-p13','generated'),('e360-generous-p15','generated'),('e360-generous-p14','original')):
        request_path.write_bytes(h.json_bytes({**request,'pass_id':identity,'mode':mode}))
        with pytest.raises(ValueError,match='cannot dispatch'):exec(compile(helper_bootstrap,'actual-P14-helper-denial','exec'),scope)  # noqa: S102 -- denied before the project import.
    limits=h.suite_recovery_amendment()['effective_roles']['suite']
    contract={'pass_id':'e360-generous-p9','role':'suite','limits':limits,
        'revalidation_amendment':h.file_spec(h.SUITE_RECOVERY_AMENDMENT)}
    (tmp_path/'clock.json').write_bytes(h.json_bytes(contract))
    real_safe=h.safe
    monkeypatch.setattr(h,'safe',lambda name,root=h.ROOT:real_safe(name,h.ROOT) if name==h.SUITE_RECOVERY_AMENDMENT else real_safe(name,root))
    monkeypatch.setattr(h.time,'perf_counter',lambda:11.)
    checked=h.operation_clock('clock.json',10.,root=tmp_path,stage='internal')
    assert checked.timer.entry==10. and checked.timer.limits['internal_s']==540 and checked._admitted_root==tmp_path.resolve()
    with monkeypatch.context() as patch:
        patch.setattr(h,'safe',lambda name,root=h.ROOT:real_safe(name,h.ROOT) if name==h.P14_RECOVERY_AMENDMENT else real_safe(name,root))
        contract={'pass_id':'e360-generous-p14','mode':'generated','role':'suite','limits':limits,
            'revalidation_amendment':h.file_spec(h.P14_RECOVERY_AMENDMENT)}
        (tmp_path/'clock.json').write_bytes(h.json_bytes(contract))
        current=h.operation_clock('clock.json',10.,root=tmp_path,stage='internal')
        assert current.timer.entry==10. and current.timer.limits['internal_s']==540
        for identity,mode in (('e360-generous-p13','generated'),('e360-generous-p15','generated'),('e360-generous-p14','original')):
            (tmp_path/'clock.json').write_bytes(h.json_bytes({**contract,'pass_id':identity,'mode':mode}))
            with pytest.raises(ValueError,match='operation denied'):h.operation_clock('clock.json',10.,root=tmp_path,stage='internal')
    context,c=journal_test_context(tmp_path,monkeypatch)
    journal=h.SuiteJournal(c,context,84)
    assert journal.max_records==508 and journal.max_records*8192+16384==4177920
    with pytest.raises(ValueError,match='max84'):h.SuiteJournal(c,context,85)
    _p30_preimport_current_cases(_P30_CURRENT,tmp_path)


def test_execution_checkpoints_reached_actual_parent_ENTRY_tamper_and_bounds(tmp_path,monkeypatch):
    isolate_compact_boundary(monkeypatch)
    value=h.execution_budget_amendment()
    contract={'pass_id':'e360-generous-p8','mode':'generated','role':'fixture',
        'revalidation_amendment':h.file_spec(h.EXECUTION_BUDGET_AMENDMENT),'limits':value['effective_roles']['fixture'],'source_sha256':h.own_sources()}
    contract,descriptor,receipt=finite_compact_contract(tmp_path,contract)
    actual_safe=h.safe
    monkeypatch.setattr(h,'safe',lambda name,root=h.ROOT:tmp_path/'contract.json' if name=='contract.json' else actual_safe(name,root))
    timer_=h.Clock(10.,contract['limits'])
    result={'contract':descriptor}
    now=[11.];monkeypatch.setattr(h.time,'perf_counter',lambda:now[0])
    with h.fixed_authority_scope(check=lambda:None) as scope:
        assert h.execution_budget_amendment()['checkpoint_policy']==h.compiled_checkpoint_policy()
        assert scope.values
        with h.descriptors_only_execution(),monkeypatch.context() as patch:
            assert h._AUTHORITY_SCOPE is None and not scope.values
            def forbidden(*args,**kwargs):raise AssertionError('passive checkpoint attempted authority IO')
            patch.setattr(h,'execution_budget_amendment',forbidden)
            patch.setattr(h,'fixed_reader',forbidden)
            patch.setattr(h,'literal_bytes',forbidden)
            patch.setattr(h,'read_spec',forbidden)
            for stage in value['checkpoint_policy']['stage_names']:
                h.checkpoint(contract,result,timer_,stage);now[0]+=1.
    result['elapsed_before_publication_s']=8.
    assert h.validate_execution_checkpoints(result,contract,receipt)['entry_perf_counter']==10.
    # A later authenticated verifier deadline must accept the earlier worker ENTRY.
    literal=h.safe(h.EXECUTION_BUDGET_AMENDMENT).read_bytes()
    literal_path=h.safe(h.EXECUTION_BUDGET_AMENDMENT,tmp_path)
    literal_path.parent.mkdir(parents=True,exist_ok=True);literal_path.write_bytes(literal)
    (tmp_path/'later-verifier.json').write_bytes(h.json_bytes({'role':'suite','limits':{
        'outer_s':120,'child_s':110,'internal_s':100,'science_s':90}}))
    later_entry=timer_.clock()
    verifier=h.operation_clock('later-verifier.json',later_entry,root=tmp_path,stage='internal')
    assert type(verifier) is h.OperationClock and verifier._admitted_root==tmp_path.resolve()
    assert verifier.timer.clock is verifier._admitted_clock and verifier.timer.entry!=receipt['entry_perf_counter']
    with monkeypatch.context() as patch:
        patch.setattr(h,'execution_budget_amendment',lambda root=h.ROOT,value=value:copy.deepcopy(value))
        with h.fixed_authority_scope(root=tmp_path,check=verifier) as boundary,h.metadata_invocation(tmp_path) as admitted:
            assert admitted and boundary.check is verifier
            assert h.validate_execution_checkpoints(result,contract,receipt,root=tmp_path)['entry_perf_counter']==10.
            wrong={**receipt,'entry_perf_counter':11.}
            with pytest.raises(ValueError,match='ENTRY'):h.validate_execution_checkpoints(result,contract,wrong,root=tmp_path)
            legacy={**receipt,'schema':'flylab.m3.e360.actual-pre-spawn.v1','authority':contract}
            with pytest.raises(ValueError):h.validate_execution_checkpoints(result,contract,legacy,root=tmp_path)
            assert verifier.timer.entry==later_entry and not boundary.failed
            assert not h._SUITE_DEADLINE
    changes=[]
    for field,item in (('entry_perf_counter',0.),('schema','wrong')):
        bad=copy.deepcopy(result);bad['execution_checkpoints'][field]=item;changes.append(bad)
    for item in (True,-1.,float('nan'),float('inf'),47.,0.):
        bad=copy.deepcopy(result);bad['execution_checkpoints']['stages'][-1]['elapsed_s']=item;changes.append(bad)
    for mutation in ('unreached','duplicate','arbitrary','malformed','oversize','reset'):
        bad=copy.deepcopy(result);rows=bad['execution_checkpoints']['stages']
        if mutation=='unreached':rows.pop()
        elif mutation=='duplicate':rows.append(copy.deepcopy(rows[-1]))
        elif mutation=='arbitrary':rows[0]['stage']='guessed_phase'
        elif mutation=='malformed':rows[0]=[]
        elif mutation=='oversize':bad['execution_checkpoints']['extra']='x'*16385
        else:bad['execution_checkpoints']['entry_perf_counter']=11.
        changes.append(bad)
    for bad in changes:
        with pytest.raises(ValueError):h.validate_execution_checkpoints(bad,contract,receipt)
    for field,item in (('revalidation_amendment',h.file_spec(h.REVALIDATION_AMENDMENT)),
                       ('limits',{**contract['limits'],'internal_s':100}),('pass_id','e360-generous-p7')):
        with pytest.raises(ValueError):h.validate_execution_checkpoints(result,{**contract,field:item},receipt)
    failure=copy.deepcopy(result);failure['execution_checkpoints']['stages']=failure['execution_checkpoints']['stages'][:2]
    failure.update(run_valid=False,effective_qualified=False,elapsed_before_publication_s=47.)
    assert len(h.validate_execution_checkpoints(failure,contract,receipt,successful=False)['stages'])==2
    failure['execution_checkpoints']['stages'][-1]['elapsed_s']=46.5
    assert h.validate_execution_checkpoints(failure,contract,receipt,successful=False)
    for field,item in (('run_valid',True),('effective_qualified',True),('elapsed_before_publication_s',46.)):
        with pytest.raises(ValueError):h.validate_execution_checkpoints({**failure,field:item},contract,receipt,successful=False)
    with pytest.raises(ValueError):h.validate_execution_checkpoints(failure,contract,receipt)


def test_execution_budget_closed_fixture_production_checkpoint_and_cap_substitution(monkeypatch):
    spec=json.loads(os.environ['E360_CURRENT_FIXTURE_EXIT'])
    closed=h.closed_exit(spec)
    contract=h.strict_json(h.read_spec(closed['contract']))
    assert contract['mode']=='generated' and contract['role']=='fixture'
    assert contract['revalidation_amendment']==h.prospective_authority_spec()
    assert contract['limits']==h.effective_policy(pass_id=contract['pass_id'])['effective_roles']['fixture']
    assert closed['source_sha256']==contract['source_sha256']
    assert all(contract['source_sha256'].get(name)==digest for name,digest in h.own_sources().items())
    value=h.strict_json(h.read_spec(closed['envelope']))
    assert value['run_valid'] and value['first_error'] is None and value['secondary_errors']==[]
    outer=h.strict_json(h.read_spec(closed['outer']))
    parent=h.strict_json(h.bounded_json_read(value['pre_spawn'],cap=262144))
    assert value['execution_checkpoints']['entry_perf_counter']==outer['entry_perf_counter']==parent['entry_perf_counter']
    assert len(value['execution_checkpoints']['stages'])==7
    actual=h.read_spec
    bounded=h.bounded_json_read
    for mutation in ('publisher_invalid','publisher_error','source_drift','bad_authority','cap_substitution','missing_checkpoints'):
        with monkeypatch.context() as patch:
            def changed(target,root=h.ROOT,*,mutation=mutation):
                raw=actual(target,root)
                if target==closed['envelope']:
                    current=h.strict_json(raw)
                    if mutation=='publisher_invalid':current['run_valid']=False
                    elif mutation=='publisher_error':current['first_error']={'message':'finite publisher failure'}
                    elif mutation=='source_drift':current['source_sha256']={}
                    elif mutation=='missing_checkpoints':current.pop('execution_checkpoints')
                    return h.json_bytes(current)
                if target==closed['contract'] and mutation in ('bad_authority','cap_substitution'):
                    current=h.strict_json(raw)
                    if mutation=='bad_authority':current['revalidation_amendment']={**current['revalidation_amendment'],'sha256':'0'*64}
                    else:current['limits']['internal_s']+=1
                    return h.json_bytes(current)
                return raw
            patch.setattr(h,'read_spec',changed)
            patch.setattr(h,'bounded_json_read',lambda target,root=h.ROOT,changed=changed,bounded=bounded,**kwargs:changed(target,root) if target==closed['contract'] else bounded(target,root,**kwargs))
            with pytest.raises(ValueError):h.closed_exit(spec)


def test_execution_budget_P8_manual_source_freshness_once_debit_no_P9():
    inherited=h.strict_json(h.read_spec(h.suite_recovery_amendment()['inherited81']))['ledger']
    passed=copy.deepcopy(inherited['e360_passes']['e360-generous-p8'])
    h.validate_p8_pass('e360-generous-p8',passed,inherited['debits'])
    for identity in ('e360-generous-p7','e360-generous-p9','e360-auto'):
        with pytest.raises(ValueError):h.validate_p8_pass(identity,passed,inherited['debits'])
    for field,item in (('modified',False),('predecessor_failed_cap_s',55),('source_sha256',inherited['e360_passes']['e360-generous-p7']['source_sha256']),('source_sha256',h.own_sources()),('concrete_fix',[])):
        with pytest.raises(ValueError):h.validate_p8_pass('e360-generous-p8',{**passed,field:item},inherited['debits'])
    failure=next(d for d in inherited['debits'] if d['descriptor']==passed['actual_failure'])
    with pytest.raises(ValueError):h.validate_p8_pass('e360-generous-p8',passed,inherited['debits']+[failure])


def test_ROOT_execution_budget_activation_and_historical_call_policy_production_fragments(tmp_path):
    import subprocess
    source=h.safe(h.BASE+'/parent-e360-root-account.ps1').read_text('utf8')
    functions=source[source.index('function Task-Path('):source.index('function Sum-Actual(')]
    call_policy=source[source.index('function Call-Policy('):source.index('function Prospective(')]
    activation_constructor=source[source.index('$executionBudgetActivation=@{'):source.index('$budgetActive=')]
    value=h.execution_budget_amendment();inherited=h.strict_json(h.read_spec(value['inherited73']))['ledger']
    payload={'authority':value,'state':inherited,'extension':h.resource_extension(),'descriptor':h.file_spec(h.EXECUTION_BUDGET_AMENDMENT),
        'historical_authorities':{str(number):h.file_spec(h.authority_path_for_pass('e360-generous-p'+str(number))) for number in range(3,8)}}
    data=tmp_path/'input.json';data.write_bytes(h.json_bytes(payload))
    script=tmp_path/'budget.ps1'
    script.write_text("param([string]$InputPath)\n$ErrorActionPreference='Stop'\n"+functions+call_policy+r'''
$p=[IO.File]::ReadAllText($InputPath)|ConvertFrom-Json -AsHashtable
$budget=$p.authority;$budgetSpec=$p.descriptor;$historicalExtension=$p.extension
$oldRevalidationSpec=$p.historical_authorities['3'];$adminRevalidationSpec=$p.historical_authorities['4']
$sourceRevalidationSpec=$p.historical_authorities['5'];$ruffRevalidationSpec=$p.historical_authorities['6']
$scopeRevalidationSpec=$p.historical_authorities['7']
$extensionSpec=$budget.resource_extension
'''+activation_constructor+r'''
function Check([bool]$ok,[string]$message){if(-not $ok){throw $message}}
$storedActivation=$executionBudgetActivation|ConvertTo-Json -Depth 100 -Compress|ConvertFrom-Json -AsHashtable
Check (Equal-Typed $storedActivation $executionBudgetActivation) 'actual activation constructor preserves exact types after JSON storage'
foreach($wrongZero in @([bool]$false,[double]0)){
    $badActivation=$storedActivation|ConvertTo-Json -Depth 100 -Compress|ConvertFrom-Json -AsHashtable
    $badActivation.new_generated_grant_s=$wrongZero
    Check (-not (Equal-Typed $badActivation $executionBudgetActivation)) 'activation zero bool or float substitution must reject'
}
Check (Execution-Budget-Activation-Allowed $p.state $true) 'exact73 should activate'
Check (-not (Execution-Budget-Activation-Allowed $p.state $false)) 'lock must reject'
foreach($key in @('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations')){
    $bad=$p.state|ConvertTo-Json -Depth 100 -Compress|ConvertFrom-Json -AsHashtable
    $bad[$key]=@{unknown=@{}}
    Check (-not (Execution-Budget-Activation-Allowed $bad $true)) 'ownership must reject'
}
foreach($id in 3,4,5,6,7,8){
    $descriptor=if($id -eq 8){$budgetSpec}else{$p.historical_authorities[[string]$id]}
    $request=@{pass_id="e360-generous-p$id";revalidation_amendment=$descriptor;call_role='scanner';mode='generated';action='scanner'}
    $policy=Call-Policy $request
    if($id -lt 8){Check ($policy.outer_s -eq 20 -and $policy.future_s -eq 240) 'historical scanner caps'}
    else{Check ($policy.outer_s -eq 15 -and $policy.future_s -eq 255) 'prospective scanner caps'}
}
foreach($wrong in @(@{pass_id='e360-generous-p7';revalidation_amendment=$budgetSpec},
                    @{pass_id='e360-generous-p8';revalidation_amendment=$scopeRevalidationSpec},
                    @{pass_id='e360-generous-p9';revalidation_amendment=$budgetSpec})){
    $wrong.call_role='scanner';$wrong.mode='generated';$wrong.action='scanner'
    $rejected=$false
    try{$null=Call-Policy $wrong}catch{$rejected=$true}
    Check $rejected 'historical/prospective authority substitution or P9 must reject'
}
$bad=$p.state|ConvertTo-Json -Depth 100 -Compress|ConvertFrom-Json -AsHashtable
$bad.e360_calls.Values|Select-Object -First 1|ForEach-Object{$_.state='begun'}
Check (-not (Execution-Budget-Activation-Allowed $bad $true)) 'uncharged must reject'
''',encoding='utf8')
    done=subprocess.run(['pwsh','-NoProfile','-File',str(script),str(data)],capture_output=True,text=True,
        timeout=8,check=False,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert done.returncode==0,done.stdout+'\n'+done.stderr


def test_supervise_P8_clock_construction_drift_cap_authority_and_expiry_close_before_Popen(tmp_path,monkeypatch):
    import time
    isolate_compact_boundary(monkeypatch)
    frozen=h.registration();amended=h.amendment();extension=h.resource_extension();budget=h.execution_budget_amendment()
    monkeypatch.setattr(h,'registration',lambda root=h.ROOT:frozen)
    monkeypatch.setattr(h,'amendment',lambda root=h.ROOT:amended)
    monkeypatch.setattr(h,'resource_extension',lambda root=h.ROOT:extension)
    monkeypatch.setattr(h,'execution_budget_amendment',lambda root=h.ROOT:budget)
    monkeypatch.setitem(sys.modules,'psutil',SimpleNamespace(Process=lambda pid:SimpleNamespace(pid=pid,create_time=lambda:456.)))
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append('Popen')
        raise AssertionError('inadmissible clock/source reached actual Popen')
    monkeypatch.setattr(h.subprocess,'Popen',forbidden)
    actual_pre_spawn=h.pre_spawn
    for mutation in ('cap_substitution','authority_substitution','contract_drift','expired','bootstrap_oversize'):
        root=tmp_path/mutation;root.mkdir()
        for name in (h.REGISTRATION,h.AMENDMENT,h.RESOURCE_EXTENSION,h.REVALIDATION_AMENDMENT,h.EXECUTION_BUDGET_AMENDMENT):
            target=h.safe(name,root);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(h.safe(name).read_bytes())
        attempt='e360-clock-'+mutation.replace('_','-');paths=h.names(attempt,'fixture')
        contract={'pass_id':'e360-generous-p8','mode':'generated','role':'fixture','attempt_id':attempt,'attempt':1,
            'paths':paths,'limits':copy.deepcopy(budget['effective_roles']['fixture']),
            'source_sha256':{},'original_calls':0,'retained_reserve_bytes':1<<20,
            'cumulative_before_s':317.1840156,'cumulative_limit_s':600,'stage_used_s':236.994,
            'stage_limit_s':600,'local_used_s':206.227,'local_limit_s':489,'reserved_s':55,
            'registration':h.file_spec(h.REGISTRATION,root),'amendment':h.file_spec(h.AMENDMENT,root),
            'resource_extension':h.file_spec(h.RESOURCE_EXTENSION,root),'revalidation_amendment':h.file_spec(h.EXECUTION_BUDGET_AMENDMENT,root)}
        if mutation=='cap_substitution':contract['limits']['internal_s']=100
        elif mutation=='authority_substitution':contract['revalidation_amendment']=h.file_spec(h.REVALIDATION_AMENDMENT,root)
        elif mutation=='bootstrap_oversize':contract['structural_padding']='x'*270000
        h.safe(paths['contract'],root).write_bytes(h.json_bytes(contract))
        if mutation=='bootstrap_oversize':
            assert 262144<h.safe(paths['contract'],root).stat().st_size<=524288
        with monkeypatch.context() as patch:
            if mutation=='contract_drift':
                def changed_pre_spawn(name,entry,*,root):
                    changed=h.strict_json(h.safe(name,root).read_bytes())
                    changed['limits']['science_s']=90
                    h.safe(name,root).write_bytes(h.json_bytes(changed))
                    return actual_pre_spawn(name,entry,root=root)
                patch.setattr(h,'pre_spawn',changed_pre_spawn)
            entry=time.perf_counter()-(56 if mutation=='expired' else 0)
            assert h.supervise(paths['contract'],entry=entry,root=root)==1
        outer=h.strict_json(h.safe(paths['outer'],root).read_bytes())
        assert outer['Popen_never_attempted'] and not outer['Popen_attempted'] and not outer['Popen_returned']
        assert outer['all_owned_ended'] and outer['reservation_released'] and outer['observed_pid_create_times']=={}
        assert outer['unknown_owned_identities']==outer['unresolved_matching_identities']=={}
        assert outer['known_identity_cleanup']==outer['missing_owned_creation_times']==[]
        assert outer['actual_full_return_pending'] and not outer['authority_admitted'] and outer['entry_perf_counter']==entry
        assert outer['unspawned_original_array_reads']==outer['unspawned_scientific_native_imports']==0
        first=outer['first_error'];assert first and first['stage']=='outer'
        if mutation=='bootstrap_oversize':
            assert first['type']=='ValueError' and first['message']=='bounded operation metadata'
            assert outer['pre_spawn'] is None and outer['envelope'] is None
            assert outer['authority_admitted'] is False and outer['actual_full_return_pending'] is True
            assert outer['all_owned_ended'] is True and outer['reservation_released'] is True
            assert not h.safe(paths['pre_spawn'],root).exists()
        elif mutation=='authority_substitution':assert 'exact pass clock authority' in first['message']
        elif mutation=='expired':
            assert first['type']=='TimeoutError' and 'outer deadline' in first['message']
            assert outer['failure_publication']['entry_perf_counter']==entry and not outer['failure_publication']['timely_success']
            assert outer['failure_publication']['closed_failed_evidence']
            failed=h.strict_json(h.read_spec(outer['failure_publication']['closed_failed_evidence'],root))
            assert failed['first_error']==first and outer['secondary_errors']
        else:assert 'operation exact immutable role caps' in first['message']
    assert calls==[]


def test_historical_P8_checkpoint_evidence_and_tampering_stay_P8():
    # Historical closed evidence is read as history, never admitted as current.
    spec={'path':h.BASE+'/parent-e346-e360-e360-generous-p8-f1-fixture-full-exit.json',
          'bytes':76557,'sha256':'afd76ae5b13d9d1f4f371efa552c0967a3ff4ecc639f914030cb9a702d3d15c3'}
    row=h.strict_json(h.read_spec(spec))
    contract=h.strict_json(h.read_spec(row['contract']))
    value=h.strict_json(h.read_spec(row['envelope']))
    receipt=h.strict_json(h.read_spec(value['pre_spawn']))
    assert contract['pass_id']=='e360-generous-p8' and row['qualified'] is True
    assert row['actual_outer_full_return_s']==26.753
    assert len(h.validate_historical_p8_checkpoints(spec,c=contract,value=value,receipt=receipt)['stages'])==7
    with pytest.raises(ValueError):h.validate_execution_checkpoints(value,contract,receipt)
    for forged in ({**spec,'sha256':'0'*64},{**spec,'bytes':True},{**spec,'path':'other.json'}):
        with pytest.raises(ValueError):h.validate_historical_p8_checkpoints(forged)
    for mutation in ('clock_reset','unreached','nonmonotone','bad_authority','cap_substitution'):
        changed=copy.deepcopy(value);authority=copy.deepcopy(contract)
        if mutation=='clock_reset':changed['execution_checkpoints']['entry_perf_counter']+=1.
        elif mutation=='unreached':changed['execution_checkpoints']['stages'].pop()
        elif mutation=='nonmonotone':changed['execution_checkpoints']['stages'][-1]['elapsed_s']=0.
        elif mutation=='bad_authority':authority['revalidation_amendment']=h.file_spec(h.REVALIDATION_AMENDMENT)
        else:authority['limits']['internal_s']+=1
        with pytest.raises(ValueError):h.validate_historical_p8_checkpoints(spec,c=authority,value=changed,receipt=receipt)


    for field,item in (('pass_id','e360-generous-p9'),('source_sha256',{}),('inputs',{'forged':True})):
        with pytest.raises(ValueError):h.validate_historical_p8_checkpoints(spec,c={**contract,field:item})
    for field,item in (('authority',{}),('schema',h.CURRENT_PRESPAWN_SCHEMA),('entry_perf_counter',287385.)):
        with pytest.raises(ValueError):h.validate_historical_p8_checkpoints(spec,receipt={**receipt,field:item})
    # Rehashed frozen leaves still cannot replace compiled immutable history.
    real=h.bounded_json_read
    for leaf in ('contract','receipt','envelope','inherited81'):
        def substituted(pin,root=h.ROOT,*,leaf=leaf,**kwargs):
            raw=real(pin,root,**kwargs)
            if pin==h.HISTORICAL_P8_ANCHORS[leaf]:
                changed=h.strict_json(raw);changed['forged_rehashed_leaf']=True
                return h.json_bytes(changed)
            return raw
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(h,'bounded_json_read',substituted)
            with pytest.raises(ValueError):h.validate_historical_p8_checkpoints(spec,c=contract,value=value,receipt=receipt)

def metadata_test_boundary(tmp_path,monkeypatch):
    # Isolate the focused finite probe from the real enclosing suite context.
    monkeypatch.setattr(h,'_SUITE_DEADLINE',None)
    monkeypatch.setattr(h,'_AUTHORITY_SCOPE',None)
    monkeypatch.setattr(h,'_AUTHORITY_FENCE_STATE',None)
    path,raw,calls=finite_scope_authority(tmp_path,monkeypatch)
    (tmp_path/'stage.json').write_bytes(h.json_bytes({'role':'suite','limits':{
        'outer_s':120,'child_s':110,'internal_s':100,'science_s':90}}))
    check=h.operation_clock('stage.json',h.time.perf_counter(),root=tmp_path,stage='internal')
    return path,raw,calls,check


def test_metadata_invocations_fresh_live_own_source_results_and_cleanup(tmp_path,monkeypatch):
    for spec in (h.P27_ACTUAL_RUFF_RECORD_SPECS['stderr.txt'],h.P29_ACTUAL_RUFF_RECORD_SPECS['stderr.txt']):
        with h.fixed_authority_scope(check=lambda:None) as scope:
            assert h.historical_p12_stream_bytes(spec)==b'' and scope.depth==0
            assert scope.history_descriptors[h.os.path.normcase(str(h.safe(spec['path'])))]==spec
            scope.fence('both_actual_completed_empty_inputs')
    _,_,calls,check=metadata_test_boundary(tmp_path,monkeypatch)
    live=tmp_path/'live.json';live.write_bytes(h.json_bytes({'budget':1}))
    def finite(spec,*,root):
        first=h.registration(root);assert h.registration(root)==first
        value=h.strict_json(h.read_spec(h.file_spec('live.json',root),root))
        own=h.sha((root/'dependency.py').read_bytes())
        return {'literal':first,'live':value,'own':own,'source_sha256':{'dependency.py':own}}
    monkeypatch.setattr(h,'_closed_exit_uncached',finite)
    with h.fixed_authority_scope(root=tmp_path,check=check) as enclosing:
        h.registration(tmp_path)
        with h.descriptors_only_execution():
            first=h.closed_exit({},root=tmp_path)
            first['literal']['items'].append(9);first['live']['budget']=99
            live.write_bytes(h.json_bytes({'budget':2}))
            second=h.closed_exit({},root=tmp_path)
            assert second['literal']['items']==[1] and second['live']['budget']==2
            assert first['own']==second['own'] and len(calls)==3
            assert h._AUTHORITY_SCOPE is None and h._AUTHORITY_FENCE_STATE is enclosing
            (tmp_path/'dependency.py').write_bytes(b'VALUE=2\n')
            with pytest.raises(ValueError,match='fence'):h.closed_exit({},root=tmp_path)
            assert enclosing.failed and not enclosing.values
        assert h._AUTHORITY_SCOPE is enclosing
    assert h._AUTHORITY_SCOPE is None and not enclosing.descriptors


def test_metadata_inherited_poison_restoration_primary_and_nested_ownership(tmp_path,monkeypatch):
    path,raw,_,check=metadata_test_boundary(tmp_path,monkeypatch)
    def rejected(spec,*,root):
        h.registration(root)
        if spec.get('drift'):path.write_bytes(raw+b' ')
        raise ValueError('primary finite validation failure')
    monkeypatch.setattr(h,'_closed_exit_uncached',rejected)
    with h.fixed_authority_scope(root=tmp_path,check=check) as enclosing:
        h.registration(tmp_path)
        with h.descriptors_only_execution():
            with pytest.raises(ValueError,match='primary finite') as clean:h.closed_exit({},root=tmp_path)
            assert not getattr(clean.value,'__notes__',[]) and not enclosing.failed
            assert h._AUTHORITY_SCOPE is None and h._AUTHORITY_FENCE_STATE is enclosing
            with pytest.raises(ValueError,match='primary finite') as drift:h.closed_exit({'drift':True},root=tmp_path)
            assert enclosing.failed and drift.value.__notes__
            path.write_bytes(raw)
            with pytest.raises(ValueError,match='poisoned'):h.closed_exit({},root=tmp_path)
        assert h._AUTHORITY_SCOPE is enclosing
    path.write_bytes(raw)
    with h.fixed_authority_scope(root=tmp_path,check=check) as enclosing:
        monkeypatch.setattr(h,'_closed_exit_uncached',lambda spec,root:{
            **h.registration(root),'source_sha256':{'dependency.py':h.sha((root/'dependency.py').read_bytes())}})
        value=h.closed_exit({},root=tmp_path)
        assert value['items']==[1] and h._AUTHORITY_SCOPE is enclosing and enclosing.values
    assert not enclosing.values and not enclosing.descriptors


def test_metadata_noop_untrusted_deadline_uses_old_uncached_path(tmp_path,monkeypatch):
    _,_,calls,_=metadata_test_boundary(tmp_path,monkeypatch)
    monkeypatch.setattr(h,'ADMIN_CHECK',lambda:None)
    def finite(spec,*,root):
        assert h._AUTHORITY_SCOPE is None
        h.registration(root);return h.registration(root)
    monkeypatch.setattr(h,'_closed_exit_uncached',finite)
    assert h.closed_exit({},root=tmp_path)['items']==[1]
    assert len(calls)==2 and h._AUTHORITY_SCOPE is None
    untrusted=h.FixedAuthorityScope(tmp_path,lambda:None)
    monkeypatch.setattr(h,'_AUTHORITY_FENCE_STATE',untrusted)
    h.closed_exit({},root=tmp_path)
    assert len(calls)==4 and not untrusted.values and not untrusted.failed
    with pytest.raises(ValueError,match='authenticated descriptors-only'):
        h.SuiteDeadline({},None,None,None,tmp_path)


def test_metadata_closure_sources_local_completion_blocks_result_after_drift(tmp_path,monkeypatch):
    path,raw,_,check=metadata_test_boundary(tmp_path,monkeypatch)
    def closure(reg,report,root,*,closure_report):
        h.registration(root)
        value={'dependency.py':h.sha((root/'dependency.py').read_bytes())}
        if report['drift']:path.write_bytes(raw+b' ')
        return value
    monkeypatch.setattr(h,'_closure_sources_uncached',closure)
    with h.fixed_authority_scope(root=tmp_path,check=check) as enclosing, h.descriptors_only_execution():
        assert h.closure_sources({}, {'drift':False},tmp_path)=={
            'dependency.py':h.sha((tmp_path/'dependency.py').read_bytes())}
        with pytest.raises(ValueError,match='fence'):h.closure_sources({}, {'drift':True},tmp_path)
        # No inherited descriptor was consumed before this local operation.
        assert not enclosing.failed and h._AUTHORITY_SCOPE is None
    path.write_bytes(raw)


def journal_test_context(tmp_path,monkeypatch):
    _,_,_,check=metadata_test_boundary(tmp_path,monkeypatch)
    boundary=h.FixedAuthorityScope(tmp_path,check)
    return SimpleNamespace(root=tmp_path,boundary=boundary,entry=check.timer.entry,
        clock=check.timer.clock,limits={'science_s':90,'internal_s':100,'output_cap_bytes':16<<20}),{
        'paths':{'output':'observations','contract':'contract.json'},
        'limits':{'output_cap_bytes':16<<20},'retained_reserve_bytes':1<<20}


def test_journal_actual_clock_writer_isolation_reached_events_and_utf8_bounds(tmp_path,monkeypatch):
    context,c=journal_test_context(tmp_path,monkeypatch)
    journal=h.SuiteJournal(c,context,1)
    def forbidden(*args,**kwargs):raise AssertionError('synthetic observer primitive')
    monkeypatch.setattr(h.time,'perf_counter',lambda:0.)
    monkeypatch.setattr(h,'safe',forbidden);monkeypatch.setattr(h,'read_spec',forbidden)
    monkeypatch.setattr(h.lib,'durable',forbidden)
    journal.record('phase_start',node='source.py::test_one',phase='call')
    journal.record('phase_report',node='source.py::test_one',phase='call',outcome='failed',
        duration=.25,longrepr='한'*3000)
    journal.record('runner_final',outcome='returned',closing=True)
    assert not journal.failed and journal.sequence==3
    rows=[h.strict_json(p.read_bytes()) for p in sorted((tmp_path/'observations').glob('*.json'))]
    assert [row['event'] for row in rows]==['phase_start','phase_report','runner_final']
    assert all(row['actual_parent_ENTRY']==context.entry and row['elapsed_s']>=0 for row in rows)
    assert len(rows[1]['failed_longrepr'].encode('utf8'))<=4096
    assert rows[1]['pytest_reported_duration_s']==.25 and rows[0]['pytest_reported_duration_s'] is None
    assert all(p.stat().st_size<=8192 for p in (tmp_path/'observations').glob('*.json'))
    assert all(row['diagnostic_only'] and not row['CASE_receipt'] and not row['qualification'] for row in rows)


def test_journal_no_overwrite_retry_partial_completion_or_deadline_extension(tmp_path,monkeypatch):
    context,c=journal_test_context(tmp_path,monkeypatch)
    journal=h.SuiteJournal(c,context,1)
    folder=tmp_path/'observations';folder.mkdir()
    occupied=folder/'pytest-journal-0000.json';occupied.write_bytes(b'partial')
    journal.record('runner_start');journal.record('runner_final',closing=True)
    assert journal.failed and journal.sequence==0 and len(journal.errors)==1
    assert occupied.read_bytes()==b'partial' and len(list(folder.glob('*.json')))==1
    context.clock=lambda:context.entry+101.
    c['paths']['output']='late'
    expired=h.SuiteJournal(c,context,1)
    expired.record('runner_final',closing=True)
    assert expired.failed and not (tmp_path/'late').exists()
    with pytest.raises(ValueError,match='aggregate bound'):h.SuiteJournal(c,context,86)


def test_journal_authentic_poison_descriptor_drift_and_reserved_final(tmp_path,monkeypatch):
    context,c=journal_test_context(tmp_path,monkeypatch)
    path=tmp_path/'boundary.json';path.write_bytes(b'{}')
    spec=h.file_spec('boundary.json',tmp_path)
    context.boundary.depth=1;context.boundary.consumed(spec,tmp_path);context.boundary.depth=0
    _p30_copy_history_table(h,tmp_path)
    history_spec=h.P25_ACTUAL_RUFF_RECORD_SPECS['stdout.txt']
    history_path=_p30_history_leaf(h,tmp_path,history_spec)
    context.boundary._consumed(history_spec,tmp_path,owner='historical')
    c['paths']['output']='history-observations'
    history_journal=h.SuiteJournal(c,context,1)
    history_path.write_bytes(b'late')
    history_journal.record('phase_start',node='test_one',phase='setup')
    assert history_journal.failed and context.boundary.failed and context.boundary.history_descriptors
    assert not (tmp_path/'history-observations').exists() and path.read_bytes()==b'{}'
    _p30_history_leaf(h,tmp_path,history_spec)
    context.boundary.failed=False;c['paths']['output']='observations'
    journal=h.SuiteJournal(c,context,1)
    path.write_bytes(b'{ }')
    journal.record('phase_start',node='test_one',phase='setup')
    assert journal.failed and not (tmp_path/'observations').exists()
    path.write_bytes(b'{}');c['paths']['output']='poisoned'
    journal=h.SuiteJournal(c,context,1);context.boundary.poison()
    journal.record('runner_final',closing=True)
    assert journal.failed and not (tmp_path/'poisoned').exists()
    _p30_history_leaf(h,tmp_path,history_spec)
    context.boundary.failed=False;c['paths']['output']='bounded'
    journal=h.SuiteJournal(c,context,1)
    for _ in range(9):journal.record('phase_start',node='test_one',phase='setup')
    assert journal.sequence==9 and not journal.failed
    journal.record('runner_final',closing=True)
    assert journal.sequence==10 and journal.total<=4<<20 and not journal.failed
    journal.record('runner_final',closing=True)
    assert journal.failed and len(list((tmp_path/'bounded').glob('*.json')))==10


def finite_production_runner(tmp_path,monkeypatch):
    """Production runner/ReceiptPlugin/CASE control flow with finite admission seams."""
    names=h.test_functions()
    # The real paid script aliases h to __main__; this finite runner seam uses
    # its already-bound canonical identity without reseeding an imported module.
    monkeypatch.setattr(h,'__name__','tools.verify_c_banc_m3_source_closure_diagnostic')
    for name in ('_SUITE_DEADLINE','_AUTHORITY_SCOPE','_AUTHORITY_FENCE_STATE'):
        monkeypatch.setattr(h,name,None)
    sources={}
    for index in range(9):
        name='finite.py' if index==0 else f'finite{index}.py'
        raw=f'FINITE={index}\n'.encode();(tmp_path/name).write_bytes(raw);sources[name]=h.sha(raw)
    suite_authority=h.suite_recovery_amendment()
    suite_limits=copy.deepcopy(suite_authority['effective_roles']['suite'])
    suite_literal=h.safe(h.SUITE_RECOVERY_AMENDMENT).read_bytes()
    for name in (h.REGISTRATION,h.AMENDMENT,h.RESOURCE_EXTENSION,h.SUITE_RECOVERY_AMENDMENT):
        path=h.safe(name,tmp_path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(suite_literal if name==h.SUITE_RECOVERY_AMENDMENT else b'{}')
    report={'finite_source_test':True}
    closure=tmp_path/'closure.json';closure.write_bytes(h.json_bytes(report))
    closure_spec=h.file_spec('closure.json',tmp_path)
    fixture=tmp_path/'fixture.json';fixture.write_bytes(b'{}')
    ini=tmp_path/'empty.ini';ini.write_bytes(b'[pytest]\n')
    owned=h.safe(h.OWNED[1],tmp_path);owned.parent.mkdir(parents=True,exist_ok=True);owned.write_bytes(b'# finite source seam\n')
    receipt_name=h.BASE+'/parent-e360-finite-case-receipt.json'
    case={'schema':'flylab.m3.e360.current-pytest-case.v1','source_sha256':sources,
        'expected_functions':names,'receipt':receipt_name,'pytest_ini':'empty.ini',
        'fixture_exit':h.file_spec('fixture.json',tmp_path),'closure_report':closure_spec,
        'registration':h.file_spec(h.REGISTRATION,tmp_path),'amendment':h.file_spec(h.AMENDMENT,tmp_path),
        'resource_extension':h.file_spec(h.RESOURCE_EXTENSION,tmp_path),
        'revalidation_amendment':h.file_spec(h.SUITE_RECOVERY_AMENDMENT,tmp_path)}
    (tmp_path/'case.json').write_bytes(h.json_bytes(case))
    limits=suite_limits
    full_sources={**sources,'closure.json':closure_spec['sha256']}
    c,spec,_=finite_compact_contract(tmp_path,{'mode':'generated','role':'suite','pass_id':'e360-generous-p9',
        'paths':{'contract':'contract.json','pre_spawn':'parent-receipt.json','output':'output'},
        'limits':limits,'source_sha256':full_sources,'inputs':{'case_contract':h.file_spec('case.json',tmp_path)},
        'retained_reserve_bytes':1<<20,**{key:case[key] for key in
            ('registration','amendment','resource_extension','revalidation_amendment','closure_report')}})
    check=h.operation_clock('contract.json',h.time.perf_counter(),root=tmp_path,stage='internal')
    timer_=h.Clock(check.timer.entry,limits)
    (tmp_path/'parent-receipt.json').write_bytes(h.json_bytes(h.compact_receipt(c,spec,timer_.entry,123,456.)))
    receipt_spec=h.file_spec('parent-receipt.json',tmp_path)
    monkeypatch.setattr(h,'own_sources',lambda root=h.ROOT:copy.deepcopy(sources))
    monkeypatch.setattr(h,'test_functions',lambda root=h.ROOT:list(names))
    monkeypatch.setattr(h,'suite_recovery_amendment',lambda root=h.ROOT:copy.deepcopy(suite_authority))
    monkeypatch.setattr(h,'prospective_authority_spec',lambda root=h.ROOT:case['revalidation_amendment'])
    monkeypatch.setattr(h,'scan_current',lambda *args,**kwargs:copy.deepcopy(report))
    monkeypatch.setattr(h,'closed_exit',lambda *args,**kwargs:{'role':'fixture','mode':'generated',
        'closure_report':closure_spec,'source_sha256':copy.deepcopy(full_sources),'pass_id':c['pass_id'],
        **{key:c[key] for key in ('registration','amendment','resource_extension','revalidation_amendment')}})
    return c,spec,receipt_spec,timer_,check,case,owned


def finite_pytest_reports(plugin,names,owned,*,failed=False,deadline=None,omit_teardown=False):
    plugin.pytest_collection_finish(SimpleNamespace(items=[SimpleNamespace(path=owned,name=name) for name in names]))
    for name in names:
        item=SimpleNamespace(nodeid=str(owned)+'::'+name)
        for phase in ('setup','call','teardown'):
            hook=getattr(plugin,'pytest_runtest_'+phase)
            reached=hook(item,nextitem=None) if phase=='teardown' else hook(item)
            next(reached)
            with pytest.raises(StopIteration):next(reached)
            bad=failed and name==names[0] and phase=='call'
            if bad and deadline is not None:deadline()
            if omit_teardown and name==names[0] and phase=='teardown':continue
            plugin.pytest_runtest_logreport(SimpleNamespace(nodeid=item.nodeid,when=phase,
                outcome='failed' if bad else 'passed',failed=bad,duration=.01,
                longrepr='first actual finite test detail '+'한'*3000 if bad else ''))
        if failed:break


def test_suite_deadline_actual_clock_identity_substitution_and_finally_restoration(tmp_path,monkeypatch):
    c,spec,receipt,timer_,check,_,_=finite_production_runner(tmp_path,monkeypatch)
    assert timer_.clock is check._admitted_clock
    with h.fixed_authority_scope(root=tmp_path,check=check) as boundary, h.descriptors_only_execution():
        with pytest.raises(ValueError,match='finite context unwind'), h.suite_deadline(c,spec,receipt,timer_,tmp_path) as context:
            assert context.clock is check._admitted_clock and context.entry==timer_.entry
            assert context.boundary is boundary and h._AUTHORITY_FENCE_STATE is boundary
            raise ValueError('finite context unwind')
        assert h._SUITE_DEADLINE is None and h._AUTHORITY_FENCE_STATE is boundary
        frozen=h.Clock(timer_.entry,timer_.limits,clock=lambda:timer_.entry)
        with pytest.raises(ValueError,match='admitted clock callable'), h.suite_deadline(c,spec,receipt,frozen,tmp_path):pass
        assert h._SUITE_DEADLINE is None and not boundary.failed
        clock=check.timer.clock;check.timer.clock=frozen.clock
        try:
            with pytest.raises(ValueError,match='admitted clock callable'), h.suite_deadline(c,spec,receipt,timer_,tmp_path):pass
        finally:check.timer.clock=clock
        assert h._SUITE_DEADLINE is None and not boundary.failed


    # Same semantic production consumer; each negative owns a fresh genuine boundary.
    for mutation in ('schema','extra','descriptor','full_contract','same_size_contract','same_size_receipt','ENTRY','expiry'):
        with monkeypatch.context() as patch:
            root=tmp_path/mutation;root.mkdir()
            current,pin,parent,clock_,admitted,_,_=finite_production_runner(root,patch)
            parent_path=root/'parent-receipt.json';contract_path=root/'contract.json'
            if mutation in ('schema','extra','ENTRY'):
                changed=h.strict_json(parent_path.read_bytes())
                if mutation=='schema':changed['schema']='flylab.m3.e360.actual-pre-spawn.v1'
                elif mutation=='extra':changed['authority']=current
                else:changed['entry_perf_counter']+=1.
                parent_path.write_bytes(h.json_bytes(changed));parent=h.file_spec('parent-receipt.json',root)
            elif mutation=='descriptor':parent={**parent,'sha256':'0'*64}
            elif mutation=='full_contract':
                changed={**current,'inputs':{'forged':True}};contract_path.write_bytes(h.json_bytes(changed))
                pin=h.file_spec('contract.json',root)
                stored=h.strict_json(parent_path.read_bytes());stored['contract']=pin
                parent_path.write_bytes(h.json_bytes(stored));parent=h.file_spec('parent-receipt.json',root)
            with h.fixed_authority_scope(root=root,check=admitted) as genuine, h.descriptors_only_execution():
                if mutation in ('same_size_contract','same_size_receipt','expiry'):
                    with h.suite_deadline(current,pin,parent,clock_,root) as context:
                        assert len(genuine.descriptors)==2 and not genuine.values
                        if mutation=='expiry':
                            late=lambda context=context:context.entry+context.limits['science_s']+1.
                            context.clock=late
                            with pytest.raises(TimeoutError):context.check('science')
                        else:
                            target=contract_path if mutation=='same_size_contract' else parent_path
                            original=target.read_bytes();changed=original.replace(b'e360-finite',b'e360-forged')
                            assert changed!=original and len(changed)==len(original);target.write_bytes(changed)
                            with pytest.raises(ValueError,match='fence'):genuine.fence('suite_deadline_after_initial_read')
                else:
                    with pytest.raises(ValueError),h.suite_deadline(current,pin,parent,clock_,root):pass
                assert h._SUITE_DEADLINE is None

def test_production_runner_first_failed_report_later_runner_deadline_secondary_and_context(tmp_path,monkeypatch):
    for kind in ('runner','deadline','returned_failure','invalid_receipt'):
        with monkeypatch.context() as patch:
            root=tmp_path/kind;root.mkdir()
            c,spec,receipt,timer_,check,case,owned=finite_production_runner(root,patch)
            later=TimeoutError('finite later deadline') if kind=='deadline' else RuntimeError('finite later runner')
            seeded=h.error_record(ValueError('existing secondary publication'),'existing-publication')
            later.publication_errors=[seeded]
            def throws(*args,later=later,**kwargs):raise later
            def main(args,plugins,*,case=case,owned=owned,kind=kind,later=later,patch=patch,throws=throws):
                plugin=plugins[0]
                finite_pytest_reports(plugin,case['expected_functions'],owned,failed=kind!='invalid_receipt',
                    deadline=(lambda:patch.setattr(h._SUITE_DEADLINE,'check',throws)) if kind=='deadline' else None,
                    omit_teardown=kind=='invalid_receipt')
                if kind=='returned_failure':return 1
                if kind=='invalid_receipt':return 0
                raise later
            patch.setitem(sys.modules,'pytest',SimpleNamespace(main=main,hookimpl=lambda **kwargs:lambda fn:fn))
            with h.fixed_authority_scope(root=root,check=check) as boundary, h.descriptors_only_execution():
                if kind=='invalid_receipt':
                    assert h.execute_current_pytest('case.json',c=c,spec=spec,receipt_spec=receipt,timer=timer_,root=root)==2
                else:
                    with pytest.raises(RuntimeError,match='actual first failed pytest report') as observed:
                        h.execute_current_pytest('case.json',c=c,spec=spec,receipt_spec=receipt,timer=timer_,root=root)
                    assert 'first actual finite test detail' in str(observed.value)
                    assert len(str(observed.value).encode('utf8'))<6144
                    if kind=='returned_failure':
                        assert any(row['stage']=='pytest-returned-failure' and 'nonzero exit code 1' in row['message']
                            for row in observed.value.publication_errors)
                    else:
                        assert any(row['message']==str(later) for row in observed.value.publication_errors)
                        assert seeded in observed.value.publication_errors
                assert h._SUITE_DEADLINE is None and h._AUTHORITY_FENCE_STATE is boundary
            if kind in ('returned_failure','invalid_receipt'):
                saved=h.strict_json(h.safe(case['receipt'],root).read_bytes())
                assert saved['exit_code']==(1 if kind=='returned_failure' else 2)
                assert saved['complete_current_cases'] is False
                if kind=='invalid_receipt':assert 'teardown' not in saved['phases'][case['expected_functions'][0]]
                with pytest.raises(ValueError):h.validate_case_receipt(saved,root=root)
            else:assert not h.safe(case['receipt'],root).exists()
            assert not list(root.glob('*qualification*'))


def test_production_runner_journal_failure_all_phases_CASE_unqualified_child_and_restore(tmp_path,monkeypatch):
    c,spec,receipt,timer_,check,case,owned=finite_production_runner(tmp_path,monkeypatch)
    # The actual affected P9 branch sees a full map distinct from the exact own9 CASE.
    assert c['pass_id']=='e360-generous-p9' and len(case['source_sha256'])==9 and len(c['source_sha256'])>9
    real_closed=h.closed_exit
    for mutation in ('missing','unrelated','stale','pass','role','direct','report'):
        with monkeypatch.context() as patch:
            fixture=real_closed(case['fixture_exit'])
            if mutation=='missing':fixture['source_sha256'].pop('closure.json')
            elif mutation=='unrelated':fixture['source_sha256']['unrelated.py']='a'*64
            elif mutation=='stale':fixture['source_sha256']['closure.json']='0'*64
            elif mutation=='pass':fixture['pass_id']='e360-generous-p10'
            elif mutation=='role':fixture['role']='suite'
            elif mutation=='direct':fixture['registration']={}
            else:fixture['closure_report']={}
            patch.setattr(h,'closed_exit',lambda *args,fixture=fixture,**kwargs:copy.deepcopy(fixture))
            with pytest.raises(ValueError):h._execute_current_pytest('case.json',root=tmp_path,c=c)
    # The same current runner must reject P14 same-pass/full-map substitutions before pytest.
    c14={**c,'pass_id':'e360-generous-p14'}
    fixture14={**real_closed(case['fixture_exit']),'pass_id':'e360-generous-p14'}
    for mutation in ('missing','unrelated','stale','pass','direct','report'):
        with monkeypatch.context() as patch:
            fixture=copy.deepcopy(fixture14)
            if mutation=='missing':fixture['source_sha256'].pop('closure.json')
            elif mutation=='unrelated':fixture['source_sha256']['unrelated.py']='a'*64
            elif mutation=='stale':fixture['source_sha256']['closure.json']='0'*64
            elif mutation=='pass':fixture['pass_id']='e360-generous-p13'
            elif mutation=='direct':fixture['registration']={}
            else:fixture['closure_report']={}
            patch.setattr(h,'closed_exit',lambda *args,fixture=fixture,**kwargs:copy.deepcopy(fixture))
            with pytest.raises(ValueError):h._execute_current_pytest('case.json',root=tmp_path,c=c14)
    real_writer=h.lib.durable
    def journal_failure(path,payload,**kwargs):
        if path.name.startswith('pytest-journal-'):raise OSError('finite observer durable failure')
        return real_writer(path,payload,**kwargs)
    monkeypatch.setattr(h.lib,'durable',journal_failure)
    def main(args,plugins):
        finite_pytest_reports(plugins[0],case['expected_functions'],owned)
        return 0
    monkeypatch.setitem(sys.modules,'pytest',SimpleNamespace(main=main,hookimpl=lambda **kwargs:lambda fn:fn))
    published=[]
    monkeypatch.setattr(h,'envelope',lambda *args:{'run_valid':False,'first_error':None,'secondary_errors':[]})
    monkeypatch.setattr(h,'CPUOnly',h.contextlib.nullcontext)
    monkeypatch.setattr(h,'actual_imports',lambda *args,**kwargs:None)
    def child_publication(c,name,value,timer,**kwargs):
        published.append((name,copy.deepcopy(value)))
    monkeypatch.setattr(h,'put',child_publication)  # Finite publication seam; no authority or Q grant.
    with h.fixed_authority_scope(root=tmp_path,check=check) as boundary, h.descriptors_only_execution():
        assert h._child(c,spec,receipt,{},timer_,root=tmp_path)==1
        assert h._SUITE_DEADLINE is None and h._AUTHORITY_FENCE_STATE is boundary
    saved=h.strict_json(h.safe(case['receipt'],tmp_path).read_bytes())
    assert saved['exit_code']!=0 and saved['complete_current_cases'] is False
    assert saved['collected_functions']==case['expected_functions']
    assert all(set(row)=={'setup','call','teardown'} and set(row.values())=={'passed'} for row in saved['phases'].values())
    with pytest.raises(ValueError):h.validate_case_receipt(saved,root=tmp_path)
    assert [name for name,_ in published]==['envelope.json','revocation.json']
    assert all(not row['run_valid'] and row['revoked'] and not row['effective_qualified'] for _,row in published)
    assert any(error['message']=='finite observer durable failure' for error in published[0][1]['secondary_errors'])
    assert not list(tmp_path.glob('*qualification*'))


def test_journal_two_actual_fences_surround_closed_readback_writer_and_retention(tmp_path,monkeypatch):
    context,c=journal_test_context(tmp_path,monkeypatch)
    boundary_file=tmp_path/'boundary.json';boundary_file.write_bytes(b'{}')
    spec=h.file_spec('boundary.json',tmp_path)
    context.boundary.depth=1;context.boundary.consumed(spec,tmp_path);context.boundary.depth=0
    events=[];real_sha=h.sha;real_writer=h.lib.durable;real_retained=h.lib.retained
    def digest(raw):events.append('fresh-descriptor-fence');return real_sha(raw)
    def writer(*args,**kwargs):
        events.append('writer-enter');result=real_writer(*args,**kwargs)
        events.append('writer-fsync-close-readback-returned');return result
    def retained(*args,**kwargs):events.append('retention');return real_retained(*args,**kwargs)
    monkeypatch.setattr(h,'sha',digest);monkeypatch.setattr(h.lib,'durable',writer)
    monkeypatch.setattr(h.lib,'retained',retained)
    journal=h.SuiteJournal(c,context,1)
    journal.record('phase_start',node='source.py::test_one',phase='setup')
    assert not journal.failed and journal.sequence==1
    assert events==['retention','fresh-descriptor-fence','writer-enter',
        'writer-fsync-close-readback-returned','fresh-descriptor-fence','retention']
    assert (tmp_path/'observations'/'pytest-journal-0000.json').exists()


def test_script_canonical_import_bridge_actual_metadata_context_conflict_tamper_and_restore(tmp_path,monkeypatch):
    source=h.safe(h.OWNED[0]).read_bytes()
    c,spec,receipt,timer_,check,_,_=finite_production_runner(tmp_path,monkeypatch)
    own=h.safe(h.OWNED[0],tmp_path);own.parent.mkdir(parents=True,exist_ok=True);own.write_bytes(source)
    c['source_sha256'][h.OWNED[0]]=h.sha(source)
    (tmp_path/'contract.json').write_bytes(h.json_bytes(c));spec=h.file_spec('contract.json',tmp_path)
    (tmp_path/'parent-receipt.json').write_bytes(h.json_bytes(h.compact_receipt(c,spec,timer_.entry,123,456.)))
    receipt=h.file_spec('parent-receipt.json',tmp_path)
    package=sys.modules['tools'];canonical='tools.verify_c_banc_m3_source_closure_diagnostic'
    member=canonical.split('.')[-1]
    previous=type(h)(canonical)
    previous.__file__=str(own);previous.ROOT=tmp_path
    monkeypatch.setitem(sys.modules,canonical,previous)
    monkeypatch.setattr(package,member,previous,raising=False)
    monkeypatch.setattr(package,'__path__',[str(tmp_path/'tools')])
    monkeypatch.setitem(sys.modules,'__main__',h)
    monkeypatch.setattr(h,'__name__','__main__')
    monkeypatch.setattr(h,'__file__',str(own));monkeypatch.setattr(h,'ROOT',tmp_path)
    with h.fixed_authority_scope(root=tmp_path,check=check) as boundary, h.descriptors_only_execution():
        with h.suite_deadline(c,spec,receipt,timer_,tmp_path) as context:
            with h.canonical_suite_module(c,context,tmp_path):
                from tools import (
                    verify_c_banc_m3_source_closure_diagnostic as imported,
                )
                assert imported is h and imported._SUITE_DEADLINE is context
                assert sys.modules[canonical] is sys.modules['__main__'] is h
                with imported.metadata_invocation(tmp_path) as admitted:
                    assert admitted and imported._AUTHORITY_SCOPE is not None
                    assert imported._AUTHORITY_FENCE_STATE is boundary
                    imported.file_spec('finite.py',tmp_path)
                assert imported._AUTHORITY_SCOPE is None and not boundary.failed
            assert sys.modules[canonical] is previous and package.__dict__[member] is previous
            with pytest.raises(ValueError,match='late module identity substitution'), h.canonical_suite_module(c,context,tmp_path):
                sys.modules[canonical]=previous
            assert sys.modules[canonical] is previous and package.__dict__[member] is previous
            assert h._SUITE_DEADLINE is context and h._AUTHORITY_FENCE_STATE is boundary
            with pytest.raises(RuntimeError,match='earliest bridge test error') as observed, h.canonical_suite_module(c,context,tmp_path):
                package.__dict__[member]=previous
                raise RuntimeError('earliest bridge test error')
            assert observed.value.__notes__ and sys.modules[canonical] is previous
            assert package.__dict__[member] is previous
            previous.ROOT=tmp_path/'wrong'
            with pytest.raises(ValueError,match='conflicting canonical'), h.canonical_suite_module(c,context,tmp_path):pass
            assert sys.modules[canonical] is previous and package.__dict__[member] is previous
            previous.ROOT=tmp_path
            previous._SUITE_DEADLINE=object()
            with pytest.raises(ValueError,match='active context'), h.canonical_suite_module(c,context,tmp_path):pass
            assert sys.modules[canonical] is previous and package.__dict__[member] is previous
            previous._SUITE_DEADLINE=None
            monkeypatch.delitem(sys.modules,canonical)
            monkeypatch.delattr(package,member)
            with h.canonical_suite_module(c,context,tmp_path):
                from tools import (
                    verify_c_banc_m3_source_closure_diagnostic as imported_missing,
                )
                assert imported_missing is h and imported_missing._SUITE_DEADLINE is context
            assert canonical not in sys.modules and member not in package.__dict__
        assert h._SUITE_DEADLINE is None and h._AUTHORITY_FENCE_STATE is boundary
