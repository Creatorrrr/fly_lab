"""E360 CPU saved-prefix diagnostic; no body or neural execution permission.

Current delegated P30 adds its exact pair2/grant0 after successful paid P29 closed109; completed P19 and earlier versions retain their frozen provenance.
Actual historical grants include P9 grant800 and P19 grant300, preserving shared1700/local1589 with no new P30 grant or reset.
Current P30 admits one reviewed generated attempt after successful P29 Ruff and a separate unpaid Native scanner Source failure; original body/neural work requires its separate downstream contract. General human autonomy removes per-attempt approval pauses.
Historical descriptions below retain their as-executed P2-P9 meaning.

ROOT sequence (all actual full returns, including failures, are separately charged):
 Preserve historical v2 grant and all p3 through p8 activations. Account recover-suite
 (alias activate-suite) once from exact inherited81 with a distinct +800 grant,
 current shared1400/local1289 and ownership0;
 callbegin REQUEST before
 every external dispatch, exact nine-call pass order. Scanner reads its admitted
 request from E360_CALL_REQUEST. --scan fresh closure, then build(role,attempt_id,
 closure_report=descriptor,adapter_qualification=Q,inputs={case_contract:spec}
 for suite only). Save proposal, debit build helper, begin live reservation,
 worker callbegin, finalize complete ledger_debits/ledger_semantics snapshots,
 --supervise CONTRACT. Store actual return; seal_full_return(CONTRACT,spec)
 writes nothing. Account actual durably records undebited pending before seal;
 debit the actual seal helper, then worker (or unqualified fallback). Release only when
 all_owned_ended/reservation_released and shared lock absent; debit seal helper.
 Prospective P8 fixture55/50/46/42; historical fixture30/24/20/16 remains exact.
 P9 suite600/560/540/500; historical suite120/110/100/90 and original12/11/10/9 stay exact.
 Standalone fixture runs production synthetic reader/publisher first. CASE
 exact schema/current-source/function/receipt/empty-ini/fixture-exit/closure
 descriptor precedes whole current suite. Q links closure + actual closed
 fixture/suite + complete three-phase receipt. Original requires this NEW Q.
 No original payload is hashed/read by scanner/build/parent qualification.
 Old registration and literal redistribution amendment are separate pins in
 contracts/receipts/envelopes/CASE/Q/full exits, alongside literal v2 resource_extension.
 Prospective direct revalidation_amendment pins approved P9 suite recovery;
 ten fixed readers retain all old literals including P7/P8 historical authority.
 index59 and old p3 reader/event retain their literal6282B historical authority.
 CASE has eleven keys; Q fourteen, including qualification_call_id. Full Q content
 is validated before publication; ordinary Q validation requires its timely exit0
 once-debited helper and identical final Q descriptor/internal completion.
 All81 costs, p1 through p8 calls and offset42 are preserved. P9 scanner20/18 reserves
 fixture55+suite600+administration120; fixture future95 and suite50. Whole nine-call reserve798.
 Manual P9 has eight modified passes maximum, no P10, auto0, additional grant800. Generated scope
 ends after Q actual debit or earlier closed failure; original needs subsequent human approval.
 Historical no-next-pass literals remain immutable.
 Reviewed old snapshots never reverse-pin final own9.
 Python never rewrites/reset the live ledger. Freeze all four sources
 before the new scan, fixture, complete current suite and Q. Array-read counts
 on successful originals mean selected decoded NPY arrays10 in NPZ archives3.
"""
# ruff: noqa: BLE001 -- preserve scientific, ownership and publication errors in closed failure records.
from __future__ import annotations

import time

ENTRY=time.perf_counter()
import argparse
import ast
import collections
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
_SCANNER_BOOTSTRAP_POLICY=None
# This fragment is inserted before project imports in both actual entrypoints.
# No admitted state or success is retained between operations.
EXECUTION_ADMISSION_PATH='verification/banc-natural-gait-m3-20260919/parent-execution-current-admission.json'
EXECUTION_RULE_SPEC={"path":"verification/banc-natural-gait-m3-20260919/parent-execution-contract-separation-sol-FIRST-fixed-rules.json","bytes":2344,"sha256":"9b8a3e011a0245c94e9863b62bebba31381be876542bfddd9e8a863b736ff9aa"}
EXECUTION_ANCHOR_SPEC={'path':'verification/banc-natural-gait-m3-20260919/parent-e360-p30-after-scanner-closed-ledger111.json','bytes':221187,'sha256':'ca3c71875072a3a1e94ca15383b902c3911c32e108deea6de8dab86ae19078dc'}

EXECUTION_HISTORY_SPEC={"path":"verification/banc-natural-gait-m3-20260919/parent-execution-contract-separation-sol-FIRST-history-table.json","bytes":220211,"sha256":"b7f4412f93f1c21625edf3d2ec9723c74cf28fb3f202c19dfe3f46722c449b57"}
EXECUTION_DIRECT_INPUT_SPEC={"path":"verification/banc-natural-gait-m3-20260919/parent-execution-contract-separation-sol-FIRST-direct-physical-inputs.json","bytes":9040,"sha256":"fb77517a11a694a91aec93d222dc753d97cd306129794558552f9e0d30adbd03"}
EXECUTION_REVIEW_PINS={
    'actual_ROOT_FIRST':{'path':'docs/BANC_M3_EXECUTION_CONTRACT_SEPARATION_ROOT_FIRST_REVIEW_20261007.md','bytes':7807,'sha256':'22ea00d80a61993a63a27807fa6ff02c83a204022ddc4ce0ec21f475e1045e04'},
    'actual_ROOT_SECOND':{'path':'docs/BANC_M3_EXECUTION_CONTRACT_SEPARATION_ROOT_SECOND_REVIEW_20261007.md','bytes':2346,'sha256':'70173acac8e04a160726186f269b2e4cd89767fd155cae2b7079e19ee0bfc48f'},
    'actual_Sol_FIRST':{'path':'verification/banc-natural-gait-m3-20260919/parent-execution-contract-separation-sol-FIRST-handoff.json','bytes':15199,'sha256':'55bfcd3349d460c41a0b2dd10c2b63fe375a33b9897f9a03705a1c3e63c1e6da'},
    'actual_Sol_R1':{'path':'verification/banc-natural-gait-m3-20260919/parent-execution-contract-separation-sol-R1-handoff.json','bytes':17944,'sha256':'c1f172af1958f72503160f8724475b4e4a77aac0dc80f806fb2a6a6aa498c009'},
}
def execution_bootstrap(root,check=lambda:None):
    """Absent admission is legacy; any present invalid admission is fatal."""
    import stat
    root=Path(root).resolve()
    def pathname(name):
        if type(name) is not str or Path(name).is_absolute():raise ValueError('relative execution metadata')
        path=root/name
        if '..' in Path(name).parts or not path.resolve().is_relative_to(root):raise ValueError('execution root containment')
        cursor=path
        while cursor!=root:
            if cursor.exists() and (cursor.is_symlink() or getattr(cursor.lstat(),'st_file_attributes',0)&stat.FILE_ATTRIBUTE_REPARSE_POINT):raise ValueError('execution no links')
            cursor=cursor.parent
        return path
    def raw(spec,cap=262144):
        check()
        if type(spec) is not dict or set(spec)!={'path','bytes','sha256'} or type(spec['bytes']) is not int or not 0<spec['bytes']<=cap or type(spec['sha256']) is not str or len(spec['sha256'])!=64:raise ValueError('execution typed exact descriptor')
        path=pathname(spec['path']);before=path.stat()
        if before.st_size!=spec['bytes']:raise ValueError('execution size')
        with path.open('rb') as stream:value=stream.read(spec['bytes']+1)
        if len(value)!=spec['bytes'] or hashlib.sha256(value).hexdigest()!=spec['sha256']:raise ValueError('execution digest')
        value.decode('utf8',errors='strict')
        again=pathname(spec['path'])
        if again!=path or again.stat().st_size!=before.st_size:raise ValueError('execution late path')
        with again.open('rb') as stream:second=stream.read(spec['bytes']+1)
        if second!=value:raise ValueError('execution reread')
        check()
        return value
    admission_path=pathname(EXECUTION_ADMISSION_PATH)
    if not admission_path.exists():return None
    if admission_path.stat().st_size>262144:raise ValueError('bounded execution admission')
    admission_raw=admission_path.read_bytes()
    admission_spec={'path':EXECUTION_ADMISSION_PATH,'bytes':len(admission_raw),'sha256':hashlib.sha256(admission_raw).hexdigest()}
    admission=json.loads(raw(admission_spec))
    if set(admission)!={'schema','state','manifest','rules','source_freeze','anchor'} or admission['schema']!='flylab.m3.execution.admission.v1' or admission['state']!='ADMITTED':raise ValueError('complete admitted execution context')
    if admission['rules']!=EXECUTION_RULE_SPEC:raise ValueError('compiled execution rules')
    rules=json.loads(raw(admission['rules']))
    manifest=json.loads(raw(admission['manifest']))
    fields={'schema','pass_id','start_index','start_calls','start_passes','rules','source_freeze','closed_anchor','registration','resource_extension','human_authorization','history_catalogue','historical_inputs','sealed_contexts','role_paths','attempts','review_pins'}
    if set(manifest)!=fields or manifest['schema']!='flylab.m3.execution.manifest.v1':raise ValueError('complete finite manifest')
    if manifest['rules']!=admission['rules'] or manifest['source_freeze']!=admission['source_freeze'] or manifest['closed_anchor']!=admission['anchor']:raise ValueError('acyclic manifest binding')
    anchor=json.loads(raw(admission['anchor'],2097152))
    floor=json.loads(raw(EXECUTION_ANCHOR_SPEC))
    if type(anchor.get('debits')) is not list or type(anchor.get('e360_calls')) is not dict or type(anchor.get('e360_passes')) is not dict:raise ValueError('typed sealed anchor roster')
    counts=(('start_index',len(anchor['debits'])),('start_calls',len(anchor['e360_calls'])),('start_passes',len(anchor['e360_passes'])))
    if any(type(manifest[k]) is not int or manifest[k]!=v for k,v in counts):raise ValueError('actual closed anchor roster')
    if anchor['debits'][:111]!=floor['debits'] or any(anchor[k].get(i)!=v for k in ('e360_calls','e360_passes') for i,v in floor[k].items()):raise ValueError('unchanged compiled preservation floor')
    if type(manifest['pass_id']) is not str or not manifest['pass_id'].startswith('e360-') or not 5<len(manifest['pass_id'])<=91 or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in manifest['pass_id']):raise ValueError('data pass identifier')
    freeze=json.loads(raw(manifest['source_freeze']))
    freeze_fields={'schema','rules','human_authorization','ROOT_review_cycle_complete','actual_ROOT_FIRST','actual_ROOT_SECOND','actual_Sol_FIRST','actual_Sol_R1','source_copies','own_sources_sha256','native_sources','native_request_scripts','collector_source','capture_caller','commit_caller','first_error_boundary','fixed_native_context'}
    if type(freeze) is not dict or set(freeze)!=freeze_fields:raise ValueError('complete exact Source freeze fields')
    if manifest['review_pins']!=EXECUTION_REVIEW_PINS:raise ValueError('actual completed review cycle pins')
    for field,spec in EXECUTION_REVIEW_PINS.items():
        if freeze[field]!=spec:raise ValueError('exact actual reviewed pin '+field)
        raw(spec)
    if freeze.get('schema')!='flylab.m3.execution.source-freeze.v1' or freeze.get('rules')!=EXECUTION_RULE_SPEC or freeze.get('human_authorization')!=manifest['human_authorization'] or type(freeze.get('source_copies')) is not list or len(freeze['source_copies'])!=4 or len(freeze.get('own_sources_sha256',{}))!=9:raise ValueError('actual execution Source freeze')
    if set(manifest['role_paths'])!=set(rules['primary_roles']) or set(manifest['attempts'])!={'fixture','suite'}:raise ValueError('finite role paths')
    for role,paths in manifest['role_paths'].items():
        if set(paths)!={'request','expected_output','completion'}:raise ValueError('finite request outputs')
        for name in paths.values():
            if not name.startswith('verification/banc-natural-gait-m3-20260919/parent-'):raise ValueError('canonical execution path')
            pathname(name)
    history=json.loads(raw(manifest['history_catalogue']))
    old_history=json.loads(raw(EXECUTION_HISTORY_SPEC))
    if type(manifest['sealed_contexts']) is not list or len(manifest['sealed_contexts'])>32:raise ValueError('finite sealed contexts')
    sealed=[];required=list(old_history['members'])
    for spec in manifest['sealed_contexts']:
        record=json.loads(raw(spec))
        if type(record) is not dict or set(record)!={'schema','pass_id','admission','manifest','source_freeze','aliases','inputs'} or record['schema']!='flylab.m3.execution.sealed-context.v1':raise ValueError('exact sealed context record')
        if type(record['aliases']) is not list or not 3<=len(record['aliases'])<=24 or type(record['inputs']) is not list or len(record['inputs'])>64:raise ValueError('finite physical historical declaration')
        sealed.append((spec,record));required.append(spec)
        required.extend(row['physical'] for row in record['aliases']);required.extend(record['inputs'])
    expected={row['path']:row for row in required}
    if any(expected[row['path']]!=row for row in required) or history.get('members')[:1204]!=old_history['members'] or {row['path']:row for row in history['members']}!=expected or len(history['members'])!=len(expected):raise ValueError('exact floor and selected historical members')
    if history['member_count']!=len(expected) or history['aggregate_member_bytes']!=sum(row['bytes'] for row in expected.values()) or history['max_leaf_bytes']!=2097152:raise ValueError('exact finite history totals')
    direct=json.loads(raw(EXECUTION_DIRECT_INPUT_SPEC))
    if manifest['historical_inputs']!=direct['required']:raise ValueError('finite actual historical read declaration')
    if manifest['registration']!={'path':'verification/banc-natural-gait-m3-20260919/parent-e360-reflected-registration.json','bytes':77417,'sha256':'40e80b01a243323e7bfb09eb9aad3608f8f36a13ffe2bea013be954be0d51fd3'}:raise ValueError('fixed registration')
    if manifest['resource_extension']!={'path':'verification/banc-natural-gait-m3-20260919/parent-e360-approved-generous-registration-v2.json','bytes':5461,'sha256':'ea5afdffb09f6a118a18e3039546a911bbc9ed1465cafb6a1fe472ce3471862b'}:raise ValueError('fixed retained resource extension')
    human={'path':'verification/banc-natural-gait-m3-20260919/parent-e360-current-human-autonomy-20261006.json','bytes':1098,'sha256':'8c912aab85ab18fc668caeba9e67bf621318a933e4a4427ad525b0e7db035224'}
    if manifest['human_authorization']!=human:raise ValueError('stable human delegation')
    labels={'ruff':'ruff','scanner':'scan','fixture_build':'fbuild','fixture_worker':'fworker','fixture_seal':'fseal','suite_build':'sbuild','suite_worker':'sworker','suite_seal':'sseal','qualification':'q'}
    for role,paths in manifest['role_paths'].items():
        call=manifest['pass_id']+'-'+labels[role];stem='verification/banc-natural-gait-m3-20260919/parent-'+call
        if paths['request']!=stem+'-final-request.json' or paths['completion']!=stem+'-completion.json':raise ValueError('fixed current request/completion template')
        if role=='ruff':output=stem+'-output.json'
        elif role=='scanner':output='verification/banc-natural-gait-m3-20260919/parent-'+manifest['pass_id']+'-closure.json'
        elif role=='qualification':output='verification/banc-natural-gait-m3-20260919/parent-'+manifest['pass_id']+'-qualification.json'
        else:
            worker=role.split('_')[0];attempt=manifest['attempts'][worker]
            if type(attempt) is not str or not attempt.startswith('e360-') or not 5<len(attempt)<=53 or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in attempt):raise ValueError('finite attempt data')
            output='verification/banc-natural-gait-m3-20260919/parent-'+attempt+'-proposal.json' if role.endswith('_build') else 'verification/banc-natural-gait-m3-20260919/parent-e346-e360-'+attempt+'-'+worker+('-outer.json' if role.endswith('_worker') else '-full-exit.json')
        if paths['expected_output']!=output:raise ValueError('fixed output template')
    if freeze.get('ROOT_review_cycle_complete') is not True or freeze.get('human_authorization')!=human:raise ValueError('genuine reviewed Source freeze')
    for field in ('actual_ROOT_FIRST','actual_ROOT_SECOND','actual_Sol_FIRST','actual_Sol_R1'):
        value=freeze.get(field)
        if type(value) is not dict or set(value)!={'path','bytes','sha256'} or type(value['bytes']) is not int or value['bytes']<=0:raise ValueError('actual reviewed pin '+field)
    current4={'tools/verify_c_banc_m3_source_closure_diagnostic.py','tests/test_c_banc_source_closure_diagnostic.py','verification/banc-natural-gait-m3-20260919/parent-e360-root-account.ps1','verification/banc-natural-gait-m3-20260919/parent-e360-root-metadata-helper.py'}
    own5={'tools/verify_c_banc_m3_candidate_body_handoff.py','tests/test_c_banc_candidate_body_handoff.py','tools/verify_c_banc_m3_startup_baseline.py','verification/banc-natural-gait-m3-20260919/parent-e267-supervise.py','verification/banc-natural-gait-m3-20260919/parent-e346-supervise-identities.py'}
    if set(freeze['own_sources_sha256'])!=current4|own5:raise ValueError('exact actual own9 path roster')
    if {row['current']['path'] for row in freeze['source_copies']}!=current4:raise ValueError('exact current Source4 paths')
    native=['verification/banc-natural-gait-m3-20260919/parent-execution-'+name+'.ps1' for name in ('capture-proposed','prepare-native','bookkeeping-native','ROOT-dispatch-adapter')]
    if [row['path'] for row in freeze['native_sources']]!=native or freeze['native_request_scripts']!=freeze['native_sources'][1:]:raise ValueError('exact Native4 order/request3')
    return {'admission':admission,'admission_spec':admission_spec,'manifest':manifest,'manifest_spec':admission['manifest'],'rules':rules,'freeze':freeze,'raw':raw,'anchor':anchor,'anchor_spec':admission['anchor'],'floor':floor,'sealed_records':sealed}

if '--scan' in sys.argv and os.environ.get('E360_CALL_REQUEST') and execution_bootstrap(Path(__file__).resolve().parents[1]) is not None:
    _boot=execution_bootstrap(Path(__file__).resolve().parents[1])
    _scanner_root=Path(__file__).resolve().parents[1]
    _request_path=(_scanner_root/os.environ['E360_CALL_REQUEST']).resolve()
    if not _request_path.is_relative_to(_scanner_root) or _request_path.stat().st_size>262144:raise ValueError('bounded contained current request')
    _request_value=json.loads(_request_path.read_bytes())
    if _request_value.get('pass_id')!=_boot['manifest']['pass_id'] or _request_value.get('mode')!='generated' or _request_value.get('revalidation_amendment')!=_boot['manifest_spec']:raise ValueError('scanner admitted data context')
    _SCANNER_BOOTSTRAP_POLICY=_boot['rules']['scanner_limits']
    if time.perf_counter()-ENTRY>_SCANNER_BOOTSTRAP_POLICY['internal_s']:raise TimeoutError('scanner same ENTRY')
elif '--scan' in sys.argv and os.environ.get('E360_CALL_REQUEST'):
    # Same real ENTRY; passive exact current bootstrap only, before project import/history.
    _request_path=(ROOT/os.environ['E360_CALL_REQUEST']).resolve()
    if not _request_path.is_relative_to(ROOT):raise ValueError('contained scanner request')
    _request_bytes=_request_path.read_bytes()
    if len(_request_bytes)>262144:raise ValueError('bounded scanner request')
    _request_value=json.loads(_request_bytes)
    if _request_value.get('pass_id')!='e360-generous-p30' or _request_value.get('mode')!='generated':raise ValueError('completed history/future/original cannot scan')
    _authority_name='verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json'
    _authority_raw=(ROOT/_authority_name).read_bytes()
    if len(_authority_raw)!=12370 or hashlib.sha256(_authority_raw).hexdigest()!='2b7412ea54d929a0a89df8679356e9bf8b1b6c643184f6291c4a8649006d6dfb':raise ValueError('scanner exact current policy')
    if _request_value.get('revalidation_amendment')!={'path':_authority_name,'bytes':12370,'sha256':'2b7412ea54d929a0a89df8679356e9bf8b1b6c643184f6291c4a8649006d6dfb'}:raise ValueError('scanner exact current authority')
    _SCANNER_BOOTSTRAP_POLICY=json.loads(_authority_raw)['scanner_limits']
    _elapsed=time.perf_counter()-ENTRY
    if not math.isfinite(_elapsed) or _elapsed<0 or _elapsed>_SCANNER_BOOTSTRAP_POLICY['internal_s']:raise TimeoutError('scanner pre-import same ENTRY')

from tools import verify_c_banc_m3_candidate_body_handoff as lib

BASE,PYTHON,SUPERVISOR,OBSERVER=lib.BASE,lib.PYTHON,lib.SUPERVISOR,lib.OBSERVER
OWNED=('tools/verify_c_banc_m3_source_closure_diagnostic.py','tests/test_c_banc_source_closure_diagnostic.py')
REGISTRATION=BASE+'/parent-e360-reflected-registration.json'
REGISTRATION_SHA='40e80b01a243323e7bfb09eb9aad3608f8f36a13ffe2bea013be954be0d51fd3'
INHERITED_SHA='63967de51be6630051f36262110a9158fe55adf91bdf61d3b334c51dc6c68563'
AMENDMENT=BASE+'/parent-e360-generated-redistribution-registration.json'
AMENDMENT_SHA='82578e33d49f451ecac5e8302f242d5928e71f4dd7e84a0d037660fc6dfd12aa'
RESOURCE_EXTENSION=BASE+'/parent-e360-approved-generous-registration-v2.json'
RESOURCE_EXTENSION_SHA='ea5afdffb09f6a118a18e3039546a911bbc9ed1465cafb6a1fe472ce3471862b'
OLD_REVALIDATION_AMENDMENT=BASE+'/parent-e360-same-budget-revalidation-amendment.json'
OLD_REVALIDATION_SHA='f9ba8cf6a1437904595c40cac5529f9a397694c71c95ee33b78841101f8731d8'
ACTIVE_REVALIDATION_AMENDMENT=BASE+'/parent-e360-admin-recovery-amendment.json'
ACTIVE_REVALIDATION_SHA='9cc0eb56e58d4be93c000d1a84f60d079c63e4b74c807b39a0164846a9a6a96e'
SOURCE_RECOVERY_AMENDMENT=BASE+'/parent-e360-ps-source-recovery-amendment.json'
SOURCE_RECOVERY_SHA='0c236d980577b8985166ef9cf7a22b5e18cb5d37a73421bc444a027ca3efaae7'
RUFF_RECOVERY_AMENDMENT=BASE+'/parent-e360-ruff-recovery-amendment.json'
RUFF_RECOVERY_SHA='0c49f109cb44b8d83e3b793326e0f3bfe30a82605caf56609796792eae923bde'
REVALIDATION_AMENDMENT=BASE+'/parent-e360-authority-scope-amendment.json'
REVALIDATION_SHA='9ecd3a7b1b29b6868e5636c3b544eb89065d2acec430b848bc5777e691f16052'
EXECUTION_BUDGET_AMENDMENT=BASE+'/parent-e360-execution-budget-amendment-v2.json'
EXECUTION_BUDGET_SHA='7f8c4014c94709f31bd9ed55d17bef6b6e27d97acfc71f5167b269ae6731215f'
SUITE_RECOVERY_AMENDMENT=BASE+'/parent-e360-suite-recovery-amendment-v1.json'
SUITE_RECOVERY_SHA='5d2222d6f53e3a05cec53680771cf14aa44ed4c777eee2ebbb3ed52650e7fa7d'
LINT_RECOVERY_AMENDMENT=BASE+'/parent-e360-lint-recovery-amendment-v1.json'
LINT_RECOVERY_SHA='b94c3feea24b3d1213e1d3272af0c19417abb72d2dcdb8cf65f60d46d0478eda'
LINT_RECOVERY_FIELDS=('e360_lint_recovery_activation','e360_lint_recovery_amendment')
P17_RECOVERY_AMENDMENT=BASE+'/parent-e360-p17-recovery-amendment.json'
P17_RECOVERY_BYTES=121989
P17_RECOVERY_SHA='569ed3e5bbbb998c7410dad8372af45e0526a802a62bb6b4e42d5137c341b295'
P17_RECOVERY_FIELDS=('e360_p17_recovery_activation','e360_p17_recovery_amendment')
CURRENT_REPAIR_POLICY='verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json'
CURRENT_REPAIR_SPEC=json.loads(r'''{"sha256":"2b7412ea54d929a0a89df8679356e9bf8b1b6c643184f6291c4a8649006d6dfb","bytes":12370,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json"}''')
HISTORICAL_P19_FIELDS=('e360_p19_recovery_activation','e360_p19_recovery_amendment')
P19_ORIGINAL_POLICY_SPEC=json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json","sha256":"61efa339496ca347fa91ada6241ec458fc265440cef89aaa2b9c11a6dfdbdc92","bytes":6880}''')
P19_FROZEN_POLICY_SPEC=json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p19-root-final-repair-policy.json","sha256":"61efa339496ca347fa91ada6241ec458fc265440cef89aaa2b9c11a6dfdbdc92","bytes":6880}''')
P19_FROZEN_FREEZE_SPEC=json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p19-root-final-freeze.json","sha256":"220bf54dde8c2e483ab446b0ff19189058b806b8f5452ab4600de5561686d7eb","bytes":11035}''')
P19_CLOSED100_SPEC=json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p19-closed-ledger100.json","sha256":"dcb36333fe9ef24c7627188b023bb20922bcc3878454a5fb4806cc11d047a587","bytes":163911}''')
P19_ACTUAL_RECORD_SPECS=json.loads(r'''{"full-return.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p19-ruff-full-return.json","sha256":"a0f3d378d235815a920bf8ad38406b93454c2dfdbd7397feadb032b1a155047e","bytes":3421},"stdout.txt":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p19-ruff-stdout.txt","sha256":"a5f7a45dbd663f49fe47f52a387ca2305f60c6e4b31c9ee85e391311a2acaf50","bytes":790},"output.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p19-ruff-output.json","sha256":"b3a16644516c9bf6f52ad07cb414e0965793a5a509eac379ea7e35549813c450","bytes":1777},"final-request.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p19-ruff-final-request.json","sha256":"9b86e22058e2689c112efe09b328164092b900c89352fd50ec46c8d94bf558bc","bytes":4342},"genuine-capture-return.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p19-ruff-genuine-capture-return.json","sha256":"5289ec2c59322069fc09f55362bcecdb7f53f2a1731eccf22c8168844820c112","bytes":1473},"ROOT-collector-trace.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p19-ruff-ROOT-collector-trace.json","sha256":"8974ade9f80d54f1e598c9ab4ba4ba2defd7b52baae4920fca225c3da1a515ac","bytes":2931},"actual.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p19-ruff-actual.json","sha256":"f943cbf2dd580e7777987878b77b51250b4a069f3897f14196d06eb876700091","bytes":1128},"stderr.txt":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p19-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","bytes":0}}''')
HISTORICAL_P20_FIELDS=('e360_p20_recovery_activation','e360_p20_recovery_amendment')
P20_ORIGINAL_POLICY_SPEC=json.loads(r'''{"sha256":"5aa17af9268331fced4d889bd936660fc6558b4386886ccd10173850da8bd9ba","bytes":7202,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json"}''')
P20_FROZEN_POLICY_SPEC=json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p20-root-final-repair-policy.json","sha256":"5aa17af9268331fced4d889bd936660fc6558b4386886ccd10173850da8bd9ba","bytes":7202}''')
P20_FROZEN_FREEZE_SPEC=json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p20-root-final-freeze.json","sha256":"cb822407d2ef2554d91a9cb6ea74110d79046d5ca490dac4ba6dcc2eaf285727","bytes":11775}''')
P20_CLOSED101_SPEC=json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p20-closed-ledger101.json","sha256":"a848728b75957b25050386ebae153405c3d9356265f0869d03d8acabb5b2bbc6","bytes":170606}''')
P20_ACTUAL_RECORD_SPECS=json.loads(r'''{"full-return.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p20-ruff-full-return.json","sha256":"b1bbd03915db8ce72f9dc40c2eff0a7f5e938ba356ee25e2f0878b1dccbf9cbe","bytes":3420},"stdout.txt":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p20-ruff-stdout.txt","sha256":"5b165d379beda4d8945dca4d7b8aaf46b49ae8feb192e5987eafa753c9ff9419","bytes":13328},"output.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p20-ruff-output.json","sha256":"32a9d5ece04e2e834206ab824c541e94e3982d802ed5cd57d0f7c7e5e40b2eb6","bytes":1782},"final-request.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p20-ruff-final-request.json","sha256":"0cf7aa5910f74c6c1e4c654952b310958ca90f7739888c696659af798d3ac207","bytes":4339},"genuine-capture-return.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p20-ruff-genuine-capture-return.json","sha256":"2c3b224c8c28d430b398259bca9df02053990ed1a28f7a766590e921b03b711e","bytes":1478},"ROOT-collector-trace.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p20-ruff-ROOT-collector-trace.json","sha256":"61f02ff5ce2a9aa7e6a0daf690e1bad28ac433d6e38e1b04f48e326385913332","bytes":2937},"actual.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p20-ruff-actual.json","sha256":"2ef4fdec5a0c38bfc93a2315b98c98b0c4b36772737037908ca33dd742fe12ee","bytes":1128},"stderr.txt":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p20-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","bytes":0}}''')
COST_AUTHORITY_PASS_RANGES=((59,60,3),(60,62,4),(62,63,5),(63,68,6),(68,73,7),(73,81,8),(81,82,9),(82,83,10),(83,88,11),(88,89,12),(89,91,13),(91,92,14),(92,93,15),(93,94,16),(94,96,17),(96,99,18),(99,100,19),(100,101,20),(101,103,21),(103,104,22),(104,105,23),(105,107,25),(107,108,27),(108,109,29))
HISTORICAL_P21_FIELDS=('e360_p21_recovery_activation','e360_p21_recovery_amendment')
P21_ORIGINAL_POLICY_SPEC=json.loads(r'''{"sha256":"c4bf79de3e8efdf90170916eb6ea24747aa1830e4799885e06dc019614b4dda5","path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json","bytes":7781}''')
P21_FROZEN_POLICY_SPEC=json.loads(r'''{"sha256":"c4bf79de3e8efdf90170916eb6ea24747aa1830e4799885e06dc019614b4dda5","path":"verification/banc-natural-gait-m3-20260919/parent-e360-p21-root-final-repair-policy.json","bytes":7781}''')
P21_FROZEN_FREEZE_SPEC=json.loads(r'''{"sha256":"556486c5ce50b00dc8ffce6230f009adee21495878564c048d9cd18cfbbc34b8","path":"verification/banc-natural-gait-m3-20260919/parent-e360-p21-root-final-freeze.json","bytes":12292}''')
P21_CLOSED103_SPEC=json.loads(r'''{"sha256":"23886a8fbcc12d0db312d2721c3a568c04e2f015682f00ce729563743a3f8c07","path":"verification/banc-natural-gait-m3-20260919/parent-e360-p21-closed-ledger103.json","bytes":178477}''')
P21_ACTUAL_RUFF_RECORD_SPECS=json.loads(r'''{"full-return.json":{"sha256":"914b01b2a735b4546ca314047d36df158360b29929bf221671c05d49fc37a550","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p21-ruff-full-return.json","bytes":3355},"final-request.json":{"sha256":"51470bec5992d1a66f0ce56178f92b069fadf223e8b1667e353473969616e41a","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p21-ruff-final-request.json","bytes":4340},"actual.json":{"sha256":"f01f8974d4da0a6743ab6ba39bd57073c5bf7dd2c5784e03ce6fb038093024ac","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p21-ruff-actual.json","bytes":1128},"ROOT-collector-trace.json":{"sha256":"596c6446320fa8fcffadc9768b4870e134b597c8d45942bfd8473fe0e2c1c5e7","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p21-ruff-ROOT-collector-trace.json","bytes":2852},"genuine-capture-return.json":{"sha256":"11ee948ccf7a0677e27c1dc721af08330f387fc34e8839efafba9e310905f261","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p21-ruff-genuine-capture-return.json","bytes":1404},"output.json":{"sha256":"779bef26c59c2cbd5b31e6aae296442ad12fcb96a24869b70b85540dec91c773","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p21-ruff-output.json","bytes":1708},"stdout.txt":{"sha256":"82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p21-ruff-stdout.txt","bytes":19},"stderr.txt":{"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p21-ruff-stderr.txt","bytes":0}}''')
P21_ACTUAL_SCANNER_RECORD_SPECS=json.loads(r'''{"full-return.json":{"sha256":"4e1b995c62bc2763627338dc4096ca6fd00f9e458d2903f18a2465ed6b538440","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p21-scan-full-return.json","bytes":2667},"final-request.json":{"sha256":"edc1258cead568c759bacaf341d73153afae9e7668cc537e2f41a967d6668379","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p21-scan-final-request.json","bytes":2207},"actual.json":{"sha256":"99a152073236fcaa219d964f470bc4f7e84d8bc53513d6ad5ee96ed4daf7b6a7","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p21-scan-actual.json","bytes":1128},"ROOT-collector-trace.json":{"sha256":"2612981d69ae0b8aa650134f64abed1881e3e9622d8e44e445f3113d0d1e93d8","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p21-scan-ROOT-collector-trace.json","bytes":3078}}''')
HISTORICAL_P22_FIELDS=('e360_p22_recovery_activation','e360_p22_recovery_amendment')
P22_ORIGINAL_POLICY_SPEC=json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json","bytes":8421,"sha256":"d94f7dd1e89a90fbe033592ea9c9cafb2969d37f491c93157edf574f7bae3d5a"}''')
P22_FROZEN_POLICY_SPEC=json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p22-root-final-repair-policy.json","bytes":8421,"sha256":"d94f7dd1e89a90fbe033592ea9c9cafb2969d37f491c93157edf574f7bae3d5a"}''')
P22_FROZEN_FREEZE_SPEC=json.loads(r'''{"bytes":11390,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p22-root-final-freeze.json","sha256":"47f523a8f636b52298f842c230a889659502381a61f863385d9ace2bf78bbadc"}''')
P22_CLOSED104_SPEC=json.loads(r'''{"bytes":185161,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p22-closed-ledger104.json","sha256":"528d1b7011419e2da2383bdab25e87e8f9330f37cbb46506597c13f5fc044dbe"}''')
P22_ACTUAL_RUFF_RECORD_SPECS=json.loads(r'''{"genuine-capture-return.json":{"sha256":"593f0cae813ae80b233a3bd38baa2ceb337747f0cad6bb3d5754ac97687e753a","bytes":1476,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p22-ruff-genuine-capture-return.json"},"output.json":{"sha256":"7e2b86a4398bca7b22ff77f11f90a9c4379eaefe37ac4eaee7c11b4c0d81bd0f","bytes":1780,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p22-ruff-output.json"},"final-request.json":{"sha256":"054dc4767b4ae04daeaa45a77fc98e7876a903b282c4262073cb56563c83e57a","bytes":4340,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p22-ruff-final-request.json"},"stdout.txt":{"sha256":"e52c71c8b484a244225c2ea309d0b5c40c611aab26d07df7798e25f5805013d3","bytes":1072,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p22-ruff-stdout.txt"},"actual.json":{"sha256":"456452db5046a99f1df85a74c611aefaee060ed82d43036e50ba00e75d9bb9a4","bytes":1128,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p22-ruff-actual.json"},"full-return.json":{"sha256":"e4ddee891fb620aaff524b5293d122a96802754593677be80dc25d43afc75920","bytes":3421,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p22-ruff-full-return.json"},"stderr.txt":{"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","bytes":0,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p22-ruff-stderr.txt"},"ROOT-collector-trace.json":{"sha256":"247246c2a0632ed7f96e8d953efbeb2bfae8840079250f326e5dd6bcadfadb2b","bytes":2933,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p22-ruff-ROOT-collector-trace.json"}}''')
HISTORICAL_P23_FIELDS=('e360_p23_recovery_activation','e360_p23_recovery_amendment')
P23_ORIGINAL_POLICY_SPEC=json.loads(r'''{"sha256":"60681e6a2f1e44bfbccf47c269142c51382cc5d709891fb6626f0edd9962ebe2","path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json","bytes":9040}''')
P23_FROZEN_POLICY_SPEC=json.loads(r'''{"sha256":"60681e6a2f1e44bfbccf47c269142c51382cc5d709891fb6626f0edd9962ebe2","path":"verification/banc-natural-gait-m3-20260919/parent-e360-p23-root-final-repair-policy.json","bytes":9040}''')
P23_FROZEN_FREEZE_SPEC=json.loads(r'''{"sha256":"612195dcf2530850d8dc10d374c849e69721ee8ed893dd5f81524853d85de5a0","bytes":11390,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p23-root-final-freeze.json"}''')
P23_CLOSED105_SPEC=json.loads(r'''{"sha256":"3b3088e43efafbbe038e022661a4eea6cf9b2b10345a506a4cc34b85c4de78ed","bytes":191858,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p23-closed-ledger105.json"}''')
P23_ACTUAL_RUFF_RECORD_SPECS=json.loads(r'''{"full-return.json":{"bytes":3421,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p23-ruff-full-return.json","sha256":"c1bbb8709245b50bb29cc6b12e1efc155a73fdfac62659b8b5778c852e1260a7"},"output.json":{"bytes":1780,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p23-ruff-output.json","sha256":"8095eb0554e77fcdecb5ac6750ce76e065e07c40ade13b093ac540afdd0f8c2c"},"genuine-capture-return.json":{"bytes":1476,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p23-ruff-genuine-capture-return.json","sha256":"ef35ec4826f91c319382d97dedc4fdd9697a079ab53a53254bd969bd8bcc01c9"},"actual.json":{"bytes":1128,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p23-ruff-actual.json","sha256":"82c7f50d085afb5703340f2f3b12b056cdffeb672f6103f18b55021905c139fc"},"final-request.json":{"bytes":4340,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p23-ruff-final-request.json","sha256":"39f177a2e78deb09532fe706b5e34d15aedf4627b4d3d5d6fec56f795d5e9fe8"},"ROOT-collector-trace.json":{"bytes":2934,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p23-ruff-ROOT-collector-trace.json","sha256":"ef198086b011c5df7a486f6356c96f20a84c1233397a06173ba1ef3a4858d31c"},"stdout.txt":{"bytes":1478,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p23-ruff-stdout.txt","sha256":"b78f50c27049f32ea6e1390663cbda2f873594176f85e2ed4f15a18a10800ca2"},"stderr.txt":{"bytes":0,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p23-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}}''')
CURRENT_P30_FIELDS=('e360_p30_recovery_activation','e360_p30_recovery_amendment')
_CURRENT_P30_PINS=tuple(json.loads(r'''[{"sha256":"2b7412ea54d929a0a89df8679356e9bf8b1b6c643184f6291c4a8649006d6dfb","bytes":12370,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json"},{"sha256":"5362d85a28b92efb89b0f66a14e173fa1d821e33acdde05a2fe214e9365f52d2","bytes":9656,"path":"docs/BANC_M3_E360_P30_CURRENT_COST_IDENTITY_ROOT_REFLECTED_20261007.md"},{"path":"docs/BANC_M3_E360_P30_CURRENT_COST_IDENTITY_ROOT_DESIGN_20261007.md","bytes":8824,"sha256":"68599f1318598c606f3cc21a0ed2410d432592a7fa553d0d0b745a23d136a4e4"},{"bytes":1098,"sha256":"8c912aab85ab18fc668caeba9e67bf621318a933e4a4427ad525b0e7db035224","path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-human-autonomy-20261006.json"},{"sha256":"ba7f61703cc4a2fbd606fdb4d623d7f1043b32db6dcc59f1589c18d967e1dde8","bytes":30887,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p30-actual-astra-review.json"},{"sha256":"d4299de1c24fb822dde3d44a11aa12bfbe4999293bb991c6d2e984f0df1f9987","bytes":4818,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p30-actual-astra-review.md"},{"sha256":"f6673785abb23e204031bf40bb9d734db0bbe2e289d90654fc5ffbdad86bdd72","bytes":1921,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p30-ROOT-actual-design-reflection.json"},{"sha256":"ce69cf7e48a2c0f7b1dae18deffb784b05f09ae182ae4379059298e53a5e7743","bytes":216320,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p30-ROOT-flat-history-refined1180.json"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-ROOT-scanner-native-admission-failure.json","bytes":2731,"sha256":"b10df60781733f43cca48bb883f16a29d53713c4e25728976046389fcaf15963"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-ROOT-scanner-admission-genuine-return.json","bytes":1606,"sha256":"18b3d770614589df82e562c4a6b799e476be4d62a11d5635c276e62d5be37291"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-ROOT-scanner-prepare-first-error.json","bytes":2386,"sha256":"b3d497c26da2d679ad8d5c79301a7b3b95c6446b6546efa73110989cea58a016"},{"bytes":206426,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-closed-ledger108.json","sha256":"f0e8583414c05d5aa99d1395018b57b6392004245fb508f114d1dd71edf80c2f"},{"bytes":208673,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-admin-held-ledger108.json","sha256":"ae4ba34b038eb053f322ddc3591a230b2f67ef4cf05db182b989786a9a8a9f7a"},{"bytes":4979,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-closed-decision.json","sha256":"ef770af24a52710522d5c9033ba2db6e784de6922579afafcea63c53c8eae004"},{"bytes":8435,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-freeze.json","sha256":"a73419cf3d3214c20a958e558cb8e0f397d1002345238f5ddf7747eaab8cd94a"},{"bytes":10788,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-repair-policy.json","sha256":"4b19ed927d9478bf0d9748e82152362e8d0c1244bbee3c8a7f2d6e2595ba348b"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-source-tool.py","sha256":"3f15eb959562556c8be693e2f976056324a061723261bc32af9051ded2b48577","bytes":690092},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-source-tests.py","sha256":"40542d38bf5fd02f9e3aab57b06ee30e980463829d3af46f94f0e1a3ac4db0de","bytes":470707},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-source-account.ps1","sha256":"337d9ce30238c9e1bf3b7637c2f5ca905a83ce4eb90b7c50c0bf7e164253d090","bytes":677643},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-source-helper.py","sha256":"f365e8c7c8a7c90c2bc3b0c20b2f4dc621231f0368297a94ad4f274bd6b903b7","bytes":12782},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-capture-proposed.ps1","sha256":"6fa1135265dcb7835622a8481b9daaf7c494cace3d9e3c89be70d14d8325368b","bytes":49908},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-prepare-native.ps1","sha256":"0028d1f9f0e06071deb379f3b308585fc9b93067e0e015cef30f16eb6007a430","bytes":7473},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-bookkeeping-native.ps1","sha256":"c89fe383b9b4f91552e6ba1842702bf8c7c851d761ab92c16a08e763d62f283a","bytes":16059},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-dispatch-adapter.ps1","sha256":"1b7eef4c99029308f64b66c364d1f53832710475659b561f74cdb54f4db5678d","bytes":14644},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-genuine-collector.txt","sha256":"46debd6e685014559b6b658686c55f7cfa529fdc834ee56d32b359c5ea69ed36","bytes":8138},{"bytes":4637,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-full-return.json","sha256":"46325a4d70098ceb77fe90ed979afeb5ec32fd8fa1ee825defcea59e28b1fffb"},{"bytes":1128,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-actual.json","sha256":"f55b6cbe0c3f2c7601e34b5386ca84020bea748d298783168d8ee216c3fca4a6"},{"bytes":4346,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-final-request.json","sha256":"3985723558efc9048a824581a24e8975e9028e83376af5a5173513c14da58040"},{"bytes":2930,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-ROOT-collector-trace.json","sha256":"fffadfe4c825817faad47b0c4f89f3624a1e37f7f3579de953883cebd490009e"},{"bytes":1472,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-capture-return.json","sha256":"d097693b5a8be6ed3e8be43c101ebeef2d257c0390cf053d544158f8d6ee1a36"},{"bytes":1777,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-output.json","sha256":"ede0a43bea8130b6290a96cc92ea8728f7275dee024d865ea95ee5802347876c"},{"bytes":794,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-stdout.txt","sha256":"48935bce1d0b453fe77a4fcb148906068a743ab2f1cebd84bffb4f0fe973ecb2"},{"bytes":0,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},{"bytes":16439,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ruff-hold-VERIFIED-NOT-COMMITTED.json","sha256":"22ff2ce8f49248450250a0262af39357641ceeabb79c9d9cfa9ad92819fbe71f"},{"bytes":1712,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-ruff-hold-reconciliation-result.json","sha256":"9e03e4a5e8b77d01a211fab670c9d550df67bbff83b94ab6c71d77d365dd9901"},{"bytes":849,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-ruff-hold-genuine-return.json","sha256":"f7d14957963d99d01ca46047287e3a3a36d71f19f9abfc76ee6620897fceb509"},{"bytes":37514,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-ruff-hold-reconcile-native.ps1","sha256":"fba8f48fc5654d62066747e4a10cf5a1902c272a6a86eed0c19a786dbd013586"},{"bytes":2865,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ruff-hold-ROOT-final-operation-freeze.json","sha256":"268741df198dcb723f3ff533966e1c763c8155cdef2b1000138c9497bc938164"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-ROOT-genuine-collector.txt","bytes":8138,"sha256":"df264afd41077daa57196076fff72fe31692977d3580c77c7984a70ac20b37d4"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-ROOT-collector-trace.json","bytes":2852,"sha256":"2795c12eae68757d600eacef9f0669f582f68e550bf95fc7f0c5223c277cb8af"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-actual.json","bytes":1129,"sha256":"bb2d5e3c76460c56394b38c84f8ca36cec5b0d7361d1e438c8e31f80f2a20036"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-stderr.txt","bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-source-account.ps1","bytes":714552,"sha256":"9162e88fae89de031119cdc4d2d14964e0acb0d0d7d24fcf5250cb6d981f494b"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-bookkeeping-native.ps1","bytes":16570,"sha256":"27b15befb84dbb84ebc616ac9de9cf278bcca72219e1d705c8a6d89f19e7dfe1"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-freeze.json","bytes":10168,"sha256":"8ee8d82b8fec0f9e8a85cc0acde08e19d4295893758298f2434314d620d4f210"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-source-tool.py","bytes":727578,"sha256":"00221b9bff24d3f3ad88fe5cd7067845ac6fa0e30152a104934a4b31c3b5f0ba"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-output.json","bytes":1709,"sha256":"7fa84b1a7afddb80f7353ed8267642d44579bde073178f4395f7954d54ccbdb9"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-ROOT-dispatch-adapter.ps1","bytes":14645,"sha256":"cf91be46d48ff60f8480f0643f1ffe26d76200cab147ddecd4d6e178779265e2"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-source-tests.py","bytes":474793,"sha256":"5481f5b9c7dfcfc79c5ec93ebc76a7e8455e046c8d4fa41f19bb7b1d63726fe1"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-prepare-native.ps1","bytes":7470,"sha256":"7621909a52e5dac7da6fa53c6fc3349e6374bd0f18f44e6ab1d6cc2084f352b8"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-repair-policy.json","bytes":12804,"sha256":"24bd4b5ba79e372443c9d57116b03df427099db553e795b1c1d63646138f047b"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-full-return.json","bytes":3348,"sha256":"f0975a24f4a46d2b08dfe1039d1f1c7b367e51926d3c1bf7d615911ed4b7efc7"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-stdout.txt","bytes":19,"sha256":"82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-capture-proposed.ps1","bytes":52584,"sha256":"af98d1910192ba5c3174e4a45e7731e0fc0a88e83df2355775f9cea8b8ea888f"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-final-request.json","bytes":4355,"sha256":"6ff3cd1bf016e88404d8e6fb9387e158960675feedf7c859e0dab78c84c4733b"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-capture-return.json","bytes":1404,"sha256":"80a5f316b7ac5df7effeb6c4845c12283e39f4facac000794a8d581bbcab6a58"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-source-helper.py","bytes":12782,"sha256":"5e345b2b9ed41c36b6e494315a9c64b2df817eca68c49aab71ecd85fe2c64eef"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-after-Ruff-closed-ledger109.json","bytes":212957,"sha256":"050f883629fcecefa2ceeb505bdabda5c183f74a045beb3382c7fec03dce4409"}]'''))
_COMPLETED_STREAM_PINS=tuple(json.loads(r'''[{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-stdout.txt","bytes":1905,"sha256":"8533cd7e1eef1d691d1fef6a81eb6dae7d9c859e3f9de1580984e922afff1601"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-stderr.txt","bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-stdout.txt","bytes":4190,"sha256":"1d4ac8043fb41b2a3a641f414a657c23dc93ee1e65b2ca4d39b9310dfa02d199"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-stderr.txt","bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p17-ruff-stdout.txt","bytes":19,"sha256":"82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p17-ruff-stderr.txt","bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}]'''))
P17_SAVED_RECORD_SPECS=json.loads(r'''{"decision":{"bytes":1955,"sha256":"5d30459d6f9d7bb59c50ccc28be634823a9a95f29d55a1b75bc2efd200e29060","path":"verification/banc-natural-gait-m3-20260919/parent-e360-p17-ROOT-closed-decision.json"},"closure":{"bytes":2290,"sha256":"bfdf737a472802c71c4742dd3f15bec8ef718de511b5af7cba89c26498c97cc6","path":"verification/banc-natural-gait-m3-20260919/parent-e360-p17-ROOT-independent-owned-closure.json"},"ruff_cost":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p17-ruff-full-return.json","bytes":3348,"sha256":"8a50ad65972e888e05c5efc2ba7209ccfdb45726d32cf6b1072ca3b909a4617c"},"ruff_actual":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p17-ruff-actual.json","bytes":1121,"sha256":"9977b7772e12924f8bbbe3954b0972e065272123ca4a71793f0c781f2df7a94f"},"ruff_request":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p17-ruff-final-request.json","bytes":4374,"sha256":"3f81108caf5eb0a21fb1c854a1252072f90544e7c59ab8427d639b3b17a5c28f"},"ruff_trace":{"sha256":"a6b521128c7677ff56d5574bd645127cf7c287d442f7af712ddf59015b82f744","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p17-ruff-ROOT-collector-trace.json","bytes":3103},"ruff_summary":{"sha256":"c1ca7f46f561cdb373d2be7da315b4a84f61a7fa6b71cef8bc4f5489ba631ab8","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p17-ruff-genuine-capture-return.json","bytes":1404},"ruff_manifest":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p17-ruff-output.json","bytes":1701,"sha256":"3aa91b2bd474486c12b0deb4c67895d7a15c8a285b4bc138061e8cf009665ba3"},"scan_cost":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p17-scan-full-return.json","bytes":2661,"sha256":"6945a6eacd310a6edc460ddf287bad665e41cfbc25b8db0d343dd818e581703f"},"scan_actual":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p17-scan-actual.json","bytes":1122,"sha256":"57ddc6605ea06e99a0e71e72cd6d7d91c04b014e1fc424470db5fbbc6f000e05"},"scan_request":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p17-scan-final-request.json","bytes":2200,"sha256":"718910da94f6df6b5b7f33770627afcca4ae4d56c13daf57514c8f62949bfdeb"},"scan_trace":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p17-scan-ROOT-collector-trace.json","sha256":"65a103d60e12613ac52487bea15ae20f03791a58a08165b1e6e52ae993bd3384","bytes":5056}}''')
CURRENT_P30_POLICY_KEYS=tuple(json.loads(r'''["schema","current_pass_id","only_current_generated_admission","authorization_kind","human_authorization","per_attempt_human_reapproval_required","actual_Astra_review","ROOT_reflected_design","closed_predecessor_ledger","predecessor_as_executed_freeze","historical_P17_literal","primary_roles","primary_caps_s","primary_total_reserved_s","primary_admin_total_s","primary_successful_suffix_after_s","future_with_unused_fallback120_s","separate_same_pending_fallback_reserved_s","whole_needed_reserved_s","fixture_primary_future_administration_s","suite_primary_future_administration_s","effective_roles","effective_helper_limits","scanner_limits","shared_limit_s","local_limit_s","local_cost_offset","new_grant_s","reset","historical_fixed_reader_capacity","original_role_admission","current_test_functions","current_test_phases","source_hash_cycle_allowed","exact_completed_stream_descriptors","no_original_GPU_body_neural_UI_in_this_stage","actual_Astra_md","ROOT_actual_reflection","original_ROOT_design","historical_P18_policy","contract_transport_cap_bytes","historical_P19_policy","historical_P19_allocation","activation_fields","historical_P20_policy","historical_P21_policy","historical_P21_history_reference","historical_P22_policy","historical_P22_history_reference","historical_P23_policy","historical_P23_history_reference","historical_P25_policy","descriptor_ownership","historical_P27_policy","current_admission_kind","paid_predecessor_success","predecessor_success_cap_s","unpaid_Source_admission_failure","unpaid_Source_genuine_return","unpaid_Source_first_error","current_cost_identity","historical_P29_policy"]'''))

P18_ORIGINAL_POLICY_SPEC=json.loads(r'''{"bytes":7227,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json","sha256":"64786eba059bfa09e2073c23bf621750c1f9c6d693d74511864b204c46da7849"}''')
P18_FROZEN_POLICY_SPEC=json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p18-root-final-repair-policy.json","bytes":7227,"sha256":"64786eba059bfa09e2073c23bf621750c1f9c6d693d74511864b204c46da7849"}''')
P18_ACTUAL_RECORD_SPECS=json.loads(r'''{"e360-generous-p18-ruff":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p18-ruff-full-return.json","bytes":3355,"sha256":"1bd5d79a25bce20cf6f3f73748aee7357c38621fa73eea1eb554f7587cf35bc5"},"e360-generous-p18-ruff-actual":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p18-ruff-actual.json","bytes":1128,"sha256":"cc0a4e48adb2ba4786ff868b37be96accf4397c7db551fb6297765b43ad81329"},"e360-generous-p18-ruff-trace":{"sha256":"41f495bbf1d566926510f82572e7402c7931e193fcc3e7162fd10f5da2eebdb4","bytes":3093,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p18-ruff-ROOT-collector-trace.json"},"e360-generous-p18-ruff-request":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p18-ruff-final-request.json","bytes":4340,"sha256":"ccfb8d89bc1c64c1023485ff8a47ca002a7bd6fd56ddb1351c37b476351d4d3b"},"e360-generous-p18-scan":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p18-scan-full-return.json","bytes":2847,"sha256":"fdfd7525a7fe5addd792b6682dbb7f0a777c7a2c7412f90c73f8b483d985f983"},"e360-generous-p18-scan-actual":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p18-scan-actual.json","bytes":1129,"sha256":"8bf373da26793b19d8b35b3db7da845466bdbef59ccd8e3b7c44995b026f2ae1"},"e360-generous-p18-scan-trace":{"bytes":1782,"sha256":"da1c1d2c7f3ed46065b5a49dcd0a83c15417a9aa3bb0d555c9614369f8a3b82a","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p18-scan-ROOT-collector-trace.json"},"e360-generous-p18-scan-request":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p18-scan-final-request.json","bytes":2207,"sha256":"4dfd5f4627798b7dd3fd571293cab77fac54fd6bfa3000b657561dd2a63bc864"},"e360-generous-p18-fbuild":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p18-fbuild-full-return.json","bytes":2685,"sha256":"e59b6bb44bb4691ab587c7a18f8dfc2655bf932426d6445a41b74d56ca5bf47b"},"e360-generous-p18-fbuild-actual":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p18-fbuild-actual.json","bytes":1135,"sha256":"ad21088da9a8c0ec8a976b644c27e5365955ef55e7e891ea36249626123c47b2"},"e360-generous-p18-fbuild-trace":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p18-fbuild-ROOT-collector-trace.json","bytes":2748,"sha256":"6684fa10b0db525311b43ffeeb6ac57c10905cadc9e9ae5c98bf050727b20626"},"e360-generous-p18-fbuild-request":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p18-fbuild-final-request.json","bytes":2590,"sha256":"a05cf196a4f9987d272440c05f62e9ee258d330d36fede694998ff6b0ec3e73a"}}''')
_CURRENT_P30_EXPECTED_POLICY=json.loads(r'''{"schema":"flylab.m3.e360.current-generated-repair-policy.v1","current_pass_id":"e360-generous-p30","only_current_generated_admission":true,"authorization_kind":"existing_general_in_scope_human_autonomy","human_authorization":{"bytes":1098,"sha256":"8c912aab85ab18fc668caeba9e67bf621318a933e4a4427ad525b0e7db035224","path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-human-autonomy-20261006.json"},"per_attempt_human_reapproval_required":false,"actual_Astra_review":{"sha256":"ba7f61703cc4a2fbd606fdb4d623d7f1043b32db6dcc59f1589c18d967e1dde8","bytes":30887,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p30-actual-astra-review.json"},"ROOT_reflected_design":{"sha256":"5362d85a28b92efb89b0f66a14e173fa1d821e33acdde05a2fe214e9365f52d2","bytes":9656,"path":"docs/BANC_M3_E360_P30_CURRENT_COST_IDENTITY_ROOT_REFLECTED_20261007.md"},"closed_predecessor_ledger":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-after-Ruff-closed-ledger109.json","bytes":212957,"sha256":"050f883629fcecefa2ceeb505bdabda5c183f74a045beb3382c7fec03dce4409"},"predecessor_as_executed_freeze":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-freeze.json","bytes":10168,"sha256":"8ee8d82b8fec0f9e8a85cc0acde08e19d4295893758298f2434314d620d4f210"},"historical_P17_literal":{"bytes":121989,"sha256":"569ed3e5bbbb998c7410dad8372af45e0526a802a62bb6b4e42d5137c341b295","path":"verification/banc-natural-gait-m3-20260919/parent-e360-p17-recovery-amendment.json"},"primary_roles":["ruff","scanner","fixture_build","fixture_worker","fixture_seal","suite_build","suite_worker","suite_seal","qualification"],"primary_caps_s":[3,30,30,60,20,40,520,25,40],"primary_total_reserved_s":768,"primary_admin_total_s":155,"primary_successful_suffix_after_s":[765,735,705,645,625,585,65,40,0],"future_with_unused_fallback120_s":[885,855,825,765,745,705,185,160,120],"separate_same_pending_fallback_reserved_s":120,"whole_needed_reserved_s":888,"fixture_primary_future_administration_s":125,"suite_primary_future_administration_s":65,"effective_roles":{"fixture":{"outer_s":60,"child_s":55,"internal_s":51,"science_s":42,"host_bytes":2147483648,"output_cap_bytes":16777216},"suite":{"outer_s":520,"child_s":480,"internal_s":450,"science_s":360,"host_bytes":2147483648,"output_cap_bytes":16777216},"saved_prefix_diagnostic":{"outer_s":12,"child_s":11,"internal_s":10,"science_s":9,"host_bytes":2147483648,"output_cap_bytes":2097152,"max_runs":1,"automatic_retries":0,"GPU_context_calls":0,"body_calls":0,"neural_calls":0}},"effective_helper_limits":{"fixture_build":{"successful_suffix_after_s":705,"action":"build","mode":"generated","internal_s":27,"worker_role":"fixture","outer_s":30},"fixture_seal":{"successful_suffix_after_s":625,"action":"seal","mode":"generated","internal_s":18,"worker_role":"fixture","outer_s":20},"suite_build":{"successful_suffix_after_s":585,"action":"build","mode":"generated","internal_s":37,"worker_role":"suite","outer_s":40},"suite_seal":{"successful_suffix_after_s":40,"action":"seal","mode":"generated","internal_s":22,"worker_role":"suite","outer_s":25},"qualification":{"successful_suffix_after_s":0,"action":"qualification","mode":"generated","internal_s":37,"worker_role":"suite","outer_s":40},"saved_prefix_build":{"successful_suffix_after_s":14,"action":"build","mode":"original","internal_s":4.5,"worker_role":"saved_prefix_diagnostic","outer_s":5},"saved_prefix_seal":{"successful_suffix_after_s":0,"action":"seal","mode":"original","internal_s":1.5,"worker_role":"saved_prefix_diagnostic","outer_s":2}},"scanner_limits":{"outer_s":30,"planned_fixture_s":60,"planned_suite_s":520,"internal_s":27,"separate_same_pending_fallback_reserved_s":120,"primary_admin_total_s":155},"shared_limit_s":1700,"local_limit_s":1589,"local_cost_offset":42,"new_grant_s":0,"reset":false,"historical_fixed_reader_capacity":18,"original_role_admission":false,"current_test_functions":84,"current_test_phases":252,"source_hash_cycle_allowed":false,"exact_completed_stream_descriptors":[{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-stdout.txt","bytes":1905,"sha256":"8533cd7e1eef1d691d1fef6a81eb6dae7d9c859e3f9de1580984e922afff1601"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-stderr.txt","bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-stdout.txt","bytes":4190,"sha256":"1d4ac8043fb41b2a3a641f414a657c23dc93ee1e65b2ca4d39b9310dfa02d199"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-stderr.txt","bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p17-ruff-stdout.txt","bytes":19,"sha256":"82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p17-ruff-stderr.txt","bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p19-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","bytes":0},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p20-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","bytes":0},{"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p21-ruff-stderr.txt","bytes":0},{"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","bytes":0,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p22-ruff-stderr.txt"},{"bytes":0,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p23-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},{"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","bytes":0,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p25-ruff-stderr.txt"},{"bytes":794,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-stdout.txt","sha256":"48935bce1d0b453fe77a4fcb148906068a743ab2f1cebd84bffb4f0fe973ecb2"},{"bytes":0,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-stdout.txt","bytes":19,"sha256":"82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-stderr.txt","bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}],"no_original_GPU_body_neural_UI_in_this_stage":true,"actual_Astra_md":{"sha256":"d4299de1c24fb822dde3d44a11aa12bfbe4999293bb991c6d2e984f0df1f9987","bytes":4818,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p30-actual-astra-review.md"},"ROOT_actual_reflection":{"sha256":"f6673785abb23e204031bf40bb9d734db0bbe2e289d90654fc5ffbdad86bdd72","bytes":1921,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p30-ROOT-actual-design-reflection.json"},"original_ROOT_design":{"path":"docs/BANC_M3_E360_P30_CURRENT_COST_IDENTITY_ROOT_DESIGN_20261007.md","bytes":8824,"sha256":"68599f1318598c606f3cc21a0ed2410d432592a7fa553d0d0b745a23d136a4e4"},"historical_P18_policy":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p18-root-final-repair-policy.json","bytes":7227,"sha256":"64786eba059bfa09e2073c23bf621750c1f9c6d693d74511864b204c46da7849"},"contract_transport_cap_bytes":524288,"historical_P19_policy":{"sha256":"61efa339496ca347fa91ada6241ec458fc265440cef89aaa2b9c11a6dfdbdc92","bytes":6880,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p19-root-final-repair-policy.json"},"historical_P19_allocation":{"sha256":"0eb0f5d45fd5ccf1eeec60694523d3444dcab8d93f395884121bc5d1562f508c","bytes":1643,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p19-ROOT-resource-allocation.json"},"activation_fields":["e360_p30_recovery_activation","e360_p30_recovery_amendment"],"historical_P20_policy":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p20-root-final-repair-policy.json","sha256":"5aa17af9268331fced4d889bd936660fc6558b4386886ccd10173850da8bd9ba","bytes":7202},"historical_P21_policy":{"sha256":"c4bf79de3e8efdf90170916eb6ea24747aa1830e4799885e06dc019614b4dda5","path":"verification/banc-natural-gait-m3-20260919/parent-e360-p21-root-final-repair-policy.json","bytes":7781},"historical_P21_history_reference":{"sha256":"5e3c190866504480f637cebb41e6417c0c0f9bf8689b1d22aa5289fa2123a92f","path":"verification/banc-natural-gait-m3-20260919/parent-e360-p22-ROOT-history-declaration-reference.json","bytes":5068},"historical_P22_policy":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p22-root-final-repair-policy.json","bytes":8421,"sha256":"d94f7dd1e89a90fbe033592ea9c9cafb2969d37f491c93157edf574f7bae3d5a"},"historical_P22_history_reference":{"bytes":6307,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p23-ROOT-history-declaration-reference-v2.json","sha256":"9b0bca8e6700bedcdb8272c179cd708d9fe4fb4299fa5c664f23f02b20f696fb"},"historical_P23_policy":{"sha256":"60681e6a2f1e44bfbccf47c269142c51382cc5d709891fb6626f0edd9962ebe2","path":"verification/banc-natural-gait-m3-20260919/parent-e360-p23-root-final-repair-policy.json","bytes":9040},"historical_P23_history_reference":{"bytes":5913,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p24-ROOT-history-declaration-reference.json","sha256":"00b25c369b46770fb6dafe3745f03cde68ed6a67654f99f82cea01984e64b206"},"historical_P25_policy":{"bytes":9672,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p25-root-final-repair-policy.json","sha256":"77d1fe0a4ac8b9486f5530a8cefba125eed5fc60f0e40c087ddb753ef8d7b6b5"},"descriptor_ownership":{"flat_history_catalogue":{"sha256":"ce69cf7e48a2c0f7b1dae18deffb784b05f09ae182ae4379059298e53a5e7743","bytes":216320,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p30-ROOT-flat-history-refined1180.json"},"complete_runtime_graph_certified":false,"ownership_before_membership":true,"history_aggregate_member_bytes":67260839,"independent_journal_both_subsets":true,"max_leaf_bytes":2097152,"dynamic_max_members":512,"table_miss_dynamic_fallback":false,"historical_max_members":1180,"schema":"flylab.m3.e360.p30.descriptor-ownership.v1","combined_max_consumed_members":1692,"catalogue_JSON_cap_bytes":262144,"unknown_historical_input":"POISON_AND_REJECT"},"historical_P27_policy":{"bytes":10788,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-repair-policy.json","sha256":"4b19ed927d9478bf0d9748e82152362e8d0c1244bbee3c8a7f2d6e2595ba348b"},"current_admission_kind":"successful-paid-predecessor-and-unpaid-Source-failure.v1","paid_predecessor_success":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-full-return.json","bytes":3348,"sha256":"f0975a24f4a46d2b08dfe1039d1f1c7b367e51926d3c1bf7d615911ed4b7efc7"},"predecessor_success_cap_s":3,"unpaid_Source_admission_failure":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-ROOT-scanner-native-admission-failure.json","bytes":2731,"sha256":"b10df60781733f43cca48bb883f16a29d53713c4e25728976046389fcaf15963"},"unpaid_Source_genuine_return":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-ROOT-scanner-admission-genuine-return.json","bytes":1606,"sha256":"18b3d770614589df82e562c4a6b799e476be4d62a11d5635c276e62d5be37291"},"unpaid_Source_first_error":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-ROOT-scanner-prepare-first-error.json","bytes":2386,"sha256":"b3d497c26da2d679ad8d5c79301a7b3b95c6446b6546efa73110989cea58a016"},"current_cost_identity":{"start_index":109,"pass_id":"e360-generous-p30"},"historical_P29_policy":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-repair-policy.json","bytes":12804,"sha256":"24bd4b5ba79e372443c9d57116b03df427099db553e795b1c1d63646138f047b"}}''')
CURRENT_HUMAN_AUTONOMY={'path':BASE+'/parent-e360-current-human-autonomy-20261006.json','bytes':1098,'sha256':'8c912aab85ab18fc668caeba9e67bf621318a933e4a4427ad525b0e7db035224'}
P16_RECOVERY_AMENDMENT=BASE+'/parent-e360-p16-recovery-amendment.json'
P16_RECOVERY_BYTES=108629
P16_RECOVERY_SHA='8733aaff9b8dbaa1f058661181a8a128b9eb948ef231c2b56616685478df2fe2'
P16_RECOVERY_FIELDS=('e360_p16_recovery_activation','e360_p16_recovery_amendment')
P15_RECOVERY_AMENDMENT=BASE+'/parent-e360-p15-recovery-amendment-v2-final.json'
P15_RECOVERY_BYTES=95430
P15_RECOVERY_SHA='88ae0799fda49d55e4d03b20bd9d1048c993f9cc6f8d1d9399be24320e42396c'
P15_RECOVERY_FIELDS=('e360_p15_recovery_activation','e360_p15_recovery_amendment')
P14_RECOVERY_AMENDMENT=BASE+'/parent-e360-p14-recovery-amendment.json'
P14_RECOVERY_BYTES=76640
P14_RECOVERY_SHA='25826341f1b25f0f4ffdc1f824159c8080f5c8ef7ee7c1cc1dff7e4ed22e5c41'
P14_RECOVERY_FIELDS=('e360_p14_recovery_activation','e360_p14_recovery_amendment')
P14_RECOVERY_SOURCE_KEYS=('actual_Astra_review','actual_Astra_review_md','actual_P12_baseline_native_preservation','actual_P13_baseline_native_preservation','actual_P14_baseline_native_preservation','actual_predecessor_failure','approved_final_proposal','capture_design_reflection','capture_reflected_design','concrete_correction_snapshot','existing_human_authorization','failed_stage_decision','historical_inherited82','historical_P10_capture_manifest','historical_P10_capture_module','historical_P10_genuine_capture_return','historical_P10_independent_owned_closure','historical_P11_actual_tool_return','historical_P11_capture_manifest','historical_P11_capture_module','historical_P11_closed_ledger88','historical_P11_contract','historical_P11_genuine_capture_return','historical_P11_inherited83','historical_P11_outer','historical_P11_ROOT_genuine_actual_returns','historical_P12_actual_request','historical_P12_actual_tool_return','historical_P12_capture_manifest','historical_P12_capture_module','historical_P12_closed_ledger89','historical_P12_final_source_freeze','historical_P12_full_return','historical_P12_genuine_capture_return','historical_P12_genuine_tool_output','historical_P12_genuine_tool_return','historical_P12_independent_owned_closure','historical_P12_inherited88','historical_P12_ROOT_genuine_actual_returns','historical_P13_closed_ledger91','historical_P13_final_source_freeze','historical_P13_independent_owned_closure','historical_P13_inherited89','historical_P13_ROOT_genuine_actual_returns','historical_P13_ruff_actual_request','historical_P13_ruff_actual_tool_return','historical_P13_ruff_capture_manifest','historical_P13_ruff_capture_module','historical_P13_ruff_full_return','historical_P13_ruff_genuine_capture_return','historical_P13_ruff_genuine_tool_return','historical_P13_scanner_actual_request','historical_P13_scanner_actual_tool_return','historical_P13_scanner_full_return','historical_P13_scanner_genuine_tool_return','historical_P13_scanner_report','inherited88','inherited89','inherited91','old_amendment','old_registration','previous_human_authorization','previous_revalidation_amendment','reflected_design','resource_extension','reviewed_design','reviewed_source_roster','ROOT_reflection','supersedes_unexecuted_P13_SOURCE_literal','supersedes_unexecuted_SOURCE_literal',)
P13_RECOVERY_AMENDMENT=BASE+'/parent-e360-p13-recovery-amendment-v2.json'
P13_RECOVERY_BYTES=62531
P13_RECOVERY_SHA='3d3c571723013be0ff94bc9ad01b88f27d52506062618e18f98faae2e92e3121'
P13_RECOVERY_FIELDS=('e360_p13_recovery_activation','e360_p13_recovery_amendment')
P13_RECOVERY_SOURCE_KEYS=('actual_Astra_review','actual_Astra_review_md','actual_P12_baseline_native_preservation','actual_P13_baseline_native_preservation','actual_predecessor_failure','approved_final_proposal','capture_design_reflection','capture_reflected_design','concrete_correction_snapshot','existing_human_authorization','failed_stage_decision','historical_inherited82','historical_P10_capture_manifest','historical_P10_capture_module','historical_P10_genuine_capture_return','historical_P10_independent_owned_closure','historical_P11_actual_tool_return','historical_P11_capture_manifest','historical_P11_capture_module','historical_P11_closed_ledger88','historical_P11_contract','historical_P11_genuine_capture_return','historical_P11_inherited83','historical_P11_outer','historical_P11_ROOT_genuine_actual_returns','historical_P12_actual_request','historical_P12_actual_tool_return','historical_P12_capture_manifest','historical_P12_capture_module','historical_P12_closed_ledger89','historical_P12_final_source_freeze','historical_P12_full_return','historical_P12_genuine_capture_return','historical_P12_genuine_tool_output','historical_P12_genuine_tool_return','historical_P12_independent_owned_closure','historical_P12_inherited88','historical_P12_ROOT_genuine_actual_returns','inherited88','inherited89','old_amendment','old_registration','previous_human_authorization','previous_revalidation_amendment','reflected_design','resource_extension','reviewed_design','reviewed_source_roster','ROOT_reflection','supersedes_unexecuted_P13_SOURCE_literal','supersedes_unexecuted_SOURCE_literal',)
P12_RECOVERY_AMENDMENT=BASE+'/parent-e360-p12-recovery-amendment-v1.json'
P12_RECOVERY_SHA='f12aa1947f3c541d3a2e018b38ddfe03d498881f8b59d5f9a074354dad2835a9'
P12_RECOVERY_FIELDS=('e360_p12_recovery_activation','e360_p12_recovery_amendment')
P12_RECOVERY_SOURCE_KEYS=('inherited88','historical_P11_inherited83','historical_P11_closed_ledger88',
    'historical_P11_ROOT_genuine_actual_returns','historical_P11_outer','historical_P11_actual_tool_return',
    'historical_P11_contract','historical_P11_capture_manifest','historical_P11_genuine_capture_return',
    'historical_P11_capture_module','actual_predecessor_failure','failed_stage_decision',
    'reviewed_design','actual_Astra_review','actual_Astra_review_md','ROOT_reflection','reflected_design',
    'existing_human_authorization','reviewed_source_roster','concrete_correction_snapshot',
    'approved_final_proposal','actual_P12_baseline_native_preservation')
P11_RECOVERY_AMENDMENT=BASE+'/parent-e360-p11-recovery-amendment-v2.json'
P11_RECOVERY_SHA='42aed77e288423e525e751a77a4f2a3145aca0bd22b7266ed2861040684563c9'
P11_RECOVERY_FIELDS=('e360_p11_recovery_activation','e360_p11_recovery_amendment')
P11_RECOVERY_SOURCE_KEYS=('inherited83','historical_inherited82','reviewed_design','actual_Astra_review',
    'actual_Astra_review_md','ROOT_reflection','reflected_design','existing_human_authorization',
    'failed_stage_decision','actual_predecessor_failure','concrete_correction_snapshot','reviewed_source_roster',
    'capture_design_reflection','capture_reflected_design','historical_P10_independent_owned_closure',
    'historical_P10_capture_manifest','historical_P10_genuine_capture_return','historical_P10_capture_module')
LINT_RECOVERY_SOURCE_KEYS=('inherited82','reviewed_design','actual_Astra_review','actual_Astra_review_md',
    'ROOT_reflection','reflected_design','existing_human_authorization','failed_stage_decision',
    'actual_predecessor_failure','concrete_correction_snapshot','reviewed_source_roster',
    'approved_final_proposal','previous_human_authorization','capture_design_reflection','capture_reflected_design')
SUITE_RECOVERY_SOURCE_KEYS=('inherited81','reviewed_design','actual_Astra_review','actual_Astra_review_md',
    'ROOT_reflection','reflected_design','existing_human_authorization','failed_stage_decision',
    'actual_predecessor_failure','concrete_correction_snapshot','reviewed_source_roster',
    'approved_final_proposal','previous_human_authorization','previous_unapproved_budget_proposal')
SUITE_RECOVERY_FIELDS=('e360_suite_recovery_activation','e360_suite_recovery_amendment','e360_suite_recovery_resource_grant')
CHECKPOINT_SCHEMA='flylab.m3.e360.execution-checkpoints.v1'
CHECKPOINT_STAGES=('pre_array_completed','diagnostics_started','synthetic_inputs_published',
    'diagnostics_recomputed','actual_imports_verified','diagnostics_published','before_envelope_publication')
CHECKPOINT_MAX_ENTRIES=16
CHECKPOINT_MAX_JSON_BYTES=16384
EXECUTION_BUDGET_SOURCE_KEYS=('inherited73','reviewed_design','actual_Astra_review','actual_Astra_review_md',
    'ROOT_reflection','reflected_design','existing_human_authorization','failed_stage_decision',
    'actual_predecessor_failure','concrete_correction_snapshot','reviewed_source_roster',
    'superseded_unexecuted_registration')
REVALIDATION_SOURCE_KEYS=('inherited59','reviewed_design','actual_Astra_review','actual_Astra_review_md',
                          'ROOT_reflection','existing_human_authorization','failed_stage_decision',
                          'actual_predecessor_failure','concrete_correction_snapshot')
ACTIVE_REVALIDATION_SOURCE_KEYS=('inherited60',*REVALIDATION_SOURCE_KEYS[1:])
SOURCE_RECOVERY_SOURCE_KEYS=('inherited62',*REVALIDATION_SOURCE_KEYS[1:])
RUFF_RECOVERY_SOURCE_KEYS=('inherited63',*REVALIDATION_SOURCE_KEYS[1:])
CURRENT_REVALIDATION_SOURCE_KEYS=('inherited68',*REVALIDATION_SOURCE_KEYS[1:],'reflected_design')
EXTENSION_SOURCE_KEYS=('inherited50','reviewed_design','direct_human_authorization','actual_Astra_review',
                       'Astra_review_md','ROOT_reflection','reflected_design')
ADMIN_CHECK=lambda:None
_PROJECT_CACHE={}
require,numeric,typed_equal=lib.require,lib.numeric,lib.typed_equal
strict_json,json_bytes,normalize,sha=lib.strict_json,lib.json_bytes,lib.normalize,lib.sha
safe,descriptor=lib.safe,lib.descriptor
retained,error_record,fail=lib.retained,lib.error_record,lib.fail
P29_CLOSED109_SPEC=json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-after-Ruff-closed-ledger109.json","bytes":212957,"sha256":"050f883629fcecefa2ceeb505bdabda5c183f74a045beb3382c7fec03dce4409"}''')
P29_DIRECT_PHYSICAL_SPECS=json.loads(r'''{"collector":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-ROOT-genuine-collector.txt","bytes":8138,"sha256":"df264afd41077daa57196076fff72fe31692977d3580c77c7984a70ac20b37d4"},"ROOT-collector-trace.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-ROOT-collector-trace.json","bytes":2852,"sha256":"2795c12eae68757d600eacef9f0669f582f68e550bf95fc7f0c5223c277cb8af"},"actual.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-actual.json","bytes":1129,"sha256":"bb2d5e3c76460c56394b38c84f8ca36cec5b0d7361d1e438c8e31f80f2a20036"},"stderr.txt":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-stderr.txt","bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},"Source2":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-source-account.ps1","bytes":714552,"sha256":"9162e88fae89de031119cdc4d2d14964e0acb0d0d7d24fcf5250cb6d981f494b"},"Native2":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-bookkeeping-native.ps1","bytes":16570,"sha256":"27b15befb84dbb84ebc616ac9de9cf278bcca72219e1d705c8a6d89f19e7dfe1"},"freeze":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-freeze.json","bytes":10168,"sha256":"8ee8d82b8fec0f9e8a85cc0acde08e19d4295893758298f2434314d620d4f210"},"Source0":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-source-tool.py","bytes":727578,"sha256":"00221b9bff24d3f3ad88fe5cd7067845ac6fa0e30152a104934a4b31c3b5f0ba"},"output.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-output.json","bytes":1709,"sha256":"7fa84b1a7afddb80f7353ed8267642d44579bde073178f4395f7954d54ccbdb9"},"Native3":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-ROOT-dispatch-adapter.ps1","bytes":14645,"sha256":"cf91be46d48ff60f8480f0643f1ffe26d76200cab147ddecd4d6e178779265e2"},"Source1":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-source-tests.py","bytes":474793,"sha256":"5481f5b9c7dfcfc79c5ec93ebc76a7e8455e046c8d4fa41f19bb7b1d63726fe1"},"Native1":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-prepare-native.ps1","bytes":7470,"sha256":"7621909a52e5dac7da6fa53c6fc3349e6374bd0f18f44e6ab1d6cc2084f352b8"},"policy":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-repair-policy.json","bytes":12804,"sha256":"24bd4b5ba79e372443c9d57116b03df427099db553e795b1c1d63646138f047b"},"full-return.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-full-return.json","bytes":3348,"sha256":"f0975a24f4a46d2b08dfe1039d1f1c7b367e51926d3c1bf7d615911ed4b7efc7"},"stdout.txt":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-stdout.txt","bytes":19,"sha256":"82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18"},"Native0":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-capture-proposed.ps1","bytes":52584,"sha256":"af98d1910192ba5c3174e4a45e7731e0fc0a88e83df2355775f9cea8b8ea888f"},"final-request.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-final-request.json","bytes":4355,"sha256":"6ff3cd1bf016e88404d8e6fb9387e158960675feedf7c859e0dab78c84c4733b"},"capture-return.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-capture-return.json","bytes":1404,"sha256":"80a5f316b7ac5df7effeb6c4845c12283e39f4facac000794a8d581bbcab6a58"},"Source3":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-source-helper.py","bytes":12782,"sha256":"5e345b2b9ed41c36b6e494315a9c64b2df817eca68c49aab71ecd85fe2c64eef"},"closed":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-after-Ruff-closed-ledger109.json","bytes":212957,"sha256":"050f883629fcecefa2ceeb505bdabda5c183f74a045beb3382c7fec03dce4409"}}''')
P30_FROZEN_POLICY_SPEC={"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p30-root-final-repair-policy.json","bytes":12370,"sha256":"2b7412ea54d929a0a89df8679356e9bf8b1b6c643184f6291c4a8649006d6dfb"}
P30_FROZEN_FREEZE_SPEC={"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p30-ROOT-scope-residual-final-freeze.json","bytes":12222,"sha256":"58bb716bc4fc713f77e26f3f9d98ad77b7e8b7428111c725103d0897357a1040"}
CURRENT_COST_IDENTITY=json.loads(r'''{"start_index":109,"pass_id":"e360-generous-p30"}''')
P29_ORIGINAL_POLICY_SPEC=json.loads(r'''{"sha256":"24bd4b5ba79e372443c9d57116b03df427099db553e795b1c1d63646138f047b","bytes":12804,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json"}''')
HISTORICAL_P29_FIELDS=json.loads(r'''["e360_p29_recovery_activation","e360_p29_recovery_amendment"]''')
P29_ACTUAL_PASS_METADATA=json.loads(r'''{"actual_failure":{"bytes":4637,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-full-return.json","sha256":"46325a4d70098ceb77fe90ed979afeb5ec32fd8fa1ee825defcea59e28b1fffb"},"next_index":1,"concrete_fix":[{"path":"docs/BANC_M3_E360_P29_NATIVE_CURRENT_POLICY_ROOT_REFLECTED_20261007.md","bytes":8670,"sha256":"7f30c86481593301451dbd75044c575fbde149ace981a65d57bee4d381acc99e"},{"path":"tools/verify_c_banc_m3_source_closure_diagnostic.py","bytes":727578,"sha256":"00221b9bff24d3f3ad88fe5cd7067845ac6fa0e30152a104934a4b31c3b5f0ba"},{"path":"tests/test_c_banc_source_closure_diagnostic.py","bytes":474793,"sha256":"5481f5b9c7dfcfc79c5ec93ebc76a7e8455e046c8d4fa41f19bb7b1d63726fe1"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-root-account.ps1","bytes":714552,"sha256":"9162e88fae89de031119cdc4d2d14964e0acb0d0d7d24fcf5250cb6d981f494b"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-root-metadata-helper.py","bytes":12782,"sha256":"5e345b2b9ed41c36b6e494315a9c64b2df817eca68c49aab71ecd85fe2c64eef"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-freeze.json","bytes":10168,"sha256":"8ee8d82b8fec0f9e8a85cc0acde08e19d4295893758298f2434314d620d4f210"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-capture-proposed.ps1","bytes":52584,"sha256":"af98d1910192ba5c3174e4a45e7731e0fc0a88e83df2355775f9cea8b8ea888f"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-prepare-native.ps1","bytes":7470,"sha256":"7621909a52e5dac7da6fa53c6fc3349e6374bd0f18f44e6ab1d6cc2084f352b8"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-bookkeeping-native.ps1","bytes":16570,"sha256":"27b15befb84dbb84ebc616ac9de9cf278bcca72219e1d705c8a6d89f19e7dfe1"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-ROOT-dispatch-adapter.ps1","bytes":14645,"sha256":"cf91be46d48ff60f8480f0643f1ffe26d76200cab147ddecd4d6e178779265e2"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-ROOT-genuine-collector.txt","bytes":8138,"sha256":"df264afd41077daa57196076fff72fe31692977d3580c77c7984a70ac20b37d4"}],"modified":true,"failed_cost":null,"revalidation_amendment":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json","bytes":12804,"sha256":"24bd4b5ba79e372443c9d57116b03df427099db553e795b1c1d63646138f047b"},"mode":"generated","failed_cap_s":0,"predecessor_failed_cap_s":3,"source_sha256":{"tools/verify_c_banc_m3_source_closure_diagnostic.py":"00221b9bff24d3f3ad88fe5cd7067845ac6fa0e30152a104934a4b31c3b5f0ba","tools/verify_c_banc_m3_startup_baseline.py":"4e853f8c4fae105102625816f3075d3b4b7d3ac7312506dd7b3e392e00607342","tools/verify_c_banc_m3_candidate_body_handoff.py":"5b43ee4ee34a8270ba123d11e99e9820b6f2d25644c027bc46a86ab9b6f7532a","tests/test_c_banc_candidate_body_handoff.py":"93476a2cb72e5bf7585d11ade9f76c207967eff8aadaf918198219bc4379cee6","verification/banc-natural-gait-m3-20260919/parent-e360-root-metadata-helper.py":"5e345b2b9ed41c36b6e494315a9c64b2df817eca68c49aab71ecd85fe2c64eef","verification/banc-natural-gait-m3-20260919/parent-e346-supervise-identities.py":"cf7b04731ed4ea8bb626047b577ae5b5b1e31cccd093415d450c839e998b52cd","tests/test_c_banc_source_closure_diagnostic.py":"5481f5b9c7dfcfc79c5ec93ebc76a7e8455e046c8d4fa41f19bb7b1d63726fe1","verification/banc-natural-gait-m3-20260919/parent-e267-supervise.py":"1d8682505f8060d3d8bcb50508811b9c4da531bdbf07c96200065bcc3324df8d","verification/banc-natural-gait-m3-20260919/parent-e360-root-account.ps1":"9162e88fae89de031119cdc4d2d14964e0acb0d0d7d24fcf5250cb6d981f494b"}}''')
P29_ACTUAL_RUFF_RECORD_SPECS=json.loads(r'''{"capture-return.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-capture-return.json","bytes":1404,"sha256":"80a5f316b7ac5df7effeb6c4845c12283e39f4facac000794a8d581bbcab6a58"},"final-request.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-final-request.json","bytes":4355,"sha256":"6ff3cd1bf016e88404d8e6fb9387e158960675feedf7c859e0dab78c84c4733b"},"ROOT-collector-trace.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-ROOT-collector-trace.json","bytes":2852,"sha256":"2795c12eae68757d600eacef9f0669f582f68e550bf95fc7f0c5223c277cb8af"},"stderr.txt":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-stderr.txt","bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},"actual.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-actual.json","bytes":1129,"sha256":"bb2d5e3c76460c56394b38c84f8ca36cec5b0d7361d1e438c8e31f80f2a20036"},"full-return.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-full-return.json","bytes":3348,"sha256":"f0975a24f4a46d2b08dfe1039d1f1c7b367e51926d3c1bf7d615911ed4b7efc7"},"stdout.txt":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-stdout.txt","bytes":19,"sha256":"82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18"},"output.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p29-ruff-output.json","bytes":1709,"sha256":"7fa84b1a7afddb80f7353ed8267642d44579bde073178f4395f7954d54ccbdb9"}}''')
P29_FROZEN_FREEZE_SPEC=json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-freeze.json","bytes":10168,"sha256":"8ee8d82b8fec0f9e8a85cc0acde08e19d4295893758298f2434314d620d4f210"}''')
P29_FROZEN_POLICY_SPEC=json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-root-final-repair-policy.json","bytes":12804,"sha256":"24bd4b5ba79e372443c9d57116b03df427099db553e795b1c1d63646138f047b"}''')
CURRENT_COST_IDENTITY['authority']=CURRENT_REPAIR_SPEC

FALSE_MILESTONES=lib.FALSE_MILESTONES
_AUTHORITY_SCOPE=None
_AUTHORITY_FENCE_STATE=None
_SUITE_DEADLINE=None
FIXED_READERS={
    'registration':(REGISTRATION,77417,REGISTRATION_SHA),
    'amendment':(AMENDMENT,3086,AMENDMENT_SHA),
    'resource_extension':(RESOURCE_EXTENSION,5461,RESOURCE_EXTENSION_SHA),
    'revalidation_amendment':(OLD_REVALIDATION_AMENDMENT,6282,OLD_REVALIDATION_SHA),
    'active_revalidation_amendment':(ACTIVE_REVALIDATION_AMENDMENT,8459,ACTIVE_REVALIDATION_SHA),
    'source_recovery_amendment':(SOURCE_RECOVERY_AMENDMENT,10441,SOURCE_RECOVERY_SHA),
    'ruff_recovery_amendment':(RUFF_RECOVERY_AMENDMENT,12323,RUFF_RECOVERY_SHA),
    'current_revalidation_amendment':(REVALIDATION_AMENDMENT,16202,REVALIDATION_SHA),
    'execution_budget_amendment':(EXECUTION_BUDGET_AMENDMENT,24334,EXECUTION_BUDGET_SHA),
    'suite_recovery_amendment':(SUITE_RECOVERY_AMENDMENT,28728,SUITE_RECOVERY_SHA),
    'lint_recovery_amendment':(LINT_RECOVERY_AMENDMENT,31323,LINT_RECOVERY_SHA),
    'p11_recovery_amendment':(P11_RECOVERY_AMENDMENT,33809,P11_RECOVERY_SHA),
    'p12_recovery_amendment':(P12_RECOVERY_AMENDMENT,50865,P12_RECOVERY_SHA),
    'p13_recovery_amendment':(P13_RECOVERY_AMENDMENT,P13_RECOVERY_BYTES,P13_RECOVERY_SHA),
    'p14_recovery_amendment':(P14_RECOVERY_AMENDMENT,P14_RECOVERY_BYTES,P14_RECOVERY_SHA),
    'p15_recovery_amendment':(P15_RECOVERY_AMENDMENT,P15_RECOVERY_BYTES,P15_RECOVERY_SHA),
    'p16_recovery_amendment':(P16_RECOVERY_AMENDMENT,P16_RECOVERY_BYTES,P16_RECOVERY_SHA),
    'p17_recovery_amendment':(P17_RECOVERY_AMENDMENT,P17_RECOVERY_BYTES,P17_RECOVERY_SHA)}


HISTORICAL_LITERAL_SPECS=tuple({'path':item[0],'bytes':item[1],'sha256':item[2]} for item in FIXED_READERS.values())
HISTORY_CATALOGUE_SPEC=json.loads(r'''{"sha256":"ce69cf7e48a2c0f7b1dae18deffb784b05f09ae182ae4379059298e53a5e7743","bytes":216320,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p30-ROOT-flat-history-refined1180.json"}''')
HISTORY_CATALOGUE_MEMBERS=1180
HISTORY_CATALOGUE_AGGREGATE_BYTES=67260839
_CURRENT_P30_DYNAMIC_PINS=tuple(json.loads(r'''[{"sha256":"2b7412ea54d929a0a89df8679356e9bf8b1b6c643184f6291c4a8649006d6dfb","bytes":12370,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json"},{"sha256":"5362d85a28b92efb89b0f66a14e173fa1d821e33acdde05a2fe214e9365f52d2","bytes":9656,"path":"docs/BANC_M3_E360_P30_CURRENT_COST_IDENTITY_ROOT_REFLECTED_20261007.md"},{"path":"docs/BANC_M3_E360_P30_CURRENT_COST_IDENTITY_ROOT_DESIGN_20261007.md","bytes":8824,"sha256":"68599f1318598c606f3cc21a0ed2410d432592a7fa553d0d0b745a23d136a4e4"},{"bytes":1098,"sha256":"8c912aab85ab18fc668caeba9e67bf621318a933e4a4427ad525b0e7db035224","path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-human-autonomy-20261006.json"},{"sha256":"ba7f61703cc4a2fbd606fdb4d623d7f1043b32db6dcc59f1589c18d967e1dde8","bytes":30887,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p30-actual-astra-review.json"},{"sha256":"d4299de1c24fb822dde3d44a11aa12bfbe4999293bb991c6d2e984f0df1f9987","bytes":4818,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p30-actual-astra-review.md"},{"sha256":"f6673785abb23e204031bf40bb9d734db0bbe2e289d90654fc5ffbdad86bdd72","bytes":1921,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p30-ROOT-actual-design-reflection.json"},{"sha256":"ce69cf7e48a2c0f7b1dae18deffb784b05f09ae182ae4379059298e53a5e7743","bytes":216320,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p30-ROOT-flat-history-refined1180.json"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-ROOT-scanner-native-admission-failure.json","bytes":2731,"sha256":"b10df60781733f43cca48bb883f16a29d53713c4e25728976046389fcaf15963"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-ROOT-scanner-admission-genuine-return.json","bytes":1606,"sha256":"18b3d770614589df82e562c4a6b799e476be4d62a11d5635c276e62d5be37291"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p29-ROOT-scanner-prepare-first-error.json","bytes":2386,"sha256":"b3d497c26da2d679ad8d5c79301a7b3b95c6446b6546efa73110989cea58a016"}]'''))

HISTORICAL_P25_FIELDS=json.loads(r'''["e360_p25_recovery_activation","e360_p25_recovery_amendment"]''')
P25_ACTUAL_RUFF_RECORD_SPECS=json.loads(r'''{"stderr.txt":{"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","bytes":0,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p25-ruff-stderr.txt"},"full-return.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p25-ruff-full-return.json","bytes":3355,"sha256":"e2052ff42ad0916a7b693a119ad03bf9dc07c0f223cf12b5c9df03e88dcddbda"},"stdout.txt":{"sha256":"82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18","bytes":19,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p25-ruff-stdout.txt"},"final-request.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p25-ruff-final-request.json","bytes":4340,"sha256":"c0d7e96b5db2385690ad04ad931d5431314f9944b281dfe5db47030986f046cb"},"ROOT-collector-trace.json":{"bytes":2861,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p25-ruff-ROOT-collector-trace.json","sha256":"e0136736c42fc1415375c93b1de226fb3bc187a8a1f309be6ef25418bbd2b0b2"},"output.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p25-ruff-output.json","bytes":1708,"sha256":"bfb25356f10b30285b2572feedbb04bbfb48b1015da9748d2fb0bbe696c9a1c2"},"genuine-capture-return.json":{"bytes":1404,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p25-ruff-genuine-capture-return.json","sha256":"0f424f68db989e11164ad03a309d5b557b399da38ba2746eba6a9a0a698331d1"},"actual.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p25-ruff-actual.json","bytes":1128,"sha256":"f4f009fdae3080349fecac9365570dc1cfbc7fe9b66a1d3f1c68b969988e45e3"}}''')
P25_ACTUAL_SCANNER_RECORD_SPECS=json.loads(r'''{"ROOT-collector-trace.json":{"sha256":"7c08bf89966356c5b2c86b1f02bea04168dccd472ed80db399346c0697585768","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p25-scan-ROOT-collector-trace.json","bytes":3808},"full-return.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p25-scan-full-return.json","bytes":2667,"sha256":"6b1b4db7d2e53b56ae40b36b8392cc2d5b2e2b5c6beae8a8cbb7bec13c87775d"},"actual.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p25-scan-actual.json","bytes":1128,"sha256":"88d60536e1d77aee08630af8f0c28962d1932da6e97e4422f398622c63830fcf"},"final-request.json":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p25-scan-final-request.json","bytes":2207,"sha256":"229cfea8a80cb7e09f3f84fe7371afc6eb9f6106ca26bebfb4afbe660710703e"}}''')
P25_CLOSED107_SPEC=json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p25-closed-ledger107.json","bytes":199728,"sha256":"097201fd2c51b8445d0415beea88248713233f127e581c5e3c7233ec9db66f43"}''')
P25_FROZEN_FREEZE_SPEC=json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p25-root-final-freeze.json","bytes":11390,"sha256":"253682c6af613f1da4531554d6eb7835c26afe14806be3fce810ebfcd7d6e23c"}''')
P25_FROZEN_POLICY_SPEC=json.loads(r'''{"sha256":"77d1fe0a4ac8b9486f5530a8cefba125eed5fc60f0e40c087ddb753ef8d7b6b5","path":"verification/banc-natural-gait-m3-20260919/parent-e360-p25-root-final-repair-policy.json","bytes":9672}''')
P25_ORIGINAL_POLICY_SPEC=json.loads(r'''{"sha256":"77d1fe0a4ac8b9486f5530a8cefba125eed5fc60f0e40c087ddb753ef8d7b6b5","path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json","bytes":9672}''')

HISTORICAL_P27_FIELDS=('e360_p27_recovery_activation','e360_p27_recovery_amendment')
P27_ORIGINAL_POLICY_SPEC=json.loads(r'''{"bytes":10788,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json","sha256":"4b19ed927d9478bf0d9748e82152362e8d0c1244bbee3c8a7f2d6e2595ba348b"}''')
P27_FROZEN_POLICY_SPEC=json.loads(r'''{"bytes":10788,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-repair-policy.json","sha256":"4b19ed927d9478bf0d9748e82152362e8d0c1244bbee3c8a7f2d6e2595ba348b"}''')
P27_FROZEN_FREEZE_SPEC=json.loads(r'''{"bytes":8435,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-freeze.json","sha256":"a73419cf3d3214c20a958e558cb8e0f397d1002345238f5ddf7747eaab8cd94a"}''')
P27_CLOSED108_SPEC=json.loads(r'''{"bytes":206426,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-closed-ledger108.json","sha256":"f0e8583414c05d5aa99d1395018b57b6392004245fb508f114d1dd71edf80c2f"}''')
P27_HELD108_SPEC=json.loads(r'''{"bytes":208673,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-admin-held-ledger108.json","sha256":"ae4ba34b038eb053f322ddc3591a230b2f67ef4cf05db182b989786a9a8a9f7a"}''')
P27_ACTUAL_RUFF_RECORD_SPECS=json.loads(r'''{"full-return.json":{"bytes":4637,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-full-return.json","sha256":"46325a4d70098ceb77fe90ed979afeb5ec32fd8fa1ee825defcea59e28b1fffb"},"actual.json":{"bytes":1128,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-actual.json","sha256":"f55b6cbe0c3f2c7601e34b5386ca84020bea748d298783168d8ee216c3fca4a6"},"final-request.json":{"bytes":4346,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-final-request.json","sha256":"3985723558efc9048a824581a24e8975e9028e83376af5a5173513c14da58040"},"ROOT-collector-trace.json":{"bytes":2930,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-ROOT-collector-trace.json","sha256":"fffadfe4c825817faad47b0c4f89f3624a1e37f7f3579de953883cebd490009e"},"capture-return.json":{"bytes":1472,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-capture-return.json","sha256":"d097693b5a8be6ed3e8be43c101ebeef2d257c0390cf053d544158f8d6ee1a36"},"output.json":{"bytes":1777,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-output.json","sha256":"ede0a43bea8130b6290a96cc92ea8728f7275dee024d865ea95ee5802347876c"},"stdout.txt":{"bytes":794,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-stdout.txt","sha256":"48935bce1d0b453fe77a4fcb148906068a743ab2f1cebd84bffb4f0fe973ecb2"},"stderr.txt":{"bytes":0,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}}''')
P27_DIRECT_PHYSICAL_SPECS=json.loads(r'''{"closed":{"bytes":206426,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-closed-ledger108.json","sha256":"f0e8583414c05d5aa99d1395018b57b6392004245fb508f114d1dd71edf80c2f"},"held":{"bytes":208673,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-admin-held-ledger108.json","sha256":"ae4ba34b038eb053f322ddc3591a230b2f67ef4cf05db182b989786a9a8a9f7a"},"decision":{"bytes":4979,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-closed-decision.json","sha256":"ef770af24a52710522d5c9033ba2db6e784de6922579afafcea63c53c8eae004"},"freeze":{"bytes":8435,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-freeze.json","sha256":"a73419cf3d3214c20a958e558cb8e0f397d1002345238f5ddf7747eaab8cd94a"},"policy":{"bytes":10788,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-repair-policy.json","sha256":"4b19ed927d9478bf0d9748e82152362e8d0c1244bbee3c8a7f2d6e2595ba348b"},"frozen_parent-e360-p27-root-final-source-tool.py":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-source-tool.py","sha256":"3f15eb959562556c8be693e2f976056324a061723261bc32af9051ded2b48577","bytes":690092},"frozen_parent-e360-p27-root-final-source-tests.py":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-source-tests.py","sha256":"40542d38bf5fd02f9e3aab57b06ee30e980463829d3af46f94f0e1a3ac4db0de","bytes":470707},"frozen_parent-e360-p27-root-final-source-account.ps1":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-source-account.ps1","sha256":"337d9ce30238c9e1bf3b7637c2f5ca905a83ce4eb90b7c50c0bf7e164253d090","bytes":677643},"frozen_parent-e360-p27-root-final-source-helper.py":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-source-helper.py","sha256":"f365e8c7c8a7c90c2bc3b0c20b2f4dc621231f0368297a94ad4f274bd6b903b7","bytes":12782},"native0":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-capture-proposed.ps1","sha256":"6fa1135265dcb7835622a8481b9daaf7c494cace3d9e3c89be70d14d8325368b","bytes":49908},"native1":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-prepare-native.ps1","sha256":"0028d1f9f0e06071deb379f3b308585fc9b93067e0e015cef30f16eb6007a430","bytes":7473},"native2":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-bookkeeping-native.ps1","sha256":"c89fe383b9b4f91552e6ba1842702bf8c7c851d761ab92c16a08e763d62f283a","bytes":16059},"native3":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-dispatch-adapter.ps1","sha256":"1b7eef4c99029308f64b66c364d1f53832710475659b561f74cdb54f4db5678d","bytes":14644},"collector":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-genuine-collector.txt","sha256":"46debd6e685014559b6b658686c55f7cfa529fdc834ee56d32b359c5ea69ed36","bytes":8138},"cost":{"bytes":4637,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-full-return.json","sha256":"46325a4d70098ceb77fe90ed979afeb5ec32fd8fa1ee825defcea59e28b1fffb"},"actual":{"bytes":1128,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-actual.json","sha256":"f55b6cbe0c3f2c7601e34b5386ca84020bea748d298783168d8ee216c3fca4a6"},"request":{"bytes":4346,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-final-request.json","sha256":"3985723558efc9048a824581a24e8975e9028e83376af5a5173513c14da58040"},"trace":{"bytes":2930,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-ROOT-collector-trace.json","sha256":"fffadfe4c825817faad47b0c4f89f3624a1e37f7f3579de953883cebd490009e"},"capture":{"bytes":1472,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-capture-return.json","sha256":"d097693b5a8be6ed3e8be43c101ebeef2d257c0390cf053d544158f8d6ee1a36"},"manifest":{"bytes":1777,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-output.json","sha256":"ede0a43bea8130b6290a96cc92ea8728f7275dee024d865ea95ee5802347876c"},"stdout":{"bytes":794,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-stdout.txt","sha256":"48935bce1d0b453fe77a4fcb148906068a743ab2f1cebd84bffb4f0fe973ecb2"},"stderr":{"bytes":0,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},"prepared":{"bytes":16439,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ruff-hold-VERIFIED-NOT-COMMITTED.json","sha256":"22ff2ce8f49248450250a0262af39357641ceeabb79c9d9cfa9ad92819fbe71f"},"result":{"bytes":1712,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-ruff-hold-reconciliation-result.json","sha256":"9e03e4a5e8b77d01a211fab670c9d550df67bbff83b94ab6c71d77d365dd9901"},"genuine_return":{"bytes":849,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-ruff-hold-genuine-return.json","sha256":"f7d14957963d99d01ca46047287e3a3a36d71f19f9abfc76ee6620897fceb509"},"operation":{"bytes":37514,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-ruff-hold-reconcile-native.ps1","sha256":"fba8f48fc5654d62066747e4a10cf5a1902c272a6a86eed0c19a786dbd013586"},"operation_freeze":{"bytes":2865,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ruff-hold-ROOT-final-operation-freeze.json","sha256":"268741df198dcb723f3ff533966e1c763c8155cdef2b1000138c9497bc938164"}}''')
P27_CLOSURE_METADATA=json.loads(r'''{"input_descriptors":[{"path":"docs/BANC_M3_E360_P27_RUFF_HOLD_RECONCILIATION_ROOT_DESIGN_20261007.md","sha256":"e065676dbb4696a0895d869835f8fd5bb2d71d219fdc0a9e3f49e22ba16f65c1","bytes":7857},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-admin-held-ledger108.json","sha256":"ae4ba34b038eb053f322ddc3591a230b2f67ef4cf05db182b989786a9a8a9f7a","bytes":208673},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-source-account.ps1","sha256":"337d9ce30238c9e1bf3b7637c2f5ca905a83ce4eb90b7c50c0bf7e164253d090","bytes":677643},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-freeze.json","sha256":"a73419cf3d3214c20a958e558cb8e0f397d1002345238f5ddf7747eaab8cd94a","bytes":8435},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-repair-policy.json","sha256":"4b19ed927d9478bf0d9748e82152362e8d0c1244bbee3c8a7f2d6e2595ba348b","bytes":10788},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-generated-repair-policy.json","sha256":"4b19ed927d9478bf0d9748e82152362e8d0c1244bbee3c8a7f2d6e2595ba348b","bytes":10788},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-current-human-autonomy-20261006.json","sha256":"8c912aab85ab18fc668caeba9e67bf621318a933e4a4427ad525b0e7db035224","bytes":1098},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-capture-return.json","sha256":"d097693b5a8be6ed3e8be43c101ebeef2d257c0390cf053d544158f8d6ee1a36","bytes":1472},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-output.json","sha256":"ede0a43bea8130b6290a96cc92ea8728f7275dee024d865ea95ee5802347876c","bytes":1777},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-final-request.json","sha256":"3985723558efc9048a824581a24e8975e9028e83376af5a5173513c14da58040","bytes":4346},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-full-return.json","sha256":"46325a4d70098ceb77fe90ed979afeb5ec32fd8fa1ee825defcea59e28b1fffb","bytes":4637},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-actual.json","sha256":"f55b6cbe0c3f2c7601e34b5386ca84020bea748d298783168d8ee216c3fca4a6","bytes":1128},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-ROOT-collector-trace.json","sha256":"fffadfe4c825817faad47b0c4f89f3624a1e37f7f3579de953883cebd490009e","bytes":2930},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-stdout.txt","sha256":"48935bce1d0b453fe77a4fcb148906068a743ab2f1cebd84bffb4f0fe973ecb2","bytes":794},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p27-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","bytes":0},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-genuine-collector.txt","sha256":"46debd6e685014559b6b658686c55f7cfa529fdc834ee56d32b359c5ea69ed36","bytes":8138},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-admin-ruff-genuine-return.json","sha256":"075a1cec7697bbeaaa23181ea3e9007b176d0f754a8f47c622245dd7289086d5","bytes":1012},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-ruff-hold-design-inputs.json","sha256":"7b973dfa05ed070eb0146feca9d819e3b8aad36e246a2654a6dd6f9d638e9e82","bytes":3465},{"path":"docs/BANC_M3_E360_P27_RUFF_HOLD_ROOT_REFLECTED_20261007.md","sha256":"c9ef96ad87f5ea207370686098c8f25ac6685284b129d040f530fdb49ceeb0b6","bytes":12685},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ruff-hold-actual-astra-review.json","sha256":"f73b8043d74c00a35ad1217a4d545315d1fe00427c068620a0ddcc70d2e9f18c","bytes":23109},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ruff-hold-actual-astra-review.md","sha256":"b638e159a08215d95525e5729ccc78af479270e07b727451f61febac254d1f38","bytes":4007},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-actual-ruff-hold-design-reflection.json","sha256":"8c18979ec9eb1a00bc013837010a8a17202bea93e8d745936ef1b85565c978e4","bytes":13569},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-ruff-hold-preapply-baseline.json","sha256":"690548e1542f8adf14193bbaddc12de8598aa678c7086aacceee70910c838e93","bytes":3103},{"sha256":"3f15eb959562556c8be693e2f976056324a061723261bc32af9051ded2b48577","path":"tools/verify_c_banc_m3_source_closure_diagnostic.py","bytes":690092},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-source-tool.py","sha256":"3f15eb959562556c8be693e2f976056324a061723261bc32af9051ded2b48577","bytes":690092},{"sha256":"40542d38bf5fd02f9e3aab57b06ee30e980463829d3af46f94f0e1a3ac4db0de","path":"tests/test_c_banc_source_closure_diagnostic.py","bytes":470707},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-source-tests.py","sha256":"40542d38bf5fd02f9e3aab57b06ee30e980463829d3af46f94f0e1a3ac4db0de","bytes":470707},{"sha256":"337d9ce30238c9e1bf3b7637c2f5ca905a83ce4eb90b7c50c0bf7e164253d090","path":"verification/banc-natural-gait-m3-20260919/parent-e360-root-account.ps1","bytes":677643},{"sha256":"f365e8c7c8a7c90c2bc3b0c20b2f4dc621231f0368297a94ad4f274bd6b903b7","path":"verification/banc-natural-gait-m3-20260919/parent-e360-root-metadata-helper.py","bytes":12782},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-root-final-source-helper.py","sha256":"f365e8c7c8a7c90c2bc3b0c20b2f4dc621231f0368297a94ad4f274bd6b903b7","bytes":12782},{"sha256":"1d8682505f8060d3d8bcb50508811b9c4da531bdbf07c96200065bcc3324df8d","path":"verification/banc-natural-gait-m3-20260919/parent-e267-supervise.py","bytes":3576},{"sha256":"93476a2cb72e5bf7585d11ade9f76c207967eff8aadaf918198219bc4379cee6","path":"tests/test_c_banc_candidate_body_handoff.py","bytes":23102},{"sha256":"4e853f8c4fae105102625816f3075d3b4b7d3ac7312506dd7b3e392e00607342","path":"tools/verify_c_banc_m3_startup_baseline.py","bytes":90298},{"sha256":"cf7b04731ed4ea8bb626047b577ae5b5b1e31cccd093415d450c839e998b52cd","path":"verification/banc-natural-gait-m3-20260919/parent-e346-supervise-identities.py","bytes":5115},{"sha256":"5b43ee4ee34a8270ba123d11e99e9820b6f2d25644c027bc46a86ab9b6f7532a","path":"tools/verify_c_banc_m3_candidate_body_handoff.py","bytes":94828},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-capture-proposed.ps1","sha256":"6fa1135265dcb7835622a8481b9daaf7c494cace3d9e3c89be70d14d8325368b","bytes":49908},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-prepare-native.ps1","sha256":"0028d1f9f0e06071deb379f3b308585fc9b93067e0e015cef30f16eb6007a430","bytes":7473},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-bookkeeping-native.ps1","sha256":"c89fe383b9b4f91552e6ba1842702bf8c7c851d761ab92c16a08e763d62f283a","bytes":16059},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-dispatch-adapter.ps1","sha256":"1b7eef4c99029308f64b66c364d1f53832710475659b561f74cdb54f4db5678d","bytes":14644},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ROOT-first-error-boundary.ps1","sha256":"523969c43f9951743e48e6b71cf85a414a9f6e47c56197fbc7ef212355826056","bytes":3881},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-approved-generous-registration-v2.json","bytes":5461,"sha256":"ea5afdffb09f6a118a18e3039546a911bbc9ed1465cafb6a1fe472ce3471862b"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p17-recovery-amendment.json","bytes":121989,"sha256":"569ed3e5bbbb998c7410dad8372af45e0526a802a62bb6b4e42d5137c341b295"},{"path":"GOAL_PLAN.md","sha256":"16898e461fede082206f9584c60b3fb11d994a7ab708cdfb19f950e35126eace","bytes":151552}],"actual_Sol_FIRST":{"sha256":"14102f230ec69c076cca9ec4a0f912dfad1cce7fa3fd43fd9db28ebe6ce84464","path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ruff-hold-sol-FIRST-handoff.json","bytes":12420},"actual_ROOT_FIRST":{"sha256":"7341cbcd28f33e0422ef2ed6ad3d31c8c9983a223d87626064894d8c00bce8ec","bytes":15105,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ruff-hold-ROOT-FIRST-actual-review.json"},"actual_ROOT_SECOND":{"sha256":"34cdc36e595ff40b9783c4cdd0020e8b3f51d86a8c850078f7e5a2065b54122b","bytes":13807,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p27-ruff-hold-ROOT-SECOND-actual-review.json"}}''')

def execution_history_bytes(spec,root):
    if spec['bytes']:
        return read_spec(spec,root,owner='historical')
    require(type(spec['bytes']) is int and Path(spec['path']).suffix=='.txt'
        and spec['sha256']==sha(b''),'exact completed empty historical stream')
    path=safe(spec['path'],root)
    require(path.read_bytes()==b'' and safe(spec['path'],root).read_bytes()==b'','completed empty stream reread')
    boundary=_AUTHORITY_SCOPE if _AUTHORITY_SCOPE is not None else _AUTHORITY_FENCE_STATE
    if boundary is not None:boundary._consumed(spec,root,owner='historical',allow_empty=True)
    return b''


def execution_archived_context(spec,record,root):
    typed_equal(strict_json(execution_history_bytes(spec,root)),record,'exact sealed context declaration')
    aliases={}
    for row in record['aliases']:
        require(type(row) is dict and set(row)=={'original','physical'},'exact historical alias')
        original,physical=row['original'],row['physical']
        require(type(original) is dict and set(original)=={'path','bytes','sha256'},'typed original logical descriptor')
        for key in ('bytes','sha256'):typed_equal(original[key],physical[key],'same-byte immutable historical alias')
        require(original['path']!=physical['path'] and original['path'] not in aliases,'unique full logical alias')
        execution_history_bytes(physical,root)
        aliases[original['path']]=(original,physical)
    for leaf in record['inputs']:execution_history_bytes(leaf,root)
    def archived(logical):
        require(logical['path'] in aliases,'explicit selected historical alias')
        original,physical=aliases[logical['path']]
        typed_equal(logical,original,'full original historical descriptor')
        return strict_json(execution_history_bytes(physical,root))
    admission=archived(record['admission']);manifest=archived(record['manifest']);freeze=archived(record['source_freeze'])
    typed_equal(admission,{'schema':'flylab.m3.execution.admission.v1','state':'ADMITTED',
        'manifest':record['manifest'],'rules':EXECUTION_RULE_SPEC,'source_freeze':record['source_freeze'],
        'anchor':manifest['closed_anchor']},'original admitted historical context')
    typed_equal(manifest['pass_id'],record['pass_id'],'historical pass declaration')
    typed_equal(manifest['source_freeze'],record['source_freeze'],'historical freeze binding')
    typed_equal(manifest['rules'],EXECUTION_RULE_SPEC,'unchanged historical fixed rules')
    typed_equal(manifest['human_authorization'],CURRENT_HUMAN_AUTONOMY,'retained historical human delegation')
    typed_equal(manifest['review_pins'],EXECUTION_REVIEW_PINS,'actual historical review pins')
    require(type(manifest['start_index']) is int and manifest['start_index']>=111,'typed historical start index')
    for key,specification in EXECUTION_REVIEW_PINS.items():typed_equal(freeze[key],specification,'historical actual review')
    require(freeze['ROOT_review_cycle_complete'] is True and len(freeze['source_copies'])==4 and len(freeze['own_sources_sha256'])==9,'historical Source4/own9')
    for row in freeze['source_copies']:
        original,physical=aliases[row['current']['path']]
        typed_equal(original,row['current'],'actual historical Source4 descriptor')
        for key in ('bytes','sha256'):typed_equal(row['current'][key],row['frozen'][key],'historical Source snapshot')
        typed_equal(physical['sha256'],freeze['own_sources_sha256'][original['path']],'actual old Source hash')
    for name,digest in freeze['own_sources_sha256'].items():
        if name in aliases:typed_equal(aliases[name][0]['sha256'],digest,'historical own Source alias')
        else:
            leaf=next((value for value in record['inputs'] if value['path']==name),None)
            require(leaf is not None,'explicit unchanged historical own Source')
            typed_equal(leaf['sha256'],digest,'unchanged own5 hash')
    for leaf in (*freeze['native_sources'],freeze['collector_source'],freeze['capture_caller'],
                 freeze['commit_caller'],freeze['first_error_boundary'],freeze['fixed_native_context']):
        require(leaf['path'] in aliases,'explicit old Native/caller alias')
        typed_equal(aliases[leaf['path']][0],leaf,'exact historical Native/caller Source')
    return {'manifest':manifest,'manifest_spec':record['manifest'],'admission_spec':record['admission'],
        'freeze':freeze,'rules':strict_json(read_spec(EXECUTION_RULE_SPEC,root,owner='dynamic')),
        'root':Path(root).resolve(),'historical':True}


def execution_sealed_anchor(context,root):
    """Flat actual postfloor closure; never execute historical code or replay old policy."""
    anchor=context['anchor'];floor=strict_json(read_spec(EXECUTION_ANCHOR_SPEC,root,owner='historical'))
    pair={'execution_context_activation','execution_context_manifest'}
    mutable={'debits','e360_calls','e360_passes','generated_shared_cumulative_used_s','generated_shared_remaining_s',
        'generated_stage_used_s','e360_generated_local_used_s','generated_new_stage_retained_bytes','e360_budget_exceeded','e360_phase'}
    require(set(anchor)==set(floor) or set(anchor)==set(floor)|pair,'finite sealed anchor fields')
    for key in set(floor)-mutable:typed_equal(anchor[key],floor[key],'unchanged preservation floor '+key)
    typed_equal(anchor['debits'][:111],floor['debits'],'all original ordered floor costs')
    for key in ('e360_calls','e360_passes'):
        for identity,value in floor[key].items():typed_equal(anchor[key].get(identity),value,'unchanged floor state')
    require(all(not anchor[key] for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'))
        and anchor['original_jobs_running']==0 and anchor['original_reserved_s']==0,'sealed ownership0')
    histories={}
    for spec,record in context['sealed_records']:
        require(record['pass_id'] not in histories,'unique historical context')
        histories[record['pass_id']]=execution_archived_context(spec,record,root)
    typed_equal(set(histories),set(anchor['e360_passes'])-set(floor['e360_passes']),'exact postfloor context roster')
    new_calls={identity:call for identity,call in anchor['e360_calls'].items() if identity not in floor['e360_calls']}
    require(all(call['pass_id'] in histories and call['state']=='charged' for call in new_calls.values()),'closed postfloor calls')
    seen={debit['cost_id'] for debit in floor['debits']};charged=set();failures={};last_context=None
    for index,debit in enumerate(anchor['debits'][111:],111):
        require(debit['cost_id'] not in seen,'once sealed cost');seen.add(debit['cost_id'])
        cost=charged_cost(debit['descriptor'],'generated',root)
        require(cost['schema']=='flylab.m3.e360.administrative-full-return.v1','finite sealed administrative extension')
        historical=histories.get(cost['pass_id']);require(historical is not None,'exact sealed cost context')
        call=new_calls.get(cost['call_id']);require(call is not None and call['state']=='charged','actual sealed cost-backed call')
        require(index>=historical['manifest']['start_index'] and cost['call_id'] not in charged,'typed historical cost interval')
        request=strict_json(execution_history_bytes(call['request'],root))
        execution_request(request,historical,root)
        actual=strict_json(execution_history_bytes(cost['actual_tool_return'],root))
        for item in (cost,actual,request,call):
            typed_equal(item['pass_id'],historical['manifest']['pass_id'],'actual historical pass')
            typed_equal(item['revalidation_amendment'],historical['manifest_spec'],'original historical manifest')
            typed_equal(item['resource_extension'],historical['manifest']['resource_extension'],'retained historical resource authority')
        for key in ('mode','cost_id','actual_outer_full_return_s'):typed_equal(cost[key],debit[key],'sealed cost/debit linkage')
        for item in (cost,actual):
            typed_equal(item['actual_outer_full_return_s'],debit['actual_outer_full_return_s'],'original sealed full return')
            typed_equal(item['actual_returned_exit_code'],debit['actual_returned_exit_code'],'original sealed exit')
            typed_equal(item['request'],call['request'],'original sealed request')
            typed_equal(item['source_sha256'] if 'source_sha256' in item else cost['source_sha256'],historical['freeze']['own_sources_sha256'],'old own9')
        typed_equal(cost['cost_id'],cost['call_id'],'administrative once identity')
        typed_equal(call['cost_id'],cost['cost_id'],'cost-backed historical call')
        require(cost['charged_once'] is True and type(debit['actual_outer_full_return_s']) is float and type(debit['actual_returned_exit_code']) is int,'typed sealed cost/exit')
        role_index=historical['rules']['primary_roles'].index(call['call_role'])
        typed_equal(call['sequence_index'],role_index,'sealed role order')
        typed_equal(call['reserved_s'],historical['rules']['primary_caps_s'][role_index],'old cap')
        typed_equal(call['future_reserved_s'],historical['rules']['future_with_unused_fallback120_s'][role_index],'old future')
        proof=p19_collector_proof(actual['collector_trace'],request,call['request'],debit['actual_outer_full_return_s'],debit['actual_returned_exit_code'],root=root,version='execution',selected_context=historical)
        typed_equal(proof['valid'],actual['collector_proof_valid'],'original historical collector')
        typed_equal(cost['collector_proof_valid'],proof['valid'],'historical cost collector')
        if call['call_role']=='ruff':
            summary=strict_json(execution_history_bytes(cost['capture_return'],root));capture=strict_json(execution_history_bytes(cost['expected_output'],root))
            typed_equal(summary['manifest_descriptor'],cost['expected_output'],'historical capture descriptor')
            typed_equal(capture['request'],call['request'],'historical capture request')
            typed_equal(capture['revalidation_amendment'],historical['manifest_spec'],'historical capture authority')
            for key in ('process_identity','streams','owned_state','ruff_exit_code','capture_complete','first_failure'):
                typed_equal(summary[key],capture[key],'original opaque historical capture')
            require(type(summary['process_identity']['start_utc_ticks']) is int and type(summary['process_identity']['process_id']) is int,'original Native opaque integer identity')
            valid=summary['capture_complete'] and summary['publication_verified'] and summary['exit_observed'] and proof['valid']
            typed_equal(cost['internal_completion_valid'],valid,'actual historical capture validity')
        elif call['call_role']=='scanner':
            typed_equal(cost['internal_completion_valid'],proof['valid'],'actual scanner transport completion remains separate from exit')
            require(cost['expected_output'] is None and debit['actual_returned_exit_code']!=0,'actual incomplete sealed scanner')
        else:
            raise ValueError('No unimplemented sealed role admission')
        if debit['actual_returned_exit_code']!=0 or debit['actual_outer_full_return_s']>call['reserved_s'] or not cost['internal_completion_valid']:
            failures[cost['pass_id']]=(debit['descriptor'],call['reserved_s'])
        charged.add(cost['call_id']);last_context=historical
    typed_equal(charged,set(new_calls),'every sealed call charged once')
    for identity,historical in histories.items():
        passed=anchor['e360_passes'][identity];calls=[call for call in new_calls.values() if call['pass_id']==identity]
        typed_equal(passed['next_index'],len(calls),'sealed pass roster')
        typed_equal(passed['source_sha256'],historical['freeze']['own_sources_sha256'],'sealed pass old own9')
        typed_equal(passed['revalidation_amendment'],historical['manifest_spec'],'sealed pass authority')
        typed_equal(passed['execution_admission'],historical['admission_spec'],'sealed pass admission')
        require(identity in failures,'terminal actual failed pass')
        typed_equal(passed['failed_cost'],failures[identity][0],'actual sealed failed cost retained')
        typed_equal(passed['failed_cap_s'],failures[identity][1],'actual sealed failed cap retained')
    if last_context is not None:
        typed_equal(anchor['execution_context_manifest'],last_context['manifest_spec'],'previous exact context pair')
        typed_equal(anchor['execution_context_activation'],{'schema':'flylab.m3.execution.activation.v1',
            'admission':last_context['admission_spec'],'manifest':last_context['manifest_spec'],
            'closed_anchor':last_context['manifest']['closed_anchor'],'new_generated_grant_s':0},'previous exact activation')
    reg=strict_json(read_spec(context['manifest']['registration'],root,owner='historical'))
    total=round(math.fsum(d['actual_outer_full_return_s'] for d in anchor['debits'] if d['mode']=='generated'),9)
    local=round(math.fsum(d['actual_outer_full_return_s'] for d in anchor['debits'][42:] if d['mode']=='generated'),9)
    for key,value in (('generated_stage_used_s',total),('e360_generated_local_used_s',local),
        ('generated_shared_cumulative_used_s',reg['generated_shared_baseline_s']+total),
        ('generated_shared_remaining_s',1700-reg['generated_shared_baseline_s']-total)):
        typed_equal(anchor[key],value,'original sealed scalar arithmetic')
    require(anchor['e360_budget_exceeded'] is False and total+reg['generated_shared_baseline_s']<=1700 and local<=1589,'retained sealed budget')
    require(type(anchor['generated_new_stage_retained_bytes']) is int and 0<=anchor['generated_new_stage_retained_bytes']<=16777216,'sealed retained bytes')
    context['historical_contexts']=histories


def execution_context(root=ROOT):
    """Fresh exact data selection with the existing cold physical fences."""
    context=execution_bootstrap(root,ADMIN_CHECK)
    if context is None:return None
    manifest=context['manifest'];freeze=context['freeze']
    for spec in (context['admission_spec'],context['manifest_spec'],manifest['rules'],EXECUTION_DIRECT_INPUT_SPEC,manifest['source_freeze'],manifest['human_authorization'],manifest['history_catalogue']):
        read_spec(spec,root,owner='dynamic')
        boundary=_AUTHORITY_SCOPE if _AUTHORITY_SCOPE is not None else _AUTHORITY_FENCE_STATE
        if boundary is not None:boundary._consumed(spec,root,owner='dynamic')
    typed_equal(manifest['human_authorization'],CURRENT_HUMAN_AUTONOMY,'same general human delegation')
    for spec in manifest['historical_inputs']:
        if spec['bytes']==0:historical_p12_stream_bytes(spec,root)
        else:read_spec(spec,root,owner='historical')
    for row in freeze['source_copies']:
        typed_equal(row['current'],file_spec(row['current']['path'],root,owner='dynamic'),'current Source4 physical descriptor')
        for key in ('bytes','sha256'):typed_equal(row['current'][key],row['frozen'][key],'same byte Source snapshot')
        read_spec(row['frozen'],root,owner='dynamic')
    typed_equal(freeze['own_sources_sha256'],own_sources(root),'actual current own9')
    for spec in freeze['native_sources']:
        read_spec(spec,root,owner='dynamic')
    require(len(freeze['native_sources'])==4,'ordered Native4')
    typed_equal(freeze['native_request_scripts'],freeze['native_sources'][1:],'request Native3')
    for key in ('collector_source','capture_caller','commit_caller','first_error_boundary','fixed_native_context'):
        read_spec(freeze[key],root,owner='dynamic')
    anchor=strict_json(read_spec(context['anchor_spec'],root,owner='dynamic'))
    require(len(anchor['debits'])==manifest['start_index'] and len(anchor['e360_calls'])==manifest['start_calls'] and len(anchor['e360_passes'])==manifest['start_passes'],'actual sealed roster')
    require(manifest['pass_id'] not in anchor['e360_passes'],'fresh context cannot reopen a historical pass')
    context['anchor']=anchor
    context['root']=Path(root).resolve()
    execution_sealed_anchor(context,root)
    return context

def execution_current_pass(root=ROOT):
    context=execution_bootstrap(root,ADMIN_CHECK)
    return 'e360-generous-p30' if context is None else context['manifest']['pass_id']

def execution_is_current(pass_id,root=ROOT):
    context=execution_bootstrap(root,ADMIN_CHECK)
    return context is not None and pass_id==context['manifest']['pass_id']

def execution_policy(context):
    rules=copy.deepcopy(context['rules'])
    helpers=rules['effective_helper_limits']
    for role in helpers:
        helpers[role]['successful_suffix_after_s']=rules['future_with_unused_fallback120_s'][rules['primary_roles'].index(role)]
    return {**rules,'helper_limits':helpers,'generated_call_order':rules['primary_roles'],
        'successful_suffix_after_s':rules['future_with_unused_fallback120_s'],
        'remaining_generated_administration_reserve_s':rules['primary_admin_total_s'],
        'fixture_future_administration_reserved_s':125,'suite_future_administration_reserved_s':65,
        'separate_same_pending_fallback_reserved_s':120,'helper_request_cap_bytes':262144}

def execution_request(request,context,root,*,held=False,allow_pending=False):
    manifest=context['manifest'];rules=context['rules'];freeze=context['freeze']
    require(type(request) is dict and request.get('schema')=='flylab.m3.e360.helper-request.v1','current request schema')
    typed_equal(request['pass_id'],manifest['pass_id'],'one data current pass')
    typed_equal(request['mode'],'generated','Original authority remains false')
    role=request['call_role'];require(role in rules['primary_roles'],'finite generated role')
    labels={'ruff':'ruff','scanner':'scan','fixture_build':'fbuild','fixture_worker':'fworker','fixture_seal':'fseal','suite_build':'sbuild','suite_worker':'sworker','suite_seal':'sseal','qualification':'q'}
    typed_equal(request['call_id'],manifest['pass_id']+'-'+labels[role],'exact finite role call identity')
    typed_equal(request['revalidation_amendment'],context['manifest_spec'],'request admitted manifest')
    typed_equal(request['resource_extension'],manifest['resource_extension'],'resource authority unchanged')
    typed_equal(request['source_sha256'],freeze['own_sources_sha256'],'request exact own9')
    typed_equal(request['collector_source'],freeze['collector_source'],'current collector pin')
    typed_equal(request['execution_admission'],context['admission_spec'],'request admission record')
    for key in ('expected_output','completion'):
        typed_equal(request[key],manifest['role_paths'][role][key],'fixed current '+key)
        safe(request[key],root)
    typed_equal(request.get('_request_path',manifest['role_paths'][role]['request']),manifest['role_paths'][role]['request'],'role request path')
    helper=rules['effective_helper_limits'].get(role)
    if helper is not None:
        for key,expected in (('action',helper['action']),('role',helper['worker_role'])):typed_equal(request[key],expected,'fixed helper '+key)
    else:
        typed_equal(request['action'],role,'finite external action')
        if role.endswith('_worker'):typed_equal(request['role'],role[:-7],'worker role')
    if role.startswith(('fixture_','suite_')):
        typed_equal(request['attempt_id'],manifest['attempts'][role.split('_')[0]],'finite attempt')
        if role.endswith(('_worker','_seal')):
            worker=role.split('_')[0]
            typed_equal(request['contract'],names(manifest['attempts'][worker],worker)['contract'],'exact admitted worker contract path')
    if held and role not in ('ruff','scanner'):
        typed_equal(strict_json(read_spec(request['closure_report'],root,owner='dynamic')),scan_current(root=root),'real current AST closure')
    if held:
        _,sem,_,_=execution_ledger_state(context,root)
        call=sem['e360_calls'].get(request['call_id'])
        require(call is not None and call['state']==('actual_pending' if allow_pending else 'begun'),'actual prospectively admitted call')
        typed_equal(call['request'],file_spec(manifest['role_paths'][role]['request'],root,owner='dynamic'),'immutable held request')
        require(not sem['e360_blocking_holds'],'unknown ownership blocks dispatch')
        if role.endswith('_seal'):require_pending_p10_seal(request,sem,root=root)
        else:require_fit(registration(root),sem,'generated',root=root)
    return helper

def execution_cost_links(debit,cost,actual,call,request,context,live,root):
    """Use the existing schema-specific paid identity primitive, not a second cost schema."""
    paid_request_identity(debit,cost,actual,live,root=root)
    typed_equal(debit['descriptor'],file_spec(debit['descriptor']['path'],root,owner='dynamic'),'actual immutable once cost descriptor')
    typed_equal(debit['mode'],'generated','current numerical debit mode')
    numeric(debit['actual_outer_full_return_s'],'actual suffix debit wall',minimum=1e-12)
    require(type(debit['actual_outer_full_return_s']) is float and type(debit['actual_returned_exit_code']) is int,'typed actual debit duration/exit')
    worker=cost['schema']=='flylab.m3.e360.actual-full-return.v1'
    require(worker or cost['schema']=='flylab.m3.e360.administrative-full-return.v1','finite current actual schemas')
    exit_key='exit_code' if worker else 'actual_returned_exit_code'
    typed_equal(cost[exit_key],debit['actual_returned_exit_code'],'schema-specific cost exit')
    typed_equal(actual['actual_returned_exit_code'],debit['actual_returned_exit_code'],'actual/debit original exit')
    typed_equal(actual['request'],call['request'],'actual immutable call request')
    if worker:
        contract=strict_json(bounded_json_read(cost['contract'],root,cap=524288))
        typed_equal(contract['pass_id'],context['manifest']['pass_id'],'worker data pass')
        typed_equal(cost['source_sha256'],contract['source_sha256'],'worker full Source map')
        own=context['freeze']['own_sources_sha256']
        typed_equal({name:cost['source_sha256'].get(name) for name in own},own,'worker own9 projection')
        require(len(cost['source_sha256'])>=len(own),'complete worker Source map')
        check_sources(cost['source_sha256'],root)
        for key in ('qualified','all_owned_ended','reservation_released','natural_closed','actual_full_return_pending'):
            require(type(cost[key]) is bool,'typed worker closure '+key)
        typed_equal(cost['actual_full_return_pending'],False,'charged worker has actual full return')
        closed=cost['all_owned_ended'] and cost['reservation_released'] and owned_closed(cost)
        require(closed or debit['cost_id'] in live['e360_blocking_holds'],'unclosed charged worker retains debit-backed ownership')
        if cost['qualified']:
            require(closed and cost['natural_closed'] and cost['exit_code']==0 and cost['first_error'] is None
                and cost['secondary_errors']==[] and cost.get('outer_revocation') is None,'actual qualified natural worker closure')
            for key in ('authority_admitted','Popen_attempted','Popen_returned'):typed_equal(cost[key],True,'qualified worker '+key)
            typed_equal(cost['Popen_never_attempted'],False,'genuine worker execution')
            outer=strict_json(read_spec(cost['outer'],root,owner='dynamic'))
            typed_equal(outer['contract'],cost['contract'],'worker outer same contract')
            typed_equal(outer['envelope'],cost['envelope'],'worker actual science envelope')
            typed_equal(outer['natural_closed'],True,'natural outer completion')
            typed_equal(outer['all_owned_ended'],True,'outer complete ownership')
        return cost['exit_code']!=0 or cost['actual_outer_full_return_s']>call['reserved_s'] or not cost['qualified']
    typed_equal(cost['source_sha256'],context['freeze']['own_sources_sha256'],'administrative own9')
    require(type(cost['internal_completion_valid']) is bool,'typed administrative completion result')
    helper=context['rules']['effective_helper_limits'].get(call['call_role'])
    if helper is not None:
        good=False
        if cost['internal_completion'] is not None and cost['expected_output'] is not None:
            completion=strict_json(read_spec(cost['internal_completion'],root,owner='dynamic'))
            required={'schema','completed','call_id','pass_id','call_role','request','resource_extension','revalidation_amendment','source_sha256','expected_output','entry_perf_counter','internal_full_return_s'}
            require(type(completion) is dict and set(completion)==required,'complete same-ENTRY helper evidence')
            for key in ('call_id','pass_id','call_role','resource_extension','revalidation_amendment','source_sha256'):typed_equal(completion[key],request[key],'helper completion '+key)
            typed_equal(completion['schema'],'flylab.m3.e360.helper-internal-completion.v1','helper completion schema')
            typed_equal(completion['request'],call['request'],'helper original request descriptor')
            typed_equal(completion['expected_output'],cost['expected_output'],'helper exact durable output')
            typed_equal(cost['expected_output'],file_spec(request['expected_output'],root,owner='dynamic'),'helper output descriptor')
            for key in ('entry_perf_counter','internal_full_return_s'):
                require(type(completion[key]) is float,'typed helper clock')
                numeric(completion[key],'actual same ENTRY helper '+key,minimum=1e-12)
            good=completion['completed'] is True and completion['internal_full_return_s']<=helper['internal_s']
        typed_equal(cost['internal_completion_valid'],good and actual['collector_proof_valid'],'same-ENTRY helper qualification')
    return cost['actual_returned_exit_code']!=0 or cost['actual_outer_full_return_s']>call['reserved_s'] or not cost['internal_completion_valid']

def execution_owned_and_scalar_state(context,live,current_calls,charged,failures,totals,local,root):
    rules=context['rules'];manifest=context['manifest'];pass_id=manifest['pass_id']
    passed=live['e360_passes'].get(pass_id)
    if passed is not None:
        require(type(passed['next_index']) is int and type(passed['failed_cap_s']) is int,'typed pass index/failed cap')
        if failures:
            cost,call=failures[-1]
            typed_equal(passed['failed_cost'],cost,'last actual failed role/necessary closure')
            typed_equal(passed['failed_cap_s'],call['reserved_s'],'failed original immutable cap')
        else:
            typed_equal(passed['failed_cost'],None,'no fabricated role failure')
            typed_equal(passed['failed_cap_s'],0,'no fabricated failed cap')
    for identity,call in current_calls.items():
        require((call['state']=='charged')==(identity in charged),'charged calls have exactly one numerical debit')
    for identity,hold in live['e360_reservations'].items():
        require(type(hold['reserved_s']) is int and hold['role'] in ('fixture','suite'),'typed finite worker reservation')
        typed_equal(hold['pending_reservations'],{'suite':520} if hold['role']=='fixture' else {},'exact pending suite hold')
        require(identity==hold['attempt_id']+':'+hold['role'],'worker reservation identity')
        typed_equal(hold['reserved_s'],rules['effective_roles'][hold['role']]['outer_s'],'immutable typed reservation cap')
        candidates=[v for v in current_calls.values() if v['attempt_id']==hold['attempt_id'] and v['call_role']==hold['role']+'_worker']
        require(len(candidates)==1 and candidates[0]['state'] in ('begun','actual_pending'),'reservation is backed by one actual worker call')
        request=strict_json(read_spec(candidates[0]['request'],root,owner='dynamic'))
        typed_equal(hold['contract'],file_spec(request['contract'],root,owner='dynamic'),'reservation exact admitted worker contract')
        typed_equal(hold['future_administration_reserved_s'],125 if hold['role']=='fixture' else 65,'typed reservation primary future')
    for identity,pending in live['e360_pending_actuals'].items():
        actual=strict_json(read_spec(pending['actual_tool_return'],root,owner='dynamic'))
        call=current_calls[pending['call_id']]
        require(call['call_role'] in ('fixture_worker','suite_worker') and call['state']=='actual_pending','pending finite actual worker')
        typed_equal(actual['call_id'],pending['call_id'],'pending actual original identity')
        typed_equal(actual['pass_id'],pass_id,'pending actual data pass')
        typed_equal(actual['revalidation_amendment'],context['manifest_spec'],'pending actual authority')
        require(type(actual['actual_returned_exit_code']) is int and type(actual['actual_outer_full_return_s']) is float,'typed pending original full return')
        numeric(actual['actual_outer_full_return_s'],'finite pending actual full return',minimum=1e-12)
        typed_equal(actual['request'],call['request'],'pending actual immutable worker request')
        typed_equal(pending['attempt_id'],call['attempt_id'],'pending worker attempt identity')
        typed_equal(pending['role'],call['call_role'][:-7],'pending worker role identity')
    for identity,hold in live['e360_blocking_holds'].items():
        cost=strict_json(read_spec(hold['full_exit'],root,owner='dynamic'))
        typed_equal(cost['cost_id'],identity,'unknown hold exact paid cost')
        typed_equal(hold['full_exit'],next(d['descriptor'] for d in live['debits'] if d['cost_id']==identity),'hold exact once numerical cost descriptor')
        require(cost.get('all_owned_ended') is not True or cost.get('reservation_released') is not True,'closed worker cannot retain an unknown hold')
    reg=strict_json(read_spec(manifest['registration'],root,owner='historical'))
    typed_equal(live['generated_shared_cumulative_used_s'],reg['generated_shared_baseline_s']+totals['generated'],'raw actual shared sum')
    remaining=1700-reg['generated_shared_baseline_s']-totals['generated']
    typed_equal(live['generated_shared_remaining_s'],remaining,'raw actual remaining')
    require(type(live['e360_budget_exceeded']) is bool,'typed budget result')
    exceeded=live['generated_shared_cumulative_used_s']>1700 or local['generated']>1589
    typed_equal(live['e360_budget_exceeded'],exceeded,'actual generated exceeded flag')
    require(not exceeded,'actual budget exceeded')
    retained=live['generated_new_stage_retained_bytes']
    require(type(retained) is int and 0<=retained<=16<<20,'actual retained bound')
    # MAX routes and same actual substitution reuse the existing production formula.
    necessary=prospective_remaining(live,'generated',root=root)
    require(local['generated']+necessary<=1589 and live['generated_shared_cumulative_used_s']+necessary<=1700,
            'actual same-pending MAX routes fit without reentering the reducer')

def execution_ledger_state(context,root,live=None):
    """Sealed prefix plus actual finite suffix; same four-result ABI, no history replay."""
    require(context['root']==Path(root).resolve(),'no cross-root current context')
    manifest=context['manifest'];anchor=context['anchor'];rules=context['rules'];freeze=context['freeze']
    if live is None:live=strict_json(read_spec(file_spec(BASE+'/parent-e358-execution-ledger.json',root,owner='dynamic'),root,owner='dynamic'))
    mutable={'debits','e360_calls','e360_passes','e360_reservations','e360_pending_actuals','e360_blocking_holds',
        'generated_shared_cumulative_used_s','generated_shared_remaining_s','generated_stage_used_s',
        'e360_generated_local_used_s','generated_new_stage_retained_bytes','e360_budget_exceeded','e360_phase'}
    extra={'execution_context_activation','execution_context_manifest'}
    require(set(live)==set(anchor)|extra,'finite current ledger field whitelist')
    for key in set(anchor)-mutable-extra:typed_equal(live[key],anchor[key],'sealed noncurrent field '+key)
    typed_equal(live['execution_context_manifest'],context['manifest_spec'],'actual context pair manifest')
    typed_equal(live['execution_context_activation'],{'schema':'flylab.m3.execution.activation.v1','admission':context['admission_spec'],'manifest':context['manifest_spec'],'closed_anchor':context['anchor_spec'],'new_generated_grant_s':0},'actual complete pair grant0')
    require(type(live['debits']) is list and len(live['debits'])>=manifest['start_index'],'sealed debit prefix')
    typed_equal(live['debits'][:manifest['start_index']],anchor['debits'],'all exact old debits in order')
    for key in ('e360_calls','e360_passes'):
        require(type(live[key]) is dict,'typed current state')
        for identity,row in anchor[key].items():typed_equal(live[key].get(identity),row,'unchanged historical '+key)
    new_passes=set(live['e360_passes'])-set(anchor['e360_passes'])
    require(new_passes<= {manifest['pass_id']},'only one current data pass')
    current_calls={k:v for k,v in live['e360_calls'].items() if k not in anchor['e360_calls']}
    rows=sorted(current_calls.items(),key=lambda item:item[1]['sequence_index'])
    require(len(rows)<=len(rules['primary_roles']),'finite current roster')
    seen={d['cost_id'] for d in anchor['debits']}
    for index,(identity,call) in enumerate(rows):
        require(type(call['sequence_index']) is int and call['sequence_index']==index,'typed unique role order')
        typed_equal(call['call_role'],rules['primary_roles'][index],'role order')
        typed_equal(call['pass_id'],manifest['pass_id'],'no historical/future current call')
        typed_equal(call['mode'],'generated','current generated only')
        typed_equal(call['reserved_s'],rules['primary_caps_s'][index],'immutable primary cap')
        typed_equal(call['future_reserved_s'],rules['future_with_unused_fallback120_s'][index],'unused fallback once suffix')
        request=strict_json(read_spec(call['request'],root,owner='dynamic'))
        execution_request(request,context,root)
        typed_equal(request['call_id'],identity,'call dictionary identity')
        typed_equal(call['request']['path'],manifest['role_paths'][call['call_role']]['request'],'actual admitted path')
        require(call['state'] in ('begun','actual_pending','charged'),'known current call state')
    require(sum(v['state']=='begun' for v in current_calls.values())<=1,'serial current dispatcher')
    if new_passes:
        passed=live['e360_passes'][manifest['pass_id']]
        typed_equal(passed['next_index'],len(rows),'actual registered role count')
        typed_equal(passed['source_sha256'],freeze['own_sources_sha256'],'single current source chain')
        typed_equal(passed['revalidation_amendment'],context['manifest_spec'],'current pass authority')
        typed_equal(passed['execution_admission'],context['admission_spec'],'actual pass admission')
        typed_equal(passed['mode'],'generated','current pass mode')
    else:require(not rows,'calls require actual pass registration')
    charged={};failures=[];last_completion=-1;completion_order=(0,1,2,4,3,5,7,6,8)
    for offset,debit in enumerate(live['debits'][manifest['start_index']:]):
        require(type(debit) is dict and debit['cost_id'] not in seen,'unique once current cost')
        seen.add(debit['cost_id'])
        cost=charged_cost(debit['descriptor'],'generated',root)
        actual=strict_json(read_spec(cost['actual_tool_return'],root,owner='dynamic'))
        typed_equal(cost['pass_id'],manifest['pass_id'],'current suffix cost identity')
        # The schema-specific primitive below distinguishes ADMIN own9 from worker full map.
        validate_cost_authority(manifest['start_index']+offset,cost,actual,root=root)
        call=current_calls.get(cost['call_id']);require(call is not None and call['state']=='charged','cost backed actual registered call')
        typed_equal(call['cost_id'],debit['cost_id'],'once cost linkage')
        rank=completion_order.index(call['sequence_index'])
        require(rank>last_completion,'actual allowed administrative/worker debit completion order');last_completion=rank
        require(cost['call_id'] not in charged,'one cost for each charged call')
        charged[cost['call_id']]=debit['descriptor']
        request=strict_json(read_spec(call['request'],root,owner='dynamic'))
        proof=p30_collector_proof(actual['collector_trace'],request,call['request'],debit['actual_outer_full_return_s'],debit['actual_returned_exit_code'],root=root)
        typed_equal(actual['collector_proof_valid'],proof['valid'],'actual collector qualification separate exit')
        if call['call_role']=='ruff':
            summary=strict_json(read_spec(cost['capture_return'],root,owner='dynamic'))
            capture=strict_json(read_spec(cost['expected_output'],root,owner='dynamic'))
            typed_equal(summary['manifest_descriptor'],cost['expected_output'],'actual capture summary descriptor')
            typed_equal(capture['request'],call['request'],'actual capture admitted request')
            for key in ('process_identity','streams','owned_state','ruff_exit_code','capture_complete','first_failure'):
                typed_equal(capture[key],summary[key],'original opaque capture '+key)
            require(type(summary['process_identity']['start_utc_ticks']) is int and type(summary['process_identity']['process_id']) is int,'exact Native integer identity')
            typed_equal(cost['internal_completion_valid'],summary['capture_complete'] and summary['publication_verified'] and summary['exit_observed'] and proof['valid'],'actual Ruff capture metadata')
        failed=execution_cost_links(debit,cost,actual,call,request,context,live,root)
        if failed:failures.append((debit['descriptor'],call))
        if failed:
            # The final failed role/cap is checked once after the complete suffix, including closure costs.
            require(all(row['sequence_index']<=call['sequence_index'] or row['call_role'].endswith('_seal') for row in current_calls.values()),'failure permits only same pending closure')
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds'):
        require(type(live[key]) is dict,'typed owned current state')
        for identity,row in live[key].items():
            typed_equal(row['mode'],'generated','no Original current ownership')
            typed_equal(row['revalidation_amendment'],context['manifest_spec'],'owned current direct authority')
            typed_equal(row['resource_extension'],manifest['resource_extension'],'owned extension')
            if key=='e360_pending_actuals':
                require(identity not in seen and identity in live['e360_reservations'],'undebited same held pending actual')
                actual=strict_json(read_spec(row['actual_tool_return'],root,owner='dynamic'))
                call=current_calls[row['call_id']]
                require(call['state']=='actual_pending','actual pending call state')
                typed_equal(actual['request'],call['request'],'pending same request')
                numeric(actual['actual_outer_full_return_s'],'actual pending measured wall',minimum=1e-12)
            elif key=='e360_blocking_holds':
                require(identity in seen and row['debit_backed'] is True and type(row['numeric_reserved_s']) is int and row['numeric_reserved_s']==0,'zero numerical debit-backed hold')
            else:
                require(row['role'] in ('fixture','suite'),'finite reservation role')
                typed_equal(row['reserved_s'],rules['effective_roles'][row['role']]['outer_s'],'actual worker reserve')
    totals={mode:round(math.fsum(d['actual_outer_full_return_s'] for d in live['debits'] if d['mode']==mode),9) for mode in ('generated','original')}
    local={mode:round(math.fsum(d['actual_outer_full_return_s'] for d in live['debits'][42:] if d['mode']==mode),9) for mode in ('generated','original')}
    for key,value in (('generated_stage_used_s',totals['generated']),('e360_generated_local_used_s',local['generated']),('original_stage_used_s',totals['original'])):typed_equal(live[key],value,'actual compensated sum '+key)
    # This scope cannot run Original work. Preserve the writer's actual JSON integer zero.
    typed_equal(live['e360_original_local_used_s'],anchor['e360_original_local_used_s'],'unchanged raw Original local scalar')
    require(local['original']==0,'no Original suffix work in generated scope')
    reg=strict_json(read_spec(manifest['registration'],root,owner='historical'))
    typed_equal(live['generated_shared_cumulative_used_s'],reg['generated_shared_baseline_s']+totals['generated'],'actual raw cumulative double')
    typed_equal(live['generated_shared_remaining_s'],1700-live['generated_shared_cumulative_used_s'],'actual remaining double')
    execution_owned_and_scalar_state(context,live,current_calls,charged,failures,totals,local,root)
    return live['debits'],live,totals,local

def execution_contract_bootstrap(name,entry,root,stage):
    boot=execution_bootstrap(root)
    require(boot is not None,'present admitted context')
    contract=strict_json(bounded_metadata_bytes(name,root))
    role=contract.get('role');require(role in ('fixture','suite'),'finite current worker')
    timer=OperationClock(entry,boot['rules']['effective_roles'][role],stage)
    timer._admitted_root=Path(root).resolve();timer._admitted_clock=timer.timer.clock
    timer()
    context=execution_context(root)
    typed_equal(contract['pass_id'],context['manifest']['pass_id'],'current worker data identity')
    typed_equal(contract['revalidation_amendment'],context['manifest_spec'],'current worker admitted authority')
    typed_equal(contract['limits'],context['rules']['effective_roles'][role],'fixed worker caps')
    typed_equal(name,names(context['manifest']['attempts'][role],role)['contract'],'canonical worker contract')
    timer()
    return timer

class FixedAuthorityScope:
    """One actual operation; parsed literals and a monotone text fence manifest only."""
    def __init__(self,root,check):
        self.root=Path(root).resolve()
        self.check=check
        self.values={}
        self.descriptors={}
        self.history_descriptors={}
        self.history_members=None
        self.dependency_owner='dynamic'
        self.literal_bytes=0
        self.depth=0
        self.failed=False
        self.thread=threading.get_ident()

    def poison(self):
        self.clear_parsed()
        self.failed=True

    def clock(self,stage=None):
        try:
            require(threading.get_ident()==self.thread,'synchronous fixed authority operation')
            ADMIN_CHECK()
            if isinstance(self.check,OperationClock):self.check(stage)
            else:self.check()
        except Exception:
            self.poison()
            raise

    def consumed(self,spec,root,*,owner=None):
        if not self.depth:return
        try:
            self._consumed(spec,root,owner=owner)
        except Exception:
            self.poison()
            raise

    def owned_items(self):
        require(not set(self.descriptors)&set(self.history_descriptors),'disjoint owned descriptor sets')
        return (*tuple(self.descriptors.items()),*tuple(self.history_descriptors.items()))

    def _history_catalogue(self):
        if self.history_members is not None:return self.history_members
        self.clock()
        current=execution_bootstrap(self.root,self.clock)
        spec=HISTORY_CATALOGUE_SPEC if current is None else current['manifest']['history_catalogue']
        path=safe(spec['path'],self.root)
        require(type(spec['bytes']) is int and 0<spec['bytes']<=262144
            and path.stat().st_size==spec['bytes'],'exact bounded history table pin')
        with path.open('rb') as stream:raw=stream.read(spec['bytes']+1)
        require(len(raw)==spec['bytes'] and sha(raw)==spec['sha256'],'exact history table bytes/SHA')
        raw.decode('utf8',errors='strict')
        require(safe(spec['path'],self.root)==path and path.stat().st_size==spec['bytes'],'late history table path/size')
        with path.open('rb') as stream:again=stream.read(spec['bytes']+1)
        require(again==raw,'history table exact reread')
        # The table itself has explicit dynamic ownership; this is not a table-miss fallback.
        self._register(spec,self.root,'dynamic',False)
        table=strict_json(raw)
        count=HISTORY_CATALOGUE_MEMBERS if current is None else table['member_count']
        aggregate=HISTORY_CATALOGUE_AGGREGATE_BYTES if current is None else table['aggregate_member_bytes']
        require(type(table) is dict and type(table.get('members')) is list
            and type(table.get('member_count')) is int and table['member_count']==count
            and len(table['members'])==count
            and type(table.get('aggregate_member_bytes')) is int
            and table['aggregate_member_bytes']==aggregate
            and type(table.get('max_leaf_bytes')) is int and table['max_leaf_bytes']==2097152,
            'typed exact finite history catalogue contract')
        members={};total=0
        for leaf in table['members']:
            self.clock()
            require(type(leaf) is dict and set(leaf)=={'path','bytes','sha256'}
                and type(leaf['path']) is str and type(leaf['bytes']) is int and 0<=leaf['bytes']<=2097152
                and type(leaf['sha256']) is str and re.fullmatch('[0-9a-f]{64}',leaf['sha256']),
                'typed historical catalogue identity')
            require(Path(leaf['path']).suffix.lower() in ('.json','.md','.py','.ps1','.txt','.ini'),
                'historical text members only')
            canonical=os.path.normcase(str(safe(leaf['path'],self.root)))
            require(canonical not in members and canonical!=os.path.normcase(str(path)),
                'unique historical membership excludes its dynamic table')
            members[canonical]=copy.deepcopy(leaf);total+=leaf['bytes']
        require(total==aggregate,'exact history aggregate')
        # Retain only root-bound identity metadata. No semantic document, proof or success is memoized.
        self.history_members=members
        self.clock()
        return members

    def _consumed(self,spec,root,*,owner=None,allow_empty=False):
        try:
            self._register(spec,root,self.dependency_owner if owner is None else owner,allow_empty)
        except Exception:
            self.poison();raise

    def _register(self,spec,root,owner,allow_empty):
        require(not self.failed,'failed fixed authority scope cannot be reused')
        require(Path(root).resolve()==self.root,'fixed scope canonical root isolation')
        require(owner in ('dynamic','historical'),'explicit descriptor ownership')
        require(type(allow_empty) is bool and type(spec) is dict and set(spec)=={'path','bytes','sha256'}
            and type(spec['path']) is str and type(spec['bytes']) is int
            and (0<=spec['bytes']<=2097152 if allow_empty else 0<spec['bytes']<=2097152)
            and type(spec['sha256']) is str and re.fullmatch('[0-9a-f]{64}',spec['sha256']),
            'fixed scope typed text descriptor')
        require(not allow_empty or owner=='historical','only exact completed history admits empty text')
        require(Path(spec['path']).suffix.lower() in ('.json','.md','.py','.ps1','.txt','.ini'),
            'fixed scope text only')
        canonical=os.path.normcase(str(safe(spec['path'],root)))
        target,other,limit=(self.history_descriptors,self.descriptors,len(self._history_catalogue())) if owner=='historical' else (self.descriptors,self.history_descriptors,512)
        require(canonical not in other,'current/history ownership collision')
        if owner=='historical':
            members=self._history_catalogue()
            require(canonical in members,'unknown historical physical dependency')
            typed_equal(members[canonical],spec,'exact historical member identity')
        if canonical in target:typed_equal(target[canonical],spec,'fixed scope consumed identity drift')
        else:
            require(len(target)<limit,'fixed scope descriptor bound')
            target[canonical]=copy.deepcopy(spec)

    def read(self,reader,validator,root):
        entered=False
        try:
            self.clock()
            require(not self.failed,'failed fixed authority scope cannot be reused')
            require(reader in FIXED_READERS,'finite fixed literal reader')
            require(Path(root).resolve()==self.root,'fixed scope reader root')
            literal=FIXED_READERS[reader]
            key=(os.path.normcase(str(self.root)),reader,literal)
            if key in self.values:
                return copy.deepcopy(self.values[key])
            require(len(self.values)<18,'fixed scope entry bound')
            require(literal[1]<=262144 and self.literal_bytes+literal[1]<=2097152,
                    'fixed scope literal byte bounds')
            self.depth+=1
            entered=True
            value=validator(root)
            require(type(value) is dict,'complete typed fixed literal dictionary')
            self.clock()
            require(len(self.values)<18 and self.literal_bytes+literal[1]<=2097152,
                    'nested complete fixed scope bounds')
            spec={'path':literal[0],'bytes':literal[1],'sha256':literal[2]}
            typed_equal(dict(self.owned_items()).get(os.path.normcase(str(safe(literal[0],root)))),spec,
                        'reader exact consumed literal descriptor')
            self.values[key]=copy.deepcopy(value)
            self.literal_bytes+=literal[1]
            return copy.deepcopy(value)
        except Exception:
            self.poison()
            raise
        finally:
            if entered:self.depth-=1

    def clear_parsed(self):
        self.values.clear()
        self.literal_bytes=0

    def fence(self,point):
        """Fresh nonmemo reads; no recursive references, values or fence result saved."""
        try:
            self._fence(point)
        except Exception:
            self.poison()
            raise

    def _fence(self,point):
        require(type(point) is str,'named actual authority boundary')
        require(not self.failed or point in ('failure_prepublication','failure_post_durable_close_readback'),
                'failed fixed authority fence cannot publish or execute')
        stage='science' if point=='post_pre_array_before_first_numeric_native_or_array' else None
        self.clock(stage)
        for canonical,spec in self.owned_items():
            self.clock(stage)
            path=safe(spec['path'],self.root)
            require(os.path.normcase(str(path))==canonical,'authority fence canonical path '+point)
            require(path.stat().st_size==spec['bytes'],'authority fence bytes '+point)
            with path.open('rb') as stream:raw=stream.read(spec['bytes']+1)
            require(len(raw)==spec['bytes'] and sha(raw)==spec['sha256'],'authority fence SHA '+point)
            raw.decode('utf8',errors='strict')
            require(safe(spec['path'],self.root)==path and path.stat().st_size==spec['bytes'],
                    'authority fence late path '+point)
            self.clock(stage)
            with path.open('rb') as stream:again=stream.read(spec['bytes']+1)
            require(again==raw,'authority fence read stability '+point)
        self.clock(stage)


@contextlib.contextmanager
def fixed_authority_scope(*,root=ROOT,check):
    """Caller supplies its actual ENTRY deadline before any scoped reader hit."""
    global _AUTHORITY_SCOPE
    if _AUTHORITY_SCOPE is not None:
        try:
            require(Path(root).resolve()==_AUTHORITY_SCOPE.root,'nested fixed scope root')
            _AUTHORITY_SCOPE.clock()
            check()
        except Exception:
            _AUTHORITY_SCOPE.poison()
            raise
        yield _AUTHORITY_SCOPE
        return
    scope=FixedAuthorityScope(root,check)
    scope.clock()
    _AUTHORITY_SCOPE=scope
    try:
        yield scope
    finally:
        scope.clear_parsed()
        scope.descriptors.clear()
        scope.history_descriptors.clear()
        scope.history_members=None
        _AUTHORITY_SCOPE=None


@contextlib.contextmanager
def dependency_ownership(root,owner):
    """Known semantic caller selects ownership before any member lookup; one actual scope."""
    boundary=_AUTHORITY_SCOPE if _AUTHORITY_SCOPE is not None else _AUTHORITY_FENCE_STATE
    if boundary is None:
        yield;return
    previous=boundary.dependency_owner
    try:
        require(Path(root).resolve()==boundary.root,'dependency owner canonical root')
        require(owner in ('historical','dynamic'),'explicit dependency owner')
        boundary.dependency_owner=owner
        yield
    except Exception:
        boundary.poison();raise
    finally:boundary.dependency_owner=previous

def historical_dependencies(*,root_first=True):
    def decorate(function):
        def wrapped(*args,**kwargs):
            root=kwargs.get('root',args[0] if root_first and args else ROOT)
            with dependency_ownership(root,'historical'):return function(*args,**kwargs)
        wrapped.__name__=function.__name__;wrapped.__doc__=function.__doc__
        return wrapped
    return decorate

def consume_source_descriptor(spec,root,*,owner=None):
    current=execution_bootstrap(root,ADMIN_CHECK) if owner is None else None
    if current is not None and owner is None:
        historical=[*current['manifest']['historical_inputs']]
        for declared,record in current['sealed_records']:
            historical.extend((declared,*record['inputs'],*(row['physical'] for row in record['aliases'])))
        if any(spec==leaf for leaf in historical):owner='historical'
        elif any(spec==leaf for leaf in (current['admission_spec'],current['manifest_spec'],current['manifest']['rules'],current['manifest']['source_freeze'],current['manifest']['history_catalogue'],*current['freeze']['native_sources'],current['freeze']['collector_source'])):owner='dynamic'
    boundary=_AUTHORITY_SCOPE
    if boundary is None:return
    selected=owner
    if selected is None:
        selected='dynamic' if any(spec==anchor for anchor in _CURRENT_P30_DYNAMIC_PINS) else boundary.dependency_owner
        # These literal identities are Source-declared historical authority, independent of table membership.
        if selected=='dynamic' and any(spec==anchor for anchor in HISTORICAL_LITERAL_SPECS):
            selected='historical'
    if selected=='historical':boundary._consumed(spec,root,owner='historical')
    else:boundary.consumed(spec,root,owner='dynamic')

def authority_fence(point):
    scope=_AUTHORITY_SCOPE if _AUTHORITY_SCOPE is not None else _AUTHORITY_FENCE_STATE
    if scope is not None:scope.fence(point)


@contextlib.contextmanager
def descriptors_only_execution():
    """External/numerical execution has no parsed bundle; preserve boundary pins."""
    global _AUTHORITY_SCOPE,_AUTHORITY_FENCE_STATE
    scope,previous=_AUTHORITY_SCOPE,_AUTHORITY_FENCE_STATE
    if scope is not None:scope.clear_parsed()
    _AUTHORITY_SCOPE=None
    _AUTHORITY_FENCE_STATE=scope if scope is not None else previous
    try:
        yield
    finally:
        _AUTHORITY_SCOPE=scope
        _AUTHORITY_FENCE_STATE=previous


def put(c,name,value,timer,**kwargs):
    if kwargs.get('emergency') is True and value.get('first_error') is not None:
        require(name in ('envelope.json','revocation.json'),'failure-only child metadata name')
        root=kwargs.get('root',ROOT)
        target=c['paths']['output']+'/'+name
        if safe(target,root).exists():target=c['paths']['output']+'/revocation.json'
        return failure_metadata(c,target,value,timer,stage='internal',root=root)
    authority_fence('prepublication')
    result=lib.put(c,name,value,timer,**kwargs)
    authority_fence('post_durable_close_readback')
    return result


class OperationClock:
    """Fixed admitted role caps on the genuine parent ENTRY; no clock reset."""
    def __init__(self,entry,limits,stage):
        self.timer=Clock(entry,limits)
        self.stage=stage

    def __call__(self,stage=None):
        self.timer.check(self.stage if stage is None else stage)


def operation_clock(name,entry,*,root=ROOT,stage='outer'):
    if execution_bootstrap(root) is not None:return execution_contract_bootstrap(name,entry,root,stage)
    # Only bounded uncached contract metadata selects an immutable clock cap;
    # full authority reproduction remains inside the real operation deadline.
    if type(name) is str and name.startswith(BASE+'/parent-e346-e360-e360-generous-p30-'):
        return current_contract_bootstrap(name,entry,root,stage)
    raw=bounded_metadata_bytes(name,root)
    contract=strict_json(raw)
    require(contract.get('pass_id') not in tuple('e360-generous-p'+str(i) for i in range(1,15))
        and contract.get('mode')!='original' and contract.get('pass_id')!='e360-generous-p16','completed history/future/original operation denied')
    require(contract.get('pass_id') not in ('e360-generous-p17','e360-generous-p18','e360-generous-p19','e360-generous-p20','e360-generous-p21','e360-generous-p30')
        and (not str(contract.get('pass_id','')).startswith('e360-generous-p') or contract['pass_id']=='e360-generous-p15'),'completed or unknown current worker bootstrap')
    role=contract['role']
    limits={'fixture':{'outer_s':30,'child_s':24,'internal_s':20,'science_s':16},
            'suite':{'outer_s':120,'child_s':110,'internal_s':100,'science_s':90},
            'saved_prefix_diagnostic':{'outer_s':12,'child_s':11,'internal_s':10,'science_s':9}}
    if contract.get('pass_id') in ('e360-generous-p8','e360-generous-p9','e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p13','e360-generous-p14','e360-generous-p15','e360-generous-p16'):
        # Establish the genuine ENTRY cap before full fixed-authority recovery.
        p16=contract['pass_id']=='e360-generous-p16'
        p15=contract['pass_id']=='e360-generous-p15'
        p14=contract['pass_id']=='e360-generous-p14'
        p13=contract['pass_id']=='e360-generous-p13'
        p12=contract['pass_id']=='e360-generous-p12'
        p11=contract['pass_id']=='e360-generous-p11'
        p10=contract['pass_id']=='e360-generous-p10'
        p9=contract['pass_id']=='e360-generous-p9'
        if p9 or p10 or p11 or p12 or p13 or p14 or p15 or p16:limits['suite']={'outer_s':600,'child_s':560,'internal_s':540,'science_s':500}
        bootstrap=OperationClock(entry,{'outer_s':55,'child_s':50,'internal_s':46,'science_s':42}
            if role=='fixture' else limits.get(role,{}),stage)
        bootstrap()
        path=P16_RECOVERY_AMENDMENT if p16 else P15_RECOVERY_AMENDMENT if p15 else P14_RECOVERY_AMENDMENT if p14 else P13_RECOVERY_AMENDMENT if p13 else P12_RECOVERY_AMENDMENT if p12 else P11_RECOVERY_AMENDMENT if p11 else LINT_RECOVERY_AMENDMENT if p10 else SUITE_RECOVERY_AMENDMENT if p9 else EXECUTION_BUDGET_AMENDMENT
        typed_equal(contract['revalidation_amendment'],file_spec(path,root),'exact pass clock authority')
        literal=bounded_metadata_bytes(path,root)
        require(len(literal)==(P16_RECOVERY_BYTES if p16 else P15_RECOVERY_BYTES if p15 else P14_RECOVERY_BYTES if p14 else P13_RECOVERY_BYTES if p13 else 50865 if p12 else 33809 if p11 else 31323 if p10 else 28728 if p9 else 24334) and sha(literal)==(P16_RECOVERY_SHA if p16 else P15_RECOVERY_SHA if p15 else P14_RECOVERY_SHA if p14 else P13_RECOVERY_SHA if p13 else P12_RECOVERY_SHA if p12 else P11_RECOVERY_SHA if p11 else LINT_RECOVERY_SHA if p10 else SUITE_RECOVERY_SHA if p9 else EXECUTION_BUDGET_SHA),'clock exact admitted literal')
        limits=strict_json(literal)['effective_roles']
        bootstrap()
    require(role in limits,'finite operation clock role')
    if 'limits' in contract:
        typed_equal({key:contract['limits'][key] for key in limits[role]},limits[role],'operation exact immutable role caps')
    checked=OperationClock(entry,limits[role],stage)
    checked._admitted_root=Path(root).resolve()
    checked._admitted_clock=checked.timer.clock
    return checked

@contextlib.contextmanager
def metadata_invocation(root):
    """Only the two reviewed metadata calls; no cache of results or live state."""
    global _AUTHORITY_SCOPE
    ambient=_AUTHORITY_SCOPE
    inherited=_AUTHORITY_FENCE_STATE
    context=_SUITE_DEADLINE
    boundary=ambient if ambient is not None else inherited
    admitted=(boundary is not None and type(boundary.check) is OperationClock
              and getattr(boundary.check,'_admitted_root',None)==Path(root).resolve())
    # A suite's deliberately synthetic test scopes cannot borrow its authority.
    if context is not None:
        admitted=admitted and boundary is context.boundary and context.root==Path(root).resolve()
    if not admitted:
        yield False
        return
    boundary.clock()
    require(not boundary.failed,'inherited metadata boundary poisoned')
    if context is not None:context.check('science')
    local=ambient if ambient is not None else FixedAuthorityScope(root,boundary.check)
    owner=ambient is None
    if owner:_AUTHORITY_SCOPE=local
    local.depth+=1
    primary=None
    try:
        yield True
    except BaseException as error:
        primary=error
        raise
    finally:
        try:
            # Refresh inherited authority even after an expected validation error.
            # Such errors only poison/clear the disposable local reader bundle.
            if context is not None:context.check('science')
            boundary.fence('metadata_inherited_completion')
            if primary is None:local.fence('metadata_local_completion')
        except Exception as secondary:
            if primary is None:raise
            primary.add_note('metadata completion secondary: '+repr(secondary))
        finally:
            if owner:
                local.clear_parsed()
                local.descriptors.clear()
                local.history_descriptors.clear()
                local.history_members=None
                _AUTHORITY_SCOPE=ambient
            local.depth-=1


class SuiteDeadline:
    """Authenticated child deadline and descriptor boundary; no parsed values."""
    def __init__(self,c,spec,receipt_spec,timer,root):
        self.root=Path(root).resolve()
        self.boundary=_AUTHORITY_FENCE_STATE
        require(_AUTHORITY_SCOPE is None and self.boundary is not None
                and type(self.boundary.check) is OperationClock
                and getattr(self.boundary.check,'_admitted_root',None)==self.root,
                'authenticated descriptors-only suite boundary')
        require(type(timer) is Clock and type(timer.entry) is float,'existing actual child timer')
        admitted_clock=self.boundary.check._admitted_clock
        require(callable(admitted_clock) and timer.clock is admitted_clock
                and self.boundary.check.timer.clock is admitted_clock,
                'suite genuine admitted clock callable, no substitution')
        typed_equal(timer.entry,self.boundary.check.timer.entry,'suite inherited actual ENTRY')
        typed_equal(timer.limits,c['limits'],'suite actual child caps')
        receipt=strict_json(bounded_json_read(receipt_spec,root,cap=262144,check=lambda:timer.check('science')))
        validate_current_receipt(c,spec,receipt,root=root,entry=timer.entry,check=lambda:timer.check('science'))
        self.entry=timer.entry
        self.clock=admitted_clock
        self.limits=copy.deepcopy(timer.limits)
        self.check('science')
        self.boundary.fence('suite_deadline_entry')

    def check(self,stage='science'):
        try:
            now=self.clock()
            require(math.isfinite(now) and now>=self.entry,'suite actual monotone ENTRY')
            if now>self.entry+self.limits[stage+'_s']:
                raise TimeoutError('E360 suite same-ENTRY '+stage+' deadline')
        except Exception:
            self.boundary.poison()
            raise


@contextlib.contextmanager
def suite_deadline(c,spec,receipt_spec,timer,root):
    global _SUITE_DEADLINE
    previous=_SUITE_DEADLINE
    require(previous is None,'one authenticated suite deadline')
    context=SuiteDeadline(c,spec,receipt_spec,timer,root)
    _SUITE_DEADLINE=context
    try:
        yield context
    finally:
        _SUITE_DEADLINE=previous


def bounded_metadata_bytes(name,root=ROOT):
    path=safe(name,root)
    require(path.stat().st_size<=262144,'bounded operation metadata')
    with path.open('rb') as stream:raw=stream.read(262145)
    require(len(raw)<=262144,'bounded operation metadata read')
    return raw


CURRENT_PRESPAWN_SCHEMA='flylab.m3.e360.actual-pre-spawn.v2'
PRESPAWN_CONTRACT_FIELDS=('role','attempt_id','attempt','source_sha256','cumulative_before_s',
    'cumulative_limit_s','reserved_s','pending_reservations','limits','stage_used_s','stage_limit_s',
    'original_calls','max_tree_rss_bytes','retained_reserve_bytes','cpu_threads','registration',
    'amendment','resource_extension','revalidation_amendment','closure_report',
    'future_administration_reserved_s','local_used_s','local_limit_s')


def enroll_json_descriptor(spec,root):
    # consumed() intentionally ignores depth0. Register at the actual boundary.
    boundary=_AUTHORITY_SCOPE if _AUTHORITY_SCOPE is not None else _AUTHORITY_FENCE_STATE
    if boundary is not None:
        try:
            boundary.clock()
            require(not boundary.failed,'failed actual JSON boundary')
            boundary._consumed(spec,root)
            boundary.clock()
        except Exception:
            boundary.poison()
            raise


def bounded_json_read(spec,root=ROOT,*,cap,check=None):
    def deadline():
        ADMIN_CHECK()
        boundary=_AUTHORITY_SCOPE if _AUTHORITY_SCOPE is not None else _AUTHORITY_FENCE_STATE
        if boundary is not None:boundary.clock()
        if check is not None:check()
    deadline()
    require(type(spec) is dict and set(spec)=={'path','bytes','sha256'}
        and type(spec['path']) is str and Path(spec['path']).suffix=='.json'
        and type(spec['bytes']) is int and 0<spec['bytes']<=cap
        and type(spec['sha256']) is str and re.fullmatch('[0-9a-f]{64}',spec['sha256']),
        'bounded exact typed JSON descriptor')
    path=safe(spec['path'],root)
    canonical=os.path.normcase(str(path))
    require(path.stat().st_size==spec['bytes'],'bounded JSON pre-read bytes')
    with path.open('rb') as stream:raw=stream.read(spec['bytes']+1)
    require(len(raw)==spec['bytes'] and sha(raw)==spec['sha256'],'bounded JSON exact bytes/SHA')
    decoded=raw.decode('utf8',errors='strict')
    strict_json(decoded)  # Strict UTF8 JSON, including duplicate/nonfinite rejection.
    deadline()
    fresh=safe(spec['path'],root)
    require(os.path.normcase(str(fresh))==canonical and fresh.stat().st_size==spec['bytes'],
        'bounded JSON late path/stat')
    with fresh.open('rb') as stream:again=stream.read(spec['bytes']+1)
    require(again==raw,'bounded JSON independent read stability')
    deadline()
    enroll_json_descriptor(spec,root)
    return raw


def bounded_json_name(name,root=ROOT,*,cap,check=None):
    # Bootstrap unknown descriptor without ever doing an unbounded read.
    ADMIN_CHECK()
    if check is not None:check()
    require(type(name) is str and Path(name).suffix=='.json','bounded JSON name')
    path=safe(name,root)
    size=path.stat().st_size
    require(type(size) is int and 0<size<=cap,'bounded JSON pre-stat cap')
    with path.open('rb') as stream:raw=stream.read(size+1)
    require(len(raw)==size,'bounded JSON initial bytes')
    return bounded_json_read(descriptor(name,raw),root,cap=cap,check=check)


def validate_p30_primary_future(contract,root=ROOT):
    identity=contract.get('pass_id')
    require(identity in ('e360-generous-p30',execution_current_pass(root)) and contract.get('mode')=='generated'
        and contract.get('role') in ('fixture','suite'),'exact current primary future role')
    value=contract.get('future_administration_reserved_s')
    require(type(value) is int,'typed current primary future125/65')
    typed_equal(value,125 if contract['role']=='fixture' else 65,'immutable current primary future125/65')
    return value

def validate_historical_p18_fallback(balance,semantics,pass_id='e360-generous-p18'):
    require(type(balance) is int and 0<=balance<=120,'historical typed P18 fallback')
    passed=semantics['e360_passes'].get(pass_id)
    rows=[row for row in semantics['e360_calls'].values() if row['pass_id']==pass_id]
    terminal=passed is not None and (passed.get('failed_cost') is not None or passed['next_index']==9)
    owned=any(semantics[key] for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds'))
    expected=0 if rows and all(row['state']=='charged' for row in rows) and terminal and not owned else 120
    typed_equal(balance,expected,'historical P18 fallback once closure')
    return balance
def compact_receipt(c,spec,entry,pid,created,*,root=ROOT):
    receipt={'schema':CURRENT_PRESPAWN_SCHEMA,'contract':spec,'entry_perf_counter':entry,
        'parent_pid':pid,'parent_create_time':created,'reservation_held':True,
        'original_array_reads':0,'scientific_native_imports':0,'Popen_before_receipt':0,
        'cumulative_after_s':None}
    receipt.update({key:c[key] for key in PRESPAWN_CONTRACT_FIELDS})
    if c.get('pass_id') in ('e360-generous-p30',execution_current_pass(root)):
        validate_p30_primary_future(c,root)
        validate_p30_fallback_balance(c['same_pending_fallback_reserved_s'],c['ledger_semantics'],root=root)
        receipt['same_pending_fallback_reserved_s']=c['same_pending_fallback_reserved_s']
    if c.get('pass_id') in ('e360-generous-p19','e360-generous-p20','e360-generous-p21','e360-generous-p22','e360-generous-p23'):
        primary=c.get('future_administration_reserved_s')
        require(type(primary) is int and primary==(125 if c['role']=='fixture' else 65),'historical P19 primary125/65')
        validate_historical_p18_fallback(c['same_pending_fallback_reserved_s'],c['ledger_semantics'],c['pass_id'])
        receipt['same_pending_fallback_reserved_s']=c['same_pending_fallback_reserved_s']
    if c.get('pass_id')=='e360-generous-p18':
        validate_historical_p18_fallback(c['same_pending_fallback_reserved_s'],c['ledger_semantics'])
        receipt['same_pending_fallback_reserved_s']=c['same_pending_fallback_reserved_s']
    fields=34 if c.get('pass_id') in ('e360-generous-p18','e360-generous-p19','e360-generous-p20','e360-generous-p21','e360-generous-p22','e360-generous-p23','e360-generous-p25','e360-generous-p27','e360-generous-p29','e360-generous-p30',execution_current_pass(root)) else 33
    require(len(receipt)==fields,'exact compact current'+str(fields)+' receipt')
    return receipt

def validate_current_receipt(c,spec,receipt,*,root=ROOT,entry=None,check=None):
    pinned=strict_json(bounded_json_read(spec,root,cap=524288,check=check))
    typed_equal(pinned,c,'complete freshly pinned admitted contract')
    require(type(receipt) is dict,'typed compact receipt')
    actual_entry=receipt.get('entry_perf_counter')
    require(type(actual_entry) is float,'typed compact actual ENTRY')
    numeric(actual_entry,'compact ENTRY',minimum=1e-12)
    require(type(receipt.get('parent_pid')) is int and receipt['parent_pid']>0
        and type(receipt.get('parent_create_time')) is float,'typed compact parent identity')
    numeric(receipt['parent_create_time'],'compact parent creation',minimum=1e-12)
    # The ambient clock times this verifier; it may follow a closed worker.
    # Only worker admission callers provide their already admitted worker ENTRY.
    if entry is not None:typed_equal(actual_entry,entry,'compact original admitted ENTRY')
    expected=compact_receipt(c,spec,actual_entry,receipt['parent_pid'],receipt['parent_create_time'],root=root)
    typed_equal(receipt,expected,'complete current'+str(len(expected))+' compact receipt')
    return receipt

def literal_bytes(name,root):
    raw=bounded_metadata_bytes(name,root)
    if _AUTHORITY_SCOPE is not None:
        consume_source_descriptor({'path':name,'bytes':len(raw),'sha256':sha(raw)},root)
    return raw


def fixed_reader(reader,validator,root):
    if _AUTHORITY_SCOPE is None:return validator(root)
    return _AUTHORITY_SCOPE.read(reader,validator,root)


def read_spec(spec,root=ROOT,*,owner=None):
    ADMIN_CHECK()
    if type(spec) is dict and type(spec.get('path')) is str and Path(spec['path']).suffix=='.ps1':
        require(set(spec)=={'path','bytes','sha256'} and type(spec.get('bytes')) is int
                and 0<spec['bytes']<=(2<<20) and type(spec.get('sha256')) is str
                and re.fullmatch('[0-9a-f]{64}',spec['sha256']),'bounded typed PowerShell source descriptor')
        path=safe(spec['path'],root)
        require(path.stat().st_size==spec['bytes'],'PowerShell source size')
        with path.open('rb') as stream:value=stream.read(spec['bytes']+1)
        require(len(value)==spec['bytes'] and sha(value)==spec['sha256'],'PowerShell source bytes/SHA')
        value.decode('utf8',errors='strict')
        path=safe(spec['path'],root)  # Recheck containment and every link after reading.
        require(path.stat().st_size==spec['bytes'],'PowerShell late source size')
        with path.open('rb') as stream:again=stream.read(spec['bytes']+1)
        require(again==value,'PowerShell source read stability')
    else:
        value=lib.read_spec(spec,root)  # Original binary suffixes still reject before I/O.
    ADMIN_CHECK()
    if _AUTHORITY_SCOPE is not None:consume_source_descriptor(spec,root,owner=owner)
    return value


def file_spec(name,root=ROOT,*,owner=None):
    ADMIN_CHECK()
    value=lib.file_spec(name,root)
    if _AUTHORITY_SCOPE is not None:consume_source_descriptor(value,root,owner=owner)
    ADMIN_CHECK()
    return value


def durable(path,data,*,cap,check=lambda:None):
    def checked():
        ADMIN_CHECK()
        check()
    checked()
    authority_fence('prepublication')
    value=lib.durable(path,data,cap=cap,check=checked)
    authority_fence('post_durable_close_readback')
    checked()
    return value


@historical_dependencies()
def _resource_extension(root=ROOT):
    """Only the literal reviewed v2 can extend resources; historical pins stay separate."""
    ADMIN_CHECK()
    raw=literal_bytes(RESOURCE_EXTENSION,root)
    require(len(raw)==5461 and sha(raw)==RESOURCE_EXTENSION_SHA,'literal generous v2 authority')
    value=strict_json(raw)
    typed_equal(value['old_registration'],file_spec(REGISTRATION,root),'extension old registration')
    typed_equal(value['old_amendment'],file_spec(AMENDMENT,root),'extension old amendment')
    for key in EXTENSION_SOURCE_KEYS:
        read_spec(value[key],root)
    expected={'local_cost_offset':42,'required_exact_prefix_length':50,'old_shared_generated_limit_s':150,
              'effective_shared_generated_limit_s':600,'old_generated_local_limit_s':39,
              'effective_generated_local_limit_s':489,'new_generated_grant_s':450,'new_shared_budget_grant':True,
              'new_original_execution_authority':False,'generated_pass_limit':2,'modified_generated_pass_limit':1,
              'automatic_retries':0,'unchanged_original_local_limit_s':20,'unchanged_original_stage_cap_s':300,
              'unchanged_original_cumulative_limit_s':10200,'unchanged_candidate_attempts':1,
              'unchanged_modified_retry_used':0,'unchanged_modified_retry_limit':1}
    for key,item in expected.items():
        typed_equal(value[key],item,'exact generous policy '+key)
    ADMIN_CHECK()
    return value


@historical_dependencies()
def _revalidation_amendment(root=ROOT):
    """Literal no-grant p3 authority; reviewed snapshots never pin final own sources."""
    ADMIN_CHECK()
    raw=literal_bytes(OLD_REVALIDATION_AMENDMENT,root)
    require(len(raw)==6282 and sha(raw)==OLD_REVALIDATION_SHA,'literal historical same-budget amendment')
    value=strict_json(raw)
    typed_equal(value['old_resource_extension'],file_spec(RESOURCE_EXTENSION,root),'preserved v2 extension')
    expected={'local_cost_offset':42,'required_exact_prefix_length':59,'generated_shared_limit_s':600,
              'generated_local_limit_s':489,'new_generated_grant_s':0,'generated_pass_limit':3,
              'modified_generated_pass_limit':2,'original_pass_limit':1,'total_pass_limit':4,
              'manual_generated_pass_id':'e360-generous-p3','predecessor_pass_id':'e360-generous-p2',
              'predecessor_failed_cap_s':120,'automatic_retries':0,'pass4_authorized':False,
              'whole_fresh_chain_required':True,'primary_total_reserved_s':263,'source_hash_cycle_allowed':False}
    for key,item in expected.items():typed_equal(value[key],item,'same-budget policy '+key)
    for key in REVALIDATION_SOURCE_KEYS:read_spec(value[key],root)
    for spec in value['reviewed_source_snapshots'].values():read_spec(spec,root)
    require(len(value['historical_concrete_fix_relocations'])==2,'exact two historical relocations')
    for row in value['historical_concrete_fix_relocations']:
        typed_equal(row['original']['bytes'],row['snapshot']['bytes'],'relocation bytes')
        typed_equal(row['original']['sha256'],row['snapshot']['sha256'],'relocation SHA')
        read_spec(row['snapshot'],root)
    return value


def revalidation_event(root=ROOT):
    authority=revalidation_amendment(root)
    return {'schema':'flylab.m3.e360.same-budget-activation.v1',
            'revalidation_amendment':file_spec(OLD_REVALIDATION_AMENDMENT,root),
            'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited59':authority['inherited59'],
            'manual_generated_pass_id':'e360-generous-p3','new_generated_grant_s':0}


@historical_dependencies()
def _active_revalidation_amendment(root=ROOT):
    """Fixed p4 admin recovery; old p3 remains an immutable historical authority."""
    ADMIN_CHECK()
    raw=literal_bytes(ACTIVE_REVALIDATION_AMENDMENT,root)
    require(len(raw)==8459 and sha(raw)==ACTIVE_REVALIDATION_SHA,'literal active admin-recovery authority')
    value=strict_json(raw)
    typed_equal(value['previous_revalidation_amendment'],file_spec(OLD_REVALIDATION_AMENDMENT,root),'preserved old p3 authority')
    typed_equal(value['old_resource_extension'],file_spec(RESOURCE_EXTENSION,root),'preserved v2 authority')
    revalidation_amendment(root)
    expected={'schema':'flylab.m3.e360.admin-recovery-amendment.v1','local_cost_offset':42,
              'required_exact_prefix_length':60,'generated_shared_limit_s':600,'generated_local_limit_s':489,
              'new_generated_grant_s':0,'generated_pass_limit':4,'modified_generated_pass_limit':3,
              'original_pass_limit':1,'total_pass_limit':5,'manual_generated_pass_id':'e360-generous-p4',
              'predecessor_pass_id':'e360-generous-p3','predecessor_failed_cap_s':3,'automatic_retries':0,
              'pass5_authorized':False,'whole_fresh_chain_required':True,'primary_total_reserved_s':263,
              'source_hash_cycle_allowed':False,'original_local_limit_s':20,'original_stage_limit_s':300,
              'original_global_limit_s':10200,'original_worker_maximum':1,'original_automatic_retries':0}
    for key,item in expected.items():typed_equal(value[key],item,'active policy '+key)
    for key in ACTIVE_REVALIDATION_SOURCE_KEYS:read_spec(value[key],root)
    for spec in value['reviewed_source_snapshots'].values():read_spec(spec,root)
    rows=value['historical_concrete_fix_relocations']
    require(type(rows) is list and len(rows)==6 and [r['pass_id'] for r in rows]==['e360-generous-p2']*2+['e360-generous-p3']*4,
            'exact six pass-bound historical relocations')
    for row in rows:
        typed_equal(row['original']['bytes'],row['snapshot']['bytes'],'historical relocation bytes')
        typed_equal(row['original']['sha256'],row['snapshot']['sha256'],'historical relocation SHA')
        read_spec(row['snapshot'],root)
    return value


def active_revalidation_event(root=ROOT):
    authority=active_revalidation_amendment(root)
    return {'schema':'flylab.m3.e360.admin-recovery-activation.v1',
            'revalidation_amendment':file_spec(ACTIVE_REVALIDATION_AMENDMENT,root),
            'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited60':authority['inherited60'],
            'previous_revalidation_amendment':authority['previous_revalidation_amendment'],
            'manual_generated_pass_id':authority['manual_generated_pass_id'],'new_generated_grant_s':authority['new_generated_grant_s']}


@historical_dependencies()
def _source_recovery_amendment(root=ROOT):
    """Fixed p5 authority; the literal p3 and p4 readers remain historical."""
    ADMIN_CHECK()
    raw=literal_bytes(SOURCE_RECOVERY_AMENDMENT,root)
    require(len(raw)==10441 and sha(raw)==SOURCE_RECOVERY_SHA,'literal PS-source-recovery authority')
    value=strict_json(raw)
    typed_equal(value['previous_revalidation_amendment'],file_spec(ACTIVE_REVALIDATION_AMENDMENT,root),'preserved p4 authority')
    typed_equal(value['historical_p3_revalidation_amendment'],file_spec(OLD_REVALIDATION_AMENDMENT,root),'preserved p3 authority')
    typed_equal(value['old_resource_extension'],file_spec(RESOURCE_EXTENSION,root),'preserved v2 grant')
    active_revalidation_amendment(root)
    expected={'schema':'flylab.m3.e360.ps-source-recovery-amendment.v1','local_cost_offset':42,
              'required_exact_prefix_length':62,'generated_shared_limit_s':600,'generated_local_limit_s':489,
              'new_generated_grant_s':0,'generated_pass_limit':5,'modified_generated_pass_limit':4,
              'original_pass_limit':1,'total_pass_limit':6,'manual_generated_pass_id':'e360-generous-p5',
              'predecessor_pass_id':'e360-generous-p4','predecessor_failed_cap_s':20,'automatic_retries':0,
              'pass6_authorized':False,'whole_fresh_chain_required':True,'primary_total_reserved_s':263,
              'minimum_current_test_functions':36,'minimum_current_test_phases':108,
              'source_hash_cycle_allowed':False,'original_local_limit_s':20,'original_stage_limit_s':300,
              'original_global_limit_s':10200,'original_worker_maximum':1,'original_automatic_retries':0}
    for key,item in expected.items():typed_equal(value[key],item,'current policy '+key)
    for key in SOURCE_RECOVERY_SOURCE_KEYS:read_spec(value[key],root)
    for spec in value['reviewed_source_snapshots'].values():read_spec(spec,root)
    rows=value['historical_concrete_fix_relocations']
    require(type(rows) is list and len(rows)==10 and [r['pass_id'] for r in rows]==['e360-generous-p2']*2+['e360-generous-p3']*4+['e360-generous-p4']*4,'exact ten pass-bound relocations')
    for row in rows:
        typed_equal(row['original']['bytes'],row['snapshot']['bytes'],'historical relocation bytes')
        typed_equal(row['original']['sha256'],row['snapshot']['sha256'],'historical relocation SHA')
        read_spec(row['snapshot'],root)
    return value


def source_recovery_event(root=ROOT):
    authority=source_recovery_amendment(root)
    return {'schema':'flylab.m3.e360.ps-source-recovery-activation.v1',
            'revalidation_amendment':file_spec(SOURCE_RECOVERY_AMENDMENT,root),
            'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited62':authority['inherited62'],
            'previous_revalidation_amendment':authority['previous_revalidation_amendment'],
            'manual_generated_pass_id':authority['manual_generated_pass_id'],'new_generated_grant_s':authority['new_generated_grant_s']}


@historical_dependencies()
def _ruff_recovery_amendment(root=ROOT):
    """Fixed p6 authority; literal p3/p4/p5 readers remain historical."""
    ADMIN_CHECK()
    raw=literal_bytes(RUFF_RECOVERY_AMENDMENT,root)
    require(len(raw)==12323 and sha(raw)==RUFF_RECOVERY_SHA,'literal Ruff-recovery authority')
    value=strict_json(raw)
    typed_equal(value['previous_revalidation_amendment'],file_spec(SOURCE_RECOVERY_AMENDMENT,root),'preserved p5 authority')
    typed_equal(value['historical_p4_revalidation_amendment'],file_spec(ACTIVE_REVALIDATION_AMENDMENT,root),'preserved p4 authority')
    typed_equal(value['historical_p3_revalidation_amendment'],file_spec(OLD_REVALIDATION_AMENDMENT,root),'preserved p3 authority')
    typed_equal(value['old_resource_extension'],file_spec(RESOURCE_EXTENSION,root),'preserved v2 grant')
    source_recovery_amendment(root)
    expected={'schema':'flylab.m3.e360.ruff-recovery-amendment.v1','local_cost_offset':42,
              'required_exact_prefix_length':63,'generated_shared_limit_s':600,'generated_local_limit_s':489,
              'new_generated_grant_s':0,'generated_pass_limit':6,'modified_generated_pass_limit':5,
              'original_pass_limit':1,'total_pass_limit':7,'manual_generated_pass_id':'e360-generous-p6',
              'predecessor_pass_id':'e360-generous-p5','predecessor_failed_cap_s':3,'automatic_retries':0,
              'pass7_authorized':False,'whole_fresh_chain_required':True,'primary_total_reserved_s':263,
              'minimum_current_test_functions':38,'minimum_current_test_phases':114,
              'source_hash_cycle_allowed':False,'original_local_limit_s':20,'original_stage_limit_s':300,
              'original_global_limit_s':10200,'original_worker_maximum':1,'original_automatic_retries':0}
    for key,item in expected.items():typed_equal(value[key],item,'current policy '+key)
    for key in RUFF_RECOVERY_SOURCE_KEYS:read_spec(value[key],root)
    for spec in value['reviewed_source_snapshots'].values():read_spec(spec,root)
    rows=value['historical_concrete_fix_relocations']
    require(type(rows) is list and len(rows)==14 and [r['pass_id'] for r in rows]==['e360-generous-p2']*2+['e360-generous-p3']*4+['e360-generous-p4']*4+['e360-generous-p5']*4,'exact fourteen pass-bound relocations')
    for row in rows:
        typed_equal(row['original']['bytes'],row['snapshot']['bytes'],'historical relocation bytes')
        typed_equal(row['original']['sha256'],row['snapshot']['sha256'],'historical relocation SHA')
        read_spec(row['snapshot'],root)
    return value


def ruff_recovery_event(root=ROOT):
    authority=ruff_recovery_amendment(root)
    return {'schema':'flylab.m3.e360.ruff-recovery-activation.v1',
            'revalidation_amendment':file_spec(RUFF_RECOVERY_AMENDMENT,root),
            'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited63':authority['inherited63'],
            'previous_revalidation_amendment':authority['previous_revalidation_amendment'],
            'manual_generated_pass_id':authority['manual_generated_pass_id'],'new_generated_grant_s':authority['new_generated_grant_s']}


FIXED_SCOPE_POLICY=strict_json(b'{"schema":"flylab.m3.e360.fixed-authority-operation-scope-policy.v1","kind":"Explicit per-operation fixed-authority bundle and descriptors-only fence manifest","reader_ids":["registration","amendment","resource_extension","revalidation_amendment","active_revalidation_amendment","source_recovery_amendment","ruff_recovery_amendment","current_revalidation_amendment"],"max_entries":16,"max_text_descriptors":512,"max_literal_json_bytes":262144,"max_aggregate_literal_json_bytes":2097152,"synchronous_after_actual_ENTRY_only":true,"key_canonical_root_reader_exact_literal":true,"complete_success_only":true,"deepcopy_return":true,"clock_on_every_hit":true,"finally_clear_preserve_first_error":true,"raw_bytes_cache":false,"ledger_values_cache":false,"Q_cache":false,"request_result_cache":false,"dynamic_process_budget_ownership_cache":false,"own_source_discovery_clock_array_cache":false,"persistent_crossprocess_preseed":false,"dependency_dropping_eviction":false,"capture_actual_consumed_direct_read_file_spec_nested_hits":true,"recursive_JSON_reference_reads":false,"fresh_nonmemo_typed_bounded_path_SHA_UTF8_clock_fence":true,"mandatory_fences":["prepublication","post_durable_close_readback","actual_Popen_after_receipt","post_pre_array_before_first_numeric_native_or_array"],"descriptors_survive_parsed_clear_to_actual_Popen":true,"parent_ENTRY_child_actual_discovery_preserved":true,"independent_fresh_own_project_installed_layout_oracle_live_admission":true}')


@historical_dependencies()
def _current_revalidation_amendment(root=ROOT):
    raw=literal_bytes(REVALIDATION_AMENDMENT,root)
    require(len(raw)==16202 and sha(raw)==REVALIDATION_SHA,'literal authority-scope amendment')
    value=strict_json(raw)
    typed_equal(value['previous_revalidation_amendment'],file_spec(RUFF_RECOVERY_AMENDMENT,root),'fixed p6 authority')
    for key,path in (('historical_p3_revalidation_amendment',OLD_REVALIDATION_AMENDMENT),
                     ('historical_p4_revalidation_amendment',ACTIVE_REVALIDATION_AMENDMENT),
                     ('historical_p5_revalidation_amendment',SOURCE_RECOVERY_AMENDMENT),
                     ('old_resource_extension',RESOURCE_EXTENSION)):
        typed_equal(value[key],file_spec(path,root),'preserved historical authority '+key)
    ruff_recovery_amendment(root)
    expected={'schema':'flylab.m3.e360.authority-scope-amendment.v1','local_cost_offset':42,
              'required_exact_prefix_length':68,'generated_shared_limit_s':600,'generated_local_limit_s':489,
              'new_generated_grant_s':0,'generated_pass_limit':7,'modified_generated_pass_limit':6,
              'original_pass_limit':1,'total_pass_limit':8,'manual_generated_pass_id':'e360-generous-p7',
              'predecessor_pass_id':'e360-generous-p6','predecessor_failed_cap_s':30,'automatic_retries':0,
              'pass8_authorized':False,'whole_fresh_chain_required':True,'primary_total_reserved_s':263,
              'minimum_current_test_functions':40,'minimum_current_test_phases':120,
              'source_hash_cycle_allowed':False,'original_local_limit_s':20,'original_stage_limit_s':300,
              'original_global_limit_s':10200,'original_worker_maximum':1,'original_automatic_retries':0}
    for key,item in expected.items():typed_equal(value[key],item,'current scope policy '+key)
    typed_equal(value['fixed_authority_scope_policy'],FIXED_SCOPE_POLICY,'literal fixed scope policy')
    typed_equal(value['new_active_fields'],['e360_authority_scope_activation','e360_authority_scope_amendment'],'only two new fields')
    for key in CURRENT_REVALIDATION_SOURCE_KEYS:read_spec(value[key],root)
    for spec in value['reviewed_source_snapshots'].values():read_spec(spec,root)
    rows=value['historical_concrete_fix_relocations']
    require(type(rows) is list and len(rows)==18 and [r['pass_id'] for r in rows]==
            ['e360-generous-p2']*2+['e360-generous-p3']*4+['e360-generous-p4']*4+['e360-generous-p5']*4+['e360-generous-p6']*4,
            'exact eighteen pass-bound relocations')
    for row in rows:
        typed_equal(row['original']['bytes'],row['snapshot']['bytes'],'historical relocation bytes')
        typed_equal(row['original']['sha256'],row['snapshot']['sha256'],'historical relocation SHA')
        read_spec(row['snapshot'],root)
    return value


def current_revalidation_event(root=ROOT):
    authority=current_revalidation_amendment(root)
    return {'schema':'flylab.m3.e360.authority-scope-activation.v1',
            'revalidation_amendment':file_spec(REVALIDATION_AMENDMENT,root),
            'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited68':authority['inherited68'],
            'previous_revalidation_amendment':authority['previous_revalidation_amendment'],
            'manual_generated_pass_id':authority['manual_generated_pass_id'],'new_generated_grant_s':authority['new_generated_grant_s']}

def registration(root=ROOT):
    context=execution_context(root)
    if context is not None:return strict_json(read_spec(context['manifest']['registration'],root,owner='historical'))
    return fixed_reader('registration',_registration,root)


def amendment(root=ROOT):
    return fixed_reader('amendment',_amendment,root)


def resource_extension(root=ROOT):
    context=execution_context(root)
    if context is not None:return execution_policy(context)
    return fixed_reader('resource_extension',_resource_extension,root)


def revalidation_amendment(root=ROOT):
    return fixed_reader('revalidation_amendment',_revalidation_amendment,root)


def active_revalidation_amendment(root=ROOT):
    return fixed_reader('active_revalidation_amendment',_active_revalidation_amendment,root)


def source_recovery_amendment(root=ROOT):
    return fixed_reader('source_recovery_amendment',_source_recovery_amendment,root)


def ruff_recovery_amendment(root=ROOT):
    return fixed_reader('ruff_recovery_amendment',_ruff_recovery_amendment,root)


def current_revalidation_amendment(root=ROOT):
    return fixed_reader('current_revalidation_amendment',_current_revalidation_amendment,root)


@historical_dependencies()
def _execution_budget_amendment(root=ROOT):
    raw=literal_bytes(EXECUTION_BUDGET_AMENDMENT,root)
    require(len(raw)==24334 and sha(raw)==EXECUTION_BUDGET_SHA,'literal prospective P8 v2 authority')
    value=strict_json(raw)
    current_revalidation_amendment(root)
    typed_equal(value['previous_revalidation_amendment'],file_spec(REVALIDATION_AMENDMENT,root),'P8 preserves P7 literal')
    expected={'schema':'flylab.m3.e360.execution-budget-amendment.v1','manual_generated_pass_id':'e360-generous-p8',
        'predecessor_pass_id':'e360-generous-p7','required_exact_prefix_length':73,'local_cost_offset':42,
        'generated_shared_limit_s':600,'generated_local_limit_s':489,'new_generated_grant_s':0,
        'generated_pass_limit':8,'modified_generated_pass_limit':7,'original_pass_limit':1,'total_pass_limit':9,
        'automatic_retries':0,'pass9_authorized':False,'original_execution_authority':False,
        'generated_scope_ends_after_Q_debit':True,'primary_total_reserved_s':273,
        'minimum_current_test_functions':52,'minimum_current_test_phases':156,
        'original_local_limit_s':20,'original_stage_limit_s':300,'original_global_limit_s':10200,
        'original_worker_maximum':1,'original_automatic_retries':0,'source_hash_cycle_allowed':False}
    for key,item in expected.items():typed_equal(value[key],item,'P8 literal policy '+key)
    scope=copy.deepcopy(FIXED_SCOPE_POLICY)
    scope['reader_ids'].append('execution_budget_amendment')
    typed_equal(value['fixed_authority_scope_policy'],scope,'ninth fixed reader same bounded scope')
    typed_equal(value['checkpoint_policy'],compiled_checkpoint_policy(),'exact compiled passive checkpoint policy')
    typed_equal(value['new_active_fields'],['e360_execution_budget_activation','e360_execution_budget_amendment'],'only P8 fields')
    for key in EXECUTION_BUDGET_SOURCE_KEYS:read_spec(value[key],root)
    for spec in value['reviewed_source_snapshots']:read_spec(spec,root)
    rows=value['historical_concrete_fix_relocations']
    require(type(rows) is list and len(rows)==22,'exact twenty-two historical relocations')
    typed_equal(rows[:18],current_revalidation_amendment(root)['historical_concrete_fix_relocations'],'all eighteen old relocations')
    for row in rows:
        snapshot=row.get('snapshot',row.get('frozen'))
        typed_equal(row['original']['bytes'],snapshot['bytes'],'relocation bytes')
        typed_equal(row['original']['sha256'],snapshot['sha256'],'relocation hash')
        read_spec(snapshot,root)
    typed_equal([row['pass_id'] for row in rows[18:]],['e360-generous-p7']*4,'four frozen P7 source maps')
    inherited=strict_json(read_spec(value['inherited73'],root))['ledger']
    require(len(inherited['debits'])==73,'immutable inherited73')
    for key in value['preserved_activation_fields']:require(key in inherited,'all ten preserved activation fields')
    return value


def execution_budget_amendment(root=ROOT):
    return fixed_reader('execution_budget_amendment',_execution_budget_amendment,root)


def execution_budget_event(root=ROOT):
    value=execution_budget_amendment(root)
    return {'schema':'flylab.m3.e360.execution-budget-activation.v1',
        'revalidation_amendment':file_spec(EXECUTION_BUDGET_AMENDMENT,root),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited73':value['inherited73'],
        'previous_revalidation_amendment':value['previous_revalidation_amendment'],
        'manual_generated_pass_id':value['manual_generated_pass_id'],'new_generated_grant_s':0}


@historical_dependencies()
def _suite_recovery_amendment(root=ROOT):
    raw=literal_bytes(SUITE_RECOVERY_AMENDMENT,root)
    require(len(raw)==28728 and sha(raw)==SUITE_RECOVERY_SHA,'literal approved P9 authority')
    value=strict_json(raw)
    old=execution_budget_amendment(root)
    typed_equal(value['previous_revalidation_amendment'],file_spec(EXECUTION_BUDGET_AMENDMENT,root),'P9 old P8 direct pin')
    typed_equal(value['resource_extension'],file_spec(RESOURCE_EXTENSION,root),'P9 preserves v2 resource extension')
    expected={'schema':'flylab.m3.e360.suite-recovery-amendment.v1','manual_generated_pass_id':'e360-generous-p9',
        'predecessor_pass_id':'e360-generous-p8','required_exact_prefix_length':81,'local_cost_offset':42,
        'old_shared_generated_limit_s':600,'old_generated_local_limit_s':489,
        'generated_shared_limit_s':1400,'generated_local_limit_s':1289,'new_generated_grant_s':800,
        'generated_pass_limit':9,'modified_generated_pass_limit':8,'original_pass_limit':1,'total_pass_limit':10,
        'automatic_retries':0,'pass9_authorized':True,'pass10_authorized':False,'original_execution_authority':False,
        'generated_scope_ends_after_Q_debit':True,'primary_total_reserved_s':798,'predecessor_failed_cap_s':120,
        'minimum_current_test_functions':71,'minimum_current_test_phases':213,'maximum_current_test_functions':84,
        'original_local_limit_s':20,'original_stage_limit_s':300,'original_global_limit_s':10200,
        'original_worker_maximum':1,'original_automatic_retries':0,'source_hash_cycle_allowed':False,
        'whole_fresh_chain_required':True,'remaining_generated_administration_reserve_s':120,
        'fixture_future_administration_reserved_s':95,'suite_future_administration_reserved_s':50,
        'successful_suffix_after_s':[795,775,750,695,680,650,50,30,0]}
    for key,item in expected.items():typed_equal(value[key],item,'exact P9 '+key)
    scope=copy.deepcopy(FIXED_SCOPE_POLICY)
    scope['reader_ids'].extend(['execution_budget_amendment','suite_recovery_amendment'])
    typed_equal(value['fixed_authority_scope_policy'],scope,'tenth reader unchanged bounded scope')
    typed_equal(value['checkpoint_policy'],compiled_checkpoint_policy(),'P9 same passive checkpoints')
    typed_equal(value['new_active_fields'],list(SUITE_RECOVERY_FIELDS[:2]),'P9 exact activation pair')
    typed_equal(value['new_resource_grant_field'],SUITE_RECOVERY_FIELDS[2],'P9 distinct grant')
    typed_equal(value['generated_call_order'],old['generated_call_order'],'same nine call order')
    roles=copy.deepcopy(old['effective_roles'])
    roles['suite'].update(outer_s=600,child_s=560,internal_s=540,science_s=500)
    typed_equal(value['effective_roles'],roles,'P9 exact worker caps original unchanged')
    typed_equal(value['scanner_limits'],{'outer_s':20,'internal_s':18,'planned_fixture_s':55,'planned_suite_s':600,'remaining_generated_administration_reserve_s':120},'P9 scanner20/18 downstream795')
    for key in SUITE_RECOVERY_SOURCE_KEYS:read_spec(value[key],root)
    for spec in value['reviewed_source_snapshots']:read_spec(spec,root)
    rows=value['historical_concrete_fix_relocations']
    require(type(rows) is list and len(rows)==26,'exact26 relocation roster')
    typed_equal(rows[:22],old['historical_concrete_fix_relocations'],'old22 relocations unchanged')
    typed_equal([row['pass_id'] for row in rows[22:]],['e360-generous-p8']*4,'only four C9 P8 relocations')
    frozen=strict_json(read_spec(value['inherited81'],root))['ledger']
    require(len(frozen['debits'])==81,'exact inherited81')
    for row in rows:
        spec=row.get('snapshot',row.get('frozen'))
        typed_equal(row['original']['bytes'],spec['bytes'],'P9 relocation exact bytes')
        typed_equal(row['original']['sha256'],spec['sha256'],'P9 relocation exact hash')
        read_spec(spec,root)
    approval=strict_json(read_spec(value['existing_human_authorization'],root))
    for key in ('generated_shared_limit_s','generated_local_limit_s','new_generated_grant_s','manual_generated_pass_id'):
        typed_equal(approval[key],value[key],'direct new approval '+key)
    typed_equal(approval['inherited81'],value['inherited81'],'approval immutable81 rather than mutable ledger SHA')
    typed_equal(approval['human_quote'],'추가 배정 승인할게','direct human approval')
    return value


def suite_recovery_amendment(root=ROOT):
    return fixed_reader('suite_recovery_amendment',_suite_recovery_amendment,root)


def suite_recovery_event(root=ROOT):
    value=suite_recovery_amendment(root)
    return {'schema':'flylab.m3.e360.suite-recovery-activation.v1',
        'revalidation_amendment':file_spec(SUITE_RECOVERY_AMENDMENT,root),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited81':value['inherited81'],
        'previous_revalidation_amendment':value['previous_revalidation_amendment'],
        'manual_generated_pass_id':value['manual_generated_pass_id'],'new_generated_grant_s':value['new_generated_grant_s'],
        'human_authorization':value['existing_human_authorization']}


def suite_recovery_grant(root=ROOT):
    value=suite_recovery_amendment(root)
    return {'schema':'flylab.m3.e360.suite-recovery-resource-grant.v1',
        'authorization':value['existing_human_authorization'],'revalidation_amendment':file_spec(SUITE_RECOVERY_AMENDMENT,root),
        'inherited81':value['inherited81'],**{key:value[key] for key in ('old_shared_generated_limit_s',
        'old_generated_local_limit_s','generated_shared_limit_s','generated_local_limit_s','new_generated_grant_s','manual_generated_pass_id')}}


@historical_dependencies()
def _lint_recovery_amendment(root=ROOT):
    """One literal human scope, grant0. Final capture pin is a separate ROOT leaf."""
    raw=literal_bytes(LINT_RECOVERY_AMENDMENT,root)
    require(len(raw)==31323 and sha(raw)==LINT_RECOVERY_SHA,'literal authorized P10 scope')
    value=strict_json(raw)
    old=suite_recovery_amendment(root)
    typed_equal(value['previous_revalidation_amendment'],file_spec(SUITE_RECOVERY_AMENDMENT,root),'P10 preserves P9 authority')
    typed_equal(value['resource_extension'],file_spec(RESOURCE_EXTENSION,root),'P10 existing v2 resources')
    expected={'schema':'flylab.m3.e360.lint-recovery-amendment.v1','manual_generated_pass_id':'e360-generous-p10',
        'predecessor_pass_id':'e360-generous-p9','required_exact_prefix_length':82,'local_cost_offset':42,
        'generated_shared_limit_s':1400,'generated_local_limit_s':1289,'new_generated_grant_s':0,
        'generated_pass_limit':10,'modified_generated_pass_limit':9,'original_pass_limit':1,'total_pass_limit':11,
        'automatic_retries':0,'pass10_authorized':True,'pass11_authorized':False,'original_execution_authority':False,
        'generated_scope_ends_after_Q_debit':True,'primary_total_reserved_s':798,'predecessor_failed_cap_s':3,
        'minimum_current_test_functions':79,'minimum_current_test_phases':237,'maximum_current_test_functions':84,
        'original_local_limit_s':20,'original_stage_limit_s':300,'original_global_limit_s':10200,
        'original_worker_maximum':1,'original_automatic_retries':0,'source_hash_cycle_allowed':False,
        'whole_fresh_chain_required':True,'previous_approved_source_patch_retained':True}
    for key,item in expected.items():typed_equal(value[key],item,'exact P10 '+key)
    for key in ('generated_call_order','generated_call_schedule','effective_helper_limits','effective_roles',
                'scanner_limits','checkpoint_policy','successful_suffix_after_s',
                'remaining_generated_administration_reserve_s','fixture_future_administration_reserved_s',
                'suite_future_administration_reserved_s'):
        typed_equal(value[key],old[key],'P10 unchanged finite P9 schedule '+key)
    scope=copy.deepcopy(old['fixed_authority_scope_policy'])
    scope['reader_ids'].append('lint_recovery_amendment')
    typed_equal(value['fixed_authority_scope_policy'],scope,'eleventh reader same bounds')
    typed_equal(value['new_active_fields'],list(LINT_RECOVERY_FIELDS),'only new P10 pair')
    typed_equal(value['preserved_activation_fields'],[*old['preserved_activation_fields'],*SUITE_RECOVERY_FIELDS[:2]],'all fourteen old fields')
    typed_equal(value['preserved_resource_grant_fields'],['e360_resource_grant',SUITE_RECOVERY_FIELDS[2]],'both old grants')
    typed_equal(value['activation_policy'],{'new_ledger_fields':list(LINT_RECOVERY_FIELDS),'changed_ledger_fields':[],
        'activation_keys':['schema','revalidation_amendment','resource_extension','inherited82',
            'previous_revalidation_amendment','manual_generated_pass_id','new_generated_grant_s','human_authorization'],
        'activation_schema':'flylab.m3.e360.lint-recovery-activation.v1','new_grant_object_allowed':False},'typed pair-only activation policy')
    for key in LINT_RECOVERY_SOURCE_KEYS:read_spec(value[key],root)
    for spec in value['reviewed_source_snapshots']:read_spec(spec,root)
    rows=value['historical_concrete_fix_relocations']
    require(type(rows) is list and len(rows)==30,'exact thirty historical maps')
    typed_equal(rows[:26],old['historical_concrete_fix_relocations'],'all twenty-six old maps')
    typed_equal([row['pass_id'] for row in rows[26:]],['e360-generous-p9']*4,'four frozen P9 v2 maps')
    frozen=strict_json(read_spec(value['inherited82'],root))['ledger']
    require(len(frozen['debits'])==82,'exact immutable82')
    typed_equal(frozen['debits'][:81],strict_json(read_spec(old['inherited81'],root))['ledger']['debits'],'82 retains81')
    for row in rows:
        spec=row.get('snapshot',row.get('frozen'))
        typed_equal(row['original']['bytes'],spec['bytes'],'P10 historical exact bytes')
        typed_equal(row['original']['sha256'],spec['sha256'],'P10 historical exact hash')
        if row['pass_id']=='e360-generous-p9':
            typed_equal(row['original']['sha256'],frozen['e360_passes']['e360-generous-p9']['source_sha256'][row['original']['path']],'P9 historical own9 mapping')
        read_spec(spec,root)
    approval=strict_json(read_spec(value['existing_human_authorization'],root))
    for key,item in {'human_quote':'추가 예산 없이 남은 한도에서 P10을 한 번 진행하도록 승인할게',
        'manual_generated_pass_id':'e360-generous-p10','manual_P10_authorized':True,'new_generated_grant_s':0,
        'existing_shared_limit_s':1400,'existing_local_limit_s':1289,'reserved_whole_generated_schedule_s':798,
        'automatic_retries':0,'P11_authorized':False,'original_execution_authority':False}.items():
        typed_equal(approval[key],item,'direct P10 human scope '+key)
    typed_equal(approval['inherited82'],value['inherited82'],'direct approval frozen82')
    typed_equal(approval['approved_proposal'],value['approved_final_proposal'],'direct reflected scope approval')
    return value


def lint_recovery_amendment(root=ROOT):
    return fixed_reader('lint_recovery_amendment',_lint_recovery_amendment,root)


def lint_recovery_event(root=ROOT):
    value=lint_recovery_amendment(root)
    return {'schema':'flylab.m3.e360.lint-recovery-activation.v1',
        'revalidation_amendment':file_spec(LINT_RECOVERY_AMENDMENT,root),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited82':value['inherited82'],
        'previous_revalidation_amendment':value['previous_revalidation_amendment'],
        'manual_generated_pass_id':value['manual_generated_pass_id'],'new_generated_grant_s':0,
        'human_authorization':value['existing_human_authorization']}


def lint_recovery_activation_allowed(live,*,lock_absent,root=ROOT):
    value=lint_recovery_amendment(root)
    frozen=strict_json(read_spec(value['inherited82'],root))['ledger']
    require(lock_absent is True and not any(key in live for key in LINT_RECOVERY_FIELDS),'once P10 scope ownership0')
    typed_equal(live,frozen,'activation exact entire inherited82 state')
    require(len(live['debits'])==82 and all(not live[key] for key in
        ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations')),'no pending/held ownership')
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(live[key],0,'typed activation ownership0')
    require(all(call['state']=='charged' for call in live['e360_calls'].values()),'all82 costs and calls charged')
    budget=current_budget_authority(root,live=live)
    require(budget['suite_recovery_active'] and budget['generated_shared_limit_s']-live['generated_shared_cumulative_used_s']>=798
        and budget['generated_local_limit_s']-live['e360_generated_local_used_s']>=798,'full798 existing allowance')
    return True


def lint_recovery_delta_valid(before,after,*,root=ROOT):
    require(set(after)==set(before)|set(LINT_RECOVERY_FIELDS),'only two new scope fields')
    for key,item in before.items():typed_equal(after[key],item,'no activation reset or budget change '+key)
    typed_equal(after[LINT_RECOVERY_FIELDS[0]],lint_recovery_event(root),'typed new scope event8')
    typed_equal(after[LINT_RECOVERY_FIELDS[1]],file_spec(LINT_RECOVERY_AMENDMENT,root),'exact new direct scope descriptor')
    return True


def historical_lint_scope_authority(root=ROOT,*,live=None):
    """Fresh scope proof separately from the unchanged current P9 budget selector."""
    if live is None:live=strict_json(bounded_metadata_bytes(registration(root)['shared_ledger_path'],root))
    present=[key in live for key in LINT_RECOVERY_FIELDS]
    require(all(present) or not any(present),'complete P10 scope pair')
    if all(present):
        require(current_budget_authority(root,live=live)['suite_recovery_active'],'P10 retains authenticated P9 grant800')
        typed_equal(live[LINT_RECOVERY_FIELDS[0]],lint_recovery_event(root),'exact P10 event8')
        typed_equal(live[LINT_RECOVERY_FIELDS[1]],file_spec(LINT_RECOVERY_AMENDMENT,root),'exact P10 scope pin')
        value=lint_recovery_amendment(root)
        frozen=strict_json(read_spec(value['inherited82'],root))['ledger']
        typed_equal(live['debits'][:82],frozen['debits'],'all82 actual costs preserved')
        for key in (*value['preserved_activation_fields'],*value['preserved_resource_grant_fields']):
            typed_equal(live[key],frozen[key],'all14 fields and both grants preserved')
        for key in ('e360_calls','e360_passes'):
            for identity,item in frozen[key].items():typed_equal(live[key][identity],item,'full historical82 '+key)
    return all(present)


@historical_dependencies()
def _p11_recovery_amendment(root=ROOT):
    """Immutable P11 authority only: no live baseline reopening or pass recursion."""
    raw=literal_bytes(P11_RECOVERY_AMENDMENT,root)
    require(len(raw)==33809 and sha(raw)==P11_RECOVERY_SHA,'exact approved P11 v2 literal')
    value=strict_json(raw)
    old=lint_recovery_amendment(root)
    expected={'schema':'flylab.m3.e360.p11-recovery-amendment.v1','manual_generated_pass_id':'e360-generous-p11',
        'predecessor_pass_id':'e360-generous-p10','required_exact_prefix_length':83,'local_cost_offset':42,
        'generated_shared_limit_s':1400,'generated_local_limit_s':1289,'new_generated_grant_s':0,
        'generated_pass_limit':11,'modified_generated_pass_limit':10,'original_pass_limit':1,'total_pass_limit':12,
        'automatic_retries':0,'pass11_authorized':True,'pass12_authorized':False,'P10_resume_authorized':False,
        'original_execution_authority':False,'generated_scope_ends_after_Q_debit':True,
        'primary_total_reserved_s':798,'predecessor_failed_cap_s':3,'minimum_current_test_functions':84,
        'minimum_current_test_phases':252,'maximum_current_test_functions':84,
        'original_local_limit_s':20,'original_stage_limit_s':300,'original_global_limit_s':10200,
        'original_worker_maximum':1,'original_automatic_retries':0,'source_hash_cycle_allowed':False,
        'whole_fresh_chain_required':True,'historical_P10_internal_completion_valid':False}
    for key,item in expected.items():typed_equal(value[key],item,'compiled P11 '+key)
    typed_equal(value['previous_revalidation_amendment'],file_spec(LINT_RECOVERY_AMENDMENT,root),'P11 old P10 pin')
    typed_equal(value['resource_extension'],file_spec(RESOURCE_EXTENSION,root),'P11 existing P9 resource')
    for key in ('generated_call_order','generated_call_schedule','effective_helper_limits','effective_roles',
        'scanner_limits','checkpoint_policy','successful_suffix_after_s','remaining_generated_administration_reserve_s',
        'fixture_future_administration_reserved_s','suite_future_administration_reserved_s'):
        typed_equal(value[key],old[key],'unchanged P11 finite schedule '+key)
    scope=copy.deepcopy(old['fixed_authority_scope_policy']);scope['reader_ids'].append('p11_recovery_amendment')
    typed_equal(value['fixed_authority_scope_policy'],scope,'current reader12 historical reader11 same bounds')
    typed_equal(value['preserved_activation_fields'],[*old['preserved_activation_fields'],*LINT_RECOVERY_FIELDS],'old16 exact')
    typed_equal(value['preserved_resource_grant_fields'],old['preserved_resource_grant_fields'],'both grants exact')
    typed_equal(value['new_active_fields'],list(P11_RECOVERY_FIELDS),'P11 pair2 only')
    typed_equal(value['activation_policy'],{'new_ledger_fields':list(P11_RECOVERY_FIELDS),'changed_ledger_fields':[],
        'activation_keys':['schema','revalidation_amendment','resource_extension','inherited83',
            'previous_revalidation_amendment','manual_generated_pass_id','new_generated_grant_s','human_authorization'],
        'activation_schema':'flylab.m3.e360.p11-recovery-activation.v1','new_grant_object_allowed':False},'P11 exact activation policy')
    for key in P11_RECOVERY_SOURCE_KEYS:read_spec(value[key],root)
    for spec in value['reviewed_source_snapshots']:read_spec(spec,root)
    frozen=strict_json(read_spec(value['inherited83'],root))['ledger']
    decision=strict_json(read_spec(value['failed_stage_decision'],root))
    typed_equal(frozen,strict_json(read_spec(decision['closed_ledger_snapshot'],root)),'entire immutable closed83')
    typed_equal(frozen['debits'][:82],strict_json(read_spec(value['historical_inherited82'],root))['ledger']['debits'],'full83 retains82')
    require(len(frozen['debits'])==83,'exact83 costs')
    approval=strict_json(read_spec(value['existing_human_authorization'],root))
    for key,item in {'human_quote':'P11 승인할게','manual_generated_pass_id':'e360-generous-p11',
        'manual_P11_authorized':True,'new_generated_grant_s':0,'existing_shared_limit_s':1400,
        'existing_local_limit_s':1289,'reserved_whole_generated_schedule_s':798,'automatic_retries':0,
        'P12_authorized':False,'original_execution_authority':False,'baseline_debits':83,
        'baseline_shared_used_s':503.9470156,'baseline_local_used_s':392.99}.items():
        typed_equal(approval[key],item,'direct P11 approval '+key)
    typed_equal(approval['baseline_live_ledger'],decision['current_ledger'],'C1 direct approval baseline descriptor')
    for key in ('bytes','sha256'):
        typed_equal(decision['current_ledger'][key],decision['closed_ledger_snapshot'][key],'baseline immutable content pin')
    rows=value['historical_concrete_fix_relocations']
    require(type(rows) is list and len(rows)==34,'old30 and frozen P10 four')
    typed_equal(rows[:30],old['historical_concrete_fix_relocations'],'old30 unchanged')
    typed_equal([row['pass_id'] for row in rows[30:]],['e360-generous-p10']*4,'four P10 frozen maps')
    for row in rows:
        spec=row.get('snapshot',row.get('frozen'))
        for key in ('bytes','sha256'):typed_equal(row['original'][key],spec[key],'exact frozen relocation '+key)
        if row['pass_id']=='e360-generous-p10':
            typed_equal(spec['sha256'],frozen['e360_passes']['e360-generous-p10']['source_sha256'][row['original']['path']],'P10 frozen4')
        read_spec(spec,root)
    require(len(value['preserved_test_functions'])==84 and len(set(value['preserved_test_functions']))==84,'source roster84')
    require(set(old['preserved_test_functions'])<=set(value['preserved_test_functions']),'historical79 retained')
    return value


def p11_recovery_amendment(root=ROOT):
    return fixed_reader('p11_recovery_amendment',_p11_recovery_amendment,root)


def p11_recovery_event(root=ROOT):
    value=p11_recovery_amendment(root)
    return {'schema':'flylab.m3.e360.p11-recovery-activation.v1','revalidation_amendment':file_spec(P11_RECOVERY_AMENDMENT,root),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited83':value['inherited83'],
        'previous_revalidation_amendment':value['previous_revalidation_amendment'],
        'manual_generated_pass_id':value['manual_generated_pass_id'],'new_generated_grant_s':0,
        'human_authorization':value['existing_human_authorization']}


def p11_recovery_activation_allowed(live,*,lock_absent,root=ROOT):
    value=p11_recovery_amendment(root)
    frozen=strict_json(read_spec(value['inherited83'],root))['ledger']
    require(lock_absent is True and not any(key in live for key in P11_RECOVERY_FIELDS),'once P11 ownership0')
    typed_equal(live,frozen,'C1 activation exact whole live83')
    require(len(live['debits'])==83 and all(not live[key] for key in
        ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations')),'no pending/held ownership')
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(live[key],0,'typed P11 ownership0')
    require(all(call['state']=='charged' for call in live['e360_calls'].values()),'all83 calls charged')
    budget=current_budget_authority(root,live=live)
    require(budget['suite_recovery_active'] and budget['generated_shared_limit_s']-live['generated_shared_cumulative_used_s']>=798
        and budget['generated_local_limit_s']-live['e360_generated_local_used_s']>=798,'entire798 existing residual')
    return True


def p11_recovery_delta_valid(before,after,*,root=ROOT):
    require(set(after)==set(before)|set(P11_RECOVERY_FIELDS),'only P11 new pair2')
    for key,item in before.items():typed_equal(after[key],item,'no P11 reset '+key)
    typed_equal(after[P11_RECOVERY_FIELDS[0]],p11_recovery_event(root),'P11 event8')
    typed_equal(after[P11_RECOVERY_FIELDS[1]],file_spec(P11_RECOVERY_AMENDMENT,root),'P11 exact direct pin')
    return True


def historical_p11_scope_authority(root=ROOT,*,live=None):
    """Uncached current P11 scope; immutable reader has already completed."""
    if live is None:live=strict_json(bounded_metadata_bytes(registration(root)['shared_ledger_path'],root))
    present=[key in live for key in P11_RECOVERY_FIELDS]
    require(all(present) or not any(present),'complete P11 pair')
    if all(present):
        require(current_budget_authority(root,live=live)['suite_recovery_active'],'P11 retained P9 grant')
        value=p11_recovery_amendment(root)
        typed_equal(live[P11_RECOVERY_FIELDS[0]],p11_recovery_event(root),'exact P11 event8')
        typed_equal(live[P11_RECOVERY_FIELDS[1]],file_spec(P11_RECOVERY_AMENDMENT,root),'P11 current pin')
        frozen=strict_json(read_spec(value['inherited83'],root))['ledger']
        require(set(live)==set(frozen)|set(P11_RECOVERY_FIELDS),'P11 exact baseline fields plus pair2 only')
        typed_equal(live['debits'][:83],frozen['debits'],'all83 actual costs preserved')
        for key in (*value['preserved_activation_fields'],*value['preserved_resource_grant_fields']):
            typed_equal(live[key],frozen[key],'old16 and grants2 '+key)
        for key in ('e360_calls','e360_passes'):
            for identity,item in frozen[key].items():typed_equal(live[key][identity],item,'whole historical83 '+key)
    return all(present)


@historical_dependencies()
def _p12_recovery_amendment(root=ROOT):
    """Immutable current13 reader. Completed P11 proof cannot select current admission."""
    raw=literal_bytes(P12_RECOVERY_AMENDMENT,root)
    require(len(raw)==50865 and sha(raw)==P12_RECOVERY_SHA,'exact approved P12 literal')
    value=strict_json(raw)
    old=p11_recovery_amendment(root)
    expected={'schema':'flylab.m3.e360.p12-recovery-amendment.v1',
        'manual_generated_pass_id':'e360-generous-p12','predecessor_pass_id':'e360-generous-p11',
        'required_exact_prefix_length':88,'local_cost_offset':42,'generated_shared_limit_s':1400,
        'generated_local_limit_s':1289,'new_generated_grant_s':0,'generated_pass_limit':12,
        'modified_generated_pass_limit':11,'original_pass_limit':1,'total_pass_limit':13,
        'automatic_retries':0,'original_local_limit_s':20,'original_stage_limit_s':300,
        'original_global_limit_s':10200,'original_worker_maximum':1,'original_automatic_retries':0,
        'source_hash_cycle_allowed':False,'whole_fresh_chain_required':True,'primary_total_reserved_s':798,
        'predecessor_failed_cap_s':55,'minimum_current_test_functions':84,'maximum_current_test_functions':84,
        'minimum_current_test_phases':252,'pass12_authorized':True,'pass13_authorized':False,
        'P11_resume_authorized':False,'P10_resume_authorized':False,'original_execution_authority':False,
        'generated_scope_ends_after_Q_debit':True}
    for key,item in expected.items():typed_equal(value[key],item,'compiled P12 '+key)
    typed_equal(value['previous_revalidation_amendment'],file_spec(P11_RECOVERY_AMENDMENT,root),'P12 immutable reader12')
    typed_equal(value['resource_extension'],file_spec(RESOURCE_EXTENSION,root),'existing resources no grant')
    for key in ('generated_call_order','generated_call_schedule','effective_helper_limits','effective_roles',
        'scanner_limits','checkpoint_policy','successful_suffix_after_s','remaining_generated_administration_reserve_s',
        'fixture_future_administration_reserved_s','suite_future_administration_reserved_s','preserved_test_functions'):
        typed_equal(value[key],old[key],'P12 exact finite inherited schedule/roster '+key)
    scope=copy.deepcopy(old['fixed_authority_scope_policy']);scope['reader_ids'].append('p12_recovery_amendment')
    typed_equal(value['fixed_authority_scope_policy'],scope,'reader13 unchanged16/512/262144/2097152')
    typed_equal(value['preserved_activation_fields'],[*old['preserved_activation_fields'],*P11_RECOVERY_FIELDS],'old18 exact')
    typed_equal(value['preserved_resource_grant_fields'],old['preserved_resource_grant_fields'],'grants2 exact')
    typed_equal(value['new_active_fields'],list(P12_RECOVERY_FIELDS),'pair2 only')
    typed_equal(value['activation_policy'],{'activation_schema':'flylab.m3.e360.p12-recovery-activation.v1',
        'new_grant_object_allowed':False,'new_ledger_fields':list(P12_RECOVERY_FIELDS),'changed_ledger_fields':[],
        'activation_keys':['schema','revalidation_amendment','resource_extension','inherited88',
        'previous_revalidation_amendment','manual_generated_pass_id','new_generated_grant_s','human_authorization']},
        'P12 activation exact8')
    rows=value['historical_concrete_fix_relocations']
    require(type(rows) is list and len(rows)==38,'old34 append P11 four')
    typed_equal(rows[:34],old['historical_concrete_fix_relocations'],'unchanged ordered old34')
    typed_equal([row['pass_id'] for row in rows[34:]],['e360-generous-p11']*4,'P11 frozen4 only')
    for row in rows:
        pinned=row.get('snapshot',row.get('frozen'))
        for key in ('bytes','sha256'):typed_equal(row['original'][key],pinned[key],'exact relocation '+key)
        read_spec(pinned,root)
    for key in P12_RECOVERY_SOURCE_KEYS:read_spec(value[key],root)
    # Integration bases have their own pins; compact Source2 deliberately differs from historic roster.
    require(len(value['reviewed_source_snapshots'])==4,'exact integration bases4')
    for spec in value['reviewed_source_snapshots']:read_spec(spec,root)
    approval=strict_json(read_spec(value['existing_human_authorization'],root))
    for key,item in {'human_quote':'P12 승인할게','manual_generated_pass_id':'e360-generous-p12',
        'manual_P12_authorized':True,'new_generated_grant_s':0,'existing_shared_limit_s':1400,
        'existing_local_limit_s':1289,'reserved_whole_generated_schedule_s':798,'automatic_retries':0,
        'P13_authorized':False,'P11_resume_authorized':False,'original_execution_authority':False,
        'baseline_debits':88,'baseline_shared_used_s':539.2770156,'baseline_local_used_s':428.32}.items():
        typed_equal(approval[key],item,'direct P12 approval '+key)
    completed_historical_p11_proof(root,value=value)
    return value


def p12_recovery_amendment(root=ROOT):
    return fixed_reader('p12_recovery_amendment',_p12_recovery_amendment,root)


@historical_dependencies()
def completed_historical_p11_proof(root=ROOT,*,value=None):
    """Acyclic immutable completed proof: no selector/own9/PID/clock/closed_exit/validate_contract."""
    if value is None:
        raw=literal_bytes(P12_RECOVERY_AMENDMENT,root)
        require(len(raw)==50865 and sha(raw)==P12_RECOVERY_SHA,'historical proof direct immutable P12 pin')
        value=strict_json(raw)
    frozen=strict_json(read_spec(value['inherited88'],root))['ledger']
    typed_equal(frozen,strict_json(read_spec(value['historical_P11_closed_ledger88'],root)),'whole closed88')
    prior=strict_json(read_spec(value['historical_P11_inherited83'],root))['ledger']
    require(len(frozen['debits'])==88 and len(frozen['e360_calls'])==38 and len(frozen['e360_passes'])==11,
        'exact completed88/38/11')
    typed_equal(frozen['debits'][:83],prior['debits'],'whole old83 prefix')
    for key in (*value['preserved_resource_grant_fields'],*value['preserved_activation_fields'][:-2]):
        typed_equal(frozen[key],prior[key],'completed historical preserved '+key)
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(frozen[key][identity],item,'whole inherited83 '+key)
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):
        typed_equal(frozen[key],{},'completed ownership0 '+key)
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(frozen[key],0,'completed original0')
    passed=frozen['e360_passes']['e360-generous-p11']
    typed_equal(passed['predecessor_failed_cap_s'],3,'historical P11 prior P10 cap stays3')
    typed_equal(passed['failed_cap_s'],55,'actual failed worker cap55')
    typed_equal(passed['next_index'],5,'completed five calls')
    typed_equal(passed['failed_cost'],value['actual_predecessor_failure'],'actual failure pin')
    typed_equal(passed['source_sha256'],value['historical_P11_frozen_own_sources_sha256'],'historic exact own9')
    own=passed['source_sha256'];require(len(own)==9,'historical own9 only')
    relocations=value['historical_concrete_fix_relocations'][34:38]
    relocated={row['original']['path']:row for row in relocations}
    require(len(relocated)==4,'exact frozen4')
    for row in relocations:
        typed_equal(row['original']['sha256'],own[row['original']['path']],'P11 original own pin')
        read_spec(row['frozen'],root)
    for name,digest in own.items():
        if name not in relocated:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'historic remaining immutable5')
    for spec in passed['concrete_fix']:
        if spec['path'] in relocated:
            typed_equal(spec,relocated[spec['path']]['original'],'registered exact historic descriptor')
            read_spec(relocated[spec['path']]['frozen'],root)
        else:read_spec(spec,root)
    costs=value['historical_P11_actual_paid_costs']
    typed_equal(frozen['debits'][83:88],costs,'exact debit order: seal before worker, never sort')
    require(len({row['cost_id'] for row in costs})==5,'five unique costs')
    returns=strict_json(read_spec(value['historical_P11_ROOT_genuine_actual_returns'],root))
    dispatch=returns['actual_calls_in_dispatch_order']
    typed_equal([row['role'] for row in dispatch],value['historical_P11_actual_dispatch_order'],'actual dispatch worker then seal')
    typed_equal(returns['actual_call_count'],5,'five genuine final returns')
    typed_equal(returns['actual_total_outer_full_return_s'],35.33,'actual five elapsed sum')
    call_rows=[frozen['e360_calls'][identity] for identity in frozen['e360_calls']
        if frozen['e360_calls'][identity]['pass_id']=='e360-generous-p11']
    require(len(call_rows)==5 and {row['sequence_index'] for row in call_rows}==set(range(5)),'five actual unique calls')
    for measured in dispatch:
        matches=[call for call in call_rows if call['call_role']==measured['role']]
        require(len(matches)==1,'one actual dispatch match')
        call=matches[0]
        matches=[row for row in costs if row['cost_id']==call['cost_id']]
        require(len(matches)==1 and call['state']=='charged','once charged actual dispatch')
        debit=matches[0];cost=strict_json(read_spec(debit['descriptor'],root))
        actual=strict_json(read_spec(cost['actual_tool_return'],root))
        request=strict_json(read_spec(call['request'],root,owner='dynamic' if call['pass_id']=='e360-generous-p30' else 'historical'))
        for key in ('call_id','pass_id'):
            typed_equal(cost[key],actual[key],'cost actual identity')
            typed_equal(cost[key],request[key],'request actual identity')
        typed_equal(request['call_role'],call['call_role'],'direct call role')
        typed_equal(request['source_sha256'],own,'historic request own9')
        typed_equal(actual['request'],call['request'],'actual same admitted request')
        typed_equal(cost['actual_outer_full_return_s'],debit['actual_outer_full_return_s'],'once elapsed')
        typed_equal(measured['actual_outer_full_return_s'],debit['actual_outer_full_return_s'],'genuine final-return elapsed')
        typed_equal(round((measured['final_return_ms']-measured['dispatch_ms'])/1000,3),
            measured['actual_outer_full_return_s'],'genuine dispatch through final return')
        typed_equal(measured['outer_cap_s'],call['reserved_s'],'actual role cap')
        typed_equal(measured['result']['exit_code'],debit['actual_returned_exit_code'],'actual tool exit')
        typed_equal(actual['actual_returned_exit_code'],debit['actual_returned_exit_code'],'schema separate actual exit')
        typed_equal(actual['actual_outer_full_return_s'],debit['actual_outer_full_return_s'],'actual measured elapsed')
        typed_equal(cost['revalidation_amendment'],value['previous_revalidation_amendment'],'historic direct authority')
        typed_equal(actual['revalidation_amendment'],cost['revalidation_amendment'],'same actual authority')
        require(cost['charged_once'] is True,'historical cost charged once')
        if measured['role']!='fixture_worker':
            typed_equal(cost['schema'],'flylab.m3.e360.administrative-full-return.v1','admin schema only four')
            typed_equal(cost['actual_returned_exit_code'],0,'historical admin exit0')
            require(cost['internal_completion_valid'] is True,'historical admin completion')
    failure=strict_json(read_spec(value['actual_predecessor_failure'],root))
    outer=strict_json(read_spec(value['historical_P11_outer'],root))
    actual=strict_json(read_spec(value['historical_P11_actual_tool_return'],root))
    contract=strict_json(read_spec(value['historical_P11_contract'],root))
    decision=strict_json(read_spec(value['failed_stage_decision'],root))
    for key,item in {'schema':'flylab.m3.e360.actual-full-return.v1','mode':'generated','role':'fixture',
        'pass_id':'e360-generous-p11','attempt_id':'e360-generous-p11-f1','call_id':'e360-generous-p11-fworker',
        'exit_code':1,'qualified':False,'actual_outer_full_return_s':8.479,'envelope':None,
        'authority_admitted':False,'Popen_never_attempted':True,'Popen_attempted':False,'Popen_returned':False,
        'all_owned_ended':True,'reservation_released':True,'charged_once':True}.items():
        typed_equal(failure[key],item,'schema-specific worker failure '+key)
    typed_equal(failure['actual_tool_return'],value['historical_P11_actual_tool_return'],'separate actual pin')
    typed_equal(actual['actual_returned_exit_code'],1,'actual-return worker exit1 separately')
    typed_equal(actual['actual_outer_full_return_s'],8.479,'actual worker elapsed separately')
    typed_equal(failure['outer'],value['historical_P11_outer'],'pinned outer')
    typed_equal(failure['contract'],value['historical_P11_contract'],'pinned historical contract')
    typed_equal(outer['pre_spawn'],None,'no invented receipt')
    typed_equal(outer['envelope'],None,'no invented envelope')
    typed_equal(outer['contract'],value['historical_P11_contract'],'outer historical contract')
    typed_equal(outer['source_sha256'],contract['source_sha256'],'outer complete historical552')
    for key in ('registration','amendment','resource_extension','revalidation_amendment'):
        typed_equal(outer[key],contract[key],'outer historical direct authority '+key)
    for key in ('first_error','all_owned_ended','reservation_released','authority_admitted','Popen_never_attempted'):
        typed_equal(outer[key],failure[key],'same original failed outer evidence '+key)
    worker=frozen['e360_calls']['e360-generous-p11-fworker']
    worker_request=strict_json(read_spec(worker['request'],root))
    typed_equal(worker['attempt_id'],contract['attempt_id'],'historic worker attempt')
    typed_equal(worker_request['attempt_id'],contract['attempt_id'],'historic request attempt')
    typed_equal(actual['request'],worker['request'],'worker distinct actual same request')
    typed_equal(actual['call_id'],failure['call_id'],'worker actual same call')
    typed_equal(actual['pass_id'],failure['pass_id'],'worker actual same pass')
    for key in ('pass_id','attempt_id','role','mode','source_sha256','closure_report','registration',
        'amendment','resource_extension','revalidation_amendment'):
        typed_equal(failure[key],contract[key],'full historic contract identity '+key)
    require(len(contract['source_sha256'])==552,'unchanged historical full552')
    typed_equal({key:contract['source_sha256'].get(key) for key in own},own,'separate historic exact own9 projection')
    for key in ('closure_report','registration','amendment','resource_extension','revalidation_amendment'):
        read_spec(contract[key],root)
    typed_equal(failure['first_error'],{'type':'ValueError','message':'fresh bounded publication','stage':'outer','notes':[]},
        'original first error')
    typed_equal(failure['secondary_errors'],[{'type':'RuntimeError','message':'actual tool returned exit code 1',
        'stage':'actual-full-return','notes':[]}],'secondary actual failure')
    typed_equal(decision['status'],'CLOSED_FAILED','closed failed decision')
    typed_equal(decision['failed_full_exit'],value['actual_predecessor_failure'],'decision exact failed full-exit')
    typed_equal(decision['closed_ledger_snapshot'],value['historical_P11_closed_ledger88'],'decision whole closed88')
    typed_equal(decision['approved_literal'],value['previous_revalidation_amendment'],'decision historic authority')
    typed_equal(decision['actual_ROOT_runtime_returns'],value['historical_P11_ROOT_genuine_actual_returns'],'decision genuine returns')
    return frozen


def validate_completed_p11_pass(pass_id,passed,debits,*,root=ROOT):
    frozen=completed_historical_p11_proof(root)
    typed_equal(pass_id,'e360-generous-p11','only completed historical P11')
    typed_equal(passed,frozen['e360_passes'][pass_id],'whole immutable historical P11 pass')
    typed_equal(debits[:88],frozen['debits'],'whole exact inherited88 costs')


def p12_recovery_event(root=ROOT):
    value=p12_recovery_amendment(root)
    return {'schema':'flylab.m3.e360.p12-recovery-activation.v1','revalidation_amendment':file_spec(P12_RECOVERY_AMENDMENT,root),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited88':value['inherited88'],
        'previous_revalidation_amendment':value['previous_revalidation_amendment'],
        'manual_generated_pass_id':value['manual_generated_pass_id'],'new_generated_grant_s':0,
        'human_authorization':value['existing_human_authorization']}


def p12_recovery_activation_allowed(live,*,lock_absent,root=ROOT):
    value=p12_recovery_amendment(root);frozen=completed_historical_p11_proof(root,value=value)
    require(lock_absent is True and not any(key in live for key in P12_RECOVERY_FIELDS),'once current ownership0')
    typed_equal(live,frozen,'activation exact whole88')
    budget=current_budget_authority(root,live=live)
    require(budget['suite_recovery_active'] and budget['generated_shared_limit_s']-live['generated_shared_cumulative_used_s']>=798
        and budget['generated_local_limit_s']-live['e360_generated_local_used_s']>=798,'whole798 existing residual')
    return True


def p12_recovery_delta_valid(before,after,*,root=ROOT):
    require(set(after)==set(before)|set(P12_RECOVERY_FIELDS),'only new P12 pair')
    for key,item in before.items():typed_equal(after[key],item,'whole old88 unchanged '+key)
    typed_equal(after[P12_RECOVERY_FIELDS[0]],p12_recovery_event(root),'exact activation8')
    typed_equal(after[P12_RECOVERY_FIELDS[1]],file_spec(P12_RECOVERY_AMENDMENT,root),'current direct literal')
    return True


def historical_p12_scope_authority(root=ROOT,*,live=None):
    if live is None:live=strict_json(bounded_metadata_bytes(registration(root)['shared_ledger_path'],root))
    present=[key in live for key in P12_RECOVERY_FIELDS]
    require(all(present) or not any(present),'complete historical P12 pair')
    if not all(present):return False
    typed_equal(live[P12_RECOVERY_FIELDS[0]],p12_recovery_event(root),'historic P12 activation')
    typed_equal(live[P12_RECOVERY_FIELDS[1]],file_spec(P12_RECOVERY_AMENDMENT,root),'historic P12 literal')
    frozen=completed_historical_p12_proof(root)
    typed_equal(live['debits'][:89],frozen['debits'],'whole completed89 prefix')
    for key in ('e360_calls','e360_passes'):
        for identity,item in frozen[key].items():typed_equal(live[key][identity],item,'whole completed89 state')
    return True


def validate_p12_pass(pass_id,passed,debits,*,root=ROOT):
    return validate_completed_p12_pass(pass_id,passed,debits,root=root)


@historical_dependencies()
def _p13_recovery_amendment(root=ROOT):
    """Immutable reader14; historical P12 evidence never supplies current admission."""
    raw=literal_bytes(P13_RECOVERY_AMENDMENT,root)
    require(len(raw)==P13_RECOVERY_BYTES and sha(raw)==P13_RECOVERY_SHA,'exact approved P13 literal')
    value=strict_json(raw)
    old=p12_recovery_amendment(root)
    expected={'schema':'flylab.m3.e360.p13-recovery-amendment.v1',
        'manual_generated_pass_id':'e360-generous-p13','predecessor_pass_id':'e360-generous-p12',
        'required_exact_prefix_length':89,'local_cost_offset':42,'generated_shared_limit_s':1400,
        'generated_local_limit_s':1289,'new_generated_grant_s':0,'generated_pass_limit':13,
        'modified_generated_pass_limit':12,'original_pass_limit':1,'total_pass_limit':14,
        'automatic_retries':0,'original_local_limit_s':20,'original_stage_limit_s':300,
        'original_global_limit_s':10200,'original_worker_maximum':1,'original_automatic_retries':0,
        'source_hash_cycle_allowed':False,'whole_fresh_chain_required':True,'primary_total_reserved_s':798,
        'predecessor_failed_cap_s':3,'minimum_current_test_functions':84,'maximum_current_test_functions':84,
        'minimum_current_test_phases':252,'pass13_authorized':True,'pass14_authorized':False,'P12_resume_authorized':False,
        'P11_resume_authorized':False,'P10_resume_authorized':False,'original_execution_authority':False,
        'generated_scope_ends_after_Q_debit':True}
    for key,item in expected.items():typed_equal(value[key],item,'compiled P13 '+key)
    typed_equal(value['previous_revalidation_amendment'],file_spec(P12_RECOVERY_AMENDMENT,root),'P12 immutable reader13')
    typed_equal(value['resource_extension'],file_spec(RESOURCE_EXTENSION,root),'existing resources no grant')
    for key in ('generated_call_order','generated_call_schedule','effective_helper_limits','effective_roles',
        'scanner_limits','checkpoint_policy','successful_suffix_after_s','remaining_generated_administration_reserve_s',
        'fixture_future_administration_reserved_s','suite_future_administration_reserved_s','preserved_test_functions'):
        typed_equal(value[key],old[key],'P12 exact finite inherited schedule/roster '+key)
    scope=copy.deepcopy(old['fixed_authority_scope_policy']);scope['reader_ids'].append('p13_recovery_amendment')
    typed_equal(value['fixed_authority_scope_policy'],scope,'reader14 unchanged16/512/262144/2097152')
    typed_equal(value['preserved_activation_fields'],[*old['preserved_activation_fields'],*P12_RECOVERY_FIELDS],'old20 exact')
    typed_equal(value['preserved_resource_grant_fields'],old['preserved_resource_grant_fields'],'grants2 exact')
    typed_equal(value['new_active_fields'],list(P13_RECOVERY_FIELDS),'pair2 only')
    typed_equal(value['activation_policy'],{'activation_schema':'flylab.m3.e360.p13-recovery-activation.v1',
        'new_grant_object_allowed':False,'new_ledger_fields':list(P13_RECOVERY_FIELDS),'changed_ledger_fields':[],
        'activation_keys':['schema','revalidation_amendment','resource_extension','inherited89',
        'previous_revalidation_amendment','manual_generated_pass_id','new_generated_grant_s','human_authorization']},
        'P13 activation exact8')
    rows=value['historical_concrete_fix_relocations']
    require(type(rows) is list and len(rows)==42,'old38 append P12 four')
    typed_equal(rows[:38],old['historical_concrete_fix_relocations'],'unchanged ordered old38')
    typed_equal([row['pass_id'] for row in rows[38:]],['e360-generous-p12']*4,'P12 frozen4 only')
    for row in rows:
        pinned=row.get('snapshot',row.get('frozen'))
        for key in ('bytes','sha256'):typed_equal(row['original'][key],pinned[key],'exact relocation '+key)
        read_spec(pinned,root)
    for key in P13_RECOVERY_SOURCE_KEYS:read_spec(value[key],root)
    # Integration bases have their own pins; compact Source2 deliberately differs from historic roster.
    require(len(value['reviewed_source_snapshots'])==4,'exact integration bases4')
    for spec in value['reviewed_source_snapshots']:read_spec(spec,root)
    approval=strict_json(read_spec(value['existing_human_authorization'],root))
    for key,item in {'human_quote':'P13 승인할게','manual_generated_pass_id':'e360-generous-p13',
        'manual_P13_authorized':True,'new_generated_grant_s':0,'existing_shared_limit_s':1400,
        'existing_local_limit_s':1289,'reserved_whole_generated_schedule_s':798,'automatic_retries':0,
        'P14_authorized':False,'P12_resume_authorized':False,'original_execution_authority':False,
        'baseline_debits':89,'baseline_shared_used_s':540.7000156,'baseline_local_used_s':429.743}.items():
        typed_equal(approval[key],item,'direct P13 approval '+key)
    typed_equal(value['prospective_native_source_paths'],['verification/banc-natural-gait-m3-20260919/parent-e360-p13-prepare-native.ps1','verification/banc-natural-gait-m3-20260919/parent-e360-p13-bookkeeping-native.ps1','verification/banc-natural-gait-m3-20260919/parent-e360-p13-ROOT-dispatch-adapter.ps1'],'exact prospective Native3 paths; future hashes only freeze/request')
    typed_equal(value['generated_scope_ends_after_first_closed_failure'],True,'first closed failure stops')
    completed_historical_p12_proof(root,value=value)
    return value


def p13_recovery_amendment(root=ROOT):
    return fixed_reader('p13_recovery_amendment',_p13_recovery_amendment,root)


def historical_p12_stream_bytes(spec,root=ROOT):
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
            if spec['bytes']:scope._consumed(spec,root,owner='historical')
            else:
                empty_paths=(BASE+'/parent-e360-generous-p12-ruff-stderr.txt',BASE+'/parent-e360-generous-p13-ruff-stderr.txt',
                    BASE+'/parent-e360-generous-p14-ruff-stdout.txt',BASE+'/parent-e360-generous-p14-ruff-stderr.txt',
                    *(row['path'] for row in _COMPLETED_STREAM_PINS if row['bytes']==0))
                require(spec['path'] in empty_paths,'fixed empty completed stderr only')
                typed_equal(spec,{'path':spec['path'],'bytes':0,
                    'sha256':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'},'fixed empty stderr anchor')
                require(Path(root).resolve()==scope.root,'stream canonical root')
                scope._consumed(spec,root,owner='historical',allow_empty=True)
            scope.clock()
        except Exception:
            scope.poison();raise
    ADMIN_CHECK()
    return raw

@historical_dependencies()
def completed_historical_p12_proof(root=ROOT,*,value=None):
    """Immutable admin failure proof, with no current selector/own discovery/clock/PID."""
    if value is None:
        raw=literal_bytes(P13_RECOVERY_AMENDMENT,root)
        require(len(raw)==P13_RECOVERY_BYTES and sha(raw)==P13_RECOVERY_SHA,'historical direct immutable P13 pin')
        value=strict_json(raw)
    old=p12_recovery_amendment(root)
    prior=completed_historical_p11_proof(root,value=old)
    frozen=strict_json(read_spec(value['inherited89'],root))['ledger']
    typed_equal(frozen,strict_json(read_spec(value['historical_P12_closed_ledger89'],root)),'whole immutable closed89')
    typed_equal(prior,strict_json(read_spec(value['historical_P12_inherited88'],root))['ledger'],'exact prior88')
    require(len(frozen['debits'])==89 and len(frozen['e360_calls'])==39 and len(frozen['e360_passes'])==12,'completed89/39/12')
    typed_equal(frozen['debits'][:88],prior['debits'],'whole ordered prefix88')
    for key in (*value['preserved_resource_grant_fields'],*old['preserved_activation_fields']):
        typed_equal(frozen[key],prior[key],'completed old field '+key)
    typed_equal(frozen[P12_RECOVERY_FIELDS[0]],p12_recovery_event(root),'historical P12 event8')
    typed_equal(frozen[P12_RECOVERY_FIELDS[1]],value['previous_revalidation_amendment'],'historical P12 literal')
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(frozen[key][identity],item,'whole completed88 '+key)
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):
        typed_equal(frozen[key],{},'completed ownership0 '+key)
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(frozen[key],0,'original0')
    passed=frozen['e360_passes']['e360-generous-p12'];own=value['historical_P12_frozen_own_sources_sha256']
    for key,item in {'mode':'generated','modified':True,'predecessor_failed_cap_s':55,'failed_cap_s':3,'next_index':1,
        'actual_failure':old['actual_predecessor_failure'],'failed_cost':value['actual_predecessor_failure'],
        'source_sha256':own,'revalidation_amendment':value['previous_revalidation_amendment']}.items():
        typed_equal(passed[key],item,'historical P12 pass '+key)
    require(type(own) is dict and len(own)==9,'historic own9')
    rows=value['historical_concrete_fix_relocations'][38:42]
    require(len(rows)==4 and [r['pass_id'] for r in rows]==['e360-generous-p12']*4,'ordered P12 frozen4')
    relocated={r['original']['path']:r for r in rows};require(len(relocated)==4,'distinct frozen4')
    freeze=strict_json(read_spec(value['historical_P12_final_source_freeze'],root))
    typed_equal(freeze['own_sources_sha256'],own,'actual frozen P12 own9')
    for row,actual_row in zip(rows,freeze['source_copies'],strict=True):
        typed_equal(row['original'],actual_row['current'],'exact historical original descriptor')
        typed_equal(row['frozen'],actual_row['frozen'],'actual frozen Source4 descriptor')
        typed_equal(row['original']['sha256'],own[row['original']['path']],'historic Source4 own pin')
        for key in ('bytes','sha256'):typed_equal(row['original'][key],row['frozen'][key],'same-byte relocation')
        read_spec(row['frozen'],root)
    for name,digest in own.items():
        if name not in relocated:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'historic direct remaining5')
    for spec in passed['concrete_fix']:
        if spec['path'] in relocated:
            typed_equal(spec,relocated[spec['path']]['original'],'historic concrete original')
            read_spec(relocated[spec['path']]['frozen'],root)
        else:read_spec(spec,root)
    costs=value['historical_P12_actual_paid_costs'];typed_equal(frozen['debits'][88:89],costs,'one exact index88 debit')
    require(len(costs)==1,'once completed Ruff debit')
    typed_equal(value['historical_P12_full_return'],value['actual_predecessor_failure'],'direct historical admin pin')
    typed_equal(value['historical_P12_actual_dispatch_order'],['ruff'],'one historic dispatch')
    debit=costs[0];cost=strict_json(read_spec(value['actual_predecessor_failure'],root))
    actual=strict_json(read_spec(value['historical_P12_actual_tool_return'],root))
    request=strict_json(read_spec(value['historical_P12_actual_request'],root))
    call_id='e360-generous-p12-ruff'
    require(type(frozen['e360_calls']) is dict and call_id in frozen['e360_calls'],'historical exact call dictionary key')
    call=frozen['e360_calls'][call_id]
    # The call ID is the ledger key; the actual historical row has cost_id and no inner call_id.
    require(type(call) is dict and set(call)=={'attempt_id','call_role','cost_id','future_reserved_s','mode','pass_id',
        'request','reserved_s','resource_extension','revalidation_amendment','sequence_index','state'},'exact historical P12 call row schema')
    typed_equal(call['cost_id'],call_id,'historical dictionary key/call cost_id')
    for item in (cost,request,actual):typed_equal(item['call_id'],call_id,'historical dictionary key/cost request actual call_id')
    for key,item in {'schema':'flylab.m3.e360.administrative-full-return.v1','mode':'generated','call_id':'e360-generous-p12-ruff',
        'cost_id':'e360-generous-p12-ruff','pass_id':'e360-generous-p12','call_role':'ruff','actual_outer_full_return_s':1.423,
        'actual_returned_exit_code':1,'charged_once':True,'internal_completion_valid':True,'capture_complete':True,
        'capture_publication_verified':True,'capture_owned_state':'exit_observed','ruff_exit_code':1,
        'capture_root_tool_exit_required':1,'internal_completion':None,'capture_proof_errors':[]}.items():
        typed_equal(cost[key],item,'actual historical admin schema '+key)
    typed_equal(debit['descriptor'],value['actual_predecessor_failure'],'cost direct pin')
    typed_equal(cost['actual_tool_return'],value['historical_P12_actual_tool_return'],'separate actual pin')
    typed_equal(cost['request'],value['historical_P12_actual_request'],'admitted request pin')
    typed_equal(call['request'],cost['request'],'historical call request')
    typed_equal(actual['request'],cost['request'],'actual same request')
    typed_equal(request['actual_failure'],old['actual_predecessor_failure'],'P12 predecessor worker retains cap55')
    typed_equal(request['concrete_fix'],passed['concrete_fix'],'historic concrete leaves')
    for key in ('pass_id','call_role','mode'):
        typed_equal(cost[key],call[key],'cost/call exact identity')
    for key in ('call_id','pass_id','call_role','mode'):
        typed_equal(cost[key],request[key],'cost/request exact identity')
    for key in ('call_id','pass_id','actual_outer_full_return_s','actual_returned_exit_code'):
        typed_equal(cost[key],actual[key],'separate actual '+key)
    for key in ('actual_outer_full_return_s','actual_returned_exit_code','cost_id','mode'):
        typed_equal(cost[key],debit[key],'exact once debit '+key)
    typed_equal(call['state'],'charged','once charged call');typed_equal(call['reserved_s'],3,'Ruff actual cap3')
    typed_equal(call['sequence_index'],0,'only first role');typed_equal(cost['source_sha256'],own,'cost historic own9')
    typed_equal(request['source_sha256'],own,'request historic own9')
    for item in (request,actual,cost,call):
        typed_equal(item['revalidation_amendment'],value['previous_revalidation_amendment'],'historical direct P12 authority')
        typed_equal(item['resource_extension'],value['resource_extension'],'existing resources')
    returns=strict_json(read_spec(value['historical_P12_ROOT_genuine_actual_returns'],root))
    genuine=strict_json(read_spec(value['historical_P12_genuine_tool_return'],root))
    typed_equal(returns['actual_calls_in_dispatch_order'],[genuine],'one genuine dispatch order')
    typed_equal(returns['actual_call_count'],1,'one genuine actual return')
    typed_equal(returns['actual_total_outer_full_return_s'],1.423,'one genuine elapsed')
    typed_equal(returns['actual_once_debit'],value['actual_predecessor_failure'],'genuine once debit pin')
    typed_equal(returns['genuine_result'],value['historical_P12_genuine_tool_return'],'genuine tool pin')
    for key,item in {'pass_id':'e360-generous-p12','call_id':'e360-generous-p12-ruff','role':'ruff',
        'actual_outer_full_return_s':1.423,'outer_cap_s':3,'genuine_FINAL_return':True}.items():
        typed_equal(genuine[key],item,'genuine FINAL '+key)
    typed_equal(round((genuine['final_return_ms']-genuine['dispatch_ms'])/1000,3),1.423,'genuine dispatch-to-FINAL')
    typed_equal(genuine['result']['exit_code'],1,'genuine external exit1')
    raw_output=read_spec(value['historical_P12_genuine_tool_output'],root)
    typed_equal(raw_output.decode('utf8'),genuine['result']['output'],'raw stored genuine tool output')
    summary=strict_json(read_spec(value['historical_P12_genuine_capture_return'],root))
    typed_equal(strict_json(raw_output),summary,'actual capture output/summary')
    manifest=strict_json(read_spec(value['historical_P12_capture_manifest'],root))
    closure=strict_json(read_spec(value['historical_P12_independent_owned_closure'],root))
    typed_equal(cost['expected_output'],value['historical_P12_capture_manifest'],'admin manifest pin')
    typed_equal(cost['capture_return'],value['historical_P12_genuine_capture_return'],'admin genuine capture pin')
    typed_equal(summary['manifest_descriptor'],value['historical_P12_capture_manifest'],'summary manifest pin')
    typed_equal(manifest['request'],cost['request'],'manifest admitted request')
    typed_equal(manifest['revalidation_amendment'],value['previous_revalidation_amendment'],'manifest historic authority')
    typed_equal(manifest['schema'],'flylab.m3.e360.p12-ruff-capture.v1','historic capture schema')
    for item in (manifest,summary):
        for key,expected in {'call_id':'e360-generous-p12-ruff','capture_complete':True,'exit_observed':True,
            'owned_state':'exit_observed','ruff_exit_code':1,'start_attempted':True,'started_observed':True,
            'start_known_failed':False,'external_cap_s':3}.items():typed_equal(item[key],expected,'historic complete capture '+key)
        typed_equal(item['process_identity'],cost['capture_process_identity'],'same pinned process identity')
    identity=cost['capture_process_identity']
    require(set(identity)=={'process_id','start_utc_ticks'} and all(type(v) is int and v>0 for v in identity.values()),'exact Int64 identity evidence')
    typed_equal(summary['root_tool_exit_required'],1,'actual Ruff tool exit1')
    typed_equal(summary['publication_verified'],True,'actual durable manifest')
    typed_equal(summary['admission_complete'],True,'actual admitted capture')
    typed_equal(summary['first_failure'],manifest['first_failure'],'actual first failure exact')
    first=manifest['first_failure']
    require(type(first) is dict and set(first)=={'slot','kind','observed_s'},'exact historic first failure')
    typed_equal(first['slot'],'ruff_exit_nonzero','actual first failure slot')
    typed_equal(first['kind'],'ruff_exit','actual first failure kind')
    require(type(first['observed_s']) is float and math.isfinite(first['observed_s']) and 0<=first['observed_s']<=3,'historic failure same cap3')
    typed_equal(summary['wrapper_error_slots'],[],'no capture wrapper errors')
    typed_equal(manifest['wrapper_errors'],{},'no manifest wrapper errors')
    typed_equal(summary['streams'],manifest['streams'],'two exact manifest streams')
    typed_equal(closure['streams'],manifest['streams'],'independent closure streams')
    require(len(manifest['streams'])==2,'two EOF streams')
    for stream,name in zip(manifest['streams'],('stdout','stderr'),strict=True):
        typed_equal(stream['stream'],name,'ordered independent stream')
        for key,expected in {'eof_observed':True,'read_fault':False,'overflow':False,'readback_verified':True}.items():
            typed_equal(stream[key],expected,'actual EOF '+key)
        spec=stream['descriptor'];typed_equal(spec,value['historical_P12_'+name],'direct stream descriptor')
        raw=historical_p12_stream_bytes(spec,root)
        typed_equal(stream['observed_bytes'],len(raw),'complete observed bytes')
        typed_equal(stream['retained_bytes'],len(raw),'complete retained bytes')
    typed_equal(closure['owned_identity'],identity,'independent pinned identity')
    typed_equal(closure['manifest'],value['historical_P12_capture_manifest'],'independent manifest')
    typed_equal(closure['genuine_capture_return'],value['historical_P12_genuine_capture_return'],'independent genuine capture')
    typed_equal(closure['charged_failure'],value['actual_predecessor_failure'],'independent charged failure')
    for key,item in {'all_owned_ended':True,'same_owned_identity_live':False,'PID_reused':False,
        'unknown_owned_identities':{},'actual_once_charge_s':1.423,'pending':0,'reservations':0,
        'blocking_holds':0,'unreturned_calls':0,'original_jobs':0,'original_reserved':0}.items():
        typed_equal(closure[key],item,'completed independent closure '+key)
    decision=strict_json(read_spec(value['failed_stage_decision'],root))
    for key,item in {'status':'CLOSED_FAILED','actual_call_count':1,'debits':89,'calls':39,'passes':12,
        'actual_total_outer_full_return_s':1.423,'actual_Ruff_exit_code':1,'actual_Ruff_diagnostic_count':20,
        'actual_tests_executed':0,'actual_phases_executed':0,'P13_authorized':False,'original_execution_authority':False}.items():
        typed_equal(decision[key],item,'immutable closed decision '+key)
    for key,item in {'failed_full_exit':value['actual_predecessor_failure'],'closed_ledger_snapshot':value['historical_P12_closed_ledger89'],
        'approved_literal':value['previous_revalidation_amendment'],'ROOT_actual_runtime_returns':value['historical_P12_ROOT_genuine_actual_returns'],
        'ROOT_independent_owned_closure':value['historical_P12_independent_owned_closure']}.items():typed_equal(decision[key],item,'closed direct proof '+key)
    return frozen


def validate_completed_p12_pass(pass_id,passed,debits,*,root=ROOT):
    frozen=completed_historical_p12_proof(root)
    typed_equal(pass_id,'e360-generous-p12','completed historical P12 only')
    typed_equal(passed,frozen['e360_passes'][pass_id],'whole immutable P12 pass')
    typed_equal(debits[:89],frozen['debits'],'whole immutable89 costs')


def p13_recovery_event(root=ROOT):
    value=p13_recovery_amendment(root)
    return {'schema':'flylab.m3.e360.p13-recovery-activation.v1','revalidation_amendment':file_spec(P13_RECOVERY_AMENDMENT,root),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited89':value['inherited89'],
        'previous_revalidation_amendment':value['previous_revalidation_amendment'],
        'manual_generated_pass_id':value['manual_generated_pass_id'],'new_generated_grant_s':0,
        'human_authorization':value['existing_human_authorization']}


def p13_recovery_activation_allowed(live,*,lock_absent,root=ROOT):
    value=p13_recovery_amendment(root);frozen=completed_historical_p12_proof(root,value=value)
    require(lock_absent is True and not any(key in live for key in P13_RECOVERY_FIELDS),'once current ownership0')
    typed_equal(live,frozen,'activation exact whole89')
    budget=current_budget_authority(root,live=live)
    require(budget['suite_recovery_active'] and budget['generated_shared_limit_s']-live['generated_shared_cumulative_used_s']>=798
        and budget['generated_local_limit_s']-live['e360_generated_local_used_s']>=798,'whole798 existing residual')
    return True


def p13_recovery_delta_valid(before,after,*,root=ROOT):
    require(set(after)==set(before)|set(P13_RECOVERY_FIELDS),'only new P13 pair')
    for key,item in before.items():typed_equal(after[key],item,'whole old89 unchanged '+key)
    typed_equal(after[P13_RECOVERY_FIELDS[0]],p13_recovery_event(root),'exact activation8')
    typed_equal(after[P13_RECOVERY_FIELDS[1]],file_spec(P13_RECOVERY_AMENDMENT,root),'current direct literal')
    return True


@historical_dependencies()
def _p14_recovery_amendment(root=ROOT):
    """Immutable reader15; historical P13 evidence never supplies current admission."""
    raw=literal_bytes(P14_RECOVERY_AMENDMENT,root)
    require(len(raw)==P14_RECOVERY_BYTES and sha(raw)==P14_RECOVERY_SHA,'exact approved P14 literal')
    value=strict_json(raw)
    old=p13_recovery_amendment(root)
    expected={'schema':'flylab.m3.e360.p14-recovery-amendment.v1',
        'manual_generated_pass_id':'e360-generous-p14','predecessor_pass_id':'e360-generous-p13',
        'required_exact_prefix_length':91,'local_cost_offset':42,'generated_shared_limit_s':1400,
        'generated_local_limit_s':1289,'new_generated_grant_s':0,'generated_pass_limit':14,
        'modified_generated_pass_limit':13,'original_pass_limit':1,'total_pass_limit':15,
        'automatic_retries':0,'original_local_limit_s':20,'original_stage_limit_s':300,
        'original_global_limit_s':10200,'original_worker_maximum':1,'original_automatic_retries':0,
        'source_hash_cycle_allowed':False,'whole_fresh_chain_required':True,'primary_total_reserved_s':798,
        'predecessor_failed_cap_s':20,'minimum_current_test_functions':84,'maximum_current_test_functions':84,
        'minimum_current_test_phases':252,'pass14_authorized':True,'pass15_authorized':False,'P13_resume_authorized':False,
        'P11_resume_authorized':False,'P10_resume_authorized':False,'original_execution_authority':False,
        'generated_scope_ends_after_Q_debit':True}
    for key,item in expected.items():typed_equal(value[key],item,'compiled P14 '+key)
    typed_equal(value['previous_revalidation_amendment'],file_spec(P13_RECOVERY_AMENDMENT,root),'P13 immutable reader14')
    typed_equal(value['resource_extension'],file_spec(RESOURCE_EXTENSION,root),'existing resources no grant')
    for key in ('generated_call_order','generated_call_schedule','effective_helper_limits','effective_roles',
        'scanner_limits','checkpoint_policy','successful_suffix_after_s','remaining_generated_administration_reserve_s',
        'fixture_future_administration_reserved_s','suite_future_administration_reserved_s','preserved_test_functions'):
        typed_equal(value[key],old[key],'P13 exact finite inherited schedule/roster '+key)
    scope=copy.deepcopy(old['fixed_authority_scope_policy']);scope['reader_ids'].append('p14_recovery_amendment')
    typed_equal(value['fixed_authority_scope_policy'],scope,'reader15 unchanged16/512/262144/2097152')
    typed_equal(value['preserved_activation_fields'],[*old['preserved_activation_fields'],*P13_RECOVERY_FIELDS],'old22 exact')
    typed_equal(value['preserved_resource_grant_fields'],old['preserved_resource_grant_fields'],'grants2 exact')
    typed_equal(value['new_active_fields'],list(P14_RECOVERY_FIELDS),'pair2 only')
    typed_equal(value['activation_policy'],{'activation_schema':'flylab.m3.e360.p14-recovery-activation.v1',
        'new_grant_object_allowed':False,'new_ledger_fields':list(P14_RECOVERY_FIELDS),'changed_ledger_fields':[],
        'activation_keys':['schema','revalidation_amendment','resource_extension','inherited91',
        'previous_revalidation_amendment','manual_generated_pass_id','new_generated_grant_s','human_authorization']},
        'P14 activation exact8')
    rows=value['historical_concrete_fix_relocations']
    require(type(rows) is list and len(rows)==46,'old42 append P13 frozen R1 four')
    typed_equal(rows[:42],old['historical_concrete_fix_relocations'],'unchanged ordered old42')
    typed_equal([row['pass_id'] for row in rows[42:]],['e360-generous-p13']*4,'P13 frozen4 only')
    for row in rows:
        pinned=row.get('snapshot',row.get('frozen'))
        for key in ('bytes','sha256'):typed_equal(row['original'][key],pinned[key],'exact relocation '+key)
        read_spec(pinned,root)
    for key in P14_RECOVERY_SOURCE_KEYS:read_spec(value[key],root)
    # Integration bases have their own pins; compact Source2 deliberately differs from historic roster.
    require(len(value['reviewed_source_snapshots'])==4,'exact integration bases4')
    for spec in value['reviewed_source_snapshots']:read_spec(spec,root)
    approval=strict_json(read_spec(value['existing_human_authorization'],root))
    for key,item in {'human_quote':'P14승인할게','manual_generated_pass_id':'e360-generous-p14',
        'manual_P14_authorized':True,'new_generated_grant_s':0,'existing_shared_limit_s':1400,
        'existing_local_limit_s':1289,'reserved_whole_generated_schedule_s':798,'automatic_retries':0,
        'P15_authorized':False,'P13_resume_authorized':False,'original_execution_authority':False,
        'baseline_debits':91,'baseline_shared_used_s':581.3300156,'baseline_local_used_s':470.373}.items():
        typed_equal(approval[key],item,'direct P14 approval '+key)
    typed_equal(value['prospective_native_source_paths'],['verification/banc-natural-gait-m3-20260919/parent-e360-p14-prepare-native.ps1','verification/banc-natural-gait-m3-20260919/parent-e360-p14-bookkeeping-native.ps1','verification/banc-natural-gait-m3-20260919/parent-e360-p14-ROOT-dispatch-adapter.ps1'],'exact prospective Native3 paths; future hashes only freeze/request')
    typed_equal(value['generated_scope_ends_after_first_closed_failure'],True,'first closed failure stops')
    completed_historical_p13_proof(root,value=value)
    return value


def p14_recovery_amendment(root=ROOT):
    return fixed_reader('p14_recovery_amendment',_p14_recovery_amendment,root)


def historical_p13_scope_authority(root=ROOT,*,live=None):
    """Uncached P13 selector; exact inherited89 and pair2."""
    if live is None:live=strict_json(bounded_metadata_bytes(registration(root)['shared_ledger_path'],root))
    present=[key in live for key in P13_RECOVERY_FIELDS]
    require(all(present) or not any(present),'complete current P13 pair')
    if not all(present):return False
    value=p13_recovery_amendment(root);frozen=strict_json(read_spec(value['inherited89'],root))['ledger']
    require(current_budget_authority(root,live=live)['suite_recovery_active'],'preserved P9 budget')
    require(set(frozen)|set(P13_RECOVERY_FIELDS)<=set(live),'exact baseline fields plus pair2')
    typed_equal(live[P13_RECOVERY_FIELDS[0]],p13_recovery_event(root),'exact current event8')
    typed_equal(live[P13_RECOVERY_FIELDS[1]],file_spec(P13_RECOVERY_AMENDMENT,root),'exact current descriptor')
    typed_equal(live['debits'][:89],frozen['debits'],'full89 exact ordered prefix')
    for key in (*value['preserved_activation_fields'],*value['preserved_resource_grant_fields']):
        typed_equal(live[key],frozen[key],'preserved old20/grants2 '+key)
    for key in ('e360_calls','e360_passes'):
        for identity,item in frozen[key].items():typed_equal(live[key][identity],item,'whole inherited89 '+key)
    return True


@historical_dependencies()
def completed_historical_p13_proof(root=ROOT,*,value=None):
    """Completed dual ADMIN proof; never enters current clocks, ownership or admission."""
    if value is None:
        raw=literal_bytes(P14_RECOVERY_AMENDMENT,root)
        require(len(raw)==P14_RECOVERY_BYTES and sha(raw)==P14_RECOVERY_SHA,'direct immutable P14 history pin')
        value=strict_json(raw)
    old=p13_recovery_amendment(root)
    prior=completed_historical_p12_proof(root,value=old)
    frozen=strict_json(read_spec(value['inherited91'],root))['ledger']
    typed_equal(frozen,strict_json(read_spec(value['historical_P13_closed_ledger91'],root)),'whole completed91')
    typed_equal(prior,strict_json(read_spec(value['historical_P13_inherited89'],root))['ledger'],'whole prior89')
    require(len(frozen['debits'])==91 and len(frozen['e360_calls'])==41 and len(frozen['e360_passes'])==13,'completed91/41/13')
    typed_equal(frozen['debits'][:89],prior['debits'],'all89 ordered costs retained')
    for key in (*old['preserved_activation_fields'],*old['preserved_resource_grant_fields']):
        typed_equal(frozen[key],prior[key],'whole historical prior field '+key)
    typed_equal(frozen[P13_RECOVERY_FIELDS[0]],p13_recovery_event(root),'completed P13 event8/grant0')
    typed_equal(frozen[P13_RECOVERY_FIELDS[1]],value['previous_revalidation_amendment'],'completed P13 literal')
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(frozen[key][identity],item,'whole historical89 '+key)
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):
        typed_equal(frozen[key],{},'completed ownership0 '+key)
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(frozen[key],0,'completed original0')
    own=value['historical_P13_frozen_own_sources_sha256'];passed=frozen['e360_passes']['e360-generous-p13']
    require(type(own) is dict and len(own)==9,'historical exact own9')
    for key,item in {'mode':'generated','modified':True,'predecessor_failed_cap_s':3,'failed_cap_s':20,'next_index':2,
        'actual_failure':old['actual_predecessor_failure'],'failed_cost':value['actual_predecessor_failure'],
        'source_sha256':own,'revalidation_amendment':value['previous_revalidation_amendment']}.items():
        typed_equal(passed[key],item,'completed P13 pass '+key)
    freeze=strict_json(read_spec(value['historical_P13_final_source_freeze'],root))
    typed_equal(freeze['own_sources_sha256'],own,'actual R1 executed own9')
    typed_equal(value['historical_concrete_fix_relocations'][:42],old['historical_concrete_fix_relocations'],'old42 unchanged')
    rows=value['historical_concrete_fix_relocations'][42:46]
    require(len(rows)==4 and [row['pass_id'] for row in rows]==['e360-generous-p13']*4,'actual P13 R1 frozen4')
    relocated={row['original']['path']:row for row in rows};require(len(relocated)==4,'distinct frozen4')
    for row,source in zip(rows,freeze['source_copies'],strict=True):
        typed_equal(row['original'],source['current'],'actual executed Source4')
        typed_equal(row['frozen'],source['frozen'],'actual frozen R1 Source4')
        typed_equal(row['original']['sha256'],own[row['original']['path']],'executed own Source4 pin')
        for key in ('bytes','sha256'):typed_equal(row['original'][key],row['frozen'][key],'same-byte frozen relocation')
        read_spec(row['frozen'],root)
    for name,digest in own.items():
        if name not in relocated:
            read_spec({'path':name,'bytes':safe(name,root).stat().st_size,'sha256':digest},root)
    for spec in passed['concrete_fix']:
        if spec['path'] in relocated:
            typed_equal(spec,relocated[spec['path']]['original'],'completed concrete Source4 descriptor')
            read_spec(relocated[spec['path']]['frozen'],root)
        else:read_spec(spec,root)
    for spec in freeze['all_P13_Native_sources']:read_spec(spec,root)
    typed_equal(value['historical_P13_actual_dispatch_order'],['ruff','scanner'],'actual role order2')
    costs=value['historical_P13_actual_paid_costs']
    typed_equal(frozen['debits'][89:91],costs,'actual index89/90 once rows')
    require(type(costs) is list and len(costs)==2,'dual once debits only')
    common={'schema','mode','charged_once','cost_id','actual_outer_full_return_s','actual_tool_return',
        'actual_returned_exit_code','original_array_reads','GPU_contexts','resource_extension','revalidation_amendment',
        'call_id','pass_id','call_role','request','source_sha256','expected_output','internal_completion','internal_completion_valid'}
    capture={'capture_return','capture_proof_errors','ruff_exit_code','capture_complete','capture_publication_verified',
        'capture_owned_state','capture_process_identity','capture_root_tool_exit_required','capture_first_failure'}
    call_keys={'attempt_id','call_role','cost_id','future_reserved_s','mode','pass_id','request','reserved_s',
        'resource_extension','revalidation_amendment','sequence_index','state'}
    returns=strict_json(read_spec(value['historical_P13_ROOT_genuine_actual_returns'],root))
    genuine_rows=[];paid=[]
    for index,(role,label,elapsed,cap,suffix) in enumerate((('ruff','ruff',1.838,3,795),('scanner','scan',38.792,20,775))):
        prefix='historical_P13_'+role+'_';call_id='e360-generous-p13-'+label
        spec=value[prefix+'full_return'];cost=strict_json(read_spec(spec,root));debit=costs[index]
        actual=strict_json(read_spec(value[prefix+'actual_tool_return'],root))
        request=strict_json(read_spec(value[prefix+'actual_request'],root))
        require(type(frozen['e360_calls']) is dict and call_id in frozen['e360_calls'],'exact historical dictionary call key')
        call=frozen['e360_calls'][call_id]
        require(type(call) is dict and set(call)==call_keys,'actual12 row; no inner call_id')
        require(type(cost) is dict and set(cost)==common|(capture if role=='ruff' else set()),'same ADMIN tag variants28/19')
        for key,item in {'schema':'flylab.m3.e360.administrative-full-return.v1','mode':'generated','charged_once':True,
            'call_id':call_id,'cost_id':call_id,'pass_id':'e360-generous-p13','call_role':role,
            'actual_outer_full_return_s':elapsed,'actual_returned_exit_code':0,'original_array_reads':0,'GPU_contexts':0,
            'internal_completion':None,'internal_completion_valid':True,'source_sha256':own}.items():
            typed_equal(cost[key],item,'actual dual ADMIN '+key)
        typed_equal(debit['descriptor'],spec,'once direct ADMIN descriptor')
        for key in ('mode','cost_id','actual_outer_full_return_s','actual_returned_exit_code'):
            typed_equal(debit[key],cost[key],'actual once debit '+key)
        typed_equal(call['cost_id'],call_id,'dictionary key/cost_id')
        typed_equal(call['state'],'charged','completed charged call')
        typed_equal(call['attempt_id'],None,'ADMIN no worker attempt')
        typed_equal(call['sequence_index'],index,'genuine dispatch sequence0/1')
        typed_equal(call['reserved_s'],cap,'immutable historical cap3/20')
        typed_equal(call['future_reserved_s'],suffix,'immutable successful suffix795/775')
        typed_equal(cost['actual_tool_return'],value[prefix+'actual_tool_return'],'separate actual descriptor')
        typed_equal(cost['request'],value[prefix+'actual_request'],'admitted exact request descriptor')
        for item in (call,actual):typed_equal(item['request'],cost['request'],'same immutable admitted request')
        for item in (cost,request,actual):typed_equal(item['call_id'],call_id,'dictionary key/cost/request/actual binding')
        for key in ('pass_id','call_role','mode'):typed_equal(call[key],cost[key],'exact call role identity')
        for key in ('pass_id','call_role','mode','source_sha256'):typed_equal(request[key],cost[key],'request exact identity')
        for key in ('pass_id','actual_outer_full_return_s','actual_returned_exit_code'):typed_equal(actual[key],cost[key],'genuine actual '+key)
        for item in (call,cost,request,actual):
            typed_equal(item['revalidation_amendment'],value['previous_revalidation_amendment'],'historical P13 direct authority')
            typed_equal(item['resource_extension'],value['resource_extension'],'existing resources')
        genuine=strict_json(read_spec(value[prefix+'genuine_tool_return'],root));genuine_rows.append(genuine);paid.append(spec)
        for key,item in {'pass_id':'e360-generous-p13','call_id':call_id,'role':role,'outer_cap_s':cap,
            'actual_outer_full_return_s':elapsed,'genuine_FINAL_return':True}.items():typed_equal(genuine[key],item,'actual genuine FINAL '+key)
        require(type(genuine['dispatch_ms']) is int and type(genuine['final_return_ms']) is int,'genuine integer clocks')
        typed_equal((genuine['final_return_ms']-genuine['dispatch_ms'])/1000,elapsed,'full dispatch-to-FINAL with no reconstruction')
        typed_equal(genuine['result']['exit_code'],0,'actual genuine exit0')
        require(genuine['result'].get('session_id') is None and type(genuine['result']['output']) is str,'genuine FINAL raw output')
        if role=='scanner':
            typed_equal(cost['expected_output'],value[prefix+'report'],'scanner exact SourceJSON report')
            typed_equal(strict_json(genuine['result']['output'].encode('utf8')),value[prefix+'report'],'scanner genuine output descriptor')
            chunks=genuine['actual_tool_chunks']
            require(len(chunks)==2 and type(chunks[0]['session_id']) is int and chunks[0]['session_id']>0,'actual scanner pending plus FINAL')
            require(chunks[0].get('exit_code') is None,'scanner pending no exit')
            typed_equal(chunks[1],genuine['result'],'actual last scanner chunk is FINAL')
            require(value[prefix+'report']['bytes']<=2097152,'existing SourceJSON bound2MiB')
            report=strict_json(read_spec(value[prefix+'report'],root))
            typed_equal(report['diagnostic_ready'],True,'diagnostic report only')
            typed_equal(report['future_ready'],False,'late historical scanner never qualification')
            require('internal_completion_valid' not in report,'internal completion belongs to ADMIN cost only')
        else:
            typed_equal(request['actual_failure'],old['actual_predecessor_failure'],'historical predecessor cap3')
            typed_equal(request['concrete_fix'],passed['concrete_fix'],'executed concrete leaves')
            summary=strict_json(read_spec(value[prefix+'genuine_capture_return'],root))
            manifest=strict_json(read_spec(value[prefix+'capture_manifest'],root))
            typed_equal(strict_json(genuine['result']['output'].encode('utf8')),summary,'opaque genuine capture output')
            typed_equal(cost['expected_output'],value[prefix+'capture_manifest'],'Ruff manifest only')
            typed_equal(cost['capture_return'],value[prefix+'genuine_capture_return'],'Ruff capture only')
            typed_equal(manifest['schema'],'flylab.m3.e360.p13-ruff-capture.v1','actual P13 capture schema')
            typed_equal(manifest['request'],cost['request'],'manifest actual request')
            typed_equal(manifest['revalidation_amendment'],value['previous_revalidation_amendment'],'manifest historic literal')
            for item in (summary,manifest):
                for key,expected in {'call_id':call_id,'capture_complete':True,'exit_observed':True,'owned_state':'exit_observed',
                    'ruff_exit_code':0,'start_attempted':True,'started_observed':True,'start_known_failed':False,'external_cap_s':3,
                    'first_failure':None}.items():typed_equal(item[key],expected,'actual Ruff capture '+key)
                typed_equal(item['process_identity'],cost['capture_process_identity'],'Ruff exact owned identity')
            identity=cost['capture_process_identity']
            require(set(identity)=={'process_id','start_utc_ticks'} and all(type(v) is int and v>0 for v in identity.values()),'exact Native Int64 identity')
            typed_equal(identity['start_utc_ticks'],639267292820475860,'actual unrounded process ticks')
            for key,item in {'capture_complete':True,'capture_publication_verified':True,'capture_owned_state':'exit_observed',
                'ruff_exit_code':0,'capture_root_tool_exit_required':0,'capture_first_failure':None,'capture_proof_errors':[]}.items():typed_equal(cost[key],item,'Ruff-only ADMIN proof '+key)
            typed_equal(summary['manifest_descriptor'],value[prefix+'capture_manifest'],'summary exact manifest')
            typed_equal(summary['publication_verified'],True,'durable historical capture')
            typed_equal(summary['admission_complete'],True,'genuine admitted capture')
            typed_equal(summary['root_tool_exit_required'],0,'genuine tool exit0')
            typed_equal(summary['wrapper_error_slots'],[],'capture no wrapper errors')
            typed_equal(manifest['wrapper_errors'],{},'manifest no wrapper errors')
            typed_equal(summary['streams'],manifest['streams'],'exact dual EOF streams')
            require(len(manifest['streams'])==2,'two ordered capture streams')
            for stream,name in zip(manifest['streams'],('stdout','stderr'),strict=True):
                typed_equal(stream['stream'],name,'ordered stream name')
                for key,item in {'eof_observed':True,'read_fault':False,'overflow':False,'readback_verified':True}.items():typed_equal(stream[key],item,'genuine stream '+key)
                typed_equal(stream['descriptor'],value[prefix+name],'direct historic stream pin')
                raw=historical_p12_stream_bytes(stream['descriptor'],root)
                for key in ('observed_bytes','retained_bytes'):typed_equal(stream[key],len(raw),'complete stream bytes')
    typed_equal(returns['actual_calls_in_dispatch_order'],genuine_rows,'actual genuine dispatch order2')
    typed_equal(returns['actual_once_debits'],paid,'plural actual once debits')
    typed_equal(returns['actual_call_count'],2,'two genuine calls')
    typed_equal(returns['actual_total_outer_full_return_s'],40.630,'all40.630 paid once')
    require(genuine_rows[0]['final_return_ms']<genuine_rows[1]['dispatch_ms'],'serial actual dispatch')
    closure=strict_json(read_spec(value['historical_P13_independent_owned_closure'],root))
    for key,item in {'owned_identity':identity,'streams':manifest['streams'],'manifest':value['historical_P13_ruff_capture_manifest'],
        'genuine_capture_return':value['historical_P13_ruff_genuine_capture_return'],'all_owned_ended':True,
        'same_owned_identity_live':False,'PID_reused':False,'unknown_owned_identities':{},'actual_once_charge_s':40.630,
        'scanner_genuine_completed_FINAL_return':True,'scanner_owned_worker_or_original_children_started':0,
        'pending':0,'reservations':0,'blocking_holds':0,'unreturned_calls':0,'original_jobs':0,'original_reserved':0,
        'closed_ledger_snapshot':value['historical_P13_closed_ledger91'],'executed_own9_preserved':True}.items():typed_equal(closure[key],item,'actual dual closure '+key)
    require('charged_failure' not in closure,'no absent P12-only closure field')
    decision=strict_json(read_spec(value['failed_stage_decision'],root))
    for key,item in {'status':'CLOSED_FAILED','first_failed_role':'scanner','actual_call_count':2,'debits':91,'calls':41,'passes':13,
        'actual_total_outer_full_return_s':40.630,'actual_Ruff_exit_code':0,'actual_Ruff_full_return_s':1.838,
        'actual_scanner_exit_code':0,'actual_scanner_full_return_s':38.792,'scanner_cap_s':20,'actual_tests_executed':0,
        'actual_phases_executed':0,'P14_authorized':False,'original_execution_authority':False,
        'failed_full_exit':value['actual_predecessor_failure'],'closed_ledger_snapshot':value['historical_P13_closed_ledger91'],
        'approved_literal':value['previous_revalidation_amendment'],'ROOT_actual_runtime_returns':value['historical_P13_ROOT_genuine_actual_returns'],
        'ROOT_independent_owned_closure':value['historical_P13_independent_owned_closure']}.items():typed_equal(decision[key],item,'immutable failed history '+key)
    return frozen


def validate_completed_p13_pass(pass_id,passed,debits,*,root=ROOT):
    frozen=completed_historical_p13_proof(root)
    typed_equal(pass_id,'e360-generous-p13','completed P13 history only')
    typed_equal(passed,frozen['e360_passes'][pass_id],'whole actual P13 historical pass')
    typed_equal(debits[:91],frozen['debits'],'whole91 exact once costs')


def p14_recovery_event(root=ROOT):
    value=p14_recovery_amendment(root)
    return {'schema':'flylab.m3.e360.p14-recovery-activation.v1','revalidation_amendment':file_spec(P14_RECOVERY_AMENDMENT,root),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited91':value['inherited91'],
        'previous_revalidation_amendment':value['previous_revalidation_amendment'],
        'manual_generated_pass_id':value['manual_generated_pass_id'],'new_generated_grant_s':0,
        'human_authorization':value['existing_human_authorization']}


def p14_recovery_activation_allowed(live,*,lock_absent,root=ROOT):
    value=p14_recovery_amendment(root);frozen=completed_historical_p13_proof(root,value=value)
    require(lock_absent is True and not any(key in live for key in P14_RECOVERY_FIELDS),'once current ownership0')
    typed_equal(live,frozen,'activation exact whole91')
    budget=current_budget_authority(root,live=live)
    require(budget['suite_recovery_active'] and budget['generated_shared_limit_s']-live['generated_shared_cumulative_used_s']>=798
        and budget['generated_local_limit_s']-live['e360_generated_local_used_s']>=798,'whole798 existing residual')
    return True


def p14_recovery_delta_valid(before,after,*,root=ROOT):
    require(set(after)==set(before)|set(P14_RECOVERY_FIELDS),'only new P14 pair')
    for key,item in before.items():typed_equal(after[key],item,'whole old91 unchanged '+key)
    typed_equal(after[P14_RECOVERY_FIELDS[0]],p14_recovery_event(root),'exact activation8')
    typed_equal(after[P14_RECOVERY_FIELDS[1]],file_spec(P14_RECOVERY_AMENDMENT,root),'current direct literal')
    return True


def historical_p14_scope_authority(root=ROOT,*,live=None):
    """Uncached P14 selector; exact inherited91 and pair2."""
    if live is None:live=strict_json(bounded_metadata_bytes(registration(root)['shared_ledger_path'],root))
    present=[key in live for key in P14_RECOVERY_FIELDS]
    require(all(present) or not any(present),'complete current P14 pair')
    if not all(present):return False
    value=p14_recovery_amendment(root);frozen=strict_json(read_spec(value['inherited91'],root))['ledger']
    require(current_budget_authority(root,live=live)['suite_recovery_active'],'preserved P9 budget')
    require(set(frozen)|set(P14_RECOVERY_FIELDS)<=set(live),'exact baseline fields plus pair2')
    typed_equal(live[P14_RECOVERY_FIELDS[0]],p14_recovery_event(root),'exact current event8')
    typed_equal(live[P14_RECOVERY_FIELDS[1]],file_spec(P14_RECOVERY_AMENDMENT,root),'exact current descriptor')
    typed_equal(live['debits'][:91],frozen['debits'],'full91 exact ordered prefix')
    for key in (*value['preserved_activation_fields'],*value['preserved_resource_grant_fields']):
        typed_equal(live[key],frozen[key],'preserved old22/grants2 '+key)
    for key in ('e360_calls','e360_passes'):
        for identity,item in frozen[key].items():typed_equal(live[key][identity],item,'whole inherited91 '+key)
    return True


def validate_p13_pass(pass_id,passed,debits,*,root=ROOT):
    raise ValueError('P13 completed history cannot supply prospective admission')
    value=p13_recovery_amendment(root)
    typed_equal(pass_id,'e360-generous-p13','one manual P13 no future')
    require(current_scope_authority(root),'P13 actual activation')
    frozen=completed_historical_p12_proof(root,value=value)
    failure=strict_json(read_spec(passed['actual_failure'],root))
    for key,item in {'schema':'flylab.m3.e360.administrative-full-return.v1','actual_returned_exit_code':1,'internal_completion_valid':True,'actual_outer_full_return_s':1.423,'charged_once':True}.items():typed_equal(failure[key],item,'P13 failed admin predecessor '+key)
    require(passed['modified'] is True and passed['mode']=='generated','modified generated P13')
    typed_equal(passed['actual_failure'],value['actual_predecessor_failure'],'actual completed P12 Ruff')
    typed_equal(passed['predecessor_failed_cap_s'],3,'completed Ruff predecessor cap3')
    require(sum(row['descriptor']==passed['actual_failure'] for row in debits)==1,'once failed Ruff debit')
    require(passed['source_sha256']!=frozen['e360_passes']['e360-generous-p12']['source_sha256'],'fresh P13 changed own9')
    typed_equal(passed['source_sha256'],own_sources(root),'current exact own9')
    require(sum(spec==value['concrete_correction_snapshot'] for spec in passed['concrete_fix'])==1,'reviewed compact correction')
    for name in (*OWNED,BASE+'/parent-e360-root-account.ps1',BASE+'/parent-e360-root-metadata-helper.py',
        value['ruff_capture_policy']['function_only_source_path'],*value['prospective_native_source_paths']):
        require(sum(spec==file_spec(name,root) for spec in passed['concrete_fix'])==1,'current concrete leaf '+name)
    for spec in passed['concrete_fix']:read_spec(spec,root)

def validate_p14_pass(pass_id,passed,debits,*,root=ROOT):
    raise ValueError('P14 completed history cannot supply prospective admission')

@historical_dependencies()
def _p15_recovery_amendment(root=ROOT):
    raw=literal_bytes(P15_RECOVERY_AMENDMENT,root)
    require(len(raw)==P15_RECOVERY_BYTES and sha(raw)==P15_RECOVERY_SHA,'exact approved P15 literal')
    value=strict_json(raw);old=p14_recovery_amendment(root)
    expected={'schema':'flylab.m3.e360.p15-recovery-amendment.v2','manual_generated_pass_id':'e360-generous-p15',
        'predecessor_pass_id':'e360-generous-p14','required_exact_prefix_length':92,'local_cost_offset':42,
        'generated_shared_limit_s':1400,'generated_local_limit_s':1289,'new_generated_grant_s':0,
        'generated_pass_limit':15,'modified_generated_pass_limit':14,'original_pass_limit':1,'total_pass_limit':16,
        'automatic_retries':0,'original_local_limit_s':20,'original_stage_limit_s':300,'original_global_limit_s':10200,
        'original_worker_maximum':1,'original_automatic_retries':0,'source_hash_cycle_allowed':False,
        'whole_fresh_chain_required':True,'primary_total_reserved_s':798,'predecessor_failed_cap_s':0,
        'minimum_current_test_functions':84,'maximum_current_test_functions':84,'minimum_current_test_phases':252,
        'pass15_authorized':True,'pass16_authorized':False,'P14_resume_authorized':False,
        'original_execution_authority':False,'generated_scope_ends_after_Q_debit':True,
        'generated_scope_ends_after_first_closed_failure':True,
        'predecessor_failure_kind':'closed_admission_failure_before_scanner_dispatch'}
    for key,item in expected.items():typed_equal(value[key],item,'compiled P15 '+key)
    typed_equal(value['previous_revalidation_amendment'],file_spec(P14_RECOVERY_AMENDMENT,root),'immutable reader15')
    for key in ('resource_extension','generated_call_order','generated_call_schedule','effective_helper_limits',
        'effective_roles','scanner_limits','checkpoint_policy','successful_suffix_after_s',
        'remaining_generated_administration_reserve_s','fixture_future_administration_reserved_s',
        'suite_future_administration_reserved_s','preserved_test_functions','preserved_resource_grant_fields'):
        typed_equal(value[key],old[key],'preserved finite P14 policy '+key)
    scope=copy.deepcopy(old['fixed_authority_scope_policy']);scope['reader_ids'].append('p15_recovery_amendment')
    typed_equal(value['fixed_authority_scope_policy'],scope,'reader16 exact old15 and unchanged bounds')
    typed_equal(value['preserved_activation_fields'],[*old['preserved_activation_fields'],*P14_RECOVERY_FIELDS],'old24')
    typed_equal(value['new_active_fields'],list(P15_RECOVERY_FIELDS),'newpair2')
    typed_equal(value['activation_policy'],{'activation_schema':'flylab.m3.e360.p15-recovery-activation.v1',
        'changed_ledger_fields':[],'new_ledger_fields':list(P15_RECOVERY_FIELDS),'new_grant_object_allowed':False,
        'activation_keys':['schema','revalidation_amendment','resource_extension','inherited92',
        'previous_revalidation_amendment','manual_generated_pass_id','new_generated_grant_s','human_authorization']},'event8')
    rows=value['historical_concrete_fix_relocations']
    require(type(rows) is list and len(rows)==50,'exact relocation50')
    typed_equal(rows[:46],old['historical_concrete_fix_relocations'],'unchanged old46')
    typed_equal([r['pass_id'] for r in rows[46:]],['e360-generous-p14']*4,'actual P14 frozen4')
    for row in rows:
        leaf=row.get('snapshot',row.get('frozen'))
        for key in ('bytes','sha256'):typed_equal(row['original'][key],leaf[key],'same-byte historical Source '+key)
        read_spec(leaf,root)
    # The exact literal pins all190 fields. Only actual descriptor objects are read here;
    # no recursive JSON reference reader, prospective nonexistent leaves or array reads.
    for key,spec in value.items():
        if type(spec) is dict and set(spec)=={'path','bytes','sha256'}:
            historical_p12_stream_bytes(spec,root) if spec['bytes']==0 else read_spec(spec,root)
    for spec in value['reviewed_source_snapshots']:read_spec(spec,root)
    for spec in value['current_correction_source_snapshots']:read_spec(spec,root)
    approval=strict_json(read_spec(value['existing_human_authorization'],root))
    for key,item in {'human_quote':'수정안 승인할게. 진행해줘','manual_generated_pass_id':'e360-generous-p15','manual_P15_authorized':True,
        'new_generated_grant_s':0,'existing_shared_limit_s':1400,'existing_local_limit_s':1289,
        'reserved_whole_generated_schedule_s':798,'automatic_retries':0,'P16_authorized':False,
        'P14_resume_authorized':False,'original_execution_authority':False,'baseline_debits':92,
        'baseline_calls':42,'baseline_passes':14,'baseline_shared_used_s':583.1700156,
        'baseline_local_used_s':472.213}.items():typed_equal(approval[key],item,'direct P15 approval '+key)
    typed_equal(value['prospective_native_source_paths'],[BASE+'/parent-e360-p15v2-'+name+'.ps1'
        for name in ('prepare-native','bookkeeping-native','ROOT-dispatch-adapter')],'Native3 canonical paths')
    completed_historical_p14_proof(root,value=value)
    return value


def p15_recovery_amendment(root=ROOT):
    return fixed_reader('p15_recovery_amendment',_p15_recovery_amendment,root)


def validate_saved_p14_termination(closure,decision,old,historical_approval,*,value):
    """Actual production typed gates shared with targeted semantic negatives."""
    # Independent saved P14 closure uses known_owned_ruff, not worker all_owned_ended.
    typed_equal(closure['schema'],'flylab.m3.e360.p14.ROOT-independent-owned-closure.v1','actual closure schema')
    typed_equal(closure['actual_actor'],'ROOT','actual closure actor')
    typed_equal(closure['frozen_ledger92'],value['historical_P14_closed_ledger92'],'actual closure whole92')
    typed_equal(closure['whole92_calls42_passes14'],True,'actual closure 92/42/14')
    typed_equal(closure['all_old91_once_debits_preserved'],True,'actual closure old91')
    typed_equal(closure['new_paid_calls'],1,'actual P14 one paid Ruff')
    typed_equal(closure['new_paid_seconds'],1.840,'actual P14 paid Ruff wall')
    for key in ('pending_actuals','reservations','blocking_holds','unreturned_calls','original_jobs','original_reserved_s'):
        typed_equal(closure[key],0,'actual closed ownership0 '+key)
    typed_equal(closure['known_owned_ruff'],{'process_id':33356,'expected_start_utc_ticks':639267587608157543,
        'observed_start_utc_ticks':None,'matching_owned_live':False,'exit_observed':True,
        'both_streams_closed':True,'no_process_termination_performed':True},'actual known-owned Ruff closure7')
    for key,item in {'schema':'flylab.m3.e360.p14.ROOT-closed-decision.v1','status':'CLOSED_ADMISSION_FAILED',
        'closed_ledger92':value['historical_P14_closed_ledger92'],'owned_closure':value['historical_P14_independent_owned_closure'],
        'active_ledger_pass_failed_cost_remains_null':True,'no_fabricated_failed_paid_cost':True,
        'new_paid_seconds':1.840,'original_IO_GPU_body_neural_UI':False,'actual_test_phases':0}.items():typed_equal(decision[key],item,'nonpaid predecessor '+key)
    # Saved decision's I/O flag and original execution authority are separate facts.
    typed_equal(old['original_execution_authority'],False,'immutable P14 literal original authority')
    typed_equal(historical_approval['original_execution_authority'],False,'immutable P14 human original authority')


@historical_dependencies()
def completed_historical_p14_proof(root=ROOT,*,value=None):
    """Immutable successful Ruff and nonpaid rejection; no prospective selector or current Source4."""
    if value is None:
        raw=literal_bytes(P15_RECOVERY_AMENDMENT,root)
        require(len(raw)==P15_RECOVERY_BYTES and sha(raw)==P15_RECOVERY_SHA,'immutable P15 history literal')
        value=strict_json(raw)
    old=p14_recovery_amendment(root);prior=completed_historical_p13_proof(root,value=old)
    frozen=strict_json(read_spec(value['inherited92'],root))['ledger']
    typed_equal(frozen,strict_json(read_spec(value['historical_P14_closed_ledger92'],root)),'exact whole92')
    typed_equal(prior,strict_json(read_spec(value['historical_P14_inherited91'],root))['ledger'],'exact wholeold91')
    require(len(frozen['debits'])==92 and len(frozen['e360_calls'])==42 and len(frozen['e360_passes'])==14,'92/42/14')
    typed_equal(frozen['debits'][:91],prior['debits'],'all91 ordered costs')
    for key in (*old['preserved_activation_fields'],*old['preserved_resource_grant_fields']):
        typed_equal(frozen[key],prior[key],'unchanged old22/grants2 '+key)
    typed_equal(frozen[P14_RECOVERY_FIELDS[0]],p14_recovery_event(root),'actual historical event8')
    typed_equal(frozen[P14_RECOVERY_FIELDS[1]],value['previous_revalidation_amendment'],'actual P14 literal')
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(frozen[key][identity],item,'whole91 historical '+key)
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):
        typed_equal(frozen[key],{},'completed ownership0 '+key)
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(frozen[key],0,'original0')
    own=value['historical_P14_frozen_own_sources_sha256'];passed=frozen['e360_passes']['e360-generous-p14']
    for key,item in {'mode':'generated','modified':True,'predecessor_failed_cap_s':20,'failed_cap_s':0,'next_index':1,
        'actual_failure':old['actual_predecessor_failure'],'failed_cost':None,'source_sha256':own,
        'revalidation_amendment':value['previous_revalidation_amendment']}.items():typed_equal(passed[key],item,'actual P14 pass '+key)
    require(type(own) is dict and len(own)==9,'historical own9')
    rows=value['historical_concrete_fix_relocations'][46:50];relocated={r['original']['path']:r for r in rows}
    require(len(relocated)==4,'actual frozen Source4')
    freeze=strict_json(read_spec(value['historical_P14_final_source_freeze'],root))
    typed_equal(freeze['own_sources_sha256'],own,'actual frozen P14 own9')
    for row,source in zip(rows,freeze['source_copies'],strict=True):
        typed_equal(row['original'],source['current'],'actual P14 current descriptor')
        typed_equal(row['frozen'],source['frozen'],'actual P14 frozen descriptor')
        typed_equal(row['original']['sha256'],own[row['original']['path']],'P14 historical Source4 pin')
        read_spec(row['frozen'],root)
    for name,digest in own.items():
        if name not in relocated:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'unchanged historical own5')
    for spec in passed['concrete_fix']:
        if spec['path'] in relocated:
            typed_equal(spec,relocated[spec['path']]['original'],'historical P14 concrete Source4')
            read_spec(relocated[spec['path']]['frozen'],root)
        else:read_spec(spec,root)
    for spec in value['historical_P14_Native4']:read_spec(spec,root)
    read_spec(value['historical_P14_collector_module'],root)
    costs=value['historical_P14_actual_paid_costs']
    require(type(costs) is list and len(costs)==1,'exact single paid P14 Ruff')
    typed_equal(frozen['debits'][91:92],costs,'actual index91 once debit')
    debit=costs[0];cost=strict_json(read_spec(value['historical_P14_ruff_full_return'],root))
    actual=strict_json(read_spec(value['historical_P14_ruff_actual_tool_return'],root))
    request=strict_json(read_spec(value['historical_P14_ruff_actual_request'],root));call_id='e360-generous-p14-ruff'
    call=frozen['e360_calls'][call_id]
    require(len(cost)==30 and len(actual)==12 and set(call)=={'attempt_id','call_role','cost_id','future_reserved_s',
        'mode','pass_id','request','reserved_s','resource_extension','revalidation_amendment','sequence_index','state'},'actual ADMIN30/actual12/call12')
    paid_request_identity(debit,cost,actual,frozen,root=root)
    for key,item in {'schema':'flylab.m3.e360.administrative-full-return.v1','mode':'generated','charged_once':True,
        'cost_id':call_id,'call_id':call_id,'pass_id':'e360-generous-p14','call_role':'ruff',
        'actual_outer_full_return_s':1.840,'actual_returned_exit_code':0,'internal_completion_valid':True,
        'collector_proof_valid':True}.items():typed_equal(cost[key],item,'actual successful Ruff '+key)
    typed_equal(debit['descriptor'],value['historical_P14_ruff_full_return'],'direct paid cost pin')
    typed_equal(cost['actual_tool_return'],value['historical_P14_ruff_actual_tool_return'],'actual return pin')
    typed_equal(cost['request'],value['historical_P14_ruff_actual_request'],'actual request pin')
    typed_equal(cost['source_sha256'],own,'cost historical own9');typed_equal(request['source_sha256'],own,'request historical own9')
    typed_equal(call['state'],'charged','once charged');typed_equal(call['reserved_s'],3,'actual cap3')
    typed_equal(call['future_reserved_s'],795,'actual suffix795');typed_equal(call['sequence_index'],0,'only Ruff')
    for item in (cost,actual,call,request):
        typed_equal(item['revalidation_amendment'],value['previous_revalidation_amendment'],'historic direct authority')
        typed_equal(item['resource_extension'],value['resource_extension'],'historic resources')
    collector=p14_collector_proof(value['historical_P14_ruff_genuine_collector_trace'],request,
        value['historical_P14_ruff_actual_request'],1.840,0,root=root)
    typed_equal(collector['valid'],True,'actual historical collector valid')
    trace=strict_json(read_spec(value['historical_P14_ruff_genuine_collector_trace'],root))
    typed_equal(trace['session_id'],None,'actual immediate FINAL no session')
    require(len(trace['chunks'])==1,'actual immediate FINAL chunk1')
    summary=strict_json(read_spec(value['historical_P14_ruff_genuine_capture_return'],root))
    manifest=strict_json(read_spec(value['historical_P14_ruff_capture_manifest'],root))
    typed_equal(strict_json(trace['chunks'][0]['result']['output'].encode('utf8')),summary,'genuine opaque capture output')
    typed_equal(cost['capture_return'],value['historical_P14_ruff_genuine_capture_return'],'cost capture pin')
    typed_equal(cost['expected_output'],value['historical_P14_ruff_capture_manifest'],'cost manifest pin')
    typed_equal(summary['manifest_descriptor'],value['historical_P14_ruff_capture_manifest'],'summary manifest pin')
    typed_equal(manifest['schema'],'flylab.m3.e360.p14-ruff-capture.v1','historic capture schema')
    typed_equal(manifest['request'],call['request'],'historic manifest request')
    identity={'process_id':33356,'start_utc_ticks':639267587608157543}
    for item in (summary,manifest):
        for key,expected in {'call_id':call_id,'capture_complete':True,'exit_observed':True,'owned_state':'exit_observed',
            'ruff_exit_code':0,'first_failure':None,'process_identity':identity}.items():typed_equal(item[key],expected,'actual capture '+key)
    typed_equal(cost['capture_process_identity'],identity,'Native Int64 actual identity')
    for name in ('stdout','stderr'):historical_p12_stream_bytes(value['historical_P14_ruff_'+name],root)
    closure=strict_json(read_spec(value['historical_P14_independent_owned_closure'],root))
    decision=strict_json(read_spec(value['historical_P14_closed_decision'],root))
    historical_approval=strict_json(read_spec(old['existing_human_authorization'],root))
    validate_saved_p14_termination(closure,decision,old,historical_approval,value=value)
    rejection=decision['first_rejection']
    for key,item in {'scanner_dispatched':False,'scanner_call_admitted':False,'scanner_report_created':False,
        'scanner_cost_created':False,'exit_code':1,'message':'Exact unique admitted collector trace',
        'scanner_request':value['historical_P14_scanner_unadmitted_request']}.items():typed_equal(rejection[key],item,'actual Native rejection '+key)
    for key in ('historical_P14_Native_scanner_admission_rejection','historical_P14_ROOT_typed_preservation'):
        read_spec(value[key],root)
    require('e360-generous-p14-scan' not in frozen['e360_calls'],'unadmitted scanner has no call')
    typed_equal(frozen['generated_shared_cumulative_used_s'],583.1700156,'actual used shared')
    typed_equal(frozen['e360_generated_local_used_s'],472.213,'actual used local')
    return frozen


def validate_completed_p14_pass(pass_id,passed,debits,*,root=ROOT):
    frozen=completed_historical_p14_proof(root)
    typed_equal(pass_id,'e360-generous-p14','completed P14 only')
    typed_equal(passed,frozen['e360_passes'][pass_id],'whole historical P14 pass')
    typed_equal(debits[:92],frozen['debits'],'whole92 ordered prefix')


def p15_recovery_event(root=ROOT):
    value=p15_recovery_amendment(root)
    return {'schema':'flylab.m3.e360.p15-recovery-activation.v1','revalidation_amendment':file_spec(P15_RECOVERY_AMENDMENT,root),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited92':value['inherited92'],
        'previous_revalidation_amendment':value['previous_revalidation_amendment'],
        'manual_generated_pass_id':'e360-generous-p15','new_generated_grant_s':0,'human_authorization':value['existing_human_authorization']}


def p15_recovery_activation_allowed(live,*,lock_absent,root=ROOT):
    value=p15_recovery_amendment(root);frozen=completed_historical_p14_proof(root,value=value)
    require(lock_absent is True and not any(key in live for key in P15_RECOVERY_FIELDS),'once P15 ownership0')
    typed_equal(live,frozen,'activation exact whole92')
    budget=current_budget_authority(root,live=live)
    require(budget['suite_recovery_active'] and budget['generated_shared_limit_s']-live['generated_shared_cumulative_used_s']>=798
        and budget['generated_local_limit_s']-live['e360_generated_local_used_s']>=798,'whole798 existing residual only')
    return True


def p15_recovery_delta_valid(before,after,*,root=ROOT):
    require(set(after)==set(before)|set(P15_RECOVERY_FIELDS),'only new P15 pair2')
    for key,item in before.items():typed_equal(after[key],item,'whole92 unchanged '+key)
    typed_equal(after[P15_RECOVERY_FIELDS[0]],p15_recovery_event(root),'exact event8')
    typed_equal(after[P15_RECOVERY_FIELDS[1]],file_spec(P15_RECOVERY_AMENDMENT,root),'literal95430')
    return True


def historical_p15_scope_authority(root=ROOT,*,live=None):
    """Only current P15; immutable completed P14 is a separate reader."""
    if live is None:live=strict_json(bounded_metadata_bytes(registration(root)['shared_ledger_path'],root))
    present=[key in live for key in P15_RECOVERY_FIELDS]
    require(all(present) or not any(present),'complete current P15 pair')
    if not all(present):return False
    value=p15_recovery_amendment(root);frozen=completed_historical_p14_proof(root,value=value)
    require(current_budget_authority(root,live=live)['suite_recovery_active'],'existing P9 budget')
    require(set(frozen)|set(P15_RECOVERY_FIELDS)<=set(live),'whole92 fields plus pair2')
    typed_equal(live[P15_RECOVERY_FIELDS[0]],p15_recovery_event(root),'current event8')
    typed_equal(live[P15_RECOVERY_FIELDS[1]],file_spec(P15_RECOVERY_AMENDMENT,root),'current literal')
    typed_equal(live['debits'][:92],frozen['debits'],'full92 ordered prefix')
    for key in (*value['preserved_activation_fields'],*value['preserved_resource_grant_fields']):typed_equal(live[key],frozen[key],'old24 grants2 '+key)
    for key in ('e360_calls','e360_passes'):
        for identity,item in frozen[key].items():typed_equal(live[key][identity],item,'whole92 history '+key)
    return True


def validate_p15_pass(pass_id,passed,debits,*,root=ROOT):
    raise ValueError('P15 completed history cannot supply prospective admission')
    value=p15_recovery_amendment(root);frozen=completed_historical_p14_proof(root,value=value)
    typed_equal(pass_id,'e360-generous-p15','one manual P15 no future/original/retry')
    require(current_scope_authority(root),'P15 actual activation')
    require(passed['mode']=='generated' and passed['modified'] is True,'modified generated P15')
    typed_equal(passed['actual_failure'],value['historical_P14_closed_decision'],'explicit nonpaid predecessor')
    typed_equal(passed['predecessor_failed_cap_s'],0,'no numeric failed paid cap invented')
    typed_equal(debits[:92],frozen['debits'],'all historical92 retained')
    require(not any(row['descriptor']==passed['actual_failure'] for row in debits),'nonpaid decision never a paid debit')
    typed_equal(passed['revalidation_amendment'],file_spec(P15_RECOVERY_AMENDMENT,root),'P15 direct authority')
    require(passed['source_sha256']!=frozen['e360_passes']['e360-generous-p14']['source_sha256'],'fresh P15 Source9')
    typed_equal(passed['source_sha256'],own_sources(root),'current exact own9')
    require(sum(spec==value['concrete_correction_snapshot'] for spec in passed['concrete_fix'])==1,'reviewed correction')
    for name in (*OWNED,BASE+'/parent-e360-root-account.ps1',BASE+'/parent-e360-root-metadata-helper.py',
        value['ruff_capture_policy']['function_only_source_path'],*value['prospective_native_source_paths'],
        value['ROOT_genuine_collector_policy']['canonical_future_source_path']):
        require(sum(spec==file_spec(name,root) for spec in passed['concrete_fix'])==1,'current concrete leaf '+name)
    for spec in passed['concrete_fix']:read_spec(spec,root)


def paid_request_identity(debit,cost,actual,live,*,root=ROOT):
    """Schema-specific identity binding; ADMIN call_id differs from worker attempt:role cost_id."""
    call_id=cost['call_id'];require(type(call_id) is str and call_id in live['e360_calls'],'paid dictionary call identity')
    call=live['e360_calls'][call_id];request=strict_json(read_spec(call['request'],root,owner='dynamic' if cost['pass_id']==execution_current_pass(root) else 'historical'))
    require(type(request) is dict,'paid typed request mapping')
    for key in ('cost_id','mode','actual_outer_full_return_s'):typed_equal(cost[key],debit[key],'paid cost/debit '+key)
    typed_equal(actual['actual_outer_full_return_s'],debit['actual_outer_full_return_s'],'paid actual/debit amount')
    typed_equal(actual['actual_returned_exit_code'],debit['actual_returned_exit_code'],'paid actual/debit exit')
    for key in ('call_id','pass_id'):
        typed_equal(cost[key],actual[key],'paid actual '+key)
        typed_equal(cost[key],request[key],'paid request '+key)
    typed_equal(call['pass_id'],cost['pass_id'],'paid call pass')
    typed_equal(call['cost_id'],cost['cost_id'],'paid call cost')
    typed_equal(request['call_role'],call['call_role'],'paid request role')
    typed_equal(actual['request'],call['request'],'paid actual request descriptor')
    typed_equal(file_spec(call['request']['path'],root),call['request'],'paid exact read request descriptor')
    for item in (cost,actual,request):
        for key in ('resource_extension','revalidation_amendment'):typed_equal(item[key],call[key],'paid direct '+key)
    if cost['schema']=='flylab.m3.e360.administrative-full-return.v1':
        typed_equal(cost['cost_id'],call_id,'ADMIN costid==callid')
        typed_equal(cost['call_role'],call['call_role'],'ADMIN role')
        typed_equal(cost['request'],call['request'],'ADMIN request descriptor')
        typed_equal(cost['actual_returned_exit_code'],actual['actual_returned_exit_code'],'ADMIN exit')
    else:
        typed_equal(cost['cost_id'],cost['attempt_id']+':'+cost['role'],'worker attempt:role costid')
        typed_equal(cost['attempt_id'],call['attempt_id'],'worker admitted attempt')
        typed_equal(request['attempt_id'],call['attempt_id'],'worker request attempt')
        typed_equal(call['call_role'],cost['role']+'_worker','worker role')
        contract=strict_json(read_spec(cost['contract'],root,owner='dynamic' if cost['pass_id']==execution_current_pass(root) else 'historical'))
        typed_equal(cost['contract']['path'],request['contract'],'worker admitted contract path')
        for key in ('attempt_id','role','pass_id'):typed_equal(cost[key],contract[key],'worker contract '+key)
        typed_equal(contract['revalidation_amendment'],call['revalidation_amendment'],'worker contract authority')
        typed_equal(cost['source_sha256'],contract['source_sha256'],'worker full contract Source map')
        for table in ('e360_pending_actuals','e360_reservations'):
            pending=live.get(table,{}).get(cost['cost_id'])
            if pending is not None:
                for key in ('attempt_id','role','mode'):typed_equal(pending[key],cost[key],'worker pending '+key)
                if 'call_id' in pending:typed_equal(pending['call_id'],call_id,'worker pending call')
                if 'contract' in pending:typed_equal(pending['contract'],cost['contract'],'worker pending contract')
                if 'actual_tool_return' in pending:typed_equal(pending['actual_tool_return'],cost['actual_tool_return'],'worker pending actual')
        typed_equal(cost['exit_code'],actual['actual_returned_exit_code'],'worker exit')
    return request


def p15_collector_proof(trace_spec,request,request_spec,seconds,exit_code,*,root=ROOT):
    """Replay immutable genuine transport metadata; never sample a new clock."""
    trace=strict_json(read_spec(trace_spec,root))
    typed_equal(trace['schema'],'flylab.m3.e360.p15v2.ROOT-genuine-collector-state.v1','collector schema')
    typed_equal(request['pass_id'],'e360-generous-p15','collector current pass only')
    for key in ('pass_id','call_id','call_role'):
        typed_equal(trace[key],request[key],'collector admitted '+key)
    typed_equal(trace['request'],request_spec,'collector exact admitted request')
    typed_equal(trace['collector_source'],request['collector_source'],'collector request source pin')
    typed_equal(trace['collector_source'],file_spec(BASE+'/parent-e360-p15v2-ROOT-genuine-collector.txt',root),'collector current source')
    typed_equal(trace['state_key'],'e360:p15v2:ROOT-genuine:'+request['call_id'],'unique collector role state')
    typed_equal(trace_spec['path'],BASE+'/parent-'+request['call_id']+'-ROOT-collector-trace.json','canonical trace publication')
    typed_equal(trace['next_role_allowed'],False,'collector cannot admit another role')
    typed_equal(trace['collection_status'],'FINAL_RECEIVED','first genuine FINAL collected independently of sticky error')
    typed_equal(trace['collection_stopped'],False,'no stopped transport may invent a later FINAL')
    typed_equal(trace['transport_error'],None,'genuine FINAL without a transport exception')
    caps={'ruff':3,'scanner':20,'fixture_build':25,'fixture_worker':55,'fixture_seal':15,
        'suite_build':30,'suite_worker':600,'suite_seal':20,'qualification':30}
    typed_equal(trace['cost_name'],request['call_id'],'collector once cost name')
    typed_equal(trace['cap_s'],caps[request['call_role']],'immutable role cap')
    require(type(trace['max_output_tokens']) is int and trace['max_output_tokens']>=1000,'complete output request bound')
    typed_equal(trace['exec_args'],{'cmd':"pwsh -NoProfile -File '"+BASE+
        "/parent-e360-p15v2-ROOT-dispatch-adapter.ps1' -CallRole '"+request['call_role']+"'",
        'workdir':'C:/Users/ckthd/Dev/Projects/fly_lab'},'actual ROOT adopted collector dispatch body')
    dispatch=trace['dispatch_ms'];chunks=trace['chunks']
    require(type(dispatch) is int and dispatch>0 and type(chunks) is list and chunks,'genuine dispatch and chunks')
    last=dispatch;known=None;first=None;first_exit=None;complete=True
    for chunk in chunks:
        received=chunk['received_ms'];result=chunk['result']
        require(type(received) is int,'original integer receive clock')
        if received<last:
            require(trace['phase']=='ERROR_HOLD' and type(trace['first_error']) is dict,'clock contradiction retained first error')
            complete=False
        require(first is None,'no result after first genuine FINAL')
        last=received
        require(type(result) is dict,'contradictory transport cannot acquire a later FINAL')
        has_session=result.get('session_id') is not None
        has_exit=result.get('exit_code') is not None
        session=result.get('session_id');returned=result.get('exit_code')
        if has_session and type(session) is int and session>0:
            if known is None:known=session
            require(known==session,'changed session cannot acquire a later FINAL')
        if has_exit and type(returned) is int and not has_session:
            first,first_exit=received,returned
        else:require(has_session and type(session) is int and session>0 and not has_exit,
                     'contradictory result shape cannot acquire a later FINAL')
        output=result.get('output')
        if (type(output) is not str or result.get('truncated') is True or result.get('output_truncated') is True
            or re.search(r'Warning: truncated output|tokens truncated|output was truncated',output or '')):complete=False
    require(first is not None,'lost FINAL remains pending and cannot invent a debit')
    typed_equal(trace['session_id'],known,'first known tool session retained')
    typed_equal(trace['first_final_ms'],first,'immutable first FINAL sample')
    typed_equal(trace['exit_code'],first_exit,'immutable first FINAL exit')
    require(type(seconds) in (int,float) and not isinstance(seconds,bool) and math.isfinite(seconds) and seconds>0,'genuine positive full return')
    elapsed=(first-dispatch)/1000.0
    require(type(trace['full_return_s']) in (int,float) and math.isfinite(trace['full_return_s'])
            and trace['full_return_s']>0,'saved FINAL duration must be a finite positive JSON number')
    require(trace['full_return_s']==elapsed and seconds==elapsed,'exact dispatch to FINAL full return')
    typed_equal(exit_code,first_exit,'actual first FINAL exit')
    require(trace['phase'] in ('FINAL_RECEIVED','ERROR_HOLD'),'closed collector state')
    if trace['phase']=='ERROR_HOLD':
        require(type(trace['first_error']) is dict,'retain first collector failure')
        complete=False
    else:
        typed_equal(trace['first_error'],None,'complete collector has no first error')
        typed_equal(trace['publication'],'REQUIRED','Native publication follows saved FINAL')
    if not any(type(c['result']) is dict and type(c['result'].get('output')) is str
        and c['result']['output'].strip() for c in chunks):complete=False
    if not complete:require(trace['phase']=='ERROR_HOLD' and type(trace['first_error']) is dict,
                            'every collector failure retains the sticky original error')
    return {'valid':complete,'trace':trace_spec,'full_return_s':elapsed,'exit_code':first_exit}




P15_SAVED_RECORD_KEYS={'cost':'historical_P15_ruff_full_return','actual':'historical_P15_ruff_actual_tool_return','request':'historical_P15_ruff_actual_request','summary':'historical_P15_ruff_genuine_capture_return','manifest':'historical_P15_ruff_capture_manifest','trace':'historical_P15_ruff_genuine_collector_trace','closure':'historical_P15_independent_owned_closure','decision':'historical_P15_closed_decision',}
P15_SAVED_RECORD_LITERALS={
    'cost':r'''{"schema":"flylab.m3.e360.administrative-full-return.v1","mode":"generated","charged_once":true,"cost_id":"e360-generous-p15-ruff","actual_outer_full_return_s":1.799,"actual_tool_return":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-actual.json","bytes":1129,"sha256":"f6716b2dc31ccdc5eb3f4bb5325306b2d19009ac1a48db9cd1ff465bcc16a83e"},"actual_returned_exit_code":1,"original_array_reads":0,"GPU_contexts":0,"resource_extension":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-approved-generous-registration-v2.json","bytes":5461,"sha256":"ea5afdffb09f6a118a18e3039546a911bbc9ed1465cafb6a1fe472ce3471862b"},"revalidation_amendment":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p15-recovery-amendment-v2-final.json","bytes":95430,"sha256":"88ae0799fda49d55e4d03b20bd9d1048c993f9cc6f8d1d9399be24320e42396c"},"call_id":"e360-generous-p15-ruff","pass_id":"e360-generous-p15","call_role":"ruff","request":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-final-request.json","bytes":4376,"sha256":"9beef9dfbe70f8831bcd8579bb3748d52018c0b8cfeeed7ec85303c85371be5e"},"source_sha256":{"tests/test_c_banc_source_closure_diagnostic.py":"f43705f5fac6dbaecd3174e297b0e2a871f43423106defc4a1d00f292daa2355","verification/banc-natural-gait-m3-20260919/parent-e360-root-account.ps1":"cf0f73ac52c9d7c11c76b4cfe29f7c0382c865305ac87bbccfc2d53cad759b5a","tools/verify_c_banc_m3_candidate_body_handoff.py":"5b43ee4ee34a8270ba123d11e99e9820b6f2d25644c027bc46a86ab9b6f7532a","verification/banc-natural-gait-m3-20260919/parent-e360-root-metadata-helper.py":"999ce67095cabf4db0956f715b97865369e01a1aa96ccd4996600e2bc4a2c57c","tests/test_c_banc_candidate_body_handoff.py":"93476a2cb72e5bf7585d11ade9f76c207967eff8aadaf918198219bc4379cee6","tools/verify_c_banc_m3_startup_baseline.py":"4e853f8c4fae105102625816f3075d3b4b7d3ac7312506dd7b3e392e00607342","verification/banc-natural-gait-m3-20260919/parent-e267-supervise.py":"1d8682505f8060d3d8bcb50508811b9c4da531bdbf07c96200065bcc3324df8d","tools/verify_c_banc_m3_source_closure_diagnostic.py":"27dc80c21db1cd9dea0c6b1884e7eb49652de148322bf672fcd0ace019a470e9","verification/banc-natural-gait-m3-20260919/parent-e346-supervise-identities.py":"cf7b04731ed4ea8bb626047b577ae5b5b1e31cccd093415d450c839e998b52cd"},"expected_output":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-output.json","bytes":1781,"sha256":"28f5dc042498dcb9b5a6d68572c41cf798f2c71625e5ece8fa6cd138fbd15b39"},"internal_completion":null,"collector_trace":{"sha256":"05d0d0c0e089977399e76d678805dd51ef1738d82f76c44e93aa1aa71071bbe7","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-ROOT-collector-trace.json","bytes":2953},"collector_proof_valid":true,"capture_return":{"sha256":"05e457ccf77011685c23fd5a05d151c7d0050dda115b3d6df89a13808d66b184","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-genuine-capture-return.json","bytes":1476},"capture_proof_errors":[],"ruff_exit_code":1,"capture_complete":true,"capture_publication_verified":true,"capture_owned_state":"exit_observed","capture_process_identity":{"start_utc_ticks":639268015933086785,"process_id":32628},"capture_root_tool_exit_required":1,"capture_first_failure":{"slot":"ruff_exit_nonzero","kind":"ruff_exit","observed_s":0.5343751},"internal_completion_valid":true}
''',
    'actual':r'''{"actual_returned_exit_code":1,"actual_outer_full_return_s":1.799,"measurement":"ROOT external helper dispatch through actual final return","original_array_reads":0,"GPU_contexts":0,"resource_extension":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-approved-generous-registration-v2.json","bytes":5461,"sha256":"ea5afdffb09f6a118a18e3039546a911bbc9ed1465cafb6a1fe472ce3471862b"},"revalidation_amendment":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p15-recovery-amendment-v2-final.json","bytes":95430,"sha256":"88ae0799fda49d55e4d03b20bd9d1048c993f9cc6f8d1d9399be24320e42396c"},"request":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-final-request.json","bytes":4376,"sha256":"9beef9dfbe70f8831bcd8579bb3748d52018c0b8cfeeed7ec85303c85371be5e"},"call_id":"e360-generous-p15-ruff","pass_id":"e360-generous-p15","collector_trace":{"sha256":"05d0d0c0e089977399e76d678805dd51ef1738d82f76c44e93aa1aa71071bbe7","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-ROOT-collector-trace.json","bytes":2953},"collector_proof_valid":true}
''',
    'request':r'''{"call_id":"e360-generous-p15-ruff","pass_id":"e360-generous-p15","call_role":"ruff","mode":"generated","completion":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-completion.json","action":"ruff","expected_output":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-output.json","actual_failure":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p14-closed-decision.json","bytes":6122,"sha256":"7a3c0f94a36b6f2f29ac3e516733a61ab24f90fed97c9c987af65aa12c0bcfea"},"concrete_fix":[{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-ROOT-design-reflection.json","bytes":2964,"sha256":"ca1ffe38e2367faab45bdd2c08362566ce0b80db35108cedf5246d91e3247c4f"},{"path":"tools/verify_c_banc_m3_source_closure_diagnostic.py","bytes":438868,"sha256":"27dc80c21db1cd9dea0c6b1884e7eb49652de148322bf672fcd0ace019a470e9"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-root-metadata-helper.py","bytes":14967,"sha256":"999ce67095cabf4db0956f715b97865369e01a1aa96ccd4996600e2bc4a2c57c"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-root-account.ps1","bytes":362547,"sha256":"cf0f73ac52c9d7c11c76b4cfe29f7c0382c865305ac87bbccfc2d53cad759b5a"},{"path":"tests/test_c_banc_source_closure_diagnostic.py","bytes":332001,"sha256":"f43705f5fac6dbaecd3174e297b0e2a871f43423106defc4a1d00f292daa2355"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-root-final-freeze.json","bytes":85760,"sha256":"91668fe3ddcfbae74851b61ea5e419858808237df60a524e59819c9bfa627799"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-capture-proposed.ps1","bytes":29710,"sha256":"4ad6aad3d2f0cd0c9c3fbd3365d897326f13c23fff12d81d75eb16fba4790a12"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-prepare-native.ps1","bytes":4246,"sha256":"4a25511f713de335bdf133e7aebaf8dfc8a214cb73c2e8c5616f006a54c62596"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-bookkeeping-native.ps1","bytes":11006,"sha256":"02380db4cfa678dab8c3487ec681cd7f39dde18795a5f97377820538bd3ee1a0"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-ROOT-dispatch-adapter.ps1","bytes":8727,"sha256":"75fd1770090e5d13c2fe4bc0b187dd4b1b12e31a9cf99f6aeea818c4d16f9aa5"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-ROOT-genuine-collector.txt","bytes":7356,"sha256":"fc2d534bc5a0e0cf4f3a03d85e882e33323ff49b3b77de44a73b82e08ba34f4a"}],"collector_source":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-ROOT-genuine-collector.txt","bytes":7356,"sha256":"fc2d534bc5a0e0cf4f3a03d85e882e33323ff49b3b77de44a73b82e08ba34f4a"},"schema":"flylab.m3.e360.helper-request.v1","source_sha256":{"tests/test_c_banc_source_closure_diagnostic.py":"f43705f5fac6dbaecd3174e297b0e2a871f43423106defc4a1d00f292daa2355","verification/banc-natural-gait-m3-20260919/parent-e360-root-account.ps1":"cf0f73ac52c9d7c11c76b4cfe29f7c0382c865305ac87bbccfc2d53cad759b5a","tools/verify_c_banc_m3_candidate_body_handoff.py":"5b43ee4ee34a8270ba123d11e99e9820b6f2d25644c027bc46a86ab9b6f7532a","verification/banc-natural-gait-m3-20260919/parent-e360-root-metadata-helper.py":"999ce67095cabf4db0956f715b97865369e01a1aa96ccd4996600e2bc4a2c57c","tests/test_c_banc_candidate_body_handoff.py":"93476a2cb72e5bf7585d11ade9f76c207967eff8aadaf918198219bc4379cee6","tools/verify_c_banc_m3_startup_baseline.py":"4e853f8c4fae105102625816f3075d3b4b7d3ac7312506dd7b3e392e00607342","verification/banc-natural-gait-m3-20260919/parent-e267-supervise.py":"1d8682505f8060d3d8bcb50508811b9c4da531bdbf07c96200065bcc3324df8d","tools/verify_c_banc_m3_source_closure_diagnostic.py":"27dc80c21db1cd9dea0c6b1884e7eb49652de148322bf672fcd0ace019a470e9","verification/banc-natural-gait-m3-20260919/parent-e346-supervise-identities.py":"cf7b04731ed4ea8bb626047b577ae5b5b1e31cccd093415d450c839e998b52cd"},"resource_extension":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-approved-generous-registration-v2.json","bytes":5461,"sha256":"ea5afdffb09f6a118a18e3039546a911bbc9ed1465cafb6a1fe472ce3471862b"},"revalidation_amendment":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p15-recovery-amendment-v2-final.json","bytes":95430,"sha256":"88ae0799fda49d55e4d03b20bd9d1048c993f9cc6f8d1d9399be24320e42396c"},"closure_report":null}
''',
    'summary':r'''{"call_id":"e360-generous-p15-ruff","capture_complete":true,"publication_verified":true,"manifest_descriptor":{"bytes":1781,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-output.json","sha256":"28f5dc042498dcb9b5a6d68572c41cf798f2c71625e5ece8fa6cd138fbd15b39"},"ruff_exit_code":1,"exit_observed":true,"owned_state":"exit_observed","admission_complete":true,"start_attempted":true,"started_observed":true,"start_known_failed":false,"process_identity":{"start_utc_ticks":639268015933086785,"process_id":32628},"streams":[{"stream":"stdout","observed_bytes":1905,"retained_bytes":1905,"eof_observed":true,"read_fault":false,"overflow":false,"readback_verified":true,"descriptor":{"bytes":1905,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-stdout.txt","sha256":"8533cd7e1eef1d691d1fef6a81eb6dae7d9c859e3f9de1580984e922afff1601"}},{"stream":"stderr","observed_bytes":0,"retained_bytes":0,"eof_observed":true,"read_fault":false,"overflow":false,"readback_verified":true,"descriptor":{"bytes":0,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}}],"first_failure":{"slot":"ruff_exit_nonzero","kind":"ruff_exit","observed_s":0.5343751},"wrapper_error_slots":[],"internal_elapsed_s":0.5690439,"external_full_return_s":null,"external_cap_s":3,"runtime_qualified":false,"root_tool_exit_required":1}
''',
    'manifest':r'''{"schema":"flylab.m3.e360.p15-ruff-capture.v1","call_id":"e360-generous-p15-ruff","pass_id":"e360-generous-p15","entry_utc":"2026-10-05T12:53:12.8543490Z","internal_observed_s":0.5543121,"external_cap_s":3,"soft_deadline_s":2,"cleanup_deadline_s":2.5,"process_identity":{"start_utc_ticks":639268015933086785,"process_id":32628},"start_attempted":true,"started_observed":true,"start_known_failed":false,"owned_state":"exit_observed","exit_observed":true,"ruff_exit_code":1,"kill_attempted":false,"kill_requested":false,"timeout_observed":false,"streams":[{"stream":"stdout","observed_bytes":1905,"retained_bytes":1905,"eof_observed":true,"read_fault":false,"overflow":false,"readback_verified":true,"descriptor":{"bytes":1905,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-stdout.txt","sha256":"8533cd7e1eef1d691d1fef6a81eb6dae7d9c859e3f9de1580984e922afff1601"}},{"stream":"stderr","observed_bytes":0,"retained_bytes":0,"eof_observed":true,"read_fault":false,"overflow":false,"readback_verified":true,"descriptor":{"bytes":0,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}}],"first_failure":{"slot":"ruff_exit_nonzero","kind":"ruff_exit","observed_s":0.5343751},"wrapper_errors":{},"capture_complete":true,"request":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-final-request.json","bytes":4376,"sha256":"9beef9dfbe70f8831bcd8579bb3748d52018c0b8cfeeed7ec85303c85371be5e"},"revalidation_amendment":{"bytes":95430,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p15-recovery-amendment-v2-final.json","sha256":"88ae0799fda49d55e4d03b20bd9d1048c993f9cc6f8d1d9399be24320e42396c"}}''',
    'trace':r'''{"schema":"flylab.m3.e360.p15v2.ROOT-genuine-collector-state.v1","pass_id":"e360-generous-p15","call_id":"e360-generous-p15-ruff","call_role":"ruff","cost_name":"e360-generous-p15-ruff","cap_s":3,"state_key":"e360:p15v2:ROOT-genuine:e360-generous-p15-ruff","request":{"bytes":4376,"sha256":"9beef9dfbe70f8831bcd8579bb3748d52018c0b8cfeeed7ec85303c85371be5e","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-final-request.json"},"collector_source":{"bytes":7356,"sha256":"fc2d534bc5a0e0cf4f3a03d85e882e33323ff49b3b77de44a73b82e08ba34f4a","path":"verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-ROOT-genuine-collector.txt"},"exec_args":{"cmd":"pwsh -NoProfile -File 'verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-ROOT-dispatch-adapter.ps1' -CallRole 'ruff'","workdir":"C:/Users/ckthd/Dev/Projects/fly_lab"},"max_output_tokens":10000,"phase":"FINAL_RECEIVED","collection_status":"FINAL_RECEIVED","collection_stopped":false,"transport_error":null,"dispatch_ms":1791204791750,"first_final_ms":1791204793549,"full_return_s":1.799,"session_id":null,"chunks":[{"received_ms":1791204793549,"result":{"chunk_id":"69da5d","wall_time_seconds":1.4454481000000001,"exit_code":1,"original_token_count":370,"output":"{\"call_id\":\"e360-generous-p15-ruff\",\"capture_complete\":true,\"publication_verified\":true,\"manifest_descriptor\":{\"bytes\":1781,\"path\":\"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-output.json\",\"sha256\":\"28f5dc042498dcb9b5a6d68572c41cf798f2c71625e5ece8fa6cd138fbd15b39\"},\"ruff_exit_code\":1,\"exit_observed\":true,\"owned_state\":\"exit_observed\",\"admission_complete\":true,\"start_attempted\":true,\"started_observed\":true,\"start_known_failed\":false,\"process_identity\":{\"start_utc_ticks\":639268015933086785,\"process_id\":32628},\"streams\":[{\"stream\":\"stdout\",\"observed_bytes\":1905,\"retained_bytes\":1905,\"eof_observed\":true,\"read_fault\":false,\"overflow\":false,\"readback_verified\":true,\"descriptor\":{\"bytes\":1905,\"path\":\"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-stdout.txt\",\"sha256\":\"8533cd7e1eef1d691d1fef6a81eb6dae7d9c859e3f9de1580984e922afff1601\"}},{\"stream\":\"stderr\",\"observed_bytes\":0,\"retained_bytes\":0,\"eof_observed\":true,\"read_fault\":false,\"overflow\":false,\"readback_verified\":true,\"descriptor\":{\"bytes\":0,\"path\":\"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-stderr.txt\",\"sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\"}}],\"first_failure\":{\"slot\":\"ruff_exit_nonzero\",\"kind\":\"ruff_exit\",\"observed_s\":0.5343751},\"wrapper_error_slots\":[],\"internal_elapsed_s\":0.5690439,\"external_full_return_s\":null,\"external_cap_s\":3,\"runtime_qualified\":false,\"root_tool_exit_required\":1}\r\n"}}],"exit_code":1,"first_error":null,"publication":"REQUIRED","next_role_allowed":false}
''',
    'closure':r'''{
  "schema": "flylab.m3.e360.p15v2.ROOT-independent-owned-closure.v1",
  "actual_actor": "ROOT",
  "date": "2026-10-05",
  "closed_ledger93": {
    "bytes": 133168,
    "sha256": "8af3bf746a8a10a2e8bcb407cd1fe421df84b91ddcf7b255ceefc4698a2a9817",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-closed-ledger93.json"
  },
  "all_old92_once_debits_preserved": true,
  "old_calls42_passes14_preserved": true,
  "whole93_calls43_passes15": true,
  "new_paid_calls": 1,
  "new_paid_seconds": 1.799,
  "pending_actuals": 0,
  "reservations": 0,
  "blocking_holds": 0,
  "unreturned_calls": 0,
  "original_jobs": 0,
  "original_reserved_s": 0,
  "known_owned_ruff": {
    "exit_observed": true,
    "both_streams_closed": true,
    "observed_start_utc_ticks": null,
    "matching_owned_live": false,
    "expected_start_utc_ticks": 639268015933086785,
    "process_id": 32628,
    "no_process_termination_performed": true
  },
  "actual_as_executed_freeze": {
    "bytes": 85760,
    "sha256": "91668fe3ddcfbae74851b61ea5e419858808237df60a524e59819c9bfa627799",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-root-final-freeze.json"
  },
  "current_own9_preserved": true,
  "protected322_preserved": true,
  "old_Native14_and_currentNative5_preserved": true,
  "original_IO_GPU_body_neural_UI": false,
  "runtime_qualified": false,
  "F1_F8_complete": false
}
''',
    'decision':r'''{
  "schema": "flylab.m3.e360.p15v2.ROOT-closed-decision.v1",
  "actual_actor": "ROOT",
  "date": "2026-10-05",
  "status": "CLOSED_FIRST_PAID_RUFF_FAILURE_AFTER_CORRECTED_ACTIVATION",
  "latest_human_quote": "수정안 승인할게. 진행해줘",
  "approved_literal": {
    "sha256": "88ae0799fda49d55e4d03b20bd9d1048c993f9cc6f8d1d9399be24320e42396c",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p15-recovery-amendment-v2-final.json",
    "bytes": 95430
  },
  "human_authorization": {
    "sha256": "5bf5a262513693261d46392f61a7f7890d1f937ee48511e920619aa18d68fdb3",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p15-correction-human-authorization.json",
    "bytes": 4826
  },
  "actual_as_executed_freeze": {
    "bytes": 85760,
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-root-final-freeze.json",
    "sha256": "91668fe3ddcfbae74851b61ea5e419858808237df60a524e59819c9bfa627799"
  },
  "actual_activation": {
    "bytes": 2158,
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-ROOT-activation-tool-evidence.json",
    "sha256": "eb4ac73b9e08e34403dd767076106e563d7c244e7cb67daf216e614d3563e73e"
  },
  "activation_attempts_this_correction": 1,
  "activation_success_this_correction": 1,
  "prior_failed_P15_v1": {
    "bytes": 4808,
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p15-closed-decision.json",
    "sha256": "d6230f638f78338262085dbd7b4160d6625e0d20cd38484682ea1d4e08c59671"
  },
  "prior_P15_v1_failure_not_overwritten": true,
  "corrected_Native_full_history_proof_actual_activation_and_callbegin_and_account_passed": true,
  "Python_full_proof_and_Native16_paid_test_NOT_RUN": true,
  "actual_Ruff_cost": {
    "bytes": 3422,
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-full-return.json",
    "sha256": "1bf44c742b4f83ca727201b814457ee0443aaf03b07df830369e8b2aa47e8960"
  },
  "actual_Ruff_stdout": {
    "bytes": 1905,
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-stdout.txt",
    "sha256": "8533cd7e1eef1d691d1fef6a81eb6dae7d9c859e3f9de1580984e922afff1601"
  },
  "actual_Ruff_exit_code": 1,
  "new_paid_roles": 1,
  "new_paid_seconds": 1.799,
  "capture_and_collector_proof_valid": true,
  "Ruff_full_return_within_cap3": true,
  "first_failure": "I001 one import separator and B023 two unbound loop map references in tests helper",
  "closed_ledger93": {
    "bytes": 133168,
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-closed-ledger93.json",
    "sha256": "8af3bf746a8a10a2e8bcb407cd1fe421df84b91ddcf7b255ceefc4698a2a9817"
  },
  "independent_owned_closure": {
    "bytes": 1448,
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-ROOT-independent-owned-closure.json",
    "sha256": "a6385fc9e49377b2c4cf2ebe07b45d00738ea777e71f32e7dbba6ee0a7c96b64"
  },
  "old92_costs_calls42_passes14_typed_preserved": true,
  "whole93_calls43_passes15": true,
  "failed_cap_s": 3,
  "next_index": 1,
  "shared_used_s": 584.9690156,
  "shared_limit_s": 1400,
  "shared_remaining_s": 815.0309844,
  "local_used_s": 474.012,
  "local_limit_s": 1289,
  "local_remaining_s": 814.988,
  "offset": 42,
  "additional_grant": 0,
  "original_cumulative": 10020.115960899504,
  "original_stage": 126.361,
  "E360_original_used": 0,
  "pending": 0,
  "reservations": 0,
  "holds": 0,
  "unreturned": 0,
  "roles_after_first_failure": 0,
  "scanner_fixture_suite_Q": 0,
  "actual_test_phases": 0,
  "actual_Native16_test_probe": 0,
  "original_GPU_body_neural_UI": 0,
  "actual_SAME_Sol_ONE_R1": {
    "bytes": 31231,
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-sol-R1-handoff.json",
    "sha256": "4fdca8eff41ae505ae58e369d9340561d311a54aec8f7bb0b554b7ade3192a07"
  },
  "ROOT_SECOND_R1": {
    "bytes": 2937,
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-ROOT-SECOND-R1-Source-review.json",
    "sha256": "e77c820f0b4684bdaee00d293c818d520170aed8b422ac3f36dd75cb1262b6d1"
  },
  "R1_OFF_ACTIVE_selected_tests": {
    "sha256": "adfb614b0952bc72ae607903d660229a2f602197a753e9be356e89a63c3ad42a",
    "bytes": 332080,
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p15v2-sol-R1-source-tests.py"
  },
  "R1_runtime_retested": false,
  "R1_not_applied_to_execution_history": true,
  "Source_as_executed_and_root_GOAL_PLAN_preserved": true,
  "P16_authorized": false,
  "original_execution_authority": false,
  "automatic_retries": 0,
  "runtime_qualified": false,
  "natural_gait": false,
  "F1_F8_complete": false,
  "goal_complete": false,
  "goal_API_status": "prior BLOCKED retained; no resume/reset/complete call"
}
''',
    'call':r'''{
  "reserved_s": 3,
  "state": "charged",
  "mode": "generated",
  "call_role": "ruff",
  "cost_id": "e360-generous-p15-ruff",
  "resource_extension": {
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-approved-generous-registration-v2.json",
    "bytes": 5461,
    "sha256": "ea5afdffb09f6a118a18e3039546a911bbc9ed1465cafb6a1fe472ce3471862b"
  },
  "pass_id": "e360-generous-p15",
  "revalidation_amendment": {
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p15-recovery-amendment-v2-final.json",
    "bytes": 95430,
    "sha256": "88ae0799fda49d55e4d03b20bd9d1048c993f9cc6f8d1d9399be24320e42396c"
  },
  "sequence_index": 0,
  "future_reserved_s": 795,
  "attempt_id": null,
  "request": {
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-final-request.json",
    "bytes": 4376,
    "sha256": "9beef9dfbe70f8831bcd8579bb3748d52018c0b8cfeeed7ec85303c85371be5e"
  }
}''',
}



@historical_dependencies()
def _p16_recovery_amendment(root=ROOT):
    raw=literal_bytes(P16_RECOVERY_AMENDMENT,root)
    require(len(raw)==P16_RECOVERY_BYTES and sha(raw)==P16_RECOVERY_SHA,'exact approved P16 literal')
    value=strict_json(raw);old=p15_recovery_amendment(root)
    expected={'schema':'flylab.m3.e360.p16-recovery-amendment.v1','manual_generated_pass_id':'e360-generous-p16',
        'predecessor_pass_id':'e360-generous-p15','required_exact_prefix_length':93,'local_cost_offset':42,
        'generated_shared_limit_s':1400,'generated_local_limit_s':1289,'new_generated_grant_s':0,
        'generated_pass_limit':16,'modified_generated_pass_limit':15,'original_pass_limit':1,'total_pass_limit':17,
        'automatic_retries':0,'original_local_limit_s':20,'original_stage_limit_s':300,'original_global_limit_s':10200,
        'original_worker_maximum':1,'original_automatic_retries':0,'source_hash_cycle_allowed':False,
        'whole_fresh_chain_required':True,'primary_total_reserved_s':798,'predecessor_failed_cap_s':3,
        'minimum_current_test_functions':84,'maximum_current_test_functions':84,'minimum_current_test_phases':252,
        'pass16_authorized':True,'pass17_authorized':False,'P15_resume_authorized':False,
        'original_execution_authority':False,'generated_scope_ends_after_Q_debit':True,
        'generated_scope_ends_after_first_closed_failure':True,
        'predecessor_failure_kind':'closed_paid_Ruff_exit1_after_corrected_activation'}
    for key,item in expected.items():typed_equal(value[key],item,'compiled P16 '+key)
    typed_equal(value['previous_revalidation_amendment'],file_spec(P15_RECOVERY_AMENDMENT,root),'immutable reader16')
    for key in ('resource_extension','generated_call_order','generated_call_schedule','effective_helper_limits',
        'effective_roles','scanner_limits','checkpoint_policy','successful_suffix_after_s',
        'remaining_generated_administration_reserve_s','fixture_future_administration_reserved_s',
        'suite_future_administration_reserved_s','preserved_test_functions','preserved_resource_grant_fields'):
        typed_equal(value[key],old[key],'preserved finite P15 policy '+key)
    policy=copy.deepcopy(old['fixed_authority_scope_policy'])
    policy['reader_ids'].append('p16_recovery_amendment');policy['max_entries']=17
    typed_equal(value['fixed_authority_scope_policy'],policy,'reader17 exact old16 capacity17')
    typed_equal(value['preserved_activation_fields'],[*old['preserved_activation_fields'],*P15_RECOVERY_FIELDS],'old26')
    typed_equal(value['new_active_fields'],list(P16_RECOVERY_FIELDS),'newpair2')
    typed_equal(value['activation_policy'],{'activation_schema':'flylab.m3.e360.p16-recovery-activation.v1',
        'changed_ledger_fields':[],'new_ledger_fields':list(P16_RECOVERY_FIELDS),'new_grant_object_allowed':False,
        'activation_keys':['schema','revalidation_amendment','resource_extension','inherited93',
        'previous_revalidation_amendment','manual_generated_pass_id','new_generated_grant_s','human_authorization']},'P16 event8')
    rows=value['historical_concrete_fix_relocations']
    require(type(rows) is list and len(rows)==54,'exact relocation54')
    typed_equal(rows[:50],old['historical_concrete_fix_relocations'],'unchanged old50')
    typed_equal([r['pass_id'] for r in rows[50:]],['e360-generous-p15']*4,'actual P15v2 frozen4')
    for row in rows:
        leaf=row.get('snapshot',row.get('frozen'))
        for key in ('bytes','sha256'):typed_equal(row['original'][key],leaf[key],'same-byte historical Source '+key)
        read_spec(leaf,root)
    for spec in value.values():
        if type(spec) is dict and set(spec)=={'path','bytes','sha256'}:
            historical_p12_stream_bytes(spec,root) if spec['bytes']==0 else read_spec(spec,root)
    for spec in value['current_P16_source_snapshots']:read_spec(spec,root)
    for spec in value['historical_P15_Native4']:read_spec(spec,root)
    approval=strict_json(read_spec(value['existing_human_authorization'],root))
    for key,item in {'human_quote':'제안해준 다음 작업 승인할 게 진행해줘.','manual_generated_pass_id':'e360-generous-p16',
        'manual_P16_authorized':True,'new_generated_grant_s':0,'existing_shared_limit_s':1400,'existing_local_limit_s':1289,
        'reserved_whole_generated_schedule_s':798,'automatic_retries':0,'P17_authorized':False,
        'P15_resume_authorized':False,'original_execution_authority':False,'baseline_debits':93,
        'baseline_calls':43,'baseline_passes':15,'baseline_shared_used_s':584.9690156,
        'baseline_local_used_s':474.012}.items():typed_equal(approval[key],item,'direct P16 approval '+key)
    typed_equal(value['prospective_native_source_paths'],[BASE+'/parent-e360-p16-'+name+'.ps1'
        for name in ('prepare-native','bookkeeping-native','ROOT-dispatch-adapter')],'P16 Native3 canonical paths')
    completed_historical_p15_proof(root,value=value)
    return value


def p16_recovery_amendment(root=ROOT):
    return fixed_reader('p16_recovery_amendment',_p16_recovery_amendment,root)


def validate_saved_p15_record(name,document):
    """Narrow production typed gate over the direct actual record; never defaults a missing field."""
    require(name in P15_SAVED_RECORD_LITERALS,'known actual P15 record')
    expected=strict_json(P15_SAVED_RECORD_LITERALS[name].encode('utf8'))
    typed_equal(document,expected,'actual P15 '+name+' exact typed record')


@historical_dependencies()
def completed_historical_p15_proof(root=ROOT,*,value=None):
    """Immutable completed P15v2 proof to whole93 END. No current selector/Source4/PID/clock/contract."""
    if value is None:
        raw=literal_bytes(P16_RECOVERY_AMENDMENT,root)
        require(len(raw)==P16_RECOVERY_BYTES and sha(raw)==P16_RECOVERY_SHA,'immutable P16 history literal')
        value=strict_json(raw)
    old=p15_recovery_amendment(root);prior=completed_historical_p14_proof(root,value=old)
    frozen=strict_json(read_spec(value['inherited93'],root))['ledger']
    typed_equal(frozen,strict_json(read_spec(value['historical_P15_closed_ledger93'],root)),'exact immutable whole93')
    typed_equal(prior,strict_json(read_spec(value['historical_P15_inherited92'],root))['ledger'],'exact wholeold92')
    require(len(frozen['debits'])==93 and len(frozen['e360_calls'])==43 and len(frozen['e360_passes'])==15,'93/43/15')
    typed_equal(frozen['debits'][:92],prior['debits'],'all92 ordered costs')
    for key in (*old['preserved_activation_fields'],*old['preserved_resource_grant_fields']):
        typed_equal(frozen[key],prior[key],'unchanged old24/grants2 '+key)
    typed_equal(frozen[P15_RECOVERY_FIELDS[0]],p15_recovery_event(root),'historical P15 event8')
    typed_equal(frozen[P15_RECOVERY_FIELDS[1]],value['historical_P15_literal'],'historical P15 literal')
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(frozen[key][identity],item,'whole92 historical '+key)
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):
        typed_equal(frozen[key],{},'completed ownership0 '+key)
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(frozen[key],0,'original0')
    own=value['historical_P15_frozen_own_sources_sha256'];passed=frozen['e360_passes']['e360-generous-p15']
    for key,item in {'mode':'generated','modified':True,'predecessor_failed_cap_s':0,'failed_cap_s':3,'next_index':1,
        'actual_failure':old['historical_P14_closed_decision'],'failed_cost':value['historical_P15_ruff_full_return'],
        'source_sha256':own,'revalidation_amendment':value['historical_P15_literal']}.items():
        typed_equal(passed[key],item,'actual P15 pass '+key)
    require(type(own) is dict and len(own)==9,'historical P15 own9')
    rows=value['historical_concrete_fix_relocations'][50:54];relocated={r['original']['path']:r for r in rows}
    require(len(relocated)==4,'actual P15 frozen Source4')
    freeze=strict_json(read_spec(value['historical_P15_final_source_freeze'],root))
    typed_equal(freeze['own_sources_sha256'],own,'actual frozen P15 own9')
    for row,source in zip(rows,freeze['source_copies'],strict=True):
        typed_equal(row['original'],source['current'],'actual P15 current descriptor')
        typed_equal(row['frozen'],source['frozen'],'actual P15 frozen descriptor')
        typed_equal(row['original']['sha256'],own[row['original']['path']],'P15 historical Source4 pin')
        read_spec(row['frozen'],root)
    for name,digest in own.items():
        if name not in relocated:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'unchanged historical P15 own5')
    for spec in passed['concrete_fix']:
        if spec['path'] in relocated:
            typed_equal(spec,relocated[spec['path']]['original'],'historical P15 concrete Source4')
            read_spec(relocated[spec['path']]['frozen'],root)
        else:read_spec(spec,root)
    for spec in value['historical_P15_Native4']:read_spec(spec,root)
    read_spec(value['historical_P15_collector_module'],root)
    costs=value['historical_P15_actual_paid_costs']
    require(type(costs) is list and len(costs)==1,'one actual paid P15 Ruff')
    typed_equal(frozen['debits'][92:93],costs,'actual index92 once debit')
    call=frozen['e360_calls']['e360-generous-p15-ruff'];validate_saved_p15_record('call',call)
    records={'call':call}
    for name,key in P15_SAVED_RECORD_KEYS.items():
        document=strict_json(read_spec(value[key],root));validate_saved_p15_record(name,document);records[name]=document
    cost,actual,request=records['cost'],records['actual'],records['request']
    paid_request_identity(costs[0],cost,actual,frozen,root=root)
    typed_equal(request['source_sha256'],own,'P15 request frozen own9')
    typed_equal(cost['source_sha256'],own,'P15 cost frozen own9')
    typed_equal(strict_json(records['trace']['chunks'][0]['result']['output'].encode('utf8')),records['summary'],'P15 opaque actual capture summary')
    typed_equal(records['closure']['closed_ledger93'],value['historical_P15_closed_ledger93'],'P15 immutable closure93 pointer')
    typed_equal(records['decision']['closed_ledger93'],value['historical_P15_closed_ledger93'],'P15 immutable decision93 pointer')
    typed_equal(records['decision']['original_execution_authority'],False,'P15 decision direct original authority false')
    historical_approval=strict_json(read_spec(value['historical_P15_human_authorization'],root))
    typed_equal(old['original_execution_authority'],False,'P15 literal original authority false')
    typed_equal(historical_approval['original_execution_authority'],False,'P15 human original authority false')
    for name in ('stdout','stderr'):historical_p12_stream_bytes(value['historical_P15_ruff_'+name],root)
    typed_equal(frozen['generated_shared_cumulative_used_s'],584.9690156,'P15 actual shared used')
    typed_equal(frozen['e360_generated_local_used_s'],474.012,'P15 actual local used')
    return frozen


def validate_completed_p15_pass(pass_id,passed,debits,*,root=ROOT):
    frozen=completed_historical_p15_proof(root)
    typed_equal(pass_id,'e360-generous-p15','completed P15 only')
    typed_equal(passed,frozen['e360_passes'][pass_id],'whole historical P15 pass')
    typed_equal(debits[:93],frozen['debits'],'whole93 ordered prefix')


def p16_recovery_event(root=ROOT):
    value=p16_recovery_amendment(root)
    return {'schema':'flylab.m3.e360.p16-recovery-activation.v1','revalidation_amendment':file_spec(P16_RECOVERY_AMENDMENT,root),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited93':value['inherited93'],
        'previous_revalidation_amendment':value['previous_revalidation_amendment'],
        'manual_generated_pass_id':'e360-generous-p16','new_generated_grant_s':0,'human_authorization':value['existing_human_authorization']}


def p16_recovery_activation_allowed(live,*,lock_absent,root=ROOT):
    value=p16_recovery_amendment(root);frozen=completed_historical_p15_proof(root,value=value)
    require(lock_absent is True and not any(key in live for key in P16_RECOVERY_FIELDS),'once P16 ownership0')
    typed_equal(live,frozen,'activation exact whole93')
    budget=current_budget_authority(root,live=live)
    require(budget['suite_recovery_active'] and budget['generated_shared_limit_s']-live['generated_shared_cumulative_used_s']>=798
        and budget['generated_local_limit_s']-live['e360_generated_local_used_s']>=798,'whole798 existing residual only')
    return True


def p16_recovery_delta_valid(before,after,*,root=ROOT):
    require(set(after)==set(before)|set(P16_RECOVERY_FIELDS),'only new P16 pair2')
    for key,item in before.items():typed_equal(after[key],item,'whole93 unchanged '+key)
    typed_equal(after[P16_RECOVERY_FIELDS[0]],p16_recovery_event(root),'exact P16 event8')
    typed_equal(after[P16_RECOVERY_FIELDS[1]],file_spec(P16_RECOVERY_AMENDMENT,root),'literal108629')
    return True


def historical_p16_scope_authority(root=ROOT,*,live=None):
    """Only current P16; immutable completed P15 is a separate reader."""
    if live is None:live=strict_json(bounded_metadata_bytes(registration(root)['shared_ledger_path'],root))
    present=[key in live for key in P16_RECOVERY_FIELDS]
    require(all(present) or not any(present),'complete current P16 pair')
    if not all(present):return False
    value=p16_recovery_amendment(root);frozen=completed_historical_p15_proof(root,value=value)
    require(current_budget_authority(root,live=live)['suite_recovery_active'],'existing P9 budget')
    require(set(frozen)|set(P16_RECOVERY_FIELDS)<=set(live),'whole93 fields plus pair2')
    typed_equal(live[P16_RECOVERY_FIELDS[0]],p16_recovery_event(root),'current P16 event8')
    typed_equal(live[P16_RECOVERY_FIELDS[1]],file_spec(P16_RECOVERY_AMENDMENT,root),'current P16 literal')
    typed_equal(live['debits'][:93],frozen['debits'],'full93 ordered prefix')
    for key in (*value['preserved_activation_fields'],*value['preserved_resource_grant_fields']):typed_equal(live[key],frozen[key],'old26 grants2 '+key)
    for key in ('e360_calls','e360_passes'):
        for identity,item in frozen[key].items():typed_equal(live[key][identity],item,'whole93 history '+key)
    return True


def validate_p16_pass(pass_id,passed,debits,*,root=ROOT):
    value=p16_recovery_amendment(root);frozen=completed_historical_p15_proof(root,value=value)
    typed_equal(pass_id,'e360-generous-p16','one manual P16 no future/original/retry')
    require(current_scope_authority(root),'P16 actual activation')
    require(passed['mode']=='generated' and passed['modified'] is True,'modified generated P16')
    typed_equal(passed['actual_failure'],value['actual_predecessor_failure'],'explicit paid P15 predecessor')
    typed_equal(passed['predecessor_failed_cap_s'],3,'actual predecessor failed cap3')
    typed_equal(debits[:93],frozen['debits'],'all historical93 retained')
    require(sum(row['descriptor']==passed['actual_failure'] for row in debits)==1,'actual P15 paid failure exactly once')
    typed_equal(passed['revalidation_amendment'],file_spec(P16_RECOVERY_AMENDMENT,root),'P16 direct authority')
    require(passed['source_sha256']!=frozen['e360_passes']['e360-generous-p15']['source_sha256'],'fresh P16 Source9')
    typed_equal(passed['source_sha256'],own_sources(root),'current exact own9')
    require(sum(spec==value['concrete_correction_snapshot'] for spec in passed['concrete_fix'])==1,'reviewed P16 correction')
    for name in (*OWNED,BASE+'/parent-e360-root-account.ps1',BASE+'/parent-e360-root-metadata-helper.py',
        value['ruff_capture_policy']['function_only_source_path'],*value['prospective_native_source_paths'],
        value['ROOT_genuine_collector_policy']['canonical_future_source_path']):
        require(sum(spec==file_spec(name,root) for spec in passed['concrete_fix'])==1,'current P16 concrete leaf '+name)
    for spec in passed['concrete_fix']:read_spec(spec,root)

def p16_collector_proof(trace_spec,request,request_spec,seconds,exit_code,*,root=ROOT):
    """Replay immutable genuine transport metadata; never sample a new clock."""
    trace=strict_json(read_spec(trace_spec,root))
    typed_equal(trace['schema'],'flylab.m3.e360.p16.ROOT-genuine-collector-state.v1','collector schema')
    typed_equal(request['pass_id'],'e360-generous-p16','collector current pass only')
    for key in ('pass_id','call_id','call_role'):
        typed_equal(trace[key],request[key],'collector admitted '+key)
    typed_equal(trace['request'],request_spec,'collector exact admitted request')
    typed_equal(trace['collector_source'],request['collector_source'],'collector request source pin')
    typed_equal(trace['collector_source'],file_spec(BASE+'/parent-e360-p16-ROOT-genuine-collector.txt',root),'collector current source')
    typed_equal(trace['state_key'],'e360:p16:ROOT-genuine:'+request['call_id'],'unique collector role state')
    typed_equal(trace_spec['path'],BASE+'/parent-'+request['call_id']+'-ROOT-collector-trace.json','canonical trace publication')
    typed_equal(trace['next_role_allowed'],False,'collector cannot admit another role')
    typed_equal(trace['collection_status'],'FINAL_RECEIVED','first genuine FINAL collected independently of sticky error')
    typed_equal(trace['collection_stopped'],False,'no stopped transport may invent a later FINAL')
    typed_equal(trace['transport_error'],None,'genuine FINAL without a transport exception')
    caps={'ruff':3,'scanner':20,'fixture_build':25,'fixture_worker':55,'fixture_seal':15,
        'suite_build':30,'suite_worker':600,'suite_seal':20,'qualification':30}
    typed_equal(trace['cost_name'],request['call_id'],'collector once cost name')
    typed_equal(trace['cap_s'],caps[request['call_role']],'immutable role cap')
    require(type(trace['max_output_tokens']) is int and trace['max_output_tokens']>=1000,'complete output request bound')
    typed_equal(trace['exec_args'],{'cmd':"pwsh -NoProfile -File '"+BASE+
        "/parent-e360-p16-ROOT-dispatch-adapter.ps1' -CallRole '"+request['call_role']+"'",
        'workdir':'C:/Users/ckthd/Dev/Projects/fly_lab'},'actual ROOT adopted collector dispatch body')
    dispatch=trace['dispatch_ms'];chunks=trace['chunks']
    require(type(dispatch) is int and dispatch>0 and type(chunks) is list and chunks,'genuine dispatch and chunks')
    last=dispatch;known=None;first=None;first_exit=None;complete=True
    for chunk in chunks:
        received=chunk['received_ms'];result=chunk['result']
        require(type(received) is int,'original integer receive clock')
        if received<last:
            require(trace['phase']=='ERROR_HOLD' and type(trace['first_error']) is dict,'clock contradiction retained first error')
            complete=False
        require(first is None,'no result after first genuine FINAL')
        last=received
        require(type(result) is dict,'contradictory transport cannot acquire a later FINAL')
        has_session=result.get('session_id') is not None
        has_exit=result.get('exit_code') is not None
        session=result.get('session_id');returned=result.get('exit_code')
        if has_session and type(session) is int and session>0:
            if known is None:known=session
            require(known==session,'changed session cannot acquire a later FINAL')
        if has_exit and type(returned) is int and not has_session:
            first,first_exit=received,returned
        else:require(has_session and type(session) is int and session>0 and not has_exit,
                     'contradictory result shape cannot acquire a later FINAL')
        output=result.get('output')
        if (type(output) is not str or result.get('truncated') is True or result.get('output_truncated') is True
            or re.search(r'Warning: truncated output|tokens truncated|output was truncated',output or '')):complete=False
    require(first is not None,'lost FINAL remains pending and cannot invent a debit')
    typed_equal(trace['session_id'],known,'first known tool session retained')
    typed_equal(trace['first_final_ms'],first,'immutable first FINAL sample')
    typed_equal(trace['exit_code'],first_exit,'immutable first FINAL exit')
    require(type(seconds) in (int,float) and not isinstance(seconds,bool) and math.isfinite(seconds) and seconds>0,'genuine positive full return')
    elapsed=(first-dispatch)/1000.0
    require(type(trace['full_return_s']) in (int,float) and math.isfinite(trace['full_return_s'])
            and trace['full_return_s']>0,'saved FINAL duration must be a finite positive JSON number')
    require(trace['full_return_s']==elapsed and seconds==elapsed,'exact dispatch to FINAL full return')
    typed_equal(exit_code,first_exit,'actual first FINAL exit')
    require(trace['phase'] in ('FINAL_RECEIVED','ERROR_HOLD'),'closed collector state')
    if trace['phase']=='ERROR_HOLD':
        require(type(trace['first_error']) is dict,'retain first collector failure')
        complete=False
    else:
        typed_equal(trace['first_error'],None,'complete collector has no first error')
        typed_equal(trace['publication'],'REQUIRED','Native publication follows saved FINAL')
    if not any(type(c['result']) is dict and type(c['result'].get('output')) is str
        and c['result']['output'].strip() for c in chunks):complete=False
    if not complete:require(trace['phase']=='ERROR_HOLD' and type(trace['first_error']) is dict,
                            'every collector failure retains the sticky original error')
    return {'valid':complete,'trace':trace_spec,'full_return_s':elapsed,'exit_code':first_exit}




P16_SAVED_RECORD_KEYS={'cost':'historical_P16_ruff_full_return','actual':'historical_P16_ruff_actual_tool_return','request':'historical_P16_ruff_actual_request','summary':'historical_P16_ruff_genuine_capture_return','manifest':'historical_P16_ruff_capture_manifest','trace':'historical_P16_ruff_genuine_collector_trace','closure':'historical_P16_independent_owned_closure','decision':'historical_P16_closed_decision'}
P16_SAVED_RECORD_LITERALS={
    'cost':r'''{"schema":"flylab.m3.e360.administrative-full-return.v1","mode":"generated","charged_once":true,"cost_id":"e360-generous-p16-ruff","actual_outer_full_return_s":1.804,"actual_tool_return":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-actual.json","bytes":1121,"sha256":"7fe34df5be1f13e0f09172c51b901d50965f875a33493a9bde0b14ea22cbe2a9"},"actual_returned_exit_code":1,"original_array_reads":0,"GPU_contexts":0,"resource_extension":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-approved-generous-registration-v2.json","bytes":5461,"sha256":"ea5afdffb09f6a118a18e3039546a911bbc9ed1465cafb6a1fe472ce3471862b"},"revalidation_amendment":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p16-recovery-amendment.json","bytes":108629,"sha256":"8733aaff9b8dbaa1f058661181a8a128b9eb948ef231c2b56616685478df2fe2"},"call_id":"e360-generous-p16-ruff","pass_id":"e360-generous-p16","call_role":"ruff","request":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-final-request.json","bytes":4362,"sha256":"679d7e8d84164b903826553716035ad4a3176d70dca029c3c3015cdf9ee25927"},"source_sha256":{"tools/verify_c_banc_m3_candidate_body_handoff.py":"5b43ee4ee34a8270ba123d11e99e9820b6f2d25644c027bc46a86ab9b6f7532a","verification/banc-natural-gait-m3-20260919/parent-e267-supervise.py":"1d8682505f8060d3d8bcb50508811b9c4da531bdbf07c96200065bcc3324df8d","verification/banc-natural-gait-m3-20260919/parent-e360-root-metadata-helper.py":"74d6776bc4ece9b4ac4b401727b16cdf3f8363048198cc896e684428468d8a15","tests/test_c_banc_source_closure_diagnostic.py":"ad9defe054f0f486b5308e660a33ceab26d8ff2c293e24ae6f72ee0c794cc852","tests/test_c_banc_candidate_body_handoff.py":"93476a2cb72e5bf7585d11ade9f76c207967eff8aadaf918198219bc4379cee6","verification/banc-natural-gait-m3-20260919/parent-e346-supervise-identities.py":"cf7b04731ed4ea8bb626047b577ae5b5b1e31cccd093415d450c839e998b52cd","tools/verify_c_banc_m3_source_closure_diagnostic.py":"0102d069ac3d606e099f28d1754e19e598cd901ad7cb52e226ff20fa18c35e11","tools/verify_c_banc_m3_startup_baseline.py":"4e853f8c4fae105102625816f3075d3b4b7d3ac7312506dd7b3e392e00607342","verification/banc-natural-gait-m3-20260919/parent-e360-root-account.ps1":"67320de0c0f44563f6a7692e720d19300428f44abd08fdbca0a52862b3295ef2"},"expected_output":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-output.json","bytes":1773,"sha256":"db4873e2ea309ac42a7b5862ea876d985111512c9aee70b75198fa64cb46a6c8"},"internal_completion":null,"collector_trace":{"sha256":"47277342e0f0e60459cb44ce5b2adb97beddfe7ed35c6dce44ef5070cc37671b","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-ROOT-collector-trace.json","bytes":3176},"collector_proof_valid":true,"capture_return":{"sha256":"7f6cb1a249d6763097568fea1cad95eb0bd74dc5a24457ba7d434b573c34a9f0","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-genuine-capture-return.json","bytes":1476},"capture_proof_errors":[],"ruff_exit_code":1,"capture_complete":true,"capture_publication_verified":true,"capture_owned_state":"exit_observed","capture_process_identity":{"process_id":10004,"start_utc_ticks":639268195783398392},"capture_root_tool_exit_required":1,"capture_first_failure":{"slot":"ruff_exit_nonzero","kind":"ruff_exit","observed_s":0.5261041},"internal_completion_valid":true}
''',
    'actual':r'''{"actual_returned_exit_code":1,"actual_outer_full_return_s":1.804,"measurement":"ROOT external helper dispatch through actual final return","original_array_reads":0,"GPU_contexts":0,"resource_extension":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-approved-generous-registration-v2.json","bytes":5461,"sha256":"ea5afdffb09f6a118a18e3039546a911bbc9ed1465cafb6a1fe472ce3471862b"},"revalidation_amendment":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p16-recovery-amendment.json","bytes":108629,"sha256":"8733aaff9b8dbaa1f058661181a8a128b9eb948ef231c2b56616685478df2fe2"},"request":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-final-request.json","bytes":4362,"sha256":"679d7e8d84164b903826553716035ad4a3176d70dca029c3c3015cdf9ee25927"},"call_id":"e360-generous-p16-ruff","pass_id":"e360-generous-p16","collector_trace":{"sha256":"47277342e0f0e60459cb44ce5b2adb97beddfe7ed35c6dce44ef5070cc37671b","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-ROOT-collector-trace.json","bytes":3176},"collector_proof_valid":true}
''',
    'request':r'''{"call_id":"e360-generous-p16-ruff","pass_id":"e360-generous-p16","call_role":"ruff","mode":"generated","completion":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-completion.json","action":"ruff","expected_output":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-output.json","actual_failure":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p15-ruff-full-return.json","bytes":3422,"sha256":"1bf44c742b4f83ca727201b814457ee0443aaf03b07df830369e8b2aa47e8960"},"concrete_fix":[{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p16-ROOT-design-reflection.json","bytes":2416,"sha256":"a04e790d4354bd10fc6526f89c078da2d796a27dec3e821c76fb74dd3c983647"},{"path":"tools/verify_c_banc_m3_source_closure_diagnostic.py","bytes":486616,"sha256":"0102d069ac3d606e099f28d1754e19e598cd901ad7cb52e226ff20fa18c35e11"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-root-metadata-helper.py","bytes":14959,"sha256":"74d6776bc4ece9b4ac4b401727b16cdf3f8363048198cc896e684428468d8a15"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-root-account.ps1","bytes":421246,"sha256":"67320de0c0f44563f6a7692e720d19300428f44abd08fdbca0a52862b3295ef2"},{"path":"tests/test_c_banc_source_closure_diagnostic.py","bytes":347603,"sha256":"ad9defe054f0f486b5308e660a33ceab26d8ff2c293e24ae6f72ee0c794cc852"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p16-root-final-freeze.json","bytes":85962,"sha256":"9390ee4e57501cd5a5bbd54ba3c28dd0700c5772c0594912ea7102277c431d62"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p16-capture-proposed.ps1","bytes":29687,"sha256":"3efcd4b3c6639c837a3973a6d3db30c52928f8367bed425a491dc78f6875d2ef"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p16-prepare-native.ps1","bytes":4237,"sha256":"73ce847a5934f9f6ff87b1f5a5d4ba8364146c84cc4090632ef7c1800a629e1c"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p16-bookkeeping-native.ps1","bytes":10975,"sha256":"1643b60ae650a08d052fad99d376e8fe48b8e43533dc6fcdf855476ed616862d"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p16-ROOT-dispatch-adapter.ps1","bytes":8694,"sha256":"da7048a44370d9e165c6afdc4bca467aa94d25e0b73cdf6b44d617b7ad869557"},{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p16-ROOT-genuine-collector.txt","bytes":7348,"sha256":"f27e63c042c5f6e56d334a320ed0a31feed7bab66c567670b1af59ebbc30645e"}],"collector_source":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p16-ROOT-genuine-collector.txt","bytes":7348,"sha256":"f27e63c042c5f6e56d334a320ed0a31feed7bab66c567670b1af59ebbc30645e"},"schema":"flylab.m3.e360.helper-request.v1","source_sha256":{"tools/verify_c_banc_m3_candidate_body_handoff.py":"5b43ee4ee34a8270ba123d11e99e9820b6f2d25644c027bc46a86ab9b6f7532a","verification/banc-natural-gait-m3-20260919/parent-e267-supervise.py":"1d8682505f8060d3d8bcb50508811b9c4da531bdbf07c96200065bcc3324df8d","verification/banc-natural-gait-m3-20260919/parent-e360-root-metadata-helper.py":"74d6776bc4ece9b4ac4b401727b16cdf3f8363048198cc896e684428468d8a15","tests/test_c_banc_source_closure_diagnostic.py":"ad9defe054f0f486b5308e660a33ceab26d8ff2c293e24ae6f72ee0c794cc852","tests/test_c_banc_candidate_body_handoff.py":"93476a2cb72e5bf7585d11ade9f76c207967eff8aadaf918198219bc4379cee6","verification/banc-natural-gait-m3-20260919/parent-e346-supervise-identities.py":"cf7b04731ed4ea8bb626047b577ae5b5b1e31cccd093415d450c839e998b52cd","tools/verify_c_banc_m3_source_closure_diagnostic.py":"0102d069ac3d606e099f28d1754e19e598cd901ad7cb52e226ff20fa18c35e11","tools/verify_c_banc_m3_startup_baseline.py":"4e853f8c4fae105102625816f3075d3b4b7d3ac7312506dd7b3e392e00607342","verification/banc-natural-gait-m3-20260919/parent-e360-root-account.ps1":"67320de0c0f44563f6a7692e720d19300428f44abd08fdbca0a52862b3295ef2"},"resource_extension":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-approved-generous-registration-v2.json","bytes":5461,"sha256":"ea5afdffb09f6a118a18e3039546a911bbc9ed1465cafb6a1fe472ce3471862b"},"revalidation_amendment":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p16-recovery-amendment.json","bytes":108629,"sha256":"8733aaff9b8dbaa1f058661181a8a128b9eb948ef231c2b56616685478df2fe2"},"closure_report":null}
''',
    'summary':r'''{"call_id":"e360-generous-p16-ruff","capture_complete":true,"publication_verified":true,"manifest_descriptor":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-output.json","sha256":"db4873e2ea309ac42a7b5862ea876d985111512c9aee70b75198fa64cb46a6c8","bytes":1773},"ruff_exit_code":1,"exit_observed":true,"owned_state":"exit_observed","admission_complete":true,"start_attempted":true,"started_observed":true,"start_known_failed":false,"process_identity":{"process_id":10004,"start_utc_ticks":639268195783398392},"streams":[{"stream":"stdout","observed_bytes":4190,"retained_bytes":4190,"eof_observed":true,"read_fault":false,"overflow":false,"readback_verified":true,"descriptor":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-stdout.txt","sha256":"1d4ac8043fb41b2a3a641f414a657c23dc93ee1e65b2ca4d39b9310dfa02d199","bytes":4190}},{"stream":"stderr","observed_bytes":0,"retained_bytes":0,"eof_observed":true,"read_fault":false,"overflow":false,"readback_verified":true,"descriptor":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","bytes":0}}],"first_failure":{"slot":"ruff_exit_nonzero","kind":"ruff_exit","observed_s":0.5261041},"wrapper_error_slots":[],"internal_elapsed_s":0.5633434,"external_full_return_s":null,"external_cap_s":3,"runtime_qualified":false,"root_tool_exit_required":1}
''',
    'manifest':r'''{"schema":"flylab.m3.e360.p16-ruff-capture.v1","call_id":"e360-generous-p16-ruff","pass_id":"e360-generous-p16","entry_utc":"2026-10-05T17:52:57.8922163Z","internal_observed_s":0.5458665,"external_cap_s":3,"soft_deadline_s":2,"cleanup_deadline_s":2.5,"process_identity":{"process_id":10004,"start_utc_ticks":639268195783398392},"start_attempted":true,"started_observed":true,"start_known_failed":false,"owned_state":"exit_observed","exit_observed":true,"ruff_exit_code":1,"kill_attempted":false,"kill_requested":false,"timeout_observed":false,"streams":[{"stream":"stdout","observed_bytes":4190,"retained_bytes":4190,"eof_observed":true,"read_fault":false,"overflow":false,"readback_verified":true,"descriptor":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-stdout.txt","sha256":"1d4ac8043fb41b2a3a641f414a657c23dc93ee1e65b2ca4d39b9310dfa02d199","bytes":4190}},{"stream":"stderr","observed_bytes":0,"retained_bytes":0,"eof_observed":true,"read_fault":false,"overflow":false,"readback_verified":true,"descriptor":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","bytes":0}}],"first_failure":{"slot":"ruff_exit_nonzero","kind":"ruff_exit","observed_s":0.5261041},"wrapper_errors":{},"capture_complete":true,"request":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-final-request.json","bytes":4362,"sha256":"679d7e8d84164b903826553716035ad4a3176d70dca029c3c3015cdf9ee25927"},"revalidation_amendment":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p16-recovery-amendment.json","sha256":"8733aaff9b8dbaa1f058661181a8a128b9eb948ef231c2b56616685478df2fe2","bytes":108629}}''',
    'trace':r'''{
  "schema": "flylab.m3.e360.p16.ROOT-genuine-collector-state.v1",
  "pass_id": "e360-generous-p16",
  "call_id": "e360-generous-p16-ruff",
  "call_role": "ruff",
  "cost_name": "e360-generous-p16-ruff",
  "cap_s": 3,
  "state_key": "e360:p16:ROOT-genuine:e360-generous-p16-ruff",
  "request": {
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-final-request.json",
    "bytes": 4362,
    "sha256": "679d7e8d84164b903826553716035ad4a3176d70dca029c3c3015cdf9ee25927"
  },
  "collector_source": {
    "bytes": 7348,
    "sha256": "f27e63c042c5f6e56d334a320ed0a31feed7bab66c567670b1af59ebbc30645e",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p16-ROOT-genuine-collector.txt"
  },
  "exec_args": {
    "cmd": "pwsh -NoProfile -File 'verification/banc-natural-gait-m3-20260919/parent-e360-p16-ROOT-dispatch-adapter.ps1' -CallRole 'ruff'",
    "workdir": "C:/Users/ckthd/Dev/Projects/fly_lab"
  },
  "max_output_tokens": 10000,
  "phase": "FINAL_RECEIVED",
  "collection_status": "FINAL_RECEIVED",
  "collection_stopped": false,
  "transport_error": null,
  "dispatch_ms": 1791222776778,
  "first_final_ms": 1791222778582,
  "full_return_s": 1.804,
  "session_id": null,
  "chunks": [
    {
      "received_ms": 1791222778582,
      "result": {
        "chunk_id": "2881e8",
        "wall_time_seconds": 1.4555954,
        "exit_code": 1,
        "original_token_count": 370,
        "output": "{\"call_id\":\"e360-generous-p16-ruff\",\"capture_complete\":true,\"publication_verified\":true,\"manifest_descriptor\":{\"path\":\"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-output.json\",\"sha256\":\"db4873e2ea309ac42a7b5862ea876d985111512c9aee70b75198fa64cb46a6c8\",\"bytes\":1773},\"ruff_exit_code\":1,\"exit_observed\":true,\"owned_state\":\"exit_observed\",\"admission_complete\":true,\"start_attempted\":true,\"started_observed\":true,\"start_known_failed\":false,\"process_identity\":{\"process_id\":10004,\"start_utc_ticks\":639268195783398392},\"streams\":[{\"stream\":\"stdout\",\"observed_bytes\":4190,\"retained_bytes\":4190,\"eof_observed\":true,\"read_fault\":false,\"overflow\":false,\"readback_verified\":true,\"descriptor\":{\"path\":\"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-stdout.txt\",\"sha256\":\"1d4ac8043fb41b2a3a641f414a657c23dc93ee1e65b2ca4d39b9310dfa02d199\",\"bytes\":4190}},{\"stream\":\"stderr\",\"observed_bytes\":0,\"retained_bytes\":0,\"eof_observed\":true,\"read_fault\":false,\"overflow\":false,\"readback_verified\":true,\"descriptor\":{\"path\":\"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-stderr.txt\",\"sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\",\"bytes\":0}}],\"first_failure\":{\"slot\":\"ruff_exit_nonzero\",\"kind\":\"ruff_exit\",\"observed_s\":0.5261041},\"wrapper_error_slots\":[],\"internal_elapsed_s\":0.5633434,\"external_full_return_s\":null,\"external_cap_s\":3,\"runtime_qualified\":false,\"root_tool_exit_required\":1}\r\n"
      }
    }
  ],
  "exit_code": 1,
  "first_error": null,
  "publication": "REQUIRED",
  "next_role_allowed": false
}
''',
    'closure':r'''{
  "schema": "flylab.m3.e360.p16.ROOT-independent-owned-closure.v1",
  "actual_actor": "ROOT",
  "date": "2026-10-06",
  "closed_ledger94": {
    "sha256": "58fe659c2ff23ddef0371730e2934a20e753752295d32c397c4c43eba92deb95",
    "bytes": 139634,
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p16-closed-ledger94.json"
  },
  "all_old93_once_debits_preserved": true,
  "old_calls43_passes15_preserved": true,
  "whole94_calls44_passes16": true,
  "new_paid_calls": 1,
  "new_paid_seconds": 1.804,
  "pending_actuals": 0,
  "reservations": 0,
  "blocking_holds": 0,
  "unreturned_calls": 0,
  "original_jobs": 0,
  "original_reserved_s": 0,
  "known_owned_ruff": {
    "expected_start_utc_ticks": 639268195783398392,
    "exit_observed": true,
    "both_streams_closed": true,
    "process_id": 10004,
    "matching_owned_live": false,
    "observed_start_utc_ticks": null,
    "no_process_termination_performed": true
  },
  "actual_as_executed_freeze": {
    "sha256": "9390ee4e57501cd5a5bbd54ba3c28dd0700c5772c0594912ea7102277c431d62",
    "bytes": 85962,
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p16-root-final-freeze.json"
  },
  "current_own9_preserved": true,
  "protected322_preserved": true,
  "old_Native19_and_currentNative5_preserved": true,
  "original_IO_GPU_body_neural_UI": false,
  "runtime_qualified": false,
  "F1_F8_complete": false
}
''',
    'decision':r'''{
  "schema": "flylab.m3.e360.p16.ROOT-closed-decision.v1",
  "actual_actor": "ROOT",
  "date": "2026-10-06",
  "status": "CLOSED_FIRST_PAID_RUFF_FAILURE_WITH_REVIEWED_OFFACTIVE_R1",
  "latest_human_quote": "제안해준 다음 작업 승인할 게 진행해줘.",
  "approved_literal": {
    "bytes": 108629,
    "sha256": "8733aaff9b8dbaa1f058661181a8a128b9eb948ef231c2b56616685478df2fe2",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p16-recovery-amendment.json"
  },
  "human_authorization": {
    "bytes": 4273,
    "sha256": "8d5a8a2b03445e7e32162634fc55f29e9448b6ab600de1bb1ddf8317ecf4adb4",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p16-human-authorization.json"
  },
  "actual_as_executed_freeze": {
    "bytes": 85962,
    "sha256": "9390ee4e57501cd5a5bbd54ba3c28dd0700c5772c0594912ea7102277c431d62",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p16-root-final-freeze.json"
  },
  "actual_activation": {
    "bytes": 2292,
    "sha256": "a89309a08cfa8ce8cfe7cec73257ce464ae7d6c7910e4cdd4a9d47b3743467f0",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p16-ROOT-activation-tool-raw-evidence.json"
  },
  "activation_attempts": 1,
  "activation_success": 1,
  "actual_callbegin_and_once_account_success": true,
  "actual_Ruff_cost": {
    "bytes": 3414,
    "sha256": "b198b53e671b1d2577d9539f8a26a5d546a454d2959070400c7bf6fa35d5b211",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-full-return.json"
  },
  "actual_Ruff_stdout": {
    "bytes": 4190,
    "sha256": "1d4ac8043fb41b2a3a641f414a657c23dc93ee1e65b2ca4d39b9310dfa02d199",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-stdout.txt"
  },
  "actual_Ruff_exit_code": 1,
  "new_paid_roles": 1,
  "new_paid_seconds": 1.804,
  "capture_and_collector_proof_valid": true,
  "Ruff_full_return_within_cap3": true,
  "first_failure": "S102 one new frozen loader exec and F821 seven undefined current/value references in copied P14 fixture block",
  "ROOT_FIRST_SECOND_missed_actual_lint_defects": true,
  "closed_ledger94": {
    "bytes": 139634,
    "sha256": "58fe659c2ff23ddef0371730e2934a20e753752295d32c397c4c43eba92deb95",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p16-closed-ledger94.json"
  },
  "independent_owned_closure": {
    "bytes": 1442,
    "sha256": "ed300dcfd056ca5da80cef0c06eb535d257f0812d1a5666ba9c2039c70b38640",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p16-ROOT-independent-owned-closure.json"
  },
  "old93_costs_calls43_passes15_typed_preserved": true,
  "whole94_calls44_passes16": true,
  "failed_cap_s": 3,
  "next_index": 1,
  "shared_used_s": 586.7730156,
  "shared_limit_s": 1400,
  "shared_remaining_s": 813.2269844,
  "local_used_s": 475.816,
  "local_limit_s": 1289,
  "local_remaining_s": 813.184,
  "offset": 42,
  "additional_grant": 0,
  "original_cumulative": 10020.115960899504,
  "original_stage": 126.361,
  "E360_original_used": 0,
  "pending": 0,
  "reservations": 0,
  "holds": 0,
  "unreturned": 0,
  "roles_after_first_failure": 0,
  "scanner_fixture_suite_Q": 0,
  "actual_test_phases": 0,
  "actual_Native16_test_probe": 0,
  "original_GPU_body_neural_UI": 0,
  "actual_SAME_Sol_ONE_R1": {
    "bytes": 10538,
    "sha256": "50423c62c7c5fb8b88914454d89ff6be8ebb54ce9fab37bbb5e1b34d0d5e0cf1",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p16-sol-R1-handoff.json"
  },
  "ROOT_SECOND_R1": {
    "bytes": 1679,
    "sha256": "97ee001ce2a27b3321aaab19a1d9f0f975061e0f2f2c2ac6dbd2b7bfc1774225",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p16-ROOT-SECOND-R1-Source-review.json"
  },
  "R1_OFF_ACTIVE_selected_tests": {
    "bytes": 346811,
    "sha256": "72bb8e3ccc2b3c63e9e95baa525e72ab8b3265c8b1ec9400666d319adf354179",
    "path": "verification/banc-natural-gait-m3-20260919/parent-e360-p16-sol-R1-source-tests.py"
  },
  "R1_runtime_retested": false,
  "R1_not_applied_to_execution_history": true,
  "P17_authorized": false,
  "original_execution_authority": false,
  "automatic_retries": 0,
  "runtime_qualified": false,
  "natural_gait": false,
  "F1_F8_complete": false,
  "goal_complete": false,
  "goal_API_status": "actual active on this resumed turn; no reset/new goal/complete or fabricated blocked",
  "prior_P15_failures_preserved": true
}
''',
    'call':r'''{"request":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p16-ruff-final-request.json","bytes":4362,"sha256":"679d7e8d84164b903826553716035ad4a3176d70dca029c3c3015cdf9ee25927"},"mode":"generated","pass_id":"e360-generous-p16","future_reserved_s":795,"call_role":"ruff","state":"charged","revalidation_amendment":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-p16-recovery-amendment.json","bytes":108629,"sha256":"8733aaff9b8dbaa1f058661181a8a128b9eb948ef231c2b56616685478df2fe2"},"reserved_s":3,"cost_id":"e360-generous-p16-ruff","attempt_id":null,"sequence_index":0,"resource_extension":{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-approved-generous-registration-v2.json","bytes":5461,"sha256":"ea5afdffb09f6a118a18e3039546a911bbc9ed1465cafb6a1fe472ce3471862b"}}''',
}

@historical_dependencies()
def _p17_recovery_amendment(root=ROOT):
    raw=literal_bytes(P17_RECOVERY_AMENDMENT,root)
    require(len(raw)==P17_RECOVERY_BYTES and sha(raw)==P17_RECOVERY_SHA,'exact approved P17 literal')
    value=strict_json(raw);require(type(value) is dict and len(value)==251,'actual current P17 literal fields251');old=p16_recovery_amendment(root)
    expected={'schema':'flylab.m3.e360.p17-recovery-amendment.v1','manual_generated_pass_id':'e360-generous-p17',
        'predecessor_pass_id':'e360-generous-p16','required_exact_prefix_length':94,'local_cost_offset':42,
        'generated_shared_limit_s':1400,'generated_local_limit_s':1289,'new_generated_grant_s':0,
        'generated_pass_limit':17,'modified_generated_pass_limit':16,'original_pass_limit':1,'total_pass_limit':18,
        'automatic_retries':0,'original_local_limit_s':20,'original_stage_limit_s':300,'original_global_limit_s':10200,
        'original_worker_maximum':1,'original_automatic_retries':0,'source_hash_cycle_allowed':False,
        'whole_fresh_chain_required':True,'primary_total_reserved_s':798,'predecessor_failed_cap_s':3,
        'minimum_current_test_functions':84,'maximum_current_test_functions':84,'minimum_current_test_phases':252,
        'pass17_authorized':True,'pass18_authorized':False,'P16_resume_authorized':False,
        'original_execution_authority':False,'generated_scope_ends_after_Q_debit':True,
        'generated_scope_ends_after_first_closed_failure':True,
        'predecessor_failure_kind':'closed_paid_Ruff_exit1_after_P16_activation'}
    for key,item in expected.items():typed_equal(value[key],item,'compiled P17 '+key)
    typed_equal(value['previous_revalidation_amendment'],file_spec(P16_RECOVERY_AMENDMENT,root),'immutable reader17')
    for key in ('resource_extension','generated_call_order','generated_call_schedule','effective_helper_limits',
        'effective_roles','scanner_limits','checkpoint_policy','successful_suffix_after_s',
        'remaining_generated_administration_reserve_s','fixture_future_administration_reserved_s',
        'suite_future_administration_reserved_s','preserved_test_functions','preserved_resource_grant_fields'):
        typed_equal(value[key],old[key],'preserved finite P16 policy '+key)
    policy=copy.deepcopy(old['fixed_authority_scope_policy'])
    policy['reader_ids'].append('p17_recovery_amendment');policy['max_entries']=18
    typed_equal(value['fixed_authority_scope_policy'],policy,'reader18 exact old17 capacity18')
    typed_equal(value['preserved_activation_fields'],[*old['preserved_activation_fields'],*P16_RECOVERY_FIELDS],'old28')
    typed_equal(value['new_active_fields'],list(P17_RECOVERY_FIELDS),'newpair2')
    typed_equal(value['activation_policy'],{'activation_schema':'flylab.m3.e360.p17-recovery-activation.v1',
        'changed_ledger_fields':[],'new_ledger_fields':list(P17_RECOVERY_FIELDS),'new_grant_object_allowed':False,
        'activation_keys':['schema','revalidation_amendment','resource_extension','inherited94',
        'previous_revalidation_amendment','manual_generated_pass_id','new_generated_grant_s','human_authorization']},'P17 event8')
    rows=value['historical_concrete_fix_relocations']
    require(type(rows) is list and len(rows)==58,'exact relocation58')
    typed_equal(rows[:54],old['historical_concrete_fix_relocations'],'unchanged old54')
    typed_equal([r['pass_id'] for r in rows[54:]],['e360-generous-p16']*4,'actual P16 frozen4')
    for row in rows:
        leaf=row.get('snapshot',row.get('frozen'))
        for key in ('bytes','sha256'):typed_equal(row['original'][key],leaf[key],'same-byte historical Source '+key)
        read_spec(leaf,root)
    for spec in value.values():
        if type(spec) is dict and set(spec)=={'path','bytes','sha256'}:
            historical_p12_stream_bytes(spec,root) if spec['bytes']==0 else read_spec(spec,root)
    for spec in value['current_P17_source_snapshots']:read_spec(spec,root)
    for spec in value['historical_P16_Native4']:read_spec(spec,root)
    typed_equal(value['existing_human_authorization'],CURRENT_HUMAN_AUTONOMY,'stable general human autonomy')
    typed_equal(value['current_human_autonomy'],CURRENT_HUMAN_AUTONOMY,'same current human autonomy')
    approval=strict_json(read_spec(CURRENT_HUMAN_AUTONOMY,root))
    for key,item in {'schema':'flylab.m3.current-human-autonomy.v1',
        'human_quote':'앞으로 내 승인 받지 말고 스스로 판단해서 진행해줘. 개입할 상황이면 내가 알아서 개입할게',
        'current_P17_repair_and_execution_authorized':True,'per_attempt_human_reapproval_required':False,
        'in_scope_repair_and_new_attempt_selection_delegated':True,'fresh_human_quote_per_future_attempt_required':False,
        'new_grant_in_this_bootstrap':0,'no_automatic_identical_redispatch_or_accounting_reset':True}.items():
        typed_equal(approval[key],item,'general delegated authority '+key)
    for key,item in {'per_attempt_human_reapproval_required':False,
        'in_scope_next_corrected_attempt_selection_delegated':True,'current_only_admission_is_not_global_human_authority':True}.items():
        typed_equal(value[key],item,'current attempt versus general authority '+key)
    typed_equal(value['prospective_native_source_paths'],[BASE+'/parent-e360-p17-'+name+'.ps1'
        for name in ('prepare-native','bookkeeping-native','ROOT-dispatch-adapter')],'P17 Native3 canonical paths')
    completed_historical_p16_proof(root,value=value)
    return value


def p17_recovery_amendment(root=ROOT):
    return fixed_reader('p17_recovery_amendment',_p17_recovery_amendment,root)


def validate_saved_p16_record(name,document):
    """Narrow production typed gate over the direct actual record; never defaults a missing field."""
    require(name in P16_SAVED_RECORD_LITERALS,'known actual P16 record')
    expected=strict_json(P16_SAVED_RECORD_LITERALS[name].encode('utf8'))
    typed_equal(document,expected,'actual P16 '+name+' exact typed record')


@historical_dependencies()
def completed_historical_p16_proof(root=ROOT,*,value=None):
    """Immutable completed P16 proof to whole94 END. No current selector/Source4/PID/clock/contract."""
    if value is None:
        raw=literal_bytes(P17_RECOVERY_AMENDMENT,root)
        require(len(raw)==P17_RECOVERY_BYTES and sha(raw)==P17_RECOVERY_SHA,'immutable P17 history literal')
        value=strict_json(raw)
    old=p16_recovery_amendment(root);prior=completed_historical_p15_proof(root,value=old)
    frozen=strict_json(read_spec(value['inherited94'],root))['ledger']
    typed_equal(frozen,strict_json(read_spec(value['historical_P16_closed_ledger94'],root)),'exact immutable whole94')
    typed_equal(prior,strict_json(read_spec(value['historical_P16_inherited93'],root))['ledger'],'exact wholeold93')
    require(len(frozen['debits'])==94 and len(frozen['e360_calls'])==44 and len(frozen['e360_passes'])==16,'94/44/16')
    typed_equal(frozen['debits'][:93],prior['debits'],'all93 ordered costs')
    for key in (*old['preserved_activation_fields'],*old['preserved_resource_grant_fields']):
        typed_equal(frozen[key],prior[key],'unchanged old24/grants2 '+key)
    typed_equal(frozen[P16_RECOVERY_FIELDS[0]],p16_recovery_event(root),'historical P16 event8')
    typed_equal(frozen[P16_RECOVERY_FIELDS[1]],value['historical_P16_literal'],'historical P16 literal')
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(frozen[key][identity],item,'whole94 historical '+key)
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):
        typed_equal(frozen[key],{},'completed ownership0 '+key)
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(frozen[key],0,'original0')
    own=value['historical_P16_frozen_own_sources_sha256'];passed=frozen['e360_passes']['e360-generous-p16']
    for key,item in {'mode':'generated','modified':True,'predecessor_failed_cap_s':3,'failed_cap_s':3,'next_index':1,
        'actual_failure':old['actual_predecessor_failure'],'failed_cost':value['historical_P16_ruff_full_return'],
        'source_sha256':own,'revalidation_amendment':value['historical_P16_literal']}.items():
        typed_equal(passed[key],item,'actual P16 pass '+key)
    require(type(own) is dict and len(own)==9,'historical P16 own9')
    rows=value['historical_concrete_fix_relocations'][54:58];relocated={r['original']['path']:r for r in rows}
    require(len(relocated)==4,'actual P16 frozen Source4')
    freeze=strict_json(read_spec(value['historical_P16_final_source_freeze'],root))
    typed_equal(freeze['own_sources_sha256'],own,'actual frozen P16 own9')
    for row,source in zip(rows,freeze['source_copies'],strict=True):
        typed_equal(row['original'],source['current'],'actual P16 current descriptor')
        typed_equal(row['frozen'],source['frozen'],'actual P16 frozen descriptor')
        typed_equal(row['original']['sha256'],own[row['original']['path']],'P16 historical Source4 pin')
        read_spec(row['frozen'],root)
    for name,digest in own.items():
        if name not in relocated:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'unchanged historical P16 own5')
    for spec in passed['concrete_fix']:
        if spec['path'] in relocated:
            typed_equal(spec,relocated[spec['path']]['original'],'historical P16 concrete Source4')
            read_spec(relocated[spec['path']]['frozen'],root)
        else:read_spec(spec,root)
    for spec in value['historical_P16_Native4']:read_spec(spec,root)
    read_spec(value['historical_P16_collector_module'],root)
    costs=value['historical_P16_actual_paid_costs']
    require(type(costs) is list and len(costs)==1,'one actual paid P16 Ruff')
    typed_equal(frozen['debits'][93:94],costs,'actual index93 once debit')
    call=frozen['e360_calls']['e360-generous-p16-ruff'];validate_saved_p16_record('call',call)
    records={'call':call}
    for name,key in P16_SAVED_RECORD_KEYS.items():
        document=strict_json(read_spec(value[key],root));validate_saved_p16_record(name,document);records[name]=document
    cost,actual,request=records['cost'],records['actual'],records['request']
    paid_request_identity(costs[0],cost,actual,frozen,root=root)
    typed_equal(request['source_sha256'],own,'P16 request frozen own9')
    typed_equal(cost['source_sha256'],own,'P16 cost frozen own9')
    typed_equal(strict_json(records['trace']['chunks'][0]['result']['output'].encode('utf8')),records['summary'],'P16 opaque actual capture summary')
    typed_equal(records['closure']['closed_ledger94'],value['historical_P16_closed_ledger94'],'P16 immutable closure94 pointer')
    typed_equal(records['decision']['closed_ledger94'],value['historical_P16_closed_ledger94'],'P16 immutable decision94 pointer')
    typed_equal(records['decision']['original_execution_authority'],False,'P16 decision direct original authority false')
    historical_approval=strict_json(read_spec(value['historical_P16_human_authorization'],root))
    typed_equal(old['original_execution_authority'],False,'P16 literal original authority false')
    typed_equal(historical_approval['original_execution_authority'],False,'P16 human original authority false')
    for name in ('stdout','stderr'):historical_p12_stream_bytes(value['historical_P16_ruff_'+name],root)
    typed_equal(frozen['generated_shared_cumulative_used_s'],586.7730156,'P16 actual shared used')
    typed_equal(frozen['e360_generated_local_used_s'],475.816,'P16 actual local used')
    return frozen


def validate_completed_p16_pass(pass_id,passed,debits,*,root=ROOT):
    frozen=completed_historical_p16_proof(root)
    typed_equal(pass_id,'e360-generous-p16','completed P16 only')
    typed_equal(passed,frozen['e360_passes'][pass_id],'whole historical P16 pass')
    typed_equal(debits[:94],frozen['debits'],'whole94 ordered prefix')


def p17_recovery_event(root=ROOT):
    value=p17_recovery_amendment(root)
    return {'schema':'flylab.m3.e360.p17-recovery-activation.v1','revalidation_amendment':file_spec(P17_RECOVERY_AMENDMENT,root),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited94':value['inherited94'],
        'previous_revalidation_amendment':value['previous_revalidation_amendment'],
        'manual_generated_pass_id':'e360-generous-p17','new_generated_grant_s':0,'human_authorization':value['existing_human_authorization']}


def p17_recovery_activation_allowed(live,*,lock_absent,root=ROOT):
    value=p17_recovery_amendment(root);frozen=completed_historical_p16_proof(root,value=value)
    require(lock_absent is True and not any(key in live for key in P17_RECOVERY_FIELDS),'once P17 ownership0')
    typed_equal(live,frozen,'activation exact whole94')
    budget=current_budget_authority(root,live=live)
    require(budget['suite_recovery_active'] and budget['generated_shared_limit_s']-live['generated_shared_cumulative_used_s']>=798
        and budget['generated_local_limit_s']-live['e360_generated_local_used_s']>=798,'whole798 existing residual only')
    return True


def p17_recovery_delta_valid(before,after,*,root=ROOT):
    require(set(after)==set(before)|set(P17_RECOVERY_FIELDS),'only new P17 pair2')
    for key,item in before.items():typed_equal(after[key],item,'whole94 unchanged '+key)
    typed_equal(after[P17_RECOVERY_FIELDS[0]],p17_recovery_event(root),'exact P17 event8')
    typed_equal(after[P17_RECOVERY_FIELDS[1]],file_spec(P17_RECOVERY_AMENDMENT,root),'literal121989')
    return True


def _prior_p17_scope_authority(root=ROOT,*,live=None):
    """Only current P17; immutable completed P16 is a separate reader."""
    if live is None:live=strict_json(bounded_metadata_bytes(registration(root)['shared_ledger_path'],root))
    present=[key in live for key in P17_RECOVERY_FIELDS]
    require(all(present) or not any(present),'complete current P17 pair')
    if not all(present):return False
    value=p17_recovery_amendment(root);frozen=completed_historical_p16_proof(root,value=value)
    validate_p17_final_freeze(root)
    require(current_budget_authority(root,live=live)['suite_recovery_active'],'existing P9 budget')
    require(set(live)==set(frozen)|set(P17_RECOVERY_FIELDS),'whole94 fields plus pair2')
    typed_equal(live[P17_RECOVERY_FIELDS[0]],p17_recovery_event(root),'current P17 event8')
    typed_equal(live[P17_RECOVERY_FIELDS[1]],file_spec(P17_RECOVERY_AMENDMENT,root),'current P17 literal')
    typed_equal(live['debits'][:94],frozen['debits'],'full94 ordered prefix')
    for key in (*value['preserved_activation_fields'],*value['preserved_resource_grant_fields']):typed_equal(live[key],frozen[key],'old28 grants2 '+key)
    for key in ('e360_calls','e360_passes'):
        for identity,item in frozen[key].items():typed_equal(live[key][identity],item,'whole94 history '+key)
    return True


def validate_p17_pass(pass_id,passed,debits,*,root=ROOT):
    value=p17_recovery_amendment(root);frozen=completed_historical_p16_proof(root,value=value)
    typed_equal(pass_id,'e360-generous-p17','one manual P17 no future/original/retry')
    require(current_scope_authority(root),'P17 actual activation')
    require(passed['mode']=='generated' and passed['modified'] is True,'modified generated P17')
    typed_equal(passed['actual_failure'],value['actual_predecessor_failure'],'explicit paid P16 predecessor')
    typed_equal(passed['predecessor_failed_cap_s'],3,'actual predecessor failed cap3')
    typed_equal(debits[:94],frozen['debits'],'all historical93 retained')
    require(sum(row['descriptor']==passed['actual_failure'] for row in debits)==1,'actual P16 paid failure exactly once')
    typed_equal(passed['revalidation_amendment'],file_spec(P17_RECOVERY_AMENDMENT,root),'P17 direct authority')
    require(passed['source_sha256']!=frozen['e360_passes']['e360-generous-p16']['source_sha256'],'fresh P17 Source9')
    typed_equal(passed['source_sha256'],own_sources(root),'current exact own9')
    require(sum(spec==value['concrete_correction_snapshot'] for spec in passed['concrete_fix'])==1,'reviewed P17 correction')
    for name in (*OWNED,BASE+'/parent-e360-root-account.ps1',BASE+'/parent-e360-root-metadata-helper.py',
        value['ruff_capture_policy']['function_only_source_path'],*value['prospective_native_source_paths'],
        value['ROOT_genuine_collector_policy']['canonical_future_source_path']):
        require(sum(spec==file_spec(name,root) for spec in passed['concrete_fix'])==1,'current P17 concrete leaf '+name)
    for spec in passed['concrete_fix']:read_spec(spec,root)

def p17_collector_proof(trace_spec,request,request_spec,seconds,exit_code,*,root=ROOT):
    """Replay immutable genuine transport metadata; never sample a new clock."""
    trace=strict_json(read_spec(trace_spec,root))
    typed_equal(trace['schema'],'flylab.m3.e360.p17.ROOT-genuine-collector-state.v1','collector schema')
    typed_equal(request['pass_id'],'e360-generous-p17','collector current pass only')
    for key in ('pass_id','call_id','call_role'):
        typed_equal(trace[key],request[key],'collector admitted '+key)
    typed_equal(trace['request'],request_spec,'collector exact admitted request')
    typed_equal(trace['collector_source'],request['collector_source'],'collector request source pin')
    typed_equal(trace['collector_source'],file_spec(BASE+'/parent-e360-p17-ROOT-genuine-collector.txt',root),'collector current source')
    typed_equal(trace['state_key'],'e360:p17:ROOT-genuine:'+request['call_id'],'unique collector role state')
    typed_equal(trace_spec['path'],BASE+'/parent-'+request['call_id']+'-ROOT-collector-trace.json','canonical trace publication')
    typed_equal(trace['next_role_allowed'],False,'collector cannot admit another role')
    typed_equal(trace['collection_status'],'FINAL_RECEIVED','first genuine FINAL collected independently of sticky error')
    typed_equal(trace['collection_stopped'],False,'no stopped transport may invent a later FINAL')
    typed_equal(trace['transport_error'],None,'genuine FINAL without a transport exception')
    caps={'ruff':3,'scanner':20,'fixture_build':25,'fixture_worker':55,'fixture_seal':16,
        'suite_build':30,'suite_worker':600,'suite_seal':20,'qualification':30}
    typed_equal(trace['cost_name'],request['call_id'],'collector once cost name')
    typed_equal(trace['cap_s'],caps[request['call_role']],'immutable role cap')
    require(type(trace['max_output_tokens']) is int and trace['max_output_tokens']>=1000,'complete output request bound')
    typed_equal(trace['exec_args'],{'cmd':"pwsh -NoProfile -File '"+BASE+
        "/parent-e360-p17-ROOT-dispatch-adapter.ps1' -CallRole '"+request['call_role']+"'",
        'workdir':'C:/Users/ckthd/Dev/Projects/fly_lab'},'actual ROOT adopted collector dispatch body')
    dispatch=trace['dispatch_ms'];chunks=trace['chunks']
    require(type(dispatch) is int and dispatch>0 and type(chunks) is list and chunks,'genuine dispatch and chunks')
    last=dispatch;known=None;first=None;first_exit=None;complete=True
    for chunk in chunks:
        received=chunk['received_ms'];result=chunk['result']
        require(type(received) is int,'original integer receive clock')
        if received<last:
            require(trace['phase']=='ERROR_HOLD' and type(trace['first_error']) is dict,'clock contradiction retained first error')
            complete=False
        require(first is None,'no result after first genuine FINAL')
        last=received
        require(type(result) is dict,'contradictory transport cannot acquire a later FINAL')
        has_session=result.get('session_id') is not None
        has_exit=result.get('exit_code') is not None
        session=result.get('session_id');returned=result.get('exit_code')
        if has_session and type(session) is int and session>0:
            if known is None:known=session
            require(known==session,'changed session cannot acquire a later FINAL')
        if has_exit and type(returned) is int and not has_session:
            first,first_exit=received,returned
        else:require(has_session and type(session) is int and session>0 and not has_exit,
                     'contradictory result shape cannot acquire a later FINAL')
        output=result.get('output')
        if (type(output) is not str or result.get('truncated') is True or result.get('output_truncated') is True
            or re.search(r'Warning: truncated output|tokens truncated|output was truncated',output or '')):complete=False
    require(first is not None,'lost FINAL remains pending and cannot invent a debit')
    typed_equal(trace['session_id'],known,'first known tool session retained')
    typed_equal(trace['first_final_ms'],first,'immutable first FINAL sample')
    typed_equal(trace['exit_code'],first_exit,'immutable first FINAL exit')
    require(type(seconds) in (int,float) and not isinstance(seconds,bool) and math.isfinite(seconds) and seconds>0,'genuine positive full return')
    elapsed=(first-dispatch)/1000.0
    require(type(trace['full_return_s']) in (int,float) and math.isfinite(trace['full_return_s'])
            and trace['full_return_s']>0,'saved FINAL duration must be a finite positive JSON number')
    require(trace['full_return_s']==elapsed and seconds==elapsed,'exact dispatch to FINAL full return')
    typed_equal(exit_code,first_exit,'actual first FINAL exit')
    require(trace['phase'] in ('FINAL_RECEIVED','ERROR_HOLD'),'closed collector state')
    if trace['phase']=='ERROR_HOLD':
        require(type(trace['first_error']) is dict,'retain first collector failure')
        complete=False
    else:
        typed_equal(trace['first_error'],None,'complete collector has no first error')
        typed_equal(trace['publication'],'REQUIRED','Native publication follows saved FINAL')
    if not any(type(c['result']) is dict and type(c['result'].get('output')) is str
        and c['result']['output'].strip() for c in chunks):complete=False
    if not complete:require(trace['phase']=='ERROR_HOLD' and type(trace['first_error']) is dict,
                            'every collector failure retains the sticky original error')
    return {'valid':complete,'trace':trace_spec,'full_return_s':elapsed,'exit_code':first_exit}




def historical_p18_collector_proof(trace_spec,request,request_spec,seconds,exit_code,*,root=ROOT):
    """Replay immutable genuine transport metadata; never sample a new clock."""
    trace=strict_json(read_spec(trace_spec,root))
    typed_equal(trace['schema'],'flylab.m3.e360.p18.ROOT-genuine-collector-state.v1','collector schema')
    typed_equal(request['pass_id'],'e360-generous-p18','collector current pass only')
    for key in ('pass_id','call_id','call_role'):
        typed_equal(trace[key],request[key],'collector admitted '+key)
    typed_equal(trace['request'],request_spec,'collector exact admitted request')
    typed_equal(trace['collector_source'],request['collector_source'],'collector request source pin')
    typed_equal(trace['collector_source'],file_spec(BASE+'/parent-e360-p18-ROOT-genuine-collector.txt',root),'collector current source')
    typed_equal(trace['state_key'],'e360:p18:ROOT-genuine:'+request['call_id'],'unique collector role state')
    typed_equal(trace_spec['path'],BASE+'/parent-'+request['call_id']+'-ROOT-collector-trace.json','canonical trace publication')
    typed_equal(trace['next_role_allowed'],False,'collector cannot admit another role')
    typed_equal(trace['collection_status'],'FINAL_RECEIVED','first genuine FINAL collected independently of sticky error')
    typed_equal(trace['collection_stopped'],False,'no stopped transport may invent a later FINAL')
    typed_equal(trace['transport_error'],None,'genuine FINAL without a transport exception')
    caps={'ruff':3,'scanner':20,'fixture_build':25,'fixture_worker':55,'fixture_seal':15,
        'suite_build':30,'suite_worker':460,'suite_seal':20,'qualification':30}
    typed_equal(trace['cost_name'],request['call_id'],'collector once cost name')
    typed_equal(trace['cap_s'],caps[request['call_role']],'immutable role cap')
    require(type(trace['max_output_tokens']) is int and trace['max_output_tokens']>=1000,'complete output request bound')
    typed_equal(trace['exec_args'],{'cmd':"pwsh -NoProfile -File '"+BASE+
        "/parent-e360-p18-ROOT-dispatch-adapter.ps1' -CallRole '"+request['call_role']+"'",
        'workdir':'C:/Users/ckthd/Dev/Projects/fly_lab'},'actual ROOT adopted collector dispatch body')
    dispatch=trace['dispatch_ms'];chunks=trace['chunks']
    require(type(dispatch) is int and dispatch>0 and type(chunks) is list and chunks,'genuine dispatch and chunks')
    last=dispatch;known=None;first=None;first_exit=None;complete=True
    for chunk in chunks:
        received=chunk['received_ms'];result=chunk['result']
        require(type(received) is int,'original integer receive clock')
        if received<last:
            require(trace['phase']=='ERROR_HOLD' and type(trace['first_error']) is dict,'clock contradiction retained first error')
            complete=False
        require(first is None,'no result after first genuine FINAL')
        last=received
        require(type(result) is dict,'contradictory transport cannot acquire a later FINAL')
        has_session=result.get('session_id') is not None
        has_exit=result.get('exit_code') is not None
        session=result.get('session_id');returned=result.get('exit_code')
        if has_session and type(session) is int and session>0:
            if known is None:known=session
            require(known==session,'changed session cannot acquire a later FINAL')
        if has_exit and type(returned) is int and not has_session:
            first,first_exit=received,returned
        else:require(has_session and type(session) is int and session>0 and not has_exit,
                     'contradictory result shape cannot acquire a later FINAL')
        output=result.get('output')
        if (type(output) is not str or result.get('truncated') is True or result.get('output_truncated') is True
            or re.search(r'Warning: truncated output|tokens truncated|output was truncated',output or '')):complete=False
    require(first is not None,'lost FINAL remains pending and cannot invent a debit')
    typed_equal(trace['session_id'],known,'first known tool session retained')
    typed_equal(trace['first_final_ms'],first,'immutable first FINAL sample')
    typed_equal(trace['exit_code'],first_exit,'immutable first FINAL exit')
    require(type(seconds) in (int,float) and not isinstance(seconds,bool) and math.isfinite(seconds) and seconds>0,'genuine positive full return')
    elapsed=(first-dispatch)/1000.0
    require(type(trace['full_return_s']) in (int,float) and math.isfinite(trace['full_return_s'])
            and trace['full_return_s']>0,'saved FINAL duration must be a finite positive JSON number')
    require(trace['full_return_s']==elapsed and seconds==elapsed,'exact dispatch to FINAL full return')
    typed_equal(exit_code,first_exit,'actual first FINAL exit')
    require(trace['phase'] in ('FINAL_RECEIVED','ERROR_HOLD'),'closed collector state')
    if trace['phase']=='ERROR_HOLD':
        require(type(trace['first_error']) is dict,'retain first collector failure')
        complete=False
    else:
        typed_equal(trace['first_error'],None,'complete collector has no first error')
        typed_equal(trace['publication'],'REQUIRED','Native publication follows saved FINAL')
    if not any(type(c['result']) is dict and type(c['result'].get('output')) is str
        and c['result']['output'].strip() for c in chunks):complete=False
    if not complete:require(trace['phase']=='ERROR_HOLD' and type(trace['first_error']) is dict,
                            'every collector failure retains the sticky original error')
    return {'valid':complete,'trace':trace_spec,'full_return_s':elapsed,'exit_code':first_exit}

@historical_dependencies()
def historical_p18_repair_policy(root=ROOT):
    raw=read_spec(P18_FROZEN_POLICY_SPEC,root);enroll_json_descriptor(P18_FROZEN_POLICY_SPEC,root)
    return strict_json(raw)

def policy_view(value,root=ROOT):
    helpers=copy.deepcopy(value['effective_helper_limits'])
    for role in value['primary_roles']:
        if role in helpers:helpers[role]['successful_suffix_after_s']=value['future_with_unused_fallback120_s'][value['primary_roles'].index(role)]
    return {**resource_extension(root),'effective_roles':value['effective_roles'],'scanner_limits':value['scanner_limits'],
        'generated_call_order':value['primary_roles'],'successful_suffix_after_s':value['future_with_unused_fallback120_s'],
        'primary_successful_suffix_after_s':value['primary_successful_suffix_after_s'],
        'primary_total_reserved_s':value['primary_total_reserved_s'],
        'remaining_generated_administration_reserve_s':value['primary_admin_total_s'],
        'fixture_future_administration_reserved_s':value['fixture_primary_future_administration_s'],
        'suite_future_administration_reserved_s':value['suite_primary_future_administration_s'],
        'separate_same_pending_fallback_reserved_s':120,'helper_limits':helpers}

def authority_spec_for_pass(pass_id,root=ROOT):
    context=execution_context(root)
    if context is not None and pass_id==context['manifest']['pass_id']:return copy.deepcopy(context['manifest_spec'])
    if context is not None and pass_id in context['historical_contexts']:return copy.deepcopy(context['historical_contexts'][pass_id]['manifest_spec'])
    if pass_id=='e360-generous-p18':return copy.deepcopy(P18_ORIGINAL_POLICY_SPEC)
    if pass_id=='e360-generous-p19':return copy.deepcopy(P19_ORIGINAL_POLICY_SPEC)
    if pass_id=='e360-generous-p20':return copy.deepcopy(P20_ORIGINAL_POLICY_SPEC)
    if pass_id=='e360-generous-p21':return copy.deepcopy(P21_ORIGINAL_POLICY_SPEC)
    if pass_id=='e360-generous-p22':return copy.deepcopy(P22_ORIGINAL_POLICY_SPEC)
    if pass_id=='e360-generous-p25':return copy.deepcopy(P25_ORIGINAL_POLICY_SPEC)
    if pass_id=='e360-generous-p23':return copy.deepcopy(P23_ORIGINAL_POLICY_SPEC)
    if pass_id=='e360-generous-p27':return copy.deepcopy(P27_ORIGINAL_POLICY_SPEC)
    if pass_id=='e360-generous-p29':return copy.deepcopy(P29_ORIGINAL_POLICY_SPEC)
    if pass_id==CURRENT_COST_IDENTITY['pass_id']:return copy.deepcopy(CURRENT_COST_IDENTITY['authority'])
    return file_spec(authority_path_for_pass(pass_id),root)

_COMPLETED_STREAM_PINS=(*_COMPLETED_STREAM_PINS,json.loads(r'''{"bytes":0,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p19-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}'''))

_COMPLETED_STREAM_PINS=(*_COMPLETED_STREAM_PINS,json.loads(r'''{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p20-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","bytes":0}'''))

def cost_authority_spec(index,pass_id,root=ROOT):
    context=execution_context(root)
    if context is not None:
        require(type(index) is int and index>=59 and type(pass_id) is str,'typed sealed/current cost identity')
        if index>=context['manifest']['start_index']:
            typed_equal(pass_id,context['manifest']['pass_id'],'exact current data pass')
            return copy.deepcopy(context['manifest_spec'])
        if index>=111:
            historical=context['historical_contexts'].get(pass_id)
            require(historical is not None and index>=historical['manifest']['start_index']
                and context['anchor']['debits'][index]['cost_id'] in context['anchor']['e360_calls']
                and context['anchor']['e360_calls'][context['anchor']['debits'][index]['cost_id']]['pass_id']==pass_id,'actual sealed context cost interval')
            return copy.deepcopy(historical['manifest_spec'])
        if index>=109:
            typed_equal(pass_id,'e360-generous-p30','actual frozen P30 paid rows')
            return copy.deepcopy(CURRENT_REPAIR_SPEC)
    """Finite historical switch; unmatched rows require the compiled current identity."""
    require(type(index) is int and index>=59 and type(pass_id) is str,'typed post59 cost index/pass')
    version=next((version for start,end,version in COST_AUTHORITY_PASS_RANGES if start<=index<end),None)
    if version is not None:
        typed_equal(pass_id,'e360-generous-p'+str(version),'cost index and exact expected historical pass')
        return authority_spec_for_pass(pass_id,root)
    require(index>=CURRENT_COST_IDENTITY['start_index'],'unclassified historical cost gap')
    typed_equal(pass_id,CURRENT_COST_IDENTITY['pass_id'],'compiled current cost pass')
    return copy.deepcopy(CURRENT_COST_IDENTITY['authority'])

def validate_cost_authority(index,cost,actual,*,root=ROOT):
    require(type(cost) is dict and type(actual) is dict and all(key in cost and key in actual for key in ('pass_id','revalidation_amendment')),'typed complete cost/actual authority inputs')
    expected=cost_authority_spec(index,cost['pass_id'],root)
    typed_equal(actual['pass_id'],cost['pass_id'],'actual expected index/pass identity')
    typed_equal(cost['revalidation_amendment'],expected,'historical/current cost amendment')
    typed_equal(actual['revalidation_amendment'],expected,'historical/current actual amendment')
    return expected

_COMPLETED_STREAM_PINS=(*_COMPLETED_STREAM_PINS,json.loads(r'''{"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p21-ruff-stderr.txt","bytes":0}'''))

_COMPLETED_STREAM_PINS=(*_COMPLETED_STREAM_PINS,json.loads(r'''{"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","bytes":0,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p22-ruff-stderr.txt"}'''))

@historical_dependencies()
def historical_p22_repair_policy(root=ROOT):
    raw=read_spec(P22_FROZEN_POLICY_SPEC,root);enroll_json_descriptor(P22_FROZEN_POLICY_SPEC,root)
    return strict_json(raw)

@historical_dependencies()
def historical_p22_source_freeze(root=ROOT):
    raw=read_spec(P22_FROZEN_FREEZE_SPEC,root);enroll_json_descriptor(P22_FROZEN_FREEZE_SPEC,root)
    freeze=strict_json(raw)
    typed_equal(freeze['approved_literal'],P22_ORIGINAL_POLICY_SPEC,'historical P22 original logical policy')
    typed_equal(freeze['current_policy_frozen'],P22_FROZEN_POLICY_SPEC,'historical P22 physical policy')
    require(len(freeze['source_copies'])==4 and len(freeze['own_sources_sha256'])==9 and len(freeze['all_P22_Native_sources'])==4,'historical P22 Source4/own9/Native4')
    changed={row['current']['path']:row for row in freeze['source_copies']}
    for row in changed.values():
        for key in ('bytes','sha256'):typed_equal(row['current'][key],row['frozen'][key],'historical P22 same bytes')
        typed_equal(row['current']['sha256'],freeze['own_sources_sha256'][row['current']['path']],'historical P22 own source')
        read_spec(row['frozen'],root);enroll_json_descriptor(row['frozen'],root)
    for name,digest in freeze['own_sources_sha256'].items():
        if name not in changed:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'unchanged own5')
    for spec in (*freeze['all_P22_Native_sources'],freeze['collector_source']):read_spec(spec,root);enroll_json_descriptor(spec,root)
    return freeze

def historical_p22_recovery_event(root=ROOT):
    policy=historical_p22_repair_policy(root);historical_p22_source_freeze(root)
    return {'schema':'flylab.m3.e360.p22-recovery-activation.v1','revalidation_amendment':copy.deepcopy(P22_ORIGINAL_POLICY_SPEC),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited103':P21_CLOSED103_SPEC,
        'previous_revalidation_amendment':copy.deepcopy(P21_ORIGINAL_POLICY_SPEC),'manual_generated_pass_id':'e360-generous-p22',
        'new_generated_grant_s':0,'human_authorization':policy['human_authorization'],'as_executed_source_freeze':P22_FROZEN_FREEZE_SPEC}

def validate_historical_p22_activation_pair(live,root=ROOT):
    require('e360_p22_recovery_resource_grant' not in live,'historical P22 grant0/no grant field')
    present=[key in live for key in HISTORICAL_P22_FIELDS]
    require(all(present) or not any(present),'complete historical P22 pair')
    if not all(present):return False
    typed_equal(live[HISTORICAL_P22_FIELDS[0]],historical_p22_recovery_event(root),'exact frozen P22 event9')
    typed_equal(live[HISTORICAL_P22_FIELDS[1]],P22_ORIGINAL_POLICY_SPEC,'historical P22 logical authority')
    return True

@historical_dependencies()
def completed_historical_p22_proof(root=ROOT):
    """Direct completed metadata; no frozen module import, current selector or new reader."""
    raw=read_spec(P22_CLOSED104_SPEC,root);enroll_json_descriptor(P22_CLOSED104_SPEC,root)
    closed=strict_json(raw);prior=completed_historical_p21_proof(root);freeze=historical_p22_source_freeze(root)
    require((len(closed['debits']),len(closed['e360_calls']),len(closed['e360_passes']))==(104,54,22),'actual104/54/22')
    typed_equal(closed['debits'][:103],prior['debits'],'all prior103 costs')
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(closed[key][identity],item,'all prior103 state')
    for key in prior:
        if key.endswith(('_activation','_amendment')) or key in ('e360_resource_grant','e360_suite_recovery_resource_grant','e360_p19_recovery_resource_grant'):
            typed_equal(closed[key],prior[key],'all prior pairs and grants3')
    require(validate_historical_p22_activation_pair(closed,root),'actual frozen P22 pair2/grant0')
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):typed_equal(closed[key],{},'closed104 ownership0')
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(closed[key],0,'closed104 original0')
    for key in prior:
        if key.startswith(('original_','e360_original_','e359_original_')):typed_equal(closed[key],prior[key],'unchanged original counters')
    passed=closed['e360_passes']['e360-generous-p22']
    typed_equal(passed['source_sha256'],freeze['own_sources_sha256'],'actual frozen P22 own9')
    typed_equal(passed['revalidation_amendment'],P22_ORIGINAL_POLICY_SPEC,'actual original P22 policy')
    typed_equal(passed['next_index'],1,'actual P22 Ruff next1')
    typed_equal(passed['failed_cap_s'],3,'actual P22 Ruff cap3')
    typed_equal(passed['failed_cost'],P22_ACTUAL_RUFF_RECORD_SPECS['full-return.json'],'actual P22 failure')
    typed_equal(passed['predecessor_failed_cap_s'],30,'P22 preserved predecessor P21 scanner cap30')
    typed_equal(passed['actual_failure'],P21_ACTUAL_SCANNER_RECORD_SPECS['full-return.json'],'P22 preserved actual P21 scanner')
    for spec in passed['concrete_fix']:historical_fix_bytes(spec,pass_id='e360-generous-p22',root=root)
    for spec in P22_ACTUAL_RUFF_RECORD_SPECS.values():historical_p12_stream_bytes(spec,root) if spec['bytes']==0 else read_spec(spec,root)
    debit=closed['debits'][103];cost=strict_json(read_spec(debit['descriptor'],root));actual=strict_json(read_spec(cost['actual_tool_return'],root))
    typed_equal(debit['descriptor'],P22_ACTUAL_RUFF_RECORD_SPECS['full-return.json'],'actual P22 once Ruff debit')
    typed_equal(debit['actual_outer_full_return_s'],1.606,'actual P22 full return')
    typed_equal(debit['actual_returned_exit_code'],1,'actual P22 exit1')
    typed_equal(cost['source_sha256'],freeze['own_sources_sha256'],'actual P22 cost own9')
    typed_equal(cost['collector_proof_valid'],True,'actual flat cost collector validity, Ruff exit1 retained')
    typed_equal(actual['collector_proof_valid'],True,'actual flat return collector validity')
    typed_equal(cost['internal_completion_valid'],True,'actual completion metadata valid, role still failed')
    validate_cost_authority(103,cost,actual,root=root)
    require('schema' not in actual,'actual P22 return has no schema')
    capture=strict_json(read_spec(P22_ACTUAL_RUFF_RECORD_SPECS['genuine-capture-return.json'],root))
    require('schema' not in capture and 'pass_id' not in capture,'actual capture observed shape')
    request=paid_request_identity(debit,cost,actual,closed,root=root);call=closed['e360_calls'][cost['call_id']]
    proof=p22_collector_proof(actual['collector_trace'],request,call['request'],1.606,1,root=root)
    typed_equal(proof['valid'],True,'actual P22 genuine first FINAL failed Ruff')
    typed_equal(closed['generated_shared_cumulative_limit_s'],1700,'historical grants3 shared')
    typed_equal(closed['e360_generated_local_limit_s'],1589,'historical grants3 local')
    typed_equal(closed['generated_shared_cumulative_used_s'],664.0600156,'actual all104 shared used')
    typed_equal(closed['e360_generated_local_used_s'],553.103,'actual all104 local used')
    return closed

def validate_completed_p22_pass(pass_id,passed,debits,*,root=ROOT):
    closed=completed_historical_p22_proof(root)
    typed_equal(pass_id,'e360-generous-p22','immutable completed P22 only')
    typed_equal(passed,closed['e360_passes'][pass_id],'actual historical P22 pass')
    typed_equal(debits[:104],closed['debits'],'actual all104 prefix')

def p23_collector_proof(trace_spec,request,request_spec,seconds,exit_code,*,root=ROOT):
    return p19_collector_proof(trace_spec,request,request_spec,seconds,exit_code,root=root,version='p23')

_COMPLETED_STREAM_PINS=(*_COMPLETED_STREAM_PINS,json.loads(r'''{"bytes":0,"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p23-ruff-stderr.txt","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}'''))

@historical_dependencies()
def historical_p23_repair_policy(root=ROOT):
    raw=read_spec(P23_FROZEN_POLICY_SPEC,root);enroll_json_descriptor(P23_FROZEN_POLICY_SPEC,root)
    return strict_json(raw)

@historical_dependencies()
def historical_p23_source_freeze(root=ROOT):
    raw=read_spec(P23_FROZEN_FREEZE_SPEC,root);enroll_json_descriptor(P23_FROZEN_FREEZE_SPEC,root)
    freeze=strict_json(raw)
    typed_equal(freeze['approved_literal'],P23_ORIGINAL_POLICY_SPEC,'historical P23 original logical policy')
    typed_equal(freeze['current_policy_frozen'],P23_FROZEN_POLICY_SPEC,'historical P23 physical policy')
    require(len(freeze['source_copies'])==4 and len(freeze['own_sources_sha256'])==9 and len(freeze['all_P23_Native_sources'])==4,'historical P23 Source4/own9/Native4')
    changed={row['current']['path']:row for row in freeze['source_copies']}
    for row in changed.values():
        for key in ('bytes','sha256'):typed_equal(row['current'][key],row['frozen'][key],'historical P23 same bytes')
        typed_equal(row['current']['sha256'],freeze['own_sources_sha256'][row['current']['path']],'historical P23 own source')
        read_spec(row['frozen'],root);enroll_json_descriptor(row['frozen'],root)
    for name,digest in freeze['own_sources_sha256'].items():
        if name not in changed:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'unchanged own5')
    for spec in (*freeze['all_P23_Native_sources'],freeze['collector_source']):read_spec(spec,root);enroll_json_descriptor(spec,root)
    return freeze

def historical_p23_recovery_event(root=ROOT):
    policy=historical_p23_repair_policy(root);historical_p23_source_freeze(root)
    return {'schema':'flylab.m3.e360.p23-recovery-activation.v1','revalidation_amendment':copy.deepcopy(P23_ORIGINAL_POLICY_SPEC),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited104':P22_CLOSED104_SPEC,
        'previous_revalidation_amendment':copy.deepcopy(P22_ORIGINAL_POLICY_SPEC),'manual_generated_pass_id':'e360-generous-p23',
        'new_generated_grant_s':0,'human_authorization':policy['human_authorization'],'as_executed_source_freeze':P23_FROZEN_FREEZE_SPEC}

def validate_historical_p23_activation_pair(live,root=ROOT):
    require('e360_p23_recovery_resource_grant' not in live,'historical P23 grant0/no grant field')
    present=[key in live for key in HISTORICAL_P23_FIELDS]
    require(all(present) or not any(present),'complete historical P23 pair')
    if not all(present):return False
    typed_equal(live[HISTORICAL_P23_FIELDS[0]],historical_p23_recovery_event(root),'exact frozen P23 event9')
    typed_equal(live[HISTORICAL_P23_FIELDS[1]],P23_ORIGINAL_POLICY_SPEC,'historical P23 logical authority')
    return True

@historical_dependencies()
def completed_historical_p23_proof(root=ROOT):
    """Direct completed metadata; no frozen module import, current selector or new reader."""
    raw=read_spec(P23_CLOSED105_SPEC,root);enroll_json_descriptor(P23_CLOSED105_SPEC,root)
    closed=strict_json(raw);prior=completed_historical_p22_proof(root);freeze=historical_p23_source_freeze(root)
    require((len(closed['debits']),len(closed['e360_calls']),len(closed['e360_passes']))==(105,55,23),'actual105/55/23')
    typed_equal(closed['debits'][:104],prior['debits'],'all prior104 costs')
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(closed[key][identity],item,'all prior104 state')
    for key in prior:
        if key.endswith(('_activation','_amendment')) or key in ('e360_resource_grant','e360_suite_recovery_resource_grant','e360_p19_recovery_resource_grant'):
            typed_equal(closed[key],prior[key],'all prior pairs and grants3')
    require(validate_historical_p23_activation_pair(closed,root),'actual frozen P23 pair2/grant0')
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):typed_equal(closed[key],{},'closed105 ownership0')
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(closed[key],0,'closed105 original0')
    for key in prior:
        if key.startswith(('original_','e360_original_','e359_original_')):typed_equal(closed[key],prior[key],'unchanged original counters')
    passed=closed['e360_passes']['e360-generous-p23']
    typed_equal(passed['source_sha256'],freeze['own_sources_sha256'],'actual frozen P23 own9')
    typed_equal(passed['revalidation_amendment'],P23_ORIGINAL_POLICY_SPEC,'actual original P23 policy')
    typed_equal(passed['next_index'],1,'actual P23 Ruff next1')
    typed_equal(passed['failed_cap_s'],3,'actual P23 Ruff cap3')
    typed_equal(passed['failed_cost'],P23_ACTUAL_RUFF_RECORD_SPECS['full-return.json'],'actual P23 failure')
    typed_equal(passed['predecessor_failed_cap_s'],3,'P23 preserved predecessor P22 Ruff cap3')
    typed_equal(passed['actual_failure'],P22_ACTUAL_RUFF_RECORD_SPECS['full-return.json'],'P23 preserved actual P22 Ruff')
    for spec in passed['concrete_fix']:historical_fix_bytes(spec,pass_id='e360-generous-p23',root=root)
    for spec in P23_ACTUAL_RUFF_RECORD_SPECS.values():historical_p12_stream_bytes(spec,root) if spec['bytes']==0 else read_spec(spec,root)
    debit=closed['debits'][104];cost=strict_json(read_spec(debit['descriptor'],root));actual=strict_json(read_spec(cost['actual_tool_return'],root))
    typed_equal(debit['descriptor'],P23_ACTUAL_RUFF_RECORD_SPECS['full-return.json'],'actual P23 once Ruff debit')
    typed_equal(debit['actual_outer_full_return_s'],2.056,'actual P23 full return')
    typed_equal(debit['actual_returned_exit_code'],1,'actual P23 exit1')
    typed_equal(cost['source_sha256'],freeze['own_sources_sha256'],'actual P23 cost own9')
    typed_equal(cost['collector_proof_valid'],True,'actual flat cost collector validity, Ruff exit1 retained')
    typed_equal(actual['collector_proof_valid'],True,'actual flat return collector validity')
    typed_equal(cost['internal_completion_valid'],True,'actual completion metadata valid, role still failed')
    validate_cost_authority(104,cost,actual,root=root)
    require('schema' not in actual,'actual P23 return has no schema')
    capture=strict_json(read_spec(P23_ACTUAL_RUFF_RECORD_SPECS['genuine-capture-return.json'],root))
    require('schema' not in capture and 'pass_id' not in capture,'actual capture observed shape')
    request=paid_request_identity(debit,cost,actual,closed,root=root);call=closed['e360_calls'][cost['call_id']]
    proof=p23_collector_proof(actual['collector_trace'],request,call['request'],2.056,1,root=root)
    typed_equal(proof['valid'],True,'actual P23 genuine first FINAL failed Ruff')
    typed_equal(closed['generated_shared_cumulative_limit_s'],1700,'historical grants3 shared')
    typed_equal(closed['e360_generated_local_limit_s'],1589,'historical grants3 local')
    typed_equal(closed['generated_shared_cumulative_used_s'],666.1160156000001,'actual all105 shared used')
    typed_equal(closed['e360_generated_local_used_s'],555.159,'actual all105 local used')
    return closed

def validate_completed_p23_pass(pass_id,passed,debits,*,root=ROOT):
    closed=completed_historical_p23_proof(root)
    typed_equal(pass_id,'e360-generous-p23','immutable completed P23 only')
    typed_equal(passed,closed['e360_passes'][pass_id],'actual historical P23 pass')
    typed_equal(debits[:105],closed['debits'],'actual all105 prefix')

def p30_collector_proof(trace_spec,request,request_spec,seconds,exit_code,*,root=ROOT):
    if execution_is_current(request['pass_id'],root):return p19_collector_proof(trace_spec,request,request_spec,seconds,exit_code,root=root,version='execution')
    context=execution_context(root)
    if context is not None and request['pass_id'] in context['historical_contexts']:
        return p19_collector_proof(trace_spec,request,request_spec,seconds,exit_code,root=root,version='execution',selected_context=context['historical_contexts'][request['pass_id']])
    return p19_collector_proof(trace_spec,request,request_spec,seconds,exit_code,root=root,version='p30')

@historical_dependencies()
def historical_p21_repair_policy(root=ROOT):
    raw=read_spec(P21_FROZEN_POLICY_SPEC,root);enroll_json_descriptor(P21_FROZEN_POLICY_SPEC,root)
    return strict_json(raw)

@historical_dependencies()
def historical_p21_source_freeze(root=ROOT):
    raw=read_spec(P21_FROZEN_FREEZE_SPEC,root);enroll_json_descriptor(P21_FROZEN_FREEZE_SPEC,root)
    freeze=strict_json(raw)
    typed_equal(freeze['approved_literal'],P21_ORIGINAL_POLICY_SPEC,'original logical P21 policy')
    typed_equal(freeze['current_policy_frozen'],P21_FROZEN_POLICY_SPEC,'physical frozen P21 policy')
    require(len(freeze['source_copies'])==4 and len(freeze['own_sources_sha256'])==9 and len(freeze['all_P21_Native_sources'])==4,'P21 Source4/own9/Native4')
    changed={row['current']['path']:row for row in freeze['source_copies']}
    for row in changed.values():
        for key in ('bytes','sha256'):typed_equal(row['current'][key],row['frozen'][key],'P21 exact frozen Source bytes')
        typed_equal(row['current']['sha256'],freeze['own_sources_sha256'][row['current']['path']],'P21 recorded own source')
        read_spec(row['frozen'],root);enroll_json_descriptor(row['frozen'],root)
    for name,digest in freeze['own_sources_sha256'].items():
        if name not in changed:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'unchanged own5')
    for spec in (*freeze['all_P21_Native_sources'],freeze['collector_source']):read_spec(spec,root);enroll_json_descriptor(spec,root)
    return freeze

def historical_p21_recovery_event(root=ROOT):
    policy=historical_p21_repair_policy(root);historical_p21_source_freeze(root)
    return {'schema':'flylab.m3.e360.p21-recovery-activation.v1','revalidation_amendment':copy.deepcopy(P21_ORIGINAL_POLICY_SPEC),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited101':P20_CLOSED101_SPEC,
        'previous_revalidation_amendment':copy.deepcopy(P20_ORIGINAL_POLICY_SPEC),'manual_generated_pass_id':'e360-generous-p21',
        'new_generated_grant_s':0,'human_authorization':policy['human_authorization'],'as_executed_source_freeze':P21_FROZEN_FREEZE_SPEC}

def validate_historical_p21_activation_pair(live,root=ROOT):
    require('e360_p21_recovery_resource_grant' not in live,'historical P21 grant0/no grant field')
    present=[key in live for key in HISTORICAL_P21_FIELDS]
    require(all(present) or not any(present),'complete historical P21 pair')
    if not all(present):return False
    typed_equal(live[HISTORICAL_P21_FIELDS[0]],historical_p21_recovery_event(root),'exact historical P21 event9')
    typed_equal(live[HISTORICAL_P21_FIELDS[1]],P21_ORIGINAL_POLICY_SPEC,'historical P21 logical authority')
    return True

@historical_dependencies()
def completed_historical_p21_proof(root=ROOT):
    """Two actual roles, direct closed103; no frozen module or invented scanner output."""
    raw=read_spec(P21_CLOSED103_SPEC,root);enroll_json_descriptor(P21_CLOSED103_SPEC,root)
    closed=strict_json(raw);prior=completed_historical_p20_proof(root);freeze=historical_p21_source_freeze(root)
    require((len(closed['debits']),len(closed['e360_calls']),len(closed['e360_passes']))==(103,53,21),'actual103/53/21')
    typed_equal(closed['debits'][:101],prior['debits'],'all prior101 costs')
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(closed[key][identity],item,'all prior101 state')
    for key in prior:
        if key.endswith(('_activation','_amendment')) or key in ('e360_resource_grant','e360_suite_recovery_resource_grant','e360_p19_recovery_resource_grant') or key.startswith(('original_','e360_original_','e359_original_')):
            typed_equal(closed[key],prior[key],'all prior pairs/grants3/original counters')
    require(validate_historical_p21_activation_pair(closed,root),'actual P21 pair2/grant0')
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):typed_equal(closed[key],{},'closed103 ownership0')
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(closed[key],0,'closed103 original0')
    passed=closed['e360_passes']['e360-generous-p21']
    typed_equal(passed['source_sha256'],freeze['own_sources_sha256'],'actual frozen P21 own9')
    typed_equal(passed['revalidation_amendment'],P21_ORIGINAL_POLICY_SPEC,'actual P21 original policy')
    typed_equal(passed['next_index'],2,'actual P21 two returned roles')
    typed_equal(passed['failed_cap_s'],30,'actual P21 failed scanner cap30')
    typed_equal(passed['failed_cost'],P21_ACTUAL_SCANNER_RECORD_SPECS['full-return.json'],'actual P21 failed scanner')
    typed_equal(passed['predecessor_failed_cap_s'],3,'P21 preserved predecessor Ruff cap3')
    typed_equal(passed['actual_failure'],P20_ACTUAL_RECORD_SPECS['full-return.json'],'P21 preserved actual P20 Ruff')
    for spec in passed['concrete_fix']:historical_fix_bytes(spec,pass_id='e360-generous-p21',root=root)
    for spec in (*P21_ACTUAL_RUFF_RECORD_SPECS.values(),*P21_ACTUAL_SCANNER_RECORD_SPECS.values()):
        historical_p12_stream_bytes(spec,root) if spec['bytes']==0 else read_spec(spec,root)
    for index,records,role,seconds,code,cap in ((101,P21_ACTUAL_RUFF_RECORD_SPECS,'ruff',1.605,0,3),(102,P21_ACTUAL_SCANNER_RECORD_SPECS,'scanner',6.868,1,30)):
        debit=closed['debits'][index];cost=strict_json(read_spec(debit['descriptor'],root));actual=strict_json(read_spec(records['actual.json'],root))
        typed_equal(debit['descriptor'],records['full-return.json'],'actual P21 once role debit')
        typed_equal(debit['actual_outer_full_return_s'],seconds,'actual P21 full return')
        typed_equal(debit['actual_returned_exit_code'],code,'actual P21 exit')
        typed_equal(cost['source_sha256'],freeze['own_sources_sha256'],'actual P21 role own9')
        typed_equal(cost['call_role'],role,'actual P21 different role')
        typed_equal(cost['internal_completion_valid'],True,'valid metadata preserves actual exit result')
        validate_cost_authority(index,cost,actual,root=root)
        request=paid_request_identity(debit,cost,actual,closed,root=root);call=closed['e360_calls'][cost['call_id']]
        typed_equal(call['reserved_s'],cap,'actual P21 role cap')
        typed_equal(call['request'],records['final-request.json'],'actual admitted P21 request')
        typed_equal(actual['collector_trace'],records['ROOT-collector-trace.json'],'actual P21 saved trace')
        proof=p21_collector_proof(actual['collector_trace'],request,call['request'],seconds,code,root=root)
        typed_equal(proof['valid'],True,'actual P21 first FINAL collector valid')
        if role=='scanner':
            typed_equal(cost['expected_output'],None,'actual scanner produced no report')
            typed_equal(cost['internal_completion'],None,'actual scanner no completion object')
    typed_equal(closed['generated_shared_cumulative_limit_s'],1700,'historical grants3 shared')
    typed_equal(closed['e360_generated_local_limit_s'],1589,'historical grants3 local')
    typed_equal(closed['generated_shared_cumulative_used_s'],662.4540156,'actual all103 shared')
    typed_equal(closed['e360_generated_local_used_s'],551.497,'actual all103 local')
    return closed

def validate_completed_p21_pass(pass_id,passed,debits,*,root=ROOT):
    closed=completed_historical_p21_proof(root)
    typed_equal(pass_id,'e360-generous-p21','immutable completed P21 only')
    typed_equal(passed,closed['e360_passes'][pass_id],'actual historical P21 pass')
    typed_equal(debits[:103],closed['debits'],'actual all103 prefix')

def p22_collector_proof(trace_spec,request,request_spec,seconds,exit_code,*,root=ROOT):
    return p19_collector_proof(trace_spec,request,request_spec,seconds,exit_code,root=root,version='p22')

@historical_dependencies()
def historical_p20_repair_policy(root=ROOT):
    raw=read_spec(P20_FROZEN_POLICY_SPEC,root);enroll_json_descriptor(P20_FROZEN_POLICY_SPEC,root)
    return strict_json(raw)

@historical_dependencies()
def historical_p20_source_freeze(root=ROOT):
    raw=read_spec(P20_FROZEN_FREEZE_SPEC,root);enroll_json_descriptor(P20_FROZEN_FREEZE_SPEC,root)
    freeze=strict_json(raw)
    typed_equal(freeze['approved_literal'],P20_ORIGINAL_POLICY_SPEC,'historical P20 original logical policy')
    typed_equal(freeze['current_policy_frozen'],P20_FROZEN_POLICY_SPEC,'historical P20 physical policy')
    require(len(freeze['source_copies'])==4 and len(freeze['own_sources_sha256'])==9 and len(freeze['all_P20_Native_sources'])==4,'historical P20 Source4/own9/Native4')
    changed={row['current']['path']:row for row in freeze['source_copies']}
    for row in changed.values():
        for key in ('bytes','sha256'):typed_equal(row['current'][key],row['frozen'][key],'historical P20 same bytes')
        typed_equal(row['current']['sha256'],freeze['own_sources_sha256'][row['current']['path']],'historical P20 own source')
        read_spec(row['frozen'],root);enroll_json_descriptor(row['frozen'],root)
    for name,digest in freeze['own_sources_sha256'].items():
        if name not in changed:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'unchanged own5')
    for spec in (*freeze['all_P20_Native_sources'],freeze['collector_source']):read_spec(spec,root);enroll_json_descriptor(spec,root)
    return freeze

def historical_p20_recovery_event(root=ROOT):
    policy=historical_p20_repair_policy(root);historical_p20_source_freeze(root)
    return {'schema':'flylab.m3.e360.p20-recovery-activation.v1','revalidation_amendment':copy.deepcopy(P20_ORIGINAL_POLICY_SPEC),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited100':P19_CLOSED100_SPEC,
        'previous_revalidation_amendment':copy.deepcopy(P19_ORIGINAL_POLICY_SPEC),'manual_generated_pass_id':'e360-generous-p20',
        'new_generated_grant_s':0,'human_authorization':policy['human_authorization'],'as_executed_source_freeze':P20_FROZEN_FREEZE_SPEC}

def validate_historical_p20_activation_pair(live,root=ROOT):
    require('e360_p20_recovery_resource_grant' not in live,'historical P20 grant0/no grant field')
    present=[key in live for key in HISTORICAL_P20_FIELDS]
    require(all(present) or not any(present),'complete historical P20 pair')
    if not all(present):return False
    typed_equal(live[HISTORICAL_P20_FIELDS[0]],historical_p20_recovery_event(root),'exact frozen P20 event9')
    typed_equal(live[HISTORICAL_P20_FIELDS[1]],P20_ORIGINAL_POLICY_SPEC,'historical P20 logical authority')
    return True

@historical_dependencies()
def completed_historical_p20_proof(root=ROOT):
    """Direct completed metadata; no frozen module import, current selector or new reader."""
    raw=read_spec(P20_CLOSED101_SPEC,root);enroll_json_descriptor(P20_CLOSED101_SPEC,root)
    closed=strict_json(raw);prior=completed_historical_p19_proof(root);freeze=historical_p20_source_freeze(root)
    require((len(closed['debits']),len(closed['e360_calls']),len(closed['e360_passes']))==(101,51,20),'actual101/51/20')
    typed_equal(closed['debits'][:100],prior['debits'],'all prior100 costs')
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(closed[key][identity],item,'all prior100 state')
    for key in prior:
        if key.endswith(('_activation','_amendment')) or key in ('e360_resource_grant','e360_suite_recovery_resource_grant','e360_p19_recovery_resource_grant'):
            typed_equal(closed[key],prior[key],'all prior pairs and grants3')
    require(validate_historical_p20_activation_pair(closed,root),'actual frozen P20 pair2/grant0')
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):typed_equal(closed[key],{},'closed101 ownership0')
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(closed[key],0,'closed101 original0')
    for key in prior:
        if key.startswith(('original_','e360_original_','e359_original_')):typed_equal(closed[key],prior[key],'unchanged original counters')
    passed=closed['e360_passes']['e360-generous-p20']
    typed_equal(passed['source_sha256'],freeze['own_sources_sha256'],'actual frozen P20 own9')
    typed_equal(passed['revalidation_amendment'],P20_ORIGINAL_POLICY_SPEC,'actual original P20 policy')
    typed_equal(passed['next_index'],1,'actual P20 Ruff next1')
    typed_equal(passed['failed_cap_s'],3,'actual P20 Ruff cap3')
    typed_equal(passed['failed_cost'],P20_ACTUAL_RECORD_SPECS['full-return.json'],'actual P20 failure')
    for spec in passed['concrete_fix']:historical_fix_bytes(spec,pass_id='e360-generous-p20',root=root)
    for spec in P20_ACTUAL_RECORD_SPECS.values():historical_p12_stream_bytes(spec,root) if spec['bytes']==0 else read_spec(spec,root)
    debit=closed['debits'][100];cost=strict_json(read_spec(debit['descriptor'],root));actual=strict_json(read_spec(cost['actual_tool_return'],root))
    typed_equal(debit['descriptor'],P20_ACTUAL_RECORD_SPECS['full-return.json'],'actual P20 once Ruff debit')
    typed_equal(debit['actual_outer_full_return_s'],1.587,'actual P20 full return')
    typed_equal(debit['actual_returned_exit_code'],1,'actual P20 exit1')
    typed_equal(cost['source_sha256'],freeze['own_sources_sha256'],'actual P20 cost own9')
    request=paid_request_identity(debit,cost,actual,closed,root=root);call=closed['e360_calls'][cost['call_id']]
    proof=p20_collector_proof(actual['collector_trace'],request,call['request'],1.587,1,root=root)
    typed_equal(proof['valid'],True,'actual P20 genuine first FINAL failed Ruff')
    typed_equal(closed['generated_shared_cumulative_limit_s'],1700,'historical grants3 shared')
    typed_equal(closed['e360_generated_local_limit_s'],1589,'historical grants3 local')
    typed_equal(closed['generated_shared_cumulative_used_s'],653.9810156000001,'actual all101 shared used')
    typed_equal(closed['e360_generated_local_used_s'],543.024,'actual all101 local used')
    return closed

def validate_completed_p20_pass(pass_id,passed,debits,*,root=ROOT):
    closed=completed_historical_p20_proof(root)
    typed_equal(pass_id,'e360-generous-p20','immutable completed P20 only')
    typed_equal(passed,closed['e360_passes'][pass_id],'actual historical P20 pass')
    typed_equal(debits[:101],closed['debits'],'actual all101 prefix')

@historical_dependencies()
def historical_p19_repair_policy(root=ROOT):
    raw=read_spec(P19_FROZEN_POLICY_SPEC,root);enroll_json_descriptor(P19_FROZEN_POLICY_SPEC,root)
    return strict_json(raw)

@historical_dependencies()
def historical_p19_source_freeze(root=ROOT):
    raw=read_spec(P19_FROZEN_FREEZE_SPEC,root);enroll_json_descriptor(P19_FROZEN_FREEZE_SPEC,root)
    freeze=strict_json(raw)
    typed_equal(freeze['approved_literal'],P19_ORIGINAL_POLICY_SPEC,'historical P19 logical policy descriptor')
    require(len(freeze['source_copies'])==4 and len(freeze['own_sources_sha256'])==9 and len(freeze['all_P19_Native_sources'])==4,'historical P19 Source4/own9/Native4')
    changed={row['current']['path']:row for row in freeze['source_copies']}
    for row in changed.values():
        for key in ('bytes','sha256'):typed_equal(row['current'][key],row['frozen'][key],'historical P19 Source4 frozen bytes')
        typed_equal(row['current']['sha256'],freeze['own_sources_sha256'][row['current']['path']],'historical P19 own source')
        read_spec(row['frozen'],root);enroll_json_descriptor(row['frozen'],root)
    for name,digest in freeze['own_sources_sha256'].items():
        if name not in changed:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'unchanged own5')
    for spec in (*freeze['all_P19_Native_sources'],freeze['collector_source']):read_spec(spec,root);enroll_json_descriptor(spec,root)
    return freeze

@historical_dependencies()
def completed_historical_p19_proof(root=ROOT):
    raw=read_spec(P19_CLOSED100_SPEC,root);enroll_json_descriptor(P19_CLOSED100_SPEC,root)
    closed=strict_json(raw);prior=completed_historical_p18_proof(root);freeze=historical_p19_source_freeze(root)
    require((len(closed['debits']),len(closed['e360_calls']),len(closed['e360_passes']))==(100,50,19),'actual100/50/19')
    typed_equal(closed['debits'][:99],prior['debits'],'exact prior99 costs')
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(closed[key][identity],item,'all prior99 state')
    for key in prior:
        if key.endswith(('_activation','_amendment')) or key in ('e360_resource_grant','e360_suite_recovery_resource_grant'):
            typed_equal(closed[key],prior[key],'all old pairs and grants2 retained')
    require(validate_p19_activation_triple(closed,root),'genuine historical P19 allocation300 triple')
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):typed_equal(closed[key],{},'closed100 ownership0')
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(closed[key],0,'closed100 original0')
    passed=closed['e360_passes']['e360-generous-p19']
    typed_equal(passed['source_sha256'],freeze['own_sources_sha256'],'actual frozen P19 own9')
    typed_equal(passed['next_index'],1,'actual P19 one Ruff return')
    typed_equal(passed['failed_cap_s'],3,'actual P19 Ruff cap3')
    typed_equal(passed['failed_cost'],P19_ACTUAL_RECORD_SPECS['full-return.json'],'actual P19 failed Ruff')
    changed={row['current']['path']:row['frozen'] for row in freeze['source_copies']}
    for spec in passed['concrete_fix']:
        if spec['path'] in changed:
            frozen=changed[spec['path']]
            for key in ('bytes','sha256'):typed_equal(spec[key],frozen[key],'exact logical P19 Source descriptor')
            read_spec(frozen,root)
        else:read_spec(spec,root)
    for spec in P19_ACTUAL_RECORD_SPECS.values():
        historical_p12_stream_bytes(spec,root) if spec['bytes']==0 else read_spec(spec,root)
    debit=closed['debits'][99];cost=strict_json(read_spec(debit['descriptor'],root));actual=strict_json(read_spec(cost['actual_tool_return'],root))
    typed_equal(debit['descriptor'],P19_ACTUAL_RECORD_SPECS['full-return.json'],'actual saved P19 once debit')
    typed_equal(debit['actual_outer_full_return_s'],1.601,'actual genuine P19 Ruff full return')
    typed_equal(debit['actual_returned_exit_code'],1,'actual genuine P19 Ruff exit1')
    typed_equal(cost['source_sha256'],freeze['own_sources_sha256'],'actual P19 cost own9')
    request=paid_request_identity(debit,cost,actual,closed,root=root)
    call=closed['e360_calls'][cost['call_id']]
    proof=p19_collector_proof(actual['collector_trace'],request,call['request'],1.601,1,root=root)
    typed_equal(proof['valid'],True,'actual P19 collector valid failed Ruff')
    typed_equal(closed['generated_shared_cumulative_limit_s'],1700,'historical grants3 shared')
    typed_equal(closed['e360_generated_local_limit_s'],1589,'historical grants3 local')
    typed_equal(closed['generated_shared_cumulative_used_s'],652.3940156,'actual all100 shared used')
    typed_equal(closed['e360_generated_local_used_s'],541.437,'actual all100 local used')
    return closed

def validate_completed_p19_pass(pass_id,passed,debits,*,root=ROOT):
    frozen=completed_historical_p19_proof(root)
    typed_equal(pass_id,'e360-generous-p19','immutable completed P19 only')
    typed_equal(passed,frozen['e360_passes'][pass_id],'actual historical P19 pass')
    typed_equal(debits[:100],frozen['debits'],'actual all100 prefix')

@historical_dependencies()
def historical_p25_repair_policy(root=ROOT):
    raw=read_spec(P25_FROZEN_POLICY_SPEC,root);enroll_json_descriptor(P25_FROZEN_POLICY_SPEC,root)
    return strict_json(raw)


@historical_dependencies()
def historical_p25_source_freeze(root=ROOT):
    raw=read_spec(P25_FROZEN_FREEZE_SPEC,root);enroll_json_descriptor(P25_FROZEN_FREEZE_SPEC,root)
    freeze=strict_json(raw)
    typed_equal(freeze['approved_literal'],P25_ORIGINAL_POLICY_SPEC,'historical P25 original logical policy')
    typed_equal(freeze['current_policy_frozen'],P25_FROZEN_POLICY_SPEC,'historical P25 physical policy')
    require(len(freeze['source_copies'])==4 and len(freeze['own_sources_sha256'])==9 and len(freeze['all_P25_Native_sources'])==4,'historical P25 Source4/own9/Native4')
    changed={row['current']['path']:row for row in freeze['source_copies']}
    for row in changed.values():
        for key in ('bytes','sha256'):typed_equal(row['current'][key],row['frozen'][key],'historical P25 same bytes')
        typed_equal(row['current']['sha256'],freeze['own_sources_sha256'][row['current']['path']],'historical P25 own source')
        read_spec(row['frozen'],root);enroll_json_descriptor(row['frozen'],root)
    for name,digest in freeze['own_sources_sha256'].items():
        if name not in changed:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'unchanged own5')
    for spec in (*freeze['all_P25_Native_sources'],freeze['collector_source']):read_spec(spec,root);enroll_json_descriptor(spec,root)
    return freeze


def historical_p25_recovery_event(root=ROOT):
    policy=historical_p25_repair_policy(root);historical_p25_source_freeze(root)
    return {'schema':'flylab.m3.e360.p25-recovery-activation.v1','revalidation_amendment':copy.deepcopy(P25_ORIGINAL_POLICY_SPEC),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited105':P23_CLOSED105_SPEC,
        'previous_revalidation_amendment':copy.deepcopy(P23_ORIGINAL_POLICY_SPEC),'manual_generated_pass_id':'e360-generous-p25',
        'new_generated_grant_s':0,'human_authorization':policy['human_authorization'],'as_executed_source_freeze':P25_FROZEN_FREEZE_SPEC}

def validate_historical_p25_activation_pair(live,root=ROOT):
    require('e360_p25_recovery_resource_grant' not in live,'historical P25 grant0/no grant field')
    present=[key in live for key in HISTORICAL_P25_FIELDS]
    require(all(present) or not any(present),'complete historical P25 pair')
    if not all(present):return False
    typed_equal(live[HISTORICAL_P25_FIELDS[0]],historical_p25_recovery_event(root),'exact frozen P25 event9')
    typed_equal(live[HISTORICAL_P25_FIELDS[1]],P25_ORIGINAL_POLICY_SPEC,'historical P25 logical authority')
    return True

@historical_dependencies()
def completed_historical_p25_proof(root=ROOT):
    """Two actual roles, direct closed103; no frozen module or invented scanner output."""
    raw=read_spec(P25_CLOSED107_SPEC,root);enroll_json_descriptor(P25_CLOSED107_SPEC,root)
    closed=strict_json(raw);prior=completed_historical_p23_proof(root);freeze=historical_p25_source_freeze(root)
    require((len(closed['debits']),len(closed['e360_calls']),len(closed['e360_passes']))==(107,57,24),'actual103/53/21')
    typed_equal(closed['debits'][:105],prior['debits'],'all prior101 costs')
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(closed[key][identity],item,'all prior101 state')
    for key in prior:
        if key.endswith(('_activation','_amendment')) or key in ('e360_resource_grant','e360_suite_recovery_resource_grant','e360_p19_recovery_resource_grant') or key.startswith(('original_','e360_original_','e359_original_')):
            typed_equal(closed[key],prior[key],'all prior pairs/grants3/original counters')
    require(validate_historical_p25_activation_pair(closed,root),'actual P25 pair2/grant0')
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):typed_equal(closed[key],{},'closed103 ownership0')
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(closed[key],0,'closed103 original0')
    passed=closed['e360_passes']['e360-generous-p25']
    typed_equal(passed['source_sha256'],freeze['own_sources_sha256'],'actual frozen P25 own9')
    typed_equal(passed['revalidation_amendment'],P25_ORIGINAL_POLICY_SPEC,'actual P25 original policy')
    typed_equal(passed['next_index'],2,'actual P25 two returned roles')
    typed_equal(passed['failed_cap_s'],30,'actual P25 failed scanner cap30')
    typed_equal(passed['failed_cost'],P25_ACTUAL_SCANNER_RECORD_SPECS['full-return.json'],'actual P25 failed scanner')
    typed_equal(passed['predecessor_failed_cap_s'],3,'P25 preserved predecessor Ruff cap3')
    typed_equal(passed['actual_failure'],P23_ACTUAL_RUFF_RECORD_SPECS['full-return.json'],'P25 preserved actual P20 Ruff')
    for spec in passed['concrete_fix']:historical_fix_bytes(spec,pass_id='e360-generous-p25',root=root)
    for spec in (*P25_ACTUAL_RUFF_RECORD_SPECS.values(),*P25_ACTUAL_SCANNER_RECORD_SPECS.values()):
        historical_p12_stream_bytes(spec,root) if spec['bytes']==0 else read_spec(spec,root)
    for index,records,role,seconds,code,cap in ((105,P25_ACTUAL_RUFF_RECORD_SPECS,'ruff',1.695,0,3),(106,P25_ACTUAL_SCANNER_RECORD_SPECS,'scanner',4.716,1,30)):
        debit=closed['debits'][index];cost=strict_json(read_spec(debit['descriptor'],root));actual=strict_json(read_spec(records['actual.json'],root))
        typed_equal(debit['descriptor'],records['full-return.json'],'actual P25 once role debit')
        typed_equal(debit['actual_outer_full_return_s'],seconds,'actual P25 full return')
        typed_equal(debit['actual_returned_exit_code'],code,'actual P25 exit')
        typed_equal(cost['source_sha256'],freeze['own_sources_sha256'],'actual P25 role own9')
        typed_equal(cost['call_role'],role,'actual P25 different role')
        typed_equal(cost['internal_completion_valid'],True,'valid metadata preserves actual exit result')
        validate_cost_authority(index,cost,actual,root=root)
        request=paid_request_identity(debit,cost,actual,closed,root=root);call=closed['e360_calls'][cost['call_id']]
        typed_equal(call['reserved_s'],cap,'actual P25 role cap')
        typed_equal(call['request'],records['final-request.json'],'actual admitted P25 request')
        typed_equal(actual['collector_trace'],records['ROOT-collector-trace.json'],'actual P25 saved trace')
        proof=historical_p25_collector_proof(actual['collector_trace'],request,call['request'],seconds,code,root=root)
        typed_equal(proof['valid'],True,'actual P25 first FINAL collector valid')
        if role=='scanner':
            typed_equal(cost['expected_output'],None,'actual scanner produced no report')
            typed_equal(cost['internal_completion'],None,'actual scanner no completion object')
    typed_equal(closed['generated_shared_cumulative_limit_s'],1700,'historical grants3 shared')
    typed_equal(closed['e360_generated_local_limit_s'],1589,'historical grants3 local')
    typed_equal(closed['generated_shared_cumulative_used_s'],672.5270156,'actual all103 shared')
    typed_equal(closed['e360_generated_local_used_s'],561.57,'actual all103 local')
    return closed

def validate_completed_p25_pass(pass_id,passed,debits,*,root=ROOT):
    closed=completed_historical_p25_proof(root)
    typed_equal(pass_id,'e360-generous-p25','immutable completed P25 only')
    typed_equal(passed,closed['e360_passes'][pass_id],'actual historical P25 pass')
    typed_equal(debits[:107],closed['debits'],'actual all103 prefix')

def historical_p25_collector_proof(trace_spec,request,request_spec,seconds,exit_code,*,root=ROOT):
    return p19_collector_proof(trace_spec,request,request_spec,seconds,exit_code,root=root,version='p25')

_COMPLETED_STREAM_PINS=(*_COMPLETED_STREAM_PINS,P25_ACTUAL_RUFF_RECORD_SPECS['stdout.txt'],P25_ACTUAL_RUFF_RECORD_SPECS['stderr.txt'])

@historical_dependencies()
def historical_p27_repair_policy(root=ROOT):
    return strict_json(read_spec(P27_FROZEN_POLICY_SPEC,root))

@historical_dependencies()
def historical_p27_source_freeze(root=ROOT):
    freeze=strict_json(read_spec(P27_FROZEN_FREEZE_SPEC,root))
    typed_equal(freeze['approved_literal'],P27_ORIGINAL_POLICY_SPEC,'historical P27 original logical policy')
    typed_equal(freeze['current_policy_frozen'],P27_FROZEN_POLICY_SPEC,'historical P27 physical policy')
    require(len(freeze['source_copies'])==4 and len(freeze['own_sources_sha256'])==9
        and len(freeze['all_P27_Native_sources'])==4,'historical P27 Source4/own9/Native4')
    changed={row['current']['path']:row for row in freeze['source_copies']}
    for row in changed.values():
        for key in ('bytes','sha256'):typed_equal(row['current'][key],row['frozen'][key],'exact P27 frozen Source alias')
        typed_equal(row['current']['sha256'],freeze['own_sources_sha256'][row['current']['path']],'historical P27 own source')
        read_spec(row['frozen'],root)
    for name,digest in freeze['own_sources_sha256'].items():
        if name not in changed:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'unchanged shared own5')
    for spec in (*freeze['all_P27_Native_sources'],freeze['collector_source']):read_spec(spec,root)
    return freeze

def historical_p27_recovery_event(root=ROOT):
    policy=historical_p27_repair_policy(root)
    return {'schema':'flylab.m3.e360.p27-recovery-activation.v1','revalidation_amendment':copy.deepcopy(P27_ORIGINAL_POLICY_SPEC),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited107':P25_CLOSED107_SPEC,
        'previous_revalidation_amendment':copy.deepcopy(P25_ORIGINAL_POLICY_SPEC),'manual_generated_pass_id':'e360-generous-p27',
        'new_generated_grant_s':0,'human_authorization':policy['human_authorization'],'as_executed_source_freeze':P27_FROZEN_FREEZE_SPEC}

def validate_historical_p27_activation_pair(live,root=ROOT):
    require('e360_p27_recovery_resource_grant' not in live,'historical P27 no grant field')
    present=[key in live for key in HISTORICAL_P27_FIELDS]
    require(all(present) or not any(present),'complete historical P27 pair2')
    if not all(present):return False
    typed_equal(live[HISTORICAL_P27_FIELDS[0]],historical_p27_recovery_event(root),'actual frozen P27 event9')
    typed_equal(live[HISTORICAL_P27_FIELDS[1]],P27_ORIGINAL_POLICY_SPEC,'historical P27 logical authority')
    return True

@historical_dependencies()
def completed_historical_p27_proof(root=ROOT):
    """Direct failed Ruff plus separately committed ownership closure; no recursive metadata reads."""
    prior=completed_historical_p25_proof(root)
    closed=strict_json(read_spec(P27_CLOSED108_SPEC,root));held=strict_json(read_spec(P27_HELD108_SPEC,root))
    freeze=historical_p27_source_freeze(root);policy=historical_p27_repair_policy(root)
    require((len(closed['debits']),len(closed['e360_calls']),len(closed['e360_passes']))==(108,58,25),'actual108/58/25')
    typed_equal(closed['debits'][:107],prior['debits'],'unchanged whole107 prefix')
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(closed[key][identity],item,'all actual prior107 state')
    for key in prior:
        if key.endswith(('_activation','_amendment')) or key in ('e360_resource_grant','e360_suite_recovery_resource_grant','e360_p19_recovery_resource_grant') or key.startswith(('original_','e360_original_','e359_original_')):
            typed_equal(closed[key],prior[key],'prior pairs/grants3/Original counters')
    require(validate_historical_p27_activation_pair(closed,root),'actual historical P27 pair2/grant0')
    # Read only the declared27 physical inputs under explicit completed-history ownership.
    for spec in P27_DIRECT_PHYSICAL_SPECS.values():
        historical_p12_stream_bytes(spec,root) if spec['bytes']==0 else read_spec(spec,root)
    passed=closed['e360_passes']['e360-generous-p27'];records=P27_ACTUAL_RUFF_RECORD_SPECS
    typed_equal(passed['source_sha256'],freeze['own_sources_sha256'],'actual frozen P27 own9')
    typed_equal(passed['revalidation_amendment'],P27_ORIGINAL_POLICY_SPEC,'actual original P27 policy')
    typed_equal(passed['next_index'],1,'P27 actual one Ruff return')
    typed_equal(passed['failed_cap_s'],3,'P27 own failed Ruff cap3')
    typed_equal(passed['failed_cost'],records['full-return.json'],'P27 own failed cost')
    typed_equal(passed['predecessor_failed_cap_s'],30,'P27 preserved P25 scanner cap30')
    typed_equal(passed['actual_failure'],P25_ACTUAL_SCANNER_RECORD_SPECS['full-return.json'],'P27 preserved P25 scanner predecessor')
    debit=closed['debits'][107];cost=strict_json(read_spec(records['full-return.json'],root))
    actual=strict_json(read_spec(records['actual.json'],root));capture=strict_json(read_spec(records['capture-return.json'],root))
    manifest=strict_json(read_spec(records['output.json'],root))
    typed_equal(debit['descriptor'],records['full-return.json'],'actual once cost index107')
    typed_equal(debit['actual_outer_full_return_s'],1.72,'actual genuine Ruff duration')
    typed_equal(debit['actual_returned_exit_code'],1,'actual failed Ruff exit1')
    typed_equal(cost['source_sha256'],freeze['own_sources_sha256'],'historical P27 cost own9')
    typed_equal(cost['internal_completion_valid'],False,'immutable admin capture argument failure')
    typed_equal(cost['capture_proof_errors'],[{'stage':'ROOT-capture-proof','message':'Typed metadata descriptor'}],'original saved capture error')
    for key in ('internal_completion','ruff_exit_code','capture_complete','capture_publication_verified','capture_owned_state',
        'capture_process_identity','capture_root_tool_exit_required','capture_first_failure'):
        typed_equal(cost[key],None,'original malformed capture direct null field')
    typed_equal(cost['capture_return'],capture,'saved malformed summary value remains immutable')
    typed_equal(capture['process_identity'],{'process_id':8976,'start_utc_ticks':639268983001446681},'exact actual Int64 identity')
    require('schema' not in actual and 'schema' not in capture and 'pass_id' not in capture,'actual return/capture shape')
    validate_cost_authority(107,cost,actual,root=root)
    request=paid_request_identity(debit,cost,actual,closed,root=root);call=closed['e360_calls'][cost['call_id']]
    typed_equal(request['concrete_fix'],passed['concrete_fix'],'original admitted P27 concrete fix metadata')
    typed_equal(call['reserved_s'],3,'actual P27 Ruff3')
    typed_equal(call['request'],records['final-request.json'],'actual P27 admitted request')
    typed_equal(actual['collector_trace'],records['ROOT-collector-trace.json'],'actual P27 own saved trace')
    collector=historical_p27_collector_proof(actual['collector_trace'],request,call['request'],1.72,1,root=root)
    typed_equal(collector['valid'],True,'genuine collector valid failed Ruff')
    typed_equal(cost['collector_proof_valid'],True,'saved flat collector validity is not lint success')
    typed_equal(actual['collector_proof_valid'],True,'saved actual flat validity')
    # No old mutable logical alias or nested43/review3 becomes a physical read authority.
    prepared=strict_json(read_spec(P27_DIRECT_PHYSICAL_SPECS['prepared'],root))
    result=strict_json(read_spec(P27_DIRECT_PHYSICAL_SPECS['result'],root))
    returned=strict_json(read_spec(P27_DIRECT_PHYSICAL_SPECS['genuine_return'],root))
    operation_freeze=strict_json(read_spec(P27_DIRECT_PHYSICAL_SPECS['operation_freeze'],root))
    logical_before={**P27_HELD108_SPEC,'path':BASE+'/parent-e358-execution-ledger.json'}
    logical_after={**P27_CLOSED108_SPEC,'path':BASE+'/parent-e358-execution-ledger.json'}
    typed_equal(prepared['status'],'VERIFIED_NOT_COMMITTED','prepared phase remains prior')
    for value in (prepared['before'],result['before'],operation_freeze['before_live']):typed_equal(value,logical_before,'old logical before maps to physical held108')
    typed_equal(prepared['held_snapshot'],P27_HELD108_SPEC,'physical held alias')
    typed_equal(prepared['attempted_live_candidate'],logical_after,'logical after maps to physical closed108')
    typed_equal(result['live'],logical_after,'committed logical live alias')
    typed_equal(result['closed108'],P27_CLOSED108_SPEC,'actual closed physical alias')
    for value in (prepared['candidate'],result['candidate']):
        typed_equal(value['path'],BASE+'/parent-e360-p27-ruff-hold-candidate-ledger108.json','candidate metadata alias only')
        for key in ('bytes','sha256'):typed_equal(value[key],P27_CLOSED108_SPEC[key],'candidate exact closed bytes')
    typed_equal(prepared['input_descriptors'],P27_CLOSURE_METADATA['input_descriptors'],'nested43 exact metadata only')
    for key in ('actual_ROOT_FIRST','actual_ROOT_SECOND','actual_Sol_FIRST'):
        typed_equal(operation_freeze[key],P27_CLOSURE_METADATA[key],'closure review pins metadata only')
    for key in ('runtime','actual_production_proofs','closure'):typed_equal(operation_freeze[key],'NOT_RUN','prior operation freeze phase retained')
    for value in (prepared['operation_source'],result['operation_source'],operation_freeze['actual_ROOT_operation']):
        typed_equal(value,P27_DIRECT_PHYSICAL_SPECS['operation'],'exact physical reviewed operation')
    typed_equal(prepared['actual_production_capture']['valid'],True,'genuine separate capture proof')
    typed_equal(prepared['actual_production_capture']['unknown'],False,'genuine separate known ownership')
    typed_equal(prepared['actual_production_capture']['manifest'],manifest,'actual manifest from capture proof')
    typed_equal(prepared['actual_production_collector']['valid'],True,'genuine separate collector proof')
    typed_equal(prepared['historical_internal_completion_valid'],False,'prepared preserves failed saved field')
    typed_equal(result['historical_internal_completion_valid'],False,'result preserves failed saved field')
    typed_equal(result['status'],'COMMITTED_READBACK_VERIFIED','actual separate commit phase')
    typed_equal(result['sole_removed_hold'],'e360-generous-p27-ruff','only named hold removed')
    typed_equal(result['hold_count'],0,'actual committed ownership0')
    native_final=returned['first_FINAL']['result']
    typed_equal(native_final['exit_code'],0,'genuine closure first FINAL exit0')
    typed_equal(native_final['chunk_id'],'6f169e','actual genuine closure chunk')
    final_value=strict_json(native_final['output'].encode('utf8'))
    typed_equal(final_value['result'],P27_DIRECT_PHYSICAL_SPECS['result'],'genuine result publication link')
    typed_equal(final_value['closed108'],P27_CLOSED108_SPEC,'genuine closed publication link')
    typed_equal(returned['full_outer_elapsed_s'],None,'unavailable full outer clock stays unavailable')
    expected=copy.deepcopy(held)
    require(set(expected['e360_blocking_holds'])=={'e360-generous-p27-ruff'},'exact sole actual held identity')
    original_hold=expected['e360_blocking_holds'].pop('e360-generous-p27-ruff')
    typed_equal(original_hold['full_exit'],records['full-return.json'],'original hold debit link')
    typed_equal(original_hold['capture_return'],capture,'original malformed hold summary')
    typed_equal(original_hold['numeric_reserved_s'],0,'zero numeric hold')
    typed_equal(original_hold['debit_backed'],True,'once charged debit-backed hold')
    typed_equal(prepared['original_hold'],original_hold,'whole saved hold immutable')
    typed_equal(closed,expected,'only named hold removed; every other typed value unchanged')
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):typed_equal(closed[key],{},'closed108 ownership0')
    typed_equal(closed['generated_shared_cumulative_used_s'],674.2470156,'actual108 shared, no reset')
    typed_equal(closed['e360_generated_local_used_s'],563.29,'actual108 local, no reset')
    typed_equal(policy['predecessor_failed_cap_s'],30,'frozen P27 predecessor remains scanner30')
    return closed

def validate_completed_p27_pass(pass_id,passed,debits,*,root=ROOT):
    closed=completed_historical_p27_proof(root)
    typed_equal(pass_id,'e360-generous-p27','immutable completed P27 only')
    typed_equal(passed,closed['e360_passes'][pass_id],'actual historical P27 pass')
    typed_equal(debits[:108],closed['debits'],'actual all108 prefix')

@historical_dependencies()
def historical_p27_collector_proof(trace_spec,request,request_spec,seconds,exit_code,*,root=ROOT):
    return p19_collector_proof(trace_spec,request,request_spec,seconds,exit_code,root=root,version='p27')

_COMPLETED_STREAM_PINS=(*_COMPLETED_STREAM_PINS,P27_ACTUAL_RUFF_RECORD_SPECS['stdout.txt'],P27_ACTUAL_RUFF_RECORD_SPECS['stderr.txt'],P29_ACTUAL_RUFF_RECORD_SPECS['stdout.txt'],P29_ACTUAL_RUFF_RECORD_SPECS['stderr.txt'])


@historical_dependencies()
def historical_p29_repair_policy(root=ROOT):
    return strict_json(read_spec(P29_FROZEN_POLICY_SPEC,root))

@historical_dependencies()
def historical_p29_source_freeze(root=ROOT):
    freeze=strict_json(read_spec(P29_FROZEN_FREEZE_SPEC,root))
    typed_equal(freeze['approved_literal'],P29_ORIGINAL_POLICY_SPEC,'P29 original logical policy')
    typed_equal(freeze['current_policy_frozen'],P29_FROZEN_POLICY_SPEC,'P29 physical policy')
    require(len(freeze['source_copies'])==4 and len(freeze['own_sources_sha256'])==9
        and len(freeze['all_P29_Native_sources'])==4,'P29 Source4/own9/Native4')
    changed={row['current']['path']:row for row in freeze['source_copies']}
    for row in changed.values():
        for key in ('bytes','sha256'):typed_equal(row['current'][key],row['frozen'][key],'P29 exact frozen alias')
        typed_equal(row['current']['sha256'],freeze['own_sources_sha256'][row['current']['path']],'P29 own source')
        read_spec(row['frozen'],root)
    for name,digest in freeze['own_sources_sha256'].items():
        if name not in changed:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'shared own5')
    for spec in (*freeze['all_P29_Native_sources'],freeze['collector_source']):read_spec(spec,root)
    return freeze

def historical_p29_recovery_event(root=ROOT):
    policy=historical_p29_repair_policy(root)
    return {'schema':'flylab.m3.e360.p29-recovery-activation.v1','revalidation_amendment':copy.deepcopy(P29_ORIGINAL_POLICY_SPEC),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited108':P27_CLOSED108_SPEC,
        'previous_revalidation_amendment':copy.deepcopy(P27_ORIGINAL_POLICY_SPEC),'manual_generated_pass_id':'e360-generous-p29',
        'new_generated_grant_s':0,'human_authorization':policy['human_authorization'],'as_executed_source_freeze':P29_FROZEN_FREEZE_SPEC}

def validate_historical_p29_activation_pair(live,root=ROOT):
    require('e360_p29_recovery_resource_grant' not in live,'P29 no grant field')
    present=[key in live for key in HISTORICAL_P29_FIELDS]
    require(all(present) or not any(present),'complete historical P29 pair2')
    if not all(present):return False
    typed_equal(live[HISTORICAL_P29_FIELDS[0]],historical_p29_recovery_event(root),'actual P29 event9')
    typed_equal(live[HISTORICAL_P29_FIELDS[1]],P29_ORIGINAL_POLICY_SPEC,'P29 logical authority')
    return True

@historical_dependencies()
def historical_p29_collector_proof(trace_spec,request,request_spec,seconds,exit_code,*,root=ROOT):
    return p19_collector_proof(trace_spec,request,request_spec,seconds,exit_code,root=root,version='p29')

@historical_dependencies()
def completed_historical_p29_proof(root=ROOT):
    """One direct P29 success appended to the actual P27 paid prefix; metadata forests are not read."""
    prior=completed_historical_p27_proof(root)
    closed=strict_json(read_spec(P29_CLOSED109_SPEC,root));freeze=historical_p29_source_freeze(root)
    require((len(closed['debits']),len(closed['e360_calls']),len(closed['e360_passes']))==(109,59,26),'actual109/59/26')
    typed_equal(closed['debits'][:108],prior['debits'],'all actual108 costs preserved')
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(closed[key][identity],item,'all prior paid state')
    for key in prior:
        if key.endswith(('_activation','_amendment')) or key in ('e360_resource_grant','e360_suite_recovery_resource_grant','e360_p19_recovery_resource_grant') or key.startswith(('original_','e360_original_','e359_original_')):
            typed_equal(closed[key],prior[key],'prior pairs/grants3/Original counters')
    require(validate_historical_p29_activation_pair(closed,root),'historical P29 pair2/grant0')
    for spec in P29_DIRECT_PHYSICAL_SPECS.values():
        historical_p12_stream_bytes(spec,root) if spec['bytes']==0 else read_spec(spec,root)
    passed=closed['e360_passes']['e360-generous-p29'];records=P29_ACTUAL_RUFF_RECORD_SPECS
    typed_equal(passed,P29_ACTUAL_PASS_METADATA,'actual P29 whole pass preserves old P27 admission')
    typed_equal(passed['source_sha256'],freeze['own_sources_sha256'],'actual P29 frozen own9')
    debit=closed['debits'][108];cost=strict_json(read_spec(records['full-return.json'],root))
    actual=strict_json(read_spec(records['actual.json'],root));capture=strict_json(read_spec(records['capture-return.json'],root))
    typed_equal(debit['descriptor'],records['full-return.json'],'P29 once paid cost108')
    for value in (debit,cost,actual):
        typed_equal(value['actual_outer_full_return_s'],2.057,'genuine P29 full return')
        typed_equal(value['actual_returned_exit_code'],0,'successful paid Ruff')
    for value in (cost,actual):typed_equal(value['collector_proof_valid'],True,'saved collector true')
    typed_equal(cost['charged_once'],True,'P29 charged exactly once')
    typed_equal(cost['internal_completion'],None,'P29 null internal payload')
    typed_equal(cost['internal_completion_valid'],True,'P29 completion metadata true')
    typed_equal(cost['capture_return'],records['capture-return.json'],'correct descriptor layer')
    typed_equal(cost['capture_proof_errors'],[],'P29 actual capture errors empty')
    typed_equal(cost['source_sha256'],freeze['own_sources_sha256'],'P29 cost own9')
    require('schema' not in actual and 'schema' not in capture and 'pass_id' not in capture,'actual observed shapes')
    typed_equal(capture['process_identity'],{'process_id':12092,'start_utc_ticks':639269092748582997},'original Native Int64')
    for key in ('capture_complete','publication_verified','exit_observed','admission_complete'):typed_equal(capture[key],True,'successful capture '+key)
    validate_cost_authority(108,cost,actual,root=root)
    request=paid_request_identity(debit,cost,actual,closed,root=root);call=closed['e360_calls'][cost['call_id']]
    typed_equal(call['request'],records['final-request.json'],'P29 original request')
    typed_equal(call['reserved_s'],3,'P29 Ruff cap3')
    typed_equal(call['future_reserved_s'],885,'P29 primary/fallback suffix')
    proof=historical_p29_collector_proof(actual['collector_trace'],request,call['request'],2.057,0,root=root)
    typed_equal(proof['valid'],True,'genuine successful P29 collector')
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):typed_equal(closed[key],{},'closed109 ownership0')
    typed_equal(closed['generated_shared_cumulative_used_s'],676.3040156000001,'exact P29 shared scalar')
    typed_equal(closed['e360_generated_local_used_s'],565.347,'exact P29 local scalar')
    return closed

def validate_completed_p29_pass(pass_id,passed,debits,*,root=ROOT):
    closed=completed_historical_p29_proof(root)
    typed_equal(pass_id,'e360-generous-p29','immutable paid P29 only')
    typed_equal(passed,closed['e360_passes'][pass_id],'actual P29 pass unchanged')
    typed_equal(debits[:109],closed['debits'],'whole109 prefix')

def p30_source_failure_proof(root=ROOT):
    """Exact unpaid Source failure, never a failed Ruff or recursive read grant."""
    policy=current_repair_policy(root)
    with dependency_ownership(root,'dynamic'):
        failure=strict_json(read_spec(policy['unpaid_Source_admission_failure'],root,owner='dynamic'))
        returned=strict_json(read_spec(policy['unpaid_Source_genuine_return'],root,owner='dynamic'))
        error=strict_json(read_spec(policy['unpaid_Source_first_error'],root,owner='dynamic'))
    for key,value in {'schema':'flylab.m3.e360.p29.ROOT-scanner-native-admission-failure.v1','actual_genuine_exit':1,
        'scanner_registered':False,'scanner_paid_or_executed':False,'counts':[109,59,26],'ownership':[0,0,0]}.items():
        typed_equal(failure[key],value,'actual unpaid Source failure '+key)
    typed_equal(failure['genuine_return'],policy['unpaid_Source_genuine_return'],'original FINAL leaf')
    typed_equal(failure['original_ErrorRecord'],policy['unpaid_Source_first_error'],'original ErrorRecord leaf')
    typed_equal(failure['successful_paid_Ruff'],policy['paid_predecessor_success'],'success separate from Source failure')
    typed_equal(failure['actual_closed109'],{**P29_CLOSED109_SPEC,'path':BASE+'/parent-e358-execution-ledger.json'},'logical closed alias binds physical immutable109')
    typed_equal(failure['actual_Source_freeze'],P29_FROZEN_FREEZE_SPEC,'as-executed Source freeze metadata')
    typed_equal(failure['current_policy'],P29_FROZEN_POLICY_SPEC,'as-executed physical policy metadata')
    typed_equal(returned['original_first_FINAL']['exit_code'],1,'original unpaid FINAL exit1')
    typed_equal(error['first_error']['Exception_Message'],'Exact cost index/pass','actual Source first error')
    typed_equal(error['first_error']['FullyQualifiedErrorId'],'Exact cost index/pass','original FQID')
    return failure
def p30_recovery_event(root=ROOT):
    policy=historical_p30_repair_policy(root);historical_p30_source_freeze(root)
    return {'schema':'flylab.m3.e360.p30-recovery-activation.v1','revalidation_amendment':copy.deepcopy(CURRENT_REPAIR_SPEC),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited109':policy['closed_predecessor_ledger'],
        'previous_revalidation_amendment':copy.deepcopy(P29_ORIGINAL_POLICY_SPEC),'manual_generated_pass_id':'e360-generous-p30',
        'new_generated_grant_s':0,'human_authorization':CURRENT_HUMAN_AUTONOMY,
        'as_executed_source_freeze':file_spec(BASE+'/parent-e360-p30-root-final-freeze.json',root)}

def validate_p30_activation_pair(live,root=ROOT):
    require('e360_p30_recovery_resource_grant' not in live,'P30 cannot add a grant field')
    present=[key in live for key in CURRENT_P30_FIELDS]
    require(all(present) or not any(present),'complete current P30 pair2; no partial policy/event')
    if not all(present):return False
    typed_equal(live[CURRENT_P30_FIELDS[0]],p30_recovery_event(root),'exact current P30 grant0 event/freeze')
    typed_equal(live[CURRENT_P30_FIELDS[1]],CURRENT_REPAIR_SPEC,'exact current P30 policy')
    return True

def p30_recovery_activation_allowed(live,*,root=ROOT,lock_absent=True):
    closed=completed_historical_p29_proof(root);historical_p30_source_freeze(root)
    require(type(lock_absent) is bool and lock_absent and not any(key in live for key in CURRENT_P30_FIELDS),'once current pair2/no lock')
    typed_equal(live,closed,'whole closed105 unchanged before activation')
    p30_source_failure_proof(root)
    limits=historical_budget_authority(root,live=live)
    require(limits['generated_shared_limit_s']-live['generated_shared_cumulative_used_s']>=888
        and limits['generated_local_limit_s']-live['e360_generated_local_used_s']>=888,'whole888 fits actual retained grants3')
    return True

def p30_recovery_delta_valid(before,after,*,root=ROOT):
    require(set(after)==set(before)|set(CURRENT_P30_FIELDS),'only P30 pair2/no grant or reset')
    for key in before:typed_equal(after[key],before[key],'all closed105 fields/scalars/grants3 unchanged')
    require(validate_p30_activation_pair(after,root),'exact pair2 delta')
    return True

@historical_dependencies()
def completed_historical_p18_proof(root=ROOT):
    """Direct exact closed99 and three actual costs; no new recursive reader/history clone."""
    policy=historical_p19_repair_policy(root)
    raw=read_spec(policy['closed_predecessor_ledger'],root)
    enroll_json_descriptor(policy['closed_predecessor_ledger'],root)
    closed=strict_json(raw)
    require((len(closed['debits']),len(closed['e360_calls']),len(closed['e360_passes']))==(99,49,18),'actual99/49/18')
    old=historical_p18_repair_policy(root)
    prior=strict_json(read_spec(old['closed_predecessor_ledger'],root))
    typed_equal(closed['debits'][:96],prior['debits'],'all prior96 costs')
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(closed[key][identity],item,'direct retained96 state')
    for key in (*p17_recovery_amendment(root)['preserved_activation_fields'],*P17_RECOVERY_FIELDS,
        *p17_recovery_amendment(root)['preserved_resource_grant_fields']):typed_equal(closed[key],prior[key],'actual old pairs/grants2')
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):typed_equal(closed[key],{},'closed P18 ownership0')
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(closed[key],0,'P18 original0')
    freeze=strict_json(read_spec(policy['predecessor_as_executed_freeze'],root))
    enroll_json_descriptor(policy['predecessor_as_executed_freeze'],root)
    typed_equal(freeze['approved_literal'],P18_ORIGINAL_POLICY_SPEC,'actual P18 original mutable policy descriptor')
    require(len(freeze['source_copies'])==4 and len(freeze['own_sources_sha256'])==9,'actual P18 Source4/own9')
    for row in freeze['source_copies']:
        for key in ('bytes','sha256'):typed_equal(row['current'][key],row['frozen'][key],'actual P18 same-byte frozen4')
        typed_equal(row['current']['sha256'],freeze['own_sources_sha256'][row['current']['path']],'actual P18 original source descriptor')
        read_spec(row['frozen'],root);enroll_json_descriptor(row['frozen'],root)
    changed={row['current']['path']:row for row in freeze['source_copies']}
    for name,digest in freeze['own_sources_sha256'].items():
        if name not in changed:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'actual immutable own5')
    for spec in (*freeze['all_P18_Native_sources'],freeze['collector_source']):
        read_spec(spec,root);enroll_json_descriptor(spec,root)
    passed=closed['e360_passes']['e360-generous-p18']
    typed_equal(passed['source_sha256'],freeze['own_sources_sha256'],'actual P18 own9')
    typed_equal(passed['next_index'],3,'P18 three actual returned roles')
    typed_equal(passed['failed_cost'],policy['predecessor_failed_build_cost'],'actual P18 failed fbuild')
    typed_equal(passed['failed_cap_s'],25,'P18 actual fbuild cap30')
    for spec in passed['concrete_fix']:
        if spec['path'] in changed:
            typed_equal(spec,changed[spec['path']]['current'],'historical P18 current descriptor mapped exactly')
            read_spec(changed[spec['path']]['frozen'],root)
        else:read_spec(spec,root)
    for spec in P18_ACTUAL_RECORD_SPECS.values():read_spec(spec,root);enroll_json_descriptor(spec,root)
    for index,seconds,code in ((96,1.528,0),(97,19.212,0),(98,21.895,1)):
        debit=closed['debits'][index];cost=strict_json(read_spec(debit['descriptor'],root))
        actual=strict_json(read_spec(cost['actual_tool_return'],root))
        typed_equal(debit['actual_outer_full_return_s'],seconds,'actual P18 full return')
        typed_equal(debit['actual_returned_exit_code'],code,'actual P18 full exit')
        typed_equal(cost['source_sha256'],freeze['own_sources_sha256'],'actual frozen P18 cost Source9')
        request=paid_request_identity(debit,cost,actual,closed,root=root)
        call=closed['e360_calls'][cost['call_id']]
        proof=historical_p18_collector_proof(actual['collector_trace'],request,call['request'],seconds,code,root=root)
        typed_equal(proof['valid'],actual['collector_proof_valid'],'actual historical collector proof')
    typed_equal(closed['e360_p18_recovery_amendment'],P18_ORIGINAL_POLICY_SPEC,'actual P18 pair policy')
    expected_event={'schema':'flylab.m3.e360.p18-recovery-activation.v1','revalidation_amendment':copy.deepcopy(P18_ORIGINAL_POLICY_SPEC),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited96':old['closed_predecessor_ledger'],
        'previous_revalidation_amendment':old['historical_P17_literal'],'manual_generated_pass_id':'e360-generous-p18',
        'new_generated_grant_s':0,'human_authorization':CURRENT_HUMAN_AUTONOMY}
    typed_equal(closed['e360_p18_recovery_activation'],expected_event,'actual unchanged P18 event8')
    typed_equal(closed['generated_shared_cumulative_used_s'],650.7930156,'actual shared650')
    typed_equal(closed['e360_generated_local_used_s'],539.836,'actual local539')
    return closed

def p19_expected_allocation(root=ROOT):
    policy=historical_p19_repair_policy(root)
    historical_p19_source_freeze(root)
    return {'schema':'flylab.m3.e360.p19.ROOT-resource-allocation.v1','current_pass_id':'e360-generous-p19',
        'policy':copy.deepcopy(P19_ORIGINAL_POLICY_SPEC),'human_authorization':CURRENT_HUMAN_AUTONOMY,
        'closed_predecessor_ledger':policy['closed_predecessor_ledger'],
        'ROOT_reflected_design':policy['ROOT_reflected_design'],'actual_Astra_review':policy['actual_Astra_review'],
        'as_executed_source_freeze':file_spec(BASE+'/parent-e360-p19-root-final-freeze.json',root),
        'ROOT_review_cycle_complete':True,'old_shared_limit_s':1400,'old_local_limit_s':1289,
        'generated_new_grant_s':300,'new_shared_limit_s':1700,'new_local_limit_s':1589,'local_cost_offset':42,
        'actual_predecessor_shared_used_s':650.7930156,'actual_predecessor_local_used_s':539.836,
        'reset':False,'original_role_admission':False}

def validate_p19_allocation_document(document,expected):
    typed_equal(document,expected,'complete actual ROOT300 allocation')
    return document

def p19_activation_objects(allocation_spec,root=ROOT):
    policy=historical_p19_repair_policy(root)
    require(allocation_spec['path']==policy['allocation_record_path'],'exact current actual allocation namespace')
    allocation=strict_json(read_spec(allocation_spec,root));enroll_json_descriptor(allocation_spec,root)
    validate_p19_allocation_document(allocation,p19_expected_allocation(root))
    event={'schema':'flylab.m3.e360.p19-recovery-activation.v1','revalidation_amendment':copy.deepcopy(P19_ORIGINAL_POLICY_SPEC),
        'resource_extension':file_spec(RESOURCE_EXTENSION,root),'inherited99':policy['closed_predecessor_ledger'],
        'previous_revalidation_amendment':P18_ORIGINAL_POLICY_SPEC,'manual_generated_pass_id':'e360-generous-p19',
        'new_generated_grant_s':300,'human_authorization':CURRENT_HUMAN_AUTONOMY,'ROOT_resource_allocation':allocation_spec}
    grant={'schema':'flylab.m3.e360.p19-generated-resource-grant.v1','allocation':allocation_spec,
        'policy':copy.deepcopy(P19_ORIGINAL_POLICY_SPEC),'human_authorization':CURRENT_HUMAN_AUTONOMY,
        'closed99':policy['closed_predecessor_ledger'],'old_shared_s':1400,'old_local_s':1289,
        'new_shared_s':1700,'new_local_s':1589,'new_generated_grant_s':300,'reset':False}
    return event,grant

def validate_p19_activation_triple(live,root=ROOT):
    fields=(*HISTORICAL_P19_FIELDS,'e360_p19_recovery_resource_grant')
    present=[key in live for key in fields]
    require(all(present) or not any(present),'complete P19 activation/policy/actual grant; partial/grant-only denied')
    if not all(present):return False
    typed_equal(live[fields[1]],P19_ORIGINAL_POLICY_SPEC,'actual exact current P19 policy')
    event,grant=p19_activation_objects(live[fields[0]]['ROOT_resource_allocation'],root)
    typed_equal(live[fields[0]],event,'complete P19 event actual300')
    typed_equal(live[fields[2]],grant,'complete P19 one actual grant300')
    return True

def p19_recovery_event(root=ROOT):
    return p19_activation_objects(file_spec(historical_p19_repair_policy(root)['allocation_record_path'],root),root)[0]

def current_scope_authority(root=ROOT,*,live=None):
    context=execution_context(root)
    if context is not None:
        execution_ledger_state(context,root,live)
        return True
    return historical_p30_scope_authority(root,live=live)

def historical_p30_scope_authority(root=ROOT,*,live=None):
    if live is None:live=strict_json(bounded_metadata_bytes(registration(root)['shared_ledger_path'],root))
    if not validate_p30_activation_pair(live,root):return False
    closed=completed_historical_p29_proof(root);historical_p30_source_freeze(root)
    require(set(live)==set(closed)|set(CURRENT_P30_FIELDS),'whole109 plus only current pair2')
    typed_equal(len(closed['debits']),CURRENT_COST_IDENTITY['start_index'],'compiled current prefix equals actual closed109')
    typed_equal(live['debits'][:CURRENT_COST_IDENTITY['start_index']],closed['debits'],'immutable all109 costs')
    for key in closed:
        if key.endswith(('_activation','_amendment')) or key in ('e360_resource_grant','e360_suite_recovery_resource_grant','e360_p19_recovery_resource_grant'):
            typed_equal(live[key],closed[key],'all old pairs and grants3')
    for key in ('e360_calls','e360_passes'):
        for identity,item in closed[key].items():typed_equal(live[key][identity],item,'immutable actual109 state')
    return True

def current_p30_policy_view(root=ROOT):
    return policy_view(current_repair_policy(root),root)

def current_contract_bytes(value,root=ROOT):
    context=execution_bootstrap(root,ADMIN_CHECK)
    identity='e360-generous-p30' if context is None else context['manifest']['pass_id']
    require(type(value) is dict and value.get('pass_id')==identity and value.get('mode')=='generated','current contract transport identity')
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
    typed_equal(strict_json(raw),value,'complete current compact typed round trip')
    return raw

def current_contract_bound(name,raw,root=ROOT):
    path=safe(name,root);exists=path.exists()
    require(type(raw) is bytes,'current payload bytes')
    if exists or len(raw)>524288:
        raise ValueError('current contract publication bytes='+str(len(raw))+' cap=524288 path='+name+' exists='+str(exists).lower())
    return raw

def publish_current_contract(name,value,root=ROOT,check=None):
    raw=current_contract_bound(name,current_contract_bytes(value,root),root)
    durable(safe(name,root),raw,cap=524288,check=check)
    typed_equal(strict_json(bounded_json_read(file_spec(name,root),root,cap=524288,check=check)),value,'current actual publication typed readback')
    return file_spec(name,root)

def current_contract_bootstrap(name,entry,root=ROOT,stage='outer'):
    """Exact purpose namespace/policy/tuple; same ENTRY before full authority history."""
    limits={'fixture':{'outer_s':60,'child_s':55,'internal_s':51,'science_s':42},
        'suite':{'outer_s':520,'child_s':480,'internal_s':450,'science_s':360}}
    namespace_role='fixture' if name.endswith('-fixture.json') else 'suite' if name.endswith('-suite.json') else None
    require(namespace_role in limits,'current purpose role namespace')
    initial=OperationClock(entry,limits[namespace_role],stage);initial()
    require(type(name) is str and name.startswith(BASE+'/parent-e346-e360-e360-generous-p30-') and name.endswith('.json'),'current contract bootstrap namespace')
    path=safe(name,root);size=path.stat().st_size
    require(size<=524288,'bounded current contract bootstrap524288')
    with path.open('rb') as stream:raw=stream.read(524289)
    require(len(raw)<=524288,'bounded current contract bootstrap read524288')
    contract=strict_json(raw)
    require(contract.get('pass_id')=='e360-generous-p30' and contract.get('mode')=='generated' and contract.get('role') in limits,'exact current contract bootstrap pass/role')
    typed_equal(contract['role'],namespace_role,'same current purpose role')
    typed_equal(contract['revalidation_amendment'],CURRENT_REPAIR_SPEC,'exact current contract bootstrap authority')
    require(name==contract['paths']['contract'],'exact current contract bootstrap path')
    clock=OperationClock(entry,limits[contract['role']],stage);clock()
    enroll_json_descriptor(descriptor(name,raw),root)
    raw_policy=bounded_metadata_bytes(CURRENT_REPAIR_POLICY,root)
    require(len(raw_policy)==CURRENT_REPAIR_SPEC['bytes'] and sha(raw_policy)==CURRENT_REPAIR_SPEC['sha256'],'current bootstrap policy exact bytes/SHA')
    policy=strict_json(raw_policy);validate_current_p30_policy(policy)
    enroll_current_p30_input(CURRENT_REPAIR_SPEC,root)
    typed_equal({key:contract['limits'][key] for key in limits[contract['role']]},limits[contract['role']],'operation exact current immutable tuple')
    clock._admitted_root=Path(root).resolve()
    clock._admitted_clock=clock.timer.clock
    clock()
    return clock

def enroll_current_p30_input(spec,root=ROOT):
    """Explicit exact current input at real depth0; no cache or fake reader depth."""
    scope=_AUTHORITY_SCOPE
    if scope is None:
        require(any(spec==anchor for anchor in _CURRENT_P30_PINS),'only exact current policy/dependency enrollment')
    else:
        try:
            require(any(spec==anchor for anchor in _CURRENT_P30_PINS),'only exact current policy/dependency enrollment')
            scope.clock();require(not scope.failed,'failed current policy scope cannot be reused')
            scope._consumed(spec,root,owner='dynamic' if any(spec==anchor for anchor in _CURRENT_P30_DYNAMIC_PINS) else 'historical')
            scope.clock()
        except Exception:
            scope.poison();raise

def current_repair_policy(root=ROOT):
    context=execution_context(root)
    if context is not None:return copy.deepcopy(context['rules'])
    return historical_p30_repair_policy(root)

def historical_p30_repair_policy(root=ROOT):
    raw=read_spec(CURRENT_REPAIR_SPEC,root)
    enroll_current_p30_input(CURRENT_REPAIR_SPEC,root)
    value=strict_json(raw)
    validate_current_p30_policy(value)
    for spec in _CURRENT_P30_PINS[1:]:
        if spec['bytes']==0:historical_p12_stream_bytes(spec,root)
        else:
            read_spec(spec,root,owner='dynamic' if any(spec==anchor for anchor in _CURRENT_P30_DYNAMIC_PINS) else 'historical')
            enroll_current_p30_input(spec,root)
    human=strict_json(read_spec(value['human_authorization'],root))
    typed_equal(value['human_authorization'],CURRENT_HUMAN_AUTONOMY,'same general human authority')
    for key,item in {'per_attempt_human_reapproval_required':False,
        'in_scope_repair_and_new_attempt_selection_delegated':True,'new_grant_in_this_bootstrap':0}.items():
        typed_equal(human[key],item,'general human '+key)
    return value

def validate_current_p30_policy(value):
    """Narrow exact current policy; planned300 is not actual grant evidence."""
    typed_equal(value,_CURRENT_P30_EXPECTED_POLICY,'compiled current P19 exact typed policy')
    return value

def validate_saved_p17_record(name,document,root=ROOT):
    require(name in P17_SAVED_RECORD_SPECS,'known actual P17 record')
    typed_equal(document,strict_json(read_spec(P17_SAVED_RECORD_SPECS[name],root)),
        'actual P17 '+name+' exact typed record')


@historical_dependencies()
def completed_historical_p17_proof(root=ROOT):
    """Exact actual two-cost history, separate from current P18 policy/admission."""
    value=historical_p18_repair_policy(root)
    old=p17_recovery_amendment(root);prior=strict_json(read_spec(old['historical_P16_closed_ledger94'],root))
    frozen=strict_json(read_spec(value['closed_predecessor_ledger'],root))
    require(len(frozen['debits'])==96 and len(frozen['e360_calls'])==46 and len(frozen['e360_passes'])==17,'actual96/46/17')
    typed_equal(frozen['debits'][:94],prior['debits'],'all94 ordered costs')
    for key in (*old['preserved_activation_fields'],*old['preserved_resource_grant_fields']):
        typed_equal(frozen[key],prior[key],'all old activation/grant '+key)
    typed_equal(frozen[P17_RECOVERY_FIELDS[0]],p17_recovery_event(root),'actual historical P17 pair')
    typed_equal(frozen[P17_RECOVERY_FIELDS[1]],file_spec(P17_RECOVERY_AMENDMENT,root),'immutable P17 literal')
    for key in ('e360_calls','e360_passes'):
        for identity,item in prior[key].items():typed_equal(frozen[key][identity],item,'all94 prior state')
    for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations'):
        typed_equal(frozen[key],{},'actual P17 owned closure '+key)
    for key in ('original_jobs_running','original_reserved_s'):typed_equal(frozen[key],0,'original0')
    freeze=strict_json(read_spec(value['predecessor_as_executed_freeze'],root))
    own=freeze['own_sources_sha256'];passed=frozen['e360_passes']['e360-generous-p17']
    typed_equal(passed['source_sha256'],own,'actual P17 frozen own9')
    typed_equal(passed['failed_cap_s'],20,'actual scanner failed cap20')
    typed_equal(passed['next_index'],2,'actual two returned roles')
    typed_equal(passed['failed_cost'],P17_SAVED_RECORD_SPECS['scan_cost'],'actual failed scanner cost')
    require(len(own)==9 and len(freeze['source_copies'])==4 and len(freeze['all_P17_Native_sources'])==4,'actual Source4/own9/Native4')
    relocated={row['current']['path']:row for row in freeze['source_copies']}
    for row in freeze['source_copies']:
        for key in ('bytes','sha256'):typed_equal(row['current'][key],row['frozen'][key],'actual same-byte frozen4')
        typed_equal(row['current']['sha256'],own[row['current']['path']],'actual ownSource4')
        read_spec(row['frozen'],root)
    for name,digest in own.items():
        if name not in relocated:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'preserved actual own5')
    for spec in passed['concrete_fix']:
        if spec['path'] in relocated:
            typed_equal(spec,relocated[spec['path']]['current'],'actual historical Source4 concrete leaf')
            read_spec(relocated[spec['path']]['frozen'],root)
        else:read_spec(spec,root)
    for spec in (*freeze['all_P17_Native_sources'],freeze['collector_source']):read_spec(spec,root)
    for name,spec in P17_SAVED_RECORD_SPECS.items():validate_saved_p17_record(name,strict_json(read_spec(spec,root)),root)
    for index,label,seconds,code in ((94,'ruff',1.876,0),(95,'scan',19.509,1)):
        debit=frozen['debits'][index];cost=strict_json(read_spec(P17_SAVED_RECORD_SPECS[label+'_cost'],root))
        actual=strict_json(read_spec(P17_SAVED_RECORD_SPECS[label+'_actual'],root))
        typed_equal(debit['descriptor'],P17_SAVED_RECORD_SPECS[label+'_cost'],'actual once debit')
        typed_equal(debit['actual_outer_full_return_s'],seconds,'actual full return')
        typed_equal(debit['actual_returned_exit_code'],code,'actual returned exit')
        paid_request_identity(debit,cost,actual,frozen,root=root)
        typed_equal(cost['source_sha256'],own,'actual two-cost own9')
    for spec in _COMPLETED_STREAM_PINS:historical_p12_stream_bytes(spec,root)
    typed_equal(frozen['generated_shared_cumulative_used_s'],608.1580156,'actual shared608')
    typed_equal(frozen['e360_generated_local_used_s'],497.201,'actual local497')
    return frozen


def historical_p17_scope_authority(root=ROOT,*,live=None):
    if live is None:live=strict_json(bounded_metadata_bytes(registration(root)['shared_ledger_path'],root))
    present=[key in live for key in P17_RECOVERY_FIELDS]
    require(all(present) or not any(present),'complete historical P17 pair')
    if not all(present):return False
    frozen=completed_historical_p17_proof(root)
    typed_equal(live['debits'][:96],frozen['debits'],'actual96 preserved prefix')
    for key in P17_RECOVERY_FIELDS:typed_equal(live[key],frozen[key],'historical P17 exact pair')
    for key in ('e360_calls','e360_passes'):
        for identity,item in frozen[key].items():typed_equal(live[key][identity],item,'completed P17 retained state')
    return True


def p19_collector_proof(trace_spec,request,request_spec,seconds,exit_code,*,root=ROOT,version='p19',selected_context=None):
    """Replay immutable genuine transport metadata; never sample a new clock."""
    trace=strict_json(read_spec(trace_spec,root))
    typed_equal(trace['schema'],'flylab.m3.e360.'+version+'.ROOT-genuine-collector-state.v1','collector schema')
    selected=selected_context
    if version=='execution' and selected is None:selected=execution_context(root)
    typed_equal(request['pass_id'],selected['manifest']['pass_id'] if version=='execution' else 'e360-generous-'+version,'collector explicit context pass')
    for key in ('pass_id','call_id','call_role'):
        typed_equal(trace[key],request[key],'collector admitted '+key)
    typed_equal(trace['request'],request_spec,'collector exact admitted request')
    typed_equal(trace['collector_source'],request['collector_source'],'collector request source pin')
    typed_equal(trace['collector_source'],selected['freeze']['collector_source'] if version=='execution' else file_spec(BASE+'/parent-e360-'+version+'-ROOT-genuine-collector.txt',root),'collector explicit context source')
    typed_equal(trace['state_key'],('e360:'+request['pass_id']+':ROOT-genuine:' if version=='execution' else 'e360:'+version+':ROOT-genuine:')+request['call_id'],'unique collector role state')
    typed_equal(trace_spec['path'],BASE+'/parent-'+request['call_id']+'-ROOT-collector-trace.json','canonical trace publication')
    typed_equal(trace['next_role_allowed'],False,'collector cannot admit another role')
    typed_equal(trace['collection_status'],'FINAL_RECEIVED','first genuine FINAL collected independently of sticky error')
    typed_equal(trace['collection_stopped'],False,'no stopped transport may invent a later FINAL')
    typed_equal(trace['transport_error'],None,'genuine FINAL without a transport exception')
    caps={'ruff':3,'scanner':30,'fixture_build':30,'fixture_worker':60,'fixture_seal':20,
        'suite_build':40,'suite_worker':520,'suite_seal':25,'qualification':40}
    typed_equal(trace['cost_name'],request['call_id'],'collector once cost name')
    typed_equal(trace['cap_s'],caps[request['call_role']],'immutable role cap')
    require(type(trace['max_output_tokens']) is int and trace['max_output_tokens']>=1000,'complete output request bound')
    typed_equal(trace['exec_args'],{'cmd':"pwsh -NoProfile -File '"+BASE+
        ("/parent-execution-ROOT-dispatch-adapter.ps1' -CallRole '" if version=='execution' else "/parent-e360-"+version+"-ROOT-dispatch-adapter.ps1' -CallRole '")+request['call_role']+"'",
        'workdir':'C:/Users/ckthd/Dev/Projects/fly_lab'},'actual ROOT adopted collector dispatch body')
    dispatch=trace['dispatch_ms'];chunks=trace['chunks']
    require(type(dispatch) is int and dispatch>0 and type(chunks) is list and chunks,'genuine dispatch and chunks')
    last=dispatch;known=None;first=None;first_exit=None;complete=True
    for chunk in chunks:
        received=chunk['received_ms'];result=chunk['result']
        require(type(received) is int,'original integer receive clock')
        if received<last:
            require(trace['phase']=='ERROR_HOLD' and type(trace['first_error']) is dict,'clock contradiction retained first error')
            complete=False
        require(first is None,'no result after first genuine FINAL')
        last=received
        require(type(result) is dict,'contradictory transport cannot acquire a later FINAL')
        has_session=result.get('session_id') is not None
        has_exit=result.get('exit_code') is not None
        session=result.get('session_id');returned=result.get('exit_code')
        if has_session and type(session) is int and session>0:
            if known is None:known=session
            require(known==session,'changed session cannot acquire a later FINAL')
        if has_exit and type(returned) is int and not has_session:
            first,first_exit=received,returned
        else:require(has_session and type(session) is int and session>0 and not has_exit,
                     'contradictory result shape cannot acquire a later FINAL')
        output=result.get('output')
        if (type(output) is not str or result.get('truncated') is True or result.get('output_truncated') is True
            or re.search(r'Warning: truncated output|tokens truncated|output was truncated',output or '')):complete=False
    require(first is not None,'lost FINAL remains pending and cannot invent a debit')
    typed_equal(trace['session_id'],known,'first known tool session retained')
    typed_equal(trace['first_final_ms'],first,'immutable first FINAL sample')
    typed_equal(trace['exit_code'],first_exit,'immutable first FINAL exit')
    require(type(seconds) in (int,float) and not isinstance(seconds,bool) and math.isfinite(seconds) and seconds>0,'genuine positive full return')
    elapsed=(first-dispatch)/1000.0
    require(type(trace['full_return_s']) in (int,float) and math.isfinite(trace['full_return_s'])
            and trace['full_return_s']>0,'saved FINAL duration must be a finite positive JSON number')
    require(trace['full_return_s']==elapsed and seconds==elapsed,'exact dispatch to FINAL full return')
    typed_equal(exit_code,first_exit,'actual first FINAL exit')
    require(trace['phase'] in ('FINAL_RECEIVED','ERROR_HOLD'),'closed collector state')
    if trace['phase']=='ERROR_HOLD':
        require(type(trace['first_error']) is dict,'retain first collector failure')
        complete=False
    else:
        typed_equal(trace['first_error'],None,'complete collector has no first error')
        typed_equal(trace['publication'],'REQUIRED','Native publication follows saved FINAL')
    if not any(type(c['result']) is dict and type(c['result'].get('output')) is str
        and c['result']['output'].strip() for c in chunks):complete=False
    if not complete:require(trace['phase']=='ERROR_HOLD' and type(trace['first_error']) is dict,
                            'every collector failure retains the sticky original error')
    return {'valid':complete,'trace':trace_spec,'full_return_s':elapsed,'exit_code':first_exit}


def p20_collector_proof(trace_spec,request,request_spec,seconds,exit_code,*,root=ROOT):
    return p19_collector_proof(trace_spec,request,request_spec,seconds,exit_code,root=root,version='p20')

def p21_collector_proof(trace_spec,request,request_spec,seconds,exit_code,*,root=ROOT):
    return p19_collector_proof(trace_spec,request,request_spec,seconds,exit_code,root=root,version='p21')

def current_p30_unused_fallback(semantics,root=ROOT):
    """Reserve retained until the current pass has actual charged and owned closure."""
    identity=execution_current_pass(root) if 'execution_context_activation' in semantics else 'e360-generous-p30'
    passed=semantics['e360_passes'].get(identity)
    if passed is None:return 120
    rows=[row for row in semantics['e360_calls'].values() if row['pass_id']==identity]
    ended=rows and all(row['state']=='charged' for row in rows)
    owned=any(semantics[key] for key in ('e360_reservations','e360_pending_actuals','e360_blocking_holds'))
    terminal=passed.get('failed_cost') is not None or passed['next_index']==9
    return 0 if ended and terminal and not owned else 120

def validate_p30_call_future(call,*,root=ROOT,pass_id=None):
    roles=('ruff','scanner','fixture_build','fixture_worker','fixture_seal','suite_build','suite_worker','suite_seal','qualification')
    caps=(3,30,30,60,20,40,520,25,40);futures=(885,855,825,765,745,705,185,160,120)
    identity=execution_current_pass(root) if pass_id is None else pass_id
    require(identity in ('e360-generous-p30',execution_current_pass(root)) and call['pass_id']==identity and call['mode']=='generated' and call['call_role'] in roles,'current finite call future')
    index=roles.index(call['call_role'])
    typed_equal(call['reserved_s'],caps[index],'current primary call cap')
    typed_equal(call['future_reserved_s'],futures[index],'current primary suffix plus unused fallback exactly once')
    return caps[index]+futures[index]

def validate_p30_fallback_balance(balance,semantics,*,root=ROOT):
    require(type(balance) is int and 0<=balance<=120,'typed current unused fallback balance')
    typed_equal(balance,current_p30_unused_fallback(semantics,root),'current fallback once until valid closure')
    return balance

def validate_completed_p17_pass(pass_id,passed,debits,*,root=ROOT):
    frozen=completed_historical_p17_proof(root)
    typed_equal(pass_id,'e360-generous-p17','completed P17 only')
    typed_equal(passed,frozen['e360_passes'][pass_id],'actual completed P17 whole pass')
    typed_equal(debits[:96],frozen['debits'],'actual96 ordered prefix')


def validate_completed_p18_pass(pass_id,passed,debits,*,root=ROOT):
    closed=completed_historical_p18_proof(root)
    typed_equal(passed,closed['e360_passes'][pass_id],'actual immutable completed P18 pass')
    typed_equal(debits[:99],closed['debits'],'actual99 prefix')

def validate_p30_pass(pass_id,passed,debits,*,root=ROOT):
    value=current_repair_policy(root);frozen=completed_historical_p29_proof(root)
    typed_equal(pass_id,'e360-generous-p30','one current finite P19 no reopen/future/original')
    require(current_scope_authority(root),'actual P19 activation')
    require(passed['mode']=='generated' and passed['modified'] is True,'fresh modified generated P19')
    typed_equal(passed['paid_predecessor_success'],value['paid_predecessor_success'],'successful paid P29 Ruff')
    typed_equal(passed['unpaid_Source_admission_failure'],value['unpaid_Source_admission_failure'],'separate unpaid Native failure')
    typed_equal(passed['admission_kind'],value['current_admission_kind'],'current explicit transition kind')
    p30_source_failure_proof(root)
    typed_equal(passed['predecessor_success_cap_s'],3,'actual P23 Ruff cap3')
    typed_equal(passed['revalidation_amendment'],CURRENT_REPAIR_SPEC,'current compact authority')
    typed_equal(debits[:109],frozen['debits'],'all105 preserved')
    require(sum(row['descriptor']==passed['paid_predecessor_success'] for row in debits)==1,'previous failure once')
    typed_equal(passed['source_sha256'],own_sources(root),'fresh current exact own9')
    require(passed['source_sha256']!=frozen['e360_passes']['e360-generous-p29']['source_sha256'],'distinct Source9')
    for name in (*OWNED,BASE+'/parent-e360-root-account.ps1',BASE+'/parent-e360-root-metadata-helper.py',
        *(BASE+'/parent-e360-p30-'+part+'.ps1' for part in ('capture-proposed','prepare-native','bookkeeping-native','ROOT-dispatch-adapter')),
        BASE+'/parent-e360-p30-ROOT-genuine-collector.txt',value['ROOT_reflected_design']['path']):
        require(sum(spec==file_spec(name,root) for spec in passed['concrete_fix'])==1,'current exact concrete leaf '+name)
    for spec in passed['concrete_fix']:read_spec(spec,root)

def current_budget_authority(root=ROOT,*,live=None):
    context=execution_context(root)
    if context is not None:
        execution_ledger_state(context,root,live)
        return {'p19_recovery_active':True,'suite_recovery_active':True,'generated_shared_limit_s':1700,'generated_local_limit_s':1589}
    return historical_budget_authority(root,live=live)

def historical_budget_authority(root=ROOT,*,live=None):
    """Uncached nonrecursive live selector. Historical policy never supplies admission."""
    if live is None:
        raw=bounded_metadata_bytes(registration(root)['shared_ledger_path'],root)
        live=strict_json(raw)
    present=[key in live for key in SUITE_RECOVERY_FIELDS]
    require(all(present) or not any(present),'complete new pair and distinct grant including grant alone')
    if all(present):
        typed_equal(live[SUITE_RECOVERY_FIELDS[0]],suite_recovery_event(root),'exact typed P9 event8')
        typed_equal(live[SUITE_RECOVERY_FIELDS[1]],file_spec(SUITE_RECOVERY_AMENDMENT,root),'exact P9 live descriptor')
        typed_equal(live[SUITE_RECOVERY_FIELDS[2]],suite_recovery_grant(root),'exact typed P9 grant10')
        require('e360_execution_budget_activation' in live and 'e360_execution_budget_amendment' in live,'P9 preserves old live P8')
        typed_equal(live['e360_execution_budget_activation'],execution_budget_event(root),'old P8 activation retained')
        typed_equal(live['e360_execution_budget_amendment'],file_spec(EXECUTION_BUDGET_AMENDMENT,root),'old P8 direct authority retained')
    p19=validate_p19_activation_triple(live,root)
    p20=validate_historical_p20_activation_pair(live,root)
    p21=validate_historical_p21_activation_pair(live,root)
    p22=validate_historical_p22_activation_pair(live,root)
    p23=validate_historical_p23_activation_pair(live,root)
    p25=validate_historical_p25_activation_pair(live,root)
    p27=validate_historical_p27_activation_pair(live,root)
    p29=validate_historical_p29_activation_pair(live,root)
    p30=validate_p30_activation_pair(live,root)
    require(not p30 or (p29 and p27 and p25 and p23 and p22 and p21 and p20 and p19),'current P30 retains completed P20 pair and actual historical grants3')
    if p19:
        old=historical_p19_repair_policy(root)
        prior=strict_json(read_spec(old['closed_predecessor_ledger'],root))
        for key in ('e360_p18_recovery_activation','e360_p18_recovery_amendment','e360_resource_grant','e360_suite_recovery_resource_grant'):
            require(key in live,'P19 historical grant requires retained prior pair/grants2')
            typed_equal(live[key],prior[key],'historical grants3 preserve exact old pair/grants2')
    require(not p19 or all(present),'P19 retains original grants2')
    shared,local=(1700,1589) if p19 else (1400,1289) if all(present) else (600,489)
    typed_equal(live['generated_shared_cumulative_limit_s'],shared,'typed current shared limit')
    typed_equal(live['e360_generated_local_limit_s'],local,'typed current local limit')
    return {'p19_recovery_active':p19,'suite_recovery_active':all(present),'generated_shared_limit_s':shared,'generated_local_limit_s':local}

def authority_path_for_pass(pass_id):
    context=execution_context(ROOT)
    if context is not None and pass_id==context['manifest']['pass_id']:return context['manifest_spec']['path']
    paths={'e360-generous-p3':OLD_REVALIDATION_AMENDMENT,'e360-generous-p4':ACTIVE_REVALIDATION_AMENDMENT,
        'e360-generous-p5':SOURCE_RECOVERY_AMENDMENT,'e360-generous-p6':RUFF_RECOVERY_AMENDMENT,
        'e360-generous-p7':REVALIDATION_AMENDMENT,'e360-generous-p8':EXECUTION_BUDGET_AMENDMENT,
        'e360-generous-p9':SUITE_RECOVERY_AMENDMENT,'e360-generous-p10':LINT_RECOVERY_AMENDMENT,'e360-generous-p11':P11_RECOVERY_AMENDMENT,'e360-generous-p12':P12_RECOVERY_AMENDMENT,'e360-generous-p13':P13_RECOVERY_AMENDMENT,'e360-generous-p14':P14_RECOVERY_AMENDMENT,'e360-generous-p15':P15_RECOVERY_AMENDMENT,'e360-generous-p16':P16_RECOVERY_AMENDMENT,'e360-generous-p17':P17_RECOVERY_AMENDMENT,'e360-generous-p18':P18_FROZEN_POLICY_SPEC['path'],'e360-generous-p19':P19_FROZEN_POLICY_SPEC['path'],'e360-generous-p20':P20_FROZEN_POLICY_SPEC['path'],'e360-generous-p21':P21_FROZEN_POLICY_SPEC['path'],'e360-generous-p22':P22_FROZEN_POLICY_SPEC['path'],'e360-generous-p23':P23_FROZEN_POLICY_SPEC['path'],'e360-generous-p25':P25_FROZEN_POLICY_SPEC['path'],'e360-generous-p27':P27_FROZEN_POLICY_SPEC['path'],'e360-generous-p29':P29_FROZEN_POLICY_SPEC['path'],'e360-generous-p30':CURRENT_REPAIR_POLICY}
    if pass_id not in paths:
        require(type(pass_id) is str and not pass_id.startswith('e360-generous-p'),'no unknown generated authority')
        return REVALIDATION_AMENDMENT  # Existing finite generated test API.
    return paths[pass_id]

def effective_policy(root=ROOT,*,pass_id=None):
    context=execution_context(root)
    if context is not None and (pass_id is None or pass_id==context['manifest']['pass_id']):return execution_policy(context)
    if context is not None and pass_id=='e360-generous-p30':return policy_view(strict_json(read_spec(P30_FROZEN_POLICY_SPEC,root,owner='historical')),root)
    """Explicit immutable historical role caps; uncached current selection is separate."""
    if type(pass_id) is str and pass_id.startswith('e360-generous-p'):
        require(pass_id in (*('e360-generous-p'+str(i) for i in range(1,24)),'e360-generous-p25','e360-generous-p27','e360-generous-p29','e360-generous-p30'),'finite historical/current policy selector')
    if pass_id=='e360-generous-p18':return policy_view(historical_p18_repair_policy(root),root)
    if pass_id=='e360-generous-p19':return policy_view(historical_p19_repair_policy(root),root)
    if pass_id=='e360-generous-p20':return policy_view(historical_p20_repair_policy(root),root)
    if pass_id=='e360-generous-p21':return policy_view(historical_p21_repair_policy(root),root)
    if pass_id=='e360-generous-p22':return policy_view(historical_p22_repair_policy(root),root)
    if pass_id=='e360-generous-p23':return policy_view(historical_p23_repair_policy(root),root)
    if pass_id=='e360-generous-p25':return policy_view(historical_p25_repair_policy(root),root)
    if pass_id=='e360-generous-p27':return policy_view(historical_p27_repair_policy(root),root)
    if pass_id=='e360-generous-p29':return policy_view(historical_p29_repair_policy(root),root)
    if pass_id=='e360-generous-p30':
        require(current_scope_authority(root),'explicit current P19 activated selector')
        return current_p30_policy_view(root)
    if pass_id is None:
        candidate=strict_json(bounded_metadata_bytes(registration(root)['shared_ledger_path'],root))
        if any(key in candidate for key in CURRENT_P30_FIELDS):
            require(current_scope_authority(root,live=candidate),'uncached current P19 selector')
            return current_p30_policy_view(root)
    extension=resource_extension(root)
    recovery=False
    lint=False
    p11=False
    p17=False; p16=False
    p15=False
    p14=False
    p13=False
    p12=False
    if pass_id is None:
        live=strict_json(safe(registration(root)['shared_ledger_path'],root).read_bytes())
        recovery=current_budget_authority(root,live=live)['suite_recovery_active']
        lint=historical_lint_scope_authority(root,live=live)
        p11=bool(live.get(P11_RECOVERY_FIELDS[0]))
        p12=historical_p12_scope_authority(root,live=live)
        p13=historical_p13_scope_authority(root,live=live)
        p17=historical_p17_scope_authority(root,live=live);p16=historical_p16_scope_authority(root,live=live);p15=historical_p15_scope_authority(root,live=live);p14=historical_p14_scope_authority(root,live=live)
        require(('e360_execution_budget_activation' in live)==('e360_execution_budget_amendment' in live),'complete P8 activation pair')
        active='e360_execution_budget_activation' in live
        if active:
            typed_equal(live['e360_execution_budget_activation'],execution_budget_event(root),'P8 live activation')
            typed_equal(live['e360_execution_budget_amendment'],file_spec(EXECUTION_BUDGET_AMENDMENT,root),'P8 live direct pin')
    else:
        require(type(pass_id) is str and pass_id.startswith('e360-'),'finite pass ID')
        require(not pass_id.startswith('e360-generous-p') or pass_id in tuple('e360-generous-p'+str(i) for i in range(1,18)),'no unactivated/future attempt authority')
        active=pass_id=='e360-generous-p8'
        recovery=pass_id=='e360-generous-p9'
        lint=pass_id=='e360-generous-p10'
        p11=pass_id=='e360-generous-p11'
        p12=pass_id=='e360-generous-p12'
        p13=pass_id=='e360-generous-p13'
        p17=pass_id=='e360-generous-p17'
        p16=pass_id=='e360-generous-p16'
        p15=pass_id=='e360-generous-p15'
        p14=pass_id=='e360-generous-p14'
    if not active and not recovery and not lint and not p11 and not p12 and not p13 and not p14 and not p15 and not p16 and not p17:return extension
    value=p17_recovery_amendment(root) if p17 else p16_recovery_amendment(root) if p16 else p15_recovery_amendment(root) if p15 else p14_recovery_amendment(root) if p14 else p13_recovery_amendment(root) if p13 else p12_recovery_amendment(root) if p12 else p11_recovery_amendment(root) if p11 else lint_recovery_amendment(root) if lint else suite_recovery_amendment(root) if recovery else execution_budget_amendment(root)
    return {**extension,**{k:value[k] for k in ('effective_roles','scanner_limits','generated_call_order',
        'successful_suffix_after_s','primary_total_reserved_s','remaining_generated_administration_reserve_s',
        'fixture_future_administration_reserved_s','suite_future_administration_reserved_s')},
        'helper_limits':value['effective_helper_limits']}

def prospective_authority_spec(root=ROOT):
    context=execution_context(root)
    if context is not None:return copy.deepcopy(context['manifest_spec'])
    if current_scope_authority(root):return copy.deepcopy(CURRENT_REPAIR_SPEC)
    live=strict_json(safe(registration(root)['shared_ledger_path'],root).read_bytes())
    current=current_budget_authority(root,live=live)
    return file_spec(P17_RECOVERY_AMENDMENT if current_scope_authority(root,live=live) else P14_RECOVERY_AMENDMENT if historical_p14_scope_authority(root,live=live) else P13_RECOVERY_AMENDMENT if historical_p13_scope_authority(root,live=live) else P12_RECOVERY_AMENDMENT if historical_p12_scope_authority(root,live=live) else LINT_RECOVERY_AMENDMENT if historical_lint_scope_authority(root,live=live) else SUITE_RECOVERY_AMENDMENT if current['suite_recovery_active'] else EXECUTION_BUDGET_AMENDMENT if 'e360_execution_budget_activation' in live else REVALIDATION_AMENDMENT,root)


def compiled_checkpoint_policy():
    """Fresh compiled constants only; no authority, data, clock or observation cache."""
    return {'schema':CHECKPOINT_SCHEMA,'max_entries':CHECKPOINT_MAX_ENTRIES,'max_json_bytes':CHECKPOINT_MAX_JSON_BYTES,
        'generated_fixture_only':True,'genuine_parent_ENTRY':True,'reached_only':True,'monotone_elapsed':True,
        'authority_or_probe':False,'stage_names':list(CHECKPOINT_STAGES)}


def validate_execution_checkpoints(value,c,receipt,*,root=ROOT,successful=True):
    """Only reached P8 fixture phases, measured from its actual closed parent receipt."""
    pass_id=c.get('pass_id')
    require(pass_id in ('e360-generous-p8','e360-generous-p9','e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p13','e360-generous-p14','e360-generous-p15','e360-generous-p16','e360-generous-p17','e360-generous-p18','e360-generous-p19','e360-generous-p20',execution_current_pass(root)) and c['mode']=='generated' and c['role']=='fixture','only genuine admitted fixture checkpoints')
    authority={'checkpoint_policy':compiled_checkpoint_policy(),'effective_roles':effective_policy(root)['effective_roles']} if execution_is_current(pass_id,root) else {**p17_recovery_amendment(root),'effective_roles':current_repair_policy(root)['effective_roles']} if pass_id==execution_current_pass(root) else p17_recovery_amendment(root) if pass_id=='e360-generous-p17' else p16_recovery_amendment(root) if pass_id=='e360-generous-p16' else p15_recovery_amendment(root) if pass_id=='e360-generous-p15' else p14_recovery_amendment(root) if pass_id=='e360-generous-p14' else p13_recovery_amendment(root) if pass_id=='e360-generous-p13' else p12_recovery_amendment(root) if pass_id=='e360-generous-p12' else p11_recovery_amendment(root) if pass_id=='e360-generous-p11' else lint_recovery_amendment(root) if pass_id=='e360-generous-p10' else suite_recovery_amendment(root) if pass_id=='e360-generous-p9' else execution_budget_amendment(root)
    policy=authority['checkpoint_policy']
    typed_equal(c['revalidation_amendment'],file_spec(authority_path_for_pass(pass_id),root),'checkpoint exact pass authority')
    typed_equal(c['limits'],authority['effective_roles']['fixture'],'checkpoint exact admitted caps')
    if pass_id=='e360-generous-p9':
        require(current_budget_authority(root)['suite_recovery_active'],'P9 checkpoints require actual current activation')
        check_sources(c['source_sha256'],root)
    if pass_id=='e360-generous-p10':
        require(historical_lint_scope_authority(root),'P10 checkpoints require actual scope activation')
        check_sources(c['source_sha256'],root)
    if pass_id=='e360-generous-p25':return policy_view(historical_p25_repair_policy(root),root)
    if pass_id=='e360-generous-p27':return policy_view(historical_p27_repair_policy(root),root)
    if pass_id=='e360-generous-p29':return policy_view(historical_p29_repair_policy(root),root)
    if pass_id==execution_current_pass(root):
        require(current_scope_authority(root),'P19 checkpoints require actual uncached current scope')
        check_sources(c['source_sha256'],root)
    validate_current_receipt(c,value['contract'],receipt,root=root)
    typed_equal(receipt['contract'],value['contract'],'actual checkpoint receipt descriptor')
    numeric(receipt['entry_perf_counter'],'actual checkpoint parent ENTRY',minimum=1e-12)
    return _validate_checkpoint_rows(value,c,receipt,policy,successful)

def _validate_checkpoint_rows(value,c,receipt,policy,successful):
    points=value.get('execution_checkpoints')
    require(type(points) is dict and set(points)=={'schema','entry_perf_counter','stages'},'typed P8 checkpoints')
    typed_equal(points['schema'],policy['schema'],'checkpoint schema')
    typed_equal(points['entry_perf_counter'],receipt['entry_perf_counter'],'same actual parent ENTRY checkpoints')
    require(type(points['entry_perf_counter']) is float,'typed actual parent ENTRY')
    rows=points['stages']
    require(type(rows) is list and len(rows)<=policy['max_entries'] and len(json_bytes(points))<=policy['max_json_bytes'],'bounded checkpoints')
    require(all(type(row) is dict for row in rows),'typed checkpoint rows')
    names=policy['stage_names']
    typed_equal([row.get('stage') for row in rows],names[:len(rows)],'reached-only ordered checkpoint names')
    if successful:typed_equal(len(rows),len(names),'all successful actual fixture checkpoints')
    last=0.0
    for row in rows:
        require(type(row) is dict and set(row)=={'stage','elapsed_s'},'exact checkpoint fields')
        elapsed=numeric(row['elapsed_s'],'actual checkpoint elapsed')
        require(type(elapsed) is float and elapsed>=last,'finite monotone actual checkpoints')
        if successful:require(elapsed<=c['limits']['internal_s'],'timely successful checkpoint')
        last=elapsed
    elapsed=numeric(value['elapsed_before_publication_s'],'actual envelope elapsed')
    require(type(elapsed) is float and last<=elapsed,'finite actual checkpoint envelope elapsed')
    if successful:require(elapsed<=c['limits']['internal_s'],'checkpoint successful envelope deadline')
    else:require(value.get('run_valid') is False and value.get('effective_qualified') is False,'failed checkpoint evidence stays unqualified')
    return points


HISTORICAL_P8_ANCHORS={
    'inherited81':{'path':BASE+'/parent-e360-generous-inherited81.json','bytes':86537,'sha256':'01e369a7fc1939596cd4b0ff37218052c4e1c24936b3414f20131039edf6d07c'},
    'full_exit':{'path':BASE+'/parent-e346-e360-e360-generous-p8-f1-fixture-full-exit.json','bytes':76557,'sha256':'afd76ae5b13d9d1f4f371efa552c0967a3ff4ecc639f914030cb9a702d3d15c3'},
    'contract':{'path':BASE+'/parent-e346-e360-e360-generous-p8-f1-fixture.json','bytes':150876,'sha256':'94b5502d68e69ec4d168ec39b6402e252a1a3599a69a79d57888a3d467f78532'},
    'receipt':{'path':BASE+'/parent-e346-e360-e360-generous-p8-f1-fixture-pre-spawn.json','bytes':255844,'sha256':'7b4e407e99daada5b6898f83de0f5658be99bd89596b3d7f21ee0c64c7055fb6'},
    'outer':{'path':BASE+'/parent-e346-e360-e360-generous-p8-f1-fixture-outer.json','bytes':76067,'sha256':'beed57134577bdd3d5119705ea35d1d59202ecedc216a2cce7fa9460c6467036'},
    'envelope':{'path':BASE+'/e360-e360-generous-p8-f1-fixture/envelope.json','bytes':92245,'sha256':'a858b696b4b2ec9aebcced8f2ff170003dee9e00ac2fc3de83d29ce6583ed664'},
    'actual_return':{'path':BASE+'/parent-e346-e360-e360-generous-p8-f1-fixture-actual-tool-return.json','bytes':877,'sha256':'16f5aa1d1f79dd05e700dd6614896917e5de0fd1015f8ce194ab8354fcbbcc3a'}}


def validate_historical_p8_checkpoints(anchor,*,root=ROOT,c=None,value=None,receipt=None):
    """Frozen proof only; no prospective admission, live PID, seal or current Q."""
    typed_equal(anchor,HISTORICAL_P8_ANCHORS['full_exit'],'only trusted fixed historical P8 anchor before IO')
    # Read exactly these compiled JSON leaves, never supplied descendant paths.
    frozen={}
    for key,pin in HISTORICAL_P8_ANCHORS.items():
        raw=bounded_json_read(pin,root,cap=262144)
        typed_equal(descriptor(pin['path'],raw),pin,'historical immutable frozen bytes '+key)
        frozen[key]=strict_json(raw)
    recovery_spec={'path':SUITE_RECOVERY_AMENDMENT,'bytes':28728,'sha256':SUITE_RECOVERY_SHA}
    recovery=strict_json(bounded_json_read(recovery_spec,root,cap=262144))
    typed_equal(recovery['inherited81'],HISTORICAL_P8_ANCHORS['inherited81'],'historical literal inherited81')
    authority_spec={'path':EXECUTION_BUDGET_AMENDMENT,'bytes':24334,'sha256':EXECUTION_BUDGET_SHA}
    authority=strict_json(bounded_json_read(authority_spec,root,cap=262144))
    row=frozen['full_exit'];contract=frozen['contract'];envelope_=frozen['envelope'];parent=frozen['receipt'];outer=frozen['outer']
    for supplied,parsed,label in ((c,contract,'contract'),(value,envelope_,'envelope'),(receipt,parent,'receipt')):
        if supplied is not None:typed_equal(supplied,parsed,'historical frozen complete '+label)
    cost_id='e360-generous-p8-f1:fixture'
    ledger=frozen['inherited81']['ledger']
    debits=[debit for debit in ledger['debits'] if debit['cost_id']==cost_id]
    typed_equal(debits,[{'cost_id':cost_id,'mode':'generated','actual_outer_full_return_s':26.753,
        'descriptor':anchor,'actual_returned_exit_code':0}],'historical unique actual once debit')
    typed_equal(row['cost_id'],cost_id,'historical worker cost')
    require(cost_id not in ledger['e360_pending_actuals'] and cost_id not in ledger['e360_reservations'],
        'historical worker pending and reservation closed')
    worker=ledger['e360_calls'][row['call_id']]
    typed_equal(worker['state'],'charged','historical worker charged state')
    typed_equal(worker['cost_id'],cost_id,'historical worker once cost identity')
    typed_equal(row['source_sha256'],contract['source_sha256'],'historical closed complete source510')
    for key in ('registration','amendment','resource_extension','revalidation_amendment','closure_report'):
        typed_equal(row[key],contract[key],'historical full return direct authority '+key)
    typed_equal(row['actual_outer_full_return_s'],26.753,'historical measured full return')
    for field,expected in (('charged_once',True),('qualified',True),('exit_code',0),
        ('actual_full_return_pending',False),('outer_revocation',None),('first_error',None),('secondary_errors',[])):
        typed_equal(row[field],expected,'historical closed worker '+field)
    for record in (row,outer):
        require(owned_closed(record),'historical closed ownership')
        for field,expected in (('natural_closed',True),('authority_admitted',True),('Popen_attempted',True),
            ('Popen_returned',True),('Popen_never_attempted',False)):
            typed_equal(record[field],expected,'historical closed actual '+field)
    for field,key in (('contract','contract'),('outer','outer'),('envelope','envelope'),('actual_tool_return','actual_return')):
        typed_equal(row[field],HISTORICAL_P8_ANCHORS[key],'historical fixed leaf linkage '+field)
    typed_equal(contract['pass_id'],'e360-generous-p8','historical original pass')
    typed_equal(contract['mode'],'generated','historical original mode')
    typed_equal(contract['role'],'fixture','historical original role')
    typed_equal(contract['limits'],authority['effective_roles']['fixture'],'historical exact original caps')
    typed_equal(contract['revalidation_amendment'],authority_spec,'historical original direct authority')
    require(len(contract['source_sha256'])==510,'historical full source510')
    historical_own=ledger['e360_passes']['e360-generous-p8']['source_sha256']
    require(len(historical_own)==9,'historical exact own9')
    typed_equal({key:contract['source_sha256'].get(key) for key in historical_own},historical_own,'historical own9 projection')
    expected=compact_receipt(contract,HISTORICAL_P8_ANCHORS['contract'],287385.5699616,
        parent['parent_pid'],parent['parent_create_time'])
    expected['schema']='flylab.m3.e360.actual-pre-spawn.v1';expected['authority']=contract
    typed_equal(parent,expected,'historical original complete fullauthority v1')
    for record in (outer,envelope_):
        typed_equal(record['contract'],HISTORICAL_P8_ANCHORS['contract'],'historical same contract')
        typed_equal(record['pre_spawn'],HISTORICAL_P8_ANCHORS['receipt'],'historical same receipt')
        typed_equal(record['source_sha256'],contract['source_sha256'],'historical whole source pins')
        for key in ('registration','amendment','resource_extension','revalidation_amendment','closure_report'):
            typed_equal(record[key],contract[key],'historical direct linkage '+key)
        typed_equal(record['first_error'],None,'historical no first error')
        typed_equal(record['secondary_errors'],[],'historical no secondary error')
    typed_equal(outer['entry_perf_counter'],parent['entry_perf_counter'],'historical original outer ENTRY')
    typed_equal(outer['envelope'],HISTORICAL_P8_ANCHORS['envelope'],'historical outer envelope')
    actual=frozen['actual_return']
    typed_equal(actual['actual_outer_full_return_s'],26.753,'historical actual return wall')
    typed_equal(actual['actual_returned_exit_code'],0,'historical actual exit')
    typed_equal(actual['call_id'],row['call_id'],'historical actual call identity')
    typed_equal(actual['pass_id'],contract['pass_id'],'historical actual pass identity')
    typed_equal(envelope_['run_valid'],True,'historical original successful envelope')
    typed_equal(envelope_['revoked'],False,'historical original unrevoked envelope')
    return _validate_checkpoint_rows(envelope_,contract,parent,authority['checkpoint_policy'],True)

def checkpoint(c,result,timer,stage,*,root=ROOT):
    if c.get('pass_id') not in ('e360-generous-p8','e360-generous-p9','e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p13','e360-generous-p14','e360-generous-p15','e360-generous-p16','e360-generous-p17','e360-generous-p18','e360-generous-p19','e360-generous-p20',execution_current_pass(root)) or c['mode']!='generated' or c['role']!='fixture':return
    # The complete policy was reproduced in fixed-reader/pre-array admission.
    # Recording on the descriptors-only science path performs no authority IO.
    points=result.setdefault('execution_checkpoints',{'schema':CHECKPOINT_SCHEMA,'entry_perf_counter':timer.entry,'stages':[]})
    rows=points['stages']
    require(len(rows)<len(CHECKPOINT_STAGES) and len(rows)<CHECKPOINT_MAX_ENTRIES,'checkpoint entry bound')
    typed_equal(stage,CHECKPOINT_STAGES[len(rows)],'actual reached checkpoint order')
    elapsed=float(time.perf_counter()-timer.entry)
    require(math.isfinite(elapsed) and elapsed>=0 and (not rows or elapsed>=rows[-1]['elapsed_s']),'same actual checkpoint clock')
    rows.append({'stage':stage,'elapsed_s':elapsed})
    require(len(json_bytes(points))<=CHECKPOINT_MAX_JSON_BYTES,'checkpoint byte bound')

@historical_dependencies(root_first=False)
def historical_fix_bytes(spec,*,pass_id,root=ROOT):
    if pass_id=='e360-generous-p30':
        freeze=strict_json(read_spec(P30_FROZEN_FREEZE_SPEC,root,owner='historical'))
        for row in freeze['source_copies']:
            if spec==row['current']:return read_spec(row['frozen'],root,owner='historical')
        if spec['path']==BASE+'/parent-e360-p30-root-final-freeze.json':
            typed_equal({key:spec[key] for key in ('bytes','sha256')},{key:P30_FROZEN_FREEZE_SPEC[key] for key in ('bytes','sha256')},'exact old logical P30 freeze')
            return read_spec(P30_FROZEN_FREEZE_SPEC,root,owner='historical')
        if spec==CURRENT_REPAIR_SPEC:return read_spec(P30_FROZEN_POLICY_SPEC,root,owner='historical')
        require(spec['path'] not in {row['current']['path'] for row in freeze['source_copies']},'unregistered historical P30 mutable Source')
        return read_spec(spec,root,owner='historical')
    """Exact pass/full-descriptor relocation only; immutable corrections read directly."""
    if pass_id=='e360-generous-p29':
        freeze=historical_p29_source_freeze(root)
        for row in freeze['source_copies']:
            if spec==row['current']:return read_spec(row['frozen'],root,owner='historical')
        require(spec['path'] not in {row['current']['path'] for row in freeze['source_copies']},'unregistered historical P29 mutable descriptor')
        return read_spec(spec,root,owner='historical')
    if pass_id=='e360-generous-p27':
        freeze=historical_p27_source_freeze(root)
        for row in freeze['source_copies']:
            if spec==row['current']:return read_spec(row['frozen'],root,owner='historical')
        require(spec['path'] not in {row['current']['path'] for row in freeze['source_copies']},'unregistered historical P27 mutable descriptor')
        return read_spec(spec,root,owner='historical')
    if pass_id=='e360-generous-p25':
        freeze=historical_p25_source_freeze(root)
        for row in freeze['source_copies']:
            if spec==row['current']:return read_spec(row['frozen'],root)
        require(spec['path'] not in {row['current']['path'] for row in freeze['source_copies']},'unregistered historical P25 mutable descriptor')
        return read_spec(spec,root)
    if pass_id=='e360-generous-p22':
        freeze=historical_p22_source_freeze(root)
        for row in freeze['source_copies']:
            if spec==row['current']:return read_spec(row['frozen'],root)
        require(spec['path'] not in {row['current']['path'] for row in freeze['source_copies']},'unregistered historical P22 mutable descriptor')
        return read_spec(spec,root)
    if pass_id=='e360-generous-p23':
        freeze=historical_p23_source_freeze(root)
        for row in freeze['source_copies']:
            if spec==row['current']:return read_spec(row['frozen'],root)
        require(spec['path'] not in {row['current']['path'] for row in freeze['source_copies']},'unregistered historical P23 mutable descriptor')
        return read_spec(spec,root)
    if pass_id=='e360-generous-p21':
        freeze=historical_p21_source_freeze(root)
        for row in freeze['source_copies']:
            if spec==row['current']:return read_spec(row['frozen'],root)
        require(spec['path'] not in {row['current']['path'] for row in freeze['source_copies']},'unregistered historical P21 mutable descriptor')
        return read_spec(spec,root)
    if pass_id=='e360-generous-p20':
        freeze=historical_p20_source_freeze(root)
        for row in freeze['source_copies']:
            if spec==row['current']:return read_spec(row['frozen'],root)
        require(spec['path'] not in {row['current']['path'] for row in freeze['source_copies']},'unregistered historical P20 mutable descriptor')
        return read_spec(spec,root)
    if pass_id=='e360-generous-p19':
        freeze=historical_p19_source_freeze(root)
        for row in freeze['source_copies']:
            if spec==row['current']:return read_spec(row['frozen'],root)
        require(spec['path'] not in {row['current']['path'] for row in freeze['source_copies']},'unregistered historical P19 mutable descriptor')
    if pass_id=='e360-generous-p17':
        freeze=strict_json(read_spec(historical_p18_repair_policy(root)['predecessor_as_executed_freeze'],root))
        for row in freeze['source_copies']:
            if spec==row['current']:return read_spec(row['frozen'],root)
        require(spec.get('path') not in freeze['own_sources_sha256'],'unregistered historical P17 mutable descriptor')
        return read_spec(spec,root)
    rows=(p17_recovery_amendment(root) if pass_id=='e360-generous-p16' else p16_recovery_amendment(root) if pass_id=='e360-generous-p15' else p15_recovery_amendment(root) if pass_id=='e360-generous-p14' else p14_recovery_amendment(root) if pass_id=='e360-generous-p13' else p13_recovery_amendment(root) if pass_id=='e360-generous-p12' else p12_recovery_amendment(root) if pass_id=='e360-generous-p11' else p11_recovery_amendment(root) if pass_id=='e360-generous-p10' else lint_recovery_amendment(root) if pass_id=='e360-generous-p9' else suite_recovery_amendment(root) if pass_id=='e360-generous-p8' else execution_budget_amendment(root) if pass_id=='e360-generous-p7' else current_revalidation_amendment(root))['historical_concrete_fix_relocations']
    for row in rows:
        if spec==row['original']:
            typed_equal(pass_id,row['pass_id'],'historical relocation pass')
            typed_equal(spec,row['original'],'exact historical descriptor')
            return read_spec(row.get('snapshot',row.get('frozen')),root)
    if pass_id=='e360-generous-p16':
        immutable_own=tuple(p17_recovery_amendment(root)['historical_P16_frozen_own_sources_sha256'])
        require(spec.get('path') not in immutable_own,'unregistered historical P16 mutable descriptor')
    if pass_id=='e360-generous-p15':
        immutable_own=tuple(p16_recovery_amendment(root)['historical_P15_frozen_own_sources_sha256'])
        require(spec.get('path') not in immutable_own,'unregistered historical P15 mutable descriptor')
    if pass_id=='e360-generous-p14':
        immutable_own=tuple(p15_recovery_amendment(root)['historical_P14_frozen_own_sources_sha256'])
        require(spec.get('path') not in immutable_own,'unregistered historical P14 mutable descriptor')
    if pass_id=='e360-generous-p13':
        immutable_own=tuple(p14_recovery_amendment(root)['historical_P13_frozen_own_sources_sha256'])
        require(spec.get('path') not in immutable_own,'unregistered historical P13 mutable descriptor')
    if pass_id=='e360-generous-p12':
        immutable_own=tuple(p13_recovery_amendment(root)['historical_P12_frozen_own_sources_sha256'])
        require(spec.get('path') not in immutable_own,'unregistered historical P12 mutable descriptor')
    if pass_id=='e360-generous-p11':
        immutable_own=tuple(p12_recovery_amendment(root)['historical_P11_frozen_own_sources_sha256'])
        require(spec.get('path') not in immutable_own,'unregistered historical P11 mutable descriptor')
    require(not (pass_id in ('e360-generous-p2','e360-generous-p3','e360-generous-p4','e360-generous-p5','e360-generous-p6','e360-generous-p7','e360-generous-p8','e360-generous-p9','e360-generous-p10') and spec.get('path') in own_sources(root)),
            'unregistered historical mutable descriptor')
    return read_spec(spec,root)

def require_pending_p10_seal(request,sem,*,root=ROOT):
    """One same admitted worker/attempt and actual call identity; closure only."""
    require(request.get('pass_id')==execution_current_pass(root) and request.get('call_role') in ('fixture_seal','suite_seal')
        and type(request.get('attempt_id')) is str and re.fullmatch('e360-[a-z0-9-]{1,48}',request['attempt_id']),
        'exact finite P10 necessary-seal request')
    matched=[]
    for row in sem['e360_pending_actuals'].values():
        call=sem['e360_calls'].get(row['call_id'])
        if call is not None and call['pass_id']==request['pass_id'] and \
                call['call_role']==request['call_role'].replace('_seal','_worker') and \
                call.get('attempt_id')==request.get('attempt_id') and call['mode']=='generated' and \
                call['state']=='actual_pending' and row['mode']=='generated':
            matched.append((row,call))
    require(len(matched)==1,'necessary seal exactly one same pending worker/attempt')
    row,call=matched[0]
    worker_request=strict_json(read_spec(call['request'],root,owner='dynamic' if call['pass_id']==execution_current_pass(root) else 'historical'))
    actual=strict_json(read_spec(row['actual_tool_return'],root))
    for key in ('call_id','pass_id','attempt_id','call_role'):
        typed_equal(worker_request[key],row['call_id'] if key=='call_id' else call[key],
            'seal exact admitted pending worker call identity')
    typed_equal(actual['call_id'],row['call_id'],'seal exact actual pending call identity')
    typed_equal(actual['pass_id'],request['pass_id'],'seal exact actual pending pass')
    typed_equal(actual['request'],call['request'],'seal actual same admitted worker request')

def p14_collector_proof(trace_spec,request,request_spec,seconds,exit_code,*,root=ROOT):
    """Replay immutable genuine transport metadata; never sample a new clock."""
    trace=strict_json(read_spec(trace_spec,root))
    typed_equal(trace['schema'],'flylab.m3.e360.p14.ROOT-genuine-collector-state.v1','collector schema')
    typed_equal(request['pass_id'],'e360-generous-p14','collector current pass only')
    for key in ('pass_id','call_id','call_role'):
        typed_equal(trace[key],request[key],'collector admitted '+key)
    typed_equal(trace['request'],request_spec,'collector exact admitted request')
    typed_equal(trace['collector_source'],request['collector_source'],'collector request source pin')
    typed_equal(trace['collector_source'],file_spec(BASE+'/parent-e360-p14-ROOT-genuine-collector.txt',root),'collector current source')
    typed_equal(trace['state_key'],'e360:p14:ROOT-genuine:'+request['call_id'],'unique collector role state')
    typed_equal(trace_spec['path'],BASE+'/parent-'+request['call_id']+'-ROOT-collector-trace.json','canonical trace publication')
    typed_equal(trace['next_role_allowed'],False,'collector cannot admit another role')
    typed_equal(trace['collection_status'],'FINAL_RECEIVED','first genuine FINAL collected independently of sticky error')
    typed_equal(trace['collection_stopped'],False,'no stopped transport may invent a later FINAL')
    typed_equal(trace['transport_error'],None,'genuine FINAL without a transport exception')
    caps={'ruff':3,'scanner':20,'fixture_build':25,'fixture_worker':55,'fixture_seal':15,
        'suite_build':30,'suite_worker':600,'suite_seal':20,'qualification':30}
    typed_equal(trace['cost_name'],request['call_id'],'collector once cost name')
    typed_equal(trace['cap_s'],caps[request['call_role']],'immutable role cap')
    require(type(trace['max_output_tokens']) is int and trace['max_output_tokens']>=1000,'complete output request bound')
    typed_equal(trace['exec_args'],{'cmd':"pwsh -NoProfile -File '"+BASE+
        "/parent-e360-p14-ROOT-dispatch-adapter.ps1' -CallRole '"+request['call_role']+"'",
        'workdir':'C:/Users/ckthd/Dev/Projects/fly_lab'},'actual ROOT adopted collector dispatch body')
    dispatch=trace['dispatch_ms'];chunks=trace['chunks']
    require(type(dispatch) is int and dispatch>0 and type(chunks) is list and chunks,'genuine dispatch and chunks')
    last=dispatch;known=None;first=None;first_exit=None;complete=True
    for chunk in chunks:
        received=chunk['received_ms'];result=chunk['result']
        require(type(received) is int,'original integer receive clock')
        if received<last:
            require(trace['phase']=='ERROR_HOLD' and type(trace['first_error']) is dict,'clock contradiction retained first error')
            complete=False
        require(first is None,'no result after first genuine FINAL')
        last=received
        require(type(result) is dict,'contradictory transport cannot acquire a later FINAL')
        has_session=result.get('session_id') is not None
        has_exit=result.get('exit_code') is not None
        session=result.get('session_id');returned=result.get('exit_code')
        if has_session and type(session) is int and session>0:
            if known is None:known=session
            require(known==session,'changed session cannot acquire a later FINAL')
        if has_exit and type(returned) is int and not has_session:
            first,first_exit=received,returned
        else:require(has_session and type(session) is int and session>0 and not has_exit,
                     'contradictory result shape cannot acquire a later FINAL')
        output=result.get('output')
        if (type(output) is not str or result.get('truncated') is True or result.get('output_truncated') is True
            or re.search(r'Warning: truncated output|tokens truncated|output was truncated',output or '')):complete=False
    require(first is not None,'lost FINAL remains pending and cannot invent a debit')
    typed_equal(trace['session_id'],known,'first known tool session retained')
    typed_equal(trace['first_final_ms'],first,'immutable first FINAL sample')
    typed_equal(trace['exit_code'],first_exit,'immutable first FINAL exit')
    require(type(seconds) in (int,float) and not isinstance(seconds,bool) and math.isfinite(seconds) and seconds>0,'genuine positive full return')
    elapsed=(first-dispatch)/1000.0
    require(type(trace['full_return_s']) in (int,float) and math.isfinite(trace['full_return_s'])
            and trace['full_return_s']>0,'saved FINAL duration must be a finite positive JSON number')
    require(trace['full_return_s']==elapsed and seconds==elapsed,'exact dispatch to FINAL full return')
    typed_equal(exit_code,first_exit,'actual first FINAL exit')
    require(trace['phase'] in ('FINAL_RECEIVED','ERROR_HOLD'),'closed collector state')
    if trace['phase']=='ERROR_HOLD':
        require(type(trace['first_error']) is dict,'retain first collector failure')
        complete=False
    else:
        typed_equal(trace['first_error'],None,'complete collector has no first error')
        typed_equal(trace['publication'],'REQUIRED','Native publication follows saved FINAL')
    if not any(type(c['result']) is dict and type(c['result'].get('output')) is str
        and c['result']['output'].strip() for c in chunks):complete=False
    if not complete:require(trace['phase']=='ERROR_HOLD' and type(trace['first_error']) is dict,
                            'every collector failure retains the sticky original error')
    return {'valid':complete,'trace':trace_spec,'full_return_s':elapsed,'exit_code':first_exit}


def validate_p17_final_freeze(root=ROOT):
    """Fresh current Source fence; no admission/result/cache and no recursive payload reads."""
    freeze=strict_json(bounded_metadata_bytes(BASE+'/parent-e360-p17-root-final-freeze.json',root))
    typed_equal(freeze['schema'],'flylab.m3.e360.p17.ROOT-final-source-freeze.v1','current ROOT freeze schema')
    typed_equal(freeze['ROOT_review_cycle_complete'],True,'ROOT review cycle complete')
    typed_equal(freeze['approved_literal'],file_spec(P17_RECOVERY_AMENDMENT,root),'freeze current literal')
    typed_equal(freeze['human_authorization'],CURRENT_HUMAN_AUTONOMY,'freeze stable general human autonomy')
    read_spec(CURRENT_HUMAN_AUTONOMY,root)
    typed_equal(freeze['own_sources_sha256'],own_sources(root),'freeze exact fresh own9')
    copies=freeze['source_copies'];native=freeze['all_P17_Native_sources'];scripts=freeze['native_request_scripts']
    require(type(copies) is list and len(copies)==4 and type(native) is list and len(native)==4
        and type(scripts) is list and len(scripts)==3,'freeze Source4 Native4 scripts3')
    current_paths=(*OWNED,BASE+'/parent-e360-root-account.ps1',BASE+'/parent-e360-root-metadata-helper.py')
    require({row['current']['path'] for row in copies}==set(current_paths),'freeze exact current mutable4')
    for row in copies:
        typed_equal(row['current'],file_spec(row['current']['path'],root),'freeze current Source descriptor')
        for key in ('bytes','sha256'):typed_equal(row['frozen'][key],row['current'][key],'freeze actual same-byte Source copy')
        read_spec(row['frozen'],root)
    names=[BASE+'/parent-e360-p17-'+name+'.ps1' for name in ('capture-proposed','prepare-native','bookkeeping-native','ROOT-dispatch-adapter')]
    typed_equal(native,[file_spec(name,root) for name in names],'freeze exact Native4 order')
    typed_equal(scripts,native[1:],'freeze native request scripts3')
    typed_equal(freeze['capture_source'],native[0],'freeze capture source')
    typed_equal(freeze['ROOT_dispatch_adapter'],native[3],'freeze ROOT dispatch adapter')
    typed_equal(freeze['collector_source'],file_spec(BASE+'/parent-e360-p17-ROOT-genuine-collector.txt',root),'freeze collector concrete leaf')
    return freeze

def validate_p30_final_freeze(root=ROOT):
    context=execution_context(root)
    if context is not None:return context['freeze']
    return historical_p30_source_freeze(root)

def historical_p30_source_freeze(root=ROOT):
    """Fresh current Source fence; no admission/result/cache and no recursive payload reads."""
    freeze=strict_json(read_spec(P30_FROZEN_FREEZE_SPEC,root,owner='historical'))
    typed_equal(freeze['schema'],'flylab.m3.e360.p30.ROOT-final-source-freeze.v1','current ROOT freeze schema')
    typed_equal(freeze['ROOT_review_cycle_complete'],True,'ROOT review cycle complete')
    typed_equal(freeze['approved_literal'],CURRENT_REPAIR_SPEC,'freeze current literal')
    typed_equal(freeze['human_authorization'],CURRENT_HUMAN_AUTONOMY,'freeze stable general human autonomy')
    read_spec(CURRENT_HUMAN_AUTONOMY,root)
    require(len(freeze['own_sources_sha256'])==9,'historical exact own9 roster')
    copies=freeze['source_copies'];native=freeze['all_P30_Native_sources'];scripts=freeze['native_request_scripts']
    require(type(copies) is list and len(copies)==4 and type(native) is list and len(native)==4
        and type(scripts) is list and len(scripts)==3,'freeze Source4 Native4 scripts3')
    current_paths=(*OWNED,BASE+'/parent-e360-root-account.ps1',BASE+'/parent-e360-root-metadata-helper.py')
    require({row['current']['path'] for row in copies}==set(current_paths),'freeze exact current mutable4')
    for row in copies:
        typed_equal(row['current']['sha256'],freeze['own_sources_sha256'][row['current']['path']],'historical current Source digest')
        for key in ('bytes','sha256'):typed_equal(row['frozen'][key],row['current'][key],'freeze actual same-byte Source copy')
        read_spec(row['frozen'],root)
    relocated={row['current']['path'] for row in copies}
    for name,digest in freeze['own_sources_sha256'].items():
        if name not in relocated:typed_equal(file_spec(name,root)['sha256'],digest,'unchanged historical own5')
    names=[BASE+'/parent-e360-p30-'+name+'.ps1' for name in ('capture-proposed','prepare-native','bookkeeping-native','ROOT-dispatch-adapter')]
    typed_equal(native,[file_spec(name,root) for name in names],'freeze exact Native4 order')
    typed_equal(scripts,native[1:],'freeze native request scripts3')
    typed_equal(freeze['capture_source'],native[0],'freeze capture source')
    typed_equal(freeze['ROOT_dispatch_adapter'],native[3],'freeze ROOT dispatch adapter')
    typed_equal(freeze['collector_source'],file_spec(BASE+'/parent-e360-p30-ROOT-genuine-collector.txt',root),'freeze collector concrete leaf')
    return freeze

def check_call_request(request,*,root=ROOT,allow_pending=False):
    context=execution_context(root)
    if context is not None:return execution_request(request,context,root,held=True,allow_pending=allow_pending)
    """Immutable call admission, never cached. ROOT callbegin precedes dispatch."""
    require(request.get('pass_id')=='e360-generous-p30',
        'closed historical P10/P11 cannot dispatch')
    require(request.get('mode')!='original','P19 generated-only; original authority false')
    extension=effective_policy(root,pass_id=request.get('pass_id'))
    if request.get('pass_id')=='e360-generous-p9':
        require(current_budget_authority(root)['suite_recovery_active'],'no preactivation P9 admission')
    if request.get('pass_id')=='e360-generous-p10':
        require(historical_lint_scope_authority(root),'no preactivation P10 admission')
        require(request['mode']=='generated','P10 generated only')
    if request.get('pass_id')=='e360-generous-p30':
        require(current_scope_authority(root) and request['mode']=='generated','P19 actual scope generated only')
    required={'schema','call_id','pass_id','call_role','mode','action','expected_output','completion',
              'source_sha256','closure_report','resource_extension','revalidation_amendment'}
    require(type(request) is dict and required<=set(request),'typed immutable call request')
    require(request['schema']=='flylab.m3.e360.helper-request.v1','call request schema')
    for key in ('call_id','pass_id'):
        require(type(request[key]) is str and re.fullmatch('e360-[a-z0-9-]{1,94}',request[key]),'typed '+key)
    role=request['call_role']
    policies=extension['helper_limits']
    if role in policies:
        policy=policies[role]
        typed_equal(request['action'],policy['action'],'helper action')
        typed_equal(request['mode'],policy['mode'],'helper mode')
        typed_equal(request['role'],policy['worker_role'],'helper actual worker role')
    else:
        require(role in ('ruff','scanner','fixture_worker','suite_worker'),'finite call role')
        typed_equal(request['mode'],'generated','generated call mode')
    typed_equal(request['resource_extension'],file_spec(RESOURCE_EXTENSION,root),'request extension')
    typed_equal(request['revalidation_amendment'],authority_spec_for_pass(request['pass_id'],root),'request exact pass authority')
    typed_equal(request['source_sha256'],own_sources(root),'request current own sources')
    if request['pass_id']=='e360-generous-p30':
        typed_equal(request['collector_source'],file_spec(BASE+'/parent-e360-p30-ROOT-genuine-collector.txt',root),'P19 admitted collector source')
    for key in ('expected_output','completion'):
        require(type(request[key]) is str and request[key].startswith((BASE+'/parent-e360-generous-p30-',BASE+'/parent-e346-e360-e360-generous-p30-')),'fresh P19 call output')
        safe(request[key],root)
    if role not in ('ruff','scanner'):
        typed_equal(strict_json(read_spec(request['closure_report'],root)),scan_current(root=root),'call real closure')
    reg=registration(root)
    _,sem,_,_=ledger_state(reg,root)
    held=sem['e360_calls'].get(request['call_id'])
    require(held is not None and held['state']==('actual_pending' if allow_pending else 'begun'),'prospectively consumed call')
    typed_equal(held['request'],file_spec(request['_request_path'],root),'immutable admitted request')
    typed_equal(held['pass_id'],request['pass_id'],'same pass')
    typed_equal(held['resource_extension'],request['resource_extension'],'held extension')
    typed_equal(held['revalidation_amendment'],request['revalidation_amendment'],'held revalidation')
    require(not sem['e360_blocking_holds'],'unknown owned process blocks calls')
    closing=role in ('fixture_seal','suite_seal') and request.get('pass_id') in ('e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p30','e360-generous-p30')
    passed=sem['e360_passes'][request['pass_id']]
    require(not passed.get('failed_cost') or closing,'failed pass permits necessary seal only')
    if closing:
        require_pending_p10_seal(request,sem,root=root)
    else:require_fit(reg,sem,request['mode'],root=root)
    if role=='saved_prefix_build':
        validate_qualification(request['adapter_qualification'],root=root)
    return policies.get(role)
OUTER_CLEANUP_MARGIN_S,OUTER_REPORT_MARGIN_S=.5,.1

SCOPES={key:False for key in ('historical_run_qualified','historical_original_publication_binding_proven',
 'historical_failure_repaired','body_reconstructed','neural_advanced','new_candidate_evaluated','numeric_alpha_search',
 'independently_observed_theta_and_selected','independently_observed_per_control_neural_ticks',
 'whole_state_reconstructed','current_GPU_transfer_reacquired')}


class Clock(lib.Clock):
    def check(self,stage='internal'):
        now=self.clock()
        require(math.isfinite(now) and now>=self.entry,'same monotone E360 ENTRY')
        if now>self.entry+self.limits[stage+'_s']:
            raise TimeoutError('E360 same-ENTRY '+stage+' deadline')


def own_sources(root=ROOT):
    ADMIN_CHECK()
    return {p:sha(safe(p,root).read_bytes()) for p in (*OWNED,*lib.OWNED,
        'tools/verify_c_banc_m3_startup_baseline.py',SUPERVISOR,OBSERVER,
        BASE+'/parent-e360-root-account.ps1',BASE+'/parent-e360-root-metadata-helper.py')}


@historical_dependencies()
def _registration(root=ROOT):
    raw=literal_bytes(REGISTRATION,root)
    require(len(raw)==77417 and sha(raw)==REGISTRATION_SHA,'frozen E360 reflected authority')
    reg=strict_json(raw)
    require(reg['inherited_cost_ledger']['sha256']==INHERITED_SHA
            and reg['inherited_cost_ledger']['bytes']==18271,'frozen inherited42 authority')
    read_spec(reg['inherited_cost_ledger'],root)
    amendment(root)
    return reg


@historical_dependencies()
def _amendment(root=ROOT):
    raw=literal_bytes(AMENDMENT,root)
    require(len(raw)==3086 and sha(raw)==AMENDMENT_SHA,'literal reviewed redistribution amendment')
    value=strict_json(raw)
    typed_equal(value['old_registration'],{'path':REGISTRATION,'bytes':77417,'sha256':REGISTRATION_SHA},'unchanged old authority')
    old=strict_json(read_spec(value['old_registration'],root))
    for key in ('inherited47','reviewed_design','actual_Astra_review','Astra_review_md','ROOT_reflection'):
        read_spec(value[key],root)
    for key,expected in {'local_cost_offset':42,'required_exact_prefix_length':47,'old_generated_local_limit_s':25,
                         'effective_generated_local_limit_s':39,'existing_unassigned_generated_redistributed_s':14,
                         'shared_generated_limit_s':150,'unchanged_original_local_limit_s':20,
                         'unchanged_original_stage_cap_s':300,'unchanged_original_cumulative_limit_s':10200,
                         'new_shared_budget_grant':False,'new_original_execution_authority':False}.items():
        typed_equal(value[key],expected,'exact amendment '+key)
    typed_equal(value['unchanged_roles'],old['roles'],'unchanged role hard caps')
    return value


def scan_project(roots,*,root=ROOT,check=lambda:None):
    """Source-only finite AST reachability. Oracles never enter this function."""
    files,edges,unresolved,chains={},[],[],{}
    canonical_files={}
    parsed={}
    exported_names={}
    parsed_sources={}
    resolved_paths,queried_layout,module_cache={},{},{}
    queue=collections.deque((p,[p]) for p in roots)
    def source_path(path):
        if path not in resolved_paths:
            check()
            value=safe(path,root)
            resolved_paths[path]=value
            queried_layout[path]=(value.is_file(),value.is_dir())
        return resolved_paths[path]
    def source_tree(path):
        canonical=os.path.normcase(str(source_path(path)))
        if canonical not in parsed:
            raw=source_path(path).read_bytes()
            parsed[canonical]=(sha(raw),ast.parse(raw.decode('utf-8-sig'),filename=path))
            check()
        parsed_sources[path]=parsed[canonical][0]
        return parsed[canonical]
    def source_exports(path):
        # Keep source_tree's parsed-source proof even on repeated exports.
        digest,tree=source_tree(path)
        key=(os.path.normcase(str(source_path(path))),digest)
        if key not in exported_names:
            names=set()
            for declaration in tree.body:
                check()
                if isinstance(declaration,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    names.add(declaration.name)
                elif isinstance(declaration,(ast.Import,ast.ImportFrom)):
                    names.update(item.asname or item.name.split('.')[0] for item in declaration.names)
                elif isinstance(declaration,(ast.Assign,ast.AnnAssign)):
                    declared=declaration.targets if isinstance(declaration,ast.Assign) else [declaration.target]
                    names.update(item.id for item in declared if isinstance(item,ast.Name))
            exported_names[key]=frozenset(names)
        return exported_names[key]
    def module_paths(name):
        if name in module_cache:
            return module_cache[name]
        parts=name.split('.')
        if parts[0] not in ('tools','flylab'):
            return [],'external'
        require(all(re.fullmatch('[A-Za-z_][A-Za-z_0-9]*',v) for v in parts),'canonical project module')
        out=[]
        for count in range(1,len(parts)+1):
            init='/'.join(parts[:count])+'/__init__.py'
            if source_path(init).is_file():
                out.append(init)
        path='/'.join(parts)+'.py'
        if source_path(path).is_file():
            out.append(path)
        directory=source_path('/'.join(parts))
        if not out or (not source_path(path).is_file() and not source_path('/'.join(parts)+'/__init__.py').is_file()):
            result=out,'namespace' if directory.is_dir() else 'absent'
        else:
            result=out,'module'
        module_cache[name]=result
        return result
    while queue:
        check()
        name,chain=queue.popleft()
        if name in files:
            continue
        require(name.endswith('.py'),'source-only discovery')
        canonical=os.path.normcase(str(source_path(name)))
        require(canonical not in canonical_files or canonical_files[canonical]==name,'canonical project path alias collision')
        canonical_files[canonical]=name
        digest,tree=source_tree(name)
        files[name]=digest
        chains[name]=chain
        package=name[:-3].replace('/','.')
        package=package.removesuffix('.__init__') if name.endswith('/__init__.py') else package.rsplit('.',1)[0]
        parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        scope_types=(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda)
        def lexical(node,parents=parents,tree=tree,scope_types=scope_types):
            cursor=parents.get(node,tree)
            while cursor is not tree and not isinstance(cursor,scope_types):
                cursor=parents.get(cursor,tree)
            return cursor
        bindings=collections.defaultdict(lambda:collections.defaultdict(list))
        dynamic_names={'import_module','__import__','spec_from_file_location','SourceFileLoader'}
        def bind(scope,target,value,bindings=bindings):
            if isinstance(target,ast.Name):
                bindings[scope][target.id].append(value)
            elif isinstance(target,(ast.Tuple,ast.List)):
                values=value.elts if isinstance(value,(ast.Tuple,ast.List)) and len(value.elts)==len(target.elts) else [None]*len(target.elts)
                for item,val in zip(target.elts,values):
                    bind(scope,item,val)
        for node in ast.walk(tree):
            if getattr(node,'lineno',0)%128==0:
                check()
            scope=lexical(node)
            if isinstance(node,ast.Import):
                for item in node.names:
                    bindings[scope][item.asname or item.name.split('.')[0]].append(('symbol',item.name if item.asname else item.name.split('.')[0]))
            elif isinstance(node,ast.ImportFrom):
                base=node.module or ''
                if node.level:
                    parts=package.split('.')
                    require(node.level<=len(parts),'relative alias within package')
                    base='.'.join(parts[:len(parts)-node.level+1]+([base] if base else []))
                for item in node.names:
                    bindings[scope][item.asname or item.name].append(('symbol',base+'.'+item.name))
                    if item.name in ('import_module','__import__','spec_from_file_location','SourceFileLoader'):
                        dynamic_names.add(item.asname or item.name)
            elif isinstance(node,ast.Assign):
                for target in node.targets:
                    bind(scope,target,node.value)
            elif isinstance(node,(ast.AnnAssign,ast.NamedExpr)):
                bind(scope,node.target,node.value)
            elif isinstance(node,(ast.AugAssign,ast.For,ast.AsyncFor)):
                bind(scope,node.target,None)
            elif isinstance(node,scope_types):
                if hasattr(node,'name'):
                    bindings[scope][node.name].append(None)
                if hasattr(node,'args'):
                    args=[*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs]
                    args += [v for v in (node.args.vararg,node.args.kwarg) if v is not None]
                    for arg in args:
                        bindings[node][arg.arg].append(None)
            elif isinstance(node,(ast.Global,ast.Nonlocal)):
                # Conservative: unusual rebinding cannot prove a finite import.
                for identifier in node.names:
                    bindings[scope][identifier].append(None)
        def values(node,kind,scope=None,seen=frozenset(),bindings=bindings,tree=tree):
            if node is None:
                return {None}
            scope=lexical(node) if scope is None else scope
            if isinstance(node,ast.Constant) and kind=='literal' and type(node.value) is str:
                return {node.value}
            if isinstance(node,ast.Name):
                cursor=scope
                while node.id not in bindings[cursor] and cursor is not tree:
                    cursor=lexical(cursor)
                if node.id in bindings[cursor]:
                    key=(id(cursor),node.id,kind)
                    if key in seen:
                        return {None}
                    result=set()
                    for definition in bindings[cursor][node.id]:
                        if isinstance(definition,tuple):
                            result.add(definition[1] if kind=='symbol' else None)
                        else:
                            result.update(values(definition,kind,cursor,seen|{key}))
                    return result or {None}
                return {node.id} if kind=='symbol' else {None}
            if isinstance(node,ast.Attribute) and kind=='symbol':
                return {v+'.'+node.attr if v is not None else None for v in values(node.value,kind,scope,seen)}
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add) and kind=='literal':
                return {a+z if a is not None and z is not None else None for a in values(node.left,kind,scope,seen)
                        for z in values(node.right,kind,scope,seen)}
            if isinstance(node,ast.Call):
                if isinstance(node.func,ast.Name) and node.func.id=='getattr' and len(node.args)==2 and kind=='symbol':
                    return {a+'.'+z if a is not None and z is not None else None for a in values(node.args[0],kind,scope,seen)
                            for z in values(node.args[1],'literal',scope,seen)}
                symbols=values(node.func,'symbol',scope,seen)
                if kind=='literal' and any(v and v.split('.')[-1]=='safe' for v in symbols) and node.args:
                    # reference.safe(name,root), core.safe(root,name), safe(CORE).
                    # A root variable is not itself a literal source path.
                    candidates=[values(arg,'literal',scope,seen) for arg in node.args[:2]]
                    usable=[v for v in candidates if any(item is not None and item.endswith('.py') for item in v)]
                    return usable[0] if len(usable)==1 else {None}
            return {None}
        def symbols(node):
            return values(node,'symbol')
        def literal(node):
            result=values(node,'literal')
            return next(iter(result)) if len(result)==1 else None
        # Finite assignment aliases are lexical, including lazy local aliases.
        dynamic_symbols={'importlib.import_module','import_module','__import__','builtins.__import__',
                         'importlib.util.spec_from_file_location','importlib.machinery.SourceFileLoader'}
        for scope,rows in bindings.items():
            for identifier in rows:
                if symbols(ast.Name(id=identifier)) & dynamic_symbols:
                    dynamic_names.add(identifier)
                for definition in rows[identifier]:
                    if isinstance(definition,ast.AST) and values(definition,'symbol',scope) & dynamic_symbols:
                        dynamic_names.add(identifier)
        def emit(node,requested,kind='import',file_target=False,attribute=False,
                 parents=parents,name=name,digest=digest,chain=chain):
            scopes=[]
            cursor=node
            while cursor in parents:
                cursor=parents[cursor]
                if isinstance(cursor,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    scopes.insert(0,cursor.name)
            row={'importer':name,'importer_sha256':digest,'line':node.lineno,'expression':ast.unparse(node),
                 'requested':requested,'kind':kind,'scope':scopes,'dormant_possible':True,'targets':[],'back_edges':[]}
            if requested is None:
                row['reason']='unsupported finite dynamic expression'
                unresolved.append(row)
                return
            if file_target:
                require(requested.endswith('.py'),'file-loader Python source only')
                targets,resolution=([requested],'file') if source_path(requested).is_file() else ([], 'absent')
            else:
                targets,resolution=module_paths(requested)
            row['resolution']=resolution
            if resolution=='absent' and attribute:
                parent,export=requested.rsplit('.',1)
                parent_targets,parent_resolution=module_paths(parent)
                exported=False
                if parent_resolution=='module':
                    module_source=parent.replace('.','/')+'.py'
                    module_source=module_source if source_path(module_source).is_file() else parent.replace('.','/')+'/__init__.py'
                    exported=export in source_exports(module_source)
                if exported:
                    row['resolution']='exported_attribute'
                    targets=parent_targets
                else:
                    row['reason']='project from-import has no proved module or exported name'
                    unresolved.append(row)
            elif resolution=='absent':
                row['reason']='project target absent'
                unresolved.append(row)
            for target in targets:
                target_hash=source_tree(target)[0]
                row['targets'].append({'path':target,'sha256':target_hash})
                if target in files:
                    require(files[target]==target_hash,'source changed/collision during AST discovery')
                    row['back_edges'].append(target)
                else:
                    queue.append((target,chain+[target]))
            edges.append(row)
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for item in node.names:
                    emit(node,item.name)
            elif isinstance(node,ast.ImportFrom):
                base=node.module or ''
                if node.level:
                    parts=package.split('.')
                    require(node.level<=len(parts),'relative import within package')
                    base='.'.join(parts[:len(parts)-node.level+1]+([base] if base else []))
                emit(node,base)
                if base.split('.')[0] in ('tools','flylab'):
                    for item in node.names:
                        if item.name!='*':
                            emit(node,base+'.'+item.name,'from-submodule-or-attribute',attribute=True)
            elif isinstance(node,ast.Call):
                resolved=symbols(node.func)
                calls=resolved & dynamic_symbols
                potential=(isinstance(node.func,ast.Name) and node.func.id in dynamic_names
                           or isinstance(node.func,ast.Attribute) and node.func.attr in dynamic_names)
                if not calls and potential:
                    emit(node,None,'ambiguous-dynamic-alias')
                if calls and (None in resolved or len(resolved)>1):
                    emit(node,None,'ambiguous-dynamic-alias')
                for call in sorted(calls):
                    if call in ('importlib.util.spec_from_file_location','importlib.machinery.SourceFileLoader'):
                        filename=node.args[1] if len(node.args)>1 else next(
                            (k.value for k in node.keywords if k.arg in ('location','path')),None)
                        emit(node,literal(filename),'literal-file-loader',file_target=True)
                        continue
                    target=literal(node.args[0]) if node.args else next(
                        (literal(k.value) for k in node.keywords if k.arg=='name'),None)
                    if call in ('__import__','builtins.__import__'):
                        level_node=node.args[4] if len(node.args)>4 else next(
                            (k.value for k in node.keywords if k.arg=='level'),None)
                        if level_node is not None:
                            if isinstance(level_node,ast.Constant) and type(level_node.value) is int and level_node.value>=0:
                                if level_node.value and target:
                                    parts=package.split('.')
                                    require(level_node.value<=len(parts),'finite relative __import__ level')
                                    target='.'.join(parts[:len(parts)-level_node.value+1]+[target])
                            else:
                                target=None
                    if target and target.startswith('.'):
                        package_arg=literal(node.args[1]) if len(node.args)>1 else next(
                            (literal(k.value) for k in node.keywords if k.arg=='package'),None)
                        if package_arg:
                            level=len(target)-len(target.lstrip('.'))
                            parts=package_arg.split('.')
                            target='.'.join(parts[:len(parts)-level+1]+[target.lstrip('.')])
                        else:
                            target=None
                    emit(node,target,'literal-dynamic')
                    if call in ('__import__','builtins.__import__') and target:
                        fromlist=node.args[3] if len(node.args)>3 else next(
                            (k.value for k in node.keywords if k.arg=='fromlist'),None)
                        if fromlist is not None:
                            if isinstance(fromlist,(ast.List,ast.Tuple)) and all(literal(v) is not None for v in fromlist.elts):
                                for item in fromlist.elts:
                                    if literal(item)!='*':
                                        emit(node,target+'.'+literal(item),'literal-fromlist',attribute=True)
                            else:
                                emit(node,None,'unresolved-fromlist')
    for name,digest in {**files,**parsed_sources}.items():
        check()
        require(sha(safe(name,root).read_bytes())==digest,'AST final source stability')
    # Cached negative lookups and namespace resolution are also part of this
    # single scan. Recheck them and every no-reparse path before publication.
    for name,path in resolved_paths.items():
        check()
        current=safe(name,root)
        require(current==path and (current.is_file(),current.is_dir())==queried_layout[name],
                'AST final project layout stability')
    return {'schema':'flylab.m3.e360.project-closure.v1','roots':list(roots),'files':dict(sorted(files.items())),
            'parsed_sources':dict(sorted(parsed_sources.items())),
            'queried_layout':{name:{'kind':'file' if flags[0] else 'directory' if flags[1] else 'absent',
                                   'canonical':os.path.normcase(str(resolved_paths[name]))}
                              for name,flags in sorted(queried_layout.items())},
            'edges':edges,'chains':dict(sorted(chains.items())),'unresolved':unresolved,
            'future_ready':not unresolved,'arbitrary_dynamic_native_completeness_proven':False}


_REAL_SCAN_PROJECT=scan_project


def cached_project(roots,authority,source_identity,*,root=ROOT,check=lambda:None,commit=True):
    """At most four same-process real ASTs; no supplied/persisted report seed."""
    base=sha(json_bytes({'root':os.path.normcase(str(root.resolve())),'authority':authority,'roots':list(roots)}))
    key=(base,sha(json_bytes(source_identity)))
    for existing in _PROJECT_CACHE:
        require(existing[0]!=base or existing==key,'cached scanner/helper source drift')
    miss=key not in _PROJECT_CACHE
    if miss:
        report=_REAL_SCAN_PROJECT(roots,root=root,check=check)
        # The real scan just checked every project SHA/layout/no-reparse path.
        # Own sources outside its graph still need a fresh identity check.
        fresh_sources=source_identity
    else:
        report=_PROJECT_CACHE[key]
        fresh_sources={**report['files'],**report['parsed_sources'],**source_identity}
    for path,digest in fresh_sources.items():
        check()
        require(sha(safe(path,root).read_bytes())==digest,'cached current source drift: '+path)
    if not miss:
        for path,layout in report['queried_layout'].items():
            check()
            current=safe(path,root)
            kind='file' if current.is_file() else 'directory' if current.is_dir() else 'absent'
            require(kind==layout['kind'] and os.path.normcase(str(current))==layout['canonical'],'cached queried layout drift: '+path)
    if commit and miss:
        if len(_PROJECT_CACHE)>=4:
            _PROJECT_CACHE.pop(next(iter(_PROJECT_CACHE)))
        _PROJECT_CACHE[key]=copy.deepcopy(report)
    if miss and not commit:
        # This fresh object has no private cache owner. scan_current takes its
        # one private raw-AST copy before enriching and committing the report.
        return report
    return copy.deepcopy(report)


def scan_current(reg=None,*,root=ROOT,check=lambda:None):
    requested_check=check
    def checked():
        ADMIN_CHECK()
        requested_check()
    check=checked
    completed_cache=None
    if reg is None:
        reg=registration(root)
    # Generated oracle tests can use a finite fake registration; those do not
    # seed the authority cache. Real admission always supplies frozen authority.
    current_context=execution_context(root)
    if current_context is not None:
        typed_equal(reg,registration(root),'current exact discovery registration')
        report=_REAL_SCAN_PROJECT(reg['discovery_roots'],root=root,check=check)
        authority_pins={**reg['protected_text_sha256'],**own_sources(root)}
        selected=[current_context['admission_spec'],current_context['manifest_spec'],current_context['manifest']['rules'],current_context['manifest']['source_freeze'],current_context['anchor_spec'],current_context['manifest']['history_catalogue'],*current_context['manifest']['historical_inputs']]
        for specification,record in current_context['sealed_records']:
            selected.extend((specification,*record['inputs'],*(row['physical'] for row in record['aliases'])))
        for spec in selected:authority_pins[spec['path']]=spec['sha256']
        check_sources(authority_pins,root,check=check)
        report['authority_source_sha256']=dict(sorted(authority_pins.items()))
    elif reg.get('inherited_cost_ledger',{}).get('sha256')==INHERITED_SHA:
        typed_equal(reg,registration(root),'cache full old registration')
        amended=amendment(root)
        identity=own_sources(root)
        ADMIN_CHECK()
        extension=resource_extension(root)
        authority={'old':reg,'amendment':amended,'resource_extension':extension,'revalidation_amendment':current_revalidation_amendment(root),'execution_budget_amendment':execution_budget_amendment(root),'suite_recovery_amendment':suite_recovery_amendment(root),'lint_recovery_amendment':lint_recovery_amendment(root),'p11_recovery_amendment':p11_recovery_amendment(root),'p12_recovery_amendment':p12_recovery_amendment(root),'p13_recovery_amendment':p13_recovery_amendment(root),'p14_recovery_amendment':p14_recovery_amendment(root)}
        report=cached_project(reg['discovery_roots'],authority,identity,root=root,check=check,commit=False)
        base=sha(json_bytes({'root':os.path.normcase(str(root.resolve())),'authority':authority,'roots':list(reg['discovery_roots'])}))
        completed_cache=((base,sha(json_bytes(identity))),copy.deepcopy(report))
        authority_pins={REGISTRATION:REGISTRATION_SHA,AMENDMENT:AMENDMENT_SHA,RESOURCE_EXTENSION:RESOURCE_EXTENSION_SHA,REVALIDATION_AMENDMENT:REVALIDATION_SHA,ACTIVE_REVALIDATION_AMENDMENT:ACTIVE_REVALIDATION_SHA,SOURCE_RECOVERY_AMENDMENT:SOURCE_RECOVERY_SHA,RUFF_RECOVERY_AMENDMENT:RUFF_RECOVERY_SHA,OLD_REVALIDATION_AMENDMENT:OLD_REVALIDATION_SHA,**reg['protected_text_sha256'],**identity}
        specifications=[*(reg[k] for k in ('plan','actual_Astra_review','Astra_review_md','ROOT_reflection','inherited_cost_ledger',
                         'body_inventory','body_native_dependencies')),*reg['historical_E359'].values(),
                        *(amended[k] for k in ('inherited47','reviewed_design','actual_Astra_review','Astra_review_md','ROOT_reflection')),
                        *(resource_extension(root)[k] for k in EXTENSION_SOURCE_KEYS),file_spec(REVALIDATION_AMENDMENT,root),*(revalidation_amendment(root)[k] for k in REVALIDATION_SOURCE_KEYS),file_spec(OLD_REVALIDATION_AMENDMENT,root),*(active_revalidation_amendment(root)[k] for k in ACTIVE_REVALIDATION_SOURCE_KEYS),file_spec(ACTIVE_REVALIDATION_AMENDMENT,root),file_spec(SOURCE_RECOVERY_AMENDMENT,root),*(source_recovery_amendment(root)[k] for k in SOURCE_RECOVERY_SOURCE_KEYS),file_spec(RUFF_RECOVERY_AMENDMENT,root),*(ruff_recovery_amendment(root)[k] for k in RUFF_RECOVERY_SOURCE_KEYS),*(current_revalidation_amendment(root)[k] for k in CURRENT_REVALIDATION_SOURCE_KEYS),file_spec(EXECUTION_BUDGET_AMENDMENT,root),*(execution_budget_amendment(root)[k] for k in EXECUTION_BUDGET_SOURCE_KEYS),file_spec(SUITE_RECOVERY_AMENDMENT,root),*(suite_recovery_amendment(root)[k] for k in SUITE_RECOVERY_SOURCE_KEYS),file_spec(LINT_RECOVERY_AMENDMENT,root),*(lint_recovery_amendment(root)[k] for k in LINT_RECOVERY_SOURCE_KEYS),file_spec(P11_RECOVERY_AMENDMENT,root),*(p11_recovery_amendment(root)[k] for k in P11_RECOVERY_SOURCE_KEYS),file_spec(P12_RECOVERY_AMENDMENT,root),*(p12_recovery_amendment(root)[k] for k in P12_RECOVERY_SOURCE_KEYS),*p12_recovery_amendment(root)['reviewed_source_snapshots'],*(r['descriptor'] for r in p12_recovery_amendment(root)['historical_P11_actual_paid_costs']),file_spec(p12_recovery_amendment(root)['ruff_capture_policy']['function_only_source_path'],root),file_spec(P13_RECOVERY_AMENDMENT,root),*(p13_recovery_amendment(root)[k] for k in P13_RECOVERY_SOURCE_KEYS),*p13_recovery_amendment(root)['reviewed_source_snapshots'],*(r['descriptor'] for r in p13_recovery_amendment(root)['historical_P12_actual_paid_costs']),file_spec(p13_recovery_amendment(root)['ruff_capture_policy']['function_only_source_path'],root),*(file_spec(name,root) for name in p13_recovery_amendment(root)['prospective_native_source_paths']),file_spec(P14_RECOVERY_AMENDMENT,root),*(p14_recovery_amendment(root)[k] for k in P14_RECOVERY_SOURCE_KEYS),*p14_recovery_amendment(root)['reviewed_source_snapshots'],file_spec(p14_recovery_amendment(root)['ruff_capture_policy']['function_only_source_path'],root),*(file_spec(name,root) for name in p14_recovery_amendment(root)['prospective_native_source_paths']),file_spec(p14_recovery_amendment(root)['ROOT_genuine_collector_policy']['canonical_future_source_path'],root)]
        specifications.extend(row['frozen'] for row in p14_recovery_amendment(root)['historical_concrete_fix_relocations'][42:46])
        for spec in specifications:
            check()
            read_spec(spec,root)
            require(spec['path'] not in authority_pins or authority_pins[spec['path']]==spec['sha256'],'authority source collision')
            authority_pins[spec['path']]=spec['sha256']
        for key in ('historical_P12_stdout','historical_P12_stderr'):
            spec=p13_recovery_amendment(root)[key];historical_p12_stream_bytes(spec,root)
            require(spec['path'] not in authority_pins or authority_pins[spec['path']]==spec['sha256'],'historical stream source collision')
            authority_pins[spec['path']]=spec['sha256']
        for key in ('historical_P13_ruff_stdout','historical_P13_ruff_stderr'):
            spec=p14_recovery_amendment(root)[key];historical_p12_stream_bytes(spec,root)
            require(spec['path'] not in authority_pins or authority_pins[spec['path']]==spec['sha256'],'completed P13 stream collision')
            authority_pins[spec['path']]=spec['sha256']
        check_sources(authority_pins,root,check=check)
        report['authority_source_sha256']=dict(sorted(authority_pins.items()))
    else:
        report=_REAL_SCAN_PROJECT(reg['discovery_roots'],root=root,check=check)
    observed=set(reg['saved_project_import_oracle'])
    for edge in report['edges']:
        edge['observed_target']=any(v['path'] in observed for v in edge['targets'])
    for path,digest in reg['saved_project_import_oracle'].items():
        require(report['files'].get(path)==digest and path in report['chains'],'observed project oracle not discovered: '+path)
    for row in reg['saved_missing_pin_oracle']:
        require(report['files'].get(row['path'])==row['actual_sha256'],'omitted-edge oracle not discovered: '+row['path'])
    deps=strict_json(read_spec(reg['body_native_dependencies'],root))
    dependency_pins={}
    for section in ('native_library_sources','installed_package_sources'):
        if not isinstance(deps[section],dict):
            continue
        for key,value in deps[section].items():
            dependency_pins[key]=value['sha256'] if isinstance(value,dict) else value
    installed={}
    for name,digest in reg['saved_installed_import_oracle'].items():
        check()
        path=safe(name,root)
        require(sha(path.read_bytes())==digest,'installed observed source drift')
        aliases=[str(path.resolve())]
        for key,value in dependency_pins.items():
            if Path(key).resolve()==path.resolve():
                require(value==digest,'installed alias collision')
                alias_path=Path(key)
                for component in (alias_path,*alias_path.parents):
                    require(not component.is_symlink() and not(getattr(component.lstat(),'st_file_attributes',0)&1024),
                            'installed alias no reparse')
                require(sha(alias_path.read_bytes())==digest,'installed alias fresh source')
                aliases.append(key)
        installed[name]={'sha256':digest,'canonical':str(path.resolve()),'aliases':sorted(set(aliases))}
    report['installed']=installed
    if completed_cache is not None:
        report['resource_extension']=file_spec(RESOURCE_EXTENSION,root)
        report['revalidation_amendment']=prospective_authority_spec(root)
    report['oracle_counts']={'project':len(reg['saved_project_import_oracle']),
                            'missing':len(reg['saved_missing_pin_oracle']),'installed':len(installed)}
    # Explicit executed CPU functions. Unknown imports in uncalled body/trial
    # functions remain visible limitations; no wildcard exemption of imports.
    pure_scopes={'tools/verify_c_banc_m3_common_descent_body.py':
                 {'startup99','array_sha','lean_row','validate_native','verify_chunk_coverage'},
                 'tools/verify_c_banc_m3_coordination_parameter_direction.py':{'load_selected','strict_npy_header'},
                 'tools/qualify_c_banc_m3_body_packet.py':{'scan_archive','member_schema','_header','archive_schema'}}
    report['diagnostic_unresolved']=[v for v in report['unresolved'] if v['importer'] in (OWNED[0],lib.OWNED[0])
        or not v['scope'] or any(k in pure_scopes.get(v['importer'],set()) for k in v['scope'])]
    report['body_only_unresolved_limitations']=[v for v in report['unresolved'] if v not in report['diagnostic_unresolved']]
    report['diagnostic_ready']=not report['diagnostic_unresolved']
    check()
    # A failed oracle/dependency/installed/clock validation cannot seed the
    # production cache. Only the completed real source scan reaches this line.
    if completed_cache is not None and completed_cache[0] not in _PROJECT_CACHE:
        if len(_PROJECT_CACHE)>=4:
            _PROJECT_CACHE.pop(next(iter(_PROJECT_CACHE)))
        _PROJECT_CACHE[completed_cache[0]]=completed_cache[1]
    return report


def closure_sources(reg,report,root=ROOT,*,closure_report=None):
    with metadata_invocation(root) as admitted:
        pins=_closure_sources_uncached(reg,report,root,closure_report=closure_report)
        if admitted:check_sources(pins,root)
        return pins


def _closure_sources_uncached(reg,report,root=ROOT,*,closure_report=None):
    context=execution_context(root)
    if context is not None:
        pins={}
        for mapping in (reg['protected_text_sha256'],report['files'],report['parsed_sources'],own_sources(root),report.get('authority_source_sha256',{})):
            for name,digest in mapping.items():
                require(name not in pins or pins[name]==digest,'full Source map merge collision')
                pins[name]=digest
        for name,row in report['installed'].items():
            require(name not in pins or pins[name]==row['sha256'],'installed Source map collision')
            pins[name]=row['sha256']
            for alias in row['aliases']:
                require(Path(alias).resolve()==safe(name,root).resolve(),'same actual installed alias')
                require(alias not in pins or pins[alias]==row['sha256'],'installed alias collision')
                pins[alias]=row['sha256']
        for spec in (context['admission_spec'],context['manifest_spec'],context['manifest']['rules'],context['manifest']['source_freeze'],*context['manifest']['historical_inputs'],*context['freeze']['native_sources'],*(context['freeze'][key] for key in ('collector_source','capture_caller','commit_caller','first_error_boundary','fixed_native_context'))):
            require(spec['path'] not in pins or pins[spec['path']]==spec['sha256'],'protocol Source map collision')
            pins[spec['path']]=spec['sha256']
        if closure_report is not None:
            require(closure_report['path'] not in pins or pins[closure_report['path']]==closure_report['sha256'],'closure report Source collision')
            pins[closure_report['path']]=closure_report['sha256']
        check_sources(pins,root)
        return pins
    pins=dict(reg['protected_text_sha256'])
    for mapping in (report['files'],report['parsed_sources'],own_sources(root),{k:v['sha256'] for k,v in report['installed'].items()}):
        for name,digest in mapping.items():
            require(name not in pins or pins[name]==digest,'source closure collision')
            pins[name]=digest
    for name,row in report['installed'].items():
        for alias in row['aliases']:
            require(Path(alias).resolve()==safe(name,root).resolve(),'same canonical installed alias only')
            require(alias not in pins or pins[alias]==row['sha256'],'alias pin collision')
            pins[alias]=row['sha256']
    pins[REGISTRATION]=REGISTRATION_SHA
    for key in ('plan','actual_Astra_review','Astra_review_md','ROOT_reflection','inherited_cost_ledger',
                'body_inventory','body_native_dependencies'):
        read_spec(reg[key],root)
        pins[reg[key]['path']]=reg[key]['sha256']
    for spec in reg['historical_E359'].values():
        read_spec(spec,root)
        pins[spec['path']]=spec['sha256']
    amended=amendment(root)
    specs=[file_spec(AMENDMENT,root),file_spec(RESOURCE_EXTENSION,root),*(amended[k] for k in ('inherited47','reviewed_design','actual_Astra_review','Astra_review_md','ROOT_reflection')),
                        *(resource_extension(root)[k] for k in EXTENSION_SOURCE_KEYS),file_spec(REVALIDATION_AMENDMENT,root),*(revalidation_amendment(root)[k] for k in REVALIDATION_SOURCE_KEYS),file_spec(OLD_REVALIDATION_AMENDMENT,root),*(active_revalidation_amendment(root)[k] for k in ACTIVE_REVALIDATION_SOURCE_KEYS),file_spec(ACTIVE_REVALIDATION_AMENDMENT,root),file_spec(SOURCE_RECOVERY_AMENDMENT,root),*(source_recovery_amendment(root)[k] for k in SOURCE_RECOVERY_SOURCE_KEYS),file_spec(RUFF_RECOVERY_AMENDMENT,root),*(ruff_recovery_amendment(root)[k] for k in RUFF_RECOVERY_SOURCE_KEYS),*(current_revalidation_amendment(root)[k] for k in CURRENT_REVALIDATION_SOURCE_KEYS),file_spec(EXECUTION_BUDGET_AMENDMENT,root),*(execution_budget_amendment(root)[k] for k in EXECUTION_BUDGET_SOURCE_KEYS),file_spec(SUITE_RECOVERY_AMENDMENT,root),*(suite_recovery_amendment(root)[k] for k in SUITE_RECOVERY_SOURCE_KEYS),file_spec(LINT_RECOVERY_AMENDMENT,root),*(lint_recovery_amendment(root)[k] for k in LINT_RECOVERY_SOURCE_KEYS),file_spec(P11_RECOVERY_AMENDMENT,root),*(p11_recovery_amendment(root)[k] for k in P11_RECOVERY_SOURCE_KEYS),file_spec(P12_RECOVERY_AMENDMENT,root),*(p12_recovery_amendment(root)[k] for k in P12_RECOVERY_SOURCE_KEYS),*p12_recovery_amendment(root)['reviewed_source_snapshots'],*(r['descriptor'] for r in p12_recovery_amendment(root)['historical_P11_actual_paid_costs']),file_spec(p12_recovery_amendment(root)['ruff_capture_policy']['function_only_source_path'],root),file_spec(P13_RECOVERY_AMENDMENT,root),*(p13_recovery_amendment(root)[k] for k in P13_RECOVERY_SOURCE_KEYS),*p13_recovery_amendment(root)['reviewed_source_snapshots'],*(r['descriptor'] for r in p13_recovery_amendment(root)['historical_P12_actual_paid_costs']),file_spec(p13_recovery_amendment(root)['ruff_capture_policy']['function_only_source_path'],root),*(file_spec(name,root) for name in p13_recovery_amendment(root)['prospective_native_source_paths']),file_spec(P14_RECOVERY_AMENDMENT,root),*(p14_recovery_amendment(root)[k] for k in P14_RECOVERY_SOURCE_KEYS),*p14_recovery_amendment(root)['reviewed_source_snapshots'],file_spec(p14_recovery_amendment(root)['ruff_capture_policy']['function_only_source_path'],root),*(file_spec(name,root) for name in p14_recovery_amendment(root)['prospective_native_source_paths']),file_spec(p14_recovery_amendment(root)['ROOT_genuine_collector_policy']['canonical_future_source_path'],root)]
    specs.extend(row['frozen'] for row in p14_recovery_amendment(root)['historical_concrete_fix_relocations'][42:46])
    p15=p15_recovery_amendment(root)
    specs.append(file_spec(P15_RECOVERY_AMENDMENT,root))
    specs.extend(spec for spec in p15.values() if type(spec) is dict and set(spec)=={'path','bytes','sha256'} and spec['bytes']>0)
    specs.extend(p15['reviewed_source_snapshots'])
    specs.extend(row['frozen'] for row in p15['historical_concrete_fix_relocations'][46:50])
    specs.extend(p15['historical_P14_Native4'])
    specs.extend(file_spec(name,root) for name in (*p15['prospective_native_source_paths'],p15['ruff_capture_policy']['function_only_source_path'],p15['ROOT_genuine_collector_policy']['canonical_future_source_path']))
    p16=p16_recovery_amendment(root)
    specs.append(file_spec(P16_RECOVERY_AMENDMENT,root))
    specs.extend(spec for spec in p16.values() if type(spec) is dict and set(spec)=={'path','bytes','sha256'} and spec['bytes']>0)
    specs.extend(p16['current_P16_source_snapshots'])
    specs.extend(row['frozen'] for row in p16['historical_concrete_fix_relocations'][50:54])
    specs.extend(p16['historical_P15_Native4'])
    specs.extend(file_spec(name,root) for name in (*p16['prospective_native_source_paths'],p16['ruff_capture_policy']['function_only_source_path'],p16['ROOT_genuine_collector_policy']['canonical_future_source_path']))
    p17=p17_recovery_amendment(root)
    specs.append(file_spec(P17_RECOVERY_AMENDMENT,root))
    specs.extend(spec for spec in p17.values() if type(spec) is dict and set(spec)=={'path','bytes','sha256'} and spec['bytes']>0)
    specs.extend(p17['current_P17_source_snapshots'])
    specs.extend(row['frozen'] for row in p17['historical_concrete_fix_relocations'][54:58])
    specs.extend(p17['historical_P16_Native4'])
    specs.extend(file_spec(name,root) for name in (*p17['prospective_native_source_paths'],p17['ruff_capture_policy']['function_only_source_path'],p17['ROOT_genuine_collector_policy']['canonical_future_source_path']))
    current=current_repair_policy(root)
    specs.extend(_CURRENT_P30_PINS)
    prior_freeze=strict_json(read_spec(current['predecessor_as_executed_freeze'],root))
    specs.extend(row['frozen'] for row in prior_freeze['source_copies'])
    specs.extend(prior_freeze['all_P29_Native_sources']);specs.append(prior_freeze['collector_source'])
    specs.extend(file_spec(BASE+'/parent-e360-p30-'+name+'.ps1',root) for name in
        ('capture-proposed','prepare-native','bookkeeping-native','ROOT-dispatch-adapter'))
    specs.append(file_spec(BASE+'/parent-e360-p30-ROOT-genuine-collector.txt',root))
    if closure_report is not None:
        specs.append(closure_report)
    for spec in specs:
        read_spec(spec,root)
        require(spec['path'] not in pins or pins[spec['path']]==spec['sha256'],'direct closure/amendment pin collision')
        pins[spec['path']]=spec['sha256']
    for key in ('historical_P12_stdout','historical_P12_stderr'):
        spec=p13_recovery_amendment(root)[key];historical_p12_stream_bytes(spec,root)
        require(spec['path'] not in pins or pins[spec['path']]==spec['sha256'],'direct stream closure collision')
        pins[spec['path']]=spec['sha256']
    for key in ('historical_P13_ruff_stdout','historical_P13_ruff_stderr'):
        spec=p14_recovery_amendment(root)[key];historical_p12_stream_bytes(spec,root)
        require(spec['path'] not in pins or pins[spec['path']]==spec['sha256'],'completed P13 stream closure collision')
        pins[spec['path']]=spec['sha256']
    return pins

def check_sources(pins,root=ROOT,*,check=lambda:None):
    for name,digest in pins.items():
        ADMIN_CHECK()
        check()
        require(type(digest) is str and re.fullmatch('[0-9a-f]{64}',digest),'typed source SHA')
        path=Path(name) if Path(name).is_absolute() else safe(name,root)
        require(path.suffix in ('.py','.json','.md','.txt','.ini','.ps1'),'source text only')
        for component in (path,*path.parents):
            require(not component.is_symlink() and not(getattr(component.lstat(),'st_file_attributes',0)&1024),
                    'source no reparse')
        require(sha(path.read_bytes())==digest,'current source closure: '+name)


def charged_cost(spec,mode,root=ROOT):
    row=strict_json(read_spec(spec,root))
    allowed={f'flylab.m3.e{stage}.{kind}-full-return.v1' for stage in (358,359,360) for kind in ('actual','administrative')}
    require(row['schema'] in allowed and row['mode']==mode and row['charged_once'] is True,'once actual cost schema/mode')
    wall=numeric(row['actual_outer_full_return_s'],'actual cost',minimum=1e-12)
    require(type(wall) is float and type(row['cost_id']) is str and row['cost_id'],'typed actual cost')
    actual=strict_json(read_spec(row['actual_tool_return'],root))
    typed_equal(actual['actual_outer_full_return_s'],wall,'actual external cost linkage')
    require(type(actual['actual_returned_exit_code']) is int,'typed actual return code')
    for key in ('actual_returned_exit_code','exit_code'):
        if key in row:
            typed_equal(row[key],actual['actual_returned_exit_code'],'actual cost exit linkage '+key)
    return row


def ledger_state(reg,root=ROOT):
    context=execution_context(root)
    if context is not None:return execution_ledger_state(context,root)
    amended=amendment(root)
    extension=resource_extension(root)
    snapshot=strict_json(read_spec(reg['inherited_cost_ledger'],root))
    prefix=snapshot['ledger']['debits']
    require(len(prefix)==42 and [v['cost_id'] for v in prefix]==reg['inherited_cost_ids']
            ==snapshot['complete_expected_cost_ids'],'immutable complete42 cost order')
    live=strict_json(safe(reg['shared_ledger_path'],root).read_bytes())
    debits=live['debits']
    budget_limits=current_budget_authority(root,live=live)
    prefix47=strict_json(read_spec(amended['inherited47'],root))
    require(len(prefix47['ledger']['debits'])==47,'immutable47 count')
    typed_equal(prefix47['ledger']['debits'][:42],prefix,'immutable47 retains old42')
    typed_equal([v['cost_id'] for v in prefix47['ledger']['debits']],prefix47['complete_expected_cost_ids'],'immutable47 complete ID order')
    prefix50=strict_json(read_spec(extension['inherited50'],root))
    require(len(prefix50['ledger']['debits'])==50,'immutable50 count')
    typed_equal(prefix50['ledger']['debits'][:47],prefix47['ledger']['debits'],'immutable50 retains47')
    typed_equal([v['cost_id'] for v in prefix50['ledger']['debits']],prefix50['complete_expected_cost_ids'],'complete50 ID order')
    require(type(debits) is list and len(debits)>=50,'complete live50 cost roster')
    typed_equal(debits[:50],prefix50['ledger']['debits'],'all immutable50 semantic fields')
    typed_equal(debits[:42],prefix,'all immutable42 semantic fields')
    typed_equal(debits[:47],prefix47['ledger']['debits'],'all immutable47 semantic fields')
    revised=revalidation_amendment(root)
    prefix59=strict_json(read_spec(revised['inherited59'],root))['ledger']
    require(len(prefix59['debits'])==59 and len(debits)>=59,'complete immutable59 authority')
    typed_equal(debits[:59],prefix59['debits'],'all immutable59 debits')
    typed_equal(live['e360_revalidation_activation'],revalidation_event(root),'explicit same-budget activation')
    typed_equal(live['e360_revalidation_amendment'],file_spec(OLD_REVALIDATION_AMENDMENT,root),'historical direct revalidation')
    active=active_revalidation_amendment(root)
    prefix60=strict_json(read_spec(active['inherited60'],root))['ledger']
    require(len(prefix60['debits'])==60 and len(debits)>=60,'complete immutable60 authority')
    typed_equal(prefix60['debits'][:59],prefix59['debits'],'60 preserves59')
    typed_equal(debits[:60],prefix60['debits'],'all immutable60 debits')
    typed_equal(live['e360_active_revalidation_activation'],active_revalidation_event(root),'explicit active recovery activation')
    typed_equal(live['e360_active_revalidation_amendment'],file_spec(ACTIVE_REVALIDATION_AMENDMENT,root),'historical active revalidation')
    source=source_recovery_amendment(root)
    prefix62=strict_json(read_spec(source['inherited62'],root))['ledger']
    require(len(prefix62['debits'])==62 and len(debits)>=62,'complete immutable62 authority')
    typed_equal(prefix62['debits'][:60],prefix60['debits'],'62 preserves60')
    typed_equal(debits[:62],prefix62['debits'],'all immutable62 debits')
    typed_equal(live['e360_source_recovery_activation'],source_recovery_event(root),'historical source recovery activation')
    typed_equal(live['e360_source_recovery_amendment'],file_spec(SOURCE_RECOVERY_AMENDMENT,root),'historical source recovery authority')
    ruff=ruff_recovery_amendment(root)
    prefix63=strict_json(read_spec(ruff['inherited63'],root))['ledger']
    require(len(prefix63['debits'])==63 and len(debits)>=63,'complete immutable63 authority')
    typed_equal(prefix63['debits'][:62],prefix62['debits'],'63 preserves62')
    typed_equal(debits[:63],prefix63['debits'],'all immutable63 debits')
    typed_equal(live['e360_ruff_recovery_activation'],ruff_recovery_event(root),'explicit Ruff recovery activation')
    typed_equal(live['e360_ruff_recovery_amendment'],file_spec(RUFF_RECOVERY_AMENDMENT,root),'current Ruff recovery authority')
    for key in ruff['old_activation_fields_preserved']:
        typed_equal(live[key],prefix63[key],'all six immutable historical activation fields')
    for key in ('e360_calls','e360_passes'):
        for identity,row in prefix63[key].items():typed_equal(live[key][identity],row,'immutable63 '+key)
    current=current_revalidation_amendment(root)
    prefix68=strict_json(read_spec(current['inherited68'],root))['ledger']
    require(len(prefix68['debits'])==68 and len(debits)>=68,'complete immutable68')
    typed_equal(prefix68['debits'][:63],prefix63['debits'],'68 retains63')
    typed_equal(debits[:68],prefix68['debits'],'all immutable68 debits')
    for key in current['old_activation_fields_preserved']:
        typed_equal(live[key],prefix68[key],'all eight historical activation fields')
    for key in ('e360_calls','e360_passes'):
        for identity,row in prefix68[key].items():typed_equal(live[key][identity],row,'immutable68 '+key)
    typed_equal(live['e360_authority_scope_activation'],current_revalidation_event(root),'current scope activation')
    typed_equal(live['e360_authority_scope_amendment'],file_spec(REVALIDATION_AMENDMENT,root),'current scope authority')
    for key in ('e360_resource_grant','e360_revalidation_activation','e360_revalidation_amendment'):
        typed_equal(live[key],prefix60[key],'historical event/grant preserved')
    for key in ('e360_calls','e360_passes'):
        for identity,row in prefix60[key].items():typed_equal(live[key][identity],row,'immutable historical '+key)
    require(len({v['cost_id'] for v in debits})==len(debits),'duplicate/replayed cost ID')
    p27_closed108=completed_historical_p27_proof(root) if len(debits)>107 else None
    p29_closed109=completed_historical_p29_proof(root) if len(debits)>108 else None
    used={'original':[],'generated':[]}
    local={'original':[],'generated':[]}
    for index,debit in enumerate(debits):
        with dependency_ownership(root,'historical' if index<109 else 'dynamic'):
            require(set(debit)=={'cost_id','mode','actual_outer_full_return_s','descriptor','actual_returned_exit_code'},'literal debit fields')
            row=charged_cost(debit['descriptor'],debit['mode'],root)
            consume_source_descriptor(debit['descriptor'],root,owner='historical' if index<109 else 'dynamic')
            for key in ('cost_id','mode','actual_outer_full_return_s'):
                typed_equal(row[key],debit[key],'live actual debit '+key)
            actual=strict_json(read_spec(row['actual_tool_return'],root))
            typed_equal(debit['actual_returned_exit_code'],actual['actual_returned_exit_code'],'live actual exit')
            if index>=50:
                typed_equal(row['resource_extension'],file_spec(RESOURCE_EXTENSION,root),'new actual cost extension')
                typed_equal(actual['resource_extension'],row['resource_extension'],'actual tool extension')
                typed_equal(row['call_id'],actual['call_id'],'actual cost call identity')
                typed_equal(row['pass_id'],actual['pass_id'],'actual cost pass identity')
            if index>=59:
                validate_cost_authority(index,row,actual,root=root)
            if index==82:
                typed_equal(row['pass_id'],'e360-generous-p10','historical index82=P10')
            if 83<=index<88:
                completed_historical_p11_proof(root)
                typed_equal(row['pass_id'],'e360-generous-p11','historical index83..87 P11')
            if index==88:
                completed_historical_p12_proof(root)
                typed_equal(row['pass_id'],'e360-generous-p12','historical index88 P12')
            if 89<=index<91:
                completed_historical_p13_proof(root)
                typed_equal(row['pass_id'],'e360-generous-p13','completed history index89/90')
            if index==91:
                completed_historical_p14_proof(root)
                typed_equal(row['pass_id'],'e360-generous-p14','exact historical index91')
                paid_request_identity(debit,row,actual,live,root=root)
            if index==92:
                whole93=completed_historical_p15_proof(root)
                typed_equal(debit,whole93['debits'][92],'exact completed P15 index92')
                paid_request_identity(debit,row,actual,live,root=root)
            if index==93:
                whole94=completed_historical_p16_proof(root)
                typed_equal(debit,whole94['debits'][93],'exact completed P16 index93')
                paid_request_identity(debit,row,actual,live,root=root)
            if 94<=index<96:
                whole96=completed_historical_p17_proof(root)
                typed_equal(debit,whole96['debits'][index],'exact completed P17 two costs')
                paid_request_identity(debit,row,actual,live,root=root)
            if 96<=index<99:
                closed99=completed_historical_p18_proof(root)
                typed_equal(debit,closed99['debits'][index],'actual historical P18 cost96..98')
                typed_equal(row['revalidation_amendment'],P18_ORIGINAL_POLICY_SPEC,'historical P18 authority original descriptor')
            if index==99:
                closed100=completed_historical_p19_proof(root)
                typed_equal(debit,closed100['debits'][99],'immutable P19 once Ruff cost')
                typed_equal(row['revalidation_amendment'],P19_ORIGINAL_POLICY_SPEC,'historical P19 original policy')
            if index==100:
                closed101=completed_historical_p20_proof(root)
                typed_equal(debit,closed101['debits'][100],'immutable P20 once Ruff cost')
                typed_equal(row['revalidation_amendment'],P20_ORIGINAL_POLICY_SPEC,'historical P20 original policy')
            if 101<=index<103:
                closed103=completed_historical_p21_proof(root)
                typed_equal(debit,closed103['debits'][index],'immutable P21 two actual costs')
            if index==103:
                closed104=completed_historical_p22_proof(root)
                typed_equal(debit,closed104['debits'][index],'immutable P22 actual Ruff cost103')
            if index==104:
                closed105=completed_historical_p23_proof(root)
                typed_equal(debit,closed105['debits'][index],'immutable P23 actual Ruff cost104')
            if 105<=index<107:
                closed107=completed_historical_p25_proof(root)
                typed_equal(debit,closed107['debits'][index],'immutable P25 actual Ruff/scanner costs')
            if index==107:
                typed_equal(debit,p27_closed108['debits'][107],'immutable P27 actual Ruff cost')
                call=live['e360_calls'][row['call_id']]
                paid_request=paid_request_identity(debit,row,actual,live,root=root)
                collector=historical_p27_collector_proof(actual['collector_trace'],paid_request,call['request'],actual['actual_outer_full_return_s'],actual['actual_returned_exit_code'],root=root)
                typed_equal(collector['valid'],actual['collector_proof_valid'],'actual historical P27 own trace')
            if index==108:
                typed_equal(debit,p29_closed109['debits'][108],'immutable successful P29 Ruff108')
                call=live['e360_calls'][row['call_id']]
                paid_request=paid_request_identity(debit,row,actual,live,root=root)
                collector=historical_p29_collector_proof(actual['collector_trace'],paid_request,call['request'],actual['actual_outer_full_return_s'],actual['actual_returned_exit_code'],root=root)
                typed_equal(collector['valid'],actual['collector_proof_valid'],'historical P29 own trace')
            if index>=109:
                require(current_scope_authority(root,live=live),'index99+ actual P19 triple')
                typed_equal(row['pass_id'],'e360-generous-p30','index99+ P19 only')
                call=live['e360_calls'][row['call_id']]
                paid_request=paid_request_identity(debit,row,actual,live,root=root)
                collector=p30_collector_proof(actual['collector_trace'],paid_request,call['request'],actual['actual_outer_full_return_s'],actual['actual_returned_exit_code'],root=root)
                typed_equal(actual['collector_proof_valid'],collector['valid'],'actual immutable collector proof')
                if row['schema']=='flylab.m3.e360.administrative-full-return.v1':
                    typed_equal(row['collector_trace'],actual['collector_trace'],'cost collector trace')
                    typed_equal(row['collector_proof_valid'],collector['valid'],'cost collector validity')
                    require(collector['valid'] or row['internal_completion_valid'] is False,'collector failure cannot qualify admin')
                else:require(collector['valid'] or row['qualified'] is False,'collector failure cannot qualify worker')
            require(debit['mode'] in used,'finite mode')
            used[debit['mode']].append(debit['actual_outer_full_return_s'])
            if index>=42:
                require(row['schema'].startswith('flylab.m3.e360.') and 'e360' in debit['cost_id'],'E360 only new costs')
                local[debit['mode']].append(debit['actual_outer_full_return_s'])
    totals={k:round(math.fsum(v),9) for k,v in used.items()}
    local={k:round(math.fsum(v),9) for k,v in local.items()}
    fixed={'original_cumulative_before_s':reg['original_cumulative_before_s'],'original_cumulative_limit_s':10200,
           'original_stage_cap_s':300,'generated_shared_cumulative_limit_s':budget_limits['generated_shared_limit_s'],
           'generated_new_stage_retained_cap_bytes':16<<20,'new_candidate_attempts':1,'modified_retry_used':0,
           'modified_retry_limit':1,'e359_reservations':{},'e359_original_local_used_s':48.181,
           'e359_generated_local_used_s':11.211,'e359_original_local_limit_s':85,'e359_generated_local_limit_s':25,
           'e360_original_local_limit_s':20,'e360_generated_local_limit_s':budget_limits['generated_local_limit_s'],
           'e360_resource_extension_path':RESOURCE_EXTENSION,'e360_resource_extension':file_spec(RESOURCE_EXTENSION,root),
           'e360_registration_path':REGISTRATION,'e360_amendment_path':AMENDMENT,'e360_amendment':file_spec(AMENDMENT,root),
           'e360_revalidation_amendment':file_spec(OLD_REVALIDATION_AMENDMENT,root),
           'e360_revalidation_activation':revalidation_event(root),
           'e360_active_revalidation_amendment':file_spec(ACTIVE_REVALIDATION_AMENDMENT,root),
           'e360_active_revalidation_activation':active_revalidation_event(root),
           'e360_source_recovery_amendment':file_spec(SOURCE_RECOVERY_AMENDMENT,root),
           'e360_source_recovery_activation':source_recovery_event(root),
           'e360_ruff_recovery_amendment':file_spec(RUFF_RECOVERY_AMENDMENT,root),
           'e360_ruff_recovery_activation':ruff_recovery_event(root),
           'e360_authority_scope_amendment':file_spec(REVALIDATION_AMENDMENT,root),
           'e360_authority_scope_activation':current_revalidation_event(root)}
    if 'e360_execution_budget_activation' in live:
        value=execution_budget_amendment(root)
        frozen=strict_json(read_spec(value['inherited73'],root))['ledger']
        require(len(debits)>=73,'P8 complete inherited73')
        typed_equal(debits[:73],frozen['debits'],'all exact inherited73 debits')
        for key in value['preserved_activation_fields']:
            typed_equal(live[key],frozen[key],'P8 all ten old fields preserved')
        for key in ('e360_resource_grant','e360_calls','e360_passes'):
            if key=='e360_resource_grant':typed_equal(live[key],frozen[key],'P8 old grant')
            else:
                for identity,row in frozen[key].items():typed_equal(live[key][identity],row,'P8 whole historical '+key)
        fixed.update(e360_execution_budget_activation=execution_budget_event(root),
                     e360_execution_budget_amendment=file_spec(EXECUTION_BUDGET_AMENDMENT,root))
    if budget_limits['suite_recovery_active']:
        value=suite_recovery_amendment(root)
        frozen=strict_json(read_spec(value['inherited81'],root))['ledger']
        require(len(debits)>=81,'P9 complete inherited81')
        typed_equal(debits[:81],frozen['debits'],'all exact inherited81 costs descriptors types')
        for key in (*value['preserved_activation_fields'],'e360_resource_grant'):
            typed_equal(live[key],frozen[key],'P9 old12/grant preserved')
        for key in ('e360_calls','e360_passes'):
            for identity,row in frozen[key].items():typed_equal(live[key][identity],row,'P9 all historical81 '+key)
        fixed.update(e360_suite_recovery_activation=suite_recovery_event(root),
            e360_suite_recovery_amendment=file_spec(SUITE_RECOVERY_AMENDMENT,root),
            e360_suite_recovery_resource_grant=suite_recovery_grant(root))
    if historical_lint_scope_authority(root,live=live):
        fixed.update(e360_lint_recovery_activation=lint_recovery_event(root),
            e360_lint_recovery_amendment=file_spec(LINT_RECOVERY_AMENDMENT,root))
    if P11_RECOVERY_FIELDS[0] in live:
        fixed.update(e360_p11_recovery_activation=p11_recovery_event(root),
            e360_p11_recovery_amendment=file_spec(P11_RECOVERY_AMENDMENT,root))
    if historical_p12_scope_authority(root,live=live):
        fixed.update(e360_p12_recovery_activation=p12_recovery_event(root),
            e360_p12_recovery_amendment=file_spec(P12_RECOVERY_AMENDMENT,root))
    if historical_p13_scope_authority(root,live=live):
        fixed.update(e360_p13_recovery_activation=p13_recovery_event(root),
            e360_p13_recovery_amendment=file_spec(P13_RECOVERY_AMENDMENT,root))
    if historical_p14_scope_authority(root,live=live):
        fixed.update(e360_p14_recovery_activation=p14_recovery_event(root),
            e360_p14_recovery_amendment=file_spec(P14_RECOVERY_AMENDMENT,root))
    if historical_p15_scope_authority(root,live=live):
        fixed.update(e360_p15_recovery_activation=p15_recovery_event(root),e360_p15_recovery_amendment=file_spec(P15_RECOVERY_AMENDMENT,root))
    if historical_p16_scope_authority(root,live=live):
        fixed.update(e360_p16_recovery_activation=p16_recovery_event(root),e360_p16_recovery_amendment=file_spec(P16_RECOVERY_AMENDMENT,root))
    if historical_p17_scope_authority(root,live=live):
        fixed.update(e360_p17_recovery_activation=p17_recovery_event(root),e360_p17_recovery_amendment=file_spec(P17_RECOVERY_AMENDMENT,root))
    if validate_p19_activation_triple(live,root):
        fixed.update({key:live[key] for key in ('e360_p18_recovery_activation','e360_p18_recovery_amendment')})
        fixed.update(e360_p19_recovery_activation=p19_recovery_event(root),e360_p19_recovery_amendment=copy.deepcopy(P19_ORIGINAL_POLICY_SPEC),
            e360_p19_recovery_resource_grant=p19_activation_objects(live['e360_p19_recovery_activation']['ROOT_resource_allocation'],root)[1])
    if validate_historical_p20_activation_pair(live,root):fixed.update(e360_p20_recovery_activation=historical_p20_recovery_event(root),e360_p20_recovery_amendment=copy.deepcopy(P20_ORIGINAL_POLICY_SPEC))
    if validate_historical_p21_activation_pair(live,root):fixed.update(e360_p21_recovery_activation=historical_p21_recovery_event(root),e360_p21_recovery_amendment=copy.deepcopy(P21_ORIGINAL_POLICY_SPEC))
    if validate_historical_p22_activation_pair(live,root):fixed.update(e360_p22_recovery_activation=historical_p22_recovery_event(root),e360_p22_recovery_amendment=copy.deepcopy(P22_ORIGINAL_POLICY_SPEC))
    if validate_historical_p23_activation_pair(live,root):fixed.update(e360_p23_recovery_activation=historical_p23_recovery_event(root),e360_p23_recovery_amendment=copy.deepcopy(P23_ORIGINAL_POLICY_SPEC))
    if validate_historical_p25_activation_pair(live,root):fixed.update(e360_p25_recovery_activation=historical_p25_recovery_event(root),e360_p25_recovery_amendment=copy.deepcopy(P25_ORIGINAL_POLICY_SPEC))
    if validate_historical_p27_activation_pair(live,root):fixed.update(e360_p27_recovery_activation=historical_p27_recovery_event(root),e360_p27_recovery_amendment=copy.deepcopy(P27_ORIGINAL_POLICY_SPEC))
    if validate_historical_p29_activation_pair(live,root):fixed.update(e360_p29_recovery_activation=historical_p29_recovery_event(root),e360_p29_recovery_amendment=copy.deepcopy(P29_ORIGINAL_POLICY_SPEC))
    if current_scope_authority(root,live=live):fixed.update(e360_p30_recovery_activation=p30_recovery_event(root),e360_p30_recovery_amendment=copy.deepcopy(CURRENT_REPAIR_SPEC))
    for key,value in fixed.items():
        typed_equal(live[key],value,'preserved live '+key)
    derived={'original_stage_used_s':totals['original'],'original_cumulative_used_s':reg['original_cumulative_before_s']+totals['original'],
             'original_remaining_s':10200-reg['original_cumulative_before_s']-totals['original'],
             'generated_stage_used_s':totals['generated'],'generated_shared_cumulative_used_s':reg['generated_shared_baseline_s']+totals['generated'],
             'generated_shared_remaining_s':budget_limits['generated_shared_limit_s']-reg['generated_shared_baseline_s']-totals['generated']}
    for key,value in derived.items():
        require(type(live[key]) is float and round(numeric(live[key],key),9)==round(value,9),'exact finite live '+key)
    for mode in local:
        typed_equal(live['e360_'+mode+'_local_used_s'],0 if local[mode]==0 else local[mode],'derived E360 local '+mode)
    holds=live['e360_reservations']
    require(type(holds) is dict and len(holds)<=1,'serial E360 role hold')
    for cost_id,held in holds.items():
        require(type(cost_id) is str and 'e360' in cost_id and cost_id not in {v['cost_id'] for v in debits}
                and set(held)=={'mode','role','reserved_s','pending_reservations'},'literal live hold')
        role=held['role']
        require(type(role) is str and role in extension['effective_roles'],'finite held role')
        typed_equal(held['mode'],'original' if role=='saved_prefix_diagnostic' else 'generated','held mode')
        typed_equal(held['reserved_s'],effective_policy(root)['effective_roles'][role]['outer_s'],'exact held reserve')
        typed_equal(held['pending_reservations'],{'suite':effective_policy(root)['effective_roles']['suite']['outer_s']} if role=='fixture' else {},'pending roles')
    original_jobs=sum(v['mode']=='original' for v in holds.values())+sum(v['mode']=='original' for v in live['e360_blocking_holds'].values())
    reserved=sum(v['reserved_s']+sum(v['pending_reservations'].values()) for v in holds.values() if v['mode']=='original')
    typed_equal(live['original_jobs_running'],original_jobs,'live original hold count')
    typed_equal(live['original_reserved_s'],reserved,'live original reserve')
    semantics={**fixed,**{k:live[k] for k in derived},'e360_reservations':holds,
               'original_jobs_running':original_jobs,'original_reserved_s':reserved,
               **{'e360_'+k+'_local_used_s':live['e360_'+k+'_local_used_s'] for k in local}}
    validate_live_calls(live,debits,root=root)
    for key in ('e360_calls','e360_passes','e360_pending_actuals','e360_blocking_holds','e360_resource_grant'):
        semantics[key]=live[key]
    return debits,semantics,totals,local

def validate_live_calls(live,debits,*,root=ROOT):
    context=execution_context(root)
    if context is not None:
        execution_ledger_state(context,root,live)
        return
    """Validate recorded admission/actual costs even while a worker awaits sealing."""
    ext=file_spec(RESOURCE_EXTENSION,root)
    grant=live['e360_resource_grant']
    typed_equal(grant,{'resource_extension':ext,'inherited50':resource_extension(root)['inherited50'],
                       'old_shared_s':150,'old_local_s':39,'new_shared_s':600,'new_local_s':489},'once grant authority')
    calls=live['e360_calls']; passes=live['e360_passes']; pending=live['e360_pending_actuals']
    blocking=live['e360_blocking_holds']
    activated='e360_revalidation_activation' in live
    active='e360_active_revalidation_activation' in live
    source='e360_source_recovery_activation' in live
    ruff='e360_ruff_recovery_activation' in live
    current='e360_authority_scope_activation' in live
    budget='e360_execution_budget_activation' in live
    recovery=current_budget_authority(root,live=live)['suite_recovery_active']
    require(budget==('e360_execution_budget_amendment' in live),'typed complete P8 activation pair')
    lint=historical_lint_scope_authority(root,live=live)
    p11=P11_RECOVERY_FIELDS[0] in live
    p12=historical_p12_scope_authority(root,live=live)
    p13=historical_p13_scope_authority(root,live=live)
    p17=historical_p17_scope_authority(root,live=live);p16=historical_p16_scope_authority(root,live=live);p15=historical_p15_scope_authority(root,live=live);p14=historical_p14_scope_authority(root,live=live)
    p27=validate_historical_p27_activation_pair(live,root)
    p29=validate_historical_p29_activation_pair(live,root)
    p30=current_scope_authority(root,live=live)
    p25=validate_historical_p25_activation_pair(live,root)
    p23=validate_historical_p23_activation_pair(live,root)
    p22=validate_historical_p22_activation_pair(live,root)
    p21=validate_historical_p21_activation_pair(live,root)
    p20=validate_historical_p20_activation_pair(live,root)
    p19=validate_p19_activation_triple(live,root)
    revised=p17_recovery_amendment(root) if p17 else p16_recovery_amendment(root) if p16 else p15_recovery_amendment(root) if p15 else p14_recovery_amendment(root) if p14 else p13_recovery_amendment(root) if p13 else p12_recovery_amendment(root) if p12 else p11_recovery_amendment(root) if p11 else lint_recovery_amendment(root) if lint else suite_recovery_amendment(root) if recovery else execution_budget_amendment(root) if budget else current_revalidation_amendment(root) if current else ruff_recovery_amendment(root) if ruff else source_recovery_amendment(root) if source else active_revalidation_amendment(root) if active else revalidation_amendment(root) if activated else None
    if activated:typed_equal(live['e360_revalidation_activation'],revalidation_event(root),'activated call authority')
    if active:typed_equal(live['e360_active_revalidation_activation'],active_revalidation_event(root),'active recovery call authority')
    if source:
        typed_equal(live['e360_source_recovery_activation'],source_recovery_event(root),'historical source call authority')
        typed_equal(live['e360_source_recovery_amendment'],file_spec(SOURCE_RECOVERY_AMENDMENT,root),'historical source direct pin')
    if ruff:
        require(active and activated,'current recovery requires both preserved activations')
        require(source,'current recovery requires preserved source activation')
        typed_equal(live['e360_ruff_recovery_activation'],ruff_recovery_event(root),'current recovery call authority')
        typed_equal(live['e360_ruff_recovery_amendment'],file_spec(RUFF_RECOVERY_AMENDMENT,root),'current recovery direct pin')
    if current:
        require(ruff,'scope requires preserved Ruff activation')
        typed_equal(live['e360_authority_scope_activation'],current_revalidation_event(root),'current scope call authority')
        typed_equal(live['e360_authority_scope_amendment'],file_spec(REVALIDATION_AMENDMENT,root),'current scope direct pin')
    if budget:
        require(current,'P8 preserves P7 activation')
        typed_equal(live['e360_execution_budget_activation'],execution_budget_event(root),'P8 actual activation')
        typed_equal(live['e360_execution_budget_amendment'],file_spec(EXECUTION_BUDGET_AMENDMENT,root),'P8 direct authority')
    require(all(type(v) is dict for v in (calls,passes,pending,blocking)) and len(passes)<=(28 if p30 else 27 if p29 else 26 if p27 else 25 if p25 else 24 if p23 else 23 if p22 else 22 if p21 else 21 if p20 else 20 if p19 else 19 if 'e360_p18_recovery_activation' in live else 18 if p17 else 17 if p16 else 16 if p15 else 15 if p14 else 14 if p13 else 13 if p12 else 12 if p11 else 11 if lint else 10 if recovery else 9 if budget else 8 if current else 7 if ruff else 6 if source else 5 if active else 4 if activated else 3),'typed finite call state')
    generated=[p for p in passes.values() if p['mode']=='generated']
    require(len(generated)<=(27 if p30 else 26 if p29 else 25 if p27 else 24 if p25 else 23 if p23 else 22 if p22 else 21 if p21 else 20 if p20 else 19 if p19 else 18 if 'e360_p18_recovery_activation' in live else 17 if p17 else 16 if p16 else 15 if p15 else 14 if p14 else 13 if p13 else 12 if p12 else 11 if p11 else 10 if lint else 9 if recovery else 8 if budget else 7 if current else 6 if ruff else 5 if source else 4 if active else 3 if activated else 2) and sum(p['mode']=='original' for p in passes.values())<=1,'finite amended passes')
    old_ids=set(strict_json(read_spec(revised['inherited94' if p17 else 'inherited93' if p16 else 'inherited92' if p15 else 'inherited91' if p14 else 'inherited89' if p13 else 'inherited88' if p12 else 'inherited83' if p11 else 'inherited82' if lint else 'inherited81' if recovery else 'inherited73' if budget else 'inherited68' if current else 'inherited63' if ruff else 'inherited62' if source else 'inherited60' if active else 'inherited59'],root))['ledger']['e360_calls']) if activated else set(calls)
    if p19:old_ids=set(completed_historical_p19_proof(root)['e360_calls'])
    if p20:old_ids=set(completed_historical_p20_proof(root)['e360_calls'])
    if p21:old_ids=set(completed_historical_p21_proof(root)['e360_calls'])
    if p22:old_ids=set(completed_historical_p22_proof(root)['e360_calls'])
    if p23:old_ids=set(completed_historical_p23_proof(root)['e360_calls'])
    if p25:old_ids=set(completed_historical_p25_proof(root)['e360_calls'])
    if p27:old_ids=set(completed_historical_p27_proof(root)['e360_calls'])
    if p29 or p30:old_ids=set(completed_historical_p29_proof(root)['e360_calls'])
    extension=resource_extension(root)
    for pass_id,passed in passes.items():
        require(type(pass_id) is str and re.fullmatch('e360-[a-z0-9-]{1,94}',pass_id),'literal pass ID')
        require(passed['mode'] in ('generated','original') and type(passed['next_index']) is int,'typed pass mode/index')
        order=extension['generated_call_order'] if passed['mode']=='generated' else ['saved_prefix_build','saved_prefix_worker','saved_prefix_seal']
        rows=sorted((v for v in calls.values() if v['pass_id']==pass_id),key=lambda v:v['sequence_index'])
        require(all(type(v['sequence_index']) is int for v in rows)
                and [v['sequence_index'] for v in rows]==list(range(len(rows))),'unique typed call sequence')
        require(0<=passed['next_index']<=len(order) and [v['call_role'] for v in rows]==order[:passed['next_index']],
                'complete prospective ordered pass calls')
        for row in rows:
            request=strict_json(read_spec(row['request'],root,owner='dynamic' if pass_id=='e360-generous-p30' else 'historical'))
            typed_equal(request['source_sha256'],passed['source_sha256'],'one pass source chain')
        if passed.get('modified') is True and pass_id not in ('e360-generous-p15','e360-generous-p16','e360-generous-p17','e360-generous-p18','e360-generous-p19','e360-generous-p20','e360-generous-p21','e360-generous-p22','e360-generous-p23','e360-generous-p25','e360-generous-p27','e360-generous-p29','e360-generous-p30'):
            require(passed['mode']=='generated' and passed.get('actual_failure') is not None
                    and type(passed.get('concrete_fix')) is list and passed['concrete_fix'],'one concretely modified pass')
            failure=charged_cost(passed['actual_failure'],'generated',root)
            require(sum(v['descriptor']==passed['actual_failure'] for v in debits)==1,'modified failure exactly once actually debited')
            require(concrete_failed_predecessor(failure,passed.get('predecessor_failed_cap_s',math.inf)),
                    'concrete failed actual predecessor')
            for spec in passed['concrete_fix']:
                historical_fix_bytes(spec,pass_id=pass_id,root=root) if pass_id in ('e360-generous-p2','e360-generous-p3','e360-generous-p4','e360-generous-p5','e360-generous-p6','e360-generous-p7','e360-generous-p8','e360-generous-p9','e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p13','e360-generous-p14') else read_spec(spec,root)
        if activated and pass_id not in ('e360-generous-p1','e360-generous-p2'):
            typed_equal(passed['revalidation_amendment'],authority_spec_for_pass(pass_id,root),'historical/current pass amendment')
            if passed['mode']=='generated':
                validator=validate_p3_pass if pass_id=='e360-generous-p3' else validate_p4_pass if pass_id=='e360-generous-p4' else validate_p5_pass if pass_id=='e360-generous-p5' else validate_p6_pass if pass_id=='e360-generous-p6' else validate_p7_pass if pass_id=='e360-generous-p7' else validate_p8_pass if pass_id=='e360-generous-p8' else validate_p9_pass if pass_id=='e360-generous-p9' else validate_p10_pass if pass_id=='e360-generous-p10' else validate_completed_p11_pass if pass_id=='e360-generous-p11' else validate_completed_p12_pass if pass_id=='e360-generous-p12' else validate_completed_p13_pass if pass_id=='e360-generous-p13' else validate_completed_p14_pass if pass_id=='e360-generous-p14' else validate_completed_p15_pass if pass_id=='e360-generous-p15' else validate_completed_p16_pass if pass_id=='e360-generous-p16' else validate_completed_p17_pass if pass_id=='e360-generous-p17' else validate_completed_p18_pass if pass_id=='e360-generous-p18' else validate_completed_p19_pass if pass_id=='e360-generous-p19' else validate_completed_p20_pass if pass_id=='e360-generous-p20' else validate_completed_p21_pass if pass_id=='e360-generous-p21' else validate_completed_p22_pass if pass_id=='e360-generous-p22' else validate_completed_p23_pass if pass_id=='e360-generous-p23' else validate_completed_p25_pass if pass_id=='e360-generous-p25' else validate_completed_p27_pass if pass_id=='e360-generous-p27' else validate_completed_p29_pass if pass_id=='e360-generous-p29' else validate_p30_pass
                validator(pass_id,passed,debits,root=root)
    ids={d['cost_id'] for d in debits}
    for call_id,call in calls.items():
        require(type(call_id) is str and re.fullmatch('e360-[a-z0-9-]{1,94}',call_id),'literal call ID')
        require(call['state'] in ('begun','actual_pending','charged'),'finite call state')
        typed_equal(call['resource_extension'],ext,'call extension')
        request=strict_json(read_spec(call['request'],root,owner='dynamic' if call['pass_id']=='e360-generous-p30' else 'historical'))
        require(type(request) is dict and request.get('schema')=='flylab.m3.e360.helper-request.v1','immutable typed call request schema')
        for key in ('call_id','pass_id','call_role','mode'):
            typed_equal(request[key],call_id if key=='call_id' else call[key],'call request '+key)
        typed_equal(request['resource_extension'],ext,'request extension')
        if activated and call_id not in old_ids:
            for value in (request,call):typed_equal(value['revalidation_amendment'],authority_spec_for_pass(call['pass_id'],root),'new call amendment')
        require(call['pass_id'] in passes,'admitted pass')
        role=call['call_role']
        call_policy=effective_policy(root,pass_id=call['pass_id'])
        policy=call_policy['helper_limits'].get(role)
        caps={'ruff':3,'scanner':call_policy['scanner_limits']['outer_s'],'fixture_worker':call_policy['effective_roles']['fixture']['outer_s'],'suite_worker':call_policy['effective_roles']['suite']['outer_s'],'saved_prefix_worker':12}
        require(policy is not None or role in caps,'finite admitted role')
        typed_equal(call['reserved_s'],policy['outer_s'] if policy else caps[role],'exact call cap')
        if call['mode']=='generated':
            suffix=call_policy['successful_suffix_after_s'][call_policy['generated_call_order'].index(role)]
        else:
            suffix=2 if role=='saved_prefix_worker' else policy['successful_suffix_after_s']
        typed_equal(call['future_reserved_s'],suffix,'distinct complete call suffix')
        if call['state']=='charged':
            require(call['cost_id'] in ids,'charged call backed by debit')
    for cost_id,row in pending.items():
        require(cost_id not in ids and cost_id in live['e360_reservations'],'pending replaces uncharged worker')
        actual=strict_json(read_spec(row['actual_tool_return'],root))
        numeric(actual['actual_outer_full_return_s'],'pending external wall',minimum=1e-12)
        require(type(actual['actual_outer_full_return_s']) is float and type(actual['actual_returned_exit_code']) is int,'pending exact external types')
        typed_equal(row['resource_extension'],ext,'pending extension')
        if activated:typed_equal(row['revalidation_amendment'],file_spec(authority_path_for_pass(calls[row['call_id']]['pass_id']),root),'pending amended authority')
        if activated:typed_equal(actual['revalidation_amendment'],row['revalidation_amendment'],'pending actual amendment')
        require(calls[row['call_id']]['state']=='actual_pending','actual transition persisted before seal')
    for cost_id,row in blocking.items():
        require(cost_id in ids and row['debit_backed'] is True and type(row['numeric_reserved_s']) is int
                and row['numeric_reserved_s']==0,'unknown debit-backed nonnumeric hold')
        typed_equal(row['resource_extension'],ext,'blocking hold extension')
        if activated:typed_equal(row['revalidation_amendment'],prospective_authority_spec(root),'blocking amended authority')

def validate_p3_pass(pass_id,passed,debits,*,root=ROOT):
    revised=revalidation_amendment(root)
    typed_equal(pass_id,revised['manual_generated_pass_id'],'only literal manual p3')
    typed_equal(passed['actual_failure'],revised['actual_predecessor_failure'],'exact p2 suite predecessor')
    typed_equal(passed['predecessor_failed_cap_s'],120,'p2 actual suite cap')
    require(passed['modified'] is True and passed['mode']=='generated','manual modified p3')
    require(sum(d['descriptor']==passed['actual_failure'] for d in debits)==1,'p3 predecessor once debited')
    require(concrete_failed_predecessor(charged_cost(passed['actual_failure'],'generated',root),120),'p3 genuine suite failure')
    require(revised['concrete_correction_snapshot'] in passed['concrete_fix'],'actual reviewed correction evidence')
    for spec in passed['concrete_fix']:historical_fix_bytes(spec,pass_id=pass_id,root=root)


def validate_p4_pass(pass_id,passed,debits,*,root=ROOT):
    revised=active_revalidation_amendment(root)
    typed_equal(pass_id,revised['manual_generated_pass_id'],'only literal manual p4')
    typed_equal(passed['actual_failure'],revised['actual_predecessor_failure'],'exact failed p3 Ruff predecessor')
    typed_equal(passed['predecessor_failed_cap_s'],revised['predecessor_failed_cap_s'],'p3 actual Ruff cap')
    require(passed['modified'] is True and passed['mode']=='generated','manual modified p4')
    require(sum(d['descriptor']==passed['actual_failure'] for d in debits)==1,'p4 predecessor once debited')
    failure=charged_cost(passed['actual_failure'],'generated',root)
    require(failure['schema']=='flylab.m3.e360.administrative-full-return.v1'
            and type(failure['actual_returned_exit_code']) is int and failure['actual_returned_exit_code']==0
            and failure['internal_completion_valid'] is False and concrete_failed_predecessor(failure,3),'preserved genuine Ruff administrative failure')
    require(revised['concrete_correction_snapshot'] in passed['concrete_fix'],'actual C4 correction evidence')
    for spec in passed['concrete_fix']:historical_fix_bytes(spec,pass_id=pass_id,root=root)


def validate_p5_pass(pass_id,passed,debits,*,root=ROOT):
    revised=source_recovery_amendment(root)
    require(type(passed['concrete_fix']) is list and type(passed['source_sha256']) is dict,'typed historical p5 source/evidence')
    typed_equal(pass_id,revised['manual_generated_pass_id'],'only literal manual p5')
    typed_equal(passed['actual_failure'],revised['actual_predecessor_failure'],'exact failed p4 scanner predecessor')
    typed_equal(passed['predecessor_failed_cap_s'],25,'p4 actual scanner cap')
    require(passed['modified'] is True and passed['mode']=='generated','manual modified p5')
    require(sum(d['descriptor']==passed['actual_failure'] for d in debits)==1,'p5 predecessor once debited')
    failure=charged_cost(passed['actual_failure'],'generated',root)
    require(failure['schema']=='flylab.m3.e360.administrative-full-return.v1'
            and type(failure['actual_returned_exit_code']) is int and failure['actual_returned_exit_code']==1
            and failure['call_role']=='scanner' and concrete_failed_predecessor(failure,20),'preserved genuine scanner exit1 failure')
    require(revised['concrete_correction_snapshot'] in passed['concrete_fix'],'actual C5 correction evidence')
    authority=ruff_recovery_amendment(root)
    executed=strict_json(read_spec(authority['inherited63'],root))['ledger']['e360_passes']['e360-generous-p5']
    typed_equal(passed['source_sha256'],executed['source_sha256'],'exact historical p5 as-executed source map')
    typed_equal(passed['concrete_fix'],executed['concrete_fix'],'all exact historical p5 concrete evidence')
    for row in authority['historical_concrete_fix_relocations']:
        if row['pass_id']=='e360-generous-p5':
            require(row['original'] in passed['concrete_fix'],'all four historical p5 source descriptors')
            typed_equal(passed['source_sha256'][row['original']['path']],row['original']['sha256'],'historical source SHA binding')
    for spec in passed['concrete_fix']:historical_fix_bytes(spec,pass_id=pass_id,root=root)


def validate_p6_pass(pass_id,passed,debits,*,root=ROOT):
    revised=ruff_recovery_amendment(root)
    typed_equal(pass_id,revised['manual_generated_pass_id'],'only literal manual p6')
    typed_equal(passed['actual_failure'],revised['actual_predecessor_failure'],'exact failed p5 Ruff predecessor')
    typed_equal(passed['predecessor_failed_cap_s'],revised['predecessor_failed_cap_s'],'actual Ruff cap3')
    require(passed['modified'] is True and passed['mode']=='generated','manual modified p6')
    require(sum(d['descriptor']==passed['actual_failure'] for d in debits)==1,'p6 predecessor once debited')
    failure=charged_cost(passed['actual_failure'],'generated',root)
    require(failure['schema']=='flylab.m3.e360.administrative-full-return.v1' and failure['call_role']=='ruff'
            and type(failure['actual_returned_exit_code']) is int and failure['actual_returned_exit_code']==1
            and failure['internal_completion_valid'] is True and concrete_failed_predecessor(failure,3),'preserved Ruff exit1/internaltrue failure')
    require(revised['concrete_correction_snapshot'] in passed['concrete_fix'],'actual C6 correction evidence')
    authority=current_revalidation_amendment(root)
    executed=strict_json(read_spec(authority['inherited68'],root))['ledger']['e360_passes']['e360-generous-p6']
    typed_equal(passed['source_sha256'],executed['source_sha256'],'whole historical p6 as-executed own9')
    typed_equal(passed['concrete_fix'],executed['concrete_fix'],'whole historical p6 evidence')
    for row in authority['historical_concrete_fix_relocations']:
        if row['pass_id']=='e360-generous-p6':
            require(row['original'] in passed['concrete_fix'],'all four historical p6 sources')
            typed_equal(passed['source_sha256'][row['original']['path']],row['original']['sha256'],'historical p6 SHA')
    for spec in passed['concrete_fix']:historical_fix_bytes(spec,pass_id=pass_id,root=root)


def validate_p7_pass(pass_id,passed,debits,*,root=ROOT):
    revised=current_revalidation_amendment(root)
    typed_equal(pass_id,revised['manual_generated_pass_id'],'only literal manual p7')
    typed_equal(passed['actual_failure'],revised['actual_predecessor_failure'],'exact p6 fixture failure')
    typed_equal(passed['predecessor_failed_cap_s'],30,'fixture outer cap30')
    require(passed['modified'] is True and passed['mode']=='generated','manual modified p7')
    require(sum(d['descriptor']==passed['actual_failure'] for d in debits)==1,'p7 predecessor once debited')
    failure=charged_cost(passed['actual_failure'],'generated',root)
    require(failure['schema']=='flylab.m3.e360.actual-full-return.v1' and failure['role']=='fixture'
            and type(failure['exit_code']) is int and failure['exit_code']==1
            and failure['qualified'] is False and failure['all_owned_ended'] is True
            and failure['reservation_released'] is True and concrete_failed_predecessor(failure,30),
            'closed actual p6 fixture worker failure')
    require(revised['concrete_correction_snapshot'] in passed['concrete_fix'],'C7 analysis plus actual changed sources')
    frozen=strict_json(read_spec(execution_budget_amendment(root)['inherited73'],root))['ledger']['e360_passes'][pass_id]
    typed_equal(passed['source_sha256'],frozen['source_sha256'],'P7 exact historical own9')
    typed_equal(passed['concrete_fix'],frozen['concrete_fix'],'P7 exact historical evidence')
    for spec in passed['concrete_fix']:historical_fix_bytes(spec,pass_id=pass_id,root=root)


def validate_p8_pass(pass_id,passed,debits,*,root=ROOT):
    value=execution_budget_amendment(root)
    typed_equal(pass_id,value['manual_generated_pass_id'],'only manual P8 no P9')
    typed_equal(passed['actual_failure'],value['actual_predecessor_failure'],'exact P7 failed fixture')
    typed_equal(passed['predecessor_failed_cap_s'],30,'historical P7 cap30')
    require(passed['modified'] is True and passed['mode']=='generated','manual modified P8')
    require(sum(d['descriptor']==passed['actual_failure'] for d in debits)==1,'P7 failure once actually charged')
    failure=charged_cost(passed['actual_failure'],'generated',root)
    require(failure['role']=='fixture' and failure['exit_code']==1 and failure['qualified'] is False
        and failure['all_owned_ended'] is True and failure['reservation_released'] is True
        and concrete_failed_predecessor(failure,30),'closed genuine P7 cap30 failure')
    require(value['concrete_correction_snapshot'] in passed['concrete_fix'],'C8 actual reviewed correction')
    p8_frozen=strict_json(read_spec(suite_recovery_amendment(root)['inherited81'],root))['ledger']['e360_passes'][pass_id]
    typed_equal(passed,p8_frozen,'P8 immutable actual pass from inherited81')
    for name in (*OWNED,BASE+'/parent-e360-root-account.ps1',BASE+'/parent-e360-root-metadata-helper.py'):
        require(any(spec['path']==name and spec['sha256']==p8_frozen['source_sha256'][name] for spec in passed['concrete_fix']),'P8 four historical C9 sources')
    frozen=strict_json(read_spec(value['inherited73'],root))['ledger']['e360_passes']['e360-generous-p7']
    require(passed['source_sha256']!=frozen['source_sha256'],'P8 changed actual own sources')
    for spec in passed['concrete_fix']:historical_fix_bytes(spec,pass_id=pass_id,root=root)


def validate_p9_pass(pass_id,passed,debits,*,root=ROOT):
    value=suite_recovery_amendment(root)
    typed_equal(pass_id,value['manual_generated_pass_id'],'only manually approved P9 no P10')
    require(current_budget_authority(root)['suite_recovery_active'],'manual P9 live authority')
    typed_equal(passed['actual_failure'],value['actual_predecessor_failure'],'exact actual P8 suite failure descriptor')
    typed_equal(passed['predecessor_failed_cap_s'],120,'immutable P8 historical suite cap120')
    require(passed['modified'] is True and passed['mode']=='generated','manual modified generated P9')
    require(sum(d['descriptor']==passed['actual_failure'] for d in debits)==1,'P8 actual suite failure once debited')
    failure=charged_cost(passed['actual_failure'],'generated',root)
    for key,item in {'pass_id':'e360-generous-p8','role':'suite','exit_code':1,'qualified':False,
        'actual_outer_full_return_s':120.053,'natural_closed':False,'all_owned_ended':True,
        'reservation_released':True,'charged_once':True}.items():typed_equal(failure[key],item,'actual closed killed P8 '+key)
    require(not failure['unresolved_matching_identities'] and not failure['unknown_owned_identities']
        and not failure['missing_owned_creation_times'],'closed killed failure owns no unresolved IDs')
    typed_equal(failure['revalidation_amendment'],file_spec(EXECUTION_BUDGET_AMENDMENT,root),'failure direct old P8 authority')
    contract=strict_json(read_spec(failure['contract'],root))
    typed_equal(contract['pass_id'],'e360-generous-p8','failed exact P8 contract pass')
    typed_equal(contract['role'],'suite','failed exact suite contract role')
    typed_equal(contract['limits'],execution_budget_amendment(root)['effective_roles']['suite'],'failure historical immutable120 caps')
    typed_equal(contract['revalidation_amendment'],failure['revalidation_amendment'],'failure exact direct contract authority')
    frozen=strict_json(read_spec(value['inherited81'],root))['ledger']['e360_passes']['e360-generous-p8']
    typed_equal(failure['source_sha256'],contract['source_sha256'],'actual complete P8 historical source closure')
    require(all(name in failure['source_sha256'] for name in frozen['source_sha256']),'complete P8 own9 in historical closure')
    typed_equal({name:failure['source_sha256'][name] for name in frozen['source_sha256']},
                frozen['source_sha256'],'actual P8 historical own9 projection')
    historical=strict_json(read_spec(lint_recovery_amendment(root)['inherited82'],root))['ledger']['e360_passes'][pass_id]
    typed_equal(passed,historical,'P9 whole actual inherited82 pass/evidence unchanged')
    require(passed['source_sha256']!=frozen['source_sha256'],'P9 concrete changed own sources')
    require(value['concrete_correction_snapshot'] in passed['concrete_fix'],'approved reviewed correction')
    for name in (*OWNED,BASE+'/parent-e360-root-account.ps1',BASE+'/parent-e360-root-metadata-helper.py'):
        require(any(spec['path']==name and spec['sha256']==historical['source_sha256'][name]
            for spec in passed['concrete_fix']),'P9 four frozen v2 sources')
    for spec in passed['concrete_fix']:historical_fix_bytes(spec,pass_id=pass_id,root=root)


def validate_p10_pass(pass_id,passed,debits,*,root=ROOT):
    """Closed historical P10: full83 and frozen4/oldremaining5, never current own9."""
    value=p11_recovery_amendment(root)  # Completed reader; no reader callback reaches this proof.
    typed_equal(pass_id,'e360-generous-p10','historical exact P10')
    frozen=strict_json(read_spec(value['inherited83'],root))['ledger']
    typed_equal(passed,frozen['e360_passes'][pass_id],'whole actual P10 pass unchanged')
    typed_equal(debits[:83],frozen['debits'],'historical full83 exact')
    cost=charged_cost(value['actual_predecessor_failure'],'generated',root)
    for key,item in {'pass_id':pass_id,'call_role':'ruff','actual_outer_full_return_s':0.885,
        'actual_returned_exit_code':1,'internal_completion_valid':False,'charged_once':True,
        'capture_complete':True,'capture_publication_verified':True,'capture_owned_state':'exit_observed'}.items():
        typed_equal(cost[key],item,'historical false P10 cost '+key)
    require(sum(row['descriptor']==value['actual_predecessor_failure'] for row in debits)==1,'P10 failure once debit')
    require(cost['capture_proof_errors']==[{'stage':'ROOT-capture-proof','message':'Typed actual capture return'}],
        'historical integer guard failure preserved')
    typed_equal(cost['source_sha256'],passed['source_sha256'],'P10 cost frozen own9')
    typed_equal(cost['revalidation_amendment'],value['previous_revalidation_amendment'],'historical P10 scope')
    call=frozen['e360_calls'][cost['call_id']]
    require(call['state']=='charged' and passed['next_index']==1 and passed['failed_cost'] is not None,'closed first Ruff failure')
    request=strict_json(read_spec(cost['request'],root))
    typed_equal(request['source_sha256'],passed['source_sha256'],'P10 request old9')
    typed_equal(request['revalidation_amendment'],cost['revalidation_amendment'],'P10 direct old scope')
    summary=strict_json(read_spec(value['historical_P10_genuine_capture_return'],root))
    manifest=strict_json(read_spec(value['historical_P10_capture_manifest'],root))
    closure=strict_json(read_spec(value['historical_P10_independent_owned_closure'],root))
    typed_equal(summary['manifest_descriptor'],value['historical_P10_capture_manifest'],'genuine manifest pin')
    typed_equal(manifest['request'],cost['request'],'P10 same admitted request')
    typed_equal(manifest['revalidation_amendment'],cost['revalidation_amendment'],'P10 manifest scope')
    for key in ('process_identity','owned_state','exit_observed','ruff_exit_code','capture_complete','streams','first_failure'):
        typed_equal(summary[key],manifest[key],'genuine capture evidence '+key)
    require(summary['capture_complete'] is True and summary['publication_verified'] is True and
        summary['exit_observed'] is True and summary['ruff_exit_code']==1 and summary['wrapper_error_slots']==[] and
        manifest['wrapper_errors']=={},'complete genuine failed Ruff capture')
    typed_equal(closure['charged_failure'],value['actual_predecessor_failure'],'independent failure closure pin')
    typed_equal(closure['genuine_capture_return'],value['historical_P10_genuine_capture_return'],'independent return pin')
    typed_equal(closure['manifest'],value['historical_P10_capture_manifest'],'independent manifest pin')
    typed_equal(closure['owned_identity'],summary['process_identity'],'same owned identity')
    typed_equal(closure['streams'],summary['streams'],'independent EOF streams')
    require(closure['all_owned_ended'] is True and closure['unknown_owned_identities']=={} and
        closure['actual_once_charge_s']==0.885 and closure['native_capture_proof_rejection_preserved'] is True,
        'independent closure never flips false cost')
    for row in summary['streams']:
        require(row['eof_observed'] is True and row['readback_verified'] is True and
            row['read_fault'] is False and row['overflow'] is False,'both historical EOF')
        spec=row['descriptor'];path=safe(spec['path'],root)
        require(path.stat().st_size==spec['bytes'] and sha(path.read_bytes())==spec['sha256'],'exact historical raw streams')
        typed_equal(row['retained_bytes'],spec['bytes'],'historical retained bytes')
        typed_equal(row['observed_bytes'],row['retained_bytes'],'historical complete bytes')
    relocated={row['original']['path'] for row in value['historical_concrete_fix_relocations'][30:]}
    require(len(relocated)==4 and len(passed['source_sha256'])==9,'P10 frozen4 oldremaining5')
    for name,digest in passed['source_sha256'].items():
        if name not in relocated:typed_equal(file_spec(name,root,owner='dynamic')['sha256'],digest,'historical remaining5 source')
    for spec in passed['concrete_fix']:historical_fix_bytes(spec,pass_id=pass_id,root=root)

def validate_p11_pass(pass_id,passed,debits,*,root=ROOT):
    """Compatibility name for completed history; never fresh prospective P11 admission."""
    return validate_completed_p11_pass(pass_id,passed,debits,root=root)

def concrete_failed_predecessor(cost,cap):
    """Only a genuine charged E360 failure permits one fresh modified generated pass."""
    allowed={'flylab.m3.e360.actual-full-return.v1','flylab.m3.e360.administrative-full-return.v1'}
    if type(cost) is not dict or cost.get('schema') not in allowed or cost.get('mode')!='generated' or cost.get('charged_once') is not True:
        return False
    worker=cost['schema']=='flylab.m3.e360.actual-full-return.v1'
    if worker and cost.get('qualified') is not False:
        return False
    code=cost.get('actual_returned_exit_code',cost.get('exit_code'))
    wall=cost.get('actual_outer_full_return_s')
    if type(code) is not int or type(wall) is not float or not math.isfinite(wall) or wall<=0:
        return False
    return code!=0 or wall>cap or (not worker and cost.get('internal_completion_valid') is False) or worker


def prospective_remaining(semantics,mode,*,exclude_call=None,root=ROOT):
    """Distinct future suffix plus unreturned caps and undebited actuals, once each."""
    calls=semantics['e360_calls']
    current=[v for k,v in calls.items() if v['mode']==mode and v['state']=='begun' and k!=exclude_call]
    pending=[strict_json(read_spec(v['actual_tool_return'],root))['actual_outer_full_return_s']
             for v in semantics['e360_pending_actuals'].values() if v['mode']==mode]
    identity=execution_current_pass(root) if 'execution_context_activation' in semantics else 'e360-generous-p30'
    for call in current:
        if call['pass_id']==identity:validate_p30_call_future(call,root=root,pass_id=identity)
    suffix=max((v['future_reserved_s'] for v in current),default=0)
    # Serial calls share the same necessary suffix; no summing overlapping futures.
    if mode=='generated' and identity in semantics['e360_passes']:
        suffix=max(suffix,current_p30_unused_fallback(semantics,root))
    call_route=math.fsum(pending)+math.fsum(v['reserved_s'] for v in current)+suffix
    if mode=='generated' and identity in semantics['e360_passes']:
        unused=current_p30_unused_fallback(semantics,root);hold_route=0
        for identity,held in semantics['e360_reservations'].items():
            if held['mode']!='generated':continue
            own=held['reserved_s']
            if identity in semantics['e360_pending_actuals']:
                own=strict_json(read_spec(semantics['e360_pending_actuals'][identity]['actual_tool_return'],root))['actual_outer_full_return_s']
            primary_future=125 if held['role']=='fixture' else 65
            hold_route=max(hold_route,own+sum(held['pending_reservations'].values())+primary_future+unused)
        call_route=max(call_route,hold_route)
    return call_route

def require_fit(reg,semantics,mode,*,root=ROOT):
    used=semantics['e360_'+mode+'_local_used_s']
    necessary=prospective_remaining(semantics,mode,root=root)
    cumulative=semantics['original_cumulative_used_s'] if mode=='original' else semantics['generated_shared_cumulative_used_s']
    stage=semantics['original_stage_used_s'] if mode=='original' else semantics['generated_stage_used_s']
    budget=current_budget_authority(root)
    require(used+necessary<=(20 if mode=='original' else budget['generated_local_limit_s'])
            and cumulative+necessary<=(10200 if mode=='original' else budget['generated_shared_limit_s'])
            and (mode!='original' or stage+necessary<=300),'live actual/pending/full future admission fits')


def names(attempt_id,role):
    require(type(attempt_id) is str and re.fullmatch('e360-[a-z0-9-]{1,48}',attempt_id),'unique literal E360 attempt ID')
    stem=BASE+'/parent-e346-e360-'+attempt_id+'-'+role
    return {'contract':stem+'.json','output':BASE+'/e360-'+attempt_id+'-'+role,
            **{key:stem+'-'+suffix for key,suffix in {'pre_spawn':'pre-spawn.json','outer':'outer.json',
            'outer_revocation':'outer-revocation.json','actual_return':'actual-tool-return.json','full_exit':'full-exit.json',
            'supervision':'supervision.json','stdout':'child.stdout.txt','stderr':'child.stderr.txt',
            'witness_stdout':'witness.stdout.txt','witness_stderr':'witness.stderr.txt'}.items()}}


def build(role,attempt_id,*,closure_report=None,adapter_qualification=None,inputs=None,root=ROOT):
    reg=registration(root)
    resource_extension(root)
    require(type(role) is str and role in reg['roles'],'explicit E360 role')
    mode='original' if role=='saved_prefix_diagnostic' else 'generated'
    original=mode=='original'
    debits,semantics,totals,local=ledger_state(reg,root)
    closure_report=file_spec(BASE+'/parent-e360-redistributed-current-closure.json',root) if closure_report is None else closure_report
    report=strict_json(read_spec(closure_report,root))
    # Recheck real AST and every file; metadata hashes alone cannot stand in for discovery.
    typed_equal(report,scan_current(reg,root=root),'CURRENT actual source AST closure')
    require(report['diagnostic_ready'] is True,'unresolved executable diagnostic dependency')
    paths=names(attempt_id,role)
    argv=[PYTHON,'-X','utf8','-B',OWNED[0],'--child',paths['contract'],'--pre-spawn',paths['pre_spawn']]
    admin=0 if execution_bootstrap(root) is not None else math.fsum(v['actual_outer_full_return_s'] for v in (strict_json(read_spec(d['descriptor'],root)) for d in debits[42:])
                   if v['mode']=='original' and 'administrative' in v['schema'])
    build_role='saved_prefix_build' if original else role+'_build'
    admitted=[(key,value) for key,value in semantics['e360_calls'].items()
              if value['call_role']==build_role and value.get('attempt_id')==attempt_id]
    require(len(admitted)==1,'one admitted build helper for current attempt')
    build_call_id,build_call=admitted[0]
    policy=effective_policy(root,pass_id=build_call['pass_id'])
    limits=dict(policy['effective_roles'][role])
    budget=current_budget_authority(root)
    require(not original or prospective_authority_spec(root)['path'] not in (EXECUTION_BUDGET_AMENDMENT,SUITE_RECOVERY_AMENDMENT,LINT_RECOVERY_AMENDMENT,P11_RECOVERY_AMENDMENT,P12_RECOVERY_AMENDMENT,P13_RECOVERY_AMENDMENT,P14_RECOVERY_AMENDMENT,P15_RECOVERY_AMENDMENT,P16_RECOVERY_AMENDMENT,P17_RECOVERY_AMENDMENT),'this current generated scope ends after Q; a later original role needs separate valid ROOT admission under general human delegation')
    future_admin=max(0,8-admin) if original else (policy['fixture_future_administration_reserved_s'] if role=='fixture' else policy['suite_future_administration_reserved_s']) if build_call['pass_id'] in ('e360-generous-p8','e360-generous-p9','e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p13','e360-generous-p14','e360-generous-p15','e360-generous-p16','e360-generous-p17','e360-generous-p18','e360-generous-p19','e360-generous-p20',execution_current_pass(root)) else policy['remaining_generated_administration_reserve_s'][role]
    return {'schema':'flylab.m3.e360.supervisor-contract.v1','mode':mode,'role':role,'attempt':1,'attempt_id':attempt_id,
            'phase':'E360-'+role,'original_calls':int(original),'registration':file_spec(REGISTRATION,root),
            'amendment':file_spec(AMENDMENT,root),'resource_extension':file_spec(RESOURCE_EXTENSION,root),'revalidation_amendment':authority_spec_for_pass(build_call['pass_id'],root),
            'pass_id':build_call['pass_id'],'build_call_id':build_call_id,'future_administration_reserved_s':future_admin,
            **({'same_pending_fallback_reserved_s':current_p30_unused_fallback(semantics)} if build_call['pass_id']==execution_current_pass(root) else {}),
            'source_sha256':closure_sources(reg,report,root,closure_report=closure_report),'closure_report':closure_report,'limits':limits,'paths':paths,
            'commands':[{'argv':argv,'stdout':paths['stdout'],'stderr':paths['stderr'],'wall_s':limits['child_s']}],
            'supervisor_output':paths['supervision'],'max_tree_rss_bytes':limits['host_bytes'],
            'cumulative_before_s':(reg['original_cumulative_before_s'] if original else reg['generated_shared_baseline_s'])+totals[mode],
            'cumulative_limit_s':10200 if original else budget['generated_shared_limit_s'],'stage_used_s':totals[mode],
            'stage_limit_s':300 if original else budget['generated_shared_limit_s']-reg['generated_shared_baseline_s'],
            'local_used_s':local[mode],'local_limit_s':20 if original else budget['generated_local_limit_s'],'reserved_s':limits['outer_s'],
            'pending_reservations':{'suite':policy['effective_roles']['suite']['outer_s']} if role=='fixture' else {},'ledger_debits':debits,'ledger_semantics':semantics,
            'adapter_qualification':adapter_qualification,'inputs':inputs or {},
            'retained_reserve_bytes':min(1<<20,limits['output_cap_bytes']//4),'cpu_threads':1}

def validate_contract(name,*,root=ROOT,child=False):
    raw=bounded_json_name(name,root,cap=524288)
    require(len(raw)<=512<<10,'bounded E360 contract')
    c=strict_json(raw)
    expected=build(c['role'],c['attempt_id'],closure_report=c['closure_report'],
                   adapter_qualification=c['adapter_qualification'],inputs=c['inputs'],root=root)
    typed_equal(c,expected,'identical full typed E360 authority')
    require(name==c['paths']['contract'],'literal fresh contract path')
    check_sources(c['source_sha256'],root)
    cost_id=c['attempt_id']+':'+c['role']
    calls=c['ledger_semantics']['e360_calls']
    require(calls[c['build_call_id']]['state']=='charged','build actual debit before worker contract')
    role=c['role']+'_worker' if c['mode']=='generated' else 'saved_prefix_worker'
    workers=[v for v in calls.values() if v['call_role']==role and v.get('attempt_id')==c['attempt_id']]
    require(len(workers)==1 and workers[0]['state']=='begun' and workers[0]['pass_id']==c['pass_id'],
            'prospective immutable worker admission')
    require(not c['ledger_semantics']['e360_blocking_holds'],'unknown closure blocks worker')
    require_fit(registration(root),c['ledger_semantics'],c['mode'],root=root)
    typed_equal(c['ledger_semantics']['e360_reservations'].get(cost_id),
                {k:c[k] for k in ('mode','role','reserved_s','pending_reservations')},'actual exact role hold')
    new=[strict_json(read_spec(v['descriptor'],root)) for v in c['ledger_debits'][42:]]
    if c['mode']=='original':
        require(not any(v.get('original_attempt_role',v.get('role'))==c['role'] for v in new),'one original diagnostic max1')
        validate_qualification(c['adapter_qualification'],root=root)
    else:
        require(c['adapter_qualification'] is None,'generated original0')
    require(set(c['inputs'])==({'case_contract'} if c['role']=='suite' else set()),'finite role inputs')
    downstream=sum(c['pending_reservations'].values())
    future=c['future_administration_reserved_s']
    if c['pass_id']==execution_current_pass(root):
        validate_p30_primary_future(c)
        validate_p30_fallback_balance(c['same_pending_fallback_reserved_s'],c['ledger_semantics'])
        future+=c['same_pending_fallback_reserved_s']
    require(c['local_used_s']+c['reserved_s']+downstream+future<=c['local_limit_s']
            and c['stage_used_s']+c['reserved_s']+downstream+future<=c['stage_limit_s']
            and c['cumulative_before_s']+c['reserved_s']+downstream+future<=c['cumulative_limit_s'],'all actual local/shared prospective caps')
    keys=('output','outer','outer_revocation','actual_return','full_exit')
    if not child:
        keys+=('pre_spawn','supervision','stdout','stderr','witness_stdout','witness_stderr')
    require(all(not safe(c['paths'][k],root).exists() for k in keys),'fresh E360 outputs/no replay')
    return c,descriptor(name,raw),registration(root)

def envelope(c,spec,receipt):
    return {'schema':'flylab.m3.e360.provisional-envelope.v1','contract':spec,'pre_spawn':receipt,
            'mode':c['mode'],'role':c['role'],'attempt_id':c['attempt_id'],'source_sha256':c['source_sha256'],
            'registration':c['registration'],'amendment':c['amendment'],'resource_extension':c['resource_extension'],'revalidation_amendment':c['revalidation_amendment'],'closure_report':c['closure_report'],
            'adapter_qualification':c['adapter_qualification'],
            'run_valid':False,'effective_qualified':False,'revoked':False,'classification':'not_started',
            'first_error':None,'secondary_errors':[],'actual_full_return_pending':True,
            'cumulative_before_s':c['cumulative_before_s'],'cumulative_after_s':None,
            'cumulative_limit_s':c['cumulative_limit_s'],**FALSE_MILESTONES}


def shared_owner(c,name,pid,created,*,root=ROOT):
    import psutil
    if c['mode']=='original':
        require(type(pid) is int and pid>0 and type(created) is float,'typed shared owner')
        typed_equal(c['ledger_semantics']['original_jobs_running'],1,'one held original owner')
        typed_equal(psutil.Process(pid).create_time(),created,'live shared owner PID/create-time')
        typed_equal(strict_json(safe(BASE+'/parent-e358-original-active.lock',root).read_bytes()),
                    {'pid':pid,'created':created,'contract':name},'actual shared lock owner')


def pre_spawn(name,entry,*,root=ROOT):
    check=operation_clock(name,entry,root=root)
    with fixed_authority_scope(root=root,check=check) as scope:
        if type(scope.check) is OperationClock:
            typed_equal(entry,scope.check.timer.entry,'parent inherited admitted worker ENTRY')
        return _pre_spawn(name,entry,root=root)


def pre_array(name,receipt_name,*,root=ROOT):
    raw=bounded_json_name(receipt_name,root,cap=262144)
    receipt=strict_json(raw)
    require(type(receipt.get('entry_perf_counter')) is float,'actual typed parent ENTRY before scope')
    admitted_entry=receipt['entry_perf_counter']  # Capture before any subsequent receipt reread.
    check=operation_clock(name,admitted_entry,root=root,stage='internal')
    with fixed_authority_scope(root=root,check=check) as scope:
        if type(scope.check) is OperationClock:
            typed_equal(admitted_entry,scope.check.timer.entry,'pre-array inherited admitted worker ENTRY')
        value=_pre_array(name,receipt_name,root=root,admitted_entry=admitted_entry)
        authority_fence('post_pre_array_before_first_numeric_native_or_array')
        return value


def supervise(name,*,entry=ENTRY,root=ROOT):
    scope_stack=contextlib.ExitStack()
    try:
        check=operation_clock(name,entry,root=root)
        check()
        scope_stack.enter_context(fixed_authority_scope(root=root,check=check))
    except Exception as error:
        scope_stack.close()
        # Construction, immutable-cap/authority and expiry failures enter the
        # same bounded publisher before Popen, retaining the actual first error.
        return _supervise(name,entry=entry,root=root,initial_error=error)
    with scope_stack:
        return _supervise(name,entry=entry,root=root)


def child(name,receipt_name):
    raw=bounded_json_name(receipt_name,cap=262144)
    entry=strict_json(raw)['entry_perf_counter']
    require(type(entry) is float,'actual parent ENTRY for child')
    check=operation_clock(name,entry,stage='internal')
    with fixed_authority_scope(check=check):
        c,spec,receipt_spec,reg,timer=pre_array(name,receipt_name)
        authority_fence('post_pre_array_before_first_numeric_native_or_array')
        with descriptors_only_execution():
            return _child(c,spec,receipt_spec,reg,timer)


def pre_popen_admission(c,name,spec,*,root=ROOT):
    """Receipt has closed; dynamic admission and all text fences remain fresh."""
    reproduced,current_spec,_=validate_contract(name,root=root,child=True)
    typed_equal(reproduced,c,'actual Popen current contract/budget/Q/hold')
    typed_equal(current_spec,spec,'actual Popen contract descriptor')
    check_sources(c['source_sha256'],root)
    if _AUTHORITY_SCOPE is not None:_AUTHORITY_SCOPE.clear_parsed()
    authority_fence('actual_Popen_after_receipt')

def _pre_spawn(name, entry, *, root=ROOT):
    c, spec, frozen = validate_contract(name, root=root)
    timer = Clock(entry, c["limits"])
    timer.check("outer")
    import psutil
    owner = psutil.Process(os.getpid())
    shared_owner(c,name,owner.pid,owner.create_time(),root=root)
    receipt=compact_receipt(c,spec,entry,owner.pid,owner.create_time())
    validate_current_receipt(c,spec,receipt,root=root,entry=entry,check=lambda:timer.check('outer'))
    require(not any(k == "torch" or k == "numpy" or k.startswith("flylab.") for k in sys.modules),
            "pre-spawn numeric/native imports0")
    durable(safe(c["paths"]["pre_spawn"], root), json_bytes(receipt), cap=256 << 10,
            check=lambda: timer.check("outer"))
    receipt_spec=descriptor(c['paths']['pre_spawn'],json_bytes(receipt))
    stored=strict_json(bounded_json_read(receipt_spec,root,cap=262144,check=lambda:timer.check('outer')))
    validate_current_receipt(c,spec,stored,root=root,entry=entry,check=lambda:timer.check('outer'))
    authority_fence('post_durable_close_readback')
    return c,spec,receipt_spec,frozen


def _pre_array(name, receipt_name, *, root=ROOT, admitted_entry):
    c, spec, frozen = validate_contract(name, root=root, child=True)
    if c.get('pass_id') in ('e360-generous-p8','e360-generous-p9','e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p13','e360-generous-p14','e360-generous-p15','e360-generous-p16','e360-generous-p17','e360-generous-p18','e360-generous-p19','e360-generous-p20',execution_current_pass(root)):
        typed_equal((p17_recovery_amendment(root) if c['pass_id']=='e360-generous-p17' else p16_recovery_amendment(root) if c['pass_id']=='e360-generous-p16' else p15_recovery_amendment(root) if c['pass_id']=='e360-generous-p15' else p14_recovery_amendment(root) if c['pass_id']=='e360-generous-p14' else p13_recovery_amendment(root) if c['pass_id']=='e360-generous-p13' else p12_recovery_amendment(root) if c['pass_id']=='e360-generous-p12' else p11_recovery_amendment(root) if c['pass_id']=='e360-generous-p11' else lint_recovery_amendment(root) if c['pass_id']=='e360-generous-p10' else suite_recovery_amendment(root) if c['pass_id']=='e360-generous-p9' else execution_budget_amendment(root))['checkpoint_policy'],compiled_checkpoint_policy(),'pre-array immutable checkpoint policy')
    require(receipt_name == c["paths"]["pre_spawn"], "same actual pre-spawn receipt path")
    receipt_raw=bounded_json_name(receipt_name,root,cap=262144)
    receipt_spec=descriptor(receipt_name,receipt_raw)
    receipt=strict_json(receipt_raw)
    validate_current_receipt(c,spec,receipt,root=root,entry=admitted_entry)
    entry = numeric(receipt["entry_perf_counter"], "ENTRY", minimum=1e-12)
    import psutil
    require(type(entry) is float and type(receipt["parent_pid"]) is int and receipt["parent_pid"] > 0
            and type(receipt["parent_create_time"]) is float,
            "typed actual parent identity")
    numeric(receipt["parent_create_time"], "parent creation time", minimum=1e-12)
    typed_equal(psutil.Process(receipt["parent_pid"]).create_time(), receipt["parent_create_time"],
            "actual live parent PID/create-time identity")
    shared_owner(c,name,receipt['parent_pid'],receipt['parent_create_time'],root=root)
    require(not any(k == "torch" or k == "numpy" or k.startswith("flylab.") for k in sys.modules),
            "pre-array numeric/native imports0")
    timer = Clock(admitted_entry, c["limits"])
    timer.check("science")
    return c, spec, receipt_spec, frozen, timer

def cleanup_matching(identities, process_api, *, timer=None):
    killed, errors = [], []
    for pid, created in reversed(list(identities.items())):
        try:
            if timer is not None:
                require(timer.remaining(OUTER_REPORT_MARGIN_S) > 0, "same-ENTRY cleanup time remaining")
            item = process_api.Process(pid)
            if item.create_time() == created:
                item.kill()
                killed.append({"pid": pid, "create_time": created})
        except process_api.NoSuchProcess:
            pass
        except Exception as error:
            errors.append(error_record(error, "matching-owned-cleanup"))
    return killed, errors


def owned_presence(identities, process_api):
    """A reused PID has ended the recorded identity; unreadable identity is unknown."""
    matching, unknown, errors = {}, {}, []
    for pid, created in identities.items():
        try:
            if process_api.Process(pid).create_time() == created:
                matching[pid] = created
        except process_api.NoSuchProcess:
            pass
        except Exception as error:
            unknown[pid] = created
            errors.append(error_record(error, "owned-identity-completion"))
    return matching, unknown, errors


def bounded_owned_completion(identities, timer, process_api, *, sleep=time.sleep):
    errors = []
    while True:
        matching, unknown, observed_errors = owned_presence(identities, process_api)
        errors.extend(observed_errors)
        if not matching or unknown:
            return matching, unknown, errors
        try:
            remaining = timer.remaining(OUTER_REPORT_MARGIN_S)
            if remaining <= 0:
                raise TimeoutError("E360 same-ENTRY owned completion deadline")
            sleep(min(.005, remaining))
        except Exception as error:
            errors.append(error_record(error, "owned-completion-wait"))
            return matching, unknown, errors


def failure_metadata(c,target,record,timer,*,stage,root=ROOT):
    """Close bounded failed metadata only; a failed fence never admits success.

    Fresh fence/clock attempts surround a separate failed evidence write. The
    final failure record retains their errors and that actual closed descriptor.
    Its final close/readback is deliberately uncached, without a timely-success
    assertion. No arrays, Q, ownership changes or deadline resets are allowed.
    """
    allowed=(c['paths'].get('outer'),c['paths'].get('outer_revocation'),
             c['paths']['output']+'/envelope.json',c['paths']['output']+'/revocation.json')
    require(type(target) is str and target in allowed and record.get('first_error') is not None,
            'explicit failed metadata authority')
    record.update(run_valid=False,effective_qualified=False,revoked=True,classification='technical_invalid')
    for label,check in (('clock',lambda:timer.check(stage)),
                        ('fence',lambda:authority_fence('failure_prepublication'))):
        try:check()
        except Exception as error:fail(record,error,'failure_prepublication-'+label)
    evidence=target+'.failure-evidence.json'
    evidence_spec=None
    try:
        payload=json_bytes(record)
        require(retained(c,root)+len(payload)+(128<<10)<=c['limits']['output_cap_bytes'],
                'failure evidence retained cap')
        path=safe(evidence,root)
        lib.durable(path,payload,cap=256<<10)
        evidence_spec=lib.file_spec(evidence,root)
    except Exception as error:
        fail(record,error,'failure-evidence-close-readback')
    for label,check in (('clock',lambda:timer.check(stage)),
                        ('fence',lambda:authority_fence('failure_post_durable_close_readback'))):
        try:check()
        except Exception as error:fail(record,error,'failure-post-close-'+label)
    record['failure_publication']={'entry_perf_counter':timer.entry,'timely_success':False,
                                   'closed_failed_evidence':evidence_spec}
    try:
        payload=json_bytes(record)
        require(retained(c,root)+len(payload)<=c['limits']['output_cap_bytes'],'failure final retained cap')
        lib.durable(safe(target,root),payload,cap=256<<10)
        return lib.file_spec(target,root)
    except Exception as error:
        # An unwritable/quota-exhausted path cannot be invented as closed. Keep
        # the original error and the additional error in the returned record.
        fail(record,error,'failure-final-close-readback')
        return None


def publish_outer(c, record, timer, *, root=ROOT):
    """Retain a provisional write and a separate late revocation, never overwrite."""
    path = safe(c["paths"]["outer"], root)
    try:
        durable(path, json_bytes(record), cap=256 << 10, check=lambda: timer.check("outer"))
        timer.check("outer")
        require(retained(c, root) + (128 << 10) <= c["limits"]["output_cap_bytes"], "outer actual retained cap")
        timer.check("outer")
    except Exception as error:
        fail(record, error, "outer-publication")
        record["natural_closed"] = False
        # Emergency evidence claims no timely success and keeps the same ENTRY.
        target = c["paths"]["outer_revocation"] if path.exists() else c["paths"]["outer"]
        failure_metadata(c,target,record,timer,stage='outer',root=root)
    return record


def _supervise(name, *, entry=ENTRY, root=ROOT, initial_error=None):
    """Observe the immutable observers; kill only PID/create-time owned identities."""
    import psutil
    lock = safe(BASE + "/parent-e358-original-active.lock", root)
    raw_c = strict_json(bounded_json_name(name,root,cap=524288))
    # A failed authority can be charged and closed, never admitted. Derive
    # failure output paths/caps from frozen role and literal canonical name.
    fallback = failed_authority_view(raw_c,name,root=root)
    held = fallback["mode"] == "original"
    token = {"pid": os.getpid(), "created": psutil.Process(os.getpid()).create_time(), "contract": name}
    lock_owned = False
    popen_attempted = False
    process = None
    identities, killed, secondary, first = {}, [], [], None
    c = spec = receipt = timer = None
    matching, unknown, missing_identities = {}, {}, []
    supervisor_killed = False
    try:
        if initial_error is not None:raise initial_error
        if held:
            durable(lock, json_bytes(token), cap=1024)
            lock_owned = True
        c, spec, receipt, _ = pre_spawn(name, entry, root=root)
        timer = Clock(entry, c["limits"])
        argv = [PYTHON, "-X", "utf8", "-B", OBSERVER, name]
        with safe(c["paths"]["witness_stdout"], root).open("xb") as out, \
                safe(c["paths"]["witness_stderr"], root).open("xb") as err:
            require(timer.remaining(OUTER_CLEANUP_MARGIN_S) > 0, "outer cleanup/publication margin")
            shared_owner(c,name,token['pid'],token['created'],root=root)
            pre_popen_admission(c,name,spec,root=root)
            popen_attempted = True
            process = subprocess.Popen(argv, cwd=root, stdout=out, stderr=err,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            owner = psutil.Process(process.pid)
            identities[owner.pid] = owner.create_time()
            while process.poll() is None:
                if timer.remaining(OUTER_CLEANUP_MARGIN_S) <= 0:
                    raise TimeoutError("E360 same-ENTRY outer work deadline with cleanup margin")
                try:
                    tree = [owner, *owner.children(recursive=True)]
                except psutil.NoSuchProcess:
                    tree = []
                rss = 0
                for item in tree:
                    try:
                        created = item.create_time()
                        require(item.pid not in identities or identities[item.pid] == created, "observed PID identity reuse")
                        identities[item.pid] = created
                        rss += item.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                require(rss <= c["limits"]["host_bytes"], "outer owned host tree cap")
                require(retained(c, root) + c["retained_reserve_bytes"] <= c["limits"]["output_cap_bytes"],
                        "outer stdout/report recursive retention")
                time.sleep(min(.005, timer.remaining(OUTER_CLEANUP_MARGIN_S)))
            code = process.returncode
            timer.check("outer")
        supervision = strict_json(safe(c["paths"]["supervision"], root).read_bytes())
        supervisor_killed = any(row["killed"] is not False for row in supervision["attempts"])
        identity_path = safe(c["paths"]["supervision"], root).with_name(
            Path(c["paths"]["supervision"]).stem + "-identities.json")
        identity = strict_json(identity_path.read_bytes())
        # Ingest actual descendant identities even when scientific work failed.
        for key, created in identity["pid_create_times"].items():
            pid = int(key)
            require(type(created) is float and math.isfinite(created) and created > 0,
                    "typed observer creation time")
            require(pid not in identities or identities[pid] == created, "matching observed ownership")
            identities[pid] = created
        missing_identities = sorted(set(supervision["owned_pids"]) - set(identities))
        secondary.extend({**row, "stage": "unchanged-creation-time-observer"} for row in identity["errors"])
        require(not missing_identities, "all observed owned identities known")
        require(supervision["success"] is True and supervision["contract_sha256"] == spec["sha256"]
                and len(supervision["attempts"]) == 1 and supervision["attempts"][0]["argv"] == c["commands"][0]["argv"]
                and supervision["attempts"][0]["killed"] is False and code == 0,
                "actual unchanged supervisor natural job exit")
        require(identity["qualified"] is True and identity["errors"] == []
                and identity["all_owned_pids_ended"] is True and identity["contract"] == spec,
                "actual unchanged creation-time observer closure")
    except Exception as error:
        first = error_record(error, "outer")
        if process is not None and process.pid not in identities:
            missing_identities.append(process.pid)
        if identities and timer is not None:
            killed, errors = cleanup_matching(identities, psutil, timer=timer)
            secondary.extend(errors)
    if c is None:
        c, spec = fallback, file_spec(name,root)
    if timer is None:
        timer = Clock(entry,c['limits'])
    matching, unknown, closure_errors = bounded_owned_completion(identities, timer, psutil)
    if popen_attempted and process is None:
        unknown['Popen-unreturned'] = 'process creation attempted; returned identity unknown'
    secondary.extend(closure_errors)
    ended = (not popen_attempted or process is not None and process.poll() is not None) and not matching and not unknown and not missing_identities
    if not ended and first is None:
        first = error_record(RuntimeError("owned identities unresolved at outer completion"), "owned-completion")
    if closure_errors and first is None:
        first, secondary = secondary[0], secondary[1:]
    # Scientific failure can naturally close; technical cleanup cannot qualify.
    natural = ended and not killed and not supervisor_killed and not secondary and not missing_identities
    if popen_attempted and first is not None and (first["type"] == "TimeoutError" or first["message"] != "actual unchanged supervisor natural job exit"):
        natural = False
    if held and ended:
        try:
            # Last actual identity check before relinquishing the reservation.
            matching, unknown, release_errors = owned_presence(identities, psutil)
            secondary.extend(release_errors)
            require(not matching and not unknown and not release_errors, "owned closure before reservation release")
            if lock_owned:
                require(strict_json(lock.read_bytes()) == token, "owned reservation lock identity")
                lock.unlink()
            require(not lock.exists(),'shared lock absent before reservation release')
            held = False
        except Exception as error:
            if first is None:
                first = error_record(error, "reservation-release")
            else:
                secondary.append(error_record(error, "reservation-release"))
            natural = False
            ended = not matching and not unknown and not missing_identities and (not popen_attempted or process is not None and process.poll() is not None)
    output = c["paths"]["output"] + "/envelope.json"
    output_spec = file_spec(output, root) if safe(output, root).exists() else None
    record = {"schema": "flylab.m3.e360.outer-provisional.v1", "contract": spec,
              'source_sha256':c['source_sha256'],
              "pre_spawn": receipt, "envelope": output_spec, "natural_closed": bool(natural),
              'registration':c['registration'],'amendment':c['amendment'],'resource_extension':c['resource_extension'],'revalidation_amendment':c['revalidation_amendment'],
              "authority_admitted": receipt is not None, "Popen_attempted": popen_attempted,
              "Popen_returned": process is not None, "Popen_never_attempted": not popen_attempted,
              "unspawned_original_array_reads": 0 if not popen_attempted else None,
              "unspawned_scientific_native_imports": 0 if not popen_attempted else None,
              "exit_code": process.returncode if process is not None else None,
              "observed_pid_create_times": {str(k): v for k, v in identities.items()},
              "all_owned_ended": bool(ended), "known_identity_cleanup": killed,
              "unchanged_supervisor_killed": supervisor_killed,
              "unresolved_matching_identities": {str(k): v for k, v in matching.items()},
              "unknown_owned_identities": {str(k): v for k, v in unknown.items()},
              "missing_owned_creation_times": missing_identities, "reservation_released": not held,
              "first_error": first, "secondary_errors": secondary,
              "elapsed_before_outer_publication_s": time.perf_counter() - entry,
              "actual_full_return_pending": True, "cumulative_before_s": c["cumulative_before_s"],
              "cumulative_after_s": None, "cumulative_limit_s": c["cumulative_limit_s"], **FALSE_MILESTONES}
    if execution_is_current(c.get('pass_id'),root) or c.get('pass_id') in ('e360-generous-p8','e360-generous-p9','e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p13','e360-generous-p14','e360-generous-p15','e360-generous-p16','e360-generous-p17','e360-generous-p18','e360-generous-p19','e360-generous-p20','e360-generous-p30'):record['entry_perf_counter']=entry
    record = publish_outer(c, record, timer, root=root)
    return 0 if record["first_error"] is None and record["secondary_errors"] == [] else 1

def failed_authority_view(raw,name,*,root=ROOT):
    """Bounded failure-record policy, not a scientific authority or retry."""
    reg=registration(root)
    amendment(root)
    resource_extension(root)
    prefix=BASE+'/parent-e346-e360-'
    require(type(name) is str and name.startswith(prefix),'failure record canonical E360 prefix')
    match=re.fullmatch(r'(e360-[a-z0-9-]{1,48})-(saved_prefix_diagnostic|fixture|suite)\.json',name[len(prefix):])
    require(match is not None,'failure record finite registered role/name')
    attempt_id,role=match.groups()
    require(role in reg['roles'],'failure record registered role')
    paths=names(attempt_id,role)
    require(name==paths['contract'],'failure record canonical contract name')
    limits=dict(effective_policy(root,pass_id=raw.get('pass_id','e360-finite-failure'))['effective_roles'][role])
    original=role=='saved_prefix_diagnostic'
    view={**raw,'paths':paths,'limits':limits,'mode':'original' if original else 'generated',
          'role':role,'attempt_id':attempt_id,'attempt':1,'original_calls':int(original),
          'source_sha256':raw.get('source_sha256') if type(raw.get('source_sha256')) is dict else {},
          'registration':file_spec(REGISTRATION,root),'amendment':file_spec(AMENDMENT,root),'resource_extension':file_spec(RESOURCE_EXTENSION,root),'revalidation_amendment':file_spec(authority_path_for_pass(raw.get('pass_id','e360-finite-failure')),root),
          'retained_reserve_bytes':min(1<<20,limits['output_cap_bytes']//4)}
    budget=current_budget_authority(root) if raw.get('pass_id') in ('e360-generous-p9','e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p13','e360-generous-p14','e360-generous-p15','e360-generous-p16','e360-generous-p17','e360-generous-p18','e360-generous-p19','e360-generous-p20',execution_current_pass(root)) else {'generated_shared_limit_s':600,'generated_local_limit_s':489}
    defaults={'cumulative_before_s':reg['original_cumulative_before_s'] if original else reg['generated_shared_baseline_s'],
              'cumulative_limit_s':10200 if original else budget['generated_shared_limit_s'],'stage_used_s':126.361 if original else 30.767,
              'stage_limit_s':300 if original else budget['generated_shared_limit_s']-reg['generated_shared_baseline_s'],
              'local_used_s':0,'local_limit_s':20 if original else budget['generated_local_limit_s'],'reserved_s':limits['outer_s']}
    for key,value in defaults.items():
        candidate=raw.get(key)
        view[key]=candidate if type(candidate) in (int,float) and math.isfinite(candidate) and candidate>=0 else value
    view['local_limit_s']=20 if original else budget['generated_local_limit_s']
    return view

def owned_closed(record):
    """Exact observable closure; qualification additionally requires success."""
    return (record.get('all_owned_ended') is True and record.get('reservation_released') is True
            and record.get('known_identity_cleanup')==[]
            and record.get('unresolved_matching_identities')=={}
            and record.get('unknown_owned_identities')=={}
            and record.get('missing_owned_creation_times')==[])


def seal_full_return(contract_name, actual_tool_return, *, root=ROOT):
    """ROOT calls after the actual exec return. Returns a record; writes nothing.

    ROOT stores result under fresh E360 .json and debits it exactly once. The
    external return schema is ROOT-owned; required fields are the two actual_*
    names documented above. Its duration must include this entire wrapper call.
    """
    raw = strict_json(bounded_json_name(contract_name,root,cap=524288))
    c = failed_authority_view(raw,contract_name,root=root)
    require(actual_tool_return["path"] == c["paths"]["actual_return"], "literal external actual tool-return path")
    actual = strict_json(read_spec(actual_tool_return, root))
    wall = numeric(actual["actual_outer_full_return_s"], "external full-return", minimum=1e-12)
    code = actual["actual_returned_exit_code"]
    require(type(code) is int, "actual exit code")
    require(type(wall) is float,'typed actual external float wall')
    outer_spec = file_spec(c["paths"]["outer"], root)
    outer = strict_json(read_spec(outer_spec, root))
    require(outer["contract"] == file_spec(contract_name, root), "closed contract actual wrapper")
    pending = outer["envelope"]
    value = strict_json(read_spec(pending, root)) if pending else None
    errors = []
    def append_errors(record):
        if record:
            rows = [record.get("first_error"), *record.get("secondary_errors", [])]
            for index, row in enumerate(rows):
                # Snapshot linkage may repeat inherited errors, while repeated
                # actual errors inside a snapshot must all remain visible.
                if row is not None and errors.count(row) < rows[:index + 1].count(row):
                    errors.append(row)
    # Preserve earliest science failure, subsequent scientific/close errors,
    # then outer supervision/publication and actual full-return errors.
    append_errors(value)
    folder = safe(c["paths"]["output"], root)
    for member in ("failure.json", "revocation.json"):
        if (folder / member).exists():
            append_errors(strict_json((folder / member).read_bytes()))
    append_errors(outer)
    if c.get('pass_id') in ('e360-generous-p8','e360-generous-p9','e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p13','e360-generous-p14','e360-generous-p15','e360-generous-p16','e360-generous-p17','e360-generous-p18','e360-generous-p19','e360-generous-p20',execution_current_pass(root)) and c['role']=='fixture' and value and value.get('run_valid') is True:
        try:
            typed_equal(value['pre_spawn'],outer['pre_spawn'],'P8 checkpoints actual outer receipt')
            receipt=strict_json(bounded_json_read(value['pre_spawn'],root,cap=262144))
            typed_equal(receipt['entry_perf_counter'],outer['entry_perf_counter'],'P8 checkpoint actual outer parent ENTRY')
            validate_execution_checkpoints(value,c,receipt,root=root)
            typed_equal(raw['limits'],effective_policy(root,pass_id=c['pass_id'])['effective_roles']['fixture'],'P8 full-return exact caps')
        except Exception as error:
            errors.append(error_record(error,'execution-checkpoint-full-return'))
    if actual.get('resource_extension')!=c['resource_extension']:
        errors.append(error_record(ValueError('actual return resource extension mismatch'),'actual-full-return'))
    if actual.get('revalidation_amendment')!=c['revalidation_amendment']:
        errors.append(error_record(ValueError('actual return revalidation amendment mismatch'),'actual-full-return'))
    if c.get('pass_id')==execution_current_pass(root):
        try:
            admitted=strict_json(read_spec(actual['request'],root))
            collector=p30_collector_proof(actual['collector_trace'],admitted,actual['request'],wall,code,root=root)
            typed_equal(actual['collector_proof_valid'],collector['valid'],'worker saved collector proof')
            require(collector['valid'],'collector failure blocks qualification after known FINAL')
        except Exception as error:errors.append(error_record(error,'actual-collector-proof'))
    revocation_spec = None
    final_outer = outer
    if safe(c["paths"]["outer_revocation"], root).exists():
        revocation_spec = file_spec(c["paths"]["outer_revocation"], root)
        final_outer = strict_json(read_spec(revocation_spec, root))
        require(final_outer["contract"] == outer["contract"] and final_outer["envelope"] == pending,
                "outer revocation exact provisional linkage")
        append_errors(final_outer)
    if code != 0:
        errors.append(error_record(RuntimeError("actual tool returned exit code " + str(code)), "actual-full-return"))
    if wall > c["limits"]["outer_s"]:
        errors.append(error_record(TimeoutError("actual full-return exceeds same-ENTRY outer cap"), "actual-full-return"))
    if c["stage_used_s"] + wall > c["stage_limit_s"] or c["cumulative_before_s"] + wall > c["cumulative_limit_s"]:
        errors.append(error_record(RuntimeError("actual full-return exceeds registered cumulative/stage budget"), "actual-full-return"))
    if c['local_used_s'] + wall > c['local_limit_s']:
        errors.append(error_record(RuntimeError('actual full-return exceeds E360 local budget'), 'actual-full-return'))
    try:
        check_sources(c['source_sha256'],root)
        typed_equal(outer['resource_extension'],c['resource_extension'],'sealed outer extension')
        typed_equal(outer['revalidation_amendment'],c['revalidation_amendment'],'sealed outer revalidation')
        if value is not None:
            require(value.get('contract')==outer['contract'] and value.get('source_sha256')==c['source_sha256'],
                    'actual full-return current publisher/source identity')
            typed_equal(value['amendment'],c['amendment'],'sealed current publisher amendment')
            typed_equal(value['resource_extension'],c['resource_extension'],'sealed current extension')
            typed_equal(value['revalidation_amendment'],c['revalidation_amendment'],'sealed revalidation')
            typed_equal(value['closure_report'],c['closure_report'],'sealed current publisher report')
    except Exception as error:
        errors.append(error_record(error,'actual-full-return-source'))
    valid = (code == 0 and wall <= c["limits"]["outer_s"] and outer["natural_closed"] is True
             and owned_closed(outer) and outer.get('authority_admitted') is True
             and outer.get('Popen_attempted') is True and outer.get('Popen_returned') is True
             and outer.get('Popen_never_attempted') is False
             and outer["first_error"] is None and outer["secondary_errors"] == [] and value is not None
             and value["run_valid"] is True and value["first_error"] is None and value["secondary_errors"] == []
             and value["revoked"] is False)
    valid &= not (folder / "failure.json").exists() and not (folder / "revocation.json").exists()
    valid &= revocation_spec is None and not errors
    valid &= c["stage_used_s"] + wall <= c["stage_limit_s"]
    valid &= c["cumulative_before_s"] + wall <= c["cumulative_limit_s"]
    valid &= c['local_used_s'] + wall <= c['local_limit_s']
    valid &= retained(c, root) + (256 << 10) <= c["limits"]["output_cap_bytes"]
    return {"schema": "flylab.m3.e360.actual-full-return.v1", "mode": c["mode"],
            "role": c["role"], "attempt_id": c["attempt_id"], "attempt": c["attempt"],
            "cost_id": c["attempt_id"] + ":" + c["role"], "charged_once": True,
            "contract": file_spec(contract_name, root), "outer": outer_spec,
            "outer_revocation": revocation_spec, "envelope": pending,
            "source_sha256": c["source_sha256"], "original_calls": c["original_calls"],
            'registration':c['registration'],'amendment':c['amendment'],'resource_extension':c['resource_extension'],'revalidation_amendment':c['revalidation_amendment'],'closure_report':c.get('closure_report'),
            "actual_tool_return": actual_tool_return, "actual_outer_full_return_s": wall,
            'call_id':actual.get('call_id'),'pass_id':actual.get('pass_id'),
            **({'collector_trace':actual['collector_trace'],'collector_proof_valid':actual['collector_proof_valid']}
               if c.get('pass_id')=='e360-generous-p17' else {}),
            "exit_code": code, "qualified": bool(valid), "all_owned_ended": final_outer["all_owned_ended"],
            "authority_admitted": outer.get('authority_admitted',False),
            "Popen_attempted": outer.get('Popen_attempted',False),"Popen_returned": outer.get('Popen_returned',False),
            "Popen_never_attempted": outer.get('Popen_never_attempted',False),
            "known_identity_cleanup": outer["known_identity_cleanup"], "first_error": errors[0] if errors else None,
            "unchanged_supervisor_killed": outer.get("unchanged_supervisor_killed", False),
            "secondary_errors": errors[1:], "natural_closed": final_outer["natural_closed"] is True,
            "unresolved_matching_identities": outer.get("unresolved_matching_identities", {}),
            "unknown_owned_identities": outer.get("unknown_owned_identities", {}),
            "missing_owned_creation_times": outer.get("missing_owned_creation_times", []),
            "reservation_released": outer.get("reservation_released", False), "actual_full_return_pending": False,
            "cumulative_before_s": c["cumulative_before_s"], "cumulative_after_s": c["cumulative_before_s"] + wall,
            "cumulative_limit_s": c["cumulative_limit_s"], "stage_after_s": c["stage_used_s"] + wall,
            'local_after_s': c['local_used_s'] + wall, 'local_limit_s': c['local_limit_s'],
            "unused_current_reservation_released_s": max(0, c["reserved_s"] - wall)
            if outer["all_owned_ended"] is True and outer.get("reservation_released") is True else 0,
            **FALSE_MILESTONES}

def closed_exit(spec, *, root=ROOT):
    with metadata_invocation(root) as admitted:
        row=_closed_exit_uncached(spec,root=root)
        if admitted:check_sources(row['source_sha256'],root)
        return row


def _closed_exit_uncached(spec, *, root=ROOT):
    row = charged_cost(spec, strict_json(read_spec(spec,root))['mode'],root)
    require(row['schema']=='flylab.m3.e360.actual-full-return.v1' and row['qualified'] is True
            and type(row['exit_code']) is int and row['exit_code']==0 and row['all_owned_ended'] is True
            and row['natural_closed'] is True and row['known_identity_cleanup']==[]
            and row['first_error'] is None and row['secondary_errors']==[]
            and row['actual_full_return_pending'] is False and row.get('outer_revocation') is None,
            'actual natural qualified E360 full return')
    require(owned_closed(row) and row.get('authority_admitted') is True
            and row.get('Popen_attempted') is True and row.get('Popen_returned') is True
            and row.get('Popen_never_attempted') is False,
            'complete released owned E360 full return')
    c = strict_json(bounded_json_read(row['contract'],root,cap=524288))
    amendment(root)
    for key,spec_ in (('registration',file_spec(REGISTRATION,root)),('amendment',file_spec(AMENDMENT,root)),('resource_extension',file_spec(RESOURCE_EXTENSION,root)),('revalidation_amendment',file_spec(authority_path_for_pass(c['pass_id']),root))):
        typed_equal(c[key],spec_,'closed current '+key)
        typed_equal(row[key],spec_,'full-return current '+key)
    typed_equal(row['closure_report'],c['closure_report'],'closed same direct source report')
    require(c['source_sha256'].get(c['closure_report']['path'])==c['closure_report']['sha256'],'closed direct report pin')
    require(row['source_sha256']==c['source_sha256'] and row['role']==c['role']
            and row['attempt_id']==c['attempt_id'] and row['attempt']==1
            and all(c['source_sha256'].get(k)==v for k,v in own_sources(root).items()), 'exact current E360 closed source')
    check_sources(c['source_sha256'],root)
    outer, value = strict_json(read_spec(row['outer'],root)), strict_json(read_spec(row['envelope'],root))
    typed_equal(c['limits'],effective_policy(root,pass_id=c['pass_id'])['effective_roles'][c['role']],'closed exact effective pass caps')
    if c.get('pass_id') in ('e360-generous-p8','e360-generous-p9','e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p13','e360-generous-p14','e360-generous-p15','e360-generous-p16','e360-generous-p17','e360-generous-p18','e360-generous-p19','e360-generous-p20',execution_current_pass(root)) and c['role']=='fixture':
        typed_equal(value['pre_spawn'],outer['pre_spawn'],'P8 checkpoint same actual outer receipt')
        receipt=strict_json(bounded_json_read(value['pre_spawn'],root,cap=262144))
        typed_equal(receipt['entry_perf_counter'],outer['entry_perf_counter'],'P8 same actual outer parent ENTRY')
        validate_execution_checkpoints(value,c,receipt,root=root)
        require(row['actual_outer_full_return_s']<=c['limits']['outer_s'],'closed P8 actual full-return deadline')
    typed_equal(outer['revalidation_amendment'],c['revalidation_amendment'],'closed outer direct amendment')
    require(outer['contract']==row['contract'] and outer['envelope']==row['envelope']
            and outer['natural_closed'] is True and outer['all_owned_ended'] is True
            and outer['first_error'] is None and outer['secondary_errors']==[]
            and value['contract']==row['contract'] and value['source_sha256']==row['source_sha256']
            and value['run_valid'] is True and value['revoked'] is False
            and value['first_error'] is None and value['secondary_errors']==[], 'current unrevoked actual publisher linkage')
    typed_equal(value['amendment'],c['amendment'],'publisher current amendment')
    typed_equal(value['resource_extension'],c['resource_extension'],'publisher current extension')
    typed_equal(value['revalidation_amendment'],c['revalidation_amendment'],'publisher revalidation')
    typed_equal(value['registration'],c['registration'],'publisher old registration')
    typed_equal(value['closure_report'],c['closure_report'],'publisher current report')
    require(owned_closed(outer) and outer.get('authority_admitted') is True
            and outer.get('Popen_attempted') is True and outer.get('Popen_returned') is True
            and outer.get('Popen_never_attempted') is False,
            'actual released owned publisher closure')
    folder = safe(c['paths']['output'],root)
    require(not any((folder/v).exists() for v in ('failure.json','revocation.json'))
            and not safe(c['paths']['outer_revocation'],root).exists(), 'no late E360 revocation')
    require(c['mode']!='original' or not safe(BASE+'/parent-e358-original-active.lock',root).exists(),
            'original shared lock absent at closed exit')
    debits,semantics,_,_=ledger_state(registration(root),root)
    matching=[v for v in debits if v['cost_id']==row['cost_id']]
    require(len(matching)==1 and matching[0]['descriptor']==spec
            and row['cost_id'] not in semantics['e360_pending_actuals']
            and row['cost_id'] not in semantics['e360_reservations']
            and not semantics['e360_blocking_holds'],'closed actual once worker debit/reservation release')
    seal_role='saved_prefix_seal' if row['mode']=='original' else row['role']+'_seal'
    seals=[v for v in semantics['e360_calls'].values() if v['call_role']==seal_role
           and v.get('attempt_id')==row['attempt_id']]
    require(len(seals)==1 and seals[0]['state']=='charged','closed actual sealing helper debit')
    seal_costs=[v for v in debits if v['cost_id']==seals[0]['cost_id']]
    require(len(seal_costs)==1,'once sealing helper')
    helper=charged_cost(seal_costs[0]['descriptor'],row['mode'],root)
    policy=effective_policy(root,pass_id=c['pass_id'])['helper_limits'][seal_role]
    require(helper['actual_returned_exit_code']==0 and helper['actual_outer_full_return_s']<=policy['outer_s']
            and helper.get('internal_completion_valid') is True,'closed helper actual timely successful completion')
    typed_equal(helper['expected_output'],spec,'seal helper identical final worker record')
    completion=strict_json(read_spec(helper['internal_completion'],root))
    typed_equal(completion['expected_output'],spec,'same ENTRY closed seal readback')
    typed_equal(completion['source_sha256'],own_sources(root),'closed helper current source')
    typed_equal(completion['resource_extension'],file_spec(RESOURCE_EXTENSION,root),'closed helper extension')
    typed_equal(completion['revalidation_amendment'],c['revalidation_amendment'],'current revalidation')
    return row

class CPUOnly:
    """Runtime call guard surrounds the actual reader/recompute/publisher path."""
    def __enter__(self):
        self.previous=sys.getprofile()
        # One guard covers nested reader/publisher scopes. Immutable code
        # objects can share the policy decision throughout this cold import.
        if getattr(self.previous,'_e360_cpu_guard',False):
            self.check=self.previous
            return self
        policy={}
        def check(frame,event,arg):
            if event=='call':
                code=frame.f_code
                forbidden=policy.get(code)
                if forbidden is None:
                    name,qualified=code.co_name,code.co_qualname
                    path=code.co_filename.replace('\\','/').lower()
                    scientific='/tools/' in path or '/flylab/' in path or '/flygym/' in path
                    # Module/class definitions are passive, including SymPy's
                    # class evaluate; invoking a scientific function is not.
                    forbidden=bool(code.co_flags & 1) and scientific and name in (
                        'resolve_original_api','advance','gpu_watch','evaluate','body','baseline','run_prefix',
                        'install_native_observer','cuda','from_checkpoint','synchronize','network_api',
                        'initial_body','body_data','body_audit','admitted_candidate','initial','lifecycle','main','child','supervise')
                    if path.endswith('/verify_c_banc_m3_source_closure_diagnostic.py'):
                        forbidden=False
                    forbidden |= ('OriginalFactory.' in qualified or 'GraphStore.load' in qualified
                                  or 'StateStore.load' in qualified or 'SSPEfficacyTrial.' in qualified
                                  or 'SSPRecoveryResourceTrial.' in qualified or 'BancWalkingSession.' in qualified)
                    forbidden |= ('/torch/cuda/' in path and name in ('_lazy_init','init','device_count','is_available',
                                   'set_device','current_device','memory_allocated','memory_reserved','get_device_properties',
                                   '_exchange_device','_maybe_exchange_device','__enter__','__exit__','synchronize'))
                    forbidden |= 'nvml' in path and name!='<module>'
                    policy[code]=bool(forbidden)
                # Passive Torch module definitions may be imported by the
                # unchanged CPU geometry dependency; no hardware API is used.
                if forbidden:
                    raise ValueError('CPU diagnostic forbids execution: '+code.co_qualname)
            elif event=='c_call':
                name=getattr(arg,'__name__','')
                module=getattr(arg,'__module__','') or ''
                native=module.lower()
                if name=='cuda' or name.startswith('_cuda_') or 'nvml' in native or 'mujo' in native:
                    raise ValueError('CPU diagnostic forbids native execution: '+name)
            if self.previous:
                self.previous(frame,event,arg)
        check._e360_cpu_guard=True
        self.check=check
        sys.setprofile(check)
        return self
    def __exit__(self,*args):
        sys.setprofile(self.previous)


def historical_metadata(reg,*,root=ROOT):
    records={k:strict_json(read_spec(v,root)) for k,v in reg['historical_E359'].items()}
    old=records['envelope.json']
    exit_=records['full_exit']
    partial=records['partial-progress.json']
    proof=records['transfer-chunks.json']
    trace=records['partial-trace.json']
    require(exit_['schema']=='flylab.m3.e359.actual-full-return.v1' and exit_['qualified'] is False
            and type(exit_['exit_code']) is int and exit_['exit_code']==1
            and exit_['natural_closed'] is True and exit_['all_owned_ended'] is True
            and exit_['reservation_released'] is True and exit_['actual_full_return_pending'] is False
            and exit_['known_identity_cleanup']==[] and not exit_['unresolved_matching_identities']
            and not exit_['unknown_owned_identities'] and not exit_['missing_owned_creation_times'],'naturally closed failed historical producer')
    require(exit_['contract']==reg['historical_E359']['contract']
            and exit_['envelope']==reg['historical_E359']['envelope.json']
            and records['contract']['registration']==reg['historical_E359']['registration']
            and records['contract']['adapter_qualification']==reg['historical_E359']['current_qualification'],
            'literal E359 failed-producer registration/Q/full-return linkage')
    if not reg.get('synthetic_scope',False):
        typed_equal(records['revocation.json'],old,'preserved historical late revocation linkage')
        typed_equal(old['actual_import_sha256'],{**reg['saved_project_import_oracle'],**reg['saved_installed_import_oracle']},
                    'complete saved123 actual source identities')
        folder=safe(reg['historical_E359']['envelope.json']['path'],root).parent
        require(not any((folder/name).exists() for name in ('baseline-manifest.json','baseline-report.json',
                   'selected-ledger.json','startup-inputs.npz')),'missing normal publication remains missing')
    typed_equal(old['first_error'],reg['expected_historical_first_error'],'historical exact first error')
    require(type(reg['expected_historical_secondary_errors']) is list,'historical secondary list schema')
    typed_equal(old['secondary_errors'],reg['expected_historical_secondary_errors'],'historical exact secondary errors')
    typed_equal(old['source_closure_mismatches'],reg['expected_historical_source_mismatches'],'historical exact missing15')
    require(old['run_valid'] is False and old['baseline_completed99'] is True and old['classification']=='technical_invalid',
            'known completion distinct from invalid publication')
    require(partial['baseline_completed99'] is False and partial['qualified'] is False
            and partial['first_error']==old['first_error'] and partial['secondary_errors']==old['secondary_errors'],
            'partial publication status unchanged')
    typed_equal(partial['progress'],{'controls':100,'rows':100,'native_samples':4950,'counters':old['counters']},
                'partial known100 boundaries and99 returned controls')
    lib.verify_work(old['counters'],old['construction'],99)
    require(old['counters'].get('advance_unreturned_unknown',False) is False
            and old['cpg_unchanged'] is True and old['cpg_hashes']==[proof['cpg_hash']]*100
            and old['transfer']==proof,'known work/100CPG/saved transfer linkage')
    return {'old':old,'partial':partial,'proof':proof,'trace':trace,'exit':exit_}


def raw_archive(policy,timer,*,root=ROOT,decode=True):
    """Partial archive: full literal headers and decompression bounds before decode."""
    import numpy as np

    from tools import verify_c_banc_m3_coordination_parameter_direction as core
    path=safe(policy['path'],root)
    spec=file_spec(policy['path'],root)
    require(spec['bytes']==policy['expected_size_from_directory']
            and spec['bytes']<=policy['max_compressed_bytes'],'literal current partial byte length')
    with zipfile.ZipFile(path) as archive:
        infos=archive.infolist()
        require(len(infos)==len(policy['members']) and len({v.filename for v in infos})==len(infos)
                and {v.filename for v in infos}=={k+'.npy' for k in policy['members']},'literal partial ZIP roster')
        total=sum(v.file_size for v in infos)
        require(total<=policy['max_uncompressed_bytes'],'partial decompression bound')
        for info in infos:
            timer.check('science')
            selected=policy['members'][info.filename[:-4]]
            require(not info.is_dir() and not info.flag_bits&1 and info.compress_type in (0,8)
                    and 0<info.file_size<=policy['max_uncompressed_bytes'],'bounded unencrypted ZIP member')
            with archive.open(info) as stream:
                header=core.strict_npy_header(stream)
            dtype=np.dtype(selected['dtype'])
            require(header['dtype']==dtype.str and header['shape']==selected['shape']
                    and header['fortran_order'] is False and not dtype.hasobject
                    and info.file_size==header['header_bytes']+math.prod(selected['shape'])*dtype.itemsize,
                    'exact partial NPY header before decode')
        values={v.filename[:-4]:np.load(io.BytesIO(archive.read(v)),allow_pickle=False) for v in infos} if decode else None
    require(file_spec(policy['path'],root)==spec,'current partial read stability, historical binding unknown')
    return values,spec,total


def read_prefix(reg,records,timer,*,root=ROOT):
    """Only two partial NPZs and E103 selected NPZ; never other132 inputs."""
    import numpy as np

    from tools import verify_c_banc_m3_coordination_parameter_direction as core
    from tools.qualify_c_banc_m3_body_packet import scan_archive
    target_policy=reg['raw_inputs']['startup']
    path=safe(target_policy['path'],root)
    require(path.stat().st_size==target_policy['bytes'],'fixed selected startup bytes')
    scanner=scan_archive(path,expected_sha256=target_policy['sha256'],descriptor_scope='observation',timely=lambda:timer.check('science'))
    startup_bytes=sum(v['uncompressed_bytes'] for v in scanner['descriptor']['members'].values())
    require(startup_bytes<=target_policy['max_uncompressed_bytes'],'all startup headers decompression bound')
    _,control_spec,control_bytes=raw_archive(reg['raw_inputs']['partial_control'],timer,root=root,decode=False)
    _,native_spec,native_bytes=raw_archive(reg['raw_inputs']['partial_native'],timer,root=root,decode=False)
    require(control_bytes+native_bytes+startup_bytes<=reg['raw_inputs']['aggregate_max_uncompressed_bytes'],'aggregate finite raw bound')
    control,decoded_control_spec,_=raw_archive(reg['raw_inputs']['partial_control'],timer,root=root)
    native,decoded_native_spec,_=raw_archive(reg['raw_inputs']['partial_native'],timer,root=root)
    require(control_spec==decoded_control_spec and native_spec==decoded_native_spec,'same all-header/decode source bytes')
    inventory=strict_json(read_spec(reg['body_inventory'],root))
    binding=inventory['selected_archives']['startup']
    require(binding['path']==target_policy['path'] and binding['bytes']==target_policy['bytes']
            and binding['sha256']==target_policy['sha256'],'fixed E103 selected inventory binding')
    require(set(binding['selected_schema'])=={'target_command','target_tendon'},'only selected startup targets')
    ledger=[]
    target=core.load_selected(root,binding,ledger,timely=lambda:timer.check('science'))
    require(all(np.isfinite(v).all() for v in control.values()),'finite partial command/tendon')
    descriptors={'partial_control':control_spec,'partial_native':native_spec,'startup':file_spec(target_policy['path'],root)}
    require(descriptors['startup']['sha256']==target_policy['sha256'],'startup current posthash')
    require(file_spec(control_spec['path'],root)==control_spec and file_spec(native_spec['path'],root)==native_spec,
            'all current partial read posthash stability')
    return control,native,target,inventory,{'current_read_descriptors':descriptors,'startup_selected_observation':ledger,
                                        'historical_raw_publication_binding_proven':False}


def validate_saved_prefix(control,native,target,records,inventory,*,b):
    import numpy as np
    old,proof,trace=records['old'],records['proof'],records['trace']
    require(set(proof['leaves'])=={'host.columns','host.rowptr','host.weights','packed.columns','packed.palette','packed.rowptr'},
            'literal six saved transfer leaves')
    b.verify_chunk_coverage(proof)
    for key in ('graph_hash','source_network_identity','source_adapter_identity'):
        typed_equal(proof[key],inventory[key],'fixed saved source '+key)
    require(proof['candidate_network_identity']==proof['source_network_identity']
            and proof['candidate_CSR_data_sha256']==proof['source_CSR_data_sha256']
            and proof['source_inverse_whole_exact'] is True and proof['source_boundary_command_and_tendon_exact'] is True
            and proof['neutral_source_CSR_equal'] is True and proof['pointer_version_guards_passed'] is True,
            'saved neutral producer proof consistency only')
    require(proof['trial_schema']=='flylab.m3-ssp-efficacy-trial.v1'
            and proof['patch_schema']=='flylab.m3-ssp-efficacy-patch.v1'
            and proof['source_trial_schema']=='flylab.m3-ssprk3-recovery-resource-trial.v1','saved trial provenance schema')
    for key in ('patch_identity','source_patch_identity','theta_sha256','selected_sha256','positions_sha256','groups_sha256','cpg_hash'):
        require(type(proof[key]) is str and re.fullmatch('[0-9a-f]{64}',proof[key]),'saved provenance '+key)
    leaves=proof['leaves']
    require(leaves['host.columns']['count']==leaves['host.weights']['count']==leaves['packed.columns']['count']
            and leaves['host.rowptr']['count']==leaves['packed.rowptr']['count']
            and leaves['packed.palette']['count']==proof['palette'] and type(proof['palette']) is int
            and 0<proof['palette']<=16384,'saved transfer dimensions/palette')
    for name,leaf in leaves.items():
        typed_equal(leaf['dtype'],'<f4' if name in ('host.weights','packed.palette') else '<i4','saved transfer dtype')
    require(b.array_sha(control['command'][0])==proof['boundary_command_sha256']
            and b.array_sha(control['tendon'][0])==proof['boundary_tendon_sha256'],'saved boundary0 byte digest')
    require(trace['landmarks']=={'scope':'actual_native','count':8,'order':list(b.NATIVE_LANDMARKS)}
            and len(trace['samples'])==100,'literal native8 landmarks/100 trace')
    for key,value in (('body_hash',proof['body_hash']),('model_hash',proof['source_network_identity']),
                      ('adapter_hash',proof['source_adapter_identity']),('graph_hash',proof['graph_hash'])):
        typed_equal(trace[key],value,'saved world/model/decoder/graph '+key)
    for tick,row in enumerate(trace['samples']):
        b.lean_row(row,point_count=8)
        require(type(row['control_tick']) is int and row['control_tick']==tick
                and abs(row['time_s']-.005*tick)<=2e-8 and not row['fault'] and row['upright']>=.9,
                'saved trace clocks/native health')
    require(b.validate_native(native)==4950 and np.all(native['upright']>=.9) and not native['nonfoot_contact'].any()
            and np.isfinite(native['slip_mm_s'][native['slip_valid']]).all(),'saved native4950 health/valid slip')
    startup=b.startup99(control['command'],control['tendon'],**target)
    require(type(old['startup']['passed']) is bool and startup['passed'] is old['startup']['passed']
            and type(old['startup']['controls']) is int and old['startup']['controls']==99,'fixed startup99 producer flags')
    for key in ('q_rms','q_max','tendon_rms','tendon_max'):
        require(type(old['startup'][key]) is float and math.isfinite(old['startup'][key])
                and abs(startup[key]-old['startup'][key])<=1e-15,'fixed startup producer equality '+key)
    return normalize(startup)


def diagnostic(reg,records,timer,*,root=ROOT):
    with CPUOnly():
        from tools import verify_c_banc_m3_common_descent_body as b
        control,native,target,inventory,observations=read_prefix(reg,records,timer,root=root)
        startup=validate_saved_prefix(control,native,target,records,inventory,b=b)
        return {'diagnostic_only':True,**SCOPES,**FALSE_MILESTONES,'H359a':'not_independently_qualified',
                'startup':startup,'startup_consistent':True,'saved_counter_transfer_scope':'producer_reported/saved_digest',
                'raw_row_alignment_scope':'saved-producer linkage assumption','observations':observations,
                'historical_input_status':{'qualified':False,'exit_code':1,'first_error':records['old']['first_error'],
                                         'secondary_errors':records['old']['secondary_errors'],
                                         'full_exit_first_error':records['exit']['first_error'],
                                         'full_exit_secondary_errors':records['exit']['secondary_errors'],
                                         'source_closure_mismatches':records['old']['source_closure_mismatches']}}


def actual_imports(c,result,*,root=ROOT):
    observed={}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename and str(filename).endswith('.py'):
            path=Path(filename).resolve()
            if path.is_relative_to(root.resolve()):
                name=path.relative_to(root.resolve()).as_posix()
                if name.startswith(('tools/','flylab/','.venv/Lib/site-packages/flygym')) or name==OWNED[0]:
                    observed[name]=sha(safe(name,root).read_bytes())
    result['actual_import_sha256']=observed
    result['source_closure_mismatches']=[{'path':k,'actual_sha256':v,'registered_sha256':c['source_sha256'].get(k)}
                                       for k,v in observed.items() if c['source_sha256'].get(k)!=v]
    require(not result['source_closure_mismatches'],'actual E360 imported source closure')



PRESERVED_C9_TEST_FUNCTIONS=(
    'test_active_admin_recovery60_history_event_and_six_pass_bound_relocations',
    'test_actual_call_admission_source_role_types_replay_and_live_suffix',
    'test_actual_E359_metadata_admission_preserves_separate_error_snapshots',
    'test_AST_cycle_relative_from_namespace_alias_constants_file_loader_and_unknown',
    'test_AST_lexical_shadow_builtin_loader_safe_and_absent_namespace_export',
    'test_AST_source_drift_and_path_escape_rejected',
    'test_bounded_PowerShell_source_reader_UTF8_SHA_types_and_binary_rejection',
    'test_closed_exit_rejects_unreleased_unknown_missing_and_unadmitted_mutations',
    'test_complete42_47_ledger_omission_duplicate_bool_and_changed_descriptor',
    'test_CURRENT_actual_AST_closure_oracles_chains_and_source_stability',
    'test_current_authority_scope68_eight_fields_eighteen_relocations_and_no_reset',
    'test_current_PS_source_recovery62_fixed_history_and_ten_relocations',
    'test_current_Ruff_recovery63_history_six_fields_fourteen_relocations',
    'test_direct_closure_amendment_pins_and_late_report_drift',
    'test_execution_budget_closed_fixture_production_checkpoint_and_cap_substitution',
    'test_execution_budget_P8_manual_source_freshness_once_debit_no_P9',
    'test_execution_budget_v2_exact73_ninth_reader_historical_caps_and_no_P9',
    'test_execution_checkpoints_reached_actual_parent_ENTRY_tamper_and_bounds',
    'test_fixed_scope_actual_child_failed_envelope_revocation_survives_drift_and_expiry',
    'test_fixed_scope_actual_pre_popen_fence_and_fresh_dynamic_admission',
    'test_fixed_scope_actual_pre_spawn_and_pre_array_fences_before_numeric',
    'test_fixed_scope_entry_literal_aggregate_and_monotone_descriptor_bounds',
    'test_fixed_scope_failure_outer_retains_primary_unknown_pending_and_closed_evidence',
    'test_fixed_scope_hit_deadline_failure_nonreuse_and_first_error_cleanup',
    'test_fixed_scope_hits_deepcopy_nested_root_descriptor_and_isolation',
    'test_fixed_scope_parent_ENTRY_science_and_internal_publication_caps',
    'test_fixed_scope_publication_pre_post_fences_bytes_path_and_late_error',
    'test_generous_literal_v2_policy_and_source_authority',
    'test_helper_same_ENTRY_callback_cold_read_and_durable_late_failure',
    'test_late_publication_revokes_and_keeps_first_error',
    'test_manual_p3_exact_once_suite_failure_no_pass4_or_fake_predecessor',
    'test_manual_p4_exact_failed_Ruff_correction_once_debit_no_p5',
    'test_manual_p5_actual_failed_scanner_C5_once_debit_and_no_p6',
    'test_manual_p6_exact_Ruff_exit1_internaltrue_C6_fresh_own9_and_no_p7',
    'test_manual_p7_exact_closed_fixture_exit1_C7_fresh_sources_no_p8',
    'test_metadata_first_error_source15_work_CPG_flags_tamper_reject',
    'test_modified_pass_semantic_worker_failure_literal_and_once_charged',
    'test_omitted_edge_oracle_does_not_seed_discovery',
    'test_owned_identity_reuse_unknown_timeout_and_same_ENTRY',
    'test_partial_archive_duplicate_extra_header_dtype_and_size_reject',
    'test_pending_actual_substitution_order_and_debit_backed_unknown',
    'test_pre_array_typed_full_authority_and_actual_lock_owner',
    'test_production_CPU_guard_forbidden_actual_call_names',
    'test_production_reader_recompute_publisher_PASS_FAIL_and_NaN_scope',
    'test_Q_helper_actual_timely_descriptor_completion_and_late_rejection',
    'test_raw_trace_target_startup_boundary_and_provenance_tamper_reject',
    'test_real_process_AST_cache_deepcopy_keys_source_and_layout_drift',
    'test_real_standalone_fixture_production_path_closed_once',
    'test_ROOT_current_scope_activation_and_complete_fixture_predecessor_production_probe',
    'test_ROOT_execution_budget_activation_and_historical_call_policy_production_fragments',
    'test_ROOT_finalize_semantic_keys_match_exact_ledger_source_schema',
    'test_ROOT_helper_entry_before_import_prepublication_and_pending_source_protocol',
    'test_ROOT_production_fallback_Q_and_modified_pass_pure_boundaries',
    'test_same_budget_literal_activation_and_exact_historical_fix_relocation',
    'test_supervise_P8_clock_construction_drift_cap_authority_and_expiry_close_before_Popen',
    'test_supervise_pre_spawn_failure_records_no_Popen_and_unknown_creation',
    'test_warm_AST_never_caches_live_budget_and_full_future_scan_bound',
    'test_whole_current_receipt_three_phases_and_strict_JSON',
)


PRESERVED_FIRST_TEST_FUNCTIONS=(
    'test_active_admin_recovery60_history_event_and_six_pass_bound_relocations',
    'test_actual_call_admission_source_role_types_replay_and_live_suffix',
    'test_actual_E359_metadata_admission_preserves_separate_error_snapshots',
    'test_AST_cycle_relative_from_namespace_alias_constants_file_loader_and_unknown',
    'test_AST_lexical_shadow_builtin_loader_safe_and_absent_namespace_export',
    'test_AST_source_drift_and_path_escape_rejected',
    'test_bounded_PowerShell_source_reader_UTF8_SHA_types_and_binary_rejection',
    'test_closed_exit_rejects_unreleased_unknown_missing_and_unadmitted_mutations',
    'test_complete42_47_ledger_omission_duplicate_bool_and_changed_descriptor',
    'test_CURRENT_actual_AST_closure_oracles_chains_and_source_stability',
    'test_current_authority_scope68_eight_fields_eighteen_relocations_and_no_reset',
    'test_current_PS_source_recovery62_fixed_history_and_ten_relocations',
    'test_current_Ruff_recovery63_history_six_fields_fourteen_relocations',
    'test_direct_closure_amendment_pins_and_late_report_drift',
    'test_execution_budget_closed_fixture_production_checkpoint_and_cap_substitution',
    'test_execution_budget_P8_manual_source_freshness_once_debit_no_P9',
    'test_execution_budget_v2_exact73_ninth_reader_historical_caps_and_no_P9',
    'test_execution_checkpoints_reached_actual_parent_ENTRY_tamper_and_bounds',
    'test_fixed_scope_actual_child_failed_envelope_revocation_survives_drift_and_expiry',
    'test_fixed_scope_actual_pre_popen_fence_and_fresh_dynamic_admission',
    'test_fixed_scope_actual_pre_spawn_and_pre_array_fences_before_numeric',
    'test_fixed_scope_entry_literal_aggregate_and_monotone_descriptor_bounds',
    'test_fixed_scope_failure_outer_retains_primary_unknown_pending_and_closed_evidence',
    'test_fixed_scope_hit_deadline_failure_nonreuse_and_first_error_cleanup',
    'test_fixed_scope_hits_deepcopy_nested_root_descriptor_and_isolation',
    'test_fixed_scope_parent_ENTRY_science_and_internal_publication_caps',
    'test_fixed_scope_publication_pre_post_fences_bytes_path_and_late_error',
    'test_generous_literal_v2_policy_and_source_authority',
    'test_helper_same_ENTRY_callback_cold_read_and_durable_late_failure',
    'test_historical_P8_checkpoint_evidence_and_tampering_stay_P8',
    'test_journal_actual_clock_writer_isolation_reached_events_and_utf8_bounds',
    'test_journal_authentic_poison_descriptor_drift_and_reserved_final',
    'test_journal_no_overwrite_retry_partial_completion_or_deadline_extension',
    'test_late_publication_revokes_and_keeps_first_error',
    'test_manual_p3_exact_once_suite_failure_no_pass4_or_fake_predecessor',
    'test_manual_p4_exact_failed_Ruff_correction_once_debit_no_p5',
    'test_manual_p5_actual_failed_scanner_C5_once_debit_and_no_p6',
    'test_manual_p6_exact_Ruff_exit1_internaltrue_C6_fresh_own9_and_no_p7',
    'test_manual_p7_exact_closed_fixture_exit1_C7_fresh_sources_no_p8',
    'test_metadata_closure_sources_local_completion_blocks_result_after_drift',
    'test_metadata_first_error_source15_work_CPG_flags_tamper_reject',
    'test_metadata_inherited_poison_restoration_primary_and_nested_ownership',
    'test_metadata_invocations_fresh_live_own_source_results_and_cleanup',
    'test_metadata_noop_untrusted_deadline_uses_old_uncached_path',
    'test_modified_pass_semantic_worker_failure_literal_and_once_charged',
    'test_omitted_edge_oracle_does_not_seed_discovery',
    'test_owned_identity_reuse_unknown_timeout_and_same_ENTRY',
    'test_partial_archive_duplicate_extra_header_dtype_and_size_reject',
    'test_pending_actual_substitution_order_and_debit_backed_unknown',
    'test_pre_array_typed_full_authority_and_actual_lock_owner',
    'test_production_CPU_guard_forbidden_actual_call_names',
    'test_production_reader_recompute_publisher_PASS_FAIL_and_NaN_scope',
    'test_Q_helper_actual_timely_descriptor_completion_and_late_rejection',
    'test_raw_trace_target_startup_boundary_and_provenance_tamper_reject',
    'test_real_process_AST_cache_deepcopy_keys_source_and_layout_drift',
    'test_real_standalone_fixture_production_path_closed_once',
    'test_ROOT_current_scope_activation_and_complete_fixture_predecessor_production_probe',
    'test_ROOT_execution_budget_activation_and_historical_call_policy_production_fragments',
    'test_ROOT_finalize_semantic_keys_match_exact_ledger_source_schema',
    'test_ROOT_helper_entry_before_import_prepublication_and_pending_source_protocol',
    'test_ROOT_production_fallback_Q_and_modified_pass_pure_boundaries',
    'test_same_budget_literal_activation_and_exact_historical_fix_relocation',
    'test_supervise_P8_clock_construction_drift_cap_authority_and_expiry_close_before_Popen',
    'test_supervise_pre_spawn_failure_records_no_Popen_and_unknown_creation',
    'test_warm_AST_never_caches_live_budget_and_full_future_scan_bound',
    'test_whole_current_receipt_three_phases_and_strict_JSON',
)


PRESERVED_R1_TEST_FUNCTIONS=(
    'test_active_admin_recovery60_history_event_and_six_pass_bound_relocations',
    'test_actual_call_admission_source_role_types_replay_and_live_suffix',
    'test_actual_E359_metadata_admission_preserves_separate_error_snapshots',
    'test_AST_cycle_relative_from_namespace_alias_constants_file_loader_and_unknown',
    'test_AST_lexical_shadow_builtin_loader_safe_and_absent_namespace_export',
    'test_AST_source_drift_and_path_escape_rejected',
    'test_bounded_PowerShell_source_reader_UTF8_SHA_types_and_binary_rejection',
    'test_closed_exit_rejects_unreleased_unknown_missing_and_unadmitted_mutations',
    'test_complete42_47_ledger_omission_duplicate_bool_and_changed_descriptor',
    'test_CURRENT_actual_AST_closure_oracles_chains_and_source_stability',
    'test_current_authority_scope68_eight_fields_eighteen_relocations_and_no_reset',
    'test_current_PS_source_recovery62_fixed_history_and_ten_relocations',
    'test_current_Ruff_recovery63_history_six_fields_fourteen_relocations',
    'test_direct_closure_amendment_pins_and_late_report_drift',
    'test_execution_budget_closed_fixture_production_checkpoint_and_cap_substitution',
    'test_execution_budget_P8_manual_source_freshness_once_debit_no_P9',
    'test_execution_budget_v2_exact73_ninth_reader_historical_caps_and_no_P9',
    'test_execution_checkpoints_reached_actual_parent_ENTRY_tamper_and_bounds',
    'test_fixed_scope_actual_child_failed_envelope_revocation_survives_drift_and_expiry',
    'test_fixed_scope_actual_pre_popen_fence_and_fresh_dynamic_admission',
    'test_fixed_scope_actual_pre_spawn_and_pre_array_fences_before_numeric',
    'test_fixed_scope_entry_literal_aggregate_and_monotone_descriptor_bounds',
    'test_fixed_scope_failure_outer_retains_primary_unknown_pending_and_closed_evidence',
    'test_fixed_scope_hit_deadline_failure_nonreuse_and_first_error_cleanup',
    'test_fixed_scope_hits_deepcopy_nested_root_descriptor_and_isolation',
    'test_fixed_scope_parent_ENTRY_science_and_internal_publication_caps',
    'test_fixed_scope_publication_pre_post_fences_bytes_path_and_late_error',
    'test_generous_literal_v2_policy_and_source_authority',
    'test_helper_same_ENTRY_callback_cold_read_and_durable_late_failure',
    'test_historical_P8_checkpoint_evidence_and_tampering_stay_P8',
    'test_journal_actual_clock_writer_isolation_reached_events_and_utf8_bounds',
    'test_journal_authentic_poison_descriptor_drift_and_reserved_final',
    'test_journal_no_overwrite_retry_partial_completion_or_deadline_extension',
    'test_journal_two_actual_fences_surround_closed_readback_writer_and_retention',
    'test_late_publication_revokes_and_keeps_first_error',
    'test_manual_p3_exact_once_suite_failure_no_pass4_or_fake_predecessor',
    'test_manual_p4_exact_failed_Ruff_correction_once_debit_no_p5',
    'test_manual_p5_actual_failed_scanner_C5_once_debit_and_no_p6',
    'test_manual_p6_exact_Ruff_exit1_internaltrue_C6_fresh_own9_and_no_p7',
    'test_manual_p7_exact_closed_fixture_exit1_C7_fresh_sources_no_p8',
    'test_metadata_closure_sources_local_completion_blocks_result_after_drift',
    'test_metadata_first_error_source15_work_CPG_flags_tamper_reject',
    'test_metadata_inherited_poison_restoration_primary_and_nested_ownership',
    'test_metadata_invocations_fresh_live_own_source_results_and_cleanup',
    'test_metadata_noop_untrusted_deadline_uses_old_uncached_path',
    'test_modified_pass_semantic_worker_failure_literal_and_once_charged',
    'test_omitted_edge_oracle_does_not_seed_discovery',
    'test_owned_identity_reuse_unknown_timeout_and_same_ENTRY',
    'test_partial_archive_duplicate_extra_header_dtype_and_size_reject',
    'test_pending_actual_substitution_order_and_debit_backed_unknown',
    'test_pre_array_typed_full_authority_and_actual_lock_owner',
    'test_production_CPU_guard_forbidden_actual_call_names',
    'test_production_reader_recompute_publisher_PASS_FAIL_and_NaN_scope',
    'test_production_runner_first_failed_report_later_runner_deadline_secondary_and_context',
    'test_production_runner_journal_failure_all_phases_CASE_unqualified_child_and_restore',
    'test_Q_helper_actual_timely_descriptor_completion_and_late_rejection',
    'test_raw_trace_target_startup_boundary_and_provenance_tamper_reject',
    'test_real_process_AST_cache_deepcopy_keys_source_and_layout_drift',
    'test_real_standalone_fixture_production_path_closed_once',
    'test_ROOT_current_scope_activation_and_complete_fixture_predecessor_production_probe',
    'test_ROOT_execution_budget_activation_and_historical_call_policy_production_fragments',
    'test_ROOT_finalize_semantic_keys_match_exact_ledger_source_schema',
    'test_ROOT_helper_entry_before_import_prepublication_and_pending_source_protocol',
    'test_ROOT_production_fallback_Q_and_modified_pass_pure_boundaries',
    'test_same_budget_literal_activation_and_exact_historical_fix_relocation',
    'test_suite_deadline_actual_clock_identity_substitution_and_finally_restoration',
    'test_supervise_P8_clock_construction_drift_cap_authority_and_expiry_close_before_Popen',
    'test_supervise_pre_spawn_failure_records_no_Popen_and_unknown_creation',
    'test_warm_AST_never_caches_live_budget_and_full_future_scan_bound',
    'test_whole_current_receipt_three_phases_and_strict_JSON',
)


def test_functions(root=ROOT):
    ADMIN_CHECK()
    tree = ast.parse(safe(OWNED[1], root).read_text("utf8"))
    functions = [v.name for v in tree.body if isinstance(v, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and v.name.startswith("test_")]
    require(functions and len(functions) == len(set(functions)), "complete finite test function roster")
    require(len(functions)<=84 and (6*len(functions)+4)*8192+16384<=4<<20,
            'full dynamic collection fits max84 journal and final reserve before dispatch')
    return sorted(functions)


def validate_case_receipt(receipt, *, root=ROOT):
    require(set(PRESERVED_R1_TEST_FUNCTIONS)<=set(receipt['expected_functions']),'all seventy R1 tests preserved')
    require(set(PRESERVED_FIRST_TEST_FUNCTIONS)<=set(receipt['expected_functions']),'all sixty-six FIRST tests preserved')
    require(set(PRESERVED_C9_TEST_FUNCTIONS)<=set(receipt['expected_functions']),'all fifty-eight C9 tests preserved')
    p8=receipt['revalidation_amendment']['path']==EXECUTION_BUDGET_AMENDMENT
    p9=receipt['revalidation_amendment']['path']==SUITE_RECOVERY_AMENDMENT
    p17=receipt['revalidation_amendment']['path']==P17_RECOVERY_AMENDMENT
    if execution_bootstrap(root) is not None:
        typed_equal(receipt['expected_functions'],test_functions(root),'same complete current roster')
        require(len(receipt['expected_functions'])==84,'whole current84')
    p16=receipt['revalidation_amendment']['path']==P16_RECOVERY_AMENDMENT
    p15=receipt['revalidation_amendment']['path']==P15_RECOVERY_AMENDMENT
    p14=receipt['revalidation_amendment']['path']==P14_RECOVERY_AMENDMENT
    p13=receipt['revalidation_amendment']['path']==P13_RECOVERY_AMENDMENT
    p12=receipt['revalidation_amendment']['path']==P12_RECOVERY_AMENDMENT
    p11=receipt['revalidation_amendment']['path']==P11_RECOVERY_AMENDMENT
    p10=receipt['revalidation_amendment']['path']==LINT_RECOVERY_AMENDMENT
    require(len(receipt['expected_functions'])>=(84 if p17 or p16 or p15 or p14 or p13 or p12 or p11 else 79 if p10 else 71 if p9 else 52 if p8 else 38),'whole current minimum cases')
    if p8 or p9 or p10 or p11 or p12 or p13 or p14 or p15 or p16 or p17:
        preserved=(p17_recovery_amendment(root) if p17 else p16_recovery_amendment(root) if p16 else p15_recovery_amendment(root) if p15 else p14_recovery_amendment(root) if p14 else p13_recovery_amendment(root) if p13 else p12_recovery_amendment(root) if p12 else p11_recovery_amendment(root) if p11 else lint_recovery_amendment(root) if p10 else suite_recovery_amendment(root) if p9 else execution_budget_amendment(root))['preserved_test_functions']
        require(set(preserved)<=set(receipt['expected_functions']),'all fifty-two old named tests preserved')
        if p11 or p12 or p13 or p14 or p15 or p16 or p17:typed_equal(receipt['expected_functions'],sorted(preserved),'C3 sorted exact84 receipt from source-order84')
    require(receipt["schema"] == "flylab.m3.e360.complete-pytest-receipt.v1"
            and receipt["source_sha256"] == own_sources(root)
            and receipt["expected_functions"] == test_functions(root)
            and receipt["collected_functions"] == receipt["expected_functions"]
            and type(receipt['exit_code']) is int and receipt["exit_code"] == 0 and receipt["collection_errors"] == []
            and type(receipt['original_calls']) is int and type(receipt['original_array_reads']) is int
            and receipt["original_calls"] == receipt["original_array_reads"] == 0
            and receipt["complete_current_cases"] is True, "all actual current test cases")
    typed_equal(receipt['source_before'],receipt['source_sha256'],'CASE source before/after')
    typed_equal(receipt['amendment'],file_spec(AMENDMENT,root),'receipt effective amendment')
    typed_equal(receipt['registration'],file_spec(REGISTRATION,root),'receipt unchanged registration')
    typed_equal(receipt['resource_extension'],file_spec(RESOURCE_EXTENSION,root),'receipt extension')
    typed_equal(receipt['revalidation_amendment'],prospective_authority_spec(root),'current revalidation')
    typed_equal(strict_json(read_spec(receipt['closure_report'],root)),scan_current(root=root),'CASE current real AST')
    require(set(receipt["phases"]) == set(receipt["expected_functions"]), "no hidden test filtering")
    for function, phases in receipt["phases"].items():
        ADMIN_CHECK()
        require(set(phases) == {"setup", "call", "teardown"}
                and all(phases[k] == "passed" for k in phases), "every function setup/call/teardown passed: " + function)
    return True


class SuiteJournal:
    """Independent bounded diagnostic observer; never a CASE receipt or Q."""
    def __init__(self,c,context,n):
        require(type(n) is int and 0<n<=84,'exact current journal roster count max84 before dispatch')
        self.max_records=6*n+4
        require(self.max_records*8192+16384<=4<<20,'precommitted journal aggregate plus final reserve bound')
        require(c['limits']['output_cap_bytes']==16<<20,'journal inside existing worker output cap')
        self.errors=[]
        self.failed=False
        self.sequence=0
        self.total=0
        self.last=0.0
        dumps=json.dumps
        self._encode=lambda value:(dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,
            separators=(',',':'))+'\n').encode('utf8')
        self._error=error_record
        # Only observation captures these primitives before pytest monkeypatches.
        # Execution-under-test keeps the module's normal dynamic checks.
        checked,resolve,digest,writer=require,safe,sha,lib.durable
        clock,entry=context.clock,context.entry
        limits=copy.deepcopy(context.limits)
        boundary=context.boundary
        invalidate=boundary.poison
        specs=copy.deepcopy(boundary.owned_items())
        output=c['paths']['output']
        retained_check=lib.retained
        owned=copy.deepcopy(c)
        root=context.root
        def elapsed(stage):
            now=clock()
            checked(math.isfinite(now) and now>=entry,'journal actual monotone ENTRY')
            if now>entry+limits[stage+'_s']:
                raise TimeoutError('E360 journal same-ENTRY '+stage+' deadline')
            checked(not boundary.failed,'journal authentic inherited poison')
            return now-entry
        def fence(stage):
            try:
                elapsed(stage)
                for canonical,spec in specs:
                    path=resolve(spec['path'],root)
                    checked(os.path.normcase(str(path))==canonical and path.stat().st_size==spec['bytes'],
                            'journal authentic descriptor path/size')
                    with path.open('rb') as stream:raw=stream.read(spec['bytes']+1)
                    checked(len(raw)==spec['bytes'] and digest(raw)==spec['sha256'],'journal authentic descriptor SHA')
                    raw.decode('utf8',errors='strict')
                    checked(resolve(spec['path'],root)==path and path.stat().st_size==spec['bytes'],
                            'journal authentic late descriptor path/size')
                    with path.open('rb') as stream:again=stream.read(spec['bytes']+1)
                    checked(again==raw,'journal authentic descriptor read stability')
                return elapsed(stage)
            except Exception:
                invalidate();raise
        self.entry=entry
        self._fence=fence
        self._elapsed=elapsed
        def publish(payload,stage,sequence):
            path=resolve(output+'/pytest-journal-'+format(sequence,'04d')+'.json',root)
            checked(retained_check(owned,root)+len(payload)+owned['retained_reserve_bytes']+16384
                    <=limits['output_cap_bytes'],'journal retained final/failure reserve')
            fence(stage)  # Actual prepublication authority; no reused fence result.
            writer(path,payload,cap=8192,check=lambda:elapsed(stage))
            checked(path.read_bytes()==payload,'journal independent closed readback')
            fence(stage)  # Fresh after fsync, close and final independent readback.
            checked(retained_check(owned,root)+owned['retained_reserve_bytes']+16384
                    <=limits['output_cap_bytes'],'journal late retained final/failure reserve')
        self._publish=publish

    def record(self,event,*,node='',phase='',outcome='',duration=None,longrepr='',closing=False):
        if self.failed:return
        try:
            stage='internal' if closing else 'science'
            elapsed=float(self._elapsed(stage))
            require(elapsed>=self.last,'journal finite monotone elapsed')
            require(self.sequence<self.max_records-(0 if closing else 1),'journal reserved final record')
            require(event in ('runner_start','collection','collection_error','phase_start','phase_report','runner_final'),
                    'finite journal events')
            require(closing==(event=='runner_final'),'only journal final uses existing internal cap')
            if event.startswith('phase_'):require(phase in ('setup','call','teardown'),'journal actual phase')
            text=str(longrepr).encode('utf8')
            if len(text)>4096:text=text[:4064].decode('utf8',errors='ignore').encode('utf8')+b'...[truncated]'
            row={'schema':'flylab.m3.e360.pytest-diagnostic-journal.v1','sequence':self.sequence,
                 'event':event,'nodeid':node,'phase':phase,'outcome':outcome,
                 'actual_parent_ENTRY':self.entry,'elapsed_s':elapsed,
                 'pytest_reported_duration_s':duration,'failed_longrepr':text.decode('utf8'),
                 'diagnostic_only':True,'CASE_receipt':False,'qualification':False}
            if duration is not None:
                require(type(duration) is float and math.isfinite(duration) and duration>=0,'pytest reported duration')
            payload=self._encode(row)
            require(len(payload)<=8192 and self.total+len(payload)<=4<<20,'bounded serialized journal')
            self._publish(payload,stage,self.sequence)
            self.sequence+=1
            self.total+=len(payload)
            self.last=elapsed
        except Exception as error:
            self.failed=True
            self.errors.append(self._error(error,'pytest-diagnostic-journal'))


@contextlib.contextmanager
def canonical_suite_module(c,context,root):
    """Transient script/canonical identity only; no new import, scope or ENTRY."""
    name='tools.verify_c_banc_m3_source_closure_diagnostic'
    member=name.split('.')[-1]
    module=sys.modules.get(__name__)
    require(module is not None and module.__dict__.get('_SUITE_DEADLINE') is context
            and module.__dict__.get('_AUTHORITY_FENCE_STATE') is context.boundary
            and _AUTHORITY_SCOPE is None,'actual authenticated runner module identity')
    if __name__!='__main__':
        require(sys.modules.get(name) is module,'already canonical runner identity')
        yield
        return
    require(type(context) is SuiteDeadline and context.root==Path(root).resolve()
            and Path(module.ROOT).resolve()==context.root,'script bridge authenticated root')
    path=safe(OWNED[0],root)
    expected=c['source_sha256'][OWNED[0]]
    require(Path(module.__file__).resolve()==path.resolve() and sha(path.read_bytes())==expected,
            'script bridge exact admitted own source')
    package=sys.modules.get('tools')
    require(package is not None and any(Path(p).resolve()==context.root/'tools' for p in package.__path__),
            'script bridge existing same-root tools package')
    missing=object()
    previous=sys.modules.get(name,missing)
    previous_attribute=package.__dict__.get(member,missing)
    if previous is not missing and previous is not module:
        require(type(previous) is type(module) and type(previous.__dict__.get('__file__')) is str
                and isinstance(previous.__dict__.get('ROOT'),(Path,str))
                and Path(previous.__file__).resolve()==path.resolve()
                and Path(previous.ROOT).resolve()==context.root
                and all(previous.__dict__.get(key) is None for key in
                    ('_SUITE_DEADLINE','_AUTHORITY_SCOPE','_AUTHORITY_FENCE_STATE')),
                'script bridge conflicting canonical module/root/active context')
    require(previous_attribute is missing or previous_attribute is module or previous_attribute is previous,
            'script bridge conflicting package attribute')
    sys.modules[name]=module
    package.__dict__[member]=module
    primary=None
    try:
        yield
    except BaseException as error:
        primary=error
        raise
    finally:
        try:
            require(sys.modules.get(name) is module and package.__dict__.get(member) is module,
                    'script bridge late module identity substitution')
            require(module.__dict__.get('_SUITE_DEADLINE') is context
                    and module.__dict__.get('_AUTHORITY_FENCE_STATE') is context.boundary,
                    'script bridge late inherited context identity')
            context.check('internal')
            require(safe(OWNED[0],root)==path and sha(path.read_bytes())==expected,
                    'script bridge late admitted source')
        except Exception as secondary:
            if primary is None:raise
            primary.add_note('module identity completion secondary: '+repr(secondary))
        finally:
            if previous is missing:sys.modules.pop(name,None)
            else:sys.modules[name]=previous
            if previous_attribute is missing:package.__dict__.pop(member,None)
            else:package.__dict__[member]=previous_attribute


def bounded_pytest_text(value,limit=4096):
    raw=str(value).encode('utf8')
    if len(raw)>limit:
        raw=raw[:limit-32].decode('utf8',errors='ignore').encode('utf8')+b'...[truncated]'
    return raw.decode('utf8')


def preserve_pytest_error(observed,later,stage,journal=None):
    """First actual failed report stays primary; later errors remain secondary."""
    report=observed['first_failed_report']
    if report is not None:
        primary=RuntimeError('actual first failed pytest report: '+report['nodeid']+' '+report['phase']
            +' failed: '+report['longrepr'])
        secondary=[error_record(later,stage),*getattr(later,'publication_errors',[])]
    else:
        primary=later
        secondary=list(getattr(later,'publication_errors',[]))
    if journal is not None:secondary.extend(journal.errors)
    primary.publication_errors=secondary
    return primary


def execute_current_pytest(case_name,*,c=None,spec=None,receipt_spec=None,timer=None,root=ROOT):
    if c is None:
        # Direct CLI retains the uncached runner and cannot invent a parent ENTRY.
        return _execute_current_pytest(case_name,root=root)
    require(c['role']=='suite' and c['mode']=='generated','actual generated suite child')
    with suite_deadline(c,spec,receipt_spec,timer,root) as context:
        return _execute_current_pytest(case_name,root=root,c=c,context=context)


def _execute_current_pytest(case_name,*,root=ROOT,c=None,context=None):
    raw = safe(case_name,root).read_bytes()
    case = strict_json(raw)
    require(set(case) == {"schema", "source_sha256", "expected_functions", "receipt", "pytest_ini", "fixture_exit","closure_report","registration","amendment","resource_extension","revalidation_amendment"}
            and case["schema"] == "flylab.m3.e360.current-pytest-case.v1"
            and case["source_sha256"] == own_sources(root) and case["expected_functions"] == test_functions(root),
            "controlled whole current pytest contract")
    require(set(PRESERVED_R1_TEST_FUNCTIONS)<=set(case['expected_functions']),'complete preserved R1 roster')
    require(set(PRESERVED_FIRST_TEST_FUNCTIONS)<=set(case['expected_functions']),'complete preserved FIRST roster')
    require(set(PRESERVED_C9_TEST_FUNCTIONS)<=set(case['expected_functions']),'complete preserved C9 roster')
    fixture=closed_exit(case['fixture_exit'],root=root)
    typed_equal(case['amendment'],file_spec(AMENDMENT,root),'CASE amendment')
    typed_equal(case['registration'],file_spec(REGISTRATION,root),'CASE registration')
    typed_equal(case['resource_extension'],file_spec(RESOURCE_EXTENSION,root),'CASE extension')
    typed_equal(case['revalidation_amendment'],prospective_authority_spec(root),'current revalidation')
    typed_equal(case['closure_report'],fixture['closure_report'],'CASE same fixture closure')
    require(fixture['role']=='fixture' and fixture['mode']=='generated','actual current new E360 fixture before suite')
    if c is not None and (execution_is_current(c.get('pass_id'),root) or c.get('pass_id') in ('e360-generous-p9','e360-generous-p10','e360-generous-p11','e360-generous-p12','e360-generous-p13','e360-generous-p14','e360-generous-p15','e360-generous-p16','e360-generous-p17','e360-generous-p18','e360-generous-p19','e360-generous-p20','e360-generous-p30')):
        typed_equal(fixture['pass_id'],c['pass_id'],'current P9 suite requires same P9 fixture')
        typed_equal(fixture['source_sha256'],c['source_sha256'],'current suite identical full fixture closure')
        typed_equal(fixture['closure_report'],c['closure_report'],'current suite identical direct report')
        for key in ('registration','amendment','resource_extension','revalidation_amendment'):
            typed_equal(fixture[key],c[key],'current suite same direct authority '+key)
        own=own_sources(root)
        typed_equal({key:fixture['source_sha256'].get(key) for key in own},own,'current suite own9 projection')
        check_sources(fixture['source_sha256'],root)
    os.environ['E360_CURRENT_FIXTURE_EXIT']=json.dumps(case['fixture_exit'],allow_nan=False)
    os.environ['E360_CURRENT_CLOSURE']=json.dumps(case['closure_report'],allow_nan=False)
    source_before=own_sources(root)
    ini = safe(case["pytest_ini"],root)
    require(ini.read_bytes().strip() == b"[pytest]", "empty owned pytest config")
    receipt_path = safe(case["receipt"],root)
    require(receipt_path.parent == safe(BASE,root) and receipt_path.name.startswith("parent-e360-")
            and not receipt_path.exists(), "fresh E360 case receipt")
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        os.environ.pop(key, None)
    journal=SuiteJournal(c,context,len(case['expected_functions'])) if context is not None else None
    if journal is not None:journal.record('runner_start')
    import pytest
    observed = {"collected": [], "phases": {}, "collection_errors": [],'first_failed_report':None}
    class ReceiptPlugin:
        def phase_start(self,item,phase):
            if context is not None:context.check('science')
            if journal is not None:journal.record('phase_start',node=item.nodeid,phase=phase)
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_setup(self,item):
            self.phase_start(item,'setup')
            yield
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self,item):
            self.phase_start(item,'call')
            yield
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_teardown(self,item,nextitem):
            self.phase_start(item,'teardown')
            yield
        def pytest_collection_finish(self, session):
            for item in session.items:
                require(item.path.resolve() == safe(OWNED[1],root).resolve(), "only current owned test file")
                require("[" not in item.name, "explicit unparameterized finite test functions")
                observed["collected"].append(item.name)
            require(sorted(observed["collected"]) == case["expected_functions"], "complete collection, no filter")
            if journal is not None:journal.record('collection',outcome='complete_exact_current_collection')
        def pytest_collectreport(self, report):
            if report.failed:
                if not observed['collection_errors'] and journal is not None:
                    journal.record('collection_error',node=report.nodeid,outcome='failed',longrepr=report.longrepr)
                observed["collection_errors"].append(str(report.longrepr)[:2048])
        def pytest_runtest_logreport(self, report):
            if report.failed and observed['first_failed_report'] is None:
                observed['first_failed_report']={'nodeid':bounded_pytest_text(report.nodeid,1024),
                    'phase':report.when,'longrepr':bounded_pytest_text(report.longrepr)}
            function = report.nodeid.split("::")[-1]
            phases = observed["phases"].setdefault(function, {})
            require(report.when not in phases, "one actual test phase")
            phases[report.when] = report.outcome
            if context is not None:context.check('science')
            if journal is not None:
                journal.record('phase_report',node=report.nodeid,phase=report.when,outcome=report.outcome,
                    duration=float(report.duration),longrepr=report.longrepr if report.failed else '')
    primary=None
    try:
        binding=canonical_suite_module(c,context,root) if context is not None else contextlib.nullcontext()
        with binding:
            code = int(pytest.main(["-q", "-p", "no:cacheprovider", "-c", str(ini), str(safe(OWNED[1],root))], plugins=[ReceiptPlugin()]))
    except BaseException as error:
        primary=preserve_pytest_error(observed,error,'pytest-runner-or-deadline')
        if primary is error:raise
        raise primary from error
    finally:
        if journal is not None:
            journal.record('runner_final',outcome='exception' if primary is not None else 'returned',closing=True)
            if primary is not None:
                primary.publication_errors=[*getattr(primary,'publication_errors',[]),*journal.errors]
                for error in journal.errors:primary.add_note('journal secondary: '+repr(error))
    if journal is not None and journal.failed:code=code or 2
    try:
        source_after=own_sources(root)
    except Exception as error:
        primary=preserve_pytest_error(observed,error,'pytest-late-source',journal)
        if primary is error:raise
        raise primary from error
    value = {"schema": "flylab.m3.e360.complete-pytest-receipt.v1", "source_sha256": source_after,
             'source_before':source_before,'closure_report':case['closure_report'],
             'amendment':case['amendment'],'registration':case['registration'],'resource_extension':case['resource_extension'],'revalidation_amendment':case['revalidation_amendment'],
             "case_contract": descriptor(case_name, raw), "expected_functions": case["expected_functions"],
             "collected_functions": sorted(observed["collected"]), "phases": observed["phases"],
             "collection_errors": observed["collection_errors"], "exit_code": code,
             "original_calls": 0, "original_array_reads": 0, "complete_current_cases": code == 0}
    try:
        validate_case_receipt(value,root=root)
    except Exception:
        value["complete_current_cases"] = False
        code = code or 2
        value["exit_code"] = code
    try:
        durable(receipt_path, json_bytes(value), cap=256 << 10)
    except Exception as error:
        primary=preserve_pytest_error(observed,error,'pytest-case-publication',journal)
        if primary is error:raise
        raise primary from error
    if journal is not None and journal.failed:
        error=RuntimeError('pytest journal publication failed; completion unqualified')
        raise preserve_pytest_error(observed,error,'pytest-journal-completion',journal)
    if context is not None and code != 0 and observed['first_failed_report'] is not None:
        error=RuntimeError('pytest returned nonzero exit code '+str(code))
        raise preserve_pytest_error(observed,error,'pytest-returned-failure',journal)
    return code

def validate_qualification_value(q,sources=None,*,root=ROOT):
    ADMIN_CHECK()
    require(set(q)=={'schema','source_sha256','original_calls','original_array_reads','closed','suite_exit','fixture_exit',
                    'case_receipt','closure_report','registration','amendment','resource_extension','revalidation_amendment','qualification_call_id'}
            and q['schema']=='flylab.m3.e360.adapter-qualification.v1' and q['closed'] is True,'new acyclic E360 Q schema')
    typed_equal(q['original_calls'],0,'Q original calls0')
    typed_equal(q['original_array_reads'],0,'Q original reads0')
    typed_equal(q['source_sha256'],own_sources(root),'Q current actual own sources')
    if sources is not None:
        require(all(sources.get(k)==v for k,v in own_sources(root).items()),'Q caller current own-source pins')
    typed_equal(q['registration'],file_spec(REGISTRATION,root),'Q fixed reflected registration')
    typed_equal(q['amendment'],file_spec(AMENDMENT,root),'Q effective amendment')
    typed_equal(q['resource_extension'],file_spec(RESOURCE_EXTENSION,root),'Q extension')
    typed_equal(q['revalidation_amendment'],prospective_authority_spec(root),'current revalidation')
    require(type(q['qualification_call_id']) is str and re.fullmatch('e360-[a-z0-9-]{1,94}',q['qualification_call_id']),'Q literal qualification call ID')
    report=strict_json(read_spec(q['closure_report'],root))
    typed_equal(report,scan_current(root=root),'Q current actual closure')
    full_sources=closure_sources(registration(root),report,root,closure_report=q['closure_report'])
    same_pass=None
    for role in ('fixture','suite'):
        ADMIN_CHECK()
        closed=closed_exit(q[role+'_exit'],root=root)
        require(closed['role']==role and closed['mode']=='generated' and closed['original_calls']==0,'new original0 closed '+role)
        if execution_bootstrap(root) is not None:typed_equal(closed['pass_id'],execution_current_pass(root),'current data Q same admitted pass')
        if q['revalidation_amendment']['path'] in (SUITE_RECOVERY_AMENDMENT,LINT_RECOVERY_AMENDMENT,P11_RECOVERY_AMENDMENT,P12_RECOVERY_AMENDMENT,P13_RECOVERY_AMENDMENT,P14_RECOVERY_AMENDMENT,P15_RECOVERY_AMENDMENT,P16_RECOVERY_AMENDMENT,P17_RECOVERY_AMENDMENT):
            typed_equal(closed['pass_id'],'e360-generous-p17' if q['revalidation_amendment']['path']==P17_RECOVERY_AMENDMENT else 'e360-generous-p16' if q['revalidation_amendment']['path']==P16_RECOVERY_AMENDMENT else 'e360-generous-p15' if q['revalidation_amendment']['path']==P15_RECOVERY_AMENDMENT else 'e360-generous-p14' if q['revalidation_amendment']['path']==P14_RECOVERY_AMENDMENT else 'e360-generous-p13' if q['revalidation_amendment']['path']==P13_RECOVERY_AMENDMENT else 'e360-generous-p12' if q['revalidation_amendment']['path']==P12_RECOVERY_AMENDMENT else 'e360-generous-p11' if q['revalidation_amendment']['path']==P11_RECOVERY_AMENDMENT else 'e360-generous-p10' if q['revalidation_amendment']['path']==LINT_RECOVERY_AMENDMENT else 'e360-generous-p9','current P9 Q requires same P9 closed role')

        contract=strict_json(bounded_json_read(closed['contract'],root,cap=524288))
        typed_equal(closed['source_sha256'],contract['source_sha256'],'Q role complete pinned contract closure')
        typed_equal(closed['source_sha256'],full_sources,'Q role reconstructed actual full closure')
        own=own_sources(root)
        typed_equal({key:closed['source_sha256'].get(key) for key in own},own,'Q role own9 projection')
        check_sources(closed['source_sha256'],root)
        typed_equal(closed['pass_id'],contract['pass_id'],'Q closed pinned role pass')
        if same_pass is None:same_pass=closed['pass_id']
        typed_equal(closed['pass_id'],same_pass,'Q identical fixture/suite pass')
        for key in ('registration','amendment','resource_extension','revalidation_amendment','closure_report'):
            typed_equal(closed[key],contract[key],'Q role direct authority '+key)
        typed_equal(contract['closure_report'],q['closure_report'],'Q identical source closure')
        typed_equal(contract['amendment'],q['amendment'],'Q same amendment')
        typed_equal(contract['resource_extension'],q['resource_extension'],'Q same extension')
        typed_equal(contract['revalidation_amendment'],q['revalidation_amendment'],'Q same revalidation')
        typed_equal(contract['registration'],q['registration'],'Q same old authority')
        if role=='suite':
            value=strict_json(read_spec(closed['envelope'],root))
            typed_equal(value['case_receipt'],q['case_receipt'],'actual closed suite receipt')
    receipt=strict_json(read_spec(q['case_receipt'],root))
    validate_case_receipt(receipt,root=root)
    case=strict_json(read_spec(receipt['case_contract'],root))
    require(case['fixture_exit']==q['fixture_exit'] and case['closure_report']==q['closure_report']
            and case['source_sha256']==q['source_sha256'] and case['amendment']==q['amendment']
            and case['registration']==q['registration'] and case['resource_extension']==q['resource_extension'] and case['revalidation_amendment']==q['revalidation_amendment'],'complete CASE/fixture/closure/currentQ linkage')
    return q


def qualification_admin_proof(q,spec,semantics,debits,*,root=ROOT):
    """Publication alone cannot qualify a Q. Actual once-debited completion must match it."""
    call=semantics['e360_calls'].get(q['qualification_call_id'])
    require(call is not None and call['state']=='charged' and call['call_role']=='qualification'
            and call['mode']=='generated','Q once charged qualification helper')
    typed_equal(call['revalidation_amendment'],q['revalidation_amendment'],'Q actual call amended authority')
    matching=[d for d in debits if d['cost_id']==call['cost_id']]
    require(len(matching)==1,'exactly one Q helper debit')
    cost=charged_cost(matching[0]['descriptor'],'generated',root)
    policy=effective_policy(root,pass_id=call['pass_id'])['helper_limits']['qualification']
    require(cost['actual_returned_exit_code']==0 and cost['actual_outer_full_return_s']<=policy['outer_s'],
            'Q helper actual timely exit0')
    require(cost.get('internal_completion_valid') is True,'Q helper actual valid internal completion')
    typed_equal(cost['request'],call['request'],'Q charged immutable request')
    typed_equal(cost['call_id'],q['qualification_call_id'],'Q charged call ID')
    typed_equal(cost['call_role'],'qualification','Q helper role')
    typed_equal(cost['pass_id'],call['pass_id'],'Q helper pass')
    typed_equal(cost['resource_extension'],q['resource_extension'],'Q charged extension')
    typed_equal(cost['revalidation_amendment'],q['revalidation_amendment'],'Q charged revalidation')
    typed_equal(cost['source_sha256'],q['source_sha256'],'Q charged current source')
    typed_equal(cost['expected_output'],spec,'Q exact descriptor after full return')
    completion=strict_json(read_spec(cost['internal_completion'],root))
    require(set(completion)=={'schema','completed','call_id','pass_id','call_role','request','resource_extension','revalidation_amendment',
                             'source_sha256','expected_output','entry_perf_counter','internal_full_return_s'}
            and completion['schema']=='flylab.m3.e360.helper-internal-completion.v1','literal helper completion schema')
    require(type(completion['entry_perf_counter']) is float,'same ENTRY typed helper clock')
    numeric(completion['entry_perf_counter'],'helper ENTRY',minimum=1e-12)
    internal=numeric(completion['internal_full_return_s'],'helper internal elapsed',minimum=1e-12)
    require(completion['completed'] is True and internal<=policy['internal_s']
            and type(internal) is float,'same ENTRY helper internal completion')
    for key,value in (('call_id',q['qualification_call_id']),('request',call['request']),
                      ('pass_id',call['pass_id']),('call_role','qualification'),
                      ('resource_extension',q['resource_extension']),('revalidation_amendment',q['revalidation_amendment']),('source_sha256',q['source_sha256']),('expected_output',spec)):
        typed_equal(completion[key],value,'Q internal linkage '+key)
    return cost


def validate_qualification(spec,sources=None,*,root=ROOT):
    q=validate_qualification_value(strict_json(read_spec(spec,root)),sources,root=root)
    debits,semantics,_,_=ledger_state(registration(root),root)
    require(not semantics['e360_pending_actuals'] and not semantics['e360_blocking_holds'],
            'no pending or unknown Q admission')
    qualification_admin_proof(q,spec,semantics,debits,root=root)
    return q


def synthetic_inputs(c,timer,*,passed=True,root=ROOT):
    """Wholly generated failed producer; exercises the same dedicated reader."""
    import numpy as np

    from tools import verify_c_banc_m3_common_descent_body as b
    def save(name,value,raw=False):
        return put(c,name.replace('_','-'),value,timer,raw=raw,root=root)
    control={'command':np.zeros((100,42),np.float64),'tendon':np.zeros((100,6),np.float64)}
    if not passed:
        control['command'][99,0]=.041
    native={'time_s':np.arange(1,4951,dtype=np.float64)*.0001,'normal_load_bw':np.zeros((4950,6),np.float64),
            'slip_mm_s':np.full((4950,6),np.nan,np.float64),'slip_valid':np.zeros((4950,6),bool),
            'nonfoot_contact':np.zeros(4950,bool),'upright':np.ones(4950,np.float64)}
    target={'target_command':np.zeros((99,42),np.float64),'target_tendon':np.zeros((99,6),np.float64)}
    raw_specs={key:save('synthetic-'+key+'.npz',lib.npz_bytes(value,cap=4<<20),raw=True)
               for key,value in (('partial_control',control),('partial_native',native),
                                 ('startup',{**target,'legitimate_header_only_extra':np.arange(3,dtype=np.float64)}))}
    raw_inputs={key:{'path':spec['path'],'expected_size_from_directory':spec['bytes'],
                    'members':{k:{'shape':list(v.shape),'dtype':v.dtype.str} for k,v in value.items()},
                    'max_compressed_bytes':1<<20,'max_uncompressed_bytes':2<<20}
                for (key,value),spec in zip((('partial_control',control),('partial_native',native)),
                                           (raw_specs['partial_control'],raw_specs['partial_native']))}
    raw_inputs['startup']={**raw_specs['startup'],'max_uncompressed_bytes':16<<20}
    raw_inputs['aggregate_max_uncompressed_bytes']=19<<20
    inv={'graph_hash':'graph','source_network_identity':'source','source_adapter_identity':'adapter',
         'selected_archives':{'startup':{**raw_specs['startup'],'kind':'npz','selected_schema':
                              {k:{'shape':list(v.shape),'dtype':v.dtype.str} for k,v in target.items()}}}}
    inventory=save('synthetic-inventory.json',inv)
    digest='a'*64
    leaves={}
    for name in ('host.columns','host.rowptr','host.weights','packed.columns','packed.palette','packed.rowptr'):
        dtype='<f4' if name in ('host.weights','packed.palette') else '<i4'
        leaves[name]={'count':3,'dtype':dtype,'chunks':[{'start':0,'end':3,'element_count':3,'dtype':dtype,'bytes':12,
                        'source_sha256':digest,'expected_sha256':digest,'actual_sha256':digest}]}
    proof={'schema':'flylab.m3.e344.transfer-chunks.v1','leaves':leaves,'source_network_identity':'source',
           'candidate_network_identity':'source','source_adapter_identity':'adapter','graph_hash':'graph','body_hash':'synthetic-world',
           'cpg_hash':digest,'candidate_CSR_data_sha256':digest,'source_CSR_data_sha256':digest,'palette':3,
           'source_inverse_whole_exact':True,'source_boundary_command_and_tendon_exact':True,
           'neutral_source_CSR_equal':True,'pointer_version_guards_passed':True,
           'trial_schema':'flylab.m3-ssp-efficacy-trial.v1','patch_schema':'flylab.m3-ssp-efficacy-patch.v1',
           'source_trial_schema':'flylab.m3-ssprk3-recovery-resource-trial.v1',
           **{k:digest for k in ('patch_identity','source_patch_identity','theta_sha256','selected_sha256','positions_sha256','groups_sha256')},
           'boundary_command_sha256':b.array_sha(control['command'][0]),'boundary_tendon_sha256':b.array_sha(control['tendon'][0])}
    counters={'controls_requested':99,'controls_host_returned':99,'controls_completed':99,'completed_rows':100,
              'neural_requested':1980,'neural_host_returned':1980,'neural_device_confirmed':1980,'neural_unreturned_unknown':False,
              'physics_requested':4950,'physics_host_returned':4950,'physics_device_confirmed':4950,'physics_unreturned_unknown':False,
              'observer_samples':4950}
    construction={'body_constructors_requested':1,'body_constructors_returned':1,'settling_physics_requested':2000,
                  'settling_physics_host_returned':2000,'neural_setup_requested':320,'neural_setup_host_returned':320,
                  'neural_capture_requested':160,'neural_capture_recorded':160,'unknown_work':False,'device_confirmed':True}
    first={'message':'actual neutral body imported source closure','notes':[],'stage':'neutral-actual-imports','type':'ValueError'}
    second={'message':'only clean completed99 publication','notes':[],'stage':'baseline-publication','type':'ValueError'}
    mismatch=[{'path':'synthetic-missing-'+str(i)+'.py','actual_sha256':digest,'registered_sha256':None} for i in range(15)]
    old={'run_valid':False,'baseline_completed99':True,'classification':'technical_invalid','first_error':first,
         'secondary_errors':[second],'source_closure_mismatches':mismatch,'construction':construction,'counters':counters,
         'transfer':proof,'cpg_hashes':[digest]*100,'cpg_unchanged':True,
         'startup':normalize(b.startup99(control['command'],control['tendon'],**target))}
    rows=[{'control_tick':i,'time_s':i*.005,'fault':None,'upright':1.,
           'body':{'position':[0.,0.,0.],'basis':np.eye(3).tolist(),'yaw':0.,
                   'legs':{leg:np.zeros((8,3),np.float64).tolist() for leg in b.LEGS}},
           'feet':{'load_bearing':[False]*6}} for i in range(100)]
    trace={'landmarks':{'scope':'actual_native','count':8,'order':list(b.NATIVE_LANDMARKS)},'samples':rows,
           'body_hash':'synthetic-world','graph_hash':'graph','model_hash':'source','adapter_hash':'adapter','parameters':{}}
    # The synthetic native label exercises the literal reader; explicit outer
    # synthetic scope prevents it becoming physical execution evidence.
    metadata={key:save('synthetic-'+key.replace('.json','')+'.json',value)
              for key,value in (('registration',{'synthetic':True}),('current_qualification',{'synthetic':True}),
                                ('envelope.json',old),('transfer-chunks.json',proof),('partial-trace.json',trace),
                                ('partial-progress.json',{'baseline_completed99':False,'qualified':False,'first_error':first,
                                   'secondary_errors':[second],'progress':{'controls':100,'rows':100,'native_samples':4950,'counters':counters}}))}
    metadata['contract']=save('synthetic-contract.json',{'registration':metadata['registration'],
                                                       'adapter_qualification':metadata['current_qualification']})
    exit_={'schema':'flylab.m3.e359.actual-full-return.v1','qualified':False,'exit_code':1,'natural_closed':True,
           'all_owned_ended':True,'reservation_released':True,'actual_full_return_pending':False,'known_identity_cleanup':[],
           'unresolved_matching_identities':{},'unknown_owned_identities':{},'missing_owned_creation_times':[],
           'contract':metadata['contract'],'envelope':metadata['envelope.json'],
           'first_error':first,'secondary_errors':[second]}
    metadata['full_exit']=save('synthetic-full-exit.json',exit_)
    reg={'historical_E359':metadata,'body_inventory':inventory,'raw_inputs':raw_inputs,
         'expected_historical_first_error':first,'expected_historical_secondary_errors':[second],
         'expected_historical_source_mismatches':mismatch,'synthetic_scope':True}
    return reg


def publish_diagnostic(c,result,value,timer,*,root=ROOT):
    value={**value,'resource_extension':c['resource_extension'],'revalidation_amendment':c['revalidation_amendment']}
    with CPUOnly():
        timer.check()
        result['diagnostic']=value
        result.update({k:value[k] for k in ('diagnostic_only','startup','startup_consistent','historical_input_status',
                                           'H359a',*SCOPES)})
        spec=put(c,'diagnostic.json',value,timer,root=root)
        result['diagnostic_manifest']=put(c,'diagnostic-manifest.json',{'schema':'flylab.m3.e360.diagnostic-manifest.v1',
            'files':{'diagnostic.json':spec},'resource_extension':c['resource_extension'],'revalidation_amendment':c['revalidation_amendment'],
            'all_fsynced_and_closed':True,'diagnostic_only':True},timer,root=root)


def execute_diagnostic_path(c,result,reg,timer,*,root=ROOT):
    """Actual non-suite child path: guard creation/import/read/math/publication."""
    with CPUOnly():
        checkpoint(c,result,timer,'diagnostics_started',root=root)
        inputs=synthetic_inputs(c,timer,root=root) if c['role']=='fixture' else reg
        checkpoint(c,result,timer,'synthetic_inputs_published',root=root)
        records=historical_metadata(inputs,root=root)
        value=diagnostic(inputs,records,timer,root=root)
        checkpoint(c,result,timer,'diagnostics_recomputed',root=root)
        value['synthetic_scope']=c['mode']=='generated'
        actual_imports(c,result,root=root)
        checkpoint(c,result,timer,'actual_imports_verified',root=root)
        publish_diagnostic(c,result,value,timer,root=root)
        checkpoint(c,result,timer,'diagnostics_published',root=root)
    return value


def _child(c,spec,receipt_spec,reg,timer,*,root=ROOT):
    result=envelope(c,spec,receipt_spec)
    try:
        checkpoint(c,result,timer,'pre_array_completed',root=root)
        authority_fence('post_pre_array_before_first_numeric_native_or_array')
        if c['role']=='suite':
            case=c['inputs']['case_contract']
            require(execute_current_pytest(case['path'],c=c,spec=spec,receipt_spec=receipt_spec,timer=timer,root=root)==0,
                    'actual complete current E360 suite')
            result['case_receipt']=file_spec(strict_json(read_spec(case,root))['receipt'],root)
        else:
            execute_diagnostic_path(c,result,reg,timer,root=root)
        # original_array_reads counts decoded selected NPY arrays: two control,
        # six native, two startup; only the same three NPZ archives are read.
        result.update(run_valid=True,classification='diagnostic_process_only',
                      original_array_reads=10 if c['mode']=='original' else 0,
                      original_input_archives_read=3 if c['mode']=='original' else 0,
                      original_arrays_decoded=10 if c['mode']=='original' else 0)
        timer.check()
        check_sources(c['source_sha256'],root)
        require(retained(c,root)+c['retained_reserve_bytes']<=c['limits']['output_cap_bytes'],'final output reserve')
    except Exception as error:
        fail(result,error,'child-diagnostic')
        result['secondary_errors'].extend(getattr(error,'publication_errors',[]))
    finally:
        if any(k=='numpy' for k in sys.modules):
            try:
                with CPUOnly():
                    actual_imports(c,result,root=root)
            except Exception as error:
                fail(result,error,'actual-source-closure')
    try:
        with CPUOnly():
            # A failed path keeps only the prefix actually reached; never invent
            # skipped science phases just to add a publication checkpoint.
            if result['run_valid']:checkpoint(c,result,timer,'before_envelope_publication',root=root)
            result['elapsed_before_publication_s']=time.perf_counter()-timer.entry
            put(c,'envelope.json',result,timer,root=root,emergency=not result['run_valid'])
            timer.check()
    except Exception as error:
        fail(result,error,'late-envelope-publication')
    if result['first_error'] is not None:
        try:
            with CPUOnly():
                put(c,'revocation.json',result,timer,root=root,emergency=True)
        except Exception as error:
            fail(result,error,'revocation-authority-fence')
    print(json.dumps({'role':c['role'],'run_valid':result['run_valid'],'actual_full_return_pending':True,
                      'resource_extension':c['resource_extension'],'revalidation_amendment':c['revalidation_amendment'],
                      **SCOPES,**FALSE_MILESTONES},allow_nan=False),flush=True)
    return 0 if result['run_valid'] else 1


def main(argv=None):
    global ADMIN_CHECK
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scan')
    group.add_argument('--supervise')
    group.add_argument('--child')
    group.add_argument('--execute-current-pytest')
    parser.add_argument('--pre-spawn')
    args=parser.parse_args(argv)
    if args.scan:
        require(_SCANNER_BOOTSTRAP_POLICY is not None,'actual current scanner request bootstrap')
        scanner=Clock(ENTRY,_SCANNER_BOOTSTRAP_POLICY)
        ADMIN_CHECK=scanner.check
        with fixed_authority_scope(check=scanner.check):
            extension=effective_policy(pass_id=_request_value['pass_id'])
            typed_equal(extension['scanner_limits'],_SCANNER_BOOTSTRAP_POLICY,'same ENTRY bootstrap/scope exact limits')
            return scan_cli(args,scanner)
    if args.supervise:
        return supervise(args.supervise)
    if args.child:
        require(args.pre_spawn is not None,'explicit actual pre-spawn')
        return child(args.child,args.pre_spawn)
    return execute_current_pytest(args.execute_current_pytest)


def scan_cli(args,scanner):
    request_name=os.environ['E360_CALL_REQUEST']
    request=strict_json(safe(request_name).read_bytes())
    request['_request_path']=request_name
    check_call_request(request)
    typed_equal(request['call_role'],'scanner','scanner admitted role')
    typed_equal(request['expected_output'],args.scan,'scanner expected fresh output')
    scanner.check()
    scan_admission(registration())
    scanner.check()
    require(args.scan.startswith(BASE+'/parent-e360-'),'fresh E360 scanner output')
    report=scan_current(check=scanner.check)
    scanner.check()
    durable(safe(args.scan),json_bytes(report),cap=16<<20,check=scanner.check)
    scanner.check()
    print(json.dumps(file_spec(args.scan),allow_nan=False),flush=True)
    scanner.check()
    return 0


def scan_admission(reg,*,root=ROOT):
    extension=effective_policy(root)
    _,semantics,totals,local=ledger_state(reg,root)
    limits=extension['scanner_limits']
    held=sum(v['reserved_s']+sum(v['pending_reservations'].values()) for v in semantics['e360_reservations'].values())
    planned=sum(limits[k] for k in ('outer_s','planned_fixture_s','planned_suite_s','primary_admin_total_s','separate_same_pending_fallback_reserved_s'))
    require(not semantics['e360_reservations'] and semantics['original_jobs_running']==0,'scan serial no held job')
    budget=current_budget_authority(root)
    require(local['generated']+held+planned<=budget['generated_local_limit_s']
            and reg['generated_shared_baseline_s']+totals['generated']+held+planned<=budget['generated_shared_limit_s'],'scan full downstream future bound')
    return planned



_COMPLETED_STREAM_PINS=(*_COMPLETED_STREAM_PINS,*json.loads(r'''[{"path":"verification/banc-natural-gait-m3-20260919/parent-e360-generous-p30-ruff-stderr.txt","bytes":0,"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}]'''))
if __name__=='__main__':
    raise SystemExit(main())
