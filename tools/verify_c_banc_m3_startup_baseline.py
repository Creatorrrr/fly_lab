# Native, ownership and publication failures must all be preserved in envelopes.
# ruff: noqa: BLE001
"""E359 neutral startup99: one baseline and one independent saved-raw audit.

ROOT first reserves a role in the live shared ledger, then calls
build_contract(role, attempt_id, mode='generated'|'original',
adapter_qualification=Q_DESCRIPTOR, inputs=INPUT_DESCRIPTORS, root=ROOT).
The returned exact CONTRACT_KEYS record is saved at paths.contract. CLI:
  --supervise CONTRACT
  --child CONTRACT --pre-spawn RECEIPT  (sole unchanged E267 child argv)
  --execute-current-pytest CASE
ROOT saves the actual tool return at paths.actual_return (fields
actual_returned_exit_code:int and actual_outer_full_return_s:finite float), calls
seal_full_return(CONTRACT, file_spec(paths.actual_return)), and durably saves its
return at paths.full_exit. Internal elapsed is never the actual tool return.

Live ledger e359_reservations is a dict keyed by attempt_id+':'+role; each value
has exactly mode, role, reserved_s, pending_reservations. ROOT holds baseline
65+audit12, or generated fixture6/suite12, within local85/25 and shared limits.
original_reserved_s is the sum of all original held role+pending reservations;
original_jobs_running is1 while an original reservation is held, including
launch and closure, and0 for generated-only work. The shared owned lock proves
the sole actual running wrapper before Popen/arrays. ROOT releases the ledger
hold only after actual full closure. Ledger debit semantics and complete
immutable28-entry prefix are checked in parent and child; caller subsets fail.
Costs after that prefix belong to E359 and are appended once, including failed
administrative work. ROOT updates the live ledger after every actual return.

CASE exact keys: schema, source_sha256, expected_functions, receipt, pytest_ini,
fixture_exit. Schema current-pytest-case.v1, source_sha256=own_sources(),
expected_functions=test_functions(), fresh parent-e359-* receipt, empty owned
[pytest] config, and a previously sealed real generated E359 fixture descriptor.
Run standalone fixture before suite. Q exact keys: schema,source_sha256,
original_calls,original_array_reads,closed,suite_exit,fixture_exit,case_receipt;
schema adapter-qualification.v1. Both exits and all current three pytest phases
must close under identical current source. No inherited Q is relabelled.

Only baseline_startup99 and baseline_audit are original roles, max1 each. Audit
inputs are baseline_exit and compare_history:bool. Neutral body runs exactly99
controls even when startup fails; no full-prefix evaluator/candidate/alpha search
exists here. Saved transfer proof is digest coverage, not GPU reacquisition.
F1-F8/natural gait/biology/adoption/M2/UI/F6 remain false.
"""
from __future__ import annotations

import time

ENTRY = time.perf_counter()
import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import verify_c_banc_m3_candidate_body_handoff as lib

BASE, PYTHON, SUPERVISOR, OBSERVER = lib.BASE, lib.PYTHON, lib.SUPERVISOR, lib.OBSERVER
OWNED = ('tools/verify_c_banc_m3_startup_baseline.py', 'tests/test_c_banc_startup_baseline.py')
REGISTRATION = BASE + '/parent-e359-reflected-registration.json'
REGISTRATION_SHA = '40322a97b17496e36200f83e41e0362eb83a6b9284cb31753454a72f00335a47'
INHERITED_SHA = 'a76c7a2044f643b63e241879fcfe51bcc29c766dceea024e69617cdd5cb4b4eb'
FALSE_MILESTONES = lib.FALSE_MILESTONES
OUTER_CLEANUP_MARGIN_S, OUTER_REPORT_MARGIN_S = .5, .1
require, numeric, typed_equal = lib.require, lib.numeric, lib.typed_equal
strict_json, json_bytes, normalize, sha = lib.strict_json, lib.json_bytes, lib.normalize, lib.sha
safe, file_spec, read_spec, descriptor, durable = lib.safe, lib.file_spec, lib.read_spec, lib.descriptor, lib.durable
retained, put, error_record, fail = lib.retained, lib.put, lib.error_record, lib.fail
monitor_check, gpu_watch, npz_bytes = lib.monitor_check, lib.gpu_watch, lib.npz_bytes
finish_owned = lib.finish_owned
CONTRACT_KEYS = {'schema','mode','role','attempt','attempt_id','phase','original_calls',
                 'registration','source_sha256','limits','commands','supervisor_output',
                 'max_tree_rss_bytes','paths','cumulative_before_s','cumulative_limit_s',
                 'stage_used_s','stage_limit_s','local_used_s','local_limit_s','reserved_s',
                 'pending_reservations','ledger_debits','ledger_semantics','adapter_qualification',
                 'inputs','retained_reserve_bytes','cpu_threads'}


class Clock(lib.Clock):
    def check(self, stage='internal'):
        now = self.clock()
        require(math.isfinite(now) and now >= self.entry, 'same monotone E359 ENTRY')
        if now > self.entry + self.limits[stage + '_s']:
            raise TimeoutError('E359 same-ENTRY ' + stage + ' deadline')


def own_sources(root=ROOT):
    return {name: sha(safe(name, root).read_bytes()) for name in
            (*OWNED, lib.OWNED[0], lib.OWNED[1], SUPERVISOR, OBSERVER)}


def registration(root=ROOT):
    raw = safe(REGISTRATION, root).read_bytes()
    require(len(raw) == 101176 and sha(raw) == REGISTRATION_SHA, 'fixed reviewed E359 registration')
    reg = strict_json(raw)
    inherited = read_spec(reg['inherited_cost_ledger'], root)
    require(len(inherited) == 12600 and sha(inherited) == INHERITED_SHA, 'fixed complete inherited cost snapshot')
    return reg


def closure_sources(reg, root=ROOT):
    pins = {}
    for key in ('science_protected_text_sha256', 'body_source_sha256', 'unchanged_adapter_source_sha256'):
        for name, digest in reg[key].items():
            require(name not in pins or pins[name] == digest, 'source pin collision')
            pins[name] = digest
            path = Path(name)
            if path.is_absolute() and path.resolve().is_relative_to(root.resolve()):
                alias = path.resolve().relative_to(root.resolve()).as_posix()
                require(alias not in pins or pins[alias] == digest, 'source alias pin collision')
                pins[alias] = digest
    for key in ('plan','actual_Astra_review','Astra_review_md','ROOT_reflection','inherited_registration',
                'inherited_adapter_qualification','inherited_cost_ledger','science_registration',
                'science_qualification','science_freeze','body_inventory','body_native_dependencies'):
        spec = reg[key]
        read_spec(spec, root)
        require(spec['path'] not in pins or pins[spec['path']] == spec['sha256'], 'metadata pin collision')
        pins[spec['path']] = spec['sha256']
    pins[REGISTRATION] = REGISTRATION_SHA
    for name, digest in own_sources(root).items():
        require(name not in pins or pins[name] == digest, 'immutable inherited adapter/current own source')
        pins[name] = digest
    return pins


def charged_cost(spec, mode, root=ROOT):
    row = strict_json(read_spec(spec, root))
    require(row['schema'] in ('flylab.m3.e358.actual-full-return.v1',
                             'flylab.m3.e358.administrative-full-return.v1',
                             'flylab.m3.e359.actual-full-return.v1',
                             'flylab.m3.e359.administrative-full-return.v1')
            and row['mode'] == mode and row['charged_once'] is True, 'once-charged actual cost schema/mode')
    wall = numeric(row['actual_outer_full_return_s'], 'actual cost', minimum=1e-12)
    require(type(wall) is float and type(row['cost_id']) is str and row['cost_id'], 'exact typed cost identity/time')
    actual = strict_json(read_spec(row['actual_tool_return'], root))
    typed_equal(actual['actual_outer_full_return_s'], wall, 'actual external wall linkage')
    require(type(actual['actual_returned_exit_code']) is int, 'typed actual return code')
    return row


def ledger_state(reg, root=ROOT):
    inherited = strict_json(read_spec(reg['inherited_cost_ledger'], root))
    live = strict_json(safe(reg['shared_ledger_path'], root).read_bytes())
    prefix = inherited['ledger']['debits']
    require(inherited['complete_expected_cost_ids'] == reg['inherited_cost_ids']
            and [v['cost_id'] for v in prefix] == reg['inherited_cost_ids']
            and len(prefix) == 28, 'exact inherited complete cost roster')
    debits = live['debits']
    require(type(debits) is list and len(debits) >= len(prefix), 'complete live inherited costs')
    typed_equal(debits[:len(prefix)], prefix, 'immutable inherited debit prefix')
    require(len({v['cost_id'] for v in debits}) == len(debits), 'no duplicate/replayed cost ID')
    used, local = {'original': [], 'generated': []}, {'original': [], 'generated': []}
    for index, debit in enumerate(debits):
        require(set(debit) == {'cost_id','mode','actual_outer_full_return_s','descriptor','actual_returned_exit_code'},
                'literal live debit fields')
        row = charged_cost(debit['descriptor'], debit['mode'], root)
        for key in ('cost_id','mode','actual_outer_full_return_s'):
            typed_equal(row[key], debit[key], 'complete live debit ' + key)
        require(type(debit['actual_returned_exit_code']) is int, 'live debit typed exit')
        actual = strict_json(read_spec(row['actual_tool_return'], root))
        typed_equal(debit['actual_returned_exit_code'], actual['actual_returned_exit_code'], 'debit actual exit linkage')
        require(debit['mode'] in used, 'finite debit mode')
        used[debit['mode']].append(debit['actual_outer_full_return_s'])
        if index >= len(prefix):
            require('e359' in debit['cost_id'] or row.get('schema','').startswith('flylab.m3.e359.'),
                    'accounted append-only E359 costs')
            local[debit['mode']].append(debit['actual_outer_full_return_s'])
    totals = {k: round(math.fsum(v),9) for k,v in used.items()}
    locals_ = {k: round(math.fsum(v),9) for k,v in local.items()}
    fixed = {'original_cumulative_before_s': reg['original_cumulative_before_s'],
             'original_cumulative_limit_s': 10200, 'original_stage_cap_s': 300,
             'new_candidate_attempts': 1, 'modified_retry_used': 0, 'modified_retry_limit': 1,
             'generated_shared_cumulative_limit_s': 150, 'generated_new_stage_retained_cap_bytes': 16 << 20,
             'e359_original_local_limit_s':85,'e359_generated_local_limit_s':25,
             'e359_registration_path':REGISTRATION}
    for key, expected in fixed.items():
        typed_equal(live[key], expected, 'fixed shared ledger ' + key)
    derived = {'original_stage_used_s': totals['original'],
               'original_cumulative_used_s': reg['original_cumulative_before_s'] + totals['original'],
               'original_remaining_s': 10200 - reg['original_cumulative_before_s'] - totals['original'],
               'generated_stage_used_s': totals['generated'],
               'generated_shared_cumulative_used_s': reg['generated_shared_baseline_s'] + totals['generated'],
               'generated_shared_remaining_s': 150 - reg['generated_shared_baseline_s'] - totals['generated']}
    for key, expected in derived.items():
        actual = numeric(live[key], 'live ' + key)
        # Ledger additions may differ by one binary rounding bit; reject bools,
        # and compare decimal nanosecond accounting rather than science values.
        require(type(actual) is float and round(actual, 9) == round(expected, 9), 'derived live debit semantic ' + key)
    local_semantics={}
    for mode in ('original','generated'):
        key='e359_'+mode+'_local_used_s'
        expected=0 if locals_[mode]==0 else locals_[mode]
        typed_equal(live[key],expected,'derived complete E359 local spend '+key)
        local_semantics[key]=live[key]
    reservations = live.get('e359_reservations', {})
    require(type(reservations) is dict, 'typed live E359 reservations')
    original_reserved = 0
    for cost_id, held in reservations.items():
        require(type(cost_id) is str and cost_id and cost_id not in {v['cost_id'] for v in debits}
                and set(held) == {'mode','role','reserved_s','pending_reservations'}, 'literal held cost reservation')
        mode, role = held['mode'], held['role']
        require(type(mode) is str and mode in ('original','generated') and type(role) is str
                and role in reg['roles'] and (mode == 'original') == role.startswith('baseline_'), 'reservation role/mode')
        typed_equal(held['reserved_s'], reg['roles'][role]['outer_s'], 'exact role reserve')
        expected_pending = ({'baseline_audit': 12} if role == 'baseline_startup99'
                            else {'suite': 12} if role == 'fixture' else {})
        typed_equal(held['pending_reservations'], expected_pending, 'finite pending reservation')
        if mode == 'original':
            original_reserved += held['reserved_s'] + sum(held['pending_reservations'].values())
    typed_equal(live['original_reserved_s'], original_reserved, 'all shared original reservations')
    original_jobs = sum(v['mode'] == 'original' for v in reservations.values())
    typed_equal(live['original_jobs_running'], original_jobs, 'held original job count')
    require(original_jobs <= 1, 'no unseen concurrent original reservation')
    require(sum(v['mode']=='generated' for v in reservations.values())<=1,'serial generated reservation')
    semantics = {**fixed, **local_semantics, **{k: live[k] for k in derived}, 'original_reserved_s': original_reserved,
                 'original_jobs_running': original_jobs, 'e359_reservations': reservations}
    return debits, semantics, totals, locals_


def validate_qualification(spec, sources=None, root=ROOT):
    q=strict_json(read_spec(spec,root))
    require(set(q)=={'schema','source_sha256','original_calls','original_array_reads','closed',
                    'suite_exit','fixture_exit','case_receipt'}
            and q['schema']=='flylab.m3.e359.adapter-qualification.v1'
            and q['source_sha256']==own_sources(root) and type(q['original_calls']) is int and q['original_calls']==0
            and type(q['original_array_reads']) is int and q['original_array_reads']==0 and q['closed'] is True,
            'current new E359 original0 Q')
    for key in ('fixture_exit','suite_exit'):
        row=closed_exit(q[key],root=root)
        require(row['role']==key.removesuffix('_exit') and row['mode']=='generated','actual generated current Q role')
        value=strict_json(read_spec(row['envelope'],root))
        require(type(value['original_array_reads']) is int and value['original_array_reads']==0,'actual original0 publisher')
        if key=='suite_exit':
            require(value['case_receipt']==q['case_receipt'],'same actual whole-case publication')
    receipt=strict_json(read_spec(q['case_receipt'],root))
    validate_case_receipt(receipt,root=root)
    case=strict_json(read_spec(receipt['case_contract'],root))
    require(case['fixture_exit']==q['fixture_exit'] and case['source_sha256']==q['source_sha256']
            and case['expected_functions']==receipt['expected_functions'],'same complete CASE/fixture/source closure')
    if sources is not None:
        require(all(sources.get(k)==v for k,v in own_sources(root).items()),'Q current original contract own-source')
    return q


def names(attempt_id, role):
    require(type(attempt_id) is str and re.fullmatch('[a-z0-9][a-z0-9-]{0,55}', attempt_id), 'literal E359 attempt ID')
    stem = BASE + '/parent-e346-e359-' + attempt_id + '-' + role
    return {'contract': stem+'.json', 'output': BASE+'/e359-'+attempt_id+'-'+role,
            'pre_spawn': stem+'-pre-spawn.json','outer':stem+'-outer.json',
            'outer_revocation':stem+'-outer-revocation.json','actual_return':stem+'-actual-tool-return.json',
            'full_exit':stem+'-full-exit.json','supervision':stem+'-supervision.json',
            'stdout':stem+'-child.stdout.txt','stderr':stem+'-child.stderr.txt',
            'witness_stdout':stem+'-witness.stdout.txt','witness_stderr':stem+'-witness.stderr.txt'}


def build_contract(role, attempt_id, *, mode='original', adapter_qualification=None, inputs=None, root=ROOT):
    reg = registration(root)
    require(role in reg['roles'] and mode in ('original','generated')
            and (mode == 'original') == role.startswith('baseline_'), 'explicit E359 role/mode')
    debits, semantics, totals, local = ledger_state(reg, root)
    limits, paths = reg['roles'][role], names(attempt_id, role)
    pending = ({'baseline_audit': 12} if role == 'baseline_startup99'
               else {'suite': 12} if role == 'fixture' else {})
    argv = [PYTHON,'-X','utf8','-B',OWNED[0],'--child',paths['contract'],'--pre-spawn',paths['pre_spawn']]
    original = mode == 'original'
    return {'schema':'flylab.m3.e359.supervisor-contract.v1','mode':mode,'role':role,'attempt':1,
            'attempt_id':attempt_id,'phase':'E359-'+role,'original_calls':int(original),
            'registration':file_spec(REGISTRATION, root),
            'source_sha256':closure_sources(reg,root) if original else own_sources(root), 'limits':dict(limits),
            'commands':[{'argv':argv,'stdout':paths['stdout'],'stderr':paths['stderr'],'wall_s':limits['child_s']}],
            'supervisor_output':paths['supervision'],'max_tree_rss_bytes':limits['host_bytes'],'paths':paths,
            'cumulative_before_s':(reg['original_cumulative_before_s'] if original else reg['generated_shared_baseline_s'])+totals[mode],
            'cumulative_limit_s':10200 if original else 150,'stage_used_s':totals[mode],
            'stage_limit_s':300 if original else 150-reg['generated_shared_baseline_s'],
            'local_used_s':local[mode],'local_limit_s':85 if original else 25,'reserved_s':limits['outer_s'],
            'pending_reservations':pending,'ledger_debits':debits,'ledger_semantics':semantics,
            'adapter_qualification':adapter_qualification,'inputs':inputs or {},
            'retained_reserve_bytes':min(1 << 20,limits['output_cap_bytes']//4),'cpu_threads':1}


def validate_contract(name, *, root=ROOT, child=False):
    raw = safe(name,root).read_bytes()
    require(len(raw) <= 256 << 10, 'bounded E359 contract')
    c = strict_json(raw)
    require(set(c) == CONTRACT_KEYS and c['schema'] == 'flylab.m3.e359.supervisor-contract.v1', 'literal E359 contract schema')
    reg = registration(root)
    for path,digest in reg['unchanged_adapter_source_sha256'].items():
        require(sha(safe(path,root).read_bytes())==digest,'immutable E358/observer dependency')
    expected = build_contract(c['role'],c['attempt_id'],mode=c['mode'],adapter_qualification=c['adapter_qualification'],
                              inputs=c['inputs'],root=root)
    typed_equal(c,expected,'current full E359 authority')
    require(name == c['paths']['contract'], 'literal supervisor contract path')
    lib.check_sources(c['source_sha256'],root)
    cost_id = c['attempt_id'] + ':' + c['role']
    reservations = c['ledger_semantics']['e359_reservations']
    require(cost_id in reservations, 'actual live role reservation before Popen/arrays')
    typed_equal(reservations[cost_id],{k:c[k] for k in ('mode','role','reserved_s','pending_reservations')},'same live role hold')
    role_costs = [strict_json(read_spec(v['descriptor'],root)) for v in c['ledger_debits'][28:]]
    require(not any(v.get('original_attempt_role',v.get('role')) == c['role'] and v['mode'] == c['mode']
                    and (v['schema']=='flylab.m3.e359.actual-full-return.v1' or 'original_attempt_role' in v)
                    for v in role_costs)
            if c['mode']=='original' else True, 'baseline/audit each max1 including failures')
    headroom = 8 if c['mode']=='original' else 7
    administrative = math.fsum(v['actual_outer_full_return_s'] for v in role_costs
                               if v['mode']==c['mode'] and 'administrative' in v['schema'])
    downstream = sum(c['pending_reservations'].values())
    require(c['local_used_s']+c['reserved_s']+downstream+max(0,headroom-administrative) <= c['local_limit_s']
            and c['stage_used_s']+c['reserved_s']+downstream <= c['stage_limit_s']
            and c['cumulative_before_s']+c['reserved_s']+downstream <= c['cumulative_limit_s'], 'local/shared/global prospective budgets')
    required_inputs = {'fixture':set(),'suite':{'case_contract'},'baseline_startup99':set(),
                       'baseline_audit':{'baseline_exit','compare_history'}}[c['role']]
    require(set(c['inputs']) == required_inputs,'finite role inputs, learned candidate forbidden')
    if c['role']=='suite':
        read_spec(c['inputs']['case_contract'],root)
    if c['mode']=='original':
        validate_qualification(c['adapter_qualification'],c['source_sha256'],root)
        lib.passive_science(reg,root)
        if c['role']=='baseline_audit':
            require(type(c['inputs']['compare_history']) is bool,'typed optional historical diagnostic')
            previous = closed_exit(c['inputs']['baseline_exit'],root=root)
            require(previous['role']=='baseline_startup99' and previous['attempt_id']==c['attempt_id']
                    and c['inputs']['baseline_exit'] in [v['descriptor'] for v in c['ledger_debits']], 'actual once-debited same baseline dependency')
    else:
        require(c['adapter_qualification'] is None,'generated original0 no original Q bypass')
    keys = ('output','outer','outer_revocation','actual_return','full_exit')
    if not child:
        keys += ('pre_spawn','supervision','stdout','stderr','witness_stdout','witness_stderr')
    require(all(not safe(c['paths'][k],root).exists() for k in keys),'fresh E359 paths/no replay')
    return c,descriptor(name,raw),reg


def envelope(c,spec,receipt):
    return {'schema':'flylab.m3.e359.provisional-envelope.v1','contract':spec,'pre_spawn':receipt,
            'mode':c['mode'],'role':c['role'],'attempt_id':c['attempt_id'],'source_sha256':c['source_sha256'],
            'registration':c['registration'],'adapter_qualification':c['adapter_qualification'],
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


def pre_spawn(name, entry, *, root=ROOT):
    c, spec, frozen = validate_contract(name, root=root)
    timer = Clock(entry, c["limits"])
    timer.check("outer")
    import psutil
    owner = psutil.Process(os.getpid())
    shared_owner(c,name,owner.pid,owner.create_time(),root=root)
    receipt = {"schema": "flylab.m3.e359.actual-pre-spawn.v1", "contract": spec,
               "authority": c, "entry_perf_counter": entry, "parent_pid": owner.pid,
               "parent_create_time": owner.create_time(), "source_sha256": c["source_sha256"],
               "role": c["role"], "attempt_id": c["attempt_id"], "attempt": c["attempt"],
               "original_array_reads": 0, "scientific_native_imports": 0,
               "Popen_before_receipt": 0, "reservation_held": True,
               "cumulative_before_s": c["cumulative_before_s"], "cumulative_after_s": None,
               "cumulative_limit_s": c["cumulative_limit_s"], "reserved_s": c["reserved_s"],
               "pending_reservations": c["pending_reservations"], "limits": c["limits"],
               "stage_used_s": c["stage_used_s"], "stage_limit_s": c["stage_limit_s"],
               "original_calls": c["original_calls"], "max_tree_rss_bytes": c["max_tree_rss_bytes"],
               "retained_reserve_bytes": c["retained_reserve_bytes"], "cpu_threads": c["cpu_threads"]}
    require(not any(k == "torch" or k == "numpy" or k.startswith("flylab.") for k in sys.modules),
            "pre-spawn numeric/native imports0")
    durable(safe(c["paths"]["pre_spawn"], root), json_bytes(receipt), cap=256 << 10,
            check=lambda: timer.check("outer"))
    return c, spec, file_spec(c["paths"]["pre_spawn"], root), frozen


def pre_array(name, receipt_name, *, root=ROOT):
    c, spec, frozen = validate_contract(name, root=root, child=True)
    require(receipt_name == c["paths"]["pre_spawn"], "same actual pre-spawn receipt path")
    receipt_spec = file_spec(receipt_name, root)
    receipt = strict_json(read_spec(receipt_spec, root))
    expected = {"schema": "flylab.m3.e359.actual-pre-spawn.v1", "contract": spec, "authority": c,
                "source_sha256": c["source_sha256"], "reservation_held": True,
                "original_array_reads": 0, "scientific_native_imports": 0,
                "Popen_before_receipt": 0, "cumulative_after_s": None}
    for key in ("role", "attempt_id", "attempt", "cumulative_before_s", "cumulative_limit_s",
                "reserved_s", "pending_reservations", "limits", "stage_used_s", "stage_limit_s",
                "original_calls", "max_tree_rss_bytes", "retained_reserve_bytes", "cpu_threads"):
        expected[key] = c[key]
    require(set(receipt) == set(expected) | {"entry_perf_counter", "parent_pid", "parent_create_time"},
            "literal same-ENTRY receipt authority fields")
    for key, value in expected.items():
        typed_equal(receipt[key], value, "same-ENTRY pre-array authority " + key)
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
    timer = Clock(entry, c["limits"])
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
                raise TimeoutError("E359 same-ENTRY owned completion deadline")
            sleep(min(.005, remaining))
        except Exception as error:
            errors.append(error_record(error, "owned-completion-wait"))
            return matching, unknown, errors


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
        payload = json_bytes(record)
        require(retained(c, root) + len(payload) <= c["limits"]["output_cap_bytes"], "outer failure retained cap")
        durable(safe(target, root), payload, cap=256 << 10)
    return record


def supervise(name, *, entry=ENTRY, root=ROOT):
    """Observe the immutable observers; kill only PID/create-time owned identities."""
    import psutil
    lock = safe(BASE + "/parent-e358-original-active.lock", root)
    raw_c = strict_json(safe(name, root).read_bytes())
    held = raw_c.get("mode") == "original"
    token = {"pid": os.getpid(), "created": psutil.Process(os.getpid()).create_time(), "contract": name}
    if held:
        durable(lock, json_bytes(token), cap=1024)
    process = None
    identities, killed, secondary, first = {}, [], [], None
    c = spec = receipt = timer = None
    matching, unknown, missing_identities = {}, {}, []
    supervisor_killed = False
    try:
        c, spec, receipt, _ = pre_spawn(name, entry, root=root)
        timer = Clock(entry, c["limits"])
        argv = [PYTHON, "-X", "utf8", "-B", OBSERVER, name]
        with safe(c["paths"]["witness_stdout"], root).open("xb") as out, \
                safe(c["paths"]["witness_stderr"], root).open("xb") as err:
            require(timer.remaining(OUTER_CLEANUP_MARGIN_S) > 0, "outer cleanup/publication margin")
            shared_owner(c,name,token['pid'],token['created'],root=root)
            process = subprocess.Popen(argv, cwd=root, stdout=out, stderr=err,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            owner = psutil.Process(process.pid)
            identities[owner.pid] = owner.create_time()
            while process.poll() is None:
                if timer.remaining(OUTER_CLEANUP_MARGIN_S) <= 0:
                    raise TimeoutError("E359 same-ENTRY outer work deadline with cleanup margin")
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
        if held and process is None:
            require(strict_json(lock.read_bytes()) == token, "unspawned reservation lock identity")
            lock.unlink()
        raise ValueError("E359 pre-spawn rejected: " + str(first))
    matching, unknown, closure_errors = bounded_owned_completion(identities, timer, psutil)
    secondary.extend(closure_errors)
    ended = process is not None and process.poll() is not None and not matching and not unknown and not missing_identities
    if not ended and first is None:
        first = error_record(RuntimeError("owned identities unresolved at outer completion"), "owned-completion")
    if closure_errors and first is None:
        first, secondary = secondary[0], secondary[1:]
    # Scientific failure can naturally close; technical cleanup cannot qualify.
    natural = ended and not killed and not supervisor_killed and not secondary and not missing_identities
    if first is not None and (first["type"] == "TimeoutError" or first["message"] != "actual unchanged supervisor natural job exit"):
        natural = False
    if held and ended:
        try:
            # Last actual identity check before relinquishing the reservation.
            matching, unknown, release_errors = owned_presence(identities, psutil)
            secondary.extend(release_errors)
            require(not matching and not unknown and not release_errors, "owned closure before reservation release")
            require(strict_json(lock.read_bytes()) == token, "owned reservation lock identity")
            lock.unlink()
            held = False
        except Exception as error:
            if first is None:
                first = error_record(error, "reservation-release")
            else:
                secondary.append(error_record(error, "reservation-release"))
            natural = False
            ended = not matching and not unknown and not missing_identities and process.poll() is not None
    output = c["paths"]["output"] + "/envelope.json"
    output_spec = file_spec(output, root) if safe(output, root).exists() else None
    record = {"schema": "flylab.m3.e359.outer-provisional.v1", "contract": spec,
              "pre_spawn": receipt, "envelope": output_spec, "natural_closed": bool(natural),
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
    record = publish_outer(c, record, timer, root=root)
    return 0 if record["first_error"] is None and record["secondary_errors"] == [] else 1


def seal_full_return(contract_name, actual_tool_return, *, root=ROOT):
    """ROOT calls after the actual exec return. Returns a record; writes nothing.

    ROOT stores result under fresh E359 .json and debits it exactly once. The
    external return schema is ROOT-owned; required fields are the two actual_*
    names documented above. Its duration must include this entire wrapper call.
    """
    c = strict_json(safe(contract_name, root).read_bytes())
    require(actual_tool_return["path"] == c["paths"]["actual_return"], "literal external actual tool-return path")
    actual = strict_json(read_spec(actual_tool_return, root))
    wall = numeric(actual["actual_outer_full_return_s"], "external full-return", minimum=1e-12)
    code = actual["actual_returned_exit_code"]
    require(type(code) is int, "actual exit code")
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
        errors.append(error_record(RuntimeError('actual full-return exceeds E359 local budget'), 'actual-full-return'))
    valid = (code == 0 and wall <= c["limits"]["outer_s"] and outer["natural_closed"] is True
             and outer["all_owned_ended"] is True and outer["known_identity_cleanup"] == []
             and outer["first_error"] is None and outer["secondary_errors"] == [] and value is not None
             and value["run_valid"] is True and value["first_error"] is None and value["secondary_errors"] == []
             and value["revoked"] is False)
    valid &= not (folder / "failure.json").exists() and not (folder / "revocation.json").exists()
    valid &= revocation_spec is None and not errors
    valid &= c["stage_used_s"] + wall <= c["stage_limit_s"]
    valid &= c["cumulative_before_s"] + wall <= c["cumulative_limit_s"]
    valid &= c['local_used_s'] + wall <= c['local_limit_s']
    valid &= retained(c, root) + (256 << 10) <= c["limits"]["output_cap_bytes"]
    return {"schema": "flylab.m3.e359.actual-full-return.v1", "mode": c["mode"],
            "role": c["role"], "attempt_id": c["attempt_id"], "attempt": c["attempt"],
            "cost_id": c["attempt_id"] + ":" + c["role"], "charged_once": True,
            "contract": file_spec(contract_name, root), "outer": outer_spec,
            "outer_revocation": revocation_spec, "envelope": pending,
            "source_sha256": c["source_sha256"], "original_calls": c["original_calls"],
            "actual_tool_return": actual_tool_return, "actual_outer_full_return_s": wall,
            "exit_code": code, "qualified": bool(valid), "all_owned_ended": outer["all_owned_ended"],
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
    row = charged_cost(spec, strict_json(read_spec(spec,root))['mode'],root)
    require(row['schema']=='flylab.m3.e359.actual-full-return.v1' and row['qualified'] is True
            and type(row['exit_code']) is int and row['exit_code']==0 and row['all_owned_ended'] is True
            and row['natural_closed'] is True and row['known_identity_cleanup']==[]
            and row['first_error'] is None and row['secondary_errors']==[]
            and row['actual_full_return_pending'] is False and row.get('outer_revocation') is None,
            'actual natural qualified E359 full return')
    c = strict_json(read_spec(row['contract'],root))
    require(row['source_sha256']==c['source_sha256'] and row['role']==c['role']
            and row['attempt_id']==c['attempt_id'] and row['attempt']==1
            and all(c['source_sha256'].get(k)==v for k,v in own_sources(root).items()), 'exact current E359 closed source')
    lib.check_sources(c['source_sha256'],root)
    outer, value = strict_json(read_spec(row['outer'],root)), strict_json(read_spec(row['envelope'],root))
    require(outer['contract']==row['contract'] and outer['envelope']==row['envelope']
            and outer['natural_closed'] is True and outer['all_owned_ended'] is True
            and outer['first_error'] is None and outer['secondary_errors']==[]
            and value['contract']==row['contract'] and value['source_sha256']==row['source_sha256']
            and value['run_valid'] is True and value['revoked'] is False
            and value['first_error'] is None and value['secondary_errors']==[], 'current unrevoked actual publisher linkage')
    folder = safe(c['paths']['output'],root)
    require(not any((folder/v).exists() for v in ('failure.json','revocation.json'))
            and not safe(c['paths']['outer_revocation'],root).exists(), 'no late E359 revocation')
    return row


def verify_work(counters, construction, count=99):
    require(type(count) is int and count==99, 'exact99 prefix; no300/full evaluator')
    lib.verify_work(counters,construction,count)
    require(counters.get('advance_unreturned_unknown',False) is False, 'known returned advance')
    return True


def source_data(reg, timer, *, b):
    """133 original inputs; invoked only after this worker's pre-array authority."""
    import numpy as np

    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    inv = strict_json(read_spec(reg['body_inventory']))
    core, ledger = ref.native(), []
    roster = reg['source_input_roster']
    selected = inv['selected_archives']
    excluded = {selected[k]['path'] for k in ('reference','audited_candidate')}
    require(len(roster)==133 and set(roster)==set(inv['files'])-excluded
            and not excluded.intersection(roster) and reg['selected_source_roles']==['recurrent_mask','startup'],
            'literal source133 excludes reference/learned candidate')
    for name,spec in roster.items():
        timer.check('science')
        path = safe(name)
        require(path.stat().st_size==spec['bytes'] and core.digest(path,lambda:timer.check('science'))==spec['sha256'],
                'original source bytes/hash inside child')
        if path.suffix=='.npy':
            with path.open('rb') as stream:
                core.strict_npy_header(stream)
        ledger.append({'path':name,'expected_sha256':spec['sha256'],'bytes':spec['bytes']})
    deps = strict_json(read_spec(reg['body_native_dependencies']))
    require(deps['network_asset_download_permitted'] is False and deps['mesh_cache_complete_currently'] is True,
            'registered offline native assets')
    for name,spec in deps['native_constructor_asset_bytes'].items():
        timer.check('science')
        path = Path(name)
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and path.stat().st_size==spec['bytes'] and core.digest(path,lambda:timer.check('science'))==spec['sha256'],
                'finite registered native constructor asset')
    b.whole_metadata(ROOT,inv)
    api = b.resolve_original_api()
    graph = api['GraphStore'].load(safe(inv['graph_bundle']))
    saved = api['StateStore'].load(safe(inv['initial_whole']),max_files=123)
    require(graph.hash==inv['graph_hash'],'exact source graph')
    api['recovery'].checked_checkpoint(saved)
    arrays = {role:core.load_selected(ROOT,selected[role],ledger,timely=lambda:timer.check('science'),generated=False)
              for role in ('recurrent_mask','startup')}
    theta = np.ones(8547,np.float64)
    groups, positions = arrays['recurrent_mask']['edge_parameters'], arrays['recurrent_mask']['positions']
    require(groups.dtype==positions.dtype==np.dtype('int64') and groups.shape==positions.shape
            and np.array_equal(np.unique(groups),np.arange(8547)), 'source8547 group coverage')
    return {'graph':graph,'saved':saved,'inventory':inv,'ledger':ledger,'theta':theta,
            'positions':positions,'groups':groups,'startup':arrays['startup']}


def select_live_source(data):
    require('selected_weights' not in data and 'source_csr' in data,'no fabricated preconstruct selected input')
    data['selected_weights'] = data['source_csr'].data[data['positions']].copy()
    return data['selected_weights']


def neutral_values(data, patch, actual_csr, actual_identity):
    import numpy as np
    theta, source = data['theta'],data['source_csr']
    require(theta.dtype==np.dtype('float64') and theta.shape==(8547,)
            and theta.tobytes()==np.ones(8547,np.float64).tobytes()
            and np.array_equal(theta,theta.astype(np.float32).astype(np.float64)), 'exact neutral float64 ones8547')
    selected = source.data[data['positions']]
    for observed in (data['selected_weights'],patch['original32'],patch['actual32']):
        require(observed.dtype==np.dtype('float32') and observed.shape==selected.shape
                and observed.tobytes()==selected.tobytes() and np.isfinite(observed).all()
                and np.all(observed!=0), 'actual source selected bytes, never fabricated')
    for key in ('data','indices','indptr'):
        want, observed = getattr(source,key),getattr(actual_csr,key)
        require(want.dtype==observed.dtype and want.shape==observed.shape
                and want.tobytes()==observed.tobytes(),'neutral full source CSR '+key+' bytes')
    require(actual_identity==data['inventory']['source_network_identity']
            and patch['source_network_identity']==patch['network_identity']==actual_identity,
            'neutral network/source identity equality')
    return True


def initial_neutral(session,data,timer,*,b,factory):
    import numpy as np
    selected = select_live_source(data)
    observed = session.snapshot()
    patch = factory.efficacy.checked_checkpoint(observed)
    neutral_values(data,patch,b.host_csr(session.network),session.network.identity)
    inverse = factory.efficacy._source_representation(observed,patch)
    factory.recovery.require_saved_whole_equal(inverse,data['saved'],'E359 exact inverse whole including wall')
    inv = data['inventory']
    require(session.control_tick==session.network.tick==0 and session.body.physics_time()==0
            and session.graph.hash==inv['graph_hash'] and session.adapter.identity==inv['source_adapter_identity']
            and session.network.interval_steps==session.neural_steps_per_control==20,'neutral source/decoder/clock0')
    require(session._cpg_digest()==data['saved']['cpg_hash']==session.cpg_hash,'exact initial source CPG')
    _, boundary = b.capture(session)
    require(b.e.exact(boundary['command'],np.asarray(data['saved']['body']['action']['joint_angles'],np.float64))
            and b.e.exact(boundary['tendon'],data['saved']['adapter']['original']['original']['original']['tendon_activation']),
            'source command/tendon boundary0')
    palette = len(np.unique(data['source_csr'].data.view(np.uint32)))
    require(0<palette<=16384,'current source palette cap')
    proof = b.verify_transfer(session.network,data['source_csr'],data['positions'],data['groups'],data['theta'],selected,
                              expected_palette=palette,timely=lambda:timer.check('science'))
    require(proof['candidate_CSR_data_sha256']==proof['source_CSR_data_sha256'],'neutral full CSR equality')
    require(patch['graph_hash']==inv['graph_hash']
            and patch['source_adapter_identity']==inv['source_adapter_identity'],'neutral patch source provenance')
    proof.update(source_inverse_whole_exact=True,source_boundary_command_and_tendon_exact=True,
                 source_network_identity=inv['source_network_identity'],source_adapter_identity=session.adapter.identity,
                 graph_hash=session.graph.hash,cpg_hash=session.cpg_hash,body_hash=session.body.model_hash,
                 theta_sha256=b.array_sha(data['theta']),selected_sha256=b.array_sha(selected),
                 positions_sha256=b.array_sha(data['positions']),groups_sha256=b.array_sha(data['groups']),
                 trial_schema=observed['schema'],patch_identity=patch['patch_identity'],
                 patch_schema=patch['schema'],source_trial_schema=patch['source_trial_schema'],
                 source_patch_identity=patch['source_patch_identity'],
                 boundary_command_sha256=b.array_sha(boundary['command']),
                 boundary_tendon_sha256=b.array_sha(boundary['tendon']),neutral_source_CSR_equal=True)
    return proof


def record_imports(result,c,actual):
    result['actual_import_sha256'] = actual
    result['source_closure_mismatches'] = [{'path':k,'actual_sha256':v,'registered_sha256':c['source_sha256'].get(k)}
                                         for k,v in actual.items() if c['source_sha256'].get(k)!=v]
    require(not result['source_closure_mismatches'],'actual neutral body imported source closure')


def run_prefix(session,factory,data,proof,c,timer,*,b,result,progress=None):
    """Exactly99; startup pass/fail cannot select a shorter/longer lifecycle."""
    controls,rows,samples,counters = [],[],[],b.new_counters()
    progress = {} if progress is None else progress
    progress.update(controls=controls,rows=rows,samples=samples,counters=counters)
    result['counters']=counters
    metadata = b.trace_metadata(session)
    progress['metadata']=metadata
    row,boundary = b.capture(session)
    rows.append(row); controls.append(boundary)
    counters['completed_rows']=1
    observer = b.LiveObserver(session,counters,samples,timely=lambda:timer.check('science'),
                              install_contact=factory.install_contact,synchronize=factory.synchronize)
    result['cpg_hashes']=[session._cpg_digest()]
    try:
        observer.install()
        for control in range(1,100):
            timer.check('science')
            counters['controls_requested'] += 1
            try:
                session.advance(1,capture=False)
            except Exception:
                counters['advance_unreturned_unknown']=True
                raise
            counters['controls_host_returned'] += 1
            factory.synchronize()
            counters['physics_device_confirmed']=counters['physics_host_returned']
            row,boundary=b.capture(session)
            rows.append(row); controls.append(boundary)
            actual_cpg=session._cpg_digest()
            result['cpg_hashes'].append(actual_cpg)
            require(session.control_tick==control and session.network.tick==20*control
                    and len(samples)==50*control and abs(session.body.physics_time()-.005*control)<=2e-8
                    and actual_cpg==proof['cpg_hash'],'actual99 feedback/clocks/CPG')
            counters['controls_completed'] += 1
            counters['completed_rows']=len(controls)
            counters['observer_samples']=len(samples)
            monitor_check(result['watch'].snapshot(),c['limits'])
        return controls,rows,samples,counters,metadata
    except Exception as error:
        fail(result,error,'neutral-prefix')
        raise
    finally:
        counters['completed_rows']=len(controls)
        counters['observer_samples']=len(samples)
        result['counters']=counters
        finish_owned(observer,None,result,timer)


def baseline(c,result,reg,timer):
    watch=session=factory=data=b=None
    construction={}
    progress={}
    controls=rows=samples=[]
    try:
        watch=gpu_watch(c)
        result['watch']=watch
        from tools import verify_c_banc_m3_common_descent_body as b
        data=source_data(reg,timer,b=b)
        factory=b.OriginalFactory()
        session=factory.construct(data,construction,lambda:timer.check('science'))
        proof=initial_neutral(session,data,timer,b=b,factory=factory)
        result['transfer']=proof
        put(c,'transfer-chunks.json',proof,timer)
        controls,rows,samples,counters,metadata=run_prefix(session,factory,data,proof,c,timer,b=b,result=result,progress=progress)
        verify_work(counters,construction)
        control_raw=b.control_arrays(controls,session,data)
        native=b.native_arrays(samples)
        require(b.validate_native(native)==4950,'actual4950 native observations')
        import numpy as np
        require(np.all(native['upright']>=.9) and not native['nonfoot_contact'].any()
                and np.isfinite(native['slip_mm_s'][native['slip_valid']]).all()
                and all(not row['fault'] for row in rows),'actual neutral native health/valid slip samples')
        startup=b.startup99(control_raw['command'],control_raw['tendon'],**data['startup'])
        result.update(run_valid=result['first_error'] is None,
                      classification='baseline_completed99' if result['first_error'] is None else 'technical_invalid',
                      reason='baseline_completed99',startup=startup,H359a='supported' if startup['passed'] else 'rejected',
                      baseline_completed99=True,counters=counters,construction=construction)
    except Exception as error:
        if result['first_error'] is None:
            fail(result,error,'baseline-startup99')
    finally:
        result.pop('watch',None)
        result['construction']=construction
        if session and factory:
            try:
                factory.synchronize()
            except Exception as error:
                fail(result,error,'final-neutral-confirmation')
        finish_owned(None,session,result,timer)
        if watch:
            try:
                result['sampler']=watch.close()
                monitor_check(result['sampler'],c['limits'],closed=True)
            except Exception as error:
                fail(result,error,'neutral-NVML-close')
        if b:
            try:
                record_imports(result,c,b.actual_imports())
            except Exception as error:
                fail(result,error,'neutral-actual-imports')
    if data and controls and result.get('baseline_completed99'):
        try:
            timer.check()
            result['cpg_unchanged']=bool(session._cpg_digest()==result['transfer']['cpg_hash'])
            require(result['cpg_unchanged'],'CPG preserved through owned close')
            for entry in data['ledger']:
                timer.check()
                require(b.e.digest(safe(entry['path']),timer.check)==entry['expected_sha256'],'neutral source posthash')
            publish_baseline(c,result,reg,timer,control_raw,native,b.lean_trace(rows,metadata),data['startup'],data['ledger'])
        except Exception as error:
            fail(result,error,'baseline-publication')
    if result['first_error'] is not None:
        result['H359a']='not_qualified'
        publish_partial(c,result,progress,timer,b=b)


def publish_baseline(c,result,reg,timer,control,native,trace,target,ledger,*,root=ROOT):
    """Production publisher, also exercised with wholly generated arrays."""
    require(result.get('baseline_completed99') is True and result['first_error'] is None
            and result['secondary_errors']==[],'only clean completed99 publication')
    for name,value in (('control-raw.npz',control),('native-contact.npz',native),('startup-inputs.npz',target)):
        put(c,name,npz_bytes(value,cap=1<<20 if name=='startup-inputs.npz' else 8<<20),timer,raw=True,root=root)
    put(c,'trace.json',trace,timer,root=root)
    transfer_path=c['paths']['output']+'/transfer-chunks.json'
    if safe(transfer_path,root).exists():
        typed_equal(strict_json(safe(transfer_path,root).read_bytes()),normalize(result['transfer']),'same early transfer publication')
    else:
        put(c,'transfer-chunks.json',result['transfer'],timer,root=root)
    put(c,'selected-ledger.json',{'inputs':normalize(ledger),'source_input_roster':reg['source_input_roster']},timer,root=root)
    put(c,'baseline-report.json',result,timer,root=root)
    members=('control-raw.npz','native-contact.npz','trace.json','startup-inputs.npz',
             'transfer-chunks.json','selected-ledger.json','baseline-report.json')
    files={v:file_spec(c['paths']['output']+'/'+v,root) for v in members}
    result['baseline_manifest']=put(c,'baseline-manifest.json',{'schema':'flylab.m3.e359.baseline-manifest.v1',
                                    'files':files,'all_fsynced_and_closed':True},timer,root=root)
    return result['baseline_manifest']


def publish_partial(c,result,progress,timer,*,b,root=ROOT):
    """Optional bounded diagnosis. Never writes a completed baseline manifest."""
    if not progress or b is None:
        return
    result['partial_progress']={'controls':len(progress['controls']),'rows':len(progress['rows']),
                                'native_samples':len(progress['samples']),
                                'counters':normalize(progress['counters'])}
    try:
        timer.check()
        require(result['first_error'] is not None,'partial diagnosis requires failure')
        import numpy as np
        controls=progress['controls']
        if controls:
            arrays={key:np.stack([v[key] for v in controls]) for key in ('command','tendon')}
            put(c,'partial-control.npz',npz_bytes(arrays,cap=8<<20),timer,raw=True,root=root)
        if progress['samples']:
            put(c,'partial-native.npz',npz_bytes(b.native_arrays(progress['samples']),cap=8<<20),timer,raw=True,root=root)
        if progress['rows'] and 'metadata' in progress:
            put(c,'partial-trace.json',b.lean_trace(progress['rows'],progress['metadata']),timer,root=root)
        put(c,'partial-progress.json',{'schema':'flylab.m3.e359.partial-diagnostic.v1',
             'baseline_completed99':False,'qualified':False,'first_error':result['first_error'],
             'secondary_errors':result['secondary_errors'],'progress':result['partial_progress']},timer,root=root)
    except Exception as error:
        fail(result,error,'partial-diagnostic-publication')


def control_fields(selected_count):
    return {'control_tick':((100,),'int64',True),'network_tick':((100,),'int64',True),
            'physics_time_s':((100,),'float64',True),'command':((100,42),'float64',True),
            'tendon':((100,6),'float64',True),'theta':((8547,),'float64',True),
            'selected_weights':((selected_count,),'float32',True)}


def native_fields():
    return {'time_s':((4950,),'float64',True),'normal_load_bw':((4950,6),'float64',True),
            'slip_mm_s':((4950,6),'float64',False),'slip_valid':((4950,6),'bool',True),
            'nonfoot_contact':((4950,),'bool',True),'upright':((4950,),'float64',True)}


def validate_prefix_raw(control,native,trace,proof,report,*,b):
    import numpy as np
    verify_work(report['counters'],report['construction'])
    require(b.validate_native(native)==4950 and np.array_equal(control['control_tick'],np.arange(100))
            and np.array_equal(control['network_tick'],np.arange(100)*20)
            and np.allclose(control['physics_time_s'],np.arange(100)*.005,rtol=0,atol=2e-8), 'literal100/4950 prefix clocks')
    require(control['theta'].tobytes()==np.ones(8547,np.float64).tobytes()
            and b.array_sha(control['theta'])==proof['theta_sha256']
            and b.array_sha(control['selected_weights'])==proof['selected_sha256'], 'independent neutral theta/source selected')
    require(len(trace['samples'])==100 and trace['body_hash']==proof['body_hash']
            and trace['graph_hash']==proof['graph_hash'] and trace['model_hash']==proof['source_network_identity']
            and trace['adapter_hash']==proof['source_adapter_identity'], 'same source/world/decoder trace')
    for tick,row in enumerate(trace['samples']):
        b.lean_row(row,point_count=8)
        require(row['control_tick']==tick and abs(row['time_s']-.005*tick)<=2e-8 and not row['fault'], 'actual trace ticks/health')
    require(np.all(native['upright']>=.9) and not native['nonfoot_contact'].any()
            and np.isfinite(native['slip_mm_s'][native['slip_valid']]).all(), 'independent native health/valid slip samples')
    require(report['cpg_unchanged'] is True and report['cpg_hashes']==[proof['cpg_hash']]*100,
            'saved actual100 unchanged CPG guards')
    return True


def startup_readback(control,target,reported,*,b):
    actual=b.startup99(control['command'],control['tendon'],**target)
    typed_equal(normalize(actual),reported,'independent fixed startup99 recomputation')
    return actual


def load_literal_npz(spec,fields,timer,*,cap=8<<20,root=ROOT):
    """Exact immutable loader policy; root is explicit for generated fixtures."""
    import numpy as np

    from tools import verify_c_banc_m3_reference_parameter_direction as ref
    raw=read_spec(spec,root,text_only=False)
    require(len(raw)<=cap,'audit archive bound')
    with lib.zipfile.ZipFile(lib.io.BytesIO(raw)) as archive:
        infos=archive.infolist()
        require(len(infos)==len(fields) and len({v.filename for v in infos})==len(infos)
                and {v.filename for v in infos}=={k+'.npy' for k in fields},'literal audit NPZ roster')
        for info in infos:
            timer.check('science')
            shape,dtype,_=fields[info.filename[:-4]]
            require(not info.is_dir() and not info.flag_bits&1 and 0<info.file_size<=cap,'bounded unencrypted member')
            with archive.open(info) as stream:
                header=ref.native().strict_npy_header(stream)
            require(header['shape']==list(shape) and header['dtype']==np.dtype(dtype).str
                    and header['fortran_order'] is False and not np.dtype(dtype).hasobject
                    and info.file_size==header['header_bytes']+math.prod(shape)*np.dtype(dtype).itemsize,
                    'literal exact saved raw header')
        arrays={v.filename[:-4]:np.load(lib.io.BytesIO(archive.read(v)),allow_pickle=False) for v in infos}
    for key,value in arrays.items():
        require(not fields[key][2] or np.isfinite(value).all(),'finite saved raw '+key)
    return arrays


def validate_neutral_proof(proof,inventory,*,b):
    import numpy as np
    b.verify_chunk_coverage(proof)
    leaves=proof['leaves']
    expected_dtypes={'host.rowptr':'int32','host.columns':'int32','host.weights':'float32',
                     'packed.rowptr':'int32','packed.columns':'int32','packed.palette':'float32'}
    require(set(leaves)==set(expected_dtypes),'complete six neutral transfer leaves')
    for label,dtype in expected_dtypes.items():
        leaf=leaves[label]
        require(type(leaf['count']) is int and leaf['count']>0
                and np.dtype(leaf['dtype'])==np.dtype(dtype),'typed nonempty neutral transfer '+label)
        for chunk in leaf['chunks']:
            require(chunk['source_sha256']==chunk['expected_sha256']==chunk['actual_sha256'],
                    'neutral unchanged source chunk '+label)
    require(leaves['host.weights']['count']==leaves['host.columns']['count']==leaves['packed.columns']['count']
            and leaves['host.rowptr']['count']==leaves['packed.rowptr']['count']>1
            and leaves['packed.palette']['count']==proof['palette'],'linked neutral CSR/packed/palette counts')
    for key in ('graph_hash','source_network_identity','source_adapter_identity'):
        typed_equal(proof[key],inventory[key],'pinned inventory neutral '+key)
    require(proof['trial_schema']=='flylab.m3-ssp-efficacy-trial.v1'
            and proof['patch_schema']=='flylab.m3-ssp-efficacy-patch.v1'
            and proof['source_trial_schema']=='flylab.m3-ssprk3-recovery-resource-trial.v1',
            'actual neutral trial/provenance schema')
    for key in ('patch_identity','source_patch_identity','boundary_command_sha256','boundary_tendon_sha256'):
        require(type(proof[key]) is str and re.fullmatch('[0-9a-f]{64}',proof[key]),'neutral provenance '+key)
    require(proof['source_inverse_whole_exact'] is True
            and proof['source_boundary_command_and_tendon_exact'] is True and proof['neutral_source_CSR_equal'] is True
            and proof['candidate_network_identity']==proof['source_network_identity']
            and proof['candidate_CSR_data_sha256']==proof['source_CSR_data_sha256']
            and type(proof['palette']) is int and 0<proof['palette']<=16384
            and proof['pointer_version_guards_passed'] is True,'saved source/neutral exact transfer coverage')


def audit(c,result,reg,timer):
    from tools import verify_c_banc_m3_common_descent_body as b
    previous=closed_exit(c['inputs']['baseline_exit'])
    parent=strict_json(read_spec(previous['envelope']))
    worker=strict_json(read_spec(previous['contract']))
    require(worker['registration']==c['registration'] and worker['source_sha256']==c['source_sha256']
            and worker['adapter_qualification']==c['adapter_qualification'], 'same current baseline registration/Q/source')
    startup,control,target,proof,inventory=audit_saved_baseline(worker,parent,reg,timer,b=b)
    comparison={'status':'comparison_unavailable','reason':'optional comparison not requested',
                'historical_run_qualified':False,'historical_original_publication_binding_proven':False}
    if c['inputs']['compare_history']:
        comparison=historical_comparison(reg,control,target,proof,timer,b=b,inventory=inventory)
    record_imports(result,c,b.actual_imports())
    result.update(run_valid=True,classification='baseline_audited99',baseline_exit=c['inputs']['baseline_exit'],
                  baseline_completed99=True,startup=startup,H359a='supported' if startup['passed'] else 'rejected',
                  comparison=comparison,saved_transfer_digest_audit=True,independent_GPU_reacquisition=False,
                  independent_body_reconstruction=False,full_prefix_evaluator_called=False)


def audit_saved_baseline(worker,parent,reg,timer,*,b,root=ROOT):
    """Actual raw-only audit core. Production calls it after closed_exit authority."""
    manifest=strict_json(read_spec(parent['baseline_manifest'],root))
    members={'control-raw.npz','native-contact.npz','trace.json','startup-inputs.npz',
             'transfer-chunks.json','selected-ledger.json','baseline-report.json'}
    require(set(manifest)=={'schema','files','all_fsynced_and_closed'}
            and manifest['schema']=='flylab.m3.e359.baseline-manifest.v1'
            and set(manifest['files'])==members and manifest['all_fsynced_and_closed'] is True,
            'literal closed new baseline manifest')
    files=manifest['files']
    for name,spec in files.items():
        require(spec['path']==worker['paths']['output']+'/'+name,'literal bound baseline member path')
        timer.check('science')
        read_spec(spec,root,text_only=not name.endswith('.npz'))
    report=strict_json(read_spec(files['baseline-report.json'],root))
    require(report['run_valid'] is True and report['reason']=='baseline_completed99'
            and report['baseline_completed99'] is True and report['first_error'] is None
            and report['secondary_errors']==[] and report['transfer']==parent['transfer'], 'clean baseline99 producer')
    proof=strict_json(read_spec(files['transfer-chunks.json'],root))
    require(proof==report['transfer'],'same published neutral transfer')
    inventory=strict_json(read_spec(reg['body_inventory'],root))
    validate_neutral_proof(proof,inventory,b=b)
    selected_count=inventory['selected_archives']['recurrent_mask']['selected_schema']['positions']['shape'][0]
    control=load_literal_npz(files['control-raw.npz'],control_fields(selected_count),timer,root=root)
    native=load_literal_npz(files['native-contact.npz'],native_fields(),timer,root=root)
    schema=inventory['selected_archives']['startup']['selected_schema']
    target=load_literal_npz(files['startup-inputs.npz'],{k:(tuple(v['shape']),v['dtype'],True) for k,v in schema.items()},timer,cap=1<<20,root=root)
    trace=strict_json(read_spec(files['trace.json'],root))
    validate_prefix_raw(control,native,trace,proof,report,b=b)
    require(b.array_sha(control['command'][0])==proof['boundary_command_sha256']
            and b.array_sha(control['tendon'][0])==proof['boundary_tendon_sha256'],'saved exact source boundary0 digest')
    ledger=strict_json(read_spec(files['selected-ledger.json'],root))
    typed_equal(ledger['source_input_roster'],reg['source_input_roster'],'same finite133 audited source')
    for name,spec in reg['source_input_roster'].items():
        records=[v for v in ledger['inputs'] if v['path']==name and 'selected' not in v]
        require(len(records)==1 and records[0]['expected_sha256']==spec['sha256']
                and records[0]['bytes']==spec['bytes'],'complete source133 producer ledger')
    for role in ('startup','recurrent_mask'):
        records=[v for v in ledger['inputs'] if v['path']==inventory['selected_archives'][role]['path'] and 'selected' in v]
        require(len(records)==1,'literal selected producer role')
        if role=='startup':
            for key,value in target.items():
                require(b.array_sha(value)==records[0]['selected'][key]['c_order_sha256'],'saved exact startup target')
        else:
            require(records[0]['selected']['positions']['c_order_sha256']==proof['positions_sha256']
                    and records[0]['selected']['edge_parameters']['c_order_sha256']==proof['groups_sha256'],'same exact selected index/group')
    startup=startup_readback(control,target,report['startup'],b=b)
    return startup,control,target,proof,inventory


def paired_errors(baseline,historical,target):
    import numpy as np

    from tools import verify_c_banc_m3_common_descent_body as body_helpers
    output={}
    for label,member,want,threshold in (('q','command','target_command',.04),('tendon','tendon','target_tendon',.025)):
        a,z=baseline[member][1:100],historical[member][1:100]
        rows={}
        for name,value in (('baseline_error',a-target[want]),('historical_error',z-target[want]),('prefix_difference',z-a)):
            absolute=np.abs(value)
            location=np.unravel_index(int(np.argmax(absolute)),absolute.shape)
            rows[name]={'rms':float(np.linalg.norm(value)/math.sqrt(value.size)), 'max':float(absolute.max()),
                        'max_location':{'control_tick':int(location[0])+1,'component':int(location[1])},
                        'threshold_exceed_locations':[{'control_tick':int(x)+1,'component':int(y)}
                                                      for x,y in np.argwhere(absolute>threshold)]}
            rows[name]['per_component']=[
                {'component':i,'leg':str(body_helpers.LEGS[i//7] if label=='q' else body_helpers.LEGS[i]),
                 'joint_index':i%7 if label=='q' else None,
                 'rms':float(np.linalg.norm(value[:,i])/math.sqrt(99)),
                 'max':float(absolute[:,i].max()),'max_control_tick':int(np.argmax(absolute[:,i]))+1}
                for i in range(value.shape[1])]
        output[label]=rows
    return output


def historical_identity(old,old_proof,old_trace,proof,inventory,*,b):
    """Necessary pairing evidence, never a historical qualification gate."""
    verify_work(old['counters'],old['construction'])
    typed_equal(old['first_error'],{'type':'ValueError','stage':'body-manifest-publication',
                'message':'actual body imported source closure','notes':[]},'retained historical first publication error')
    require(old['cpg_unchanged'] is True,'historical CPG unchanged through known prefix')
    b.verify_chunk_coverage(old_proof)
    require(old['transfer']==old_proof,'historical envelope transfer linkage')
    for key in ('source_network_identity','source_adapter_identity','graph_hash'):
        typed_equal(old_proof[key],inventory[key],'historical pinned source '+key)
    for key in ('source_network_identity','source_adapter_identity','graph_hash','cpg_hash','source_CSR_data_sha256'):
        typed_equal(old_proof[key],proof[key],'paired exact source '+key)
    for key,value in (('body_hash',proof['body_hash']),('graph_hash',old_proof['graph_hash']),
                      ('model_hash',old_proof['candidate_network_identity']),
                      ('adapter_hash',old_proof['source_adapter_identity'])):
        typed_equal(old_trace[key],value,'paired historical world/model/decoder '+key)
    require(len(old_trace['samples'])==100,'paired known prefix trace100')
    return True


def historical_comparison(reg,control,target,proof,timer,*,b,inventory):
    result={'status':'comparison_unavailable','reason':None,'diagnostic_only':True,
            'historical_run_qualified':False,'historical_original_publication_binding_proven':False,
            'new_candidate_evaluated':False,'numeric_alpha_search_performed':False,'historical_manifest_missing':True,
            'historical_actual_import_evidence_missing':True}
    try:
        # A voluntary skip occurs before any old numeric read. Real late timeout
        # still reaches the outer audit's final deadline and revokes its result.
        if timer.entry+timer.limits['science_s']-timer.clock()<2:
            result['reason']='insufficient reserved audit margin before historical reads'
            return result
        policy=reg['historical_diagnostic']
        metadata={Path(name).name:strict_json(read_spec(spec)) for name,spec in policy['metadata_files'].items()
                  if name.endswith('.json')}
        for name,spec in policy['metadata_files'].items():
            timer.check('science'); read_spec(spec)
        old=metadata['envelope.json']; old_exit=metadata['parent-e346-e358-candidate1-body-full-exit.json']
        old_contract=metadata['parent-e346-e358-candidate1-body.json']
        require(old_exit['envelope']==policy['metadata_files'][BASE+'/e358-candidate1-body/envelope.json']
                and old_exit['contract']==policy['metadata_files'][BASE+'/parent-e346-e358-candidate1-body.json'],
                'literal historical exit/contract/envelope diagnostic linkage')
        for name,snapshot in (('tools/verify_c_banc_m3_candidate_body_handoff.py','parent-e358-as-executed-adapter.py'),
                              ('tests/test_c_banc_candidate_body_handoff.py','parent-e358-as-executed-tests.py')):
            require(old_contract['source_sha256'][name]==policy['metadata_files'][BASE+'/'+snapshot]['sha256'],
                    'historical executed source snapshot, no current validity repair')
        result['historical_first_error']=old['first_error']; result['historical_secondary_errors']=old['secondary_errors']
        result['historical_exit_code']=old_exit['exit_code']
        require(old_exit['qualified'] is False and old_exit['exit_code']==1 and old['run_valid'] is False
                and old['classification']=='technical_invalid' and 'actual_import_sha256' not in old
                and not safe(BASE+'/e358-candidate1-body/body-manifest.json').exists(),'preserved failed historical publication')
        old_proof=metadata['transfer-chunks.json']; old_ledger=metadata['selected-ledger.json']; old_trace=metadata['trace.json']
        historical_identity(old,old_proof,old_trace,proof,inventory,b=b)
        for name,spec in reg['source_input_roster'].items():
            rows=[v for v in old_ledger['inputs'] if v['path']==name and 'selected' not in v]
            require(len(rows)==1 and rows[0]['expected_sha256']==spec['sha256'] and rows[0]['bytes']==spec['bytes'],
                    'paired whole/graph/mask/startup source133')
        for role,key,field in (('recurrent_mask','positions','positions_sha256'),('recurrent_mask','edge_parameters','groups_sha256')):
            rows=[v for v in old_ledger['inputs'] if v['path']==inventory['selected_archives'][role]['path'] and 'selected' in v]
            require(len(rows)==1 and rows[0]['selected'][key]['c_order_sha256']==proof[field],'paired selected index/group')
        raw_specs={Path(name).name:file_spec(name) for name in policy['raw_paths']}
        require(all(v['bytes']<=policy['per_raw_cap_bytes'] for v in raw_specs.values()),'finite historical raw bounds')
        result['current_historical_byte_descriptors']=raw_specs
        old_control=load_literal_npz(raw_specs['control-raw.npz'],control_fields(control['selected_weights'].size),timer)
        old_native=load_literal_npz(raw_specs['native-contact.npz'],native_fields(),timer)
        fields={role+'.'+k:(tuple(v['shape']),v['dtype'],True) for role in ('reference','startup')
                for k,v in inventory['selected_archives'][role]['selected_schema'].items()}
        evaluation=load_literal_npz(raw_specs['evaluation-inputs.npz'],fields,timer)
        import numpy as np
        require(b.validate_native(old_native)==4950 and np.array_equal(old_control['control_tick'],control['control_tick'])
                and np.array_equal(old_control['network_tick'],control['network_tick'])
                and np.array_equal(old_control['physics_time_s'],control['physics_time_s']), 'paired exact boundary/native clocks')
        require(np.isfinite(old_native['slip_mm_s'][old_native['slip_valid']]).all(),'historical valid-mask finite slip policy')
        for key in ('command','tendon'):
            require(b.e.exact(old_control[key][0],control[key][0]),'paired identical source '+key+' boundary0')
        for key,value in target.items():
            require(b.e.exact(evaluation['startup.'+key],value),'paired identical startup target bytes')
        require(b.array_sha(old_control['theta'])==old_proof['theta_sha256']
                and b.array_sha(old_control['selected_weights'])==old_proof['selected_sha256']
                and old_ledger['candidate']==old_proof['candidate'],'historical current theta/selected linkage')
        for tick,row in enumerate(old_trace['samples']):
            b.lean_row(row,point_count=8)
            require(row['control_tick']==tick and row['time_s']==control['physics_time_s'][tick],'paired trace index/time')
        timer.check('science')
        result.update(status='diagnostic_only',reason=None,paired=paired_errors(control,old_control,target),
                      historical_startup_recomputed=b.startup99(old_control['command'],old_control['tendon'],**target))
    except Exception as error:
        result['reason']=type(error).__name__+': '+str(error)[:1024]
    return result


def test_functions(root=ROOT):
    tree = ast.parse(safe(OWNED[1], root).read_text("utf8"))
    functions = [v.name for v in tree.body if isinstance(v, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and v.name.startswith("test_")]
    require(functions and len(functions) == len(set(functions)), "complete finite test function roster")
    return sorted(functions)


def validate_case_receipt(receipt, *, root=ROOT):
    require(receipt["schema"] == "flylab.m3.e359.complete-pytest-receipt.v1"
            and receipt["source_sha256"] == own_sources(root)
            and receipt["expected_functions"] == test_functions(root)
            and receipt["collected_functions"] == receipt["expected_functions"]
            and type(receipt['exit_code']) is int and receipt["exit_code"] == 0 and receipt["collection_errors"] == []
            and type(receipt['original_calls']) is int and type(receipt['original_array_reads']) is int
            and receipt["original_calls"] == receipt["original_array_reads"] == 0
            and receipt["complete_current_cases"] is True, "all actual current test cases")
    require(set(receipt["phases"]) == set(receipt["expected_functions"]), "no hidden test filtering")
    for function, phases in receipt["phases"].items():
        require(set(phases) == {"setup", "call", "teardown"}
                and all(phases[k] == "passed" for k in phases), "every function setup/call/teardown passed: " + function)
    return True


def execute_current_pytest(case_name):
    raw = safe(case_name).read_bytes()
    case = strict_json(raw)
    require(set(case) == {"schema", "source_sha256", "expected_functions", "receipt", "pytest_ini", "fixture_exit"}
            and case["schema"] == "flylab.m3.e359.current-pytest-case.v1"
            and case["source_sha256"] == own_sources() and case["expected_functions"] == test_functions(),
            "controlled whole current pytest contract")
    fixture=closed_exit(case['fixture_exit'])
    require(fixture['role']=='fixture' and fixture['mode']=='generated','actual current new E359 fixture before suite')
    os.environ['E359_CURRENT_FIXTURE_EXIT']=json.dumps(case['fixture_exit'],allow_nan=False)
    ini = safe(case["pytest_ini"])
    require(ini.read_bytes().strip() == b"[pytest]", "empty owned pytest config")
    receipt_path = safe(case["receipt"])
    require(receipt_path.parent == safe(BASE) and receipt_path.name.startswith("parent-e359-")
            and not receipt_path.exists(), "fresh E359 case receipt")
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        os.environ.pop(key, None)
    import pytest
    observed = {"collected": [], "phases": {}, "collection_errors": []}
    class ReceiptPlugin:
        def pytest_collection_finish(self, session):
            for item in session.items:
                require(item.path.resolve() == safe(OWNED[1]).resolve(), "only current owned test file")
                require("[" not in item.name, "explicit unparameterized finite test functions")
                observed["collected"].append(item.name)
            require(sorted(observed["collected"]) == case["expected_functions"], "complete collection, no filter")
        def pytest_collectreport(self, report):
            if report.failed:
                observed["collection_errors"].append(str(report.longrepr)[:2048])
        def pytest_runtest_logreport(self, report):
            function = report.nodeid.split("::")[-1]
            phases = observed["phases"].setdefault(function, {})
            require(report.when not in phases, "one actual test phase")
            phases[report.when] = report.outcome
    code = int(pytest.main(["-q", "-p", "no:cacheprovider", "-c", case["pytest_ini"], OWNED[1]], plugins=[ReceiptPlugin()]))
    value = {"schema": "flylab.m3.e359.complete-pytest-receipt.v1", "source_sha256": own_sources(),
             "case_contract": descriptor(case_name, raw), "expected_functions": case["expected_functions"],
             "collected_functions": sorted(observed["collected"]), "phases": observed["phases"],
             "collection_errors": observed["collection_errors"], "exit_code": code,
             "original_calls": 0, "original_array_reads": 0, "complete_current_cases": code == 0}
    try:
        validate_case_receipt(value)
    except Exception:
        value["complete_current_cases"] = False
        code = code or 2
    durable(receipt_path, json_bytes(value), cap=256 << 10)
    return code


def child(name,receipt_name):
    c,spec,receipt_spec,reg,timer=pre_array(name,receipt_name)
    result=envelope(c,spec,receipt_spec)
    try:
        if c['mode']=='generated':
            if c['role']=='suite':
                case=c['inputs']['case_contract']
                require(execute_current_pytest(case['path'])==0,'actual complete E359 pytest passed')
                result['case_receipt']=file_spec(strict_json(read_spec(case))['receipt'])
            else:
                put(c,'fixture.json',{'value':[1,2,3],'original_array_reads':0},timer)
            result.update(run_valid=True,classification='generated_boundary_only',original_array_reads=0)
        elif c['role']=='baseline_startup99':
            baseline(c,result,reg,timer)
        elif c['role']=='baseline_audit':
            audit(c,result,reg,timer)
        timer.check()
        require(retained(c)+c['retained_reserve_bytes']<=c['limits']['output_cap_bytes'],'final retained reserve')
    except Exception as error:
        fail(result,error,'child-final')
    try:
        result['elapsed_before_publication_s']=time.perf_counter()-timer.entry
        put(c,'envelope.json',result,timer,emergency=not result['run_valid'])
        timer.check()
    except Exception as error:
        fail(result,error,'late-envelope-publication')
    if result['first_error'] is not None:
        put(c,'revocation.json',result,timer,emergency=True)
    print(json.dumps({'role':c['role'],'run_valid':result['run_valid'],'classification':result['classification'],
                      'actual_full_return_pending':True,**FALSE_MILESTONES},allow_nan=False),flush=True)
    return 0 if result['run_valid'] else 1


def main(argv=None):
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--supervise'); group.add_argument('--child'); group.add_argument('--execute-current-pytest')
    parser.add_argument('--pre-spawn')
    args=parser.parse_args(argv)
    if args.supervise:
        require(args.pre_spawn is None,'parent owns actual pre-spawn receipt')
        return supervise(args.supervise)
    if args.child:
        require(args.pre_spawn is not None,'child requires actual pre-spawn receipt')
        return child(args.child,args.pre_spawn)
    require(args.pre_spawn is None,'controlled whole-test entrypoint')
    return execute_current_pytest(args.execute_current_pytest)


if __name__=='__main__':
    raise SystemExit(main())
