"""K paid Generated qualification. Temporary ledgers and fees are synthetic."""
import copy
import json
import os

from tests import banc_m3_checkpoint_fixture as f
from tools import banc_m3_checkpoint_runtime as k
from tools import banc_m3_conserved_resource as r


def test_math_k_checkpoint_ordered_accounting_negative_remaining(tmp_path):
    fixture = f.case(tmp_path, "generated-k-negative-accounting")
    fixture.admit()
    fixture.python_charge(600.0, 124, suspended=True)
    state = k.validate(fixture.ledger, fixture.root)
    assert fixture.ledger["generated_shared_remaining_s"] < 0
    assert fixture.ledger["direct_science_generated_used_s"] == 1636.744
    assert fixture.ledger["debits"][-1]["actual_outer_full_return_s"] == 600.0
    assert fixture.ledger["direct_science_blocking_holds"] == {k.BOOTSTRAP_CALL: k.HOLD}
    f.negative(lambda: k.eligibility(fixture.ledger, state))
    _a, policy, grant = k.anchor(fixture.root)
    totals = k.totals(fixture.ledger, policy, grant, k.OLD_CAPS)
    for key, value in totals.items():
        assert fixture.ledger[key] == value and type(fixture.ledger[key]) is float
    assert totals["original_remaining_s"] == 15600.0 - grant["original_cumulative_before_s"] - sum(
        d["actual_outer_full_return_s"] for d in fixture.ledger["debits"][165:] if d["mode"] == "original")


def test_math_k_checkpoint_exact_prefix_types_and_legal_states(tmp_path):
    fixture = f.case(tmp_path, "generated-k-types")
    baseline = copy.deepcopy(fixture.ledger)
    assert k.validate(baseline, fixture.root)["phase"] == "charged"
    for key, value in (("original_jobs_running", True), ("direct_science_generated_limit_s", 1964),
                       ("original_remaining_s", 0.0), ("direct_science_policy", None), ("extra", 0)):
        f.negative(lambda key=key, value=value: k.validate(f.mutation(baseline, key, value), fixture.root))
    wrong = copy.deepcopy(baseline)
    wrong["debits"][0]["actual_outer_full_return_s"] += .001
    f.negative(lambda: k.validate(wrong, fixture.root))
    assert fixture.admit()["phase"] == "admitted"
    call = k.BOOTSTRAP_CALL
    token = {"schema": "flylab.m3.direct-science.lock-owner.v1", "call_id": call,
             "request": fixture.request_spec, "owner_pid": 411, "owner_creation_ticks": 412,
             "entry_counter_s": 1.0, "job_name": "Local\\generated-synthetic-K"}
    token_spec = fixture.put(r.SHARED_LOCK, token)
    identity = {"owner_pid": 411, "owner_creation_ticks": 412, "worker_pid": 413, "worker_creation_ticks": 414}
    fixture.ledger["direct_science_workers"] = {call: {"request": fixture.request_spec, "state": "begun", "lock_token": token_spec, **identity}}
    assert k.validate(fixture.ledger, fixture.root)["phase"] == "begun"
    fixture.ledger["direct_science_blocking_holds"] = {call: {"request": fixture.request_spec, "state": "ownership_unknown", **identity}}
    assert k.validate(fixture.ledger, fixture.root)["phase"] == "unknown"
    f.negative(lambda: k.eligibility(fixture.ledger, k.validate(fixture.ledger, fixture.root)))
    fixture.ledger["direct_science_workers"] = {}
    fixture.ledger["direct_science_blocking_holds"] = {}
    fixture.ledger["direct_science_reservations"] = {}
    fixture.ledger["direct_science_pending_actuals"] = {call: {"request": fixture.request_spec, "state": "actual_pending", "ownership_closed": True}}
    assert k.validate(fixture.ledger, fixture.root)["phase"] == "closed-pending"
    wrong = copy.deepcopy(fixture.ledger)
    wrong["direct_science_pending_actuals"][call]["ownership_closed"] = 1
    f.negative(lambda: k.validate(wrong, fixture.root))
    duplicate = b'{"a":1,"a":2}'
    from tools.verify_c_banc_m3_startup_ray import json_load
    f.negative(lambda: json_load(duplicate))
    f.negative(lambda: json_load(b'{"fee":NaN}'))


def test_gate_k_checkpoint_production_python_native_parity(tmp_path):
    fixture = f.case(tmp_path, "generated-k-parity")
    fixture.admit()
    expected = k.validate(fixture.ledger, fixture.root)
    lock_value = {'schema': 'flylab.m3.direct-science.lock-owner.v1', 'fixture_synthetic': True}
    lock_spec = fixture.put(r.SHARED_LOCK, lock_value)
    assert k.data(lock_spec, fixture.root, 16384) == lock_value
    foreign_lock = fixture.put(k.BASE + '/generated-k-foreign-metadata.lock', lock_value)
    f.negative(lambda: k.data(foreign_lock, fixture.root, 16384))
    lock_read = fixture.native("$value=K-Read (Pin '" + r.SHARED_LOCK + "') 16384;Same $value.schema 'flylab.m3.direct-science.lock-owner.v1' 'Existing literal ownership JSON';$rejected=$false;try{$null=K-Read (Pin '" + foreign_lock['path'] + "') 16384}catch{$rejected=$true};Need $rejected 'Foreign lock metadata refused before read';Write-Output 'PASS synthetic existing ownership token reader'")
    f.observed_pass(lock_read)
    result = fixture.native("$l=K-Read (Pin $ledgerName) 524288;$s=K-Validate $l;$s|ConvertTo-Json -Depth 10 -Compress")
    f.observed_pass(result)
    actual = json.loads(result.stdout)
    assert actual == expected
    from unittest.mock import patch
    with patch.object(r, "j_history", side_effect=AssertionError("legacy recursion")), patch.object(r, "j_lint_cleanup", side_effect=AssertionError("legacy recursion")):
        assert r.current_caps(fixture.ledger, fixture.root, lambda *_: (_ for _ in ()).throw(AssertionError("legacy fallback"))) == k.OLD_CAPS
    native = fixture.native("function Load-CheckpointLegacy {throw 'forbidden K fee loader'};function Conserved-J-History {throw 'forbidden K recursion'};$l=K-Read (Pin $ledgerName) 524288;$s=K-Validate $l;Same $s.caps (K-OldCaps) 'K parity';Write-Output 'PASS synthetic K production validator'")
    f.observed_pass(native)
    original = f.case(tmp_path, 'generated-k-original-metadata')
    assert original.original_evidence()[3] is True
    bad_science, bad_cost = [], []
    for index, (key, value) in enumerate((('schema', 'wrong'), ('mode', 'generated'), ('role', 'contact_body_audit'),
            ('call_id', k.CALLS[0] + '-alias'), ('output', original.science['output'] + '-alias'),
            ('inputs', original.science['contract']), ('grant', original.science['contract']),
            ('roster', original.science['contract']), ('local_debit_offset', True), ('resource_increment_s', 1),
            ('historical_manifest', original.science['contract']), ('source_freeze', k.J_FREEZE))):
        wrong = f.mutation(original.science, key, value)
        f.negative(lambda wrong=wrong: k.original_metadata(wrong, original.q, original.root))
        bad_science.append(original.put(k.BASE + '/generated-k-science-negative-' + str(index) + '.json', wrong))
    wrong = copy.deepcopy(original.science)
    wrong['limits']['internal_s'] += 1
    f.negative(lambda: k.original_metadata(wrong, original.q, original.root))
    bad_science.append(original.put(k.BASE + '/generated-k-science-internal-negative.json', wrong))
    trace, outer = k.data(original.trace_spec, original.root), k.data(original.outer_spec, original.root)
    for index, key in enumerate(('request', 'supervisor_request', 'role', 'grant', 'inputs', 'roster', 'contract', 'body_retry', 'local_debit_offset', 'resource_increment_s')):
        wrong = f.mutation(original.cost, key, 164 if key == 'local_debit_offset' else (1 if key == 'resource_increment_s' else 'tampered'))
        f.negative(lambda wrong=wrong: k.fee_values(wrong, original.q, original.request_spec, trace, outer, original.root))
        pin = original.put(original.cost_spec['path'], wrong)
        debit = dict(original.debit, descriptor=pin)
        f.negative(lambda pin=pin, debit=debit: k.fee_proof(pin, debit, original.source, original.root))
        bad_cost.append(original.put(k.BASE + '/generated-k-cost-negative-' + str(index) + '.json', wrong))
    original.put(original.cost_spec['path'], original.cost)
    shell_trace, shell_outer = copy.deepcopy(trace), copy.deepcopy(outer)
    shell_trace.update(exit_code=1, exec_args={'cmd': '& .venv/Scripts/python.exe -m tools.banc_m3_startup_ray_supervisor --request ' + original.request_spec['path']})
    shell_outer.update(exit_code=124, worker_exit_code=124, timeout=True, first_error={'type': 'TimeoutError'})
    shell_cost = dict(original.cost, exit_code=1, native_supervisor_exit_code=124,
                      actual_collector_shell_exit_code=1, shell_deadline_status_pair_observed=True)
    k.fee_values(shell_cost, original.q, original.request_spec, shell_trace, shell_outer, original.root)
    shell_bad = []
    for key, value in (('native_supervisor_exit_code', 1), ('actual_collector_shell_exit_code', 124), ('shell_deadline_status_pair_observed', 1)):
        wrong = f.mutation(shell_cost, key, value)
        f.negative(lambda wrong=wrong: k.fee_values(wrong, original.q, original.request_spec, shell_trace, shell_outer, original.root))
        shell_bad.append(wrong)
    shell_pin = original.put(k.BASE + '/generated-k-shell-pair-values.json', {'trace': shell_trace, 'outer': shell_outer, 'valid': shell_cost, 'invalid': shell_bad})
    science_names = ','.join("'" + x['path'] + "'" for x in bad_science)
    cost_names = ','.join("'" + x['path'] + "'" for x in bad_cost)
    native = original.native("""
$q=K-Read (Pin 'REQUEST');$s=K-Read $q.science_request;$null=K-OriginalMetadata $s $q
$cost=K-Read (Pin 'COST');$d=@{cost_id=$q.call_id;mode='original';actual_outer_full_return_s=1.0;actual_returned_exit_code=[long]0;descriptor=(Pin 'COST')};$null=K-FeeProof $d.descriptor $d $q.source_freeze
$t=K-Read $cost.trace;$o=K-Read $cost.outer
foreach($name in @(SCIENCE)){$rejected=$false;try{$null=K-OriginalMetadata (K-Read (Pin $name)) $q}catch{$rejected=$true};Need $rejected 'Actual Native common science metadata negative'}
foreach($name in @(COSTS)){$rejected=$false;try{K-FeeValues (K-Read (Pin $name)) $q (Pin 'REQUEST') $t $o}catch{$rejected=$true};Need $rejected 'Actual Native admitted cost value negative'}
$pair=K-Read (Pin 'SHELL');Need (K-ExitMatch $pair.trace $pair.outer (Pin 'REQUEST')) 'Actual immutable shell deadline pair';K-FeeValues $pair.valid $q (Pin 'REQUEST') $pair.trace $pair.outer
foreach($value in $pair.invalid){$rejected=$false;try{K-FeeValues $value $q (Pin 'REQUEST') $pair.trace $pair.outer}catch{$rejected=$true};Need $rejected 'Exact Native observed shell-pair field'}
Write-Output 'PASS synthetic Original immutable metadata and cost value parity'
""".replace('REQUEST', original.request_spec['path']).replace('COSTS', cost_names).replace('COST', original.cost_spec['path']).replace('SCIENCE', science_names).replace('SHELL', shell_pin['path']))
    f.observed_pass(native)
    negatives = []
    for index, (key, value) in enumerate((("original_jobs_running", True), ("direct_science_generated_limit_s", 1964),
                                         ("startup_ray_original_grants", {}), ("extra", 0))):
        wrong = f.mutation(fixture.ledger, key, value)
        f.negative(lambda wrong=wrong: k.validate(wrong, fixture.root))
        negatives.append(fixture.put(k.BASE + "/generated-k-native-negative-" + str(index) + ".json", wrong))
    paths = ",".join("'" + pin['path'] + "'" for pin in negatives)
    native = fixture.native("foreach($name in @(PATHS)){$rejected=$false;try{$null=K-Validate (K-Read (Pin $name) 524288)}catch{$rejected=$true};Need $rejected 'Actual Native prefix/type/mixed caps/grant negative'};$rejected=$false;try{$null=Conserved-Json '{\"a\":1,\"a\":2}'}catch{$rejected=$true};Need $rejected 'Actual Native duplicate JSON';Write-Output 'PASS synthetic Native negative parity'".replace("PATHS", paths))
    f.observed_pass(native)


def test_gate_k_checkpoint_exact_recovery_allocation_and_negatives(tmp_path):
    fixture = f.case(tmp_path, "generated-k-deltas")
    fixture.full_source()
    from unittest.mock import patch
    bounded = fixture.put(k.BASE + '/generated-k-read-cap.py', 'synthetic Source text')
    reads = []
    real_raw = r.raw_pin
    def observed_raw(spec, root, cap, **kwargs):
        reads.append(cap)
        return real_raw(spec, root, cap, **kwargs)
    with patch.object(r, 'raw_pin', side_effect=observed_raw), patch.object(r, 'source_text', side_effect=AssertionError('old2MiB Source reader')):
        k.bounded_source(bounded, fixture.root, 65536)
    assert reads == [65536]
    f.negative(lambda: k.bounded_source(bounded, fixture.root, bounded['bytes'] - 1))
    relative_path = fixture.root / bounded['path']
    assert relative_path.read_bytes() == r.raw_pin(bounded, fixture.root, 65536)
    oversized = r.fixture_text_write(fixture.root, k.BASE + '/generated-k-oversized.py', b'x' * 65537)
    f.negative(lambda: k.bounded_source(oversized, fixture.root, 65536))
    foreign = dict(bounded, path=str(relative_path))
    f.negative(lambda: k.bounded_source(foreign, fixture.root, 65536))
    cap_result = fixture.native("$rejected=$false;try{$null=K-SourceRaw (Safe 'OVERSIZED') 65536}catch{$rejected=$true};Need $rejected 'Actual Source cap, not descriptor only';$raw=K-SourceRaw (Safe 'BOUNDED') 65536;Need ($raw.Length -lt 65536) 'Whole bounded text';Write-Output 'PASS synthetic actual Source read caps'".replace('OVERSIZED', oversized['path']).replace('BOUNDED', bounded['path']))
    f.observed_pass(cap_result)
    baseline, _p, _g = k.anchor(fixture.root)
    fixture.ledger = baseline
    (fixture.root / fixture.ledger_path).write_bytes(r.raw_pin(k.ANCHOR, fixture.root, 524288))
    result = fixture.native("$null=Start-LedgerTransaction 'generated-synthetic-K-recovery';try{$l=Conserved-Json $script:ledgerTransaction.before;$a=K-RecoveryAction $l (Pin 'RECOVERY') $script:ledgerTransaction.before;$null=K-Validate $a;$null=Save-Ledger $a;Write-Output 'PASS synthetic recovery'}finally{if($script:ledgerTransaction -and -not $script:ledgerTransaction.uncertain){Stop-LedgerTransaction $script:ledgerTransaction}}".replace("RECOVERY", k.RECOVERY_PATH))
    f.observed_pass(result)
    recovered = k.data(r.text_spec(fixture.ledger_path, fixture.root, 524288), fixture.root, 524288)
    assert recovered == k.recovered_snapshot(fixture.recovery, fixture.root)
    assert recovered["debits"] == baseline["debits"] and recovered["direct_science_calls"] == baseline["direct_science_calls"]
    assert set(recovered) - set(baseline) == {k.KEY}
    for key in set(baseline) - {"direct_science_blocking_holds"}:
        assert recovered[key] == baseline[key]
    fixture.ledger = recovered
    fixture.admit()
    debit = fixture.python_charge()
    rows = k.roster(fixture.freeze, fixture.root)["bootstrap"]
    xml = '<testsuites><testsuite tests="4" failures="0" errors="0" skipped="0">' + ''.join(
        '<testcase classname="' + x["classname"] + '" name="' + x["name"] + '" />' for x in rows) + '</testsuite></testsuites>'
    junit = r.fixture_text_write(fixture.root, k.BASE + "/parent-contact-learning-k-bootstrap-junit.xml", xml.encode("utf8"))
    before = fixture.put(k.BASE + "/generated-k-allocation-before.json", fixture.ledger)
    event = {"schema": "flylab.m3.checkpoint-allocation.v1", "epoch": k.EPOCH, "before_ledger": before, "recovery": fixture.recovery,
             "source_freeze": fixture.source, "bootstrap": debit["descriptor"], "bootstrap_junit": junit,
             "old_caps": k.OLD_CAPS, "new_caps": k.CAPS, "delta_s": {"generated_global": 400, "generated_local": 400, "original_global": -400},
             "combined_global_s": 17900, "new_resource_grant_s": 0, "refund_s": 0}
    fixture.put(k.ALLOCATION_PATH, event)
    result = fixture.native("$null=Start-LedgerTransaction 'generated-synthetic-K-allocation';try{$l=Conserved-Json $script:ledgerTransaction.before;$a=K-AllocationAction $l (Pin 'ALLOCATION') $script:ledgerTransaction.before;$null=Save-Ledger $a;Write-Output 'PASS synthetic allocation'}finally{if($script:ledgerTransaction -and -not $script:ledgerTransaction.uncertain){Stop-LedgerTransaction $script:ledgerTransaction}}".replace("ALLOCATION", k.ALLOCATION_PATH))
    f.observed_pass(result)
    after = k.data(r.text_spec(fixture.ledger_path, fixture.root, 524288), fixture.root, 524288)
    assert k.validate(after, fixture.root)["caps"] == k.CAPS
    changed = {key for key in fixture.ledger if fixture.ledger[key] != after[key]}
    assert changed == {"generated_shared_cumulative_limit_s", "direct_science_generated_limit_s", "original_cumulative_limit_s", "generated_shared_remaining_s", "original_remaining_s"}
    assert set(after) - set(fixture.ledger) == {k.ALLOCATION_KEY}
    assert after["debits"] == fixture.ledger["debits"]
    f.negative(lambda: k.validate(f.mutation(after, "original_cumulative_limit_s", 15600), fixture.root))


def test_gate_k_checkpoint_actual_atomic_foreign_lock_and_stale_raw(tmp_path):
    fixture = f.case(tmp_path, "generated-k-atomic")
    path = fixture.root / fixture.ledger_path
    before = path.read_bytes()
    lock_path = fixture.root / r.SHARED_LOCK
    token = {"schema": "flylab.m3.direct-science.lock-owner.v1", "owner_pid": os.getpid(), "fixture_synthetic": True}
    lock = r.SharedLock(lock_path, token)
    try:
        f.negative(lambda: r.SharedLock(lock_path, token))
        changed = copy.deepcopy(fixture.ledger)
        f.negative(lambda: r.atomic_ledger_publish(path, b"stale", changed, lock, validate=lambda value: k.validate(value, fixture.root)))
        assert path.read_bytes() == before
        raw = r.atomic_ledger_publish(path, before, changed, lock, validate=lambda value: k.validate(value, fixture.root))
        assert path.read_bytes() == raw and json.loads(raw) == changed
        assert lock_path.read_bytes() == lock.raw
    finally:
        lock.release(True)
    assert not lock_path.exists()
    result = fixture.native("$null=Start-LedgerTransaction 'generated-synthetic-K-atomic';try{$l=Conserved-Json $script:ledgerTransaction.before;$pin=Save-Ledger $l;Same (K-Read $pin 524288) $l 'Actual atomic replace/readback';Write-Output 'PASS synthetic Native atomic'}finally{if($script:ledgerTransaction -and -not $script:ledgerTransaction.uncertain){Stop-LedgerTransaction $script:ledgerTransaction}}")
    f.observed_pass(result)


def test_gate_k_checkpoint_fee_source_drift_overrun_unknown_duplicate(tmp_path):
    fixture = f.case(tmp_path, "generated-k-fee")
    fixture.admit()
    fixture.evidence(600.0, 124, suspended=True)
    # Current executable Source is absent/drifted in this generated root. The
    # immutable request/freeze/trace/outer remains intact and payable.
    f.negative(lambda: k.source_fence(fixture.source, fixture.root))
    result = fixture.native_commit()
    f.observed_pass(result)
    live = k.data(r.text_spec(fixture.ledger_path, fixture.root, 524288), fixture.root, 524288)
    assert live["debits"][-1]["actual_outer_full_return_s"] == 600.0 and live["generated_shared_remaining_s"] < 0
    assert live["direct_science_blocking_holds"] == {k.BOOTSTRAP_CALL: k.HOLD}
    assert k.validate(live, fixture.root)["failed"] is True
    duplicate = fixture.native_commit()
    assert duplicate.returncode != 0
    unchanged = k.data(r.text_spec(fixture.ledger_path, fixture.root, 524288), fixture.root, 524288)
    assert unchanged == live
    unknown = f.case(tmp_path, "generated-k-unknown")
    unknown.admit()
    unknown.unknown_owner()
    token_path = unknown.root / r.SHARED_LOCK
    token_raw = token_path.read_bytes()
    before_raw = (unknown.root / unknown.ledger_path).read_bytes()
    outer = unknown.evidence(600.0, 124, uncertain=True)
    assert k.closure(outer) is False
    illegal = []
    for index, (key, value) in enumerate((('all_owned_ended', True), ('reservation_released', True),
            ('worker_started', True), ('resume_outcome', 'RESUMED'), ('worker_pid', None), ('unknown_owned_identities', {}))):
        wrong = f.mutation(outer, key, value)
        f.negative(lambda wrong=wrong: k.closure(wrong))
        illegal.append(unknown.put(k.BASE + '/generated-k-uncertainty-negative-' + str(index) + '.json', wrong))
    names = ','.join("'" + x['path'] + "'" for x in illegal)
    result = unknown.native("foreach($name in @(NAMES)){$rejected=$false;try{$null=K-Closure (K-Read (Pin $name))}catch{$rejected=$true};Need $rejected 'Exact Native uncertainty layout'};Write-Output 'PASS synthetic uncertainty metadata parity'".replace('NAMES', names))
    f.observed_pass(result)
    blocked = unknown.native_commit()
    assert blocked.returncode != 0 and token_path.read_bytes() == token_raw
    assert (unknown.root / unknown.ledger_path).read_bytes() == before_raw
    assert not (unknown.root / (k.BASE + '/parent-' + k.BOOTSTRAP_CALL + '-full-return.json')).exists()
    unknown.python_charge(600.0, 124, uncertain=True)
    unknown_state = k.validate(unknown.ledger, unknown.root)
    hold = unknown.ledger['direct_science_blocking_holds'][k.BOOTSTRAP_CALL]
    assert hold['state'] == 'ownership_unknown' and hold['accounting_hold'] == k.HOLD
    assert unknown.ledger['direct_science_workers'] and unknown.ledger['direct_science_reservations']
    assert unknown.ledger['generated_shared_remaining_s'] < 0 and unknown_state['failed'] is True
    parity = unknown.native("function Load-CheckpointLegacy {throw 'forbidden unknown fee loader'};$l=K-Read (Pin $ledgerName) 524288;$s=K-Validate $l;$s|ConvertTo-Json -Depth 20 -Compress")
    f.observed_pass(parity)
    assert json.loads(parity.stdout) == unknown_state and token_path.read_bytes() == token_raw
    pending = f.case(tmp_path, 'generated-k-uncommitted-unknown')
    pending.admit()
    pending.evidence(600.0, 124, uncertain=True)
    pending_before = (pending.root / pending.ledger_path).read_bytes()
    receipt = pending.native_commit()
    f.observed_pass(receipt)
    observed = json.loads(receipt.stdout)
    assert observed['committed'] is False and observed['status'] == 'UNCOMMITTED_OWNERSHIP_UNKNOWN'
    assert observed['cost_published'] is False and observed['owner_and_reservation_retained'] is True
    assert (pending.root / pending.ledger_path).read_bytes() == pending_before
    assert not (pending.root / (k.BASE + '/parent-' + k.BOOTSTRAP_CALL + '-full-return.json')).exists()
    refused = f.case(tmp_path, "generated-k-no-create")
    refused.admit()
    outer = refused.evidence(1.0, 1, no_create=True)
    assert k.closure(outer) is True and outer["stdout_eof"] is False and outer["stderr_eof"] is False
    outer["stdout_eof"] = True
    f.negative(lambda: k.closure(outer))
    refused.evidence(1.0, 1, no_create=True)
    paid_refusal = refused.native_commit()
    f.observed_pass(paid_refusal)
    refused_live = k.data(r.text_spec(refused.ledger_path, refused.root, 524288), refused.root, 524288)
    assert refused_live['debits'][-1]['actual_outer_full_return_s'] == 1.0
    assert refused_live['direct_science_blocking_holds'] == {k.BOOTSTRAP_CALL: k.FAILURE_HOLD}
    assert k.validate(refused_live, refused.root)['failed'] is True
    wrong = copy.deepcopy(live)
    wrong["startup_ray_original_grants"] = {}
    f.negative(lambda: k.validate(wrong, fixture.root))
