"""Generated metadata-only production Native checks; no Original arrays or worker.

Batch Native cases in three hosts under the paid ROOT gate. Source pinning and
all action mutations occur only in explicit temporary roots.
"""
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "verification/banc-natural-gait-m3-20260919"
OWNER = BASE + "/parent-direct-science-account.ps1"
PREPARE = BASE + "/parent-contact-learning-prepare-native.ps1"
CONTRACT = BASE + "/parent-contact-learning-contract-native.ps1"
LEDGER = BASE + "/parent-e358-execution-ledger.json"
LITERAL = "앞으로 내 승인 받지 말고 스스로 판단해서 진행해줘. 개입할 상황이면 내가 알아서 개입할게"


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf8")
    path.write_bytes(raw)
    return {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def rebound(source, root, variable):
    anchor = f"${variable}=[IO.Path]::GetFullPath('C:/Users/ckthd/Dev/Projects/fly_lab')"
    assert source.count(anchor) == 1
    return source.replace(anchor, f"${variable}=[IO.Path]::GetFullPath('{root.as_posix().replace(chr(39), chr(39)*2)}')")


def run_batch(root, source):
    assert shutil.which("pwsh"), "Production Native fixture requires pwsh; no skip"
    script = root / "batch.ps1"
    script.write_text(source, encoding="utf8")
    result = subprocess.run([shutil.which("pwsh"), "-NoProfile", "-File", str(script)], capture_output=True, timeout=24, check=False)
    assert result.returncode == 0, result.stderr.decode("utf8", errors="replace")
    assert b"ALL_CASES_PASS" in result.stdout


def owner_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    closed = {key: {} for key in (
        "e360_reservations", "e360_pending_actuals", "e360_blocking_holds", "e359_reservations",
        "direct_science_workers", "direct_science_reservations", "direct_science_pending_actuals", "direct_science_blocking_holds",
        "startup_ray_original_workers", "startup_ray_original_reservations", "startup_ray_original_pending_actuals", "startup_ray_original_blocking_holds")}
    before = {**closed, "debits": [{"cost_id": f"historic-{i}", "mode": "original"} for i in range(118)],
              "original_jobs_running": 0, "original_reserved_s": 0, "original_cumulative_limit_s": 16200,
              "original_stage_cap_s": 6300, "startup_ray_original_local_limit_s": 6000,
              "original_cumulative_used_s": 11646.536960899506, "original_stage_used_s": 1752.7820000000002,
              "startup_ray_original_local_used_s": 1626.4210000000003,
              "generated_stage_used_s": 648.022, "generated_shared_cumulative_used_s": 728.2120156000001,
              "generated_shared_cumulative_limit_s": 1700,
              "generated_shared_remaining_s": 1700.0 - 80.1900156 - 648.022}
    prior = put(root, BASE + "/initial-before.json", before)
    policy = put(root, BASE + "/initial-policy.json", {"schema": "flylab.m3.direct-science.policy.v1", "before_ledger": prior,
                 "generated_local_limit_s": 300, "generated_shared_limit_s": 1700, "new_generated_grant_s": 0,
                 "original_authorized": False, "minimum_close_reserved_s": 20, "local_debit_offset": 118,
                 "generated_stage_before_s": 648.022, "generated_shared_before_s": 728.2120156000001,
                 "generated_shared_baseline_s": 80.1900156})
    live = copy.deepcopy(before)
    live.update(direct_science_policy=policy, direct_science_generated_limit_s=300, direct_science_generated_used_s=277.638,
                direct_science_calls={"historical-paid": {"state": "charged", "mode": "generated"}},
                generated_stage_used_s=648.022 + 277.638, generated_shared_cumulative_used_s=728.2120156000001 + 277.638,
                generated_shared_remaining_s=1700.0 - 80.1900156 - (648.022 + 277.638))
    live["debits"].append({"cost_id": "historical-paid", "mode": "generated", "actual_outer_full_return_s": 277.638})
    put(root, LEDGER, live)
    authority = put(root, BASE + "/authority.json", {"schema": "flylab.m3.contact-metadata.current-human-autonomy.v1",
                    "literal_latest_user": LITERAL, "direct_human_authority": True, "automatic_authority": True, "new_resource_grant_s": 0})
    anchor = put(root, BASE + "/transfer-before.json", live)
    event = put(root, BASE + "/transfer.json", {"schema": "flylab.m3.contact-metadata.generated-transfer.v1",
                "transfer_id": "contact-metadata-20261007-80", "transfer_s": 80, "from_local_limit_s": 300,
                "to_local_limit_s": 380, "new_resource_grant_s": 0, "initial_policy": policy,
                "direct_human_auto_authority": authority, "before_ledger": anchor})
    script = root / "owner.ps1"
    script.write_text(rebound((ROOT / OWNER).read_text(encoding="utf8"), root, "taskRoot"), encoding="utf8")
    return live, event, script


def test_native_archive_actual_metadata_and_boundary_drift(tmp_path):
    # The positive uses the actual plain archived8 metadata and production function.
    owner = (ROOT / OWNER).read_text(encoding="utf8").split("$ledger=Get-Content -LiteralPath (Safe $ledgerName)", 1)[0]
    owner = owner[owner.index("$ErrorActionPreference="):]
    contract = (ROOT / CONTRACT).read_text(encoding="utf8")
    script = owner + "\n" + contract + r"""
$roster=Read-Pin (Pin ($taskBase+'/parent-contact-conditioned-input-roster-20261007-a.json'))
$archive=Read-Pin $roster.geometry_source_archive;$old=Read-Pin $roster.geometry_source_freeze
Need (Contact-Archive $archive $old $roster.geometry_source_freeze) 'Actual archive positive'
"""
    scratch = tmp_path.as_posix().replace("'", "''")
    script += f"\n$taskRoot=[IO.Path]::GetFullPath('{scratch}')\n"
    archive_root = tmp_path / BASE / "parent-local-foot-geometry-as-executed-e"
    archive_root.mkdir(parents=True)
    items, old = [], {"sources": {}}
    for i in range(8):
        raw = f"# historic text {i}\n".encode()
        p = archive_root / f"source{i}.py"
        p.write_bytes(raw)
        descriptor = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        items.append({**descriptor, "original_path": f"source{i}.py", "archive_path": str(p)})
        old["sources"][f"source{i}.py"] = descriptor
    put(tmp_path, "archive.json", {"source_freeze": {"fixture": "historic"}, "sources": items})
    put(tmp_path, "old.json", old)
    script += r"""
$a=Read-Pin (Pin 'archive.json');$o=Read-Pin (Pin 'old.json')
Need (Contact-Archive $a $o $a.source_freeze) 'Scratch production positive'
$original=$a.sources[0].archive_path
foreach($path in @('relative.py', (Join-Path $taskRoot ($taskBase+'/parent-local-foot-geometry-as-executed-e-sibling/source0.py')), (Join-Path $taskRoot ($taskBase+'/parent-local-foot-geometry-as-executed-e/../outside.py')))){
  $a.sources[0].archive_path=$path;$refused=$false
  try{$null=Contact-Archive $a $o $a.source_freeze}catch{$refused=$true};Need $refused 'Absolute/boundary/traversal rejection'
}
$a.sources[0].archive_path=$original
$historicRaw=[IO.File]::ReadAllBytes($original)
$changed=$historicRaw.Clone();$changed[0]=[byte]($changed[0]+1);[IO.File]::WriteAllBytes($original,$changed)
$refused=$false;try{$null=Contact-Archive $a $o $a.source_freeze}catch{$refused=$true};Need $refused 'Byte drift rejected'
[IO.File]::WriteAllBytes($original,$historicRaw)
$foreign=Join-Path $taskRoot 'foreign-archive';$null=[IO.Directory]::CreateDirectory($foreign);[IO.File]::WriteAllBytes((Join-Path $foreign 'source0.py'),$historicRaw)
$junction=Join-Path (Split-Path $original -Parent) 'linked';$null=New-Item -ItemType Junction -Path $junction -Target $foreign
$a.sources[0].archive_path=Join-Path $junction 'source0.py'
$refused=$false;try{$null=Contact-Archive $a $o $a.source_freeze}catch{$refused=$true};Need $refused 'Reparse ancestor refused'
$a.sources[0].archive_path=$original
$a.sources[0].bytes=2097153;$o.sources[$a.sources[0].original_path].bytes=2097153
$refused=$false;try{$null=Contact-Archive $a $o $a.source_freeze}catch{$refused=$true};Need $refused 'Read size capped before archive read'
'ALL_CASES_PASS'
"""
    run_batch(tmp_path, script)


def test_native_transfer_conservation_negatives_and_actual_limiter_once(tmp_path):
    live, event, owner = owner_fixture(tmp_path)
    from tools import banc_m3_startup_ray_supervisor as supervisor
    assert supervisor.generated_budget_limit(live, tmp_path) == 300
    transferred = copy.deepcopy(live)
    transferred.update(direct_science_generated_limit_s=380, direct_science_generated_transfer=event)
    assert supervisor.generated_budget_limit(transferred, tmp_path) == 380
    first_event = json.loads((tmp_path/event["path"]).read_text(encoding="utf8"))
    extra_value = {"schema": "flylab.m3.contact-body-retry.generated-transfer.v1", "transfer_id": "contact-body-retry-20261007-80",
                   "transfer_s": 80, "from_local_limit_s": 380, "to_local_limit_s": 460, "new_resource_grant_s": 0,
                   "initial_policy": transferred["direct_science_policy"], "first_transfer": event,
                   "direct_human_auto_authority": first_event["direct_human_auto_authority"],
                   "before_ledger": put(tmp_path, BASE+"/extra-python-before.json", transferred)}
    extra = put(tmp_path, BASE+"/extra-python-event.json", extra_value)
    second = dict(copy.deepcopy(transferred), direct_science_generated_limit_s=460, direct_science_generated_additional_transfer=extra)
    assert supervisor.generated_budget_limit(second, tmp_path) == 460
    admitted = copy.deepcopy(second)
    admitted["direct_science_calls"]["current"] = {"state": "admitted"}
    assert supervisor.generated_budget_limit(admitted, tmp_path) == 460
    final_value = {"schema": "flylab.m3.contact-body-retry.generated-final-transfer.v1",
                   "transfer_id": "contact-body-retry-20261007-final-140", "transfer_s": 140,
                   "from_local_limit_s": 460, "to_local_limit_s": 600, "new_resource_grant_s": 0,
                   "initial_policy": second["direct_science_policy"], "first_transfer": event, "additional_transfer": extra,
                   "direct_human_auto_authority": first_event["direct_human_auto_authority"],
                   "before_ledger": put(tmp_path, BASE+"/final-python-before.json", second)}
    final = put(tmp_path, BASE+"/final-python-event.json", final_value)
    third = dict(copy.deepcopy(second), direct_science_generated_limit_s=600, direct_science_generated_final_transfer=final)
    assert supervisor.generated_budget_limit(third, tmp_path) == 600
    admitted_final = copy.deepcopy(third)
    admitted_final["direct_science_calls"]["current"] = {"state": "admitted"}
    assert supervisor.generated_budget_limit(admitted_final, tmp_path) == 600
    continuation_value = {"schema": "flylab.m3.contact-body-retry.generated-continuation-transfer.v1",
                          "transfer_id": "contact-body-retry-20261007-continuation-300", "transfer_s": 300,
                          "from_local_limit_s": 600, "to_local_limit_s": 900, "new_resource_grant_s": 0,
                          "initial_policy": third["direct_science_policy"], "first_transfer": event,
                          "additional_transfer": extra, "final_transfer": final,
                          "direct_human_auto_authority": first_event["direct_human_auto_authority"],
                          "before_ledger": put(tmp_path, BASE+"/continuation-python-before.json", third)}
    continuation = put(tmp_path, BASE+"/continuation-python-event.json", continuation_value)
    fourth = dict(copy.deepcopy(third), direct_science_generated_limit_s=900, direct_science_generated_continuation_transfer=continuation)
    assert supervisor.generated_budget_limit(fourth, tmp_path) == 900
    admitted_continuation = copy.deepcopy(fourth)
    admitted_continuation["direct_science_calls"]["current"] = {"state": "admitted"}
    assert supervisor.generated_budget_limit(admitted_continuation, tmp_path) == 900
    for kind in ("amount", "policy", "first", "additional", "final", "authority", "prefix", "call", "total", "dangling", "limit", "stale", "anchor_continuation"):
        invalid, altered = copy.deepcopy(fourth), copy.deepcopy(continuation_value)
        if kind == "amount":
            altered["transfer_s"] = 301
        elif kind in ("policy", "first", "additional", "final", "authority"):
            field = {"policy": "initial_policy", "first": "first_transfer", "additional": "additional_transfer",
                     "final": "final_transfer", "authority": "direct_human_auto_authority"}[kind]
            altered[field]["sha256"] = "0"*64
        elif kind == "prefix":
            invalid["debits"][0]["cost_id"] = "changed"
        elif kind == "call":
            invalid["direct_science_calls"]["historical-paid"]["state"] = "admitted"
        elif kind == "total":
            invalid["direct_science_generated_used_s"] += 1
        elif kind == "dangling":
            del invalid["direct_science_generated_final_transfer"]
        elif kind == "limit":
            invalid["direct_science_generated_limit_s"] = 901
        else:
            anchor = copy.deepcopy(second if kind == "stale" else third)
            if kind == "anchor_continuation":
                anchor["direct_science_generated_continuation_transfer"] = continuation
            altered["before_ledger"] = put(tmp_path, BASE+"/continuation-python-"+kind+"-before.json", anchor)
        invalid["direct_science_generated_continuation_transfer"] = put(tmp_path, BASE+"/continuation-python-"+kind+".json", altered)
        try:
            supervisor.generated_budget_limit(invalid, tmp_path)
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError("Unpinned or invalid continuation allocation accepted")
    for lower in (live, transferred, second, third):
        invalid = dict(copy.deepcopy(lower), direct_science_generated_continuation_transfer=continuation)
        try:
            supervisor.generated_budget_limit(invalid, tmp_path)
        except ValueError as error:
            assert str(error) == "no continuation pointer in lower allocation"
        else:
            raise AssertionError("Continuation pointer accepted under lower allocation")
    for kind in ("amount", "policy", "first", "additional", "authority", "prefix", "call", "total", "dangling", "limit", "stale", "anchor_final"):
        invalid, altered = copy.deepcopy(third), copy.deepcopy(final_value)
        if kind == "amount":
            altered["transfer_s"] = 141
        elif kind in ("policy", "first", "additional", "authority"):
            field = {"policy": "initial_policy", "first": "first_transfer", "additional": "additional_transfer",
                     "authority": "direct_human_auto_authority"}[kind]
            altered[field]["sha256"] = "0"*64
        elif kind == "prefix":
            invalid["debits"][0]["cost_id"] = "changed"
        elif kind == "call":
            invalid["direct_science_calls"]["historical-paid"]["state"] = "admitted"
        elif kind == "total":
            invalid["direct_science_generated_used_s"] += 1
        elif kind == "dangling":
            del invalid["direct_science_generated_additional_transfer"]
        elif kind == "limit":
            invalid["direct_science_generated_limit_s"] = 601
        else:
            anchor = copy.deepcopy(transferred if kind == "stale" else second)
            if kind == "anchor_final":
                anchor["direct_science_generated_final_transfer"] = final
            altered["before_ledger"] = put(tmp_path, BASE+"/final-python-"+kind+"-before.json", anchor)
        invalid["direct_science_generated_final_transfer"] = put(tmp_path, BASE+"/final-python-"+kind+".json", altered)
        try:
            supervisor.generated_budget_limit(invalid, tmp_path)
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError("Unpinned or invalid final allocation accepted")
    for lower in (live, transferred, second):
        invalid = dict(copy.deepcopy(lower), direct_science_generated_final_transfer=final)
        try:
            supervisor.generated_budget_limit(invalid, tmp_path)
        except ValueError as error:
            assert str(error) == "no final pointer in lower allocation"
        else:
            raise AssertionError("Final pointer accepted under lower allocation")
    for kind in ("amount", "policy", "first", "prefix", "call", "total", "dangling", "limit"):
        invalid = copy.deepcopy(second)
        altered = copy.deepcopy(extra_value)
        if kind == "amount":
            altered["transfer_s"] = 81
        elif kind in ("policy", "first"):
            altered["initial_policy" if kind == "policy" else "first_transfer"]["sha256"] = "0"*64
        elif kind == "prefix":
            invalid["debits"][0]["cost_id"] = "changed"
        elif kind == "call":
            invalid["direct_science_calls"]["historical-paid"]["state"] = "admitted"
        elif kind == "total":
            invalid["direct_science_generated_used_s"] += 1
        elif kind == "dangling":
            del invalid["direct_science_generated_transfer"]
        else:
            invalid["direct_science_generated_limit_s"] = 461
        invalid["direct_science_generated_additional_transfer"] = put(tmp_path, BASE+"/extra-python-"+kind+".json", altered)
        try:
            supervisor.generated_budget_limit(invalid, tmp_path)
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError("Unpinned or invalid second allocation accepted")
    for mutation in ("limit", "pin", "history"):
        invalid = copy.deepcopy(transferred)
        if mutation == "limit":
            invalid["direct_science_generated_limit_s"] = 381
        elif mutation == "pin":
            invalid["direct_science_generated_transfer"]["sha256"] = "0" * 64
        else:
            invalid["debits"][0]["cost_id"] = "rewritten"
        try:
            supervisor.generated_budget_limit(invalid, tmp_path)
        except ValueError:
            pass
        else:
            raise AssertionError("Supervisor accepted an unpinned allocation or changed history")
    put(tmp_path, "original-live.json", live)
    put(tmp_path, "event-spec.json", event)
    freeze = put(tmp_path, "freeze.json", {"schema": "explicit.generated.fixture.v1"})
    request_specs = {}
    for label, cap, seconds in (("within", 100, 40), ("overrun", 40, 70), ("reject", 90, 1)):
        call = "direct-science-metadata-" + label
        output = BASE + "/" + call
        request = put(tmp_path, BASE + "/" + call + "-request.json", {
            "schema": "flylab.m3.direct-science.supervision.v1", "mode": "generated", "role": "execution", "call_id": call,
            "source_freeze": freeze, "output": output, "limits": {"outer_s": cap}})
        put(tmp_path, output + "/outer.json", {"request": request, "exit_code": 0, "all_owned_ended": True,
            "reservation_released": True, "stdout_eof": True, "stderr_eof": True, "unknown_owned_identities": {}})
        trace = put(tmp_path, BASE + "/" + call + "-trace.json", {"schema": "flylab.m3.direct-science.genuine-return.v1",
            "phase": "FINAL", "first_error": None, "dispatch_ms": 1000, "first_final_ms": 1000 + seconds * 1000,
            "full_return_s": float(seconds), "exit_code": 0, "request": request, "source_freeze": freeze})
        request_specs[label] = {"request": request, "trace": trace["path"]}
    put(tmp_path, "requests.json", request_specs)
    p = owner.as_posix().replace("'", "''")
    script = f"$ErrorActionPreference='Stop'\n$owner='{p}'\n" + r"""
$root=$PSScriptRoot;$base='verification/banc-natural-gait-m3-20260919';$ledger=Join-Path $root ($base+'/parent-e358-execution-ledger.json')
$original=[IO.File]::ReadAllText((Join-Path $root 'original-live.json'));$event=Get-Content (Join-Path $root 'event-spec.json') -Raw
function Act($action,$value){
  if($action -cin @('generated-transfer','generated-transfer-additional','generated-transfer-final','generated-transfer-continuation')){& $owner -Action $action -TransferDescriptorJSON $value}
  elseif($action -ceq 'admit'){& $owner -Action $action -RequestDescriptorJSON $value}
  else{& $owner -Action $action -TracePath $value}
  if($LASTEXITCODE -ne 0){throw 'Native action failed'}
}
function Refuse($action,$value,[string]$expectedReason=''){
  $raw=[IO.File]::ReadAllBytes($ledger);$failed=$false;$reason=''
  try{Act $action $value}catch{$failed=$true;$reason=$_.Exception.Message}
  if(-not $failed -or [Convert]::ToBase64String($raw) -cne [Convert]::ToBase64String([IO.File]::ReadAllBytes($ledger))){throw 'Refusal changed ledger or accepted'}
  if($expectedReason -cne '' -and $reason -cne $expectedReason){throw ('Wrong semantic refusal: expected '+$expectedReason+'; got '+$reason)}
}
function Scratch-Put([string]$name,$value){
  $raw=[Text.UTF8Encoding]::new($false).GetBytes(($value|ConvertTo-Json -Depth 100))
  [IO.File]::WriteAllBytes((Join-Path $root $name),$raw)
  return @{path=$name;bytes=[long]$raw.Length;sha256=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($raw)).ToLowerInvariant()}
}
$originalEventSpec=$event|ConvertFrom-Json -AsHashtable
$originalEvent=Get-Content (Join-Path $root $originalEventSpec.path) -Raw|ConvertFrom-Json -AsHashtable
function Consistent-Event($live,[string]$label){
  # Exact descriptors bind the mutated synthetic live state, never a stale one.
  $before=Scratch-Put ($base+'/negative-'+$label+'-before.json') $live
  $e=@{schema=$originalEvent.schema;transfer_id=$originalEvent.transfer_id;transfer_s=80;from_local_limit_s=300;to_local_limit_s=380;new_resource_grant_s=0;
    initial_policy=$live.direct_science_policy;before_ledger=$before;direct_human_auto_authority=$originalEvent.direct_human_auto_authority}
  $spec=Scratch-Put ($base+'/negative-'+$label+'-event.json') $e
  return ($spec|ConvertTo-Json -Compress)
}
foreach($field in @('direct_science_workers','direct_science_reservations','direct_science_pending_actuals','direct_science_blocking_holds','startup_ray_original_workers','startup_ray_original_reservations','startup_ray_original_pending_actuals','startup_ray_original_blocking_holds','e360_reservations','e360_pending_actuals','e360_blocking_holds','e359_reservations')){
  $l=$original|ConvertFrom-Json -AsHashtable;$l[$field]['open']=@{};[IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100))
  Refuse 'generated-transfer' (Consistent-Event $l ('open-'+$field))
}
foreach($field in @('e359_reservations','e360_reservations','e360_pending_actuals','e360_blocking_holds')){
  foreach($kind in @('missing','wrongtype')){
    $l=$original|ConvertFrom-Json -AsHashtable
    if($kind -ceq 'missing'){$l.Remove($field)}else{$l[$field]=@()}
    [IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100))
    Refuse 'generated-transfer' (Consistent-Event $l ($kind+'-'+$field)) 'Every required ownership state is a closed dictionary'
  }
}
foreach($field in @('direct_science_generated_used_s','generated_stage_used_s','generated_shared_cumulative_used_s','generated_shared_remaining_s','direct_science_generated_limit_s')){
  $l=$original|ConvertFrom-Json -AsHashtable;$l[$field]+=1;[IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100))
  $expected='Exact live ordered generated accounting'
  if($field -ceq 'direct_science_generated_limit_s'){$expected='Live effective generated limit agrees with event'}
  Refuse 'generated-transfer' (Consistent-Event $l ('sum-'+$field)) $expected
}
$l=$original|ConvertFrom-Json -AsHashtable;$l.direct_science_calls['historical-paid'].state='admitted';[IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100))
Refuse 'generated-transfer' (Consistent-Event $l 'uncharged') 'All existing calls charged'
$l=$original|ConvertFrom-Json -AsHashtable;$l['unrelated_stale']=1;[IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100))
Refuse 'generated-transfer' $event 'Exact full fresh transfer snapshot fields'
# Cold-pinned initial-before/policy/live/event descriptors agree throughout.
# Only each intended semantic field disagrees; insufficient accounting is valid.
foreach($kind in @('offset','baseline','insufficient')){
  $l=$original|ConvertFrom-Json -AsHashtable;$pol=Get-Content (Join-Path $root $l.direct_science_policy.path) -Raw|ConvertFrom-Json -AsHashtable
  $prior=Get-Content (Join-Path $root $pol.before_ledger.path) -Raw|ConvertFrom-Json -AsHashtable
  $expected='Immutable initial policy and exact debit offset'
  if($kind -ceq 'offset'){$pol.local_debit_offset=117}
  elseif($kind -ceq 'baseline'){$pol.generated_shared_baseline_s+=1;$expected='Pinned initial shared baseline and cap'}
  else{
    $pol.generated_stage_before_s=1500.0;$pol.generated_shared_before_s=1580.1900156
    $prior.generated_stage_used_s=1500.0;$prior.generated_shared_cumulative_used_s=1580.1900156;$prior.generated_shared_remaining_s=1700.0-80.1900156-1500.0
    $l.debits=$prior.debits;$l.direct_science_generated_used_s=0.0;$l.generated_stage_used_s=1500.0;$l.generated_shared_cumulative_used_s=1580.1900156;$l.generated_shared_remaining_s=$prior.generated_shared_remaining_s
    $expected='Sufficient existing shared remaining'
  }
  $pol.before_ledger=Scratch-Put ($base+'/negative-'+$kind+'-initial-before.json') $prior
  $l.direct_science_policy=Scratch-Put ($base+'/negative-'+$kind+'-policy.json') $pol
  [IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100))
  Refuse 'generated-transfer' (Consistent-Event $l $kind) $expected
}
[IO.File]::WriteAllText($ledger,$original)
$r=Get-Content (Join-Path $root 'requests.json') -Raw|ConvertFrom-Json -AsHashtable
Refuse 'admit' ($r.within.request|ConvertTo-Json -Compress) # 300 with close20 blocks.
Act 'generated-transfer' $event
$closed380=[IO.File]::ReadAllText($ledger)
$after=Get-Content $ledger -Raw|ConvertFrom-Json -AsHashtable
$after.Remove('direct_science_generated_transfer');$after.direct_science_generated_limit_s=300
if(($after|ConvertTo-Json -Depth 100 -Compress) -cne (($original|ConvertFrom-Json -AsHashtable)|ConvertTo-Json -Depth 100 -Compress)){throw 'Transfer mutated history, usage or Original quota'}
Refuse 'generated-transfer' $event
# 100+20 exceeds available102.362, use bounded40 to admit before40 actual.
$q=Get-Content (Join-Path $root $r.within.request.path) -Raw|ConvertFrom-Json -AsHashtable;$q.limits.outer_s=40
$qRaw=[Text.Encoding]::UTF8.GetBytes(($q|ConvertTo-Json -Depth 100));[IO.File]::WriteAllBytes((Join-Path $root $r.within.request.path),$qRaw)
$r.within.request.bytes=$qRaw.Length;$r.within.request.sha256=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($qRaw)).ToLowerInvariant()
# Re-pin generated trace and outer after request edit; this is scratch metadata.
foreach($name in @($r.within.trace,($q.output+'/outer.json'))){$v=Get-Content (Join-Path $root $name) -Raw|ConvertFrom-Json -AsHashtable;$v.request=$r.within.request;[IO.File]::WriteAllText((Join-Path $root $name),($v|ConvertTo-Json -Depth 100))}
Act 'admit' ($r.within.request|ConvertTo-Json -Compress);Act 'commit' $r.within.trace
$l=Get-Content $ledger -Raw|ConvertFrom-Json -AsHashtable
if($l.direct_science_generated_used_s -ne (277.638+40.0) -or $l.direct_science_blocking_holds.Count -ne 0){throw 'False300 hold or wrong ordered charge'}
Refuse 'admit' ($r.reject.request|ConvertTo-Json -Compress)
Act 'admit' ($r.overrun.request|ConvertTo-Json -Compress);Act 'commit' $r.overrun.trace
$l=Get-Content $ledger -Raw|ConvertFrom-Json -AsHashtable
if($l.debits.Count -ne 121 -or $l.direct_science_generated_used_s -ne (277.638+40.0+70.0) -or $l.direct_science_blocking_holds.Count -ne 1){throw 'Actual over380 not once charged with hold'}
Refuse 'commit' $r.overrun.trace
# Separate known closed380 branch; the prior real over380 hold is preserved above.
[IO.File]::WriteAllText($ledger,$closed380)
function Second-Event($live,[string]$label){
  $prior=Scratch-Put ($base+'/second-'+$label+'-before.json') $live
  $first=Get-Content (Join-Path $root $live.direct_science_generated_transfer.path) -Raw|ConvertFrom-Json -AsHashtable
  $v=@{schema='flylab.m3.contact-body-retry.generated-transfer.v1';transfer_id='contact-body-retry-20261007-80';transfer_s=80;from_local_limit_s=380;to_local_limit_s=460;new_resource_grant_s=0;
    initial_policy=$live.direct_science_policy;first_transfer=$live.direct_science_generated_transfer;direct_human_auto_authority=$first.direct_human_auto_authority;before_ledger=$prior}
  return Scratch-Put ($base+'/second-'+$label+'-event.json') $v
}
$l=$closed380|ConvertFrom-Json -AsHashtable;$second=Second-Event $l 'positive'
foreach($kind in @('amount','policy','first')){
  $v=Get-Content (Join-Path $root $second.path) -Raw|ConvertFrom-Json -AsHashtable
  if($kind -ceq 'amount'){$v.transfer_s=81}elseif($kind -ceq 'policy'){$v.initial_policy.sha256='0'*64}else{$v.first_transfer.sha256='0'*64}
  $bad=Scratch-Put ($base+'/second-bad-'+$kind+'.json') $v
  Refuse 'generated-transfer-additional' ($bad|ConvertTo-Json -Compress)
}
foreach($kind in @('prefix','call','owner')){
  $l=$closed380|ConvertFrom-Json -AsHashtable
  if($kind -ceq 'prefix'){$l.debits[0].cost_id='changed'}elseif($kind -ceq 'call'){$l.direct_science_calls['historical-paid'].state='admitted'}else{$l.direct_science_workers['unknown']=@{}}
  [IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100));$bad=Second-Event $l $kind
  Refuse 'generated-transfer-additional' ($bad|ConvertTo-Json -Compress)
}
# A consistently pinned shared state that supports380, but cannot support460.
$l=$closed380|ConvertFrom-Json -AsHashtable;$pol=Get-Content (Join-Path $root $l.direct_science_policy.path) -Raw|ConvertFrom-Json -AsHashtable
$initial=Get-Content (Join-Path $root $pol.before_ledger.path) -Raw|ConvertFrom-Json -AsHashtable
$pol.generated_stage_before_s=1200.0;$pol.generated_shared_before_s=1280.1900156
$initial.generated_stage_used_s=1200.0;$initial.generated_shared_cumulative_used_s=1280.1900156;$initial.generated_shared_remaining_s=1700.0-80.1900156-1200.0
$pol.before_ledger=Scratch-Put ($base+'/second-exhaust-initial.json') $initial;$l.direct_science_policy=Scratch-Put ($base+'/second-exhaust-policy.json') $pol
$l.generated_stage_used_s=1200.0+[double]$l.direct_science_generated_used_s;$l.generated_shared_cumulative_used_s=1280.1900156+[double]$l.direct_science_generated_used_s;$l.generated_shared_remaining_s=1700.0-80.1900156-[double]$l.generated_stage_used_s
$pre=($l|ConvertTo-Json -Depth 100)|ConvertFrom-Json -AsHashtable;$pre.Remove('direct_science_generated_transfer');$pre.direct_science_generated_limit_s=300
$first=$originalEvent|ConvertTo-Json -Depth 100|ConvertFrom-Json -AsHashtable;$first.initial_policy=$l.direct_science_policy;$first.before_ledger=Scratch-Put ($base+'/second-exhaust-first-before.json') $pre
$l.direct_science_generated_transfer=Scratch-Put ($base+'/second-exhaust-first.json') $first
[IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100));$bad=Second-Event $l 'exhaust'
Refuse 'generated-transfer-additional' ($bad|ConvertTo-Json -Compress) 'Sufficient existing shared for additional allocation'
[IO.File]::WriteAllText($ledger,$closed380)
Act 'generated-transfer-additional' ($second|ConvertTo-Json -Compress)
$closed460=[IO.File]::ReadAllText($ledger)
$after=Get-Content $ledger -Raw|ConvertFrom-Json -AsHashtable
if($after.direct_science_generated_limit_s -ne 460){throw 'Native effective460'}
$after.Remove('direct_science_generated_additional_transfer');$after.direct_science_generated_limit_s=380
if(($after|ConvertTo-Json -Depth 100 -Compress) -cne (($closed380|ConvertFrom-Json -AsHashtable)|ConvertTo-Json -Depth 100 -Compress)){throw 'Second transfer changed other fields'}
Refuse 'generated-transfer-additional' ($second|ConvertTo-Json -Compress)
Refuse 'generated-transfer-additional' ((Scratch-Put ($base+'/second-alternate.json') (Get-Content (Join-Path $root $second.path) -Raw|ConvertFrom-Json -AsHashtable))|ConvertTo-Json -Compress)
# Fresh scratch receipts exercise above380 admission, closing reserve, and460 overrun.
function Second-Receipt([string]$name,[long]$cap,[long]$seconds){
  $call='direct-science-second-'+$name;$out=$base+'/'+$call
  $q=@{schema='flylab.m3.direct-science.supervision.v1';mode='generated';role='execution';call_id=$call;source_freeze=$r.within.request.source_freeze;output=$out;limits=@{outer_s=$cap}}
  $q.source_freeze=(Get-Content (Join-Path $root $r.within.request.path) -Raw|ConvertFrom-Json -AsHashtable).source_freeze
  $null=[IO.Directory]::CreateDirectory((Join-Path $root $out));$spec=Scratch-Put ($base+'/'+$call+'-request.json') $q
  $null=Scratch-Put ($out+'/outer.json') @{request=$spec;exit_code=[long]0;all_owned_ended=$true;reservation_released=$true;stdout_eof=$true;stderr_eof=$true;unknown_owned_identities=@{}}
  $trace=Scratch-Put ($base+'/'+$call+'-trace.json') @{schema='flylab.m3.direct-science.genuine-return.v1';phase='FINAL';first_error=$null;dispatch_ms=[long]1000;first_final_ms=[long](1000+1000*$seconds);full_return_s=[double]$seconds;exit_code=[long]0;request=$spec;source_freeze=$q.source_freeze}
  return @{request=$spec;trace=$trace.path}
}
$within=Second-Receipt 'within' 100 100;$reject=Second-Receipt 'reject' 80 1;$over=Second-Receipt 'over' 40 90
Act 'admit' ($within.request|ConvertTo-Json -Compress);Act 'commit' $within.trace
$l=Get-Content $ledger -Raw|ConvertFrom-Json -AsHashtable
if($l.direct_science_generated_used_s -ne (277.638+100.0) -or $l.direct_science_blocking_holds.Count -ne 0){throw 'Above380 availability paid under460'}
Refuse 'admit' ($reject.request|ConvertTo-Json -Compress)
Act 'admit' ($over.request|ConvertTo-Json -Compress);Act 'commit' $over.trace
$l=Get-Content $ledger -Raw|ConvertFrom-Json -AsHashtable
if($l.debits.Count -ne 121 -or $l.direct_science_generated_used_s -ne (277.638+100.0+90.0) -or $l.direct_science_blocking_holds.Count -ne 1){throw 'Actual over460 not once charged with hold'}
Refuse 'commit' $over.trace
# Separate known closed460 state; both older overrun holds remain verified above.
[IO.File]::WriteAllText($ledger,$closed460)
function Final-Event($live,[string]$label){
  $prior=Scratch-Put ($base+'/final-'+$label+'-before.json') $live
  $first=Get-Content (Join-Path $root $live.direct_science_generated_transfer.path) -Raw|ConvertFrom-Json -AsHashtable
  return Scratch-Put ($base+'/final-'+$label+'-event.json') @{schema='flylab.m3.contact-body-retry.generated-final-transfer.v1';transfer_id='contact-body-retry-20261007-final-140';transfer_s=140;from_local_limit_s=460;to_local_limit_s=600;new_resource_grant_s=0;
    initial_policy=$live.direct_science_policy;first_transfer=$live.direct_science_generated_transfer;additional_transfer=$live.direct_science_generated_additional_transfer;direct_human_auto_authority=$first.direct_human_auto_authority;before_ledger=$prior}
}
$l=$closed460|ConvertFrom-Json -AsHashtable;$final=Final-Event $l 'positive'
foreach($kind in @('amount','policy','first','additional','authority')){
  $v=Get-Content (Join-Path $root $final.path) -Raw|ConvertFrom-Json -AsHashtable
  if($kind -ceq 'amount'){$v.transfer_s=141}else{$field=@{policy='initial_policy';first='first_transfer';additional='additional_transfer';authority='direct_human_auto_authority'}[$kind];$v[$field].sha256='0'*64}
  Refuse 'generated-transfer-final' ((Scratch-Put ($base+'/final-bad-'+$kind+'.json') $v)|ConvertTo-Json -Compress)
}
foreach($kind in @('prefix','call','owner')){
  $l=$closed460|ConvertFrom-Json -AsHashtable
  if($kind -ceq 'prefix'){$l.debits[0].cost_id='changed'}elseif($kind -ceq 'call'){$l.direct_science_calls['historical-paid'].state='admitted'}else{$l.direct_science_workers['unknown']=@{}}
  [IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100));$bad=Final-Event $l $kind
  Refuse 'generated-transfer-final' ($bad|ConvertTo-Json -Compress)
}
foreach($state in @($original,$closed380,$closed460)){
  $l=$state|ConvertFrom-Json -AsHashtable;$l.direct_science_generated_final_transfer=$final
  [IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100))
  Refuse 'admit' ($r.within.request|ConvertTo-Json -Compress) 'No final pointer in lower allocation'
}
foreach($kind in @('stale','anchor-final','dangling')){
  $l=$closed460|ConvertFrom-Json -AsHashtable;$l.direct_science_generated_limit_s=600
  $v=Get-Content (Join-Path $root $final.path) -Raw|ConvertFrom-Json -AsHashtable
  if($kind -ceq 'dangling'){$l.Remove('direct_science_generated_additional_transfer');$l.direct_science_generated_final_transfer=$final}
  else{
    $a=if($kind -ceq 'stale'){$closed380|ConvertFrom-Json -AsHashtable}else{$closed460|ConvertFrom-Json -AsHashtable}
    if($kind -ceq 'anchor-final'){$a.direct_science_generated_final_transfer=$final}
    $v.before_ledger=Scratch-Put ($base+'/final-'+$kind+'-anchor.json') $a
    $l.direct_science_generated_final_transfer=Scratch-Put ($base+'/final-'+$kind+'-reader.json') $v
  }
  [IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100));Refuse 'admit' ($r.within.request|ConvertTo-Json -Compress)
}
# Consistently re-pin a shared state supporting460, but not600, through both prior events.
$l=$closed460|ConvertFrom-Json -AsHashtable;$pol=Get-Content (Join-Path $root $l.direct_science_policy.path) -Raw|ConvertFrom-Json -AsHashtable
$initial=Get-Content (Join-Path $root $pol.before_ledger.path) -Raw|ConvertFrom-Json -AsHashtable
$pol.generated_stage_before_s=1100.0;$pol.generated_shared_before_s=1180.1900156
$initial.generated_stage_used_s=1100.0;$initial.generated_shared_cumulative_used_s=1180.1900156;$initial.generated_shared_remaining_s=1700.0-80.1900156-1100.0
$pol.before_ledger=Scratch-Put ($base+'/final-exhaust-initial.json') $initial;$l.direct_science_policy=Scratch-Put ($base+'/final-exhaust-policy.json') $pol
$l.generated_stage_used_s=1100.0+[double]$l.direct_science_generated_used_s;$l.generated_shared_cumulative_used_s=1180.1900156+[double]$l.direct_science_generated_used_s;$l.generated_shared_remaining_s=1700.0-80.1900156-[double]$l.generated_stage_used_s
$pre=($l|ConvertTo-Json -Depth 100)|ConvertFrom-Json -AsHashtable;$pre.Remove('direct_science_generated_transfer');$pre.Remove('direct_science_generated_additional_transfer');$pre.direct_science_generated_limit_s=300
$first=$originalEvent|ConvertTo-Json -Depth 100|ConvertFrom-Json -AsHashtable;$first.initial_policy=$l.direct_science_policy;$first.before_ledger=Scratch-Put ($base+'/final-exhaust-first-before.json') $pre
$l.direct_science_generated_transfer=Scratch-Put ($base+'/final-exhaust-first.json') $first
$pre=($l|ConvertTo-Json -Depth 100)|ConvertFrom-Json -AsHashtable;$pre.Remove('direct_science_generated_additional_transfer');$pre.direct_science_generated_limit_s=380
$l.direct_science_generated_additional_transfer=Second-Event $pre 'final-exhaust'
[IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100));$bad=Final-Event $l 'exhaust'
Refuse 'generated-transfer-final' ($bad|ConvertTo-Json -Compress) 'Sufficient existing shared for final allocation'
[IO.File]::WriteAllText($ledger,$closed460)
$v=Get-Content (Join-Path $root $final.path) -Raw|ConvertFrom-Json -AsHashtable
$v.before_ledger=Scratch-Put ($base+'/final-stale-before.json') ($closed380|ConvertFrom-Json -AsHashtable)
$stale=Scratch-Put ($base+'/final-stale-before-event.json') $v
Refuse 'generated-transfer-final' ($stale|ConvertTo-Json -Compress) 'Exact full fresh final snapshot fields'
Act 'generated-transfer-final' ($final|ConvertTo-Json -Compress)
$closed600=[IO.File]::ReadAllText($ledger)
$after=Get-Content $ledger -Raw|ConvertFrom-Json -AsHashtable
if($after.direct_science_generated_limit_s -ne 600){throw 'Native effective600'}
$after.Remove('direct_science_generated_final_transfer');$after.direct_science_generated_limit_s=460
if(($after|ConvertTo-Json -Depth 100 -Compress) -cne (($closed460|ConvertFrom-Json -AsHashtable)|ConvertTo-Json -Depth 100 -Compress)){throw 'Final transfer changed other fields'}
Refuse 'generated-transfer-final' ($final|ConvertTo-Json -Compress)
Refuse 'generated-transfer-final' ((Scratch-Put ($base+'/final-alternate.json') (Get-Content (Join-Path $root $final.path) -Raw|ConvertFrom-Json -AsHashtable))|ConvertTo-Json -Compress)
$seed=Second-Receipt 'final-seed' 100 100
Act 'admit' ($seed.request|ConvertTo-Json -Compress);Act 'commit' $seed.trace
$l=Get-Content $ledger -Raw|ConvertFrom-Json -AsHashtable
if(($l.direct_science_generated_used_s+100+20) -le 460 -or ($l.direct_science_generated_used_s+100+20) -gt 600){throw '600 within must require new allocation under per-call120'}
$within=Second-Receipt 'final-within' 100 100;$reject=Second-Receipt 'final-reject' 110 1;$over=Second-Receipt 'final-over' 60 130
Act 'admit' ($within.request|ConvertTo-Json -Compress);Act 'commit' $within.trace
$l=Get-Content $ledger -Raw|ConvertFrom-Json -AsHashtable
if($l.direct_science_generated_used_s -ne (277.638+100.0+100.0) -or $l.direct_science_blocking_holds.Count -ne 0){throw 'Above460 availability paid under600'}
Refuse 'admit' ($reject.request|ConvertTo-Json -Compress)
Act 'admit' ($over.request|ConvertTo-Json -Compress);Act 'commit' $over.trace
$l=Get-Content $ledger -Raw|ConvertFrom-Json -AsHashtable
if($l.debits.Count -ne 122 -or $l.direct_science_generated_used_s -ne (277.638+100.0+100.0+130.0) -or $l.direct_science_blocking_holds.Count -ne 1){throw 'Actual over600 not once charged with hold'}
Refuse 'commit' $over.trace
# Separate closed600 before seed usage. Retain the verified over600 hold above.
[IO.File]::WriteAllText($ledger,$closed600)
function Continuation-Event($live,[string]$label){
  $prior=Scratch-Put ($base+'/continuation-'+$label+'-before.json') $live
  $first=Get-Content (Join-Path $root $live.direct_science_generated_transfer.path) -Raw|ConvertFrom-Json -AsHashtable
  return Scratch-Put ($base+'/continuation-'+$label+'-event.json') @{schema='flylab.m3.contact-body-retry.generated-continuation-transfer.v1';transfer_id='contact-body-retry-20261007-continuation-300';transfer_s=300;from_local_limit_s=600;to_local_limit_s=900;new_resource_grant_s=0;
    initial_policy=$live.direct_science_policy;first_transfer=$live.direct_science_generated_transfer;additional_transfer=$live.direct_science_generated_additional_transfer;final_transfer=$live.direct_science_generated_final_transfer;direct_human_auto_authority=$first.direct_human_auto_authority;before_ledger=$prior}
}
$l=$closed600|ConvertFrom-Json -AsHashtable;$continuation=Continuation-Event $l 'positive'
foreach($kind in @('amount','policy','first','additional','final','authority')){
  $v=Get-Content (Join-Path $root $continuation.path) -Raw|ConvertFrom-Json -AsHashtable
  if($kind -ceq 'amount'){$v.transfer_s=301}else{$field=@{policy='initial_policy';first='first_transfer';additional='additional_transfer';final='final_transfer';authority='direct_human_auto_authority'}[$kind];$v[$field].sha256='0'*64}
  Refuse 'generated-transfer-continuation' ((Scratch-Put ($base+'/continuation-bad-'+$kind+'.json') $v)|ConvertTo-Json -Compress)
}
foreach($kind in @('prefix','call','owner')){
  $l=$closed600|ConvertFrom-Json -AsHashtable
  if($kind -ceq 'prefix'){$l.debits[0].cost_id='changed'}elseif($kind -ceq 'call'){$l.direct_science_calls['historical-paid'].state='admitted'}else{$l.direct_science_workers['unknown']=@{}}
  [IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100));$bad=Continuation-Event $l $kind
  Refuse 'generated-transfer-continuation' ($bad|ConvertTo-Json -Compress)
}
foreach($state in @($original,$closed380,$closed460,$closed600)){
  $l=$state|ConvertFrom-Json -AsHashtable;$l.direct_science_generated_continuation_transfer=$continuation
  [IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100))
  Refuse 'admit' ($r.within.request|ConvertTo-Json -Compress) 'No continuation pointer in lower allocation'
}
foreach($kind in @('stale','anchor-continuation','dangling')){
  $l=$closed600|ConvertFrom-Json -AsHashtable;$l.direct_science_generated_limit_s=900
  $v=Get-Content (Join-Path $root $continuation.path) -Raw|ConvertFrom-Json -AsHashtable
  if($kind -ceq 'dangling'){$l.Remove('direct_science_generated_final_transfer');$l.direct_science_generated_continuation_transfer=$continuation}
  else{
    $a=if($kind -ceq 'stale'){$closed460|ConvertFrom-Json -AsHashtable}else{$closed600|ConvertFrom-Json -AsHashtable}
    if($kind -ceq 'anchor-continuation'){$a.direct_science_generated_continuation_transfer=$continuation}
    $v.before_ledger=Scratch-Put ($base+'/continuation-'+$kind+'-anchor.json') $a
    $l.direct_science_generated_continuation_transfer=Scratch-Put ($base+'/continuation-'+$kind+'-reader.json') $v
  }
  [IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100));Refuse 'admit' ($r.within.request|ConvertTo-Json -Compress)
}
# Consistently pinned shared state supports600, but not900.
$l=$closed600|ConvertFrom-Json -AsHashtable;$pol=Get-Content (Join-Path $root $l.direct_science_policy.path) -Raw|ConvertFrom-Json -AsHashtable
$initial=Get-Content (Join-Path $root $pol.before_ledger.path) -Raw|ConvertFrom-Json -AsHashtable
$pol.generated_stage_before_s=900.0;$pol.generated_shared_before_s=980.1900156
$initial.generated_stage_used_s=900.0;$initial.generated_shared_cumulative_used_s=980.1900156;$initial.generated_shared_remaining_s=1700.0-80.1900156-900.0
$pol.before_ledger=Scratch-Put ($base+'/continuation-exhaust-initial.json') $initial;$l.direct_science_policy=Scratch-Put ($base+'/continuation-exhaust-policy.json') $pol
$l.generated_stage_used_s=900.0+[double]$l.direct_science_generated_used_s;$l.generated_shared_cumulative_used_s=980.1900156+[double]$l.direct_science_generated_used_s;$l.generated_shared_remaining_s=1700.0-80.1900156-[double]$l.generated_stage_used_s
$pre=($l|ConvertTo-Json -Depth 100)|ConvertFrom-Json -AsHashtable;$pre.Remove('direct_science_generated_transfer');$pre.Remove('direct_science_generated_additional_transfer');$pre.Remove('direct_science_generated_final_transfer');$pre.direct_science_generated_limit_s=300
$first=$originalEvent|ConvertTo-Json -Depth 100|ConvertFrom-Json -AsHashtable;$first.initial_policy=$l.direct_science_policy;$first.before_ledger=Scratch-Put ($base+'/continuation-exhaust-first-before.json') $pre
$l.direct_science_generated_transfer=Scratch-Put ($base+'/continuation-exhaust-first.json') $first
$pre=($l|ConvertTo-Json -Depth 100)|ConvertFrom-Json -AsHashtable;$pre.Remove('direct_science_generated_additional_transfer');$pre.Remove('direct_science_generated_final_transfer');$pre.direct_science_generated_limit_s=380
$l.direct_science_generated_additional_transfer=Second-Event $pre 'continuation-exhaust'
$pre=($l|ConvertTo-Json -Depth 100)|ConvertFrom-Json -AsHashtable;$pre.Remove('direct_science_generated_final_transfer');$pre.direct_science_generated_limit_s=460
$l.direct_science_generated_final_transfer=Final-Event $pre 'continuation-exhaust'
[IO.File]::WriteAllText($ledger,($l|ConvertTo-Json -Depth 100));$bad=Continuation-Event $l 'exhaust'
Refuse 'generated-transfer-continuation' ($bad|ConvertTo-Json -Compress) 'Sufficient existing shared for continuation allocation'
[IO.File]::WriteAllText($ledger,$closed600)
$v=Get-Content (Join-Path $root $continuation.path) -Raw|ConvertFrom-Json -AsHashtable
$v.before_ledger=Scratch-Put ($base+'/continuation-stale-before.json') ($closed460|ConvertFrom-Json -AsHashtable)
Refuse 'generated-transfer-continuation' ((Scratch-Put ($base+'/continuation-stale-event.json') $v)|ConvertTo-Json -Compress) 'Exact full fresh continuation snapshot fields'
Act 'generated-transfer-continuation' ($continuation|ConvertTo-Json -Compress)
$after=Get-Content $ledger -Raw|ConvertFrom-Json -AsHashtable
if($after.direct_science_generated_limit_s -ne 900){throw 'Native effective900'}
$after.Remove('direct_science_generated_continuation_transfer');$after.direct_science_generated_limit_s=600
if(($after|ConvertTo-Json -Depth 100 -Compress) -cne (($closed600|ConvertFrom-Json -AsHashtable)|ConvertTo-Json -Depth 100 -Compress)){throw 'Continuation changed other fields'}
Refuse 'generated-transfer-continuation' ($continuation|ConvertTo-Json -Compress)
Refuse 'generated-transfer-continuation' ((Scratch-Put ($base+'/continuation-alternate.json') (Get-Content (Join-Path $root $continuation.path) -Raw|ConvertFrom-Json -AsHashtable))|ConvertTo-Json -Compress)
for($i=0;$i -lt 3;$i++){
  $seed=Second-Receipt ('continuation-seed-'+$i) 100 100
  Act 'admit' ($seed.request|ConvertTo-Json -Compress);Act 'commit' $seed.trace
}
$l=Get-Content $ledger -Raw|ConvertFrom-Json -AsHashtable
if(($l.direct_science_generated_used_s+100+20) -le 600 -or ($l.direct_science_generated_used_s+100+20) -gt 900){throw '900 within must require new allocation under per-call120'}
for($i=0;$i -lt 2;$i++){
  $within=Second-Receipt ('continuation-within-'+$i) 100 100
  Act 'admit' ($within.request|ConvertTo-Json -Compress);Act 'commit' $within.trace
}
$l=Get-Content $ledger -Raw|ConvertFrom-Json -AsHashtable
if($l.direct_science_generated_used_s -ne (277.638+100.0+100.0+100.0+100.0+100.0) -or $l.direct_science_blocking_holds.Count -ne 0){throw '900 sequential bounded actual receipts'}
$reject=Second-Receipt 'continuation-reject' 110 1;$over=Second-Receipt 'continuation-over' 60 130
Refuse 'admit' ($reject.request|ConvertTo-Json -Compress)
Act 'admit' ($over.request|ConvertTo-Json -Compress);Act 'commit' $over.trace
$l=Get-Content $ledger -Raw|ConvertFrom-Json -AsHashtable
if($l.debits.Count -ne 125 -or $l.direct_science_generated_used_s -ne (277.638+100.0+100.0+100.0+100.0+100.0+130.0) -or $l.direct_science_blocking_holds.Count -ne 1){throw 'Actual over900 not once charged with hold'}
Refuse 'commit' $over.trace
'ALL_CASES_PASS'
"""
    run_batch(tmp_path, script)


def test_native_suffix_preflight_drift_and_no_replay(tmp_path):
    live, _, _ = owner_fixture(tmp_path)
    put(tmp_path, "live.json", live)
    source = (ROOT / PREPARE).read_text(encoding="utf8").split("$contractName=$replayBase+", 1)[0]
    source = source[source.index("$ErrorActionPreference="):]
    source = rebound(source, tmp_path, "replayRoot")
    script = source + r"""
$l=Read-Metadata 'live.json';$call='contact-learning-20261007-a-direction'
$old=Contact-Targets $call '';$new=Contact-Targets $call 'b'
$null=Publish $old.science @{old='failed unadmitted evidence'};$oldPin=Spec $old.science
Contact-Fresh $l $call $new
foreach($suffix in @('../b','B','b-c','b_','123','abcdefghi')){
  $failed=$false;try{$null=Contact-Targets $call $suffix}catch{$failed=$true};Need $failed 'Invalid suffix refused'
}
foreach($name in $new.Values){$null=Publish $name @{fixture='preserved'};$failed=$false;try{Contact-Fresh $l $call $new}catch{$failed=$true};Need $failed 'All three target collisions refused';[IO.File]::Delete((Target $name))}
$null=Same-Pin $oldPin
$initial=Spec ($replayBase+'/parent-e358-execution-ledger.json');$beforeName=$new.before
[IO.File]::Copy((Target $initial.path),(Target $beforeName),$false);$before=Spec $beforeName
Contact-Snapshot $before $initial
# Live drift is rejected and remains present, with snapshot and old evidence retained.
$ledgerFile=Target $initial.path;[IO.File]::AppendAllText($ledgerFile,' ');$drift=Spec $initial.path
$failed=$false;try{Contact-Snapshot $before $initial}catch{$failed=$true};Need $failed 'Ledger drift refused'
$null=Same-Pin $drift;$null=Same-Pin $before;$null=Same-Pin $oldPin
$l.direct_science_calls[$call]=@{state='admitted'}
$failed=$false;try{Contact-Fresh $l $call (Contact-Targets $call 'c')}catch{$failed=$true};Need $failed 'Different suffix cannot replay admitted fixed call'
'ALL_CASES_PASS'
"""
    run_batch(tmp_path, script)
