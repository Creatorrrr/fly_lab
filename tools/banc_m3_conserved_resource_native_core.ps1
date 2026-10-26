
# Common production metadata transaction. These definitions are also extracted by
# generated temporary-root fixtures; no acquisition occurs while definitions load.
function Conserved-Count($value){
  Need ($value -is [Collections.IDictionary]) 'Actual IDictionary count'
  [long]$n=0;foreach($entry in $value.GetEnumerator()){$n++};return $n
}
function Conserved-Keys($value){Need ($value -is [Collections.IDictionary]) 'Actual dictionary enumeration';foreach($entry in $value.GetEnumerator()){Write-Output $entry.Key}}
function Conserved-Same($actual,$expected,[string]$label){
  if($null -eq $expected -or $null -eq $actual){Need ($null -eq $expected -and $null -eq $actual) $label;return}
  if($expected -is [Collections.IDictionary]){
    Need ($actual -is [Collections.IDictionary] -and (Conserved-Count $actual) -eq (Conserved-Count $expected)) ($label+' real fields')
    $names=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach($entry in $actual.GetEnumerator()){Need ($entry.Key -is [string] -and $names.Add($entry.Key)) ($label+' unique ordinal keys')}
    foreach($entry in $expected.GetEnumerator()){Need ($names.Contains($entry.Key)) ($label+' exact key');Conserved-Same $actual[$entry.Key] $entry.Value ($label+'.'+$entry.Key)}
  }elseif($expected -is [array]){
    Need ($actual -is [array] -and $actual.Length -eq $expected.Length) ($label+' typed array')
    for($i=0;$i -lt $expected.Length;$i++){Conserved-Same $actual[$i] $expected[$i] $label}
  }elseif($expected -is [long] -or $expected -is [int]){
    Need (($actual -is [long] -or $actual -is [int]) -and $actual -eq $expected) ($label+' integer, never bool/float')
  }elseif($expected -is [double]){
    Need ($actual -is [double] -and [double]::IsFinite($actual) -and [double]::IsFinite($expected) -and $actual -eq $expected) ($label+' exact finite binary64')
  }else{Need ($actual.GetType() -eq $expected.GetType() -and $actual -ceq $expected) ($label+' type/value')}
}
function Conserved-ExactKeys($value,[string[]]$expected){
  Need ((Conserved-Count $value) -eq $expected.Length) 'Exact genuine dictionary field count'
  $names=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
  foreach($entry in $value.GetEnumerator()){Need ($entry.Key -is [string] -and $names.Add($entry.Key)) 'Unique ordinal dictionary keys'}
  foreach($name in $expected){Need ($names.Contains($name)) 'No Count/Keys shadow or substituted field'}
}
function Conserved-Json($raw){
  $text=if($raw -is [byte[]]){$utf8.GetString($raw)}else{[string]$raw}
  function Test-ConservedJson($element){
    if($element.ValueKind -eq [Text.Json.JsonValueKind]::Object){
      $seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
      foreach($property in $element.EnumerateObject()){Need ($seen.Add($property.Name)) 'Duplicate raw JSON event/key rejected';Test-ConservedJson $property.Value}
    }elseif($element.ValueKind -eq [Text.Json.JsonValueKind]::Array){foreach($item in $element.EnumerateArray()){Test-ConservedJson $item}}
  }
  $document=[Text.Json.JsonDocument]::Parse($text,[Text.Json.JsonDocumentOptions]::new())
  try{Test-ConservedJson $document.RootElement}finally{$document.Dispose()}
  return $text|ConvertFrom-Json -AsHashtable
}
function Conserved-ReadPin($spec,[long]$cap=1048576){
  Conserved-ExactKeys $spec @('path','bytes','sha256')
  Need ($spec.path -is [string] -and ($spec.bytes -is [long] -or $spec.bytes -is [int]) -and $spec.bytes -gt 0 -and $spec.bytes -le $cap -and $spec.sha256 -is [string] -and $spec.sha256 -cmatch '^[0-9a-f]{64}$') 'Typed bounded epoch descriptor'
  $file=Safe $spec.path;$stamp=([IO.FileInfo]$file).LastWriteTimeUtc.Ticks
  Need (([IO.FileInfo]$file).Length -eq $spec.bytes) 'Epoch exact bytes before read'
  $raw=[IO.File]::ReadAllBytes($file)
  Need ($raw.Length -eq $spec.bytes -and [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($raw)).ToLowerInvariant() -ceq $spec.sha256 -and
    [Convert]::ToBase64String($raw) -ceq [Convert]::ToBase64String([IO.File]::ReadAllBytes($file)) -and ([IO.FileInfo]$file).LastWriteTimeUtc.Ticks -eq $stamp) 'Epoch cold hash/full readback/stamp'
  return Conserved-Json $raw
}
function Conserved-SourcePin($spec,[string]$name){
  Conserved-ExactKeys $spec @('path','bytes','sha256');Conserved-Same $spec.path $name 'Same critical production Source path'
  Need (($spec.bytes -is [long] -or $spec.bytes -is [int]) -and $spec.bytes -gt 0 -and $spec.bytes -le 262144 -and $spec.sha256 -is [string] -and $spec.sha256 -cmatch '^[0-9a-f]{64}$') 'Typed bounded critical Source descriptor'
  $file=Safe $spec.path;$stamp=([IO.FileInfo]$file).LastWriteTimeUtc.Ticks
  Need (([IO.FileInfo]$file).Length -eq $spec.bytes) 'Critical Source exact bytes before read'
  $raw=[IO.File]::ReadAllBytes($file)
  Need ($raw.Length -eq $spec.bytes -and [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($raw)).ToLowerInvariant() -ceq $spec.sha256 -and
    [Convert]::ToBase64String($raw) -ceq [Convert]::ToBase64String([IO.File]::ReadAllBytes($file)) -and ([IO.FileInfo]$file).LastWriteTimeUtc.Ticks -eq $stamp) 'Critical Source full current hash/readback/stamp'
  $null=$utf8.GetString($raw)
}
function Conserved-BootstrapSources {return @('tools/banc_m3_conserved_resource.py','tools/banc_m3_conserved_resource_native_core.ps1','tools/banc_m3_conserved_resource_native_epoch.ps1',($taskBase+'/parent-direct-science-account.ps1'),'tools/banc_m3_startup_ray_supervisor.py','tests/test_c_banc_contact_runtime_journal.py')}
function Conserved-BootstrapDispatchable {return $true} # first bounded attempt permitted; successful fit remains unobserved
function Conserved-BootstrapAdmission($q,$l){
  Conserved-Closed $l
  Conserved-Same $q.call_id 'direct-science-20261008-conserved-bootstrap' 'Single unused bootstrap call'
  Conserved-Same $q.mode 'generated' 'Generated temporary experiment only'
  Conserved-Same $q.role 'execution' 'Full bootstrap execution role'
  Conserved-Same $q.generated_behavior 'command' 'Full fixed bootstrap command'
  Conserved-Same $q.generated_argv @('-m','pytest','tests/test_c_banc_contact_runtime_journal.py::test_gate_conserved_bootstrap_production_transaction_and_native_parity','--tb=short','-q',('--junitxml='+$taskBase+'/parent-conserved-resource-bootstrap-junit.xml')) 'Entire fixed bootstrap; no subset'
  foreach($pair in @(@('outer_s',11),@('child_s',8),@('cleanup_s',3))){Conserved-Same $q.limits[$pair[0]] ([long]$pair[1]) 'Exact old-budget bootstrap bounds'}
  Conserved-Same $q.limits.host_bytes ([long]2147483648) 'Fixed bounded bootstrap host'
  Conserved-Same $q.limits.output_cap_bytes ([long]8388608) 'Full bounded bootstrap output'
  Conserved-Same $q.ledger_path ($taskBase+'/parent-e358-execution-ledger.json') 'Same actual ledger'
  Conserved-Same $q.shared_original_lock_path ($taskBase+'/parent-e358-original-active.lock') 'Same actual shared exclusion'
  Conserved-Same $q.output ($taskBase+'/parent-direct-science-20261008-conserved-bootstrap') 'Fixed fresh bootstrap output'
  Conserved-Same $q.source_freeze.path ($taskBase+'/parent-contact-learning-source-freeze-20261008-e.json') 'Current reviewed complete Source path'
  Need (-not $l.Contains('direct_science_conserved_allocation') -and -not $l.direct_science_calls.Contains($q.call_id)) 'No future funds or replay'
  Conserved-Same $l.generated_shared_cumulative_limit_s ([long]1700) 'Old shared cap'
  Conserved-Same $l.direct_science_generated_limit_s ([long]964) 'Old local cap'
  Conserved-Same $l.original_cumulative_limit_s ([long]16200) 'Old Original cap'
  Need ([double]$l.direct_science_generated_used_s+11.0+20.0 -le 964.0 -and [double]$l.generated_shared_remaining_s -ge 31.0) 'Whole attempt plus distinct close20 fits actual old budget'
  Conserved-BootstrapSource $q.source_freeze
}
function Conserved-Pins {
  return '{"historical":{"path":"verification/banc-natural-gait-m3-20260919/parent-native-count-conformance-closed282-20261008-a.json","bytes":453918,"sha256":"405ffe995f1241f2ff3263efc916c9e69897b18c37028da3a9aab82c6d37f661"},"policy":{"path":"verification/banc-natural-gait-m3-20260919/parent-direct-science-generated-policy.json","bytes":1496,"sha256":"693b5193d09a83f5de3b48b99c13199fb3eed2a955c044aa1e97a5a8a9546403"},"j":{"path":"verification/banc-natural-gait-m3-20260919/parent-startup-ray-original-grant-20261007-j.json","bytes":2227,"sha256":"7245e218f5c9abae27a849f880f37bc6013df87ae5e514947887183b4a2dba2f"},"human":{"path":"verification/banc-natural-gait-m3-20260919/parent-contact-metadata-current-human-autonomy-20261007-a.json","bytes":324,"sha256":"476afe58a7d98697e9f6edf94f6ac6d788d5447062d6a980a8ea5e552b5759ad"},"root":{"path":"docs/BANC_M3_CONSERVED_RESOURCE_REALLOCATION_ROOT_REFLECTED_20261008.md","bytes":9222,"sha256":"1694f6233a68123aa0f72ce3c5fcc043a1d11f975bd42b008c68be33d77853ed"},"roster":{"path":"verification/banc-natural-gait-m3-20260919/parent-contact-learning-qualification-roster-20261008-e.json","bytes":40437,"sha256":"be96770622f6fc0584c3f2ff427708f47aa852b7b3a083a3576ff11eb34fcba0"},"e":{"path":"verification/banc-natural-gait-m3-20260919/parent-contact-body-runtime-20261008-e.json","bytes":10332,"sha256":"4259bc545045b6b505291b37ca15e38e1ba8344f314f7292d014d254a06c1849"}}'|ConvertFrom-Json -AsHashtable
}
function Conserved-OwnerKeys {return @('e359_reservations','e360_reservations','e360_pending_actuals','e360_blocking_holds','direct_science_workers','direct_science_reservations','direct_science_pending_actuals','direct_science_blocking_holds','startup_ray_original_workers','startup_ray_original_reservations','startup_ray_original_pending_actuals','startup_ray_original_blocking_holds')}
function Conserved-Transfers {return @('direct_science_generated_transfer','direct_science_generated_additional_transfer','direct_science_generated_final_transfer','direct_science_generated_continuation_transfer','direct_science_generated_runtime_transfer','direct_science_generated_runtime_residual_transfer')}
function Conserved-TailKeys {return @((Conserved-OwnerKeys)+@('direct_science_generated_used_s','generated_stage_used_s','generated_shared_cumulative_used_s','generated_shared_remaining_s','original_cumulative_used_s','original_stage_used_s','startup_ray_original_local_used_s','original_remaining_s','debits','direct_science_calls','original_jobs_running','original_reserved_s','direct_science_qualification'))}
function Assert-LedgerOwner($owner){
  Need ($null -ne $owner -and [Object]::ReferenceEquals($owner,$script:ledgerTransaction) -and $owner.local_pid -eq $PID -and $owner.active -ceq $true) 'Actual current local metadata-writer capability'
  Need ($owner.creation -eq [Diagnostics.Process]::GetCurrentProcess().StartTime.ToUniversalTime().ToFileTimeUtc()) 'Same current invocation creation identity'
  Need ([Convert]::ToBase64String($owner.raw) -ceq [Convert]::ToBase64String([IO.File]::ReadAllBytes((Safe ($taskBase+'/parent-e358-original-active.lock'))))) 'Authenticate exact own token; never foreign supervisor reuse'
}
function Start-LedgerTransaction([string]$action){
  Need ($null -eq $script:ledgerTransaction) 'One metadata transaction per invocation'
  $token=@{schema='flylab.m3.direct-science.metadata-writer.v1';action=$action;owner_pid=[long]$PID;
    owner_creation_ticks=[long][Diagnostics.Process]::GetCurrentProcess().StartTime.ToUniversalTime().ToFileTimeUtc();invocation_id=[guid]::NewGuid().ToString('N')}
  $raw=$utf8.GetBytes(($token|ConvertTo-Json -Depth 10 -Compress)+[char]10);Need ($raw.Length -le 16384) 'Bounded new local metadata token'
  $stream=[IO.FileStream]::new((Safe ($taskBase+'/parent-e358-original-active.lock')),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
  try{$stream.Write($raw,0,$raw.Length);$stream.Flush($true)}finally{$stream.Dispose()}
  Need ([Convert]::ToBase64String($raw) -ceq [Convert]::ToBase64String([IO.File]::ReadAllBytes((Safe ($taskBase+'/parent-e358-original-active.lock'))))) 'Common exclusion first exact fsync readback'
  $owner=@{raw=$raw;local_pid=[long]$PID;creation=$token.owner_creation_ticks;active=$true;uncertain=$false;before=$null}
  $script:ledgerTransaction=$owner;Assert-LedgerOwner $owner
  $owner.before=[IO.File]::ReadAllBytes((Safe $ledgerName));Need ($owner.before.Length -le 524288) 'Fresh bounded ledger only after own exclusion'
  return $owner
}
function Stop-LedgerTransaction($owner){
  Assert-LedgerOwner $owner;Need ($owner.uncertain -ceq $false) 'Uncertain persistence retains hold/exclusion; no success'
  [IO.File]::Delete((Safe ($taskBase+'/parent-e358-original-active.lock')))
  Need (-not [IO.File]::Exists((Safe ($taskBase+'/parent-e358-original-active.lock')))) 'Exact locally owned metadata exclusion released'
  $owner.active=$false;$script:ledgerTransaction=$null
}
function Ledger-LockAbsentOrOwn {
  if(-not [IO.File]::Exists((Safe ($taskBase+'/parent-e358-original-active.lock')))){return $true}
  if($null -eq $script:ledgerTransaction){return $false}
  Assert-LedgerOwner $script:ledgerTransaction;return $true
}
function Publish-LedgerTransaction($value,$owner){
  Assert-LedgerOwner $owner;Need ($owner.uncertain -ceq $false) 'Known transaction only'
  $target=Safe $ledgerName;$before=$owner.before
  Need ([Convert]::ToBase64String($before) -ceq [Convert]::ToBase64String([IO.File]::ReadAllBytes($target))) 'Exact fresh raw-before comparison'
  $raw=$utf8.GetBytes(($value|ConvertTo-Json -Depth 100)+[char]10);Need ($raw.Length -le 524288) 'Bounded complete ledger mutation'
  $null=Conserved-Json $raw
  $temporary=Safe ($taskBase+'/parent-ledger-transaction-'+[guid]::NewGuid().ToString('N')+'.tmp')
  $stream=[IO.FileStream]::new($temporary,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
  try{$stream.Write($raw,0,$raw.Length);$stream.Flush($true)}finally{$stream.Dispose()}
  Need ([Convert]::ToBase64String($raw) -ceq [Convert]::ToBase64String([IO.File]::ReadAllBytes($temporary))) 'Full CreateNew temporary fsync/readback'
  Assert-LedgerOwner $owner
  Need ([Convert]::ToBase64String($before) -ceq [Convert]::ToBase64String([IO.File]::ReadAllBytes($target))) 'Same before bytes immediately before atomic replace'
  $backup=Safe ($taskBase+'/parent-ledger-before-'+[guid]::NewGuid().ToString('N')+'.json');Need (-not [IO.File]::Exists($backup)) 'New unique before backup'
  $owner.uncertain=$true
  [IO.File]::Replace($temporary,$target,$backup)
  Need ([Convert]::ToBase64String($raw) -ceq [Convert]::ToBase64String([IO.File]::ReadAllBytes($target))) 'One atomic publication exact full readback; no overwrite retry'
  Conserved-Same (Conserved-Json ([IO.File]::ReadAllBytes($target))) $value 'Typed full committed ledger'
  $pin=@{path=$ledgerName;bytes=[long]$raw.Length;sha256=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($raw)).ToLowerInvariant()}
  $owner.uncertain=$false;Stop-LedgerTransaction $owner
  return $pin
}
