# Private text metadata operands; shared core, Safe/Original readers stay unchanged.
function Conserved-TextPath([string]$name){
Need ($taskRoot -is [string] -and -not $taskRoot.StartsWith('\\',[StringComparison]::Ordinal) -and -not $taskRoot.StartsWith('//',[StringComparison]::Ordinal)) 'Normal trusted text root, no caller namespace/UNC'
$windows=[OperatingSystem]::IsWindows()
if($windows){Need ($taskRoot -cmatch '^[A-Za-z]:[\\/]' -and -not $taskRoot.Substring(2).Contains(':')) 'One local rooted drive';$rootParts=$taskRoot.Substring(3).Replace('\','/').Split('/')}
else{Need ($taskRoot.StartsWith('/',[StringComparison]::Ordinal) -and -not $taskRoot.Contains('\') -and -not $taskRoot.Contains(':')) 'Normal absolute POSIX text root';$rootParts=$taskRoot.Substring(1).Split('/')}
Need ($rootParts -cnotcontains '..' -and $rootParts -cnotcontains '.' -and -not $taskRoot.Contains([char]0)) 'No lexical root traversal'
Need ($name.Length -gt 0 -and $name.Length -le 4096 -and $name -cmatch '^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$') 'Literal text relative grammar before stat: no namespace/drive/UNC/ADS'
foreach($part in @($rootParts)+@($name.Split('/'))){Need ($part -cne '.' -and $part -cne '..' -and -not $part.EndsWith('.') -and $part -notmatch '^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$') 'No traversal/device/Win32 alias text operand'}
$normal=[IO.Path]::GetFullPath($taskRoot);$comparison=if($windows){[StringComparison]::OrdinalIgnoreCase}else{[StringComparison]::Ordinal}
$rootItem=[IO.DirectoryInfo]::new($normal);$ancestors=@();while($null -ne $rootItem){$ancestors=@($rootItem.FullName)+$ancestors;$rootItem=$rootItem.Parent}
foreach($ancestor in $ancestors){$operand=if($windows){'\\?\'+$ancestor}else{$ancestor};$attributes=[IO.File]::GetAttributes($operand);Need (($attributes -band [IO.FileAttributes]::Directory) -and -not ($attributes -band [IO.FileAttributes]::ReparsePoint)) 'Actual trusted text-root lstat, missing is not safe'}
$parts=$name.Split('/');$current=$normal
for($i=0;$i -lt $parts.Length;$i++){
$current=[IO.Path]::Combine($current,$parts[$i]);Need ($current.StartsWith($normal+[IO.Path]::DirectorySeparatorChar,$comparison)) 'Same lexical contained text namespace'
$operand=if($windows){'\\?\'+$current}else{$current};$attributes=[IO.File]::GetAttributes($operand)
Need (-not ($attributes -band [IO.FileAttributes]::ReparsePoint)) 'Real text lstat for every ancestor'
if($i -lt $parts.Length-1){Need ($attributes -band [IO.FileAttributes]::Directory) 'Actual text parent directory'}else{Need (-not ($attributes -band [IO.FileAttributes]::Directory)) 'Actual text leaf file'}
}
return $operand
}
function Conserved-TextStamp([string]$file){
$i=[IO.FileInfo]::new($file);$i.Refresh()
return @([long]$i.Length,[long]$i.LastWriteTimeUtc.Ticks,[long]$i.CreationTimeUtc.Ticks,[long]$i.Attributes)
}
function Conserved-TextRead([string]$file,[long]$cap){
$s=[IO.FileStream]::new($file,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read);$m=[IO.MemoryStream]::new();$buffer=[byte[]]::new(65536)
try{while(($n=$s.Read($buffer,0,[int][Math]::Min($buffer.Length,$cap+1-$m.Length))) -gt 0){$m.Write($buffer,0,$n);Need ($m.Length -le $cap) 'Bounded full text read, no race size expansion'};return ,$m.ToArray()}finally{$s.Dispose();$m.Dispose()}
}
function Conserved-TextRaw($spec,[long]$cap=1048576,[bool]$allowEmpty=$false){
Conserved-ExactKeys $spec @('path','bytes','sha256')
$minimum=if($allowEmpty){0}else{1}
Need (($spec.bytes -is [long] -or $spec.bytes -is [int]) -and $spec.bytes -ge $minimum -and $spec.bytes -le $cap -and $cap -gt 0 -and $cap -le 2097152 -and $spec.sha256 -is [string] -and $spec.sha256 -cmatch '^[0-9a-f]{64}$') 'Strict typed finite text pin'
$file=Conserved-TextPath $spec.path;$stamp=Conserved-TextStamp $file;Conserved-Same $stamp[0] ([long]$spec.bytes) 'Exact bounded text pre-read size'
$raw=Conserved-TextRead $file $cap;$null=$utf8.GetString($raw);$again=Conserved-TextRead $file $cap;$null=Conserved-TextPath $spec.path
Need ($raw.Length -eq $spec.bytes -and [Convert]::ToBase64String($raw) -ceq [Convert]::ToBase64String($again) -and [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($raw)).ToLowerInvariant() -ceq $spec.sha256) 'Full repeated text bytes/hash'
Conserved-Same (Conserved-TextStamp $file) $stamp 'Full text readback stamp'
return ,$raw
}
function Conserved-TextJson($spec,[long]$cap=1048576){return Conserved-Json (Conserved-TextRaw $spec $cap)}
function Conserved-TextPin([string]$name,[long]$cap=262144,[bool]$allowEmpty=$false){
$file=Conserved-TextPath $name;$stamp=Conserved-TextStamp $file;Need ($stamp[0] -le $cap -and ($stamp[0] -gt 0 -or $allowEmpty)) 'Finite text identity'
$raw=Conserved-TextRead $file $cap;$pin=@{path=$name;bytes=[long]$raw.Length;sha256=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($raw)).ToLowerInvariant()}
Conserved-Same (Conserved-TextStamp $file) $stamp 'Text identity stamp';$null=Conserved-TextRaw $pin $cap $allowEmpty;return $pin
}
function Conserved-SourceShape($spec,[string]$name){
Conserved-ExactKeys $spec @('path','bytes','sha256');Conserved-Same $spec.path $name 'Literal text Source key'
Need (($spec.bytes -is [long] -or $spec.bytes -is [int]) -and $spec.bytes -ge 0 -and $spec.bytes -le 2097152 -and $spec.sha256 -is [string] -and $spec.sha256 -cmatch '^[0-9a-f]{64}$' -and [IO.Path]::GetExtension($name) -cin @('.py','.ps1','.js','.json','.txt','.ini','.md','.xml')) 'Unchanged finite text Source descriptor'
}
function Conserved-Closed($l){
foreach($key in (Conserved-OwnerKeys)){Conserved-Same $l[$key] @{} ('Allocation empty '+$key)}
Conserved-Same $l.original_jobs_running ([long]0) 'No Original job';Conserved-Same $l.original_reserved_s ([long]0) 'No Original reservation'
foreach($entry in $l.direct_science_calls.GetEnumerator()){Conserved-Same $entry.Value.state 'charged' 'Every actual call charged once'}
}
function Conserved-Prefix($current,$before){
Need ($current.debits -is [array] -and $current.debits.Length -ge $before.debits.Length) 'Entire actual debit prefix'
Conserved-Same @($current.debits|Select-Object -First $before.debits.Length) @($before.debits) 'Every old actual debit/mode/fee'
foreach($entry in $before.direct_science_calls.GetEnumerator()){Conserved-Same $current.direct_science_calls[$entry.Key] $entry.Value 'Every old once-charged call'}
}
function Conserved-Totals($l,$p,$g,[long]$generatedCap,[long]$originalCap){
[double]$generated=0;[double]$original=0;$ids=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
for($i=0;$i -lt $l.debits.Length;$i++){
$d=$l.debits[$i];Need ($d.cost_id -is [string] -and $ids.Add($d.cost_id) -and $d.mode -cin @('generated','original')) 'No duplicate actual event or mode drift'
if(($i -ge 118 -and $d.mode -ceq 'generated') -or ($i -ge 165 -and $d.mode -ceq 'original')){
Need (($d.actual_outer_full_return_s -is [double] -or $d.actual_outer_full_return_s -is [long] -or $d.actual_outer_full_return_s -is [int]) -and [double]::IsFinite([double]$d.actual_outer_full_return_s) -and $d.actual_outer_full_return_s -ge 0) 'Typed finite actual fee, never bool'
if($d.mode -ceq 'generated'){$generated+=[double]$d.actual_outer_full_return_s}else{$original+=[double]$d.actual_outer_full_return_s}
}
}
[double]$stage=[double]$p.generated_stage_before_s+$generated
$values=@{direct_science_generated_used_s=$generated;generated_stage_used_s=$stage;generated_shared_cumulative_used_s=[double]$p.generated_shared_before_s+$generated;
generated_shared_remaining_s=[double]$generatedCap-[double]$p.generated_shared_baseline_s-$stage;
original_cumulative_used_s=[double]$g.original_cumulative_before_s+$original;original_stage_used_s=[double]$g.original_stage_before_s+$original;
startup_ray_original_local_used_s=[double]$g.science_local_used_before_s+$original;original_remaining_s=[double]$originalCap-[double]$g.original_cumulative_before_s-$original}
foreach($entry in $values.GetEnumerator()){Need ([double]::IsFinite($entry.Value) -and $entry.Value -ge 0) 'Finite nonnegative ordered totals';Conserved-Same $l[$entry.Key] $entry.Value ('Ordered binary64 '+$entry.Key)}
}
function Conserved-CostProof($cost,$debit){
return _Conserved-ActualCostProof $cost $debit 'generated'
}
function _Conserved-ActualCostProof($cost,$debit,[ValidateSet('generated','original','historical-bootstrap','historical-H-lint')][string]$family){
Need ($family -cin @('generated','original','historical-bootstrap','historical-H-lint')) 'Only fixed genuine cost envelopes'
$generated=$family -cne 'original';$timeout=$family -cin @('historical-bootstrap','historical-H-lint')
if($family -ceq 'historical-bootstrap'){Conserved-Same $cost (Conserved-TextJson (Conserved-F-Pins).failed.cost) 'Only immutable historical bootstrap';Conserved-Same $debit.cost_id 'direct-science-20261008-conserved-bootstrap' 'No H history alias'}
if($family -ceq 'historical-H-lint'){
$m=Conserved-I-Archives;$pins=@{};foreach($x in (Conserved-I-HPaths).GetEnumerator()){$pins[$x.Key]=$m[$x.Value].source}
Conserved-Same $debit @{cost_id='direct-science-20261008-contact-h-lint';mode='generated';actual_outer_full_return_s=[double]13.253;actual_returned_exit_code=[long]124;descriptor=$pins.cost} 'Only immutable historical H fee'
Conserved-Same $cost (Conserved-TextJson $pins.cost) 'Entire actual H saved cost';foreach($x in @('request','trace','outer','freeze')){$key=if($x -ceq 'freeze'){'source_freeze'}else{$x};Conserved-Same $cost[$key] $pins[$x] 'Exact actual H pins'}
}
Conserved-Same $cost.cost_id $debit.cost_id 'Fee/debit ID';Conserved-Same $cost.actual_outer_full_return_s $debit.actual_outer_full_return_s 'Fee/debit seconds';Conserved-Same $cost.exit_code $debit.actual_returned_exit_code 'Fee/debit actual exit'
Need ($cost.actual_outer_full_return_s -is [double] -and [double]::IsFinite($cost.actual_outer_full_return_s) -and $cost.actual_outer_full_return_s -gt 0) 'Genuine positive binary64 actual'
foreach($key in @('charged_once','all_owned_ended','reservation_released')){Conserved-Same $cost[$key] $true 'Actual known closed fee'}
Conserved-Same $cost.cap_exceeded $false 'Actual fee within cap'
$outer=Conserved-TextJson $cost.outer;$trace=Conserved-TextJson $cost.trace
if($family -ceq 'historical-H-lint'){
Need ($trace.dispatch_ms -is [long] -and $trace.first_final_ms -is [long]) 'Actual H Int64 clocks'
Conserved-Same $trace.dispatch_ms ([long]1791444054523) 'Actual H dispatch';Conserved-Same $trace.first_final_ms ([long]1791444067776) 'Actual H FINAL'
}
foreach($key in @('all_owned_ended','reservation_released','stdout_eof','stderr_eof')){Conserved-Same $outer[$key] $true 'Known complete actual owner'}
Conserved-Same $outer.unknown_owned_identities @{} 'No unknown owner';Conserved-Same $outer.timeout $timeout 'Exact fixed historical timeout or no current timeout';Conserved-Same $outer.exit_code $cost.exit_code 'Actual owner exit'
Need ($trace.schema -ceq 'flylab.m3.direct-science.genuine-return.v1' -and $trace.phase -ceq 'FINAL' -and $null -eq $trace.first_error -and ($trace.dispatch_ms -is [long] -or $trace.dispatch_ms -is [int]) -and ($trace.first_final_ms -is [long] -or $trace.first_final_ms -is [int]) -and $trace.first_final_ms -gt $trace.dispatch_ms) 'Real first FINAL clock'
if($family -cne 'generated'){Need ($trace.dispatch_ms -is [long] -and $trace.first_final_ms -is [long]) 'Preserved actual Contact Int64 FINAL clock'}
Conserved-Same ([double]($trace.first_final_ms-$trace.dispatch_ms)/1000.0) $cost.actual_outer_full_return_s 'Actual fee exact clock';Conserved-Same $trace.full_return_s $cost.actual_outer_full_return_s 'First FINAL cost'
$ownerSpec=if($generated){$cost.request}else{$cost.supervisor_request}
foreach($pair in @(@($trace.exit_code,$cost.exit_code),@($trace.source_freeze,$cost.source_freeze),@($trace.request,$ownerSpec),@($outer.request,$ownerSpec))){Conserved-Same $pair[0] $pair[1] 'Actual Source/owner/fee binding'}
$q=Conserved-TextJson $ownerSpec;Conserved-Same $q.call_id $cost.cost_id 'Actual admitted call';Conserved-Same $q.source_freeze $cost.source_freeze 'Actual admitted Source'
Need ($cost.actual_outer_full_return_s -le $q.limits.outer_s) 'Actual within admitted cap'
if($generated){Need ($q.mode -ceq 'generated' -and $q.role -cin @('lint','numerical','execution')) 'Real generated fee only';return $q}
Conserved-Same $q.science_request $cost.request 'Actual admitted Original science request';$request=Conserved-TextJson $cost.request
foreach($key in @('role','source_freeze','grant','inputs','roster','contract','local_debit_offset','resource_increment_s')){Conserved-Same $cost[$key] $request[$key] ('Actual Original cost/request '+$key)}
foreach($pair in @(@($request.call_id,$cost.cost_id),@($q.role,$request.role),@($q.grant,$request.grant),@($q.output,($request.output+'-supervision')))){Conserved-Same $pair[0] $pair[1] 'Same admitted Original call/role/J/output'}
foreach($key in @('outer_s','child_s','cleanup_s','host_bytes','output_cap_bytes')){Conserved-Same $q.limits[$key] $request.limits[$key] 'Actual Original fixed owner bounds'}
return $request
}
function Conserved-JUnit($spec,$rows){
$null=Conserved-ReadPinXml $spec
[xml]$xml=$utf8.GetString((Conserved-TextRaw $spec 1048576));$suites=@($xml.testsuites.testsuite);Need ($suites.Length -eq 1) 'One complete actual JUnit suite'
$suite=$suites[0];Need ($suite.failures -ceq '0' -and $suite.errors -ceq '0' -and $suite.skipped -ceq '0' -and $suite.tests -ceq [string]$rows.Length) 'No skip/error/failure or missing case'
$cases=@($suite.testcase);Need ($cases.Length -eq $rows.Length) 'Exact successful case count'
$seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
for($i=0;$i -lt $rows.Length;$i++){Conserved-Same ([string]$cases[$i].name) $rows[$i].name 'Every exact parameterized name';Conserved-Same ([string]$cases[$i].classname) $rows[$i].classname 'Every exact module/class';Need ($seen.Add($cases[$i].classname+'::'+$cases[$i].name) -and $cases[$i].ChildNodes.Count -eq 0) 'No duplicate or skipped/failed/error testcase'}
}
function Conserved-ReadPinXml($spec){
Conserved-ExactKeys $spec @('path','bytes','sha256');Need (($spec.bytes -is [long] -or $spec.bytes -is [int]) -and $spec.bytes -gt 0 -and $spec.bytes -le 1048576) 'Bounded actual XML'
$null=Conserved-TextRaw $spec 1048576;return $true
}
function Conserved-BootstrapSource($spec){
$f=Conserved-TextJson $spec 262144
Conserved-Same $f.schema 'flylab.m3.contact-learning.source-freeze.v1' 'Actual bootstrap freeze schema'
Conserved-Same $f.ROOT_review_cycle_complete $true 'Actual reviewed bootstrap Source'
Need ($f.sources -is [Collections.IDictionary]) 'Typed bootstrap Source dictionary'
foreach($name in (Conserved-BootstrapSources)){
Need ($f.sources.Contains($name)) ('Missing critical bootstrap Source '+$name)
Conserved-SourcePin $f.sources[$name] $name
}
# Dynamic cost-linked descriptors cover the embedded account definitions too.
# No own/future content hashes are embedded in this Source.
}
function Conserved-Bootstrap($spec,$before){
$p=Conserved-TextJson $spec;Conserved-ExactKeys $p @('schema','cost','source_freeze','fixture_node','junit')
Conserved-Same $p.schema 'flylab.m3.conserved-resource.bootstrap.v1' 'Actual bootstrap schema'
$node='tests/test_c_banc_contact_runtime_journal.py::test_gate_conserved_bootstrap_production_transaction_and_native_parity'
Conserved-Same $p.fixture_node $node 'One complete fixed meaningful bootstrap'
$d=$before.debits[-1];Conserved-Same $d.cost_id 'direct-science-20261008-conserved-bootstrap' 'Sole paid bootstrap';Conserved-Same $d.descriptor $p.cost 'Fresh actual before includes once fee'
$c=Conserved-TextJson $p.cost;$q=Conserved-CostProof $c $d;Conserved-Same $c.exit_code ([long]0) 'Real successful closed bootstrap';Conserved-Same $c.source_freeze $p.source_freeze 'Actual bootstrap Source'
Conserved-BootstrapSource $p.source_freeze # before any allocation/cap success
Conserved-Same $q.generated_argv @('-m','pytest',$node,'--tb=short','-q',('--junitxml='+$taskBase+'/parent-conserved-resource-bootstrap-junit.xml')) 'Full fixed bootstrap argv'
Conserved-Same $q.generated_behavior 'command' 'Generated temporary fixture only'
Conserved-Same $q.limits.outer_s ([long]11) 'Old budget outer11';Conserved-Same $q.limits.child_s ([long]8) 'Child8 same ENTRY';Conserved-Same $q.limits.cleanup_s ([long]3) 'Cleanup3 plus distinct close20'
Conserved-JUnit $p.junit @(@{classname='tests.test_c_banc_contact_runtime_journal';name=$node.Substring($node.IndexOf('::',[StringComparison]::Ordinal)+2)})
}
function Conserved-AllocationBefore($e){
Conserved-ExactKeys $e @('schema','event_id','before_ledger','historical_closed282','initial_policy','transfers','j_grant','direct_human_auto_authority','root_decision','old_caps','new_caps','delta_s','combined_global_s','new_resource_grant_s','bootstrap')
Conserved-Same $e.schema 'flylab.m3.conserved-resource-allocation.v1' 'One explicit cap epoch';Conserved-Same $e.event_id 'conserved-resource-20261008-600' 'One event ID'
$pins=Conserved-Pins
foreach($pair in @(@('historical_closed282','historical'),@('initial_policy','policy'),@('j_grant','j'),@('direct_human_auto_authority','human'))){Conserved-Same $e[$pair[0]] $pins[$pair[1]] 'Immutable actual legacy/authority pin'}
Conserved-Same $e.old_caps @{generated_global=[long]1700;generated_local=[long]964;original_global=[long]16200} 'Old paired caps'
Conserved-Same $e.new_caps @{generated_global=[long]2300;generated_local=[long]1564;original_global=[long]15600} 'New paired caps'
Conserved-Same $e.delta_s @{generated_global=[long]600;generated_local=[long]600;original_global=[long]-600} 'No new global resource'
Conserved-Same $e.combined_global_s ([long]17900) 'Conserved combined integer caps';Conserved-Same $e.new_resource_grant_s ([long]0) 'No resource grant'
Conserved-Same $e.root_decision @{source=$pins.root;pool_transfer_s=[long]600;human_numerical_approval=$false} 'ROOT600 prospective choice only'
Conserved-Same (Pin $pins.root.path) $pins.root 'Exact reflected ROOT Source'
$a=Conserved-TextJson $pins.human;Conserved-Same $a @{schema='flylab.m3.contact-metadata.current-human-autonomy.v1';direct_human_authority=$true;automatic_authority=$true;new_resource_grant_s=[long]0;literal_latest_user='앞으로 내 승인 받지 말고 스스로 판단해서 진행해줘. 개입할 상황이면 내가 알아서 개입할게'} 'Actual literal autonomy'
$before=Conserved-TextJson $e.before_ledger 524288;$historical=Conserved-TextJson $pins.historical 524288
Need (-not $before.Contains('direct_science_conserved_allocation') -and (Generated-Budget $before) -eq 964) 'Real legacy before validated under old caps; never reset-cap clone'
Conserved-Closed $before;Conserved-Prefix $before $historical
Need ($before.debits.Length -eq $historical.debits.Length+1 -and (Conserved-Count $before.direct_science_calls) -eq (Conserved-Count $historical.direct_science_calls)+1) 'Fresh sole paid bootstrap tail; historical282 retained'
$tail=Conserved-TailKeys;foreach($entry in $historical.GetEnumerator()){if($entry.Key -cnotin $tail){Conserved-Same $before[$entry.Key] $entry.Value 'Immutable historical before field'}}
foreach($pair in @(@('generated_shared_cumulative_limit_s',1700),@('direct_science_generated_limit_s',964),@('original_cumulative_limit_s',16200))){Conserved-Same $before[$pair[0]] ([long]$pair[1]) 'Exact old paired live caps'}
Conserved-Same $before.direct_science_policy $pins.policy 'Immutable actual policy'
$transfers=@{};foreach($key in (Conserved-Transfers)){$transfers[$key]=$before[$key]};Conserved-Same $e.transfers $transfers 'All six exact historical event descriptors'
Conserved-Same $before.startup_ray_original_grants['startup-ray-original-20261007-j'].grant $pins.j 'Registered J unchanged16200/6300/6000'
$policy=Conserved-TextJson $pins.policy;$grant=Conserved-TextJson $pins.j;Conserved-Totals $before $policy $grant 1700 16200;Conserved-Bootstrap $e.bootstrap $before
return @{before=$before;policy=$policy;grant=$grant}
}
function Conserved-AllocatedSnapshot($before,$spec,$p,$g){
$after=Conserved-Json ($before|ConvertTo-Json -Depth 100)
$after.generated_shared_cumulative_limit_s=[long]2300;$after.direct_science_generated_limit_s=[long]1564;$after.original_cumulative_limit_s=[long]15600
$after.generated_shared_remaining_s=2300.0-[double]$p.generated_shared_baseline_s-[double]$before.generated_stage_used_s
[double]$total=0;foreach($d in $before.debits|Select-Object -Skip 165){if($d.mode -ceq 'original'){$total+=[double]$d.actual_outer_full_return_s}}
$after.original_remaining_s=15600.0-[double]$g.original_cumulative_before_s-$total;$after.direct_science_conserved_allocation=$spec
return $after
}
function Conserved-ActiveEventPaths {return @(($taskBase+'/parent-conserved-resource-allocation-20261008-a.json'),($taskBase+'/parent-conserved-resource-prequalification-20261008-f-fixed.json'))}
function Conserved-F-ForbiddenSources {return @(($taskBase+'/parent-conserved-resource-prequalification-reviewed-source-20261008-f-fixed.json'),($taskBase+'/parent-conserved-resource-prequalification-20261008-f-fixed.json'),($taskBase+'/parent-contact-learning-source-freeze-20261008-f-fixed.json'),($taskBase+'/parent-contact-learning-qualified-20261008-f-fixed.json'))}
function Conserved-HistoricalWriterSources {return @(($taskBase+'/parent-direct-science-account.ps1'),'tools/banc_m3_conserved_resource_native_core.ps1','tools/banc_m3_conserved_resource_native_epoch.ps1')}
function Conserved-F-FixupSources {return @('tools/banc_m3_conserved_resource.py','tools/banc_m3_conserved_resource_native_epoch.ps1',($taskBase+'/parent-direct-science-account.ps1'),($taskBase+'/parent-contact-learning-contract-native.ps1'),'tests/test_c_banc_contact_runtime_journal.py')}
function Conserved-Caps($l){
if($l.Contains('direct_science_qualification_lint_cleanup')){return Conserved-J-Caps $l}
if($l.Contains('direct_science_preflight_recovery')){return Conserved-I-Caps $l}
if($l.Contains('direct_science_qualification_lint_repair')){return Conserved-H-Caps $l}
if($l.Contains('direct_science_qualification_amendment')){return Conserved-G-Caps $l}
if(-not $l.Contains('direct_science_conserved_allocation')){
Need ($l.generated_shared_cumulative_limit_s -ne 2300 -and $l.direct_science_generated_limit_s -ne 1564 -and $l.original_cumulative_limit_s -ne 15600) 'New/mixed caps require exact event'
return @{generated_global=[long]1700;generated_local=[long]964;original_global=[long]16200}
}
$spec=$l.direct_science_conserved_allocation;Need ($spec.path -cin @(Conserved-ActiveEventPaths)) 'Exactly two immutable event routes'
$e=Conserved-TextJson $spec;$data=Conserved-RouteBefore $spec $e
$expected=Conserved-AllocatedSnapshot $data.before $spec $data.policy $data.grant;Conserved-Prefix $l $data.before
$names=@(Conserved-Keys $expected);Conserved-ExactKeys $l $names;$tail=Conserved-TailKeys
foreach($entry in $expected.GetEnumerator()){if($entry.Key -cnotin $tail){Conserved-Same $l[$entry.Key] $entry.Value 'Immutable allocated history/policy/J/live caps'}}
Conserved-Totals $l $data.policy $data.grant 2300 15600
return @{generated_global=[long]2300;generated_local=[long]1564;original_global=[long]15600}
}
function Contact-E-GeneratedPlan([string]$role){
Need ($role -cin @('lint','math','gate-a','gate-b')) 'Four distinct actual qualification roles'
$pins=Conserved-Pins;$r=Conserved-TextJson $pins.roster
Conserved-Same $r.epoch 'contact-conserved-20261008-e' 'Fixed complete Source roster'
$roles=@('lint','math','gate-a','gate-b');$i=[array]::IndexOf($roles,$role);$caps=@(@(15,10,5),@(25,20,5),@(120,110,10),@(120,110,10))
if($role -ceq 'lint'){
$argv=@('-m','ruff','check','tools/banc_m3_contact_learning.py','tools/verify_c_banc_m3_contact_learning.py','tools/analyze_c_banc_m3_contact_learning.py','tools/banc_m3_contact_learning_admission.py','tools/banc_m3_startup_ray_supervisor.py','tools/verify_c_banc_m3_startup_ray.py','tools/banc_m3_contact_runtime_journal.py','tools/banc_m3_conserved_resource.py','tests/test_c_banc_contact_runtime_journal.py','tests/test_c_banc_contact_learning.py','tests/test_c_banc_contact_metadata.py','tests/test_c_banc_startup_ray_execution.py','tests/test_c_banc_contact_body_retry.py')
}else{$kind=$role.Replace('-','_');$argv=@('-m','pytest')+@($r[$kind]|ForEach-Object{$_.node_id})+@('--tb=short','-q',('--junitxml='+$taskBase+'/parent-contact-learning-e-'+$role+'-junit.xml'))}
return @{call='direct-science-20261008-contact-e-'+$role;role=@('lint','numerical','execution','execution')[$i];argv=$argv;outer=$caps[$i][0];child=$caps[$i][1];cleanup=$caps[$i][2]}
}
function Conserved-Qualification($q,$l){
if($l.Contains('direct_science_qualification_lint_cleanup')){return Conserved-J-Qualification $q $l}
if($l.Contains('direct_science_preflight_recovery')){return Conserved-I-Qualification $q $l}
if($l.Contains('direct_science_qualification_lint_repair')){return Conserved-H-Qualification $q $l}
if($l.Contains('direct_science_qualification_amendment')){return Conserved-G-Qualification $q $l}
if($l.Contains('direct_science_conserved_allocation') -and $l.direct_science_conserved_allocation.path -ceq ($taskBase+'/parent-conserved-resource-prequalification-20261008-f-fixed.json')){return Conserved-F-Qualification $q $l}
$pins=Conserved-Pins;$caps=Conserved-Caps $l;Conserved-Same $caps.generated_global ([long]2300) 'Real paired allocation required'
Conserved-Same $q.schema 'flylab.m3.direct-science.qualification.v2' 'Four-cost/three-JUnit new epoch';Conserved-Same $q.epoch 'contact-conserved-20261008-e' 'Exact E epoch';Conserved-Same $q.old_E360 'UNPASSED' 'Old failure unchanged'
Conserved-Same $q.allocation $l.direct_science_conserved_allocation 'Same real allocation';Conserved-Same $q.qualification_roster $pins.roster 'All complete fixed node IDs'
Conserved-Same $q.source_freeze.path ($taskBase+'/parent-contact-learning-source-freeze-20261008-e.json') 'New full same Source'
$freeze=Conserved-TextJson $q.source_freeze;Conserved-Same $freeze.ROOT_review_cycle_complete $true 'Actual full review'
Conserved-BootstrapSource $q.source_freeze
foreach($name in @((Conserved-BootstrapSources)+@($pins.roster.path,$pins.e.path,'tests/test_c_banc_contact_body_retry.py','tests/test_c_banc_startup_ray_execution.py'))){Need ($freeze.sources.Contains($name)) 'All new Source frozen';Conserved-Same (Pin $name) $freeze.sources[$name] 'Exact current Source'}
Conserved-Same $freeze.sources[$pins.roster.path] $pins.roster 'Fixed complete roster in Source'
$e=Conserved-TextJson $l.direct_science_conserved_allocation;$anchor=Conserved-TextJson $e.before_ledger 524288;[long]$offset=$anchor.debits.Length
Need ($q.actual_costs -is [array] -and $q.actual_costs.Length -eq 4) 'Every four genuine actual fees'
for($i=0;$i -lt 4;$i++){
$selectedRole=@('lint','math','gate-a','gate-b')[$i];$plan=Contact-E-GeneratedPlan $selectedRole;$d=$l.debits[$offset+$i]
Conserved-Same $d.cost_id $plan.call 'Ordered sole four actual calls';Conserved-Same $d.descriptor $q.actual_costs[$i] 'Actual once fee descriptor'
$cost=Conserved-TextJson $d.descriptor;$request=Conserved-CostProof $cost $d;Conserved-Same $cost.exit_code ([long]0) 'Actual successful role';Conserved-Same $cost.source_freeze $q.source_freeze 'All four full same Source'
Conserved-Same $request.role $plan.role 'Fixed qualification role';Conserved-Same $request.generated_argv $plan.argv 'Entire exact modules/nodes; no permissive -k'
foreach($key in @('outer_s','child_s','cleanup_s')){$limit=if($key -ceq 'outer_s'){$plan.outer}elseif($key -ceq 'child_s'){$plan.child}else{$plan.cleanup};Conserved-Same $request.limits[$key] ([long]$limit) 'Fixed real caps'}
Conserved-Same $l.direct_science_calls[$cost.cost_id] @{state='charged';mode='generated';request=$cost.request;reserved_s=[long]$plan.outer} 'All four known closed charged'
}
Need ($q.tests -is [array] -and $q.tests.Length -eq 3) 'All math/gate-A/gate-B actual JUnits'
$r=Conserved-TextJson $pins.roster
for($i=0;$i -lt 3;$i++){
$role=@('math','gate-a','gate-b')[$i];$rows=@($r[$role.Replace('-','_')]);$item=$q.tests[$i]
Conserved-Same $item.file.path ($taskBase+'/parent-contact-learning-e-'+$role+'-junit.xml') 'Fixed actual complete JUnit'
Conserved-Same $item.node_ids @($rows|ForEach-Object{$_.node_id}) 'Complete actual parameterized IDs';Conserved-Same $item.names @($rows|ForEach-Object{$_.name}) 'Complete actual names';Conserved-Same $item.classnames @($rows|ForEach-Object{$_.classname}) 'Complete module/classes';Conserved-Same $item.count ([long]$rows.Length) 'Full count'
Conserved-JUnit $item.file $rows
}
Need ($q.evidence -is [array] -and $q.evidence.Length -ge 4) 'Every actual role observation; no shard alone qualifies';foreach($spec in $q.evidence){$null=Conserved-TextJson $spec 2097152}
return [long]($offset+4)
}

function Conserved-F-Pins {return '{"failed":{"outer":{"path":"verification/banc-natural-gait-m3-20260919/parent-direct-science-20261008-conserved-bootstrap/outer.json","sha256":"673c634d9b1d01a60ad94a8d8442ffd7de45e9fbdbcae31ece280b881a311757","bytes":1547},"trace":{"path":"verification/banc-natural-gait-m3-20260919/parent-direct-science-20261008-conserved-bootstrap-genuine-trace.json","sha256":"b8528a3f5202e9c6147677829ce56f9fbf581d891f259fb83a65fdeb2f7e63b0","bytes":3078},"request":{"path":"verification/banc-natural-gait-m3-20260919/parent-direct-science-20261008-conserved-bootstrap-request.json","sha256":"115974fdaec579e0388aa222c287b3c0e7a32c2e8f0b7181fa7958196cbd8354","bytes":1231},"native_observation":{"path":"verification/banc-natural-gait-m3-20260919/parent-conserved-bootstrap-actual-failure-20261008/generated-conserved-native-observation.json","sha256":"daa175989a4983226ed4157e0c1c80bf9cfd2e424e9cdefd3e4b4450325ee1f2","bytes":740},"junit":{"path":"verification/banc-natural-gait-m3-20260919/parent-conserved-resource-bootstrap-junit.xml","sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","bytes":0},"cost":{"path":"verification/banc-natural-gait-m3-20260919/parent-direct-science-20261008-conserved-bootstrap-full-return.json","sha256":"cd04912465bad2d267928d8bedf0d446bc51ac5264dedbe4f49d83fe79c6ff65","bytes":1304},"source_freeze":{"path":"verification/banc-natural-gait-m3-20260919/parent-contact-learning-source-freeze-20261008-e.json","sha256":"a0b5c0e1d7f77708754e303dfa7450f1c5c89120bd12ec6663bfe7cb1bcaaebc","bytes":249298}},"addendum":{"path":"docs/BANC_M3_POST_BOOTSTRAP_RESOURCE_ROUTE_ROOT_DEPENDENCY_ADDENDUM_20261008.md","sha256":"20af9d1e74933c393bea2eaabb8ed13a7ea7cd9713e38187691975072fe226bb","bytes":1819},"astra":{"json":{"path":"verification/banc-natural-gait-m3-20260919/parent-post-bootstrap-resource-route-astra-review-20261008/review.json","sha256":"5b26e6e749da601c9937f175e9904a9ad6c17af2e873b3a599a9e743dd7911d0","bytes":12894},"markdown":{"path":"verification/banc-natural-gait-m3-20260919/parent-post-bootstrap-resource-route-astra-review-20261008/review.md","sha256":"82295c54781ddde5150d0c109e49b99ac85cc3d8936e2e8d2b479c4a3ed28da9","bytes":11018}},"root":{"path":"docs/BANC_M3_POST_BOOTSTRAP_RESOURCE_ROUTE_ROOT_REFLECTED_20261008.md","sha256":"1a36744c198707ede56cefde3b393d60c7d3dfef4ddffbd0fb17a49d32c30ba0","bytes":11137},"closed":{"path":"verification/banc-natural-gait-m3-20260919/parent-conserved-resource-prequalification-closed283-20261008-f.json","sha256":"40c35c8f08f65a036d678db9d562385425068c04e89bbae2367389cca667c7a2","bytes":454791},"body":{"path":"verification/banc-natural-gait-m3-20260919/parent-contact-body-runtime-20261008-f-fixed.json","sha256":"43c59780c88db868befa19c132087302007673c1c2cecae02d7526bf94cbb537","bytes":11151},"admin":{"path":"verification/banc-natural-gait-m3-20260919/parent-conserved-bootstrap-administrative-observations-20261008.json","sha256":"4810e9b55110553ce2877208ea4d381845021263bd25249034ae9c37545d149a","bytes":2307},"source_before":{"path":"verification/banc-natural-gait-m3-20260919/parent-post-bootstrap-resource-route-ROOT-review-20261008/source-before-valid.json","sha256":"ebfd5d975947f47766f17def16c4835f568707aca00c98f91a4ca82dd0d8b97e","bytes":5381},"original_body":{"path":"verification/banc-natural-gait-m3-20260919/parent-contact-body-runtime-20261008-f.json","sha256":"3266e2fda20702fc0a182269cb8edb8aaa3e08894ddb844de9cb3676ca269ac7","bytes":11133},"original_review":{"path":"verification/banc-natural-gait-m3-20260919/parent-conserved-resource-prequalification-reviewed-source-20261008-f.json","sha256":"7d3d18610e821e512e4618c7048f7ab2e87c01786e4dc80b9d58ba605140e4cc","bytes":205973},"original_event":{"path":"verification/banc-natural-gait-m3-20260919/parent-conserved-resource-prequalification-20261008-f.json","sha256":"49617e5236a11b8c367bba1c5ba0dcb0f09051ba02d361139bf744ea87f7e012","bytes":17051},"refusal":{"path":"verification/banc-natural-gait-m3-20260919/parent-post-bootstrap-resource-route-administrative-allocation-refusal-20261008-f.json","sha256":"f1435c615c975eb58902f39a1886c30dfd414e031360d37cc176583560f7fcc4","bytes":2307},"adoption":{"path":"verification/banc-natural-gait-m3-20260919/parent-post-bootstrap-resource-route-ROOT-residual-20261008/ROOT-source-adoption.json","sha256":"ff6dd02cfcbc59377fdddb51f4efc7a1f44adedf50c7407f8af1eeab5afd0647","bytes":2565},"fixup_design":{"path":"docs/BANC_M3_POST_BOOTSTRAP_RESOURCE_ROUTE_ROOT_FIXUP_REFLECTED_20261008.md","sha256":"34d4689ee796c50b8dac323a2b833fceceaa53529c6f095c42a30339276f4988","bytes":5965},"fixup_before":{"path":"verification/banc-natural-gait-m3-20260919/parent-post-bootstrap-root-fixup-ROOT-before-20261008/source-before.json","sha256":"2123376c2b03589ee964688c34a525a01a75fe057d98ec3479af618ab0fa0fa1","bytes":6778},"fixup_astra":{"markdown":{"path":"verification/banc-natural-gait-m3-20260919/parent-post-bootstrap-root-fixup-astra-review-20261008/review.md","sha256":"e88c28a5f4dbc3eb672ab1c078b2ab65e4ce5e6721e63ffebe01b2dc8c9d5b4a","bytes":11427},"json":{"path":"verification/banc-natural-gait-m3-20260919/parent-post-bootstrap-root-fixup-astra-review-20261008/review.json","sha256":"4dbc6e0c191e2f839cac7d70ab48248e2d6861cd7db18f8c6fc914a97e8910af","bytes":22166}},"fixup_report_path":"docs/BANC_M3_POST_BOOTSTRAP_RESOURCE_ROUTE_ROOT_FIXUP_20261008.md"}'|ConvertFrom-Json -AsHashtable}

function Conserved-F-Text($spec){
Conserved-ExactKeys $spec @('path','bytes','sha256')
Need ($spec.path -is [string] -and ($spec.bytes -is [long] -or $spec.bytes -is [int]) -and $spec.bytes -ge 0 -and $spec.bytes -le 2097152 -and $spec.sha256 -is [string] -and $spec.sha256 -cmatch '^[0-9a-f]{64}$') 'Typed bounded text Source descriptor'
$extensions=@('.py','.ps1','.js','.json','.txt','.ini','.md','.xml');Need ([IO.Path]::GetExtension($spec.path) -cin $extensions) 'Text only; no Original arrays'
if([IO.Path]::IsPathRooted($spec.path)){
$file=[IO.Path]::GetFullPath($spec.path);$item=[IO.FileInfo]$file
while($null -ne $item){if($item.Exists){Need (-not ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) 'No foreign Source reparse'};if($item -is [IO.DirectoryInfo]){$item=$item.Parent}else{$item=$item.Directory}}
}else{$null=Conserved-TextRaw $spec 2097152 $true;return}
$stamp=([IO.FileInfo]$file).LastWriteTimeUtc.Ticks;Need (([IO.FileInfo]$file).Length -eq $spec.bytes) 'Exact text Source bytes'
$raw=[IO.File]::ReadAllBytes($file);$null=$utf8.GetString($raw)
Need ($raw.Length -eq $spec.bytes -and [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($raw)).ToLowerInvariant() -ceq $spec.sha256 -and
[Convert]::ToBase64String($raw) -ceq [Convert]::ToBase64String([IO.File]::ReadAllBytes($file)) -and ([IO.FileInfo]$file).LastWriteTimeUtc.Ticks -eq $stamp) 'Full cold Source hash/readback/stamp'
}
function Conserved-F-Sources {return @((Conserved-BootstrapSources)+@('tools/banc_m3_contact_learning_admission.py',($taskBase+'/parent-contact-learning-contract-native.ps1'),'tools/verify_c_banc_m3_contact_learning.py','tools/verify_c_banc_m3_startup_ray.py','tools/banc_m3_contact_runtime_journal.py','tests/test_c_banc_contact_body_retry.py','tests/test_c_banc_startup_ray_execution.py',($taskBase+'/parent-contact-learning-qualification-roster-20261008-e.json'),($taskBase+'/parent-contact-body-runtime-20261008-f-fixed.json')))}
function Contact-F-GeneratedPlan([string]$role){
$old=Contact-E-GeneratedPlan $role;$i=[array]::IndexOf(@('lint','math','gate-a','gate-b'),$role);$argv=@($old.argv)
if($role -cne 'lint'){$argv[-1]='--junitxml='+$taskBase+'/parent-contact-learning-f-'+$role+'-junit.xml'}
$call='direct-science-20261008-contact-f-'+$role
return @{call=$call;role=$old.role;argv=$argv;limits=@{outer_s=[long]$old.outer;child_s=[long]$old.child;cleanup_s=[long]$old.cleanup;host_bytes=[long]2147483648;output_cap_bytes=[long]8388608};output=$taskBase+'/parent-'+$call}
}
function Conserved-F-Plan {
$pins=Conserved-Pins;$calls=@();foreach($role in @('lint','math','gate-a','gate-b')){$calls+=Contact-F-GeneratedPlan $role}
return @{epoch='contact-prequalification-20261008-f';roster_epoch='contact-conserved-20261008-e';qualification_roster=$pins.roster;calls=$calls;generated_close_s=[long]20;original_release_calls=@('contact-learning-20261008-f-body','contact-learning-20261008-f-body-audit')}
}
function Conserved-F-Fixup($p){
$pins=Conserved-F-Pins;Conserved-ExactKeys $p @('schema','design','astra_review','before_source','root_fixup')
Conserved-Same $p.schema 'flylab.m3.post-bootstrap-root-fixup.provenance.v1' 'Separate fixup provenance'
Conserved-Same $p.design $pins.fixup_design 'Actual ROOT reflection';Conserved-F-Text $p.design
Conserved-Same $p.astra_review $pins.fixup_astra 'Actual auxiliary Astra reports';foreach($item in $p.astra_review.Values){Conserved-F-Text $item}
Conserved-Same $p.before_source $pins.fixup_before 'Distinct before14, never historical13'
$before=Conserved-TextJson $p.before_source;Conserved-Same $before.schema 'flylab.m3.post-bootstrap-root-fixup.source-before.v1' 'Actual before14 schema'
Conserved-Same $before.source_metadata_only $true 'Source archive only';Conserved-Same $before.scientific_execution 'NOT_RUN' 'No conformance receipt';Conserved-Same $before.allocation_committed $false 'Original F unallocated'
Need ($before.snapshots -is [array] -and $before.snapshots.Length -eq 14) 'Full separate before14'
$seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach($item in $before.snapshots){
Conserved-ExactKeys $item @('source','archive');Need ($seen.Add($item.source.path)) 'Unique before14 Source'
$arch=@{path=$taskBase+'/parent-post-bootstrap-root-fixup-ROOT-before-20261008/before/'+$item.source.path;bytes=$item.source.bytes;sha256=$item.source.sha256}
Conserved-Same $item.archive $arch 'Entire exact archive';Conserved-F-Text $item.archive
}
foreach($name in (Conserved-F-FixupSources)){Need ($seen.Contains($name)) 'All five current Sources archived'}
Conserved-Same $p.root_fixup.path $pins.fixup_report_path 'Only final ROOT report path'
Need ($p.root_fixup.bytes -le 65536) 'Bounded dynamic final ROOT report';Conserved-F-Text $p.root_fixup
}
function Conserved-F-Refusal {
$pins=Conserved-F-Pins;$r=Conserved-TextJson $pins.refusal
Conserved-Same $r.schema 'flylab.m3.post-bootstrap-resource-route.administrative-allocation-refusal.v1' 'Actual failed administration'
Conserved-Same $r.actual_action 'conserved-allocation' 'Same action';Conserved-Same $r.actual_event $pins.original_event 'Original unallocated F event'
Conserved-Same $r.actual_source_adoption $pins.adoption 'Actual Source adoption';$null=Conserved-TextJson $pins.adoption
Conserved-Same $r.actual_tool_first_final.exit_code ([long]1) 'Actual first FINAL failed'
Need (-not $r.actual_tool_first_final.Contains('session_id') -and $r.actual_tool_first_final.output.Contains('Exactly two schema-bound prospective routes')) 'Known final refusal, never ongoing/success'
Conserved-Same $r.observed_unchanged_ledger @{path=$taskBase+'/parent-e358-execution-ledger.json';bytes=$pins.closed.bytes;sha256=$pins.closed.sha256} 'Actual after-read closed283'
foreach($pair in @(@('observed_allocation_present',$false),@('observed_shared_lock_absent',$true),@('debits',[long]283),@('direct_calls',[long]165),@('scientific_execution','NOT_RUN'),@('qualification','NOT_RUN'),@('original','NOT_RUN'),@('scientific_cost_created',$false),@('scientific_fees_rewritten',$false))){Conserved-Same $r[$pair[0]] $pair[1] 'Actual administrative refusal facts'}
}
function Conserved-F-ReviewMetadata($spec){
$pins=Conserved-F-Pins;Conserved-Same $spec.path ($taskBase+'/parent-conserved-resource-prequalification-reviewed-source-20261008-f-fixed.json') 'Single reviewed F Source vector'
$r=Conserved-TextJson $spec
Conserved-ExactKeys $r @('schema','epoch','sources','root_reflection','dependency_addendum','astra_review','root_reviews','fixup_provenance')
Conserved-Same $r.schema 'flylab.m3.conserved-resource-prequalification.reviewed-source.v2' 'Fixed Source review schema';Conserved-Same $r.epoch 'contact-prequalification-20261008-f' 'Fresh F review epoch'
foreach($pair in @(@('root_reflection','root'),@('dependency_addendum','addendum'))){Conserved-Same $r[$pair[0]] $pins[$pair[1]] 'Current ROOT binding';Conserved-F-Text $pins[$pair[1]]}
Conserved-Same $r.astra_review $pins.astra 'Actual Astra review';foreach($entry in $pins.astra.GetEnumerator()){Conserved-F-Text $entry.Value}
Need ($r.root_reviews -is [array] -and $r.root_reviews.Length -ge 1 -and $r.root_reviews.Length -le 3) 'Actual final ROOT provenance'
$allowed=@('FIRST_REVIEW','SECOND_REVIEW','RESIDUAL')|ForEach-Object{'docs/BANC_M3_POST_BOOTSTRAP_RESOURCE_ROUTE_ROOT_'+$_+'_20261008.md'}
$seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach($p in $r.root_reviews){Need ($p.path -cin $allowed -and $seen.Add($p.path)) 'No foreign/duplicate ROOT review';Conserved-F-Text $p}
Need ($seen.Contains('docs/BANC_M3_POST_BOOTSTRAP_RESOURCE_ROUTE_ROOT_FIRST_REVIEW_20261008.md')) 'Actual new ROOT FIRST'
$original=Conserved-TextJson $pins.original_review # historical metadata, never cold-read its superseded Source
Conserved-Same $original.schema 'flylab.m3.conserved-resource-prequalification.reviewed-source.v1' 'Immutable original F manifest'
Conserved-Same $r.root_reviews $original.root_reviews 'All original actual ROOT provenance'
Need ($original.sources -is [Collections.IDictionary] -and $original.sources.Count -eq 722) 'Entire inherited F722'
Conserved-F-Fixup $r.fixup_provenance
Need ($r.sources -is [Collections.IDictionary]) 'Typed current Source vector'
Conserved-ExactKeys $r.sources @((Conserved-Keys $original.sources)+@($pins.body.path))
foreach($entry in $original.sources.GetEnumerator()){if($entry.Key -cnotin @(Conserved-F-FixupSources)){Conserved-Same $r.sources[$entry.Key] $entry.Value 'Unrelated inherited Source pin retained'}}
foreach($name in (Conserved-F-Sources)){Need ($r.sources.Contains($name)) 'Complete production/current fixture Source'}
$forbidden=@(Conserved-F-ForbiddenSources)
foreach($entry in $r.sources.GetEnumerator()){
Need ($entry.Key -is [string] -and $entry.Key -cnotin $forbidden -and $entry.Key -cnotmatch 'direct-science-20261008-contact-f-|contact-learning-20261008-f-body') 'No own/event/future child Source hash cycle'
Conserved-Same $entry.Value.path $entry.Key 'Source vector exact path';Conserved-SourceShape $entry.Value $entry.Key
}
$old=Conserved-Pins;Conserved-Same $r.sources[$old.roster.path] $old.roster 'Unchanged complete E roster';Conserved-Same $r.sources[$pins.original_body.path] $pins.original_body 'Original static F retained';Conserved-Same $r.sources[$pins.body.path] $pins.body 'One fixed static F body scope'
return $r
}
function Conserved-F-FailedHistory($before){
$pins=Conserved-F-Pins;$d=$before.debits[-1]
Conserved-Same $d.cost_id 'direct-science-20261008-conserved-bootstrap' 'Spent bootstrap retained';Conserved-Same $d.descriptor $pins.failed.cost 'Actual failed cost'
$c=Conserved-TextJson $pins.failed.cost;$q=_Conserved-ActualCostProof $c $d 'historical-bootstrap'
Conserved-Same $c.exit_code ([long]124) 'V1 failure stays124';Conserved-Same $c.actual_outer_full_return_s ([double]9.208) 'Full once-paid failed fee'
foreach($key in @('request','outer','trace','source_freeze')){Conserved-Same $c[$key] $pins.failed[$key] 'Immutable failed evidence'}
Conserved-Same $q.call_id 'direct-science-20261008-conserved-bootstrap' 'No spent-call alias'
Conserved-Same $q.generated_argv @('-m','pytest','tests/test_c_banc_contact_runtime_journal.py::test_gate_conserved_bootstrap_production_transaction_and_native_parity','--tb=short','-q',('--junitxml='+$pins.failed.junit.path)) 'Whole spent fixture'
Conserved-Same $q.limits @{outer_s=[long]11;child_s=[long]8;cleanup_s=[long]3;host_bytes=[long]2147483648;output_cap_bytes=[long]8388608} 'Actual failed envelope'
Conserved-Same $q.source_freeze $pins.failed.source_freeze 'As-executed E immutable'
$f=Conserved-TextJson $pins.failed.source_freeze;$n=Conserved-TextJson $pins.failed.native_observation
foreach($key in @('returncode','worker_pid','worker_creation_ticks')){Conserved-Same $n[$key] $null 'Native NOT_STARTED'}
Conserved-Same $n.pre_spawn_refused $true 'Actual parser refusal';Conserved-Same $n.first_error @{type='AttributeError';message="'str' object has no attribute 'decode'"} 'Actual failure'
Conserved-F-Text $pins.failed.junit;Conserved-Same $pins.failed.junit.bytes ([long]0) 'Empty JUnit is not success'
$a=Conserved-TextJson $pins.admin
foreach($key in @('admission','commit')){Conserved-Same $a[$key].final_tool_exit_code ([long]0) 'Bookkeeping return only'}
Conserved-Same $a.admission.parsed_actual_stdout.admitted 'direct-science-20261008-conserved-bootstrap' 'Actual admitted failed call';Conserved-Same $a.admission.parsed_actual_stdout.request $pins.failed.request 'Actual request'
$out=$a.commit.parsed_actual_stdout;Conserved-Same $out.committed $true 'Actual once settlement';Conserved-Same $out.cost $pins.failed.cost 'Committed failure'
Conserved-Same $out.actual_s ([double]9.208) 'No administrative credit';Conserved-Same $out.exit_code ([long]124) 'Bookkeeping not science'
$live=@{path=$taskBase+'/parent-e358-execution-ledger.json';bytes=$pins.closed.bytes;sha256=$pins.closed.sha256};Conserved-Same $out.ledger $live 'Actual final mutable-ledger observation'
$s=Conserved-TextJson $pins.source_before;Conserved-Same $s.schema 'flylab.m3.post-bootstrap-resource-route.source-before.v1' 'Actual before Source record'
Need ($s.source_snapshots -is [array] -and $s.source_snapshots.Length -eq 13) 'Full before Source13';$seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach($item in $s.source_snapshots){
Conserved-ExactKeys $item @('path','bytes','sha256','archive');Need ($seen.Add($item.path)) 'Unique before Source'
Conserved-Same $item.archive ($taskBase+'/parent-post-bootstrap-resource-route-ROOT-review-20261008/before/'+$item.path) 'Exact preserved archive'
Conserved-F-Text @{path=$item.archive;bytes=$item.bytes;sha256=$item.sha256}
if($item.path -cin @(Conserved-HistoricalWriterSources)){Conserved-Same $f.sources[$item.path] @{path=$item.path;bytes=$item.bytes;sha256=$item.sha256} 'Old operations used old writer, never new allocator'}
}
}
function Conserved-F-BeforeMetadata($e){
Conserved-ExactKeys $e @('schema','event_id','scope','before_ledger','historical_closed282','initial_policy','transfers','j_grant','direct_human_auto_authority','root_decision','reviewed_source','astra_review','administrative_observations','failed_bootstrap','source_before','qualification_plan','old_caps','new_caps','delta_s','combined_global_s','new_resource_grant_s','superseded_administrative_refusal')
$pins=Conserved-F-Pins;$old=Conserved-Pins
$original=Conserved-TextJson $pins.original_event;Conserved-Same $original.schema 'flylab.m3.conserved-resource-prequalification.v1' 'Immutable original F event'
$expected=Conserved-Json ($original|ConvertTo-Json -Depth 100);$expected.schema='flylab.m3.conserved-resource-prequalification.v2';$expected.event_id='conserved-prequalification-20261008-f-fixed-600';$expected.reviewed_source=$e.reviewed_source;$expected.superseded_administrative_refusal=$pins.refusal
Conserved-Same $e $expected 'Only three replaced fields and one actual-refusal field';Conserved-F-Refusal
foreach($pair in @(@('schema','flylab.m3.conserved-resource-prequalification.v2'),@('event_id','conserved-prequalification-20261008-f-fixed-600'),@('scope','PREQUALIFICATION_ONLY'),@('before_ledger',$pins.closed),@('historical_closed282',$old.historical),@('initial_policy',$old.policy),@('j_grant',$old.j),@('direct_human_auto_authority',$old.human),@('astra_review',$pins.astra),@('administrative_observations',$pins.admin),@('failed_bootstrap',$pins.failed),@('source_before',$pins.source_before),@('combined_global_s',[long]17900),@('new_resource_grant_s',[long]0),@('qualification_plan',(Conserved-F-Plan)),@('superseded_administrative_refusal',$pins.refusal))){Conserved-Same $e[$pair[0]] $pair[1] 'Exact fixed F event field'}
Conserved-Same $e.old_caps @{generated_global=[long]1700;generated_local=[long]964;original_global=[long]16200} 'Old paired caps'
Conserved-Same $e.new_caps @{generated_global=[long]2300;generated_local=[long]1564;original_global=[long]15600} 'New paired caps'
Conserved-Same $e.delta_s @{generated_global=[long]600;generated_local=[long]600;original_global=[long]-600} 'Conserved integer deltas'
Conserved-Same $e.root_decision @{source=$pins.root;pool_transfer_s=[long]600;human_numerical_approval=$false} 'ROOT decision not human numeric approval'
Generated-Authority $old.human
$before=Conserved-TextJson $pins.closed 524288;$history=Conserved-TextJson $old.historical 524288
Need (-not $before.Contains('direct_science_conserved_allocation') -and (Generated-Budget $before) -eq 964) 'Real old-cap closed283, never reset clone'
Conserved-Closed $before;Conserved-Prefix $before $history
Need ($before.debits.Length -eq 283 -and (Conserved-Count $before.direct_science_calls) -eq 165 -and $history.debits.Length -eq 282 -and (Conserved-Count $history.direct_science_calls) -eq 164) 'Entire282 plus sole actual failed bootstrap'
$tail=Conserved-TailKeys;foreach($entry in $history.GetEnumerator()){if($entry.Key -cnotin $tail){Conserved-Same $before[$entry.Key] $entry.Value 'Every immutable historical field'}}
foreach($pair in @(@('generated_shared_cumulative_limit_s',1700),@('direct_science_generated_limit_s',964),@('original_cumulative_limit_s',16200))){Conserved-Same $before[$pair[0]] ([long]$pair[1]) 'Strict old paired caps'}
Conserved-Same $before.direct_science_policy $old.policy 'Old policy retained';$transfers=@{};foreach($key in (Conserved-Transfers)){$transfers[$key]=$before[$key]};Conserved-Same $e.transfers $transfers 'All six old transfers'
Conserved-Same $before.startup_ray_original_grants['startup-ray-original-20261007-j'].grant $old.j 'Immutable J16200/6300/6000'
$p=Conserved-TextJson $old.policy;$g=Conserved-TextJson $old.j;Conserved-Totals $before $p $g 1700 16200
Conserved-F-FailedHistory $before
return @{before=$before;policy=$p;grant=$g}
}
function Conserved-F-Review($spec){$r=Conserved-F-ReviewMetadata $spec;foreach($p in $r.sources.Values){Conserved-F-Text $p};return $r}
function Conserved-F-Before($e){$data=Conserved-F-BeforeMetadata $e;$null=Conserved-F-Review $e.reviewed_source;return $data}
function Conserved-RouteBefore($spec,$e){
if($spec.path -ceq ($taskBase+'/parent-conserved-resource-allocation-20261008-a.json')){Conserved-Same $e.schema 'flylab.m3.conserved-resource-allocation.v1' 'Strict old v1 path/schema';return Conserved-AllocationBefore $e}
Conserved-Same $spec.path ($taskBase+'/parent-conserved-resource-prequalification-20261008-f-fixed.json') 'Only exact second event path';Conserved-Same $e.schema 'flylab.m3.conserved-resource-prequalification.v2' 'Strict F path/schema'
return Conserved-F-Before $e
}
function Conserved-F-Unused($l){
Conserved-Closed $l
foreach($call in @('direct-science-20261008-contact-f-lint','direct-science-20261008-contact-f-math','direct-science-20261008-contact-f-gate-a','direct-science-20261008-contact-f-gate-b','contact-learning-20261008-f-body','contact-learning-20261008-f-body-audit')){
Need (-not $l.direct_science_calls.Contains($call) -and $call -cnotin @($l.debits|ForEach-Object{$_.cost_id})) 'All fixed F scientific IDs remain unused at allocation'
$stem=$taskBase+'/parent-'+$call
foreach($name in @($stem,($stem+'-request.json'),($stem+'-full-return.json'),($stem+'-genuine-trace.json'))){Need (-not [IO.File]::Exists((Safe $name)) -and -not [IO.Directory]::Exists((Safe $name))) 'No prior fixed F request/output/cost/trace'}
}
foreach($role in @('math','gate-a','gate-b')){Need (-not [IO.File]::Exists((Safe ($taskBase+'/parent-contact-learning-f-'+$role+'-junit.xml')))) 'No existing fixed F JUnit'}
foreach($name in @((($taskBase+'/parent-contact-learning-source-freeze-20261008-f-fixed.json')),($taskBase+'/parent-contact-learning-qualified-20261008-f-fixed.json'))){Need (-not [IO.File]::Exists((Safe $name))) 'No premature fixed freeze or qualification at allocation'}
}
function Conserved-F-Freeze($spec,$e,$eventSpec){
Conserved-Same $spec.path ($taskBase+'/parent-contact-learning-source-freeze-20261008-f-fixed.json') 'Fresh F freeze only'
$f=Conserved-TextJson $spec;Conserved-Same $f.schema 'flylab.m3.contact-learning.source-freeze.v1' 'Inherited full Source schema';Conserved-Same $f.ROOT_review_cycle_complete $true 'Actual reviewed current Source'
Conserved-Same $f.qualification_epoch 'contact-prequalification-20261008-f' 'Distinct F epoch';Conserved-Same $f.reviewed_source $e.reviewed_source 'Same reviewed vector';Conserved-Same $f.prequalification_event $eventSpec 'Freeze after event; acyclic'
$r=Conserved-F-Review $e.reviewed_source;Conserved-Same $f.sources $r.sources 'Entire same current Source vector, no legacy relabelling'
return $f
}
function Conserved-F-GeneratedRequest($q,$plan,$source,[bool]$fresh){
Conserved-ExactKeys $q @('schema','mode','call_id','role','source_freeze','shared_original_lock_path','ledger_path','limits','generated_behavior','generated_argv','output')
$expected=@{schema='flylab.m3.direct-science.supervision.v1';mode='generated';call_id=$plan.call;role=$plan.role;source_freeze=$source;shared_original_lock_path=$taskBase+'/parent-e358-original-active.lock';ledger_path=$taskBase+'/parent-e358-execution-ledger.json';limits=$plan.limits;generated_behavior='command';generated_argv=$plan.argv;output=$plan.output}
Conserved-Same $q $expected 'Full exact F generated request; no authorization alias'
if($fresh){Need (-not [IO.Directory]::Exists((Safe $plan.output)) -and -not [IO.File]::Exists((Safe $plan.output))) 'Unused fixed actual output'}
}
function Conserved-F-Progress($l,$e,$eventSpec,$source){
$anchor=Conserved-TextJson $e.before_ledger 524288;$tail=@($l.debits|Select-Object -Skip $anchor.debits.Length);Need ($tail.Length -le 6) 'No fifth generated/retry/other Original tail'
[long]$completed=[Math]::Min($tail.Length,4)
for($i=0;$i -lt $completed;$i++){
$plan=Contact-F-GeneratedPlan (@('lint','math','gate-a','gate-b')[$i]);$d=$tail[$i]
Conserved-Same $d.cost_id $plan.call 'Ordered F fee';Conserved-Same $d.mode 'generated' 'Generated qualifier only'
$c=Conserved-TextJson $d.descriptor;$q=Conserved-CostProof $c $d;Conserved-Same $c.exit_code ([long]0) 'Any paid failure blocks progression/replacement';Conserved-Same $c.source_freeze $source 'All genuine fees same fresh Source'
Conserved-F-GeneratedRequest $q $plan $source $false
Conserved-Same $l.direct_science_calls[$plan.call] @{state='charged';mode='generated';request=$c.request;reserved_s=$plan.limits.outer_s} 'Actual once charged call'
}
for($i=4;$i -lt $tail.Length;$i++){
$call=@('contact-learning-20261008-f-body','contact-learning-20261008-f-body-audit')[$i-4];$d=$tail[$i]
Conserved-Same $d.cost_id $call 'Only sole F body then audit';Conserved-Same $d.mode 'original' 'Actual body Original'
$c=Conserved-TextJson $d.descriptor;$q=_Conserved-ActualCostProof $c $d 'original'
Conserved-Same $c.exit_code ([long]0) 'Failed body cannot enable audit/repeat';Conserved-Same $c.source_freeze $source 'Same F physical Source';Conserved-Same $q.call_id $call 'Actual fixed F scope'
Conserved-Same $l.direct_science_calls[$call] @{state='charged';mode='original';request=$c.supervisor_request;reserved_s=[long](@(300,120)[$i-4])} 'Once actual body/audit charged'
}
$null=Conserved-F-Freeze $source $e $eventSpec
return @{completed=$completed;bodies=[long]($tail.Length-$completed);anchor=$anchor}
}

function Conserved-F-Qualification($q,$l){
$caps=Conserved-Caps $l;Conserved-Same $caps @{generated_global=[long]2300;generated_local=[long]1564;original_global=[long]15600} 'Paired F caps'
$eventSpec=$l.direct_science_conserved_allocation;Conserved-Same $eventSpec.path ($taskBase+'/parent-conserved-resource-prequalification-20261008-f-fixed.json') 'F qualification only F event'
$e=Conserved-TextJson $eventSpec;$pins=Conserved-Pins
Conserved-ExactKeys $q @('schema','epoch','old_E360','allocation','reviewed_source','qualification_roster','source_freeze','actual_costs','tests','evidence')
foreach($pair in @(@('schema','flylab.m3.direct-science.qualification.v3'),@('epoch','contact-prequalification-20261008-f'),@('old_E360','UNPASSED'),@('allocation',$eventSpec),@('reviewed_source',$e.reviewed_source),@('qualification_roster',$pins.roster))){Conserved-Same $q[$pair[0]] $pair[1] 'Exact F qualification'}
$p=Conserved-F-Progress $l $e $eventSpec $q.source_freeze;Conserved-Same $p.completed ([long]4) 'All four genuine successful costs'
$costs=@($l.debits|Select-Object -Skip $p.anchor.debits.Length -First 4|ForEach-Object{$_.descriptor});Conserved-Same $q.actual_costs $costs 'Exact four actual once costs'
Need ($q.tests -is [array] -and $q.tests.Length -eq 3) 'All complete actual JUnits';$r=Conserved-TextJson $pins.roster
for($i=0;$i -lt 3;$i++){
$role=@('math','gate-a','gate-b')[$i];$rows=@($r[$role.Replace('-','_')]);$item=$q.tests[$i]
Conserved-ExactKeys $item @('file','node_ids','names','classnames','count')
Conserved-Same $item.file.path ($taskBase+'/parent-contact-learning-f-'+$role+'-junit.xml') 'New actual F JUnit'
Conserved-Same $item.node_ids @($rows|ForEach-Object{$_.node_id}) 'Every full parameterized ID';Conserved-Same $item.names @($rows|ForEach-Object{$_.name}) 'Every full name';Conserved-Same $item.classnames @($rows|ForEach-Object{$_.classname}) 'Every module/class';Conserved-Same $item.count ([long]$rows.Length) 'No subset'
Conserved-JUnit $item.file $rows
}
Conserved-Same $q.evidence $q.actual_costs 'Every actual cost/owner/trace; no authorization bool'
return [long]($p.anchor.debits.Length+4)
}
function Conserved-F-Dispatch($l,$q,$requestSpec,[string]$phase){
if($l.Contains('direct_science_qualification_lint_cleanup')){return Conserved-J-Dispatch $l $q $requestSpec $phase}
if($l.Contains('direct_science_preflight_recovery')){return Conserved-I-Dispatch $l $q $requestSpec $phase}
if($l.Contains('direct_science_qualification_lint_repair')){return Conserved-H-Dispatch $l $q $requestSpec $phase}
if($l.Contains('direct_science_qualification_amendment')){return Conserved-G-Dispatch $l $q $requestSpec $phase}
$null=Conserved-Caps $l # global missing-pointer/mixed-cap refusal before science/request access
if(-not $l.Contains('direct_science_conserved_allocation')){return $null}
if($l.direct_science_conserved_allocation.path -cne ($taskBase+'/parent-conserved-resource-prequalification-20261008-f-fixed.json')){return $null}
Need ($phase -cin @('closed','admitted')) 'Exact closed/admitted dispatch phase'
$eventSpec=$l.direct_science_conserved_allocation;$e=Conserved-TextJson $eventSpec;$source=$q.source_freeze
$p=Conserved-F-Progress $l $e $eventSpec $source
if($q.mode -ceq 'generated'){
Need ($p.completed -lt 4 -and $p.bodies -eq 0) 'Only next unused fixed full F qualification; no fifth/retry'
$role=@('lint','math','gate-a','gate-b')[$p.completed];$plan=Contact-F-GeneratedPlan $role;Conserved-F-GeneratedRequest $q $plan $source $true
}else{
Conserved-Same $q.mode 'original' 'Finite mode';Need ($p.completed -eq 4 -and $p.bodies -lt 2) 'All four before any Original'
$qs=$l.direct_science_qualification;Need ($qs -is [Collections.IDictionary] -and $qs.path -ceq ($taskBase+'/parent-contact-learning-qualified-20261008-f-fixed.json')) 'Legacy qualification never releases F'
$qual=Conserved-TextJson $qs;$null=Conserved-F-Qualification $qual $l;Conserved-Same $qual.source_freeze $source 'Same qualified F Source'
Conserved-Same $q.call_id (@('contact-learning-20261008-f-body','contact-learning-20261008-f-body-audit')[$p.bodies]) 'Only sole same-candidate body/audit'
Conserved-Same $q.role (@('contact_candidate_body','contact_body_audit')[$p.bodies]) 'No generic/tracking/replay/geometry/legacy bypass'
Conserved-ExactKeys $q @('schema','mode','call_id','role','source_freeze','shared_original_lock_path','ledger_path','limits','output','science_request','grant')
Conserved-Same $q.schema 'flylab.m3.direct-science.supervision.v1' 'Actual same supervisor protocol';Conserved-Same $q.grant (Conserved-Pins).j 'Immutable J no restore'
$contract=Conserved-TextJson (Conserved-TextJson (Conserved-F-Pins).body).contract;$limits=@{};foreach($key in @('outer_s','child_s','cleanup_s','host_bytes','output_cap_bytes')){$limits[$key]=$contract.limits[$q.role][$key]}
Conserved-Same $q.limits $limits 'Unchanged actual body/audit bounds';Conserved-Same $q.output ($taskBase+'/parent-'+$q.call_id+'/result-supervision') 'Fresh fixed F output'
Conserved-Same $q.shared_original_lock_path ($taskBase+'/parent-e358-original-active.lock') 'Same common F exclusion';Conserved-Same $q.ledger_path ($taskBase+'/parent-e358-execution-ledger.json') 'Same actual ledger'
}
Conserved-Same (Conserved-TextJson $requestSpec) $q 'Exact actual dispatch descriptor'
$calls=@{};foreach($entry in $p.anchor.direct_science_calls.GetEnumerator()){$calls[$entry.Key]=$entry.Value}
foreach($d in @($l.debits|Select-Object -Skip $p.anchor.debits.Length)){$calls[$d.cost_id]=$l.direct_science_calls[$d.cost_id]}
if($phase -ceq 'admitted'){$calls[$q.call_id]=@{state='admitted';mode=$q.mode;request=$requestSpec;reserved_s=$q.limits.outer_s}}
Conserved-Same $l.direct_science_calls $calls 'No extra admitted alias/stale charged call'
foreach($key in (Conserved-OwnerKeys)){
$expected=@{};if($phase -ceq 'admitted' -and $q.mode -ceq 'generated' -and $key -ceq 'direct_science_reservations'){$expected[$q.call_id]=@{request=$requestSpec;reserved_s=$q.limits.outer_s}}
Conserved-Same $l[$key] $expected ('Exact phase/current own reservation/worker/pending/HOLD '+$key)
}
Conserved-Same $l.original_jobs_running ([long]0) 'No pre-spawn worker';Conserved-Same $l.original_reserved_s ([long]0) 'Original reserve begins at actual begin'
Need (-not [IO.File]::Exists((Safe $q.output)) -and -not [IO.Directory]::Exists((Safe $q.output))) 'Unused fixed actual output'
return @{completed=$p.completed;bodies=$p.bodies;epoch='contact-prequalification-20261008-f'}
}

function Conserved-G-Paths {(_Conserved-Record 'g').paths}
function _Conserved-RecordPin {return @{path='verification/banc-natural-gait-m3-20260919/parent-j-immutable-attempt-records-20261008.json';bytes=[long]16830;sha256='60cb1084cefbb8204178c15ffa319da0683d137b1d9d8da460591a5ab7945db0'}}
function _Conserved-RecordShape($d){
Conserved-ExactKeys $d @('schema','families');Conserved-Same $d.schema 'flylab.m3.native-immutable-attempt-records.v1' 'Record schema';Conserved-ExactKeys $d.families @('g','h','i','j')
foreach($family in @('g','h','i','j')){
$r=$d.families[$family];$field=@{g='amendment';h='repair';i='recovery';j='cleanup'}[$family];$keys=@('pins','paths','epoch');if($family -cne 'g'){$keys+='predecessor'};Conserved-ExactKeys $r $keys
$keys=@('before','root','closed','astra');if($family -cne 'g'){$keys+='registered'};if($family -cin @('i','j')){$keys+='admitted'};Conserved-ExactKeys $r.pins $keys;Conserved-ExactKeys $r.pins.astra @('markdown','json')
foreach($k in $keys){if($k -ceq 'astra'){$pins=@($r.pins.astra.markdown,$r.pins.astra.json)}else{$pins=@($r.pins[$k])};foreach($p in $pins){Conserved-SourceShape $p $p.path;Need ($p.bytes -gt 0) 'Record pin'}}
Conserved-ExactKeys $r.paths @('body',$field,'review','source','qualification')
$paths=@($r.paths.Values);if($family -cne 'g'){$keys=@('review','freeze','request','trace','outer','stdout','stderr','cost','result');$keys+=if($family -ceq 'h'){'amendment'}elseif($family -ceq 'i'){'repair'}else{'recovery'};if($family -cne 'h'){$keys+=@('registered','admitted')};Conserved-ExactKeys $r.predecessor $keys;$paths+=@($r.predecessor.Values)}
foreach($p in $paths){Need ($p -is [string] -and $p.Length -gt 0) 'Record path'}
$e=$r.epoch;Conserved-ExactKeys $e @('key','epoch','schema','offset','end','calls','caps','metadata','review','freeze','body','generatedPlan','pins','currentReview','envelope','origin')
foreach($k in @('key','epoch','origin')){Need ($e[$k] -is [string] -and $e[$k].Length -gt 0) 'Record string'}
$i='ghij'.IndexOf($family);$u=$family.ToUpperInvariant();$p='Conserved-'+$u+'-';$call='contact-learning-20261008-'+$family+'-body'
$expected=@{offset=[long](284+$i);end=[long](288+$i);envelope=[long]$(if($i -lt 2){300}else{575});schema='flylab.m3.direct-science.qualification.v'+(4+$i);calls=@($call,($call+'-audit'));caps=$p+'Caps';review=$p+'ReviewMetadata';freeze=$p+'Freeze';body=$p+'Body';pins=$p+'Pins';currentReview=$p+'CurrentReview';metadata=$p+@{g='Amendment';h='Repair';i='Recovery';j='Cleanup'}[$family];generatedPlan='Contact-'+$u+'-GeneratedPlan'}
foreach($pair in $expected.GetEnumerator()){Conserved-Same $e[$pair.Key] $pair.Value 'Record values'}
}}
function _Conserved-Record([string]$family){
Need ($family -cin @('g','h','i','j')) 'Record family'
$d=Conserved-TextJson (_Conserved-RecordPin) 262144;_Conserved-RecordShape $d
$r=$d.families[$family];foreach($p in $r.pins.Values){if($p.Contains('bytes')){$p.bytes=[long]$p.bytes}else{foreach($v in $p.Values){$v.bytes=[long]$v.bytes}}}
foreach($k in @('offset','end','envelope')){$r.epoch[$k]=[long]$r.epoch[$k]};return $r
}
function Conserved-G-Pins {(_Conserved-Record 'g').pins}
function Conserved-G-Replacements {return @('tools/banc_m3_conserved_resource.py','tools/banc_m3_conserved_resource_native_epoch.ps1',($taskBase+'/parent-direct-science-account.ps1'),'tools/banc_m3_startup_ray_supervisor.py','tools/banc_m3_contact_learning_admission.py',($taskBase+'/parent-contact-learning-contract-native.ps1'),'tests/test_c_banc_contact_runtime_journal.py')}
function Conserved-G-FPaths {
return @{allocation=$taskBase+'/parent-conserved-resource-prequalification-20261008-f-fixed.json';review=$taskBase+'/parent-conserved-resource-prequalification-reviewed-source-20261008-f-fixed.json';freeze=$taskBase+'/parent-contact-learning-source-freeze-20261008-f-fixed.json';trace=$taskBase+'/parent-direct-science-20261008-contact-f-lint-genuine-return.json';request=$taskBase+'/parent-direct-science-20261008-contact-f-lint-request.json';admitted=$taskBase+'/parent-f-lint-actual-admitted455814-20261008.json';outer=$taskBase+'/parent-direct-science-20261008-contact-f-lint/outer.json';cost=$taskBase+'/parent-direct-science-20261008-contact-f-lint-full-return.json';result=$taskBase+'/parent-f-lint-settlement-ROOT-actual-result-20261008.json'}
}
function Conserved-G-Archives {
$pins=Conserved-G-Pins;$before=Conserved-TextJson $pins.before
Conserved-ExactKeys $before @('schema','source_metadata_only','actual_f_lint_once_settled','authoring_note','scientific_execution','registration_committed','snapshots','closed284')
foreach($pair in @(@('schema','flylab.m3.g-qualification-repair.source-before.v1'),@('source_metadata_only',$true),@('actual_f_lint_once_settled',$true),@('scientific_execution','NOT_RUN'),@('registration_committed',$false),@('closed284',$pins.closed))){Conserved-Same $before[$pair[0]] $pair[1] 'Actual pre-G archive map'}
Need ($before.snapshots -is [array] -and $before.snapshots.Length -eq 23) 'All23 immutable historical mappings';$mapped=@{}
for($i=0;$i -lt 23;$i++){
$item=$before.snapshots[$i];Conserved-ExactKeys $item @('source','archive');Conserved-ExactKeys $item.source @('path','bytes','sha256')
Need (-not $mapped.Contains($item.source.path)) 'Unique historical mapping'
$expected=@{path=$taskBase+'/parent-g-before-20261008/'+$i.ToString('00')+'-'+$item.source.sha256.Substring(0,16)+'.txt';bytes=$item.source.bytes;sha256=$item.source.sha256}
Conserved-Same $item.archive $expected 'Literal numbered full-content archive';$null=Conserved-TextRaw $item.archive 524288;$mapped[$item.source.path]=$item
}
foreach($name in @((Conserved-G-Replacements)+@((Conserved-G-FPaths).Values)+@((Conserved-F-Pins).body.path,$pins.closed.path))){Need ($mapped.Contains($name)) 'Complete fixed pre-G history map'}
return $mapped
}
function Conserved-G-HistoricalReview {
$mapped=Conserved-G-Archives;$r=Conserved-F-ReviewMetadata $mapped[(Conserved-G-FPaths).review].source
Need ((Conserved-Count $r.sources) -eq 723) 'Entire immutable F723 vector'
foreach($name in (Conserved-G-Replacements)){Conserved-Same $r.sources[$name] $mapped[$name].source 'Exact superseded F Source';$null=Conserved-TextRaw $mapped[$name].archive 262144}
return @{review=$r;mapped=$mapped}
}
function Contact-G-GeneratedPlan([string]$role){
$old=Contact-F-GeneratedPlan $role;$argv=@($old.argv);if($role -cne 'lint'){$argv[-1]='--junitxml='+$taskBase+'/parent-contact-learning-g-'+$role+'-junit.xml'}
$call='direct-science-20261008-contact-g-'+$role;return @{call=$call;role=$old.role;argv=$argv;limits=$old.limits;output=$taskBase+'/parent-'+$call}
}
function Conserved-G-Plan {
$calls=@();foreach($role in @('lint','math','gate-a','gate-b')){$calls+=Contact-G-GeneratedPlan $role}
return @{epoch='contact-qualification-repair-20261008-g';roster_epoch='contact-conserved-20261008-e';qualification_roster=(Conserved-Pins).roster;calls=$calls;generated_close_s=[long]20;original_release_calls=@('contact-learning-20261008-g-body','contact-learning-20261008-g-body-audit')}
}
function Conserved-G-ReviewMetadata($spec,$h){return _Conserved-Epoch-ReviewMetadata $spec $h 'g'}
function Conserved-G-Body($spec){return _Conserved-Epoch-Body $spec 'g'}

function Conserved-G-CurrentReview($spec){return _Conserved-Epoch-CurrentReview $spec 'g'}

function Conserved-G-History {
$h=Conserved-G-HistoricalReview;$paths=Conserved-G-FPaths;$pins=@{};foreach($entry in $paths.GetEnumerator()){$pins[$entry.Key]=$h.mapped[$entry.Value].source}
$e=Conserved-TextJson $pins.allocation;Conserved-Same $e.reviewed_source $pins.review 'Actual unchanged F allocation/review'
$data=Conserved-F-BeforeMetadata $e;$allocated=Conserved-AllocatedSnapshot $data.before $pins.allocation $data.policy $data.grant
$freeze=Conserved-TextJson $pins.freeze
foreach($pair in @(@('schema','flylab.m3.contact-learning.source-freeze.v1'),@('ROOT_review_cycle_complete',$true),@('qualification_epoch','contact-prequalification-20261008-f'),@('reviewed_source',$pins.review),@('prequalification_event',$pins.allocation),@('sources',$h.review.sources))){Conserved-Same $freeze[$pair[0]] $pair[1] 'Entire actual as-executed F freeze'}
$admitted=Conserved-TextJson $pins.admitted 524288;$q=Conserved-TextJson $pins.request;Conserved-F-GeneratedRequest $q (Contact-F-GeneratedPlan 'lint') $pins.freeze $false
$expected=Conserved-Json ($allocated|ConvertTo-Json -Depth 100);$call='direct-science-20261008-contact-f-lint'
$expected.direct_science_calls[$call]=@{state='admitted';mode='generated';request=$pins.request;reserved_s=[long]15};$expected.direct_science_reservations=@{};$expected.direct_science_reservations[$call]=@{request=$pins.request;reserved_s=[long]15}
Conserved-Same $admitted $expected 'Actual admitted283/166; no cold mutable witness read'
$anchor=Conserved-TextJson (Conserved-G-Pins).closed 524288;Conserved-Closed $anchor;Conserved-Prefix $anchor $allocated
Need ($anchor.debits.Length -eq 284 -and (Conserved-Count $anchor.direct_science_calls) -eq 166) 'Actual once closed284/166';Conserved-ExactKeys $anchor @(Conserved-Keys $allocated)
$tail=Conserved-TailKeys;foreach($entry in $allocated.GetEnumerator()){if($entry.Key -cnotin $tail){Conserved-Same $anchor[$entry.Key] $entry.Value 'Immutable actual F allocation field'}}
$d=$anchor.debits[-1];Conserved-Same $d @{cost_id=$call;mode='generated';actual_outer_full_return_s=[double]1.755;actual_returned_exit_code=[long]1;descriptor=$pins.cost} 'Sole exact paid F failure'
$cost=Conserved-TextJson $pins.cost;Conserved-Same (Conserved-CostProof $cost $d) $q 'Actual failed request proof'
foreach($key in @('trace','request','outer')){Conserved-Same $cost[$key] $pins[$key] 'Immutable actual F settlement evidence'};Conserved-Same $cost.source_freeze $pins.freeze 'Actual failed F Source'
$calls=Conserved-Json ($allocated.direct_science_calls|ConvertTo-Json -Depth 100);$calls[$call]=@{state='charged';mode='generated';request=$pins.request;reserved_s=[long]15};Conserved-Same $anchor.direct_science_calls $calls 'All165 plus sole once-charged F failure'
$outer=Conserved-TextJson $pins.outer;Conserved-Same $outer.origin 'ROOT_ADMINISTRATIVE_RECONSTRUCTION' 'ROOT witness, never runtime supervisor receipt'
Conserved-Same $outer.current_ledger @{path=$taskBase+'/parent-e358-execution-ledger.json';bytes=$pins.admitted.bytes;sha256=$pins.admitted.sha256} 'Historical observation binds immutable admitted snapshot'
Conserved-Same $outer.immutable_admitted_ledger $pins.admitted 'Exact immutable admitted identity'
foreach($pair in @(@('job_created',$false),@('worker_started',$false),@('worker_pid',$null),@('worker_creation_ticks',$null),@('stream_state','NO_WORKER_STREAMS_OPENED'),@('qualification','NOT_QUALIFIED'))){Conserved-Same $outer[$pair[0]] $pair[1] 'Actual F pre-Job facts'}
$result=Conserved-TextJson $pins.result;Conserved-Same $result.actual_administrative_commit.first_final_tool.exit_code ([long]0) 'Actual unchanged canonical once commit'
Conserved-Same $result.independent_actual_readback.closed284 (Conserved-G-Pins).closed 'Actual ROOT closed284';Conserved-Same $result.independent_actual_readback.actual_cost $pins.cost 'Actual ROOT cost'
Conserved-Totals $anchor $data.policy $data.grant 2300 15600
return @{anchor=$anchor;policy=$data.policy;grant=$data.grant;history=$h;pins=$pins}
}
function Conserved-G-Amendment($spec){
$paths=Conserved-G-Paths;$gPins=Conserved-G-Pins;Conserved-Same $spec.path $paths.amendment 'One immutable G pointer path';$a=Conserved-TextJson $spec;$h=Conserved-G-History
Conserved-ExactKeys $a @('schema','epoch','scope','allocation','historical_closed284','failed_f_lint','source_before','reviewed_source','qualification_plan','new_resource_grant_s','refund_s')
foreach($pair in @(@('schema','flylab.m3.contact-qualification-amendment.v1'),@('epoch','contact-qualification-repair-20261008-g'),@('scope','PREQUALIFICATION_ONLY'),@('allocation',$h.pins.allocation),@('historical_closed284',$gPins.closed),@('failed_f_lint',$h.pins),@('source_before',$gPins.before),@('qualification_plan',(Conserved-G-Plan)),@('new_resource_grant_s',[long]0),@('refund_s',[long]0))){Conserved-Same $a[$pair[0]] $pair[1] 'Strict immutable G amendment'}
$null=Conserved-G-ReviewMetadata $a.reviewed_source $h.history
$h.amendment=$a;return $h
}
function Conserved-G-Registration($l,$spec,[byte[]]$beforeRaw){return _Conserved-Epoch-Registration $l $spec $beforeRaw 'g'}
function Conserved-G-Caps($l){return _Conserved-Epoch-Caps $l 'g'}
function Conserved-G-Freeze($spec,$a,$amendmentSpec){
$paths=Conserved-G-Paths;Conserved-Same $spec.path $paths.source 'Fresh literal G freeze';$f=Conserved-TextJson $spec
$mapped=Conserved-G-Archives;$prior=Conserved-TextJson $mapped[(Conserved-G-FPaths).freeze].source
$expected=Conserved-Json ($prior|ConvertTo-Json -Depth 100);$expected.qualification_epoch='contact-qualification-repair-20261008-g';$expected.reviewed_source=$a.reviewed_source;$expected.qualification_amendment=$amendmentSpec;$expected.sources=$f.sources;$expected.body_retry=$f.body_retry
Conserved-Same $f $expected 'Whole inherited F metadata plus exact G fields; old provenance stays historical'
Conserved-Same $f.ROOT_review_cycle_complete $true 'Actually reviewed G Source';Conserved-Same $f.schema 'flylab.m3.contact-learning.source-freeze.v1' 'Same freeze protocol'
$review=Conserved-G-CurrentReview $a.reviewed_source;Conserved-Same $f.sources $review.sources 'Entire G724, no old bytes presented as current';Conserved-Same $f.body_retry $review.sources[$paths.body] 'Only reviewed static G scope'
return $f
}
function _Conserved-Epoch-Progress($l,$a,$amendmentSpec,$source,[ValidateSet('g','h','i','j')][string]$family){
$f=_Conserved-Epoch $family
$tail=@($l.debits|Select-Object -Skip $f.offset);Need ($tail.Length -le 6) 'No fifth G/H role/retry/other Original';[long]$completed=[Math]::Min($tail.Length,4)
for($i=0;$i -lt $tail.Length;$i++){
$d=$tail[$i];$generated=$i -lt 4
if($generated){$plan=& $f.generatedPlan (@('lint','math','gate-a','gate-b')[$i]);$call=$plan.call;$mode='generated'}else{$call=$f.calls[$i-4];$mode='original'}
Conserved-Same $d.cost_id $call 'Exact ordered G cost';Conserved-Same $d.mode $mode 'Exact G mode';$cost=Conserved-TextJson $d.descriptor
if($generated){$q=Conserved-CostProof $cost $d;Conserved-F-GeneratedRequest $q $plan $source $false;$owner=$cost.request;$reserve=$plan.limits.outer_s}
else{$q=_Conserved-ActualCostProof $cost $d 'original';Conserved-Same $q.call_id $call 'Sole G body/audit';$owner=$cost.supervisor_request;$reserve=[long](@(300,120)[$i-4])}
Conserved-Same $cost.exit_code ([long]0) 'Spent G failure blocks every later role/replacement/retry';Conserved-Same $cost.source_freeze $source 'Same actually executed G Source'
Conserved-Same $l.direct_science_calls[$call] @{state='charged';mode=$mode;request=$owner;reserved_s=$reserve} 'Once charged exact G call'
}
$null=& $f.freeze $source $a $amendmentSpec;return @{completed=$completed;bodies=[long][Math]::Max(0,$tail.Length-4)}
}
function _Conserved-Epoch-Qualification($q,$l,[ValidateSet('g','h','i','j')][string]$family){
$f=_Conserved-Epoch $family;$null=& $f.caps $l;$h=& $f.metadata $l[$f.key]
$keys=@('schema','epoch','old_E360','allocation','qualification_amendment','reviewed_source','qualification_roster','source_freeze','actual_costs','tests','evidence');if($family -cin @('h','i','j')){$keys+=@('qualification_lint_repair');Conserved-Same $q.qualification_lint_repair $l.direct_science_qualification_lint_repair 'Same immutable H repair pointer'}
if($family -cin @('i','j')){$keys+=@('preflight_recovery');Conserved-Same $q.preflight_recovery $l.direct_science_preflight_recovery 'Same immutable I pointer'}
if($family -ceq 'j'){$keys+=@('qualification_lint_cleanup');Conserved-Same $q.qualification_lint_cleanup $l[$f.key] 'Same immutable J cleanup pointer'}
Conserved-ExactKeys $q $keys
foreach($pair in @(@('schema',$f.schema),@('epoch',$f.epoch),@('old_E360','UNPASSED'),@('allocation',$l.direct_science_conserved_allocation),@('qualification_amendment',$l.direct_science_qualification_amendment),@('reviewed_source',$h.amendment.reviewed_source),@('qualification_roster',(Conserved-Pins).roster))){Conserved-Same $q[$pair[0]] $pair[1] 'Exact G/H qualification'}
$progress=_Conserved-Epoch-Progress $l $h.amendment $l[$f.key] $q.source_freeze $family;Conserved-Same $progress.completed ([long]4) 'Exactly four genuine fresh successful G/H roles'
Conserved-Same $q.actual_costs @($l.debits|Select-Object -Skip $f.offset -First 4|ForEach-Object{$_.descriptor}) 'All exact once real G/H costs';Conserved-Same $q.evidence $q.actual_costs 'Four actual owner/trace/cost evidence'
Need ($q.tests -is [array] -and $q.tests.Length -eq 3) 'All three complete G JUnits';$roster=Conserved-TextJson (Conserved-Pins).roster
for($i=0;$i -lt 3;$i++){
$role=@('math','gate-a','gate-b')[$i];$rows=$roster[$role.Replace('-','_')];$test=$q.tests[$i]
Conserved-ExactKeys $test @('file','node_ids','names','classnames','count');Conserved-Same $test.file.path ($taskBase+'/parent-contact-learning-'+$family+'-'+$role+'-junit.xml') 'Literal fresh G/H JUnit'
foreach($pair in @(@('node_ids',@($rows|ForEach-Object{$_.node_id})),@('names',@($rows|ForEach-Object{$_.name})),@('classnames',@($rows|ForEach-Object{$_.classname})),@('count',[long]$rows.Length))){Conserved-Same $test[$pair[0]] $pair[1] 'Whole unchanged19/21/32 G JUnit'}
Conserved-JUnit $test.file $rows
}
return $f.end
}
function _Conserved-Epoch-Dispatch($l,$q,$requestSpec,[string]$phase,[ValidateSet('g','h','i','j')][string]$family){
$f=_Conserved-Epoch $family;$null=& $f.caps $l;Need ($phase -cin @('closed','admitted')) 'Exact G/H dispatch phase'
$h=& $f.metadata $l[$f.key];$source=$q.source_freeze;$progress=_Conserved-Epoch-Progress $l $h.amendment $l[$f.key] $source $family
if($q.mode -ceq 'generated'){
Need ($progress.completed -lt 4 -and $progress.bodies -eq 0) 'Only next unused whole G role, no fifth/subset/retry'
$plan=& $f.generatedPlan (@('lint','math','gate-a','gate-b')[$progress.completed]);Conserved-F-GeneratedRequest $q $plan $source $true
if($progress.completed -eq 0){$envelope=$f.envelope;Need ($l.direct_science_generated_used_s+$envelope -le 1564 -and $l.generated_shared_remaining_s -ge $envelope) 'Entire fixed family plan plus distinct close20'}
}else{
Conserved-Same $q.mode 'original' 'Finite G Original mode';Need ($progress.completed -eq 4 -and $progress.bodies -lt 2) 'Global Original block before every legacy-specific input'
$qualified=$l.direct_science_qualification;Need ($qualified -is [Collections.IDictionary] -and $qualified.path -ceq $f.paths.qualification) 'Old qualifications never release current G/H'
$qualification=Conserved-TextJson $qualified;$null=_Conserved-Epoch-Qualification $qualification $l $family;Conserved-Same $qualification.source_freeze $source 'Same qualified G/H Source'
Conserved-ExactKeys $q @('schema','mode','call_id','role','source_freeze','shared_original_lock_path','ledger_path','limits','output','science_request','grant')
$call=$f.calls[$progress.bodies];$role=@('contact_candidate_body','contact_body_audit')[$progress.bodies]
Conserved-Same $q.call_id $call 'Only literal G body then conditional audit';Conserved-Same $q.role $role 'All legacy Original families stay blocked';Conserved-Same $q.schema 'flylab.m3.direct-science.supervision.v1' 'Same supervisor protocol';Conserved-Same $q.grant (Conserved-Pins).j 'Immutable J lineage'
$freeze=Conserved-TextJson $source;$body=& $f.body $freeze.body_retry;$contract=Conserved-TextJson $body.contract;$limits=@{}
foreach($key in @('outer_s','child_s','cleanup_s','host_bytes','output_cap_bytes')){$limits[$key]=$contract.limits[$role][$key]};Conserved-Same $q.limits $limits 'Unchanged body/audit limits'
Conserved-Same $q.output ($taskBase+'/parent-'+$call+'/result-supervision') 'Fresh G physical output';Conserved-Same $q.shared_original_lock_path ($taskBase+'/parent-e358-original-active.lock') 'Same common exclusion';Conserved-Same $q.ledger_path $ledgerName 'Same actual ledger'
}
Conserved-Same (Conserved-TextJson $requestSpec) $q 'Actual descriptor/current G request'
$calls=Conserved-Json ($h.anchor.direct_science_calls|ConvertTo-Json -Depth 100);foreach($d in $l.debits|Select-Object -Skip $f.offset){$calls[$d.cost_id]=$l.direct_science_calls[$d.cost_id]}
if($phase -ceq 'admitted'){$calls[$q.call_id]=@{state='admitted';mode=$q.mode;request=$requestSpec;reserved_s=$q.limits.outer_s}}
Conserved-Same $l.direct_science_calls $calls 'No alias/stale charged/admitted G call'
foreach($key in (Conserved-OwnerKeys)){
$own=@{};if($phase -ceq 'admitted' -and $q.mode -ceq 'generated' -and $key -ceq 'direct_science_reservations'){$own[$q.call_id]=@{request=$requestSpec;reserved_s=$q.limits.outer_s}}
Conserved-Same $l[$key] $own ('Exact G owner/phase/HOLD '+$key)
}
Conserved-Same $l.original_jobs_running ([long]0) 'No worker before spawn';Conserved-Same $l.original_reserved_s ([long]0) 'No runtime reserve before begin'
Need (-not [IO.File]::Exists((Safe $q.output)) -and -not [IO.Directory]::Exists((Safe $q.output))) 'Fresh fixed G output'
return @{completed=$progress.completed;bodies=$progress.bodies;epoch=$f.epoch}
}
function _Conserved-Epoch-Settlement($l,$q,$requestSpec,$outer,[ValidateSet('g','h','i','j')][string]$family){
# Historical accounting/admitted identity only: live Source failure is chargeable.
$f=_Conserved-Epoch $family;$null=& $f.caps $l;$paths=$f.paths;Conserved-Same $q.source_freeze.path $paths.source 'Actual admitted literal G/H Source descriptor'
Conserved-Same $l.direct_science_calls[$q.call_id] @{state='admitted';mode=$q.mode;request=$requestSpec;reserved_s=$q.limits.outer_s} 'Same actual admitted G once request'
if($q.mode -ceq 'generated'){
$roles=@('lint','math','gate-a','gate-b');$calls=@($roles|ForEach-Object{'direct-science-20261008-contact-'+$family+'-'+$_});$i=[array]::IndexOf($calls,$q.call_id);Need ($i -ge 0) 'Only fixed G/H generated settlement'
Conserved-F-GeneratedRequest $q (& $f.generatedPlan $roles[$i]) $q.source_freeze $false
}else{
Conserved-Same $q.mode 'original' 'Fixed G/H Original settlement';Need ($q.call_id -cin $f.calls) 'Only fixed G/H Original calls'
}
if($outer.refusal_origin -ceq $f.origin){
Conserved-Same $outer.schema 'flylab.m3.direct-science.outer.v1' 'Actual runtime supervisor-origin G receipt';Conserved-Same $outer.request $requestSpec 'Exact initial refusal request';Conserved-Same $outer.qualification_amendment $l.direct_science_qualification_amendment 'Same saved immutable G pointer'
if($family -cin @('h','i','j')){
Conserved-Same $outer.qualification_lint_repair $l.direct_science_qualification_lint_repair 'Same saved immutable H pointer';Conserved-Same $outer.source_freeze $q.source_freeze 'Exact saved admitted H freeze'
if($family -cin @('i','j')){Conserved-Same $outer.preflight_recovery $l.direct_science_preflight_recovery 'Same saved immutable I pointer'}
if($family -ceq 'j'){Conserved-Same $outer.qualification_lint_cleanup $l[$f.key] 'Same saved immutable J pointer'}
foreach($pair in @(@('runtime_reservation_state','NEVER_CREATED'),@('root_admission_reservation_state','STILL_PRESENT_PENDING_NATIVE_COMMIT'),@('ledger_mutated',$false),@('charged',$false),@('qualified',$false))){Conserved-Same $outer[$pair[0]] $pair[1] 'Exact lexical H pending admission semantics'}
foreach($pair in @(@('stdout',$null),@('stderr',$null),@('stdout_eof_observed',$false),@('stderr_eof_observed',$false),@('unknown_owned_identities',@{}))){Conserved-Same $outer[$pair[0]] $pair[1] 'No actual H streams or observed EOF'}
Conserved-ExactKeys $outer.first_error @('type','message','original_message_chars','message_truncated');Need ($outer.first_error.type -is [string] -and $outer.first_error.message -is [string]) 'Actual saved H primary error'
}
foreach($pair in @(@('pre_spawn_refused',$true),@('job_created',$false),@('worker_started',$false),@('worker_pid',$null),@('worker_creation_ticks',$null),@('worker_exit_code',$null),@('stream_state','NO_WORKER_STREAMS_OPENED'),@('runtime_reservation_created',$false))){Conserved-Same $outer[$pair[0]] $pair[1] 'Narrow lexical initial G refusal'}
Need (-not $outer.Contains('trace') -and -not $outer.Contains('dispatch_ms') -and -not $outer.Contains('first_final_ms') -and -not $outer.Contains('actual_outer_full_return_s')) 'No fabricated future collector evidence'
Assert-LedgerOwner $script:ledgerTransaction;$raw=$script:ledgerTransaction.before
Conserved-Same $outer.initial_ledger @{path=$ledgerName;bytes=[long]$raw.Length;sha256=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($raw)).ToLowerInvariant()} 'Actual saved admitted raw ledger unchanged before commit'
foreach($key in (Conserved-OwnerKeys)){$own=@{};if($q.mode -ceq 'generated' -and $key -ceq 'direct_science_reservations'){$own[$q.call_id]=@{request=$requestSpec;reserved_s=$q.limits.outer_s}};Conserved-Same $l[$key] $own 'Initial no foreign/unknown ownership'}
}
}
function _Conserved-Epoch-OriginalAccounting($q,$outer,$live,[ValidateSet('g','h','i','j')][string]$family){
# Once settlement is historical/admitted metadata only, including Source failures.
$f=_Conserved-Epoch $family;$science=Conserved-TextJson $q.science_request;$h=& $f.metadata $live[$f.key];$history=if($family -ceq 'g'){$h.history}else{$h};$r=& $f.review $h.amendment.reviewed_source $history
Conserved-Same $science.schema 'flylab.m3.contact-learning.request.v1' 'Actual admitted G science metadata'
foreach($key in @('call_id','role','grant','source_freeze')){Conserved-Same $science[$key] $q[$key] 'Same actual admitted Original identity'}
$i=[array]::IndexOf($f.calls,$q.call_id);Need ($i -ge 0) 'Only G/H body/audit'
Conserved-Same $science.role (@('contact_candidate_body','contact_body_audit')[$i]) 'Fixed G/H role';Conserved-Same $science.body_retry $r.sources[$f.paths.body] 'Immutable admitted static descriptor pin, no live Source success'
$contract=Conserved-TextJson $science.contract;Conserved-Same $science.contract.path ($taskBase+'/parent-contact-learning-stage-20261007-a.json') 'Same immutable Stage-A'
Conserved-Same $science.limits $contract.limits[$science.role] 'Unchanged full G physical limits'
foreach($key in @('outer_s','child_s','cleanup_s','host_bytes','output_cap_bytes')){Conserved-Same $q.limits[$key] $science.limits[$key] 'Actual admitted physical envelope'}
Conserved-Same $science.grant (Conserved-Pins).j 'Same immutable J';if($outer.refusal_origin -ceq $f.origin){Conserved-Same $outer.original_payload_accesses ([long]0) 'No Original payload before lexical guard'}
return $science
}

function Conserved-H-Paths {(_Conserved-Record 'h').paths}
function Conserved-H-Pins {(_Conserved-Record 'h').pins}
function Conserved-H-GPaths {(_Conserved-Record 'h').predecessor}
function Conserved-H-Archives {return _Conserved-Epoch-Archives 'h'}

function Contact-H-GeneratedPlan([string]$role){
$old=Contact-G-GeneratedPlan $role;$argv=@($old.argv);if($role -cne 'lint'){$argv[-1]='--junitxml='+$taskBase+'/parent-contact-learning-h-'+$role+'-junit.xml'}
$call='direct-science-20261008-contact-h-'+$role;return @{call=$call;role=$old.role;argv=$argv;limits=$old.limits;output=$taskBase+'/parent-'+$call}
}
function Conserved-H-Plan {
$calls=@();foreach($r in @('lint','math','gate-a','gate-b')){$calls+=Contact-H-GeneratedPlan $r}
return @{epoch='contact-qualification-lint-repair-20261008-h';roster_epoch='contact-conserved-20261008-e';qualification_roster=(Conserved-Pins).roster;calls=$calls;generated_close_s=[long]20;original_release_calls=@('contact-learning-20261008-h-body','contact-learning-20261008-h-body-audit')}
}
function Conserved-H-History {return _Conserved-HI-History 'h'}
function Conserved-H-ReviewMetadata($spec,$h){return _Conserved-Epoch-ReviewMetadata $spec $h 'h'}
function _Conserved-Epoch-ReviewMetadata($spec,$h,[ValidateSet('g','h','i','j')][string]$family){
$f=_Conserved-Epoch $family;$p=& $f.pins;$paths=$f.paths;$schema=@{g='flylab.m3.contact-qualification.reviewed-source.v1';h='flylab.m3.contact-qualification.reviewed-source.v2';i='flylab.m3.contact-qualification.reviewed-source.v3';j='flylab.m3.contact-qualification.reviewed-source.v4'}[$family];$epoch=$f.epoch;$rootPrefix=@{g='docs/BANC_M3_G_QUALIFICATION_REPAIR_ROOT_';h='docs/BANC_M3_H_QUALIFICATION_LINT_REPAIR_ROOT_';i='docs/BANC_M3_I_PREFLIGHT_OVERHEAD_RECOVERY_ROOT_';j='docs/BANC_M3_J_QUALIFICATION_LINT_CLEANUP_ROOT_'}[$family];$solPrefix=@{g='/parent-g-qualification-repair-sol-';h='/parent-h-qualification-lint-repair-sol-';i='/parent-i-preflight-overhead-sol-';j='/parent-j-qualification-lint-cleanup-sol-'}[$family];Conserved-Same $spec.path $paths.review 'One acyclic reviewed H manifest';$r=Conserved-TextJson $spec
Conserved-ExactKeys $r @('schema','epoch','sources','replacements','root_reflection','before_source','astra_review','root_reviews','sol_reports')
foreach($pair in @(@('schema',$schema),@('epoch',$epoch),@('root_reflection',$p.root),@('before_source',$p.before),@('astra_review',$p.astra))){Conserved-Same $r[$pair[0]] $pair[1] 'Reviewed H metadata'}
$null=Conserved-TextRaw $p.root 65536;foreach($pin in $p.astra.Values){$null=Conserved-TextRaw $pin 65536}
$allowed=@('FIRST_REVIEW','SECOND_REVIEW','RESIDUAL')|ForEach-Object{$rootPrefix+$_+'_20261008.md'};$seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
Need ($r.root_reviews -is [array] -and $r.root_reviews.Length -ge 2 -and $r.root_reviews.Length -le 3) 'Actual ROOT FIRST/SECOND optional residual'
foreach($pin in $r.root_reviews){Need ($pin.path -cin $allowed -and $seen.Add($pin.path)) 'No foreign/duplicate ROOT review';$null=Conserved-TextRaw $pin 1048576};foreach($name in @($allowed|Select-Object -First 2)){Need ($seen.Contains($name)) 'Actual ROOT FIRST and SECOND'}
$allowed=@();foreach($role in @('FIRST','R1')){foreach($ext in @('md','json')){$allowed+=($taskBase+$solPrefix+$role+'-20261008/'+$role+'.'+$ext)}};$seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
Need ($r.sol_reports -is [array] -and $r.sol_reports.Length -cin @(2,4)) 'Actual sole Sol FIRST optional R1'
foreach($pin in $r.sol_reports){Need ($pin.path -cin $allowed -and $seen.Add($pin.path)) 'No foreign/duplicate Sol report';$null=Conserved-TextRaw $pin 1048576};foreach($name in @($allowed|Select-Object -First 2)){Need ($seen.Contains($name)) 'Actual FIRST Markdown and JSON'}
$added=@($paths.body);if($family -ceq 'j'){$data=_Conserved-RecordPin;$added+=$data.path}
$seven=@(Conserved-G-Replacements);Conserved-ExactKeys $r.sources @((Conserved-Keys $h.review.sources)+$added);Conserved-ExactKeys $r.replacements $seven
foreach($entry in $r.sources.GetEnumerator()){$name=$entry.Key;$pin=$entry.Value;Conserved-SourceShape $pin $name
if($name -cin $seven){Need ($pin.bytes -gt 0 -and $pin.bytes -le 262144) 'Existing changed Source cap';Conserved-Same $r.replacements[$name] @{before=$h.mapped[$name].source;after=$pin} 'Exact before-H/after-H replacements'}
elseif($family -ceq 'j' -and $name -ceq $data.path){Conserved-Same $pin $data 'Fixed record DATA'}
elseif($name -cne $paths.body){Conserved-Same $pin $h.review.sources[$name] 'Unrelated/installed/static G pins unchanged'}
};return $r
}
function Conserved-H-Body($spec){return _Conserved-Epoch-Body $spec 'h'}

function Conserved-H-CurrentReview($spec){return _Conserved-Epoch-CurrentReview $spec 'h'}

function Conserved-H-Repair($spec){
$p=Conserved-H-Pins;Conserved-Same $spec.path (Conserved-H-Paths).repair 'One separately typed immutable H pointer';$a=Conserved-TextJson $spec;$h=Conserved-H-History
Conserved-ExactKeys $a @('schema','epoch','scope','allocation','qualification_amendment','historical_closed285','failed_g_lint','source_before','reviewed_source','qualification_plan','new_resource_grant_s','refund_s')
foreach($pair in @(@('schema','flylab.m3.contact-qualification-lint-repair.v1'),@('epoch','contact-qualification-lint-repair-20261008-h'),@('scope','PREQUALIFICATION_ONLY'),@('allocation',$h.anchor.direct_science_conserved_allocation),@('qualification_amendment',$h.anchor.direct_science_qualification_amendment),@('historical_closed285',$p.closed),@('failed_g_lint',$h.pins),@('source_before',$p.before),@('qualification_plan',(Conserved-H-Plan)),@('new_resource_grant_s',[long]0),@('refund_s',[long]0))){Conserved-Same $a[$pair[0]] $pair[1] 'Immutable H repair'}
$null=Conserved-H-ReviewMetadata $a.reviewed_source $h;$h.amendment=$a;return $h
}
function Conserved-H-Registration($l,$spec,[byte[]]$beforeRaw){return _Conserved-Epoch-Registration $l $spec $beforeRaw 'h'}
function Conserved-H-Caps($l){return _Conserved-Epoch-Caps $l 'h'}
function Conserved-H-Freeze($spec,$a,$repairSpec){
$paths=Conserved-H-Paths;Conserved-Same $spec.path $paths.source 'Literal fresh H freeze';$f=Conserved-TextJson $spec 262144;$m=Conserved-H-Archives;$expected=Conserved-TextJson $m[(Conserved-G-Paths).source].source
$expected.qualification_epoch='contact-qualification-lint-repair-20261008-h';$expected.reviewed_source=$a.reviewed_source;$expected.qualification_amendment=$a.qualification_amendment;$expected.qualification_lint_repair=$repairSpec;$expected.sources=$f.sources;$expected.body_retry=$f.body_retry
Conserved-Same $f $expected 'Whole G metadata plus H fields and both immutable pointers';$r=Conserved-H-CurrentReview $a.reviewed_source;Conserved-Same $f.sources $r.sources 'Entire current H725';Conserved-Same $f.body_retry $r.sources[$paths.body] 'Only reviewed static H';return $f
}

# Private closed literal families; callers cannot supply epochs, offsets or caps.
function _Conserved-Epoch([ValidateSet('g','h','i','j')][string]$family){
Need ($family -cin @('g','h','i','j')) 'Only four literal families, no absent/foreign selector'
$r=_Conserved-Record $family;$e=$r.epoch;$e.paths=$r.paths;return $e
}
function Conserved-G-Progress($l,$a,$pointer,$source){return _Conserved-Epoch-Progress $l $a $pointer $source 'g'}
function Conserved-H-Progress($l,$a,$pointer,$source){return _Conserved-Epoch-Progress $l $a $pointer $source 'h'}
function Conserved-G-Qualification($q,$l){return _Conserved-Epoch-Qualification $q $l 'g'}
function Conserved-H-Qualification($q,$l){return _Conserved-Epoch-Qualification $q $l 'h'}
function Conserved-G-Dispatch($l,$q,$requestSpec,[string]$phase){return _Conserved-Epoch-Dispatch $l $q $requestSpec $phase 'g'}
function Conserved-H-Dispatch($l,$q,$requestSpec,[string]$phase){return _Conserved-Epoch-Dispatch $l $q $requestSpec $phase 'h'}
function Conserved-G-Settlement($l,$q,$requestSpec,$outer){_Conserved-Epoch-Settlement $l $q $requestSpec $outer 'g'}
function Conserved-H-Settlement($l,$q,$requestSpec,$outer){_Conserved-Epoch-Settlement $l $q $requestSpec $outer 'h'}
function Conserved-G-OriginalAccounting($q,$outer,$live){return _Conserved-Epoch-OriginalAccounting $q $outer $live 'g'}
function Conserved-H-OriginalAccounting($q,$outer,$live){return _Conserved-Epoch-OriginalAccounting $q $outer $live 'h'}

function Conserved-I-Paths {(_Conserved-Record 'i').paths}
function Conserved-I-Pins {(_Conserved-Record 'i').pins}
function Conserved-I-HPaths {(_Conserved-Record 'i').predecessor}
function Conserved-I-Archives {return _Conserved-Epoch-Archives 'i'}

function Contact-I-GeneratedPlan([string]$role){
$r=@('lint','math','gate-a','gate-b');$i=[array]::IndexOf($r,$role);Need ($i -ge 0) 'Only four whole I roles';$p=Contact-H-GeneratedPlan $role;$p.call='direct-science-20261008-contact-i-'+$role;$p.output=$taskBase+'/parent-'+$p.call
if($role -cne 'lint'){$p.argv[-1]='--junitxml='+$taskBase+'/parent-contact-learning-i-'+$role+'-junit.xml'};$limits=@(@(40,35,5),@(35,30,5),@(120,110,10),@(360,340,20))[$i];$p.limits.outer_s=[long]$limits[0];$p.limits.child_s=[long]$limits[1];$p.limits.cleanup_s=[long]$limits[2];return $p
}
function Conserved-I-Plan {return @{epoch='contact-preflight-overhead-recovery-20261008-i';roster_epoch='contact-conserved-20261008-e';qualification_roster=(Conserved-Pins).roster;calls=@(@('lint','math','gate-a','gate-b')|ForEach-Object{Contact-I-GeneratedPlan $_});generated_close_s=[long]20;original_release_calls=@('contact-learning-20261008-i-body','contact-learning-20261008-i-body-audit')}}
function Conserved-I-History {return _Conserved-HI-History 'i'}
function Conserved-I-ReviewMetadata($spec,$h){return _Conserved-Epoch-ReviewMetadata $spec $h 'i'}
function Conserved-I-Body($spec){return _Conserved-Epoch-Body $spec 'i'}

function Conserved-I-CurrentReview($spec){return _Conserved-Epoch-CurrentReview $spec 'i'}

function Conserved-I-Recovery($spec){
$p=Conserved-I-Pins;Conserved-Same $spec.path (Conserved-I-Paths).recovery 'One immutable typed I descriptor';$a=Conserved-TextJson $spec;$h=Conserved-I-History
Conserved-ExactKeys $a @('schema','epoch','scope','allocation','qualification_amendment','qualification_lint_repair','historical_closed286','failed_h_lint','source_before','reviewed_source','qualification_plan','new_resource_grant_s','refund_s')
foreach($pair in @(@('schema','flylab.m3.contact-preflight-recovery.v1'),@('epoch','contact-preflight-overhead-recovery-20261008-i'),@('scope','PREQUALIFICATION_ONLY'),@('allocation',$h.anchor.direct_science_conserved_allocation),@('qualification_amendment',$h.anchor.direct_science_qualification_amendment),@('qualification_lint_repair',$h.anchor.direct_science_qualification_lint_repair),@('historical_closed286',$p.closed),@('failed_h_lint',$h.pins),@('source_before',$p.before),@('qualification_plan',(Conserved-I-Plan)),@('new_resource_grant_s',[long]0),@('refund_s',[long]0))){Conserved-Same $a[$pair[0]] $pair[1] 'Exact immutable I descriptor'}
$null=Conserved-I-ReviewMetadata $a.reviewed_source $h;$h.amendment=$a;return $h
}
function Conserved-I-Registration($l,$spec,[byte[]]$beforeRaw){return _Conserved-Epoch-Registration $l $spec $beforeRaw 'i'}
function Conserved-I-Caps($l){return _Conserved-Epoch-Caps $l 'i'}
function Conserved-I-Freeze($spec,$a,$pointer){
$paths=Conserved-I-Paths;Conserved-Same $spec.path $paths.source 'Literal fresh I freeze';$f=Conserved-TextJson $spec 262144;$m=Conserved-I-Archives;$e=Conserved-TextJson $m[(Conserved-H-Paths).source].source
$e.qualification_epoch='contact-preflight-overhead-recovery-20261008-i';$e.reviewed_source=$a.reviewed_source;$e.sources=$f.sources;$e.body_retry=$f.body_retry;$e.preflight_recovery=$pointer;Conserved-Same $f $e 'Whole immutable H clone plus only I pointer';$r=Conserved-I-CurrentReview $a.reviewed_source;Conserved-Same $f.sources $r.sources 'Whole current I726';Conserved-Same $f.body_retry $r.sources[$paths.body] 'Only reviewed static I';return $f
}
function Conserved-I-Progress($l,$a,$pointer,$source){return _Conserved-Epoch-Progress $l $a $pointer $source 'i'}
function Conserved-I-Qualification($q,$l){return _Conserved-Epoch-Qualification $q $l 'i'}
function Conserved-I-Dispatch($l,$q,$requestSpec,[string]$phase){return _Conserved-Epoch-Dispatch $l $q $requestSpec $phase 'i'}
function Conserved-I-Settlement($l,$q,$requestSpec,$outer){_Conserved-Epoch-Settlement $l $q $requestSpec $outer 'i'}
function Conserved-I-OriginalAccounting($q,$outer,$live){return _Conserved-Epoch-OriginalAccounting $q $outer $live 'i'}

function _Conserved-Epoch-Registration($l,$spec,[byte[]]$beforeRaw,[ValidateSet('g','h','i','j')][string]$family){
$f=_Conserved-Epoch $family;$p=& $f.pins;$raw=Conserved-TextRaw $p.closed 524288;Need ($null -ne $beforeRaw -and [Convert]::ToBase64String($beforeRaw) -ceq [Convert]::ToBase64String($raw)) 'Entire exact immutable raw-before'
Need (-not $l.Contains($f.key)) 'No duplicate/alias/overwrite registration';$h=& $f.metadata $spec;Conserved-Closed $l;Conserved-Same $l $h.anchor 'Entire fixed historical closed anchor'
Need ($l.direct_science_generated_used_s+$f.envelope -le 1564 -and $l.generated_shared_remaining_s -ge $f.envelope) 'Whole fixed plan and distinct close20 in both pools';$null=& $f.currentReview $h.amendment.reviewed_source
$after=Conserved-Json ($l|ConvertTo-Json -Depth 100);$after[$f.key]=$spec;Conserved-ExactKeys $after @((Conserved-Keys $l)+@($f.key));foreach($e in $l.GetEnumerator()){Conserved-Same $after[$e.Key] $e.Value 'Only one immutable pointer; all old fields exact'};return $after
}
function _Conserved-Epoch-Caps($l,[ValidateSet('g','h','i','j')][string]$family){
$f=_Conserved-Epoch $family;$h=& $f.metadata $l[$f.key];Conserved-Same $l.direct_science_conserved_allocation $h.amendment.allocation 'Same old allocation'
if($family -cin @('h','i','j')){Conserved-Same $l.direct_science_qualification_amendment $h.amendment.qualification_amendment 'Immutable G pointer'};if($family -cin @('i','j')){Conserved-Same $l.direct_science_qualification_lint_repair $h.amendment.qualification_lint_repair 'Immutable H pointer'};if($family -ceq 'j'){Conserved-Same $l.direct_science_preflight_recovery $h.amendment.preflight_recovery 'Immutable I pointer'}
$e=Conserved-Json ($h.anchor|ConvertTo-Json -Depth 100);$e[$f.key]=$l[$f.key];Conserved-ExactKeys $l @(Conserved-Keys $e);Conserved-Prefix $l $h.anchor;$tail=Conserved-TailKeys;foreach($x in $e.GetEnumerator()){if($x.Key -cnotin $tail){Conserved-Same $l[$x.Key] $x.Value 'Immutable full accounting; no pointer tail exemption'}};Conserved-Totals $l $h.policy $h.grant 2300 15600
return @{generated_global=[long]2300;generated_local=[long]1564;original_global=[long]15600}
}

function _Conserved-HI-History([ValidateSet('h','i')][string]$family){
Need ($family -cin @('h','i')) 'Closed history selector';$isI=$family -ceq 'i'
if($isI){$m=Conserved-I-Archives;$p=Conserved-I-Pins;$paths=Conserved-I-HPaths;$parent=Conserved-H-Repair $m[$paths.repair].source;$r=Conserved-H-ReviewMetadata $m[$paths.review].source $parent;$pointer='direct_science_qualification_lint_repair';$metadata=$m[$paths.repair].source;$priorSpec=$parent.mapped[(Conserved-G-Paths).source].source;$old=Conserved-H-Paths;$oldEpoch='contact-qualification-lint-repair-20261008-h';$expectedCount=725;$duration=[double]13.253;$exit=[long]124}
else{$m=Conserved-H-Archives;$p=Conserved-H-Pins;$paths=Conserved-H-GPaths;$parent=Conserved-G-Amendment $m[$paths.amendment].source;$r=Conserved-G-ReviewMetadata $m[$paths.review].source $parent.history;$pointer='direct_science_qualification_amendment';$metadata=$m[$paths.amendment].source;$priorSpec=$parent.history.mapped[(Conserved-G-FPaths).freeze].source;$old=Conserved-G-Paths;$oldEpoch='contact-qualification-repair-20261008-g';$expectedCount=724;$duration=[double]11.355;$exit=[long]1}
$pins=@{};foreach($x in $paths.GetEnumerator()){$pins[$x.Key]=$m[$x.Value].source};Conserved-Same $parent.amendment.reviewed_source $pins.review 'Executed review pointer';Need ((Conserved-Count $r.sources) -eq $expectedCount) 'Whole historical Source';foreach($name in (Conserved-G-Replacements)){Conserved-Same $r.sources[$name] $m[$name].source 'Exact seven executed archives'}
$registered=Conserved-TextJson $p.registered 524288;$e=Conserved-Json ($parent.anchor|ConvertTo-Json -Depth 100);$e[$pointer]=$metadata;Conserved-Same $registered $e 'Anchor plus sole historical pointer';Conserved-Closed $registered
$prior=Conserved-TextJson $priorSpec;$prior.qualification_epoch=$oldEpoch;$prior.reviewed_source=$pins.review;$prior.qualification_amendment=if($isI){$parent.amendment.qualification_amendment}else{$pins.amendment};if($isI){$prior.qualification_lint_repair=$pins.repair};$prior.sources=$r.sources;$prior.body_retry=$r.sources[$old.body];Conserved-Same (Conserved-TextJson $pins.freeze) $prior 'Entire immutable executed freeze; no live Source permission';if($isI){$null=Conserved-H-Body $r.sources[$old.body];$plan=Contact-H-GeneratedPlan 'lint'}else{$null=Conserved-G-Body $r.sources[$old.body];$plan=Contact-G-GeneratedPlan 'lint'}
$q=Conserved-TextJson $pins.request;Conserved-F-GeneratedRequest $q $plan $pins.freeze $false
if($isI){$admitted=Conserved-Json ($registered|ConvertTo-Json -Depth 100);$admitted.direct_science_calls[$q.call_id]=@{state='admitted';mode='generated';request=$pins.request;reserved_s=[long]15};$admitted.direct_science_reservations=@{};$admitted.direct_science_reservations[$q.call_id]=@{request=$pins.request;reserved_s=[long]15};Conserved-Same (Conserved-TextJson $p.admitted 524288) $admitted 'Entire registered to admitted H'}
$d=@{cost_id=$q.call_id;mode='generated';actual_outer_full_return_s=$duration;actual_returned_exit_code=$exit;descriptor=$pins.cost};$cost=Conserved-TextJson $pins.cost;$actual=if($isI){_Conserved-ActualCostProof $cost $d 'historical-H-lint'}else{Conserved-CostProof $cost $d};Conserved-Same $actual $q 'Actual known-closed paid failure';foreach($key in @('request','trace','outer','freeze')){$target=if($key -ceq 'freeze'){'source_freeze'}else{$key};Conserved-Same $cost[$target] $pins[$key] 'Immutable actual fee pin'}
$outer=Conserved-TextJson $pins.outer;foreach($pair in @(@('all_owned_ended',$true),@('reservation_released',$true),@('unknown_owned_identities',@{}),@('stdout_eof',$true),@('stderr_eof',$true),@('timeout',$isI),@('worker_exit_code',$exit),@('stdout',$pins.stdout),@('stderr',$pins.stderr))){Conserved-Same $outer[$pair[0]] $pair[1] 'Actual process/Job/EOF closure'}
if($isI){Conserved-Same $outer.first_error.type 'TimeoutError' 'Actual H error';Conserved-Same $outer.first_error.message 'same ENTRY pre-resume child interval expired' 'Actual H error'}else{Conserved-Same $outer.first_error $null 'Actual G no supervisor error'};foreach($key in @('owner_pid','owner_creation_ticks','worker_pid','worker_creation_ticks')){Need (($outer[$key] -is [long] -or $outer[$key] -is [int]) -and $outer[$key] -gt 0) 'Positive actual runtime identity'};foreach($key in @('stdout','stderr')){$raw=Conserved-TextRaw $pins[$key] 262144 $true;if($isI){Need ($raw.Length -eq 0) 'Actual empty H streams'}}
$trace=Conserved-TextJson $pins.trace;foreach($pair in @(@('phase','FINAL'),@('publication','PENDING'),@('first_error',$null),@('next_role_allowed',$false))){Conserved-Same $trace[$pair[0]] $pair[1] 'Actual FINAL failure, never permission'}
$e=Conserved-Json ($registered|ConvertTo-Json -Depth 100);$e.debits=@($e.debits)+@($d);$e.direct_science_calls[$d.cost_id]=@{state='charged';mode='generated';request=$pins.request;reserved_s=[long]15};[double]$generated=0;foreach($debit in $e.debits|Select-Object -Skip 118){if($debit.mode -ceq 'generated'){Need (($debit.actual_outer_full_return_s -is [double] -or $debit.actual_outer_full_return_s -is [long] -or $debit.actual_outer_full_return_s -is [int]) -and [double]::IsFinite([double]$debit.actual_outer_full_return_s) -and $debit.actual_outer_full_return_s -ge 0) 'Typed ordered history';$generated+=[double]$debit.actual_outer_full_return_s}};[double]$stage=[double]$parent.policy.generated_stage_before_s+$generated
$e.direct_science_generated_used_s=$generated;$e.generated_stage_used_s=$stage;$e.generated_shared_cumulative_used_s=[double]$parent.policy.generated_shared_before_s+$generated;$e.generated_shared_remaining_s=2300.0-[double]$parent.policy.generated_shared_baseline_s-$stage;$closed=Conserved-TextJson $p.closed 524288;Conserved-Same $closed $e 'Whole old prefix plus sole actual paid failure; every other field exact';Conserved-Closed $closed;Conserved-Totals $closed $parent.policy $parent.grant 2300 15600
$result=Conserved-TextJson $pins.result;Conserved-Same $result.actual_once_commit.cost $pins.cost 'Actual Native once receipt';if($isI){Conserved-Same $result.independent_closed_readback.closed286 $p.closed 'Actual closed286';Conserved-Same $result.actual_lint.immutable_trace $pins.trace 'Actual H FINAL'}else{Conserved-Same $result.independent_closed_readback.closed285 $p.closed 'Actual closed285'}
return @{anchor=$closed;policy=$parent.policy;grant=$parent.grant;review=$r;mapped=$m;pins=$pins}
}

function Conserved-J-Paths {(_Conserved-Record 'j').paths}
function Conserved-J-Pins {(_Conserved-Record 'j').pins}
function Conserved-J-IPaths {(_Conserved-Record 'j').predecessor}
function Conserved-J-Archives {return _Conserved-Epoch-Archives 'j'}

function Contact-J-GeneratedPlan([string]$role){
$r=@('lint','math','gate-a','gate-b');$i=[array]::IndexOf($r,$role);Need ($i -ge 0) 'Only four whole J roles';$p=Contact-I-GeneratedPlan $role;$p.call='direct-science-20261008-contact-j-'+$role;$p.output=$taskBase+'/parent-'+$p.call
if($role -cne 'lint'){$p.argv[-1]='--junitxml='+$taskBase+'/parent-contact-learning-j-'+$role+'-junit.xml'};return $p
}
function Conserved-J-Plan {return @{epoch='contact-qualification-lint-cleanup-20261008-j';roster_epoch='contact-conserved-20261008-e';qualification_roster=(Conserved-Pins).roster;calls=@(@('lint','math','gate-a','gate-b')|ForEach-Object{Contact-J-GeneratedPlan $_});generated_close_s=[long]20;original_release_calls=@('contact-learning-20261008-j-body','contact-learning-20261008-j-body-audit')}}
function Conserved-J-History {
$m=Conserved-J-Archives;$p=Conserved-J-Pins;$paths=Conserved-J-IPaths;$parent=Conserved-I-Recovery $m[$paths.recovery].source;$r=Conserved-I-ReviewMetadata $m[$paths.review].source $parent
$pins=@{};foreach($x in $paths.GetEnumerator()){$pins[$x.Key]=$m[$x.Value].source};Conserved-Same $parent.amendment.reviewed_source $pins.review 'Executed I review';Need ((Conserved-Count $r.sources) -eq 726) 'Whole historical I726';foreach($name in (Conserved-G-Replacements)){Conserved-Same $r.sources[$name] $m[$name].source 'Seven as-executed I archives including ROOT residual'}
$registered=Conserved-TextJson $p.registered 524288;$e=Conserved-Json ($parent.anchor|ConvertTo-Json -Depth 100);$e.direct_science_preflight_recovery=$pins.recovery;Conserved-Same $registered $e 'Whole closed286 plus sole I pointer';Conserved-Closed $registered
$prior=Conserved-TextJson $parent.mapped[(Conserved-H-Paths).source].source;$prior.qualification_epoch='contact-preflight-overhead-recovery-20261008-i';$prior.reviewed_source=$pins.review;$prior.sources=$r.sources;$prior.body_retry=$r.sources[(Conserved-I-Paths).body];$prior.preflight_recovery=$pins.recovery;Conserved-Same (Conserved-TextJson $pins.freeze) $prior 'Whole executed I freeze H clone';$null=Conserved-I-Body $r.sources[(Conserved-I-Paths).body]
$q=Conserved-TextJson $pins.request;Conserved-F-GeneratedRequest $q (Contact-I-GeneratedPlan 'lint') $pins.freeze $false
$admitted=Conserved-Json ($registered|ConvertTo-Json -Depth 100);$admitted.direct_science_calls[$q.call_id]=@{state='admitted';mode='generated';request=$pins.request;reserved_s=[long]40};$admitted.direct_science_reservations=@{};$admitted.direct_science_reservations[$q.call_id]=@{request=$pins.request;reserved_s=[long]40};Conserved-Same (Conserved-TextJson $p.admitted 524288) $admitted 'Entire actual I admission reservation40'
$d=@{cost_id=$q.call_id;mode='generated';actual_outer_full_return_s=[double]20.284;actual_returned_exit_code=[long]1;descriptor=$pins.cost};$cost=Conserved-TextJson $pins.cost;Conserved-Same (Conserved-CostProof $cost $d) $q 'Ordinary Generated no-timeout I actual fee';foreach($key in @('request','trace','outer','freeze')){$target=if($key -ceq 'freeze'){'source_freeze'}else{$key};Conserved-Same $cost[$target] $pins[$key] 'Exact actual failed I pin'}
$outer=Conserved-TextJson $pins.outer;foreach($pair in @(@('all_owned_ended',$true),@('reservation_released',$true),@('unknown_owned_identities',@{}),@('stdout_eof',$true),@('stderr_eof',$true),@('timeout',$false),@('worker_exit_code',[long]1),@('first_error',$null),@('stdout',$pins.stdout),@('stderr',$pins.stderr))){Conserved-Same $outer[$pair[0]] $pair[1] 'Distinct actual I non-timeout closure'}
foreach($key in @('owner_pid','owner_creation_ticks','worker_pid','worker_creation_ticks')){Need (($outer[$key] -is [long] -or $outer[$key] -is [int]) -and $outer[$key] -gt 0) 'Positive actual I owner/worker'};$stdout=Conserved-TextRaw $pins.stdout 262144;Need ($stdout.Length -eq 1067) 'Actual I stdout1067';$stderr=Conserved-TextRaw $pins.stderr 262144 $true;Need ($stderr.Length -eq 0) 'Actual I stderr0'
$trace=Conserved-TextJson $pins.trace;foreach($pair in @(@('phase','FINAL'),@('publication','PENDING'),@('first_error',$null),@('next_role_allowed',$false))){Conserved-Same $trace[$pair[0]] $pair[1] 'I failed no replay/success'};Need ($trace.dispatch_ms -is [long] -and $trace.first_final_ms -is [long]) 'Typed actual I clocks';Conserved-Same $trace.dispatch_ms ([long]1791453078261) 'Literal actual I dispatch';Conserved-Same $trace.first_final_ms ([long]1791453098545) 'Literal actual I first FINAL'
$e=Conserved-Json ($registered|ConvertTo-Json -Depth 100);$e.debits=@($e.debits)+@($d);$e.direct_science_calls[$d.cost_id]=@{state='charged';mode='generated';request=$pins.request;reserved_s=[long]40};[double]$generated=0
foreach($debit in $e.debits|Select-Object -Skip 118){if($debit.mode -ceq 'generated'){Need (($debit.actual_outer_full_return_s -is [double] -or $debit.actual_outer_full_return_s -is [long] -or $debit.actual_outer_full_return_s -is [int]) -and [double]::IsFinite([double]$debit.actual_outer_full_return_s) -and $debit.actual_outer_full_return_s -ge 0) 'Typed ordered I history';$generated+=[double]$debit.actual_outer_full_return_s}};[double]$stage=[double]$parent.policy.generated_stage_before_s+$generated
$e.direct_science_generated_used_s=$generated;$e.generated_stage_used_s=$stage;$e.generated_shared_cumulative_used_s=[double]$parent.policy.generated_shared_before_s+$generated;$e.generated_shared_remaining_s=2300.0-[double]$parent.policy.generated_shared_baseline_s-$stage;$closed=Conserved-TextJson $p.closed 524288;Conserved-Same $closed $e 'Whole286debits168calls plus sole I20.284/exit1 charged40';Conserved-Closed $closed;Conserved-Totals $closed $parent.policy $parent.grant 2300 15600
$result=Conserved-TextJson $pins.result;Conserved-Same $result.actual_once_commit.cost $pins.cost 'Actual once I Native receipt';Conserved-Same $result.independent_closed_readback.closed287 $p.closed 'Actual whole closed287';Conserved-Same $result.actual_lint.immutable_trace $pins.trace 'Actual first FINAL I trace';Conserved-Same $result.qualification 'NOT_QUALIFIED' 'I permanently failed';Conserved-Same $trace.chunks $result.actual_lint.chunks 'Whole actual I collector chunks no error/truncation';Need ($trace.chunks -is [array] -and $trace.chunks.Length -eq 2) 'Two actual I receipts';$last=$trace.chunks[-1];Conserved-Same $last.received_ms $trace.first_final_ms 'Actual first FINAL receipt';Conserved-ExactKeys $last.result @('chunk_id','wall_time_seconds','exit_code','original_token_count','output');Conserved-Same (Conserved-Json $last.result.output) $outer 'Whole genuine I supervisor output'
return @{anchor=$closed;policy=$parent.policy;grant=$parent.grant;review=$r;mapped=$m;pins=$pins}
}
function Conserved-J-ReviewMetadata($spec,$h){return _Conserved-Epoch-ReviewMetadata $spec $h 'j'}
function Conserved-J-Body($spec){return _Conserved-Epoch-Body $spec 'j'}

function Conserved-J-CurrentReview($spec){return _Conserved-Epoch-CurrentReview $spec 'j'}

function Conserved-J-Cleanup($spec){
$p=Conserved-J-Pins;Conserved-Same $spec.path (Conserved-J-Paths).cleanup 'One typed immutable J cleanup';$a=Conserved-TextJson $spec;$h=Conserved-J-History
Conserved-ExactKeys $a @('schema','epoch','scope','allocation','qualification_amendment','qualification_lint_repair','preflight_recovery','historical_closed287','failed_i_lint','source_before','reviewed_source','qualification_plan','new_resource_grant_s','refund_s')
foreach($pair in @(@('schema','flylab.m3.contact-qualification-lint-cleanup.v1'),@('epoch','contact-qualification-lint-cleanup-20261008-j'),@('scope','PREQUALIFICATION_ONLY'),@('allocation',$h.anchor.direct_science_conserved_allocation),@('qualification_amendment',$h.anchor.direct_science_qualification_amendment),@('qualification_lint_repair',$h.anchor.direct_science_qualification_lint_repair),@('preflight_recovery',$h.anchor.direct_science_preflight_recovery),@('historical_closed287',$p.closed),@('failed_i_lint',$h.pins),@('source_before',$p.before),@('qualification_plan',(Conserved-J-Plan)),@('new_resource_grant_s',[long]0),@('refund_s',[long]0))){Conserved-Same $a[$pair[0]] $pair[1] 'Exact14 immutable J descriptor'}
$null=Conserved-J-ReviewMetadata $a.reviewed_source $h;$h.amendment=$a;return $h
}
function Conserved-J-Registration($l,$spec,[byte[]]$beforeRaw){return _Conserved-Epoch-Registration $l $spec $beforeRaw 'j'}
function Conserved-J-Caps($l){return _Conserved-Epoch-Caps $l 'j'}
function Conserved-J-Freeze($spec,$a,$pointer){
$paths=Conserved-J-Paths;Conserved-Same $spec.path $paths.source 'Literal fresh J freeze';$f=Conserved-TextJson $spec 262144;$m=Conserved-J-Archives;$e=Conserved-TextJson $m[(Conserved-I-Paths).source].source
$e.qualification_epoch='contact-qualification-lint-cleanup-20261008-j';$e.reviewed_source=$a.reviewed_source;$e.sources=$f.sources;$e.body_retry=$f.body_retry;$e.qualification_lint_cleanup=$pointer;Conserved-Same $f $e 'Whole immutable I clone plus J pointer';$r=Conserved-J-CurrentReview $a.reviewed_source;Conserved-Same $f.sources $r.sources 'Whole current J727';Conserved-Same $f.body_retry $r.sources[$paths.body] 'Only reviewed static J';return $f
}
function Conserved-J-Progress($l,$a,$pointer,$source){return _Conserved-Epoch-Progress $l $a $pointer $source 'j'}
function Conserved-J-Qualification($q,$l){return _Conserved-Epoch-Qualification $q $l 'j'}
function Conserved-J-Dispatch($l,$q,$requestSpec,[string]$phase){return _Conserved-Epoch-Dispatch $l $q $requestSpec $phase 'j'}
function Conserved-J-Settlement($l,$q,$requestSpec,$outer){_Conserved-Epoch-Settlement $l $q $requestSpec $outer 'j'}
function Conserved-J-OriginalAccounting($q,$outer,$live){return _Conserved-Epoch-OriginalAccounting $q $outer $live 'j'}

function _Conserved-Epoch-Archives([ValidateSet('h','i','j')][string]$family){
Need ($family -cin @('h','i','j')) 'Closed archive selector';$f=_Conserved-Epoch $family;$p=& $f.pins;$b=Conserved-TextJson $p.before
$info=@{
h=@{schema='flylab.m3.h-qualification-repair.source-before.v1';failed='actual_g_lint_once_settled';closed='closed285';count=[long]25;labels=@('Exact before-H map','All25 immutable before-H pairs','Unique before-H identity','Literal full-content before-H archive');paths=(Conserved-H-GPaths);extra=@($p.registered.path,$p.closed.path,(Conserved-G-Paths).body)}
i=@{schema='flylab.m3.i-preflight-overhead-recovery.source-before.v1';failed='actual_h_lint_once_settled';closed='closed286';count=[long]29;labels=@('Exact immutable pre-I map','Whole29 pairs','Unique pre-I identity','Whole inert pre-I archive');paths=(Conserved-I-HPaths);extra=@($p.closed.path,(Conserved-H-Pins).before.path,(Conserved-H-Pins).closed.path,(Conserved-H-Paths).body,(Conserved-Pins).roster.path)}
j=@{schema='flylab.m3.j-qualification-lint-cleanup.source-before.v1';failed='actual_i_lint_once_settled';closed='closed287';count=[long]29;labels=@('Exact immutable pre-J map','Whole29 pairs','Unique pre-J identity','Whole inert pre-J archive');paths=(Conserved-J-IPaths);extra=@($p.closed.path,(Conserved-I-Pins).before.path,(Conserved-I-Pins).closed.path,(Conserved-I-Paths).body,(Conserved-Pins).roster.path)}
}[$family]
Conserved-ExactKeys $b @('schema','source_metadata_only',$info.failed,'scientific_execution','registration_committed','snapshots',$info.closed)
foreach($pair in @(@('schema',$info.schema),@('source_metadata_only',$true),@($info.failed,$true),@('scientific_execution','NOT_RUN'),@('registration_committed',$false),@($info.closed,$p.closed))){Conserved-Same $b[$pair[0]] $pair[1] $info.labels[0]}
Need ($b.snapshots -is [array] -and $b.snapshots.Length -eq $info.count) $info.labels[1];$m=@{}
for($i=0;$i -lt $info.count;$i++){$x=$b.snapshots[$i];Conserved-ExactKeys $x @('source','archive');Conserved-ExactKeys $x.source @('path','bytes','sha256');Need (-not $m.Contains($x.source.path)) $info.labels[2];Conserved-Same $x.archive @{path=$taskBase+'/parent-'+$family+'-before-20261008/'+$i.ToString('00')+'-'+$x.source.sha256.Substring(0,16)+'.txt';bytes=$x.source.bytes;sha256=$x.source.sha256} $info.labels[3];$null=Conserved-TextRaw $x.archive 524288 $true;$m[$x.source.path]=$x}
$names=@((Conserved-G-Replacements)+@($info.paths.Values)+$info.extra+@('tools/banc_m3_conserved_resource_native_core.ps1','tools/verify_c_banc_m3_startup_ray.py','tools/banc_m3_contact_runtime_journal.py','tools/verify_c_banc_m3_contact_learning.py','tests/test_c_banc_contact_body_retry.py'));Conserved-ExactKeys $m $names;return $m
}
function _Conserved-Epoch-Body($spec,[ValidateSet('g','h','i','j')][string]$family){
$f=_Conserved-Epoch $family;$p=& $f.pins;$paths=$f.paths
$info=@{
g=@{schema='flylab.m3.contact-learning.body-runtime.v4';labels=@('Only static G body','Whole unchanged F candidate/history/limits plus fixed G scope');archive='Conserved-G-Archives';prior=$null;field='qualification_amendment_path';pointer=$paths.amendment;closed='historical_closed284';failed='failed_f_lint';cost=(Conserved-G-FPaths).cost}
h=@{schema='flylab.m3.contact-learning.body-runtime.v5';labels=@('Only static H body','Whole static G: exactly six changes and three additions; G/F lineage intact');archive='Conserved-H-Archives';prior='Conserved-G-Body';priorPath=(Conserved-G-Paths).body;field='qualification_lint_repair_path';pointer=$paths.repair;closed='historical_closed285';failed='failed_g_lint';cost=(Conserved-H-GPaths).cost}
i=@{schema='flylab.m3.contact-learning.body-runtime.v6';labels=@('Only static I body','Entire H object, six changes and three additions');archive='Conserved-I-Archives';prior='Conserved-H-Body';priorPath=(Conserved-H-Paths).body;field='preflight_recovery_path';pointer=$paths.recovery;closed='historical_closed286';failed='failed_h_lint';cost=(Conserved-I-HPaths).cost}
j=@{schema='flylab.m3.contact-learning.body-runtime.v7';labels=@('Only static J body','Whole static I six changes three additions');archive='Conserved-J-Archives';prior='Conserved-I-Body';priorPath=(Conserved-I-Paths).body;field='qualification_lint_cleanup_path';pointer=$paths.cleanup;closed='historical_closed287';failed='failed_i_lint';cost=(Conserved-J-IPaths).cost}
}[$family]
Conserved-Same $spec.path $paths.body $info.labels[0]
if($family -ceq 'g'){$e=Conserved-TextJson (Conserved-F-Pins).body;$m=& $info.archive}else{$m=& $info.archive;$e=& $info.prior $m[$info.priorPath].source}
$e.schema=$info.schema;$e.retry_id='contact-learning-20261008-'+$family;$e.calls=$f.calls;$e.qualification_epoch=$f.epoch;$e.source_review_path=$paths.review;$e.source_freeze_path=$paths.source;$e[$info.field]=$info.pointer;$e[$info.closed]=$p.closed;$e[$info.failed]=$m[$info.cost].source
$r=Conserved-TextJson $spec;Conserved-Same $r $e $info.labels[1];return $r
}
function _Conserved-Epoch-CurrentReview($spec,[ValidateSet('g','h','i','j')][string]$family){
$f=_Conserved-Epoch $family;$history=@{g='Conserved-G-HistoricalReview';h='Conserved-H-History';i='Conserved-I-History';j='Conserved-J-History'}[$family];$h=& $history;$r=& $f.review $spec $h
foreach($pin in $r.sources.Values){Conserved-F-Text $pin};$null=& $f.body $r.sources[$f.paths.body];return $r
}
