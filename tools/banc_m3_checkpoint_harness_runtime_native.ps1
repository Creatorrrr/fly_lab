$kb=$taskBase
$z0='source_freeze';$z1='direct_science_calls';$z2='request';$z3='unknown_owned_identities';$z4='exit_code';$z5='direct_science_blocking_holds';$z6='limits';$z7='worker_creation_state';$z8='anchor';$z9='sources';$za='direct_science_generated_used_s';$zb='all_owned_ended';$zc='science_request';$zd='reservation_released';$ze='call_id';$zf='generated_shared_remaining_s';$zg='debits';$zh='worker_exit_code';$zi='first_error';$zj='schema';$zk='source';$zl='artifacts';$zm='before_ledger';$zn='qualification_epoch';$zo='resume_outcome';$zp='shell_deadline_status_pair_observed';$zq='generated_shared_cumulative_limit_s';$zr='startup_ray_original_local_used_s';$zs='qualification';$zt='resume_attempted';$zu='actual_collector_shell_exit_code';$zv='direct_science_generated_limit_s';$zw='original_jobs_running';$zx='worker_pid';$zy='original_remaining_s';$zz='reviewed_source';$z10='original_reserved_s';$z11='historical_manifest';$z12='as_executed_sources';$z13='original_cumulative_before_s';$z14='worker_started';$z15='first_final_ms';$z16='exec_args';$z17='supervisor_request';$z18='native_supervisor_exit_code';$z19='original_cumulative_limit_s';$z1a='direct_science_reservations';$z1b='actual_outer_full_return_s';$z1c='ROOT_review_cycle_complete';$z1d='stdout_eof';$z1e='stderr_eof';$z1f='body_retry';$z1g='generated_global';$z1h='original_global';$z1i='generated_local';$z1j='dispatch_ms';
function K-Keys($v,[string[]]$a){Conserved-ExactKeys $v $a}
function K-Int($v,[bool]$a=$false){Need (($v -is [long] -or $v -is [int]) -and (-not $a -or $v -gt 0)) 'K exact integer, never bool or float';return [long]$v}
function K-Number($v,[bool]$_m=$true){Need (($v -is [double] -or $v -is [long] -or $v -is [int]) -and [double]::IsFinite([double]$v) -and (-not $_m -or $v -ge 0)) 'K finite typed accounting number';return [double]$v}
function K-Json($a){
$b=if($a -is [byte[]]){$utf8.GetString($a)}else{[string]$a};$c=[Text.Json.JsonDocument]::Parse($b,[Text.Json.JsonDocumentOptions]::new())
try{if($b.Contains('\u')){$null=Conserved-Json $b}else{
$s=[Newtonsoft.Json.Linq.JsonLoadSettings]::new();$s.DuplicatePropertyNameHandling='Error'
$r=[Newtonsoft.Json.JsonTextReader]::new([IO.StringReader]::new($b))
try{$r.CloseInput=$true;$r.DateParseHandling='None';$r.MaxDepth=64;$r.FloatParseHandling='Double'
try{$null=[Newtonsoft.Json.Linq.JToken]::Load($r,$s)}catch{$e=$_.Exception.GetBaseException();if($e -is [Newtonsoft.Json.JsonReaderException] -and $e.Message.StartsWith("Property with the name '",[StringComparison]::Ordinal)){throw 'Duplicate raw JSON event/key rejected'};throw}
}finally{$r.Close()}}}finally{$c.Dispose()}
return $b|ConvertFrom-Json -AsHashtable
}
function K-Clone($v){return K-Json ($v|ConvertTo-Json -Depth 100 -Compress)}
function K-Paths {return @{
recovery="$kb/s28-d19bz-recovery.json";allocation=$kb+'/s28-allocation-r64.json';$zk=$kb+'/s28-s1.json';review=$kb+'/s28-r1.json';contract=$kb+'/c5.json';body=$kb+'/b5.json';roster=$kb+'/t5.json';qualified=$kb+'/s28-qualified-r64.json'
}}
function K-Pins {return @{
$z8=@{path=$kb+'/parent-l12-cold-settled306-HOLD-20261008.json';bytes=[long]480516;sha256='7748aaa0695f12d25b46b613c6d9b8226a1680410b5645d80378aa00393e66a3'};
policy=@{path=$kb+'/parent-direct-science-generated-policy.json';bytes=[long]1496;sha256='693b5193d09a83f5de3b48b99c13199fb3eed2a955c044aa1e97a5a8a9546403'};
grant=@{path=$kb+'/parent-startup-ray-original-grant-20261007-j.json';bytes=[long]2227;sha256='7245e218f5c9abae27a849f880f37bc6013df87ae5e514947887183b4a2dba2f'};
jfreeze=@{path=$kb+'/l13-s1.json';bytes=[long]262136;sha256='51afdfc45124190a03354cbc1892aada5d105959bd1f74c812f802ef147a6998'};
jbody=@{path=$kb+'/l13r2-before/'+$kb+'/b5.json';bytes=[long]19735;sha256='a9628aaf192f3b1085ff7d403f7cb17c9542cab4c32df7bc97c83b60f3a07d77'};
oldroster=@{path=$kb+'/l13r2-before/'+$kb+'/t5.json';bytes=[long]47801;sha256='fcf5b40f439941b0b16cd7f87cdb16cd5be80675d97b4cdcd18283f45a55b764'}
}}

function K-MatchInit {
Need ($script:KManagedSourceAccepted -ceq $true) 'Managed backend only inside accepted owned/admitted action'
$source=@'
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Globalization;
using System.Management.Automation;
using System.Threading;

public static class S23ManagedMatch_3407ce9d34bda83cc22314cc878809c47a85fba832f94e79959de9daba0c8cdd
{
    public static long Entries;
    public static long Matches;
    public static bool Match(object actual, object expected)
    {
        Interlocked.Increment(ref Entries);
        bool result = Visit(actual, expected,
            new HashSet<object>(ReferenceEqualityComparer.Instance),
            new HashSet<object>(ReferenceEqualityComparer.Instance));
        if (result) Interlocked.Increment(ref Matches);
        return result;
    }
    private static bool MapType(Type type)
    {
        return type == typeof(OrderedHashtable) || type == typeof(Hashtable)
            || type == typeof(OrderedDictionary);
    }
    private static bool Extended(object value)
    {
        // Public inspection only. Never unwrap PSObject or execute a member getter.
        PSMemberTypes custom = PSMemberTypes.NoteProperty | PSMemberTypes.AliasProperty
            | PSMemberTypes.ScriptProperty | PSMemberTypes.CodeProperty
            | PSMemberTypes.ScriptMethod | PSMemberTypes.CodeMethod;
        foreach (PSMemberInfo member in PSObject.AsPSObject(value).Members)
            if ((member.MemberType & custom) != 0) return true;
        return false;
    }
    private static bool EntriesOf(IDictionary value, out Dictionary<string, object> entries)
    {
        entries = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (DictionaryEntry item in value)
        {
            if (!(item.Key is string key) ||
                string.Equals(key, "Count", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(key, "Keys", StringComparison.OrdinalIgnoreCase) ||
                !entries.TryAdd(key, item.Value)) return false;
        }
        return entries.Count == value.Count;
    }
    private static bool Visit(object actual, object expected, HashSet<object> left, HashSet<object> right)
    {
        // Never use reference identity as a positive subtree proof.
        if (actual is null || expected is null) return actual is null && expected is null;
        if (actual is PSObject || expected is PSObject || Extended(actual) || Extended(expected)) return false;
        Type a = actual.GetType(), b = expected.GetType();
        if (MapType(a) && MapType(b))
        {
            if (!left.Add(actual)) return false;
            if (!right.Add(expected)) { left.Remove(actual); return false; }
            try
            {
                IDictionary ad = (IDictionary)actual, bd = (IDictionary)expected;
                if (!EntriesOf(ad, out var ae) || !EntriesOf(bd, out var be) || ae.Count != be.Count)
                    return false;
                foreach (var item in be)
                {
                    if (!ae.TryGetValue(item.Key, out var av) || !ad.Contains(item.Key) || !bd.Contains(item.Key))
                        return false;
                    // Check enumeration keys and public lookup, including builtin comparer behavior.
                    if (!Visit(ad[item.Key], bd[item.Key], left, right)) return false;
                }
                return true;
            }
            finally { left.Remove(actual); right.Remove(expected); }
        }
        if (actual is Array aa && expected is Array ba)
        {
            if (aa.Rank != 1 || ba.Rank != 1 || aa.GetLowerBound(0) != 0 || ba.GetLowerBound(0) != 0 || aa.Length != ba.Length)
                return false;
            if (!left.Add(actual)) return false;
            if (!right.Add(expected)) { left.Remove(actual); return false; }
            try
            {
                for (int i = 0; i < aa.Length; i++)
                    if (!Visit(aa.GetValue(i), ba.GetValue(i), left, right)) return false;
                return true;
            }
            finally { left.Remove(actual); right.Remove(expected); }
        }
        if (a != b) return false;
        if (a == typeof(bool)) return (bool)actual == (bool)expected;
        if (a == typeof(string)) return LanguagePrimitives.Equals(actual, expected, false, CultureInfo.InvariantCulture);
        if (a == typeof(int)) return (int)actual == (int)expected;
        if (a == typeof(long)) return (long)actual == (long)expected;
        if (a == typeof(double))
        {
            double x = (double)actual, y = (double)expected;
            return double.IsFinite(x) && double.IsFinite(y) && x == y;
        }
        return false;
    }
}
'@
$j='3407ce9d34bda83cc22314cc878809c47a85fba832f94e79959de9daba0c8cdd';$h='f7cd7de315ef02391a11802f5231b6869ba4a4c8ea741f07f7b4b7631d684f34';$n='S23ManagedMatch_3407ce9d34bda83cc22314cc878809c47a85fba832f94e79959de9daba0c8cdd'
Need ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($utf8.GetBytes($source))).ToLowerInvariant() -ceq $h) 'Literal rendered C# SHA, separate template identity'
if($null -eq $script:KManagedCache){
Need ($null -eq $script:KManagedOrigin) 'Partial backend cache never name-lookup reuse'
$start=[Diagnostics.Stopwatch]::GetTimestamp()
$types=@(Add-Type -TypeDefinition $source -PassThru -ErrorAction Stop -WarningAction Stop)
$end=[Diagnostics.Stopwatch]::GetTimestamp()
Need ($types.Length -eq 1 -and $types[0] -is [type]) 'Actual Add-Type returned one Type'
$type=$types[0];$assembly=$type.Assembly
Need ($type.FullName -ceq $n -and $assembly -is [Reflection.Assembly]) 'Actual returned type/assembly identity'
$method=$type.GetMethod('Match',[Reflection.BindingFlags]'Public,Static',$null,[type[]]@([object],[object]),$null)
Need ($method -is [Reflection.MethodInfo] -and $method.DeclaringType -eq $type -and $method.IsPublic -and $method.IsStatic -and $method.ReturnType -eq [bool] -and -not $method.IsGenericMethod) 'Exact public static Boolean Match(object,object)'
$p=$method.GetParameters();Need ($p.Length -eq 2 -and $p[0].ParameterType -eq [object] -and $p[1].ParameterType -eq [object]) 'Exact backend signature'
$script:R41=[long[]]@(0,0);$script:KManagedOrigin=[object[]]@($type,$assembly,$method,$source,$h,$j)
$script:KManagedCache=@{type=$type;assembly=$assembly;method=$method;source=$source;source_hash=$h;template_hash=$j;compile_start_ticks=$start;compile_end_ticks=$end;frequency=[long][Diagnostics.Stopwatch]::Frequency;warm_reuses=[long]0}
}else{
$cache=$script:KManagedCache;$origin=$script:KManagedOrigin
Need ($cache -is [Collections.IDictionary] -and $cache.Count -eq 10 -and $origin -is [array] -and $origin.Length -eq 6) 'Whole backend cache shape'
foreach($name in @('type','assembly','method','source','source_hash','template_hash','compile_start_ticks','compile_end_ticks','frequency','warm_reuses')){Need ($cache.Contains($name)) 'Complete backend cache fields'}
Need ([object]::ReferenceEquals($cache.type,$origin[0]) -and [object]::ReferenceEquals($cache.assembly,$origin[1]) -and [object]::ReferenceEquals($cache.method,$origin[2])) 'No foreign Type/Assembly/Method cache'
Need ($cache.type -is [type] -and $cache.assembly -is [Reflection.Assembly] -and $cache.method -is [Reflection.MethodInfo] -and $cache.type.Assembly -eq $cache.assembly -and $cache.method.DeclaringType -eq $cache.type -and $cache.type.FullName -ceq $n -and $cache.method.IsPublic -and $cache.method.IsStatic -and $cache.method.ReturnType -eq [bool]) 'Warm exact actual returned backend identity'
Need ([string]::Equals($cache.source,$source,[StringComparison]::Ordinal) -and [string]::Equals($origin[3],$source,[StringComparison]::Ordinal) -and $cache.source_hash -ceq $h -and $origin[4] -ceq $h -and $cache.template_hash -ceq $j -and $origin[5] -ceq $j) 'No stale/partial literal backend cache'
foreach($name in @('compile_start_ticks','compile_end_ticks','frequency','warm_reuses')){Need ($cache[$name] -is [long]) 'Exact backend observation integer'}
Need ($cache.compile_start_ticks -ge 0 -and $cache.compile_end_ticks -ge $cache.compile_start_ticks -and $cache.frequency -eq [Diagnostics.Stopwatch]::Frequency -and $cache.frequency -gt 0 -and $cache.warm_reuses -ge 0) 'Monotonic actual compile/warm provenance'
$p=$cache.method.GetParameters();Need ($p.Length -eq 2 -and $p[0].ParameterType -eq [object] -and $p[1].ParameterType -eq [object] -and -not $cache.method.IsGenericMethod) 'Warm exact signature'
$cache.warm_reuses++
}
}
function K-Same($actual,$expected,[string]$label){
K-MatchInit
Need ($script:R41 -is [long[]] -and $script:R41.Length -eq 2 -and $script:R41[0] -ge 0 -and $script:R41[1] -ge 0 -and $script:R41[1] -le $script:R41[0] -and $script:R41[0] -lt [long]::MaxValue) 'Reference counters'
$script:R41[0]++;Same $actual $expected $label;$script:R41[1]++
}
function K-Key {return 'direct_science_checkpoint_harness_s28_recovery'}
function K-AllocationKey {return 'direct_science_checkpoint_harness_s28_allocation'}
function K-QualificationKey {return 'direct_science_checkpoint_harness_s28_qualification'}
function K-Epoch {return 'contact-checkpoint-harness-20261009-s28'}
function K-HistoryCaps {return @{$z1g=[long]3400;$z1i=[long]2664;$z1h=[long]14500}}
function K-OldCaps {return @{$z1g=[long]6530;$z1i=[long]5794;$z1h=[long]13370}}
function K-NewCaps {return D19-Caps}
function K-Calls {return @(foreach($p in 'bootstrap-r64 r64-lint r64-math r64-gate-a r64-gate-b'.Split(' ')){'direct-science-20261009-contact-s28-'+$p})+@('contact-learning-20261009-s28-r64-body','contact-learning-20261009-s28-r64-body-audit')}
function K-PinShape($_p,[string]$a){K-Keys $_p @('path','bytes','sha256');Same $_p.path $a 'Literal K metadata path';Need ((K-Int $_p.bytes) -gt 0 -and $_p.bytes -le 524288 -and $_p.sha256 -is [string] -and $_p.sha256 -cmatch '^[0-9a-f]{64}$') 'Typed bounded K pin'}
function K-Raw($_p,[long]$a=262144,[bool]$b=$false,$o=$false){
K-Keys $_p @('path','bytes','sha256');Need ((K-Int $_p.bytes) -ge 0 -and $_p.bytes -le $a -and ($b -or $_p.bytes -gt 0)) 'Bounded K pin';Need ($_p.sha256 -is [string] -and $_p.sha256 -cmatch '^[0-9a-f]{64}$') 'SHA pin'
Need ($_p.path -is [string] -and $_p.path -cmatch '^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$') 'Relative text path';$c=K-SourceRaw (Safe $_p.path) $a $o
Same ([long]$c.Length) ([long]$_p.bytes) 'Whole K evidence bytes';Same ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($c)).ToLowerInvariant()) $_p.sha256 'Whole K SHA';return ,$c
}
function K-LedgerPaths {$kn='verification/banc-natural-gait-m3-20260919';return @(foreach($p in '!00terminal319-before!13 !01admitted318-before!13 !01settlement-pre-backup!13 !01settlement-replace-backup!13 !0212-root/a318!13 !0214-root/a318!13 !0427-!06318-HOLD!12 !0213-root/a318!13 !0428-anchor318-before-recovery!13 !0214-root/a318!13 !0213-root/a318!13 !0212-root/a318!13 !0211-root/a318!13 !0210-root/a318!13 !029-root/a318!13 !028-root/a318!13 !027-root/a318!13 !026-root/a318!13 !025-root/a318!13 !024-root/a318!13 !0428-anchor318-before-recovery!13 !0428-recovered318-before-bootstrap!13 !0328-!06319!0920261009!13 !0328-!07323!1020261009!13 !0328-!08324!1120261009!13 !0428-!06319-HOLD!12 !0427-anchor317-before-recovery!13 !0427-recovered317-before-bootstrap!13 !0327-!06318!0920261009!13 !0327-!07322!1020261009!13 !0327-!08323!1120261009!13 !0427-!06318-HOLD!12 !0425-!06317-HOLD!12 !0426-anchor317-before-recovery!13 !0426-recovered317-before-bootstrap!13 !0326-!06318!0920261009!13 !0326-!07322!1020261009!13 !0326-!08323!1120261009!13 !0426-!06318-HOLD!12 !0424-!06316-HOLD!12 !0425-anchor316-before-recovery!13 !0425-recovered316-before-bootstrap!13 !0325-!06317!0920261009!13 !0325-!07321!1020261009!13 !0325-!08322!1120261009!13 !0425-!06317-HOLD!12 !0423-!06315-HOLD!12 !0424-anchor315-before-recovery!13 !0424-recovered315-before-bootstrap!13 !0324-!06316!0920261009!13 !0324-!07320!1020261009!13 !0324-!08321!1120261009!13 !0424-!06316-HOLD!12'.Split(' ')){$p.Replace('!13','.json').Replace('!12','-20261009.json').Replace('!11','-before-audit-').Replace('!10','-before-body-').Replace('!09','-before-allocation-').Replace('!08','body-settled').Replace('!07','qualification-settled').Replace('!06','bootstrap-settled').Replace('!05','@/').Replace('!04','@/s').Replace('!03','@/parent-s').Replace('!02','@/s28-f').Replace('!01','@/s28-f15-root/').Replace('!00','@/s28-f15b-root/').Replace('@/',($kn+'/'))}) + @(foreach($p in '^l12-cold-settled306-HOLD@ ^l12-bootstrap-admitted305=cold-settlement@ ^l11-%305-HOLD@ ^l10-cold-settled304-HOLD@ ^l10-bootstrap-admitted303=cold-settlement@ ^l9r2-%303-HOLD@ ^l9-cold-settled302-HOLD@ ^l8-%301-HOLD@ ^l7-cold-settled300-HOLD@ ^e358-execution-ledger ^l13r2-%307=allocation@ ^l13r2-?311=body@ ^l13r2-!312=audit@ ^l9-bootstrap-admitted301=cold-settlement@ ^l7-bootstrap-admitted299=cold-settlement@ ^l6-math-settled299-HOLD@ ^l3-%292=allocation@ generated-l13r2-allocation-before generated-l13r2-original-before s16-%310-HOLD~ s16-%310-HOLD~ ^s15-%309=allocation~ ^s15-?313=body~ ^s15-!314=audit~ ^l13r2-%307-HOLD@ s14-anchor307-after-repair400 ^s14-%308=allocation~ ^s14-?312=body~ ^s14-!313=audit~ s14-%308-HOLD~ s15-%309-HOLD~ ^s16-%310=allocation~ ^s16-?314=body~ ^s16-!315=audit~ s16-anchor309-after-repair400-v2 s16-recovered309=bootstrap ^s17-%311=allocation~ ^s17-?315=body~ ^s17-!316=audit~ s17-%311-HOLD~ s18-anchor311-after-repair20 s17-recovered310=bootstrap ^s18-%312=allocation~ ^s18-?316=body~ ^s18-!317=audit~ s18-%312-HOLD~ s19-anchor312-after-repair200 s18-recovered311=bootstrap ^s19-%313=allocation~ ^s19-?317=body~ ^s19-!318=audit~ s19-%313-HOLD~ s22-anchor313-after-repair110 s19-recovered312=bootstrap ^s22-%314=allocation~ ^s22-?318=body~ ^s22-!319=audit~ s22-%314-HOLD~ s23-anchor314-after-grant2000 ^s23-%315=allocation~ ^s23-?319=body~ ^s23-!320=audit~'.Split(' ')){"$kb/$($p.Replace('=','-before-').Replace('^','parent-').Replace('@','-20261008').Replace('~','-20261009').Replace('%','bootstrap-settled').Replace('?','qualification-settled').Replace('!','body-settled')).json"}) + @(($kn+'/s28-f16-root/before/'+$kn+'/parent-e358-execution-ledger.json'),($kn+'/s28-f15c-root/before-full4-canonical319.json'),($kn+'/s28-f15c-root/full4-admitted319.json'),($kn+'/s28-f18-root/b320.json'),($kn+'/s28-f18-root/a320.json'),($kn+'/s28-f18-root/t321.json')) +@(foreach($p in 'before-admission-canonical321 admitted321 terminal322'.Split(' ')){$kn+'/s28-f18-root/residual02/r3-'+$p+'.json'}) + @((K-Read $script:KCurrentContract).d19_metadata_paths -replace '^','verification/')}
$kn='verification/banc-natural-gait-m3-20260919'
$script:KCurrentContract=[ordered]@{path=($kn+'/c5.json');bytes=[long]130832;sha256='9398fcc3a61b0e8019b2599c40ca8dd7e0ebaa4e60302b71a74c2dd7a1a32d78'}
$script:KHistoryAuthority=[ordered]@{path=($kn+'/s28-f15b-root/before/'+$kn+'/c5.json');bytes=[long]124922;sha256='9d65539915dbe98563ae5599df9d91c422541411f51608bc5ebe14fab8655ed8'}
$script:KFrozenPins='p/pins/previous_recovery c/d13/recovery/previous_source c/d12/known/before p/pins/refinement_anchor c/d12/known/tool_return @5 p/pins/catalog c/d12/gate/actual_known_layout_compatibility_diagnostic c/d12/gate/journal c/d12/gate/source_model c/d12/recovery/previous_recovery c/d12/recovery/previous_source p/pins/original p/pins/repair c/d12/recovery/catalog p/latest_failure c/d12/recovery/technical_refusal c/d13/tool/request p/pins/previous_source c/d13/known/tool_return c/d13/known/actual_journal p/pins/previous_contract c/d13/gate/actual_known_compiler_type_refusal c/d13/recovery/before_ledger c/d13/recovery/catalog c/d13/recovery/technical_refusal anchor v/source_before p/pins/previous_catalog p/pins/technical_refusal @30'.Split(' ');$script:KFrozenMissing=[ordered]@{path='verification/banc-natural-gait-m3-20260919/s28-f13-root/c5-before.json';bytes=[long]74674;sha256='3a87bc0f34b8a36ddd5153940f2b17fccbdd990b32459aea28ae92e51be58b0a'}
$script:KHistoryBeforeSource=[ordered]@{path=($kn+'/s28-f15b-root/before/'+$kn+'/s28-s1.json');bytes=[long]262142;sha256='b8f00cb6fe06b0fa611b10b62d60f39d5a8e130887767f9e6c791b0973e35bbc'}
$script:KBeforeSource=[ordered]@{path=($kn+'/s28-f18-root/residual03/inputs/'+$kn+'/s28-s1.json');bytes=[long]262142;sha256='7acb957d85b4987472452f576c2f851ad575b13d6c571b47d9d37f0f55c1c0e3'}
function K-Read($_p,[long]$a=262144){$b=$_p.path -ceq ($kb+'/parent-e358-original-active.lock');Need ($_p.path -is [string] -and ([IO.Path]::GetExtension($_p.path) -ceq '.json' -or $b)) 'Metadata JSON or exact existing ownership token before any operand read';if($b){$a=[Math]::Min($a,16384)};Need ($a -le 262144 -or $_p.path -cin (K-LedgerPaths)) 'Finite named ledger metadata only';return K-Json (K-Raw $_p $a)}
function K-Totals($l,$p,$g,$a){
[double]$_g=0;[double]$b=0;$c=@{};Need ($l[$zg] -is [array]) 'Ordered K debit array'
for($i=0;$i -lt $l[$zg].Length;$i++){$d=$l[$zg][$i];Need ($d.cost_id -is [string] -and -not $c.Contains($d.cost_id) -and $d.mode -cin @('generated','original')) 'Unique typed debit identity';$c[$d.cost_id]=$true;$e=K-Number $d[$z1b];if($i -ge 118 -and $d.mode -ceq 'generated'){$_g+=$e};if($i -ge 165 -and $d.mode -ceq 'original'){$b+=$e}}
[double]$f=[double]$p.generated_stage_before_s+$_g
$v=@{$za=$_g;generated_stage_used_s=$f;generated_shared_cumulative_used_s=[double]$p.generated_shared_before_s+$_g;$zf=[double]$a[$z1g]-[double]$p.generated_shared_baseline_s-$f;original_cumulative_used_s=[double]$g[$z13]+$b;original_stage_used_s=[double]$g.original_stage_before_s+$b;$zr=[double]$g.science_local_used_before_s+$b;$zy=[double]$a[$z1h]-[double]$g[$z13]-$b}
foreach($x in $v.Values){$null=K-Number $x $false};return $v
}
function K-Anchor {
$b=K-Pins;$a=K-Read $b[$z8] 524288;$p=K-Read $b.policy;$g=K-Read $b.grant
Need ($a[$zg].Length -eq 306 -and (Conserved-Count $a[$z1]) -eq 188) 'Literal closed300 charged182';Same $a.direct_science_policy $b.policy 'Immutable policy';$v=K-Totals $a $p $g (K-HistoryCaps);foreach($k in $v.Keys){Same $a[$k] $v[$k] ('Ordered anchor '+$k)};return @{$z8=$a;policy=$p;grant=$g}
}
function K-HistoryPins {return @{closed305=@{path=$kb+'/parent-l11-bootstrap-settled305-HOLD-20261008.json';bytes=[long]479395;sha256='039a2cdf42d4dc9cfbe02daafd6487c8199e8da4bd272ded430626c69108c8fa'};source11=@{path=$kb+'/l11-s1.json';bytes=[long]262136;sha256='fa8ce9f6d795d1bb5b53ecb131ce302b147c3b2d65742a591c1ea3e251e8d877'};source12=@{path=$kb+'/l12-s1.json';bytes=[long]262136;sha256='f89aee03161b4cb6e1916f27d06eaf5a80d3bdbbe60c810fbfe076c5d8243519'};cold12=@{path=$kb+'/l12-cold-actual.json';bytes=[long]2161;sha256='1d5e85e7e541807aaba30df8b226ffd0f035dd1b04c10525c0273530771a621d'};timeout=@{path=$kb+'/l11-actual.json';bytes=[long]1722;sha256='042547cf24bee0d9b644eabae32562a0587f8e73297652acf34598d7426c6091'};closed304=@{path=$kb+'/parent-l10-cold-settled304-HOLD-20261008.json';bytes=[long]478269;sha256='8ee279a65965b578841cdd67e028f5e98017a5ad57fa07506c89a765aa8ff461'};cold10=@{path=$kb+'/l10-cold-actual.json';bytes=[long]2055;sha256='b71204781cbacd2f3794b8c5440ccdf9ca8f7b5c4d793100bf5f71f7ce87c571'};closed303=@{path=$kb+'/parent-l9r2-bootstrap-settled303-HOLD-20261008.json';bytes=[long]477147;sha256='dcd13d8bd47e1f878bc0b83acdad6d7fa9b67023102ae16ef629f84a067b1e7c'};latest=@{path=$kb+'/l9r2-actual.json';bytes=[long]2263;sha256='3a36455df74e5fa367adae1d5775ae6f0115f386e4ae9fb076a2f5aad492223e'};old301=@{path=$kb+'/parent-l8-bootstrap-settled301-HOLD-20261008.json';bytes=[long]474869;sha256='70872adcfb9654c2219f17b80bcd2acfdfadc5b2195cf40fc3dfcd1fe25f46fc'};cold9=@{path=$kb+'/l9-cold-actual.json';bytes=[long]2065;sha256='cf10476b8bb4a9546ead22e330466b5113e269e985bd95dc11f6ebde72796bf4'};old300=@{path=$kb+'/parent-l7-cold-settled300-HOLD-20261008.json';bytes=[long]473757;sha256='19c52087c920c29d2f51c9dc9b7a03ec84127877b6ff70bb8e70c26d72884218'};oldrecovery=@{path=$kb+'/l8-recovery1.json';bytes=[long]2410;sha256='3a2178d3d9a4fba0d0b31d4cb2fc138db3cfc2e9a09785d36d3422fab5485f3b'};oldsource=@{path=$kb+'/l8-s1.json';bytes=[long]262134;sha256='91d3a533e3d73f1aaae7af2bce236724452e7f90cd5e12720f42072948825d12'};report=@{path=$kb+'/l8-actual.json';bytes=[long]2209;sha256='6ee7571754780fa913f7462fbf5725c4ad17eeab1ab17997903f3e1140933dc7'}}}
function K-ColdPaths {return @{witness=$kb+'/l7-cold-witness.json';$zk=$kb+'/l7-s2.json';legacy=$kb+'/parent-l6-math-settled299-HOLD-20261008.json';report=$kb+'/l7-cold-actual.json';$z2=$kb+'/parent-direct-science-20261008-contact-l7-bootstrap-request.json';trace=$kb+'/parent-direct-science-20261008-contact-l7-bootstrap-genuine-return.json';recovery=$kb+'/l7-recovery1.json';cost=$kb+'/parent-direct-science-20261008-contact-l7-bootstrap-full-return.json';before=$kb+'/parent-l7-bootstrap-admitted299-before-cold-settlement-20261008.json'}}
$script:KPC=[Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
$script:KPB=$false;$script:KPJ=$null
function K-HRead($a,[long]$b=262144){$v=K-Read $a $b;$script:KPO.Add(@{pin=(K-Clone $a);cap=$b;kind='json'});return $v}
function K-HRaw($a,[long]$b=262144){$c=$a.bytes -eq 0;if($c){K-EmptyPin $a};$v=K-Raw $a $b $c;$script:KPO.Add(@{pin=(K-Clone $a);cap=$b;kind=if($c){'raw0'}else{'raw'}});return ,$v}
function K-EmptyPin($a){Need ($a.path -cin @('20261008-contact-l11','20261008-contact-l13r2','20261009-contact-s14','20261009-contact-s15','20261009-contact-s16','20261009-contact-s19','20261009-contact-s23','20261009-contact-s24','20261009-contact-s25','20261009-contact-s27'|ForEach-Object {"$kb/parent-direct-science-$_-bootstrap/stderr.bin"})) 'Exact ten empty paths';Same $a @{path=$a.path;bytes=[long]0;sha256='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'} 'Exact empty pin'}
function K-HistoryOperand($a){K-Keys $a @('pin','cap','kind');Need ($a.kind -is [string] -and $a.kind -cin @('json','raw','raw0')) 'Reader kind';if($a.kind -ceq 'raw0'){K-EmptyPin $a.pin}else{K-PinShape $a.pin $a.pin.path};Same (K-Int $a.cap) (K-ReaderCap $a.pin.path) 'Path cap every reader';if($a.kind -ceq 'json'){Need ([IO.Path]::GetExtension($a.pin.path) -ceq '.json') 'JSON namespace'};$null=K-Raw $a.pin $a.cap ($a.kind -ceq 'raw0')}
function K-Failed($p,$rp,$f,$ap,$a,$id,$d,$g,$d0,$d1,$h,$j,$rk){
$_c='direct-science-20261008-contact-'+$id+'-bootstrap';$k=$kb+'/parent-'+$_c;$ar=$kb+'/'+$d+'/'
$r=K-HRead $rp;Need ($id -cin @('l8','l9r2','l11')) 'Three fixed failed cases';$to=$id -ceq 'l11';$l=$id -cne 'l8';$b=if($l){$r}else{$r.bootstrap};Same $(if($l){$r[$z8]}else{$r.ledger}) $ap 'Actual301';Same $r[$z0] $f 'Executed L8 Source';if($to){foreach($pr in @(@('full4','INCOMPLETE'),@($z4,[long]124),@('JUnit','NOT_CREATED'),@('stdout','.F'),@('actual_s',[double]170.664))){Same $r[$pr[0]] $pr[1] 'Actual incomplete timeout'}
Same @($g,$d0,$d1,$f,$ap,$d) @([double]170.664,[long]1791503352604,[long]1791503523268,(K-HistoryPins).source11,(K-HistoryPins).closed305,'l12-before') 'Only fixed actual L11 timeout';K-PinShape $a ($kb+'/l11-recovery1.json')
Same (K-HRead $a) @{$zj='flylab.m3.checkpoint-recovery.v1';epoch='contact-checkpoint-harness-20261008-l11';$zm=(K-HistoryPins).closed304;$z0=$f;failed_bootstrap=(K-HistoryPins).report;exact_l9_cold=(K-HistoryPins).cold9;latest_failure=(K-HistoryPins).latest;exact_l10_cold=(K-HistoryPins).cold10;released_hold='direct-science-20261008-contact-l10-bootstrap';reason='KNOWN_EXACT_ROOT_COLD_L10_FAILURE_RETAINED';new_resource_grant_s=[long]0;refund_s=[long]0} 'Whole actual L11 recovery'
}else{foreach($_a in @(@('passed',[long]$(if($l){1}else{3})),@('failed',[long]$(if($l){3}else{1})),@('errors',[long]0),@('skipped',[long]0))){Same $b[$_a[0]] $_a[1] 'Actual failed4'};Same $b[$(if($l){'actual_s'}else{'fee_s'})] $g 'Actual failed4';if(-not $l){Same $b[$z4] ([long]1) 'Actual failed4'}
}
$m=K-HRead $f
foreach($s in @((K-Paths).body,(K-Paths).roster)){$u=K-Clone $m[$z9][$s];$u.path=$ar+$s;K-PinShape $u ($ar+$s);$null=K-HRaw $u;if($s -ceq (K-Paths).roster){$_o=K-HRead $u}}
foreach($v in @('tools/banc_m3_checkpoint_harness_runtime.py','tools/banc_m3_checkpoint_harness_runtime_native.ps1',($kb+'/parent-direct-science-account.ps1'))){$u=K-Clone $m[$z9][$v];$u.path=$ar+$v;$w=if($v -cin (K-Replacements)[0..1]){65536}else{262144};$null=K-HRaw $u $w}
$cp=$b.cost;$jp=if($to){$null}else{$b.junit};K-PinShape $cp ($k+'-full-return.json');if(-not $to){K-PinShape $jp ($kb+'/parent-contact-learning-'+$id+'-bootstrap-junit.xml')};$c=K-HRead $cp
foreach($_a in @(@($c[$z2],($k+'-request.json')),@($c.trace,($k+'-genuine-return.json')),@($c.outer,($k+'/outer.json')))){K-PinShape $_a[0] $_a[1]}
$q=K-HRead $c[$z2];$t=K-HRead $c.trace;$o=K-HRead $c.outer
$y=@($_o.bootstrap.name)
$z=@('-m','pytest')+@($y|ForEach-Object {'tests/test_c_banc_checkpoint_harness_runtime.py::'+$_})+@('--tb=short','-q',('--junitxml='+$kb+'/parent-contact-learning-'+$id+'-bootstrap-junit.xml'))
Same $q @{$zj='flylab.m3.direct-science.supervision.v1';$ze=$_c;mode='generated';role='execution';$z0=$f;$z6=@{outer_s=[long]190;child_s=[long]170;cleanup_s=[long]20;host_bytes=[long]2147483648;output_cap_bytes=[long]8388608};output=$kb+'/parent-'+$_c;ledger_path=$kb+'/parent-e358-execution-ledger.json';shared_original_lock_path=$kb+'/parent-e358-original-active.lock';generated_behavior='command';generated_argv=$z} 'Whole actual L8 request'
foreach($_a in @(@($zj,'flylab.m3.direct-science.genuine-return.v1'),@($ze,$_c),@('mode','generated'),@($z2,$c[$z2]),@($z0,$f),@('phase','FINAL'),@($zi,$null),@($z1j,$d0),@($z15,$d1),@('full_return_s',$g),@($z4,[long]$(if($to){124}else{1})))){Same $t[$_a[0]] $_a[1] 'Actual L8 trace'}
$cmd='$ErrorActionPreference=''Stop'''+[char]10+'& .venv/Scripts/python.exe -m tools.banc_m3_startup_ray_supervisor --request '+$c[$z2].path+[char]10+'exit $LASTEXITCODE';Same $t[$z16].cmd $cmd 'Actual command';Same $t[$z16].workdir 'C:/Users/ckthd/Dev/Projects/fly_lab' 'Actual root';Same $o[$z2] $c[$z2] 'Actual outer';Need ((K-Closure $o) -and $o[$z4] -eq $(if($to){124}else{1}) -and $o[$z7] -ceq 'RESUMED' -and $o.timeout -ceq $to -and $(if($to){$o[$zi].type -ceq 'TimeoutError' -and $o[$zi].message -ceq 'same ENTRY child interval, owned Job termination required'}else{$null -eq $o[$zi]})) 'Actual parent known closure'
if($to){Same $o.secondary_errors @() 'No timeout secondary error';foreach($v in @('owner_pid','owner_creation_ticks',$zx,'worker_creation_ticks')){$null=K-Int $o[$v] $true};Need ($o.owner_pid -ne $o[$zx]) 'Distinct timeout owner/worker'
foreach($pr in @(@('stdout',[long]2,'56df4f6ec7b11cb0d5f94b41ec7e21b122f4fdf68c717f43ee40be1003f33d0f'),@('stderr',[long]0,'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'))){$u=@{path=$k+'/'+$pr[0]+'.bin';bytes=$pr[1];sha256=$pr[2]};Same $o[$pr[0]] $u 'Exact timeout raw log';Same ($utf8.GetString((K-HRaw $u))) $(if($pr[0] -ceq 'stdout'){'.F'}else{''}) 'Whole actual incomplete log'}
foreach($pr in @(@('cost',$cp),@('trace',$c.trace),@($z2,$c[$z2]),@('outer',$c.outer))){Same $r[$pr[0]] $pr[1] 'Whole timeout correlated record'}}

$ec=@{$zj='flylab.m3.direct-science.full-return.v1';cost_id=$_c;mode='generated';$z2=$c[$z2];$z0=$f;trace=$c.trace;outer=$c.outer;$z1b=$g;$z4=[long]$(if($to){124}else{1});charged_once=$true;cap_exceeded=$false};foreach($key in @($z14,$zd,$zo,$zt,$z7,'creation_attempted',$zb)){$ec[$key]=$o[$key]};Same $c $ec 'Whole once cost'
if(-not $to){[xml]$x=$utf8.GetString((K-HRaw $jp));Need ($x.DocumentElement.PSBase.Name -ceq 'testsuites' -and @($x.testsuites.testsuite).Count -eq 1) 'Whole suite';$suite=$x.testsuites.testsuite;foreach($_a in @(@('tests','4'),@('failures',$(if($l){'3'}else{'1'})),@('errors','0'),@('skipped','0'))){Same $suite.GetAttribute($_a[0]) $_a[1] 'Whole failed4'};$cases=@($suite.testcase);Need ($cases.Length -eq 4) 'Full4';for($i=0;$i -lt 4;$i++){Same $cases[$i].GetAttribute('classname') 'tests.test_c_banc_checkpoint_harness_runtime' 'Class';Same $cases[$i].GetAttribute('name') $y[$i] 'Ordered4';$els=@($cases[$i].ChildNodes|Where-Object {$_.NodeType -eq [Xml.XmlNodeType]::Element});if($h[$i]){Need ($els.Count -eq 0) 'First3 PASS'}else{Need ($els.Count -eq 1 -and $els[0].PSBase.Name -ceq 'failure') 'Sole last failure'}}
if($l){Need ($cases[3].OuterXml.Contains('same parent ENTRY Native deadline; owned descendant termination required')) 'Same ENTRY timeout'}else{K-PinShape $r.nested_failure ($kb+'/l8-nested-last-observation.json');$n=K-HRead $r.nested_failure;Need ($n.returncode -eq 124 -and $n.timeout -ceq $true -and $n[$zb] -ceq $true -and $n[$z1d] -ceq $true -and $n[$z1e] -ceq $true -and (Conserved-Count $n[$z3]) -eq 0) 'Nested closure';Same $n[$zi] @{type='TimeoutError';message='same parent ENTRY Native deadline; owned descendant termination required'} 'Same ENTRY timeout'}
}
$e=K-Clone $p[$z8];$e[$rk]=$a;$e[$z5]=@{};$e[$z5][$_c]='L stage unsuccessful; successor blocked';$e[$z1][$_c]=@{$z2=$c[$z2];state='charged';mode='generated';reserved_s=[long]190};$e[$zg]+=,@{cost_id=$_c;mode='generated';$z1b=$g;actual_returned_exit_code=[long]$(if($to){124}else{1});descriptor=$cp};$totals=K-Totals $e $p.policy $p.grant (K-HistoryCaps);foreach($key in $totals.Keys){$e[$key]=$totals[$key]};$closed=K-HRead $ap 524288;Same $closed $e 'Whole301 old300 retained';Need ((Conserved-Count $closed[$z1]) -eq $j) 'Actual183';return $closed
}
function K-HistoryProof($_p){
$hp=K-HistoryPins;$v=K-HRead $_p
Same $v @{$zj='flylab.m3.checkpoint-recovery.v1';epoch='contact-checkpoint-harness-20261008-l13r2';$zm=(K-Pins)[$z8];$z0=$v[$z0];failed_bootstrap=$hp.report;exact_l9_cold=$hp.cold9;latest_failure=$hp.latest;exact_l10_cold=$hp.cold10;latest_timeout=$hp.timeout;exact_l12_cold=$hp.cold12;released_hold='direct-science-20261008-contact-l12-bootstrap';reason='KNOWN_EXACT_ROOT_COLD_L12_FAILURE_RETAINED';new_resource_grant_s=[long]0;refund_s=[long]0} 'Whole failed recovery';K-PinShape $v[$z0] ($kb+'/l13r2-s1.json')
$p=K-ColdProof;$p[$z8]=K-Failed $p $hp.report $hp.oldsource $hp.old301 $hp.oldrecovery 'l8' 'l9-before' ([double]170.472) ([long]1791495442844) ([long]1791495613316) @($true,$true,$true,$false) 183 'direct_science_checkpoint_harness_l8_recovery'
$p[$z8]=K-Cold9Proof $p
$l=K-HRead $hp.closed303 524288;$rp=$l.direct_science_checkpoint_harness_l9r2_recovery;K-PinShape $rp ($kb+'/l9r2-recovery1.json')
Same $rp @{path=$kb+'/l9r2-recovery1.json';bytes=[long]987;sha256='7dbd315c34dd718bbc85cd2f95c2d4d8c2b05951adc1012ccc4f173874990fe0'} 'Whole actual previous recovery';$rv=K-HRead $rp;$os=(K-HRead $hp.latest)[$z0];$ca=(K-HRead $hp.cold9).anchor
Same $rv @{$zj='flylab.m3.checkpoint-recovery.v1';epoch='contact-checkpoint-harness-20261008-l9r2';$zm=$ca;$z0=$os;failed_bootstrap=$hp.report;exact_l9_cold=$hp.cold9;released_hold='direct-science-20261008-contact-l9-bootstrap';reason='KNOWN_EXACT_ROOT_COLD_L9_FAILURE_RETAINED';new_resource_grant_s=[long]0;refund_s=[long]0} 'Whole old R2 recovery'
$p[$z8]=K-Failed $p $hp.latest $os $hp.closed303 $rp 'l9r2' 'l10-before' ([double]170.146) ([long]1791499127903) ([long]1791499298049) @($false,$false,$true,$false) 185 'direct_science_checkpoint_harness_l9r2_recovery'
$p[$z8]=K-Cold9Proof $p $true;$cl=K-HRead $hp.closed305 524288;$rp=$cl.direct_science_checkpoint_harness_l11_recovery
$p[$z8]=K-Failed $p $hp.timeout $hp.source11 $hp.closed305 $rp 'l11' 'l12-before' ([double]170.664) ([long]1791503352604) ([long]1791503523268) @() 187 'direct_science_checkpoint_harness_l11_recovery';$p[$z8]=K-Cold9Proof $p 'l12';return @{event=$v;$z8=$p[$z8];policy=$p.policy;grant=$p.grant}
}
function K-S15Pins {$__n='verification/banc-natural-gait-m3-20260919/';return @{'original'=@{'path'=($__n+'s27-bootstrap-settled318-HOLD-20261009.json');'sha256'='70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e';'bytes'=[long]495417};'anchor'=@{'bytes'=[long]495417;'sha256'='70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e';'path'=($__n+'s28-anchor318-before-recovery.json')};'catalog'=@{'path'=($__n+'h28.json');'bytes'=[long]261912;'sha256'='e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'};'repair'=@{'path'=($__n+'e28.json');'bytes'=[long]2443;'sha256'='e5e284cbaa1f67f1dd6e14bdf3fe74c5f902d751c387f6850700323c23609c93'};'before'=@{'path'=($__n+'s27-s1.json');'sha256'='35b6cf63018f61f3fde34d47045d16f5e3c97dc53bac64b1fe7814aed0aa61c6';'bytes'=[long]262139};'refinement_anchor'=@{'path'=($__n+'s28-f14-root/a318.json');'bytes'=[long]495559;'sha256'='fadfb1604441f5c2cc481e15b07eeaf10a7a7529cedc8969bc496196ea221274'};'previous_recovery'=@{'path'=($__n+'s28-recovery10.json');'bytes'=[long]1903;'sha256'='27f8b420a0f7cdb3447aac416cd2ac71cf499ee8b5ea719ff05c968fd6f1184e'};'previous_source'=@{'path'=($__n+'s28-f14-root/s1-before.json');'bytes'=[long]262140;'sha256'='f1b802af060e385f03955b5cbba5830cc9ee42ccf757b9800016a45afd7cabfd'};'previous_contract'=@{'path'=($__n+'s28-f14-root/c5-before.json');'bytes'=[long]97623;'sha256'='9690bd2ec7a65d33432071d09d9050c6d949c92d6a20852fc92072737f7dcb41'};'previous_catalog'=@{'path'=($__n+'s28-f14-root/h28-before.json');'bytes'=[long]261912;'sha256'='e7ed396ebea2d616853b35c708669102b1652bf94d1f2fe50d9b117adb89de9e'};'technical_refusal'=@{'path'=($__n+'s28-f13-root/actual-fast-warm-gate.json');'bytes'=[long]2818;'sha256'='042051f83aaa9c347c72d943e76d0958cf22afeccb5fd3c929bf0241cd5f5a1f'}}}
function K-ReaderCap($a){if($a -is [string]){foreach($pf in @(($kb+'/s28-f16-root/before/'),($kb+'/s28-f18-root/before/'))){if($a -ceq ($pf+'verification/banc-natural-gait-m3-20260919/parent-direct-science-account.ps1')){return [long]393216};if($a -cin @(($pf+'tools/banc_m3_checkpoint_harness_runtime.py'),($pf+'tools/banc_m3_checkpoint_harness_runtime_native.ps1'))){return [long]131072}}};
if($a -cin @('tools/banc_m3_checkpoint_harness_runtime.py','tools/banc_m3_checkpoint_harness_runtime_native.ps1')){return [long]131072}
if($a -ceq ($kb+'/parent-direct-science-account.ps1')){return [long]393216}
foreach($n in @(24,25,26,27,28)){$pf='verification/banc-natural-gait-m3-20260919/s'+$n+'-before/';$ps=@(($pf+'tools/banc_m3_checkpoint_harness_runtime.py'),($pf+'tools/banc_m3_checkpoint_harness_runtime_native.ps1'),($pf+'verification/banc-natural-gait-m3-20260919/parent-direct-science-account.ps1'));if($ps -ccontains $a){return (@{($ps[0])=[long]131072;($ps[1])=[long]131072;($ps[2])=[long]393216})[$a]}}
if($a -ceq ($kb+'/parent-e358-original-active.lock')){return [long]16384}
$lp=@(K-LedgerPaths);if($a -cin $lp){return [long]524288}
foreach($p in @('tools/banc_m3_checkpoint_harness_runtime.py','tools/banc_m3_checkpoint_harness_runtime_native.ps1')){if($a -ceq $p -or $a.EndsWith('/'+$p,[StringComparison]::Ordinal)){return [long]65536}}
return [long]262144
}
function K-CatalogAuth($v){
K-Keys $v @($zj,'complete_status','project_execution','rows','Original_arrays','trust_basis','deduplicated_path_cap_kind','parent_catalog');Same $v[$zj] 'flylab.m3.s28.ROOT-static-historical-catalog.v1' 'Catalog schema';Need ($v.rows -is [array] -and $v.rows.Length -eq 779) 'Catalog actual finite rows';$seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal);$paths=[Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal);$policy=@();$empty=0
for($i=0;$i -lt 779;$i++){$a=$v.rows[$i];K-Keys $a @('ordinal','pin','cap','kind','reason');Same (K-Int $a.ordinal) ([long]$i) 'Catalog ordinal';Need ($a.reason -is [string]) 'Reader reason';K-HistoryOperand @{pin=$a.pin;cap=$a.cap;kind=$a.kind};$key=@($a.pin.path,$a.cap,$a.kind)|ConvertTo-Json -Compress;Need ($seen.Add($key)) 'Reader tuple';if($paths.ContainsKey($a.pin.path)){Same $paths[$a.pin.path] $a.pin 'Pin conflict'}else{$paths.Add($a.pin.path,$a.pin)};if($a.kind -ceq 'raw0'){$empty++};$policy+=,@($a.pin.path,$a.pin.bytes,$a.pin.sha256,$a.cap,$a.kind)}
Need ($paths.Count -eq 583 -and $empty -eq 10) 'Actual finite paths/old emptyeight plus S25';return ,$policy
}
function K-PinnedAnchor {$__n='verification/banc-natural-gait-m3-20260919/';
$p=K-S15Pins;$b=K-Pins;$v=K-Read $p.repair;Same $v @{'schema'='flylab.m3.s28.ROOT-zero-resource-source-continuation.v1';'epoch'='contact-checkpoint-harness-20261009-s28';'status'='PROSPECTIVE_METADATA_ONLY_NO_LEDGER_RESOURCE_DELTA';'trust'='ROOT_WHOLE318_ACTUAL_S27_FAILURE_SUPPORT_DECLINED_EXPLICIT_TRUST_NOT_HISTORY_REPLAY';'before_ledger'=@{'bytes'=[long]495417;'sha256'='70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e';'path'=($__n+'parent-e358-execution-ledger.json')};'source_before'=@{'bytes'=[long]262139;'sha256'='35b6cf63018f61f3fde34d47045d16f5e3c97dc53bac64b1fe7814aed0aa61c6';'path'=($__n+'s27-s1.json')};'anchor'=@{'bytes'=[long]495417;'sha256'='70175b4bc56c19412b64500343ebc72bc538a7b3776dd60733d43efeeb59476e';'path'=($__n+'s28-anchor318-before-recovery.json')};'old_caps'=@{'generated_global'=[long]6530;'original_global'=[long]13370;'generated_local'=[long]5794};'new_caps'=@{'generated_global'=[long]6530;'original_global'=[long]13370;'generated_local'=[long]5794};'delta_s'=@{'generated_global'=[long]0;'original_global'=[long]0;'generated_local'=[long]0};'combined_global_s'=[long]19900;'new_resource_grant_s'=[long]0;'refund_s'=[long]0;'reset'=$false;'retained_e23_event'=@{'bytes'=[long]2130;'sha256'='1e70cd7f9280407321f40c5c53d2336a4cd2f1f328ad6d743c435b3f9c9f1d2c';'path'=($__n+'e23.json')};'retained_e23_pointer'='direct_science_checkpoint_extension_s23_2000';'exact_ledger_resource_delta'=@();'canonical_new_extension_pointer'=$false;'provenance'=@{'root_design'=@{'bytes'=[long]8964;'sha256'='d461a02466a69ae131975590ca2610edeac8b7f661d4c329a5b061b38b8a1e58';'path'='docs/M3_28D.md'};'astra_json'=@{'bytes'=[long]107955;'sha256'='567ff76f3033c612efcb1f56a7e0ee955f4867182581e50030fc67c9aa17ce6e';'path'=($__n+'s28-aj.json')};'root_cause'=@{'bytes'=[long]11668;'sha256'='a3fdc896a0f39060f379b08a433b44bb59be17e49c57e3d0de33dbe4f9b12d33';'path'=($__n+'s28-root-cause.json')};'root_reflection'=@{'bytes'=[long]11407;'sha256'='85614debeff6238e5b0516fd303d2179dc01d67ebb7727d11abef5c57c995363';'path'='docs/M3_28R.md'};'astra_markdown'=@{'bytes'=[long]30684;'sha256'='218cf3b65877bb7d47ed9eed16ba99f6f4cf88185c5e3ad9c17818ad4e5f3b39';'path'=($__n+'s28-am.md')};'before_index'=@{'bytes'=[long]4915;'sha256'='7c9ca2a35817b59ab1f91bbf699f48e1ff40a8394044b2c9d911b1e305255bd5';'path'=($__n+'s28-before-index.json')}}} 'Whole exact zero-resource S28 event including canonical descriptor metadata'
Same @($v.before_ledger.bytes,$v.before_ledger.sha256) @($p.original.bytes,$p.original.sha256) 'Canonical before metadata and immutable sealed318 raw identity'
$a=K-Read $p.original 524288;$g=K-Read $b.grant;$pol=K-Read $b.policy
Need ($a[$zg] -is [array] -and  $a[$zg].Length -eq 318 -and (Conserved-Count $a[$z1]) -eq 200) 'Trusted whole318/paid200'
Same $a.direct_science_policy $b.policy 'Immutable policy';Same $a[$z5] @{'direct-science-20261009-contact-s27-bootstrap'='L stage unsuccessful; successor blocked'} 'Exact sole actual S27 HOLD'
foreach($k in (Conserved-OwnerKeys)){if($k -cne $z5){Same $a[$k] @{} 'Closed owners'}};Same $a[$zw] ([long]0) 'Original jobs0';Same $a[$z10] ([long]0) 'Original reserve0'
$c=K-OldCaps;$fields=@{$z1g=$zq;$z1i=$zv;$z1h=$z19};foreach($k in $fields.Keys){Same $a[$fields[$k]] $c[$k] 'Unchanged actual e23 caps'}
$t=K-Totals $a $pol $g $c;foreach($k in $t.Keys){Same $a[$k] $t[$k] 'Ordered actual316 totals offsets118/165'};Same $a.direct_science_checkpoint_extension_s23_2000 $v.retained_e23_event 'Actual e23 pointer retained'
foreach($x in @(@($za,[long]4661411048063999410),@($zf,[long]4654475215397681216),@($zy,[long]4652583079272156350))){Need ($a[$x[0]] -is [double]) 'Actual binary64';Same ([BitConverter]::DoubleToInt64Bits($a[$x[0]])) $x[1] 'Exact ordered binary64 bits'}
$e=K-Read $p.anchor 524288;Same ([Convert]::ToBase64String((K-Raw $p.original 524288))) ([Convert]::ToBase64String((K-Raw $p.anchor 524288))) 'Same whole318 raw bytes; mutable canonical is never re-read';K-Same $e $a 'Whole318 zero resource delta; all history/e23/e25/Original retained'
return @{$z8=$e;policy=$pol;grant=$g}
}
$script:KPredecessor={param($n,$whole)
$kp=K-Paths;$c=@($n.archives|Where-Object {$_.logical_current_before.path -ceq $kp.contract});Need ($c.Length -eq 1) 'Unique predecessor contract';$p=$c[0];$o=K-Read $p.immutable_archive;$s=K-Read $n.adoption_before_Source;$r=@($n.archives|Where-Object {$_.logical_current_before.path -ceq $kp.review});Need ($r.Length -eq 1) 'Unique predecessor review';$r=$r[0];$v=K-Read $r.immutable_archive;foreach($x in @($s.checkpoint_contract,$s.sources[$kp.contract],$v.sources[$kp.contract])){& $whole $x $p.logical_current_before};& $whole $s.sources $v.sources;& $whole $s.reviewed_source $r.logical_current_before;return ,$o
}
$script:KWhole={param($a,$b)
Same $a $b 'Whole types/arrays';if($a -is [double]){Need ($b -is [double] -and [BitConverter]::DoubleToInt64Bits($a) -eq [BitConverter]::DoubleToInt64Bits($b)) 'Whole binary64 bits'};if($a -is [Collections.IDictionary]){Same @($a.Keys) @($b.Keys) 'Whole dictionary order';foreach($k in $a.Keys){& $whole $a[$k] $b[$k]}}elseif($a -is [array]){for($i=0;$i -lt $a.Length;$i++){& $whole $a[$i] $b[$i]}}
}
function K-Recovery($_p){return D19-R $_p;
$d3=$false;try{Need (-not $script:KPB -and $taskRoot -is [string]) 'Root/reentrancy';$d3=$true;$script:KPB=$true;$script:KCatalogManifestContract=$null;K-PinShape $_p (K-Paths).recovery;$v=K-Read $_p;
$whole=$script:KWhole
$frozenContract=K-Read $script:KCurrentContract;$b=$frozenContract.terminal_bridge
foreach($k in @('terminal','admitted','pre_backup','replace_backup','cost','receipt','actor','audit','tool','failure','failed_tool','request')){$null=K-Raw $b[$k] (K-ReaderCap $b[$k].path)}
foreach($x in $b.archives){$null=K-Raw $x.immutable_archive 262144;Same @($x.logical_current_before.bytes,$x.logical_current_before.sha256) @($x.immutable_archive.bytes,$x.immutable_archive.sha256) 'Exact old logical archive identity'}
$t=K-Read $b.terminal 524288;$a=K-Read $b.admitted 524288
Same ([Convert]::ToBase64String((K-Raw $b.admitted 524288))) ([Convert]::ToBase64String((K-Raw $b.pre_backup 524288))) 'Exact admitted backup';Same ([Convert]::ToBase64String((K-Raw $b.pre_backup 524288))) ([Convert]::ToBase64String((K-Raw $b.replace_backup 524288))) 'Distinct exact backups'
$c=K-Read $b.cost;$r=K-Read $b.receipt;$c9=K-Read $b.actor;$d2=K-Read $b.audit;$tool=K-Read $b.tool
& $whole $r.actual_metadata_actor $c9;& $whole $d2.actual_actor $c9
Same @($c9.pid,$c9.creation_filetime_1601) @([long]26436,[long]134361160074558301) 'Metadata actor only'
Same @($d2.first_FINAL,$d2.session_spent,$d2.exit_code,$tool.genuine_return_kind,$tool.session_id,$tool.session_spent) @('396ac0',[long]55000,[long]0,'FIRST_WRITE_STDIN_FINAL',[long]55000,$true) 'Spent real settlement'
foreach($x in @(@('cost','cost'),@('immutable_before','pre_backup'),@('replace_backup','replace_backup'),@('old_request','request'))){& $whole $r[$x[0]] $b[$x[1]]};Same @($r.immutable_after.bytes,$r.immutable_after.sha256) @($b.terminal.bytes,$b.terminal.sha256) 'Actual after and terminal raw identity'
$call='direct-science-20261009-contact-s28-bootstrap';Same @($c.call_id,$c.reported_tool_wall_s,$c.measurement_basis,$c.returned_exit_code,$c.scientific_runtime) @($call,[double]0.4983084,'genuine_exec_return.wall_time_seconds',[long]1,'NOT_RUN') 'Once mixed fee; never legacy FeeProof'
Same @($c.original_supervisor.status,$c.original_supervisor.pid,$c.stdout_eof,$c.stderr_eof) @('NOT_CAPTURED',$null,$false,$false) 'No old identity/timer/EOF'
Need ($t.debits.Length -eq 319 -and (Conserved-Count $t.direct_science_calls) -eq 201 -and @($t.debits|Where-Object {$_.cost_id -ceq $call}).Length -eq 1) 'Terminal319/201/once'
Same $t.direct_science_blocking_holds @{$call=$b.hold} 'Only exact named HOLD';foreach($k in (Conserved-OwnerKeys)){if($k -cne 'direct_science_blocking_holds'){Same $t[$k] @{} 'No other ownership'}};Same @($t.original_jobs_running,$t.original_reserved_s) @([long]0,[long]0) 'No Original jobs/reserve'
& $whole $t.debits[-1] ([ordered]@{actual_outer_full_return_s=[double]0.4983084;mode='generated';descriptor=$b.cost;cost_id=$call;actual_returned_exit_code=[long]1})
$rev=K-Clone $t;foreach($k in $b.nine_delta){if($a.Contains($k)){$rev[$k]=K-Clone $a[$k]}else{$rev.Remove($k)}};& $whole $rev $a
$old=K-Read $script:KHistoryBeforeSource;$oa=$null;$ra=$null;foreach($x in $b.archives){if($x.logical_current_before.path -ceq (K-Paths).review){$oa=K-Read $x.immutable_archive};if($x.logical_current_before.path -ceq ($kb+'/s28-recovery12.json')){$ra=$x.logical_current_before}}
& $whole $old.sources $oa.sources;$c2=K-Clone $script:KHistoryAuthority;$c2.path=(K-Paths).contract;& $whole $old.checkpoint_contract $c2;& $whole $t[(K-Key)] $ra
$nb=(K-Read $script:KCurrentContract).actual320
$oc=& $script:KPredecessor $nb $whole;$ob=& $script:KPredecessor $oc.actual320 $whole
$b9={param($value,$maps)
$out=K-Clone $value;foreach($map in @($maps|Sort-Object {$_.parts.Length} -Descending)){$node=$out;foreach($part in $map.parts){$node=$node[$part]};$keys=@($node.Keys);Same ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($utf8.GetBytes(($keys|ConvertTo-Json -Compress)))).ToLowerInvariant()) $map.keys_sha256 'Fixed bridge original key order';Same @($map.indices|Sort-Object) @((0..($keys.Length-1))|ForEach-Object {[long]$_}) 'Exact finite bridge permutation';$c0=[ordered]@{};foreach($i in $map.indices){$c0[$keys[$i]]=$node[$keys[$i]]};if($map.parts.Length -eq 0){$out=$c0}else{$c3=$out;for($j=0;$j -lt $map.parts.Length-1;$j++){$c3=$c3[$map.parts[$j]]};$c3[$map.parts[-1]]=$c0}};return $out
}
$priorHold=$b.hold;$baseCall=$call;foreach($step in @(@($ob.actual320,[double]115.76,($baseCall+'-r1'),[long]320,[long]202),@($oc.actual320,[double]0.672,($baseCall+'-r2'),[long]321,[long]203),@($nb,[double]123.633,($baseCall+'-r3'),[long]322,[long]204))){$nb=$step[0];$amount=$step[1];Same $nb.old_failed_call $step[2] 'Fixed paid identity';foreach($pin in $nb.raw_pins){$null=K-Raw $pin (K-ReaderCap $pin.path)}
$e=K-Clone $t;Same $e[$z5] @{$call=$priorHold} 'Exact prior HOLD only';$e[$z5].Remove($call);$e[(K-Key)]=$nb.old_continuation
& $whole (& $b9 $e $nb.order_relations.before) (K-Read $nb.before_full4 524288)
$q=K-Read $nb.old_request;$cost=K-Read $nb.old_cost;Same @($q.call_id,$q.mode,$q.limits.outer_s,$cost.cost_id,$cost.request,$cost.charged_once,$cost.actual_outer_full_return_s,$cost.exit_code,$cost.cap_exceeded) @($nb.old_failed_call,'generated',[long]180,$nb.old_failed_call,$nb.old_request,$true,$amount,[long]1,$false) 'Exact real r1 admission and once paid115.76'
$e.direct_science_calls[$nb.old_failed_call]=[ordered]@{request=$nb.old_request;state='admitted';mode=$q.mode;reserved_s=$q.limits.outer_s};$e.direct_science_reservations[$nb.old_failed_call]=[ordered]@{request=$nb.old_request;reserved_s=$q.limits.outer_s}
& $whole (& $b9 $e $nb.order_relations.admitted) (K-Read $nb.admitted_full4 524288)
$capture=K-Read $nb.admission_capture;Need ($capture -is [Collections.IDictionary]) 'Whole fixed actual admission capture';$trace=K-Read $cost.trace;$outer=K-Read $cost.outer;Same @($trace.phase,$trace.first_error,$trace.call_id,$trace.request,$trace.source_freeze,$trace.full_return_s,$trace.exit_code,$outer.request,$outer.exit_code) @('FINAL',$null,$nb.old_failed_call,$nb.old_request,$q.source_freeze,$amount,[long]1,$nb.old_request,[long]1) 'Real immutable request/first FINAL/outer';Need (K-Closure $outer) 'Real r1 known closure';K-FeeValues $cost $q $nb.old_request $trace $outer
$e.direct_science_calls[$nb.old_failed_call].state='charged';$e.direct_science_reservations=@{};$e.direct_science_workers=@{};$e.direct_science_pending_actuals=@{};$e[$z5]=@{};$e[$z5][$nb.old_failed_call]=$nb.old_failed_hold
$e.debits+=,[ordered]@{cost_id=$nb.old_failed_call;descriptor=$nb.old_cost;mode='generated';actual_outer_full_return_s=$cost.actual_outer_full_return_s;actual_returned_exit_code=$cost.exit_code};$c4=K-Totals $e (K-Read (K-Pins).policy) (K-Read (K-Pins).grant) (K-OldCaps);foreach($key in $c4.Keys){$e[$key]=$c4[$key]}
& $whole (& $b9 $e $nb.order_relations.paid) (K-Read $nb.terminal 524288);$t=K-Read $nb.terminal 524288
Same @([long]$t.debits.Length,(Conserved-Count $t.direct_science_calls)) @($step[3],$step[4]) 'Fixed paid counts';$call=$nb.old_failed_call;$priorHold=$nb.old_failed_hold}
$c8=@{anchor=$t;policy=(K-Read (K-Pins).policy);grant=(K-Read (K-Pins).grant)}
$b6=$nb;$b6.event_keys=$b.event_keys;$b6.event_schema=$b.event_schema;$null=K-SourceFence $v.source_freeze
$sf=$v[$z0];K-PinShape $sf (K-Paths)[$zk];$f=K-Read $sf;Same $f[$zn] (K-Epoch) 'Source epoch';Same $f[$z1c] $true 'ROOT Source';$script:KManagedSourceAccepted=$true;$cp=$f.checkpoint_contract;Same $cp $f[$z9][(K-Paths).contract] 'Contract';$ct=K-Read $cp;Same $ct.epoch (K-Epoch) 'Contract epoch';$script:KCatalogManifestContract=K-Clone $cp;$bind=$ct.pinned_checkpoint;K-Keys $bind @('pins','trust','coverage','latest_failure','latest_cost','policy','grant','provenance','repair_delta','post_counts');$p=K-S15Pins;Same $bind.pins $p 'Fixed pins';Same $bind.trust 'ROOT_WHOLE318_ACTUAL_S27_FAILURE_SUPPORT_DECLINED_EXPLICIT_TRUST_NOT_HISTORY_REPLAY' 'Trust boundary';Same $bind.repair_delta @() 'No new repair delta';Same $bind.post_counts ([ordered]@{'debits'=[long]322;'paid_calls'=[long]204;'bootstrap_debits'=[long]323;'bootstrap_calls'=[long]205;'qualified_debits'=[long]327;'qualified_calls'=[long]209;'body_debits'=[long]328;'body_calls'=[long]210;'audit_debits'=[long]329;'audit_calls'=[long]211}) 'Exact S28 baseline and successor counts';foreach($pin in @($bind.coverage,$bind.latest_failure,$bind.latest_cost,$bind.policy,$bind.grant)+@($bind.provenance.Values)){$null=K-Raw $pin 262144};Same @($bind.policy,$bind.grant) @((K-Pins).policy,(K-Pins).grant) 'Policy/grant'
K-Keys $v $b6.event_keys;$b7=@{};foreach($x in $b6.archives){$b7[$x.logical_current_before.path]=$x.logical_current_before}
Same $v @{schema=$b6.event_schema;epoch=(K-Epoch);before_ledger=$b6.terminal;source_freeze=$sf;review=$f.reviewed_source;settlement=$b6.commit_capture;cost=$b6.old_cost;audit=$b6.failure;previous_recovery=$b6.old_continuation;old_source=$b7[(K-Paths).source];old_review=$b7[(K-Paths).review];old_contract=$b7[(K-Paths).contract];root_reviews=$v.root_reviews;sol_reports=$v.sol_reports;old_caps=(K-OldCaps);new_caps=(K-NewCaps);new_resource_grant_s=[long]0;refund_s=[long]0;reset=$false} 'Exact terminal continuation event'

K-PinShape $f[$zz] (K-Paths).review;$rv=K-Read $f[$zz];K-Keys $rv @('schema','epoch','sources','replacements','additions','before_source','root_reflection','astra_review','root_reviews','sol_reports');Same @($f.schema,$rv.schema,$rv.epoch) @('flylab.m3.contact-learning.source-freeze.v1','flylab.m3.checkpoint-reviewed-source.v1',(K-Epoch)) 'Exact Source review schemas and epoch';Same $cp $rv.sources[(K-Paths).contract] 'Source vector contract equals current review';Same $f[$z9] $rv.sources 'Whole Source vector equals review before first catalog and every cache';& $whole $v.root_reviews $rv.root_reviews;& $whole $v.sol_reports $rv.sol_reports;$d4=K-Provenance $rv;
$p=K-S15Pins;$c5=K-Read $p.previous_contract;$b5=$c5.pinned_checkpoint.pins;$c1=K-Read $b5.technical_refusal;$gate=K-Read $p.technical_refusal;$receipt=K-Read $gate.actual_tool_return;$c6=@($b5.previous_source,$b5.previous_contract,$b5.technical_refusal,$c1.actual_tool_return,$c1.actual_known_layout_compatibility_diagnostic,$c1.journal,$gate.actual_tool_return,$gate.actual_known_compiler_type_refusal,$gate.journal,$receipt.request);$rp=K-CatalogVerified (K-Read $p.catalog);$null=K-Raw $p.repair;$null=K-Raw $p[$z8] 524288;$pins=@(foreach($pin in $frozenContract.actual320.frozen31){$row=[ordered]@{};foreach($key in $pin.Keys){$row[$key]=if($key -ceq 'bytes'){[long]$pin[$key]}else{$pin[$key]}};$row})
$key=@($taskRoot,'terminal320-v1',(K-Epoch),$sf,$cp,$f[$zz],$rv.root_reviews,$rv.sol_reports,$d4,$p,$c6,$pins,$b6,(K-Pins).policy,(K-Pins).grant,$rp)|ConvertTo-Json -Depth 30 -Compress
if(-not $script:KPC.ContainsKey($key)){$script:KPC.Clear();$script:KPJ=$null;$script:KPC.Add($key,(K-Clone $c8));$script:KPJ=@($key,($script:KPC[$key]|ConvertTo-Json -Depth 100 -Compress))};$j=$script:KPJ;Need ($script:KPC.Count -eq 1 -and $j -is [array] -and $j.Length -eq 2 -and $j[0] -is [string] -and $j[0] -ceq $key -and $j[1] -is [string] -and $j[1].Length -gt 0) 'Exact serialized companion';K-CatalogJournal 'companion-parse' 'BEGIN';$h=K-Json $j[1];K-CatalogJournal 'companion-parse' 'END';K-Keys $h @($z8,'policy','grant');K-CatalogJournal 'companion-compare' 'BEGIN';K-Same $h $script:KPC[$key] 'Whole normalized cache pair';K-CatalogJournal 'companion-compare' 'END';$h.event=$v;return $h
}catch{$script:KManagedSourceAccepted=$false;$script:KCatalogManifestContract=$null;$script:KPC.Clear();$script:KPJ=$null;throw}finally{if($d3){$script:KPB=$false}}
}
function K-RecoveredSnapshot($_p){return D19-S $_p;$h=K-Recovery $_p;$a=$h[$z8];$a[(K-Key)]=$_p;$call='direct-science-20261009-contact-s28-bootstrap-r3';Same $a[$z5] @{$call=((K-Read $script:KCurrentContract).actual320).old_failed_hold} 'Only exact old-call HOLD';$a[$z5].Remove($call);return $a}

function K-Roster($_z){
$a=K-Read @{path=$kb+'/s28-before/'+(K-Paths).roster;bytes=[long]47801;sha256='53cc880df410197b07f32ae15258c1cd062d6f2bd150955bde6d781b1ecec7d7'};$v=K-Read $_z[$z9][(K-Paths).roster];$b=K-Clone $a;$b.epoch=K-Epoch;Same $v $b 'Whole tested L23/61 and exact A25/B36';Need ($v.math.Length -eq 23 -and $v.gate.Length -eq 61) 'Full inherited roster'
foreach($c in @('math','gate','gate_a','gate_b')){$d=@{};foreach($e in $v[$c]){Same $e.node_id ($e.module+'::'+$e.name) 'Full parameter node';Same $e.classname ($e.module.Substring(0,$e.module.Length-3).Replace('/','.')) 'Full classname';Need (-not $d.Contains($e.node_id)) 'No duplicate node';$d[$e.node_id]=$true}}
foreach($f in $v.indivisible_groups){$g=@($v.gate|Where-Object{$_.module -ceq $f}|ForEach-Object{$_.node_id});Need (@($g|Where-Object{$_ -cnotin @($v.gate_a.node_id)}).Length -eq 0 -or @($g|Where-Object{$_ -cnotin @($v.gate_b.node_id)}).Length -eq 0) 'Whole Native group'}
Same @(@($v.gate_a.node_id+$v.gate_b.node_id)|Sort-Object -CaseSensitive) @($v.gate.node_id|Sort-Object -CaseSensitive) 'Full disjoint gate union';return $v
}
function K-Plan([string]$_c,$_z){
$i=[array]::IndexOf((K-Calls),$_c);Need ($i -ge 0) 'Only seven literal K calls';$a=@(@(4000,3980,20),@(40,35,5),@(120,90,30),@(170,155,15),@(5400,5380,20),@(300,285,15),@(120,100,20));$b=@('execution','lint','numerical','execution','execution','contact_candidate_body','contact_body_audit');$t=$a[$i]
$c=@{outer_s=[long]$t[0];child_s=[long]$t[1];cleanup_s=[long]$t[2];host_bytes=[long]2147483648;output_cap_bytes=[long]8388608};$p=@{call=$_c;role=$b[$i];mode=if($i -lt 5){'generated'}else{'original'};$z6=$c;output=$kb+'/parent-'+$_c}
if($i -eq 1){$p.argv=@('-m','ruff','check','tools/banc_m3_contact_learning.py','tools/verify_c_banc_m3_contact_learning.py','tools/analyze_c_banc_m3_contact_learning.py','tools/banc_m3_contact_learning_admission.py','tools/banc_m3_startup_ray_supervisor.py','tools/verify_c_banc_m3_startup_ray.py','tools/banc_m3_contact_runtime_journal.py','tools/banc_m3_conserved_resource.py','tests/test_c_banc_contact_runtime_journal.py','tests/test_c_banc_contact_learning.py','tests/test_c_banc_contact_metadata.py','tests/test_c_banc_startup_ray_execution.py','tests/test_c_banc_contact_body_retry.py','tools/banc_m3_checkpoint_runtime.py','tests/test_c_banc_checkpoint_runtime.py','tests/banc_m3_checkpoint_fixture.py','tools/banc_m3_checkpoint_harness_runtime.py','tests/test_c_banc_checkpoint_harness_runtime.py','tests/banc_m3_checkpoint_harness_fixture.py')} 
elseif($i -lt 5){$d=@('bootstrap','lint','math','gate_a','gate_b')[$i];$e=@('r64-bootstrap','r64-lint','r64-math','r64-gate-a','r64-gate-b')[$i];$_o=K-Roster $_z;$p.argv=@('-m','pytest')+@($_o[$d].node_id)+@('--tb=short','-q',('--junitxml='+$kb+'/parent-contact-learning-s28-'+$e+'-junit.xml'));if(!$i){$p.argv+='--maxfail=1'}}
else{$ct=K-Read $_z.contract;$p[$z6]=@{};foreach($k in $c.Keys){$p[$z6][$k]=$ct[$z6][$b[$i]][$k]};foreach($_a in @(@('outer_s',$t[0]),@('child_s',$t[1]),@('cleanup_s',$t[2]))){Same $p[$z6][$_a[0]] ([long]$_a[1]) 'Unchanged Original envelope'};$p.output+='/result-supervision'};return $p
}
function K-Identity($q,$_j,$_f){D19-I $q $_j;
K-PinShape $_f (K-Paths)[$zk];$f=K-Read $_f;$p=K-Plan $q[$ze] $f;$a=@($zj,$ze,'mode','role',$z0,$z6,'output','ledger_path','shared_original_lock_path')+$(if($p.mode -ceq 'generated'){@('generated_behavior','generated_argv')}else{@('grant',$zc)});K-Keys $q $a
foreach($_a in @(@($zj,'flylab.m3.direct-science.supervision.v1'),@($z0,$_f),@($ze,$p.call),@('mode',$p.mode),@('role',$p.role),@($z6,$p[$z6]),@('output',$p.output),@('ledger_path',$ledgerName),@('shared_original_lock_path',($kb+'/parent-e358-original-active.lock')))){Same $q[$_a[0]] $_a[1] 'Exact K request identity'}
if($q.mode -ceq 'generated'){Same $q.generated_behavior 'command' 'One exact generated command';Same $q.generated_argv $p.argv 'Whole fixed K argv'}else{Same $q.grant (K-Pins).grant 'Unchanged J grant';$s=K-Read $q[$zc];foreach($k in @($ze,'role','grant',$z0)){Same $q[$k] $s[$k] 'Same science/owner'};foreach($k in $q[$z6].Keys){Same $q[$z6][$k] $s[$z6][$k] 'Same physical envelope'};$null=K-OriginalMetadata $s $q};Same (K-Read $_j) $q 'Intact admitted request';return $f
}
function K-Closure($o){
foreach($k in @($z14,'creation_attempted','job_created',$zt,$zb,$zd,$z1d,$z1e)){Need ($o[$k] -is [bool]) ('Mandatory K typed '+$k)}
Need ($o[$z7] -cin @('NO_CREATE_ATTEMPT','CREATED_SUSPENDED_NOT_RESUMED','RESUMED','UNCERTAIN_CREATION') -and $o[$zo] -cin @('NOT_ATTEMPTED','RESUMED','UNCERTAIN')) 'Finite creation/resume outcome';Need ($o[$z3] -is [Collections.IDictionary]) 'Typed unknown identities';Need ((Conserved-Count $o[$z3]) -le 1024) 'Bounded owned identities';foreach($_e in $o[$z3].GetEnumerator()){Need ($_e.Key -is [string] -and $_e.Key -cmatch '^[1-9][0-9]*$' -and [long]$_e.Key -gt 0) 'Typed unknown PID';$null=K-Int $_e.Value $true};foreach($k in @('owner_pid','owner_creation_ticks')){$null=K-Int $o[$k] $true};Same $o[$zd] $o[$zb] 'Release only with known whole closure'
if($o[$z7] -ceq 'NO_CREATE_ATTEMPT'){foreach($k in @($z14,'creation_attempted',$zt,$z1d,$z1e)){Same $o[$k] $false 'No-create never observed EOF'};foreach($k in @($zx,'worker_creation_ticks',$zh)){Same $o[$k] $null 'No invented identity/exit'};Same $o[$zo] 'NOT_ATTEMPTED' 'No invented resume';Same $o[$z3] @{} 'No invented no-create identities';return ($o[$zb] -and $o[$zd])}
Same $o.creation_attempted $true 'Actual creation attempt';if($o[$z7] -ceq 'UNCERTAIN_CREATION'){
Same $o[$zb] $false 'Unknown liability retained';Same $o[$zd] $false 'Unknown reservation retained';Same $o.job_created $true 'Actual owned Job';Same $o[$z14] $false 'Resume never confirmed';Same $o[$zo] $(if($o[$zt]){'UNCERTAIN'}else{'NOT_ATTEMPTED'}) 'Finite uncertainty phase'
if($o[$zt]){foreach($k in @($zx,'worker_creation_ticks')){$null=K-Int $o[$k] $true};Same $o[$z3][[string]$o[$zx]] $o.worker_creation_ticks 'Retained suspended identity';Need ($o[$zx] -ne $o.owner_pid) 'Distinct observed suspended worker/owner'}else{foreach($k in @($zx,'worker_creation_ticks',$zh)){Same $o[$k] $null 'No unconfirmed identity/exit'};Same $o[$z3] @{} 'No observed identity before confirmed creation';Same $o[$z1d] $false 'No reader before confirmed creation';Same $o[$z1e] $false 'No reader before confirmed creation'};if($null -ne $o[$zh]){$null=K-Int $o[$zh]};return $false
}
foreach($k in @($zx,'worker_creation_ticks')){$null=K-Int $o[$k] $true};Need ($o[$zx] -ne $o.owner_pid) 'Distinct worker/owner';Same $o.job_created $true 'Actual owned Job';if($null -ne $o[$zh]){$null=K-Int $o[$zh]};Same $o[$zt] ($o[$z7] -ceq 'RESUMED') 'Actual resume attempt';Same $o[$z14] ($o[$z7] -ceq 'RESUMED') 'Started means resumed';Same $o[$zo] $(if($o[$z14]){'RESUMED'}else{'NOT_ATTEMPTED'}) 'Actual resume'
if($o[$zb]){$null=K-Int $o[$zh];Same $o[$z3] @{} 'Closed means no unknown identities';Same $o[$z1d] $true 'Known closure requires actual EOF';Same $o[$z1e] $true 'Known closure requires actual EOF'}else{Same $o[$z3][[string]$o[$zx]] $o.worker_creation_ticks 'Retain every unclosed identity'};return ($o[$zb] -and $o[$zd] -and (Conserved-Count $o[$z3]) -eq 0 -and $o[$z1d] -and $o[$z1e])
}
function K-ExitMatch($t,$o,$_r){$null=K-Int $t[$z4];$null=K-Int $o[$z4];if($t[$z4] -eq $o[$z4]){return $true};return ($t[$z4] -eq 1 -and $o[$z4] -eq 124 -and $o[$zh] -eq 124 -and $o.timeout -ceq $true -and $o[$zi].type -ceq 'TimeoutError' -and $o[$zb] -ceq $true -and $o[$zd] -ceq $true -and $o[$z1d] -ceq $true -and $o[$z1e] -ceq $true -and (Conserved-Count $o[$z3]) -eq 0 -and $t[$z16].cmd -is [string] -and $t[$z16].cmd.EndsWith('& .venv/Scripts/python.exe -m tools.banc_m3_startup_ray_supervisor --request '+$_r.path,[StringComparison]::Ordinal))}
function K-FeeValues($_d,$q,$_r,$t,$o){if($q.mode -ceq 'original'){$_s=K-Read $q[$zc];Same $_d[$z2] $q[$zc] 'Same immutable admitted science request';Same $_d[$z17] $_r 'Same admitted supervision request';foreach($a in @('role','grant','inputs','roster','contract',$z1f,'local_debit_offset','resource_increment_s')){Same $_d[$a] $_s[$a] 'Cost admitted science value'}};if($o[$z4] -ne $t[$z4]){Same $_d[$z18] $o[$z4] 'Actual observed Native shell pair';Same $_d[$zu] $t[$z4] 'Actual observed collector shell pair';Same $_d[$zp] $true 'Exact observed shell pair'}}
function K-FeeProof($fp,$a,$_f){return D19-P $fp $a $_f;
K-PinShape $fp ($kb+'/parent-'+$a.cost_id+'-full-return.json');$_d=K-Read $fp;$_r=if($a.mode -ceq 'generated'){$_d[$z2]}else{$_d[$z17]};$q=K-Read $_r;$null=K-Identity $q $_r $_f;$t=K-Read $_d.trace 262144;$o=K-Read $_d.outer 262144
Same $t[$z2] $_r 'Genuine request';Same $t[$z0] $_f 'Genuine source binding';Same $o[$z2] $_r 'Immutable outer binding';Same $t[$zj] 'flylab.m3.direct-science.genuine-return.v1' 'Actual collector';Same $t.phase 'FINAL' 'First genuine FINAL';Same $t[$zi] $null 'No collector uncertainty';$null=K-Int $t[$z1j] $true;$null=K-Int $t[$z15] $true;[double]$b=($t[$z15]-$t[$z1j])/1000.0;Need ((K-Number $b) -gt 0) 'Positive genuine fee';Same $t.full_return_s $b 'Genuine full return'
foreach($_a in @(@('cost_id',$q[$ze]),@('mode',$q.mode),@($z0,$_f),@($z1b,$b),@($z4,$t[$z4]),@('charged_once',$true),@('cap_exceeded',($b -gt $q[$z6].outer_s)))){Same $_d[$_a[0]] $_a[1] 'Exact incurred once cost'}
Same $a @{cost_id=$q[$ze];mode=$q.mode;$z1b=$b;actual_returned_exit_code=$t[$z4];descriptor=$fp} 'Whole exact paid debit';Need (K-ExitMatch $t $o $_r) 'Same actual exit or existing proven shell deadline pair';$_k=@($zj,'cost_id','mode',$z2,$z0,'trace','outer',$z1b,$z4,'charged_once',$zb,$zd,'cap_exceeded',$z14,'creation_attempted',$z7,$zt,$zo);if($q.mode -ceq 'original'){$_k+=@($z17,'role','grant','inputs','roster','contract',$z1f,'local_debit_offset','resource_increment_s')};if($o[$z4] -ne $t[$z4]){$_k+=@($z18,$zu,$zp)};K-Keys $_d $_k;Same $_d[$zj] $(if($q.mode -ceq 'generated'){'flylab.m3.direct-science.full-return.v1'}else{'flylab.m3.contact-learning.full-return.v1'}) 'Exact mode cost schema';Same $o[$zj] 'flylab.m3.direct-science.outer.v1' 'Actual supervisor outer schema';K-FeeValues $_d $q $_r $t $o;foreach($c in @($zb,$zd,$z14,'creation_attempted',$z7,$zt,$zo)){Same $_d[$c] $o[$c] 'Exact cost closure and creation'}return @{q=$q;cost=$_d;outer=$o;closed=(K-Closure $o);$z2=$_r}
}
function K-AccountingHold($l,[string]$a,[string]$_c){$b=$(if($a -ceq 'generated'){'direct_science_'}else{'startup_ray_original_'})+'blocking_holds';if($l[$b][$_c] -is [Collections.IDictionary]){$l[$b][$_c].accounting_hold='Actual closure or resource cap unresolved'}else{$l[$b][$_c]='Actual closure or resource cap unresolved'}}
function K-Ownership($e,$q,$_r,$a,[bool]$b=$false){
$_c=$q[$ze];$c=$q.mode -ceq 'original';$_i=if($c){'startup_ray_original_'}else{'direct_science_'};$wk=$_i+'workers';$rs=$_i+'reservations';$pd=$_i+'pending_actuals';$hd=$_i+'blocking_holds';$w=$a[$wk];$d=$a[$rs];$f=$a[$pd];$g=$a[$hd];$free=(Conserved-Count $w) -eq 0 -and (Conserved-Count $g) -eq 0
if($free -and (Conserved-Count $f) -eq 0 -and -not $b){if(-not $c){$e[$rs]=@{};$e[$rs][$_c]=[ordered]@{$z2=$_r;reserved_s=$q[$z6].outer_s}};return 'admitted'}
if($free -and (Conserved-Count $d) -eq 0){$e[$pd]=@{};if(-not $b){$e[$pd][$_c]=[ordered]@{$z2=$_r;state='actual_pending';ownership_closed=$true}};return $(if($b){'charged'}else{'closed-pending'})}
K-Keys $w @($_c);$v=$w[$_c];$i=@('owner_pid','owner_creation_ticks',$zx,'worker_creation_ticks');$j=@($z2,'state','lock_token')+$i;if($c){$j+=@('grant','worker_max','reserved_s')};K-Keys $v $j;foreach($k in $i){$null=K-Int $v[$k] $true};Need ($v.owner_pid -ne $v[$zx]) 'Distinct owner/worker';K-PinShape $v.lock_token ($kb+'/parent-e358-original-active.lock');$l=K-Read $v.lock_token 16384;Same $l[$zj] 'flylab.m3.direct-science.lock-owner.v1' 'Owned token schema';Same $l[$z2] $_r 'Owned actual request';Same $l[$ze] $_c 'Owned literal call';foreach($ik in @('owner_pid','owner_creation_ticks')){Same $l[$ik] $v[$ik] 'Owned process identity'};Need ($l.entry_counter_s -is [double] -and (K-Number $l.entry_counter_s) -gt 0) 'Same positive original ENTRY'
$m=[ordered]@{$z2=if($c){$q[$zc]}else{$_r}};if($c){$m.grant=$q.grant};$m.state='begun';if($c){$m.worker_max=1L;$m.reserved_s=$q[$z6].outer_s};$m.lock_token=$v.lock_token;foreach($k in $i){$m[$k]=$v[$k]};$_v=[ordered]@{$z2=$m[$z2]};if($c){$_v.grant=$q.grant;$e[$zw]=1L;$e[$z10]=$q[$z6].outer_s};$_v.reserved_s=$q[$z6].outer_s;$e[$wk]=@{};$e[$wk][$_c]=$m;$e[$rs]=@{};$e[$rs][$_c]=$_v
if((Conserved-Count $g) -gt 0){$h=[ordered]@{$z2=$_r;state='ownership_unknown'};foreach($k in $i){$h[$k]=$v[$k]};$e[$hd]=@{};$e[$hd][$_c]=$h;return 'unknown'};Need (-not $b) 'Charged unresolved owner needs HOLD';return 'begun'
}
function K-Validate($l){return D19-V $l;
$h=K-Recovery $l[(K-Key)];$e=$h[$z8];$e[(K-Key)]=$l[(K-Key)];$e[$z5]=@{};$_f=$h.event[$z0];$b=@($l[$zg]|Select-Object -Skip 322);Need ($b.Length -le 7) 'Finite seven-call tail';$c=K-OldCaps;$g=$false;$j=K-Calls
for($i=0;$i -lt $b.Length;$i++){$d=$b[$i];Same $d.cost_id $j[$i] 'Next literal debit';Need (-not $g) 'Failure never success prerequisite';$f=K-FeeProof $d.descriptor $d $_f;$q=$f.q;Need ($f.closed) 'UNCOMMITTED_OWNERSHIP_UNKNOWN without fee/debit/totals';$e[$zg]+=@($d);$e[$z1][$q[$ze]]=@{$z2=$f[$z2];state='charged';mode=$q.mode;reserved_s=$q[$z6].outer_s};$g=$f.cost[$z4] -ne 0 -or $f.cost.cap_exceeded;if($f.cost[$z4] -ne 0){$_x=$(if($q.mode -ceq 'generated'){'direct_science_'}else{'startup_ray_original_'})+'blocking_holds';$e[$_x][$q[$ze]]='L stage unsuccessful; successor blocked'};if($f.cost.cap_exceeded){$hk=$(if($q.mode -ceq 'generated'){'direct_science_'}else{'startup_ray_original_'})+'blocking_holds';K-AccountingHold $e $q.mode $q[$ze]}}
if($l.Contains((K-AllocationKey))){Need ($b.Length -ge 1) 'Allocation follows actual bootstrap';$sp=$l[(K-AllocationKey)];K-PinShape $sp (K-Paths).allocation;$v=K-Read $sp;K-Keys $v @($zj,'epoch',$zm,'recovery',$z0,'bootstrap','bootstrap_junit','old_caps','new_caps','delta_s','combined_global_s','new_resource_grant_s','refund_s');foreach($_a in @(@($zj,'flylab.m3.checkpoint-allocation.v1'),@('epoch',(K-Epoch)),@('recovery',$l[(K-Key)]),@($z0,$_f),@('old_caps',(K-OldCaps)),@('new_caps',(K-NewCaps)),@('delta_s',@{$z1g=[long]0;$z1i=[long]0;$z1h=[long]0}),@('combined_global_s',[long]19900),@('new_resource_grant_s',[long]0),@('refund_s',[long]0),@('bootstrap',$b[0].descriptor))){Same $v[$_a[0]] $_a[1] 'Exact allocation field'}
$_b=K-Read $v[$zm] 524288;Need (-not $_b.Contains((K-AllocationKey)) -and $_b[$zg].Length -eq 323) 'Fresh closed bootstrap before allocation';$bs=K-Validate $_b;K-Eligible $_b $bs;$null=K-Success $_b[$zg][-1] $_f 'bootstrap' $v.bootstrap_junit;$e[(K-AllocationKey)]=$sp;$c=K-NewCaps;$e[$zq]=$c[$z1g];$e[$zv]=$c[$z1i];$e[$z19]=$c[$z1h]
}else{Need ($b.Length -le 1) 'No qualification from old caps'}
if($l.Contains((K-QualificationKey))){Need ($l.Contains((K-AllocationKey)) -and $b.Length -ge 5) 'Whole four costs first';$sp=$l[(K-QualificationKey)];K-PinShape $sp (K-Paths).qualified;K-Qualification (K-Read $sp) $l $_f;$e[(K-QualificationKey)]=$sp};Need ($b.Length -le 5 -or $l.Contains((K-QualificationKey))) 'Original requires K qualification'
$o=K-Totals $e $h.policy $h.grant $c;foreach($k in $o.Keys){$e[$k]=$o[$k]};$p=@((Conserved-Keys $l[$z1])|Where-Object {-not $e[$z1].Contains($_)});Need ($p.Length -le 1) 'Sole next admitted call';$r='charged'
if($p.Length -eq 1){Need (-not $g -and $b.Length -lt 7) 'No successor after failure/end';$_c=$j[$b.Length];Same $p @($_c) 'No old/replay/alias admission';$a=$l[$z1][$_c];K-Keys $a @($z2,'state','mode','reserved_s');$q=K-Read $a[$z2];$null=K-Identity $q $a[$z2] $_f;Same $a @{$z2=$a[$z2];state='admitted';mode=$q.mode;reserved_s=$q[$z6].outer_s} 'Exact whole admission';Need ($b.Length -ne 1 -or $l.Contains((K-AllocationKey))) 'No lint before allocation';Need ($b.Length -ne 5 -or $l.Contains((K-QualificationKey))) 'No body before full qualification';if($b.Length -eq 6){K-ValidBody $b[5] $_f (K-Read $q[$zc])};$e[$z1][$_c]=$a;$r=K-Ownership $e $q $a[$z2] $l}
if($b.Length -gt 0){$d=$b[-1];$s=($d.mode -ceq 'generated' -and ($e[$za] -gt $c[$z1i] -or $e[$zf] -lt 0)) -or ($d.mode -ceq 'original' -and ($e[$zy] -lt 0 -or $e.original_stage_used_s -gt 6300 -or $e[$zr] -gt 6000));if($s){$hk=$(if($d.mode -ceq 'generated'){'direct_science_'}else{'startup_ray_original_'})+'blocking_holds';K-AccountingHold $e $d.mode $d.cost_id}}
K-Same $l $e 'Whole reconstructed K legal state, entire immutable289 retained';return @{caps=$c;completed=$b.Length;phase=$r;failed=$g;$zk=$_f;epoch=(K-Epoch)}
}
function K-Eligible($l,$s){Need ($s.phase -ceq 'charged' -and -not $s.failed) 'Successful known charged prerequisite';foreach($k in (Conserved-OwnerKeys)){Same $l[$k] @{} 'Closed eligible owner'};Need ($l[$zf] -ge 0 -and $l[$zy] -ge 0 -and $l[$za] -le $s.caps[$z1i]) 'Execution remaining resource'}
function K-JUnit($_p,$a){
$b=K-Raw $_p 262144;[xml]$x=$utf8.GetString($b);Need ($x.DocumentElement.PSBase.Name -ceq 'testsuites' -and @($x.testsuites.testsuite).Count -eq 1) 'One whole actual suite';$s=$x.testsuites.testsuite;foreach($k in @('failures','errors','skipped')){Same $s.GetAttribute($k) '0' 'No missing/error/failed/skipped qualification'};Same $s.GetAttribute('tests') ([string]$a.Length) 'Exact count';$c=@($s.testcase);Need ($c.Length -eq $a.Length) 'All actual nodes';for($i=0;$i -lt $a.Length;$i++){Same $c[$i].GetAttribute('classname') $a[$i].classname 'Exact classname';Same $c[$i].GetAttribute('name') $a[$i].name 'Exact parameterized name';Need (@($c[$i].ChildNodes|Where-Object {$_.NodeType -eq [Xml.XmlNodeType]::Element}).Count -eq 0) 'No testcase failure/skip/error'}
}
function K-Success($d,$_f,[string]$a,$b=$null){$f=K-FeeProof $d.descriptor $d $_f;Need ($f.closed -and $f.cost[$z4] -eq 0 -and $f.cost.cap_exceeded -ceq $false -and $f.outer[$z14] -ceq $true -and $f.outer[$z7] -ceq 'RESUMED') 'Successful resumed once paid known closed role';if($a -cne 'lint'){$c=if($a -ceq 'bootstrap'){'r64-bootstrap'}else{'r64-'+$a.Replace('_','-')};K-PinShape $b ($kb+'/parent-contact-learning-s28-'+$c+'-junit.xml');$_o=K-Roster (K-Read $_f);K-JUnit $b $_o[$a]};return $f.q}
function K-Qualification($q,$l,$_f){$bn=(K-Read $script:KCurrentContract).baseline_debits;Need ($bn -is [long] -and $bn -gt 0) 'Exact baseline';K-Keys $q @($zj,'epoch',$z0,'actual_costs','tests');Same $q[$zj] 'flylab.m3.direct-science.qualification.v8' 'Distinct K qualification';Same $q.epoch (K-Epoch) 'K epoch';Same $q[$z0] $_f 'One immutable tested Source';$a=@($l[$zg]|Select-Object -Skip ($bn+1) -First 4);Need ($a.Length -eq 4 -and $q.tests.Length -eq 3) 'Four separate costs/three whole JUnits';Same $q.actual_costs @($a.descriptor) 'Whole four actual fees';$b=@('lint','math','gate_a','gate_b');for($i=0;$i -lt 4;$i++){Same $a[$i].cost_id (K-Calls)[$i+1] 'Exact qualification role';$null=K-Success $a[$i] $_f $b[$i] $(if($i -eq 0){$null}else{$q.tests[$i-1]})}}
function K-Replacements {return @('tools/banc_m3_checkpoint_harness_runtime.py','tools/banc_m3_checkpoint_harness_runtime_native.ps1','tests/banc_m3_checkpoint_fixture.py','tests/banc_m3_checkpoint_harness_fixture.py','tests/test_c_banc_checkpoint_harness_runtime.py','tests/test_c_banc_contact_runtime_journal.py','tools/banc_m3_conserved_resource.py',($kb+'/parent-direct-science-account.ps1'),($kb+'/b5.json'),($kb+'/c5.json'),($kb+'/t5.json'))}
function K-Additions {return ,@()}
function K-SourceRaw([string]$a,[long]$c,$o=$false){
Need ($o -is [bool] -and (-not $o -or $a -cmatch '[\\/](stdout|stderr)\.bin$')) 'Opaque stream'
Need ($c -gt 0 -and $c -le 524288 -and $a -cmatch '^[A-Za-z]:[\\/]' -and -not $a.Substring(2).Contains(':') -and -not $a.StartsWith('\\')) 'Local text drive';foreach($d in $a.Substring(3).Replace('\','/').Split('/')){Need ($d -cmatch '^[A-Za-z0-9_.-]+$' -and $d -cnotin @('.','..') -and -not $d.EndsWith('.') -and $d -cnotmatch '^(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$') 'Source path grammar'}
$a=[IO.Path]::GetFullPath($a);$e=[IO.FileInfo]$a;Need ($e.Exists -and $e.Length -le $c -and -not ($e.Attributes -band [IO.FileAttributes]::Directory)) 'Bounded actual regular Source';$f=$e.Length;$g=$e.LastWriteTimeUtc.Ticks;$h=$e.CreationTimeUtc.Ticks;$u=$e;while($null -ne $u){Need ($u.Exists -and -not ($u.Attributes -band [IO.FileAttributes]::ReparsePoint)) 'No reparse ancestors';$u=if($u -is [IO.DirectoryInfo]){$u.Parent}else{$u.Directory}}
$j=[byte[][]]::new(2);for($i=0;$i -lt 2;$i++){$s=[IO.File]::Open($a,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read);$m=[IO.MemoryStream]::new();$b=[byte[]]::new(65536);try{while(($n=$s.Read($b,0,[int][Math]::Min($b.Length,$c+1-$m.Length))) -gt 0){$m.Write($b,0,$n);Need ($m.Length -le $c) 'Source read cap'};$j[$i]=$m.ToArray()}finally{$s.Dispose();$m.Dispose()}}
$k=[IO.FileInfo]$a;Need ($k.Length -eq $f -and $k.LastWriteTimeUtc.Ticks -eq $g -and $k.CreationTimeUtc.Ticks -eq $h) 'Source stamp/readback';Same ([Convert]::ToBase64String([byte[]]$j[0])) ([Convert]::ToBase64String([byte[]]$j[1])) 'Whole Source reread';$u=$k;while($null -ne $u){Need ($u.Exists -and -not ($u.Attributes -band [IO.FileAttributes]::ReparsePoint)) 'No reparse reread';$u=if($u -is [IO.DirectoryInfo]){$u.Parent}else{$u.Directory}};$l=[byte[]]$j[0];if(-not $o){$null=$utf8.GetString($l)};return ,$l
}
function K-Provenance($rv){
K-Keys $rv @('schema','epoch','sources','replacements','additions','before_source','root_reflection','astra_review','root_reviews','sol_reports');Same $rv.schema 'flylab.m3.checkpoint-reviewed-source.v1' 'Complete current review schema';Same $rv.epoch (K-Epoch) 'Current review epoch';K-Keys $rv.astra_review @('markdown','json')
$cp=$rv.sources[(K-Paths).contract];K-PinShape $cp (K-Paths).contract;Same $cp ([ordered]@{'path'='verification/banc-natural-gait-m3-20260919/c5.json';'bytes'=[long]130832;'sha256'='9398fcc3a61b0e8019b2599c40ca8dd7e0ebaa4e60302b71a74c2dd7a1a32d78'}) 'Whole current compiled contract pin';$ct=K-Read $cp;Same $ct.comparison_profile 'REFERENCE_SAME_V1' 'Reference profile';Same @($ct.schema,$ct.epoch) @('flylab.m3.checkpoint-contract.v1',(K-Epoch)) 'Current provenance contract schema epoch';Same $ct.actual320.adoption_before_Source $script:KBeforeSource 'Compiled adoption predecessor';$p=$ct.provenance;K-Keys $p @('root_design','astra_json','root_cause','root_reflection','astra_markdown','before_index','source_before','immutable_inputs','root_coverage','reader_closure')
Same @($p.astra_markdown,$p.astra_json,$p.root_reflection,$p.root_coverage) @($rv.astra_review.markdown,$rv.astra_review.json,$rv.root_reflection,$rv.root_reflection) 'Current direct provenance equals reviewed fields'
foreach($a in @(@($p.root_design,'docs/M3_28D18.md'),@($p.astra_markdown,($kb+'/s28-d18-astra/review.md')),@($p.astra_json,($kb+'/s28-d18-astra/review.json')),@($p.root_reflection,'docs/M3_28R18.md'))){K-PinShape $a[0] $a[1];$null=K-Raw $a[0] 262144}
$d5=K-Read (K-BoundedPin (K-Paths)[$zk]);Same @($d5.ROOT_review_cycle_complete,$d5.qualification_epoch,$d5.checkpoint_contract,$d5.sources) @($true,(K-Epoch),$cp,$rv.sources) 'Actual accepted Source/review before evidence';Same $d5.reviewed_source (K-BoundedPin (K-Paths).review) 'Actual raw reviewed Source'
$whole=$script:KWhole
$old=K-Read $script:KHistoryAuthority;& $whole $ct.checkpoint_continuity_evidence ([ordered]@{authority=$script:KHistoryAuthority;member='checkpoint_continuity_evidence'})
$pins=@(foreach($pin in $ct.actual320.frozen31){$row=[ordered]@{};foreach($key in $pin.Keys){$row[$key]=if($key -ceq 'bytes'){[long]$pin[$key]}else{$pin[$key]}};$row})
foreach($pin in $pins){$null=K-Raw $pin (K-ReaderCap $pin.path)}
foreach($name in @('d12','d13')){
$e=$old.checkpoint_continuity_evidence[$name];$gate=$e.gate;$p=K-S15Pins
$gp=if($name -ceq 'd13'){$p.technical_refusal}else{(K-Read $p.previous_contract).pinned_checkpoint.pins.technical_refusal}
$known=if($name -ceq 'd13'){$gate.actual_known_compiler_type_refusal}else{$gate.actual_known_layout_compatibility_diagnostic}
$raw=K-Raw $gate.journal 262144;$text=$utf8.GetString($raw);Need ($text.EndsWith([string][char]10) -and -not $text.Contains([string][char]13) -and $raw.Length -le 524288) 'Fixed journal framing';$rows=@();foreach($line in $text.TrimEnd([char]10).Split([char]10)){Need ($utf8.GetByteCount($line)+1 -le 384) 'Fixed row cap';$rows+=,(K-Json $line)}
$rp=if($name -ceq 'd12'){$p.previous_recovery}else{(K-Read $gate.actual_tool_return).request}
& $whole @((K-Read $gate.actual_tool_return),(K-Read $known),(K-Read $gp),(K-Read $rp),$rows) @($e.tool,$e.known,$e.gate,$e.recovery,$e.journal_rows)
}
foreach($spec in @($rv.root_reviews)+@($rv.sol_reports)){K-PinShape $spec ([string]$spec.path);$null=K-Raw $spec 262144}
return ,@($ct.provenance.root_design,$ct.provenance.astra_markdown,$ct.provenance.astra_json,$ct.provenance.root_reflection)
}
function K-SourceFence($_p){return D19-F $_p;
$script:KManagedSourceAccepted=$false;$script:KCatalogManifestContract=$null;try{
K-PinShape $_p (K-Paths)[$zk];$f=K-Read $_p;Same $f[$z1c] $true 'Actual ROOT adopted Source';Same $f[$zn] (K-Epoch) 'K current Source';K-PinShape $f[$zz] (K-Paths).review;$rv=K-Read $f[$zz];K-Keys $rv @($zj,'epoch',$z9,'replacements','additions','before_source','root_reflection','astra_review','root_reviews','sol_reports');Same $rv[$zj] 'flylab.m3.checkpoint-reviewed-source.v1' 'Distinct K review';Same $rv.epoch (K-Epoch) 'K Source review epoch';Same $f.checkpoint_contract $f[$z9][(K-Paths).contract] 'Current Source contract before provenance';Same $f.checkpoint_contract $rv.sources[(K-Paths).contract] 'Review Source contract agreement';$null=K-Provenance $rv;Same $rv.before_source $script:KBeforeSource 'Exact continuation Source archive';$a=K-Read $script:KBeforeSource;$b=$a[$z9];$c=K-Clone $a;$c[$zn]=K-Epoch;$c[$zz]=$f[$zz];$c[$z1f]=$f[$z9][(K-Paths).body];$c[$z9]=$f[$z9];$c.checkpoint_contract=$f[$z9][(K-Paths).contract];Same $f $c 'Whole inherited freeze plus finite K changes';Need ((Conserved-Count $b) -eq 871) 'Whole K736';K-Keys $f[$z9] (Conserved-Keys $b);Same $rv[$z9] $f[$z9] 'Reviewed exact vector';Same $rv.replacements (K-Replacements) 'Only declared replacements';Need ($rv.additions -is [array] -and $rv.additions.Length -eq 0) 'Exact empty additions array, never null';foreach($k in (Conserved-Keys $b)){if($k -cnotin (K-Replacements)){Same $f[$z9][$k] $b[$k] 'Whole unchanged Source pin'}}
foreach($se in $f[$z9].GetEnumerator()){$si=$se.Value;Same $si.path $se.Key 'Same whole Source namespace';Need ([IO.Path]::GetExtension($si.path) -cin @('.py','.ps1','.json','.txt','.ini','.md','.js','.xml')) 'Text Source only';$_current=$si.path -cin @('tools/banc_m3_checkpoint_harness_runtime.py','tools/banc_m3_checkpoint_harness_runtime_native.ps1',($kb+'/parent-direct-science-account.ps1'));$_y=-not $_current -and $b.Contains($si.path) -and $b[$si.path].bytes -gt 262144;$d=if($si.path -cin @('tools/banc_m3_checkpoint_harness_runtime.py','tools/banc_m3_checkpoint_harness_runtime_native.ps1')){[long]131072}elseif($si.path -ceq ($kb+'/parent-direct-science-account.ps1')){[long]393216}elseif($_y){Same $si $b[$si.path] 'Only exact inherited J evidence cap';[long]524288}else{[long]262144};Need ($si.bytes -le $d) 'Source cap';if([IO.Path]::IsPathRooted($si.path)){Need ($b.Contains($si.path)) 'Installed roots only from fixed J vector';Same $si $b[$si.path] 'Fixed inherited installed pin';Need ($si.path -notlike '*.npz' -and [IO.Path]::GetExtension($si.path) -cin @('.py','.ps1','.json','.txt','.ini','.md')) 'Installed text Source only';$e=K-SourceRaw $si.path $d;Need ($e.Length -le $d) 'Bounded installed Source';Same ([long]$e.Length) $si.bytes 'Whole installed bytes';Same ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($e)).ToLowerInvariant()) $si.sha256 'Whole installed SHA'}else{$null=K-Raw $si $d $true}}
$script:KManagedSourceAccepted=$true;K-Same $f $c 'Whole inherited freeze plus finite K changes';K-Same $rv[$z9] $f[$z9] 'Reviewed exact vector';Same $f.checkpoint_contract $f[$z9][(K-Paths).contract] 'Accepted current code contract';$script:KCatalogManifestContract=K-Clone $f.checkpoint_contract;$null=K-Roster $f;Same $f[$z1f] $f[$z9][(K-Paths).body] 'Static acyclic body';return $f
}catch{$script:KManagedSourceAccepted=$false;$script:KCatalogManifestContract=$null;$script:KPC.Clear();$script:KPJ=$null;throw}
}

function K-Dispatch($l,$q,$_r,[string]$a){$bn=(K-Read $script:KCurrentContract).baseline_debits;Need ($bn -is [long] -and $bn -gt 0) 'Exact baseline';
$s=K-Validate $l;Need ($a -cin @('closed','admitted')) 'Finite dispatch phase';if($a -ceq 'closed'){K-Eligible $l $s;Same $q[$ze] (K-Calls)[$s.completed] 'Next literal call'}else{Same $s.phase 'admitted' 'Exact pre-worker admission';Same $l[$z1][$q[$ze]][$z2] $_r 'Same admitted outer'};$null=K-Identity $q $_r $s[$zk]
if($q.mode -ceq 'generated'){$_h=@(9755,5755,5715,5595,5425)[[array]::IndexOf((K-Calls),$q[$ze])];Need ($l[$za]+$_h -le $s.caps[$z1i] -and $l[$zf] -ge $_h) 'Whole paired Generated envelope'}else{Need ($l.Contains((K-QualificationKey))) 'Full K qualification before Original';$oe=if($q[$ze] -ceq (K-Calls)[5]){[long]600}else{$q[$z6].outer_s+180};foreach($_a in @(@('original_cumulative_used_s',[long]12870),@('original_stage_used_s',[long]6300),@($zr,[long]6000))){Need ($l[$_a[0]]+$oe -le $_a[1]) 'Whole body/audit/close600 or audit plus close180'}};for($i=1;$i -lt [Math]::Min($s.completed,5);$i++){$b=@('lint','math','gate_a','gate_b')[$i-1];$c=if($b -ceq 'lint'){$null}else{K-BoundedPin ($kb+'/parent-contact-learning-s28-r64-'+$b.Replace('_','-')+'-junit.xml')};$null=K-Success $l[$zg][$bn+$i] $s[$zk] $b $c};return $s
}
function K-BodyBinding($q,$ct){
$a=@((K-Calls)[5],(K-Calls)[6]);$i=[array]::IndexOf($a,$q[$ze]);Need ($i -ge 0 -and $q.role -ceq @('contact_candidate_body','contact_body_audit')[$i]) 'Only K body/audit';Same $q[$z0].path (K-Paths)[$zk] 'K science Source';$f=K-Read $q[$z0];Same $q[$z1f] $f[$z9][(K-Paths).body] 'Frozen K body';$v=K-Read $q[$z1f];$bp=(K-Read (K-S15Pins).before)[$z9][(K-Paths).body];Same $bp @{path=(K-Paths).body;bytes=[long]20516;sha256='ac3786738fad5fe69e4934c0382c1a4508444402d387e1f113017f7eaec92276'} 'Executed beforeB5';$bp=K-Clone $bp;$bp.path=$kb+'/s28-before/'+$bp.path;$e=K-Clone (K-Read $bp)
$e.calls=$a;$e[$zn]=K-Epoch;$e.source_review_path=(K-Paths).review;$e.source_freeze_path=(K-Paths)[$zk];$e.checkpoint_contract_path=(K-Paths).contract;$e.checkpoint_recovery_path=(K-Paths).recovery;$e.checkpoint_allocation_path=(K-Paths).allocation;Same $v $e 'Whole beforeB5 plus seven routing changes';foreach($k in @('contract','grant','inputs','roster')){Same $q[$k] $v[$k] 'Exact candidate metadata'};Same $q[$z6] $ct[$z6][$q.role] 'Unchanged science limits';return $v
}
function K-Candidate($q,$a,$r){$b=K-Read $r.historical_manifests[1];$c=K-Read $b[$zl].report;Same $c.candidate $r.candidate 'Whole original audited candidate';Same $a.candidate $r.candidate 'Same candidate k';Same $b[$zl].candidate_arrays $r.candidate_arrays 'Literal Original arrays metadata';$d=K-Read $q[$zl].manifest;Same $d[$z0] $(if($q.role -ceq 'contact_candidate_body'){$r.source_k}else{$q[$z0]}) 'Immediate exact provenance';Same $d.role $(if($q.role -ceq 'contact_candidate_body'){'contact_candidate_audit'}else{'contact_candidate_body'}) 'Immediate role';Same $d.run_valid $true 'Valid predecessor';$e=K-Clone $d[$zl];$e.manifest=$q[$zl].manifest;Same $q[$zl] $e 'Whole artifact pins'}
function K-ValidBody($d,$_f,$q){$f=K-FeeProof $d.descriptor $d $_f;Need ($f.closed -and $f.cost[$z4] -eq 0 -and -not $f.cost.cap_exceeded -and $f.q[$ze] -ceq (K-Calls)[5]) 'Audit conditional actual body';$m=K-Read $q[$zl].manifest;Same $m[$z2] $f.cost[$z2] 'Actual body manifest';Same $m.run_valid $true 'Actual run valid';$a=K-Read $m[$zl].report;Need ($a.run_valid -ceq $true -and $null -eq $a[$zi] -and $a.secondary_errors.Length -eq 0) 'Body report actually valid';$r=K-BodyBinding $q (K-Read $q.contract);K-Candidate $q $a $r}
function K-OriginalMetadata($_s,$q){
$a=K-Read (K-Pins).jbody;foreach($b in @('contract','grant','inputs','roster')){Same $_s[$b] $a[$b] 'Fixed immutable science pin before reading'};K-PinShape $_s[$z0] (K-Paths)[$zk];K-PinShape $_s[$z1f] (K-Paths).body;K-PinShape $_s[$zs] (K-Paths).qualified;$ct=K-Read $_s.contract;Same $ct[$zj] 'flylab.m3.contact-learning.stage.v1' 'Fixed science contract schema';Same $ct.roles @('contact_direction','contact_candidate_audit','contact_candidate_body','contact_body_audit') 'Whole fixed science roles';$c=K-BodyBinding $_s $ct
foreach($_a in @(@($zj,'flylab.m3.contact-learning.request.v1'),@('mode','original'),@('ledger_path',$ledgerName),@('shared_original_lock_path',($kb+'/parent-e358-original-active.lock')),@('local_debit_offset',[long]165),@('resource_increment_s',[long]0),@('output',($kb+'/parent-'+$_s[$ze]+'/result')))){Same $_s[$_a[0]] $_a[1] 'Finite K science field'};Same $q.mode 'original' 'Same Original outer';Same $q.output ($_s.output+'-supervision') 'Whole science/outer output relation';foreach($b in @($ze,'role','grant',$z0,'ledger_path','shared_original_lock_path')){Same $q[$b] $_s[$b] 'Same K Original metadata'};foreach($b in $q[$z6].Keys){Same $_s[$z6][$b] $q[$z6][$b] 'Whole exact outer science cap'};Same $_s[$z6] $ct[$z6][$_s.role] 'Whole science limits including internal'
$_o=K-Read $_s.roster;$d=K-Read $_s.grant;Same $_s.grant (K-Pins).grant 'Fixed existing grant';foreach($_a in @(@($zj,'flylab.m3.startup-ray.original-grant.v1'),@('grant_id','startup-ray-original-20261007-j'),@('reset',$false),@('resource_increment_s',[long]0),@('local_debit_offset',[long]165),@('worker_max',[long]1),@('automatic_restarts',[long]0))){Same $d[$_a[0]] $_a[1] 'Existing finite grant'};Same $_o[$zj] 'flylab.m3.contact-conditioned-learning.input-roster.v1' 'Fixed input roster schema';Same $ct.roster $_s.roster 'Fixed contract roster';Same $_o.grant $_s.grant 'Same immutable roster grant';Same $_o.original_inputs $_s.inputs 'Same immutable input binder';Same $d.inputs $_s.inputs 'Same grant input lineage';$e=K-Read $_s.inputs;Need ($e -is [Collections.IDictionary]) 'Intact metadata binder, no payload reads';$f=K-Read $_o.previous_geometry_roster;Same $_s[$z11] $f[$z11] 'Fixed historical manifest lineage';$null=K-Read $_s[$z11]
$_z=K-Read $_s[$z0];Same $_z[$zj] 'flylab.m3.contact-learning.source-freeze.v1' 'Same K freeze schema';Same $_z[$z1c] $true 'Reviewed immutable K freeze';Same $_z[$zn] (K-Epoch) 'Finite K epoch';Same $_z[$z9][$_s.contract.path] $_s.contract 'Same frozen contract';Same $_z.human_authorization $d.human_authorization 'Unchanged human scope';$_n=K-Read $_s[$zs];Same $_n[$zj] 'flylab.m3.direct-science.qualification.v8' 'Same K qualification schema';Same $_n.epoch (K-Epoch) 'Same K qualification epoch';Same $_n[$z0] $_s[$z0] 'Same admitted qualified freeze';$_b=K-Read $_s[$zm] 524288;Same $_b[(K-QualificationKey)] $_s[$zs] 'Before carries actual K qualification';$g=K-Read $_b[(K-Key)];Same $g[$z0] $_s[$z0] 'Same recovered Source lineage';Need ($_b.Contains((K-AllocationKey))) 'Existing conserved allocation before Original';return $c
}
function K-OriginalRequest($_s,$q,$a){
$s=K-Validate $a;$r=K-OriginalMetadata $_s $q;$null=K-SourceFence $_s[$z0];Same $_s[$zs] $a[(K-QualificationKey)] 'Actual K registered qualification';K-Qualification (K-Read $_s[$zs]) $a $_s[$z0];$_b=K-Read $_s[$zm] 524288;Same $a $_b 'Fresh exact Original before';foreach($k in @($ze,'role','grant',$z0)){Same $q[$k] $_s[$k] 'Same Original owner'};Same $_s.local_debit_offset ([long]165) 'Original offset retained';Same $_s.resource_increment_s ([long]0) 'No reset';if($_s.role -ceq 'contact_body_audit'){K-ValidBody $a[$zg][-1] $_s[$z0] $_s}else{$b=K-Read $r.historical_manifests[1];K-Candidate $_s (K-Read $b[$zl].report) $r};return K-Read (K-Pins).grant
}
function K-BoundedPin([string]$_q,[long]$a=262144){
$lp=K-LedgerPaths;if($_q -ceq ($kb+'/parent-e358-original-active.lock')){$a=[long]16384}elseif($_q -cin $lp){$a=[long]524288}else{Need ($a -le 262144) 'New L metadata descriptor cap'};$b=K-SourceRaw (Safe $_q) $a;return @{path=$_q;bytes=[long]$b.Length;sha256=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($b)).ToLowerInvariant()}
}
function K-Commit($l,[string]$_l){return D19-C $l $_l;
Assert-LedgerOwner $script:ledgerTransaction;Need ($script:ledgerTransaction.uncertain -ceq $false) 'Certain common fee owner';
$feeSemanticOnly={param($a,$b)
Same $a $b 'Whole fee typed membership/arrays';if($a -is [double]){Need ($b -is [double] -and [BitConverter]::DoubleToInt64Bits($a) -eq [BitConverter]::DoubleToInt64Bits($b)) 'Whole fee binary64'};if($a -is [Collections.IDictionary]){foreach($k in $a.Keys){& $feeSemanticOnly $a[$k] $b[$k]}}elseif($a -is [array]){for($i=0;$i -lt $a.Length;$i++){& $feeSemanticOnly $a[$i] $b[$i]}}
}
$whole={param($a,$b)
& $feeSemanticOnly $a $b;if($a -is [Collections.IDictionary]){Same @($a.Keys) @($b.Keys) 'Protected fee dictionary order';foreach($k in $a.Keys){& $whole $a[$k] $b[$k]}}elseif($a -is [array]){for($i=0;$i -lt $a.Length;$i++){& $whole $a[$i] $b[$i]}}
}
& $whole $l (K-Json $script:ledgerTransaction.before);
$ct=K-Read $script:KCurrentContract;$a9=$ct.actual320
$bind=$ct.pinned_checkpoint;foreach($pin in @($bind.coverage,$bind.latest_failure,$bind.latest_cost,$bind.policy,$bind.grant)+@($bind.provenance.Values)){$null=K-Raw $pin 262144}
foreach($key in 'terminal admitted pre_backup replace_backup cost receipt actor audit tool failure failed_tool request'.Split(' ')){$null=K-Raw $ct.terminal_bridge[$key] (K-ReaderCap $ct.terminal_bridge[$key].path)};foreach($pin in $ct.terminal_bridge.archives){$null=K-Raw $pin.immutable_archive 262144}
foreach($pin in $a9.raw_pins){$null=K-Raw $pin (K-ReaderCap $pin.path)}
$oc=$ct;foreach($i in 1,2){$oc=& $script:KPredecessor $oc.actual320 $whole;foreach($pin in $oc.actual320.raw_pins){$null=K-Raw $pin (K-ReaderCap $pin.path)}}
$a0=K-Read $a9.terminal 524288;Need ($a0.debits.Length -eq 322 -and (Conserved-Count $a0.direct_science_calls) -eq 204) 'Exact paid322/204 baseline'
$a8=$l[(K-Key)];K-PinShape $a8 (K-Paths).recovery;$event=K-Read $a8;$sf=$event.source_freeze;K-PinShape $sf (K-Paths).source;$a2=K-Read $sf;$rv=K-Read $a2.reviewed_source
$null=K-Provenance $rv;Same $rv.before_source $script:KBeforeSource 'Fee historical SourceFence authority';$b1=K-Read $script:KBeforeSource;$a1=K-Clone $b1;$a1.qualification_epoch=K-Epoch;$a1.reviewed_source=$a2.reviewed_source;$a1.body_retry=$a2.sources[(K-Paths).body];$a1.sources=$a2.sources;$a1.checkpoint_contract=$a2.sources[(K-Paths).contract];& $feeSemanticOnly $a2 $a1
K-Keys $a2.sources (Conserved-Keys $b1.sources);& $feeSemanticOnly $a2.sources $rv.sources;Same $rv.replacements (K-Replacements) 'All ordered11';Same $rv.additions @() 'No Source additions';foreach($key in $b1.sources.Keys){if($key -cnotin (K-Replacements)){& $feeSemanticOnly $a2.sources[$key] $b1.sources[$key]}}
foreach($key in (K-Replacements)){$pin=$a2.sources[$key];K-Keys $pin @('path','bytes','sha256');Same $pin.path $key 'Fee Source name';$cap=if($key -cin (K-Replacements)[0..1]){131072}elseif($key -ceq ($kb+'/parent-direct-science-account.ps1')){393216}else{262144};Need ((K-Int $pin.bytes) -ge 0 -and $pin.bytes -le $cap -and $pin.sha256 -is [string] -and $pin.sha256 -cmatch '^[0-9a-f]{64}$') 'Fee Source metadata'}
$null=K-Roster $a2;
K-Keys $event $ct.terminal_bridge.event_keys;$a3=@{};foreach($pin in $a9.archives){$a3[$pin.logical_current_before.path]=$pin.logical_current_before}
$want=[ordered]@{schema=$ct.terminal_bridge.event_schema;epoch=(K-Epoch);before_ledger=$a9.terminal;source_freeze=$sf;review=$a2.reviewed_source;settlement=$a9.commit_capture;cost=$a9.old_cost;audit=$a9.failure;previous_recovery=$a9.old_continuation;old_source=$a3[(K-Paths).source];old_review=$a3[(K-Paths).review];old_contract=$a3[(K-Paths).contract];root_reviews=$rv.root_reviews;sol_reports=$rv.sol_reports;old_caps=(K-OldCaps);new_caps=(K-NewCaps);new_resource_grant_s=[long]0;refund_s=[long]0;reset=$false};Same $event $want 'Exact independently authenticated event fields';& $whole $event.root_reviews $rv.root_reviews;& $whole $event.sol_reports $rv.sol_reports
$oldCall='direct-science-20261009-contact-s28-bootstrap-r3';Same $a0.direct_science_blocking_holds @{$oldCall='L stage unsuccessful; successor blocked'} 'Only old failed r3 HOLD';$a0[(K-Key)]=$a8;$a0.direct_science_blocking_holds.Remove($oldCall)
$h=K-Anchor;$h[$z8]=$a0;$h.event=$event
$a5=$null
$fee={param($fp,$a,$_f)
K-PinShape $fp ($kb+'/parent-'+$a.cost_id+'-full-return.json');$b3=if($null -ne $a5 -and $fp.path -ceq $a5.pin.path){$a5.pin}else{K-BoundedPin ($kb+'/parent-'+$a.cost_id+'-full-return.json')};Same $fp $b3 'Exact computed fee tuple';$b3=[ordered]@{path=$b3.path;bytes=$b3.bytes;sha256=$b3.sha256};$_d=if($null -ne $a5 -and $fp.path -ceq $a5.pin.path){Same $fp $a5.pin 'Exact provisional fee tuple';K-Json $a5.raw}else{K-Read $fp};$_r=if($a.mode -ceq 'generated'){$_d[$z2]}else{$_d[$z17]};$q=K-Read $_r;$null=K-Identity $q $_r $_f;$t=K-Read $_d.trace 262144;$o=K-Read $_d.outer 262144
Same $t[$z2] $_r 'Genuine request';Same $t[$z0] $_f 'Genuine source binding';Same $o[$z2] $_r 'Immutable outer binding';Same $t[$zj] 'flylab.m3.direct-science.genuine-return.v1' 'Actual collector';Same $t.phase 'FINAL' 'First genuine FINAL';Same $t[$zi] $null 'No collector uncertainty';$null=K-Int $t[$z1j] $true;$null=K-Int $t[$z15] $true;[double]$b=($t[$z15]-$t[$z1j])/1000.0;Need ((K-Number $b) -gt 0) 'Positive genuine fee';Same $t.full_return_s $b 'Genuine full return'
foreach($_a in @(@('cost_id',$q[$ze]),@('mode',$q.mode),@($z0,$_f),@($z1b,$b),@($z4,$t[$z4]),@('charged_once',$true),@('cap_exceeded',($b -gt $q[$z6].outer_s)))){Same $_d[$_a[0]] $_a[1] 'Exact incurred once cost'}
Same $a @{cost_id=$q[$ze];mode=$q.mode;$z1b=$b;actual_returned_exit_code=$t[$z4];descriptor=$fp} 'Whole exact paid debit';Need (K-ExitMatch $t $o $_r) 'Same actual exit or existing proven shell deadline pair';$_k=@($zj,'cost_id','mode',$z2,$z0,'trace','outer',$z1b,$z4,'charged_once',$zb,$zd,'cap_exceeded',$z14,'creation_attempted',$z7,$zt,$zo);if($q.mode -ceq 'original'){$_k+=@($z17,'role','grant','inputs','roster','contract',$z1f,'local_debit_offset','resource_increment_s')};if($o[$z4] -ne $t[$z4]){$_k+=@($z18,$zu,$zp)};K-Keys $_d $_k;Same $_d[$zj] $(if($q.mode -ceq 'generated'){'flylab.m3.direct-science.full-return.v1'}else{'flylab.m3.contact-learning.full-return.v1'}) 'Exact mode cost schema';Same $o[$zj] 'flylab.m3.direct-science.outer.v1' 'Actual supervisor outer schema';K-FeeValues $_d $q $_r $t $o;foreach($c in @($zb,$zd,$z14,'creation_attempted',$z7,$zt,$zo)){Same $_d[$c] $o[$c] 'Exact cost closure and creation'}return @{q=$q;cost=$_d;outer=$o;closed=(K-Closure $o);$z2=$_r;pin=$b3}
}
$a4={param($l)
$e=K-Clone $h[$z8];$e[(K-Key)]=$a8;$e[$z5]=@{};$_f=$h.event[$z0];$b=@($l[$zg]|Select-Object -Skip 322);Need ($b.Length -le 7) 'Finite seven-call tail';$c=K-OldCaps;$g=$false;$j=K-Calls
for($i=0;$i -lt $b.Length;$i++){$d=$b[$i];Same $d.cost_id $j[$i] 'Next literal debit';Need (-not $g) 'Failure never success prerequisite';$f=& $fee $d.descriptor $d $_f;$q=$f.q;Need ($f.closed) 'UNCOMMITTED_OWNERSHIP_UNKNOWN without fee/debit/totals';$paid=[ordered]@{cost_id=$q[$ze];mode=$q.mode;$z1b=$f.cost[$z1b];actual_returned_exit_code=$f.cost[$z4];descriptor=$f.pin};Same $d $paid 'Independent whole paid row';$e[$zg]+=,$paid;$e[$z1][$q[$ze]]=@{$z2=$f[$z2];state='charged';mode=$q.mode;reserved_s=$q[$z6].outer_s};$g=$f.cost[$z4] -ne 0 -or $f.cost.cap_exceeded -or -not $f.closed;if(-not $f.closed){$null=K-Ownership $e $q $f[$z2] $l $true;$m=$(if($q.mode -ceq 'generated'){'direct_science_'}else{'startup_ray_original_'})+'workers';foreach($n in @('owner_pid','owner_creation_ticks',$zx,'worker_creation_ticks')){Same $f.outer[$n] $e[$m][$q[$ze]][$n] 'Unclosed evidence retains admitted identity'}};if($f.closed -and $f.cost[$z4] -ne 0){$_x=$(if($q.mode -ceq 'generated'){'direct_science_'}else{'startup_ray_original_'})+'blocking_holds';$e[$_x][$q[$ze]]='L stage unsuccessful; successor blocked'};if($f.cost.cap_exceeded){$hk=$(if($q.mode -ceq 'generated'){'direct_science_'}else{'startup_ray_original_'})+'blocking_holds';K-AccountingHold $e $q.mode $q[$ze]}}
if($l.Contains((K-AllocationKey))){Need ($b.Length -ge 1) 'Allocation follows actual bootstrap';$sp=$l[(K-AllocationKey)];K-PinShape $sp (K-Paths).allocation;$v=K-Read $sp;K-Keys $v @($zj,'epoch',$zm,'recovery',$z0,'bootstrap','bootstrap_junit','old_caps','new_caps','delta_s','combined_global_s','new_resource_grant_s','refund_s');foreach($_a in @(@($zj,'flylab.m3.checkpoint-allocation.v1'),@('epoch',(K-Epoch)),@('recovery',$l[(K-Key)]),@($z0,$_f),@('old_caps',(K-OldCaps)),@('new_caps',(K-NewCaps)),@('delta_s',@{$z1g=[long]0;$z1i=[long]0;$z1h=[long]0}),@('combined_global_s',[long]19900),@('new_resource_grant_s',[long]0),@('refund_s',[long]0),@('bootstrap',$b[0].descriptor))){Same $v[$_a[0]] $_a[1] 'Exact allocation field'}
$_b=K-Read $v[$zm] 524288;Need (-not $_b.Contains((K-AllocationKey)) -and $_b[$zg].Length -eq 323) 'Fresh closed bootstrap before allocation';$bs=& $a4 $_b;Need ($bs.phase -ceq 'charged' -and -not $bs.failed) 'Successful known charged prerequisite';foreach($key in (Conserved-OwnerKeys)){Same $_b[$key] @{} 'Closed eligible owner'};Need ($_b[$zf] -ge 0 -and $_b[$zy] -ge 0 -and $_b[$za] -le $bs.caps[$z1i]) 'Execution remaining resource';$null=K-Success $_b[$zg][-1] $_f 'bootstrap' $v.bootstrap_junit;$e[(K-AllocationKey)]=$sp;$c=K-NewCaps;$e[$zq]=$c[$z1g];$e[$zv]=$c[$z1i];$e[$z19]=$c[$z1h]
}else{Need ($b.Length -le 1) 'No qualification from old caps'}
if($l.Contains((K-QualificationKey))){Need ($l.Contains((K-AllocationKey)) -and $b.Length -ge 5) 'Whole four costs first';$sp=$l[(K-QualificationKey)];K-PinShape $sp (K-Paths).qualified;K-Qualification (K-Read $sp) $l $_f;$e[(K-QualificationKey)]=$sp};Need ($b.Length -le 5 -or $l.Contains((K-QualificationKey))) 'Original requires K qualification'
$o=K-Totals $e $h.policy $h.grant $c;foreach($k in $o.Keys){$e[$k]=$o[$k]};$p=@((Conserved-Keys $l[$z1])|Where-Object {-not $e[$z1].Contains($_)});Need ($p.Length -le 1) 'Sole next admitted call';$r='charged'
if($p.Length -eq 1){Need (-not $g -and $b.Length -lt 7) 'No successor after failure/end';$_c=$j[$b.Length];Same $p @($_c) 'No old/replay/alias admission';$a=$l[$z1][$_c];K-Keys $a @($z2,'state','mode','reserved_s');$b2=if($b.Length -eq 0){$ct.actual320.current_request_path}else{$a[$z2].path};K-PinShape $a[$z2] $b2;$_r=K-BoundedPin $b2;$q=K-Read $_r;$null=K-Identity $q $_r $_f;Same $a @{$z2=$_r;state='admitted';mode=$q.mode;reserved_s=$q[$z6].outer_s} 'Exact whole admission';Need ($b.Length -ne 1 -or $l.Contains((K-AllocationKey))) 'No lint before allocation';Need ($b.Length -ne 5 -or $l.Contains((K-QualificationKey))) 'No body before full qualification';if($b.Length -eq 6){K-ValidBody $b[5] $_f (K-Read $q[$zc])};$e[$z1][$_c]=@{$z2=$_r;state='admitted';mode=$q.mode;reserved_s=$q[$z6].outer_s};$r=K-Ownership $e $q $_r $l}
if($b.Length -gt 0){$d=$b[-1];$s=($d.mode -ceq 'generated' -and ($e[$za] -gt $c[$z1i] -or $e[$zf] -lt 0)) -or ($d.mode -ceq 'original' -and ($e[$zy] -lt 0 -or $e.original_stage_used_s -gt 6300 -or $e[$zr] -gt 6000));if($s){$hk=$(if($d.mode -ceq 'generated'){'direct_science_'}else{'startup_ray_original_'})+'blocking_holds';K-AccountingHold $e $d.mode $d.cost_id}}
& $feeSemanticOnly $l $e;return @{caps=$c;completed=$b.Length;phase=$r;failed=$g;$zk=$_f;epoch=(K-Epoch)}
}
$a=& $a4 $l;$tp=K-BoundedPin $_l;$t=K-Read $tp 262144;Same $t[$zj] 'flylab.m3.direct-science.genuine-return.v1' 'Actual collector';Same $t.phase 'FINAL' 'First FINAL';Same $t[$zi] $null 'No uncertain collector';$null=K-Int $t[$z1j] $true;$null=K-Int $t[$z15] $true;[double]$_t=($t[$z15]-$t[$z1j])/1000.0;Need ((K-Number $_t) -gt 0) 'Finite genuine fee';Same $t.full_return_s $_t 'Exact genuine elapsed';$q=K-Read $t[$z2];$_c=$q[$ze];$null=K-Identity $q $t[$z2] $a[$zk]
Need ($l[$z1].Contains($_c) -and $l[$z1][$_c].state -ceq 'admitted' -and $_c -cnotin @($l[$zg].cost_id)) 'Actual once call';Same $l[$z1][$_c][$z2] $t[$z2] 'Same admitted owner';Same $t[$z0] $a[$zk] 'Intact execution Source binding';$op=K-BoundedPin ($q.output+'/outer.json');$o=K-Read $op 262144;Same $o[$z2] $t[$z2] 'Actual supervisor outer';Need (K-ExitMatch $t $o $t[$z2]) 'Same observed exit or proven shell pair';$b=K-Closure $o;if(-not $b){return @{committed=$false;status='UNCOMMITTED_OWNERSHIP_UNKNOWN';trace=$tp;outer=$op;actual_s=$_t;cost_published=$false;owner_and_reservation_retained=$true}}
$_d=[ordered]@{$zj='flylab.m3.direct-science.full-return.v1';cost_id=$_c;mode=$q.mode;$z2=$t[$z2];$z0=$t[$z0];trace=$tp;outer=$op;$z1b=$_t;$z4=$t[$z4];charged_once=$true;$zb=$o[$zb];$zd=$o[$zd];cap_exceeded=($_t -gt $q[$z6].outer_s);$z14=$o[$z14];$z7=$o[$z7];creation_attempted=$o.creation_attempted;$zt=$o[$zt];$zo=$o[$zo]}
if($o[$z4] -ne $t[$z4]){$_d[$z18]=$o[$z4];$_d[$zu]=$t[$z4];$_d[$zp]=$true};if($q.mode -ceq 'original'){$_s=K-Read $q[$zc];$_d[$zj]='flylab.m3.contact-learning.full-return.v1';$_d[$z2]=$q[$zc];$_d[$z17]=$t[$z2];foreach($k in @('role','grant','inputs','roster','contract',$z1f,'local_debit_offset','resource_increment_s')){$_d[$k]=$_s[$k]}}
$fn=$kb+'/parent-'+$_c+'-full-return.json';$b4=[IO.File]::Exists((Safe $fn));$raw=$utf8.GetBytes(($_d|ConvertTo-Json -Depth 100)+[char]10);$fp=[ordered]@{path=$fn;bytes=[long]$raw.Length;sha256=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($raw)).ToLowerInvariant()};if($b4){$fp=K-BoundedPin $fn;$raw=K-Raw $fp 262144;& $feeSemanticOnly (K-Json $raw) $_d};$a5=@{pin=$fp;raw=$raw}
$d=[ordered]@{cost_id=$_c;mode=$q.mode;$z1b=$_t;actual_returned_exit_code=$t[$z4];descriptor=$fp};$null=& $fee $fp $d $a[$zk];$l=K-Clone $l;$l[$zg]+=@($d);$l[$z1][$_c].state='charged';$_i=if($q.mode -ceq 'generated'){'direct_science_'}else{'startup_ray_original_'}
if($b){$l[$_i+'workers'].Remove($_c);$l[$_i+'reservations'].Remove($_c);$l[$_i+'pending_actuals'].Remove($_c);if($q.mode -ceq 'original'){$l[$zw]=[long]0;$l[$z10]=[long]0}}
$c=K-Totals $l $h.policy $h.grant $a.caps;foreach($k in $c.Keys){$l[$k]=$c[$k]};$e=$_d.cap_exceeded -or ($q.mode -ceq 'generated' -and ($l[$za] -gt $a.caps[$z1i] -or $l[$zf] -lt 0)) -or ($q.mode -ceq 'original' -and ($l[$zy] -lt 0 -or $l.original_stage_used_s -gt 6300 -or $l[$zr] -gt 6000))
if($b -and $t[$z4] -ne 0){$l[$_i+'blocking_holds'][$_c]='L stage unsuccessful; successor blocked'};if($e){K-AccountingHold $l $q.mode $_c};$null=& $a4 $l;$feePayload=$utf8.GetBytes(($l|ConvertTo-Json -Depth 100)+[char]10);Need ($feePayload.Length -le 524288) 'Bounded full paid ledger before either publication';$null=Conserved-Json $feePayload;Assert-LedgerOwner $script:ledgerTransaction;$feeBefore={Same ([Convert]::ToBase64String((K-Raw (K-BoundedPin $ledgerName 524288) 524288))) ([Convert]::ToBase64String($script:ledgerTransaction.before)) 'Fresh exact fee raw-before'}; & $feeBefore;$script:ledgerTransaction.uncertain=$true;try{if(-not $b4){Same (Write-New $fn $_d) $fp 'Exact new cost tuple'};Same ([Convert]::ToBase64String((K-Raw $fp 262144))) ([Convert]::ToBase64String($raw)) 'Exact paid cost raw readback';Assert-LedgerOwner $script:ledgerTransaction;& $feeBefore;$script:ledgerTransaction.uncertain=$false;$f=Save-Ledger $l}catch{if($null -ne $script:ledgerTransaction){$script:ledgerTransaction.uncertain=$true};throw};return @{committed=$true;cost=$fp;actual_s=$_t;$z4=$t[$z4];ledger=$f}
}
function K-RecoveryAction($l,$a,[byte[]]$_w){return D19-A $l $a $_w;
Assert-LedgerOwner $script:ledgerTransaction
Same ([Convert]::ToBase64String($_w)) ([Convert]::ToBase64String($script:ledgerTransaction.before)) 'Real common-owner before'
$b=(K-Read $script:KCurrentContract).actual320;Same ([Convert]::ToBase64String($_w)) ([Convert]::ToBase64String((K-Raw $b.terminal 524288))) 'Exact terminal raw before Source acceptance'
$current=K-Json $_w;$whole={param($a,$b)
Same $a $b 'Whole types/arrays';if($a -is [double]){Need ($b -is [double] -and [BitConverter]::DoubleToInt64Bits($a) -eq [BitConverter]::DoubleToInt64Bits($b)) 'Whole binary64 bits'};if($a -is [Collections.IDictionary]){Same @($a.Keys) @($b.Keys) 'Whole dictionary order';foreach($k in $a.Keys){& $whole $a[$k] $b[$k]}}elseif($a -is [array]){for($i=0;$i -lt $a.Length;$i++){& $whole $a[$i] $b[$i]}}
}
& $whole $l $current
$h=K-Recovery $a;$null=K-SourceFence $h.event.source_freeze;$result=K-RecoveredSnapshot $a
$reverse=K-Clone $result;$reverse[(K-Key)]=K-Clone $l[(K-Key)];$call=$b.old_failed_call;$reverse.direct_science_blocking_holds[$call]=$b.old_failed_hold
& $whole $reverse $l
return $result
}
function K-AllocationAction($l,$a,[byte[]]$_w){
Assert-LedgerOwner $script:ledgerTransaction;$s=K-Validate $l;K-Eligible $l $s;Need ($s.completed -eq 1 -and -not $l.Contains((K-AllocationKey))) 'Only successful closed bootstrap allocation';K-PinShape $a (K-Paths).allocation;$v=K-Read $a;$a=[ordered]@{path=$a.path;bytes=$a.bytes;sha256=$a.sha256};$b=K-Raw $v[$zm] 524288;Same ([Convert]::ToBase64String($_w)) ([Convert]::ToBase64String($b)) 'Fresh exact raw-before allocation';$null=K-SourceFence $s[$zk];$e=K-Clone $l;$e[(K-AllocationKey)]=$a;$c=K-NewCaps;$e[$zq]=$c[$z1g];$e[$zv]=$c[$z1i];$e[$z19]=$c[$z1h];$h=K-Anchor;$d=K-Totals $e $h.policy $h.grant $c;foreach($k in $d.Keys){$e[$k]=$d[$k]};$null=K-Validate $e;return $e
}
function K-CatalogJournal([string]$leaf,[string]$boundary,[object[]]$counters=@()) {
Need ($leaf -cin @('catalog-support','catalog-init','catalog-backend','catalog-schema','catalog-IO','companion-parse','companion-compare') -and $boundary -cin @('BEGIN','END')) 'Finite S28 leaf journal site'
Need ($taskRoot -is [string] -and $PSCommandPath -is [string] -and [IO.Path]::IsPathRooted($PSCommandPath)) 'Actual rooted journal command context'
$g7=[IO.Path]::GetFullPath($PSCommandPath);$i9=[IO.Path]::GetFullPath($taskRoot)
$g8=Safe ($kb+'/parent-direct-science-account.ps1')
if([string]::Equals($g7,$g8,[StringComparison]::OrdinalIgnoreCase)){
$h6=$script:ledgerTransaction;Assert-LedgerOwner $h6
K-Keys $h6 @('raw','local_pid','creation','active','uncertain','before')
Need ($h6.raw -is [byte[]] -and $h6.raw.Length -gt 0 -and $h6.raw.Length -le 16384 -and $h6.local_pid -is [long] -and $h6.creation -is [long] -and $h6.active -is [bool] -and $h6.active -ceq $true -and $h6.uncertain -is [bool] -and $h6.uncertain -ceq $false -and $h6.before -is [byte[]] -and $h6.before.Length -le 524288) 'Bounded verified current metadata capability'
$h5=K-Json $h6.raw;K-Keys $h5 @('schema','action','owner_pid','owner_creation_ticks','invocation_id')
Same $h5.schema 'flylab.m3.direct-science.metadata-writer.v1' 'Exact existing metadata token schema'
Need ($Action -is [string] -and $Action -cin @('checkpoint-recovery','checkpoint-allocation','admit','commit','qualification') -and $h5.action -is [string] -and $h5.action -ceq $Action -and $h5.action.Length -le 21) 'Exact bounded current metadata action'
Need ($h5.owner_pid -is [long] -and $h5.owner_pid -gt 0 -and $h5.owner_creation_ticks -is [long] -and $h5.owner_creation_ticks -gt 0) 'Exact existing metadata identity types'
Same $h5.owner_pid $h6.local_pid 'Verified local metadata PID';Same $h5.owner_creation_ticks $h6.creation 'Verified local metadata creation'
Need ($h5.invocation_id -is [string] -and $h5.invocation_id.Length -eq 32 -and $h5.invocation_id -cmatch '^[0-9a-f]{32}$') 'Existing typed bounded32hex invocation id'
$h2='canonical';$h3=$h5.invocation_id;$path=Safe ($kb+'/s28-catalog-leaves-'+$h3+'.jsonl')
}else{
$i8=[IO.FileInfo]::new($g7);$g9=$i8.Directory
Need ($i8.Name -ceq 'generated-native.ps1' -and $i8.Exists -and -not ($i8.Attributes -band [IO.FileAttributes]::Directory) -and $g9.Name -cmatch '^generated-native-observation-[0-9]{6}$' -and $g9.Name -cne 'generated-native-observation-000000' -and [string]::Equals($g9.Parent.FullName,$i9,[StringComparison]::OrdinalIgnoreCase)) 'Exact existing Generated observation command'
$h0=Safe ($g9.Name+'/generated-native.ps1');Need ([string]::Equals($g7,$h0,[StringComparison]::OrdinalIgnoreCase)) 'Contained actual Generated wrapper path'
$h1=[Environment]::GetEnvironmentVariable('FLYLAB_DIRECT_SCIENCE_OWNER')
Need ($h1 -is [string] -and $h1.Length -gt 0 -and $utf8.GetByteCount($h1) -le 16384) 'Existing bounded real parent ENTRY environment'
$h4=K-Json $h1;K-Keys $h4 @('schema','call_id','request','owner_pid','owner_creation_ticks','entry_counter_s','job_name')
Same $h4.schema 'flylab.m3.direct-science.lock-owner.v1' 'Actual inherited supervisor parent schema'
Need ($h4.call_id -is [string] -and $h4.call_id -cin @((K-Calls)[0],(K-Calls)[3],(K-Calls)[4]) -and $h4.owner_pid -is [long] -and $h4.owner_pid -gt 0 -and $h4.owner_creation_ticks -is [long] -and $h4.owner_creation_ticks -gt 0 -and $h4.entry_counter_s -is [double] -and [double]::IsFinite($h4.entry_counter_s) -and $h4.entry_counter_s -gt 0 -and $h4.job_name -is [string] -and $h4.job_name.Length -le 256 -and $h4.job_name.StartsWith('Local\flylab-startup-ray-',[StringComparison]::Ordinal)) 'Same actual typed parent ENTRY identity'
K-PinShape $h4.request $(if($h4.call_id -ceq (K-Calls)[0]){$kb+'/s28-d19bz-full4-request.json'}else{$kb+'/parent-'+$h4.call_id+'-request.json'});Need ($h4.request.bytes -le 262144) 'Actual bounded parent request descriptor'
$h2='generated';$h3=$g7+'|'+[BitConverter]::DoubleToInt64Bits($h4.entry_counter_s);$path=[IO.Path]::Combine($g9.FullName,'s28-catalog-leaves.jsonl')
}
if($null -ne $script:KCatalogJournal){
K-Keys $script:KCatalogJournal @('path','context','identity','command','root','ordinal','bytes','ticks','frequency')
foreach($key in @('path','context','identity','command','root')){Need ($script:KCatalogJournal[$key] -is [string]) 'Exact retained journal context types'}
foreach($key in @('ordinal','bytes','ticks','frequency')){Need ($script:KCatalogJournal[$key] -is [long]) 'Exact retained journal counters'}
Need ($script:KCatalogJournal.ordinal -ge 0 -and $script:KCatalogJournal.ordinal -le 1152 -and $script:KCatalogJournal.bytes -ge 0 -and $script:KCatalogJournal.bytes -le 524288 -and [string]::Equals($script:KCatalogJournal.command,$g7,[StringComparison]::Ordinal) -and [string]::Equals($script:KCatalogJournal.root,$i9,[StringComparison]::Ordinal)) 'No partial foreign journal context adoption'
if($h2 -ceq 'canonical' -and $script:KCatalogJournal.identity -cne $h3){
Need ($script:KCatalogJournal.context -ceq 'canonical' -and $script:KCatalogJournal.identity -cmatch '^[0-9a-f]{32}$' -and [string]::Equals($script:KCatalogJournal.path,(Safe ($kb+'/s28-catalog-leaves-'+$script:KCatalogJournal.identity+'.jsonl')),[StringComparison]::Ordinal)) 'Only new authenticated canonical transaction selects fresh journal'
}
}
if($null -eq $script:KCatalogJournal -or ($h2 -ceq 'canonical' -and $script:KCatalogJournal.identity -cne $h3)){
$script:KCatalogJournal=@{path=$path;context=$h2;identity=$h3;command=$g7;root=$i9;ordinal=[long]0;bytes=[long]0;ticks=[long]0;frequency=[long][Diagnostics.Stopwatch]::Frequency}
}else{
Need ($script:KCatalogJournal.context -ceq $h2 -and $script:KCatalogJournal.identity -ceq $h3 -and [string]::Equals($script:KCatalogJournal.path,$path,[StringComparison]::Ordinal) -and [string]::Equals($script:KCatalogJournal.command,$g7,[StringComparison]::Ordinal) -and [string]::Equals($script:KCatalogJournal.root,$i9,[StringComparison]::Ordinal)) 'Same authenticated journal context; unknown context never reused'
}
$j=$script:KCatalogJournal;$h7=[long][Diagnostics.Stopwatch]::GetTimestamp()
Need ($j.ordinal -ge 0 -and $j.ordinal -lt 1152 -and $h7 -ge $j.ticks -and $j.frequency -eq [Diagnostics.Stopwatch]::Frequency -and $j.frequency -gt 0) 'S28 finite ordered same-frequency journal'
$h8=@();$h9='';$i0=@()
if($leaf -ceq 'catalog-support' -and $boundary -ceq 'END' -and $counters.Length){
Need ($counters.Length -eq 6) 'Compact first refusal tuple'
foreach($i6 in @(0,1,2,3,4)){Need ($counters[$i6] -is [int]) 'Refusal tuple numeric types'}
Need ($counters[0] -ge 0 -and $counters[0] -le 3 -and $counters[1] -ge -1 -and $counters[1] -le 779 -and $counters[2] -ge 0 -and $counters[2] -le 10 -and ((($counters[3] -ge 0 -and $counters[3] -le 12) -or ($counters[3] -ge 14 -and $counters[3] -le 19) -or ($counters[3] -ge 22 -and $counters[3] -le 35) -or ($counters[3] -ge 38 -and $counters[3] -le 137)) -or ($counters[0] -eq 0 -and $counters[1] -eq -1 -and $counters[2] -eq 0 -and $counters[3] -eq 13 -and $counters[4] -eq 0 -and $counters[5] -ceq 'DIAGNOSTIC_UNAVAILABLE')) -and $counters[4] -ge 0 -and $counters[4] -le 8 -and $counters[5] -is [string] -and $counters[5].Length -gt 0 -and $counters[5].Length -le 96 -and $counters[5] -cmatch '^[A-Za-z0-9_.+\x60\[\]]+$') 'Bounded safe first refusal tuple'
$i0=$counters
}elseif($counters.Length){Need ($counters.Length -eq 9) 'Actual catalog counter snapshot';$h8=@($counters[0..6]);$h9=$counters[7]}
$i1=$leaf -cin @('catalog-init','catalog-schema','catalog-IO')
$i2=[ordered]@{schema='flylab.m3.s28.catalog-leaf.v1';ordinal=$j.ordinal;leaf=$leaf;boundary=$boundary;ticks=$h7;frequency=$j.frequency;nested=$i1;inclusive=$true;actual=$h8;order_digest=$h9}
if($h2 -ceq 'canonical' -and $j.ordinal -eq 0){
Need ($leaf -ceq 'catalog-support' -and $boundary -ceq 'BEGIN' -and $counters.Length -eq 0) 'First canonical journal boundary is authenticated support BEGIN'
$i2.owner=@($h5.action,$h5.owner_pid,$h5.owner_creation_ticks,$h5.invocation_id)
}
if($i0.Length){$i2.refusal=$i0}
$line=($i2|ConvertTo-Json -Depth 5 -Compress)+[char]10
$i5=$utf8.GetBytes($line)
if($i5.LongLength -gt 384 -and $i0.Length){$i2.refusal=@($i0[0..4])+'UNKNOWN';$line=($i2|ConvertTo-Json -Depth 5 -Compress)+[char]10;$i5=$utf8.GetBytes($line)}
Need ($i5.LongLength -le 384 -and $j.bytes+$i5.LongLength -le 524288) 'S28 derived finite record/byte caps'
$i4=$null;$i3=$null
try{
$i7=if($j.ordinal -eq 0){[IO.FileMode]::CreateNew}else{[IO.FileMode]::Open}
$i4=[IO.File]::Open($j.path,$i7,[IO.FileAccess]::Write,[IO.FileShare]::Read)
Need ($i4.Length -eq $j.bytes) 'Exact prior actual journal append boundary'
$null=$i4.Seek(0,[IO.SeekOrigin]::End);$i4.Write($i5,0,$i5.Length);$i4.Flush($true)
$j.bytes+=$i5.LongLength;$j.ordinal++;$j.ticks=$h7
}catch{$i3=$_.Exception;throw}finally{
if($null -ne $i4){try{$i4.Dispose()}catch{if($null -eq $i3){throw};try{[Console]::Error.WriteLine('S28 journal secondary cleanup failure')}catch{}}}
}
}
function K-CatalogInit {
Need ($script:KManagedSourceAccepted -ceq $true) 'Catalog backend only inside accepted owned/admitted action'
$null=K-SourceFence (K-BoundedPin (K-Paths)[$zk])
$d8=[ordered]@{'path'='verification/banc-natural-gait-m3-20260919/c5.json';'bytes'=[long]130832;'sha256'='9398fcc3a61b0e8019b2599c40ca8dd7e0ebaa4e60302b71a74c2dd7a1a32d78'}
Need ($script:KCatalogManifestContract -is [Collections.IDictionary]) 'Actual accepted catalog contract descriptor'
K-PinShape $script:KCatalogManifestContract (K-Paths).contract
Same $script:KCatalogManifestContract $d8 'Independent accepted catalog contract origin'
$d9=K-Read $d8
Same $d9.epoch (K-Epoch) 'Current catalog source contract epoch'
$code=$d9.catalog_backend_code
K-Keys $code @('schema','source','template_sha256','rendered_sha256')
Same $code.schema 'flylab.m3.s28.catalog-code.v1' 'Exact authenticated catalog code envelope schema'
Need ($code.source -is [string] -and $code.template_sha256 -is [string] -and $code.rendered_sha256 -is [string] -and $code.template_sha256 -cmatch '^[0-9a-f]{64}$' -and $code.rendered_sha256 -cmatch '^[0-9a-f]{64}$') 'Typed catalog code envelope'
$f9=$code.source;$e0=$utf8.GetBytes($f9)
Need ($e0.Length -gt 0 -and $e0.Length -le 262144) 'Bounded whole catalog source UTF8'
foreach($f8 in $f9.ToCharArray()){$f7=[int]$f8;Need (($f7 -ge 32 -and $f7 -le 126) -or $f7 -eq 9 -or $f7 -eq 10) 'Exact ASCII catalog source formatting'}
$e1='5571aff448167222167082010d2de8e8a0d07e35f00b2fca0ebd7b30cee7eb3d';$e2='7a696f991deece427f20422d5d589e61cd7a9f8fd1407953579b823906782d16';$e3='S28Catalog_5571aff448167222167082010d2de8e8a0d07e35f00b2fca0ebd7b30cee7eb3d'
Same $code.rendered_sha256 $e2 'Hardcoded rendered catalog source identity'
Same $code.template_sha256 $e1 'Hardcoded catalog template identity'
$e4='(?<![A-Za-z0-9_])'+[regex]::Escape($e3)+'(?![A-Za-z0-9_])'
$e5=[regex]::Matches($f9,$e4,[Text.RegularExpressions.RegexOptions]::CultureInvariant)
Need ($e5.Count -eq 1) 'Exact sole catalog type identifier'
$e6=$e5[0];$template=$f9.Substring(0,$e6.Index)+'S28Catalog_@IDENTITY@'+$f9.Substring($e6.Index+$e6.Length)
Need (([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($utf8.GetBytes($template)))).ToLowerInvariant() -ceq $e1) 'Whole decoded catalog template SHA'
Need ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($utf8.GetBytes($f9))).ToLowerInvariant() -ceq $e2) 'Literal S28 rendered C# SHA, separate template identity'
$e8=@(@{name='Supports';parameters=[type[]]@([object]);result=[bool]},@{name='Verify';parameters=[type[]]@([object],[string],[object],[Action[string,string,object[]]]);result=[object[]]},@{name='Snapshot';parameters=[type[]]@();result=[object[]]})
if($null -eq $script:KCatalogCache){
Need ($null -eq $script:KCatalogOrigin) 'Partial catalog backend never name-lookup adoption'
$g5=[long][Diagnostics.Stopwatch]::GetTimestamp()
$g0=@(Add-Type -TypeDefinition $f9 -PassThru -ErrorAction Stop -WarningAction Stop)

$g6=[long][Diagnostics.Stopwatch]::GetTimestamp()
if(-not ($g0.Length -eq 1 -and $g0[0] -is [type])){
$f4='Actual catalog Add-Type one returned Type';$g1=$g0.Length;$f5=$f4+'; total='+$g1+'; displayed=0; omitted='+$g1+'; diagnostics_unavailable'
try{
$g2=[Collections.Generic.List[object]]::new()
for($n=0;$n -lt [Math]::Min(8,$g1);$n++){
$item=$g0[$n];$g3=$item -is [type]
$name=if($g3){$item.FullName}elseif($null -eq $item){'<null>'}else{$item.GetType().FullName}
if($null -eq $name){$name='<null FullName>'};$name=[string]$name;$cut=$name.Length -gt 128
if($cut){$limit=128;if([char]::IsHighSurrogate($name[$limit-1])){$limit--};$name=$name.Substring(0,$limit)}
$g2.Add([ordered]@{kind=if($g3){'Type'}else{'non-Type CLR kind'};name=$name;name_truncated=$cut})
}
do{
$detail=([ordered]@{total=$g1;displayed=$g2.Count;omitted=$g1-$g2.Count;items=$g2.ToArray()}|ConvertTo-Json -Depth 4 -Compress)
$f6=$f4+'; '+$detail
if($utf8.GetByteCount($f6) -le 4096){$f5=$f6;break}
if($g2.Count -eq 0){break};$g2.RemoveAt($g2.Count-1)
}while($true)
}catch{}
throw $f5
}
$type=$g0[0];$f2=$type.Assembly;Need ($type.FullName -ceq $e3 -and $f2 -is [Reflection.Assembly]) 'Actual returned catalog Type/Assembly'
$f0=@();foreach($e7 in $e8){
$f1=$type.GetMethod($e7.name,[Reflection.BindingFlags]'Public,Static',$null,$e7.parameters,$null)
Need ($f1 -is [Reflection.MethodInfo] -and $f1.DeclaringType -eq $type -and $f1.IsPublic -and $f1.IsStatic -and -not $f1.IsGenericMethod -and $f1.ReturnType -eq $e7.result) 'Exact actual catalog public static nongeneric method'
$e9=$f1.GetParameters();Need ($e9.Length -eq $e7.parameters.Length) 'Exact actual catalog signature length';for($i=0;$i -lt $e9.Length;$i++){Need ($e9[$i].ParameterType -eq $e7.parameters[$i]) 'Exact actual catalog parameter'};$f0+=,$f1
}
$script:KCatalogOrigin=[object[]]@($type,$f2,$f0[0],$f0[1],$f0[2],$f9,$e2,$e1)
$script:KCatalogCache=@{type=$type;assembly=$f2;support=$f0[0];verify=$f0[1];snapshot=$f0[2];source=$f9;source_hash=$e2;template_hash=$e1;compile_start_ticks=$g5;compile_end_ticks=$g6;frequency=[long][Diagnostics.Stopwatch]::Frequency;warm_reuses=[long]0}
}else{
$g4=$script:KCatalogCache;$f3=$script:KCatalogOrigin
K-Keys $g4 @('type','assembly','support','verify','snapshot','source','source_hash','template_hash','compile_start_ticks','compile_end_ticks','frequency','warm_reuses')
Need ($f3 -is [array] -and $f3.Length -eq 8) 'Whole catalog backend origin'
Need ([object]::ReferenceEquals($g4.type,$f3[0]) -and [object]::ReferenceEquals($g4.assembly,$f3[1]) -and $g4.type -is [type] -and $g4.assembly -is [Reflection.Assembly] -and $g4.type.Assembly -eq $g4.assembly -and $g4.type.FullName -ceq $e3) 'No foreign actual catalog Type/Assembly'
$names=@('support','verify','snapshot');for($i=0;$i -lt 3;$i++){
$f1=$g4[$names[$i]];$e7=$e8[$i]
Need ([object]::ReferenceEquals($f1,$f3[2+$i]) -and $f1 -is [Reflection.MethodInfo] -and $f1.DeclaringType -eq $g4.type -and $f1.Name -ceq $e7.name -and $f1.IsPublic -and $f1.IsStatic -and -not $f1.IsGenericMethod -and $f1.ReturnType -eq $e7.result) 'No foreign or stale actual catalog Method'
$e9=$f1.GetParameters();Need ($e9.Length -eq $e7.parameters.Length) 'Warm exact catalog signature length';for($n=0;$n -lt $e9.Length;$n++){Need ($e9[$n].ParameterType -eq $e7.parameters[$n]) 'Warm exact catalog parameter'}
}
Need ([string]::Equals($g4.source,$f9,[StringComparison]::Ordinal) -and [string]::Equals($f3[5],$f9,[StringComparison]::Ordinal) -and $g4.source_hash -ceq $e2 -and $f3[6] -ceq $e2 -and $g4.template_hash -ceq $e1 -and $f3[7] -ceq $e1) 'No stale catalog literal/source/hash cache'
foreach($name in @('compile_start_ticks','compile_end_ticks','frequency','warm_reuses')){Need ($g4[$name] -is [long]) 'Exact catalog observation integer'}
Need ($g4.compile_start_ticks -ge 0 -and $g4.compile_end_ticks -ge $g4.compile_start_ticks -and $g4.frequency -eq [Diagnostics.Stopwatch]::Frequency -and $g4.frequency -gt 0 -and $g4.warm_reuses -ge 0) 'Actual monotonic catalog compile/warm provenance'
$g4.warm_reuses++
}
}
function K-CatalogInvoke([Reflection.MethodInfo]$method,[object[]]$arguments) {
try{return ,($method.Invoke($null,$arguments))}catch{
$cause=$_.Exception
while($null -ne $cause.InnerException -and ($cause -is [Reflection.TargetInvocationException] -or $cause -is [Management.Automation.MethodInvocationException] -or $cause -is [Management.Automation.RuntimeException])){$cause=$cause.InnerException}
[Runtime.ExceptionServices.ExceptionDispatchInfo]::Capture($cause).Throw();throw
}
}
function K-CatalogVerified($v) {

$null=K-SourceFence (K-BoundedPin (K-Paths)[$zk])
K-CatalogJournal 'catalog-support' 'BEGIN'
K-CatalogJournal 'catalog-init' 'BEGIN';K-CatalogInit;K-CatalogJournal 'catalog-init' 'END'
$ka0=[object[]]::new(1);$ka0[0]=$v
$kc0=$script:KCatalogCache;$ky0=$kc0.type;$kt0=[Threading.Thread]::CurrentThread.ManagedThreadId
Need ([object]::ReferenceEquals($ky0,$script:KCatalogOrigin[0]) -and [object]::ReferenceEquals($ky0.Assembly,$kc0.assembly) -and $kc0.source_hash -ceq $script:KCatalogOrigin[6]) 'Source-owned actual refusal Type/Assembly/hash'
$kf0=[Reflection.BindingFlags]'NonPublic,Static,DeclaredOnly';$ks0=$ky0.GetField('RefusalSlot',$kf0);$kq0=$ky0.GetField('RefusalSequence',$kf0)
foreach($kd0 in @($ks0,$kq0)){Need ($kd0 -is [Reflection.FieldInfo] -and $kd0.DeclaringType -eq $ky0 -and $kd0.IsPrivate -and $kd0.IsStatic -and -not $kd0.IsInitOnly -and $kd0.IsDefined([ThreadStaticAttribute],$false)) 'Exact private source-owned thread-local refusal field'}
Need ($ks0.FieldType -eq [object[]] -and $kq0.FieldType -eq [long]) 'Fixed private refusal field types'
$kj0=$kq0.GetValue($null);Need ($kj0 -is [long] -and $kj0 -ge 0 -and $kj0 -lt [long]::MaxValue) 'Bounded actual same-thread refusal generation'
try{$kp0=K-CatalogInvoke $kc0.support $ka0}catch{try{[Console]::Error.WriteLine('DIAGNOSTIC_UNAVAILABLE')}catch{};throw}
Need ($kp0 -is [bool]) 'Exact pure catalog support Boolean'
$kr0=@()
try{
$u1=$ks0.GetValue($null);$kz0=$kq0.GetValue($null)
Need ([Threading.Thread]::CurrentThread.ManagedThreadId -eq $kt0 -and $u1 -is [object[]] -and $u1.Length -eq 9 -and $u1[0] -is [int] -and $u1[0] -eq 1 -and $u1[1] -is [int] -and $u1[1] -eq $kt0 -and $u1[2] -is [long] -and $kz0 -is [long] -and $u1[2] -eq $kz0 -and $kz0 -eq $kj0+1) 'Exact current-thread nonstale refusal slot'
foreach($ki0 in @(3,4,5,6,7)){Need ($u1[$ki0] -is [int]) 'Refusal numeric enum types'}
Need ($u1[3] -ge 0 -and $u1[3] -le 3 -and $u1[4] -ge 0 -and $u1[4] -le 10 -and (($u1[5] -ge 0 -and $u1[5] -le 12) -or ($u1[5] -ge 14 -and $u1[5] -le 19) -or ($u1[5] -ge 22 -and $u1[5] -le 35) -or ($u1[5] -ge 38 -and $u1[5] -le 137)) -and $u1[6] -ge -1 -and $u1[6] -le 779 -and $u1[7] -ge 0 -and $u1[7] -le 8 -and $u1[8] -is [string] -and $u1[8].Length -gt 0 -and $u1[8].Length -le 96 -and $u1[8] -cmatch '^[A-Za-z0-9_.+\x60\[\]]+$') 'Fixed bounded refusal enums/row/depth/ASCII kind'
if($kp0){Need ($u1[3] -eq 0 -and $u1[4] -eq 0 -and $u1[5] -eq 0 -and $u1[6] -eq -1 -and $u1[7] -eq 0 -and $u1[8] -ceq 'NONE') 'Supported call leaves reset refusal sentinel'}
else{Need ($u1[5] -gt 0) 'Unsupported call has mandatory first refusal';$kr0=@($u1[3],$u1[6],$u1[4],$u1[5],$u1[7],$u1[8])}
}catch{
$ke0=$_.Exception
try{K-CatalogJournal 'catalog-support' 'END' @([int]0,[int]-1,[int]0,[int]13,[int]0,'DIAGNOSTIC_UNAVAILABLE')}catch{try{[Console]::Error.WriteLine('DIAGNOSTIC_UNAVAILABLE')}catch{}}
[Runtime.ExceptionServices.ExceptionDispatchInfo]::Capture($ke0).Throw();throw
}
K-CatalogJournal 'catalog-support' 'END' $kr0
if(-not $kp0){return ,(K-CatalogAuth $v)}
$ka0=[object[]]::new(4);$ka0[0]=$v;$ka0[1]=$taskRoot
$ka0[2]=@{path='';bytes=[long]0;sha256='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}
$ka0[3]=[Action[string,string,object[]]]{param($leaf,$boundary,$counters) K-CatalogJournal $leaf $boundary $counters}
K-CatalogJournal 'catalog-backend' 'BEGIN'
$kv0=K-CatalogInvoke $script:KCatalogCache.verify $ka0
$km0=K-CatalogInvoke $script:KCatalogCache.snapshot ([object[]]@())
Need ($kv0 -is [object[]] -and $km0 -is [object[]] -and $km0.Length -eq 9) 'Detached actual catalog output/counters'
K-CatalogJournal 'catalog-backend' 'END' $km0
return ,$kv0
}
&{$a=(K-Read $script:KCurrentContract).d19_bind;[byte[]]$b=(K-SourceRaw (Safe $a[0]) 393216)[$a[1]..$a[2]];Same ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($b))) $a[3];iex $utf8.GetString($b)}
