# P26 고정 과거 목록의 보수적 합집합 — ROOT 설계, 2026-10-06

이 문서는 실제 Astra 검토를 위한 설계다. 현재 canonical Source나 실행 정책을 바꾸지 않으며, 검토 결과를 ROOT가 반영하기 전 Sol 구현을 재개하지 않는다. 기존 전체 v3 설계의 구조·실제 P25 전임자·예산·개발 순서는 유지한다. 바꾸려는 부분은 정확한 고정 역사 목록의 회원 수와 Source coverage 판단이다.

## 실제 근거와 정정

같은 GPT-6.1 Sol/xhigh가 읽기 전용 finite audit를 안정 인계했다. `parent-e360-p26-sol-FIRST-coverage-consolidated-handoff.json`116410B/SHA3af81bfa85b37d774a2897953db8b0a3c9ad84792766a0762db6847cc311111c, 설명3547B/SHAe9096b1e3b011d091af9389b022b45f7fa445d7158f7d30736f35f62942cc726다. Source FIRST 구현/ROOT FIRST/sole ONE_R1/runtime은 미수행이다. 현재712 목록은 완전 coverage에 부족하다.

고정18 reader의 명시적 메타데이터, 8개 지정 historical freeze/policy, closed107의 실제 cost42..106/57 requests 및 지정 cost/actual/manifest stream 필드에서 정확한 누락411개를 보고했다. raw audit2281597B/SHA8962454e3d8d29231dfb221c24d8e2a0376c23db598fe034431c77ac99e24a3e는 Source 전용이며 production JSON에 주입하지 않는다. relevance713154B/SHA2fe0b18027b08a3a989cfd15303a85d35a6509c263271f8d67b7a31cec13d5e3의 원래 분류341 required/70 unresolved는 그대로 보존한다.

ROOT가 실제 `_p17_recovery_amendment`3376–3378의 `for spec in value.values()` 직접 reader와3379–3380 snapshot/Native loop를 확인했고, Sol이 stable handoff에서 reviewed_design/reflected_design 두 누락의 직접 소비를 확인했다. 따라서 확인된 required는343개, 남은 relevance unresolved는68개다. 총411개와 원래 분류 보고서는 바뀌지 않는다. 나머지68개가 사용되지 않거나 모두 필수라고 주장하지 않는다. 관측된 identity 충돌/shape 오류/physical mismatch는0이나 전체 호출 그래프·별도 제외 경로의 coverage는 미확인이다.

## 제안하는 회원 선택

ROOT는 기존712 + 명시적으로 이름이 주어진411개 전체의 **정확한 보수적 합집합1123**을 선택하려 한다. required343와 unresolved68의 구분은 증거에서 유지한다. 불명확한 항목을 런타임에서 탐색·추가하는 것보다 기존 메타데이터가 이미 명시한 유한 후보를 고정하는 선택이다. 불명확한 항목의 필요성·소비·검증 성공을 회원 포함만으로 인증하지 않는다.

ROOT의 실제 Source 전용 준비는 모든1123개 root-contained relative text leaf에 stat/bytes/SHA/strictUTF8/두 번의 동일 읽기를 수행했다. current mutable 제외와 canonical 중복을 독립 확인했다. 후보는 `parent-e360-p26-ROOT-flat-history-refined1123.json`205062B/SHA581f73814c6f3f411ab2b1985b403b3062f1622fcbefe85ceed5b1c3de79f74b, count1123/aggregate59932737B/maxleaf2097152B다. JSON은 기존262144B 상한 안이다. 준비 proof586217B/SHAfdadc28f08a896c26c190a59e8d65e36f0e05981f8f885bafb194d8876af5abc 역시 Source audit 전용이다.

dynamic512 + exact history1123, combined maximum1635로 **기존 total512 및 v3의 combined1224 계약을 명시적으로 개정**한다. 정확한 count/aggregate/membership/table pin을 정책·Source metadata에서 묶는다. arbitrary 추가 상한, eviction/reset/spill, metadata-only 회원의 강제 실행을 도입하지 않는다.

## 실제 소유권·검사 경로

v3의 한 actual scope/ENTRY 및 두 monotone 소유 목록을 유지한다. 고정 목록에 포함된 leaf를 실제로 소비했을 때만 owned-history subset으로 등록한다. 일반/current descriptor는 dynamic512 한도를 받는다. 두 목록은 canonical path 기준으로 서로 배타적이고 회원 identity 충돌·명시적 unknown historical member·missing required member는 poison/fail-closed한다. historical unknown을 dynamic overflow fallback으로 허용하지 않는다.

parsed-clear, descriptors_only_execution, failure handling 동안 두 consumed 목록이 살아 있고, 원래 모든 applicable fence가 두 목록의 모든 소비된 leaf를 같은 cold path/stat/bytes/SHA/strictUTF8/two-read/clock으로 검사한다. 읽지 않은1123개를 fence마다 전부 읽거나 parsed historical document/budget/selector/admission/fence 성공을 cache하지 않는다. table 파일을 런타임에서 읽으면 그 table pin도 dynamic manifest에 등록한다. 기존 logical→physical relocation은 실제 pass/context resolver가 계속 수행한다. Source audit의 설명용 alias map으로 production resolver를 교체하지 않는다. scratch-root는 다른 root의 파일·검증 성공을 빌리지 않는다.

기존 public consumed의 depth 정책과 private/direct enrollment를 바꾸지 않는다. empty/positive의 실제 depth0 등록, late drift/poison, fixed-reader consumed literal 확인을 두 목록에 맞게 유지한다. `.descriptors`를 직접 참조하는 lookup/cleanup과 empty stream 경로도 빠짐없이 반영한다. direct dynamic512/513, 양쪽 subset·중복·identity conflict·unknown history·parsed-clear·prepublication/pre-Popen/pre-array를 기존 named tests에서 실제 production 경로로 확인한다. mirrored predicate/new helper/framework/probe를 추가하지 않는다.

## Source coverage에 대한 검토 요청

합집합1123은 이 finite audit가 명시한411개 누락을 모두 담는다. 이것을 실제 runtime 전체 descriptor census, 모든 inherited callback의 coverage, original array/foreign Source/oversize 경로 검증 또는 timing PASS로 해석하지 않는다. original mutable36 seed/foreign93/full11-child14/oversize E3184033033B 제외와 기존 분리 경로는 유지한다.

실제 Astra는 정확한 finite 합집합이 Source 구현을 시작하기에 충분한지 판단한다. 기존 named18 reader/8 historical contexts의 직접 descriptor collection loop·field와 inherited callbacks에 **구체적으로 더 빠진 physical descriptor가 확인되면** 그 정확한 descriptor·Source 함수/필드 근거를 요구하고 현재 설계를 REVISE한다. unknown을 dynamic으로 흘리거나 finite audit의 미검증 범위를 성공으로 바꾸어 통과시키지 않는다. 새 범용 audit 프로젝트나 runtime recursive walk는 요구하지 않는다.

## 보존과 후속 실행

실제 Source/paid 전임자는 frozen P25와 closed107/57calls/24paidpasses/failed scanner30/next2다. current P26 비용은107부터, future27이며 unpaidP24를 paid로 만들지 않는다. actual Native activation107 및 same PROGRAM16의 whole107 positive/106·lock·pair negatives/Boolean consumer를 유지한다. Source4/Native4/collector/policy/helper/bootstrap/receipt/Q ABI, 원래84names/252phases/fixed18/Native15/full11/child14/nonownQ, original hot four-return/single-call 경로를 유지한다.

shared672.5270156/local561.57와 limits1700/1589, whole888, caps3/30/30/60/20/40/520/25/40, primary suffix765/735/**705**/645/625/585/65/40/0 및 모든 future/fallback **705** 값은 바뀌지 않는다. catalogue 숫자가 budget705 치환에 쓰이지 않는다. first closed failure/실제 FINAL clock/opaque output/Native int64/owned close/once debit를 그대로 유지한다. 이번 Source 준비·설계에는 grant/activation/original/body/CUDA/UI 실행이 없다.

실제 Astra 리뷰→ROOT 전체 반영→same Sol 적절한 effort FIRST→ROOT FIRST→필요시 same Sol sole ONE_R1→ROOT SECOND→ROOT 잔여→실제 apply/freeze/activation/genuine gate 순서다. 유효한 필요한 gate 후 반복 검증을 늘리지 않고 기존 startup 보존 한 후보의 same-instance0→99→300/실제1.5s body milestone으로 전환한다. original local0/20에 기존46–48s prefix가 들어가지 않으므로 실제 충분한 original allocation/저장/close/audit를 먼저 등록한다. fixed F1–F8/M2/UI/anatomy와 과거 실패를 유지한다. full goal ACTIVE/INCOMPLETE, Source/lint/Q/단기 성공은 자연·생물학적 보행 완료가 아니다.
