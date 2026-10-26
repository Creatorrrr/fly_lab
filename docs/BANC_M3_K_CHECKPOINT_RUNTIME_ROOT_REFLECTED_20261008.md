# M3 K 체크포인트 실행 복구 — ROOT 반영 구현 계약

2026-10-08. 기존 M3 유한 목표 ACTIVE / F1–F8 INCOMPLETE. 현재 실제 Source는 J이고, K 실행·문법·속도·bootstrap·qualification·보행·생물학은 NOT_RUN 또는 UNKNOWN이다. 이번 구현은 실제 후보-k 신체 평가로 돌아가기 위한 한 단계다. 별도 proof 목표를 만들지 않는다. 사용자 최신 자율 판단 지시는 유지되며 추가 승인 요청은 하지 않는다.

## 실제 리뷰와 ROOT 결정

ROOT가 실제 gpt-6-astra/max의 review.md 전체10241바이트 / 4d36fd9bd659d4dc5ffe0c6312fc37316129acf216b0f5bfafd816326d84d2ae 및 review.json 전체17812바이트 / 06f57a6e6180175b11026e742013cdfdeccb944dcdd3fbe961d180c6675594f8을 읽고 재해시했다. 위치는 verification/banc-natural-gait-m3-20260919/parent-k-checkpoint-runtime-design-astra-review-20261008이다. 판정 ACCEPT_DESIGN_WITH_REQUIRED_REFLECTION_AMENDMENTS의 R1–R6를 모두 아래 계약에 채택한다. ROOT는 선언된30 입력도 독립 재해시하여 MATCH를 확인했다.

원 설계 docs/BANC_M3_K_CHECKPOINT_RUNTIME_ROOT_DESIGN_20261008.md는15092 / 0225683d4cf56932c347fef72ff90d4cd9d65b0cb269b49c2046205691718ce9이며 변경하지 않는다. 입력 manifest7249 / 6b96f8286b0c4dcab0d1bfe13effe74939803a456248f100d78d8c07a29db08b 및 K-before source map15761 / 26666fb0be28e4b0efc68eda07c6764e63e8e339fe24c903fd81c0a9ec6b5a58도 고정된다. source map의30 archive는 inert한 원본 전체 바이트 사본이다.

이 리뷰는 실행 통과나 ledger 변경 자체의 근거가 아니다. ROOT는 아래 계약에 따른 Source-only 구현을 실제 gpt-6.1-sol/max에 맡긴다. canonical 채택과 Native 행정 transaction 및 실제 보호 실행은 ROOT의 Source 검토와 명시된 전제 조건 후 별도로 수행한다. 400초는 ROOT의 전향적인 보존 배분이며 새 인간 수치 승인이 아니다.

## R1 — 닫힌 기준점과 유한 상태

정확한 settled288 pin은 Source literal이어야 한다: verification/banc-natural-gait-m3-20260919/parent-j-lint-settled288-HOLD-20261008.json,460487 / ba0cb692cae00e15a7693c8ae01279dc9d8b0af771380ad1af7c363051c9b55a. caller/event/cache가 anchor를 바꿀 수 없다. invocation마다 bounded raw와 duplicate-aware typed JSON을 인증한다. live ledger의 whole key set, immutable values,288 debit prefix,170 charged-call mapping,policy/grants/과거 pointer를 정확히 유지한다. accounting 기준점이 current executable Source 적합성이나 과학 결과를 대신하지 않는다.

K recovery와 allocation을 분리한다. recovery는 유일한 J per-call overrun HOLD의 정확한 제거와 recovery pointer 하나만 바꾼다. allocation은 allocation pointer와 globalGenerated2700/local1964/Original15200 및 두 remaining 값만 바꾼다. 합계17900·actual fee·usage·offset118/165·history는 같다. old caps는2300/1564/15600이고 old allocation event는 유지한다.

구현은 recovery, bootstrap, allocation, 네 qualification, qualification registration, body, 조건부 audit에 대해 admitted/begun/closed-pending/unknown/charged의 exact typed layout과 가능한 각 delta를 선언한다. broad TAIL_KEYS로 dictionary 전체를 예외 처리하지 않는다. 새 call은 fixed sequence의 다음 literal identity만 허용한다. paid failure는 다음 단계 성공 전제가 아니다. extra/old/replay/alias call, 임의 pointer·hold·owner 변경, 잘못된 caps나 Source 변화는 거부한다. unknown은 책임/소유를 유지하는 정산 상태이며 실행 적격이 아니다.

합산은 기존 binary64 debit order 및 cap-minus-baseline-minus-stage 방식 그대로다. cap-minus-cumulative-used shortcut은 사용하지 않는다. Astra가 실제 Original 값에서4.547473508864641e-13 차이를 확인했다. 정수/실수/bool 구분, ordinal key set, duplicate·nonfinite 거부를 유지한다. history의 proof graph를 다시 재귀 실행할 필요 없이 고정 prefix의 bounded numeric 합을 검사한다.

## R2 — 닫힌 비용은 current Source와 독립하여 정산

K Native fee의 admitted-identity, request/trace/outer/closure, once-fee, accounting-validity의 전체 dependency closure는 account에 inline으로 유지한다. core14933은 byte-identical하게 정확히 한 번 포함한다. 외부 epoch/helper load, current executable Source 검증, 과거 recursive validators를 K settlement에서 호출하지 않는다. intact immutable evidence가 있는 종료한 call은 executable drift가 있더라도 청구한다. immutable evidence가 없거나 부패하면 PENDING/HOLD를 유지하며 PASS를 만들지 않는다.

비용 기록의 유효성과 다음 실행 허용을 별개로 판단한다. nonzero/timeout/per-call 또는 전체 overrun에도 exact finite genuine fee를 한 번 게시한다. cap_exceeded와 HOLD를 남긴다. 실제 지출로 remaining이 음수가 된 경우, 그것만으로 incurred debit 게시를 막지 않는다. eligibility에서 부족한 자원·hold·unknown을 거부한다. 순서를 바꿔 환불하거나 unknown을0으로 취급하지 않는다.

K outer는 mandatory worker_started/resume outcome을 포함하며 no-create-attempt, created-suspended-not-resumed, resumed, uncertain creation을 구분한다. suspended worker도 실제 Job 종료와 두 observed EOF가 필요하다. lexical no-create refusal에서는 EOF를 만들어 넣지 않는다. optional perf_counter phase 값은 genuine collector의 dispatch→first FINAL 비용을 대신하지 않는다. 기존 J 기록은 수정하지 않는다.

## R3 — 실제 모든 경로의 K 우선 분기

K ledger는 보존된 J pointer보다 우선한다. Native action setup/Current-Policy/caps/Generated-Budget/admit/commit/qualification/grant·rebind 금지 및 Python global/locked dispatch/begin·closed·unknown/refusal/budget, contact binding/history/candidate/worker/audit journal까지 route matrix로 표시한다. supervisor279–281/953–954의 caps!=NEW_CAPS fallback과 Native commit1900/1916의 J-first는 K를 명시적으로 다뤄야 한다. 기존 NEW_CAPS2300 상수를 바꾸어 역사를 혼합하지 않는다.

recovery action은 canonical Source가 K로 채택된 뒤에도 old J current-Source validator를 먼저 호출하지 않고 정확한 baseline·known J proof로 전용 분기한다. K current_caps/begin/closed/settlement에서 j_history/Conserved-J-History 또는 과거 current review를 호출하면 테스트가 실패하도록 한다.

actual SharedLock 획득 후 fresh request/freeze/전체 현재 Source를 worker 생성 전에 검사한다. invocation-local reuse는 이 cold check를 대체할 수 없다. common Native exclusion/Job/creation ticks/same ENTRY/deadline/cleanup/두 EOF/atomic durability를 유지한다. Source validation은 현재 실행 경로에 필요하고, already incurred fee를 막는 조건이 아니다. Original cold fence·candidate·8 preparation journal slot·CPU/CUDA 경로는 고정된다.

existing conserved_run_native(test journal1363–1396)에 actual K bootstrap90와 gate-B360 parent만 추가한다. real parent token/request/Job/exclusion/creation identity, exact progress와 ENTRY deadlines를 인증한다. K임을 가장하기 위해 token이나 q를 old call로 바꾸지 않는다. 모든 기존 parent assertion은 보존한다. 공통 low-level runner를 추출하거나 명시적 K entry를 만들 수 있지만 그 구현·dependency·현재 Source가 fixture 전체에서 인증되어야 한다.

## R4 — legacy executable와 기존 fixture의 정확한 Source

canonical tools/banc_m3_conserved_resource_native_epoch.ps1의 실제 J126162 / a790ccde241733ed7e23bc8bbba63d4a9833c02708567907b385ef0d33af8116을 그대로 둔다. lazy loader는 dedicated immutable executable tools/banc_m3_checkpoint_legacy_epoch_native.ps1에서 이 전체 바이트를 인증하고 올바른 PowerShell scope로 로드한다. public legacy signatures와 taskRoot/extraction sentinels를 보존한다. current mutable historical path의 unauthenticated fallback, arbitrary executable archive, encoded executable/eval/Invoke-Expression은 금지한다. K fee는 loader와 무관하다.

G/H/I temporary roots는 일곱 canonical historical Source를 복원하므로 그 자리에서 current J pin을 요구하면 실패한다. 새 fixed legacy dependency와 immutable records를 first wrapper 전에 explicit staging한다. conserved_copy1142–1156의 archived exact-pair lookup에는 Source literal의 pinned K-before map만 추가해 post-J bytes를 찾는다. j_fixture3758–3774가 요구하는 as-executed J operands를 이 map의 source/archive exact pair에서 복원한다. metadata operand Source와 실행할 dedicated legacy Source는 구분한다.

현재 account262144/test262143에서 기존 assertion·parameter identity와 보호349/375/403 chain을 삭제하지 않는다. 새 K tests/helper는 별도 파일에 둔다. 기존 fixture plumbing 변경이 필요하면 actual before/after byte ranges와 모든 보호 statement의 byte evidence를 보고한다. Source/comment 또는 새 private fixture helper의 비의미적 정리는 가능하나 scientific assertions를 약화하지 않는다. actual AFTER sizes와 disjoint byte accounting으로 상한을 증명한다.

## R5 — 고정된 Source와 순환 없는 실제 publication

유일한 체인은 ROOT/Astra provenance + static K contract → reviewed Source vector → freeze → recovery → bootstrap request/actual proof → fresh closed allocation event → qualification → Original requests다. Source/static descriptor/freeze는 자기 또는 미래 cost/event/request/qualification hash를 pin하지 않는다. future fixed path는 hash pin이 아니다. tested freeze는 bootstrap 뒤 변경하지 않으며 allocation은 실제 성공 bootstrap과 같은 Source에 결속한다. Source가 바뀌면 이전 bootstrap을 새 Source에 재사용할 수 없다.

J728 identity 전체를 유지하며 선언된 replacements만 새 pin으로 바꾸고 새 checkpoint modules/dedicated legacy executable/fixture modules/K roster/static metadata를 명시한다. key-set 전체와 모든 unchanged pins를 비교한다. release-time whole Source 인증은 ROOT가 수행한다. Astra가728 전체 Source를 읽었다고 주장하지 않는다. current J freeze255412의6732 headroom 안에 모든 추가 dependency가 맞는 concrete JSON을 만들고 actual byte fit를 보고한다. ledger anchor는 evidence cap524288이며 executable Source인 것처럼 취급하지 않는다. checkpoint modules는 각각65536 이하, 나머지 bounded new/changed Source/fixture/freeze는262144 이하. Python Ruff에 모든 기존 targets와 새 Python dependencies를 포함하고 PS dependency는 paid Native tests로 검사한다.

ROOT가 사용할 fixed metadata/action names는 다음과 같다. 아직 published runtime pin이 아니다.

- static checkpoint contract: parent-contact-checkpoint-contract-20261008-k.json
- static body: parent-contact-body-runtime-20261008-k.json
- reviewed Source: parent-contact-checkpoint-reviewed-source-20261008-k.json
- freeze: parent-contact-learning-source-freeze-20261008-k.json
- recovery: parent-contact-checkpoint-recovery-20261008-k.json / ledger direct_science_checkpoint_recovery / Native checkpoint-recovery
- allocation: parent-contact-checkpoint-allocation-20261008-k.json / ledger direct_science_checkpoint_allocation / Native checkpoint-allocation
- qualification: parent-contact-learning-qualified-20261008-k.json / ledger direct_science_checkpoint_qualification
- fixed bootstrap direct-science-20261008-contact-k-bootstrap; qualification direct-science-20261008-contact-k-lint, -math, -gate-a, -gate-b
- body contact-learning-20261008-k-body 및 body-audit; maximum one each, audit only valid body.

metadata는 모두 verification/banc-natural-gait-m3-20260919 아래다. Source proposal 단계에서 runtime recovery/allocation/qualification은 blueprint만 게시하며 canonical/ledger authority로 쓰지 않는다. qualify transaction은 old direct_science_qualification을 덮어쓰지 않고 K pointer를 추가하며 old grant/rebind를 K에서 차단한다.

## R6 — 실제 meaningful bootstrap에서 신체 평가까지

보호 실행 전에 bootstrap argv/node IDs/JUnit module/class/name 및 전체 K math/gate roster를 정확히 고정한다. generated temporary roots에서 production Python/Native parity, full prefix/types/legal state negatives, recovery/allocation exact deltas, foreign lock/stale raw/atomic replace+readback, paid Source-drift refusal, invalid grant/mixed caps, unknown owner/EOF, duplicate commit와 overrun을 검사한다. 실제 live ledger/Original payload를 fixture가 수정·읽지 않는다. fixture cost/PASS는 synthetic이며 real bootstrap 비용·JUnit은 실제 execution에서만 나온다.

bootstrap90/75/15+Generated close20은 old2300/1564/15600의 남은 direct527.2560000000001에서만 쓴다. 실제 exit0·complete successful JUnit·once-paid·known closed/noHOLD·fresh actual before·같은 tested Source가 있어야 allocation+400/-400이 가능하다. 새 cap으로 자기 bootstrap을 지불하지 않는다. Source review/행정 recovery는 empirical bootstrap PASS가 아니다.

full K qualification은 old19 math/21 gate-A/32 gate-B와 모든 parameter group을 보존하고 새 사례를 exact roster에 더한다. disjoint union/indivisible Native groups를 검증한다. lint40/35/5 +math35/30/5 +gate-A120/110/10 +gate-B360/340/20 +close20=575이며 네 actual 비용/세 exact JUnit은 bootstrap과 별도다. first role 전에두 pool 전체575를 확인한다. missing/skipped/duplicate/error/failure/Source drift/unknown 비용은 Original을 차단한다.

통과하면 같은 후보-k의 실제 body300/285/internal270/cleanup15를 한 번 수행한다. body genuine exit0와 run_valid=true일 때만 audit120/100/internal90/cleanup20, Original close180을 수행한다. ROOT 채택 후보나 자연 보행 완료가 아니라 actual startup·contact·movement/phase·health gate와 F1–F8를 원 기준으로 판정한다. 실제 preparation/neural/physics UNKNOWN은0이 아니다. 이후 남은 과학적 gap을 계속 다룬다.

## Sol handoff와 ROOT 검토

FIRST owner는 actual gpt-6.1-sol/max 한 명이다. 출력 directory verification/banc-natural-gait-m3-20260919/parent-k-checkpoint-runtime-sol-FIRST-20261008의 sibling proposals와 FIRST.md/FIRST.json만 쓴다. canonical Source·ledger·plan·old artifacts 수정, 실행/import/parser/AST/compile/lint/pytest/Native/process/CUDA/Original payload/NPZ/header/stat/hash는 금지한다. ordinary bounded PS/.NET bytes/hash/text/JSON, rg, literal edits만 사용한다. no subagents. 모델 effort max는 실제 통합 경로·Native/Python 상태 계약의 난이도 때문이다.

원 설계/본 반영/Astra actual reports/immutable inputs/source-before map을 먼저 읽고 현재 Source에 적용한다. 기존 failure48.607/124와 current HOLD/모든 비용을 그대로 둔다. 최소 concrete changes는 account/resource/supervisor/contact admission/contact Native/worker 또는 journal route/fixture plumbing/new K modules+legacy dependency+K roster/static contract/body다. 필요한 Source closure 전체를 명시한다. FIRST는 actual inputs/outputs, full diff, route matrix, legal state/delta table, inline fee dependency closure, preserved assertions/APIs, cap proof, full vector/roster+blueprint와 unresolved runtime facts를 보고한다. 거대 Base64 dump 대신 inert raw Source copies/작은 complete report·범위별 byte evidence를 제공한다.

ROOT FIRST가 실제 reports 및 수정 Source를 읽고 필요하면 같은 Sol에게 sole R1을 맡긴다. ROOT SECOND 뒤 남은 수정은 ROOT가 마무리한다. Source가 concretely reviewable하고 모든 invariants와 fit가 맞아야 canonical adoption을 한다. 그 뒤 real Native recovery/admit/collector/once commit/readback로 실행을 검증한다. 알려진 failure를 지우거나 approval을 기다리는 방식으로 진행을 끝내지 않는다.
