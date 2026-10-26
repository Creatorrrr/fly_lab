# M3 K — 실행 검증 체크포인트와 보존된 자원 복구: ROOT 설계 초안

2026-10-08. 기존 M3 유한 목표를 계속한다. F1–F8는 미완료다. 이 문서는 새로운 목표, 예산 초기화, 생물학적 보행 통과 또는 새로운 인간 승인을 뜻하지 않는다. 최신 직접 사용자 지시는 “앞으로 내 승인 받지 말고 스스로 판단해서 진행해줘. 개입할 상황이면 내가 알아서 개입할게”다. 내부 설계·구현·검증과 기존 총자원 내의 배분은 ROOT가 판단한다. 파괴적 외부 효과나 새 자원 총액은 범위 밖이다.

## 결과에 연결되는 현재 결정

BANC 신경망 → 모터 → 실제 물리의 후보-k 평가로 돌아가려면 실행 전에 반복되는 과거 감사 비용을 줄여야 한다. 실제 J 실행은 48.607초/exit124로 한 번 청구됐으며 Ruff는 실행되지 않았다. owner 생성부터 suspended worker 생성까지 31.2450283초이고, begin 이후 같은 ENTRY의 pre-resume 검사에서 child35가 만료됐다. 이 수치는 전체 검증의 병목을 보여주지만 개별 함수의 측정 시간이나 생물학적 실패를 보여주지는 않는다.

닫힌 기준점은 parent-j-lint-settled288-HOLD-20261008.json의 정확한 460487바이트 / ba0cb692cae00e15a7693c8ae01279dc9d8b0af771380ad1af7c363051c9b55a다. 실제 ledger는 이 바이트와 같고, 288 debit / 170 paid call, reservation·pending·owner·Original job 0이다. 물리 shared lock은 없다. 유일한 hold는 direct-science-20261008-contact-j-lint의 “Actual closure or resource cap unresolved”다. 실제 J cost는 48.607/124, cap_exceeded=true, all_owned_ended=true, reservation_released=true이며 두 EOF와 known creation identity가 확인됐다. 총액 초과나 미확정 종료를 이와 같은 hold로 일반화해서 해제해서는 안 된다.

현재 Generated global2300/local1564, Original15600, 합계17900이다. direct527.2560000000001/shared535.0439843999998/Original3336.9150391004946이 남는다. 기존 J의 전체 qualification575는 맞지 않는다. ROOT는 유용한 검증 뒤 Generated+400 / Original-400, 즉 global2700/local1964/Original15200을 제안한다. 이 400은 ROOT의 전향적 배분 판단이며 별도 인간 수치 승인이 아니다. 실제 비용·baseline·offset·실패·기존 epoch 이벤트는 모두 보존한다.

## 변경 불가 조건

F1–F8, startup99 및 startup q/tendon 오차 상한, 짧은1.5초 gate, 후보 alpha0.0001/index4/theta·selected hash, 원래 unused F6, M2/UI를 변경하지 않는다. 외부 CPG·강제 위상·teacher 출력·root 이동을 추가하지 않는다. Original 입력 payload/NPZ/header/stat/hash/array/GPU/import는 실제 Original admission 전 읽지 않는다. 모든 known failure 비용은 단 한 번 유지하며 unknown 비용이나 종료를 0으로 처리하지 않는다. 기존 J/G/H/I Source·결과·freeze·descriptor·cost·ledger anchor를 덮어쓰거나 성공으로 재명명하지 않는다.

현재 실행 guard 및 accounting의 내부 구조는 바꿀 수 있다. 역사 Source와 모든 기존 assertion을 보존하되, 실행마다 재귀 감사하는 방식은 고정 요구 사항이 아니다. 과거 전체 감사와 현재 실행 적합성을 구분하는 새로운 체크포인트는 기존 한도 내 구현 변경이다.

## 구현 구조

1. 현행 닫힌 ledger288을 Source에 고정된 pin으로 인증하는 checkpoint runtime을 만든다. 이 anchor의 정확한 raw bytes와 duplicate/type-safe JSON을 매 invocation 인증하고, 기준 이전 immutable 필드·debit prefix·paid call mapping·policy·모든 과거 allocation/amendment pointer·실패를 live ledger와 정확히 대조한다. 다음 꼬리만 전향적인 기존 transition 규칙과 ordered binary64 합으로 검증한다. key-set, typed finite numbers, bool-as-number, 중복 call/cost, cap 혼합, 비인가 pointer/hold/owner 변화를 거부한다. anchor 값은 미래의 실행 성공이나 전체 과거 Source 재실행을 주장하지 않는다. ROOT 실제 J 결과와 온전한 기존 provenance는 보존되며 cold audit 경로에서 여전히 읽을 수 있다.

2. tools/banc_m3_checkpoint_runtime.py와 tools/banc_m3_checkpoint_runtime_native.ps1(각 최대65536바이트)을 대칭적으로 통합한다. Native/Python 현재 caps, request/admission, locked dispatch, begin/closed/unknown, settlement, qualification, Original routing이 같은 체크포인트 계약을 쓴다. K key가 없는 역사 ledger는 기존 경로를 유지한다. K key가 있는 ledger는 과거 j_history/Conserved-J-History 재귀 검증에 되돌아가거나 새로운 family wrapper로 그 재귀를 복제하지 않는다. K 이후 과거 call ID는 재실행·재등록할 수 없다.

3. account.ps1의 262144바이트 상한을 해결한다. 현재 whole core14933과 whole J epoch126162가 inline이며 local 부분121049이다. J ledger288 이후 미결 요청이 없다는 고정 증거를 전제로, 현재 K fee/closure engine을 account 안에 완전히 inline 보존한다. 기존 epoch Source는 canonical 파일을 그대로 유지하고, pin을 인증한 후 역사 진단/기존 함수 호출에서만 lazy load한다. 실제 K commit의 fee·identity·closure는 외부 epoch/helper Source가 바뀌어도 중단되거나 성공으로 간주되지 않아야 한다. 외부 파일에서 K accounting 자체를 dot-source하는 방식, encoded executable, 무검증 eval/Invoke-Expression은 금지한다. public legacy API와 old fixture 동작을 보존하고 무엇이 lazy인지 명시한다. core는 변경하지 않는다. 새 executable 파일은 모두 현재 freeze 및 lint에 포함한다. account 및 개별 Source/fixture/freeze 상한262144를 유지한다. 상한에 맞추기 위한 기존 assertion 삭제는 금지한다.

4. 현재 Source vector는 기존 J728 전체 identity를 보존하고 바뀐 실행 파일만 전향적으로 replacement로 표시하며 새 실행 파일·새 tests·static K metadata를 명시적으로 추가한다. Source에 새로운 dependency/import가 생기면 freeze의 완전한 closure에 넣는다. historical ledger anchor는 bounded ledger proof(524288)의 대상이며 executable Source cap으로 위장하지 않는다. request/freeze/전체 Source는 actual SharedLock 안에서 실제 worker 생성 전에 검증한다. current_caps와 begin/closed는 authenticate된 anchor/ledger만 검증하고 전체 과거 Source/vector를 다시 재귀 해석하지 않는다. 전역 캐시나 이전 invocation의 PASS 재사용은 금지한다. invocation-local pin reuse는 동일 raw snapshot 안에서만 가능하고 lock 획득 뒤 fresh bytes 검사를 대신하지 않는다. Source가 current 역할 실행 전에 달라지면 거부하지만 이미 종료한 known call의 fee settlement는 보존한다. 자기 자신의 미래 freeze/manifest/cost hash를 Source에 넣지 않는다.

5. 기존 SharedLock와 Native transaction exclusion, 실제 owner PID+creation ticks, Job/suspended spawn/assign/resume, 같은 ENTRY child/outer/cleanup clocks, 두 EOF, atomic CreateNew/fsync/replace/full readback, 첫 오류와 secondary, unknown hold를 보존한다. process-wide 실행 메모이제이션, foreign token, 검사 직전 deadline 재설정은 금지한다. perf_counter 기반 실제 phase elapsed를 outer에 선택적으로 기록한다: locked validation, spawn, begin, resume 여부, worker interval, close. 이것은 genuine collector의 dispatch→first FINAL 비용을 대체하지 않는다. 실패한 pre-resume는 worker_started=false로 분명하게 남긴다. 새 telemetry 필드의 Native/Python typed schema를 맞춘다.

## 복구와 실행의 단일 경로

A. ROOT Source 설계 → 실제 gpt-6-astra/max 리뷰 → ROOT 반영 → 실제 gpt-6.1-sol 구현(난이도상 max) → ROOT FIRST. 필요하면 같은 Sol의 sole R1 → ROOT SECOND → 남은 수정은 ROOT. 구현은 sibling proposals만 쓴다. Source-only 구간에서 Python/import/AST/compile/lint/pytest/Native 실행/프로세스/CUDA/Original probes는 금지한다. PS/.NET의 bounded bytes/hash/text/JSON 읽기와 literal edits만 가능하다. reviewer/implementer는 canonical ledger/Source/plan을 수정하지 않는다.

B. ROOT가 실제 Source를 검토·adopt하고 immutable K reviewed Source/freeze/static descriptor를 게시한 뒤, 전용 Native checkpoint-recovery transaction을 한 번 실행한다. 이 transaction은 fresh locked live raw가 정확한 settled288 anchor이고 모든 process/owner/reservation/pending이 없으며 유일한 J hold가 위 known-closed fully-paid per-call overrun임을 확인한다. J cost·genuine return·outer의 고정 pin과 typed cost/owner/EOF proof를 읽고, 이미 낸48.607 및 cap_exceeded=true를 유지한다. 변경은 K recovery pointer 추가와 그 정확한 J hold 제거뿐이다. cap·남은 자원·cost·usage·old history는 변하지 않는다. recovery descriptor는 해제 이유와 J 실패를 영구적으로 보존한다. 다른 hold·unknown closure·전체 budget breach·외국 lock은 모두 거부한다. Source review가 경험적 bootstrap을 통과했다는 주장으로 쓰이지 않는다.

C. 기존 남은 Generated 한도로 단일 실제 K bootstrap을 수행한다. fixed generated-python role, fresh call/output, outer90/child75/cleanup15, host2GiB/output8MiB, old global2300/local1564/Original15600. ROOT가 완전히 선언한 production fixture node(s)가 실제 Native/Python checkpoint equality, exact delta/recovery rejection, same lock/atomic operation, Source drift와 paid settlement, stale bytes, duplicate/mixed caps, invalid grant, unknown owner/EOF를 검사한다. 실제 live ledger를 synthetic test가 직접 수정해서는 안 된다. bootstrap fixture는 reviewed production primitive를 호출하고 fabricated PASS receipt를 쓰지 않는다. subset는 bootstrap이며 full qualification이 아니다. 전용 once-charge collector는 unchanged이며 실제 admission과 독립 readback 뒤에만 호출한다. 전체 genuine return을 원래 cap의 ledger에 단 한 번 청구하고 known closure와 no hold 확인 뒤에만 다음 단계로 간다. 실패하면 fee/hold를 보존하고 코드 또는 설계를 고친다. 같은 실패 실행을 근거 없이 반복하지 않는다.

D. bootstrap이 실제로 통과하고 한 번 청구된 fresh closed tail을 기준으로, Source에 정의된 cap-allocation transaction을 실시한다. runtime event는 bootstrap request/trace/outer/cost/JUnit, actual fresh before ledger, adopted ROOT/Astra/Sol Source provenance를 정확히 결속한다. 미래 hash는 사전에 만들어 넣지 않는다. 변경은 new K allocation pointer, Generated2700/local1964/Original15200 및 두 remaining 값뿐이다. 기존 모든 cap/transfer event와 failure를 유지하며 합계17900·actual usage가 동일하다. 전향적인 bootstrap이 자기 validation 자금을 쓰지 않았음을 확인한다. 앞선 HOLD 해제는 자원 증가가 아니다.

E. K qualification은 기존 J의 모든 lint targets와 19/21/32 전체 고정 roster를 보존하고, 새 production fixture/assertions의 정확한 node identities를 추가한다. 기존 math·gate 사례/parameter group을 삭제하거나 free probe/collect-only를 PASS로 쓰지 않는다. 별도 immutable K roster에서 math/gate A/gate B의 정확한 disjoint union과 모든 Native 관련 indivisible groups를 정의한다. 기존 E roster는 그대로 둔다. lint40/35/5, math35/30/5, gate-A120/110/10, gate-B360/340/20, Generated close20(총575)을 유지한다. 첫 call 전 두 pool의 전체575가 맞아야 한다. bootstrap과 모든 K full costs는 별개이며 bootstrap 결과는 qualification cost로 재사용하지 않는다. 각 다음 역할 전 실제 exit0/known closed/paid actual pin/JUnit full identity와 exact current Source를 확인한다. missing/duplicate/skipped/error/failed node, timeout, changed Source 또는 unknown fee/EOF는 Original을 막는다. tests는 gait 증거가 아니다.

F. 네 qualification 비용과 전체3 JUnit이 통과하고 fresh closed resource가 충분할 때, 동일 후보-k의 실제 body300/285/internal270/cleanup15를 딱 한 번 수행한다. genuine exit0 및 run_valid=true일 때만 audit120/100/internal90/cleanup20(Original close180)을 허용한다. actual request/freeze/ledger snapshot을 새 K literal ID로 결속하며 기존 J ID를 재사용하지 않는다. 후보 입력·Original prepare stage8 journal·Source worker routing을 기존 과학 계약 그대로 사용한다. 실제 preparation/neural/physics가 미확정이면 UNKNOWN으로 기록하고, body가 실행되었어도 짧은 gate나 F1–F8를 못 넘으면 채택하지 않는다. 여기까지 반환하면 신경망/제어 후보 개선의 다음 결정으로 돌아간다; proof infrastructure 자체를 완료 목표로 바꾸지 않는다.

## 판정, 자원과 남은 불확실성

설계의 가설은 “immutable288 checkpoint와 현재 Source 검증의 분리로 recursive preflight와 begin 경로가 줄어들고 실제 worker가 cap 안에서 resume한다”이다. 경쟁 설명은 Native process startup/전체 gate fixture 비용 자체가 여전히 주 병목이라는 것이다. J의 genuine48.607와 worker NOT_STARTED를 baseline으로 유지한다. K bootstrap과 다음 whole lint의 실제 phase/worker_started/genuine 비용으로 판별하고, 여러 변경을 결합했으므로 함수별 인과 효과는 주장하지 않는다. 적합한 run은 실제 owned termination+두 EOF+once paid+fresh Source/JUnit proof를 갖춘다. elapsed 감소만으로 transaction correctness나 자연 보행을 PASS 처리하지 않는다.

기존527.256 직접 자원은 bootstrap90과 close20을 감당한다. +400 전향적 배분 후 bootstrap가 최대90을 썼더라도 direct 최소837.256/shared845.0439844로 whole575와 close가 맞는다. Original은2936.9150391004946이 남아 body300+audit120+close180이 맞는다. 이들은 예약 가능성 계산이며 실행 시간 보장은 아니다. actual overrun/unknown closure에는 기존 HOLD 규칙을 유지한다. K 실패가 있으면 현재 전체 총액 안에서 다음 근거 있는 수정을 판단한다. 자동 반복·새 budget grant·비용 환불·cap reset은 없다.

핵심 위험은 checkpoint에 과거 검증 권한을 과도하게 부여하는 것, K fee settlement의 외부 Source drift 의존, lazy legacy API 회귀, Source closure 누락, typed parity 불일치, 부모/worker timing 부정확성, 새 tests 또는 freeze cap 초과다. Astra는 이 경계와 production end-to-end path를 먼저 검토한다. Sol은 concrete changed files, full Source diff, old assertion/legacy API preservation, hard caps, static exact roster, actual execution NOT_RUN과 미해결 항목을 보고한다. ROOT는 답변/파일 존재를 통과 증거로 삼지 않고 실제 수정 Source를 읽는다.
