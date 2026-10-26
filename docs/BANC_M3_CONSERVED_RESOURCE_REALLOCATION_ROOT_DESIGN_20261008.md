# M3 검증 재개를 위한 미사용 자원 재배분 — ROOT 설계

2026-10-08. 이전 goal turn은 검토된 입력 준비 timing Source3개와 실제 리뷰/수정 cycle을 남겼으므로 progress였다. 그러나 실제 gait와 full qualification은 미완료다. 이번 목적은 그 Source와 원래 전체 검증을 실제로 확인하고 실제 whole-BANC body를 다시 평가하는 경로를 여는 것이다. Source-only 문서나 작은 fixture 성공을 F1–F8의 대체 결과로 삼지 않는다.

## ROOT 판단과 권한

최신 직접 사용자 지시 `앞으로 내 승인 받지 말고 스스로 판단해서 진행해줘. 개입할 상황이면 내가 알아서 개입할게`에 따라 ROOT가 실행 준비와 미사용 local compute 배분을 판단한다. 사용자가600초라는 숫자를 승인했다고 주장하지 않는다. 과거 승인/예산 정책은 역사적 계약이며 덮어쓰지 않는다. 새 금융 지출이나 외부 서비스/배포는 없다. Root GOAL_PLAN, M2/UI, unused F6, BANC provenance, 신경 단독 구동과 F1–F8을 유지한다.

현재 별도 cap이 전체 검증을 차단한다. 그 cap을 묵묵히 넘는 대신 **미사용 Original 한도600초를 Generated 검증 한도로 prospective 전용**하는 계약을 검토한다. 이는 원본 데이터 작업을 generated라고 부르거나 generated 비용을 Original로 청구하는 대체가 아니다. 각 실제 call의 mode/실제 첫 FINAL 비용 분류는 그대로다. Combined cap17900초를 늘리거나 비용을 지우지 않는다. Astra가 권한 또는 conservation 조건이 부족하다고 판단하면 그 부분을 채택하지 않고 실행을 보류한다. 이 설계만으로 예산을 변경하거나 Source를 적용하지 않는다.

## 현재 기준과 prospective 값

Canonical ledger는 `verification/banc-natural-gait-m3-20260919/parent-e358-execution-ledger.json`, closed copy는 `parent-native-count-conformance-closed282-20261008-a.json`이다. 두 개는453918B/SHA256 405ffe995f1241f2ff3263efc916c9e69897b18c37028da3a9aab82c6d37f661,282 debits/164 calls다. Generated shared used1660.4940155999998/cap1700/remaining39.505984400000216, direct generated used932.2819999999998/local cap964/unused31.71800000000019다. Original used12263.084960899505/global16200/remaining3936.9150391004946, stage6300/science local6000이다. Close reserve20는 그대로다. 모든 worker/reservation/pending/hold는0이다.

| 한도 | 현재 | 단일 재배분 뒤 |
|---|---:|---:|
| Generated shared global |1700|2300|
| Direct Generated local |964|1564|
| Original global |16200|15600|
| Original stage/science local |6300/6000|6300/6000, 더 낮은 global15600과 함께 제한|
| 두 global cap의 합 |17900|17900|
| 기존 누적 used·debits·calls·실패·기존 allocation 사건 |현재 값|정확히 불변|

부동소수점 remaining은 기존 Native writer의 ordered actual-cost 산술과 동일한 규칙으로 새 cap에 대해 계산한다. 전/후 floating remaining의 단순 덧셈 bit 일치를 conservation의 정의로 사용하지 않는다. Integer cap sum과 모든 실제 cost/used prefix 불변으로 검증한다. 미래 Actual은 계속 once debit이며 UNKNOWN return은0이 아니라 hold다.

새 allocation 사건은 한 번만 등록하고 immutable closed282 anchor, 기존 policy·모든 transfer·J grant, literal 최신 인간 지시와 ROOT의600초 판단, 구/신 cap, delta, sum, schema/version을 고정한다. Cap delta는+600/+600/−600이며 추가 global resource0이다. 기존 close를 빌리거나 반복 가능한 allocation 명령을 만들지 않는다. Historical grants/cost/source/qualification를 수정하지 않는다. Old source가 새 Original cap을 이해하지 못하면 fail closed한다.

## 방법과 위험

기존+4 등의 Generated local 전용은 같은1700 안에서 이루어졌으나 그 shared 잔여 자체가 이제 부족하다. 새 이름의 작은 Count probe나14/30 subset은 의도한 전체 검증을 열지 못한다. 측정되지 않은 Source/이력 캐시나 fence 생략도 채택하지 않는다. Conserved600 재배분 + reviewed preparation timing + 현재 실제 이력으로 고정한 후속 body 계약을 선택한다.

주요 위험은 cap 두 개를 따로 써서 한쪽만 변경하는 것, 새로운 writer를 이용한 재청구/재설정, 과거 grant/global16200를 현재15600과 혼동하는 것, Generated supervisor만 새 정책을 받아들이고 Native admit/commit이 서로 다른 계산을 쓰는 것, stale d280/162를 재사용하는 것, accounting 변경의 bootstrap 검증 순환이다. 각각에 대해 fail-closed 동작·고정 현재 anchor·Native/Python parity·외부 typed owner·부록의 실행 단계 및 검증을 요구한다.

## 최소 구현 범위

ROOT 설계→actual GPT-6 Astra/max review→ROOT 반영→actual gpt-6.1-sol/max FIRST→ROOT FIRST→필요 시 동일 Sol sole R1→ROOT SECOND→ROOT residual을 따른다. 현재 source contract들에는964/1700/16200와 old d-history가 상수로 박혀 있다. 임의 broad literal replacement를 금지한다.

1. Native accounting/administration과 Python supervisor에 한 번의 prospective allocation event를 검증하는 경로를 추가한다. 현재 ledger anchor와 모든 사용량/구계약/pending/owned closure를 완전 확인하고, 같은 lock 아래 cap·remaining·event를 한 번에 commit한다. Unknown/torn write에는 재시도·복구라고 주장하지 않고 HOLD한다. 실제 배분 commit은 메타데이터 작업이며 과학적 실행 비용0이라는 뜻으로 전체 준비 비용을0으로 기록하지 않는다.
2. 기존 initial policy, transfer300→380→460→600→900→960→964, source-qualified history와 ledger actual debit prefix를 유지한다. 새1564는 이 신규 event를 검증한 경우에만 허용하고,2300 Generated/15600 Original 두 current cap을 함께 검증한다. 기존 snapshot을 새 cap으로 재해석하지 않는다. Historical usage는 그대로, future admission/commit/current budget은 prospective caps다.
3. 이전 ROOT SECOND 완료 timing3개 Source의 Source edits를 integration proposal에 포함한다. Startup 기존문장/검사/콜백 순서와 journal event64/active14/emit16/wire32KiB/stream512KiB/science270/worker285/flush1/terminal을 유지한다. 기존 Source 전체/테스트 assertions를 보존한다.
4. 모든 기존 full math14/gate30와 신규 timing fixtures, 필요한 resource/owner/conservation fixtures를 포함하는 **전체 qualification**을 명시적 새 roster/JUnit evidence로 확인한다. 필터로 새/실패 테스트를 빼지 않는다. Counts/names와 Native/Python 실현 variant를 actual execution과 맞춘다. Source equivalence waiver나 테스트 성공을 body 증거로 쓰지 않는다.
5. 실제 ledger tail에 gate68 failed timeout, old Native probe1 failed, Count conformance1 Native-not-started와 원래 all costs를 포함하는 acyclic fresh e-body/e-audit 계약을 작성한다. Old d descriptor는 보존하고 stale로 유지한다. Future e-body before ledger는 새 full qualification의 실제3비용이 once charged된 뒤의 exact prefix로 만들어야 한다. 未観測 future cost/pin/count를 지금 꾸미지 않는다. Candidate-k/inputs/raw gates/startup99/1.5s body/health/arm initializer/전체BANC는 그대로다. Journal을 새 e-body에 현재 owner origin으로 연결하는 데 필요한 mapping만 바꾼다.

변경 가능한 정확한 파일 목록은 raw Source 조사 후 Astra/ROOT 반영안에서 확정한다. Delegates는 canonical current bytes에 근거한 sibling proposal에만 쓰고 canonical/ledger/active plans를 수정하지 않는다. 추가 subagent/subprocess/Python import/AST/compile/pytest/Ruff/Native/CUDA/Original payload/header/hash 금지다. Source·metadata 텍스트/해시 읽기와 지정 sibling 결과의 CreateNew/fsync/full byte readback만 허용한다.

## Bootstrap와 실제 실행 순서

재배분 writer는 과학적 검증을 우회하는 권한이 없다. ROOT는 actual source review cycle과 unchanged closed282 current metadata·literal 인간 위임을 확인한 뒤, 기존 administrative Native lock/write 패턴을 따른 유한 allocation commit만 할 수 있다. 그 writer에 대한 full empirical 검증을 commit 전 현재 한도에서 수행할 수 없다면 이 검증 순환·한계를 명시하고 Astra가 안전한 bootstrap 조건 또는 차단 verdict를 결정하게 한다. 기존 Source qualification을 요구하는 Original 역할은 끝까지 차단한다.

조건을 만족한 뒤 ROOT가 canonical Source를 구체적으로 적용하고 신규 Source freeze를 만든다. 한 번의 full lint outer15/child10/cleanup5, full math outer25/child20/cleanup5, full execution gate outer280/child270/cleanup10과 close20를 reserve한다. 총 envelope340은 재배분 뒤 local unused 약631.718/shared remaining 약639.506 안에 들어간다. Host/output/실제 Job/ENTRY deadline과 separate current Native admit→기존 genuine collector→first FINAL fee once commit은 기존 안전 계약을 유지한다. 수치340은 실행 성공/시간의 예측이 아니라 hard maximum reserve다.

어느 gate든 실패/timeout/unknown이면 그 raw 결과/clock/cost/Source/owned closure를 보존하고 body를 실행하지 않는다. 자동 retry0, Source 변경 없는 같은 gate 이름 변경 재실행0이다. 실제 실패가 변경한 근거로 수정을 설계할 경우 별도 reviewed Source/current reserve가 필요하다. Canonical integration은 reversible development지만 qualification은 성공으로 소급하지 않는다.

전체 qualification이 실제 통과하고 모든 비용/ownership이 known closed이면 current fresh e-body envelope outer300/child285/internal270/cleanup15과 조건부 audit outer15/child10/cleanup5만 설정한다. Latest frozen source/current ledger/independent saved candidate provenance에서 실행한다. Whole-BANC→motor→actual FlyBody99 startup+1.5s의 fixed gates를 검증하고 단계 시간이 획득되면 그 증거로 다음 최적화/후속10s F3–F5 여부를 정한다. Body를 테스트나 prepared manifest로 대체하지 않는다. 실제 body timeout이면 acquired8슬롯/active/global/first-error/work만 해석하고 없는 단계를UNKNOWN으로 남긴다. F6/F7/F8로 진행하는 것은 original finite-goal의 요구를 유지하며 앞선 실물 gate를 통과한 경우에만 선택한다.

## 사전 질문과 판정

Hypothesis: 검증 한도만이 현재 실제 whole-body 평가 재개의 선행 blocker이며, conservative prospective 재배분·full Source qualification·fresh actual history가 그 경로를 연다. Competing explanation: 새 accounting bootstrap 또는 Native history validation/whole suite가 실패하거나, 실제 입력 준비/신체 실행 자체가 허용270초에 완료되지 않는다.

분리 판정: allocation validity/Source qualification/runtime body scientific verdict/gait milestone을 각각 판단한다. Cost or setup 실패는 gait failure나 Count 의미 반증이 아니다. Qualified actual short body가 모든 fixed gates를 만족할 때만 동일 후보의10s 평가로 이어간다. 최종 goal은 모든 F1–F8 실제 충족 전까지 ACTIVE/INCOMPLETE, candidate-k UNADOPTED/biology UNPROVEN/native VJP UNVERIFIED다.

## 근거 파일

- Current ledger와 immutable closed282 pins는 위와 같고 live owners/holds는 ROOT가 다시 확인한다.
- `docs/BANC_M3_BODY_PREPARATION_TIMING_ROOT_SECOND_REVIEW_20261008.md`:5109B/SHA256 bbb241c42b9b6c9cbaa8770300e18fbe12ed16cf8356b930fcd16edb44972356.
- `verification/banc-natural-gait-m3-20260919/parent-body-preparation-timing-sol-R1-20261008`의3 source와 R1.md/json를 읽고 준비 instrumentation을 재사용한다.
- `tools/banc_m3_startup_ray_supervisor.py`、`tools/banc_m3_contact_learning_admission.py`、`tools/verify_c_banc_m3_contact_learning.py`、Native `parent-direct-science-account.ps1`、`parent-direct-science-prepare-native.ps1`의 현재 raw code를 읽는다.
- Initial generated policy, latest residual4 transfer, direct human autonomy metadata와 J historical grant는 Source/metadata로만 읽고 actual 사용자 지시를 대신하는 authority로 쓰지 않는다.

새로운 cap/ledger/Source를 아직 적용하지 않았다. 이번 설계의 확인과 실행에는 Root설계→Astra actual review→ROOT 반영과 구현/리뷰가 필요하다. 이미 실패한 conformance/qualification을 취소하거나 삭제하지 않는다.
