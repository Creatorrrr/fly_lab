# Native 검증 비용 개선 설계 — 2026-10-08

목적은 동일한 BANC 후보의 실제 몸체 실행에 도달하는 것이다. 현재 검증 코드가 큰 이력 구조를 반복 대조하며 시간을 소비하므로, 검증 내용을 유지하면서 이 계산의 비용을 줄이는 방향을 검토한다. 이 문서는 ROOT 설계이며 실제 Astra 리뷰, 구현 또는 성능 통과를 뜻하지 않는다. 사용자의 자율 진행 지시를 적용하고 추가 승인 질문을 만들지 않는다.

## 확인한 상태

Runtime 단계는 실제 Astra/max 리뷰 → ROOT 반영 → GPT-6.1 Sol/max FIRST → ROOT FIRST → 동일 Sol sole R1 → ROOT SECOND → ROOT 잔여 수정 순서를 마쳤다. Source-z는 196158 bytes, SHA256 `5145ff896b6ef416f1437e6390a5912a98ee54a28e7f900c0c0659116794c40d`다. Actual lint66은 exit0/2.222초, math67은 14 PASS/exit0/5.81초다. Gate68은 full return88.609초/exit124, child88초 timeout, known-ended/양쪽 EOF/once cost였다. Stdout24 bytes는 완료 표시24개이며 30개 전체 JUnit이나 qualification은 없다. 이것을 24개 전체 합격, 보행 실패 또는 생성 검증 통과로 승격하지 않는다. Source 선언 순서상 다음 d-history 그룹이 병목 후보이나 실제 세부 단계는 UNKNOWN이다.

실제 결과는 `verification/banc-natural-gait-m3-20260919/parent-contact-runtime-ROOT-actual-result-20261008-z.json` 5718 bytes/SHA `6cfc6f355472286c23af34beb5ae99d477adfe4305a51e291129fbc0664adffb`에 보존했다. Executed-z16 archive index는 5342 bytes/SHA `8d8658e942e889f8c460de2b950991fd731b264587a6dc2606d7e0c1ec7eadad`다. Closed280/162 snapshot은 452155 bytes/SHA `583cea595792cb65df694514667e95befceeec5cebd91c06ad5df5c25b3e5212`다. d-body와 d-audit는 아직 실행하지 않았다. c-body의 원래 deadline 실패와 work UNKNOWN은 그대로 남는다.

같은 프로젝트의 실제 Source-x gate62 JUnit에서 `test_native_transfer_conservation_negatives_and_actual_limiter_once`는18.313초, `test_gate_retry_python_native_history21_22_and_strict_negatives`는22.381초였다. 이는 CPU의 Native 검증 경로가 실질 비용을 차지한 직접 증거다. 새로운 z의 세부 시간 또는 c의 CUDA/물리 병목 증거는 아니다. 현재 Native `Same`은 dictionary/array의 모든 요소에 PowerShell 함수를 재귀 호출하고 scalar마다 `Need`를 호출한다. `Generated-Budget`은 이전 배정 snapshot을 각 단계에서 다시 재귀 검증한다. 비용 원인이 typed 대조, JSON read/hash, 재귀 반복 중 어느 쪽인지는 아직 분리 측정하지 않았다.

## ROOT가 추천하는 좁은 변경

우선 Native 메타데이터의 typed 대조 계산을 개선한다. Read-Pin의 containment, reparse 금지, cap, byte/hash/readback/mtime 검증과 모든 실제 prefix/call/transfer 검증은 그대로 유지한다. 데이터나 검증 결과를 process 사이에서 캐시하지 않는다. 자원 규칙, Source pin, 후보, 호출 이력 또는 first-error를 검사에서 빼지 않는다.

구현 후보는 JSON에 사용되는 타입만 다루는 bounded compiled comparator다. 비교 대상과 반환값만 받으며 파일, 배열 payload, solver, CUDA, 시간 원점, ledger mutation에 접근하지 않는다. dictionary의 key 순서는 무시하되 key 집합과 exact key semantics를 유지하고 array 순서와 scalar runtime type/값을 정확히 대조한다. Bool과 정수, Int64와 Double, null과 빈 문자열, nested field 누락을 혼동하면 안 된다. 기존 PowerShell comparator는 부정 결과와 지원하지 않는 타입의 진단 경로로 유지한다. Compile 실패나 지원 범위 밖의 객체가 발생하면 원래 strict 경로로 돌아가며 성공을 만들어내지 않는다. 큰 이력 대조에서만 lazy compile하여 작은 counter 검사와 단순 refusal에 추가 비용을 만들지 않는 방법을 먼저 검토한다.

성능 가설은 동일한 metadata/case/assertions에서 interpreted 함수 호출 비용이 감소한다는 것이다. 경쟁 설명은 대부분의 시간이 Read-Pin과 반복된 전체 validation에 있어 comparator 변경만으로는 충분하지 않다는 것이다. 어떤 설명이 맞아도 실제 측정 전 속도 향상률이나 전체 검사 통과를 발표하지 않는다. 안정적인 immutable-file lease 없이 history proof cache를 도입하지 않는다. 반복 budget 검증의 구조적 중복 제거는 별도 설계 분기이며 이번 comparator 변경에 섞지 않는다.

현재 변경 범위 후보는 Native account의 Same helper와 그 helper를 의미 있게 검증하는 기존 scratch Native test 그룹이다. Prepare의 다른 comparator를 바꿀 필요가 있다면 실제 scope를 명시하여 리뷰받는다. 과거 기록의 함수나 snapshot을 편집하지 않고 새 proposal과 새 Source로 보존한다. 전체 Runtime14/30 names/classes/old11/24 assertions, owner9/metadata3, native counter type checks와 d-history 및 실패 경계 검사를 유지한다. 같은 case 내부의 추가 typed comparator assert는 가능하며 축소나 skip은 없다.

## 자원과 실행 판단

현재 generated used921.5029999999998/local964, local unused42.497000000000184, shared remaining50.28498440000021/1700, minimum close20다. Original12263.084960899505/16200/remaining3936.9150391004946와 모든 owners/pending/holds/reservations0를 확인했다. 기존 full8+12+96+close20=136은 남은 generated 자원에 들어가지 않는다. 이 문서로 새 grant, reset, close 여유 차용, Original→generated 대체, 짧은 subset qualification 또는 Source-equivalence 예외를 허용하지 않는다.

따라서 현재는 Source-only 설계/리뷰/검토 가능한 구현 proposal까지 진행할 수 있다. 새 보호 검사를 dispatch하려면 해당 체크의 전체 envelope 및 종료 여유가 실제 남은 한도에 들어가고, 전체 mandatory qualification과 단일 d-body/조건부 감사까지의 실행 계약이 구체적으로 성립해야 한다. 남은 한도에 맞추려고 아직 측정하지 않은 full30 검사에 낙관적인 짧은 timeout을 붙이지 않는다. 코드 제안, 검토 결과, 과거 PASS는 새 Source qualification을 대신하지 않는다. 자원 판단은 ROOT가 맡으며 사용자에게 반복 승인 질문을 하지 않는다.

실행 가능한 다음 bounded 진단이 설계된다면 old/candidate 동일 Native cases와 고정 fixture에 대해 genuine collector/별도 admission/first FINAL/once cost를 사용한다. 단일 comparator benchmark는 Native 개선의 진단일 뿐 전체14/30 또는 보행 증거가 아니며, 다음 fresh Source history에 실제 비용과 성공/실패를 모두 포함해야 한다. Unknown return은 hold/no replay다. 선행 실패의 gate68/c 비용은 수정하거나 재사용하지 않는다.

## 개발 및 수용 경계

ROOT 설계 → 실제 GPT-6 Astra/max 리뷰 → ROOT 반영 → 실제 GPT-6.1 Sol의 적절한 effort FIRST → ROOT FIRST → 필요 시 동일 Sol sole R1 → ROOT SECOND → ROOT 잔여의 사용자 지정 순서를 적용한다. 이번 FIRST는 Source-only proposal이며 canonical/ledger/역사/Original를 직접 쓰거나 검사하지 않는다. Native equality의 semantic coverage와 scope를 ROOT가 우선 리뷰하고 실제 protected 검증 여부는 자원 계약에 따라 판단한다.

F1–F8 ACTIVE/INCOMPLETE, candidate-k unadopted, biology UNPROVEN, exact native VJP UNVERIFIED다. M2/UI/unused F6/root GOAL_PLAN151552 bytes/SHA `16898e461fede082206f9584c60b3fb11d994a7ab708cdfb19f950e35126eace`, 모든 BANC provenance, 외부 CPG/교사/위상/궤적/root 교정 금지, startup 네 기준과 short 기준을 보존한다. Native 검사 비용 개선은 실제 자연 보행 수용 판정이 아니다.
