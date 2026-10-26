# E360 P9: 승인된 생성 검증 복구안의 자원 권한 통합 설계

2026-10-04 ROOT 설계. 사용자의 직접 승인 "추가 배정 승인할게"는 최종 제안 `docs/BANC_M3_E360_SUITE_RECOVERY_FINAL_PROPOSAL_20261004.md`의 추가 생성 검증 800초를 승인한다. 공유 600→1400초, E360 지역 489→1289초로 늘리고 81개 과거 차감, 사용량, 원본 한도 및 실패를 보존한다. 기존 goal-planner finite-goal과 F1–F8은 ACTIVE/미완료다. 이번 단계는 실제 생성 검증의 복구이며 자연 보행이나 원본 실행의 증거가 아니다.

## 현재 증거와 변경 가설

P8 생성 fixture는 실제 전체 반환 26.753초에 성공했지만 전체 suite는 120.053초에 실패로 닫혔다. 기존 suite 소유 자식은 110.03245170001173초에 종료되었고 stdout은 `..................F...`뿐이다. 19번째 기대값과 반복 metadata 검사 비용은 소스 기반 가설이며 실제 traceback이나 개별 시간으로 확인되지 않았다. 과거 총 81개 차감, 생성 공유 502.45501559999997초, E360 지역 391.498초는 그대로다. 보류/예약/미차감 작업은 없다.

ROOT가 작성하고 실제 Astra6/max, 실제 Sol6.1/high FIRST 및 같은 Sol 한 번의 R1, ROOT SECOND/N1/N2로 리뷰한 최종 패치를 구현의 기반으로 사용한다. 최종 source proposal `parent-e360-suite-recovery-root-final-source-proposal.json` (14210 bytes, SHA256 12676978e80e0592a6f63f626728ff4cddc7812ed4181bd1ece523c91fa972c4) 및 최종 tool/tests (248335/172276 bytes, SHA256 c6d14ebda7b1dace691fd2ac5f7f0646139fac3e99cdbe6105e57f574f90c859 / ccd46f60657d35d6ea33c224ff28ecff910a303e3c699884f4b526026f9c0e93)를 보존한다. 현재 활성 소스는 C9이므로 최종 제안 tool/tests를 먼저 기반으로 적용한 뒤 P9 통합을 더한다. account/helper는 C9와 같은 최종 제안 사본을 기반으로 한다.

고정된 가설: 실제 ENTRY/clock, cold live/source validation 및 fresh durable publication fence를 유지하면서 승인된 추가 자원과 P9를 코드 전체에서 일관되게 선택하면, 새 전체 fixture→suite→Q 체인이 과거 권한과 비용을 손상하지 않고 판정 가능한 실제 실패 또는 성공으로 닫힐 수 있다. suite 600초 내 완료 가능성은 아직 미검증이다. 실패하면 관측된 첫 실패와 실제 full-return 비용을 보존하고 P10/자동 재시도를 하지 않는다.

## 순서와 소유권

ROOT 설계 → 실제 GPT-6 Astra/max 읽기 전용 리뷰 → ROOT 반영 → 새 불변 승인/P9 literal → 실제 GPT-6.1 Sol/high NEW FIRST 구현 → ROOT FIRST → 필요하면 같은 Sol ONE R1 → ROOT SECOND → 남은 수정 ROOT만 → 최종 소스 freeze/자원 활성화 → 승인된 실제 생성 9-call 체인. 이전 suite patch의 R1과 이번 새 통합 범위의 R1을 혼동하지 않는다. Sol R2는 없다. Sol은 다음 네 소스만 편집하고 ledger/literal/문서/원본/다른 프로젝트 소스는 변경하지 않는다.

- `tools/verify_c_banc_m3_source_closure_diagnostic.py`
- `tests/test_c_banc_source_closure_diagnostic.py`
- `verification/banc-natural-gait-m3-20260919/parent-e360-root-account.ps1`
- `verification/banc-natural-gait-m3-20260919/parent-e360-root-metadata-helper.py`

모든 Python/import/AST/compile/Ruff/pytest/helper/worker/probe는 사전 admission 및 실제 전체 반환 한 번의 차감이 필요하다. 설계/리뷰/구현 동안 실행하지 않는다. native PowerShell 텍스트/해시/구문 분석, Git diff/check는 가능하다. 322개 protected text 및 다른 own5는 그대로 유지한다. 원본 NPZ/NPY/header/hash/value IO 금지.

## 추가 승인과 불변 P9 literal

ROOT는 새 `parent-e360-suite-recovery-human-authorization.json`에 실제 직접 승인 문구, final proposal MD(7823 bytes, SHA a638672b6a72422070c68712dba3d675b0996a984924dfad08586053bd81370b), 최종 source proposal, 기존 승인 및 ledger81 descriptor를 기록한다. 과거 제안의 unapproved 필드는 당시 기록으로 남기고 덮어쓰지 않는다.

새 `parent-e360-suite-recovery-amendment-v1.json`을 만들고 실제 bytes/SHA를 네 소스에 고정한다. 이 literal은 기존 v2 resource extension 및 P3–P8 literal을 변경하지 않는다. 최소 계약은 exact81 prefix, local offset42, P9 한 번, 이전 P8 suite 실패 cap120/실제120.053/exit1 및 소유 자식 종료/once debit/closed failure, 추가 grant800, shared1400/local1289, generated9/modified8/original1/total10, auto0, P10 false, 원본 신규권한 false, Q 차감 후 생성 범위 종료다. 과거 생성과 원본 비용·역사적 요청의 cap/직접 authority를 유지한다.

기존 12개 활성화 필드와 `e360_resource_grant`는 exact typed 값 그대로 유지한다. 새 `e360_suite_recovery_activation`, `e360_suite_recovery_amendment` 쌍과 별도 `e360_suite_recovery_resource_grant`만 추가한다. 새 grant가 current limit을 변경하는 유일한 근거이며 이전 grant 객체를 수정하지 않는다. 사전 exact81/소유 작업0/미차감0/예약0 확인 뒤 native once activation으로 적용한다. bool/float와 정수를 구분하고 PowerShell native int0과 JSON long0 문제는 불변 권한 값의 타입을 사용해 해결한다. 사용량 리셋이나 이전 scalar 덮어쓰기 우회는 금지한다.

독립 source pins는 inherited81, 직접 승인, 최종 승인 제안, 실제 P8 실패/닫힘, Astra 리뷰/ROOT 반영/검토 source roster/스냅샷을 포함한다. 활성 최종 P9 source SHA를 literal에 다시 넣는 순환을 만들지 않는다. 최종 active own9는 요청/receipt/closed-exit/Q에서 별도로 실제 검증한다.

고정 reader `suite_recovery_amendment`를 열 번째로 추가한다. 기존 9 readers/16 entries/512 descriptors/256KiB literal/2MiB total 및 fresh descriptors-only mandatory fences를 유지한다. relocation은 기존 P8 22개 순서와 값 그대로 + P8 C9 최종 사본 네 개 = 26개다. historical P8 mutable source relocation은 새 literal을 사용하며 P3–P7 과거 mapping과 literal caps는 그대로다. 권한 선언은 old P8가 새 P9 own hash로 현재 Q 자격을 얻도록 만들지 않는다.

## 정책 선택과 끝까지 일관된 admission

현재 정책은 live ledger의 완전한 새 활성화 쌍/새 grant/typed descriptor를 uncached 검증한 경우에만 P9를 선택한다. 명시적 historical pass는 해당 old literal과 caps를 유지한다. `authority_path_for_pass`, `prospective_authority_spec`, `effective_policy`, call request schema, pass/failure state, source closure, closed-exit, receipt, helper completion, Q와 fallback까지 P9를 연결한다. 새로운 수치로 과거 P8를 재해석하지 않는다.

tool pre-import bootstrap 및 metadata helper의 ENTRY 직후 bounded request/literal 선택에도 P9 직접 descriptor를 연결한다. actual OperationClock/root/callable identity, actual parent ENTRY는 변하지 않는다. 모든 worker/helper/scanner cap은 정확한 요청의 literal에서 선택하고 P8 bootstrap caps를 새 P9 suite600에 잘못 적용하지 않는다. 늦게 생성한 clock으로 시간을 리셋하지 않는다.

PowerShell account는 별도 once activation action을 제공한다. 전역 inherited 검증과 과거 debit call-policy는 0–80 전부 이전 authority로 검증한다. 81 이후의 새 P9 debit만 새 literal을 사용한다. after-grant fit로 전체798 예약을 먼저 확인하고 실패하면 ledger를 쓰지 않는다. Checked-Call/Validate-Call-State/live suffix/begin/debit/pending/close/helper semantics/full-return fallback/remaining fields/stage and cumulative params/final exact-key schema 모두 현재1400/1289를 사용한다. 현재 기존 ledger original/stage 값은 변화하지 않는다. 최종 성공 기준은 activation 선언이 아니라 실제 admitted 9-call chain의 모든 비용 차감과 Q qualified이다.

## 고정된 새 생성 실행 일정

| call | outer/internal/child/science (초) | successful suffix (초) |
|---|---:|---:|
| Ruff | 3 | 795 |
| scan | 20/18 | 775 |
| fixture-build | 25/22 | 750 |
| fixture-worker | 55/46/50/42 | 695 |
| fixture-seal | 15/13 | 680 |
| suite-build | 30/27 | 650 |
| suite-worker | 600/540/560/500 | 50 |
| suite-seal | 20/18 | 30 |
| Q | 30/27 | 0 |

총798초. 역할 worker 표기는 outer/internal/child/science이며 실제 effective_roles key 순서는 기존과 같다. scanner necessary20+fixture55+suite600+administration120=795, fixture future administration95, suite future administration50. 새로운 grant 뒤 잔여 shared897.5449844/local897.502, 최대 예약 후 여유99.5449844/99.502. paid dispatch는 실제 승인이 이미 있으나 source 리뷰/activation/request admission을 먼저 끝낸 뒤 수행한다.

P9 fixture에도 실제 parent ENTRY 기반 동일 reached-only 7단계 checkpoint를 연결한다. 역사적 P8 checkpoint validation은 당시 P8 authority/caps/own source에 유지한다. 새 current fixture는 P9/current own9로만 자격을 얻는다. checkpoint는 passive diagnostic이며 Q 권한이나 completion 대신 쓰지 않는다. CASE11/Q14/helper12 key 계약은 유지한다.

승인된 patch의 genuine SuiteDeadline, canonical script import bridge, two fresh journal fences, real durable fsync/close/readback, captured actual timer/writer, first failure preservation/N1/N2, fresh local metadata completion, poison 및 finally cleanup을 유지한다. 전체 71개/213 phase baseline과 58/66/70 보호 roster를 보존한다. 새 통합의 meaningful regressions는 현재 full suite에 포함한다. filtering/skip/xfail/임의 삭제 금지. 실제 whole collection roster를 동적으로 일치시킨다. 6N+4 journal records/8KiB/4MiB/16KiB extra reserve를 유지하고 N이84를 넘으면 dispatch 전 거부한다. journal은 diagnostic이며 CASE/Q 자격이 아니다.

## 검증과 중단 판정

source-only ROOT 리뷰는 명시적 history caps, 신규 approval/typed grant/limits, partial pair/type substitution, once replay, exact81/permanent closed P8 suite failure, source freshness/26 relocation, future-suffix fit, pre-import clocks, helper modes, P9/current fixture/P8 history, protected322 및 full roster를 검사한다. 이미 승인된 최종 patch regressions를 보존한다.

최종 source freeze 후 Ruff→scan→fixture-build→fixture-worker→fixture-seal→suite-build→suite-worker→suite-seal→Q를 새 admitted source/request로 한 번 실행한다. 모든 Python call의 actual full return을 먼저 durable 기록하고 정확히 한 번 차감한다. worker는 실제 closed observer/receipt/first failure를 seal에서 확인한다. 반환이 outer cap을 넘으면 실제 시간 그대로 실패 및 차감하며 cap로 잘라 쓰지 않는다. fixture/현재 source/whole-suite CASE가 하나라도 실패하면 해당 실패를 닫고 후속 Q/원본을 실행하지 않는다. pending/holds/소유 자식이 남으면 unknown을 실패로 보존하고 owned scope만 정리한다. 실제 코드 실패는 traceback/첫 failed report/journal의 available evidence로 판정하고 추정과 구분한다.

Q actual debit 이후 또는 먼저 발생한 실제 닫힌 실패에서 이번 P9 범위를 종료한다. P10/자동 재시도/무료 보조 Python/원본/1.5초 및10초 몸체 실행 권한은 생기지 않는다. 원본 helper build5/4.5, worker12/11/10/9, seal2/1.5와 original20/300/10200을 유지한다. generated PASS는 source closure 복구 증거만이며 F1–F8, 자연 보행, 생물학적 기전 및 원본 timing feasibility는 미완료로 보고한다.
