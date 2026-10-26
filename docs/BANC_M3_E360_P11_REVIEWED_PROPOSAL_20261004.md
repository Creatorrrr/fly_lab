**승인 제안: 확인된 P10 소스 오류를 고친 뒤, 추가 배정 없이 기존 잔여 한도에서 수동 P11 한 번으로 생성 검증을 진행한다. P11은 아직 승인·활성화·실행되지 않았다.**

전체 목표는 [활성 M3 계획](BANC_NATURAL_GAIT_M3_GOAL_PLAN_20260919.md)의 BANC 신경 출력만으로 실제 여섯 다리가 협응해 걷는 F1–F8이다. 이 제안은 원본 신체 평가로 다시 진입하기 위한 생성 검증 복구 단계다. 교사·proxy·소스 리뷰·테스트·Q 성공을 자연 보행 또는 생물학적 기전의 검증으로 인정하지 않는다. 기존 M2/UI·해부학적 provenance·실패한 후보·누적 비용·candidate1/retry0/retrylimit1과 원본20/300/10200 조건을 유지한다.

직접 승인된 P10은 첫 Ruff exit1/B023 14건으로 종료했다. 실제 ROOT dispatch부터 최종 반환까지의 0.885초를 한 번 차감했고, 전체 비용은 83개다. 캡처 검증기의 정수 타입 불일치 때문에 남은 원래 invalid flag와 오류도 보존했다. 별도의 실제 소유 종료 증거로 해당 비용에 연결된 hold만 닫았다. 현재 pending/reservation/hold/begun/jobs는 0이며 실행 당시 소스와 원장은 그대로다. [P10 실제 결과](BANC_M3_E360_P10_RESULT_20261004.md)를 참고한다.

ROOT의 [초기 제안](BANC_M3_E360_P11_WITHIN_REMAINING_PROPOSAL_20261004.md)을 실제 GPT-6 Astra/max가 원자료부터 소스 리뷰했다. [리뷰 JSON](../verification/banc-natural-gait-m3-20260919/parent-e360-P11-astra-preparation-review.json)과 [리뷰 설명](../verification/banc-natural-gait-m3-20260919/parent-e360-P11-astra-preparation-review.md)의 ASTRA-P11-01을 ROOT가 반영했다. 이전 동일 Sol ONE R1 이후의 ROOT 잔여 수정이며 Sol R2는 없다.

별도 [테스트 수정 사본](../verification/banc-natural-gait-m3-20260919/parent-e360-p10-root-residual-v2-tests.py)은 기존 B023 입력 고정에 더해 정상 exit1와 관련 exit0 변형의 정수를 명시적인 Int64로 맞춘다. 음성 사례가 타입 불일치 때문에 우연히 통과하지 않도록 한다. [회계 수정 사본](../verification/banc-natural-gait-m3-20260919/parent-e360-p10-root-residual-proposed-account.ps1)은 생산 cap3 비교만 Int64로 맞춘다. 전역 엄격 비교, bool/double 거부, 의도적으로 잘못된 observed_s 정수 사례와 lint 규칙을 유지한다. [리뷰 패치](../verification/banc-natural-gait-m3-20260919/parent-e360-p10-ROOT-residual-v2-review.patch)는 활성 소스와 두 사본의 차이다. 활성 경로에 적용하지 않았다.

새 P11 통합은 실제 직접 승인 이후 GPT-6.1 Sol/high FIRST → ROOT FIRST → 필요하면 같은 Sol ONE R1 → ROOT SECOND → ROOT 잔여 수정 순서로 개발한다. 기존 Python/PowerShell 함수와 기존 테스트 안에서 다음 연결만 완성한다.

- P11 scope는 full83/offset42/old16 activation·amendment 필드/두 grant를 보존하고 새 pair2만 추가한다. grant0, shared1400/local1289, generated11/modified10/original1/total12이며 P12·자동 재시도·원본 실행은 false다. 승인 기록과 새 literal은 실제 직접 승인 이전에 작성하지 않는다.
- P10의 정확한 과거 pass/call/cost와 실행 당시 frozen tool/tests/account/helper 네 사본을 연결한다. 기존 relocation30 뒤에 네 개만 더해 P11에서는34다. 과거 P10은 새 current own9와 비교하지 않는다. P10의 exit1/0.885초/once debit/원래 internal_completion_valid=false, capture 오류·기존 capture leaf/ROOT freeze/genuine return/manifest/독립 owned closure를 그대로 요구한다.
- P9 예산 선택자를 유지하고 별도의 uncached whole83 P11 scope를 선택한다. 과거 P2–P10의 policy를 명시적으로 보존하고 debit index82는 P10, index83 이후는 승인된 P11로만 연결한다. unknown pass와 원본 실행을 거부한다.
- 같은 P11 descriptor를 scanner bootstrap·clock·helper import 전 상한·request/checkpoint/worker/CASE/Q/fallback 및 native policy/begin/actual/admin/finalize/Future-Admin에 연결한다. 현재 소스 확인은 P11에서 수행하며, 실패 seal은 정확히 같은 pending worker/attempt/call만 허용한다.
- 현재 reader12와 과거 P10 literal reader11을 별도로 확인한다. 16 entries/512 descriptors/각 literal256KiB/합계2MiB, 84개 이름·252 예정 phases와 과거79, journal4MiB/record8KiB/두 durable fence를 유지한다. 새 테스트 이름과 별도 검증 프레임워크는 추가하지 않는다.
- 기존 capture 구현을 고정 P11 함수/schema/path/authority로 연결하고 별도 freeze/concrete_fix leaf로 둔다. own-source의 열 번째 키로 만들지 않는다. 원래 P10 module을 보존한다. 각각1MiB raw stream/8KiB read buffer, 16KiB manifest/8KiB compact return, 실제 소유 identity·두 EOF·fsync/close/readback을 유지한다.

| 순서 | 외부 full-return 상한 | 내부 상한 | 이후 성공 suffix 예약 |
|---|---:|---:|---:|
| Ruff | 3초 | — | 795초 |
| Scanner | 20초 | 18초 | 775초 |
| Fixture build | 25초 | 22초 | 750초 |
| Fixture worker | 55초 | 46초 | 695초 |
| Fixture seal | 15초 | 13초 | 680초 |
| Suite build | 30초 | 27초 | 650초 |
| Suite worker | 600초 | 540초 | 50초 |
| Suite seal | 20초 | 18초 | 30초 |
| Qualification | 30초 | 27초 | 0초 |

최대 예약은 798초다. 현재 shared 잔여896.0529844초/local 잔여896.010초에서 예약 후 여유는 각각98.0529844초/98.010초다. fixture child50/science42, suite child560/science500, future administration95/50을 유지한다. 모든 실제 Python/import/AST/compile/Ruff/helper/test/probe는 실행 전에 ROOT admission을 받고, 실패·초과를 포함한 dispatch부터 최종 반환까지 시간을 한 번 차감한다. Native text/hash/source parse·metadata bookkeeping을 준비 작업으로 사용한다. 실제 소요 시간과 descriptor fit은 아직 검증하지 않았다.

첫 닫힌 실패에서 후속 검증을 중단한다. worker 실패 뒤에는 이미 예약된 동일 pending worker의 seal·actual debit·소유 종료만 마무리하고 새 검증/Q/원본을 시작하지 않는다. Q actual debit 또는 그보다 앞선 닫힌 실패에서 P11 한 번의 범위를 끝낸다. Original build5/4.5·seal2/1.5은 이전 실측보다 작으므로 Q 통과만으로 원본 신체 평가를 실행하지 않는다. 실제 current Q와 별도 실행 가능성·권한 검토가 필요하다.

ROOT의 이번 확인은 소스 이름·순서84와 과거79 유지, native PowerShell 소스 파싱 오류0, 실제 ledger83 보존이다. Ruff/Python 구문·import/fixture/suite/Q·원본/GPU의 이번 추가 실행은0이다. 두 소스 수정의 실행 검증, P11 통합·최종 소스 freeze·권한·활성화·실행은 미완료다. 이 문서는 proposal이며 실행 authority가 아니다.

마지막 직접 승인은 P10 한 번이었다. 그 pass는 이미 종료됐고 활성 P10 literal의 P11 권한은 false다. 예산 배정 없이 진행하더라도 P11이라는 다음 수동 실행 범위에는 직접 승인이 필요하다. 승인 대기는 그 실행과 권한 활성화만 막는다. 전체 finite-goal, 이전 승인·자원 한도와 실패 기록을 보존하며 문서·자동 continuation·소스 리뷰를 인간의 새 승인으로 사용하지 않는다. 최신 직접 지시를 우선하고 무관한 작업을 보존한다. 기존 증거를 낮춘 기준으로 통과시키지 않는다. F1–F8의 실제 효과가 확인돼야 goal complete이며, blocked는 같은 실제 blocker가 세 연속 goal turn에 반복되고 의미 있는 독립 진행이 없는 경우에만 판단한다.
