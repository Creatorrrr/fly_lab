# E360 소스 검사 복구 반영 계획 — 실행 승인 전 제안

기존 M3 finite-goal의 F1–F8을 유지한다. 실제 Astra 6/max의 `parent-e360-source-scan-recovery-astra-review.json` R1–R5를 반영했다. 원본 초안과 리뷰, 50개 실제 비용은 별도로 보존한다. 이 문서는 추가 예산 지급이나 새 실행 authority가 아니다.

## 확인된 결과와 현재 한도

같은 GPT-6.1 Sol/high의 FIRST와 ONE R1 후 ROOT SECOND를 마쳤다. Actual-cost/debit JSON-only 검증과 bounded UTF-8 source descriptor 검증을 분리했고 테스트 반복문 바인딩을 수정했다. 수정 당시 실제 Ruff는 0.409초/exit1과 0.571초/exit0이다. 그 뒤 실제 소스 검사 4.121초/exit1은 내부3.5초 제한에 걸렸고 현재 보고서를 발행하지 못했다. 마지막 세 비용을 포함한 총50개 비용을 한 번씩 기록했다.

생성 공유 사용126.9030156/150초, E360 지역 사용15.946/39초다. 공유 잔여23.0969844초와 지역 잔여23.054초 모두 현재 다음 단계 예약26초에 부족하다. 원본은 전체10020.115960899504/10200초, 공동126.361/300초, E360 0/20초이며 모든 hold/job0, candidate1/retry0을 유지한다. 현재 fixture/whole suite/Q/원본 저장 구간 진단은 미실행이다.

## ROOT 잔여 소스 수정

이번 implementation의 같은 Sol ONE R1 뒤이므로 추가 Sol R2 대신 ROOT가 세 변경을 마무리한다. 최초 실제 `_REAL_SCAN_PROJECT`의 전체 SHA/layout/no-reparse 검사를 유지하고 MISS 직후의 중복 프로젝트 검사만 줄인다. MISS의 own identity 검사와 HIT의 전체 파일·parsed source·own identity·positive/negative/namespace layout·canonical/no-reparse 검사는 매번 유지한다. 실제 scan_current의 authority/oracle/dependency/installed/clock 검증 후에만 캐시에 넣는다.

MISS+commit=False의 아직 캐시에 들어가지 않은 새 보고서는 바로 반환한다. HIT와 commit=True의 private cache/반환 nested object 복사 격리는 유지한다. scan_current는 enrichment 전에 private raw AST 사본을 하나 고정한다. 파일별 canonical parsed AST/SHA에 연결된 top-level exported-name 집합을 한 scan에서 한 번만 만든다. 기존 함수/클래스/import alias/단순 Name assignment 판정과 모든 negative lookup을 유지하며 tuple/nested/conditional/__all__/dynamic export를 새로 인정하지 않는다.

기존23개 case 함수/69개 setup-call-teardown phase를 유지하고, 기존 lexical/cache case 안에 MISS commit=False, nested copy isolation, deadline failure no-seed, alias/assignment/negative export의 행동 검증을 보강한다. 현재 소스에 대한 테스트나 성능 향상은 아직 검증하지 않았다. 과거 통과 소스/closure를 현재 결과로 쓰지 않는다. `safe()`, 역할별 clock, ledger/Q/admission freshness, installed alias 검사 및 원본 범위를 줄이지 않는다.

## 승인 요청의 구체적인 범위

생성 검증 공유 한도 **150→170초, 추가20초**, E360 지역 한도 **39→59초**를 제안한다. 동시에 불충분한 helper 합계4초 예약을 다음 **19초의 유한 호출표**로 바꾸는 것까지 승인 제안에 명시한다. 기존 worker hard cap은 변경하지 않는다. 이 제안은 한 번의 유한 검증 재개 시도이며 성공 보장이나 자동 재시도 권한이 아니다.

| 순서 | 생성 단계 | 실제 외부 full-return 최대 예약 | 다음 미실행 단계 최대 예약 |
| --- | --- | ---: | ---: |
| 1 | 최종 targeted Ruff | 1초 | 41초 |
| 2 | 새 현재 source scan | 4초, internal3.5초 | 37초 |
| 3 | ROOT fixture build helper | 5초 | 32초 |
| 4 | standalone production fixture worker | 6초, child5/internal4/science3 | 26초 |
| 5 | ROOT fixture seal helper | 2초 | 24초 |
| 6 | ROOT suite build helper | 5초 | 19초 |
| 7 | 전체 current suite worker | 12초, child10/internal8/science7 | 7초 |
| 8 | ROOT suite seal helper | 2초 | 5초 |
| 9 | ROOT current Q publication/validation helper | 5초 | 0초 |

Helper는 build5/seal2/build5/seal2/Q5의 **19초**다. Scanner의 미래 helper reserve19, fixture build 후 미래 administration reserve14, suite build 후 미래 reserve7로 정책을 함께 바꾸는 제안이다. 최초 최대 예약은1+4+5+6+2+5+12+2+5=**42초**다. 공유126.9030156+42=168.9030156≤170, 지역15.946+42=57.946≤59다. 실제 시간 충분성은 미검증이며 이 표는 실패 가능한 한도이다.

각 외부 한도는 ROOT dispatch부터 실제 최종 반환, helper의 import·cold AST·source 검증·fsync/close/reread·보고 반환·필요한 정리까지 포함한다. JSON request 작성과 순수 PowerShell metadata accounting은 기존 계약과 동일한 bookkeeping이며 Python 시간을 대신하지 않는다. Helper별 같은 ENTRY 내부 한도와 ROOT 외부 한도 판정을 구현하고, 실패 시 qualification을 발행/승격하지 않는다. Fixture parent/child의 cold scan을 기존 same-ENTRY deadline에 포함한다. 프로세스 간 prewarm/cache, ENTRY reset, 부분 case 실행은 허용하지 않는다.

각 단계가 반환하면 성공/실패/overrun의 실제 시간을 먼저 exact-once debit한다. Worker는 actual return→seal→worker debit/owned closure로 닫고, seal helper 비용도 따로 기록한다. Failed worker의 정리/봉인/비용 기록은 필요한 닫힘 작업이며 다음 과학 단계로 진행하는 허가가 아니다. 각 helper 및 worker 뒤 실제 사용·잔여·owned identity·남은 suffix fit을 다시 확인한다. 비정상 종료, 한도 초과, missing identity, 보고/소스/비용 linkage 실패 시 현재 Q 및 원본 admission을 닫는다. 한도 초과 비용을 상한으로 잘라 기록하지 않는다.

승인 전에는 code/text 리뷰·ROOT 잔여 수정·계획 갱신만 한다. Ruff/scanner/helper/fixture/suite 실행, 새 registration 발행, live 한도 변경은 하지 않는다. 승인 후 별도 ROOT 설계/actual Astra 리뷰/반영을 거친 authority integration을 GPT-6.1 Sol의 적절한 effort로 구현하고 정해진 FIRST/ONE R1/ROOT SECOND/ROOT 잔여 절차를 따른다. Frozen old registration와 redistribution amendment, exact inherited50 및 local offset42, 실제 human 승인 근거, 이 반영 계획/actual review/ROOT reflection을 새 acyclic authority에 연결한다. 모든 admission/contract/CASE/Q/failed closure/late publication/seal/account의 유효 한도와 helper 정책을 함께 일치시킨다. 원장 사용을47/50부터 다시 시작하지 않는다.

최종 source/authority freeze 후 표의 각 단계를 최대1회 순서대로 진행한다. 새 source report/실제 standalone production fixture/전체 현재 case receipt/새 Q의 실제 full return과 durable readback이 모두 닫힌 뒤에만 기존 원본 saved-prefix CPU 진단12/11/10/9, 지역20초, 최대1회/자동0을 고려한다. 원본 graph/whole/theta 재구성, 새 body/NN/GPU, alpha 검색, 추가 baseline/candidate 실행은 없다.

저장 구간 진단이 성공해도 E359/E358의 과거 source/publication/body 실패를 valid로 바꾸지 않는다. 자연 보행, 생물학 기전, F1–F8, M2/UI는 미완료다. 후속 실제 startup/1.5초 bout/cycle/alternation/slip 및10초·미사용 상태·6목표·정지/재시작 검증은 별도 고정 기준과 남은 원본 예산에 따라 진행한다.
