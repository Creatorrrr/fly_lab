# E360 현재 소스 검사 실패 후 복구 제안 — 2026-10-03

이 문서는 같은 finite-goal의 소스 수정과 검증 재개 제안이다. 새 실행 승인이나 예산 지급, 과거 실패의 qualification을 의미하지 않는다. F1–F8, 생물학 미검증, M2/UI와 실패 후보를 보존한다. 원본 BANC/body 재실행과 수정 후보/alpha 탐색은 이 단계에 포함하지 않는다.

## 현재 확인한 사실

기존 실제 Astra/max 설계 리뷰 → ROOT 반영 → 실제 GPT-6.1 Sol/high FIRST → ROOT FIRST → 같은 Sol ONE R1 → ROOT SECOND를 마쳤다. ROOT SECOND의 두 수정은 테스트 반복문 변수 default binding과 bounded UTF-8 source descriptor 검증이다. Actual-cost/debit는 JSON-only 검증을 유지한다. 실제 Ruff 0.409초/exit1을 누락 없이 기록한 뒤 최종 Ruff 0.571초/exit0을 기록했다.

새 소스의 최초 실제 검사 `e360-root-redistribution-closure1`은 4.121초/exit1이다. `scan_current → cached_project`에서 실제 AST traversal과 그 뒤의 반복 검사 중 내부 3.5초 제한에 걸렸다. 새 보고서는 발행되지 않았다. 원본 배열 읽기, 새 신체/신경/GPU 실행, 현재 standalone fixture/whole suite/Q는 모두 0이다. 과거 8.473초/exit0 탐색 보고서는 현재 qualification으로 재사용하지 않는다.

50개 실제 debit을 유지한다. E360 생성 사용 15.946/39초, 생성 공유 사용 126.9030156/150초, 잔여 23.0969844초다. 원본 공동 사용 126.361/300초, 전체 10020.115960899504/10200초, E360 원본 0/20초다. candidate1/retry0 및 모든 hold0은 그대로다. 현재 등록된 다음 검사4 + standalone fixture6 + whole suite12 + helper4 = 26초는 남은 생성 한도에 들어가지 않는다. 새로운 이름, 소스 수정, 세션 변경으로 비용을 reset하지 않는다.

## ROOT가 마무리할 소스 수정 설계

같은 implementation의 ONE R1 이후 남은 수정이므로 이번 수정은 ROOT가 맡는다. 실제 Astra/max가 다음 세 변경을 먼저 리뷰한다. 검증을 생략하는 persisted/cross-process cache를 만들지 않는다.

1. `cached_project`의 최초 실제 scan은 `_REAL_SCAN_PROJECT`가 이미 수행한 마지막 전체 프로젝트 SHA/layout/no-reparse 검사를 보존한다. 직후 동일한 프로젝트 전체를 다시 검사하는 부분은 cache hit에서만 수행한다. 최초 scan에도 scanner/helper `source_identity`의 SHA/no-reparse 검사는 유지한다. Source/layout drift는 거부하며 조용히 rescan하여 통과시키지 않는다. Cache seed는 실제 scan_current의 oracle/installed/dependency/clock 검증이 모두 끝난 뒤에만 허용한다.
2. 완료된 실제 scan을 cache에 넣기 전의 중복 deepcopy를 줄인다. Cache-owned object와 반환 object는 서로 공유하지 않는다. `commit=False`의 최초 scan은 아직 private cache가 없는 새 객체이므로 반환용 deepcopy가 필요 없다. Cache hit와 `commit=True`는 반환 복사 격리를 유지한다. scan_current가 저장할 원본 AST 결과는 별도 deepcopy 하나로 고정한다. Caller mutation으로 cache가 바뀌지 않아야 한다.
3. 한 실제 scan에서 같은 파일의 top-level exported-name 집합을 한 번만 만든다. 파일별 parsed AST와 SHA에 결합된 scan-local 집합이다. 기존 함수/클래스/import alias/단순 assignment/annotated assignment 판정 의미를 그대로 유지한다. Negative module lookup과 namespace/no-reparse/layout 기록은 계속 수행한다. Oracles로 파일이나 edges를 seed하지 않는다.

이 변경은 소스 분석에 근거한 성능 가설이다. 타이머 통과나 속도 개선을 아직 주장하지 않는다. 기존 23개 함수/69개 setup-call-teardown phase 및 실제 standalone production fixture를 모두 유지한다. Source pin, old registration, amendment, exact inherited42/47/50 cost linkage, ledger fresh reread, CASE9/Q11, owned identity/actual return/fsync-close-reread, CPU guard와 원본 범위는 유지한다. 조건이나 역할별 hard cap을 낮추거나 통과를 간주하지 않는다.

## 검증 재개에 필요한 구체적인 승인 제안

제안은 생성 검증 공유 한도 **150→170초, 추가20초**와 E360 지역 한도 **39→59초**다. 승인되지 않은 상태에서는 source-only 리뷰/수정/문서화만 수행한다. 아직 새로운 registration을 발행하거나 live ledger의 한도/사용을 변경하지 않는다.

승인 후에도 기존 50개 비용과 추가 발생 비용을 정확히 한 번 누적한다. 별도 acyclic amendment는 기존 old registration + redistribution amendment + exact inherited50 snapshot + 이 제안 + 실제 Astra 리뷰 + ROOT 반영을 pin한다. 새로운 source/Q/report를 registration의 자기 hash에 넣지 않는다. 모든 admission/contract/CASE/Q/late publication/seal/account는 기존 authority와 새 authority를 함께 확인한다. 고정 original20/300/10200, 후보/retry, 원본 진단1/자동0과 원본 범위는 변경하지 않는다.

재개 순서는 최종 source freeze → 실제 targeted Ruff 최대1초 → 수정된 현재 최초 source scan outer4/internal3.5 최대1회 → 새 standalone fixture6/5/4/3 → whole current suite12/10/8/7 → 새 Q → 저장된 partial prefix 원본 CPU 진단12/11/10/9 최대1회/자동0이다. Generated metadata helper 미래 reserve4초, fixture4/suite2를 유지하고 각 호출의 실제 full return을 기록한다. Source scan/fixture/suite/helper가 한도나 closure 검증을 실패하면 결과와 비용을 보존하고 Q/원본 admission을 닫는다. 추가20초는 성공 보장이나 실패 후 자동 재실행 권한이 아니다. 실패 후 새 실행은 변경 근거, 실제 잔여, 설계 리뷰를 다시 확인한다.

생성 다음 단계의 초기 최대 예약은 Ruff1 + scan4 + fixture6 + suite12 + helper4 = 27초다. 현재 실제 공유 사용126.9030156 +27 =153.9030156≤170, 지역15.946+27=42.946≤59다. 원본은 생성과 별도 실제 원장에 누적하며 F1–F8 미달을 유지한다. 역사적 E359 baseline1과 E358 body 오류는 새 CPU diagnostic의 성공으로 복구/qualification하지 않는다.

## 결과 기준과 남은 목표

새 source 검사, 실제 production fixture, 전체 현재 case receipt와 새 Q가 닫혀야 원본 배열을 읽는다. 원본은 등록된 세 NPZ의 선택10 NPY 배열만 CPU에서 읽고 clock/header/finite/health/saved proof/producer startup 값을 엄격히 재계산한다. 과거 실패의 첫 오류와 부차 오류를 분리해서 유지한다.

이 단계가 통과해도 자연 보행 완료가 아니다. 같은 후보의 startup/1.5초 실제 bout와 완전 cycle/교대/slip 근거, 고정 10초 F4/F5, 미사용 초기 상태/6개 목표/stop-restart/UI/M2/provenance는 여전히 별도 미충족이다. 후속 신경 후보는 새 ROOT 설계 → 실제 Astra/max 리뷰/반영 → GPT-6.1 Sol의 적절한 effort 구현 → ROOT FIRST → 같은 Sol ONE R1 → ROOT SECOND → ROOT 잔여 수정 절차로 진행한다.
