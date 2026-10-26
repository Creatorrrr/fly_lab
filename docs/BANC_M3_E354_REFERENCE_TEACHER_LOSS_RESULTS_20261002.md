# E354 결과: 구현 검증 통과, 원본 실행 종료 기록 실패

ROOT 설계 → 실제 GPT-6 Astra/max 리뷰·반영 → 실제 GPT-6.1 Sol/high FIRST → ROOT FIRST → SAME Sol ONE R1 → ROOT SECOND 순서를 수행했다. 처음 발견한 결함은 실제 pytest 대신 NOOP 실행 종료 기록으로 CPU 자격을 부여할 수 있다는 점이었다. Sol R1은 현재 전체 pytest와 등록된 ROOT 추가 테스트의 argv·소스 pin을 강제하도록 수정했다. 현재 소스 테스트13 + ROOT 독립 회귀4 =17개 및 현재 소스 Ruff가 통과했다. ROOT SECOND에 제품 소스 잔여 수정은 없었다.

현재 소스는 `tools/banc_m3_reference_cadence_teacher.py` SHA10b693b2e4817b78e1160476c7d185835ee862696c009a6f159e7bdccee5142a, test SHA6b01daadd930bffc484db3c6295da4714baf2b701ffede47c22cabada51bf776다. whole300/window16의 f64 normalized MSE, 서로 독립된 q-only autograd, clone-detach, fixed index 및 finite/device 계약이 생성 데이터에서 검증됐다. GPU/신경 파라미터 VJP/보행 개선은 검증하지 않았다.

현재 17개 CPU 자격과 passive freeze 뒤 원본 run10s + audit5s를 prospectively 예약했다. 원본 실행은 실제 전체 tool return2.8036107s / chunk5d29f7 / exit1로 종료했다. ROOT가 작성한 E267 실행 계약에 필수 `cumulative_limit_s`를 누락했다. 불변 E267은 작업 뒤 종료 보고서를 저장하는 finally에서 KeyError를 냈고 E346도 없는 보고서를 읽으려다 실패했다. 실제 ROOT observer가 관측한 PID/생성시각은 전부 종료됐고 cleanup0이지만, E267 감독 결과와 E346 identity 결과는 존재하지 않는다. worker exit0을 추정하거나 합성하지 않는다.

worker raw261811B, receipt758B, result4807B는 저장돼 있다. result는 계산 완료와 full/window 기울기 반환 각1회를 **선언**하고 whole loss65.72568460423544/window loss53.679815983943776/prefix0/startup0/self0을 기록한다. 이것들은 외부 종료 실패 뒤 독립 감사되지 않은 값이다. ROOT는 새 raw/원본 NPZ byte를 열거나 해시하지 않았다. 저장된 provisional `external_completion_qualified=false`는 그대로 유지한다.

E354 판정은 `H354=not_evaluated_due_failed_outer_closure`, diagnostic completion false다. 기존 감사 gate의 expected_exit0을 우회하지 않고 감사0/재실행0이다. 실패 계약/로그/데이터는 보존한다. 실제 실패 비용2.8036107s를 한 번 차감해 원본 누적9871.733085099502/9900s, 남은28.266914900497795s, reserve/jobs0이다. Sol 생성 누적17.7711743/20s, ROOT15.6234634/20s이며 한도를 연장하지 않았다.

다음 단계는 별도 E355 forensic recovery를 ROOT가 설계하고 실제 Astra 리뷰를 받는 것이다. 저장된 acquisition을 새 독립 NumPy 수식으로 검증할 수는 있지만, 실패한 E354 완료나 worker exit를 소급해 변경하는 설계는 허용하지 않는다. 현재 goal/F1–F8/M2/UI/F6/생물학적 미검증과 원본 campaign9900s를 유지한다.

권위 기록: `verification/banc-natural-gait-m3-20260919/parent-e354-final-decision.json`, `parent-e354-failed-full-return.json`, `parent-e354-root-second-review.json`.
