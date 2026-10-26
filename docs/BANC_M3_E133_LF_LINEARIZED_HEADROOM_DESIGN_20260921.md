# E133: 저장된 LF 미분의 국소 개선 여지 진단

2026-09-21 · M3/H133 · 실행 전 부모 설계 · 실제 보행 후보 채택 없음

E132는 feasible LF 목적을4.6096% 개선했지만 마지막 feasible34→39의 추가 감소는 약0.018%였고, 초기99 제약 경계에 가깝다. 이것만으로 수렴이나 모델 표현력 부족을 결론낼 수 없다. 후속 계산을 늘리기 전에 **저장된 정확한 미분의 선형화에서 원래 제약과 bounds를 지키는 추가 하강 여지가 있는지** 확인한다. 실제 신경·기하·신체는 재실행하지 않는다.

H133은 E132 선택점 주변의 사전 제한 영역에서 원래 목적의1차 예측 감소가1% 이상인 방향이 존재한다는 국소 주장이다. 경쟁 설명은 현재 활성 초기 제약과 파라미터 bounds가 그 크기의1차 개선을 제한한다는 것이다. 선형화는 실제 유한 이동의 응답을 검증하지 않으므로 어느 결과도 새 후보, 전역 도달 불가능성 또는 자연 보행 판정이 아니다.

입력은 E13239번째 raw/선택proposal/결과/부모최종검토, E128 fixture와 원래 `constraint_values`/`objective_and_jac` 소스다. 모든 사용 파일의 SHA를 새 freeze에 고정한다. 원래 소스 및 E128–E132 결과를 변경하지 않는다. LF273delta/17gain,3문맥/99초기표본/7관절,4761제약과 원래 cap0.99960004/selector0.99980001을 유지한다.

저장 B_initial99와 parameter_muscle_map으로 q/ltm Jacobian을 복원한다. q의 근육 축은 저장 q_coefficient의 **local muscle axis**를 사용하고 이것이 fixture moment의 global muscle_indices와 일치하는지 검증한다. LTM은 fixture ltm에 대응하는 local muscle만 미분한다. 복원한 q/ltm 오차로 계산한12ratio는 원자료와 동일해야 한다. `t0=max(losses)`, prior와 gradient는 원래 E128 식을 재사용한다.

고정 반경 rho=[.005,.02,.05] 각각에서 변수 y=(delta/gain 변화,t 변화)에 대해 `min g·y`, `-J·y <= c`, 원래 파라미터 box와 `abs(y_i)<=rho*(upper_i-lower_i)`의 교집합을 푼다. t 변화는 이 trust box를 적용하지 않는다. 이 LP의 목적·제약·bounds 전체와 각 반환 x/fun/slack/status/iteration/warning/시간을 원자료로 보존한다. LP 값은 **1차 예측**으로만 표시한다.

SciPy `linprog(method='highs-ds')`, presolve=true, primal/dual feasibility tolerance1e-9, `time_limit=1.0`초/호출을 사용한다. HiGHS threads=1 요청은 SciPy가 전달하는 원래 warning과 함께 기록하며, CPU/BLAS thread1도 실제 기록한다. 최대3회,반경순서 고정,시작 전 내부마감까지2초 미만이면 다음 LP는 not_run으로 남긴다. 반복·후보 재평가·반경 추가·재시작은 없다.

수치 유효성은 모든 입력·ID·shape·finite·bounds/초기selector일치,요청/반환 수,solver status 및 원래 A,b/box에서 독립 잔차 확인으로 판단한다. status0 해의 최대 부등식·box 위반 <=1e-7, fun과독립 g·y 차<=1e-9를 요구한다. status2는 해당 선형화 LP의 infeasible로 기록하고 무효한 해를 꾸미지 않는다. 시간/수치실패·미실행은 unresolved다.

유효한 최종 rho=.05 LP가 최적 반환되고 예측 목적 감소 >=1%면 H133 supported, 최적 반환되어1%미만이면 이 국소 선형화 범위에서 refuted다. 마지막 LP가 infeasible/미완료/무효이면 inconclusive다. 작은 두 반경은 동일식의 진단이며 사후 반경 선택으로 주판정을 바꾸지 않는다. active bounds,near-active startup constraints,예측 감소를 전부 보고한다. 보행후보/실제 신경/신체/F6/기하 호출은 항상0이다.

원본 분석 프로세스1회/retry0,내부8초/외부10초/supervisor12초다. 기존 실제 보수적누적1513.182953200012초를1514로 올려12초 배정,1526<=기존shared1530초로 유지한다. 요청한 CPU LP와 실제 경과 시간을 원본 분석 비용에 포함하고, 부모 저장배열 검산·구현 생성테스트는 이전과 같은 별도 기록에 남긴다. LP당1초의 solver한도와 전체외부한도를 모두 적용한다.

부모 설계→gpt-5.6-sol/high 구현→부모1차 리뷰→필요한 첫 수정은 같은 Sol→부모2차 리뷰→남는 수정은 부모 직접 처리한다. 기존 미분과 제약식의 조립·LP·계상 연결 범위라 high를 실제 사용한다. Sol 소유는 새 `tools/analyze_c_banc_m3_lf_linearized_headroom.py`, `tests/test_c_banc_lf_linearized_headroom.py`뿐이다. 부모는 freeze/supervisor/독립검토를 소유한다. 테스트는 생성 CPU 배열만 사용하며 원본 읽기/원본 LP/신경/기하를 실행하지 않는다.

CLI는 `--freeze --freeze-sha --source-review --source-review-sha --out`. 정확한 두 소스 해시와 `accepted_for_single_original_lf_linearized_headroom=true`, `blocking_defects=[]`인 부모 최종소스 receipt가 필요하다. 원본 실행 전까지 원본 분석0이다.
