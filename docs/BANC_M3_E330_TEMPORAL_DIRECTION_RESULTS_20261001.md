# E330: 시간 목적의 실제 BANC 방향 검증 결과

원본 실행과 독립 검증은 유효하지만 방향 가설은 기각이다. 국소 손실 감소를 신경 연속 실행이나 실제 보행 성공으로 바꾸지 않았다. 사전 기준에 따라 연속 신경·신체는 미실행이며 동일 후보 F1–F8과 목표는 미완료/active다.

## 구현과 리뷰

현재 세션 설계→actual `gpt-6-astra/max` 설계 리뷰·정상화 추가 리뷰→root 반영→actual `gpt-6.1-sol/xhigh` 구현→root FIRST를 완료했다. FIRST blocking defect0으로 Sol R1은 불필요했고 root production 잔여수정0이다. 생성 CPU17 tests/Ruff, root 독립 CPU14, CUDA 연결17/실제 정상화 전달11, 전체 passive173 source/1248 input closure가 통과했다. 독립 생성 수치미분 최대차이는1.6339100270996456e-12다. 생성 CUDA 연결의 추가 GPU는92274688bytes였다.

E330 runner SHA `fa7bf0ce867d844bdce0c02e4ffa7ee2270df30c69487988441de924b976e997`, tests SHA `5d44a37a71b7b89e9b7c17287729853880fdb30e76d4e66e527916b9244514e6`. 기존 E329/과거 구현은 불변이다. CPU/GPU 재계산 정상화의 bytes 차이를 기록하고, 실제 손실 denominator는 E329 승인 f64 bytes SHA `c5324b972080cdaffc3e7e011e190fed78a94cb26107984e3e2934e18e0609d3`와 정확히 일치한다.

## 실제 결과

| 항목 | 중립 | plus | minus |
|---|---:|---:|---:|
| training loss | 8.103502495971870 | 8.104091073106778 | 8.101377246862823 |
| successor loss | 7.281789146803972 | 7.282275413955963 | 7.279682421296293 |
| all15 coordination | 0.821713349167898 | 0.821815659150815 | 0.821694825566529 |
| LR diagnostic | 0.609344286071986 | 0.609425631176667 | 0.609348629729816 |

50개20ms 조건부 창에서 중립의 네 평균 항은 E329의 full200 pairs와 기존1e-10 기준으로 일치했다. 관절/힘줄 명령 및 신경 끝 상태는 원래 E318을 재현했다. 실제 반올림된 효능의 관측 차이는0.0027138262439549976, surrogate VJP 예측은0.004406558794261441, 상대오차0.3841393316959369로 사전5% 기준을 실패했다. `local_direction=refuted`, `local_descent=true`이며 minus 감소량은0.0021252491090475445다. 감소만으로 방향 예측 합격을 대체하지 않았다.

전체 native CSR palette는 plus13087/minus13011로16384 이하이며 두 방향 모두 전달 가능하다. 선택 가중치의 변경 bit 수는 plus377747/minus377112, theta1 그룹은6966/6982다. 비선택 CSR/선택 부호/토폴로지는 보존했다. native 표현 한도가 이번 방향 기각의 이유는 아니다. 연속 실행은 `local_direction_not_supported`로 미실행이며 startup/연속 손실/실제 신체/보행 결과가 새로 생기지 않았다. E330 handoff는 신체 실행 자격이 없다.

## 독립 판정과 비용

감독 종료 뒤 root가 immutable 결과·입력·소스·receipt·raw 명령·8547개 gradient/실제 f32 theta·full CSR palette를 NumPy로 재계산했고932 checks가 통과했다. 이 PASS는 기각 판정이 원자료와 일치한다는 의미다. 전체 BANC/신체 재실행 없이 사후 감사4.117547800000466초, 최대 tree RSS430968832bytes였다.

원본 supervisor1회166.52443790000143초, 최대 tree RSS3103399936bytes, 재시도0/신체0/optimizer0이다. Forward12000ticks, recompute4000ticks, backward4000blockticks/200blocks, decoder replay300controls, forward150/backward50rollouts가 요청/반환/확인됐다. 선행 조건 실패로 연속6000ticks는 요청하지 않았다. 새 유한 단계 배정300초는 원본 전에 기존9060→9360초로 기록했고, 원래 실패 누적을 지우지 않았다. 원본 비용은9034.148870999521→9200.673308899522초로 한 번 합산했다. 예약은0/실행 중 원본0이며 잔여 단계 배정159.326691100478초다. 사용자 전체 예산은 신설하지 않았다.

Sol 생성 검사 실패 두 번의 로그도 보존했다(시스템 Python 선택 오류/구문·Ruff 오류). 생성 CPU 테스트 누적4.690697000001819초, root 생성 CPU1.8458019999998214초, CUDA4.241600600000311초이며 원본 실행과 별도다.

## 다음 설계의 근거

이미 저장된50개 창의 residual을 검토하면 control140/208이 큰 오차를 차지하고, 상위5개 창이 절대 residual의98.79492800323215%를 차지한다. 이 집중만으로 신경 VJP 오류나 위상 투영 오류를 확정할 수 없다. 다음은 저장 자료에서 objective의 관절 명령 미분과 선택된 reference branch를 분리해 `관측−예측`을 수치적으로 분해하는 새 진단이다. 기각 기준을 완화하거나 E330 원본을 재시도하지 않는다.

증거는 `verification/banc-natural-gait-m3-20260919/e330-temporal-direction/`, `parent-e330-original-supervision.json`, `parent-e330-original-raw-audit.json`, `parent-e330-residual-design-inspection.json` 및 고정 freeze/source review에 있다. M2/UI/F6는 그대로 보존했다.
