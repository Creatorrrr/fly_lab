# E187 이후: MN144018 전체 incoming의 3초 동기 관측 1회

E187의 발전은 큰 입력을 찾았다는 점만이 아니라, 경계 절대 입력 coverage97–99%로도 사건 간 전류 차이73.16%가 미관측 잔여에 남는다는 것을 확인한 것이다. 따라서 알려진 relay54014를 일괄 약화하는 효능 대조를 선택하지 않는다. 이 잔여는 미관측185개 선행뉴런과 누적 오차의 합이며 특정 생물학적 억제 경로로 명명하지 않는다.

다음 E188의 단일 결정은 **195개 incoming source의 endpoint 기여를 모두 기록하면 E186의 고정 제동·추진 조건 차이를 어떤 기존 연결로 설명할 수 있는가**다. E150/E185 metadata와 현재 E187의 source mask를 우선 확인하고, 다른 실행인 E03/E39 또는 경계뿐인 E178로 시간축을 대신하지 않는다. 새 관측 없이 같은 조건의 연속 기록이 발견되면 그 자료의 exact prefix를 먼저 검증해 재사용한다.

없다면 기존 E185와 동일한 full158706 BANC/CUDA·FlyBody·theta0·초기상태에서3초 baseline을1회 실행하는 E188 사전 설계를 고정한다. 가중치/근육/감각 encoding/신체/정책·command에는 변경0이다. 원래 CSR에서 MN144018의195개 incoming edge/pre 순서를 먼저 동결한다. 매 control endpoint0..600에서 그 pre rate/mask와 MN current/drive/adaptation/rate를 기록하며 full-state current를 새로운 수식으로 바꾸지 않는다. 기존 packed CUDA `_current` 관측을 사용하고 원래 full network/neural clock/physics clock을 유지한다. observer 실패/누락은 즉시 실패로 남기며 몰래 재시작하지 않는다.

관측의 필수 유효성은 기존 E185601 control/30000 native prefix와 M2/UI 무변경, 각 시점195개 항의 full CSR ordered CUDA sum과 실제 current bits 일치, time0 anchor와600 intervals의 구분, 같은11사건/선택8사건/52window 조건 평균이다. 미관측185개를 각각 채운 결과와 이전10개 항을 모두 보존한다. float64 descriptive sum과 CUDA total 차이를 별도로 남겨 E187 residual을 분해한다. 어느 조건에서든 불리한 사건이나 양의 반력 표본이 없는 사건을 삭제하지 않는다. 특정 source의 signed contribution/반례가 확인되기 전 효능 개입·fit·gain/tau sweep을 하지 않는다.

예측은 두 갈래다. A: 실제 미관측 입력의 시간 변화가 E187 잔여 차이의 주된 부분이고 입력 누적 재현은 이를 확인한다. B: 출처별 차이가 작거나 서로 상쇄돼 한 연결을 선택하기 어렵거나, 관측 시계/재현 문제가 남는다. A에서도 보행 인과성은 미검증이다. B는 후보 선정 실패로 기록한다. 모델 밖 CPG/위상/궤적/root 교정과 임의 muscle coefficients는 여전히 금지한다.

부모가 소스 경로와 shadow observer 연결 위치를 먼저 설계하고 실제 `gpt-6-sol/high`가 구현한다. FIRST→같은Sol첫수정→SECOND→부모잔여수정을 유지한다. 원본 실행 전 정확한 감독/입력/hash gate, 산출물, 건강성 stop, 예측/대조, cleanup을 고정한다. 현재 원본 E188은0이다. 상속 누적4351.08495669975/4994초, 남음642.91504330025초에서 GPU1·신경3초12000steps·물리3초30000steps의 관측1회에 감독 벽시간최대150초(300초 공통상한보다 작음)를 우선 배정하는 설계 초안이다. 내부 실행/cleanup·분석 몫은 실제 사용 러너의 시계를 확인한 최종 설계에서 분리한다. 전체 예산을 새로 시작하거나 성공률·F1–F8을 낮추지 않는다.

이 문서는 후속 관측의 결정과 설계 범위이며 구현/원본실행 승인을 통과한 source gate 또는 성공 결과가 아니다. 기존 사용자 승인 범위의 로컬 작업으로 다음 단계에서 구체화하며 새 사용자 승인을 요구하지 않는다. F6 노출0, M2/UI 보존, 후보 미채택, 자연 보행/M3 전체 미완료다.
