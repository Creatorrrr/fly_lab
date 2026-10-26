# E163 여섯 다리 신경 출력과 제동 시점의 저장 자료 대조

E162의 일정 RM group70 감쇠는 지속 보행을 개선하지 않았다. 같은 효능을 다시 탐색하지 않고, E150의 실제 E1 리듬과 근육 출력이 여섯 다리의 추진·제동 및 속도 중단에 어떻게 정렬되는지 확인한다. E152는 속도 중단 사건을, E154/E157은 1–2초의 signed force를, E161은 RM 출력과 수직 운동을 따로 다뤘다. 이번 범위는 이 자료들의 시간 연결과 기존 세 좌우 쌍 밖의 E1 관계다. 새 신경·물리·후보 선택·fit은 없다.

경쟁 설명은 다리 사이 신경 리듬의 관계가 일관되지 않는 경우와, 신경 리듬은 일관되지만 근육 출력/접촉이 전진에 적합하게 이어지지 않는 경우다. 두 현상은 함께 존재할 수 있다. 시점의 연관성이 원인이나 특정 개입 효과를 입증하지 않는다. 기존 기록에는 10초 전체 signed force가 없으므로 1–2초의 힘을 전체 구간으로 확대하지 않는다.

## 고정 자료와 순서

E150 recruitment-signals, recruitment-metadata, assembled-calibration, native-recruitment-state, adapter-provenance, trace 및 start/state.json의 정적 groups를 읽는다. JSON을 직접 읽을 뿐 StateStore를 생성하거나 복원하지 않는다. E152 e150/features, event-stacks 및 summary, E157 native-replay 및 layout을 함께 쓴다. source-phase-summary의 기존 결과와 과거 실패는 보존한다.

물리 다리 순서는 LF, LM, LH, RF, RM, RH다. 소프트웨어의 E1 core group 대응은 2,5,4,1,3,0이며 각각 neuron ID 53861,71439,152280,92648,97727,91167이다. 이 대응은 E37 projection-provenance의 직접 MN projection을 원래 side/effector와 대조한 할당이며 PAIRS 이름만으로 추정하지 않는다. E1 자체의 effector annotation은 없으므로 실제 동물의 E1 기능을 별도로 검증한 주장은 아니다. calibration의 core_cell_types=0 및 observed_ids의 유일한 대응을 검사한다. 96개 muscle의 소유권과 391 MN pool은 저장 groups/columns/membership에서 확인한다. LTM은 15,30,46,63,78,94이며 위치 moment=0이 물리 영향=0이라는 뜻은 아니다.

control 0..2000의 간격은 .005초, 신경 기록은 실제 .005..10초 2000행, native 힘은 1.0001..2초 10000행 및 control201..400 각각50행이다. 시간 절대오차 2e-8초, rate/command 대응 1e-10, 힘 적분 closure 1e-12를 쓴다. tick/index/ID, 유한값, 저장 availability와 완료 표식을 검사한다. q target의 rest+.02*moment*activation 재구성은 control1..2000에서 검사한다. control0은 실제 복원 관측을 유지하며 decoder 출력으로 간주하지 않는다.

## 계산 계약

1. 실제 E1의 5ms rate에서 기존 prominence20 peak와 동일한 3점 포물선 보정식을 사용한다. 초기 rate를 만들어 붙이지 않는다. 각 다리의 실제 인접 peak 사이만 선형 phase를 정의하며 양 끝 밖은 NaN이다. phase는 분석용 표지이며 controller에 쓰지 않는다. 원래 rate와 peak 시각, control별 phase/availability를 저장한다.

2. 모든 15개 다리 쌍의 phase 차이(first minus second)를 고정 [.5,2.5), [2.5,5), [5,10) 창에서 보고한다. query는 .5부터 .025초 간격이며 여섯 다리 peak로 공통 bracketing된 점만 사용한다. mean=arg(mean(exp(2πi*delta)))/(2π) mod1, R=절댓값, sample 수를 기록한다. 3창 사이 최대 원형 거리도 보고한다. R≥.8 및 창 사이 최대 거리≤.15는 기존 source 기준에 맞춘 **시간 일관성 진단**이며 F4 또는 생물학적 tripod 판정이 아니다. 어떤 동측 phase를 바람직한 값으로 미리 간주하지 않는다.

3. E152와 동일하게 [.5,10) 안의 인접 control에서 speed가 >.5에서 ≤.5mm/s로 내려가는 모든 crossing을 재구성한다. ±20control의 온전한 창만 포함하고 기존 event-stacks의 46개 index/41offset과 정확히 비교한다. 각 사건의 시각, E1 phase, 96개 MN pool rate·실제 activation, 42개 q target·후방 차분 dq/dt, 속도와 지지하중을 같은 축에 저장한다. t0의 dq/dt는 unavailable, 첫 차분은 관측과 명령 사이 변화로 명시하고 분석 창에는 포함하지 않는다. crossing의 각 다리 phase에 대해 1차와 2차 원형 harmonic의 mean/R/count를 전체 창 및 위 세 창에서 보고한다. 한 주기당 두 중단이 있으면 1차 R만으로 관계가 없다고 결론내리지 않는다. 서로 겹치는 사건은 독립 표본이 아니다.

4. E157 native force를 control별50행씩 묶는다. mean signed force, native 양의 힘 적분, 음의 힘 크기 적분, 평균하중 및 support≥.02BW 비율을 다리별로 보존한다. 먼저 native에서 부호를 나눈 뒤 적분하며 block mean에 부호를 적용하지 않는다. positive−negative=.005*mean force의 closure를 검사한다. 저장 세계 힘·수평 heading의 내적도 force_forward와 대조한다. 해당 명령 tick의 target과 native ctrl[8:50] 대응을 검사한다.

5. 힘 사건은 위와 같은 속도 crossing 중 control[k−20,k+20]가 모두 201..400에 들어오는 것만 포함한다. 제외된 모든 crossing에 이유를 남긴다. E1 phase 부족은 phase 통계만 unavailable로 처리하고 유효한 힘 사건을 버리지 않는다. ±.1초 force/load/support, activation/target/phase를 같은 event 축에 저장한다. 사건별 pre는 offset−19..0의20개 block, 즉 (t−.1,t], post는 offset1..20, 즉 (t,t+.1]다. 모든 다리의 양/음 impulse를 각각 합하고 모든 사건의 mean/median을 보고한다. 평균힘과 실제 근육 힘/기여를 혼동하지 않는다.

6. phase bin은 [0,1)을 같은 폭의12개로 고정한다. 자신의 E1 phase를 기준으로 96개 muscle의 pool rate/activation 및 해당 다리의7개 target을 위 세 창에서 평균한다. count와 unavailable mask를 함께 저장한다. 1–2초의 force도 해당 control tick의 E1 endpoint phase로 binning한다. 한 control의 신경/명령 값은 그 control의50개 물리 단계에 유지된 값이며, bin 표지는 신경 native phase를 관측한 것이 아니다. 각 force bin은 count, mean signed force, positive/negative impulse 합을 기록한다. 최적 lag·bin 수·peak 문턱·후보 파라미터를 찾는 sweep은 하지 않는다.

## 해석·검증·자원

도구의 유효성은 위 mapping·수식·창·시간·해시 계약의 일치다. 모든 쌍의 시간 일관성이 통과하더라도 보행 협응이 옳다는 뜻은 아니다. 출력/힘/중단의 반복되는 관계는 다음 단일 개입의 근거 후보일 뿐이다. 1–2초에 충분한 사건 또는 bracket이 없으면 해당 항목을 unavailable로 남긴다. 후보 채택, F1–F8, 전체 목표의 완료 판정은 항상 false다. F6·M2·활성 UI는 변경하지 않는다.

부모 설계 → 실제 gpt-5.6-sol/medium 첫 구현 → 부모 첫 리뷰 → 같은 Sol의 필요한 첫 수정 → 부모 두 번째 리뷰/잔여 수정 순서다. 고정된 저장 배열의 통계와 시점 정렬이므로 medium을 사용한다. 소유 파일은 tools/analyze_c_banc_m3_six_leg_timing.py와 tests/test_c_banc_six_leg_timing.py다. 부모는 독립 수식 검산과 기존 source-to-leg provenance 검토를 병행한다. 모델/물리 통합을 새로 구현할 때에는 high, 신경/CUDA/상태 구조 변경에는 xhigh를 별도 실제 호출에 적용한다.

입력/재사용 소스/과거 실패/M2를 전후 해시 확인한다. 생성 자료로 bin 경계, 원형 wrap, 두 번 반복되는 사건, missing bracket, 음·양 native 힘 상쇄, 정확한 event block 구간, 잘못된 tick/ID/완료표식, NumPy 스칼라 JSON 및 늦은 저장 실패를 검증한다. 원시 유도 배열은 수치 closure assert와 요약 판정보다 먼저 저장한다. source gate와 실제 frozen 계약만 읽는 preflight 후 원본 분석1회/자동재시도0으로 실행한다. 저장 실패는 성공 판정을 취소하고 원래 오류와 원시 증거를 보존한다.

CPU1/GPU0, body/StateStore/physics/neural/fit=0. 내부55/외부70/감독80초, 총 출력64MiB/여유1GiB다. 직전 원본 보수누적2767.0572974초를 올림한2768초에서 감독80초를 더한 사전2848초다. 직전2956초는 종료된 배정 상한이며 실제 누적 사용량을 초기화하지 않는다. 이 실험은 저장 자료 진단이고 자연 보행 개선 자체는 아직 미검증이다.
