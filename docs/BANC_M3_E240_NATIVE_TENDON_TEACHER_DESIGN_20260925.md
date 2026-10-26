# E240: 원래 BANC 힘줄 출력을 유지한 관절 교사 보행

E239는 E85와 exact인 교사10초의 BANC 입력을 수집했다. 그러나 교사가 신체에 가한 여섯 발끝 힘줄.2와 원래 신경 decoder 사이에는 startup RMS .0379619/max .1999993 차이가 있었다. .5초 이후 차이는 작지만, 이를 무시하고 q만 학습하면 시작 신체의 감각 입력이 달라질 수 있다. E239는 유효한 고정 힘줄 기준 자료로 보존하며 자율 보행으로 채택하지 않는다.

H240은 **관절 q는 E85 기록을 적용하고 힘줄은 실제 BANC decoder에 맡겨도 건강하고 반복적인10초 기준 보행을 만들 수 있다**는 것이다. 경쟁 설명은 E85 보행이 신경 decoder가 주지 못하는 초기 힘줄 유지에 의존한다는 것이다. 이 비교는 실제 자율 후보와 같은 힘줄 경로의 학습 자료를 만들 수 있는지 판정한다. q 교사가 있는 움직임이므로 최종 신경 단독 보행 성공은 아니다.

## 변경과 불변

E239와 같은 E224 start의 원래 MotorTimescale 신경 상태를 복원하고 같은 E208 signed-input 관측기를 사용한다. 원래158706뉴런/CSR/시간상수/drive/감각필터/391MN/96근육/초기 activation을 그대로 유지한다. E223 effector는 사용하지 않는다. E85의2000개 기록 q를 neural decode 직후 body 경계에서 대신 적용하되, **그 호출에서 decoder가 제공한 8개 tendon 값을 그대로 신체에 전달**한다. 복부의 원래.2 값도 추가 조정하지 않는다. 실제 applied_tendon과 decoded_tendon은 매 control byte exact이어야 한다. 외부에서 미리 계산한 tendon, 이전 control의 tendon, teacher의.2 배열을 대입하지 않는다. adhesion은 원래 false이고 보간/신체/world는 같으며 body.advance를 두 번 호출하지 않는다.

기존 E238/E239 소스와 module global은 변경하지 않는다. 새 인스턴스 경계와 실행기는 새 두 파일에서 구현하고 검증된 관측·clock·publication helper는 재사용할 수 있다. 저장된 E238 parity와 E239 완주 검산 hash를 선행 확인한다. 새 전체 start는 E239 observed-start와 exact다. 첫80 RK stage(첫 control의 물리 action 이전) 세 입력391MN 및 control0/1 original motor readout은 E239와 byte exact여야 한다. 이후 신체/신경 상태는 힘줄 개입으로 달라질 수 있으므로 E85 final exact를 요구하거나 의도된 차이를 기술 오류로 처리하지 않는다.

새 원본은 한 arm의2000control/10초/40000neural ticks/100000physics steps/160000RK stages다. 입력·readout·decoded/applied q/tendon·adhesion·control/native·whole observed-start/final checkpoint를 E239와 같은 schema로 저장한다. 첫 control 검증은 다음 control 전에 수행해 초기 불일치 시 중단한다. 관측기가 실행 중이므로 첫80 stage는 종료 후 검산할 수 있으며, 첫control neuron/readout·초기 whole state는 즉시 확인한다. 완주 후 신경 입력 전체 finite, 카운터391=160000/overflow0/armed0과 명령·시간축을 확인한다. 부분 오류의 원래 원인·clock·raw를 보존하고 자동 재시도0이다.

## 판정

먼저 수집 유효성(동일 start/입력 첫 구간, 실제 적용 q 및 tendon 경로, finite·clock·카운터·hash·자원)을 판정한다. 유효한 새 물리 궤적을 기존 `tools/analyze_c_banc_biological_gait.py`의 같은 .5≤t<10초 창과 native100us 기록으로 평가한다. 전진>=10mm, net/path>=.5, 전체 native upright>=.9/nonfoot0/수치오류0; 원래 전진 bout 규칙 아래 각 다리 complete cycles>=10, 좌우3쌍 평균 위상.5±.15·R>=.5; LF/RF/LH/RH 무릎95–5폭>=25/30/40/35도와 각 다리 period CV<=.3을 그대로 사용한다. E85/E239와 같은 평가기로 비교하며 임계값이나 창을 바꾸지 않는다.

모든 기준을 만족하면 **관절 교사와 원래 힘줄 출력이 양립하는 학습 자료**로만 인정하고, 이 입력과 같은 control의 E85 applied q를 짝지은 fit을 다음 단계에서 설계한다. 전체 q 적합 전에 건강성이 떨어지거나 반복 기준이 실패하면 학습을 진행하지 않고 힘줄 인터페이스의 초기 조건/전달 문제를 검토한다. 이 경우 단순 기준 완화나 다른 기간을 성공 구간으로 선택하지 않는다. 기존 E197 startup proxy를 새 목표로 자동 변환하지 않는다. 최종 F1–F8와 F6 미노출/E150/M2/UI는 유지한다.

## 실행·자원

실제 gpt-6-sol/high가 `tools/observe_c_banc_m3_native_tendon_teacher.py`, `tests/test_c_banc_native_tendon_teacher.py`를 구현한다. 신경/물리 호출 경계와 부분 오류 회계가 결합된 변경이어서 high를 사용한다. 부모 FIRST→필요시 같은 Sol 첫 수정→부모 SECOND→잔여 수정은 부모 순서다. 생성 CPU 검사는 현재 decoder tendon을 같은 호출에 전달하는지, q만 대체하는지, 오류 때 경계 복구·teacher 실패 기록이 보존되는지 확인한다. 수학 kernel 변경이 없으므로 새 생성 CUDA 실행은 하지 않는다. 원본은 부모만 실행하며 GPU 중복0이다.

CPU1/GPU1/원본job1/retry0/fit0/solver0, 신체1/신경생성7, setup280실행/140capture와 과학 시간을 구분한다. 실제 E239 감독129.5977초를 기준으로 내부160초(저장15초 예약)/외부175초/감독190초, RSS6GiB/추가GPU4GiB/출력2GiB/free3GiB다. 기존 원본 누적6408.073387799643/6475초·잔여66.926612200357초를 보존하고 원본 전에 배정+130→6605초·잔여196.926612200357초에서190초를 예약한다. 부모의 저장 검산·고정 평가기는 별도 계측한다. 새 전역 사용자 시간상한은 만들지 않는다.
