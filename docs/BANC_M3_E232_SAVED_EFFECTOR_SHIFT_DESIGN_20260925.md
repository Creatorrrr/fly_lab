# E232: 조건부 학습과 실제 보행의 저장 출력·기억 차이

E231의 경골 단일 hold도 후진과 제동을 늘려 기각됐다. 임의 관절 고정 탐색은 종료한다. E218은 E150 기록 입력에 대해 좋은 command proxy와 startup을 통과했지만 startup 여유는 약4e−8rad뿐이다. 실제 E224/E225는 startup RMS .018052184585/max .058065758583로 실패했다. E223의 새 반응·기억은 원래 신경 상태를 보존하지만, 관절 출력 변화 이후 신체 감각이 달라지면 학습 때의 신경 입력도 유지된다고 가정할 수 없다.

기존 E03–E29의 교사 재생/정적 decoder/full-circuit 학습 실패를 보존한다. 같은 제안을 재학습으로 반복하기 전에 **이미 저장된 E217/E218 조건부 계산과 E224/E225 실제 effector**를 대조한다. 이 단계는 저장 분석1회이며 BANC/신체/CUDA/fit 실행0이다. 신규 확인 상태와 F6는 읽지 않는다.

## 고정 질문과 비교

주 질문은 실제 startup 실패의 명령 차이를 부동소수점 readout, free effector 동역학, fixed48 MN, 초기 activation 차이로 정확히 분리할 수 있는가다. 경쟁 설명은 구현/초기상태 불일치가 첫5ms부터 크거나 실제 입력·기억이 학습 조건에서 이탈하는 것이다. 같은 시각의 비교는 같은 위상의 생물학적 보행 비교가 아니다. 현재 입력 E/I/drive의 모든 native stage가 실제 보행 파일에는 없으므로 완전한9D feature 차이나 feedback 인과를 입증하지 않는다.

E224 effector control0..600과 E225 continuation600..1663을 control600 seam 전 필드 exact 후 한 번만 합쳐1664행을 만든다. E217 parameters의 motor_ids391/free343/fixed48, membership96, pool96×391, moment42×96, rest42 및 E224 시작 snapshot의 실제 ID를 대조한다. E218 reconstruction의 rate/q_direct/activation/tendon과 E217 h_fast/h_slow를 동일 control 행에 정렬한다. 기준 startup은 E197 `recorded_command`의1..99행이며 기존 .018RMS/.04max를 그대로 사용한다. row0은 held 명령으로 별도 취급한다.

명령 차이는 동일한10ms EMA와 .02moment를 사용해 다음 rate 차이를 각각 전달한 합으로 분리한다.

- free 동역학: 실제 free_effector64 − E218 rate[free].
- readout 양자화: 실제 decoder_input32[free]의 float64값 − 실제 free_effector64.
- fixed48: 실제 decoder_input32[fixed] − E218 rate[fixed].
- 초기 activation: 실제 activation0 − E218 activation0의 EMA 감쇠항.

실제 free decoder가 effector64의 float32 cast이고 fixed decoder가 실제 MN rate인지를 전 행 확인한다. 각 성분은 .005초의 α=−expm1(−.005/.01)로 순차 전달하며, t≥1에서 실제−조건부 command의 합성 오차≤2e−10rad를 요구한다. 실제/조건부 각 activation·command도 원래 pool/EMA로 직접 재검산한다. 근육 값의 clamp나 출력 보정은 없다. row0 command와 activation 초기값을 섞지 않는다.

각 성분의 startup RMS/max·최대 위치와 그 성분을 실제 command에서 대수적으로 뺐을 때의 기존 startup 판정을 보고한다. 이는 저장 신호 대수 분해이고 신체 개입 결과가 아니다. 모든 startup 실패와 반대 성분의 상쇄를 숨기지 않는다.

## 인과 기억 관측

실제 free MN의 h_fast/h_slow와 E217의 같은6개 memory 좌표 차이를 원시 배열로 보존한다. 첫행/첫5ms, startup1..99, 초반100..600, 연장601..1663을 별도로 요약한다. 첫5ms h_max≤1e−8, effector_max≤1e−5, q_max≤1e−7이면 `first_tick_within_diagnostic_tolerance`; 아니면 초기/수치 경로 검토가 우선이다. 이 수치는 생물학적 동등성 기준이 아니라 E223 생성 오차보다 넓게 잡은 구현 진단 기준이다. 이후 최초 각 threshold 초과 control과0..1663 전 시계열을 기록한다.

E217의 동결64 center에서 좌표3..8의6D memory만 취해 `min_j sum((memory-center)^2)/(2*sigma2)`를 각 free MN에 계산한다. 학습2001행의 동일 projected-distance 최대값과 비교해 actual1664행의 초과(`max_train+1e−10`)를 기록한다. chunk32 이하, 고정center·sigma 그대로, 모든 nearest 동률은 최초index다. 이는 **6D 투영에서 관측된 학습 범위 초과**이지 전체9D probability나 모델 전체의 외삽 증명이 아니다. 9D 현재 입력을 복원하거나 미관측값을0으로 채우지 않는다.

결과 분기: 첫5ms부터 허용 오차를 넘으면 추가 학습 전에 구현/초기상태/입력 연산 순서부터 검토한다. 첫5ms는 맞고 이후 동역학/기억 차이가 커지며 양자화만 제거해도startup이 실패하면 단순 정밀도 변경으로 해결됐다고 주장하지 않고, 실제 feedback 경로의 stage 입력을 수집하는 별도 실험을 설계한다. 양자화 제거만으로 두startup조건이 복원되면 정밀도 경로가 다음 우선 판별이다. 이 분석만으로 어느 후보도 채택하지 않는다.

## 구현·검증·자원

실제 gpt-6-sol/high가 `tools/analyze_c_banc_m3_effector_distribution_shift.py`, `tests/test_c_banc_effector_distribution_shift.py`만 구현한다. 부모 FIRST→필요시 같은 Sol 첫수정→SECOND→부모 잔여수정. 합성 선형 EMA 분해, held row0, 중복 seam 및 nearest center/동률·비유한 입력 검사만으로 새 코드를 확인한다. 원본 분석은 부모가 소스/원자료 해시를 고정한 뒤1회만 실행한다. 부모는 별도 수식의 EMA/성분합·기존startup·nearest 거리 표본을 검산한다. 기존 동결 source를 변경하거나 mutable global patch하지 않는다.

출력에는 정렬된 control/원시 차이·성분별command, projected nearest와학습최대/초과mask, summary, advance-ledger 및 manifest가 필요하다. raw를 먼저 저장하고 요약 실패도 보존한다. 실패/부분 분석은 전체 판정을 만들지 않는다. B/H 전체를 불필요하게 적재하지 않고 필요한 NPZ 멤버만 읽는다.

CPU1/GPU0/저장자료 원본 분석job1/retry0/neural0/physics0/fit0/solver0. 내부25초(마감5초예약)/외부30/감독35초,RSS2GiB/출력128MiB/free1GiB. 기존 분석의 보수 회계 방식에 맞춰 이 분석job 감독 비용도 누적에 더한다. 현재6076.219747399653/6104초·잔여27.780252600347012초를 보존하고 **실행 전 단계 배정+10→6114초**, 새 잔여37.780252600347012초 중35초를 예약한다. 준비/생성/독립 부모검산·미계측 읽기는 별도이며 새로운 전체시간/토큰 상한을 만들지 않는다. E150최선/M2/UI/F6미노출/실패 이력/F1–F8 및 전체 목표 미완료를 유지한다.
