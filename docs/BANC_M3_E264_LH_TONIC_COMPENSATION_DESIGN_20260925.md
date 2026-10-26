# E264 LH 감각 연결 복원과 기준 전류의 분리 — 2026-09-25

E263에서 기준 감각 U0 아래 W6가 실제 q19 초기 명령 차이를 거의 재현했다(L1잔차0.606%). E262의 물리 자극 반환은 작았다. 같은 decoder 재적합은 E168–175와 E242–244의 제한에 겹친다. 이번에는 새 적응 상태·느린 필터·부호 반전 없이 **하나의 기존 뉴런111970에 대한 상수 기준 전류**만 시험한다. 감각→BANC→기존 운동 변환의 구조를 보존하는 공학적 교정 가설이며, 생물학적으로 확인된 homeostasis라고 부르지 않는다.

## 개입과 고정 교정식

W6의 원래 여섯 CSR 간선과 모든 신경 모수/마스크/decoder는 그대로 둔다. baseline E257의 controls1..99에서 각 기존 source6의 실제 발화와 output mask를 써서 `dI(t)=sum((W6-W0)*r_B(t)*mask)`를 float64로 계산한다. 보정값은 `bias64=-mean(dI(t))`, 실제 주입값은 이를 한 번 float32로 바꾼 값이다. 범위·개별 간선·창을 결과에 맞춰 고르지 않으며 fit/solver/grid는0이다. 이 값은 endpoint 전류 평균이지 연속 시간 전류 적분이 아니다.

기존 drive[111970]는0임을 확인한다. 보정 팔 Z에서는 control1부터 모든 신경 interval에서 오직 이 항에 bias32를 더한다. 기준 팔 X에서는0이다. 시간 이후 켜지는 gate나 저장 파형은 없다. 증가한 감각 결합의 정적 성분을 제한하는 상수 유효 입력일 뿐, source 발화를 음수로 만들거나 기존 간선을 삭제하지 않는다. 모든 센서 입력·DNg100 drive·navigation 배경과 나머지158705개 drive는 같아야 한다. 신경 엔진 identity는 같은 W6이며 bias/도출 규칙을 포함한 별도 실험 configuration identity를 보존한다.

## 한 번의 원본과 분기

1. 동일 E260 W6 control0 전체 신경 상태에서 X_U0를100controls(.5초) 재생한다. 모든101개 MN391/기존core24/관측82 및 전체 시작·끝 상태가 E260 X와 exact여야 한다. q19 산술은 E263 X와1e−10 이내여야 하며 실패하면 Z 전에 중단한다.
2. 같은 control0에서 Z_U0를 같은100controls로 실행하고, B/X/Z의 bank391×3 및 q19 세근육42/43/44를 비교한다. 신경 상태 외 bank는 동등한 원래 control0에서 시작한다. 실제 신체를 생성하거나 진행하지 않는다.
3. X_U0 끝 상태로 X_low/X_high, Z_U0 끝 상태로 Z_low/Z_high를 각각40controls(.2초) 진행한다. 전체 마지막 U0 drive를 held하며 기존 port85 source23 항만 E248의 고정 low/high(8.758351424802523/13.260401038721753을 float32로 변환)로 바꾼다. X bias0/Z bias32는 유지한다. 각 쌍은 같은 상태에서 시작한다. 고정56개 free LH MN의 controls1..40 발화 차이를 cap200으로 나눈 RMS를 R_X/R_Z로 구한다. 서로 다른 operating point 간 비교이며 단일경로 매개율은 아니다.
4. 두 번째 새 W6 모델이 저장된 Z_U0 끝 신경 상태와 bank를 읽어 Z_low 40controls를 재현해야 한다. 전체41행 관측/모든391MN 및 끝 전체 state·bank가 exact여야 한다. 새 state 변수는 없지만 configuration bias와 실제 drive가 함께 복원돼야 한다.

순서 X_U0, Z_U0, X_low, X_high, Z_low, Z_high, disk_Z_low를 고정한다. 전체400controls=8000neural ticks, 신경 생성2/setup120실행+capture40기록이다. X_U0의0bias exact 미래와 disk 미래는 다른 검증으로 분리한다. 모든 원래 rate/adaptation/drive·부호·CSR/actual packed CUDA·float32 주입 및 원래 물리 외 상태를 검증한다. baseline B는 E260의 이미 검증된 저장 팔을 사용한다. 모든 개입과 비교가 신경 전용임을 표시한다.

## 사전 판정과 다음 결정

H264는 다음 조건의 conjunction이다.

- startup1..99 q19에서 `RMS(qZ-qB) <= .2*RMS(qX-qB)`. 비교 분모≤1e−12이면 판정 불가다.
- 같은 창의 가상 Z 오차 `error_B+(qZ-qB)` 최대절댓값≤.04rad. 이는 저장 기준 목표에 대한 명령 보호이며 실제 신체 자세의 보증은 아니다.
- R_Z≥1e−4 AND R_Z≥.8R_X. 신경 반응 크기를 보존하지 못한 단순 입력 약화는 지지하지 않는다.
- MN47329의 Z startup 평균이 기준 B로부터 멀어진 정도가 X보다 크지 않고, Z의 rate 범위가0보다 커야 한다. 각 팔 mean/min/max/zero/cap200 비율과 core24 차이를 함께 보고한다.

과학적 유효성, H264, 후보 채택, F1–F8 완료를 따로 판정한다. 지지할 때만 같은 보정의 짧은 실제 신체 대조를 별도 설계한다. 미지지면 상수 평균 보정 한 가지의 실패로 남기고 bias 크기/창/τ 탐색으로 반복하지 않는다. 이 결과로 동적 중심화가 자동 정당화되지는 않는다. 초기 명령·감각 반응이 유지돼도 기존 E150의 연속 전진/협응 문제를 해결했다고 주장하지 않는다.

## 구현·검토·자원

실제 gpt-6-sol/high가 tools/probe_c_banc_m3_lh_tonic_compensation.py와 tests/test_c_banc_lh_tonic_compensation.py만 구현한다. 두 입력 조건/시간축/whole 상태 복원과 CUDA 실행을 함께 다루므로 high를 사용한다. 부모 FIRST→필요시 같은 Sol 첫수정→부모 SECOND→남은 수정은 부모 순서다. E250 constructor·E254 원래 drive·E260 준비/관측 원리를 명시적으로 재사용할 수 있으나 원래 파일 수정·schema/global monkeypatch·의미 없는 재명명은 금지한다. 실제 필요한 W6 모델2개만 생성한다.

입력 hash와 상수 도출값을 freeze에 고정하고 source review 뒤 원본1/retry0/CPU1/GPU1이다. internal45(마감8)/external50/supervisor55초, host6GiB/추가GPU3GiB/출력256MiB/min disk1GiB/min GPU3GiB, body·physics·fit·solver0. E260의4모델/10000tick 원본24.0103초를 근거로2모델/8000tick에55초를 보수적으로 배정한다. 누적7082.651317799613초를 보존하고 미래 단계배정7101→7138(+37)로55.348682200387초를 확보한다. 사용자 전체 예산을 새로 정의하거나 과거 비용을 지우지 않는다.

출력은 verification/banc-natural-gait-m3-20260919/e264-lh-tonic-compensation/에 각 시작/끝 신경 state와 decoder bank, 발화·drive·적응·q19·상수 도출·CSR/실제backend·결과·오류·호출/부분 진행량·manifest를 보존한다. 준비/생성/부모 감사는 별도 비용이며 미측정 개발null. E150부분/E257기각/E261무효/M2/UI/F6미노출 및 동일 최종 후보의 F1–F8 미완료를 유지한다.
