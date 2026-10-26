# E255 연결 복원과 감각 이력의 교차 대조 — 2026-09-25

E254는 MN6373의 초기 미관측 입력 증가가 기존 intrinsic-timescale 회로의 row135246(INXXX466)에서 주로 발생함을 보여 주었다. 원자료 주석은 ACH 예측/좌측 intrinsic single_leg_neuromere이며 특정 LH 기능이나 생물학적 리듬 기능을 입증하지 않는다. 현재 구현에서는 기존 original24/core_adaptation/intrinsic_timescale38에 포함돼 있다. 기존 두 arm은 연결과 신체에서 돌아온 감각 이력이 함께 달라졌으므로 두 요인을 분리한다.

## 고정 비교와 가능한 결론

W0=E251 baseline CSR, W1=25개 기존 감각→bridge 연결을 복원한 candidate CSR. U0/U1은 각각 baseline/candidate의 저장 old_filtered와 원래 순서로 생성한 drive다. 같은 control0 동적 상태에서 각 조합을 control100까지 0.5초 재생한다. 저장 입력을 고정한 신경 반사실 대조이며 새 신체 폐루프나 자율 보행 검증이 아니다. 입력 U1을 W0에 넣는 조합의 물리적 실현 가능성은 검증하지 않는다.

한 프로세스에서 두 CUDA 모델을 순차 생성한다. W0 모델: 00 재생 후 정확한 시작 snapshot으로 복원하고 01 재생. W1 모델: 11 재생 후 같은 모델의 시작 snapshot으로 복원하고 10 재생. 각 arm은100controls×20tick이다. 모델 간에 identity metadata를 교환하지 않는다. CSR와 packed CUDA backend/설정/최초 전체 snapshot은 기존 E250 constructor 검증을 재사용한다. 원래 두 시작 상태의 동적 배열과 tick/dt/mask가 같고 예상 identity 차이만 존재함을 확인한다.

00/11은 E254의 처음101 endpoint의 union1236 rate/drive/adaptation, 5MN actual-current, edge-current64, 100개 drive-values 모두 bitexact여야 한다. 01/10은 네트워크를 실제로 전진시키며 자신의 전체 초기/최종 상태와 raw를 남긴다. control100 전체망의 과거 reference는 없으므로 새 final의 전체동등성을 주장하지 않는다. 두 diagonal이 모두 통과해야만 효과를 판정한다. 시작과 마지막 관측 전후 snapshot exact 및 모든 clock/finite/마스크/원본 입력 주입을 확인한다. 실패 시 이미 실행한 일과 부분 raw/최종 snapshot을 보존하고 판정 unavailable이다.

관측/수식은 E254와 같은 고정5MN 모든incoming1741개, union1236, 실제 _current와 float64 edge 항을 사용한다. 101endpoint×4=404 current probes, 추가 neural tick0. 원래 old_filtered(float64)→float32 scatter→descending→navigation 순서를 유지한다. 00/10은 U0, 01/11은 U1이어야 하며 생성 drive의 일치를 raw로 확인한다. 출력 상태 관측과 원본 참조 비교는 서로 구분한다.

## 고정 가설과 분해

각 endpoint 변수 Y에 대해 T=Y11−Y00, C=Y10−Y00, S=Y01−Y00, I=Y11−Y10−Y01+Y00를 float64로 계산한다. T=C+S+I를 모든 행에서 검산한다. 순서가 다른 조건부 효과 C1=Y11−Y01, S1=Y11−Y10 및 두 순서의 평균도 함께 보고한다. 비선형 상호작용이 있으므로 C/T를 유일한 인과 기여율로 부르지 않는다.

H255의 단일 대상은 E254에서 고정한 CSR444229, source135246→MN6373다. startup endpoint1..99에서 edge-current64의 signed 합 T와 C를 사용해 T>0, C>0, abs(T−C)≤0.2*abs(T)일 때만 `H255_connectivity_effect_retained`를 지지한다. 이는 원래 감각 이력에서도 전체 관측 증가의80% 범위가 연결 변경으로 유지되는지에 대한 개발 기준이며 생물학적 기준이 아니다. 결과를 보고 대상/창/문턱을 바꾸지 않는다.

추가로 source135246/bridge20156/bridge111970/MN6373/MN22787의 rate/drive/−adaptation, 고정5MN의 actual current와 float64 current/margin, edge20156→22787과444229를 같은 분해로 저장한다. startup1..99와case64/84를 보고하며 steady 이후로 일반화하지 않는다. 양·음 항은 endpoint별로 분리한 뒤 합산한다. 가설 판정은 bool과 원래 T/C/S/I 수치를 같이 제공한다.

지지하면 기존 회로의 연결 복원이 만든 직접적인 초기 출력 변화에 맞춰 감각-회로 인터페이스 교정을 설계한다. 미달이면 S/I와 두 순서 효과를 근거로 움직임에서 돌아온 감각의 증폭을 먼저 검토한다. 어느 결과도 gain 격자, 임의 간선 삭제, 전 다리 복원, 신체 재실행 또는 M3 채택을 자동 실행하는 근거로 쓰지 않는다. 다음 변경은 목표와 비교를 다시 동결한다.

## 구현과 자원

부모가 설계하고 실제 gpt-6-sol/high가 `tools/probe_c_banc_m3_lh_crossed_input.py`, `tests/test_c_banc_lh_crossed_input.py`만 구현한다. E254 prepare/drive/incoming/관측 및 E250 constructor를 재사용하되 고정301행/CONTROLS300/helper finalizer를 E255에 잘못 적용하지 않는다. 전역 상수 monkeypatch 금지. FIRST→필요시 같은Sol 첫수정→부모SECOND→남은수정부모. 생성 검사로 네 조합 라우팅/전체상태 reset/diagonal 참조 행·shape/분해 항/엄격 경계·signed 상쇄/관측 비변경과 실패 뒤 snapshot·ledger·success 철회를 검증한다. 부모 승인이 원본 전 필수다.

원본1/retry0/CPU1/GPU1, constructor2/setup실행120 중capture40/과학8000tick/404probes, 신체·physics·fit·solver0. internal38초(마감6)/external43/supervisor45초. host4GiB, 추가장치GPU3GiB/최소GPUfree3GiB, 출력256MiB/최소disk1GiB. 원본 E254는20.83초에12000tick/602probes를 끝냈고 이번은8000tick/404probes와 상태 저장이 조금 더 있으므로 이 한도를 선택했다. 기존 누적6877.641533999635/6933초·잔여55.358466000365초에서45초예약, 증액0.

출력 `verification/banc-natural-gait-m3-20260919/e255-lh-crossed-input/`에 네 arm raw/전체시작·끝/동적reset검사, CSR/incoming/drive/parameters binding, constructor/arm별 requested-entered-completed·unknown partial, errors/result/manifest를 원본 write-once 보존한다. 요청 수를 완료 수로 세지 않으며 cleanup/publication 오류에서 성공을 철회한다. 부모는 원본 E254/각drive/실제CSR/float32 current/분해 수식을 독립 재계산한다. 모든 개발/생성/감사 비용은 별도, 미측정은null이다. E251기각/E150/M2/UI/F6/기존실패/F1–F8 미완료를 유지한다.
