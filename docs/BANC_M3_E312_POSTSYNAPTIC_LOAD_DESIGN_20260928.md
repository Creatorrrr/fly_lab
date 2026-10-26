# E312: 수신 뉴런의 활동 부하에 의존하는 재귀 전달

Root 사전 설계, 2026-09-28. E310의 presynaptic resource branch와 E311의 고정 E277 decoder branch를 확대하지 않는다. E198/E208은 signed current를 단일 값으로 합산하는 현재 가정을 검토할 근거이며 새로운 생리 기전의 증거는 아니다. E209–E218의 MN 자기입력 기저, E224 병렬 effector, E242–E244 재적합과 구분하여 **실제 premotor 재귀 상태 생성**에 비분리적 수신 상태를 넣는 한 공학적 가설을 검증한다.

실측 막전위·반전전위·수용기 시간상수가 없으므로 conductance-based 생리 모델을 복원했다고 부르지 않는다. 단위가 확인된 기존 current와 threshold의 비로 정의한 **postsynaptic divisive load**만 구현한다. 새 reversal voltage나 측정되지 않은 세포 유형을 만들지 않는다. 생물학적 shunting의 정답이라는 주장은 하지 않는다.

## 고정 기전

전체 E150 해부학적 graph·가중치·부호·기존 neuron r/z·모터·단조 모집·신체·감각 경로를 유지한다. 비교 적분기는 E307에서 startup을 통과한 SSPRK33이다. 대상 S는 기존 E289의 P7136→P7136 nonzero725849 간선 전체이며, core 보호라는 이름으로 일부를 추가 배제하지 않는다. 기존 premotor와 MN 집합은 서로 겹치지 않는다. Source resource u는 전부0/R1이며 E306/E309의 active resource와 결합하지 않는다.

각 대상 postsynaptic 뉴런 i에서, 실제 같은 SSP stage의 live masked rate로 다음을 계산한다.

- U_i = sum over selected incoming edges of abs(w_ij * masked_rate_j). 원래 부호가 같은/다른 모든 선택 입력의 총량이다. E와 I 합산 전 총량을 쓰지만 원래 net current의 부호는 유지한다.
- L_i = U_i / (threshold_i + U_i). 원래 threshold_i > 0이어야 한다. 이는 무차원 [0,1) 부하이며 입력 전류의 단위와 threshold 단위가 같다.
- dh_i/dt = (L_i − h_i) / tau_i, h_i(0)=0. tau_i는 기존 해당 뉴런의 실제 membrane tau32를 f64로 변환한 값이다. 새로운 시간상수·시간 gate·위상·teacher가 없다.
- g_i = 1 / (1 + lambda * h_i). Neutral lambda=0, 단일 후보 lambda=1만 허용한다. 0≤h≤1이면 .5≤g≤1이며 sign이나 topology를 바꾸지 않는다.
- 선택 간선의 원래 packed f32 product에 g_i를 곱하고 동일 warp CSR 순서로 전체 입력을 합산한다. 비선택 간선은 원래 product다. 따라서 수학적으로 J_i = J_nonselected,i + g_i*J_selected,i이며 실제 f32 식과 차이를 숨기지 않는다. lambda0는 원래 표현을 직접 사용하여 neural r/z/mask/tick이 기존 SSP/u0와 bytes exact여야 한다.

U는 shunting을 적용하기 전 원래 선택 product의 절댓값이다. 같은 net current라도 총량·과거 수신 상태가 다르면 반응이 달라진다. 이는 source별 독립 R이 아니라 여러 presynaptic 신호가 공유하는 postsynaptic 상태다. E/I별 독립 receptor state나 막전위 방정식을 구현한 것으로 표현하지 않는다. core rate·startup 보존은 주장하지 않고 이후 실제 검사한다.

## 수치·상태 계약

기존 dt=.00025와 SSPRK33의 세 단계: Euler, .75*base+.25*Euler, (1/3)*base+(2/3)*Euler. r/z는 원래 E303 f32 연산, h는 no-FMA f64로 같은 old stage U와 h를 사용하고 neuron overwrite 전에 갱신한다. h Euler는 `(1-q)*h+q*L`, q=dt/f64(tau32). dt≤min(tau)/2와 기존 adaptation 제한을 검사한다. clipping/projection 없이 모든 stage에서 finite/r,z≥0 및 0≤h≤1를 검사한다. 원래 resource는 u0/R1만 허용하고 active u는 생성·복원 모두 거절한다.

Gross U의 각 term과 warp32 합산은 원래 f32 FTZ 순서, 마지막 값만 f64로 변환하여 L/h/g를 계산한다. g는 f64→f32 후 원래 선택 f32 product에 추가 곱한다. lambda0에서 추가 곱을 생략한다. CUDA pointwise는 원래 add/mul/div/tanh 계약 및 no FMA, GPU kernel은 allocation/cpu-copy 없이 captured graph에서 동작한다. 오류 flag는 sticky, 실제 stage 배열은 교정하지 않는다. 원래 양수 subnormal/음수 subnormal을 임의의0으로 대체해 validation을 우회하지 않는다.

새 schema/precision/model identity에는 원래 model identity, selected CSR 위치·pre/post/weights, premotor/MN 순서, 대상 postsynaptic ID, threshold/tau32 원본 및 lambda가 들어간다. CSR 내용과 captured pointer 무결성을 검증한다. 새 h와 lambda/선택 metadata는 snapshot/disk/future에 포함하고, 불완전·다른 schema·dtype·순서·변경 metadata·out-of-range state는 live state 변경 전에 거절한다. reset은 h0 및 오류0, motor/source mask는 원래 의미를 보존한다. 기존 E303 schema/원본을 새 모델로 몰래 복원하지 않는다. Source snapshot 전체를 이식하는 factory는 원래 동적 상태를 명시적으로 확인한다.

## 구현·리뷰와 이번 범위

Root 설계 → 실제 gpt-6-sol/xhigh 구현 → root FIRST → 필요 시 같은 Sol R1 → root SECOND → root 잔여 수정. CUDA fused CSR, coupled SSP, metadata/복원 난이도로 xhigh를 사용한다. 소유 파일은 `tools/banc_m3_postsynaptic_load.py`, `tests/test_c_banc_postsynaptic_load.py` 두 개다. 기존 E303/MotorTimescale 코드·평가기를 수정하지 않는다. 기존 validated 코드의 상속/재사용을 우선하고 숨은 global monkeypatch를 금지한다.

필수 생성 검사는 2-sign net-equal/gross-different와 h-history 차이, 선택/비선택 경계·mask·warp32 초과 행, lambda0 기존 SSP exact, 같은 stage 독립 scalar reference, 양수·음수 subnormal validation, h positivity/sticky fault/no clipping, 중간/nonzero snapshot·disk/future exact, 손상 복원 atomicity 및 zero/reset/identity를 포함한다. 기존 신경 단독 생성 constructor와 body/F6를 구분한다. 새로운 모든 생성 CPU 시도 합80초/CPU1/host2GiB로 제한하고 raw 실패·지원 읽기·미계측 비용을 보존한다. 구현 agent는 실제 CUDA/원본 graph/신체/solver/fit/원본 수치분석을 실행하지 않는다.

Root source review 뒤 별도로 생성 CUDA qualification을 준비한다. 최소 39cell/warp 초과 sparse fixture에서 독립 scalar oracle, direct vs graph, lambda0 SSP neutral exact, 실제 nonzero disk future와 invalid-state rejection을 검사한다. 한 root 감독 프로세스/재시도0/60초, host2GiB/additionalGPU512MiB/output16MiB로 제한한다. 이는 생성 검증이며 전체 BANC 원본 비용과 분리한다. 소스 검토 전 실행하지 않는다.

## 실제 보행으로 이어지는 결정

이 단계 통과는 runtime qualification일 뿐 후보 채택이 아니다. 통과 뒤 별도 E313 계약으로 원래 neutral에서 같은 SSP/u0와 단일 lambda1을 비교한다. 고정 원래 .5초 startup을 먼저 검사하고, 통과한 동일 후보만 1.5초 short로 연속 진행한다. 실제 U/h/g·core/MN·bank·activation·contacts와 whole provenance/counters를 저장한다. 기존 neural-only exact와 신체 전체 F2는 구별한다. Startup 실패 또는 short의 지속 bout/여섯 완전주기·기존 전진/slip 미달이면 이 고정 후보를 기각한다. Parameter grid·사후 neuron 선택·교사 외부 위상·몸 root 보정은 하지 않는다. 좋은 short 결과일 때만 동일 후보10초/F1–F8을 다음에 설계한다.

현재 원본 누적8416.895937599536/8560초, 잔여143.1040624004636초이며 이번 E312 원본 예약0이다. E313 실제 원본은 이후 source gate와 사전 예약 전에는 실행하지 않는다. 원래 startup/F1–F8/M2/UI/F6 미노출과 모든 과거 실패·비용을 보존한다. 목표는 active·미완료다.
