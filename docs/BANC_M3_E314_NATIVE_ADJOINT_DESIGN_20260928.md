# E314: 현재 SSP 신경 방정식의 학습 경로

상태: 구현·생성 검증 허용, 원본 실행 예약 0. 원래 finite-goal/F1–F8은 그대로 active이며 본 단계 통과는 자연 보행 성공이 아니다.

## 판단을 바꾸는 근거

E313 λ1은 실제 신체 .2156초에 upright .9 미만으로 기각됐다. 입력을 일괄 감쇠하거나 λ를 탐색하지 않는다. E150은 리듬 자체가 없는 모델이 아니다. E152의 전체 창에는 다리별 21–22회 반복이 있고, 46회 전진 중단과 최장 bout .215초 때문에 정식 적격 주기가 0이다. E163의 후기 source 위상 결집도 높다. E73은 다른 source·전달 계통에서 개발 시작의 F4/F5를 만족했으나 원형 경로로 F3를 실패했다. 이들을 합쳐 성공하거나 시정수 하나의 인과 효과로 해석하지 않는다.

다음 학습의 목적은 이미 있는 반복을 늘리는 것이 아니라, 한 주기 안의 근육 동작을 바꾸어 지지 중 제동을 줄이면서 startup을 보존하는 것이다. 기존 E107 형태 대조와 E152–E163 접촉 진단을 활용하고 같은 저장자료 분석을 별도 캠페인으로 반복하지 않는다. E269–E287의 기록 source를 고정한 전달 적합과 달리 전체 신경 되먹임에 대한 미분이 필요하다. E29의 옛 E20/RK4/단일 필터 코드는 현재 E150/SSP/근육별 adaptation 및 단조 모집과 다르므로 그대로 학습하지 않는다. E24/E25에서 큰 장기 형식 미분이 실제 float32 작은 개입을 예측하지 못했던 실패도 유지한다.

## 구현 경계

고정 원래 CSR의 기존 선택 간선에 양수 공유 효능을 적용하는 sparse current와, 현재 SSPRK(3,3)의 functional forward/역미분을 구현한다. 전체 신경을 계산하며 새 node/edge, 외부 시계·위상·CPG·기록 명령 또는 몸체 보정은 만들지 않는다. 이 도구는 offline 학습용이다. 실제 제품/UI, 기존 E312/E313 소스와 원본 증거는 수정하지 않는다.

첫 그룹 정의는 기존 premotor→premotor anatomical mask를 전달할 수 있는 일반 입력 인터페이스다. 학습 mask와 최종 물리 목적함수/실험 길이는 후속 원본 계약에서 동결한다. 이번 생성 시험에는 작은 임의 CSR만 쓴다. 순수 oscillation loss, 교사 절대 위상 또는 12Hz 일치는 채택 기준으로 도입하지 않는다. 실제 gait 제약을 받는 학습 전에 현 모델의 forward와 유효한 국소 미분인지 검증해야 한다.

### Sparse current 계약

`GroupedCurrent(weights, edge_ids, group_ids, group_count, device)`와 `current(masked_rate, efficacy)`를 제공한다. weights는 정규화된 sorted CSR float32 원본이며 행렬을 변형하지 않는다. 선택은 unique int64 기존 nonzero positions, group ids는 contiguous 0..G-1이다. efficacy는 finite positive float32 또는 float64 tensor이다. float32로 한 번 변환한 `theta32`를 쓰며 선택 weight는 `mul_exact(original_weight32, theta32[group])`로 계산한 뒤 원래 CSR FTZ product/reduction 순서를 적용한다. theta32==1인 경우 원래 weight를 그대로 쓰므로 neutral 원래 packed CSR와 exact다. 기존 base lookup과 간선별 group을 사용하여 그룹별 새 weight의 unique 수가 packed 14-bit 한도를 넘는 문제를 피한다. CPU forward는 같은 32-lane/FTZ 규칙의 작은 scalar oracle을 사용한다.

역미분은 rounded computation 자체의 수학적 도함수라는 주장을 하지 않는 **실수 방정식의 surrogate VJP**이다. rate VJP는 같은 theta32를 반영한 W(theta)^T, efficacy VJP는 선택 간선별 upstream_post×original_weight×masked_rate_pre의 그룹합이다. 저장된 tensor version은 forward 이후 변경을 감지해야 하고, 서로 다른 두 효능의 살아 있는 autograd graph가 간섭하지 않아야 한다. float64의 deterministic 그룹 reduction을 선호하며 구현이 비결정적이면 이를 명시하고 오차를 검사한다. 미래 방향은 finite-size native 개입으로 별도 검증한다. 이 단계에서는 solver/fit 0이다.

### SSP functional 계약

`SSPAdjointModel(operator, tau, coefficient, threshold, cap, gain, inverse_adaptation_tau, dt)` 및 `step(rate, adaptation, drive, efficacy, mask)`를 제공한다. 모든 physiology는 현재 resolved per-cell float32 배열이며 갱신 대상이 아니다. h/tau<=.5, nonnegative gain/inverse, finite positive tau/cap, 유효 shape/device를 검증한다. 상태를 in-place 변경하지 않고 `(new_rate,new_adaptation)`를 반환한다. mask는 rate에 먼저 곱하고 각각의 stage에서 재계산한다. drive는 control 동안 고정하는 caller 입력이다.

세 stage는 원래 `(base_weight, work_weight)=(0,1),(.75,.25),(f32(1/3),f32(2/3))`. 원래 `drive+current-threshold-z`, coefficient, matching_tanh/clamp, q_mem/q_adapt, Euler/convex 순서와 pointwise no-FTZ/no-FMA를 CUDA kernel에서 유지한다. CPU는 동일 float32 scalar 순서로 계산한다. 역미분은 cap*tanh의 양수 활성 구간 미분, q_mem/q_adapt, base-state 두 경로와 모든 재귀 current 경로를 전달한다. rectifier의 정확한 kink는 0 subgradient로 정의한다. native 적응은 `gain*r`이며 gain이 작다고 운동 출력이 차단된 것으로 해석하지 않는다.

전체 adjoint의 장기 신뢰성 또는 신체 미분을 이번 검증으로 주장하지 않는다. 단조 decoder까지 확장하기 전 이 핵심 경로가 통과해야 한다. 신경 모델 저장/제품 wrapper는 본 단계에 추가하지 않는다.

## 생성 검증과 자원

Root 설계 → actual gpt-6-sol/xhigh 구현 → root FIRST → 필요한 첫 수정은 같은 Sol → root SECOND → 잔여 수정 root. sparse(2파일)와 SSP(2파일)는 소유를 나누고 인터페이스만 공유한다.

- CPU generated: 39-neuron CSR(32-lane 이상 행·부호·비선택·mask·여러 group), 선택 중립 exact, dense float64 독립 VJP/finite difference(매끄러운 영역), 여러 효능 graph·변경 거부·nonfinite/shape 검증. 실제 원본/BANC/GraphStore/body/solver 0.
- CUDA generated: 한 GPU 작업만, neutral current와 실제 `SSPRecoveryResourceNetwork` u0의 같은 작은 모델 step/여러 step byte parity. nonneutral current와 pointwise scalar oracle 및 adjoint/dense tolerance를 따로 검증. 생성 원본 모델은 아님. CSR subnormal FTZ와 pointwise 보존을 구별한다.
- 자원: 구현 에이전트별 CPU 생성 검사 wall 합계 90초, root CPU 생성 90초. CUDA는 root의 소스 승인 뒤 공유 60초/host RSS 2GiB/추가 GPU allocation 1GiB/출력 100MiB. 시험 실패·timeout도 사용량에 포함하고 자동 재시도하지 않는다. 읽기·편집 비용은 도구 반환값으로 기록하며 과거 원본 누적 8495.160237499575/8560초와 잔여64.83976250042542초는 그대로다.

중립 exact 또는 의미 있는 독립 VJP가 미달하면 실제 학습을 실행하지 않고 수치 구현 결함을 수정한다. 통과하면 한정된 실제 E150/SSP 입력의 neutral forward와 finite-size directional response를 후속 계약으로 동결한다. 그 결과가 지지할 때만 startup 제약과 실제 stance 전환의 목표를 가진 한 학습 시도를 허용한다. 모든 F1–F8/M2/UI/F6 미노출/이전 기각과 비용을 보존한다.
