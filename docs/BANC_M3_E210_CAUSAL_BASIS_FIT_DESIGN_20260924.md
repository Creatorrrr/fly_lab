# E210: 고정 인과 반응 기저의 여섯 다리 조건부 적합

E209의 실제 signed native 입력에서 CUDA 기저 계산과 독립 검산이 통과했다. 아직 학습된 계수·새 native 신경 모델·개선된 보행은 없다. 이 실험은 기록된 입력과 고정된 기저에서 beta를 적합하는 한정 학습이며, 전체 BANC–신체 피드백의 성과를 대신하지 않는다. F1–F8, 원래 E150/M2, UI, 실패 이력과 미사용 F6 상태를 보존한다.

초안과 실제 gpt-6-sol/medium 읽기 감사를 검토하여 여섯 다리의 독립 문제를 선택했다. 원시 전역 행렬 76608×5488을 만들지 않는다. 다리별 startup RMS 제약이 원래 전역 startup 제약보다 엄격한 충분조건임을 명시한다. 실패를 원래 전역 문제의 불가능성으로 주장하지 않는다.

## 고정 자료와 변수

E209 scale/prototype/폭/16ms 기억/gate, B/H, original tau/cap/초기값과 10ms canonical decoder를 고정한다. 원래 소유 근육과 moment를 바꾸지 않는다. 자유 MN343×16, 총 beta5488개만 학습하며 각 beta는0..1이다. fixed48 LTM MN의 조건부 rate는 원래 E201값이고 D=0이다. 기존 MN의 재선택이나 약한 column 제거, 새 trust ball, signed correction 변수변환을 하지 않는다.

LF/LM/LH/RF/RM/RH의 변수 수는976/864/896/992/880/880이다. 각 다리 소유 MN의 D와 `.02*moment[:,membership]`로 해당7관절의 `A[t,j,m,k]`를 구성한다. 모든 근육의 해당 다리 밖 moment가0인지 확인한다. 교차 다리 영향이 있으면 문제 분해를 거부한다. 다리와 MN/기저의 순서는 원래 고정 순서를 유지한다.

목표는 E197에 저장된 기존1824개 row100..1923, velocity1823쌍100→101..1922→1923, startup99개 row1..99다. row0은 원래 held command다. 각 후보의 q는 `q_intercept+A*beta`이고 velocity는 동일한 두 endpoint의 차이/.005다. 원래 baseline RMS도 동일 고정 자료에서 계산한다. 새 목표 창·위상 이동·시간 재정렬은 없다.

## 한 다리의 볼록 문제

각 다리를 LF,LM,LH,RF,RM,RH 순서로 최대 한 번씩 풀며 총 solver 호출은 최대6회다. 한 다리의 q residual12768개를 `(원래 해당 다리 qRMS*sqrt(12768))`로 나눈 `C*beta+c`에 대해 `||C*beta+c||²+1e-8*mean(beta²)`를 최소화한다. 물리 beta를 직접 변수로 사용한다. 목적함수 상수도 저장해서 보고된 scalar objective를 직접 검산한다.

제약은 다음과 같다.

- beta0..1. 사후 clip/계수 보정을 하지 않는다.
- 해당 다리 qRMS와 velocity RMS가 원래 해당 다리보다 작거나 같다.
- 해당 다리 startup RMS<=.018, 모든99×7 startup 절대오차<=.04.

q/velocity/startup 및 startup 점별 수치 제약의 bound에는 `PAD=1-1e-6`을 곱해 반올림 여유를 둔다. beta box에는 이 PAD를 적용하지 않는다. 이 여유와 다리별 startup 제한은 원래 조건보다 엄격하므로 미해결/불가능 결과를 전역 불가능 증거로 승격하지 않는다. 다리별20% q 제한은 부과하지 않는다. 원래 전체20% 개선은 여섯 결과를 합친 후 별도 검사한다.

q와 velocity norm은 각 다리의 augmented Gram을 사용한 고정된 수치 압축으로 SOC를 구성할 수 있다. `tools/fit_c_banc_m3_constrained_tangent.py::compress_norm`의 PSD 잔차 검사를 읽기 재사용한다. 최소 고유값이−1e-10*max(1,max_eigen)보다 작거나 재구성 Frobenius 상대오차>1e-10이면 중단한다. 양수 고유값을 임의 threshold로 제거하지 않는다. 693행 startup norm은 원시 행을 그대로 쓰며 불필요한 큰 전역 Gram을 만들지 않는다. SOC는 `A_solver*beta+s=b, s∈SOC`에서 `[0;-matrix]`, `[limit;offset]` 규약이다. solver P는 대칭 목적 이차행렬의 upper triangle이며 선형항·상수·정규화를 함께 저장한다.

Clarabel 실제1CPU, max_iter150, tol_gap_abs/tol_gap_rel/tol_feas=1e-8, time_limit=20s/다리다. 그 외 버전 기본값을 기록한다. 각 상태/Solved 여부/반복/시간/primal·dual residual·원래 beta/슬랙·dual 및 모든 solver 입력을 즉시 저장한다. solver 예외·제한·미수렴·불가능을 숫자0이나 후보로 대체하지 않는다. 독립된 다른 다리 문제는 정해진 순서로 계속할 수 있으나 원본 재시도와 설정 변경은 없다. 새 호출 전 내부 시간 잔여가20초 미만이면 남은 호출을 차단한다.

## 실제 후보 판정과 대조

모든 다리에서 Solved와 유한한 반환값이 있어야 완성 후보를 만든다. 반환된 raw beta가 물리 box0..1 밖이면 사후 clamp하지 않고 후보 미달로 기록한다. 직접 H+B*beta에서 자유343 rate를 만들고 fixed48은 원래 rate를 유지한다. 이 rate를 실제 순차 canonical decoder로 전달한 q와 affine q의 차이가atol2e-12/rtol2e-12를 만족해야 한다. 고정 LTM activation/tendon은 원래 canonical 계산과 exact여야 한다.

solver 입력 A/b를 사용한 primal cone 검사와 원시 q/velocity/startup 데이터의 직접 검사 모두 수행한다. solver 수치 유효성 residual 허용은 정규화된1e-6이며, **후보 통과는 원래 실제 숫자 기준**으로 따로 판단한다: 전체 qRMS<=.8*0.056686385349786386, 전체 및 다리별 velocity/다리별 q 비악화, startup qRMS<=.018/max<=.04, tendonRMS<=.009/max<=.025. 추가 학습 조건도 직접 검사해 위반을 숨기지 않는다. 이 proxy는 F1–F8의 대체나 자연 보행의 필요조건이라는 주장이 아니다.

완성된 계수가 있으면 통과 여부와 관계없이 두 신경 출력(원래/새 반응)을 두 decoder(E150 monotone/canonical)로 읽은2×2 대조를 저장한다. 원래 rate+monotone과 원래 native q의 수치 잔차를 별도 측정하고 canonical 변경을 반올림이라고 부르지 않는다. monotone 비교는 저장된 초기 bank/activation을 유지한다. 후보 채택과 실제 native 모델 통합은 이 실험에 포함하지 않는다.

모든 원시 cone/목표/scale/해부학/계수/명령·startup residual·실패·manifest와 비용을 남긴다. 조건부 후보가 나오면 별도 native source 설계에서 실제 signed 입력과 기억을 실행 때 계산하며 외부 time/phase/teacher를 주입하지 않는다. fixed48은 기록 rate 파일 재생이 아니라 현재 입력에서 원래 식을 적용한다. 전체 BANC–신체10초 검증을 거쳐야 새 보행 개선으로 평가한다.

## 구현·검토·자원

부모가 설계하고 실제 gpt-6-sol/high가 새 `tools/fit_c_banc_m3_causal_neural_basis.py`, `tests/test_c_banc_causal_neural_basis_fit.py`만 구현한다. 고정 기저의 leg-local 이차/SOC 문제이므로 새로운 미분 신경 kernel 없이 high를 사용한다. 부모 FIRST→필요시 동일Sol 첫수정→부모SECOND/잔여수정→원본1회 순서다. 기존 구현과 freeze를 수정하지 않는다.

생성 수학/모의 시험은 SOC 부호·물리 beta box·Gram 압축·소유권·표본 시간축·실패 후 raw 보존을 확인한다. 부모 FIRST 뒤 작은 실제 Clarabel 대조는 하나의 생성 문제에 solver1회만 실행한다. 알려진 제약 경계의 해를 독립 계산하여 비교하고, CUDA/전체BANC/신체는0이다. 원본 데이터의 적합은 source gate 이후1프로세스에서 최대6solver 호출만이다. 사전 비용 점검은 원래 입력 크기·할당 예정 byte와 기존 E206 기록의 읽기 범위에 한정하며 원래 목표의 예비 적합을 하지 않는다.

원본 CPU1/GPU0/process1, 내부180/solver합120/외부220/감독240벽초, RSS8GiB/출력768MiB/free disk4GiB다. E209까지 누적5516.421354399707초를 보존하며 단계 배정5624→5784(+160초), 잔여267.578645600293초에서240초를 사전예약한다. 사용자 전체 예산을 새로 만들지 않는다. 생성·준비·부모 리뷰는 별도 계상하고 미측정을0으로 간주하지 않는다. 중단 시 완료 다리와 미실행 다리를 구분하고 원본 자동 재시도는0이다. F1–F8 active·미완료 상태를 유지한다.
