# E316: 현재 단조 근육 변환의 functional 미분 연결

M3 finite-goal의 offline 학습 계산 단계다. E314–E315는 실제 전체 BANC의 처음 20tick exact와 1ms 국소 방향 미분만 통과했다. 자연 보행, 장기 미분, 학습 성공과 구분한다. E150의 반복 자체는 이미 있으므로 한 주기 안의 근육 동작을 개선하는 후속 학습을 준비한다. F1–F8, E150/E313 및 모든 이전 실패, M2·활성 UI·F6 미노출을 유지한다.

## H316와 판별

현재 E150의 MN→근육→관절 변환을 고정한 채 GPU의 functional 상태와 자동 미분으로 표현할 수 있다는 공학적 가설이다. 경쟁 설명은 고정 근육과 가변 근육의 서로 다른 상태를 합치거나, filter 갱신 순서·hinge 경계·float64 reduction 차이 때문에 실제 변환과 달라지는 경우다. 기존 E269 GPU decoder에는 조건부 source tangent가 있으나 현재 전체 SSP autograd 그래프에 연결되는 일반 VJP가 없어 이 인터페이스를 추가한다. 기존 kernel·운동계수·runtime은 수정하지 않는다.

구현 소유는 `tools/banc_m3_monotone_adjoint.py`, `tests/test_c_banc_monotone_adjoint.py` 두 신규 파일이다. Root 설계 → actual gpt-6-sol/xhigh 구현 → root FIRST → 필요한 같은 Sol 첫 수정 → root SECOND → 잔여 문제 root 수정 순서다. 현재는 생성 CPU 구현·검증만 허용하며 root 승인 뒤 생성 CUDA, 별도 pin/reservation 뒤 실제 기록 재생으로 순차 진행한다.

## 고정 수식과 상태

입력은 기존 순서의 391 MN rate, float64 bank(391×3), float64 original activation(96)이다. 신경에서 온 float32 rate를 float64로 변환하되 gradient 연결은 유지한다. 모든 기존 calibration/membership/kappa/slopes/LTM/motor IDs를 그대로 사용하고 학습하지 않는다. 매 control은 실제 `dt`를 받으며 기본 .005초, 유효 범위 (0,.05]다.

1. `pooled = pool @ rate64`; 원래 activation은 `A + (-expm1(-dt/.01))*(pooled-A)`로 96개 모두 갱신한다.
2. `bank += (-expm1(-dt/[.005,.01,.04]))*(rate64[:,None]-bank)`.
3. Variable muscle은 기존 support에서 `sum(low*bank)+sum((high-low)*max(bank-kappa,0))`다. LTM 또는 singleton인 fixed muscle은 갱신한 original activation을 쓴다. 같은 근육의 다른 상태를 합치지 않는다.
4. `q = rest + .02*(moment @ activation)`, `tendon = .2*tanh(activation[ltm]/10)`.
5. `motor_cut=True`는 bank/original/activation/tendon을 정확한 0으로, q를 실제 body neutral로 바꾼다. rate 및 이전 상태에 대한 gradient는 0이고 그 다음 재시작은 영 상태와 같다. 생성 API에는 몸체가 없으므로 neutral은 별도 고정 입력이다.

API는 `MonotoneDecoder(calibration, tau, kappa, slopes, membership, neutral, *, device='cpu')`와 `.step(rate, bank, original_activation, *, dt=.005, motor_cut=False)`다. 결과 dict의 고정 필드는 `bank`, `original_activation`, `activation`, `q`, `tendon`이다. 입력을 바꾸지 않는 functional 연산이며 모든 결과는 float64다. 실제 기존 constructor의 숫자·shape·anatomical support·fixed canonical slopes·per-muscle 0.5–2 및 MN slope floor를 검증한다. sorted unique motor IDs와 6 LTM rows를 보존한다. Constructor가 수치 배열을 소유 복사한다. 모델 생성·GraphStore·body·StateStore load를 module import/constructor에서 실행하지 않는다.

Float64 행렬 곱과 fixed mask로 계산하여 근육마다 Python GPU 동기화를 하지 않는다. CUDA의 장치는 index까지 해결하고 값 검사에는 device predicate를 사용한다. 실행 경로에서 CPU scalar 추출·CPU copy·per-muscle host 분기를 하지 않는다. 정확한 hinge `bank==kappa`에는 hinge derivative 0, 즉 low slope를 택한다. `dt`와 운동계수는 학습하지 않는다. 원래 float32 신경 반올림의 미분 한계는 E314와 동일하게 남는다.

## 생성 검증과 승인 조건

고정 생성 seed 31601. 불균등·섞인 anatomical membership, variable의 증가/감소 high slope, LTM와 singleton, 비영 bank/original state, 가변 dt, hinge 양쪽과 정확한 경계, cut/restart, 둘 이상의 살아 있는 rollout을 검사한다. CPU에서는 기존 `AnatomicalRecruitmentAdapter`+`MonotoneRecruitmentAdapter` 자체와 비교하고, 별도 scalar/NumPy 수식 및 유한차분으로 rate/bank/original gradient를 검산한다. Input mutation, invalid shape/dtype/negative/nonfinite/device 및 부적합 fixed/support를 거부한다.

Float64 forward 기준은 bank/original/activation 최대 절대차 ≤1e-10, q/tendon ≤1e-11이다. 정확한 cut/영 상태와 경계 derivative는 별도 equality로 판정한다. Hinge에서 떨어진 smooth 생성 사례의 gradient는 독립 f64 reference/FD와 상대 L2 ≤1e-6 또는 절대 ≤1e-8을 요구한다. 생성 CUDA에도 같은 기준을 적용하며 prepared step의 host scalar/copy 없음을 확인한다. 현재 SSP와 작은 생성 신경을 연결해 rate→decoder→q 역미분이 연결되고 유한한지 검사한다. 생성 통과만으로 학습·원본 동역학을 승인하지 않는다.

CPU Sol 전체 90초, root 전체 90초(각 CPU1/host2GiB), 생성 CUDA는 root 검토 뒤 단일 GPU 전체 45초/host2GiB/GPU추가512MiB/출력100MiB로 한정한다. 실제기록 재생은 코드와 입력을 pin하고 저장된 MN 출력과 native q를 비교하는 별도 root 계약 후 최대20초의 단회 감독으로 진행한다. 새 신경·물리 step·신체·fit·teacher·holdout는 모두 0이며 원본 동역학 원장은 갱신하지 않고 저장 원본 분석의 실제 wall을 기존 원본 누적에 한 번만 더한다. 현재 누적8501.121034199608/8560초·잔여58.8789658003916초·예약0, 증액0이다. 구현 실패/최초 검증 실패/비용은 보존한다.

## 후속 결정

Forward 또는 독립 gradient 미달은 먼저 수치 결함을 고친다. 통과하면 현재 모델의 시작 조건 제약과 위상 독립적인 운동 형태 목적함수를 사전 고정한다. E315의 premotor 평균 상승이나 decoder proxy의 감소를 보행 개선으로 승격하지 않는다. 후속 학습의 실제 구간·finite-size direction·원본 startup/short 판별은 새 계약에서 정하며 이번 단계에는 포함하지 않는다.
