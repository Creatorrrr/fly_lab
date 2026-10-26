# E197: 운동신경 모집 문턱을 근육 적분 앞에 배치

부모 설계, 2026-09-24. M3 finite-goal과 F1–F8을 유지한다. E196은 감각 전달을 증가시켰지만 지속 전진을 개선하지 못했다. 현재 질문은 같은 운동신경 이력에서 짧은 문턱 초과를 보존하는 전달 상태가 부족한 명령 파형을 만들 수 있는가다. 기존 E148/E169의 `hinge(EMA(rate))`를 `EMA(hinge(rate))`로 바꾼다. gain, 문턱, 시간상수 탐색이 아니다. 성체 근육 생리가 확인됐다는 주장도 아니다.

actual gpt-6-sol/high의 M2/M3 소스 감사에서 이 정확한 함수는 찾지 못했다. M2 E27/E64/E72의 joint tanh 순서, rise/fall 상태, per-unit saturation은 다른 함수다. E168/E170/E175의 고정 70ms resource 계열도 다르다. E107 shape-only 성공과 E163의 리듬 유지에도 반복되는 전진 중단이 선택 근거다. 경쟁 설명은 이 문턱 이력으로도 현재 신경 출력의 부족한 파형을 표현하지 못한다는 것이다.

## 개입과 원본 자료

E164/E169와 동일한 E150 control0..2000, 391MN, 해부학 소유권, 고정19/가변77근육, 가변330MN, kappa391, tau=[.005,.01,.04]초를 사용한다. 양수 2branch×3filter 전달, rest, moment, .02 gain, LTM과 singleton의 원래 활성은 그대로다. E164 freeze의 입력 파일과 원래 snapshot/clock/bank 재구성을 확인한다. 기존 결과와 실패를 변경하지 않는다.

각 MN i와 filter k의 상태 L,H는 0에서 시작한다. row0은 advance하지 않는다. n=1..2000에서 해당 행의 rate를 사용하고 alpha_k=-expm1(-.005/tau_k)로 다음을 계산한다.

```
L[n] = L[n-1] + alpha * (min(rate[n], kappa) - L[n-1])
H[n] = H[n-1] + alpha * (max(rate[n] - kappa, 0) - H[n-1])
a_m[n] = sum_owned_filters(w_low * L[n] + w_high * H[n])
q[n] = rest + .02 * moment @ a[n]
```

고정19개 활성은 원래 decoder의 recorded original_activation을 사용하며, 이것은 미래 runtime에서 원래 adapter의 현재 활성으로 공급해야 한다. 가변 근육에는 recorded activation/command 보정을 넣지 않는다. 특히 `q_recorded + B_new*(w-w0)`는 실행 불가능한 시간별 offset이므로 금지한다. tick0의 실제 pre-decode activation/command는 보존하고 decoder 식 비교는 tick1부터 한다. row0 rate를 적분하거나 두 bank를 현재 rate로 초기화하지 않는다.

L+H가 원래 raw-rate EMA와 일치해야 한다. 생성 검증에서는 상수, 문턱 교차 pulse, kappa0, 변화하는 실제 dt, prefix 독립성을 검사한다. low=high일 때 기존 선형 bank와 같은 활성인지는 확인하되, 기존 E148의 서로 다른 low/high w0가 새 함수의 baseline이라는 주장은 하지 않는다. 실제 L/H와 old post-filter basis의 차이, 같은 w0에서의 명령 변화도 저장한다. 단순 이름 변경은 허용하지 않는다.

## 한 번의 조건부 적합

비교 지표는 기존 실제 E150 명령이며 새로운 w0 출력으로 기준을 바꾸지 않는다. E85 own-leg 참조/offset과 E103 startup, command tick100..1923의1824행, velocity1823쌍, startup1..99를 보존한다. command RMS는 원래 대비 .8 이하, 전체/각 다리 velocity와 각 다리 command는 비악화, startup q RMS .018/max .04, LTM RMS .009/max .025다. 참조와 phase는 오프라인 적합용이며 실행 입력이 아니다.

계수 순서는 원래 graph column 오름차순330MN, filter0/1/2, low/high interleave의1980개다. 변수는 w1980, ell990, u990, margin t1로3961개다. 물리적 true family는 w/ell/u∈[0,2], t∈[-1,1], ell≤low/high≤u, MN별 sum_filter ell≥1e-6/n_owned, 근육별 sum ell≥.5와 sum u≤2다. 원래 물리 판정 허용치는1e-12다.

수치용 inner 문제는 w lower/upper에 g=1e-9, MN/근육484행 RHS에 delta=1e-7의 내부 여유만 둔다. ell/u와 envelope는 원래 값이다. 기각용 true dual에서는 계수 box를 g만큼 복원하고 물리484행을 delta+1e-12만큼 복원한다. 작은 음수 계수를 clipping하지 않는다.

기존 E169와 같은3961변수/20682선형행/15SOC/18개 local thin QR 구조를 사용하되 affine offset은 **고정 활성의 실제 명령에서 참조를 뺀 값**이다. startup 직접 행, 전체/다리별 오차 norm, 공통 margin을 보존한다. linear행은 box7922, envelope3960, MN330, muscle154, startup±8316 순서다. 물리행은11882:12366이다. SOC 순서는 전체command/velocity, 각다리command/velocity, startup이다. truncated QR/SVD는 사용하지 않는다.

새 함수는 기존 E150을 포함하지 않으므로 old w0 feasible witness를 강제하지 않는다. old w0 및 canonical equal-branch 전달의 실제 slack을 진단으로 저장하되 불가능한 포함성 주장을 하지 않는다. QR/direct 비교는 w0, zero, 고정 cosine 벡터로1e-12+1e-10*norm 허용치, 최종 반환에서도 직접 명령/velocity/startup 지표로 다시 판정한다. decoder의 동등 계산 비교는 activation/q에1e-12+1e-12*scale, 차분 velocity에 그 양 끝 q 오차 한도를 dt로 나눈 전파 한도를 적용한다. 이 수치 정합 한도는 보행/적합 합격 한도를 완화하지 않는다.

Clarabel0.11.1, CPU1/max_threads1, qdldl, max_iter100, time_limit20초, gap/feas/infeas1e-10, P0/q_t=-1. constructor1/solve1/반환1/자동retry0이며 원본 데이터로 문턱/시간상수/초기값을 다시 고르지 않는다. Solved/AlmostSolved와 유한 반환만 정상 adjudication한다. 다른 상태는 invalid 또는 inconclusive로 남기며 검증하지 않은 infeasibility certificate를 과학적 기각으로 쓰지 않는다.

실제 계수·보호·직접 지표를 모두 통과하면 supported_saved_prefilter_proposal이다. 통과점이 없고 같은 true 문제에 대한 독립 검산 projected dual margin upper<0이면 refuted_saved_prefilter_family다. 나머지는 valid_rejection_inconclusive이며 둘이 모순되면 invalid다. 양수cone/Lorentz projection pad1e-12, box support와 upper pad1e-8의 기존 E145 방법을 재사용한다. 엄밀한 구간 증명이나 BANC 전체 불가능성 주장이 아니다.

## 개발, 실행, 다음 분기

부모 설계 → actual gpt-6-sol/high 첫 구현 → 부모 FIRST → 같은 Sol 첫 수정 → 부모 SECOND/잔여 수정. Sol 소유 파일은 `tools/fit_c_banc_m3_prefilter_recruitment.py`, `tests/test_c_banc_prefilter_recruitment.py`다. 기존 E145/E164/E168/E169의 검산된 로더/입력검증/metrics/cone/dual/publication helper를 읽기 전용 재사용할 수 있다. 모듈 전역 monkeypatch나 기존 소스 수정 없이 새 함수 차이를 명시한다. 부모는 독립 생성/반환 검산과 freeze/감독/기록을 소유한다.

소스 리뷰와 hash gate 이전에는 원본 배열 수치 계산/fit0이다. 합성 검증은 새 상태의 비가환성·인과성·zero initialization, 실제 소유권/positive 제약, 직접/affine/QR 일치, 한 solve와 실패 시 raw 보존을 확인한다. pre-solve에 입력과 L/H/basis/mapping/offset/QR/problem을 저장하고 bare solver 반환, candidate 출력은 판정보다 먼저 남긴다. source/design/input hash, state/clock/source provenance와 완료/실패 원인을 보존한다. 오류 뒤 동일 원본 자동 재실행은 없다.

원본 process1/fit1/GPU0/새 neural·physical0, CPU1, fit wall30/internal95/external110/supervisor120초, output384MiB/free2GiB다. 누적4997.566624599716초와 기존 상한5144초를 유지한다. 잔여146.433375400284초 중120초를 예약하므로 배정 확대가 없다. 준비·합성·부모 검산은 별도 기록하며 미계측 비용을0으로 쓰지 않는다. F6노출0, M2/UI보존, 후보미채택이다.

통과하면 같은 전달의 저장/복원·cut을 포함한 runtime adapter와 단일 CUDA 폐루프 검증을 설계한다. 기각되면 이 함수의 추가 계수 적합을 중단하고 신경 출력 생성 또는 다른 근육 상태의 근거를 검토한다. 저장 적합은 실제 보행 개선이 아니며 전체 F1–F8 목표는 계속 active다.
