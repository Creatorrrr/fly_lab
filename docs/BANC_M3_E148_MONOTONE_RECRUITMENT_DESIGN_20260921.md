# E148 입력 수준에 따른 단조 모집 전달 — 사전 설계

2026-09-21. 부모가 설계하고 실제 Sol/high가 구현한다. E147은 valid/reject이며 변경하지 않는다. 같은 후보의 실제 F1–F8이 최종 목표다. 이 단계의 조건부 제안이나 수치 검사 통과는 보행 완료가 아니다.

## 질문과 범위

H148: 원래 운동뉴런·근육 대응, 5/10/40ms 필터와 양의 전달 방향을 보존하되 입력 수준에 따라 전달 기울기가 달라지는 함수를 사용하면, E147과 같은 관절·속도·초기 자세 조건을 동시에 만족하는 전달 모수를 얻을 수 있는가?

E145–E147은 정적 합성·선형 시간 필터·근육별 크기 조절의 조합으로 공동 조건을 충족하지 못했다. E147 잔차는 RF joint5와 LF joint0/5/6에서 크며, 초기의 양·음 최대 잔차가 여러 시점에서 활성이다. 다음 변경은 gain 상한 확대나 시간상수 탐색이 아니라 **단조인 입력-출력 곡선의 형태**다. 경험적 발화율-근력 관계를 복원했다는 주장은 하지 않는다. 이는 위치 명령을 생성하는 기존 공학적 인터페이스의 가설이다.

동결한 E146 bank (2000,391,3)를 그대로 읽는다. 새 EMA, 신경, CUDA, 신체 또는 LP 실행은 없다. E143 원래 baseline, E103 초기99와 E142 anatomy/moment/rest, 실제1827 endpoint 및1826 인접 velocity pair를 유지한다. F6·활성 UI·M2는 사용하지 않는다. 새 연결, signed recruitment, 외부 위상/시계/교사/궤적/root 보정은 없다.

## 입력 곡선과 모수

운동뉴런 i의 knot κ_i는 기존10ms bank의 **엄격히 양수인 2000개 표본 중 중앙값과1Hz 중 큰 값**이다. 양수 표본이 없으면1Hz다. NumPy median/quantile(method=linear)의 동일한 정의를 사용한다. 같은 κ_i를 세 필터에 적용한다. knot 선정에는 명령 목표나 residual을 사용하지 않고 추가 knot 탐색도 하지 않는다. 선택 규칙을 먼저 고정하고 원본 계산 때 실제κ를 raw로 보존한다.

각 bank 값 b≥0에 φ_low=min(b,κ), φ_high=max(b−κ,0)를 계산한다. 두 조각의 합은 수학적으로 b이다. 원래 같은 근육에 속한 MN만 a_m=Σ_i,k(w_low φ_low+w_high φ_high)에 기여하며 w는 비음수다. 함수는0에서0이고 연속·단조이며 외부 oscillator/state를 추가하지 않는다. 필터 상태가 고정되면 출력은 상수다. 기울기가 서로 같으면 E147의 선형 전달을 포함한다. 이것은 힌지별 독립 위상/시간 선택이 아니다.

LTM6 및 원래 singleton 근육은 정확히 기존 canonical10ms 출력으로 둔다. 가변 근육에만 새 함수를 쓴다. q=rest+.02*moment*a와 힘줄 식은 그대로다. 가변 baseline의 부동소수점 reduction 차이는 명시적으로 수치 비교하고 고정 근육 출력은 bit-exact로 요구한다. baseline 성능 기준은 새 곡선으로 재평가한 값이 아니라 동결한 원래 canonical baseline이다.

가변 원래 MN/filter pair 개수를 n이라고 한다. pair 순서는 E147과 동일한 muscle 오름차순→motor-major3*i+k다. coefficient 순서는 pair마다 low,high다. 최적화 벡터는 [2n coefficient, n lower envelope ell, n upper envelope u, margin t]이다. 해석용 envelope는 최종 실제 coefficient의 min/max로 계산하며 solver auxiliary와 혼동하지 않는다.

각 coefficient에는 branch guard1e−6/해당 근육 원래MN수와 상한2를 둔다. ell≥0, u≥0 및 각각≤2, ell≤w_low/w_high, u≥w_low/w_high를 둔다. 근육마다 Σell≥.500001, Σu≤1.999999다. 따라서 모든 입력 수준의 합산 기울기가 같은 .5–2 범위 안에 있도록 보수적으로 제한한다. 이는 생리 측정값이 아니라 E147 크기 범위를 유지하기 위한 제한이다.

실제 판정은 coefficient≥0, 같은 anatomical support, 원래 MN별 Σ_k min(w_low,w_high)≥1e−6/MN수(atol1e−12), 근육별 Σ_i,k min(w_low,w_high)≥.5−1e−12와 Σ_i,k max(w_low,w_high)≤2+1e−12다. fixed weight/output 정확 보존을 요구한다. 사후 clipping·정규화·gain 보정·후보 조합은 하지 않는다. solver가 반환한 한 벡터만 판정한다.

canonical witness는 middle pool을 low/high에 복사하고 ell=u=그 pair weight로 둔다. lifted witness는 원래 각 weight의1e−6씩 fast/slow로 나누고 middle1−2e−6를 둔 다음 양쪽 slope와 ell/u에 복사한다. t=−.251에서 원래 baseline과의 허용 가능한 부동소수점 차이, 실제 cone feasibility를 검증한다. E147 반환 weight도 같은 방식으로 lift해 저장 출력과의 대응을 수치 확인한다(최대 절대 오차1e−12). E147가 합격했다고 바꾸지는 않는다.

## SOCP와 수치 검산

목적은 max t, t∈[−1,1]이다. 실제 q와 인접 q velocity, 초기99에 대한 affine matrix의 열은 .02*φ*moment다. envelope 열은 모두0이다. fixed activation/rest만 affine offset에 둔다. 실제 anatomical17/15/16/17/15/16 group 및 원래 MN column과 cross-leg moment0를 검증하며 16개 고정 분할을 사용하지 않는다.

E147와 같은 전체/다리별 command·velocity 및 전체 startup15 SOC를 사용한다. 전체 command 한도는 baselineRMS*.8, velocity와 각 다리 command/velocity는 baseline, startupq RMS .018/max .04, tendon RMS .009/max .025다. norm/limit≤1−t와 초기99*42의 ±sample 제약을 유지한다. thin QR은 truncation하지 않으며 envelope의0열은 QR 입력에서 제외하고 embedding에만 포함한다. canonical/lifted 직접 norm·QR norm을 보존한다.

첫 nonnegative cone의 row 순서:

1. −coefficient≤−guard, +coefficient≤2.
2. −ell≤0, +ell≤2, −u≤0, +u≤2.
3. ell−w_low≤0, ell−w_high≤0, w_low−u≤0, w_high−u≤0.
4. −row_sum(ell)≤−.500001, +row_sum(u)≤1.999999.
5. −t≤1, +t≤1.
6. startup_matrix/.04+t≤1−startup_offset/.04, 이어서 음의 쪽도 같은 표본 순서로 추가한다.

Clarabel0.11.1/CPU1/max_threads1/qdldl/max_iter100/time_limit20, tol_gap_abs/rel/feas/infeas_abs/rel1e−10와 나머지 defaults다. 기존 safe solver wrapper를 재사용한다. projected dual/self-dual cone 내부pad1e−12*(1+tailnorm), r=q+A.T*z, 추가boundpad1e−8을 유지한다. dual용 box는 모든 coefficient/ell/u∈[0,2], t∈[−1,1]이다. 수치 상한이 음수이면 이번 guarded monotone single-knot family에만 한정한 수치적 반증이며, 생물학적/전체BANC 불가능성은 아니다.

## 保存と担当

새 두 파일만 Sol이 구현한다: tools/fit_c_banc_m3_monotone_recruitment.py, tests/test_c_banc_monotone_recruitment.py. E144/E145/E146/E147 승인 helper는 읽기 전용으로 재사용한다. 부모는 freeze, 생성자료의 독립 검산, 원본 감독과 결과 기록을 맡는다. 부모 FIRST review 후 첫 수정은 같은 Sol/high, 부모 SECOND review 후 잔여 수정은 부모가 맡는다. native 신경/CUDA/state 변경은 이 구현에 포함하지 않으며 이후 필요하면 실제 새 Sol/xhigh에게 별도 위임한다.

CLI는 --freeze/--freeze-sha256/--source-review/--source-review-sha256/--output이다. 부모 final source review의 accepts_single_monotone_recruitment_fit=true 및 blocking_defects=[]와 소스 해시를 요구한다. 출력은 conic-problem.npz, monotone-recruitment-raw.npz, freeze-snapshot.json, source-review-snapshot.json, execution-raw.json, summary.json이다. 새 출력 디렉터리만 만들고 원본 재실행/덮어쓰기하지 않는다.

raw에는 원래bank/실제knot/파생basis/전체 variable map/canonical·lifted·E147 witness/slope와 envelope/A/P/b/q/cones/전체QR/criterion, solver 설정·history·반환x/s/z·dual보정·잔차, 실제 candidate coefficient/min-max budget/activation/command/velocity/startup/tendon/지표를 저장한다. 이미 생성된 자료는 그 다음 계산·판정·저장 전에 caller-owned partial에 게시한다. pre-solve raw 저장에 실패하면 solver는 진입하지 않으며 첫 오류와 secondary 오류를 구분한다. 실패 경로의 실제 solver 진입·derived value 수를 기록한다. raw를 먼저 저장한 뒤 proposal/자원 판정을 한다.

validity와 proposal을 분리한다. 공동 물리 조건을 모두 통과하면 conditional_monotone_recruitment_proposal만 표시하며 실제 채택은 하지 않는다. 미달+음의 상한이면 family refuted, 나머지는 inconclusive다. F1–F8은 동일 후보의 native/물리/확인/회귀/통합 증거가 있을 때만 통과한다.

## 자원

원본1프로세스/재시도0/solver1, CPU1/GPU0, 새 EMA/neural/CUDA/physics/LP0. fit20/internal45/external50/supervisor59초, raw192MiB/free1GiB다. E147 원본프로세스 보수 누적2211.2814538000384초를 보존하고 prior ceil2212+59=2271로 **기존2271 배정 안**에 둔다. 이는 전체 프로젝트 수명 비용이 아니라 기존 원본프로세스 누적 계정이며, 생성자료·개발·부모 검산 비용은 기존 방식대로 별도 기록한다. 모델 시간·원본 실행 성공을 늘려 완료 기준을 대신하지 않는다. 추가 원본 실행은 부모 최종 source gate 이후 한 번만 가능하다.
