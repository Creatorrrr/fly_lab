# E171: 저장된 전달 특징의 제한 없는 표현력 진단

Milestone은 BANC 자연 보행 M3의 고정 F1–F8이며 계속 pending이다. Experiment ID E171, Hypothesis ID H171, 종류 diagnosis이다. E169 원본 invalid와 별도 완료 affine 인증, E170 valid 조건부 실패를 그대로 보존한다. 이번 계산은 새로운 보행 후보나 원본 run 재시도가 아니다.

## 질문과 비교

H171: E150의 같은 관측 신호로 만든 저장 affine 특징에, 양수·예산·시작 보호 제약을 제거하면 전체 목표 명령 RMS를 baseline보다20% 이상 줄이는 유한한 계수가 존재하는지 확인한다. E169 static과 E170 alpha+F*beta를 각각 한번 비교한다. 경쟁 설명은 현재 시간 특징의 표현력 자체가 부족하거나, 수치 rank/큰 상쇄로 추정이 불안정하다는 것이다. 양수 제약 등이 병목이라는 인과 결론은 이 비교만으로 확정하지 않는다.

두 가족×여섯 다리의 command QR에 각각 SVD 한 번, 합계12회 신규 fitting 계산을 수행한다. command만 fit하고 저장 velocity/startup QR에는 같은 계수를 대입해 평가한다. 원본 bank/rate/history/activation/state/target command를 열거나 재계산하지 않는다. 새로운 EMA, history recurrence, neural, CUDA, physics, body, 학습 rollout은0이다. 참조 궤적은 기존 affine offset에 포함된 진단 목적의 교사이며 runtime에 투입하지 않는다. F6 미사용 초기 상태는 노출하지 않는다.

## 입력과 소유권

입력은 E169/E170 `pre-solve-raw.npz`에서 명시한 키만 선택한다: 18개 `qr_{command|velocity|startup}_{lf|lm|lh|rf|rm|rh}`와 대응 `_active`, `pair_motors`, `pair_filters`, `pair_muscles`, `w0`, `target_indices`, `pair_start`, `pair_end`. `np.load(..., allow_pickle=False)`의 선택 키 접근만 허용하고 archive 전체 순회/복사 및 기존 fitter import는 금지한다. 고정 support와 muscle/leg metadata는 부모 freeze의 sample contract를 사용한다.

동결은 두 원시 파일, 원래 manifest/result/freeze/최종 review와 source의 SHA256을 포함한다. 원래 두 freeze의 공통 입력 hash와 sample contract의 동일성을 metadata로 확인한다. source gate는 새 구현·시험·설계·freeze hash를 확인한 뒤에만 원본 선택 키를 열도록 한다. 출력 폴더가 이미 있으면 실행하지 않으며 자동 재시도는0이다. 기존 입력과 소스는 읽기 전용이다.

static coefficient1980 = ascending original graph-column330×filter3×low/high2. history coefficient3960 = alpha1980, beta1980. pair990개의 motor/filter/muscle 소유권, 실제 noncontiguous original motor 목록, 변수77muscle, 고정19muscle를 확인한다. 다리별 active는 static의 같은 leg 소속 low/high index, history는 alpha active 뒤에 alpha active+1980이어야 한다. 같은 다리의 command/velocity/startup active는 동일하고 여섯 다리는 전체 계수를 중복 없이 덮어야 한다. w0는 두 가족이 exact 같아야 한다. baseline은 static w0, history [w0,0]이다.

target tick100..1923의1824행, velocity1823인접 쌍, startup1..99의99행, 다리별7관절을 유지한다. 각 QR은 augmented [M,offset]의 저장 thin R이며 열 수 active+1, 행 수 min(원래 행수,active+1)이다. 원래 행수는 각각12768,12761,693이다. QR residual 성분은 원래 관절/시간 성분이 아니다. 따라서 startup RMS는 계산하지만 pointwise max, clipping, activation 및 native gate는 **평가하지 않으며 pass로 표시하지 않는다**.

## 고정 선형대수와 수치 의미

float64, NumPy2.5.3, CPU/BLAS1을 사용한다. 가족 순서는 static/history, 다리 순서는 lf/lm/lh/rf/rm/rh다. 각 `M=R[:,:-1]`, `o=R[:,-1]`에 `np.linalg.svd(M, full_matrices=False)`를 한번 호출한다. `cutoff = eps(float64)*max(M.shape)*s[0]`, 엄격히 `s>cutoff`인 성분만 사용한다. all-zero이면 rank0, x=0이다. `x = -Vh[keep,:].T @ ((U[:,keep].T @ o)/s[keep])`이며 coefficient clipping/정규화/regularization/column scaling과 cutoff 탐색은0이다. 반환 x는 매 다리 계산 후 즉시 원형 저장한다. 실패한 SVD를 다른 driver나 다른 cutoff로 재시도하지 않는다.

U,s,Vh,keep,x,cutoff,실제 `M@x+o`, projected residual `o-U_keep@(U_keep.T@o)`,baseline residual를 보존한다. rank, full/retained condition number(null for undefined/infinite), cutoff 주변 [cutoff/10,cutoff*10] singular count, 계수 최대절대값/L2를 기록한다. 원본 QR와 SVD raw는 hash로 추적한다.

유한성, singular descending/nonnegative, 상대 Frobenius reconstruction≤1e-10, U/V orthogonality max≤1e-10을 검사한다. retained rank stationarity `U_keep.T@actual_residual`와 원래 normal residual `M.T@actual_residual`도 기록한다. 실제/projected residual norm 차이는 `128*eps*max(shape)*(||M||F*||x||2+||o||2)`+1e-12 이하인지 확인한다. 이 양은 보수적인 **heuristic numerical resolution indicator**이며 엄밀한 오차 인증이나 가족 전체의 잔차 하한이 아니다. 해당 지표의 다리별 제곱합 제곱근/√(1824*42)가 baseline command RMS의0.1%보다 크면 비교를 numerically unresolved로 분류한다. threshold의 양쪽 경계가 이 지표 안이면20% 달성 판정도 unresolved이다. 수치 검사가 실패하면 invalid/not_tested로 분리하고 raw는 남긴다. singular cutoff 주변 수치 및 dropped direction은 명시하며, 미달을 전체 표현 불가능성으로 확대하지 않는다.

각 kind RMS = QR residual norm/√(원래 샘플수*관절수)이다. 전체 RMS는 다리별 squared norm을 합해42관절로 계산한다. 두 가족 baseline의 command/velocity/startup norm은 abs1e-12+rel1e-10 내에서 같아야 한다. 각 fit의 실제 command norm이 baseline보다 이 허용치 이상 나빠지면 검사 실패다. 목표는 original command RMS ratio≤0.8이며 velocity와 startup은 fit에 포함하지 않는다. baseline 및 반환 계수의 모든 metric을 저장한다.

## 물리 제약 위반 진단과 결정

static low/high 또는 history alpha/beta는 음수와2초과를 exact count하고 크기를 기록한다. history alpha+beta의0미만/2초과도 기록한다. 각 pair의 F∈[0,1] 전체 extrema는 static의 두 branch, history의 네 endpoint(alpha low/high와 alpha+beta low/high)의 min/max다. beta가 음수인 진단해에서 alpha만으로 최소값을 계산하지 않는다. pair min/max를 합해 MN floor1e-6/n_owned, muscle lower0.5/upper2를 atol1e-12로 비교한다. 모든77변수 근육 및330MN의 raw extrema/slack를 남긴다. 이는 원래 제약을 만족시키기 위한 재적합이 아니며 실제 activation 검증을 대신하지 않는다.

유효하고 수치 해상도가 충분한 반환 해가20%에 도달하면 `supported_returned_saved_affine_witness`이며 신호에 일부 활용 가능한 표현이 있음을 보고한다. 어떤 제약을 위반했는지와 startup/velocity 악화를 근거로 전달·초기화 조사의 우선순위를 정한다. 어느 가족도 도달하지 못하면 `inconclusive_not_attained_at_fixed_cutoff`이며 rank/불안정을 검토한 뒤 신호 특징 또는 기존 BANC 동역학을 조사한다. 수치 해상도가 불충분하면 `inconclusive_numerical_resolution`이다. history-static 차이는 같은 finite-precision rank 규칙 아래의 진단값일 뿐 history가 자연 보행을 개선했다는 뜻이 아니다. 양쪽 모두 실물 보행, 생물학적 타당성, F1–F8 완료를 주장하지 않는다. adoption=not_applicable, candidate_adopted=false, whole_goal_complete=false를 고정한다.

## 실행과 검토

부모가 설계/freeze/source gate/독립 검토를 소유한다. 실제 gpt-6-sol/high가 `tools/diagnose_c_banc_m3_saved_qr_capacity.py`와 `tests/test_c_banc_saved_qr_capacity.py`의 FIRST 구현을 맡는다. 첫 수정이 필요하면 같은 Sol, SECOND 리뷰 후 잔여 수정은 부모가 맡는다. 생성 full-rank/rank-deficient/zero 및 cutoff 경계, 비연속 소유권, QR RMS/pointwise 구별, signed beta extrema, 반환 보존/단일12-call/금지된 입력 키 접근을 시험한다. 부모는 별도 생성 closed-form/scalar oracle로 검토한다.

CPU1, 최초 original process1, SVD fit12, QR factorization0, 자동 retry0. 내부90초/외부105초/supervisor120초, raw128MiB/free1GiB, 실행 시작 전 Threadpool 정보와 버전을 저장한다. 각 완료 solve raw를 검증 전에 저장하고 failure와 primary/secondary publication 오류를 보존한다. 같은 프로세스의 마지막 성공 raw를 예외로 잃지 않는다. provenance 검사는 원본 key load 전 수행한다. 감독 프로세스는 자원/timeout/exit/raw크기를 별도 기록한다.

이전 보수 original 누적3045.490792799974초를 유지한다. 기존 prospective bound3093초를 새 실행 전3166초(ceil3045.490792799974+120)로 확장한다. 이전 비용을 초기화하지 않으며 생성자료 시험/metadata preflight/저장결과 검증 비용을 구분한다. 새 solver가12회라는 사실을 offline이란 이유로 fit0으로 바꾸지 않는다. 실행 후 결과와 다음 행동을 progress/checkpoint에 기록하고 F1–F8 미달을 유지한다.
