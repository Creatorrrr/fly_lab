# E168 사용량 의존 전달 상태의 단일 조건부 적합

2026-09-23 부모 설계. M3 finite-goal과 F1–F8을 그대로 유지한다. E167은 E150 최대 절반 감쇠 가족에서 전체 RMS 하한 .036416431992rad를 얻어 기존20% screen목표 .045349108280rad에 대한 가능성을 배제하지 못했다. 독립183검산까지 완료한 진전이다. 이 낙관적 구간은 실제 계수/시간 상태의 해가 아니다. E165/E166은 전달 고갈의 인과 근거가 아니다.

## 질문과 경쟁 설명

H168은 같은 MN의 활동 이력에 따라 사용 중 효능이 감소하고 휴지 중 회복하는 상태를 고정해, 기존 신경 출력·소유권·초기 자세·전달 보호 제약을 유지하며 유용한 참조 파형 오차를 더 줄일 수 있는가다. 경쟁 설명은 시간 일관성과 기존 하한/초기 자세를 유지하면 E167의 낙관적 여유를 사용할 수 없다는 것이다. E164는 양의 fast-minus-slow onset, E123/E138은 신경망 내부 적응, E148은 정적 bank hinge였다. 이번 식은 고정된 신경 출력 뒤에서 같은 MN 기여의 효능을 감소시키므로 다르다.

tau=.07s는 공학적 prior이며 성체 다리 NMJ 실측값이 아니다. 유충과 성체 비행근 관찰을 성체 다리의 직접 증거로 쓰지 않는다. 이 실험은 저장 출력의 조건부 수치 적합이며, 생물학적 기전 또는 폐루프 보행의 검증이 아니다.

## 고정 입력과 처리

E164의 동결 입력/출처/완료·시계·초기 snapshot 배열/해부학/참조/strict startup 검증을 읽기 전용으로 재사용한다. e164.load_original_inputs(base_freeze, merged_hashes)와 prepare_saved(data, base_freeze, raw, pending)를 사용할 수 있다. E164 실행기/preflight/optimizer를 호출하거나 source/원자료를 수정하지 않는다. 이 helper가 검산하는 기존 onset 배열은 새 전달의 입력으로 사용하지 않는다. 새 freeze는 base_freeze 자체와 그 전체 참조 입력을 해시로 묶는다. StateStore/GraphStore/body 객체는 만들지 않는다.

E150 bank(2001,391,3), 실제 activation(2001,96), original_activation, endpoint motor_rate(2001,391), 현재 control_tick0..2000, neural_tick20배,5ms clock을 사용한다. dtype/shape/finite/비음수/availability/소유권은 손실 변환 전에 확인한다. raw 입력은 assertion 전 저장한다. fixed19와 variable77/330은 E164의 동결 목록을 사용한다. parameter NPZ에 존재하지 않는 fixed_muscles 필드를 요구하지 않는다. E164 checkpoint가 제공하는 고정 목록도 비교한다. 근육군17/15/16/17/15/16과 실제 moment cross-leg0를 보존한다.

E148의 low/high 양수 branch로 c_i[k]=sum_j(low_ij*min(bank_ij,kappa_i)+high_ij*max(bank_ij-kappa_i,0))를 계산한다. c>=0이며 variable 근육의 합과 기록 activation은 atol1e-12로 같아야 한다. high-low가 음수인 정상 branch를 음의 해부학 연결로 해석하지 않는다. 기존 activation/moment/rest/target/tendon/recurrence 재구성도 원래 E164 한도를 유지한다. t0 target은 원래 decoder equality 대상에서 제외한다.

## 새로운 상태와 자유도

R_i[0]=1이다. k=1..2000에서 같은 행의 endpoint rate r_i[k]로 s=r/(r+kappa)를 계산한다. r=kappa=0이면 s=0; kappa=0,r>0이면 s=1이다. 모든 입력은 finite/비음수여야 하며 overflow를 묵인하지 않는다. R*=1/(1+s), R_i[k]=R*+(R_i[k-1]-R*)*exp(-(1+s)*.005/.07)이다. 현재 bank와 **업데이트 후 R[k]**를 조합해 phi_i[k]=(1-R_i[k])*c_i[k]를 만든다. t0 phi는0이다. 새 EMA/신경 rollout 없이 자원 상태2000회만 진행한다. tau/kappa/lag/offset/시작 R을 탐색하지 않는다.

모든 비음수 rate 이력에서 .5<=R<=1, 0<=phi<=.5c이다. 기존 MN391 중 variable 근육에 속한330개만 lambda_i∈[0,1]를 갖는다. a'_m=a_recorded_m-sum_owned lambda_i*phi_i이며 fixed19는 기록값을 그대로 둔다. 새 연결, 외부 phase/시간 진동/교사 출력은 모델 입력이 아니다.

명령은 q'=q_base+.02*((a'-a_recorded)@moment.T)로 정의한다. 여기서 저장자료 실험의 q_base는 실제 E150 recorded_command이며, 향후 runtime에서는 그 시점의 기존 BANC decoder 출력이다. 저장 궤적을 runtime에 주입하는 설계가 아니다. lambda0은 activation과 command를 exact하게 보존한다. 별도로 q'와 rest+.02*(a'@moment.T)의 tick1..2000 residual<=1e-12를 검사한다. 이 baseline 부동소수점 차이는 E167의1e-10rad 이상 확장 안에 포함된다. LTM 출력은 기존 출력 그대로이며 변경되는 근육에 LTM는 없다.

## 전체 이력 보호와 적합

기존 slope로 mu_i=sum_j min(low_ij,high_ij), U_i=sum_j max(low_ij,high_ij), n_m=원래 소유 MN수를 정의한다. R 범위를 사용한 모든 이력의 충분 선형 제약은 (1-.5lambda_i)*mu_i>=1e-6/n_m, sum_owned(1-.5lambda_i)*mu_i>=.5다. 기존 sum_owned U_i<=2는 입력 invariant로 검증하며 감쇠이므로 자동 보존된다. 이는 R이 주어졌을 때 양수 branch의 보호이지 활동 이력 전체에 대한 동적 단조성 증명이 아니다. 작은 위반을 clip/정규화하지 않는다.

변수는 graph-column 오름차순330 lambda와 마지막 margin t의331개다. 목적은 max t, t∈[-1,1]이다. E164의77g+330beta layout/양수 onset 열/증폭 예산행을 재사용하지 않는다. 각 MN 열은 -.02*phi_i*moment[:,owner(i)]이고 constant는 실제 q_base다.

E164 목표/기준을 바꾸지 않는다: target index100..1923의1824행과1823개의 실제 인접 velocity, 원래 E85 own-leg 참조/offset과 E103 startup1..99. 전체 command 한도는 E150 baseline*.8, 전체 및6다리 velocity와6다리 command 한도는 각각 E150 baseline, startup qRMS<=.018 및 samplewise max<=.04, LTM RMS<=.009/max<=.025다. 15 SOC에 norm/limit<=1-t를 사용하고 startup pointwise 양/음 제약에도 t를 넣는다. 한도가0이면 임의 epsilon으로 바꾸지 않는다. 이20% screen은 자연 보행의 새 필요조건이 아니다.

E145의 truncation 없는 thin QR, SOC row 생성, safe solve/callback 및 보정 dual helper는 읽기 전용으로 재사용한다. E164 Criterion/ConicProblem 데이터 구조와 metrics_from_recorded/solver_raw의 차원 일반 부분을 사용할 수 있다. E164 _leg_affine/_compress_blocks/build_conic_problem/direct_metrics는 기존 layout/witness를 가정하므로 그대로 사용하지 않는다. 새 QR witness는 lambda0, linspace(-.375,.625,330), cos(arange(330)*.173)다. 압축 전/후 norm은1e-12+1e-10*directnorm 이내여야 한다. 실제 solver baseline witness는 lambda0,t=-.251로 고정하고 원래 신호에서 선형/SOC/startup feasible인지 fit 전에 raw 뒤 검증한다. 실패하면 fit0의 기술적 실패로 남기며 t를 다시 고르지 않는다.

Clarabel0.11.1, qdldl, CPU/max_threads1, max_iter100,time_limit20,tol_gap_abs/rel/feas/infeas_abs/rel1e-10, 나머지 defaults. solver 생성/solve 한 번, 반환 벡터 하나만 검사한다. raw x/s/z/history/settings와 callback/constructor/solve 카운트를 반환 직후 저장하고 후속 계산을 한다. lambda의 작은 음수/1초과도 clip하지 않는다. 제안 조건은 모든 lambda가 정확히[0,1], 모든 기존 command/velocity/startup/fixed/소유권/상태 제약 통과이며 전달 예산·재구성 수치 검산은 physical atol1e-12를 유지한다. solver 상태명만으로 채택하지 않는다.

331개 실제 box로 보정 dual margin upper bound를 계산한다(P0, 마지막 q=-1, cone projection pad1e-12, upper bound pad1e-8). 모든 물리조건을 통과하면 conditional_resource_proposal이며 native/보행 검증은 아직 pending이다. 제안 실패이고 보정 upper<0이면 refuted_fixed_resource_family로 이 신호·고정tau·보호 제약 가족에만 적용한다. 나머지 유효 실패는 valid_rejection_inconclusive다. 기술적 오류는 과학적 반증이 아니다.

## 검증·실행·자원

부모 설계 → 실제 gpt-6-sol/high FIRST 구현 → 부모 FIRST 리뷰 → 필요시 같은 Sol FIRST 수정 → 부모 SECOND 리뷰와 잔여 직접 수정. 동적 상태의 수치 적합과 파일·solver 계약을 연결하므로 high다. 소유 파일 tools/fit_c_banc_m3_resource_attenuation.py와 tests/test_c_banc_resource_attenuation.py. E167 및 과거 helper/자료/신체/UI/M2는 읽기 전용이다.

CLI --freeze/--freeze-sha256/--source-review/--source-review-sha256/--out. 실제 schema/해시 gate와 독립 cwd subprocess, 생성 fixture full-run 및 수치 단위 테스트를 원본 전에 확인한다. 생성 시험은 R의 상수입력 폐형식·행 시점·zero kappa/zero rate·감소/회복·R 범위·lambda0·양수 기여·fixed19·근육 소유권·all-history 보호·QR/15SOC/331차원/dual/solver실패와 raw 보존을 다룬다. 원본을 fixture로 읽지 않는다.

원자료/상태/phi/재구성/affine/QR/witness/문제행렬을 수치 assertion 전에 pre-solve raw로 보존한다. 그 저장 실패 시 solve0이다. 솔버 반환 직후 raw 저장, 후보 activation/q/velocity/startup/tendon도 판정보다 먼저 저장한다. post source/입력/snapshot 검사와 마지막 output hash·저장·용량·free 검사까지 마친 뒤 마지막 time gate를 적용한다. 늦은 실패에는 성공 receipt를 취소하고 최초 오류/raw/temp를 남긴다. 반환·저장 timestamp의 측정 범위를 일치시킨다.

원본 process1/자동재시도0, CPU1/GPU0, fit<=1, 자원 상태 update2000, neural/physics/body/StateStore/GraphStore0. fit20/internal40/external50/supervisor55초, output192MiB/free1GiB. E167까지 실패 포함 보수누적2897.3572213초를 올림한2898+55=2953초를 사전 배정하며 기존2953 범위 안이다. 생성 검증/보존 해시/부모 독립 검산은 별도 기록한다. 모든 원본 입력과 과거 실패를 보존한다. F6노출0, UI/M2변경0, 후보 채택false, 전체 목표는 active다.
