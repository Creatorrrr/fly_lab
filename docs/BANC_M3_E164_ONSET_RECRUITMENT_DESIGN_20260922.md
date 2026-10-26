# E164 같은 운동뉴런의 활동 상승 성분을 이용하는 전달 적합

2026-09-22. 부모 설계. E163는 저장 진단으로 완료했고, 새로운 보행 후보는 아직 없다. E164는 **고정된 E150 신경 기록에서만 평가하는 단일 볼록 적합**이다. 물리·신경 실행 또는 최종 controller 통합은 포함하지 않는다. F1–F8/M2/UI/F6 및 모든 과거 실패는 보존한다.

## 근거와 질문

E163 후기에는 모든15 E1쌍 R≥.99306이고 속도 저하가 두 위상에 집중된다. 이 연관성만으로 coupling 문제를 원인으로 단정하지 않는다. E106 timing-only는 적격 주기를 회복하지 못했고 E107 shape-only는 회복했다. E148의 단조 전달은 실제 E150에서 전진/무릎폭을 개선했지만 연속 보행은 미달이다.

E123–E142는 MN 내부의 적응 상태를 바꿨고, E146–E148은 같은 MN의5/10/40ms EMA를 각각 양수로 조합하거나 각 branch의 활동 수준 hinge를 사용했다. **서로 다른 filter의 차분을 양수로 정류한 전달은 해당 계보에서 시험하지 않았다.** E146의 affine 좌표 안에 나타나는 anchor 차분은 물리 출력의 onset branch가 아니다.

H164: 이미 존재하는 같은 MN의 빠른/느린 filter 차이에서 활동 상승 성분을 얻고, 기존 전달과 제한적으로 재분배하면 기존 초기 자세를 유지하면서 각 다리의 유용한 기준 파형 재구성을 더 개선할 수 있는가? 이는 위치 명령을 만드는 공학적 전달 가정이다. 측정된 근육 생리나 실제 생물학적 기전이라고 부르지 않는다. 단순 gain 확대, 시간상수 sweep, E93 coupling 재fit은 하지 않는다.

## 고정 자료·재구성

E150 native-recruitment-state의 bank(2001,391,3), activation/original_activation(2001,96), motor_rate 및 control/network/physics clock와 availability를 읽는다. 기존 parameter bundle의 tau=[.005,.01,.04], kappa391, slopes(96,391,3,2), membership391, 같은 adapter/calibration의 moment/rest/pool/ltm/IDs와 start/state.json groups를 직접 읽는다. StateStore/GraphStore/body 객체를 생성하지 않는다.

E163 raw-derived의 control phase/availability 및 E150 E1 입력 provenance를 사용한다. 해당 원본과 부모207검산/final review를 해시로 연결한다. phase는 목표 표본 작성에만 쓰며 새 전달의 입력에는 절대 들어가지 않는다. E85 commands와 E103 startup-inputs를 읽는다. 원래 전체10초에서 control1..2000만 decoder 출력이며 control0은 복원 관측이다.

구조와 시계 검사는 E163와 같다(5ms control,20 neural ticks/control, clock atol2e−8). bank0/activation0이 원래 초기 상태와 일치하고 all2001 availability/complete를 만족해야 한다. 저장된 이웃 bank와 endpoint MN rate를 대입한 recurrence residual을 기록한다. 새 순차 EMA rollout은 없으며 저장 recurrence2000개를 대조할 뿐이다. 원래 runtime의 alpha=-expm1(-.005/tau), bank+=alpha*(rate-bank)와 같은 float64 식을 검사한다.

기존 activation은 fixed LTM/singleton에는 저장 original_activation을 그대로 쓰고, 나머지는 원래 E149 계산 순서(sum(low*bank)+sum((high-low)*max(bank-kappa,0)))로 재구성한다. activation·42 target·6LTM·rate/roster/mapping을 원기록과 비교한다. numeric residual은 assertion 전에 raw에 저장하며 activation/target/tendon 재구성 atol1e−12, bank recurrence atol1e−12를 유지한다. t0 target은 decoder 등식에서 제외한다.

## 새 전달과 정확한 자유도

저장 bank의 fast/slow 값으로 각 MN i의 p_i=max(b_i,5ms−b_i,40ms,0)를 정의한다. **하나의 onset branch만**, knot/lag/tau/offset 탐색 없이 사용한다. slow branch를 빼는 것은 같은 MN의 역사 의존 전달 가정이며 새로운 음의 신경 연결이 아니다. 동일 정상 입력의 두 bank가 같아지면 p=0이다. 단, 이 식이 생물학적 흥분성 시냅스 단조성을 보존한다고 주장하지 않는다.

가변 muscle m의 activation은 a'_m=g_m*a_m+Σ_{i∈m} beta_i*p_i다. a_m은 frozen E148 전달을 그대로 적용한 E150 activation이며 기존 slope/kappa를 다시 fit하지 않는다. 각 muscle의 기존 MN만 기여한다. 출력 q=rest+.02*(activation@moment.T), LTM 식/상한·rest/moment/body/서보/힘줄은 바꾸지 않는다. beta는 비음수, 원래 위치 moment의 부호는 유지한다. 이 식은 최종 실행에도 MN의 현재/과거 bank만 필요하며 외부 위상/시간/교사/참조 배열은 필요하지 않는다.

고정 muscle은 기존 LTM6 또는 원래 MN singleton 전체다. g=1/beta=0 및 activation/tendon bit-exact를 요구한다. 가변 muscle 오름차순의 g, 그 다음 가변 muscle에 속한 MN을 graph column 오름차순으로 나열한 beta, 마지막 margin t 순서로 벡터를 만든다. muscle count를16씩 자르지 않고 groups/columns/membership의17/15/16/17/15/16 소유권과 실제 moment의 cross-leg0를 검증한다.

기존 slope로 mu_i=Σ_k min(w_low,w_high), L_m=Σ_i mu_i, U_m=Σ_i,k max(w_low,w_high)를 계산해 raw에 보존한다. 각 variable muscle의 원래 MN수는 n_m이다. 선형 제약:

- .5≤g_m≤1, 0≤beta_i≤2.
- g_m*mu_i≥1e−6/n_m (기존 MN 기여 floor).
- g_m*L_m≥.5 (기존 tonic lower budget).
- g_m*U_m+Σ_i beta_i≤2 (원래 upper envelope 안에서 tonic/onset 재분배).

이는 tonic 하한과 비음수 전달 크기의 보수적 예산이다. 새 함수 전체의 모든 history 방향 도함수가 양수라는 뜻은 아니다. beta에 양수 floor를 강제하지 않아 기존 전달 g=1/beta=0을 정확히 포함한다. baseline witness를 먼저 raw로 저장하고 affine/직접 계산이 동일한지 확인한다. 기존 가중치가 예산에 들어가는지는 직접 검산하며, 위반 시 clip/정규화/자동 수정하지 않는다.

## 목표, 제약, 선택

E163 phase가 여섯 다리 모두 available이며 .5≤control_time<10인 control index를 그대로 사용한다. 구간은 하나의 연속 index열이어야 한다. freeze가 이 기존 mask의 index/hash/count를 고정한다. 누락된 끝 구간을 외삽하지 않는다. 목표는 원래 own_leg_targets 방식: 각 다리 E1 phase−E85 offsets[0,.5,0,.5,0,.5]에서 같은 다리7좌표를 보간한다. 참조는 기존 default E85 time≥5; LF phase column0과 저장6offset 일치를 확인한다. 기존5..8→8..10 보간검사 RMS≤.002rad를 유지한다. 새 cross-leg phase 목표는 없다.

baseline은 **이번 E150 실제 명령**의 해당 target/adjacent velocity 오차다. E142를 새 baseline으로 사용하지 않는다. 인접 velocity=(q[k+1]−q[k])/.005이며 실제 연속 mask의 인접 쌍만 쓴다. 초기99는 원래 E103 control1..99 target_command/target_tendon과 비교한다. phase를 startup 목표로 대체하지 않는다.

단일 SOCP는 max t, t∈[−1,1]. 전체 command/velocity, 다리별6 command/6 velocity, 전체 startup command의15개 SOC를 기존 방식 norm/limit≤1−t로 구성한다. 한도는 전체 command baselineRMS*.8, 전체 velocity와 각 다리 command/velocity baselineRMS, startup qRMS .018이다. 원래 startup samplewise |error|/.04+t≤1의 양·음 제약을 추가한다. LTM는 고정이므로 startup tendon RMS≤.009/max≤.025를 별도 확인한다. 완성 후보의 물리 단위 직접 검산을 solver status보다 우선한다.

단일 공학적 제안은 overall command RMS20% 이상 개선, overall/각 다리 velocity 및 각 다리 command 비악화, 원래 strict startup, 해부학/예산/fixed-output를 모두 만족해야 한다. 재구성 개선을 실제 보행 개선으로 부르지 않는다. 이20%는 이번 출력 가정의 제안 조건이며 자연 보행의 새로운 필요조건이 아니다. norm0 한도는 임의 epsilon으로 바꾸지 않고 계약 오류로 보존한다.

원래 E145 thin QR을 truncation 없이 다리별 affine 열과 offset에 적용한다. 압축 전/후 witness와 생성 벡터 norm을 검산(1e−12+1e−10*directnorm)하고 모든 QR을 저장한다. 희소 전역 embedding을 사용하며 거대한 dense global matrix는 만들지 않는다. g=1/beta=0/t=−.251 witness의 feasibility를 raw 이후 확인한다. 이를 만족하지 못하면 fit에 진입하지 않는다.

Clarabel0.11.1, CPU1/max_threads1/qdldl/max_iter100/time_limit20, tol_gap_abs/rel/feas/infeas_abs/rel1e−10, 그 밖에는 defaults. 한 번 생성·solve만 허용하며 반환한 한 벡터만 평가한다. solver 실패/근소한 음수 beta를 clipping하거나 다른 iteration을 선택하지 않는다. 실제 direct 물리 단위 예산 atol1e−12이며 equality/fixed 보존은 명시한 exact/재구성 한도대로다.

E145 safe solve wrapper와 callback ledger를 읽기 전용으로 재사용한다. P=0, q 마지막−1; 실제 모든 g/beta/t box를 사용한 dual 보정과 self-dual cone projection(pad1e−12), 상한pad1e−8을 계산한다. proposal 실패이며 보정 margin upper bound<0이면 이 **고정된 tonic-rescaling+onset family/고정 신경 신호**에만 수치적 반증으로 보고한다. 불가능성의 exact proof 또는 BANC 전체 반증은 아니다. 그 외 실패는 inconclusive/valid rejection을 구분한다.

## 구현·저장·자원

부모 설계 → **실제 gpt-5.6-sol/high** 첫 구현 → 부모 첫 리뷰 → 같은 Sol 첫 수정 → 부모 두 번째 리뷰/잔여 수정. 기존 state 배열을 사용하는 새로운 affine/볼록 최적화와 파일 검증을 연결하므로 high를 사용한다. 새 신경/CUDA/state runtime 구조는 이 단계에 없다. 소유 파일은 tools/fit_c_banc_m3_onset_recruitment.py, tests/test_c_banc_onset_recruitment.py 두 개다. 이전 helper와 결과는 읽기 전용이다.

CLI --freeze/--freeze-sha256/--source-review/--source-review-sha256/--out. 부모 final gate와 contract-only preflight 이후 원본1회/자동재시도0. 실행 전후 frozen 입력/소스/과거·M2 및 live freeze/gate/snapshot을 확인한다. 중복 hash map은 병합해 전후 각 한 번 읽는다. 원본 raw의 수치 제약 검사 전에 입력 비교 residual/phase label/onset feature/affine map/constraints/witness/QR을 pre-solve raw로 보존한다. pre-solve 저장에 실패하면 solver0이다. solver 반환x/s/z/history/settings를 이후 계산 전에 partial에 게시하고 즉시 별도 raw를 저장한다. 후보 activation/q/velocity/startup/tendon과 scalar 결과도 최종 판정보다 먼저 저장한다.

기술적 유효성과 conditional proposal을 별도 필드로 둔다. 늦은 저장 오류/마지막 저장 시간·용량 초과는 이미 존재한 성공 receipt를 취소하며 원래 오류·partial·임시 파일은 보존한다. solver constructor/callback 실패와 solve 진입을 구분한다. JSON NumPy scalar 변환을 포함한 full generated fixture, 부호/소유권/zero-onset baseline/steady bank/fast-slow 교환/QR/15 SOC/초기 point/invalid clock/늦은 저장/terminal resource 테스트를 만든다. 원본1회 실행은 부모가 감독한다.

CPU1/GPU0, body/GraphStore/StateStore/neural/physics/new EMA rollout0, solver≤1. fit20/internal80/external95/supervisor105초, output192MiB/free1GiB. E163 실제 보수누적2789.927225499996초를 올림한2790에 감독105를 더한 사전2895초다. 이전2848은 종료된 배정 상한이고 누적 비용을 초기화하지 않는다. 성공해도 native adapter/실제BANC 보행/차단·저장/확인·회귀는 별도 다음 설계가 필요하다. 전체 goal과 F1–F8는 active다.
