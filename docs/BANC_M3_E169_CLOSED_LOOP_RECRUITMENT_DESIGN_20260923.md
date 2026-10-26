# E169 현재 폐루프 입력에서 양수 전달 계수 재적합

2026-09-23 부모 설계. M3 finite-goal과 F1–F8을 유지한다. 직전 목표 턴은 E168 원본 적합과 독립 검산, 제약 해석을 완료해 다음 방법 선택을 바꾼 **progress**다. E168의 기각·비용·미채택과 전체 미완료를 보존한다.

## 질문과 선택 근거

H169: E150의 현재 폐루프 BANC 운동 출력에 기존 양수 단일-knot 전달 기울기를 다시 적합하면, 기존 해부학·MN 하한·근육 합산 보호·초기 자세를 유지하면서 같은 조건부 명령 오차 20% 감소를 얻을 수 있는가?

E148은 E142 계열 저장 bank에서 조건부 후보를 만들었고 E150의 실제 보행은 지속성 기준에 미달했다. E164 onset 및 E168 resource는 E150 자료를 사용했지만 E148 기울기 전체를 다시 선택하지 않았다. E168 산출물에서 21/77개 근육은 감쇠 하한 여유가 작고 17개는 상한에도 근접하므로 작은 gain/감쇠 보정보다 branch 재배분을 먼저 확인한다. 다른 56개에는 여유가 있어 보호 제약 하나를 원인으로 단정하지 않는다.

경쟁 설명은 현재 신경 출력과 기존 전달 가족 자체가 같은 목표를 충분히 표현하지 못한다는 것이다. 입력 재적합과 새 동적 상태의 효과를 함께 변경하지 않는다. 기존 5/10/40ms bank, kappa와 low/high 함수는 그대로며 E168 자원 상태는 사용하지 않는다. 생물학적 근력·운동단위 측정의 복원이나 실제 폐루프 보행의 성공을 주장하는 실험이 아니다.

## 입력과 불변 조건

E164/E168과 같은 E150 2001행(control0..2000), bank(2001,391,3), actual/original activation(2001,96), motor_rate와 clock/complete/availability/체크포인트/해부학/실제 명령·힘줄을 사용한다. E168의 capture_loaded_input와 validate_loaded_input를 읽기 전용으로 재사용해 원형을 저장하고 dtype/shape/finite를 검사한 뒤 E164 prepare_saved로 기존 재구성을 검산한다. E164의 onset/layout은 새 특징이나 solver mapping에 사용하지 않는다. 원본 로더는 E164의 해시 검증된 파일/NPY reference 로더이며 StateStore/GraphStore/body를 만들지 않는다.

고정 19개 근육과 가변 77개/330MN, 17/15/16/17/15/16 다리 소유권, moment의 cross-leg0, rest/.02 gain, LTM 출력, kappa391와 tau[.005,.01,.04]를 보존한다. 실제 전달 node/edge를 제거하거나 추가하지 않는다. 참조·phase는 오프라인 목적에만 쓰고 최종 decoder 입력에 넣지 않는다.

목표는 그대로 tick100..1923의1824행과1823 인접 velocity다. E85 own-leg 참조/offset, E103 startup1..99, 전체 명령 RMS 기준의 .8, 전체/각 다리 velocity와 각 다리 command 비악화, startup q RMS .018/max .04와 LTM RMS .009/max .025를 유지한다.

## 가변 전달과 baseline 포함성

variable_motors는 원래 graph-column 오름차순330개, 각각 filter0/1/2 순으로 pair990개다. 같은 pair의 low/high를 interleave한 계수 w1980개를 사용한다. 원래 w0는 같은 순서의 E148 기울기다. b_low=min(bank,kappa), b_high=max(bank−kappa,0)이며 모두 비음수다. knot·bank·지연·위상·tau를 탐색하지 않는다.

저장자료 평가는 delta_a_m=sum_owned[(w_low−w0_low)b_low+(w_high−w0_high)b_high], a'=a_recorded+delta_a, q'=q_recorded+.02*(delta_a@moment.T)로 정의한다. w=w0이면 activation/q/velocity/startup이 exact baseline이어야 한다. 고정 19개에는 delta0이다.

별도 native 대응 검산은 기존 MonotoneRecruitmentAdapter와 같은 low*bank+(high−low)*positive_hinge 식으로 새 기울기를 평가하고 고정 근육은 원래 activation을 보존한다. tick1..2000의 activation/q 및 선택 인접 velocity를 anchored 결과와 atol1e−12로 비교한다. native 적용은 새 기울기를 현재 신경 bank에 적용하는 것이다. 기록 activation/command/교사 궤적을 runtime에 주입하지 않는다. t0은 기존과 같이 decoder equality에서 제외한다.

## 원래 물리 보호를 표현하는 SOCP

벡터 순서는 w1980, pair별 lower ell990, upper u990, margin t1로 **3961개**다. w/ell/u는[0,2], t는[-1,1]. 각 pair에서 ell<=w_low,ell<=w_high,w_low<=u,w_high<=u다. 원래 MN마다 sum_filter ell>=1e−6/n_m, 원래 가변 근육마다 sum_owned ell>=.5 및 sum_owned u<=2다. 이 epigraph는 실제 양수 전달 family의 MN sum(min)와 근육 sum(min/max) 보호를 표현한다. 최종 검사는 보조변수가 아닌 실제 계수 min/max로 한다.

E148의 solver 내부 수치 여유(.500001/1.999999 및 각 filter-branch guard)는 물리 보호 기준과 다르다. 이번에는 원래 실제 물리 기준(.5/2 및 MN별 합산 floor)을 직접 표현한다. 기존 E148 계수는 물리 여유 약1e−6가 있지만 내부 상한을 약7e−10 벗어날 수 있어 같은 계수를 엄격한 내부 guard의 feasible witness로 잘못 부르지 않는다. 물리·초기 자세·오차 판정이나 F1–F8은 낮추지 않는다. branch0은 허용해도 모든 원래 MN의 합산 양수 하한과 연결 provenance는 유지한다. 이는 과거 E148 내부-guard 가족과 정확히 같은 최적화 영역이라는 주장이 아니다.

선형 cone 행 순서는 모든3961 변수 lower/upper box7922행, ell−low/ell−high/low−u/high−u의3960행, MN floor330행, 근육 lower77/upper77행, startup positive4158/negative4158행으로 총20682행이다. 각 startup 양/음 제약과15SOC에는 같은 margin을 넣고 max t를 적합한다. affine coefficient 열은 .02*b_branch*moment[:,owner], offset은 recorded command−matrix*w0다. ell/u 열은0이다.

baseline witness는 [w0,min(w0_low,w0_high),max(w0_low,w0_high),−.251]로 고정한다. 원시 witness/slack을 저장한 뒤 물리 포함성과 모든 선형/SOC feasible을 검사한다. 실패하면 fit0의 기술적 실패이며 witness나 margin을 다시 고르지 않는다.

다리별 command/velocity/startup의18개 QR을 한 번씩 만들고 전체/다리별15SOC에서 재사용한다. envelope0열은 QR에 넣지 않고 embedding에서 추가한다. truncation 없는 thin QR과 E145 SOC helper를 재사용한다. QR 검산 vector는 w0, zero1980, .5+.5*cos(arange(1980)*.173)이며 직접/압축 norm의 허용차는1e−12+1e−10*direct norm이다. 실제 datum마다 shape를 검산하고 E148의2000/1827행 또는 예전 column mapping을 재사용하지 않는다.

Clarabel0.11.1, CPU1/max_threads1, qdldl,max_iter100,time_limit20, gap_abs/rel/feas/infeas_abs/rel1e−10 및 나머지 defaults, P0와 마지막 q=-1이다. constructor/solve 각1회이며 한 반환 벡터만 판정한다. w의 작은 음수나2초과를 clip하지 않는다. physical floor/budget/reconstruction은 기존 atol1e−12다.

실제3961 box와 cone projection pad1e−12, 보수적 upper pad1e−8을 사용해 dual margin 상한을 계산한다. 엄격한 solver b에 대한 값과, 최종 물리 허용오차를 포함한 b_relaxed에 대한 값을 함께 저장한다. b_relaxed는 MN floor330 및 근육 lower77/upper77의 행11882..12365(0-based)에만 physical_atol=1e−12를 더한다. box/epigraph/SOC/startup은 그대로다. 기각에는 **이 완화된 물리 보호 집합의 상한**을 사용해 엄격한 cone와 허용오차 판정 범위를 혼동하지 않는다. 모든 물리조건 통과 시 conditional_closed_loop_recruitment_proposal, 미달이며 검산한 relaxed upper<0이면 refuted_current_positive_branch_family, 그 외 valid_rejection_inconclusive다. 오류/시간 초과는 invalid_not_tested이며 과학적 기각으로 바꾸지 않는다. solver 상태명만으로 판정하지 않는다.

## 구현·검증·실행

부모 설계 → 실제 gpt-6-sol/high 첫 구현 → 부모 FIRST 리뷰 → 같은 Sol 첫 수정 → 부모 SECOND 리뷰와 잔여 직접 수정. 새 소유 파일은 tools/fit_c_banc_m3_closed_loop_recruitment.py와 tests/test_c_banc_closed_loop_recruitment.py다. E145/E148/E164/E168 helper와 과거 자료는 읽기 전용이다. E164 Criterion/ConicProblem, metrics_from_recorded/solver_raw/problem_arrays의 차원 일반 부분과 E145 safe solver를 재사용할 수 있다. E168 run/validate_freeze/build_problem/candidate_gates는 다른 실험 계약이므로 그대로 호출하지 않는다.

CLI --freeze/--freeze-sha256/--source-review/--source-review-sha256/--out, 실제 schema/hash gate와 독립 cwd 진입을 검증한다. 생성 시험은 양수 branch·MN 소유권·w0 baseline exact·fixed19·native 재구성·실제 min/max와 envelope 동치·3961변수/20682행/15SOC·18QR/box dual·초기 raw 실패·solver 반환 후 오류·최종 저장 시간/용량 실패를 다룬다. 원본 자료를 fixture로 사용하지 않는다.

입력 원형, basis/mapping/w0/재구성, affine/QR/problem/witness를 assertion 전에 보존한다. affine는 전체 dense 직접 행렬을 복제하지 않고 **branch basis, moment, 계수별 owner/인덱스, q_recorded, w0 및 각 행 선택**의 무손실 인수분해 표현으로 저장한다. 이 표현에서 직접 affine/offset을 결정론적으로 재구성할 수 있다. QR18개와 최종 문제의 sparse CSC P/A/b/q/cone/box 및 witness를 저장하며 전체·다리별로 같은 QR/행렬의 원시 복사본을 중복 저장하지 않는다. 생성 full-shape 저장/QR 비용을 동결 전에 측정하고 구현 후 실제 생성 실행으로 확인한다. pre-solve 저장 실패 시 fit0이다. solver bare 반환은 후속 I/O 전에 caller-owned raw에 게시하며, solver raw와 후보 activation/명령/속도/힘줄/가중치를 판정보다 먼저 저장한다. 첫 오류·완료 계산·secondary/temp를 보존하고 늦은 실패는 성공 receipt를 취소한다. 마지막 hash/snapshot/output hash/쓰기/용량/free/readback 이후 time gate를 적용한다. 종료 후 부모가 실제 물리식·행렬·dual을 별도로 검산한다.

원본 process1/자동재시도0, CPU1/GPU0, fit1 이하, 새 EMA/resource/neural/physics/body/StateStore/GraphStore0이다. fit20/internal70/external85/supervisor95초, output192MiB/free1GiB를 배정한다. 과거 보수 누적2927.27736190008초를 보존하고 ceil2928+95=**3023초**로 사전 배정한다. 마지막2953초보다70초 증가하며 계정 초기화가 아니다. 3961변수 재적합/원시자료와 늘어난 해시 검증 비용을 포함한 이 단일 실행의 배정이다. 생성 검증·부모 검산은 기존처럼 별도 기록한다. F6/M2/UI/과거 실패 보존, 후보 채택false, 전체 goal active다.

조건부 후보 통과는 다음 native 동일 후보 검증으로 이어진다. 실제 지속 보행과 F1–F8을 통과하기 전 M3로 통합하지 않는다. 재기각은 이 현재 신호와 전달 가족의 한계를 뜻하며, 실패 후 새 근거에 따라 목표 내부의 다음 구조를 선택한다.
