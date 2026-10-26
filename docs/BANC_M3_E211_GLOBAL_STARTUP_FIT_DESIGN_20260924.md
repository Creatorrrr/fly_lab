# E211: 원래 전신 출발 제약의 결합 적합

E210 원본 여섯 문제는 세 AlmostSolved와 세 PrimalInfeasible로 끝났다. 부모는 물리 행렬과 dual을 독립 검산했다. E210에서 추가한 각 다리 startup RMS .018은 원래 전신 startup RMS .018보다 엄격하다. 저장된 infeasibility certificate는 전신 조건의 다리별 필요조건 .018*sqrt(6)에서 더 이상 불가능을 증명하지 못한다. 후속은 기저/신경식/목표/합격 기준을 바꾸는 재탐색이 아니라 원래 전신 norm을 정확히 적용하는 한 번의 결합 문제다. E210 실패와 추가 제약을 보존하고 같은 문제의 재시도로 세지 않는다.

## 고정 데이터와 물리 변수

E210 원본 여섯 `pre-solve.npz`, norm factors와 manifest/부모 검산을 고정한다. E209 B/H/D·scale·center·sigma·16ms 기억·gate·cap·tau·canonical decoder 및 원래 E150/E197 표본과 목표를 유지한다. 자유343 MN×16=5488 beta의 물리 box0..1, fixed48 LTM와 근육별 소유권은 그대로다. 다리 순서는 LF/LM/LH/RF/RM/RH, 폭976/864/896/992/880/880이다. 각 block의 motor_columns가 전역 free343을 중복 없이 덮고 원래 own-leg moment와 맞는지 확인한다. 약한 column pruning/변수 scaling/사후 clip은 없다.

목적은 E210 다리별 목적의 합이다. 전역 P는 저장된 여섯 upper-triangle P의 block diagonal, linear는 이어붙인 값, constant는 합이다. 즉 sum_leg(||C_leg*beta_leg+c_leg||²+1e-8*mean(beta_leg²))다. 목적 가중치를 새로 튜닝하지 않는다. 원래 전체 q20% 개선은 직접 평가에서 그대로 검사한다.

## 원래 출발 제약을 보존하는 결합

각 저장된 local A/b에서 첫 세 cone, 즉 beta box와 모든 startup 점별 .04 제한의 nonnegative cone, command와 velocity 두 norm cone을 그대로 가져와 해당 variable block에 배치한다. E210의 네 번째 cone인 각 다리 startup RMS .018만 제거한다. 대신 startup_matrix(693×n_leg)의 block diagonal M(4158×5488), leg-major startup_offset e(4158)을 구성한다. 모든 다리/관절/99시점을 한 번씩 포함하므로 leg-major 순열은 원래 시간-major 전신 RMS와 정확히 같은 norm이다.

단일 global SOC는 `[0;-M/sqrt(4158)]*beta+s=[.018*PAD;e/sqrt(4158)]`다. PAD=1-1e-6은 원래 E210대로다. box에 PAD를 적용하지 않고 per-leg q/velocity/점별 startup PAD도 그대로 유지한다. cone 순서는 다리별 nonnegative, qSOC, vSOC를 여섯 번 이어붙인 뒤 globalStartupSOC 하나, 총19개다. 전체 A는34451×5488이다. 전역 Gram/eigendecomposition을 계산하지 않고 이미 검산한 sparse local cone/P를 재사용한다. 전역 dense q Jacobian도 만들지 않는다.

이 결합은 같은 E209 기저에서 **원래 전신 startup 조건을 보존**한다. 더 엄격했던 각 다리 충분조건에 매달리지 않는 것이며 F1–F8 및 원래 proxy 문턱의 완화가 아니다. 전체/다리별 q와 velocity, startup pointwise, beta box 및 고정 tendon 조건은 남아 있다.

## 실제 반환·검산·대조

한 번의 Clarabel CPU1, max_threads1, max_iter150, tol_gap_abs/rel/feas1e-8, time_limit180초를 사전에 고정한다. 다른 기본값과 버전을 저장하고 원본 solve는 최대1회, warmup/원본 탐색/재시도0이다. 180초는 solver가 반복 경계에서 확인하는 soft limit이며 하드 외부 감독은270초다. 준비와 constructor 후 남은 내부240초가180초 이상인지 검사하여 부족하면 실제 solve를 차단한다.

모든 원시 P/A/b/linear/constant/cone 종류·크기/offset/열 소유권·startup M/e, solver x/s/z, 설정/status/residual/objective/호출·constructor 시간을 판정 전에 저장한다. 저장 실패는 미실행을 실행으로 세지 않는다. 기대된 infeasible/time-limit/AlmostSolved는 정상적으로 기록한 미해결이고 코드/shape/저장 실패와 구분한다. finite raw beta는 진단을 보존하되 `Solved`와 strict box0..1을 통과한 완성 beta만 후보 대조를 만든다. 기준 변경이나 clip으로 결과를 승격하지 않는다.

완성 beta는 H+cap*B*beta에서 rate를 재구성하고 fixed48은 이번 조건부 대조에서 원래 rate를 유지한다. 실제 canonical 순차 decoder와 affine q를 atol/rtol2e-12로 비교한다. 고정 LTM activation/tendon은 exact다. 새/원래 rate×monotone/canonical 2×2를 direct/proxy 실패에도 보존한다. 근육별 해부학은 바꾸지 않는다. 실제 native 통합은 별도이며 여기서는 저장 rate 재생을 자율 신경망으로 부르지 않는다.

원래 합격 proxy: 전체 qRMS<=.8*.056686385349786386, 전체와 다리별 velocity·다리별 q 비악화, 전신 startup qRMS<=.018/max<=.04, tendonRMS<=.009/max<=.025. 각 다리 startup RMS는 보고하지만 더 이상 합격 게이트가 아니다. 원시 전역 cone의 primal 및 direct 숫자를 각각 확인하며 부모가 원자료를 독립 검산한다. solver 수치 residual1e-6은 수치 검사이고 실제 proxy를 느슨하게 만드는 값이 아니다. 불가능 certificate는 이 고정 조건부 기저에만 해당하며 자연 보행 전체의 불가능 주장으로 쓰지 않는다.

## 절차와 자원

부모 설계→실제 gpt-6-sol/high가 새 `tools/fit_c_banc_m3_global_startup.py`, `tests/test_c_banc_global_startup_fit.py`만 구현→부모 FIRST→필요시 동일Sol 첫수정→부모 SECOND/잔여수정 순서다. E210/E209 원본과 소스는 동결한다. 생성 수학/모의 검증은 block 순서/희소 연결/원시 norm 등가/목적상수/원본 per-leg startup 게이트 제거/실패 후 raw 보존을 확인한다. 부모 리뷰 이후 실제 생성 Clarabel1회만 실행한다: box0..1, 목표 y=(.1,.1), startup center=(.8,.2), global RMS<=.25의 analytic 해(.45,.15). 이 해는 전신 조건을 지키면서 첫 다리 .25 제한을 초과하므로 결합 의미도 검증한다. 원본 데이터의 예비 solve는 없다.

원본 프로세스1/CPU1/GPU0/solver1, 내부240/solver180/외부270/감독300초, RSS8GiB/출력512MiB/free4GiB다. 기존 여섯 sparse A는 약10.8M nonzero 수준이며 dense 전역 Gram을 만들지 않는다. 실제 factorization 메모리/시간은 미검증이며 감독 한도에서 남은 원시 결과와 실패를 보존한다. 부모는 준비 단계에서 행수·nnz·예정 byte만 점검하고 별도 원본 탐색을 하지 않는다.

E210까지 보수적 원본 누적5616.236763699704초를 보존하며 단계 배정5784→6004(+220), 잔여387.763236300296초에서300초를 사전 예약한다. 생성/준비/부모 검산 비용은 별도 기록하고 미측정을0으로 세지 않는다. 사용자 전체 예산을 신설하지 않는다. M2/최선E150/UI/F6·실패 이력/F1–F8 active·미완료를 보존하며, 조건부 fit 성공도 최종 보행 완성이 아니다.
