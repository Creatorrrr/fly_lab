# E277 — 원래 문제의 최적성 인증과 수용된 이동 폭을 쓰는 조건부 학습

2026-09-27 root 사전 설계. 기존 finite-goal과 F1–F8은 유지한다. E275는 학습 RMS 0.15007225375308153 / 중립 0.1613248403800355로 약 6.98% 개선했지만, 요구 20%와 LM/RM 개별 개발 기준에 미달한 **valid reject**다. E276 단위 변환은 원래 세 문제 중 1개만 Solved여서 **valid reject**다. 둘을 소급 승격하지 않는다.

## 근거와 가설

`verification/banc-natural-gait-m3-20260919/parent-e276-slack-location-analysis.json`은 저장된 원본 해만 읽었다. 선형 제약과 전체 상수인 LTM cone의 affine 잔차는 반올림 수준이다. 큰 잔차는 q SOC의 보조 slack과 `b-A*d` 사이에서 발생한다. 따라서 상수 LTM 제약 삭제를 다음 수정으로 채택하지 않는다.

`parent-e276-original-kkt-analysis.json`에서 E275의 LM/RF/RH 원래 d와 z는 원래 primal/dual cone에 들어가고, 원래 목적함수의 primal-dual gap은 각각 5.1701e-12, 8.9017e-11, 8.5722e-10, stationarity infinity norm은 약 2.1e-15 이하다. 이 수동 계산은 새 solver 실행 0회이며 신경/보행 성공 증거가 아니다. E275에서 채택 47개에 비해 거절 제안 240개가 있었고, 작은 alpha로 채택해도 다음 radius가 그대로여서 평가를 반복 소비했다.

가설: 원래 QP/SOC 문제의 독립 최적성 인증으로 수치 slack 때문에 종료되는 경우를 구분하고, 수용된 alpha를 다음 radius에 반영하면 같은 평가 한도에서 유효한 학습 단계가 늘어난다. 경쟁 설명은 소스 필터의 표현력/교사와 신경 상태의 불일치다. 특히 LM/RM 개발 오차 악화는 이 수치 수정만으로 해결된다고 가정하지 않는다. 이번 한 번의 결합 변경으로 두 구성요소의 개별 인과 효과를 주장하지 않는다. 다시 미달하면 단순 반복 횟수 증가나 solver 설정 탐색 대신 신호/학습 목표의 구조를 재검토한다.

## 변경과 불변 항목

새 `tools/banc_m3_conic_certificate.py`, `tests/test_c_banc_conic_certificate.py`, `tools/fit_c_banc_m3_certified_source.py`, `tests/test_c_banc_certified_source_fit.py`만 제품/테스트 변경한다. E275/E276 및 이전 소스는 수정하지 않는다. 실제 gpt-6-sol/xhigh가 구현하고 root FIRST, 필요 시 같은 Sol 첫 수정, root SECOND, 남은 수정은 root가 수행한다. 재위임하지 않는다. Sol은 작은 생성 CPU 문제만 실행하며 원본 자료/900변수 benchmark/CUDA/BANC/body/fit 권한은 0이다.

기존 원본 d 문제, P=1e-3 I, q=gradient/maxabs, 원래 99개 시간점 peak와 두 SOC, trust/coefficient/triple 경계, Clarabel 0.11.1/100iter/2초/thread1/tol1e-8, 실제 nonlinear eligibility와 Armijo, LF/LM/LH/RF/RM/RH 순서, 24outer/80unique/30초/공유180초/원본290초를 유지한다. E276 scaling은 사용하지 않는다. 중립 재현, train-only 단계 1..1200, 최종 단회 f32 rounding 및 1..2000 평가, startup q .018/.04와 LTM .009/.025, train RMS 비율 .8 / 개발 1.0 / 다리별 1.05를 유지한다. 모델/교사/수용 기준/F6/M2/E150/UI를 바꾸지 않는다.

**내부 방향의 사전 판정 계약만 명시적으로 변경한다.** E275/E276는 `Solved` 문자열을 필수로 요구했으며 그 판정은 유지한다. E277부터 `Solved` 또는 `AlmostSolved`인 raw 결과에 동일한 독립 primal/dual 인증을 요구한다. 어느 상태도 문자열만으로 통과하지 않는다. 시간/iteration 한도, infeasible, numerical error 등 다른 상태는 인증 여부와 관계없이 거부한다. 기존 신경 후보 실패를 통과로 재해석하는 변경이 아니다.

## 독립 인증 수학과 API

`conic_certificate(problem, direction, solver_s, solver_z, status)`는 JSON 가능 receipt와 복원한 `primal_slack`을 반환한다. 원래 입력 불변. float64 유한 shape, canonical CSC/finite A/P, 원래 3 cone의 정확한 row count, P=1e-3 I를 확인한다. 다른 Hessian을 대각으로 오해하지 않는다. 원래 입력 제약 metadata에 대해서도 E275 `conic_residuals`를 호출한다. 비유한/잘못된 배열은 인증하지 않고 malformed problem은 명시적으로 실패한다.

1. `s_exact = b-A*d`를 계산하되 raw `solver_s`를 덮어쓰지 않는다. 원래 `conic_residuals` minimum >= -1e-8과 모든 `s_exact`의 cone slack >= -1e-8을 함께 요구한다. raw affine infinity norm과 raw cone slack도 기록한다.
2. z의 nonnegative 각 원소 >= 0, 각 SOC의 z0 >= norm(z_tail)를 **허용오차/clip/projection 없이** 요구한다. z를 변형하지 않는다.
3. `w=q+A.T*z`, `stationarity=P*d+w`. infinity norm <= `1e-8*(1+maxabs(q))`.
4. primal `p=.5*d.T*P*d+q.T*d`, dual lower bound `l=-.5*dot(w,w)/.001-b.T*z`. 이는 실제 원래 convex quadratic의 Lagrange 하한이다. finite p/l와 `gap=p-l`을 기록한다. `scale=1+max(abs(p),abs(l))`에 대해 `-1e-10*scale <= gap <= 1e-8*scale`, 그리고 `abs(dot(s_exact,z)) <= 1e-8*scale`를 요구한다. 음의 미세 gap 범위는 float64 산술 확인용이며 원래 primal gate는 그대로다.
5. 상태가 Solved/AlmostSolved 중 하나이고 1–4를 모두 만족할 때만 인증한다. 모든 개별 검사와 원래 상태를 남긴다. 원래 E275 valid 여부도 별도 보존한다. 불완전 해/검사 실패는 bounded rejection이며 성공으로 보완하지 않는다.

`solve_certified(problem, time_limit, *, solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter)`는 E275 `solve_direction`을 정확히 1회 호출한다. retry/fallback/warm start/추가 solve 0. 원래 d/s/z와 receipt를 보존하고, certificate와 primal_slack 및 원래 valid를 추가하며 새 valid는 인증 결과다. constructor/solve 실패의 primary exception, partial solution과 비용을 유지한다. 인증 단계 오류도 동일 primary에 기존 solver receipt/raw를 첨부하고 인증 단계 비용/상태를 기록한다. 알 수 없는 비용을 0으로 위장하지 않는다.

## 실제 학습 루프와 저장

E275 `fit_conic_leg`와 같은 독립 신규 루프 `fit_certified_leg`를 작성하되 위 solver와 radius 정책 두 부분만 변경한다. 수용 alpha가 1이면 기존처럼 `min(.5, 2*r)`, 1 미만이면 `max(1e-6, alpha*r)`로 다음 radius를 설정한다. 전부 거절이면 r/2, 최소 radius 종료 조건은 그대로다. 각 제안에 채택 당시 r/alpha/next_radius를 남긴다. 정해진 alpha 목록 및 actual feasibility/학습 비악화/음의 slope/Armijo를 그대로 확인한다. 개발 값은 최종 판정에만 사용하며 선택/재선택하지 않는다.

E275의 오류/예산 만료/최초 exception 보존/중간 history/selected persistence/원본 만료 재전파를 그대로 유지한다. `abort_run`을 무시하지 않는다. SOC 파일에 원래 d/s/z와 추가 primal_slack을 함께 남기고 history에 인증의 모든 수치/판정을 보존한다. 비용은 actual constructor/solve/iterations와 인증 wall 및 Solved/AlmostSolved 중 실제 인증한 수를 구분한다. 단회 최종 f32 평가와 후보 선택은 기존 base 함수로 수행한다.

새 runner schema/CONTRACT/source-review에는 원래 E270 입력·E273 exact CUDA gate와 E275 동결본을 근거로 실제 12 source hash를 고정한다. 새 freeze/hash 검증을 우회하거나 monkeypatch하지 않는다. import는 파일 읽기/원본/solver 실행을 하지 않는다.

## 검증과 원본 실행 경계

생성 작은 convex 문제에서 direct minimizer/검증 가능한 dual 하한과 원래 판정의 일치를 확인한다. 상태가 Solved여도 잘못된 d/z/중간 raw는 거부하며 AlmostSolved라도 primal/dual/gap 중 한 항목이라도 실패하면 거부한다. 내부 raw slack만 어긋나고 원래 d/z가 인증되는 경우는 raw와 잔차를 보존한 채 새 정책으로 구분한다. P off-diagonal, cone row mismatch, NaN/inf, 불완전 raw, primary/secondary 실패, 원본 만료 identity, alpha<1의 radius 반영 및 평가 경계/선택 불변을 검사한다. root는 저장된 동일 900변수 E275 benchmark와 LM/RF/RH 3개에 solver 재실행 없이 인증을 적용해 독립 원래 수학과 비교한다. 이는 과거 E275/E276의 성공 여부를 바꾸지 않는다.

이 검증 이후에만 root가 중립에서 시작하는 한 번의 원본 fit을 별도 동결/예약한다. 후보 선택 1회, neural 적분 최대 forward12,984,000/derivative12,480,000step, whole BANC/body 0이다. 최대 144 Clarabel 호출, 290초/6GiB/한 CUDA 작업. 현재 원본 누적 7609.155666199575 / 배정 7805초, 잔여 195.84433380042537초이며 **이 문서는 아직 원본 예약이 아니다**. 필요 배정 조정은 사전에 별도 기록하고 과거 비용을 누적한다.

유효 실행은 중립 재현/입력·소스 일치/유한 기록과 실제 원본 최종 판정 완료로 결정한다. 방향 알고리즘의 유용성은 중도 solver 거부/accepted 및 rejected 비율을 E275와 함께 기술한다. 후보 승격은 원래 전체 conditional gate를 모두 통과해야만 가능하다. 통과해도 실제 신체/F1–F8의 완료가 아니며 E272 계약의 후보 결합을 사전 갱신한 뒤 실제 body 검증이 필요하다. 미달 시 원래 F1–F8과 unmet claims를 유지하고 구조적 다음 설계로 간다.
