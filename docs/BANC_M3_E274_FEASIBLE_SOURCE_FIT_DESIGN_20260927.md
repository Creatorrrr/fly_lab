# E274 — 시작 안정성을 유지하는 source 계수 학습

2026-09-27, root 설계. 동일한 M3 finite-goal과 F1–F8을 유지한다. E270은 유효한 단회 실험이었지만 고정 기준을 통과하지 못했다. 306개 평가 중 대부분의 다리에서 개선 후보가 원래 startup 한도를 크게 넘었고, 각 다리의 30초가 소진됐다. 실패와 원본 소스는 `verification/banc-natural-gait-m3-20260919/e270-original-reviewed-sources/` 및 `parent-e270-original-final-review.json`에 보존됐다. 수치 반올림이나 selector margin을 완화할 근거가 없다.

가설 H274: 같은 E267의 1660개 기존 간선, 같은 비음수 3계수와 고정 E150 decoder에서, 실제 startup 제약을 만족하는 단계만 채택하는 학습은 E270보다 유효한 개선 후보를 찾기 쉽다. 대안 설명은 이 국소 파라미터화의 안정적인 영역에서 필요한 개선량 자체가 나오지 않는다는 것이다. E273은 같은 수학의 조건부 CUDA 실행만 가속하며, 두 변경을 합친 최종 결과를 optimizer만의 인과 효과로 해석하지 않는다.

## 구현 소유권과 리뷰

실제 gpt-6-sol/high가 다음 파일만 수정한다.

- `tools/fit_c_banc_m3_feasible_source.py` — 새로운 runner와 feasible-step optimizer.
- `tests/test_c_banc_feasible_source_fit.py` — 생성 CPU 검사.
- `tools/fit_c_banc_m3_source_coefficients.py` — `execute_original(..., *, leg_fit=None)`과 `(leg_fit or fit_leg)(...)`의 명시적 주입만 추가한다. E270 기본 실행, 계약, 기존 root 오류 보존 수정을 유지한다.

E269/E273 engine 파일과 기존 E270 test 파일은 수정하지 않는다. root FIRST → 같은 Sol 첫 수정 → root SECOND → root 잔여 수정 순서다. 에이전트는 생성 CPU 검사와 Ruff만 실행한다. 원본 자료, CUDA, BANC, body 실행은 0회다.

## 고정 입력과 불변 조건

E270의 원본 입력 SHA, 여섯 다리 LF/LM/LH/RF/RM/RH 순서, leg contraction, startup 99개 행, train 1:1201, development 1201:2001, 중립 재현 검사 109개, fixed outside-leg SSE, 선택 및 최종 판정 코드를 그대로 재사용한다. optimizer에는 training 96000 stages와 해당 목표만 보인다. 개발 구간은 최초 중립 재현과 최종 단회 판정에서만 사용하며, fitting objective와 gradient에는 들어가지 않는다. 기존 개발 구간은 이미 노출된 development이고 독립 확인 데이터라고 부르지 않는다. F6 두 pose는 계속 미노출이다.

계수는 f64, 각각 [0,8], triple 합은 [0.125×(1+1e-6), 8×(1-1e-6)]이다. objective는 기존 `train_MSE/neutral_train_MSE + 1e-4*mean((c-neutral)^2)`다. startup 제약은 기존 `startup_constraints`의 전역 q RMS .018, LTM RMS .009, q 최대 .04, LTM 최대 .025이며 selector 상대 margin 1e-4를 유지한다. 이것을 다리별 RMS로 바꾸지 않는다. 매 다리 이후 선택된 startup SSE가 다음 다리의 고정 outside SSE에 반영된다.

## Feasible-step 알고리즘

중립 계수에서 시작하여 실제 평가값이 유한하고 interior coefficient/startup 제약을 만족하는지 확인한다. 다리당 outer step은 최대 24회, 실제 고유 평가 최대 80회, 다리 30초, 공유 fitting 180초, 원본 supervisor 290초를 유지한다. 휴지나 예산 종료 후 재시작하지 않는다.

현재 채택된 feasible c의 objective gradient g, startup constraint h>=0, Jacobian J에서 `scipy.optimize.linprog(method='highs')`로 방향 d를 구한다. 목적함수는 g/max(abs(g))와 d의 내적이며 g가 정확히 0이면 stationary로 종료한다. 정규화는 LP의 수치 스케일만 바꾸며 실제 목적함수는 그대로다.

- trust radius 초기 .05, 최대 .5, 최소 1e-6. 각 d는 [-radius,+radius]와 c+d의 [0,8]을 동시에 만족한다.
- triple 합의 interior bound를 c+d에 적용한다.
- 선형 startup 제약은 `J d >= -0.8 h`로 두어 현재 선형화 여유의 20%를 남긴다.
- HiGHS primal/dual feasibility tolerance 1e-9, maxiter 2000, LP time limit은 min(1초, 남은 다리/공유/원본 시간)이다. 성공한 유한 해만 사용하고 LP 상태, nit, 실제 wall, 방향, 계수 및 선형 제약 residual을 저장한다. LP 실패/시간 초과는 해당 다리의 bounded 종료이지 파라미터 전체의 불가능성 증명이 아니다. SLSQP 호출은 0회다.
- 후보 alpha는 1, .5, .25, .125, .0625, .03125 순서다. c+alpha*d를 [0,8]로 clip하는 것은 floating-point box 잔여 오차 정리에 한정한다. triple interior 또는 유한성 실패는 신경 계산 전에 거부한다.
- 각 실제 후보에 대해 기존 startup 제약 모두 >=0, train MSE <= neutral, `g dot (candidate-current) < 0`, `candidate_objective <= current_objective + 1e-4*g dot (candidate-current)`가 성립해야 다음 iterate로 채택한다. 원래의 true constraints를 계산하므로 최대 오차가 발생하는 시간 인덱스가 바뀌어도 검사한다.
- 첫 통과 후보를 채택한다. alpha=1이 통과하면 radius를 두 배로 하되 .5 이하로 한다. 여섯 alpha가 모두 실패하면 radius를 반으로 줄이고 다음 outer step으로 간다. 최소 radius 아래나 영 방향이면 종료한다. 무한 backtracking이나 자동 restart는 없다.

완료된 모든 실제 평가는 기존 evaluator에 즉시 남긴다. 최종 다리 선택은 기존 `choose_candidate`와 같이 strict feasible 및 train MSE 비악화 후보 중 최소 objective, 동률 첫 후보다. budget-stop 직전에 계산이 완료된 후보도 보존한다. 미시작 다리는 중립이다. 선택/LP 이력 저장의 이차 실패가 원래 계산 오류를 덮어쓰지 않도록 기존 `preserve_primary`를 사용한다.

계수는 마지막에 한 번 f32로 반올림하고 여섯 다리의 startup/full teacher를 한 번 평가한다. 채택 조건은 E270과 동일하게 원래 strict startup, 전역 train RMS <= 중립의 .8, development RMS <= 중립, 각 다리 train/development RMS <= 중립의 1.05다. 이 조건을 통과해도 조건부 지원이며 자연 보행의 증거는 아니다. 통과한 동일 후보만 이미 설계한 E272 실제 BANC/body screen으로 간다.

## 실행 계약과 검증

새로운 E274 schema/CONTRACT/source-review를 사용한다. source pin은 E269 engine/test, E270 runner/test, E274 runner/test 총 6개다. E270의 four-source/SLSQP validator에 새 의미를 억지로 넣지 않는다. 안전한 입력 경로, SHA 검증, native gate는 재사용하고 E273 생성 CUDA exactness/speed gate도 명시적으로 요구한다. global monkeypatch는 금지한다. 함수 주입은 호출 인자로만 전달한다.

생성 CPU 검사는 feasible 경로, 비선형 제약 때문에 거부되는 제안과 backtracking, 고정 다리 외 SSE, 경계 계수, time/unique budget, 완료 후보 보존, LP 실패, 오류/저장 실패, 원본 entrypoint의 명시적 주입과 E270 기본 경로 보존을 다룬다. 기존 E270 검사도 유지한다. 실행한 작은 LP의 실제 호출/iterations/wall을 보고한다.

원본 cost에는 `solver_calls`가 SLSQP가 아닌 실제 LP 호출 수임을 명시하고 leg optimizer 호출(최대 6), outer step(최대 144), HiGHS iteration, 평가/거부/채택/alpha, 조건부 forward/derivative/current 비용을 따로 기록한다. LP마다 잔여 예산과 time limit을 남긴다. 전체 BANC 시간이나 물리 시간으로 환산하지 않는다. 실패/미완료 비용은 0으로 대체하지 않는다.

현재 E274 원본 예약은 0이다. root 리뷰 및 E273 생성 비교 후 단회 원본 예약을 현재 누적 자원에 추가하고, freeze를 저장한 뒤에만 실행한다. M2/E150/활성 UI와 F1–F8은 그대로이며, 새 이름으로 과거 비용과 실패가 초기화되지 않는다.
