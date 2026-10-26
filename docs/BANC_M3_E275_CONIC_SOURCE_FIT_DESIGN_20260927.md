# E275 — 모든 시작 시간점과 RMS 곡률을 반영하는 source 학습

2026-09-27 root 설계. M3의 동일 후보 F1–F8, E150/M2, BANC provenance, 미노출 F6와 현재 UI를 유지한다. E270의 유효 reject 및 E274의 실행 오류/비용/부분 관찰을 보존한다. 기존 개발 구간은 development이며 새로운 독립 확인으로 해석하지 않는다.

## 바꾸는 이유와 불변 조건

E274는 401개 실제 조건부 평가, 71개 HiGHS 문제에서 29개 feasible 단계를 채택했다. 그러나 366개 제안이 실제 제약을 위반했고, 첫 다리 LF의 74개 거부 중 한 joint 최대 오차 위반이 73개였다. LM/RF/RM은 global RMS 위반도 대부분이었다. 현재 한 시간점의 argmax 미분과 SSE의 1차 근사만 사용하므로, 다른 시간점의 peak와 Jacobian 제곱 항을 놓친다는 설명을 검증한다. 경쟁 설명은 신경 출력의 비선형성이 이 source 계수 표현의 유효 영역을 여전히 좁힌다는 것이다.

E274 원본은 마지막 RH neutral q가 baseline과 bit exact인데 MSE 합산의 C/F 순서 차이 6.938893903907228e-18 때문에 중단됐다. root는 `mean_square`를 추가해 baseline/candidate/recorded training MSE를 동일한 C-order f64 순서로 계산하도록 수정했다. 이는 허용 오차를 늘리는 변경이 아니다. 과거 source6개와 모든 결과는 `e274-original-reviewed-sources/`에 보존했다. 저장된 부분 train RMS .160531951931(중립 .161324840380)은 최종 반올림/개발 판정이 아니며, 전체 학습의 성공을 의미하지 않는다.

H275: 실제 startup 모든 행을 제약에 넣고 선형화된 q/LTM의 **제곱 노름 자체**를 제한하면, 큰 비유효 방향을 줄여 같은 단회 학습 예산에서 개선을 더 얻을 수 있다. 원래의 실제 nonlinear startup/계수/최종 train .8/development1.0/leg1.05 gate는 바꾸지 않는다. 계수 1660×3, 5/40ms history, 입력/모델/E150 decoder, train1:1201 및 고정 다리 밖 SSE도 동일하다.

## 소유권과 API

실제 gpt-6-sol/xhigh가 새로운 `tools/fit_c_banc_m3_conic_source.py`, `tests/test_c_banc_conic_source_fit.py`를 구현한다. 추가로 `tools/fit_c_banc_m3_source_coefficients.py`에는 다음 작은 확장만 허용한다.

- `LegEvaluator(..., *, retain_startup_jacobians=False)`를 추가하고 true일 때 기존에 계산한 startup q Jacobian[99,7,P], LTM Jacobian[99,1,P]를 cache entry에 보존한다. 새 신경 평가/미분 호출은 없다. 메모리 회계는 기존 `array_bytes(evaluator.cache)`에 포함된다.
- `execute_original(..., evaluator_factory=LegEvaluator)`를 keyword-only 인자로 추가하고 원래 evaluator 생성 위치에서만 사용한다. 새 runner는 `functools.partial(LegEvaluator, retain_startup_jacobians=True)`와 conic leg fitter를 명시적으로 주입한다. 기존 호출과 E274의 기본 동작은 그대로다.

현재 root의 `mean_square` 및 `abort_run` 수정은 유지한다. E269 engine, E274 runner/test, 기존 E270 test, 계획/원본 파일은 수정하지 않는다. root FIRST → 같은 Sol 첫 수정 → root SECOND → root 잔여 수정 순서를 따른다. 원본/CUDA/BANC/body는 에이전트 0회, 생성 CPU/Ruff만 허용한다. 재위임하지 않는다.

## 방향 subproblem

현재 strict feasible c, training gradient g, startup 오차 q_e/l_e, Jacobian J_q/J_l에서 f64 sparse Clarabel 문제를 만든다. 입력 Jacobian은 각 error와 동일한 C-order로 펼친다. 변수 d는 P개의 계수 변화량이다. 목적은 `g/max(abs(g)) dot d + .5e-3 * ||d||²`다. 작은 양의 proximal 항은 제약의 null 방향을 불필요하게 움직이지 않도록 방향 생성에만 쓰며, 실제 training objective와 최종 선택은 기존 그대로다. g가 정확히 0이면 solver 없이 종료한다.

기존 trust radius 초기 .05/최대 .5/최소1e-6과 coefficient [0,8], triple 합 interior[.125*(1+1e-6),8*(1-1e-6)]를 유지한다. 각 d는 box와 trust bounds를 동시에 만족한다.

각 startup 행마다 q의 peak limit `.04*(1-1e-4)`, LTM `.025*(1-1e-4)`에 대해 다음 선형 제약을 모두 사용한다. 일부 argmax 행만 선택하지 않는다.

`J d <= .8*(peak - error)` 및 `-J d <= .8*(peak + error)`.

global RMS는 다른 다리의 fixed SSE를 그대로 빼서 남은 해당 다리의 허용 SSE R²를 계산한다. q의 global count는 99×42, LTM은99×6이며 RMS .018/.009와 동일 selector margin1e-4를 적용한다. 각 norm에 대해 current SSE를 s라 하면 `r² = s + .8*(R²-s)`로 현재 여유의20%를 남긴다. 부호/유한성/현재 feasibility를 확인하고 다음 SOC를 사용한다.

`||error + J d||₂ <= sqrt(r²)`.

Clarabel의 `A d + slack = b` 표현에서 SOC는 첫 행 A=0, b=radius; 나머지 A=-J, b=error이다. 두 개 SOC를 사용한다. peak 및 coefficient/triple constraints는 nonnegative cone으로 묶는다. `P=1e-3*I`, 선형항은 정규화 g다. 따라서 원래 nonlinear 신경 모델을 SOCP로 대체하지 않으며, 방향의 예측에만 사용한다.

설정은 설치된 Clarabel 버전을 확인해 freeze에 기록한다. `verbose=False`, `max_iter=100`, `tol_gap_abs=tol_gap_rel=tol_feas=1e-8`, `time_limit=min(2초, 남은 leg/shared/original)`이다. 가능한 라이브러리 설정으로 CPU thread1을 고정하고 실제 설정을 기록한다. status `Solved`의 유한한 해만 사용하며 독립 coefficient/linear/SOC 최소 residual >=-1e-8을 확인한다. 실패/시간 초과는 해당 다리의 bounded 종료로 보존하고 자동 재시도/다른 solver fallback은 없다. solver constructor/solve 요청·완료·실패·iterations·wall을 구분한다.

## 실제 평가와 자원

다리 순서 LF/LM/LH/RF/RM/RH, outer 최대24, 고유 실제 평가80/다리, 다리30초/공유fit180초/원본290초를 유지한다. 각 방향의 alpha1,.5,.25,.125,.0625,.03125를 실제 기존 evaluator로 검사한다. 계수 interior, 모든 기존 실제 nonlinear startup 제약>=0, train MSE 비악화, 음의 방향 미분 및 Armijo1e-4를 통과해야 다음 iterate로 채택한다. 원래 radius 조절과 완료된 모든 strict-feasible 후보 중 최소 objective/첫 동률 선택, 미처리 다리 neutral, 단회 f32 rounding/최종 예측도 그대로다.

각 방향 생성 시 coefficient, gradient scale/normalized gradient, q_e/l_e와 전체 J_q/J_l, norm radii, bounds, solver x/s/z와 solver info/residual을 저장해 root가 별도 식으로 확인할 수 있게 한다. 배열은 `.npz`에 저장하고 거대한 행렬을 JSON에 넣지 않는다. static matrix 재구성에 충분한 값만 남겨 중복 저장을 피한다. 완료/미완료 평가·예산/저장 실패의 원래 예외와 `abort_run`을 상위까지 보존한다. E274의 검증된 helpers를 재사용해도 되지만 LP 문제와 비용을 conic solver라고 잘못 이름 붙이지 않는다. `solver_calls_kind`는 실제 Clarabel SOCP 호출이다.

기존 E270 final artifacts/raw/candidate/selection/checks/result 형식을 유지해 저장 자료 audit를 재사용한다. E275 별도 schema/CONTRACT/source-review를 사용한다. source pin은 E269 engine/test, E270 runner/test, E274 runner/test, E275 runner/test 총8개다. 기존 원본 E270 freeze의 입력/맥락/목표/native gate는 유지하고 E273 exactness/speed gate도 요구한다. historical live source pins는 다시 강제하지 않는다. 글로벌 monkeypatch나 원본 승인 자동 추론은 금지한다.

생성 CPU 검사는 실제 Clarabel의 작은 문제, SOC 부호, C/F memory-order 동일성, 활성 peak 시간점 교체, SSE 1차 모델이 놓치는 제곱 항, 실제 nonlinear backtracking, 완료/미완료/저장 실패, explicit factory 주입을 다룬다. 실제 작은 solver 비용을 보고하고 큰 원본 크기 성능 시험은 root 별도 예약까지 실행하지 않는다.

원본 예약은 현재0이다. root 생성 검증에서 900변수/693q/99LTM의 고정 생성 Jacobian 문제 한 개(학습 아님)로 solver1회/최대2초 및 메모리 한도를 확인한 뒤 원본 여부를 결정한다. 통과한 구현만 동일 원본 입력에서 단회 학습한다. 결과의 고정 conditional gates가 모두 통과한 동일 후보만 E272 실제 BANC/body로 갈 수 있다. 새 이름으로 비용·실패·F1–F8 미달을 초기화하지 않는다.
