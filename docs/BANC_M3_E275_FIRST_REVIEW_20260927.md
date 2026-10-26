# E275 root FIRST review — 2026-09-27

검토 대상은 실제 gpt-6-sol/xhigh FIRST다. 원본 학습은 아직0이다. source8개를 `verification/banc-natural-gait-m3-20260919/e275-FIRST-reviewed-sources/`에 보존했고 root 생성 계약 SHA는 `d31664c177717a87dea394869ce087484d498d24a281dae4aca10e21651d2088`이다.

## 검증된 부분

기존 E270/E274를 포함한 생성 CPU90개와 Ruff가 통과했다. root의 고정 생성900변수/4778제약/547110 nonzero 문제는 Clarabel1회/17iter로 Solved, constructor .083027899964초, solve1.555908999988초, 전체 wrapper1.642280300031초로 사전2초를 통과했다. 별도 계산한 coefficient/모든 peak/SOC 최소 residual은6.184497358674435e-13, objective 방향 미분은-14.418574655510545였다. 같은 수치 기준과 실제 nonlinear evaluator를 유지한다. 이 결과는 원본 신경 학습·실제 보행의 통과가 아니다.

root 생성 supervisor5.989002400020초, peakRSS158605312bytes, 원본/CUDA/BANC/body0. 작은 실제 Clarabel4constructor/4solve/39iter, 기존 HiGHS46호출/0iter, SLSQP1호출/2iter/3평가를 별도 기록했다. 에이전트가 실행한 앞선 작은 생성 비용은 별도로 보존한다.

## 수정 필요: 원본 전체 시간 만료 예외의 보존

`fit_conic_leg`의 final exception handler는 `BudgetExceeded("original")`까지 선택값으로 반환한다. 그러면 `run_fit_pass`의 다음 `start_leg()` 또는 최종 prediction 전 deadline 검사에서 **다른** BudgetExceeded가 새로 발생해, 최초 예외의 객체·notes·관련 증거 연결을 잃는다. root의 생성 시계(전체270→291초)/fake evaluator/fake solver1회 재현에서 완료 평가2개와 선택eval1은 보존되었으나 최초 notes가 상위에서 사라졌다. actual solver/신경0이며 재현은 `parent-e275-first-probe.py/json`이다.

같은 Sol의 첫 수정은 다음 두 파일로 한정한다.

- `tools/fit_c_banc_m3_conic_source.py`: history/selection 저장을 끝낸 뒤 원본 전체 만료이면 최초 예외를 그대로 raise한다. leg/shared_fit/unique 제한은 기존처럼 완료한 strict-feasible 선택값을 반환하고, 저장 실패의 abort_run 및 원래 예외를 유지한다.
- `tests/test_c_banc_conic_source_fit.py`: 실제 상위 `run_fit_pass`를 포함해 original 만료 시 같은 예외 객체와 notes가 상위까지 도달하고 추가 다리/solver가 실행되지 않는 생성 회귀를 추가한다. original 만료의 완료/미완료 평가를 구분하고 저장 실패 시에도 같은 원래 예외를 보존한다.

수학·solver·900변수 benchmark·입력/허용치/최종 gate/base/engine은 바꾸지 않는다. 관련 생성 CPU만 실행하며 큰 생성 benchmark 반복, 원본/CUDA/body는0이다. root가 SECOND 리뷰하고 잔여 수정이 있다면 root가 맡는다. 현재 원본 누적7494.618436499561/7805초와 새 원본 예약0을 유지한다.
