# E276 — 동일 방향 문제의 단위 조정 진단

2026-09-27 root 설계. E275는 원본 단회112.421847100초를 사용한 **유효 reject**다. 학습 RMS .150072253753/중립 .161324840380, 개발 .092535497316/.097982864968로 개선됐지만 train .8 및 LM/RM 개별 development1.05 gate를 통과하지 못했다. startup은 통과했고 실제body0이다. F1–F8, 원래 gate와 과거 비용을 유지한다.

관찰: LM/RF/RH는 각각 실제 평가41/5/7회 뒤 Clarabel `AlmostSolved`에서 종료했다. 이 세 마지막 해의 원래 단위 독립 residual은 각각1.263746052249104e-15,6.984647235586117e-15,1.0322168467191695e-13으로 feasible였으나 solver r_prim은4.837061744073e-8/5.336136311585e-8/3.158164766869e-8이었다. 고정 `Solved`만 허용하는 E275 계약에 따라 거부한 판정을 보존한다. 원본 문제의 높은 민감도와 계수/명령 단위 차이가 수치 수렴을 어렵게 한다는 가설을 검증하며, 그 원인을 확정하지 않는다.

## 범위와 소유권

새 후보 학습이나 `AlmostSolved` 채택이 아니다. 원래 feasible set과 optimum을 보존하는 **양의 변수/행 단위 변환**만 구현한다. 실제 gpt-6-sol/high가 `tools/banc_m3_equilibrated_conic.py`, `tests/test_c_banc_equilibrated_conic.py`만 구현한다. E275/E274/E270/engine/source/실험raw/계획은 수정하지 않는다. root FIRST→같은 Sol 첫 수정→root SECOND→root 잔여 수정 절차다. Sol은 작은 생성 CPU 문제만 실행하고 큰900변수·원본·CUDA·BANC·body는0회, 재위임0이다.

API는 `equilibrate_problem(original)`과 `solve_equilibrated(original, time_limit, *, solver_factory=clarabel.DefaultSolver, wall_clock=time.perf_counter)`다. 입력은 E275 `conic_problem` 결과이며 original을 변형하지 않는다. 상위 학습기에 연결하지 않는 독립 helper다.

## 고정 변환

원래 방향을 d, trust radius를 r>0라 하고 d=r*u로 둔다. 목적함수 전체를 양의 r로 나누므로 `P'=r*P`, `q'=q`이며 최소점은 동일하다.

원래 제약은 A*d+s=b. 먼저 A_r=r*A를 만든다. Nonnegative cone의 각 행 i는 `scale_i = 1/max(max_j(abs(A_r[i,j])), abs(b[i]), 1e-12)`로 정규화한다. 각 SOC는 **cone 전체에 한 개 동일한** 양의 scale을 적용한다. 그 scale은 같은 cone의 모든 A_r/b 원소 절댓값 최대와1e-12의 max의 역수다. SOC 내부 행마다 다른 scale을 적용하지 않는다.

즉 D는 양의 대각이며 SOC별 상수, `A'=D*A*r`, `b'=D*b`다. 원래 primal과 dual로 복원하는 식은 `d=r*u`, `s=D^-1*s'`, `z=r*D*z'`다. 물리 단위 목적값은 transformed objective의 r배다. 이 식과 `P*d+q+A.T*z`의 KKT 관계를 별도 검사한다. 원래 배열의 dtype/shape/finite/양수 radius와 cone row count, scale finite/positive를 검증한다. cone 종류/순서는 E275와 같고 scaling 최적화·탐색은 하지 않는다.

Clarabel0.11.1/max_threads1/max_iter100/abs,rel,feas1e-8/time_limit min(2초, 전달된 잔여 시간)을 유지한다. fallback/retry0, `Solved`와 유한한 해만 허용한다. E275 `conic_residuals(original,d)`로 원래 coefficient/peak/SOC minimum>=-1e-8을 재검사한다. 이 tolerance나 최종 실제 nonlinear/F1–F8 gate는 바꾸지 않는다. 원래 코드처럼 미세 box 잔여는 향후 실제 제안 경로에서 기존 clip 후 재평가하며 이 helper는 자동 clip하지 않는다.

반환은 E275 `solve_direction`과 같은 direction/s/z/valid/residuals/receipt에 transformed_x/transformed_s/transformed_z/variable_scale/row_scale을 추가한다. 모든 원래/변환된 raw 해와 설정, status, constructor/solve 요청·완료·실패/iterations/wall, 원래 단위 objective/KKT norm을 보존한다. 실패 예외에는 동일 primary를 유지하며 receipt와 가능한 partial solution을 첨부한다. 원본 신경 실행을 helper import나 검사에서 호출하지 않는다.

## 사전 검증과 원본 진단 경계

생성 작은 문제는 affine slack 및 primal/dual 복원, cone별 동일scale, SOC 부호, 원래 입력 불변성, 작거나 큰 Jacobian, strict `Solved` 판정과 불완전/solver/storage 오류를 검사한다. 원본 최적점이나 gait 성능은 생성 테스트로 주장하지 않는다.

root는 이전과 동일한 seed27520260927의900변수 생성 fixture 한 개에 새 solver1회/2초를 예약한다. 이전 원래 단위 문제의 saved objective와 비교하고 원래 residual, descent, wall2초를 검사한다. 이 시험도 wrapper의 수치/성능 검사이며 신경 학습0이다. 중복 반복이나 seed/크기/기준 변경은 없다.

생성/코드 검토를 통과하면 root가 별도로 **원본 저장 문제3개(LM의socp-011,RF의socp-001,RH의socp-002)**에서 새 helper를 각각1회만 실행하는 진단을 예약한다. 각2초/100iter, supervisor전체15초/2GiB, 원본 적분/CUDA/BANC/body0이며 원본 optimization 비용으로 전체wall을 누적한다. 이 세 점은 이미 노출된 개발 과정의 증거다. 현재 이 문서 작성 시 원본 예약0, 누적7607.040283599581/7805초, 잔여197.959716400419초다.

진단은3/3 `Solved`, 원래 minimum residual>=-1e-8, 원래 physical subproblem objective가 기존저장값보다1e-6 초과 악화하지 않음을 요구한다. 통과하더라도 기존 E275 결과를 재선택·승격하지 않는다. 다음 학습은 별도 가설/고정 계약으로 설계한다. 현재 FIRST에서 보인 작은 alpha 반복/다리별 개발 악화도 별도 과제로 남고, solver 수치 수렴만 해결하면 자연 보행이 된다고 해석하지 않는다.
