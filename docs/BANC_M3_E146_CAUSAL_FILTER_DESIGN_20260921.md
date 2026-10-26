# E146 — 신경 출력에 대한 인과적 근육 활성 시간 응답

2026-09-21. 부모 설계, 실제 Sol/high FIRST 구현, 부모 FIRST review, 필요한 경우 같은 Sol FIRST correction, 부모 SECOND review, 이후 부모 잔여 수정 순서다. 기존 finite M3의 F1–F8 및 실패 기록은 그대로다. 이 문서와 freeze를 원본 계산 전에 작성한다.

E145의 10ms 단일 필터와 정적 정규화 pool은 저장 신호에서 정한 복합 제안 조건을 만족하지 못했다. 이를 반복하지 않고 각 원래 운동뉴런 출력에 빠른/기존/느린 인과적 활성 응답을 허용한다. H146은 이 제한된 시간 응답 확장이 동일한 초기 안전 조건과 후기 command RMS20% 감소, velocity/다리별 비악화를 동시에 만족하는 조건부 제안을 만들 수 있다는 것이다. 경쟁 설명은 시간 응답을 바꾸어도 입력 신호와 관절 전달 구조 자체가 제한된다는 것이다. 이 제안 조건은 자연 보행의 필요조건이 아니며 실패로 모든 신경 보행의 불가능성을 주장하지 않는다.

## 고정 신호와 모델

E143 raw의 recorded_motor_rate[2000,391]을 한 번 읽는다. dt=.005s, 고정 tau=[.005,.01,.04]s, 각 branch의 초기 상태0, b_next=b+(-expm1(-dt/tau))*(rate-b)를 2000회 수행한다. 원본 391 MN, 96 근육 membership, moment/rest, 관절 전달 계수 .02, 신경 파라미터, 표적/위상/시간창은 고정한다. middle10ms branch는 E143 saved filtered_motor_rate와 array_equal이어야 한다. branch bank 형태는 [2000,391,3], 평탄화 열은 motor-major 3*i+k이다. 필터에는 미래 신호, 교사, 시계 기반 위상 또는 외부 CPG가 없다.

변수 W[96,391,3]은 원래 MN이 속한 근육에만 존재한다. 활성 a[t,m]=sum(i,k) W[m,i,k]*b[t,i,k], q=rest+.02*moment*a이다. LTM6 및 원래 singleton 행은 원래 pool을 middle branch에 두고 다른 branch0으로 고정한다. 따라서 LTM 및 singleton 출력은 그대로다. 그 외 근육은 합1의 비음수 branch 가중치를 적합한다. 원래 MN별 branch 합은 최소1e-6/해당 근육 MN수이고 모든 원래 MN을 보존한다.

수치 solver에는 각 가변 branch에 1e-6/원래 MN수의 양의 내부 guard를 둔다. 이는 최종 비음수 및 MN별 floor보다 엄격한 사전 제약이다. 후보는 사후 clipping/정규화하지 않는다. 원래 정적 pool은 guard 없는 표현에 정확히 포함되며, guarded family에는 임의로 정확히 포함된다고 주장하지 않는다. 검사용 lifted baseline은 원래 각 MN pool 가중치의 1e-6씩을 fast/slow로 옮기고 middle은 (1-2e-6)로 둔다. 원래 pool의 균등성을 검증하고, 이 벡터의 t=-.251 feasibility와 원래 baseline 대비 오차를 기록한다. 예상 조건이 다르면 첫 실패를 남기고 중단한다. 음의 dual 상한은 이 guarded family에 대해서만 해석한다.

각 가변 근육에서 원래 마지막 MN의 middle10ms를 anchor로 정하고 나머지 supported branch 열은 오름차순으로 둔다. anchor=1-sum(free)라는 사전 affine 좌표계를 사용한다. fixed 행은 원래 값을 직접 유지한다. canonical baseline은 원래10ms branch만 사용하는 출력이며 E144/E145의 값과 같다. candidate/fixed 출력은 동일한 저장10ms 배열을 사용하고 tendon은 .2*tanh(a_LTM/10)이다. 실제 합산 방식에서 fixed 출력이 bit-exact 유지되는지 확인한다.

이는 공학적 활성 응답 가설이다. 서로 다른 활성/비활성 시정수의 비선형 근육 모델, 힘-길이-속도 관계, 또는 실측 초파리 근육 힘을 구현했다는 뜻이 아니다. MuJoCo 공식 stateful actuator의 exact filter 식은 수치 모델 근거이며, tau의 생물학적 타당성을 증명하지 않는다: https://mujoco.readthedocs.io/en/stable/computation/index.html#stateful-actuators .

## 단일 적합과 직접 판정

기존 E145의 QR 압축, SOCP 및 dual 진단을 재사용하되 basis/anchor/fixed 행을 명시적으로 구성한다. 원래1827 endpoint, 실제 인접1826 velocity pair와 초기99를 사용한다. command/velocity/startup 다리별 affine 식에서 thin QR의 전체 R을 보존하며 열/특이값 절삭은 없다. 독립 직접 norm과 압축 norm을 canonical/lifted baseline 및 생성자료에서 검사한다.

목적 max t, -1<=t<=1, 같은15 SOC와 초기99*42의 +/-조건을 사용한다. 전체 command norm 한도는 baseline RMS*.8, 전체 velocity는 baseline RMS, 여섯 다리 각 command/velocity는 해당 baseline RMS, startup 전체 RMS .018rad 및 각 샘플 절댓값 .04rad이다. 모든 물리 norm 제약을 해당 한도로 나누어 <=1-t로 둔다. tendon 초기 RMS<=.009/max<=.025는 고정 출력에서 직접 확인한다.

Clarabel0.11.1 한 번, CPU1/max_threads1/qdldl, max_iter100/time_limit30s, tol_gap_abs/rel/feas/infeas_abs/rel1e-10, 나머지 defaults. raw에는 실제 settings, 호출 진입 수, callback history, x/s/z/status/잔차를 보존한다. pre-solve에는 필터 bank, affine anchor, A/b/q/cones/QR을 저장한다. 결과 raw는 후보판정 및 자원 판정 전에 보존한다. constructor/callback/solve/저장/시간초과 실패 시 첫 예외와 이미 생성된 raw를 유지한다. 추가 fit/재시도0.

제안은 반환 벡터 하나의 실제 전체 q/velocity/startup/tendon 오차, 정확한 fixed 행/출력, 행합 atol1e-12, 각 branch>=0, 원래 MN별 합 floor 허용1e-12, 같은 해부학적 support를 모두 직접 만족해야 한다. solver status 또는 t만으로 통과시키지 않는다. runtime1e-15 행합 호환성은 별도 기록한다. raw dual 및 cone 내부 보정 dual, stationarity/box 보정에 따른 수치 t 상한은 E145와 같은 방법과 pad를 사용한다. 이는 부동소수점 진단이며 정확한 유리수 증명이 아니다.

한정된 세 branch 가설은 valid run에서 실제 제안이 있으면 supported, 충분한 음의 수치 상한이면 guarded family 내 refuted, 그 외는 inconclusive로 판정한다. 조건부 제안은 채택이나 자연 보행 완료가 아니다. 통과하면 별도 prospective native10초/저장 future/차단 검증을 설계하며 같은 후보 F1–F8을 요구한다. 실패하면 원래 결과를 보존하고 시간 응답 또는 전달 구조의 다음 결정을 내린다. F6/활성 UI/M2는 읽거나 실행 변경하지 않는다.

## 자원과 구현 범위

원본1프로세스, 재시도0, CPU1/GPU0, solver1, 신경/CUDA/물리/LP0, 저장 신호의 bank2000 vector updates(391*3*2000 scalar updates). fit30/internal60/external65/supervisor70초, raw128MiB/시작여유1GiB. 기존 보수 누적2187.4898345999536을 보존하고 prior2188+70=2258로 배정하여 기존2271초 한도 안이다. 생성자료 테스트·부모 검산 비용은 별도 실제 측정한다.

Sol은 tools/fit_c_banc_m3_causal_filter.py와 tests/test_c_banc_causal_filter.py만 구현한다. 기존 E145/E144 코드와 모든 봉인 원본은 수정하지 않는다. 부모가 설계/freeze/독립 검산/감독/진행 기록을 소유한다. high는 저장자료 필터·희소 수치식·단일 convex solver 통합에 해당한다. 이후 native neural/CUDA/checkpoint 구현은 xhigh로 별도 맡긴다. 원본 파일을 읽어 fit/필터하는 실행은 부모 최종 source gate 이후 한 번만 허용한다. FIRST 구현과 FIRST correction 검증은 생성자료만 사용한다.
