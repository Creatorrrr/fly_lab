# E170 Stage B V2 활동 이력 양수 전달 적합

2026-09-23 부모 설계. Stage A helper와 기존 E169의 실패, 별도 저장 affine 인증, 전체 M3 F1–F8 및 미사용 F6 초기상태를 보존한다. 실제 구현은 gpt-6-sol/high, 부모 FIRST 리뷰, 필요시 같은 Sol FIRST 수정, 부모 SECOND 리뷰와 잔여 수정 순서다.

## 가설과 판정 범위

H170: E150 저장 BANC 운동 신호에 현재 활동과 과거 활동을 함께 사용하는 양수 전달을 적합하면, 원래 물리 보호와 startup 한도 아래 E85 조건부 명령 RMS를 20% 줄일 수 있는가. 반대 설명은 이 유계 이력 상태를 추가해도 표현력이 부족하다는 것이다. E169 static alpha 전체 재적합 및 E168 고정 alpha 감쇠와 다른 가설이다. 최종 생물학적 기전, 실제 폐루프 보행 또는 F1–F8 성공을 이 저장자료 시험으로 주장하지 않는다.

Stage A의 dt=.005/tau=.07 recurrence R와 F=2(1-R), 실제 MN graph-column 소유권 및 3개 기존 EMA bank를 유지한다. alpha+F*beta의 alpha와 beta를 공동 선택한다. kappa, 시간상수, 위상, 학습 신경 회로 또는 대상 mask를 탐색하지 않는다. alpha=w0,beta0는 physical baseline이며 실제 decode endpoint1..2000의 원래 명령을 정확히 재현해야 한다.

같은 E164 입력 로더와 E168 원형 capture/strict validate, E164 prepare_saved를 사용한다. 원본의 고정19근육/77가변/330가변MN/391전체MN 및 다리별 소유권을 freeze로 검사한다. 원래 실제 신경/물리 trace의 2001행, target100..1923의1824행,1823velocity, startup1..99를 유지한다. E85 및 E103 참조는 오프라인 목적에만 사용한다. Stage A history_state를 한 번 실행(2000 update)하고 검증된 E168 resource와 직접 대조한다. 저장 resource, rate, kappa, bank의 원형/해시/shape/dtype 및 tick 일치를 먼저 검사하고 모든 결과를 assertion 전에 raw로 보존한다.

## 계산 순서와 독립 검산

E170-B0 진단에서 실제 tick helper는 baseline 기록과 bit exact였다. 단순 시간축 벡터화는 velocity gate에 미달했다. E170-B1에서 각 low*bank 및 (high-low)*hinge 곱을 C-contiguous(T,ownedMN,3)로 만든 뒤 (T,-1)의 마지막 축 sum을 사용하고 tick별 GEMV를 수행하자 사전 고정 세 벡터 모두 activation/command/velocity가 bit exact였다. 진단 raw는 보존하며 기준을 바꾸지 않는다.

새 candidate 함수는 독립적인 벡터화 식으로 activation과 tick별 command를 계산한다. native oracle은 Stage A evaluate_history를 별도로 호출한다. 각자의 command에서 같은 pair_start/end로 velocity를 계산하여 activation/command/velocity 각각 absolute 1e-12를 유지한다. recorded+delta/GEMM으로 native 비교를 하지 않는다. baseline alpha=w0,beta0는 recorded activation 전체, 실제 decode endpoint1..2000의 command 및 기존 baseline metrics와 정확히 같아야 한다. canonical command tick0와 기록 tick0의 차이는 별도 raw로 보존하고 어느 쪽도 덮어쓰지 않는다. 후보는 clipping 또는 후처리하지 않는다.

solver는 low/high branch basis와 F*branch를 독립 구성한 실수 affine 모델이다. 각 다리의 direct matrix/offset과 QR norm을 baseline, zero 및 사전 cosine 벡터에서 검산하고 반환 후보에서도 18개 norm을 직접 native 명령/velocity/startup 오차와 대조한다. 기존 QR absolute1e-12+relative1e-10 기준을 사용하며 pointwise matrix residual도 raw에 보존한다. 이는 native 순서 일치와 별개인 선형대수 검산이다. 큰 solver 행렬을 실제 tick oracle인 것처럼 취급하지 않는다. 참조의 시간차분과 q/velocity 구간을 각자 정확히 사용한다. norm gate가 실패하면 technical invalidity이며 조용히 원본 적합을 다시 하지 않는다. affine certificate의 배제 범위는 저장된 유한 정밀도 affine 가족으로 제한한다.

## 최적화 계약

pair990의 순서는 원래 graph-column 오름차순330 × filter0/1/2이며 각 pair는 low/high 순서다. alpha1980,beta1980,ell990,u990,t1로 총5941변수다. alpha/beta의 실제 물리 box는[0,2], endpoint alpha+beta<=2이고 모든 F∈[0,1]에서 보호한다. pair마다 ell<=alpha_low,alpha_high, u>=alpha_low+beta_low,alpha_high+beta_high다. MN의 ell합>=1e-6/n_owned, 근육의 ell합>=.5, u합<=2다. 실제 gate는 auxiliary 대신 직접 min(alpha) 및 max(alpha+beta)를 합산한다. box/support/fixed는 exact이며 MN/muscle budget만1e-12를 사용한다. ell,u는[0,2],t는[-1,1]이다.

새 solver만 alpha/beta 하한을1e-9로 둔다. 이는 E169에서 관찰된 미세 음수 반환값을 피하기 위한 사전 수치 여유이며 물리 허용치를 완화하지 않는다. physical baseline beta0가 inner cone에 포함된다고 주장하지 않는다. inner witness는 alpha=(1-d)*w0+d*c, d=1e-5, c의 모든 owned branch=1.25/(3*n_owned), beta의 가변 branch=1e-9, ell=min(alpha),u=max(alpha+beta),t=-.251이다. 이 witness의 실제 전체 slack을 저장/검산한 뒤에만 solve한다. 실패하면 fit0에서 멈춘다.

행0:11882는 -I,+I boxes;11882:15842는3960 epigraph;15842:16172는330MN;16172:16249는77lower;16249:16326는77upper;16326:20484와20484:24642는startup ± max다. first cone24642 및 같은15SOC/18QR를 사용한다. objective P0,q 마지막 -1로 margin을 최대화한다. 전체 command RMS .8, 전체/각 다리 velocity와 각 다리 command 비악화, startup qRMS .018/max .04 및 고정 LTM RMS .009/max .025를 유지한다.

strict inner dual과 true-physical relaxed dual을 둘 다 저장한다. 후자는 lower-box RHS b[0:3960]에1e-9를 더하고 b[15842:16326]에1e-12를 더하며 stationarity support의 lower[0:3960]를0으로 되돌린다. 그 외 A/q/box/SOC/startup/epigraph는 유지한다. cone projection pad1e-12, bound pad1e-8을 사용한다. true-physical upper<0이어도 저장 affine 가족 배제만 판정하며 native의 실제 폐루프/생물학 전체를 반증하지 않는다. inner dual만으로 넓은 physical 가족을 배제하지 않는다.

## 구현, 저장과 자원

새 source `tools/fit_c_banc_m3_history_recruitment.py`와 시험 `tests/test_c_banc_history_recruitment_fit.py`만 Sol이 소유한다. 기존 E169 run의 단일 solve, 설정 검증, snapshot, 원형 선저장, 첫 I/O 실패 시 완료된 solver/candidate의 caller-owned fallback 보존, primary/secondary error 구분을 재사용한다. E169/Stage A/기존 시험을 수정하지 않는다. Stage B freeze와 최종 source gate는 부모가 소유한다. 원본 실행은 최종 gate 전까지0이다. CPU1, Clarabel0.11.1/qdldl, max_iter100, 기존 tolerance1e-10, solver1회/fit30초, 내부95초/외부110초/supervisor120초, raw320MiB/free1GiB를 고정한다. 자동 재시도/추가 fit/파라미터 탐색은0이다. factorized basis와18QR, problem sparse arrays를 저장하고 큰 global dense affine 복제는 만들지 않는다.

이전 보수 원본 누적2961.394460999989초와 B0/B1 저장진단11.527756599942223초를 보존한다. 합계2972.922217599931초를 올림한2973+120=3093초를 이번 prospective 누적 bound로 배정한다(이전 bound3023에서70초 증가). 생성 full-shape 용량 stress는256264022bytes였다. 새 기록이나 이름으로 이전 비용을 초기화하지 않는다. solver/feature/recurrence/검증 진단 비용은 구분한다. 원본 실행에는 body/store/새 EMA/신경/물리/CUDA0이다. GUI/서버/M2/F6도 그대로 유지한다.

생성 시험은 실제 noncontiguous graph-column 소유권, 양수/음수 계수 raw, 다양한 F 및 beta0, robust 물리 보호, alpha+beta epigraph 부호, 정확한 전체 행 offset, true dual 하한 복원, inner/physical baseline 구분, 잘못된 입력 원형 저장, single solve 및 첫 publication 실패에서 solver/candidate 보존을 다룬다. 부모는 별도 scalar 수식으로 matrix/QR/cone와 원시 계수를 검산한다.

validity와 hypothesis/proposal/adoption/whole goal을 따로 기록한다. native/입력/행렬/자원 실패는 invalid_not_tested, 같은 반환 후보가 물리/직접 성능 gate를 모두 통과하면 조건부 후보다. 통과 후보는 다음 단계의 전체 BANC 실제 몸체에서 F1–F8을 검증해야 한다. 가설 배제 또는 inconclusive면 기존 가설의 제한을 근거로 다음 in-scope 구조를 설계하며 목표를 완료 처리하지 않는다.


## V2: 최초 decode 이전 기록의 비교 범위 정정

V1 설계/동결/초안 결과를 보존한다. 원본 fit 전 B0 저장 raw만 다시 읽어 baseline activation 전체 오차0, command1..2000 오차0, command tick0 오차0.32668292642150043을 확인했다. tick0는 실제 최초 decode 전 초기 몸체 명령이며 Stage A q0는 동일 activation을 새 decoder로 계산한 가상 출력이다. 이 두 값을 동일하다고 요구한 V1 설계는 시간 정렬 오류였다. V2에서는 원래 E164/E169 native 비교와 같이 실제 decode endpoint1..2000을 비교한다. target100..1923/velocity1823쌍/startup1..99/모든 허용치 및 물리 조건은 그대로다. 초기 기록이나 canonical q0를 교체하지 않으며 baseline raw에 q0 차이와 endpoint 차이를 모두 남긴다. fixture도 이 경계를 다루고 원래 activation은 모든 tick에서 동일해야 한다. 이는 보행 통과 기준 완화나 기존 실패 재분류가 아니다. V1 원본 fit은0이다.
