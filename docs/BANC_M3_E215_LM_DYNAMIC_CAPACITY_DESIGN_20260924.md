# E215: LM의 시간상수를 보존한 동적 전달 용량

E214의 고정 9차원 입력 반응은 전역 startup 조건에서도 PrimalInfeasible였다. E212는 각 시점 activation을 독립적으로 골랐으므로 막/근육 필터의 도달 가능성과 반응 함수의 표현력을 분리하지 못했다. E214 dual에서 LM 명령 항의 기여가 가장 컸다는 것은 다음 검사 대상을 고르는 근거이며 단독 원인 증명이 아니다. 이번 진단은 LM 하나의 **필요조건을 완화한 문제**를 한 번 푼다. 원래 시간상수, cap, gate, 근육 소유권, canonical 10ms 전달을 유지하고 시간마다 반응의 크기를 자유롭게 선택한다. 최종 신경 후보나 외부 제어기로 채택하지 않는다.

## 고정 입력과 정확한 필터 소거

E201 motor-parameters, E203 free343/fixed48 mask, E213 B/H/affine, E197 target, E150 trace를 사용한다. 자유 LM MN54개/근육14개를 근육 오름차순으로 선택한다. 각 근육의 MN은 동일한 tau를 가져야 한다. 실제 예외인 MN92104(col220, muscle19)는 singleton이고 tau=.014046886935830116s이며 나머지 LM53개는 .019999999552965164s다. 이를 평균내지 않는다. 자유 initial_rate와 B[0]은 정확히0이어야 한다. 고정 LTM의 기록 입력 조건부 효과와 초기 activation은 q_intercept에 남긴다.

dt=.00025, 20 neural ticks/control, control=.005다. 각 근육 g의 막 동차 decay는 F_g=(1-dt/tau+(dt/tau)^2/2-(dt/tau)^3/6+(dt/tau)^4/24)^20. 신경 RK4의 양의 입력 가중치 및 stage gate>=0에 따라 한 control의 임의 bounded response는 endpoint에서 하나의 forcing으로 합칠 수 있다. E213 mass[t,m]=sum_k B[t,m,k]에서 J[t,m]=cap_m*(mass[t,m]-F_g*mass[t-1,m]). G_g=sum_own pool[g,m]*cap_m, S[t,g]=sum_own pool[g,m]*J[t,m]/G_g다. G>0, own pool>=0와 다른 근육 pool0을 확인한다. 모든 raw J/S를 보존한다. 부동소수 반올림의 작은 음수를 숨기지 않으며, 원래 물리 범위를 낙관적으로 넓힌 forcing 범위 [-2e-12,S+2e-12]를 사전 사용한다. S<-2e-12이면 중단한다. 이 epsilon은 정규화된 pooled-rate innovation 단위다.

alpha=-expm1(-.005/.01), d=1-alpha. 추가 activation/G를 x[t,g]라 하고 x[0,g]=x[-1,g]=0으로 정의한다. 막 rate와 근육 EMA를 소거하면

`T x = x[t]-(d+F_g)*x[t-1]+d*F_g*x[t-2] = alpha*forcing[t,g]`.

변수는 frame1..1923 × LM14근육의 time-major x26922개다. 0<=x<=1과 위 forcing 양쪽 bound를 모두 선형 부등식으로 둔다. 새 초기 rate/activation을 자유 변수로 넣지 않는다. startup99행과 main1824행을 이어붙이며 mainvelocity는 frame100..1923의 기존1823쌍만 쓴다. q=q_intercept[frame,LM7]+L*x, L=.02*moment[LM7,own14]*G. 시간축·목표·위상은 바꾸지 않는다. LM 이외 다리의 행이나 고정48 rate를 다시 맞추지 않는다.

## 한 번의 sparse convex 문제

목적은 LM main 명령 잔차의 원래 LM baseline RMS로 정규화한 평균제곱이다. ridge나 새 가중항은 없다. startup pointwise |error|<=.04, main LM command RMS<=원래 LM baseline, LM velocity RMS<=원래 LM baseline을 제약한다. PAD=1로 더 좁히지 않는다. 전역 startup RMS .018, 전역20%명령 개선, 다른 다리, tendon 조건은 이 단일 LM 필요조건 문제에 넣지 않는다. 특히 LM startup RMS .018 같은 더 강한 대체조건을 추가하지 않는다. main/startup/velocity SOC는 원행 그대로 sparse로 만들고 dense Gram 압축/SVD는 하지 않는다. P는 main sparse map의 Gram 상삼각, linear와 목적 상수를 저장한다. cone 순서는 하나의 Nonnegative(동적 하한/상한, x하한/상한, startup 양쪽), command SOC, velocity SOC다.

Clarabel max_threads1/max_iter150/time_limit90s/tol_gap_abs=tol_gap_rel=tol_feas=1e-8, 원본1 solve/retry0이다. pre-solve CSC(P,A), b/linear/constant, 각 물리 map/offset, F/G/S/J/mass/소유권/표본/한도, 모든 solver settings를 원본 호출 전에 저장한다. 반환 x/s/z/status/objective/residual/iterations/time은 판정 전에 저장한다. 시간이나 저장 실패를 자동 재시도하지 않는다. 생성 데이터 검사는 수식/행 조립/모의 오류를 검사하고 부모 리뷰 뒤 작은 해석 가능한 실제 Clarabel 한 번만 허용한다(별도 외부30초). 기존 E211 실제 생성 결과는 보조 증거일 뿐 새 행 조립의 검사를 대신하지 않는다.

Solved이면 raw x에서 forcing와 pooled-rate를 복원하고, 두 단계 필터를 다시 순방향 계산해 q를 비교한다. clip/활성화 보정은 하지 않는다. raw feasibility를 normalized box/dynamic absolute1e-8, startup .04+1e-8, command/velocity RMS limit+1e-8 기준으로 수치 보고하고 오차와 허용치를 명시한다. LM startup RMS 및 명령20% proxy는 참고로만 계산한다. 이는 자유 forcing의 조건부 수치 witness이지 BANC 반응 함수/센서 피드백/자연 보행이 아니다.

PrimalInfeasible이면 cone(z), A^Tz, b^Tz와 각 cone 기여를 독립 검산한다. x box0..1을 이용한 수치 분리 조건은 b^Tz < sum(min((A^Tz)_i,0))이며 cone오차와 잔차를 함께 보고한다. 검산이 성립하면 고정 입력과 canonical 전달에서 원래 필요조건도 이 범위로는 만족하기 어렵다는 근거다. 이것을 자연 보행 전체의 불가능성, 생물학적 시간상수의 오류, 원래 E150 전달의 한계로 확대하지 않는다. AlmostSolved/시간초과/검산 실패는 unresolved다. 한 LM feasible만으로 여섯 다리 전체 feasible라고 하지 않는다.

## 구현과 자원

부모 설계 → 기존 실제 gpt-6-sol/high 구현 → 부모 FIRST → 필요시 같은 Sol 첫 수정 → 부모 SECOND/잔여 수정. 소유 파일은 tools/analyze_c_banc_m3_dynamic_capacity.py와 tests/test_c_banc_dynamic_capacity.py다. 동역학 소거·sparse SOC·원시 저장·회계가 결합되어 high를 쓴다. 부모는 별도 독립 행/필터/dual 검산을 준비한다.

원본 CPU1/GPU0/신경0/물리0/process1/solver1, 내부150s/solver90s soft/외부180s/감독200s, RSS8GiB/출력256MiB/free4GiB. solver constructor 전후에 내부시간에서90초가 남는지 확인한다. 원본 누적5684.636283099706/배정6004초, 잔여319.3637169002941초에서200초 예약, 배정 증액0이다. 준비/생성/부모 검산은 별도 계상한다. 미측정 설계/agent 비용을0으로 기록하지 않는다. 원본 새 재시도나 자동 sweep은 없다.

기존 E150 최선, M2, E214 음성 결과, UI와 F1–F8를 보존한다. F6 미노출, 실제 전체 BANC/신체 검증 미진행, 목표 active/미완료다. 다음 방향은 이 한정 결과를 검토한 뒤 별도 설계한다.
