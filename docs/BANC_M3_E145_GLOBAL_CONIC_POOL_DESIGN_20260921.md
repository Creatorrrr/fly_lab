# E145 — 물리 조건 전체를 직접 다루는 정적 pool 볼록 적합

2026-09-21. 부모 설계, 실제 Sol/high 구현. E144는 정상 종료한 제한 탐색이지만 LF 초기 feasible점이 없어 제안이 없었다. 그 실패와 모든 원본·비용을 보존한다. 동일 SLSQP 반복이 아니라, E144의 추가적인 다리별 초기 RMS 비례 배분을 제거하고 원래 합성 물리 기준을 직접 다루는 하나의 SOCP를 구성한다. 최종 자연 보행 F1–F8은 변경하지 않는다.

H145: E142 저장 신호에 동일한 양의 근육 내 정적 가중치 하나를 적용하여 원래99의 전체 초기 기준, 후기 관절 RMS20% 감소, 전체/각 다리 속도 비악화와 각 다리 관절 비악화를 동시에 만족할 수 있다. 실패 시 수치 상태와 조건부 여유 상한을 기록해 정적 읽기 변환의 한계를 판단한다. 이 공학적 조건은 자연 보행의 필요조건이 아니며, 실패를 모든 BANC 보행의 불가능성으로 일반화하지 않는다.

고정 입력은 E144/E143/E103과 동일한 b[2000,391], pool[96,391], membership/moment/rest/ltm, 1827 target endpoint, 1826 실제 인접 velocity pair, 초기99 target이다. 새 EMA/신경/CUDA/물리/LP0. six LTM 및 singleton 근육 행은 원래 그대로, moment/rest/10ms EMA/신경/위상/시간창/표적은 고정한다. 가변 행은 원래 모든 MN을 보존하고 각 가중치>=1e-6/멤버수, <=1, 행 합1이다.

각 가변 근육의 열을 원래 MN 열 번호 순서대로 둔다. 마지막 열을 anchor로 정해 나머지 멤버 가중치 z를 자유변수로 사용하고 anchor=1-sum(z)로 구성한다. 전체 가중치는 x=a+Bz이다. 이는 solver 이전에 정한 affine 좌표계이며 사후 정규화나 clipping이 아니다. baseline z0는 원래 pool이다. 최종 배열을 직접 검사하여 행합1e-12, 양의 동일 지원, floor 위반 허용1e-12이며 반드시>0을 적용한다. 기존 runtime의1e-15 행합 호환성은 별도 관측으로 기록한다.

E144와 같은 D[t,j,i]=.02*b[t,i]*moment[j,membership_i], c=baseline-D*x0를 먼저 구성한 뒤 x=a+Bz를 대입한다. 전체 실제 인접 endpoint 차이/.005에서 pair_end-2를 골라 속도를 만든다. 각 다리 command/velocity/startup 잔차를 E[z;1]로 표현한다. 큰 norm 벡터는 thin QR의 R로 대체할 수 있다: E=QR이면 ||E[z;1]||=||R[z;1]||이다. 열 삭제·singular value 절삭은 금지한다. baseline 및 고정 생성 벡터에서 직접 잔차와 압축 norm 차이를 기록·검사한다. 원래 다리별 RMS 배분은 사용하지 않는다.

목적은 하나의 공통 무차원 여유 t를 최대화하는 것이다. 변수는 (z,t), P=0, q_t=-1, -1<=t<=1이다. 각 제약을 물리 한도로 나눈 norm<=1-t 형태로 둔다.

- 전체 command norm/(sqrt(1827*42)*(.8*baseline_command_RMS))<=1-t.
- 전체 velocity norm/(sqrt(1826*42)*baseline_velocity_RMS)<=1-t.
- 여섯 다리 각각 command/velocity norm/(sqrt(N*7)*해당 baseline_RMS)<=1-t. 12개 조건.
- 전체 startup norm/(sqrt(99*42)*.018)<=1-t.
- 초기99의 각 관절 샘플 +error/.04<=1-t 및 -error/.04<=1-t.
- 원래 pool 범위와 -1<=t<=1은 별도 선형 제약이다. tendon은 LTM 행 고정이므로 원래99 출력 exact와 .009 RMS/.025 max를 별도로 필수 검사한다.

t=0은 원래 E144 합성 물리 조건과 동치다. 원래 baseline은 후기 command ratio1.25이므로 t=-.25에서 가능하다(초기 max도 이 완화 범위 안인지 실제 검사). 따라서 t>=-1은 실현 가능한 최적해를 배제하지 않는 수치 범위다. baseline 조건이 예상과 다르면 첫 오류를 보존하고 실행을 중단한다. E144에서 선택한 일부 다리나 임의 조합으로 시작하지 않는다.

Clarabel0.11.1의 직접 Python API로 단 하나의 SOCP를 푼다. 실제 wheel SHA와 설치 파일 SHA는 freeze에 고정한다. 기존 numpy/scipy 등은 업그레이드하지 않았다. CPU1/max_threads1/direct_solve_method=qdldl, max_iter100, time_limit30s, verboseFalse, tol_gap_abs/tol_gap_rel/tol_feas=1e-10, tol_infeas_abs/tol_infeas_rel=1e-10, 나머지는 설치 버전 기본값이다. 모든 최종 settings를 raw에 저장한다. 임의 restarts/재시도/원본 solver 추가 호출0. callback은 반복 정보만 기록하며 제공되지 않는 중간 x를 기록했다고 주장하지 않는다.

원본 fit 전 matrix A,b,q, cone 종류/차원, affine support/anchor와 QR R을 저장한다. 종료 뒤 실제 x/s/z, status/iterations/solve_time/목적/잔차와 실제 정적 pool의 전체 물리 오차를 먼저 저장하고 판정한다. 저장 실패·resource 초과는 성공으로 남기지 않고 E144의 검증된 실패 기록·close/stat 보호 흐름을 재사용한다. 외부 감독은 PID/정상 종료/원본 hash·필수 저장 이후의 resource receipt를 보존한다.

정적 제안은 solver.success 문자열이나 t의 반올림에 의존하지 않고 원래 모든 실제 물리 조건으로 판정한다. 후보 벡터의 사후 clipping/정규화/constraint relaxation 금지. 하나의 반환 벡터만 사용하며 임의 중간 조합을 선택하지 않는다. 조건부 제안이 통과해도 보행 채택/F1–F8 완료가 아니며 native10초와 future/motor-cut/최종 모든 기준은 여전히 필요하다.

보조 수치 상한: conic dual z를 각 self-dual cone에 투영하고 SOC 선두를 max(선두,norm(tail))+1e-12*(1+norm(tail))로 안쪽 보정한다. 이는 후보 가중치 보정이 아니라 별도 dual 진단이다. r=q+A.T*z, L=-b.T*z+sum(min(r_i*l_i,r_i*u_i))는 알려진 box z_free∈[0,1], t∈[-1,1]를 이용한 최소 목적의 보수적 하한이므로 t의 수치 상한은 -L이다. 원래 dual, 보정 dual, cone margin, stationarity residual, dot 항과 box 보정치를 모두 저장한다. 추가1e-8*(1+|b.T*z|+sum|box terms|)를 t 상한에 더한다. 부모가 별도 코드로 재계산한다. 부동소수점 수치 상한과 solver status를 명시하며 exact rational 불가능성 증명으로 주장하지 않는다. F1–F8나 생물학적 불가능성의 증명이 아니다.

원본1프로세스/재시도0/solver1, CPU1/GPU0, fit30/internal60/external65/supervisor70초, raw128MiB/시작여유1GiB다. E144 실제 보수누적2178.7443611을 보존하고 prior2179+70=2249로 사전 배정한다. 기존2268/전체2271초 내다. 패키지 다운로드·설치·부모 생성자료 검사 비용은 별도 기록하고 원래 scientific process 비용을 초기화하지 않는다.

구현은 tools/fit_c_banc_m3_conic_pool.py 및 tests/test_c_banc_conic_pool.py 두 파일만 담당한다. E144 등 봉인 파일/활성UI/M2/F6는 불변이다. 부모 설계→실제 Sol/high FIRST 구현→부모 FIRST review→같은Sol FIRST correction(필요 시)→부모 SECOND review→부모 잔여 수정 순서를 지킨다. 저장 신호의 희소 선형식과 conic API 연결이므로 high이며 native/CUDA/checkpoint 변경은 후속 별도 xhigh 작업이다. 원본 실행은 부모 최종 source gate 뒤에만 허용한다.

공식 API 근거: https://clarabel.org/stable/python/getting_started_py/ 및 https://clarabel.org/stable/api_settings/ . solver 상태와 실제 물리 제약의 직접 검산을 구분한다.
