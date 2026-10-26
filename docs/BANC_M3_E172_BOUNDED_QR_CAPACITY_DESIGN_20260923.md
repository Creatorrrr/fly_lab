# E172: 크기를 제한한 저장 affine 표현력

Experiment E172 / Hypothesis H172 / Milestone M3 F1–F8 / diagnosis. E171의 유효한 수치 미결 및 V1 저장 schema 실패를 보존한다. 전체 목표는 실제 BANC 폐루프 자연 보행이며 이번 진단으로 대체하지 않는다.

H172는 기존 E150 저장 신호의 static 또는 history 특징에서 계수 크기2를 유지해 전체 command RMS를20% 이상 줄이는 유한 witness를 얻을 수 있는지 묻는다. 경쟁 설명은 작은 singular direction에 큰 계수를 주어야만 명목상 감소가 생기거나, 현재 제한된 특징/크기로 해당 참조를 표현하기 어렵다는 것이다. 두 가족의 signed coefficient 공간은 진단용 superset이며 음수 계수를 생물학적 근육 모델에 채택하지 않는다. 유효 witness도 F1–F8·보행·생물학의 증거가 아니다.

## 보존 입력과 비교 의미

E171 V2의18 input pin,239 shared input metadata와 sample contract를 재사용하고 새 freeze로 hash를 확인한다. 같은 E169/E170 `pre-solve-raw.npz`의43 canonical key만 읽는다. `pair_start`/`pair_end`는 저장 `affine_rows_velocity_start/end`로 연결한다. 모든 키의 ZIP directory 존재를 값 로드 전에 확인한다. original bank/rate/feature/state/activation/target-command member와 archive 전체 복사는 금지한다. 원래1824 command ticks100..1923,1823 velocity 쌍,99 startup ticks1..99,42관절/다리별7관절을 유지한다.

검토 완료된 E171 helper를 읽기 전용으로 재사용할 수 있다: dtype/shape/noncontiguous motor330/77muscle/19fixed/990pair/active/mapping validation, 선택 key loader, RMS 평가, signed endpoint physical diagnostics. E171 run/preflight/solve_block/SVD는 호출하지 않는다. 기존 전체 fitter/원본 loader도 import하지 않는다. source gate는 E172 소스·시험·설계·freeze 및 재사용 E171 source hash를 확인한 뒤에만 원본 선택 키를 열도록 한다. baseline은 static w0, history [w0,0]이며 두 가족의 per-leg/global command/velocity/startup norm 일치를 abs1e-12+rel1e-10로 사전 확인한다.

## 고정 SOCP 두 개

가족 순서 static, history. 각 가족당 Clarabel solve 한 번, 합계 최대2회 신규 fitting이다. 새 SVD/QR/EMA/history recurrence/신경/CUDA/물리/몸체 계산은0이다. solver/cutoff/regularization/seed 탐색이나 자동 재시도는0이다.

static c1980은 graph-column330×filter3×low/high2. history c3960은 alpha1980 다음 beta1980이다. command QR 여섯 개에서 `R[:,:-1]`을 active index로 원래 전역 열에 배치해 sparse M으로 연결하고, 마지막 열을 o로 연결한다. row 순서는 lf/lm/lh/rf/rm/rh다. 기존 QR 값의 truncation/scaling/recompression은0이다. command residual은 `M@c+o`이며 M은 static(1986,1980), history(3966,3960)이다. baseline norm B>0, target T=0.8B로 고정한다.

변수 x=[c,t], t∈[-1,1], 목적 maximize t, 곧 P=0/q[-1]=-1인 min q'x다. 단일 Lorentz cone은 `[T(1-t), M@c+o]`이다. Clarabel `A x+s=b`에서 SOC head A의 t열=T/b=T, tail A=[-M,0]/b=o다. static n1981/linear3962/SOC1987/A(5949,1981); history n3961/linear11882/SOC3967/A(15849,3961).

true diagnostic box는 각 coefficient[-2,2]이며 history의 alpha+beta 각1980 endpoint도[-2,2]다. 원래 양수/근육 예산/시작/속도 보호 조건은 목적의 제약으로 넣지 않고 평가만 한다. 이 superset은 원래 E169/E170 물리 후보 공간을 포함하나, 최종 보행 후보의 조건은 완화하지 않는다.

실제 수치 solve에는 사전 고정 guard g=1e-7을 사용해 coefficient와 history endpoint를[-2+g,2-g]로 제한한다. t box는[-1,1] 그대로다. clipping/정규화는0이다. linear row 순서: -I_n, +I_n, history의 -[I1980,I1980,0], +[I1980,I1980,0]. 첫 두 block의 coefficient RHS=2-g/t RHS=1, history endpoint RHS=2-g다. static linear rows0:3962, history0:11882다. true family 인증에서는 coefficient box RHS에 g를 더하되 두 t row는 유지하고 history endpoint RHS에도 g를 더한다. stationarity box lower/upper는 coefficient±2/t±1로 복원한다. inner solver 영역에 대한 인증으로 true diagnostic 영역을 배제하지 않는다.

별도 inner witness는 `(1-g)*baseline`, t=-0.251로 고정한다. 실제 original baseline과 섞지 않는다. 입력·row·baseline·inner witness·M-vs-local-QR 검증 및 raw 선저장 후에만 solve를 시작한다. matrix와 direct QR residual은 baseline, inner witness, deterministic c=sin(arange(ncoef)*0.173) 세 벡터에서 검사하며 별도 fit으로 선택하지 않는다. 반환 c에서도 같은 독립 표현을 비교한다. abs1e-12+rel1e-10의 pointwise/norm 허용치를 사용한다.

## 반환값, bound와 판정

bare return x/s/z/status/iterations/solve_time은 get_settings나 평가 전에 caller-owned raw에 보존한다. 첫 publication 실패에는 distinct fallback 파일로 같은 값을 남기고 fit을 반복하지 않는다. input/problem/returned/candidate 오류와 primary/secondary I/O/resource 오류를 구분한다. 첫 가족이 완료되고 둘째가 실패하면 첫 가족의 완료 자료는 보존하되 전체 비교의 성공으로 승격하지 않는다.

Conic dual z는 nonnegative 부분 max(z,0), Lorentz projection 후 head에 `1e-12*(1+||tail||)`를 더해 검산한다. projected cone membership와 finite 값을 확인한다. `h=q+A.T@z`, `m_j=min(h_j*lower_j,h_j*upper_j)`이면 margin upper는 `b.T@z-sum(m)+1e-8*(1+abs(b.T@z)+sum(abs(m)))`다. inner와 true b/box의 stationarity/raw 값을 모두 남긴다. 부모가 별도 scalar fsum으로 다시 계산한다. 이 finite-precision bound는 저장 M/o의 제한된 affine 문제에 대한 결과이며 실제 native 동역학 또는 생물학 전체의 엄밀한 반증은 아니다.

유효 returned witness에는 원시 계수의 exact[-2,2]와 history endpoint exact[-2,2], t exact[-1,1], finite 값, 직접 QR RMS ratio≤0.8을 요구한다. 각 leg의 수치 해상도 지표 `128*eps*max(local M.shape)*(||local M||F*||c_active||2+||offset||2)+1e-12`를 제곱합/√(1824*42)로 합한다. 이는 E171과 같은 heuristic이며 엄밀한 오차 인증은 아니다. baseline RMS의0.1% 이하여야 하고 실제 command RMS+지표가0.8×baseline RMS 이하이어야 한다. 큰 계수의 E171 결과나 정규화한 반환값으로 witness를 대체하지 않는다.

가족별 판정은 직접 유효 witness가 있으면 supported_returned_bounded_affine_witness, 없고 true projected dual upper<0이면 refuted_bounded_signed_saved_affine_family, 그 외 inconclusive다. witness와 음수 true upper가 동시에 나타나면 invalid_numerical_contradiction이다. solver 상태만으로 지지/반증하지 않는다. Solved/AlmostSolved의 완전 유한 반환과 모든 validity/resource gate를 확인한다. 다른 상태/실패는 invalid/not_tested이고 완료 하위 증거만 별도 기록한다. 두 가족이 모두 유효한 경우 하나라도 witness면 scoped H172 지지, 둘 다 음수 인증이면 두 bounded 가족 기각, 나머지는 미결이다.

같은 반환 계수를 velocity/startup QR에 대입해 다리별/global RMS를 기록한다. QR 성분을 원래 pointwise 성분으로 해석하지 않으며 startup max/실제 activation/native/clip은 not_evaluated다. E171 physical diagnostics를 사용해 alpha/beta/alpha+beta 부호·크기, 네 endpoint 전체 min/max,330MN floor 및77근육 .5–2 예산 위반을 남긴다. adoption=not_applicable/candidate_adopted=false/whole_goal_complete=false는 고정한다.

## 실행·검토·자원

실제 gpt-6-sol/high는 두 신규 파일 `tools/diagnose_c_banc_m3_bounded_qr.py`, `tests/test_c_banc_bounded_qr.py`의 FIRST 구현을 맡는다. 부모 FIRST 리뷰→같은 Sol 첫 수정→부모 SECOND 리뷰→부모 잔여 수정 순서다. freeze와 source gate, 부모 generated/scalar 검산은 부모가 소유한다. 최종 gate 전 original load/fit0이다. 생성 시험은 실제非連續 소유권/전역 scatter/한SOC head-tail 부호/box와endpoint 행/inner→true dual 복원/known feasible·excluded cases/두 번 이하 solve/bare return·첫 I/O 실패·둘째 가족 실패 보존을 다룬다. 원본 데이터에 대한 수치 가정은 metadata와 sealed source에 근거하며 생성 시험의 범위를 실제 보행 검증으로 확대하지 않는다.

Python 환경의 NumPy2.5.3/SciPy1.18.1/Clarabel0.11.1, CPU/BLAS1을 사용한다. qdldl/max_threads1/verboseFalse/max_iter100/time_limit30초, tol_gap_abs/rel/tol_feas/tol_infeas_abs/rel=1e-10으로 고정하고 실제 전체 설정을 저장한다. family fit wall≤40초, total fit wall≤80초, internal120/external135/supervisor145초. raw160MiB/free1GiB. 한 프로세스/최대2 solver/자동 retry0, 출력 경로가 존재하면 거절한다. 마지막 평가·저장 이후에도 자원 한도를 확인하고 처음 오류를 보존한다. sparse problem, local QR 또는 immutable input references, return, metadata/manifest/source snapshot을 남긴다.

이전 보수 원본 누적3049.0294185999082초를 유지한다. prospective bound는 ceil3049.0294185999082+145=3195초로 이전3167에서28초 확장한다. 생성 시험·metadata 검사·저장 결과 검산 비용은 별도 기록하고 비용을 초기화하지 않는다. M2/UI/F6를 변경하지 않는다. 다음 행동은 유효 witness/negative certificate/미결에 따라 전달 제약 또는 기존 BANC 동역학을 구체화하는 것이며, 자연 보행 기준을 낮추는 방향이 아니다.
