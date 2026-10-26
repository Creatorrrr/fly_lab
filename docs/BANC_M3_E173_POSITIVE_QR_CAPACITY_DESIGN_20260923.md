# E173: 양수성만 복원한 저장 affine 표현력

Experiment E173 / H173 / M3 F1–F8 / diagnosis. 지난 E172는 실제 구현·두 리뷰·원본2fit·독립 검산으로 다음 결정을 바꾼 progress였다. 고정 크기의 signed 계수로 command 오차41.67%/48.36%를 줄였지만 음수 구동과 startup/근육 조건 때문에 채택하지 않았다. 이 관측과 모든 과거 실패를 보존한다.

H173은 **같은 저장 신호·참조·계수 크기에서 양수 계수만으로 command RMS20% 감소가 가능한가**이다. 경쟁 설명은 양수 전달 자체가 제한되거나, 양수 전달은 가능하지만 추가 근육/startup/velocity 보호 조건과의 결합이 문제라는 것이다. 이번 유일한 true 영역 변화는 E172 signed[-2,2]를 nonnegative[0,2]로 바꾸는 것이다. 긍정 결과도 동시 보호 조건이나 native 자연 보행의 성공이 아니다.

## 입력·모델 대응

E172 freeze와 source/result/검산을 pin하고 E169/E170의 같은43 selected canonical key만 읽는다. pair_start/end는 affine_rows_velocity_start/end alias이며 ZIP 존재·hash·sample contract를 값 로드 전에 확인한다. 330개 비연속 MN의990pair,77 variable muscle/19fixed, static1980/history3960, command100..1923의1824표본, velocity1823/startup99표본,6다리×7관절이다. 원본 bank/rate/history state/target command를 새로 읽거나 계산하지 않는다.

E172의 검토된 순수 함수(명령 M/o 구성, baseline/local-global 비교, RMS/해상도, dual projection/box bound, solver 설정·bare return·저장 helper)와 E171의 선택 loader/검증/physical helper를 source hash로 고정해 재사용한다. **E172/E171 run/preflight/SVD 및 과거 full fitter는 호출하지 않는다.** E172 소스·시험은 수정하지 않는다. 새 파일은 tools/diagnose_c_banc_m3_positive_qr.py와 tests/test_c_banc_positive_qr.py다. E173 schema·source gate·출력 경로는 별도로 검증한다.

baseline은 static w0, history[w0,0] 그대로다. baseline이 true 양수 영역과 history endpoint 안에 있는지 확인한다. per-leg/global command/velocity/startup norm은 두 가족 사이 abs1e-12+rel1e-10로 같아야 한다. 부모 저장 검산에서 E173 M/A/q/SOC offset/target과 E172의 해당 자료가 정확히 같은지도 확인한다. 명령 오차 비교 분모와 특징을 바꿔 양수성 효과로 부르지 않는다.

## 고정 conic 문제

각 가족당 Clarabel solve1회, 순서 static→history, 합계최대2회다. M/o는 같은 여섯 command QR의 active index scatter이며 static M(1986,1980), history M(3966,3960). x=[c,t], t∈[-1,1], P=0/q_t=-1. B=||M baseline+o||>0, T=.8B. SOC는 [T(1−t),Mc+o]이며 A head t열=T, tail=[−M,0], b head=T/tail=o다.

true coefficient는 모든 c∈[0,2]. history의 alpha+beta 각1980개도[0,2]다. 음수 beta를 양수 activation의 감소로 해석하여 허용하지 않는다. 별도 근육 예산/floor·startup·velocity 제한은 E172와 같이 fitting에서 제외하고 반환점에 대한 진단으로 남긴다.

inner guard g=1e-7. coefficient 영역[g,2−g], history endpoint[g,2−g], t는[-1,1]이다. 선형 행 순서는 E172와 같은 -I_n,+I_n, history -[I,I,0],+[I,I,0]. 첫 -I block RHS는 coefficient -g/t1, +I는2−g/t1. history negative endpoint RHS=-g, positive endpoint RHS=2−g. true RHS는 모든 coefficient 및 endpoint 행에 g를 더하고 두 t 행(n−1,2n−1)과 SOC는 변경하지 않는다. true support는 coefficient lower0/upper2, inner support lowerg/upper2−g, 두 경우 t lower−1/upper1이다. E172의 대칭 support 또는 음수 lower를 남기지 않는다.

차원은 static n1981/linear3962/SOC1987/A(5949,1981), history n3961/linear11882/SOC3967/A(15849,3961)로 같다. history endpoint lower는 coefficient nonnegative 조건에 의해 중복되지만 같은 행 구조와 true 복원을 유지한다.

사전 inner 가능점은 **c_inner=(1−4g)*baseline+2g** (각 coefficient에 상수2g), t=−.251이다. 이는 true box의 baseline을 내부 anchor .5와4g 비율로 섞는다. static 최대2−6g, history endpoint 최대2−4g로 inner coefficient/endpoint box 안에 놓인다. 원래 baseline을 이 가능점으로 바꿔 보고하지 않는다. 수치 SOC 가능성은 고정 tolerance1e−12로 원본 solve 전에 확인하고 실패하면 그 실험을 invalid로 남긴다. 사후 g/t 변경은 없다.

고정 세 probe는 original baseline, 위 inner witness, c=sin(arange(ncoef)*.173)다. sine은 제약 가능점이 아닌 순수 행렬 검산용이다. global sparse와 local QR의 각 성분·norm은 abs1e−12+rel1e−10로 비교한다. 반환 계수도 같은 검산을 거친다. 입력·문제·가능점·probe는 solve 전에 저장한다.

## 원시 반환과 판정

bare x/s/z/status/iterations/solve_time은 get_settings·평가 전에 저장하며 첫 I/O 실패에는 distinct fallback을 남기고 refit하지 않는다. 원래 오류가 resource/저장 후속 오류에 덮이지 않아야 한다. 두 번째 가족 실패 시 첫 완료 증거를 보존하고 전체 성공으로 승격하지 않는다. Solved/AlmostSolved와 완전 finite 반환, 실제 전체 설정·한도·입력·행렬 검증을 요구한다.

dual은 E172와 같은 nonnegative/Lorentz projection과 head pad1e−12*(1+tailnorm)를 사용한다. h=q+A.Tz, m=min(h*lower,h*upper), upper=b.Tz−sum(m)+1e−8*(1+abs(b.Tz)+sum(abs(m)))을 inner와 true 각각에 대해 저장한다. true 영역 기각에는 projected true upper<0만 쓴다. raw/inner 숫자를 true 인증으로 대체하지 않는다. 부모는 sparse CSC와 scalar fsum으로 독립 검산한다. finite-precision 저장 문제의 수치 인증이며 formal interval 또는 native/biology 전체 증명이 아니다.

유효 witness는 raw c exact[0,2], history alpha+beta exact[0,2], t exact[-1,1], finite, direct QR command RMS+해상도 지표≤.8×baseline RMS다. E172와 같은 per-leg heuristic 128*eps*max(Mleg.shape)*(||Mleg||F*||cactive||+||offset||)+1e−12를 제곱합 후 sqrt(1824*42)로 나누며 baseline RMS의.001 이하여야 한다. 작은 음수라도 clipping·정규화·사후 tolerance로 양수 판정하지 않는다. 수치용 inner 위반은 별도 기록하되 최종 witness는 사전 true 범위로 판정한다.

가족별 direct witness이면 supported_returned_positive_affine_witness, 없고 true upper<0이면 refuted_nonnegative_saved_affine_family, 그 외 inconclusive다. witness와 음수 upper가 동시에 있으면 invalid_numerical_contradiction이다. 두 가족 모두 유효한 경우 하나라도 witness면 제한 H173 지지, 둘 다 음수 인증이면 두 양수 가족 기각, 그 외 미결이다. solver 상태명만으로 판정하지 않는다.

반환 계수의 per-leg/global command·startup·velocity norm과 원래 표본 수 기반 RMS, alpha/beta/endpoint 부호와 크기, 모든 네 signed endpoint 기반 MN floor/근육 예산 위반을 보존한다. startup pointwise max, native activation/clipping/실제 보행은 not_evaluated다. candidate_adopted=false/adoption=not_applicable/whole_goal_complete=false/F1–F8 pending이다.

## 실행·자원·다음 결정

부모 설계→actual gpt-6-sol/high FIRST 구현→부모 FIRST 리뷰→동일 Sol 첫 수정→부모 SECOND 리뷰→부모 잔여 수정 순서를 적용한다. high는 비대칭 support, true/inner 복원, 원시 결과 보존을 함께 다루는 수치 구현이기 때문이다. 생성 시험은 양수 feasible/excluded analytic cases, 실제 비연속 소유권/차원, 음수 반환 exact 거절, true 영역과 inner 영역, 두 t 행 보존, 원래 RMS 분모, 두 solver 이하 및 실패 증거 보존을 검증한다. 원본 gate 전 original load/fit0이다.

NumPy2.5.3/SciPy1.18.1/Clarabel0.11.1, CPU/BLAS1, qdldl/max_threads1/verboseFalse/max_iter100/time_limit30초/tol_gap_abs·rel·feas·infeas_abs·rel1e−10을 유지한다. 가족fit≤40초/fit합≤80초/internal120/external135/supervisor145초/raw160MiB/free1GiB, 원본process1/최대2fits/자동retry0. serialization 후에도 resource gate를 확인한다. 새 SVD/QR/features/history/신경/물리/F6/UI 변경0이다.

이전 원본 보수누적3052.308218299877초를 보존한다. prospective bound는 ceil3052.308218299877+145=3198초로 이전3195에서3초 확장한다. 생성 시험·metadata·저장 검산 비용은 별도 기록하고 과거 비용을 초기화하지 않는다.

두 양수 가족이 모두 인증으로 기각되면 현재 저장 특징의 정적/해당 history 계수 탐색을 계속하지 않고, 기존 해부학 안의 신경·수용체 또는 근육 활성 동역학을 바꾸는 후보로 전환한다. 양수 witness가 있으면 추가 근육/startup/velocity 조건과의 결합을 근거에 따라 해결한다. 어느 경우에도 이번 진단을 실제 자연 보행의 채택 기준으로 쓰지 않는다.
