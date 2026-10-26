# E175: 양수 endpoint 이력 전달의 전체 보호 조건 검사

E174에서 startup 보호만으로 정적 양수 전달의 저장 command 오차 20% 감소가 배제됐다. 다음 질문은 최근 신경 활동에 따라 증가하거나 감소하는 양수 전달이 같은 목표와 원래 보호 조건을 함께 충족할 수 있는가다. E170은 증가 계수 beta≥0만 허용했다. 이번에는 c(F)=(1−F)a+Fb, a,b∈[0,2], F=2(1−R)∈[0,1]로 양쪽 endpoint를 자유롭게 둔다. E170 좌표는 alpha=a, beta=b−a다. 음수 beta는 양수 실제 출력의 감소이며 음수 근육 힘이 아니다. 외부 위상·CPG·teacher 시간표를 실행 입력에 넣지 않는다. 기존 R/F의 70ms 동역학은 공학적 가설이며 생물학적 확인으로 주장하지 않는다.

단일 저장 SOCP 진단이다. 실제 신경·신체, 새로운 history update/feature/QR/SVD, UI/M2/F6 변경은 0이다. 비교 기준은 원래 w0와 같은 E170 저장 affine 데이터다. 가설이 지지되어도 native 일치와 실제 폐루프 보행 검증을 별도로 거쳐야 하며 이 실행으로 후보를 채택하지 않는다. F1–F8은 그대로 미완료다.

## 데이터와 문제 정의

E170 pre-solve-raw.npz에서 기존 canonical QR 43개와 CSC problem 10개 member만 읽는다. pair_start/end의 affine_rows_velocity_start/end alias를 유지한다. hash/ZIP metadata/NPY header 확인만 사전 허용하고 원본 수치는 부모 최종 source gate 이후에 읽는다. E170 변수 5941, 선형 24642, box 11882, epigraph 11882:15842, 실제 MN/근육 15842:16326, startup ±16326:24642, 15 SOC를 확인한다. 소유권은 실제 비연속 graph column과 원래 77근육/330MN/990pair/1980branch에 따른다. command1824tick, velocity1823pair, startup99tick과 원래 다리 순서를 유지한다.

새 변수는 a1980,b1980,ell990,u990,t1이다. T는 old=(alpha=a,beta=b−a,ell,u,t)를 만드는 5941×5941 sparse 변환이다. 기존 box/epigraph를 제거하고 다음 행을 구성한다. P=0,q_t=−1이며 나머지 q는 0이다.

| 행 | 내용 |
|---|---|
| 0:11882 | 새 −I/I box. a,b inner [g,2−g], ell/u [0,2], t [−1,1], g=1e−7 |
| 11882:19802 | pair마다 ell≤a_low,a_high,b_low,b_high 및 u≥같은 네 endpoint. 네 lower 블록 다음 네 upper 블록, 각 블록990행. inner RHS는 −eta, eta=1e−8 |
| 19802:20286 | 기존 15842:16326 MN floor/근육 budget 484행. coefficient 열0, ell/u/t 좌표 그대로. inner RHS는 원래 RHS−delta, delta=1e−7 |
| 20286:28602 | 기존 startup ±16326:24642 행에 T 적용, RHS 그대로 |
| 28602:끝 | 기존 15 SOC 행에 T 적용, RHS와 cone 순서/정규화 그대로 |

MN은 sum_filter min(네 endpoint)≥1e−6/n_owned, 근육은 sum min≥.5 및 sum max≤2다. envelope를 alpha와 alpha+beta의 두 선택값만으로 검사하지 않는다. startup q RMS .018/max .04, 전체 command RMS .8×baseline, 전체 velocity 및 다리별 command/velocity RMS≤각 baseline을 함께 보존한다. 각 기준에 원래 공통 margin t를 사용하며 t=0에서 원래 조건이다.

True 문제는 같은 A에 box 계수 lower rows0:3960와 upper rows5941:9901의 b를 g만큼 복원하고, 모든7920 envelope행은 eta를 복원하고, 484물리행은 delta+원래 physical_atol1e−12를 복원한다. 나머지 b는 그대로다. True support는 a/b/ell/u[0,2],t[−1,1]; inner support는 a/b[g,2−g]다. 수치용 guard로 true 가족을 배제하지 않는다.

Baseline은 a=b=w0다. 고정 anchor는 각 근육 MN수 n_m에 대해 v=(1−.001)w0+.001/(3n_m), a=b=v, ell=pair_min(v)−eta, u=pair_max(v)+eta, t=−.5다. 원래 baseline 물리 보호 및 모든 inner linear/SOC slack≥−1e−12를 확인한다. 초기 가능점은 solver warm start나 새 탐색이 아니며 하나만 계산한다. 실패하면 문제와 계산된 anchor/slack/물리 배열을 먼저 보존하고 invalid_not_tested로 종료한다. t=−.5는 가능점 확인용이며 판정 기준을 바꾸지 않는다.

## 변환 검증과 판정

각 local QR [M_alpha,M_beta,offset]를 [(M_alpha−M_beta),M_beta,offset]로 옮긴다. baseline a=b=w0가 old alpha=w0,beta=0와 같은 residual을 내는지 확인한다. 모든 15 SOC의 행렬/offset/limit와 원래 local QR를 절대1e−12+상대1e−10으로 비교한다. command/velocity/startup 및 다리별 RMS는 반환된 실제 a,b로 다시 계산한다. startup pointwise는 원래 비압축 ±행에서 복원하며 반환점과 baseline 모두 직접 norm과 QR norm을 같은 허용치로 확인한다. 원래 sparse startup 계수의 ±대칭, RHS합2, aux열0와 t열1을 확인한다. 행렬/범위/시간 정렬 불일치는 invalid다.

해상도 지표는 변환된 endpoint local QR마다 128eps×max(shape)×(norm(M)norm(endpoint)+norm(offset))+1e−12이며 kind별 제곱합을 합쳐 RMS로 환산한다. 전체와 다리별 command/velocity의 지표는 해당 baseline RMS의.001 이하여야 한다. 모든 반환 RMS+지표가 각 원래 한도 이하여야 witness다. startup RMS+지표≤.018, 직접 pointwise max≤.04, 실제 네 endpoint min/max의 모든 물리 보호(원래1e−12), 계수 exact[0,2]와 t exact[−1,1], 유한값도 모두 필요하다. auxiliary 허용치로 실제 위반을 숨기거나 계수를 clipping하지 않는다.

Clarabel 반환 x/s/z/status/iteration/time을 상태 검사 전에 저장한다. Solved 또는 AlmostSolved만 유효하다. 모든 inner/true linear slack과 15 SOC slack, baseline/반환 local QR residual, 직접 startup 벡터, 네 endpoint min/max/MN/근육 합을 저장한다. Dual은 기존 E174처럼 비음수 cone과 각 Lorentz cone으로 따로 project하고 SOC head pad1e−12를 더한다. h=q+Aᵀz에 대해 U=bᵀz−sum(min(h*lower,h*upper))+1e−8*(1+abs(bᵀz)+sum(abs(box_terms)))를 inner/true 각각 계산한다.

직접 witness면 supported_saved_endpoint_witness, witness가 없고 유한 true U<0이면 refuted_saved_endpoint_family, 둘 다 아니면 inconclusive다. witness와 음수 상한이 동시에 나오면 invalid다. 이는 유한 정밀도 저장 affine 진단이며 엄밀한 구간 증명 또는 BANC 전체 불가능성 증명이 아니다. 배제되면 같은 endpoint 가족의 재적합을 멈추고 신경 모집/활성 동역학 변경을 설계한다. 미결은 실패 원인을 구분하여 다음 방법을 정하며 사후 허용치 확대/같은 실행 재시도는 하지 않는다.

## 자원과 개발 담당

원본 process1/fit1/retry0. Clarabel0.11.1/NumPy2.5.3/SciPy1.18.1, CPU1/qdldl/max_threads1/max_iter100/time_limit30, gap/feas/infeas tolerance1e−10와 나머지 기본값을 기록한다. fit40초/internal75/external85/supervisor95초/raw160MiB/free1GiB다. 원본 누적3059.892423899961초와 기존 배정3202초를 유지한다. ceil3060+95=3155초는 기존3202초 이내이며 이번에 배정을 늘리지 않는다. 생성 검산/저장 검산 비용은 별도 기록하고 이전 원본비용은 초기화하지 않는다.

부모가 설계와 독립 생성 oracle을 작성한다. 실제 gpt-6-sol/high가 tools/diagnose_c_banc_m3_endpoint_history.py와 tests/test_c_banc_endpoint_history.py의 첫 구현을 수행한다. 원래 protection split의 보존·cone·dual·publication helper는 hash를 고정하여 재사용할 수 있으나 기존 소스는 변경하지 않는다. 부모 FIRST 리뷰 후 같은 Sol이 첫 수정을 맡고 부모 SECOND 리뷰 후 필요한 잔여 수정은 부모가 수행한다. source snapshot/hash gate를 통과하기 전 원본 수치 계산은0이다. 실패 원시 자료와 원래 판정을 보존한다. 공개 앱 동작 및 자연보행 개선이 완료됐다고 보고하지 않는다.
