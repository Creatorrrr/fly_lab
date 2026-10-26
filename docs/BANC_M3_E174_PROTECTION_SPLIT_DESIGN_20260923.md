# E174: 같은 양수 static 신호에서 두 보호 집합을 각각 복원

## 판별 질문과 범위

E173의 완료된 static 반환점은 command 오차를 25.22% 줄이지만 startup와 근육 보호를 동시에 위반했다. 어느 위반이 개선을 제한하는지 한 점만으로 판단하지 않는다. 같은 E169 저장 affine 신호와 양수 계수 c∈[0,2]에 (A) startup q RMS/max 또는 (B) MN floor/근육 min-max 예산을 각각 복원하여 두 개의 고정된 SOCP를 한 번씩 푼다. E173 history의 실패·전체 invalid_not_tested를 보존한다. 새 특징, history 재시도, native 신경·물리 실행, UI/M2/F6 변경은 없다.

두 가족 모두 기존 command 목표 0.8×baseline, 원래 1824 command/1823 velocity/99 startup 표본, 원래 QR, coefficient 범위와 물리 한도를 유지한다. 각 가족의 직접 witness 또는 true 영역 dual 상한을 판단한다. 어느 한 보호 집합의 상한이 음수이면 해당 보호만으로도 이 저장 static 목표를 제한한다. 두 witness가 존재해도 두 집합의 동시 충족이나 다른 per-leg/velocity 조건의 충족은 따라오지 않는다. 실제 보행 후보의 채택 권한은 없다.

## 입력과 정확한 문제

E169 pre-solve-raw의 기존 43 canonical member와 problem_A의 CSC data/indices/indptr/shape, problem_b/q/cone_dims/cone_kinds/lower/upper만 읽는다. pair_start/end는 기존 affine_rows_velocity_start/end alias다. metadata key와 hash를 동결 전에 확인하고 수치값은 최종 부모 source gate 이후에만 읽는다. E169의 초기 nonnegative 20682행, 3961열과 15 SOC, q 마지막 −1 및 모든 원래 bounds를 검사한다. 원래 변수는 c1980, ell990, u990, t1이다. 저장 command와 startup SOC는 원래 QR의 정규화 행렬/offset으로 독립 비교하고 startup ± pointwise의 계수 반대 부호/offset 합/여분 열 0을 검사한다. 절대1e−12+상대1e−10 기준을 유지한다.

| 가족 | 변수와 선형 행 | SOC |
|---|---|---|
| startup | c1980,t1. 새 lower/upper box 3962행 다음 기존 row12366:20682의 startup ±.04 행 8316개. 총 linear12278. | 기존 첫 command SOC와 마지막 startup SOC(.018×sqrt(99×42)) |
| muscle | c1980,ell990,u990,t1. 새 box7922행 다음 기존 row7922:12366의 envelope/MN floor/근육 예산4444행. 총 linear12366. | 기존 첫 command SOC |

원래 sparse 행/열을 선택하여 순서와 정규화를 보존한다. P=0, q 마지막−1, t∈[−1,1]. command 및 startup SOC와 startup pointwise에는 E169의 같은 t가 남는다. t=0에서 원래 기준과 같다. startup는 envelopes를 제거하고 그 열이 해당 행에서0인지 검사한다. muscle의 MN floor는 각 MN의 sum_filter min(c_low,c_high)≥1e−6/n_owned, 근육 sum min≥.5 및 sum max≤2다. 실제 판정은 auxiliary ell/u 대신 실제 계수에서 계산한다. LTM/fixed19는 이 가변 계수로 바뀌지 않지만 이번에 native나 LTM 재검증을 수행했다고 주장하지 않는다.

inner box는 c에만 g=1e−7을 적용해 [g,2−g]로 축소한다. ell/u는[0,2],t는[−1,1] 유지다. true b는 lower c 행0:1980 및 upper c 행n:n+1980에 g를 더한다. muscle만 MN/budget 행11882:12366에 원래 physical_atol=1e−12를 더한다. 다른 행은 그대로다. true box support는 c/ell/u[0,2],t[−1,1], inner support는 c[g,2−g]다. inner dual만으로 true 영역을 배제하지 않는다.

baseline은 원래 w0다. 수치용 가능점은 각 근육 소유 MN 수 n_m으로 a_i=1/(3n_m), λ=6g×max(n_m), c_in=(1−λ)w0+λa로 한 번만 만든다. 각 pair의 ell=min(c_in),u=max(c_in),t=−.251이다. 이 convex anchor는 MN/근육 보호를 보존하고 c를 box 내부로 옮긴다. 원래 baseline physical 조건, λ∈(0,1), 모든 inner linear/SOC slack≥−1e−12를 원본 solve 전에 확인한다. 실패하면 원시값을 보존하고 invalid_not_tested, 대체 가능점 탐색은0이다.

## 판정과 검산

각 solve 반환의 x/s/z/status/iteration/time을 먼저 원시 저장한 뒤 상태·수치를 검증한다. 허용 상태는 Solved 또는 AlmostSolved이며 그 밖의 상태는 해당 실행을 invalid_not_tested로 종료한다. 먼저 완료된 가족의 증거는 전체 미완료 판정을 유지한 채 별도로 검산할 수 있다. clipping/작은 음수 수용/재시도/여러 시작점은0이다.

원래 local QR로 command/velocity/startup 및 다리별 norm/RMS를 계산하고 모든 가족에서 startup 최대값·MN floor·근육 보호를 사후 보고한다. QR 해상도 지표는 E173의 128eps×max(shape)×(norm(M)norm(c)+norm(offset))+1e−12를 kind별 적용한다. command 지표≤baseline RMS의.001이며 candidate RMS+지표≤.8baseline일 때 command witness다. startup 가족은 추가로 startup RMS+지표≤.018 및 pointwise 최대값≤.04, muscle 가족은 실제 min/max로 기존 physical_atol1e−12 floor/budget 통과가 필요하다. coefficient exact [0,2], t exact [−1,1], 유한값은 두 가족 모두 필요하다. 보조변수 slack과 inner/true 전체 선형 및 각 SOC slack을 함께 저장한다. native/생물학/보행 witness라고 부르지 않는다.

비음수 및 각 Lorentz cone으로 z를 따로 project하고 SOC head pad1e−12를 적용한다. h=q+Aᵀz에 대해 U=bᵀz−sum(min(h×lower,h×upper))+1e−8×(1+abs(bᵀz)+sum(abs(box_terms)))를 inner/true 각각 계산한다. 유한 true U<0이고 직접 witness가 없으면 해당 저장 보호 가족에서 배제, witness면 supported, 그 밖은 inconclusive다. 두 판단이 모순되면 invalid다. 유한 정밀도 상한이며 엄밀한 구간 증명은 아니다.

## 자원과 담당

가족 순서 startup→muscle, 원본 process1/최대fit2/retry0. Clarabel0.11.1/NumPy2.5.3/SciPy1.18.1, CPU1/qdldl/max_threads1/max_iter100/time_limit30, gap/feas/infeas tolerance1e−10 및 나머지 기본값을 모두 기록한다. fit당40초/합80초/internal120/external135/supervisor145초/raw160MiB/free1GiB. 이전 원본 보수누적3056.341625199886초와 이전 배정3198초를 보존하며 prospective ceil3057+145=3202초다. 새 원본 실행은 source gate 이전0이다.

실제 gpt-6-sol/high가 첫 구현과 첫 수정을 수행한다. sparse 행/열 선택, 다중 SOC, support box/물리 relaxation 및 실패 원시 보존을 함께 다루므로 high를 선택한다. 소유 파일은 tools/diagnose_c_banc_m3_protection_split.py와 tests/test_c_banc_protection_split.py다. 부모는 생성 scalar/full-shape 검산과 FIRST 리뷰, 같은 Sol의 첫 수정 후 SECOND 리뷰와 잔여 직접 수정을 수행한다. 부모가 최종 source hash gate를 만든 뒤에만 원본 실행한다. 입력 metadata/hash와 모든 반환을 검증·보존하고 후기 저장/시간 실패로 성공 판정을 남기지 않는다. 전체 F1–F8은 변경하지 않는다.
