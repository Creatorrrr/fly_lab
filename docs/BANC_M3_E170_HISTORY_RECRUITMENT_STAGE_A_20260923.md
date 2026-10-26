# E170 활동 이력 전달 가설과 Stage A 구현 계획

부모 설계, 2026-09-23. 전체 M3 F1–F8을 유지한다. E169 원본은 native velocity 재구성 미통과로 incomplete/invalid_not_tested이고 후보 미채택이다. 완료된 저장 affine SOCP의 별도 인증만 음수 margin 상한을 확인했다. 이 경계를 보존하며 같은 고정 계수 재적합은 반복하지 않는다.

## 방법 선택

다음 가설 H170은 현재 신호뿐 아니라 이전 활동량에 따라 달라지는 양수 전달이다. E168에서 검산된 유계 R 상태를 다시 사용하되, 고정 기존 기울기에 감쇠만 가하던 E168과 달리 tonic과 history 기울기를 함께 선택한다. E169는 기존 전달 전체를 재적합했으므로 새 상태와 단순 재적합의 혼동도 줄인다. 더 많은 정적 knot를 추가하는 방법보다 시간 이력의 표현력을 먼저 구분하고, 큰 신경 회로 변경에 앞서 기존 저장자료로 제한된 가설을 확인한다.

이 선택은 생물학적 원인 판정이 아니다. 성체 다리의 NMJ facilitation을 측정하거나 검증한 것으로 부르지 않는다. 기존 전달은 위치 명령 proxy이며 근력을 복원하는 모델이라고 하지 않는다.

현재 입력 r와 기존 kappa로 s=r/(r+kappa), 0/0이면0이다. dt=.005, tau=.07을 고정하고 R[0]=1, R*=1/(1+s), R[k]=R*+(R[k−1]−R*)exp(−(1+s)dt/tau)다. 현재 control의 rate가 사용 가능해진 뒤 상태를 갱신하고 F=2(1−R)를 적용한다. R∈[.5,1], F∈[0,1], F[0]=0이며 미래/교사 위상을 입력으로 사용하지 않는다. 이미 계산한 E168 R를 재사용할 가능성은 Stage B에서 해시와 상태 일치를 먼저 검증한다.

각 MN/filter/low-high 기울기는 alpha+F*beta다. alpha,beta≥0이고 tonic alpha의 pair min을 ell, alpha+beta의 pair max를 u로 제한한다. MN sum ell≥1e−6/n, 근육 sum ell≥.5 및 sum u≤2를 보존한다. 이 보호는 유한 저장 궤적만이 아니라 모든 F∈[0,1]에서 적용하므로 실제 trace보다 보수적일 수 있다. F[0]=0은 beta 항을 없앨 뿐, 자유로운 alpha까지 원래 초기 출력과 같게 만들지는 않는다. 초기 bank/자세를 유지하고 원래 startup1..99 한도를 그대로 검사한다.

예정 Stage B는 alpha1980,beta1980,ell990,u990,t1의5941변수, nonnegative24642행과 같은15SOC다. E169의 graph-column→filter→low/high 순서, E1501824 target/1823 velocity와99startup, E85 own-leg reference, 고정19/가변77/330MN/391 해부학, 20% command와 각 다리 비악화 기준을 유지한다. baselinealpha=w0,beta0가 기존 exact baseline이다. 물리 허용오차를 포함하는 dual은 행15842:16326에만1e−12를 더해야 한다. 최종 source/fit 설계는 아직 동결하지 않았고 추가 원본 실행 권한은0이다.

## Stage A: 순수 수치 helper만 구현

새 파일 `tools/banc_m3_history_recruitment.py`와 `tests/test_c_banc_history_recruitment.py`를 실제 gpt-6-sol/medium이 FIRST 구현한다. 이 단계는 새 fitter/adapter/StateStore/GraphStore/body/신경/물리/CLI 실행기를 만들지 않는다. 모든 원본 NPZ/fit과 원본 상태 계산은0이다. 부모는 동시에 full-shape QR/저장 용량을 생성 자료로 측정하고 E169 결과를 기록한다.

필수 공개 동작은 다음과 같다.

1. `history_state(rate, kappa)`는 float64 rate(T,391),kappa(391)에서 위 고정 갱신의 R,F를 반환한다. 입력을 변경하지 않고 첫 행의 R1/F0를 유지한다. finite/nonnegative/실수 dtype/shape를 검사하며 새 tau나 knot를 찾지 않는다.
2. `evaluate_history(bank, original_activation, alpha, beta, kappa, membership, fixed_muscles, moment, rest, history)`는 activation(T,96),command(T,42)를 반환한다. alpha/beta(96,391,3,2), bank(T,391,3),history(T,391)다. 각 tick·근육별로 low=alpha_low+F*beta_low, high=alpha_high+F*beta_high, activation=sum(low*bank)+sum((high−low)*positive_hinge)를 계산하고 고정19는 original_activation을 그대로 사용한다. command는 실제 adapter와 같은 **각 tick의** `rest+.02*(moment@activation)` 순서다. 실제 기준은 `tools/banc_m3_monotone_recruitment.py`의 `_activation_from`과 `decode`다. 후보 진단을 위해 유한한 음수 계수를 clipping하지 않고 계산할 수 있지만, 물리 gate는 이를 반드시 실패로 판정한다. shape/원형을 caller가 보존한 뒤 helper를 호출하는 규약을 명시한다.
3. `parameter_gates(alpha, beta, baseline_slopes, membership, fixed_muscles)`는 실제 계수에서 exact [0,2], 원래 해부학적 support0, 고정 alpha exact/고정 beta0, MN min floor 및 근육 robust min/max budget을 계산해 bool gate와 원시 min/max를 반환한다. auxiliary ell/u를 통과 판단에 사용하지 않는다. 모든 비음수와 box는 exact, floor/budget만 기존1e−12다. 가변 근육에는 최소3개 filter의 min 합을 적용한다.

입력 shape와 dtype를 명료하게 고정하고 적절한 type hints/docstrings를 둔다. 계약 메타데이터 JSON을 읽을 수 있으나 원본 NPZ를 fixture로 쓰지 않는다. 생성 시험은 scalar recurrence, F0/무입력/지속입력/서로 다른 MN, beta0가 실제 per-tick monotone 식과 command 순서를 정확히 재현함, 각 근육 소유권/고정 근육, zero/history1의 정확한 slope extrema와 보호, 음수 계수 미-clipping/비통과, input nonmutation 및 잘못된 dtype/shape를 확인한다. 새로운 과대 검증 프레임워크를 만들지 않는다.

특히 E169의 아주 작은 command 합산 차이가 velocity에서 증폭된 실패를 보존한다. 다음 원본 fit 전에 실제 tick별 계산 순서와 baseline 포함성을 생성 자료 및 허용된 저장 baseline 검산으로 확인해야 한다. 원래 허용치를 결과를 보고 완화하지 않는다.

부모 FIRST 리뷰 후 필요한 첫 수정은 같은 Sol/medium이 수행하고, 부모 SECOND 리뷰 후 남는 수정은 부모가 맡는다. Stage A의 helper/시험 통과는 가설·보행 성공이 아니다. 부모가 이를 검토한 후에야 Stage B의 정확한 native/affine 비교법, 단일 solver, 저장·시간 cap, frozen inputs와 증거를 정하고 실제 gpt-6-sol/high에게 fitter 구현을 맡길 수 있다. 현재 원본 보수 누적2961.394460999989/3023초는 보존하며 Stage B 비용은 사전 배정 전까지0이다.
