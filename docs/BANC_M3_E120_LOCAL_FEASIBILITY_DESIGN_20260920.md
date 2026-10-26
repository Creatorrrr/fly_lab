# E120: 모든 초기 sample을 유지한 국소 개선 가능성 진단

2026-09-20 · 부모 설계 · Sol/xhigh 구현/첫 수정 · 부모 재리뷰/잔여 직접 수정

## 질문과 고정 범위

E119의 선택점은 원래 초기99행 및 selection cap을 통과하지만 형태 개선은 실패했고, 6개 다리 모두100 unique 평가 한도에 도달했다. E120은 저장된 미분만으로 같은 작은 범위 안에 개선 방향이 남아 있는지 조사한다. E119의 반복 수를 늘리거나 새 신경/물리 응답을 실행하지 않는다. 이 국소 affine loss와 선형 residual 문제의 결과를 실제 비선형 개선, 전역 표현 용량, 자연 보행 또는 생물학적 기전 증명으로 해석하지 않는다.

전체 F1–F8 및 기존 판정은 불변이다. 마지막 실제 E113의 F3/F4/F5 실패, M2 원본, F6 두 초기 자세 미노출을 유지한다. 외부 CPG/위상/궤적/root 교정은 허용하지 않는다. 입력은 E119 선택 eval LF100/LM99/LH100/RF16/RM100/RH99의 delta/bounds, 두 loss와 gradient, q/LTM signed errors, factored startup Jacobian이다. E115 원래 bounds/edge partition 및 E119 선택 delta/manifest/부모 검증과 SHA를 동결한다. 모델 응답/미분/fit/LP 실행 전 source와 입력을 검토한다.

## 같은 h box에서 두 문제 비교

각 다리 n개의 h 변수에 epigraph t를 추가한다. h 하한=max(original_lower-delta,-.01), 상한=min(original_upper-delta,.01)이며 h=0이 포함되어야 한다. 원래 delta bounds를 확대하거나 LP h를 clipping하지 않는다. 두 문맥의 loss를 L, gradient를 G라 하면 G_i h-t <= -L_i 이다. 목적은 t+p h+P0이며 p=1.2e-4*delta/1572, P0=6e-5*sum(delta²)/1572다. prior도 선형화한 국소 문제임을 명시한다. t>=0으로 두며 local objective는 max(0,max(L+Gh))+p h+P0다.

t의 유한 상한은 1+max(0,max_i(L_i+sum_j max(G_ij*h_lower_j,G_ij*h_upper_j)))로 정한다. 모든 h box 점의 최소 feasible t가 이 값보다 작으므로 최적값을 없애지 않는다. 이 유한 box로 수치 stationarity 오차를 포함한 dual lower bound를 계산할 수 있다.

shape_box는 위 두 shape 행과 box만 사용한다. startup_box는 다음 모든 sample 행 및 RMS 조건을 추가한다. SELECTOR_CAP=(1-1e-4)^2는 E119의 실제 선택 기준 그대로다. 더 엄격한 optimizer 내부 margin 대신 실제 선택 feasible set을 진단하며 실제 채택 기준은 변하지 않는다.

Jq=v[:,:,None,:]*q_coefficient[None,None,:,:], Jz=v*ltm_gain[:,:,None]*ltm_coefficient[None,None,:]. q는(2,99,7), z는(2,99), v는(2,99,n)다. local residual은 q(h)=q+Jq h, z(h)=z+Jz h다.

행 순서는 shape2, +q1386, -q1386, +z198, -z198(C order), 이후 RMS cut 순서다. signed 행은 +/-Jq/.04*h <= sqrt(CAP) -/+ q/.04 및 +/-Jz/.025*h <= sqrt(CAP) -/+ z/.025다. 시작3170행에 문맥0 qRMS, 문맥1 qRMS, 문맥0 zRMS, 문맥1 zRMS의 h=0 supporting cut4개를 더해3174행부터 시작한다.

각 RMS 함수 g(h)는 mean((q(h)/.018)^2) 또는 mean((z(h)/.009)^2)이며 g<=CAP가 원래 국소 제약이다. 미분은 2*mean(residual*J)/scale²다. 중심 k에서 supporting cut은 grad(g(k))*h <= CAP-g(k)+grad(g(k))*k+padding이다. padding=1e-12*(1+abs(g(k))+sum(abs(grad)*(abs(k)+max(abs(h_lower),abs(h_upper)))))다. padding은 부동소수 supporting-plane 계산의 outer relaxation에만 적용한다. 후보 feasibility에는 padding이나 tolerance를 적용하지 않는다.

shape_box LP1회와 startup_box 최대8회/다리를 순서대로 실행한다. startup LP 결과에서 실제 local RMS g>CAP인 항목의 cut만 위 순서대로 append한다. 이전 행/중심/범위는 변경하지 않는다. 각 LP 입력 행렬/labels와 cut 중심·g·gradient·padding을 저장한다. 마지막8회 결과에 추가 cut을 만들지 않는다.

## feasible upper bound와 dual lower bound

LP 결과 h의 모든 box/signed/RMS 조건을 직접 검사한다. LP KKT tolerance를 실제 후보의 허용치로 재사용하지 않는다. 항상 feasible h=0을 초기 upper point로 보존한다. LP h가 실제 local feasible이면 alpha=1을 사용한다. 아니면 원점에서 해당 h까지 선분에 대해 정확히60회 이분법으로 feasible alpha 하한을 구한다. 각 중간점은 모든 box/signed/RMS 조건을 그대로 검사한다. alpha=0이 남으면 원점 그대로다. 이것은 명시적인 local ray repair이며 새 모델 호출이나 clipping이 아니다. 원래 LP x와 별도 repaired h/alpha, residual/ratios/objective를 모두 저장한다. 실제 local feasible 점 중 목적값이 가장 작은 점만 upper bound를 갱신한다.

SciPy inequality marginal y_raw에서 y=min(y_raw,0)으로 certificate용 부호를 맞춘다. 원본 marginal은 보존한다. d=c-A.T@y라 할 때 dual lower = b@y + sum(min(d*lower,d*upper))+P0 이다. 어떤 y<=0에도 유효한 finite-box Lagrangian lower bound이므로 작은 stationarity 오차를 최소값 증거로 혼동하지 않는다. 계산 오차에 대해 allowance=1e-9*(1+abs(b)@abs(y)+sum((abs(c)+abs(A).T@abs(y))*max(abs(lower),abs(upper))))를 빼고 nextafter(-inf)를 적용한다. 이는 보수적인 float64 수치 증거이며 기호적/임의정밀도 증명이라고 표기하지 않는다.

LP status0/success, shape/finite 배열, primal/dual signs, returned residuals, c@x와 fun, stationarity, complementarity 및 finite-box dual objective를 독립 검산한다. KKT 허용은1e-8*(1+해당 항의 최대 크기), solver primal/dual tolerance는1e-9다. 원본 x/marginals에 이 tolerance를 적용하며 local feasible 판정은 엄격한 원래 bounds/CAP를 따른다. shape_box는 원래 LP 결과의 h가 엄격한 box 밖이면 ray repair로 upper를 구하되 startup 제약은 적용하지 않는다.

각 startup 반복에서 누적 max(lower)와 min(feasible upper)를 보고한다. gap 허용=1e-6*(1+abs(origin_objective))이다. gap<=허용이면 종료하며 작은 음의 gap이 생겨도 이를 숨기지 않고 -허용 미만이면 오류다. 실제 local feasible LP 점이 나왔어도 gap이 크면 미해결로 끝내고 추가 LP 재시도를 하지 않는다. violated RMS가 없고 gap이 크면 numerical_gap_unresolved, 8회면 cut_limit_unresolved다.

origin objective 대비 feasible upper의 감소가1e-4*(1+abs(origin_objective)) 이상이면 feasible_descent_demonstrated다. origin-lower가 그 임계값 이하이고 gap도 작으면 limited_local_descent다. 그 외는 unresolved다. 이 임계값은 다음 연구 방향을 구분하는 진단용이며 foot/velocity20%나 F1–F8 채택 기준을 대신하지 않는다. startup 없는/있는 감소 범위 및 constraint cost interval도 함께 보고한다. feasible 감소가 남으면 다음 단계의 소수 비선형 acceptance를 별도 설계한다. 감소가 제한되어도 정적 가중치 전체/생물학적 용량의 불가능성을 주장하지 않는다.

## 구현·원자료·실행 한도

새 tools/analyze_c_banc_m3_local_feasibility.py 및 tests/test_c_banc_local_feasibility.py 두 파일만 Sol이 구현한다. 기존 source와 원자료를 수정하지 않는다. 결합된 제약/수치 경계/원자료 복구 문제이므로 기존 gpt-5.6-sol/xhigh를 사용한다. 부모는 별도 NumPy 대수·primal/dual/cut/판정 검산을 작성한다. 첫 리뷰 수정은 같은 Sol, 재리뷰 후 잔여 수정은 부모가 한다. toy CPU tests는 sample switching/4RMS/cut supporting property/ray strict bounds/dual sign과 finite t/원자료 예외 회계/작은 OA 제한을 검증한다. 원래 저장 gradient를 쓰는 LP는 부모 최종 source 검토 후 한 프로세스만 실행한다.

SciPy1.18.1 highs-ds, presolve=True,time_limit=1초,primal_feasibility_tolerance=dual_feasibility_tolerance=1e-9,threads=1. OMP/MKL/OPENBLAS 각1이다. threads 전달에 관한 기존의 정확한 Unrecognized options/HiGHS warning만 허용하며 다른 warning은 실패다. SciPy 공식 설명의 RHS sensitivity marginal 부호를 사용한다: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html . 원래 input 최대54LP(6*(1+8)), 총32RMS cuts/다리 이하, 재시작/자동재시도0. CUDA/응답/미분 bank/fit/native/body/calibration/F6 모두0이다. 작은 CPU LP 진단을 CUDA 개선이나 보행 실행으로 표기하지 않는다.

내부 wall80초, 외부100초(child timeout95초). E119 완료 누적770.2679146999726초를771로 올려 새 shared ceiling871초를 사용한다. 과거 한도는 그대로다. 각 LP 직전/직후 및 마지막 결과 저장까지 내부 deadline을 확인한다. 이미 존재하는 e120-local-feasibility/ 출력은 거부한다. 입력 NPZ/필수 start receipt 저장 성공 뒤에만 LP를 호출한다. lp_started/lp_returned/lp_materialized/lp_raw_saved를 구분한다. 반환 후 ledger나 materialization/저장 실패가 나면 이미 반환한 raw/scalars와 입력을 별도 rescue하고 원래 예외를 유지한다. raw arrays와 status/warnings를 assertion 전에 저장한다. 실패도 원본 manifest, 외부 stdout/stderr/exit/timeout/wall을 보존하며 제한을 늘려 재실행하지 않는다.

부모 검증은 저장된 배열만 사용하며 원래 LP/forward를 재실행하지 않는다. 모든 원본 manifest/input/source SHA, 6개 선택점, exact original bounds, 행/gradient/중심/cut ordering, local feasible upper, 수치 dual lower/gap/분류, 호출 수/한도, M2/E110/E113/E114/E115/E116/E117/E118/E119 보존을 검사한다. 결과가 유효하더라도 candidate_adopted=false/native_gait_tested=false/F6_exposed=false 및 전체 goal active를 유지한다.
