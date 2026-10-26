# E269 — source 계수 적합을 위한 CUDA 조건부 응답과 실제 E150 전달

2026-09-27. E268 원본이 통과했고, startup .5초/teacher10초의 모든168000 RK4단계에391MN post-drive total과51source raw rate/mask/history가 있다. 자료·신경 시작/끝·기존발화와이력을 부모388검사로 확인했다. 현재 원본누적7187.140763999607/7225초,잔여37.859236000393초다. 이 문서는 구현설계이며 원본계산/fit/신체의새예약이 아니다.

## 목적과 고정 경계

E267 기존1660간선의 세 계수, 총4980개만 맞추는 하나의 제한된 fit을 준비한다. 기존 E150의 실제 monotone adapter, 신경gain/τ, CSR/노드/부호, source5/40msτ를 고정한다. E128의1572간선+96gain·단일10ms출력경로를 복사해 다른 모델을 만들지 않는다. 최종 후보는 동일계수의 전체재귀망과신체에서 판정한다. 조건부 fit만으로 F1–F8을 통과했다고 하지 않는다.

두 문맥은 길이가 다르며 매문맥이 독립적인 전체 시작 상태를 가진다. startup100controls와 teacher2000controls,20neural steps/control,4RK4stage/step이다. fit 학습은 teacher1..1200,1201..2000은 이미 관찰한 개발 비교다. F6미사용초기상태는 사용하지 않는다. 구체적 목적/solver/자원/후보선택은 첫원본fit전에 root가 별도동결한다.

## 조건부 신경 식

`I = recorded_total + sum_e W_e*mask_pre*((c0-1)*r_pre + c1*float32(hfast_pre) + c2*float32(hslow_pre))`.

기록 total은 drive가이미더해진실제float32값이다. source rate와mask는 같은RK4단계값이며history는같은단계f64를native혼합처럼한번f32로반올림한뒤f64조건부식으로올린다. W/신경물성은nativef32에서f64로올린값이다. c는f64최적화계수이며모든원소≥0,각triple합[.125,8]이다. 최종c를한번f32로반올림한뒤범위를재검사한다.

`target = cap * max(tanh(coefficient*(I-threshold-z)),0)`;
`dr = (target-r)/membrane_tau`;
`dz = (cell_gain*r-z)*cell_inverse_adaptation_tau`.

막τ와세포inverse_adaptation_tau를혼동하지않고실제배열을쓴다. 전체BANC재귀망을생성하지않으며source입력은고정이다. 이는동일float32커널의exact재현이아니라f64조건부근사다. neutral재현허용치와gradient검증은계수적합전에고정/검사한다.

각모수p=(edge,channel)의한MN민감도는초기0에서네RK4단계를동일순서로적분한다. `u=[raw_rate,float32(hfast),float32(hslow)]`, `dI/dp=W*mask*u[channel]`, `phi=cap*coefficient*(1-tanh²)` (positive branch만,경계0의미분0)다.

`d(dr)/dp = (phi*(dI/dp - dz/dp)-drate/dp)/membrane_tau`;
`d(dz)/dp = (cell_gain*drate/dp-dz/dp)*cell_inverse_adaptation_tau`.

source 이력에대한재귀민감도를여기에추가하지않는다. 그효과는실제전체재귀후보에서확인한다. 각간선은한MN에만입력되므로모수당한MN/한근육의민감도만필요하다. parameter순서는원래edge순서×channel0/1/2다.

## E150 전달과 미분

근거파일은 `tools/banc_m3_monotone_recruitment.py`, `tools/banc_m3_rhythmic_recruitment.py`, `e150-native-monotone-body/candidate-neutral`의adapter와 `e150-fixed-monotone-parameters.npz`다. 저장adapter의원래calibration/groups/parameters/bank/activation/original.activation을사용한다.

모든MN에3개bank를실제control끝rate로갱신한다: `b += alpha*(rate-b)`, `alpha=-expm1(-.005/[.005,.010,.040])`. variable근육은 `sum(low*b)+sum((high-low)*max(b-kappa,0))`. membership 밖슬로프는0이어야한다. fixed근육19개(6LTM포함)는기존anatomical경로 `a_fixed += -expm1(-.005/.010)*(pool@rate-a_fixed)`다. variable근육에별도10msEMA를더하지않는다.

bank tangent도동일3필터로갱신한다. variable activation tangent는 `sum((low+(high-low)*(bank>kappa))*dbank)`, hinge경계0미분을선택한다. fixed근육은pool값을곱한MN민감도의10msEMA다. 모수는한근육에만영향을주므로압축B[sample,param]와parameter_muscle[param]로표현할수있다.

q=`rest+.02*(moment@activation)`;LTM=`.2*tanh(activation[ltm]/10)`다. initial row는입력의bank/activation/rate/adaptation을보존하고decode한번을추가하지않는다. actualadhesion과복부두힘줄.2는실제body시험에서원래경로를유지한다;이조건부엔진에학습변수를추가하지않는다.

## 소유/API/안전

실제gpt-6-sol/xhigh가 `tools/banc_m3_source_coefficient_engine.py`, `tests/test_c_banc_source_coefficient_engine.py`만구현한다. root FIRST→필요시같은Sol첫수정→SECOND→root잔여수정이다. 기존제품/E267/E268/계획/검증자료는수정하지않는다. Solver/원본자료실행/전체BANC/신체/다른에이전트는허용하지않는다. CPU생성테스트와Ruff만실행하고비용을보고한다. CUDA생성시험도 root 사전예약후root가실행한다.

하나의context를검증하여GPU에올리는엔진으로구현하고서로길이다른context를root가구성한다. 원자료shape검사후복사한다. N≤391,P≤51,E≤1660,M≤96,Q≤42,stages≤160000이며stage수는80의배수다. source/motor/edge ID는명시적int64·unique,edge_post/source는명시적local인덱스,edgeweightf32·nonzero다. 같은source/post중복간선·잘못된membership/한MN다중근육/pool·비유한/음수rate·잘못된mask·역순stage/shape를거부한다. 모터부분집합/다리부분집합은입력에서명시적으로수축하며암묵적인행추정은금지한다. 실제원본정확한숫자와source identity/hash는root freeze에서검증한다.

입력항목: native stage total f32[S,N], source rate/mask f32[S,P], history f64[S,P,2]; initialr/z f32[N]; native tau/coefficient/threshold/cap/gain/inverse_tau f32[N]; edge_ids/motor_ids/source_ids int64,edge_post/edge_source int64[E],weights f32[E]. decoder는membership int64[N],pool f64[M,N],low/high f64[N,3],kappa f64[N],fixed bool[M],bank0 f64[N,3],activation0 및original_activation0 f64[M],moment f64[Q,M],rest f64[Q],ltm int64[L≤6]다. 모든경로의유효성과초기상태를검증한다.

권장API는pureCPU `prepare_problem(mapping)`와CUDA전용 `CudaSourceCoefficientEngine(mapping)`; `solve(coefficients, derivative=True, host=False)`; 기록MN을실제GPUdecoder에통과시키는`decode_rates(rates, host=False)`; 압축민감도에서activation cotangent곱/필요한sample의command Jacobian추출이다. 독립CPU decoder/oracle은별도이름으로구분한다. stage_index=int64 arange(S)와stage_order=start/half1/half2/end를명시하지만,이index만으로다른원자료의역순을증명할수는없으므로root입력SHA와묶는다. 반환값은rate/adaptation endpoints,bank/activation/q/LTM,선택적으로B및parameter_muscle,branch/phi,실행receipt다. fit반복은host=False를사용하고필요한q/LTM·startup99Jacobian·gradient만복사한다. metered host_copy helper와cotangent곱/전송receipt를제공하며매반복에전체phi/bank를내려받지않는다. 자료는호출자와분리하고입력검증실패시GPU생성을하지않는다. CUDA/CuPy만사용하고CPU fallback은없다.

CUDA double,`--fmad=false`,고정stream;긴kernel은최대1000neural steps로분할하여Windows TDR위험을제한한다. chunk끝의state/derivative와sample기록순서를보존한다. 입력upload/solve/derivative/host copy의요청·완료개수/byte·wall을구분한다. error뒤부분배열과완료범위를반환가능하게하고자동재시도하지않는다. 검증/모수/메모리한도는대형할당전검사하고GPU전체엔진추정4GiB상한을둔다.

## 생성 검증과 후속

CPU생성helper검사와독립small NumPy RK4 oracle을포함한다. root예약CUDA에서neutral·비중립·마스크차단·서로다른inverseτ·RK4half1/half2분리·chunk경계·nonzeroinitialbank·fixed/variable/hinge및관절/LTM을검사한다. analytic B/command/tendon tangent를중심차분과비교한다(hinge/rectification 경계에서떨어진fixture). 모든fit항목을검사할수있도록3채널·두부호·공유source·한근육복수MN을fixture에넣는다. full원본자료/optimizer는실행하지않는다.

원본학습전 root는E268neutral MN/endpoints와 실제E150decoder 경로를 확인하고,source계수적합의단한번한도/목적/시작보호/개발비교/반올림후선택/실제body연결을동결한다. 이엔진개발을또다른국소진단연쇄로바꾸지않는다.
