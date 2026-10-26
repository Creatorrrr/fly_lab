# E332: 선분 하나를 선택하지 않는 연속 시간 학습 목적

2026-10-01 현재 세션 설계. Actual gpt-6-astra/max의 `parent-e332-astra-design-review.md` 조건부 GO와 필수M1–M3를 반영했으며 반영본을 actual gpt-6.1-sol/xhigh에게 구현하도록 한다. E331 root3374검사는 E330 residual의 nearest-index 고정 효과 절대합이99.4488%임을 확인했다.140 RF 비국소 선택뿐 아니라208 LM/284 LH 인접 선택 변경에서도 큰 차이가 있었다. 이 비중은 인과 책임이나 비국소 불연속만의 비중이 아니다. E330 actual38.4139%>5% 기각/연속·신체 미실행은 불변이다.

## 질문·범위·금지

H332: 기존128-anchor reference와 고정 5ms step scale을 이용한 Gaussian 가중 평균은 nearest index/phase argmin 없이 미분 가능한 successor/coordination 목적을 만들고, reference의 전진·정지·역순·다리 불일치를 구별하며, 이미 저장된 E330 ± 명령 변화에서 유한 변화와 명령 미분의 불일치를 줄이는가?

E332는 새 objective 정의·생성 CPU/CUDA 자격·저장 reference 및50창 분석까지다. 효능 VJP/신경 replay/θ변경/optimizer/신체/감각 runtime/F6/M2/UI 변경은0이다. E330 gtheta는 이전 objective의 미분이므로 새 목적의 T로 재사용하지 않는다. Q와O만 비교한다. 이 단계로 신경 방향 검증이나 보행 성공을 주장하지 않는다. Reference는 오프라인 학습 loss에만 사용한다. 최종 실행에 외부 phase·CPG·trajectory·root 보정/새 연결을 추가하지 않는다.

## 사전 입력과 고정 수식

E331 완료 supervisor와 root3374검사/최종source review/최종결정/result/ledger/manifest/reference/raw를 SHA 계약으로 고정한다. E331 contract의175source/1470input 및 새 source, E331 immutable51raw file와 proof closure를 상속한다. NPZ 숫자·Torch 전에 gate를 완료한다. 기존 E329 승인 curves C[6,128,7]f64/normalization n[6]f64 bytes, joint/leg order,frequency2.382760007990865/dt.005/all15 coefficient1을 그대로 사용한다. Constructor 정상화 검증 후 승인 n을 실제 loss tensor로 넣고 bytes를 검증한다. E329/E330/E331 소스는 변경하지 않는다.

Anchor phase φ_s=s/128와 future reference A_l,s=C_l(φ_s+frequency*dt)는 q와 무관한 상수다. Anchors는 동일 승인 polyline의 고정 보간으로 생성하고 실제 bytes를 저장한다. Reference 7관절 command step mean-MSE n_l를 Euclidean squared distance로 환산한 **τ_l=7*n_l**를 Gaussian의 고정 scale로 사용한다. 임의 bandwidth·coefficient·anchor count·step-size sweep·control별 온도·학습 가능한 온도는 없다. Gaussian exponent는 −||x_l−C_l,s||²/τ_l이고 factor2를 추가하지 않는다.

`w_l,s(x)=softmax_s(−||x_l−C_l,s||²/τ_l)`.

`a_l(x)=sum_s w_l,s(x)*A_l,s`.

`z_l(x)=sum_s w_l,s(x)*exp(2πi*s/128)`.

Previous=concat(fixed_original_boundary,q[:3]),next=q[4,42]이며 boundary만 고정한다. Successor는 mean_(time,leg,joint) `(next−a(previous))²/n_l`다. Coordination은next에서 mean_(time,15unorderedpairs) `1−Re(z_a*conj(z_b))`;LR진단은same3pairs다. training=successor+coordination. 실제 구현은real/imag 두 행렬과쌍내적으로 해도 같은수식이어야 한다. Weight·future target·z 미분은 모두live다. Hardnearest/argmax/angle/phaseunwrap/selected segment/clamp 없이 모든128anchor가 참여한다. Softmax의 안정적 내부 max subtraction은 수치 처리를 위한 것으로 학습 target의 discrete 선택이 아니다. Underflow weight0은 기록하고 epsilon로 z를 normalize하지 않는다.

z의 크기를 나누지 않아 R≈0에서도 training loss와미분이유한하고, 불확실한 phase의 coordination은1근처로 남는다. R 자체는exact0에서미분불가능하므로진단으로만쓰고loss에넣지않는다. Confidence와phase를혼합하는 목적이라는 것을 명시하며 R을1로 가정하지 않는다. 높은 R도truephase식별성보장이아니다. Gaussian a는각다리의referenceconvexhull을자신안으로보내는연속함수이므로정적 successor 고정점이적어도하나존재한다. 이는total loss의전역최소가정적이라는증명은아니다. 실제고정점이centroid라는보장도없다. 대칭원형reference의중심에서는a=z=0,successor0/coordination1/traininggradient0이가능하며비영미분을강요하지않는다. 초기stochasticity·clock/phase메모리·runtimeoverride로보완하지않고아래고정식별성검사로알려진반례를판정한다. Referenceforward가exact0이어야한다고가정하지않는다. Referenceblur·고정온도·offcurve가새목적의tradeoff다.

Optional `evaluate(q[300,42],tendon[300,6])`는 기존 steady200pairs/rawcontrol100→101..299→300과 startup99targets의 동일 qRMS/max,tendonRMS/max 및 .018/.04/.009/.025 guards를 유지한다. Startuploss를새objective나candidate선택으로대체하지않는다. 실제신경/decoder전달은후속별도설계다.

## 생성 자격과 saved 분석

Sol 구현 소유는 `tools/banc_m3_smooth_temporal.py`, `tools/analyze_c_banc_m3_smooth_temporal.py`, `tests/test_c_banc_smooth_temporal.py` 세파일뿐이다. 클래스는 CPU/CUDAf64device를엄격히검사하고weights/targets/z/R/perleg/pairloss를반환한다. Independent NumPyGaussian/scalar와generatedq168좌표centralFD로previous내부연결/target/coordination/1/50gq를검증한다. 분리된generatedinput에서CPU/CUDAvalue/cotangentagreement를검증한다. z≈0/anchorseam/competingloops/adjacentnearvertex/zeroresidual/underflow/nonfinite/shape와frozenfixedconstants/partialfirsterror/publicationfailure자격을포함한다. CUDA기술오류는 scientificrefutation으로쓰지않고실패·최초오류·cost를남긴다.

Saved CPU 단회는E329승인C/n와E331reference를 exactbytes비교한뒤 다음을계산한다.

1. 모든128sharedanchorphase에서 reference forward(next=C(p+inc),previous=C(p)), static(next=previous), reverse(next=C(p−inc))의 successor/training평균·leg값을기록한다. Fixedscramble는next6legs에offsets[0,1/6,2/6,3/6,4/6,5/6]를적용한다. 모든leg의static/reverse successor평균−forward successor평균>1e−8, aggregate static/reverse training평균−forward training평균>1e−8이어야progress식별성이pass다. Scramblecoordination−matchedforwardcoordination>1e−8을요구한다. Exactzero/nonzerogradient 자체로보행채택하지않는다.
2. 위 모든reference점과고정정적curve평균point, referenceanchorstatic, E318/E327저장q에서weights/R/entropy/maxweight/targetblur/offcurve거리와loss를보고한다. Static집합 S는6다리globalphase가맞는128anchor x_s와다리별curve평균x_mean의129개고정점이다. `static_margin=min_(x in S)L(x,x)−mean_s L(x_s,C(p_s+inc))>1e−8`이어야known_static_discrimination=true다. L은unscaledloss이며단일pair와sameboundary/4반복행window는같은값이어야한다. 가장낮은static을평균으로숨기지않는다. 이는새최적화/후보탐색이아니라고정진단reduction이며통과해도다른정적고정점이없다는증명은아니다. E318/E327는E329에서승인한두raw경로/301clock/steadyraw100→101..299→300을exactreuse한다. 후자가reference보다낮다는이유로candidate를채택하지않는다. R≈0에서math유한성은PASS일수있지만학습식별성제한은명시한다. 추가다중초기seed/후보선택·F6는없다. Targetblur는reference에서meanjoint-MSE(a(C(p))−C(p+inc))/n_l로정의하고offcurve거리/entropy/R를함께읽는다. Phase구간별min/max를보고하며reference시간이동후score불변을강요하지않는다.
3. E330 저장50창neutral/plus/minus q와원래q[c]boundary를exact비교한다. 새 CPU losses/gq=grad(Lneutral/50,q) 및δq에대해O=(Lplus−Lminus)/50,Q=gq·δq와O−Q를계산·raw저장한다. 기존T/E330step방향은새gradient로이름바꾸지않는다. 새O/Qsignedsum,abs기여합,top창과component/leg/pair를보고하고신경cotangent/biology책임을만들지않는다.
4. 사전 input qualification은analysis/입력valid,위referenceprogress식별성,known_static_discrimination와다음두command비율의conjunction이다. `D_signed=max(abs(sum(O)),abs(sum(Q)))`, `e_signed=abs(sum(O−Q))/D_signed`; `D_L1=max(sum(abs(O)),sum(abs(Q)))`, `e_L1=sum(abs(O−Q))/D_L1`로고정한다. 두분모>0이고두비율각<=.05여야finite_command_qualification=true다. 0분모비율은JSONnull,자격false/이유no_direction_information이고계산가능한나머지비율은보고한다. O/Q모두0이면absoluteclosure는유효할수있다. 비유한값은invalid이며극소비영signal의실제크기도보고한다. 이전E330의5%기준을바꾸거나기각을소급재분류하지않는다. 이새자격은기존저장δq에서새objective의유한변화검사일뿐효능gtheta방향의자격이아니다. 실패시τ/sweep으로고치지않고결과를보존하여다음구조적가설을설계한다.

Raw에는C/n/τ/A(상수futureanchors)/V(원형cos,sinbasis)/referencecaseq,weights/R/target/loss,50창boundary/q/gq/δq/O/Q/perleg/pair값을재현가능하게저장한다. 보존size한도내필요key만읽고원래gtheta/rates/adaptation/fullCSR/neuralloader를읽거나호출하지않는다. Source/inputposthash,validity/firsterror/partialcost,strictfiniteJSON/manifest/freshoutput을요구한다. Root는supervisor종료뒤독립NumPy/FD/실제raw대조를한다.

## Actual Astra M3 반영: 상수·수치·실행 계약

원초안SHA5a310358677686d8e020fd145f4a39fc76b19bf5ebc9c0f1e035db5836999d87와리뷰SHA0aad9c1397dc2522c754c4e6e8ed01ec102aa0c716e73d2f4aaedf8fe6f37770를불변보존한다. M1signed/L1, M2정적고정점/129static최저margin/1e−8 및M3상수·허용오차·closure/별도rootaudit비용을모두반영했다. Optional시간순열은이번자격에추가하지않고leg-offsetscramble를시간순열검사라고부르지않는다. Covariance미분과R/offcurve/blur해석을반영한다.

Canonical상수는승인C/n에서CPU NumPy<f8/C-order로한번생성한다. n의constructor검증은rtol1e−12/atol1e−15다. τ[6]=7*n, A[6,128,7]는승인E329 `numpy_interpolate(C,tile(arange(128)/128)+inc)`를(128,6,7)→(6,128,7)C-order로옮긴값, V[128,2]는angle=2*np.pi*(arange(128,dtype=f64)/128)의cos/sin두열이다. 실제tensor전송전shape/dtype/finite/τ>0/bytesSHA를기록한다. CPU/CUDAconstructor독립재계산n/τ/A/V를canonicalCPU값과rtol1e−12/atol1e−15로먼저검증한뒤실제loss에canonicalbytes를그대로복사하여host-copySHA로검사한다. 실패를overwrite로숨기지않는다. SavedE85CUDA실행은0이다.

| 비교 | 고정 허용오차 |
|---|---|
| unscaled CPU L/성분/target/z/weights vs 독립NumPy | rtol1e−10/atol1e−10 |
| 168좌표centralFD(h=1e−6rad) vs unscaled∇L | rtol1e−6/atol1e−7; gq scaled atol2e−9 |
| covariance닫힌식 vs unscaledautograd∇L | rtol1e−8/atol1e−8; gq scaled atol2e−10 |
| generatedCPU/CUDA unscaledL/성분 | rtol1e−10/atol1e−10 |
| generatedCPU/CUDA unscaled∇L | rtol1e−8/atol1e−8; gq scaled atol2e−10 |
| /50identity·성분/좌표기여closure | rtol1e−12/atol1e−12 |

L/50의value를직접비교할때absolute허용오차도1/50로줄인다. Covariance미분은 `Da=(2/τ)sum w(A−a)(C−mu)^T`, `Dz=(2/τ)sum w(v−z)(C−mu)^T`;coordleggradient는 `−Dz_l^T sum_(k!=l)z_k/(T*15)`다. q내부3행에는다음행target미분이들어가고boundary만고정한다. Root independent수식helper `parent-e332-independent-math.py`는Sol읽기전용이다. Entropy는0log0=0으로안전하게계산하며τ/z에epsilon을더하지않는다. finite입력의square/logit/gradientoverflow도failclosed한다.

Inheritedgate는E331exact source/inputmap+새3sources+immutable51rawfiles와result/ledger/manifest/auditsource/3374checksalltrue/완료supervisor/SECOND승인/finaldecision의hashlink를확인한다. E33050창순서와세arm원래경계/301clock/E331q·원래E330q bytes를대조한다. 실험개수만맞추는gate는불충분하다. 성분O/Q와명령좌표별Q의의미를분리하고sumclosure를검사하며인과leg책임으로쓰지않는다.

## 리뷰·자원·후속 제한

Root설계→actualAstra6/max→root반영→actualGPT6.1Sol/xhigh→rootFIRST→필요시같은SolR1→rootSECOND→잔여root수정의사용자정책을지킨다. xhigh는새smoothobjective의fullcotangent/CPUCUDA/상속SHA계약과부분실패처리를함께다루기때문이다.

Sol/root generatedCPU각90초(FIRST/R1/SECOND누적), rootgeneratedCUDA60초, savedCPU supervisor120/internal110/calculation+rehash100/finalization10초, host2GiB/output128MiB/free1GiB/CPUthread1. Root independent savedrawaudit도별도supervision/costreceipt에서same120/110/100/10·2GiB/128MiB/1GiB계약을사전재사용한다. 이는새원본배정/사용자총예산이아니다. GeneratedCUDA Torch256MiB/context포함additionalGPU1GiB;actualCUDAqualification증거를CPUtests로대체하지않는다. CUDA종료/synchronize와부분비용을확인한다. CUDAoriginal0/savedCPU에는CUDA숨김. 신경/신체original0;누적9200.673308899522/9360초/예약0불변이며새원본배정은없다. 모든실패를보존하고stage비용을별도로기록한다.

E332가valid/inputqualificationPASS여도효능backprop·θcandidate·reset없는연속rollout·신체handoff는새E333사전설계및same5%/startup/native조건을통과해야한다. E330실패계약과동일후보F1–F8/최종신경단독/M2/UI/F6는불변이다. 자연스러운보행·생물학적기전·wholegoal완료는false다.
