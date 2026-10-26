# E354: 검증된 기준 보행을 BANC 학습의 목표값으로 연결

ROOT reflected design v2. 실제 GPT-6 Astra/max 리뷰 `parent-e354-astra-design-review.json` SHA90add6401a5eb13700e97589d8130e6c70e7a1d80afa6b428fa89981b02730af와 MD SHA5de213e3f02b69809febfef816ae7414d0d3fd4d24ac8155a457ce9d0779d685의 M1–M3를 반영했다. 사용자 지정 순서를 유지한다: 현재 세션 설계 → 실제 GPT-6 Astra/max 리뷰 → ROOT 반영 → 실제 GPT-6.1 Sol/high 구현 → ROOT FIRST → 결함이 있으면 같은 Sol ONE R1 → ROOT SECOND → 잔여 ROOT 수정 → 다음 단계. high는 원본 접근·교사 provenance·분리된 autograd·독립 검산·실패/자원 계약 구현의 복잡성 때문에 사용한다. 이 문서는 구현이나 원본 실행 자격이 아니다.

활성 목표는 동일 BANC 후보의 F1–F8이며 아직 미완료다. E353은 단일 고정2.382760007990865Hz 교사 R를 실제 같은 신체에서 한 번 실행하고 독립 감사했다. 301 boundaries/15000 native/constructor2000/300 returned actions, forward1.9749839267771232mm/slip1.399688502351197mm/s/bout.995s, LF2 LM2 LH1 RF2 RM1 RH2 complete cycles, dual health와 고정 H353 모두 통과했다. `parent-e353-final-decision.json` SHAca4ca3e038aad892c3251d2c64d1ed4d0a2434d91d44942fd00dd451552a927a가 실제 worker18.8660253s/audit3.2023443s full return과 source/acquisition/독립 판정을 연결한다. 원래 E350 기술 실패와 H348 여섯 주기 실패를 바꾸지 않는다.

핵심 공백은 **좋은 교사 궤적을 신경 출력에 대한 학습 오차로 연결하는 것**이다. 기존 phase-free successor/coordination proxy는 별도로 유지한다. E353의 시간·위상 계산을 최종 실행 controller에 넣지 않는다. 새 손실의 교사 label은 오프라인 상수다. 최종 runtime에는 tick/phase/reference lookup/teacher force/root 교정/외부 CPG가 없어야 하고, 실제 BANC 전체 native dynamics에서 새 VJP와 후보 물리 검증을 거쳐야 한다. 이 단계는 그 전에 데이터와 q-only 미분 계약을 완성한다. 신경 gradient나 parameter candidate/body/자율 보행이 이번 결과라는 주장은 금지한다.

## 단일 질문과 범위

H354: qualified R의 제어명령을 정확한 오프라인 supervision으로 연결했을 때, source E335 command와의 정규화 MSE 및 q-only derivative를 실제 Torch64 producer와 독립 NumPy/math.fsum auditor가 같은 값으로 계산하는가? 교사 자신의 loss/gradient0, prefix controls1..100 error0, c0 restore와 NPZ literal의 구분, 기존 네 창의 정확한 control indexing을 함께 확인한다. 이것은 새로운 공통 방향/parameter VJP/훈련 step/학습 개선 실험이 아니다.

소유 소스는 `tools/banc_m3_reference_cadence_teacher.py`, `tests/test_c_banc_reference_cadence_teacher.py` 두 개다. 첫 파일에 lazy Torch offline objective와 새 own original diagnostic/audit gate를 둔다. 기존 runtime/goal/M2/UI/F6/old sources/supervisors/등록/교사 raw는 편집하지 않는다. Torch import와 objective construction은 원본 gate 뒤 또는 생성 fixture 안에서만 한다. module passive import/help는 Torch/GraphStore/body/CUDA/NVML를 실행하지 않는다. 유한 reader/strict JSON/scalar lowering/publisher는 frozen E351 primitives를 재사용할 수 있으나 E351/E353 old run/gate/전체 recover/loop를 호출하거나 generated=True로 actual을 우회하지 않는다.

`ReferenceCadenceTeacher`는 분리된 f64 target_q[300,42]와 normalization[6] 상수만 보유한다. constructor에서 입력 finite/positive normalization·shape/dtype를 확인하고 detach+clone한다. target/norm은 requires_grad=false, storage와 caller 변경에서 격리한다. device는 q와 같고 explicit cpu/cuda를 지원하는 tensor 경로로 구현하되 이 단계의 actual CUDA 자격은 false다. 첫 compiler/forward 때 새로운 reference phase·control clock을 계산하지 않는다. hot q 경로에 scalar extraction/CPU copies/NumPy/state reads는 없다. `_assert_async` 등 현재 Torch device-local finite 판정을 사용하고 dtype/device/shape를 엄격히 검사한다.

M1/M2의 실제 API 경계: constructor는 explicit device를 한 번 고정하고 f64 target/norm을 그 device에 detach+clone한 뒤 **private clone에서** window target을 만든다. hot q 입력은 exact device/dtype/shape여야 하며 암묵적 이동·변환을 거부한다. finite 입력뿐 아니라 loss 및 실제 반환 q gradient도 device-local finite predicate로 검사한다. constructor 이후 caller target/norm 변경이 whole/window 값과 미분을 바꾸지 않아야 한다. `banc_m3_phase_free_temporal.py`/E343 worker는 constants/helper를 얻으려고 import하지 않으며 text provenance만 상속한다. 원본 reader는 default-denied E351 reader를 새 E354 gate가 통과한 뒤 명시적으로 열고, 생성 reader는 workspace 밖 owned fixture 또는 명시적 E354 fixture adapter만 허용한다. E351 generated namespace/global을 변경하거나 actual에 generated=True를 쓰지 않는다.

whole API는 q[300,42]만 받는다. `L_full = mean_{c=1..300,l=1..6,j=1..7}((q[c,l,j]-target[c,l,j])^2 / normalization[l])`. 네 창 API는 q[4,4,42]만 받고, 사전 고정 starts=(100,164,228,292)의 controls=(101..104,165..168,229..232,293..296)에 해당하는 target를 constructor에서 선택한다. `L_window`은16controls*42 전체 mean이며 창마다 별도로 평균해 다시4로 나누는 것과 같다. prefix1..100와 startup1..99 diagnostic은 separate means로 명명해 full300/window16와 혼동하지 않는다. 새 loss는 old S/C 의미나 coefficient/scale50를 대체하거나 재라벨하지 않는다.

독립 q derivative는 각각 `2*(q-target)/(300*6*7*normalization[l])`와 `2*(q-target)/(4*4*6*7*normalization[l])`다. source q는 저장된 control command[1:301]이며 실제 theta candidate와의 매핑/미분은 없다. 원본 Torch producer에서 full leaf와 독립 window leaf 각각 autograd.grad1회, 총 q-only2회로 원자료를 수집한다. 신경 forward/backward/recompute/graph/optimizer/body/CPG/GPU0다. 교사 자기 loss/gradient는 algebraic zero를 따로 확인하며 이를 source 개선이나 parameter gradient로 부르지 않는다. four-window reset provenance는 E343의 fixed indexing과 연결하되 원래 saved neural reset state/feedback을 R와 같다고 주장하지 않는다. 원래 sensory replay를 새 교사 물리 입력으로 바꾸는 것은 다음 native VJP 설계의 범위다.

M2의 0-based mapping은 `Q_full[i]=source_command[i+1]`, `T_full[i]=teacher_command[i+1]`, `Q_window[w,k]=Q_full[starts[w]+k]`, k=0..3이다. 따라서 controls=starts[w]+k+1이고 target rows=[100:104],[164:168],[228:232],[292:296]이다. independent auditor는 producer index helper 없이 이 고정 mapping을 검산하고 같은 위치에서 `g_window[w,k]=(300/16)*g_full[starts[w]+k]`를 확인한다. 이 관계에 /4 또는 /50을 추가하지 않는다. prefix/startup은 각각 controls1..100/1..99의 separate mean이다.

## 선택 입력과 자격

등록에는 원본을 새로 hash/header/value 조회하지 않고 기존 closed acquisition descriptors·JSON qualification과 stats만 쓴다. actual fresh scientific input은 정확히3NPZ, 모두 authorized CLI 안에서 once cache로 읽는다.

- E335 `control-raw.npz`: command f64[301,42], control_tick i64[301], physics_time_s f64[301]만 선택. 기존 SHA35e0de41eefa01b2445bd77ab8740dc718403c65751a0486e57024bb97e5b072/546482B 및 selected header/CRC를 상속한다. tendon 및 다른 extra members는 ZIP directory metadata-only, NPY member/header unopened, value0.
- E332 `reference.npz`: normalization f64[6]만 선택. 기존 SHA140e23c865d7e110cd93699b41612c1ffb02eb88dc100ac435c4dcba5acf6c01/124353B와 selected header/CRC를 상속한다. curves/temperature/phase/future anchors는 ZIP directory metadata-only, NPY member/header unopened, value0.
- E353 `cadence-plan.npz`: command f64[301,42], control_tick i64[301], control_time_s f64[301]만 선택. SHA/bytes/selected header/CRC는 sealed current result의 actual acquisition descriptor와 qualified E353 auditor consumed_digest를 연결해 상속한다. source_z/R/phi/phi_R/blend/source_tick는 ZIP directory metadata-only, NPY member/header unopened, value0.

M1 reader 계약은 selected7만 NPY header/CRC/value 검사를 하고 archive.open한다. 원본 전체 파일 byte digest 및 ZIP directory 확인은 once cache 범위이며, extras의 NPY header를 추가로 읽지 않는다. 실제 최소 E351 helper/source edge를 current source proof에 pin한다.

qualified E353 final decision/report/worker seal/run+audit contract/supervision/actual identities/full-return/force-policy/model/cadence registration·reflection/current source2를 finite JSON/PY chain으로 pin한다. historical qualified teacher의 explicit external true qualification은 ROOT final decision 범위에만 있다. old saved worker/audit provisional flags는 그대로 false다. all-six cycles와 H353 conjunction, exact source/model/frequency/worker acquisition linkage를 검증한다. 새 등록/자격 없이 target teacher를 바꿀 수 없다. finite actual imported source closure와 module's lazy dependencies를 명시하고 필요 없는 recursive raw를 읽지 않는다.

original3 arrays의 dtype/Corder/NPY1.0/shape/CRC/full byte digest, clocks c*.005 tolerance2e-8, literal control0 values 및 source/teacher prefix c0..100 finite-byte exact를 확인한다. c0는 objective에 넣지 않고 original restore action이라고 하지 않는다. actor에는 control clock/normalization을 재추정하거나 frequency/offset/amplitude를 탐색하는 옵션이 없다. original reader original=True authorization은 current source2/registration/reflection/design/ROOT current generated qualification/freeze/review/reservation/정확한 sole argv를 통과한 뒤에만 열린다.

## 원자료·독립 검산·판정

fresh output `verification/banc-natural-gait-m3-20260919/e354-reference-teacher-loss`의 roster는 `raw.npz`, `receipt.json`, `result.json`, 실패 `failure.json`뿐이다. M3 completed raw는 다음 exact13 members만 허용한다. scalar shape=()도 header acquisition에 포함한다.

| member | exact dtype | exact shape |
|---|---|---|
| source_q, target_q, full_gq | <f8 | [300,42] 각각 |
| normalization | <f8 | [6] |
| window_source_q, window_target_q, window_gq | <f8 | [4,4,42] 각각 |
| window_control_tick | <i8 | [4,4] |
| full_loss, window_loss, prefix_loss, startup_loss, teacher_self_loss | <f8 | [] 각각 |

partial은 exact13의 **실제 acquired subset**만 허용하며 acquired/missing 및 현재 단계가 receipt에 연결된다. padding/추정0이고 요청만 한 gradient를 returned/acquired로 기록하지 않는다. full/window는 서로 독립된 f64 same-device leaf이며 각각 autograd.grad requested1/returned1을 실제 관측한다. 원본 요청/반환 총2회를 넘기거나 window gradient를 full slice로 대체하지 않는다. target/norm requires_grad=false와 actual q leaf/dtype/device scope를 기록한다. teacher_self_loss는 실제 직접 평가1회지만 teacher self-gradient0는 **algebraic identity, measured autograd0**로 별도 기록한다. prefix/startup0는 저장 label byte 일치이며 새 BANC startup/gait pass가 아니다.

compressed new raw는 buffer stage 후 acquisition member headers/digests를 만들고 모든 raw+JSON+reserved64KiB emergency 합계<=4MiB를 raw 첫 write 전에 검사한다. stage/cache64MiB/host2GiB/CPU threads1. 오류/late deadline/fsync/close/stdout는 already saved data를 삭제하지 않고 minimal bounded failure64KiB를 남긴다. counts/first+secondary를 보존한다. partial/technical에서 full mathematical PASS나 신경 작업을 합성하지 않는다. first raw/full result 전에 cap을 확인하고 final external nonzero가 completed result보다 우선한다. success와 failure모두4MiB, cap 확대0다.

별도 `e354-reference-teacher-audit/report.json`에 한 번만 독립 감사한다. closed source/freeze/worker actual full-return/acquisition seal/current audit reservation/sole argv가 필요하다. original3/newraw1 once cache, finite JSON no neural/Torch body execution audit. math.fsum으로 loss를 새로 계산하고 직접 derivative formula/index oracle로 full/window raw를 검산한다. producer objective 또는 Torch autograd를 auditor에서 호출하지 않는다. 정규화/MSE/scalar/mean scaling/index/dtype/shape/clock/teacher0/prefix0/raw acquisition/actual counts 및 worker stored results linkage를 각각 확인한다. math tolerance는 rtol1e-11, atol1e-12로 고정하며 bytes를 검증하는 역할과 수식 rounding tolerance를 구분한다. count signal0는 honest nullable information이지만 ordinary source c>=101 mismatch를 math-pass로 개선했다고 하지 않는다.

M3 auditor는 original3와 newraw1의 byte identity/index를 먼저 확인하고 full exact13 roster 및 q-only requested2/returned2를 검산한다. acquired partial은 기술적 evidence이며 mathematical qualification은 not_evaluated다. completed worker는 expected_exit=0인 qualified actual full-return 없이는 **new raw를 열기 전에** reject한다. source/audit의 NaN/Inf/overflow·late deadline·close/stdout error·full-exit nonzero는 이미 계산된 math PASS보다 우선한다. E351 generic publish_outcome의 diagnostic_verified/success 키를 새 H354로 승격시키지 않는다. generated producer→audit positive 및 representative late-gradient/publication failure를 같은 고정 생성 예산에서 검증한다.

H354는 label fidelity와 q-only derivative consistency가 통과한 `qualified_offline_teacher_loss` 또는 invalid/not_evaluated다. `parameter_VJP_qualified`, `gradient_step_improved`, `candidate_adopted`, `BANC_autonomy`, `F1_F8_complete`, `biology`, `natural_gait`, `M2_UI_F6_changed`는 모두false다. 통과 후에는 현재 whole BANC anatomical efficacy groups에 대한 **새 직접 R VJP**와 fresh native response 설계를 할 자격만 생긴다. old S proxy를 새로운 R parameter derivative로 선형 조합하여 대체하지 않는다.

## 자원 및 검증 순서

원본 누적 **9868.929474399501/9900s**, 남은31.070525600498513s, reserve/jobs0. original campaign cap을 reset/연장/waive하지 않는다. 이 단계는 one read-only Torch q-only original external **10/internal8s** + independent NumPy audit **5/internal4s**, 합계15s를 final ROOT current source qualification 이후 prospectively 예약한다. NN/body/GPU0, retry0. known actual outer tool full-return을 각각once charge, unknown closure는 각 cap 차감. unused allocation해제는 이전 비용 소거가 아니다.

E267/E346 불변 source와 실제 outer full-exit witness pattern을 그대로 사용한다. E353 prefix adapter는 source/prefix 계약이 E353 전용이므로 몰래 변경/우회하지 않는다. ROOT는 같은 메타데이터 witness 패턴을 E354 새 own admin 파일로 구현해 original_payload_reads0와 known PID/create-time close/actual returned exit/tool wall/late error를 연결한다. 독립 mathematical PASS+worker/audit0+no kill+RSS/byte caps+all identities ended가 있어야 ROOT final diagnostic true다. E346 narrow elapsed만으로 full exit나15s budget을 증명하지 않는다.

Sol FIRST+optional SAME ONE R1 generated **20s/8MiB**, ROOT FIRST+SECOND+residual generated **20s/8MiB**, 실패·mutating formatter·help/freeze 포함 실제 full-return 비용이다. toy body/neural/CUDA0, generated CPU Torch f64/q-only와 fixture publication/gates만. whole graph/GPU benchmark/repeated large suites0. 테스트는 full/windows analytic gradient 및 numerical finite difference(작은 fixture), normalization/clone-detach/mutation isolation, exact4-window index/mean16 vsmean4(scale fault), wrongdtype/device/nonfinite/zero norm, optional original3 selected headers/cache0 preread, caps/emergency/late failure, completed worker nonzero→reject-before-newraw, standalone CLI, current supervised generated producer→independent auditor positive/perturbed gradient/index/result rejection을 포함한다. module hot tensor path의 host scalar/clock/lookup 부재는 AST와 fixture로 확인하고 이것을 actualCUDA performance라 부르지 않는다.

ROOT final review에 결함이 없으면 형식적 R1은 실행하지 않는다. 있으면 SAME Sol ONE R1 뒤 ROOT SECOND에서 ROOT가 잔여를 마무리한다. actual source/qualified mathematical loss를 다음 native R VJP 설계로 전달하고 active goal을 유지한다. H354 또는 teacher-only 결과를 F1–F8/생물학적 보행 완료로 표시하지 않는다.
