# E335: E334 단일 후보의 실제 신체·감각 폐루프 검증

현재 세션 설계이며, 실제 Astra6/max의 조건부 진행 리뷰 M1–M7을 아래에 반영했다. 실제 GPT6.1Sol/xhigh 구현, root FIRST, 필요한 경우 같은 Sol R1, root SECOND 및 남은 root 수정 순서로 진행한다. 현재 원본 누적9431.526588899522/9660초, 잔여228.4734111004782초, 예약·활성0이다. 사용자 총 예산을 새로 만들지 않는다.

## 질문과 동일 후보

E334는 유효한 fresh control0·158706-neuron CUDA300controls/6000ticks 고정 감각 입력에서 startup 네 guard를 통과했고, steady200 smooth training7.77211086209774→7.662243283944549로1.413613%감소했다. Root NumPy5094 raw 검사PASS, external 감독25.8276879초 완료다. 이는 실제 감각이 바뀌는 신체 보행의 증명이 아니다. E333invalid/E330방향기각/E327신체기각은 그대로다.

H335: **동일한** actual theta64 SHA `d7c8f19833c7364d8f345a5ae4dec2444a4be20606d5b23daf267e15525a7e40`와 originalCSR에한번f32반올림한patchedCSR data SHA `fabd71c6b7b1153cd060901f88edbb7964403598623c913cee2dc3005de5c249`가 fresh original whole control0의 실제 신체·감각 feedback에서 기존 startup·물리적 짧은 보행 screen과 새 smooth 시간 목적을 함께 통과하는가?

신체1개, control1..300, neural6000ticks, physics15000steps/1.5초, 중간reset0, 외부CPG/phase/trajectory/root보정0, 새theta/gradient/optimizer/seed선택/line-search/원본재시도0이다. 고정E318 drives는 새body에 주입하지 않는다. 매 control 실제body observation에서 생성한 감각·drive로신경을진행한다. E334 terminal r/z/bank/activation을 body초기상태로 이식하지 않는다. C/n/τ/A/V 및 target99는 오프라인 평가 상수로만 쓰고 runtime에 입력하지 않는다. UI/M2/F6미노출과동일후보 F1–F8/해부학적노드·연결 provenance는 불변이다.

## Closure·전송·구성

Sol소유는 새 `tools/verify_c_banc_m3_smooth_secant_body.py`와 `tests/test_c_banc_smooth_secant_body.py`만이다. 이전180/183source·E318/E327/E334 helpers/results/raw/audit·계획·체크포인트를수정하지않는다. 새bodygate/run/lifecycle/classifier는직접구현한다. Immutable 공개 `tools/banc_m3_ssp_efficacy_trial.py`의 `SSPEfficacyTrial`, `patch_csr`, 원래 SSP trial 및 E318/E310 native construction/advance/readout/contact/capture/journal/evaluator API를명시적config로재사용한다. 기존 E327 `run/gate/PATHS/LIMITS` monkeypatch0, 과거E326입력을새후보로위장0이다.

Freeze는 E334원본freeze의 source183/input2328를 그대로 상속해 이전digest를현재파일로덮어쓰지않고, E318bodyproducer의additional source/input을collision-reject merge한다. Immutable SSP효능층/E327공개API/모든새own2 및 실제review/reflection/현재source-generated자격·감독·독립audit source를추가한다. E334rootfinaldecision SHA `6838c9282c2c6fd919567dd65464e267a48057bad42862902cf0d9d56e226d2a`, result/ledger/manifest/source/receipt·원본supervisor/contract·rootraw-v2/source/completed audit supervisor/contract·원본report를각각exact SHA/schema/producerlink로연결한다. 유효신경자격·원본6000receipt/confirmed1·rootraw5094alltrue·external완료/비용한번합산·전체goalactive·body0/비채택/F1–F8false를검사한다. 새literaloutput은 `verification/banc-natural-gait-m3-20260919/e335-smooth-secant-body`로fresh다. Gate완료전NPZpayload/GraphStore/StateStore/Torch/CuPy/body/neural loader0이다.

Loader는 별도 inherited E318 config를원래공개bodyloader에전달한다. Old E327loader의E326theta를읽지않는다. E334 `theta-star.npz`의theta64/32, `selected-weights-star.npz`의positions/groups/original32/actual32, `patched-csr-star.npz`의data/indices/indptr만선택적으로읽는다. 원래CSR·E29mask/현재E289roster에대해oldbyteclosure와각dtype/shape·정확한ordering을대조한다. 모든8547efficacy는positivefinitef32-promoted,group누락/초과/중복positions/zero/sign변경은차단한다. SourceCSR+θ→patchedCSR의대응과전체 palette8040를다시확인하며 **patchedCSR에theta를재적용하지않는다**.

Source whole는 E318재현 E307 control0이다. 원래 SSPRecoveryResourceTrial.from_checkpoint로body1개를만들고 SSPEfficacyTrial로 **원래CSR**에한번패치해같은SSPRK33/u0 network를구성한다. Native 실제네트워크data/indices/indptr를E334patchedCSR과exact비트대조한다. 선택roster/theta/source/candidate identity·physiology/core/motor/resource설정·r/z/drive/fullmask/resource/tick을검사한다. 효능층과새network metadata를제외한whole body/adapter/감각/rng/clock/wall기준은원래control0와exact해야한다. 새 network는새model_hash를가져야하고원래hash와동일하다고표시하지않는다. 실제adapter391motorIDs/3bank/96activation/parameters는E334approveddecoder와동일해야한다. CandidateCSR의실제내용과fused-readout/CUDApacked/20step 설정을기록한다.

SSPEfficacyTrial whole snapshot/from_checkpoint API는그대로 사용하고 기존층provenance/schema를숨기지않는다. 생성fixture에서dynamicroundtrip/변조거부/소유권 정리를검증한다. 이번원본에서는whole restore용secondbody0이고 F2/F8정식증명으로승격하지않는다. Source/network rebuild·GPU graph capture의알려진setupwork와미반환work를constructionledger에기록하며unknownstep/capture/frame이하나라도있으면invalid다.

## 원자료·평가·실패

각control requested/Python-returned/readout/acquisition/native-step/device-confirmed를분리하고source의검증된동기native API가확인한비용만confirmed로쓴다. Readout/기록오류를simulation 미실행으로되돌리지않는다. 원래 control-journal·native AdvanceLedger·constructionledger를보존한다. 부분신경/물리실패는원래예외receipt/clock을보존하며실제로반환된값만저장한다. 신경 또는물리 interval이반환했지만capture가실패하면returnedtick/step와unavailable raw를구분한다. 최초오류는후속journal/save/detach/close/게시오류가덮어쓰지못한다. Cleanup은현재ownedtrial만대상으로하고unrelatedPID/process/GPU를정리하지않는다.

Normal raw는 observed-start/final whole(StateStore 원래schema), patched-edge-roster.npz/actual patched-csr.npz, control-raw301, native-contact15000, trace301, activity/recruitment, control-journal, 아래 M3의 필수 live-input chunks와 events, evaluation raw/scalars, source/receipt/advance ledger/output manifest를포함한다. Largeforward/endpoint배열의중복0이며새runner가직접쓰는NPZ는uncompressed고정이다. 기존helper의저장방식/스키마를변경하지않고그대로보존할수있으며변경의성능효과를동등작업량benchmark로부르지않는다. CPUevaluations before irreversible science completion에서도실제 획득 자료는먼저보존한다. Nativeupright<.9/nonfootcontact에서기존health stop을유지하고정상300control완료로위장하지않는다.

평가는같은fixedC/n/τ/A/V의SmoothTemporalObjective(approvednormalization)로candidate및저장E318neutral의steady200/startup99를계산하며constructor재계산과실제사용CPU tensor bytes를확인한다. legacy E327 `evaluate_saved`/fixedshort evaluator 및shape는같은source/inputs로readonly재사용하고새classification은직접한다. E318neutral의원래startup/forward/slip/bout/cycle/health 및shape를E327기존결과·E334neutral과대조해평가불일치를차단한다. Neutral자료는추가body실행없이읽는다.

Screen통과는engineeringvalid·healthycomplete301/15000·startupqRMS≤.018/qmax≤.04/tendonRMS≤.009/max≤.025·actualsmoothtraining≤neutral−1e−9·legacyshape<같은neutral·[.5,1.5)forward≥1.6790085779712631mm·하중가중slip≤1.8026436968249955mm/s·longestforwardbout≥.3초·여섯다리eligiblecycles각>0을동일후보에서모두요구한다. 좌우협응/관절실제폭/period/uptime는진단으로보존하며짧은screen이정식F3–F8성공은아니다. 한항목실패면rejected; resource/clock/closure/publication실패면invalid다. Startup/smooth/legacy/물리gates는독립NumPy/evaluator감사에서도원래threshold그대로재적용하고producer와guard판정이경계에서불일치하면qualificationwithheld다. α/θ/threshold를사후변경하지않는다.

Result/ledger/manifest게시후readback·정합성/실제파일bytes·출력·deadline/resources를다시검사한다. 게시후오류는측정 science를보존하되run_valid/body screen 자격/handoff를철회한다. Emergency/repair는clock을reset하지않으며모든sink실패시diskagreement를보장하지않는다. External완료+owned종료+root terminalbytes독립 audit없이는원본을유효한 결과로 채택하지않는다. Observerdetach/sessionclose실패도scientificPASS로덮지않는다.

## 자원과 자격

단회 **supervisor80/internal70/preparation+science60/finalization10초**다. Runentry gate이전s0고정, science끝=s0+60, final끝=min(s0+70,finalbegin+10)로gate/load/construct/300interval/rawsave/evaluate/posthash를포함한다. 이전E327감독57.6626362초는예산근거이고새closure/목적/후반저장까지60초안에끝난다는측정증명은아니다. 외부80cutoff와poll/kill/wait뒤실제 종료 elapsed를구분하고초과/unknownwork는invalid,원본재시도0이다. CPU1/GPU1/ownedRAM6GiB/additionalGPU4GiB/output1GiB/free disk4GiB를동결한다. Torch CUDA를CuPy보다먼저초기화하고baseline/sampledGPU/Torchpeak와미포착peak·외부GPU작업의한계를기록한다. Window/service/현재UI를종료하지않는다.

Sol/root 생성CPU각90초(실패/FIRST/R1/SECOND포함)/host2GiB,rootCUDA30초/Torchpeakreserved≤256MiB/context포함sampledadditional≤1GiB를사용한다. Root현행sourceSHA와fresh CPU/CUDA report·완료owned슈퍼바이저·source-producer를originalgate에연결한다. CPUexplicitfixture와동적native SSP 소형 network/Gpuactualonecontrol20tick를구분하고neutraltheta exact·source→θ→patchednativeonly once·original/candidate identity·indices/position/group ordering·physiology/state transfer/readout/whole roundtrip/변조/owned closure를검사한다. Native vs현재grouped-current SSPrk33forward는 같은source·actualf32theta·state·drive에서 rate/adaptation의 바이트 완전 일치를 요구한다. 이전 E327 자격에 비영점 tolerance가 있었다고 간주하지 않는다. 원본의 추가 one-control calibration/parity forward는 0이다.

새runner의생성fakebody로정상300interval계수·neuralreturned후capture failure·partialphysics/native clock error·construction callback failure·hostrawsave/journal failure·health stop·cleanup/detach·postcommit/write/readback시간 초과·emergency+repair sink 실패·최초 오류/불변 deadline·원본CLI fulldimensions/body1/device 고정를검증한다. CPU테스트가생물 보행을증명한다고주장하지않는다. Generatedproof현재source 재변경 시fresh재자격이다.

독립 원자료감사는savedCPU **30 external/25 internal/20 계산+posthash/5 final 초**,host2GiB/output64MiB/원본추가0이다. Productionrunner/NN/신체를import하지않고필수StateStore leaf만읽어whole-source clock/dynamic/auditeddecoder·actualCSR_once-round·raw301/native15000·신체timestamp/receipt/terminalresource/cost및smooth NumPycovariance를계산한다. Bodyshort는immutablepure saved evaluator를 명시적인 source SHA로재사용하며판정loop를도구의과학값과분리한다. Loss/startup수치대조rtol1e−10/atol1e−10·θ/CSR/ordering/state는exact·clock의기존2e−8 tolerance는유지한다. 실패포함30초를초과하면감사자격은 미확립이며한도연장/새body0이다.

현잔여228.4734111004782초안에서source/review/generated자격/rootSECOND/passivegate후 **E33580만**별도예약한다. Original누적9660한도변경0/사용자총budget신설0/비용actual감독한번합산/후속 단계미예약이다. Screen 실패 시 후보 채택 없이 원인 검토로 돌아가며, 성공하면동일 후보의 장시간 F1–F8 통합/wholefuture/cut/F6/기존 navigation 검증을새root설계·actualAstrareview로진행한다. 이번짧은screen만으로자연스러운생물학적보행을 완료했다고 선언하지않는다.

## 실제 Astra 리뷰 반영: 필수 구현 계약 M1–M7

리뷰는 `verification/banc-natural-gait-m3-20260919/parent-e335-astra-design-review.md`, SHA `0f66a9374bf04bef868ad476f68cd212c1b9b5915a2690bdfeb71f52848a79c3`이다. 앱에 이미 있는 CLI0.159.2에 `gpt-6-astra`와 `model_reasoning_effort=max`를 지정했고 read-only 실행 exit0/turn.completed를 확인했다. 호스트의 선택·완료 증거이며 backend 독립 attestation은 없다. 파일 변경·원본 계산·예약을 승인한 리뷰가 아니다. Root가 수정 요청 전부를 수용한다.

### M1: 생산 평가와 독립 감사

생산 평가에서만 E327 `evaluate_saved`를 재사용한다. Root 독립 감사의 신체 평가 API는 `parent-e327-saved-metrics-v2.py:evaluate`이며 기존 E327 감사 producer의 SHA `b75f80920ba59886b93cf24fc157a3c483c51a92693fa43a3c33ff3055442117`를 그대로 상속한다. 이 경로의 pure saved gait evaluator 및 NumPy smooth reference만 import하고 production runner/Torch/NN/body를 import하지 않는다. StateStore 전체 loader 대신 manifest/state JSON의 semantic path와 각 NPY manifest SHA/dtype/shape를 검증한 제한 reader로 필요한 leaf만 읽는다. 전체 파일 closure의 stream SHA 검증은 payload 해석과 구분한다.

평가 배열은 startup `raw_command[1:100]`, smooth `raw_command[100:300] → raw_command[101:301]`, legacy shape `raw_command[101:301]`로 고정한다. Neutral은 기존 E318 301 raw와 동일 slice다.

### M2: 실행되는 packed CUDA 배열과 decoder 대응

`source_matrix`의 host CSR뿐 아니라 실제 native `_rowptr`, `_columns`, `_weight_values`를 읽어 증명한다. Packed columns를 uint32로 해석해 하위18bit column과 상위14bit palette index로 분리하고 palette f32 bits로 전체 가중치를 복원한다. rowptr 정수값·column 순서·복원 f32 전체 bits를 E334 CSR 및 실제 host CSR과 exact 비교한다. 실제 장치/dtype/shape/내용 digest와 palette8040을 기록한다. Generated fixture는 host CSR은 같은데 packed rowptr/column/palette가 변조된 경우도 거부해야 한다.

E334 `candidate_native_identity`는 후보 전송용 CSR+theta 합성 digest이고 실제 SSP `network.identity/model_hash`가 아니다. 두 필드는 분리하고 새 actual model_hash를 snapshot/provenance에 사용한다. Decoder는 ordered391 motor IDs, calibration의 moment/rest/pool/activation_tau/neutral/ltm, tau/kappa/membership, native slopes의 `[...,0] → low`, `[...,1]-[...,0] → delta`, fixed_muscles→96 bool fixed_mask, source bank391×3와 original96 activation 대응을 고정한다. E334 실제 decoder 증거와 데이터 bytes까지 비교하고 hash만으로 대신하지 않는다. Native whole의 bank/activation 및 original adapter 상태도 fresh control0와 exact다.

### M3: live 감각 입력 필수 원자료

자기 소유 body/adapter/network 인스턴스의 원래 bound method를 관측용으로 감싼다. 호출/반환 값·순서·kwargs·감각 지연을 변경하거나 encoder를 추가 호출하지 않는다. Encoder 내부에서 실제 반환된 `body.leg_observation()` 및 `body.knee_kinematics(include_direction=False)`, 실제 adapter.encode 반환과 실제 advance_readout 입력을 필수 관측한다. Encoder 단계에서 발생한 호출만 분리해 기록하고 기존 capture의 추가 관측과 혼동하지 않는다.

각 control의 prior body pose/rotation, leg angles42/velocity42/support6/non-support6, knee flexion6/velocity6, body time, neural before/after, control before/after, sensory-cut/dt/descending drive/IDs/navigation 설정, encoder/request/return/acquisition/confirmation event를 저장한다. Encoded drive와 실제 DN 합산 후 native 입력은 각각 `(158706,) float32`의 실제 내용이며 변조하지 않는다. Returned motor readout은 `(391,) float32`이다. 관측의 dtype/shape/내용 SHA 및 event 시각과 raw NPZ 값을 연결한다. 25control 단위 uncompressed `live-input-001-025.npz` 등의 chunks는 encoded_drive/actual_drive/motor readout·실제 관측·clock 배열을 저장하고, 완료후 대형 배열의 중복 retention은 제거한다. Partial 반환 자료는 availability/각 field control IDs를 유지하며 없는 raw를 zero나 완성 endpoint로 대체하지 않는다. Native full rate/adaptation의 매 control 복제는 필수가 아니며 observed whole start/final과 기존 activity가 이를 보완한다. Held E318 drives의 새 실행 주입·일치 요구는 0이다. Fake body에서 관측 변화→원래 encoder→native 입력 변화를 검증한다.

### M4: 실제 반환과 관측 획득의 분리

Native advance_readout 원래 호출의 Python 반환 직후 successful return/20ticks readout을 먼저 자체 메모리 receipt에 기록하고 후속 journal/capture로 지우지 않는다. 반환 이전 오류에는 실제 native exception receipt/네트워크 tick를 보존하되 tick 증가만으로 동기확인을 선언하지 않는다. Physics는 원래 `_step_physics`를 별도 instance return counter로 감싼 뒤 기존 native contact observer를 설치한다. 따라서 physics가 반환한 직후 counter가 올라가고 뒤의 contact acquisition/health 오류는 returned step을 없애지 않는다. Native clock 오류, contact 행 개수, Python 반환, 동기 readout 확인을 별도 필드로 보존한다. Cleanup은 native contact observer→원래 physics counter→encoder/body/network wrappers 순서로 원래 bound method를 복원하며 최초 오류를 보존한다.

같은 구성의 예상 construction은 setup320neural ticks, 그 중 capture 기록160ticks, settling2000physics steps다. 본 실행6000/15000과 따로 source construction ledger에 기록하고 실제 반환/미반환 setup가 하나라도 예상과 다르거나 unknown이면 invalid다. Source constructor 및 candidate graph rebuild에서 발생한 모든 setup를 포함한다. Generated 시험에는 callback 객체 반환 후 persist 실패, 신경 반환후 journal/capture 실패, physics 반환후 contact 실패, 부분 physics/clock 실패, owned 객체 회수 실패를 추가한다.

### M5: 별도 자격의 exact 범위

Fake body/whole schema CPU 자격, 실제 소형 native SSP 20tick CUDA 자격, native/functional decoder 자격을 별도 report로 분리한다. E327과 같은 native/grouped-current rate/adaptation 비교는 exact bytes다. Theta neutral 및 once-round CSR, packed 복원, state transfer/restore/future도 exact다. Decoder 입력·상수 대응은 exact다. Decoder의 동일 실제 motor rates와 순서를 사용하는 native-versus-functional 연산은 NumPy/Torch의 f64 reduction 순서가 달라 기존 `tests/test_c_banc_monotone_adjoint.py:108` 기준을 그대로 고정한다: CPU bank/original_activation/activation rtol0/atol1e−10, q/tendon rtol0/atol1e−11; 기존 CUDA-vs-CPU test309는 모든 출력 rtol0/atol1e−10이다. 이는 신경 f32 exact 요구를 완화하지 않으며 원본 관측 전에 지정한 기존 decoder 범위다. 임의 epsilon을 새로 넣지 않는다. Scientific objective/startup 독립 판정의 기존 rtol/atol1e−10 및 clock2e−8도 유지한다.

### M6: 단계표와 실제 I/O 사전 확인

한도80/70/60/10을 바꾸지 않는다. `s0`는 gate 전 한 번만 생성한다. `s0..s0+60` 안에 static gate→payload gate→construction→start/raw evidence→live controls/chunk writes→final whole→detach/close→raw save→neutral/candidate smooth 및 legacy 평가→closure/output posthash를 모두 포함한다. Final 게시·readback·정합성·recursive bytes·resource 확인은 `min(s0+70, final_begin+10)` 이내다. Emergency 및 repair 뒤에도 동일 절대 clock을 다시 검사한다. Setup/payload/hash/loop/capture/save/evaluation/close/publication 각 시간을 기록하되 측정 공백은 unknown으로 표시한다.

Sol/root generated CPU 예산 안에서 synthetic full-size encoded/actual drives300×158706 f32, motor/observation chunks와 streaming hash/readback의 I/O를 실행해 bytes·elapsed·추가 RAM을 남긴다. 기존 E31861.96/E32757.66초와 분리한 prospective 작업량 확인이며 원본60초 완료 증명으로 부르지 않는다. Recursive output 집계는 StateStore 하위 디렉터리·pending·실패 산출물 전체를 포함한다. 원본 시간 초과는 retry/연장/자료 생략 없이 invalid다. 독립 감사도 실패 포함30초 한도를 유지한다.

### M7: static gate와 payload gate 연결표

| 생산 증거 | static gate의 필수 연결 |
|---|---|
| E334 freeze | 상속 source183/input2328 exact; E318 source97/input402 모두 포함; E327 추가 source는 과거 producer pin에서 collision-reject merge |
| E334 finaldecision | SHA `6838c9282c2c6fd919567dd65464e267a48057bad42862902cf0d9d56e226d2a`; schema/run_valid/effective qualification/raw qualification/동일후보 handoff 및 nonadoption/body0/goalactive |
| E334 result/ledger/source/receipt/manifest | 정확 SHA, schema, freeze/review/producer links, 일치하는 terminal 필드, once6000confirm, 원본1/body0, 정상7raw 및2immutable metadata manifest closure |
| E334 원본 supervisor/contract | exact SHA, successful exit0/killedfalse/oneattempt/cost한번9431.526588899522 및9660한도; current owned termination 증거와 역사적 PID 재사용을 구분 |
| E334 root raw v2 | 정확 파일/SHA/schema/`passed=true`/5094 checks 모두true/terminal SHA 전부일치/audit source·freeze·review·original supervision SHA |
| E334 audit v2 완료 감독·contract | exact SHA/exit0/killedfalse/original0 및 앞 실패 포함 감사비용10.4332357/30; 최초 감사의 `formal_qualification`만으로 수용하지 않음 |
| E335 current generated/source review | 실제 own2/current SHA·CPU/CUDA별 report·supervisor/contract/helper SHA·root FIRST/R1/SECOND/reflection 링크 및 blocking0 |
| E335 output/limits | literal fresh output, nooriginalactive/noreservation-before-finalgate, fixedlimits, NPZ/GraphStore/StateStore/Torch/CuPy load 이전 static gate 완료 |

Static gate 변조 사례는 first audit로 치환, v2 checks 하나 false, terminal SHA 불일치, source SHA·producer link·완료 감독·고정 output/clock mismatch를 거부한다. Payload gate는 그 후 selected NPZ/state를 읽어 실제 theta64/32·CSR·mask/roster·whole source/decoder를 검증하고 construction 전에 끝난다. Old E318 config는 별도 전달하고 E327/E326 loader·global monkeypatch는 사용하지 않는다. 두 gate가 통과했다고 새 body 결과나 생물학 자격이 미리 통과한 것은 아니다.
