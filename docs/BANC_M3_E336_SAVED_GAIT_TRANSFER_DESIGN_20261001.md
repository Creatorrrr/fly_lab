# E336 사전 설계: 저장된 신경 명령과 실제 보행의 전달 진단

현재 세션이 실제 Astra6/max 리뷰의 M1–M8을 반영한 설계다. 아래 반영 계약이 초안의 불명확한 표현을 대체한다. 사용자가 지정한 root 설계→Astra6/max 리뷰→반영→GPT6.1Sol 적정 effort 구현→root FIRST→필요시 같은 Sol R1→root SECOND/남은 root 수정 순서를 따른다. 최종 F1–F8·M2·UI·F6·해부학 provenance·신경 단독 계약과 과거 invalid/rejected 결과를 유지한다.

## 문제, 주장과 가능한 결론

E334의 고정 입력 신경 screen과 E335의 실제 감각 폐루프는 동일 theta다. E335 원본은 유효하고 독립 7082검사가 모두 PASS지만 smooth loss 개선 0.9473%가 실제 보행 품질로 이어지지 않았다. Slip과 bout·eligible cycles·legacy shape 기준에 미달하여 후보를 기각했다. 완료 external marker/final decision을 읽고 원본 종료와 비용 단회 합산을 확인한 뒤 진단한다.

H336은 저장 trajectory에서 **명령의 위상 진행 부족, 신체 추적 손실, 실제 감각 입력 조건의 차이**를 분리해서 다음 개발 결정을 좁힐 수 있다는 가설이다. E318 neutral 실제 보행, E334 같은 theta의 held-input 신경 command300, E335 같은 theta의 live 실제 보행을 비교한다. E334에는 신체가 없으므로 실제 보행 비교군으로 세지 않는다. E335-E318은 같은 입력 시계에서도 다른 상태에서 발생한 trajectory이고 E335-E334도 폐루프 조건 변화와 이후 상태 변화가 함께 있으므로 상관·차이를 피드백의 인과 효과 또는 body Jacobian으로 부르지 않는다.

모델·입력·theta·gain·시간상수·학습창·임계값은 변경하지 않는다. 새 신경·CUDA·신체 실행, optimizer/gradient·고정점 solver·위상/CPG 출력·시드 탐색은 0이다. Offline 저장 q의 soft encoding과 기하학 분석만 실행한다. 정적 weighted-target 잔차가 작아도 동역학의 고정점/안정성/모든 정적 해의 부재를 입증하지 않는다.

## 입력과 byte closure

E335 freeze의 source203/input2418와 E335 완료 decision의 evidence 전체를 exact collision-reject merge한다. E334 완료 decision/5094 raw audit v2/원본 supervisor·contract 및 E335 7082 v3/audit supervisor·contract/원본 supervisor·contract/원본 accounting/외부 완료 marker의 실제 SHA와 링크를 확인한다. 두 source와 root의 current source-review/generated qualification을 별도로 연결한다. 슬래시·absolute/repo-relative의 동일 경로 alias는 정규화하되 저장된 producer pin을 다시 계산하여 대체하지 않는다. 같은 경로에 다른 SHA·repo escape·중복 alias conflict는 거부한다.

Static gate는 freeze/review/현재 source/input의 streaming SHA 및 completed producer의 schema·verdict·checks·terminal SHA를 검증한다. 이 gate 전 NPZ 수치·StateStore loader·GraphStore·Torch·CuPy·신경·신체 loader는 0이다. Gate 이후에도 NPZ는 필요한 key만 선택해서 읽고 원본률/adaptation·전체 CSR·whole NPY를 수치로 읽지 않는다. 모든 실제 분석 입력은 고정 manifest SHA/bytes에서 선택하고 posthash한다. Meta/header 확인은 payload 수치 해석과 구분한다.

| 원자료 | 읽는 내용과 경계 |
|---|---|
| E332 canonical reference.npz | E333 freeze의 paths.e332_result로 위치를 해석하고 curves(6,128,7), normalization(6), temperature, future_anchors, phase_basis만 선택; 부모 NPZ로부터 임의 다른 reference를 찾지 않음 |
| E318/E335 control-raw.npz | command301×42, tendon301×6, motor_rate32 301×391 및 control/network/physics clocks; activation은 이번 분석 불필요 |
| E334 continuation-forward.npz | q300×42와 tendon300×6만; terminal rate/adaptation/bank/activation key는 해석하지 않음 |
| E318/E335 trace.json | 301 contiguous samples, joint actual/targets/tracking-error42, body geometry/basis/pose, feet, adhesion, time와 clock; 기존 trace source schema |
| E318/E335 native-contact.npz | time15000, normal load, slip/slip_valid, upright/nonfoot; 필요 key만 |
| E335 live-input 12chunks | control/availability·clock·prior pose·angles/velocity/support/knee·encoded/actual drive·returned motor; 각 필드별 IDs300 연속 exact |
| E318 full-drive 12chunks | drive25×158706, motor_rate25×391, control/network_tick만; full rate/adaptation/mask를 펼치지 않음 |

큰 drive는 25control 한 chunk씩 처리해 per-control scalar를 저장하고 즉시 해제한다. 모든 원본 raw·실패·whole 상태를 수정하지 않는다. F6 파일을 읽거나 확인 상태를 고르지 않는다. E335 cue cut에서도 기존 tonic 20포트 -5000 및 bilateral DNg100 +5000은 유지되며, 이를 전부 0이라고 가정하지 않는다. 해당 기존 engineering input 가설을 생물학적 afferent 복원으로 해석하지 않는다.

## 고정 시계와 전달 대조

Control 경계는 c=0..300, t=.005c, network tick20c, native t=.0001·1..15000이다. E334 q[i]는 c=i+1이고 E335 raw q[c]와 대조한다. 시작 구간은 c1..99, steady command 분석은 q[100:301] 및 edge q[100:300]→q[101:301], 물리 평가는 기존 [.5,1.5) 200samples와 그 native substeps다.

Live chunk control c의 angles/knee/prior pose는 **신경 계산 전 경계 c−1**이다. Trace joint_angles[c−1] 및 trace geometry[c−1]와 대조한다. 같은 control의 motor_readout은 raw motor_rate32[c]와 exact다. Driveheld[c]와 actual_live[c]는 같은 row/control을 비교하며 그 상태가 같다고 가정하지 않는다. Command q[c]는 경계 c에서 끝나는 한 control interval의 target이며 trace joint_targets[c]와 exact 대응한다. 신체 추적은 trace actual[c]−target[c]를 사용하고 저장된 joint_tracking_error_rad=target−actual의 음수와 검증한다. 사후 최적 lag를 찾지 않는다. Command increment q[c]−q[c−1], 실제 joint increment actual[c]−actual[c−1]을 같은 축/순서로 기록한다.

Source walking_trace는 실제 joint_angles/targets를 모두 기록한다. source body.leg_observation은 같은 qpos_ids 순서를 사용한다. 기존 geometric knee interior와 knee input flexion은 서로 보각이며, input[c] = pi−deg2rad(interior(trace[c−1]))를 기존 tolerance로 검증한다. Live position/rotation에는 고정 flylab/common.py의 S=[[1,0,0],[0,0,1],[0,-1,0]]를 적용해 trace position=S@position 및 basis=S@rotation과 대조한다. Support sensor에서 adhesion을 뺀 receptor load와 native floor load는 정의가 다르므로 둘을 같다고 요구하지 않는다. 순간 joint velocity를 표본 간 finite difference와 exact 비교하지 않는다. 단위는 command/actual rad, geometry interior deg, normalized foot AP/leg length, load BW, native slip mm/s다.

## 사전 고정 분석

1. **명령의 진행:** 고정 pure NumPy parent-e332-independent-math.py와 transitive parent-e329-independent-qualification.py의 수식을 따르는 새 **값 전용** 경로로 동일 full128 soft weights/target/z/R와 loss만 구한다. 원본에서 두 helper의 encode/terms/window/main을 호출하지 않으며 Jacobian/gradient/finite difference는 0이다. Generated fixture에서만 기존 pure helper와 값 대조가 가능하다. Steady smooth scalar는 완료 E334/E335 audit의 값과 기존 rtol/atol1e−10으로 대조한다. Encoded phase는 atan2(z_y,z_x)/2pi이고 서로 인접한 두 표본 모두 R≥.5인 경우에만 principal signed delta를 계산한다. Gap을 이어 unwrap하지 않으며 per-leg valid count·contiguous segments·delta percentile/net signed progress·reverse count를 저장한다. 낮은 R이면 invalid/unknown으로 표시한다. 샘플링 alias 가능성 및 soft global phase와 해부학적 위상의 차이를 명시한다.
2. **목적의 모호성:** 각 다리 successor residual, q−soft_future_target(q)의 static residual, actual command increment와 reference anchor increment scale을 저장한다. Per-leg normalized RMS 및 full128 weight entropy/R를 원자료로 제공한다. 곡선 C_l(s)는 공통 LF-reference 위상에 정의되어 다리별 offset을 이미 포함하므로 encoded global phase agreement를 같은 다리 해부학적 위상으로 해석하지 않는다. 잔차가 작은 관찰만으로 정적 attractor를 선언하지 않는다. 추가 고정점 반복·parameter 탐색은 없다.
3. **신체 전달:** E318/E335 command·actual per-leg/joint 95−5 span, 고정 zero-lag tracking RMS/max, command/actual increment RMS와 기준 step scale 대비 비율을 기록한다. 실제 geometric knee 및 foot AP 95−5와 움직임/전진 bout 분할을 고정 offline gait.load_model/analyze로 평가한다. 저진폭일 때 span 비율 분모가 0이면 null/undefined로 표시하고 epsilon으로 큰 비율을 만들지 않는다. 전체 구간에서 실제 oscillator가 있어도 공식 eligible bouts에서 제외될 수 있음을 보고한다.
4. **접촉과 위상:** 기존 물리 evaluator의 forward/slip/bout/cycles는 그대로 재현한다. Native per-leg load-weighted slip/support fraction·contact transitions와 boundary adhesion을 별도로 보존한다. 공식 eligible-cycle 및 phase invalidity를 그대로 둔다. 모든 [.5,1.5) 표본을 이용한 exploratory phase/events는 기존 gait.analyze의 moving=window 경로에 별도 라벨로 저장할 수 있으나 공식 통과/다음 threshold 선택의 근거로 세지 않는다. Contact transitions를 걸음 수로 세지 않는다.
5. **입력 조건 차이:** E318 held vs E335 actual drive의 per-control changed-column count, RMS/L2/max, 동시 motor RMS/max, E334held vs E335live command/tendon per-leg 차이를 정해진 row에서 구한다. 임의 가장 유리한 뉴런/창/lag를 선택하지 않는다. Tonic actual−encoded의 source-pinned 벡터를 재확인하고 live observation과 trace의 경계 대응을 검증한다. 차이의 증가를 loop gain 또는 인과 피드백 실패로 해석하지 않는다.

부동소수 exact metadata/ordering/controlIDs/source SHA는 exact, 신경 readout exact, physical clock tolerance2e−8, 기존 scalar 대조 rtol/atol1e−10을 유지한다. Geometric 보각 계산은 기존 source-defined 연산의 rtol/atol1e−10으로 고정한다. 새로운 diagnostic confidence R≥.5는 공식 gate 변경이 아니며 결과 후 다른 cutoff로 바꾸지 않는다.

## 판정과 다음 결정

이번 단계의 성공은 현재 closure와 시계 대조, 고정 지표의 raw 재현, 제한된 해석을 갖춘 **유효한 진단 보고서**다. Biological success/신체 후보 채택/F1–F8 완료를 의미하지 않는다. Root 독립 감사가 같은 원자료에서 명령 진행·실제 tracking·slip 및 input-difference scalar를 다시 계산하고 source/terminal/owned 종료·비용을 닫기 전 다음 후보 설계를 확정하지 않는다.

고정 해석 순서는 (a) 신경 명령이 기존 reference step·범위에 비해 얼마나 진행하는지, (b) 같은 축에서 신체 추적과 실제 foot/knee 운동에서 얼마나 전달되는지, (c) held/live 조건에서 얼마나 달라지는지다. 연속값에 근거한 후속 가설 제안으로 제한하며 '부족/충분/큼'을 자동 분기 임계값으로 쓰지 않는다. 수치만으로 하나를 단독 원인으로 기각/지지하지 않는다. 명령 진행과 reference step의 관계를 토대로 학습 목적의 temporal progress/범위와 실제 신경 모집 전달을 검토한다. Actual tracking/foot/knee 전달 값으로 물리/근육 전달 검토의 필요성을 기술한다. Held/live 차이로 실제 입력을 포함한 학습 모델·폐루프 민감도 가설을 제안할 수 있으나 아직 body Jacobian 검증은 없다. 설명이 혼재하면 새 causal probe를 좁혀 사전 설계한다. Blind theta 재시도와 결과를 통과시키기 위한 threshold 변경은 없다.

## 구현 소유권과 자원

GPT6.1Sol/high가 새 tools/analyze_c_banc_m3_saved_gait_transfer.py 및 tests/test_c_banc_saved_gait_transfer.py 두 파일만 구현한다. Offline NumPy/형상·시계·I/O·gate/lifecycle 연동으로 high를 선택하며 새 CUDA 미분/신경 동역학은 없다. Source/input closure·preregistered configuration·test dimensions는 원본과 fixture 경계를 나눈다. 원본 CLI는 고정 fresh output verification/banc-natural-gait-m3-20260919/e336-saved-gait-transfer, 고정 inputs/dimensions/device CPU다. 소형 test fixture dimension override를 원본에서 허용하지 않는다. Raw diagnosis arrays는 uncompressed NPZ(큰 drive 복제 없음), report/result/advance-ledger/source/output-manifest/receipts로 작성하고 모든 terminal postread/hash/resource check 후 유효성을 결정한다. Save/readback/postpublication 실패는 raw를 보존하고 자격을 철회한다. First error와 절대 clock은 repair에서도 유지한다.

Generated CPU는 Sol/root 각각 실패 포함90초, host2GiB/output64MiB다. Synthetic phase known-motion/constant/uniform ambiguous curve/R threshold/gap/no unwrap·leg offsets·tracking row/knee complement·zero denominator/native weighted slip/drive chunk memory/typed shape와 producer-link/file tamper·late deadline·publication failure를 검사한다. 중요한 실제 source fault case를 independent root helper에서 검사한다. 별도 CUDA 자격은 불필요하며 generated/diagnostic에서 Torch/CuPy/neural/body0을 확인한다.

Saved 원본 진단은 **external40/internal35/prehash+load+calculation+posthash30/final5초**, CPU1/owned host2GiB/output64MiB/free disk1GiB다. 기존 E335 감사가21.9883초(세 시도 포함)였으며 새 selective drive 비교의 시간은 미측정이다. Generated full-size chunk의 선택적 읽기·해제/I/O를 사전 확인하고 한도 초과면 원본 retry/한도 확장 없이 invalid를 보존한다. Fixed s0는 gate전 한 번, science_end=s0+30, final_end=min(s0+35,final_begin+5), supervisor40이다. Root의 독립 saved audit는 실패 포함20초/internal17/science14/final3·host2GiB/output32MiB이며 neural/body/original 진단 재실행0이다.

신경·신체 original 실행은0이지만 **이 저장 원본 수치 진단은 original-analysis1로40초 예약하고 실제 감독 시간을 기존9660초 원본 회계에 한 번만 합산**한다. 원본 cumulative9493.171036399524·잔여166.82896360047562·예약0을 현재 유지하며, root FIRST/R1/SECOND/current generated qualification/static passive gate 뒤에만40을 예약한다. 새 budget·배정 증가·사용자 총 토큰 한도는 만들지 않는다. 다음 후보·body/장시간/F6 실행은 아직 미예약이다.

## 실제 Astra6/max 리뷰 반영 계약 M1–M8

Actual CLI0.159.2 gpt-6-astra/model_reasoning_effort=max/read-only/ephemeral가 exit0 및 turn.completed로373.485초에 종료했다. Review SHA315e0ce730ed7d140ad4c2fae15f083f62185a19499d3ddf7e400a5a075a4924, 원본 초안 SHA4df7f5b48a27b34f53b67403628dc4329ff00f426e932dbf177866dce6103c06이다. Host 호출·완료 증거이고 backend 내부 모델의 독립 attestation은 없다. M1–M8을 모두 수용하며 원본 실행을 승인한 리뷰로 해석하지 않는다.

**M1 입력 schema와 ordering.** Raw clocks/control_ids/*_control_ids는 int64, body_before/after 및 physics_time_s는 f64다. command/q301 또는300×42·tendon×6·actual/targets/tracking42·angles/velocities42·knee6·position3·rotation3×3·support/non_support6는 f64, raw motor_rate32/returned motor391·encoded/actual/held drive158706는 f32다. Chunk은25행×12, 각 NPZ의 field IDs는 chunk control IDs와 exact/누락·중복·재정렬 없이 대응한다. E318 drive chunk clocks는 control_tick1..300/network_tick20c이며 motor_rate key는 f32×391다. Live availability/event는 별도 live-input-events.json의 chunk metadata를 명시적으로 읽어 성공 반환/시계/필수 단일 호출과 대조한다. Native slip_valid/nonfoot은 bool, normal_load_bw/slip/upright/time은 원래 dtype를 NPZ header와 고정 부모 producer descriptor에서 확인해 freeze에 명시한다. 헤더를 보고 임의로 입력을 다른 dtype로 cast하여 허용하지 않는다.

Node ordering은 동일 graph/nodes producer SHA·node count·source model metadata와 기존 E335 once-patched CSR의 indices/indptr exact 증거로 연결한다. MN ordering은 E318/E335 observed-start에서 제한 parent-e335-audit-readers.py의 adapter/calibration/motor_ids leaf(391,int64)만 읽어 exact 비교한다. 다른 whole 수치 leaf는 0이다. 이 두 작은 숫자 leaf의 접근도 original saved 진단 gate 뒤의 수치 read 목록에 기록한다. E334의 native actual-input receipt/descriptor가 E318 full-drive chunk SHA 및 해당 drive row digest/controlIDs/host/device dtype에 대응하는 것을 completed E334 audit v2의 SHA·checks·terminal links와 receipt 원자료에서 확인한다. 단순158706/391 shape만으로 ordering을 가정하지 않는다.

**M2 값 전용·canonical 상수.** Temperature=7n, INC=2.382760007990865×.005, future는 고정128-anchor linear interpolation C_l(s+INC), basis=(cos2pi s,sin2pi s)다. 기존 상수 재구성 tolerance rtol1e−12/atol1e−15로 저장 canonical 배열과 대조한 뒤 **저장 canonical 값**을 전 계산에서 일관되게 사용한다. Encode value-only는 squared-distance→temperature division→max-shift softmax→weight/F/z/R/entropy만 계산한다. Scalar successor=mean_l(mean_edges,j(residual²)/n_l), coordination=mean_edges(mean_all15(1−dot(z_l,z_m)))다. Gradient/Jacobian 출력을 만들지 않고 기존 helper main은 금지한다. Loss 배열의 previous c100..299와 following c101..300을 고정한다.

**M3 부호·좌표.** Trace stored error=target−actual, 새 tracking=actual−target=−stored_error. 무릎 sensor=pi−deg2rad(geometric interior), UI position/basis=S@physical_position/S@physical_rotation이며 위 본문과 같다. Source metadata exact 및 계산값은 기존1e−10 tolerance를 유지한다. Support와 floor normal/adhesion의 정의를 합치지 않는다.

**M4 표본과 창.** 공식 span/tracking/command-vs-actual 비교는 c100..299의 동일200경계, increment 비교는 이 안의 c100→101..c298→299의199edge다. Smooth/phase/static residual diagnostics는 c100..300의201경계·200edge를 사용하고 별도 라벨을 둔다. E334[i]를 c=i+1로 변환한 뒤 같은 control label로 선택한다. Native 공식 mask는 고정 parent-e327-saved-metrics-v2.py SHA b75f80920ba59886b93cf24fc157a3c483c51a92693fa43a3c33ff3055442117의 **저장 실제 time_s>=.5 & time_s<1.5** 식이다. Nominal t=k*.0001이면 k5000..14999/10000표본이며 boundary×50 방식으로 대체하지 않는다. Actual mask의 first/last tick/count와 nominal mask 차이도 기록한다. Clock tolerance2e−8로 허용된 boundary floating jitter 때문에 mask가 다르면 이를 숨기거나 retime하지 않고 동일 기존 evaluator 수식으로 재현하며 차이를 명시한다. Generated fixture는 정확 endpoint와 ±floating boundary 경우를 포함한다.

**M5 공식 물리 수식.** Geometry는 gait.load_model/analyze, 공식 forward/slip/bout/cycle/startup/health는 위 SHA의 pure evaluate(folder) 또는 정확 수식에 따른다. Signed forward=첫 body basis의 XZ forward 단위벡터에 [.5,1.5) 마지막−첫 XZ position을 투영한다. Native support=actual time mask & load>=.02BW; loadfinite 및 모든 support의 slip_valid/finite를 요구하며 전체 slip=sum(load*slip)/sum(load)다. Per-leg도 같은 support/valid/load분모를 쓰고 분모0이면null·샘플0으로 기록한다. 전체 slip은 per-leg 평균이 아니다. Native support fraction은 창 내 load>=.02비율, transition은 이 bool의 adjacent 변경 수이며 step 수가 아니다. Boundary adhesion 및 receptor support는 별도로 기술한다. Brake request가 없는 고정 창임을 확인한다. Gait 자동 sensitivity/exploratory 결과는 descriptive label만 부여하고 공식 판정을 재선택하지 않는다.

**M6 통계 정의.** RMS_l=sqrt(mean_time,j(x_lj²)); normalized RMS_l=RMS_l/sqrt(n_l). Increment 기준은 고정 sqrt(n_l), reference span은 per-joint 128anchor p95−p05다. Command/actual 및 held/live span ratio의 분자·분모를 명시하며 command-vs-actual 전달 비율은 actual span/command span, reference 대비는 각각 span/reference span이다. 분모0은null로 기록하고 epsilon을 넣지 않는다. Percentile은 p05/median/p95로 고정, 독립 반복/모집단CI가 아니다. Valid adjacent R>=.5 두 끝점에서 principal angle atan2(cross(z_prev,z_next),dot)/2pi를 계산하되 정확 반회전(abs(delta)==.5)은 ambiguous/invalid다. Reverse는 valid delta<0의 개수다. Segment별 sum(delta) 및 edge_count*.005초·expected_progress=edge_count*INC를 함께 기록하고 gap을 넘는 진행량은 만들지 않는다. R은 weight concentration이며 reference 근접성/통계 신뢰도가 아니다. Reference 최소 anchor 거리·static 잔차도 같은 행에서 제공하되 거리 cutoff를 추가하지 않는다. MN 차이는 theta와 상태/input을 함께 바꾼 관찰이다.

**M7 f32 산술.** Drive changed count는 저장 f32 element의 exact `!=` 개수, RMS/L2/max는 subtract 전에 f64로 promote 후 node축 mean/sum/max다. 같은 규칙을 motor f32/command f64 차이에 명시한다. Tonic 검증은 encoded f32 copy→source ordered descending_ids에 descending_drive f32 addition→고정 tonic f32 vector의 in-place addition 순서를 재현한 expected actual bytes와 비교한다. `actual−encoded==±5000` exact를 요구하지 않는다. 원래 v3가 source/UID로 입증한20ports/DNg100은 고정 SHA의 audit metadata에서 가져와 source-config/profile와 연결한다. Finite large value에서 f32 rounding이 발생하는 generated fixture를 포함한다.

**M8 전체 작업량·감사.** root는 input/source 전 목록의 bytes·NPZ ZIP compression/method 및 필요한 member의 compressed/uncompressed size를 **header/metadata만** 읽어 고정 inventory를 만든다. Protected F6 registration/path가 closure와 충돌하면 원본 gate를 거부하며 필수 closure를 몰래 제외하지 않는다. 이번 inventory에서 이미 개발에 사용된 입력과 미등록/미사용 F6 상태를 구분하고 미사용 F6 수치·새 초기상태 선택은0이다.

Generated CPU90·host2GiB·output64MiB 안에서 synthetic full25×158706 f32 encoded/actual/held 한 chunk의 ZIP_STORED/ZIP_DEFLATED selective-read 및 해제·value-only201row 계산·순차12read 작업을 평가한다. 동일 synthetic 파일을 반복 읽어 처리하며12개의 대형 duplicate를 쓰지 않는다. CPU1 환경변수(OMP/OPENBLAS/MKL/NUMEXPR 등1) 및 owned supervisor RSS로 제한한다. Root는 actual pinned 파일의 static streaming SHA와 필요한 ZIP member의 **바이트 hash/decompression만**을 사전 byte-I/O probe할 수 있으나 NumPy로 수치 payload를 해석하지 않는다. 그 비용도 root generated90에 포함하고 closed source에 실제 helper/argv/contract/supervisor·보고서SHA를 연결한다. Inventory·I/O·synthetic compute 각각을 구분하며 원본40/감사20 완료의 보증이라고 주장하지 않는다. 원본 clock이 초과하면 invalid/재시도0이며 audit 실패 포함20초 내 끝나지 않으면 qualification withheld다.

독립 saved 감사는 producer scalar를 계산 입력으로 사용하지 않는다. Raw에서 세 trajectory의 value-only smooth/phase/step/static residual·두 body의 same-window actual span/tracking·공식 native slip/signedforward/bout/cycles·12chunk drive RMS/L2/max/changed-count·MN readout/tonic/경계 정렬의 핵심을 별도 root 수식으로 재계산하고, terminal/source/input SHA 및 original40/cost/owned 종료를 확인한다. 같은 원자료와 수학 정의에 대한 구현 독립성만 주장하며 독립 생물학적 표본이 아니다. 실패 자료·과거 verdict를 보존하고 최신 유효 producer의 exact schema/allchecks/links와 혼동하지 않는다.
