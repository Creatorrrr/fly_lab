# E121: 목표 입력 포트의 64배 억제 증폭과 실제 보행의 간섭

2026-09-20 · 부모 설계 · Sol/xhigh 개발/첫 수정 · 부모 재리뷰/잔여 직접 수정

## 방향 선택의 근거

E120은 원래 두 기록 문맥과 작은 가중치 범위에서 초기 관절 제약이 형태 개선 방향을 크게 제한함을 확인했다. 그 뒤 기존 신경 상태를 이용한 시간 응답 가설을 검토했지만, 현재 코드와 실제 전체 상태에서 더 직접적인 개입 대상이 확인됐다.

`BancTargetEncoder.calibrate_weights`는 목표 입력용20개 GABA 포트에서 기존 다리 운동신경으로 가는872개 연결을64배 증폭한다. 382/391 MN이 이 투사를 받는다. 이 포트들은 기존 모집 적합의49source에 들어 있지 않다. 목표 cue cut 시에도 encoder는 포트에-5000, DNg100에5000의 tonic drive를 주며 회전/정지 요청만 제거한다. 따라서 목표가 없을 때 포트가 반드시 침묵한다는 보장은 없다.

부모의 저장 전체 상태 감사에서 E110의1.1초에는 회전/정지 요청 없이 IN19A003 index83877(root720575941541544028)가129.63684로 발화했고, 포트→MN 억제 전류 최대치는81391.1976이었다. E113의 같은 체크포인트에서는 이 포트가 침묵했다. 기존 입력 차이 분석과 일치한다. 이 차이가 보행 실패의 원인인지, 같은 run의 다른 시각에도 나타나는지는 아직 검증하지 않았다. 관찰과 인과 개입을 구분한다.

정적 감사 첫 실행은 `no_calibration_overlap`라는 예상이 틀려 오류로 남겼다. 실제로872개 모두130943-entry calibration에 이미 등록되어 있으며, 각 values와 source_values는 원래64배 값 그대로였다. 저장 raw만으로 이를 확인하고 별도 overlap-addendum을 남겼다. 실패 결과를 삭제/재실행하지 않았고 모델 전진은0이다. 따라서 새 edge 추가나 encoder 전역 수정 없이 해당872개 cal.values를 원래 증폭 전 값으로 바꾸는 독립적인 가설을 만들 수 있다.

성체 다리 MN의 모집과 고유 감각 입력이 균일하지 않다는 근거는 [Azevedo et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7347388/)에 있다. 선택적 억제가 운동 pool의 지연 모집에 관여한 직접 개입은 [Zwart et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4980426/)의 유충 연구다. 이들은 여기의20개 성체 포트나64배 계수를 검증한 자료가 아니다. 특히 새로운 양의 활동 피드백을 성체 MN의 확정된 생리로 가정할 근거가 충분하지 않아, 먼저 기존 공학적 증폭을 제거하는 한 번의 실제 개입을 선택한다.

## 고정 개입과 불변 조건

기준선은 이미 검증한 실제 E113 전체10초다. 같은 E113 calibration과 원래 E32 whole-zero/body/world에서 출발한다. 유일한 변경은 `BancTargetEncoder`에 선언된20개 포트에서 기존391MN 중382개로 가는872개 edge 값이다. 정확히 E113 float32값/64, 즉 `rate_weights(graph)`의 증폭 전 같은 signed float32값으로 바꾼다. 64는2의 거듭제곱이므로 저장 fixture와 exact float32 일치를 요구한다. 세포/edge를 결과에 따라 선택하지 않고 전체 선언20포트를 사용한다.

새 runner는 graph와 encoder/원래 calibration에서 이872개를 독립적으로 찾아 적용한 뒤 부모 `parent-e121-expected-calibration.npz`와 전 필드를 비교한다. `edge_ids`는 이미 모두 등록되어 있어 추가/삭제하지 않는다. `source_values`는 기존 calibration의 출처인64배 값을 그대로 유지하고, 변경된 `values`의 factor1/64와 증폭 전 base-weight를 별도 patch provenance로 기록한다. 이것을 실제 생체 synaptic strength로 표기하지 않는다.

선택된1572개 source49→391MN 모집 가중치/그 efficacy metadata, 나머지130071개 등록 values, 나머지 전체 CSR, 모든 node/edge/sign, 모든 source 수신 row, 38tau/full158706 실제tau, adaptation/gain/threshold/a/cap, muscle pool/moment/rest/activation/adhesion/tendon, drive/cut/목표 cue 설정, 원래 whole-zero 및 신체·지형을 유지한다. 제품 M2/UI/서버/기존 verifier와 기존 source는 수정하지 않는다. 외부 CPG/phase/trajectory/time ramp/root/pose 보정과 prewarm은 금지한다. 실제 목표 회귀는 다음 단계이며 이 진단에서 목표 포트의 drive나 navigation 기준을 보정하지 않는다.

`parent-e121-port-patch.npz`는872개 graph edge/calibration position/port20/pre/post/before/after를 명시한다. 부모 expected fixture를 새 runner가 그대로 복사한 것만으로 구현 검증을 대신하지 않는다. 기존 `rate_weights`, encoder64 및 E113 cal로 before/after 전체CSR을 다시 만들어, 변경 범위/부호/원래weight 및 부모 fixture와 비교한다.

## 실제 실행과 관측

새 단일 runner `tools/verify_c_banc_m3_e121_port_interference.py`와 `tests/test_c_banc_port_interference.py`만 Sol이 구현한다. CPU calibration 검토와 failure/toy tests 후 부모 최종 source 리뷰를 거친다. 결합된 전체 BANC/CUDA 신체·snapshot·미세단계 관측·실패 회계 작업이므로 기존 Sol/xhigh를 사용한다. 첫 리뷰의 첫 수정은 같은 Sol, 두 번째 리뷰 후 잔여 수정은 부모다.

원래 E113의 검증된 `prepare_neutral_checkpoint`, `AdvanceLedger`, native observer/contact arrays, `body_quality`, `supported_slip`, `observation_ids`, source diagnostics 등 범용 helper를 수정 없이 재사용할 수 있다. E113 main/run_body/하드코딩된 source 준비 판정·비교·resource screen은 E121 실행으로 가장하여 호출하지 않는다. 새 coordinator와 output metadata는 실제 E121로 명명한다. 모델을 만든 즉시 cleanup 책임을 확보하며 prepare/main/clone 실패 시 모두 close하고 원래 예외 및 각 cleanup 오류를 보존한다.

실제 준비는 parent fixture를 만족하는 새 neutral을 만든 뒤 전체 sparseW/tau 및 원래 E32/E113의 body/world/rate/adaptation/drive/output_mask/tick을 exact 검증한다. whole-neutral disk 저장/복원 exact 뒤에만 본 실험을 시작한다. 생성 과정 CUDA graph warmup은 zero/원래 state 복원 증거를 남기며 실험 clock advance로 오기하지 않는다. 해시가 달라질 수밖에 없는 model/calibration identity 외 실제 상태가 같아야 한다.

원래 full158706 BANC/CUDA와 실제 FlyBody에서 한 main10초/2000controls만 실행한다. control마다 기존20 neural steps(250us)와50 physical steps(100us), native100000개를 요구한다. 1초 checkpoint에서 main을 멈추고 clone20controls/.1초(1000native)를 한 번 진행한 뒤 닫는다. main1.1초와 clone1.1초의 whole state가 exact여야 한다. 총 actual neural/physical 각각 최대10.1초다. 다른 seed/포즈/terrain/후보, F6 두 미사용 자세는 노출하지 않는다. baseline actual run은 재실행하지 않는다.

기존 E113과 동일한 basic442/detailed446/metadata55 및391MN/96muscle time traces를 같은 ID/순서로 저장한다. 여기에 선언20개 target port를 별도 추가 관측한다. 포트의 모든 실제 rate/recurrent_current/drive/adaptation, root/cell type/nt/port order,20→391 static actual weights, target cut/turn/braking를 저장한다. detailed에 이미 포함된 포트가 있어도 포트 전용 배열은20개 전체 순서로 유지하고 별도의 neuronal intervention을 가하지 않는다. 기존 매control `_current` 한 번의 반환을 상세/포트 관측에 함께 사용하여 불필요한 전체 CSR 계산을 추가하지 않는다. t0도 저장하며 warmup이나 실패한 control을 완료 sample로 취급하지 않는다.

원래 native health guard(비발 접촉/비유한 상태/upright<.9/fault)를 최초100us에서 적용한다. health stop은 측정된 실패로 prefix/시계/원자료를 보존하며, F 수치 미달이나 source screen 미달로 임의 조기정지하지 않는다. 오류/한도/관측·저장 실패는 automatic retry 없이 종료한다. advance의 attempted/before/after clocks와 actual neural/physics 비용을 main/clone별로 구분하고 반환·raw저장 여부를 보고한다. 필수 start receipt 실패 시 실제 advance0, 이미 진행한 후의 receipt/host 전송 오류에서는 진행량과 가능한 raw를 복구한다. sample 수로 실제 시간이 같다고 가정하지 않는다.

## 판정과 증거

raw trace/포트/MN/native contacts/clock/checkpoint를 수치 assertion보다 먼저 저장한다. 원래10초/.5–10초/F3–F5 evaluator와 E113의 판정 함수를 수정 없이 사용한다. F3는 초기방향 전진10mm, net/path.5, native 건강이며 F4/F5의 실제 Thorax-local forward bouts/완전주기10개·좌우 반주기·R·무릎폭·CV는 기존 그대로다. 원래99 startup q/LTM, supported slip, 여섯 source cadence/nineLR 및 실제 LF source 준비 screen도 별도로 보고한다. E119 선택점이나 E112b 조건부 prediction을 이번 변경의 예측인 것처럼 재사용하지 않는다.

원래 E113 실제 trace/startup/native/slip와 같은 window에서 비교한다. 모든 F3–F5 통과 +원래99 startup +wholefuture exact +health/resource를 만족해야 `supports_next_final_candidate_validation`이다. 일부 개선은 partial이며, criteria가 미달이면 gait improvement 채택을 하지 않는다. F1–F8 전체 completion은 항상 별도이고 이 runner는 candidate_adopted=false/F6_exposed=false를 유지한다. F3–F5가 모두 미달해도 원자료와 포트 시간 패턴이 이후 가설에 필요한 경우 그 결론을 한정하여 기록한다.

포트 기여는 실제 저장 port_rate와 actual20→391 signedW로 계산한다. 활동 시점과 MN/관절/전진 중단의 동시 관계, 실제 회전/정지 요청 여부, before64배 weight를 같은 관측 rate에 적용한 산술값을 별도로 저장한다. 마지막 값은 추가 신경 응답이나 변화 전 trajectory의 예측이 아니다. E113의 같은20포트 전체 시계열은 없으므로 없는 기준선 데이터를 만들어 비교하지 않는다. E1101.1초의 큰 억제 관찰은 개입의 동기이며 E113 실패 원인의 확정 증거가 아니다.

부모는 모든 원본 SHA,872개 변경과 전 calibration/전체W/상태 불변, 원래 F3–F5/99/slip 수치, 시계/실행 수, 완전한 저장 미래, 포트 전류 대수 및 M2/E110–E120 보존을 독립 검산한다. 원래 신경·물리 재실행은 하지 않는다. 실제 개선 후보가 생긴 경우에만 원래 F6/F7/F8 및 제품 통합으로 진행하고, 판정 기준·실패/노출을 바꾸지 않는다.

## 자원과 기록

하나의 GPU, native CUDA/기존 packed 구현/동일float32 방정식, OMP/MKL/OPENBLAS/Torch CPU thread1. 준비45초, body150초, 내부 전체195초, 외부210초(child205초). 본 run의 새 control 시작은 body/전체 deadline의8초 전까지로 제한하여 raw/정리 시간을 남긴다. 준비·최종 결과/cleanup 저장까지 시간에 포함한다. main+clone actual 각10.1초/총2020controls 상한, original fit/LP/조건부응답0, 한 프로세스/재실행0이다.

E120 완료 누적777.305040900초를778로 올려 새 shared ceiling988초를 사용한다. 과거 한도는 소급 변경하지 않는다. 별도 parent 정적설계 감사의 오류 포함 외부4.791472초(내부3.876603초)와 저장raw addendum은 모델 없는 설계 비용으로 따로 기록한다. 출력 `e121-port-interference/`가 이미 있으면 거부한다. 모든 stdout/stderr/exit/timeout/원본 manifest/source hash와 실패 원자료를 보존한다. 현재 계획만으로 실행하지 않으며, 입력/source 동결과 부모 최종 source review가 실제 한 번의 실행의 마지막 조건이다.
