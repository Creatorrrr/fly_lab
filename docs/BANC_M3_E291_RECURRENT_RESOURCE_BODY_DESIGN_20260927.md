# E291 — 고정 재귀 전달 자원 후보의 실제 신체 판별

2026-09-27 root 설계. E289/E290의 구현·tiny 수치 검증이 승인된 뒤 별도 동결·예약하여 수행할 실제 BANC–신체 진단이다. 현재 원본 누적7869.382029599514/7925초, 예약0. 이 문서와 생성 테스트는 원본 실행 허가 gate를 대체하지 않는다. 사용자 원래 자연 보행 F1–F8와 기존 모든 실패를 유지한다.

## 한 후보와 판정

E150 원래 neutral body/adapter/network checkpoint를 기준으로, 정적 직접 motor-presynaptic VNC intrinsic population7136 안의 기존 nonzero mutual edges725849에만 E289 source-resource를 적용한다. 원래 CSR의 current bytes/부호/선택 규칙과 motor391을 확인한다. 원래 E29 weights를 대체 사용하지 않는다. roster는 `verification/banc-natural-gait-m3-20260919/e289-recurrent-resource-roster.npz`, SHA256 `1a758419034490ae20070ebe4d82dc8797241c065fb6c87f9b1ba6f6d4c6abd2`다. Unique source7136에 공유 resource 하나씩이며 source R=1, tau=.150초, candidate u=.05 / neutral u=0로 고정한다. Decoder/source-filter/CPG/입력 교사/새 간선/신체·물리 설정 수정이나 fit/grid는 없다.

1. Neutral 새 layer를 원래 tick0 checkpoint에서 만들고 .005초(1control/20neural/50physics)만 실행한다. 기존 E150 trace2/native50/recruitment2와 exact 비교한다. 마지막 whole state도 E279 저장 neutral/final에서 `remove_source_layer`한 원래 whole와, 새 state에서 `remove_resource_layer`한 whole를 비교한다. StateStore의 tuple/list·numpy scalar 표현만 명시적으로 정규화하며 ndarray dtype/bytes/과학 상태의 숫자 오차 허용은 없다. 독립 실행의 성능 측정값인 최상위 `wall_s`만 별도 관측 비용으로 보존하고 endpoint scientific equality에서 제외한다(아래 FIRST 정정 근거). 초기 상태·동일 저장 상태 복원은 wall_s도 exact다. Resource는 전부 정확히1이다. 통과해야 candidate를 생성한다.
2. 같은 original neutral에서 candidate를 시작해100controls=.5초. 기존 strict startup(q controls1..99 RMS≤.018/peak≤.04, LTM RMS≤.009/peak≤.025 및 원래 native health)를 그대로 적용한다. 음성이면 실제 기각으로 종료하고 raw/whole 또는 독립 partial checkpoints를 저장한다. 조건부 q 예측이 없으므로 prediction 산출·모조 prediction 입력은 없다.
3. Startup 통과시 새 세션 없이 같은 session300controls=1.5초까지 연속 실행한다. `verify_c_banc_m3_source_decoder_body.assess_short`의 승인된 동일 기준을 사용한다. [.5,1.5) 실제 전진≥1.6790085779712631mm, 실제 clock/finite/upright≥.9/nonfoot0, slip≤E257 baseline1.8026436968249955mm/s이고 support유효/누락0, eligible bout≥.3초, 여섯 다리 모두 eligible 완전주기>baseline0을 요구한다. 각 주기는 aep<pep<next_aep로 유효 bout 안에 있어야 하며 접촉전환 횟수로 대체하지 않는다. .3 비교는 기존 고정sample격자(samplecount−1)*.005로 한다.

양성일 때만 같은 parameter/roster/decoder의 별도10초/F2/F6/F7/F8로 확장한다. 본 단계 natural_gait_complete/F1_F8_complete/F2_complete/F6_exposed/adopted는 항상false. 기각시 같은 후보의 startup기준·u/tau·seed를 결과에 맞춰 바꾸거나 반복하지 않는다. 추가 구조·메커니즘 판단은 관측 결과로 다음 설계에서 한다.

## 구현 소유권과 API

실제 gpt-6-sol/high가 새 `tools/verify_c_banc_m3_recurrent_resource_body.py`와 `tests/test_c_banc_recurrent_resource_body.py` 두 파일만 구현한다. root FIRST→같은 Sol 첫수정→root SECOND→root 잔여수정 절차를 따른다. 기존 E279/E288/E289/E290와 모든 frozen 파일은 읽기 전용이다. 동일 process에서 그 module globals를 monkeypatch하지 않는다.

`gate(freeze_path,freeze_sha,review_path,review_sha,output)`, `NativeDependencies`, 자체 `run_arm`, `execute`, CLI를 제공한다. E279의 observer/clock/journal/short raw/cleanup 및 E288 assess_short는 그대로 읽기 전용 재사용 가능하다. 그러나 E279 `run_arm`의 source_history51×2 및 conditional prediction 부분은 사용할 수 없다. 자체 run_arm은 새 state와 실제 gate만 다룬다. E290 wrapper API는 `RecurrentResourceTrial(base,edge_ids,premotor_ids,utilization,recovery_tau)`, `remove_resource_layer`/`checked_checkpoint`다.

관측은 control0 포함 매5ms rows/원래 recruitment/원래 motor391와 premotor7136의 sorted union rate/drive/adaptation 및 R7136를 저장한다. R nativef64/finite/[0,1], rate/drive/adaptation nativef32, tick/physicalclock정확성을 확인한다. 완성되지 않은 readout도 completefalse와 이미 받은 각 array를 보존한다. Body native100µs contact/load/feet/nonfoot 및 full whole state/adapter held controls/clock/sensory/monotonebank/신경metadata·resource를 저장한다. R 변화량/min/max는 기술통계일 뿐 선택기준이나 생리 판정이 아니다.

Neutral whole 비교에는 session endpoint가 필요하므로 raw metadata에 마지막 검증 snapshot을 넘기거나 deps가 session을 보유한 동안 수행하되 종료 후 GPU state를 숨겨 보유하지 않는다. 최대 live whole/body1, 전체 body constructors2. 각각의 returned session은 ledger/observer publication이 실패해도 holder로 회수하여 exactly once close한다. 실패시 첫 예외·후속 cleanup/부분저장 오류·실제 진행 clock·setup/settle를 보존한다. 원본 body는 생성검증에서 호출하지 않는다.

## 동결과 비용

새 freeze/review schema `flylab.m3.e291.recurrent-resource-body.freeze.v1` / `.review.v1`. OWNED 두 파일, SOURCE_FILES에 실제 inherited 실행 경로를 포함한다. Paths는 graph_bundle,neutral_checkpoint,baseline_trace,baseline_native,baseline_recruitment,baseline_short_result,baseline_short_trace,baseline_short_native,startup,selection,baseline_neutral_final_checkpoint. E287 candidate/selection/predictions 경로는 없다. 새 freeze는 exact original graph/neural/adapter identity, computed neutral/candidate network identity, selected counts/roster, 고정 u/tau, source/input/package/output/설계/기존 M2 보존 SHA를 모두 핀한다. E289/E290 최종 root source review와 root tiny qualification receipt(approvedtrue, 대상 sourceSHA 일치)가 있어야 original gate 통과한다. Invalid/missing/mutated roster/review/input/output는 body 생성 전에 fail-closed한다.

원본1/retry0, scientific neural≤6020ticks/physics≤15050steps, fullbodyconstructors≤2, science≤85초+finalization≤30초/internal115초/supervisor125초, hostpeak≤6GiB/addGPU≤4GiB/output≤2GiB/freeDisk≥4GiB, BLASthread1. Setup/capture/warmup/settle는 실제 ledger로 따로 기록하며 scientific clock과 합치지 않는다. 실제 비용·실패를 초기화하지 않는다. 전체 원본 잔여에 맞는 prospective 단계 배정·예약은 root가 소스/tiny review 후 실행 전에 문서화한다.

구현 생성검증은 CPU≤90초, fake5ms/100µs session과 작은metadata/원본0/CUDA0/body0/solver0. Hash/path/package gate 변조, neutralprefix와 wholeendpoint 실패시 candidate0, control100 startup실패시 continuation0, 동일session300및short6legcycle 기준, Rshape/dtype/bounds, closed ownership/recoveredreturn/finalization/partialcheckpoint/memory·time 실패를 시험한다. 생물학적/자연보행 성공으로 표시하는 출력은 허용하지 않는다.

## FIRST 리뷰 중 설계 정정 — 독립 실행의 관측 wall 비용

원본 개시 전 root 코드 리뷰에서 `BancWalkingSession.advance`가 실제 `perf_counter()` 경과를 `wall_s`에 누적한다는 사실을 확인했다. FIRST는 root 초기 설계의 모든 endpoint 필드 equality를 따랐지만, 독립적으로 실행한 neutral 두 개의 관측 wall 비용까지 같은 값으로 요구할 수 없다. Root passive checkpoint probe는 E150/E279 실저장 trace/native/recruitment/whole metadata와 새 neutral resource 변환이 모두 exact인 것을 확인한 뒤 wall_s 하나만 다른 경우의 거절을 재현했다. 원본/신체/CUDA/solver0이고 이 FIRST 실패와19.061947초 감독 비용을 보존한다.

같은 Sol 첫 수정에서 endpoint 비교의 top-level wall_s만 두 개의 유한 비음수 Python float 관측값으로 별도 기록한다. 원본 시작 wall보다 작으면 거절하며, 해당 필드를 뺀 모든 전체 과학 state/parameter/clock/array는 exact다. 실제 snapshot·checkpoint·비용의 wall 값은 수정하거나 맞추지 않는다. 반환 receipt는 `excluded_fields:["wall_s"]`와 양쪽 관측값/비교범위를 명시한다. 초기 whole 비교 및 E290 동일 저장 상태 restore의 wall_s는 계속 exact다. 이는 원본 결과를 보고 보행 기준을 완화한 변경이 아니며 F1–F8·startup·short·신경/신체 시간 및 원본 비용 한도는 불변이다.
