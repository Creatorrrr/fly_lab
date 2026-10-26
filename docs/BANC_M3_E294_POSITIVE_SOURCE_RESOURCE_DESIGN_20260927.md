# E294 — 억제성 전달 가중치를 보존한 고정 재귀 자원 후보

Root설계,2026-09-27. E292 모든부호 resource 후보는0.3124초 자세기준 실패로 기각됐다. E293 동일저장rate 분해에서0.05/.10초 억제완화/흥분손실의 크기비는1.9078/1.8969, 순양변화 post비율77.13%/75.08%였다. Root 독립 sparse matrix 재계산은63×7136 모든성분에서 최대오차1.3643e−12로 확인했다. 이는 국소 전달 항 변화이며 미래반사실·보행인과증거가 아니다. 고정부호를 구별하지 않은 약화가 억제를 크게 낮췄으므로 다음은 그 구조를 구별하는 한 후보다. 기존 u/tau값의작은grid나실패후seed선택은 하지 않는다.

## 한 개 고정 후보와 과학 판정

E289 원래725849 edge/7136 source/7136 premotor/391 motor 및 originalCSR bytes/위치를 전부 유지한다. Source별 선택outgoing weight에 양수만 있으면 u=.05, 음수가 하나라도 있으면 u=0으로 한다. 따라서 mixed-sign source도 u=0이며 데이터에서 부호가혼합되지않는다고추정하지않는다. tau=.150은 전부동일, R초기1. Neutral은 전source u=0. 기존 E292와 바뀌는 과학parameter는 이 정적 부호 규칙에 따른 utilization vector 하나뿐이다. Source ID는 descriptive ranking, activity, leg, 시간, 보행결과로 고르지 않는다.

u=0 source의 R은정확히1이므로 선택된 모든음수edge는 원래 CSR 연산식을 유지한다. 이 말은 **같은 시점rate에 대한 전달가중치**가 바뀌지않는다는 뜻이다. 신경상태가달라지면서 억제신경rate자체가달라질수있으므로 실제미래 억제current까지baseline과같다고주장하지않는다. 양수source의 dynamic 자원만 달라지며 kernel/RK4/decoder/body/물리/inputs는 기존승인E289/E290/E291을재사용한다. 공학적가설이며 BANC에서측정한short-termplasticity가아니다.

신체 판정은 E292와같다: 원래E150 neutral .005 whole scientific+trace/native/recruitment exact→.5초 strict startup(q RMS≤.018/peak≤.04,LTM≤.009/.025,nativehealth)→통과하면같은session1.5초. 원래 short forward/slip/bout/6legcycle기준과 F1–F8/M2/UI/F6미노출을보존한다. 동일후보10초는short통과후별도예약한다. 미통과는기각이며u/tau/기준/seed변경재시도없다. 원본1/retry0,science85+final30/internal115/super125초,body≤2/live1,scientific neural≤6020/physics≤15050,host6GiB/addGPU4GiB/output2GiB/freeDisk4GiB/CPUthread1.

## 구현과검토

실제 gpt-6-sol/high가 신규 `tools/verify_c_banc_m3_positive_resource_body.py`와 `tests/test_c_banc_positive_resource_body.py`두파일만구현한다. 기존소스/데이터/docs는읽기전용이다. Root FIRST→필요시같은Sol R1→root SECOND→root잔여수정 순서를적용한다.

`positive_source_selection(base_roster)`는 원래E289roster의 exact키와topology/typedmetadata를검증한뒤 deepcopy하여 utilization만위규칙으로교체한다. CPU 작은회로에서 mixedsource는u0,음수edge원래expression/R1,양수전용만u.05,neutral와input불변을검증한다. 기존tinyCUDA는mixedu0/nonzero를이미승인했으며 kernel수정없다. 현재E289/E290root tiny/source review SHA가같아야한다.

NativeDependencies는E291의 승인된construct/initial/capture/compare_neutral/run_arm/execute를읽기전용으로재사용한다. Monkeypatch/moduleglobals변경금지. 새로운`load`는 **literal frozen E292 baseline provenance**로E291.NativeDependencies.load(base_freeze)를호출해원래입력·checkpoint/graph를검증한뒤,별도동결된새selection과계산한identity를검증하여 data.selection/roster/freeze의candidate metadata만명시적으로교체한다. Baseline load를새candidate를검증한것처럼표시하지않는다. 실제construct는새data의vector를반드시사용한다. 같은7136source를유지해inheritedRshape검증과API는바뀌지않는다.

Gate는새freeze/review schema `flylab.m3.e294.positive-resource-body.freeze.v1`/`.review.v1`,exactsource/input/package/design/output/limits/criteria를확인한다. Root가E292 base_freeze/base_review/base_original_audit(spec path+sha256)를핀하고,기존원본graph/body/decoder/neuralbaseline/input/preservationSHA와 source/tinyapproval lineage를그대로검증한다. Base 원본이유효기각이고새selection이base에서정적으로계산한bytes와exact인지확인한다. Source identity는utilizationvector와모든metadata로계산하며neutral은같고candidate는새identity다. `source-manifest.json`은새runner의OWNED와실제재사용source를정확히표시한다(필요하면deps.write에서새dict로해당manifest를교체;원래moduleglobals수정안함). Runtime환경·CPUthread·GPU추가메모리·cleanup은기존E291동일기능을유지한다.

Runner는ROOT,OWNED,SOURCE_FILES,LIMITS,STARTUP_LIMITS,FREEZE_SCHEMA,REVIEW_SCHEMA,`gate`/`NativeDependencies`/`execute`또는runCLI를노출하고root에게freeze필드/입력alias를명시한다. Input 목록에는원래E292모든입력(원래roster포함)과새selection이모두필요하다. output은새`e294-positive-resource-body`다. E291결과schema를재사용해도diagnostic_id는E294이고모든최종보행승격flag는false다.

생성검증max60초/2GiB,실제original/body/CUDA/fit/solver0. 정책tiny손계산·mixed/negative preservation·잘못된source/utilvector/roster/hashes거절·실제construct선택vector전달·neutral실패후candidate0·startup실패후continuation0·동일session300분기·manifest·cleanup을핵심으로한다. Root source검토후현재단계원본잔여77.535825100491초를이월하고필요한125초예약을실행전에기록한다. 이전원본누적7967.464174899509초와모든실패를보존한다.
