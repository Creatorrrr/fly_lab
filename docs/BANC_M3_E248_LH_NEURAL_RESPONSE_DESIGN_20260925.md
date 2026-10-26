# E248 LH 감각 입력과 기존 내부 출력의 신경 반응 대조 — 2026-09-25

M3/F1–F8의 다음 한정된 진단이다. E247은 실제 두 단계 경로의 존재를 확인했고 E246에서 LH MN 상태 표현이 미달했다. E248은 신체를 생성하지 않은 전체 BANC에서 감각 입력을 바꾸었을 때 LH MN 반응이 생기며, 선택된 내부 출력이 그 반응에 관여하는지 확인한다. 모델의 자율운동 생성, 주기성, 몸의 자연보행, 특정 단일 간선의 인과성은 판정하지 않는다.

## H248와 고정 대조

현재 E150/E245 전체158,706뉴런,11,584,852개 선언CSR,기존계수/마스크/적응/타임스케일을 보존한다. E245 교사 최종 network tick40000의 전체 rate/adaptation/drive/mask로 각 arm을 복원한다. 전체 drive는 저장된 마지막 held drive를 바탕으로 해당 감각포트만 교체한다. DNg100 tonic 및 다른 감각/음수 입력을 보존한다. 관측 신체를 재생하거나 teacher command를 출력하지 않는다.

E247의 LH 포트 중 기존 유효 dynamic 경로로 가장 많은 서로 다른 free LH MN에 닿는 포트를 선택하며 동률은 원래 port ID 최소다. 이는 경로 구조만에 의한 개발 선택이다. 선택 결과는 port85(claw), 원래 감각23뉴런, 내부 bridge4뉴런[20156, 63026, 111970, 152280], 도달free LH MN55개다. 대조 범위는 실제56개 free LH MN 전체다. 입력low/high는 E247 steady TRAIN100..1200의 port85 실제 old_filtered min/max 8.758351424802523/13.260401038721753를 float32로 변환한 값이며, 포트의 실제23개 뉴런 모두에 동일주입한다. 외부 입력은 postsynaptic drive이고 CSR에 들어가기 전 rate에 합산되는 값이 아니다.

고정6 arms 순서: intact_low, intact_high, bridge_cut_low, bridge_cut_high, sensory_cut_low, sensory_cut_high. 모든 arm은 같은 saved network에서 시작하여40controls×20ticks=200ms 진행한다. bridge cut은 위4개 원래뉴런의 모든 outgoing contribution을0으로 만들며, sensory cut은 해당23개 뉴런의 outgoing을0으로 만든다. 기존0마스크와 합집합을 사용한다. rate를 강제0으로 하거나 CSR 간선/weight를 변경하지 않는다. bridge 개입은 해당 뉴런의 전체 outgoing 절단이므로 두 특정 간선의 개별 인과성으로 해석하지 않는다.

주판정: 56개 free LH MN의 rate/cap(원래cap200)을 controls1..40에 대해 사용한다. delta_intact=high−low, delta_bridge=bridge_high−bridge_low. R=RMS(delta_intact/cap), D=RMS((delta_intact−delta_bridge)/cap). H248 지지는 R≥1e−4 AND D≥1e−4 AND D≥0.2R이다. 반응 존재와 내부 의존 차이를 각각 보고한다. D는 상호작용 크기이며 반드시 차단에 따른 감쇠가 아니다. 조건미달은 이작동점/포트/200ms 대조에서만미달이며, 경로의존재나장시간보행학습불가능으로일반화하지않는다. 주요경쟁설명은 입력/내부경로포화, 직접우회/다른내부경로, 선택한작동점의작은반응이다. 현재고정시험으로각원인을단독확정하지않는다.

유효성: sensory_cut high/low에서 모든비해당감각뉴런의 full rate가 각41행 모두 bitexact해야 한다. 이검사는 입력차이의outgoing전달경계를확인하며MN만보지않는다. 해당23개 감각뉴런의 입력/발화차이는 별도정량보고하고 H기준을바꾸지않는다. 모든 rate/adaptation finite와유효영역,모든실제mask/drive,전체CSR/parameter/staticidentity,원래metadata exact를요구한다. 오류/제한초과는invalid/not_tested, 완료된raw를보존한다.

## 실제 신경망 복원과 독립 미래 대조

신체없는 MotorTimescaleNetwork만 사용한다. 가중치는E150 actual-global-csr의정확한float32CSR다. membrane tau는원래assembled-calibration의tau_ids/values로.02배열에패치하고E150 actual membrane_tau와exact검사한다. a1/threshold7.5/cap200, adaptation_gain8/tau.3/dt.00025/CUDA packed/capture_steps20, core/motor/timescale은저장patch의원래IDs/계수다. network/model및각layeridentity와staticgain/tau/inverse를검사하고baseline복원snapshot전체exact를요구한다. 기본/실험Trial 또는FlyBody 생성금지다.

최초모델에서baseline neural snapshot을기존StateStore로새 neural-start에저장한다. 각arm의초기동적상태와mask/drive는실제확인한다. full rate를41행전부,최종fullsnapshot을arm별저장한다. 별도의두번째동일모델은신규disk neural-start에서복원하고명시적으로같은intact_low full-drive schedule을40controls진행한다. 매full rate와최종fullsnapshot이첫모델intact_low와bitexact여야한다. 이는새로정한held-input 미래의일치이며과거E245의미래를재현했다고부르지않는다. 새disk정체/복원검사실패시다른arm결과도채택하지않는다.

생성자는모델당60setup CUDA ticks(두warmup+한capture)를수행한다. 이를과학적tick과분리해2constructors/120setup/7×800=5600scientificticks로계상한다. 각operation의requested/entered/completed와unknown_partial을진입전에내구기록하며실패를0work로바꾸지않는다. 한원본프로세스에서순차진행하고원본재시도0이다.

## 구현, 검토, 자원 및 결과 결정

부모설계→실제gpt-6-sol/high 구현→부모 FIRST→필요시같은Sol첫수정→부모SECOND→잔여문제부모수정. 구현소유파일은tools/probe_c_banc_m3_lh_neural_response.py와tests/test_c_banc_lh_neural_response.py 두개다. 기존production/model파일변경금지,새일반프레임워크금지,기존StateStore/constructors/supervisor패턴재사용. 생성fixture tests로frozen선택/실제mask합집합/metric/차단검사/실패결과무효화/작업ledger를검사하며원본/GPU모델은승인된single original에서만구성한다.

기존누적6740.33810969962/6800초,잔여59.66189030038004초에서supervisor55초를예약하며추가배정0이다. internal44초(최종저장6초포함),external50초,supervisor55초,CPU1/GPU1,hostRSS6GiB/추가GPU6GiB/output384MiB/free disk2GiB/free GPU7GiB. 본래2model/6arms/1reference를넘는작업,solver/fit/신체/physics/F6/UI0. 출처조사미측정시간null은0비용이아니다. topology선택준비시간은별도기록한다. 예산을넘으면원본을중단하고완료된결과/부분raw와실제비용을보존한다.

출력verification/banc-natural-gait-m3-20260919/e248-lh-neural-response에config/selection/provenance,각arm full rate/raw 및baseline/end snapshots,metrics/result/ledger/errors/manifest를보존한다. source review가freeze/source/inputs/package를통과하기전원본arrays/model읽기금지다. 부모는저장raw에서모든metric/차단/미래/상태를별도검산한다.

유효하고H248지지면현재실제경로를유지한채LH MN동작범위/적응/기존synapse gain의최소개입을다음설계로고른다. R미달이면입력범위/각층반응의포화를먼저분석하며바로body rollout/fit하지않는다. R통과D미달이면이번4뉴런을유일전달회로라고부르지않고기존우회경로를검토한다. 어느경우도E150/M2교체/F2완료/자연보행목표완료를선언하지않는다. F1–F8최종계약과F6미노출을보존한다.
