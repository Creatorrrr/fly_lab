# E295 — 저장된 핵심 회로 변화와 전달 손실의 위치 확인

Root 설계, 2026-09-28. E294는 실제 .5초 건강성/LTM를 유지했지만 startup q RMS .2280028963/max1.3178885184로 유효기각했다. E292/E294의 u/tau 격자나 startup 기준 완화로 돌아가지 않는다. 원래 F1–F8, 모든 과거 실패와 비용, M2/UI, 미노출 F6를 보존한다.

Root 저장 탐색에서 E150과 E257의 동일24core rate 첫100행이 exact이고, E294의 해당 rate RMS 변화45.7697(기준 RMS110.2774)를 확인했다. 기존 core-target 양수3064간선도 자원 소모에 노출됐다. Sol/xhigh 읽기 검토는 일정 rate의 R∞=1/(1+tau*u*r) 때문에 원래 자원식이 정상상태 전달도 줄이고, R초기1은 비영 후보의 orbit 보존이 아님을 확인했다. 정상상태 전달을 복원하는 별도 비선형 가설은 onset 오차를 보장하지 않으므로 아직 actual 후보로 고르지 않는다.

이번 질문은 **선택된 양수 재귀 입력 손실이 기존 core 내부에서 주로 생기는지, 다른 premotor에서 core로 들어오는 경로에서 생기는지, 그리고 실제 core/운동신경/근육/명령 변화와 감각 drive 변화가 어느 저장 시점부터 관측되는지**다. 합계만으로 생물학적 원인·미래 반사실을 확정하지 않는다. 새로운 모델·신경·물리·CUDA·solver·fit·후보 선택은 0이다.

## 고정 입력과 범위

기존 E294의 positive roster, candidate activity/native recruitment/trace, result/ledger/root raw audit/source manifest/output manifest/freeze/source review 및 그 SHA를 사용한다. Baseline은 원래 E150 recruitment-signals/native-recruitment-state/trace와 E257 baseline core-activity다. Core24는 원래 core-activity의 ID 전부이고, 결과 기반 순위로 고르지 않는다. 원래 E150 전역 CSR인 E257 baseline-csr.npz는 선택 간선 위치와 core에 들어오는 전체 nonzero 간선 수/부호를 확인하는 데만 사용한다. 전체 baseline/candidate recurrent current를 관측했다고 부르지 않는다.

모든 새 입력과 실행 source는 root가 실행 전에 SHA로 동결한다. 기존 manifest가 제공하는 각 사용 raw의 SHA와 새 inventory가 같아야 한다. E294 root 감사가 pass/run_valid이고 actual_startup/short/adopted false인지 확인한다. E294 native5000/controls100, neural2000, whole final 가용과 음수 source R exact1, source/graph/adapter 정체성 및 보존 SHA는 기존 검증 증거와 결합해 확인한다. 입력은 읽기 전용이다.

E294 activity의101행(0..100control)과24core,391MN을 사용한다. Baseline E150 recruitment-signals는100행(1..100control)이며 observed_ids가 **정렬되지 않았다**. 명시적인 unique ID→column map으로만 정렬한다. E257 core의 [1:101]과 E150 [:100]의 rate/drive/adaptation을 모두 exact 비교하고, time grid는 원래2e-8초 허용치로 확인한다. 관찰되지 않은 baseline 0행을 만들지 않는다. Core24/MN391 baseline·candidate의 rate/drive/adaptation,96근육 활성,42관절 명령,6힘줄 명령의 원래 저장 정의/순서·행 대응을 검증한다. Native recruitment motor_rate와 activity의 motor 열이 exact인지 확인한다. 근육 activation은 baseline signals/native recruitment와 후보 native recruitment에서 대응한다. 서로 다른 저장 정의라 exact가 아니면 결과를 쓰기 전에 root에게 반환하며 임의로 섞지 않는다.

## 계산과 해석

순수 함수 `core_decomposition(selection, activity, core_ids)`를 노출한다. E293 checked_activity의 typed metadata/전체 행/공유 resource 검증을 재사용할 수 있으나 승인된 E293 파일은 편집하지 않는다. 같은 candidate rate에서 core24로 향하는 선택 간선만 사용하여 source가 core24인 군과 나머지 군을 분리한다. 각 군/각 core/각101행에 float64 수학합 Jpos=sum(w*r), JposR=sum(w*r*R), loss=Jpos-JposR, Jneg와 JnegR을 저장한다. 두 군 합은 모든 선택 core-input 합과 같아야 하고 음수항은 E294에서 exact 동일해야 한다. 모델 입력 단위이며 암페어·CUDA 순서 exact current·전체 current가 아니다. 비선택 간선이나 후보에 없는 신경 rate를 0으로 대입하지 않는다.

모든 core와 그룹의101행 원자료, baseline/candidate 대응100행 신호와 차이를 저장한다. 평균/RMS/최대 변화 및 처음 exact 차이를 관측한 control을 필드별·셀별/근육별/관절별로 기록한다. 첫 차이는5ms 저장 격자의 관측 시점이고 인과 선행·substep onset 증명이 아니다. 고정 요약 구간은 controls1..10(.005–.05),11..20(.055–.10),21..100(.105–.5)이다. 고정 표시는 controls1,5,10,15,20,40,60,80,100이다. 모든 셀을 제공하고 일부 best cell만 고르지 않는다. 한 번의 짧은 이행 궤적에서 주기·위상 통과나 신호의 정상상태 tonic/phasic 분리를 추정하지 않는다.

양수 loss의 core-source/other-source 비중은 분모가 양수인 경우만 정의한다. 0이면 null/available=false로 남긴다. 활동 신호 변화·drive 변화와 손실은 기술 통계로만 연결한다. 이 결과만으로 core 보호 후보나 DC normalization 후보를 승격하지 않는다. 다음 actual 구조를 정하려면 전체 기존 경로에서의 과학적 의미와 고정 시작/보행 판별을 별도로 설계한다.

## 구현·검토·자원

실제 gpt-6-sol/high가 신규 `tools/analyze_c_banc_m3_core_resource_localization.py`, `tests/test_c_banc_core_resource_localization.py` 두 파일만 구현한다. Root FIRST→필요시 같은Sol 첫수정→root SECOND→root 잔여수정 순서다. 기존 데이터/구현/docs/checkpoint는 읽기 전용이다. CLI는 새 freeze/review SHA와 fresh output을 요구한다. ROOT/OWNED/SOURCE_FILES/INPUT_NAMES/LIMITS/FREEZE_SCHEMA/REVIEW_SCHEMA/gate/run을 노출하고 구체적 필드를 root에 전달한다. Execution은 새 `e295-core-resource-localization` 출력만 쓴다. source/input/output manifest와 제한/오류 영수증을 저장한다. 모든 adoption/natural gait/F1–F8/F6/causal/runtime selection 플래그는 false다.

생성 검사30초/1GiB, 저장 분석 한 번30초/2GiB/output64MiB/CPUthread1이다. 원본/model/body/CUDA/solver/fit0이며 이미 발생한 원본 누적8022.108435499544/8095초를 유지한다. 생성 fixture는 작은 회로의 core/other source 분할, 양/음/공유 source, 빈 그룹/분모0, unsorted baseline IDs, 잘못된 dtype/누락/반복ID/clock/hash/incomplete 행을 확인한다. Root는 별도 작은 손계산과 실제 저장 자료의 sparse matrix 재계산으로 검토한다. 결과를 보기 전 새 원본 body 예약은 하지 않는다.
