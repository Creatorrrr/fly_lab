# E247: 저장된 실제 신경 연결에서 개입 가능한 전달 경로 확인

E246은 R0/R1/R3의 고정 거리 상태 구별 조건을 통과했고 R2/R4는 미달했다. 이를 감각 정보 손실이나 특정 시냅스의 원인으로 단정하지 않는다. E247은 **현재 모델에 이미 있는 경로**를 확인해 다음 한정된 신경 반응 개입을 설계하는 저장 자료 진단이다. 새 신경·신체·CUDA 실행, fit, weight 변경, 다리 명령 생성은 없다. 최종 F1–F8/M2/E150/UI/F6 조건은 그대로다.

## H247과 비교 의미

H247: 현재 교사 관측에서 변하는 실제 감각 입력과 감각 발화가, 변하는 기존 내부51개 중 일부를 거쳐 각 다리의 free MN에 도달할 수 있는 **유효한 두 단계 연결**을 갖는다. 경쟁 설명은 이 선택된 내부 세포의 직접 출력이 일부 다리/근육에 없거나, 해당 연결이 mask/zero weight/정적인 상태 때문에 이번 좁은 경로 후보에 포함되지 않는다는 것이다.

이것은 해부학 및 모델 가중치에 따른 경로 존재 판정이다. 연결의 실현된 인과적 기여, 제어 가능성, 학습 가능성이나 보행 성공을 증명하지 않는다. 경로 부재도 다른 세포·긴 경로의 부재를 뜻하지 않는다. 세포의 soma_side/cell_type에서 담당 다리를 추정하지 않는다.

## 고정 입력과 정합성

- E150의 `actual-global-csr.npz`와 `csr-provenance.json`, `network-layer-identities.json`을 사용한다. E245 observed-start/final의 network model_hash와 E150 network_identity가 일치해야 한다. E245는 원래 MotorTimescaleTrial을 복원했고 관측기만 바꾼 실험이다. 새 모델을 만들지 않는다.
- 저장 CSR의 indptr/indices는 원래 BANC bundle 배열과 exact 일치해야 한다. 원래 anatomical counts/weights와 CSR provenance의 data/indices/indptr SHA도 확인한다. 모든 원래 11,584,852개 항목을 유지한 모델이며 분석 대상만 잘라 기록한다. 비영 native weight가 원래 zero/sign을 넘어 새 연결을 만들거나 부호를 뒤집지 않았는지 전체 배열에서 확인한다.
- E245 roster/metadata/signal 및 시작·종료 network output_mask를 사용한다. mask는 동일한 실제 이진 배열이어야 하고 graph/model/roster/MN ID를 검증한다. source_annotations는 원래 검증된 roster에서 그대로 가져온다.
- E241 parameters/groups의 MN391, free343/fixed48, membership/leg를 유지한다. E246의 판정·독립 검토와 R3 열 ID도 대조한다. 원래391MN 모두를 경로 표에 남기고 fixed48은 판정의 free 대상과 구분한다.

## 동적 상태의 고정 정의

TRAIN 전체1..1200 및 steady TRAIN100..1200에서 float64 population mean/std/min/max를 기록한다. **steady std≥1e−4**인 원래 열을 dynamic으로 표시한다. 내부51개와 감각3604개는 selected_rate, 각 port241개는 실제 주입용 old_filtered를 쓴다. startup 영향과 단순 tonic 입력을 구분하기 위한 동일한 수치 문턱이며, 결과를 보고 바꾸지 않는다. controls1201..2000은 이 통계나 선택에 사용하지 않는다. 뒤구간 자체도 이미 노출된 개발 자료다.

R3는 기존 core_adaptation442에서 MN391을 뺀 원래 순서의51개다. DNg100 두 개를 임의로 제거하거나 CPG라고 이름 붙이지 않는다. +5000 tonic 입력이 유지된 사실을 기록하고 동적 여부는 위 고정 통계로 판단한다.

## 경로와 출력

W[post,pre]와 원래 global edge ID를 이용해 다음 세 가족의 **모든 declared edge**를 추출한다. native zero 및 mask된 항목도 보존한다.

1. sensory3604 → internal51
2. internal51 → motor391
3. sensory3604 → motor391 (짧은 비교 경로)

각 edge에 global edge ID, pre/post 실제 row, anatomical count, 원래 graph weight, native weight(float32), source output_mask 및 native_weight×mask(float32)를 저장한다. 각 node는 원래 ID/annotation 및 감각 port·다리 또는 motor 근육·다리와만 연결한다. 없는 edge는 생성하지 않는다.

두 단계 eligible chain은 다음을 모두 만족한다: sensory의 port old_filtered와 sensory rate가 steady에서 dynamic, sensory→internal weight×mask≠0, internal rate가 steady에서 dynamic, internal→motor weight×mask≠0, 도착 MN은 free343. 감각 다리는 실제 port 소속을 기록하며 도착 다리와 같다고 요구하지 않는다. 같은 다리/교차 다리 수를 각각 보고한다.

각 다리에 eligible chain 도착 free MN이 하나 이상 있으면 `existing_dynamic_two_hop_coverage=true`다. 여섯 다리 각각의 결과와 전체 AND, 도달 MN/근육 수, 미도달 근육·MN ID, sign 조합 및 직접 sensory→MN 비교를 모두 저장한다. 경로가 있는 각 free MN에 대해 낮은 global internal→MN edge ID, 이어 낮은 sensory→internal edge ID 순서로 첫 witness를 저장한다. 가장 유리한 경로나 뉴런을 탐색해 기준을 바꾸지 않는다. 전체 free343에 대한 완전한 route 목록은 raw로 재구성 가능해야 한다.

후속 선택: 여섯 다리 모두 좁은 경로 조건을 만족하면 기존 두 단계 경로를 보존한 짧은 paired neural response 대조를 설계한다. E246에서 미달한 LH 전달을 먼저 검토하되 이번 자료에서 weight를 맞추거나 관절 명령을 만들지 않는다. 일부 다리가 없으면 그 다리의 더 긴 실제 경로 또는 관측 범위를 다음 질문으로 삼는다. 어떤 경우에도 신규 직접 fanout, 외부 위상/CPG/교사/KNN 정책을 최종 후보에 넣지 않는다.

## 구현·검토와 한도

부모 설계 → 실제gpt-6-sol/medium 구현 → 부모FIRST → 필요시 같은Sol 첫수정 → 부모SECOND → 남은수정은부모 순서다. 소유 파일은 `tools/analyze_c_banc_m3_existing_routes.py`, `tests/test_c_banc_existing_routes.py` 두 개. 기존 CPU 분석의 hash/NPZ 저장 도우미는 재사용할 수 있고 과거 파일은 수정하지 않는다. medium은 새 신경 적분 없이 고정 CSR 행과 검증된 rosters를 조합하는 작업 난이도에 맞춘다.

생성 테스트는 CSR 방향/global ID, zero/mask 보존, source dynamic/port dynamic의 독립 조건, free/fixed MN 분리, startup/tail이 steady 선택에 들어가지 않음, 결정적 witness, 부분/늦은 오류 판정 철회를 검증한다. 부모는 별도 직접 edge 조회로 모든 witness와 coverage/미도달 목록을 검산한다.

원본 CPU 분석1회/retry0, CPU1/GPU0, fit/solver/neural/body0. internal30초(마무리5초 포함), external40초, supervisor45초, RSS2GiB, 출력64MiB, 최소여유512MiB. 기존6738.985683899624/6800초, 잔여61.01431610037616초 중45초 예약, 추가 배정0. 이 저장 분석도 원본 누적 회계에 더한다. 생성 검사·부모 검산은 별도 측정, 개발/자문 미측정 시간은null이다. 실패·부분 출력과 오류 순서를 보존하고, 늦은 저장·기한 오류도 root/nested 판정을 철회한다. 사후 한도 변경이나 원본 재시도는 하지 않는다.
