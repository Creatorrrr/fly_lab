# E123: 기존 BANC 운동신경의 근육별 활동 적응과 상태 복원

2026-09-21 KST · 부모 상세 설계 · Sol/xhigh 구현·첫 수정 · 부모 리뷰·잔여 수정

## 목적과 완료 경계

E122에서 전체 관절 추적 RMS는 E121 .001877705rad/교사 .002032111rad이지만, 지지 발의 전방 움직임과 미끄럼은 E121에서 더 컸다. 이전 E107 외부 형태 개입은 보행 주기를 회복했고 시점만 바꾼 E106은 실패했다. 정적 가중치와 초기99 제약의 E119/E120 국소 한계를 유지하면서, 기존391MN의 현재 gain0 활동 적응을 근육별로 조절할 수 있게 하고 실제 전체 신경망에서 상태·출력 전달을 검증한다.

이는 E123 제안의 첫 구현 단계다. 고정 감각 입력에서의 짧은 신경 진단은 보행 적합, 두 실제 관측 문맥의 조건부 fidelity, 폐루프 보행 또는 F1–F8 통과가 아니다. 기능 진단을 통과하면 원래 초기99와 두 실제 관측 문맥을 유지한 모집 형태 적합의 상세 설계로 이어간다. 여기서 gain1은 적응 경로가 실제로 작동하는지 확인하는 진단값이며, 보행 개선 후보로 선택한 값이 아니다. 전체 자연보행 목표를 줄이거나 완료로 바꾸지 않는다.

## 변경 범위

Sol이 작성할 파일은 `tools/banc_m3_motor_adaptation.py`, `tools/verify_c_banc_m3_motor_adaptation.py`, `tests/test_c_banc_motor_adaptation.py` 세 개다. 기존 CoreAdaptationNetwork/AdaptiveRateNetwork, M2, 평가기, 제품/UI/서버는 수정하지 않는다. 다른 파일·원본 결과를 덮어쓰지 않는다. 전체 F1–F8, 초기99 qRMS.02/qmax.05/LTM RMS.01/max.03, 기존 형태20% 기준을 유지한다.

새 `MotorAdaptationNetwork`는 기존 CoreAdaptationNetwork를 확장한다. 선택된 기존 MN ID에만 유한 float32 gain을 명시적으로 적용한다. ID는 고유한 기존 정수 index이며 입력 순서를 보존한다. gain 범위[0,8]은 현재 기본 모델 gain8 안에서의 공학적 제한이며 실측 생리값이라는 주장은 하지 않는다. 나머지 모든 셀의 실제 _cell_gain, 전체 CSR 및 부호/값, tau/a/threshold/cap, dt와 RK4 계산 순서를 유지한다. 원래 적응 미분 `(gain_i*rate_i-adaptation_i)/.3`을 CPU와 기존 CUDA fused kernel 모두 사용하며 새 상태 방정식·CPG·주기 입력·시간 ramp는 추가하지 않는다.

선택 gain은 CUDA 그래프 warmup/capture 전에 존재해야 한다. _cell_gain tensor를 capture 후 교체하거나 CPU에서만 바꾸는 구현은 거부한다. 기존 gain0과 동일한 벡터에서도 새 기능의 별도 identity를 유지하되 원래 identity와 모든 차이를 설명할 수 있어야 한다. 새 identity에는 source core identity, 선택 ID 순서, 실제 float32 gain 전체가 포함된다. snapshot/restore는 명시 schema·원래 core metadata·모든 rate/adaptation/drive/output_mask/tick과 선택 gain metadata를 검증한다. 잘못된 gain/ID/hash를 복원하기 전에 거부하고 기존 상태를 바꾸지 않는다. set_muted의 기존 outgoing-cut 의미와 source 자체 적응은 유지한다.

`MotorAdaptationTrial`은 실제 IntrinsicRecruitmentTrial 위에 적용하는 연구용 wrapper다. 기존391 motor_ids/96muscle pool의 각MN단일membership를 확인하여96근육 gain을391MN gain으로 매핑한다. 근육 이름/columns/ID provenance, cal/rest/moment/pool, body/world/adapter/입력 경로를 유지하고 전체 snapshot에 patch/source identity를 저장한다. from_checkpoint가 원래 layer를 복원한 뒤 동일 gain layer를 복원하도록 한다. 실패 시 자신이 소유한 신체를 닫고 원래 예외를 보존한다. E123 실제 실행은 이 wrapper의 신체를 구성하지 않는다. 실제 신체 사용과 완전한 몸통 미래 검증은 다음 폐루프 후보 검증에서 별도로 필요하다.

## 원래 자료와 네트워크 재구성

`e123-frozen-design.json`에 설계, E121 calibration/metadata/anatomy와 checkpoint-1s 전체 StateStore, 원본211 manifest/부모 검토, E122 원본21/부모 검토, graph bundle, 실제 읽는 source SHA를 동결한다. graph bundle은 `data/acquisitions/banc888-windows-20260910/bundle`의 전체158706 노드와 원래 모든 CSR 연결이다. GraphStore.load와 rate_weights의 원래 계산, BancTargetEncoder(forward_drive=5000)의 기존 port64 calibration 후 E121 cal.edge_ids의 values를 정확히 적용한다. E121은 이미872 target-port weights의64증폭을 제거한 후보라는 사실을 유지한다.

원래 full tau=float32 .02에 cal.tau_ids/tau_values만 적용하고 a1/threshold7.5/cap200 기본값을 사용한다. 이 재구성이 기존 network identity `bfc89ead7088892f8c7fb479d25010c35a9c590825e777b8e8c9b9a3d01674d7`와 정확히 같아야 한다. 추측한 매개변수로 identity 검증을 생략하지 않는다. cal/observed metadata와391 motor root roster를 GraphStore의 실제 index로 교차 확인한다. 부모는 별도의 정적 CSR/매개변수/identity 검산을 한 번 수행한다(신경 구성/전진0, 별도 비용).

출발 상태는 이미 노출된 E121 1초 checkpoint이며 native tick4000, neural dt.00025s, 원래 body state와 근육 activation을 보존한다. 전체158706 실제 rate/adaptation/drive/mask를 StateStore에서 hash 검증 후 읽는다. 원래 MN gain391개는0, 기본 gain8/tau.3이다. 저장한 drive를 진단 동안 고정한다. 이것은 새로운 감각 폐루프나 원래 E121 1.0→1.05 미래의 재생이 아니다. 저장된 기존 감각 입력에 대한 전체 신경망 동역학 검증이다.

## 단 한 번의 실제 CUDA 진단

부모 최종 source review 후 단일 프로세스에서 아래 arm을 순서대로 각각 한 번 실행한다. 실제 신체 생성/물리 전진/새 pose/F6/fit/LP는0이다. 새 `e123-motor-adaptation` 출력이 있으면 거부한다. 각 arm은 같은1초 신경 상태와 원래96근육 activation에서 시작하며, 새 모델 hash/명시 gain metadata만 변환한다. 변화 전후의 full CSR와 모든 비선택 physiology를 확인한다.

1. **original_core**: 기존 CoreAdaptationNetwork, 저장된 original core gain0; 고정 drive로10×20 native step=.05초.
2. **zero_motor_gain**: 새 network,391MN gain0; 같은.05초. 매5ms 모든158706 rate/adaptation, drive/mask/tick 및 원래 decoder의96activation/42관절/6힘줄이 original_core와 bit exact여야 한다. source identity와 추가 metadata 차이만 별도로 기록한다.
3. **positive_motor_gain**: 새 network,391MN gain1; 같은.05초. 전체 상태를 raw로 먼저 저장한다. 선택 MN의 adaptation이 유한하게 실제 증가하고 원래 neural rate/근육 요청과 차이가 생기는지 기록한다. 나머지 physiology/CSR 및 graph/node provenance는 동일해야 한다. 이 차이는 개선이나 생물학적 검증이 아니다.
4. positive .05초에서 신경+96근육+cal/held-drive를 StateStore로 저장한다. 원본과 disk clone이 각각20step=.005초 더 진행한 전체 신경·근육·요청 미래가 bit exact인지 검증한다. clone 생성/복원도 제한 시간 안에서 시작한다.

원래 logical neural advances는 baseline.05+zero.05+positive.055+clone.005=.16초/640 native tick이며, 이것을 신체 시간으로 표기하지 않는다. 초기화용 CUDA warmup/capture의 계산도 별도 setup으로 기록하며, restore 후 clock이 원래4000인지 확인한다. 준비 중 계산과 과학적 arm response를 숨겨 합치지 않는다. 내부에서 다시 원래 arm을 실행하거나 silent retry하지 않는다.

decoder는 기존 `activation += -expm1(-.005/.01)*(pool@rate-activation)` 및 `rest+.02*moment@activation`, LTM `.2*tanh(activation[ltm]/10)`을 사용한다. 각 arm 시작은 저장된1초 activation이고 새 적합·초기화를 하지 않는다. body.step은 호출하지 않는다. arm별10회 및 positive/clone 추가1회씩 총32 decoder control을 별도 계수한다.

## 검증·기록·자원

합성 테스트는 구조가 다른 소수 셀의 양수/음수 연결, 서로 다른 gain, 비영 rate/adaptation, silent/cut/drive 변화, 독립 DOP853 기준(양 상태 atol/rtol2e-5), gain0 기존 CPU/CUDA 경로 일치, snapshot/손상 거부/정확한 미래, invalid ID/gain을 확인한다. CPU와 CUDA 수치 일치를 실제로 실행하여 기록하며, CUDA 테스트를 skip하고 지원 완료로 표기하지 않는다. 합성 모델 계산 횟수/모델초/벽초는 별도로 계수한다. 합성 반복으로 결과를 선택하지 않는다.

실제 raw는 각5ms full rate/adaptation과391MN/96activation/42request/6tendon, tick, held drive/mask, model/gain identity를 보존한다. 비교·숫자판정 전에 현재까지의 raw를 저장한다. 진행 ledger는 파일 저장 성공과 독립적이며 advance가 성공한 뒤 관측/저장 실패가 발생해도 이미 진행한 tick을 잃지 않는다. 모든 완성/부분 arm과 메모리에 계산된 raw를 독립 저장하고 최초 예외·후속 저장/정리 오류를 함께 남긴다. 준비·비교·StateStore·CUDA 동기화·close·마지막 저장까지 비용에 포함한다. 모델 초기화와 매control/clone 시작에 시간 여유를 검사한다.

한 GPU, CPU thread1, 내부45초/외부60초/child55초, 새 모델·control 시작에는8초 저장·정리 여유를 둔다. 기존 누적925/shared988초를 유지한다(외부60초를 모두 사용해도985초). 부모 정적 검산과 합성 테스트 비용은 별도 공개하며 실행 failure와 ceiling을 뒤에서 바꾸지 않는다. 부모 감독 프로세스가 실제 child를 종료/확인하며 관측 timeout만으로 재실행하지 않는다.

유효한 neural implementation diagnostic은 정확한 원래 재구성·gain0 실제 전체 동등성·양수 gain 실제효과·저장 미래·자원·원본보존을 모두 요구한다. `candidate_adopted=false`, `F6_exposed=false`, `whole_goal_complete=false`를 유지한다. 성공 후에도 다음 조건부 모집 형태 및 실제 보행 검증을 계속한다. 부모 설계→Sol/xhigh 구현→부모 첫 리뷰→같은 Sol 첫 수정→부모 재리뷰·잔여 직접 수정 순서를 지킨다.
