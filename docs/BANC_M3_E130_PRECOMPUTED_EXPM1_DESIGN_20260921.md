# E130: 단계 불변 expm1 재사용의 수치·비용 비교

2026-09-21 · 부모 사전 설계 · H130/M3 · F1–F8 미완료

E129의 동일 LF 한 점에서 전체 평가2.225543초 중 joint_forward1.774796초, edge_bank0.209658초였다. 동역학을 바꾸지 않는 CUDA 비용 개선을 먼저 검증한다. 소스의 `expm1(delta[fit[j]])`는 delta가 한 평가 내내 고정인데도 단계·수신 연결 루프 안에 있다. 이를 평가당 한 번 계산해 재사용하면 반복 전진 비용이 감소한다는 것이 H130이다. 경쟁 설명은 순차 RK4/tanh/다른 메모리 작업이 우세하여 재사용 효과가 작다는 것이다. 실제 기계 명령별 비중은 아직 측정하지 않았다.

## 변경과 불변 조건

새 `tools/banc_m3_precomputed_expm1.py`는 E128의 검증된 소스를 수정하지 않는 프로세스 내부 context manager `precomputed_expm1(engine)`를 제공한다. 원본 CUDA source에서 정확히 한 곳의 `expm1(delta[fit[j]])`를 미리 계산한 double 배열 접근으로 바꾼 전진 커널을 별도로 만든다. 같은 CUDA builtin `expm1`/float64/기존 compiler flags를 사용한 작은 CUDA 커널로 평가별 delta 전체를 GPU에서 계산한다. 전진 호출에만 새 배열을 전달하고 원래 delta는 edge-bank/미분에 그대로 전달한다. 각 누적 항의 `source * base * expm1` 곱셈 및 더하기 순서를 보존한다. 가중치를 미리 곱하거나 source 전류를 합치는 다른 변경은 하지 않는다.

원본 `_kernel` factory는 finally에서 정확히 복원한다. 입력 delta를 변경하지 않고 추가 CPU↔GPU 복사·numpy 단계 루프·시간 간격 변경·CUDA graph·fast-math·float32 전환을 넣지 않는다. native_mode는 원본을 호출하여 불필요한 준비를 하지 않는다. 새 준비·전진의 요청/반환/동기화·불명확한 완료, GPU 임시 배열 크기와 오류를 별도로 기록한다. 이미 있는 engine의 원자료/부분 오류 복구는 그대로 사용한다. 이 adapter는 격리된 연구 프로세스에서만 사용한다.

E128/E129의 입력·기하·96근육/391MN/69source/2444union/1572fit edge·초기99·loss/제약·분기·감도·컴파일 설정을 보존한다. 원본fit/BANC/신체/F6/제품 UI는 실행하지 않는다. 기존 M2, E128 reject, E121 F4/F5 미달을 변경하지 않는다.

## 비교 입력과 노출

이미 공개된 E128 LF best-objective 기록13의 **비영점** delta/gain를 고정한다. 선택은 현재 profile 결과를 본 뒤 별도로 사전 기록한 개발용 점이며 holdout이 아니다. 원래 raw13의 명시적69MN/17muscle/273edge와 mapping, delta/gain, loss/parts/ratios/초기99/B/gradient를 그대로 사용한다.

실제 RTX4080/CUDA,one CPU thread,동일 fixture와 Torch geometry를 사용하여 원본과 후보 engine을 각각 한 번 구성한다. 각 backend에 E128의 synthetic tiny1MN/20ticks/.005조건부초/1bank를 한 번 실행하여 커널 준비 비용을 따로 기록한다. 신경/기하의 원본점 추가 warmup은0이다.

원본 평가 순서는 **baseline,candidate / candidate,baseline / baseline,candidate**의3쌍이다. 매회 같은 delta/gain,같은3 contexts/2001끝점/native160000단계,`evaluate_leg(...leg=0,index=1)`을 사용하며 각각 별도 출력 폴더에 raw를 먼저 저장한다. 원본은6 LF batches/18 conditional responses/180조건부초/720000 contextticks/49680000 scalar MNticks/786240000 edge-stage ticks/6banks/36000 decoder끝점/기하 forward·backward 각각32838행이다. 데이터 순서는 고정하며 결과를 보고 pair를 추가하거나 고르지 않는다.

E129의 검토된 관찰 래퍼를 동일하게 적용한다. 후보 `joint_forward` event 구간에는 준비 커널과 변경된 전진 커널을 모두 포함한다. 추가 event나 준비 비용을 빼서 비교하지 않는다. 두 backend의 생성/고정 업로드/정리 및 한 번의 tiny 비용과 전체 외부 비용도 보존한다. 전체 평가 CPU wall과 CUDA stream event는 중첩이므로 합하지 않는다. 첫 Torch 호출 비용과 실행 순서/캐시 한계를 명시한다.

## 판정과 보존

모든 pair에서 여덟 응답,전체 branch bits,명시적ID/mapping,delta/gain,초기99/B/오차/gradient를 비교한다. 수치 허용은 E129와 같은 응답/미분1e-8,loss/parts/ratios1e-10이며 branch/ID/delta/gain는 exact다. baseline이 기존 raw13의 공통 저장 필드와도 같은 기준으로 일치해야 한다. 실제 각 결과가 알려진 완료여야 하며, 불완전한 raw/event나 정합 실패는 비용 진단으로만 남긴다.

성능 채택 조건은 모든 수치 비교 통과,3쌍 모두 원본/후보 **전진 CUDA 구간 비율≥1.25**,3쌍의 전체 평가 wall 비율 중앙값≥1.2,오류/불명확한 완료0이다. 조건 미달이면 이 최적화를 채택하지 않는다. 이 판정은 해당 조건부 적합 엔진의 한 점 성능에 한정하며 전체 BANC/실제 보행의 가속 또는 자연 보행 성공이 아니다. 적합 재개와 실제 보행은 후속 사전 설계에서 별도로 판정한다.

원본 프로세스1회/retry0,내부25초/외부30초/supervisor35초다. E129까지 실제 보수적 누적1404.8167935000383초를1405로 올리고35초를 배정한다. 배정 종료1440≤기존1530초 공유 한도이며 이전 비용을 지우거나 초기화하지 않는다. 남은90초 이상은 후속 판단 전 보존한다. 실패한 준비/부분 실행/테스트 비용도 남긴다.

## 구현·리뷰

새 adapter, `tools/verify_c_banc_m3_precomputed_expm1.py`, `tests/test_c_banc_precomputed_expm1.py`만 Sol이 구현한다. 수학 변경 없는 CUDA hoisting과 래퍼/paired comparison 작업이므로 **gpt-5.6-sol/high**를 실제 도구 호출에 사용한다. 부모가 설계·입력/hash·원본 supervisor·독립 원자료 검토를 담당한다. 부모 첫 리뷰→같은 Sol 첫 수정→부모 두 번째 리뷰→잔여 부모 직접수정 순서다.

생성 입력 테스트는 양/음의 delta,서로 다른 중간단계,이질적gain 및 정류 경계를 포함한 작은 조건에서 원본과 후보의8출력/전체B/branch를 비교하고,변경 없는 native mode,원본 delta와 factory 복원,준비/전진 실패의 부분계상을 확인한다. 원본 입력의 합성 위장과 미기록 warmup은 허용하지 않는다. 각 test invocation의 실제 synthetic 호출/ticks/bank/외부 시간과 실패를 남긴다. 부모 최종 검토 전 원본 실행은0이다.

검증 CLI는 `--freeze --freeze-sha --source-review --source-review-sha --out`이고,정확한3소스 해시를 담은 부모 receipt의 `accepted_for_single_original_precomputed_expm1_comparison=true`,`blocking_defects=[]`를 요구한다. 결과·ledger·raw·source/env·전체 파일 manifest를 보존하고 오류 시 원래 예외와 이용 가능한 부분 원자료/부가 오류를 유지한다.
