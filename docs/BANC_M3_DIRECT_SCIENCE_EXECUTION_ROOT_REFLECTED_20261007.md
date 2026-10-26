# 새 과학 실행 경로: Astra 검토 반영

ROOT 설계 `BANC_M3_DIRECT_SCIENCE_EXECUTION_ROOT_DESIGN_20261007.md`를 실제 gpt-6-astra/max 검토 결과(D1–D3)에 맞춰 수정한다. 검토 JSON은 `verification/banc-natural-gait-m3-20260919/parent-direct-science-execution-astra-review.json` (12755 bytes, SHA256 ab5c32dab404b36bdc2c3a2cbc4386494b1a5af3d78cd932ba8ada72b3fc89c3)이다. 사용자 자율 진행 지시에 따라 추가 승인 질문 없이 실행한다. 기존 E360 84/252, fixture/suite/Q는 미통과로 보존한다.

1. 새 startup-ray Source, 실제 import helpers, Native caller/supervisor, 정확한 수치 테스트를 ROOT FIRST → 같은 Sol 한 번의 R1(필요 시) → ROOT SECOND 순서로 검토하고 봉인한다. 실제 lint/수치 검사와 별도로 production stdlib entry/pre-array gate의 정상·잘못된 Source/input/grant/offset/mode/role·bool/nonfinite·외부 경로·중복 요청·late drift 거부를 작은 생성 자료로 관찰한다. 원본 파일 접근 전에 거부되는지 실제 경로에서 확인한다. 코드나 caller가 바뀌면 그 검증을 재사용하지 않는다.
2. 작은 생성 worker를 실제 supervisor로 실행해 정상 종료, nonzero, deadline 종료, 공유 Original lock 경쟁, 소유한 프로세스 종료, 중복 비용 거부를 확인한다. 새 call ID, 유한 outer cap, 정리 여유 및 후속 종료 예약을 먼저 admission하고 기존 generated 한도에서 실제 dispatch→첫 최종 full return 비용을 한 번씩 기록한다. 실제 PID/생성 identity와 양쪽 스트림 종료를 보존한다. 종료 소유권을 확인할 수 없으면 원본 실행을 막는다. 기존 shared lock을 원자적으로 획득하고 다른 소유자의 lock은 지우지 않는다. JSON 필드 타입과 limit-baseline-stage 연산 순서를 명시하고 과거 비용은 수정하지 않는다.
3. 이 새 범위의 직접 qualification이 실제 완료된 다음에만 별도 Original+6000을 발급한다. 당시 ledger의 debit offset을 불변으로 묶으며 기존118개 비용과 Original 누적10020.115960899504/stage126.361을 유지한다. global16200/stage6300/new-local6000, worker1/auto0, 역할 cap120/1200/120/300/300/120/120/180을 manifest→request→pre-spawn→pre-array→worker에 연결한다. 발급 전에는 원본 배열·CUDA·신체를 실행하지 않는다.
4. 실제 packet에서 E318 control0/drive1…99와 neutral identity 및 theta1 full99 재현을 검사한다. 고정 alpha 다섯 개, 기존 R/reconciliation, float32 배포/CSR 대응을 실제 감사한다. neutral/candidate 각각 한 인스턴스로0→99→300을 실행하고 native counters/health와 raw body metrics를 독립 재계산한다. body 실패 뒤 다른 alpha를 고르지 않는다. startup 및 short screen 실패는 과학적 거부로 기록한다.

F1–F8, startup99, short screen, BANC 해부학적 provenance, 외부 CPG/교사/위상/궤적/root 보정 금지, M2/UI 및 실패·미사용 F6 상태를 유지한다. 새 qualification은 실행 안전성을 증명하는 제한된 관찰이며 신경망 보행이나 생물학적 기전 성공을 의미하지 않는다. 실제 과학 결과가 다음 행동을 결정한다. 역사 proof forest나 과거 정책 replay를 확장하지 않는다.

현재 상태: 설계 반영 완료; 새 Source FIRST 검토·직접 qualification·Original grant·과학 실행은 아직 미완료다.
