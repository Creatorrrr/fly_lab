# E176: 하중 피드백 복원 후보의 상태 구현 검증

RM의 기존 감각→운동신경 연결 다섯 개만 바꾸는 연구용 wrapper를 구현했다. 작은 생성 CPU 신경망에서 상태 이식·저장·복원·미래 출력을 확인했다. **실제 BANC 또는 신체 실행은 아직 없으며, 보행 개선 후보를 채택하지 않았다.** F1–F8과 전체 목표는 미완료다.

`tools/banc_m3_load_feedback.py`는 E150 MonotoneRecruitmentTrial을 한 번 받아 같은 MotorTimescaleNetwork로 신경망을 재구축한다. 기존 CSR 양끝과 source 값이 정확히 맞을 때만 다섯 `.00003/.00006` 값을 원래 rate 모델의 `.03/.06` float32로 복원한다. 비선택 연결, 부호, 감각 encoder, 근육 전달, 신경 모수와 현재 상태는 보존한다. 원래 값에1000을 곱해 근사하지 않는다. 새 kernel이나 외부 보행 주기를 넣지 않았다.

가중치에 따른 신경망 identity 네 계층과 원래 모델을 복원하는 provenance를 분리했다. 저장한 rate/adaptation/drive/output mask/tick을 유지하고, 다시 열 때 원래 E150 경로를 복원한 후 같은 다섯 효능을 적용한다. 새 의미적 checksum은 실행에 걸린 wall_s만 제외하며, 디스크에서 같은 snapshot을 복원할 때는 wall_s도 정확히 확인한다. checksum은 일반적인 저장 상태 변경 검출용이며 공격자 인증 수단은 아니다.

## 개발과 두 차례 리뷰

독립 소스 조사는 실제 gpt-6-sol/medium, 첫 구현과 첫 수정은 같은 실제 gpt-6-sol/xhigh가 수행했다. 새로운 상태 저장·복원과 소유권 처리가 있어 xhigh를 선택했다. 부모 설계→Sol 첫 구현→부모 FIRST 리뷰→같은 Sol 첫 수정→부모 SECOND 리뷰→부모 잔여 수정 순서를 지켰다.

FIRST에서 부모는 서로 다른 wall_s 때문에 같은 미래가 불일치로 보이는 문제와, 중첩 적용은 허용하면서 그 snapshot은 복원하지 못하는 문제를 작은 생성 신경망으로 재현했다. 첫 실패 자료와 당시 소스는 그대로 보존했다. 같은 Sol이 checksum의 runtime 분리 및 지원하지 않는 base의 조기 거부를 수정했다. SECOND에서 부모는 같은 snapshot의 runtime 복원 대조를 엄격하게 유지하고 프로젝트 Ruff의 B008 기본 인자 문제를 수정했다.

최종 부모 실행은 **37개 focused tests, Ruff check/format, 독립 검산43개 PASS**다. 별도 검산은 직접 만든 7-cell 행렬로 구성한 실제 CPU MotorTimescaleNetwork를 기준으로 선택 weight, 비선택 topology, 비영 신경 상태, 미래 출력, StateStore disk roundtrip을 대조했다. 해당 최종 oracle는 CPU network5개/160 neural steps를 실행했으며 원본 data/GraphStore/CUDA/물리/fit은0이다. 신체와 adapter 직렬화 경계는 fake이므로 실제 E150의 전체 복원 검증을 대신하지 않는다. pytest의 PyTorch CSR beta 안내1개는 보행 시뮬레이션 warning이 아니다.

부모 FIRST/SECOND/final oracle를 합친 기록은 CPU network15개/480steps다. 중첩 실패 재현과 pytest의 별도 생성 비용도 존재하며, 이 숫자에 포함했다고 주장하지 않는다. 부모 validation wall과 로그를 저장했고 자식의 측정되지 않은 전체 시험 시간은 unknown으로 남겼다. 원본 누적3085.011408499908/3202초는 변하지 않았다.

## 다음 실제 대조

`docs/BANC_M3_E177_NATIVE_FEEDBACK_COMPARISON_DRAFT_20260923.md`에 baseline/candidate를 모두 같은 wrapper로 실행하는 비교를 작성했다. baseline은 원래 효능 그대로, candidate만 다섯 값을 복원한다. 원래 E150 출력을 먼저 재현하고, 같은 초기 상태에서 실제 BANC/CUDA·저장 미래·strict startup·10초 F3–F5·미끄러짐을 비교하는 방향이다. 아직 E177 입력·자원·판정 동결과 원본 source gate는 없다.

E161의 저장 분석에서 해당 flexor의 명령 변화가 아래쪽으로 투영됐기 때문에, 연결 복원 자체를 발 들기나 자연 보행으로 해석하지 않는다. 여섯 다리 전체의 감각 연결을 복원한 것도 아니다. M2, 활성 UI, 미사용 F6, 기존 부분 결과와 실패는 유지했다.

주요 증거는 `verification/banc-natural-gait-m3-20260919/`의 `parent-e176-FIRST-review.md`, `parent-e176-SECOND-review.md`, `parent-e176-generated-final-review.json`, `e176-parent-final-generated/result.json`, `e176-parent-final-validation/`와 단계별 소스 snapshot에 있다. 원본 실행 gate로 사용하는 문서가 아니다.
