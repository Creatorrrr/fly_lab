# M3 신체 입력 준비 비용 분리 — ROOT 설계, 2026-10-08

## 목적과 현재 증거

현재 finite-goal은 전체 BANC 신경 출력만으로 실제 신체의 F1–F8을 충족하는 자연 보행이다. Candidate-k는 미채택이고, M2/UI·미사용 F6·신경 해부학 provenance·기존 실패와 비용을 보존한다. 이번 작업의 범위는 실제 신체 실행에 앞선 입력 준비의 관측 가능성이다. 보행 개선, CUDA 개선, 성능 개선이나 qualification 성공으로 판정하지 않는다.

직전 두 진단은 Count assertion 실패와 준비 시간 부족으로 종료되었다. 가장 최근 call `direct-science-20261008-native-count-conformance1`은 실제 첫 FINAL 4.249초/exit1이며 Native를 시작하지 못했다. 폐쇄된 ledger는 282 debits/164 calls이고 generated 사용 932.2819999999998초, local cap964초, shared remaining39.505984400000216초, 종료 reserve20초다. Original remaining3936.9150391004946초는 generated 대체 예산으로 사용하지 않는다. 전체8+12+96+20=136초 qualification은 현재 한도에 들어가지 않는다. 이전 실패를 새 이름으로 재실행하지 않는다.

`arm_data()`가 candidate의 입력 준비 경로를 끝까지 수행하면 명시적 `pre_array` full fence가382회 호출되는 구조다. 이는 소스에서 센 조건부 횟수이며 실제 호출 수·시간이나 병목의 증거가 아니다. 기존 `arm-data` 기록은 전체 준비를 한 덩어리로 표시하므로, 다음 실제 실행에서 어느 입력 단계에 도달했는지와 그 단계 안의 검증 시간이 필요하다.

## 방법 선택

1. Native 비교 가속을 계속하는 방법은 Python 신체 입력 경로에 직접 적용되지 않고, 현재 전체 qualification을 실행할 여유도 없다.
2. Source/이력을 한번 확인하고 성공을 재사용하거나 파일별 fence를 묶는 방법은 중간 변경 감지와 fresh admission 의미를 바꾼다. 실제 병목이 확인되지 않았으므로 이번에는 구현하지 않는다.
3. 기존 RuntimeJournal에 고정된 준비 단계와 그 안의 fence 관측을 추가한다. 검증·입력 순서·신체 동작을 유지하고 미래 actual result가 다음 구조 변경을 결정하게 한다. 이를 선택한다. 추가 로그 시스템이나 파일별 이벤트를 만들지 않는다.

## 구현 계약

Canonical을 지금 교체하지 않는다. 다음 Source 두 개와 필요한 기존 테스트의 수정안을 새 FIRST/R1 proposal에 작성한다.

- `tools/verify_c_banc_m3_startup_ray.py`: `arm_data()`의 기존 문장을 같은 순서로 유지하고, 기존 `phase(timer, name)`를 이용해 아래 준비 단계만 감싼다.
- `tools/banc_m3_contact_runtime_journal.py`: 고정 준비 단계의 시도/반환/실패/진행 중 상태, inclusive wall, 그 단계 안의 fence 시도/반환/실패 및 nested wall을 기존 bounded snapshot에 싣는다.
- 기존 `tests/test_c_banc_contact_runtime_journal.py` 또는 별도 작은 테스트 수정안: fake clock과 생성 fixture로 의미를 검증한다. 이 단계에서 실행하지 않는다.

고정 단계는 `input-original-files`, `input-native-assets`, `input-whole-metadata`, `input-resolve-api`, `input-graph-load`, `input-state-load`, `input-selected-arrays`, `input-result-check`다. Original roster loop 전체와 asset loop 전체에 각각 하나의 phase를 둔다. 파일별/관절별/제어 step별 event나 stream write는 추가하지 않는다. 각 phase는 `opaque=False`를 명시하거나 기존 기본값을 쓴다. 기존 opaque phase의 stream emission 수를 늘리지 않는다.

기존 입력 작업의 order, 각 full fence, digest/header 검사, no-reparse 검사, API/GraphStore/StateStore/selected load, initializer 및 검사 식은 모두 유지한다. 모듈 import는 phase helper를 사용할 최소 추가 외에 이동하지 않는다. neutral/candidate 두 arm과 journal이 없는 caller도 지원한다. No-journal 경로에서 원래 데이터/예외/콜백 순서를 유지한다.

RuntimeJournal의 deadline origin, science270/worker285/flush1초, event64/emit16/wire32KiB/stream512KiB, 오류 보존/terminal reserve, work counter 및 trust/run_valid 의미를 유지한다. 준비 요약은 고정8개 슬롯만 사용하고 bounded encoding의 필수 보존 필드에 포함한다. 준비가 시작되지 않았다는 관측은 신체 constructor/neural/physics 작업이0이라는 주장을 허용하지 않는다.

위험은 중첩 시간의 이중 합산, 오류 unwind 중 현재 단계 상실, deadline 부근의 진단 오류가 original first error를 가리는 것, unreturned 작업을0으로 쓰는 것, Source-only 구현을 실제 측정으로 부르는 것이다. 각 슬롯에서 stage inclusive wall과 nested fence wall을 분리하고 합산하지 말라고 명시한다. fence 오류는 기존 first-error 규칙을 유지하고 같은 오류 객체를 다시 전파한다. 준비 phase의 clock/상태 관측은 실제 데이터 접근을 대체하지 않는다. 가짜 시간은 fixture 결과이며 실제 stage 시간으로 기록하지 않는다.

추가된 단계는 RuntimeJournal의 고정 allowlist에만 등록한다. 성공하면 원래 순서대로 반환, 실패하면 원래 예외 전파, 진행 중이면 `NOT_RETURNED`, 미진입이면 `NOT_STARTED`다. 들어오지 않은 다음 단계를 성공 또는 실패로 채우지 않는다. 준비 요약을 validity/ownership/Source/qualification/과학 결과의 증거로 사용하는 consumer를 추가하지 않는다.

## 리뷰와 다음 검증

ROOT 설계 → 실제 GPT-6 Astra/max 리뷰 → ROOT 반영 → 실제 GPT-6.1 Sol/high FIRST → ROOT FIRST → 필요한 경우 같은 Sol의 sole R1 → ROOT SECOND → ROOT 잔여 수정 순서를 따른다. Delegate는 Source 읽기와 자신에게 지정한 proposal/report 파일 작성만 할 수 있다. subprocess·Python 실행/import/AST/compile/pytest/Ruff·Native/CUDA/Original payload/header/hash·canonical·ledger·계획 수정 및 추가 delegation은 금지한다.

리뷰는 raw existing Source를 먼저 보고 데이터/검증 순서가 바뀌는지, 현재 journal stream budget을 침범하는지, exception과 nested fence 상태가 보존되는지를 판단한다. 적절한 fixture는 성공한 두 준비 단계, 두번째 단계 내부 fence 실패, 중단된 단계/미시작된 다음 단계, 독립 snapshot, journal 부재, bounded encoding/terminal retention을 다룬다. 기존 테스트나 기준을 제거하지 않는다.

ROOT는 텍스트 diff와 호출 순서 보존을 확인하고 proposal만 보존한다. 현재 한도에서는 full runtime qualification과 실제 body run을 하지 않는다. 미래 full qualified Source/current fresh contract가 갖추어졌을 때만 다음 실제 body result의 acquired preparation timing을 해석한다. 가장 오래 걸리는 단계와 nested fence wall을 비교한다. Full fence가 우세하면 fresh mutation detection을 보존하는 구조적 방법을 별도 설계하고, constructor/state load가 우세하면 그 경로를 우선하며, neural/physics가 우세하면 실제 CUDA 경로를 분석한다. 관측이 없거나 잘렸으면 해당 단계 비용은 UNKNOWN으로 유지한다.

## 입력 증거

다음 항목은 코드/진단 메타데이터다. Original binary에 대한 접근은 허용하지 않는다.

- `tools/verify_c_banc_m3_startup_ray.py`: 77875B, SHA256 a89bf6053cdc6774db6efe30640bf5a5e6c0f0a42eb7253a09f718c3e8003a14.
- `tools/banc_m3_contact_learning_admission.py`: 104010B, SHA256 2855307ab64a87e5824989e8d525a989d5024d83c67220f899b7ae4c8bc58a69.
- `tools/banc_m3_contact_runtime_journal.py`、`tests/test_c_banc_contact_runtime_journal.py`、`tools/verify_c_banc_m3_contact_learning.py`: 현재 실제 텍스트를 읽고 각 출력에 input pin을 기록한다.
- `docs/BANC_M3_BODY_METADATA_FENCE_SOURCE_NOTE_20261008.md`: 3481B/SHA256 5d791df87b7d9c0172d1e7c27555ea38171ca97ad4a19222cabf9b7c9ce7a421.
- `docs/BANC_M3_NATIVE_COUNT_CONFORMANCE_ROOT_ACTUAL_RESULT_20261008.md`: 3942B/SHA256 d5f8980f71c40ad4a8fff6ca7a943a5093fa175ae9d777f78a610827c5967148.
- `verification/banc-natural-gait-m3-20260919/parent-native-count-conformance-closed282-20261008-a.json`: 453918B/SHA256 405ffe995f1241f2ff3263efc916c9e69897b18c37028da3a9aab82c6d37f661. 현재 mutable ledger와 동일한 폐쇄 사본이며 전체 원본 데이터가 아니다.

목표는 ACTIVE/INCOMPLETE다. 이번 Source 작업은 보호 실행 비용0이라는 뜻으로 과학적 작업 비용/준비 시간을0이라고 기록하지 않는다. 기존 한도·비용·Source qualification 실패·candidate 미채택·biology UNPROVEN·native VJP UNVERIFIED를 유지한다.
