# 신체 입력 준비 관측 — ROOT 반영 및 Sol FIRST 계약

2026-10-08. 실제 GPT-6 Astra/max의 review.md6432B/SHA256 b7d686104eee29da63c6a52a3b11bcecfc6360f1cb3b3cfe93c32a8e8c4793de, review.json8800B/SHA256 7cd06db4daeff55a0eb877b12a07f7311d2831c51d87238c7e5247f823009ff2를 ROOT가 전체 읽고 입력9개 pins를 재확인했다. 판정은 조건부 Source-only 구현이며, 실제 단계 시간/성능/보행은 UNKNOWN이다. [ROOT 원설계](BANC_M3_BODY_PREPARATION_TIMING_ROOT_DESIGN_20261008.md)를 다음 조건으로 구체화한다.

## 반영한 B1–B5

- B1: startup의 기존844–853을 original-files,854–862를 native-assets,863–864를 whole-metadata,865를 resolve-api,866–867을 graph-load,868–871을 state-load,872–876을 selected-arrays,877–885를 result-check로 감싼다. 정확한 이름은 원설계의 `input-*`8개다. 각 preceding fence를 그대로 포함한다. graph.hash/checked_checkpoint는 StateStore.load 뒤에 둔다. Preamble의 import/ref.native/neutral registration과 arm_data 바깥 neutral reference load1103–1105는 이8단계로 측정하지 않는다.
- B2: 8개 고정 슬롯은 per-journal lifetime 누적이다. attempted/completed/failed, 미시작·진행 중·반환·실패 상태, 완료/실패 시 inclusive wall과 stage 안 nested fence wall을 분리한다. Fence 입구에서 가장 가까운 활성 준비 phase를 고정하고 종료 시 그 슬롯에 귀속한다. 준비 phase 밖의 fence는 기존 global만 갱신한다. 반복 snapshot은 누적값을 변경하지 않고 partial elapsed를 따로 투영한다. 기존 clock 값을 재사용하며 파일별 event/clock/stream write를 추가하지 않는다.
- B3: 원래 callback 예외가 있을 때 진단 unwind/clock/bookkeeping 오류가 같은 예외 객체와 first_error를 가리지 않게 한다. 기존 외부 phase/fence의 unwind도 함께 보존해야 한다. 복구 가능한 진단 오류는 secondary/incomplete로만 남기고, 관측할 수 없는 wall은 UNKNOWN이다. 정상 경로의 owner/science/clock 거부는 계속 거부하며 callback을 재시도하거나 늦게 실행하지 않는다. Pre-entry refusal과 entered-but-unreturned를 구분한다. Root 설계의 기존 generic phase를 무조건 호출하는 제안은 이 오류 보존 조건에 종속한다.
- B4: 모든 추가 phase는 non-opaque이고 추가 stream write0이다. Event64/emit16/wire32KiB/stream512KiB/flush1/science270/worker285와 terminal once/reserve는 유지한다. Event/secondary trimming 이후8슬롯, global fence, active state, first error, work와 terminal 필수 관측을 보존한다. Cap16384/일반32KiB와 반복 진입·압력 fixture를 추가하되 기존 assertions를 삭제하지 않는다.
- B5: 상태는 실제 snapshot의 observed_s 시점 관측이다. Non-opaque 준비 단계는 메모리에만 있다가 기존 snapshot에 실린다. Hard stop 전에 새 snapshot을 못 얻었으면 그 뒤 진입/비용은 UNKNOWN이다. NOT_STARTED를 이후에도 미진입이었다거나 constructor/neural/physics 작업이0이었다는 증거로 해석하지 않는다. 새 call 활성화나 consumer authority는 추가하지 않는다.

## Sol 작업 지시

실제 gpt-6.1-sol/high Source-only FIRST를 사용한다. R=C:/Users/ckthd/Dev/Projects/fly_lab, B=verification/banc-natural-gait-m3-20260919. 새 `B/parent-body-preparation-timing-sol-FIRST-20261008` 안에 다음 세 proposal과 FIRST.md/FIRST.json만 작성한다.

1. proposals/tools/verify_c_banc_m3_startup_ray.py
2. proposals/tools/banc_m3_contact_runtime_journal.py
3. proposals/tests/test_c_banc_contact_runtime_journal.py

Canonical 현재 bytes에서 시작해 위 Source만 수정한다. 기존 journal 테스트는 전부 그대로 유지하고 의미 있는 fixture를 추가한다. 두 성공 단계와 반복 진입, 두번째 단계 fence failure의 귀속/동일 예외/다음 단계 미시작, active partial 및 detached snapshot, 예외 unwind 관측 오류, neutral/candidate 및 no-journal의 생성 substitute 입력/검사/callback 순서, 기존 event/byte/emit/terminal 한도를 검증하도록 작성한다. Original 파일을 fixture로 읽거나 따라가지 않는다. 단순 Source 문자열 검사를 실행 증명으로 쓰지 않는다.

읽기는 원설계/본 반영안/Astra 두 결과와 raw startup/journal/tests/admission/worker 및 필요한 Source 텍스트만 가능하다. 코드/메타데이터 텍스트 읽기와 해시, 자신에게 지정한 proposal/report의 exclusive create/fsync/full byte readback만 허용한다. Subprocess·Python 실행/import/AST/compile/pytest/Ruff·Native/CUDA/Original payload/header/hash·canonical/ledger/계획/root GOAL_PLAN 쓰기·추가 delegate는 금지한다. Tests/성능/biology/가이트는 NOT_RUN/UNVERIFIED로 보고한다. 보고에는 입력/출력 exact pins와 B1–B5 대응, 알려진 한계만 간결히 적는다.

ROOT FIRST를 기다린다. 수정이 필요하면 동일 Sol의 sole R1만 허용하고 ROOT SECOND 뒤 잔여는 ROOT가 마무리한다. FIRST/R1 파일을 덮어쓰지 않는다. 최종 proposal도 canonical 적용 전 전체 qualification이 필요하며 현재 한도에는 들어가지 않는다. 새 grant/reset/Original 대체/축소 qualifier/자동 재실행0, closed282/164와 모든 이전 실패/비용 유지다. F1–F8 ACTIVE/INCOMPLETE, candidate-k UNADOPTED, M2/UI·unused F6·BANC provenance 보존, biology UNPROVEN/native VJP UNVERIFIED다.
