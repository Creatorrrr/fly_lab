# 신체 입력 준비 관측 — ROOT FIRST 리뷰

2026-10-08. 실제 Sol/high FIRST.md6529B/SHA256 46d87404629450daae2806d47ac0885473793a454c9461e19a3890bebab3281e 및 FIRST.json8916B/SHA256 09e890a05aeb5178425b014efe3e0c6f5ce604313cc683ea93cc211c5043d696를 ROOT가 전체 읽었다. Journal 전체 변경 diff, startup 전체 diff, 추가 test Source를 모두 읽고 입력12개·출력4개 pins를 재검증했다. 일부 도구 출력의 생략 구간은 별도 읽기로 보충했다. 판정은 **보완 필요, Source-only sole R1**다. 성능/qualification/body/가이트는 NOT_RUN이다.

ROOT의 독립 텍스트 비교로 arm_data의 기존 비공백 문장 순서가 동일하고, 함수 밖 텍스트 및 기존 tests 전체 prefix가 동일함을 확인했다. 여덟 단계 경계, StateStore 뒤 graph.hash, non-opaque 기록, lifetime 누적/nearest entry 귀속, partial projection, first-error 보존의 Source 방향은 설계에 맞는다. 이는 실행 동치나 테스트 통과가 아니다.

## 동일 Sol의 sole R1

1. **R1: 기존 중첩 한도 보존.** 기존 `start()`는 `len(self.active) < len(PHASES)`로14단계 한도를 사용했다. 새8개 이름을 PHASES에 추가해 한도가22로 증가했다. 단계 allowlist 확대와 활성 stack 한도를 분리해 기존14를 명시적으로 유지한다. 기존 deadline/event/emit/wire/stream/terminal 한도도 그대로 둔다. 생성 fixture에서14개 기존 phase frame이 열린 동안15번째 prep phase 진입이 거부되고 callback이 실행되지 않으며, 기존 frame들은 정상적으로 닫히는지 검사한다. 기존 enum 이름이나 테스트를 삭제하지 않는다.

2. **R2: pre-entry 관측 실패 fixture.** 현재 clock/event/finish/error-record fixture는 callback이 이미 진입한 뒤의 unwind 실패를 다룬다. 추가 preparation START `_event`가 yield 전에 실패하는 별도 생성 fixture를 추가한다. 원래 예외 객체와 first_error를 유지하고 callback0회, attempted1/entry_refused1/completed0/failed0, latest state NOT_STARTED, matching active frame 제거와 다음 단계 미시작을 확인한다. Enclosing arm-data phase에서도 unwind가 원래 예외를 보존하는지 확인한다. 정상 owner/science 거부와 entered-but-unreturned 의미를 변경하지 않는다. 기존 assertions 유지.

FIRST 파일을 덮어쓰지 말고 새 `verification/banc-natural-gait-m3-20260919/parent-body-preparation-timing-sol-R1-20261008`에 같은3개 proposal과 R1.md/R1.json만 CreateNew/fsync/전체 readback으로 작성한다. FIRST input pins와 ROOT FIRST review를 확인한다. Source/text/hash 작업만 허용하며 모든 보호 실행/import/AST/compile/pytest/Ruff/Native/CUDA/Original 및 canonical/ledger/plan 쓰기0이다. Delegate 추가0. R1 후 ROOT SECOND로 반환하며 Sol R2는 없다.

전체 후속 qualification은 기존 테스트와 새 fixture를 모두 포함해야 한다. 이전14/30을 임의 subset으로 줄여 적용 자격을 만들지 않는다. 현재 최소136초조차 한도에 들어가지 않으며 이번에는 canonical 교체·actual body 실행0이다. Closed282/164의 모든 실패·비용·caps 및 Source-z gate68 실패를 보존한다. Source-only proposal은 candidate-k 채택이나 F1–F8 충족을 뜻하지 않는다. M2/UI·unused F6·BANC provenance를 유지하고 biology UNPROVEN/native VJP UNVERIFIED다.
