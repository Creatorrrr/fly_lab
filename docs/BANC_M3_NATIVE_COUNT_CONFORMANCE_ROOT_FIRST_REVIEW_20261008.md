# Native Count 관측 진단 ROOT FIRST — 2026-10-08

판정은 보완 필요다. 실제 Sol/high FIRST의 Source35818B/SHA8027664fda05d4df23af71618b49b96bbe315529d515ca69af78fcbe51c990ad, FIRST.md9807B/SHA4766c49f6898fd000410a864cba67864455a37f96660ad3fb00b654e46202f5c, FIRST.json13787B/SHAb0fa75be32e415d630029d3e3e2a5d80dc8e26eea200aec22cd90061fa55582b를 전체 읽고 실제 input10/output2 pins 및 FIRST 자체를 확인했다. Current canonical 원장 SHA5fc8105681c4c47f28112c093be2ef9db64e00e62125a808e3051aa25f127f06은 그대로다. Python/Native/테스트 실행은 없었다.

새 fixed call·독립 baseline·변경 없는 R1 definitions·fixture compile 한 곳·24 fresh operands·12 fixed rows·각 row의 semantic 판정 전 flush·typed owner/절대 deadline·bounded Native streams가 구현됐다. 공학적 Source 구현이며 실제 runtime/Count 의미/종료/qualification은 NOT_RUN이다. 다음 세 보완을 동일 Sol의 sole R1으로 요청한다.

## R1 [P1] 실제 spawn 뒤 모든 예외 경로에서도 시작·소유·종료 상태를 보존

Source `capture()`에서 Popen은 outer try 밖에 있고 reader.start, cleanup/handle close도 예외를 낼 수 있다. 이 경우 parent의 phase Native/native None 분기는 이미 생성한 Native를 NOT_STARTED로 바꿀 수 있다. 실제 capture state/identity/cleanup 정보가 사라지고 report 표현이 실제 작업과 어긋난다.

Popen 전 시작하지 않은 상태와 successful Popen 후 started 상태를 구분해, 프로세스 생성 이후의 reader/identity/ownership/cleanup/pipe/handle 오류를 같은 owned state 안에서 취득·보존한다. 성공적인 생성 뒤에는 NOT_STARTED를 만들지 않는다. 생성된 프로세스는 bounded cleanup에 들어가야 하고 known_close는 실제 종료·동일 identity·양쪽 EOF·resource close 성공의 조합이어야 한다. Partial capture record를 반환할 수 있어야 하며 unknown close는 owner collector HOLD/no-replay로 남긴다. Parent report/marker는 actual all-owned/first FINAL/once fee를 대신하지 않는다. Close 오류를 감춘 known close 또는 새 child allowance는 허용하지 않는다.

## R2 [P1] setup의 getter 영향과 실제 비교 호출의 getter 관측을 분리

현재 row는 method 뒤 CountReads/Trace만 담고, 초기화는 PowerShell indexer assignment64개를 거친다. Setup이 실제 getter를 호출했는지 아직 측정하지 않았으므로 raw after 값만으로 comparator의 reached getter를 단정할 수 없다.

Fresh pair를 만들고 Role/FailCount와64 entries를 설정한 직후, 비교 호출 전에 direct declared FieldInfo로 a/b CountReads/Trace를 취득해 row에 before 값으로 보존한다. Operand Count를 직접 호출하거나 fixture literal을 바꾸지 않는다. Existing after/raw 값을 그대로 보존하고 subtraction/normalization으로 고정 a1b1/a1b0/a1b1 기준을 맞추지 않는다. 판정은 before의 CountReads0/Trace empty 및 기존 after/typed parity/trace/no-output 계약을 모두 요구한다. Setup 오염이 있으면 취득한 원자료를 보존하고 proposal을 reject한다. 안전한12개 관측 취득을 먼저 완료한 뒤 의미 판정한다.

## R3 [P1] 취득된 row의 첫 검증 오류가 다른 durable row의 존재를 숨기지 않도록 수정

Parent record loop는 `validate_row()`의 첫 오류로 전체 loop를 종료하고, rows에 append하지 못한 모든 ID를 missing_ids로 표시한다. 이미 디스크에 저장된 뒤의 row들도 읽지 못하며, 존재하는 invalid/torn row와 실제 없는 row를 구분하지 못한다. Native Summary ids 비교도 현재는 valid rows만 대상으로 한다.

Actual deadline/ownership이 허용하는 동안 fixed12 각각의 bounded read 결과를 취득한다. No-file, present-but-torn/unreadable, raw-read-but-schema/type-invalid, valid row를 구분하고 첫 acquisition/validation 오류와 각 ID의 descriptor/read 결과를 보존한다. 한 row의 schema/type 오류는 후속 안전한 fixed metadata read를 막지 않아야 한다. Valid ownership/deadline failure는 affected 작업을 중단한다. Missing-on-disk와 invalid/not-read는 별도로 기록하며 Native Summary는 실제 persisted observation roster와 valid semantic roster를 혼동하지 않는다. 완전한 typed12/closure/work 조건이 성립하지 않으면 conformance false다. 행이 존재하거나 exit0인 것만으로 completion을 주장하지 않는다.

## 유지·반영 사항

ROOT가 앞서 전달한 counter clarification은 적절하다. Baseline은 candidate CheapFallback을 증가시키지 않으므로 이 절대 before/after 값은 모두 보존하되 method 간 동일값을 요구하지 않는다. Semantic parity는 accepted/typed exception/no-output/operand getter counters/traces/known field/type identity에 적용하며 production compile/fast/fallback/visited/work0·Attempted/Failed false·Type null은 두 method 모두 요구한다. 이는 counters를 삭제하거나 Count 기준을 낮추는 변경이 아니다.

기존 pinned C# class body/namespace-only 변경·baseline rename three anchors·R1 helper bytes, fixed12/24/64, one fixture compile/no production compile, call max1/repeats0, 10/6/4와 Native3/cleanup1/margin0.25를 유지한다. Full-history/warm/lifecycle/qualifier/body/ledger action은 추가하지 않는다. FIRST는 불변이다. 새 sibling `parent-native-count-conformance-sol-R1-20261008`에서 Source/R1.md/R1.json을 CreateNew/fsync/전체 readback으로 출판한다. ROOT SECOND 뒤 잔여는 ROOT가 담당하며 Sol R2는 없다.

보호 실행·canonical/ledger 변경·새 grant·예산 reset·Original 대체·criteria 완화는0이다. Closed281/163와 모든 기존 실패·비용·Source, F1–F8 ACTIVE/INCOMPLETE·M2/UI·unusedF6·BANC provenance·candidate-k 미채택·biology/VJP 미증명은 유지한다.
