# S14 historical read coverage

이 문서는 실제 Source/metadata 읽기로 작성한 coverage 대조이며, 과거 semantic verifier를 새로 실행했다는 증거가 아니다. catalog v3는130 (path,cap,kind) tuples/126 paths이다. 현재 trust boundary는 이전 턴에 전체 typed307을 독립 재구성한 불변 ledger와 실제 raw record/Source 핀이다. v1→v2→v3의 보완 기록을 보존했다.

| 읽기 경로 | catalog에 포함한 범위 |
|---|---|
| Native K-ColdProof / Python _l9_cold_proof | l8-recovery1.exact_l7_cold의9 metadata pins, legacy299/before299/closed300, 정책·grant,4 executed Source archives. L7 body/roster는 frozen vector와 pin을 비교하며 해당 함수가 새로 파일을 읽지 않는다. |
| K-Failed / _l10_failed_bootstrap: L8 | 실제 report/bootstrap Source,5 old archives l9-before, cost/request/trace/outer/whole4 JUnit/nested failure/closed301/recovery. Native의 old roster JSON read는 raw와 별도 kind로 포함. |
| shared cold9 | 실제 report의 before/witness/cost/trace/source/request/recovery,5 cause archives l9-cold-before,closed302. |
| failedR2 | 실제 report/source,cost/request/trace/outer/JUnit/closed303/recovery,5 archives l10-before 및 roster JSON 별도 kind. |
| shared cold10 | 실제 report와7 correlated pins,5 cause archives l10-cold-before,closed304. |
| L11 timeout | report/source/recovery/cost/request/trace/outer/closed305,5 archives l12-before 및 roster JSON, stdout2/정확한 empty stderr. JUnit은 실제 NOT_CREATED이므로 가짜 pin 없음. |
| shared cold12 | report와7 correlated pins,5 cause archives l12-cold-before,closed306. .852 once/180 release와 falseEOF/null owner는 보존된 record의 의미이며 본 catalog raw auth를 semantic replay라고 부르지 않는다. |
| latest L13R2 charged tail / K-FeeProof / Python fee_proof | actual full4 timeout report/recovery/source/request/trace/cost/outer,stdout F1byte/정확한 empty stderr/closed307. actual Source11개를 s14-before에 freeze871 descriptor/current bytes/SHA와 대조해 포함. |
| latest request identity / K-Plan→K-Roster | 최신S13R2 before-roster(l13r2-before/t5), executed roster(s14-before/t5)를 JSON kind로 포함. 둘의 raw roster Source도 포함. Source before l13-s1 포함. Generated bootstrap plan은 이 Source/roster와4 exact node argv이며 Original plan/array를 수행하지 않는다. |

기존 current callback 기록의 repeated 같은 tuple은 deduplicate한다. 각 positive/cached recovery use는 모든 unique tuples를 다시 raw 인증한다. raw와 json 같은 파일 읽기는 reader kind가 다르므로 서로 합치지 않는다. ordinal0..129, pin/cap/kind/empty-path policy와 catalog 자체 고정 pin을 확인해야 하며 수량만으로 completeness를 주장하지 않는다.

모든 module65536, JSON/raw262144, 정확히 명명된 ledger524288, lock16384, inherited18 large Source524288 경계를 유지한다. raw0는 L11/L13R2 exact stderr0+emptySHA 두 경로만 허용한다. Source14 전체871 검증은 이 history catalog와 별도로 수행한다. current before-source S13R2 및 사전400 후 prospective anchor는 final Source/runtime에서 명시적 핀으로 연결해야 한다. catalog가 현재 mutable executable bytes를 대신하지 않도록 s14-before11의 whole Source 검증을 포함했다.

과거 prefix/fees/held failure를 무효화·재환급·재실행하거나 Original NPZ를 읽는 작업은 없었다. 본 작업에서 실제 정본/Source/장부 변화는 아직 없다. 새 전체4/23/61 및 실제 BANC body/F1–F8 통과 여부는 별도 실제 실행으로 검증해야 한다.

