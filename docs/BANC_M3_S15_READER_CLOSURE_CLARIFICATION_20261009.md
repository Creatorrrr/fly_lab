# S15 ROOT design clarification — 2026-10-09

本補正 governs the catalog portion of docs/BANC_M3_S15_TYPED_BINARY_READER_DESIGN_20261009.md and docs/BANC_M3_S15_HISTORY_COVERAGE_20261009.md. The original design/coverage and v1/v2 catalogs remain immutable publication records, not adoption or completeness claims.

Actual Astra/max review found missing direct S14 recovery raw targets. ROOT added7 exact rows in v2: S14 catalog itself(json), repair400(raw), post307(raw524288), old S14 coverage(raw), old provenance3(raw). Parent_catalog and an embedded descriptor do not reauthenticate the original target.

Actual direct reader-kind closure required6 further rows in v3: L13R2 failed report(raw), its actual cost(raw), policy(raw), grant(raw), original307(raw524288), post307(json524288). Existing JSON reader rows remain. The archived c5 already has a json row156; ROOT inspected it and the reviewer corrected their earlier all-raw archive assumption. Internal raw work of a JSON reader is not double counted.

Final prospective catalog: verification/banc-natural-gait-m3-20260919/s15-root-catalog-v3.json51999/d90b324ef85ce19c0438d32a7c4c84e2ba40168798e29fc2d7265b86ba774c2d,171 reader tuples/161 paths, exact three raw0. It contains original130 + actual latest17 + executed archive11 + direct-read-target7 + additional direct-reader-kind6. ROOT independently bounded-hashed all171 operands and checked exact ordinals, unique path/cap/kind tuples and same-path pin consistency; no historical semantics/project execution/Original access occurred. Actual review and final contract must bind this v3, not v1/v2. Source-only count/hash identity is not runtime or completeness proof.

Module/archive65536, ordinary262144, exact named historical/actual/post ledgers524288 and lock16384 remain. The c5 trust, coverage, Source and provenance contract must explicitly accept the complete new whole308 reader closure. Cache/safe positive raw reads, whole typed current ledger and failed cost preservation remain as designed.

The narrow typed byte[][]/byte[] optimization, no new helper/Add-Type/content cache/flat key/profiler, same full4/qualification, existing caps3800/3064/14100 and total17900, full4+close200 and PASS-only700, F1–F8 criteria and user-authorized workflow are unchanged. Performance contribution and first F cause remain hypotheses. No new paid call/Source adoption/ledger mutation is part of this clarification.
