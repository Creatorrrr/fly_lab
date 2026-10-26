# S16 ROOT historical reader closure

2026-10-09. Source-only 준비. 실제 S16 코드는 아직 작성하지 않았으며 최종 Source callsite와 Astra 리뷰를 다시 대조해야 한다.

B=verification/banc-natural-gait-m3-20260919. Draft B/s16-root-catalog-v1.json65930/61ed78452d906c07a415359a98423f9afb78a1c3684826f823d1f6653446d364:212 reader rows,195 unique paths,raw0 four. ROOT는 전체212의 원래 target bytes를 각cap 뒤에 인증하고 ordinal0..211/동일path pin/중복tuple를 확인했다. old171 rows는 S15catalog와 전체 typed 동일하다. target authentication은 historical semantic replay·runtime PASS·speed·생물학적 보행을 증명하지 않는다.

| 구간 | 실제 추가된 대상 및 consumer |
|---|---|
|0..170|이전 S15 catalog 전체. 과거 Source/roster/원래 서보·장부/실패·비용·old recovery closure의 원래 path/cap/kind를 그대로 유지한다.|
|171..186|실제 S15 Source/review/adoption/source proof/ROOT FIRST/SECOND/recovery/recovery actual/whole settled309/failure/request/genuine trace/cost/outer/stdout3/stderr0. 새 failure과 Source lineage의 고정 target이며 exact raw0 예외는 실제 S15 stderr만 추가한다.|
|187..197|실제 실행된 S15 replacement11의 s16-before/<originalpath> 전체 archive. 모듈2는65536, 나머지는262144. JSON3은 json kind, 나머지8은 raw kind. 이전 Source/roster/body binding은 이 원래 bytes를 사용하며 current Source fallback은 없다.|
|198..205|S15 recovery가 직접 읽던 old catalog JSON, old coverage raw, provenance3 raw, old S14 failure raw/cost raw/whole308 raw. JSON descriptor와 raw reader kind를 별개로 보존한다.|
|206..211|actual309 raw, latest S15 failure raw/cost raw, prospective ordered post309v2 JSON/raw, 새 prospective repair400 event raw. post309v1의 aggregate arithmetic 오류는 authority에서 제외하고 보존한다.|

핵심 fixture/production reader policy는 일반262144, runtime Source module65536, exact existing ownership lock16384, 유한하게 지정한 ledger524288이다. S16 구현 시 actual S15 whole309와 prospective post309v2 경로를 hard finite ledger 목록에 추가해야 한다. 기존 S14 anchor/post/예전 ledger aliases와 reader controls에 필요한 paths는 그대로 둔다. raw0는 L11, L13R2, S14, S15 bootstrap stderr 네 경로만 허용하고 path/bytes0/empty SHA를 정확히 검사한다.

새 current S16 Source/c5/provenance는 catalog에 자기참조로 넣지 않고 매 positive fresh 인증한다. 기존 policy/grant와 current recovery/Source/c5/coverage/latest failure/cost/provenance도 계속 cache 밖에서 확인한다. 모든212 target raw는 cold와 warm에서 전체 읽기/비교/UTF8/cap/stamp/reparse/SHA를 지킨다. immutable object proof cache와 detached 반환은 유지하며 이미 detached인 anchor를 K-RecoveredSnapshot에서 중복 복제하는 부분만 제거하는 수정안이 리뷰 중이다.

S16 original=anchor는 B/s16-anchor309-after-repair400-v2.json384613/9e198f4705be83e412d938896e84bacd08f07a2c1c01d5eca024b0be2c04553f 이지만 현재 canonical은 actual309/HOLD484147/59c6402e1be0124e590b5e4d9f6ebe5622f53ee7df3a63cb97e3401aebdd7543이고 새400 action은 NOT_APPLIED다. fresh recovery 전에 actual owned400/readback로 post309를 정말 적용하고 독립 whole typed 확인해야 한다. immutable prospective repair event의 publicationstatus는 후일에도 실제 적용 receipt와 구분한다.

이 목록은 ROOT가 읽은 현재 Source와 S16 설계가 요구하는 closure이다. 구현 완료 후 추가 또는 다른 read callsite가 발견되면 실제 전체 target를 재인증하고 revision 및 이유를 보존해야 한다. catalog 자체의 whole hash만으로 closure를 완료했다고 주장하지 않는다. Original NPZ는 읽기·stat·hash·실행하지 않았다.

