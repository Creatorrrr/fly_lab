# Native 검증 비용 ROOT 반영 — 2026-10-08

실제 GPT-6 Astra/max의 review.md31633 bytes/SHA `cc195892bf2ee768aa951303089ac78ceb47041445844d43bf19a7a1f3f16b43`와 review.json42669 bytes/SHA `9e6bcf4bb21609286ccf15765b83d8565064d5e01d74dd86635021b367ad9950`의 전체 내용을 읽었다. ROOT는 모든 현재 Source16/보존 Source16/input9 핀을 실제 파일과 확인했다. Source-only 리뷰이며 컴파일·테스트·속도 또는 자연 보행 PASS가 아니다. 원 설계7858 bytes/SHA b61b977199c5844ea8802270c3735c51399c326539c24b870e60d93745c833f1와 의미 검토3732 bytes/SHA93c263f3e64f668b39b5f63cd6f1c42983e55401c9fc7cecbb69b7ec9df46890는 불변으로 남긴다.

## F1–F7 반영

F1을 채택한다. Account Same은 b를 기준으로 branch하며 a.Contains(b key)의 방향과 실제 dictionary comparer를 유지한다. Container CLR type 동일성을 새 조건으로 추가하지 않는다. 알려진 inert dictionary class만 지원하고 non-string key/custom comparer와 미지원 객체는 fallback한다. Scalar는 정확한 CLR type을 먼저 확인하고 원래 -ceq 의미를 보존한다. Prepare Contact-SameTyped와 JSON parser 옵션은 바꾸지 않는다. LanguagePrimitives.Equals의 후보 사용은 실제 differential diagnosis 전까지 미검증이다.

F2를 채택한다. 원 Same을 Same-Strict로 보존하고 그 재귀 호출도 Same-Strict로만 연결한다. 새 Same signature/무출력/원래 label 규칙은 같다. Helper는 Equal/Different/Unsupported를 구분하며, 완전한 지원 tree 방문만 Equal이 된다. Different는 원래 root부터 Same-Strict를 실행해 기존 오류를 복구하고 원래 경로가 뜻밖에 성공하면 comparator disagreement로 fail closed한다. Unsupported/depth/node/compile/invocation 실패는 baseline fallback이며 fast success가 아니다. ReferenceEquals로 방문하지 않은 tree를 통과시키지 않는다.

F3을 채택한다. 모든 helper/literal/initializer는 기존 공통 function Need..function Write-New 앞 추출 구간 안에 둔다. 실제 두 extraction 및 같은 pwsh에서 반복 & owner 호출을 고려한다. Process-local로 재사용할 수 있는 것은 정확한 Source로 직접 compile해 얻은 owned Type capability와 bounded code-only 상태뿐이다. 임의 preloaded type 이름이나 self-claimed marker로 신뢰를 만들지 않는다. Data, snapshot, comparison proof/result 캐시는 없다. Eager compile/외부 compiler 파일/출력 assembly/env 우회도 없다.

F4를 채택한다. Top-level count 같은 값으로만 cheap eligibility를 확인한다. 큰 fixture는 모든 노드를 실제 방문한 supported Equal/late mismatch/compile 및 fallback witness를 요구한다. Depth/node 한도는 helper의 work bound이며 원래 metadata acceptance cap을 대체하지 않는다. 실제 runtime parser types를 Source에서 단정하지 않고 cold/warm/compile/fast/fallback을 별도 보고한다. 기존 named Native groups에 의미 있는 추가 assertions를 넣되 old assertion/name/class/selection/cap을 제거하지 않는다.

F5를 채택하고 원 설계27–29줄의 실행 판단을 명시적으로 구분한다. Full8/12/96+close20=136이 현재 자원에 들어가지 않으므로 전체 qualification/adoption 및 d-body는 차단한다. 독립된 진단은 adoption에 앞서 수행 가능한 별도 계약이다. ROOT의 아래 bounded contract와 Source review가 모두 성립하면 proposal 자체의 의미·비용 정보를 얻기 위한 단일 generated probe를 수행할 수 있다. Full14/30 또는 실제 gait qualification으로 사용하지 않는다. 새 grant/reset/close 차용/Original 대체/Source-equivalence 예외는0이다.

F6을 채택한다. 여섯 predecessor 재귀에서64 activations가 발생할 수 있다는 판단은 Source 복잡도 추론이며 actual trace/time이 아니다. 이번 scope에 budget 재귀 제거·lease·history proof cache를 섞지 않는다. Comparator 진단의 개선이 end-to-end 비용 회복을 뜻하지 않는다. File/hash/parse가 지배적인 경우 별도의 구조 설계를 작성한다.

F7을 채택한다. 진단을 실행하면 실제 새 generated debit/call을 미래 fresh history에 그대로 포함해야 한다. 현재 d before280/162 및277:280의66/67/68 계약은 추가 행을 수용하지 못한다. 따라서 probe 뒤 d가 실행되었다는 결과나 현재 계약의 적격성을 만들지 않는다. 실패한 gate68을 성공 또는 probe로 대체하지 않고 old Sources/costs/candidate도 불변이다.

## 진단의 ROOT 소유 계약

현재 Source와 실제 기존 genuine collector를 ROOT가 읽었다. Native account840–856은 새 generic generated call/source descriptor/단회 freshness와 current local/shared/close 여유를 검사한다. Supervisor808–909는 generated의 ROOT_review_cycle_complete Source closure, 두 production entrypoints, actual admitted call/one reservation, pending/hold 없음, finite existing generated budget, -m pytest/ruff 및 .py target의 frozen inclusion을 검사한다. 기존 collector는 별도 admitted-role, 정확한 workspace, 미사용 call ID, 실제 exec/write_stdin 결과의 첫 FINAL과 opaque chunks를 기록하고 truncated/unknown은 hold한다. ROOT가 이 Source 경로를 확인했다는 사실은 end-to-end probe 실행 PASS가 아니다.

단일 미사용 call ID는 `direct-science-20261008-native-same-probe1`이다. Outer20/child12/cleanup8, host2147483648 bytes, output8388608 bytes를 사용하고 automatic repeats0/max probe1이다. Current local unused 약42.497/shared remaining50.28498440000021에 outer20+minimum close20=40이 들어간다. 20초는 stop bound이며 완료 예상 시간이 아니다. 과거8/12/96·Original300/120·close20/180 및 original/science limits를 바꾸지 않는다.

Baseline은 immutable as-executed-z/account128514 bytes/SHA `ecae6c7a1edae79aa137e8c459da92b6124272693a00d624174d242c5dc7714f`다. Candidate는 실제 Sol FIRST/필요 시 sole R1/ROOT reviews를 거친 proposal만 사용한다. Metadata fixture는 closed280452155 bytes/SHA `583cea595792cb65df694514667e95befceeec5cebd91c06ad5df5c25b3e5212`의 bounded JSON이며 내부 Original payload descriptors를 따라가지 않는다. 두 distinct parse tree를 대조해 reference identity shortcut으로 비용을 축소하지 않는다. Native는 helper extraction까지만 사용하고 ledger action을 실행하지 않는다.

Case order와 work는 pre-run freeze에 고정한다. 작은 scalar/no compile; null/empty와 exact Bool/Int32/Int64/Double; directional dictionary comparer와 container cross-pairs; array order/nested missing/extra/error-output parity; Unicode/culture differential; large full-tree fast witness와 late mismatch; unsupported/nonfinite/negative-zero/depth/node fallback; compile/invocation/collision와 extraction/repeated invocation lifecycle; identical long history의 baseline/candidate cold+3 warm comparisons 순서다. 환경/parser type inventory, compile attempt/time, visited nodes, fast/fallback counters, result/error/no-output parity와 wall time을 실제 값으로만 소형 보고서에 저장한다. 특수 scalar는 검증 전 unsupported fallback을 유지한다.

ROOT가 새로운 diagnostic Source freeze와 CreateNew request를 출판하고 현재 Native에 별도 admit한 후 기존 genuine collector를 사용한다. Python/pytest/Native compile/semantic/cost 실행은 이 실제 admitted generated envelope 안에서만 한다. ROOT가 실제 FINAL을 once commit하고 끝난 identities/EOF/actual charge를 검토한다. Unknown return은 hold/no replay다. Canonical/ledger contract/candidate/body는 proposal 또는 probe 때문에 바꾸지 않는다. Native 부모 bookkeeping이 ledger에 등록하는 것은 실제 admit/cleanup/once cost뿐이다.

진단 complete semantic parity와 actual fast/cold/warm witness가 모두 있으면 comparator-only 가설의 좁은 증거로 판단한다. Timeout/partial output/fallback-only이면 performance INCONCLUSIVE이며 baseline 결과도 새 qualification으로 쓰지 않는다. 기존 fullRuntime14/30, old11/24 assertions/owner9/metadata3는 여전히 future mandatory qualification이다. 현재20초 진단 뒤 남은 자원으로 full qualification이 가능하다고 예측하지 않는다.

## 구현과 남은 목표

다음은 실제 GPT-6.1 Sol/xhigh Source-only FIRST다. 정확한 비교·fallback·재진입 수명·Native Source extraction·진단 계획이 연결되므로 xhigh를 선택한다. Agent는 새 sibling proposal/manifest에만 쓴다. ROOT FIRST 후 수정이 필요하면 같은 Sol sole R1, 이어 ROOT SECOND/ROOT 잔여를 적용한다. ROOT가 all Source/pins와 진단 경로를 리뷰하기 전 실행하지 않는다.

Source-z lint66/math67 실제 PASS와 gate68 실제 timeout/failed-y/c-body UNKNOWN을 보존한다. Generated used921.5029999999998/local964/shared1700/close20, Original12263.084960899505/16200/remaining3936.9150391004946, all owners0는 현재 불변이다. Full F1–F8 ACTIVE/INCOMPLETE, candidate-k UNADOPTED, biology UNPROVEN, exact native VJP UNVERIFIED이며 M2/UI/unused F6/root GOAL_PLAN와 모든 BANC provenance 및 외부 CPG/teacher/phase/trajectory/root 교정 금지를 보존한다. 리뷰와 진단은 자연 보행 완료가 아니다.
