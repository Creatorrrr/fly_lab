# Anchor DATA 리뷰 반영과 ROOT 잔여 수정

실제 GPT-6 Astra/max의 `parent-execution-anchor-data-astra-review.json`(14152B/SHA ff65dc596555fbf0a4625c5ad0abdec9a256cda1487afbb3935033c65725a681)과 읽기용 MD(4657B/SHA16354b4051484ec29ee6a101224ddaeb1f09f63faa69f80f65400b19083a45e9)를 ROOT가 읽었다. 판정 REVISE와 R1–R3를 아래 수정에 반영한다. 기존 Sol FIRST/sole R1 이후 ROOT가 맡은 잔여 수정이며 새로운 Sol R1은 열지 않는다.

현재 anchor는 admission/manifest DATA로 선택하고 실제 counts를 비교한다. compiled closed111과 그1204 history tuples는 보존 바닥이다. 바닥 이후의 실제 행만 당시 context와 함께 평평하게 검증한다. 현재 known extension은 실패 Ruff 행111이며 실패·1.762초·원래 logical descriptors와18자리 Native identity를 보존한다. 새 행112를 실제로 실행하기 전 성공했다고 취급하지 않는다.

이전 설정/소스를 `parent-execution-structure-as-executed-context.json`의16 original→same-byte physical aliases로 보존했다. ROOT는 Native CreateNew/Flush/readback을 수행했다. 13 inputs는 당시 request/cost/actual/collector/capture/output/두 raw stream과 변경하지 않은 own5다. 별도 history table의 기존1204 tuples는 그대로이고 실제 선택되는30 records를 더한1234/aggregate72112863이다. 이 metadata 출판은 runtime 검증이 아니다. Astra는 이 보충 자료 전체를 인증하지 않았으므로 ROOT가 실제 선택과 cold fence 연결을 검증한다.

context는 archived logical descriptor를 전체(path,bytes,SHA)로 확인한 후 선언된 physical alias를 읽는다. path만으로 교체하거나 H 누락을 D로 넘기지 않는다. 새 current Source는 현재 freeze, 과거 cost/request는 당시 own9와 비교한다. 기존 collector/capture의 실제 clock·exit·opaque identity 검사를 명시된 historical context로 재사용한다. 바닥 이전 정책을 재귀 실행하거나 과거 frozen 코드를 실행하지 않는다.

등록 전 whole live==selected closed anchor와 ownership0을 요구한다. 등록 후 key union 및 정확한 새 pair/grant0만 허용하고 이전 비용·call/pass·Original·예산을 보존한다. remaining은 실제 writer의 limit-baseline-stage 순서로 계산한다. commit-aware caller는 manifest.start_index를 사용하며 descriptor-first/once/불확실 commit 처리 ABI를 유지한다.

현재84 names/252 phases,37 slots,Native15+같은 PROGRAM16/8초,Source14/full11/late-fourth/nonowned Q와 negative 의미는 유지한다. 역사 P30 테스트는 compiled floor111, 현재 테스트는 실제 선택된 context와 새 paid Ruff를 쓴다. Source 적용과 actual lint→scanner→fixture→suite→Q를 거쳐, 별도 Original 계약에서 실제 startup/보행으로 이어간다. 전체 F1–F8은 미완료이며 이번 Source 검토를 보행 성공으로 표시하지 않는다.
