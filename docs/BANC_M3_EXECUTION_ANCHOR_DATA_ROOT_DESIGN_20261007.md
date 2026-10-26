# 고정 계약의 원장 시작점도 현재 데이터로 선택한다

목적은 기존 생성 검사를 끝내고 같은 BANC 후보 control0→99→300의 실제 보행으로 이어가는 것이다. 전체 F1–F8, 외부 CPG/교사/위상/궤적/root 보정 금지, BANC provenance, M2/UI와 실패·원본 비용은 유지한다. 별도 목표나 검증 프로젝트를 만들지 않는다.

전 단계는 실제 Source 적용과 실제 Ruff/once 비용 기록이라는 진전이었다. 현재 live와 immutable `parent-execution-after-ruff-closed-ledger112.json`은 같은226844B/SHA d0510dbadb2fb80f1b5e285657da62783f1f5ee8594edfea3939d3064fcafd44다. 비용112/calls62/passes28, 모두 charged, 예약/pending/hold0다. e360-structure Ruff는 exit1/full1.762/cap3이며 SIM211과 F821 두 건이다. ROOT prospective Source에서 세 오류를 수정했다. 이 결과는 보행/CUDA/Original 실행이 아니다.

실제 Native writer는 remaining을 `limit-baseline-stage` 순서로 계산한다. reducer의 `limit-(baseline+stage)` 순서가 이 값을 거부했다. 실제 원장 값이나 실패 판정에 epsilon/반올림 보정을 넣지 않고 writer와 같은 계산 순서를 유지한다. commit-aware caller에 남은109 guard도 manifest의 실제 start_index로 고친다.

현재 Source에는 anchor111/61/27을 컴파일한 곳이 남아 있다. 이를112로 복제하는 새 번호별 계층은 만들지 않는다. 고정 schema/역할/명령/Source·clock·ownership·once-cost·budget 규칙을 유지하고, 다음 attempt의 sealed anchor와 실제 counts는 Native가 봉인한 데이터로 선택한다.

최소 변경안은 admission.anchor와 manifest.closed_anchor를 동일한 typed descriptor로 읽고 start_index/start_calls/start_passes를 실제 anchor의 lengths와 비교하는 것이다. compiled closed111은 변경 불가한 보존 바닥으로 남긴다. 새 anchor는 최소한 그111 ordered debits/61 calls/27 passes와 기존 authority/grants/Original/science 값을 보존하고, 모든 추가 비용의 request→actual→cost→debit, typed pass/call·종료·Source9·현재 당시 manifest/collector를 검증한 sealed closure여야 한다. metadata의 true 선언이나 caller supplied counts만으로 이를 통과시키지 않는다.

다음 현재 scope가 과거 mutable manifest/admission/source-freeze 경로를 덮어쓰기 전에, 실제 당시 bytes를 별도 immutable physical 파일로 보존한다. 원래 논리 descriptor와 물리 alias의 bytes/SHA를 같이 바인딩한다. closed112와 실제 실패 cost/request/trace/capture/Source9를 과거 것으로 읽고, 다음 현재 Source/manifest/새 실제 first-cost와 섞지 않는다. 완전한 과거111 정책을 다시 재귀 실행하는 방식은 피한다. 실제 선택이 읽는 유한 physical dependencies와 Source map/cold fence를 유지한다.

첫 등록의 state는 이전 sealed anchor 전체와 같아야 한다. 이전 execution_context_activation/manifest pair를 새 pair로 바꾸는 좁은 전환만 허용하고, debit/call/pass/counter/Original 값을 변경하지 않는다. 새 pass ID만 데이터로 새로 선택한다. 후속 reducer는 새 anchor prefix와 해당 scope의 finite suffix만 검사한다. 역사/current test는 기존84 names/252 phases, Native15+동일 PROGRAM16/8초/37 slots, Source14/full11/late-fourth/nonowned Q 및 부정 의미를 보존한다. 테스트를 skip하거나 합성 paid 첫 행을 만들지 않는다.

남은 shared/local은1018.1519844000001/1018.109초다. 기존 primary768+fallback120=888, grant0/resetfalse를 유지한다. timing은 미확인이다. Original cumulative10020.115960899504/stage126.361/local0/20과 이번 generated-only 권한을 보존한다. 이 수정 뒤 실제 lint→scanner→fixture→suite→Q, 충분한 별도 Original 배정/과학 manifest, 실제 동일 후보0→99→300 및 F1–F8을 계속한다. 승인 질문은 하지 않는다.

실제 Astra/max는 이 데이터 경계에 필요한 최소 보존·closure·trust 조건과 Source 변경 위치를 리뷰한다. 지금 ROOT는 이전 sole Sol R1 이후의 잔여 수정을 담당한다. 새 Sol 수정 사이클은 열지 않는다. 변경이 기존 반영 설계를 넘어서는 새로운 구현 단계라면 그 범위를 명시하고 사용자 지정 설계/리뷰/구현 순서를 지킨다.
