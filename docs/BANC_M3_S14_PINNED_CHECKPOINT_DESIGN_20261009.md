# S14: 재검증 병목을 줄여 실제 BANC 보행 실험을 재개

유한 목표와 F1–F8, BANC 신경 단독 출력, 해부학 provenance, M2/UI, 후보 실패 및 미검증 구분을 유지한다. 체크포인트나 테스트 통과는 보행 목표의 완료가 아니다. 사용자는 현재 세션 설계 → 실제 GPT-6 Astra/max 설계 검토 → GPT-6.1 Sol 적절한 effort 구현 → ROOT 1차 리뷰 → 필요 시 같은 Sol 1회 수정 → ROOT 2차/잔여 수정 순서를 승인했고, 최신 직접 지시로 통상 승인 질문을 생략한다. S14는 반복적인 L 잔여 수정과 구분한 하나의 구조적 개선 단계이며 기존 FIRST/R1/실패를 다시 쓰지 않는다.

## 확인한 현재 상태
정본 및 불변 closed307: verification/banc-natural-gait-m3-20260919/parent-l13r2-bootstrap-settled307-HOLD-20261008.json481656/5bab3c24057d981ae68367f4f3d18b132b2f68194d426305c1197dbb687ee358. 전체307 debit/189 paid calls, actual160.783 once, failure HOLD 단 하나, owner/reservation/pending empty, Original jobs/reserve0, lock absent. 이전 턴의 전체 typed307 독립 재구성과 이번 raw/metadata 확인 일치. 실제 full4 INCOMPLETE/124, stdout F, no JUnit; 원래 보존된 Native 배치의 lock/typed-state/legacy-control 3 receipts와 synthetic Original metadata PASS만 확인. 전체 negative group/primitive controls 완료 없음. 시간 병목의 내부 profiler 인과는 미측정이다.
Current Source l13r2-s1.json262140/cd8ea2d80090ef9d6c8ca3d920a8180c1786486388cc9bb98fb76fc4b13d8ae8, all871/11/93/18/protected421/138/core/inline authenticated in adoption. Modules Python63108/Native61095, account 약260KB. 과거 failed graphs는 Current K-HistoryProof/_history_proof에서 매 최초 프로세스 재구성, cached use에서도 모든 raw operands를 재인증하고 full typed clones/whole ledger comparisons를 수행한다. 이전 작은 iteration/바인딩 수정만으로 full4를 시간 안에 마치지 못했다.

## 구조적 수정 가설과 동등성 경계
완전 독립 대조를 이미 끝낸 closed307를 불변 기준점으로 사용하면, 과거 failure graph의 repeated JSON parse/reconstruction은 생략하면서 같은 full prefix와 raw drift rejection을 유지할 수 있다. 단순 해시 한 개나 요약 수치로 과거를 대신하지 않는다. 검증된 모든 old historical read operands를 Source/metadata 읽기만으로 정확한 pin/cap/kind 목록으로 고정하고 현재 c5.json에 넣는다. metadata collector는 기존 _history_proof 및 K-HRead/K-HRaw 사용처를 정적으로 대조해 중복/순서/종류를 기록한다. 새 file loader/import/parser/과학 실행은 쓰지 않는다. Scope/count가 확정되지 않으면 구현 전에 보완한다.

현재 Source는 c5.json의 전체 해시를 freeze한다. c5에는 closed307/원래 정책·grant 및 raw dependency catalog를 포함한다. catalog는 실제 report, recovery, request, trace, cost, witness, JUnit/로그와 old Source archives를 포함한다. Original NPZ는 metadata descriptor만 유지하며 읽기/stat/hash 금지. Current source-freeze871는 별도 전체 검증한다. 기존 historical drift/missing tests에 노출된 l9-before archive와 nested failure를 반드시 포함한다. zero-byte log는 L11 stderr 및 실제 L13R2 stderr라는 정확한 경로만 허용한다. cap policy는 JSON/raw262144, modules65536, 정확히 이름 붙인 ledger524288, token16384, inherited18 Source524288 그대로다.

각 positive recovery/validate와 cached proof use에서 전체 catalog를 bounded raw/stamp/readback/hash로 다시 인증한다. cold recovery는 등록된 immutable307와 정책/grant 전체를 읽어 raw SHA, exact typed caps/policy/recovery/charged-call/old prefix, ordered binary64 totals, closed liabilities 및 알려진 sole failure를 확인한다. 그 다음 typed defensive clone을 반환한다. 과거 기록을 수정하거나 fees를 다시 charge하지 않는다. 현재 ledger는 이 전체 baseline 및 fresh finite seven-call tail만으로 같은 legal states를 재구성한다. Current source fence/actual Job/ENTRY/twoEOF/unknown-owner/refusal/once-cost 검증은 그대로다. OLD historical Source/functions/보존 파일은 삭제하지 않는다; 현재 graph 재실행 대신 검증된 checkpoint trust boundary를 명시한다. 이것은 acceptance model 변경이므로 실제 review에서 동등성/허점과 누락 catalog를 반드시 검토한다.

거부 최적화는 타입이 잘못된 scalar/cap 또는 불가능한 root shape에 대한 early rejection만 허용한다. positive fast accept, signature-only cache, 누락 operand 승인, blob summary 비교, late error 숨김을 만들지 않는다. 같은 Python/Native typed state 및 모든 기존 positive/negative/assertion/primitive cases를 유지한다. 변경이 private artifacts에만 있으면 아직 구현 채택/실행 PASS가 아니다.

## 승인된 총한도 안의 자원 배분
현재 globalGenerated3400/local2664/Original14500,total17900. 실제 남음 direct40.392999999999574/shared48.18098439999949/Original2236.9150391004946. 새 grant/refund/reset 없이 알려진 closed307에서 사전 repair400을 conserved internal transfer한다: global3800/local3064/Original14100,total17900. local은 global 안의 sublimit이며 추가 예산이 아니다. 사전 회계 delta는 caps3, ordered remaining2, 정확한 immutable repair pointer 외에 없다. 기존 모든307 debit/189calls/cost/Source/HOLD와 owner maps 보존. 자체 결정한 방향 변경은 이 문서와 실제 ROOT reflection/active plan에 명시한다.

사전 재배분 후 direct440.393/shared448.181/Original1836.915, bootstrap180/160/20+close20=200 충족. 전체4 PASS/sameSource/known closed/once eligible fee를 확인한 뒤 기존 별도 allocation700/-700을 유지한다: global4500/local3764/Original13400,total17900. bootstrap worst180 이후 direct960.393/shared968.181 >= full qualification690; Original1136.915 >= body/audit/close600. 실제 cap과 liability를 각 admission 직전 확인한다. 실패하면 정확한 비용/HOLD를 보존하고 남은 full envelope가 없는 call은 시작하지 않는다. 원래 계획700 NOT_RUN은 그대로 남기고 과거600/700 소급 변경 금지.

repair400은 fixed before307/prospective whole checkpoint 및 actual own metadata token으로 검증한 bounded administrative transaction만 허용한다. private future snapshot과 새 Source를 먼저 검토·인증하고, 실제 장부 업데이트는 exact ROOT owned atomic publication/readback 및 독립 whole delta proof로 수행한다. stale/foreign token/unknown owner면 미발행. 별도 fee나 과학 테스트를 발명하지 않는다. transferred anchor는 baseline307/189를 유지하고 이후 bootstrap308/190, qualification309–312/194, body313/195, conditional audit314/196이다.

## 구현·검증 경계 및 다음 행동
1. ROOT는 기존 whole read dependency 목록과 원래 Source11/12/13R2/cold307를 Source/metadata-only로 확정한다. actual Astra/max가 이 checkpoint trust model, catalog completeness/caps, conservation/ownership, gating와 raw-drift negatives를 검토한다.
2. 검토 반영 후 실제 GPT-6.1 Sol xhigh가 private Source/metadata drafts만 구현한다. ROOT 1차 전체 리뷰; 같은 Sol soleR1; ROOT 2차/잔여 수정. 새 science test/helper/smoke/profiler/Original probe는 금지, existing stages 안에서만 실제 실행한다. old Sol completed cycles를 새로 이름 바꾸지 않는다.
3. Source caps/full871/11/93/18/provenance/protected421/138/core/inline 검증과 actual adoption 및 owned repair400 ledger delta. Fresh identity/source/recovery/request, known failure만 release.
4. genuine separately admitted full4, 기존 math23/full61(A25/B36) 및 모든 부정 검사. full4 PASS만700; 원래 typed/atomic/foreign lock/drift/unknown/overrun/duplicate 및 old/new Same/JSON controls의 assertions 유지. 이후 네 별도 qualification calls와 실제 Original body/audit를 진행한다. No truncated/per-case timer reset/subset success.
5. F1–F8 actual gait evidence를 얻고 후보 채택 판단. 위 단계 성공은 회계/실행 경로 개선일 뿐이며 자연 보행의 결과가 아니다. fixed8inputs/candidate4/prep/startup99/short/M2/UI 및 biologic limitations 그대로.

비교 예측: 기존 Source는 actual full4를160.783 안에 완료하지 못했다. 새 Source는 같은4 cases/같은180·160·20 및 실제 반환 계측에서 완료해야 한다. 동시 수정이 있으므로 개선의 각 요소에 대한 독립 인과 효과를 주장하지 않는다. 전체 실행 성공/조건부 proxy/실제 BANC body/생물학 의미를 별도로 보고한다.

