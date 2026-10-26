# P29 현재 정책 검증 수정 — 실제 Astra 리뷰를 반영한 ROOT 설계

2026-10-07. 목표는 실제 보행 실험을 막은 현재 정책 검증 결함을 고치고, 기존 검사를 통과한 다음 같은 후보의 실제 신체 실험으로 이어가는 것이다. 이 수정 자체는 보행 개선이나 생물학적 기전의 증거가 아니다.

실제 GPT-6 Astra/max 리뷰 JSON `parent-e360-p29-actual-astra-review.json`(27893B, SHA256 `4ce4bf3a8bcdb482b2aee03bbf36a9e07329fabd7980acb9b2bbb8d94d475591`)과 MD(4372B, `051fd79ba1d850f2c40d1ffa824735ba3b8e63f57dc904401dc8517a777cea22`)를 ROOT가 읽었다. R1–R3를 모두 채택한다. 원 설계 `docs/BANC_M3_E360_P29_NATIVE_CURRENT_POLICY_ROOT_DESIGN_20261007.md`와 선택 입력35개 inventory가 규정한 변경 범위·자원·보존 조건도 유지한다. 충돌 시 이 반영 설계의 구체적인 수정 지시가 우선한다.

## 실제 상태와 수정 이유

P28 실제 활성화는 한 번 실행되어 exit1로 끝났다. 원래 ErrorRecord는 Current-Policy3884 → Recovery-Literal4786 → main6047을 가리킨다. 현재 정책11340B/SHA37859e7…를 과거 P27 정책10788B/SHA4b19ed…와 비교한 오류다. 전체 실행 시간은 확인할 수 없으며 재구성하지 않는다. bookkeeping149도 현재 SHA와 과거 크기10788을 혼합한다. 후자는 아직 실행하지 않은 코드에서 확인한 결함이다.

원장은 실제 P27 closed108(206426B/SHAf0e85834…)이며108 debit/58 call/25 paid pass, hold/pending/reservation0이다. P28은 활성화 쌍·pass·call·debit를 만들지 않았다. 소스 전임자는 실제 미지불 실패 P28, 지불된 전임자는 P27 실패 Ruff1.72초/exit1/cap3/next1이다. P24/P26/P28의 미지불 기록을 지불 이력으로 만들지 않는다.

## 구현할 세 가지

1. **현재 정책의 compiled descriptor를 통일한다.** account에 새 P29 정책의 정확한 canonical path/typed bytes/SHA tuple을 한 번 정의하고, Current-Policy·Recovery-Event·Scope·Recovery-Literal.spec·fresh pass·final freeze·현재 bootstrap이 이를 사용한다. Describe는 관측값이며 신뢰 기준을 대신하지 않는다. 과거 P2710788/4b19ed tuple과 실제 활성화 branch의 P27closed108 증명은 유지한다.

   같은 account의 첫 ledger/history read 앞에 작고 실제인 entry guard를 둔다. 기존 Checked-Source/Checked-Read와 경로·UTF8·크기·해시 검증을 이용한다. 전체 Current-Policy/history/human pipeline을 앞당기지 않는다. 이후 전체 정책·Source·human·freeze·admission 검사는 그대로 남으며 매 실행 다시 검사한다. 별도 preflight·새 reader·cache·권한 생략은 만들지 않는다.

2. **각 실제 entrypoint의 현재 tuple을 일관되게 사용한다.** bookkeeping의 trace와 request guard는 동일한 로컬 compiled tuple을 전체 typed path/bytes/SHA로 비교한다. mixed old-size/new-SHA guard를 수정한다. account를 실행하거나 import해서 상수를 공유하지 않는다. 각 preimport entrypoint에 필요한 로컬 literal은 새 정책 출판 후 같은 descriptor인지 독립적으로 확인한다.

   리뷰의 finite_site_matrix 전체를 실제 코드 위치로 확인한다: account current guard/event/scope/freeze/fresh pass/actual activation, 현재 Call-Policy의 selected5→projected3, Python tool·scanner·helper preimport, Native4·capture·dispatcher·collector, receipt/Q·own9, ROOT caller. `10788` 전역 치환이나 namespace 변경만으로 고쳤다고 판단하지 않는다. capture226/227과 dispatcher77의 역사 tuple은 보존한다. 105/106=P25,107=P27,108+=P29이며 historical P27은 이미 증명된 ledger와 frozen policy를 사용한다.

3. **같은 PROGRAM16에서 실제 entry guard를 검증한다.** 기존 tests1363은 Sum-Actual부터 첫 ledger read까지 definitions로 복사하므로 entry call을 그대로 삽입하면 root/context 전에 실행된다. definitions와 실제 entry guard 사이의 명확한 추출 경계를 만들고, 기존 PROGRAM16에서 case root/base가 설정된 뒤 같은 production guard 또는 정확한 실제 entry slice를 호출한다.

   기존 named test 안에서 canonical path/typed tuple/size/SHA 변조를 실제 guard로 거부하는 사례를 추가한다. JSON 값만 비교하는 mirror validator를 만들지 않는다. 작은 거부 사례는 history replay 전에 끝낸다. whole108/full cost/paid105–107/정책/실제 roster END positive와 Boolean true activation/일반 false negative를 보존한다. 37 extraction slots, Native15+single PROGRAM16, one process8초,84 ordered names/252 phases는 그대로다. 실제 시간은 실행으로만 판단한다.

## 유한한 입력·보존·자원

ROOT의 실제 baseline `parent-e360-p29-ROOT-actual-preimplementation-baseline.json`(76983B, `6bbc077d1ab91acb016426919d67e93dd325f78704fdeb95c2063a82122bf23a`)은 protected322, 과거 Native84, 현재 own9, 선택35 입력, root GOAL과 원장을 새로 확인했다. 실행한 것은 Source metadata 관측뿐이다. ROOT는 반영 후 같은 실제 GPT-6.1 Sol/xhigh에 FIRST 구현을 위임한다.

실제 실패 P28의 보존된 Source freeze9569/fd322…와 residual account712558/42f53c…가 실행된 소스다. pre-residual freeze8703/164c…와 accountd261…는 별도 이전 기록이다. FIRST/R1/ROOT reviews·genuine return·first ErrorRecord·실패·frozen Source/Native/policy를 수정하지 않는다. 새 산출물만 작성하고 canonical·live ledger·root GOAL은 ROOT 적용 때까지 그대로 둔다.

현재 코드가 P28 실패 provenance를 읽어야 한다면 선택 입력6/7/8/9/10/12의 정확한 물리적 여섯 파일만 현재 D512 membership에 먼저 선언한다: as-executed freeze, residual account, failure, genuine return, original ErrorRecord, frozen policy. 이 허용 범위는 새 reader나 nested descriptor의 재귀적 읽기를 허용하지 않는다. 필요 없는 production 읽기를 추가하지 않는다. H1160(212387B/SHA33f036a0…, 총64980982B), D512, combined1672, maxleaf2MiB/table262144B, fixed18 readers의 독립2MiB aggregate는 그대로 유지한다. table-miss dynamic fallback은 금지한다.

현재P29/future30 denied, activation pair2/grant0, 등록108/59/26, 첫 debit index108/count109이다. shared674.2470156/local563.29, 한도1700/1589, 잔여1025.7529844/1025.71, primary768+fallback120=888, margin137.71을 유지한다. 원 설계와 실제 리뷰의 caps/suffix/MAX825/705/future125/65/fixture60/55/51/42/suite520/480/450/360 및 모든705 수치를 그대로 유지한다. reset/new grant0, full11/Source14/late fourth/nonown Q·두 journals·poison·fences·ENTRY·deadlines도 보존한다.

FIRST는 전체 변경 diff와 실제 prospective ROOT operation ABI, 각 현재 정책 site의 source 위치/typed tuple, Native ParseFile 결과, NOT_RUN 상태를 함께 인계하고 안정된 파일 출판 후 멈춘다. ROOT FIRST에서 수정이 필요하면 같은 Sol의 P29 sole ONE_R1을 사용한다. ROOT SECOND 뒤 남은 수정은 ROOT가 마무리하고 실제 apply/freeze/genuine activation으로 이어간다. 이미 사용한 P28 R1은 다시 열지 않는다.

## 실행과 최종 결과

Source 단계에서는 Native UTF8/text/JSON/SHA/regex/ParseFile metadata만 사용한다. Python/import/AST/compile/Ruff/tests/PROGRAM/account/helper/ScriptBlock/private probe/Original arrays·headers·checkpoints/GPU/body/UI/network는 아직 실행하지 않는다. Source 검토를 runtime/timing PASS로 바꾸지 않는다.

리뷰·적용 뒤 필요한 순서의 실제 검사를 수행한다. genuine 첫 FINAL을 그대로 보존하고, 원래 opaque Native capture summary를 출판한 다음 정확한 descriptor를 admin에 전달한다. 순수 Capture-Proof는 내부 Checked-Read된 summary를 받는다. 실패 비용과 ownership을 한 번만 기록하고, 같은 실패를 변경 없이 반복하지 않는다.

현재 Original allowance20초는 과거 startup46–48초보다 작다. 필요한 검사 통과 후 실제 allowance를 맞춘 뒤 동일 후보0→99→300/1.5초 신체 screen으로 이어간다. startup4·실제 contact/slip/displacement·6다리 qualified cycle·bout/native health를 원 기준대로 판단하고 이후 전체 F1–F8로 이어간다. 외부 CPG/교사/위상/궤적/root 교정을 최종 보행에 쓰지 않으며 M2/UI·BANC 해부학 provenance·기존 실패·미사용 초기상태 노출을 유지한다. 현재 F1–F8은 INCOMPLETE다. 사람의 자율 진행 지시에 따라 별도 승인 질문 없이 진행한다.
